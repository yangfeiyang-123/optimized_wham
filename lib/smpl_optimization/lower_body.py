from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from lib.smpl_optimization.losses import (
    contact_ground_loss,
    foot_lock_loss,
    penetration_loss,
    smoothness_loss,
    stance_anchor_xz_loss,
    stance_anchor_y_loss,
)
from lib.smpl_optimization.metrics import (
    beta_variation_max_abs,
    contact_foot_sliding,
    foot_penetration_depth,
    pose_delta_max_abs,
    root_vertical_jitter,
)
from lib.smpl_optimization.reports import build_validation_summary, to_jsonable
from lib.smpl_optimization.smpl_forward import expand_betas, smpl_forward_axis_angle
from lib.world_grounded.foot_points import get_record_contact, get_record_foot_points
from lib.world_grounded.contact_segments import extract_stance_segments, kinematic_stance_gate
from lib.world_grounded.stance_anchor import compute_stance_anchors, rasterize_anchors


LOWER_BODY_SMPL_JOINTS = (1, 2, 4, 5, 7, 8, 10, 11)


@dataclass
class LowerBodyOptimizerConfig:
    fps: float
    ground_y: float = 0.0
    max_root_y_shift: float = 0.25
    foot_clearance: float = 0.0
    sliding_threshold: float = 0.15
    enable_pose_pass: bool = False
    pose_iterations: int = 80
    pose_lr: float = 0.03
    device: str = "cuda"
    chunk_size: int = 256
    w_contact_ground: float = 20.0
    w_penetration: float = 50.0
    w_foot_lock: float = 5.0
    w_pose_smooth: float = 2.0
    w_root_smooth: float = 5.0
    w_wham_prior: float = 2.0
    w_root_prior: float = 20.0
    use_stance_anchor_loss: bool = True
    w_stance_anchor_xz: float = 80.0
    w_stance_anchor_y: float = 60.0
    stance_enter_threshold: float = 0.55
    stance_exit_threshold: float = 0.30
    stance_min_frames: int = 4
    stance_merge_gap: int = 2
    use_huber: bool = True
    huber_delta: float = 0.03
    max_lower_body_pose_delta: float = 0.7
    # Kinematic stance gate: a foot may anchor only when it is both near the ground
    # and moving slowly, so genuine steps/pivots are not pinned to a single spot.
    use_kinematic_stance_gate: bool = True
    stance_height_threshold: float = 0.05
    stance_speed_threshold: float = 0.15
    # Percentile of the per-frame lowest-foot height used for the deterministic ground
    # shift. The median (50) aligns the shift to the *typical* lowest foot: on a clip
    # whose feet already sit on the ground it is ~0 (no-op), while a clip whose whole
    # baseline sank underground is still lifted. Crucially it is not dragged by the few
    # deep-penetration or high-flight frames of a jump, which the absolute minimum was —
    # that lifted every planted frame off the floor. Residual per-frame penetration is
    # cleaned up locally by the pose pass's penetration_loss.
    ground_shift_percentile: float = 50.0


_SHIFTED_3D_SEQUENCE_KEYS = ("verts", "feet_world", "feet_refined", "feet")


def optimize_record(record: dict, config: LowerBodyOptimizerConfig) -> tuple[dict, dict]:
    out_record = copy.deepcopy(record)
    original_trans_world = _trans_world_xyz(record)

    before_quality = _quality_report(record, config)
    shift = _compute_frame_y_shift(record, config)
    _apply_frame_y_shift(out_record, shift)
    pose_report = {"pose_optimizer_used": False}
    if config.enable_pose_pass:
        out_record, pose_report = optimize_lower_body_pose_smpl(out_record, config)
        optimized_pose_key = pose_report.get("optimized_pose_key")
        if optimized_pose_key in ("pose", "pose_world"):
            _sync_lower_body_pose_fields(out_record, source_key=str(optimized_pose_key))
    _clamp_total_root_y_shift(out_record, original_trans_world, config)
    after_quality = _quality_report(out_record, config)

    pose_delta_report = _pose_delta_report(record, out_record)
    beta_variation_after = beta_variation_max_abs(out_record.get("betas", np.asarray([])))
    frame_count_unchanged = _frame_count(record) == _frame_count(out_record)
    opensim_report = {
        "status": "not_run",
        "used_for_success": False,
        "ik_rms_not_worse": True,
        "ground_clearance_not_worse": True,
    }
    deterministic_root_y_shift_max_abs = float(np.max(np.abs(shift))) if shift.size else 0.0
    pose_root_residual_y_max_abs = float(pose_report.get("root_residual_y_max_abs", 0.0) or 0.0)
    root_y_shift_budget = max(float(config.max_root_y_shift), 0.0)
    total_root_y_shift_max_abs = _total_root_y_shift_max_abs(record, out_record)

    lower_body_optimization_report = {
        "applied_root_y_shift": shift,
        "max_applied_root_y_shift": float(round(float(np.max(shift)), 8)) if shift.size else 0.0,
        "mean_applied_root_y_shift": float(round(float(np.mean(shift)), 8)) if shift.size else 0.0,
        "max_root_y_shift": float(config.max_root_y_shift),
        "root_y_shift_budget": float(root_y_shift_budget),
        "deterministic_root_y_shift_max_abs": float(round(deterministic_root_y_shift_max_abs, 8)),
        "pose_root_residual_y_max_abs": float(round(pose_root_residual_y_max_abs, 8)),
        "total_root_y_shift_max_abs": float(round(total_root_y_shift_max_abs, 8)),
        "ground_y": float(config.ground_y),
        "foot_clearance": float(config.foot_clearance),
        "foot_point_source": before_quality["foot_point_source"],
        "root_shift_applied": bool(np.any(shift > 0.0)),
        "root_shift_reason": _root_shift_reason(record, shift),
        "before": before_quality,
        "after": after_quality,
        **pose_report,
    }

    ground_contact_report = {
        "before": {
            "source": before_quality["foot_point_source"],
            "contact_available": before_quality["contact_available"],
            "penetration": before_quality["penetration"],
            "sliding": before_quality["sliding"],
        },
        "after": {
            "source": after_quality["foot_point_source"],
            "contact_available": after_quality["contact_available"],
            "penetration": after_quality["penetration"],
            "sliding": after_quality["sliding"],
        },
    }

    validation_summary = build_validation_summary(
        beta_variation_after_max_abs=beta_variation_after,
        frame_count_unchanged=frame_count_unchanged,
        before=before_quality,
        after=after_quality,
        pose_delta=pose_delta_report,
        opensim=opensim_report,
    )

    reports = {
        "lower_body_optimization_report": lower_body_optimization_report,
        "ground_contact_report": ground_contact_report,
        "pose_delta_report": pose_delta_report,
        "validation_summary": validation_summary,
    }
    return out_record, to_jsonable(reports)


def lower_body_pose_mask(num_pose_dims: int = 72) -> np.ndarray:
    mask = np.zeros((num_pose_dims,), dtype=bool)
    for joint in LOWER_BODY_SMPL_JOINTS:
        start = joint * 3
        if start + 3 <= num_pose_dims:
            mask[start : start + 3] = True
    return mask


def _contact_for_feet(
    record: dict,
    n_frames: int,
    n_feet: int,
    config: Optional["LowerBodyOptimizerConfig"] = None,
    foot_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    contact = get_record_contact(record, n_frames=n_frames, n_points=n_feet)
    if contact is None:
        return np.zeros((n_frames, n_feet), dtype=np.float32)
    contact = _fit_contact_shape(contact, n_frames=n_frames, n_points=n_feet)
    # Gate the contact that drives contact_ground_loss / foot_lock in the pose pass by the
    # low-and-slow kinematic mask. WHAM contact confidence stays non-zero even on airborne
    # frames (a jump/smash keeps ~0.2-0.5), so an ungated contact_ground_loss pulls the
    # lifted foot back to the floor and flattens the jump. Gating lets genuine flight
    # frames drop out while planted frames keep their ground constraint.
    if (
        config is not None
        and getattr(config, "use_kinematic_stance_gate", False)
        and foot_points is not None
    ):
        gate = kinematic_stance_gate(
            np.asarray(foot_points)[:n_frames],
            fps=float(config.fps),
            ground_y=float(config.ground_y),
            height_threshold=float(config.stance_height_threshold),
            speed_threshold=float(config.stance_speed_threshold),
            vertical_axis=1,
        )
        contact = contact * _fit_contact_shape(gate.astype(np.float32), n_frames=n_frames, n_points=n_feet)
    return contact


def optimize_lower_body_pose_smpl(
    record: dict,
    config: LowerBodyOptimizerConfig,
    frame_weights: Optional[np.ndarray] = None,
) -> tuple[dict, dict]:
    from lib.models import build_body_model

    pose_key = "pose_world" if "pose_world" in record else "pose"
    if pose_key not in record or "betas" not in record or "trans_world" not in record:
        return copy.deepcopy(record), {
            "pose_optimizer_used": False,
            "reason": "missing_pose_or_betas_or_trans_world",
        }

    out = copy.deepcopy(record)
    pose_np = np.asarray(record[pose_key], dtype=np.float32)
    if pose_np.ndim != 2 or len(pose_np) == 0:
        return out, {"pose_optimizer_used": False, "reason": "empty_sequence"}

    n_frames = int(pose_np.shape[0])
    trans_np = np.asarray(record["trans_world"], dtype=np.float32)
    if trans_np.ndim != 2 or trans_np.shape[1] < 3 or trans_np.shape[0] == 0:
        return out, {
            "pose_optimizer_used": False,
            "reason": "missing_pose_betas_or_trans_world",
        }

    trans_np = _fit_2d_sequence(trans_np[:, :3], n_frames=n_frames, n_cols=3)
    betas_np = expand_betas(record["betas"], n_frames)
    betas_np = _fit_2d_sequence(betas_np, n_frames=n_frames, n_cols=10)

    requested_device = str(config.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    pose_mask_np = lower_body_pose_mask(pose_np.shape[1])
    pose_mask = torch.tensor(pose_mask_np, dtype=torch.bool, device=device)
    frame_weights_np = np.ones((n_frames,), dtype=np.float32)
    if frame_weights is not None:
        frame_weights_np = _fit_frame_weights(frame_weights, n_frames=n_frames)

    final_pose_np = pose_np.copy()
    final_trans_np = trans_np.copy()
    feet_chunks = []
    verts_chunks = []
    final_root_residual_y_max_abs = 0.0
    loss_value = 0.0
    anchor_targets_np, anchor_mask_np, anchor_report = _build_anchor_targets_for_pose_pass(
        record,
        config,
        n_frames=n_frames,
    )

    # Foot points for the kinematic contact gate (see _contact_for_feet). Computed once
    # here so genuine flight frames drop out of the ground/lock constraints across chunks.
    gate_foot_points = _safe_foot_points(record)
    gate_points_np = gate_foot_points.points if gate_foot_points is not None else None

    chunk_size = max(1, int(config.chunk_size))
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        window_len = end - start

        pose_base = torch.tensor(pose_np[start:end], dtype=torch.float32, device=device)
        trans_base = torch.tensor(trans_np[start:end], dtype=torch.float32, device=device)
        betas = torch.tensor(betas_np[start:end], dtype=torch.float32, device=device)
        pose_residual = torch.zeros_like(pose_base, requires_grad=True)
        root_residual = torch.zeros_like(trans_base, requires_grad=True)
        optimizer = torch.optim.Adam([pose_residual, root_residual], lr=float(config.pose_lr))
        model = build_body_model(str(device), batch_size=window_len)

        with torch.no_grad():
            initial = smpl_forward_axis_angle(model, pose_base, betas, trans_base)
            n_feet = int(initial["feet"].shape[1])

        contact_np = _contact_for_feet(
            record,
            n_frames=n_frames,
            n_feet=n_feet,
            config=config,
            foot_points=gate_points_np,
        )[start:end]
        contact = torch.tensor(contact_np, dtype=torch.float32, device=device)
        frame_w = torch.tensor(frame_weights_np[start:end, None], dtype=torch.float32, device=device)
        anchor_targets = None
        anchor_mask = None
        if config.use_stance_anchor_loss and anchor_targets_np is not None and anchor_mask_np is not None:
            fitted_targets = _fit_anchor_targets(anchor_targets_np[start:end], window_len, n_feet)
            fitted_mask = _fit_contact_shape(anchor_mask_np[start:end], n_frames=window_len, n_points=n_feet)
            anchor_targets = torch.tensor(fitted_targets, dtype=torch.float32, device=device)
            anchor_mask = torch.tensor(fitted_mask, dtype=torch.float32, device=device)

        for _ in range(int(config.pose_iterations)):
            optimizer.zero_grad()
            masked_pose = pose_base + pose_residual * pose_mask[None, :]
            root_y = torch.clamp(
                root_residual[:, 1],
                -float(config.max_root_y_shift),
                float(config.max_root_y_shift),
            )
            masked_root = trans_base + torch.stack(
                [torch.zeros_like(root_y), root_y, torch.zeros_like(root_y)],
                dim=-1,
            )
            smpl = smpl_forward_axis_angle(model, masked_pose, betas, masked_root)
            foot = smpl["feet"]
            foot_y = foot[:, :, 1]
            weighted_contact = contact * frame_w
            loss = (
                config.w_contact_ground * contact_ground_loss(foot_y, weighted_contact, config.ground_y)
                + config.w_penetration * penetration_loss(foot_y, config.ground_y)
                + config.w_foot_lock * foot_lock_loss(foot, weighted_contact)
                + config.w_pose_smooth * smoothness_loss(masked_pose[:, pose_mask])
                + config.w_root_smooth * smoothness_loss(masked_root)
                + config.w_wham_prior * torch.mean((pose_residual[:, pose_mask]) ** 2)
                + config.w_root_prior * torch.mean(root_residual * root_residual)
            )
            if anchor_targets is not None and anchor_mask is not None:
                loss = (
                    loss
                    + config.w_stance_anchor_xz
                    * stance_anchor_xz_loss(
                        foot,
                        anchor_targets[..., [0, 2]],
                        anchor_mask,
                        huber_delta=config.huber_delta,
                    )
                    + config.w_stance_anchor_y
                    * stance_anchor_y_loss(
                        foot,
                        ground_y=config.ground_y,
                        mask=anchor_mask,
                        huber_delta=config.huber_delta,
                    )
                )
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())

        with torch.no_grad():
            final_pose = (pose_base + pose_residual * pose_mask[None, :]).detach().cpu().numpy().astype(np.float32)
            final_root_residual = root_residual.detach().cpu().numpy().astype(np.float32)
            final_root_residual[:, 0] = 0.0
            final_root_residual[:, 2] = 0.0
            final_root_residual[:, 1] = np.clip(
                final_root_residual[:, 1],
                -float(config.max_root_y_shift),
                float(config.max_root_y_shift),
            )
            final_trans = (trans_np[start:end] + final_root_residual).astype(np.float32)
            final = smpl_forward_axis_angle(
                model,
                torch.tensor(final_pose, dtype=torch.float32, device=device),
                betas,
                torch.tensor(final_trans, dtype=torch.float32, device=device),
            )

        final_pose_np[start:end] = final_pose
        final_trans_np[start:end] = final_trans
        feet_chunks.append(final["feet"].detach().cpu().numpy().astype(np.float32))
        verts_chunks.append(final["vertices"].detach().cpu().numpy().astype(np.float32))
        if len(final_root_residual):
            final_root_residual_y_max_abs = max(
                final_root_residual_y_max_abs,
                float(np.max(np.abs(final_root_residual[:, 1]))),
            )

    final_feet = np.concatenate(feet_chunks, axis=0).astype(np.float32)
    final_verts = np.concatenate(verts_chunks, axis=0).astype(np.float32)

    out[pose_key] = final_pose_np
    _sync_lower_body_pose_fields(out, source_key=pose_key)
    out["trans_world"] = final_trans_np
    out["feet_refined"] = final_feet
    if "feet_world" in record:
        out["feet_world"] = final_feet.copy()
    if "feet" in record:
        out["feet"] = final_feet.copy()
    out["verts"] = final_verts

    return out, {
        "pose_optimizer_used": True,
        "iterations": int(config.pose_iterations),
        "final_loss": loss_value,
        "optimized_pose_key": pose_key,
        "optimized_pose_dims": np.where(pose_mask_np)[0].tolist(),
        "root_residual_y_max_abs": final_root_residual_y_max_abs,
        **anchor_report,
    }


def _quality_report(record: dict, config: LowerBodyOptimizerConfig) -> dict:
    foot_result = _safe_foot_points(record)
    trans_world = record.get("trans_world", np.asarray([]))

    if foot_result is None:
        penetration = foot_penetration_depth(np.asarray([]), ground_y=config.ground_y)
        sliding = contact_foot_sliding(
            np.asarray([]),
            np.asarray([]),
            fps=config.fps,
            threshold=config.sliding_threshold,
        )
        return {
            "foot_point_source": None,
            "foot_point_labels": [],
            "contact_available": False,
            "penetration": penetration,
            "sliding": sliding,
            "root": root_vertical_jitter(trans_world),
            "beta_variation_max_abs": beta_variation_max_abs(record.get("betas", np.asarray([]))),
        }

    points = foot_result.points
    contact = get_record_contact(record, n_frames=points.shape[0], n_points=points.shape[1])
    if contact is None:
        contact = np.zeros(points.shape[:2], dtype=np.float32)
        contact_available = False
    else:
        contact = _fit_contact_shape(contact, n_frames=points.shape[0], n_points=points.shape[1])
        contact_available = True

    # Sliding is only meaningful on genuinely-planted frames. WHAM contact confidence
    # saturates for near-ground footage, so measuring speed on every high-confidence
    # frame would flag real steps as "sliding" and reward freezing the feet. Gate the
    # contact used for the sliding metric by the same low-and-slow kinematics used for
    # anchoring; penetration/coverage keep the raw confidence.
    sliding_contact = _gated_contact_for_sliding(points, contact, config)

    return {
        "foot_point_source": foot_result.source,
        "foot_point_labels": foot_result.labels,
        "contact_available": contact_available,
        "num_stance_segments": _stance_segment_count(contact, foot_result.labels, config),
        "stance_coverage_ratio": float(np.mean(contact > float(config.stance_exit_threshold))) if contact.size else 0.0,
        "penetration": foot_penetration_depth(points[:, :, 1], ground_y=config.ground_y),
        "sliding": contact_foot_sliding(
            points,
            sliding_contact,
            fps=config.fps,
            threshold=config.sliding_threshold,
        ),
        "root": root_vertical_jitter(trans_world),
        "beta_variation_max_abs": beta_variation_max_abs(record.get("betas", np.asarray([]))),
    }


def _gated_contact_for_sliding(
    points: np.ndarray, contact: np.ndarray, config: LowerBodyOptimizerConfig
) -> np.ndarray:
    """Intersect contact confidence with the low-and-slow kinematic stance gate."""
    if not config.use_kinematic_stance_gate:
        return contact
    gate = kinematic_stance_gate(
        points,
        fps=float(config.fps),
        ground_y=float(config.ground_y),
        height_threshold=float(config.stance_height_threshold),
        speed_threshold=float(config.stance_speed_threshold),
        vertical_axis=1,
    )
    return np.asarray(contact, dtype=np.float32) * gate.astype(np.float32)


def _compute_frame_y_shift(record: dict, config: LowerBodyOptimizerConfig) -> np.ndarray:
    if "trans_world" not in record:
        return np.zeros(_frame_count(record), dtype=np.float32)

    foot_result = _safe_foot_points(record)
    if foot_result is None:
        return np.zeros(_frame_count(record), dtype=np.float32)

    # Ground alignment: shift the whole clip up so the feet's lowest point reaches the
    # ground. HOW we summarise "lowest point" depends on whether the pose pass runs:
    #   * Pose pass ON: use a robust percentile (median) of the per-frame lowest foot.
    #     WHAM's world frame jitters and jump/smash clips mix a few deep-penetration frames
    #     with genuine flight; the absolute minimum lets one spurious frame lift the whole
    #     sequence and float every planted frame. Residual per-frame penetration is cleaned
    #     up locally by the pose pass's penetration_loss.
    #   * Pose pass OFF: this shift is the ONLY grounding mechanism, so fall back to the
    #     absolute minimum to guarantee no frame is left penetrating.
    lowest_per_frame = np.min(foot_result.points[:, :, 1], axis=1)
    if config.enable_pose_pass:
        q = float(np.clip(config.ground_shift_percentile, 0.0, 100.0))
        min_y = float(np.percentile(lowest_per_frame, q))
    else:
        min_y = float(np.min(lowest_per_frame))
    target_y = float(config.ground_y) + float(config.foot_clearance)
    required_shift = max(target_y - min_y, 0.0)
    capped_shift = min(required_shift, max(float(config.max_root_y_shift), 0.0))
    return np.full(_frame_count(record), capped_shift, dtype=np.float32)


def _apply_frame_y_shift(record: dict, shift: np.ndarray) -> None:
    if shift.size == 0:
        return

    if "trans_world" in record:
        trans_world = np.asarray(record["trans_world"]).copy()
        if trans_world.ndim == 2 and trans_world.shape[1] >= 2:
            n_frames = min(trans_world.shape[0], shift.shape[0])
            trans_world[:n_frames, 1] += shift[:n_frames]
            record["trans_world"] = trans_world

    for key in _SHIFTED_3D_SEQUENCE_KEYS:
        if key not in record:
            continue
        value = np.asarray(record[key]).copy()
        if value.ndim == 3 and value.shape[-1] >= 2:
            n_frames = min(value.shape[0], shift.shape[0])
            value[:n_frames, :, 1] += shift[:n_frames, None]
            record[key] = value


def _trans_world_xyz(record: dict) -> Optional[np.ndarray]:
    if "trans_world" not in record:
        return None
    trans_world = np.asarray(record["trans_world"], dtype=np.float32)
    if trans_world.ndim != 2 or trans_world.shape[0] == 0 or trans_world.shape[1] < 2:
        return None
    return trans_world[:, :3].copy()


def _clamp_total_root_y_shift(
    record: dict,
    original_trans_world: Optional[np.ndarray],
    config: LowerBodyOptimizerConfig,
) -> None:
    if original_trans_world is None or "trans_world" not in record:
        return

    trans_world = np.asarray(record["trans_world"]).copy()
    if trans_world.ndim != 2 or trans_world.shape[0] == 0 or trans_world.shape[1] < 2:
        return

    n_frames = min(trans_world.shape[0], original_trans_world.shape[0])
    if n_frames == 0:
        return

    budget = max(float(config.max_root_y_shift), 0.0)
    original_y = original_trans_world[:n_frames, 1]
    clamped_y = np.clip(trans_world[:n_frames, 1], original_y - budget, original_y + budget)
    correction = clamped_y - trans_world[:n_frames, 1]
    if not np.any(correction):
        return

    trans_world[:n_frames, 1] = clamped_y
    record["trans_world"] = trans_world
    for key in _SHIFTED_3D_SEQUENCE_KEYS:
        if key not in record:
            continue
        value = np.asarray(record[key]).copy()
        if value.ndim == 3 and value.shape[-1] >= 2:
            rows = min(value.shape[0], n_frames)
            value[:rows, :, 1] += correction[:rows, None]
            record[key] = value


def _total_root_y_shift_max_abs(original: dict, optimized: dict) -> float:
    original_trans_world = _trans_world_xyz(original)
    optimized_trans_world = _trans_world_xyz(optimized)
    if original_trans_world is None or optimized_trans_world is None:
        return 0.0

    n_frames = min(original_trans_world.shape[0], optimized_trans_world.shape[0])
    if n_frames == 0:
        return 0.0
    return float(np.max(np.abs(optimized_trans_world[:n_frames, 1] - original_trans_world[:n_frames, 1])))


def _fit_contact_shape(contact: np.ndarray, *, n_frames: int, n_points: int) -> np.ndarray:
    fitted = np.zeros((n_frames, n_points), dtype=np.float32)
    contact = np.asarray(contact, dtype=np.float32)
    if contact.ndim != 2 or contact.shape[0] == 0 or contact.shape[1] == 0:
        return fitted

    rows = min(n_frames, contact.shape[0])
    cols = min(n_points, contact.shape[1])
    fitted[:rows, :cols] = contact[:rows, :cols]
    return fitted


def _fit_anchor_targets(value: np.ndarray, n_frames: int, n_points: int) -> np.ndarray:
    fitted = np.zeros((n_frames, n_points, 3), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return fitted
    rows = min(n_frames, arr.shape[0])
    cols = min(n_points, arr.shape[1])
    fitted[:rows, :cols] = arr[:rows, :cols]
    return fitted


def _fit_2d_sequence(value: np.ndarray, *, n_frames: int, n_cols: int) -> np.ndarray:
    fitted = np.zeros((n_frames, n_cols), dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        return fitted

    rows = min(n_frames, value.shape[0])
    cols = min(n_cols, value.shape[1])
    fitted[:rows, :cols] = value[:rows, :cols]
    if rows < n_frames:
        fitted[rows:] = fitted[rows - 1]
    return fitted


def _fit_frame_weights(frame_weights: np.ndarray, *, n_frames: int) -> np.ndarray:
    fitted = np.ones((n_frames,), dtype=np.float32)
    weights = np.asarray(frame_weights, dtype=np.float32).reshape(-1)
    if weights.size == 0:
        return fitted

    rows = min(n_frames, weights.shape[0])
    fitted[:rows] = weights[:rows]
    if rows < n_frames:
        fitted[rows:] = fitted[rows - 1]
    return fitted


def _build_anchor_targets_for_pose_pass(
    record: dict,
    config: LowerBodyOptimizerConfig,
    *,
    n_frames: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    if not config.use_stance_anchor_loss:
        return None, None, {"stance_anchor_loss_used": False}
    foot_result = _safe_foot_points(record)
    if foot_result is None:
        return None, None, {"stance_anchor_loss_used": False, "stance_anchor_reason": "missing_foot_points"}

    points = foot_result.points
    contact = get_record_contact(record, n_frames=n_frames, n_points=points.shape[1])
    if contact is None:
        return None, None, {"stance_anchor_loss_used": False, "stance_anchor_reason": "missing_contact"}
    contact = _fit_contact_shape(contact, n_frames=n_frames, n_points=points.shape[1])
    gate = _stance_kinematic_gate(points[:n_frames], config)
    segments, stance_mask = extract_stance_segments(
        contact,
        foot_result.labels,
        enter_threshold=config.stance_enter_threshold,
        exit_threshold=config.stance_exit_threshold,
        min_segment_len=config.stance_min_frames,
        merge_gap=config.stance_merge_gap,
        kinematic_gate=gate,
    )
    anchors = compute_stance_anchors(points[:n_frames], contact, segments, ground_y=config.ground_y)
    targets, anchor_mask = rasterize_anchors(anchors, n_frames=n_frames, n_feet=points.shape[1])
    return (
        targets,
        anchor_mask.astype(np.float32),
        {
            "stance_anchor_loss_used": bool(anchors),
            "kinematic_stance_gate_used": bool(gate is not None),
            "num_stance_segments": int(len(segments)),
            "num_stance_anchors": int(len(anchors)),
            "stance_coverage_ratio": float(np.mean(stance_mask)) if stance_mask.size else 0.0,
        },
    )


def _stance_kinematic_gate(points: np.ndarray, config: LowerBodyOptimizerConfig) -> Optional[np.ndarray]:
    """Foot points are Y-up in this stage; gate stance on low-and-slow feet."""
    if not config.use_kinematic_stance_gate:
        return None
    return kinematic_stance_gate(
        points,
        fps=float(config.fps),
        ground_y=float(config.ground_y),
        height_threshold=float(config.stance_height_threshold),
        speed_threshold=float(config.stance_speed_threshold),
        vertical_axis=1,
    )


def _stance_segment_count(contact: np.ndarray, labels: list[str], config: LowerBodyOptimizerConfig) -> int:
    if contact.size == 0:
        return 0
    segments, _ = extract_stance_segments(
        contact,
        labels,
        enter_threshold=config.stance_enter_threshold,
        exit_threshold=config.stance_exit_threshold,
        min_segment_len=config.stance_min_frames,
        merge_gap=config.stance_merge_gap,
    )
    return int(len(segments))


def _sync_lower_body_pose_fields(record: dict, *, source_key: str) -> None:
    if "pose" not in record or "pose_world" not in record:
        return

    if source_key not in ("pose", "pose_world"):
        return

    target_key = "pose" if source_key == "pose_world" else "pose_world"
    source = np.asarray(record[source_key])
    target = np.asarray(record[target_key]).copy()
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        return

    pose_mask = lower_body_pose_mask(source.shape[1])
    target[:, pose_mask] = source[:, pose_mask]
    record[target_key] = target


def _root_shift_reason(record: dict, shift: np.ndarray) -> str:
    if "trans_world" not in record:
        return "missing_trans_world"
    if shift.size == 0 or not np.any(shift > 0.0):
        return "no_shift_required"
    return "applied"


def _pose_delta_report(original: dict, optimized: dict) -> dict:
    original_pose = original.get("pose", np.asarray([]))
    optimized_pose = optimized.get("pose", np.asarray([]))
    max_abs = pose_delta_max_abs(original_pose, optimized_pose)
    original_pose = np.asarray(original_pose)
    optimized_pose = np.asarray(optimized_pose)
    upper_max_abs = max_abs
    lower_max_abs = max_abs
    if (
        original_pose.ndim == 2
        and optimized_pose.ndim == 2
        and original_pose.shape == optimized_pose.shape
        and original_pose.shape[1] > 0
    ):
        lower_mask = lower_body_pose_mask(original_pose.shape[1])
        upper_mask = ~lower_mask
        lower_max_abs = (
            float(np.max(np.abs(optimized_pose[:, lower_mask] - original_pose[:, lower_mask])))
            if np.any(lower_mask)
            else 0.0
        )
        upper_max_abs = (
            float(np.max(np.abs(optimized_pose[:, upper_mask] - original_pose[:, upper_mask])))
            if np.any(upper_mask)
            else 0.0
        )
    return {
        "upper_body_max_abs": upper_max_abs,
        "lower_body_max_abs": lower_max_abs,
        "total_max_abs": max_abs,
    }


def _safe_foot_points(record: dict) -> Optional[Any]:
    try:
        return get_record_foot_points(record)
    except ValueError:
        return None


def _frame_count(record: dict) -> int:
    for key in ("trans_world", "pose", "betas", "feet_refined", "feet_world", "feet", "verts"):
        if key not in record:
            continue
        value = np.asarray(record[key])
        if value.ndim > 0:
            return int(value.shape[0])
    return 0
