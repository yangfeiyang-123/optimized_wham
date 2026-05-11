from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lib.smpl_optimization.losses import (
    contact_ground_loss,
    foot_lock_loss,
    penetration_loss,
    smoothness_loss,
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


def _contact_for_feet(record: dict, n_frames: int, n_feet: int) -> np.ndarray:
    contact = get_record_contact(record, n_frames=n_frames, n_points=n_feet)
    if contact is None:
        return np.zeros((n_frames, n_feet), dtype=np.float32)
    return _fit_contact_shape(contact, n_frames=n_frames, n_points=n_feet)


def optimize_lower_body_pose_smpl(
    record: dict,
    config: LowerBodyOptimizerConfig,
    frame_weights: np.ndarray | None = None,
) -> tuple[dict, dict]:
    from lib.models import build_body_model

    if "pose" not in record or "betas" not in record or "trans_world" not in record:
        return copy.deepcopy(record), {
            "pose_optimizer_used": False,
            "reason": "missing_pose_betas_or_trans_world",
        }

    out = copy.deepcopy(record)
    pose_np = np.asarray(record["pose"], dtype=np.float32)
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

        contact_np = _contact_for_feet(record, n_frames=n_frames, n_feet=n_feet)[start:end]
        contact = torch.tensor(contact_np, dtype=torch.float32, device=device)
        frame_w = torch.tensor(frame_weights_np[start:end, None], dtype=torch.float32, device=device)

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

    out["pose"] = final_pose_np
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
        "optimized_pose_dims": np.where(pose_mask_np)[0].tolist(),
        "root_residual_y_max_abs": final_root_residual_y_max_abs,
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

    return {
        "foot_point_source": foot_result.source,
        "foot_point_labels": foot_result.labels,
        "contact_available": contact_available,
        "penetration": foot_penetration_depth(points[:, :, 1], ground_y=config.ground_y),
        "sliding": contact_foot_sliding(
            points,
            contact,
            fps=config.fps,
            threshold=config.sliding_threshold,
        ),
        "root": root_vertical_jitter(trans_world),
        "beta_variation_max_abs": beta_variation_max_abs(record.get("betas", np.asarray([]))),
    }


def _compute_frame_y_shift(record: dict, config: LowerBodyOptimizerConfig) -> np.ndarray:
    if "trans_world" not in record:
        return np.zeros(_frame_count(record), dtype=np.float32)

    foot_result = _safe_foot_points(record)
    if foot_result is None:
        return np.zeros(_frame_count(record), dtype=np.float32)

    min_y = float(np.min(foot_result.points[:, :, 1]))
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


def _trans_world_xyz(record: dict) -> np.ndarray | None:
    if "trans_world" not in record:
        return None
    trans_world = np.asarray(record["trans_world"], dtype=np.float32)
    if trans_world.ndim != 2 or trans_world.shape[0] == 0 or trans_world.shape[1] < 2:
        return None
    return trans_world[:, :3].copy()


def _clamp_total_root_y_shift(
    record: dict,
    original_trans_world: np.ndarray | None,
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


def _safe_foot_points(record: dict) -> Any | None:
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
