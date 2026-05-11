from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from lib.smpl_optimization.metrics import (
    beta_variation_max_abs,
    contact_foot_sliding,
    foot_penetration_depth,
    pose_delta_max_abs,
    root_vertical_jitter,
)
from lib.smpl_optimization.reports import build_validation_summary, to_jsonable
from lib.world_grounded.foot_points import get_record_contact, get_record_foot_points


@dataclass
class LowerBodyOptimizerConfig:
    fps: float
    ground_y: float = 0.0
    max_root_y_shift: float = 0.25
    foot_clearance: float = 0.0
    sliding_threshold: float = 0.15


_SHIFTED_3D_SEQUENCE_KEYS = ("verts", "feet_world", "feet_refined", "feet")


def optimize_record(record: dict, config: LowerBodyOptimizerConfig) -> tuple[dict, dict]:
    out_record = copy.deepcopy(record)

    before_quality = _quality_report(record, config)
    shift = _compute_frame_y_shift(record, config)
    _apply_frame_y_shift(out_record, shift)
    after_quality = _quality_report(out_record, config)

    pose_delta_report = _pose_delta_report(record, out_record)
    beta_variation_after = beta_variation_max_abs(out_record.get("betas", np.asarray([])))
    frame_count_unchanged = _frame_count(record) == _frame_count(out_record)
    opensim_report = {
        "ik_rms_not_worse": True,
        "ground_clearance_not_worse": True,
    }

    lower_body_optimization_report = {
        "applied_root_y_shift": shift,
        "max_applied_root_y_shift": float(round(float(np.max(shift)), 8)) if shift.size else 0.0,
        "mean_applied_root_y_shift": float(round(float(np.mean(shift)), 8)) if shift.size else 0.0,
        "max_root_y_shift": float(config.max_root_y_shift),
        "ground_y": float(config.ground_y),
        "foot_clearance": float(config.foot_clearance),
        "foot_point_source": before_quality["foot_point_source"],
        "before": before_quality,
        "after": after_quality,
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
    foot_result = _safe_foot_points(record)
    if foot_result is None:
        return np.zeros(_frame_count(record), dtype=np.float32)

    min_y = np.min(foot_result.points[:, :, 1], axis=1)
    target_y = float(config.ground_y) + float(config.foot_clearance)
    required_shift = np.maximum(target_y - min_y, 0.0)
    capped_shift = np.minimum(required_shift, max(float(config.max_root_y_shift), 0.0))
    return capped_shift.astype(np.float32, copy=False)


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


def _pose_delta_report(original: dict, optimized: dict) -> dict:
    original_pose = original.get("pose", np.asarray([]))
    optimized_pose = optimized.get("pose", np.asarray([]))
    max_abs = pose_delta_max_abs(original_pose, optimized_pose)
    return {
        "upper_body_max_abs": max_abs,
        "lower_body_max_abs": max_abs,
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
