"""Conservative Stage7 whole-body SMPL pose smoothing candidate generation."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from lib.smpl_optimization.metrics import pose_smoothness, root_translation_smoothness


LOWER_BODY_JOINTS = (1, 2, 4, 5, 7, 8, 10, 11)
ROOT_AND_TRUNK_JOINTS = (0, 3, 6, 9, 12, 13, 14, 15)


@dataclass(frozen=True, init=False)
class WholeBodySmoothConfig:
    lower_body_weight: float
    trunk_weight: float
    upper_body_weight: float
    max_lower_body_delta: float
    max_whole_body_delta: float
    max_root_delta: float
    max_root_vertical_delta: float

    def __init__(
        self,
        lower_body_weight: float = 0.35,
        trunk_weight: float = 0.55,
        upper_body_weight: float = 0.8,
        max_lower_body_delta: float = 0.05,
        max_whole_body_delta: float = 0.12,
        max_root_delta: float = 0.03,
        max_root_vertical_delta: float = 0.015,
        max_vertical_delta: float | None = None,
    ):
        if max_vertical_delta is not None:
            max_root_vertical_delta = max_vertical_delta

        object.__setattr__(self, "lower_body_weight", lower_body_weight)
        object.__setattr__(self, "trunk_weight", trunk_weight)
        object.__setattr__(self, "upper_body_weight", upper_body_weight)
        object.__setattr__(self, "max_lower_body_delta", max_lower_body_delta)
        object.__setattr__(self, "max_whole_body_delta", max_whole_body_delta)
        object.__setattr__(self, "max_root_delta", max_root_delta)
        object.__setattr__(self, "max_root_vertical_delta", max_root_vertical_delta)

    @property
    def max_vertical_delta(self):
        return self.max_root_vertical_delta


def _clean_float(value):
    return float(round(float(value), 8))


def build_region_weights(config=None) -> np.ndarray:
    config = config or WholeBodySmoothConfig()
    weights = np.full((24, 3), float(config.upper_body_weight), dtype=np.float32)
    weights[list(ROOT_AND_TRUNK_JOINTS), :] = float(config.trunk_weight)
    weights[list(LOWER_BODY_JOINTS), :] = float(config.lower_body_weight)
    return weights


def _smooth_three_point(values):
    values = np.asarray(values)
    candidate = values.astype(np.float64, copy=True)
    if values.ndim == 0 or values.shape[0] < 3:
        return candidate
    candidate[1:-1] = (values[:-2] + values[1:-1] + values[2:]) / 3.0
    return candidate


def _clip_delta(candidate, baseline, limit):
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    limit = max(float(limit) - 1e-7, 0.0)
    return np.asarray(baseline, dtype=np.float64) + np.clip(delta, -limit, limit)


def _record_pose_key(record):
    if "pose_world" in record:
        return "pose_world"
    if "pose" in record:
        return "pose"
    return None


def _record_trans_key(record):
    if "trans_world" in record:
        return "trans_world"
    if "trans" in record:
        return "trans"
    return None


def _root_report(record, candidate, trans_key):
    if trans_key is None:
        return {}, {}, {"max_abs": 0.0, "vertical_max_abs": 0.0}, False

    baseline = np.asarray(record[trans_key])
    smoothed = np.asarray(candidate[trans_key])
    if baseline.ndim != 2 or baseline.shape[1] < 3:
        return {}, {}, {"max_abs": 0.0, "vertical_max_abs": 0.0}, False

    delta = smoothed[:, :3].astype(np.float64) - baseline[:, :3].astype(np.float64)
    return (
        root_translation_smoothness(baseline[:, :3]),
        root_translation_smoothness(smoothed[:, :3]),
        {
            "max_abs": _clean_float(np.max(np.abs(delta))) if delta.size else 0.0,
            "vertical_max_abs": _clean_float(np.max(np.abs(delta[:, 1]))) if delta.size else 0.0,
        },
        True,
    )


def _base_report(record, candidate, pose_key, trans_key, candidate_generated, reason=None):
    if pose_key is None:
        pose_before = np.zeros((0, 72), dtype=np.float32)
        pose_after = pose_before
        pose_delta = np.zeros((0, 24, 3), dtype=np.float64)
    else:
        pose_before = np.asarray(record[pose_key])
        pose_after = np.asarray(candidate[pose_key])
        if pose_before.ndim == 2 and pose_before.shape[1] >= 72:
            pose_before = pose_before[:, :72]
            pose_after = pose_after[:, :72]
            pose_delta = pose_after.reshape(pose_after.shape[0], 24, 3).astype(
                np.float64
            ) - pose_before.reshape(pose_before.shape[0], 24, 3).astype(np.float64)
        else:
            pose_delta = np.zeros((0, 24, 3), dtype=np.float64)

    pose_after_report = pose_smoothness(pose_after)
    root_before, root_after, root_delta, root_available = _root_report(
        record, candidate, trans_key
    )
    report = {
        "candidate_generated": bool(candidate_generated),
        "pose_smoothness": pose_after_report,
        "pose_smoothness_before": pose_smoothness(pose_before),
        "pose_smoothness_after": pose_after_report,
        "root_smoothness": root_after if root_available else {},
        "root_smoothness_available": root_available,
        "root_smoothness_before": root_before,
        "root_smoothness_after": root_after,
        "pose_delta": {
            "lower_body_max_abs": _clean_float(
                np.max(np.abs(pose_delta[:, LOWER_BODY_JOINTS, :]))
            )
            if pose_delta.size
            else 0.0,
            "whole_body_max_abs": _clean_float(np.max(np.abs(pose_delta)))
            if pose_delta.size
            else 0.0,
        },
        "root_delta": root_delta,
    }
    if reason is not None:
        report["reason"] = reason
    return report


def smooth_record(record, config=None) -> tuple[dict, dict]:
    config = config or WholeBodySmoothConfig()
    candidate = deepcopy(record)
    pose_key = _record_pose_key(record)
    trans_key = _record_trans_key(record)

    if pose_key is None:
        report = _base_report(
            record, candidate, pose_key, trans_key, False, reason="missing_pose"
        )
        return candidate, report

    pose = np.asarray(record[pose_key])
    if pose.ndim != 2 or pose.shape[1] < 72:
        report = _base_report(
            record, candidate, pose_key, trans_key, False, reason="missing_pose_values"
        )
        return candidate, report
    if pose.shape[0] < 3:
        report = _base_report(
            record, candidate, pose_key, trans_key, False, reason="too_short_pose"
        )
        return candidate, report

    pose_dtype = pose.dtype
    baseline_pose = pose[:, :72].reshape(pose.shape[0], 24, 3)
    smoothed_pose = _smooth_three_point(baseline_pose)
    weights = build_region_weights(config).reshape(1, 24, 3)
    weighted_pose = baseline_pose.astype(np.float64) + (
        smoothed_pose - baseline_pose.astype(np.float64)
    ) * weights
    weighted_pose = _clip_delta(weighted_pose, baseline_pose, config.max_whole_body_delta)
    lower = weighted_pose[:, LOWER_BODY_JOINTS, :]
    lower_base = baseline_pose[:, LOWER_BODY_JOINTS, :]
    weighted_pose[:, LOWER_BODY_JOINTS, :] = _clip_delta(
        lower, lower_base, config.max_lower_body_delta
    )

    candidate_pose = np.asarray(candidate[pose_key]).copy()
    candidate_pose[:, :72] = weighted_pose.reshape(pose.shape[0], 72).astype(
        pose_dtype, copy=False
    )
    candidate[pose_key] = candidate_pose

    if trans_key is not None:
        trans = np.asarray(record[trans_key])
        if trans.ndim == 2 and trans.shape[0] >= 3 and trans.shape[1] >= 3:
            trans_dtype = trans.dtype
            baseline_trans = trans[:, :3]
            smoothed_trans = _smooth_three_point(baseline_trans)
            clipped_trans = _clip_delta(smoothed_trans, baseline_trans, config.max_root_delta)
            clipped_trans[:, 1] = _clip_delta(
                clipped_trans[:, 1],
                baseline_trans[:, 1],
                config.max_root_vertical_delta,
            )
            candidate_trans = np.asarray(candidate[trans_key]).copy()
            candidate_trans[:, :3] = clipped_trans.astype(trans_dtype, copy=False)
            candidate[trans_key] = candidate_trans

    report = _base_report(record, candidate, pose_key, trans_key, True)
    return candidate, report
