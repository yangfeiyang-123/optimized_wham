from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RootOptimizationResult:
    optimized_trans_world: np.ndarray
    delta: np.ndarray
    report: dict


def _safe_savgol(values: np.ndarray, window: int = 9, polyorder: int = 2) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 5 or window <= 1:
        return values.copy()
    win = min(int(window), len(values) if len(values) % 2 == 1 else len(values) - 1)
    if win <= polyorder or win < 3:
        return values.copy()
    if win % 2 == 0:
        win -= 1
    return _moving_average(values, win)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    radius = max(1, int(window) // 2)
    out = np.empty_like(values, dtype=np.float64)
    for idx in range(len(values)):
        lo = max(0, idx - radius)
        hi = min(len(values), idx + radius + 1)
        out[idx] = np.mean(values[lo:hi], axis=0)
    return out


def _penetration_cm(foot_y: np.ndarray, ground_y: float) -> tuple[float, float]:
    penetration = np.maximum(0.0, float(ground_y) - np.asarray(foot_y, dtype=np.float64))
    return float(np.mean(penetration) * 100.0), float(np.max(penetration) * 100.0)


def _root_accel_score(trans: np.ndarray, fps: float) -> float:
    if len(trans) < 3:
        return 0.0
    acc = (trans[2:] - 2.0 * trans[1:-1] + trans[:-2]) * float(fps) ** 2
    return float(np.mean(np.linalg.norm(acc, axis=-1)))


def _validate_inputs(
    trans_world: np.ndarray,
    foot_points: np.ndarray,
    contact_confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trans_world = np.asarray(trans_world, dtype=np.float64)
    foot_points = np.asarray(foot_points, dtype=np.float64)
    contact_confidence = np.asarray(contact_confidence, dtype=np.float64)

    if trans_world.ndim != 2 or trans_world.shape[1] != 3:
        raise ValueError(f"trans_world must have shape [T, 3], got {trans_world.shape}")
    if foot_points.ndim != 3 or foot_points.shape[-1] != 3:
        raise ValueError(f"foot_points must have shape [T, K, 3], got {foot_points.shape}")
    if foot_points.shape[0] != trans_world.shape[0]:
        raise ValueError("trans_world and foot_points must have the same frame count.")
    if foot_points.shape[:2] != contact_confidence.shape:
        raise ValueError("foot_points and contact_confidence frame/point dimensions must match.")
    if trans_world.shape[0] == 0:
        raise ValueError("trans_world must contain at least one frame.")

    return (
        np.nan_to_num(trans_world, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(foot_points, nan=0.0, posinf=0.0, neginf=0.0),
        np.clip(np.nan_to_num(contact_confidence, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0),
    )


def optimize_root_translation(
    trans_world: np.ndarray,
    foot_points: np.ndarray,
    contact_confidence: np.ndarray,
    ground_y: float,
    fps: float,
    smooth_window: int = 9,
    contact_threshold: float = 0.35,
) -> RootOptimizationResult:
    trans_world, foot_points, contact_confidence = _validate_inputs(
        trans_world,
        foot_points,
        contact_confidence,
    )

    smoothed_trans = _safe_savgol(trans_world, smooth_window)
    delta = smoothed_trans - trans_world
    foot_after_smooth = foot_points + delta[:, None, :]

    active = contact_confidence > float(contact_threshold)
    correction_y = np.zeros(len(trans_world), dtype=np.float64)
    for frame_idx in range(len(trans_world)):
        mask = active[frame_idx]
        if np.any(mask):
            min_y = float(np.min(foot_after_smooth[frame_idx, mask, 1]))
            correction_y[frame_idx] = max(0.0, float(ground_y) - min_y)
        else:
            min_y = float(np.min(foot_after_smooth[frame_idx, :, 1]))
            correction_y[frame_idx] = max(0.0, float(ground_y) - min_y) * 0.25

    correction_y = _safe_savgol(correction_y[:, None], smooth_window)[:, 0]
    correction_y = np.maximum(correction_y, 0.0)
    delta[:, 1] += correction_y
    optimized = trans_world + delta

    before_mean, before_max = _penetration_cm(foot_points[..., 1], ground_y)
    after_mean, after_max = _penetration_cm((foot_points + delta[:, None, :])[..., 1], ground_y)
    report = {
        "root_acceleration_before": _root_accel_score(trans_world, fps),
        "root_acceleration_after": _root_accel_score(optimized, fps),
        "foot_penetration_mean_cm_before": before_mean,
        "foot_penetration_max_cm_before": before_max,
        "foot_penetration_mean_cm_after": after_mean,
        "foot_penetration_max_cm_after": after_max,
        "root_delta_mean_cm": float(np.mean(np.linalg.norm(delta, axis=-1)) * 100.0),
        "root_delta_max_cm": float(np.max(np.linalg.norm(delta, axis=-1)) * 100.0),
    }
    return RootOptimizationResult(
        optimized_trans_world=optimized.astype(np.float32),
        delta=delta.astype(np.float32),
        report=report,
    )
