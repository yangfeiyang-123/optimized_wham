from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.ndimage import median_filter
except Exception:  # pragma: no cover - scipy is available in the target env.
    median_filter = None


@dataclass
class GroundEstimate:
    ground_y: float
    contact_confidence: np.ndarray
    sample_weights: np.ndarray
    contact_source: str
    ground_confidence: str


def _as_float_array(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _validate_foot_points(foot_points) -> np.ndarray:
    foot_points = _as_float_array(foot_points, "foot_points")
    if foot_points.ndim != 3 or foot_points.shape[-1] != 3:
        raise ValueError(f"foot_points must have shape [T, K, 3], got {foot_points.shape}")
    return foot_points


def _validate_contact(wham_contact, expected_shape: tuple[int, int]) -> np.ndarray | None:
    if wham_contact is None:
        return None
    contact = np.asarray(wham_contact, dtype=np.float64)
    if contact.shape != expected_shape:
        raise ValueError(f"wham_contact shape {contact.shape} does not match foot points {expected_shape}")
    return np.nan_to_num(contact, nan=0.0, posinf=1.0, neginf=0.0)


def _validate_fps(fps: float) -> float:
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be a positive finite value, got {fps!r}")
    return fps


def _validate_min_weighted_samples(min_weighted_samples: int) -> int:
    value = int(min_weighted_samples)
    if value < 1:
        raise ValueError("min_weighted_samples must be a positive integer.")
    return value


def _finite_values(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    return _weighted_quantile(values, weights, 0.5)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same flattened shape.")

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        finite = _finite_values(values)
        if finite.size == 0:
            raise ValueError("Cannot compute weighted median without finite values.")
        return float(np.nanmedian(finite))

    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = float(np.clip(quantile, 0.0, 1.0)) * float(cumulative[-1])
    return float(values[np.searchsorted(cumulative, cutoff, side="left")])


def _velocity(foot_points: np.ndarray, fps: float) -> np.ndarray:
    vel = np.zeros_like(foot_points, dtype=np.float64)
    if len(foot_points) > 1:
        vel[1:] = (foot_points[1:] - foot_points[:-1]) * float(fps)
        vel[0] = vel[1]
    return np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)


def _score_less(value: np.ndarray, threshold: float, sigma: float) -> np.ndarray:
    scaled = (np.asarray(value, dtype=np.float64) - threshold) / max(float(sigma), 1e-6)
    scaled = np.clip(scaled, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(scaled))


def _score_near_abs(value: np.ndarray, threshold: float, sigma: float) -> np.ndarray:
    return _score_less(np.abs(value), threshold, sigma)


def _smooth_confidence(confidence: np.ndarray, smooth_size: int) -> np.ndarray:
    if smooth_size <= 1 or len(confidence) <= 1:
        return confidence
    if median_filter is not None:
        return median_filter(confidence, size=(min(smooth_size, len(confidence)), 1), mode="nearest")

    size = min(int(smooth_size), len(confidence))
    radius_left = size // 2
    radius_right = size - radius_left - 1
    padded = np.pad(confidence, ((radius_left, radius_right), (0, 0)), mode="edge")
    out = np.empty_like(confidence)
    for idx in range(len(confidence)):
        out[idx] = np.median(padded[idx : idx + size], axis=0)
    return out


def _low_percentile_ground(heights: np.ndarray) -> float:
    finite = _finite_values(heights)
    if finite.size == 0:
        raise ValueError("Cannot estimate ground without finite foot heights.")
    return float(np.percentile(finite, 5.0))


def _initial_weights(foot_points: np.ndarray, fps: float, wham_contact: np.ndarray | None) -> tuple[np.ndarray, str]:
    vel = _velocity(foot_points, fps)
    horizontal_speed = np.linalg.norm(vel[..., [0, 2]], axis=-1)
    vertical_speed = np.abs(vel[..., 1])
    speed_weight = _score_less(horizontal_speed, 0.35, 0.15) * _score_less(vertical_speed, 0.35, 0.15)
    finite_height = np.isfinite(foot_points[..., 1]).astype(np.float64)
    weights = speed_weight * finite_height
    if wham_contact is None:
        low_height = _low_percentile_ground(foot_points[..., 1])
        height_weight = _score_less(foot_points[..., 1] - low_height, 0.08, 0.04)
        return weights * height_weight, "fallback_velocity_height"
    return weights * np.clip(wham_contact, 0.0, 1.0), "wham_contact"


def _fallback_ground_from_weights(heights: np.ndarray, weights: np.ndarray) -> float:
    if int(np.sum(weights > 0.05)) > 0:
        return _weighted_quantile(heights, weights, 0.2)
    return _low_percentile_ground(heights)


def compute_contact_confidence(
    foot_points: np.ndarray,
    fps: float,
    ground_y: float,
    wham_contact: np.ndarray | None = None,
    height_threshold: float = 0.08,
    horizontal_speed_threshold: float = 0.25,
    vertical_speed_threshold: float = 0.25,
    smooth_size: int = 5,
) -> np.ndarray:
    foot_points = _validate_foot_points(foot_points)
    fps = _validate_fps(fps)
    if not np.isfinite(float(ground_y)):
        raise ValueError(f"ground_y must be finite, got {ground_y!r}")
    wham_contact = _validate_contact(wham_contact, foot_points.shape[:2])

    vel = _velocity(foot_points, fps)
    height = foot_points[..., 1] - float(ground_y)
    horizontal_speed = np.linalg.norm(vel[..., [0, 2]], axis=-1)
    vertical_speed = np.abs(vel[..., 1])

    height_score = _score_near_abs(height, height_threshold, height_threshold * 0.5)
    horizontal_score = _score_less(
        horizontal_speed,
        horizontal_speed_threshold,
        horizontal_speed_threshold * 0.5,
    )
    vertical_score = _score_less(
        vertical_speed,
        vertical_speed_threshold,
        vertical_speed_threshold * 0.5,
    )
    confidence = height_score * horizontal_score * vertical_score
    confidence *= np.isfinite(foot_points[..., 1]).astype(np.float64)

    if wham_contact is not None:
        confidence *= np.clip(wham_contact, 0.0, 1.0)

    confidence = _smooth_confidence(confidence, smooth_size)
    return np.clip(np.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def estimate_ground(
    foot_points: np.ndarray,
    fps: float,
    wham_contact: np.ndarray | None = None,
    min_weighted_samples: int = 8,
) -> GroundEstimate:
    foot_points = _validate_foot_points(foot_points)
    fps = _validate_fps(fps)
    min_weighted_samples = _validate_min_weighted_samples(min_weighted_samples)
    wham_contact = _validate_contact(wham_contact, foot_points.shape[:2])

    heights = foot_points[..., 1]
    weights, contact_source = _initial_weights(foot_points, fps, wham_contact)
    active = int(np.sum(weights > 0.05))
    if wham_contact is not None and active > 0:
        ground_y = weighted_median(heights, weights)
    elif wham_contact is not None and np.any(wham_contact > 0.0):
        contact_only_weights = np.clip(wham_contact, 0.0, 1.0) * np.isfinite(heights).astype(np.float64)
        ground_y = weighted_median(heights, contact_only_weights)
    elif wham_contact is None:
        ground_y = _fallback_ground_from_weights(heights, weights)
    else:
        ground_y = _fallback_ground_from_weights(heights, weights)

    contact_confidence = compute_contact_confidence(foot_points, fps, ground_y, wham_contact)
    refined_weights = np.maximum(weights, contact_confidence)
    refined_active = int(np.sum(refined_weights > 0.05))
    if wham_contact is not None and refined_active >= min_weighted_samples:
        ground_y = weighted_median(heights, refined_weights)
        contact_confidence = compute_contact_confidence(foot_points, fps, ground_y, wham_contact)
        refined_weights = np.maximum(weights, contact_confidence)

    final_active = int(np.sum(refined_weights > 0.05))
    if final_active >= 30:
        ground_confidence = "high"
    elif final_active >= min_weighted_samples:
        ground_confidence = "medium"
    else:
        ground_confidence = "low"

    return GroundEstimate(
        ground_y=float(ground_y),
        contact_confidence=contact_confidence.astype(np.float32),
        sample_weights=np.clip(refined_weights, 0.0, 1.0).astype(np.float32),
        contact_source=contact_source,
        ground_confidence=ground_confidence,
    )
