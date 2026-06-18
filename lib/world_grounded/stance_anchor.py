from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lib.world_grounded.contact_segments import StanceSegment


@dataclass(frozen=True)
class FootAnchor:
    foot_index: int
    label: str
    start: int
    end: int
    anchor_xyz: np.ndarray
    anchor_xz: np.ndarray
    ground_y: float
    confidence: float
    source: str = "weighted_median"


def compute_stance_anchors(
    foot_points: np.ndarray,
    contact_confidence: np.ndarray,
    segments: list[StanceSegment],
    ground_y: float,
) -> list[FootAnchor]:
    points = np.asarray(foot_points, dtype=np.float32)
    confidence = np.asarray(contact_confidence, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"foot_points must have shape [T, K, 3], got {points.shape}")
    if confidence.shape != points.shape[:2]:
        raise ValueError("contact_confidence must match foot_points frame/foot dimensions.")

    anchors: list[FootAnchor] = []
    for segment in segments:
        point_window = points[segment.start : segment.end, segment.foot_index]
        weight_window = confidence[segment.start : segment.end, segment.foot_index]
        if point_window.size == 0:
            continue
        x = _weighted_lower_median(point_window[:, 0], weight_window)
        z = _weighted_lower_median(point_window[:, 2], weight_window)
        anchor_xyz = np.asarray([x, float(ground_y), z], dtype=np.float32)
        anchors.append(
            FootAnchor(
                foot_index=int(segment.foot_index),
                label=str(segment.label),
                start=int(segment.start),
                end=int(segment.end),
                anchor_xyz=anchor_xyz,
                anchor_xz=anchor_xyz[[0, 2]].astype(np.float32),
                ground_y=float(ground_y),
                confidence=float(segment.confidence_mean),
            )
        )
    return anchors


def rasterize_anchors(
    anchors: list[FootAnchor],
    n_frames: int,
    n_feet: int,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.zeros((int(n_frames), int(n_feet), 3), dtype=np.float32)
    mask = np.zeros((int(n_frames), int(n_feet)), dtype=np.bool_)
    for anchor in anchors:
        start = max(0, min(int(n_frames), int(anchor.start)))
        end = max(start, min(int(n_frames), int(anchor.end)))
        foot_index = int(anchor.foot_index)
        if foot_index < 0 or foot_index >= int(n_feet):
            continue
        targets[start:end, foot_index, :] = anchor.anchor_xyz
        mask[start:end, foot_index] = True
    return targets, mask


def anchors_to_jsonable(anchors: list[FootAnchor], *, coordinate_system: str, ground_y: float) -> dict:
    return {
        "coordinate_system": str(coordinate_system),
        "ground_y": float(ground_y),
        "anchors": [
            {
                "foot_index": int(anchor.foot_index),
                "label": anchor.label,
                "start": int(anchor.start),
                "end": int(anchor.end),
                "anchor_xyz": [float(v) for v in anchor.anchor_xyz],
                "anchor_xz": [float(v) for v in anchor.anchor_xz],
                "ground_y": float(anchor.ground_y),
                "confidence": float(anchor.confidence),
                "source": anchor.source,
            }
            for anchor in anchors
        ],
    }


def _weighted_lower_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    if not np.any(weights > 0.0):
        return float(np.median(values))

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    half = 0.5 * float(np.sum(sorted_weights))
    cumulative = np.cumsum(sorted_weights)
    idx = int(np.searchsorted(cumulative, half, side="left"))
    idx = min(idx, len(sorted_values) - 1)
    return float(sorted_values[idx])
