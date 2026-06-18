from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FootPointResult:
    points: np.ndarray
    labels: list[str]
    source: str


DEFAULT_FOOT_LABELS = [
    "left_heel_or_ankle",
    "left_toe_or_forefoot",
    "right_heel_or_ankle",
    "right_toe_or_forefoot",
]


def _validate_points(points: np.ndarray, source: str) -> Optional[FootPointResult]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[0] == 0 or points.shape[1] == 0:
        return None
    labels = DEFAULT_FOOT_LABELS[: points.shape[1]]
    if len(labels) < points.shape[1]:
        labels.extend([f"foot_point_{idx}" for idx in range(len(labels), points.shape[1])])
    return FootPointResult(points=points, labels=labels, source=source)


def get_record_foot_points(record: dict) -> FootPointResult:
    for key in ("feet_refined", "feet_world", "feet"):
        if key not in record:
            continue
        result = _validate_points(record[key], key)
        if result is not None:
            return result

    if "verts" in record:
        verts = np.asarray(record["verts"], dtype=np.float32)
        if verts.ndim == 3 and verts.shape[-1] == 3 and verts.shape[0] > 0 and verts.shape[1] > 0:
            return FootPointResult(
                points=_fallback_feet_from_low_vertices(verts),
                labels=DEFAULT_FOOT_LABELS.copy(),
                source="verts_low_fallback",
            )

    raise ValueError("Record does not contain valid feet_refined, feet_world, feet, or verts.")


def _fallback_feet_from_low_vertices(verts: np.ndarray) -> np.ndarray:
    n_frames = verts.shape[0]
    points = np.zeros((n_frames, 4, 3), dtype=np.float32)
    for frame_idx, frame in enumerate(verts):
        finite = np.all(np.isfinite(frame), axis=-1)
        frame = frame[finite]
        if len(frame) == 0:
            continue

        y_cut = np.percentile(frame[:, 1], 5.0)
        low = frame[frame[:, 1] <= y_cut]
        if len(low) < 4:
            low = frame[np.argsort(frame[:, 1])[: max(4, min(20, len(frame)))]]

        x_mid = np.median(low[:, 0])
        left = low[low[:, 0] <= x_mid]
        right = low[low[:, 0] > x_mid]
        for side_idx, side_points in enumerate((left, right)):
            if len(side_points) == 0:
                side_points = low
            heel = side_points[np.argmin(side_points[:, 2])]
            toe = side_points[np.argmax(side_points[:, 2])]
            points[frame_idx, side_idx * 2] = heel
            points[frame_idx, side_idx * 2 + 1] = toe
    return points


def get_record_contact(record: dict, n_frames: int, n_points: int) -> Optional[np.ndarray]:
    if "contact" not in record:
        return None
    contact = np.asarray(record["contact"], dtype=np.float32)
    if contact.ndim != 2 or contact.shape[0] == 0:
        return None

    contact = np.nan_to_num(contact[:n_frames], nan=0.0, posinf=1.0, neginf=0.0)
    contact = np.clip(contact, 0.0, 1.0)
    if contact.shape[1] == n_points:
        return contact
    if contact.shape[1] > n_points:
        return contact[:, :n_points]

    padded = np.zeros((len(contact), n_points), dtype=np.float32)
    padded[:, : contact.shape[1]] = contact
    return padded
