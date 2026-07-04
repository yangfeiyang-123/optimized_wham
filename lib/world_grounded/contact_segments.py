from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass(frozen=True)
class StanceSegment:
    foot_index: int
    label: str
    start: int
    end: int
    confidence_mean: float
    confidence_min: float
    length: int


def kinematic_stance_gate(
    foot_points: np.ndarray,
    fps: float,
    *,
    ground_y: float = 0.0,
    height_threshold: float = 0.05,
    speed_threshold: float = 0.15,
    vertical_axis: int = 1,
) -> np.ndarray:
    """Per-frame [T, K] bool mask: a foot may be in stance only when it is low AND slow.

    WHAM's contact-confidence signal saturates for footage where the actor never
    fully leaves the ground (e.g. a badminton pivot), so hysteresis on confidence
    alone marks every frame as stance and the downstream anchor pins erase real
    footwork. Gating stance on the foot's own kinematics restores genuine steps:
    a frame counts as a stance candidate only when the foot point is within
    ``height_threshold`` of the ground and moving slower than ``speed_threshold``
    horizontally. Callers AND this with the confidence-based segmentation.

    ``foot_points`` is [T, K, 3] in the same world frame as the anchors (WHAM Y-up
    by default, hence ``vertical_axis=1``; the two horizontal axes form the speed).
    """
    points = np.asarray(foot_points, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"foot_points must have shape [T, K, 3], got {points.shape}")
    n_frames, n_feet = points.shape[0], points.shape[1]
    if n_frames == 0:
        return np.zeros((0, n_feet), dtype=bool)

    horizontal_axes = [axis for axis in range(3) if axis != int(vertical_axis)]
    height = points[:, :, int(vertical_axis)] - float(ground_y)

    horizontal = points[:, :, horizontal_axes]
    # Centered horizontal velocity: |x[t+1] - x[t-1]| * fps / 2, with one-sided
    # differences at the endpoints. A centered estimate keeps a foot's touchdown and
    # takeoff frames (where one side is still moving) correctly classified as motion.
    speed = np.zeros((n_frames, n_feet), dtype=np.float64)
    if n_frames >= 3:
        centered = (horizontal[2:] - horizontal[:-2]) * (float(fps) / 2.0)
        speed[1:-1] = np.linalg.norm(centered, axis=2)
    if n_frames >= 2:
        speed[0] = np.linalg.norm((horizontal[1] - horizontal[0]) * float(fps), axis=1)
        speed[-1] = np.linalg.norm((horizontal[-1] - horizontal[-2]) * float(fps), axis=1)

    low = height <= float(height_threshold)
    slow = speed <= float(speed_threshold)
    return (low & slow).astype(bool)


def extract_stance_segments(
    contact_confidence: np.ndarray,
    foot_labels: Union[list[str], tuple[str, ...]],
    enter_threshold: float = 0.55,
    exit_threshold: float = 0.30,
    min_segment_len: int = 4,
    merge_gap: int = 2,
    kinematic_gate: Optional[np.ndarray] = None,
) -> tuple[list[StanceSegment], np.ndarray]:
    confidence = np.asarray(contact_confidence, dtype=np.float32)
    if confidence.ndim != 2:
        raise ValueError(f"contact_confidence must have shape [T, K], got {confidence.shape}")
    n_frames, n_feet = confidence.shape
    labels = _fit_labels(foot_labels, n_feet)

    gate = _fit_gate(kinematic_gate, n_frames, n_feet)

    segments: list[StanceSegment] = []
    for foot_index in range(n_feet):
        raw = _raw_segments_for_foot(
            confidence[:, foot_index],
            foot_index=foot_index,
            label=labels[foot_index],
            enter_threshold=float(enter_threshold),
            exit_threshold=float(exit_threshold),
            gate=None if gate is None else gate[:, foot_index],
        )
        merged = _merge_close_segments(raw, confidence[:, foot_index], max(0, int(merge_gap)))
        segments.extend(segment for segment in merged if segment.length >= int(min_segment_len))

    segments.sort(key=lambda item: (item.start, item.foot_index, item.end))
    stance_mask = np.zeros((n_frames, n_feet), dtype=np.bool_)
    for segment in segments:
        stance_mask[segment.start : segment.end, segment.foot_index] = True
    return segments, stance_mask


def _fit_gate(gate: Optional[np.ndarray], n_frames: int, n_feet: int) -> Optional[np.ndarray]:
    if gate is None:
        return None
    fitted = np.asarray(gate, dtype=bool)
    if fitted.shape != (n_frames, n_feet):
        raise ValueError(
            f"kinematic_gate must match contact_confidence shape {(n_frames, n_feet)}, got {fitted.shape}"
        )
    return fitted


def _fit_labels(labels: Union[list[str], tuple[str, ...]], n_feet: int) -> list[str]:
    fitted = [str(label) for label in labels[:n_feet]]
    while len(fitted) < n_feet:
        fitted.append(f"foot_{len(fitted)}")
    return fitted


def _raw_segments_for_foot(
    values: np.ndarray,
    *,
    foot_index: int,
    label: str,
    enter_threshold: float,
    exit_threshold: float,
    gate: Optional[np.ndarray] = None,
) -> list[StanceSegment]:
    segments: list[StanceSegment] = []
    active = False
    start = 0
    for frame_idx, value in enumerate(values):
        gated_in = True if gate is None else bool(gate[frame_idx])
        if not active and value >= enter_threshold and gated_in:
            active = True
            start = frame_idx
        elif active and (value <= exit_threshold or not gated_in):
            segments.append(_make_segment(foot_index, label, start, frame_idx, values))
            active = False
    if active:
        segments.append(_make_segment(foot_index, label, start, len(values), values))
    return segments


def _merge_close_segments(
    segments: list[StanceSegment],
    values: np.ndarray,
    merge_gap: int,
) -> list[StanceSegment]:
    if not segments:
        return []

    merged = [segments[0]]
    for segment in segments[1:]:
        prev = merged[-1]
        gap = segment.start - prev.end
        if gap <= merge_gap and prev.length > 1 and segment.length > 1:
            merged[-1] = _make_segment(prev.foot_index, prev.label, prev.start, segment.end, values)
        else:
            merged.append(segment)
    return merged


def _make_segment(
    foot_index: int,
    label: str,
    start: int,
    end: int,
    values: np.ndarray,
) -> StanceSegment:
    window = np.asarray(values[start:end], dtype=np.float32)
    if window.size == 0:
        mean = 0.0
        min_value = 0.0
    else:
        mean = float(np.mean(window))
        min_value = float(np.min(window))
    return StanceSegment(
        foot_index=int(foot_index),
        label=str(label),
        start=int(start),
        end=int(end),
        confidence_mean=mean,
        confidence_min=min_value,
        length=int(max(0, end - start)),
    )
