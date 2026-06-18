from __future__ import annotations

from dataclasses import dataclass
from typing import Union

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


def extract_stance_segments(
    contact_confidence: np.ndarray,
    foot_labels: Union[list[str], tuple[str, ...]],
    enter_threshold: float = 0.55,
    exit_threshold: float = 0.30,
    min_segment_len: int = 4,
    merge_gap: int = 2,
) -> tuple[list[StanceSegment], np.ndarray]:
    confidence = np.asarray(contact_confidence, dtype=np.float32)
    if confidence.ndim != 2:
        raise ValueError(f"contact_confidence must have shape [T, K], got {confidence.shape}")
    n_frames, n_feet = confidence.shape
    labels = _fit_labels(foot_labels, n_feet)

    segments: list[StanceSegment] = []
    for foot_index in range(n_feet):
        raw = _raw_segments_for_foot(
            confidence[:, foot_index],
            foot_index=foot_index,
            label=labels[foot_index],
            enter_threshold=float(enter_threshold),
            exit_threshold=float(exit_threshold),
        )
        merged = _merge_close_segments(raw, confidence[:, foot_index], max(0, int(merge_gap)))
        segments.extend(segment for segment in merged if segment.length >= int(min_segment_len))

    segments.sort(key=lambda item: (item.start, item.foot_index, item.end))
    stance_mask = np.zeros((n_frames, n_feet), dtype=np.bool_)
    for segment in segments:
        stance_mask[segment.start : segment.end, segment.foot_index] = True
    return segments, stance_mask


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
) -> list[StanceSegment]:
    segments: list[StanceSegment] = []
    active = False
    start = 0
    for frame_idx, value in enumerate(values):
        if not active and value >= enter_threshold:
            active = True
            start = frame_idx
        elif active and value <= exit_threshold:
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
