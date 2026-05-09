from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

PER_FRAME_FIELDS = {
    "pose",
    "pose_world",
    "trans",
    "trans_world",
    "verts",
    "joints",
    "joints_world",
}


def to_numpy(value: Any) -> np.ndarray:
    """Convert tensor-like or array-like values to a NumPy array."""
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        return value.numpy()
    return np.asarray(value)


def track_length(record: Mapping[str, Any]) -> int:
    for key in ("frame_ids", "frame_id", "pose", "pose_world"):
        if key in record:
            return len(record[key])
    raise ValueError("Cannot infer track length from WHAM record.")


def track_frame_ids(record: Mapping[str, Any]) -> np.ndarray:
    if "frame_ids" in record:
        return np.asarray(to_numpy(record["frame_ids"]), dtype=np.int64)
    if "frame_id" in record:
        return np.asarray(to_numpy(record["frame_id"]), dtype=np.int64)
    return np.arange(track_length(record), dtype=np.int64)


def _has_track_length(value: Any, n_frames: int) -> bool:
    array = to_numpy(value)
    return array.ndim > 0 and array.shape[0] == n_frames


def _selected_samples(
    results: Mapping[Any, Mapping[str, Any]],
) -> tuple[list[int], dict[int, tuple[int, Any, int]]]:
    samples: dict[int, tuple[int, Any, int]] = {}
    for key, record in results.items():
        n_frames = track_length(record)
        for local_idx, frame_id in enumerate(track_frame_ids(record)):
            frame_id = int(frame_id)
            current = samples.get(frame_id)
            candidate = (n_frames, key, local_idx)
            if current is None or candidate[0] > current[0]:
                samples[frame_id] = candidate

    if not samples:
        raise ValueError("No frames found in WHAM output.")

    return sorted(samples), samples


def _merge_per_frame_field(
    field: str,
    ordered_frames: list[int],
    samples: Mapping[int, tuple[int, Any, int]],
    results: Mapping[Any, Mapping[str, Any]],
) -> np.ndarray | None:
    values = []
    for frame_id in ordered_frames:
        _, key, local_idx = samples[frame_id]
        record = results[key]
        if field not in record:
            return None
        value = record[field]
        if not _has_track_length(value, track_length(record)):
            return None
        values.append(to_numpy(value)[local_idx])
    return np.stack(values, axis=0)


def _merge_betas(
    ordered_frames: list[int],
    samples: Mapping[int, tuple[int, Any, int]],
    results: Mapping[Any, Mapping[str, Any]],
) -> np.ndarray | None:
    values = []
    for frame_id in ordered_frames:
        _, key, local_idx = samples[frame_id]
        record = results[key]
        if "betas" not in record:
            return None

        betas = to_numpy(record["betas"])
        if betas.ndim == 1:
            values.append(betas.copy())
        elif betas.ndim == 2 and betas.shape[0] == track_length(record):
            values.append(betas[local_idx])
        else:
            return None
    return np.stack(values, axis=0)


def merge_tracks(results: Mapping[Any, Mapping[str, Any]]) -> dict[str, Any]:
    ordered_frames, samples = _selected_samples(results)
    merged: dict[str, Any] = {}

    for field in PER_FRAME_FIELDS:
        value = _merge_per_frame_field(field, ordered_frames, samples, results)
        if value is not None:
            merged[field] = value

    betas = _merge_betas(ordered_frames, samples, results)
    if betas is not None:
        merged["betas"] = betas

    merged["frame_ids"] = np.asarray(ordered_frames, dtype=np.int64)
    merged["frame_id"] = np.asarray(ordered_frames, dtype=np.int64)
    return merged


def select_track(results: Mapping[Any, Mapping[str, Any]], track_id: str | None = "merge"):
    if track_id is None:
        key = sorted(results.keys(), key=lambda x: str(x))[0]
        return key, results[key]
    if track_id in results:
        return track_id, results[track_id]
    for key, record in results.items():
        if str(key) == str(track_id):
            return key, record
    if track_id == "merge":
        return "merged", merge_tracks(results)
    if track_id == "longest":
        key = max(results.keys(), key=lambda k: track_length(results[k]))
        return key, results[key]
    raise KeyError(f"Track id {track_id!r} not found. Available: {list(results.keys())}")
