from pathlib import Path
import re

import numpy as np


LOWER_BODY_MARKERS = {
    "hip_l",
    "hip_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
    "ltoe",
    "rtoe",
    "toe_l",
    "toe_r",
    "l_foot_touch",
    "r_foot_touch",
}

_NON_LOWER_BODY_HINTS = (
    "head",
    "shoulder",
    "elbow",
    "wrist",
    "chest",
    "pelvis",
)

_IK_LOG_PATTERN = re.compile(
    r"Frame\s+(?P<frame>\d+).*?"
    r"RMS\s*=\s*(?P<rms>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?).*?"
    r"max\s*=\s*(?P<max>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?).*?"
    r"\((?P<marker>[^)]*)\)"
)


def _neutral_metrics():
    return {
        "valid": False,
        "parse_status": "no_frames",
        "num_frames": 0,
        "mean_rms": None,
        "max_rms": None,
        "max_marker_name": "",
        "frames": [],
    }


def _clean_float(value):
    return float(round(float(value), 8))


def _normalize_marker_name(name) -> str:
    return str(name or "").strip().lower()


def is_lower_body_marker(name) -> bool:
    return _normalize_marker_name(name) in LOWER_BODY_MARKERS


def classify_marker(name) -> str:
    marker = _normalize_marker_name(name)
    if not marker:
        return "unknown"
    if marker in LOWER_BODY_MARKERS:
        return "lower_body"
    if any(hint in marker for hint in _NON_LOWER_BODY_HINTS):
        return "non_lower_body"
    return "unknown"


def parse_ik_log_metrics(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    frames = []

    for line in text.splitlines():
        match = _IK_LOG_PATTERN.search(line)
        if match is None:
            continue

        frames.append(
            {
                "frame": int(match.group("frame")),
                "rms": float(match.group("rms")),
                "max": float(match.group("max")),
                "max_marker": match.group("marker").strip(),
            }
        )

    if not frames:
        return _neutral_metrics()

    rms_values = np.asarray([frame["rms"] for frame in frames], dtype=np.float64)
    max_frame = max(frames, key=lambda frame: frame["max"])
    return {
        "valid": True,
        "parse_status": "ok",
        "num_frames": len(frames),
        "mean_rms": _clean_float(np.mean(rms_values)),
        "max_rms": _clean_float(np.max(rms_values)),
        "max_marker_name": max_frame["max_marker"],
        "frames": frames,
    }


def _weight_from_rms(rms, rms_threshold, min_weight, max_weight):
    if rms_threshold <= 0:
        return max_weight
    severity = max(0.0, (float(rms) / float(rms_threshold)) - 1.0)
    return min(max_weight, min_weight + (max_weight - min_weight) * severity)


def _apply_smoothed_weight(weights, frame, min_weight, center_weight, smooth_radius):
    if smooth_radius <= 0:
        weights[frame] = max(weights[frame], center_weight)
        return

    start = max(0, frame - smooth_radius)
    stop = min(len(weights) - 1, frame + smooth_radius)
    span = smooth_radius + 1
    amplitude = center_weight - min_weight
    for index in range(start, stop + 1):
        distance = abs(index - frame)
        factor = (span - distance) / span
        candidate = min_weight + amplitude * factor
        weights[index] = max(weights[index], candidate)


def build_feedback_weights(
    metrics: dict,
    num_frames: int,
    rms_threshold=0.08,
    min_weight=1.0,
    max_weight=2.5,
    smooth_radius=1,
) -> dict:
    n = max(0, int(num_frames))
    min_weight = float(min_weight)
    max_weight = float(max_weight)
    smooth_radius = max(0, int(smooth_radius))
    weights = np.full(n, min_weight, dtype=np.float32)
    num_weighted_frames = 0
    ignored_non_lower_body_frames = []
    ignored_unknown_marker_frames = []

    for frame_metrics in metrics.get("frames", []):
        frame = int(frame_metrics.get("frame", -1))
        if not 0 <= frame < n:
            continue
        if float(frame_metrics.get("rms", 0.0)) <= float(rms_threshold):
            continue

        marker_class = classify_marker(frame_metrics.get("max_marker", ""))
        if marker_class == "lower_body":
            center_weight = _weight_from_rms(
                frame_metrics.get("rms", 0.0),
                rms_threshold,
                min_weight,
                max_weight,
            )
            _apply_smoothed_weight(weights, frame, min_weight, center_weight, smooth_radius)
            num_weighted_frames += 1
        elif marker_class == "non_lower_body":
            ignored_non_lower_body_frames.append(frame)
        else:
            ignored_unknown_marker_frames.append(frame)

    return {
        "weights": weights.astype(np.float32, copy=False),
        "num_frames": n,
        "num_weighted_frames": num_weighted_frames,
        "min_weight": float(np.min(weights)) if n else min_weight,
        "max_weight": float(np.max(weights)) if n else min_weight,
        "mean_weight": float(np.mean(weights)) if n else min_weight,
        "ignored_non_lower_body_frames": ignored_non_lower_body_frames,
        "ignored_unknown_marker_frames": ignored_unknown_marker_frames,
    }
