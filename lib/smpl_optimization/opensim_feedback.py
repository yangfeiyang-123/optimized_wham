from pathlib import Path
import re

import numpy as np


_IK_LOG_PATTERN = re.compile(
    r"Frame\s+(?P<frame>\d+).*?"
    r"RMS\s*=\s*(?P<rms>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?).*?"
    r"max\s*=\s*(?P<max>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?).*?"
    r"\((?P<marker>[^)]*)\)"
)


def _neutral_metrics():
    return {
        "num_frames": 0,
        "mean_rms": 0.0,
        "max_rms": 0.0,
        "max_marker_name": "",
        "frames": [],
    }


def _clean_float(value):
    return float(round(float(value), 8))


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
        "num_frames": len(frames),
        "mean_rms": _clean_float(np.mean(rms_values)),
        "max_rms": _clean_float(np.max(rms_values)),
        "max_marker_name": max_frame["max_marker"],
        "frames": frames,
    }


def build_feedback_weights(metrics: dict, num_frames: int, rms_threshold=0.08) -> np.ndarray:
    weights = np.ones(int(num_frames), dtype=np.float32)
    for frame_metrics in metrics.get("frames", []):
        frame = int(frame_metrics.get("frame", -1))
        if 0 <= frame < num_frames and float(frame_metrics.get("rms", 0.0)) > float(rms_threshold):
            weights[frame] = 2.0
    return weights
