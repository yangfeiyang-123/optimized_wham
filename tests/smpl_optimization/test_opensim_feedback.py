from pathlib import Path

import numpy as np

from lib.smpl_optimization.opensim_feedback import (
    build_feedback_weights,
    parse_ik_log_metrics,
)


def test_parse_ik_log_metrics_extracts_rms_values(tmp_path: Path):
    log = tmp_path / "ik.log"
    log.write_text(
        "\n".join(
            [
                "[info] Frame 0 (t = 0.0): marker error: RMS = 0.05, max = 0.10 (ankle_r)",
                "[info] Frame 1 (t = 0.1): marker error: RMS = 0.08, max = 0.20 (knee_l)",
            ]
        ),
        encoding="utf-8",
    )
    metrics = parse_ik_log_metrics(log)
    assert metrics["num_frames"] == 2
    assert metrics["mean_rms"] == 0.065
    assert metrics["max_marker_name"] == "knee_l"


def test_parse_ik_log_metrics_returns_neutral_metrics_for_empty_log(tmp_path: Path):
    log = tmp_path / "empty.log"
    log.write_text("", encoding="utf-8")
    metrics = parse_ik_log_metrics(log)
    assert metrics == {
        "num_frames": 0,
        "mean_rms": 0.0,
        "max_rms": 0.0,
        "max_marker_name": "",
        "frames": [],
    }


def test_build_feedback_weights_marks_bad_frames():
    metrics = {"frames": [{"frame": 0, "rms": 0.03}, {"frame": 1, "rms": 0.12}]}
    weights = build_feedback_weights(metrics, num_frames=3, rms_threshold=0.08)
    assert weights.tolist() == [1.0, 2.0, 1.0]


def test_build_feedback_weights_ignores_out_of_range_frames():
    metrics = {
        "frames": [
            {"frame": -1, "rms": 0.12},
            {"frame": 1, "rms": 0.12},
            {"frame": 3, "rms": 0.12},
        ]
    }
    weights = build_feedback_weights(metrics, num_frames=3, rms_threshold=0.08)
    assert weights.dtype == np.float32
    assert weights.tolist() == [1.0, 2.0, 1.0]
