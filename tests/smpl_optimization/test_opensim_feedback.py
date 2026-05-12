import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.opensim_feedback import (  # noqa: E402
    build_feedback_weights,
    classify_marker,
    is_lower_body_marker,
    parse_ik_log_metrics,
)


def test_parse_ik_log_metrics_extracts_frames(tmp_path):
    log = tmp_path / "ik.log"
    log.write_text(
        "\n".join(
            [
                "[info] Frame 0 (t = 0.0): marker error: RMS = 0.040, max = 0.100 (RightShoulder)",
                "[info] Frame 1 (t = 0.1): marker error: RMS = 0.120, max = 0.220 (ankle_l)",
                "[info] Frame 2 (t = 0.2): marker error: RMS = 0.090, max = 0.180 (RTOE)",
            ]
        ),
        encoding="utf-8",
    )

    metrics = parse_ik_log_metrics(log)

    assert metrics["valid"] is True
    assert metrics["parse_status"] == "ok"
    assert metrics["num_frames"] == 3
    assert metrics["mean_rms"] == 0.08333333
    assert metrics["max_rms"] == 0.12
    assert metrics["max_marker_name"] == "ankle_l"
    assert metrics["frames"][1]["max_marker"] == "ankle_l"


def test_parse_ik_log_metrics_marks_empty_logs_invalid(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("[info] Running tool without frame rows\n", encoding="utf-8")

    metrics = parse_ik_log_metrics(log)

    assert metrics["valid"] is False
    assert metrics["parse_status"] == "no_frames"
    assert metrics["num_frames"] == 0
    assert metrics["mean_rms"] is None
    assert metrics["max_rms"] is None


def test_marker_classification_is_case_insensitive_for_lower_body():
    assert is_lower_body_marker("ankle_l")
    assert is_lower_body_marker("RTOE")
    assert is_lower_body_marker("knee_r")
    assert not is_lower_body_marker("RightShoulder")
    assert not is_lower_body_marker("head_retarget")
    assert classify_marker("RTOE") == "lower_body"
    assert classify_marker("RightShoulder") == "non_lower_body"


def test_build_feedback_weights_ignores_non_lower_body_marker_spikes():
    metrics = {
        "frames": [
            {"frame": 0, "rms": 0.20, "max": 0.40, "max_marker": "RightShoulder"},
            {"frame": 1, "rms": 0.20, "max": 0.40, "max_marker": "ankle_l"},
            {"frame": 2, "rms": 0.04, "max": 0.07, "max_marker": "ankle_l"},
        ]
    }

    result = build_feedback_weights(metrics, num_frames=4, rms_threshold=0.08, smooth_radius=0)

    np.testing.assert_allclose(result["weights"], np.asarray([1.0, 2.5, 1.0, 1.0], dtype=np.float32))
    assert result["num_weighted_frames"] == 1
    assert result["ignored_non_lower_body_frames"] == [0]


def test_build_feedback_weights_scales_near_threshold_errors():
    metrics = {
        "frames": [
            {"frame": 1, "rms": 0.081, "max": 0.10, "max_marker": "ankle_l"},
        ]
    }

    result = build_feedback_weights(metrics, num_frames=3, rms_threshold=0.08, smooth_radius=0)

    assert 1.0 < result["weights"][1] < 2.5


def test_build_feedback_weights_smooths_from_scaled_center_weight():
    metrics = {
        "frames": [
            {"frame": 2, "rms": 0.081, "max": 0.10, "max_marker": "ankle_l"},
        ]
    }

    result = build_feedback_weights(metrics, num_frames=5, rms_threshold=0.08, smooth_radius=1)

    weights = result["weights"]
    assert 1.0 < weights[2] < 2.5
    assert 1.0 < weights[1] < weights[2]
    assert 1.0 < weights[3] < weights[2]


def test_build_feedback_weights_smooths_lower_body_windows():
    metrics = {
        "frames": [
            {"frame": 2, "rms": 0.20, "max": 0.40, "max_marker": "ankle_l"},
        ]
    }

    result = build_feedback_weights(metrics, num_frames=5, rms_threshold=0.08, smooth_radius=1)

    weights = result["weights"]
    assert weights[2] == 2.5
    assert weights[1] > 1.0
    assert weights[3] > 1.0
    assert weights[0] == 1.0
    assert weights[4] == 1.0
