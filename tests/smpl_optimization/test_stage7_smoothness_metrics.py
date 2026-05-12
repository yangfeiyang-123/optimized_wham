import numpy as np

from lib.smpl_optimization.metrics import (
    pose_smoothness,
    root_translation_smoothness,
    temporal_derivative_summary,
)


def test_temporal_derivative_summary_reports_zero_for_constant_signal():
    values = np.ones((6, 3), dtype=np.float32)
    summary = temporal_derivative_summary(values, order=3)
    assert summary == {"rms": 0.0, "max_abs": 0.0}


def test_temporal_derivative_summary_treats_1d_input_as_single_frame():
    values = np.arange(72.0)
    assert temporal_derivative_summary(values, order=1) == {"rms": 0.0, "max_abs": 0.0}
    assert temporal_derivative_summary(values, order=2) == {"rms": 0.0, "max_abs": 0.0}
    assert temporal_derivative_summary(values, order=3) == {"rms": 0.0, "max_abs": 0.0}


def test_pose_smoothness_detects_jerky_pose():
    smooth = np.zeros((8, 72), dtype=np.float32)
    jerky = smooth.copy()
    jerky[4, 10] = 1.0

    smooth_report = pose_smoothness(smooth)
    jerky_report = pose_smoothness(jerky)

    assert jerky_report["jerk"]["rms"] > smooth_report["jerk"]["rms"]
    assert jerky_report["acceleration"]["max_abs"] > 0.0


def test_root_translation_smoothness_handles_short_sequences():
    report = root_translation_smoothness(np.zeros((2, 3), dtype=np.float32))
    assert report["velocity"]["rms"] == 0.0
    assert report["acceleration"]["rms"] == 0.0
    assert report["jerk"]["rms"] == 0.0


def test_pose_smoothness_treats_1d_pose_as_single_frame():
    report = pose_smoothness(np.arange(72.0))
    assert report["velocity"]["rms"] == 0.0
    assert report["acceleration"]["rms"] == 0.0
    assert report["jerk"]["rms"] == 0.0


def test_root_translation_smoothness_treats_1d_translation_as_single_frame():
    report = root_translation_smoothness(np.array([0.0, 1.0, 3.0]))
    assert report["velocity"]["rms"] == 0.0
    assert report["acceleration"]["rms"] == 0.0
    assert report["jerk"]["rms"] == 0.0
