import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.lower_body import (
    LowerBodyOptimizerConfig,
    _frame_weight_report,
    _fit_frame_weights,
    optimize_record,
)
from scripts.optimize_smpl_lower_body import load_frame_weights


def test_config_accepts_frame_weights():
    weights = np.asarray([1.0, 2.0, 1.5], dtype=np.float32)
    config = LowerBodyOptimizerConfig(fps=60.0, frame_weights=weights)
    np.testing.assert_allclose(config.frame_weights, weights)


def test_load_frame_weights_reads_numpy_file(tmp_path):
    path = tmp_path / "weights.npy"
    np.save(path, np.asarray([1.0, 2.5], dtype=np.float32))
    weights = load_frame_weights(path)
    np.testing.assert_allclose(weights, np.asarray([1.0, 2.5], dtype=np.float32))
    assert weights.dtype == np.float32


def test_frame_weight_report_has_neutral_defaults_when_missing():
    report = _frame_weight_report(None, 3)
    assert report == {
        "provided": False,
        "num_frames": 3,
        "min": 1.0,
        "max": 1.0,
        "mean": 1.0,
        "used_for_pose_pass": False,
    }


def test_frame_weight_report_rejects_mismatched_length():
    weights = np.asarray([1.0, 2.0], dtype=np.float32)
    try:
        _frame_weight_report(weights, 3, used_for_pose_pass=True)
    except ValueError as exc:
        assert "match num_frames" in str(exc)
    else:
        raise AssertionError("expected mismatched frame weights to raise")


def test_fit_frame_weights_rejects_invalid_values():
    for weights in (
        np.asarray([1.0, -0.1], dtype=np.float32),
        np.asarray([1.0, np.inf], dtype=np.float32),
    ):
        try:
            _fit_frame_weights(weights, n_frames=2)
        except ValueError as exc:
            assert "non-finite or negative" in str(exc)
        else:
            raise AssertionError("expected invalid frame weights to raise")


def test_optimize_record_reports_weights_ignored_when_pose_pass_disabled():
    weights = np.asarray([1.0, 2.0], dtype=np.float32)
    record = {
        "trans_world": np.zeros((2, 3), dtype=np.float32),
        "pose": np.zeros((2, 72), dtype=np.float32),
        "betas": np.zeros((2, 10), dtype=np.float32),
    }
    _, reports = optimize_record(
        record,
        LowerBodyOptimizerConfig(fps=60.0, frame_weights=weights, enable_pose_pass=False),
    )

    frame_report = reports["lower_body_optimization_report"]["frame_weights"]
    assert frame_report == {
        "provided": True,
        "num_frames": 2,
        "min": 1.0,
        "max": 2.0,
        "mean": 1.5,
        "used_for_pose_pass": False,
        "ignored_reason": "pose_pass_disabled",
    }
