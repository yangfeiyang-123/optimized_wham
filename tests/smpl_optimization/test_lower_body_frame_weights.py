import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.lower_body import LowerBodyOptimizerConfig, _frame_weight_report
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
        "min_weight": 1.0,
        "max_weight": 1.0,
        "mean_weight": 1.0,
    }
