import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.whole_body_smooth import (  # noqa: E402
    WholeBodySmoothConfig,
    build_region_weights,
    smooth_record,
)


def test_build_region_weights_prioritizes_lower_body():
    weights = build_region_weights()

    assert weights.shape == (24, 3)
    assert weights[1].max() < weights[16].max()


def test_default_bounds_do_not_exceed_stage7_selector_gates():
    config = WholeBodySmoothConfig()

    assert config.max_lower_body_delta <= 0.05
    assert config.max_whole_body_delta <= 0.15
    assert config.max_root_delta <= 0.03
    assert config.max_root_vertical_delta <= 0.015


def test_smooth_record_preserves_betas_and_bounds_pose_delta():
    pose = np.zeros((5, 75), dtype=np.float32)
    pose[2, 4] = 1.0
    pose[:, 72:] = np.asarray([10.0, 11.0, 12.0], dtype=np.float32)
    betas = np.arange(10, dtype=np.float32)
    record = {
        "pose_world": pose.copy(),
        "betas": betas.copy(),
        "trans_world": np.zeros((5, 3), dtype=np.float32),
    }
    config = WholeBodySmoothConfig(
        max_lower_body_delta=0.05,
        max_whole_body_delta=0.08,
        max_root_delta=0.1,
        max_root_vertical_delta=0.02,
    )

    candidate, report = smooth_record(record, config)

    assert report["candidate_generated"] is True
    np.testing.assert_array_equal(candidate["betas"], betas)
    np.testing.assert_array_equal(candidate["pose_world"][:, 72:], pose[:, 72:])
    assert report["pose_delta"]["lower_body_max_abs"] <= config.max_lower_body_delta
    assert report["pose_delta"]["whole_body_max_abs"] <= config.max_whole_body_delta
    assert np.max(np.abs(candidate["pose_world"][:, :72] - pose[:, :72])) <= (
        config.max_whole_body_delta + 1e-7
    )
    np.testing.assert_array_equal(record["pose_world"], pose)


def test_smooth_record_drops_stale_derived_geometry_after_pose_change():
    pose = np.zeros((5, 72), dtype=np.float32)
    pose[2, 4] = 1.0
    record = {
        "pose_world": pose,
        "trans_world": np.zeros((5, 3), dtype=np.float32),
        "betas": np.zeros((5, 10), dtype=np.float32),
        "feet_world": np.ones((5, 4, 3), dtype=np.float32),
        "feet_refined": np.ones((5, 4, 3), dtype=np.float32),
        "verts": np.ones((5, 10, 3), dtype=np.float32),
        "joints_world": np.ones((5, 24, 3), dtype=np.float32),
    }

    candidate, report = smooth_record(record)

    assert report["candidate_generated"] is True
    for key in ("feet_world", "feet_refined", "feet", "verts", "joints_world"):
        assert key not in candidate
    assert "feet_world" in record


def test_smooth_record_reports_selector_ready_smoothness_when_trans_exists():
    pose = np.zeros((5, 72), dtype=np.float32)
    pose[2, 20] = 0.5
    trans = np.zeros((5, 3), dtype=np.float32)
    trans[2, 1] = 0.03
    record = {"pose": pose, "trans": trans}

    _, report = smooth_record(record)

    assert report["pose_smoothness"] == report["pose_smoothness_after"]
    assert report["root_smoothness"] == report["root_smoothness_after"]
    assert report["root_smoothness_available"] is True
    assert report["pose_smoothness"]["jerk"]["rms"] >= 0.0
    assert report["root_smoothness"]["jerk"]["rms"] >= 0.0


def test_missing_trans_marks_root_smoothness_unavailable():
    pose = np.zeros((5, 72), dtype=np.float32)
    pose[2, 4] = 0.5
    record = {"pose": pose}

    _, report = smooth_record(record)

    assert report["root_smoothness_available"] is False
    assert report["root_smoothness"] == {}
    assert report["root_delta"] == {"max_abs": 0.0, "vertical_max_abs": 0.0}


def test_extra_pose_values_beyond_smpl_pose_remain_unchanged():
    pose = np.zeros((5, 78), dtype=np.float32)
    pose[2, 4] = 1.0
    pose[:, 72:] = np.arange(30, dtype=np.float32).reshape(5, 6)

    candidate, _ = smooth_record({"pose": pose})

    np.testing.assert_array_equal(candidate["pose"][:, 72:], pose[:, 72:])


def test_smooth_record_preserves_short_sequences():
    pose = np.zeros((2, 72), dtype=np.float32)
    record = {"pose": pose.copy(), "betas": np.ones(10, dtype=np.float32)}

    candidate, report = smooth_record(record)

    assert report["candidate_generated"] is False
    assert "too_short" in report["reason"]
    np.testing.assert_array_equal(candidate["pose"], pose)
    np.testing.assert_array_equal(candidate["betas"], record["betas"])
    assert report["root_delta"]["max_abs"] == 0.0
    assert report["root_smoothness_available"] is False
    assert report["root_smoothness"] == {}
