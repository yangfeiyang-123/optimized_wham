import numpy as np

from lib.smpl_optimization.metrics import (
    beta_variation_max_abs,
    contact_foot_sliding,
    foot_penetration_depth,
    root_vertical_jitter,
    pose_delta_max_abs,
)


def test_beta_variation_max_abs_zero_for_fixed_beta():
    betas = np.tile(np.arange(10, dtype=np.float32), (4, 1))
    assert beta_variation_max_abs(betas) == 0.0


def test_beta_variation_max_abs_treats_1d_beta_as_single_frame():
    betas = np.arange(10, dtype=np.float32)
    assert beta_variation_max_abs(betas) == 0.0


def test_beta_variation_max_abs_detects_frame_change():
    betas = np.zeros((3, 10), dtype=np.float32)
    betas[2, 4] = 0.25
    assert beta_variation_max_abs(betas) == 0.25


def test_foot_penetration_depth_reports_max_and_mean():
    foot_y = np.array([[0.1, -0.2], [-0.05, 0.0]], dtype=np.float32)
    report = foot_penetration_depth(foot_y, ground_y=0.0)
    assert report["max_penetration"] == 0.2
    assert np.isclose(report["mean_penetration"], 0.0625)
    assert report["num_penetrating"] == 2


def test_contact_foot_sliding_uses_contact_mask():
    points = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.2, 0.0, 0.0]],
            [[0.5, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    contact = np.array([[1.0], [1.0], [0.0]], dtype=np.float32)
    report = contact_foot_sliding(points, contact, fps=10.0, threshold=1.0)
    assert np.isclose(report["mean_contact_speed"], 2.0)
    assert report["num_sliding"] == 1


def test_contact_foot_sliding_ignores_vertical_motion():
    points = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    contact = np.ones((3, 1), dtype=np.float32)
    report = contact_foot_sliding(points, contact, fps=10.0, threshold=1.0)
    assert report["mean_contact_speed"] == 0.0
    assert report["max_contact_speed"] == 0.0
    assert report["num_sliding"] == 0


def test_root_vertical_jitter_is_second_difference_rms():
    trans = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    report = root_vertical_jitter(trans)
    assert report["rms_vertical_accel"] > 0.0


def test_root_vertical_jitter_returns_zero_for_1d_input():
    trans = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    report = root_vertical_jitter(trans)
    assert report["rms_vertical_accel"] == 0.0
    assert report["max_vertical_accel"] == 0.0


def test_root_vertical_jitter_returns_zero_for_rank_3_input():
    report = root_vertical_jitter(np.zeros((3, 1, 3), dtype=np.float32))
    assert report["rms_vertical_accel"] == 0.0
    assert report["max_vertical_accel"] == 0.0


def test_pose_delta_max_abs_compares_matching_prefix():
    original = np.zeros((2, 72), dtype=np.float32)
    corrected = original.copy()
    corrected[1, 10] = -0.5
    assert pose_delta_max_abs(original, corrected) == 0.5


def test_pose_delta_max_abs_treats_1d_pose_as_single_frame():
    original = np.zeros(72, dtype=np.float32)
    corrected = np.zeros((2, 72), dtype=np.float32)
    assert pose_delta_max_abs(original, corrected) == 0.0


def test_pose_delta_max_abs_flattens_non_frame_dimensions_before_comparing_prefix():
    original = np.zeros((1, 2, 3), dtype=np.float32)
    corrected = np.zeros((1, 2), dtype=np.float32)
    assert pose_delta_max_abs(original, corrected) == 0.0
