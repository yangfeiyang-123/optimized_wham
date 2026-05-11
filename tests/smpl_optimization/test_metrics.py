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


def test_root_vertical_jitter_is_second_difference_rms():
    trans = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    report = root_vertical_jitter(trans)
    assert report["rms_vertical_accel"] > 0.0


def test_pose_delta_max_abs_compares_matching_prefix():
    original = np.zeros((2, 72), dtype=np.float32)
    corrected = original.copy()
    corrected[1, 10] = -0.5
    assert pose_delta_max_abs(original, corrected) == 0.5
