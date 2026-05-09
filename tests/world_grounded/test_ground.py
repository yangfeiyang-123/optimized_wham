import numpy as np
import pytest

import lib.world_grounded.ground as ground_module
from lib.world_grounded.ground import (
    compute_contact_confidence,
    estimate_ground,
    weighted_median,
)


def test_weighted_median_ignores_low_weight_outlier():
    values = np.array([0.0, 0.01, 0.02, -2.0], dtype=np.float64)
    weights = np.array([1.0, 1.0, 1.0, 0.001], dtype=np.float64)

    assert abs(weighted_median(values, weights) - 0.01) < 1e-6


def test_estimate_ground_uses_contact_over_raw_minimum():
    foot_points = np.zeros((20, 4, 3), dtype=np.float64)
    foot_points[..., 1] = 0.05
    foot_points[:, 0, 1] = 0.0
    foot_points[5, 3, 1] = -1.0
    contact = np.zeros((20, 4), dtype=np.float64)
    contact[:, 0] = 1.0
    contact[5, 3] = 0.0

    result = estimate_ground(foot_points, fps=60.0, wham_contact=contact)

    assert abs(result.ground_y) < 0.02
    assert result.contact_source == "wham_contact"
    assert result.ground_confidence in {"medium", "high"}


def test_compute_contact_confidence_shape_and_range():
    foot_points = np.zeros((6, 4, 3), dtype=np.float64)
    contact = np.ones((6, 4), dtype=np.float64)

    conf = compute_contact_confidence(
        foot_points,
        fps=30.0,
        ground_y=0.0,
        wham_contact=contact,
    )

    assert conf.shape == (6, 4)
    assert float(conf.min()) >= 0.0
    assert float(conf.max()) <= 1.0


def test_estimate_ground_falls_back_without_contact():
    foot_points = np.zeros((12, 4, 3), dtype=np.float64)
    foot_points[..., 1] = 0.12
    foot_points[:, 1, 1] = 0.02

    result = estimate_ground(foot_points, fps=30.0)

    assert result.contact_source == "fallback_velocity_height"
    assert abs(result.ground_y - 0.02) < 0.02
    assert result.contact_confidence.shape == (12, 4)


def test_estimate_ground_all_zero_contact_falls_back_to_low_percentile():
    foot_points = np.zeros((10, 4, 3), dtype=np.float64)
    foot_points[..., 1] = 0.2
    foot_points[:, 2, 1] = 0.03
    contact = np.zeros((10, 4), dtype=np.float64)

    result = estimate_ground(foot_points, fps=30.0, wham_contact=contact)

    assert result.contact_source == "wham_contact"
    assert abs(result.ground_y - 0.03) < 0.03
    assert result.ground_confidence == "low"


def test_estimate_ground_sparse_contact_ignores_non_contact_outlier():
    foot_points = np.zeros((8, 4, 3), dtype=np.float64)
    foot_points[..., 1] = 0.15
    foot_points[0, 0, 1] = 0.0
    foot_points[0, 3, 1] = -1.0
    contact = np.zeros((8, 4), dtype=np.float64)
    contact[0, 0] = 1.0

    result = estimate_ground(foot_points, fps=30.0, wham_contact=contact)

    assert abs(result.ground_y) < 1e-6
    assert result.ground_confidence == "low"


def test_estimate_ground_without_contact_downweights_low_outlier():
    foot_points = np.zeros((12, 4, 3), dtype=np.float64)
    foot_points[..., 1] = 0.12
    foot_points[:, 0, 1] = 0.0
    foot_points[0, 1, 1] = -0.05

    result = estimate_ground(foot_points, fps=30.0)

    assert abs(result.ground_y) < 0.02


def test_estimate_ground_rejects_contact_shape_mismatch():
    foot_points = np.zeros((5, 4, 3), dtype=np.float64)
    contact = np.ones((5, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="wham_contact"):
        estimate_ground(foot_points, fps=30.0, wham_contact=contact)


@pytest.mark.parametrize("fps", [0.0, -30.0, np.nan, np.inf])
def test_estimate_ground_rejects_invalid_fps(fps):
    foot_points = np.zeros((5, 4, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="fps"):
        estimate_ground(foot_points, fps=fps)


@pytest.mark.parametrize("fps", [0.0, -30.0, np.nan, np.inf])
def test_compute_contact_confidence_rejects_invalid_fps(fps):
    foot_points = np.zeros((5, 4, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="fps"):
        compute_contact_confidence(foot_points, fps=fps, ground_y=0.0)


def test_estimate_ground_rejects_invalid_min_weighted_samples():
    foot_points = np.zeros((5, 4, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="min_weighted_samples"):
        estimate_ground(foot_points, fps=30.0, min_weighted_samples=0)


def test_contact_smoothing_fallback_matches_scipy(monkeypatch):
    if ground_module.median_filter is None:
        pytest.skip("scipy median_filter is not available")

    foot_points = np.zeros((7, 4, 3), dtype=np.float64)
    foot_points[0, :, 1] = 0.5

    scipy_conf = compute_contact_confidence(foot_points, fps=30.0, ground_y=0.0, smooth_size=5)
    monkeypatch.setattr(ground_module, "median_filter", None)
    fallback_conf = compute_contact_confidence(foot_points, fps=30.0, ground_y=0.0, smooth_size=5)

    assert np.allclose(fallback_conf, scipy_conf)


def test_estimate_ground_ignores_nan_and_inf_samples():
    foot_points = np.zeros((8, 4, 3), dtype=np.float64)
    foot_points[..., 1] = 0.04
    foot_points[:, 0, 1] = 0.0
    foot_points[0, 1, 1] = np.nan
    foot_points[1, 2, 1] = np.inf
    contact = np.ones((8, 4), dtype=np.float64)

    result = estimate_ground(foot_points, fps=30.0, wham_contact=contact)

    assert np.isfinite(result.ground_y)
    assert np.all(np.isfinite(result.contact_confidence))
    assert np.all(np.isfinite(result.sample_weights))
