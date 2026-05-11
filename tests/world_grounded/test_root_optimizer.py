import numpy as np
import pytest

from lib.world_grounded.foot_points import get_record_contact, get_record_foot_points
from lib.world_grounded.root_optimizer import optimize_root_translation


def test_optimize_root_translation_reduces_penetration_and_preserves_shape():
    trans = np.zeros((12, 3), dtype=np.float64)
    foot_points = np.zeros((12, 4, 3), dtype=np.float64)
    foot_points[..., 1] = -0.10
    contact_conf = np.ones((12, 4), dtype=np.float64)

    result = optimize_root_translation(
        trans_world=trans,
        foot_points=foot_points,
        contact_confidence=contact_conf,
        ground_y=0.0,
        fps=30.0,
    )

    corrected_feet = foot_points + result.delta[:, None, :]
    assert result.optimized_trans_world.shape == (12, 3)
    assert result.delta.shape == (12, 3)
    assert float(np.min(corrected_feet[..., 1])) > -1e-5
    assert result.report["foot_penetration_max_cm_after"] <= result.report["foot_penetration_max_cm_before"]


def test_optimize_root_translation_smooths_spike():
    trans = np.zeros((15, 3), dtype=np.float64)
    trans[7, 1] = 1.0
    foot_points = np.zeros((15, 4, 3), dtype=np.float64)
    contact_conf = np.zeros((15, 4), dtype=np.float64)

    result = optimize_root_translation(trans, foot_points, contact_conf, ground_y=0.0, fps=30.0)

    assert result.optimized_trans_world[7, 1] < 0.75


def test_optimize_root_translation_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same frame count"):
        optimize_root_translation(
            np.zeros((3, 3)),
            np.zeros((2, 4, 3)),
            np.zeros((2, 4)),
            ground_y=0.0,
            fps=30.0,
        )


def test_get_record_foot_points_prefers_refined_feet():
    record = {
        "feet_world": np.zeros((3, 4, 3), dtype=np.float32),
        "feet_refined": np.ones((3, 4, 3), dtype=np.float32),
    }

    result = get_record_foot_points(record)

    assert result.source == "feet_refined"
    assert result.points.shape == (3, 4, 3)
    assert np.allclose(result.points, 1.0)


def test_get_record_contact_pads_and_clips():
    record = {"contact": np.array([[-1.0, 0.5], [2.0, np.nan]], dtype=np.float32)}

    contact = get_record_contact(record, n_frames=2, n_points=4)

    assert contact.shape == (2, 4)
    assert np.allclose(contact[:, 2:], 0.0)
    assert float(contact.min()) >= 0.0
    assert float(contact.max()) <= 1.0


def test_get_record_foot_points_uses_vertices_fallback():
    verts = np.array(
        [
            [
                [-1.0, 0.0, -1.0],
                [-1.0, 0.0, 1.0],
                [1.0, 0.0, -1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )

    result = get_record_foot_points({"verts": verts})

    assert result.source == "verts_low_fallback"
    assert result.points.shape == (1, 4, 3)
