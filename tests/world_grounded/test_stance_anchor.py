import numpy as np

from lib.world_grounded.contact_segments import StanceSegment
from lib.world_grounded.stance_anchor import compute_stance_anchors, rasterize_anchors


def test_compute_stance_anchors_uses_weighted_median_and_ground_y():
    points = np.zeros((5, 1, 3), dtype=np.float32)
    points[:, 0, 0] = [0.0, 1.0, 2.0, 100.0, 3.0]
    points[:, 0, 1] = 9.0
    points[:, 0, 2] = [0.0, 1.0, 2.0, 100.0, 3.0]
    confidence = np.array([[1.0], [1.0], [1.0], [0.01], [1.0]], dtype=np.float32)
    segment = StanceSegment(0, "left_foot", 0, 5, 0.802, 0.01, 5)

    anchors = compute_stance_anchors(points, confidence, [segment], ground_y=0.25)

    assert len(anchors) == 1
    assert np.allclose(anchors[0].anchor_xyz, [2.0, 0.25, 2.0])
    assert anchors[0].source == "weighted_median"


def test_compute_stance_anchors_selects_weighted_median_at_cumulative_crossing():
    points = np.zeros((2, 1, 3), dtype=np.float32)
    points[:, 0, 0] = [0.0, 10.0]
    points[:, 0, 2] = [0.0, 10.0]
    confidence = np.array([[0.49], [0.51]], dtype=np.float32)
    segment = StanceSegment(0, "left_foot", 0, 2, 0.5, 0.49, 2)

    anchors = compute_stance_anchors(points, confidence, [segment], ground_y=0.0)

    assert np.allclose(anchors[0].anchor_xyz, [10.0, 0.0, 10.0])


def test_rasterize_anchors_keeps_one_global_target_across_frames():
    anchor = compute_stance_anchors(
        np.array([[[1.0, 0.0, 2.0]], [[3.0, 0.0, 4.0]]], dtype=np.float32),
        np.ones((2, 1), dtype=np.float32),
        [StanceSegment(0, "left_foot", 0, 2, 1.0, 1.0, 2)],
        ground_y=0.0,
    )[0]

    targets, mask = rasterize_anchors([anchor], n_frames=4, n_feet=1)

    assert mask[:, 0].tolist() == [True, True, False, False]
    assert np.allclose(targets[0, 0], targets[1, 0])
