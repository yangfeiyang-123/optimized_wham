import numpy as np

from lib.world_grounded.body_graph import BODY_GRAPH, graph_labels
from lib.world_grounded.body_keypoints import (
    H36M_JOINT_NAMES,
    assemble_body_graph_keypoints,
    body_keypoints_from_verts,
)


def _distinct_h36m_feet(n_frames=2):
    """Build h36m/feet arrays where every joint has a unique, identifiable coordinate."""
    h36m = np.zeros((n_frames, len(H36M_JOINT_NAMES), 3), dtype=np.float32)
    for j in range(len(H36M_JOINT_NAMES)):
        h36m[:, j] = [j + 1, 0.0, 0.0]
    feet = np.zeros((n_frames, 4, 3), dtype=np.float32)
    for f in range(4):
        feet[:, f] = [100 + f, 0.0, 0.0]
    return h36m, feet


def test_assemble_returns_21_keypoints_in_graph_order():
    h36m, feet = _distinct_h36m_feet()
    out = assemble_body_graph_keypoints(h36m, feet)
    assert out.shape == (2, 21, 3)
    assert len(graph_labels()) == 21


def test_assemble_maps_each_keypoint_to_expected_source():
    h36m, feet = _distinct_h36m_feet()
    out = assemble_body_graph_keypoints(h36m, feet)
    labels = graph_labels()
    idx = {name: i for i, name in enumerate(labels)}
    h36m_id = {name: i for i, name in enumerate(H36M_JOINT_NAMES)}

    # Torso / limb joints come straight from H36M (encoded as j+1 on the x axis).
    assert out[0, idx["pelvis"], 0] == h36m_id["hip"] + 1
    assert out[0, idx["neck"], 0] == h36m_id["neck"] + 1
    assert out[0, idx["left_wrist"], 0] == h36m_id["lwrist"] + 1
    assert out[0, idx["right_ankle"], 0] == h36m_id["rankle"] + 1

    # Feet come from the feet regressor (encoded as 100+f); heel/toe order matters.
    assert out[0, idx["left_heel"], 0] == 100 + 0
    assert out[0, idx["left_toe"], 0] == 100 + 1
    assert out[0, idx["right_heel"], 0] == 100 + 2
    assert out[0, idx["right_toe"], 0] == 100 + 3

    # Thorax is the midpoint between the two shoulders.
    expected = 0.5 * ((h36m_id["lshoulder"] + 1) + (h36m_id["rshoulder"] + 1))
    assert out[0, idx["thorax"], 0] == expected


def test_body_keypoints_from_verts_matches_manual_regression():
    rng = np.random.default_rng(0)
    verts = rng.standard_normal((3, 12, 3)).astype(np.float32)
    j_h36m = rng.standard_normal((17, 12)).astype(np.float32)
    j_feet = rng.standard_normal((4, 12)).astype(np.float32)

    out = body_keypoints_from_verts(verts, j_h36m, j_feet)
    h36m = np.einsum("kv,tvc->tkc", j_h36m, verts)
    feet = np.einsum("kv,tvc->tkc", j_feet, verts)
    expected = assemble_body_graph_keypoints(h36m, feet)

    assert out.shape == (3, 21, 3)
    assert np.allclose(out, expected)


def test_graph_labels_and_keypoint_sources_are_consistent():
    # Every BODY_GRAPH keypoint must have a defined source in the assembly table.
    from lib.world_grounded.body_keypoints import _KEYPOINT_SOURCES

    assert set(BODY_GRAPH["keypoints"]) == set(_KEYPOINT_SOURCES)
