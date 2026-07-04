import numpy as np

from lib.world_grounded.contact_segments import extract_stance_segments, kinematic_stance_gate


def test_extract_stance_segments_uses_hysteresis_for_one_segment():
    confidence = np.array([[0.0], [0.6], [0.5], [0.4], [0.31], [0.2]], dtype=np.float32)

    segments, stance_mask = extract_stance_segments(
        confidence,
        ["left_foot"],
        enter_threshold=0.55,
        exit_threshold=0.30,
        min_segment_len=2,
        merge_gap=0,
    )

    assert len(segments) == 1
    assert segments[0].start == 1
    assert segments[0].end == 5
    assert stance_mask[:, 0].tolist() == [False, True, True, True, True, False]


def test_extract_stance_segments_merges_short_gaps_and_drops_short_segments():
    confidence = np.array(
        [[0.0], [0.7], [0.7], [0.1], [0.7], [0.7], [0.0], [0.8], [0.0]],
        dtype=np.float32,
    )

    segments, stance_mask = extract_stance_segments(
        confidence,
        ["right_foot"],
        min_segment_len=3,
        merge_gap=1,
    )

    assert [(s.start, s.end, s.length) for s in segments] == [(1, 6, 5)]
    assert stance_mask[:, 0].tolist() == [False, True, True, True, True, True, False, False, False]


def test_extract_stance_segments_returns_empty_for_no_contact():
    segments, stance_mask = extract_stance_segments(np.zeros((4, 2)), ["left", "right"])

    assert segments == []
    assert stance_mask.shape == (4, 2)
    assert not stance_mask.any()


def test_kinematic_gate_excludes_fast_and_airborne_frames():
    # One foot, always "in contact" per confidence, but physically moving mid-clip.
    confidence = np.full((10, 1), 0.9, dtype=np.float32)
    points = np.zeros((10, 1, 3), dtype=np.float32)  # Y-up: axis 1 is vertical
    points[:, 0, 0] = np.array([0, 0, 0, 0, 0.3, 0.6, 0.9, 0.9, 0.9, 0.9])  # slides in x, frames ~3-6
    gate = kinematic_stance_gate(points, fps=30.0, height_threshold=0.05, speed_threshold=0.15)

    _, mask = extract_stance_segments(
        confidence, ["left"], min_segment_len=1, merge_gap=0, kinematic_gate=gate
    )
    assert not mask[4:6, 0].any()  # fast-moving frames are not stance
    assert mask[0, 0] and mask[-1, 0]  # stationary frames remain stance


def test_kinematic_gate_excludes_lifted_foot():
    confidence = np.full((6, 1), 0.9, dtype=np.float32)
    points = np.zeros((6, 1, 3), dtype=np.float32)
    points[:, 0, 1] = np.array([0, 0, 0.2, 0.2, 0, 0])  # foot lifts on frames 2-3
    gate = kinematic_stance_gate(points, fps=30.0, height_threshold=0.05, speed_threshold=1e9)

    _, mask = extract_stance_segments(
        confidence, ["left"], min_segment_len=1, merge_gap=0, kinematic_gate=gate
    )
    assert not mask[2:4, 0].any()  # airborne frames are not stance


def test_kinematic_gate_is_opt_in_and_backward_compatible():
    # Without a gate, confidence hysteresis alone decides stance (unchanged behaviour).
    confidence = np.array([[0.0], [0.6], [0.5], [0.4], [0.31], [0.2]], dtype=np.float32)
    _, mask = extract_stance_segments(
        confidence, ["left"], enter_threshold=0.55, exit_threshold=0.30, min_segment_len=2, merge_gap=0
    )
    assert mask[:, 0].tolist() == [False, True, True, True, True, False]

