import numpy as np

from lib.world_grounded.contact_segments import extract_stance_segments


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
