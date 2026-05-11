import numpy as np

from lib.world_grounded.tracks import merge_tracks, select_track


def test_merge_tracks_preserves_full_frame_range_and_prefers_longer_overlap():
    results = {
        0: {
            "frame_ids": np.arange(0, 4),
            "pose": np.zeros((4, 72), dtype=np.float32),
            "trans_world": np.zeros((4, 3), dtype=np.float32),
        },
        1: {
            "frame_ids": np.arange(3, 7),
            "pose": np.ones((4, 72), dtype=np.float32),
            "trans_world": np.ones((4, 3), dtype=np.float32),
        },
    }

    merged = merge_tracks(results)

    assert merged["frame_ids"].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert merged["pose"].shape == (7, 72)
    assert np.allclose(merged["pose"][3], np.zeros(72))
    assert np.allclose(merged["pose"][4], np.ones(72))


def test_select_track_longest():
    results = {
        "a": {"frame_ids": np.arange(2), "pose": np.zeros((2, 72))},
        "b": {"frame_ids": np.arange(5), "pose": np.zeros((5, 72))},
    }

    track_id, record = select_track(results, "longest")

    assert track_id == "b"
    assert len(record["pose"]) == 5


def test_merge_tracks_omits_missing_per_frame_fields_without_stale_data():
    results = {
        0: {
            "frame_ids": np.arange(0, 2),
            "pose": np.zeros((2, 72), dtype=np.float32),
            "trans_world": np.zeros((2, 3), dtype=np.float32),
            "joints": np.zeros((2, 24, 3), dtype=np.float32),
        },
        1: {
            "frame_ids": np.arange(2, 4),
            "pose": np.ones((2, 72), dtype=np.float32),
            "trans_world": np.ones((2, 3), dtype=np.float32),
        },
    }

    merged = merge_tracks(results)

    assert merged["frame_ids"].tolist() == [0, 1, 2, 3]
    assert "joints" not in merged
    assert "trans_world" in merged


def test_merge_tracks_expands_static_betas_per_selected_frame():
    results = {
        0: {
            "frame_ids": np.arange(0, 2),
            "pose": np.zeros((2, 72), dtype=np.float32),
            "betas": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        },
        1: {
            "frame_ids": np.arange(2, 4),
            "pose": np.ones((2, 72), dtype=np.float32),
            "betas": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        },
    }

    merged = merge_tracks(results)

    assert merged["betas"].shape == (4, 3)
    assert np.allclose(merged["betas"][:2], [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert np.allclose(merged["betas"][2:], [[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])


def test_merge_tracks_ignores_unknown_matching_length_arrays():
    results = {
        0: {
            "frame_ids": np.arange(0, 2),
            "pose": np.zeros((2, 72), dtype=np.float32),
            "scores": np.array([0.1, 0.2], dtype=np.float32),
        },
        1: {
            "frame_ids": np.arange(2, 4),
            "pose": np.ones((2, 72), dtype=np.float32),
            "scores": np.array([0.3, 0.4], dtype=np.float32),
        },
    }

    merged = merge_tracks(results)

    assert "scores" not in merged


def test_merge_tracks_preserves_contact_and_feet_fields():
    results = {
        0: {
            "frame_ids": np.arange(0, 2),
            "pose": np.zeros((2, 72), dtype=np.float32),
            "contact": np.zeros((2, 4), dtype=np.float32),
            "feet_refined": np.zeros((2, 4, 3), dtype=np.float32),
        },
        1: {
            "frame_ids": np.arange(2, 4),
            "pose": np.ones((2, 72), dtype=np.float32),
            "contact": np.ones((2, 4), dtype=np.float32),
            "feet_refined": np.ones((2, 4, 3), dtype=np.float32),
        },
    }

    merged = merge_tracks(results)

    assert merged["contact"].shape == (4, 4)
    assert merged["feet_refined"].shape == (4, 4, 3)
    assert np.allclose(merged["contact"][:2], 0.0)
    assert np.allclose(merged["contact"][2:], 1.0)


def test_select_track_supports_stringified_int_lookup():
    results = {
        1: {"frame_ids": np.arange(2), "pose": np.zeros((2, 72))},
        2: {"frame_ids": np.arange(3), "pose": np.ones((3, 72))},
    }

    track_id, record = select_track(results, "2")

    assert track_id == 2
    assert len(record["pose"]) == 3


def test_select_track_exact_reserved_string_keys_before_sentinels():
    results = {
        "merge": {"frame_ids": np.arange(1), "pose": np.zeros((1, 72))},
        "longest": {"frame_ids": np.arange(2), "pose": np.ones((2, 72))},
        "other": {"frame_ids": np.arange(3), "pose": np.full((3, 72), 2.0)},
    }

    merge_id, merge_record = select_track(results, "merge")
    longest_id, longest_record = select_track(results, "longest")

    assert merge_id == "merge"
    assert len(merge_record["pose"]) == 1
    assert longest_id == "longest"
    assert len(longest_record["pose"]) == 2
