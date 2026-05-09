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
