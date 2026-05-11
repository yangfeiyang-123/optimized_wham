import numpy as np

from scripts.world_grounded_smpl_optimizer import optimize_record


def test_optimize_record_aligns_estimated_ground_to_zero():
    n_frames = 8
    ground_y = -0.87
    record = {
        "frame_ids": np.arange(n_frames, dtype=np.int64),
        "pose": np.zeros((n_frames, 72), dtype=np.float32),
        "betas": np.zeros((n_frames, 10), dtype=np.float32),
        "trans_world": np.zeros((n_frames, 3), dtype=np.float32),
        "feet_refined": np.zeros((n_frames, 4, 3), dtype=np.float32),
        "feet_world": np.zeros((n_frames, 4, 3), dtype=np.float32),
        "verts": np.zeros((n_frames, 5, 3), dtype=np.float32),
        "contact": np.ones((n_frames, 4), dtype=np.float32),
    }
    record["feet_refined"][..., 1] = ground_y
    record["feet_world"][..., 1] = ground_y
    record["verts"][..., 1] = ground_y

    optimized, ground_report, _, quality_report = optimize_record(record, fps=30.0)

    assert np.allclose(optimized["feet_refined"][..., 1], 0.0, atol=1e-6)
    assert np.allclose(optimized["feet_world"][..., 1], 0.0, atol=1e-6)
    assert np.allclose(optimized["verts"][..., 1], 0.0, atol=1e-6)
    assert np.allclose(optimized["trans_world"][:, 1], -ground_y, atol=1e-6)
    assert ground_report["ground_alignment_offset_y"] == -ground_report["ground_y"]
    assert quality_report["ground_y_after_alignment"] == 0.0
