import numpy as np

from lib.smpl_optimization.lower_body import LowerBodyOptimizerConfig, optimize_record


def make_record():
    betas = np.tile(np.arange(10, dtype=np.float32), (4, 1))
    trans = np.zeros((4, 3), dtype=np.float32)
    pose = np.zeros((4, 72), dtype=np.float32)
    feet = np.array(
        [
            [[0.0, -0.10, 0.0], [0.3, 0.05, 0.0]],
            [[0.2, -0.05, 0.0], [0.3, 0.05, 0.0]],
            [[0.4, 0.02, 0.0], [0.3, 0.05, 0.0]],
            [[0.6, 0.04, 0.0], [0.3, 0.05, 0.0]],
        ],
        dtype=np.float32,
    )
    contact = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    return {"betas": betas, "trans_world": trans, "pose": pose, "feet_refined": feet, "contact": contact}


def test_optimize_record_keeps_beta_and_frame_count():
    record = make_record()
    out, reports = optimize_record(record, LowerBodyOptimizerConfig(fps=30.0))
    assert out["betas"].shape == record["betas"].shape
    assert np.max(np.abs(out["betas"] - record["betas"])) == 0.0
    assert len(out["trans_world"]) == len(record["trans_world"])
    assert reports["validation_summary"]["checks"]["frame_count_unchanged"] is True


def test_optimize_record_reduces_penetration_in_foot_arrays():
    record = make_record()
    out, reports = optimize_record(record, LowerBodyOptimizerConfig(fps=30.0))
    assert np.min(out["feet_refined"][:, :, 1]) >= -1e-6
    assert reports["validation_summary"]["checks"]["smpl_foot_penetration_reduced"] is True


def test_optimize_record_does_not_change_upper_body_pose():
    record = make_record()
    out, reports = optimize_record(record, LowerBodyOptimizerConfig(fps=30.0))
    assert np.max(np.abs(out["pose"][:, :3] - record["pose"][:, :3])) == 0.0
    assert reports["pose_delta_report"]["upper_body_max_abs"] == 0.0


def test_optimize_record_does_not_mutate_input_record():
    record = make_record()
    original_feet = record["feet_refined"].copy()
    original_trans = record["trans_world"].copy()

    optimize_record(record, LowerBodyOptimizerConfig(fps=30.0))

    np.testing.assert_array_equal(record["feet_refined"], original_feet)
    np.testing.assert_array_equal(record["trans_world"], original_trans)


def test_optimize_record_without_feet_returns_sensible_reports():
    record = {
        "betas": np.zeros((2, 10), dtype=np.float32),
        "trans_world": np.zeros((2, 3), dtype=np.float32),
        "pose": np.zeros((2, 72), dtype=np.float32),
    }

    out, reports = optimize_record(record, LowerBodyOptimizerConfig(fps=30.0))

    assert out["trans_world"].shape == record["trans_world"].shape
    assert reports["lower_body_optimization_report"]["foot_point_source"] is None
    assert reports["lower_body_optimization_report"]["max_applied_root_y_shift"] == 0.0
    assert reports["validation_summary"]["checks"]["frame_count_unchanged"] is True


def test_optimize_record_pads_short_contact_arrays_for_quality_metrics():
    record = make_record()
    record["contact"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    out, reports = optimize_record(record, LowerBodyOptimizerConfig(fps=30.0))

    assert out["feet_refined"].shape[0] == 4
    assert reports["ground_contact_report"]["before"]["contact_available"] is True
    assert "mean_contact_speed" in reports["ground_contact_report"]["after"]["sliding"]


def test_optimize_record_validation_summary_succeeds_for_simple_record():
    _, reports = optimize_record(make_record(), LowerBodyOptimizerConfig(fps=30.0))

    assert reports["validation_summary"]["success"] is True
    assert reports["validation_summary"]["checks"]["root_vertical_jitter_not_worse"] is True


def test_optimize_record_without_trans_world_does_not_shift_cached_feet():
    record = make_record()
    original_feet = record["feet_refined"].copy()
    del record["trans_world"]

    out, reports = optimize_record(record, LowerBodyOptimizerConfig(fps=30.0))
    lower_report = reports["lower_body_optimization_report"]

    np.testing.assert_array_equal(out["feet_refined"], original_feet)
    assert lower_report["max_applied_root_y_shift"] == 0.0
    assert lower_report["mean_applied_root_y_shift"] == 0.0
    assert lower_report["root_shift_applied"] is False
    assert lower_report["root_shift_reason"] == "missing_trans_world"


def test_optimize_record_caps_root_y_shift():
    record = make_record()

    out, reports = optimize_record(record, LowerBodyOptimizerConfig(fps=30.0, max_root_y_shift=0.05))

    assert np.max(out["trans_world"][:, 1] - record["trans_world"][:, 1]) <= 0.05 + 1e-6
    assert np.min(out["feet_refined"][:, :, 1]) < 0.0
    assert reports["lower_body_optimization_report"]["max_applied_root_y_shift"] == 0.05


def test_optimize_record_marks_opensim_validation_not_run_by_default():
    _, reports = optimize_record(make_record(), LowerBodyOptimizerConfig(fps=30.0))

    opensim = reports["validation_summary"]["opensim"]
    assert opensim["status"] == "not_run"
    assert opensim["used_for_success"] is False
