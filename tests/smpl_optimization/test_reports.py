from lib.smpl_optimization.reports import build_validation_summary, to_jsonable


def test_to_jsonable_converts_numpy_scalars():
    import numpy as np

    data = {"x": np.float32(1.5), "items": [np.int64(2)]}
    assert to_jsonable(data) == {"x": 1.5, "items": [2]}


def test_to_jsonable_converts_numpy_scalar_dict_keys_to_json_strings():
    import json
    import numpy as np

    result = to_jsonable({np.int64(1): np.float32(2)})

    assert result == {"1": 2.0}
    json.dumps(result)


def test_validation_summary_marks_success_when_all_thresholds_pass():
    summary = build_validation_summary(
        beta_variation_after_max_abs=0.0,
        frame_count_unchanged=True,
        before={"penetration": {"max_penetration": 0.2}, "sliding": {"mean_contact_speed": 0.3}, "root": {"rms_vertical_accel": 0.2}},
        after={"penetration": {"max_penetration": 0.05}, "sliding": {"mean_contact_speed": 0.1}, "root": {"rms_vertical_accel": 0.1}},
        pose_delta={"upper_body_max_abs": 0.01, "lower_body_max_abs": 0.3},
        opensim={"ik_rms_not_worse": True, "ground_clearance_not_worse": True},
    )
    assert summary["success"] is True
    assert summary["checks"]["smpl_foot_penetration_reduced"] is True


def test_validation_summary_allows_zero_noop_metrics():
    summary = build_validation_summary(
        beta_variation_after_max_abs=0.0,
        frame_count_unchanged=True,
        before={"penetration": {"max_penetration": 0.0}, "sliding": {"mean_contact_speed": 0.0}, "root": {"rms_vertical_accel": 0.0}},
        after={"penetration": {"max_penetration": 0.0}, "sliding": {"mean_contact_speed": 0.0}, "root": {"rms_vertical_accel": 0.0}},
        pose_delta={"upper_body_max_abs": 0.0, "lower_body_max_abs": 0.0},
        opensim={"ik_rms_not_worse": True, "ground_clearance_not_worse": True},
    )

    assert summary["success"] is True
    assert summary["checks"]["smpl_foot_penetration_reduced"] is True
    assert summary["checks"]["smpl_contact_foot_sliding_reduced"] is True
    assert summary["checks"]["root_vertical_jitter_not_worse"] is True


def test_validation_summary_marks_degraded_when_beta_changes():
    summary = build_validation_summary(
        beta_variation_after_max_abs=1e-3,
        frame_count_unchanged=True,
        before={"penetration": {"max_penetration": 0.2}, "sliding": {"mean_contact_speed": 0.3}, "root": {"rms_vertical_accel": 0.2}},
        after={"penetration": {"max_penetration": 0.05}, "sliding": {"mean_contact_speed": 0.1}, "root": {"rms_vertical_accel": 0.1}},
        pose_delta={"upper_body_max_abs": 0.01, "lower_body_max_abs": 0.3},
        opensim={"ik_rms_not_worse": True, "ground_clearance_not_worse": True},
    )
    assert summary["success"] is False
    assert summary["checks"]["beta_variation_after_max_abs"] is False


def test_validation_summary_keeps_not_run_opensim_out_of_primary_checks():
    summary = build_validation_summary(
        beta_variation_after_max_abs=0.0,
        frame_count_unchanged=True,
        before={"penetration": {"max_penetration": 0.0}, "sliding": {"mean_contact_speed": 0.0}, "root": {"rms_vertical_accel": 0.0}},
        after={"penetration": {"max_penetration": 0.0}, "sliding": {"mean_contact_speed": 0.0}, "root": {"rms_vertical_accel": 0.0}},
        pose_delta={"upper_body_max_abs": 0.0, "lower_body_max_abs": 0.0},
        opensim={
            "status": "not_run",
            "used_for_success": False,
            "ik_rms_not_worse": True,
            "ground_clearance_not_worse": True,
        },
    )

    assert summary["success"] is True
    assert summary["opensim"]["status"] == "not_run"
    assert summary["opensim"]["used_for_success"] is False
    assert "opensim_ik_rms_not_worse" not in summary["checks"]
    assert summary["opensim_checks"]["opensim_ik_rms_not_worse"] is True


def test_validation_summary_includes_opensim_checks_when_used_for_success():
    summary = build_validation_summary(
        beta_variation_after_max_abs=0.0,
        frame_count_unchanged=True,
        before={"penetration": {"max_penetration": 0.0}, "sliding": {"mean_contact_speed": 0.0}, "root": {"rms_vertical_accel": 0.0}},
        after={"penetration": {"max_penetration": 0.0}, "sliding": {"mean_contact_speed": 0.0}, "root": {"rms_vertical_accel": 0.0}},
        pose_delta={"upper_body_max_abs": 0.0, "lower_body_max_abs": 0.0},
        opensim={
            "status": "run",
            "used_for_success": True,
            "ik_rms_not_worse": False,
            "ground_clearance_not_worse": True,
        },
    )

    assert summary["success"] is False
    assert summary["checks"]["opensim_ik_rms_not_worse"] is False
    assert summary["checks"]["opensim_ground_clearance_not_worse"] is True
    assert summary["opensim_checks"]["opensim_ik_rms_not_worse"] is False
