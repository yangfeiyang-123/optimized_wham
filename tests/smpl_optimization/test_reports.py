from lib.smpl_optimization.reports import build_validation_summary, to_jsonable


def test_to_jsonable_converts_numpy_scalars():
    import numpy as np

    data = {"x": np.float32(1.5), "items": [np.int64(2)]}
    assert to_jsonable(data) == {"x": 1.5, "items": [2]}


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
