import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.stage7_selection import select_stage7_result  # noqa: E402


def _summary(
    *,
    mean_rms=0.05,
    max_rms=0.12,
    range_violations=0,
    coordinate_jumps=3,
    coordinate_jerk=0.4,
    penetration=0.002,
    sliding_mean=0.1,
    sliding_max=0.3,
    whole_body_jerk=0.8,
    root_jerk=0.2,
    beta_var=0.0,
    frames=100,
    lower_delta=0.02,
    whole_body_delta=0.08,
    root_translation_delta=0.01,
    root_vertical_delta=0.005,
):
    return {
        "opensim": {
            "mean_rms": mean_rms,
            "max_rms": max_rms,
            "motion": {
                "num_range_violations": range_violations,
                "num_coordinate_jumps": coordinate_jumps,
                "coordinate_jerk": {"rms": coordinate_jerk},
            },
        },
        "smpl": {
            "beta_variation_after_max_abs": beta_var,
            "frames": frames,
            "penetration": {"max_penetration": penetration},
            "sliding": {
                "mean_contact_speed": sliding_mean,
                "max_contact_speed": sliding_max,
            },
            "pose_delta": {
                "lower_body_max_abs": lower_delta,
                "whole_body_max_abs": whole_body_delta,
            },
            "root_delta": {
                "max_abs": root_translation_delta,
                "vertical_max_abs": root_vertical_delta,
            },
            "pose_smoothness": {"jerk": {"rms": whole_body_jerk}},
            "root_smoothness": {"jerk": {"rms": root_jerk}},
        },
    }


def test_accepts_candidate_when_smoother_and_gates_pass():
    baseline = _summary(whole_body_jerk=0.8, root_jerk=0.2)
    candidate = _summary(whole_body_jerk=0.6, root_jerk=0.15)

    result = select_stage7_result(baseline, candidate)

    assert result["selected"] == "candidate"
    assert result["accepted"]
    assert result["checks"]["at_least_one_smoothness_metric_improved"]
    assert result["target_improvements"]["whole_body_pose_jerk_improved"]
    assert result["target_improvements"]["root_translation_jerk_improved"]


def test_rejects_candidate_when_penetration_worse():
    baseline = _summary(penetration=0.002)
    candidate = _summary(penetration=0.006, whole_body_jerk=0.6)

    result = select_stage7_result(baseline, candidate)

    assert result["selected"] == "baseline"
    assert not result["accepted"]
    assert not result["checks"]["smpl_max_foot_penetration_not_worse"]


def test_rejects_candidate_without_smoothness_improvement():
    baseline = _summary()
    candidate = _summary()

    result = select_stage7_result(baseline, candidate)

    assert result["selected"] == "baseline"
    assert not result["accepted"]
    assert not result["checks"]["at_least_one_smoothness_metric_improved"]
    assert not any(result["target_improvements"].values())


def test_missing_required_numbers_fail_closed():
    baseline = _summary()
    candidate = _summary(whole_body_jerk=0.6)
    del candidate["opensim"]["mean_rms"]
    del candidate["smpl"]["pose_delta"]

    result = select_stage7_result(baseline, candidate)

    assert result["selected"] == "baseline"
    assert not result["accepted"]
    assert not result["checks"]["opensim_mean_rms_not_worse"]
    assert not result["checks"]["lower_body_pose_delta_small"]
    assert not result["checks"]["whole_body_pose_delta_small"]


def test_missing_plan_shaped_smoothness_value_fails_closed():
    baseline = _summary()
    candidate = _summary()
    del candidate["smpl"]["pose_smoothness"]

    result = select_stage7_result(baseline, candidate)

    assert result["selected"] == "baseline"
    assert not result["accepted"]
    assert not result["target_improvements"]["whole_body_pose_jerk_improved"]
    assert not result["checks"]["at_least_one_smoothness_metric_improved"]


def test_tolerance_boundaries_are_inclusive():
    baseline = _summary(
        mean_rms=0.05,
        max_rms=0.12,
        penetration=0.002,
        sliding_mean=0.1,
        sliding_max=0.3,
        whole_body_jerk=0.8,
    )
    candidate = _summary(
        mean_rms=0.051,
        max_rms=0.126,
        penetration=0.004,
        sliding_mean=0.102,
        sliding_max=0.315,
        whole_body_jerk=0.6,
        lower_delta=0.05,
        whole_body_delta=0.15,
        root_translation_delta=0.03,
        root_vertical_delta=0.015,
    )

    result = select_stage7_result(baseline, candidate)

    assert result["accepted"]
    assert all(result["checks"].values())


def test_accepts_plan_shaped_summary_when_pose_jerk_improves():
    baseline = {
        "opensim": {
            "mean_rms": 0.05,
            "max_rms": 0.12,
            "motion": {
                "num_range_violations": 0,
                "num_coordinate_jumps": 3,
                "coordinate_jerk": {"rms": 0.4},
            },
        },
        "smpl": {
            "frames": 100,
            "beta_variation_after_max_abs": 0.0,
            "penetration": {"max_penetration": 0.002},
            "sliding": {
                "mean_contact_speed": 0.1,
                "max_contact_speed": 0.3,
            },
            "pose_delta": {
                "lower_body_max_abs": 0.02,
                "whole_body_max_abs": 0.08,
            },
            "root_delta": {
                "max_abs": 0.01,
                "vertical_max_abs": 0.005,
            },
            "pose_smoothness": {"jerk": {"rms": 0.8}},
            "root_smoothness": {"jerk": {"rms": 0.2}},
        },
    }
    candidate = {
        "opensim": {
            "mean_rms": 0.05,
            "max_rms": 0.12,
            "motion": {
                "num_range_violations": 0,
                "num_coordinate_jumps": 3,
                "coordinate_jerk": {"rms": 0.4},
            },
        },
        "smpl": {
            "frames": 100,
            "beta_variation_after_max_abs": 0.0,
            "penetration": {"max_penetration": 0.002},
            "sliding": {
                "mean_contact_speed": 0.1,
                "max_contact_speed": 0.3,
            },
            "pose_delta": {
                "lower_body_max_abs": 0.02,
                "whole_body_max_abs": 0.08,
            },
            "root_delta": {
                "max_abs": 0.01,
                "vertical_max_abs": 0.005,
            },
            "pose_smoothness": {"jerk": {"rms": 0.6}},
            "root_smoothness": {"jerk": {"rms": 0.2}},
        },
    }

    result = select_stage7_result(baseline, candidate)

    assert result["selected"] == "candidate"
    assert result["accepted"]
    assert result["target_improvements"]["whole_body_pose_jerk_improved"]


def test_zero_baseline_ratio_gate_allows_tiny_absolute_regression():
    baseline = _summary(sliding_mean=0.0, whole_body_jerk=0.8)
    candidate = _summary(sliding_mean=5e-9, whole_body_jerk=0.6)

    result = select_stage7_result(baseline, candidate)

    assert result["selected"] == "candidate"
    assert result["accepted"]
    assert result["checks"]["smpl_contact_sliding_mean_not_worse"]


def test_zero_baseline_ratio_gate_rejects_large_absolute_regression():
    baseline = _summary(sliding_mean=0.0, whole_body_jerk=0.8)
    candidate = _summary(sliding_mean=1e-4, whole_body_jerk=0.6)

    result = select_stage7_result(baseline, candidate)

    assert result["selected"] == "baseline"
    assert not result["accepted"]
    assert not result["checks"]["smpl_contact_sliding_mean_not_worse"]
