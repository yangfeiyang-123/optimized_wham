"""Selection checks for the Stage6 OpenSim feedback iteration."""


def _get(summary, *path):
    value = summary
    for key in path:
        value = value[key]
    return value


def _reduced(candidate_value, baseline_value):
    return candidate_value < baseline_value


def select_stage6_result(
    baseline,
    candidate,
    *,
    root_y_budget,
    upper_body_delta_limit=0.05,
    lower_body_delta_limit=1.2,
):
    """Select iter_01 only when Stage6 hard checks pass and a target improves."""

    baseline_mean_rms = _get(baseline, "opensim", "mean_rms")
    candidate_mean_rms = _get(candidate, "opensim", "mean_rms")
    baseline_max_rms = _get(baseline, "opensim", "max_rms")
    candidate_max_rms = _get(candidate, "opensim", "max_rms")
    baseline_penetration = _get(
        baseline, "smpl", "penetration", "max_penetration"
    )
    candidate_penetration = _get(
        candidate, "smpl", "penetration", "max_penetration"
    )
    baseline_contact_speed = _get(
        baseline, "smpl", "sliding", "mean_contact_speed"
    )
    candidate_contact_speed = _get(
        candidate, "smpl", "sliding", "mean_contact_speed"
    )
    baseline_root_jitter = _get(baseline, "smpl", "root", "rms_vertical_accel")
    candidate_root_jitter = _get(candidate, "smpl", "root", "rms_vertical_accel")

    target_improvements = {
        "lower_body_mean_rms_reduced": _reduced(
            _get(candidate, "opensim", "lower_body_mean_rms"),
            _get(baseline, "opensim", "lower_body_mean_rms"),
        ),
        "max_penetration_reduced": _reduced(
            candidate_penetration,
            baseline_penetration,
        ),
        "mean_contact_speed_reduced": _reduced(
            candidate_contact_speed,
            baseline_contact_speed,
        ),
        "root_rms_vertical_accel_reduced": _reduced(
            candidate_root_jitter,
            baseline_root_jitter,
        ),
    }

    checks = {
        "beta_variation_after_max_abs_zero": _get(
            candidate, "smpl", "beta_variation_after_max_abs"
        )
        == 0,
        "frame_count_unchanged": _get(candidate, "smpl", "frames")
        == _get(baseline, "smpl", "frames"),
        "opensim_mean_rms_not_worse": candidate_mean_rms <= baseline_mean_rms,
        "opensim_max_rms_not_worse": candidate_max_rms <= baseline_max_rms,
        "smpl_max_foot_penetration_not_worse": (
            candidate_penetration <= baseline_penetration
        ),
        "smpl_contact_foot_sliding_not_worse": (
            candidate_contact_speed <= baseline_contact_speed
        ),
        "root_vertical_jitter_not_worse": candidate_root_jitter <= baseline_root_jitter,
        "upper_body_pose_delta_small": _get(
            candidate, "smpl", "pose_delta", "upper_body_max_abs"
        )
        <= upper_body_delta_limit,
        "lower_body_pose_delta_bounded": _get(
            candidate, "smpl", "pose_delta", "lower_body_max_abs"
        )
        <= lower_body_delta_limit,
        "root_y_total_shift_within_budget": _get(
            candidate, "smpl", "total_root_y_shift_max_abs"
        )
        <= root_y_budget,
        "at_least_one_target_metric_improved": any(target_improvements.values()),
    }
    accepted = all(checks.values())

    return {
        "selected": "iter_01" if accepted else "iter_00",
        "accepted": accepted,
        "checks": checks,
        "target_improvements": target_improvements,
        "baseline": baseline,
        "candidate": candidate,
    }
