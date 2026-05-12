"""Selection checks for the Stage6 OpenSim feedback iteration."""


def _number(data, *keys, default=None):
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return value


def _not_worse(candidate_value, baseline_value):
    if candidate_value is None or baseline_value is None:
        return False
    return candidate_value <= baseline_value


def _improved(candidate_value, baseline_value):
    if candidate_value is None or baseline_value is None:
        return False
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

    baseline_mean_rms = _number(baseline, "opensim", "mean_rms")
    candidate_mean_rms = _number(candidate, "opensim", "mean_rms")
    baseline_max_rms = _number(baseline, "opensim", "max_rms")
    candidate_max_rms = _number(candidate, "opensim", "max_rms")
    baseline_lower_body_mean_rms = _number(
        baseline, "opensim", "lower_body_mean_rms"
    )
    candidate_lower_body_mean_rms = _number(
        candidate, "opensim", "lower_body_mean_rms"
    )
    baseline_penetration = _number(
        baseline, "smpl", "penetration", "max_penetration"
    )
    candidate_penetration = _number(
        candidate, "smpl", "penetration", "max_penetration"
    )
    baseline_contact_speed = _number(
        baseline, "smpl", "sliding", "mean_contact_speed"
    )
    candidate_contact_speed = _number(
        candidate, "smpl", "sliding", "mean_contact_speed"
    )
    baseline_root_jitter = _number(baseline, "smpl", "root", "rms_vertical_accel")
    candidate_root_jitter = _number(candidate, "smpl", "root", "rms_vertical_accel")
    baseline_frames = _number(baseline, "smpl", "frames")
    candidate_frames = _number(candidate, "smpl", "frames")
    candidate_beta_variation = _number(
        candidate, "smpl", "beta_variation_after_max_abs"
    )
    candidate_upper_body_delta = _number(
        candidate, "smpl", "pose_delta", "upper_body_max_abs"
    )
    candidate_lower_body_delta = _number(
        candidate, "smpl", "pose_delta", "lower_body_max_abs"
    )
    candidate_root_y_shift = _number(
        candidate, "smpl", "total_root_y_shift_max_abs"
    )

    target_improvements = {
        "opensim_lower_body_marker_rms_reduced": _improved(
            candidate_lower_body_mean_rms,
            baseline_lower_body_mean_rms,
        ),
        "smpl_foot_penetration_reduced": _improved(
            candidate_penetration,
            baseline_penetration,
        ),
        "smpl_contact_foot_sliding_reduced": _improved(
            candidate_contact_speed,
            baseline_contact_speed,
        ),
        "root_vertical_jitter_reduced": _improved(
            candidate_root_jitter,
            baseline_root_jitter,
        ),
    }

    checks = {
        "beta_variation_after_max_abs_zero": candidate_beta_variation == 0.0,
        "frame_count_unchanged": (
            baseline_frames is not None
            and candidate_frames is not None
            and candidate_frames == baseline_frames
        ),
        "opensim_mean_rms_not_worse": _not_worse(
            candidate_mean_rms, baseline_mean_rms
        ),
        "opensim_max_rms_not_worse": _not_worse(candidate_max_rms, baseline_max_rms),
        "smpl_max_foot_penetration_not_worse": _not_worse(
            candidate_penetration, baseline_penetration
        ),
        "smpl_contact_foot_sliding_not_worse": _not_worse(
            candidate_contact_speed, baseline_contact_speed
        ),
        "root_vertical_jitter_not_worse": _not_worse(
            candidate_root_jitter, baseline_root_jitter
        ),
        "upper_body_pose_delta_small": (
            candidate_upper_body_delta is not None
            and candidate_upper_body_delta <= upper_body_delta_limit
        ),
        "lower_body_pose_delta_bounded": (
            candidate_lower_body_delta is not None
            and candidate_lower_body_delta <= lower_body_delta_limit
        ),
        "root_y_total_shift_within_budget": (
            candidate_root_y_shift is not None
            and candidate_root_y_shift <= root_y_budget
        ),
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
