"""Selection checks for the Stage7 whole-body smoothness candidate."""


def _number(data, *keys, default=None):
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return value


def _not_worse_abs(candidate, baseline, tolerance):
    if candidate is None or baseline is None:
        return False
    return candidate <= baseline + tolerance


def _not_worse_ratio(candidate, baseline, ratio):
    if candidate is None or baseline is None:
        return False
    return candidate <= baseline * (1.0 + ratio)


def _improved(candidate, baseline):
    if candidate is None or baseline is None:
        return False
    return candidate < baseline


def select_stage7_result(
    baseline,
    candidate,
    *,
    foot_penetration_tolerance=0.002,
    contact_sliding_mean_ratio=0.02,
    contact_sliding_max_ratio=0.05,
    lower_body_delta_limit=0.05,
    whole_body_delta_limit=0.15,
    root_translation_delta_limit=0.03,
    root_vertical_delta_limit=0.015,
    opensim_mean_rms_ratio=0.02,
    opensim_max_rms_ratio=0.05,
):
    """Select the Stage7 candidate only when safety gates pass and smoothness improves."""

    baseline_frames = _number(baseline, "smpl", "frames")
    candidate_frames = _number(candidate, "smpl", "frames")
    candidate_beta_variation = _number(
        candidate, "smpl", "beta_variation_after_max_abs"
    )

    baseline_penetration = _number(
        baseline, "smpl", "penetration", "max_penetration"
    )
    candidate_penetration = _number(
        candidate, "smpl", "penetration", "max_penetration"
    )
    baseline_contact_mean = _number(
        baseline, "smpl", "sliding", "mean_contact_speed"
    )
    candidate_contact_mean = _number(
        candidate, "smpl", "sliding", "mean_contact_speed"
    )
    baseline_contact_max = _number(
        baseline, "smpl", "sliding", "max_contact_speed"
    )
    candidate_contact_max = _number(
        candidate, "smpl", "sliding", "max_contact_speed"
    )

    candidate_lower_body_delta = _number(
        candidate, "smpl", "pose_delta", "lower_body_max_abs"
    )
    candidate_whole_body_delta = _number(
        candidate, "smpl", "pose_delta", "whole_body_max_abs"
    )
    candidate_root_translation_delta = _number(
        candidate, "smpl", "root_delta", "translation_max_abs"
    )
    candidate_root_vertical_delta = _number(
        candidate, "smpl", "root_delta", "vertical_max_abs"
    )

    baseline_mean_rms = _number(baseline, "opensim", "mean_rms")
    candidate_mean_rms = _number(candidate, "opensim", "mean_rms")
    baseline_max_rms = _number(baseline, "opensim", "max_rms")
    candidate_max_rms = _number(candidate, "opensim", "max_rms")
    baseline_range_violations = _number(
        baseline, "opensim", "coordinate_range_violations"
    )
    candidate_range_violations = _number(
        candidate, "opensim", "coordinate_range_violations"
    )
    baseline_coordinate_jumps = _number(baseline, "opensim", "coordinate_jumps")
    candidate_coordinate_jumps = _number(candidate, "opensim", "coordinate_jumps")

    baseline_whole_body_jerk = _number(
        baseline, "smpl", "smoothness", "whole_body_pose_jerk"
    )
    candidate_whole_body_jerk = _number(
        candidate, "smpl", "smoothness", "whole_body_pose_jerk"
    )
    baseline_root_translation_jerk = _number(
        baseline, "smpl", "smoothness", "root_translation_jerk"
    )
    candidate_root_translation_jerk = _number(
        candidate, "smpl", "smoothness", "root_translation_jerk"
    )
    baseline_coordinate_jerk = _number(baseline, "opensim", "coordinate_jerk")
    candidate_coordinate_jerk = _number(candidate, "opensim", "coordinate_jerk")

    target_improvements = {
        "whole_body_pose_jerk_improved": _improved(
            candidate_whole_body_jerk,
            baseline_whole_body_jerk,
        ),
        "root_translation_jerk_improved": _improved(
            candidate_root_translation_jerk,
            baseline_root_translation_jerk,
        ),
        "opensim_coordinate_jerk_improved": _improved(
            candidate_coordinate_jerk,
            baseline_coordinate_jerk,
        ),
        "opensim_coordinate_jumps_improved": _improved(
            candidate_coordinate_jumps,
            baseline_coordinate_jumps,
        ),
    }

    checks = {
        "frame_count_unchanged": (
            baseline_frames is not None
            and candidate_frames is not None
            and candidate_frames == baseline_frames
        ),
        "beta_variation_after_max_abs_zero": candidate_beta_variation == 0.0,
        "smpl_max_foot_penetration_not_worse": _not_worse_abs(
            candidate_penetration,
            baseline_penetration,
            foot_penetration_tolerance,
        ),
        "smpl_contact_sliding_mean_not_worse": _not_worse_ratio(
            candidate_contact_mean,
            baseline_contact_mean,
            contact_sliding_mean_ratio,
        ),
        "smpl_contact_sliding_max_not_worse": _not_worse_ratio(
            candidate_contact_max,
            baseline_contact_max,
            contact_sliding_max_ratio,
        ),
        "lower_body_pose_delta_small": (
            candidate_lower_body_delta is not None
            and candidate_lower_body_delta <= lower_body_delta_limit
        ),
        "whole_body_pose_delta_small": (
            candidate_whole_body_delta is not None
            and candidate_whole_body_delta <= whole_body_delta_limit
        ),
        "root_translation_delta_small": (
            candidate_root_translation_delta is not None
            and candidate_root_translation_delta <= root_translation_delta_limit
        ),
        "root_vertical_delta_small": (
            candidate_root_vertical_delta is not None
            and candidate_root_vertical_delta <= root_vertical_delta_limit
        ),
        "opensim_mean_rms_not_worse": _not_worse_ratio(
            candidate_mean_rms,
            baseline_mean_rms,
            opensim_mean_rms_ratio,
        ),
        "opensim_max_rms_not_worse": _not_worse_ratio(
            candidate_max_rms,
            baseline_max_rms,
            opensim_max_rms_ratio,
        ),
        "opensim_coordinate_range_violations_not_worse": _not_worse_abs(
            candidate_range_violations,
            baseline_range_violations,
            0.0,
        ),
        "opensim_coordinate_jumps_not_worse": _not_worse_abs(
            candidate_coordinate_jumps,
            baseline_coordinate_jumps,
            0.0,
        ),
        "at_least_one_smoothness_metric_improved": any(
            target_improvements.values()
        ),
    }
    accepted = all(checks.values())

    return {
        "selected": candidate if accepted else baseline,
        "accepted": accepted,
        "checks": checks,
        "target_improvements": target_improvements,
        "baseline": baseline,
        "candidate": candidate,
    }
