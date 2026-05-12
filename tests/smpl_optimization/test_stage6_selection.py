import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.stage6_selection import select_stage6_result  # noqa: E402


def _summary(
    *,
    mean_rms=0.05,
    max_rms=0.12,
    lower_body_mean_rms=0.05,
    penetration=0.002,
    sliding=0.1,
    root_jitter=0.2,
    beta_var=0.0,
    frames=100,
    upper_delta=0.0,
    lower_delta=0.3,
    root_shift=0.1,
):
    return {
        "opensim": {
            "mean_rms": mean_rms,
            "max_rms": max_rms,
            "lower_body_mean_rms": lower_body_mean_rms,
        },
        "smpl": {
            "beta_variation_after_max_abs": beta_var,
            "frames": frames,
            "penetration": {"max_penetration": penetration},
            "sliding": {"mean_contact_speed": sliding},
            "root": {"rms_vertical_accel": root_jitter},
            "pose_delta": {
                "upper_body_max_abs": upper_delta,
                "lower_body_max_abs": lower_delta,
            },
            "total_root_y_shift_max_abs": root_shift,
        },
    }


def test_selects_iter_01_when_hard_checks_pass_and_target_improves():
    baseline = _summary(lower_body_mean_rms=0.08, penetration=0.003)
    candidate = _summary(lower_body_mean_rms=0.06, penetration=0.002)

    result = select_stage6_result(baseline, candidate, root_y_budget=0.25)

    assert result["selected"] == "iter_01"
    assert result["checks"]["at_least_one_target_metric_improved"]


def test_rolls_back_when_opensim_mean_rms_worsens():
    baseline = _summary(mean_rms=0.05)
    candidate = _summary(mean_rms=0.07, lower_body_mean_rms=0.04)

    result = select_stage6_result(baseline, candidate, root_y_budget=0.25)

    assert result["selected"] == "iter_00"
    assert not result["checks"]["opensim_mean_rms_not_worse"]


def test_rolls_back_when_no_metric_improves():
    baseline = _summary()
    candidate = _summary()

    result = select_stage6_result(baseline, candidate, root_y_budget=0.25)

    assert result["selected"] == "iter_00"
    assert not result["checks"]["at_least_one_target_metric_improved"]


def test_rolls_back_when_root_shift_exceeds_budget():
    baseline = _summary()
    candidate = _summary(lower_body_mean_rms=0.04, root_shift=0.4)

    result = select_stage6_result(baseline, candidate, root_y_budget=0.25)

    assert result["selected"] == "iter_00"
    assert not result["checks"]["root_y_total_shift_within_budget"]
