import json
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(to_jsonable(key)): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def _nested_number(data: dict, *keys: str) -> float | None:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reduced(before: dict, after: dict, *keys: str) -> bool:
    before_value = _nested_number(before, *keys)
    after_value = _nested_number(after, *keys)
    return before_value is not None and after_value is not None and after_value < before_value


def _not_worse(before: dict, after: dict, *keys: str) -> bool:
    before_value = _nested_number(before, *keys)
    after_value = _nested_number(after, *keys)
    return before_value is not None and after_value is not None and after_value <= before_value


def build_validation_summary(
    *,
    beta_variation_after_max_abs: float,
    frame_count_unchanged: bool,
    before: dict,
    after: dict,
    pose_delta: dict,
    opensim: dict,
) -> dict:
    upper_body_max_abs = _nested_number(pose_delta, "upper_body_max_abs")
    lower_body_max_abs = _nested_number(pose_delta, "lower_body_max_abs")
    opensim_status = str(opensim.get("status", "run"))
    opensim_used_for_success = bool(opensim.get("used_for_success", opensim_status != "not_run"))
    opensim = {
        **opensim,
        "status": opensim_status,
        "used_for_success": opensim_used_for_success,
    }
    opensim_ik_rms_not_worse = (
        True if not opensim_used_for_success else bool(opensim.get("ik_rms_not_worse", False))
    )
    opensim_ground_clearance_not_worse = (
        True if not opensim_used_for_success else bool(opensim.get("ground_clearance_not_worse", False))
    )

    checks = {
        "beta_variation_after_max_abs": beta_variation_after_max_abs == 0.0,
        "frame_count_unchanged": bool(frame_count_unchanged),
        "smpl_foot_penetration_reduced": _not_worse(before, after, "penetration", "max_penetration"),
        "smpl_contact_foot_sliding_reduced": _not_worse(before, after, "sliding", "mean_contact_speed"),
        "root_vertical_jitter_not_worse": _not_worse(before, after, "root", "rms_vertical_accel"),
        "upper_body_pose_delta_small": upper_body_max_abs is not None and upper_body_max_abs <= 0.05,
        "lower_body_pose_delta_bounded": lower_body_max_abs is not None and lower_body_max_abs <= 1.2,
        "opensim_ik_rms_not_worse": opensim_ik_rms_not_worse,
        "opensim_ground_clearance_not_worse": opensim_ground_clearance_not_worse,
    }

    summary = {
        "success": all(checks.values()),
        "checks": checks,
        "before": before,
        "after": after,
        "pose_delta": pose_delta,
        "opensim": opensim,
    }
    return to_jsonable(summary)
