from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class QualityGateConfig:
    max_foot_penetration_cm: float = 3.0
    max_stance_sliding_cm: float = 5.0
    max_stance_contact_speed_mps: float = 12.0
    max_root_delta_y_cm: float = 25.0
    max_root_delta_xz_cm: float = 5.0
    max_lower_body_pose_delta_rad: float = 0.8
    max_contact_switch_rate: float = 0.25
    min_stance_coverage: float = 0.10
    require_fixed_beta: bool = True
    allowed_ground_confidence: tuple[str, ...] = ("medium", "high")


def evaluate_quality_gates(report: dict[str, Any], config: Optional[QualityGateConfig] = None) -> dict[str, Any]:
    cfg = config or QualityGateConfig()
    failed: list[dict[str, Any]] = []
    passed: list[str] = []

    _check_max(failed, passed, "max_foot_penetration_cm", _get(report, "foot_penetration_max_cm_after"), cfg.max_foot_penetration_cm)
    _check_max(failed, passed, "max_stance_sliding_cm", _get(report, "stance_sliding_max_cm_after"), cfg.max_stance_sliding_cm)
    _check_max(
        failed,
        passed,
        "max_stance_contact_speed_mps",
        _get(report, "stance_contact_speed_max_mps_after", "after.sliding.max_contact_speed"),
        cfg.max_stance_contact_speed_mps,
    )
    _check_max(failed, passed, "max_root_delta_y_cm", _get(report, "root_delta_y_max_cm"), cfg.max_root_delta_y_cm)
    _check_max(failed, passed, "max_root_delta_xz_cm", _get(report, "root_delta_xz_max_cm"), cfg.max_root_delta_xz_cm)
    _check_max(
        failed,
        passed,
        "max_lower_body_pose_delta_rad",
        _get(report, "lower_body_pose_delta_max_abs", "pose_delta_report.total_max_abs"),
        cfg.max_lower_body_pose_delta_rad,
        missing_passes=True,
    )
    _check_max(failed, passed, "max_contact_switch_rate", _get(report, "contact_switch_rate"), cfg.max_contact_switch_rate)
    _check_min(
        failed,
        passed,
        "min_stance_coverage",
        _get(report, "stance_coverage_ratio", "contact_coverage"),
        cfg.min_stance_coverage,
    )

    ground_confidence = str(_get(report, "ground_confidence", default="")).lower()
    if ground_confidence in set(cfg.allowed_ground_confidence):
        passed.append("ground_confidence")
    else:
        failed.append(
            {
                "name": "ground_confidence",
                "value": ground_confidence,
                "threshold": list(cfg.allowed_ground_confidence),
            }
        )

    beta_variation = float(_get(report, "beta_variation_after_max_abs", default=0.0) or 0.0)
    if cfg.require_fixed_beta and beta_variation > 1e-7:
        failed.append({"name": "fixed_beta", "value": beta_variation, "threshold": 1e-7})
    else:
        passed.append("fixed_beta")

    tier = _quality_tier(report, failed, cfg)
    return {
        "usable_for_training": tier in ("A", "B", "C"),
        "quality_tier": tier,
        "failed_gates": failed,
        "passed_gates": passed,
        "recommendation": "exclude_from_training" if tier == "D" else "usable_with_curriculum",
    }


def _quality_tier(report: dict[str, Any], failed: list[dict[str, Any]], cfg: QualityGateConfig) -> str:
    if failed:
        return "D"
    penetration = float(_get(report, "foot_penetration_max_cm_after", default=0.0) or 0.0)
    sliding = float(_get(report, "stance_sliding_max_cm_after", default=0.0) or 0.0)
    if penetration <= cfg.max_foot_penetration_cm * 0.5 and sliding <= cfg.max_stance_sliding_cm * 0.5:
        return "A"
    if penetration <= cfg.max_foot_penetration_cm and sliding <= cfg.max_stance_sliding_cm:
        return "B"
    return "C"


def _check_max(
    failed: list[dict[str, Any]],
    passed: list[str],
    name: str,
    value: Any,
    threshold: float,
    *,
    missing_passes: bool = False,
) -> None:
    if value is None and missing_passes:
        passed.append(name)
        return
    numeric = float(value or 0.0)
    if numeric > float(threshold):
        failed.append({"name": name, "value": numeric, "threshold": float(threshold)})
    else:
        passed.append(name)


def _check_min(failed: list[dict[str, Any]], passed: list[str], name: str, value: Any, threshold: float) -> None:
    numeric = float(value or 0.0)
    if numeric < float(threshold):
        failed.append({"name": name, "value": numeric, "threshold": float(threshold)})
    else:
        passed.append(name)


def _get(report: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in report:
            return report[key]
        current: Any = report
        found = True
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break
        if found:
            return current
    return default
