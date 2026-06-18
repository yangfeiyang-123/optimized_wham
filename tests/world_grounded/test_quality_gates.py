from lib.world_grounded.quality_gates import QualityGateConfig, evaluate_quality_gates


def test_quality_gates_rejects_bad_reference_and_reports_failed_gate():
    report = {
        "ground_confidence": "high",
        "beta_variation_after_max_abs": 0.0,
        "foot_penetration_max_cm_after": 1.0,
        "stance_sliding_max_cm_after": 8.0,
        "root_delta_y_max_cm": 3.0,
        "root_delta_xz_max_cm": 0.0,
        "contact_switch_rate": 0.05,
        "stance_coverage_ratio": 0.5,
    }

    result = evaluate_quality_gates(report, QualityGateConfig(max_stance_sliding_cm=5.0))

    assert result["usable_for_training"] is False
    assert result["quality_tier"] == "D"
    assert result["failed_gates"][0]["name"] == "max_stance_sliding_cm"


def test_quality_gates_assigns_tier_a_for_clean_reference():
    report = {
        "ground_confidence": "high",
        "beta_variation_after_max_abs": 0.0,
        "foot_penetration_max_cm_after": 0.8,
        "stance_sliding_max_cm_after": 2.0,
        "root_delta_y_max_cm": 5.0,
        "root_delta_xz_max_cm": 0.0,
        "contact_switch_rate": 0.05,
        "stance_coverage_ratio": 0.5,
    }

    result = evaluate_quality_gates(report)

    assert result["usable_for_training"] is True
    assert result["quality_tier"] == "A"
    assert result["failed_gates"] == []


def test_quality_gates_rejects_contact_speed_spike():
    report = {
        "ground_confidence": "high",
        "beta_variation_after_max_abs": 0.0,
        "foot_penetration_max_cm_after": 0.5,
        "stance_sliding_max_cm_after": 2.0,
        "stance_contact_speed_max_mps_after": 9.5,
        "root_delta_y_max_cm": 3.0,
        "root_delta_xz_max_cm": 0.0,
        "contact_switch_rate": 0.05,
        "stance_coverage_ratio": 0.5,
    }

    result = evaluate_quality_gates(report, QualityGateConfig(max_stance_contact_speed_mps=6.0))

    assert result["usable_for_training"] is False
    assert result["quality_tier"] == "D"
    assert result["failed_gates"][0]["name"] == "max_stance_contact_speed_mps"


def test_quality_gates_use_contact_coverage_when_stance_coverage_missing():
    report = {
        "ground_confidence": "high",
        "beta_variation_after_max_abs": 0.0,
        "foot_penetration_max_cm_after": 0.8,
        "stance_sliding_max_cm_after": 2.0,
        "root_delta_y_max_cm": 5.0,
        "root_delta_xz_max_cm": 0.0,
        "contact_switch_rate": 0.05,
        "contact_coverage": 0.5,
    }

    result = evaluate_quality_gates(report)

    assert result["usable_for_training"] is True
    assert "min_stance_coverage" in result["passed_gates"]
