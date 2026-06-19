import json

import numpy as np

from lib.world_grounded.reference_bundle import export_reference_bundle


def test_export_reference_bundle_writes_manifest_motion_and_contact(tmp_path):
    record = {
        "pose": np.zeros((3, 72), dtype=np.float32),
        "trans_world": np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], dtype=np.float32),
        "betas": np.zeros((3, 10), dtype=np.float32),
        "gender": "neutral",
        "feet_world": np.zeros((3, 2, 3), dtype=np.float32),
        "contact": np.array([[1.0, 0.0], [0.8, 0.0], [0.0, 0.9]], dtype=np.float32),
    }

    manifest_path = export_reference_bundle(
        record,
        tmp_path,
        sequence="demo",
        fps=60.0,
        quality_report={"usable_for_training": True, "quality_tier": "A"},
        source={"canonical_pkl": "canonical.pkl"},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    motion = np.load(tmp_path / manifest["motion_npz"])
    contact = np.load(tmp_path / manifest["contact_npz"], allow_pickle=True)

    assert manifest["version"] == "contact_reference_bundle_v1"
    assert manifest["coordinate_system"] == "amass_zup"
    assert manifest["num_frames"] == 3
    assert np.allclose(motion["trans"][0], [1.0, -3.0, 2.0])
    assert contact["stance_mask"].shape == (3, 2)
    assert contact["coordinate_system"].item() == "amass_zup"
    assert np.allclose(motion["poses"][0, :3], [np.pi / 2.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(motion["root_orient"][0], motion["poses"][0, :3])
    assert motion["pose_body"].shape == (3, 69)


def test_export_reference_bundle_preserves_failed_validation_as_unusable(tmp_path):
    record = {
        "pose": np.zeros((2, 72), dtype=np.float32),
        "trans_world": np.zeros((2, 3), dtype=np.float32),
        "betas": np.zeros((10,), dtype=np.float32),
        "feet_world": np.zeros((2, 1, 3), dtype=np.float32),
        "contact": np.ones((2, 1), dtype=np.float32),
    }

    manifest_path = export_reference_bundle(
        record,
        tmp_path,
        sequence="bad_demo",
        fps=60.0,
        quality_report={"success": False, "checks": {"smpl_contact_foot_sliding_reduced": False}},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["quality"]["usable_for_training"] is False
    assert manifest["quality"]["quality_tier"] == "D"


def test_export_reference_bundle_writes_body_keypoints_and_laplacian(tmp_path):
    body_keypoints = np.arange(3 * 21 * 3, dtype=np.float32).reshape(3, 21, 3) / 100.0
    record = {
        "pose": np.zeros((3, 72), dtype=np.float32),
        "trans_world": np.zeros((3, 3), dtype=np.float32),
        "betas": np.zeros((10,), dtype=np.float32),
        "feet_world": np.zeros((3, 2, 3), dtype=np.float32),
        "contact": np.ones((3, 2), dtype=np.float32),
        "body_keypoints": body_keypoints,
    }

    manifest_path = export_reference_bundle(
        record,
        tmp_path,
        sequence="body_demo",
        fps=60.0,
        quality_report={"usable_for_training": True, "quality_tier": "A"},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    motion = np.load(tmp_path / manifest["motion_npz"], allow_pickle=True)

    assert "body_keypoints" in motion
    assert "body_keypoint_labels" in motion
    assert "body_laplacian" in motion
    assert motion["body_keypoints"].shape == (3, 21, 3)
    assert motion["body_laplacian"].shape == (3, 21, 3)
    assert motion["body_keypoints_coordinate_system"].item() == "amass_zup"


def test_export_reference_bundle_runs_quality_gates_for_raw_metric_report(tmp_path):
    record = {
        "pose": np.zeros((4, 72), dtype=np.float32),
        "trans_world": np.zeros((4, 3), dtype=np.float32),
        "betas": np.zeros((10,), dtype=np.float32),
        "feet_world": np.zeros((4, 2, 3), dtype=np.float32),
        "contact": np.ones((4, 2), dtype=np.float32),
    }
    raw_report = {
        "foot_penetration_max_cm_after": 0.5,
        "stance_sliding_max_cm_after": 9.0,
        "root_delta_y_max_cm": 1.0,
        "root_delta_xz_max_cm": 0.0,
        "contact_switch_rate": 0.0,
        "stance_coverage_ratio": 1.0,
        "ground_confidence": "high",
        "beta_variation_after_max_abs": 0.0,
    }

    manifest_path = export_reference_bundle(
        record,
        tmp_path,
        sequence="bad_metric_demo",
        fps=60.0,
        quality_report=raw_report,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["quality"]["usable_for_training"] is False
    assert manifest["quality"]["quality_tier"] == "D"
    assert manifest["quality"]["failed_gates"][0]["name"] == "max_stance_sliding_cm"


def test_export_reference_bundle_does_not_trust_claimed_quality_when_metrics_fail(tmp_path):
    record = {
        "pose": np.zeros((4, 72), dtype=np.float32),
        "trans_world": np.zeros((4, 3), dtype=np.float32),
        "betas": np.zeros((10,), dtype=np.float32),
        "feet_world": np.zeros((4, 2, 3), dtype=np.float32),
        "contact": np.ones((4, 2), dtype=np.float32),
    }
    raw_report = {
        "usable_for_training": True,
        "quality_tier": "A",
        "foot_penetration_max_cm_after": 0.5,
        "stance_sliding_max_cm_after": 9.0,
        "root_delta_y_max_cm": 1.0,
        "root_delta_xz_max_cm": 0.0,
        "contact_switch_rate": 0.0,
        "stance_coverage_ratio": 1.0,
        "ground_confidence": "high",
        "beta_variation_after_max_abs": 0.0,
    }

    manifest_path = export_reference_bundle(
        record,
        tmp_path,
        sequence="claimed_good_metric_bad",
        fps=60.0,
        quality_report=raw_report,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["quality"]["usable_for_training"] is False
    assert manifest["quality"]["quality_tier"] == "D"
    assert manifest["quality"]["failed_gates"][0]["name"] == "max_stance_sliding_cm"


def test_export_reference_bundle_does_not_trust_success_when_checks_fail(tmp_path):
    record = {
        "pose": np.zeros((4, 72), dtype=np.float32),
        "trans_world": np.zeros((4, 3), dtype=np.float32),
        "betas": np.zeros((10,), dtype=np.float32),
        "feet_world": np.zeros((4, 2, 3), dtype=np.float32),
        "contact": np.ones((4, 2), dtype=np.float32),
    }

    manifest_path = export_reference_bundle(
        record,
        tmp_path,
        sequence="claimed_success_checks_bad",
        fps=60.0,
        quality_report={"success": True, "checks": {"smpl_contact_foot_sliding_reduced": False}},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["quality"]["usable_for_training"] is False
    assert manifest["quality"]["quality_tier"] == "D"
    assert manifest["quality"]["failed_gates"][0]["name"] == "smpl_contact_foot_sliding_reduced"


def test_export_reference_bundle_does_not_mark_missing_quality_as_usable(tmp_path):
    record = {
        "pose": np.zeros((4, 72), dtype=np.float32),
        "trans_world": np.zeros((4, 3), dtype=np.float32),
        "betas": np.zeros((10,), dtype=np.float32),
        "feet_world": np.zeros((4, 2, 3), dtype=np.float32),
        "contact": np.ones((4, 2), dtype=np.float32),
    }

    manifest_path = export_reference_bundle(
        record,
        tmp_path,
        sequence="unknown_quality",
        fps=60.0,
        quality_report={},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["quality"]["usable_for_training"] is False
    assert manifest["quality"]["quality_tier"] == "D"
    assert manifest["quality"]["failed_gates"][0]["name"] == "missing_quality_report"


def test_export_reference_bundle_records_lower_body_stage_and_root_smooth_axes(tmp_path):
    record = {
        "pose": np.zeros((4, 72), dtype=np.float32),
        "trans_world": np.zeros((4, 3), dtype=np.float32),
        "betas": np.zeros((10,), dtype=np.float32),
        "feet_world": np.zeros((4, 2, 3), dtype=np.float32),
        "contact": np.ones((4, 2), dtype=np.float32),
    }

    manifest_path = export_reference_bundle(
        record,
        tmp_path,
        sequence="lower_body_demo",
        fps=60.0,
        quality_report={"usable_for_training": True, "quality_tier": "A"},
        source={"corrected_smpl_pkl": "corrected_smpl.pkl"},
        root_smooth_axes=("x", "y"),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    processing = json.loads((tmp_path / manifest["processing_report_json"]).read_text(encoding="utf-8"))

    assert processing["stages"] == [
        "fixed_beta",
        "world_grounded",
        "lower_body",
        "contact_preserving_export",
    ]
    assert processing["parameters"]["root_smooth_axes"] == ["x", "y"]


def test_export_reference_bundle_pads_short_betas_to_ten_dims(tmp_path):
    record = {
        "pose": np.zeros((4, 72), dtype=np.float32),
        "trans_world": np.zeros((4, 3), dtype=np.float32),
        "betas": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "feet_world": np.zeros((4, 2, 3), dtype=np.float32),
        "contact": np.ones((4, 2), dtype=np.float32),
    }

    manifest_path = export_reference_bundle(
        record,
        tmp_path,
        sequence="short_betas",
        fps=60.0,
        quality_report={"usable_for_training": True, "quality_tier": "A"},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    motion = np.load(tmp_path / manifest["motion_npz"])

    assert motion["betas"].shape == (10,)
    assert np.allclose(motion["betas"][:3], [1.0, 2.0, 3.0])
    assert np.allclose(motion["betas"][3:], 0.0)
