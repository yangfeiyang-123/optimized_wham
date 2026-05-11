import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

import scripts.optimize_smpl_lower_body as lower_body_script


def test_optimize_smpl_lower_body_script_writes_outputs(tmp_path):
    input_pkl = tmp_path / "input.pkl"
    out_dir = tmp_path / "out"
    record = {
        "betas": np.zeros((2, 10), dtype=np.float32),
        "pose": np.zeros((2, 72), dtype=np.float32),
        "trans_world": np.zeros((2, 3), dtype=np.float32),
        "feet_refined": np.array([[[0.0, -0.1, 0.0]], [[0.0, 0.0, 0.0]]], dtype=np.float32),
        "contact": np.ones((2, 1), dtype=np.float32),
    }
    joblib.dump({"0": record}, input_pkl)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/optimize_smpl_lower_body.py",
            "--input-pkl",
            str(input_pkl),
            "--out-dir",
            str(out_dir),
            "--fps",
            "30",
            "--track-id",
            "0",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "corrected_smpl_pkl:" in result.stdout
    assert (out_dir / "corrected_smpl.pkl").exists()
    assert (out_dir / "validation_summary.json").exists()

    corrected = joblib.load(out_dir / "corrected_smpl.pkl")
    np.testing.assert_array_equal(corrected["0"]["betas"], record["betas"])

    for report_name in (
        "lower_body_optimization_report",
        "ground_contact_report",
        "pose_delta_report",
        "validation_summary",
    ):
        assert (out_dir / f"{report_name}.json").exists()

    validation_summary = json.loads((out_dir / "validation_summary.json").read_text(encoding="utf-8"))
    assert validation_summary["success"] is True


def test_optimize_smpl_lower_body_script_accepts_pipeline_cli_options(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "optimize_smpl_lower_body.py",
            "--input-pkl",
            "input.pkl",
            "--out-dir",
            "out",
            "--fps",
            "30",
            "--track-id",
            "0",
            "--max-root-y-shift",
            "0.1",
            "--device",
            "cpu",
            "--enable-pose-pass",
            "--pose-iterations",
            "1",
        ],
    )

    args = lower_body_script.parse_args()

    assert args.max_root_y_shift == pytest.approx(0.1)
    assert args.device == "cpu"
    assert args.enable_pose_pass is True
    assert args.pose_iterations == 1
