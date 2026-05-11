import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_wham_ablation import (  # noqa: E402
    STAGES,
    build_lower_body_cmd,
    safe_ascii_name,
    write_metrics_csv,
)


def test_ablation_stage_ids_are_expected():
    assert [stage["id"] for stage in STAGES] == [
        "00_raw_wham",
        "01_fixed_beta",
        "02_world_grounded",
        "03_root_y_only",
        "04_lower_body_full",
    ]


def test_safe_ascii_name_keeps_output_ascii():
    value = safe_ascii_name("5月1日-2")

    value.encode("ascii")
    assert value.endswith("_43e0a6ee")


def test_lower_body_command_pose_pass_flag_differs_by_stage(tmp_path):
    args = argparse.Namespace(
        fps=60.0,
        track_id="merge",
        lower_body_max_root_y_shift=0.25,
        lower_body_pose_iterations=80,
        device="cuda",
    )
    input_pkl = tmp_path / "input.pkl"
    out_dir = tmp_path / "out"

    root_only = build_lower_body_cmd(args, input_pkl, out_dir, enable_pose_pass=False)
    full = build_lower_body_cmd(args, input_pkl, out_dir, enable_pose_pass=True)

    assert "--enable-pose-pass" not in root_only
    assert "--pose-iterations" not in root_only
    assert "--enable-pose-pass" in full
    assert full[full.index("--pose-iterations") + 1] == "80"


def test_write_metrics_csv_preserves_stage_rows(tmp_path):
    path = tmp_path / "metrics.csv"
    rows = [
        {
            "stage_id": "00_raw_wham",
            "description": "raw",
            "pkl": "raw.pkl",
            "frames": 10,
            "beta_variation_max_abs": 0.5,
            "extra_metric": "kept",
        },
        {
            "stage_id": "01_fixed_beta",
            "description": "fixed",
            "pkl": "fixed.pkl",
            "frames": 10,
            "beta_variation_max_abs": 0.0,
        },
    ]

    write_metrics_csv(path, rows)

    with path.open("r", newline="", encoding="utf-8") as f:
        loaded = list(csv.DictReader(f))

    assert loaded[0]["stage_id"] == "00_raw_wham"
    assert loaded[0]["extra_metric"] == "kept"
    assert loaded[1]["stage_id"] == "01_fixed_beta"
    assert loaded[1]["beta_variation_max_abs"] == "0.0"
