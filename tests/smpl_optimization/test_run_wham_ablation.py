import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_wham_ablation import (  # noqa: E402
    STAGES,
    build_demo_cmd,
    build_lower_body_cmd,
    build_opensim_feedback_cmd,
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
        "05_opensim_feedback",
    ]


def test_safe_ascii_name_keeps_output_ascii():
    value = safe_ascii_name("5月1日-2")

    value.encode("ascii")
    assert value.endswith("_43e0a6ee")


def test_demo_command_uses_posix_video_path_for_demo_sequence_parsing(tmp_path):
    args = argparse.Namespace(
        pose_backend="rtmpose",
        max_frames=None,
        run_smplify=False,
        fast=False,
    )
    video = Path(r"D:\Main\Research\video.mp4")

    cmd = build_demo_cmd(args, video, tmp_path)

    assert cmd[cmd.index("--video") + 1] == "D:/Main/Research/video.mp4"


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


def make_opensim_feedback_args(*, free_root: bool) -> argparse.Namespace:
    return argparse.Namespace(
        fps=60.0,
        track_id="merge",
        device="cuda",
        chunk_size=128,
        lower_body_max_root_y_shift=0.2,
        lower_body_pose_iterations=40,
        retarget_config="configs/retarget/test.yaml",
        alignment="anatomical",
        axis_conversion="mujoco_to_opensim",
        root_calibration="constant_from_free_ik",
        root_calib_frames=7,
        opensim_cmd="opensim-cmd",
        ik_accuracy=1e-5,
        free_root=free_root,
    )


def test_opensim_feedback_command_uses_world_pkl_and_fixed_root_by_default(tmp_path):
    args = make_opensim_feedback_args(free_root=False)
    input_pkl = tmp_path / "world.pkl"
    out_dir = tmp_path / "05_opensim_feedback"

    cmd = build_opensim_feedback_cmd(args, input_pkl, out_dir)

    assert cmd[:2] == [sys.executable, "scripts/opensim_feedback_loop.py"]
    assert cmd[cmd.index("--input-pkl") + 1] == str(input_pkl)
    assert cmd[cmd.index("--out-dir") + 1] == str(out_dir)
    assert cmd[cmd.index("--max-root-y-shift") + 1] == "0.2"
    assert cmd[cmd.index("--pose-iterations") + 1] == "40"
    assert cmd[cmd.index("--retarget-config") + 1] == "configs/retarget/test.yaml"
    assert cmd[cmd.index("--root-calib-frames") + 1] == "7"
    assert cmd[cmd.index("--opensim-cmd") + 1] == "opensim-cmd"
    assert cmd[cmd.index("--ik-accuracy") + 1] == "1e-05"
    assert "--fix-root" in cmd
    assert "--free-root" not in cmd


def test_opensim_feedback_command_uses_free_root_when_requested(tmp_path):
    args = make_opensim_feedback_args(free_root=True)
    input_pkl = tmp_path / "world.pkl"
    out_dir = tmp_path / "05_opensim_feedback"

    cmd = build_opensim_feedback_cmd(args, input_pkl, out_dir)

    assert "--free-root" in cmd
    assert "--fix-root" not in cmd


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
