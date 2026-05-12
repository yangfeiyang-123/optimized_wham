import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.video_to_fixed_smpl_to_opensim import (  # noqa: E402
    build_parser,
    build_whole_body_smooth_cmd,
)


def _args(free_root=True):
    return argparse.Namespace(
        fps=60.0,
        track_id="merge",
        device="cuda",
        chunk_size=256,
        retarget_config="configs/retarget/smpl_to_mimicmsk_opensim.yaml",
        alignment="anatomical",
        axis_conversion="mujoco_to_opensim",
        root_calibration="constant_from_free_ik",
        root_calib_frames=10,
        opensim_cmd="C:/OpenSim/bin/opensim-cmd.exe",
        free_root=free_root,
    )


def test_build_parser_accepts_whole_body_smooth_flag():
    args = build_parser().parse_args(
        [
            "--video",
            "examples/forehand_clear/video1.mp4",
            "--fps",
            "60",
            "--world-grounded",
            "--optimize-lower-body",
            "--whole-body-smooth",
            "--free-root",
        ]
    )

    assert args.whole_body_smooth is True


def test_build_whole_body_smooth_cmd_contains_stage7_flags(tmp_path):
    cmd = build_whole_body_smooth_cmd(
        _args(True), tmp_path / "selected.pkl", tmp_path / "stage7"
    )

    assert "scripts/whole_body_smoothness_optimizer.py" in cmd
    assert "--input-pkl" in cmd
    assert cmd[cmd.index("--input-pkl") + 1].endswith("selected.pkl")
    assert "--out-dir" in cmd
    assert cmd[cmd.index("--out-dir") + 1].endswith("stage7")
    assert "--free-root" in cmd
    assert "--fix-root" not in cmd


def test_build_whole_body_smooth_cmd_can_fix_root(tmp_path):
    cmd = build_whole_body_smooth_cmd(
        _args(False), tmp_path / "selected.pkl", tmp_path / "stage7"
    )

    assert "--fix-root" in cmd
    assert "--free-root" not in cmd
