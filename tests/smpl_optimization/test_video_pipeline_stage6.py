import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.video_to_fixed_smpl_to_opensim import build_opensim_feedback_cmd  # noqa: E402


def _args(free_root=True):
    return argparse.Namespace(
        fps=60.0,
        track_id="merge",
        device="cuda",
        chunk_size=256,
        lower_body_max_root_y_shift=0.25,
        lower_body_pose_iterations=80,
        retarget_config="configs/retarget/smpl_to_mimicmsk_opensim.yaml",
        alignment="anatomical",
        axis_conversion="mujoco_to_opensim",
        root_calibration="constant_from_free_ik",
        root_calib_frames=10,
        opensim_cmd="C:/OpenSim/bin/opensim-cmd.exe",
        free_root=free_root,
    )


def test_build_opensim_feedback_cmd_contains_stage6_flags(tmp_path):
    cmd = build_opensim_feedback_cmd(
        _args(True), tmp_path / "world.pkl", tmp_path / "stage6"
    )
    assert "scripts/opensim_feedback_loop.py" in cmd
    assert "--input-pkl" in cmd
    assert cmd[cmd.index("--input-pkl") + 1].endswith("world.pkl")
    assert "--free-root" in cmd
    assert "--fix-root" not in cmd


def test_build_opensim_feedback_cmd_can_fix_root(tmp_path):
    cmd = build_opensim_feedback_cmd(
        _args(False), tmp_path / "world.pkl", tmp_path / "stage6"
    )
    assert "--fix-root" in cmd
    assert "--free-root" not in cmd
