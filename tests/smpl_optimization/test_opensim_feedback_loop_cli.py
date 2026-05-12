import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.opensim_feedback_loop import (  # noqa: E402
    build_lower_body_cmd,
    build_retarget_cmd,
    iter_paths,
)


def _args(free_root=True):
    return argparse.Namespace(
        fps=60.0,
        track_id="merge",
        max_root_y_shift=0.25,
        device="cuda",
        pose_iterations=80,
        retarget_config="configs/retarget/smpl_to_mimicmsk_opensim.yaml",
        chunk_size=256,
        alignment="anatomical",
        axis_conversion="mujoco_to_opensim",
        root_calibration="constant_from_free_ik",
        root_calib_frames=10,
        opensim_cmd="C:/OpenSim/bin/opensim-cmd.exe",
        free_root=free_root,
        ik_accuracy=1e-4,
    )


def test_iter_paths_are_named_for_stage6(tmp_path):
    paths = iter_paths(tmp_path)

    assert paths["iter_00_smpl"].name == "iter_00_stage5_smpl.pkl"
    assert paths["iter_01_smpl"].name == "iter_01_opensim_feedback_smpl.pkl"
    assert paths["selected_smpl"].name == "selected_corrected_smpl.pkl"


def test_lower_body_cmd_uses_frame_weights_when_given(tmp_path):
    args = _args()
    cmd = build_lower_body_cmd(
        args,
        tmp_path / "input.pkl",
        tmp_path / "out",
        tmp_path / "weights.npy",
    )

    assert "--enable-pose-pass" in cmd
    assert "--frame-weights" in cmd
    assert cmd[cmd.index("--frame-weights") + 1].endswith("weights.npy")


def test_lower_body_cmd_omits_frame_weights_when_not_given(tmp_path):
    args = _args()
    cmd = build_lower_body_cmd(args, tmp_path / "input.pkl", tmp_path / "out", None)

    assert "--enable-pose-pass" in cmd
    assert "--frame-weights" not in cmd


def test_retarget_cmd_requests_log_run_ik_and_free_root(tmp_path):
    args = _args(free_root=True)
    cmd = build_retarget_cmd(
        args,
        tmp_path / "input.pkl",
        tmp_path / "retarget",
        tmp_path / "ik.log",
    )

    assert "--run-ik" in cmd
    assert "--free-root" in cmd
    assert "--opensim-log" in cmd
    assert cmd[cmd.index("--opensim-log") + 1].endswith("ik.log")


def test_retarget_cmd_omits_free_root_when_false(tmp_path):
    args = _args(free_root=False)
    cmd = build_retarget_cmd(
        args,
        tmp_path / "input.pkl",
        tmp_path / "retarget",
        tmp_path / "ik.log",
    )

    assert "--run-ik" in cmd
    assert "--opensim-log" in cmd
    assert "--free-root" not in cmd
