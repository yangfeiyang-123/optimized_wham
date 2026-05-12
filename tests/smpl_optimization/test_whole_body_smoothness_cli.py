import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.whole_body_smoothness_optimizer import (  # noqa: E402
    build_retarget_cmd,
    find_opensim_mot,
    stage7_paths,
)


def _args(free_root=True):
    return argparse.Namespace(
        fps=60.0,
        track_id="merge",
        device="cuda",
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


def test_stage7_paths_are_named_for_whole_body_smoothness(tmp_path):
    paths = stage7_paths(tmp_path)

    assert paths["baseline_smpl"].name == "baseline_smpl.pkl"
    assert paths["candidate_smpl"].name == "smooth_candidate_smpl.pkl"
    assert paths["selected_smpl"].name == "selected_smooth_smpl.pkl"
    assert paths["baseline_opensim"].name == "baseline_opensim"
    assert paths["candidate_opensim"].name == "candidate_opensim"
    assert paths["report"].name == "stage7_smoothness_report.json"


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


def test_find_opensim_mot_prefers_opensim_ik(tmp_path):
    fallback = tmp_path / "other.mot"
    preferred = tmp_path / "opensim_ik.mot"
    fallback.write_text("fallback", encoding="utf-8")
    preferred.write_text("preferred", encoding="utf-8")

    assert find_opensim_mot(tmp_path) == preferred


def test_find_opensim_mot_raises_when_no_mot_exists(tmp_path):
    try:
        find_opensim_mot(tmp_path)
    except FileNotFoundError as exc:
        assert str(tmp_path) in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
