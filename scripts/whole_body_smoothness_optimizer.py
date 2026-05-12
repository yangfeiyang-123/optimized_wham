"""Run Stage7 whole-body SMPL smoothness candidate selection."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.metrics import (  # noqa: E402
    beta_variation_max_abs,
    foot_penetration_depth,
    pose_smoothness,
    root_translation_smoothness,
)
from lib.smpl_optimization.opensim_feedback import parse_ik_log_metrics  # noqa: E402
from lib.smpl_optimization.opensim_motion import summarize_mot_coordinates  # noqa: E402
from lib.smpl_optimization.reports import to_jsonable, write_json  # noqa: E402
from lib.smpl_optimization.stage7_selection import select_stage7_result  # noqa: E402
from lib.smpl_optimization.whole_body_smooth import (  # noqa: E402
    WholeBodySmoothConfig,
    smooth_record,
)
from lib.world_grounded.tracks import select_track, to_numpy  # noqa: E402


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print()
    print("$ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def stage7_paths(out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    return {
        "baseline_smpl": out_dir / "baseline_smpl.pkl",
        "candidate_smpl": out_dir / "smooth_candidate_smpl.pkl",
        "selected_smpl": out_dir / "selected_smooth_smpl.pkl",
        "baseline_opensim": out_dir / "baseline_opensim",
        "baseline_log": out_dir / "baseline_opensim" / "opensim_ik.log",
        "candidate_opensim": out_dir / "candidate_opensim",
        "candidate_log": out_dir / "candidate_opensim" / "opensim_ik.log",
        "report": out_dir / "stage7_smoothness_report.json",
    }


def build_retarget_cmd(
    args: argparse.Namespace,
    input_pkl: Path,
    out_dir: Path,
    log_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/retarget_smpl_to_opensim.py",
        str(input_pkl),
        "--config",
        str(args.retarget_config),
        "--out-dir",
        str(out_dir),
        "--fps",
        str(args.fps),
        "--device",
        str(args.device),
        "--chunk-size",
        str(args.chunk_size),
        "--alignment",
        str(args.alignment),
        "--axis-conversion",
        str(args.axis_conversion),
        "--root-calibration",
        str(args.root_calibration),
        "--root-calib-frames",
        str(args.root_calib_frames),
        "--track-id",
        str(args.track_id),
        "--ik-accuracy",
        str(args.ik_accuracy),
        "--opensim-log",
        str(log_path),
        "--run-ik",
    ]
    if args.opensim_cmd:
        cmd += ["--opensim-cmd", str(args.opensim_cmd)]
    if args.free_root:
        cmd.append("--free-root")
    return cmd


def load_selected_record(pkl: Path, track_id: str) -> tuple[dict, str, dict]:
    data = joblib.load(pkl)
    selected_track_id, record = select_track(data, track_id)
    return data, str(selected_track_id), record


def replace_selected_record(data, selected_track_id: str, record: dict) -> dict:
    if selected_track_id == "merged":
        return {"merged": record}

    updated = deepcopy(data)
    for key in list(updated.keys()):
        if str(key) == str(selected_track_id):
            updated[key] = record
            return updated
    updated[selected_track_id] = record
    return updated


def _record_frames(record: dict) -> int:
    for key in ("trans_world", "trans", "pose_world", "pose", "frame_ids", "frame_id"):
        if key in record:
            return int(np.asarray(to_numpy(record[key])).shape[0])
    return 0


def _record_pose(record: dict) -> np.ndarray:
    for key in ("pose_world", "pose"):
        if key in record:
            pose = np.asarray(to_numpy(record[key]))
            return pose[:, :72] if pose.ndim == 2 and pose.shape[1] >= 72 else pose
    return np.empty((0, 72), dtype=np.float32)


def _record_root(record: dict) -> np.ndarray:
    for key in ("trans_world", "trans"):
        if key in record:
            trans = np.asarray(to_numpy(record[key]))
            return trans[:, :3] if trans.ndim == 2 and trans.shape[1] >= 3 else trans
    return np.empty((0, 3), dtype=np.float32)


def _zero_pose_delta() -> dict:
    return {
        "lower_body_max_abs": 0.0,
        "whole_body_max_abs": 0.0,
    }


def _zero_root_delta() -> dict:
    return {
        "max_abs": 0.0,
        "vertical_max_abs": 0.0,
    }


def _penetration(record: dict) -> dict:
    for key in ("feet_world", "feet", "joints_world", "joints"):
        if key not in record:
            continue
        values = np.asarray(to_numpy(record[key]))
        if values.ndim >= 3 and values.shape[-1] >= 2:
            return foot_penetration_depth(values[..., 1])
    return foot_penetration_depth(np.asarray([], dtype=np.float32))


def summarize_smpl(record: dict, smooth_report: dict | None = None) -> dict:
    summary = {
        "beta_variation_after_max_abs": beta_variation_max_abs(
            record.get("betas", np.asarray([]))
        ),
        "frames": _record_frames(record),
        "penetration": _penetration(record),
        "sliding": {
            "mean_contact_speed": 0.0,
            "max_contact_speed": 0.0,
            "num_sliding": 0,
        },
        "pose_delta": _zero_pose_delta(),
        "root_delta": _zero_root_delta(),
        "pose_smoothness": pose_smoothness(_record_pose(record)),
        "root_smoothness": root_translation_smoothness(_record_root(record)),
    }
    if smooth_report:
        for key in (
            "pose_delta",
            "root_delta",
            "pose_smoothness",
            "root_smoothness",
        ):
            if key in smooth_report:
                summary[key] = smooth_report[key]
    return summary


def find_opensim_mot(opensim_dir: Path) -> Path:
    opensim_dir = Path(opensim_dir)
    preferred = opensim_dir / "opensim_ik.mot"
    if preferred.exists():
        return preferred
    candidates = sorted(opensim_dir.glob("*.mot"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No .mot file found in {opensim_dir}")


def _opensim_summary(log_path: Path, opensim_dir: Path) -> dict:
    opensim = parse_ik_log_metrics(log_path)
    opensim["motion"] = summarize_mot_coordinates(find_opensim_mot(opensim_dir))
    return opensim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--track-id", default="merge")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--retarget-config",
        default="configs/retarget/smpl_to_mimicmsk_opensim.yaml",
    )
    parser.add_argument(
        "--alignment",
        choices=["mapping", "anatomical"],
        default="anatomical",
    )
    parser.add_argument(
        "--axis-conversion",
        choices=["mujoco_to_opensim", "opensim_to_mujoco", "identity"],
        default="mujoco_to_opensim",
    )
    parser.add_argument(
        "--root-calibration",
        choices=["none", "constant_from_free_ik"],
        default="constant_from_free_ik",
    )
    parser.add_argument("--root-calib-frames", type=int, default=10)
    parser.add_argument("--opensim-cmd", default=None)
    parser.add_argument("--ik-accuracy", type=float, default=1e-4)
    parser.add_argument("--free-root", dest="free_root", action="store_true", default=True)
    parser.add_argument("--fix-root", dest="free_root", action="store_false")
    parser.add_argument("--lower-body-weight", type=float, default=0.35)
    parser.add_argument("--trunk-weight", type=float, default=0.55)
    parser.add_argument("--upper-body-weight", type=float, default=0.8)
    parser.add_argument("--max-lower-body-delta", type=float, default=0.05)
    parser.add_argument("--max-whole-body-delta", type=float, default=0.12)
    parser.add_argument("--max-root-delta", type=float, default=0.03)
    parser.add_argument("--max-root-vertical-delta", type=float, default=0.015)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _smooth_config(args: argparse.Namespace) -> WholeBodySmoothConfig:
    return WholeBodySmoothConfig(
        lower_body_weight=args.lower_body_weight,
        trunk_weight=args.trunk_weight,
        upper_body_weight=args.upper_body_weight,
        max_lower_body_delta=args.max_lower_body_delta,
        max_whole_body_delta=args.max_whole_body_delta,
        max_root_delta=args.max_root_delta,
        max_root_vertical_delta=args.max_root_vertical_delta,
    )


def main() -> int:
    args = parse_args()
    input_pkl = Path(args.input_pkl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = stage7_paths(out_dir)

    shutil.copy2(input_pkl, paths["baseline_smpl"])
    data, selected_track_id, record = load_selected_record(
        paths["baseline_smpl"], args.track_id
    )
    candidate_record, smooth_report = smooth_record(record, _smooth_config(args))
    candidate_data = replace_selected_record(data, selected_track_id, candidate_record)
    joblib.dump(candidate_data, paths["candidate_smpl"])

    run(
        build_retarget_cmd(
            args,
            paths["baseline_smpl"],
            paths["baseline_opensim"],
            paths["baseline_log"],
        )
    )
    run(
        build_retarget_cmd(
            args,
            paths["candidate_smpl"],
            paths["candidate_opensim"],
            paths["candidate_log"],
        )
    )

    baseline = {
        "smpl": summarize_smpl(record),
        "opensim": _opensim_summary(paths["baseline_log"], paths["baseline_opensim"]),
    }
    candidate = {
        "smpl": summarize_smpl(candidate_record, smooth_report),
        "opensim": _opensim_summary(
            paths["candidate_log"], paths["candidate_opensim"]
        ),
    }
    selection = select_stage7_result(baseline, candidate)
    selection["smooth_report"] = smooth_report
    selection["selected_track_id"] = selected_track_id

    selected_source = (
        paths["candidate_smpl"]
        if selection["selected"] == "candidate"
        else paths["baseline_smpl"]
    )
    shutil.copy2(selected_source, paths["selected_smpl"])
    write_json(paths["report"], to_jsonable(selection))

    print(f"selected: {selection['selected']}")
    print(f"selected_smooth_smpl: {paths['selected_smpl']}")
    print(f"stage7_smoothness_report: {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
