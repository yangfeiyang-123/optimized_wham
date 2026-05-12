from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.metrics import beta_variation_max_abs
from lib.smpl_optimization.opensim_feedback import (
    build_feedback_weights,
    classify_marker,
    parse_ik_log_metrics,
)
from lib.smpl_optimization.reports import to_jsonable, write_json
from lib.smpl_optimization.stage6_selection import select_stage6_result
from lib.world_grounded.tracks import select_track


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print()
    print("$ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def iter_paths(out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    return {
        "iter_00_dir": out_dir / "iter_00_stage5",
        "iter_00_smpl": out_dir / "iter_00_stage5_smpl.pkl",
        "iter_00_opensim": out_dir / "iter_00_opensim",
        "iter_00_log": out_dir / "iter_00_opensim" / "opensim_ik.log",
        "weights": out_dir / "opensim_feedback_weights.npy",
        "feedback_report": out_dir / "opensim_feedback_report.json",
        "iter_01_dir": out_dir / "iter_01_opensim_feedback",
        "iter_01_smpl": out_dir / "iter_01_opensim_feedback_smpl.pkl",
        "iter_01_opensim": out_dir / "iter_01_opensim",
        "iter_01_log": out_dir / "iter_01_opensim" / "opensim_ik.log",
        "selection_report": out_dir / "stage6_selection_report.json",
        "selected_smpl": out_dir / "selected_corrected_smpl.pkl",
    }


def build_lower_body_cmd(
    args: argparse.Namespace,
    input_pkl: Path,
    out_dir: Path,
    frame_weights: Path | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/optimize_smpl_lower_body.py",
        "--input-pkl",
        str(input_pkl),
        "--out-dir",
        str(out_dir),
        "--fps",
        str(args.fps),
        "--track-id",
        str(args.track_id),
        "--max-root-y-shift",
        str(args.max_root_y_shift),
        "--device",
        str(args.device),
        "--enable-pose-pass",
        "--pose-iterations",
        str(args.pose_iterations),
    ]
    if frame_weights is not None:
        cmd += ["--frame-weights", str(frame_weights)]
    return cmd


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


def load_selected_record(pkl: Path, track_id: str) -> tuple[str, dict]:
    data = joblib.load(pkl)
    selected_track_id, record = select_track(data, track_id)
    return str(selected_track_id), record


def copy_selected(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def num_record_frames(record: dict) -> int:
    for key in ("trans_world", "trans", "pose_world", "pose"):
        if key in record:
            return int(np.asarray(record[key]).shape[0])
    return 0


def _read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_smpl_report(pkl: Path, report_dir: Path, track_id: str) -> dict:
    _, record = load_selected_record(pkl, track_id)
    validation = _read_json_if_exists(report_dir / "validation_summary.json")
    lower_report = _read_json_if_exists(report_dir / "lower_body_optimization_report.json")
    pose_delta = validation.get("pose_delta") or lower_report.get("pose_delta") or {}
    after = validation.get("after") or lower_report.get("after") or {}
    return {
        "beta_variation_after_max_abs": beta_variation_max_abs(
            record.get("betas", np.asarray([]))
        ),
        "frames": num_record_frames(record),
        "penetration": after.get("penetration", {}),
        "sliding": after.get("sliding", {}),
        "root": after.get("root", {}),
        "pose_delta": pose_delta,
        "total_root_y_shift_max_abs": lower_report.get(
            "total_root_y_shift_max_abs", 0.0
        ),
    }


def lower_body_mean_rms(metrics: dict) -> float:
    values = [
        float(frame["rms"])
        for frame in metrics.get("frames", [])
        if classify_marker(frame.get("max_marker", "")) == "lower_body"
    ]
    return float(round(float(np.mean(values)), 8)) if values else 0.0


def build_summary(smpl_pkl: Path, report_dir: Path, ik_log: Path, track_id: str) -> dict:
    opensim = parse_ik_log_metrics(ik_log)
    opensim["lower_body_mean_rms"] = lower_body_mean_rms(opensim)
    return {
        "opensim": opensim,
        "smpl": summarize_smpl_report(smpl_pkl, report_dir, track_id),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage6 OpenSim feedback loop.")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--track-id", default="merge")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-root-y-shift", type=float, default=0.25)
    parser.add_argument("--pose-iterations", type=int, default=80)
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
    parser.add_argument("--rms-threshold", type=float, default=0.08)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_pkl = Path(args.input_pkl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = iter_paths(out_dir)

    run(build_lower_body_cmd(args, input_pkl, paths["iter_00_dir"], None))
    copy_selected(paths["iter_00_dir"] / "corrected_smpl.pkl", paths["iter_00_smpl"])
    run(
        build_retarget_cmd(
            args,
            paths["iter_00_smpl"],
            paths["iter_00_opensim"],
            paths["iter_00_log"],
        )
    )

    _, iter_00_record = load_selected_record(paths["iter_00_smpl"], args.track_id)
    iter_00_ik = parse_ik_log_metrics(paths["iter_00_log"])
    feedback = build_feedback_weights(
        iter_00_ik,
        num_record_frames(iter_00_record),
        rms_threshold=args.rms_threshold,
    )
    np.save(paths["weights"], feedback["weights"])
    feedback_report = {key: value for key, value in feedback.items() if key != "weights"}
    write_json(paths["feedback_report"], to_jsonable(feedback_report))

    run(build_lower_body_cmd(args, input_pkl, paths["iter_01_dir"], paths["weights"]))
    copy_selected(paths["iter_01_dir"] / "corrected_smpl.pkl", paths["iter_01_smpl"])
    run(
        build_retarget_cmd(
            args,
            paths["iter_01_smpl"],
            paths["iter_01_opensim"],
            paths["iter_01_log"],
        )
    )

    baseline = build_summary(
        paths["iter_00_smpl"],
        paths["iter_00_dir"],
        paths["iter_00_log"],
        args.track_id,
    )
    candidate = build_summary(
        paths["iter_01_smpl"],
        paths["iter_01_dir"],
        paths["iter_01_log"],
        args.track_id,
    )
    selection = select_stage6_result(
        baseline,
        candidate,
        root_y_budget=args.max_root_y_shift,
    )
    selection["feedback"] = to_jsonable(feedback_report)
    selected_source = (
        paths["iter_01_smpl"] if selection["selected"] == "iter_01" else paths["iter_00_smpl"]
    )
    copy_selected(selected_source, paths["selected_smpl"])
    write_json(paths["selection_report"], to_jsonable(selection))

    print(f"selected: {selection['selected']}")
    print(f"selected_corrected_smpl: {paths['selected_smpl']}")
    print(f"stage6_selection_report: {paths['selection_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
