"""Run WHAM SMPL ablations for fixed beta, ground alignment, and lower-body optimization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.lower_body import lower_body_pose_mask
from lib.smpl_optimization.metrics import (
    beta_variation_max_abs,
    contact_foot_sliding,
    foot_penetration_depth,
    pose_delta_max_abs,
    root_vertical_jitter,
)
from lib.smpl_optimization.reports import to_jsonable
from lib.world_grounded.foot_points import get_record_contact, get_record_foot_points
from lib.world_grounded.ground import estimate_ground
from lib.world_grounded.tracks import select_track


STAGES = [
    {
        "id": "00_raw_wham",
        "description": "Original WHAM output.",
    },
    {
        "id": "01_fixed_beta",
        "description": "WHAM output after fixed-beta canonicalization.",
    },
    {
        "id": "02_world_grounded",
        "description": "Fixed-beta output aligned to the estimated ground plane.",
    },
    {
        "id": "03_root_y_only",
        "description": "World-grounded output with deterministic root-y foot penetration correction.",
    },
    {
        "id": "04_lower_body_full",
        "description": "World-grounded output with root-y and lower-body pose optimization.",
    },
    {
        "id": "05_opensim_feedback",
        "description": "OpenSim feedback loop selected corrected SMPL.",
    },
]


def active_stages(*, run_opensim_feedback_loop: bool, run_whole_body_smooth: bool) -> list[dict[str, str]]:
    stages = [stage for stage in STAGES if stage["id"] != "05_opensim_feedback" or run_opensim_feedback_loop]
    if run_whole_body_smooth:
        stages.append(
            {
                "id": f"{len(stages):02d}_whole_body_smooth",
                "description": "Conservative whole-body smoothness selected SMPL.",
            }
        )
    return stages


METRIC_FIELDS = [
    "stage_id",
    "description",
    "pkl",
    "frames",
    "beta_variation_max_abs",
    "estimated_ground_y",
    "ground_confidence",
    "min_foot_y",
    "max_foot_y",
    "penetration0_max",
    "penetration0_mean",
    "penetration0_count",
    "penetration_estimated_max",
    "penetration_estimated_mean",
    "penetration_estimated_count",
    "contact_mean_speed",
    "contact_max_speed",
    "contact_num_sliding",
    "root_rms_vertical_accel",
    "root_max_vertical_accel",
    "pose_delta_total_max_abs",
    "pose_delta_upper_max_abs",
    "pose_delta_lower_max_abs",
    "retarget_dir",
    "opensim_motion",
]


def safe_ascii_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned or 'sequence'}_{digest}"


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print()
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def build_demo_cmd(args: argparse.Namespace, video: Path, output_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        "demo.py",
        "--video",
        video.as_posix(),
        "--output_pth",
        str(output_root),
        "--save_pkl",
        "--pose-backend",
        args.pose_backend,
    ]
    if args.max_frames is not None:
        cmd += ["--max-frames", str(args.max_frames)]
    if args.run_smplify:
        cmd.append("--run_smplify")
    if args.fast:
        cmd.append("--fast")
    return cmd


def build_fixed_beta_cmd(
    args: argparse.Namespace,
    raw_pkl: Path,
    fixed_pkl: Path,
    wham_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/canonicalize_wham_fixed_beta.py",
        "--wham-pkl",
        str(raw_pkl),
        "--out-pkl",
        str(fixed_pkl),
        "--report",
        str(wham_dir / "fixed_beta_report.json"),
        "--beta-out",
        str(wham_dir / "beta_fixed.npy"),
        "--tracking-results",
        str(wham_dir / "tracking_results.pth"),
        "--device",
        args.device,
        "--chunk-size",
        str(args.chunk_size),
        "--keep-percentile",
        str(args.keep_percentile),
    ]
    return cmd


def build_world_grounded_cmd(args: argparse.Namespace, fixed_pkl: Path, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/world_grounded_smpl_optimizer.py",
        "--input-pkl",
        str(fixed_pkl),
        "--out-dir",
        str(out_dir),
        "--fps",
        str(args.fps),
        "--track-id",
        str(args.track_id),
        "--device",
        args.device,
    ]


def build_lower_body_cmd(
    args: argparse.Namespace,
    input_pkl: Path,
    out_dir: Path,
    *,
    enable_pose_pass: bool,
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
        str(args.lower_body_max_root_y_shift),
        "--device",
        args.device,
    ]
    if enable_pose_pass:
        cmd += ["--enable-pose-pass", "--pose-iterations", str(args.lower_body_pose_iterations)]
    return cmd


def build_retarget_cmd(args: argparse.Namespace, input_pkl: Path, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/retarget_smpl_to_opensim.py",
        str(input_pkl),
        "--config",
        args.retarget_config,
        "--out-dir",
        str(out_dir),
        "--fps",
        str(args.fps),
        "--device",
        args.device,
        "--chunk-size",
        str(args.chunk_size),
        "--alignment",
        args.alignment,
        "--axis-conversion",
        args.axis_conversion,
        "--root-calibration",
        args.root_calibration,
        "--root-calib-frames",
        str(args.root_calib_frames),
        "--track-id",
        str(args.track_id),
        "--ik-accuracy",
        str(args.ik_accuracy),
    ]
    if args.opensim_cmd:
        cmd += ["--opensim-cmd", args.opensim_cmd]
    if args.free_root:
        cmd.append("--free-root")
    if args.run_ik:
        cmd.append("--run-ik")
    return cmd


def build_opensim_feedback_cmd(args: argparse.Namespace, input_pkl: Path, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/opensim_feedback_loop.py",
        "--input-pkl",
        str(input_pkl),
        "--out-dir",
        str(out_dir),
        "--fps",
        str(args.fps),
        "--track-id",
        str(args.track_id),
        "--device",
        str(args.device),
        "--chunk-size",
        str(args.chunk_size),
        "--max-root-y-shift",
        str(args.lower_body_max_root_y_shift),
        "--pose-iterations",
        str(args.lower_body_pose_iterations),
        "--retarget-config",
        str(args.retarget_config),
        "--alignment",
        str(args.alignment),
        "--axis-conversion",
        str(args.axis_conversion),
        "--root-calibration",
        str(args.root_calibration),
        "--root-calib-frames",
        str(args.root_calib_frames),
        "--ik-accuracy",
        str(args.ik_accuracy),
    ]
    if args.opensim_cmd:
        cmd += ["--opensim-cmd", str(args.opensim_cmd)]
    if args.free_root:
        cmd.append("--free-root")
    else:
        cmd.append("--fix-root")
    return cmd


def build_whole_body_smooth_cmd(args: argparse.Namespace, input_pkl: Path, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/whole_body_smoothness_optimizer.py",
        "--input-pkl",
        str(input_pkl),
        "--out-dir",
        str(out_dir),
        "--fps",
        str(args.fps),
        "--track-id",
        str(args.track_id),
        "--device",
        args.device,
        "--chunk-size",
        str(args.chunk_size),
        "--retarget-config",
        args.retarget_config,
        "--alignment",
        args.alignment,
        "--axis-conversion",
        args.axis_conversion,
        "--root-calibration",
        args.root_calibration,
        "--root-calib-frames",
        str(args.root_calib_frames),
        "--ik-accuracy",
        str(args.ik_accuracy),
    ]
    if args.opensim_cmd:
        cmd += ["--opensim-cmd", args.opensim_cmd]
    if args.free_root:
        cmd.append("--free-root")
    return cmd


def frame_count(record: dict) -> int:
    for key in ("trans_world", "pose_world", "pose", "betas", "feet_refined", "feet_world", "feet", "verts"):
        value = record.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim > 0:
            return int(arr.shape[0])
    return 0


def select_record(pkl: Path, track_id: str) -> tuple[str, dict]:
    data = joblib.load(pkl)
    selected_track_id, record = select_track(data, track_id)
    return str(selected_track_id), record


def pose_delta_report(baseline: dict | None, record: dict) -> dict[str, float]:
    if baseline is None:
        return {
            "pose_delta_total_max_abs": 0.0,
            "pose_delta_upper_max_abs": 0.0,
            "pose_delta_lower_max_abs": 0.0,
        }
    pose_key = "pose_world" if "pose_world" in baseline and "pose_world" in record else "pose"
    original = np.asarray(baseline.get(pose_key, np.asarray([])))
    corrected = np.asarray(record.get(pose_key, np.asarray([])))
    total = pose_delta_max_abs(original, corrected)
    upper = total
    lower = total
    if original.ndim == 2 and corrected.ndim == 2 and original.shape == corrected.shape and original.shape[1]:
        mask = lower_body_pose_mask(original.shape[1])
        upper_mask = ~mask
        if np.any(mask):
            lower = float(np.max(np.abs(corrected[:, mask] - original[:, mask])))
        if np.any(upper_mask):
            upper = float(np.max(np.abs(corrected[:, upper_mask] - original[:, upper_mask])))
    return {
        "pose_delta_total_max_abs": float(round(float(total), 8)),
        "pose_delta_upper_max_abs": float(round(float(upper), 8)),
        "pose_delta_lower_max_abs": float(round(float(lower), 8)),
    }


def compute_metrics(
    stage: dict[str, str],
    pkl: Path,
    *,
    fps: float,
    track_id: str,
    baseline_record: dict | None,
    retarget_dir: Path | None = None,
) -> dict[str, Any]:
    selected_track_id, record = select_record(pkl, track_id)
    n_frames = frame_count(record)
    row: dict[str, Any] = {
        "stage_id": stage["id"],
        "description": stage["description"],
        "track_id": selected_track_id,
        "pkl": str(pkl),
        "frames": n_frames,
        "retarget_dir": str(retarget_dir) if retarget_dir else "",
        "opensim_motion": str(retarget_dir / "opensim_ik.mot") if retarget_dir else "",
    }
    row["beta_variation_max_abs"] = beta_variation_max_abs(record.get("betas", np.asarray([])))

    trans_world = np.asarray(record.get("trans_world", np.asarray([])))
    root = root_vertical_jitter(trans_world)
    row["root_rms_vertical_accel"] = root["rms_vertical_accel"]
    row["root_max_vertical_accel"] = root["max_vertical_accel"]
    row.update(pose_delta_report(baseline_record, record))

    try:
        foot = get_record_foot_points(record)
        points = foot.points[:n_frames]
        contact = get_record_contact(record, points.shape[0], points.shape[1])
        if contact is None:
            contact = np.zeros(points.shape[:2], dtype=np.float32)
        ground = estimate_ground(points, fps=fps, wham_contact=contact)
        penetration0 = foot_penetration_depth(points[:, :, 1], ground_y=0.0)
        penetration_estimated = foot_penetration_depth(points[:, :, 1], ground_y=ground.ground_y)
        sliding = contact_foot_sliding(points, contact, fps=fps)
        row.update(
            {
                "foot_source": foot.source,
                "estimated_ground_y": float(round(float(ground.ground_y), 8)),
                "ground_confidence": ground.ground_confidence,
                "min_foot_y": float(round(float(np.min(points[:, :, 1])), 8)),
                "max_foot_y": float(round(float(np.max(points[:, :, 1])), 8)),
                "penetration0_max": penetration0["max_penetration"],
                "penetration0_mean": penetration0["mean_penetration"],
                "penetration0_count": penetration0["num_penetrating"],
                "penetration_estimated_max": penetration_estimated["max_penetration"],
                "penetration_estimated_mean": penetration_estimated["mean_penetration"],
                "penetration_estimated_count": penetration_estimated["num_penetrating"],
                "contact_mean_speed": sliding["mean_contact_speed"],
                "contact_max_speed": sliding["max_contact_speed"],
                "contact_num_sliding": sliding["num_sliding"],
            }
        )
    except Exception as exc:
        row.update(
            {
                "foot_source": "",
                "estimated_ground_y": "",
                "ground_confidence": f"unavailable: {exc}",
                "min_foot_y": "",
                "max_foot_y": "",
                "penetration0_max": "",
                "penetration0_mean": "",
                "penetration0_count": "",
                "penetration_estimated_max": "",
                "penetration_estimated_mean": "",
                "penetration_estimated_count": "",
                "contact_mean_speed": "",
                "contact_max_speed": "",
                "contact_num_sliding": "",
            }
        )
    return row


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(METRIC_FIELDS)
    extras = sorted({key for row in rows for key in row if key not in fieldnames})
    fieldnames.extend(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-pth", default="output/ablation")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--track-id", default="merge")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--pose-backend", choices=["vitpose", "rtmpose"], default="rtmpose")
    parser.add_argument("--run-smplify", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--skip-wham", action="store_true")
    parser.add_argument("--keep-percentile", type=float, default=80.0)
    parser.add_argument("--lower-body-max-root-y-shift", type=float, default=0.25)
    parser.add_argument("--lower-body-pose-iterations", type=int, default=80)
    parser.add_argument("--retarget-config", default="configs/retarget/smpl_to_mimicmsk_opensim.yaml")
    parser.add_argument("--alignment", choices=["mapping", "anatomical"], default="anatomical")
    parser.add_argument(
        "--axis-conversion",
        choices=["mujoco_to_opensim", "opensim_to_mujoco", "identity"],
        default="mujoco_to_opensim",
    )
    parser.add_argument("--root-calibration", choices=["none", "constant_from_free_ik"], default="constant_from_free_ik")
    parser.add_argument("--root-calib-frames", type=int, default=10)
    parser.add_argument("--opensim-cmd", default=None)
    parser.add_argument("--ik-accuracy", type=float, default=1e-4)
    parser.add_argument("--free-root", action="store_true")
    parser.add_argument("--run-ik", action="store_true")
    parser.add_argument("--skip-retarget", action="store_true")
    parser.add_argument("--run-opensim-feedback-loop", action="store_true")
    parser.add_argument("--run-whole-body-smooth", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = Path(args.video).resolve()
    ensure_file(video, "input video")
    output_root = Path(args.output_pth).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sequence = video.stem
    wham_dir = output_root / sequence
    ablation_root = output_root / "_ablation" / safe_ascii_name(sequence)
    ablation_root.mkdir(parents=True, exist_ok=True)

    raw_pkl = wham_dir / "wham_output.pkl"
    fixed_pkl = wham_dir / "canonical_wham_output.pkl"
    world_dir = ablation_root / "02_world_grounded"
    world_pkl = world_dir / "optimized_canonical_wham_output.pkl"
    root_y_dir = ablation_root / "03_root_y_only"
    root_y_pkl = root_y_dir / "corrected_smpl.pkl"
    lower_body_dir = ablation_root / "04_lower_body_full"
    lower_body_pkl = lower_body_dir / "corrected_smpl.pkl"
    feedback_dir = ablation_root / "05_opensim_feedback"
    feedback_pkl = feedback_dir / "selected_corrected_smpl.pkl"

    if not args.skip_wham:
        run(build_demo_cmd(args, video, output_root))
    ensure_file(raw_pkl, "raw WHAM pkl")

    run(build_fixed_beta_cmd(args, raw_pkl, fixed_pkl, wham_dir))
    run(build_world_grounded_cmd(args, fixed_pkl, world_dir))
    run(build_lower_body_cmd(args, world_pkl, root_y_dir, enable_pose_pass=False))
    run(build_lower_body_cmd(args, world_pkl, lower_body_dir, enable_pose_pass=True))
    if args.run_opensim_feedback_loop:
        run(build_opensim_feedback_cmd(args, world_pkl, feedback_dir))

    stage_pkls = {
        "00_raw_wham": raw_pkl,
        "01_fixed_beta": fixed_pkl,
        "02_world_grounded": world_pkl,
        "03_root_y_only": root_y_pkl,
        "04_lower_body_full": lower_body_pkl,
    }
    if args.run_opensim_feedback_loop:
        stage_pkls["05_opensim_feedback"] = feedback_pkl

    if args.run_whole_body_smooth:
        smooth_stage_id = "06_whole_body_smooth" if args.run_opensim_feedback_loop else "05_whole_body_smooth"
        smooth_dir = ablation_root / smooth_stage_id
        smooth_input_pkl = feedback_pkl if args.run_opensim_feedback_loop else lower_body_pkl
        run(build_whole_body_smooth_cmd(args, smooth_input_pkl, smooth_dir))
        stage_pkls[smooth_stage_id] = smooth_dir / "selected_smooth_smpl.pkl"

    stages = active_stages(
        run_opensim_feedback_loop=args.run_opensim_feedback_loop,
        run_whole_body_smooth=args.run_whole_body_smooth,
    )

    retarget_dirs: dict[str, Path] = {}
    if not args.skip_retarget:
        for stage in stages:
            stage_id = stage["id"]
            retarget_dir = ablation_root / stage_id / "opensim_retarget"
            retarget_dirs[stage_id] = retarget_dir
            run(build_retarget_cmd(args, stage_pkls[stage_id], retarget_dir))

    _, baseline_record = select_record(raw_pkl, args.track_id)
    rows = []
    for stage in stages:
        stage_id = stage["id"]
        rows.append(
            compute_metrics(
                stage,
                stage_pkls[stage_id],
                fps=args.fps,
                track_id=args.track_id,
                baseline_record=baseline_record,
                retarget_dir=retarget_dirs.get(stage_id),
            )
        )

    summary = {
        "video": str(video),
        "output_root": str(output_root),
        "ablation_root": str(ablation_root),
        "stages": stages,
        "metrics": rows,
    }
    metrics_csv = ablation_root / "ablation_metrics.csv"
    summary_json = ablation_root / "ablation_summary.json"
    write_metrics_csv(metrics_csv, rows)
    write_json(summary_json, summary)

    print()
    print("Ablation complete")
    print(f"ablation_root: {ablation_root}")
    print(f"metrics_csv: {metrics_csv}")
    print(f"summary_json: {summary_json}")
    for row in rows:
        print(
            f"{row['stage_id']}: beta_var={row['beta_variation_max_abs']} "
            f"penetration0_max={row['penetration0_max']} "
            f"contact_mean_speed={row['contact_mean_speed']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
