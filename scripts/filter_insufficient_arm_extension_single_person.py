from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np


REPO = Path("/data3/yangfeiyang/WorkSpace/optimized_wham")
DEFAULT_OUT = REPO / "output/forehand_clear/InsufficientArmExtension"
DEFAULT_VIDEO_DIR = REPO / "BadmintonVideos/forehand_clear/wrong"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def frame_count(record: dict[str, Any]) -> int:
    for key in ("frame_ids", "frame_id", "pose", "pose_world", "betas"):
        if key in record:
            arr = to_numpy(record[key])
            if arr.ndim > 0:
                return int(arr.shape[0])
    return 0


def track_metrics(tracking: dict[Any, dict[str, Any]]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for track_id, record in tracking.items():
        bbox = np.asarray(record["bbox"], dtype=np.float64)
        side_px = bbox[:, 2] * 200.0
        metrics[str(track_id)] = {
            "frames": float(len(bbox)),
            "mean_cx": float(np.mean(bbox[:, 0])),
            "mean_cy": float(np.mean(bbox[:, 1])),
            "mean_side_px": float(np.mean(side_px)),
            "median_side_px": float(np.median(side_px)),
        }
    return metrics


def choose_main_track(metrics: dict[str, dict[str, float]], min_size_ratio: float) -> tuple[str, bool, float]:
    ordered = sorted(
        metrics.items(),
        key=lambda item: (item[1]["median_side_px"], item[1]["mean_side_px"]),
        reverse=True,
    )
    selected = ordered[0][0]
    if len(ordered) == 1:
        return selected, False, float("inf")
    ratio = ordered[0][1]["median_side_px"] / max(ordered[1][1]["median_side_px"], 1e-6)
    return selected, ratio < min_size_ratio, float(ratio)


def resolve_key(mapping: dict[Any, Any], selected: str) -> Any:
    for key in mapping:
        if str(key) == selected:
            return key
    raise KeyError(f"Track {selected!r} not found in {list(mapping.keys())}")


def backup_once(src: Path, backup_dir: Path) -> None:
    if not src.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = backup_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)


def write_source_json(path: Path, video: Path, seq_dir: Path, selected_track: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "video": str(video),
                "selected_person_track_id": selected_track,
                "selection_rule": "largest median bbox side in tracking_results.pth",
                "wham_pkl": str(seq_dir / "wham_output.pkl"),
                "canonical_pkl": str(seq_dir / "canonical_wham_output.pkl"),
                "world_grounded_pkl": str(seq_dir / "world_grounded/optimized_canonical_wham_output.pkl"),
                "corrected_smpl_pkl": str(seq_dir / "lower_body_corrected/corrected_smpl.pkl"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def run_stage(status_writer, log, seq_name: str, stage: str, cmd: list[str], cwd: Path) -> None:
    status_writer(seq_name, "START", stage)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp()}] start {stage}\n")
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=f, check=True)
        f.write(f"[{timestamp()}] done {stage}\n")
    status_writer(seq_name, "DONE", stage)


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter InsufficientArmExtension WHAM outputs to one standing player.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--pose-iterations", type=int, default=80)
    parser.add_argument("--min-size-ratio", type=float, default=1.3)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    video_dir = args.video_dir.resolve()
    status_path = out_dir / "person_filter_status.tsv"
    selection_path = out_dir / "person_filter_selection.csv"
    report_path = out_dir / "person_filter_selection.json"
    log_root = out_dir / "person_filter_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    selections: list[dict[str, Any]] = []
    status_file = status_path.open("w", newline="", encoding="utf-8")
    status_writer_csv = csv.writer(status_file, delimiter="\t")
    status_writer_csv.writerow(["timestamp", "video", "status", "stage"])
    status_file.flush()

    def write_status(video: str, status: str, stage: str) -> None:
        status_writer_csv.writerow([timestamp(), video, status, stage])
        status_file.flush()
        print(f"[{timestamp()}] {video} {status} {stage}", flush=True)

    try:
        seq_dirs = sorted(p for p in out_dir.iterdir() if p.is_dir() and p.name not in {"person_filter_previews", "person_filter_logs"})
        for seq_dir in seq_dirs:
            tracking_path = seq_dir / "tracking_results.pth"
            wham_path = seq_dir / "wham_output.pkl"
            if not tracking_path.exists() or not wham_path.exists():
                continue

            seq_name = seq_dir.name
            tracking = joblib.load(tracking_path)
            wham = joblib.load(wham_path)
            if len(tracking) <= 1 and len(wham) <= 1:
                write_status(seq_name, "SKIP", "already_single_person")
                continue

            metrics = track_metrics(tracking)
            selected_track, uncertain, ratio = choose_main_track(metrics, args.min_size_ratio)
            selections.append(
                {
                    "video": f"{seq_name}.mp4",
                    "selected_track_id": selected_track,
                    "uncertain": uncertain,
                    "size_ratio": ratio,
                    "metrics": metrics,
                }
            )
            if uncertain:
                write_status(seq_name, "UNCERTAIN", f"size_ratio={ratio:.3f}")
                continue

            write_status(seq_name, "SELECT", f"track={selected_track},size_ratio={ratio:.3f}")
            backup_dir = seq_dir / "person_filter_backup"
            backup_once(tracking_path, backup_dir)
            backup_once(wham_path, backup_dir)
            backup_once(seq_dir / "canonical_wham_output.pkl", backup_dir)
            backup_once(seq_dir / "fixed_beta_report.json", backup_dir)
            backup_once(seq_dir / "beta_fixed.npy", backup_dir)
            backup_once(seq_dir / "world_grounded/optimized_canonical_wham_output.pkl", backup_dir / "world_grounded")
            backup_once(seq_dir / "lower_body_corrected/corrected_smpl.pkl", backup_dir / "lower_body_corrected")

            selected_key_tracking = resolve_key(tracking, selected_track)
            selected_key_wham = resolve_key(wham, selected_track)
            selected_tracking = {selected_key_tracking: tracking[selected_key_tracking]}
            selected_wham = {selected_key_wham: wham[selected_key_wham]}
            joblib.dump(selected_tracking, tracking_path)
            joblib.dump(selected_wham, wham_path)

            selection_json = seq_dir / "person_filter_selection.json"
            selection_json.write_text(
                json.dumps(selections[-1], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log = log_root / f"{seq_name}.log"
            log.write_text(f"[{timestamp()}] selected track {selected_track}; size_ratio={ratio:.3f}\n", encoding="utf-8")

            canonical_pkl = seq_dir / "canonical_wham_output.pkl"
            fixed_report = seq_dir / "fixed_beta_report.json"
            world_dir = seq_dir / "world_grounded"
            world_pkl = world_dir / "optimized_canonical_wham_output.pkl"
            lower_dir = seq_dir / "lower_body_corrected"
            corrected_pkl = lower_dir / "corrected_smpl.pkl"
            reference_dir = seq_dir / "reference_bundle"
            source_json = reference_dir / "source.json"
            video = video_dir / f"{seq_name}.mp4"

            run_stage(
                write_status,
                log,
                seq_name,
                "fixed_beta",
                [
                    sys.executable,
                    "scripts/canonicalize_wham_fixed_beta.py",
                    "--wham-pkl",
                    str(wham_path),
                    "--out-pkl",
                    str(canonical_pkl),
                    "--report",
                    str(fixed_report),
                    "--beta-out",
                    str(seq_dir / "beta_fixed.npy"),
                    "--tracking-results",
                    str(tracking_path),
                    "--device",
                    args.device,
                    "--chunk-size",
                    str(args.chunk_size),
                    "--keep-percentile",
                    "80",
                ],
                REPO,
            )
            run_stage(
                write_status,
                log,
                seq_name,
                "world_grounded",
                [
                    sys.executable,
                    "scripts/world_grounded_smpl_optimizer.py",
                    "--input-pkl",
                    str(canonical_pkl),
                    "--out-dir",
                    str(world_dir),
                    "--fps",
                    str(args.fps),
                    "--track-id",
                    "merge",
                    "--device",
                    args.device,
                    "--root-smooth-axes",
                    "y",
                ],
                REPO,
            )
            run_stage(
                write_status,
                log,
                seq_name,
                "lower_body_corrected",
                [
                    sys.executable,
                    "scripts/optimize_smpl_lower_body.py",
                    "--input-pkl",
                    str(world_pkl),
                    "--out-dir",
                    str(lower_dir),
                    "--fps",
                    str(args.fps),
                    "--track-id",
                    "merge",
                    "--max-root-y-shift",
                    "0.25",
                    "--device",
                    args.device,
                    "--enable-pose-pass",
                    "--pose-iterations",
                    str(args.pose_iterations),
                ],
                REPO,
            )
            write_source_json(source_json, video, seq_dir, selected_track)
            run_stage(
                write_status,
                log,
                seq_name,
                "reference_bundle",
                [
                    sys.executable,
                    "scripts/export_contact_preserving_reference.py",
                    "--input-pkl",
                    str(corrected_pkl),
                    "--out-dir",
                    str(reference_dir),
                    "--sequence",
                    seq_name,
                    "--fps",
                    str(args.fps),
                    "--track-id",
                    "merge",
                    "--quality-report",
                    str(lower_dir / "validation_summary.json"),
                    "--source-json",
                    str(source_json),
                    "--stance-enter-threshold",
                    "0.55",
                    "--stance-exit-threshold",
                    "0.30",
                    "--root-smooth-axes",
                    "y",
                ],
                REPO,
            )
            write_status(seq_name, "DONE", "single_person_filter")
    finally:
        status_file.close()

    selection_path.write_text(
        "video,selected_track_id,uncertain,size_ratio,metrics\n"
        + "\n".join(
            f"{row['video']},{row['selected_track_id']},{row['uncertain']},{row['size_ratio']:.6f},\"{row['metrics']}\""
            for row in selections
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8")

    uncertain_rows = [row for row in selections if row["uncertain"]]
    if uncertain_rows:
        print("Uncertain selections need manual review:", flush=True)
        for row in uncertain_rows:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        return 2
    print(f"[{timestamp()}] filter complete; selections={len(selections)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
