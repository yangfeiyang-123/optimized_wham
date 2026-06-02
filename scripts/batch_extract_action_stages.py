"""Batch extract action SMPL stages into stage-specific folders."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.world_grounded.tracks import select_track
from scripts.run_wham_ablation import safe_ascii_name


STAGE_DIRS = {
    "stage1_raw": ("video", "wham_output.pkl", True),
    "stage2_fixed_beta": ("video", "canonical_wham_output.pkl", True),
    "stage3_world_grounded": ("ablation", "02_world_grounded/optimized_canonical_wham_output.pkl", False),
    "stage4_root_y_only": ("ablation", "03_root_y_only/corrected_smpl.pkl", False),
    "stage5_lower_body_full": ("ablation", "04_lower_body_full/corrected_smpl.pkl", False),
    "stage6_opensim_feedback": ("ablation", "05_opensim_feedback/selected_corrected_smpl.pkl", False),
}


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print()
    print("$ " + " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=check)


def frame_count(record: dict) -> int:
    for key in ("frame_id", "frame_ids", "pose_world", "pose", "trans_world", "trans"):
        if key in record:
            return len(record[key])
    return 0


def write_single_track(src: Path, dst: Path, *, merge: bool) -> tuple[str, int]:
    data = joblib.load(src)
    if merge:
        track_id, record = select_track(data, "merge")
        data = {track_id: record}
    else:
        track_id, record = select_track(data, "merge")
        data = {track_id: record}
    dst.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, dst)
    return str(track_id), frame_count(record)


def stage_sources(video: Path, output_root: Path) -> dict[str, Path]:
    seq = video.stem
    wham_dir = output_root / seq
    ablation_dir = output_root / "_ablation" / safe_ascii_name(seq)
    sources = {}
    for stage_name, (source_kind, rel_path, _merge) in STAGE_DIRS.items():
        source_root = wham_dir if source_kind == "video" else ablation_dir
        sources[stage_name] = source_root / rel_path
    return sources


def write_fallback_stage6(video: Path, output_root: Path, *, exit_code: int) -> None:
    sources = stage_sources(video, output_root)
    required = [
        "stage1_raw",
        "stage2_fixed_beta",
        "stage3_world_grounded",
        "stage4_root_y_only",
        "stage5_lower_body_full",
    ]
    missing = [stage for stage in required if not sources[stage].exists()]
    if missing:
        missing_text = ", ".join(f"{stage}: {sources[stage]}" for stage in missing)
        raise RuntimeError(f"Cannot fallback Stage6 because required sources are missing: {missing_text}")

    stage6_src = sources["stage6_opensim_feedback"]
    if stage6_src.exists():
        return

    stage6_src.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sources["stage5_lower_body_full"], stage6_src)
    report = {
        "stage": "stage6_opensim_feedback",
        "status": "fallback_to_stage5",
        "reason": "OpenSim IK feedback failed after Stage1-5 artifacts were generated; Stage5 lower-body full correction was copied as the Stage6 output so batch processing can continue.",
        "run_wham_ablation_exit_code": exit_code,
        "source_pkl": str(sources["stage5_lower_body_full"]),
        "output_pkl": str(stage6_src),
    }
    (stage6_src.parent / "stage6_opensim_feedback_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"stage6_opensim_feedback: fallback_to_stage5 {stage6_src.name}", flush=True)


def clean_output(output_root: Path, video_names: list[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in video_names:
        target = output_root / name
        if target.exists():
            shutil.rmtree(target)
    ablation = output_root / "_ablation"
    if ablation.exists():
        shutil.rmtree(ablation)
    for stage in [*STAGE_DIRS, "stage7_whole_body_smooth"]:
        stage_dir = output_root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        for pkl in stage_dir.glob("*.pkl"):
            pkl.unlink()
        for report in stage_dir.glob("*.json"):
            report.unlink()


def copy_stages(video: Path, output_root: Path) -> dict[str, int]:
    seq = video.stem
    sources = stage_sources(video, output_root)
    frames: dict[str, int] = {}

    for stage_name, (source_kind, rel_path, merge) in STAGE_DIRS.items():
        src = sources[stage_name]
        dst = output_root / stage_name / f"{seq}_{stage_name}.pkl"
        track_id, n_frames = write_single_track(src, dst, merge=merge)
        frames[stage_name] = n_frames
        print(f"{stage_name}: {dst.name} track={track_id} frames={n_frames}", flush=True)

        if stage_name == "stage6_opensim_feedback":
            fallback_report = src.parent / "stage6_opensim_feedback_report.json"
            if fallback_report.exists():
                shutil.copy2(
                    fallback_report,
                    output_root / stage_name / f"{seq}_stage6_opensim_feedback_report.json",
                )

    stage6 = output_root / "stage6_opensim_feedback" / f"{seq}_stage6_opensim_feedback.pkl"
    stage7 = output_root / "stage7_whole_body_smooth" / f"{seq}_stage7_whole_body_smooth.pkl"
    shutil.copy2(stage6, stage7)
    frames["stage7_whole_body_smooth"] = frames["stage6_opensim_feedback"]
    report = {
        "stage": "stage7_whole_body_smooth",
        "status": "conservative_fallback",
        "reason": "No runnable Stage7 whole-body smoothness optimizer is present in this checkout; selected Stage6 SMPL was copied unchanged.",
        "source_pkl": str(stage6),
        "output_pkl": str(stage7),
        "frames": frames["stage7_whole_body_smooth"],
    }
    (output_root / "stage7_whole_body_smooth" / f"{seq}_stage7_whole_body_smooth_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"stage7_whole_body_smooth: {stage7.name} frames={frames['stage7_whole_body_smooth']}", flush=True)
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-pth", required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pose-backend", choices=["vitpose", "rtmpose"], default="rtmpose")
    parser.add_argument("--opensim-cmd", default=r"C:\Users\yangfeiyang\Downloads\OpenSim 4.5\bin\opensim-cmd.exe")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_pth).resolve()
    videos = sorted(input_dir.glob("*.mp4"), key=lambda p: int(p.stem.replace("video", "")))
    videos = [video for video in videos if int(video.stem.replace("video", "")) >= args.start_index]
    if args.end_index is not None:
        videos = [video for video in videos if int(video.stem.replace("video", "")) <= args.end_index]
    if args.clean:
        clean_output(output_root, [video.stem for video in sorted(input_dir.glob("*.mp4"))])

    for video in videos:
        print(f"\n=== {video.stem} ===", flush=True)
        result = run(
            [
                sys.executable,
                "scripts/run_wham_ablation.py",
                "--video",
                str(video),
                "--output-pth",
                str(output_root),
                "--fps",
                str(args.fps),
                "--device",
                args.device,
                "--pose-backend",
                args.pose_backend,
                "--skip-retarget",
                "--run-opensim-feedback-loop",
                "--opensim-cmd",
                args.opensim_cmd,
                "--free-root",
            ],
            check=False,
        )
        if result.returncode != 0:
            print(
                f"run_wham_ablation failed for {video.stem} with exit code {result.returncode}; attempting Stage6 fallback if Stage1-5 are complete.",
                flush=True,
            )
            write_fallback_stage6(video, output_root, exit_code=result.returncode)
        copy_stages(video, output_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
