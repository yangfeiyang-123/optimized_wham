"""Run video -> WHAM SMPL -> fixed-beta SMPL -> OpenSim IK retarget.

This is the top-level pipeline for the current WorldSMPLGen workflow. It
intentionally retargets only canonical_wham_output.pkl, so the OpenSim stage
never consumes WHAM's frame-varying raw beta sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENSIM_CMD = r"C:\Users\yangfeiyang\Downloads\OpenSim 4.5\bin\opensim-cmd.exe"


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def video_sequence_name(video: Path) -> str:
    # Match demo.py's split('/') + split('.') behavior while avoiding Windows
    # backslash issues by passing the video path to demo.py in POSIX form.
    return ".".join(video.name.split(".")[:-1]) or video.stem


def is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def safe_ascii_name(value: str) -> str:
    cleaned = "".join(ch if ord(ch) < 128 else "_" for ch in value)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned).strip("._-")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned or 'sequence'}_{digest}"


def default_retarget_out_dir(output_root: Path, wham_dir: Path, sequence: str) -> Path:
    candidate = wham_dir / "opensim_retarget_fixed_beta"
    if is_ascii_path(candidate):
        return candidate

    safe_dir = output_root / "_opensim_retarget_fixed_beta" / safe_ascii_name(sequence)
    if is_ascii_path(safe_dir):
        return safe_dir

    return (
        REPO_ROOT
        / "output"
        / "_opensim_retarget_fixed_beta"
        / safe_ascii_name(sequence)
    )


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_fixed_beta_report(report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tracks = report.get("tracks", {})
    max_after = 0.0
    for track_report in tracks.values():
        max_after = max(
            max_after, float(track_report.get("beta_variation_after_max_abs", 0.0))
        )
    report["pipeline_beta_variation_after_max_abs"] = max_after
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--output-pth",
        default="output/demo",
        help="WHAM output root. demo.py creates one subdir per video.",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="FPS for generated TRC/MOT files."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for fixed-beta and retarget SMPL forward.",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--track-id",
        default="merge",
        help="Track id to retarget. Default 'merge' stitches WHAM ID switches into one timeline.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Debug limit passed to retarget only.",
    )

    parser.add_argument(
        "--calib", default=None, help="Optional WHAM camera calibration file."
    )
    parser.add_argument("--estimate-local-only", action="store_true")
    parser.add_argument("--force-global", action="store_true")
    parser.add_argument("--visualize-wham", action="store_true")
    parser.add_argument("--run-smplify", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--pose-backend", choices=["vitpose", "rtmpose"], default="rtmpose"
    )
    parser.add_argument("--pose-model", default=None)
    parser.add_argument("--detector-ckpt", default=None)
    parser.add_argument("--detector-imgsz", type=int, default=None)
    parser.add_argument("--detector-scale", type=float, default=None)
    parser.add_argument("--feature-batch-size", type=int, default=None)
    parser.add_argument("--min-track-frames", type=int, default=None)

    parser.add_argument(
        "--keep-percentile",
        type=float,
        default=80.0,
        help="Beta samples kept by canonicalizer.",
    )
    parser.add_argument(
        "--retarget-config", default="configs/retarget/smpl_to_mimicmsk_opensim.yaml"
    )
    parser.add_argument("--retarget-out-dir", default=None)
    parser.add_argument(
        "--alignment", choices=["mapping", "anatomical"], default="anatomical"
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
    parser.add_argument(
        "--free-root",
        dest="free_root",
        action="store_true",
        default=True,
        help="Diagnostic retarget mode; disables fixed root.",
    )
    parser.add_argument(
        "--fix-root",
        dest="free_root",
        action="store_false",
        help="Force OpenSim root coordinates to follow the WHAM-derived root MOT.",
    )
    parser.add_argument(
        "--skip-ik",
        action="store_true",
        help="Write retarget assets but do not run OpenSim IK.",
    )
    parser.add_argument("--opensim-cmd", default=DEFAULT_OPENSIM_CMD)
    parser.add_argument(
        "--skip-wham",
        action="store_true",
        help="Reuse an existing wham_output.pkl in the video output dir.",
    )
    parser.add_argument(
        "--skip-fixed-beta",
        action="store_true",
        help="Reuse an existing canonical_wham_output.pkl.",
    )
    parser.add_argument(
        "--world-grounded",
        action="store_true",
        help="Run SMPL-layer ground/contact optimizer before OpenSim retarget.",
    )
    parser.add_argument("--world-grounded-out-dir", default=None)
    parser.add_argument(
        "--optimize-lower-body",
        action="store_true",
        help="Run lower-body SMPL optimizer before OpenSim retarget.",
    )
    parser.add_argument("--lower-body-out-dir", default=None)
    parser.add_argument("--lower-body-max-root-y-shift", type=float, default=0.25)
    parser.add_argument("--disable-lower-body-pose-pass", action="store_true")
    parser.add_argument("--lower-body-pose-iterations", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = Path(args.video).resolve()
    require_file(video, "input video")

    output_root = Path(args.output_pth).resolve()
    sequence = video_sequence_name(video)
    wham_dir = output_root / sequence
    wham_pkl = wham_dir / "wham_output.pkl"
    canonical_pkl = wham_dir / "canonical_wham_output.pkl"
    fixed_beta_report = wham_dir / "fixed_beta_report.json"
    retarget_out = (
        Path(args.retarget_out_dir).resolve()
        if args.retarget_out_dir
        else default_retarget_out_dir(output_root, wham_dir, sequence)
    )
    world_grounded_out = (
        Path(args.world_grounded_out_dir).resolve()
        if args.world_grounded_out_dir
        else output_root / "_world_grounded" / safe_ascii_name(sequence)
    )
    lower_body_out = (
        Path(args.lower_body_out_dir).resolve()
        if args.lower_body_out_dir
        else output_root / "_lower_body_optimized" / safe_ascii_name(sequence)
    )

    if not args.skip_wham:
        wham_cmd = [
            sys.executable,
            "demo.py",
            "--video",
            video.as_posix(),
            "--output_pth",
            str(output_root),
            "--save_pkl",
        ]
        if args.calib:
            wham_cmd += ["--calib", str(Path(args.calib).resolve())]
        if args.estimate_local_only:
            wham_cmd.append("--estimate_local_only")
        if args.force_global:
            wham_cmd.append("--force_global")
        if args.visualize_wham:
            wham_cmd.append("--visualize")
        if args.run_smplify:
            wham_cmd.append("--run_smplify")
        if args.fast:
            wham_cmd.append("--fast")
        for cli_name, value in [
            ("--pose-backend", args.pose_backend),
            ("--pose-model", args.pose_model),
            ("--detector-ckpt", args.detector_ckpt),
            ("--detector-imgsz", args.detector_imgsz),
            ("--detector-scale", args.detector_scale),
            ("--feature-batch-size", args.feature_batch_size),
            ("--min-track-frames", args.min_track_frames),
        ]:
            if value is not None:
                wham_cmd += [cli_name, str(value)]
        run(wham_cmd)

    require_file(wham_pkl, "WHAM pkl")

    if not args.skip_fixed_beta:
        fixed_cmd = [
            sys.executable,
            "scripts/canonicalize_wham_fixed_beta.py",
            "--wham-pkl",
            str(wham_pkl),
            "--out-pkl",
            str(canonical_pkl),
            "--report",
            str(fixed_beta_report),
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
        run(fixed_cmd)

    require_file(canonical_pkl, "fixed-beta WHAM pkl")
    require_file(fixed_beta_report, "fixed-beta report")
    beta_report = load_fixed_beta_report(fixed_beta_report)
    if beta_report["pipeline_beta_variation_after_max_abs"] > 1e-7:
        raise RuntimeError(
            "Fixed beta validation failed: "
            f"max variation after canonicalization is {beta_report['pipeline_beta_variation_after_max_abs']}"
        )

    retarget_input_pkl = canonical_pkl
    if args.world_grounded:
        wg_cmd = [
            sys.executable,
            "scripts/world_grounded_smpl_optimizer.py",
            "--input-pkl",
            str(canonical_pkl),
            "--out-dir",
            str(world_grounded_out),
            "--fps",
            str(args.fps),
            "--track-id",
            str(args.track_id),
            "--device",
            args.device,
        ]
        run(wg_cmd)
        retarget_input_pkl = world_grounded_out / "optimized_canonical_wham_output.pkl"
        require_file(retarget_input_pkl, "world-grounded optimized pkl")

    if args.optimize_lower_body:
        lb_cmd = [
            sys.executable,
            "scripts/optimize_smpl_lower_body.py",
            "--input-pkl",
            str(retarget_input_pkl),
            "--out-dir",
            str(lower_body_out),
            "--fps",
            str(args.fps),
            "--track-id",
            str(args.track_id),
            "--max-root-y-shift",
            str(args.lower_body_max_root_y_shift),
            "--device",
            args.device,
        ]
        if not args.disable_lower_body_pose_pass:
            lb_cmd += [
                "--enable-pose-pass",
                "--pose-iterations",
                str(args.lower_body_pose_iterations),
            ]
        run(lb_cmd)
        retarget_input_pkl = lower_body_out / "corrected_smpl.pkl"
        require_file(retarget_input_pkl, "lower-body corrected SMPL pkl")

    retarget_cmd = [
        sys.executable,
        "scripts/retarget_smpl_to_opensim.py",
        str(retarget_input_pkl),
        "--config",
        args.retarget_config,
        "--out-dir",
        str(retarget_out),
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
        "--opensim-cmd",
        args.opensim_cmd,
    ]
    retarget_cmd += ["--track-id", str(args.track_id)]
    if args.max_frames is not None:
        retarget_cmd += ["--max-frames", str(args.max_frames)]
    if args.free_root:
        retarget_cmd.append("--free-root")
    if not args.skip_ik:
        retarget_cmd.append("--run-ik")
    run(retarget_cmd)

    print("\nPipeline complete")
    print(f"raw_wham_pkl: {wham_pkl}")
    print(f"fixed_beta_pkl: {canonical_pkl}")
    print(f"fixed_beta_report: {fixed_beta_report}")
    if args.world_grounded:
        print(f"world_grounded_dir: {world_grounded_out}")
        print(
            f"optimized_fixed_beta_pkl: {world_grounded_out / 'optimized_canonical_wham_output.pkl'}"
        )
    if args.optimize_lower_body:
        print(f"lower_body_optimized_dir: {lower_body_out}")
        print(f"corrected_smpl_pkl: {retarget_input_pkl}")
    print(f"retarget_dir: {retarget_out}")
    print(f"opensim_motion: {retarget_out / 'opensim_ik.mot'}")
    print(
        f"beta_variation_after_max_abs: {beta_report['pipeline_beta_variation_after_max_abs']:.8g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
