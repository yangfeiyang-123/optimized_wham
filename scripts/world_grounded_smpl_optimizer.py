from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.world_grounded.foot_points import get_record_contact, get_record_foot_points
from lib.world_grounded.ground import estimate_ground
from lib.world_grounded.reports import contact_coverage, contact_switch_rate, write_json
from lib.world_grounded.root_optimizer import optimize_root_translation
from lib.world_grounded.tracks import select_track


def _copy_shifted_sequence(record: dict, key: str, n_frames: int, delta: np.ndarray) -> None:
    if key not in record:
        return
    arr = np.asarray(record[key], dtype=np.float32).copy()
    if arr.ndim == 3 and arr.shape[-1] == 3:
        n = min(n_frames, len(arr))
        arr[:n] = arr[:n] + delta[:n, None, :]
        record[key] = arr


def optimize_record(
    record: dict,
    fps: float,
    align_ground_to_zero: bool = True,
    root_smooth_axes: tuple[str, ...] = ("y",),
    max_xz_delta: float = 0.03,
    max_y_delta: float = 0.30,
) -> tuple[dict, dict, dict, dict]:
    if "trans_world" not in record:
        raise ValueError("Record must contain trans_world for root translation optimization.")

    out = copy.deepcopy(record)
    trans_world = np.asarray(record["trans_world"], dtype=np.float32)
    foot = get_record_foot_points(record)
    n_frames = min(len(trans_world), len(foot.points))
    if n_frames == 0:
        raise ValueError("Cannot optimize an empty record.")

    foot_points = foot.points[:n_frames]
    contact = get_record_contact(record, n_frames, foot_points.shape[1])
    ground = estimate_ground(foot_points, fps=fps, wham_contact=contact)
    root = optimize_root_translation(
        trans_world[:n_frames],
        foot_points,
        ground.contact_confidence,
        ground_y=ground.ground_y,
        fps=fps,
        smooth_axes=root_smooth_axes,
        max_xz_delta=max_xz_delta,
        max_y_delta=max_y_delta,
    )

    delta = root.delta.astype(np.float32)
    ground_alignment_offset_y = -float(ground.ground_y) if align_ground_to_zero else 0.0
    total_delta = delta.copy()
    total_delta[:, 1] += np.float32(ground_alignment_offset_y)

    out["trans_world"] = trans_world.copy()
    out["trans_world"][:n_frames] = root.optimized_trans_world
    out["trans_world"][:n_frames, 1] += np.float32(ground_alignment_offset_y)
    for key in ("verts", "feet_world", "feet_refined", "feet"):
        _copy_shifted_sequence(out, key, n_frames, total_delta)

    ground_report = {
        "ground_y": ground.ground_y,
        "ground_alignment_offset_y": ground_alignment_offset_y,
        "ground_y_after_alignment": float(ground.ground_y + ground_alignment_offset_y),
        "ground_confidence": ground.ground_confidence,
        "contact_source": ground.contact_source,
        "foot_source": foot.source,
        "foot_labels": foot.labels,
    }
    contact_report = {
        "contact_coverage": contact_coverage(ground.contact_confidence),
        "contact_switch_rate": contact_switch_rate(ground.contact_confidence),
        "contact_confidence_shape": list(ground.contact_confidence.shape),
    }
    quality_report = {
        **ground_report,
        **contact_report,
        **root.report,
        "num_frames": int(n_frames),
    }
    return out, ground_report, contact_report, quality_report


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize fixed-beta WHAM SMPL for ground/contact consistency.")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--track-id", default="merge")
    parser.add_argument("--run-retarget", action="store_true")
    parser.add_argument("--retarget-config", default="configs/retarget/smpl_to_mimicmsk_opensim.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opensim-cmd", default=r"C:\Users\yangfeiyang\Downloads\OpenSim 4.5\bin\opensim-cmd.exe")
    parser.add_argument(
        "--no-align-ground-to-zero",
        action="store_true",
        help="Keep the optimized trajectory in WHAM world height instead of shifting estimated ground to OpenSim Y=0.",
    )
    parser.add_argument(
        "--root-smooth-axes",
        default="y",
        help="Comma-separated root axes to smooth before vertical penetration correction. Default preserves horizontal X/Z.",
    )
    parser.add_argument("--max-xz-delta", type=float, default=0.03)
    parser.add_argument("--max-y-delta", type=float, default=0.30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_pkl = Path(args.input_pkl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = joblib.load(input_pkl)
    track_id, record = select_track(results, args.track_id)
    optimized_record, ground_report, contact_report, quality_report = optimize_record(
        record,
        fps=args.fps,
        align_ground_to_zero=not args.no_align_ground_to_zero,
        root_smooth_axes=tuple(axis.strip() for axis in args.root_smooth_axes.split(",") if axis.strip()),
        max_xz_delta=args.max_xz_delta,
        max_y_delta=args.max_y_delta,
    )

    optimized = {track_id: optimized_record}
    optimized_pkl = out_dir / "optimized_canonical_wham_output.pkl"
    joblib.dump(optimized, optimized_pkl)
    write_json(out_dir / "ground_plane.json", ground_report)
    write_json(out_dir / "contact_report.json", contact_report)
    write_json(out_dir / "quality_report.json", quality_report)
    write_json(
        out_dir / "optimization_report.json",
        {
            "input_pkl": str(input_pkl),
            "output_pkl": str(optimized_pkl),
            "track_id": str(track_id),
        },
    )

    print(f"track_id: {track_id}")
    print(f"frames: {quality_report['num_frames']}")
    print(f"optimized_pkl: {optimized_pkl}")
    print(f"ground_y: {ground_report['ground_y']:.8f}")
    print(f"ground_alignment_offset_y: {ground_report['ground_alignment_offset_y']:.8f}")
    print(f"ground_confidence: {ground_report['ground_confidence']}")

    if args.run_retarget:
        retarget_dir = out_dir / "retarget"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "retarget_smpl_to_opensim.py"),
            str(optimized_pkl),
            "--config",
            args.retarget_config,
            "--out-dir",
            str(retarget_dir),
            "--fps",
            str(args.fps),
            "--device",
            args.device,
            "--track-id",
            str(track_id),
            "--free-root",
            "--run-ik",
            "--opensim-cmd",
            args.opensim_cmd,
        ]
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
