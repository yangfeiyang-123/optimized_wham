from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.lower_body import LowerBodyOptimizerConfig, optimize_record
from lib.smpl_optimization.reports import write_json
from lib.world_grounded.tracks import select_track


def parse_args():
    parser = argparse.ArgumentParser(description="Write corrected SMPL output with lower-body ground/contact optimization.")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--track-id", default="merge")
    parser.add_argument("--ground-y", type=float, default=0.0)
    parser.add_argument("--max-root-y-shift", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--enable-pose-pass", action="store_true")
    parser.add_argument("--pose-iterations", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_pkl = Path(args.input_pkl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = joblib.load(input_pkl)
    track_id, record = select_track(results, args.track_id)
    optimized_record, reports = optimize_record(
        record,
        LowerBodyOptimizerConfig(
            fps=args.fps,
            ground_y=args.ground_y,
            max_root_y_shift=args.max_root_y_shift,
            enable_pose_pass=args.enable_pose_pass,
            pose_iterations=args.pose_iterations,
            device=args.device,
        ),
    )

    corrected_smpl_pkl = out_dir / "corrected_smpl.pkl"
    joblib.dump({track_id: optimized_record}, corrected_smpl_pkl)
    for name, report in reports.items():
        write_json(out_dir / f"{name}.json", report)

    validation_success = bool(reports.get("validation_summary", {}).get("success", False))
    write_json(
        out_dir / "optimization_report.json",
        {
            "input_pkl": str(input_pkl),
            "corrected_smpl_pkl": str(corrected_smpl_pkl),
            "track_id": str(track_id),
        },
    )

    print(f"track_id: {track_id}")
    print(f"corrected_smpl_pkl: {corrected_smpl_pkl}")
    print(f"validation_success: {validation_success}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
