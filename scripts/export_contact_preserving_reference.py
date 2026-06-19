from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Union

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.world_grounded.reference_bundle import export_reference_bundle
from lib.world_grounded.tracks import select_track


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export contact-preserving reference bundle for ASI-PPO/MuscleMimic.")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--track-id", default="merge")
    parser.add_argument("--quality-report", default=None)
    parser.add_argument("--source-json", default=None)
    parser.add_argument("--stance-enter-threshold", type=float, default=0.55)
    parser.add_argument("--stance-exit-threshold", type=float, default=0.30)
    parser.add_argument("--stance-min-frames", type=int, default=4)
    parser.add_argument("--stance-merge-gap", type=int, default=2)
    parser.add_argument("--root-smooth-axes", default="y")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_pkl = Path(args.input_pkl).resolve()
    out_dir = Path(args.out_dir).resolve()
    results = joblib.load(input_pkl)
    track_id, record = select_track(results, args.track_id)
    quality = _read_json(args.quality_report) if args.quality_report else {}
    source = _read_json(args.source_json) if args.source_json else {"input_pkl": str(input_pkl), "track_id": str(track_id)}
    source.setdefault("input_pkl", str(input_pkl))
    source.setdefault("track_id", str(track_id))

    manifest = export_reference_bundle(
        record,
        out_dir,
        sequence=args.sequence,
        fps=args.fps,
        quality_report=quality,
        source=source,
        stance_enter_threshold=args.stance_enter_threshold,
        stance_exit_threshold=args.stance_exit_threshold,
        stance_min_frames=args.stance_min_frames,
        stance_merge_gap=args.stance_merge_gap,
        root_smooth_axes=tuple(axis.strip() for axis in args.root_smooth_axes.split(",") if axis.strip()),
    )
    print(f"reference_manifest: {manifest}")
    return 0


def _read_json(path: Union[str, Path]) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
