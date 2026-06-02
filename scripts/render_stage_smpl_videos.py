"""Render staged WHAM/SMPL pkl files into overlay videos for Demo outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio
import joblib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import get_cfg_defaults
from lib.models import build_body_model
from lib.vis.renderer import Renderer
from lib.vis.run_vis import run_vis_on_demo


STAGES = [
    "stage1_raw",
    "stage2_fixed_beta",
    "stage3_world_grounded",
    "stage4_root_y_only",
    "stage5_lower_body_full",
    "stage6_opensim_feedback",
    "stage7_whole_body_smooth",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-global", action="store_true")
    parser.add_argument(
        "--camera-smpl-only",
        action="store_true",
        help="Render only the camera-view SMPL mesh on a blank background.",
    )
    return parser


def render_camera_smpl_only(cfg, video: Path, results: dict, output_path: Path, smpl) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    focal_length = (width**2 + height**2) ** 0.5
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl.faces)
    default_r = torch.eye(3)
    default_t = torch.zeros(3)

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        mode="I",
        format="FFMPEG",
        macro_block_size=1,
    )

    try:
        for frame_i in range(length):
            background = np.ones((height, width, 3), dtype=np.uint8) * 255
            renderer.create_camera(default_r, default_t)
            for val in results.values():
                frame_ids = np.asarray(val["frame_ids"])
                frame_i2 = np.where(frame_ids == frame_i)[0]
                if len(frame_i2) == 0:
                    continue
                verts = torch.from_numpy(val["verts"][frame_i2[0]]).to(cfg.DEVICE)
                background = renderer.render_mesh(verts, background)
            writer.append_data(background)
    finally:
        writer.close()
        cap.release()


def main() -> int:
    args = build_parser().parse_args()
    video = Path(args.video)
    output_root = Path(args.output_root)
    sequence = args.sequence if args.sequence is not None else video.stem

    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/yamls/demo.yaml")
    cfg.DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    smpl = build_body_model(cfg.DEVICE, 512)
    for stage in STAGES:
        stage_dir = output_root / stage
        pkl = stage_dir / f"{sequence}_{stage}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"stage pkl not found: {pkl}")

        print(f"render: {pkl}", flush=True)
        data = joblib.load(pkl)

        if args.camera_smpl_only:
            target = stage_dir / f"{sequence}_{stage}_smpl_only.mp4"
            render_camera_smpl_only(cfg, video, data, target, smpl)
        else:
            run_vis_on_demo(
                cfg,
                str(video),
                data,
                str(stage_dir),
                smpl,
                vis_global=not args.no_global,
            )

            rendered = stage_dir / "output.mp4"
            target = stage_dir / f"{sequence}_{stage}_smpl.mp4"
            if not rendered.exists():
                raise FileNotFoundError(f"renderer did not write: {rendered}")
            if target.exists():
                target.unlink()
            rendered.replace(target)
        print(f"video: {target}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
