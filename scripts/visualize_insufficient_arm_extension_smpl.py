from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import imageio
import joblib
import numpy as np
import torch

REPO = Path("/data3/yangfeiyang/WorkSpace/optimized_wham")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lib.models import build_body_model
from lib.vis.renderer import Renderer, get_global_cameras


DEFAULT_ROOT = REPO / "output/forehand_clear/InsufficientArmExtension"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def natural_index(path: Path) -> int:
    match = re.search(r"-(\d+)$", path.name)
    if match:
        return int(match.group(1))
    return 10**9


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def select_single_record(results: dict[Any, dict[str, Any]], source: Path) -> tuple[str, dict[str, Any]]:
    if len(results) != 1:
        raise ValueError(f"Expected exactly one SMPL track in {source}, found {list(results.keys())}")
    key = next(iter(results.keys()))
    return str(key), results[key]


def prepare_global_verts(record: dict[str, Any], device: str) -> torch.Tensor:
    if "verts" not in record:
        raise ValueError("Record does not contain verts.")
    verts_np = to_numpy(record["verts"]).astype(np.float32)
    if verts_np.ndim != 3 or verts_np.shape[-1] != 3:
        raise ValueError(f"Expected verts shape [T, V, 3], got {verts_np.shape}")
    verts = torch.from_numpy(verts_np).float()
    verts[..., 1] = verts[..., 1] - verts[..., 1].min()
    center_xz = verts.mean(1).mean(0)[[0, 2]]
    verts[..., 0] -= center_xz[0]
    verts[..., 2] -= center_xz[1]
    return verts.to(device)


def render_sequence(
    *,
    seq_dir: Path,
    output_mp4: Path,
    index: int,
    fps: float,
    renderer: Renderer,
    faces: torch.Tensor,
    device: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    pkl = seq_dir / "lower_body_corrected" / "corrected_smpl.pkl"
    results = joblib.load(pkl)
    track_id, record = select_single_record(results, pkl)
    verts = prepare_global_verts(record, device)
    n_frames = int(verts.shape[0])

    mean = verts.mean(1)
    span = mean.max(0).values - mean.min(0).values
    mesh_span = verts.reshape(-1, 3).max(0).values - verts.reshape(-1, 3).min(0).values
    ground_scale = float(max(span[[0, 2]].max().item(), mesh_span[[0, 2]].max().item(), 1.0) * 1.8)
    renderer.set_ground(ground_scale, 0.0, 0.0)
    global_r, global_t, lights = get_global_cameras(
        verts.detach().cpu(),
        device,
        distance=float(max(4.0, ground_scale * 1.3)),
        position=(-4.0, 3.2, 4.0),
    )
    global_r = global_r.to(device)
    global_t = global_t.to(device)

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_mp4), fps=fps, mode="I", format="FFMPEG", macro_block_size=1)
    try:
        for frame_i in range(n_frames):
            cameras = renderer.create_camera(global_r[frame_i], global_t[frame_i])
            colors = torch.ones((1, 4), dtype=torch.float32, device=device)
            colors[..., :3] = torch.tensor([0.88, 0.88, 0.84], dtype=torch.float32, device=device)
            img = renderer.render_with_ground(
                verts[frame_i : frame_i + 1],
                faces,
                colors,
                cameras,
                lights,
            )
            label = f"{index:03d} {seq_dir.name}  frame {frame_i + 1}/{n_frames}  track {track_id}"
            cv2.putText(img, label, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img, label, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 1, cv2.LINE_AA)
            writer.append_data(img)
            if (frame_i + 1) % 60 == 0 or frame_i + 1 == n_frames:
                print(f"[{timestamp()}] {seq_dir.name} frame {frame_i + 1}/{n_frames}", flush=True)
    finally:
        writer.close()

    return {
        "index": index,
        "sequence": seq_dir.name,
        "track_id": track_id,
        "frames": n_frames,
        "fps": fps,
        "source_pkl": str(pkl),
        "output_mp4": str(output_mp4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render ordered SMPL visualizations for InsufficientArmExtension outputs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "tmp")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "render_status.tsv"
    manifest_path = out_dir / "render_manifest.csv"

    seq_dirs = sorted(
        [
            p
            for p in root.iterdir()
            if p.is_dir()
            and p.name not in {"tmp", "person_filter_previews", "person_filter_logs"}
            and (p / "lower_body_corrected" / "corrected_smpl.pkl").exists()
        ],
        key=natural_index,
    )

    print(f"[{timestamp()}] found {len(seq_dirs)} sequences", flush=True)
    smpl = build_body_model(args.device, batch_size=1)
    renderer = Renderer(
        args.width,
        args.height,
        (args.width**2 + args.height**2) ** 0.5,
        args.device,
        smpl.faces,
    )
    faces = renderer.faces.clone().squeeze(0)

    rows: list[dict[str, Any]] = []
    with status_path.open("w", newline="", encoding="utf-8") as status_file:
        status_writer = csv.writer(status_file, delimiter="\t")
        status_writer.writerow(["timestamp", "index", "total", "sequence", "status", "output_mp4"])
        status_file.flush()

        for index, seq_dir in enumerate(seq_dirs, start=1):
            safe_name = seq_dir.name.replace(" ", "_")
            output_mp4 = out_dir / f"{index:03d}_{safe_name}_smpl.mp4"
            status_writer.writerow([timestamp(), index, len(seq_dirs), seq_dir.name, "START", str(output_mp4)])
            status_file.flush()
            print(f"[{timestamp()}] START {index}/{len(seq_dirs)} {seq_dir.name}", flush=True)

            if output_mp4.exists() and output_mp4.stat().st_size > 0 and not args.overwrite:
                row = {
                    "index": index,
                    "sequence": seq_dir.name,
                    "track_id": "",
                    "frames": "",
                    "fps": args.fps,
                    "source_pkl": str(seq_dir / "lower_body_corrected" / "corrected_smpl.pkl"),
                    "output_mp4": str(output_mp4),
                }
                rows.append(row)
                status_writer.writerow([timestamp(), index, len(seq_dirs), seq_dir.name, "SKIP_EXISTS", str(output_mp4)])
                status_file.flush()
                print(f"[{timestamp()}] SKIP_EXISTS {seq_dir.name}", flush=True)
                continue

            row = render_sequence(
                seq_dir=seq_dir,
                output_mp4=output_mp4,
                index=index,
                fps=args.fps,
                renderer=renderer,
                faces=faces,
                device=args.device,
                width=args.width,
                height=args.height,
            )
            rows.append(row)
            status_writer.writerow([timestamp(), index, len(seq_dirs), seq_dir.name, "DONE", str(output_mp4)])
            status_file.flush()
            print(f"[{timestamp()}] DONE {index}/{len(seq_dirs)} {seq_dir.name}", flush=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "sequence", "track_id", "frames", "fps", "source_pkl", "output_mp4"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{timestamp()}] render complete: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
