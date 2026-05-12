"""Run Stage7 whole-body SMPL smoothness candidate selection."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.metrics import (  # noqa: E402
    beta_variation_max_abs,
    contact_foot_sliding,
    foot_penetration_depth,
    pose_smoothness,
    root_translation_smoothness,
)
from lib.smpl_optimization.opensim_feedback import parse_ik_log_metrics  # noqa: E402
from lib.smpl_optimization.opensim_motion import summarize_mot_coordinates  # noqa: E402
from lib.smpl_optimization.reports import to_jsonable, write_json  # noqa: E402
from lib.smpl_optimization.smpl_forward import (  # noqa: E402
    expand_betas,
    smpl_forward_axis_angle,
)
from lib.smpl_optimization.stage7_selection import select_stage7_result  # noqa: E402
from lib.smpl_optimization.whole_body_smooth import (  # noqa: E402
    WholeBodySmoothConfig,
    smooth_record,
)
from lib.world_grounded.foot_points import (  # noqa: E402
    get_record_contact,
    get_record_foot_points,
)
from lib.world_grounded.tracks import select_track, to_numpy  # noqa: E402


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print()
    print("$ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def stage7_paths(out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    return {
        "baseline_smpl": out_dir / "baseline_smpl.pkl",
        "candidate_smpl": out_dir / "smooth_candidate_smpl.pkl",
        "selected_smpl": out_dir / "selected_smooth_smpl.pkl",
        "baseline_opensim": out_dir / "baseline_opensim",
        "baseline_log": out_dir / "baseline_opensim" / "opensim_ik.log",
        "candidate_opensim": out_dir / "candidate_opensim",
        "candidate_log": out_dir / "candidate_opensim" / "opensim_ik.log",
        "report": out_dir / "stage7_smoothness_report.json",
    }


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


def load_selected_record(pkl: Path, track_id: str) -> tuple[dict, str, dict]:
    data = joblib.load(pkl)
    selected_track_id, record = select_track(data, track_id)
    return data, str(selected_track_id), record


def replace_selected_record(data, selected_track_id: str, record: dict) -> dict:
    if selected_track_id == "merged":
        return {"merged": record}

    updated = deepcopy(data)
    for key in list(updated.keys()):
        if str(key) == str(selected_track_id):
            updated[key] = record
            return updated
    updated[selected_track_id] = record
    return updated


def _record_frames(record: dict) -> int:
    for key in ("trans_world", "trans", "pose_world", "pose", "frame_ids", "frame_id"):
        if key in record:
            return int(np.asarray(to_numpy(record[key])).shape[0])
    return 0


def _record_pose(record: dict) -> np.ndarray:
    for key in ("pose_world", "pose"):
        if key in record:
            pose = np.asarray(to_numpy(record[key]))
            return pose[:, :72] if pose.ndim == 2 and pose.shape[1] >= 72 else pose
    return np.empty((0, 72), dtype=np.float32)


def _record_root(record: dict) -> np.ndarray:
    for key in ("trans_world", "trans"):
        if key in record:
            trans = np.asarray(to_numpy(record[key]))
            return trans[:, :3] if trans.ndim == 2 and trans.shape[1] >= 3 else trans
    return np.empty((0, 3), dtype=np.float32)


def _record_pose_key(record: dict) -> str | None:
    for key in ("pose_world", "pose"):
        if key in record:
            return key
    return None


def _record_trans_key(record: dict) -> str | None:
    for key in ("trans_world", "trans"):
        if key in record:
            return key
    return None


def refresh_smpl_derived_geometry(
    record: dict,
    *,
    device: str,
    chunk_size: int,
) -> dict:
    pose_key = _record_pose_key(record)
    trans_key = _record_trans_key(record)
    if pose_key is None or trans_key is None or "betas" not in record:
        return {
            "refreshed": False,
            "reason": "missing_pose_trans_or_betas",
        }

    pose = np.asarray(to_numpy(record[pose_key]), dtype=np.float32)
    trans = np.asarray(to_numpy(record[trans_key]), dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] < 72 or trans.ndim != 2 or trans.shape[1] < 3:
        return {
            "refreshed": False,
            "reason": "invalid_pose_or_trans_shape",
        }

    n_frames = min(int(pose.shape[0]), int(trans.shape[0]))
    if n_frames <= 0:
        return {
            "refreshed": False,
            "reason": "empty_sequence",
        }
    pose = pose[:n_frames, :72]
    trans = trans[:n_frames, :3]
    betas = expand_betas(record["betas"], n_frames)

    import torch
    from lib.models import build_body_model

    requested_device = str(device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    torch_device = torch.device(requested_device)

    feet_chunks = []
    verts_chunks = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        model = build_body_model(str(torch_device), batch_size=end - start)
        with torch.no_grad():
            smpl = smpl_forward_axis_angle(
                model,
                torch.tensor(pose[start:end], dtype=torch.float32, device=torch_device),
                torch.tensor(betas[start:end], dtype=torch.float32, device=torch_device),
                torch.tensor(trans[start:end], dtype=torch.float32, device=torch_device),
            )
        feet_chunks.append(smpl["feet"].detach().cpu().numpy().astype(np.float32))
        verts_chunks.append(smpl["vertices"].detach().cpu().numpy().astype(np.float32))

    fresh_feet = np.concatenate(feet_chunks, axis=0).astype(np.float32)
    fresh_verts = np.concatenate(verts_chunks, axis=0).astype(np.float32)
    record["feet_refined"] = fresh_feet
    record["feet_world"] = fresh_feet.copy()
    record["feet"] = fresh_feet.copy()
    record["verts"] = fresh_verts
    return {
        "refreshed": True,
        "num_frames": int(n_frames),
        "pose_key": pose_key,
        "trans_key": trans_key,
        "feet_shape": list(fresh_feet.shape),
        "verts_shape": list(fresh_verts.shape),
    }


def _zero_pose_delta() -> dict:
    return {
        "lower_body_max_abs": 0.0,
        "whole_body_max_abs": 0.0,
    }


def _zero_root_delta() -> dict:
    return {
        "max_abs": 0.0,
        "vertical_max_abs": 0.0,
    }


def _penetration(record: dict) -> dict:
    for key in ("feet_world", "feet", "joints_world", "joints"):
        if key not in record:
            continue
        values = np.asarray(to_numpy(record[key]))
        if values.ndim >= 3 and values.shape[-1] >= 2:
            return foot_penetration_depth(values[..., 1])
    return foot_penetration_depth(np.asarray([], dtype=np.float32))


def _sliding(record: dict, fps: float) -> tuple[dict, bool]:
    try:
        foot_points = get_record_foot_points(record)
    except ValueError:
        return {
            "mean_contact_speed": None,
            "max_contact_speed": None,
            "num_sliding": None,
        }, False

    points = foot_points.points
    raw_contact = np.asarray(record["contact"]) if "contact" in record else None
    if (
        raw_contact is None
        or raw_contact.ndim != 2
        or raw_contact.shape != (points.shape[0], points.shape[1])
    ):
        return {
            "mean_contact_speed": None,
            "max_contact_speed": None,
            "num_sliding": None,
        }, False

    contact = get_record_contact(record, points.shape[0], points.shape[1])
    if (
        contact is None
        or contact.ndim != 2
        or contact.shape[0] != points.shape[0]
        or contact.shape[1] != points.shape[1]
    ):
        return {
            "mean_contact_speed": None,
            "max_contact_speed": None,
            "num_sliding": None,
        }, False
    return contact_foot_sliding(points, contact, fps), True


def summarize_smpl(record: dict, fps: float, smooth_report: dict | None = None) -> dict:
    sliding, sliding_available = _sliding(record, fps)
    summary = {
        "beta_variation_after_max_abs": beta_variation_max_abs(
            record.get("betas", np.asarray([]))
        ),
        "frames": _record_frames(record),
        "penetration": _penetration(record),
        "sliding": sliding,
        "sliding_available": sliding_available,
        "pose_delta": _zero_pose_delta(),
        "root_delta": _zero_root_delta(),
        "pose_smoothness": pose_smoothness(_record_pose(record)),
        "root_smoothness": root_translation_smoothness(_record_root(record)),
    }
    if smooth_report:
        for key in (
            "pose_delta",
            "root_delta",
            "pose_smoothness",
            "root_smoothness",
        ):
            if key in smooth_report:
                summary[key] = smooth_report[key]
    return summary


def find_opensim_mot(opensim_dir: Path) -> Path:
    opensim_dir = Path(opensim_dir)
    preferred = opensim_dir / "opensim_ik.mot"
    if preferred.exists():
        return preferred
    raise FileNotFoundError(f"Required opensim_ik.mot not found in {opensim_dir}")


def _opensim_summary(log_path: Path, opensim_dir: Path) -> dict:
    opensim = parse_ik_log_metrics(log_path)
    opensim["motion"] = summarize_mot_coordinates(find_opensim_mot(opensim_dir))
    return opensim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--track-id", default="merge")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=256)
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
    parser.add_argument("--lower-body-weight", type=float, default=0.35)
    parser.add_argument("--trunk-weight", type=float, default=0.55)
    parser.add_argument("--upper-body-weight", type=float, default=0.8)
    parser.add_argument("--max-lower-body-delta", type=float, default=0.05)
    parser.add_argument("--max-whole-body-delta", type=float, default=0.12)
    parser.add_argument("--max-root-delta", type=float, default=0.03)
    parser.add_argument("--max-root-vertical-delta", type=float, default=0.015)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _smooth_config(args: argparse.Namespace) -> WholeBodySmoothConfig:
    return WholeBodySmoothConfig(
        lower_body_weight=args.lower_body_weight,
        trunk_weight=args.trunk_weight,
        upper_body_weight=args.upper_body_weight,
        max_lower_body_delta=args.max_lower_body_delta,
        max_whole_body_delta=args.max_whole_body_delta,
        max_root_delta=args.max_root_delta,
        max_root_vertical_delta=args.max_root_vertical_delta,
    )


def main() -> int:
    args = parse_args()
    input_pkl = Path(args.input_pkl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = stage7_paths(out_dir)

    shutil.copy2(input_pkl, paths["baseline_smpl"])
    data, selected_track_id, record = load_selected_record(
        paths["baseline_smpl"], args.track_id
    )
    candidate_record, smooth_report = smooth_record(record, _smooth_config(args))
    smooth_report["derived_geometry_refresh"] = refresh_smpl_derived_geometry(
        candidate_record,
        device=args.device,
        chunk_size=args.chunk_size,
    )
    candidate_data = replace_selected_record(data, selected_track_id, candidate_record)
    joblib.dump(candidate_data, paths["candidate_smpl"])

    run(
        build_retarget_cmd(
            args,
            paths["baseline_smpl"],
            paths["baseline_opensim"],
            paths["baseline_log"],
        )
    )
    run(
        build_retarget_cmd(
            args,
            paths["candidate_smpl"],
            paths["candidate_opensim"],
            paths["candidate_log"],
        )
    )

    baseline = {
        "smpl": summarize_smpl(record, args.fps),
        "opensim": _opensim_summary(paths["baseline_log"], paths["baseline_opensim"]),
    }
    candidate = {
        "smpl": summarize_smpl(candidate_record, args.fps, smooth_report),
        "opensim": _opensim_summary(
            paths["candidate_log"], paths["candidate_opensim"]
        ),
    }
    selection = select_stage7_result(baseline, candidate)
    selection["smooth_report"] = smooth_report
    selection["selected_track_id"] = selected_track_id

    selected_source = (
        paths["candidate_smpl"]
        if selection["selected"] == "candidate"
        else paths["baseline_smpl"]
    )
    shutil.copy2(selected_source, paths["selected_smpl"])
    write_json(paths["report"], to_jsonable(selection))

    print(f"selected: {selection['selected']}")
    print(f"selected_smooth_smpl: {paths['selected_smpl']}")
    print(f"stage7_smoothness_report: {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
