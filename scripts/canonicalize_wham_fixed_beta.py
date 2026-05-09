import argparse
import copy
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import constants as _C


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def track_len(record):
    if "frame_ids" in record:
        return len(record["frame_ids"])
    if "frame_id" in record:
        return len(record["frame_id"])
    betas = to_numpy(record["betas"])
    return 1 if betas.ndim == 1 else len(betas)


def find_track(mapping, track_id):
    if mapping is None:
        return None
    if track_id in mapping:
        return mapping[track_id]
    for key, value in mapping.items():
        try:
            if int(key) == int(track_id):
                return value
        except Exception:
            pass
    return None


def load_tracking_results(path):
    if path is None or not path.exists():
        return None
    return joblib.load(path)


def frame_quality_from_tracking(tracking_record, n_frames):
    if tracking_record is None or "keypoints" not in tracking_record:
        return np.ones(n_frames, dtype=np.float64)

    keypoints = to_numpy(tracking_record["keypoints"])
    if keypoints.ndim != 3 or keypoints.shape[-1] < 3:
        return np.ones(n_frames, dtype=np.float64)

    conf = keypoints[:n_frames, :, 2].astype(np.float64)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.clip(conf, 0.0, None)
    weights = conf.mean(axis=1)

    if len(weights) < n_frames:
        weights = np.pad(weights, (0, n_frames - len(weights)), constant_values=weights.mean() if len(weights) else 1.0)
    if not np.any(weights > 0):
        weights = np.ones(n_frames, dtype=np.float64)
    return weights[:n_frames]


def weighted_median(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) == 0:
        raise ValueError("Cannot compute beta_fixed from zero beta samples.")

    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, 0.0, None)
    if weights.shape[0] != values.shape[0] or weights.sum() <= 0:
        weights = np.ones(values.shape[0], dtype=np.float64)

    med = np.zeros(values.shape[1], dtype=np.float64)
    for dim in range(values.shape[1]):
        order = np.argsort(values[:, dim])
        sorted_values = values[order, dim]
        sorted_weights = weights[order]
        cdf = np.cumsum(sorted_weights)
        cutoff = 0.5 * sorted_weights.sum()
        med[dim] = sorted_values[np.searchsorted(cdf, cutoff, side="left")]
    return med


def estimate_fixed_beta(samples, weights, keep_percentile=80.0):
    samples = np.asarray(samples, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 10:
        raise ValueError(f"Expected beta samples with shape [N, 10], got {samples.shape}.")

    base_median = np.median(samples, axis=0)
    dist = np.linalg.norm(samples - base_median[None], axis=1)
    cutoff = np.percentile(dist, keep_percentile)
    valid = dist <= cutoff
    if valid.sum() < max(3, int(0.1 * len(samples))):
        valid = np.ones(len(samples), dtype=bool)

    filtered_weights = weights[valid]
    if filtered_weights.sum() <= 0:
        filtered_weights = np.ones(valid.sum(), dtype=np.float64)

    beta_fixed = weighted_median(samples[valid], filtered_weights).astype(np.float32)
    stats = {
        "num_samples": int(len(samples)),
        "num_valid_samples": int(valid.sum()),
        "keep_percentile": float(keep_percentile),
        "distance_cutoff": float(cutoff),
        "distance_mean": float(dist.mean()),
        "distance_max": float(dist.max()),
    }
    return beta_fixed, valid, stats


def collect_beta_samples(results, tracking_results):
    samples = []
    weights = []
    sample_tracks = []

    for track_id, record in results.items():
        if "betas" not in record:
            continue
        betas = to_numpy(record["betas"]).astype(np.float64)
        n_frames = track_len(record)
        tracking_record = find_track(tracking_results, track_id)

        if betas.ndim == 1:
            samples.append(betas.reshape(1, 10))
            weights.append(np.array([frame_quality_from_tracking(tracking_record, n_frames).mean()], dtype=np.float64))
            sample_tracks.append(np.array([str(track_id)], dtype=object))
        elif betas.ndim == 2 and betas.shape[1] == 10:
            beta_n = min(len(betas), n_frames)
            samples.append(betas[:beta_n])
            weights.append(frame_quality_from_tracking(tracking_record, beta_n))
            sample_tracks.append(np.array([str(track_id)] * beta_n, dtype=object))
        else:
            raise ValueError(f"Track {track_id} has unsupported betas shape {betas.shape}.")

    if not samples:
        raise ValueError("No betas found in WHAM output.")

    return np.concatenate(samples, axis=0), np.concatenate(weights, axis=0), np.concatenate(sample_tracks, axis=0)


def forward_axis_angle(smpl, pose72, betas, transl=None, device="cpu"):
    pose72 = np.asarray(pose72, dtype=np.float32)
    betas = np.asarray(betas, dtype=np.float32)
    body_pose = torch.from_numpy(pose72[:, 3:]).float().to(device)
    global_orient = torch.from_numpy(pose72[:, :3]).float().to(device)
    betas_t = torch.from_numpy(betas).float().to(device)
    kwargs = {
        "body_pose": body_pose,
        "global_orient": global_orient,
        "betas": betas_t,
    }
    if transl is not None:
        kwargs["transl"] = torch.from_numpy(np.asarray(transl, dtype=np.float32)).float().to(device)
    with torch.no_grad():
        output = smpl.get_output(**kwargs)
    return {
        "vertices": output.vertices.detach().cpu().numpy(),
        "joints": output.joints.detach().cpu().numpy(),
        "offset": output.offset.detach().cpu().numpy(),
    }


def run_in_chunks(pose72, betas, transl=None, device="cpu", chunk_size=256):
    from lib.models import build_body_model

    pose72 = np.asarray(pose72, dtype=np.float32)
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        betas = np.repeat(betas[None], len(pose72), axis=0)
    outputs = {"vertices": [], "joints": [], "offset": []}

    for start in range(0, len(pose72), chunk_size):
        end = min(start + chunk_size, len(pose72))
        smpl = build_body_model(device, batch_size=end - start)
        transl_chunk = None if transl is None else np.asarray(transl, dtype=np.float32)[start:end]
        chunk = forward_axis_angle(
            smpl,
            pose72[start:end],
            betas[start:end],
            transl=transl_chunk,
            device=device,
        )
        for key in outputs:
            outputs[key].append(chunk[key])

    return {key: np.concatenate(value, axis=0) for key, value in outputs.items()}


def bone_length_summary(joints):
    parents = _C.BMODEL.PARENTS.detach().cpu().numpy()
    joints = np.asarray(joints, dtype=np.float64)
    edges = [(idx, int(parent)) for idx, parent in enumerate(parents) if parent >= 0 and idx < joints.shape[1] and parent < joints.shape[1]]
    if not edges:
        return {"num_edges": 0, "max_cv": 0.0, "mean_cv": 0.0, "max_deviation": 0.0}

    cvs = []
    max_devs = []
    for child, parent in edges:
        lengths = np.linalg.norm(joints[:, child] - joints[:, parent], axis=1)
        mean = float(lengths.mean())
        std = float(lengths.std())
        cvs.append(0.0 if mean == 0 else std / mean)
        max_devs.append(float(np.max(np.abs(lengths - mean))))

    return {
        "num_edges": int(len(edges)),
        "max_cv": float(np.max(cvs)),
        "mean_cv": float(np.mean(cvs)),
        "max_deviation": float(np.max(max_devs)),
    }


def beta_variation(betas):
    betas = np.asarray(betas, dtype=np.float64)
    if betas.ndim == 1 or len(betas) <= 1:
        return 0.0
    return float(np.max(np.abs(betas - betas[0:1])))


SEQUENCE_FIELDS_TO_TRIM = [
    "contact",
    "feet_world",
    "feet_refined",
    "feet_cam",
    "feet_local",
]


def trim_sequence_fields(out_record, n_frames):
    for key in SEQUENCE_FIELDS_TO_TRIM:
        if key not in out_record:
            continue
        value = to_numpy(out_record[key])
        if hasattr(value, "__len__") and len(value) >= n_frames:
            out_record[key] = value[:n_frames].astype(np.float32)


def canonicalize_track(record, beta_fixed, device, chunk_size):
    out_record = copy.deepcopy(record)
    if "pose" not in record:
        raise ValueError("Only WHAM demo-style outputs with a 'pose' field are supported in this script.")

    pose = to_numpy(record["pose"]).astype(np.float32)
    old_betas = to_numpy(record["betas"]).astype(np.float32)
    if old_betas.ndim == 1:
        old_betas_seq = np.repeat(old_betas[None], len(pose), axis=0)
    else:
        old_betas_seq = old_betas[: len(pose)]
    fixed_betas_seq = np.repeat(beta_fixed[None], len(pose), axis=0).astype(np.float32)

    old_zero = run_in_chunks(pose, old_betas_seq, device=device, chunk_size=chunk_size)
    new_zero = run_in_chunks(pose, fixed_betas_seq, device=device, chunk_size=chunk_size)

    if "trans" in record:
        old_trans = to_numpy(record["trans"]).astype(np.float32)[: len(pose)]
        full_cam = old_trans + old_zero["offset"]
        out_record["trans"] = (full_cam - new_zero["offset"]).astype(np.float32)
        out_record["verts"] = (new_zero["vertices"] + full_cam[:, None, :]).astype(np.float32)

    if "pose_world" in record and "trans_world" in record:
        pose_world = to_numpy(record["pose_world"]).astype(np.float32)
        old_trans_world = to_numpy(record["trans_world"]).astype(np.float32)[: len(pose_world)]
        old_world = run_in_chunks(pose_world, old_betas_seq[: len(pose_world)], transl=old_trans_world, device=device, chunk_size=chunk_size)
        new_world_zero = run_in_chunks(pose_world, fixed_betas_seq[: len(pose_world)], device=device, chunk_size=chunk_size)
        old_pelvis = old_world["joints"][:, 0]
        new_pelvis_zero = new_world_zero["joints"][:, 0]
        out_record["trans_world"] = (old_pelvis - new_pelvis_zero).astype(np.float32)

    out_record["betas"] = fixed_betas_seq.astype(np.float32)
    trim_sequence_fields(out_record, len(pose))

    report = {
        "num_frames": int(len(pose)),
        "beta_variation_before": beta_variation(old_betas_seq),
        "beta_variation_after": beta_variation(fixed_betas_seq),
        "bone_lengths_before": bone_length_summary(old_zero["joints"]),
        "bone_lengths_after": bone_length_summary(new_zero["joints"]),
    }
    if "trans" in record and "verts" in record:
        old_verts = to_numpy(record["verts"]).astype(np.float32)
        report["camera_vertices_changed_mean_abs"] = float(np.mean(np.abs(out_record["verts"] - old_verts)))
        report["camera_vertices_changed_max_abs"] = float(np.max(np.abs(out_record["verts"] - old_verts)))

    return out_record, report


def canonicalize_wham_pkl(args):
    wham_pkl = Path(args.wham_pkl)
    out_pkl = Path(args.out_pkl) if args.out_pkl else wham_pkl.with_name("canonical_wham_output.pkl")
    report_path = Path(args.report) if args.report else out_pkl.with_name("fixed_beta_report.json")
    beta_path = Path(args.beta_out) if args.beta_out else out_pkl.with_name("beta_fixed.npy")
    tracking_path = Path(args.tracking_results) if args.tracking_results else wham_pkl.with_name("tracking_results.pth")

    results = joblib.load(wham_pkl)
    tracking_results = load_tracking_results(tracking_path)

    samples, weights, sample_tracks = collect_beta_samples(results, tracking_results)
    beta_fixed, valid_mask, beta_stats = estimate_fixed_beta(samples, weights, keep_percentile=args.keep_percentile)

    canonical = copy.deepcopy(results)
    track_reports = {}
    for track_id, record in results.items():
        if "betas" not in record:
            continue
        canonical_record, track_report = canonicalize_track(record, beta_fixed, args.device, args.chunk_size)
        canonical[track_id] = canonical_record
        track_reports[str(track_id)] = track_report

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    beta_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(canonical, out_pkl)
    np.save(beta_path, beta_fixed)

    report = {
        "source_wham_pkl": str(wham_pkl),
        "output_wham_pkl": str(out_pkl),
        "tracking_results": str(tracking_path) if tracking_results is not None else None,
        "shared_beta_scope": "clip",
        "beta_method": "weighted_median_after_beta_outlier_filter",
        "beta_fixed": beta_fixed,
        "beta_estimation": beta_stats,
        "sample_tracks": sorted(set(sample_tracks.tolist())),
        "num_tracks": int(len(track_reports)),
        "global_beta_variation_before": beta_variation(samples),
        "global_beta_variation_after": 0.0,
        "tracks": track_reports,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_value(report), f, indent=2)

    return out_pkl, report_path, beta_path


def parse_args():
    parser = argparse.ArgumentParser(description="Canonicalize WHAM SMPL betas to one fixed clip-level beta.")
    parser.add_argument("--wham-pkl", required=True, help="Path to WHAM demo-style wham_output.pkl.")
    parser.add_argument("--out-pkl", default=None, help="Output pkl path. Defaults to canonical_wham_output.pkl next to input.")
    parser.add_argument("--report", default=None, help="Output JSON report path.")
    parser.add_argument("--beta-out", default=None, help="Output beta_fixed.npy path.")
    parser.add_argument("--tracking-results", default=None, help="Optional tracking_results.pth path for keypoint confidence weights.")
    parser.add_argument("--device", default="cpu", help="Torch device for SMPL forward, e.g. cpu or cuda.")
    parser.add_argument("--chunk-size", type=int, default=256, help="SMPL forward chunk size.")
    parser.add_argument("--keep-percentile", type=float, default=80.0, help="Percentile of beta-distance samples to keep.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_pkl, report_path, beta_path = canonicalize_wham_pkl(args)
    print(f"Saved canonical WHAM pkl: {out_pkl}")
    print(f"Saved fixed beta: {beta_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
