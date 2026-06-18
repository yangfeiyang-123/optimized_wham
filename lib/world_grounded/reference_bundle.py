from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from lib.world_grounded.body_graph import BODY_GRAPH, graph_labels, laplacian_coordinates
from lib.world_grounded.contact_segments import extract_stance_segments
from lib.world_grounded.foot_points import get_record_contact, get_record_foot_points
from lib.world_grounded.quality_gates import evaluate_quality_gates
from lib.world_grounded.stance_anchor import anchors_to_jsonable, compute_stance_anchors


BUNDLE_VERSION = "contact_reference_bundle_v1"


def export_reference_bundle(
    record: dict[str, Any],
    out_dir: Union[str, Path],
    *,
    sequence: str,
    fps: float,
    quality_report: Optional[dict[str, Any]] = None,
    source: Optional[dict[str, Any]] = None,
    stance_enter_threshold: float = 0.55,
    stance_exit_threshold: float = 0.30,
    stance_min_frames: int = 4,
    stance_merge_gap: int = 2,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    poses = _poses(record)
    n_frames = int(poses.shape[0])
    trans_yup = _fit_2d(record.get("trans_world", record.get("trans", np.zeros((n_frames, 3)))), n_frames, 3)
    betas = _betas(record, n_frames)
    foot = get_record_foot_points(record)
    foot_points_yup = _fit_3d(foot.points, n_frames, len(foot.labels), 3)
    contact = get_record_contact(record, n_frames=n_frames, n_points=foot_points_yup.shape[1])
    if contact is None:
        contact = np.zeros(foot_points_yup.shape[:2], dtype=np.float32)
    contact = _fit_2d(contact, n_frames, foot_points_yup.shape[1])
    body_keypoints_yup, body_labels = _body_keypoints(record, n_frames)
    body_payload: dict[str, Any] = {}
    if body_keypoints_yup is not None:
        body_keypoints_zup = _yup_to_zup(body_keypoints_yup.reshape(-1, 3)).reshape(body_keypoints_yup.shape)
        body_payload = {
            "body_keypoints": body_keypoints_zup.astype(np.float32),
            "body_keypoint_labels": np.asarray(body_labels),
            "body_laplacian": laplacian_coordinates(body_keypoints_zup, body_labels).astype(np.float32),
            "body_keypoints_coordinate_system": np.asarray("amass_zup"),
        }

    segments, stance_mask = extract_stance_segments(
        contact,
        foot.labels,
        enter_threshold=stance_enter_threshold,
        exit_threshold=stance_exit_threshold,
        min_segment_len=stance_min_frames,
        merge_gap=stance_merge_gap,
    )
    anchors_yup = compute_stance_anchors(foot_points_yup, contact, segments, ground_y=0.0)
    anchors_zup = [
        type(anchor)(
            foot_index=anchor.foot_index,
            label=anchor.label,
            start=anchor.start,
            end=anchor.end,
            anchor_xyz=_yup_to_zup(anchor.anchor_xyz[None, :])[0],
            anchor_xz=_yup_to_zup(anchor.anchor_xyz[None, :])[0][[0, 1]],
            ground_y=0.0,
            confidence=anchor.confidence,
            source=anchor.source,
        )
        for anchor in anchors_yup
    ]

    frame_ids = np.arange(n_frames, dtype=np.int32)
    motion_npz = out_path / "motion.npz"
    contact_npz = out_path / "contact_schedule.npz"
    stance_anchors_json = out_path / "stance_anchors.json"
    body_graph_json = out_path / "body_graph.json"
    quality_json = out_path / "quality_report.json"
    processing_json = out_path / "processing_report.json"
    manifest_json = out_path / "manifest.json"

    motion_payload = {
        "poses": poses.astype(np.float32),
        "trans": _yup_to_zup(trans_yup).astype(np.float32),
        "betas": betas.astype(np.float32),
        "gender": np.asarray(str(record.get("gender", "neutral"))),
        "mocap_framerate": np.asarray(float(fps), dtype=np.float32),
        "mocap_frame_rate": np.asarray(float(fps), dtype=np.float32),
        "frame_ids": frame_ids,
        "coordinate_system": np.asarray("amass_zup"),
        "source_coordinate_system": np.asarray("wham_yup"),
        **body_payload,
    }
    np.savez(motion_npz, **motion_payload)
    np.savez(
        contact_npz,
        contact_confidence=contact.astype(np.float32),
        stance_mask=stance_mask.astype(np.bool_),
        foot_points=_yup_to_zup(foot_points_yup.reshape(-1, 3)).reshape(foot_points_yup.shape).astype(np.float32),
        foot_labels=np.asarray(foot.labels),
        frame_ids=frame_ids,
        ground_y=np.asarray(0.0, dtype=np.float32),
        coordinate_system=np.asarray("amass_zup"),
        source_coordinate_system=np.asarray("wham_yup"),
    )
    _write_json(stance_anchors_json, anchors_to_jsonable(anchors_zup, coordinate_system="amass_zup", ground_y=0.0))
    _write_json(body_graph_json, BODY_GRAPH)

    quality = _normalize_quality_report(quality_report or {})
    _write_json(quality_json, quality)

    processing = {
        "version": BUNDLE_VERSION,
        "stages": ["fixed_beta", "world_grounded", "contact_preserving_export"],
        "parameters": {
            "stance_enter_threshold": float(stance_enter_threshold),
            "stance_exit_threshold": float(stance_exit_threshold),
            "root_smooth_axes": ["y"],
        },
    }
    _write_json(processing_json, processing)

    manifest = {
        "version": BUNDLE_VERSION,
        "sequence": str(sequence),
        "num_frames": n_frames,
        "fps": float(fps),
        "unit": "meter",
        "coordinate_system": "amass_zup",
        "source_coordinate_system": "wham_yup",
        "axis_conversion": "Rx(+90deg)",
        "up_axis": "z",
        "motion_npz": motion_npz.name,
        "contact_npz": contact_npz.name,
        "stance_anchors_json": stance_anchors_json.name,
        "body_graph_json": body_graph_json.name,
        "quality_report_json": quality_json.name,
        "processing_report_json": processing_json.name,
        "body_keypoints_available": bool(body_payload),
        "source": dict(source or {}),
        "quality": quality,
    }
    _write_json(manifest_json, manifest)
    return manifest_json


def _yup_to_zup(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0]
    out[..., 1] = -arr[..., 2]
    out[..., 2] = arr[..., 1]
    return out


def _poses(record: dict[str, Any]) -> np.ndarray:
    value = record.get("pose", record.get("pose_world", record.get("poses", np.zeros((0, 72), dtype=np.float32))))
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("record must contain non-empty pose/pose_world/poses with shape [T, D].")
    return arr


def _betas(record: dict[str, Any], n_frames: int) -> np.ndarray:
    arr = np.asarray(record.get("betas", np.zeros((10,), dtype=np.float32)), dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == n_frames:
        return arr[0].copy()
    return arr.reshape(-1)[:10]


def _fit_2d(value: Any, n_frames: int, n_cols: int) -> np.ndarray:
    fitted = np.zeros((n_frames, n_cols), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return fitted
    rows = min(n_frames, arr.shape[0])
    cols = min(n_cols, arr.shape[1])
    fitted[:rows, :cols] = arr[:rows, :cols]
    if rows < n_frames and rows > 0:
        fitted[rows:] = fitted[rows - 1]
    return fitted


def _fit_3d(value: Any, n_frames: int, n_points: int, n_cols: int) -> np.ndarray:
    fitted = np.zeros((n_frames, n_points, n_cols), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return fitted
    rows = min(n_frames, arr.shape[0])
    points = min(n_points, arr.shape[1])
    cols = min(n_cols, arr.shape[2])
    fitted[:rows, :points, :cols] = arr[:rows, :points, :cols]
    if rows < n_frames and rows > 0:
        fitted[rows:] = fitted[rows - 1]
    return fitted


def _body_keypoints(record: dict[str, Any], n_frames: int) -> tuple[Optional[np.ndarray], list[str]]:
    for key in ("body_keypoints", "joints_world", "joints"):
        if key not in record:
            continue
        arr = np.asarray(record[key], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[0] > 0:
            labels = _body_keypoint_labels(record, arr.shape[1])
            return _fit_3d(arr, n_frames, len(labels), 3), labels
    return None, []


def _body_keypoint_labels(record: dict[str, Any], n_points: int) -> list[str]:
    raw = record.get("body_keypoint_labels", record.get("joint_labels"))
    if raw is not None:
        labels = [str(label) for label in np.asarray(raw).tolist()]
    else:
        labels = graph_labels()[:n_points]
    if len(labels) < n_points:
        labels.extend(f"body_keypoint_{idx}" for idx in range(len(labels), n_points))
    return labels[:n_points]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_quality_report(report: dict[str, Any]) -> dict[str, Any]:
    quality = dict(report)
    if "usable_for_training" in quality and "quality_tier" in quality:
        return quality
    if "success" in quality:
        success = bool(quality.get("success", False))
        quality.setdefault("usable_for_training", success)
        quality.setdefault("quality_tier", "B" if success else "D")
        if not success:
            quality.setdefault("failed_gates", _failed_validation_checks(quality))
            quality.setdefault("recommendation", "exclude_from_training")
        return quality
    if _looks_like_metric_report(quality):
        gated = evaluate_quality_gates(quality)
        return {**quality, **gated}
    quality.setdefault("usable_for_training", True)
    quality.setdefault("quality_tier", "B")
    return quality


def _looks_like_metric_report(report: dict[str, Any]) -> bool:
    metric_keys = {
        "foot_penetration_max_cm_after",
        "stance_sliding_max_cm_after",
        "root_delta_y_max_cm",
        "root_delta_xz_max_cm",
        "contact_switch_rate",
        "stance_coverage_ratio",
    }
    return bool(metric_keys & set(report))


def _failed_validation_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    checks = report.get("checks", {})
    if not isinstance(checks, dict):
        return []
    return [
        {"name": str(name), "value": bool(value), "threshold": True}
        for name, value in checks.items()
        if not bool(value)
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
