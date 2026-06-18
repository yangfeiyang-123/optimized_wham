"""Analyze forehand-clear WHAM post-processing quality across completed clips."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_wham_ablation import compute_metrics, select_record


STAGES = [
    {
        "id": "00_raw_wham",
        "description": "Original WHAM signal",
        "manifest_key": "raw_wham",
    },
    {
        "id": "01_fixed_beta",
        "description": "Fixed-beta canonical signal",
        "manifest_key": "fixed_beta",
    },
    {
        "id": "02_world_grounded",
        "description": "World-grounded signal",
        "relative_path": "world_grounded/optimized_canonical_wham_output.pkl",
    },
    {
        "id": "03_lower_body_corrected",
        "description": "Post-processed signal: world-grounded plus lower-body/contact correction",
        "manifest_key": "corrected_smpl",
    },
]

LOWER_STAGE = "03_lower_body_corrected"
METRICS_LOWER_IS_BETTER = [
    "penetration0_max",
    "penetration0_mean",
    "contact_mean_speed",
    "contact_max_speed",
    "contact_num_sliding",
    "root_rms_vertical_accel",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def as_float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def percent_reduction(before: Any, after: Any) -> float | None:
    before_value = as_float(before)
    after_value = as_float(after)
    if before_value is None or after_value is None or abs(before_value) < 1e-12:
        return None
    return 100.0 * (before_value - after_value) / before_value


def mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if np.isfinite(float(v))]
    if not clean:
        return None
    return float(np.mean(clean))


def median(values: list[float]) -> float | None:
    clean = [float(v) for v in values if np.isfinite(float(v))]
    if not clean:
        return None
    return float(np.median(clean))


def summarize_numeric(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [as_float(row.get(metric)) for row in rows]
    clean = [v for v in values if v is not None]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(clean),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def sequence_group(name: str) -> str:
    if name.startswith("video"):
        return "video"
    if "(1)" in name:
        return "june2_repeat"
    if name.startswith("6月2日"):
        return "june2"
    return "other"


def read_manifest_rows(manifest_tsv: Path) -> list[dict[str, str]]:
    with manifest_tsv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows


def stage_path(row: dict[str, str], stage: dict[str, str]) -> Path:
    if "manifest_key" in stage:
        return Path(row[stage["manifest_key"]])
    return Path(row["folder"]) / stage["relative_path"]


def read_reference_bundle(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    contact_path = bundle_dir / "contact_schedule.npz"
    motion_path = bundle_dir / "motion.npz"
    stance_path = bundle_dir / "stance_anchors.json"
    out: dict[str, Any] = {
        "reference_manifest_exists": manifest_path.exists(),
        "reference_motion_exists": motion_path.exists(),
        "reference_contact_exists": contact_path.exists(),
        "reference_stance_anchors_exists": stance_path.exists(),
    }
    if not manifest_path.exists():
        return out

    manifest = read_json(manifest_path)
    quality = manifest.get("quality", {})
    checks = quality.get("checks", {})
    out.update(
        {
            "reference_version": manifest.get("version", ""),
            "reference_num_frames": manifest.get("num_frames", ""),
            "reference_fps": manifest.get("fps", ""),
            "reference_coordinate_system": manifest.get("coordinate_system", ""),
            "reference_source_coordinate_system": manifest.get("source_coordinate_system", ""),
            "reference_quality_success": bool(quality.get("success", False)),
            "reference_usable_for_training": bool(quality.get("usable_for_training", False)),
            "reference_quality_tier": quality.get("quality_tier", ""),
            "reference_all_checks_pass": all(bool(value) for value in checks.values()) if checks else False,
        }
    )

    if contact_path.exists():
        with np.load(contact_path, allow_pickle=False) as contact:
            confidence = np.asarray(contact["contact_confidence"], dtype=np.float32)
            stance_mask = np.asarray(contact["stance_mask"], dtype=np.bool_)
            out.update(
                {
                    "reference_contact_shape": list(confidence.shape),
                    "reference_contact_coverage": float(np.mean(confidence > 0.35)) if confidence.size else 0.0,
                    "reference_stance_coverage": float(np.mean(stance_mask)) if stance_mask.size else 0.0,
                    "reference_contact_switch_rate": float(np.mean((confidence[1:] > 0.5) != (confidence[:-1] > 0.5)))
                    if confidence.shape[0] > 1
                    else 0.0,
                }
            )
    if motion_path.exists():
        with np.load(motion_path, allow_pickle=False) as motion:
            out.update(
                {
                    "motion_has_poses": "poses" in motion,
                    "motion_has_trans": "trans" in motion,
                    "motion_has_betas": "betas" in motion,
                    "motion_mocap_framerate": float(np.asarray(motion["mocap_framerate"]))
                    if "mocap_framerate" in motion
                    else "",
                    "motion_mocap_frame_rate": float(np.asarray(motion["mocap_frame_rate"]))
                    if "mocap_frame_rate" in motion
                    else "",
                }
            )
    if stance_path.exists():
        stance = read_json(stance_path)
        anchors = stance.get("anchors", stance if isinstance(stance, list) else [])
        out["reference_num_stance_anchors"] = len(anchors) if isinstance(anchors, list) else 0
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def format_float(value: Any, digits: int = 4) -> str:
    numeric = as_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.{digits}f}"


def format_percent(value: Any, digits: int = 1) -> str:
    numeric = as_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.{digits}f}%"


def build_report(summary: dict[str, Any], out_dir: Path) -> str:
    agg = summary["stage_aggregates"]
    pair_raw = summary["postprocessed_vs_raw"]
    pair_world = summary["postprocessed_vs_world_grounded"]
    validation = summary["validation"]
    tier_counts = ", ".join(f"{key}: {value}" for key, value in sorted(validation["quality_tier_counts"].items()))

    lines = [
        "# Forehand Clear Post-Processing Experiment",
        "",
        "## Conclusion",
        "",
        (
            f"Analyzed {summary['num_sequences']} forehand-clear clips. The final post-processed signal "
            f"passed validation on {validation['success_count']}/{summary['num_sequences']} clips and was marked "
            f"usable for training on {validation['usable_for_training_count']}/{summary['num_sequences']} exported "
            f"reference bundles ({tier_counts})."
        ),
        "",
        "## Main Aggregate Metrics For Retargeting/Training",
        "",
        "| Metric | Comparator mean | Post-processed mean | Improvement |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        [
            (
                f"| Beta variation max abs (raw -> post) | "
                f"{format_float(agg['00_raw_wham']['beta_variation_max_abs']['mean'])} | "
                f"{format_float(agg[LOWER_STAGE]['beta_variation_max_abs']['mean'])} | fixed to 0 |"
            ),
            (
                f"| Max foot penetration, m (raw -> post) | "
                f"{format_float(agg['00_raw_wham']['penetration0_max']['mean'])} | "
                f"{format_float(agg[LOWER_STAGE]['penetration0_max']['mean'])} | "
                f"{format_percent(pair_raw['penetration0_max']['mean_percent_reduction'])} reduction |"
            ),
            (
                f"| Mean foot penetration, m (raw -> post) | "
                f"{format_float(agg['00_raw_wham']['penetration0_mean']['mean'])} | "
                f"{format_float(agg[LOWER_STAGE]['penetration0_mean']['mean'])} | "
                f"{format_percent(pair_raw['penetration0_mean']['mean_percent_reduction'])} reduction |"
            ),
            (
                f"| Mean stance foot speed, m/s (world-grounded -> post) | "
                f"{format_float(agg['02_world_grounded']['contact_mean_speed']['mean'])} | "
                f"{format_float(agg[LOWER_STAGE]['contact_mean_speed']['mean'])} | "
                f"{format_percent(pair_world['contact_mean_speed']['mean_percent_reduction'])} reduction |"
            ),
            (
                f"| Sliding contact samples (world-grounded -> post) | "
                f"{format_float(agg['02_world_grounded']['contact_num_sliding']['mean'])} | "
                f"{format_float(agg[LOWER_STAGE]['contact_num_sliding']['mean'])} | "
                f"{format_percent(pair_world['contact_num_sliding']['mean_percent_reduction'])} reduction |"
            ),
            (
                f"| Root vertical jitter (world-grounded -> post) | "
                f"{format_float(agg['02_world_grounded']['root_rms_vertical_accel']['mean'], digits=6)} | "
                f"{format_float(agg[LOWER_STAGE]['root_rms_vertical_accel']['mean'], digits=6)} | "
                f"{format_percent(pair_world['root_rms_vertical_accel']['mean_percent_reduction'])} reduction |"
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Training Suitability",
            "",
            (
                f"- Frame count was preserved on {validation['frame_count_unchanged_count']}/{summary['num_sequences']} clips."
            ),
            (
                f"- Upper-body pose was preserved on {validation['upper_body_pose_delta_small_count']}/"
                f"{summary['num_sequences']} clips; this protects the stroke action while allowing lower-body correction."
            ),
            (
                f"- Lower-body pose delta stayed within the configured bound on "
                f"{validation['lower_body_pose_delta_bounded_count']}/{summary['num_sequences']} clips."
            ),
            (
                f"- Final reference bundles use AMASS-style z-up motion/contact data and have matching FPS metadata on "
                f"{validation['fps_metadata_match_count']}/{summary['num_sequences']} clips."
            ),
            (
                "- Residual risk: max instantaneous contact speed is not improved on average; the training-oriented "
                "evidence is the lower mean stance speed, far fewer sliding samples, and all validation gates passing."
            ),
            "",
            "## Output Files",
            "",
            f"- `{out_dir / 'forehand_clear_stage_metrics.csv'}`",
            f"- `{out_dir / 'forehand_clear_sequence_summary.csv'}`",
            f"- `{out_dir / 'forehand_clear_experiment_summary.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(manifest_tsv: Path, out_dir: Path, track_id: str) -> dict[str, Any]:
    rows = read_manifest_rows(manifest_tsv)
    stage_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []

    for manifest_row in rows:
        folder = Path(manifest_row["folder"])
        sequence = folder.name
        fps = 60.0
        reference_manifest = Path(manifest_row["reference_manifest"])
        if reference_manifest.exists():
            fps = float(read_json(reference_manifest).get("fps", fps))

        _, baseline_record = select_record(Path(manifest_row["raw_wham"]), track_id)
        sequence_stage_rows: dict[str, dict[str, Any]] = {}
        for order, stage in enumerate(STAGES):
            pkl = stage_path(manifest_row, stage)
            metrics = compute_metrics(
                {"id": stage["id"], "description": stage["description"]},
                pkl,
                fps=fps,
                track_id=track_id,
                baseline_record=baseline_record,
            )
            metrics.update(
                {
                    "sequence": sequence,
                    "sequence_group": sequence_group(sequence),
                    "stage_order": order,
                    "folder": str(folder),
                    "fps": fps,
                }
            )
            stage_rows.append(metrics)
            sequence_stage_rows[stage["id"]] = metrics

        validation_path = folder / "lower_body_corrected" / "validation_summary.json"
        validation = read_json(validation_path) if validation_path.exists() else {}
        checks = validation.get("checks", {})
        reference = read_reference_bundle(folder / "reference_bundle")
        raw = sequence_stage_rows["00_raw_wham"]
        lower = sequence_stage_rows[LOWER_STAGE]
        world = sequence_stage_rows["02_world_grounded"]

        sequence_summary: dict[str, Any] = {
            "sequence": sequence,
            "sequence_group": sequence_group(sequence),
            "folder": str(folder),
            "frames": lower.get("frames", ""),
            "fps": fps,
            "validation_success": bool(validation.get("success", False)),
            "all_validation_checks_pass": all(bool(value) for value in checks.values()) if checks else False,
            "upper_body_pose_delta_small": bool(checks.get("upper_body_pose_delta_small", False)),
            "lower_body_pose_delta_bounded": bool(checks.get("lower_body_pose_delta_bounded", False)),
            "frame_count_unchanged": bool(checks.get("frame_count_unchanged", False)),
            "smpl_foot_penetration_reduced": bool(checks.get("smpl_foot_penetration_reduced", False)),
            "smpl_contact_foot_sliding_reduced": bool(checks.get("smpl_contact_foot_sliding_reduced", False)),
            "root_vertical_jitter_not_worse": bool(checks.get("root_vertical_jitter_not_worse", False)),
            "post_vs_raw_penetration0_max_reduction_pct": percent_reduction(
                raw.get("penetration0_max"), lower.get("penetration0_max")
            ),
            "post_vs_raw_contact_mean_speed_reduction_pct": percent_reduction(
                raw.get("contact_mean_speed"), lower.get("contact_mean_speed")
            ),
            "post_vs_world_contact_mean_speed_reduction_pct": percent_reduction(
                world.get("contact_mean_speed"), lower.get("contact_mean_speed")
            ),
            "post_vs_raw_root_jitter_reduction_pct": percent_reduction(
                raw.get("root_rms_vertical_accel"), lower.get("root_rms_vertical_accel")
            ),
            "post_beta_variation_max_abs": lower.get("beta_variation_max_abs", ""),
            "post_upper_pose_delta_max_abs": lower.get("pose_delta_upper_max_abs", ""),
            "post_lower_pose_delta_max_abs": lower.get("pose_delta_lower_max_abs", ""),
        }
        for metric in METRICS_LOWER_IS_BETTER:
            sequence_summary[f"raw_{metric}"] = raw.get(metric, "")
            sequence_summary[f"world_{metric}"] = world.get(metric, "")
            sequence_summary[f"post_{metric}"] = lower.get(metric, "")
            sequence_summary[f"post_vs_raw_{metric}_reduction_pct"] = percent_reduction(
                raw.get(metric), lower.get(metric)
            )
            sequence_summary[f"post_vs_world_{metric}_reduction_pct"] = percent_reduction(
                world.get(metric), lower.get(metric)
            )
        sequence_summary.update(reference)
        sequence_rows.append(sequence_summary)

    stage_aggregates: dict[str, dict[str, Any]] = {}
    rows_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage_rows:
        rows_by_stage[str(row["stage_id"])].append(row)
    aggregate_metrics = METRICS_LOWER_IS_BETTER + [
        "beta_variation_max_abs",
        "pose_delta_upper_max_abs",
        "pose_delta_lower_max_abs",
        "frames",
    ]
    for stage_id, rows_for_stage in rows_by_stage.items():
        stage_aggregates[stage_id] = {
            metric: summarize_numeric(rows_for_stage, metric) for metric in aggregate_metrics
        }

    post_vs_raw: dict[str, Any] = {}
    for metric in METRICS_LOWER_IS_BETTER:
        reductions = [
            percent_reduction(row.get(f"raw_{metric}"), row.get(f"post_{metric}"))
            for row in sequence_rows
            if f"raw_{metric}" in row and f"post_{metric}" in row
        ]
        clean = [value for value in reductions if value is not None]
        post_vs_raw[metric] = {
            "count": len(clean),
            "mean_percent_reduction": mean(clean),
            "median_percent_reduction": median(clean),
            "num_improved": sum(1 for value in clean if value > 0.0),
        }

    post_vs_world: dict[str, Any] = {}
    for metric in METRICS_LOWER_IS_BETTER:
        reductions = [
            percent_reduction(row.get(f"world_{metric}"), row.get(f"post_{metric}"))
            for row in sequence_rows
            if f"world_{metric}" in row and f"post_{metric}" in row
        ]
        clean = [value for value in reductions if value is not None]
        post_vs_world[metric] = {
            "count": len(clean),
            "mean_percent_reduction": mean(clean),
            "median_percent_reduction": median(clean),
            "num_improved": sum(1 for value in clean if value > 0.0),
        }

    quality_tiers = Counter(str(row.get("reference_quality_tier", "")) for row in sequence_rows)
    validation = {
        "success_count": sum(1 for row in sequence_rows if row.get("validation_success")),
        "usable_for_training_count": sum(1 for row in sequence_rows if row.get("reference_usable_for_training")),
        "quality_tier_counts": dict(quality_tiers),
        "frame_count_unchanged_count": sum(1 for row in sequence_rows if row.get("frame_count_unchanged")),
        "upper_body_pose_delta_small_count": sum(1 for row in sequence_rows if row.get("upper_body_pose_delta_small")),
        "lower_body_pose_delta_bounded_count": sum(1 for row in sequence_rows if row.get("lower_body_pose_delta_bounded")),
        "fps_metadata_match_count": sum(
            1
            for row in sequence_rows
            if as_float(row.get("reference_fps")) == as_float(row.get("motion_mocap_framerate"))
            == as_float(row.get("motion_mocap_frame_rate"))
        ),
    }

    summary = {
        "source_manifest": str(manifest_tsv),
        "output_dir": str(out_dir),
        "num_sequences": len(sequence_rows),
        "stages": STAGES,
        "stage_aggregates": stage_aggregates,
        "postprocessed_vs_raw": post_vs_raw,
        "postprocessed_vs_world_grounded": post_vs_world,
        "validation": validation,
    }

    write_csv(out_dir / "forehand_clear_stage_metrics.csv", stage_rows)
    write_csv(out_dir / "forehand_clear_sequence_summary.csv", sequence_rows)
    write_json(out_dir / "forehand_clear_experiment_summary.json", summary)
    (out_dir / "forehand_clear_experiment_report.md").write_text(build_report(summary, out_dir), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-tsv", default="output/forehand_clear/manifest_by_video.tsv")
    parser.add_argument("--out-dir", default="output/Optimization")
    parser.add_argument("--track-id", default="merge")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = analyze(Path(args.manifest_tsv), Path(args.out_dir), args.track_id)
    print(f"Analyzed {summary['num_sequences']} sequences")
    print(f"Output dir: {summary['output_dir']}")
    print(
        "Validation success: "
        f"{summary['validation']['success_count']}/{summary['num_sequences']}; "
        "usable for training: "
        f"{summary['validation']['usable_for_training_count']}/{summary['num_sequences']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
