#!/usr/bin/env python3
"""Bridge optimized_wham outputs with the MuscleMimic datasets layout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.video_fps import detect_video_fps  # noqa: E402


OPTIMIZED_WHAM_ROOT = Path("/data3/yangfeiyang/WorkSpace/optimized_wham")
MUSCLEMIMIC_ROOT = Path("/data3/yangfeiyang/WorkSpace/musclemimic")
DATASETS_ROOT = MUSCLEMIMIC_ROOT / "datasets"

CANONICAL_WHAM_STAGES = {"raw_wham", "optimized_wham"}
CANONICAL_TRAJECTORY_STAGES = {"raw", "optimized"}
AMASS_TRAJECTORY_CONTAINERS = {"amass_npz", "amass_npz_legacy"}
GMR_TRAJECTORY_CONTAINERS = {"gmr_cache"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class Operation:
    kind: str
    source: str | None
    destination: str
    reason: str


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: list[str]
    cwd: Path
    output: Path | None = None


@dataclass(frozen=True)
class DatasetPipeline:
    action_dir: Path
    video: Path
    sequence: str
    raw_wham_dir: Path
    optimized_wham_dir: Path
    raw_wham_pkl: Path
    canonical_pkl: Path
    optimized_pkl: Path
    reference_bundle_dir: Path
    raw_motion: str
    optimized_motion: str
    raw_amass: Path
    optimized_amass: Path
    raw_manifest: Path
    optimized_manifest: Path
    raw_gmr_root: Path
    optimized_gmr_root: Path
    steps: list[PipelineStep]

    @property
    def commands(self) -> list[list[str]]:
        return [step.command for step in self.steps]


def _is_action_dir(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith("_") and path.name != "\u793a\u8303"


def _action_dirs(datasets_root: Path, actions: Sequence[str] | None) -> list[Path]:
    if actions:
        return [datasets_root / action for action in actions]
    if not datasets_root.exists():
        return []
    return sorted(path for path in datasets_root.iterdir() if _is_action_dir(path))


def required_contract_dirs(action_dir: Path) -> list[Path]:
    return [
        action_dir / "raw_video",
        action_dir / "wham" / "raw_wham",
        action_dir / "wham" / "optimized_wham",
        action_dir / "muscle_trajectory" / "raw",
        action_dir / "muscle_trajectory" / "optimized",
        action_dir / "manifests",
    ]


def plan_contract_operations(datasets_root: Path, actions: Sequence[str] | None = None) -> list[Operation]:
    operations: list[Operation] = []
    for action_dir in _action_dirs(datasets_root, actions):
        for directory in required_contract_dirs(action_dir):
            if not directory.exists():
                operations.append(
                    Operation(
                        kind="mkdir",
                        source=None,
                        destination=str(directory),
                        reason="required dataset contract directory",
                    )
                )
        operations.extend(_plan_wham_stage_migrations(action_dir))
        operations.extend(_plan_remaining_wham_child_archives(action_dir))
        operations.extend(_plan_trajectory_stage_migrations(action_dir))
    return operations


def plan_compact_trajectory_operations(datasets_root: Path, actions: Sequence[str] | None = None) -> list[Operation]:
    operations: list[Operation] = []
    for action_dir in _action_dirs(datasets_root, actions):
        for stage in ("raw", "optimized"):
            operations.extend(_plan_compact_trajectory_stage(action_dir, stage))
    return operations


def _plan_compact_trajectory_stage(action_dir: Path, stage: str) -> list[Operation]:
    stage_root = action_dir / "muscle_trajectory" / stage
    if not stage_root.exists():
        return []

    operations: list[Operation] = []
    dirs_to_prune: set[Path] = set()
    for source in sorted(stage_root.rglob("*")):
        if source.is_dir():
            dirs_to_prune.add(source)
            continue
        rel = source.relative_to(stage_root)
        if source.parent == stage_root and _is_final_trajectory_npz(source):
            continue

        destination = _compact_trajectory_destination(action_dir, stage, rel, source)
        if source.resolve() == destination.resolve():
            continue
        operations.append(
            Operation(
                kind="merge_dir",
                source=str(source),
                destination=str(destination),
                reason=_compact_trajectory_reason(stage_root, destination),
            )
        )

    for directory in sorted(dirs_to_prune, key=lambda path: len(path.parts), reverse=True):
        operations.append(
            Operation(
                kind="rmdir_empty",
                source=str(directory),
                destination=str(directory),
                reason="remove empty compacted trajectory directory",
            )
        )
    return operations


def _compact_trajectory_destination(action_dir: Path, stage: str, rel: Path, source: Path) -> Path:
    stage_root = action_dir / "muscle_trajectory" / stage
    first = rel.parts[0]
    if first in AMASS_TRAJECTORY_CONTAINERS:
        wham_stage = "raw_wham" if stage == "raw" else "optimized_wham"
        return action_dir / "wham" / wham_stage / "_amass_intermediate" / rel
    if first in GMR_TRAJECTORY_CONTAINERS and _is_final_trajectory_npz(source):
        return stage_root / source.name
    if _is_final_trajectory_npz(source):
        return stage_root / source.name
    return action_dir / "manifests" / "legacy_muscle_trajectory" / stage / rel


def _compact_trajectory_reason(stage_root: Path, destination: Path) -> str:
    destination_text = str(destination)
    if "_amass_intermediate" in destination_text:
        return "preserve AMASS intermediate outside muscle_trajectory"
    if destination.parent == stage_root:
        return "flatten final trajectory file into stage root"
    return "archive non-trajectory muscle_trajectory artifact"


def _is_final_trajectory_npz(path: Path) -> bool:
    return path.suffix.lower() == ".npz" and not path.stem.lower().endswith("_analysis")


def _plan_wham_stage_migrations(action_dir: Path) -> list[Operation]:
    wham_root = action_dir / "wham"
    if not wham_root.exists():
        return []

    operations: list[Operation] = []
    legacy_stages = {
        "stage1": "raw_wham",
        "stage1_raw": "raw_wham",
        "raw": "raw_wham",
        "best": "optimized_wham",
        "optimized": "optimized_wham",
    }
    for legacy_name, stage in legacy_stages.items():
        source = wham_root / legacy_name
        if source.exists():
            operations.append(
                Operation(
                    kind="merge_dir",
                    source=str(source),
                    destination=str(wham_root / stage),
                    reason=f"legacy WHAM stage {legacy_name} -> {stage}",
                )
            )

    ignored = set(legacy_stages) | CANONICAL_WHAM_STAGES
    for child in sorted(path for path in wham_root.iterdir() if path.is_dir() and path.name not in ignored):
        for stage in sorted(CANONICAL_WHAM_STAGES):
            source = child / stage
            if source.exists():
                operations.append(
                    Operation(
                        kind="merge_dir",
                        source=str(source),
                        destination=str(wham_root / stage / child.name),
                        reason=f"lift nested WHAM stage {child.name}/{stage}",
                    )
                )
    operations.extend(_plan_recursive_legacy_wham_stages(wham_root))
    return operations


def _plan_recursive_legacy_wham_stages(wham_root: Path) -> list[Operation]:
    operations: list[Operation] = []
    planned_sources: list[Path] = []
    for source in sorted((path for path in wham_root.rglob("*") if path.is_dir()), key=lambda p: len(p.parts)):
        if _is_in_canonical_wham_stage(source, wham_root):
            continue
        if source.parent == wham_root and source.name in {"stage1", "stage1_raw", "raw", "best", "optimized"}:
            continue
        if any(parent in planned_sources for parent in source.parents):
            continue

        stage = _classify_legacy_wham_stage(source.name)
        if stage is None:
            continue
        destination = wham_root / stage / _legacy_wham_destination_rel(source, wham_root)
        operations.append(
            Operation(
                kind="merge_dir",
                source=str(source),
                destination=str(destination),
                reason=f"recursive legacy WHAM stage {source.relative_to(wham_root)} -> {stage}",
            )
        )
        planned_sources.append(source)

    for source in sorted(path for path in wham_root.rglob("*") if path.is_file()):
        if _is_in_canonical_wham_stage(source, wham_root):
            continue
        if any(parent in planned_sources for parent in source.parents):
            continue
        stage = _classify_loose_wham_file(source.name)
        if stage is None:
            continue
        rel = source.relative_to(wham_root)
        operations.append(
            Operation(
                kind="merge_dir",
                source=str(source),
                destination=str(wham_root / stage / rel),
                reason=f"loose WHAM file {rel} -> {stage}",
            )
        )
    return operations


def _plan_remaining_wham_child_archives(action_dir: Path) -> list[Operation]:
    wham_root = action_dir / "wham"
    if not wham_root.exists():
        return []

    operations: list[Operation] = []
    for child in sorted(path for path in wham_root.iterdir() if path.is_dir()):
        if child.name in CANONICAL_WHAM_STAGES:
            continue
        if _is_empty_dir(child):
            operations.append(
                Operation(
                    kind="rmdir_empty",
                    source=str(child),
                    destination=str(child),
                    reason="remove empty noncanonical WHAM directory",
                )
            )
            continue
        operations.append(
            Operation(
                kind="merge_dir",
                source=str(child),
                destination=str(action_dir / "manifests" / "legacy_wham" / child.name),
                reason="archive remaining noncanonical WHAM directory",
            )
        )
    return operations


def _is_empty_dir(path: Path) -> bool:
    try:
        next(path.iterdir())
    except StopIteration:
        return True
    except OSError:
        return False
    return False


def _is_in_canonical_wham_stage(path: Path, wham_root: Path) -> bool:
    canonical_roots = [wham_root / stage for stage in CANONICAL_WHAM_STAGES]
    return any(path == root or root in path.parents for root in canonical_roots)


def _classify_legacy_wham_stage(name: str) -> str | None:
    lower = name.lower()
    if lower in {"stage1", "stage1_raw", "raw", "raw_video"} or lower.startswith("00_raw_wham"):
        return "raw_wham"
    if lower in {"best", "optimized"}:
        return "optimized_wham"
    if lower in {"01_fixed_beta"}:
        return "optimized_wham"
    if lower.startswith(("stage2", "stage3", "stage4", "stage5", "stage6", "stage7")):
        return "optimized_wham"
    if lower.startswith(("02_", "03_", "04_", "05_", "06_", "07_")):
        return "optimized_wham"
    return None


def _classify_loose_wham_file(name: str) -> str | None:
    lower = name.lower()
    if lower == "wham_output.pkl":
        return "raw_wham"
    if lower in {"canonical_wham_output.pkl", "optimized_canonical_wham_output.pkl", "corrected_smpl.pkl"}:
        return "optimized_wham"
    return None


def _legacy_wham_destination_rel(source: Path, wham_root: Path) -> Path:
    rel_parent = source.parent.relative_to(wham_root)
    if source.name in {"stage1", "stage1_raw", "raw", "best", "optimized"}:
        return rel_parent
    return rel_parent / source.name


def _plan_trajectory_stage_migrations(action_dir: Path) -> list[Operation]:
    trajectory_root = action_dir / "muscle_trajectory"
    if not trajectory_root.exists():
        return []

    operations: list[Operation] = []
    for container in ("amass_npz", "gmr_cache", "amass_npz_legacy"):
        container_root = trajectory_root / container
        if not container_root.exists():
            continue
        for family in sorted(path for path in container_root.iterdir() if path.is_dir()):
            for legacy_name, stage in (("raw", "raw"), ("best", "optimized"), ("optimized", "optimized")):
                source = family / legacy_name
                if source.exists():
                    operations.append(
                        Operation(
                            kind="merge_dir",
                            source=str(source),
                            destination=str(trajectory_root / stage / container / family.name),
                            reason=f"legacy trajectory {container}/{family.name}/{legacy_name} -> {stage}",
                        )
                    )

    ignored = CANONICAL_TRAJECTORY_STAGES | {"amass_npz", "gmr_cache", "amass_npz_legacy"}
    for child in sorted(path for path in trajectory_root.iterdir() if path.is_dir() and path.name not in ignored):
        stage = _infer_trajectory_stage(child)
        operations.append(
            Operation(
                kind="merge_dir",
                source=str(child),
                destination=str(trajectory_root / stage / child.name),
                reason=f"classified legacy trajectory group as {stage}",
            )
        )
    return operations


def _infer_trajectory_stage(path: Path) -> str:
    text = path.name.lower()
    if "raw" in text or "stage1" in text:
        return "raw"
    if "best" in text or "optimized" in text:
        return "optimized"
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        name = file_path.name.lower()
        if "best" in name or "optimized" in name or "lower_body" in name or "corrected" in name:
            return "optimized"
        if "raw" in name or "stage1" in name:
            return "raw"
    return "optimized"


def apply_operations(operations: Sequence[Operation], *, write: bool) -> None:
    for operation in operations:
        if not write:
            print(_format_operation(operation))
            continue
        if operation.kind == "mkdir":
            Path(operation.destination).mkdir(parents=True, exist_ok=True)
        elif operation.kind == "merge_dir":
            if operation.source is None:
                raise ValueError(f"merge_dir operation missing source: {operation}")
            _merge_path(Path(operation.source), Path(operation.destination))
        elif operation.kind == "rmdir_empty":
            if operation.source is None:
                raise ValueError(f"rmdir_empty operation missing source: {operation}")
            try:
                Path(operation.source).rmdir()
            except OSError:
                pass
        else:
            raise ValueError(f"Unsupported operation kind: {operation.kind}")


def _format_operation(operation: Operation) -> str:
    if operation.source is None:
        return f"{operation.kind}: {operation.destination}  # {operation.reason}"
    return f"{operation.kind}: {operation.source} -> {operation.destination}  # {operation.reason}"


def _merge_path(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if source.resolve() == destination.resolve():
        return
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for child in sorted(source.iterdir(), key=lambda p: p.name):
            _move_entry(child, destination / child.name)
        _prune_empty_parents(source)
        return
    _move_entry(source, destination)


def _move_entry(source: Path, destination: Path) -> None:
    if source.is_dir() and destination.exists() and destination.is_dir():
        _merge_path(source, destination)
        return
    target = _available_destination(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


def _available_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination
    for index in range(1, 10_000):
        candidate = destination.with_name(f"{destination.stem}__dup{index}{destination.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find available destination for {destination}")


def _prune_empty_parents(path: Path) -> None:
    current = path
    while True:
        try:
            current.rmdir()
        except OSError:
            return
        parent = current.parent
        if parent == current or parent.name in {"wham", "muscle_trajectory"}:
            return
        current = parent


def iter_raw_videos(action_dir: Path) -> list[Path]:
    raw_video_root = action_dir / "raw_video"
    if not raw_video_root.exists():
        return []
    return sorted(
        path
        for path in raw_video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def sequence_name_for_video(action_dir: Path, video: Path) -> str:
    same_stem_count = sum(1 for candidate in iter_raw_videos(action_dir) if candidate.stem == video.stem)
    if same_stem_count <= 1:
        return video.stem

    raw_video_root = action_dir / "raw_video"
    try:
        relative = video.relative_to(raw_video_root).with_suffix("")
    except ValueError:
        return video.stem
    return "__".join(relative.parts)


def build_pipeline(
    *,
    action_dir: Path,
    video: Path,
    sequence_name: str | None = None,
    optimized_wham_root: Path,
    musclemimic_root: Path,
    fps: float | None,
    device: str,
    pose_backend: str,
    optimize_lower_body: bool,
    track_id: str = "merge",
    gender: str = "neutral",
    quality_tier: str = "B",
    project_stance_contacts: bool = True,
    python_exe: str | None = None,
    wham_python_exe: str | None = None,
    musclemimic_python_exe: str | None = None,
) -> DatasetPipeline:
    default_python = python_exe or sys.executable
    wham_python = wham_python_exe or default_python
    musclemimic_python = musclemimic_python_exe or default_python
    sequence = sequence_name or video.stem

    # Auto-detect the true video frame rate so the whole pipeline is time-correct with no
    # per-dataset --fps. A wrong fps silently time-scales the motion (a 30 fps clip run at
    # 60 fps comes out half-length). Explicit --fps still overrides.
    if fps is None:
        detected = detect_video_fps(video)
        if detected is None:
            raise SystemExit(
                f"Could not detect fps for {video}; pass --fps explicitly."
            )
        fps = float(detected)
        print(f"[fps] {sequence}: auto-detected {fps:g} fps from {video.name}")

    raw_wham_root = action_dir / "wham" / "raw_wham"
    raw_wham_dir = raw_wham_root / sequence
    raw_wham_pkl = raw_wham_dir / "wham_output.pkl"

    optimized_wham_dir = action_dir / "wham" / "optimized_wham" / sequence
    canonical_pkl = optimized_wham_dir / "canonical_wham_output.pkl"
    fixed_beta_report = optimized_wham_dir / "fixed_beta_report.json"
    world_grounded_dir = optimized_wham_dir / "world_grounded"
    world_grounded_pkl = world_grounded_dir / "optimized_canonical_wham_output.pkl"
    lower_body_dir = optimized_wham_dir / "lower_body_corrected"
    lower_body_pkl = lower_body_dir / "corrected_smpl.pkl"
    optimized_pkl = lower_body_pkl if optimize_lower_body else world_grounded_pkl
    reference_bundle_dir = optimized_wham_dir / "reference_bundle"

    raw_stage_root = action_dir / "muscle_trajectory" / "raw"
    optimized_stage_root = action_dir / "muscle_trajectory" / "optimized"
    raw_motion = sequence
    optimized_motion = sequence
    raw_amass = raw_wham_root / f"{sequence}.npz"
    optimized_amass = action_dir / "wham" / "optimized_wham" / f"{sequence}.npz"
    raw_gmr_root = raw_stage_root
    optimized_gmr_root = optimized_stage_root
    raw_manifest = action_dir / "manifests" / "raw" / f"{sequence}.txt"
    optimized_manifest = action_dir / "manifests" / "optimized" / f"{sequence}.txt"

    steps = [
        PipelineStep(
            name="wham_raw",
            command=[
                wham_python,
                str(optimized_wham_root / "demo.py"),
                "--video",
                str(video),
                "--output_pth",
                str(raw_wham_root),
                "--sequence-name",
                sequence,
                "--save_pkl",
                "--pose-backend",
                pose_backend,
            ],
            cwd=optimized_wham_root,
            output=raw_wham_pkl,
        ),
        PipelineStep(
            name="fixed_beta",
            command=[
                wham_python,
                str(optimized_wham_root / "scripts" / "canonicalize_wham_fixed_beta.py"),
                "--wham-pkl",
                str(raw_wham_pkl),
                "--out-pkl",
                str(canonical_pkl),
                "--report",
                str(fixed_beta_report),
                "--beta-out",
                str(optimized_wham_dir / "beta_fixed.npy"),
                "--tracking-results",
                str(raw_wham_dir / "tracking_results.pth"),
                "--device",
                device,
            ],
            cwd=optimized_wham_root,
            output=canonical_pkl,
        ),
        PipelineStep(
            name="world_grounded",
            command=[
                wham_python,
                str(optimized_wham_root / "scripts" / "world_grounded_smpl_optimizer.py"),
                "--input-pkl",
                str(canonical_pkl),
                "--out-dir",
                str(world_grounded_dir),
                "--fps",
                str(fps),
                "--track-id",
                track_id,
                "--device",
                device,
            ],
            cwd=optimized_wham_root,
            output=world_grounded_pkl,
        ),
    ]

    if optimize_lower_body:
        steps.append(
            PipelineStep(
                name="lower_body",
                command=[
                    wham_python,
                    str(optimized_wham_root / "scripts" / "optimize_smpl_lower_body.py"),
                    "--input-pkl",
                    str(world_grounded_pkl),
                    "--out-dir",
                    str(lower_body_dir),
                    "--fps",
                    str(fps),
                    "--track-id",
                    track_id,
                    "--device",
                    device,
                    "--enable-pose-pass",
                ],
                cwd=optimized_wham_root,
                output=lower_body_pkl,
            )
        )

    quality_report = (
        lower_body_dir / "validation_summary.json"
        if optimize_lower_body
        else world_grounded_dir / "quality_report.json"
    )
    steps.extend(
        [
            PipelineStep(
                name="reference_bundle",
                command=[
                    wham_python,
                    str(optimized_wham_root / "scripts" / "export_contact_preserving_reference.py"),
                    "--input-pkl",
                    str(optimized_pkl),
                    "--out-dir",
                    str(reference_bundle_dir),
                    "--sequence",
                    sequence,
                    "--fps",
                    str(fps),
                    "--track-id",
                    track_id,
                    "--quality-report",
                    str(quality_report),
                    "--source-json",
                    str(reference_bundle_dir / "source.json"),
                ],
                cwd=optimized_wham_root,
                output=reference_bundle_dir / "manifest.json",
            ),
            PipelineStep(
                name="raw_wham_to_amass",
                command=[
                    musclemimic_python,
                    str(musclemimic_root / "BadmintonMimic" / "scripts" / "convert_wham_to_amass.py"),
                    "--input",
                    str(raw_wham_pkl),
                    "--output",
                    str(raw_amass),
                    "--fps",
                    str(fps),
                    "--force-fps",
                    "--gender",
                    gender,
                    "--merge-tracks",
                ],
                cwd=musclemimic_root,
                output=raw_amass,
            ),
            PipelineStep(
                name="optimized_wham_to_amass",
                command=[
                    musclemimic_python,
                    str(musclemimic_root / "BadmintonMimic" / "scripts" / "convert_wham_to_amass.py"),
                    "--input",
                    str(optimized_pkl),
                    "--output",
                    str(optimized_amass),
                    "--fps",
                    str(fps),
                    "--force-fps",
                    "--gender",
                    gender,
                    "--merge-tracks",
                ],
                cwd=musclemimic_root,
                output=optimized_amass,
            ),
            PipelineStep(
                name="retarget_raw",
                command=[
                    musclemimic_python,
                    str(musclemimic_root / "BadmintonMimic" / "scripts" / "run_retarget.py"),
                    "--manifest",
                    str(raw_manifest),
                    "--amass-root",
                    str(raw_wham_root),
                    "--gmr-cache-root",
                    str(raw_gmr_root),
                    "--fps",
                    str(int(fps) if float(fps).is_integer() else fps),
                ],
                cwd=musclemimic_root,
                output=raw_gmr_root / f"{sequence}.npz",
            ),
            PipelineStep(
                name="retarget_optimized",
                command=[
                    musclemimic_python,
                    str(musclemimic_root / "BadmintonMimic" / "scripts" / "run_retarget.py"),
                    "--manifest",
                    str(optimized_manifest),
                    "--amass-root",
                    str(action_dir / "wham" / "optimized_wham"),
                    "--gmr-cache-root",
                    str(optimized_gmr_root),
                    "--fps",
                    str(int(fps) if float(fps).is_integer() else fps),
                    *(
                        ["--stance-bundle", str(reference_bundle_dir / "manifest.json")]
                        if project_stance_contacts
                        else []
                    ),
                ],
                cwd=musclemimic_root,
                output=optimized_gmr_root / f"{sequence}.npz",
            ),
        ]
    )

    return DatasetPipeline(
        action_dir=action_dir,
        video=video,
        sequence=sequence,
        raw_wham_dir=raw_wham_dir,
        optimized_wham_dir=optimized_wham_dir,
        raw_wham_pkl=raw_wham_pkl,
        canonical_pkl=canonical_pkl,
        optimized_pkl=optimized_pkl,
        reference_bundle_dir=reference_bundle_dir,
        raw_motion=raw_motion,
        optimized_motion=optimized_motion,
        raw_amass=raw_amass,
        optimized_amass=optimized_amass,
        raw_manifest=raw_manifest,
        optimized_manifest=optimized_manifest,
        raw_gmr_root=raw_gmr_root,
        optimized_gmr_root=optimized_gmr_root,
        steps=steps,
    )


def execute_pipeline(pipeline: DatasetPipeline, *, run: bool, force: bool) -> None:
    if not run:
        _print_pipeline(pipeline)
        return

    _ensure_pipeline_dirs(pipeline)
    _write_reference_source(pipeline)
    for step in pipeline.steps:
        if step.name == "retarget_raw":
            _upsert_manifest(pipeline.raw_manifest, pipeline.raw_motion)
        elif step.name == "retarget_optimized":
            _upsert_manifest(pipeline.optimized_manifest, pipeline.optimized_motion)

        if not force and step.output is not None and step.output.exists():
            print(f"[SKIP] {step.name}: {step.output}")
            continue
        print("$ " + " ".join(step.command), flush=True)
        subprocess.run(step.command, cwd=str(step.cwd), check=True)


def _ensure_pipeline_dirs(pipeline: DatasetPipeline) -> None:
    for directory in required_contract_dirs(pipeline.action_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for directory in (
        pipeline.optimized_wham_dir,
        pipeline.raw_amass.parent,
        pipeline.optimized_amass.parent,
        pipeline.raw_gmr_root,
        pipeline.optimized_gmr_root,
        pipeline.raw_manifest.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _write_reference_source(pipeline: DatasetPipeline) -> None:
    pipeline.reference_bundle_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": str(pipeline.video),
        "raw_wham_pkl": str(pipeline.raw_wham_pkl),
        "canonical_pkl": str(pipeline.canonical_pkl),
        "optimized_pkl": str(pipeline.optimized_pkl),
    }
    (pipeline.reference_bundle_dir / "source.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _upsert_manifest(path: Path, motion: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if path.exists():
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if motion not in lines:
        lines.append(motion)
    path.write_text("\n".join(line for line in lines if line) + "\n", encoding="utf-8")


def _print_pipeline(pipeline: DatasetPipeline) -> None:
    print(f"video: {pipeline.video}")
    print(f"raw_wham: {pipeline.raw_wham_dir}")
    print(f"optimized_wham: {pipeline.optimized_wham_dir}")
    print(f"raw_trajectory: {pipeline.action_dir / 'muscle_trajectory' / 'raw'}")
    print(f"optimized_trajectory: {pipeline.action_dir / 'muscle_trajectory' / 'optimized'}")
    for step in pipeline.steps:
        print(f"[{step.name}] cwd={step.cwd}")
        print("  $ " + " ".join(step.command))


def _parse_action_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    contract = subparsers.add_parser("contract", help="Create or migrate action directories to the dataset contract.")
    contract.add_argument("--datasets-root", type=Path, default=DATASETS_ROOT)
    contract.add_argument("--actions", default=None, help="Comma-separated action names. Defaults to every action dir.")
    contract.add_argument("--write", action="store_true", help="Apply operations. Default only prints a dry-run.")
    contract.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")

    compact = subparsers.add_parser(
        "compact-trajectories",
        help="Keep only final trajectory files under muscle_trajectory stage roots.",
    )
    compact.add_argument("--datasets-root", type=Path, default=DATASETS_ROOT)
    compact.add_argument("--actions", default=None, help="Comma-separated action names. Defaults to every action dir.")
    compact.add_argument("--write", action="store_true", help="Apply operations. Default only prints a dry-run.")
    compact.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")

    pipeline = subparsers.add_parser("pipeline", help="Build or run the optimized_wham -> MuscleMimic pipeline.")
    pipeline.add_argument("--datasets-root", type=Path, default=DATASETS_ROOT)
    pipeline.add_argument("--action", required=True)
    pipeline.add_argument("--video", type=Path, default=None, help="Specific raw video. Defaults to all videos in the action.")
    pipeline.add_argument("--optimized-wham-root", type=Path, default=OPTIMIZED_WHAM_ROOT)
    pipeline.add_argument("--musclemimic-root", type=Path, default=MUSCLEMIMIC_ROOT)
    pipeline.add_argument(
        "--python-exe",
        default=None,
        help="Fallback Python for all child steps when stage-specific interpreters are not set.",
    )
    pipeline.add_argument(
        "--wham-python-exe",
        default=None,
        help="Python used for optimized_wham demo/optimization/reference-bundle steps.",
    )
    pipeline.add_argument(
        "--musclemimic-python-exe",
        default=None,
        help="Python used for WHAM->AMASS conversion and MuscleMimic retargeting steps.",
    )
    pipeline.add_argument("--fps", type=float, default=None,
                          help="FPS override; if omitted, auto-detected from each source video.")
    pipeline.add_argument("--device", default="cuda")
    pipeline.add_argument("--pose-backend", choices=["vitpose", "rtmpose"], default="rtmpose")
    pipeline.add_argument("--track-id", default="merge")
    pipeline.add_argument("--gender", default="neutral")
    pipeline.add_argument("--quality-tier", choices=["A", "B", "C", "all"], default="B")
    pipeline.add_argument("--optimize-lower-body", action="store_true")
    pipeline.add_argument(
        "--stance-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project stance contacts onto MyoFullBody after GMR (OmniRetarget-style "
        "foot-sticking, using the reference bundle). Enabled by default; use "
        "--no-stance-projection to fall back to plain GMR.",
    )
    pipeline.add_argument("--limit", type=int, default=None)
    pipeline.add_argument("--run", action="store_true", help="Execute commands. Default only prints a dry-run.")
    pipeline.add_argument("--force", action="store_true", help="Re-run steps even when their output exists.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "contract":
        actions = _parse_action_list(args.actions)
        operations = plan_contract_operations(args.datasets_root, actions=actions)
        apply_operations(operations, write=args.write)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps([asdict(operation) for operation in operations], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        print(f"operations={len(operations)}")
        return 0

    if args.command == "compact-trajectories":
        actions = _parse_action_list(args.actions)
        operations = plan_compact_trajectory_operations(args.datasets_root, actions=actions)
        apply_operations(operations, write=args.write)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps([asdict(operation) for operation in operations], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        print(f"operations={len(operations)}")
        return 0

    if args.command == "pipeline":
        action_dir = args.datasets_root / args.action
        videos = [args.video] if args.video is not None else iter_raw_videos(action_dir)
        if args.limit is not None:
            videos = videos[: args.limit]
        if not videos:
            raise SystemExit(f"No raw videos found for action: {action_dir}")
        for video in videos:
            pipeline = build_pipeline(
                action_dir=action_dir,
                video=video,
                sequence_name=sequence_name_for_video(action_dir, video),
                optimized_wham_root=args.optimized_wham_root,
                musclemimic_root=args.musclemimic_root,
                fps=args.fps,
                device=args.device,
                pose_backend=args.pose_backend,
                optimize_lower_body=args.optimize_lower_body,
                track_id=args.track_id,
                gender=args.gender,
                quality_tier=args.quality_tier,
                project_stance_contacts=args.stance_projection,
                python_exe=args.python_exe,
                wham_python_exe=args.wham_python_exe,
                musclemimic_python_exe=args.musclemimic_python_exe,
            )
            execute_pipeline(pipeline, run=args.run, force=args.force)
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
