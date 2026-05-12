# Stage7 Whole-Body Smoothness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative Stage7 whole-body smoothness pass that improves jitter only when SMPL and OpenSim validation gates do not regress.

**Architecture:** Add focused utilities for temporal smoothness metrics, OpenSim `.mot` coordinate validation, Stage7 candidate selection, and a CLI script that generates a smooth candidate, runs OpenSim validation, and writes `selected_smooth_smpl.pkl` with rollback. Integrate the script into the main video pipeline and ablation runner after Stage5/Stage6.

**Tech Stack:** Python stdlib, `numpy`, `joblib`, existing `lib.smpl_optimization` helpers, existing `scripts/retarget_smpl_to_opensim.py`, pytest.

---

## File Structure

- Modify `lib/smpl_optimization/metrics.py`: add generic temporal derivative metrics for root and pose arrays.
- Create `lib/smpl_optimization/opensim_motion.py`: parse OpenSim `.mot` files and compute coordinate range/jump/smoothness metrics.
- Create `lib/smpl_optimization/stage7_selection.py`: implement hard gates, improvement gates, and rollback decision.
- Create `lib/smpl_optimization/whole_body_smooth.py`: generate a conservative smooth candidate using bounded residual smoothing.
- Create `scripts/whole_body_smoothness_optimizer.py`: CLI orchestration for candidate generation, OpenSim validation, and selected output.
- Modify `scripts/video_to_fixed_smpl_to_opensim.py`: add `--whole-body-smooth` pipeline stage.
- Modify `scripts/run_wham_ablation.py`: add optional Stage7 group after lower-body or Stage6 selected output.
- Create tests under `tests/smpl_optimization/` for each new module and CLI command construction.

---

### Task 1: Add Smoothness Metrics

**Files:**
- Modify: `lib/smpl_optimization/metrics.py`
- Test: `tests/smpl_optimization/test_stage7_smoothness_metrics.py`

- [ ] **Step 1: Write failing tests for temporal derivative metrics**

Create `tests/smpl_optimization/test_stage7_smoothness_metrics.py`:

```python
import numpy as np

from lib.smpl_optimization.metrics import (
    pose_smoothness,
    root_translation_smoothness,
    temporal_derivative_summary,
)


def test_temporal_derivative_summary_reports_zero_for_constant_signal():
    values = np.ones((6, 3), dtype=np.float32)
    summary = temporal_derivative_summary(values, order=3)
    assert summary == {"rms": 0.0, "max_abs": 0.0}


def test_pose_smoothness_detects_jerky_pose():
    smooth = np.zeros((8, 72), dtype=np.float32)
    jerky = smooth.copy()
    jerky[4, 10] = 1.0

    smooth_report = pose_smoothness(smooth)
    jerky_report = pose_smoothness(jerky)

    assert jerky_report["jerk"]["rms"] > smooth_report["jerk"]["rms"]
    assert jerky_report["acceleration"]["max_abs"] > 0.0


def test_root_translation_smoothness_handles_short_sequences():
    report = root_translation_smoothness(np.zeros((2, 3), dtype=np.float32))
    assert report["velocity"]["rms"] == 0.0
    assert report["acceleration"]["rms"] == 0.0
    assert report["jerk"]["rms"] == 0.0
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_stage7_smoothness_metrics.py -q
```

Expected: FAIL because `pose_smoothness`, `root_translation_smoothness`, and `temporal_derivative_summary` are not defined.

- [ ] **Step 3: Implement metric helpers**

Append to `lib/smpl_optimization/metrics.py`:

```python
def temporal_derivative_summary(values, order=1):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or values.shape[0] <= int(order):
        return {"rms": 0.0, "max_abs": 0.0}

    flat = values.reshape(values.shape[0], -1)
    derivative = np.diff(flat, n=int(order), axis=0)
    if derivative.size == 0:
        return {"rms": 0.0, "max_abs": 0.0}

    return {
        "rms": _clean_float(np.sqrt(np.mean(derivative**2))),
        "max_abs": _clean_float(np.max(np.abs(derivative))),
    }


def pose_smoothness(pose):
    pose = np.asarray(pose)
    return {
        "velocity": temporal_derivative_summary(pose, order=1),
        "acceleration": temporal_derivative_summary(pose, order=2),
        "jerk": temporal_derivative_summary(pose, order=3),
    }


def root_translation_smoothness(trans):
    trans = np.asarray(trans)
    return {
        "velocity": temporal_derivative_summary(trans, order=1),
        "acceleration": temporal_derivative_summary(trans, order=2),
        "jerk": temporal_derivative_summary(trans, order=3),
    }
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_stage7_smoothness_metrics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add lib\smpl_optimization\metrics.py tests\smpl_optimization\test_stage7_smoothness_metrics.py
git commit -m "feat: add stage7 smoothness metrics"
```

---

### Task 2: Add OpenSim Motion Validation

**Files:**
- Create: `lib/smpl_optimization/opensim_motion.py`
- Test: `tests/smpl_optimization/test_opensim_motion.py`

- [ ] **Step 1: Write failing tests for `.mot` parsing and coordinate checks**

Create `tests/smpl_optimization/test_opensim_motion.py`:

```python
from pathlib import Path

from lib.smpl_optimization.opensim_motion import parse_mot, summarize_mot_coordinates


def test_parse_mot_reads_header_and_rows(tmp_path: Path):
    mot = tmp_path / "sample.mot"
    mot.write_text(
        "name sample\n"
        "endheader\n"
        "time\thip_flexion_r\tknee_angle_r\n"
        "0.0\t1.0\t2.0\n"
        "0.1\t1.5\t2.5\n",
        encoding="utf-8",
    )

    parsed = parse_mot(mot)

    assert parsed["columns"] == ["time", "hip_flexion_r", "knee_angle_r"]
    assert parsed["data"].shape == (2, 3)


def test_summarize_mot_coordinates_counts_jumps(tmp_path: Path):
    mot = tmp_path / "jump.mot"
    mot.write_text(
        "endheader\n"
        "time\tknee_angle_r\n"
        "0.0\t0.0\n"
        "0.1\t0.1\n"
        "0.2\t5.0\n",
        encoding="utf-8",
    )

    summary = summarize_mot_coordinates(mot, jump_threshold=1.0)

    assert summary["finite"] is True
    assert summary["num_coordinate_jumps"] == 1
    assert summary["coordinate_jerk"]["rms"] == 0.0


def test_summarize_mot_coordinates_counts_range_violations(tmp_path: Path):
    mot = tmp_path / "range.mot"
    mot.write_text(
        "endheader\n"
        "time\tknee_angle_r\n"
        "0.0\t0.0\n"
        "0.1\t200.0\n",
        encoding="utf-8",
    )

    summary = summarize_mot_coordinates(
        mot,
        coordinate_ranges={"knee_angle_r": (-120.0, 160.0)},
    )

    assert summary["num_range_violations"] == 1
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_opensim_motion.py -q
```

Expected: FAIL because `lib.smpl_optimization.opensim_motion` does not exist.

- [ ] **Step 3: Implement `.mot` parser and validator**

Create `lib/smpl_optimization/opensim_motion.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from lib.smpl_optimization.metrics import temporal_derivative_summary


DEFAULT_COORDINATE_RANGES = {
    "hip_flexion_r": (-60.0, 140.0),
    "hip_flexion_l": (-60.0, 140.0),
    "knee_angle_r": (-120.0, 160.0),
    "knee_angle_l": (-120.0, 160.0),
    "ankle_angle_r": (-90.0, 90.0),
    "ankle_angle_l": (-90.0, 90.0),
    "lumbar_extension": (-90.0, 90.0),
    "lumbar_bending": (-90.0, 90.0),
    "lumbar_rotation": (-120.0, 120.0),
    "elbow_flex_r": (-20.0, 170.0),
    "elbow_flex_l": (-20.0, 170.0),
}


def parse_mot(path: Path | str) -> dict:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end = i + 1
            break

    table_lines = [line.strip() for line in lines[header_end:] if line.strip()]
    if not table_lines:
        return {"columns": [], "data": np.zeros((0, 0), dtype=np.float64)}

    columns = table_lines[0].split()
    rows = []
    for line in table_lines[1:]:
        parts = line.split()
        if len(parts) != len(columns):
            continue
        rows.append([float(part) for part in parts])

    data = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(columns)), dtype=np.float64)
    return {"columns": columns, "data": data}


def summarize_mot_coordinates(
    path: Path | str,
    *,
    coordinate_ranges: dict[str, tuple[float, float]] | None = None,
    jump_threshold: float = 45.0,
) -> dict:
    parsed = parse_mot(path)
    columns = parsed["columns"]
    data = parsed["data"]
    if data.size == 0 or not columns:
        return {
            "finite": True,
            "num_coordinates": 0,
            "num_range_violations": 0,
            "num_coordinate_jumps": 0,
            "coordinate_velocity": {"rms": 0.0, "max_abs": 0.0},
            "coordinate_acceleration": {"rms": 0.0, "max_abs": 0.0},
            "coordinate_jerk": {"rms": 0.0, "max_abs": 0.0},
        }

    finite = bool(np.isfinite(data).all())
    coord_start = 1 if columns[0].lower() == "time" else 0
    coord_columns = columns[coord_start:]
    coords = data[:, coord_start:]
    ranges = coordinate_ranges or DEFAULT_COORDINATE_RANGES

    range_violations = 0
    for idx, name in enumerate(coord_columns):
        if name not in ranges:
            continue
        low, high = ranges[name]
        values = coords[:, idx]
        range_violations += int(np.count_nonzero((values < low) | (values > high)))

    if coords.shape[0] <= 1:
        jumps = 0
    else:
        jumps = int(np.count_nonzero(np.abs(np.diff(coords, axis=0)) > float(jump_threshold)))

    return {
        "finite": finite,
        "num_coordinates": int(coords.shape[1]),
        "num_range_violations": int(range_violations),
        "num_coordinate_jumps": int(jumps),
        "coordinate_velocity": temporal_derivative_summary(coords, order=1),
        "coordinate_acceleration": temporal_derivative_summary(coords, order=2),
        "coordinate_jerk": temporal_derivative_summary(coords, order=3),
    }
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_opensim_motion.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add lib\smpl_optimization\opensim_motion.py tests\smpl_optimization\test_opensim_motion.py
git commit -m "feat: add opensim motion validation"
```

---

### Task 3: Add Stage7 Selection Gates

**Files:**
- Create: `lib/smpl_optimization/stage7_selection.py`
- Test: `tests/smpl_optimization/test_stage7_selection.py`

- [ ] **Step 1: Write failing selection tests**

Create `tests/smpl_optimization/test_stage7_selection.py`:

```python
from lib.smpl_optimization.stage7_selection import select_stage7_result


def _summary(*, pose_jerk=1.0, root_jerk=1.0, penetration=0.001, sliding=0.1, mean_rms=0.04, max_rms=0.08):
    return {
        "smpl": {
            "frames": 10,
            "beta_variation_after_max_abs": 0.0,
            "penetration": {"max_penetration": penetration},
            "sliding": {"mean_contact_speed": sliding, "max_contact_speed": sliding * 2.0},
            "pose_smoothness": {"jerk": {"rms": pose_jerk}},
            "root_smoothness": {"jerk": {"rms": root_jerk}},
            "pose_delta": {"lower_body_max_abs": 0.01, "whole_body_max_abs": 0.02},
            "root_delta": {"max_abs": 0.01, "vertical_max_abs": 0.005},
        },
        "opensim": {
            "mean_rms": mean_rms,
            "max_rms": max_rms,
            "motion": {
                "finite": True,
                "num_range_violations": 0,
                "num_coordinate_jumps": 0,
                "coordinate_jerk": {"rms": 1.0},
            },
        },
    }


def test_accepts_candidate_when_smoother_and_gates_pass():
    baseline = _summary(pose_jerk=2.0)
    candidate = _summary(pose_jerk=1.0)

    result = select_stage7_result(baseline, candidate)

    assert result["accepted"] is True
    assert result["selected"] == "candidate"


def test_rejects_candidate_when_penetration_worse():
    baseline = _summary(penetration=0.001, pose_jerk=2.0)
    candidate = _summary(penetration=0.010, pose_jerk=1.0)

    result = select_stage7_result(baseline, candidate)

    assert result["accepted"] is False
    assert result["selected"] == "baseline"
    assert result["checks"]["smpl_max_foot_penetration_not_worse"] is False


def test_rejects_candidate_without_smoothness_improvement():
    baseline = _summary(pose_jerk=1.0, root_jerk=1.0)
    candidate = _summary(pose_jerk=1.0, root_jerk=1.0)

    result = select_stage7_result(baseline, candidate)

    assert result["accepted"] is False
    assert result["checks"]["at_least_one_smoothness_metric_improved"] is False
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_stage7_selection.py -q
```

Expected: FAIL because `stage7_selection.py` does not exist.

- [ ] **Step 3: Implement fail-closed selector**

Create `lib/smpl_optimization/stage7_selection.py`:

```python
from __future__ import annotations


def _number(data, *keys, default=None):
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _not_worse_abs(candidate, baseline, tolerance):
    if candidate is None or baseline is None:
        return False
    return candidate <= baseline + float(tolerance)


def _not_worse_ratio(candidate, baseline, ratio):
    if candidate is None or baseline is None:
        return False
    return candidate <= baseline * (1.0 + float(ratio))


def _improved(candidate, baseline):
    if candidate is None or baseline is None:
        return False
    return candidate < baseline


def select_stage7_result(
    baseline,
    candidate,
    *,
    foot_penetration_tolerance=0.002,
    contact_sliding_mean_ratio=0.02,
    contact_sliding_max_ratio=0.05,
    lower_body_delta_limit=0.05,
    whole_body_delta_limit=0.15,
    root_translation_delta_limit=0.03,
    root_vertical_delta_limit=0.015,
    opensim_mean_rms_ratio=0.02,
    opensim_max_rms_ratio=0.05,
):
    baseline_frames = _number(baseline, "smpl", "frames")
    candidate_frames = _number(candidate, "smpl", "frames")
    candidate_beta = _number(candidate, "smpl", "beta_variation_after_max_abs")

    baseline_penetration = _number(baseline, "smpl", "penetration", "max_penetration")
    candidate_penetration = _number(candidate, "smpl", "penetration", "max_penetration")
    baseline_sliding_mean = _number(baseline, "smpl", "sliding", "mean_contact_speed")
    candidate_sliding_mean = _number(candidate, "smpl", "sliding", "mean_contact_speed")
    baseline_sliding_max = _number(baseline, "smpl", "sliding", "max_contact_speed")
    candidate_sliding_max = _number(candidate, "smpl", "sliding", "max_contact_speed")

    candidate_lower_delta = _number(candidate, "smpl", "pose_delta", "lower_body_max_abs")
    candidate_whole_delta = _number(candidate, "smpl", "pose_delta", "whole_body_max_abs")
    candidate_root_delta = _number(candidate, "smpl", "root_delta", "max_abs")
    candidate_root_vertical_delta = _number(candidate, "smpl", "root_delta", "vertical_max_abs")

    baseline_mean_rms = _number(baseline, "opensim", "mean_rms")
    candidate_mean_rms = _number(candidate, "opensim", "mean_rms")
    baseline_max_rms = _number(baseline, "opensim", "max_rms")
    candidate_max_rms = _number(candidate, "opensim", "max_rms")
    baseline_range_violations = _number(baseline, "opensim", "motion", "num_range_violations")
    candidate_range_violations = _number(candidate, "opensim", "motion", "num_range_violations")
    baseline_jumps = _number(baseline, "opensim", "motion", "num_coordinate_jumps")
    candidate_jumps = _number(candidate, "opensim", "motion", "num_coordinate_jumps")

    baseline_pose_jerk = _number(baseline, "smpl", "pose_smoothness", "jerk", "rms")
    candidate_pose_jerk = _number(candidate, "smpl", "pose_smoothness", "jerk", "rms")
    baseline_root_jerk = _number(baseline, "smpl", "root_smoothness", "jerk", "rms")
    candidate_root_jerk = _number(candidate, "smpl", "root_smoothness", "jerk", "rms")
    baseline_coord_jerk = _number(baseline, "opensim", "motion", "coordinate_jerk", "rms")
    candidate_coord_jerk = _number(candidate, "opensim", "motion", "coordinate_jerk", "rms")

    improvements = {
        "whole_body_pose_jerk_improved": _improved(candidate_pose_jerk, baseline_pose_jerk),
        "root_translation_jerk_improved": _improved(candidate_root_jerk, baseline_root_jerk),
        "opensim_coordinate_jerk_improved": _improved(candidate_coord_jerk, baseline_coord_jerk),
        "opensim_coordinate_jumps_improved": _improved(candidate_jumps, baseline_jumps),
    }

    checks = {
        "frame_count_unchanged": baseline_frames is not None and candidate_frames == baseline_frames,
        "beta_variation_after_max_abs_zero": candidate_beta == 0.0,
        "smpl_max_foot_penetration_not_worse": _not_worse_abs(
            candidate_penetration, baseline_penetration, foot_penetration_tolerance
        ),
        "smpl_contact_sliding_mean_not_worse": _not_worse_ratio(
            candidate_sliding_mean, baseline_sliding_mean, contact_sliding_mean_ratio
        ),
        "smpl_contact_sliding_max_not_worse": _not_worse_ratio(
            candidate_sliding_max, baseline_sliding_max, contact_sliding_max_ratio
        ),
        "lower_body_pose_delta_small": candidate_lower_delta is not None and candidate_lower_delta <= lower_body_delta_limit,
        "whole_body_pose_delta_small": candidate_whole_delta is not None and candidate_whole_delta <= whole_body_delta_limit,
        "root_translation_delta_small": candidate_root_delta is not None and candidate_root_delta <= root_translation_delta_limit,
        "root_vertical_delta_small": candidate_root_vertical_delta is not None and candidate_root_vertical_delta <= root_vertical_delta_limit,
        "opensim_mean_rms_not_worse": _not_worse_ratio(candidate_mean_rms, baseline_mean_rms, opensim_mean_rms_ratio),
        "opensim_max_rms_not_worse": _not_worse_ratio(candidate_max_rms, baseline_max_rms, opensim_max_rms_ratio),
        "opensim_coordinate_range_violations_not_worse": (
            baseline_range_violations is not None
            and candidate_range_violations is not None
            and candidate_range_violations <= baseline_range_violations
        ),
        "opensim_coordinate_jumps_not_worse": (
            baseline_jumps is not None and candidate_jumps is not None and candidate_jumps <= baseline_jumps
        ),
        "at_least_one_smoothness_metric_improved": any(improvements.values()),
    }
    accepted = all(checks.values())
    return {
        "selected": "candidate" if accepted else "baseline",
        "accepted": accepted,
        "checks": checks,
        "target_improvements": improvements,
        "baseline": baseline,
        "candidate": candidate,
    }
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_stage7_selection.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add lib\smpl_optimization\stage7_selection.py tests\smpl_optimization\test_stage7_selection.py
git commit -m "feat: add stage7 safety selector"
```

---

### Task 4: Generate Conservative Smooth Candidate

**Files:**
- Create: `lib/smpl_optimization/whole_body_smooth.py`
- Test: `tests/smpl_optimization/test_whole_body_smooth.py`

- [ ] **Step 1: Write failing tests for bounded candidate generation**

Create `tests/smpl_optimization/test_whole_body_smooth.py`:

```python
import numpy as np

from lib.smpl_optimization.whole_body_smooth import (
    WholeBodySmoothConfig,
    build_region_weights,
    smooth_record,
)


def test_build_region_weights_has_24_smpl_joints():
    weights = build_region_weights()
    assert weights.shape == (24, 3)
    assert weights[1].max() < weights[16].max()


def test_smooth_record_keeps_beta_unchanged_and_bounds_pose_delta():
    pose = np.zeros((8, 72), dtype=np.float32)
    pose[4, 16 * 3] = 1.0
    record = {
        "pose": pose.copy(),
        "trans": np.zeros((8, 3), dtype=np.float32),
        "betas": np.ones((8, 10), dtype=np.float32),
    }

    out, report = smooth_record(record, WholeBodySmoothConfig(max_whole_body_delta=0.15))

    assert np.allclose(out["betas"], record["betas"])
    assert report["pose_delta"]["whole_body_max_abs"] <= 0.15


def test_smooth_record_preserves_short_sequences():
    record = {
        "pose": np.zeros((2, 72), dtype=np.float32),
        "trans": np.zeros((2, 3), dtype=np.float32),
    }

    out, report = smooth_record(record, WholeBodySmoothConfig())

    assert np.allclose(out["pose"], record["pose"])
    assert report["candidate_generated"] is False
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_whole_body_smooth.py -q
```

Expected: FAIL because `whole_body_smooth.py` does not exist.

- [ ] **Step 3: Implement bounded residual smoothing**

Create `lib/smpl_optimization/whole_body_smooth.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from lib.smpl_optimization.metrics import pose_delta_max_abs, pose_smoothness, root_translation_smoothness


LOWER_BODY_JOINTS = (1, 2, 4, 5, 7, 8, 10, 11)
ROOT_AND_TRUNK_JOINTS = (0, 3, 6, 9, 12, 13, 14, 15)


@dataclass
class WholeBodySmoothConfig:
    lower_body_alpha: float = 0.08
    root_trunk_alpha: float = 0.15
    upper_body_alpha: float = 0.25
    trans_alpha: float = 0.10
    max_lower_body_delta: float = 0.05
    max_whole_body_delta: float = 0.15
    max_root_delta: float = 0.03
    max_root_vertical_delta: float = 0.015


def build_region_weights(config: WholeBodySmoothConfig | None = None) -> np.ndarray:
    config = config or WholeBodySmoothConfig()
    weights = np.full((24, 3), float(config.upper_body_alpha), dtype=np.float32)
    for joint in ROOT_AND_TRUNK_JOINTS:
        weights[joint, :] = float(config.root_trunk_alpha)
    for joint in LOWER_BODY_JOINTS:
        weights[joint, :] = float(config.lower_body_alpha)
    return weights


def _smooth_three_point(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] < 3:
        return values.copy()
    out = values.copy()
    out[1:-1] = 0.25 * values[:-2] + 0.5 * values[1:-1] + 0.25 * values[2:]
    return out


def _clip_delta(candidate: np.ndarray, baseline: np.ndarray, limit: float) -> np.ndarray:
    delta = np.clip(candidate - baseline, -float(limit), float(limit))
    return baseline + delta


def _record_pose_key(record: dict) -> str:
    return "pose_world" if "pose_world" in record else "pose"


def _record_trans_key(record: dict) -> str:
    return "trans_world" if "trans_world" in record else "trans"


def smooth_record(record: dict, config: WholeBodySmoothConfig | None = None) -> tuple[dict, dict]:
    config = config or WholeBodySmoothConfig()
    out = deepcopy(record)
    pose_key = _record_pose_key(record)
    trans_key = _record_trans_key(record)
    pose = np.asarray(record.get(pose_key, np.asarray([])), dtype=np.float32)
    trans = np.asarray(record.get(trans_key, np.asarray([])), dtype=np.float32)
    if pose.ndim != 2 or pose.shape[0] < 4 or pose.shape[1] < 72:
        return out, {"candidate_generated": False, "reason": "sequence_too_short_or_pose_missing"}

    pose_24 = pose[:, :72].reshape(pose.shape[0], 24, 3)
    smoothed_pose = _smooth_three_point(pose_24)
    weights = build_region_weights(config)[None, :, :]
    candidate_pose_24 = pose_24 + weights * (smoothed_pose - pose_24)
    candidate_pose_24[:, LOWER_BODY_JOINTS, :] = _clip_delta(
        candidate_pose_24[:, LOWER_BODY_JOINTS, :],
        pose_24[:, LOWER_BODY_JOINTS, :],
        config.max_lower_body_delta,
    )
    candidate_pose_24 = _clip_delta(candidate_pose_24, pose_24, config.max_whole_body_delta)
    candidate_pose = pose.copy()
    candidate_pose[:, :72] = candidate_pose_24.reshape(pose.shape[0], 72)

    candidate_trans = trans.copy()
    if trans.ndim == 2 and trans.shape[0] >= 4 and trans.shape[1] >= 3:
        smoothed_trans = _smooth_three_point(trans)
        candidate_trans = trans + float(config.trans_alpha) * (smoothed_trans - trans)
        candidate_trans = _clip_delta(candidate_trans, trans, config.max_root_delta)
        candidate_trans[:, 1] = _clip_delta(
            candidate_trans[:, 1],
            trans[:, 1],
            config.max_root_vertical_delta,
        )

    out[pose_key] = candidate_pose.astype(pose.dtype, copy=False)
    if trans.size:
        out[trans_key] = candidate_trans.astype(trans.dtype, copy=False)

    lower_original = pose_24[:, LOWER_BODY_JOINTS, :].reshape(pose.shape[0], -1)
    lower_candidate = candidate_pose_24[:, LOWER_BODY_JOINTS, :].reshape(pose.shape[0], -1)
    root_delta = np.abs(candidate_trans - trans) if candidate_trans.size and trans.size else np.asarray([0.0])
    report = {
        "candidate_generated": True,
        "pose_smoothness_before": pose_smoothness(pose),
        "pose_smoothness_after": pose_smoothness(candidate_pose),
        "root_smoothness_before": root_translation_smoothness(trans),
        "root_smoothness_after": root_translation_smoothness(candidate_trans),
        "pose_delta": {
            "lower_body_max_abs": pose_delta_max_abs(lower_original, lower_candidate),
            "whole_body_max_abs": pose_delta_max_abs(pose, candidate_pose),
        },
        "root_delta": {
            "max_abs": float(round(float(np.max(root_delta)), 8)),
            "vertical_max_abs": float(round(float(np.max(np.abs(candidate_trans[:, 1] - trans[:, 1]))), 8)) if candidate_trans.ndim == 2 and trans.ndim == 2 and trans.shape[1] > 1 else 0.0,
        },
    }
    return out, report
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_whole_body_smooth.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add lib\smpl_optimization\whole_body_smooth.py tests\smpl_optimization\test_whole_body_smooth.py
git commit -m "feat: add conservative whole-body smoother"
```

---

### Task 5: Add Stage7 CLI Script

**Files:**
- Create: `scripts/whole_body_smoothness_optimizer.py`
- Test: `tests/smpl_optimization/test_whole_body_smoothness_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/smpl_optimization/test_whole_body_smoothness_cli.py`:

```python
import argparse
from pathlib import Path

from scripts.whole_body_smoothness_optimizer import build_retarget_cmd, stage7_paths


def test_stage7_paths_are_stable(tmp_path: Path):
    paths = stage7_paths(tmp_path)
    assert paths["baseline_smpl"].name == "baseline_smpl.pkl"
    assert paths["candidate_smpl"].name == "smooth_candidate_smpl.pkl"
    assert paths["selected_smpl"].name == "selected_smooth_smpl.pkl"
    assert paths["report"].name == "stage7_smoothness_report.json"


def test_build_retarget_cmd_uses_log_and_free_root(tmp_path: Path):
    args = argparse.Namespace(
        fps=60.0,
        device="cpu",
        chunk_size=128,
        retarget_config="configs/retarget/smpl_to_mimicmsk_opensim.yaml",
        alignment="anatomical",
        axis_conversion="mujoco_to_opensim",
        root_calibration="constant_from_free_ik",
        root_calib_frames=10,
        track_id="merge",
        ik_accuracy=1e-4,
        opensim_cmd=None,
        free_root=True,
    )

    cmd = build_retarget_cmd(args, tmp_path / "input.pkl", tmp_path / "opensim", tmp_path / "ik.log")

    assert "--opensim-log" in cmd
    assert "--free-root" in cmd
    assert "--run-ik" in cmd
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_whole_body_smoothness_cli.py -q
```

Expected: FAIL because `scripts/whole_body_smoothness_optimizer.py` does not exist.

- [ ] **Step 3: Implement CLI orchestration**

Create `scripts/whole_body_smoothness_optimizer.py` with these public functions and main flow:

```python
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.metrics import beta_variation_max_abs, pose_smoothness, root_translation_smoothness
from lib.smpl_optimization.opensim_feedback import parse_ik_log_metrics
from lib.smpl_optimization.opensim_motion import summarize_mot_coordinates
from lib.smpl_optimization.reports import to_jsonable, write_json
from lib.smpl_optimization.stage7_selection import select_stage7_result
from lib.smpl_optimization.whole_body_smooth import WholeBodySmoothConfig, smooth_record
from lib.world_grounded.tracks import select_track


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
        "candidate_opensim": out_dir / "candidate_opensim",
        "baseline_log": out_dir / "baseline_opensim" / "opensim_ik.log",
        "candidate_log": out_dir / "candidate_opensim" / "opensim_ik.log",
        "report": out_dir / "stage7_smoothness_report.json",
    }


def build_retarget_cmd(args, input_pkl: Path, out_dir: Path, log_path: Path) -> list[str]:
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
```

The main implementation must:

```python
def load_selected_record(pkl: Path, track_id: str) -> tuple[dict, object, dict]:
    data = joblib.load(pkl)
    selected_track_id, record = select_track(data, track_id)
    return data, selected_track_id, record


def replace_selected_record(data: dict, selected_track_id: str, record: dict) -> dict:
    out = dict(data)
    out[selected_track_id] = record
    return out


def summarize_smpl(record: dict, smooth_report: dict | None = None) -> dict:
    pose = record.get("pose_world", record.get("pose", []))
    trans = record.get("trans_world", record.get("trans", []))
    summary = {
        "frames": int(len(pose)) if hasattr(pose, "__len__") else 0,
        "beta_variation_after_max_abs": beta_variation_max_abs(record.get("betas", [])),
        "pose_smoothness": pose_smoothness(pose),
        "root_smoothness": root_translation_smoothness(trans),
    }
    if smooth_report:
        summary["pose_delta"] = smooth_report.get("pose_delta", {})
        summary["root_delta"] = smooth_report.get("root_delta", {})
    else:
        summary["pose_delta"] = {"lower_body_max_abs": 0.0, "whole_body_max_abs": 0.0}
        summary["root_delta"] = {"max_abs": 0.0, "vertical_max_abs": 0.0}
    return summary


def find_opensim_mot(opensim_dir: Path) -> Path:
    mot = opensim_dir / "opensim_ik.mot"
    if mot.exists():
        return mot
    candidates = sorted(opensim_dir.glob("*.mot"))
    if not candidates:
        raise FileNotFoundError(f"No OpenSim .mot file found in {opensim_dir}")
    return candidates[0]
```

`main()` must copy the input to `baseline_smpl.pkl`, write `smooth_candidate_smpl.pkl`, run retarget+IK for both files, summarize metrics, call `select_stage7_result`, copy the selected file to `selected_smooth_smpl.pkl`, and write `stage7_smoothness_report.json`.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_whole_body_smoothness_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Run script help**

Run:

```powershell
python scripts\whole_body_smoothness_optimizer.py --help
```

Expected: prints CLI help with `--input-pkl`, `--out-dir`, `--fps`, `--track-id`, `--free-root`, `--fix-root`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add scripts\whole_body_smoothness_optimizer.py tests\smpl_optimization\test_whole_body_smoothness_cli.py
git commit -m "feat: add stage7 smoothness optimizer cli"
```

---

### Task 6: Integrate Stage7 Into Main Pipeline

**Files:**
- Modify: `scripts/video_to_fixed_smpl_to_opensim.py`
- Test: `tests/smpl_optimization/test_video_pipeline_stage7.py`

- [ ] **Step 1: Write failing pipeline command-construction test**

Create `tests/smpl_optimization/test_video_pipeline_stage7.py`:

```python
from scripts.video_to_fixed_smpl_to_opensim import build_parser


def test_video_pipeline_accepts_whole_body_smooth_flag():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--video",
            "examples/forehand_clear/video1.mp4",
            "--fps",
            "60",
            "--world-grounded",
            "--optimize-lower-body",
            "--whole-body-smooth",
            "--free-root",
        ]
    )

    assert args.whole_body_smooth is True
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_video_pipeline_stage7.py -q
```

Expected: FAIL because `--whole-body-smooth` is not defined.

- [ ] **Step 3: Add CLI flag and stage routing**

In `scripts/video_to_fixed_smpl_to_opensim.py`, add:

```python
parser.add_argument("--whole-body-smooth", action="store_true")
parser.add_argument("--whole-body-smooth-out-dir", default=None)
```

After Stage5/Stage6 selected SMPL path is known and before final retargeting, insert:

```python
if args.whole_body_smooth:
    whole_body_dir = (
        Path(args.whole_body_smooth_out_dir).resolve()
        if args.whole_body_smooth_out_dir
        else output_pth / "_whole_body_smooth" / run_id
    )
    whole_body_cmd = [
        sys.executable,
        "scripts/whole_body_smoothness_optimizer.py",
        "--input-pkl",
        str(retarget_input_pkl),
        "--out-dir",
        str(whole_body_dir),
        "--fps",
        str(args.fps),
        "--track-id",
        str(args.track_id),
        "--device",
        str(args.device),
        "--chunk-size",
        str(args.chunk_size),
        "--retarget-config",
        str(args.retarget_config),
        "--alignment",
        str(args.alignment),
        "--axis-conversion",
        str(args.axis_conversion),
        "--root-calibration",
        str(args.root_calibration),
        "--root-calib-frames",
        str(args.root_calib_frames),
    ]
    if args.opensim_cmd:
        whole_body_cmd += ["--opensim-cmd", str(args.opensim_cmd)]
    if args.free_root:
        whole_body_cmd.append("--free-root")
    run(whole_body_cmd)
    retarget_input_pkl = whole_body_dir / "selected_smooth_smpl.pkl"
    pipeline_outputs["whole_body_smooth_dir"] = str(whole_body_dir)
    pipeline_outputs["selected_smooth_smpl_pkl"] = str(retarget_input_pkl)
```

Use existing local variable names where they differ; the behavior must be the same.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_video_pipeline_stage7.py tests\smpl_optimization\test_video_pipeline_stage6.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add scripts\video_to_fixed_smpl_to_opensim.py tests\smpl_optimization\test_video_pipeline_stage7.py
git commit -m "feat: add stage7 pipeline flag"
```

---

### Task 7: Integrate Stage7 Into Ablation Runner

**Files:**
- Modify: `scripts/run_wham_ablation.py`
- Modify: `tests/smpl_optimization/test_run_wham_ablation.py`

- [ ] **Step 1: Extend ablation tests**

Add this test to `tests/smpl_optimization/test_run_wham_ablation.py`:

```python
def test_active_stages_append_stage7_after_selected_source():
    from scripts.run_wham_ablation import active_stages

    ids = [stage["id"] for stage in active_stages(run_opensim_feedback_loop=False, run_whole_body_smooth=True)]
    assert ids[-2] == "04_lower_body_full"
    assert ids[-1] == "05_whole_body_smooth"

    ids_with_stage6 = [
        stage["id"]
        for stage in active_stages(run_opensim_feedback_loop=True, run_whole_body_smooth=True)
    ]
    assert ids_with_stage6[-2] == "05_opensim_feedback"
    assert ids_with_stage6[-1] == "06_whole_body_smooth"
```

Add this command-construction test:

```python
def test_whole_body_smooth_command_uses_previous_stage_pkl(tmp_path):
    from scripts.run_wham_ablation import build_whole_body_smooth_cmd

    args = argparse.Namespace(
        fps=60.0,
        track_id="merge",
        device="cuda",
        chunk_size=128,
        retarget_config="configs/retarget/test.yaml",
        alignment="anatomical",
        axis_conversion="mujoco_to_opensim",
        root_calibration="constant_from_free_ik",
        root_calib_frames=7,
        opensim_cmd="opensim-cmd",
        ik_accuracy=1e-5,
        free_root=True,
    )
    input_pkl = tmp_path / "previous.pkl"
    out_dir = tmp_path / "06_whole_body_smooth"

    cmd = build_whole_body_smooth_cmd(args, input_pkl, out_dir)

    assert cmd[:2] == [sys.executable, "scripts/whole_body_smoothness_optimizer.py"]
    assert cmd[cmd.index("--input-pkl") + 1] == str(input_pkl)
    assert cmd[cmd.index("--out-dir") + 1] == str(out_dir)
    assert "--free-root" in cmd
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_run_wham_ablation.py -q
```

Expected: FAIL because Stage7 is not defined.

- [ ] **Step 3: Add ablation flag and command**

In `scripts/run_wham_ablation.py`, add parser flag:

```python
parser.add_argument("--run-whole-body-smooth", action="store_true")
```

Replace direct iteration over `STAGES` in `main()` with a helper:

```python
def active_stages(*, run_opensim_feedback_loop: bool, run_whole_body_smooth: bool) -> list[dict[str, str]]:
    stages = [
        stage
        for stage in STAGES
        if stage["id"] != "05_opensim_feedback" or run_opensim_feedback_loop
    ]
    if run_whole_body_smooth:
        stages.append(
            {
                "id": f"{len(stages):02d}_whole_body_smooth",
                "description": "Conservative whole-body smoothness selected SMPL.",
            }
        )
    return stages
```

Add command construction:

```python
def build_whole_body_smooth_cmd(args: argparse.Namespace, input_pkl: Path, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/whole_body_smoothness_optimizer.py",
        "--input-pkl",
        str(input_pkl),
        "--out-dir",
        str(out_dir),
        "--fps",
        str(args.fps),
        "--track-id",
        str(args.track_id),
        "--device",
        args.device,
        "--chunk-size",
        str(args.chunk_size),
        "--retarget-config",
        args.retarget_config,
        "--alignment",
        args.alignment,
        "--axis-conversion",
        args.axis_conversion,
        "--root-calibration",
        args.root_calibration,
        "--root-calib-frames",
        str(args.root_calib_frames),
        "--ik-accuracy",
        str(args.ik_accuracy),
    ]
    if args.opensim_cmd:
        cmd += ["--opensim-cmd", str(args.opensim_cmd)]
    if args.free_root:
        cmd.append("--free-root")
    return cmd
```

After optional Stage6 execution, add:

```python
if args.run_whole_body_smooth:
    smooth_stage_id = "06_whole_body_smooth" if args.run_opensim_feedback_loop else "05_whole_body_smooth"
    smooth_dir = ablation_root / smooth_stage_id
    smooth_input = feedback_pkl if args.run_opensim_feedback_loop else lower_body_pkl
    run(build_whole_body_smooth_cmd(args, smooth_input, smooth_dir))
    stage_pkls[smooth_stage_id] = smooth_dir / "selected_smooth_smpl.pkl"
```

Use `stages = active_stages(...)` for retarget loops, metric loops, and summary:

```python
stages = active_stages(
    run_opensim_feedback_loop=args.run_opensim_feedback_loop,
    run_whole_body_smooth=args.run_whole_body_smooth,
)
```

Then iterate over `stages` instead of `STAGES`:

```python
for stage in stages:
    stage_id = stage["id"]
    retarget_dir = ablation_root / stage_id / "opensim_retarget"
    retarget_dirs[stage_id] = retarget_dir
    run(build_retarget_cmd(args, stage_pkls[stage_id], retarget_dir))
```

The summary should store the active stage list:

```python
summary = {
    "video": str(video),
    "output_root": str(output_root),
    "ablation_root": str(ablation_root),
    "stages": stages,
    "metrics": rows,
}
```

Do not remove the existing `STAGES` constant; tests still use it to verify the base stage order.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_run_wham_ablation.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add scripts\run_wham_ablation.py tests\smpl_optimization\test_run_wham_ablation.py
git commit -m "feat: add stage7 ablation option"
```

---

### Task 8: End-to-End Verification

**Files:**
- No source files unless verification reveals a bug.

- [ ] **Step 1: Run all focused unit tests**

Run:

```powershell
python -m pytest tests\smpl_optimization -q
```

Expected: PASS.

- [ ] **Step 2: Run Stage7 smoke on video1 without rerunning WHAM if possible**

Use an existing Stage5 or Stage6 pkl if available:

```powershell
python scripts\whole_body_smoothness_optimizer.py `
  --input-pkl output\forehand_clear\_ablation\video1_*\04_lower_body_full\corrected_smpl.pkl `
  --out-dir output\forehand_clear\_stage7_smoke\video1 `
  --fps 60 `
  --track-id merge `
  --device cuda `
  --free-root
```

If PowerShell wildcard expansion does not produce a single file, replace the `--input-pkl` value with the exact `corrected_smpl.pkl` path.

Expected: command completes, writes `selected_smooth_smpl.pkl`, and writes `stage7_smoothness_report.json`.

- [ ] **Step 3: Inspect report acceptance**

Run:

```powershell
python -c "import json; p='output/forehand_clear/_stage7_smoke/video1/stage7_smoothness_report.json'; d=json.load(open(p,encoding='utf-8')); print(d['selection']['accepted']); print(d['selection']['checks'])"
```

Expected: Either accepted is `True`, or accepted is `False` with failed gates explaining rollback. The selected file must exist in both cases.

- [ ] **Step 4: Run pipeline help**

Run:

```powershell
python scripts\video_to_fixed_smpl_to_opensim.py --help
```

Expected: help includes `--whole-body-smooth`.

- [ ] **Step 5: Run ablation help**

Run:

```powershell
python scripts\run_wham_ablation.py --help
```

Expected: help includes `--run-whole-body-smooth`.

- [ ] **Step 6: Commit verification fixes only if needed**

If verification required source changes:

```powershell
git status --short
git add lib\smpl_optimization\metrics.py lib\smpl_optimization\opensim_motion.py lib\smpl_optimization\stage7_selection.py lib\smpl_optimization\whole_body_smooth.py scripts\whole_body_smoothness_optimizer.py scripts\video_to_fixed_smpl_to_opensim.py scripts\run_wham_ablation.py tests\smpl_optimization
git commit -m "fix: stabilize stage7 verification"
```

Do not commit generated output directories.
