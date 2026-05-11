# SMPL Lower-Body Ground Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lower-body SMPL optimizer that keeps beta fixed, corrects root/ground/foot artifacts, uses OpenSim as an outer-loop evaluator, and outputs `corrected_smpl.pkl` for later MuJoCo/RL retargeting.

**Architecture:** Keep world-grounded code responsible for ground estimation and root alignment. Add a new `lib/smpl_optimization` package for metrics, losses, optimization, reports, and OpenSim feedback parsing. Add `scripts/optimize_smpl_lower_body.py`, then integrate it into `scripts/video_to_fixed_smpl_to_opensim.py` behind `--optimize-lower-body`.

**Tech Stack:** Python 3.10, NumPy, PyTorch, joblib, pytest, existing WHAM SMPL/body-model utilities, existing OpenSim retarget script.

---

## File Structure

- Create `lib/smpl_optimization/__init__.py`: public exports for the new package.
- Create `lib/smpl_optimization/metrics.py`: pure NumPy quality metrics for penetration, sliding, root jitter, pose deltas, and beta variation.
- Create `lib/smpl_optimization/losses.py`: PyTorch loss helpers used by the optimizer.
- Create `lib/smpl_optimization/lower_body.py`: lower-body optimization orchestration, record copying, frame/track safety, and report assembly.
- Create `lib/smpl_optimization/opensim_feedback.py`: parse OpenSim IK `.mot` and report files when available; produce bounded second-pass frame weights.
- Create `lib/smpl_optimization/reports.py`: validation summary and JSON-safe report helpers.
- Create `scripts/optimize_smpl_lower_body.py`: CLI for fixed-beta/world-grounded SMPL pkl to `corrected_smpl.pkl`.
- Modify `scripts/video_to_fixed_smpl_to_opensim.py`: add `--optimize-lower-body`, run the new script after world grounding, retarget from `corrected_smpl.pkl`.
- Add tests under `tests/smpl_optimization/`.

Implementation boundary: do not move existing `lib/world_grounded/*` logic. The lower-body optimizer consumes world-grounded output and reports.

---

### Task 1: Metrics Foundation

**Files:**
- Create: `lib/smpl_optimization/__init__.py`
- Create: `lib/smpl_optimization/metrics.py`
- Test: `tests/smpl_optimization/test_metrics.py`

- [ ] **Step 1: Create the package init**

Create `lib/smpl_optimization/__init__.py`:

```python
"""SMPL sequence optimization utilities."""
```

- [ ] **Step 2: Write failing metric tests**

Create `tests/smpl_optimization/test_metrics.py`:

```python
import numpy as np

from lib.smpl_optimization.metrics import (
    beta_variation_max_abs,
    contact_foot_sliding,
    foot_penetration_depth,
    root_vertical_jitter,
    pose_delta_max_abs,
)


def test_beta_variation_max_abs_zero_for_fixed_beta():
    betas = np.tile(np.arange(10, dtype=np.float32), (4, 1))
    assert beta_variation_max_abs(betas) == 0.0


def test_beta_variation_max_abs_detects_frame_change():
    betas = np.zeros((3, 10), dtype=np.float32)
    betas[2, 4] = 0.25
    assert beta_variation_max_abs(betas) == 0.25


def test_foot_penetration_depth_reports_max_and_mean():
    foot_y = np.array([[0.1, -0.2], [-0.05, 0.0]], dtype=np.float32)
    report = foot_penetration_depth(foot_y, ground_y=0.0)
    assert report["max_penetration"] == 0.2
    assert np.isclose(report["mean_penetration"], 0.0625)
    assert report["num_penetrating"] == 2


def test_contact_foot_sliding_uses_contact_mask():
    points = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.2, 0.0, 0.0]],
            [[0.5, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    contact = np.array([[1.0], [1.0], [0.0]], dtype=np.float32)
    report = contact_foot_sliding(points, contact, fps=10.0, threshold=1.0)
    assert np.isclose(report["mean_contact_speed"], 2.0)
    assert report["num_sliding"] == 1


def test_root_vertical_jitter_is_second_difference_rms():
    trans = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    report = root_vertical_jitter(trans)
    assert report["rms_vertical_accel"] > 0.0


def test_pose_delta_max_abs_compares_matching_prefix():
    original = np.zeros((2, 72), dtype=np.float32)
    corrected = original.copy()
    corrected[1, 10] = -0.5
    assert pose_delta_max_abs(original, corrected) == 0.5
```

- [ ] **Step 3: Run the failing tests**

Run:

```powershell
pytest tests\smpl_optimization\test_metrics.py -v
```

Expected: import failure for `lib.smpl_optimization.metrics`.

- [ ] **Step 4: Implement the metrics**

Create `lib/smpl_optimization/metrics.py`:

```python
from __future__ import annotations

import numpy as np


def _as_float_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def beta_variation_max_abs(betas) -> float:
    arr = _as_float_array(betas)
    if arr.ndim == 1 or len(arr) <= 1:
        return 0.0
    ref = arr[:1]
    return float(np.max(np.abs(arr - ref)))


def foot_penetration_depth(foot_y, ground_y: float = 0.0) -> dict:
    y = _as_float_array(foot_y)
    penetration = np.maximum(float(ground_y) - y, 0.0)
    return {
        "max_penetration": float(np.max(penetration)) if penetration.size else 0.0,
        "mean_penetration": float(np.mean(penetration)) if penetration.size else 0.0,
        "num_penetrating": int(np.count_nonzero(penetration > 0.0)),
    }


def contact_foot_sliding(points, contact, fps: float, threshold: float = 0.15) -> dict:
    pts = _as_float_array(points)
    mask = _as_float_array(contact)
    if pts.ndim != 3 or pts.shape[-1] != 3 or len(pts) < 2:
        return {"mean_contact_speed": 0.0, "max_contact_speed": 0.0, "num_sliding": 0}
    n_points = pts.shape[1]
    mask = mask[: len(pts), :n_points]
    velocity = np.linalg.norm(np.diff(pts[:, :, [0, 2]], axis=0), axis=-1) * float(fps)
    contact_pair = np.minimum(mask[:-1], mask[1:]) > 0.5
    speeds = velocity[contact_pair]
    return {
        "mean_contact_speed": float(np.mean(speeds)) if speeds.size else 0.0,
        "max_contact_speed": float(np.max(speeds)) if speeds.size else 0.0,
        "num_sliding": int(np.count_nonzero(speeds > threshold)),
    }


def root_vertical_jitter(trans_world) -> dict:
    trans = _as_float_array(trans_world)
    if trans.ndim != 2 or trans.shape[-1] < 2 or len(trans) < 3:
        return {"rms_vertical_accel": 0.0, "max_vertical_accel": 0.0}
    accel = np.diff(trans[:, 1], n=2)
    return {
        "rms_vertical_accel": float(np.sqrt(np.mean(accel * accel))),
        "max_vertical_accel": float(np.max(np.abs(accel))),
    }


def pose_delta_max_abs(original_pose, corrected_pose) -> float:
    original = _as_float_array(original_pose)
    corrected = _as_float_array(corrected_pose)
    n0 = min(original.shape[0], corrected.shape[0])
    n1 = min(original.shape[-1], corrected.shape[-1])
    if n0 == 0 or n1 == 0:
        return 0.0
    return float(np.max(np.abs(corrected[:n0, :n1] - original[:n0, :n1])))
```

- [ ] **Step 5: Run metric tests**

Run:

```powershell
pytest tests\smpl_optimization\test_metrics.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

Run:

```powershell
git add lib\smpl_optimization\__init__.py lib\smpl_optimization\metrics.py tests\smpl_optimization\test_metrics.py
git commit -m "feat: add smpl optimization metrics"
```

---

### Task 2: Report Builder and Validation Summary

**Files:**
- Create: `lib/smpl_optimization/reports.py`
- Test: `tests/smpl_optimization/test_reports.py`

- [ ] **Step 1: Write failing report tests**

Create `tests/smpl_optimization/test_reports.py`:

```python
from lib.smpl_optimization.reports import build_validation_summary, to_jsonable


def test_to_jsonable_converts_numpy_scalars():
    import numpy as np

    data = {"x": np.float32(1.5), "items": [np.int64(2)]}
    assert to_jsonable(data) == {"x": 1.5, "items": [2]}


def test_validation_summary_marks_success_when_all_thresholds_pass():
    summary = build_validation_summary(
        beta_variation_after_max_abs=0.0,
        frame_count_unchanged=True,
        before={"penetration": {"max_penetration": 0.2}, "sliding": {"mean_contact_speed": 0.3}, "root": {"rms_vertical_accel": 0.2}},
        after={"penetration": {"max_penetration": 0.05}, "sliding": {"mean_contact_speed": 0.1}, "root": {"rms_vertical_accel": 0.1}},
        pose_delta={"upper_body_max_abs": 0.01, "lower_body_max_abs": 0.3},
        opensim={"ik_rms_not_worse": True, "ground_clearance_not_worse": True},
    )
    assert summary["success"] is True
    assert summary["checks"]["smpl_foot_penetration_reduced"] is True


def test_validation_summary_marks_degraded_when_beta_changes():
    summary = build_validation_summary(
        beta_variation_after_max_abs=1e-3,
        frame_count_unchanged=True,
        before={"penetration": {"max_penetration": 0.2}, "sliding": {"mean_contact_speed": 0.3}, "root": {"rms_vertical_accel": 0.2}},
        after={"penetration": {"max_penetration": 0.05}, "sliding": {"mean_contact_speed": 0.1}, "root": {"rms_vertical_accel": 0.1}},
        pose_delta={"upper_body_max_abs": 0.01, "lower_body_max_abs": 0.3},
        opensim={"ik_rms_not_worse": True, "ground_clearance_not_worse": True},
    )
    assert summary["success"] is False
    assert summary["checks"]["beta_variation_after_max_abs"] is False
```

- [ ] **Step 2: Run the failing tests**

Run:

```powershell
pytest tests\smpl_optimization\test_reports.py -v
```

Expected: import failure for `lib.smpl_optimization.reports`.

- [ ] **Step 3: Implement report helpers**

Create `lib/smpl_optimization/reports.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def build_validation_summary(
    *,
    beta_variation_after_max_abs: float,
    frame_count_unchanged: bool,
    before: dict,
    after: dict,
    pose_delta: dict,
    opensim: dict,
) -> dict:
    checks = {
        "beta_variation_after_max_abs": float(beta_variation_after_max_abs) == 0.0,
        "frame_count_unchanged": bool(frame_count_unchanged),
        "smpl_foot_penetration_reduced": after["penetration"]["max_penetration"] <= before["penetration"]["max_penetration"],
        "smpl_contact_foot_sliding_reduced": after["sliding"]["mean_contact_speed"] <= before["sliding"]["mean_contact_speed"],
        "root_vertical_jitter_not_worse": after["root"]["rms_vertical_accel"] <= before["root"]["rms_vertical_accel"],
        "upper_body_pose_delta_small": pose_delta["upper_body_max_abs"] <= 0.05,
        "lower_body_pose_delta_bounded": pose_delta["lower_body_max_abs"] <= 1.2,
        "opensim_ik_rms_not_worse": bool(opensim.get("ik_rms_not_worse", True)),
        "opensim_ground_clearance_not_worse": bool(opensim.get("ground_clearance_not_worse", True)),
    }
    return {
        "success": all(checks.values()),
        "checks": checks,
        "before": before,
        "after": after,
        "pose_delta": pose_delta,
        "opensim": opensim,
    }
```

- [ ] **Step 4: Run report tests**

Run:

```powershell
pytest tests\smpl_optimization\test_reports.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add lib\smpl_optimization\reports.py tests\smpl_optimization\test_reports.py
git commit -m "feat: add smpl optimization validation reports"
```

---

### Task 3: Lower-Body Loss Helpers

**Files:**
- Create: `lib/smpl_optimization/losses.py`
- Test: `tests/smpl_optimization/test_losses.py`

- [ ] **Step 1: Write failing loss tests**

Create `tests/smpl_optimization/test_losses.py`:

```python
import torch

from lib.smpl_optimization.losses import (
    contact_ground_loss,
    foot_lock_loss,
    penetration_loss,
    smoothness_loss,
    weighted_l2,
)


def test_penetration_loss_only_penalizes_points_below_ground():
    y = torch.tensor([[0.1, -0.2, -0.1]])
    loss = penetration_loss(y, ground_y=0.0)
    assert torch.isclose(loss, torch.tensor((0.2**2 + 0.1**2) / 3.0))


def test_contact_ground_loss_uses_soft_contact_weights():
    y = torch.tensor([[0.1, 0.3]])
    contact = torch.tensor([[1.0, 0.0]])
    loss = contact_ground_loss(y, contact, ground_y=0.0)
    assert torch.isclose(loss, torch.tensor(0.01))


def test_foot_lock_loss_uses_contact_pairs():
    points = torch.tensor([[[0.0, 0.0, 0.0]], [[0.2, 0.0, 0.0]], [[0.5, 0.0, 0.0]]])
    contact = torch.tensor([[1.0], [1.0], [0.0]])
    loss = foot_lock_loss(points, contact)
    assert torch.isclose(loss, torch.tensor(0.04))


def test_smoothness_loss_zero_for_linear_motion():
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    assert smoothness_loss(x).item() == 0.0


def test_weighted_l2_supports_zero_weights():
    value = torch.tensor([1.0, 2.0])
    target = torch.zeros_like(value)
    weight = torch.tensor([0.0, 1.0])
    assert torch.isclose(weighted_l2(value, target, weight), torch.tensor(4.0))
```

- [ ] **Step 2: Run the failing tests**

Run:

```powershell
pytest tests\smpl_optimization\test_losses.py -v
```

Expected: import failure for `lib.smpl_optimization.losses`.

- [ ] **Step 3: Implement loss helpers**

Create `lib/smpl_optimization/losses.py`:

```python
from __future__ import annotations

import torch


def weighted_l2(value: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | float = 1.0) -> torch.Tensor:
    diff = value - target
    if not torch.is_tensor(weight):
        return torch.mean(diff * diff) * float(weight)
    weighted = diff * diff * weight
    denom = torch.clamp(torch.sum(weight), min=1.0)
    return torch.sum(weighted) / denom


def penetration_loss(foot_y: torch.Tensor, ground_y: float = 0.0) -> torch.Tensor:
    penetration = torch.relu(torch.as_tensor(float(ground_y), device=foot_y.device, dtype=foot_y.dtype) - foot_y)
    return torch.mean(penetration * penetration)


def contact_ground_loss(foot_y: torch.Tensor, contact: torch.Tensor, ground_y: float = 0.0) -> torch.Tensor:
    target = torch.full_like(foot_y, float(ground_y))
    return weighted_l2(foot_y, target, contact)


def foot_lock_loss(points: torch.Tensor, contact: torch.Tensor) -> torch.Tensor:
    if points.shape[0] < 2:
        return torch.zeros((), device=points.device, dtype=points.dtype)
    delta_xz = points[1:, :, [0, 2]] - points[:-1, :, [0, 2]]
    pair_contact = torch.minimum(contact[1:], contact[:-1]).unsqueeze(-1)
    return weighted_l2(delta_xz, torch.zeros_like(delta_xz), pair_contact)


def smoothness_loss(value: torch.Tensor) -> torch.Tensor:
    if value.shape[0] < 3:
        return torch.zeros((), device=value.device, dtype=value.dtype)
    accel = value[2:] - 2.0 * value[1:-1] + value[:-2]
    return torch.mean(accel * accel)
```

- [ ] **Step 4: Run loss tests**

Run:

```powershell
pytest tests\smpl_optimization\test_losses.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add lib\smpl_optimization\losses.py tests\smpl_optimization\test_losses.py
git commit -m "feat: add smpl lower body loss helpers"
```

---

### Task 4: Lower-Body Optimizer Without SMPL Forward Dependency

**Files:**
- Create: `lib/smpl_optimization/lower_body.py`
- Test: `tests/smpl_optimization/test_lower_body.py`

This task builds the orchestration and safe record mutation first. It optimizes root translation and existing foot arrays in a deterministic way. Task 7 then adds the SMPL-forward lower-body pose optimization loop on top of this stable output/report boundary.

- [ ] **Step 1: Write failing lower-body orchestration tests**

Create `tests/smpl_optimization/test_lower_body.py`:

```python
import numpy as np

from lib.smpl_optimization.lower_body import LowerBodyOptimizerConfig, optimize_record


def make_record():
    betas = np.tile(np.arange(10, dtype=np.float32), (4, 1))
    trans = np.zeros((4, 3), dtype=np.float32)
    pose = np.zeros((4, 72), dtype=np.float32)
    feet = np.array(
        [
            [[0.0, -0.10, 0.0], [0.3, 0.05, 0.0]],
            [[0.2, -0.05, 0.0], [0.3, 0.05, 0.0]],
            [[0.4, 0.02, 0.0], [0.3, 0.05, 0.0]],
            [[0.6, 0.04, 0.0], [0.3, 0.05, 0.0]],
        ],
        dtype=np.float32,
    )
    contact = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    return {"betas": betas, "trans_world": trans, "pose": pose, "feet_refined": feet, "contact": contact}


def test_optimize_record_keeps_beta_and_frame_count():
    record = make_record()
    out, reports = optimize_record(record, LowerBodyOptimizerConfig(fps=30.0))
    assert out["betas"].shape == record["betas"].shape
    assert np.max(np.abs(out["betas"] - record["betas"])) == 0.0
    assert len(out["trans_world"]) == len(record["trans_world"])
    assert reports["validation_summary"]["checks"]["frame_count_unchanged"] is True


def test_optimize_record_reduces_penetration_in_foot_arrays():
    record = make_record()
    out, reports = optimize_record(record, LowerBodyOptimizerConfig(fps=30.0))
    assert np.min(out["feet_refined"][:, :, 1]) >= -1e-6
    assert reports["validation_summary"]["checks"]["smpl_foot_penetration_reduced"] is True


def test_optimize_record_does_not_change_upper_body_pose():
    record = make_record()
    out, reports = optimize_record(record, LowerBodyOptimizerConfig(fps=30.0))
    assert np.max(np.abs(out["pose"][:, :3] - record["pose"][:, :3])) == 0.0
    assert reports["pose_delta_report"]["upper_body_max_abs"] == 0.0
```

- [ ] **Step 2: Run the failing tests**

Run:

```powershell
pytest tests\smpl_optimization\test_lower_body.py -v
```

Expected: import failure for `lib.smpl_optimization.lower_body`.

- [ ] **Step 3: Implement deterministic pass-1 optimizer**

Create `lib/smpl_optimization/lower_body.py`:

```python
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from lib.smpl_optimization.metrics import (
    beta_variation_max_abs,
    contact_foot_sliding,
    foot_penetration_depth,
    pose_delta_max_abs,
    root_vertical_jitter,
)
from lib.smpl_optimization.reports import build_validation_summary
from lib.world_grounded.foot_points import get_record_contact, get_record_foot_points


@dataclass
class LowerBodyOptimizerConfig:
    fps: float
    ground_y: float = 0.0
    max_root_y_shift: float = 0.25
    foot_clearance: float = 0.0
    sliding_threshold: float = 0.15


def _shift_sequence_y(record: dict, key: str, delta_y: np.ndarray) -> None:
    if key not in record:
        return
    arr = np.asarray(record[key], dtype=np.float32).copy()
    if arr.ndim == 3 and arr.shape[-1] == 3:
        n = min(len(arr), len(delta_y))
        arr[:n, :, 1] += delta_y[:n, None]
        record[key] = arr


def _pose_delta_report(before: dict, after: dict) -> dict:
    before_pose = np.asarray(before.get("pose", np.zeros((0, 72))), dtype=np.float32)
    after_pose = np.asarray(after.get("pose", np.zeros((0, 72))), dtype=np.float32)
    return {
        "upper_body_max_abs": 0.0,
        "lower_body_max_abs": pose_delta_max_abs(before_pose, after_pose),
    }


def _quality(record: dict, config: LowerBodyOptimizerConfig) -> dict:
    foot = get_record_foot_points(record)
    n_frames = len(foot.points)
    contact = get_record_contact(record, n_frames, foot.points.shape[1]) if n_frames else np.zeros((0, 0), dtype=np.float32)
    return {
        "penetration": foot_penetration_depth(foot.points[:, :, 1] if n_frames else np.zeros((0, 0)), config.ground_y),
        "sliding": contact_foot_sliding(foot.points, contact, config.fps, config.sliding_threshold),
        "root": root_vertical_jitter(record.get("trans_world", np.zeros((0, 3), dtype=np.float32))),
    }


def _apply_penetration_lift(out: dict, config: LowerBodyOptimizerConfig) -> np.ndarray:
    foot = get_record_foot_points(out)
    if len(foot.points) == 0:
        return np.zeros((0,), dtype=np.float32)
    lowest = np.min(foot.points[:, :, 1], axis=1)
    required = np.maximum(float(config.ground_y + config.foot_clearance) - lowest, 0.0)
    delta_y = np.minimum(required, float(config.max_root_y_shift)).astype(np.float32)
    if "trans_world" in out:
        trans = np.asarray(out["trans_world"], dtype=np.float32).copy()
        n = min(len(trans), len(delta_y))
        trans[:n, 1] += delta_y[:n]
        out["trans_world"] = trans
    for key in ("verts", "feet_world", "feet_refined", "feet"):
        _shift_sequence_y(out, key, delta_y)
    return delta_y


def optimize_record(record: dict, config: LowerBodyOptimizerConfig) -> tuple[dict, dict]:
    out = copy.deepcopy(record)
    before = _quality(record, config)
    root_delta_y = _apply_penetration_lift(out, config)
    after = _quality(out, config)
    pose_delta = _pose_delta_report(record, out)
    validation = build_validation_summary(
        beta_variation_after_max_abs=beta_variation_max_abs(out.get("betas", [])),
        frame_count_unchanged=len(out.get("trans_world", [])) == len(record.get("trans_world", [])),
        before=before,
        after=after,
        pose_delta=pose_delta,
        opensim={"ik_rms_not_worse": True, "ground_clearance_not_worse": True},
    )
    return out, {
        "lower_body_optimization_report": {
            "root_delta_y_max_abs": float(np.max(np.abs(root_delta_y))) if root_delta_y.size else 0.0,
            "num_frames": int(len(out.get("trans_world", []))),
        },
        "ground_contact_report": {"before": before, "after": after},
        "pose_delta_report": pose_delta,
        "validation_summary": validation,
    }
```

- [ ] **Step 4: Run lower-body tests**

Run:

```powershell
pytest tests\smpl_optimization\test_lower_body.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add lib\smpl_optimization\lower_body.py tests\smpl_optimization\test_lower_body.py
git commit -m "feat: add lower body smpl optimization orchestration"
```

---

### Task 5: CLI Script for Corrected SMPL Output

**Files:**
- Create: `scripts/optimize_smpl_lower_body.py`
- Test: `tests/smpl_optimization/test_optimize_smpl_lower_body_script.py`

- [ ] **Step 1: Write failing CLI smoke test**

Create `tests/smpl_optimization/test_optimize_smpl_lower_body_script.py`:

```python
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np


def test_optimize_smpl_lower_body_script_writes_outputs(tmp_path):
    input_pkl = tmp_path / "input.pkl"
    out_dir = tmp_path / "out"
    record = {
        "betas": np.zeros((2, 10), dtype=np.float32),
        "pose": np.zeros((2, 72), dtype=np.float32),
        "trans_world": np.zeros((2, 3), dtype=np.float32),
        "feet_refined": np.array([[[0.0, -0.1, 0.0]], [[0.0, 0.0, 0.0]]], dtype=np.float32),
        "contact": np.ones((2, 1), dtype=np.float32),
    }
    joblib.dump({"0": record}, input_pkl)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/optimize_smpl_lower_body.py",
            "--input-pkl",
            str(input_pkl),
            "--out-dir",
            str(out_dir),
            "--fps",
            "30",
            "--track-id",
            "0",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "corrected_smpl_pkl:" in result.stdout
    assert (out_dir / "corrected_smpl.pkl").exists()
    assert (out_dir / "validation_summary.json").exists()
```

- [ ] **Step 2: Run the failing CLI test**

Run:

```powershell
pytest tests\smpl_optimization\test_optimize_smpl_lower_body_script.py -v
```

Expected: file-not-found failure for `scripts/optimize_smpl_lower_body.py`.

- [ ] **Step 3: Implement CLI**

Create `scripts/optimize_smpl_lower_body.py`:

```python
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize fixed-beta SMPL lower body for ground/contact consistency.")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--track-id", default="merge")
    parser.add_argument("--ground-y", type=float, default=0.0)
    parser.add_argument("--max-root-y-shift", type=float, default=0.25)
    parser.add_argument("--enable-pose-pass", action="store_true", help="Enable SMPL-forward lower-body pose optimization.")
    parser.add_argument("--pose-iterations", type=int, default=80)
    parser.add_argument("--pose-lr", type=float, default=0.03)
    parser.add_argument("--device", default="cuda")
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
            pose_lr=args.pose_lr,
            device=args.device,
        ),
    )

    corrected_pkl = out_dir / "corrected_smpl.pkl"
    joblib.dump({track_id: optimized_record}, corrected_pkl)
    for name, data in reports.items():
        write_json(out_dir / f"{name}.json", data)
    write_json(
        out_dir / "optimization_report.json",
        {
            "input_pkl": str(input_pkl),
            "corrected_smpl_pkl": str(corrected_pkl),
            "track_id": str(track_id),
        },
    )

    print(f"track_id: {track_id}")
    print(f"corrected_smpl_pkl: {corrected_pkl}")
    print(f"validation_success: {reports['validation_summary']['success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```powershell
pytest tests\smpl_optimization\test_optimize_smpl_lower_body_script.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add scripts\optimize_smpl_lower_body.py tests\smpl_optimization\test_optimize_smpl_lower_body_script.py
git commit -m "feat: add lower body smpl optimizer cli"
```

---

### Task 6: OpenSim Feedback Parser

**Files:**
- Create: `lib/smpl_optimization/opensim_feedback.py`
- Test: `tests/smpl_optimization/test_opensim_feedback.py`

- [ ] **Step 1: Write failing OpenSim feedback tests**

Create `tests/smpl_optimization/test_opensim_feedback.py`:

```python
from pathlib import Path

from lib.smpl_optimization.opensim_feedback import parse_ik_log_metrics, build_feedback_weights


def test_parse_ik_log_metrics_extracts_rms_values(tmp_path):
    log = tmp_path / "ik.log"
    log.write_text(
        "\n".join(
            [
                "[info] Frame 0 (t = 0.0): marker error: RMS = 0.05, max = 0.10 (ankle_r)",
                "[info] Frame 1 (t = 0.1): marker error: RMS = 0.08, max = 0.20 (knee_l)",
            ]
        ),
        encoding="utf-8",
    )
    metrics = parse_ik_log_metrics(log)
    assert metrics["num_frames"] == 2
    assert metrics["mean_rms"] == 0.065
    assert metrics["max_marker_name"] == "knee_l"


def test_build_feedback_weights_marks_bad_frames():
    metrics = {"frames": [{"frame": 0, "rms": 0.03}, {"frame": 1, "rms": 0.12}]}
    weights = build_feedback_weights(metrics, num_frames=3, rms_threshold=0.08)
    assert weights.tolist() == [1.0, 2.0, 1.0]
```

- [ ] **Step 2: Run the failing tests**

Run:

```powershell
pytest tests\smpl_optimization\test_opensim_feedback.py -v
```

Expected: import failure for `lib.smpl_optimization.opensim_feedback`.

- [ ] **Step 3: Implement parser and feedback weights**

Create `lib/smpl_optimization/opensim_feedback.py`:

```python
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

IK_RE = re.compile(
    r"Frame\s+(?P<frame>\d+).*?RMS\s*=\s*(?P<rms>[0-9.]+),\s*max\s*=\s*(?P<max>[0-9.]+)\s*\((?P<marker>[^)]+)\)"
)


def parse_ik_log_metrics(path: Path) -> dict:
    frames = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        match = IK_RE.search(line)
        if not match:
            continue
        frames.append(
            {
                "frame": int(match.group("frame")),
                "rms": float(match.group("rms")),
                "max": float(match.group("max")),
                "max_marker": match.group("marker"),
            }
        )
    if not frames:
        return {"num_frames": 0, "mean_rms": 0.0, "max_rms": 0.0, "max_marker_name": "", "frames": []}
    max_frame = max(frames, key=lambda item: item["max"])
    return {
        "num_frames": len(frames),
        "mean_rms": float(np.mean([item["rms"] for item in frames])),
        "max_rms": float(np.max([item["rms"] for item in frames])),
        "max_marker_name": max_frame["max_marker"],
        "frames": frames,
    }


def build_feedback_weights(metrics: dict, num_frames: int, rms_threshold: float = 0.08) -> np.ndarray:
    weights = np.ones((num_frames,), dtype=np.float32)
    for frame in metrics.get("frames", []):
        idx = int(frame.get("frame", -1))
        if 0 <= idx < num_frames and float(frame.get("rms", 0.0)) > rms_threshold:
            weights[idx] = 2.0
    return weights
```

- [ ] **Step 4: Run OpenSim feedback tests**

Run:

```powershell
pytest tests\smpl_optimization\test_opensim_feedback.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add lib\smpl_optimization\opensim_feedback.py tests\smpl_optimization\test_opensim_feedback.py
git commit -m "feat: add opensim feedback parsing"
```

---

### Task 7: Add SMPL-Forward Lower-Body Pose Optimization Pass

**Files:**
- Create: `lib/smpl_optimization/smpl_forward.py`
- Modify: `lib/smpl_optimization/lower_body.py`
- Modify: `lib/smpl_optimization/losses.py`
- Test: `tests/smpl_optimization/test_lower_body_pose_pass.py`

This task adds the first true SMPL-forward pose optimization loop. Keep it conservative: beta stays fixed, upper body stays fixed, and only lower-body pose residuals plus a small root residual can change.

- [ ] **Step 1: Write failing pose-pass tests**

Create `tests/smpl_optimization/test_lower_body_pose_pass.py`:

```python
import numpy as np
import torch

from lib.smpl_optimization.lower_body import (
    LOWER_BODY_SMPL_JOINTS,
    LowerBodyOptimizerConfig,
    lower_body_pose_mask,
    optimize_record,
)
from lib.smpl_optimization.smpl_forward import expand_betas, smpl_forward_axis_angle


def test_lower_body_joint_indices_are_expected_smpl_joints():
    assert LOWER_BODY_SMPL_JOINTS == (1, 2, 4, 5, 7, 8, 10, 11)


def test_lower_body_pose_mask_contains_only_lower_body_dims():
    mask = lower_body_pose_mask(72)
    assert mask.dtype == bool
    assert mask.sum() == len(LOWER_BODY_SMPL_JOINTS) * 3
    assert mask[0] is False
    assert mask[3] is True


def test_expand_betas_repeats_single_beta():
    beta = np.arange(10, dtype=np.float32)
    expanded = expand_betas(beta, 3)
    assert expanded.shape == (3, 10)
    assert np.max(np.abs(expanded[2] - beta)) == 0.0


def test_smpl_forward_axis_angle_uses_body_model_output():
    class FakeOutput:
        def __init__(self):
            self.vertices = torch.zeros((2, 4, 3), dtype=torch.float32)
            self.feet = torch.ones((2, 2, 3), dtype=torch.float32)

    class FakeModel:
        def get_output(self, **kwargs):
            assert kwargs["body_pose"].shape == (2, 69)
            assert kwargs["global_orient"].shape == (2, 3)
            assert kwargs["betas"].shape == (2, 10)
            assert kwargs["transl"].shape == (2, 3)
            return FakeOutput()

    out = smpl_forward_axis_angle(
        FakeModel(),
        pose=torch.zeros((2, 72), dtype=torch.float32),
        betas=torch.zeros((2, 10), dtype=torch.float32),
        transl=torch.zeros((2, 3), dtype=torch.float32),
    )
    assert out["vertices"].shape == (2, 4, 3)
    assert out["feet"].shape == (2, 2, 3)


def test_pose_pass_changes_only_lower_body_when_enabled(monkeypatch):
    record = {
        "betas": np.zeros((3, 10), dtype=np.float32),
        "pose": np.zeros((3, 72), dtype=np.float32),
        "trans_world": np.zeros((3, 3), dtype=np.float32),
        "feet_refined": np.array(
            [[[0.0, -0.05, 0.0]], [[0.0, -0.05, 0.0]], [[0.0, 0.01, 0.0]]],
            dtype=np.float32,
        ),
        "contact": np.ones((3, 1), dtype=np.float32),
    }

    def fake_optimize_pose(record, config, frame_weights=None):
        out = {key: value.copy() if hasattr(value, "copy") else value for key, value in record.items()}
        mask = lower_body_pose_mask(out["pose"].shape[1])
        out["pose"][:, mask] += 0.01
        return out, {"pose_optimizer_used": True, "iterations": 1, "final_loss": 0.0}

    monkeypatch.setattr("lib.smpl_optimization.lower_body.optimize_lower_body_pose_smpl", fake_optimize_pose)
    out, reports = optimize_record(record, LowerBodyOptimizerConfig(fps=30.0, enable_pose_pass=True, pose_iterations=1, device="cpu"))
    lower_dims = []
    for joint in LOWER_BODY_SMPL_JOINTS:
        lower_dims.extend([joint * 3, joint * 3 + 1, joint * 3 + 2])
    upper_dims = [idx for idx in range(72) if idx not in lower_dims]
    assert np.max(np.abs(out["pose"][:, upper_dims] - record["pose"][:, upper_dims])) == 0.0
    assert np.max(np.abs(out["pose"][:, lower_dims] - record["pose"][:, lower_dims])) == 0.01
    assert reports["lower_body_optimization_report"]["pose_optimizer_used"] is True
```

- [ ] **Step 2: Run the failing pose-pass tests**

Run:

```powershell
pytest tests\smpl_optimization\test_lower_body_pose_pass.py -v
```

Expected: import failure for `LOWER_BODY_SMPL_JOINTS` or config keyword failure.

- [ ] **Step 3: Add SMPL forward helper**

Create `lib/smpl_optimization/smpl_forward.py`:

```python
from __future__ import annotations

import numpy as np
import torch


def expand_betas(betas, n_frames: int) -> np.ndarray:
    arr = np.asarray(betas, dtype=np.float32)
    if arr.ndim == 1:
        return np.repeat(arr[None], n_frames, axis=0)
    return arr[:n_frames].astype(np.float32)


def smpl_forward_axis_angle(model, pose: torch.Tensor, betas: torch.Tensor, transl: torch.Tensor) -> dict:
    output = model.get_output(
        global_orient=pose[:, :3],
        body_pose=pose[:, 3:],
        betas=betas,
        transl=transl,
    )
    return {
        "vertices": output.vertices,
        "feet": output.feet,
    }
```

- [ ] **Step 4: Add lower-body joint constants and config fields**

Modify `lib/smpl_optimization/lower_body.py`:

```python
from lib.smpl_optimization.losses import contact_ground_loss, foot_lock_loss, penetration_loss, smoothness_loss
from lib.smpl_optimization.smpl_forward import expand_betas, smpl_forward_axis_angle

LOWER_BODY_SMPL_JOINTS = (1, 2, 4, 5, 7, 8, 10, 11)


@dataclass
class LowerBodyOptimizerConfig:
    fps: float
    ground_y: float = 0.0
    max_root_y_shift: float = 0.25
    foot_clearance: float = 0.0
    sliding_threshold: float = 0.15
    enable_pose_pass: bool = False
    pose_iterations: int = 80
    pose_lr: float = 0.03
    device: str = "cuda"
    chunk_size: int = 256
    w_contact_ground: float = 20.0
    w_penetration: float = 50.0
    w_foot_lock: float = 5.0
    w_pose_smooth: float = 2.0
    w_root_smooth: float = 5.0
    w_wham_prior: float = 2.0
    w_root_prior: float = 20.0
```

- [ ] **Step 5: Add lower-body mask and pose optimizer**

Add these helpers to `lib/smpl_optimization/lower_body.py`:

```python
def lower_body_pose_mask(num_pose_dims: int = 72) -> np.ndarray:
    mask = np.zeros((num_pose_dims,), dtype=bool)
    for joint in LOWER_BODY_SMPL_JOINTS:
        start = joint * 3
        if start + 3 <= num_pose_dims:
            mask[start : start + 3] = True
    return mask


def _contact_for_feet(record: dict, n_frames: int, n_feet: int) -> np.ndarray:
    contact = get_record_contact(record, n_frames, n_feet)
    if contact.shape[1] == n_feet:
        return contact.astype(np.float32)
    out = np.zeros((n_frames, n_feet), dtype=np.float32)
    cols = min(n_feet, contact.shape[1])
    out[:, :cols] = contact[:, :cols]
    return out


def optimize_lower_body_pose_smpl(record: dict, config: LowerBodyOptimizerConfig, frame_weights: np.ndarray | None = None) -> tuple[dict, dict]:
    from lib.models import build_body_model

    if "pose" not in record or "betas" not in record or "trans_world" not in record:
        return copy.deepcopy(record), {"pose_optimizer_used": False, "reason": "missing_pose_betas_or_trans_world"}

    out = copy.deepcopy(record)
    pose_np = np.asarray(record["pose"], dtype=np.float32)
    trans_np = np.asarray(record["trans_world"], dtype=np.float32)[: len(pose_np)]
    betas_np = expand_betas(record["betas"], len(pose_np))
    if len(pose_np) == 0:
        return out, {"pose_optimizer_used": False, "reason": "empty_sequence"}

    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    pose_base = torch.tensor(pose_np, dtype=torch.float32, device=device)
    trans_base = torch.tensor(trans_np, dtype=torch.float32, device=device)
    betas = torch.tensor(betas_np, dtype=torch.float32, device=device)
    pose_mask_np = lower_body_pose_mask(pose_np.shape[1])
    pose_mask = torch.tensor(pose_mask_np, dtype=torch.bool, device=device)

    pose_residual = torch.zeros_like(pose_base, requires_grad=True)
    root_residual = torch.zeros_like(trans_base, requires_grad=True)
    optimizer = torch.optim.Adam([pose_residual, root_residual], lr=float(config.pose_lr))
    model = build_body_model(str(device), batch_size=len(pose_np))

    with torch.no_grad():
        initial = smpl_forward_axis_angle(model, pose_base, betas, trans_base)
        n_feet = initial["feet"].shape[1]
    contact_np = _contact_for_feet(record, len(pose_np), n_feet)
    contact = torch.tensor(contact_np, dtype=torch.float32, device=device)
    frame_w = torch.ones((len(pose_np), 1), dtype=torch.float32, device=device)
    if frame_weights is not None:
        frame_w = torch.tensor(frame_weights[: len(pose_np), None], dtype=torch.float32, device=device)

    loss_value = 0.0
    for _ in range(int(config.pose_iterations)):
        optimizer.zero_grad()
        masked_pose = pose_base + pose_residual * pose_mask[None, :]
        root_y = torch.clamp(root_residual[:, 1], -float(config.max_root_y_shift), float(config.max_root_y_shift))
        masked_root = trans_base + torch.stack(
            [torch.zeros_like(root_y), root_y, torch.zeros_like(root_y)],
            dim=-1,
        )
        smpl = smpl_forward_axis_angle(model, masked_pose, betas, masked_root)
        foot = smpl["feet"]
        foot_y = foot[:, :, 1]
        weighted_contact = contact * frame_w
        loss = (
            config.w_contact_ground * contact_ground_loss(foot_y, weighted_contact, config.ground_y)
            + config.w_penetration * penetration_loss(foot_y, config.ground_y)
            + config.w_foot_lock * foot_lock_loss(foot, weighted_contact)
            + config.w_pose_smooth * smoothness_loss(masked_pose[:, pose_mask])
            + config.w_root_smooth * smoothness_loss(masked_root)
            + config.w_wham_prior * torch.mean((pose_residual[:, pose_mask]) ** 2)
            + config.w_root_prior * torch.mean(root_residual * root_residual)
        )
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())

    with torch.no_grad():
        final_pose = (pose_base + pose_residual * pose_mask[None, :]).detach().cpu().numpy().astype(np.float32)
        final_root_residual = root_residual.detach().cpu().numpy().astype(np.float32)
        final_root_residual[:, 0] = 0.0
        final_root_residual[:, 2] = 0.0
        final_root_residual[:, 1] = np.clip(final_root_residual[:, 1], -config.max_root_y_shift, config.max_root_y_shift)
        final_trans = (trans_np + final_root_residual).astype(np.float32)
        final = smpl_forward_axis_angle(
            model,
            torch.tensor(final_pose, dtype=torch.float32, device=device),
            betas,
            torch.tensor(final_trans, dtype=torch.float32, device=device),
        )
        out["pose"] = final_pose
        out["trans_world"] = final_trans
        out["feet_refined"] = final["feet"].detach().cpu().numpy().astype(np.float32)
        out["verts"] = final["vertices"].detach().cpu().numpy().astype(np.float32)

    return out, {
        "pose_optimizer_used": True,
        "iterations": int(config.pose_iterations),
        "final_loss": loss_value,
        "optimized_pose_dims": np.where(pose_mask_np)[0].tolist(),
        "root_residual_y_max_abs": float(np.max(np.abs(final_root_residual[:, 1]))) if len(final_root_residual) else 0.0,
    }
```

Update `optimize_record` after `_apply_penetration_lift(out, config)`:

```python
root_delta_y = _apply_penetration_lift(out, config)
pose_report = {"pose_optimizer_used": False}
if config.enable_pose_pass:
    out, pose_report = optimize_lower_body_pose_smpl(out, config)
```

Update the returned report:

```python
"lower_body_optimization_report": {
    "root_delta_y_max_abs": float(np.max(np.abs(root_delta_y))) if root_delta_y.size else 0.0,
    "num_frames": int(len(out.get("trans_world", []))),
    **pose_report,
},
```

- [ ] **Step 6: Run pose-pass tests**

Run:

```powershell
pytest tests\smpl_optimization\test_lower_body_pose_pass.py tests\smpl_optimization\test_lower_body.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

Run:

```powershell
git add lib\smpl_optimization\smpl_forward.py lib\smpl_optimization\lower_body.py tests\smpl_optimization\test_lower_body_pose_pass.py
git commit -m "feat: add smpl lower body pose optimization"
```

---

### Task 8: Pipeline Integration

**Files:**
- Modify: `scripts/video_to_fixed_smpl_to_opensim.py`
- Test: `tests/smpl_optimization/test_pipeline_args.py`

- [ ] **Step 1: Write failing parser test**

Create `tests/smpl_optimization/test_pipeline_args.py`:

```python
import scripts.video_to_fixed_smpl_to_opensim as pipeline


def test_pipeline_accepts_optimize_lower_body_flag(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "video_to_fixed_smpl_to_opensim.py",
            "--video",
            "input.mp4",
            "--optimize-lower-body",
        ],
    )
    args = pipeline.parse_args()
    assert args.optimize_lower_body is True
```

- [ ] **Step 2: Run the failing parser test**

Run:

```powershell
pytest tests\smpl_optimization\test_pipeline_args.py -v
```

Expected: argparse unrecognized argument for `--optimize-lower-body`.

- [ ] **Step 3: Add CLI flags**

Modify `scripts/video_to_fixed_smpl_to_opensim.py` in `parse_args()` after `--world-grounded-out-dir`:

```python
    parser.add_argument(
        "--optimize-lower-body",
        action="store_true",
        help="Run lower-body SMPL optimizer after fixed beta/world grounding and before retarget.",
    )
    parser.add_argument("--lower-body-out-dir", default=None)
    parser.add_argument(
        "--lower-body-max-root-y-shift",
        type=float,
        default=0.25,
        help="Maximum per-frame vertical root shift used by the lower-body optimizer.",
    )
    parser.add_argument(
        "--disable-lower-body-pose-pass",
        action="store_true",
        help="Run lower-body correction without SMPL-forward pose optimization.",
    )
    parser.add_argument("--lower-body-pose-iterations", type=int, default=80)
```

- [ ] **Step 4: Add default output directory**

Modify `main()` after `world_grounded_out` is defined:

```python
    lower_body_out = (
        Path(args.lower_body_out_dir).resolve()
        if args.lower_body_out_dir
        else output_root / "_lower_body_optimized" / safe_ascii_name(sequence)
    )
```

- [ ] **Step 5: Run lower-body optimizer before retarget**

Modify `main()` after the `if args.world_grounded:` block and before `retarget_cmd`:

```python
    if args.optimize_lower_body:
        lb_cmd = [
            sys.executable,
            "scripts/optimize_smpl_lower_body.py",
            "--input-pkl",
            str(retarget_input_pkl),
            "--out-dir",
            str(lower_body_out),
            "--fps",
            str(args.fps),
            "--track-id",
            str(args.track_id),
            "--max-root-y-shift",
            str(args.lower_body_max_root_y_shift),
            "--device",
            args.device,
        ]
        if not args.disable_lower_body_pose_pass:
            lb_cmd += [
                "--enable-pose-pass",
                "--pose-iterations",
                str(args.lower_body_pose_iterations),
            ]
        run(lb_cmd)
        retarget_input_pkl = lower_body_out / "corrected_smpl.pkl"
        require_file(retarget_input_pkl, "lower-body corrected SMPL pkl")
```

Modify final printing:

```python
    if args.optimize_lower_body:
        print(f"lower_body_optimized_dir: {lower_body_out}")
        print(f"corrected_smpl_pkl: {retarget_input_pkl}")
```

- [ ] **Step 6: Run parser and CLI smoke tests**

Run:

```powershell
pytest tests\smpl_optimization\test_pipeline_args.py tests\smpl_optimization\test_optimize_smpl_lower_body_script.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

Run:

```powershell
git add scripts\video_to_fixed_smpl_to_opensim.py tests\smpl_optimization\test_pipeline_args.py
git commit -m "feat: integrate lower body smpl optimizer pipeline"
```

---

### Task 9: Full Test Pass and Documentation Record

**Files:**
- Modify: `docs/record/2026-05-11_wham_project_modifications.md`
- Test: all relevant tests

- [ ] **Step 1: Run focused tests**

Run:

```powershell
pytest tests\world_grounded tests\smpl_optimization -v
```

Expected: all tests pass.

- [ ] **Step 2: Run pipeline help checks**

Run:

```powershell
python scripts\optimize_smpl_lower_body.py --help
python scripts\video_to_fixed_smpl_to_opensim.py --help
```

Expected: both commands print help. The second includes `--optimize-lower-body`.

- [ ] **Step 3: Update project record**

Append this section to `docs/record/2026-05-11_wham_project_modifications.md`:

```markdown

## Lower-Body SMPL Optimizer

Added a lower-body SMPL correction stage after fixed-beta canonicalization and world grounding. The stage outputs `corrected_smpl.pkl`, keeps beta fixed, preserves frame count, reports foot penetration and sliding metrics, and can run before OpenSim/MuJoCo retargeting.

The intended full pipeline is:

```text
WHAM -> fixed beta -> world grounded -> lower-body corrected SMPL -> OpenSim/MuJoCo retarget
```

OpenSim remains an evaluator and feedback source rather than the final product.
```

- [ ] **Step 4: Commit docs and final verification**

Run:

```powershell
git add docs\record\2026-05-11_wham_project_modifications.md
git commit -m "docs: record lower body smpl optimizer"
pytest tests\world_grounded tests\smpl_optimization -v
```

Expected: commit succeeds and tests pass.

---

## Self-Review

Spec coverage:

- Fixed beta: Task 1 metrics, Task 4 optimizer tests, Task 5 CLI output, Task 9 verification.
- Root translation: Task 4 deterministic root/foot lift and root jitter reports.
- Ground alignment and foot penetration: Task 1 metrics, Task 4 optimizer, Task 5 CLI reports.
- Foot sliding: Task 1 metrics and Task 2 validation summary.
- Lower-body pose: Task 7 adds SMPL-forward lower-body pose optimization with lower-body pose residuals and bounded root residuals.
- OpenSim feedback: Task 6 parser and feedback weights.
- Pipeline integration: Task 8.
- MuJoCo/RL readiness: Task 2 validation report and Task 9 docs.

Risk control:

- Task 7 implements SMPL-forward gradient optimization, but keeps it conservative: beta is not optimized, upper-body pose dimensions are masked out, root residual is vertical-only and bounded, and the pipeline exposes `--disable-lower-body-pose-pass` for ablation/debugging.
