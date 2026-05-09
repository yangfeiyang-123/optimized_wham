# World-Grounded SMPL Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first-version SMPL-layer world-grounded optimizer that uses WHAM contact/feet signals to estimate ground, smooth/correct root translation, preserve fixed beta, and export OpenSim MOT through the existing retarget pipeline.

**Architecture:** Add a focused `lib/world_grounded/` package for track merging, foot/contact extraction, ground estimation, root correction, reports, and pkl writing. Extend WHAM output to save contact/feet and add a script plus pipeline flag that runs optimizer before free-root OpenSim retarget.

**Tech Stack:** Python, NumPy, SciPy where already available, PyTorch/SMPL for fallback foot extraction, joblib, pytest, existing WHAM/OpenSim scripts.

---

## File Structure

- Modify `demo.py`: save WHAM `contact`, `feet_world`, and `feet_refined` when available.
- Modify `scripts/canonicalize_wham_fixed_beta.py`: preserve and length-trim contact/feet fields when writing canonical records.
- Create `lib/world_grounded/__init__.py`: package export surface.
- Create `lib/world_grounded/tracks.py`: track length, frame ids, and merge policy shared with optimizer.
- Create `lib/world_grounded/foot_points.py`: preferred WHAM feet provider and SMPL fallback provider.
- Create `lib/world_grounded/ground.py`: weighted median, contact confidence, ground estimator, contact report.
- Create `lib/world_grounded/root_optimizer.py`: root translation smoothing and vertical ground correction.
- Create `lib/world_grounded/reports.py`: quality metrics and JSON serialization helpers.
- Create `scripts/world_grounded_smpl_optimizer.py`: CLI entry for SMPL optimizer.
- Modify `scripts/video_to_fixed_smpl_to_opensim.py`: add `--world-grounded` and retarget optimized pkl when enabled.
- Add tests under `tests/world_grounded/`.

---

### Task 1: Save WHAM Contact and Feet Signals

**Files:**
- Modify: `demo.py`
- Test: manual pkl inspection command

- [ ] **Step 1: Add contact/feet fields to WHAM result saving**

In `demo.py`, inside the `# ========= Store results ========= #` block after existing assignments:

```python
        if "contact" in pred:
            results[_id]["contact"] = pred["contact"].detach().cpu().squeeze(0).numpy()
        if "feet" in pred:
            results[_id]["feet_world"] = pred["feet"].detach().cpu().squeeze(0).numpy()
        if "feet_refined" in pred:
            results[_id]["feet_refined"] = pred["feet_refined"].detach().cpu().squeeze(0).numpy()
```

Expected field shapes:

```text
contact: [T, 4]
feet_world: [T, 4, 3]
feet_refined: [T, 4, 3] when available
```

- [ ] **Step 2: Run syntax check**

Run:

```powershell
python -m py_compile demo.py
```

Expected: exit code 0.

- [ ] **Step 3: Run a short WHAM smoke test**

Use an existing short video and a fresh output folder:

```powershell
python demo.py `
  --video examples\IMG_9732_4s.mp4 `
  --output_pth output\wg_smoke `
  --save_pkl `
  --pose-backend rtmpose `
  --estimate_local_only
```

Expected: `output\wg_smoke\IMG_9732_4s\wham_output.pkl` exists.

- [ ] **Step 4: Inspect saved fields**

Run:

```powershell
@'
from pathlib import Path
import joblib, numpy as np
p = Path("output/wg_smoke/IMG_9732_4s/wham_output.pkl")
data = joblib.load(p)
for tid, rec in data.items():
    print("track", tid)
    for key in ["contact", "feet_world", "feet_refined"]:
        print(key, np.asarray(rec[key]).shape if key in rec else None)
'@ | python -
```

Expected: `contact` and `feet_world` are present with frame count matching `pose`.

- [ ] **Step 5: Commit**

```powershell
git add demo.py
git commit -m "Save WHAM contact and feet signals"
```

---

### Task 2: Preserve Contact/Feet Through Fixed-Beta Canonicalization

**Files:**
- Modify: `scripts/canonicalize_wham_fixed_beta.py`
- Test: `tests/world_grounded/test_canonical_fields.py`

- [ ] **Step 1: Add helper to trim sequence fields**

Add near `beta_variation()`:

```python
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
```

- [ ] **Step 2: Call helper in canonicalize_track**

Inside `canonicalize_track()`, after `out_record["betas"] = ...`:

```python
    trim_sequence_fields(out_record, len(pose))
```

- [ ] **Step 3: Add test for preserving fields**

Create `tests/world_grounded/test_canonical_fields.py`:

```python
import numpy as np

from scripts.canonicalize_wham_fixed_beta import trim_sequence_fields


def test_trim_sequence_fields_preserves_contact_and_feet():
    record = {
        "contact": np.ones((5, 4), dtype=np.float64),
        "feet_world": np.ones((5, 4, 3), dtype=np.float64) * 2.0,
        "unrelated": "keep",
    }

    trim_sequence_fields(record, 3)

    assert record["contact"].shape == (3, 4)
    assert record["feet_world"].shape == (3, 4, 3)
    assert record["contact"].dtype == np.float32
    assert record["feet_world"].dtype == np.float32
    assert record["unrelated"] == "keep"
```

- [ ] **Step 4: Run test**

Run:

```powershell
python -m pytest tests\world_grounded\test_canonical_fields.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```powershell
git add scripts\canonicalize_wham_fixed_beta.py tests\world_grounded\test_canonical_fields.py
git commit -m "Preserve WHAM contact fields during canonicalization"
```

---

### Task 3: Add Track Merge Utilities

**Files:**
- Create: `lib/world_grounded/__init__.py`
- Create: `lib/world_grounded/tracks.py`
- Test: `tests/world_grounded/test_tracks.py`

- [ ] **Step 1: Create package init**

Create `lib/world_grounded/__init__.py`:

```python
"""World-grounded SMPL optimization utilities."""
```

- [ ] **Step 2: Write failing tests**

Create `tests/world_grounded/test_tracks.py`:

```python
import numpy as np

from lib.world_grounded.tracks import merge_tracks, select_track


def test_merge_tracks_preserves_full_frame_range_and_prefers_longer_overlap():
    results = {
        0: {
            "frame_ids": np.arange(0, 4),
            "pose": np.zeros((4, 72), dtype=np.float32),
            "trans_world": np.zeros((4, 3), dtype=np.float32),
        },
        1: {
            "frame_ids": np.arange(3, 7),
            "pose": np.ones((4, 72), dtype=np.float32),
            "trans_world": np.ones((4, 3), dtype=np.float32),
        },
    }

    merged = merge_tracks(results)

    assert merged["frame_ids"].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert merged["pose"].shape == (7, 72)
    assert np.allclose(merged["pose"][3], np.zeros(72))
    assert np.allclose(merged["pose"][4], np.ones(72))


def test_select_track_longest():
    results = {
        "a": {"frame_ids": np.arange(2), "pose": np.zeros((2, 72))},
        "b": {"frame_ids": np.arange(5), "pose": np.zeros((5, 72))},
    }

    track_id, record = select_track(results, "longest")

    assert track_id == "b"
    assert len(record["pose"]) == 5
```

- [ ] **Step 3: Implement tracks.py**

Create `lib/world_grounded/tracks.py`:

```python
from __future__ import annotations

import copy
from typing import Any

import numpy as np


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def track_length(record: dict) -> int:
    if "frame_ids" in record:
        return len(record["frame_ids"])
    if "frame_id" in record:
        return len(record["frame_id"])
    if "pose" in record:
        return len(record["pose"])
    if "pose_world" in record:
        return len(record["pose_world"])
    raise ValueError("Cannot infer track length from record.")


def track_frame_ids(record: dict) -> np.ndarray:
    if "frame_ids" in record:
        return np.asarray(record["frame_ids"], dtype=np.int64)
    if "frame_id" in record:
        return np.asarray(record["frame_id"], dtype=np.int64)
    return np.arange(track_length(record), dtype=np.int64)


def merge_tracks(results: dict) -> dict:
    samples: dict[int, tuple[int, Any, int]] = {}
    for key, record in results.items():
        n = track_length(record)
        for local_idx, frame_id in enumerate(track_frame_ids(record)):
            candidate = (n, key, local_idx)
            current = samples.get(int(frame_id))
            if current is None or candidate[0] > current[0]:
                samples[int(frame_id)] = candidate

    if not samples:
        raise ValueError("No frames found in records.")

    ordered_frames = sorted(samples)
    first_key = samples[ordered_frames[0]][1]
    merged = copy.deepcopy(results[first_key])
    all_keys = set().union(*(record.keys() for record in results.values()))

    for field in all_keys:
        values = []
        can_merge = True
        for frame_id in ordered_frames:
            _, key, local_idx = samples[frame_id]
            record = results[key]
            if field not in record:
                can_merge = False
                break
            value = record[field]
            if not hasattr(value, "__len__") or len(value) != track_length(record):
                can_merge = False
                break
            values.append(to_numpy(value)[local_idx])
        if can_merge:
            merged[field] = np.stack(values, axis=0)

    merged["frame_ids"] = np.asarray(ordered_frames, dtype=np.int64)
    merged["frame_id"] = np.asarray(ordered_frames, dtype=np.int64)
    return merged


def select_track(results: dict, track_id: str | None = "merge"):
    if track_id == "merge":
        return "merged", merge_tracks(results)
    if track_id == "longest":
        key = max(results.keys(), key=lambda k: track_length(results[k]))
        return key, results[key]
    if track_id is None:
        key = sorted(results.keys(), key=lambda x: str(x))[0]
        return key, results[key]
    if track_id in results:
        return track_id, results[track_id]
    for key, value in results.items():
        if str(key) == str(track_id):
            return key, value
    raise KeyError(f"Track id {track_id!r} not found. Available: {list(results.keys())}")
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m pytest tests\world_grounded\test_tracks.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Refactor retarget to use shared track utilities**

In `scripts/retarget_smpl_to_opensim.py`, replace its local `track_length`, `track_frame_ids`, `merge_tracks`, and `select_track` definitions with:

```python
from lib.world_grounded.tracks import select_track
```

Keep the CLI default as `--track-id merge`.

- [ ] **Step 6: Run retarget smoke without IK**

Run:

```powershell
python scripts\retarget_smpl_to_opensim.py output\demo\5月1日-1\canonical_wham_output.pkl `
  --track-id merge `
  --out-dir output\demo\_opensim_retarget_fixed_beta\plan_track_smoke `
  --device cuda `
  --fps 60
```

Expected output contains:

```text
track_id: merged
frames: 216
```

- [ ] **Step 7: Commit**

```powershell
git add lib\world_grounded\__init__.py lib\world_grounded\tracks.py tests\world_grounded\test_tracks.py scripts\retarget_smpl_to_opensim.py
git commit -m "Add shared track merge utilities"
```

---

### Task 4: Add Contact-Aware Ground Estimator

**Files:**
- Create: `lib/world_grounded/ground.py`
- Test: `tests/world_grounded/test_ground.py`

- [ ] **Step 1: Write tests for weighted median and contact-aware ground**

Create `tests/world_grounded/test_ground.py`:

```python
import numpy as np

from lib.world_grounded.ground import (
    compute_contact_confidence,
    estimate_ground,
    weighted_median,
)


def test_weighted_median_ignores_low_weight_outlier():
    values = np.array([0.0, 0.01, 0.02, -2.0], dtype=np.float64)
    weights = np.array([1.0, 1.0, 1.0, 0.001], dtype=np.float64)

    assert abs(weighted_median(values, weights) - 0.01) < 1e-6


def test_estimate_ground_uses_contact_over_raw_minimum():
    foot_points = np.zeros((20, 4, 3), dtype=np.float64)
    foot_points[..., 1] = 0.05
    foot_points[:, 0, 1] = 0.0
    foot_points[5, 3, 1] = -1.0
    contact = np.zeros((20, 4), dtype=np.float64)
    contact[:, 0] = 1.0
    contact[5, 3] = 0.0

    result = estimate_ground(foot_points, fps=60.0, wham_contact=contact)

    assert abs(result.ground_y) < 0.02
    assert result.contact_source == "wham_contact"
    assert result.ground_confidence in {"medium", "high"}


def test_compute_contact_confidence_shape_and_range():
    foot_points = np.zeros((6, 4, 3), dtype=np.float64)
    contact = np.ones((6, 4), dtype=np.float64)

    conf = compute_contact_confidence(foot_points, fps=30.0, ground_y=0.0, wham_contact=contact)

    assert conf.shape == (6, 4)
    assert float(conf.min()) >= 0.0
    assert float(conf.max()) <= 1.0
```

- [ ] **Step 2: Implement ground.py**

Create `lib/world_grounded/ground.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter


@dataclass
class GroundEstimate:
    ground_y: float
    contact_confidence: np.ndarray
    sample_weights: np.ndarray
    contact_source: str
    ground_confidence: str


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float(np.nanmedian(values))
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * np.sum(weights)
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _velocity(foot_points: np.ndarray, fps: float) -> np.ndarray:
    vel = np.zeros_like(foot_points, dtype=np.float64)
    if len(foot_points) > 1:
        vel[1:] = (foot_points[1:] - foot_points[:-1]) * float(fps)
        vel[0] = vel[1]
    return vel


def _score_less(value: np.ndarray, threshold: float, sigma: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((value - threshold) / max(sigma, 1e-6)))


def _score_near_abs(value: np.ndarray, threshold: float, sigma: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((np.abs(value) - threshold) / max(sigma, 1e-6)))


def compute_contact_confidence(
    foot_points: np.ndarray,
    fps: float,
    ground_y: float,
    wham_contact: np.ndarray | None = None,
    height_threshold: float = 0.08,
    horizontal_speed_threshold: float = 0.25,
    vertical_speed_threshold: float = 0.25,
    smooth_size: int = 5,
) -> np.ndarray:
    foot_points = np.asarray(foot_points, dtype=np.float64)
    vel = _velocity(foot_points, fps)
    height = foot_points[..., 1] - float(ground_y)
    horizontal_speed = np.linalg.norm(vel[..., [0, 2]], axis=-1)
    vertical_speed = np.abs(vel[..., 1])

    height_score = _score_near_abs(height, height_threshold, height_threshold * 0.5)
    speed_score = _score_less(horizontal_speed, horizontal_speed_threshold, horizontal_speed_threshold * 0.5)
    vertical_score = _score_less(vertical_speed, vertical_speed_threshold, vertical_speed_threshold * 0.5)
    confidence = height_score * speed_score * vertical_score

    if wham_contact is not None:
        confidence = confidence * np.clip(np.asarray(wham_contact, dtype=np.float64), 0.0, 1.0)

    if smooth_size and smooth_size > 1:
        confidence = median_filter(confidence, size=(smooth_size, 1), mode="nearest")

    return np.clip(confidence, 0.0, 1.0)


def _initial_weights(foot_points: np.ndarray, fps: float, wham_contact: np.ndarray | None) -> tuple[np.ndarray, str]:
    vel = _velocity(foot_points, fps)
    horizontal_speed = np.linalg.norm(vel[..., [0, 2]], axis=-1)
    vertical_speed = np.abs(vel[..., 1])
    speed_weight = _score_less(horizontal_speed, 0.35, 0.15) * _score_less(vertical_speed, 0.35, 0.15)
    if wham_contact is None:
        return speed_weight, "fallback_velocity_height"
    return speed_weight * np.clip(wham_contact, 0.0, 1.0), "wham_contact"


def estimate_ground(
    foot_points: np.ndarray,
    fps: float,
    wham_contact: np.ndarray | None = None,
    min_weighted_samples: int = 8,
) -> GroundEstimate:
    foot_points = np.asarray(foot_points, dtype=np.float64)
    if foot_points.ndim != 3 or foot_points.shape[-1] != 3:
        raise ValueError(f"foot_points must have shape [T, K, 3], got {foot_points.shape}")
    if wham_contact is not None:
        wham_contact = np.asarray(wham_contact, dtype=np.float64)
        if wham_contact.shape != foot_points.shape[:2]:
            raise ValueError(f"wham_contact shape {wham_contact.shape} does not match {foot_points.shape[:2]}")

    weights, source = _initial_weights(foot_points, fps, wham_contact)
    heights = foot_points[..., 1]
    enough = int(np.sum(weights > 0.05)) >= min_weighted_samples
    if enough:
        ground_y = weighted_median(heights, weights)
    else:
        ground_y = float(np.percentile(heights[np.isfinite(heights)], 5))

    contact_conf = compute_contact_confidence(foot_points, fps, ground_y, wham_contact)
    refined_weights = np.maximum(weights, contact_conf)
    if int(np.sum(refined_weights > 0.05)) >= min_weighted_samples:
        ground_y = weighted_median(heights, refined_weights)
        contact_conf = compute_contact_confidence(foot_points, fps, ground_y, wham_contact)
        final_weights = np.maximum(refined_weights, contact_conf)
    else:
        final_weights = refined_weights

    active = int(np.sum(final_weights > 0.05))
    if active >= 30:
        confidence = "high"
    elif active >= min_weighted_samples:
        confidence = "medium"
    else:
        confidence = "low"

    return GroundEstimate(
        ground_y=float(ground_y),
        contact_confidence=contact_conf.astype(np.float32),
        sample_weights=final_weights.astype(np.float32),
        contact_source=source,
        ground_confidence=confidence,
    )
```

- [ ] **Step 3: Run tests**

Run:

```powershell
python -m pytest tests\world_grounded\test_ground.py -v
```

Expected: 3 passed.

- [ ] **Step 4: Commit**

```powershell
git add lib\world_grounded\ground.py tests\world_grounded\test_ground.py
git commit -m "Add contact-aware ground estimator"
```

---

### Task 5: Add Foot Point Provider and Root Translation Optimizer

**Files:**
- Create: `lib/world_grounded/foot_points.py`
- Create: `lib/world_grounded/root_optimizer.py`
- Test: `tests/world_grounded/test_root_optimizer.py`

- [ ] **Step 1: Write root optimizer tests**

Create `tests/world_grounded/test_root_optimizer.py`:

```python
import numpy as np

from lib.world_grounded.root_optimizer import optimize_root_translation


def test_optimize_root_translation_reduces_penetration_and_preserves_shape():
    trans = np.zeros((12, 3), dtype=np.float64)
    foot_points = np.zeros((12, 4, 3), dtype=np.float64)
    foot_points[..., 1] = -0.10
    contact_conf = np.ones((12, 4), dtype=np.float64)

    result = optimize_root_translation(
        trans_world=trans,
        foot_points=foot_points,
        contact_confidence=contact_conf,
        ground_y=0.0,
        fps=60.0,
    )

    corrected_feet = foot_points + result.delta[:, None, :]
    assert result.optimized_trans_world.shape == (12, 3)
    assert corrected_feet[..., 1].min() > -0.01
    assert result.report["foot_penetration_max_cm_after"] < result.report["foot_penetration_max_cm_before"]


def test_optimize_root_translation_smooths_spike():
    trans = np.zeros((15, 3), dtype=np.float64)
    trans[7, 1] = 1.0
    foot_points = np.zeros((15, 4, 3), dtype=np.float64)
    contact_conf = np.zeros((15, 4), dtype=np.float64)

    result = optimize_root_translation(trans, foot_points, contact_conf, ground_y=0.0, fps=30.0)

    assert result.optimized_trans_world[7, 1] < 0.75
```

- [ ] **Step 2: Implement foot_points.py**

Create `lib/world_grounded/foot_points.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FootPointResult:
    points: np.ndarray
    labels: list[str]
    source: str


DEFAULT_FOOT_LABELS = ["left_heel_or_ankle", "left_toe_or_forefoot", "right_heel_or_ankle", "right_toe_or_forefoot"]


def get_record_foot_points(record: dict) -> FootPointResult:
    for key in ["feet_refined", "feet_world", "feet"]:
        if key in record:
            points = np.asarray(record[key], dtype=np.float32)
            if points.ndim == 3 and points.shape[-1] == 3:
                return FootPointResult(points=points, labels=DEFAULT_FOOT_LABELS[: points.shape[1]], source=key)

    if "verts" in record:
        verts = np.asarray(record["verts"], dtype=np.float32)
        if verts.ndim == 3 and verts.shape[-1] == 3:
            # Fallback: use low vertices split by x side. This is deliberately conservative and reported as fallback.
            n = verts.shape[0]
            points = np.zeros((n, 4, 3), dtype=np.float32)
            for t in range(n):
                frame = verts[t]
                y_cut = np.percentile(frame[:, 1], 5)
                low = frame[frame[:, 1] <= y_cut]
                if len(low) < 4:
                    low = frame[np.argsort(frame[:, 1])[: max(4, min(20, len(frame)))]]
                left = low[low[:, 0] <= np.median(low[:, 0])]
                right = low[low[:, 0] > np.median(low[:, 0])]
                for side_idx, side_points in enumerate([left, right]):
                    if len(side_points) == 0:
                        side_points = low
                    heel = side_points[np.argmin(side_points[:, 2])]
                    toe = side_points[np.argmax(side_points[:, 2])]
                    points[t, side_idx * 2] = heel
                    points[t, side_idx * 2 + 1] = toe
            return FootPointResult(points=points, labels=DEFAULT_FOOT_LABELS, source="verts_low_fallback")

    raise ValueError("Record does not contain feet_refined, feet_world, feet, or verts for foot point extraction.")


def get_record_contact(record: dict, n_frames: int, n_points: int) -> np.ndarray | None:
    if "contact" not in record:
        return None
    contact = np.asarray(record["contact"], dtype=np.float32)
    if contact.ndim != 2:
        return None
    contact = contact[:n_frames]
    if contact.shape[1] == n_points:
        return contact
    if contact.shape[1] > n_points:
        return contact[:, :n_points]
    padded = np.zeros((len(contact), n_points), dtype=np.float32)
    padded[:, : contact.shape[1]] = contact
    return padded
```

- [ ] **Step 3: Implement root_optimizer.py**

Create `lib/world_grounded/root_optimizer.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


@dataclass
class RootOptimizationResult:
    optimized_trans_world: np.ndarray
    delta: np.ndarray
    report: dict


def _safe_savgol(values: np.ndarray, window: int = 9, polyorder: int = 2) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 5:
        return values.copy()
    win = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if win <= polyorder:
        return values.copy()
    return savgol_filter(values, window_length=win, polyorder=polyorder, axis=0, mode="interp")


def _penetration_cm(foot_y: np.ndarray, ground_y: float) -> tuple[float, float]:
    penetration = np.maximum(0.0, float(ground_y) - foot_y)
    return float(np.mean(penetration) * 100.0), float(np.max(penetration) * 100.0)


def _root_accel_score(trans: np.ndarray, fps: float) -> float:
    if len(trans) < 3:
        return 0.0
    acc = (trans[2:] - 2.0 * trans[1:-1] + trans[:-2]) * float(fps) ** 2
    return float(np.mean(np.linalg.norm(acc, axis=-1)))


def optimize_root_translation(
    trans_world: np.ndarray,
    foot_points: np.ndarray,
    contact_confidence: np.ndarray,
    ground_y: float,
    fps: float,
    smooth_window: int = 9,
    contact_threshold: float = 0.35,
) -> RootOptimizationResult:
    trans_world = np.asarray(trans_world, dtype=np.float64)
    foot_points = np.asarray(foot_points, dtype=np.float64)
    contact_confidence = np.asarray(contact_confidence, dtype=np.float64)

    if trans_world.ndim != 2 or trans_world.shape[1] != 3:
        raise ValueError(f"trans_world must have shape [T, 3], got {trans_world.shape}")
    if foot_points.shape[:2] != contact_confidence.shape:
        raise ValueError("foot_points and contact_confidence frame/point dimensions must match.")

    smoothed_trans = _safe_savgol(trans_world, smooth_window)
    delta = smoothed_trans - trans_world

    foot_after_smooth = foot_points + delta[:, None, :]
    active = contact_confidence > contact_threshold
    correction_y = np.zeros(len(trans_world), dtype=np.float64)
    for t in range(len(trans_world)):
        mask = active[t]
        if np.any(mask):
            min_contact_y = float(np.min(foot_after_smooth[t, mask, 1]))
            correction_y[t] = max(0.0, float(ground_y) - min_contact_y)
        else:
            min_y = float(np.min(foot_after_smooth[t, :, 1]))
            correction_y[t] = max(0.0, float(ground_y) - min_y) * 0.25

    correction_y = _safe_savgol(correction_y[:, None], smooth_window)[:, 0]
    delta[:, 1] += correction_y
    optimized = trans_world + delta

    before_mean, before_max = _penetration_cm(foot_points[..., 1], ground_y)
    after_mean, after_max = _penetration_cm((foot_points + delta[:, None, :])[..., 1], ground_y)

    report = {
        "root_acceleration_before": _root_accel_score(trans_world, fps),
        "root_acceleration_after": _root_accel_score(optimized, fps),
        "foot_penetration_mean_cm_before": before_mean,
        "foot_penetration_max_cm_before": before_max,
        "foot_penetration_mean_cm_after": after_mean,
        "foot_penetration_max_cm_after": after_max,
        "root_delta_mean_cm": float(np.mean(np.linalg.norm(delta, axis=-1)) * 100.0),
        "root_delta_max_cm": float(np.max(np.linalg.norm(delta, axis=-1)) * 100.0),
    }
    return RootOptimizationResult(optimized_trans_world=optimized.astype(np.float32), delta=delta.astype(np.float32), report=report)
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m pytest tests\world_grounded\test_root_optimizer.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add lib\world_grounded\foot_points.py lib\world_grounded\root_optimizer.py tests\world_grounded\test_root_optimizer.py
git commit -m "Add root translation ground correction"
```

---

### Task 6: Add Optimizer CLI and Reports

**Files:**
- Create: `lib/world_grounded/reports.py`
- Create: `scripts/world_grounded_smpl_optimizer.py`
- Test: optimizer smoke command

- [ ] **Step 1: Implement reports.py**

Create `lib/world_grounded/reports.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


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


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(data), indent=2), encoding="utf-8")


def contact_switch_rate(contact_confidence: np.ndarray, threshold: float = 0.5) -> float:
    active = np.asarray(contact_confidence) > threshold
    if len(active) < 2:
        return 0.0
    return float(np.mean(active[1:] != active[:-1]))


def contact_coverage(contact_confidence: np.ndarray, threshold: float = 0.35) -> float:
    return float(np.mean(np.asarray(contact_confidence) > threshold))
```

- [ ] **Step 2: Implement world_grounded_smpl_optimizer.py**

Create `scripts/world_grounded_smpl_optimizer.py`:

```python
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.world_grounded.foot_points import get_record_contact, get_record_foot_points
from lib.world_grounded.ground import estimate_ground
from lib.world_grounded.reports import contact_coverage, contact_switch_rate, write_json
from lib.world_grounded.root_optimizer import optimize_root_translation
from lib.world_grounded.tracks import select_track


def optimize_record(record: dict, fps: float) -> tuple[dict, dict, dict, dict]:
    if "trans_world" not in record:
        raise ValueError("Record must contain trans_world for root translation optimization.")
    out = copy.deepcopy(record)
    foot = get_record_foot_points(record)
    n = min(len(out["trans_world"]), len(foot.points))
    foot_points = foot.points[:n]
    contact = get_record_contact(record, n, foot_points.shape[1])
    ground = estimate_ground(foot_points, fps=fps, wham_contact=contact)
    root = optimize_root_translation(
        np.asarray(out["trans_world"][:n], dtype=np.float32),
        foot_points,
        ground.contact_confidence,
        ground_y=ground.ground_y,
        fps=fps,
    )

    delta = root.delta
    out["trans_world"] = np.asarray(out["trans_world"], dtype=np.float32).copy()
    out["trans_world"][:n] = root.optimized_trans_world
    if "verts" in out:
        out["verts"] = np.asarray(out["verts"], dtype=np.float32).copy()
        out["verts"][:n] = out["verts"][:n] + delta[:, None, :]
    for key in ["feet_world", "feet_refined", "feet"]:
        if key in out:
            arr = np.asarray(out[key], dtype=np.float32).copy()
            arr[:n] = arr[:n] + delta[:, None, :]
            out[key] = arr

    ground_report = {
        "ground_y": ground.ground_y,
        "ground_confidence": ground.ground_confidence,
        "contact_source": ground.contact_source,
        "foot_source": foot.source,
        "foot_labels": foot.labels,
    }
    contact_report = {
        "contact_coverage": contact_coverage(ground.contact_confidence),
        "contact_switch_rate": contact_switch_rate(ground.contact_confidence),
        "contact_confidence_shape": list(ground.contact_confidence.shape),
    }
    quality_report = {
        **ground_report,
        **contact_report,
        **root.report,
        "num_frames": int(n),
    }
    return out, ground_report, contact_report, quality_report


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize fixed-beta WHAM SMPL for ground/contact consistency.")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--track-id", default="merge")
    parser.add_argument("--run-retarget", action="store_true")
    parser.add_argument("--retarget-config", default="configs/retarget/smpl_to_mimicmsk_opensim.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opensim-cmd", default=r"C:\Users\yangfeiyang\Downloads\OpenSim 4.5\bin\opensim-cmd.exe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_pkl = Path(args.input_pkl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = joblib.load(input_pkl)
    track_id, record = select_track(results, args.track_id)
    optimized_record, ground_report, contact_report, quality_report = optimize_record(record, fps=args.fps)

    optimized = {track_id: optimized_record}
    optimized_pkl = out_dir / "optimized_canonical_wham_output.pkl"
    joblib.dump(optimized, optimized_pkl)
    write_json(out_dir / "ground_plane.json", ground_report)
    write_json(out_dir / "contact_report.json", contact_report)
    write_json(out_dir / "quality_report.json", quality_report)
    write_json(out_dir / "optimization_report.json", {"input_pkl": str(input_pkl), "output_pkl": str(optimized_pkl), "track_id": str(track_id)})

    print(f"track_id: {track_id}")
    print(f"frames: {quality_report['num_frames']}")
    print(f"optimized_pkl: {optimized_pkl}")
    print(f"ground_y: {ground_report['ground_y']:.8f}")
    print(f"ground_confidence: {ground_report['ground_confidence']}")

    if args.run_retarget:
        retarget_dir = out_dir / "retarget"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "retarget_smpl_to_opensim.py"),
            str(optimized_pkl),
            "--config",
            args.retarget_config,
            "--out-dir",
            str(retarget_dir),
            "--fps",
            str(args.fps),
            "--device",
            args.device,
            "--track-id",
            str(track_id),
            "--free-root",
            "--run-ik",
            "--opensim-cmd",
            args.opensim_cmd,
        ]
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run syntax check**

Run:

```powershell
python -m py_compile scripts\world_grounded_smpl_optimizer.py lib\world_grounded\reports.py
```

Expected: exit code 0.

- [ ] **Step 4: Run optimizer smoke without OpenSim**

Run:

```powershell
python scripts\world_grounded_smpl_optimizer.py `
  --input-pkl output\demo\5月1日-1\canonical_wham_output.pkl `
  --out-dir output\demo\world_grounded_smoke `
  --fps 60 `
  --track-id merge
```

Expected files:

```text
output\demo\world_grounded_smoke\optimized_canonical_wham_output.pkl
output\demo\world_grounded_smoke\ground_plane.json
output\demo\world_grounded_smoke\contact_report.json
output\demo\world_grounded_smoke\quality_report.json
```

- [ ] **Step 5: Inspect quality report**

Run:

```powershell
Get-Content output\demo\world_grounded_smoke\quality_report.json
```

Expected: includes `foot_penetration_max_cm_before`, `foot_penetration_max_cm_after`, and `ground_confidence`.

- [ ] **Step 6: Commit**

```powershell
git add lib\world_grounded\reports.py scripts\world_grounded_smpl_optimizer.py
git commit -m "Add world grounded SMPL optimizer CLI"
```

---

### Task 7: Integrate World-Grounded Optimizer Into Video Pipeline

**Files:**
- Modify: `scripts/video_to_fixed_smpl_to_opensim.py`
- Test: pipeline smoke with `--world-grounded --skip-ik`

- [ ] **Step 1: Add CLI arguments**

In `parse_args()` add:

```python
    parser.add_argument("--world-grounded", action="store_true", help="Run SMPL-layer ground/contact optimizer before OpenSim retarget.")
    parser.add_argument("--world-grounded-out-dir", default=None)
```

- [ ] **Step 2: Compute optimized pkl path**

After `retarget_out` is computed in `main()`:

```python
    world_grounded_out = (
        Path(args.world_grounded_out_dir).resolve()
        if args.world_grounded_out_dir
        else output_root / "_world_grounded" / safe_ascii_name(sequence)
    )
```

- [ ] **Step 3: Run optimizer before retarget when enabled**

Before building `retarget_cmd`, insert:

```python
    retarget_input_pkl = canonical_pkl
    if args.world_grounded:
        wg_cmd = [
            sys.executable,
            "scripts/world_grounded_smpl_optimizer.py",
            "--input-pkl",
            str(canonical_pkl),
            "--out-dir",
            str(world_grounded_out),
            "--fps",
            str(args.fps),
            "--track-id",
            str(args.track_id),
            "--device",
            args.device,
        ]
        run(wg_cmd)
        retarget_input_pkl = world_grounded_out / "optimized_canonical_wham_output.pkl"
```

Then change the retarget command to use:

```python
        str(retarget_input_pkl),
```

- [ ] **Step 4: Print world-grounded output in summary**

In the final summary, add:

```python
    if args.world_grounded:
        print(f"world_grounded_dir: {world_grounded_out}")
        print(f"optimized_fixed_beta_pkl: {world_grounded_out / 'optimized_canonical_wham_output.pkl'}")
```

- [ ] **Step 5: Run syntax check**

Run:

```powershell
python -m py_compile scripts\video_to_fixed_smpl_to_opensim.py
```

Expected: exit code 0.

- [ ] **Step 6: Run pipeline smoke without OpenSim IK**

Run:

```powershell
python scripts\video_to_fixed_smpl_to_opensim.py `
  --video examples\5月1日-1.mp4 `
  --output-pth output\demo `
  --device cuda `
  --fps 60 `
  --skip-wham `
  --skip-fixed-beta `
  --world-grounded `
  --skip-ik `
  --retarget-out-dir output\demo\_opensim_retarget_fixed_beta\wg_pipeline_smoke
```

Expected:

```text
optimized_fixed_beta_pkl: ...optimized_canonical_wham_output.pkl
track_id: merged
frames: 216
beta_variation_after_max_abs: 0
```

- [ ] **Step 7: Commit**

```powershell
git add scripts\video_to_fixed_smpl_to_opensim.py
git commit -m "Integrate world grounded optimizer into video pipeline"
```

---

### Task 8: Final Verification and Documentation Update

**Files:**
- Modify: `docs/superpowers/specs/2026-05-09-world-grounded-smpl-optimizer-design.md` only if implementation details changed.
- Test: full test suite subset and short OpenSim smoke.

- [ ] **Step 1: Run all world-grounded tests**

Run:

```powershell
python -m pytest tests\world_grounded -v
```

Expected: all tests pass.

- [ ] **Step 2: Run full optimizer + retarget short smoke**

Run:

```powershell
python scripts\video_to_fixed_smpl_to_opensim.py `
  --video examples\5月1日-1.mp4 `
  --output-pth output\demo `
  --device cuda `
  --fps 60 `
  --skip-wham `
  --skip-fixed-beta `
  --world-grounded `
  --max-frames 30 `
  --retarget-out-dir output\demo\_opensim_retarget_fixed_beta\wg_ik_smoke
```

Expected:

```text
Pipeline complete
beta_variation_after_max_abs: 0
opensim_motion: ...\opensim_ik.mot
```

- [ ] **Step 3: Compare quality report**

Run:

```powershell
Get-Content output\demo\_world_grounded\5_1_-1_3d3c17ec\quality_report.json
```

Expected: `foot_penetration_max_cm_after` is less than or equal to `foot_penetration_max_cm_before`. If not, record the failure in the final answer and do not claim ground correction worked.

- [ ] **Step 4: Confirm fixed beta remains fixed**

Run:

```powershell
@'
from pathlib import Path
import joblib, numpy as np
p = Path("output/demo/_world_grounded/5_1_-1_3d3c17ec/optimized_canonical_wham_output.pkl")
data = joblib.load(p)
for tid, rec in data.items():
    betas = np.asarray(rec["betas"])
    print(tid, float(np.max(np.abs(betas - betas[0:1]))))
'@ | python -
```

Expected: every printed value is `0.0`.

- [ ] **Step 5: Commit final docs if changed**

If spec or README notes changed:

```powershell
git add docs\superpowers\specs\2026-05-09-world-grounded-smpl-optimizer-design.md
git commit -m "Document world grounded optimizer usage"
```

- [ ] **Step 6: Final status**

Report:

- changed files
- test commands and results
- output paths
- whether ground metrics improved
- whether OpenSim warnings remain

