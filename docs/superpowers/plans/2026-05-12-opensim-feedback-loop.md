# Stage6 OpenSim Feedback Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Stage6 v1 so OpenSim IK feedback can weight a second SMPL lower-body optimization pass, then accept or roll back to Stage5 based on auditable metrics.

**Architecture:** Keep OpenSim as a black-box feedback and validation backend. Add focused utilities for IK log parsing, lower-body marker classification, feedback weight construction, and selected-output acceptance. A new `scripts/opensim_feedback_loop.py` orchestrates `iter_00 -> OpenSim -> weights -> iter_01 -> OpenSim -> selection`, and the existing video pipeline calls it behind `--opensim-feedback-loop`.

**Tech Stack:** Python, NumPy, joblib, pytest, existing WHAM SMPL/OpenSim scripts, OpenSim CLI through `opensim-cmd.exe`.

---

## File Structure

- Modify `lib/smpl_optimization/opensim_feedback.py`
  - Owns OpenSim IK log parsing, lower-body marker classification, capped/smoothed feedback weights, and summary statistics.

- Create `lib/smpl_optimization/stage6_selection.py`
  - Owns accept/rollback decision logic for `iter_00` vs `iter_01`.

- Modify `lib/smpl_optimization/lower_body.py`
  - Threads optional `frame_weights` through `optimize_record()` and records frame weight stats in reports.

- Modify `scripts/optimize_smpl_lower_body.py`
  - Adds `--frame-weights` so Stage6 can reuse the Stage5 optimizer without duplicating logic.

- Modify `scripts/retarget_smpl_to_opensim.py`
  - Adds optional OpenSim log capture while keeping existing console output.

- Create `scripts/opensim_feedback_loop.py`
  - Orchestrates Stage6 v1 and writes `selected_corrected_smpl.pkl`.

- Modify `scripts/video_to_fixed_smpl_to_opensim.py`
  - Adds `--opensim-feedback-loop` and calls the Stage6 script after `--optimize-lower-body`.

- Create tests:
  - `tests/smpl_optimization/test_opensim_feedback.py`
  - `tests/smpl_optimization/test_stage6_selection.py`
  - `tests/smpl_optimization/test_opensim_feedback_loop_cli.py`

---

### Task 1: Strengthen OpenSim Feedback Utilities

**Files:**
- Modify: `lib/smpl_optimization/opensim_feedback.py`
- Test: `tests/smpl_optimization/test_opensim_feedback.py`

- [ ] **Step 1: Write failing tests for parser, marker filtering, and feedback weights**

Create `tests/smpl_optimization/test_opensim_feedback.py`:

```python
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.opensim_feedback import (  # noqa: E402
    build_feedback_weights,
    classify_marker,
    is_lower_body_marker,
    parse_ik_log_metrics,
)


def test_parse_ik_log_metrics_extracts_frames(tmp_path):
    log = tmp_path / "ik.log"
    log.write_text(
        "\n".join(
            [
                "[info] Frame 0 (t = 0.0): marker error: RMS = 0.040, max = 0.100 (RightShoulder)",
                "[info] Frame 1 (t = 0.1): marker error: RMS = 0.120, max = 0.220 (ankle_l)",
                "[info] Frame 2 (t = 0.2): marker error: RMS = 0.090, max = 0.180 (RTOE)",
            ]
        ),
        encoding="utf-8",
    )

    metrics = parse_ik_log_metrics(log)

    assert metrics["num_frames"] == 3
    assert metrics["mean_rms"] == 0.08333333
    assert metrics["max_rms"] == 0.12
    assert metrics["max_marker_name"] == "ankle_l"
    assert metrics["frames"][1]["max_marker"] == "ankle_l"


def test_marker_classification_is_case_insensitive_for_lower_body():
    assert is_lower_body_marker("ankle_l")
    assert is_lower_body_marker("RTOE")
    assert is_lower_body_marker("knee_r")
    assert not is_lower_body_marker("RightShoulder")
    assert not is_lower_body_marker("head_retarget")
    assert classify_marker("RTOE") == "lower_body"
    assert classify_marker("RightShoulder") == "non_lower_body"


def test_build_feedback_weights_ignores_non_lower_body_marker_spikes():
    metrics = {
        "frames": [
            {"frame": 0, "rms": 0.20, "max": 0.40, "max_marker": "RightShoulder"},
            {"frame": 1, "rms": 0.20, "max": 0.40, "max_marker": "ankle_l"},
            {"frame": 2, "rms": 0.04, "max": 0.07, "max_marker": "ankle_l"},
        ]
    }

    result = build_feedback_weights(metrics, num_frames=4, rms_threshold=0.08, smooth_radius=0)

    np.testing.assert_allclose(result["weights"], np.asarray([1.0, 2.5, 1.0, 1.0], dtype=np.float32))
    assert result["num_weighted_frames"] == 1
    assert result["ignored_non_lower_body_frames"] == [0]


def test_build_feedback_weights_smooths_lower_body_windows():
    metrics = {
        "frames": [
            {"frame": 2, "rms": 0.20, "max": 0.40, "max_marker": "ankle_l"},
        ]
    }

    result = build_feedback_weights(metrics, num_frames=5, rms_threshold=0.08, smooth_radius=1)

    weights = result["weights"]
    assert weights[2] == 2.5
    assert weights[1] > 1.0
    assert weights[3] > 1.0
    assert weights[0] == 1.0
    assert weights[4] == 1.0
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_opensim_feedback.py -q
```

Expected: FAIL because `classify_marker()` and the new dictionary-returning `build_feedback_weights()` behavior do not exist yet.

- [ ] **Step 3: Implement lower-body marker classification and capped/smoothed weights**

Replace `lib/smpl_optimization/opensim_feedback.py` with:

```python
from pathlib import Path
import re

import numpy as np


_IK_LOG_PATTERN = re.compile(
    r"Frame\s+(?P<frame>\d+).*?"
    r"RMS\s*=\s*(?P<rms>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?).*?"
    r"max\s*=\s*(?P<max>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?).*?"
    r"\((?P<marker>[^)]*)\)"
)

LOWER_BODY_MARKERS = {
    "hip_l",
    "hip_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
    "ltoe",
    "rtoe",
    "toe_l",
    "toe_r",
    "l_foot_touch",
    "r_foot_touch",
}

NON_LOWER_BODY_HINTS = (
    "head",
    "shoulder",
    "elbow",
    "wrist",
    "chest",
    "pelvis",
)


def _neutral_metrics():
    return {
        "num_frames": 0,
        "mean_rms": 0.0,
        "max_rms": 0.0,
        "max_marker_name": "",
        "frames": [],
    }


def _clean_float(value):
    return float(round(float(value), 8))


def _normalize_marker_name(name: str) -> str:
    return str(name or "").strip().lower()


def is_lower_body_marker(name: str) -> bool:
    normalized = _normalize_marker_name(name)
    return normalized in LOWER_BODY_MARKERS


def classify_marker(name: str) -> str:
    normalized = _normalize_marker_name(name)
    if normalized in LOWER_BODY_MARKERS:
        return "lower_body"
    if any(hint in normalized for hint in NON_LOWER_BODY_HINTS):
        return "non_lower_body"
    return "unknown"


def parse_ik_log_metrics(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    frames = []

    for line in text.splitlines():
        match = _IK_LOG_PATTERN.search(line)
        if match is None:
            continue

        frames.append(
            {
                "frame": int(match.group("frame")),
                "rms": float(match.group("rms")),
                "max": float(match.group("max")),
                "max_marker": match.group("marker").strip(),
            }
        )

    if not frames:
        return _neutral_metrics()

    rms_values = np.asarray([frame["rms"] for frame in frames], dtype=np.float64)
    max_frame = max(frames, key=lambda frame: frame["max"])
    return {
        "num_frames": len(frames),
        "mean_rms": _clean_float(np.mean(rms_values)),
        "max_rms": _clean_float(np.max(rms_values)),
        "max_marker_name": max_frame["max_marker"],
        "frames": frames,
    }


def _smooth_weights(weights: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0 or weights.size == 0:
        return weights

    smoothed = weights.copy()
    elevated = np.where(weights > 1.0)[0]
    for frame in elevated:
        for offset in range(1, radius + 1):
            decay = 1.0 / float(offset + 1)
            value = 1.0 + (weights[frame] - 1.0) * decay
            left = frame - offset
            right = frame + offset
            if left >= 0:
                smoothed[left] = max(smoothed[left], value)
            if right < len(smoothed):
                smoothed[right] = max(smoothed[right], value)
    return smoothed


def build_feedback_weights(
    metrics: dict,
    num_frames: int,
    rms_threshold=0.08,
    min_weight=1.0,
    max_weight=2.5,
    smooth_radius=1,
) -> dict:
    n = max(0, int(num_frames))
    weights = np.ones(n, dtype=np.float32) * float(min_weight)
    ignored_non_lower_body_frames = []
    ignored_unknown_marker_frames = []

    for frame_metrics in metrics.get("frames", []):
        frame = int(frame_metrics.get("frame", -1))
        if frame < 0 or frame >= n:
            continue
        rms = float(frame_metrics.get("rms", 0.0))
        if rms <= float(rms_threshold):
            continue

        marker = str(frame_metrics.get("max_marker", ""))
        marker_class = classify_marker(marker)
        if marker_class == "non_lower_body":
            ignored_non_lower_body_frames.append(frame)
            continue
        if marker_class == "unknown":
            ignored_unknown_marker_frames.append(frame)
            continue

        scale = rms / max(float(rms_threshold), 1e-8)
        weights[frame] = min(float(max_weight), max(float(min_weight), float(min_weight) + (scale - 1.0)))

    weights = np.clip(_smooth_weights(weights, smooth_radius), float(min_weight), float(max_weight)).astype(np.float32)
    return {
        "weights": weights,
        "num_frames": n,
        "num_weighted_frames": int(np.count_nonzero(weights > float(min_weight))),
        "min_weight": _clean_float(np.min(weights)) if weights.size else float(min_weight),
        "max_weight": _clean_float(np.max(weights)) if weights.size else float(min_weight),
        "mean_weight": _clean_float(np.mean(weights)) if weights.size else float(min_weight),
        "ignored_non_lower_body_frames": ignored_non_lower_body_frames,
        "ignored_unknown_marker_frames": ignored_unknown_marker_frames,
    }
```

- [ ] **Step 4: Run tests for feedback utilities**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_opensim_feedback.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add lib\smpl_optimization\opensim_feedback.py tests\smpl_optimization\test_opensim_feedback.py
git commit -m "feat: add opensim feedback weights"
```

---

### Task 2: Add Stage6 Selection Logic

**Files:**
- Create: `lib/smpl_optimization/stage6_selection.py`
- Test: `tests/smpl_optimization/test_stage6_selection.py`

- [ ] **Step 1: Write failing selection tests**

Create `tests/smpl_optimization/test_stage6_selection.py`:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.stage6_selection import select_stage6_result  # noqa: E402


def _summary(
    *,
    mean_rms=0.05,
    max_rms=0.12,
    lower_body_mean_rms=0.05,
    penetration=0.002,
    sliding=0.1,
    root_jitter=0.2,
    beta_var=0.0,
    frames=100,
    upper_delta=0.0,
    lower_delta=0.3,
    root_shift=0.1,
):
    return {
        "opensim": {
            "mean_rms": mean_rms,
            "max_rms": max_rms,
            "lower_body_mean_rms": lower_body_mean_rms,
        },
        "smpl": {
            "beta_variation_after_max_abs": beta_var,
            "frames": frames,
            "penetration": {"max_penetration": penetration},
            "sliding": {"mean_contact_speed": sliding},
            "root": {"rms_vertical_accel": root_jitter},
            "pose_delta": {
                "upper_body_max_abs": upper_delta,
                "lower_body_max_abs": lower_delta,
            },
            "total_root_y_shift_max_abs": root_shift,
        },
    }


def test_selects_iter_01_when_hard_checks_pass_and_target_improves():
    baseline = _summary(lower_body_mean_rms=0.08, penetration=0.003)
    candidate = _summary(lower_body_mean_rms=0.06, penetration=0.002)

    result = select_stage6_result(baseline, candidate, root_y_budget=0.25)

    assert result["selected"] == "iter_01"
    assert result["checks"]["at_least_one_target_metric_improved"]


def test_rolls_back_when_opensim_mean_rms_worsens():
    baseline = _summary(mean_rms=0.05)
    candidate = _summary(mean_rms=0.07, lower_body_mean_rms=0.04)

    result = select_stage6_result(baseline, candidate, root_y_budget=0.25)

    assert result["selected"] == "iter_00"
    assert not result["checks"]["opensim_mean_rms_not_worse"]


def test_rolls_back_when_no_metric_improves():
    baseline = _summary()
    candidate = _summary()

    result = select_stage6_result(baseline, candidate, root_y_budget=0.25)

    assert result["selected"] == "iter_00"
    assert not result["checks"]["at_least_one_target_metric_improved"]


def test_rolls_back_when_root_shift_exceeds_budget():
    baseline = _summary()
    candidate = _summary(lower_body_mean_rms=0.04, root_shift=0.4)

    result = select_stage6_result(baseline, candidate, root_y_budget=0.25)

    assert result["selected"] == "iter_00"
    assert not result["checks"]["root_y_total_shift_within_budget"]
```

- [ ] **Step 2: Run selection tests and verify they fail**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_stage6_selection.py -q
```

Expected: FAIL because `lib.smpl_optimization.stage6_selection` does not exist.

- [ ] **Step 3: Implement selection logic**

Create `lib/smpl_optimization/stage6_selection.py`:

```python
from __future__ import annotations

from typing import Any


def _number(data: dict, *keys: str, default: float = 0.0) -> float:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return float(default)
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _not_worse(baseline: dict, candidate: dict, *keys: str, tolerance: float = 1e-8) -> bool:
    return _number(candidate, *keys) <= _number(baseline, *keys) + float(tolerance)


def _improved(baseline: dict, candidate: dict, *keys: str, tolerance: float = 1e-8) -> bool:
    return _number(candidate, *keys) < _number(baseline, *keys) - float(tolerance)


def select_stage6_result(
    baseline: dict,
    candidate: dict,
    *,
    root_y_budget: float,
    upper_body_delta_limit: float = 0.05,
    lower_body_delta_limit: float = 1.2,
) -> dict:
    checks = {
        "beta_variation_after_max_abs_zero": _number(candidate, "smpl", "beta_variation_after_max_abs") == 0.0,
        "frame_count_unchanged": int(_number(candidate, "smpl", "frames")) == int(_number(baseline, "smpl", "frames")),
        "opensim_mean_rms_not_worse": _not_worse(baseline, candidate, "opensim", "mean_rms"),
        "opensim_max_rms_not_worse": _not_worse(baseline, candidate, "opensim", "max_rms"),
        "smpl_max_foot_penetration_not_worse": _not_worse(
            baseline, candidate, "smpl", "penetration", "max_penetration"
        ),
        "smpl_contact_foot_sliding_not_worse": _not_worse(
            baseline, candidate, "smpl", "sliding", "mean_contact_speed"
        ),
        "root_vertical_jitter_not_worse": _not_worse(
            baseline, candidate, "smpl", "root", "rms_vertical_accel"
        ),
        "upper_body_pose_delta_small": _number(candidate, "smpl", "pose_delta", "upper_body_max_abs")
        <= float(upper_body_delta_limit),
        "lower_body_pose_delta_bounded": _number(candidate, "smpl", "pose_delta", "lower_body_max_abs")
        <= float(lower_body_delta_limit),
        "root_y_total_shift_within_budget": _number(candidate, "smpl", "total_root_y_shift_max_abs")
        <= float(root_y_budget),
    }
    target_improvements = {
        "opensim_lower_body_marker_rms_reduced": _improved(
            baseline, candidate, "opensim", "lower_body_mean_rms"
        ),
        "smpl_foot_penetration_reduced": _improved(
            baseline, candidate, "smpl", "penetration", "max_penetration"
        ),
        "smpl_contact_foot_sliding_reduced": _improved(
            baseline, candidate, "smpl", "sliding", "mean_contact_speed"
        ),
        "root_vertical_jitter_reduced": _improved(
            baseline, candidate, "smpl", "root", "rms_vertical_accel"
        ),
    }
    checks["at_least_one_target_metric_improved"] = any(target_improvements.values())
    selected = "iter_01" if all(checks.values()) else "iter_00"
    return {
        "selected": selected,
        "accepted": selected == "iter_01",
        "checks": checks,
        "target_improvements": target_improvements,
        "baseline": baseline,
        "candidate": candidate,
    }
```

- [ ] **Step 4: Run selection tests**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_stage6_selection.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add lib\smpl_optimization\stage6_selection.py tests\smpl_optimization\test_stage6_selection.py
git commit -m "feat: add stage6 selection checks"
```

---

### Task 3: Thread Frame Weights Through Lower-Body Optimization

**Files:**
- Modify: `lib/smpl_optimization/lower_body.py`
- Modify: `scripts/optimize_smpl_lower_body.py`
- Test: `tests/smpl_optimization/test_lower_body_frame_weights.py`

- [ ] **Step 1: Write failing tests for frame weight fitting and CLI loading**

Create `tests/smpl_optimization/test_lower_body_frame_weights.py`:

```python
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.lower_body import LowerBodyOptimizerConfig  # noqa: E402
from scripts.optimize_smpl_lower_body import load_frame_weights  # noqa: E402


def test_config_accepts_frame_weights():
    weights = np.asarray([1.0, 2.0, 1.5], dtype=np.float32)
    config = LowerBodyOptimizerConfig(fps=60.0, frame_weights=weights)

    np.testing.assert_allclose(config.frame_weights, weights)


def test_load_frame_weights_reads_numpy_file(tmp_path):
    path = tmp_path / "weights.npy"
    np.save(path, np.asarray([1.0, 2.5], dtype=np.float32))

    weights = load_frame_weights(path)

    np.testing.assert_allclose(weights, np.asarray([1.0, 2.5], dtype=np.float32))
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_lower_body_frame_weights.py -q
```

Expected: FAIL because `LowerBodyOptimizerConfig.frame_weights` and `load_frame_weights()` do not exist.

- [ ] **Step 3: Add `frame_weights` to the config and optimizer call**

Modify `LowerBodyOptimizerConfig` in `lib/smpl_optimization/lower_body.py`:

```python
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
    frame_weights: Optional[np.ndarray] = None
    w_contact_ground: float = 20.0
    w_penetration: float = 50.0
    w_foot_lock: float = 5.0
    w_pose_smooth: float = 2.0
    w_root_smooth: float = 5.0
    w_wham_prior: float = 2.0
    w_root_prior: float = 20.0
```

Modify the pose pass call in `optimize_record()`:

```python
if config.enable_pose_pass:
    out_record, pose_report = optimize_lower_body_pose_smpl(
        out_record,
        config,
        frame_weights=config.frame_weights,
    )
```

Add frame weight stats to `lower_body_optimization_report`:

```python
"frame_weights": _frame_weight_report(config.frame_weights, _frame_count(record)),
```

Add this helper near `_fit_frame_weights()`:

```python
def _frame_weight_report(frame_weights: Optional[np.ndarray], n_frames: int) -> dict:
    fitted = _fit_frame_weights(frame_weights, n_frames=n_frames) if frame_weights is not None else np.ones((n_frames,), dtype=np.float32)
    return {
        "used": frame_weights is not None,
        "num_frames": int(n_frames),
        "min": float(round(float(np.min(fitted)), 8)) if fitted.size else 1.0,
        "max": float(round(float(np.max(fitted)), 8)) if fitted.size else 1.0,
        "mean": float(round(float(np.mean(fitted)), 8)) if fitted.size else 1.0,
        "num_weighted_frames": int(np.count_nonzero(fitted > 1.0)),
    }
```

- [ ] **Step 4: Add `--frame-weights` to the CLI**

Modify `scripts/optimize_smpl_lower_body.py`:

```python
import numpy as np
```

Add before `parse_args()`:

```python
def load_frame_weights(path: Path | None):
    if path is None:
        return None
    weights = np.load(Path(path))
    return np.asarray(weights, dtype=np.float32).reshape(-1)
```

Add parser argument:

```python
parser.add_argument("--frame-weights", default=None, help="Optional .npy frame weights for pose pass.")
```

Load weights in `main()`:

```python
frame_weights = load_frame_weights(Path(args.frame_weights) if args.frame_weights else None)
```

Pass into config:

```python
frame_weights=frame_weights,
```

- [ ] **Step 5: Run frame weight tests and existing lower-body tests**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_lower_body_frame_weights.py tests\smpl_optimization\test_opensim_feedback.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add lib\smpl_optimization\lower_body.py scripts\optimize_smpl_lower_body.py tests\smpl_optimization\test_lower_body_frame_weights.py
git commit -m "feat: support lower-body frame weights"
```

---

### Task 4: Capture OpenSim IK Logs in Retarget Script

**Files:**
- Modify: `scripts/retarget_smpl_to_opensim.py`
- Test: `tests/smpl_optimization/test_retarget_log_capture.py`

- [ ] **Step 1: Write failing tests for log capture helper**

Create `tests/smpl_optimization/test_retarget_log_capture.py`:

```python
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.retarget_smpl_to_opensim import write_process_log  # noqa: E402


def test_write_process_log_combines_stdout_and_stderr(tmp_path):
    result = subprocess.CompletedProcess(
        args=["opensim-cmd", "run-tool"],
        returncode=0,
        stdout="Frame 0 RMS = 0.1\n",
        stderr="warning line\n",
    )
    path = tmp_path / "opensim_ik.log"

    write_process_log(path, result)

    text = path.read_text(encoding="utf-8")
    assert "Frame 0 RMS = 0.1" in text
    assert "warning line" in text
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_retarget_log_capture.py -q
```

Expected: FAIL because `write_process_log()` does not exist.

- [ ] **Step 3: Implement log capture helper and optional log path**

Add to `scripts/retarget_smpl_to_opensim.py` near `run_opensim()`:

```python
def write_process_log(path: Path, result: subprocess.CompletedProcess) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if result.stdout:
        lines.append(str(result.stdout))
    if result.stderr:
        lines.append(str(result.stderr))
    path.write_text("\n".join(lines), encoding="utf-8")
```

Change `run_opensim()` signature and body:

```python
def run_opensim(opensim_cmd: str, setup_xml: Path, log_path: Path | None = None) -> None:
    if not is_ascii_path(setup_xml):
        raise ValueError(
            "OpenSim 4.5 on Windows cannot reliably read non-ASCII setup/model paths. "
            f"Use an ASCII --out-dir, current setup path is: {setup_xml}"
        )
    cmd = [opensim_cmd, "run-tool", str(setup_xml)]
    if log_path is None:
        subprocess.run(cmd, cwd=str(setup_xml.parent), check=True)
        return

    result = subprocess.run(
        cmd,
        cwd=str(setup_xml.parent),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    write_process_log(log_path, result)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
```

Add parser argument:

```python
parser.add_argument("--opensim-log", default=None, help="Optional path to capture OpenSim IK stdout/stderr.")
```

After `setup_xml` is defined in `main()`:

```python
opensim_log = Path(args.opensim_log).resolve() if args.opensim_log else None
```

Use `opensim_log` only for the final IK call:

```python
run_opensim(opensim_cmd, setup_xml, opensim_log)
```

Keep calibration calls without log capture:

```python
run_opensim(opensim_cmd, calibration_setup)
```

- [ ] **Step 4: Run log capture test**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_retarget_log_capture.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts\retarget_smpl_to_opensim.py tests\smpl_optimization\test_retarget_log_capture.py
git commit -m "feat: capture opensim ik logs"
```

---

### Task 5: Implement Stage6 Orchestration Script

**Files:**
- Create: `scripts/opensim_feedback_loop.py`
- Test: `tests/smpl_optimization/test_opensim_feedback_loop_cli.py`

- [ ] **Step 1: Write failing tests for command builders and selected path**

Create `tests/smpl_optimization/test_opensim_feedback_loop_cli.py`:

```python
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.opensim_feedback_loop import (  # noqa: E402
    build_lower_body_cmd,
    build_retarget_cmd,
    iter_paths,
)


def _args():
    return argparse.Namespace(
        fps=60.0,
        track_id="merge",
        max_root_y_shift=0.25,
        device="cuda",
        pose_iterations=80,
        retarget_config="configs/retarget/smpl_to_mimicmsk_opensim.yaml",
        chunk_size=256,
        alignment="anatomical",
        axis_conversion="mujoco_to_opensim",
        root_calibration="constant_from_free_ik",
        root_calib_frames=10,
        opensim_cmd="C:/OpenSim/bin/opensim-cmd.exe",
        free_root=True,
        ik_accuracy=1e-4,
    )


def test_iter_paths_are_named_for_stage6(tmp_path):
    paths = iter_paths(tmp_path)

    assert paths["iter_00_smpl"].name == "iter_00_stage5_smpl.pkl"
    assert paths["iter_01_smpl"].name == "iter_01_opensim_feedback_smpl.pkl"
    assert paths["selected_smpl"].name == "selected_corrected_smpl.pkl"


def test_lower_body_cmd_uses_frame_weights_when_given(tmp_path):
    args = _args()
    cmd = build_lower_body_cmd(args, tmp_path / "input.pkl", tmp_path / "out", tmp_path / "weights.npy")

    assert "--enable-pose-pass" in cmd
    assert "--frame-weights" in cmd
    assert cmd[cmd.index("--frame-weights") + 1].endswith("weights.npy")


def test_retarget_cmd_requests_log_and_free_root(tmp_path):
    args = _args()
    cmd = build_retarget_cmd(args, tmp_path / "input.pkl", tmp_path / "retarget", tmp_path / "ik.log")

    assert "--run-ik" in cmd
    assert "--free-root" in cmd
    assert "--opensim-log" in cmd
    assert cmd[cmd.index("--opensim-log") + 1].endswith("ik.log")
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_opensim_feedback_loop_cli.py -q
```

Expected: FAIL because `scripts/opensim_feedback_loop.py` does not exist.

- [ ] **Step 3: Create the Stage6 script**

Create `scripts/opensim_feedback_loop.py`:

```python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.lower_body import LowerBodyOptimizerConfig, optimize_record
from lib.smpl_optimization.metrics import beta_variation_max_abs
from lib.smpl_optimization.opensim_feedback import build_feedback_weights, parse_ik_log_metrics
from lib.smpl_optimization.reports import to_jsonable, write_json
from lib.smpl_optimization.stage6_selection import select_stage6_result
from lib.world_grounded.tracks import select_track


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print()
    print("$ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def iter_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "iter_00_dir": out_dir / "iter_00_stage5",
        "iter_00_smpl": out_dir / "iter_00_stage5_smpl.pkl",
        "iter_00_opensim": out_dir / "iter_00_opensim",
        "iter_00_log": out_dir / "iter_00_opensim" / "opensim_ik.log",
        "weights": out_dir / "opensim_feedback_weights.npy",
        "feedback_report": out_dir / "opensim_feedback_report.json",
        "iter_01_dir": out_dir / "iter_01_opensim_feedback",
        "iter_01_smpl": out_dir / "iter_01_opensim_feedback_smpl.pkl",
        "iter_01_opensim": out_dir / "iter_01_opensim",
        "iter_01_log": out_dir / "iter_01_opensim" / "opensim_ik.log",
        "selection_report": out_dir / "stage6_selection_report.json",
        "selected_smpl": out_dir / "selected_corrected_smpl.pkl",
    }


def build_lower_body_cmd(
    args: argparse.Namespace,
    input_pkl: Path,
    out_dir: Path,
    frame_weights: Path | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/optimize_smpl_lower_body.py",
        "--input-pkl",
        str(input_pkl),
        "--out-dir",
        str(out_dir),
        "--fps",
        str(args.fps),
        "--track-id",
        str(args.track_id),
        "--max-root-y-shift",
        str(args.max_root_y_shift),
        "--device",
        args.device,
        "--enable-pose-pass",
        "--pose-iterations",
        str(args.pose_iterations),
    ]
    if frame_weights is not None:
        cmd += ["--frame-weights", str(frame_weights)]
    return cmd


def build_retarget_cmd(args: argparse.Namespace, input_pkl: Path, out_dir: Path, log_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/retarget_smpl_to_opensim.py",
        str(input_pkl),
        "--config",
        args.retarget_config,
        "--out-dir",
        str(out_dir),
        "--fps",
        str(args.fps),
        "--device",
        args.device,
        "--chunk-size",
        str(args.chunk_size),
        "--alignment",
        args.alignment,
        "--axis-conversion",
        args.axis_conversion,
        "--root-calibration",
        args.root_calibration,
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
        cmd += ["--opensim-cmd", args.opensim_cmd]
    if args.free_root:
        cmd.append("--free-root")
    return cmd


def load_selected_record(pkl: Path, track_id: str) -> tuple[str, dict]:
    data = joblib.load(pkl)
    selected_track_id, record = select_track(data, track_id)
    return str(selected_track_id), record


def summarize_smpl_report(pkl: Path, report_dir: Path, track_id: str) -> dict:
    _, record = load_selected_record(pkl, track_id)
    validation_path = report_dir / "validation_summary.json"
    lower_report_path = report_dir / "lower_body_optimization_report.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    lower_report = json.loads(lower_report_path.read_text(encoding="utf-8")) if lower_report_path.exists() else {}
    after = validation.get("after", lower_report.get("after", {}))
    pose_delta = validation.get("pose_delta", {})
    return {
        "beta_variation_after_max_abs": beta_variation_max_abs(record.get("betas", np.asarray([]))),
        "frames": int(np.asarray(record.get("trans_world", record.get("pose", []))).shape[0]),
        "penetration": after.get("penetration", {}),
        "sliding": after.get("sliding", {}),
        "root": after.get("root", {}),
        "pose_delta": pose_delta,
        "total_root_y_shift_max_abs": lower_report.get("total_root_y_shift_max_abs", 0.0),
    }


def lower_body_mean_rms(metrics: dict) -> float:
    values = [
        float(frame["rms"])
        for frame in metrics.get("frames", [])
        if str(frame.get("max_marker", "")).strip()
    ]
    return float(round(float(np.mean(values)), 8)) if values else 0.0


def build_summary(smpl_pkl: Path, report_dir: Path, ik_log: Path, track_id: str) -> dict:
    opensim = parse_ik_log_metrics(ik_log)
    opensim["lower_body_mean_rms"] = lower_body_mean_rms(opensim)
    return {
        "opensim": opensim,
        "smpl": summarize_smpl_report(smpl_pkl, report_dir, track_id),
    }


def copy_selected(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage6 OpenSim feedback loop.")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--track-id", default="merge")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-root-y-shift", type=float, default=0.25)
    parser.add_argument("--pose-iterations", type=int, default=80)
    parser.add_argument("--retarget-config", default="configs/retarget/smpl_to_mimicmsk_opensim.yaml")
    parser.add_argument("--alignment", choices=["mapping", "anatomical"], default="anatomical")
    parser.add_argument("--axis-conversion", choices=["mujoco_to_opensim", "opensim_to_mujoco", "identity"], default="mujoco_to_opensim")
    parser.add_argument("--root-calibration", choices=["none", "constant_from_free_ik"], default="constant_from_free_ik")
    parser.add_argument("--root-calib-frames", type=int, default=10)
    parser.add_argument("--opensim-cmd", default=None)
    parser.add_argument("--ik-accuracy", type=float, default=1e-4)
    parser.add_argument("--free-root", action="store_true", default=True)
    parser.add_argument("--rms-threshold", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_pkl = Path(args.input_pkl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = iter_paths(out_dir)

    run(build_lower_body_cmd(args, input_pkl, paths["iter_00_dir"], None))
    copy_selected(paths["iter_00_dir"] / "corrected_smpl.pkl", paths["iter_00_smpl"])
    run(build_retarget_cmd(args, paths["iter_00_smpl"], paths["iter_00_opensim"], paths["iter_00_log"]))

    _, iter_00_record = load_selected_record(paths["iter_00_smpl"], args.track_id)
    n_frames = int(np.asarray(iter_00_record.get("trans_world", iter_00_record.get("pose", []))).shape[0])
    iter_00_ik = parse_ik_log_metrics(paths["iter_00_log"])
    feedback = build_feedback_weights(iter_00_ik, n_frames, rms_threshold=args.rms_threshold)
    np.save(paths["weights"], feedback["weights"])
    write_json(paths["feedback_report"], {k: v for k, v in feedback.items() if k != "weights"})

    run(build_lower_body_cmd(args, input_pkl, paths["iter_01_dir"], paths["weights"]))
    copy_selected(paths["iter_01_dir"] / "corrected_smpl.pkl", paths["iter_01_smpl"])
    run(build_retarget_cmd(args, paths["iter_01_smpl"], paths["iter_01_opensim"], paths["iter_01_log"]))

    baseline = build_summary(paths["iter_00_smpl"], paths["iter_00_dir"], paths["iter_00_log"], args.track_id)
    candidate = build_summary(paths["iter_01_smpl"], paths["iter_01_dir"], paths["iter_01_log"], args.track_id)
    selection = select_stage6_result(baseline, candidate, root_y_budget=args.max_root_y_shift)
    selected_source = paths["iter_01_smpl"] if selection["selected"] == "iter_01" else paths["iter_00_smpl"]
    copy_selected(selected_source, paths["selected_smpl"])
    write_json(paths["selection_report"], selection)

    print(f"selected: {selection['selected']}")
    print(f"selected_corrected_smpl: {paths['selected_smpl']}")
    print(f"stage6_selection_report: {paths['selection_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_opensim_feedback_loop_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts\opensim_feedback_loop.py tests\smpl_optimization\test_opensim_feedback_loop_cli.py
git commit -m "feat: add opensim feedback loop script"
```

---

### Task 6: Add Top-Level Pipeline Flag

**Files:**
- Modify: `scripts/video_to_fixed_smpl_to_opensim.py`
- Test: `tests/smpl_optimization/test_video_pipeline_stage6.py`

- [ ] **Step 1: Write failing tests for Stage6 command builder**

Create `tests/smpl_optimization/test_video_pipeline_stage6.py`:

```python
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.video_to_fixed_smpl_to_opensim import build_opensim_feedback_cmd  # noqa: E402


def test_build_opensim_feedback_cmd_contains_stage6_flags(tmp_path):
    args = argparse.Namespace(
        fps=60.0,
        track_id="merge",
        device="cuda",
        chunk_size=256,
        lower_body_max_root_y_shift=0.25,
        lower_body_pose_iterations=80,
        retarget_config="configs/retarget/smpl_to_mimicmsk_opensim.yaml",
        alignment="anatomical",
        axis_conversion="mujoco_to_opensim",
        root_calibration="constant_from_free_ik",
        root_calib_frames=10,
        opensim_cmd="C:/OpenSim/bin/opensim-cmd.exe",
        free_root=True,
    )

    cmd = build_opensim_feedback_cmd(args, tmp_path / "world.pkl", tmp_path / "stage6")

    assert "scripts/opensim_feedback_loop.py" in cmd
    assert "--input-pkl" in cmd
    assert cmd[cmd.index("--input-pkl") + 1].endswith("world.pkl")
    assert "--free-root" in cmd
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_video_pipeline_stage6.py -q
```

Expected: FAIL because `build_opensim_feedback_cmd()` does not exist.

- [ ] **Step 3: Add command builder and parser flag**

Modify `scripts/video_to_fixed_smpl_to_opensim.py`.

Add parser arguments:

```python
parser.add_argument(
    "--opensim-feedback-loop",
    action="store_true",
    help="Run Stage6 OpenSim feedback loop after world-grounded lower-body optimization.",
)
parser.add_argument("--opensim-feedback-out-dir", default=None)
```

Add validation after existing lower-body validation:

```python
if args.opensim_feedback_loop and not args.world_grounded:
    parser.error("--opensim-feedback-loop requires --world-grounded.")
```

Add default output directory near `lower_body_out`:

```python
opensim_feedback_out = (
    Path(args.opensim_feedback_out_dir).resolve()
    if args.opensim_feedback_out_dir
    else output_root / "_opensim_feedback" / safe_ascii_name(sequence)
)
```

Add function near other command helpers:

```python
def build_opensim_feedback_cmd(args, input_pkl: Path, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/opensim_feedback_loop.py",
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
        "--max-root-y-shift",
        str(args.lower_body_max_root_y_shift),
        "--pose-iterations",
        str(args.lower_body_pose_iterations),
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
        "--opensim-cmd",
        args.opensim_cmd,
    ]
    if args.free_root:
        cmd.append("--free-root")
    return cmd
```

After `retarget_input_pkl` is set to lower-body output, run Stage6 when requested:

```python
if args.opensim_feedback_loop:
    feedback_input = world_grounded_out / "optimized_canonical_wham_output.pkl"
    feedback_cmd = build_opensim_feedback_cmd(args, feedback_input, opensim_feedback_out)
    run(feedback_cmd)
    retarget_input_pkl = opensim_feedback_out / "selected_corrected_smpl.pkl"
    require_file(retarget_input_pkl, "Stage6 selected corrected SMPL pkl")
```

Print final Stage6 paths:

```python
if args.opensim_feedback_loop:
    print(f"opensim_feedback_dir: {opensim_feedback_out}")
    print(f"selected_corrected_smpl_pkl: {retarget_input_pkl}")
```

- [ ] **Step 4: Run top-level pipeline tests**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_video_pipeline_stage6.py tests\smpl_optimization\test_opensim_feedback_loop_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts\video_to_fixed_smpl_to_opensim.py tests\smpl_optimization\test_video_pipeline_stage6.py
git commit -m "feat: add stage6 pipeline flag"
```

---

### Task 7: Add Ablation Stage6 Row

**Files:**
- Modify: `scripts/run_wham_ablation.py`
- Modify: `tests/smpl_optimization/test_run_wham_ablation.py`

- [ ] **Step 1: Write failing ablation stage test**

Modify `test_ablation_stage_ids_are_expected()` in `tests/smpl_optimization/test_run_wham_ablation.py`:

```python
def test_ablation_stage_ids_are_expected():
    assert [stage["id"] for stage in STAGES] == [
        "00_raw_wham",
        "01_fixed_beta",
        "02_world_grounded",
        "03_root_y_only",
        "04_lower_body_full",
        "05_opensim_feedback",
    ]
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_run_wham_ablation.py::test_ablation_stage_ids_are_expected -q
```

Expected: FAIL because `05_opensim_feedback` is not yet in `STAGES`.

- [ ] **Step 3: Add Stage6 to ablation only when requested**

Add to `STAGES` in `scripts/run_wham_ablation.py`:

```python
{
    "id": "05_opensim_feedback",
    "description": "OpenSim feedback loop selected corrected SMPL.",
},
```

Add parser flag:

```python
parser.add_argument("--run-opensim-feedback-loop", action="store_true")
```

In `main()`, after `lower_body_pkl`:

```python
feedback_dir = ablation_root / "05_opensim_feedback"
feedback_pkl = feedback_dir / "selected_corrected_smpl.pkl"
```

If `args.run_opensim_feedback_loop`, run:

```python
run(
    [
        sys.executable,
        "scripts/opensim_feedback_loop.py",
        "--input-pkl",
        str(world_pkl),
        "--out-dir",
        str(feedback_dir),
        "--fps",
        str(args.fps),
        "--track-id",
        str(args.track_id),
        "--device",
        args.device,
        "--chunk-size",
        str(args.chunk_size),
        "--max-root-y-shift",
        str(args.lower_body_max_root_y_shift),
        "--pose-iterations",
        str(args.lower_body_pose_iterations),
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
        "--opensim-cmd",
        str(args.opensim_cmd),
        "--free-root",
    ]
)
```

Add to `stage_pkls` only when it exists:

```python
if args.run_opensim_feedback_loop:
    stage_pkls["05_opensim_feedback"] = feedback_pkl
```

When computing rows, skip Stage6 unless requested:

```python
for stage in STAGES:
    if stage["id"] == "05_opensim_feedback" and not args.run_opensim_feedback_loop:
        continue
```

- [ ] **Step 4: Run ablation tests**

Run:

```powershell
python -m pytest tests\smpl_optimization\test_run_wham_ablation.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts\run_wham_ablation.py tests\smpl_optimization\test_run_wham_ablation.py
git commit -m "feat: add stage6 ablation option"
```

---

### Task 8: Full Verification

**Files:**
- No required source edits unless tests expose a defect.

- [ ] **Step 1: Run focused unit tests**

Run:

```powershell
python -m pytest tests\smpl_optimization -q
```

Expected: all `tests/smpl_optimization` tests pass.

- [ ] **Step 2: Run world-grounded regression tests**

Run:

```powershell
python -m pytest tests\world_grounded -q
```

Expected: all `tests/world_grounded` tests pass.

- [ ] **Step 3: Run a dry pipeline command without OpenSim feedback**

Run a short existing path if data is available:

```powershell
python scripts\video_to_fixed_smpl_to_opensim.py `
  --video D:\Main\Research\IEProgram\Bad_Muskeleton\WorldSMPLGen\WHAM\examples\forehand_clear\video1.mp4 `
  --output-pth output\stage6_smoke `
  --device cuda `
  --fps 60 `
  --world-grounded `
  --optimize-lower-body `
  --skip-ik
```

Expected: pipeline still reaches lower-body output and prints `corrected_smpl_pkl`.

- [ ] **Step 4: Run one real Stage6 command with OpenSim**

Run:

```powershell
python scripts\video_to_fixed_smpl_to_opensim.py `
  --video D:\Main\Research\IEProgram\Bad_Muskeleton\WorldSMPLGen\WHAM\examples\forehand_clear\video1.mp4 `
  --output-pth output\stage6_smoke `
  --device cuda `
  --fps 60 `
  --world-grounded `
  --optimize-lower-body `
  --opensim-feedback-loop `
  --free-root
```

Expected:

```text
selected_corrected_smpl_pkl: ...\selected_corrected_smpl.pkl
opensim_feedback_dir: ...
```

Check files:

```powershell
Test-Path output\stage6_smoke\_opensim_feedback\*\selected_corrected_smpl.pkl
Test-Path output\stage6_smoke\_opensim_feedback\*\stage6_selection_report.json
```

Both should return `True`.

- [ ] **Step 5: Commit verification fixes if any**

If verification required fixes:

```powershell
git add lib\smpl_optimization scripts tests\smpl_optimization
git commit -m "fix: stabilize stage6 feedback loop"
```

If no fixes were needed, do not create an empty commit.

---

## Self-Review Notes

Spec coverage:

- OpenSim log parsing is covered by Task 1 and Task 4.
- Lower-body-only feedback filtering is covered by Task 1.
- Frame weight threading into SMPL optimization is covered by Task 3.
- Stage6 orchestration and selected output are covered by Task 5.
- Acceptance/rollback is covered by Task 2 and Task 5.
- Top-level command integration is covered by Task 6.
- Ablation visibility is covered by Task 7.
- Verification is covered by Task 8.

Scope boundary:

- This plan implements Stage6 v1 only.
- Stage6 v2 is intentionally not implemented here because it requires marker/side-specific loss routing and OpenSim coordinate plausibility thresholds. That should be a separate plan after v1 is validated.
