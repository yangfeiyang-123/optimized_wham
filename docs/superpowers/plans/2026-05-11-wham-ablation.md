# WHAM Ablation Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one command that generates five WHAM/SMPL ablation stages and writes comparable metrics.

**Architecture:** Create `scripts/run_wham_ablation.py` as a thin orchestration script around existing WHAM, fixed-beta, world-grounded, lower-body, and optional retarget scripts. Add tests for stage definitions and metric CSV/JSON behavior without running CUDA/OpenSim.

**Tech Stack:** Python stdlib, `joblib`, `numpy`, existing `lib.world_grounded` and `lib.smpl_optimization` helpers.

---

### Task 1: Add Ablation Runner

**Files:**
- Create: `scripts/run_wham_ablation.py`

- [ ] **Step 1: Define five stages**

The script must emit exactly these stage IDs:

```python
[
    "00_raw_wham",
    "01_fixed_beta",
    "02_world_grounded",
    "03_root_y_only",
    "04_lower_body_full",
]
```

- [ ] **Step 2: Reuse existing scripts**

The runner calls:

```text
demo.py
scripts/canonicalize_wham_fixed_beta.py
scripts/world_grounded_smpl_optimizer.py
scripts/optimize_smpl_lower_body.py
scripts/retarget_smpl_to_opensim.py
```

`03_root_y_only` calls lower-body optimizer without `--enable-pose-pass`; `04_lower_body_full` includes `--enable-pose-pass`.

- [ ] **Step 3: Write metrics**

Write:

```text
ablation_summary.json
ablation_metrics.csv
```

Metrics include beta variation, foot penetration, contact sliding, root vertical jitter, pose deltas, frame count, and output path.

### Task 2: Add Tests

**Files:**
- Create: `tests/smpl_optimization/test_run_wham_ablation.py`

- [ ] **Step 1: Test stage IDs**

Assert stage order equals the five approved groups.

- [ ] **Step 2: Test CSV writer**

Use fake metric rows and assert headers/values are written consistently.

- [ ] **Step 3: Test command construction**

Check root-only lower-body command does not include `--enable-pose-pass`, while full lower-body command does.

### Task 3: Verify

- [ ] **Step 1: Run focused tests**

```powershell
python -m pytest tests\smpl_optimization\test_run_wham_ablation.py -q
```

- [ ] **Step 2: Run script help**

```powershell
python scripts\run_wham_ablation.py --help
```

