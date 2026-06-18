# Reference Bundle Training Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the non-RL-experiment pieces that connect Optimized-WHAM contact-preserving reference bundles to ASI-PPO/MuscleMimic training inputs and diagnostics.

**Architecture:** Optimized-WHAM remains the reference producer and exports validated `contact_reference_bundle_v1` bundles in `amass_zup`. BadmintonMimic/MuscleMimic remains the consumer and adds lightweight, testable loaders, phase sampling, cache building, and diagnostics without launching PPO training.

**Tech Stack:** Python, NumPy, JSON/JSONL, pytest, existing `BadmintonMimic.data.reference_bundle` and `lib.world_grounded.reference_bundle`.

---

### Task 1: Optimized-WHAM Bundle Quality And Body Graph

**Files:**
- Create: `lib/world_grounded/body_graph.py`
- Modify: `lib/world_grounded/reference_bundle.py`
- Test: `tests/reference_bundle/test_export_bundle.py`

- [ ] **Step 1: Write failing tests**

Add tests that export a bundle with `body_keypoints`, then assert `motion.npz` contains `body_keypoints`, `body_keypoint_labels`, `body_laplacian`, and that a bad quality report is normalized to `quality_tier == "D"` and `usable_for_training is False`.

- [ ] **Step 2: Run tests to verify failure**

Run: `PYTHONPATH=. python -m pytest tests/reference_bundle/test_export_bundle.py -q`
Expected: FAIL because those arrays and quality-gate normalization are not implemented.

- [ ] **Step 3: Implement body graph utilities and bundle export**

Move the body graph definition to `lib/world_grounded/body_graph.py`, add `laplacian_coordinates(points, labels, graph)`, use it from `reference_bundle.py`, and call `evaluate_quality_gates()` when raw reports do not already include `quality_tier`.

- [ ] **Step 4: Run tests to verify pass**

Run: `PYTHONPATH=. python -m pytest tests/reference_bundle/test_export_bundle.py tests/world_grounded/test_quality_gates.py -q`
Expected: PASS.

### Task 2: ASI-PPO Reference Loader And Phase Manager

**Files:**
- Modify: `/data3/yangfeiyang/WorkSpace/musclemimic/BadmintonMimic/data/reference_bundle.py`
- Create: `/data3/yangfeiyang/WorkSpace/musclemimic/BadmintonMimic/asi/reference_phase.py`
- Test: `/data3/yangfeiyang/WorkSpace/musclemimic/tests/unit/test_reference_bundle_loader.py`
- Test: `/data3/yangfeiyang/WorkSpace/musclemimic/tests/unit/test_reference_phase_manager.py`

- [ ] **Step 1: Write failing tests**

Add tests that require `load_reference_bundle()` to expose optional `body_keypoints`, `body_laplacian`, and labels, plus tests for mapping control steps to reference frames at different FPS values.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/test_reference_bundle_loader.py tests/unit/test_reference_phase_manager.py -q`
Expected: FAIL because the new fields and phase manager do not exist.

- [ ] **Step 3: Implement loader fields and phase manager**

Extend `ReferenceBundle` with optional body arrays and add `ReferencePhaseManager` with `frame_at_control_step()`, `sample_indices()`, and `effective_ref_stride`.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/unit/test_reference_bundle_loader.py tests/unit/test_reference_phase_manager.py -q`
Expected: PASS.

### Task 3: ASI-PPO Tracking Cache And Diagnostics

**Files:**
- Create: `/data3/yangfeiyang/WorkSpace/musclemimic/BadmintonMimic/asi/tracking_cache.py`
- Create: `/data3/yangfeiyang/WorkSpace/musclemimic/BadmintonMimic/asi/diagnostics.py`
- Create: `/data3/yangfeiyang/WorkSpace/musclemimic/BadmintonMimic/scripts/build_tracking_reference_cache.py`
- Test: `/data3/yangfeiyang/WorkSpace/musclemimic/tests/unit/test_tracking_reference_cache.py`
- Test: `/data3/yangfeiyang/WorkSpace/musclemimic/tests/unit/test_tracking_diagnostics.py`

- [ ] **Step 1: Write failing tests**

Add tests that build a cache from a reference bundle manifest and assert it writes `tracking_reference_cache.npz` and `retarget_report.json`; add diagnostics tests that classify high root error and stance foot slip frames.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/test_tracking_reference_cache.py tests/unit/test_tracking_diagnostics.py -q`
Expected: FAIL because cache and diagnostics modules do not exist.

- [ ] **Step 3: Implement cache builder and diagnostics**

Implement NumPy-only cache building that writes motion/contact/body arrays plus metadata, and diagnostics that produces frame-indexed failures suitable for feedback to Optimized-WHAM.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/unit/test_tracking_reference_cache.py tests/unit/test_tracking_diagnostics.py -q`
Expected: PASS.

### Task 4: Manifest Builder And Config Contract

**Files:**
- Modify: `/data3/yangfeiyang/WorkSpace/musclemimic/BadmintonMimic/scripts/build_contact_tracking_manifest.py`
- Modify: `/data3/yangfeiyang/WorkSpace/musclemimic/BadmintonMimic/configs/asi_ppo/contact_preserving_tracking.yaml`
- Modify: `/data3/yangfeiyang/WorkSpace/musclemimic/pyproject.toml`
- Test: `/data3/yangfeiyang/WorkSpace/musclemimic/tests/unit/test_build_contact_tracking_manifest.py`

- [ ] **Step 1: Write failing tests**

Add tests that require the JSONL builder to record `coordinate_system`, `cache_path`, and `effective_ref_stride` when cache metadata exists.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/test_build_contact_tracking_manifest.py -q`
Expected: FAIL because these fields are not emitted.

- [ ] **Step 3: Implement manifest metadata and CLI entry point**

Update the manifest builder to read optional cache reports, add a `badminton-build-tracking-reference-cache` console entry point, and add cache paths to the config.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/unit/test_build_contact_tracking_manifest.py -q`
Expected: PASS.

### Task 5: Verification

**Files:**
- No new files.

- [ ] **Step 1: Run Optimized-WHAM verification**

Run: `PYTHONPATH=. python -m pytest tests/reference_bundle/test_export_bundle.py tests/world_grounded/test_quality_gates.py tests/world_grounded/test_contact_segments.py tests/world_grounded/test_stance_anchor.py -q`
Expected: PASS.

- [ ] **Step 2: Run MuscleMimic verification**

Run: `uv run pytest tests/unit/test_reference_bundle_loader.py tests/unit/test_reference_phase_manager.py tests/unit/test_tracking_reference_cache.py tests/unit/test_tracking_diagnostics.py tests/unit/test_build_contact_tracking_manifest.py tests/unit/test_badminton_fps_contract.py -q`
Expected: PASS.
