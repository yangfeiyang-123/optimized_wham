# World-Grounded SMPL Optimizer Design

Date: 2026-05-09

## Goal

Build the first production-useful World-Grounded SMPL optimizer for the current WHAM -> fixed-beta SMPL -> OpenSim pipeline.

The optimizer operates primarily in the SMPL layer. It takes a fixed-beta WHAM sequence, uses WHAM contact/feet signals plus SMPL foot geometry to estimate a robust horizontal ground, improves motion smoothness and foot-ground consistency, then sends the optimized SMPL sequence through the existing OpenSim retarget path.

This is the selected first-version approach. OpenSim remains a validation/export backend, not an optimization loop.

## Non-Goals

- Do not implement full OpenSim closed-loop optimization in v1.
- Do not solve camera homography or court-line-based ground estimation in v1.
- Do not optimize SMPL beta; beta remains fixed from `canonical_wham_output.pkl`.
- Do not aggressively change full-body pose. V1 default optimizes root translation only. Lower-body pose changes are experimental and must be behind `--optimize-lower-body`.
- Do not make fixed-root OpenSim IK the default. Free-root retarget remains the default for OpenSim export.

## Current Context

Existing pipeline:

1. `demo.py` runs WHAM and writes `wham_output.pkl`.
2. `scripts/canonicalize_wham_fixed_beta.py` writes `canonical_wham_output.pkl`, `beta_fixed.npy`, and `fixed_beta_report.json`.
3. `scripts/video_to_fixed_smpl_to_opensim.py` runs the full video pipeline.
4. `scripts/retarget_smpl_to_opensim.py` converts fixed-beta SMPL to 17 retarget markers and runs OpenSim IK.

Recent findings:

- WHAM raw beta varies by frame; fixed beta is mandatory.
- OpenSim fixed-root currently makes marker error much worse, so free-root should remain default.
- WHAM can split a single person into multiple track IDs; default retarget should merge tracks.
- OpenSim can fail on non-ASCII paths; OpenSim retarget outputs should use ASCII-safe directories.
- The model's `Geometry` directory should be copied into retarget outputs to avoid mesh warnings.
- WHAM internally predicts contact probability and feet positions, but current `wham_output.pkl` does not save them.

## Inputs

Required:

- `canonical_wham_output.pkl`
- `fixed_beta_report.json`
- FPS

Preferred:

- WHAM predicted contact probability: `contact: [T, 4]`
- WHAM predicted/refined foot points: `feet_world: [T, 4, 3]` or equivalent
- Frame IDs for merged track handling

Fallback:

- If WHAM contact is unavailable, estimate contact from reconstructed SMPL foot point height and velocity.
- If WHAM feet are unavailable, extract foot points from SMPL vertices/joints.

## Outputs

The optimizer writes an output directory containing:

- `optimized_canonical_wham_output.pkl`
- `ground_plane.json`
- `contact_report.json`
- `quality_report.json`
- `optimization_report.json`
- `retarget/opensim_ik.mot`
- `retarget/smpl_17pt_markers.trc`
- `retarget/MimicMSK_OpenSim_retarget_markers.osim`

The optimized pkl keeps the WHAM-style structure, including fixed `betas`, so downstream scripts can consume it like the current `canonical_wham_output.pkl`.

## Architecture

### 1. WHAM Signal Export

Update WHAM output saving to include contact and feet signals when available:

- `contact`: contact probabilities from WHAM, shape `[T, 4]`
- `feet_world`: world/refined feet from WHAM, shape `[T, 4, 3]`
- `feet_cam` or `feet_local`, optional diagnostic fields if useful

The canonicalizer should preserve these fields when writing fixed-beta output.

### 2. Track Merge Handling

The optimizer should consume a single canonical record. If the input contains multiple tracks, it should support the same policy as retarget:

- `merge`: stitch ID switches by frame ID, preferring the longer track on overlap
- `longest`: use the longest track
- explicit numeric track ID

Default: `merge`.

### 3. Foot Point Provider

A foot point provider builds the points used for ground/contact optimization:

- Preferred source: WHAM `feet_world`
- Fallback source: SMPL output foot joints/regressed feet
- Deferred source: selected SMPL foot vertices for heel/toe/forefoot

The provider should return:

- foot points `[T, K, 3]`
- semantic labels
- source metadata

For v1, `K=4` is acceptable. The design should leave room for heel/toe/forefoot expansion.

### 4. Contact-Aware Ground Estimator

The ground is a horizontal plane in the current OpenSim/SMPL working frame:

```text
ground_y = constant
normal = +Y
```

The estimator must not use a raw minimum foot height. It should use a robust, contact-aware estimate:

1. Compute foot height samples from foot point y coordinates.
2. Compute foot velocity:
   - horizontal speed from x/z velocity
   - vertical speed from y velocity
3. Build sample weights:
   - high WHAM contact probability increases weight
   - low horizontal speed increases weight
   - low vertical speed increases weight
   - outlier foot heights are downweighted
4. Estimate `ground_y` with a weighted median or trimmed weighted percentile.
5. Iterate once:
   - estimate initial `ground_y`
   - update contact confidence using height and velocity
   - re-estimate `ground_y`

Fallback if WHAM contact is missing:

- use low-speed foot points plus trimmed low percentile
- save `contact_source: fallback_velocity_height`

### 5. Contact Confidence

Contact confidence should be soft, not binary:

```text
contact_conf = wham_contact * height_score * speed_score * vertical_speed_score
```

Where:

- `height_score` is high near the estimated ground
- `speed_score` is high when horizontal foot speed is low
- `vertical_speed_score` is high when vertical speed is low

Apply temporal smoothing to contact confidence with a small median or moving-average window. The report should include contact coverage and contact switch rate.

### 6. Motion Smoothing

The first version should improve smoothness conservatively:

- Smooth root translation with a low-pass or Savitzky-Golay style filter.
- Smooth root orientation only if the representation is stable.
- Do not smooth lower-body pose by default. If `--optimize-lower-body` is enabled, apply only small temporal regularization to hips/knees/ankles.

The smoother must preserve action timing. It should not remove fast badminton footwork or jump/landing transitions.

### 7. Ground and Foot Correction

The first version should focus on root translation correction rather than aggressive pose editing:

- Correct global/root vertical offset so high-confidence contact foot points lie on or slightly above `ground_y`.
- Penalize penetration below ground.
- Penalize foot skating only during high-confidence contact.
- Keep corrections close to the original canonical SMPL motion.

The v1 optimizer may implement this as a staged numeric optimizer over root translation:

Stage 1:

- optimize root y offset and smooth root translation
- reduce foot penetration
- keep original motion close

Stage 2:

- optimize per-frame root translation residuals
- apply contact anti-skate loss
- apply temporal acceleration loss

Lower-body pose optimization is not part of the v1 default path. If implemented during v1, it must be gated by `--optimize-lower-body`, constrained to small changes in hips/knees/ankles, and excluded from the default quality comparison.

### 8. OpenSim Export

After optimization:

1. Write `optimized_canonical_wham_output.pkl`.
2. Run existing retarget script on the optimized pkl.
3. Use free-root OpenSim IK by default.
4. Use ASCII-safe output directories.
5. Copy OpenSim `Geometry` assets into retarget output.

The pipeline should preserve the current command style:

```powershell
python scripts/video_to_fixed_smpl_to_opensim.py `
  --video path/to/video.mp4 `
  --output-pth output/demo `
  --device cuda `
  --fps 60 `
  --world-grounded
```

The pipeline flag is `--world-grounded`.

## Data Flow

```text
video
  -> WHAM demo.py
  -> wham_output.pkl with contact/feet
  -> canonicalize fixed beta
  -> canonical_wham_output.pkl with fixed beta and contact/feet
  -> world-grounded SMPL optimizer
  -> optimized_canonical_wham_output.pkl
  -> OpenSim retarget, free-root
  -> opensim_ik.mot
```

## Error Handling

- If no foot/contact signal exists, warn and use velocity/height fallback.
- If fewer than a minimum number of contact samples exist, fall back to robust low percentile.
- If the estimated ground has poor confidence, write the output but mark `ground_confidence: low`.
- If OpenSim output path is non-ASCII, use an ASCII-safe output path.
- If OpenSim IK fails, keep optimized SMPL outputs and report that retarget failed.

## Quality Metrics

`quality_report.json` should include:

- `beta_variation_after_max_abs`
- `num_frames`
- `ground_y`
- `ground_confidence`
- `contact_source`
- `contact_coverage`
- `foot_penetration_mean_cm`
- `foot_penetration_max_cm`
- `foot_skating_contact_mean_cm_s`
- `root_acceleration_before`
- `root_acceleration_after`
- `opensim_ik_rms_mean` if OpenSim runs
- `opensim_ik_rms_max` if OpenSim runs

## Testing

Minimum tests:

- Unit test weighted ground estimation with synthetic foot/contact data.
- Unit test fallback ground estimation when contact is missing.
- Unit test track merge keeps full frame range.
- Smoke test optimizer on an existing fixed-beta pkl with `--skip-ik`.
- Smoke test OpenSim retarget on a short sequence.

Manual validation:

- Compare raw vs optimized foot height curves.
- Compare raw vs optimized root acceleration.
- Compare raw vs optimized OpenSim marker RMS.
- Visualize whether contact foot points stay near ground without destroying motion timing.

## First-Version Success Criteria

The first version is successful if:

- Beta remains fixed.
- The full sequence length is preserved after track merge.
- Foot penetration visibly decreases.
- Contact foot skating decreases or does not get worse.
- Root motion is smoother without removing the main action.
- OpenSim free-root IK still produces low marker error.
- Reports make failures diagnosable.

## Deferred Work

- Full SMPL lower-body pose optimizer.
- OpenSim closed-loop optimization.
- Inverse dynamics residual validation.
- Homography/court-line ground estimation.
- Learned contact predictor.
- Multi-person tracking selection beyond merge/longest/manual ID.
