# SMPL Lower-Body Ground Optimizer Design

Date: 2026-05-11

## Goal

Build the next-stage optimizer after WHAM and fixed-beta canonicalization. The primary output is a corrected SMPL time-series file that can later be retargeted to MuJoCo for RL training.

OpenSim is not the final product. It is used as a biomechanical evaluator and feedback source to improve the SMPL sequence. The first production target is:

```text
corrected_smpl.pkl
```

The corrected SMPL sequence should keep WHAM's useful global/root estimate, keep body shape fixed, reduce foot penetration and foot sliding, improve lower-body biological plausibility, and preserve the original video motion as much as possible.

## Non-Goals

- Do not optimize SMPL beta. Beta must remain fixed after canonicalization.
- Do not rewrite WHAM inference.
- Do not make OpenSim the only output.
- Do not require differentiating through OpenSim in the first version.
- Do not optimize full-body pose equally. Upper body and torso are mostly preservation targets.
- Do not assume the first version produces final RL-ready motion without validation.

## Current Baseline

The project already adds these stages on top of WHAM:

1. Fixed-beta canonicalization, producing a stable beta across all frames.
2. Contact-aware ground estimation from WHAM contact and foot trajectories.
3. World-grounded root translation correction.
4. SMPL-to-OpenSim retargeting with 17 reference points.
5. Optional OpenSim IK output for inspection.

The remaining problem is that the SMPL sequence can still have lower-body artifacts:

- feet can still penetrate or float relative to the intended ground;
- contact frames can slide;
- root translation can hide lower-body errors;
- lower-body SMPL pose can remain implausible even when marker error is acceptable;
- OpenSim/MuJoCo consumers may expose errors not obvious in SMPL-only visualization.

## Recommended Architecture

Use a hybrid closed-loop optimizer:

```text
WHAM raw output
  -> fixed beta canonical SMPL
  -> contact-aware ground estimation
  -> root low-frequency correction
  -> SMPL lower-body optimizer pass 1
  -> OpenSim IK/analyze feedback
  -> SMPL lower-body optimizer pass 2
  -> corrected_smpl.pkl
  -> MuJoCo/OpenSim validation exports
```

The main optimizer remains in SMPL space so it can directly output a corrected SMPL pkl. OpenSim runs as a low-frequency outer-loop evaluator because it is slow and non-differentiable in the current toolchain.

## Optimization Variables

Hard-fixed:

- SMPL beta.
- Frame count and timestamps.
- Upper-body pose, except for very small optional smoothing if explicitly enabled later.
- Original WHAM global orientation by default.

Optimized:

- Small residual on root translation.
- Left/right hip pose residuals.
- Left/right knee pose residuals.
- Left/right ankle pose residuals.
- Left/right toe pose residuals.

The optimizer should apply residuals on top of the canonical WHAM SMPL sequence rather than replacing the sequence from scratch.

## Root Translation Policy

Root translation is allowed to move, but only as a constrained low-frequency correction.

Purpose:

- align the body to the estimated ground;
- reduce vertical jitter;
- avoid global drift from bad WHAM frames;
- support contact consistency when the entire body height is biased.

Limits:

- root residuals should be small relative to the original WHAM trajectory;
- vertical root correction should be smooth;
- root should not be the main way to fix foot penetration;
- lower-body pose should handle pose-specific foot errors.

The report must separate root correction from pose correction so failures are diagnosable.

## Ground and Contact Model

Ground should remain a horizontal plane in the first version, but the estimator must be robust:

- use WHAM contact probability when available;
- combine contact with foot height distribution;
- use foot velocity to discount sliding or airborne frames;
- use multiple foot points rather than only ankle markers;
- produce confidence metrics and fallback status.

The optimizer should treat contact as soft evidence, not a hard truth. Incorrect contact labels must not force a swinging foot onto the ground.

## Foot Model

Foot constraints should use a multi-point foot representation:

- toe;
- heel;
- ankle;
- available SMPL foot vertices or a configured foot vertex subset;
- existing WHAM `feet_refined` or `feet_world` when available.

The optimizer should distinguish:

- contact foot: should be near ground and should not slide in XZ;
- swing foot: may be above ground and should not be forced down;
- uncertain contact: receives weaker ground and lock weights.

## Loss Terms

Required losses:

```text
L_beta_fixed        hard constraint; beta is not an optimization variable
L_ground_contact    high-confidence contact foot points should be near ground
L_penetration       foot points should not go below ground
L_foot_lock         high-confidence contact foot points should not slide in XZ
L_root_smooth       root translation residual should be smooth and low-frequency
L_pose_smooth       lower-body pose should avoid frame-to-frame jitter
L_wham_prior        optimized SMPL should stay close to original WHAM
L_upper_preserve    upper body should remain unchanged or nearly unchanged
L_joint_limit_soft  lower-body pose should avoid obvious biological violations
```

OpenSim feedback should be represented as outer-loop signals in the first version:

```text
OpenSim IK RMS
OpenSim marker max error
OpenSim lower-body coordinate range warnings
OpenSim foot/body ground clearance
OpenSim contact-related failure frames
```

These signals should influence the second optimization pass through adjusted weights or frame masks, not through direct OpenSim gradients.

## OpenSim Feedback Loop

The first version should use OpenSim as an evaluator in a bounded loop:

1. Optimize SMPL with internal ground/contact/smoothness losses.
2. Retarget the result to OpenSim.
3. Run OpenSim IK/analyze.
4. Read generated reports.
5. Identify frames with poor lower-body marker error, bad foot clearance, or suspicious joint ranges.
6. Run a second SMPL optimization pass with adjusted weights or masks.
7. Save final corrected SMPL and validation reports.

This keeps the system practical while still using OpenSim to guide the SMPL correction.

## MuJoCo/RL Readiness

The final corrected SMPL pkl is intended for later MuJoCo retargeting and RL training. The optimizer must therefore report motion quality in terms useful for RL:

- contact foot sliding;
- foot penetration depth;
- root vertical jitter;
- lower-body pose delta from WHAM;
- sudden velocity or acceleration spikes;
- frame continuity;
- ground clearance after MuJoCo/OpenSim-axis conversion if available.

MuJoCo validation is not required inside the first optimizer implementation, but the output format and reports should make the next MuJoCo stage straightforward.

## Outputs

The lower-body optimizer should create a new output directory containing:

```text
corrected_smpl.pkl
lower_body_optimization_report.json
ground_contact_report.json
pose_delta_report.json
opensim_feedback_report.json
validation_summary.json
```

Optional artifacts:

```text
before_after_smpl_preview.mp4
foot_height_plot.png
foot_sliding_plot.png
opensim_ik_after.mot
```

The pipeline should clearly print which SMPL file is the final corrected output.

## Pipeline Integration

Add a pipeline flag:

```text
--optimize-lower-body
```

Expected pipeline:

```text
python scripts/video_to_fixed_smpl_to_opensim.py `
  --video <video> `
  --output-pth output/demo `
  --device cuda `
  --fps <fps> `
  --world-grounded `
  --optimize-lower-body
```

The pipeline should still allow users to run only part of the stack:

- fixed beta only;
- fixed beta plus world grounding;
- fixed beta plus world grounding plus lower-body SMPL optimization;
- optional OpenSim retarget/IK after correction.

## Validation Criteria

A run should be considered successful only if the validation summary reports:

```text
beta_variation_after_max_abs == 0
frame_count_unchanged == true
smpl_foot_penetration_reduced == true
smpl_contact_foot_sliding_reduced == true
root_vertical_jitter_not_worse == true
upper_body_pose_delta_small == true
lower_body_pose_delta_bounded == true
opensim_ik_rms_not_worse == true
opensim_ground_clearance_not_worse == true
```

If any criterion fails, the optimizer should still save outputs but mark the run as degraded. The report should identify the failing frames and likely cause.

## Failure Handling

Known failure modes and responses:

- Bad WHAM pose: keep correction bounded and report that input quality is insufficient.
- Bad contact labels: soften contact weights using height and velocity evidence.
- Ground estimate uncertainty: use conservative root correction and mark confidence low.
- OpenSim IK failure: keep SMPL pass-1 output and write an OpenSim failure report.
- Excessive pose change: reject or downweight the second pass.
- Remaining toe penetration: report residual penetration and affected frames.
- Output stale directory: include optimizer version and input hash in reports.

## Implementation Boundaries

The implementation should be split into focused modules:

```text
lib/smpl_optimization/lower_body.py
lib/smpl_optimization/losses.py
lib/smpl_optimization/reports.py
lib/smpl_optimization/opensim_feedback.py
scripts/optimize_smpl_lower_body.py
```

Existing world-grounded modules should remain responsible for ground estimation and basic root translation alignment. The new lower-body optimizer should consume their output rather than duplicating that logic.

## Testing Strategy

Unit tests:

- beta remains fixed after optimization;
- frame count and track structure remain unchanged;
- contact masks are padded/clipped safely;
- loss terms handle missing contact and missing foot data;
- root residual is bounded;
- lower-body pose residual is bounded;
- report files contain required keys.

Synthetic sequence tests:

- foot penetration is reduced;
- contact foot sliding is reduced;
- root jitter is not increased;
- swing foot is not incorrectly pinned to the ground;
- OpenSim failure does not prevent saving corrected SMPL.

Pipeline tests:

- `--world-grounded --optimize-lower-body` produces `corrected_smpl.pkl`;
- `--skip-ik` still works;
- existing fixed-beta and retarget behavior remain compatible.

## Confidence Assessment

The design has high confidence as an engineering direction because it keeps the corrected SMPL as the main product while using OpenSim for biomechanical feedback. It does not have absolute confidence until tested on real videos and downstream MuJoCo retargeting.

The design therefore requires measurable validation instead of relying on visual judgment alone.

