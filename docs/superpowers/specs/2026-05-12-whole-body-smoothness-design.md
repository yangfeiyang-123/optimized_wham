# Stage7 Conservative Whole-Body Smoothness Optimizer Design

## Purpose

Stage7 adds a conservative whole-body smoothness pass after the current WHAM correction pipeline. Its goal is to reduce visible jitter and frame-to-frame motion roughness while preserving the corrected SMPL result produced by Stage5 or the selected Stage6 result.

This stage is intentionally conservative. It is not allowed to make the current result worse just to improve a smoothness score. If a candidate motion fails validation, Stage7 selects the unchanged baseline.

## Current Baseline

The existing pipeline already improves raw WHAM in these stages:

1. Fixed beta canonicalization keeps body shape constant across frames.
2. World grounding estimates a ground height and shifts the SMPL sequence to reduce foot penetration.
3. Lower-body optimization reduces foot penetration, foot sliding, and lower-body contact artifacts.
4. OpenSim feedback can generate frame weights from IK marker errors, but the current Stage6 test selected the Stage5 result when feedback did not improve target metrics.

Stage7 starts from the best available corrected SMPL file:

- Prefer Stage6 `selected_corrected_smpl.pkl` when the OpenSim feedback loop has run.
- Otherwise use Stage5 `corrected_smpl.pkl`.

## Non-Goals

Stage7 does not try to infer a new action, change swing mechanics, or perform aggressive pose correction.

It does not optimize SMPL beta. It does not perform differentiable OpenSim optimization in the first version. It does not accept a result solely because the motion looks smoother.

## Design Summary

Stage7 is a selector with rollback:

1. Load the corrected SMPL baseline.
2. Generate one smooth candidate with small residual changes.
3. Evaluate the baseline and candidate with SMPL metrics.
4. Run OpenSim IK on both baseline and candidate.
5. Parse IK logs and OpenSim motion outputs.
6. Accept the candidate only if all safety gates pass and at least one smoothness target improves.
7. Otherwise output the baseline unchanged as the selected result.

The final user-facing output is always `selected_smooth_smpl.pkl`.

## Optimization Variables

The optimizer uses residual variables relative to the baseline:

- `betas`: fixed, no optimization.
- `root translation`: small residual only.
- `global orientation`: small residual only.
- `body_pose`: whole-body residual with region-specific limits.

Region-specific constraints:

- Lower body has the strictest prior and the smallest allowed change.
- Root and pelvis have strict translation and orientation budgets.
- Spine and trunk may change slightly more than lower body.
- Upper body and arms may change the most, but still remain close to the baseline.

This prevents the optimizer from using root drift or lower-body distortion to hide jitter.

## Losses

The candidate generation objective combines these losses:

- `baseline_prior_loss`: keeps all optimized parameters close to the input SMPL sequence.
- `pose_velocity_loss`: discourages unnecessary high-frequency pose changes.
- `pose_acceleration_loss`: reduces frame-to-frame acceleration spikes.
- `pose_jerk_loss`: reduces third-order pose roughness, the main smoothness target.
- `root_acceleration_loss`: reduces root translation acceleration spikes.
- `root_jerk_loss`: reduces root translation jerk.
- `lower_body_guard_loss`: heavily penalizes lower-body deviation.
- `contact_guard_loss`: preserves existing foot contact, foot lock, and penetration behavior.
- `root_shift_guard_loss`: prevents large global shifts in root translation.

The first version should weight acceleration and jerk more than velocity. This preserves intentional fast motion while reducing roughness.

## Validation Metrics

Stage7 compares baseline and candidate on both SMPL-side and OpenSim-side metrics.

SMPL-side metrics:

- beta variation maximum absolute value.
- foot penetration maximum and mean.
- contact foot sliding mean and maximum.
- root vertical jitter.
- root acceleration and jerk RMS.
- whole-body pose acceleration and jerk RMS.
- lower-body pose delta from baseline.
- whole-body pose delta from baseline.

OpenSim-side metrics:

- IK marker RMS mean.
- IK marker RMS maximum.
- maximum marker error.
- marker error outlier count.
- coordinate finite check.
- coordinate range violation count.
- coordinate frame-to-frame jump count.
- coordinate velocity, acceleration, and jerk summary where available.

## Acceptance Gates

The candidate is accepted only if all hard gates pass:

- Frame count is unchanged.
- Beta variation remains zero or unchanged from the baseline if the input is not fixed-beta.
- Foot penetration does not exceed the baseline by more than a small tolerance.
- Contact sliding does not exceed the baseline by more than a small tolerance.
- Lower-body pose delta remains below the strict lower-body threshold.
- Whole-body pose delta remains below the global threshold.
- Root translation shift remains below the root budget.
- OpenSim IK mean RMS does not worsen beyond tolerance.
- OpenSim IK max RMS does not worsen beyond tolerance.
- OpenSim coordinate range violations do not increase.
- OpenSim coordinate jump count does not increase.

First-version default tolerances:

- Foot penetration max may worsen by at most `0.002 m`.
- Contact sliding mean may worsen by at most `2 percent`.
- Contact sliding max may worsen by at most `5 percent`.
- Lower-body pose delta max must stay below `0.05 rad`.
- Whole-body pose delta max must stay below `0.15 rad`.
- Root translation delta max must stay below `0.03 m`.
- Root vertical delta max must stay below `0.015 m`.
- OpenSim IK mean RMS may worsen by at most `2 percent`.
- OpenSim IK max RMS may worsen by at most `5 percent`.
- Coordinate frame-to-frame jump count must not increase.
- Coordinate range violation count must not increase.

The candidate must also satisfy at least one improvement gate:

- Whole-body pose jerk improves.
- Root translation jerk improves.
- OpenSim coordinate jerk improves.
- OpenSim coordinate jump score improves.

If any hard gate fails, Stage7 rejects the candidate and selects the baseline.

## Rollback Behavior

Rollback is part of the intended design, not an error condition.

When the candidate is rejected:

- `selected_smooth_smpl.pkl` is copied from the baseline.
- The report records `accepted: false`.
- The report includes every failed gate and the baseline/candidate values.
- The downstream retargeting step continues with the baseline.

This makes Stage7 safe for batch use across all videos.

## Outputs

For each video, Stage7 writes:

- `baseline_smpl.pkl`
- `smooth_candidate_smpl.pkl`
- `selected_smooth_smpl.pkl`
- `baseline_opensim/`
- `candidate_opensim/`
- `stage7_smoothness_report.json`

The report must contain:

- input path.
- accepted/rejected decision.
- selected output path.
- SMPL metrics before and after.
- OpenSim metrics before and after.
- failed gate list.
- improvement gate list.
- all optimizer hyperparameters.

## Command Integration

The main pipeline should expose a flag:

```powershell
python scripts\video_to_fixed_smpl_to_opensim.py `
  --video <video_path> `
  --output-pth output\demo `
  --device cuda `
  --fps 60 `
  --world-grounded `
  --optimize-lower-body `
  --whole-body-smooth `
  --free-root
```

The ablation runner should add a sixth or seventh stage depending on whether Stage6 is enabled:

- `05_opensim_feedback` when OpenSim feedback loop is enabled.
- `06_whole_body_smooth` or `07_whole_body_smooth` after the selected corrected SMPL.

## Testing Plan

Unit tests:

- Region mask construction is deterministic and covers expected SMPL joints.
- Smoothness metrics report lower values for known smoothed synthetic motion.
- Acceptance selection rejects candidates with worse penetration.
- Acceptance selection rejects candidates with worse OpenSim IK metrics.
- Acceptance selection accepts candidates that improve smoothness without violating gates.
- Rollback produces a selected file identical to the baseline when rejected.

Smoke test:

- Run Stage7 on `examples/forehand_clear/video1.mp4`.
- Compare baseline and selected result.
- Confirm selected result is either safer/smoother or exactly the baseline.

Batch validation:

- Run all ten `forehand_clear` videos.
- Produce a summary table with accepted count, rejected count, smoothness changes, foot-contact changes, and OpenSim changes.
- No selected output may worsen hard-gate metrics versus its baseline.

## Confidence Statement

There is no 100 percent proof that the optimized motion is more physically real because the project does not have ground-truth 3D motion capture or measured kinetics. The practical confidence target is different: Stage7 should be safe to use because it only accepts a candidate when measurable SMPL and OpenSim validation metrics do not get worse.

The strategy is therefore high-confidence for non-regression, not for perfect physical truth.
