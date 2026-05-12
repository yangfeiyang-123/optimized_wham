# Stage6 OpenSim Feedback Loop Design

Date: 2026-05-12

## Goal

Stage6 的目标是在当前 WHAM 后处理流水线中加入 OpenSim 反馈闭环，使最终 SMPL 不只是满足 SMPL 空间的脚地几何约束，也能对 OpenSim IK 的下肢拟合问题做出响应。

当前推荐输出是：

```text
selected_corrected_smpl.pkl
```

它代表经过 stage6 选择后的最终 SMPL。如果 OpenSim feedback 版本通过验收，它指向反馈优化后的结果；如果反馈优化变差，它回退到 stage5 的结果。

## Current Pipeline

当前已有五个消融阶段：

```text
stage1_raw
  = raw WHAM output

stage2_fixed_beta
  = fixed_beta(stage1_raw)

stage3_world_grounded
  = world_grounded(stage2_fixed_beta)

stage4_root_y_only
  = root_y_only(stage3_world_grounded)

stage5_lower_body_full
  = lower_body_pose + root_y optimization(stage3_world_grounded)
```

Stage5 的优化发生在 SMPL 空间中。它用 WHAM contact、SMPL foot points、ground_y、penetration、foot sliding、smoothness 和 WHAM prior 来优化下肢 pose 与 root_y。OpenSim 当前只负责 retarget、IK 输出和人工/指标验证，不参与修改 `corrected_smpl.pkl`。

Stage6 要解决这个缺口：

```text
SMPL -> OpenSim
```

变成：

```text
SMPL -> OpenSim -> SMPL -> OpenSim -> selected result
```

## Non-Goals

- 不在第一版中实现 OpenSim 到 SMPL pose 的直接反解。
- 不对 OpenSim IK 进行可微反向传播。
- 不把 OpenSim `.mot` 直接当作 SMPL pose 的替代结果。
- 不优化 beta；beta 必须保持 fixed。
- 不无条件覆盖 stage5。Stage6 的反馈结果必须通过验收，否则回退。
- 不做无限迭代。Stage6 v1 最多只做一轮 OpenSim feedback。

## Stage6 v1: Conservative OpenSim Feedback Weighting

### Summary

第一版采用保守闭环：

```text
stage3_world_grounded
  -> stage5 lower_body_full
  -> iter_00_stage5_smpl.pkl
  -> OpenSim retarget + IK
  -> parse IK frame RMS and max marker
  -> build lower-body feedback weights
  -> weighted lower_body_full optimization
  -> iter_01_opensim_feedback_smpl.pkl
  -> OpenSim retarget + IK
  -> accept or rollback
  -> selected_corrected_smpl.pkl
```

OpenSim 不直接告诉 SMPL 往哪个方向改。OpenSim 只告诉 SMPL optimizer 哪些帧、哪些下肢区域在 OpenSim IK 中更值得关注。真正的几何优化方向仍来自 SMPL 的可微 loss。

### Inputs

Required:

- `optimized_canonical_wham_output.pkl` from stage3
- fixed beta fields
- WHAM contact fields if available
- SMPL foot fields if available
- retarget config `configs/retarget/smpl_to_mimicmsk_opensim.yaml`
- OpenSim command path
- FPS

Optional:

- existing stage5 `corrected_smpl.pkl`
- previous OpenSim IK log
- max frame limit for debug runs

### Outputs

Stage6 output directory should contain:

```text
iter_00_stage5_smpl.pkl
iter_00_opensim/
iter_00_opensim/opensim_ik.mot
iter_00_opensim/opensim_ik.log

opensim_feedback_weights.npy
opensim_feedback_report.json

iter_01_opensim_feedback_smpl.pkl
iter_01_opensim/
iter_01_opensim/opensim_ik.mot
iter_01_opensim/opensim_ik.log

stage6_selection_report.json
selected_corrected_smpl.pkl
```

The pipeline should never require downstream tools to guess which result is best. They should consume `selected_corrected_smpl.pkl`.

### Feedback Signal

OpenSim feedback is extracted from IK log lines:

```text
Frame N: marker error: RMS = ..., max = ... (marker_name)
```

For each frame, parse:

- frame index
- RMS marker error
- max marker error
- max marker name

Only lower-body marker errors can drive lower-body feedback:

```text
hip_l, hip_r
knee_l, knee_r
ankle_l, ankle_r
LTOE, RTOE
toe_l, toe_r
l_foot_touch, r_foot_touch
```

Non-lower-body markers are diagnostic only:

```text
head
shoulder
elbow
wrist
chest
pelvis
```

If the max marker is not lower-body, that frame must not receive strong lower-body feedback from OpenSim.

### Weight Construction

Frame feedback weights should start at 1.0.

For lower-body frames with high OpenSim RMS, increase weight up to a capped value:

```text
min_weight = 1.0
max_weight = 2.5
```

Weights must be temporally smoothed so isolated single-frame spikes do not produce sharp pose edits. The feedback should be strongest on contiguous lower-body error windows.

Weights must be combined with contact confidence. A frame should only receive strong foot/contact feedback when contact is plausible:

```text
contact_conf =
  wham_contact
  * foot_height_near_ground_score
  * low_foot_speed_score
```

This protects badminton footwork from being over-locked during fast steps, lunges, takeoff, and landing transitions.

### Weighted SMPL Optimization

The second SMPL optimization pass reuses the current stage5 optimizer, but passes `frame_weights` into `optimize_lower_body_pose_smpl`.

The weighted loss keeps the same loss family:

```text
contact_ground_loss
penetration_loss
foot_lock_loss
pose_smoothness_loss
root_smoothness_loss
wham_pose_prior
root_prior
```

OpenSim feedback only scales the importance of selected frames. It does not become a direct differentiable geometry target.

### Acceptance Criteria

`iter_01_opensim_feedback_smpl.pkl` is accepted only if all hard checks pass:

```text
beta_variation_after_max_abs == 0
frame_count unchanged
OpenSim mean RMS not worse
OpenSim max RMS not worse
SMPL max foot penetration not worse
SMPL contact foot sliding not worse
root vertical jitter not worse
upper-body pose delta small
lower-body pose delta bounded
root_y total shift <= configured budget
```

Additionally, at least one target metric must improve:

```text
OpenSim lower-body marker RMS reduced
or SMPL foot penetration reduced
or SMPL contact foot sliding reduced
or root vertical jitter reduced
```

If any hard check fails, `selected_corrected_smpl.pkl` must be stage5 `iter_00_stage5_smpl.pkl`.

### Why This Is Safe Enough

This strategy does not guarantee `iter_01` is always better. It guarantees that the selected output does not accept an unverified degradation.

The practical confidence comes from:

```text
attempt OpenSim-informed improvement
measure it
accept only if safer metrics pass
rollback otherwise
```

## Risk Review and Mitigations

### Risk: IK RMS high for upper body, not lower body

Mitigation:

Only lower-body marker names create lower-body feedback weights. Upper-body marker errors are written to the report but do not drive leg optimization.

### Risk: Free-root IK hides root or scale errors

Mitigation:

OpenSim RMS is not sufficient for acceptance. The selector also checks SMPL foot penetration, sliding, root jitter, pose delta, beta stability, and frame count.

### Risk: WHAM contact is wrong

Mitigation:

Do not use WHAM contact alone. Combine contact with height-near-ground and low-foot-speed confidence.

### Risk: Fast badminton steps are incorrectly foot-locked

Mitigation:

Only high-confidence static contact windows receive strong foot-lock weighting. Fast or airborne frames receive weak or no foot-lock amplification.

### Risk: OpenSim marker mapping error drives bad SMPL changes

Mitigation:

OpenSim feedback is a frame weight, not a direct target. The optimization direction still comes from SMPL foot-ground/contact geometry.

### Risk: One local window improves while global motion worsens

Mitigation:

Acceptance checks cover full-sequence metrics, including mean sliding, max penetration, root jitter, and OpenSim mean/max RMS.

### Risk: Multiple feedback rounds oscillate

Mitigation:

Stage6 v1 uses only one feedback iteration. Multi-iteration behavior is deferred to v2.

### Risk: OpenSim is slow

Mitigation:

Support a debug frame cap for development, but final acceptance must run on the full sequence used for reporting.

## Stage6 v2: Marker-Aware and Coordinate-Aware Feedback

Stage6 v2 should be implemented only after v1 is stable on representative videos.

The v2 goal is to make feedback more specific without removing rollback protection.

### V2 Direction

V2 extends v1 from frame-level feedback to marker-aware lower-body feedback:

```text
OpenSim marker error
  -> lower-body region weights
  -> side-specific and joint-specific SMPL losses
```

Example:

```text
max marker = ankle_l
  -> increase left ankle/toe/foot contact losses
  -> optionally increase left knee/ankle pose smoothness
  -> do not increase right leg losses
```

```text
max marker = knee_r
  -> increase right hip/knee/ankle pose consistency
  -> do not over-weight foot lock unless foot contact is also high
```

### V2 Additions

1. Marker-to-region map:

```text
hip_l -> left_hip_chain
knee_l -> left_knee_chain
ankle_l -> left_ankle_chain
LTOE -> left_foot_contact
hip_r -> right_hip_chain
knee_r -> right_knee_chain
ankle_r -> right_ankle_chain
RTOE -> right_foot_contact
```

2. Side-specific weights:

```text
left_leg_frame_weights
right_leg_frame_weights
left_foot_contact_weights
right_foot_contact_weights
```

3. Loss routing:

```text
foot marker error
  -> contact_ground, penetration, foot_lock

knee/hip marker error
  -> lower-body pose smoothness and WHAM prior balance

mixed lower-body errors
  -> frame-level lower-body weight
```

4. Optional OpenSim coordinate plausibility checks:

Read `opensim_ik.mot` and flag biologically suspicious lower-body coordinates, such as knee hyperextension or extreme ankle values, using conservative thresholds. In v2 these checks should be validation gates, not direct SMPL targets.

### V2 Non-Goals

- Still do not directly solve OpenSim-to-SMPL inverse mapping.
- Still do not use OpenSim as a differentiable model.
- Still keep selected-output rollback.
- Still keep beta fixed.

### V2 Acceptance

V2 must keep all v1 acceptance checks and add:

```text
side-specific lower-body marker RMS not worse
OpenSim lower-body coordinate plausibility not worse
no new extreme knee/ankle coordinate outliers
```

V2 can be accepted as the new default only after it beats v1 on a batch such as:

```text
examples/forehand_clear/video1.mp4 ... video10.mp4
```

## Command Shape

The eventual user-facing command should remain close to the existing pipeline:

```powershell
python scripts\video_to_fixed_smpl_to_opensim.py `
  --video path\to\video.mp4 `
  --output-pth output\demo `
  --device cuda `
  --fps 60 `
  --world-grounded `
  --optimize-lower-body `
  --opensim-feedback-loop `
  --free-root
```

For direct stage6 execution, a separate script can be clearer:

```powershell
python scripts\opensim_feedback_loop.py `
  --input-pkl output\demo\_world_grounded\...\optimized_canonical_wham_output.pkl `
  --out-dir output\demo\_opensim_feedback\... `
  --fps 60 `
  --device cuda `
  --free-root
```

The top-level pipeline can call this script when `--opensim-feedback-loop` is enabled.

## Reports

The key report is:

```text
stage6_selection_report.json
```

It should include:

- selected iteration: `iter_00` or `iter_01`
- reason for selection
- hard acceptance checks
- target metric improvements
- OpenSim metrics before and after
- SMPL ground/contact metrics before and after
- pose delta report
- feedback weight statistics
- ignored non-lower-body marker errors

This report is required because stage6 is only trustworthy if its decision is auditable.

## Testing

Unit tests:

- Parse OpenSim IK logs into per-frame metrics.
- Classify lower-body vs non-lower-body markers.
- Build capped and smoothed feedback weights.
- Ensure non-lower-body marker errors do not increase lower-body weights.
- Ensure selection rolls back when any hard check fails.
- Ensure selection accepts when all hard checks pass and one target metric improves.

Integration tests:

- Run stage6 with `--skip-ik` using synthetic IK metrics.
- Run stage6 on a short real sequence with OpenSim available.
- Verify `selected_corrected_smpl.pkl` exists and points to the accepted result.

Manual validation:

- Compare stage5 vs stage6 selected SMPL visually.
- Compare OpenSim IK RMS.
- Compare foot penetration and contact sliding.
- Confirm failed feedback iterations do not replace selected output.

## Success Criteria

Stage6 v1 is successful if:

- It can run after stage5 without changing existing stage1-stage5 behavior.
- It produces `selected_corrected_smpl.pkl`.
- It accepts OpenSim feedback only when core metrics do not worsen.
- It rolls back safely otherwise.
- Its reports explain the decision.
- It improves at least some representative videos without making failed cases worse.

Stage6 v2 is successful if:

- Marker-aware feedback improves lower-body OpenSim errors beyond v1.
- Left/right leg feedback is correctly isolated.
- OpenSim coordinate plausibility checks catch obvious bad IK outputs.
- V2 remains protected by the same selected-output rollback mechanism.

