# 3. Optimized-WHAM 和 ASI-PPO 之间的联系

> 目标：定义 Optimized-WHAM 与 ASI-PPO 之间的清晰工程边界、数据合同、目录结构、坐标系约定、质量筛选机制和反馈闭环。核心原则是：**Optimized-WHAM 负责生成高质量 reference；ASI-PPO 负责消费 reference 并训练肌骨模型复现轨迹。**

---

## 0. 总体关系

两者不应该混在一起。

```text
Optimized-WHAM
  = 视频到高质量 SMPL / reference bundle 的生成器

ASI-PPO
  = 高质量 reference bundle 到肌骨控制策略的训练器
```

更具体地说：

```text
视频输入
  -> Optimized-WHAM
      -> WHAM
      -> fixed beta
      -> world-grounded
      -> contact-preserving lower-body optimization
      -> reference bundle
  -> ASI-PPO
      -> load reference bundle
      -> retarget / build tracking cache
      -> PPO / MuscleMimic training
      -> evaluation diagnostics
```

你的目标是不做球拍、不做手部、不做羽毛球任务，因此两者连接的核心数据不是 object interaction，而是：

```text
全身运动轨迹；
root 轨迹；
body/joint reference；
foot contact schedule；
stance anchors；
quality report；
body graph structure。
```

---

## 1. 工程边界

### 1.1 Optimized-WHAM 负责

```text
1. 从视频运行 WHAM；
2. 固定 SMPL beta；
3. 合并或选择 track；
4. 提取 foot points；
5. 估计 ground；
6. 修正 root vertical 和 foot penetration；
7. 提取 contact confidence；
8. 提取 stance segments；
9. 生成 stance anchors；
10. 通过 lower-body optimizer 降低脚滑和穿地；
11. 计算 body graph / quality metrics；
12. 输出 reference bundle；
13. 给每个 clip 打 quality tier。
```

### 1.2 ASI-PPO 负责

```text
1. 读取 reference bundle manifest；
2. 验证坐标系、fps、frame count、quality tier；
3. 将 reference 转成肌骨模型 tracking targets；
4. 加载 contact schedule；
5. 构造 observation；
6. 构造轨迹复现 reward；
7. 构造 foot contact reward；
8. 训练 PPO / MuscleMimic policy；
9. 输出 tracking evaluation；
10. 把失败信息反馈给 reference 数据筛选流程。
```

### 1.3 不应该混淆的职责

| 问题 | 应该在哪边解决 |
|---|---|
| WHAM beta 逐帧变化 | Optimized-WHAM |
| 视频中脚穿地 | Optimized-WHAM |
| 支撑脚 contact 检测 | Optimized-WHAM |
| reference 坐标系转换 | Optimized-WHAM 导出时完成，ASI-PPO 只验证 |
| reference 是否适合训练 | Optimized-WHAM 给出 quality tier，ASI-PPO 可二次过滤 |
| 肌骨模型 body/site mapping | ASI-PPO |
| PPO reward | ASI-PPO |
| 肌肉激活平滑 | ASI-PPO |
| policy failure diagnostics | ASI-PPO |
| 根据失败数据更新 quality gate | 两者之间反馈闭环 |

---

## 2. 推荐目录结构

### 2.1 Optimized-WHAM 输出目录

```text
optimized_wham/
  output/demo/<sequence>/
    wham_output.pkl
    canonical_wham_output.pkl
    fixed_beta_report.json

  output/demo/_world_grounded/<sequence_safe>/
    optimized_canonical_wham_output.pkl
    ground_plane.json
    contact_report.json
    quality_report.json
    optimization_report.json

  output/demo/_lower_body_optimized/<sequence_safe>/
    corrected_smpl.pkl
    lower_body_optimization_report.json
    ground_contact_report.json
    pose_delta_report.json
    validation_summary.json

  output/demo/_reference_bundles/<sequence_safe>/
    manifest.json
    motion.npz
    contact_schedule.npz
    stance_anchors.json
    body_graph.json
    quality_report.json
    processing_report.json
```

### 2.2 ASI-PPO 输入目录

可以不复制数据，直接通过 manifest 指向 Optimized-WHAM 输出：

```text
asi_strengthen_musclemimic/
  data/reference_manifests/
    train_A.jsonl
    train_AB.jsonl
    val_A.jsonl
    debug_one_clip.jsonl

  data/reference_cache/
    <sequence_safe>/
      tracking_reference.npz
      retarget_report.json
      cache_manifest.json

  runs/contact_tracking/<experiment_name>/
    checkpoints/
    logs/
    eval_reports/
```

### 2.3 JSONL manifest

ASI-PPO 的训练 manifest 不直接存数组，只存每个 reference bundle 的路径和摘要：

```json
{"manifest": "D:/.../_reference_bundles/seq1/manifest.json", "quality_tier": "A", "num_frames": 216, "fps": 60.0}
{"manifest": "D:/.../_reference_bundles/seq2/manifest.json", "quality_tier": "B", "num_frames": 184, "fps": 60.0}
```

这样 ASI-PPO 可以灵活组合训练集，而不需要复制大文件。

---

## 3. 数据合同：reference bundle v1

这是两个项目最重要的连接点。

### 3.1 manifest.json

```json
{
  "version": "contact_reference_bundle_v1",
  "sequence": "5月1日-1",
  "sequence_safe": "5_1_-1_3d3c17ec",
  "num_frames": 216,
  "fps": 60.0,
  "unit": "meter",
  "coordinate_system": "amass_zup",
  "up_axis": "z",
  "motion_npz": "motion.npz",
  "contact_npz": "contact_schedule.npz",
  "stance_anchors_json": "stance_anchors.json",
  "body_graph_json": "body_graph.json",
  "quality_report_json": "quality_report.json",
  "processing_report_json": "processing_report.json",
  "source": {
    "video": "examples/5月1日-1.mp4",
    "wham_pkl": "output/demo/5月1日-1/wham_output.pkl",
    "canonical_pkl": "output/demo/5月1日-1/canonical_wham_output.pkl",
    "world_grounded_pkl": "output/demo/_world_grounded/.../optimized_canonical_wham_output.pkl",
    "corrected_smpl_pkl": "output/demo/_lower_body_optimized/.../corrected_smpl.pkl"
  },
  "quality": {
    "usable_for_training": true,
    "quality_tier": "A",
    "ground_confidence": "high",
    "beta_variation_after_max_abs": 0.0,
    "foot_penetration_max_cm_after": 0.8,
    "stance_sliding_max_cm_after": 2.1
  }
}
```

ASI-PPO 只需要知道 manifest 的路径，其余都从 manifest 解析。

---

## 4. 坐标系约定

### 4.1 关键原则

坐标系必须由 Optimized-WHAM 在导出时统一好。ASI-PPO 只做验证，不应该重复猜测或重复转换。

推荐合同：

```text
Optimized-WHAM 内部可以使用 WHAM Y-up；
reference bundle 对外统一导出为 AMASS/MuJoCo Z-up；
manifest 中 coordinate_system 必须写 amass_zup；
ASI-PPO 如果读到非 amass_zup，默认报错。
```

### 4.2 为什么要这样

当前 WHAM / SMPL world 常见是 Y-up，而 AMASS / MuJoCo / MuscleMimic 常见是 Z-up。如果重复转换或忘记转换，会出现：

```text
人躺倒；
地面方向错误；
foot contact reward 锁错轴；
root height reward 追错轴；
PPO 学不到东西。
```

### 4.3 转换规则

WHAM Y-up 到 AMASS Z-up 可以使用 Rx(+90deg)：

```python
R_align = Rotation.from_euler("x", 90.0, degrees=True)
new_root_R = R_align * Rotation.from_rotvec(root_aa)
new_trans = trans @ R_align.as_matrix().T
```

但注意：这个转换只能做一次。

### 4.4 manifest 中必须记录

```json
{
  "source_coordinate_system": "wham_yup",
  "coordinate_system": "amass_zup",
  "axis_conversion": "Rx(+90deg)",
  "up_axis": "z"
}
```

### 4.5 ASI-PPO 侧验证

```python
if manifest["coordinate_system"] != "amass_zup":
    raise ValueError("ASI-PPO expects amass_zup reference bundles. Re-export from Optimized-WHAM.")
```

---

## 5. 时间和 FPS 约定

### 5.1 Optimized-WHAM 输出

`motion.npz` 和 `contact_schedule.npz` 必须有相同帧数：

```text
motion poses: [T, D]
contact_confidence: [T, K]
stance_mask: [T, K]
```

manifest 中：

```json
{
  "num_frames": 216,
  "fps": 60.0
}
```

### 5.2 ASI-PPO 消费

PPO control dt 可能不是 `1 / fps`。因此 ASI-PPO 需要 ReferencePhaseManager 处理：

```text
reference fps -> simulation control dt
```

例如：

```text
reference fps = 60
control_dt = 1/30
每个 control step 对应 reference 前进 2 帧
```

如果不能整除，应插值：

```text
root position：linear interpolation；
quaternion：slerp；
joint position：linear 或角度插值；
contact mask：nearest neighbor；
contact confidence：linear 或 nearest，建议 nearest。
```

### 5.3 常见问题

```text
reference fps 写错会导致动作快慢错误；
contact schedule 不重采样会错位；
PPO 以为脚该接触时，reference 其实已经 swing；
动作复现误差异常大。
```

解决：

```text
ASI-PPO 加载时打印 effective_ref_stride；
evaluation 报告 time_alignment_error；
所有 rollout 存 reference frame index。
```

---

## 6. 身体映射关系

### 6.1 Optimized-WHAM 输出 body graph

`body_graph.json`：

```json
{
  "version": "body_graph_v1",
  "keypoints": [
    "pelvis",
    "thorax",
    "head",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_toe",
    "left_heel",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_toe",
    "right_heel",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist"
  ],
  "edges": [
    ["pelvis", "thorax"],
    ["pelvis", "left_hip"],
    ["left_hip", "left_knee"],
    ["left_knee", "left_ankle"]
  ],
  "default_weights": {
    "pelvis": 1.0,
    "thorax": 1.0,
    "head": 0.5,
    "left_wrist": 0.2,
    "right_wrist": 0.2
  }
}
```

### 6.2 ASI-PPO 需要 body mapping

ASI-PPO 侧新增：

```text
configs/body_mapping/smpl_to_myo_fullbody.yaml
```

示例：

```yaml
pelvis:
  myo_body: pelvis
  weight: 1.0
thorax:
  myo_body: torso
  weight: 1.0
left_hip:
  myo_body: femur_l
  weight: 0.8
left_knee:
  myo_body: tibia_l
  weight: 0.8
left_ankle:
  myo_body: talus_l
  weight: 1.0
left_toe:
  myo_site: toe_l
  weight: 1.0
left_heel:
  myo_site: heel_l
  weight: 1.0
right_toe:
  myo_site: toe_r
  weight: 1.0
right_heel:
  myo_site: heel_r
  weight: 1.0
left_wrist:
  myo_body: ulna_l
  weight: 0.2
right_wrist:
  myo_body: ulna_r
  weight: 0.2
```

注意：手部不做，不等于上肢完全不管。建议：

```text
肩/肘中等权重；
腕低权重；
手指权重 0；
不使用手掌/手指 site。
```

---

## 7. 质量筛选闭环

### 7.1 Optimized-WHAM 输出质量

Optimized-WHAM 必须输出：

```json
{
  "quality_tier": "A",
  "usable_for_training": true,
  "foot_penetration_max_cm_after": 0.8,
  "stance_sliding_max_cm_after": 2.1,
  "root_delta_y_max_cm": 8.2,
  "root_delta_xz_max_cm": 0.1,
  "contact_switch_rate": 0.08,
  "body_laplacian_delta_mean_cm": 1.5
}
```

### 7.2 ASI-PPO 训练集构建

ASI-PPO 不直接扫所有 bundle，而是通过脚本筛选：

```bash
python BadmintonMimic/scripts/build_contact_tracking_manifest.py \
  --reference-root output/demo/_reference_bundles \
  --out data/reference_manifests/train_AB.jsonl \
  --include-tiers A,B
```

### 7.3 ASI-PPO 训练后反馈

ASI-PPO evaluation 输出：

```json
{
  "sequence": "5月1日-1",
  "success_rate": 0.35,
  "mean_body_error_cm": 12.8,
  "mean_foot_sliding_cm_s": 9.4,
  "failure_frames": [48, 51, 52],
  "failure_reason": "stance_contact_tracking_failed"
}
```

这些信息可以回写到一个单独文件：

```text
output/demo/_reference_bundles/<sequence>/asi_feedback.json
```

### 7.4 反馈如何用

如果某个 clip 的 Optimized-WHAM quality 看起来不错，但 ASI-PPO 总是失败，说明可能有以下问题：

```text
SMPL reference 对肌骨模型不可行；
body mapping 错；
contact schedule 错；
动作太难，需要 curriculum；
PPO reward 权重不合理；
reference fps / 坐标系错。
```

下一轮可以：

```text
把该 clip 降级为 C；
加入 hard_clips stage；
检查 failure frame 的 stance anchors；
检查 body graph error；
检查 retarget_cache 是否有大误差。
```

---

## 8. 端到端命令建议

### 8.1 从视频生成 reference bundle

在 Optimized-WHAM：

```powershell
python scripts\video_to_fixed_smpl_to_opensim.py `
  --video examples\5月1日-1.mp4 `
  --output-pth output\demo `
  --device cuda `
  --fps 60 `
  --world-grounded `
  --optimize-lower-body `
  --contact-preserving `
  --export-reference-bundle `
  --skip-ik
```

输出：

```text
output/demo/_reference_bundles/<sequence_safe>/manifest.json
```

### 8.2 构建 ASI-PPO 训练 manifest

在 ASI-PPO 仓库：

```powershell
python BadmintonMimic\scripts\build_contact_tracking_manifest.py `
  --reference-root D:\path\to\optimized_wham\output\demo\_reference_bundles `
  --out data\reference_manifests\train_AB.jsonl `
  --include-tiers A,B `
  --min-frames 60
```

### 8.3 生成 retarget cache

```powershell
python BadmintonMimic\scripts\build_tracking_reference_cache.py `
  --manifest data\reference_manifests\train_AB.jsonl `
  --body-mapping configs\body_mapping\smpl_to_myo_fullbody.yaml `
  --out data\reference_cache
```

### 8.4 训练 PPO

```powershell
python BadmintonMimic\scripts\train_asi_ppo.py `
  --config configs\asi_ppo\contact_preserving_tracking.yaml `
  --reference-manifest data\reference_manifests\train_AB.jsonl `
  --reference-cache data\reference_cache `
  --run-name contact_tracking_v1
```

### 8.5 评估

```powershell
python BadmintonMimic\scripts\evaluate_contact_tracking.py `
  --checkpoint runs\contact_tracking_v1\checkpoints\latest.pt `
  --manifest data\reference_manifests\val_A.jsonl `
  --out runs\contact_tracking_v1\eval_reports\val_A.json
```

---

## 9. 两边的版本管理

### 9.1 Reference bundle version

每次数据合同改变都要升级版本：

```text
contact_reference_bundle_v1
contact_reference_bundle_v2
```

ASI-PPO loader 应显式支持版本：

```python
SUPPORTED_BUNDLE_VERSIONS = {"contact_reference_bundle_v1"}
```

如果版本不支持，报错。

### 9.2 processing_report.json

Optimized-WHAM 应记录：

```json
{
  "optimized_wham_commit": "abc123",
  "command": "python scripts/video_to_fixed_smpl_to_opensim.py ...",
  "stages": [
    "wham",
    "fixed_beta",
    "world_grounded",
    "lower_body",
    "contact_preserving_export"
  ],
  "parameters": {
    "stance_enter_threshold": 0.55,
    "stance_exit_threshold": 0.30,
    "root_smooth_axes": ["y"]
  }
}
```

ASI-PPO 训练日志也要记录：

```json
{
  "asi_ppo_commit": "def456",
  "reference_bundle_version": "contact_reference_bundle_v1",
  "reference_manifest": "train_AB.jsonl",
  "reward_config": "contact_preserving_tracking.yaml"
}
```

这样以后才能复现实验。

---

## 10. 两边都要做的验证

### 10.1 Optimized-WHAM 导出前验证

```text
motion.npz 存在；
contact_schedule.npz 存在；
manifest.json 存在；
num_frames 一致；
coordinate_system == amass_zup；
quality_tier 存在；
beta_variation_after_max_abs == 0；
root_delta_xz 默认很小；
foot_penetration_after 不变差；
stance_sliding_after 不变差。
```

### 10.2 ASI-PPO 加载前验证

```text
manifest version 支持；
坐标系是 amass_zup；
fps 合法；
frame 数 > min_frames；
quality.usable_for_training true，除非允许低质量；
contact frame 数与 motion frame 数一致；
body_mapping 覆盖必要 keypoints。
```

### 10.3 ASI-PPO 训练中验证

每个 rollout 记录：

```text
reference frame index；
root error；
body error；
foot contact error；
termination reason。
```

### 10.4 ASI-PPO 训练后验证

```text
mean root tracking error；
mean body tracking error；
foot sliding；
foot penetration；
episode success rate；
failure frame distribution。
```

---

## 11. 最容易出问题的连接点

### 11.1 坐标系重复转换

症状：

```text
模型躺倒；
脚接触方向错误；
root height 完全不对。
```

预防：

```text
manifest 强制 coordinate_system；
ASI-PPO 只接受 amass_zup；
禁止训练脚本自己再次调用 Y-up -> Z-up。
```

### 11.2 contact 和 motion frame 错位

症状：

```text
脚明明在空中，reward 要求它贴地；
脚明明支撑，reward 不锁脚。
```

预防：

```text
motion.npz 和 contact_schedule.npz 都保存 frame_ids；
加载时校验 frame_ids 一致；
resampling 时同步处理 contact。
```

### 11.3 foot label 顺序不一致

症状：

```text
左脚 contact 被用到右脚；
toe/heel 混乱；
foot lock reward 反而让脚滑。
```

预防：

```text
contact_schedule.npz 保存 foot_labels；
ASI-PPO body_mapping 显式匹配 label；
不要依赖数组顺序。
```

### 11.4 Optimized-WHAM 修正过度

症状：

```text
reference 很干净，但不像原视频；
PPO 复现了错误动作；
评估原始视频误差变大。
```

预防：

```text
quality_report 记录 original_vs_corrected error；
pose_delta / root_delta 超阈值降级；
保留原始和修正后 reference 供对比。
```

### 11.5 ASI-PPO body mapping 错

症状：

```text
reward 很低；
某些 body error 极大；
模型出现奇怪姿态。
```

预防：

```text
build_tracking_reference_cache.py 输出 retarget_report；
可视化 body target 和模型 body；
每个 keypoint 单独报告平均误差。
```

### 11.6 reference 对肌骨模型不可行

症状：

```text
Optimized-WHAM 看起来没问题，但 PPO 无论如何学不好；
需要巨大肌肉激活才能接近；
episode 经常摔倒。
```

预防：

```text
在 ASI-PPO 中增加 retarget feasibility report；
报告 joint limit violation；
报告 required velocity；
把不可行片段降级为 C/D。
```

---

## 12. 推荐开发里程碑

### Milestone 1：统一 reference bundle

Optimized-WHAM 输出：

```text
manifest.json
motion.npz
contact_schedule.npz
quality_report.json
```

ASI-PPO 能读取 manifest 并打印摘要。

验收：

```text
单个 clip 能从 Optimized-WHAM 输出，并被 ASI-PPO loader 正确读取。
```

### Milestone 2：quality-based manifest builder

ASI-PPO 可以根据质量分级构建训练集。

验收：

```text
train_A.jsonl / train_AB.jsonl 正确生成；
低质量 clips 被排除。
```

### Milestone 3：contact-aware reward

ASI-PPO 使用 contact schedule 做 foot contact reward。

验收：

```text
训练后 foot sliding 比无 contact reward 下降；
body tracking 不明显变差。
```

### Milestone 4：body graph reward

ASI-PPO 加入 Laplacian body graph reward。

验收：

```text
姿态整体协调性提高；
局部 body tracking error 没有明显恶化；
肌肉 effort 不明显增加。
```

### Milestone 5：feedback loop

ASI-PPO evaluation 输出 failure diagnostics，Optimized-WHAM 可以读取或人工查看。

验收：

```text
失败 clips 能被定位到具体 frame 和原因；
下一轮数据筛选能排除或降级这些 clips。
```

---

## 13. 最终端到端目标

最终两个项目之间应该形成这样一条稳定链路：

```text
1. Optimized-WHAM 从视频生成高质量 reference bundle
   - 地面一致
   - fixed beta
   - 支撑脚稳定
   - 穿地少
   - 结构保持
   - 有质量分级

2. ASI-PPO 读取 reference bundle
   - 不猜坐标系
   - 不自己修 contact
   - 不处理视频噪声
   - 只做肌骨轨迹复现训练

3. ASI-PPO 输出 tracking diagnostics
   - root/body/joint/contact/muscle 指标
   - failure frame
   - failure reason

4. 诊断结果反馈给 Optimized-WHAM
   - 调整 quality gate
   - 调整 stance anchor
   - 排除坏 clips
   - 改善下一批 reference
```

一句话总结：

```text
Optimized-WHAM 是 reference 生产者；
ASI-PPO 是 reference 消费者；
两者通过 manifest + motion/contact/body_graph/quality 的数据合同连接；
所有视频噪声、地面、接触、尺度问题尽量在 Optimized-WHAM 解决；
所有肌骨控制、reward、policy、evaluation 问题在 ASI-PPO 解决。
```

这就是在你当前目标下最合理、最工程可落地的系统分工。
