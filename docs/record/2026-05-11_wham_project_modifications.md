# WHAM 基础上的项目修正与改进记录

记录时间：2026-05-11  
项目目录：`D:\Main\Research\IEProgram\Bad_Muskeleton\WorldSMPLGen\WHAM`

## 目标

原始 WHAM 主要输出视频对应的 SMPL 时序结果，但在本项目里需要继续服务于 OpenSim/MimicMSK 肌骨模型。因此，当前项目在原始 WHAM 基础上做了几类扩展：

1. 让 WHAM 输出的 SMPL 序列具有固定 beta。
2. 把 WHAM/SMPL 序列重定向到 MimicMSK OpenSim 模型。
3. 使用 WHAM 的接触概率和脚点信息估计地面，并做 world-grounded root 修正。
4. 将估计地面归一到 OpenSim 的 `Y=0`，避免模型整体落在 OpenSim 地面以下。
5. 提供一条从视频到 fixed-beta SMPL，再到 OpenSim IK motion 的自动流水线。

## 原始 WHAM 的主要问题

在当前任务目标下，原始 WHAM 有几个不足：

- `betas` 是逐帧输出，人体形状会随时间漂移，不适合作为同一个人的肌骨重定向输入。
- 原始 WHAM 输出没有直接保存后续 ground/contact 优化需要的接触概率和 refined 脚点。
- WHAM 坐标系里的地面高度不等于 OpenSim 可视化和 IK 使用的 `Y=0`。
- WHAM 可能出现 track id 切换，直接取单个 track 会截断动作。
- OpenSim retarget 需要固定的 17 个对应点、TRC 文件、IK setup、root MOT 等中间文件，原始 WHAM 没有这条链路。

## 已完成的核心改动

### 1. WHAM 输出增加 contact 和 feet 信息

修改文件：

- `demo.py`
- `lib/models/wham.py`

新增保存字段：

- `contact`: WHAM 预测的脚部接触概率，形状通常为 `[T, 4]`。
- `feet_world`: 世界坐标脚点。
- `feet_refined`: trajectory refinement 之后重新计算的脚点。

同时，`lib/models/wham.py` 中的 refined feet 现在不仅训练阶段计算，推理阶段也会输出。这样后续地面估计不会使用 refinement 前的旧脚点。

### 2. 固定 beta canonicalization

新增/修改文件：

- `scripts/canonicalize_wham_fixed_beta.py`

功能：

- 从 WHAM 原始 `wham_output.pkl` 中估计一个 clip-level fixed beta。
- 把每一帧的 `betas` 替换为同一个固定 beta。
- 重新计算或校正相关 SMPL 输出，减少 fixed beta 后的骨架/mesh 不一致。
- 保留并裁剪 `contact`、`feet_world`、`feet_refined` 等时序字段。
- 输出：
  - `canonical_wham_output.pkl`
  - `beta_fixed.npy`
  - `fixed_beta_report.json`

验证指标：

```text
beta_variation_after_max_abs: 0
```

这表示 fixed-beta 后，每一帧 beta 完全一致。

### 3. Track merge 工具

新增文件：

- `lib/world_grounded/tracks.py`

功能：

- 支持 `--track-id merge`，把 WHAM 中可能发生的 ID switch 拼接成一个连续 timeline。
- 支持 `--track-id longest` 和指定 track id。
- merge 时保留关键时序字段：
  - `pose`
  - `trans_world`
  - `betas`
  - `contact`
  - `feet_world`
  - `feet_refined`
  - `verts`
- 避免把静态字段误当成逐帧字段，尤其避免 1D `betas` 被错误切成标量。

当前默认流水线使用：

```text
--track-id merge
```

### 4. SMPL 到 OpenSim 的 17 点 retarget

新增/修改文件：

- `configs/retarget/smpl_to_mimicmsk_opensim.yaml`
- `scripts/retarget_smpl_to_opensim.py`
- `scripts/validate_smpl_opensim_retarget.py`

功能：

- 使用 MuscleMimic/MyoFullBody 参考方式中的 17 个标准点。
- 把 SMPL 关节点映射到 OpenSim marker 或 virtual marker。
- 生成：
  - `smpl_17pt_markers.trc`
  - `wham_fixed_root.mot`
  - `smpl_to_opensim_ik_setup.xml`
  - 添加 virtual markers 后的 `.osim` 模型
  - OpenSim IK 输出 `opensim_ik.mot`

当前默认策略：

- 默认 `--free-root`，不强行锁死 OpenSim root。
- 可选 `--fix-root`，但之前实测 fixed root 容易导致 IK marker error 很大，因此不作为默认。

### 5. OpenSim 几何资源处理

在 retarget 阶段增加了 Geometry 资源复制/组织逻辑，解决之前 OpenSim 大量提示：

```text
Couldn't find file 'Geometry/xxx.stl'
```

的情况。当前目标是让生成的 retarget 输出目录能尽量自包含，OpenSim 加载模型时不再因为相对 Geometry 路径缺失而产生大量 warning。

### 6. World-grounded SMPL optimizer

新增文件：

- `lib/world_grounded/ground.py`
- `lib/world_grounded/foot_points.py`
- `lib/world_grounded/root_optimizer.py`
- `lib/world_grounded/reports.py`
- `scripts/world_grounded_smpl_optimizer.py`

核心输入：

- fixed-beta 后的 `canonical_wham_output.pkl`
- WHAM contact 概率
- WHAM refined feet/world feet

核心输出：

- `optimized_canonical_wham_output.pkl`
- `ground_plane.json`
- `contact_report.json`
- `quality_report.json`
- `optimization_report.json`

地面估计逻辑：

1. 优先使用 `feet_refined`，其次 `feet_world`、`feet`，最后才 fallback 到 mesh 低点。
2. 使用 WHAM `contact` 作为接触概率。
3. 同时计算脚点水平速度和垂直速度，低速脚点权重更高。
4. 用加权中位数估计地面高度 `ground_y`。
5. 再根据接近地面、速度低、contact 高等条件计算 `contact_confidence`。
6. 对稀疏 contact、无 contact、异常 outlier、NaN/Inf、非法 fps 等情况做了防护。

当前示例结果：

```text
ground_y: -0.87002873
ground_confidence: high
contact_source: wham_contact
foot_source: feet_refined
```

### 7. Root translation 平滑和穿地修正

`lib/world_grounded/root_optimizer.py` 对 `trans_world` 做两件事：

1. 平滑 root translation，降低 root 抖动。
2. 根据 contact foot 与估计地面的关系，向上修正穿地。

示例质量指标：

```text
root_acceleration_before: 3.4529
root_acceleration_after:  1.7108

foot_penetration_mean_cm_before: 0.7558
foot_penetration_mean_cm_after:  0.1248

foot_penetration_max_cm_before: 3.1595
foot_penetration_max_cm_after:  2.4897
```

这说明当前版本主要改善了 root 平滑和平均穿地，但最大穿地仍有残余，需要后续 foot locking / lower-body correction。

### 8. 地面对齐到 OpenSim Y=0

最新修正：

- `scripts/world_grounded_smpl_optimizer.py`
- `tests/world_grounded/test_world_grounded_optimizer.py`

原先 world-grounded 只保证脚相对 WHAM 估计地面不穿地，但 WHAM 地面可能是：

```text
ground_y = -0.87002873
```

这会导致导入 OpenSim 后，人体整体仍然落在 OpenSim 的 `Y=0` 地面以下。

现在默认会执行：

```text
ground_alignment_offset_y = -ground_y
```

即把整段轨迹沿 Y 方向平移，使：

```text
ground_y_after_alignment = 0.0
```

示例：

```text
ground_y: -0.87002873
ground_alignment_offset_y: 0.87002873
ground_y_after_alignment: 0.0
```

也就是把整个人、脚点、mesh、`trans_world` 统一上移约 `0.87m`，让 WHAM 估计地面和 OpenSim 地面一致。

如果需要保留 WHAM 原始世界高度，可以给 optimizer 传：

```text
--no-align-ground-to-zero
```

但当前主流水线默认会对齐到 OpenSim `Y=0`。

### 9. 一键流水线脚本

新增/修改文件：

- `scripts/video_to_fixed_smpl_to_opensim.py`

完整流程：

```text
video
  -> WHAM demo.py
  -> wham_output.pkl
  -> canonicalize fixed beta
  -> canonical_wham_output.pkl
  -> world-grounded optimizer
  -> optimized_canonical_wham_output.pkl
  -> SMPL-to-OpenSim retarget
  -> OpenSim IK opensim_ik.mot
```

常用命令：

```powershell
python scripts\video_to_fixed_smpl_to_opensim.py `
  --video D:\Main\Research\IEProgram\Bad_Muskeleton\WorldSMPLGen\WHAM\examples\5月1日-2.mp4 `
  --output-pth output\demo `
  --device cuda `
  --fps 60 `
  --world-grounded
```

如果已经有 WHAM 和 fixed-beta 输出，只想重新跑 world-grounded 与 retarget：

```powershell
python scripts\video_to_fixed_smpl_to_opensim.py `
  --video D:\Main\Research\IEProgram\Bad_Muskeleton\WorldSMPLGen\WHAM\examples\5月1日-2.mp4 `
  --output-pth output\demo `
  --device cuda `
  --fps 60 `
  --world-grounded `
  --skip-wham `
  --skip-fixed-beta
```

建议调试时指定新输出目录，避免误看旧结果：

```powershell
python scripts\video_to_fixed_smpl_to_opensim.py `
  --video D:\Main\Research\IEProgram\Bad_Muskeleton\WorldSMPLGen\WHAM\examples\5月1日-2.mp4 `
  --output-pth output\demo `
  --device cuda `
  --fps 60 `
  --world-grounded `
  --skip-wham `
  --skip-fixed-beta `
  --world-grounded-out-dir output\demo\_world_grounded\5_1_-2_align0 `
  --retarget-out-dir output\demo\_opensim_retarget_fixed_beta\5_1_-2_align0
```

## 当前验证情况

当前已有测试：

- `tests/world_grounded/test_canonical_fields.py`
- `tests/world_grounded/test_tracks.py`
- `tests/world_grounded/test_ground.py`
- `tests/world_grounded/test_root_optimizer.py`
- `tests/world_grounded/test_world_grounded_optimizer.py`

最近验证结果：

```text
tests/world_grounded: 35 passed
py_compile: passed
```

真实数据 `5月1日-2.mp4` 中，world-grounded 输出显示：

```text
frames: 202
beta_variation_after_max_abs: 0
ground_y: -0.87002873
ground_alignment_offset_y: 0.87002873
ground_y_after_alignment: 0.0
```

用地面对齐后的 pkl 生成 TRC 后，OpenSim marker 高度大致为：

```text
ankle_l: 0.040 ~ 0.156 m
LTOE:   -0.006 ~ 0.058 m
ankle_r: 0.035 ~ 0.107 m
RTOE:   -0.011 ~ 0.048 m
pelvis: 0.927 ~ 0.973 m
```

这说明人体已经不再整体落在 OpenSim 地面下方，只剩脚尖级别的轻微残余穿地。

## 当前仍存在的问题

1. 最大穿地还没有完全消除  
   当前 root-level 修正能明显降低平均穿地，但最大穿地仍可能在 1-3 cm 量级。

2. 下肢局部动作仍需要优化  
   当前版本主要优化 root translation 和地面对齐，没有直接优化膝、踝、趾关节。

3. OpenSim IK 不包含真实接触约束  
   OpenSim IK 目前仍是 marker fitting，没有加入 foot-ground contact constraint，所以脚尖仍可能轻微穿地或滑动。

4. `--free-root` 虽然稳定，但不保证绝对 root 轨迹被完全遵循  
   当前默认 `--free-root` 是为了避免 fixed-root 导致 IK 误差大，但这也意味着 OpenSim 会自己拟合 root。

5. 仍需视频可视化评估  
   数值指标显示改进，但最终动作是否自然，还需要并排视频或 OpenSim 可视化检查。

## 后续建议

下一阶段不建议继续大幅移动 root，而应该进入更局部的下肢优化：

1. 加 foot locking：
   - 对高 contact confidence 的脚点减少滑动。
   - 对脚尖/脚跟穿地做局部 correction。

2. 加 lower-body pose optimizer：
   - 优化 hip/knee/ankle/toe 对应角度。
   - 保持 root 基本不变。
   - 用 OpenSim 可行关节范围作为约束。

3. 加地面接触约束：
   - contact 脚不能低于 OpenSim `Y=0`。
   - contact 脚水平速度尽量小。

4. 增加对比可视化：
   - raw WHAM
   - fixed-beta
   - world-grounded
   - OpenSim retarget

## 总结

当前项目已经从“原始 WHAM 视频到 SMPL”扩展为：

```text
视频
  -> WHAM SMPL
  -> fixed-beta SMPL
  -> contact-aware world-grounded SMPL
  -> OpenSim 17 点 retarget
  -> OpenSim IK motion
```

最重要的改进是：

- beta 已固定。
- track 可 merge。
- WHAM contact 和 refined feet 被保存并用于地面估计。
- root translation 已平滑。
- 平均脚部穿地显著降低。
- WHAM 地面已默认对齐到 OpenSim `Y=0`。
- 一键流水线已经可以从视频直接生成 OpenSim IK motion。

当前还没完成的是更精细的 foot locking 和 lower-body pose/contact 优化，这是下一阶段 `world_grounded_smpl optimizer` 应该继续推进的方向。

## Lower-Body SMPL Optimizer

Added a lower-body SMPL correction stage after fixed-beta canonicalization and world grounding. The stage outputs `corrected_smpl.pkl`, keeps beta fixed, preserves frame count, reports foot penetration and sliding metrics, and can run before OpenSim/MuJoCo retargeting.

The intended full pipeline is:

```text
WHAM -> fixed beta -> world grounded -> lower-body corrected SMPL -> OpenSim/MuJoCo retarget
```

OpenSim remains an evaluator and feedback source rather than the final product.

Current recommended command:

```text
python scripts\video_to_fixed_smpl_to_opensim.py --video <video> --output-pth output\demo --device cuda --fps <fps> --world-grounded --optimize-lower-body
```

`--optimize-lower-body` now requires `--world-grounded`.
