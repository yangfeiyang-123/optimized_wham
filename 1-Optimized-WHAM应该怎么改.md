# 1. Optimized-WHAM 应该怎么改

> 目标：在**不加入手部动作、不加入球拍、不建模羽毛球**的前提下，把 Optimized-WHAM 从“视频 SMPL 恢复与地面修正工具”升级成一个 **Contact-Preserving / Body-Structure-Preserving Reference Generator**。它的任务不是做最终控制，而是为 ASI-PPO / MuscleMimic 提供更干净、更稳定、更符合肌骨模型可实现性的参考轨迹。

---

## 0. 总体定位

你当前真正需要的不是完整 OmniRetarget 的 robot-object-terrain 交互，也不是球拍、羽毛球、击球点建模。你要借鉴的是 OmniRetarget 的这条核心思路：

```text
低质量视频 SMPL 轨迹
  -> 先通过接触、身体结构、运动学约束优化成高质量 reference
  -> 再交给下游 PPO / MuscleMimic 学习
```

因此，Optimized-WHAM 应该承担以下职责：

```text
视频 / WHAM 输出
  -> 固定 beta
  -> 估计世界地面
  -> 修正 root 高度和穿地
  -> 识别 foot contact / stance segment
  -> 对支撑脚做 near-hard locking
  -> 优化 lower-body pose 和 root，使脚底接触稳定
  -> 保持身体结构和原始动作相似
  -> 导出 ASI-PPO 可直接消费的 reference bundle
```

它不应该承担：

```text
不做手部精细动作；
不建模球拍；
不建模羽毛球；
不做击球任务奖励；
不直接训练 PPO；
不强行把所有问题都放到 OpenSim IK 或 PPO 阶段解决。
```

---

## 1. 当前 Optimized-WHAM 的已有基础

根据你当前项目结构，Optimized-WHAM 已经有非常好的基础。当前 pipeline 大体是：

```text
WHAM 原始输出
  -> canonicalize_wham_fixed_beta.py
  -> world_grounded_smpl_optimizer.py
  -> optimize_smpl_lower_body.py，可选
  -> retarget_smpl_to_opensim.py
```

现有代码中已经有：

### 1.1 fixed beta

`video_to_fixed_smpl_to_opensim.py` 顶部说明里已经明确：OpenSim 阶段只消费 `canonical_wham_output.pkl`，避免使用 WHAM 原始逐帧变化的 beta。这个方向是正确的，因为肌骨模型复现轨迹时最怕身体尺度逐帧变化。

### 1.2 world-grounded optimizer

`world_grounded_smpl_optimizer.py` 当前已经会：

```text
读取 canonical WHAM pkl；
提取 foot points；
提取 contact；
估计 ground_y；
优化 trans_world；
把地面平移到 Y=0；
输出 optimized_canonical_wham_output.pkl；
输出 ground_plane.json、contact_report.json、quality_report.json。
```

这已经相当接近“视频 SMPL 清洗器”。

### 1.3 contact confidence 和 ground estimation

`lib/world_grounded/ground.py` 已经会基于：

```text
foot height；
horizontal foot speed；
vertical foot speed；
WHAM contact，可选；
median filtering；
weighted median ground estimation。
```

输出：

```text
ground_y
contact_confidence
sample_weights
contact_source
ground_confidence
```

这个模块应该继续保留，并作为后续 stance segment 的输入。

### 1.4 root translation optimizer

`lib/world_grounded/root_optimizer.py` 已经会：

```text
平滑 root translation；
根据 foot penetration 修正 root vertical；
输出 foot_penetration before/after；
输出 root_acceleration before/after；
输出 root_delta_mean/max。
```

这个模块需要改，因为它现在的平滑逻辑可能会影响 X/Z 轨迹复现，具体见后文。

### 1.5 lower-body optimizer

`lib/smpl_optimization/lower_body.py` 已经包含：

```text
contact_ground_loss
penetration_loss
foot_lock_loss
smoothness_loss
WHAM prior
root prior
lower body pose pass
```

这是最应该继续强化的模块。

---

## 2. 新的总体架构

建议把 Optimized-WHAM 的目标架构升级为：

```text
lib/world_grounded/
  ground.py                       # 已有：地面估计和 contact confidence
  root_optimizer.py               # 改造：默认只修 vertical，不轻易改 X/Z
  contact_segments.py             # 新增：从 contact confidence 提取 stance segments
  stance_anchor.py                # 新增：为每个 stance segment 建立 foot anchor
  quality_gates.py                # 新增：训练数据质量筛选规则
  reference_bundle.py             # 新增：导出 ASI-PPO 消费的数据合同

lib/smpl_optimization/
  lower_body.py                   # 改造：加入 stance anchor loss / swing clearance
  losses.py                       # 改造：加入 Huber、segment anchor、velocity limit loss
  body_graph.py                   # 新增：人体结构图和 Laplacian body loss / metrics
  metrics.py                      # 改造：增加 stance sliding、body graph error 等指标

scripts/
  world_grounded_smpl_optimizer.py
  optimize_smpl_lower_body.py
  export_contact_preserving_reference.py  # 新增或并入现有 pipeline
  video_to_fixed_smpl_to_opensim.py       # 改造：串联新阶段
```

推荐 pipeline：

```text
python scripts/video_to_fixed_smpl_to_opensim.py \
  --video xxx.mp4 \
  --output-pth output/demo \
  --fps 60 \
  --world-grounded \
  --optimize-lower-body \
  --contact-preserving \
  --export-reference-bundle \
  --skip-ik
```

最终输出：

```text
output/demo/_reference_bundles/<sequence>/
  reference_motion.npz
  contact_schedule.npz
  body_graph.json
  quality_report.json
  processing_report.json
  manifest.json
```

---

## 3. 最重要改动一：root_optimizer 默认只修 Y，不要默认平滑 X/Z

### 3.1 问题

你现在的目标是“尽可能复现轨迹”。那么 root 的 X/Z 平面轨迹非常重要。如果 root optimizer 对 `trans_world[:, 0]` 和 `trans_world[:, 2]` 做 moving average，可能会：

```text
改变真实水平位移；
降低轨迹复现精度；
把快速步伐或身体移动抹平；
导致下游 PPO 学到的 root tracking 与原视频不一致；
让评估时原始轨迹误差反而变大。
```

当前 root optimizer 中：

```python
smoothed_trans = _safe_savgol(trans_world, smooth_window)
delta = smoothed_trans - trans_world
```

这会对 XYZ 全部生效。对你的目标来说，默认行为应该改成：

```text
默认只修 vertical，也就是 Y-up 下的 Y；
X/Z 只在显式开启时做轻微平滑；
并且 X/Z 平滑要有限幅。
```

### 3.2 新增配置

修改 `RootOptimizationConfig` 或给 `optimize_root_translation()` 添加参数：

```python
@dataclass
class RootOptimizerConfig:
    fps: float
    ground_y: float = 0.0
    smooth_window: int = 9
    contact_threshold: float = 0.35
    smooth_axes: tuple[str, ...] = ("y",)  # 默认只修 Y
    max_xz_delta: float = 0.03             # 米，默认最多改 3cm
    max_y_delta: float = 0.30              # 米，根据视频质量可调
    preserve_horizontal: bool = True
```

也可以保持函数式接口：

```python
def optimize_root_translation(
    trans_world,
    foot_points,
    contact_confidence,
    ground_y,
    fps,
    smooth_window=9,
    contact_threshold=0.35,
    smooth_axes=("y",),
    max_xz_delta=0.03,
    max_y_delta=0.30,
):
    ...
```

### 3.3 实现策略

```python
smoothed = _safe_savgol(trans_world, smooth_window)
delta = np.zeros_like(trans_world)

if "x" in smooth_axes:
    delta[:, 0] = np.clip(smoothed[:, 0] - trans_world[:, 0], -max_xz_delta, max_xz_delta)
if "y" in smooth_axes:
    delta[:, 1] = np.clip(smoothed[:, 1] - trans_world[:, 1], -max_y_delta, max_y_delta)
if "z" in smooth_axes:
    delta[:, 2] = np.clip(smoothed[:, 2] - trans_world[:, 2], -max_xz_delta, max_xz_delta)
```

然后再叠加 penetration correction：

```python
delta[:, 1] += correction_y
```

最后仍然限制总 Y 改变量：

```python
delta[:, 1] = np.clip(delta[:, 1], -max_y_delta, max_y_delta)
```

### 3.4 新增报告字段

`quality_report.json` 里必须增加：

```json
{
  "root_delta_x_mean_cm": 0.0,
  "root_delta_x_max_cm": 0.0,
  "root_delta_y_mean_cm": 1.7,
  "root_delta_y_max_cm": 8.2,
  "root_delta_z_mean_cm": 0.0,
  "root_delta_z_max_cm": 0.0,
  "root_smooth_axes": ["y"],
  "horizontal_root_preserved": true
}
```

### 3.5 验收标准

```text
默认情况下 root X/Z 改变量应接近 0；
foot_penetration_max_cm_after <= before；
root_delta_y_max_cm 不超过 max_y_delta；
轨迹复现任务中，水平 root 误差不应因为预处理变大。
```

### 3.6 可能问题

```text
如果原始 WHAM X/Z 本身有明显抖动，不修 X/Z 会把抖动传给 PPO；
如果修 X/Z，又可能改变原始轨迹。
```

建议策略：

```text
默认不修 X/Z；
只在 quality report 中记录 root_xz_acceleration；
如果某个 clip 的 root_xz_acceleration 过大，再手动或通过 quality gate 标记为 low quality；
不要默认自动平滑所有水平运动。
```

---

## 4. 最重要改动二：新增 contact_segments.py

### 4.1 目的

OmniRetarget 最值得你借鉴的是 foot sticking。当前 `foot_lock_loss` 是相邻帧速度软惩罚，强度不够。要先把 contact confidence 转成稳定的 stance segments。

### 4.2 文件位置

新增：

```text
lib/world_grounded/contact_segments.py
```

### 4.3 数据结构

```python
from dataclasses import dataclass

@dataclass
class StanceSegment:
    foot_index: int
    label: str
    start: int          # inclusive
    end: int            # exclusive
    confidence_mean: float
    confidence_min: float
    length: int
```

### 4.4 输入输出

输入：

```text
contact_confidence: np.ndarray, shape [T, K]
foot_labels: list[str]
```

输出：

```text
segments: list[StanceSegment]
stance_mask: np.ndarray, shape [T, K], bool
```

### 4.5 核心算法

不要直接 `contact_confidence > threshold`，应使用 hysteresis：

```text
enter_threshold = 0.55
exit_threshold  = 0.30
min_segment_len = 4 frames
merge_gap       = 2 frames
```

伪代码：

```python
def extract_stance_segments(conf, labels, enter=0.55, exit=0.30, min_len=4, merge_gap=2):
    segments = []
    for k in range(conf.shape[1]):
        active = False
        start = None
        for t in range(T):
            c = conf[t, k]
            if not active and c >= enter:
                active = True
                start = t
            elif active and c <= exit:
                end = t
                maybe_add_segment(start, end)
                active = False
        if active:
            maybe_add_segment(start, T)
    segments = remove_short_segments(segments, min_len)
    segments = merge_close_segments(segments, merge_gap)
    return segments
```

### 4.6 为什么要 hysteresis

WHAM contact 和视频估计足部速度会抖。如果用一个阈值，会出现：

```text
接触/离地频繁跳变；
stance segment 被切碎；
foot anchor 不稳定；
PPO 中 contact reward 也会抖。
```

hysteresis 可以避免这一点。

### 4.7 测试

新增：

```text
tests/world_grounded/test_contact_segments.py
```

测试用例：

```text
1. 连续高置信度应生成一个 segment；
2. 短暂低于 enter 但高于 exit 不应断开；
3. 小 gap 应合并；
4. 过短 segment 应删除；
5. 空 contact 应返回空 segments。
```

---

## 5. 最重要改动三：新增 stance_anchor.py，实现 near-hard foot locking

### 5.1 目的

把每个 stance segment 中的 toe/heel XZ 位置固定到一个 anchor，减少 foot skating。

这里不要直接粗暴改 SMPL 点，而是生成 anchor target，交给 lower-body optimizer 去满足。

### 5.2 文件位置

```text
lib/world_grounded/stance_anchor.py
```

### 5.3 数据结构

```python
@dataclass
class FootAnchor:
    foot_index: int
    label: str
    start: int
    end: int
    anchor_xyz: np.ndarray      # shape [3]
    anchor_xz: np.ndarray       # shape [2]
    ground_y: float
    confidence: float
    source: str                 # median / first_frame / weighted_median
```

### 5.4 Anchor 计算方式

推荐使用 weighted median，不要用首帧。原因是 WHAM 某一帧可能异常。

```python
def compute_anchor(points, confidence, segment, ground_y):
    p = points[segment.start:segment.end, segment.foot_index]  # [L, 3]
    w = confidence[segment.start:segment.end, segment.foot_index]
    x = weighted_median(p[:, 0], w)
    z = weighted_median(p[:, 2], w)
    y = ground_y
    return np.array([x, y, z], dtype=np.float32)
```

### 5.5 输出 sidecar

`stance_anchors.json`：

```json
{
  "coordinate_system": "wham_y_up",
  "ground_y": 0.0,
  "anchors": [
    {
      "foot_index": 0,
      "label": "left_heel_or_ankle",
      "start": 12,
      "end": 38,
      "anchor_xyz": [0.123, 0.0, 1.482],
      "confidence": 0.84
    }
  ]
}
```

### 5.6 质量指标

新增 metrics：

```text
stance_sliding_mean_cm_before
stance_sliding_max_cm_before
stance_sliding_mean_cm_after
stance_sliding_max_cm_after
stance_anchor_error_mean_cm_after
stance_anchor_error_max_cm_after
num_stance_segments
stance_coverage_ratio
```

---

## 6. 改造 lower_body.py：从 soft foot lock 升级到 segment anchor loss

### 6.1 当前问题

当前 loss 大致是：

```python
loss = (
    w_contact_ground * contact_ground_loss(...)
  + w_penetration * penetration_loss(...)
  + w_foot_lock * foot_lock_loss(...)
  + w_pose_smooth * smoothness_loss(...)
  + w_root_smooth * smoothness_loss(...)
  + w_wham_prior * prior
  + w_root_prior * root_prior
)
```

其中 `foot_lock_loss` 惩罚的是相邻帧 foot velocity。它有用，但对明显脚滑的视频不够强。

### 6.2 新增配置

在 `LowerBodyOptimizerConfig` 增加：

```python
use_stance_anchor_loss: bool = True
w_stance_anchor_xz: float = 80.0
w_stance_anchor_y: float = 60.0
w_swing_clearance: float = 3.0
min_swing_clearance: float = 0.02
stance_enter_threshold: float = 0.55
stance_exit_threshold: float = 0.30
stance_min_frames: int = 4
stance_merge_gap: int = 2
use_huber: bool = True
huber_delta: float = 0.03
max_lower_body_pose_delta: float = 0.7
max_root_y_shift: float = 0.25
```

### 6.3 新增 loss：stance_anchor_loss

文件：`lib/smpl_optimization/losses.py`

```python
def huber_l2(error: torch.Tensor, delta: float) -> torch.Tensor:
    abs_err = torch.abs(error)
    quad = torch.minimum(abs_err, torch.as_tensor(delta, device=error.device, dtype=error.dtype))
    lin = abs_err - quad
    return 0.5 * quad ** 2 + delta * lin


def stance_anchor_loss(points, anchor_targets, anchor_mask, huber_delta=0.03):
    # points: [T, K, 3]
    # anchor_targets: [T, K, 3]
    # anchor_mask: [T, K]
    error = points - anchor_targets
    loss_xyz = huber_l2(error, huber_delta)
    weight = anchor_mask.unsqueeze(-1)
    denom = torch.clamp(weight.sum() * 3.0, min=1.0)
    return (loss_xyz * weight).sum() / denom
```

也可以拆成 XZ 和 Y：

```python
def stance_anchor_xz_loss(points, target_xz, mask):
    error = points[..., [0, 2]] - target_xz
    ...


def stance_anchor_y_loss(points, ground_y, mask):
    error = points[..., 1] - ground_y
    ...
```

### 6.4 anchor target 构建

在 lower-body optimizer 中：

```python
segments = extract_stance_segments(contact_confidence, labels, ...)
anchors = compute_stance_anchors(foot_points, contact_confidence, segments, ground_y)
anchor_targets, anchor_mask = rasterize_anchors(anchors, n_frames, n_feet)
```

其中：

```text
anchor_targets: [T, K, 3]
anchor_mask: [T, K]
```

### 6.5 与 pose pass 的结合

在 `optimize_lower_body_pose_smpl()` 中，每个 chunk 做 forward 后得到 `foot`，然后加入：

```python
loss = loss + config.w_stance_anchor_xz * stance_anchor_xz_loss(
    foot,
    anchor_xz[start:end],
    anchor_mask[start:end],
)

loss = loss + config.w_stance_anchor_y * stance_anchor_y_loss(
    foot,
    config.ground_y,
    anchor_mask[start:end],
)
```

### 6.6 注意 chunk 边界

如果一个 stance segment 跨过 chunk 边界，anchor target 仍然应当全局一致。不要在每个 chunk 内重新计算 anchor，否则会造成 foot target 在边界跳变。

正确流程：

```text
整段序列先计算 contact segments；
整段序列先计算 anchors；
把 anchor_targets 切片传入每个 chunk。
```

### 6.7 可能问题

#### 问题 A：真实动作里脚可能有轻微旋转或滑移

有些动作支撑脚不是完全固定，尤其是转体时 toe/heel 可能发生 pivot。如果同时锁 toe 和 heel 的 XZ，可能过度约束。

解决：

```text
先只锁置信度最高的 foot point；
或分别锁 toe/heel，但权重降低；
允许 pivot：锁 foot center，放松 toe/heel 相对旋转；
对快速转身动作降低 w_stance_anchor_xz。
```

#### 问题 B：contact 检测错误

如果把 swing phase 错判成 stance，会把脚锁到空中或错误位置。

解决：

```text
contact_confidence 必须同时满足高度、速度、WHAM contact；
ground_confidence low 时不启用 hard stance lock；
contact segment 太短时删除；
anchor_error 优化后仍大则标记 low quality。
```

#### 问题 C：下肢姿态被优化得离原 WHAM 太远

解决：

```text
增加 WHAM prior；
限制 max_lower_body_pose_delta；
报告 pose_delta_report；
如果 lower_body_pose_delta 超阈值，该 clip 不进入训练集。
```

---

## 7. 新增 body_graph.py：保持人体结构关系

### 7.1 为什么需要 body graph

你的目标不是只让某几个 joint point 对齐，而是让肌骨模型复现整体动作。SMPL 和肌骨模型的骨段比例、joint 定义不完全一致，只做 keypoint L2 很容易出现：

```text
局部点对齐，但整体姿态变形；
pelvis 跟上了，上身朝向不自然；
脚跟上了，膝盖角度异常；
某些 limb 被拉伸式匹配；
PPO 训练时肌肉激活很大。
```

借鉴 OmniRetarget 的 Laplacian deformation，可以在没有物体的情况下做一个简化版：

```text
Body Graph Laplacian Loss
```

### 7.2 文件位置

```text
lib/smpl_optimization/body_graph.py
```

### 7.3 推荐 keypoints

即使不做手部动作，也建议保留腕部作为低权重 keypoint，因为上肢摆动会影响躯干和整体轨迹。

```python
BODY_KEYPOINTS = [
    "pelvis",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_hip", "left_knee", "left_ankle", "left_toe", "left_heel",
    "right_hip", "right_knee", "right_ankle", "right_toe", "right_heel",
]
```

### 7.4 推荐 edges

```python
BODY_GRAPH_EDGES = [
    ("pelvis", "spine"),
    ("spine", "thorax"),
    ("thorax", "neck"),
    ("neck", "head"),
    ("thorax", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("thorax", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_toe"),
    ("left_ankle", "left_heel"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_toe"),
    ("right_ankle", "right_heel"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "right_shoulder"),
]
```

### 7.5 Laplacian 计算

```python
def build_neighbors(num_nodes, edges):
    neighbors = [[] for _ in range(num_nodes)]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    return neighbors


def laplacian_coordinates(points, neighbors):
    # points: [..., N, 3]
    out = []
    for i, nbr in enumerate(neighbors):
        if not nbr:
            out.append(points[..., i, :])
        else:
            avg = points[..., nbr, :].mean(axis=-2)
            out.append(points[..., i, :] - avg)
    return np.stack(out, axis=-2)
```

### 7.6 在 Optimized-WHAM 中怎么用

Optimized-WHAM 中 body graph 有两个用途：

#### 用途 A：作为 lower-body 优化的 regularization / metric

优化 lower-body 时，不能破坏整体身体结构。可以计算优化前后 body graph Laplacian error：

```text
body_laplacian_delta_mean_cm
body_laplacian_delta_max_cm
```

如果优化后 body graph 改变太大，说明修脚修过头了。

#### 用途 B：导出给 ASI-PPO

ASI-PPO 可以用 body graph tracking reward，而不是只用 body positions。Optimized-WHAM 要输出：

```text
body_graph.json
body_keypoints_ref.npz
```

### 7.7 不建议现在做的事

不要在 Optimized-WHAM 里直接做完整 SMPL-to-Myo retarget。Optimized-WHAM 应该仍然保持“SMPL reference generator”的身份。真正 Myo 模型 body mapping 可以放到 ASI-PPO 或一个中间转换工具里。

---

## 8. 新增 reference_bundle.py：统一输出给 ASI-PPO

### 8.1 为什么需要 reference bundle

当前 WHAM -> AMASS `.npz` 只保留人体 pose/trans/betas/fps。可是对你的 PPO 训练而言，还需要：

```text
contact schedule；
foot anchors；
quality report；
body graph；
coordinate system；
scale；
frame ids；
processing history。
```

这些不应该散落在多个目录里，应输出统一 bundle。

### 8.2 推荐输出结构

```text
output/demo/_reference_bundles/<sequence>/
  motion.npz
  contact_schedule.npz
  stance_anchors.json
  body_graph.json
  quality_report.json
  processing_report.json
  manifest.json
```

### 8.3 motion.npz

```python
np.savez(
    motion_npz,
    poses=poses.astype(np.float32),                 # [T, 72] or [T, D]
    trans=trans.astype(np.float32),                 # [T, 3]
    betas=betas.astype(np.float32),                 # [10] or [T, 10] fixed
    gender=np.asarray(gender),
    mocap_framerate=np.asarray(fps, dtype=np.float32),
    frame_ids=frame_ids.astype(np.int32),
    coordinate_system=np.asarray("amass_zup"),
    source_coordinate_system=np.asarray("wham_yup"),
)
```

注意：给 ASI-PPO 的推荐坐标系应统一为 `amass_zup` / MuJoCo Z-up，避免 ASI-PPO 侧重复做 Y-up 到 Z-up。

### 8.4 contact_schedule.npz

```python
np.savez(
    contact_npz,
    contact_confidence=contact_confidence.astype(np.float32),   # [T, K]
    stance_mask=stance_mask.astype(np.bool_),                   # [T, K]
    foot_points=foot_points.astype(np.float32),                 # [T, K, 3]
    foot_labels=np.asarray(foot_labels),
    ground_y=np.asarray(ground_y, dtype=np.float32),
    coordinate_system=np.asarray("amass_zup"),
)
```

如果在 Optimized-WHAM 内部是 Y-up，那么导出前必须统一转换 foot_points 和 ground plane。

### 8.5 manifest.json

```json
{
  "version": "contact_reference_bundle_v1",
  "sequence": "5月1日-1",
  "num_frames": 216,
  "fps": 60.0,
  "coordinate_system": "amass_zup",
  "up_axis": "z",
  "unit": "meter",
  "motion_npz": "motion.npz",
  "contact_npz": "contact_schedule.npz",
  "stance_anchors_json": "stance_anchors.json",
  "body_graph_json": "body_graph.json",
  "quality_report_json": "quality_report.json",
  "source": {
    "wham_pkl": ".../wham_output.pkl",
    "canonical_pkl": ".../canonical_wham_output.pkl",
    "corrected_smpl_pkl": ".../corrected_smpl.pkl"
  },
  "quality": {
    "usable_for_training": true,
    "ground_confidence": "high",
    "foot_penetration_max_cm_after": 0.8,
    "stance_sliding_max_cm_after": 1.9,
    "beta_variation_after_max_abs": 0.0
  }
}
```

### 8.6 工程原则

```text
ASI-PPO 只读 manifest.json；
不要让 ASI-PPO 猜文件路径；
不要让 ASI-PPO 自己判断坐标系；
所有是否可训练的质量判断尽量在 Optimized-WHAM 阶段完成；
ASI-PPO 可以再次验证，但不负责修 reference。
```

---

## 9. quality_gates.py：把质量报告用于训练数据筛选

### 9.1 目的

坏 reference 会严重影响 PPO。Optimized-WHAM 应该输出一个明确字段：

```json
"usable_for_training": true
```

而不是让 ASI-PPO 训练时才发现轨迹有问题。

### 9.2 新增文件

```text
lib/world_grounded/quality_gates.py
```

### 9.3 推荐阈值

第一版可用阈值：

```python
@dataclass
class QualityGateConfig:
    max_foot_penetration_cm: float = 3.0
    max_stance_sliding_cm: float = 5.0
    max_root_delta_y_cm: float = 25.0
    max_root_delta_xz_cm: float = 5.0
    max_lower_body_pose_delta_rad: float = 0.8
    max_contact_switch_rate: float = 0.25
    min_stance_coverage: float = 0.10
    require_fixed_beta: bool = True
    allowed_ground_confidence: tuple[str, ...] = ("medium", "high")
```

### 9.4 输出格式

```json
{
  "usable_for_training": false,
  "failed_gates": [
    {
      "name": "max_stance_sliding_cm",
      "value": 8.7,
      "threshold": 5.0
    }
  ],
  "passed_gates": ["fixed_beta", "ground_confidence"],
  "recommendation": "exclude_from_stage0_keep_for_debug"
}
```

### 9.5 多级质量

不要只有 true/false，建议分级：

```text
A: clean，可进入 stage0；
B: usable，可进入普通训练；
C: hard，可进入 curriculum 后期；
D: reject，不训练。
```

---

## 10. 修改 video_to_fixed_smpl_to_opensim.py

### 10.1 新增 CLI

```python
parser.add_argument("--contact-preserving", action="store_true")
parser.add_argument("--export-reference-bundle", action="store_true")
parser.add_argument("--root-smooth-axes", default="y")
parser.add_argument("--stance-enter-threshold", type=float, default=0.55)
parser.add_argument("--stance-exit-threshold", type=float, default=0.30)
parser.add_argument("--quality-tier", choices=["A", "B", "C", "all"], default="B")
```

### 10.2 调用逻辑

```text
WHAM
  -> fixed-beta
  -> world-grounded
  -> lower-body optimized
  -> contact-preserving reference bundle export
  -> optional OpenSim IK
```

注意：`--contact-preserving` 应要求 `--world-grounded`：

```python
if args.contact_preserving and not args.world_grounded:
    parser.error("--contact-preserving requires --world-grounded")
```

如果启用 stance anchor lower-body 优化，也应要求 `--optimize-lower-body` 或自动启用：

```python
if args.contact_preserving and not args.optimize_lower_body:
    print("[WARN] --contact-preserving works best with --optimize-lower-body")
```

---

## 11. 测试方案

### 11.1 单元测试

新增：

```text
tests/world_grounded/test_contact_segments.py
tests/world_grounded/test_stance_anchor.py
tests/world_grounded/test_quality_gates.py
tests/smpl_optimization/test_body_graph.py
tests/smpl_optimization/test_stance_anchor_loss.py
tests/reference_bundle/test_export_bundle.py
```

### 11.2 集成测试

命令：

```powershell
python scripts/video_to_fixed_smpl_to_opensim.py `
  --video examples\5月1日-1.mp4 `
  --output-pth output\demo `
  --device cuda `
  --fps 60 `
  --skip-wham `
  --skip-fixed-beta `
  --world-grounded `
  --optimize-lower-body `
  --contact-preserving `
  --export-reference-bundle `
  --skip-ik
```

预期输出：

```text
Pipeline complete
world_grounded_dir: ...
lower_body_optimized_dir: ...
reference_bundle_dir: ...
usable_for_training: true/false
quality_tier: A/B/C/D
```

### 11.3 回归指标

每次修改后必须检查：

```text
foot_penetration_max_cm_after <= before
stance_sliding_max_cm_after <= before
root_delta_xz_max_cm 默认接近 0
beta_variation_after_max_abs = 0
frame_count unchanged
coordinate_system 正确
motion.npz 和 contact_schedule.npz frame 数一致
```

---

## 12. 最小可行版本 MVP

如果你希望先快速落地，不要一次实现全部，可以按这个顺序：

### MVP-1：root optimizer vertical-only

```text
修改 root_optimizer.py；
默认 smooth_axes=("y",)；
报告 X/Y/Z delta；
保证水平轨迹不被默认改变。
```

### MVP-2：contact segment + stance anchor report

```text
新增 contact_segments.py；
新增 stance_anchor.py；
暂时只报告，不参与优化；
输出 stance_anchors.json 和 stance sliding metrics。
```

### MVP-3：stance anchor loss 进入 lower-body pose pass

```text
把 stance anchor target 加入 lower_body.py；
优化 foot XZ/Y；
输出 before/after 对比。
```

### MVP-4：reference bundle export

```text
统一导出 motion.npz、contact_schedule.npz、quality_report.json、manifest.json；
ASI-PPO 只从 manifest 读取。
```

---

## 13. 可能存在的问题和应对

### 13.1 WHAM contact 不可靠

表现：

```text
接触期碎裂；
空中脚被判断为接触；
真正支撑脚没被识别。
```

应对：

```text
融合高度、速度、WHAM contact；
hysteresis；
最短 segment；
quality report 标记 low confidence；
ground_confidence low 时不启用强 foot lock。
```

### 13.2 脚底点定义不准

SMPL ankle/toe/heel 与肌骨模型 foot site 不完全对应。

应对：

```text
明确 foot_labels；
导出 foot point source；
后续 ASI-PPO 里做 foot site mapping；
不要混用 left/right 或 toe/heel 顺序。
```

### 13.3 坐标系错误

WHAM 是 Y-up，AMASS/MuJoCo 常用 Z-up。重复转换会导致角色躺倒或地面方向错误。

应对：

```text
manifest 中强制写 coordinate_system；
motion.npz、contact_schedule.npz 坐标系必须一致；
ASI-PPO 只接受 amass_zup；
如果是 wham_yup，必须拒绝或显式转换。
```

### 13.4 修正 reference 后偏离原视频

表现：

```text
脚不滑了，但姿态看起来不像原视频；
lower body pose delta 太大；
root height 改太多。
```

应对：

```text
加入 WHAM prior；
限制 pose delta 和 root delta；
报告 original_vs_corrected MPJPE；
把过度修正的 clip 标记为 C/D 级。
```

### 13.5 动作中真实存在脚部 pivot

正手高远球或转身动作中，支撑脚可能以 toe 为轴旋转。如果同时锁 toe 和 heel，可能过约束。

应对：

```text
先锁 foot center；
toe/heel 只用于高度，不强锁 XZ；
或对 toe/heel 使用较低 XZ 权重；
检测 pivot motion 后降低 anchor 权重。
```

### 13.6 优化变慢

lower-body pose pass 会调用 SMPL forward，加入 anchor loss 后可能更慢。

应对：

```text
保留 chunk_size；
先使用 deterministic Y shift + anchor report；
只对质量较差 clip 启用 pose pass；
默认 iterations 40，debug 时 10，最终导出时 80。
```

### 13.7 下游 PPO 仍然学不好

可能不是 PPO 问题，而是 reference 对肌骨模型不可行。

应对：

```text
增加 joint limit / velocity limit metrics；
增加 body graph error；
让 ASI-PPO 返回 tracking failure frame；
把失败 frame 回写到 reference quality report。
```

---

## 14. 最终你应该得到什么

完成改造后，Optimized-WHAM 应该从：

```text
WHAM 输出清洗 + OpenSim retarget 前处理
```

升级为：

```text
Contact-Preserving SMPL Reference Generator
```

它应该输出的是：

```text
地面一致；
固定 beta；
root vertical 合理；
支撑脚不滑；
穿地少；
下肢姿态不过度偏离原视频；
身体结构保持；
有质量分级；
有明确坐标系；
能被 ASI-PPO 直接读取的 reference bundle。
```

这正是你从 OmniRetarget 中应该借鉴的核心：**不要把低质量视频轨迹直接丢给 PPO，而是在 Optimized-WHAM 阶段先生成高质量、接触保持、肌骨模型更容易复现的参考轨迹。**
