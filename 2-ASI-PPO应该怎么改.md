# 2. ASI-PPO 应该怎么改

> 目标：在**不加入手部动作、不加入球拍、不加入羽毛球任务奖励**的前提下，把 ASI-PPO / MuscleMimic 的训练目标明确收敛到一件事：**让肌骨模型尽可能稳定、准确、物理合理地复现 Optimized-WHAM 提供的全身参考轨迹**。

---

## 0. 总体定位

ASI-PPO 不应该承担视频清洗、地面估计、foot contact 修复、SMPL fixed beta 等前处理任务。这些应由 Optimized-WHAM 完成。

ASI-PPO 应该做的是：

```text
读取 Optimized-WHAM 产出的高质量 reference bundle；
把 reference retarget 到肌骨模型可跟踪的状态空间；
设计以轨迹复现为核心的 observation、reward、termination、curriculum；
训练肌骨模型通过肌肉激活或控制动作复现 reference；
输出 tracking quality 和 failure diagnostics。
```

当前你不做手部、不拿球拍，因此 ASI-PPO 的核心应从“羽毛球任务学习”转为：

```text
Contact-Aware Whole-Body Musculoskeletal Motion Imitation
```

也就是：

```text
全身轨迹复现 + 足底接触一致 + 肌肉控制平滑 + 关节/动力学合理
```

---

## 1. ASI-PPO 应该消费什么数据

ASI-PPO 不应该直接消费零散的 WHAM pkl，也不应该自己猜测 contact、ground、coordinate system。它应该只读取 Optimized-WHAM 输出的 manifest。

推荐输入结构：

```text
reference_bundle_dir/
  manifest.json
  motion.npz
  contact_schedule.npz
  body_graph.json
  quality_report.json
  processing_report.json
```

### 1.1 manifest.json

ASI-PPO 入口只接收：

```bash
--reference-manifest path/to/manifest.json
```

manifest 示例：

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
  "body_graph_json": "body_graph.json",
  "quality_report_json": "quality_report.json",
  "quality": {
    "usable_for_training": true,
    "quality_tier": "A"
  }
}
```

ASI-PPO 启动时必须验证：

```text
coordinate_system == amass_zup；
unit == meter；
num_frames 与 npz 内部一致；
fps 与训练配置能对齐；
quality.usable_for_training == true，除非显式 --allow-low-quality-reference。
```

---

## 2. 新增 ReferenceBundle 数据加载层

### 2.1 文件建议

在 ASI-PPO / BadmintonMimic 侧新增：

```text
BadmintonMimic/asi/data/reference_bundle.py
BadmintonMimic/asi/data/reference_manifest.py
BadmintonMimic/asi/data/contact_schedule.py
BadmintonMimic/asi/data/body_graph.py
```

如果当前项目没有 `asi/` 目录，可以放在：

```text
BadmintonMimic/badminton_mimic/data/
```

重点是：形成一个独立的数据合同层，而不是把读取逻辑散落在训练脚本中。

### 2.2 数据类

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class ReferenceBundle:
    root_orient: np.ndarray          # [T, 3] axis-angle or converted representation
    pose_body: np.ndarray            # [T, D]
    trans: np.ndarray                # [T, 3]
    betas: np.ndarray                # [10] or [T, 10]
    fps: float
    frame_ids: np.ndarray            # [T]
    contact_confidence: np.ndarray   # [T, K]
    stance_mask: np.ndarray          # [T, K]
    foot_points: np.ndarray          # [T, K, 3]
    foot_labels: list[str]
    body_graph: dict
    quality: dict
    coordinate_system: str
    manifest_path: Path
```

### 2.3 加载流程

```python
def load_reference_bundle(manifest_path: str | Path) -> ReferenceBundle:
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
    base = Path(manifest_path).parent

    assert manifest["coordinate_system"] == "amass_zup"
    assert manifest["unit"] == "meter"

    motion = np.load(base / manifest["motion_npz"], allow_pickle=True)
    contact = np.load(base / manifest["contact_npz"], allow_pickle=True)
    body_graph = json.load(open(base / manifest["body_graph_json"], "r", encoding="utf-8"))
    quality = json.load(open(base / manifest["quality_report_json"], "r", encoding="utf-8"))

    validate_frame_count(motion, contact, manifest)
    validate_quality(quality)

    return ReferenceBundle(...)
```

### 2.4 为什么必须有这一层

如果没有统一数据层，后续会出现：

```text
训练脚本自己转换坐标系；
reward 函数自己读取 contact；
evaluation 又用另一套 foot labels；
一个脚本认为是 Y-up，另一个认为是 Z-up；
frame 数不一致时静默错位；
训练坏了却不知道是 reference 问题还是 PPO 问题。
```

ReferenceBundle 层就是为了避免这些工程问题。

---

## 3. Retarget 到肌骨模型：不要只用 pose，要生成 tracking targets

### 3.1 问题

ASI-PPO 不能只拿 SMPL 的 axis-angle pose 当 reward。肌骨模型通常有自己的：

```text
body names；
joint names；
site names；
root qpos 定义；
关节自由度；
肌肉控制维度；
foot site 位置。
```

所以需要把 ReferenceBundle 转换成肌骨模型的 tracking targets。

### 3.2 新增 RetargetCache

建议新增：

```text
BadmintonMimic/asi/retarget/reference_cache.py
```

数据结构：

```python
@dataclass
class TrackingReference:
    qpos_ref: np.ndarray             # [T, nq] optional
    qvel_ref: np.ndarray             # [T, nv] optional
    root_pos_ref: np.ndarray         # [T, 3]
    root_rot_ref: np.ndarray         # [T, 4] quaternion
    body_pos_ref: np.ndarray         # [T, B, 3]
    body_rot_ref: np.ndarray         # [T, B, 4]
    body_linvel_ref: np.ndarray      # [T, B, 3]
    body_angvel_ref: np.ndarray      # [T, B, 3]
    joint_pos_ref: np.ndarray        # [T, J]
    joint_vel_ref: np.ndarray        # [T, J]
    foot_pos_ref: np.ndarray         # [T, F, 3]
    foot_contact_ref: np.ndarray     # [T, F]
    body_graph_lap_ref: np.ndarray   # [T, N, 3]
    valid_mask: np.ndarray           # [T]
```

### 3.3 生成方式

可以有两种路线：

#### 路线 A：已有 GMR / retarget 结果作为 qpos_ref

```text
Optimized-WHAM motion.npz
  -> 现有 GMR / MuscleMimic retarget
  -> qpos_ref, qvel_ref
  -> rollout tracking
```

优点：工程量小。

缺点：如果 GMR retarget 自身有脚滑或关节异常，需要额外检查。

#### 路线 B：用 body targets 做 reference，不强依赖 qpos_ref

```text
SMPL / OpenSim keypoints
  -> map 到肌骨模型 body/site targets
  -> PPO reward 主要追 body position/orientation
```

优点：更稳健，避免 qpos 维度和模型自由度不一致。

缺点：reward 和 mapping 设计更复杂。

建议第一阶段采用混合方式：

```text
qpos_ref 用于 joint tracking；
body_pos_ref/body_rot_ref 用于主 tracking；
foot_contact_ref 用于 contact reward；
body_graph_lap_ref 用于结构保持 reward。
```

---

## 4. Observation 设计

你不做球拍、不做外部物体，因此 observation 应保持干净，不要引入无关信息。

### 4.1 推荐 observation

```text
当前 proprioception：
  root height
  root orientation
  root linear/angular velocity
  joint positions
  joint velocities
  muscle activations or previous action

reference phase：
  phase index or normalized time
  reference root error in local frame
  reference body position error in local frame
  reference joint position error
  future reference summary，1/3/5/10 frames

contact reference：
  left/right foot contact phase
  stance mask
  time-to-contact-switch，可选
```

### 4.2 不建议加入

```text
不要加入 racket；
不要加入 shuttle；
不要加入 hand pose target，如果你明确不要手部；
不要加入 court target；
不要加入任务成功奖励。
```

### 4.3 root-relative 表达

reference body positions 最好转到当前 root 局部坐标：

```python
body_pos_error_local = R_root_current.T @ (body_pos_ref - body_pos_current)
```

这样 PPO 更容易学习，不会过度依赖世界绝对坐标。

---

## 5. Reward 设计：以轨迹复现为核心

### 5.1 总体公式

推荐 reward：

```text
R =
  w_root_pos       * r_root_pos
+ w_root_rot       * r_root_rot
+ w_body_pos       * r_body_pos
+ w_body_rot       * r_body_rot
+ w_joint_pos      * r_joint_pos
+ w_joint_vel      * r_joint_vel
+ w_body_graph     * r_body_graph
+ w_foot_contact   * r_foot_contact
+ w_foot_lock      * r_foot_lock
+ w_penetration    * r_no_penetration
+ w_action_smooth  * r_action_smooth
+ w_muscle_effort  * r_muscle_effort
+ w_joint_limit    * r_joint_limit
```

其中前六项是轨迹复现，后面是让复现更稳定、更物理。

### 5.2 root tracking

```python
e_root = ||root_pos - root_pos_ref||^2
r_root_pos = exp(-k_root_pos * e_root)
```

注意：root X/Z 和 root height 权重可以分开：

```text
root_xz_weight 高：保持轨迹；
root_z/up_weight 中等：避免为了贴合视频高度造成失衡；
root orientation 权重较高：保持身体朝向。
```

如果在 MuJoCo 中是 Z-up，则：

```text
root horizontal = x/y
root vertical = z
```

不要和 WHAM Y-up 混淆。

### 5.3 body position tracking

```python
e_body = mean_i w_i * ||body_pos_i - body_pos_ref_i||^2
r_body_pos = exp(-k_body_pos * e_body)
```

推荐 body 权重：

```text
pelvis / torso: 高
head / neck: 中
upper legs / lower legs / feet: 高
upper arms: 中
forearms/wrists: 低或中低
hands/fingers: 0 或不使用
```

因为你不要手部动作，手指和细粒度手部应该完全不进入 reward。腕部可以低权重保留，因为它影响上肢整体姿态。

### 5.4 body orientation tracking

```python
e_rot = quaternion_distance(body_rot, body_rot_ref)
r_body_rot = exp(-k_body_rot * e_rot)
```

注意不要对所有小 body 都高权重。否则模型可能为了匹配不可靠视频估计，牺牲稳定性。

### 5.5 joint position / velocity tracking

```python
e_q = mean_j w_j * (q_j - q_ref_j)^2
r_joint_pos = exp(-k_joint * e_q)
```

这项要适度。肌骨模型的关节定义未必等同于 SMPL/GMR 的关节角。如果过高，会导致 PPO 追一个不完全可实现的 qpos。

建议：

```text
body_pos/body_rot 是主 reward；
joint_pos 是辅助 reward；
joint_vel 用于时序；
手部 joint 不参与。
```

### 5.6 body graph / Laplacian reward

这是从 OmniRetarget 借鉴但适配你目标的关键 reward。

对于肌骨模型当前 body keypoints 和 reference body keypoints，计算 body graph Laplacian：

```python
L_i = p_i - mean(p_j for j in neighbors(i))
```

reward：

```python
e_lap = mean_i ||L_i_current - L_i_ref||^2
r_body_graph = exp(-k_lap * e_lap)
```

它的作用：

```text
保持身体整体结构；
减少只追局部点造成的姿态变形；
缓解 SMPL 与肌骨模型比例不一致；
让 torso-leg-arm 的协调性更好。
```

建议初始权重：

```yaml
reward:
  body_graph:
    weight: 0.15
    scale: 20.0
```

不要一开始过高，否则会和 body position tracking 重复或冲突。

### 5.7 foot contact reward

使用 Optimized-WHAM 的 `stance_mask` 或 `contact_confidence`。

#### stance 高度 reward

```python
if contact_ref[foot] > threshold:
    penalize |foot_height - ground_height|
```

#### stance velocity reward

```python
if contact_ref[t, foot] and contact_ref[t-1, foot]:
    penalize ||foot_xy_vel||
```

在 MuJoCo Z-up 下：

```text
horizontal = x/y
vertical = z
```

#### swing phase

swing phase 不应该锁脚，只需要：

```text
不穿地；
合理 clearance；
不要强行贴地。
```

### 5.8 penetration / ground reward

```python
penetration = relu(ground_z - foot_z)
penalty = penetration^2
```

这项可以作为 penalty，不一定用 exp reward。

### 5.9 muscle effort / action smoothness

为了轨迹复现，不能让模型用极端肌肉激活追踪噪声。

```python
r_effort = -mean(action^2)
r_action_rate = -mean((action_t - action_{t-1})^2)
```

权重不要太高，否则会牺牲 tracking。

---

## 6. 推荐 reward 权重初始配置

可以新建：

```text
configs/asi_ppo/contact_preserving_tracking.yaml
```

示例：

```yaml
task:
  name: contact_preserving_fullbody_tracking
  use_racket: false
  use_hand_detail: false
  reference_manifest: null
  allow_low_quality_reference: false

reference:
  future_offsets: [1, 3, 5, 10]
  coordinate_system: amass_zup
  resample_to_control_dt: true
  drop_hand_joints: true
  wrist_weight: 0.25
  finger_weight: 0.0

reward:
  root_pos:
    weight: 0.80
    scale: 8.0
  root_rot:
    weight: 0.60
    scale: 4.0
  body_pos:
    weight: 1.50
    scale: 12.0
  body_rot:
    weight: 0.80
    scale: 3.0
  joint_pos:
    weight: 0.50
    scale: 4.0
  joint_vel:
    weight: 0.25
    scale: 0.5
  body_graph:
    weight: 0.20
    scale: 20.0
  foot_contact_height:
    weight: 0.45
    scale: 80.0
  foot_contact_velocity:
    weight: 0.45
    scale: 8.0
  foot_penetration:
    weight: 0.30
    scale: 100.0
  action_rate:
    weight: 0.05
  muscle_effort:
    weight: 0.02
  joint_limit:
    weight: 0.10

termination:
  enable: true
  max_root_error_m: 0.75
  max_body_error_m: 0.80
  max_torso_angle_error_rad: 1.2
  max_foot_penetration_m: 0.12
  grace_frames: 20

curriculum:
  enabled: true
  stages:
    - name: clean_root_body
      quality_tiers: [A]
      rewards_enabled: [root_pos, root_rot, body_pos, body_rot]
      max_steps: 200000
    - name: add_joint_tracking
      quality_tiers: [A, B]
      rewards_enabled: [root_pos, root_rot, body_pos, body_rot, joint_pos, joint_vel]
      max_steps: 400000
    - name: add_contact
      quality_tiers: [A, B]
      rewards_enabled: [root_pos, root_rot, body_pos, body_rot, joint_pos, joint_vel, foot_contact_height, foot_contact_velocity]
      max_steps: 600000
    - name: hard_clips
      quality_tiers: [A, B, C]
      max_steps: 1000000
```

这些权重只是初始值。实际训练时应根据 tracking diagnostics 调整。

---

## 7. Curriculum 设计

不要一开始让 PPO 同时追全身、contact、joint velocity、muscle effort。建议分阶段。

### Stage 0：只训练干净短片段

输入：

```text
quality_tier = A
clip 长度短
root 变化不剧烈
contact_confidence high
```

reward：

```text
root_pos
root_rot
body_pos
body_rot
action_smooth
```

目的：让模型先学会基本姿态复现。

### Stage 1：加入 joint tracking

加入：

```text
joint_pos
joint_vel
body_graph
```

目的：让肌骨模型不只是 body point 对齐，也尽量关节轨迹相似。

### Stage 2：加入 foot contact reward

加入：

```text
foot_contact_height
foot_contact_velocity
foot_penetration
```

目的：让脚底接触更物理，减少跟踪时滑脚。

### Stage 3：加入更长、更难片段

加入 B/C 级 clips，但不要加入 D 级。C 级可以作为 hard cases。

### Stage 4：全量微调

小 learning rate，训练更稳定的策略。

---

## 8. Manifest-based dataset builder

### 8.1 新增脚本

```text
BadmintonMimic/scripts/build_contact_tracking_manifest.py
```

输入：

```bash
python BadmintonMimic/scripts/build_contact_tracking_manifest.py \
  --reference-root output/demo/_reference_bundles \
  --out manifests/contact_tracking_train.jsonl \
  --include-tiers A,B \
  --min-frames 60 \
  --max-foot-penetration-cm 3.0 \
  --max-stance-sliding-cm 5.0
```

输出 JSONL：

```json
{"manifest": ".../seq1/manifest.json", "quality_tier": "A", "num_frames": 216, "fps": 60.0}
{"manifest": ".../seq2/manifest.json", "quality_tier": "B", "num_frames": 184, "fps": 60.0}
```

### 8.2 为什么用 JSONL

```text
方便增量添加 clips；
方便训练时 streaming 读取；
方便记录每个 clip 的质量；
方便排查某个 clip 失败。
```

---

## 9. 训练环境需要的改造

### 9.1 Reference phase manager

新增：

```text
BadmintonMimic/asi/envs/reference_phase.py
```

职责：

```text
根据 episode start frame 和当前 simulation step 返回 reference frame；
处理 reference fps 与 control dt 不一致；
支持循环、截断、随机起点；
支持未来帧查询。
```

伪代码：

```python
class ReferencePhaseManager:
    def __init__(self, ref, control_dt):
        self.ref = ref
        self.control_dt = control_dt
        self.ref_dt = 1.0 / ref.fps

    def frame_at_step(self, episode_start_frame, step):
        t = step * self.control_dt
        ref_float = episode_start_frame + t / self.ref_dt
        return interpolate_reference(self.ref, ref_float)
```

### 9.2 Contact reference interpolation

contact 不建议线性插值成奇怪值。可以：

```text
contact_confidence 用最近邻或线性均可；
stance_mask 用最近邻；
contact switch time 用 discrete frame 计算。
```

### 9.3 训练随机起点

对一条长 clip，不要每次从 frame 0 开始。建议：

```python
start_frame = random.randint(0, T - episode_len - 1)
```

但对于很难的动作，早期 curriculum 可以固定从 0 开始。

---

## 10. Evaluation / Diagnostics

PPO 训练后必须输出比 reward 更细的指标，否则无法判断问题来自 reference 还是 policy。

### 10.1 新增评估脚本

```text
BadmintonMimic/scripts/evaluate_contact_tracking.py
```

输入：

```bash
python BadmintonMimic/scripts/evaluate_contact_tracking.py \
  --checkpoint runs/xxx/checkpoint.pt \
  --manifest manifests/contact_tracking_val.jsonl \
  --out reports/contact_tracking_eval.json
```

### 10.2 指标

```json
{
  "mean_root_pos_error_cm": 4.2,
  "mean_root_rot_error_deg": 6.1,
  "mean_body_pos_error_cm": 5.8,
  "mean_joint_error_rad": 0.18,
  "mean_body_graph_error_cm": 3.1,
  "stance_foot_sliding_cm_s": 2.9,
  "foot_penetration_max_cm": 1.4,
  "episode_success_rate": 0.82,
  "mean_episode_length": 182.4,
  "failure_reasons": {
    "root_error": 3,
    "fall": 5,
    "foot_penetration": 1
  }
}
```

### 10.3 Failure frame dump

对失败 episode 输出：

```text
clip id；
start frame；
failure frame；
root error；
body error；
foot contact state；
current qpos；
reference qpos；
```

这可以反馈给 Optimized-WHAM，判断是否是 reference 本身坏。

---

## 11. 不做手部动作时的具体处理

### 11.1 数据层

如果 reference 里有 hand pose：

```text
可以保留在 motion.npz 中，但 ASI-PPO 不使用；
loader 可以读取但 drop_hand_joints=True；
finger joints 权重设为 0。
```

### 11.2 Reward 层

```text
finger body / hand sites 不进入 body_pos reward；
wrist 可保留低权重，避免前臂完全乱动；
elbow/shoulder 中等权重；
torso/pelvis/lower body 高权重。
```

### 11.3 Termination 层

不要因为手部误差终止 episode。

### 11.4 Evaluation 层

报告时区分：

```text
full_body_error，包括上肢；
core_body_error，不包括手指/手掌；
lower_body_error；
torso_error。
```

这样你可以证明“不建模手部”并不影响主要轨迹复现目标。

---

## 12. ASI-PPO 侧可能存在的问题

### 12.1 reference 太难，PPO 一开始学不动

表现：

```text
episode 很快终止；
reward 接近 0；
肌肉激活很大；
身体倒地。
```

解决：

```text
从 A 级短片段开始；
降低 termination 严格度；
先关掉 foot_contact reward；
先只追 root 和 torso；
逐步加入 joint/body/contact。
```

### 12.2 contact reward 和 body tracking 冲突

如果 reference 中脚底 contact 仍不完美，contact reward 会和 body tracking 打架。

解决：

```text
contact reward 只对 confidence > 0.7 的帧启用；
低 confidence 帧只做 penetration penalty；
quality_tier B/C 的 contact reward 权重降低。
```

### 12.3 肌骨模型无法追高频抖动

视频 SMPL 有抖动，即使 Optimized-WHAM 修过，也可能存在高频噪声。

解决：

```text
ReferenceBundle 中提供 smoothed reference；
PPO reward 使用 velocity/acceleration 平滑后的 target；
保留原始 reference 只用于评估。
```

### 12.4 坐标系双重转换

这是最常见工程错误。

解决：

```text
ASI-PPO 只接受 amass_zup；
如果 manifest 是 wham_yup，直接报错；
不要在多个地方重复 Y-up -> Z-up。
```

### 12.5 qpos_ref 和 body_pos_ref 不一致

GMR retarget 出来的 qpos_ref 可能与 body_pos_ref 不完全一致。

解决：

```text
生成 RetargetCache 后重新 rollout 一次 qpos_ref；
用 rollout 得到的 body_pos_ref_train 作为训练 target；
原 SMPL body_pos_ref 作为评估参考；
报告 retarget_error。
```

### 12.6 reward 权重过拟合某些 clips

解决：

```text
按 quality tier 分组评估；
每组单独看 tracking error；
不要只看平均 reward；
使用 validation clips。
```

---

## 13. ASI-PPO 最小可行改造路线

### MVP-1：ReferenceBundle loader

```text
只读取 manifest、motion.npz、contact_schedule.npz；
验证坐标系和 frame 数；
训练时仍使用原有 reward。
```

### MVP-2：manifest-based dataset builder

```text
用 Optimized-WHAM quality report 筛选训练数据；
只训练 A/B 级 clips。
```

### MVP-3：foot contact reward

```text
加入 stance height 和 stance velocity reward；
先只对 high confidence contact 启用。
```

### MVP-4：body graph reward

```text
从 body keypoints 计算 Laplacian；
加入低权重结构保持 reward。
```

### MVP-5：完整 curriculum

```text
clean -> joint -> contact -> hard clips。
```

---

## 14. 最终 ASI-PPO 应该变成什么

完成改造后，ASI-PPO 不再是“随便读 AMASS npz，然后让肌骨模型模仿”。它应该变成：

```text
Contact-Aware Musculoskeletal Motion Imitation Trainer
```

它的输入是 Optimized-WHAM 产出的干净 reference bundle；它的训练目标是：

```text
root 轨迹复现；
躯干和四肢 body pose 复现；
关节轨迹尽量复现；
身体结构关系保持；
支撑脚接触一致；
脚不穿地、不明显滑动；
肌肉激活平滑且不过度；
不依赖手部、球拍、羽毛球任务。
```

这就是在你当前目标下，从 OmniRetarget 最值得借鉴到 ASI-PPO 的部分：**把高质量 reference 和 contact schedule 显式喂给 PPO，让 PPO 学“可物理执行的轨迹复现”，而不是让 PPO 从噪声视频轨迹中自己猜接触和修错误。**
