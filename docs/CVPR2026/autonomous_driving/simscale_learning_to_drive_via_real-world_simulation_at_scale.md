---
title: >-
  [论文解读] SimScale: Learning to Drive via Real-World Simulation at Scale
description: >-
  [CVPR 2026 (Oral)][自动驾驶][仿真数据] 提出 SimScale 框架，通过对现有驾驶日志进行轨迹扰动 + 反应式环境仿真 + 神经渲染生成大规模高保真模拟数据，配合伪专家轨迹监督和 sim-real co-training 策略，使端到端规划器在 NAVSIM v2 上取得显著提升（navhard +8.6 EPDMS），且性能随仿真数据量平滑扩展。
tags:
  - CVPR 2026 (Oral)
  - 自动驾驶
  - 仿真数据
  - 端到端规划
  - 仿真到现实
  - 数据扩展
  - 伪专家轨迹
  - 神经渲染
  - co-training
---

# SimScale: Learning to Drive via Real-World Simulation at Scale

**会议**: CVPR 2026 (Oral)  
**arXiv**: [2511.23369](https://arxiv.org/abs/2511.23369)  
**代码**: [OpenDriveLab/SimScale](https://github.com/OpenDriveLab/SimScale)  
**作者**: Haochen Tian, Tianyu Li, Haochen Liu, Jiazhi Yang 等 (CASIA, OpenDriveLab@HKU, Xiaomi EV)
**领域**: autonomous_driving  
**关键词**: 仿真数据, 端到端规划, 仿真到现实, 数据扩展, 伪专家轨迹, 神经渲染, co-training

## 一句话总结
提出 SimScale 框架，通过对现有驾驶日志进行轨迹扰动 + 反应式环境仿真 + 神经渲染生成大规模高保真模拟数据，配合伪专家轨迹监督和 sim-real co-training 策略，使端到端规划器在 NAVSIM v2 上取得显著提升（navhard +8.6 EPDMS），且性能随仿真数据量平滑扩展。

## 研究背景与动机

全自动驾驶需要在广泛场景中学习合理决策，包括安全关键和分布外 (OOD) 场景。然而：

- **数据分布偏差**：人类专家采集的真实数据以常规驾驶为主，安全关键场景（急刹、险避让）和 OOD 场景严重不足
- **示范偏差 (demonstration bias)**：模仿学习策略仅暴露于专家分布内的状态，无法学习从偏离状态恢复的能力
- **现有仿真方案的局限**：
    - 传统仿真器 (CARLA/MetaDrive)：渲染真实感不足，sim-to-real gap 大
    - 基于 NeRF/3DGS 的神经渲染：质量高但缺乏场景交互性（非反应式环境）
    - 纯轨迹扰动：只产生新状态，缺乏对应的高质量传感器观测

**核心思路**：在已有真实驾驶日志上扰动 ego 轨迹产生新状态，结合反应式环境模拟其他交通参与者的反应，再用神经渲染生成高保真多视角图像，最后为新状态生成伪专家监督轨迹，从而以可扩展方式合成海量训练数据。

## 方法详解

### 整体流程：扰动 → 反应 → 渲染 → 标注

SimScale 的仿真数据生成流程分为三个核心模块：

### 1. 轨迹扰动 (Trajectory Perturbation)

在时间 $T$ 到 $T+H$ 之间对 ego 车辆的原始轨迹施加扰动，生成偏离正常行驶路线的新状态序列。扰动方式包括横向偏移、速度变化等，使 ego 进入原始数据中未出现的状态空间。

### 2. 反应式环境仿真 (Reactive Environment Rollout)

对于扰动后的 ego 状态，环境中的其他交通参与者（车辆、行人等）需要做出相应反应。采用反应式仿真引擎（基于 MTGS 等），确保仿真场景的物理合理性和交互一致性，避免出现穿模、碰撞等不合理现象。

### 3. 神经渲染 (Neural Rendering)

利用先进的 3D Gaussian Splatting (3DGS) 技术，根据扰动后的 ego 位姿和反应式环境状态，生成高保真的多视角相机观测图像，为端到端模型提供视觉输入。

### 4. 伪专家轨迹生成 (Pseudo-Expert Trajectory)

为仿真状态提供动作监督标签。论文比较了两种策略：

- **Recovery-based（恢复式）**：在扰动结束时刻 $T+H$，直接规划一条从当前偏离状态恢复到合理行驶状态的轨迹。类似 DAgger 的思想，教模型"犯错后如何纠正"
- **Planner-based（规划器式）**：使用规则化规划器 PDM 在仿真环境中重新规划最优轨迹，提供更优质的动作监督

### 5. Sim-Real Co-Training 策略

将仿真数据与真实数据混合训练，采用简单的联合训练策略（无需复杂的域适应）。对不同类型的端到端规划器均适用：

- **回归式策略** (LTF / Transfuser)：直接回归轨迹点
- **扩散式策略** (DiffusionDrive)：基于扩散模型生成轨迹分布
- **打分式策略** (GTRS-Dense)：对候选轨迹进行打分排序。此类策略还支持"仅用奖励"模式 (rewards only)，即仿真数据只提供奖励信号而非模仿学习监督

## 实验关键数据

评估基于 NAVSIM v2 基准，包含 navhard（高难度安全关键场景）和 navtest（常规测试集）两个 split。

### Table 1: Model Zoo 主要结果（EPDMS 指标）

| 模型 | 骨干网络 | Co-Train 模式 | navhard EPDMS | navhard 提升 | navtest EPDMS | navtest 提升 |
|---|---|---|---|---|---|---|
| LTF | ResNet34 | w/ pseudo-expert | 30.3 | **+6.9** | 84.4 | **+2.9** |
| DiffusionDrive | ResNet34 | w/ pseudo-expert | 32.6 | **+5.1** | 85.9 | **+1.7** |
| GTRS-Dense | ResNet34 | w/ pseudo-expert | 46.1 | **+7.8** | 84.0 | **+1.7** |
| GTRS-Dense | ResNet34 | rewards only | 46.9 | **+8.6** | 84.6 | **+2.3** |
| GTRS-Dense | V2-99 | w/ pseudo-expert | 47.7 | **+5.8** | 84.5 | **+0.5** |
| GTRS-Dense | V2-99 | rewards only | 48.0 | **+6.1** | 84.8 | **+0.8** |

**关键发现**：
- 所有策略类型均从仿真数据中获益，navhard 提升尤为显著（+5.1 ~ +8.6）
- GTRS-Dense + rewards only 模式达到最大 navhard 提升 (+8.6)，表明打分式策略不需要伪专家轨迹标签，仅靠奖励信号即可充分利用仿真数据
- navtest 上也有一致提升 (+0.5 ~ +2.9)，说明仿真数据同时改善泛化能力

### Table 2: 扩展性分析——仿真数据量 vs 性能

| 仿真数据轮数 | 仿真 token 数 | GTRS navhard (pseudo-expert) | GTRS navhard (rewards only) | LTF navhard |
|---|---|---|---|---|
| 0 (仅真实数据) | 0 | 38.3 | 38.3 | 23.4 |
| 1 轮 (round 0) | ~65K | 42.5 | 43.1 | 27.8 |
| 3 轮 (round 0-2) | ~166K | 44.8 | 45.6 | 29.5 |
| 5 轮 (round 0-4) | ~236K | 46.1 | 46.9 | 30.3 |

**扩展性洞察**：
- 性能随仿真数据量平滑增长，未见明显饱和
- 即使不增加真实数据，仅扩展仿真数据即可持续获得收益
- 不同策略架构展现不同的扩展特性：打分式策略扩展最好，扩散式策略次之

## 亮点与洞察

- **CVPR 2026 Oral**：获评口头报告，认可度高
- **完整的仿真-训练闭环**：从扰动到反应到渲染到标注到训练，形成完整可扩展的数据增强管线
- **伪专家应具有探索性**：Recovery-based 伪专家让模型学会从错误中恢复，比 planner-based 在某些场景下更有效，说明数据多样性比轨迹最优性更重要
- **多模态建模激发扩展性**：扩散式和打分式策略比回归式策略更能利用扩展的仿真数据，因为它们建模了轨迹分布而非单点估计
- **Reward is All You Need**：GTRS-Dense 在 rewards only 模式下表现最佳，表明对于打分式策略，仿真数据上无需做模仿学习，仅提供奖励信号即可
- **Sim-Real Gap 可控**：简单的 co-training 策略即可有效，无需域自适应/域随机化等复杂技术，归因于神经渲染的高保真度
- **已开源数据和代码**：TB 级仿真数据 + 训练代码 + 模型权重全部公开，可复现性强

## 局限性

- **依赖基础设施**：需要高质量的 3DGS 神经渲染模型 (MTGS) 和反应式仿真引擎，前置成本高
- **仿真数据规模巨大**：5 轮仿真产生数 TB 传感器数据，存储和 I/O 开销显著
- **场景多样性受限于原始日志**：扰动只能在已有场景的邻域内生成变体，无法创造全新场景类型（如原始数据无雪天，仿真也无法生成雪天）
- **评估局限**：主要在 NAVSIM v2 闭环评估，未在其他基准（如 nuPlan、CARLA 闭环）上验证
- **伪专家质量上限**：PDM 规划器自身的性能上限决定了伪专家的质量天花板
- **未探索更长的仿真时长和多轮交互**：当前仿真窗口为固定 6 秒，更长时间的仿真和累积误差处理尚未涉及

## 相关工作

- **端到端自动驾驶规划**：UniAD、VAD、Transfuser 等直接从传感器到轨迹的端到端方法，受限于训练数据中安全关键场景不足
- **驾驶场景仿真**：CARLA/MetaDrive（传统渲染）到 NeRF/3DGS 神经渲染（高保真但静态）再到反应式仿真（如 DriveArena、MTGS），SimScale 在反应式仿真基础上加入可扩展的伪专家生成
- **数据扩展与 co-training**：DAgger 系列（在线交互）、DROID/Scaling-up（大规模数据收集），SimScale 走仿真扩展路线，避免额外真实数据采集成本
- **打分式规划**：GTRS 等基于奖励打分的轨迹选择范式，本文证明其在 sim-real 场景下的独特优势（rewards only）

## 评分

- 新颖性: 4/5 — 将轨迹扰动+反应式仿真+神经渲染+伪专家的完整闭环框架化，并首次系统性研究端到端规划器的仿真数据 scaling law
- 实验充分度: 5/5 — 3 种策略架构 x 2 种骨干 x 2 种伪专家 x 5 轮扩展，消融全面，已开源数据和代码
- 写作质量: 4/5 — 结构清晰，核心洞见提炼到位（三个 scaling insight），CVPR Oral 水准
- 价值: 5/5 — 为端到端自动驾驶提供了可扩展的仿真数据增强范式，开源生态完善，实用性极强

<!-- RELATED:START -->

## 相关论文

- [Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos](learning_to_drive_is_a_free_gift_large-scale_label-free_autonomy_pretraining_fro.md)
- [SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model](../../CVPR2025/autonomous_driving/scenediffuser_city-scale_traffic_simulation_via_a_generative_world_model.md)
- [LookOut: Real-World Humanoid Egocentric Navigation](../../ICCV2025/autonomous_driving/lookout_real-world_humanoid_egocentric_navigation.md)
- [EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video](../../ICLR2026/autonomous_driving/egodex_learning_dexterous_manipulation_from_large-scale_egocentric_video.md)
- [Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving](drive_my_way_preference_alignment_of_vision-language-action_model_for_personaliz.md)

<!-- RELATED:END -->
