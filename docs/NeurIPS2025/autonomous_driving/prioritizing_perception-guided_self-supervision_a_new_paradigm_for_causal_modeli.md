---
title: >-
  [论文解读] Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving
description: >-
  [NeurIPS 2025][自动驾驶][因果混淆] 通过感知输出（车道线、agent 轨迹）和自监督学习来建立因果关系，解决端到端自动驾驶中的因果混淆问题，在 Bench2Drive 闭环评估上实现 SOTA（Driving Score 78.08）。
tags:
  - NeurIPS 2025
  - 自动驾驶
  - 因果混淆
  - 自监督学习
  - 端到端驾驶
  - 感知引导
  - 闭环评估
---

# Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving

**会议**: NeurIPS 2025  
**arXiv**: [2511.08214](https://arxiv.org/abs/2511.08214)  
**代码**: 有  
**领域**: 自动驾驶 / 端到端决策  
**关键词**: 因果混淆, 自监督学习, 端到端驾驶, 感知引导, 闭环评估

## 一句话总结

通过感知输出（车道线、agent 轨迹）和自监督学习来建立因果关系，解决端到端自动驾驶中的因果混淆问题，在 Bench2Drive 闭环评估上实现 SOTA（Driving Score 78.08）。

## 研究背景与动机

**领域现状**：端到端自动驾驶系统在开环评估中表现良好，但在闭环场景中性能严重下降。

**现有痛点**：因果混淆（causal confusion）是根本原因——模型无法将驾驶行为与主要环境因素关联，而是学到了噪声信号中的虚假因果关系。现有方法主要关注输入噪声（传感器噪声等），忽视了监督信号本身的噪声。

**核心矛盾**：模仿学习范式过度依赖专家轨迹，而专家数据本身包含大量噪音（驾驶风格、时间延迟、控制误差等）。

**切入角度**：与其设计复杂的网络架构，不如改变监督信号的来源——从依赖专家轨迹转向依赖感知输出（lane centerlines、agent trajectories）来指导规划。

**核心 idea**：正向约束（MTPS/STPS）确保基本驾驶行为正确 + 负向约束（NTPS）强化安全互动 = 完整因果推理框架。

## 方法详解

### 整体框架

PGS（Perception-Guided Self-Supervision）建立在标准端到端架构上，包含感知模块（输出 lane centerlines 和 dynamic objects 的 future trajectories）、运动预测+规划统一模块、三层自监督机制（MTPS、STPS、NTPS）。

### 关键设计

1. **多模态轨迹规划自监督 MTPS（目标车道选择）**

    - 功能：将多模态驾驶决策重新表述为**车道选择问题**
    - 核心思路：从感知输出的所有 lane centerlines 中，用几何滤波器选出 ego 相关的 3 条车道（当前、左、右），用 MLP 预测每条车道的选择得分（softmax 归一化）
    - 设计动机：车道信息本身就包含了所有可行的横向选择，监督信号来自 expert trajectory 终点与各车道的距离，避免了专家驾驶风格的干扰

2. **空间轨迹规划自监督 STPS（基于 lane centerline）**

    - 功能：用 lane centerline 作为纯空间参考，替代有时间噪音的专家轨迹
    - 核心思路：对 expert trajectory 的每个点查找最近的 target lane centerline 上的点，距离 $\leq w$ 的用 centerline 点替代，否则保留原点
    - 设计动机：Lane centerlines 天然连接进入/离开车道，避免了累积误差导致的车道偏离

3. **负向轨迹规划自监督 NTPS（动态物体互动）**

    - 功能：用 predicted future bounding boxes 作为负向信号，强制 ego 轨迹避免碰撞
    - 核心思路：用 Separating Axis Theorem (SAT) 检测碰撞，对碰撞时刻计算距离 margin 并最大化 $\max(0, \beta - \|Traj_{ego}^t - Traj_{obj}^t\|_2)$
    - 设计动机：正向 supervision 说明应该做什么，负向 supervision 指导不应该做什么

### 损失函数 / 训练策略

$$L'_{total} = L_{total} + w_{MTPS} L_{MTPS} + w_{STPS} L_{STPS} + w_{NTPS} L_{NTPS}$$

两阶段训练：Stage 1（6 epochs）感知学习；Stage 2（6 epochs）联合感知和规划优化。超参：$w_{MTPS}=1.0, w_{STPS}=0.3, w_{NTPS}=1.0$。

## 实验关键数据

### 主实验（Bench2Drive 基准）

| 方法 | Driving Score↑ | Success Rate↑ | Efficiency↑ |
|------|---------------|---------------|-------------|
| VAD-Base | 42.35 | 15.00% | 157.94 |
| UniAD-Base | 45.81 | 16.36% | 129.21 |
| DriveTransformer | 63.46 | 35.01% | 100.64 |
| DiffAD | 67.92 | 38.64% | - |
| **PGS (本文)** | **78.08** | **48.64%** | **181.31** |

### 消融实验（多场景能力）

| 场景 | VAD | DriveTransformer | DiffAD | **PGS** |
|------|-----|-----------------|--------|---------|
| Merging | 8.11% | 17.57% | 30% | **35%** |
| Overtaking | 24.44% | 35% | 35.55% | **73.33%** |
| Emergency Brake | 18.64% | 48.36% | 46.66% | **55%** |
| Give Way | 20% | 40% | 40% | **60%** |
| 平均 | 18.07% | 38.60% | 38.79% | **53.40%** |

### 关键发现

- 相比 VAD-Base 基线：Driving Score 提升 35.73 分（+84%），Success Rate 从 15%→48.64%（+223%）
- Overtaking 场景 73.33% 成功率远超其他方法，证明了因果推理对复杂交互场景的帮助
- 用简单的 VAD 架构就超过了更复杂的方法，证明了范式转变的价值

## 亮点与洞察

- **因果思维转变**：从"输入噪音"转向"监督噪音"是非常精准的问题诊断。同样的思路可迁移到其他模仿学习任务。
- **最小化架构改动**：不需要复杂网络，仅需改变训练方式就能大幅提升，体现了方法的优雅性。
- **闭环 vs 开环的差异**：虽然开环 L2 error 反而略高于一些方法，但闭环 Driving Score 大幅领先，这对自动驾驶研究的评估方法选择有重要启示。

## 局限与展望

- Lane centerline 的可用性假设：仅当高精地图可用且感知准确时 STPS 才有效
- 闭环评估仍在模拟器中，真实场景的表现需进一步验证
- 三个 loss 的权重比例可能需要针对不同道路场景调整
- 推理速度与实时性的权衡未深入讨论

## 相关工作与启发

- **vs ChauffeurNet**：ChauffeurNet 随机 drop ego-motion 缓解因果混淆，PGS 直接改变监督信号来源，更根本
- **vs DriveAdapter**：DriveAdapter 用 privileged information 蒸馏，PGS 不依赖额外信息，仅利用已有感知输出

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 因果混淆从监督角度的全新诠释
- 实验充分度: ⭐⭐⭐⭐⭐ Bench2Drive 闭环测试，多场景消融
- 写作质量: ⭐⭐⭐⭐ 清晰流畅，逻辑严密
- 价值: ⭐⭐⭐⭐⭐ 解决了核心问题，方法简单有效

<!-- RELATED:START -->

## 相关论文

- [Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)
- [DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving](drivedpo_policy_learning_via_safety_dpo_for_end-to-end_autonomous_driving.md)
- [RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](../../ICCV2025/autonomous_driving/unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)
- [AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)

<!-- RELATED:END -->
