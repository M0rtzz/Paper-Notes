---
title: >-
  [论文解读] APPLE: Toward General Active Perception via Reinforcement Learning
description: >-
  [ICLR 2026][机器人][active perception] 提出APPLE——一种结合强化学习与监督学习的通用主动感知框架，将主动感知建模为POMDP，奖励函数设计为RL奖励减去预测损失，梯度自然分解为策略梯度和预测损失梯度两部分…
tags:
  - "ICLR 2026"
  - "机器人"
  - "active perception"
  - "reinforcement-learning"
  - "POMDP"
  - "supervised learning"
  - "off-policy"
  - "ViViT"
  - "CrossQ"
---

# APPLE: Toward General Active Perception via Reinforcement Learning

**会议**: ICLR 2026  
**arXiv**: [2505.06182](https://arxiv.org/abs/2505.06182)  
**领域**: 主动感知 / 强化学习  
**关键词**: active perception, reinforcement-learning, POMDP, supervised learning, off-policy, ViViT, CrossQ

## 一句话总结

提出APPLE——一种结合强化学习与监督学习的通用主动感知框架，将主动感知建模为POMDP，奖励函数设计为RL奖励减去预测损失，梯度自然分解为策略梯度和预测损失梯度两部分，基于off-policy算法（SAC/CrossQ）和共享ViViT骨干网络，在5个不同任务基准上验证通用性，其中CrossQ变体无需逐任务调参且训练效率提高53%。

## 研究背景与动机

**主动感知的核心挑战**：主动感知要求智能体通过主动控制传感器（如移动相机视角、执行触觉探索）来获取信息，同时完成感知预测任务，需要同时优化"如何感知"和"如何预测"。

**现有方法的碎片化**：当前主动感知方法通常针对特定任务和传感模态设计（如主动物体识别、主动触觉感知），缺乏统一的框架适用于多种任务。

**RL与预测任务的耦合难题**：纯RL方法需要设计奖励函数来间接评估感知质量，难以直接优化预测性能；纯监督学习方法无法学习感知策略。

**On-policy方法的失败**：实验发现REINFORCE和PPO等on-policy方法在主动感知任务上完全失败，因为探索效率过低且奖励信号稀疏。

**超参数敏感性**：现有方法往往需要针对每个任务精心调整超参数，限制了实际应用的通用性。

**计算效率需求**：实际部署场景要求高效的训练和推理，需要在不牺牲性能的前提下减少计算开销。

## 方法详解

### 整体框架

APPLE把主动感知统一建模成一个POMDP：智能体在每一步依据历史观测选择控制传感器的动作（移动视角、触觉探索等），拿到新观测后再更新自己对目标的预测。它的巧妙之处在于让一套奖励同时驱动"怎么感知"和"怎么预测"两件事——奖励被定义为RL任务奖励减去当前预测损失，于是优化这个奖励的梯度天然劈成策略梯度和预测损失梯度两支，分别去训练感知策略和预测模型，整个系统跑在off-policy算法（SAC/CrossQ）和一个共享的ViViT骨干之上。

### 关键设计

**1. 奖励即"RL奖励减预测损失"：用一个标量同时管住感知与预测。** 主动感知最别扭的地方是两个目标互相纠缠：纯RL框架得手工设计代理奖励去间接评估感知质量，调起来很玄；纯监督学习又压根学不出感知策略。APPLE把奖励直接写成 $r_t = r_t^{RL} - \mathcal{L}_{pred}(\hat{y}_t, y)$，即任务奖励扣掉这一步的监督预测损失。这样做的好处在求梯度时才显现出来——整体目标的梯度可以分解为 $\nabla J = \nabla_\theta J_{RL} + \nabla_\phi \mathcal{L}_{pred}$，策略参数 $\theta$ 走标准策略梯度去优化感知行为，预测参数 $\phi$ 走监督损失被直接优化。一来不必再为"看得好不好"硬编代理奖励，二来预测模型仍然享受监督信号的直接梯度，两条路各走各的却由同一个奖励缝在一起。

**2. 用off-policy（SAC/CrossQ）替代on-policy：先让探索能跑起来。** 作者实测发现REINFORCE、PPO这类on-policy方法在所有主动感知任务上几乎全军覆没，原因是探索效率太低、奖励信号又稀疏，根本攒不出有用的轨迹。于是APPLE改用off-policy路线，靠经验回放反复利用历史样本来撑住探索，给出APPLE-SAC和APPLE-CrossQ两个变体。其中CrossQ用批归一化替掉了SAC里的target network，把"target网络更新频率"这个最折磨人的超参直接消掉，因此比SAC更鲁棒、也更省事——这正是后面它能不调参跨任务通用、训练还快53%的来源。

**3. 共享ViViT骨干：把逐步积累的观测当成一段视频来读。** 主动感知天然是一个观测一帧帧攒起来的过程，APPLE索性把历史观测序列看作视频，用Video Vision Transformer（ViViT）风格的时空注意力骨干去聚合跨时间步的信息。关键是这个骨干被策略网络和预测网络共享：一方面省下大量参数，另一方面序列化的时空建模恰好契合"观测随交互逐步丰富"的结构，让感知策略和预测都建立在同一套不断更新的时空表征上。

## 实验关键数据

### 主实验

| 任务 | APPLE-SAC | APPLE-CrossQ | 最优基线 | 基线方法 |
|------|-----------|-------------|---------|---------|
| MHSB (分类) | 94.2% | **95.1%** | 89.7% | InfoGain |
| CircleSquare (检测) | 0.82 IoU | **0.84 IoU** | 0.76 IoU | Random |
| TactileMNIST (识别) | 92.8% | **93.5%** | 88.3% | Coverage |
| Volume (估计) | 0.031 MSE | **0.028 MSE** | 0.045 MSE | Heuristic |
| Toolbox (6DoF) | 78.5% | **80.2%** | 71.4% | AcTPa |

### 消融实验

| 方法/变体 | 平均排名 | 训练时间 (相对) | 超参调整需求 |
|-----------|---------|----------------|-------------|
| APPLE-CrossQ | **1.2** | **1.0x** | 低 |
| APPLE-SAC | 1.8 | 1.53x | 中 |
| REINFORCE | 4.5 | 0.8x | 高（效果差） |
| PPO | 4.8 | 1.1x | 高（效果差） |
| 纯监督 (无RL) | 3.2 | 0.6x | 低 |

### 关键发现

1. **On-policy方法完全失败**：REINFORCE和PPO在所有5个基准上均无法学到有效策略，验证了off-policy方法对主动感知的必要性。
2. **CrossQ全面优于SAC**：跨任务平均排名更高，训练速度快53%，且无需调整target network超参。
3. **通用性验证**：同一框架和超参设定在5个差异巨大的任务上均取得SOTA或接近SOTA。
4. **RL+监督优于纯监督**：去掉RL部分后性能显著下降，说明学习感知策略的重要性。

## 亮点与洞察

1. **统一框架**：首次提出适用于多种传感模态和任务类型的通用主动感知框架。
2. **优雅的梯度分解**：奖励-损失设计使策略梯度和预测梯度自然分离，理论清晰。
3. **重要的负面结果**：on-policy方法完全失败的发现对主动感知社区有重要参考价值。
4. **实用性突出**：CrossQ变体几乎不需要调参，显著降低了实际应用门槛。

## 局限与展望

1. **离散动作空间**：当前实验均为离散动作，连续动作空间（如连续视角控制）的效果未验证。
2. **模拟环境为主**：5个基准均为模拟环境，真实物理场景的泛化性有待验证。
3. **计算资源需求**：ViViT骨干的计算开销在资源受限的嵌入式平台上可能成为瓶颈。
4. **长时间序列**：当前实验的感知步数较短（5-20步），更长序列的性能趋势未探索。

## 相关工作与启发

- **主动感知**：Bajcsy et al. (2018) 的主动感知综述；AcTPa (Liang et al., 2025) 的触觉主动感知
- **Off-policy RL**：SAC (Haarnoja et al., 2018), CrossQ (Bhatt et al., 2024) 的高效off-policy方法
- **视觉Transformer**：ViViT (Arnab et al., 2021) 的视频理解架构
- **POMDP求解**：Kaelbling et al. (1998) 的POMDP理论框架

## 评分

- 新颖性: ⭐⭐⭐⭐ 统一框架和梯度分解设计新颖
- 实验充分度: ⭐⭐⭐⭐ 5个基准覆盖多种模态和任务类型
- 写作质量: ⭐⭐⭐⭐ 框架清晰，实验详实
- 价值: ⭐⭐⭐⭐ 通用主动感知框架的实际应用潜力大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] AnyTouch 2: General Optical Tactile Representation Learning For Dynamic Tactile Perception](anytouch_2_general_optical_tactile_representation_learning_for_dynamic_tactile_p.md)
- [\[NeurIPS 2025\] Real-World Reinforcement Learning of Active Perception Behaviors](../../NeurIPS2025/robotics/real-world_reinforcement_learning_of_active_perception_behaviors.md)
- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](../../CVPR2026/robotics/sapave_active_perception_manipulation_vla_roboti.md)
- [\[ICLR 2026\] Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments](partially_equivariant_reinforcement_learning_in_symmetry-breaking_environments.md)
- [\[ICLR 2026\] MVR: Multi-view Video Reward Shaping for Reinforcement Learning](mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
