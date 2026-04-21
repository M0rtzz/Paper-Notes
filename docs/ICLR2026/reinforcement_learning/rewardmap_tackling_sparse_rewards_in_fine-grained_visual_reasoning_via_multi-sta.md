---
title: >-
  [论文解读] RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning
description: >-
  [ICLR 2026][多模态大模型] 提出RewardMap框架，通过难度感知的细节奖励设计和从简单感知到复杂推理的多阶段RL课程学习策略，克服细粒度视觉推理中的稀疏奖励问题。
tags:
  - ICLR 2026
  - 多模态大模型
  - 视觉推理
  - 稀疏奖励
  - 多阶段RL
  - 地铁路线规划
---

# RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning

**会议**: ICLR 2026  
**arXiv**: [2510.02240](https://arxiv.org/abs/2510.02240)  
**代码**: [项目页面](https://fscdc.github.io/RewardMap)  
**领域**: 强化学习  
**关键词**: 多模态大模型, 视觉推理, 稀疏奖励, 多阶段RL, 地铁路线规划

## 一句话总结

提出RewardMap框架，通过难度感知的细节奖励设计和从简单感知到复杂推理的多阶段RL课程学习策略，克服细粒度视觉推理中的稀疏奖励问题。

## 研究背景与动机

细粒度视觉推理（如地铁路线规划）是多模态大模型（MLLM）的核心挑战。ReasonMap基准揭示了即使先进的MLLM在结构化、信息密集的视觉场景中也难以进行空间推理。

将标准RL方法（如GRPO）直接应用于此类复杂任务面临**稀疏奖励瓶颈**：
- 成功信号仅在长推理链末端给出（最终答案对/错）
- 任务难度进一步放大稀疏性——大多数采样得到的奖励 $r_i \approx 0$
- 在GRPO中，当所有采样都失败时，组内优势 $\hat{A}_i$ 趋近零，梯度信号微弱，收敛困难

传统SFT虽提供密集监督，但无法赋予模型长链决策的推理能力。核心矛盾是任务复杂度与监督信号密度的错配。

本文的切入点：（1）构建ReasonMap-Plus数据集作为密集奖励冷启动源；（2）设计从易到难的多阶段RL训练，从感知逐步过渡到推理。

## 方法详解

### 整体框架

RewardMap包含两个核心组件：（1）难度感知的奖励设计，在格式和正确性奖励基础上增加细节奖励；（2）多阶段GRPO训练课程，从简单VQA到复杂路线规划逐步推进。

### 关键设计

1. **ReasonMap-Plus数据集构建**:

    - 功能：构建4018个VQA问题覆盖5种扩展题型，30个城市13个国家
    - 核心思路：设计全局计数、局部计数、判断题3大类问题，利用Metro Data自动生成答案
    - 设计动机：VQA题型简单、奖励密集，适合作为RL冷启动，训练模型的基础视觉理解能力

2. **难度感知的细节奖励**:

    - 功能：在正确性奖励外增加部分分数奖励
    - 核心思路：$R = W_{\text{difficulty}}(R_{\text{format}} + R_{\text{correctness}} + \alpha \times R_{\text{detail}})$
    - 细节奖励对起点/终点、路线名、换乘站、路段数分别给予奖惩
    - 难度权重 $W_{\text{difficulty}} = W_{\text{map}} + W_{\text{question}}$，综合地图难度和换乘次数
    - 设计动机：缓解规划任务中的稀疏奖励，即使最终答案错误也能从部分正确的信息中学习

3. **多阶段GRPO课程学习**:

    - 功能：按全局课程原则将训练分为多阶段
    - 核心思路：判断题 → 计数题 → 规划题（视觉理解 → 视觉推理）。每阶段内随机打乱样本
    - 设计动机：（1）低层级任务奖励密集，支持有效冷启动；（2）逐步桥接感知和推理，避免直接面对困难任务时的训练崩溃；（3）局部随机性防止过拟合固定课程轨迹

### 损失函数 / 训练策略

使用GRPO的标准策略梯度目标，以组相对优势驱动更新。关键不同在于奖励函数的设计（三层奖励+难度加权）和数据调度策略（多阶段课程）。冷启动阶段直接使用RL而非SFT，确保奖励信号与任务目标从一开始就对齐。

## 实验关键数据

### 主实验（Qwen2.5-VL-7B-Instruct）

| 方法 | ReasonMap加权准确率(S/L) | ReasonMap-Plus加权准确率 |
|------|-------------------------|------------------------|
| 基础模型 | 13.28%/7.12% | 44.21% |
| +RL (GRPO) | 26.22%/26.04% | 44.64% |
| +RL (REINFORCE++) | 27.17%/27.60% | - |
| +RewardMap（完整） | **最优** | **最优** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅格式+正确性奖励 | 基线性能 | 稀疏奖励下学习困难 |
| +细节奖励 | 显著提升 | 部分分数缓解稀疏性 |
| +难度权重 | 进一步提升 | 难题贡献更多学习信号 |
| +多阶段课程 | 最佳性能 | 冷启动策略有效 |

### 关键发现
- RewardMap训练的模型在6个外部基准上平均提升3.47%，说明能力泛化性好
- 使用RL冷启动优于SFT冷启动，避免了SFT导致的过拟合和认知僵化
- 参考模型对比中，GPT-5在ReasonMap上达到59.98%/62.50%，显示出该任务的极高难度
- Seed1.5-VL和GPT-4o在ReasonMap-Plus上分别达到73.58%和64.42%

## 亮点与洞察

- **问题定义有价值**：地铁路线规划是MLLM视觉推理的天然测试场，任务本身兼具实用性和科学价值
- **RL替代SFT做冷启动**是一个有洞察力的设计选择，避免了奖励与损失函数的错配
- **细节奖励设计巧妙**：利用规划任务的结构性（起点、终点、换乘站等可独立验证）分解奖励

## 局限与展望

- 细节奖励的设计依赖于任务特定的结构信息，泛化到其他视觉推理任务需要重新设计
- 难度权重的超参数（$\gamma_e, \gamma_m, \gamma_h, \beta_0, \beta_1$）需要预设
- 当前仅在Qwen2.5-VL模型族上验证，对其他架构的泛化性未知

## 相关工作与启发

- ReasonMap（Feng et al., 2025b）是本文的基准和数据基础
- GRPO（Shao et al., 2024）提供了RL优化框架
- 课程RL（Parashar et al., 2025）的从易到难策略启发了多阶段设计
- 启示：对于复杂推理任务，奖励工程（reward engineering）可能比算法创新更为关键

## 评分
- 新颖性: ⭐⭐⭐⭐ 多阶段RL冷启动替代SFT的思路有新意，但各组件较标准
- 实验充分度: ⭐⭐⭐⭐ 多基准验证包括外部泛化，有消融研究
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，框架图示清晰
- 价值: ⭐⭐⭐⭐ 为MLLM视觉推理的RL训练提供了实用方案

<!-- RELATED:START -->

## 相关论文

- [DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition](dive-k_differential_visual_reasoning_for_fine-grained_image_recognition.md)
- [LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning](../../CVPR2026/reinforcement_learning/msrl_scaling_generative_multimodal_reward_modeling_via_multi-stage_reinforcement.md)
- [MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding](mergemix_a_unified_augmentation_paradigm_for_visual_and_multi-modal_understandin.md)
- [Learning from Synthetic Data Improves Multi-hop Reasoning](learning_from_synthetic_data_improves_multi-hop_reasoning.md)

<!-- RELATED:END -->
