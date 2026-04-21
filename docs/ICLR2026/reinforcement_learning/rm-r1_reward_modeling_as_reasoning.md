---
title: >-
  [论文解读] RM-R1: Reward Modeling as Reasoning
description: >-
  [ICLR 2026][奖励模型] 将奖励建模重新定义为推理任务，提出RM-R1系列推理奖励模型（ReasRM），通过推理蒸馏+RL训练以及Chain-of-Rubrics（CoR）机制，在三大奖励模型基准上平均超越70B和GPT-4o模型达4.9%。
tags:
  - ICLR 2026
  - 奖励模型
  - 推理
  - Chain-of-Rubrics
  - 生成式奖励模型
  - RLVR
---

# RM-R1: Reward Modeling as Reasoning

**会议**: ICLR 2026  
**arXiv**: [2505.02387](https://arxiv.org/abs/2505.02387)  
**代码**: [GitHub](https://github.com/RM-R1-UIUC/RM-R1)  
**领域**: 强化学习  
**关键词**: 奖励模型, 推理, Chain-of-Rubrics, 生成式奖励模型, RLVR

## 一句话总结

将奖励建模重新定义为推理任务，提出RM-R1系列推理奖励模型（ReasRM），通过推理蒸馏+RL训练以及Chain-of-Rubrics（CoR）机制，在三大奖励模型基准上平均超越70B和GPT-4o模型达4.9%。

## 研究背景与动机

奖励模型是RLHF中对齐LLM的核心组件。现有方法分为两类：（1）**ScalarRM**——将RM训练为分类器输出标量分数，不透明、无推理过程；（2）**GenRM**——生成文本判断，有一定透明度但推理往往肤浅、不可靠，导致性能不如ScalarRM。

作者观察到，准确的奖励建模天然需要推理：推断评判者潜在标准、在多标准间权衡、模拟潜在后果等。Figure 1的例子清楚展示了这一点——普通指令模型过拟合数据表面模式，而推理模型能评估回复的深层影响。

核心问题：**能否将奖励建模作为推理任务来处理？**

本文提出**推理奖励模型（ReasRM）**这一新类别，强调在判断过程中使用长而连贯的推理链，并设计了两阶段训练流程（蒸馏+RL）和分类别的Chain-of-Rubrics推理策略。

## 方法详解

### 整体框架

RM-R1训练分两阶段：（1）推理蒸馏——使用o3/Claude等oracle模型合成高质量推理轨迹训练基模型；（2）强化学习——使用GRPO和可验证奖励进一步优化推理判断能力。

### 关键设计

1. **推理蒸馏（阶段一）**:

    - 功能：用oracle模型合成推理轨迹 $r^{(i)}$，构建蒸馏数据 $y_{\text{trace}}^{(i)} = r^{(i)} \oplus l^{(i)}$
    - 核心思路：最小化NLL损失 $\mathcal{L}_{\text{distill}}(\theta) = -\sum \log r_\theta(y_t|x,y_{<t})$，引导模型学会结构化推理
    - 设计动机：直接用指令模型做GenRM性能差，需要推理轨迹示范来引导。仅需约8.7K样本即可达到竞争力

2. **Chain-of-Rubrics (CoR) 推理策略**:

    - 功能：根据任务类型采用不同的推理策略
    - 核心思路：模型先将问题分类为Chat或Reasoning——Chat类型生成评分标准(rubric)后评估；Reasoning类型先自行解题再对比答案
    - 设计动机：不同类型的偏好判断关注点不同——chat关注礼貌性、安全性等文本标准，推理关注逻辑正确性和答案准确性

3. **GRPO强化学习（阶段二）**:

    - 功能：以判断正确性为奖励，使用GRPO优化策略
    - 核心思路：$\mathcal{R}(x,j|y_a,y_b) = \begin{cases} 1 & \text{if } \hat{l}=l \\ -1 & \text{otherwise} \end{cases}$
    - 设计动机：蒸馏容易过拟合特定模式，RL通过探索增强泛化能力和批判性思维

### 损失函数 / 训练策略

第一阶段用标准NLL损失训练推理蒸馏；第二阶段用GRPO优化，最大化 $\mathbb{E}[\mathcal{R}(x,j)] - \beta D_{KL}(r_\theta \| r_{\text{ref}})$，参考模型为蒸馏阶段得到的模型。值得注意的是仅使用正确性奖励（无格式奖励），因为蒸馏后模型已学会格式。

## 实验关键数据

### 主实验（三大基准平均）

| 模型 | RewardBench | RM-Bench | RMB | 平均 |
|------|------------|---------|-----|------|
| INF-ORM-70B (ScalarRM) | 95.1 | 70.9 | 70.5 | 78.8 |
| GPT-4o (GenRM) | 86.7 | 72.5 | 73.8 | 77.7 |
| Self-taught-eval-70B | 90.2 | 71.4 | 67.0 | 76.2 |
| **RM-R1-14B (ours)** | **88.9** | **81.5** | **68.5** | **79.6** |
| **RM-R1-32B (ours)** | **90.9** | **83.9** | **69.8** | **81.5** |

### 消融实验（Qwen-2.5-Instruct-32B，RewardBench）

| 方法 | Chat | Chat Hard | Safety | Reasoning | 平均 |
|------|------|-----------|--------|-----------|------|
| Instruct原模型 | 95.8 | 74.3 | 86.8 | 86.3 | 85.8 |
| +Cold Start RL | 92.5 | 81.5 | 89.7 | 94.4 | 89.5 |
| +RL+Rubrics | 93.0 | 82.5 | 90.8 | 94.2 | 90.1 |
| +RL+Rubrics+QC | 92.3 | 82.6 | 91.6 | 96.3 | 90.8 |
| **RM-R1 (完整)** | **95.3** | **83.1** | **91.9** | **95.2** | **91.4** |

### 关键发现
- RM-R1在RM-Bench上超越之前最佳8.7%，在数学和代码上分别达91.8%和74.1%
- 推理能力对奖励建模至关重要——蒸馏提供基础，RL进一步增强泛化
- 模型规模scaling效果好——7B到32B呈现近似线性的相对提升
- 推理长度scaling也有效——更长的推理链带来更好的判断性能

## 亮点与洞察

- **将RM与推理深度结合**：首次系统性地将长链推理引入奖励建模，建立了ReasRM这一新类别
- **数据效率极高**：仅8.7K样本蒸馏即可达到竞争力，远少于DeepSeek-Distilled的800K
- **CoR设计巧妙**：区分chat和reasoning的不同评判策略，反映了人类打分的实际认知过程
- **SFT vs RL对比有洞察**：Table 3显示推理训练（RL）一致优于SFT，即使在同一蒸馏数据上

## 局限与展望

- CoR的分类（Chat vs Reasoning）可能过于简化，更细粒度的任务分类可能更优
- 依赖oracle模型（o3/Claude）生成蒸馏数据，增加了成本
- 当前奖励设计仅使用二元正确性（±1），更细粒度的奖励信号可能进一步提升

## 相关工作与启发

- DeepSeek-GRM系列是直接竞争者，但未开源且依赖更多数据
- JudgeLRM也是ReasRM但性能明显落后，凸显了训练方案的重要性
- 启示：在RM训练中，"如何推理"比"看多少数据"更重要

## 评分
- 新颖性: ⭐⭐⭐⭐ 将推理引入RM的方向很好，但蒸馏+RL框架较为标准
- 实验充分度: ⭐⭐⭐⭐⭐ 三大基准、详细消融、scaling分析、case study
- 写作质量: ⭐⭐⭐⭐ 结构清晰，Figure 1动机例子说明力强
- 价值: ⭐⭐⭐⭐⭐ 建立了ReasRM新范式，开源代码和模型推动社区发展

<!-- RELATED:START -->

## 相关论文

- [UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)
- [MMhops-R1: Multimodal Multi-hop Reasoning](../../AAAI2026/reinforcement_learning/mmhops-r1_multimodal_multi-hop_reasoning.md)
- [MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning](../../CVPR2026/reinforcement_learning/msrl_scaling_generative_multimodal_reward_modeling.md)
- [TextShield-R1: Reinforced Reasoning for Tampered Text Detection](../../AAAI2026/reinforcement_learning/textshield-r1_reinforced_reasoning_for_tampered_text_detection.md)
- [Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](exploration_vs_exploitation_rethinking_rlvr_through_clipping_entropy_and_spuriou.md)

<!-- RELATED:END -->
