---
title: >-
  [论文解读] When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners
description: >-
  [NeurIPS 2025][多语言推理] 受认知神经科学启发（人脑的推理与语言处理相对独立），在 LLM 的激活空间中识别并消除语言特定成分，实现语言与推理的解耦，从而在免训练条件下一致性地提升多语言推理性能。
tags:
  - NeurIPS 2025
  - 多语言推理
  - 语言-推理解耦
  - 因果干预
  - 激活空间
  - 免训练
---

# When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners

**会议**: NeurIPS 2025  
**arXiv**: [2505.15257](https://arxiv.org/abs/2505.15257)  
**代码**: [GitHub](https://github.com/MuyuenLP/Language-Reasoning-Disentangle)  
**领域**: 强化学习  
**关键词**: 多语言推理, 语言-推理解耦, 因果干预, 激活空间, 免训练

## 一句话总结

受认知神经科学启发（人脑的推理与语言处理相对独立），在 LLM 的激活空间中识别并消除语言特定成分，实现语言与推理的解耦，从而在免训练条件下一致性地提升多语言推理性能。

## 研究背景与动机

推理型 LLM（如 DeepSeek-R1、QwQ）在英语/中文等高资源语言上表现出色，但在中低资源语言上的推理能力显著不足。这一**多语言推理鸿沟**限制了 LLM 的全球适用性，加剧了语言间 AI 能力的不平等。

核心问题是：多语言推理性能差距的底层原因是什么？

本文的假设来自**认知神经科学**的发现——人脑中负责语言理解/生产的区域在推理任务中**几乎不活跃**，人类语言进化的目的是沟通而非推理。受此启发，作者假设 LLM 中的推理能力和语言处理同样是**可分离**的。如果语言特定的表征成分干扰了推理过程，那么**移除**这些成分应该能提升非英语的推理表现——让推理能力从高资源语言"自由迁移"到低资源语言。

这一假设的妙处在于：如果成立，就意味着无需昂贵的多语言后训练，只需在推理时做简单的表征干预即可增强跨语言推理。

## 方法详解

### 整体框架

方法分为三步：(1) 识别语言特定子空间；(2) 在推理时通过投影消除语言特定成分；(3) 在中间层消除但在高层保留以维持语言输出保真度。

### 关键设计

1. **语言特定子空间识别**：对于处理 $L$ 种语言输入的模型，在每一层计算各语言的平均表征 $\boldsymbol{m}_l = \frac{1}{n}\sum_{i=1}^{n}\boldsymbol{e}_l^i$（取最终 token 的嵌入），拼接成矩阵 $\boldsymbol{M} \in \mathbb{R}^{d \times L}$。通过正交分解将 $\boldsymbol{M}$ 拆分为**语言无关子空间** $\boldsymbol{M}_a$（跨语言共享语义）和**语言特定子空间** $\boldsymbol{M}_s$（编码语言差异）。分解通过 SVD 高效求解，目标为：

$$\min_{\boldsymbol{M}_a, \boldsymbol{M}_s, \boldsymbol{\Gamma}} \|\boldsymbol{M} - \boldsymbol{M}_a \mathbbm{1}^\top - \boldsymbol{M}_s \boldsymbol{\Gamma}^\top\|_F^2 \quad \text{s.t.} \quad \text{Span}(\boldsymbol{M}_a) \perp \text{Span}(\boldsymbol{M}_s)$$

2. **激活消融（Activation Ablation）**：推理时，对每个隐藏表征 $\boldsymbol{h}$ 投影去除其沿语言特定子空间 $\boldsymbol{M}_s$ 的分量：

$$\hat{\boldsymbol{h}} = \boldsymbol{h} - \lambda \boldsymbol{M}_s^\top \boldsymbol{M}_s \boldsymbol{h}$$

其中 $\lambda$ 控制消融强度。这去除了语言特异性变化，让剩余表征 $\hat{\boldsymbol{h}}$ 更好地反映语言无关的推理过程。设计动机直接来自假设验证——如果语言成分确实干扰推理，投影去除就应该提升性能。

3. **分层干预策略**：实验发现全层干预会破坏输出语言的保真度（模型倾向于输出英语）。关键洞察是**高层的语言特定信号对维持目标语言输出至关重要**。因此采用"中间层消除 + 高层保留"的策略——在中低层去除语言特定成分以释放推理能力迁移，在高层重新注入以维持语言输出一致性。

### 验证实验

通过两个验证确认被去除的成分确实是语言信号：(1) PCA 可视化显示，消融后非英语表征向英语簇收敛；(2) 语言保真度指标显示，消融强度越大，模型越倾向于输出英语（即使输入为其他语言）。

## 实验关键数据

### 主实验

在 MGSM（数学推理）上，11 种语言的平均准确率：

| 模型 | 原始 AVG | +解耦 AVG | 提升 | 说明 |
|------|---------|----------|------|------|
| Qwen-2.5-3B-Instruct | 56.36 | **58.51** | +2.15 | 非推理小模型 |
| Qwen-2.5-7B-Instruct | 69.82 | **70.76** | +0.94 | 非推理模型 |
| Qwen-3-8B-Thinking | 84.15 | **85.42** | +1.27 | 推理型模型也有效 |
| R1-Distill-Qwen-14B | 65.24 | **67.24** | +2.00 | 蒸馏推理模型 |
| QwQ-32B | 83.31 | **84.62** | +1.31 | 32B 大推理模型也受益 |

在 XWinograd（常识推理）和 M-MMLU（知识密集 QA）上提升更为显著。

### 跨基准详细对比（XWinograd / M-MMLU）

| 模型 | XWinograd 原始 | XWinograd +解耦 | M-MMLU 原始 | M-MMLU +解耦 |
|------|--------------|----------------|-------------|--------------|
| Qwen-2.5-3B | 65.07 | **70.18** (+5.11) | 52.62 | **56.63** (+4.01) |
| Qwen-2.5-7B | 68.13 | **74.00** (+5.87) | 61.25 | **63.88** (+2.63) |
| Qwen-3-8B-Think | 85.14 | **87.99** (+2.85) | 73.94 | **76.19** (+2.25) |

### 关键发现

- **一致性提升**：在所有 10 个测试模型（推理型+通用型）、3 个基准、11 种语言上都观测到提升，证明假设的普适性。
- **低资源语言受益更多**：如 Swahili 在多个模型上准确率提升超过 10%，部分情况下性能翻倍以上。
- **免训练 vs 后训练**：该免训练干预方法在效果上可媲美甚至超越 SFT/RL 等多语言后训练方法，但计算开销几乎为零。
- **语言干扰强度 vs 推理准确率**：分析显示隐藏状态中语言特定信号越强，推理准确率越低——直接支持"语言信号干扰推理"的因果机制。

## 亮点与洞察

- **跨学科灵感**：从认知神经科学（语言-推理独立性）到 AI 的假设迁移，优雅且有说服力。
- **免训练方法**：无需微调、无需额外数据，仅在推理时做矩阵投影，实际部署成本极低。
- "Less is More"的哲学——去除语言信号反而增强推理，暗示当前 LLM 中语言和推理表征的纠缠可能是制约多语言性能的深层原因。
- 分层干预策略的发现（中层去除+高层保留）为理解 Transformer 各层的功能分工提供了实验证据。

## 局限与展望

- 虽然中文和英语都是高资源语言，但中文表征也向英语收敛——作者推测与预训练数据中英语占比有关，但具体机制未明。
- 消融强度 $\lambda$ 和干预层范围目前需要对不同模型调参。
- 方法假设语言和推理可以线性分离（通过 SVD），对于更复杂的非线性纠缠可能不够。
- 对极端低资源语言（训练数据很少的语言）效果有限（如 Swahili 在部分模型上提升有限）。

## 相关工作与启发

- 与多语言表征对齐研究（语言中立表征、跨语言迁移）形成理论连接。
- 和 representation engineering/activation steering 方向相关，但目标不同（这里是提升推理，而非控制行为）。
- 可以与多语言后训练方法（SFT/RL）结合使用，作为互补增强策略。
- 启发：LLM 内部可能存在更多可解耦的"功能子空间"，如知识 vs 创造力、事实 vs 风格。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 假设来源独特（认知神经科学），验证方式优雅（因果干预）
- **实验充分度**: ⭐⭐⭐⭐⭐ — 10个模型×11种语言×3个基准，规模庞大且结论一致
- **写作质量**: ⭐⭐⭐⭐⭐ — 故事线清晰，从假设到验证到分析层层递进
- **价值**: ⭐⭐⭐⭐⭐ — 免训练提升多语言推理，理论洞察+实践价值兼备

<!-- RELATED:START -->

## 相关论文

- [Checklists Are Better Than Reward Models For Aligning Language Models](checklists_are_better_than_reward_models_for_aligning_langua.md)
- [RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning](rl_tango_reinforcing_generator_and_verifier_together_for_lan.md)
- [Horizon Reduction Makes RL Scalable](horizon_reduction_makes_rl_scalable.md)
- [Training Language Models to Reason Efficiently](training_language_models_to_reason_efficiently.md)
- [Less is More: Clustered Cross-Covariance Control for Offline RL](../../ICLR2026/reinforcement_learning/less_is_more_clustered_cross-covariance_control_for_offline_rl.md)

<!-- RELATED:END -->
