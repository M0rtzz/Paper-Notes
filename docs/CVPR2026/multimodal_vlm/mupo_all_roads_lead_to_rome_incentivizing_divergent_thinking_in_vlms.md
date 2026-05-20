---
title: >-
  [论文解读] MUPO: All Roads Lead to Rome - Incentivizing Divergent Thinking in Vision-Language Models
description: >-
  [CVPR 2026][多模态VLM][强化学习] MUPO 揭示了 GRPO 训练导致推理多样性坍缩的问题——模型过早收敛到少数推理策略而丢弃大多数替代方案。通过将响应分组进行局部化优势估计并引入多样性奖励，MUPO 激励 VLM 保持发散思维，在多个推理基准上提升 2-7%。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "强化学习"
  - "GRPO"
  - "发散思维"
  - "推理多样性"
  - "视觉语言模型"
---

# MUPO: All Roads Lead to Rome - Incentivizing Divergent Thinking in Vision-Language Models

**会议**: CVPR 2026  
**arXiv**: [2604.00479](https://arxiv.org/abs/2604.00479)  
**代码**: [https://xytian1008.github.io/MUPO/](https://xytian1008.github.io/MUPO/)  
**领域**: LLM推理 / 多模态VLM  
**关键词**: 强化学习, GRPO, 发散思维, 推理多样性, 视觉语言模型

## 一句话总结

MUPO 揭示了 GRPO 训练导致推理多样性坍缩的问题——模型过早收敛到少数推理策略而丢弃大多数替代方案。通过将响应分组进行局部化优势估计并引入多样性奖励，MUPO 激励 VLM 保持发散思维，在多个推理基准上提升 2-7%。

## 研究背景与动机

RL（特别是 GRPO）已成为增强 VLM 推理能力的主流方法。但作者发现了一个关键矛盾：

**RL 模型深而窄，Base 模型浅而广**：RL 模型在单次尝试时准确率更高（推理更深入），但给多次尝试机会时，Base 模型能解决更多不同的问题（策略更多样）。例如几何题，RL 模型总是用方程求解（容易出逻辑错误），而 Base 模型有时会用验证式策略简洁地得到答案。

**多样性坍缩**：通过追踪 GRPO 训练过程，发现推理多样性在训练早期就急剧下降到可忽略水平。模型迅速收敛到少数"占优"策略，丢弃了大量潜在替代路径。这导致：(1) 利用优先于探索，陷入局部最优；(2) 扩展性差，收敛的推理无法覆盖广泛的问题类型。

## 方法详解

### 整体框架

MUPO 是 GRPO 的即插即用替代。将模型的多个响应分成多个组，组内进行局部化优势估计，组间引入多样性奖励，鼓励不同组代表不同的推理策略。

### 关键设计

1. **多组策略优化**:

    - 功能：保持推理策略的多样性，避免所有响应收敛到同一策略
    - 核心思路：将 GRPO 中全局计算优势的方式改为分组局部化优势估计。将 $K$ 个响应分成 $G$ 组，每组独立计算优势值。这使得每个组可以独立维护自己的"最优策略"，而不被全局主导策略淹没。直觉上，每个组是一种推理策略的独立实现
    - 设计动机：GRPO 的全局优势计算导致少数高奖励策略获得极大的优势值，抑制了其他策略的更新信号

2. **多样性奖励**:

    - 功能：促进组间推理策略的分离
    - 核心思路：在准确率和格式奖励之外，加入多样性奖励——衡量不同组之间的推理嵌入距离。鼓励组间距离越大越好，使不同组代表真正不同的推理路径
    - 设计动机：仅分组但不鼓励差异化，组可能仍然收敛到相似策略。多样性奖励提供了分离的显式激励

3. **深度+广度的统一**:

    - 功能：让模型同时具备单路径的深度推理和多路径的广度覆盖
    - 核心思路：组内优化保证每条策略都被充分优化（深度），组间多样性保证维持多种策略（广度）。这与人类解题类似——给多次尝试时会从不同角度思考，每个角度都认真推理
    - 设计动机：这正是发散思维的本质——不是简单生成不同答案，而是用不同方法思考同一个问题

### 损失函数 / 训练策略

标准 RL 训练流程，MUPO 替代 GRPO 作为策略优化算法。准确率奖励 + 格式奖励 + 多样性奖励。

## 实验关键数据

### 主实验

| 模型 | MathVerse | LogicVista | WeMath | HallusionBench | 平均提升 |
|------|-----------|-----------|--------|----------------|---------|
| GRPO 基线 | 基线 | 基线 | 基线 | 基线 | — |
| **MUPO-Thinker-7B** | +提升 | +提升 | +提升 | +提升 | **2~7%** |

在多个推理基准上一致提升 2-7%，建立新 SOTA。

### 消融实验

| 配置 | acc@1 | acc@4 | 多样性 | 说明 |
|------|-------|-------|--------|------|
| GRPO | 高 | 有限提升 | 低（坍缩） | 深而窄 |
| Base 模型 | 较低 | 大幅提升 | 高 | 浅而广 |
| MUPO | **最高** | **最高** | **高** | 深且广 |

### 关键发现

- acc@k 分析揭示了 RL 和 Base 模型的根本差异：k=1 时 RL 赢，k>1 时 Base 赢。这说明多样性本身就是一种能力
- GRPO 的多样性坍缩在训练极早期就发生（<10%训练步），说明这是算法层面的问题而非训练不足
- 多样性与准确率呈正相关——更多样的推理策略提高了找到正确答案的概率

## 亮点与洞察

- **发散思维 vs 收敛思维**：将心理学的发散/收敛思维概念引入 RL 训练，提供了理解 GRPO 局限性的新视角
- **多样性坍缩的诊断**：用嵌入距离量化推理多样性并追踪训练动态，是一种可复用的分析方法
- **acc@k 作为补充指标**：不仅看单次准确率，还看多次尝试能解决多少问题，这对评估推理模型更全面
- **即插即用替代 GRPO**：MUPO 可以直接替换 GRPO，不需要修改其他训练流程

## 局限与展望

- 分组数量 G 是超参数，最优值可能因任务而异
- 多样性奖励的权重需要调节，过大可能牺牲单路径准确率
- 当前主要验证在数学/逻辑推理任务，在其他任务（如开放式生成）的效果待验证
- 未来可探索自适应分组和动态多样性权重

## 相关工作与启发

- **vs GRPO/DeepSeekMath**: GRPO 追求深度推理但牺牲了广度，MUPO 同时保持两者
- **vs DAPO/GVPO**: 这些方法从采样角度优化 GRPO，但没有解决多样性坍缩问题
- **vs Best-of-N/Self-Consistency**: 这些是推理时扩展策略，MUPO 是训练时策略，两者可组合

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 对GRPO多样性坍缩的诊断和发散思维的引入非常有洞察力
- 实验充分度: ⭐⭐⭐⭐⭐ 行为分析+训练动态+多基准验证，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 分析深入，图示清晰，逻辑连贯
- 价值: ⭐⭐⭐⭐⭐ 对RL训练推理模型的方法论有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](../../ICLR2026/multimodal_vlm/vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)
- [\[CVPR 2026\] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)
- [\[CVPR 2026\] Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking](circuit_tracing_in_vision-language_models_understanding_the_internal_mechanisms_.md)
- [\[CVPR 2026\] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](cropvlm_learning_to_zoom_for_fine_grained_vision_language_perception.md)
- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](understanding_task_transfer_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
