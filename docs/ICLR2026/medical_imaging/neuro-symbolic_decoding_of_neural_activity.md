---
title: >-
  [论文解读] Neuro-Symbolic Decoding of Neural Activity
description: >-
  [ICLR 2026][医学图像][fMRI解码] 提出 NEURONA，一个神经符号框架用于 fMRI 解码和概念基础，通过将视觉场景分解为符号程序（概念的逻辑组合），在 fMRI 问答任务上显著优于端到端神经解码和线性模型。
tags:
  - "ICLR 2026"
  - "医学图像"
  - "fMRI解码"
  - "神经符号"
  - "概念基础"
  - "思维语言假说"
  - "视觉问答"
---

# Neuro-Symbolic Decoding of Neural Activity

**会议**: ICLR 2026  
**arXiv**: [2603.03343](https://arxiv.org/abs/2603.03343)  
**代码**: 无  
**领域**: 神经科学 / 多模态  
**关键词**: fMRI解码, 神经符号, 概念基础, 思维语言假说, 视觉问答

## 一句话总结
提出 NEURONA，一个神经符号框架用于 fMRI 解码和概念基础，通过将视觉场景分解为符号程序（概念的逻辑组合），在 fMRI 问答任务上显著优于端到端神经解码和线性模型。

## 研究背景与动机

**领域现状**：认知科学的"思维语言"假说认为人类思维以结构化、组合性的表征运作。fMRI 神经解码在过去几十年取得了大量进展，从线性映射到深度学习方法。

**现有痛点**：现有神经解码方法要么使用简单线性模型（可解释但表达力不足），要么使用端到端神经网络（强大但黑盒）。两者都无法很好地捕捉概念间的组合关系和逻辑结构。

**核心矛盾**：fMRI 信号编码了丰富的视觉概念，但直接从 fMRI 预测自然语言答案跨越了太大的语义鸿沟——需要同时理解场景结构、概念语义和问题意图。

**本文目标** 如何从 fMRI 活动中解码结构化的概念表征，而不是直接预测端到端答案？

**切入角度**：利用图像和视频 fMRI 数据集中自然编码的复合概念，将解码过程分解为符号程序执行。

**核心 idea**：将 fMRI 解码分解为"fMRI→概念检测→符号程序执行→答案"的神经符号流水线，比端到端方法更准确且更可解释。

## 方法详解

### 整体框架
NEURONA 不直接从 fMRI 端到端预测答案，而是先把大脑活动落到一组离散的视觉概念上，再用符号程序在这些概念上做逻辑推理得到答案。整个流水线串起"fMRI→概念检测→程序执行→答案"三个阶段，其中只有概念检测需要训练，问题编译和程序执行都是零样本的。

### 关键设计

**1. fMRI 概念基础模块：把高维体素信号压成离散概念集**

直接从几千维的 fMRI 体素去预测自然语言答案，语义鸿沟太大。NEURONA 先训练一组线性探针 $\{f_c : \mathbb{R}^V \to [0,1]\}_{c \in \mathcal{C}}$，每个探针对应一个视觉概念 $c$（如"狗""红色""跑"），从体素激活 $\mathbf{x} \in \mathbb{R}^V$ 直接读出该概念出现的概率。训练标签并非人工标注，而是用预训练视觉-语言模型（CLIP）对配对刺激图像做零样本检测得到的伪标签，配合二元交叉熵损失优化。这一步把高维连续信号离散化成一个概念集合，让后续推理只需在有限的符号空间内进行，复杂度大幅下降；这也是全流水线唯一需要训练的部分。

**2. 问题到程序的编译：用 LLM 把自然语言问题翻成可执行逻辑**

要在概念集上做推理，先得把问题变成机器能跑的程序。NEURONA 用 LLM（如 GPT-4）把自然语言问题编译成一段领域特定语言（DSL）程序，DSL 的原语包括 `detect(concept)`、`AND/OR/NOT` 逻辑运算、`count`、`spatial_relation` 等，例如 "Is the dog running?" 被编译为 `AND(detect(dog), detect(running))`。这种符号化表示天然带来组合泛化：遇到训练时没见过的概念组合，不必重新训练任何模型，只要生成对应的新程序即可。

**3. 符号程序执行引擎：在概念概率上做确定性推理**

最后一步由一个程序解释器递归执行编译出的程序：`detect(c)` 查询概念基础模块的输出 $f_c(\mathbf{x})$，逻辑运算直接作用在概率值上（`AND` 取最小、`OR` 取最大、`NOT` 取 $1-p$），最终对结果阈值化得到答案。整个执行过程是确定性的，因此可以精确追溯每个答案是由哪几个概念检测、经过哪些逻辑步骤得出的——可解释性不是事后解释，而是推理机制本身自带的。

## 实验关键数据

### 主实验

| 方法 | BOLD5000-QA Overall | CNeuroMod-QA Overall |
|------|-------------------|---------------------|
| Linear | 0.4692 | - |
| End-to-end Neural | ~0.50 | ~0.45 |
| NEURONA | **显著更高** | **显著更高** |

### 消融实验

| 配置 | 准确率 | 说明 |
|------|--------|------|
| NEURONA (完整) | 最优 | 神经符号三阶段 |
| 仅线性解码 | 0.47 | 无组合推理能力 |
| 仅端到端 | ~0.50 | 黑盒，缺乏结构 |
| 无 CLIP 伪标签 | 下降 | 概念基础质量降低 |

### 关键发现
- 神经符号方法在 fMRI QA 上显著优于纯线性和纯端到端方法
- 动作类和位置类问题的提升尤为明显，说明这些概念在 fMRI 中有清晰的神经表征
- 概念基础模块的准确性是整个系统的瓶颈
- 符号程序提供了完全的可解释性——可以追溯每个答案的推理过程

## 亮点与洞察
- **思维语言假说的计算验证**：通过神经符号方法展示概念组合确实更好地映射 fMRI 活动，间接支持了认知科学假说
- **可解释性的免费午餐**：符号程序不仅提升性能还提供完全透明的推理，对神经科学研究有重要价值

## 局限与展望
- 概念库的覆盖范围限制了可回答的问题类型
- fMRI 数据集规模较小（几十到几百个样本），泛化性存疑
- 伪标签质量依赖 CLIP 的零样本能力，对偏离自然图像的刺激可能不可靠
- DSL 设计需要领域知识，不同任务需不同的原语集

## 相关工作与启发
- **vs BrainBERT/Mind-Vis**: 端到端解码方法直接从 fMRI 生成文本/图像，但缺乏结构化推理能力
- **vs Neurosymbolic AI (VQA)**: 类似 NS-VQA 将视觉问答分解为感知+推理，NEURONA 将此思路引入 fMRI 领域

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将神经符号方法应用于 fMRI 解码，连接认知科学和 AI
- 实验充分度: ⭐⭐⭐ 数据集较小，定量比较有限
- 写作质量: ⭐⭐⭐⭐ 跨学科但可读性好
- 价值: ⭐⭐⭐⭐ 开辟了 fMRI 解码的新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification](../../NeurIPS2025/medical_imaging/firegnn_neuro-symbolic_graph_neural_networks_with_trainable_fuzzy_rules_for_inte.md)
- [\[ICLR 2026\] Towards Interpretable Visual Decoding with Attention to Brain Representations](towards_interpretable_visual_decoding_with_attention_to_brain_representations.md)
- [\[ICLR 2026\] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)
- [\[NeurIPS 2025\] Generalizable, Real-Time Neural Decoding with Hybrid State-Space Models](../../NeurIPS2025/medical_imaging/generalizable_real-time_neural_decoding_with_hybrid_state-space_models.md)
- [\[AAAI 2026\] Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks](../../AAAI2026/medical_imaging/neural_bandit_based_optimal_llm_selection_for_a_pipeline_of_tasks.md)

</div>

<!-- RELATED:END -->
