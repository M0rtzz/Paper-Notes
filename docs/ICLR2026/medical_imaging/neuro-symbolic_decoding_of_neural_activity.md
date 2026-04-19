---
title: >-
  [论文解读] Neuro-Symbolic Decoding of Neural Activity
description: >-
  [ICLR 2026][医学图像][fMRI解码] 提出 NEURONA，一个神经符号框架用于 fMRI 解码和概念基础，通过将视觉场景分解为符号程序（概念的逻辑组合），在 fMRI 问答任务上显著优于端到端神经解码和线性模型。
tags:
  - ICLR 2026
  - 医学图像
  - fMRI解码
  - 神经符号
  - 概念基础
  - 思维语言假说
  - 视觉问答
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
NEURONA 采用三阶段流水线：(1) 概念基础：从 fMRI 活动中检测场景包含的视觉概念（如"狗"、"红色"、"跑"）；(2) 程序合成：将自然语言问题转化为概念上的逻辑程序（如 "Is the dog running?" → AND(detect(dog), detect(running))）；(3) 程序执行：在检测到的概念集上执行程序得到答案。

### 关键设计

1. **fMRI 概念基础模块 (Concept Grounding)**:

    - 做什么：从 fMRI 体素模式中检测场景包含的视觉概念
    - 核心思路：训练线性探针集合 $\{f_c : \mathbb{R}^V \to [0,1]\}_{c \in \mathcal{C}}$，每个探针 $f_c$ 从 fMRI 活动 $\mathbf{x} \in \mathbb{R}^V$ 中预测概念 $c$ 的存在概率。使用预训练视觉-语言模型（如 CLIP）的零样本预测作为伪标签来训练
    - 设计动机：将高维 fMRI 信号转化为离散概念集合，大幅降低后续推理的复杂度

2. **问题到程序的编译 (Question-to-Program Compilation)**:

    - 做什么：将自然语言问题转化为可执行的符号程序
    - 核心思路：使用 LLM（如 GPT-4）将问题编译为领域特定语言（DSL）的程序。DSL 包含 detect(concept)、AND/OR/NOT 逻辑运算、count、spatial_relation 等原语
    - 设计动机：符号程序提供了组合泛化能力——新概念组合无需重新训练，只需新的程序

3. **符号程序执行引擎**:

    - 做什么：在概念检测结果上执行符号程序得到最终答案
    - 核心思路：程序解释器递归执行每个原语：detect(c) 查询概念基础模块的输出 $f_c(\mathbf{x})$；逻辑运算对概率值操作（AND 取最小、OR 取最大）；最终阈值化得到答案
    - 设计动机：确定性执行保证可解释性和可追溯性——可以精确知道哪些概念检测影响了最终答案

### 损失函数 / 训练策略
概念基础模块使用二元交叉熵损失训练，伪标签来自 CLIP 零样本检测。整个流水线只有概念基础模块需要训练，程序合成和执行是零样本的。

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

## 相关论文

- [FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification](../../NeurIPS2025/medical_imaging/firegnn_neuro-symbolic_graph_neural_networks_with_trainable_fuzzy_rules_for_inte.md)
- [Intrinsic Lorentz Neural Network](intrinsic_lorentz_neural_network.md)
- [Towards Interpretable Visual Decoding with Attention to Brain Representations](towards_interpretable_visual_decoding_with_attention_to_brain_representations.md)
- [SONIC: Spectral Oriented Neural Invariant Convolutions](sonic_spectral_oriented_neural_invariant_convolutions.md)
- [Generalizable, Real-Time Neural Decoding with Hybrid State-Space Models](../../NeurIPS2025/medical_imaging/generalizable_real-time_neural_decoding_with_hybrid_state-space_models.md)

<!-- RELATED:END -->
