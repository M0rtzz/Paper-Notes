---
title: >-
  [论文解读] An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal
description: >-
  [ACL 2026][LLM/NLP][惊奇度理论] 通过在花园路径句上微调神经语言模型，证明了存在一个神经 LM 能够通过惊奇度（surprisal）同时解释花园路径效应和自然阅读时间，为惊奇度理论提供了存在性证明。
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "惊奇度理论"
  - "花园路径效应"
  - "人类阅读时间"
  - "语言模型微调"
  - "心理语言学"
---

# An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal

**会议**: ACL 2026  
**arXiv**: [2604.18293](https://arxiv.org/abs/2604.18293)  
**代码**: [github](https://github.com/osekilab/RE-GPE)  
**领域**: LLM/NLP  
**关键词**: 惊奇度理论, 花园路径效应, 人类阅读时间, 语言模型微调, 心理语言学

## 一句话总结

通过在花园路径句上微调神经语言模型，证明了存在一个神经 LM 能够通过惊奇度（surprisal）同时解释花园路径效应和自然阅读时间，为惊奇度理论提供了存在性证明。

## 研究背景与动机

**领域现状**：惊奇度理论（Surprisal Theory）认为人类句子处理的难度与词的负对数概率（surprisal）成线性关系。近年来，研究者们使用语言模型作为人类预测的代理来验证这一假说。

**现有痛点**：尽管神经 LM 的惊奇度能较好地捕捉自然语料上的人类阅读时间，但在需要句法消歧的句子（如花园路径句 "the horse raced past the barn fell"）上，它严重低估了处理难度——仅能预测人类阅读减速的 1/10 到 1/30。

**核心矛盾**：这一失败引发了两种可能解释的争论——是神经 LM 的概率估计与人类不同，还是花园路径效应本质上无法归结为惊奇度？近期多项研究倾向于后者，即认为惊奇度理论不足以解释此类现象。

**本文目标**：探究第一种可能性——是否真的不可能构建一个能通过惊奇度解释花园路径效应的神经语言模型。

**切入角度**：不再评估现成的 LM，而是通过微调使 LM 的惊奇度估计与人类实际阅读时间对齐。

**核心 idea**：通过在花园路径句上微调 GPT-2，使其惊奇度更好地匹配人类阅读时间，从而提供一个"存在性证明"——存在神经 LM 能同时解释花园路径效应和自然阅读时间。

## 方法详解

### 整体框架

采用 Kiegeland et al. (2024) 的微调方法，将惊奇度驱动的阅读时间估计与实际人类阅读时间对齐。在花园路径句上微调 GPT-2（S/M/L），然后在三个维度评估：(i) 是否泛化到未见花园路径项目；(ii) 是否保持对自然语料阅读时间的预测能力；(iii) 是否保持一般 LM 能力。

### 关键设计

1. **基于岭回归的阅读时间估计**:

    - 功能：将 LM 惊奇度映射为阅读时间估计
    - 核心思路：特征向量包含当前位置及前两个位置的惊奇度（捕获溢出效应）和控制变量（词长、位置等），通过岭回归估计回归系数
    - 设计动机：在 ROI 之外的"普通"阅读时间上估计系数，这些系数也应能解释受句法消歧影响的 ROI 区域阅读时间

2. **带正则化的微调损失函数**:

    - 功能：引导 LM 产生更接近人类阅读模式的惊奇度分布
    - 核心思路：损失函数包含两项——最小化实际与估计阅读时间的残差平方，以及惩罚回归系数偏离初始系数的程度
    - 设计动机：第二项正则化防止 LM 通过降低 ROI 外的惊奇度来人为放大 ROI 处的估计阅读时间，确保模型学到合理的概率分布变化

3. **留一交叉验证框架**:

    - 功能：在小规模数据上严格评估泛化能力
    - 核心思路：每折留出一对来自每种花园路径构式的句对作测试，训练集和测试集在歧义动词和 ROI 词上无重叠
    - 设计动机：确保评估结果反映真正的泛化能力而非过拟合

### 损失函数 / 训练策略

损失函数 $\mathcal{L}_B(\theta)$ 包含：(1) 均方残差项，衡量预测与实际阅读时间差异；(2) 系数漂移惩罚项，防止回归系数偏离初始值太远。使用平衡批采样策略，每个 batch 包含等量的各类花园路径构式句对。

## 实验关键数据

### 主实验

| 模型 | 构式 | ROI 1 预微调覆盖率 | ROI 1 后微调覆盖率 |
|------|------|-------------------|-------------------|
| GPT-2 Small | MVRR | 7% | 73% |
| GPT-2 Small | NPS | 19% | 83% |
| GPT-2 Small | NPZ | 15% | 73% |

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| 单构式微调→跨构式迁移 | MVRR微调→NPS 51.5ms (基线9.6ms) | 跨构式迁移有效 |
| SRC/ORC不对称微调 | 仅捕获22%人类效应 | 惊奇度理论不适用场景下微调效果有限 |
| 自然语料预测力 | 微调后全面提升 | 花园路径微调意外改善自然文本预测 |

### 关键发现
- 微调后的 GPT-2 Small 表现最佳，在 ROI 1 处能捕获约 73%-83% 的人类阅读减速，远超微调前的 7%-19%
- 微调后模型正确复现了人类阅读减速幅度的构式间排序（MVRR > NPZ > NPS）
- 在自然语料上，微调后的 LM 对人类阅读时间的预测能力反而提升
- 单构式微调也能迁移到其他构式，暗示模型学到了花园路径效应的通用机制
- 但在 SRC/ORC 不对称（惊奇度理论被认为不适用的现象）上，微调效果有限，说明方法并非万能

## 亮点与洞察
- 巧妙的研究设计：将理论争论转化为可操作的实证问题——不是证明惊奇度理论"正确"，而是提供一个存在性证明
- 微调在花园路径句上的改进同时提升了自然语料的预测力，暗示原始 LM 与人类预测存在系统性偏差
- SRC/ORC 的负面结果同样有价值，说明方法的局限性与惊奇度理论的适用边界吻合
- 提出了深刻的理论问题：如果 LM 空间无界，惊奇度理论可能在实践中不可证伪

## 局限与展望
- 数据规模较小（24 对 × 3 种构式），需在更大规模数据上验证
- 未探究微调改变了 LM 的哪些内部机制
- 存在性证明虽然重要，但不等于说明现有 LM 自然学到了正确的人类概率分布
- 作者提出两个方向改善惊奇度理论的可证伪性：约束概率分布、要求解析分布的心理现实性

## 相关工作与启发
- **vs van Schijndel & Linzen (2021)**: 他们认为神经 LM 惊奇度无法解释花园路径效应是反对惊奇度理论的证据，本文挑战了这一结论
- **vs Kiegeland et al. (2024)**: 他们在自然语料上微调 LM，本文将此方法扩展到花园路径句
- **vs Huang et al. (2024)**: 他们系统展示了 LM 惊奇度低估花园路径效应的程度，本文提供了反例

## 评分
- 新颖性: ⭐⭐⭐⭐ 将理论争论转化为构造性存在性证明，思路新颖
- 实验充分度: ⭐⭐⭐⭐ 多维度评估，正面与负面结果均有，但数据规模较小
- 写作质量: ⭐⭐⭐⭐⭐ 理论背景阐述清晰，行文逻辑严谨
- 价值: ⭐⭐⭐⭐ 对计算心理语言学理论讨论有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Expect the Unexpected? Testing the Surprisal of Salient Entities](expect_the_unexpected_testing_the_surprisal_of_salient_entities.md)
- [\[ACL 2025\] Can Input Attributions Explain Inductive Reasoning in In-Context Learning?](../../ACL2025/llm_nlp/can_input_attributions_explain_inductive_reasoning_in_in-context_learning.md)
- [\[ACL 2025\] Neural Topic Modeling with Large Language Models in the Loop](../../ACL2025/llm_nlp/neural_topic_modeling_with_large_language_models_in_the_loop.md)
- [\[ACL 2025\] Information Locality as an Inductive Bias for Neural Language Models](../../ACL2025/llm_nlp/information_locality_as_an_inductive_bias_for_neural_language_models.md)
- [\[ACL 2026\] Big AI is Accelerating the Metacrisis: What Can We Do?](big_ai_is_accelerating_the_metacrisis_what_can_we_do.md)

</div>

<!-- RELATED:END -->
