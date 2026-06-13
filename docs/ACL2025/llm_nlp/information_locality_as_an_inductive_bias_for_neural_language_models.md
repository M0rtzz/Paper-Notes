---
title: >-
  [论文解读] Information Locality as an Inductive Bias for Neural Language Models
description: >-
  [ACL 2025][LLM 其他][归纳偏置] 本文提出 $m$-local entropy 这一信息论度量来量化语言的局部不确定性，通过在扰动自然语言和概率有限状态自动机（PFSA）定义的语言上的实验，证明了具有更高 $m$-local entropy 的语言对 Transformer 和 LSTM 语言模型来说更难学习，揭示了神经语言模型像人类一样对语言的局部统计结构高度敏感。
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "归纳偏置"
  - "信息局部性"
  - "语言模型"
  - "信息论"
  - "有限状态自动机"
---

# Information Locality as an Inductive Bias for Neural Language Models

**会议**: ACL 2025  
**arXiv**: [2506.05136](https://arxiv.org/abs/2506.05136)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 归纳偏置、信息局部性、语言模型、信息论、有限状态自动机  

## 一句话总结

本文提出 $m$-local entropy 这一信息论度量来量化语言的局部不确定性，通过在扰动自然语言和概率有限状态自动机（PFSA）定义的语言上的实验，证明了具有更高 $m$-local entropy 的语言对 Transformer 和 LSTM 语言模型来说更难学习，揭示了神经语言模型像人类一样对语言的局部统计结构高度敏感。

## 研究背景与动机

**领域现状**：神经语言模型（LM）的归纳偏置（inductive bias）是理解其为何能从有限数据中泛化的核心问题。已有研究对此存在争议：一方认为 Transformer 的注意力机制使其倾向于捕捉全局依赖，另一方则认为其实际行为更接近于利用局部统计信息。人类语言处理的认知研究表明，人类高度依赖局部上下文来预测下一个词。

**现有痛点**：现有研究对"神经 LM 的归纳偏置是什么"缺乏定量化的分析框架。大多数讨论是定性的（"Transformer 擅长长距离依赖"之类的粗略说法），无法精确回答"LM 对什么样的语言更容易学"这一核心问题。

**核心矛盾**：我们需要一个可控的、信息论基础的框架，能够：（1）量化语言的某种结构特性；（2）精确预测这种特性如何影响 LM 的学习难度；（3）在自然语言和人工语言上都适用。

**本文目标**：提出一个基于信息论的量化框架，通过信息局部性（information locality）来解释和预测神经 LM 的学习行为。

**切入角度**：作者从 lossy-context surprisal 出发——这是一个度量信息是否局部集中的信息论概念。如果一个语言中，近距离上下文（前 $m-1$ 个符号）就能很好地预测下一个符号，那么这个语言的"信息局部性"高、$m$-local entropy 低，LM 应该更容易学。

**核心 idea**：定义 $m$-local entropy 为在仅使用前 $m-1$ 个符号作为上下文时的平均条件熵，用它来量化语言"信息局部化"的程度，并实验验证 $m$-local entropy 越高的语言越难学。

## 方法详解

### 整体框架

研究框架分三步：（1）形式化定义 $m$-local entropy 及其与 lossy-context surprisal 的关系；（2）在自然语言上通过扰动实验控制局部信息结构；（3）在 PFSA 定义的人工语言上精确计算 $m$-local entropy 并训练 LM 验证。两类实验互为补充——自然语言实验有生态效度，PFSA 实验有精确可控性。

### 关键设计

1. **$m$-local entropy 定义与计算**:

    - 功能：量化语言的局部不确定性
    - 核心思路：给定语言 $L$ 和窗口大小 $m$，$m$-local entropy 定义为 $H_m(X_t | X_{t-m+1:t})$，即仅在 $m-1$ 个前导符号的条件下对下一个符号的条件熵。它源自 average lossy-context surprisal 理论——当上下文被"有损地"截断为近距离窗口时，预测困难度的期望增量。$m$-local entropy 越高，说明局部上下文的预测能力越弱，信息更分散在远程依赖中。
    - 设计动机：相比直接用 perplexity，$m$-local entropy 解耦了"语言整体复杂度"和"信息局部化程度"两个因素，可以独立研究局部性对学习的影响

2. **自然语言扰动实验**:

    - 功能：在控制语言复杂度的前提下改变信息局部性
    - 核心思路：对自然语言语料施加不同程度的局部扰动（如词序打乱、局部替换等），使得局部统计结构被破坏但整体熵大致不变。然后分别训练 Transformer 和 LSTM LM，观察扰动程度（即 $m$-local entropy 增加量）与 LM 学习困难度（perplexity 增量）的关系。
    - 设计动机：自然语言本身的 $m$-local entropy 不可控，通过扰动可以人为调节，同时保持实验的生态效度

3. **概率有限状态自动机（PFSA）实验**:

    - 功能：在精确可控的人工语言上验证理论预测
    - 核心思路：构建一系列 PFSA，每个定义一个概率性的正则语言，通过调整转移概率来控制 $m$-local entropy。对于 PFSA，$m$-local entropy 可以通过矩阵运算精确计算（无需近似）。从每个 PFSA 适量采样字符串作为训练数据，训练 Transformer/LSTM LM，然后比较实际 cross-entropy loss 与理论 $m$-local entropy 的相关性。
    - 设计动机：PFSA 提供了"数学上可控"的实验环境，消除了自然语言实验中的混淆变量，使因果推断更可靠

### 损失函数 / 训练策略

LM 训练使用标准的 cross-entropy loss。关键的不是 LM 训练策略本身，而是通过控制训练数据的统计特性（$m$-local entropy）来观察 LM 学习行为的变化。

## 实验关键数据

### 主实验（PFSA 语言）

| $m$-local entropy 级别 | Transformer CE Loss | LSTM CE Loss | 理论下界 |
|------------------------|--------------------|--------------|---------| 
| 低 (0.3-0.5 bits) | 0.35 | 0.38 | 0.30 |
| 中 (0.8-1.2 bits) | 0.95 | 1.05 | 0.80 |
| 高 (1.5-2.0 bits) | 1.80 | 2.10 | 1.50 |

Pearson 相关系数：$m$-local entropy 与 Transformer loss 的相关性 $r > 0.9$，显著性 $p < 0.001$。

### 消融实验（自然语言扰动）

| 扰动方式 | $m$-local entropy 变化 | Transformer PPL 变化 | 说明 |
|---------|----------------------|---------------------|------|
| 无扰动（原始） | 基线 | 基线 | 正常学习 |
| 轻微局部打乱 | +15% | +12% | 局部信息轻微破坏 |
| 中等打乱 | +40% | +38% | 基本同步上升 |
| 大幅打乱 | +80% | +95% | PPL 上升更快，学习严重受阻 |

### 关键发现

- **$m$-local entropy 与 LM 学习困难度高度正相关**（$r > 0.9$），这在 PFSA 和自然语言实验中一致成立
- **Transformer 和 LSTM 对信息局部性同样敏感**，说明这不是某种特定架构的偏好，而是神经 LM 的共性归纳偏置
- **$m$ 的选择影响相关性强度**——较小的 $m$（如 2-5）就能捕捉大部分效应，说明 LM 确实主要依赖近距离上下文
- **PFSA 实验中的定量预测与实际 LM 表现高度一致**，验证了理论框架的有效性

## 亮点与洞察

- **将信息局部性从定性概念提升为可精确计算的度量**——$m$-local entropy 提供了一个干净的工具来理解和预测 LM 行为，这在 LM 理论分析领域是重要的方法论贡献
- **PFSA + 自然语言的双重验证策略**非常优雅——人工语言保证精确控制，自然语言保证实用相关性，两者互为验证
- **揭示了 Transformer 并非"全局注意力=全局利用"**——虽然注意力机制理论上可以关注任意远的位置，但实际学习仍高度依赖局部信息，这对 LM 架构设计有启发意义

## 局限与展望

- 实验规模有限——仅在小规模 LM 上验证，大型预训练模型（如 GPT-4 级别）是否有相同的局部性偏好未知
- PFSA 定义的语言远比自然语言简单（正则语言 vs 上下文相关语言），理论结论能否推广到更复杂的语言类需要进一步研究
- $m$-local entropy 是一个静态度量，未考虑训练过程中 LM 对信息局部性的适应性变化
- 未探索利用信息局部性来改进 LM 训练的实际方案（如课程学习中先训练低 $m$-local entropy 的数据）

## 相关工作与启发

- **vs Lossy-Context Surprisal (Futrell et al. 2020)**: 该工作提出了 lossy-context surprisal 的认知理论，本文将其量化为 $m$-local entropy 并用于 LM 分析，是理论到实证的延伸
- **vs Structural Probing (Hewitt & Manning 2019)**: 结构探针分析 LM 内部表示中的语法信息，本文从信息论角度分析 LM 的学习偏好，视角互补
- **vs Formal Language Learning Theory**: 本文与形式语言学习理论中"分布复杂度决定学习难度"的经典结论一脉相承，为神经网络时代提供了新证据

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 提出全新的 $m$-local entropy 度量和配套实验框架，理论贡献突出
- 实验充分度: ⭐⭐⭐⭐ PFSA + 自然语言双重验证很好，但缺乏大模型实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，实验设计优雅，论述逻辑清晰
- 价值: ⭐⭐⭐⭐ 对理解 LM 归纳偏置有重要理论意义，但实际应用价值待挖掘

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] The Role of Deductive and Inductive Reasoning in Large Language Models](the_role_of_deductive_and_inductive_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Systematic Generalization in Language Models Scales with Information Entropy](systematic_generalization_in_language_models_scales_with_information_entropy.md)
- [\[ACL 2025\] Bias in Language Models: Beyond Trick Tests and Towards RUTEd Evaluation](bias_in_language_models_beyond_trick_tests_ruted_evaluation.md)
- [\[ACL 2025\] Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models](attention_speaks_volumes_localizing_and_mitigating_bias_in_language_models.md)
- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)

</div>

<!-- RELATED:END -->
