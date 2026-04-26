---
title: >-
  [论文解读] Language Models Grow Less Humanlike beyond Phase Transition
description: >-
  [ACL 2025][LLM/NLP][预训练相变] 本文发现语言模型在预训练过程中与人类阅读行为的对齐（PPP）经历先升后降的拐点现象，通过关联和因果实验证明此拐点由预训练中的相变（specialized attention heads 的快速涌现）导致，且相变并非直接产生有害的注意力模式，而是改变了模型后续的学习动态使其持续偏离人类。
tags:
  - ACL 2025
  - LLM/NLP
  - 预训练相变
  - 心理计量预测力
  - 注意力头
  - 人类阅读行为
  - 语言模型认知对齐
---

# Language Models Grow Less Humanlike beyond Phase Transition

**会议**: ACL 2025  
**arXiv**: [2502.18802](https://arxiv.org/abs/2502.18802)  
**代码**: 无  
**领域**: LLM NLP / 认知科学  
**关键词**: 预训练相变、心理计量预测力、注意力头、人类阅读行为、语言模型认知对齐

## 一句话总结

本文发现语言模型在预训练过程中与人类阅读行为的对齐（PPP）经历先升后降的拐点现象，通过关联和因果实验证明此拐点由预训练中的相变（specialized attention heads 的快速涌现）导致，且相变并非直接产生有害的注意力模式，而是改变了模型后续的学习动态使其持续偏离人类。

## 研究背景与动机

**领域现状**：心理语言学研究表明，语言模型的 surprisal（句子中每个词的负对数概率）可以预测人类的阅读行为指标（如注视时长、自定步速阅读时间等）。这种能力被称为心理计量预测力（psychometric predictive power, PPP）。PPP 通常在预训练早期随训练步数增加而改善，因为模型逐渐学会更好地编码语言统计规律。

**现有痛点**：PPP 的改善存在一个拐点——达到峰值后要么趋于平稳，要么开始下降。这一拐点现象已被多个研究独立观察到，但没有统一的解释。已有的理论包括：词频效应（模型过度关注低频词）、注意力中的近因偏差、上下文窗口大小等，但没有一个理论能解释拐点为什么存在以及它与预训练动态的关系。

**核心矛盾**：直觉上，更好的语言模型（perplexity 更低）应该更好地预测人类行为，但实际上训练越多、PPP 反而可能下降——为什么模型变得"更聪明"但"更不像人"？

**本文目标**：找到 PPP 拐点的根本原因，并解释其与预训练动态的关系。

**切入角度**：作者假设 PPP 拐点与预训练中的相变（phase transition）相关。相变是指模型在训练过程中某些能力突然涌现的现象，表现为特化注意力头（如 induction heads）的快速出现。这些头专门负责 in-context learning 等能力，其涌现可能改变模型对语言统计规律的编码方式。

**核心 idea**：预训练相变（特化注意力头的快速涌现）是 PPP 拐点的根因。相变不是通过产生"坏的"注意力模式直接损害 PPP，而是改变了模型后续的学习动态，使继续训练持续偏离人类阅读行为模式。

## 方法详解

### 整体框架

研究分三个阶段进行：(1) 关联分析——在多个预训练 checkpoint 上测量 PPP 和相变指标，验证两者在时间上的对应关系；(2) 因果实验——通过消融和替换特定注意力头，验证相变确实导致了 PPP 下降；(3) 机制分析——区分相变的直接效应和对后续学习动态的间接效应。

### 关键设计

1. **PPP 测量方法**:

    - 功能：量化模型 surprisal 对人类阅读行为的预测能力
    - 核心思路：使用多个人类阅读行为数据集（eye-tracking、self-paced reading），将模型在各 checkpoint 产生的逐词 surprisal 作为回归特征预测人类阅读时间，用 $\Delta$LogLik（加入 surprisal 后的对数似然提升）作为 PPP 指标。分别在 GPT-2 级别模型的多个中间 checkpoint 上计算 PPP 曲线。
    - 设计动机：PPP 是连接语言模型和认知科学的标准桥梁指标，其拐点行为已被多次观察但未被解释。

2. **相变检测方法**:

    - 功能：定位预训练过程中特化注意力头涌现的时间点
    - 核心思路：追踪模型在各 checkpoint 上 induction heads 的数量和强度。Induction heads 是一种特化的注意力头模式，负责执行 in-context copying（看到 "A B ... A" 序列时预测 "B"）。通过在合成的重复序列上测试注意力头的匹配分数来检测。当 induction heads 数量在短时间内急剧增加时，标记为相变发生点。
    - 设计动机：Olsson et al. (2022) 已经观察到 induction heads 在 Transformer 训练中的突然涌现构成一种相变。本文假设这种相变同时触发了 PPP 的拐点。

3. **因果干预实验**:

    - 功能：从相关性推进到因果性
    - 核心思路：设计三组实验——(a) 在相变后的 checkpoint 上消融（ablate）特化注意力头，观察 PPP 是否恢复；(b) 用相变前 checkpoint 的注意力头替换相变后的对应头，测试是否能"逆转"相变的影响；(c) 比较消融后模型继续训练的 PPP 轨迹与正常训练的差异，区分直接效应和间接效应（对后续学习动态的改变）。
    - 设计动机：关联分析只能说明相变和 PPP 拐点同时发生，因果实验才能确认前者导致了后者。

### 损失函数 / 训练策略

本文是分析性工作，不涉及新的训练策略。使用预训练 GPT-2 级别模型的多个中间 checkpoint 进行分析。

## 实验关键数据

### 主实验（关联分析）

PPP 拐点与相变时间点的对应关系：

| 数据集 | PPP 拐点 (步数) | 相变开始 (步数) | 相变结束 (步数) | 时间对应 |
|--------|----------------|----------------|----------------|---------|
| Dundee (eye-tracking) | ~3k-5k | ~3k | ~5k | 拐点在相变期间 |
| Natural Stories | ~3k-5k | ~3k | ~5k | 拐点在相变期间 |
| Brown Corpus | ~3k-5k | ~3k | ~5k | 拐点在相变期间 |

### 消融实验（因果验证）

| 干预方式 | PPP 变化 | 说明 |
|---------|---------|------|
| 不干预（基线） | 相变后持续下降 | PPP 在拐点后稳步下降 |
| 消融 induction heads | 小幅恢复 | 直接效应较小 |
| 替换为相变前的 heads | 小幅恢复 | 确认相变是必要条件 |
| 消融后继续训练 | PPP 下降减缓 | 间接效应才是主因 |

### 关键发现

- **相变和 PPP 拐点在时间上高度对齐**：在多个数据集和多种阅读行为指标上一致。相变期间 induction heads 数量急剧增加，PPP 同时达到峰值然后开始下降。
- **相变的间接效应远大于直接效应**：消融相变后涌现的注意力头只能带来 PPP 的小幅恢复。但如果在相变前分叉训练（避免相变发生），PPP 的后续轨迹会显著不同。这说明相变不是通过产出"不好的"注意力模式直接损害 PPP，而是改变了模型的整体学习动态——相变后模型进入了一种新的优化轨道，持续训练在这条轨道上会不断偏离人类阅读模式。
- **"更强的模型更不像人"的机制解释**：相变使模型获得了更强的 in-context learning 能力（perplexity 下降），但这种能力的学习过程改变了模型内部的表征方式，使其编码语言信息的方式与人类大脑处理语言的方式产生了系统性偏差。

## 亮点与洞察

- **优雅的因果推理链**：从观察（PPP 拐点）→ 假设（相变导致）→ 关联验证（时间对齐）→ 因果验证（消融实验）→ 机制区分（直接 vs 间接效应），逻辑链条完整且令人信服。
- **"相变改变学习动态"的洞察**：不是相变产出的具体结构有问题，而是相变改变了模型后续的学习路径。这个发现对理解 LLM 的训练动态有更广泛的意义——能力的涌现可能伴随着其他维度上的系统性变化。
- **认知科学与 AI 的交叉价值**：如果我们希望 LLM 与人类认知对齐，理解它们在训练过程中何时、为什么偏离人类是关键的第一步。这项工作提示我们，规模化训练可能天然会导致与人类认知的偏离。

## 局限与展望

- **模型规模受限**：实验仅在 GPT-2 级别（~100M 参数）的模型上验证。更大模型（如 GPT-3/4 级别）的相变动态可能不同——它们的相变可能发生在不同的训练阶段，或者可能有多个相变点。
- **因果实验的局限性**：消融注意力头是一种粗粒度的干预。模型的内部表征在相变后的变化可能是分布式的，无法通过消融特定组件完全捕捉。
- **PPP 指标的局限**：PPP 只捕捉了人类阅读行为的某些方面（如注视时长），不能代表人类语言理解的全貌。模型在其他认知指标（如 N400 ERP）上的行为可能不同。
- **未提供改进方案**：论文解释了为什么 PPP 下降但未提出如何保持 PPP 的方法。未来可以探索的方向包括：在相变后使用人类阅读数据作为辅助训练目标、调整学习率策略来缓和相变的影响、或者设计保留认知对齐的训练课程。

## 相关工作与启发

- **vs Olsson et al. (2022)**：Olsson 等人发现了 induction heads 的相变现象，但未将其与认知科学指标联系起来。本文展示了这种相变的"副作用"——虽然获得了 in-context learning 能力，但偏离了人类认知。
- **vs Oh & Schuler (2023)**：Oh & Schuler 观察到 PPP 拐点并提出了词频效应假说。本文提供了一个更底层的统一解释——相变改变了学习动态，词频效应可能只是其表征之一。
- **vs Scaling Laws**：传统 scaling laws 认为更大的模型 = 更好的语言建模 = 更好的一切。本文提示了一个反例维度：在"像人类一样处理语言"这件事上，更强的模型可能更差。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将预训练相变与认知科学指标联系，"间接效应"的发现非常深刻
- 实验充分度: ⭐⭐⭐⭐ 关联+因果实验设计严谨，多个数据集验证，但模型规模受限
- 写作质量: ⭐⭐⭐⭐ 推理链清晰，但 HTML 版本不可用限制了对细节的了解
- 价值: ⭐⭐⭐⭐⭐ 对理解 LLM 训练动态和认知对齐有深远意义，跨学科贡献大

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Perspective Transition of Large Language Models for Solving Subjective Tasks](perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)
- [\[ACL 2025\] Bias in Language Models: Beyond Trick Tests and Towards RUTEd Evaluation](bias_in_language_models_beyond_trick_tests_ruted_evaluation.md)
- [\[ICML 2025\] Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence](../../ICML2025/llm_nlp/beyond_induction_heads_in-context_meta_learning_induces_multi-phase_circuit_emer.md)
- [\[ACL 2025\] Soundwave: Less is More for Speech-Text Alignment in LLMs](soundwave_less_is_more_for_speech-text_alignment_in_llms.md)
- [\[ACL 2025\] Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More](lm_graph_search_supervision.md)

<!-- RELATED:END -->
