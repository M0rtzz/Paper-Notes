---
title: >-
  [论文解读] Expect the Unexpected? Testing the Surprisal of Salient Entities
description: >-
  [ACL 2026][LLM/NLP][信息密度均匀性] 本文研究全局显著实体（discourse-level salient entities）与惊异度（surprisal）的关系，通过 70K+ 手工标注的实体提及和新颖的最小对提示方法，发现全局显著实体本身更出人意料（更高 surprisal）…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "信息密度均匀性"
  - "话语显著性"
  - "惊异度"
  - "实体突出度"
  - "语篇结构"
---

# Expect the Unexpected? Testing the Surprisal of Salient Entities

**会议**: ACL 2026  
**arXiv**: [2604.10724](https://arxiv.org/abs/2604.10724)  
**代码**: 无  
**领域**: 计算语言学 / 信息论  
**关键词**: 信息密度均匀性, 话语显著性, 惊异度, 实体突出度, 语篇结构

## 一句话总结

本文研究全局显著实体（discourse-level salient entities）与惊异度（surprisal）的关系，通过 70K+ 手工标注的实体提及和新颖的最小对提示方法，发现全局显著实体本身更出人意料（更高 surprisal），但它们系统性地降低周围内容的 surprisal，且该效应随体裁变化——话题连贯性高的文本中效应最强。

## 研究背景与动机

**领域现状**：信息密度均匀性（UID）假说认为说话者倾向于在话语中均匀分布信息，使惊异度大致恒定。然而，多项研究发现系统性偏离——语音学约束（词首高 surprisal）、句法约束、话语结构约束等"竞争性压力"会产生局部非均匀性。

**现有痛点**：(1) 先前的 UID 研究基本忽略了话语参与者的相对显著性——哪些实体是文本的"主角"；(2) 关于显著实体本身是否更可预测还是更出人意料，现有结果相互矛盾；(3) 多种因素（语法角色、近期性、指称形式等）影响实体可预测性，难以在自然语境中分离显著性效应。

**核心矛盾**：一方面，显著实体因反复提及而更可预测；另一方面，它们作为信息承载者可能包含更高信息量。两种效应如何在篇章层面交互？

**本文目标**：首次系统研究全局实体显著性与 surprisal 的关系，区分实体自身的 surprisal 和实体对周围内容 surprisal 的影响。

**切入角度**：利用 GUM-SAGE 数据集的手工标注（基于摘要一致性的显著性评分）和 16 种体裁的多样性，结合最小对提示方法控制混淆因素。

**核心 idea**：全局显著实体扮演"锚点"角色——它们本身承载更多信息（高 surprisal），但通过建立话题期望显著降低后续内容的不确定性，形成局部 surprisal"低谷"。

## 方法详解

### 整体框架

研究分三个层次展开：(1) RQ1——分析自然语料中显著实体的 surprisal 特征（控制位置、长度、嵌套等混淆因素）；(2) RQ2——使用最小对提示范式（替换显著 vs 非显著实体）测量实体对文档内容可预测性的因果影响；(3) RQ3——跨 16 种体裁比较效应强度。

### 关键设计

1. **基于摘要一致性的全局显著性度量**:

    - 功能：提供量化的话语级实体重要性评分
    - 核心思路：利用 GUM-SAGE 数据集，每篇文档有 5 份独立摘要。若一个实体在所有 5 份摘要中都被提及，得分为 5（最显著）；仅在 1 份中出现得分为 1；从未出现得分为 0（约 84.5% 的实体）。数据包含超过 70K 个实体提及，覆盖 31K 个独立实体
    - 设计动机：基于"如果一个实体是显著的，就很难写出不提及它的摘要"这一直觉——摘要一致性提供了稳健的、可操作的显著性定义

2. **最小对提示范式 (Minimal-Pair Prompting)**:

    - 功能：控制混淆因素，测量实体对后续内容 surprisal 的因果效应
    - 核心思路：对同一文档内容，分别以显著实体和非显著实体作为提示前缀，比较语言模型对后续文本的 surprisal。如果显著实体真的增强了文档内容的可预测性，那么以显著实体为提示时后续内容的 surprisal 应更低
    - 设计动机：自然语料中多种因素协同作用，无法分离显著性的独立贡献。最小对设计通过固定其他因素，只变化实体身份，实现了准因果推断

3. **跨体裁分析**:

    - 功能：揭示显著性-surprisal 关系的调节因素
    - 核心思路：GUM 语料库涵盖 16 种体裁（学术论文、传记、vlog、对话、法庭记录、散文、小说、论坛等），分别分析效应强度。预期话题连贯性强的文本（如学术论文——聚焦单一主题）效应最强，话题切换频繁的文本（如对话）效应最弱
    - 设计动机：如果显著性效应通过话题期望机制运作，那么话题一致性应是关键调节因素

### 损失函数 / 训练策略

本文为分析性工作，不涉及模型训练。使用语言模型计算 surprisal（负对数概率），在 GUM v11 语料库（250K+ tokens，16 种体裁）上进行统计分析。

## 实验关键数据

### 主实验

| 研究问题 | 核心发现 |
|----------|----------|
| RQ1: 显著实体自身 surprisal | 全局显著实体的 surprisal 显著高于非显著实体，控制位置、长度、嵌套后仍成立 |
| RQ2: 对周围内容的影响 | 显著实体系统性降低后续内容的 surprisal，创造局部"低谷" |
| RQ3: 体裁差异 | 效应在话题连贯文本（学术论文）中最强，在对话语境中最弱 |

### 消融实验

| 分析维度 | 结果 |
|----------|------|
| 显著性分数 vs surprisal | 正相关——得分越高，实体本身 surprisal 越高 |
| 最小对：显著 vs 非显著提示 | 显著实体提示下后续内容 surprisal 显著更低 |
| 话题连贯 vs 话题切换体裁 | 话题连贯体裁中效应强度约为话题切换体裁的 2-3 倍 |

### 关键发现

- 全局显著实体"更出人意料"但"使上下文更可预测"——两个看似矛盾的发现实际上反映了不同层面的信息组织
- 这一模式类似于语音学中的"词首高 surprisal"现象——信息在局部不均匀但在更大尺度上服务于整体均匀性
- 体裁效应符合话题连贯性假说，为 UID 竞争压力框架增添了指称结构这一新维度
- 约 84.5% 的实体得分为 0（非显著），表明大多数实体是"配角"

## 亮点与洞察

- "显著实体是信息锚点"的洞察优雅统一了两个方向的发现——自身高 surprisal 是因为承载关键信息，降低周围 surprisal 是因为建立了强话题期望
- 最小对提示方法巧妙地将因果推理引入观察性语料分析，可推广到其他话语现象研究
- 将 UID 框架中的"竞争压力"扩展到指称结构维度——之前的工作只考虑了语音学、句法和话语结构

## 局限与展望

- 仅使用英语数据，跨语言泛化性未知
- 显著性基于摘要一致性，可能偏向于可提取的信息而非深层主题重要性
- 语言模型计算的 surprisal 不等于人类认知 surprisal
- 未探索动态显著性——实体的局部显著性可能随话语推进而变化

## 相关工作与启发

- **vs Centering Theory**: 后者关注局部注意显著性（语法角色、近期性），本文关注全局话语显著性——两者互补
- **vs Clark et al. (2023)**: 后者发现句法约束限制了 UID 的实现程度，本文发现指称结构约束也类似
- **vs Tsipidi et al. (2024)**: 后者发现话语结构预测 surprisal 轮廓的非均匀性，本文将此扩展到实体显著性维度

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究全局实体显著性与 surprisal 的关系，最小对方法新颖
- 实验充分度: ⭐⭐⭐⭐ 70K 标注、16 种体裁覆盖广泛，但仅限英语
- 写作质量: ⭐⭐⭐⭐⭐ 研究问题层次分明，分析逻辑严密，结论清晰
- 价值: ⭐⭐⭐⭐ 为 UID 理论增加了重要的指称结构维度，对话语处理和语言模型评估有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal](clozing_the_gap_exploring_why_language_model_surprisal_outperforms_cloze_surpris.md)
- [\[ACL 2026\] An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal](an_existence_proof_for_neural_language_models_that_can_explain_garden-path_effec.md)
- [\[ACL 2025\] GIFT-SW: Gaussian Noise Injected Fine-Tuning of Salient Weights for LLMs](../../ACL2025/llm_nlp/gift-sw_gaussian_noise_injected_fine-tuning_of_salient_weights_for_llms.md)
- [\[ACL 2025\] The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](../../ACL2025/llm_nlp/token_granularity_impact.md)
- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](../../ACL2025/llm_nlp/leveraging_in-context_learning_for_political_bias_testing_of_llms.md)

</div>

<!-- RELATED:END -->
