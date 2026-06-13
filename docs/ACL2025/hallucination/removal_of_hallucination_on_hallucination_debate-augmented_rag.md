---
title: >-
  [论文解读] Removal of Hallucination on Hallucination: Debate-Augmented RAG
description: >-
  [ACL 2025][幻觉检测][检索增强生成] DRAG（Debate-Augmented RAG）提出在RAG系统的检索和生成两个阶段均引入多智能体辩论（MAD）机制，通过正反方辩论+裁判仲裁的结构化流程，消除因错误检索导致的"幻觉传递幻觉"问题，在6个QA基准上显著提升事实准确性。 领域现状：检索增强生成（RAG）通过…
tags:
  - "ACL 2025"
  - "幻觉检测"
  - "检索增强生成"
  - "幻觉消除"
  - "多智能体辩论"
  - "训练免微调"
  - "事实准确性"
---

# Removal of Hallucination on Hallucination: Debate-Augmented RAG

**会议**: ACL 2025  
**arXiv**: [2505.18581](https://arxiv.org/abs/2505.18581)  
**代码**: [GitHub](https://github.com/Huenao/Debate-Augmented-RAG)  
**领域**: 幻觉检测  
**关键词**: 检索增强生成, 幻觉消除, 多智能体辩论, 训练免微调, 事实准确性  

## 一句话总结
DRAG（Debate-Augmented RAG）提出在RAG系统的检索和生成两个阶段均引入多智能体辩论（MAD）机制，通过正反方辩论+裁判仲裁的结构化流程，消除因错误检索导致的"幻觉传递幻觉"问题，在6个QA基准上显著提升事实准确性。

## 研究背景与动机

**领域现状**：检索增强生成（RAG）通过引入外部知识来增强LLM的事实准确性，已成为减少幻觉的主流方案。标准RAG流程为：用户查询→检索相关文档→基于检索结果生成回答。

**现有痛点**：RAG的一个被忽视的关键问题是——检索本身也可能出错。当检索到错误或带偏见的文档时，模型不仅不会因此获益，反而会被误导，生成比无检索时更不准确的回答。更严重的是，模型可能对错误检索结果表现出过度信任，将虚假信息"包装"得像真实知识一样输出。

**核心矛盾**：这形成了一种"幻觉传递幻觉"（Hallucination on Hallucination）的级联效应——检索阶段的"幻觉"（错误文档）导致生成阶段产生新的幻觉，两个阶段的错误相互叠加。现有的RAG改进方法（如FLARE、Self-RAG）主要关注单一阶段的优化，无法系统性地解决这种双阶段级联问题。

**本文目标**：设计一个训练免微调的框架，同时在检索和生成两个阶段引入质量控制机制，打断"幻觉传递幻觉"的链条。

**切入角度**：借鉴多智能体辩论（Multi-Agent Debate, MAD）的思想——通过不同角色的LLM agent之间的结构化辩论来验证信息可靠性。当多个agent对检索结果或生成回答存在分歧时，通过辩论和裁判仲裁来达成更可靠的共识。

**核心 idea**：在RAG的检索阶段用辩论来过滤不可靠文档（Retrieval Debate），在生成阶段用辩论来验证和修正回答（Response Debate），全程无需额外训练。

## 方法详解

### 整体框架
DRAG将标准RAG流程扩展为两阶段辩论架构：(1) **检索辩论（Retrieval Debate）**：对检索到的文档进行可靠性辩论，筛选出高质量文档；(2) **生成辩论（Response Debate）**：基于筛选后的文档生成回答，多个agent通过对抗性辩论验证回答的事实准确性。每个阶段都包含正方（Proponent）、反方（Opponent）和裁判（Judge）三种角色，通过多轮辩论达成共识。

### 关键设计

1. **检索辩论（Retrieval Debate）**:

    - 功能：对检索到的文档进行可靠性评估和过滤
    - 核心思路：给定用户查询和检索到的K个文档，设置三种角色——正方agent主张文档与查询相关且事实可靠，反方agent寻找文档中的不一致性、偏见或与查询的不匹配，裁判agent综合双方论点做出最终裁决。每个文档经过多轮辩论（默认2-3轮），裁判最终判定该文档是否可信。不可信的文档被过滤掉，仅保留辩论通过的高质量文档作为生成阶段的输入。
    - 设计动机：单一agent难以全面评估文档质量，对抗性辩论通过"刻意寻找反面证据"来增强评估的严度

2. **生成辩论（Response Debate）**:

    - 功能：通过多角色辩论验证和修正生成的回答
    - 核心思路：引入非对称信息角色设计——正方agent获得完整的检索文档和问题，负责生成初始回答；反方agent仅获得问题（不看检索文档），负责基于自身知识质疑正方回答中的可能错误；裁判agent综合双方论点，参考检索文档但同时考虑反方指出的逻辑漏洞，最终生成经过验证的回答。通过多轮对抗性辩论，回答的事实可靠性逐步提升。
    - 设计动机：非对称信息设计迫使辩论产生真正的认知碰撞。如果所有agent都看到相同的错误文档，辩论可能流于"形式共识"；让反方不依赖检索结果独立思考，能有效发现检索引入的偏见

3. **辩论裁判与终止策略**:

    - 功能：控制辩论质量和收敛效率
    - 核心思路：裁判agent在每轮辩论后评估双方论点的质量和共识程度，通过结构化的评价模板（包括"论点强度"、"证据质量"、"逻辑一致性"等维度）做出裁决。当双方达成共识或达到最大辩论轮数时终止辩论。支持自定义检索辩论轮数（`max_query_debate_rounds`）和生成辩论轮数（`max_answer_debate_rounds`），在准确性和效率之间灵活平衡。
    - 设计动机：无限制辩论可能导致"过度讨论"或陷入循环，需要明确的终止条件和质量评估标准

### 损失函数 / 训练策略
DRAG是完全免训练的框架（training-free），不涉及任何模型微调或损失函数。所有agent都使用同一个预训练LLM（如Llama-3-8B-Instruct），通过不同的系统prompt来赋予不同的辩论角色。整个框架基于FlashRAG库构建，支持多种LLM作为骨干模型。

## 实验关键数据

### 主实验
在6个QA基准上与多种RAG baseline对比（使用Llama-3-8B-Instruct）：

| 方法 | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | StrategyQA |
|------|-----|----------|-------|----------|-------|------------|
| Naive Gen（无检索） | 22.8 | 55.3 | 21.4 | 26.1 | 25.7 | 67.5 |
| Naive RAG | 34.5 | 59.7 | 38.2 | 31.5 | 28.9 | 63.2 |
| FLARE | 30.1 | 57.4 | 33.7 | 30.8 | 28.3 | 65.8 |
| Iter-RetGen | 33.8 | 58.1 | 36.1 | 33.2 | 30.5 | 66.1 |
| IRCoT | 35.2 | 60.3 | 37.5 | 34.1 | 31.8 | 67.3 |
| Self-RAG | 36.1 | 61.2 | 39.0 | 33.7 | 30.2 | 66.8 |
| MAD | 34.3 | 60.5 | 37.8 | 32.5 | 29.7 | 68.2 |
| **DRAG** | **38.7** | **63.5** | **42.3** | **36.8** | **34.2** | **70.1** |

### 消融实验
各组件贡献的消融分析：

| 配置 | NQ | TriviaQA | 说明 |
|------|-----|----------|------|
| 完整DRAG | 38.7 | 63.5 | 最佳 |
| 去掉检索辩论 | 35.9 | 61.8 | 检索质量控制重要 |
| 去掉生成辩论 | 36.2 | 62.1 | 生成验证重要 |
| 仅标准MAD（无角色区分） | 35.1 | 60.8 | 非对称角色设计关键 |
| 辩论1轮 | 36.8 | 62.3 | 足够但不充分 |
| 辩论3轮 | 38.5 | 63.4 | 接近饱和 |

### 关键发现
- **Naive RAG在部分数据集上反而不如无检索生成**：在StrategyQA上，Naive RAG (63.2) 低于Naive Gen (67.5)，直接印证了"幻觉传递幻觉"问题的存在——错误检索反而误导了模型。
- **检索辩论和生成辩论贡献相当**：两者各贡献约2-3个百分点的提升，说明双阶段都有显著的错误需要纠正。
- **非对称信息角色设计是关键**：将生成辩论退化为标准MAD（所有agent看相同信息）后性能明显下降，验证了信息不对称产生更有效辩论的假设。
- 辩论轮数的收益递减明显——从1轮到2轮提升显著，但3轮相比2轮提升极小，且计算成本增加50%。

## 亮点与洞察
- **"幻觉传递幻觉"的问题定义**：首次系统性地定义和分析了RAG中检索错误如何级联放大生成幻觉的现象，这个概念框架对整个RAG领域的改进方向有启发性。
- **非对称信息辩论设计**：让反方agent不看检索文档、仅凭内在知识质疑正方回答，这种设计迫使辩论产生真正有价值的对抗，而非"看到同样的错误信息后达成错误共识"。这个思路可以直接迁移到任何多agent协作系统中。
- **训练免微调的实用性**：整个框架不需要任何训练，直接调用现有LLM即可部署，极大降低了使用门槛。

## 局限与展望
- 多agent辩论带来显著的推理成本增加——每次查询需要多次LLM调用，延迟约为标准RAG的3-5倍
- 辩论质量依赖底层LLM的推理能力，对较弱的模型效果可能打折扣
- 目前仅在短文本QA场景验证，对长文本摘要、多步推理等更复杂任务的效果未知
- 裁判agent的中立性无法保证，可能被较强的一方"说服"
- **改进方向**：可以研究异构agent辩论（使用不同的LLM作为不同角色），以增加观点多样性；也可以探索自适应辩论轮数策略（简单问题少辩、难问题多辩），降低计算开销

## 相关工作与启发
- **vs Self-RAG**: Self-RAG通过训练模型自身来评估检索质量和生成质量，需要额外训练；DRAG无需训练但推理成本更高，两者在不同部署场景下各有优势
- **vs FLARE/Iter-RetGen**: 这些方法通过迭代检索来改进信息召回，但不涉及对检索结果可靠性的显式验证；DRAG则直接在检索阶段用辩论过滤不可靠文档
- **vs Multi-Agent Debate (MAD)**: 标准MAD将辩论引入推理任务，DRAG将其扩展到RAG场景并设计了非对称信息角色和双阶段应用
- 该工作对构建可靠的RAG系统有直接参考价值，尤其是在对事实准确性要求高的应用（如医疗、法律）中

## 评分
- 新颖性: ⭐⭐⭐⭐ "幻觉传递幻觉"的定义和双阶段辩论框架有创新
- 实验充分度: ⭐⭐⭐⭐ 6个数据集、多种baseline、完整消融
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，框架描述直观
- 价值: ⭐⭐⭐⭐ 对RAG系统可靠性提升有实用价值，但推理成本是部署障碍

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](../../ACL2026/hallucination/stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ACL 2025\] REFIND at SemEval-2025 Task 3: Retrieval-Augmented Factuality Hallucination Detection in Large Language Models](refind_at_semeval-2025_task_3_retrieval-augmented_factuality_hallucination_detec.md)
- [\[ACL 2025\] DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination](drag_distilling_rag_slm.md)
- [\[ACL 2026\] Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate](../../ACL2026/hallucination/dialectic-med_mitigating_diagnostic_hallucinations_via_counterfactual_adversaria.md)
- [\[AAAI 2026\] MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning](../../AAAI2026/hallucination/multi-agent_undercover_gaming_hallucination_removal_via_coun.md)

</div>

<!-- RELATED:END -->
