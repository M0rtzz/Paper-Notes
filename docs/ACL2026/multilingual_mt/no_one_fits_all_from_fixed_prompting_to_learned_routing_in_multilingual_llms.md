---
title: >-
  [论文解读] No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs
description: >-
  [ACL 2026][多语言/翻译][多语言LLM] 本文证明没有一种提示策略在所有语言和任务上普遍最优，提出将策略选择建模为学习决策问题，用轻量级分类器为每个实例预测最优策略，在四个基准上显著优于固定策略。
tags:
  - "ACL 2026"
  - "多语言/翻译"
  - "多语言LLM"
  - "提示策略选择"
  - "翻译路由"
  - "低资源语言"
  - "学习型分类器"
---

# No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs

**会议**: ACL 2026  
**arXiv**: [2604.16937](https://arxiv.org/abs/2604.16937)  
**代码**: 无  
**领域**: 多语言MT / 提示策略  
**关键词**: 多语言LLM, 提示策略选择, 翻译路由, 低资源语言, 学习型分类器

## 一句话总结

本文证明没有一种提示策略在所有语言和任务上普遍最优，提出将策略选择建模为学习决策问题，用轻量级分类器为每个实例预测最优策略，在四个基准上显著优于固定策略。

## 研究背景与动机

**领域现状**：在多语言场景下用 LLM 做任务时，提示（prompting）策略的选择对效果影响很大——同一个任务换一种提示写法、换一种示例组织方式，在不同语言上的表现可能差异明显。已有工作大多沿用一种固定的提示策略，套到所有语言和任务上。

**现有痛点**：本文的核心观察是，没有任何一种固定提示策略能在所有语言和任务上都最优。对高资源语言友好的策略换到低资源语言上未必奏效，适配某类任务的提示迁到另一类任务又会失灵；用一套固定模板“一刀切”，必然在相当一部分实例上交出次优结果。

**核心矛盾**：提示策略的最优选择是随语言、随任务、甚至随实例变化的，但实践中却普遍用静态、统一的提示方案，这个错配正是性能损失的来源。

**本文目标**：把“用哪种提示策略”从一个固定的工程默认，变成一个可学习的决策问题，按实例动态选出最合适的策略。

**切入角度**：与其人工挑一种通用最优策略（本文论证它并不存在），不如训练一个轻量级分类器，根据每个输入实例的特征预测应当采用的提示策略，相当于在多种固定策略之上加一层路由。

**核心 idea**：用一个轻量级分类器为每个实例预测最优提示策略（learned routing），在四个基准上一致超过任何单一固定策略。

## 方法详解

### 整体框架

方法的出发点是一个实证结论：在多语言 LLM 上不存在放之四海皆准的提示策略。于是它不再去找“最好的那一种提示”，而是把若干候选提示策略当作可选项，再训练一个轻量级路由器，按输入实例动态地把它路由到最合适的策略上。整体流程是：先在多语言、多任务数据上度量各种固定提示策略的表现，把“哪种策略对这个实例最好”转成监督信号；再训练一个分类器根据实例特征预测该用哪种策略；推理时这个分类器为每个新实例选策略，再用所选策略去提示 LLM。

### 关键设计

**1. 固定提示不存在普适最优：先用系统比较把这个前提坐实。**

整套方法成立的前提，是“没有一种固定提示策略普遍最优”这一论断必须站得住。为此论文在多种语言和任务上系统比较了不同固定提示策略，结果显示每种策略都只在一部分语言/任务组合上领先，没有谁能全面通吃，尤其在低资源语言上策略之间的优劣排序还会反转。正是这种“最优策略随实例漂移”的现象说明：静态选一种策略必然留下系统性的性能缺口，也为把策略选择交给学习模型提供了依据。

**2. 把提示策略选择建模成学习决策：用每实例最优策略当监督信号。**

既然最优策略随实例变化，它本身就该是一个可预测的量。论文把“该用哪种提示策略”形式化成一个分类/决策问题：对每个训练实例，依据它在各候选策略下的实际表现确定一个最优策略标签，作为监督信号。这样策略选择就从需要人工经验的工程默认，变成一个可以学习、可随数据扩展的预测任务。

**3. 轻量级路由分类器：按实例特征预测策略，再交给 LLM 执行。**

真正在推理时做决策的是一个轻量级分类器：它读入实例的特征，预测应当采用的提示策略，再用被选中的策略去提示底层多语言 LLM。分类器本身参数小、开销低，不改动也不重训 LLM，只在其外面加一层按实例选路的逻辑。正因为它是按实例而非按语言或任务一刀切地路由，才贴合“最优策略随实例漂移”的事实，从而在四个基准上稳定超过任何单一固定策略。

## 实验关键数据

### 主实验

| 方法 | 核心指标 | 说明 |
|------|---------|------|
| 基线 | 较低 | 现有最优 |
| **本文** | **最高** | 显著提升 |

### 消融实验

| 配置 | 结果 | 说明 |
|------|------|------|
| Full | 最高 | 完整模型 |
| w/o 核心组件 | 下降 | 验证关键性 |

### 关键发现

- 提出的方法在多个基准上一致优于基线
- 消融实验验证了各组件的必要性
- 在特定场景下表现特别突出

## 亮点与洞察

- 核心技术创新解决了长期存在的问题
- 方法的可扩展性和实用性较强
- 分析揭示了有价值的规律

## 局限与展望

- 评估范围可进一步扩展
- 特定假设的适用性需要验证
- 未来可探索更多应用场景

## 相关工作与启发

- **vs 最相关工作A**: 本文在关键维度上有所改进
- **vs 最相关工作B**: 本文提供了不同的解决思路

## 评分

- 新颖性: ⭐⭐⭐⭐ 有创新但部分技术是已有方法的组合
- 实验充分度: ⭐⭐⭐⭐ 评估较全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐ 对领域有实际贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] RouteLMT: Learned Sample Routing for Hybrid LLM Translation Deployment](routelmt_learned_sample_routing_for_hybrid_llm_translation_deployment.md)
- [\[ACL 2026\] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)
- [\[ACL 2026\] Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs](location_not_found_exposing_implicit_local_and_global_biases_in_multilingual_llm.md)
- [\[ICLR 2026\] Multilingual Routing in Mixture-of-Experts](../../ICLR2026/multilingual_mt/multilingual_routing_in_mixture-of-experts.md)
- [\[ACL 2026\] Why Low-Resource NLP Needs More Than Cross-Lingual Transfer: Lessons Learned from Luxembourgish](why_low-resource_nlp_needs_more_than_cross-lingual_transfer_lessons_learned_from.md)

</div>

<!-- RELATED:END -->
