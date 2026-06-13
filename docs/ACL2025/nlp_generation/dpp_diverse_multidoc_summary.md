---
title: >-
  [论文解读] Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries
description: >-
  [ACL 2025][文本生成][multi-document summarization] 提出将多文档摘要解耦为关键点抽取→DPP多样性选择→重写三步流水线，通过行列式点过程（DPP）进行原则性内容选择，显著提升LLM多文档摘要的源文档覆盖率。 - 问题定义：多文档多样性摘要（MDDS）要求从多篇报道同一新闻事件的文章中…
tags:
  - "ACL 2025"
  - "文本生成"
  - "multi-document summarization"
  - "DPP"
  - "content selection"
  - "source coverage"
  - "personalized summary"
---

# Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries

**会议**: ACL 2025  
**arXiv**: [2505.21859](https://arxiv.org/abs/2505.21859)  
**代码**: 未公开  
**领域**: 其他  
**关键词**: multi-document summarization, DPP, content selection, source coverage, personalized summary  

## 一句话总结

提出将多文档摘要解耦为关键点抽取→DPP多样性选择→重写三步流水线，通过行列式点过程（DPP）进行原则性内容选择，显著提升LLM多文档摘要的源文档覆盖率。

## 研究背景与动机

- **问题定义**：多文档多样性摘要（MDDS）要求从多篇报道同一新闻事件的文章中生成覆盖多元观点的摘要，现有LLM因"lost in the middle"注意力偏差导致覆盖率不足
- **现有方法局限**：直接将所有文档拼接后让LLM一步生成摘要，内容选择和文本生成耦合在一起，LLM的位置偏差会导致开头/结尾文档被过度关注
- **核心假设**：LLM不擅长内容选择，将其与文本生成解耦并用原则性方法（DPP）替代LLM做内容选择可提升覆盖率
- **额外目标**：通过将用户意图相关性融入DPP核矩阵，生成个性化摘要

## 方法详解

### 整体框架

LLM + DPP三步流水线：(1) 用LLM从每篇源文档中抽取原子关键点（key points）；(2) 用DPP从所有关键点中选择多样性子集；(3) 用LLM将选中关键点重写为连贯摘要。

### 关键设计

- **关键点抽取**：使用zero-shot提示让LLM将每篇文档 $d_i$ 分解为原子关键点集合 $K_i$，每个关键点捕获一个独立的信息单元，确保信息粒度足够细
- **DPP多样性选择**：用Transformer编码器（DeBERTa-V3 BERTScore嵌入）将关键点编码为向量，构建高斯核矩阵 $L$ 衡量关键点间相似度，通过DPP推理（谱方法精确采样）选择多样性最大的子集 $K_{sel}$，选中数量由核矩阵特征值自动决定
- **相关性加权DPP**：对于query-focused任务，用e5-mistral-7b-instruct计算每个关键点与用户意图的相关性分数 $R_i$，构造 $L' = RLR^T$ 平衡多样性与相关性

### 损失函数

无需训练，DPP选择基于核矩阵的组合优化（贪心近似），所有LLM步骤均为zero-shot提示。评估使用LLM-as-judge（GPT-4o）判断摘要是否能正确回答源文档相关问题，人工验证与LLM判断的一致率达86.4%（可回答性）和95.3%（正确性）。

## 实验

### 主实验

| 方法 | DiverseSumm覆盖率 | | | | DiverseSumm Augmented覆盖率 | | | |
|------|-------|-------|--------|--------|-------|-------|--------|--------|
| | GPT 3.5 | GPT 4o | Claude | Llama | GPT 3.5 | GPT 4o | Claude | Llama |
| Naive LLM | 0.332 | 0.552 | 0.478 | 0.243 | 0.267 | 0.481 | 0.425 | 0.219 |
| All KPs | 0.347 | 0.544 | 0.568 | 0.346 | 0.257 | 0.462 | 0.411 | 0.237 |
| LLM-Selected KPs | 0.437 | 0.575 | 0.537 | 0.338 | 0.385 | 0.541 | 0.514 | 0.309 |
| **LLM + DPP** | **0.471** | **0.581** | **0.592** | **0.365** | **0.385** | **0.554** | **0.547** | **0.323** |

### 消融实验（DPP核选择）

| 核函数 | GPT 3.5 | GPT 4o | Claude |
|--------|---------|--------|--------|
| Gaussian σ=0.1 | 0.449 | **0.615** | **0.635** |
| Gaussian σ=1 | **0.471** | 0.581 | 0.592 |
| Gaussian σ=10 | 0.434 | 0.591 | 0.520 |
| Linear | 0.465 | 0.589 | 0.586 |

### 关键发现

- LLM + DPP在所有4个LLM上一致性地取得最高覆盖率，证明DPP内容选择的有效性
- 显式关键点选择（LLM-Selected KPs和LLM + DPP）普遍优于全量关键点（All KPs），说明仅缩短上下文不够，需要主动选择
- DPP选择的关键点比LLM选择的关键点覆盖更多源文档（分布更均匀）
- LLM + DPP有效缓解了位置偏差：Llama的尾部偏差、GPT-4o的头部偏差均显著降低
- 覆盖率提升并非来自更长摘要——各方法平均摘要长度无显著差异

## 亮点

- 将原则性统计方法（DPP）与LLM提示流水线优雅结合，展示了并非所有步骤都需要用LLM完成
- 清晰揭示了不同LLM在多文档场景下的位置偏差模式（头部/尾部/中间偏差）及其对覆盖率的影响
- 方法简洁、模块化、即插即用，适用于不同LLM后端，无需额外训练
- 通过合成问题扩展和人工验证确保评估的可靠性

## 局限性

- 仅在新闻领域的DiverseSumm benchmark上评估，其他领域（如科技、医疗）泛化性未知
- DPP核函数和参数（σ）需手动调优，不同LLM的最优配置可能不同
- query-focused任务的相关性评估依赖额外的retrieval模型（e5-mistral-7b），增加了系统复杂度和推理成本
- 关键点抽取质量依赖LLM本身的能力，弱模型可能产生低质量关键点
- 未与最新的长上下文优化方法（如RAG、context compression）进行对比

## 相关工作

- 多文档摘要：DiverseSumm benchmark（Huang et al. 2024）、层次摘要（Chang et al. 2024）
- LLM注意力偏差："Lost in the middle"（Liu et al. 2024）
- DPP在NLP中的应用：文档摘要（Kulesza et al. 2012）、推荐系统多样性（Chen et al. 2018）
- 原子声明分解：Kim et al. 2024, Krishna et al. 2023, Padmakumar & He
- 查询聚焦摘要：Daumé III & Marcu 2006, Vig et al. 2022
- LLM-as-Judge评估：Li et al. 2024, Balepur et al. 2024

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

> **备注**：本文在245个新闻事件上实验，使用GPT-3.5/GPT-4o/Claude-3-Sonnet/LLaMA-3.1四种LLM评估，结果具有较强的一致性和说服力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MDCure: A Scalable Pipeline for Multi-Document Instruction-Following](mdcure_a_scalable_pipeline_for_multi-document_instruction-following.md)
- [\[ACL 2025\] LAQuer: Localized Attribution Queries in Content-grounded Generation](laquer_localized_attribution.md)
- [\[ACL 2025\] KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding](kodcode_a_diverse_challenging_and_verifiable_synthetic_dataset_for_coding.md)
- [\[ACL 2025\] MOSAIC: Multiple Observers Spotting AI Content](mosaic_multiple_observers_spotting_ai_content.md)
- [\[ACL 2025\] Principled Understanding of Generalization for Generative Transformer Models in Arithmetic Reasoning Tasks](principled_generalization_arithmetic.md)

</div>

<!-- RELATED:END -->
