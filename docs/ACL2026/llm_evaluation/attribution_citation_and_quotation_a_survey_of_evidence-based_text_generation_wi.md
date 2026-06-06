---
title: >-
  [论文解读] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models
description: >-
  [ACL 2026][LLM评测][证据基础文本生成] 本文系统综述了 134 篇关于 LLM 证据基础文本生成的论文，首次提出统一分类学（归因方式 × 引用特征 × 任务），分析了 300 个评估指标并归纳为七大维度六种方法，为该碎片化领域提供了全景式参考框架。
tags:
  - "ACL 2026"
  - "LLM评测"
  - "证据基础文本生成"
  - "引用归因"
  - "LLM可信性"
  - "评估框架"
  - "RAG"
---

# Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models

**会议**: ACL 2026  
**arXiv**: [2508.15396](https://arxiv.org/abs/2508.15396)  
**代码**: [https://github.com/faerber-lab/AttributeCiteQuote](https://github.com/faerber-lab/AttributeCiteQuote)  
**领域**: 综述/NLP  
**关键词**: 证据基础文本生成, 引用归因, LLM可信性, 评估框架, RAG

## 一句话总结

本文系统综述了 134 篇关于 LLM 证据基础文本生成的论文，首次提出统一分类学（归因方式 × 引用特征 × 任务），分析了 300 个评估指标并归纳为七大维度六种方法，为该碎片化领域提供了全景式参考框架。

## 研究背景与动机

**领域现状**：LLM 面临幻觉生成和知识局限等可信性挑战，越来越多研究关注"证据基础文本生成"——让 LLM 输出可追溯到支撑证据。但该领域高度碎片化：有的叫"引用"（citation, 75% 论文使用），有的叫"归因"（attribution, 62%），有的叫"引述"（quotation, 13%），且各自评估实践孤立。

**现有痛点**：(1) 缺乏统一术语和分类体系，研究者难以定位自己的工作；(2) 评估标准不一致——300 个指标但仅 2 个框架（ALCE、G-Eval）被多篇论文复用；(3) RAG 虽然流行但仅是七种相关方法之一，过度聚焦 RAG 会遗漏其他重要方法。

**核心矛盾**：快速增长的研究兴趣（2024 年论文数是 2023 年的 3.4 倍，75% 论文发表在 2023 年之后）vs. 缺乏统一视角来整合和比较不同方法。

**本文目标**：提供第一个专门针对 LLM 证据基础文本生成的系统性综述，建立统一分类学，分析评估实践，识别研究趋势和未来方向。

**切入角度**：采用 PRISMA 协议进行系统映射研究，从 805 篇去重论文中筛选出 134 篇相关论文，使用多面分类方法构建分类学。

**核心 idea**：将"引用"、"归因"和"引述"统一为"证据基础文本生成"范式，通过三维分类学和七维评估框架为碎片化领域提供系统化视角。

## 方法详解

### 综述框架/分类体系

### 整体分类学

本文提出三维独立分类学，捕捉证据基础文本生成的核心设计选择：

- **维度一：归因方式** — 内容如何关联到证据（参数式 vs. 非参数式）
- **维度二：引用特征** — 证据的形式和呈现（模态、粒度、样式、可见性、频率）
- **维度三：任务** — 应用场景（QA、接地生成、摘要等六类任务）

### 关键分类维度

1. **归因方式 (Attribution Approach)**:

    - 功能：描述 LLM 如何将生成内容关联到支撑证据
    - 核心分类：**参数式**（25 篇）——纯 LLM（利用现有能力，占参数式的 72%）、模型中心（修改架构/训练）、数据中心（策划/增强数据）。**非参数式**（126 篇）——后检索（58%，RAG 为代表）、后生成（18%，先生成再检索证据）、生成中（4%，动态判断是否需要检索）、上下文内（20%，用户直接提供证据）
    - 关键发现：参数式归因严重不足，模型和数据中心方法尤其缺乏关注。非参数式中后检索占主导，但生成中归因（如 Self-RAG）是有前景但被低估的新方向

2. **引用特征 (Citation Characteristics)**:

    - 功能：描述证据的具体呈现方式
    - 核心分类：**引用模态**——文本 96%、图表、表格、视觉。**证据级别**——文档级 43%、段落级 40%、句子级 12%、token 级 2%。**引用样式**——行内引用 62%、引用报告、段落展示、叙述性引用、高亮梯度、引述。**可见性**——最终回复 91% vs. 中间文本。**频率**——多重引用 64% vs. 单一引用
    - 关键发现：非文本证据模态（图表、表格、图像）严重未开发（仅 4%）；细粒度证据（句子级、token 级）虽占比小但增长更快

3. **任务分布 (Task Landscape)**:

    - 功能：映射该领域的应用场景
    - 核心分类：QA 和接地文本生成为主导任务，摘要、事实验证为中等，引用文本生成和相关工作生成为新兴任务
    - 关键发现：评估实践主要围绕 QA 任务开发，可能不适用于新兴任务（如引用文本生成需要更多关于引用选择推理的评估）

### LLM 集成方式

**训练**（45% 论文使用）：以监督微调为主，主要用于改善归因行为。预训练较少使用。
**提示**（78% 论文使用）：以零/少样本提示为主。专门针对引用行为的策略包括 chain-of-citation、chain-of-quote 和 conflict-aware 提示。

## 实验关键数据

### 文献分析/覆盖范围

### 评估指标体系

**300 个指标按七大评估维度分类**

| 评估维度 | 何时使用 | 主要方法 | 代表指标（复用次数） |
|---------|---------|---------|-------------------|
| 归因 (Attribution) | 无标注证据时 | NLI 为主 | Citation NLI P/R/F1 (33/33/16), Auto-AIS (11), FActScore (7) |
| 引用 (Citation) | 有标注证据时 | 检索为主 | Citation Retrieval P/R/F1 (6/6/5), Citation Accuracy (2) |
| 正确性 (Correctness) | 始终需要 | 词汇重叠/NLI | Exact Match (12), BLEU-N (5), Claim Recall (17) |
| 语言质量 | 模型被修改时 | LLM-as-Judge | G-Eval Fluency (4), MAUVE (21), Perplexity (4) |
| 保留度 | 后生成归因时 | 词汇重叠 | Preservation-Levenshtein (3), F1-AP (2) |
| 相关性 | 用户场景 | LLM-as-Judge | G-Eval Relevance (3), RAGAS (2) |
| 检索 | 非参数归因时 | 检索指标 | P@k (4), R@k (4), MRR (3) |

### 评估指导原则

| 维度类别 | 何时评估 | 说明 |
|---------|---------|------|
| **核心维度** | 归因或引用 + 正确性 | 始终应评估正确性；归因和引用根据证据可用性二选一 |
| **上下文维度** | 语言质量、保留度、相关性、检索 | 取决于任务设计和系统架构 |

### 关键发现

- 仅 2 个框架（ALCE、G-Eval）和 2 个基准被多篇论文复用，评估标准化严重不足
- 134 篇论文中识别出 19 个框架、11 个基准和 231 个数据集
- 文本在引用模态中占 96%，多模态证据几乎空白
- 参数式归因虽然对理解模型内部知识和数据溯源至关重要，但严重被忽视
- 人工评估在正确性维度仍占主导，反映了自动指标在捕捉语义错误方面的局限

## 亮点与洞察

- 将"引用"、"归因"和"引述"统一为"证据基础文本生成"是重要的概念贡献，消除了长期的术语混乱
- 七维评估指南（Table 1）为实践者提供了清晰的指标选择建议——核心维度 vs. 上下文维度的区分极为实用
- 参数式归因的三分法（纯 LLM / 模型中心 / 数据中心）比此前的二分法更精细
- 识别出生成中归因（in-generation）作为有前景但被低估的方向——仅占 4% 但代表了更紧密集成检索和生成的趋势
- 指出了引用行为可能存在类似人类作者的偏差，呼吁研究 LLM 引用推理的可解释性

## 局限与展望

- 单一搜索字符串可能遗漏部分相关研究（敏感性分析显示仅 4% 额外发现）
- 仅覆盖英文论文，可能低估非英语研究
- 人工筛选和分类不可避免引入一定主观性
- **四大未来方向**：(1) 参数式和混合归因的深入研究；(2) 标准化评估框架（当前 300 指标仅 2 框架被复用）；(3) 可解释的引用推理——理解 LLM 为何选择特定来源；(4) 多模态证据支持——从 96% 文本向图表、表格、图像扩展

## 相关工作与启发

- **vs Li et al. (2023a)**: 唯一先前相关综述，但已严重过时（75%+ 论文发表在其之后），且未覆盖完整范式
- **vs Huang & Chang (2024)**: 立场论文仅强调引用重要性，未系统综述
- **vs RAG surveys**: RAG 综述仅覆盖后检索这一种方法，本文覆盖七种归因方式
- **vs 幻觉/接地综述**: 聚焦不同侧面，本文专注于证据生成而非检测

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次对证据基础文本生成进行全面统一分类，三维分类学设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 134 篇论文、300 个指标、19 框架、231 数据集、11 基准的覆盖范围极全面
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，多维度分析平衡，每节附有精炼的 "Takeaways" 总结
- 价值: ⭐⭐⭐⭐⭐ 对快速增长但碎片化领域的全景梳理，对研究者和实践者都有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)
- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/llm_evaluation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)

</div>

<!-- RELATED:END -->
