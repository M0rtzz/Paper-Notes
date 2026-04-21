---
title: >-
  [论文解读] Demo: Guide-RAG: Evidence-Driven Corpus Curation for Retrieval-Augmented Generation in Long COVID
description: >-
  [NeurIPS 2025 (GenAI for Health Workshop)][医学图像][RAG] 系统评估了六种 RAG 语料库配置用于长新冠（Long COVID）临床问答，发现将临床指南与高质量系统综述结合的 GS-4 配置在 faithfulness、relevance 和 comprehensiveness 三维度上一致优于单指南和大规模文献库方案，并提出 Guide-RAG 框架和 LongCOVID-CQ 评估数据集。
tags:
  - NeurIPS 2025 (GenAI for Health Workshop)
  - 医学图像
  - RAG
  - 长新冠
  - 语料库策展
  - LLM-as-a-judge
  - 临床问答
---

# Demo: Guide-RAG: Evidence-Driven Corpus Curation for Retrieval-Augmented Generation in Long COVID

**会议**: NeurIPS 2025 (GenAI for Health Workshop)  
**arXiv**: [2510.15782](https://arxiv.org/abs/2510.15782)  
**代码**: 无  
**领域**: 医学图像 / 医学NLP  
**关键词**: RAG, 长新冠, 语料库策展, LLM-as-a-judge, 临床问答

## 一句话总结
系统评估了六种 RAG 语料库配置用于长新冠（Long COVID）临床问答，发现将临床指南与高质量系统综述结合的 GS-4 配置在 faithfulness、relevance 和 comprehensiveness 三维度上一致优于单指南和大规模文献库方案，并提出 Guide-RAG 框架和 LongCOVID-CQ 评估数据集。

## 研究背景与动机

**领域现状**：AI 聊天机器人在临床医学中快速普及，但对于复杂、异质、证据尚不充分的新兴疾病（如长新冠），设计有效的临床辅助 AI 面临重大挑战

**现有痛点**：
   - 纯 LLM 无检索容易产生幻觉和引用错误
   - 依赖 PubMed 等大规模文献库检索会引入大量低质量或无关内容（information overload），且存在"lost-in-the-middle"效应
   - 仅用单一临床指南又过于狭窄，无法覆盖快速演变的研究证据
   - 现有医学 QA 基准多为选择题格式，无法评估决策支持能力

**核心矛盾**：在证据快速演变、缺乏共识的新兴疾病场景下，检索范围过窄（miss evolving evidence）和过宽（引入噪声）之间存在根本张力

**切入角度**：以长新冠为案例，系统比较不同语料库策展策略对 RAG 性能的影响

**核心 idea**：将临床指南与少量高质量系统综述结合（GS-4），可以在窄共识文档和无过滤初级文献之间取得最佳平衡

## 方法详解

### 整体框架
以 GPT-4o 为基础 LLM，构建六种语料库配置进行对比实验。使用 LongCOVID-CQ（20 个专家临床问题）作为评测集，通过 LLM-as-a-judge pairwise 比较框架评估回答质量。

### 关键设计

1. **六种语料库配置**

    - **NR-0**: 无检索的 GPT-4o（对照组）
    - **G-1**: 仅用 AAPM&R 长新冠多学科指南（单一文档）
    - **GS-4**: 指南 + 3 篇高质量系统综述（共 4 个文档，核心方案）
    - **R-110**: 指南中 110 篇参考文献原文
    - **PM**: PubMed 全库检索（混合稀疏-稠密检索）
    - **WS**: GPT-4o web search（限定同行评审文献）
    - 设计动机：从"最窄"(G-1) 到"最宽"(PM) 覆盖完整谱系，验证"策展质量 vs 覆盖广度"的权衡

2. **LongCOVID-CQ 评测数据集**

    - 由长新冠专科医师生成 20 个临床问题，涵盖诊断、管理策略和机制
    - 不同于选择题基准，采用开放式问答反映真实临床信息需求

3. **三维度 LLM-as-a-judge 评估**

    - **Faithfulness**: 回答是否有检索文档支撑，可追溯引用
    - **Relevance**: 回答是否直接针对问题，无冗余内容
    - **Comprehensiveness**: 是否全面覆盖（长新冠有 200+ 已知症状）
    - 评分机制：GPT-4o 做 pairwise 比较，随机化呈现顺序避免位置偏差，显式控制长度偏差，win 记 100 / lose 记 0 / tie 各 50

### 损失函数 / 训练策略
本文是评估框架而非训练模型，不涉及损失函数。关键的技术选择包括：对小语料库（G-1, GS-4, R-110）采用稠密检索，对 PubMed 采用混合稀疏-稠密检索。

## 实验关键数据

### 主实验：Pairwise Win Rate (%)

| 配置 | vs G-1 | vs R-110 | vs PM | vs WS | 平均 Overall Win Rate |
|------|--------|----------|-------|-------|----------------------|
| **GS-4** | ~60 | ~62 | ~60 | ~65 | **57.5-65** |
| G-1 | - | 57.5 (faithfulness) | ~55 | ~58 | ~55 |
| R-110 | 42.5 | - | ~50 | ~52 | ~48 |
| PM | ~40 | ~50 | - | ~50 | **最佳 relevance** |

GS-4（指南+3 篇系统综述，仅 4 个文档）在 overall、faithfulness、comprehensiveness 三项上均排名第一。

### 消融/分析：不同维度表现

| 维度 | 最佳配置 | 说明 |
|------|---------|------|
| Faithfulness | GS-4 | 策展文档提供高质量可引用来源 |
| Relevance | PM | 大规模库提供更广泛的相关内容 |
| Comprehensiveness | GS-4 | 系统综述本身就是证据综合，覆盖广 |
| Overall | GS-4 | 三维度综合最优 |

### 关键发现
- GS-4 仅用 4 个文档即超越了 110 篇参考文献（R-110）和 PubMed 全库，说明**语料库质量远比数量重要**
- 综合指南优于其构成引用（G-1 > R-110 in faithfulness），说明综合文档提供了更好的检索锚定
- PubMed 在 relevance 上略优，但在 faithfulness 和 comprehensiveness 上表现差——大规模检索容易引入过度自信但证据基础薄弱的回答
- 专家评审发现 R-110 和 PM 产生的回答看似具体但实际存在误导（如推荐长新冠患者运动训练，这恰恰是专家告诫避免的）

## 亮点与洞察
- **"少即是多"的反直觉发现**：4 个精选文档击败了 39M+ PubMed 和 110 篇参考文献，说明在新兴疾病 RAG 中策展质量是核心，这个原则可推广到其他证据快速演变的领域
- **三维度评估框架设计严谨**：faithfulness 保证可信、relevance 避免信息过载、comprehensiveness 确保完整性，这套评估范式可复用于其他医学 RAG 系统
- **临床专家质性分析有力**：不只靠定量 win rate，还通过具体案例展示大规模语料库如何产生看似权威但实际危险的建议

## 局限与展望
- 仅聚焦长新冠单一疾病，未验证对其他新兴疾病的泛化性
- 仅用 GPT-4o 作为 judge，缺少与人类评估的对照验证
- 评测集仅 20 个问题，统计稳健性有限
- 不同语料库的检索策略不同（稠密 vs 混合），可能引入检索方法偏差
- 未做检索参数（chunk size、embedding 模型、reranking 阈值）的系统消融

## 相关工作与启发
- **vs Medical Graph RAG**: 后者用知识图谱结构化检索，Guide-RAG 用策展语料库，两者可互补——结构化知识图谱 + 策展文档可能是更好方案
- **vs RAGAS 框架**: RAGAS 提供自动化 RAG 评估指标，Guide-RAG 在此基础上增加了临床特定维度（comprehensiveness for 200+ 症状）
- **vs 一般医学 QA benchmarks (MedQA, PubMedQA)**: 这些用选择题，Guide-RAG 的 LongCOVID-CQ 是开放式专家问题，更贴近真实临床

## 评分
- 新颖性: ⭐⭐⭐ 系统比较有价值但方、法上无新模型/算法，更偏实证研究
- 实验充分度: ⭐⭐⭐ 六种配置对比全面，但评测集小、缺少人类评估
- 写作质量: ⭐⭐⭐⭐ 动机清晰、实验设计合理、专家分析有深度
- 价值: ⭐⭐⭐⭐ 为新兴疾病 RAG 系统的语料库设计提供了实践指导原则

<!-- RELATED:START -->

## 相关论文

- [Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](../../ACL2025/medical_imaging/omni_rag_medical.md)
- [On the Vulnerability of Applying Retrieval-Augmented Generation within Knowledge-Intensive Application Domains](../../ICML2025/medical_imaging/on_the_vulnerability_of_applying_retrieval-augmented_generation_within_knowledge.md)
- [MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](../../ACL2025/medical_imaging/medbiorag_semantic_search_and_retrieval-augmented_generation_with_large_language.md)
- [Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies](../../ACL2025/medical_imaging/urca_biomedical_evidence_extraction.md)
- [RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification](raxss_retrieval-augmented_sparse_sampling_for_explainable_variable-length_medica.md)

<!-- RELATED:END -->
