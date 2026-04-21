---
title: >-
  [论文解读] Hierarchical Retrieval with Evidence Curation for Open-Domain Financial QA
description: >-
  [ACL 2025][NLP理解][金融问答] HiREC 提出分层检索与证据策展框架，先检索相关文档再从中选取段落，并通过过滤无关段落 + 自动生成补充查询来补全缺失信息，在包含 14.5 万篇 SEC 文档的 LOFin 基准上相比最优 RAG 基线提升 13%+ 答案准确率。
tags:
  - ACL 2025
  - NLP理解
  - 金融问答
  - 分层检索
  - 证据策展
  - RAG
  - 标准化文档
---

# Hierarchical Retrieval with Evidence Curation for Open-Domain Financial QA

**会议**: ACL 2025  
**arXiv**: [2505.20368](https://arxiv.org/abs/2505.20368)  
**代码**: [GitHub](https://github.com/deep-over/LOFin-bench-HiREC)  
**领域**: NLP理解  
**关键词**: 金融问答, 分层检索, 证据策展, RAG, 标准化文档

## 一句话总结

HiREC 提出分层检索与证据策展框架，先检索相关文档再从中选取段落，并通过过滤无关段落 + 自动生成补充查询来补全缺失信息，在包含 14.5 万篇 SEC 文档的 LOFin 基准上相比最优 RAG 基线提升 13%+ 答案准确率。

## 研究背景与动机

**领域痛点**：金融标准化文档（如 SEC 年报 10-K）跨公司、跨时期使用统一模板，表结构高度相似（如 Amazon、Meta、Walmart 的运营收入表仅数值不同），传统 RAG 方法难以区分这些近重复文本，导致检索混淆和冗余。

**数据集不足**：现有金融 QA 基准（FinanceBench 仅 150 题/368 文档，SEC-QA 无固定测试集）规模小、不开放，无法反映真实场景。

**比较类问题的挑战**：金融 QA 中大量比较类问题（如"Amazon 和 Walmart 2023 年运营收入差多少"）需要从多个文档中完整检索所有必要证据，单次检索往往遗漏关键信息。

## 方法详解

### 整体框架

HiREC 包含两大核心组件，采用迭代式流程（最多 $i_{\max}=3$ 轮）：

**A. 分层检索（Hierarchical Retrieval）**
- Step 1: 文档级检索 → 缩小搜索空间
- Step 2: 段落级检索 → 从候选文档中选取最相关段落

**B. 证据策展（Evidence Curation）**
- Step 3: 段落过滤 → 移除无关段落
- Step 4: 可回答性判断 → 评估证据是否充分
- Step 5: 补充查询生成 → 若不充分则生成新查询触发下一轮检索

**C. 答案生成**
- 数值题用 Program-of-Thought (PoT)，文本题用 Chain-of-Thought (CoT)

### 关键设计

**1. 文档索引与检索（Document Retriever）**

标准化文档内容冗长且格式统一，单一向量难以捕获所有关键信息。HiREC 采用**封面摘要索引**策略：
- 用 LLM 从每篇文档封面提取核心区分信息（公司名、报告类型、财年周期）生成摘要 $d'$
- 用 bi-encoder（E5 模型）预计算摘要嵌入 $\mathbf{v}_d = E^D(d')$ 存入文档库
- 检索时：查询 $q$ → LLM 转换为精炼查询 $q'$（去除干扰性金融术语）→ 密集检索取 $k'_D$ 候选 → cross-encoder（DeBERTa-v3）重排取 top-$k_D$（$k_D=5$）

**2. 段落检索器（Passage Retriever）**

- 在检索到的文档集 $\mathcal{D}_r$ 内，用 cross-encoder 对每个段落 $p$ 计算 $\text{CrossEncoder}^P(q, p)$，取 top-$k_P$（$k_P=5$）段落
- **关键改进**：标准预训练 reranker 处理金融表格能力差。本文在 FinQA 训练集上**微调 cross-encoder**：对每个问题 $q$ 和证据表格 $p$，采样 $n_{\text{neg}}=8$ 个负样本（非证据页的表格），用二元交叉熵损失训练：

$$\mathcal{L} = \sum_{(q,p) \in \mathcal{X}} \left[ -\log(\text{CE}^P(q,p)) - \sum_{p' \in \mathcal{P}^-} \log(1 - \text{CE}^P(q,p')) \right]$$

**3. 证据策展三步流程**

三个模块用单次 LLM 调用完成：

- **段落过滤器**：从 $\mathcal{P}_r$ 中移除与问题无关的段落，保留至多 $k'_P=10$ 个段落组成 $\mathcal{P}_f$，同时考虑之前迭代中已确认相关的段落
- **可回答性检查器**：评估 $\mathcal{P}_f$ 是否包含足够信息回答问题。若充分则进入答案生成；否则触发补充检索
- **补充问题生成器**：分析 $\mathcal{P}_f$ 中的信息缺口，生成补充查询 $q_c$，用于下一轮分层检索（例：只检索到 Amazon 运营收入 → 生成"Walmart 2023 运营收入"的补充查询）

**4. LOFin 基准构建**

- **语料库**：从 SEC EDGAR 收集 S&P 500 公司 2001-2025 年的 10-K/10-Q/8-K 文件，共 145,897 篇/516 家公司
- **QA 对**：从 FinQA（闭域→开域转换）+ FinanceBench（直接采用）+ SEC-QA 多文档模板手工构造，共 1,595 对
- **证据标注**：BM25 + NLI 两步自动匹配 + 人工校验

### 损失函数 / 训练策略

- 段落检索器微调：DeBERTa-v3，$n_{\text{neg}}=8$，batch size 128，3 epochs，lr $2 \times 10^{-7}$，单卡 RTX 4090
- 其余 LLM 模块（查询转换、摘要、证据策展）使用 Qwen-2.5-7B-Instruct
- 答案生成使用 GPT-4o

## 实验关键数据

### 主实验：LOFin-1.4k 上的全面对比

| 方法 | Page Recall | Answer Acc | 平均段落数 |
|------|-----------|-----------|-----------|
| GPT-4o (Zero-shot) | - | 13.92 | - |
| Perplexity | - | 10.55 | - |
| Self-RAG | 18.96 | 7.63 | 10.0 |
| RQ-RAG | 18.54 | 8.34 | 36.0 |
| IRCoT | 25.15 | 22.31 | 20.0 |
| Dense | 34.78 | 29.22 | 10.0 |
| HHR | 33.31 | 28.67 | 10.0 |
| **HiREC** | **45.35** | **42.36** | **3.7** |

HiREC 相比 Dense 提升 10%+ page recall 和 13%+ answer accuracy，且仅用 3.7 个段落。

### 消融实验

| 配置 | Page Precision | Page Recall | Answer Acc |
|------|---------------|-------------|-----------|
| HiREC (完整) | 21.79 | 45.35 | 42.36 |
| w/o HR (无分层检索) | 14.75 | 34.16 | 32.76 |
| w/o EC (无证据策展) | 4.70 | 41.41 | 36.70 |
| w/o Fine-tuning | 21.07 | 42.77 | 40.13 |
| w/o Filter (无过滤) | 8.43 | 50.19 | 42.08 |

- 去除分层检索（HR）影响最大（Acc -9.6%），证明文档级预筛选的核心价值
- 去除过滤器虽然 recall 最高（50.19），但 accuracy 未提升，说明引入了冲突/错误信息
- 即使不微调 reranker，HiREC 仍超越 Dense 基线 10%+

### 跨生成器分析

| 方法 | Qwen-2.5-7B | DeepSeek-14B | GPT-4o |
|------|------------|-------------|--------|
| Dense | 23.87 | 30.77 | 29.22 |
| HiREC | 32.32 | **38.76** | **42.36** |

HiREC + DeepSeek-14B (38.76) 超越 Dense + GPT-4o (29.22) 达 9%+，说明高质量检索可弥补生成器能力差距。

### 关键发现

- **分层检索是性能核心**：通过先定位正确公司/文档，大幅减少近重复文本的混淆
- **迭代证据策展持续改善**：每轮迭代后 recall 和 precision 均稳步提升，同时每查询段落数减少
- **成本效率优势明显**：HiREC 检索阶段 token 消耗仅为 IRCoT 的 ~45%，生成阶段仅 ~30%
- **小模型可替代大模型做策展**：Qwen-2.5-7B 即可有效执行证据策展，无需高昂 API 费用

## 亮点与洞察

1. **封面摘要索引**是针对标准化文档的巧妙设计——用最具区分力的信息（公司名+时期）而非全文来做文档级检索
2. **补充查询生成**解决了比较类问题的本质困难——自动识别信息缺口并补全，而非依赖用户重新提问
3. **LOFin 基准**：14.5 万文档的大规模开放域金融 QA 数据集，比现有最大基准大 100 倍
4. **检索质量 > 生成器能力**：用小模型 + 优质检索胜过大模型 + 普通检索的发现有很强实践意义

## 局限与展望

1. **领域特异性**：封面摘要索引策略依赖金融文档的结构化特征，泛化到其他标准化文档（如法律合同、医疗记录）需要领域适配
2. **LLM 依赖**：证据策展的三个模块均依赖 LLM，在极端成本敏感场景下开销仍不可忽视
3. **多文档推理上限**：当前最多处理 $k_D=5$ 个文档，对需要跨数十份文档推理的复杂分析场景可能不足
4. **表格理解**：虽已微调段落检索器，但金融表格的复杂计算推理仍是核心瓶颈（数值表类准确率仅 37%）

## 相关工作与启发

- **金融 RAG**：GraphRAG、HybridRAG 等图结构方法与本文的分层方法互补
- **迭代检索**：IRCoT、Self-RAG 使用之前检索上下文作为后续查询输入；HiREC 不同——专门发现缺失信息而非复用已有上下文
- **启发**：(1) 标准化文档检索应"先识别文档再检索段落"；(2) 补充查询生成可应用于任何需要多证据汇聚的 QA 场景

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|------------|------|
| 新颖性 | 7 | 分层检索 + 证据策展 + 补充查询的组合设计有效 |
| 技术深度 | 7 | 多阶段流水线设计完整，微调策略合理 |
| 实验充分性 | 9 | 大规模基准 + 多方法对比 + 消融 + 跨模型分析 + 成本分析 |
| 写作质量 | 8 | 框架图清晰，伪代码完整，错误类型分析深入 |
| 实用价值 | 8 | LOFin 基准和 HiREC 框架可直接落地金融场景 |
| 总分 | 7.5 | 面向真实金融场景的实用 RAG 框架，工程与学术价值兼具 |

<!-- RELATED:START -->

## 相关论文

- [Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering](embqa_embedding_odqa.md)
- [NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](neusym_rag_pdf_qa.md)
- [Towards a More Generalized Approach in Open Relation Extraction](generalized_open_relation_extract.md)
- [A Variational Approach for Mitigating Entity Bias in Relation Extraction](a_variational_approach_for_mitigating_entity_bias_in_relation_extraction.md)
- [Disambiguate First, Parse Later: Generating Interpretations for Ambiguity Resolution in Semantic Parsing](disambiguate_first_parse_later_generating_interpretations_for_ambiguity_resoluti.md)

<!-- RELATED:END -->
