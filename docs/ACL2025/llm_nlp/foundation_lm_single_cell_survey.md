---
title: >-
  [论文解读] A Survey on Foundation Language Models for Single-cell Biology
description: >-
  [ACL 2025 (Long Paper)][LLM/NLP][single-cell biology] 首篇从语言建模视角系统综述单细胞生物学基础语言模型，将现有工作划分为PLM（从头预训练）和LLM（利用已有大模型）两大类，全面分析tokenization策略、预训练/微调范式以及下游任务体系，并指出当前领域在数据质量、统一评测和scaling law方面的核心挑战。
tags:
  - ACL 2025 (Long Paper)
  - LLM/NLP
  - single-cell biology
  - foundation language model
  - pre-trained language model
  - gene expression
  - tokenization
---

# A Survey on Foundation Language Models for Single-cell Biology

**会议**: ACL 2025 (Long Paper)  
**arXiv**: 无  
**代码**: 无  
**领域**: 计算生物学 / NLP交叉  
**关键词**: single-cell biology, foundation language model, pre-trained language model, gene expression, tokenization  

## 一句话总结
首篇从语言建模视角系统综述单细胞生物学基础语言模型，将现有工作划分为PLM（从头预训练）和LLM（利用已有大模型）两大类，全面分析tokenization策略、预训练/微调范式以及下游任务体系，并指出当前领域在数据质量、统一评测和scaling law方面的核心挑战。

## 研究背景与动机

**跨领域迁移趋势**: 语言模型（BERT、GPT等）的成功已渗透至计算生物学领域。研究者发现可以将**细胞类比为"句子"，基因类比为"词/token"**，从而用语言模型构建统一的单细胞基础模型。

**统一表示的价值**: 这类模型能获得跨数据集、跨任务的通用细胞表示，在细胞类型注释、基因扰动预测、药物响应等下游任务上超越传统专用模型，避免了为每个任务单独设计模型的高昂成本。

**综述动机**: 已有综述大多从Transformer架构角度分析单细胞模型（如Lan et al. 2024, Szalata et al. 2024），缺乏从"语言建模"这一NLP核心范式出发的系统分析。本文填补这一空白，用NLP社区更熟悉的PLM vs LLM二分法重新组织该领域知识。

## 方法详解

### 整体框架
将单细胞基础语言模型分为两大阵营：

1. **Single-cell PLMs**（预训练语言模型）：将基因视为token、细胞视为句子，从头在大规模单细胞数据上预训练。典型代表：scBERT、scGPT、GeneFormer、scFoundation等。

2. **Single-cell LLMs**（大语言模型）：不从头预训练，利用已有通用LLM（GPT-2/3.5/4、LLaMA、T5），通过将细胞数据转为文本后微调或直接推理。典型代表：Cell2Sentence、GenePT、scELMo等。

### 关键设计

1. **Tokenization策略（PLM端）**

    将细胞的基因表达矩阵 (N x G) 转化为语言模型可理解的格式，三大方向：
    - **离散化**: Binning将连续表达值离散为整数区间（scBERT, CellLM）；Rank Value Encoding按表达量排序用基因词表编码（GeneFormer系列）
    - **连续嵌入**: 利用蛋白质语言模型获取基因嵌入（UCE, scPRINT）；可学习层映射（CellPLM）；分层贝叶斯下采样处理稀疏性（scFoundation）
    - **辅助信息融入**: 整合元数据（细胞状态、器官来源、供体信息、测序技术）或蛋白质基础模型的先验知识

2. **预训练范式（PLM端）**

    - **掩码语言建模（MLM）**: 最主流，随机掩码15-30%基因后重建。采用者：scBERT, UCE, GeneFormer, CellPLM, scFoundation, Nicheformer
    - **下一个token预测（NTP）**: 自回归预训练，仅tGPT和scGPT采用。在单细胞领域不流行，原因：(1) 数据规模比文本仍小得多；(2) 基因表达稀疏导致大量ground truth为零，模型倾向学到平凡零值解
    - **多任务预训练**: 在MLM基础上叠加对比学习、分类、细胞生成、元数据预测、去噪等监督信号（CellLM, LangCell, scCello, scPRINT, scMulan, GeneCompass, CellFM）

3. **细胞-文本转换与微调范式（LLM端）**

    **转换方式**:
    - Cell-to-Sentence：按表达量排序选top-100基因名拼成自然语言句子（Cell2Sentence, CHATCELL, CELLama）
    - Text-level Gene Embeddings：用LLM获取每个基因的功能描述嵌入，再用表达值加权组合（GenePT, scELMo, scInterpreter）

    **微调范式**:
    - 指令微调：将任务转为QA格式（Cell2Sentence, CHATCELL）
    - 嵌入微调：直接利用细胞/基因嵌入进行监督微调（目前主流）
    - 免调优：LLM作为agent直接生成Python代码执行分析（scChat）

## 实验

### 模型对比总览

| 模型 | 类型 | Tokenization | 预训练范式 | 数据规模 |
|------|------|-------------|-----------|---------|
| scBERT | PLM | Binning | MLM | 1M cells |
| GeneFormer | PLM | Rank Value | MLM | 27.4M cells |
| scGPT | PLM | Binning+Meta | NTP | 33M cells |
| scFoundation | PLM | Downsampling | MLM | 50M cells |
| GeneCompass | PLM | Ranking+Meta | Multi-task | 126M cells |
| CellFM | PLM | Padding+MLP | Multi-task | 100M cells |
| Nicheformer | PLM | Ranking+Meta | MLM | 57M cells |
| Cell2Sentence | LLM | Cell-to-Text | 指令微调 | GPT-2 base |
| GenePT | LLM | Text Embedding | 嵌入微调 | GPT-3.5 base |
| CELLama | LLM | Cell-to-Text | 指令微调 | LLaMA-13B base |

### 下游任务体系

| 任务层级 | 具体任务 |
|----------|---------|
| **细胞级** | 细胞类型注释、新细胞类型发现、批次效应校正、细胞聚类、多组学整合、细胞生成 |
| **基因级** | 基因网络分析、基因扰动预测、基因功能/表达预测 |
| **药物相关** | 药物敏感性预测、药物响应建模 |
| **空间相关** | 空间转录组补全、空间标签预测、空间组成分析 |

### 关键发现
- MLM在单细胞领域显著优于NTP：数据规模和稀疏性是NTP表现不佳的主因
- 多任务预训练整合了自监督和监督信号，通常效果最好
- 数据规模从1M扩展到126M cells带来了一致的性能提升，但scaling law尚不明确
- 仅scGPT和scELMo验证了多组学整合能力，该方向空间广阔
- Cell-to-Sentence方式简单直观但信息损失大（仅保留top-100基因），Text-level Embedding更忠实但计算开销更高

## 亮点
- **清晰的分类体系**: PLM vs LLM的二分法，配合tokenization三策略、预训练三范式、LLM三种微调模式的精细子分类，使读者快速建立全景视图
- **语言建模新视角**: 首次完全从NLP的语言建模角度审视单细胞基础模型，而非传统的生物信息学视角，对NLP社区更友好
- **Cell=Sentence类比的系统化**: 将基因视为token、细胞视为句子的统一框架简洁优雅，是跨领域知识迁移的典范
- **系统的挑战分析**: 从数据质量（稀疏性/批次效应/多组学缺乏）、模型设计（统一tokenizer/scaling law未现）、评测协议（缺统一基准）三方面深入分析

## 局限性
- 综述本身无原创实验，对各模型的经验效果缺乏横向量化对比（因模型大多在私有数据集上评测）
- 作者自己承认侧重技术分析，对设计背后的**生物学动机**讨论不足——为什么某种tokenization在生物学上更合理？
- 时效性受限：后续大量新模型（如CellVerse等）涌现但未覆盖
- 缺乏统一benchmark是整个领域的痛点，论文指出但未提出具体解决方案
- 现有最大单细胞PLM不到1B参数，scaling行为是否与NLP领域类似仍不清楚

## 相关工作
- **vs Lan et al. (2024), Szalata et al. (2024)**: 从Transformer架构角度分析，本文首次从语言建模视角（PLM vs LLM二分法）组织知识
- **vs LLM4Cell (Dip et al., 2025)**: 后者覆盖了agentic models（如scChat），时间更晚
- **vs 传统综述**: 传统生物信息学综述关注实验方法和生物学洞见，本文关注建模范式和NLP技术迁移

## 评分
- 新颖性: ⭐⭐⭐ 综述无新方法，但首次从语言建模视角切入是一个有价值的新角度
- 实验充分度: ⭐⭐⭐ 无原创实验，模型总结表格较完整但缺乏量化对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰、分类体系完整、配图直观，适合快速入门
- 对我的价值: ⭐⭐⭐ 对了解NLP与计算生物学交叉领域有参考价值

<!-- RELATED:START -->

## 相关论文

- [Large Language Models in Bioinformatics: A Survey](large_language_models_in_bioinformatics_a_survey.md)
- [Knowledge Boundary of Large Language Models: A Survey](knowledge_boundary_survey.md)
- [Recent Advances in Speech Language Models: A Survey](recent_advances_in_speech_language_models_a_survey.md)
- [A Survey of Large Language Models in Psychotherapy: Current Landscape and Future Directions](a_survey_of_large_language_models_in_psychotherapy_current_landscape_and_future_.md)
- [When Large Language Models Meet Speech: A Survey on Integration Approaches](when_large_language_models_meet_speech_a_survey_on_integration_approaches.md)

<!-- RELATED:END -->
