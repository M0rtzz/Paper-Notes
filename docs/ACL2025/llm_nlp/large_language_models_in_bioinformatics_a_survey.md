---
title: >-
  [论文解读] Large Language Models in Bioinformatics: A Survey
description: >-
  [ACL 2025][LLM/NLP][survey] 本文系统综述了大语言模型在生物信息学四大领域（DNA/基因组、RNA、蛋白质、单细胞分析）的应用进展，涵盖 30+ 代表性模型的架构、任务和数据集，并讨论了数据稀缺、计算复杂度、跨组学整合等核心挑战和未来方向。
tags:
  - ACL 2025
  - LLM/NLP
  - survey
  - bioinformatics
  - DNA
  - RNA
  - protein
  - single-cell
---

# Large Language Models in Bioinformatics: A Survey

**会议**: ACL 2025  
**arXiv**: [2503.04490](https://arxiv.org/abs/2503.04490)  
**代码**: 无  
**领域**: LLM效率  
**关键词**: survey, bioinformatics, DNA, RNA, protein, single-cell

## 一句话总结

本文系统综述了大语言模型在生物信息学四大领域（DNA/基因组、RNA、蛋白质、单细胞分析）的应用进展，涵盖 30+ 代表性模型的架构、任务和数据集，并讨论了数据稀缺、计算复杂度、跨组学整合等核心挑战和未来方向。

## 研究背景与动机

**领域现状**：大语言模型在 NLP 中取得了突破性进展，研究者已开始将 LLM 应用于生物信息学的各种任务中——包括 DNA 序列功能预测、RNA 结构预测、蛋白质功能推断和单细胞转录组分析。近年来生物信息学的 LLM 数量呈爆发式增长。

**现有痛点**：生物数据与自然语言数据有本质差异（序列类型、数据规模、标注成本等），将 LLM 有效适配到生物信息学任务面临独特挑战。现有各种方法分散在不同子领域，缺乏系统性的梳理和比较分析。

**核心矛盾**：一方面 LLM 在生物序列建模上展现了巨大潜力，另一方面数据稀缺、计算资源需求高、跨模态整合困难等问题制约了其进一步发展。此前的综述未能全面覆盖 DNA、RNA、蛋白质和单细胞四大领域的最新进展。

**本文目标** 提供一个全面、系统的综述，覆盖 LLM 在生物信息学各子领域的代表性方法，归纳架构范式，分析共性挑战，指明未来方向。

**切入角度**：按生物序列类型（DNA → RNA → 蛋白质 → 单细胞）分章组织，按模型架构范式（Encoder-only / Decoder-only / Encoder-Decoder）分类，形成清晰的二维综述框架。

**核心 idea**：首个全面覆盖 LLM 在 DNA、RNA、蛋白质和单细胞四大生物信息学领域的系统综述，提供模型分类、计算成本量化和未来方向展望。

## 方法详解

### 整体框架

这是一篇综述论文，不涉及新方法提出。组织结构为：预备知识（三种架构范式）→ DNA 和基因组学 → RNA（结构与功能）→ 蛋白质（预测与设计）→ 单细胞分析 → 挑战与未来方向。核心贡献是对 30+ 代表性模型的系统分类和对比。

### 关键设计

1. **三种架构范式的系统对比**:

    - Encoder-only（如 DNABERT、ProteinBERT、scBERT）：双向自注意力捕捉序列上下文，擅长表示学习和下游分类/功能预测任务。平均训练资源需求适中（~43GB 显存，~14 天）
    - Decoder-only（如 ProGen2、Evo、DNAGPT）：自回归生成方式，适合序列生成和从头设计任务。训练最快（~46GB，~5 天），但单向注意力难以捕捉长程双向依赖
    - Encoder-Decoder（如 RoseTTAFold、ESM-3、scGPT）：序列到序列转换，适合跨模态映射和需要双向理解的结构化输出。功能最强但计算资源需求也最高（~81GB，~40 天）

2. **四大应用领域梳理**:

    - DNA/基因组：从 DNABERT（功能预测）到 DNABERT2（跨物种）到 Evo（统一 DNA/RNA/蛋白质），发展脉络是从单物种单任务走向跨物种跨分子的大统一模型
    - RNA：二级结构预测（RiNALMo、ERNIE-RNA 效果最优）→ 三级结构预测（RhoFold+ 端到端）→ 功能预测 → 序列生成（RNA-GPT、RNA-DCGen）
    - 蛋白质：结构预测（AlphaFold2/3 达到原子级精度）→ 功能推断（ESM-1b、ProtTrans）→ 设计工程（ProtGPT2、ESM-3 多模态预测与设计），形成预测-理解-设计的完整链条
    - 单细胞：scBERT、Geneformer（2990 万转录组预训练）、scFoundation（1 亿参数）、scGPT（3300 万转录组预训练 + 多组学），实现细胞类型标注、扰动预测、批次整合等任务的迁移学习

3. **挑战与未来方向总结**:

    - 三大挑战：数据稀缺与偏差（偏向模式生物和常见疾病）、计算复杂度（长生物序列对标准 Transformer 不友好）、跨组学整合不足（大多模型仍在单模态上训练）
    - 三大方向：混合 AI 模型（LLM + GNN + 知识图谱 + 符号 AI）→ 多模态跨组学整合（同时处理 DNA + RNA + 蛋白质 + 表观遗传数据）→ 临床转化（模型验证、合规、伦理）

### 损失函数 / 训练策略

综述论文不涉及具体训练策略。综述的共性总结是：自监督预训练（MLM 或自回归）+ 下游任务微调是主流范式。

## 实验关键数据

### 主实验

本文为综述，无自有实验。以下汇总综述中的代表性模型对比：

| 模型 | 架构 | 领域 | 关键成就 |
|------|------|------|----------|
| AlphaFold2 | 特殊架构 | Protein | CASP14 原子级精度蛋白质结构预测 |
| ESM-3 | Enc-Dec | Protein | 多模态蛋白质预测与设计 |
| DNABERT2 | Enc-only | DNA | 多物种基因组功能高效分析 |
| Evo | Dec-only | DNA/RNA/Protein | 首个跨 DNA/RNA/蛋白质的统一基础模型 |
| scGPT | Enc-Dec | scRNA | 3300 万单细胞预训练，多组学分析 |
| RhoFold+ | Enc-only | RNA | 端到端 RNA 三维结构预测 |

### 模型规模与计算成本统计

| 架构类型 | 平均显存/设备 | 平均训练时长 |
|----------|-------------|-------------|
| Encoder-only | ~43 GB | ~14 天 |
| Decoder-only | ~46 GB | ~5 天 |
| Encoder-Decoder | ~81 GB | ~40 天 |

### 关键发现

- Encoder-only 模型在分类和功能预测任务中表现稳健，训练效率适中，是目前最常用的架构
- Decoder-only 模型训练最快但对长程双向依赖捕捉弱，主要用于序列生成和从头设计
- Encoder-Decoder 功能最强但资源消耗最大，是蛋白质结构预测和单细胞基础模型的首选
- 单模态训练是当前主流限制，跨组学整合（DNA+RNA+蛋白质+表观遗传）是关键突破方向
- 数据稀缺和标注偏差（偏向模式生物和常见疾病）制约了模型的泛化能力

## 亮点与洞察

- **全面的模型矩阵表**（Table 1）：汇总了 30+ 模型的架构、数据集、任务和能力，是一份高效的参考速查手册，省去大量文献调研时间
- **计算成本量化**：罕见地提供了不同架构的平均 GPU 显存和训练时间统计，对资源有限的研究者选型有切实的参考价值
- **清晰的未来方向定位**：混合 AI 模型（LLM + GNN + 知识图谱）、多模态跨组学整合、临床转化三大方向指引明确

## 局限与展望

- **范围限制**：未覆盖表观基因组学和宏基因组学等重要领域
- **缺乏统一基准测试**：综述仅汇总各文献自报结果，未在统一条件下进行模型对比实验，难以做出严格公平的性能排序
- **快速过时风险**：该领域发展极快，截至 2025 年初的综述可能很快需要更新
- 未深入讨论生物序列 tokenization 策略差异（如 k-mer、BPE、单核苷酸等）对性能的影响，这是一个关键技术点

## 相关工作与启发

- **vs 此前综述**：此前综述多聚焦单一领域（如蛋白质或基因组），本文首次横跨 DNA/RNA/蛋白质/单细胞四大领域，综合性更强
- **vs 通用 NLP 综述**：生物信息学的 LLM 面临独特挑战——生物序列的 tokenization、极长序列处理（基因组可达数十亿碱基）、跨模态对齐等问题在通用 NLP 中不存在
- Evo 模型尝试统一 DNA/RNA/蛋白质的思路值得关注，可能代表了生物基础模型的未来方向

## 评分

- 新颖性: ⭐⭐⭐ 综述论文不强调方法新颖性，但跨四大领域的全面性是核心贡献
- 实验充分度: ⭐⭐⭐ 综述无自有实验，但模型覆盖面广且包含计算成本量化
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分类体系合理，Table 1 非常有价值
- 价值: ⭐⭐⭐⭐ 对于想了解 LLM 在生物信息学应用全貌的读者是很好的入门材料

<!-- RELATED:START -->

## 相关论文

- [Knowledge Boundary of Large Language Models: A Survey](knowledge_boundary_survey.md)
- [Recent Advances in Speech Language Models: A Survey](recent_advances_in_speech_language_models_a_survey.md)
- [A Survey of Large Language Models in Psychotherapy: Current Landscape and Future Directions](a_survey_of_large_language_models_in_psychotherapy_current_landscape_and_future_.md)
- [When Large Language Models Meet Speech: A Survey on Integration Approaches](when_large_language_models_meet_speech_a_survey_on_integration_approaches.md)
- [Pragmatics in the Era of Large Language Models: A Survey on Datasets, Evaluation, Opportunities and Challenges](pragmatics_survey.md)

<!-- RELATED:END -->
