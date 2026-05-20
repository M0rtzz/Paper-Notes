---
title: >-
  [论文解读] Faithful Summarization of Consumer Health Queries: A Cross-Lingual Framework with LLMs
description: >-
  [NeurIPS 2025][医学图像][医学文本摘要] 提出结合 TextRank 抽取式句子选择和医学命名实体识别 (NER) 来引导 LLM 生成忠实医学摘要的框架，在英文 MeQSum 和孟加拉语 BanglaCHQ-Summ 数据集上通过微调 LLaMA-2-7B 实现质量和忠实性的一致提升…
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "医学文本摘要"
  - "忠实性"
  - "跨语言"
  - "TextRank"
  - "命名实体识别"
  - "LLaMA"
---

# Faithful Summarization of Consumer Health Queries: A Cross-Lingual Framework with LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2511.10768](https://arxiv.org/abs/2511.10768)  
**代码**: 未公开  
**领域**: 医学图像  
**关键词**: 医学文本摘要, 忠实性, 跨语言, TextRank, 命名实体识别, LLaMA

## 一句话总结

提出结合 TextRank 抽取式句子选择和医学命名实体识别 (NER) 来引导 LLM 生成忠实医学摘要的框架，在英文 MeQSum 和孟加拉语 BanglaCHQ-Summ 数据集上通过微调 LLaMA-2-7B 实现质量和忠实性的一致提升，SummaC 达 0.57，人工评估 82% 摘要保留关键医学信息。

## 研究背景与动机

**在线健康咨询增长**：疫情推动的在线问诊平台成为重要医疗信息来源，但冗长且重复的患者提问给医护人员造成负担——需在回答前先识别核心关注点

**传统指标不足**：现有摘要评估主要依赖 ROUGE/BERTScore 等质量指标，衡量词汇/语义相似度，但**忽视忠实性**——即摘要与源文的事实一致性

**医学场景的特殊风险**：抽象式模型常产生内在错误（错误表示实体/关系）和外在错误（引入无支持事实），在医学领域即使微小歪曲也可能误导患者/医生，危及健康结果

**忠实性被低估**：相比可读性和一般准确性，忠实性在医学摘要中研究不足，缺乏专门提升忠实性的框架

**跨语言空白**：医学摘要忠实性的跨语言评估几乎为零，低资源语言（如孟加拉语）的医学 NLP 严重匮乏

## 方法详解

### 整体框架

三阶段管线：**医学 NER + TextRank 抽取 → LLM 微调 → Best-of-N 选择**

### Step 1: 预处理与相关句子抽取

1. 标准化数据集为 question-summary 格式
2. 识别重叠医学实体和否定词，确保关键信息被保留
3. 应用 TextRank 算法提取包含医学实体和查询相关词的句子
4. **核心目的**：在 LLM 抽象式生成前，保证输入已锚定在医学重要内容上

TextRank 基于图的排序算法，将句子作为节点、句子间相似度作为边权，迭代计算句子重要性得分：

$$S(V_i) = (1 - d) + d \cdot \sum_{V_j \in \text{In}(V_i)} \frac{w_{ji}}{\sum_{V_k \in \text{Out}(V_j)} w_{jk}} S(V_j)$$

其中 $d$ 为阻尼因子，$w_{ji}$ 为句子 $j$ 到 $i$ 的边权。

### Step 2: LLM 微调

- **基座模型**：LLaMA-2-7B
- **训练方式**：LoRA (Low-Rank Adaptation) 参数高效微调
- **输入**：TextRank 筛选的含医学实体的句子
- **输出**：简洁忠实的摘要

### Step 3: Best-of-N 选择

生成多个候选摘要（温度 $t = 0.7$），按两种策略选最优：
- **ROUGE-1 选择**：最大化词汇覆盖（质量导向）
- **SummaC 选择**：最大化事实一致性（忠实性导向）

温度扫描 $t \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ 发现权衡：低温利于 ROUGE，高温利于 SummaC，$t = 0.7$ 为平衡点。

### 评估指标体系

- **质量指标**：ROUGE-1/2/L, BERTScore
- **忠实性指标**：SummaC（NLI-based 事实一致性）, AlignScore（语义对齐）
- **可读性**：Flesch Reading Ease (FRE)

## 实验关键数据

### MeQSum（英文，1000 样本）

| 设置 | R1 | R2 | RL | BERT | 可读性 | SummaC | AlignScore |
|------|-----|-----|-----|------|--------|--------|------------|
| Zero-shot | 21.97 | 6.48 | 19.98 | 0.60 | 65.16 | 0.28 | 21.80 |
| 微调 (无 TR) | 44.23 | 27.36 | 41.55 | 0.71 | 70.21 | 0.31 | 38.45 |
| 微调 + TR | 47.07 | 29.44 | 44.08 | 0.72 | 70.69 | 0.37 | 45.65 |
| Best-of-3 (R1) | **50.50** | **34.38** | **47.74** | **0.74** | 71.56 | 0.40 | 39.24 |
| Best-of-3 (SummaC) | 48.27 | 31.38 | 45.34 | 0.73 | 71.56 | **0.57** | **45.91** |

### 与 SOTA 对比

| 模型 | R1 | R2 | RL | BERT | SummaC | AlignScore |
|------|-----|-----|-----|------|--------|------------|
| Mixtral-8x7B-Inst. | 32.47 | 36.38 | 16.86 | 0.72 | - | - |
| BioBART + FaMeSumm | 31.76 | 11.71 | 29.64 | 0.74 | 0.46 | - |
| **Ours (Best-of-3)** | **50.50** | **34.38** | **47.74** | **0.74** | **0.57** | **0.46** |

### BanglaCHQ-Summ（孟加拉语，2350 样本）

| 设置 | R1 | R2 | RL | BERT | SummaC |
|------|-----|-----|-----|------|--------|
| Zero-shot | 19.10 | 8.21 | 18.97 | 0.62 | 0.22 |
| 微调 (无 TR) | 28.24 | 14.22 | 24.54 | 0.71 | 0.26 |
| 微调 + TR | 30.71 | 15.71 | 28.95 | 0.74 | 0.28 |
| Best-of-3 (R1) | **32.35** | **16.32** | **29.09** | **0.76** | 0.29 |
| Best-of-3 (SummaC) | 30.92 | 15.74 | 27.35 | 0.73 | **0.32** |

### 关键发现

1. **TextRank 持续有效**：加入 TextRank 在英/孟两种语言上都一致提升质量和忠实性
2. **Best-of-N 策略显著**：从 3 个候选中选最优比单次生成大幅提升——R1 选择优化质量，SummaC 选择优化忠实性
3. **知识迁移成功**：框架从英文到孟加拉语保持有效，验证跨语言泛化能力
4. **人工评估验证**：医学博士评估 82% 的摘要同时保留所有关键信息且事实一致
5. **温度权衡**：存在质量-忠实性权衡——$t = 0.7$ 为最佳平衡点

## 亮点

- ⭐⭐⭐⭐ **忠实性聚焦**：首次将忠实性作为医学文本摘要的核心优化目标，而非仅追求 ROUGE
- ⭐⭐⭐ **跨语言验证**：在英文和孟加拉语（低资源语言）上均验证有效，是首个跨语言医学摘要忠实性评估
- ⭐⭐⭐ **实用管线**：TextRank + NER + LoRA 微调 + Best-of-N 的管线简洁实用，易于复现
- ⭐⭐⭐ **人工验证**：82% 的人工评估通过率提供了超越自动指标的可信度
- ⭐⭐ **清晰消融**：每个组件的增量贡献通过消融实验清晰展示

## 局限与展望

1. **单一 LLM**：仅在 LLaMA-2-7B 上验证，未测试更新/更大的模型（如 LLaMA-3、Mistral）
2. **数据集规模小**：MeQSum 仅 1000 样本，BanglaCHQ-Summ 2350 样本，统计显著性受限
3. **两种语言**：仅覆盖英文和孟加拉语，其他低资源语言的泛化性未知
4. **TextRank 局限**：基于词频的图排序可能在复杂医学叙述中遗漏语义重要但词频低的信息
5. **评估维度有限**：SummaC/AlignScore 本身也是近似指标，可能无法捕获所有类型的事实不一致

## 总体评价 ⭐⭐⭐

方向正确——将忠实性作为医学摘要的核心关注是重要贡献。方法设计朴实但有效（TextRank + NER 引导 + Best-of-N）。跨语言评估是亮点。但整体规模偏小（单一 LLM、小数据集、两种语言），技术贡献的深度有限——更像是一个系统性的工程整合而非方法创新。作为 workshop paper（Muslims in ML @ NeurIPS）定位合理。

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings](../../ACL2025/medical_imaging/evaluation_of_llms_in_medical_text_summarization_the_role_of_vocabulary_adaptati.md)
- [\[NeurIPS 2025\] UniSite: The First Cross-Structure Dataset and Learning Framework for End-to-End Ligand Binding Site Detection](unisite_the_first_cross-structure_dataset_and_learning_framework_for_end-to-end_.md)
- [\[ACL 2025\] Are LLMs Effective Psychological Assessors? Leveraging Adaptive RAG for Interpretable Mental Health Screening through Psychometric Practice](../../ACL2025/medical_imaging/are_llms_effective_psychological_assessors_leveraging_adaptive_rag_for_interpret.md)
- [\[NeurIPS 2025\] The Biased Oracle: Assessing LLMs' Understandability and Empathy in Medical Diagnoses](the_biased_oracle_assessing_llms_understandability_and_empathy_in_medical_diagno.md)
- [\[NeurIPS 2025\] EEGReXferNet: A Lightweight Gen-AI Framework for EEG Subspace Reconstruction via Cross-Subject Transfer Learning and Channel-Aware Embedding](eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)

</div>

<!-- RELATED:END -->
