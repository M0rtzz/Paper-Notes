---
title: >-
  [论文解读] Incorporating Domain Knowledge into Materials Tokenization
description: >-
  [ACL 2025][预训练][材料科学] 提出 MATTER——一种面向材料科学的领域感知分词框架，通过训练材料概念检测器 MatDetector 并将检测结果注入分词的合并排序中，避免领域术语碎片化，在生成和分类任务上分别平均提升 4% 和 2%。 领域现状 领域现状：领域现状：语言模型在材料科学中的应用日益增多（Mat…
tags:
  - "ACL 2025"
  - "预训练"
  - "材料科学"
  - "分词"
  - "领域知识"
  - "BPE/WordPiece"
  - "概念检测"
  - "MatDetector"
---

# Incorporating Domain Knowledge into Materials Tokenization

**会议**: ACL 2025  
**arXiv**: [2506.11115](https://arxiv.org/abs/2506.11115)  
**代码**: [https://github.com/yerimoh/MATTER](https://github.com/yerimoh/MATTER)  
**作者**: Yerim Oh, Jun-Hyung Park, Junho Kim, SungHo Kim, SangKeun Lee  
**机构**: Korea University, Hankuk University of Foreign Studies  
**领域**: LLM预训练  
**关键词**: 材料科学, 分词, 领域知识, BPE/WordPiece, 概念检测, MatDetector

## 一句话总结

提出 MATTER——一种面向材料科学的领域感知分词框架，通过训练材料概念检测器 MatDetector 并将检测结果注入分词的合并排序中，避免领域术语碎片化，在生成和分类任务上分别平均提升 4% 和 2%。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**：语言模型在材料科学中的应用日益增多（MatSciBERT, BatteryBERT 等），但这些模型沿用了通用 NLP 的频率驱动分词方法（BPE、WordPiece）。

**现有方法的不足**：

### 现有痛点

**现有痛点**：材料相关术语（化学式、材料名称）在语料中出现频率极低，而频率驱动分词优先保留高频词

### 核心矛盾

**核心矛盾**：低频的材料概念被拆分成语义无关的子词。例如 "germanium"（锗元素）被拆为 "german" + "ium"，完全丧失化学含义

### 解决思路

**解决思路**：之前的子词分词改进方法如 SAGE、PickyBPE 面向通用领域，未针对材料科学场景

**核心动机**：需要在分词训练阶段就引入领域知识，确保材料概念在分词时保持结构和语义完整性，而不仅仅依赖词频统计。

## 方法详解

### 整体框架

MATTER 在 WordPiece 分词算法基础上进行三个关键修改：

1. **词频计算**：保留原始词频作为基础
2. **材料知识注入**：使用 MatDetector 检测材料概念并赋予概率权重
3. **重排序合并**：根据调整后的频率重新排列 token 合并顺序，优先合并材料相关的 token 对

### 关键设计一：MatDetector（材料概念检测器）

MatDetector 是一个用于检测文本中材料概念的 NER 工具，使用 Trewartha et al. (2022) 的架构。其训练数据构建流程：

1. **材料概念提取**：从 PubChem 数据库搜索材料相关概念，提取 80K 个材料概念（化学名称、IUPAC 名称、同义词、分子式）
2. **材料语料爬取**：使用这些概念在 Semantic Scholar 上搜索，收集约 42K 篇科学论文
3. **数据标注**：用 PubChem 材料概念自动标注语料，生成 NER 数据集，标签为"material name"、"material formula"和"other"
4. **数据增强**：对格式不一致、OCR 错误等噪声进行标准化处理，将数据集扩充 4 倍

对于词 $w$，MatDetector 输出其作为材料概念的概率 $\hat{y}_{mat}(w)$：

$$\hat{y}(w) = \arg\max_{c \in C} \frac{1}{n} \sum_{i=1}^{n} P(t_i, c)$$

若预测标签属于"material"类别，则赋值 $\hat{y}_{mat}(w)$；否则为空。

### 关键设计二：频率调整与重排序

对于被 MatDetector 识别为材料概念的词，使用 log-odds 加权调整频率：

$$\text{freq}_{mat}(w) = \text{freq}_{origin}(w) + \lambda \cdot \frac{\hat{y}_{mat}(w)}{1 - \hat{y}_{mat}(w)}$$

- $\lambda$ 为材料重要性因子，控制领域知识注入的强度
- 概率越高的材料概念获得越大的频率加成，从而在合并过程中被优先保留
- 非材料概念词的频率保持不变

### 关键设计三：基于材料知识的合并排序

使用调整后的频率 $\text{freq}_{mat}$ 计算 token 对的合并得分 $\text{MatScore}(t_L, t_R)$，替代原始的纯频率得分。在迭代合并过程中：

1. 选择 MatScore 最高的 token 对进行合并
2. 创建新 token 并加入词汇表
3. 在语料中替换所有该 token 对的出现
4. 重新计算更新后 token 集的得分

## 实验关键数据

### 评估任务

**生成任务（MatSci-NLP）**：7 个子任务
- NER、关系分类（RC）、事件论元提取（EAE）、段落分类（PC）、合成动作检索（SAR）、句子分类（SC）、槽填充（SF）

**分类任务**：5 个子任务
- NER-SOFC、NER-Matscholar、SF、RC、PC

### 主要实验结果（生成任务 Macro-F1）


### 主实验

| 分词方法 | NER | RC | EAE | PC | SAR | SC | SF | 平均 |
|---------|-----|-----|-----|-----|-----|-----|-----|------|
| BPE | 47.1 | 47.2 | 36.3 | 40.2 | 41.8 | 47.6 | 16.7 | 42.0 |
| WordPiece | 56.1 | 58.5 | 29.4 | 58.9 | 74.6 | 60.3 | 32.6 | 52.9 |
| SAGE | 57.0 | 61.6 | 28.3 | 59.6 | 67.4 | 61.6 | 35.0 | 52.9 |
| PickyBPE | 41.7 | **65.1** | 36.5 | 40.2 | 66.1 | 47.6 | 23.1 | 45.8 |
| **MATTER** | **59.3** | 59.1 | **36.9** | **67.6** | **79.3** | **64.9** | **38.0** | **57.9** |

MATTER 在 7 个任务中的 5 个取得最优 Macro-F1，平均提升约 **5 个百分点**。

### 分类任务结果

MATTER 在分类任务上也取得了一致性的提升，相对于基线方法平均 Micro-F1 提升约 2%。特别是在 PickyBPE 于分类任务的 Micro-F1 上表现最强（因其清理了中间 junk token），但 MATTER 在 Macro-F1 上仍保持优势。

### 关键发现

1. **材料概念频率极低**：材料概念在 150K 篇材料相关论文中的频率远低于通用词汇，验证了频率驱动分词碎片化材料术语的直觉
2. **碎片化严重影响性能**：以 "germanium" → "german" + "ium" 为例，碎片化导致模型误解词义
3. **MATTER 有效减少碎片化**：保留了更完整的材料概念 token

## 亮点与洞察

1. **问题定义精准**：抓住了材料科学 NLP 中分词的核心痛点——低频领域术语被碎片化
2. **方法简洁有效**：仅修改分词的频率计算和合并顺序即可获得显著提升，不改变模型架构
3. **数据工程扎实**：从 PubChem 提取 80K 材料概念 → Semantic Scholar 爬取 42K 论文 → 自动标注 + 4x 增强，产出可复用资源
4. **通用框架**：MATTER 的方法论可推广到其他领域（如医学、法律），只需替换 MatDetector 为对应领域的概念检测器

## 局限与展望

1. 仅在 BERT 类模型上验证，未探索 GPT/LLaMA 等自回归模型上的效果
2. MatDetector 基于 PubChem 数据库构建，其覆盖范围受限于数据库内容
3. $\lambda$ 参数的选取需要调优，论文未给出明确的指导准则
4. 未分析分词后词汇表中材料相关 token 的具体变化（如保留了哪些完整概念）
5. 实验仅限于英文材料科学文本

## 相关工作

- **子词分词**：BPE (Sennrich et al., 2016), WordPiece (Wu et al., 2016), SAGE (Yehezkel & Pinter, 2023), PickyBPE (Chizhov et al., 2024)
- **材料科学语言模型**：MatSciBERT (Gupta et al., 2022), BatteryBERT (Huang & Cole, 2022)
- **材料概念检测**：ChemDataExtractor (Kumar et al., 2024) — 基于生物医学数据，在材料领域准确率受限

## 评分

⭐⭐⭐⭐ (4/5)

问题动机明确，方法简洁且实用，实验覆盖面广（7+5 个下游任务）。在领域专用分词这个被忽视的方向上做出了有价值的贡献。不足在于仅限 BERT 类模型且 $\lambda$ 参数缺乏理论指导。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling](dual_stage_curriculum_learning_sequence_labeling.md)
- [\[ACL 2025\] Adversarial Tokenization](adversarial_tokenization.md)
- [\[ACL 2025\] How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training](how_do_llms_acquire_new_knowledge_a_knowledge_circuits_perspective_on_continual_.md)
- [\[ACL 2025\] Retrofitting Large Language Models with Dynamic Tokenization](retrofitting_large_language_models_with_dynamic_tokenization.md)
- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](splintering_nonconcatenative_languages_for_better_tokenization.md)

</div>

<!-- RELATED:END -->
