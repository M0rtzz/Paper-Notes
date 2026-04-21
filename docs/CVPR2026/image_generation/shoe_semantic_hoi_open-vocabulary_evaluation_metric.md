---
title: >-
  [论文解读] SHOE: Semantic HOI Open-Vocabulary Evaluation Metric
description: >-
  [CVPR 2026][图像生成][开放词汇HOI检测] 提出SHOE评估框架，通过将HOI预测分解为动词和物体分别计算LLM驱动的语义相似度，替代传统mAP的精确匹配方式，在开放词汇HOI检测评估中达到85.73%的人类判断一致性，超过人类标注者之间78.61%的平均一致性。
tags:
  - CVPR 2026
  - 图像生成
  - 开放词汇HOI检测
  - 语义相似度评估
  - LLM评分
  - WordNet
  - 评估指标
---

# SHOE: Semantic HOI Open-Vocabulary Evaluation Metric

**会议**: CVPR 2026  
**arXiv**: [2604.01586](https://arxiv.org/abs/2604.01586)  
**代码**: https://github.com/majnoa/SHOE  
**领域**: 人体-物体交互检测 / 评估指标  
**关键词**: 开放词汇HOI检测, 语义相似度评估, LLM评分, WordNet, 评估指标

## 一句话总结

提出SHOE评估框架，通过将HOI预测分解为动词和物体分别计算LLM驱动的语义相似度，替代传统mAP的精确匹配方式，在开放词汇HOI检测评估中达到85.73%的人类判断一致性，超过人类标注者之间78.61%的平均一致性。

## 研究背景与动机

1. **领域现状**：人体-物体交互（HOI）检测是视觉理解的基础任务，标准评估指标为mAP，依赖于预测与标签的精确分类匹配。
2. **现有痛点**：mAP将HOI类别视为离散标签，语义相近但词汇不同的预测（如"lean on couch"和"sit on couch"）会被判为错误；同时数据集标注不完整，合理但未标注的预测被惩罚为假阳。
3. **核心矛盾**：随着VLM和MLLM的崛起，模型能生成超越固定标签集的开放词汇预测，但现有评估协议无法公正衡量这些灵活输出的质量。
4. **本文目标**：设计一个语义感知的柔性评估框架，支持开放词汇HOI预测的分级匹配评估。
5. **切入角度**：将HOI分解为动词和物体两个独立组件，分别用多个LLM的平均评分计算语义相似度，避免全HOI对组合爆炸。
6. **核心 idea**：通过WordNet消歧 + 多LLM语义评分实现HOI分解式柔性匹配评估。

## 方法详解

### 整体框架

输入为预测的HOI三元组$(b_h, b_o, v, o)$和GT HOI，经过边界框匹配后，将动词和物体分别映射到WordNet同义词集（synset），查询预计算的LLM相似度表，合成实例级相似度分数，最终聚合得到Soft-mAP或mF1分数。

### 关键设计

1. **WordNet Synset映射与消歧**:

    - 功能：将HOI的动词和物体标签映射到语义明确的WordNet同义词集
    - 核心思路：每个动词/物体对应一个sense-specific synset，消除一词多义的歧义。对于物体，利用WordNet层级结构的邻域扩展（上位词、下位词）；对于动词，由于WordNet动词分类较浅且碎片化，手动整理约7,150个HOI相关动词synset进行匹配
    - 设计动机：直接比较原始词汇会受到词汇多义性干扰，用synset确保语义比较反映真实含义

2. **多LLM语义相似度评分**:

    - 功能：为每对动词-动词和物体-物体计算0-4分的语义相似度
    - 核心思路：先用Qwen3-32B进行全量初筛（约850K动词对比较），筛掉零相似的对；再用DeepSeek-V3、Llama-4-Maverick-17B、Yi-1.5-34B-Chat、Gemini-2.5-Pro四个LLM对非零对进一步评分，取平均值。LLM根据synset的gloss定义在5分制上打分
    - 设计动机：单一LLM评分有偏差，多模型平均提高鲁棒性；动词相似度的模型间Pearson相关较低(0.50-0.72)而物体较高(最高r=0.84)，说明动词语义确实更复杂

3. **分解式可扩展评估设计**:

    - 功能：将HOI相似度分解为$\text{sim}(p,g) = f(\text{sim}_v(v^p, v^g), \text{sim}_o(o^p, o^g))$
    - 核心思路：采用算术平均$w=0.5$聚合动词和物体相似度。这种分解使得相似度表只需计算$V^2 + O^2$次，而非暴力枚举的$(V \times O)^2$次。支持将HICO-DET的600个HOI类扩展到3800万个语义相关HOI
    - 设计动机：暴力计算每对HOI的相似度随词汇量二次增长不可行，分解策略使大规模开放词汇评估在计算上可行

### 损失函数 / 训练策略

SHOE本身不涉及训练，而是一个评估指标框架。它提供两种聚合模式：
- **有置信度模式**：兼容mAP式排序评估，计算Soft-AP和Soft-mAP
- **无置信度模式**：直接对所有预测平等计算soft precision/recall/F1，适用于VLM等无原生置信度的模型

## 实验关键数据

### 主实验

| 方法 | 类型 | mAP | SHOE mAP |
|------|------|-----|----------|
| HOLA (ViT-L) | Default | 39.05 | 39.92 |
| LAIN (ViT-B) | Zero-shot | 34.60 | 35.37 |
| THID | Open-Vocab | 22.01 | 22.04 |
| GPT-4.1 + DETR | VLM | 49.50 | 61.67 |
| InternVL3-38B + DETR | VLM | 42.00 | 58.03 |
| Qwen2.5-VL-32B + DETR | VLM | 34.83 | **66.03** |

### 消融实验

| 评估指标 | 与人类判断一致性(%) |
|----------|---------------------|
| SHOE (Standard, 算术平均) | **85.73** |
| SHOE (几何平均) | 84.29 |
| SHOE (最小值) | 84.01 |
| DeepSeek-V3 (直接LLM评分) | 83.34 |
| Gemini-2.5-Pro | 77.52 |
| CLIP-ViT-B (gloss) | 59.11 |
| WordNet WUP | 57.09 |
| SentenceBERT | 54.09 |
| mAP direct-match | 38.90 |

### 关键发现

- Qwen2.5-VL-32B标准mAP最低(34.83)但SHOE mAP最高(66.03)，说明该模型有很强的语义理解但不完全复现HICO-DET的精确标签
- VLM类方法在SHOE mAP下显著优于传统方法，揭示了mAP无法捕捉的真实能力差异
- 超参数调优显示"同动词不同物体"场景下最优权重$w^*=0.267$偏向物体相似度，"不同动词同物体"下$w^*=0.733$偏向动词，但因用户研究规模有限仍用$w=0.5$
- 用Qwen3-32B筛掉的零相似动词对，其他LLM不同意率仅0.245%~1.318%，验证了筛选策略的可靠性

## 亮点与洞察

- **分解思路极其优雅**：将HOI相似度拆为动词和物体独立比较，计算复杂度从$(V \times O)^2$降到$V^2 + O^2$，使HICO-DET的600类扩展到3800万类成为可能。这个思路可以推广到任何需要组合语义比较的评估场景
- **超越人类一致性**：SHOE达到85.73%与平均人类评分的一致性，而人类标注者之间平均一致性仅78.61%。这说明多LLM平均确实能产生比单个人类更稳定的语义判断
- **评估指标即基础设施**：相似度查找表只需构建一次，后续评估直接查表，极大降低了重复使用成本

## 局限与展望

- 目前仅在HICO-DET上验证，其他HOI数据集（如SWIG-HOI）也存在标注不完整问题，需要扩展验证
- 用户研究规模偏小（500对，5位标注者），在更大规模人类评估中的稳定性需要进一步验证
- 对VLM的置信度代理（token概率）可能不可靠，如何更好地为开放式生成模型获取校准的置信度仍是开放问题
- 语义相似度的"黄金标准"本身因人而异，特定领域（如医疗、法律场景）的HOI评估可能需要领域定制

## 相关工作与启发

- **vs mAP (标准评估)**: mAP执行严格精确匹配，SHOE引入语义梯度匹配，两者互补——mAP衡量精确再现能力，SHOE衡量语义理解能力
- **vs CLIP-based相似度**: CLIP在HOI对比较中仅59.11%一致性，说明通用视觉-语言嵌入不足以捕捉HOI语义的细微差异
- **vs 直接LLM评分**: 直接用LLM评整个HOI对最高达83.34%，但SHOE分解策略达85.73%且更可扩展

## 评分

- 新颖性: ⭐⭐⭐⭐ 分解式语义评估思路新颖，但核心仍是用LLM评分+平均
- 实验充分度: ⭐⭐⭐⭐ 用户研究、多基线对比、Qwen筛选验证等都比较完备
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，图表专业，公式表达完整
- 价值: ⭐⭐⭐⭐ 为开放词汇HOI评估提供了实用工具，但影响范围限于HOI社区

<!-- RELATED:START -->

## 相关论文

- [OpenDPR: Open-Vocabulary Change Detection via Vision-Centric Diffusion-Guided Prototype Retrieval for Remote Sensing Imagery](opendpr_open-vocabulary_change_detection_via_vision-centric_diffusion-guided_pro.md)
- [EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories](emma_concept_erasure_benchmark_with_comprehensive_semantic_metrics_and_diverse_c.md)
- [PolyGraph Discrepancy: a classifier-based metric for graph generation](../../ICLR2026/image_generation/polygraph_discrepancy_a_classifier-based_metric_for_graph_generation.md)
- [Multi-Metric Preference Alignment for Generative Speech Restoration](../../AAAI2026/image_generation/multi-metric_preference_alignment_for_generative_speech_restoration.md)
- [Conjuring Semantic Similarity](../../ICLR2026/image_generation/conjuring_semantic_similarity.md)

<!-- RELATED:END -->
