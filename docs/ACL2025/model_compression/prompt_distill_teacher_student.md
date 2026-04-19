---
title: >-
  [论文解读] Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation
description: >-
  [ACL 2025][模型压缩][LLM数据标注] 提出CanDist框架，借鉴人类面对不确定性时的"模糊规避"心理，引导LLM输出多个候选标签而非单一标签(候选标注)，再通过分布精炼(Distribution Refinery)策略蒸馏到小语言模型(SLM)获得最终标注，从理论到实验证明候选标注蒸馏优于单一标注。
tags:
  - ACL 2025
  - 模型压缩
  - LLM数据标注
  - 候选标注
  - 知识蒸馏
  - teacher-student
  - 文本分类
---

# Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation

**会议**: ACL 2025  
**arXiv**: [2506.03857](https://arxiv.org/abs/2506.03857)  
**代码**: [有](https://github.com/MingxuanXia/CanDist)  
**领域**: 模型蒸馏 / 数据标注  
**关键词**: LLM数据标注, 候选标注, 知识蒸馏, teacher-student, 文本分类

## 一句话总结

提出CanDist框架，借鉴人类面对不确定性时的"模糊规避"心理，引导LLM输出多个候选标签而非单一标签(候选标注)，再通过分布精炼(Distribution Refinery)策略蒸馏到小语言模型(SLM)获得最终标注，从理论到实验证明候选标注蒸馏优于单一标注。

## 研究背景与动机

**领域现状**: LLM驱动的自动数据标注已广泛应用于文本分类、NER、情感分析等NLP任务，显著降低人工标注成本。

**现有痛点**: 现有方法采用"激进策略"——强迫LLM为每个样本输出单一确定标签。当LLM对困难样本不确定时，这种策略往往产生完全错误的标注，不仅浪费计算资源，还严重损害下游任务的数据质量。

**核心矛盾**: LLM对下游任务的知识有限，面对不确定样本时被迫"过度自信"地给出单一答案，导致错误标注的概率大增。

**本文目标**: 当LLM不确定时，能否让它给出更有价值的输出(而非完全错误的标签)？

**切入角度**: 受人类行为中"模糊规避"(Ambiguity Aversion)心理启发——面对不确定性时，人类倾向保守行事而非过度自信。将此思想注入LLM标注过程，允许LLM输出多个可能标签(候选标注)，然后用SLM蒸馏出正确标签。

**核心 idea**: 让不确定的LLM给出候选集而非单一标签，再用SLM从候选集中蒸馏出正确答案——保守策略优于过度自信。

## 方法详解

### 整体框架

CanDist分为两阶段：(1) **候选标注(Prompt Candidates)**——用两种提示策略(CA_add: 先给一个答案再追加可能答案; CA_all: 直接给出所有可能答案)引导LLM输出候选标签集；(2) **蒸馏标注(Distill)**——训练SLM在LLM的候选标签约束下，通过分布精炼逐步识别正确标签。

### 关键设计

1. **候选标注提示策略(CA_add / CA_all)**
    - 功能：引导LLM在不确定时输出多个可能标签
    - 核心思路：CA_add在标准提示后添加"如果不确定，请包含其他可能选项"；CA_all直接要求输出"所有可能的类型"
    - 设计动机：CA_all在1-α-error指标上比单一标注(SA)提升18-27%，大幅提高正确标签的覆盖率

2. **分布精炼(Distribution Refinery, DR)**
    - 功能：从候选标签中动态识别真实标签
    - 核心思路：利用DNN的记忆效应——SLM先学会简单模式，使部分真实标签从假阳性标签中浮现。训练目标分布初始化为候选标签上的均匀分布，后续迭代中用SLM预测的softmax输出在候选集内重归一化来更新目标分布
    - 设计动机：直接在候选标签均匀分布上训练是次优的，需要动态精炼来凸显正确标签

3. **样本过滤与分布锐化**
    - 功能：处理候选集不包含正确标签的边缘情况，并加速收敛
    - 核心思路：过滤"out-of-candidate"样本(SLM最大预测落在候选集外)；对可靠样本(类别内小loss样本)用温度参数锐化分布；对高置信度的out-of-candidate样本用其预测标签训练
    - 设计动机：少量样本的正确标签不在候选集中，会干扰蒸馏过程

### 损失函数 / 训练策略

整体训练目标为交叉熵损失 $\mathcal{L}_{dr} = \frac{1}{n}\sum_{i=1}^{n} l_{ce}(\boldsymbol{p}_i, \hat{\boldsymbol{q}}_i)$，其中目标分布 $\hat{\boldsymbol{q}}$ 根据样本类别动态调整：可靠样本用温度 $\gamma$ 锐化、普通in-candidate样本用标准DR分布、高置信候选外样本用预测类别作one-hot目标。使用RoBERTa-Base作为SLM，GPT-3.5作为Teacher LLM。

## 实验关键数据

### 主实验

| 方法 | TREC | MA | DBP | AGN | RCT | BANK |
|------|------|-----|-----|-----|-----|------|
| Zero-shot | 72.20 | 63.12 | 93.94 | 87.24 | 61.83 | 68.41 |
| Few-shot | 77.20 | 63.40 | 95.40 | 88.05 | 65.85 | 68.86 |
| FreeAL | 82.33 | 64.13 | 97.92 | 88.64 | 68.32 | 74.58 |
| **CanDist_add** | **83.13** | **64.23** | **98.72** | **89.46** | **69.77** | **76.27** |
| CanDist_all | 87.80 | 64.20 | 98.65 | 88.78 | 70.57 | 75.97 |
| SFT(有标注) | 97.80 | 64.54 | 98.78 | 92.29 | 84.52 | 93.31 |

### 消融实验

| 消融设置 | 平均训练集准确率 |
|---------|---------------|
| CanDist_add | 79.16 |
| CanDist_add + LLM Select | 75.42 (-3.74) |
| CanDist_all | 78.86 |
| CanDist_all + LLM Select | 74.96 (-3.90) |
| Few-shot (SA) | 74.79 |

### 关键发现

- 候选标注(CA_all)相比单一标注(SA)在1-α-error上提升14-27%，F1分数也一致更高
- CanDist在6个数据集上全面超越所有LLM和SLM基线，包括FreeAL
- 用SLM蒸馏候选标注效果远优于让LLM自己从候选中二次选择(平均差3.74-3.90%)——说明SLM的蒸馏能力优于LLM的二次判断
- 理论证明(Theorem 1)：从top-2候选蒸馏比从top-1单标注蒸馏具有更宽松的100%准确率达成条件

## 亮点与洞察

- **类比精妙**：将人类"模糊规避"心理应用于LLM标注，idea自然且有说服力
- **理论保证**：严格证明候选标注蒸馏比单一标注蒸馏具有更好的噪声容忍上界
- 分布精炼策略巧妙利用DNN记忆效应——先记住简单样本→从候选中浮现真实标签→迭代精炼
- 方法极其轻量且通用——只需修改prompt就能获得候选标注，不需要重新训练LLM
- 揭示了一个反直觉结论：SLM蒸馏候选集 > LLM直接从候选集中再选择

## 局限与展望

- 仅在文本分类任务上验证，序列标注、生成类任务的适用性未知
- 候选标注策略依赖LLM能正确理解"请输出所有可能标签"的指令，对指令跟随能力差的模型可能失效
- CA_all可能导致候选集过大(接近全标签空间)，在类别数极多的场景下效果可能退化
- 分布精炼的超参(温度γ、高置信阈值τ、小loss比例δ)需要调节
- 未探索非LLM场景(如人类标注团队)的候选标注范式

## 相关工作与启发

- 与Self-Consistency(SC)的区别：SC利用LLM的随机性(temperature采样)，CanDist提示LLM输出内在不确定性
- 与FreeAL的关系：FreeAL也是SLM协作标注的先驱，但仅蒸馏单一标注；CanDist证明蒸馏候选标注严格更优
- 启发：面对LLM不确定性，让它坦率承认不确定(输出候选集)比强迫它给出一个答案更有价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ (候选标注范式+蒸馏的组合idea新颖，理论分析加分)
- **实验充分度**: ⭐⭐⭐⭐ (6个数据集、多基线对比、消融充分，但任务类型单一)
- **写作质量**: ⭐⭐⭐⭐⭐ (动机阐述清晰，图表直观，理论与实验结合好)
- **价值**: ⭐⭐⭐⭐ (提供了LLM数据标注的新范式，对NLP从业者有实用价值)

<!-- RELATED:START -->

## 相关论文

- [Adversarially Robust Distillation by Reducing the Student-Teacher Variance Gap](../../ECCV2024/model_compression/adversarially_robust_distillation_by_reducing_the_student-teacher_variance_gap.md)
- [STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](stun_moe_pruning.md)
- [Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation](data_laundering_artificially_boosting_benchmark_results_through_knowledge_distil.md)
- [500xCompressor: Generalized Prompt Compression for Large Language Models](500xcompressor_generalized_prompt_compression_for_large_language_models.md)
- [OuroMamba: A Data-Free Quantization Framework for Vision Mamba](../../ICCV2025/model_compression/ouromamba_a_data-free_quantization_framework_for_vision_mamba.md)

<!-- RELATED:END -->
