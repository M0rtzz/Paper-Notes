---
title: >-
  [论文解读] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction
description: >-
  [NeurIPS 2025 (Workshop: Multi-modal Foundation Models for Life Sciences)][自监督学习][TabPFN] 本文提出OligoICP方法，利用TabPFN模型的预测分位数间距（IQR）作为无标签模型选择启发式指标，在siRNA敲低效率预测中实现了优于专用SOTA模型和朴素集成的性能。
tags:
  - "NeurIPS 2025 (Workshop: Multi-modal Foundation Models for Life Sciences)"
  - 自监督学习
  - TabPFN
  - siRNA功效预测
  - 不确定性引导
  - 后验集成
  - 模型选择
---

# Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction

**会议**: NeurIPS 2025 (Workshop: Multi-modal Foundation Models for Life Sciences)  
**arXiv**: [2510.02476](https://arxiv.org/abs/2510.02476)  
**代码**: 无  
**领域**: 生物信息学, 表格基础模型  
**关键词**: TabPFN, siRNA功效预测, 不确定性引导, 后验集成, 模型选择

## 一句话总结

本文提出OligoICP方法，利用TabPFN模型的预测分位数间距（IQR）作为无标签模型选择启发式指标，在siRNA敲低效率预测中实现了优于专用SOTA模型和朴素集成的性能。

## 研究背景与动机

- siRNA通过切割mRNA转录本沉默靶基因，是有前景的治疗模态，但设计高敲低效率的siRNA是关键挑战
- 生物分子功效数据集通常**小、异质、来自不同实验技术**
- 上下文学习器（如TabPFN）在小表格数据上表现出色，但性能**高度敏感于所提供的上下文**
- 简单使用更多数据不保证更好——大数据集可能超出ICL的计算限制，或与预训练分布不匹配
- **核心问题**：在无标签情况下如何选择集成中的最佳模型？

## 方法详解

### 整体框架

OligoICP方法流程：
1. 构建特征集（one-hot + trimer计数 + 热力学参数 = 574维特征）
2. 训练400个TabPFN模型集成（每个模型随机选k个训练子集，k∈[1,20]）
3. 用模型的IQR作为不确定性度量
4. 选择均值IQR最低的top 10%模型进行集成平均

### 关键设计

**特征工程**（574维）：
- siRNA 19-mer one-hot编码（4×19 = 76维）
- mRNA 57-nt one-hot编码（5×57 = 285维）
- siRNA trimer计数（64维）+ mRNA trimer计数（125维）
- 热力学参数（吉布斯自由能变化、焓变等）

**不确定性度量 — IQR**：
- TabPFN可输出预测分布的分位数估计
- IQR = 85%分位数 - 15%分位数（期望在分布内数据上70%正确率）
- 关键发现：IQR与真实预测误差呈**负相关**（高IQR → 低准确率）
- IQR在模型级聚合后，均值IQR与模型相关系数的Pearson r = -0.42

**模型选择策略**：
- 全部使用：平均400个模型预测（Full ensemble mean）
- OligoICP：仅平均IQR最低的top 10%模型（约40个）
- 基线：单一模型（所有可用数据训练）

### 数据来源

- Huesken数据集：2361个数据点，29个mRNA靶标
- Target1：295+366+9个数据点（来自3个机构的专利）
- Target2：252个数据点

## 实验关键数据

### TabPFN vs 专用SOTA（OligoFormer）

| 数据集 | TabPFN MAE↓ | OligoFormer MAE↓ | TabPFN Corr↑ | OligoFormer Corr↑ |
|--------|-----------|----------------|-------------|------------------|
| Huesken (ID) | 0.087±0.004 | 0.096 | 0.677±0.042 | 0.630 |
| Target1 (A, OOD) | 0.245 | 0.251 | 0.244 | 0.158 |
| Target1 (B, OOD) | 0.159 | 0.180 | 0.200 | 0.082 |

### 模型选择策略比较

| 数据集 | OligoICP MAE | Full ensemble MAE | All data single MAE | Oracle best MAE |
|--------|-------------|-------------------|--------------------|--------------------|
| Target1 (A) | 0.270±0.005 | 0.268±0.002 | 0.278 | 0.197 |
| Target1 (B) | 0.174±0.001 | 0.169±0.001 | 0.172 | 0.149 |
| Target2 | **0.185±0.001** | 0.189±0.001 | 0.186 | 0.161 |

| 数据集 | OligoICP Corr | Full ensemble Corr | All data single Corr | Oracle best Corr |
|--------|-------------|-------------------|--------------------|--------------------|
| Target1 (A) | **0.278±0.015** | 0.257±0.012 | 0.051 | 0.544 |
| Target1 (B) | 0.072±0.005 | 0.086±0.020 | 0.112 | 0.430 |
| Target2 | **0.246±0.015** | 0.230±0.002 | 0.230 | 0.384 |

### 关键发现

- TabPFN + 简单特征即可**超越专用的OligoFormer**，尤其在OOD场景
- IQR与预测误差之间存在可观察的负相关趋势
- OligoICP在相关系数上改进显著（Target1(A): 0.051→0.278，提升5倍+）
- 与Oracle最优模型仍有差距（0.278 vs 0.544），表明模型选择策略仍有改进空间
- 单一全量数据模型的相关系数可能极低（0.051），说明"更多数据≠更好"

## 亮点与洞察

- **通用表格模型可超越领域特定模型**——挑战了"专用模型总是更好"的假设
- IQR作为无标签模型选择指标的概念简洁且有效
- 为处理超出ICL单次前馈限制的大规模上下文数据提供了自然解决方案
- 额外计算量可接受（各模型仅处理有限数据，推理可并行化）

## 局限与展望

- MAE上改进不明显，主要改进体现在相关系数
- Target1(B)对所有方法都具有挑战性，OligoICP未能改善
- Oracle结果显示模型选择策略仍有很大改进空间
- 仅在siRNA任务上验证，需扩展到更广泛的生物分子预测任务
- 特征数量(574)超出TabPFN预训练限制，使用了"忽略预训练限制"标志
- 未与LoCalPFN等更高级的上下文选择策略比较

## 相关工作与启发

- TabPFN/TabPFNv2在小表格数据上的成功在生物医药领域的首批应用实例
- 后验集成加模型选择是Auto-sklearn等AutoML系统的经典策略，但IQR引导是新方法
- 对药物发现中的序列设计任务具有直接实用价值

## 评分

⭐⭐⭐ — 实用性强，通用模型超越专用模型的发现有价值，但方法创新有限，实验规模较小。

<!-- RELATED:START -->

## 相关论文

- [TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models](mitra_mixed_synthetic_priors_for_enhancing_tabular_foundation_models.md)
- [Robust Tabular Foundation Models](../../AAAI2026/self_supervised/robust_tabular_foundation_models.md)
- [Towards Benchmarking Foundation Models for Tabular Data With Text](../../ICML2025/self_supervised/towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection](../../ICML2025/self_supervised/foundation_model_insights_and_a_multi-model_approach_for_superior_fine-grained_o.md)

<!-- RELATED:END -->
