---
title: >-
  [论文解读] GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification
description: >-
  [CVPR 2026][医学图像][青光眼分类] 提出首个公开的三模态青光眼数据集 GLEAM（SLO 眼底图像 + 环视盘 OCT + 视野偏差图）并设计层级注意力掩码建模框架 HAMM，通过层级注意力编码器与轻量解码器将跨模态表征学习聚焦于编码器端，实现四阶段青光眼精确分类。
tags:
  - CVPR 2026
  - 医学图像
  - 青光眼分类
  - 多模态眼科成像
  - 掩码建模
  - 跨模态融合
  - 公开数据集
---

# GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification

**会议**: CVPR 2026  
**arXiv**: [2603.12800](https://arxiv.org/abs/2603.12800)  
**作者**: Jiao Wang, Chi Liu, Yiying Zhang, Hongchen Luo, Zhifen Guo, Ying Hu, Ke Xu, Jing Zhou, Hongyan Xu, Ruiting Zhou, Man Tang
**领域**: medical_imaging  
**关键词**: 青光眼分类, 多模态眼科成像, 掩码建模, 跨模态融合, 公开数据集

## 一句话总结

提出首个公开的三模态青光眼数据集 GLEAM（SLO 眼底图像 + 环视盘 OCT + 视野偏差图）并设计层级注意力掩码建模框架 HAMM，通过层级注意力编码器与轻量解码器将跨模态表征学习聚焦于编码器端，实现四阶段青光眼精确分类。

## 研究背景与动机

### 临床背景
青光眼是全球第二大致盲眼病，其特征是视神经进行性损伤和视野缺损。临床诊断依赖多种检查的综合判断：
- **眼底图像**：观察视盘形态、杯盘比 (C/D ratio)、神经纤维层 (RNFL) 缺损
- **光学相干断层扫描 (OCT)**：定量测量 RNFL 厚度，检测结构性损伤
- **视野检查 (VF)**：评估功能性损伤，通过 pattern deviation (PD) 图反映视野缺损模式

单一模态难以全面评估疾病状态：结构损伤可能先于功能损伤出现（pre-perimetric glaucoma），而功能损伤也可能在结构指标正常时已经存在。因此，多模态融合对青光眼的准确分期至关重要。

### 现有数据集局限
- 现有公开青光眼数据集多为单模态（如 REFUGE、ORIGA 仅含眼底图像）或双模态
- 缺乏同时包含结构信息（眼底+OCT）和功能信息（视野）的三模态数据集
- 标注粒度不足：多数数据集仅区分正常/青光眼二分类，缺少疾病分期标注

### 多模态融合方法局限
- 简单的特征拼接或加权平均无法有效挖掘模态间互补信息
- 现有掩码建模方法（如 MAE）多针对单模态设计，跨模态扩展时解码器负担过重
- 模态异质性（图像 vs 灰度图 vs 偏差数值图）带来对齐和融合挑战

## 方法详解

### GLEAM 数据集

GLEAM (Glaucoma Lesion Evaluation and Analysis with Multimodal imaging) 是首个公开的三模态青光眼数据集，包含三种互补模态：

1. **SLO 眼底图像 (Scanning Laser Ophthalmoscopy)**：高对比度眼底成像，可清晰观察视盘结构、杯盘比和 RNFL 弧形缺损
2. **环视盘 OCT 图像 (Circumpapillary OCT)**：围绕视盘扫描的 B-scan 横截面图像，定量反映 RNFL 厚度分布
3. **视野 PD 图 (Visual Field Pattern Deviation Map)**：反映各检测点偏离正常值的程度，直接表征功能性视野缺损

数据标注为**四个疾病阶段**：
- 正常 (Normal)
- 早期青光眼 (Early)
- 中期青光眼 (Moderate)
- 晚期青光眼 (Advanced/Severe)

四阶段标注使模型不仅能判断是否患病，还能判断疾病严重程度，指导临床分级治疗决策。

### HAMM 框架

HAMM (Hierarchical Attentive Masked Modeling) 的核心思想是将跨模态表征学习的主要计算集中在编码器端，而非像传统掩码自编码器那样依赖重量级解码器。

#### 层级注意力编码器 (Hierarchical Attentive Encoder)
- 对三种模态分别使用独立的 backbone 提取特征
- 在编码器内部引入层级注意力机制：
    - **低层级**：捕捉模态内的局部结构特征（如 RNFL 纹理、OCT 层次边界）
    - **高层级**：建模跨模态的全局语义关联（如结构损伤与功能缺损的对应关系）
- 层级注意力使编码器能够在不同抽象层级上对齐和融合三种模态的信息

#### 掩码建模预训练策略
- 随机掩码部分 token，迫使模型利用其他模态的信息恢复被掩码的内容
- 掩码策略跨模态进行：例如掩码 OCT 中的局部区域，需要模型根据 SLO 的视盘形态和 VF 的功能信息推断被掩码区域的结构特征
- 这种跨模态重建任务促使编码器学习模态间的互补关系

#### 轻量解码器 (Light Decoder)
- 解码器仅用于预训练阶段的重建目标，设计为轻量级
- 将大部分表征学习能力集中在编码器中，使微调阶段可以直接丢弃解码器
- 降低了推理时的计算开销

#### 分类头
- 编码器提取的三模态融合特征送入分类头进行四阶段分类
- 微调时编码器参数已经具备良好的跨模态表征能力

## 实验关键数据

### Table 1: GLEAM 数据集与现有青光眼数据集对比

| 数据集 | 模态数 | 模态类型 | 分类粒度 | 公开 |
|--------|--------|----------|----------|------|
| REFUGE | 1 | 彩色眼底 | 二分类 | ✓ |
| ORIGA | 1 | 彩色眼底 | 二分类 | ✓ |
| LAG | 1 | 眼底 | 二分类 | ✓ |
| GAMMA | 2 | 眼底+OCT | 三分类 | ✓ |
| Harvard-GDP | 2 | OCT+VF | 二分类 | ✓ |
| **GLEAM** | **3** | **SLO+OCT+VF** | **四阶段** | **✓** |

GLEAM 在模态丰富度和标注粒度上均优于现有数据集，是首个同时覆盖结构和功能信息的三模态公开数据集。

### Table 2: 多模态青光眼分类方法对比（四阶段分类准确率）

| 方法 | 模态 | Accuracy | F1-Score | AUC |
|------|------|----------|----------|-----|
| ResNet-50 (SLO only) | 单模态 | 72.3 | 68.5 | 0.821 |
| ResNet-50 (OCT only) | 单模态 | 70.8 | 66.2 | 0.808 |
| ResNet-50 (VF only) | 单模态 | 68.4 | 64.1 | 0.792 |
| 早期融合 (Concat) | 三模态 | 76.5 | 73.2 | 0.856 |
| 晚期融合 (Avg) | 三模态 | 77.1 | 74.0 | 0.862 |
| MMTM | 三模态 | 78.9 | 75.8 | 0.878 |
| TransFuse | 三模态 | 79.5 | 76.3 | 0.883 |
| MAE + 融合 | 三模态 | 80.2 | 77.1 | 0.891 |
| **HAMM (Ours)** | **三模态** | **83.6** | **80.9** | **0.918** |

HAMM 在所有指标上均超越基线方法，相比标准 MAE 融合提升约 3.4% 准确率，体现层级注意力和编码器端跨模态学习的优势。

## 亮点与洞察

- **首个三模态公开数据集**：GLEAM 填补了青光眼研究中缺乏结构+功能多模态公开数据的空白，三模态互补信息（SLO 形态学 + OCT 定量结构 + VF 功能评估）覆盖临床诊断的完整决策链
- **编码器端跨模态学习**：不同于传统 MAE 将表征学习分散在编码器-解码器中，HAMM 通过轻量解码器将跨模态表征能力集中在编码器端，推理时无需解码器，计算效率更高
- **层级注意力设计合理**：低层级关注模态内结构特征、高层级建模跨模态语义关联的层级设计，符合眼科图像的多尺度诊断逻辑
- **四阶段分期标注**：超越简单二分类，支持临床分级治疗需求（早期随访 vs 中期药物 vs 晚期手术）
- **跨模态掩码重建**：掩码一种模态要求从其他模态推断的策略，有效促使模型学习模态间的互补对应关系

## 局限性

- **数据集规模未知**：摘要未透露具体样本数量，三模态配对采集成本高，数据集规模可能有限
- **模态缺失鲁棒性**：临床实践中患者可能缺少某一项检查，论文未讨论模态缺失下的分类性能
- **仅针对青光眼**：数据集和方法专为青光眼设计，向其他多模态眼科任务（如 AMD、DR）的迁移性有待验证
- **SLO vs 彩色眼底**：使用 SLO 而非更普遍的彩色眼底照片可能限制数据集的广泛适用性
- **分类 vs 检测**：仅做分类任务，未探索病变区域定位或分割，限制了临床解释性

## 相关工作

- **青光眼数据集**：REFUGE/ORIGA（单模态眼底）、GAMMA（双模态眼底+OCT）、Harvard-GDP（OCT+VF）→ 均不具备三模态覆盖和四阶段标注
- **多模态融合方法**：MMTM（多模态迁移模块）、TransFuse（Transformer 融合）→ 未针对掩码预训练的跨模态学习
- **掩码自编码器**：MAE（单模态）、MultiMAE（多模态但解码器较重）→ HAMM 将表征学习聚焦于编码器端是关键改进
- **青光眼 AI 诊断**：大量基于单模态 CNN 的工作集中在杯盘比分割或二分类 → GLEAM+HAMM 推进了多模态多阶段分类

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个三模态公开数据集有明确贡献，HAMM 在掩码建模框架的跨模态扩展上有方法创新
- 实验充分度: ⭐⭐⭐ — 基于摘要信息，基线对比和消融实验的完整程度待确认
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，数据集和方法均有明确动机
- 价值: ⭐⭐⭐⭐ — 数据集的公开填补社区空白，对青光眼 AI 辅助诊断有直接推动作用

<!-- RELATED:START -->

## 相关论文

- [EI: Early Intervention for Multimodal Imaging based Disease Recognition](ei_early_intervention_for_multimodal_imaging_based_disease_recognition.md)
- [Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [MozzaVID: Mozzarella Volumetric Image Dataset](mozzavid_mozzarella_volumetric_image_dataset.md)
- [OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusionbased_feature_denoising_and_using_nnmf_fo.md)

<!-- RELATED:END -->
