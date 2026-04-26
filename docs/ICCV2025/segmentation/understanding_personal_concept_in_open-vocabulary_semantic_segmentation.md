---
title: >-
  [论文解读] Personalized OVSS: Understanding Personal Concept in Open-Vocabulary Semantic Segmentation
description: >-
  [ICCV 2025][图像分割][个性化分割] 首次提出个性化开放词汇语义分割（Personalized OVSS）任务，设计基于文本提示调优的即插即用方法，通过"负掩码提案"抑制假阳性和视觉嵌入注入丰富个性化概念表征，仅用少量图像-掩码对即可识别用户感兴趣的特定物体实例，同时保持原有OVSS性能。
tags:
  - ICCV 2025
  - 图像分割
  - 个性化分割
  - 开放词汇语义分割
  - 文本提示调优
  - 负掩码提案
  - 少样本学习
---

# Personalized OVSS: Understanding Personal Concept in Open-Vocabulary Semantic Segmentation

**会议**: ICCV 2025  
**arXiv**: [2507.11030](https://arxiv.org/abs/2507.11030)  
**代码**: 无  
**领域**: 图像分割  
**关键词**: 个性化分割, 开放词汇语义分割, 文本提示调优, 负掩码提案, 少样本学习

## 一句话总结

首次提出个性化开放词汇语义分割（Personalized OVSS）任务，设计基于文本提示调优的即插即用方法，通过"负掩码提案"抑制假阳性和视觉嵌入注入丰富个性化概念表征，仅用少量图像-掩码对即可识别用户感兴趣的特定物体实例，同时保持原有OVSS性能。

## 研究背景与动机

开放词汇语义分割（OVSS）可以用任意文本描述分割图像，但无法理解个人化概念——例如识别"我的杯子"需要区分它与其他杯子的不同。这在实际应用中极为关键：

**OVSS的局限**：现有OVSS模型设计用于区分不同类别（如杯子 vs 瓶子），而非区分同类别内的特定实例（"我的杯子"vs "其他杯子"）

**少样本分割的不足**：(a)不支持开放词汇分割——只能分割给定的目标类别；(b)不考虑在同类别物体中区分特定实例

**个性化需求**：机器人助手场景中，用户希望说"拿我的杯子"即可，无需每次提供详细描述

**文本提示调优的假阳性问题**：直接进行text prompt tuning虽能识别目标概念，但会把其他相似物体也误识别为个性化概念（如把所有鸟都识别为"我的鸟"）

**核心动机**：需要一种即插即用方法，仅用少量图像-掩码对学习个人视觉概念，同时保持对其他类别的正常分割能力。

## 方法详解

### 整体框架

在现成OVSS模型（如SAN、ODISE）基础上，新增三个轻量化组件：
1. 可学习文本嵌入 $\textbf{T}_{\text{per}}$（学习个性化概念）
2. 负掩码提案（抑制假阳性）
3. 视觉嵌入注入（丰富个性化表征）

### 关键设计

1. **文本提示调优（Text Prompt Tuning）**：

    - 初始化一个可学习文本嵌入 $\textbf{T}_{\text{per}} \in \mathbb{R}^{1 \times D}$，用目标类别名的文本嵌入初始化（如"a photo of black footed albatross"）
    - 拼接到原有词汇的文本嵌入 $\textbf{T}_{\text{open}}$ 后：$\textbf{T} = [\textbf{T}_{\text{open}}; \textbf{T}_{\text{per}}]$
    - 使用标准分割损失训练：$\mathcal{L}_{seg} = \lambda_1\mathcal{L}_{dice} + \lambda_2\mathcal{L}_{bce} + \lambda_3\mathcal{L}_{cls}$
    - **关键发现**：单纯文本提示调优提高了recall（能识别目标），但precision大幅下降（把其他类似物体也误识别为目标），即假阳性严重
    - 设计动机：文本提示调优是最直接的个性化方法，但需要配合负掩码来控制假阳性

2. **负掩码提案（Negative Mask Proposal）**：

    - **负掩码嵌入**：通过对原有掩码嵌入的可学习线性组合得到 $\textbf{Z}_{\text{neg}} = \textbf{W}_{\text{Z}} \textbf{Z}_{\text{open}}$，$\textbf{W}_{\text{Z}} \in \mathbb{R}^{1 \times N}$
    - **负掩码**：通过可学习卷积层 $\textbf{W}_{\text{M}}$ 从原有掩码提案生成 $\textbf{M}_{\text{neg}} = \textbf{W}_{\text{M}} \textbf{M}_{\text{open}}$
    - **监督信号**：
        - 负掩码嵌入学习均匀匹配除个性化概念外的所有词汇：$\mathcal{L}^{\text{neg}}_{\text{Z}} = -\sum_{i \neq k} \frac{1}{V-1}\log S[i,j]$
        - 负掩码以 $1 - \textbf{M}_{\text{gt}}$ 为GT进行BCE监督：$\mathcal{L}^{\text{neg}}_{\text{M}} = -(1-\textbf{M}_{\text{gt}})\log(\textbf{M}_{\text{neg}}) - \textbf{M}_{\text{gt}}\log(1-\textbf{M}_{\text{neg}})$
    - 拼接后：$\textbf{Z} = [\textbf{Z}_{\text{open}}; \textbf{Z}_{\text{neg}}]$, $\textbf{M} = [\textbf{M}_{\text{open}}; \textbf{M}_{\text{neg}}]$
    - 设计动机：显式学习"非目标区域"的表征，给模型提供反例信号；相比Yo'LLaVA需要收集大量负样本，本方法从已有掩码提案中自动生成负掩码，无需额外数据

3. **视觉嵌入注入（Injection of Visual Embeddings）**：

    - 用CLIP图像编码器提取特征图 $\textbf{F} = \textbf{I}_{\text{enc}}(\textbf{X})$
    - 用掩码提取目标区域特征并平均池化：$\textbf{F}_{\text{per}} = \frac{1}{\sum \mathbb{1}(\textbf{M}'_{\text{gt}}=1)} \sum \textbf{F} \odot \textbf{M}'_{\text{gt}}$
    - 多张图像的 $\textbf{F}_{\text{per}}$ 取平均
    - 与文本嵌入插值融合：$\textbf{T}_{\text{per}}^{vis} = \alpha \cdot \textbf{F}_{\text{per}} + (1-\alpha) \cdot \textbf{T}_{\text{per}}$
    - 设计动机：单模态（纯文本或纯视觉）的提示调优表征能力有限；文本-视觉联合能更好地编码个性化概念的细粒度外观特征

### 损失函数 / 训练策略

- 总损失：$\mathcal{L}_{total} = \mathcal{L}_{seg} + \lambda^{\text{neg}}_{\text{Z}}\mathcal{L}^{\text{neg}}_{\text{Z}} + \lambda^{\text{neg}}_{\text{M}}\mathcal{L}^{\text{neg}}_{\text{M}}$
- 仅训练 $\textbf{T}_{\text{per}}$, $\textbf{W}_{\text{M}}$, $\textbf{W}_{\text{Z}}$ 三组参数，OVSS模型完全冻结
- 仅需200次迭代即可完成个性化训练
- 支持K=1,3,5张参考图像

## 实验关键数据

### 主实验

| 数据集 | 方法 | IoU$^{\text{per}}$ (K=5) | mIoU (K=5) | IoU$^{\text{per}}$提升 |
|---|---|---|---|---|
| FSS$^{\text{per}}$ | SAN | 41.08 | 55.68 | baseline |
| FSS$^{\text{per}}$ | SAN + Ours | **56.80** | 55.85 | **+15.72** |
| CUB$^{\text{per}}$ | SAN | 68.25 | 77.32 | baseline |
| CUB$^{\text{per}}$ | SAN + Ours | **76.80** | 78.29 | **+8.55** |
| ADE$^{\text{per}}$ | SAN | 6.88 | 17.20 | baseline |
| ADE$^{\text{per}}$ | SAN + Ours | **26.15** | 17.19 | **+19.27** |
| FSS$^{\text{per}}$ | ODISE + Ours | **34.05** | 22.94 | +23.36 |
| ADE$^{\text{per}}$ | ODISE + Ours | **13.43** | 12.18 | +12.19 |

### 消融实验

| 配置 | mIoU | IoU$^{\text{per}}$ | IoU$^{\text{per}}_{\text{precision}}$ | IoU$^{\text{per}}_{\text{recall}}$ | 说明 |
|---|---|---|---|---|---|
| 无个性化 | 77.32 | 68.25 | 92.25 | 72.95 | 高precision低recall |
| +文本提示调优 | 77.89 | 69.70 | 74.75↓ | 91.04↑ | recall升但precision剧降 |
| +文本+负掩码 | 77.89 | 73.71 | 80.07↑ | 90.17 | 负掩码有效抑制假阳性 |
| +文本+视觉注入 | 77.65 | 65.94 | 70.06↓ | 91.58↑ | 视觉注入进一步提升recall但precision更差 |
| +文本+负掩码+视觉注入 | **78.29** | **76.80** | **84.51** | **89.07** | 三者协同最优 |

### 关键发现

- **负掩码是核心**：文本提示调优导致precision从92.25暴跌至74.75，负掩码将其恢复至80.07
- **视觉注入需配合负掩码**：单独使用视觉注入反而降低performance（65.94 < 68.25），但与负掩码结合后达到最佳（76.80）
- **mIoU基本不变**：所有配置下原始OVSS性能（mIoU）保持在77-78之间，说明方法不损害原有能力
- **K=1也有效**：仅1张参考图像就能提升IoU$^{\text{per}}$（SAN: 41.08→49.80），验证了实用性
- **跨模型一致性**：在SAN和ODISE两个不同OVSS模型上均有显著提升

## 亮点与洞察

- **问题提出有价值**：个性化OVSS是一个被忽视但实际需求强烈的任务方向，论文首次定义了完整的任务设置和评估体系
- **假阳性问题的发现和解决**：深入分析了text prompt tuning导致假阳性的原因（precision/recall解耦分析），并用负掩码提案优雅解决
- **即插即用设计**：方法可直接应用于任何现成OVSS模型（SAN、ODISE等），无需修改模型架构
- **极少参数量**：仅需训练一个文本嵌入向量 + 两个线性层，200次迭代即可完成
- **基准建设贡献**：建立了FSS$^{\text{per}}$、CUB$^{\text{per}}$、ADE$^{\text{per}}$三个新基准

## 局限与展望

- 当同类物体外观极其相似时（如只有微小logo差异的杯子），区分能力可能不足
- 评估基准主要基于FSS-1000和CUB-200，真实场景的复杂度可能更高
- 视觉嵌入注入使用简单的线性插值（$\alpha$固定），可探索更复杂的融合方式
- 未考虑多个个性化概念同时存在的情况
- 参考图像中的遮挡和视角变化对性能的影响未充分分析

## 相关工作与启发

- 与Yo'LLaVA的个性化VQA任务相关，但Yo'LLaVA需要大量负样本，本方法自动生成
- SAN和ODISE代表了两种不同的OVSS范式（CLIP-based vs Diffusion-based），方法在两者上都有效说明通用性强
- 负掩码提案的思想可以推广到其他需要抑制假阳性的个性化任务（如个性化检测、个性化VQA）
- 文本+视觉联合提示调优的有效性与CoCoOp、MaPLe等多模态提示学习一致

## 评分

- **新颖性**: ⭐⭐⭐⭐ 任务定义新颖且实用，负掩码提案设计巧妙
- **实验充分度**: ⭐⭐⭐⭐ 两个模型+三个数据集+详细消融+定性分析，缺少与更多个性化方法的对比
- **写作质量**: ⭐⭐⭐⭐ 问题阐述清楚，方法描述直观，图表有说服力
- **价值**: ⭐⭐⭐⭐ 开辟了个性化OVSS新赛道，基准和方法都有参考价值

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation](floss_free_lunch_in_openvocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Stepping Out of Similar Semantic Space for Open-Vocabulary Segmentation](stepping_out_of_similar_semantic_space_for_open-vocabulary_segmentation.md)
- [\[ICCV 2025\] Auto-Vocabulary Semantic Segmentation](auto-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Effective SAM Combination for Open-Vocabulary Semantic Segmentation](../../CVPR2025/segmentation/effective_sam_combination_for_open-vocabulary_semantic_segmentation.md)

<!-- RELATED:END -->
