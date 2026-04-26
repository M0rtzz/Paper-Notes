---
title: >-
  [论文解读] Eliminating Feature Ambiguity for Few-Shot Segmentation
description: >-
  [ECCV 2024][图像分割][少样本分割] 提出AENet插件网络，通过挖掘判别性查询前景区域来消除特征歧义，增强交叉注意力中的前景-前景匹配，可即插即用地提升现有少样本分割方法性能（SCCAN 1-shot在PASCAL-5i上+3.0%）。
tags:
  - ECCV 2024
  - 图像分割
  - 少样本分割
  - 特征歧义
  - 交叉注意力
  - 判别性先验掩码
  - 插件网络
---

# Eliminating Feature Ambiguity for Few-Shot Segmentation

**会议**: ECCV 2024  
**arXiv**: [2407.09842](https://arxiv.org/abs/2407.09842)  
**代码**: 有 (https://github.com/Sam1224/AENet)  
**领域**: 图像分割  
**关键词**: 少样本分割, 特征歧义, 交叉注意力, 判别性先验掩码, 插件网络

## 一句话总结

提出AENet插件网络，通过挖掘判别性查询前景区域来消除特征歧义，增强交叉注意力中的前景-前景匹配，可即插即用地提升现有少样本分割方法性能（SCCAN 1-shot在PASCAL-5i上+3.0%）。

## 研究背景与动机

少样本分割（Few-Shot Segmentation, FSS）旨在利用少量标注的support样本分割包含任意类别的query图像。核心思路是学习类别无关的模式：找到query特征中与support前景（FG）特征相似的部分，分类为前景。

**现有方法的发展脉络**：
- **原型方法**（PFENet, BAM等）：将support FG特征压缩为原型，与query特征比较。但原型压缩会导致**信息损失**和**空间结构破坏**。
- **交叉注意力方法**（CyCTR, SCCAN, HDMNet等）：利用交叉注意力在query和未压缩的support FG特征之间进行像素级匹配，选择性激活query中与support FG相同类别的特征。

**核心问题——特征歧义（Feature Ambiguity）**：

作者首次识别了交叉注意力方法中的一个被忽视的关键问题。由于深层backbone（如ResNet50 Block4）的**大感受野**，提取的FG/BG像素特征不可避免地混入了周围的BG/FG特征，尤其在FG与BG的边界区域。这导致：

**FG特征被BG污染**：query的FG像素包含了FG（目标物体）和BG（背景物体）的混合特征。例如，鸟的像素特征中也包含了栅栏的特征。

**BG特征被FG污染**：support的BG像素也可能包含FG信息（如边界处的人），导致它们与query FG产生虚假的高相似度。

**匹配失效**：由于query FG和support FG特征分别混入了不同类别的BG特征，它们之间的相似度变小，交叉注意力分数降低，query FG无法充分聚合support的FG信息。

**直觉验证**：通过可视化先验掩码可以直接观察到问题——FG先验 $M_{Prior}^{FG}$ 中有大量BG区域被错误激活（它们混入了FG特征所以与support FG相似），BG先验 $M_{Prior}^{BG}$ 中support BG也能与query FG匹配。

**解决思路**：抑制那些同时与support FG和BG都相似的**歧义区域**（说明这些区域被大量BG特征污染），保留最具判别性的query FG区域，用这些"纯净"的FG特征去修正被污染的query和support特征。

## 方法详解

### 整体框架

AENet是一个即插即用的插件网络，由两个核心模块组成：

- **先验生成器（Prior Generator, PG）**：无学习参数的先验掩码生成模块，用于定位判别性query FG区域
- **歧义消除器（Ambiguity Eliminator, AE）**：利用判别性FG特征修正query和support特征

AENet可插入任何基于交叉注意力的FSS方法。以SCCAN为例：用PG替换原有的PMA模块，在每个SCCA块前插入一个AE模块。

### 关键设计

1. **先验生成器（PG）**：利用高层特征 $F_Q^h$、$F_S^h$ 和support掩码 $M_S$ 生成判别性先验掩码。

    - 首先通过全局平均池化获取support的FG和BG原型：$P_S^{FG} = GAP(F_S^h, M_S)$，$P_S^{BG} = GAP(F_S^h, 1-M_S)$
    - 分别计算query特征与两个原型的余弦相似度，归一化后得到 $M_{Prior}^{FG}$ 和 $M_{Prior}^{BG}$
    - **关键操作——截断减法**：$M_{Prior}^{Disc} = ReLU(M_{Prior}^{FG} - M_{Prior}^{BG})$
    - **设计动机**：同时与support FG和BG都相似的区域（歧义区域）在减法后值趋近0或为负，被ReLU截断。剩余高响应区域是真正具有判别性的FG区域，受BG污染最少。
    - 最终将 $M_{Prior}^{FG}$ 和 $M_{Prior}^{Disc}$ 拼接作为最终先验掩码，前者提供粗略FG定位，后者提供判别性锚点。
    - **内存优势**：每次计算复杂度为 $HW \times 1$，远低于PFENet/SCCAN的 $HW \times HW$。

2. **歧义消除器（AE）**：利用判别性FG区域实际修正特征。

    - 中层query特征 $F_Q$ 通过线性层投影为 $K$ 和 $V$，support特征 $F_S$ 投影为 $Q$
    - 使用PG计算判别性掩码 $M^{Disc}$，并用辅助BCE损失 $\mathcal{L}_{aux} = BCE(M^{Disc}, M_Q)$ 监督
    - 通过矩阵乘法提取判别性query FG原型：$P_Q^{FG} = Softmax(M^{Disc}) \otimes V$
    - 计算support和query FG原型的余弦相似度：$\alpha = (Cosine(P_S^{FG}, P_Q^{FG}) + 1) / 2$
    - 加权融合得到综合FG原型：$P^{FG} = \alpha \cdot P_S^{FG} + (1-\alpha) \cdot P_Q^{FG}$
    - 将 $P^{FG}$ 扩展后与query/support特征拼接并通过线性层修正：$F_* = Linear(F_* \| P^{FG})$
    - **设计动机**：$P^{FG}$ 融合了最纯净的support和query FG信息，与原始被污染的特征拼接后，可增大FG信息在混合特征中的比例，从而增强FG-FG匹配。

3. **Transformer包装**：AE模块被Transformer块包装，输出经修正的query和support特征送入后续交叉注意力块。

### 损失函数 / 训练策略

以SCCAN为例，总损失为：

$$\mathcal{L} = Dice(\hat{M}_Q, M_Q) + \lambda \cdot \frac{1}{N} \sum_{i=1}^{N} BCE(M_i^{Disc}, M_Q)$$

其中 $\lambda=1$，$N$ 为注意力块数量。主损失为Dice损失（保持原baseline不变），辅助损失为每个AE模块输出的判别性掩码上的BCE损失。

## 实验关键数据

### 主实验

PASCAL-5i 上 ResNet50 backbone 结果（mIoU%）：

| 方法 | 1-shot Mean | 提升 | 5-shot Mean | 提升 |
|------|------------|------|------------|------|
| CyCTR | 64.2 | - | 65.6 | - |
| CyCTR + AENet | **69.0** | +4.8 | **72.6** | +7.0 |
| SCCAN | 66.8 | - | 70.3 | - |
| SCCAN + AENet | **69.8** | +3.0 | **74.1** | +3.8 |
| HDMNet | 69.4 | - | 71.8 | - |
| HDMNet + AENet | **70.3** | +0.9 | **74.2** | +2.4 |

COCO-20i 上 ResNet50 backbone 结果（mIoU%）：

| 方法 | 1-shot Mean | 提升 | 5-shot Mean | 提升 |
|------|------------|------|------------|------|
| CyCTR | 40.3 | - | 45.6 | - |
| CyCTR + AENet | **47.0** | +6.7 | **52.4** | +6.8 |
| SCCAN | 46.3 | - | 53.9 | - |
| SCCAN + AENet | **49.4** | +3.1 | **56.7** | +2.8 |
| HDMNet | 49.6 | - | 55.3 | - |
| HDMNet + AENet | **51.3** | +1.7 | **57.1** | +1.8 |

### 消融实验

组件消融（PASCAL-5i, ResNet50, 1-shot）：

| PG | AE | BAM | Mean mIoU | 提升 |
|----|----|----|-----------|------|
| ✗ | ✗ | ✗ | 66.8 | baseline |
| ✓ | ✗ | ✗ | 67.8 | +1.0 |
| ✗ | ✓ | ✗ | 67.9 | +1.1 |
| ✓ | ✓ | ✗ | 68.3 | +1.5 |
| ✓ | ✓ | ✓ | **69.8** | +3.0 |

AE中减法操作的重要性：

| AE配置 | Mean mIoU | 说明 |
|--------|-----------|------|
| 无AE | 66.8 | baseline |
| $M^{FG}$（无减法） | 66.9 | 仅用FG信息几乎无效 |
| $M^{Disc}$（有减法） | **67.9** | 减法操作是关键 |

### 关键发现

- **AENet在更困难数据集上提升更大**：COCO-20i上CyCTR提升6.7%，远超PASCAL-5i的4.8%。因为COCO图像中小物体多、背景复杂，特征歧义问题更严重。
- **减法操作是核心**：不做减法仅用FG信息修正几乎无提升（66.8→66.9），因为模型会学到类别特定的解耦模式，无法泛化到新类别。减法提供了类别无关的引导。
- **损失权重 $\lambda=1$ 最优**：即使 $\lambda=0$（不加辅助监督），mIoU已达69%+，说明判别性掩码本身的特征修正就有效。

## 亮点与洞察

- 首次识别了特征歧义问题对FSS中交叉注意力匹配的负面影响，问题定义精准
- PG的减法操作极其简洁（无学习参数），却非常有效，体现了"做减法"的设计美学
- 作为插件网络的设计使其易于集成到多种baseline中，实用性强
- AE中 $\alpha$ 加权融合query和support FG原型的设计巧妙——根据两者的一致性动态调整权重

## 局限与展望

- 当FG物体极小时（占图像比例<5%），判别性区域可能不足以提供有效的修正信号
- PG依赖原型级别的FG/BG计算，当support中存在多个实例时可能不够精细
- 可探索在4D相关方法（如HSNet, VAT）上的适配
- 可考虑多尺度判别性掩码融合，利用不同层级特征的互补性

## 相关工作与启发

- PFENet的先验掩码思想被优雅地扩展——从仅计算FG相似度到同时考虑FG和BG相似度的差异
- SCCAN的自校准交叉注意力与AENet互补——前者处理BG特征错误匹配，后者处理FG特征被BG污染
- 减法操作的class-agnostic特性值得在其他few-shot任务中探索

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次识别特征歧义问题，减法操作简洁有效
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个baseline × 两个数据集 × 两个backbone × 详细消融
- **写作质量**: ⭐⭐⭐⭐ 问题阐述清晰，可视化验证充分
- **价值**: ⭐⭐⭐⭐ 即插即用设计具有很好的实用性，在多个baseline上一致提升

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] UniFS: Universal Few-Shot Instance Perception with Point Representations](unifs_universal_few-shot_instance_perception_with_point_representations.md)
- [\[ICCV 2025\] Object-level Correlation for Few-Shot Segmentation](../../ICCV2025/segmentation/object-level_correlation_for_few-shot_segmentation.md)
- [\[ECCV 2024\] FREST: Feature Restoration for Semantic Segmentation under Multiple Adverse Conditions](frest_feature_restoration_for_semantic_segmentation_under_multiple_adverse_condi.md)
- [\[ICCV 2025\] MOVE: Motion-Guided Few-Shot Video Object Segmentation](../../ICCV2025/segmentation/move_motion-guided_few-shot_video_object_segmentation.md)
- [\[NeurIPS 2025\] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](../../NeurIPS2025/segmentation/sansa_unleashing_the_hidden_semantics_in_sam2_for_few-shot_segmentation.md)

<!-- RELATED:END -->
