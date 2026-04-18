---
title: >-
  [论文解读] Object-level Correlation for Few-Shot Segmentation
description: >-
  [ICCV 2025][图像分割][图像分割] 提出 OCNet，通过模仿生物视觉过程构建**目标级别**（而非图像级别）的 support-query 关联，先挖掘查询图像中的通用物体，再从中识别目标物体，有效抑制背景中的无关物体噪声。
tags:
  - ICCV 2025
  - 图像分割
  - object-level correlation
  - prototype learning
  - optimal transport
  - hard pixel noise
---

# Object-level Correlation for Few-Shot Segmentation

**会议**: ICCV 2025  
**arXiv**: [2509.07917](https://arxiv.org/abs/2509.07917)  
**代码**: 无  
**领域**: segmentation  
**关键词**: few-shot segmentation, object-level correlation, prototype learning, optimal transport, hard pixel noise

## 一句话总结

提出 OCNet，通过模仿生物视觉过程构建**目标级别**（而非图像级别）的 support-query 关联，先挖掘查询图像中的通用物体，再从中识别目标物体，有效抑制背景中的无关物体噪声。

## 研究背景与动机

少样本语义分割（FSS）的核心在于构建 support 目标和 query 图像之间的关联。现有方法主要建立**图像级别关联**（support 目标 ↔ 整个 query 图像），存在以下问题：

**Hard pixel noise**：关联中包含无关背景物体（如真实背景物体、基类物体、无关新类物体）
2. 一些后处理方法（如 BAM、ABCB）试图消除部分噪声，但仍无法处理**无关新类物体**（如 query 图中同时出现狗和人，但只需分割狗）
3. 在多新类物体共现时，图像级关联难以准确识别目标

**生物视觉启发**：人类视觉系统先以预注意方式计算全局显著性（找到通用物体），然后基于任务线索从中选择目标。目标识别在通用物体中比在整幅图像中更有效。

## 方法详解

### 整体框架

OCNet 由两个核心模块组成：
1. **GOMM**（General Object Mining Module）：从 query 图像中挖掘通用物体特征
2. **CCM**（Correlation Construction Module）：在 support 目标和 query 通用物体之间建立目标级关联

流程：预训练骨干提取特征 → GOMM 生成通用物体特征 $F_g$ → CCM 利用 support 原型与 $F_g$ 构建目标级关联 $F_c$ → FPN 解码器预测

### 关键设计

1. **GOMM - 通用物体挖掘模块**：

    - **通用物体掩码生成**：由于无 query ground truth，使用 CAM 获取原始通用物体掩码，融合 support-query 高层特征余弦相似度先验，通过阈值 $\tau=0.6$ 分割：
    $M_g = \mathbb{1}_\tau(\text{Max}(\text{Cosine}(F_q^h, F_s^h) \oplus \text{CAM}(F_q^h)))$
    - **初始通用物体特征**：随机初始化通用物体原型 $P_g \in \mathbb{R}^{N_g \times C}$，与 query 特征计算余弦相似度分配，拼接后 1×1 卷积生成 $F_{ig}$
    - **信息补全**：通过交叉注意力融合 $F_{ig}$ 和 $F_q$：$F_g = \text{Atten}(F_q, F_{ig}, F_{ig}) + F_q$
    - 设计动机：通用物体掩码虽不完美，但适度不完整有利于原型的泛化和重建能力

2. **CCM - 关联构建模块**：

    - **Support 原型获取**：使用多频率池化（MFP）从 support 特征生成原型 $P_s \in \mathbb{R}^{L \times C}$（$L=49$）
    - **前景/背景原型选择**：通过欧氏距离比较原型激活掩码 $M_{sp}$ 与真实掩码 $M_s$ 的相似性，TopK 为前景原型索引 $ID_t$，LowK 为背景原型索引 $ID_l$
    - **最优传输分配**：将原型分配建模为 OT 问题，使用 Sinkhorn 算法（$\epsilon=0.05$）求解最优传输矩阵 $T^*$，生成原型分配掩码 $M_{pa}$
    - **关联构建**：用分配掩码监督原型分配，通过矩阵乘法融合 support 和 query 信息，得到目标级关联 $F_c = \text{Alloc}(P_q, \text{Argmax}(\hat{M}_{pa})) \oplus F_g$
    - 设计动机：前景原型捕获目标信息，背景原型主动抑制噪声像素（之前方法忽略了背景原型的作用）

3. **前景+背景原型双重机制**：

    - 与之前 FPTrans 等只用前景原型不同，CCM 同时利用背景原型
    - 前景原型负责激活目标区域，背景原型负责抑制 hard pixel noise
    - 通过分配掩码将二者统一在最优传输框架下
    - 设计动机：抑制噪声与增强目标同等重要

### 损失函数 / 训练策略

总损失由三部分组成：$\mathcal{L}_f = \mathcal{L}_t + \mathcal{L}_g + \mathcal{L}_p$
- $\mathcal{L}_t = \text{CE}(\hat{M}_q, M_q)$：目标分割损失
- $\mathcal{L}_g = \text{CE}(\hat{M}_g, M_g)$：通用物体分割损失
- $\mathcal{L}_p = \text{CE}(\hat{M}_{pa}, M_{pa})$：原型分配损失

训练配置：SGD 优化器，lr=0.005，batch size=4；PASCAL-5^i 训练 200 epochs，COCO-20^i 训练 75 epochs；图像裁剪到 473×473（PASCAL）或 641×641（COCO）。

## 实验关键数据

### 主实验 (表格)

**PASCAL-5^i 1-shot/5-shot（ResNet-50）**：

| 方法 | 1-shot Mean mIoU | 1-shot FB-IoU | 5-shot Mean mIoU | 5-shot FB-IoU |
|------|-----------------|---------------|-----------------|---------------|
| BAM (CVPR'22) | 67.8 | 79.7 | 70.9 | 82.2 |
| AENet (ECCV'24) | 69.8 | 80.8 | 74.1 | 84.5 |
| ABCB (CVPR'24) | 70.6 | - | 73.6 | - |
| HMNet (NeurIPS'24) | 70.4 | 81.6 | 74.1 | 84.4 |
| **OCNet** | **71.4** | **82.2** | **74.5** | **84.7** |

**COCO-20^i 1-shot/5-shot（ResNet-50）**：

| 方法 | 1-shot Mean mIoU | 1-shot FB-IoU | 5-shot Mean mIoU | 5-shot FB-IoU |
|------|-----------------|---------------|-----------------|---------------|
| AENet (ECCV'24) | 49.4 | 73.6 | 56.7 | 76.5 |
| ABCB (CVPR'24) | 50.0 | - | 55.1 | - |
| **OCNet** | **51.5** | **73.7** | **57.0** | **76.8** |

### 消融实验 (表格)

**GOMM 和 CCM 模块消融（PASCAL-5^i, 1-shot, ResNet-50）**：

| GOMM | CCM | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Mean |
|------|-----|--------|--------|--------|--------|------|
| ✗ | ✗ | 67.5 | 73.4 | 66.5 | 61.6 | 67.3 |
| ✓ | ✗ | 69.9 | 74.2 | 68.3 | 63.9 | 69.1 |
| ✗ | ✓ | 71.9 | 74.7 | 69.8 | 63.0 | 69.9 |
| ✓ | ✓ | **73.5** | **75.9** | **71.1** | **64.9** | **71.4** |

- GOMM 单独贡献 +1.8% mIoU
- CCM 单独贡献 +2.6% mIoU
- 组合贡献 +4.1% mIoU，说明两模块互补有效

### 关键发现

- 目标级关联相比图像级关联在所有设定下均更优，证明了"先找物体再识别目标"策略的有效性
- 背景原型的引入对抑制 hard pixel noise 至关重要（之前方法忽略了这一点）
- 通用物体掩码虽然不完美，但适度不完整反而有利于原型泛化
- 在 COCO-20^i 等更具挑战性的数据集上，OCNet 的优势更加明显
- VGG-16 和 ResNet-50 两种骨干均取得一致提升

## 亮点与洞察

1. **生物视觉启发**：模仿人类"显著性→目标选择"的两阶段视觉处理过程，将抽象的认知过程转化为可计算的模块
2. **从图像级到目标级的范式转变**：不再将整个 query 图像与 support 匹配，而是先提取通用物体再精准对应
3. **前景+背景原型双重机制**：利用背景原型主动抑制噪声，而非仅被动过滤
4. **最优传输建模分配**：将原型到像素的分配建模为 OT 问题，获得全局最优的分配方案

## 局限与展望

- CAM 生成的通用物体掩码质量不稳定，有时可能遗漏重要目标
- 最优传输求解增加了计算开销，Sinkhorn 迭代次数需要权衡
- 仅在 PASCAL-5^i 和 COCO-20^i 上验证，未在更多领域（如医学、遥感）测试
- 当 query 图像中只有单个物体时，目标级关联的优势可能不如多物体场景明显
- 未与最新的基于大规模预训练模型（如 SAM）的方法比较

## 相关工作与启发

- 目标级关联的思想可推广到其他密集预测任务中 support-query 匹配的场景
- "先分割通用物体，再识别目标"的两阶段策略在开放世界任务中可能更有价值
- OT 建模原型分配是一个值得深入探索的方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ 从图像级到目标级关联的转变有创意，生物视觉启发合理
- **实验充分度**: ⭐⭐⭐⭐ 双数据集、双骨干、充分消融，定性分析清晰
- **写作质量**: ⭐⭐⭐⭐ 动机阐述清楚，方法描述详细，图示信息丰富
- **价值**: ⭐⭐⭐⭐ 提供了 FSS 领域新的思路，但提升幅度有限（~1-2% mIoU）
