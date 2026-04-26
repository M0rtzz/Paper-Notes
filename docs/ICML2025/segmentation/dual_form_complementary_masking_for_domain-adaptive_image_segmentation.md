---
title: >-
  [论文解读] Dual form Complementary Masking for Domain-Adaptive Image Segmentation
description: >-
  [ICML 2025][图像分割][无监督域自适应] 提出 MaskTwins 框架，将掩码重建理论化为稀疏信号重建问题，证明互补掩码对（dual form complementary masks）在提取域无关特征方面具有理论优势，并在端到端训练中通过互补掩码一致性约束实现域自适应分割。
tags:
  - ICML 2025
  - 图像分割
  - 无监督域自适应
  - 互补掩码
  - 掩码图像建模
  - 稀疏信号重建
  - 一致性正则化
  - 语义分割
---

# Dual form Complementary Masking for Domain-Adaptive Image Segmentation

**会议**: ICML 2025

**arXiv**: [2507.12008](https://arxiv.org/abs/2507.12008)

**作者**: Jiawen Wang, Yinda Chen, Xiaoyu Liu, Che Liu, Dong Liu, Jianqing Gao, Zhiwei Xiong

**领域**: 分割 (域自适应语义分割)

**关键词**: 无监督域自适应, 互补掩码, 掩码图像建模, 稀疏信号重建, 一致性正则化, 语义分割

## 一句话总结

提出 MaskTwins 框架，将掩码重建理论化为稀疏信号重建问题，证明互补掩码对（dual form complementary masks）在提取域无关特征方面具有理论优势，并在端到端训练中通过互补掩码一致性约束实现域自适应分割。

## 研究背景与动机

无监督域自适应（UDA）旨在利用有标注的源域数据和无标注的目标域数据，弥合域间差异。现有方法主要分为统计矩对齐、对抗学习和自训练三类。近期 MIC 等工作将掩码图像建模（MIM）与一致性正则化相结合用于 UDA，但存在两个关键不足：

**缺乏理论基础**：现有方法仅将掩码视为输入图像的一种特殊扰动形式，缺乏对掩码重建有效性的理论分析

**未充分利用互补性**：前期工作（如 MIC）使用随机掩码，未深入研究互补掩码在单模态场景下的潜力

本文从压缩感知（Compressed Sensing）理论出发，将掩码重建重新定义为稀疏信号重建问题，首次从理论上证明互补掩码在信息保留、泛化界和特征一致性三方面均优于随机掩码。

## 方法详解

### 整体框架

MaskTwins 采用教师-学生架构，核心流程：

1. 对源域数据计算有监督分割损失
2. 对目标域图像生成互补掩码对 $(D, 1-D)$，产生两个互补视图
3. 教师模型（EMA 更新）对无掩码目标图像生成伪标签
4. 学生模型对两个互补掩码图像分别预测，通过一致性损失和互补掩码损失进行约束

### 理论分析

**视觉数据模型**：输入图像 $X = S + E + N$，其中 $S$ 为稀疏信号，$E$ 为环境因子，$N \sim \mathcal{N}(0, \sigma^2 I)$ 为高斯噪声。

**互补掩码定义**：$D \in \{0,1\}^{H \times W}$，每个元素独立采样自 $\text{Bernoulli}(0.5)$，互补掩码对为 $(D, \mathbf{1}-D)$。

**核心定理**：

- **信息保留（Theorem 1）**：$\mathbb{E}[\text{IP}(X_D, X_{1-D})] \geq \mathbb{E}[\text{IP}(X_{R_1}, X_{R_2})]$
- **泛化界（Theorem 2）**：互补掩码的泛化界更紧，不含 $\sqrt{HWC}$ 的额外项
- **特征一致性（Theorem 3）**：互补掩码的特征一致性误差不含环境因子 $\|E\|_F$ 项

### 关键设计：互补掩码学习

对目标域图像 $X^T$，生成互补掩码图像对：

$$X^T_{cm} = \{D \odot X^T, (1-D) \odot X^T\}$$

掩码按 patch 级采样：$D_{mb+1:(m+1)b, nb+1:(n+1)b} \sim \text{Bernoulli}(1-r)$

**互补掩码损失**：

$$\mathcal{L}^T_{cm} = \mathbb{E}[\|p^T_{j,D}, p^T_{j,1-D}\|_2]$$

**掩码一致性学习损失**：

$$\mathcal{L}^T_{cl} = \mathbb{E}[\lambda \cdot \mathcal{L}_{ce}(p^T_{j,D}, \hat{y}^T_j) + (1-\lambda) \cdot \mathcal{L}_{ce}(p^T_{j,1-D}, \hat{y}^T_j)]$$

### 损失函数

$$\mathcal{L}_{total} = \mathcal{L}^S_{sup} + \mathcal{L}^T_{cl} + \lambda_{cm} \mathcal{L}^T_{cm}$$

其中 $\mathcal{L}^S_{sup} = \mathbb{E}[-y^S_i \log(p^S_i)]$ 为源域有监督交叉熵损失。教师模型通过 EMA 更新：$\phi_{t+1} \leftarrow \alpha \phi_t + (1-\alpha)\theta_t$。

## 实验关键数据

### 主实验：SYNTHIA→Cityscapes 语义分割（mIoU, %）

| 方法 | Road | SW | Build | TL | TS | Veg. | Sky | PR | Rider | Car | Bus | Motor | Bike | mIoU |
|------|------|------|-------|------|------|------|------|------|-------|------|------|-------|------|------|
| DAFormer | 84.5 | 40.7 | 88.4 | 55.0 | 54.6 | 86.0 | 89.8 | 73.2 | 48.2 | 87.2 | 53.2 | 53.9 | 61.7 | 67.4 |
| HRDA | 85.2 | 47.7 | 88.8 | 65.7 | 60.9 | 85.3 | 92.9 | 79.4 | 52.8 | 89.0 | 64.7 | 63.9 | 64.9 | 72.4 |
| MIC | 86.6 | 50.5 | 89.3 | 66.7 | 63.4 | 87.1 | 94.6 | 81.0 | 58.9 | 90.1 | 61.9 | 67.1 | 64.3 | 74.0 |
| **MaskTwins** | **96.0** | **70.1** | **89.5** | **66.8** | 62.1 | **89.1** | 94.3 | **81.5** | **59.7** | **90.5** | **66.6** | **67.7** | 63.6 | **76.7** |

**关键发现**：MaskTwins 以 **+2.7 mIoU** 超越 SOTA（MIC），sidewalk 类别提升惊人（50.5→70.1, +19.6 IoU），road 类别 +4.8 IoU。

### 生物图像分割：线粒体语义分割（IoU, %）

| 方法 | V2L1 | V2L2 | R2H | H2R |
|------|------|------|------|------|
| DA-ISC | 68.7 | 74.3 | 74.8 | 79.4 |
| CAFA | 71.8 | 75.4 | 76.3 | 80.6 |
| **MaskTwins** | **75.0** | **78.6** | **78.4** | **81.9** |

### 消融实验

| CL | CMask | RMask | EMA | AdaIN | mIoU |
|----|-------|-------|-----|-------|------|
| - | - | - | - | - | 53.7 |
| ✓ | - | ✓ | - | - | 74.3 |
| ✓ | - | ✓ | ✓ | ✓ | 75.2 |
| ✓ | ✓ | - | - | - | 76.0 |
| ✓ | ✓ | - | ✓ | ✓ | **76.7** |

**关键发现**：互补掩码（CMask）vs 随机掩码（RMask）：76.0 vs 74.3，仅替换掩码策略即可获得 +1.7 mIoU 提升。

### 超参数消融

| 掩码比例 $r$ | mIoU | | Patch 大小 $b$ | mIoU |
|-------------|------|-|---------------|------|
| 0.1 | 72.0 | | 32 | 76.2 |
| 0.2 | 74.6 | | 64 | **76.7** |
| 0.3 | 75.4 | | 128 | 75.9 |
| 0.4 | 76.5 | | 256 | 75.6 |
| 0.5 | **76.7** | | 512 | 75.0 |

最优配置：$r=0.5$, $b=64$（约 1/16 输入尺寸）。

## 亮点与洞察

1. **理论驱动的方法设计**：首次将掩码重建与压缩感知理论联系，从信息保留、泛化界、特征一致性三个维度严格证明互补掩码的优越性，理论与实验完美吻合
2. **简洁高效**：MaskTwins 不引入额外可学习参数，仅通过改变掩码策略（随机→互补）即可获得显著性能提升
3. **跨领域通用性**：在自然图像（Cityscapes）、EM 线粒体分割和 3D 突触检测上均取得 SOTA，从 2D 到 3D 场景均有效
4. **sidewalk 特别提升**：最难适应的 sidewalk 类别提升 +19.6 IoU，说明互补掩码特别擅长学习需要上下文关系的类别

## 局限性

1. 对小目标类别改进有限，互补掩码可能导致小目标被完全遮盖
2. 理论分析基于线性特征提取假设，与深度网络的非线性特性存在差距
3. 仅在合成→真实的 UDA 场景上验证，缺乏真实→真实的域适应实验
4. 掩码 patch 大小和比例需要针对不同任务手动调整

## 相关工作与启发

- **MIC (CVPR 2023)**：首次在 UDA 中使用掩码一致性，但仅用随机掩码且缺乏理论分析
- **HRDA (ECCV 2022)**：多分辨率域适应框架，MaskTwins 以其为基础架构
- **压缩感知理论**：互补掩码的信号重建优势可推广到其他需要域不变特征的任务

**启发**：互补掩码的理论框架可扩展到多视图学习（论文已证明 K 个互补掩码的多视图定理），为视频域自适应分割、多模态融合等方向提供理论指导。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 从压缩感知角度为互补掩码提供理论基础，视角新颖 |
| 技术深度 | 5 | 完备的理论证明（5个定理）+ 详尽的实验验证 |
| 实验充分性 | 5 | 6个数据集、自然/生物/3D 场景、全面消融 |
| 写作质量 | 4 | 结构清晰，理论推导严谨 |
| 实用价值 | 4 | 即插即用的掩码策略，无额外参数开销 |
| 总评 | 4.4 | 理论与实践结合的典范之作 |

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Balanced Learning for Domain Adaptive Semantic Segmentation](balanced_learning_for_domain_adaptive_semantic_segmentation.md)
- [\[CVPR 2026\] Masked Representation Modeling for Domain-Adaptive Segmentation](../../CVPR2026/segmentation/mrm_masked_representation_modeling_domain_adaptive.md)
- [\[NeurIPS 2025\] Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation](../../NeurIPS2025/segmentation/towards_unsupervised_domain_bridging_via_image_degradation_in_semantic_segmentat.md)
- [\[CVPR 2025\] Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation](../../CVPR2025/segmentation/dual-agent_optimization_framework_for_cross-domain_few-shot_segmentation.md)
- [\[CVPR 2026\] Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](../../CVPR2026/segmentation/seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)

<!-- RELATED:END -->
