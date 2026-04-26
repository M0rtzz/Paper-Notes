---
title: >-
  [论文解读] SACB-Net: Spatial-Awareness Convolutions for Medical Image Registration
description: >-
  [CVPR 2025][医学图像][医学图像配准] 提出3D空间感知卷积块（SACB），通过对特征图进行无监督聚类并为不同空间区域生成自适应卷积核，结合金字塔流估计器实现多尺度形变场组合，在脑部和腹部CT配准任务上超越现有SOTA方法。
tags:
  - CVPR 2025
  - 医学图像
  - 医学图像配准
  - 空间自适应卷积
  - 金字塔流估计
  - 形变场
  - 3D配准
---

# SACB-Net: Spatial-Awareness Convolutions for Medical Image Registration

**会议**: CVPR 2025  
**arXiv**: [2503.19592](https://arxiv.org/abs/2503.19592)  
**代码**: https://github.com/x-xc/SACB_Net  
**领域**: 医学图像  
**关键词**: 医学图像配准, 空间自适应卷积, 金字塔流估计, 形变场, 3D配准

## 一句话总结

提出3D空间感知卷积块（SACB），通过对特征图进行无监督聚类并为不同空间区域生成自适应卷积核，结合金字塔流估计器实现多尺度形变场组合，在脑部和腹部CT配准任务上超越现有SOTA方法。

## 研究背景与动机

**领域现状**：深度学习医学图像配准方法已取得显著进展，主流方法包括基于U-Net的VoxelMorph系列、模型驱动的Fourier-Net/B-Spline方法，以及级联/金字塔的多尺度方法如LapIRN、ModeT等。

**现有痛点**：几乎所有现有方法都依赖空间共享的卷积核——即同一层内所有空间位置使用相同的卷积权重。然而在医学图像中，不同解剖区域（如脑灰质、白质、脑脊液）的形变特征差异很大，共享卷积核无法捕捉非局部区域的空间变化信息，导致形变场估计不够精确。

**核心矛盾**：标准卷积的空间不变性与医学图像配准中形变的空间异质性之间存在根本矛盾——组织形态与形变紧密相关，不同区域的体素/特征需要不同的关注度。

**本文目标**：(1) 设计能感知空间区域差异的卷积机制；(2) 在金字塔框架中集成该机制以处理大形变。

**切入角度**：作者从2D图像全色锐化中的内容自适应卷积获得启发，将其扩展到3D医学配准场景。核心观察是：如果能基于特征相似性将体素分成不同的空间聚类，然后为每个聚类生成专属的卷积核，就能实现空间感知的特征提取。

**核心 idea**：用无监督K-Means聚类将特征图划分为语义相似的空间区域，利用聚类中心通过MLP生成区域特异的自适应卷积核权重和偏置，从而取代传统的空间共享卷积。

## 方法详解

### 整体框架

SACB-Net采用"共享编码器 + 金字塔流估计器"的架构。输入一对移动图像和固定图像，共享编码器提取5个尺度的多分辨率特征金字塔。然后从最粗尺度（scale5）到最细尺度（scale1），逐级进行SACB特征增强和相似度匹配计算，通过流组合逐步生成最终形变场。

### 关键设计

1. **3D空间感知卷积块（SACB）**:

    - 功能：增强特征图的空间表达能力，使不同解剖区域获得不同的特征处理
    - 核心思路：SACB由三部分组成——空间上下文估计模块、自适应核生成器、残差连接。首先将特征图$\mathbf{F}$展开为局部patches并计算空间均值，然后reshape后用GPU K-Means聚类得到聚类索引矩阵$S$和每个聚类的中心$S_n^c$。接着两个MLP分别从中心$S_n^c$生成空间权重$\mathcal{F}_w(S_n^c)$和偏置$\mathcal{F}_b(S_n^c)$，权重与全局可学习卷积核$\mathbf{W}$逐元素相乘得到区域特异的卷积核$\mathbf{W}_n = \mathcal{F}_w(S_n^c) \odot \mathbf{W}$。最终通过残差连接输出增强特征：$\hat{\mathbf{F}} = \mathbf{F} + \sigma(SAC(\mathbf{F}))$
    - 设计动机：不使用标签信息做聚类是因为移动和固定图像的标注在特征空间中可能不一致，且标签通常稀缺。MLP生成权重而非直接学习多套卷积核，既保持了参数效率又实现了自适应

2. **金字塔流估计器**:

    - 功能：多尺度渐进式估计形变场，处理大形变
    - 核心思路：从最粗尺度开始，每级先用SACB增强移动和固定图像特征，再计算相似度匹配得分$M_{sim}$。匹配得分通过固定特征与展开的移动特征的内积并Softmax得到，然后乘以3D位置网格$G$得到子形变流。上一级的流上采样后用于warp当前级的移动特征，再计算残差流并与上采样流组合。最终级（scale1）不做匹配而用两层卷积直接估计
    - 设计动机：粗到细的策略让低分辨率处理大位移、高分辨率处理精细对齐。搜索窗口$k=3$在最粗尺度对应较大实际位移，在最细尺度只对应1个体素，因此在scale1不做匹配以节省计算

3. **共享编码器**:

    - 功能：为移动和固定图像提取多尺度特征金字塔
    - 核心思路：5层3D卷积加4次平均池化下采样，每层包含3D卷积+InstanceNorm+LeakyReLU(0.1)。移动和固定图像共享同一编码器权重
    - 设计动机：权重共享保证了两支特征的一致性表示空间，有助于后续的相似度匹配

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{sim}(I_m \circ (\phi + \text{Id}), I_f) + \lambda \mathcal{L}_{reg}$，其中相似度项使用NCC（归一化互相关），正则化项为形变场梯度的L2范数$\|\nabla\phi\|_2^2$，鼓励平滑形变。使用Adam优化器，学习率$10^{-4}$，batch size为1，在A100 40GB GPU上训练。

## 实验关键数据

### 主实验

| 数据集 | 指标 | SACB-Net | ModeT (2nd) | LKU | TransMorph | 参数量 |
|--------|------|----------|-------------|-----|------------|-------|
| IXI (30 ROIs) | Dice↑ | **0.769** | 0.758 | 0.765 | 0.754 | 1.11M |
| IXI | \|J\|<0%↓ | 0.083 | 0.114 | 0.109 | 1.579 | — |
| LPBA (54 ROIs) | Dice↑ | **0.731** | 0.721 | 0.706 | 0.695 | — |
| LPBA | HD95↓ | **5.862** | 5.969 | 6.452 | 6.564 | — |
| Abdomen CT | Dice↑ | **0.588** | 0.550 | 0.423 | 0.444 | — |
| Abdomen CT | HD95↓ | **18.253** | 20.351 | 24.252 | 24.187 | — |

### 消融实验

| 配置 | IXI Dice | LPBA Dice | Abdomen Dice |
|------|----------|-----------|--------------|
| 无SACB (baseline) | 0.7643 | 0.7141 | 0.5374 |
| SACB @ scale5 only | 0.7668 | 0.7217 | 0.5444 |
| SACB @ scale5-4 | 0.7679 | 0.7241 | 0.5498 |
| SACB @ scale5-3 | 0.7671 | 0.7266 | 0.5685 |
| SACB @ scale5-2 (N=5) | 0.7683 | 0.7294 | 0.5849 |
| SACB @ scale5-2 (N=7) | **0.7691** | **0.7309** | **0.5881** |
| SACB @ scale5-2 (N=11) | 0.7684 | 0.7300 | 0.5875 |

### 关键发现

- SACB在所有尺度上的效果是累积的，尤其在腹部CT大形变场景提升最大（Dice从0.537提升到0.588，提升约5%）
- 聚类数N=7是最优选择，过多聚类（N=11）反而轻微下降，可能是过度分割导致聚类不够有意义
- 空间维度的patch均值做聚类优于通道维度均值，说明空间局部性对配准至关重要
- 在大形变的腹部CT任务上优势最明显（Dice比次优ModeT高3.8%），验证了金字塔+空间自适应的互补效果

## 亮点与洞察

- **用无监督聚类替代标签驱动的区域划分**：巧妙回避了配准中标签不一致和标签稀缺的问题，同时让模型自动发现有意义的空间分区。这个思路可以迁移到其他需要空间自适应处理的3D任务
- **自适应卷积核 = 全局核 × 区域权重**：不为每个区域独立学习完整卷积核，而是通过MLP生成调制权重来调整全局核，参数效率非常高
- **小模型大效果**：1.11M参数超越46.77M的TransMorph，说明在配准任务中归纳偏置（空间自适应）比模型容量更重要

## 局限与展望

- K-Means聚类是非微分的，不能端到端训练聚类过程，聚类质量依赖特征表示的好坏
- 聚类数N需要手动调节，不同解剖区域的最优聚类数可能不同
- 未考虑微分同胚约束，|J|<0%虽然较低但不为零
- 在scale1使用简单卷积替代匹配可能损失精细对齐能力，可探索更高效的全分辨率匹配方案

## 相关工作与启发

- **vs LKU**: LKU用大核卷积建模长程依赖，但仍是空间共享核；SACB-Net用自适应核实现空间变化感知，Dice更高且参数少一半
- **vs ModeT**: ModeT用多头注意力加权融合多尺度流，SACB-Net用空间聚类自适应增强特征后再做匹配，在大形变场景优势更明显
- **vs TransMorph**: Transformer虽能捕捉长程依赖但参数量巨大（46.77M vs 1.11M），且不具备显式的空间自适应性

## 评分

- 新颖性: ⭐⭐⭐⭐ 将2D内容自适应卷积扩展到3D医学配准是首创，思路清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、与14种方法对比、多维度消融非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，但公式符号偶有不一致
- 价值: ⭐⭐⭐⭐ 空间自适应卷积思路有广泛适用性，参数效率高，实用价值强

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] CARL: A Framework for Equivariant Image Registration](carl_a_framework_for_equivariant_image_registration.md)
- [\[AAAI 2026\] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](../../AAAI2026/medical_imaging/decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)
- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)
- [\[ECCV 2024\] NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration](../../ECCV2024/medical_imaging/textttnephi_neural_deformation_fields_for_approximately_diff.md)
- [\[CVPR 2025\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)

<!-- RELATED:END -->
