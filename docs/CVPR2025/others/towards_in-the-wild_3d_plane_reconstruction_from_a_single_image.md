---
title: >-
  [论文解读] Towards In-the-Wild 3D Plane Reconstruction from a Single Image
description: >-
  [CVPR 2025][平面重建] ZeroPlane 提出了首个跨域零样本3D平面重建框架，通过构建包含14个数据集/56万标注的大规模平面基准数据集，并设计法向量-偏移解耦的分类-回归范式和像素几何增强嵌入模块，实现了在室内外多样场景中显著优于现有方法的泛化性能。
tags:
  - "CVPR 2025"
  - "平面重建"
  - "零样本泛化"
  - "单图3D重建"
  - "跨域学习"
  - "Transformer"
---

# Towards In-the-Wild 3D Plane Reconstruction from a Single Image

**会议**: CVPR 2025  
**arXiv**: [2506.02493](https://arxiv.org/abs/2506.02493)  
**代码**: [https://github.com/jcliu0428/ZeroPlane](https://github.com/jcliu0428/ZeroPlane)  
**领域**: LLM评测  
**关键词**: 平面重建, 零样本泛化, 单图3D重建, 跨域学习, Transformer

## 一句话总结

ZeroPlane 提出了首个跨域零样本3D平面重建框架，通过构建包含14个数据集/56万标注的大规模平面基准数据集，并设计法向量-偏移解耦的分类-回归范式和像素几何增强嵌入模块，实现了在室内外多样场景中显著优于现有方法的泛化性能。

## 研究背景与动机

**领域现状**：单图3D平面重建是3D视觉的基础任务，应用于AR、定位建图和机器人领域。现有方法（PlaneNet、PlaneRCNN、PlaneTR、PlaneRecTR）主要在单一数据集上训练和测试，集中于室内场景（ScanNet）或户外场景（Synthia），各自在对应域内表现优异。

**现有痛点**：(1) 没有方法尝试过跨域泛化的平面重建——室内外场景的几何尺度、平面分布差异巨大；(2) 高质量的户外平面标注数据稀缺，现有户外数据集缺少密集平面掩码标注；(3) 现有方法在低分辨率（256×192）上训练和测试，限制了重建质量。

**核心矛盾**：室内和户外场景中平面的几何参数分布差异极大（室内偏移通常<5m，户外可达数十甚至上百米），现有方法将法向量和偏移耦合为一个缩放向量 $\mathbf{n}/d$ 进行回归，在混合训练时无法适应如此大的几何范围变化。

**本文目标**：构建统一的跨域3D平面重建系统，能够在室内外多样场景（甚至 in-the-wild 数据）中实现零样本泛化。

**切入角度**：受零样本深度估计（MiDaS、Depth Anything）启发——在混合大规模数据上训练统一模型可显著提升泛化能力。作者将此思路拓展到更具挑战性的平面重建任务。

**核心 idea**：解耦法向量和偏移的表示，对两者分别采用"分类-回归"范式（先分类到最近的聚类中心/exemplar，再回归残差），降低跨域几何参数学习的难度。

## 方法详解

### 整体框架

ZeroPlane 基于 Mask2Former 和 PlaneRecTR 的检测-分割框架。输入单张图像，通过 DINOv2-Base 编码器提取多尺度特征，经 DPT 像素解码器得到多分辨率特征图。可学习的平面查询（query）在 Transformer 解码器中与图像特征交互，输出实例级预测：分割掩码、分类分数、法向量和偏移。额外的像素级深度/法线预测作为辅助任务，其特征通过几何增强模块注入平面查询。

### 关键设计

1. **法向量-偏移解耦 + 分类-回归范式**:

    - 功能：解决跨域混合训练中几何参数回归困难的核心问题
    - 核心思路：将平面参数 $\mathbf{n}/d$ 解耦为法向量 $\mathbf{n}$ 和偏移 $d$ 分别学习。对每个参数采用分类-回归策略：先用 K-Means 从混合训练数据中聚类得到 $K_n=7$ 个法向量 exemplar 和 $K_d=20$ 个偏移 exemplar（偏移按20m阈值分成室内/户外两组各聚10个）。MLP 分类头预测属于哪个 exemplar，回归头预测相对残差。最终预测：$\mathbf{n} = \hat{\mathbf{n}}^{(i)} + \mathbf{r_n}^{(i)}$, $d = \hat{d}^{(j)} + r_d^{(j)}$
    - 设计动机：直接回归需要网络同时处理差异巨大的几何尺度（室内<5m vs 户外>50m），非常困难。分类先将问题粗定位到合适范围，回归只需精调残差，大幅降低学习难度

2. **像素几何增强平面嵌入模块**:

    - 功能：将底层像素级几何线索（深度、法线）注入平面查询，增强实例级预测
    - 核心思路：编码器后附加两个 CNN 模块分别预测像素深度图 $\mathbf{D}$ 和法线图 $\mathbf{N}$（作为辅助任务）。将它们投影到嵌入空间 $\mathbf{F_D}$, $\mathbf{F_N}$，然后平面查询通过交叉注意力与几何特征交互：$\mathbf{X_D} = \text{Attn}(\mathbf{Q}, \mathbf{F_D})$, $\mathbf{X_N} = \text{Attn}(\mathbf{Q}, \mathbf{F_N})$。最终嵌入 $\mathbf{X} = \mathbf{X_F} + \mathbf{X_D} + \mathbf{X_N}$
    - 设计动机：仅将像素深度/法线作为辅助任务训练时提升有限，因为平面查询无法直接利用这些底层几何信息。注意力机制让查询能从像素级几何预测中发现细粒度上下文线索（如平面边界）

3. **大规模跨域平面基准数据集**:

    - 功能：提供覆盖多样室内外环境的高分辨率密集平面标注
    - 核心思路：整合14个数据集（7室内+4户外），共56万高分辨率（640×480）标注。对缺少语义标注的数据集，使用 Mask2Former 获取全景分割结果作为伪真值，然后在每个物体/区域的反投影点云上拟合平面。对户外场景借助合成数据集（精确深度图）或立体相机数据生成平面标注
    - 设计动机：现有平面数据集仅覆盖室内且低分辨率，无法支撑跨域训练。高分辨率标注是高质量平面重建的前提

### 损失函数 / 训练策略

- **总损失**：$L = \lambda_c L_c + \lambda_m L_m + \lambda_{n_c} L_{n_c} + \lambda_{n_r} L_{n_r} + \lambda_{d_c} L_{d_c} + \lambda_{d_r} L_{d_r} + \lambda_{p_d} L_{p_d} + \lambda_{p_n} L_{p_n}$
- 分类用交叉熵损失，回归用 L1 损失，掩码用交叉熵+Dice 联合损失
- 使用二部图匹配策略（Hungarian matching）匹配预测和真值
- **训练**：AdamW 优化器，lr=1e-4，batch size=16，50K步，在40K/47K步各衰减10倍

## 实验关键数据

### 主实验 — 零样本评测

| 方法 | NYUv2 Recall@0.1m | NYUv2 Recall@5° | 7-Scenes Recall@0.1m |
|------|-------------------|-----------------|---------------------|
| PlaneRecTR (S) | 10.13 | 16.42 | - |
| PlaneRecTR (M) | 14.29 | 24.97 | 10.97 |
| **ZeroPlane-DINO-B (M)** | **17.86** | **37.29** | **17.19** |
| ZeroPlane-Dust3R (M) | 21.20 | 38.32 | 19.14 |

### 消融实验

| 设计 | NYUv2 Recall@0.1m |
|------|-------------------|
| 耦合表示（n/d 直接回归） | ~12 |
| 解耦 + 直接回归 | ~14 |
| 解耦 + 分类-回归 | **17.86** |
| 无几何增强模块 | 降低约1-2% |
| 辅助任务但无注意力注入 | 提升有限 |

### 关键发现

- 混合训练（M）显著优于单数据集训练（S），证实了跨域数据的互补价值
- 解耦法向量/偏移 + 分类-回归相比直接回归提升巨大，特别在深度重建精度上
- 几何增强嵌入模块通过注意力机制注入有效，仅作辅助任务则帮助有限
- 更强的编码器（DINOv2-L、Dust3R）进一步提升性能，但基础版已大幅领先
- 在户外零样本数据上优势更加明显，体现了跨域泛化能力

## 亮点与洞察

- 首次将零样本泛化思想引入3D平面重建领域，开辟了新的研究方向
- "分类-回归"范式巧妙解决了跨域几何参数分布差异大的难题，具有通用性
- 构建的大规模跨域基准数据集本身就是重要贡献，可支撑后续研究
- DINOv2 编码器的引入显著提升了特征的跨域鲁棒性

## 局限与展望

- 户外数据主要来自合成数据集或立体相机，真实户外场景的标注仍然困难
- 偏移 exemplar 的室内/户外分割阈值（20m）是人工设定的，可能不够灵活
- 未处理非平面区域的重建（如曲面），仅关注平面
- 可考虑结合深度基础模型（如 Depth Anything）进一步提升几何估计精度

## 相关工作与启发

- **PlaneRecTR**：之前的 SOTA Transformer 平面检测器，ZeroPlane 在此基础上引入跨域训练策略
- **MiDaS/Depth Anything**：零样本深度估计的成功直接启发了本文
- **Mask2Former**：提供了整体架构和二部图匹配策略
- **Dust3R**：作为替代编码器进一步提升了几何感知能力

## 评分

- **新颖性**: 8/10 — 首次提出跨域零样本平面重建，分类-回归范式设计精巧
- **实验充分度**: 9/10 — 14个数据集、多种编码器、全面消融，零样本评测覆盖室内外
- **写作质量**: 8/10 — 问题分析透彻，方法动机清晰
- **价值**: 8/10 — 数据集+方法双贡献，开辟3D平面重建的新研究方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Image Reconstruction from Readout-Multiplexed Single-Photon Detector Arrays](image_reconstruction_from_readout-multiplexed_single-photon_detector_arrays.md)
- [\[ECCV 2024\] Intrinsic Single-Image HDR Reconstruction](../../ECCV2024/others/intrinsic_single-image_hdr_reconstruction.md)
- [\[CVPR 2025\] MagicArticulate: Make Your 3D Models Articulation-Ready](magicarticulate_make_your_3d_models_articulation-ready.md)
- [\[NeurIPS 2025\] HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild](../../NeurIPS2025/others/houselayout3d_a_benchmark_and_training-free_baseline_for_3d_layout_estimation_in.md)
- [\[CVPR 2025\] Regor: Progressive Correspondence Regenerator for Robust 3D Registration](progressive_correspondence_regenerator_for_robust_3d_registration.md)

</div>

<!-- RELATED:END -->
