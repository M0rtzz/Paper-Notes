---
title: >-
  [论文解读] CrossSDF: 3D Reconstruction of Thin Structures From Cross-Sections
description: >-
  [CVPR 2025][医学图像][SDF] 提出 CrossSDF，通过从 2D 截面符号距离场重建 3D SDF，结合混合编码（哈希网格 + 随机傅里叶特征）和对称差损失，首次实现对薄管状结构（如血管）的精确重建。
tags:
  - "CVPR 2025"
  - "医学图像"
  - "SDF"
  - "cross-section reconstruction"
  - "thin structures"
  - "neural implicit fields"
  - "hash encoding"
---

# CrossSDF: 3D Reconstruction of Thin Structures From Cross-Sections

**会议**: CVPR 2025  
**arXiv**: [2412.04120](https://arxiv.org/abs/2412.04120)  
**代码**: [项目页面](https://iamsalvatore.io/cross_sdf)  
**机构**: University of Edinburgh / University of British Columbia
**领域**: 医学影像 / 3D 重建  
**关键词**: SDF, cross-section reconstruction, thin structures, neural implicit fields, hash encoding, medical imaging

## 一句话总结
提出 CrossSDF，通过从 2D 截面符号距离场重建 3D SDF，结合混合编码（哈希网格 + 随机傅里叶特征）和对称差损失，首次实现对薄管状结构（如血管）的精确重建。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**：从平面截面数据重建 3D 结构是医学影像、制造业和地形学中的经典问题。CT/MRI 扫描产生稀疏的 2D 截面数据，需从中重建完整的 3D 形状。

**现有痛点**：

### 现有痛点

**现有痛点**：点云重建方法**：截面之间数据稀疏，常导致失败

### 核心矛盾

**核心矛盾**：传统插值方法**（OReX 等）：产生严重的"棱状"伪影（laddering artifacts），在平行截面间出现阶梯状结构

### 解决思路

**解决思路**：Indicator function 方法**：不连续性导致法线和光滑性伪影，且退化为不稳定的二分类问题

### 补充说明

**补充说明**：所有现有方法在重建**薄结构**（如血管、神经组织）时表现糟糕，要么过度平滑要么断裂

**核心矛盾**：2D SDF 和期望的 3D SDF 在一般情况下不一致——直接拟合 2D SDF 会迫使 3D 表面法线与截面平行，产生"阶梯效应"。

**切入角度**：仅利用 2D SDF 的零集信息（内/外分类），通过对称差损失引导 3D SDF 学习，释放模型从"必须是 2D SDF"的约束。

**核心 idea**：对称差损失（释放平面内约束） + 混合编码（哈希+RFF，消除网格伪影） + 自适应采样 = 精确薄结构重建。

## 方法详解

### 输入输出
- **输入**：$n$ 个任意方向的 2D 平面，每个平面上有闭合轮廓（截面与目标几何体的交线）
- **输出**：神经 3D SDF $f(\mathbf{x}; \boldsymbol{\theta})$，零集定义重建几何体

### 关键设计

1. **自适应采样（Adaptive Sampling）**

    - 功能：确保薄结构内部被充分采样
    - 问题：均匀采样会欠采样小截面积结构；OReX 的固定半径采样对薄结构也不适用
    - 解决：直接在每个轮廓内部采样至阈值数量，利用 2D 截面的内/外标注进行精确分区
    - 效果：小截面积轮廓不会被忽略

2. **混合编码（Hybrid Encoding）**

    - 功能：结合哈希网格编码 $\gamma_{\text{hash}}$ 和随机傅里叶特征编码 $\gamma_{\text{RF}}$
    - 哈希网格：捕捉细节，但在网格边界产生插值伪影
    - 随机傅里叶特征：鼓励光滑表示，减少网格伪影
    - 融合方式：$\mathbf{z}_{\text{comb}} = \mathbf{z}_{\text{hash}} + \alpha \cdot \mathbf{z}_{\text{RF}}$，$\alpha$ 为平衡因子
    - 最终输入：$\mathbf{z}_{\text{final}} = [\mathbf{z}_{\text{comb}} | \mathbf{x}]$，拼接原始 3D 坐标

3. **对称差损失（Symmetric Difference Loss）**

    - 功能：仅在预测和目标内/外分类不一致的区域驱动优化
    - 核心洞察：截面上 2D SDF 包含的关于 3D 几何的唯一信息是内/外分类
    - 两个集合：$\Omega_{\text{on}} = \{\mathbf{x} | f_{2D}(\mathbf{x}) = 0\}$（零集采样点） 和对称差区域
    - 优势：当预测和目标轮廓完全重合时损失为零，释放网络自由学习 3D SDF
    - 效果：消除直接拟合 2D SDF 导致的阶梯伪影

4. **Eikonal 正则化**

    - 约束 $|\nabla f| = 1$，保证 SDF 性质
    - 对稀疏输入特别有效

### 网络架构
- 两个通道：哈希网格通道和 RFF 通道，各有一层隐藏层 MLP
- 融合后接 SDF MLP 输出最终距离值

## 实验关键数据

### 与现有方法的定量比较（合成数据）

| 方法 | 适用场景 | 薄结构保持 | 拓扑连续性 | 表面平滑性 |
|------|---------|-----------|-----------|-----------|
| OReX | 任意截面 | 差 | 差 | 差（棱状伪影） |
| Points2Surf | 点云 | 中 | 中 | 中 |
| POCO | 点云 | 中 | 中 | 好 |
| DeepSDF | 潜空间 | 差 | 好 | 好（过度平滑） |
| **CrossSDF** | **任意截面** | **好** | **好** | **好** |

### 消融实验

| 配置 | 薄结构质量 | 表面光滑度 | 阶梯伪影 |
|------|-----------|-----------|---------|
| 仅哈希编码 | 好 | 差（网格伪影） | 少 |
| 仅 RFF 编码 | 差（过度平滑） | 好 | 少 |
| 直接拟合 2D SDF | 好 | 差 | 严重 |
| 无自适应采样 | 差（丢失小结构） | — | — |
| **完整方法** | **好** | **好** | **无** |

### 关键发现
- 对称差损失是消除阶梯伪影的关键
- 混合编码比单一编码在细节保持和光滑性上取得更好平衡
- 自适应采样对小截面积结构至关重要

## 技术细节补充

### 网络参数
- 哈希网格分辨率：多尺度，从 $16^3$ 到 $2048^3$
- RFF 编码维度 $d$：保证与哈希编码输出维度匹配
- 缩放因子 $\alpha$：初始化时让两路编码幅度大致相等
- SDF MLP：标准 8 层全连接，带 skip connection

### Eikonal 正则化
$$\mathcal{L}_{eik} = \mathbb{E}_{\mathbf{x}}[(|\nabla f(\mathbf{x})| - 1)^2]$$
在截面之间的 3D 空间随机采样点上施加，提供平滑性先验。

## 亮点与洞察
- **对称差损失**的设计精妙：将几何约束从"学习 2D SDF"放松到"仅学习内/外分类一致"，既保留了充分的监督信号又释放了自由度
- **混合编码**解法简洁：一个捕细节、一个保光滑，互补性强
- 医学影像中薄血管重建是真实刚需，方法的实用价值高
- 支持任意方向非平行截面，比仅处理平行截面的方法通用性更强
- 方法核心是损失函数设计，网络架构无需特殊修改，可迁移到其他隐式场表示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Thin-Shell-SfT: Fine-Grained Monocular Non-Rigid 3D Surface Tracking with Neural Deformation Fields](thin-shell-sft_fine-grained_monocular_non-rigid_3d_surface_tracking_with_neural_.md)
- [\[ICML 2026\] Foundation VAEs for 3D CT Reconstruction, Augmentation, and Generation](../../ICML2026/medical_imaging/foundation_vaes_for_3d_ct_reconstruction_augmentation_and_generation.md)
- [\[NeurIPS 2025\] EEGReXferNet: A Lightweight Gen-AI Framework for EEG Subspace Reconstruction via Cross-Subject Transfer Learning and Channel-Aware Embedding](../../NeurIPS2025/medical_imaging/eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)
- [\[CVPR 2025\] vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation](vesselfm_a_foundation_model_for_universal_3d_blood_vessel_segmentation.md)
- [\[CVPR 2025\] VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging](vista3d_a_unified_segmentation_foundation_model_for_3d_medical_imaging.md)

</div>

<!-- RELATED:END -->
