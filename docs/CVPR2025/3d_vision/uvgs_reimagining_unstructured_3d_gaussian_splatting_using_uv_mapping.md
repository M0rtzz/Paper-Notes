---
title: >-
  [论文解读] UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping
description: >-
  [CVPR 2025][3D视觉][3D高斯溅射] UVGS通过球面映射将无序的3D高斯溅射（3DGS）转化为结构化的2D UV图表示，并进一步压缩为3通道Super UVGS图像，使预训练的2D图像基础模型（VAE、扩散模型）可以零样本直接应用于3DGS的生成与压缩。 3D高斯溅射（3DGS）在3D对象和场景建模中展现了卓…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "UV映射"
  - "结构化表示"
  - "扩散模型"
  - "3D生成"
---

# UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping

**会议**: CVPR 2025  
**arXiv**: [2502.01846](https://arxiv.org/abs/2502.01846)  
**代码**: [项目页面](https://ivl.cs.brown.edu/uvgs)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, UV映射, 结构化表示, 扩散模型, 3D生成

## 一句话总结

UVGS通过球面映射将无序的3D高斯溅射（3DGS）转化为结构化的2D UV图表示，并进一步压缩为3通道Super UVGS图像，使预训练的2D图像基础模型（VAE、扩散模型）可以零样本直接应用于3DGS的生成与压缩。

## 研究背景与动机

3D高斯溅射（3DGS）在3D对象和场景建模中展现了卓越的质量和效率，但其**无序、离散、置换不变**的特性给生成任务带来了重大挑战：

- **无结构性**：类似点云，3DGS缺乏空间结构，无法直接与基于图像的生成模型（CNN、Transformer）兼容
- **置换不变性**：任意排列的高斯集合代表同一对象，但神经网络不具备置换不变性
- **异构属性**：每个高斯包含14维异构属性（位置3D + 旋转4D + 缩放3D + 颜色3D + 不透明度1D），分布差异大

现有的结构化方案存在问题：
- **体素网格**（GaussianCube）：计算量大、高分辨率困难、体素化导致信息损失
- **三平面表示**：质量与内存的折衷，可能丢失细节
- **直接属性预测**（DiffGS）：仅限类别级生成，学习通用概率函数困难
- **Splatter Image**：3D不感知的投影，多视图一致性差

## 方法详解

### 整体框架

UVGS管线包含三步：(1) 球面映射将3DGS的$N$个高斯投影到$M \times N$的14通道UV图；(2) 多分支映射网络将14通道UVGS压缩为3通道Super UVGS图像；(3) Super UVGS可直接输入预训练的2D基础模型进行压缩或生成。逆过程通过逆映射网络和逆球面映射重建3DGS。

### 关键设计

**设计一：球面映射 — 将无序高斯映射为结构化2D UV图**

- **功能**：解决3DGS的无序性和置换不变性，建立高斯间的局部和全局对应关系
- **核心思路**：将3DGS对象内接于以其几何中心为球心的球面上。对每个高斯计算球面坐标$(\rho_i, \theta_i, \phi_i)$，将方位角$\theta$和极角$\phi$映射到UV图像坐标。每个UV像素存储14维高斯属性。多对一冲突通过Dynamic Selection（保留同一射线上最高不透明度的高斯）解决
- **设计动机**：球面映射自然地为3D点提供了确定性的2D排列，同一对象的任何随机排列都映射到相同的UVGS表示。邻近高斯在3D空间中的邻近关系被保留到2D UV图中，使CNN可以有效学习局部和全局特征

**设计二：Super UVGS + 多分支映射网络 — 异构属性的统一3通道表示**

- **功能**：将14通道异构属性压缩到3通道图像，实现与预训练2D基础模型的零样本兼容
- **核心思路**：前向映射网络包含三个分支——位置分支（处理$\sigma$）、变换分支（处理$r, s$）、外观分支（处理$c, o$），各自提取特征图后拼接输入中央分支（多层Conv + BN + ReLU），最终层用$\tanh$激活输出3通道Super UVGS。逆映射网络以对称结构重建14通道UVGS
- **设计动机**：位置、旋转、缩放、颜色具有截然不同的值分布——相邻高斯的位置和颜色变化平滑，但旋转和缩放可能剧烈变化。分支处理策略避免梯度异常和收敛缓慢，使每个分支专注于各自属性的特有特征

**设计三：零样本基础模型集成 — 直接利用预训练2D模型**

- **功能**：实现3DGS的高效压缩（99.5%+）和无条件/条件生成
- **核心思路**：Super UVGS可直接输入预训练的图像AE/VAE/VQVAE进行重建（无需微调），实现99.5%以上的存储压缩。在VAE潜在空间训练LDM（无条件或文本条件），生成的潜在向量解码为Super UVGS，再通过逆映射重建3DGS对象
- **设计动机**：2D基础模型已在海量数据上训练，拥有强大的图像理解能力。Super UVGS "看起来像图像"且结构化，预训练的VAE可以直接泛化（零样本重建PSNR仅比UVGS降低~0.3dB）

### 损失函数

映射网络训练使用MSE + LPIPS组合损失：$\mathcal{L}_{uvgs} = \mathcal{L}_{mse} + \lambda \cdot \mathcal{L}_{UV-lpips}$，其中LPIPS损失分别对位置、缩放、旋转、颜色四个属性计算，$\lambda$在训练过程中从0增加到10。

## 实验关键数据

### 主实验：重建质量与压缩率（Objaverse Cars/Full）

| 方法 | PSNR (C/F) | LPIPS (C/F) | 压缩率 |
|------|-----------|------------|--------|
| 原始3DGS | 34.6/34.2 | 0.02/0.02 | 0% |
| UVGS (K=1) | 31.3/31.1 | 0.06/0.06 | 53.0% |
| UVGS (K=4) | 34.2/33.2 | 0.02/0.03 | 33.3% |
| Super UVGS (K=1) | 31.2/31.1 | 0.07/0.08 | **89.7%** |
| VAE (K=1) | 30.6/30.9 | 0.07/0.09 | **99.5%** |

### 生成质量对比

| 方法 | 问题 |
|------|------|
| DiffTF | 低质量、低分辨率 |
| Get3D | 3D不一致、伪影多 |
| GaussianCube | 对称性不一致 |
| **UVGS (Ours)** | 高质量、高分辨率、3D一致 |

### 关键发现

- 预训练的图像VAE可以零样本重建Super UVGS，PSNR仅降低~0.3-0.6dB
- Super UVGS本身实现89.7%压缩，经VAE编码后达99.5%压缩，质量损失极小
- 512×512的UV图可存储高达262K个唯一高斯
- 文本条件3DGS生成在复杂几何对象上也产生高质量结果
- 首次展示了3DGS修复（inpainting）实验

## 亮点与洞察

1. **简洁优雅的核心思路**：用球面映射为无序高斯赋予结构，无需任何学习即可完成3D→2D映射
2. **零样本泛化的发现**：预训练VAE直接处理Super UVGS的发现非常意外且有价值
3. **可扩展性**：增加UV分辨率即可容纳更多高斯，用多层UV图处理复杂对象

## 局限与展望

- 球面映射对非凸对象存在多对一冲突，需要多层UV图
- 当前仅测试了物体级3DGS，场景级表示需要更复杂的映射策略
- Super UVGS的3通道压缩不可避免地损失部分信息，尤其在旋转和缩放属性上
- 未来可探索更高效的UV映射方案和更大规模的3DGS生成

## 相关工作与启发

- **GaussianCube**：体素网格结构化3DGS，但计算密集且分辨率受限
- **DiffGS**：连续函数表示3DGS，仅限类别级
- **Splatter Image**：图像式3DGS表示，但缺乏3D感知
- 启发：**找到合适的表示转换是连接不同领域模型的关键**——将3D问题转化为2D问题后，可以直接利用成熟的2D基础设施

## 评分

⭐⭐⭐⭐ — 核心思路简洁且有启发性，球面映射为3DGS赋予结构的方案直观有效。零样本VAE泛化的发现是有趣的经验贡献。对3DGS生成领域提供了新的范式。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping](gausshdr_high_dynamic_range_gaussian_splatting_via_learning_unified_3d_and_2d_lo.md)
- [\[CVPR 2025\] ActiveGAMER: Active GAussian Mapping through Efficient Rendering](activegamer_active_gaussian_mapping_through_efficient_rendering.md)
- [\[CVPR 2026\] OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting](../../CVPR2026/3d_vision/onlinepg_online_open-vocabulary_panoptic_mapping_with_3d_gaussian_splatting.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[CVPR 2025\] PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting](pup_3d-gs_principled_uncertainty_pruning_for_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
