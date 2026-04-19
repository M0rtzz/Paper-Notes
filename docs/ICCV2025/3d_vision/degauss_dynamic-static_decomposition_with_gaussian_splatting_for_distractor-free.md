---
title: >-
  [论文解读] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction
description: >-
  [ICCV 2025][3D视觉][3D高斯泼溅] 提出 DeGauss，一种基于解耦的动态-静态高斯泼溅的自监督框架，通过前景动态高斯和背景静态高斯的概率掩码组合，实现从随意捕获的图像集到高度动态的自我中心视频的广泛场景下的无干扰 3D 重建。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D高斯泼溅
  - 动态场景重建
  - 动静态分解
  - 无干扰重建
  - 自监督
---

# DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction

**会议**: ICCV 2025  
**arXiv**: [2503.13176](https://arxiv.org/abs/2503.13176)  
**代码**: [GitHub](https://batfacewayne.github.io/DeGauss.io/)  
**领域**: 3D视觉  
**关键词**: 3D高斯泼溅, 动态场景重建, 动静态分解, 无干扰重建, 自监督

## 一句话总结

提出 DeGauss，一种基于解耦的动态-静态高斯泼溅的自监督框架，通过前景动态高斯和背景静态高斯的概率掩码组合，实现从随意捕获的图像集到高度动态的自我中心视频的广泛场景下的无干扰 3D 重建。

## 研究背景与动机

从真实世界捕获的视频/图像中重建干净的静态 3D 场景是计算机视觉的核心挑战。现有方法面临的困境包括：

**静态假设的局限**：NeRF 和 3DGS 在受控的静态场景下表现优秀，但在包含动态元素（行人、移动物体）的日常捕获中，动态内容被建模为视角相关的伪影（"floaters"），严重降低重建质量。

**自我中心视频的独特挑战**：头戴式相机录制的视频包含剧烈相机运动、运动模糊、频繁的人-物交互，以及操作者自身的身体，使得动静态分离极为困难。

**现有去干扰方法的不足**：
   - **基于残差的方法**（NeRF-on-the-go, WildGaussians, SpotlessSplats）利用重建损失残差和语义特征识别动态区域，但对初始化敏感，在类内动静态歧义和场景形变下泛化能力差
   - **NeRF 基的分解方法**（Nerf-W, NeuralDiff, D2NeRF）通过动态分支建模，但训练慢，且在稀疏输入下动态分支可能分割不完全，在高动态视频中又可能过度分割

**关键洞察**：动态高斯擅长时序建模但容易过拟合训练视角、对新视角泛化差；静态高斯虽然无法处理运动但跨视角表示更稳定。将两者优势互补，通过解耦的前景-背景设计实现协调优化。

## 方法详解

### 整体框架

DeGauss 用两组高斯分别建模：前景动态高斯 $\mathcal{G}_f$ 配合时空变形模块处理动态元素，背景静态高斯 $\mathcal{G}_b$ 处理静态内容。通过前景高斯光栅化得到的概率掩码控制两者的组合，使得两个分支可以独立优化又协调互补。

### 关键设计

1. **前景可变形高斯 (Foreground Deformable Gaussian)**：在标准高斯属性之外增加了三个特征 $\{m_f, m_b, b\}$，分别对应前景概率、背景概率和亮度控制。时空变形模块采用 HexPlane 编码器 $\mathcal{H}$ 提取时空特征 $\mathbf{f_d} = \mathcal{H}(\mathcal{G}_f, t)$，再通过多头 MLP 解码器 $\mathcal{D}$ 预测所有属性的变形量：
   $$\mathcal{G}'_f = \Delta\mathcal{G}_f + \mathcal{G}_f$$
   每个属性（位置、旋转、缩放、不透明度、颜色、掩码、亮度）各用独立的 MLP 预测。

2. **概率组合掩码光栅化 (Probabilistic Composition Mask)**：通过可微渲染从前景高斯的 $m_f', m_b'$ 属性得到原始前景概率 $\mathbf{M}_f$ 和背景概率 $\mathbf{M}_b$，再归一化为概率掩码：
   $$\mathbf{P}_f = \frac{\mathbf{M}_f}{\mathbf{M}_f + \mathbf{M}_b + \epsilon}, \quad \mathbf{P}_b = \frac{\mathbf{M}_b}{\mathbf{M}_f + \mathbf{M}_b + \epsilon}$$
   这种概率形式天然抑制中间值（接近0.5），推动预测向 0 或 1 两极化，产生清晰的动静态分解。

3. **背景亮度控制 (Background Brightness Control)**：日常捕获中光照变化显著，前景高斯的高表达力容易过度分割光照变化区域为动态。引入亮度控制掩码 $\mathbf{B}$ 增强背景分支对非朗伯效应的建模能力，通过分段线性激活函数将亮度映射到 $[0.5, 1.25+]$ 范围，防止将暗色动态物体误作亮度控制，同时支持过曝光建模：
   $$\hat{\mathbf{C}}_b = \hat{\mathbf{B}} * \mathbf{C}_b$$

4. **无监督场景分解**：最终组合渲染 $\hat{\mathbf{C}} = \mathbf{P}_f * \mathbf{C}_f + \mathbf{P}_b * \hat{\mathbf{B}} * \mathbf{C}_b$。与 NeRF 基方法在渲染过程中沿光线混合密度不同，DeGauss 在渲染后独立合成前景和背景，避免了早期光线截断导致的局部最优和静态场景细节缺失。

5. **部分不透明度重置 (Partial Opacity Reset)**：对 SpotlessSplats 等方法，周期性完全重置不透明度会导致训练不稳定。得益于前景-背景解耦设计的额外稳定性，DeGauss 可以进行 50% 比例的周期性部分不透明度重置，有效控制高斯密度、处理 floaters 和避免局部最优。

### 损失函数 / 训练策略

损失分为两部分，梯度分离用于控制高斯自适应密度化过程：
- **主损失 $\mathcal{L}_\text{main}$**（用于密度化）= $\mathcal{L}_1 + \mathcal{L}_\text{diversity} + \mathcal{L}_\text{reg} + \mathcal{L}_\text{depth} + \mathcal{L}_f + \mathcal{L}_b$
- **工具损失 $\mathcal{L}_\text{uti}$**（不用于密度化）= $\mathcal{L}_\text{SSIM} + \mathcal{L}_\text{entropy} + \mathcal{L}_\text{brightness} + \mathcal{L}_\text{scale}$

训练分两阶段：粗训练阶段（1K迭代，禁用变形模块），精细训练阶段（20K-120K迭代，端到端联合优化）。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | SpotlessSplats | 提升 |
|--------|------|------|----------------|------|
| NeRF-on-the-go (Mean) | PSNR↑ | 23.91 | 23.42 | +0.49 |
| NeRF-on-the-go (Mean) | SSIM↑ | 0.819 | 0.813 | +0.006 |
| NeRF-on-the-go (Mean) | LPIPS↓ | 0.113 | 0.145 | -0.032 |
| Neu3D (Mean) | LPIPS↓ | 0.047 | - | SOTA |
| Neu3D (Mean) | PSNR↑ | 31.52 | - | 与4DGS相当 |

在 NeRF-on-the-go 全部 6 个场景上 LPIPS 均优于 SpotlessSplats。在 Neu3D 数据集上 LPIPS 显著优于所有基线方法（0.047 vs 4DGS 的 0.058）。

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| 完整模型 | 23.91 | 0.819 | 0.113 | 全部组件 |
| w/o 亮度控制 | 23.54 | 0.814 | 0.118 | 光照歧义导致过度分割 |
| w/o 部分重置 | 23.56 | 0.814 | 0.117 | floaters残留 |
| w/o 深度损失 | 23.68 | 0.816 | 0.113 | 几何约束缺失 |
| w/o 背景掩码元素 | 23.83 | 0.817 | 0.115 | 分解不够干净 |

### 关键发现

- 解耦的后渲染组合方式比 NeRF 的沿光线混合（渲染中组合）更鲁棒，梯度流更完整
- 亮度控制掩码对解决光照变化引起的动静态歧义至关重要
- 方法从随意图像集到高动态自我中心长视频（2800-5000帧）都能稳定工作
- 动态前景建模能力使方法同时具备高质量的动态场景表示能力

## 亮点与洞察

- 设计极为简洁优雅：解耦的前景-背景高斯 + 概率掩码组合 + 亮度控制，三个简单设计组合实现了强大的泛化性
- 概率掩码通过归一化天然趋向二值化，不需要额外正则化
- 前景高斯中允许不贡献颜色但参与掩码/亮度计算的"工具高斯"，增加了表达力
- 从图像集到长视频的广泛评估展示了方法的通用性

## 局限与展望

- HyperNeRF 数据集的相机位姿不准确，未做定量评估
- 长视频（数千帧）训练迭代需 120K，计算开销较大
- 对极端运动模糊（如快速手部运动）的处理能力未充分验证
- 未探索利用语义先验（如 SAM/DINO）进一步提升分解质量的可能性

## 相关工作与启发

- SpotlessSplats 的基于扩散特征聚类的方法受初始化限制，这揭示了显式分解方法的必要性
- 4DGS 的 HexPlane 编码器被成功复用为前景变形的时空编码器
- 解耦设计思路可推广至多体动态重建、语义分割等下游任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 解耦前景-背景高斯的概率组合设计新颖实用
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖图像集、自我中心视频、固定多视角等多种数据源
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，实验丰富
- 价值: ⭐⭐⭐⭐ 建立了泛化性很强的无干扰重建baseline

<!-- RELATED:START -->

## 相关论文

- [BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curv.md)
- [Proactive Scene Decomposition and Reconstruction](proactive_scene_decomposition_and_reconstruction.md)
- [No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)
- [StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)

<!-- RELATED:END -->
