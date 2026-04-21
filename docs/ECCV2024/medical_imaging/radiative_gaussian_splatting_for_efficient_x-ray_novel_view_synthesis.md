---
title: >-
  [论文解读] Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis
description: >-
  [ECCV 2024][医学图像][3D Gaussian Splatting] 提出 X-Gaussian，首个将 3D 高斯泼溅（3DGS）应用于 X 射线新视角合成的框架，通过设计辐射高斯点云模型（替代球谐函数）和角度位姿长方体均匀初始化策略（替代 SfM），在性能上超越 SOTA NeRF 方法 6.5 dB 的同时，实现 73× 推理加速和仅 15% 的训练时间。
tags:
  - ECCV 2024
  - 医学图像
  - 3D Gaussian Splatting
  - X-ray imaging
  - novel view synthesis
  - CT reconstruction
  - differentiable rendering
---

# Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis

**会议**: ECCV 2024  
**arXiv**: [2403.04116](https://arxiv.org/abs/2403.04116)  
**代码**: [caiyuanhao1998/X-Gaussian](https://github.com/caiyuanhao1998/X-Gaussian)  
**领域**: 医学图像  
**关键词**: 3D Gaussian Splatting, X-ray imaging, novel view synthesis, CT reconstruction, differentiable rendering

## 一句话总结

提出 X-Gaussian，首个将 3D 高斯泼溅（3DGS）应用于 X 射线新视角合成的框架，通过设计辐射高斯点云模型（替代球谐函数）和角度位姿长方体均匀初始化策略（替代 SfM），在性能上超越 SOTA NeRF 方法 6.5 dB 的同时，实现 73× 推理加速和仅 15% 的训练时间。

## 研究背景与动机

**领域现状**: X 射线新视角合成（NVS）旨在从已有投影中生成未扫描角度的 X 射线图像，对减少患者辐射暴露和辅助 CT 重建具有重要临床价值。当前方法主要基于 NeRF，沿射线采样大量 3D 点并逐一计算，训练和推理速度极慢。

**现有痛点**: 即使最高效的 NeRF 方法 NAF 也需要超过 1 小时训练，推理速度仅 2 fps，远不能满足临床实时性需求。而 3DGS 在自然光成像中展现了远胜 NeRF 的效率，但尚未有人将其应用于 X 射线领域。

**核心矛盾**: 3DGS 的两个核心组件——球谐函数（SH）和 Structure-from-Motion（SfM）初始化——都不适用于 X 射线成像。SH 用于建模视角相关的各向异性颜色，但 X 射线辐射强度是各向同性的；SfM 依赖特征检测与匹配，但 X 射线图像是灰度低对比度的且存在透射重叠，SfM 精度严重下降。

**本文目标** 如何将 3DGS 的高效渲染优势迁移到 X 射线成像领域，同时解决物理成像模型不匹配和初始化策略不适用的问题。

**切入角度**: 从 X 射线成像的物理本质出发——穿透衰减而非表面反射——重新设计高斯点云的辐射强度表示函数和光栅化流程，并利用 X 射线扫描仪参数直接计算相机参数和初始化点云。

**核心 idea**: 基于 X 射线各向同性穿透成像的物理特性，设计无视角依赖的辐射高斯点云模型和免 SfM 的长方体均匀初始化策略，从根本上实现 3DGS 在 X 射线 NVS 中的高效适配。

## 方法详解

### 整体框架

X-Gaussian 的流程分为三步：(1) 使用 ACUI 策略从 X 射线扫描仪参数计算内外参矩阵，并在包围被扫描物体的长方体内均匀采样初始点云；(2) 辐射高斯点云模型学习每个 3D 点的辐射特征；(3) 通过可微辐射光栅化（DRR）渲染出 X 射线投影图像，与真值计算损失进行优化。

### 关键设计

1. **辐射高斯点云模型（Radiative Gaussian Point Cloud Model）**: 核心创新是用辐射强度响应函数（RIRF）替代原始 3DGS 中的球谐函数（SH）。每个高斯点云学习一个特征向量 $\mathbf{f} \in \mathbb{R}^{N_f}$，辐射强度通过以下函数计算：

   $$\mathbf{i}(\mathbf{f}) = \text{RIRF}(\mathbf{f}) = \text{Sigmoid}(\boldsymbol{\lambda} \odot \mathbf{f})$$

   其中 $\boldsymbol{\lambda} \in \mathbb{R}^{N_f}$ 是常数权重向量。关键在于该函数**不包含视角方向 $\mathbf{d}$**，符合 X 射线的各向同性特性。与 SH 对比：SH 建模的是视角相关的各向异性颜色 $\mathbf{c}(\mathbf{d}, \mathbf{k}) = \sum_{l,m} k_l^m Y_l^m(\theta, \phi)$，而 RIRF 完全排除了视角依赖性。

   设计动机：自然光成像依赖表面反射，同一点从不同角度看颜色不同（各向异性）；X 射线成像依赖穿透衰减，辐射密度是物质固有属性，与观察方向无关（各向同性）。

2. **可微辐射光栅化（Differentiable Radiative Rasterization, DRR）**: 给定视角，将 3D 高斯投影到 2D 探测器平面。投影中心坐标通过外参和内参矩阵变换：

   $$\tilde{\mathbf{t}}_i = \mathbf{M}_{ext} \tilde{\boldsymbol{\mu}}_i, \quad \tilde{\mathbf{u}}_i = \mathbf{M}_{int} \tilde{\mathbf{t}}_i$$

   3D 协方差矩阵通过雅可比矩阵变换到相机坐标系：$\boldsymbol{\Sigma}_i' = \mathbf{J}_i \mathbf{W}_i \boldsymbol{\Sigma}_i \mathbf{W}_i^\top \mathbf{J}_i^\top$。像素 $p$ 的强度通过 alpha blending 计算：

   $$\mathbf{I}(p) = \sum_{j \in \mathcal{N}} \mathbf{i}_j \sigma_j \prod_{k=1}^{j-1}(1-\sigma_k), \quad \sigma_j = \alpha_j P(\mathbf{x}_j | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)$$

   由于去除了与视角方向相关的计算，DRR 的前向和反向过程均比 RGB 光栅化更快。整个 DRR 使用 CUDA 实现以实现高效 GPU 并行。

3. **角度位姿长方体均匀初始化（ACUI）**: 完全绕过 SfM，直接利用圆锥束 X 射线扫描仪的已知参数计算相机矩阵：

   $$\mathbf{M}_{ext} = \begin{bmatrix} -\sin\phi & \cos\phi & 0 & 0 \\ 0 & 0 & -1 & 0 \\ -\cos\phi & -\sin\phi & 0 & L_{SO} \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

   其中 $L_{SO}$ 是光源到物体的距离，$\phi$ 是方位角。点云初始化通过在包围物体的长方体内以间隔 $d$ 均匀采样：

   $$\mathcal{P} = \left\{ \left(\frac{n_1 S_1 d}{M_1}, \frac{n_2 S_2 d}{M_2}, \frac{n_3 S_3 d}{M_3}\right) \right\}$$

   设计动机：X 射线图像灰度低对比度且具有透射重叠特性，导致 SfM 的特征检测和匹配精度严重下降；而圆锥束扫描的几何参数完全已知，可以直接计算相机内外参，省去 SfM 的耗时运算。

### 损失函数 / 训练策略

训练目标为 $\mathcal{L}_1$ 损失和 SSIM 损失的加权和：

$$\mathcal{L} = (1-\gamma)\mathcal{L}_1 + \gamma\mathcal{L}_{\text{SSIM}}$$

其中 $\gamma = 0.2$。使用 Adam 优化器（$\beta_1=0.9$, $\beta_2=0.999$），训练 $2 \times 10^4$ 次迭代。点云位置学习率从 $1.9 \times 10^{-4}$ 指数衰减到 $1.9 \times 10^{-6}$。使用原始 3DGS 的自适应密度控制策略动态调整点云数量 $N_p$。

## 实验关键数据

### 主实验：新视角合成性能比较

| 方法 | 推理速度 | 训练时间 | 平均PSNR(dB) | 平均SSIM |
|------|---------|---------|-------------|---------|
| InTomo | 0.62 fps | 125 min | 30.187 | 0.9611 |
| NeRF | 0.14 fps | 313 min | 30.289 | 0.9617 |
| TensoRF | 0.77 fps | 178 min | 30.477 | 0.9194 |
| NeAT | 1.78 fps | 69 min | 34.201 | 0.9366 |
| NAF | 2.01 fps | 63 min | 36.942 | 0.9627 |
| **X-Gaussian** | **148 fps** | **9 min** | **43.404** | **0.9993** |

X-Gaussian 比最佳 NeRF 方法 NAF **高 6.5 dB**，推理速度是其 **73 倍**，训练时间仅为其 **15%**。

### 消融实验：各组件贡献分析

| 配置 | PSNR(dB) | SSIM | 训练时间(s) | 推理速度(fps) |
|------|---------|------|-----------|-------------|
| 原始 3DGS (baseline) | 37.21 | 0.9813 | 1898 | 64 |
| + ACUI (替换SfM) | 38.87 (+1.66) | 0.9871 | 1172 (↓34%) | 72 |
| + DRR (替换RGB光栅化) | **43.40 (+4.53)** | **0.9993** | **538 (↓54%)** | **148 (2.1×)** |

初始化策略对比：

| 策略 | PSNR(dB) | 训练时间(s) | 推理速度(fps) |
|------|---------|-----------|-------------|
| Random | 41.33 | 601 | 112 |
| Spherical | 42.84 | 575 | 136 |
| FDK | **43.47** | 1394 | 93 |
| **Cubic (ACUI)** | 43.40 | **538** | **148** |

### 关键发现

1. **辐射高斯点云模型贡献最大**：DRR 替换 RGB 光栅化后 PSNR 提升 4.53 dB（从 38.87→43.40），同时训练时间减半，推理速度翻倍，证明了针对 X 射线各向同性特性的建模至关重要。
2. **ACUI 高效且有效**：虽然 FDK 初始化精度略高（+0.07 dB），但其训练时间是 ACUI 的 2.59 倍且推理更慢。ACUI 取得了性能与效率的最佳平衡。
3. **稀疏视角 CT 重建有显著实用价值**：使用 X-Gaussian 生成新视角投影辅助 ASD-POCS 重建，PSNR 提升 13.53 dB（从 17.03→30.56），远超 NAF 的 10.88 dB 提升。
4. **收敛速度远快于原始 3DGS 和 NAF**：在训练 1000 次迭代时，X-Gaussian 的点云已基本形成物体轮廓；训练 60 秒时渲染质量已远超 NAF 训练 180 秒的结果。
5. **特征维度 $N_f=16$ 是最优性价比**：$N_f=16$ 与 $N_f=32$ 性能差距仅 0.013 dB，但前者训练更快推理更快。

## 亮点与洞察

- **物理驱动的模型设计**：从 X 射线成像的物理本质（穿透衰减、各向同性）出发重新设计 3DGS 的每个组件，不是简单的"减去 SH"而是重新思考了辐射强度的表示方式，RIRF 的设计简洁高效。
- **极致的效率提升**：148 fps 的推理速度使实时 X 射线视角合成成为可能，9 分钟的训练时间大幅降低了临床应用门槛。
- **端到端的系统设计**：从初始化到点云模型再到光栅化，每个环节都针对 X 射线特性进行了定制，形成了一个完整的技术栈。
- **稀疏视角 CT 重建的实际应用**展示了该方法超越纯学术研究的临床价值——可以用更少的 X 射线扫描实现高质量 CT 重建，直接减少患者辐射剂量。

## 局限与展望

1. **实现复杂度高**：核心 CUDA 实现调试困难，可解释性较低，不如 PyTorch 实现易于复现和修改。
2. **仅适用于圆锥束扫描场景**：ACUI 策略依赖特定的扫描几何，对非标准扫描配置（如螺旋 CT、不等角间隔）需要额外适配。
3. **未考虑多能量 X 射线**：实际扫描中 X 射线是多能量的，不同能量的衰减特性不同，当前模型假设单一衰减系数可能在复杂场景下不够精确。
4. **缺少真实临床数据验证**：实验使用 TIGRE 工具箱仿真生成投影，未在真实 X 射线设备采集的数据上验证。
5. **可扩展至其他穿透成像模态**：如超声、近红外光谱等，但需要针对各模态的物理特性重新设计辐射模型。

## 相关工作与启发

- **3DGS** [Kerbl et al.]: RGB 场景的高斯泼溅表示与光栅化，是本文的基础框架，本文证明了 3DGS 的设计理念可以迁移到非自然光成像领域。
- **NAF** [Zha et al.]: 基于 Instant-NGP 的 X 射线 NeRF 方法，是此前 SOTA，本文将其作为主要对比基线。
- **NeAT** [Rückert et al.]: 神经自适应层析成像，在部分场景上优于 NAF，但速度较慢。
- **TensoRF** [Chen et al.]: 张量辐射场分解，RGB NeRF 中的效率优化方法，但直接应用于 X 射线性能欠佳。
- 本文的核心启发：当将通用方法（3DGS）迁移到特定领域（X 射线）时，需要从**物理第一性原理**出发重新审视每个组件的适用性，而非简单套用。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次将 3DGS 应用于 X 射线 NVS，RIRF 和 ACUI 设计思路清晰，物理驱动的方法论值得借鉴
- **实验充分度**: ⭐⭐⭐⭐⭐ 5 个场景的全面评估，丰富的消融实验（分解消融、初始化对比、参数分析、收敛分析），下游 CT 重建应用验证
- **写作质量**: ⭐⭐⭐⭐ 方法部分公式推导完整，图示清晰，但 CUDA 实现细节较少
- **实用价值**: ⭐⭐⭐⭐⭐ 148fps 推理速度和 9min 训练时间使其具备临床应用潜力，稀疏视角 CT 重建的提升对减少患者辐射有直接价值

<!-- RELATED:START -->

## 相关论文

- [PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI](../../AAAI2026/medical_imaging/pings-x_physics-informed_normalized_gaussian_splatting_with_axes_alignment_for_e.md)
- [GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction](../../CVPR2026/medical_imaging/gaussianpile_a_unified_sparse_gaussian_splatting_framework_for_slice-based_volum.md)
- [FOXES: A Framework For Operational X-ray Emission Synthesis](../../NeurIPS2025/medical_imaging/foxes_a_framework_for_operational_x-ray_emission_synthesis.md)
- [A Cephalometric Landmark Regression Method Based on Dual-Encoder for High-Resolution X-Ray Image](a_cephalometric_landmark_regression_method_based_on_dual-encoder_for_high-resolu.md)
- [Co-synthesis of Histopathology Nuclei Image-Label Pairs using a Context-Conditioned Joint Diffusion Model](co-synthesis_of_histopathology_nuclei_image-label_pairs_using_a_context-conditio.md)

<!-- RELATED:END -->
