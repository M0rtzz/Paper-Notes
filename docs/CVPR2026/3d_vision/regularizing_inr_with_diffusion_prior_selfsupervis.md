---
title: >-
  [论文解读] Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron Computed Tomography Data
description: >-
  [CVPR 2026][3D视觉][Neutron CT] 提出 Diffusive INR (DINR) 框架，在 DD3IP 扩散重建流程中用 INR 替代传统 DIS，并通过近端损失函数将扩散模型去噪估计作为正则化先验注入 INR 优化过程，在仅 4-5 个视角的极端稀疏中子 CT 条件下实现超越 MBIR(qGGMRF)、DD3IP 和纯 INR 的重建质量。
tags:
  - CVPR 2026
  - 3D视觉
  - Neutron CT
  - Implicit Neural Representation
  - 扩散模型
  - Sparse-view Reconstruction
  - Inverse Problem
---

# Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron Computed Tomography Data

**会议**: CVPR 2026  
**arXiv**: [2603.10947](https://arxiv.org/abs/2603.10947)  
**代码**: 即将公开  
**领域**: 3D视觉  
**关键词**: Neutron CT, Implicit Neural Representation, Diffusion Prior, Sparse-view Reconstruction, Inverse Problem

## 一句话总结

提出 Diffusive INR (DINR) 框架，在 DD3IP 扩散重建流程中用 INR 替代传统 DIS，并通过近端损失函数将扩散模型去噪估计作为正则化先验注入 INR 优化过程，在仅 4-5 个视角的极端稀疏中子 CT 条件下实现超越 MBIR(qGGMRF)、DD3IP 和纯 INR 的重建质量。

## 研究背景与动机

中子计算机断层扫描 (Neutron CT) 因其独特的氢分布表征能力，在燃料电池制造、锂电池研究、植物水分运输和混凝土结构监测等领域有不可替代的作用。然而中子束流量远低于 X 射线，单次曝光需要更长时间，使得稀疏视图采集成为刚需。

传统的 Filtered Back Projection (FBP) 在投影视图数低于 Nyquist 要求时产生严重伪影。Model-Based Iterative Reconstruction (MBIR) 引入手工先验（如 TV、qGGMRF）改善了这一问题，但这些先验建模能力有限。近年来两条技术路线展现出潜力：

- **隐式神经表示 (INR)**：用 MLP 将坐标映射到衰减系数，提供连续、内存高效的体积表示，且易于集成物理前向模型。但 INR 存在频谱偏差——偏向低频分量，稀疏监督下高频结构重建不佳。
- **扩散模型先验**：DD3IP 框架通过 Steerable Conditional Diffusion (SCD) 将预训练扩散模型适配到 OOD 推理数据，实现跨正交空间维度的一致 3D 重建。DD3IP 核心发现是其框架对 DIS（扩散逆问题求解器）的选择是不可知的——可以插入任意 SOTA 求解器。

本文的动机很自然：**将 INR 作为 DD3IP 框架中的 DIS**，结合 INR 的连续表示能力与扩散模型的强生成先验，同时用扩散估计正则化 INR 以弥补其频谱偏差。

## 方法详解

### 整体框架

DINR 运行在 DD3IP 框架内。设观测模型为 $y = Ax + n$，其中 $x$ 是 3D 衰减系数体积，$A$ 是平行束 CT 投影矩阵，$n$ 是加性噪声。DINR 的完整流程如下：

1. **初始化阶段**：用纯数据保真损失预训练 INR 权重 $\phi_T$；加载合成数据预训练的扩散模型权重 $\theta_T$；用 FBP 重建 $A^*y$ 加噪初始化扩散起点 $x_T$
2. **扩散迭代**（$t = T \to 1$）：
    - 更新扩散模型权重 $\theta_{t-1}$ 以适配当前 OOD 数据——最小化 $\text{MSE}(A D_\theta(x_t|y), y)$
    - 扩散去噪得到当前估计 $\hat{x}_t = D_{\theta_{t-1}}(x_t|y)$
    - 用近端损失更新 INR 权重 $\phi_{t-1}$，将 $\hat{x}_t$ 作为正则化目标
    - DDIM 采样生成下一步估计 $x_{t-1}$，其中后验均值由 INR 输出 $F_{\phi_{t-1}}(S, A^*y)$ 提供
3. **最终步**（$t=1$）：直接输出 INR 重建结果，不再加噪

### 关键设计

1. **近端正则化 INR 损失 (Proximal INR Loss)**
    - 功能：将扩散模型的去噪估计注入 INR 优化，弥补 INR 的低频偏差
    - 核心思路：损失函数由两项组成——数据保真项 $\text{MSE}(A F_\phi(S, A^*y), y)$ 确保与投影数据一致；近端项 $\rho \cdot \text{MSE}(\hat{x}_t, F_\phi(S, A^*y))$ 将 INR 输出拉向扩散模型当前估计。完整公式为 $\mathcal{L}_\phi = \text{MSE}(AF_\phi, y) + \rho \cdot \text{MSE}(\hat{x}_t, F_\phi)$
    - 设计动机：扩散先验擅长高频结构建模，INR 擅长数据一致性约束，近端形式实现两者优势互补。$\rho$ 一个超参数即可控制平衡

2. **FBP 增强的 SIREN INR 架构**
    - 功能：提供连续的 3D 体积表示，同时利用 FBP 粗重建加速收敛
    - 核心思路：INR 采用 SIREN 架构（周期性正弦激活函数的 MLP），输入为 3D 坐标网格 $S$ 和 FBP 重建 $A^*y$ 的拼接。SIREN 的周期性激活天然适合捕获高频信号
    - 设计动机：FBP 虽然质量差，但提供了有意义的初始估计，帮助 INR 更快收敛到合理解；坐标输入使重建分辨率无关

3. **噪声缩放参数 $\omega$ 与初始化策略**
    - 功能：控制扩散起点中信号与噪声的相对比例
    - 核心思路：$x_T = \sqrt{\alpha_T} A^*y + \sqrt{1 - \alpha_T} \cdot \epsilon \cdot \omega$，$\omega$ 越大噪声越强，扩散先验影响越大；$\omega$ 越小则更信任 FBP 估计
    - 设计动机：不同稀疏度下 FBP 质量差异巨大，$\omega$ 提供了自适应调节的灵活性。实验表明不同视图数下最优 $\omega$ 不同（4 视图用 0.2，8-32 视图用 0.02 或 0.002）

### 损失函数 / 训练策略

- **INR 初始化**：$\rho = 0$，纯数据保真损失，不使用扩散先验
- **扩散迭代中**：$\rho$ 选择使近端项与数据保真项的比值约为 $10^{-5}$（合成数据）或 $10^{-6}$（真实数据）
- **扩散模型预训练**：仅在合成椭球体数据上训练 UNet DDPM，推理时通过 SCD 权重更新适配 OOD 混凝土微结构数据
- **DDIM 采样**：使用确定性 DDIM 反向过程，$\eta$ 控制随机性，噪声使用球面插值

## 实验关键数据

### 主实验

**合成数据**（2×256×256 体积，平行束投影）：

| 视图数 | FBP (PSNR/SSIM) | INR (SIREN) | DD3IP | **DINR** |
|--------|-----------------|-------------|-------|----------|
| 4 | 19.31 / 0.08 | 14.76 / 0.18 | 26.17 / 0.25 | **26.27 / 0.24** |
| 8 | 21.67 / 0.18 | 28.15 / 0.35 | 28.37 / 0.34 | **28.56 / 0.38** |
| 16 | 25.27 / 0.30 | 30.34 / 0.54 | 31.21 / 0.61 | **31.30 / 0.63** |
| 32 | 29.62 / 0.43 | 32.85 / 0.66 | 32.91 / 0.74 | **33.43 / 0.76** |

DINR 在所有稀疏度下均取得最高 PSNR。32 视图下比 DD3IP 高 0.52 dB，4 视图超稀疏下比纯 INR 高 11.51 dB。

**真实中子 CT 数据**（混凝土微结构，1091 视角 360° 扫描降采样）：

| 视图数 | FBP | MBIR (qGGMRF) | INR | DD3IP | **DINR** |
|--------|-----|---------------|-----|-------|----------|
| 5 | 19.90 / 0.10 | 21.02 / 0.04 | 20.18 / 0.03 | 20.89 / 0.06 | **21.27 / 0.05** |
| 9 | 22.90 / 0.33 | **26.00 / 0.38** | 24.08 / 0.27 | 25.41 / 0.34 | 25.22 / 0.35 |
| 17 | 25.91 / 0.55 | **28.10 / 0.58** | 27.30 / 0.54 | 28.04 / 0.62 | 27.56 / 0.62 |
| 33 | 30.11 / 0.73 | 31.00 / 0.77 | 29.70 / 0.71 | 31.19 / 0.79 | **31.37 / 0.77** |

5 视图下 DINR 以 21.27 dB 超越精心调参的 MBIR (21.02 dB)；33 视图下 DINR (31.37 dB) 也优于 MBIR (31.00 dB)。9 和 17 视图下 MBIR 整体 PSNR 更高，但这主要因为背景区域平滑带来的优势。

### 消融实验

论文采用 ROI 尺度分析代替传统消融实验：

- **ROI 尺度与方法优势的关系**：在 64×96 全区域裁剪到 8×8 像素的渐进 ROI 中，当 ROI < 48×48 时 DINR 开始超越其他方法，< 32×32 时优势显著。这证明 DINR 在微结构区域（而非平坦背景）的重建保真度更高
- **$\omega$ 参数影响**：4 视图最优 $\omega = 0.2$（需要更强扩散先验），8 视图最优 $\omega = 0.02$，32 视图最优 $\omega = 0.02$（更依赖数据保真）
- **$\rho$ 参数设定**：合成数据中近端项与保真项比值 $10^{-5}$，真实数据 $10^{-6}$，表明真实数据中需更弱的扩散约束

### 关键发现

- 纯 INR 在 4 视图下 PSNR 仅 14.76 dB（比 FBP 还低），说明没有先验的 INR 在极端稀疏下完全失效
- DINR 在 4 视图下达到 26.27 dB，扩散先验将 INR 性能提升了 11.51 dB
- MBIR 在中等稀疏度（9-17 视图）下 PSNR 更高，但其优势来自背景平滑而非微结构保真——ROI 分析揭示了全局指标的误导性
- 扩散模型仅在合成椭球体上训练，却成功迁移到完全不同的混凝土微结构，展示了 SCD 权重适应机制的 OOD 泛化能力

## 亮点与洞察

- **模块化设计哲学**：利用 DD3IP 对 DIS 的不可知性，将 INR 无缝插入扩散框架。这种"积木式"设计使得未来可以替换更好的 INR 架构或扩散模型
- **近端损失的优雅性**：一个 $\rho$ 参数实现了物理数据约束与学习先验的平衡，公式简洁，实现直接
- **ROI 尺度分析方法**：揭示了全局 PSNR/SSIM 在科学成像中的局限性——当关注微结构细节时，需要在不同空间尺度上评估。这一分析方法本身是一个方法论贡献
- **合成到真实的迁移**：扩散模型在简单几何（椭球体）上训练，通过 SCD 权重微调即可适配复杂微结构，降低了训练数据需求

## 局限与展望

- **体积规模受限**：仅测试了 2×256×256 的极小体积（2 个切片），未验证大规模 3D 体积重建的可行性
- **缺乏系统消融**：作者自认缺少对 FBP 输入、网络架构、扩散步数等的消融实验
- **超参数手动调节**：$\rho$ 和 $\omega$ 都需要针对数据特性手动搜索，无自适应策略
- **计算效率未报告**：INR 优化 + 扩散迭代 + SCD 权重更新的总耗时未给出，可能远慢于 FBP 或 MBIR
- **9-17 视图段表现不突出**：在中等稀疏度下 DINR 未超越精调 MBIR，限制了其实用范围
- **评估指标单一**：仅用 PSNR/SSIM，缺少分割精度、边界保真度等面向应用的指标

## 相关工作与启发

- **DD3IP [Chung & Ye, ECCV 2024]**：DINR 的直接前驱，提供了框架和 DIS 可替换的关键发现。DINR 在合成数据上 PSNR 提升 0.1-0.5 dB
- **SIREN [Sitzmann et al., NeurIPS 2020]**：DINR 采用的 INR 骨干，周期性激活函数缓解频谱偏差但不能完全解决
- **SCD [Barbano et al., 2023]**：提供了 OOD 适配扩散先验的理论基础和权重更新机制
- **启发**：近端正则化耦合思路可推广到 NeRF、3D Gaussian Splatting 等其他 INR 重建任务；OOD 适配策略对数据稀缺的科学成像（电子显微、冷冻电镜）有参考价值

## 评分

- **新颖性**: ⭐⭐⭐ 核心是 DD3IP 框架的增量改进，将 CG 求解器替换为 INR+近端正则，思路自然但非突破性
- **实验充分度**: ⭐⭐ 仅 2 层体积、无系统消融、无运行时间、无下游任务评估
- **写作质量**: ⭐⭐⭐ 方法推导清晰完整，但篇幅偏短，ROI 分析是亮点
- **实用价值**: ⭐⭐⭐ 对中子 CT 稀疏重建有直接应用价值，但通用性和规模化验证不足

<!-- RELATED:START -->

## 相关论文

- [Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron CT Data](../../CVPR2025/3d_vision/regularizing_inr_with_diffusion_prior_self-supervised_3d_reconstruction_of_neutr.md)
- [DuoMo: Dual Motion Diffusion for World-Space Human Reconstruction](duomo_dual_motion_diffusion_for_world-space_human_reconstruction.md)
- [E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training](e-rayzer_self-supervised_3d_reconstruction_as_spatial_visual_pre-training.md)
- [PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)
- [M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation](m3dlayout_a_multi-source_dataset_of_3d_indoor_layouts_and_structured_description.md)

<!-- RELATED:END -->
