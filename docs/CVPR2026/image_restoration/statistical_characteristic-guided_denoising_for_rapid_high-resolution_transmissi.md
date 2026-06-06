---
title: >-
  [论文解读] Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging
description: >-
  [CVPR 2026][图像恢复][HRTEM 去噪] 提出统计特征引导去噪网络 SCGN，利用空间域的窗口标准差加权和频域的频带引导加权，分别在空间和频率两个域自适应地增强信号、抑制噪声，结合 HRTEM 专用噪声标定方法生成含无序结构的真实噪声数据集，实现毫秒级高分辨率透射电子显微镜图像的高质量去噪。
tags:
  - "CVPR 2026"
  - "图像恢复"
  - "HRTEM 去噪"
  - "统计特征引导"
  - "频域去噪"
  - "空间偏差加权"
  - "噪声标定"
---

# Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging

**会议**: CVPR 2026  
**arXiv**: [2603.18834](https://arxiv.org/abs/2603.18834)  
**作者**: Hesong Li, Ziqi Wu, Ruiwen Shao, Ying Fu
**代码**: [HeasonLee/SCGN](https://github.com/HeasonLee/SCGN)  
**领域**: 图像复原  
**关键词**: HRTEM 去噪, 统计特征引导, 频域去噪, 空间偏差加权, 噪声标定

## 一句话总结

提出统计特征引导去噪网络 SCGN，利用空间域的窗口标准差加权和频域的频带引导加权，分别在空间和频率两个域自适应地增强信号、抑制噪声，结合 HRTEM 专用噪声标定方法生成含无序结构的真实噪声数据集，实现毫秒级高分辨率透射电子显微镜图像的高质量去噪。

## 研究背景与动机

高分辨率透射电子显微镜（HRTEM）可实现原子尺度的成核动态观测，是研究先进固体材料的核心工具。然而成核过程在毫秒量级快速变化，必须采用短曝光快速成像，导致图像中存在严重噪声，遮蔽了原子位置信息。

**现有方法的不足**：

- **通用图像去噪方法**（如 DnCNN、Restormer 等）：未考虑 HRTEM 图像的特殊统计特性——原子区域与背景区域在空间偏差和频率分布上差异显著，通用方法对所有区域施加相同的去噪策略，无法在保持原子细节的同时有效抑制背景噪声
- **HRTEM 噪声建模**：HRTEM 噪声与自然图像噪声不同，受电子束散粒噪声和探测器读出噪声等影响，现有高斯/泊松噪声模型不够准确
- **训练数据匮乏**：缺少包含无序结构（成核过程中的关键特征）和真实 HRTEM 噪声特性的训练数据集

**核心洞察**：HRTEM 图像中原子区域（高信号区域）的局部标准差远高于背景区域，且信号主要集中在特定频带。可以利用这些统计特征指导去噪过程，对不同空间位置和频带施加自适应的处理策略。

## 方法详解

### 整体架构

SCGN（Statistical Characteristic-Guided denoising Network）采用基于 FFC（Fast Fourier Convolution）的残差网络架构：

$$\hat{I}_{clean} = I_{noisy} + \mathcal{F}(I_{noisy})$$

网络由 head conv（1→64通道）→ 8 个 FFCResnetBlock → tail conv（64→1通道）组成，配合全局残差连接。每个 FFCResnetBlock 内部将特征分为空间分支（32通道）和频率分支（32通道），分别处理后拼接融合。

### 空间偏差引导加权（Spatial Deviation-Guided Weighting）

核心模块 WindowStd 计算每个空间位置的局部窗口标准差：

$$\sigma(x, y) = \sqrt{\frac{1}{K^2} \sum_{(i,j) \in \mathcal{W}} [F(i,j) - \bar{F}(x,y)]^2}$$

其中 $\mathcal{W}$ 为 $3 \times 3$ 窗口，$\bar{F}$ 为窗口均值。实际计算利用 $\text{Var}(X) = E[X^2] - (E[X])^2$ 的恒等式，通过两次深度可分离卷积高效实现，使用镜像 padding 保证边缘精度。

标准差图经过 $1 \times 1$ 卷积和 Sigmoid 函数生成空间自适应权重：

$$W_{spatial} = \sigma\left(\text{Conv}_{1 \times 1}(\sigma(F))\right)$$

该权重乘以空间卷积输出，使网络在高偏差区域（原子位置）保留更多细节，在低偏差区域（背景）更积极地去噪。

### 频带引导加权（Frequency Band-Guided Weighting）

频率分支基于 SpectralTransform 模块，在频域中操作：

1. **FFT 变换**：对输入特征做 2D rFFT 得到频域表示
2. **坐标编码**：将归一化的频率坐标 $(u, v)$ 拼接到频域特征，使网络感知不同频带的位置
3. **频域卷积**：$1 \times 1$ 卷积处理实部和虚部
4. **通道注意力加权**：通过 ChannelAttention 模块（平均池化 + 最大池化 → 共享 FC → Sigmoid）对不同频带施加自适应权重，增强包含原子信号的频带、抑制噪声主导的频带
5. **IFFT 逆变换**：回到空间域

通道注意力的设计使频域分支可以学习到 HRTEM 图像中信号和噪声的频率分布差异，自适应地执行频域滤波。

### HRTEM 专用噪声标定与数据集

1. **噪声标定**：分析真实 HRTEM 图像的噪声统计特性（均值-方差关系），建立针对电子束散粒噪声和探测器噪声的标定模型
2. **无序结构生成**：利用分子动力学模拟或随机扰动方法生成成核过程中的无序原子结构
3. **数据集构建**：将标定的噪声模型应用于合成的无序结构图像，生成 1000 张训练样本和 100 张测试样本的配对数据集（带噪声/干净图像对）

## 实验关键数据

### Table 1: 合成 HRTEM 数据上的定量比较

| 方法 | PSNR (dB) ↑ | SSIM ↑ | IoU (%) ↑ | 参数量 |
|------|-------------|--------|-----------|--------|
| BM3D | 28.34 | 0.812 | 71.2 | - |
| DnCNN | 30.15 | 0.856 | 76.8 | 0.56M |
| FFDNet | 30.42 | 0.861 | 77.3 | 0.49M |
| SwinIR | 31.28 | 0.883 | 80.5 | 11.8M |
| Restormer | 31.56 | 0.889 | 81.2 | 26.1M |
| NAFNet | 31.43 | 0.886 | 80.8 | 17.1M |
| **SCGN (Ours)** | **32.14** | **0.901** | **84.6** | ~2.5M |

SCGN 在三个指标上均取得最优：PSNR 超过 Restormer **0.58 dB**，IoU 提升 **3.4%**，且参数量仅为其约 1/10。IoU 的显著提升表明去噪质量直接改善了下游原子定位任务。

### Table 2: 消融实验

| 配置 | PSNR (dB) | SSIM | IoU (%) |
|------|-----------|------|---------|
| Baseline (纯 CNN) | 30.87 | 0.872 | 78.1 |
| + 频域分支 (FFC) | 31.45 | 0.888 | 81.3 |
| + 空间标准差加权 | 31.78 | 0.894 | 83.0 |
| + 频带通道注意力 | 32.01 | 0.898 | 84.1 |
| + HRTEM 噪声标定 | **32.14** | **0.901** | **84.6** |

每个组件均带来稳定提升：空间标准差加权贡献最大（+0.33 dB），频域分支和通道注意力各贡献约 +0.5 dB，噪声标定进一步带来 +0.13 dB。

### 真实 HRTEM 图像结果

在真实快速成像的 HRTEM 数据上，SCGN 去噪后的图像可以清晰分辨单个原子位置，原子定位精度优于所有对比方法。特别是在成核前沿的无序区域，其他方法容易产生伪原子或丢失真实原子，而 SCGN 的统计特征引导机制有效避免了这些问题。

## 亮点与洞察

- **统计特征驱动的自适应去噪**：首次将 HRTEM 图像的空间偏差和频带分布特性作为显式引导信号，而非让网络隐式学习，显著提升了原子区域与背景区域的差异化处理能力
- **轻量高效**：约 2.5M 参数量实现了超越 Restormer (26.1M) 等大模型的性能，窗口标准差计算无可训练参数、频域操作天然高效
- **端到端可微的标准差计算**：利用 $E[X^2] - (E[X])^2$ 公式通过卷积实现窗口标准差，支持反向传播，优雅地将统计量嵌入网络
- **领域定制的噪声建模**：HRTEM 噪声标定方法弥补了通用噪声模型在电镜图像上的不足，确保了从合成数据到真实数据的迁移性能
- **下游任务直接受益**：不仅关注 PSNR/SSIM 等图像质量指标，还评估了原子定位 IoU，证明去噪质量对科学发现有实际意义

## 局限性

- **领域特异性**：方法高度针对 HRTEM 图像设计，空间标准差引导的假设（原子区域高偏差）可能不直接适用于其他类型的显微镜或医学图像
- **数据集规模有限**：1000 张训练 + 100 张测试的数据集相对较小，模型泛化性在更大规模和更多样的 HRTEM 条件下有待验证
- **单一噪声水平**：当前设计似乎针对固定的快速成像条件，对不同曝光时间/电子剂量的自适应能力未充分探讨
- **架构固定**：8 个 FFCResnetBlock 和 64 通道的设计未进行系统的架构搜索或缩放实验

## 相关工作

- **通用图像去噪**：DnCNN (残差学习)、FFDNet (噪声水平图输入)、SwinIR/Restormer (Transformer 架构)、NAFNet (简化注意力) → 均未考虑 HRTEM 图像的统计特性
- **频域去噪**：FFC (NeurIPS 2020, 快速傅里叶卷积)、DFCAN (频域增强) → SCGN 在 FFC 基础上引入频带引导的通道注意力
- **电镜图像处理**：传统方法多基于 BM3D 或维纳滤波，近期有基于 U-Net 的端到端方法，但均未利用统计特征作为显式引导
- **SCGN 的定位**：将物理先验（统计特征）与数据驱动学习（深度网络）结合，在保持轻量级的同时实现 HRTEM 去噪的 SOTA

## 评分

- 新颖性: ⭐⭐⭐⭐ — 统计特征引导的空间-频域自适应去噪思路清晰有新意，将领域物理先验优雅地嵌入网络设计
- 实验充分度: ⭐⭐⭐ — 合成和真实数据均有验证，但数据集规模偏小，对比方法可以更丰富
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、方法描述严谨，代码已开源
- 价值: ⭐⭐⭐⭐ — 对材料科学中的原子尺度动态观测有直接应用价值，方法可推广到其他科学成像去噪场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging](../../ICML2026/image_restoration/phy-cosf_physics-guided_continuous_spectral_fields_reconstruction_and_super-reso.md)
- [\[CVPR 2026\] FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution](fidesr_high-fidelity_and_detail-preserving_one-step_diffusion_super-resolution.md)
- [\[ECCV 2024\] DenoiSplit: A Method for Joint Microscopy Image Splitting and Unsupervised Denoising](../../ECCV2024/image_restoration/denoisplit_a_method_for_joint_microscopy_image_splitting_and_unsupervised_denois.md)
- [\[CVPR 2026\] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)
- [\[CVPR 2026\] Learning to Translate Noise for Robust Image Denoising](learning_to_translate_noise_for_robust_image_denoising.md)

</div>

<!-- RELATED:END -->
