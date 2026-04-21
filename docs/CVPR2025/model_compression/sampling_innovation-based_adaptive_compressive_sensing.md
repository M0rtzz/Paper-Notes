---
title: >-
  [论文解读] Sampling Innovation-Based Adaptive Compressive Sensing
description: >-
  [CVPR 2025][模型压缩][自适应压缩感知] 提出 SIB-ACS 框架，通过"采样创新"准则（衡量采样增量带来的重建误差下降）指导多阶段自适应采样分配，并设计主成分压缩域网络（PCCD-Net）进行高保真图像重建，显著超越 SOTA 压缩感知方法。
tags:
  - CVPR 2025
  - 模型压缩
  - 自适应压缩感知
  - 采样创新
  - 负反馈机制
  - 深度展开网络
  - 主成分梯度下降
---

# Sampling Innovation-Based Adaptive Compressive Sensing

**会议**: CVPR 2025  
**arXiv**: [2503.13241](https://arxiv.org/abs/2503.13241)  
**代码**: [GitHub](https://github.com/giant-pandada/SIB-ACS_CVPR2025)  
**领域**: 模型压缩  
**关键词**: 自适应压缩感知, 采样创新, 负反馈机制, 深度展开网络, 主成分梯度下降

## 一句话总结

提出 SIB-ACS 框架，通过"采样创新"准则（衡量采样增量带来的重建误差下降）指导多阶段自适应采样分配，并设计主成分压缩域网络（PCCD-Net）进行高保真图像重建，显著超越 SOTA 压缩感知方法。

## 研究背景与动机

压缩感知（CS）利用信号稀疏性进行欠采样重建，广泛应用于医学成像、高光谱成像等领域。均匀压缩感知（UCS）对所有图像块使用相同采样率，无法适配区域复杂度差异。自适应压缩感知（ACS）根据图像块内容动态分配采样资源，但在未知场景下（无 ground truth）面临关键挑战。

现有 ACS 方法存在两类问题：(1) 测量域方法（基于测量误差、余弦相似度等）——欠采样测量数据本身是病态的，判断不准确；(2) 图像域方法（基于重建图像复杂度分析）——分析指标与采样形成正反馈循环，导致采样集中在初始分配区域，无法纠错。

本文提出基于"采样创新"的负反馈机制来解决这些问题：通过比较采样增量前后的重建差异来判断哪些区域最需要额外采样，且随着主成分被恢复，创新值自然下降形成负反馈。

## 方法详解

### 整体框架

SIB-ACS 包含两大模块：(1) 自适应采样模块（ASM）——通过创新准则和多阶段负反馈实现自适应采样分配；(2) 图像重建模块（PCCD-Net）——通过主成分和压缩域双路径的近端梯度下降进行高效重建。

### 关键设计1：采样创新准则

**功能**：准确判断各图像块需要多少额外采样资源。

**核心思路**：定义"创新"为采样增量带来的重建图像变化量 $\alpha = \|\hat{\mathbf{x}}_{\text{IS}} - \hat{\mathbf{x}}_{\text{HM}}\|_2^2$，即增量采样后的重建与历史测量重建的差异。各块的自适应采样数按创新值比例分配：$M_n = M_{\text{ASR}} \cdot \frac{\|\alpha_n\|_2^2}{\sum_n \|\alpha_n\|_2^2}$。

**设计动机**：创新值直接估计了重建误差的下降量，方向与最小化重建误差目标一致。更重要的是，创新是相对度量——随着图像块主成分被恢复，创新值自然下降，形成负反馈，避免了正反馈导致的采样集中问题。

### 关键设计2：创新引导的多阶段自适应采样

**功能**：通过迭代负反馈逐步消除各块的残余创新。

**核心思路**：每个 AS 阶段包含三步：(1) 创新采样（IS）——基于上一阶段分布进行均匀探测；(2) 创新估计（IE）——用轻量网络分别重建 IS 前后的图像并计算差异；(3) 自适应采样（AS）——按创新权重分配新采样。初始阶段使用均匀采样率 $SR_{\text{init}}$，然后迭代 $S$ 个阶段。每块最大采样率设为 $s/S$。

**设计动机**：多阶段框架利用负反馈逐步修正采样分配误差，每阶段的创新估计独立于前一阶段的错误，收敛性更好。

### 关键设计3：主成分压缩域网络（PCCD-Net）

**功能**：在自适应采样场景下高效进行高保真图像重建。

**核心思路**：采用深度展开网络，每个迭代阶段包含两条并行路径。PCPGD 路径：将 $C$ 通道特征聚合为单通道主成分图像，在图像域计算梯度后扩展回特征域。CDPGD 路径：将特征压缩到 $L$ 维压缩域，在通道维度上计算梯度。最终两路径特征互补相加：$\mathbf{X}^k = \mathbf{X}^k_p + \mathbf{X}^k_c$。

**设计动机**：ACS 中复杂区域采样率更高，采样矩阵更大，导致特征域梯度下降的计算开销剧增。通过主成分路径压缩 GD 操作维度，在控制计算成本的同时利用压缩域路径补充细节特征。

### 损失函数

使用 $\ell_1$ 损失和 SSIM 损失的加权组合：$\mathcal{L}(\Theta) = \mathcal{L}_{l_1}(\Theta) + \mu \mathcal{L}_{\text{SSIM}}(\Theta)$，同时优化像素精度和纹理质量。

## 实验关键数据

### 主实验：BSD68 数据集 PSNR (dB) 对比

| 方法 | SR=0.10 | SR=0.25 | SR=0.50 | 平均 |
|------|---------|---------|---------|------|
| CPP-Net (CVPR'24, UCS) | 28.41 | 32.25 | 37.30 | 33.33 |
| CASNet (TIP'22, ACS) | 28.41 | 32.31 | 37.49 | 33.41 |
| AMS-Net (TMM'22, ACS) | 29.36 | 33.53 | 39.20 | 34.73 |
| **SIB-ACS (ours)** | **29.54** | **34.35** | **41.14** | **35.83** |

### Urban100 数据集 PSNR (dB) 对比

| 方法 | SR=0.10 | SR=0.25 | SR=0.50 | 平均 |
|------|---------|---------|---------|------|
| CPP-Net (CVPR'24, UCS) | 28.48 | 33.37 | 38.29 | 34.24 |
| AMS-Net (TMM'22, ACS) | 28.04 | 33.22 | 38.33 | 34.06 |
| **SIB-ACS (ours)** | **30.72** | **36.30** | **42.96** | **37.35** |

### 关键发现

- SIB-ACS 在 BSD68 上平均 PSNR 比最强 UCS 方法（CPP-Net）高 2.5 dB，比最强 ACS 方法（AMS-Net）高 1.1 dB。
- 在 Urban100（纹理丰富场景）上优势更明显，平均 PSNR 比 AMS-Net 高 3.29 dB，说明负反馈机制对复杂区域的自适应分配更准确。
- 低采样率（SR=0.10）下 SIB-ACS 的优势最显著，因为资源稀缺时准确的分配策略更关键。
- PCCD-Net 的双路径设计在降低计算成本的同时保持了与全特征域 GD 相当的重建质量。

## 亮点与洞察

1. **负反馈机制设计精巧**：采样创新本质上是一个自限制的度量——重建越好创新越低——天然避免了正反馈陷阱，这是对现有 ACS 框架的根本性改进。
2. **PCCD-Net 双路径互补**：主成分路径处理主要结构，压缩域路径补充细节，两者互补类似于低频+高频分解思想。
3. **单模型多采样率**：通过多阶段框架实现单一模型处理任意采样率的自适应感知。

## 局限与展望

- 多阶段采样增加了总采样和重建时间，实时性受限。
- 轻量重建网络（IE 中用于快速重建）的精度可能影响创新估计的准确性。
- 当前以块级别进行自适应分配，更细粒度的像素级分配可能进一步提升效果。
- 实验主要在自然图像上验证，对医学/遥感等特殊领域的适用性待探索。

## 相关工作与启发

- **OCTUF (CVPR'23)**、**CPP-Net (CVPR'24)**：强 UCS 基线，本文 ACS 策略在其基础上通过更智能的采样分配获得显著提升。
- **深度展开网络 (DUN)**：PCCD-Net 继承了 DUN 的物理信息注入优势，同时通过双路径压缩解决了 ACS 场景下 GD 维度膨胀的计算问题。
- **AMS-Net (TMM'22)**：此前最强 ACS 方法，但缺乏负反馈机制，本文在所有采样率上超越。

## 评分

⭐⭐⭐⭐ — 采样创新+负反馈的设计理念简洁而有效，PCCD-Net 的双路径架构实用性强。实验全面，在多个基准上取得了显著的 PSNR 提升，尤其在纹理丰富场景上表现突出。

<!-- RELATED:START -->

## 相关论文

- [Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling](../../ECCV2024/model_compression/adaptive_compressed_sensing_with_diffusionbased_posterior_sa.md)
- [Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing](../../ECCV2024/model_compression/adaptive_selection_of_samplingreconstruction_in_fourier_comp.md)
- [AutoSSVH: Exploring Automated Frame Sampling for Efficient Self-Supervised Video Hashing](autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_h.md)
- [Adaptive Stochastic Coefficients for Accelerating Diffusion Sampling](../../NeurIPS2025/model_compression/adaptive_stochastic_coefficients_for_accelerating_diffusion_sampling.md)
- [HyperLoRA: Parameter-Efficient Adaptive Generation for Portrait Synthesis](hyperlora_parameter-efficient_adaptive_generation_for_portrait_synthesis.md)

<!-- RELATED:END -->
