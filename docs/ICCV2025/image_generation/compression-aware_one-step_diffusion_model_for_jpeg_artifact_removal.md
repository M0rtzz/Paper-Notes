---
title: >-
  [论文解读] Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal
description: >-
  [ICCV 2025][图像生成][JPEG伪影去除] 提出 CODiff，一种压缩感知的单步扩散模型用于 JPEG 伪影去除，核心是设计了压缩感知视觉嵌入器（CaVE）通过显式+隐式双重学习策略提取 JPEG 压缩先验，引导扩散模型实现高质量复原，在 LIVE-1、Urban100、DIV2K-Val 上全面超越现有方法同时推理效率极高。
tags:
  - ICCV 2025
  - 图像生成
  - JPEG伪影去除
  - 单步扩散模型
  - 压缩先验
  - 双重学习
  - 图像复原
---

# Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal

**会议**: ICCV 2025  
**arXiv**: [2502.09873](https://arxiv.org/abs/2502.09873)  
**代码**: [github.com/jp-guo/CODiff](https://github.com/jp-guo/CODiff)  
**领域**: 图像生成/图像复原  
**关键词**: JPEG伪影去除, 单步扩散模型, 压缩先验, 双重学习, 图像复原

## 一句话总结

提出 CODiff，一种压缩感知的单步扩散模型用于 JPEG 伪影去除，核心是设计了压缩感知视觉嵌入器（CaVE）通过显式+隐式双重学习策略提取 JPEG 压缩先验，引导扩散模型实现高质量复原，在 LIVE-1、Urban100、DIV2K-Val 上全面超越现有方法同时推理效率极高。

## 研究背景与动机

### 问题定义

JPEG 伪影去除旨在从压缩图像中去除块效应、色带等失真，恢复丢失的视觉信息。在高压缩率（如 QF=5）下，信息损失极为严重，传统 CNN/Transformer 方法力不从心。

### 已有方法的不足

**CNN/Transformer 方法**（FBCNN、PromptCIR 等）：在高压缩率下效果有限，因为丢失的信息无法从剩余线索中完全恢复

**多步扩散模型**（DiffBIR、SUPIR）：虽然有强大的生成先验，但多步去噪导致 50 步推理，计算开销巨大（188T MACs，50 秒）

**现有单步扩散模型**（OSEDiff）：推理高效但忽略了 JPEG 压缩先验，无法区分压缩伪影和自然图像特征

**压缩先验利用不足**：之前的质量因子（QF）学习方法仅将 QF（单个整数）作为学习目标，捕获的信息太有限；量化表方法只提供静态数值

### 核心矛盾

如何在保持单步推理效率的同时，有效提取和利用 JPEG 压缩先验来指导扩散模型？

**核心 idea**：设计一个压缩感知的视觉嵌入器（CaVE），通过"显式学习（预测 QF）+ 隐式学习（重建高质量图像）"的双重策略全面捕获 JPEG 压缩特性，然后将提取的先验注入单步扩散模型。

## 方法详解

### 整体框架

CODiff 采用两阶段训练：
- **Stage 1**：训练 CaVE 以提取 JPEG 压缩先验嵌入
- **Stage 2**：将 CaVE 提取的先验注入预训练的 StableDiffusion 模型（通过 LoRA 微调），用感知损失 + GAN 损失训练生成器

### 关键设计

#### 1. **压缩感知视觉嵌入器（CaVE）**

- **功能**：将低质量图像 $\mathbf{I}_L$ 编码为一组特征向量 $\mathbf{c}_L = \{\mathbf{c}_{L_k} \in \mathbb{R}^d\}_{k=1}^K$，作为 JPEG 压缩先验
- **核心架构**：UNet 编码器 + 轻量 QF 预测器 + UNet 解码器
- **设计动机**：利用 UNet 的多尺度特征提取能力，同时从多个分辨率捕获压缩相关信息

#### 2. **双重学习策略（Dual Learning）**

- **显式学习**：训练 CaVE 从嵌入预测 QF，使用 $\ell_1$ 损失：$\mathcal{L}_{QF} = \frac{1}{B}\sum_{i=1}^{B}\|QF_{pred}^i - QF_{gt}^i\|_1$
    - 动机：使嵌入能明确区分不同压缩级别
    - 局限：仅靠 QF 预测无法泛化到未见过的压缩级别（t-SNE 可视化证实）

- **隐式学习**：训练 CaVE 从嵌入重建高质量图像，使用 $\ell_1$ 损失：$\mathcal{L}_{rec} = \frac{1}{B}\sum_{i=1}^{B}\|\hat{\mathbf{I}}_H^i - \mathbf{I}_H^i\|_1$
    - 动机：重建目标迫使嵌入捕获压缩过程的完整信息，而非仅仅一个 QF 整数

- **联合目标**：$\mathcal{L}_{CaVE} = \mathcal{L}_{QF} + \lambda \cdot \mathcal{L}_{rec}$，其中 $\lambda=1000$
    - 关键发现：双重学习后，CaVE 能有效区分训练时**未见过**的压缩级别（QF=1, 5），而纯显式学习无法做到

#### 3. **单步扩散生成器**

- **功能**：将低质量图像的 latent 表示作为输入（替代高斯噪声），一步去噪恢复高质量图像
- **核心公式**：$\hat{\mathbf{z}}_H = \frac{\mathbf{z}_L - \sqrt{1-\bar{\alpha}_{T_L}} \varepsilon_\theta(\mathbf{z}_L; \mathbf{c}_L, T_L)}{\sqrt{\bar{\alpha}_{T_L}}}$
- **训练**：VAE encoder + UNet 通过 LoRA（rank=16）微调，VAE decoder 冻结
- **判别器**：使用预训练 SD UNet encoder + 轻量 MLP

### 损失函数 / 训练策略

Stage 2 的总损失：$\mathcal{L} = \mathcal{L}_{per} + \lambda_G \mathcal{L}_{\mathcal{G}}$
- 感知损失：$\mathcal{L}_{per} = \mathcal{L}_2(\hat{\mathbf{I}}_H, \mathbf{I}_H) + \lambda_D \mathcal{L}_{DISTS}(\hat{\mathbf{I}}_H, \mathbf{I}_H)$
- GAN 损失：标准对抗损失，$\lambda_G = 5 \times 10^{-3}$
- 不使用蒸馏（OSEDiff 依赖蒸馏），而是用 GAN 增强真实感

训练细节：
- Stage 1: 200K iterations, 4×A6000
- Stage 2: 100K iterations, 4×A6000, AdamW, lr=5e-5
- 训练 QF 范围 8-95，patch 大小 256×256

## 实验关键数据

### 主实验

LIVE-1 数据集（QF=5，感知质量指标）：

| 方法 | 步数 | LPIPS↓ | DISTS↓ | MUSIQ↑ | MANIQA↑ | CLIPIQA↑ |
|------|------|--------|--------|--------|---------|----------|
| JPEG | — | 0.4384 | 0.3242 | 40.33 | 0.2294 | 0.1716 |
| FBCNN | 1 | 0.3736 | 0.2353 | 63.56 | 0.3425 | 0.2763 |
| PromptCIR | 1 | 0.3797 | 0.2334 | 60.34 | 0.2790 | 0.2655 |
| DiffBIR* | 50 | 0.3509 | 0.2035 | 58.09 | 0.2812 | 0.3776 |
| OSEDiff* | 1 | 0.2675 | 0.1653 | 65.51 | 0.3417 | 0.5623 |
| **CODiff** | **1** | **0.2062** | **0.1121** | **73.16** | **0.5321** | **0.7212** |

计算效率对比（1024×1024 输入）：

| 方法 | 步数 | 参数量(G) | MACs(T) | 时间(s) |
|------|------|----------|---------|---------|
| DiffBIR | 50 | 1.52 | 188.24 | 50.81 |
| SUPIR | 50 | 4.49 | 464.29 | 24.33 |
| OSEDiff | 1 | 1.40 | 10.39 | 0.65 |
| **CODiff** | **1** | **1.00** | **9.46** | **0.57** |

### 消融实验

Prompt 生成方式对比（LIVE-1, QF=5）：

| 方法 | LPIPS↓ | MUSIQ↑ | MANIQA↑ |
|------|--------|--------|---------|
| Empty string | 0.3485 | 62.56 | 0.3793 |
| Learnable | 0.3471 | 63.39 | 0.3900 |
| DAPE | 0.3463 | 62.54 | 0.3793 |
| **CaVE (ours)** | **0.3426** | **67.13** | **0.4584** |

双重学习策略的有效性也通过 t-SNE 可视化清晰展示：仅显式学习的 CaVE 无法区分未见 QF（1, 5），而双重学习后聚类清晰分离。

### 关键发现

- CODiff 在所有三个数据集、所有 QF 级别上全面超越现有方法，包括 50 步的 DiffBIR 和 SUPIR
- 在 QF=5 极端压缩下优势最明显（LPIPS 从 0.2675 降至 0.2062，MANIQA 从 0.3417 升至 0.5321）
- 推理速度比 DiffBIR 快 89 倍（0.57s vs 50.81s），参数量减少 34%
- CaVE 是性能提升的核心（比 DAPE 的 MANIQA 高 18%）
- 双重学习显著优于纯显式或纯隐式学习

## 亮点与洞察

- **压缩先验的巧妙利用**：不仅预测 QF，还通过重建目标让模型理解完整的压缩过程，这是一个通用思路——用多任务学习丰富表示
- **轻量化设计**：CaVE 是 UNet encoder（不需要完整 UNet），远比 ControlNet、DAPE 等辅助模块轻量
- **不依赖蒸馏**：与 OSEDiff 不同，CODiff 用 GAN 替代蒸馏，不受教师模型上限约束
- **t-SNE 可视化**：直观展示了双重学习如何帮助泛化到未见压缩级别

## 局限与展望

- 训练 QF 范围 8-95，对极低 QF（1-7）的泛化依赖双重学习的隐式扩展能力
- 仅针对 JPEG 压缩设计，未探索 WebP、HEIF 等现代压缩格式
- GAN 训练可能引入模式坍塌风险
- 未在真实压缩图像（非合成降质）上评估
- CaVE 的 UNet decoder 仅用于 Stage 1 训练，Stage 2 丢弃，存在一定计算浪费

## 相关工作与启发

- FBCNN 首先提出预测可调 QF 作为压缩先验，CODiff 在此基础上通过双重学习大幅提升
- OSEDiff 证明了单步扩散在图像复原中的可行性，CODiff 证明了领域特定先验的重要性
- CaVE 的设计思路（多任务提取表示）可以推广到其他降质类型（模糊、噪声）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 双重学习策略和压缩先验注入扩散模型是有效且新颖的设计
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集、多 QF 级别、全面的指标体系、消融完整
- **写作质量**: ⭐⭐⭐⭐ — 图示清晰，t-SNE 可视化有说服力
- **价值**: ⭐⭐⭐⭐⭐ — 实际应用价值高，推理快且效果好，代码开源

<!-- RELATED:START -->

## 相关论文

- [SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](sanasprint_onestep_diffusion_with_continuoustime_consistency.md)
- [OSDFace: One-Step Diffusion Model for Face Restoration](../../CVPR2025/image_generation/osdface_one-step_diffusion_model_for_face_restoration.md)
- [Steering One-Step Diffusion Model with Fidelity-Rich Decoder for Fast Image Compression](../../AAAI2026/image_generation/steering_one-step_diffusion_model_with_fidelity-rich_decoder_for_fast_image_comp.md)
- [Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [Structure-Guided Diffusion Models for High-Fidelity Portrait Shadow Removal](structure-guided_diffusion_models_for_high-fidelity_portrait_shadow_removal.md)

<!-- RELATED:END -->
