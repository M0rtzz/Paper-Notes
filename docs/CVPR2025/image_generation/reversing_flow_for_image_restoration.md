---
title: >-
  [论文解读] Reversing Flow for Image Restoration
description: >-
  [CVPR 2025][图像生成][连续归一化流] ResFlow 提出将图像退化过程建模为确定性连续归一化流（而非随机扩散过程），通过辅助变量消解退化的不可逆性实现可逆建模，采用熵守恒调度策略，仅需 4 步采样即可完成高质量图像复原，在去雪/去雨/去雾/去噪/去压缩伪影等任务上达到 SOTA。
tags:
  - CVPR 2025
  - 图像生成
  - 连续归一化流
  - 图像复原
  - 确定性退化路径
  - 辅助过程
  - 熵守恒
---

# Reversing Flow for Image Restoration

**会议**: CVPR 2025  
**arXiv**: [2506.16961](https://arxiv.org/abs/2506.16961)  
**代码**: 无（有 Project Page）  
**领域**: 图像复原 / 生成模型  
**关键词**: 连续归一化流, 图像复原, 确定性退化路径, 辅助过程, 熵守恒

## 一句话总结

ResFlow 提出将图像退化过程建模为确定性连续归一化流（而非随机扩散过程），通过辅助变量消解退化的不可逆性实现可逆建模，采用熵守恒调度策略，仅需 4 步采样即可完成高质量图像复原，在去雪/去雨/去雾/去噪/去压缩伪影等任务上达到 SOTA。

## 研究背景与动机

**领域现状**：图像复原旨在从退化的低质量（LQ）图像恢复高质量（HQ）图像。基于扩散/分数的生成模型是当前主流，包括 DDRM、IR-SDE、I2SB、ResShift、RDDM 等。它们通常将退化过程建模为随机前向过程，学习反向过程来恢复图像。

**现有痛点**：(1) 从高斯噪声开始反向过程是不必要且低效的，因为 LQ 图像中已包含大量结构信息；(2) 即使 IR-SDE、I2SB 等将 LQ 图像融入前向过程，仍将退化视为渐进扩散的随机过程，引入额外复杂性；(3) 随机性导致训练和推理效率低，通常需要数十到上百步采样。

**核心矛盾**：图像退化本质上是**不可逆**的（信息被擦除），体现在互信息的递减——DPI（数据处理不等式）表明退化过程中中间状态与 HQ 之间的互信息单调递减。但确定性 ODE 描述的是可逆过程（保持互信息恒定），两者矛盾。因此不能直接用 ODE 建模退化。

**本文目标**：将退化过程建模为确定性路径而非随机路径，实现快速且高质量的图像复原。

**切入角度**：从互信息的角度分析退化的不可逆性，引入辅助变量过程 $\{\mathbf{y}_t\}$ 来编码退化中丢失的信息，使增广后的联合过程 $\mathbf{z}_t = [\mathbf{x}_t; \mathbf{y}_t]$ 成为可逆的 ODE。

**核心 idea**：用辅助变量弥补退化造成的信息损失——当 $\mathbf{x}_t$ 接近 LQ 时互信息减少，$\mathbf{y}_t$ 补偿这部分信息使总互信息恒定，从而用确定性流实现可逆退化建模。

## 方法详解

### 整体框架

ResFlow 定义增广状态 $\mathbf{z}_t^T = [\mathbf{x}_t^T; \mathbf{y}_t^T]$，其中 $\mathbf{x}_0 = \mathbf{x}_{HQ}$, $\mathbf{x}_1 = \mathbf{x}_{LQ}$, $\mathbf{y}_0 = 0$, $\mathbf{y}_1 \sim \mathcal{N}(0, I)$。训练一个速度场网络 $\mathbf{v}_\theta(\mathbf{x}_t, \mathbf{y}_t, t)$ 通过 velocity matching 学习。推理时从 $t=1$（LQ + 随机 $\mathbf{y}_1$）积分到 $t=0$，取 $\hat{\mathbf{x}}_0$ 作为复原结果，只需 4 步。

### 关键设计

1. **增广退化流 (Augmented Degradation Flow)**:

    - 功能：通过辅助变量使不可逆的退化过程变为可逆的 ODE
    - 核心思路：观察到确定性 ODE $\partial \mathbf{z}_t / \partial t = \mathbf{v}(\mathbf{z}_t, t)$ 保持互信息恒定（Proposition 1），但退化过程互信息递减。引入辅助变量 $\mathbf{y}_t$ 与"不确定性范围"耦合——当 $\mathbf{x}_t$ 失去与 HQ 的互信息时，$\mathbf{y}_t$ 承载补偿的信息量。$\mathbf{y}_1$ 为高斯噪声（最大熵），$\mathbf{y}_0 = 0$（复原完成时无需辅助信息）。训练时 $\mathbf{y}_t$ 与 $\mathbf{x}_0$ 独立耦合，但训练好的速度网络在推理时产生确定性耦合
    - 设计动机：直接用 ODE 无法建模不可逆退化，增广状态空间后 ODE 的互信息守恒性被正确利用

2. **熵守恒退化调度 (Entropy-Preserving Schedule)**:

    - 功能：定义数据分量和辅助分量各自的插值路径
    - 核心思路：数据分量走直线路径 $\alpha_t^x = 1 - t$, $\sigma_t^x = t$（欧几里得空间测地线）。辅助分量采用非线性调度 $\sigma_t^y = \beta \cdot (1 - t + \beta)^{-1}$（$\beta = 10$），使总熵在整个流程中保持恒定。这基于可逆过程中熵应守恒的直觉
    - 设计动机：与直接估计 $\mathbb{E}[\mathbf{z}_0 | \mathbf{z}_t]$ 的参数化相比，velocity matching 避免了在 $t \to 0$ 附近的高离散化误差

3. **自适应损失加权 (Loss Weighting)**:

    - 功能：平衡不同时间步的训练梯度
    - 核心思路：时间越接近 $t=1$（LQ 端），$\mathbf{x}_t$ 与 HQ 的互信息越少，速度预测越困难。损失加权函数 $\lambda(t) = (\cos(\frac{\pi}{2}(t-2)) + 1)^\gamma$（$\gamma = 1.75$）使 $t$ 接近 1 时权重增大，确保模型在困难区域投入更多学习力度
    - 设计动机：近 LQ 端的速度估计更重要也更困难，需要更大的损失权重来补偿

### 损失函数 / 训练策略

Velocity matching loss：$\min_\theta \mathbb{E}[\int_0^1 \lambda(t) \|\mathbf{v}_\theta(\mathbf{x}_t, \mathbf{y}_t, t) - \dot{\mathbf{z}}_t\|^2 dt]$。无需模拟 ODE，直接匹配目标速度，训练高效。采用 DDPM 的 U-Net 架构，时间步通过 adaptive layer norm 注入。在 256 分辨率裁剪上训练，全分辨率测试。均匀时间调度、4 步采样。

## 实验关键数据

### 主实验

**合成数据集**：

| 任务 | 指标 | 之前 SOTA | ResFlow |
|------|------|----------|---------|
| 去雪 (Snow100K) | PSNR/SSIM/LPIPS | 30.92/0.917/0.034 | **31.86**/0.917/**0.030** |
| 去雨 (Outdoor-Rain) | PSNR/SSIM | 30.99/0.934 | **32.82**/**0.936** |
| 去雾 (Dense-Haze) | PSNR/SSIM | 17.07/0.63 | **17.12**/0.59 |

**真实数据集**：

| 任务 | 指标 | 之前 SOTA | ResFlow |
|------|------|----------|---------|
| 去噪 (SIDD) | PSNR/SSIM | 40.02/0.960 | **42.26**/**0.962** |
| 去雾 (NH-HAZE) | PSNR | 20.66 | **21.44** |
| 去雨 (LHP) | PSNR/SSIM | 34.33/0.946 | **34.54**/0.939 |

### 消融实验

散焦去模糊 (DPDD Combined)：ResFlow 达到 PSNR 最高，超越 Restormer (25.98) 和 FocalNet (26.18)。

### 关键发现

- 仅需 4 步采样即达到 SOTA，比扩散方法（通常 50-100 步）快一个数量级以上
- 确定性路径比随机路径更适合图像复原——退化是已知的，不需要随机性
- 熵守恒调度比直线调度效果更好
- 辅助变量在概念上编码了退化丢失的信息，在推理时 $\mathbf{y}_1$ 的不同采样产生不同复原结果，保留了生成模型的多样性

## 亮点与洞察

- **信息论视角**非常优雅——用互信息守恒/递减来解释为什么 ODE 不能直接建模退化，以及为什么需要辅助变量
- **Proposition 1** 的证明将 ODE 与信息论联系起来，为 ResFlow 提供了坚实的理论基础
- 将 flow matching 框架适配到图像复原的方式非常自然，完全避免了扩散模型的冗余随机性
- 4 步采样的极致效率对实际部署友好

## 局限与展望

- 辅助变量 $\mathbf{y}_1$ 的随机采样意味着每次复原结果略有不同，对需要确定性输出的应用可能需要额外处理
- 论文未讨论超分辨率等其他低级视觉任务
- 对于极端退化（如信息几乎完全丢失），辅助变量的补偿能力可能不足
- 与预训练扩散模型的结合方式未探索——能否从预训练 Stable Diffusion 适配？

## 相关工作与启发

- **Flow Matching / Continuous Normalizing Flow** 在生成建模中的成功被迁移到图像复原
- 与 InDI（增量估计 HQ）相比，ResFlow 的 velocity matching 避免了接近 HQ 时的误差敏感性
- 辅助变量消解不可逆性的思路可推广到其他 ill-posed 逆问题

## 评分

- **新颖性**: 9/10 — 信息论驱动的增广流设计理论优美、思路新颖
- **实验充分度**: 8/10 — 覆盖 5 个任务多个数据集，但部分任务改进幅度小
- **写作质量**: 9/10 — 理论推导清晰，图示直观
- **价值**: 8/10 — 4 步高质量复原对实际应用很有价值

<!-- RELATED:START -->

## 相关论文

- [Navigating Image Restoration with VAR's Distribution Alignment Prior](navigating_image_restoration_with_vars_distribution_alignment_prior.md)
- [Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)
- [GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration](gendeg_diffusion-based_degradation_synthesis_for_generalizable_all-in-one_image_.md)
- [Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)](zero-shot_image_restoration_using_few-step_guidance_of_consistency_models_and_be.md)
- [OSDFace: One-Step Diffusion Model for Face Restoration](osdface_one-step_diffusion_model_for_face_restoration.md)

<!-- RELATED:END -->
