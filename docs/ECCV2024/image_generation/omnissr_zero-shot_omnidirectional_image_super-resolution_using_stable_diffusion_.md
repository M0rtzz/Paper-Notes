---
title: >-
  [论文解读] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model
description: >-
  [ECCV 2024][图像生成] 提出 OmniSSR，首个基于扩散模型的零样本全向图像超分方法，通过十八面切线投影信息交互（OTII）和梯度分解（GD）校正技术，利用 Stable Diffusion 的图像先验实现保真度和真实感的平衡，无需任何训练或微调。
tags:
  - ECCV 2024
  - 图像生成
---

# OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model

**会议**: ECCV 2024  
**arXiv**: [2404.10312](https://arxiv.org/abs/2404.10312)  
**领域**: 图像生成

## 一句话总结

提出 OmniSSR，首个基于扩散模型的零样本全向图像超分方法，通过十八面切线投影信息交互（OTII）和梯度分解（GD）校正技术，利用 Stable Diffusion 的图像先验实现保真度和真实感的平衡，无需任何训练或微调。

## 研究背景与动机

全向图像（ODI）在 VR、监控等场景中应用广泛，超分辨率可提升视觉细节。现有全向图像超分方法（ODISR）面临两大挑战：

**端到端方法的局限**：大多数方法（如 SphereSR、OSRT）只能产生确定性输出，数据保真度好但感知质量差（过度平滑），且训练需要大量高分辨率 ODI（采集成本高）

**直接应用扩散先验的困难**：如果将 ERP 图像分割为切线投影（TP）图像后逐一独立超分，重投影回 ERP 时重叠区域会出现不一致，破坏全局连续性

本文的核心思路是：利用 Stable Diffusion 的强图像先验，在 ERP 和 TP 之间反复转换以融合信息，并通过梯度分解校正保证一致性约束。

## 方法详解

### 整体框架

OmniSSR 的流程分三步：

1. **预处理**：将低分辨率 ERP 图像上采样后投影为 18 个 TP 图像
2. **迭代去噪**：在每个去噪步中，TP 图像通过 SD（含 time-aware adapter）去噪，再进行 OTII 信息交互和 GD 校正
3. **后处理**：将最终 TP 图像逆投影回 ERP 格式并做最终 GD 校正

### 关键设计

**十八面切线投影信息交互（OTII）**：

- 单张 ERP 图像由 18 个 TP 图像表示
- 在每个去噪步中，将中间结果在 ERP 和 TP 格式之间交替转换：TP→ERP（信息融合）→TP（回到可用 SD 的平面域）
- 解决了独立处理 TP 图像导致的全局不连续问题
- **预上采样策略**：在投影变换前先对源图像做双三次上采样，显著减轻投影变换中的信息损失

**梯度分解（GD）校正**：

将超分建模为逆问题 $\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{n}$，通过梯度下降得到解析近似解：

$$\tilde{\mathbf{E}}_{0|t} = \mathbf{E}_{0|t} + \gamma \mathbf{A}^\dagger(\mathbf{E}_{init} - \mathbf{A}\mathbf{E}_{0|t})$$

分解为两项：
- $\gamma \mathbf{A}^\dagger \mathbf{E}_{init}$：保证与输入的一致性（保真度）
- $(\mathbf{I} - \gamma \mathbf{A}^\dagger \mathbf{A})\mathbf{E}_{0|t}$：迭代更新 SD 生成结果（真实感）

超参数 $\gamma$ 平衡保真度和视觉质量，设置 $\gamma_p=1.0$, $\gamma_e=1.0$, $\gamma_l=0.5$。

同时在潜空间中进行去噪结果融合：$\tilde{\mathbf{z}}_{0|t}^{(i)} = (1-\gamma_l)\mathbf{z}_{0|t}^{(i)} + \gamma_l \mathcal{E}(\tilde{\mathbf{x}}_{0|t}^{(i)})$

### 损失函数

OmniSSR 是零样本方法，无需训练损失。其核心优化在推理时通过 GD 校正实现，本质上是对凸优化问题 $\arg\min_{\mathbf{x}} \|\mathbf{y} - \mathbf{Ax}\|_2^2 + \lambda \mathcal{R}(\mathbf{x})$ 的梯度下降近似求解。

## 实验关键数据

### 主实验

**ODI-SR 和 SUN 360 数据集上的定量比较（×4 超分）**：

| 方法 | WS-PSNR↑ | WS-SSIM↑ | FID↓ | LPIPS↓ |
|------|----------|----------|------|--------|
| Bicubic | 25.43 | 0.7059 | 50.84 | 0.3755 |
| DDRM | 25.43 | 0.7367 | 32.69 | 0.3206 |
| DPS | 24.75 | 0.6594 | 120.74 | 0.4911 |
| GDP | 23.16 | 0.6692 | 77.43 | 0.4260 |
| PSLD | 21.72 | 0.5498 | 107.99 | 0.5329 |
| StableSR | 23.33 | 0.6577 | 49.95 | 0.3135 |
| **OmniSSR** | **25.77** | **0.7279** | **30.97** | **0.2977** |

**与端到端监督方法对比（×4 SR，ODI-SR）**：

| 方法 | WS-PSNR↑ | FID↓ | LPIPS↓ | NIQE↓ | DISTS↓ |
|------|----------|------|--------|-------|--------|
| SwinIR | 26.76 | 27.94 | 0.3321 | 5.3961 | 0.1710 |
| OSRT | 26.89 | 27.39 | 0.3258 | 5.4364 | 0.1695 |
| **OmniSSR** | 25.77 | **30.97** | **0.2977** | **5.2891** | **0.1541** |

### 消融实验

**OTII 和 GD 校正的逐步消融（ODI-SR ×2）**：

| 输入类型 | OTII | GD | WS-PSNR↑ | FID↓ | LPIPS↓ |
|---------|------|-----|----------|------|--------|
| ERP | ✗ | ✗ | 22.69 | 44.87 | 0.3039 |
| TP | ✗ | ✗ | 23.53 | 43.91 | 0.3113 |
| TP | ✓ | ✗ | 23.74 | 65.35 | 0.3748 |
| TP | ✗ | ✓(仅后处理) | 26.77 | 15.41 | 0.1691 |
| TP | ✓ | ✓ | **28.58** | **13.01** | **0.1575** |

### 关键发现

- OmniSSR 在零样本设置下，保真度指标（WS-PSNR）接近监督方法，而感知质量指标（LPIPS、NIQE、DISTS）全面超越
- OTII + GD 的组合效果远超各自单独使用，PSNR 从 22.69 提升到 28.58（+5.89dB）
- 预上采样策略使 ERP↔TP 转换的 PSNR 从 28.98 提升至 38.18（(4,4) 设置），大幅减缓了投影变换的信息损失
- GD 校正在每步去噪中应用比仅在后处理中应用效果更好

## 亮点与洞察

1. **零样本范式**：完全不需要全向图像数据训练，利用现成的平面图像 SD 先验，解决了高分辨率 ODI 数据稀缺的问题
2. **OTII 设计精巧**：通过 ERP↔TP 的反复转换在去噪过程中融合全局信息，弥补了独立处理 TP 图像的连续性缺陷
3. **GD 校正理论清晰**：基于凸优化的梯度下降框架，提供了保真度-真实感平衡的解析解，超参数 $\gamma$ 有明确的物理意义
4. **通用扩展性**：框架可扩展到 ODI 编辑、ODI 修复、3D 高斯泼溅增强等任务

## 局限性

- 推理速度慢：每张 1024×2048 ERP 图像约需 14 分钟，难以实时超分
- ERP↔TP 的多次转换消耗额外推理时间
- GD 校正的超参数 $\gamma$ 需手动网格搜索调节，尚未做到自适应

## 评分

⭐⭐⭐⭐ (4/5)

- 新颖性：★★★★★ — 首个扩散模型零样本 ODISR 方法
- 技术：★★★★ — OTII 和 GD 理论推导严谨，组合效果显著
- 实验：★★★★ — 多数据集多指标评估，消融充分
- 实用性：★★★ — 推理速度制约实际部署

<!-- RELATED:START -->

## 相关论文

- [Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)
- [XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](xpsr_cross-modal_priors_for_diffusion-based_image_super-resolution.md)
- [Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)
- [SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [Memory-Efficient Fine-Tuning for Quantized Diffusion Model](memory-efficient_fine-tuning_for_quantized_diffusion_model.md)

<!-- RELATED:END -->
