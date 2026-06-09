---
title: >-
  [论文解读] ReNoise: Real Image Inversion Through Iterative Noising
description: >-
  [ECCV 2024][图像生成] 提出 ReNoise 迭代重噪方法改进扩散模型的图像反演质量，通过在每个反演步骤多次应用 UNet 并平均预测来提升轨迹估计精度，尤其适用于少步扩散模型（SDXL Turbo、LCM）。
tags:
  - "ECCV 2024"
  - "图像生成"
---

# ReNoise: Real Image Inversion Through Iterative Noising

**会议**: ECCV 2024  
**arXiv**: [2403.14602](https://arxiv.org/abs/2403.14602)  
**领域**: 图像生成

## 一句话总结

提出 ReNoise 迭代重噪方法改进扩散模型的图像反演质量，通过在每个反演步骤多次应用 UNet 并平均预测来提升轨迹估计精度，尤其适用于少步扩散模型（SDXL Turbo、LCM）。

## 研究背景与动机

- 文本引导的扩散模型编辑需要先将真实图像反演到扩散模型域中
- 反演问题的核心困难：去噪步骤不可逆——模型训练时是 $z_t \to z_{t-1}$，但反演需要 $z_{t-1} \to z_t$
- DDIM Inversion 通过线性假设近似反演方向，但假设在步长大时误差显著
- **新挑战**：近期少步扩散模型（SDXL Turbo 仅需 1-4 步）使得每步跨度极大，传统反演方法完全失效
- 现有改进方法（Null-Text Inversion）需要耗时优化（3分钟 vs 13秒），不适合交互式编辑

## 方法详解

### 整体框架

ReNoise 是一种元算法，可与任意采样器（DDIM、Ancestral-Euler、LCM sampler）组合使用：

1. 在每个反演时间步 $t$，基于 $z_{t-1}$ 估计 $z_t$
2. 通过迭代重噪产生一系列 $z_t$ 的估计 $\{z_t^{(k)}\}_{k=1}^{\mathcal{K}+1}$
3. 对最终几次估计取加权平均，得到更精确的 $z_t$
4. 可选地加入可编辑性增强和噪声校正

### 关键设计

**1. 迭代重噪（Iterative Renoising）**

原始 DDIM Inversion 用 $\epsilon_\theta(z_{t-1}, t)$ 近似 $\epsilon_\theta(z_t, t)$，但 $z_{t-1}$ 与 $z_t$ 可能相距甚远。

ReNoise 的改进：
- 第一次近似得到 $z_t^{(1)}$（等同于 DDIM Inversion）
- 第 $k$ 次迭代：用更接近真实 $z_t$ 的 $z_t^{(k)}$ 作为 UNet 输入，获得更精确的方向估计
- $z_t^{(k+1)} = \text{InverseStep}(z_{t-1}, \epsilon_\theta(z_t^{(k)}, t))$

**关键直觉**：每次迭代都从 $z_{t-1}$ 出发，但使用更准确的方向，逐步逼近真实的 $z_t$。

**2. 预测平均**

固定点迭代可能非单调收敛，因此对最后几次估计取加权平均：

$$z_t^{(\text{avg})} = \sum_{k=1}^{\mathcal{K}} w_k \cdot z_t^{(k)}$$

平均策略有效抑制了非单调振荡带来的误差。

**3. 可编辑性增强**

反演得到的噪声预测可能偏离高斯白噪声统计特性，损害可编辑性。通过两个正则化损失改善：

- $\mathcal{L}_{\text{patch-KL}}$：鼓励预测噪声与随机噪声在 patch 级别的 KL 散度最小
- $\mathcal{L}_{\text{pair}}$：惩罚像素对之间的相关性

**4. 非确定性采样器的噪声校正**

对于 $\rho_t > 0$ 的采样器（如 DDPM、Ancestral-Euler），通过优化外部噪声 $\epsilon_t$ 弥补反演和去噪轨迹之间的差距，同时保持噪声分布特性。

### 损失函数

采样器反演的通用形式：

$$z_t = \frac{z_{t-1} - \psi_t \epsilon_\theta(z_t, t, c) - \rho_t \epsilon_t}{\phi_t}$$

其中 $\phi_t$、$\psi_t$、$\rho_t$ 是采样器特定参数。可编辑性损失 $\mathcal{L}_{\text{edit}} = \mathcal{L}_{\text{patch-KL}} + \mathcal{L}_{\text{pair}}$。

### 收敛性分析

通过 Taylor 展开分析迭代收敛条件：

$$\|\Delta^{(k+1)}\| \leq \frac{\psi_t}{\phi_t} \cdot \|\frac{\partial \epsilon_\theta}{\partial z}\|_{z_t^{(k-1)}} \cdot \|\Delta^{(k)}\| + O(\|\Delta^{(k)}\|^2)$$

实验验证缩放 Jacobian 范数始终 <1，确认算法在实践中收敛。连续估计间距离呈指数下降。

## 实验关键数据

### 主实验

固定 100 次 UNet 操作的图像重建比较（SDXL）：

| 反演步数 | 推理步数 | ReNoise步数 | L2↓ | PSNR↑ | LPIPS↓ |
|---------|---------|------------|-----|-------|--------|
| 50 | 50 | 0 | 0.00364 | 26.023 | 0.06273 |
| 75 | 25 | 0 | 0.00382 | 25.466 | 0.06605 |
| 80 | 20 | 0 | 0.00408 | 25.045 | 0.07099 |
| 90 | 10 | 0 | 0.01023 | 20.249 | 0.10305 |
| 25 | 25 | 2 | 0.00182 | **29.569** | 0.03637 |
| 20 | 20 | 3 | **0.00167** | **29.884** | **0.03633** |
| 10 | 10 | 8 | 0.00230 | 28.156 | 0.04678 |

### 消融实验

SDXL Turbo 上各组件的增量效果：

| 配置 | L2↓ | PSNR↑ | LPIPS↓ |
|------|-----|-------|--------|
| Euler Inversion | 0.0700 | 11.784 | 0.20337 |
| + 1 ReNoise | 0.0552 | 12.796 | 0.20254 |
| + 4 ReNoise | 0.0249 | 16.521 | 0.14821 |
| + 9 ReNoise | 0.0126 | 19.702 | 0.10850 |
| + Averaging | **0.0087** | 21.491 | **0.08832** |
| + Edit Losses | 0.0276 | 18.432 | 0.12616 |
| + Noise Correction (完整) | 0.0196 | **22.077** | 0.08469 |

### 关键发现

1. **ReNoise 方式优于增加反演步数**：相同 UNet 操作次数下，20步反演+3步ReNoise（PSNR=29.884）远超 80步反演+0步ReNoise（PSNR=25.045）
2. 在 SDXL Turbo（4步模型）上，ReNoise 将 PSNR 从 11.784 提升到 22.077（+87%）
3. 平均策略是关键组件——将 PSNR 从 19.702 提升到 21.491
4. 可编辑性增强损失会略微降低重建质量（PSNR 21.491 → 18.432），但保证了编辑能力
5. 噪声校正有效弥补了可编辑性损失带来的重建退化
6. 反演速度仅 13 秒，远快于 Null-Text Inversion 的 3 分钟
7. 该方法适配多种模型（SD、SDXL、SDXL Turbo、LCM）和采样器

## 亮点与洞察

- **计算效率优先的设计哲学**：不增加总操作数，而是重新分配操作（减少步数、增加每步迭代），实现高效反演
- **通用性强**：适配确定性和非确定性采样器、标准和少步模型，真正的元算法
- **理论和实践的完美结合**：从 ODE 后向 Euler 求解到固定点迭代，理论分析与实验验证一致
- **少步模型的实用反演方案**：首次让 SDXL Turbo（4步）的反演编辑成为可能

## 局限性

- 重建和可编辑性之间存在固有权衡，完美重建的潜在码往往不够可编辑
- 虽然理论分析了收敛条件，但实际中收敛速度依赖于模型和数据
- 对提示文本（prompt）敏感——不同 prompt 可能导致不同的反演质量
- 可编辑性增强的超参数（每步迭代中哪些权重 $w_k > 0$）需要调优
- 少步模型上的编辑效果虽远超先前方法，但仍不及标准 50 步模型的编辑质量

## 评分

- 创新性：⭐⭐⭐⭐ — 迭代重噪思路直观优雅
- 实用性：⭐⭐⭐⭐⭐ — 通用、快速、少步模型适配
- 表现力：⭐⭐⭐⭐ — 重建质量大幅提升
- 综合评分：8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)
- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [\[ECCV 2024\] Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [\[ECCV 2024\] Removing Distributional Discrepancies in Captions Improves Image-Text Alignment](removing_distributional_discrepancies_in_captions_improves_image-text_alignment.md)

</div>

<!-- RELATED:END -->
