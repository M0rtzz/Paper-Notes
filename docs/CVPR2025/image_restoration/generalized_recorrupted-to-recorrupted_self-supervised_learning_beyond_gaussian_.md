---
title: >-
  [论文解读] Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise
description: >-
  [CVPR 2025][图像恢复][自监督去噪] 本文提出Generalized R2R (GR2R)，将原始R2R自监督去噪框架从高斯噪声推广到自然指数族（NEF）分布——包括Poisson/Gamma/Binomial噪声，证明GR2R损失是有监督损失的无偏估计，并且SURE可视为其特例，在低光成像和SAR等应用中达到接近监督学习的性能。
tags:
  - CVPR 2025
  - 图像恢复
  - 自监督去噪
  - 非高斯噪声
  - 自然指数族
  - 无偏估计
  - Recorrupted-to-Recorrupted
---

# Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise

**会议**: CVPR 2025  
**arXiv**: [2412.04648](https://arxiv.org/abs/2412.04648)  
**代码**: https://github.com/bemc22/GeneralizedR2R (有)  
**领域**: 图像复原  
**关键词**: 自监督去噪, 非高斯噪声, 自然指数族, 无偏估计, Recorrupted-to-Recorrupted

## 一句话总结
本文提出Generalized R2R (GR2R)，将原始R2R自监督去噪框架从高斯噪声推广到自然指数族（NEF）分布——包括Poisson/Gamma/Binomial噪声，证明GR2R损失是有监督损失的无偏估计，并且SURE可视为其特例，在低光成像和SAR等应用中达到接近监督学习的性能。

## 研究背景与动机
1. **领域现状**：深度学习图像去噪依赖干净-噪声配对数据（监督学习），但在医学成像、科学成像等领域干净数据稀缺。自监督方法兴起：Noise2Noise需要独立噪声对（难获取），Blind Spot Networks不看中心像素（次优），SURE仅适用于高斯/部分指数族噪声且需计算散度。
2. **现有痛点**：R2R通过合成重新腐蚀生成训练对，是有监督损失的无偏估计，但仅在高斯噪声上验证。实际应用中大量遇到非高斯噪声：Poisson噪声（光子计数/低光成像）、Gamma噪声（合成孔径雷达SAR）、log-Rayleigh噪声等。
3. **核心矛盾**：R2R的理论证明依赖高斯分布的特性（$\epsilon - \omega/\tau$ 和 $\epsilon + \tau\omega$ 的独立性），不能直接推广到其他噪声分布。
4. **本文目标** 将R2R推广到整个自然指数族噪声分布，提供理论保证和实用算法。
5. **切入角度**：利用NEF的可分解性质 $\boldsymbol{y} = (1-\alpha)\boldsymbol{y}_1 + \alpha\boldsymbol{y}_2$，从单个噪声观测中构造独立的噪声对。
6. **核心 idea**：利用自然指数族分布的可加性/可分性从单张噪声图像生成独立噪声对，使R2R框架适用于Poisson/Gamma等非高斯噪声。

## 方法详解

### 整体框架
给定噪声图像 $\boldsymbol{y} \sim p(\boldsymbol{y}|\boldsymbol{x})$（属于NEF），GR2R通过分布特定的重新腐蚀策略生成两个独立的噪声图像 $\boldsymbol{y}_1$ 和 $\boldsymbol{y}_2$，其中 $\boldsymbol{y}_1$ 作为输入、$\boldsymbol{y}_2$ 作为目标，用MSE或负对数似然(NLL)损失训练去噪网络 $f$。推理时通过Monte Carlo平均缓解重新腐蚀带来的额外噪声。

### 关键设计

1. **非高斯加性噪声的R2R推广（Proposition 1）**:
    - 功能：将R2R的原始腐蚀方案推广到任意加性噪声
    - 核心思路：对于 $\boldsymbol{y} = \boldsymbol{x} + \boldsymbol{\epsilon}$，仍使用原始R2R的腐蚀 $\boldsymbol{y}_1 = \boldsymbol{y} + \tau\boldsymbol{\omega}$, $\boldsymbol{y}_2 = \boldsymbol{y} - \boldsymbol{\omega}/\tau$，但证明只要 $\boldsymbol{\omega}$ 能匹配 $\boldsymbol{\epsilon}$ 的低阶矩即可近似无偏：线性估计器只需匹配二阶矩 $\mathbb{E}\omega_i^2 = \mathbb{E}\epsilon_i^2$，二次估计器额外需匹配三阶矩 $\mathbb{E}\omega_i^3 = \frac{1}{\tau}\mathbb{E}\epsilon_i^3$。
    - 设计动机：对于如log-Rayleigh这种非对称噪声，原始R2R用高斯 $\omega$（对称分布，三阶矩为0）会产生偏差。匹配三阶矩后，在log-Rayleigh噪声上PSNR从25.32提升到29.47dB。

2. **自然指数族的R2R推广（Theorem 1）**:
    - 功能：为Poisson/Gamma/Binomial等非加性噪声提供精确的无偏估计
    - 核心思路：利用NEF的可分解性质设计分布特定的腐蚀方案：对Poisson噪声 $\boldsymbol{z} \sim \mathcal{P}(\boldsymbol{x}/\gamma)$，从 $\boldsymbol{z}$ 中二项抽样 $\boldsymbol{\omega} \sim \text{Bin}(\boldsymbol{z}, \alpha)$，得到 $\boldsymbol{y}_1 = (\boldsymbol{y}-\gamma\boldsymbol{\omega})/(1-\alpha)$；对Gamma噪声 $\boldsymbol{y} \sim \mathcal{G}(\ell, \ell/\boldsymbol{x})$，用Beta分布分裂 $\boldsymbol{\omega} \sim \text{Beta}(\ell\alpha, \ell(1-\alpha))$，$\boldsymbol{y}_1 = \boldsymbol{y} \circ (1-\omega)/(1-\alpha)$。公式 $\boldsymbol{y}_2 = \frac{1}{\alpha}\boldsymbol{y} - \frac{1-\alpha}{\alpha}\boldsymbol{y}_1$ 统一适用。证明了 $\mathbb{E}_{\boldsymbol{y}|\boldsymbol{x}} \mathcal{L}_{GR2R} = \mathbb{E}_{\boldsymbol{y}_1|\boldsymbol{x}} \|f(\boldsymbol{y}_1)-\boldsymbol{x}\|_2^2$。
    - 设计动机：利用NEF分布的可分解性质是关键数学洞察——一个NEF随机变量可分解为两个独立的、同族的随机变量之和。

3. **负对数似然(NLL)损失**:
    - 功能：利用噪声分布信息提升去噪性能
    - 核心思路：MSE损失未利用噪声分布先验，NLL损失 $\mathcal{L}_{GR2R-NLL}^\alpha = \mathbb{E}\{\phi(f(\boldsymbol{y}_1)) - \boldsymbol{y}_2^\top \eta(f(\boldsymbol{y}_1))\}$ 直接最大化重新腐蚀观测的似然，对Poisson为 $-\gamma\boldsymbol{y}_2^\top\log f(\boldsymbol{y}_1) + \mathbf{1}^\top f(\boldsymbol{y}_1)$，对Gamma为 $\log f(\boldsymbol{y}_1) + \boldsymbol{y}_2/f(\boldsymbol{y}_1)$，证明其最优解仍是条件均值MMSE估计。
    - 设计动机：NLL损失在低信噪比（高噪声）下比MSE更鲁棒，因为它考虑了噪声的实际分布形状。

### 损失函数 / 训练策略
- MSE损失：$\mathcal{L}_{GR2R-MSE}^\alpha = \mathbb{E}_{\boldsymbol{y}_1,\boldsymbol{y}_2|\boldsymbol{y}} \|f(\boldsymbol{y}_1) - \boldsymbol{y}_2\|_2^2$
- NLL损失：分布特定的负对数似然
- 超参数 $\alpha$：控制 $\boldsymbol{y}_1$ 和 $\boldsymbol{y}_2$ 的信噪比分配，最优值通常在 $[0.1, 0.2]$
- 推理时Monte Carlo平均：$\hat{\boldsymbol{x}} \approx \frac{1}{J}\sum_{j=1}^J \hat{f}(\boldsymbol{y}_1^{(j)})$，$J=5\sim15$
- 网络架构：DRUnet（Poisson）和DnCNN（Gamma/Gaussian），GR2R对架构无关

## 实验关键数据

### 主实验（Poisson去噪, PSNR/SSIM on DIV2K）

| 噪声级别(γ) | PURE | Neigh2Neigh | GR2R-NLL | GR2R-MSE | 监督学习 |
|------------|------|-------------|----------|----------|---------|
| 0.01 | 32.69/0.919 | 33.37/0.929 | 33.90/0.935 | 33.92/0.935 | 33.96/0.933 |
| 0.1 | 24.37/0.631 | 28.27/0.827 | 28.30/0.827 | 28.35/0.827 | 28.39/0.827 |
| 0.5 | 22.98/0.623 | 24.90/0.651 | **25.07/0.716** | 24.69/0.698 | 25.32/0.727 |
| 1.0 | 17.94/0.469 | 23.56/0.653 | **23.69/0.658** | 23.49/0.646 | 23.85/0.668 |

### 消融实验（log-Rayleigh加性噪声, PSNR/SSIM）

| 配置 | PSNR | SSIM | 说明 |
|------|------|------|------|
| R2R (仅匹配2阶矩) | 25.32±0.79 | 0.576±0.08 | 原始R2R用高斯ω |
| **GR2R (匹配3阶矩)** | **29.47±1.51** | **0.813±0.04** | +4.15dB |
| 监督学习 | 29.93±1.50 | 0.831±0.04 | 上界 |

### 关键发现
- 匹配三阶矩带来4.15dB的巨大提升（25.32→29.47），证明了高阶矩匹配对非对称噪声的重要性
- GR2R在Poisson去噪上几乎追平监督学习（差距仅0.06-0.16dB），大幅超越PURE
- NLL损失在高噪声（低计数 $\gamma=0.5, 1.0$）下优于MSE，因为它利用了Poisson分布的形状信息
- MSE损失在低噪声（高计数 $\gamma=0.01$）下与NLL持平，因为此时Poisson近似高斯
- 最优 $\alpha$ 在 $[0.1, 0.2]$，需要在输入信噪比和目标噪声之间取平衡
- GR2R对架构无关，可直接用于任何去噪网络

## 亮点与洞察
- **理论优美性**：证明了GR2R-MSE在 $\alpha \to 0$ 时退化为SURE，建立了R2R和SURE之间的数学联系。SURE可看作GR2R的特例，而GR2R避免了SURE需计算散度项的缺点
- **NEF可分解性质的巧妙利用**：Poisson变量可通过二项分裂、Gamma变量可通过Beta分裂生成独立对，这是利用概率论经典结果解决深度学习问题的典范
- **NLL损失的引入**打破了自监督去噪只用MSE的惯例，为不同噪声分布量身定制了loss函数
- 该方法可迁移到任何NEF噪声的逆问题：MRI重建（Rician噪声）、光声成像、天文图像处理等

## 局限与展望
- 目前的Monte Carlo推理（$J$次前向传播）增加了测试时计算量，可以探索单次推理的方案
- 对于混合噪声（如Poisson-Gaussian）需要进一步推导腐蚀方案
- $\alpha$ 的最优选择目前靠经验消融，缺乏理论指导
- 对于信号依赖的噪声（如heteroscedastic Gaussian），噪声方差随信号变化，需要额外处理
- 实验数据集（DIV2K 900张训练图）相对较小，在大规模医学数据上的验证不足

## 相关工作与启发
- **vs R2R**: 原始R2R仅处理高斯噪声，GR2R推广到整个NEF，是真正的理论泛化
- **vs SURE/PURE/GSURE**: SURE需要计算散度（数值近似），PURE和GSURE各有局限；GR2R完全避免散度计算，且证明SURE是GR2R的特例
- **vs Noise2Void/Neigh2Neigh**: 这些方法通过约束网络结构（blind spot）来实现自监督，但最优解非MMSE；GR2R的解就是MMSE最优估计
- **vs Noise2Score**: 需要近似噪声分布的score函数，依赖Tweedie公式；GR2R直接从噪声样本构造训练对，更简单直接

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 理论贡献突出，将R2R从高斯推广到NEF，建立了与SURE的联系
- 实验充分度: ⭐⭐⭐⭐ 覆盖了高斯/Poisson/Gamma/log-Rayleigh噪声，但数据集有限
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，表格总结清晰，定理-命题-证明结构规范
- 价值: ⭐⭐⭐⭐ 为非高斯噪声的自监督去噪提供了统一理论框架，代码开源，实用性强

<!-- RELATED:START -->

## 相关论文

- [Rotation-Equivariant Self-Supervised Method in Image Denoising](rotation-equivariant_self-supervised_method_in_image_denoising.md)
- [Blind2Sound: Self-Supervised Image Denoising without Residual Noise](../../ICCV2025/image_restoration/blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [The Effect of Optimal Self-Distillation in Noisy Gaussian Mixture Model](../../NeurIPS2025/image_restoration/the_effect_of_optimal_self-distillation_in_noisy_gaussian_mixture_model.md)
- [MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes](../../NeurIPS2025/image_restoration/moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc.md)
- [TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation](../../ICML2025/image_restoration/timedart_a_diffusion_autoregressive_transformer_for_self-supervised_time_series_.md)

<!-- RELATED:END -->
