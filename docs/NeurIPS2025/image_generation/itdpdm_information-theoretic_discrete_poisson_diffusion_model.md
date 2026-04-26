---
title: >-
  [论文解读] ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model
description: >-
  [NeurIPS 2025][图像生成][扩散模型] 提出 ItDPDM（信息论离散泊松扩散模型），通过泊松噪声信道和泊松重建损失（PRL）实现非负离散数据的精确似然估计，避免了 ELBO 近似和 dequantization，在合成数据及 CIFAR-10 和 MIDI 音乐上取得优于现有离散扩散模型的似然估计。
tags:
  - NeurIPS 2025
  - 图像生成
  - 扩散模型
  - Poisson Process
  - information theory
  - Likelihood Estimation
  - Symbolic Music
---

# ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model

**会议**: NeurIPS 2025  
**arXiv**: [2505.05082](https://arxiv.org/abs/2505.05082)  
**代码**: 有（论文中提供链接）  
**领域**: 图像生成  
**关键词**: Discrete Diffusion, Poisson Process, information theory, Likelihood Estimation, Symbolic Music

## 一句话总结

提出 ItDPDM（信息论离散泊松扩散模型），通过泊松噪声信道和泊松重建损失（PRL）实现非负离散数据的精确似然估计，避免了 ELBO 近似和 dequantization，在合成数据及 CIFAR-10 和 MIDI 音乐上取得优于现有离散扩散模型的似然估计。

## 研究背景与动机

**领域现状**：扩散模型在连续域（图像）取得巨大成功，但处理本质离散数据（符号音乐、计数数据）时存在根本性局限：
   - 连续模型需 dequantization 将离散数据映射到连续空间，引入量化间隙和训练-测试不匹配。
   - 现有离散扩散模型（D3PM、LTJ）依赖变分下界（ELBO），非精确似然估计。

**现有痛点**：
   - 连续 DDPM 在双模、偏斜等离散分布上无法准确学习 PMF（如 NYC Taxi 分布中错失第二模态）。
   - LTJ 使用二项稀疏（binomial thinning）和 ELBO 损失——ELBO 非精确似然，且需精心调节去噪步数 $T$。
   - ELBO 引入两层近似：(1) 变分界本身的松弛，(2) Monte Carlo 积分。

**核心矛盾**：离散数据建模需要直接操作概率质量函数（PMF），而非通过概率密度函数（PDF）间接处理。

**本文目标**：为非负离散数据设计信息论上精确的扩散模型，消除 ELBO 近似。

**切入角度**：类比高斯信道上的 I-MMSE 恒等式（互信息导数=最小均方误差），建立泊松信道上的 I-MPRL 恒等式（互信息导数=最小泊松重建损失）。

**核心 idea**：I-MPRL identity 精确将泊松去噪损失与数据似然联系起来，$-\log P(x) = \int_0^\infty \text{mprl}(x, \gamma) d\gamma$。

## 方法详解

### 整体框架

- **泊松噪声信道**：给定非负输入 $x \geq 0$，输出 $z_\gamma \sim \mathcal{P}(\gamma x)$，SNR 参数 $\gamma$。
- **前向过程**：从清晰图像（$\gamma \to \infty$）逐步退化到黑色图像（$\gamma \to 0$，零光子），与高斯扩散的白噪声起点形成鲜明对比。
- **反向过程**：从黑色图像逐步恢复，通过 $z_{\gamma_{t-1}} \sim \text{Poisson}(\gamma_{t-1} \hat{x}_0)$。

### 关键设计

#### 1. 泊松重建损失（PRL）
- **功能**：替代均方误差（MSE）作为去噪损失。
- **核心公式**：
  $$l(x, \hat{x}) = x \log\frac{x}{\hat{x}} - x + \hat{x}$$
  这是 Bregman 散度的一种形式，是泊松分布对数矩母函数的凸共轭。
- **性质**（Lemma 1）：
    - 非负性：$l(x, \hat{x}) \geq 0$，当且仅当 $x = \hat{x}$ 时为零。
    - 凸性：对 $\hat{x}$ 和 $x$ 都是凸的。
    - 低估惩罚：当 $\hat{x} \to 0$ 时 $l \to \infty$，适合非负数据。
    - 最优估计器为条件期望 $E[X|Z_\gamma]$。
- **设计动机**：MSE 假设连续输出，对离散 PMF 建模引入量化误差；PRL 直接对应 PMF 的质量函数。

#### 2. I-MPRL 恒等式（核心理论）
- **功能**：建立互信息与 MPRL 的微分关系。
- **核心公式**：
  $$\frac{d}{d\gamma} I(x; z_\gamma) = \text{mprl}(\gamma)$$
  积分得精确似然：
  $$-\log P(x) = \int_0^\infty \text{mprl}(x, \gamma) d\gamma$$
- **对比高斯 I-MMSE**：$\frac{d}{d\gamma} I(x; z_\gamma) = \frac{1}{2}\text{mmse}(\gamma)$。ItDPDM 的泊松版本在结构上完全类似。
- **意义**：消除 ELBO 的第一层近似——PRL 损失与真实似然之间是精确（非变分）关系。

#### 3. NLL 上界与重要性采样
- **NLL 上界**：
  $$E[-\log P(x)] = \int_0^\infty \text{mprl}(\gamma) d\gamma \leq \int_0^\infty \mathbb{E}[l(X, \hat{X})] d\gamma$$
  上界的松弛仅来自去噪器 $\hat{X}$ 的次优性（而非变分近似）。
- **log-SNR 参数化**：$\alpha = \log \gamma$，积分变为 $\int_{-\infty}^\infty e^\alpha \text{mprl}(\alpha) d\alpha$。
- **尾部界**：解析推导积分范围 $[\alpha_0, \alpha_1]$ 外的上界，确保截断误差可控。
- **重要性采样**：用 Logistic 分布采样 $\alpha$，MC 估计误差 $O(1/\sqrt{n})$。

#### 4. 输入归一化
- 泊松噪声是非加性的（$z_\gamma \sim \mathcal{P}(\gamma x)$ 而非 $z_\gamma = x + n$），均值和方差随 $\gamma$ 增长。
- 归一化：$\tilde{Z}_\gamma = Z_\gamma / (1 + \gamma)$，保证输入在 $[0, X]$ 范围内。

### 损失函数/训练策略

- 训练损失：$L = \sum_{i \in B} \text{PRL}(x_i, \hat{x}_i) / q(\alpha)$，其中 $q(\alpha)$ 为重要性采样权重。
- 去噪器：条件 MLP（合成数据）或 U-Net/ConvTransformer（真实数据）。
- 采样：从 $z_{\gamma_T} = \mathbf{0}$（黑色图像）开始，逐步增大 SNR 采样。

## 实验关键数据

### 合成离散数据（Wasserstein-1 距离 ↓）

| 分布 | DDPM | LTJ | **ItDPDM** |
|------|------|-----|-----------|
| PoissMix | 3.76 | 1.21 | **0.99** |
| ZIP | 2.31 | 0.69 | **0.56** |
| NBinomMix | 4.89 | **1.15** | 1.39 |
| Zipf | 1.51 | 0.73 | **0.48** |
| Yule-Simon | 0.32 | 0.17 | **0.14** |

ItDPDM 在 6 个离散分布中的 4 个取得最优 WD，尤其在 Zipf（重尾）和 PoissMix 上优势明显。

### 真实数据 NLL（↓ 越低越好）

**CIFAR-10 子集**：

| 噪声+损失 | DDPM backbone | IDDPM backbone |
|-----------|:------------:|:--------------:|
| Gaussian + MSE | 0.44 | 0.48 |
| Gaussian + PRL | 0.27 | 0.32 |
| Poisson + MSE | 0.23 | 0.22 |
| **Poisson + PRL (ItDPDM)** | **0.18** | **0.17** |

**Lakh MIDI（符号音乐）**：ItDPDM NLL = **4.61×10⁻⁵** vs Gaussian+MSE NLL = 0.51。

### 生成质量

| 方法 | FID (dB↓) | SSIM ↑ | FAD ↓ | Consistency ↑ |
|------|-----------|--------|-------|---------------|
| DDPM | **0** | **0.93** | 0.89 | 0.91 |
| LTJ | 0.30 | 0.90 | 0.66 | 0.92 |
| D3PM | 2.93 | 0.86 | **0.61** | **0.98** |
| **ItDPDM** | 0.18 | 0.91 | 0.64 | 0.94 |

### 交叉训练消融

| 设置 | NLL (DDPM) |
|------|-----------|
| Gaussian + MSE | 0.44 |
| Gaussian + PRL | 0.27 |
| Poisson + MSE | 0.23 |
| **Poisson + PRL** | **0.18** |

PRL 在高斯噪声下也优于 MSE（0.27 vs 0.44），说明 PRL 对离散数据有独立价值。

### 关键发现

- I-MPRL identity 实现了精确（非变分）似然估计——消除 ELBO 的理论松弛。
- PRL 训练收敛更快，在低 SNR 区域损失更低。
- 泊松扩散在非泊松分布（Zipf、Yule-Simon）上也泛化良好，不限于泊松型数据。
- MC 估计误差在 $n > 1000$ 时约 $10^{-2}$ 级别。

## 亮点与洞察

- **精确似然**：I-MPRL identity 建立了 PRL 损失与真实 NLL 的精确等式（非上界、非下界）——这是相对于所有 ELBO 方法的本质优势。
- **处理非负整数的自然选择**：泊松噪声是计数数据的自然扰动，避免了到连续空间的映射。
- **信息论优雅性**：泊松信道的 I-MPRL 与高斯信道的 I-MMSE 完美对偶。
- **PRL 的通用性**：即使配合高斯噪声，PRL 也优于 MSE 用于离散数据。

## 局限与展望

- **概念验证阶段**：受限于训练计算，CIFAR-10 和 MIDI 上尚未达到 SOTA 生成质量（FID）。
- 仅评估了无条件生成；条件生成和 OOD 泛化未探索。
- Logistic 采样参数固定，未进行超参数搜索。
- 大规模高分辨率数据集上的验证缺失。
- 泊松噪声非加性且非源可分离，使反向去噪比高斯更复杂。

## 相关工作与启发

- **IT Gaussian Diffusion (Kong et al., 2023)**：用 I-MMSE 恒等式进行连续数据的 IT 似然估计。ItDPDM 是其离散泊松对偶。
- **LTJ (Chen & Zhou, 2023)**：二项稀疏 + ELBO 的离散扩散，不可逆且依赖变分近似。
- **D3PM (Austin et al., 2021)**：离散状态空间扩散模型，使用 masking/uniform 转移矩阵。
- **Blackout Diffusion / Beta Diffusion**：不可逆先验，无可处理似然。
- **启发**：信息论工具（互信息、信道编码）为生成模型提供了强大的分析框架，有望推广到更多离散域。

## 评分

⭐⭐⭐⭐ (4/5)

理论贡献突出（I-MPRL identity 实现精确似然），方法与高斯扩散形成优雅对偶。合成实验验证充分。主要不足是实际数据上的生成质量和规模化验证有限——作者明确指出这是 proof-of-concept。未来用更大模型和更长训练有望大幅提升。

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Information Theoretic Learning for Diffusion Models with Warm Start](information_theoretic_learning_for_diffusion_models_with_warm_start.md)
- [\[NeurIPS 2025\] Information-Theoretic Discrete Diffusion](information-theoretic_discrete_diffusion.md)
- [\[NeurIPS 2025\] MMG: Mutual Information Estimation via the MMSE Gap in Diffusion](mmg_mutual_information_estimation_via_the_mmse_gap_in_diffusion.md)
- [\[NeurIPS 2025\] BlurDM: A Blur Diffusion Model for Image Deblurring](blurdm_a_blur_diffusion_model_for_image_deblurring.md)
- [\[NeurIPS 2025\] BADiff: Bandwidth Adaptive Diffusion Model](badiff_bandwidth_adaptive_diffusion_model.md)

<!-- RELATED:END -->
