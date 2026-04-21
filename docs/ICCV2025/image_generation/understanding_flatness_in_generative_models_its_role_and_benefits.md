---
title: >-
  [论文解读] Understanding Flatness in Generative Models: Its Role and Benefits
description: >-
  [ICCV 2025][图像生成][平坦最小值] 本文首次系统研究损失景观平坦性在生成模型（尤其是扩散模型）中的角色与优势，理论证明平坦极小值可增强对先验分布扰动的鲁棒性，实验表明 SAM 能有效提升扩散模型的平坦性，从而改善生成质量、降低暴露偏差和量化误差。
tags:
  - ICCV 2025
  - 图像生成
  - 平坦最小值
  - 扩散模型
  - SAM
  - 泛化性
  - 量化鲁棒性
  - 暴露偏差
---

# Understanding Flatness in Generative Models: Its Role and Benefits

**会议**: ICCV 2025  
**arXiv**: [2503.11078](https://arxiv.org/abs/2503.11078)  
**领域**: 扩散模型·图像生成  
**关键词**: 平坦最小值, 扩散模型, SAM, 泛化性, 量化鲁棒性, 暴露偏差  

## 一句话总结

本文首次系统研究损失景观平坦性在生成模型（尤其是扩散模型）中的角色与优势，理论证明平坦极小值可增强对先验分布扰动的鲁棒性，实验表明 SAM 能有效提升扩散模型的平坦性，从而改善生成质量、降低暴露偏差和量化误差。

## 研究背景与动机

平坦极小值（flat minima）在判别式学习中被广泛研究，已知它能增强泛化性和分布偏移鲁棒性。然而在生成模型中，平坦性的角色几乎未被探索。

核心疑问在于：**扩散模型中的平坦极小值是否会迫使模型无论输入变化都生成相似内容？** 如果是，这将与生成多样性矛盾。如果不是，平坦性如何影响生成建模？

作者发现：

**扩散模型天然就具有相当的平坦性**（可能因为训练多种噪声水平），标准SAM强度几乎无效
2. 需要**显著更强的正则化**才能在扩散模型中诱导有意义的平坦性
3. 平坦性带来的好处包括：**更好的FID**、**减少暴露偏差**、**增强量化鲁棒性**

## 方法详解

### 理论框架

**定义 1（Δ-平坦极小值）**：最小值 $\theta^*$ 满足在 $\|\delta\|_2 \leq \Delta$ 范围内损失不变。

**核心定理（Theorem 1）**：参数扰动 $\theta + \delta$ 可等价转化为先验分布的扰动：

$$\hat{p}(\mathbf{x}) = e^{-I(\mathbf{x},\delta)} p(\mathbf{x})$$

其中 $I(\mathbf{x},\delta) = \frac{1}{2}\mathbf{x}^\top(\delta\mathbf{W}^\top)\mathbf{x} + \mathbf{x}^\top\delta(\mathbf{U}^\top\mathbf{e}) + C$。

**推论 1（扩散模型版本）**：对于噪声先验 $\epsilon \sim \mathcal{N}(0,\mathbf{I})$，扰动后仍为高斯分布 $\hat{\epsilon} = \mathcal{N}(\boldsymbol{\mu}_\delta, \Sigma_\delta)$。

**定理 2（平坦性与分布鲁棒性的桥梁）**：Δ-平坦极小值具有 $\mathcal{E}$-分布间隙鲁棒性，且：

$$\mathcal{E} \leq \frac{1}{2}\left[\sum_{i=1}^d(\sigma_i - \log\sigma_i) - d + \frac{\sigma_d}{m^2}\|\mathbf{U}^\top\mathbf{e}\|_2^2 \Delta^2\right]$$

### 两个实际好处

1. **减少暴露偏差**：平坦极小值抑制了扰动估计的损失值，缓解了迭代采样中的误差累积
2. **量化鲁棒性**：量化后的参数 $\hat{\theta} = \theta + \Delta$ 可视为参数扰动，平坦性保证了损失稳定

### 优化方法比较

比较了四种促进平坦性的方法：
- **EMA**：滑动平均权重，间接平坦
- **SWA**：随机权重平均，直接追求平坦
- **IP**：输入扰动，通过 Lipschitz 条件间接平坦
- **SAM**：显式最小化锐度感知损失 $\max_{\|p\|_2 \leq \rho}[L(\theta+p) - L(\theta)]$，最有效

## 实验

### 主实验：FID 对比

| 方法 | CIFAR-10 (20步) | CIFAR-10 (100步) | LSUN Tower (100步) | FFHQ (100步) |
|------|----------------|-----------------|-------------------|-------------|
| ADM | 34.47 | 8.80 | 8.57 | 7.53 |
| +EMA | 10.63 | 4.06 | 2.49 | 6.19 |
| +SAM | 9.01 | 3.83 | 4.79 | 5.29 |
| +SAM+EMA | **7.00** | **3.18** | **2.30** | **5.04** |
| +SAM+SWA | 7.27 | **2.96** | **2.27** | 4.17 |

SAM+EMA/SWA 组合在三个数据集上一致取得最优 FID。

### 量化鲁棒性实验

| 方法 | FID (32bit) | FID (8bit) | 退化幅度 |
|------|------------|-----------|---------|
| ADM | 34.47 | 48.02 | +13.65 |
| +SAM | 9.01 | 8.94 | **-0.07** |

SAM 训练的模型在从 32 位直接转为 8 位量化时，FID 几乎不退化（-0.07 vs +13.65），展现出极强的量化鲁棒性。

### 暴露偏差分析

| 方法 | $\|\epsilon_\theta\|^2$ gap |
|------|--------------------------|
| ADM | +11.39 |
| +SAM | +3.32 |

SAM 模型的 $\|\epsilon\|^2$ gap 仅为 ADM 的约 1/3，验证了平坦性有效减少暴露偏差。

## 亮点与洞察

1. **首次系统研究**平坦性在生成模型中的角色，提供了理论保证
2. 发现扩散模型**天然平坦**——标准 SAM 强度无法产生显著效果，需要更强的正则化
3. 理论优雅地建立了**参数空间扰动→先验分布扰动**的等价关系
4. 平坦性的实际好处明确：改善 FID、减少暴露偏差、增强量化鲁棒性

## 局限性

- 理论分析基于简化的随机特征模型（random feature model），与实际深层网络有差距
- 实验仅在无条件 DDPM 上进行，未验证条件生成（如 text-to-image）
- SAM 需要双倍梯度计算，训练开销增加约 2 倍
- 仅测试了 32→8 位量化，更细粒度的量化分析缺失

## 相关工作

- 平坦性与泛化：SAM、SWA 等优化器文献
- 扩散模型暴露偏差：Input Perturbation (IP) 方法
- 模型量化：PTQ4DM、Q-Diffusion 等扩散模型量化技术

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 5 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| **综合** | **4.2** |

<!-- RELATED:START -->

## 相关论文

- [Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes](../../ICML2025/image_generation/understanding_and_mitigating_memorization_in_generative_models_via_sharpness_of_.md)
- [CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models](transformed_low-rank_adaptation_via_tensor_decomposition_and_its_applications_to.md)
- [Why Diffusion Models Don't Memorize: The Role of Implicit Regularization](../../NeurIPS2025/image_generation/why_diffusion_models_dont_memorize_the_role_of_implicit_regularization.md)
- [Compositional Scene Understanding through Inverse Generative Modeling](../../ICML2025/image_generation/compositional_scene_understanding_through_inverse_generative_modeling.md)

<!-- RELATED:END -->
