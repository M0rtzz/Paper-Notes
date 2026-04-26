---
title: >-
  [论文解读] The Spacetime of Diffusion Models: An Information Geometry Perspective
description: >-
  [ICLR 2026][图像生成][时空几何] 从信息几何视角提出扩散模型的"时空"概念，证明标准拉回几何在扩散模型中退化为直线，转而引入 Fisher-Rao 度量的时空几何，并导出可实际计算的散度编辑距离（DiffED）和转移路径采样方法。
tags:
  - ICLR 2026
  - 图像生成
  - 时空几何
  - Fisher-Rao度量
  - 拉回几何
  - 扩散编辑距离
  - 转移路径采样
---

# The Spacetime of Diffusion Models: An Information Geometry Perspective

**会议**: ICLR 2026  
**arXiv**: [2505.17517](https://arxiv.org/abs/2505.17517)  
**代码**: [GitHub](https://github.com/rafalkarczewski/spacetime-geometry)  
**领域**: 扩散模型 / 信息几何 / 理论分析  
**关键词**: 时空几何, Fisher-Rao度量, 拉回几何, 扩散编辑距离, 转移路径采样

## 一句话总结

从信息几何视角提出扩散模型的"时空"概念，证明标准拉回几何在扩散模型中退化为直线，转而引入 Fisher-Rao 度量的时空几何，并导出可实际计算的散度编辑距离（DiffED）和转移路径采样方法。

## 研究背景与动机

理解扩散模型中间噪声状态 $\mathbf{x}_t$ 的信息演化是一个开放问题：

**拉回几何的失败**：在生成模型中，通常通过拉回环境度量来研究数据的内在几何。然而在扩散模型中，这一方法存在根本问题

**缺乏对中间状态几何结构的理解**：现有工作主要聚焦采样和训练，对信息如何在噪声流程中演化缺乏分析

**需要原则性的距离和路径概念**：现有的图像相似度指标（LPIPS等）缺乏生成过程的几何基础

## 方法详解

### 1. 拉回几何的退化（核心负面结论）

**定理**：确定性 PF-ODE 解码器 $\mathbf{x}_T \mapsto \mathbf{x}_0(\mathbf{x}_T)$ 的拉回度量

$$\mathbf{G}_{\text{PB}}(\mathbf{x}_T) = \left(\frac{\partial \mathbf{x}_0}{\partial \mathbf{x}_T}\right)^\top \left(\frac{\partial \mathbf{x}_0}{\partial \mathbf{x}_T}\right)$$

导致所有测地线在数据空间中解码为**直线段**。

**原因**：扩散模型中潜在空间和数据空间维度相同，解码器在环境空间中操作，无法捕获数据流形的内在结构。

### 2. 信息几何的无记忆性问题

随机解码器（逆 SDE）的 Fisher-Rao 度量：

$$\mathbf{G}_{\text{IG}}(\mathbf{x}_T) = \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0|\mathbf{x}_T)}[\nabla_{\mathbf{x}_T}\log p(\mathbf{x}_0|\mathbf{x}_T) \nabla_{\mathbf{x}_T}\log p(\mathbf{x}_0|\mathbf{x}_T)^\top]$$

但由于无记忆性：$p(\mathbf{x}_T|\mathbf{x}_0) \approx p_T(\mathbf{x}_T)$，Fisher-Rao 度量在 $\mathbf{x}_T$ 处塌缩为零。

### 3. 潜在时空

**核心创新**：引入 $(D+1)$ 维时空 $\mathbf{z} = (\mathbf{x}_t, t) \in \mathbb{R}^D \times (0, T]$

- 索引所有噪声水平下的去噪分布族 $\{p(\mathbf{x}_0|\mathbf{x}_t)\}$
- 恢复非退化的几何结构
- 清洁数据被识别为时空点 $(\mathbf{x}, 0)$

### 4. 指数族结构与可计算能量

**命题**：去噪分布形成指数族，时空曲线能量有闭式近似：

$$\mathcal{E}(\boldsymbol{\gamma}) \approx \frac{N-1}{2}\sum_{n=0}^{N-2}(\boldsymbol{\eta}(\mathbf{z}_{n+1}) - \boldsymbol{\eta}(\mathbf{z}_n))^\top(\boldsymbol{\mu}(\mathbf{z}_{n+1}) - \boldsymbol{\mu}(\mathbf{z}_n))$$

其中自然参数和期望参数：

$$\boldsymbol{\eta}(\mathbf{x}_t, t) = \left(\frac{\alpha_t}{\sigma_t^2}\mathbf{x}_t, -\frac{\alpha_t^2}{2\sigma_t^2}\right)$$

$$\boldsymbol{\mu}(\mathbf{x}_t, t) = \left(\mathbb{E}[\mathbf{x}_0|\mathbf{x}_t], \mathbb{E}[\|\mathbf{x}_0\|^2|\mathbf{x}_t]\right)$$

**计算方式**：通过 Tweedie 公式和 Hutchinson 技巧，仅需单次 Jacobian-向量积（JVP）即可估计。

### 5. 扩散编辑距离（DiffED）

$$\text{DiffED}(\mathbf{x}^a, \mathbf{x}^b) = \ell(\boldsymbol{\gamma})$$

其中 $\boldsymbol{\gamma}$ 是连接 $(\mathbf{x}^a, 0)$ 和 $(\mathbf{x}^b, 0)$ 的时空测地线。

**直觉解释**：测地线追踪最小编辑序列——添加足够噪声以忘掉 $\mathbf{x}^a$ 的特有信息，然后去噪以引入 $\mathbf{x}^b$ 的特有信息。距离衡量沿路径去噪分布的总变化量。

### 6. 转移路径采样

对于 Boltzmann 分布 $q(\mathbf{x}) \propto \exp(-U(\mathbf{x}))$：
- 估计两个低能态间的时空测地线
- 使用退火 Langevin 动力学沿测地线采样
- 支持约束变体（低方差路径、区域回避）

## 实验

### 采样轨迹比较
- PF-ODE 路径与能量最小化测地线非常相似
- 测地线在早期采样阶段弯曲稍少

### 扩散编辑距离

| 性质 | 结果 |
|------|------|
| 与 LPIPS 相关性 | ~-7%（捕获不同信息） |
| 与 SSIM 相关性 | ~53% |
| 端点越不相似 | 中间噪声越强 |

DiffED 捕获的是结构级编辑成本，而非感知相似度。

### 转移路径采样（丙氨酸二肽）

| 方法 | MaxEnergy↓ | 能量评估次数↓ |
|------|-----------|-------------|
| MCMC-固定长度 | 42.54±7.42 | 1.29B |
| MCMC-变长 | 58.11±18.51 | 21.02M |
| Doob's Lagrangian | 66.24±1.01 | 38.4M |
| **时空测地线（本文）** | **37.36±0.60** | **16M (+16M)** |
| 下界 | 36.42 | — |

本文方法最接近下界，且能量评估次数少几个数量级。

### 约束路径
- 生成的路径有效避免高能区域
- 不像 Doob's Lagrangian 那样坍缩到单一路径

## 亮点

1. **深刻的理论洞察**：证明拉回几何在扩散模型中的根本失败
2. **时空概念的优雅性**：统一所有噪声水平的几何结构
3. **可计算**：利用指数族性质推导无模拟估计器
4. **多领域应用**：编辑距离+分子动力学
5. **计算效率**：能量估计仅需单次JVP

## 局限性

1. 时空测地线不能作为替代采样方法（需要提前知道端点）
2. 在高维数据上 Hutchinson 估计器可能引入方差
3. DiffED 的计算成本仍高于简单距离度量
4. 依赖于去噪器的质量（$\hat{\mathbf{x}}_0$ 的近似误差）
5. 转移路径采样需要已知能量函数

## 相关工作

- **黎曼几何+生成模型**：Arvanitidis (2018/2022)、Park (2023)
- **扩散模型几何**：Domingo-Enrich (2025)无记忆性分析
- **转移路径采样**：Holdijk (2023)、Doob's Lagrangian (Du 2024)
- **信息几何**：Fisher-Rao度量、Amari (2016)

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 时空几何概念极具深度和原创性
- **实用性**: ⭐⭐⭐⭐ — DiffED和转移路径有实际价值
- **实验**: ⭐⭐⭐⭐ — 理论验证充分，分子动力学结果强
- **写作**: ⭐⭐⭐⭐⭐ — 理论优雅，表达精准

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[ICLR 2026\] Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)
- [\[ICLR 2026\] When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis](when_scores_learn_geometry_rate_separations_under_the_manifold_hypothesis.md)
- [\[NeurIPS 2025\] InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](../../NeurIPS2025/image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)
- [\[NeurIPS 2025\] Information Theoretic Learning for Diffusion Models with Warm Start](../../NeurIPS2025/image_generation/information_theoretic_learning_for_diffusion_models_with_warm_start.md)

<!-- RELATED:END -->
