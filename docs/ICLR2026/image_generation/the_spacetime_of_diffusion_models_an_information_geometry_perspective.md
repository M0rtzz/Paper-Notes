---
title: >-
  [论文解读] The Spacetime of Diffusion Models: An Information Geometry Perspective
description: >-
  [ICLR 2026][图像生成][时空几何] 从信息几何视角提出扩散模型的"时空"概念，证明标准拉回几何在扩散模型中退化为直线，转而引入 Fisher-Rao 度量的时空几何，并导出可实际计算的散度编辑距离（DiffED）和转移路径采样方法。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "时空几何"
  - "Fisher-Rao度量"
  - "拉回几何"
  - "扩散编辑距离"
  - "转移路径采样"
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

### 整体框架

本文的出发点是为扩散模型的中间噪声状态找一个非退化的几何结构：先证明两种自然做法（确定性解码器的拉回度量、随机解码器的 Fisher-Rao 度量）都在数据空间退化，再把噪声水平 $t$ 显式纳入坐标，构造 $(D+1)$ 维"潜在时空" $\mathbf{z}=(\mathbf{x}_t,t)$。在这个时空里去噪分布族形成指数族，曲线能量有闭式近似从而可实际计算，由此派生出图像间的扩散编辑距离（DiffED）和分子构象间的转移路径采样两个应用。

### 关键设计

**1. 拉回几何的退化：解释为什么不能直接拉回环境度量。** 研究数据内在几何的标准套路是把环境度量沿解码器拉回，但确定性 PF-ODE 解码器 $\mathbf{x}_T \mapsto \mathbf{x}_0(\mathbf{x}_T)$ 在扩散模型里行不通。本文证明其拉回度量

$$\mathbf{G}_{\text{PB}}(\mathbf{x}_T) = \left(\frac{\partial \mathbf{x}_0}{\partial \mathbf{x}_T}\right)^\top \left(\frac{\partial \mathbf{x}_0}{\partial \mathbf{x}_T}\right)$$

会让所有测地线在数据空间中解码成**直线段**。根源在于扩散模型的潜在空间与数据空间同维，解码器始终在环境空间里操作，根本没有机会编码数据流形的内在弯曲，于是几何信息被抹平。这是一个核心负面结论，直接堵死了"照搬生成模型几何分析"的路。

**2. 信息几何的无记忆性塌缩：第二条自然路径同样失败。** 换用随机解码器（逆 SDE），其 Fisher-Rao 度量为

$$\mathbf{G}_{\text{IG}}(\mathbf{x}_T) = \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0|\mathbf{x}_T)}[\nabla_{\mathbf{x}_T}\log p(\mathbf{x}_0|\mathbf{x}_T)\, \nabla_{\mathbf{x}_T}\log p(\mathbf{x}_0|\mathbf{x}_T)^\top]$$

看似能捕获概率流形的曲率，但在最大噪声水平 $\mathbf{x}_T$ 处，前向过程已经"忘掉"了数据，即 $p(\mathbf{x}_T|\mathbf{x}_0) \approx p_T(\mathbf{x}_T)$，去噪分布几乎不随 $\mathbf{x}_T$ 变化，于是 Fisher-Rao 度量整体塌缩为零。两条路都死在同一个地方：只盯着单一噪声水平的切片，信息要么被抹直、要么被噪声淹没。

**3. 潜在时空：把噪声水平升格为坐标轴。** 核心创新是不再固定某个 $t$，而是引入 $(D+1)$ 维时空 $\mathbf{z}=(\mathbf{x}_t,t)\in\mathbb{R}^D\times(0,T]$，让一个点同时携带状态和它所处的噪声水平。这样整族去噪分布 $\{p(\mathbf{x}_0|\mathbf{x}_t)\}$ 被同一坐标系索引起来，干净数据 $\mathbf{x}$ 被识别为时空底面上的点 $(\mathbf{x},0)$。沿时间轴移动相当于在不同噪声水平间切换观察尺度，几何结构因此重新变得非退化——设计 1、2 中"被抹平/被塌缩"的信息，正是被压在了被忽略的时间维度里，显式加回来就恢复了。

**4. 指数族结构与可计算能量：让抽象几何能真的算出来。** 时空之所以有用，关键在于去噪分布沿这条曲线形成一个指数族，于是时空曲线 $\boldsymbol\gamma$ 的能量有闭式近似

$$\mathcal{E}(\boldsymbol{\gamma}) \approx \frac{N-1}{2}\sum_{n=0}^{N-2}(\boldsymbol{\eta}(\mathbf{z}_{n+1}) - \boldsymbol{\eta}(\mathbf{z}_n))^\top(\boldsymbol{\mu}(\mathbf{z}_{n+1}) - \boldsymbol{\mu}(\mathbf{z}_n))$$

其中自然参数 $\boldsymbol{\eta}(\mathbf{x}_t,t)=\left(\tfrac{\alpha_t}{\sigma_t^2}\mathbf{x}_t,\,-\tfrac{\alpha_t^2}{2\sigma_t^2}\right)$、期望参数 $\boldsymbol{\mu}(\mathbf{x}_t,t)=\left(\mathbb{E}[\mathbf{x}_0|\mathbf{x}_t],\,\mathbb{E}[\|\mathbf{x}_0\|^2|\mathbf{x}_t]\right)$ 都能由去噪器读出。借助 Tweedie 公式拿到一阶矩、再用 Hutchinson 技巧估二阶矩，整条曲线能量只需单次 Jacobian-向量积（JVP）即可无模拟地估计，把"求测地线"从积分微分方程降成了可优化的离散和。

**5. 扩散编辑距离 DiffED：把图像相似度变成几何距离。** 有了可算能量，就能定义两张图之间的距离

$$\text{DiffED}(\mathbf{x}^a, \mathbf{x}^b) = \ell(\boldsymbol{\gamma})$$

其中 $\boldsymbol{\gamma}$ 是连接时空点 $(\mathbf{x}^a,0)$ 与 $(\mathbf{x}^b,0)$ 的测地线、$\ell$ 是其长度。这条测地线对应一段最经济的编辑序列：先加噪到足够忘掉 $\mathbf{x}^a$ 的专有信息，再去噪逐步引入 $\mathbf{x}^b$ 的专有信息，距离量化的是沿途去噪分布的总变化量。因此 DiffED 衡量的是结构层面的"改写成本"，而非 LPIPS 那种感知相似度，两张图越不像、测地线就要爬到越高的噪声水平才能完成迁移。

**6. 转移路径采样：同一框架迁移到分子动力学。** 同样的时空测地线可直接用于在 Boltzmann 分布 $q(\mathbf{x})\propto\exp(-U(\mathbf{x}))$ 的两个低能态之间找过渡路径：先估计连接两态的时空测地线作为"骨架"，再沿这条骨架跑退火 Langevin 动力学采样具体路径。该框架天然支持约束变体——比如要求低方差路径或回避指定区域——只需在测地线求解时加入相应约束，无需重训模型，从而把图像编辑里推出的几何机制平移到了构象转移问题上。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[ICLR 2026\] Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[ICML 2026\] Geometry-Aware Tabular Diffusion](../../ICML2026/image_generation/geometry-aware_tabular_diffusion.md)
- [\[ICLR 2026\] When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis](when_scores_learn_geometry_rate_separations_under_the_manifold_hypothesis.md)

</div>

<!-- RELATED:END -->
