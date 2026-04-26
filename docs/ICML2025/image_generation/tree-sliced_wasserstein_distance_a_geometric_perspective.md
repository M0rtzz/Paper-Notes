---
title: >-
  [论文解读] Tree-Sliced Wasserstein Distance: A Geometric Perspective
description: >-
  [ICML2025][图像生成][Sliced Wasserstein] 提出 Tree-Sliced Wasserstein distance on Systems of Lines (TSW-SL)，用树状线系统替代 SW 中的一维直线作为投影域，保留拓扑结构的同时保持闭合解的高效计算，在梯度流、风格迁移和生成模型上超越 SW 及其变体。
tags:
  - ICML2025
  - 图像生成
  - Sliced Wasserstein
  - 树度量
  - Radon 变换
  - 最优传输
  - 生成模型
---

# Tree-Sliced Wasserstein Distance: A Geometric Perspective

**会议**: ICML2025  
**arXiv**: [2406.13725](https://arxiv.org/abs/2406.13725)  
**代码**: [GitHub](https://github.com/mint-vu/TSW-SL)  
**领域**: image_generation / 最优传输  
**关键词**: Sliced Wasserstein, 树度量, Radon 变换, 最优传输, 生成模型

## 一句话总结

提出 Tree-Sliced Wasserstein distance on Systems of Lines (TSW-SL)，用树状线系统替代 SW 中的一维直线作为投影域，保留拓扑结构的同时保持闭合解的高效计算，在梯度流、风格迁移和生成模型上超越 SW 及其变体。

## 研究背景与动机

### 领域现状

**领域现状**：OT 计算瓶颈**：最优传输（Wasserstein 距离）在支撑点数量 $n$ 上有超三次计算复杂度，难以直接应用。

### 现有痛点

**现有痛点**：SW 的局限**：Sliced Wasserstein (SW) 通过投影到一维直线利用闭合解加速，但一维投影导致输入分布拓扑信息丢失，尤其在高维时表现不佳。

### 核心矛盾

**核心矛盾**：核心动机**：寻找比一维直线更复杂、但仍允许闭合 OT 解的几何域。树度量空间恰好满足此条件——其上的 Wasserstein 距离有闭合表达式。

## 方法详解

### 整体框架

1. **树系统 (Tree System)**：将 $k$ 条 $\mathbb{R}^d$ 中的直线按树结构连接，形成树系统 $(\mathcal{L}, \mathcal{T})$。
2. **分裂映射 (Splitting Map)**：$\alpha: \mathbb{R}^d \to \Delta_{k-1}$ 将高维点的质量按比例分配到各条直线上。
3. **树系统上的 Radon 变换**：将高维分布推前到树系统上，然后在树度量下计算 OT。

### 关键设计

- **树系统的度量化 (Theorem 3.2)**：树系统上的拓扑空间 $\Omega_{\mathcal{L}}$ 可被树度量 $d_{\mathcal{L}}$ 度量化，两点间距离等于唯一路径的测度：
  $$d_{\mathcal{L}}(a,b) = \mu_{\mathcal{L}}(P_{a,b})$$
- **系统 Radon 变换 (Definition 4.1)**：
  $$\mathcal{R}^\alpha_{\mathcal{L}} f(x,l) = \int_{\mathbb{R}^d} f(y) \cdot \alpha(y)_l \cdot \delta(t_x - \langle y - x_l, \theta_l \rangle) \, dy$$
  该变换的注入性得到证明 (Theorem 4.2)。
- **TSW-SL 距离 (Definition 5.1)**：
  $$\text{TSW-SL}(\mu,\nu) = \int_{\mathbb{T}} W_{d_{\mathcal{L}},1}(\mathcal{R}^\alpha_{\mathcal{L}} f_\mu, \mathcal{R}^\alpha_{\mathcal{L}} f_\nu) \, d\sigma(\mathcal{L})$$
  通过 Monte Carlo 采样 $L$ 个树系统近似，每个树系统包含 $k$ 条直线。

### 损失/复杂度

- 时间复杂度 $O(Lkn\log n + Lkdn)$，与 SW 在相同投影方向数下等价。
- TSW-SL 是 $\mathcal{P}(\mathbb{R}^d)$ 上的度量 (Theorem 5.2)，当 $k=1$ 时退化为标准 SW。

## 实验关键数据

### 主实验：梯度流

| 方法 | Swiss Roll (iter 2500) | 25 Gaussians (iter 2500) |
|---|---|---|
| SW | 1.05e-3 | 2.20e-2 |
| MaxSW | 3.45e-3 | - |
| **TSW-SL** | **显著优于 SW** | **显著优于 SW** |

在 Swiss Roll 和 25 Gaussians 数据上，TSW-SL 在所有迭代轮次中均取得更低的 Wasserstein 距离。

### 图像风格迁移 & 生成模型

- 风格迁移：TSW-SL 生成的风格化图像质量优于 SW 及 MaxSW。
- 生成模型（MNIST/CIFAR-10）：TSW-SL 在 FID 分数上一致优于 SW 及变体，同时保持相近的计算时间。

### 消融实验

- **树系统中直线数 $k$ 的影响**：$k$ 增大通常带来更好性能，但边际收益递减。
- **分裂映射选择**：均匀分裂映射 ($\alpha = 1/k$) 已有较好效果，距离加权分裂映射可进一步提升。

## 亮点与洞察

1. **理论完备**：从拓扑结构→度量空间→Radon 变换→注入性→度量性，形成完整的数学框架。
2. **优雅的泛化**：TSW-SL 自然包含 SW 作为 $k=1$ 的特例，理论上更丰富的拓扑结构捕获更多高维信息。
3. **实用性强**：与 SW 同等计算复杂度，可直接替换现有 SW 管线中的距离度量。
4. **树度量的妙用**：利用树上 OT 的闭合解避免了通用高维 OT 的计算难题。

## 局限与展望

- 目前使用链式树结构（chain-like），更一般的树采样策略（如星形、随机树）的效果有待探索。
- 分裂映射 $\alpha$ 目前是预设的，可考虑学习最优分裂映射。
- 仅验证了 $p=1$ 的情况（树度量闭合解限制），$p>1$ 的推广需要额外工作。
- 高维高 $k$ 场景下采样效率和方差控制尚需深入分析。

## 相关工作与启发

- **Sliced Wasserstein 系列**：SW, MaxSW, Subspace SW 等通过不同投影策略改进 OT 效率。
- **Tree Wasserstein (TW)**：Le et al. (2019) 提出树度量上 OT 的闭合解，本文将其推广到连续树系统。
- **非线性投影**：Kolouri et al. (2019) 用神经网络参数化投影方向，本文从几何域角度出发互补。
- **子空间方法**：Alvarez-Melis et al. (2018), Bonet et al. (2023) 投影到低维子空间而非一维直线。
- **启发**：
    - 可进一步探索更丰富的几何域（如图、流形）上的 sliced OT 变体。
    - 树系统的随机采样策略可与注意力机制结合，自适应调整树结构。
    - 分裂映射的学习可视为一种软聚类，与混合模型有潜在联系。

## 技术细节补充

- **树系统的采样 (Algorithm 1)**：从均匀分布 $\mathcal{U}([-1,1]^d)$ 采样起点 $x_1$，$\mathcal{U}(\mathbb{S}^{d-1})$ 采样方向 $\theta_1$；后续每条线的起点在前一条线上随机选取，方向独立采样。
- **Monte Carlo 估计**：$\widehat{\text{TSW-SL}}(\mu,\nu) = \frac{1}{L}\sum_{l=1}^L W_{d_{\mathcal{L}_l},1}(\mathcal{R}^\alpha_{\mathcal{L}_l}f_\mu, \mathcal{R}^\alpha_{\mathcal{L}_l}f_\nu)$，其中每个树系统的 OT 由闭合解 (Equation 5) 直接计算。
- **离散分布的处理**：支持点被投影到树系统的各条线上，权重由 splitting map 分配，然后排序计算 TW。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Tree-Sliced Wasserstein Distance with Nonlinear Projection](tree-sliced_wasserstein_distance_with_nonlinear_projection.md)
- [\[NeurIPS 2025\] Schrödinger Bridge Matching for Tree-Structured Costs and Entropic Wasserstein Barycentres](../../NeurIPS2025/image_generation/schrödinger_bridge_matching_for_tree-structured_costs_and_entropic_wasserstein_b.md)
- [\[ICML 2025\] ContinualFlow: Learning and Unlearning with Neural Flow Matching](continualflow_learning_and_unlearning_with_neural_flow_matching.md)
- [\[ICML 2025\] Model Immunization from a Condition Number Perspective](model_immunization_from_a_condition_number_perspective.md)
- [\[ICML 2025\] Broadband Ground Motion Synthesis by Diffusion Model with Minimal Condition](broadband_ground_motion_synthesis_by_diffusion_model_with_minimal_condition.md)

<!-- RELATED:END -->
