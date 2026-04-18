---
title: >-
  [论文解读] Tree-Sliced Wasserstein Distance with Nonlinear Projection
description: >-
  [ICML2025][图像生成][最优传输] 提出非线性投影框架下的 Tree-Sliced Wasserstein（TSW）距离，通过 Circular/Spatial 两种非线性 Radon 变换替代原有线性投影，在保持度量良定义和单射性的同时，在梯度流、自监督学习和生成模型等任务上显著优于已有 SW 和 TSW 变体。
tags:
  - ICML2025
  - 图像生成
  - 最优传输
  - Tree-Sliced Wasserstein距离
  - 非线性投影
  - Radon变换
  - 概率度量
  - 球面度量
---

# Tree-Sliced Wasserstein Distance with Nonlinear Projection

**会议**: ICML2025  
**arXiv**: [2505.00968](https://arxiv.org/abs/2505.00968)  
**代码**: [thanhqt2002/NonlinearTSW](https://github.com/thanhqt2002/NonlinearTSW)  
**领域**: image_generation  
**关键词**: 最优传输, Tree-Sliced Wasserstein距离, 非线性投影, Radon变换, 概率度量, 球面度量

## 一句话总结
提出非线性投影框架下的 Tree-Sliced Wasserstein（TSW）距离，通过 Circular/Spatial 两种非线性 Radon 变换替代原有线性投影，在保持度量良定义和单射性的同时，在梯度流、自监督学习和生成模型等任务上显著优于已有 SW 和 TSW 变体。

## 研究背景与动机
**最优传输（OT）** 提供了概率测度空间上几何意义明确的度量，但计算复杂度高。**Sliced Wasserstein（SW）距离** 利用一维 OT 的闭式解降低复杂度，通过 Radon 变换将高维度量投影到一维线上。

**Tree-Sliced 方法** 是 SW 的替代方案：
- 用树形度量空间替代一维线
- 引入分裂机制（splitting map）将测度分配到多条连接的线上
- 更好捕获积分域的拓扑结构，同时保持低计算开销

**现有限制**：Tree-Sliced 框架仍局限于线性投影（超平面积分域），而 SW 中已有多种非线性投影增强方案（Circular RT、Spatial RT 等）。

**核心问题**：能否将非线性投影框架引入 Tree-Sliced 方法，兼得树结构优势和非线性投影的表达能力？

## 方法详解

### 基础回顾

**Radon 变换**：$\mathcal{R}f(t, \theta) = \int_{\mathbb{R}^d} f(y) \cdot \delta(t - \langle y, \theta \rangle) dy$

**广义 Radon 变换（GRT）**：用一般函数 $g(y, \psi)$ 替代内积 $\langle y, \theta \rangle$，投影沿超曲面而非超平面

**空间 Radon 变换（SRT）**：先用单射连续映射 $h: \mathbb{R}^d \to \mathbb{R}^{d_\theta}$ 变换空间，再做线性投影

### 1. 圆形 Radon 变换在线系统上的推广（CRTSL）

$$\mathcal{CR}^{\alpha}_{\mathcal{L},r} f(x_i + t \cdot \theta_i) = \int_{\mathbb{R}^d} f(y) \cdot \alpha(y, \mathcal{L})_i \cdot \delta(t - \|y - x_i - r\theta_i\|_2) \, dy$$

- 用欧氏距离函数替代内积投影，积分域从超平面变为（超）球面
- $\alpha(y, \mathcal{L})_i$ 是分裂映射，将函数 $f$ 的质量分配到线系统的各条线上
- 参数 $r$ 控制圆心偏移

### 2. 空间 Radon 变换在线系统上的推广（SRTSL）

$$\mathcal{H}^{\alpha}_{\mathcal{L}} f(x_i + t \cdot \theta_i) = \int_{\mathbb{R}^d} f(y) \cdot \alpha(h(y), \mathcal{L})_i \cdot \delta(t - \langle h(y) - x_i, \theta_i \rangle) \, dy$$

- 先用单射映射 $h$ 将数据变换到新空间 $\mathbb{R}^{d_\theta}$，再在新空间做 Tree-Sliced
- $h$ 的选择：逐分量奇数多项式 $h(x_1,...,x_d) = (f_1(x_1),...,f_d(x_d))$（如 $f_i(x)=x+x^3$），或拼接神经网络 $h(x) = (x, \phi(x))$

### 单射性保证

**定理 4.2**：若分裂映射 $\alpha$ 是 $\text{E}(d)$-不变的，则 CRTSL 是单射的。

**定理 4.3**：若分裂映射 $\alpha$ 是 $\text{E}(d_\theta)$-不变的，则 SRTSL 是单射的。

分裂映射选择基于距离的 softmax：$\alpha(x, \mathcal{L})_l = \text{softmax}(\{d(x, \mathcal{L})_i\}_{i=1}^k)$，天然满足 $\text{E}(d)$-不变性。

### 3. 新距离定义

**CircularTSW**：$\text{CircularTSW}(\mu, \nu) = \int_{\mathbb{T}^d_k} \text{W}(\mathcal{CR}^{\alpha}_{\mathcal{L},r} f_\mu, \mathcal{CR}^{\alpha}_{\mathcal{L},r} f_\nu) \, d\sigma(\mathcal{L})$

**SpatialTSW**：$\text{SpatialTSW}(\mu, \nu) = \int_{\mathbb{T}^{d_\theta}_k} \text{W}(\mathcal{H}^{\alpha}_{\mathcal{L}} f_\mu, \mathcal{H}^{\alpha}_{\mathcal{L}} f_\nu) \, d\sigma(\mathcal{L})$

**定理 5.3**：CircularTSW 和 SpatialTSW 都是 $\mathcal{P}(\mathbb{R}^d)$ 上的度量。

### 4. CircularTSW$_{r=0}$ 的计算优势
当 $r=0$ 时，所有支撑点在树的 $k$ 条线上有相同投影坐标，排序只需做一次（而非 $k$ 次），复杂度从 $O(Lkn\log n)$ 降至 $O(Ln\log n + Lkd_\theta n)$。

### 5. 球面扩展（SpatialSTSW）
将框架推广到球面 $\mathbb{S}^d$ 上的测度，使用球面树（Spherical Trees）和 $\text{O}(d_\theta+1)$-不变分裂映射。

## 实验关键数据

### 计算效率（运行时间对比）

| 方法 | 相对 SW 速度 |
|------|------------|
| SW（vanilla） | 1× |
| Db-TSW | 较慢 |
| CircularTSW | 略慢 |
| **CircularTSW$_{r=0}$** | **接近 SW** |

CircularTSW$_{r=0}$ 是唯一能接近 vanilla SW 速度的 Tree-Sliced 方法。

### 应用表现（梯度流、生成模型等）
- 在梯度流实验中，SpatialTSW 和 CircularTSW 收敛速度和质量均优于 SW 和线性 TSW
- 在去噪扩散 GAN 和自监督学习任务上显著优于近期 SW/TSW 变体
- 球面数据上，SpatialSTSW 优于 Spherical SW 和 Spherical TSW

### 关键发现
- 非线性投影显著增强了 Tree-Sliced 距离的度量效果，尤其在高维数据上
- CircularTSW$_{r=0}$ 在树系统框架下表现良好，但在原始 Sliced 设置下表现差——证明了树结构和分裂映射的必要性
- 拼接神经网络的 $h(x)=(x,\phi(x))$ 带来可学习参数，性能-计算权衡灵活

## 亮点与洞察
1. **非线性投影与树结构的结合**是本文核心创新——两者各有优势，组合后效果显著
2. **CRTSL 和 SRTSL 的单射性证明**为度量的良定义提供了坚实理论保障
3. **CircularTSW$_{r=0}$ 的计算技巧**精巧：利用 $r=0$ 时坐标相同的特性避免重复排序，实现接近 vanilla SW 的速度
4. **球面扩展**体现了框架的通用性，不局限于欧氏空间
5. 分裂映射使用基于距离的 softmax，简洁优雅且自动满足群不变性

## 局限与展望
1. 非线性投影函数 $h$ 的选择对性能影响大，但缺乏系统化的选择指南
2. 理论上 CRTSL 的单射性依赖 $\text{E}(d)$-不变性假设，实际实现中 softmax 近似可能引入误差
3. 实验主要在中小规模数据上验证，大规模高维数据（如百万级图像）的扩展性未充分评估
4. 与其他 OT 近似方法（如 Sinkhorn、entropic OT）的理论和实验比较不足

## 相关工作与启发
- **Tran et al., 2025c/b**：Tree-Sliced 框架先驱，本文的直接基础
- **Kolouri et al., 2019**：广义 Radon 变换在 SW 中的应用（Generalized SW）
- **Chen et al., 2022**：空间 Radon 变换（Augmented SW）
- **Bonneel et al., 2015**：经典 Sliced Wasserstein 距离
- 启示：度量设计中"投影方式"和"度量空间结构"是两个独立可优化的维度

## 评分
- 新颖性: ⭐⭐⭐⭐ （非线性投影+树结构的组合，理论框架完整）
- 实验充分度: ⭐⭐⭐⭐ （梯度流、生成模型、自监督学习、球面数据全覆盖）
- 写作质量: ⭐⭐⭐⭐ （数学严谨、符号清晰，但公式密度较高）
- 价值: ⭐⭐⭐⭐ （推进了 Sliced OT 领域的前沿，开源代码可复现）
