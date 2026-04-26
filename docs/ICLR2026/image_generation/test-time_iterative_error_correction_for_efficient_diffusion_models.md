---
title: >-
  [论文解读] Test-Time Iterative Error Correction for Efficient Diffusion Models
description: >-
  [ICLR 2026][图像生成][迭代误差校正] 提出 IEC（Iterative Error Correction），一种测试时的即插即用方法，通过迭代修正高效扩散模型的推理误差，将误差累积从指数增长降低为线性增长。
tags:
  - ICLR 2026
  - 图像生成
  - 迭代误差校正
  - 测试时增强
  - 量化扩散
  - 特征缓存
  - 误差传播
---

# Test-Time Iterative Error Correction for Efficient Diffusion Models

**会议**: ICLR 2026  
**arXiv**: [2511.06250](https://arxiv.org/abs/2511.06250)  
**代码**: [GitHub](https://github.com/zysxmu/IEC)  
**领域**: 扩散模型 / 模型效率 / 测试时优化  
**关键词**: 迭代误差校正, 测试时增强, 量化扩散, 特征缓存, 误差传播

## 一句话总结

提出 IEC（Iterative Error Correction），一种测试时的即插即用方法，通过迭代修正高效扩散模型的推理误差，将误差累积从指数增长降低为线性增长。

## 研究背景与动机

### 领域现状

**领域现状**：高效扩散模型（量化、特征缓存等）在部署后面临挑战：

**近似误差不可避免**：量化和缓存引入的误差会随时间步指数累积

**部署后模型不可修改**：

### 现有痛点

**现有痛点**：存储限制和部署策略使参数不可变

### 核心矛盾

**核心矛盾**：原始高精度权重可能已不可获取

### 解决思路

**解决思路**：重新执行效率流水线成本高

**现有方法是部署前方案**：时间步级量化参数、非均匀缓存策略等需要重新执行流水线

**核心问题**：能否在不重复模型效率流水线的情况下，提升已部署扩散模型的性能？

## 方法详解

### 1. 误差传播分析

DDIM 采样过程：$x_{t-1} = A_t x_t + B_t \epsilon_\theta(x_t, t)$

引入高效近似后的误差递推：

$$\delta_{t-1} = (A_t + B_t J_t) \delta_t + B_t \epsilon_\theta^\delta$$

最终累积误差：

$$\delta_0 = \sum_{i=1}^T \left(\prod_{j=i+1}^T (A_j + B_j J_j)\right)(B_i \epsilon_\theta^\delta)$$

**关键发现**：实验证明 $\|A_t + B_t J_t\| > 1$ 在所有时间步成立，意味着误差**指数放大**。

### 2. 迭代误差校正（IEC）

在每个时间步引入修正迭代：

$$x_{t-1}^{(k+1)} = x_{t-1}^{(k)} + \lambda (A_t x_t + B_t \epsilon_\theta(x_{t-1}^{(k)}, t) - x_{t-1}^{(k)})$$

等价于不动点迭代 $x_{t-1}^* = G(x_{t-1}^*)$，其中：

$$G(x) = (1-\lambda)x + \lambda(A_t x_t + B_t \epsilon_\theta(x, t))$$

### 3. 收敛性证明

利用 Banach 不动点定理：

- 映射 $G$ 的 Jacobian：$\nabla G(x) = (1-\lambda)I + \lambda B_t J_t$
- Lipschitz 常数：$L = \|(1-\lambda)I + \lambda B_t J_t\|$
- 由于 $B_t < 0$，适当的正 $\lambda$ 可确保 $L < 1$
- 实验验证 $\lambda \in [0.1, 0.7]$ 时 $\|\nabla G(x)\| < 1$ 对所有时间步成立
- 实践中设 $\lambda = 0.5$

### 4. 误差抑制效果

IEC 收敛后每步误差有界：$\|\delta_{t-1}^{(\infty)}\| \leq \frac{C}{1-L}$

关键：IEC 消除了对前一时间步误差的依赖，使总累积误差从指数增长变为线性增长：$\delta_0^{\text{IEC}} = \sum_{j=1}^T \delta_j^x$

### 5. 实际使用

- 最大迭代次数 K=1（实际仅需1次额外前向传递）
- 阈值 $\tau = 10^{-5}$
- 可选择性应用于部分时间步
- 对于量化方法：每个时间步应用
- 对于缓存方法：仅在非缓存时间步应用

## 实验

### 设置
- 模型：DDPM、LDM、Stable Diffusion
- 效率技术：量化（W4A8/W8A8）、DeepCache、CacheQuant
- 数据集：CIFAR-10、LSUN-Churches、LSUN-Bedrooms、ImageNet、MS-COCO
- 指标：FID、IS、CLIP Score

### 主要结果（量化 + IEC）


### 主实验

| 数据集 | 精度 | 基线 FID | +IEC FID | 改善 |
|--------|------|---------|---------|------|
| CIFAR-10 | W8A8 | 较高 | 显著降低 | 大幅改善 |
| CIFAR-10 | W4A8 | 很高 | 明显降低 | 大幅改善 |
| LSUN-Churches | W8A8 | 较高 | 降低 | 改善 |
| LSUN-Bedrooms | W8A8 | 较高 | 降低 | 改善 |

### DeepCache + IEC


### 消融实验

| 数据集 | 缓存策略 | 基线 FID | +IEC FID |
|--------|---------|---------|---------|
| CIFAR-10 | N=10 | 较高 | 降低 |
| ImageNet | N=10 | 较高 | 降低 |

### CacheQuant + IEC
在量化+缓存的混合效率方案上同样有效。

### Stable Diffusion 上的结果
- MS-COCO 上 FID 和 CLIP Score 均有改善
- 仅在第一步应用 IEC 即可获得显著提升

### 灵活性分析

| 策略 | 效果 |
|------|------|
| 不使用 IEC | 基线性能 |
| 首尾各A步 | 可调节的质量-效率权衡 |
| 所有步应用 | 最大质量提升 |

通过调节应用 IEC 的时间步数量，用户可以细粒度控制效率-质量权衡。

## 亮点与洞察

1. **理论严谨**：从误差传播分析到收敛性证明的完整理论链
2. **即插即用**：无需重训、无需架构修改、无需原始模型
3. **广泛适用**：跨不同效率技术（量化、缓存、混合）均有效
4. **灵活可控**：用户可自由选择应用程度来权衡效率与质量
5. **测试时方法的新思路**：借鉴测试时缩放理念应用于生成模型

## 局限与展望

1. 每次 IEC 迭代需要额外的前向传递，增加推理时间
2. 理论分析基于 DDIM，对其他采样器（如 DPM-Solver）的适用性需进一步验证
3. $\lambda$ 的最优值可能因模型和数据而异
4. 对于误差极大的极低位量化（如 W2），IEC 的改善可能有限
5. 未讨论与测试时训练方法的关系

## 相关工作

- **扩散模型量化**：PTQ4DM、Q-Diffusion、TDQ
- **特征缓存**：DeepCache、CacheQuant
- **测试时缩放**：TTT (Snell 2024)、REPA
- **高效采样**：DDIM、DPM-Solver、一致性模型

## 评分

- **创新性**: ⭐⭐⭐⭐ — 指数到线性的误差抑制，理论贡献清晰
- **实用性**: ⭐⭐⭐⭐⭐ — 部署后优化，真正的即插即用
- **实验**: ⭐⭐⭐⭐ — 跨模型、跨技术、跨数据集验证
- **写作**: ⭐⭐⭐⭐ — 理论推导严谨，实验设置合理

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model](vfscale_intrinsic_reasoning_through_verifier-free_test-time_scalable_diffusion_m.md)
- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [\[ICLR 2026\] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)
- [\[CVPR 2026\] Physics-Consistent Diffusion for Efficient Fluid Super-Resolution via Multiscale Residual Correction](../../CVPR2026/image_generation/physics-consistent_diffusion_for_efficient_fluid_super-resolution_via_multiscale.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)

<!-- RELATED:END -->
