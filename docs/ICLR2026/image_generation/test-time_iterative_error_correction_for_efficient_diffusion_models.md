---
title: >-
  [论文解读] Test-Time Iterative Error Correction for Efficient Diffusion Models
description: >-
  [ICLR 2026][图像生成][迭代误差校正] 提出 IEC（Iterative Error Correction），一种测试时的即插即用方法，通过迭代修正高效扩散模型的推理误差，将误差累积从指数增长降低为线性增长。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "迭代误差校正"
  - "测试时增强"
  - "量化扩散"
  - "特征缓存"
  - "误差传播"
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

### 整体框架

IEC 把高效扩散模型的"近似误差为什么会失控"先讲清楚，再给出对症的解法：在 DDIM 的每一步采样里插入一个轻量的不动点修正循环，让当前步先收敛到自洽解再往下走，从而切断误差在时间步之间的级联放大。整套方法不碰模型权重、不改架构、不需要原始高精度模型，纯粹是测试时的即插即用补丁。

### 关键设计

**1. 误差传播分析：先证明误差为什么会指数爆炸。** DDIM 的单步更新写成 $x_{t-1} = A_t x_t + B_t \epsilon_\theta(x_t, t)$，量化或缓存引入的近似让每一步多出一个扰动 $\epsilon_\theta^\delta$，于是状态误差满足递推 $\delta_{t-1} = (A_t + B_t J_t)\delta_t + B_t \epsilon_\theta^\delta$，其中 $J_t$ 是噪声预测网络对输入的 Jacobian。把递推展开到第 0 步，得到累积误差 $\delta_0 = \sum_{i=1}^T \big(\prod_{j=i+1}^T (A_j + B_j J_j)\big)(B_i \epsilon_\theta^\delta)$。这个连乘项是问题的根源——作者实测发现 $\|A_t + B_t J_t\| > 1$ 在所有时间步都成立，意味着早期的小误差会被后续步骤反复放大，最终呈指数级累积。这一步分析没有给出解法，但把"误差耦合在相邻时间步之间"这个病灶定位出来，后面的设计正是冲着它去的。

**2. 迭代误差校正：用不动点迭代把每一步逼到自洽。** 既然误差靠时间步之间的耦合放大，IEC 的做法是在每个时间步内部反复修正，直到当前步的预测和它自己的输出一致。具体地，固定 $x_t$，对 $x_{t-1}$ 做迭代 $x_{t-1}^{(k+1)} = x_{t-1}^{(k)} + \lambda\big(A_t x_t + B_t \epsilon_\theta(x_{t-1}^{(k)}, t) - x_{t-1}^{(k)}\big)$，步长 $\lambda$ 控制每次修正的幅度。这等价于求映射 $G(x) = (1-\lambda)x + \lambda\big(A_t x_t + B_t \epsilon_\theta(x, t)\big)$ 的不动点 $x_{t-1}^* = G(x_{t-1}^*)$。直觉上，原本的一次性单步预测会把近似误差直接写进 $x_{t-1}$ 带向下一步，而迭代到自洽解相当于让这一步内部先"自我纠偏"，不把未消化的误差外溢出去。

**3. 收敛性证明：说明迭代为什么一定收敛而不是发散。** 迭代要有意义，$G$ 必须是压缩映射。$G$ 的 Jacobian 为 $\nabla G(x) = (1-\lambda)I + \lambda B_t J_t$，对应 Lipschitz 常数 $L = \|(1-\lambda)I + \lambda B_t J_t\|$，由 Banach 不动点定理只要 $L < 1$ 迭代就唯一收敛。关键观察是 DDIM 中 $B_t < 0$，因此选一个适当的正 $\lambda$ 能把 $(1-\lambda)I$ 和 $\lambda B_t J_t$ 两项抵消到单位球内；作者实测 $\lambda \in [0.1, 0.7]$ 时 $\|\nabla G(x)\| < 1$ 对所有时间步都成立，实践中取 $\lambda = 0.5$。这一步把第 2 点的"启发式修正"坐实成了有理论保证的收敛过程。

**4. 误差抑制效果：从指数增长降到线性增长。** 收敛之后每一步的残余误差被压到有界范围 $\|\delta_{t-1}^{(\infty)}\| \leq \frac{C}{1-L}$。更重要的是，因为每一步都被独立逼到自洽解，IEC 切断了 $\delta_{t-1}$ 对前一步 $\delta_t$ 的依赖——总累积误差不再是第 1 点里那个连乘项，而退化成各步独立误差的简单求和 $\delta_0^{\text{IEC}} = \sum_{j=1}^T \delta_j^x$。这正是 IEC 的核心收益：误差累积从指数增长被根治为线性增长，与第 1 点的分析首尾呼应。

**5. 实际使用：一次额外前向就够，且可按需施加。** 落地时迭代成本极低，最大迭代次数取 $K=1$（实际只需 1 次额外前向传递），配合阈值 $\tau = 10^{-5}$ 做早停。施加位置可以灵活选择：量化方法在每个时间步都用，缓存方法只在非缓存时间步用（缓存步本身没有重算误差），混合方案两者结合。这种选择性应用让用户能在质量和额外开销之间细粒度权衡，而不必对每一步都付出代价。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model](vfscale_intrinsic_reasoning_through_verifier-free_test-time_scalable_diffusion_m.md)
- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [\[ICML 2026\] Linearizing Vision Transformer with Test-Time Training](../../ICML2026/image_generation/linearizing_vision_transformer_with_test-time_training.md)
- [\[ICLR 2026\] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)
- [\[ICML 2026\] Quantifying Error Propagation and Model Collapse in Diffusion Models](../../ICML2026/image_generation/quantifying_error_propagation_and_model_collapse_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
