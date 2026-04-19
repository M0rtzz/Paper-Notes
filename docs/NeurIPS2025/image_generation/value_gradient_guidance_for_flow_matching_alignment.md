---
title: >-
  [论文解读] Value Gradient Guidance for Flow Matching Alignment
description: >-
  [NeurIPS 2025][图像生成][流匹配] 提出VGG-Flow方法，利用最优控制理论中的Hamilton-Jacobi-Bellman方程，将流匹配模型对齐问题转化为"残差速度场匹配值函数梯度"的梯度匹配任务，实现高效且保持先验分布的奖励对齐。
tags:
  - NeurIPS 2025
  - 图像生成
  - 流匹配
  - 人类偏好对齐
  - 最优控制
  - HJB方程
  - 值函数梯度
---

# Value Gradient Guidance for Flow Matching Alignment

**会议**: NeurIPS 2025  
**arXiv**: [2512.05116](https://arxiv.org/abs/2512.05116)  
**代码**: [项目页面](https://vggflow25.github.io)  
**领域**: 流匹配 / 模型对齐  
**关键词**: 流匹配, 人类偏好对齐, 最优控制, HJB方程, 值函数梯度

## 一句话总结

提出VGG-Flow方法，利用最优控制理论中的Hamilton-Jacobi-Bellman方程，将流匹配模型对齐问题转化为"残差速度场匹配值函数梯度"的梯度匹配任务，实现高效且保持先验分布的奖励对齐。

## 研究背景与动机

流匹配模型（如Stable Diffusion 3）是当前最强大的连续分布生成方法之一，广泛用于图像、视频和3D物体生成。与扩散模型不同，流匹配模型使用确定性ODE进行采样，路径更直且更容易建模。

将流匹配模型与人类偏好对齐（RLHF）面临独特挑战：

**缺乏概率流**：扩散模型每步采样是随机的，可以自然地使用随机最优控制方法。但流匹配模型的ODE采样路径是确定性的，无法直接应用扩散模型的对齐方法（如GFlowNet-based微调）。

**先验保持**：直接在计算图上最大化奖励（如ReFL、DRaFT）只能找到奖励模型的模式，不能真正对齐到目标分布，容易导致reward hacking和模式坍缩。

**Adjoint Matching虽然原理完善**，但需要将流匹配ODE转换为等价SDE并求解adjoint ODE，计算开销大。

核心矛盾：如何在保持概率正确性的前提下，高效且稳健地对齐流匹配模型？

本文从确定性最优控制出发，提出一种更高效的替代方案。

## 方法详解

### 整体框架

VGG-Flow将流匹配对齐建模为确定性最优控制问题。定义微调目标为：

$$\min_\theta \mathbb{E}_{x_0 \sim p_0, \dot{x}_t = v_\theta(x_t, t)} \left[\frac{\lambda}{2} \int_0^1 \|\tilde{v}_\theta(x_t, t)\|^2 dt - r(x_1)\right]$$

其中$\tilde{v}_\theta = v_\theta - v_{\text{base}}$为残差速度场，$\lambda$为正则化系数。目标含义：最大化终端奖励$r(x_1)$的同时，通过累积$\ell_2$代价约束微调后的速度场不要偏离基础模型太远。

### 关键设计

1. **值梯度匹配（Value Gradient Matching）**：由HJB方程的一阶条件推导出最优控制律：

$$\tilde{v}^*(x, t) = -\frac{1}{\lambda} \nabla V(x, t)$$

即最优残差速度场应等于值函数梯度的负方向。这是整个方法的核心：如果我们能准确估计值函数梯度$\nabla V(x,t)$，那么对齐问题就归结为一个简单的梯度匹配问题。

2. **值一致性方程**：将最优控制律代入HJB方程，得到值函数梯度$g_\phi(x,t) \triangleq \nabla V_\phi(x,t)$的演化方程：

$$\frac{\partial}{\partial t} g_\phi = [\nabla g_\phi]^T \left(\frac{1}{\lambda} g_\phi - v_{\text{base}}(x,t)\right) - [\nabla v_{\text{base}}(x,t)]^T g_\phi$$

带边界条件$g_\phi(x, 1) = -\nabla r(x)$。这一偏微分方程通过有限差分高效离散化。

3. **前瞻参数化（Forward-looking Parametrization）**：直接求解上述PDE需要较长时间。受DreamFusion启发，提出用单步Euler预测$\hat{x}_1$的奖励梯度加残差网络来参数化：

$$g_\phi(x, t) \triangleq -\eta_t \cdot \text{stop-gradient}(\nabla_{x_t} r(\hat{x}_1(x_t, t))) + \nu_\phi(x_t, t)$$

其中$\hat{x}_1 = x_t + (1-t) \cdot \text{stop-gradient}(v(x_t, t))$。这提供了良好的初始化（接近$t=1$时$\nu_\phi$应接近零），加速收敛。

### 损失函数 / 训练策略

总训练目标包含三部分：

$$\mathcal{L}_{\text{total}}(\theta, \phi) = \mathcal{L}_{\text{matching}}(\theta) + \mathcal{L}_{\text{consistency}}(\phi) + \alpha \mathcal{L}_{\text{boundary}}(\phi)$$

- **匹配损失**（更新$\theta$）：$\mathcal{L}_{\text{matching}} = \mathbb{E}\|\tilde{v}_\theta(x_t, t) + \beta g_\phi(x_t, t)\|^2$
- **一致性损失**（更新$\phi$）：$\mathcal{L}_{\text{consistency}}$约束$g_\phi$满足HJB梯度方程
- **边界损失**（更新$\phi$）：$\mathcal{L}_{\text{boundary}} = \mathbb{E}\|g_\phi(x_1, 1) + \nabla r(x_1)\|^2$

训练流程：模拟ODE轨迹→更新值梯度模型$g_\phi$→更新速度场$v_\theta$。使用LoRA（rank=8）在SD3的注意力层上微调，值梯度网络为缩小版SD-v1.5 U-Net。

## 实验关键数据

### Aesthetic Score对齐（400步微调）

| 方法 | Reward↑ | DreamSim多样性↑(×10⁻²) | FID↓ |
|------|---------|----------------------|------|
| Base (SD3) | 5.99 | 23.12 | 212 |
| VGG-Flow | **8.24** | **22.12** | **375** |
| ReFL | 10.00 | 5.59 | 1338 |
| DRaFT | 9.54 | 7.78 | 1518 |
| Adjoint Matching | 6.87 | 22.34 | 465 |

### 多奖励模型对比

| 方法 | HPSv2 Reward↑ | HPSv2 Diversity↑ | PickScore Reward↑ | PickScore Diversity↑ |
|------|--------------|------------------|-------------------|---------------------|
| VGG-Flow | **3.86** | **18.40** | **23.21** | **20.93** |
| ReFL | 3.87 | 14.08 | 23.19 | 17.71 |
| DRaFT | 3.76 | 15.05 | 23.00 | 19.03 |
| AM | 3.59 | 14.11 | 22.78 | 19.70 |

### 关键发现

- VGG-Flow在奖励和多样性/先验保持之间达到最佳Pareto前沿
- ReFL和DRaFT在Aesthetic Score上轻易达到9+的奖励值，但意味着基础模型先验完全丢失（FID>1300）
- VGG-Flow在相同奖励水平下，DreamSim多样性高出3-4倍，FID低3-4倍
- Adjoint Matching相比VGG-Flow收敛更慢且计算开销更大（需要4 GPU，需float32计算）
- 温度$\beta$的消融显示：更高$\beta$收敛更快但多样性和先验保持更差，$\eta_t$的时间调度对最终性能影响不大

## 亮点与洞察

- 从确定性最优控制出发是关键创新，避免了将ODE转换为SDE的额外开销（区别于Adjoint Matching）
- 前瞻参数化利用了rectified flow的近似线性性质，提供了高效的值梯度初始化
- 与PMP的联系分析揭示了HJB方法的计算优势：分摊学习$\nabla V$而非逐轨迹求解adjoint方程
- stop-gradient操作是实用的工程trick，源自DreamFusion但有理论解释

## 局限与展望

- 基于松弛目标，微调分布仅在$\lambda$较小时才良好近似KL正则化分布
- 使用有限差分和禁用二阶梯度带来不可避免的偏差
- 存在与标准RL相同的探索-利用权衡，超参数设置倾向模式坍缩
- 未探索更好的架构设计，这在大模型微调中被证明很重要

## 相关工作与启发

- 与Adjoint Matching的核心区别：AM基于随机最优控制，需要ODE→SDE转换和adjoint ODE求解；VGG-Flow直接在确定性ODE上操作
- ReFL和DRaFT是计算图截断方法，不具有概率正确性，容易reward hacking
- 最优控制在扩散模型对齐中的应用不断增多，VGG-Flow为流匹配提供了对应方案

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次将确定性最优控制的HJB方程应用于流匹配对齐，前瞻参数化设计巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ 三种奖励模型、多组消融、Pareto前沿分析，在SD3上的实验具有说服力
- **写作质量**: ⭐⭐⭐⭐ 理论推导清晰，与PMP/AM的联系讨论深入
- **价值**: ⭐⭐⭐⭐⭐ 为流匹配模型对齐提供了高效实用的方案，对SD3等大模型有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [Curly Flow Matching for Learning Non-gradient Field Dynamics](curly_flow_matching_for_learning_non-gradient_field_dynamics.md)
- [Gradient Variance Reveals Failure Modes in Flow-Based Generative Models](gradient_variance_reveals_failure_modes_in_flow-based_generative_models.md)
- [Flow Matching Neural Processes](flow_matching_neural_processes.md)
- [Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints](physics-constrained_flow_matching_sampling_generative_models_with_hard_constrain.md)
- [A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)

<!-- RELATED:END -->
