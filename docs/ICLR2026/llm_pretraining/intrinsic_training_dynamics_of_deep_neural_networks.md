---
title: >-
  [论文解读] Intrinsic Training Dynamics of Deep Neural Networks
description: >-
  [ICLR 2026][预训练][intrinsic dynamics] 本文研究深度神经网络梯度流训练中，参数空间的轨迹何时可以被"提升"到低维本征空间并表示为内禀的黎曼梯度流，提出了基于守恒律的内禀可恢复性（intrinsic recoverability）准则…
tags:
  - "ICLR 2026"
  - "预训练"
  - "intrinsic dynamics"
  - "gradient flow"
  - "conservation laws"
  - "implicit bias"
  - "Riemannian metric"
---

# Intrinsic Training Dynamics of Deep Neural Networks

**会议**: ICLR 2026  
**arXiv**: [2508.07370](https://arxiv.org/abs/2508.07370)  
**代码**: 无  
**领域**: 深度学习理论 / 优化动力学  
**关键词**: intrinsic dynamics, gradient flow, conservation laws, implicit bias, Riemannian metric

## 一句话总结

本文研究深度神经网络梯度流训练中，参数空间的轨迹何时可以被"提升"到低维本征空间并表示为内禀的黎曼梯度流，提出了基于守恒律的内禀可恢复性（intrinsic recoverability）准则，并将结果推广到任意深度的 ReLU 网络和线性网络。

## 研究背景与动机

**隐式偏置的核心问题**：深度学习理论的核心挑战之一是理解梯度训练是否会促使参数趋向某些低维结构（稀疏、低秩等），即所谓的隐式偏置（implicit bias）问题。

**提升变量框架**：许多分析将参数 $\theta$ 通过架构相关的映射 $\phi$ "提升"为 $z = \phi(\theta)$，例如线性网络中 $\phi(\theta) = U_L \cdots U_1$，ReLU 网络中 $\phi(\theta) = (u_j v_j^\top)_j$。

**内在动力学的重要性**：若能证明 $z(t)$ 遵循内在Riemannian梯度流，则可以利用凸优化理论工具（如mirror flow）来分析隐式正则化效应。

**现有条件过强**：Li et al. (2022) 的commuting条件在实际中很少满足；Marcotte et al. (2023) 的involutive条件仅适用于两层ReLU网络。

**平衡初始化的限制**：线性网络中，之前的内在动力学结果依赖于严格的balanced初始化条件 $U_{i+1}^\top U_{i+1} = U_i U_i^\top$。

**缺乏统一理论**：对于一般DAG架构的深层ReLU网络、非平衡初始化的线性网络以及无限深线性网络，缺乏统一的内在动力学分析框架。

## 方法详解

### 整体框架

论文围绕一个统一的问题展开：对梯度流 $\dot{\theta}(t) = -\nabla \ell(\theta(t))$，把参数提升为 $z(t) = \phi(\theta(t))$ 后，其动力学写成 $\dot{z}(t) = -M(\theta(t)) \nabla f(z(t))$，度量矩阵 $M(\theta) = \partial\phi(\theta) \partial\phi(\theta)^\top$，问什么时候 $M(\theta(t))$ 可以只用 $z(t)$ 和初始化 $\theta_0$ 表达——一旦能，$z$ 就服从一个内禀的黎曼梯度流，便能套用凸优化工具分析隐式正则化。为此作者把"内在性"拆成三个由强到弱的层级，证明它们依次蕴含：内在可恢复性 $\Rightarrow$ 内在度量 $\Rightarrow$ 内在动力学，再分别在任意深度 ReLU 网络与松弛平衡的线性网络上落地。

### 关键设计

**1. 内在动力学性质（Definition 2.6）：把度量与任务解耦。** 最弱也最直接的诉求是，无论训练用什么损失，提升变量都遵循同一条内禀梯度流。形式上，称 $\theta_0$ 相对于 $\phi$ 具有内在动力学性质，当且仅当存在仅依赖架构与初始化的函数 $K_{\theta_0}$，使 $M(\theta(t)) = K_{\theta_0}(\phi(\theta(t)))$ 对所有可微 $f$ 成立。这样度量矩阵就不再显式依赖具体的 $\theta$ 或数据，而只是 $z$ 的函数，$z(t)$ 因此可被看作一个自洽的低维系统。

**2. 内在度量性质（Definition 2.10）：用守恒律把轨迹钉在低维流形上。** 直接要求 $K_{\theta_0}$ 对所有 $\theta$ 成立往往太苛刻，但梯度流天然带有守恒量——这正是放松条件的入口。作者引入守恒律 $\mathbf{h}$，把约束限制到流形 $\mathcal{M}_{\theta_0} = \{\theta : \mathbf{h}(\theta) = \mathbf{h}(\theta_0)\}$ 上，只需存在 $K_{\theta_0}$ 使 $M(\theta) = K_{\theta_0}(\phi(\theta))$ 对所有 $\theta \in \mathcal{M}_{\theta_0}$ 成立。由于轨迹本就被守恒律封闭在 $\mathcal{M}_{\theta_0}$ 内，内在度量性质足以推出内在动力学，却比后者更容易验证。

**3. 内在可恢复性（Definition 2.15）与等价判据（Theorem 2.17）：最强条件归结为一道线性代数题。** 最强的层级要求从 $\phi(\theta)$ 与 $\mathbf{h}(\theta)$ 的取值就能唯一还原 $\theta$，即"提升 + 守恒量"不丢信息。它的价值在于可被局部验证：Theorem 2.17 证明内在可恢复性等价于核交集条件 $\ker\partial\phi(\theta) \cap \ker\partial\mathbf{h}(\theta) = \{0\}$。这把一个关于全局动力学的抽象命题，转化为逐点检查两个雅可比矩阵零空间是否只相交于原点，从而可计算、可判定。

**4. ReLU 网络的 Frobenius 性质（Theorem 3.1 & Corollary 3.3）：用 Lie 代数判据绕开守恒律的显式构造。** 验证内在可恢复性的难点在于找全守恒律，作者改用一个代数充分条件——Frobenius 性质，即由 $\phi$ 诱导的向量场族在 Lie 括号下封闭。Theorem 3.1 证明任意 DAG 架构 ReLU 网络的 path-lifting $\phi_{\text{ReLU}}$ 在非零参数构成的稠密集上满足该性质，于是 Corollary 3.3 对几乎所有初始化都建立了最强的内在可恢复性。反直觉的是，ReLU 的分段结构使其对称群更小、守恒律更丰富，已知守恒律（对角项差）对它已是完备的（Corollary 3.4），因此比线性网络更容易建立内在动力学。

**5. 松弛平衡条件下的线性网络（Theorem 3.8 & 3.9）：把平衡初始化从一个点放宽成一族。** 线性网络以往的内在动力学结果依赖严格平衡初始化 $U_{i+1}^\top U_{i+1} = U_i U_i^\top$（对应守恒量 $S = 0$），实践中几乎不可能精确满足。作者将其放宽为松弛平衡 $S = \lambda I$，允许各层差一个标量倍数，并据此推导出内在动力学的闭式表达式（覆盖两层、深层乃至无限深的线性神经 ODE）。Theorem 3.9 进一步证明在 $r \leq \max(n,m)$ 时松弛平衡是内在度量性质的必要条件，从而在线性网络上给出了一个充要刻画。

### 损失函数 / 训练策略

本文为纯理论工作，分析连续时间梯度流。损失 $\ell(\theta) = f(\phi(\theta))$ 中的 $f$ 取任意可微函数，所有结论与具体损失形式和数据集无关。

## 实验关键数据

### 主实验

本文为纯理论贡献，核心结果以定理形式给出：

| 网络类型 | 映射 $\phi$ | 内在性质 | 条件 |
|---------|-----------|---------|------|
| 任意 DAG ReLU 网络 | $\phi_{\text{ReLU}}$ (path-lifting) | 内在可恢复性 ✓ | 非零参数（稠密集） |
| 两层线性网络 | $\phi_{\text{Lin}} = UV^\top$ | 内在度量 ✓/✗ | 松弛平衡 ✓ / 非松弛 ✗ |
| 深层线性网络 | $\phi_{\text{Lin}} = U_L \cdots U_1$ | 内在动力学 ✓ | 松弛平衡条件 |
| 线性神经ODE | 无穷深极限 | 内在动力学 ✓ | 松弛平衡 + 闭式度量 |

### 消融实验

| 比较维度 | 之前结果 | 本文推广 |
|---------|---------|---------|
| ReLU 网络深度 | 两层 | 任意 DAG 架构 |
| 线性网络初始化 | 严格平衡 $\lambda = 0$ | 松弛平衡 $S = \lambda I$ |
| 线性网络层数 | 两层 | 任意深度 + 无限深 |
| 守恒律完备性 | 经验验证 | 理论证明（Corollary 3.4） |

### 关键发现

- ReLU 网络在稠密初始化集上满足最强的内在可恢复性质
- 已知守恒律（对角项差）对 ReLU 网络是完备的
- 线性网络的松弛平衡条件是内在度量性质的充要条件
- 三层 ReLU 网络首次给出了内在动力学的闭式表达式
- 线性神经 ODE 在松弛平衡初始化下也有闭式的内在度量

## 亮点与洞察

1. **统一框架**：三层递进定义清晰揭示不同内在性概念的强弱关系
2. **ReLU 的良好性质**：反直觉地，ReLU 的非线性分段结构使得对称群更小、守恒律更丰富，比线性网络更容易建立内在动力学
3. **核包含准则的力量**：Theorem 2.14 提供了简洁工具来证明"负面结果"
4. **Lie 代数判据**：基于 Frobenius 性质的实用代数检验方法，避免直接构造守恒律
5. **跨架构适用**：统一处理了 ReLU、线性、注意力层和无限深网络

## 局限与展望

1. 仅分析连续梯度流，未覆盖离散优化算法（SGD、Adam）
2. 仅建立了内在动力学（步骤i），未推进到 mirror flow（步骤ii）
3. 线性网络在 $r > \max(n, m)$ 时仍是 open problem
4. 缺乏数值验证实验
5. Frobenius 性质在注意力层上不成立，需间接分析

## 相关工作与启发

- **Arora et al. (2019)**：balanced 初始化与守恒律 → 本文推广为 relaxed balanced
- **Marcotte et al. (2023)**：involutive 条件 → 本文弱化为 Frobenius 条件
- **Li et al. (2022)**：commuting 条件（Frobenius 的特例）→ mirror flow
- **Gonon et al. (2024)**：path-lifting 框架 → 本文证明其满足 Frobenius 性质
- **启发**：为后续分析 warped mirror flow 和实际架构的隐式偏置奠定基础

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 建立了完整的层级理论，ReLU 的普适结果是重要突破
- **实验充分度**: ⭐⭐⭐ 纯理论工作，定理严谨但缺数值验证
- **写作质量**: ⭐⭐⭐⭐⭐ 定义、定理层层递进，框架优雅清晰
- **价值**: ⭐⭐⭐⭐ 为理解隐式偏置提供重要理论基石

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Generalization Bounds for Rank-sparse Neural Networks](../../NeurIPS2025/llm_pretraining/generalization_bounds_for_rank-sparse_neural_networks.md)
- [\[ICLR 2026\] MoMa: A Simple Modular Deep Learning Framework for Material Property Prediction](moma_a_modular_deep_learning_framework_for_material_property_prediction.md)
- [\[ICML 2025\] Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks](../../ICML2025/llm_pretraining/bayesian_neural_scaling_law_extrapolation_with_prior-data_fitted_networks.md)
- [\[NeurIPS 2025\] Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks](../../NeurIPS2025/llm_pretraining/alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)
- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](../../NeurIPS2025/llm_pretraining/neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)

</div>

<!-- RELATED:END -->
