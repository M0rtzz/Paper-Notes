---
title: >-
  [论文解读] Stable Minima of ReLU Neural Networks Suffer from the Curse of Dimensionality: The Neural Shattering Phenomenon
description: >-
  [NeurIPS 2025 Spotlight][物理/科学计算][稳定极小值] 本文研究了两层过参数化 ReLU 网络中稳定极小值（flat minima）的泛化性质，证明虽然平坦性确实蕴含泛化，但其收敛速率随输入维度指数级恶化（即存在维度灾难），与不受维度灾难影响的低范数解（weight decay）形成指数级分离；并揭示了"neural shattering"现象作为高维失败的几何机制。
tags:
  - "NeurIPS 2025 Spotlight"
  - "物理/科学计算"
  - "稳定极小值"
  - "ReLU网络"
  - "维度灾难"
  - "隐式偏置"
  - "非参数估计"
---

# Stable Minima of ReLU Neural Networks Suffer from the Curse of Dimensionality: The Neural Shattering Phenomenon

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2506.20779](https://arxiv.org/abs/2506.20779)  
**代码**: 无  
**领域**: 科学计算  
**关键词**: 稳定极小值, ReLU网络, 维度灾难, 隐式偏置, 非参数估计

## 一句话总结
本文研究了两层过参数化 ReLU 网络中稳定极小值（flat minima）的泛化性质，证明虽然平坦性确实蕴含泛化，但其收敛速率随输入维度指数级恶化（即存在维度灾难），与不受维度灾难影响的低范数解（weight decay）形成指数级分离；并揭示了"neural shattering"现象作为高维失败的几何机制。

## 研究背景与动机
**领域现状**: 深度学习中过参数化模型存在无穷多全局极小值，梯度下降(GD)似乎能找到泛化良好的解。大量工作从动态稳定性(dynamical stability)角度研究 GD 的隐式偏置——稳定极小值是 GD 能"稳定收敛"到的那些极小值，其特征是 loss 曲率有界（即 Hessian 最大特征值 $\leq 2/\eta$）。

**现有痛点**: 已有工作（Mulayoff et al., 2021; Qiao et al., 2024）要么假设插值(interpolation)，要么局限于单变量输入。Qiao et al. (2024) 在单变量情形给出了良好的风险界，但仅限于数据支撑严格内部的区间，且未涉及多变量/高维情形。

**核心矛盾**: 平坦极小值在低维（d=1）下表现良好，但在高维情形下的行为完全未知。直觉上，稳定性/平坦性应该蕴含某种正则性从而帮助泛化——但这个正则性在高维下是否足够？

**本文目标**: 精确回答"两层过参数化 ReLU 网络的稳定极小值在高维非插值情形下的泛化表现如何"这一根本问题。

**切入角度**: 建立稳定极小值的函数空间刻画（加权变分范数），然后分别推导泛化间隙和 MSE 的上下界，并通过基于球面覆盖的新型极小极大下界构造揭示几何机制。

**核心 idea**: 高维球面上存在指数多的"方向帽"(spherical caps)，使得 ReLU 神经元可以仅激活极少数据点却保持大权重，从而"欺骗"平坦性标准——这就是 neural shattering，它导致稳定极小值在高维下不可避免地遭遇维度灾难。

## 方法详解

### 整体框架
本文是一篇理论分析论文，核心工作是：
1. 建立稳定极小值的函数空间特征（加权变分空间 $V_g$）
2. 推导泛化间隙的上下界（Theorem 3.5）
3. 推导非参数估计 MSE 的上下界（Theorem 3.6, 3.7）
4. 实验验证维度灾难和 neural shattering 现象

### 关键设计
1. **加权变分(半)范数**: 定义数据依赖的权函数 $g(\mathbf{u}, t)$，它综合了数据在方向 $\mathbf{u}$ 上超过阈值 $t$ 的概率、条件期望距离和条件期望位置。当输入均匀分布于单位球时，$g(\mathbf{u}, t) \asymp (1-|t|)^{d+2}$，即靠近边界时权重急剧衰减。这意味着权重大但仅在边界附近激活的神经元在此范数下可以有很小的代价。
2. **稳定性蕴含正则性（Theorem 3.2）**: 对任意两层 ReLU 网络，$|f_\theta|_{V_g} \leq \frac{\lambda_{max}(\nabla^2 \mathcal{L}(\theta))}{2} - \frac{1}{2} + (R+1)\sqrt{2\mathcal{L}(\theta)}$。结合稳定性条件（Proposition 2.1），稳定极小值必然属于 $V_g$ 的有界子集。
3. **泛化间隙上界（Theorem 3.5）**: 对稳定极小值，泛化间隙以 $n^{-1/(2d+4)}$ 的速率收敛到 0。虽然确实泛化，但速率随维度 $d$ 指数级恶化。
4. **泛化间隙下界（Theorem 3.5）**: 存在不可避免的 $n^{-2/(d+1)}$ 下界，证实维度灾难不是分析的产物，而是本质特性。
5. **MSE 上界（Theorem 3.6）**: 优化过的稳定极小值的 MSE 上界为 $\tilde{O}(n^{-1/(2d+4)})$。
6. **极小极大下界（Theorem 3.7）**: 对 $d > 1$，任何估计器在 $V_g$ 上的极小极大 MSE 下界为 $\Omega(n^{-2/(d+1)})$；对 $d = 1$，下界为 $\Omega(n^{-1/2})$。

### 核心反例构造（neural shattering 的理论基础）
下界证明的关键是一个新的 packing 论证：
- 在 $d-1$ 维单位球面 $\mathbb{S}^{d-1}$ 上堆放 $M = \exp(\Omega(d))$ 个互不相交的球面帽
- 每个帽对应一个 ReLU 神经元 $\varphi_i(\mathbf{x}) = c \cdot \phi(\mathbf{u}_i^T \mathbf{x} - t)$，仅在帽覆盖的方向上激活
- 由于权函数 $g$ 在边界处急剧衰减，这些边界附近的高权重神经元在变分范数约束下是"廉价的"
- 由此构造出指数多个"难学习"的函数，形成 minimax 下界

### 损失函数 / 训练策略
- 理论分析基于平方损失 $\mathcal{L}(\theta) = \frac{1}{2n} \sum_{i=1}^n (y_i - f_\theta(x_i))^2$
- 稳定性条件：$\lambda_{max}(\nabla^2_\theta \mathcal{L}(\theta)) \leq 2/\eta$（等价于 GD 的动态稳定性）
- 实验中使用标准 GD 训练，Kaiming 初始化，梯度裁剪阈值 50

## 实验关键数据

### 主实验：维度灾难验证

| 维度 d | GD (大学习率 η=0.2) log-MSE 斜率 | Weight Decay (η=0.01, λ=0.1) log-MSE 斜率 |
|--------|-----------------------------------|---------------------------------------------|
| 1 | ~-0.35 | >-0.50 |
| 2 | ~-0.25 | >-0.50 |
| 3 | ~-0.18 | >-0.50 |
| 4 | ~-0.13 | >-0.50 |
| 5 | ~-0.10 | >-0.50 |

GD 训练下的 MSE 收敛斜率（对数-对数图）随维度急剧下降，从 d=1 的约 -0.35 降至 d=5 的约 -0.10，验证了理论预测的维度灾难。而 weight decay 训练在所有维度下斜率都保持在 -0.5 以上，不受维度影响。

### 消融实验：Neural Shattering 现象

| 训练方式 | 每个神经元平均激活率 | 权重范数 | 训练 MSE |
|----------|---------------------|---------|---------|
| GD (η=0.9, 无 WD) | ≤ 10% | 大 | ≈ 1.105 |
| GD (η=0.01, WD=0.1) | 高（几乎全部激活） | 小且有界 | ≈ 0.055 |

宽度 2048 的网络在 512 个 10 维线性目标的含噪样本上训练。大学习率 GD 进入 edge-of-stability 机制后，Hessian 最大特征值稳定在 $2/\eta \approx 2.2$ 附近，但每个神经元仅对不到 10% 的数据点激活，权重范数很大——这正是 neural shattering。

### 关键发现
1. 稳定极小值（flat minima）的泛化性能确实随维度恶化，MSE 收敛速率从 $n^{-1/6}$ (d=1) 衰退到接近 0 (d=5)
2. 低范数解 (weight decay) 不受维度灾难影响，收敛斜率始终 ≥ -0.5
3. Neural shattering 是高维下的固有现象：GD 通过使神经元稀疏激活（而非缩小权重）来满足平坦性约束
4. 实验中观察到的 shattering 模式与下界构造中的"难学习"函数完全吻合

## 亮点与洞察
1. **平坦性 ≠ 泛化的万能钥匙**: 长期以来人们认为 flat minima 一定泛化好，本文首次在理论上证明了其在高维下的根本局限性，给出了 flat solutions 与 low-norm solutions 之间的指数级分离
2. **Neural shattering 的发现**: 揭示了一个之前未被认识到的现象——高维下 ReLU 神经元可以通过在球面边界的指数多方向上稀疏激活来"欺骗"平坦性标准，这是维度灾难的几何根源
3. **数据依赖的函数空间刻画**: 加权变分空间 $V_g$ 精确刻画了稳定极小值的归纳偏置，权函数的边界衰减行为 $(1-|t|)^{d+2}$ 是关键
4. **理论与实验的一致性**: 下界构造的几何直觉（球面覆盖 → 稀疏激活）在实验中被完美验证
5. **全域分析**: 不同于 Qiao et al. (2024) 仅在内部区间的分析，本文覆盖了全输入域，也刻画了网络的外推行为

## 局限与展望
1. 仅分析两层 ReLU 网络，更深架构的稳定极小值行为未知
2. 输入分布假设为单位球上均匀分布，更一般分布下的权函数 $g$ 更复杂
3. 上界（$n^{-1/(2d+4)}$）与下界（$n^{-2/(d+1)}$）之间仍有间隙，紧致的速率尚待确定
4. 未考虑 SGD（随机梯度下降）、batch normalization 等实际训练技巧的影响
5. 未涉及自适应学习率优化器（Adam 等），这些可能会改变稳定性特征
6. 维度灾难是否可以通过结构化先验（如低维流形假设）缓解，值得进一步研究

## 相关工作与启发
- **Mulayoff et al. (2021)**: 首次建立单变量插值情形的稳定极小值函数空间刻画，本文扩展到高维非插值
- **Qiao et al. (2024)**: 单变量非插值情形的泛化和风险界（但仅限内部区间），本文推广到高维和全域
- **Nacson et al. (2023)**: 多变量插值情形的刻画，本文去除了插值假设
- **Bach (2017)**: 证明低范数（weight decay）解不受维度灾难影响，本文建立了与 flat solutions 的指数级分离
- **Edge-of-stability (Cohen et al., 2020)**: 实验发现 GD 训练中 Hessian 特征值在 $2/\eta$ 附近振荡，本文的稳定性定义直接对应此现象
- **启发**: 仅依赖隐式正则化（平坦性）可能不够，在高维问题中显式正则化（如 weight decay）可能是必要的

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次证明 flat minima 的维度灾难 + neural shattering 的发现和理论解释
- 实验充分度: ⭐⭐⭐⭐ 合成实验充分验证理论预测，但缺少真实数据/深网络验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论严谨，动机清晰，几何直觉与形式化证明完美结合
- 价值: ⭐⭐⭐⭐⭐ 对理解深度学习泛化有根本性贡献，挑战了"flat minima = good generalization"的主流观点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Exoplanet Formation Inference Using Conditional Invertible Neural Networks](exoplanet_formation_inference_using_conditional_invertible_neural_networks.md)
- [\[NeurIPS 2025\] A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction](a_variational_manifold_embedding_framework_for_nonlinear_dimensionality_reductio.md)
- [\[AAAI 2026\] PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](../../AAAI2026/physics/physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)
- [\[NeurIPS 2025\] Neural Green's Functions](neural_greens_functions.md)
- [\[ICML 2025\] Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](../../ICML2025/physics/differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)

</div>

<!-- RELATED:END -->
