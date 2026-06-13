---
title: >-
  [论文解读] Deep Learning for Continuous-Time Stochastic Control with Jumps
description: >-
  [NeurIPS 2025][随机控制] 提出两种基于模型的深度学习算法（GPI-PINN 和 GPI-CBU）来求解含跳跃的有限时域连续时间随机控制问题，通过迭代训练策略网络和价值网络，避免了状态动力学的离散化和模拟，在高维场景中表现出色。 连续时间随机控制问题广泛存在于动态决策场景中，其核心是求解 Hamilton-Ja…
tags:
  - "NeurIPS 2025"
  - "随机控制"
  - "跳跃扩散"
  - "HJB方程"
  - "深度学习"
  - "Actor-Critic"
---

# Deep Learning for Continuous-Time Stochastic Control with Jumps

**会议**: NeurIPS 2025  
**arXiv**: [2505.15602](https://arxiv.org/abs/2505.15602)  
**代码**: [GitHub](https://github.com/jdupret97/Deep-Learning-for-CT-Stochastic-Control-with-Jumps)  
**领域**: 其他  
**关键词**: 随机控制, 跳跃扩散, HJB方程, 深度学习, Actor-Critic

## 一句话总结

提出两种基于模型的深度学习算法（GPI-PINN 和 GPI-CBU）来求解含跳跃的有限时域连续时间随机控制问题，通过迭代训练策略网络和价值网络，避免了状态动力学的离散化和模拟，在高维场景中表现出色。

## 研究背景与动机

连续时间随机控制问题广泛存在于动态决策场景中，其核心是求解 Hamilton-Jacobi-Bellman (HJB) 方程。传统方法面临三大挑战：

**维度灾难**：有限差分、有限元等经典方法在高维问题上不可行

**跳跃处理困难**：当系统动力学包含随机跳跃时，HJB 方程退化为 PIDE（偏积分微分方程），需要在每个时空采样点计算跳跃期望 $\mathbb{E}[V(t, x+\gamma(t,x,Z_1,a))]$，计算量巨大

**隐式最优控制**：当最优控制无法显式求解时，无法直接代入 HJB 方程化简，需要同时近似价值函数和最优控制

现有深度学习方法要么是无模型的 RL（不利用已知动力学，精度低），要么是基于时间离散化的局部方法（存在离散化误差，不易泛化到未见区域）。

## 方法详解

### 整体框架

考虑控制问题：

$$\sup_\alpha \mathbb{E}\left[\int_0^T f(t, X_t^\alpha, \alpha_t) dt + F(X_T^\alpha)\right]$$

被控过程遵循跳跃-扩散动力学：

$$dX_t^\alpha = \beta(t,X_t^\alpha,\alpha_t)dt + \sigma(t,X_t^\alpha,\alpha_t)dW_t + \int_E \gamma(t,X_{t-}^\alpha,z,\alpha_t)N^\alpha(dz,dt)$$

其中 $N^\alpha$ 是具有可控强度 $\lambda(t,X_{t-}^\alpha,\alpha_t)$ 的 Cox 过程。

方法核心：用两个神经网络 $V_\theta$ 和 $\alpha_\phi$ 分别近似价值函数和最优控制，通过 Actor-Critic 方式迭代训练，基于 Feynman-Kac 公式和验证定理保证正确性。

### 关键设计

#### 算法一：GPI-PINN

利用 PINN 方法最小化受控 HJB 方程的残差。关键技巧是通过 **Proposition 3.1** 避免显式计算梯度和 Hessian：

定义辅助函数 $\psi(h)$，使得：

$$\psi''(0) = \partial_t v(t,x) + \beta^\top(t,x,a)\nabla_x v(t,x) + \frac{1}{2}\text{Tr}[\sigma\sigma^\top(t,x,a)\nabla_x^2 v(t,x)]$$

从而将梯度和 Hessian 的计算替换为单变量函数 $\psi''(0)$ 的评估，计算代价仅为 $n \cdot \text{cost}(v)$ 的小倍数。

**价值网络更新**（最小化 PIDE 残差）：

$$\mathscr{L}_1(\theta, \phi) = \xi_1 \mathbb{E}_{(t,x)\sim\mu} \mathcal{H}^2(t,x,\theta,\phi) + \xi_2 \mathbb{E}_{x\sim\nu}(V_\theta(T,x) - F(x))^2$$

**控制网络更新**（最大化 Hamiltonian）：

$$\mathscr{L}_2(\theta, \phi) = -\mathbb{E}_{(t,x)\sim\mu} \mathcal{H}(t,x,\theta,\phi)$$

采用 RAD（Residual-based Adaptive Distribution）方法自适应更新采样分布。

**GPI-PINN 的局限**：仍需在每个采样点计算跳跃期望，且梯度步骤涉及三阶导数。

#### 算法二：GPI-CBU

利用连续时间 Bellman 更新规则，引入**无期望算子** $G_\zeta$：

$$G_\zeta(t,x,z,v,a) = v(t,x) + \zeta[\partial_t v + f + \beta^\top \nabla_x v + \frac{1}{2}\text{Tr}[\sigma\sigma^\top \nabla_x^2 v] + \lambda(v(t,x+\gamma(t,x,z,a)) - v(t,x))]$$

关键点：$G_\zeta$ 不需要计算跳跃期望（仅需单次跳跃评估），也不需要三阶导数。

**Proposition 4.1** 保证了最小化 $\mathbb{E}[(g(Y_t) - G_\zeta(t,Y_t,Z_1,V^\alpha,\alpha(t,Y_t)))^2]$ 可以恢复正确的价值函数。

价值网络更新损失：

$$\mathscr{L}_1^{(k)}(\theta) = \xi_1 \mathbb{E}_{(t,x,z)\sim\mu\otimes\mathcal{Z}}(V_\theta(t,x) - G_\zeta(t,x,z,\theta^{(k)},\phi^{(k)}))^2 + \xi_2 \mathbb{E}_{x\sim\nu}(V_\theta(T,x) - F(x))^2$$

### 损失函数 / 训练策略

两种算法都采用迭代的 Actor-Critic 训练：
1. **Step 1**（Critic）：固定控制网络，更新价值网络使其满足 HJB 方程
2. **Step 2**（Actor）：固定价值网络，更新控制网络使其最大化 Hamiltonian

网络架构使用 DGM（Deep Galerkin Method），需要 $C^2$ 激活函数。超参数 $\zeta=1$ 在速度和精度间提供了良好权衡（负缩放因子导致损失爆炸）。

## 实验关键数据

### 主实验

**线性二次调节器（LQR）含跳跃**（$d=10$）：

与 RL 和离散时间方法对比：

| 方法 | 类型 | 精度 (log MAE_V) |
|------|------|-----------------|
| PPO | 无模型 RL | 最差 |
| SAC | 无模型 RL | 次差 |
| Han & E (2016) | 有模型离散时间 | 中等（存在离散化误差） |
| **GPI-CBU** | **有模型连续时间** | **最优** |

GPI-PINN vs GPI-CBU：
- 无跳跃时：两者精度相近，GPI-CBU 因避免三阶导数略快
- **有跳跃时**：GPI-CBU 显著快于 GPI-PINN（避免了跳跃期望计算）
- GPI-PINN 收敛更稳定，GPI-CBU 计算成本更低

**高维 LQR 含跳跃**（$d=50$）：
- GPI-PINN 不可行（计算量过大）
- GPI-CBU 仍能高精度近似价值函数和最优控制
- 附录报告了高达 $d=150$ 的结果

### 消融实验

**最优消费-投资问题含跳跃**（$n=25$ 种资产，$d=52$ 状态变量）：
- 包含随机波动率、随机跳跃强度和随机利率
- GPI-CBU 训练损失收敛，提供了实际经济决策问题的可行解
- 在简化版本（常数系数）中，GPI-CBU 结果与 Runge-Kutta 参考解几乎不可区分

### 关键发现

1. **模型信息至关重要**：利用已知动力学的模型方法（GPI-PINN/CBU）远优于不利用的 RL 方法（PPO/SAC）
2. **GPI-CBU 解决了跳跃问题的核心计算瓶颈**：无期望算子避免了在每个采样点对跳跃分布积分
3. **全局近似的优势**：局部方法（Han & E 2016）仅在最优轨迹附近学好，对未探索区域泛化差
4. **收敛稳定性 vs 效率的权衡**：GPI-PINN 通过平均多次跳跃更稳定，GPI-CBU 单次跳跃评估更高效

## 亮点与洞察

1. **优雅的数学推导**：Proposition 3.1 将梯度和 Hessian 计算转化为单变量二阶导数，简洁且实用；Proposition 4.1 为 CBU 方法提供了理论基础
2. **连续时间的优势**：在连续时间框架下直接求解，避免离散化误差，提供全局解（任意时空点可查询）
3. **GPI-CBU 的核心创新**：通过递归式更新规则，将跳跃期望从损失函数中完全消除，使高维跳跃控制问题首次可解
4. **Actor-Critic 与 PIDE 求解的融合**：将 RL 中的 GPI 思想与偏微分方程数值方法优雅结合

## 局限与展望

1. **需要已知动力学模型**：在经济金融等领域，动力学模型通常需从数据中推断，作者建议可先用模型学习算法预学习
2. **GPI-CBU 的收敛稳定性**：使用单次跳跃估计导致方差较大，负缩放因子 $\zeta$ 会导致损失爆炸
3. **超参数敏感性**：$\xi_1, \xi_2, \zeta$ 的选择对性能有显著影响
4. **未处理约束和路径依赖问题**：当前框架假设 Markovian 反馈控制和无约束动作空间
5. **网络架构选择**：DGM 架构的适用性与更现代架构（如 Transformer）的比较缺失

## 相关工作与启发

- **Han & E (2016)**：基于时间离散化的深度学习控制方法，本文的连续时间方法避免了其离散化误差
- **PINN (Raissi et al.)**：物理信息神经网络，GPI-PINN 在其基础上扩展到控制问题
- **DGM (Sirignano & Spiliopoulos)**：提供了网络架构设计；经验表明可改善 PINN 性能
- **Duarte et al. (2024)**：提出了避免梯度和 Hessian 计算的技巧，本文适配到有限时域含终端条件的场景
- 对科学计算和量化金融有重要实用价值：50维消费-投资问题和150维 LQR 问题的可解性为实际应用打开了大门

## 评分

- 新颖性: ⭐⭐⭐⭐ (GPI-CBU 的无期望更新规则是核心创新，解决了跳跃控制的关键瓶颈)
- 实验充分度: ⭐⭐⭐⭐ (理论验证充分，有解析解对比，多维度扩展，但实际应用场景有限)
- 写作质量: ⭐⭐⭐⭐⭐ (数学推导严谨，算法描述清晰，理论与实验结合紧密)
- 价值: ⭐⭐⭐⭐ (为高维含跳跃随机控制提供了首个实用的深度学习求解方案)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Deep Continuous-Time State-Space Models for Marked Event Sequences](deep_continuous-time_state-space_models_for_marked_event_sequences.md)
- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)
- [\[NeurIPS 2025\] Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds](finite-time_analysis_of_stochastic_nonconvex_nonsmooth_optimization_on_the_riema.md)
- [\[NeurIPS 2025\] Continuous Thought Machines](continuous_thought_machines.md)
- [\[NeurIPS 2025\] Uncertainty Estimation by Flexible Evidential Deep Learning](uncertainty_estimation_by_flexible_evidential_deep_learning.md)

</div>

<!-- RELATED:END -->
