---
title: >-
  [论文解读] Efficient Adaptive Experimentation with Noncompliance
description: >-
  [NeurIPS 2025][医学图像][自适应实验] 提出 AMRIV——首个面向带非依从性（noncompliance）的自适应实验的半参数高效、多重鲁棒的ATE估计器，结合方差最优的工具变量分配策略和序贯推断保证。
tags:
  - NeurIPS 2025
  - 医学图像
  - 自适应实验
  - 工具变量
  - 非依从性
  - 半参数效率
  - 因果推断
---

# Efficient Adaptive Experimentation with Noncompliance

**会议**: NeurIPS 2025  
**arXiv**: [2505.17468](https://arxiv.org/abs/2505.17468)  
**代码**: [GitHub](https://github.com/CausalML/Adaptive-IV)  
**领域**: 医学图像  
**关键词**: 自适应实验, 工具变量, 非依从性, 半参数效率, 因果推断

## 一句话总结

提出 AMRIV——首个面向带非依从性（noncompliance）的自适应实验的半参数高效、多重鲁棒的ATE估计器，结合方差最优的工具变量分配策略和序贯推断保证。

## 研究背景与动机

**领域现状**：自适应实验通过根据累积数据调整分配策略来高效估计处理效应，已被 FDA 正式认可。直接分配处理的自适应 ATE 估计已有成熟工具包（A2IPW、Neyman 分配等）。

**现有痛点**：在很多现实场景中，**处理不能直接分配**，只能通过**工具变量（IV）鼓励**。例如：
   - TripAdvisor 实验：可以随机化注册界面（IV），但用户是否订阅（处理）是自愿的
   - 临床试验：医生可以推荐药物（IV），但患者是否遵从（处理）不可控
   - 存在**非依从性（noncompliance）**导致处理和 IV 不一致，传统方法会产生偏差

**核心矛盾**：
   - 自适应实验的理论在**直接分配**场景已成熟，但**仅能分配工具变量而处理内生**的场景几乎空白
   - 现有 IV 方法（DeepIV、MRIV 等）是非自适应的或不追求半参数效率

**本文目标**：将完整的现代半参数工具——高效影响函数、自适应策略学习、鲁棒插补估计、随时有效推断——引入带非依从性的自适应 IV 设置。

**切入角度**：基于 Wang & Tchetgen Tchetgen 的非混杂依从性假设和多重鲁棒影响函数，推广到自适应设置。

**核心 idea**：推导带 IV 的自适应实验的半参数效率界和最优分配策略，并构建序贯估计器实现该界。

## 方法详解

### 整体框架

**问题设置**：$T$ 轮序贯实验，每轮观察协变量 $X_t$，分配工具变量 $Z_t \sim \pi_t(\cdot|X_t, \mathcal{H}_{t-1})$，观察处理 $A_t = A_t(Z_t)$ 和结果 $Y_t$。目标估计 ATE $\tau = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]$。

**关键假设**：
- **Assumption 1（标准 IV）**：排除性约束、独立性（$Z \perp U|X$）、相关性（$\text{Cov}(Z,A|X) \neq 0$）
- **Assumption 2（非混杂依从性）**：$Y(1) - Y(0) \perp A(1) - A(0) | X$

**ATE 识别**：$\tau = \mathbb{E}_X\left[\frac{\delta^Y(X)}{\delta^A(X)}\right]$，其中 $\delta^Y(X)$ 和 $\delta^A(X)$ 分别是 IV 诱导的结果和处理变化。

### 关键设计

#### 1. 半参数效率界（Theorem 1）

$$V_{\text{eff}}(\pi) = \mathbb{E}\left[\frac{1}{\delta^A(X)^2}\left(\frac{\sigma^2(1,X)}{\pi(X)} + \frac{\sigma^2(0,X)}{1-\pi(X)}\right) + (\delta(X) - \tau)^2\right]$$

其中残差方差 $\sigma^2(z,X) = \text{Var}(Y - A\delta(X) | Z=z, X)$。

#### 2. 最优工具变量分配（Corollary 2）

$$\pi^*(X) = \frac{\sqrt{\sigma^2(1,X)}}{\sqrt{\sigma^2(1,X)} + \sqrt{\sigma^2(0,X)}}$$

**关键洞察**：
- 最优策略向残差方差更大的臂倾斜
- 残差方差**同时依赖结果噪声和依从性噪声**——与标准 ATE 设置的Neyman分配不同
- 当 $\delta^A(X) \to 1$（完美依从）退化为经典 Neyman 分配
- 当 $\delta^A(X) \to 0$（低依从）趋向均匀分配

#### 3. AMRIV 估计器

$$\hat{\tau}_T^{\text{AMRIV}} = \frac{1}{T}\sum_{t=1}^T \phi(X_t, Z_t, A_t, Y_t; \pi_t, \hat{\eta}_t)$$

影响函数 $\phi$ 是基于自适应策略 $\pi_t$ 和序贯估计的 nuisance $\hat{\eta}_t$ 的再中心化高效影响函数：

$$\phi = \frac{2Z-1}{Z\pi(X) + (1-Z)(1-\pi(X))} \cdot \frac{1}{\delta^A(X)} [Y - A\delta(X) - \mu^Y(0,X) + \mu^A(0,X)\delta(X)] + \delta(X)$$

**关键特性**：所有 nuisance 估计仅使用历史数据 $\mathcal{H}_{t-1}$，确保近鞅结构。

#### 4. 算法组件

- **Burn-in 阶段**：固定策略 $\pi_{\text{init}}$（如均匀随机化）$T_0$ 轮
- **自适应阶段**：插补最优策略 $\tilde{\pi}_t$ + 截断 $\pi_t = \text{clip}(\tilde{\pi}_t, 1/k_t, 1-1/k_t)$
- **残差方差估计**：两阶段交叉拟合消除有限样本偏差
- **Nuisance 学习器**：可用任意非参数回归器（k-NN、随机森林、神经网络）

### 理论保证

#### Theorem 3（渐近正态性）

$$\sqrt{T}(\hat{\tau}_T^{\text{AMRIV}} - \tau) \xrightarrow{d} \mathcal{N}(0, V_{\text{eff}}(\pi))$$

当 $\pi = \pi^*$ 时达到半参数效率。仅需 $L_2$ 一致性（无需 Donsker 条件）。

#### Theorem 4（收敛率）

$$|\hat{\tau}_T^{\text{AMRIV}} - \tau| = O_p(T^{-1/2}) + O_p(\|\hat{\delta}_T^A - \delta^A\|_2 \cdot \|\hat{\delta}_T - \delta\|_2)$$

#### Corollary 5（多重鲁棒性）

只要 $\hat{\delta}$ 或 $\hat{\delta}^A$ 之一 $L_2$-一致，AMRIV 就是一致的。比静态 MRIV 鲁棒性更强（因为自适应控制 $\pi_t$ 赋予了对 $\mu^Y, \mu^A$ 误指定的额外鲁棒性）。

## 实验关键数据

### 合成数据实验（T=2000, 1000 trajectories）

**单侧非依从**：$\mu^A(0,X) = 0$，依从率 $\delta^A(x) = \sigma(-2x)$

| 指标 | AMRIV | AMRIV-NA | DM | DM-NA | A2IPW | Oracle |
|------|-------|---------|----|----|-------|--------|
| 效率(Norm. MSE) | 接近Oracle | 恒定gap | 随T增大 | 随T增大 | 有偏 | 1.0 |
| 一致性 | ✓ 收敛 | ✓ 收敛 | ✓ | ✓ | ✗ 不收敛 | ✓ |
| 95% CI 覆盖率 | **名义水平** | **名义水平** | 不足 | 不足 | 严重不足 | — |

### 关键观察

1. **AMRIV 逼近 Oracle 基准**（使用真实 nuisance），自适应版本显著优于非自适应
2. **A2IPW 有偏且不收敛**——因为没有修正处理选择中的未观测混杂
3. **AMRIV-MS（误指定版本）保持一致**但覆盖率略低于名义水平
4. **DM 方法在 δ 误指定时发散**，而 AMRIV-MS 仍收敛——体现多重鲁棒性
5. 自适应设计在**低依从区域尤其有益**：将更多分配给 $Z=1$ 以补偿稀疏处理uptake

### 半合成数据（TripAdvisor）

结果与合成实验一致：自适应 IV 分配提升效率，AMRIV 实现最优覆盖和一致性。

## 亮点与洞察

- **填补重要空白**：首次将完整的半参数自适应实验工具包引入 IV/非依从性设置
- **效率界的精细分析**：揭示了最优分配同时平衡结果方差和依从方差的非平凡结构
- **Neyman 分配的优雅泛化**：完美依从 → 经典 Neyman；低依从 → 趋向均匀
- **多重鲁棒性比静态更强**：自适应控制 $\pi_t$ 提供额外鲁棒维度
- **随时有效推断**：支持序贯停止决策（通过渐近置信序列）

## 局限与展望

1. **Assumption 2（非混杂依从性）较强**：实践中可能不满足，此时估计的是 ACLATE 而非 ATE
2. **计算开销**：每轮更新所有 nuisance 估计（虽然可用 mini-batch）
3. **仅考虑二元 IV 和二元处理**：多值设置的扩展不直接
4. **有界性假设**（Assumption 3）在重尾分布下可能不适用
5. **截断参数 $k_t$ 的选择**缺乏系统性指导

## 相关工作与启发

- 泛化了 A2IPW (Kato et al.)、Cook et al. 的自适应 ATE 方法到 IV 设置
- 建立在 MRIV (Wang & Tchetgen Tchetgen) 的静态半参数框架上
- **核心启发**：在不能直接控制处理的场景（这在医学、社会科学中极常见），自适应分配鼓励/工具变量可以显著提升因果效应估计的效率

## 评分

⭐⭐⭐⭐ (4/5)

**理由**：问题动机强烈（非依从性是现实痛点），理论贡献完整（效率界+最优策略+收敛率+多重鲁棒+随时有效推断），实验虽基于模拟但验证充分。局限在于较强的依从性假设和仅限二元设置。
