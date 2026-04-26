---
title: >-
  [论文解读] Safety Certificate against Latent Variables with Partially Unidentifiable Dynamics
description: >-
  [ICML2025][安全证书] 提出基于概率空间不变性条件的安全证书设计方法，利用因果强化学习从含潜变量的离线数据中学习边际化 Q 函数，在离线与在线统计分布不一致的情况下仍能保证长期安全性，并证明了安全动作的持续可行性。
tags:
  - ICML2025
  - 安全证书
  - 潜变量
  - 部分可辨识动力学
  - 因果强化学习
  - 控制障碍函数
  - 分布偏移
  - 前门调整
---

# Safety Certificate against Latent Variables with Partially Unidentifiable Dynamics

**会议**: ICML2025  
**arXiv**: [2506.17927](https://arxiv.org/abs/2506.17927)  
**代码**: 无  
**领域**: 安全RL / Safe Control  
**关键词**: 安全证书, 潜变量, 部分可辨识动力学, 因果强化学习, 控制障碍函数, 分布偏移, 前门调整

## 一句话总结

提出基于概率空间不变性条件的安全证书设计方法，利用因果强化学习从含潜变量的离线数据中学习边际化 Q 函数，在离线与在线统计分布不一致的情况下仍能保证长期安全性，并证明了安全动作的持续可行性。

## 研究背景与动机

在自动驾驶等场景中，系统常受到**不可观测的潜变量** $W_t$（如路面湿滑程度、行人意图）的影响。这些潜变量带来两大挑战：

**动力学部分不可辨识**：完整状态 $(X_t, W_t)$ 无法获取，底层转移核 $\mathcal{P}(X_{t+1}, W_{t+1} | X_t, W_t, U_t)$ 无法完全辨识

**离线-在线分布偏移**：离线数据中行为策略 $\pi^b$ 依赖潜变量（如人类司机看到路滑会更用力刹车），导致 $P_{\text{offline}}(X_{t+1}|X_t,U_t) \neq P_{\text{online}}(X_{t+1}|X_t,U_t)$

现有安全控制方法（如控制障碍函数 CBF、Lyapunov 方法）通常假设：

- 完全已知的系统动力学或完美模拟器
- 完全可观的状态
- 分布一致性

这些假设在存在潜变量时均不成立。仅保证**近视安全**（myopic safety）也不够：短期内潜变量的影响可能不明显，但可能导致**不可恢复状态**（如高速驶入低限速区）。

**核心问题**：如何在潜变量导致分布偏移、动力学部分不可辨识的条件下，高效保证随机系统的长期安全？

## 方法详解

### 问题建模：混淆马尔可夫决策过程

系统建模为混淆 MDP $(\mathcal{X}, \mathcal{U}, \mathcal{W}, \mathcal{P}, H)$：

- $X_t \in \mathcal{X}$：可观状态
- $U_t \in \mathcal{U}$：控制动作
- $W_t \in \mathcal{W}$：不可观潜变量
- Assumption 2.1 保证可观状态满足 Markov 性质：$P(X_{t+1}|X_t,U_t) = P(X_{t+1}|\{X_\tau\}_{\tau \le t}, \{U_\tau\}_{\tau \le t})$

长期安全目标要求：

$$\mathbb{P}^{\hat{\pi},\pi}(C(X_t) \cap C(X_{t+1}) \cap \cdots \cap C(X_H) | X_0) \ge 1 - \epsilon, \quad \forall t$$

### 核心创新 1：概率空间不变性条件

传统方法在**状态空间**建立前向不变性条件，需要完整动力学和状态可观性。本文在**概率空间**建立不变性条件。

定义长期安全概率函数：

$$\Psi^\pi(x,t) := \mathbb{P}^\pi(C(X_t) \cap \cdots \cap C(X_H) | X_t = x)$$

**Proposition 3.1** 证明该函数等价于辅助 MDP 上的边际化值函数：

$$V^\pi([x^T, k]^T) = \Psi^\pi(x, H-k)$$

辅助 MDP 的关键设计：当安全事件 $C(\hat{X}_t)$ 不成立时，状态冻结 $\hat{X}_{t+1} = \hat{X}_t$；reward 设为终端时刻安全指示函数 $r = \mathbf{1}\{k=0\}\mathbf{1}\{C(x)\}$。

### 核心创新 2：基于 Q 函数的安全证书

**Theorem 3.2**：若 $\Psi^\pi(X_0, 0) > 1 - \epsilon$ 且在所有时刻满足：

$$\mathbb{E}[V^\pi(\hat{Y}_{t+1}) | \hat{Y}_t, U_t] - V^\pi(\hat{Y}_t) \ge 0$$

则安全目标成立。该条件的含义是：每步动作不降低期望安全概率（类比超鞅条件）。

由于在线转移分布未知，上式无法直接计算。**Lemma 3.3** 给出等价但可计算的形式——**安全证书**：

$$S(X_t, U_t, t) := Q^\pi(\hat{Y}_t, U_t) - \mathbb{E}_{U \sim \pi}[Q^\pi(\hat{Y}_t, U)] \ge 0$$

直觉：选择的动作的 Q 值不低于策略 $\pi$ 下的平均 Q 值。

### 核心创新 3：持续可行性保证

**Theorem 3.4** 证明在所有时刻，总存在满足安全证书的动作 $U_t \in \mathcal{U}$。证明思路简洁：取 $u^* = \arg\max_u Q^\pi(\hat{Y}_t, u)$，由最大值性质即可得证。

### 因果强化学习桥接分布偏移

利用中介变量 $M_t$ 和**前门调整**（front-door adjustment）公式，从离线数据估计无偏的 Q 函数：

1. **Algorithm 1**：从原始离线数据 $\mathcal{D}$ 构造辅助数据集 $\tilde{\mathcal{D}}$
2. 通过迭代求解最小二乘问题（Eq. 44）学习 $Q_M^\pi$
3. 利用 Eq. 54 从 $Q_M^\pi$ 恢复 $Q^\pi$

在线控制阶段（Algorithm 2）求解优化问题：

$$\arg\min_{u} J(U^n, u) \quad \text{s.t.} \quad S(X_t, u, t) \ge 0$$

在保证安全的前提下，尽量接近名义策略 $\pi^n$ 的动作。

## 实验关键数据

### 实验设置

简化驾驶场景，离散状态空间：

| 设置项 | 内容 |
|---|---|
| 状态 $X_t$ | 二维整数向量 $[位置, 速度]^T$ |
| 动作 $U_t$ | $\{-3,-2,-1,0,1\}$（加/减速） |
| 潜变量 $W_t$ | $\{0,1,2,3\}$（路面湿滑度，减弱制动效果） |
| Episode 长度 $H$ | 10 |
| 安全阈值 $\epsilon$ | 0.2 |
| 模拟次数 | 100 次模拟 × 100 条轨迹 |
| 安全约束 | 不同路段的变速限制 |
| 基线 | 离散时间控制障碍函数 (DTCBF, Cosner et al. 2023) |

### 主要结果

- **提出方法**：在无法观测潜变量、无法获取真实动力学的条件下，基于因果 RL 估计的 Q 函数成功将长期安全概率维持在 $1 - \epsilon = 0.8$ 以上，满足安全目标 (Figure 2)
- **DTCBF 基线**：使用离线统计估计的安全条件在分布偏移下失效，无法满足长期安全目标，安全概率在后续时步显著下降
- 95% 置信区间验证了结果的统计显著性

## 亮点与洞察

1. **理论贡献突出**：首次将因果强化学习与安全证书设计结合，建立了从离线有偏数据到在线安全保证的完整理论链路
2. **概率空间不变性条件**：绕开了对完整动力学和状态可观性的依赖，通过 MDP 值函数巧妙将安全概率量化问题转化为标准 RL 问题
3. **持续可行性证明**：Theorem 3.4 保证安全约束"永远不会把自己逼入死角"，这是许多安全 RL 方法缺乏的关键性质
4. **框架通用性**：虽然示例使用了前门调整 (Shi et al. 2024)，但该框架可兼容其他因果 RL 技术（工具变量、后门调整等）

## 局限与展望

1. **仅有简化数值实验**：只在低维离散驾驶场景验证，缺乏连续高维环境、真实机器人等实验，说服力有限
2. **离散空间假设**：Algorithm 2 中 Q 函数的估计和前门调整公式基于离散状态/动作空间，向连续空间扩展需要额外函数逼近
3. **Assumption 2.1 的限制**：要求潜变量 $W_t$ 在给定 $X_t$ 后条件独立于历史，排除了潜变量有长期记忆的场景
4. **中介变量假设**：Assumption 3.5 中前门调整需要可观的中介变量 $M_t$，在很多实际系统中不易满足
5. **Q 函数估计误差的影响**：理论结果假设 Q 函数精确已知，未分析有限样本估计误差传播到安全保证的影响
6. **计算复杂度**：在线优化问题 (55) 在高维动作空间中的求解效率未讨论

## 相关工作与启发

- **安全控制**：控制障碍函数 (Ames et al. 2016/2019)、预测安全过滤器 (Wabersich et al. 2021)、随机 CBF (Clark 2021) — 均需完整动力学
- **因果 RL**：混淆 MDP 下的值函数估计 (Wang et al. 2021b, Shi et al. 2024, Bennett & Kallus 2024) — 提供了处理分布偏移的工具
- **本文的桥梁作用**：首次将因果 RL 的去混淆能力用于安全证书设计，有望启发更多"因果推断 × 安全控制"交叉研究

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （首创因果RL+安全证书框架，概率空间不变性条件新颖）
- 实验充分度: ⭐⭐ （仅一个低维离散模拟实验）
- 写作质量: ⭐⭐⭐⭐ （理论推导严谨清晰，符号体系统一）
- 价值: ⭐⭐⭐⭐ （开辟了重要新方向，但实验验证不足限制了即时影响力）

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization](../../NeurIPS2025/reinforcement_learning/dynamics-aligned_latent_imagination_in_contextual_world_models_for_zero-shot_gen.md)
- [\[ICLR 2026\] Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts](../../ICLR2026/reinforcement_learning/dual-robust_cross-domain_offline_reinforcement_learning_against_dynamics_shifts.md)
- [\[ICML 2025\] Embedding Safety into RL: A New Take on Trust Region Methods](embedding_safety_into_rl_a_new_take_on_trust_region_methods.md)
- [\[ICML 2025\] PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable RL](pigdreamer_privileged_information_guided_world_models_for_safe_partially_observa.md)
- [\[ICML 2025\] Learning Dynamics under Environmental Constraints via Measurement-Induced Bundle Structures](learning_dynamics_under_environmental_constraints_via_measurement-induced_bundle.md)

<!-- RELATED:END -->
