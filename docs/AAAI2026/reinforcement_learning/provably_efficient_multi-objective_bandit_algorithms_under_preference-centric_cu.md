---
title: >-
  [论文解读] Provably Efficient Multi-Objective Bandit Algorithms under Preference-Centric Customization
description: >-
  [AAAI 2026][multi-objective bandits] 首次从理论角度研究显式用户偏好下的多目标多臂赌博机（MO-MAB）定制化优化问题，提出 PAMO-MAB 框架并针对"未知偏好"和"隐藏偏好"两种场景分别设计 PRUCB-UP 和 PRUCB-HP 算法，通过偏好估计 + 偏好感知优化的双组件框架实现近最优遗憾界，证明了 preference-free 算法在 Pareto 前沿冲突时必然产生 $\Omega(T)$ 线性遗憾。
tags:
  - AAAI 2026
  - multi-objective bandits
  - user preference
  - Pareto optimality
  - UCB
  - regret bound
  - preference estimation
---

# Provably Efficient Multi-Objective Bandit Algorithms under Preference-Centric Customization

**会议**: AAAI 2026  
**arXiv**: [2502.13457](https://arxiv.org/abs/2502.13457)  
**代码**: 无  
**领域**: Online Learning / Multi-Armed Bandits  
**关键词**: multi-objective bandits, user preference, Pareto optimality, UCB, regret bound, preference estimation

## 一句话总结

首次从理论角度研究显式用户偏好下的多目标多臂赌博机（MO-MAB）定制化优化问题，提出 PAMO-MAB 框架并针对"未知偏好"和"隐藏偏好"两种场景分别设计 PRUCB-UP 和 PRUCB-HP 算法，通过偏好估计 + 偏好感知优化的双组件框架实现近最优遗憾界，证明了 preference-free 算法在 Pareto 前沿冲突时必然产生 $\Omega(T)$ 线性遗憾。

## 背景与动机

### 核心问题

传统 MO-MAB 问题以 Pareto 最优为目标，使用全局策略（global policy）为所有用户选择臂。然而，不同用户对各目标维度有不同偏好：

- **场景示例**：餐厅推荐中，用户1偏好口味、用户2偏好价格，Pareto 前沿上的同一餐厅可能让一个用户满意但让另一个用户不满
- **根本矛盾**：当 Pareto 前沿包含 2 个以上冲突臂时（$|\mathcal{O}^*| \geq 2$），任何不考虑用户偏好的算法对某些用户必然产生线性遗憾

### 现有方法局限

1. **Pareto 前沿估计类**（Pareto-UCB, Pareto-TS）：估计整个前沿后随机选臂，无用户定制
2. **标量化类**（S-UCB, GGI）：用固定标量化函数将多维奖励压缩为标量，但函数与用户无关
3. **字典序类**（Lexicographic MO-MAB）：目标严格分层，无法建模用户间的连续权衡偏好

### 本文贡献

- 形式化 Preference-Aware MO-MAB（PAMO-MAB）问题，首次引入显式用户偏好框架
- 证明 preference-free 算法的 $\Omega(T)$ 下界（Proposition 1）
- 针对隐藏偏好和未知偏好两种场景提出近最优算法

## 方法详解

### 问题定义

$N$ 个用户、$K$ 个臂、$D$ 个目标维度。每轮 $t$，用户 $n$ 有随机偏好 $\boldsymbol{c}_t^n \in \mathbb{R}^D$（均值 $\bar{\boldsymbol{c}}^n$），选择臂 $a_t^n$ 后观察 $D$ 维奖励 $\boldsymbol{r}_{a_t^n, t}$。总体奖励定义为内积 $g_{a_t^n, t} = {\boldsymbol{c}_t^n}^\top \boldsymbol{r}_{a_t^n, t}$，目标是最小化累积遗憾：

$$R(T) = \sum_{t=1}^T \sum_{n=1}^N \bar{\boldsymbol{c}}^{n\top}(\boldsymbol{\mu}_{a_t^{n*}} - \boldsymbol{\mu}_{a_t^n})$$

### 整体框架：偏好估计 + 偏好感知优化

所有算法共享统一的两组件框架：

1. **偏好估计（Preference Estimation）**：从 bandit 反馈推断用户偏好向量
2. **偏好感知优化（Preference-Aware Optimization）**：基于偏好估计选择臂，使决策与用户意图对齐

### 场景一：隐藏偏好（PRUCB-HP）

用户仅提供各目标评分和总体评分，偏好需从两者关系中推断。存在两大独特挑战：

#### 挑战1: 随机映射问题

总体奖励 $g_t = (\bar{\boldsymbol{c}} + \boldsymbol{\zeta}_t)^\top \boldsymbol{r}_t$，残差噪声 $\boldsymbol{\zeta}_t^\top \boldsymbol{r}_t$ 依赖于输入（奖励越大噪声越大），标准回归模型失效。

#### 挑战2: 局部 vs 全局探索困境

- **局部探索**：选择高奖励臂以发现更好臂
- **全局探索**：选择多样臂以学习偏好
- 两者冲突：高奖励臂反而不利于偏好估计（噪声更大）

#### 关键设计 I: WLS 偏好估计器

采用加权最小二乘（WLS）进行偏好学习，权重设为奖励 $\ell_2$ 范数平方的倒数：

$$w_t^n = \frac{\omega}{\|\boldsymbol{r}_{a_t^n, t}\|_2^2}$$

- 高奖励样本分配小权重 → 抑制大残差噪声
- 低奖励样本分配大权重 → 增强其对估计的贡献
- **核心引理**：经变换后残差噪声变为与输入无关的 $R' = \sqrt{\omega}R$ 次高斯，恢复标准回归条件

#### 关键设计 II: 双探索策略

臂选择策略融合两种 bonus：

$$a_t = \arg\max_{i \in \mathcal{A}_t^n} (\hat{\boldsymbol{c}}_t^n)^\top \hat{\boldsymbol{r}}_{i,t} + B_{i,t}^{n,r} + B_{i,t}^{n,c}$$

- **奖励 bonus** $B^{n,r}$：基于 Hoeffding 不等式的标准 UCB bonus，鼓励局部探索
- **偏好 bonus** $B^{n,c}$：基于伪信息增益（pseudo information gain）的 bonus，$\beta_t \|\hat{\boldsymbol{r}}_{i,t} + \rho_{i,t}^\alpha \boldsymbol{e}\|_{(\boldsymbol{V}_{t-1}^n)^{-1}}$，鼓励全局探索

### 场景二：未知偏好（PRUCB-UP）

用户在选臂后显式提供偏好反馈。相比隐藏偏好场景大幅简化：

- **偏好估计**：直接使用经验均值 $\hat{\boldsymbol{c}}_{t+1}^n = ((t-1)\hat{\boldsymbol{c}}_t^n + \boldsymbol{c}_t^n) / t$
- **优化策略**：仅需奖励 bonus，无需偏好 bonus：$a_t^n = \arg\max_{i} (\hat{\boldsymbol{c}}_t^n)^\top (\hat{\boldsymbol{r}}_{i,t} + \rho_{i,t}^\alpha \boldsymbol{e})$

### 理论分析

#### 下界（Proposition 1）

当 $|\mathcal{O}^*| \geq 2$ 时，任何 preference-free 算法存在某些用户子集使得 $R(T) = \Omega(T)$。证明思路：对任意在偏好 $\boldsymbol{c}_1$ 下次线性遗憾的 preference-free 算法，由于策略与偏好无关，切换到对立偏好 $\boldsymbol{c}_2$ 下策略不变，但最优臂改变，遗憾必为线性。

#### PRUCB-HP 上界（Theorem 1）

隐藏偏好场景下，当 $t \geq M$（$M$ 为与 $T$ 无关的常数），遗憾渐近为：

$$R(T) = \tilde{O}(D\sqrt{T})$$

关键技术难点在于上界累积偏好 bonus $\sum_t B_{i,t}^c$：需处理 Gram 矩阵 $\boldsymbol{V}_t$ 由真实奖励构造但 bonus 中使用估计奖励的不一致性，通过转换到期望 Gram 矩阵 $\mathbb{E}[\boldsymbol{V}_t]$ 解决。

#### PRUCB-UP 上界（Theorem 2）

未知偏好场景下：

$$R(T) = O(KN\delta \log T + KN\delta D^2)$$

偏好估计误差造成的遗憾仅为关于 $D$ 和 $\delta$ 的常数项，表明偏好学习对总遗憾影响有限。

## 实验关键数据

### 主结果：不同偏好场景下的遗憾对比

| 算法 | 偏好环境 | 遗憾趋势 | 是否次线性 | 核心特点 |
|------|----------|----------|-----------|---------|
| **PRUCB-HP** | 隐藏偏好 | 次线性增长 | ✓ | WLS偏好估计 + 双探索 |
| **PRUCB-UP** | 未知偏好 | 次线性增长 | ✓ | 经验均值偏好估计 |
| PRUCB-UP (GT) | 已知偏好 | 次线性增长 | ✓ | 真实偏好上界 |
| S-UCB | 两种场景 | 线性增长 | ✗ | 固定等权标量化 |
| S-MOSS | 两种场景 | 线性增长 | ✗ | 固定等权 + MOSS策略 |
| Pareto-UCB | 两种场景 | 线性增长 | ✗ | 估计Pareto前沿 |
| Pareto-TS | 两种场景 | 线性增长 | ✗ | Thompson采样 |
| OFUL | 隐藏偏好 | 线性增长 | ✗ | 线性bandit基线 |
| UCB / MOSS | 隐藏偏好 | 线性增长 | ✗ | 仅使用标量反馈 |

### 消融实验：WLS 偏好估计器 vs 标准线性回归

| 样本数 | WLS 估计器误差 | 标准 LR 估计器误差 | WLS 优势 |
|--------|---------------|-------------------|---------|
| 20 | 较低 | 较高 | 在少量样本下即显示优势 |
| 50 | 明显低于 LR | 中等偏差 | 差距扩大 |
| 80 | 接近真实值 [0.5, 0.5] | 仍有偏差 | WLS 收敛更快 |
| 200 | 几乎无偏 | 偏差仍存在 | WLS 始终一致优于 LR |

> 实验设置：2D 玩具实例，Arm-1 均值 [0.2, 0.2]（dominated），Arm-2 均值 [0.8, 0.8]（Pareto-optimal）。偏好真实均值 [0.5, 0.5]，方差 0.05。T=5000, 10次重复取平均。

## 亮点与洞察

1. **理论意义深远**：首次证明 preference-free 算法在冲突 Pareto 前沿下必然失败（$\Omega(T)$ 下界），为 MO-MAB 定制化提供了坚实的理论动机
2. **反直觉发现**：高奖励臂（Pareto 最优臂）反而不利于偏好学习——奖励越大，由随机偏好引入的噪声越大。这颠覆了"好臂信息多"的直觉
3. **WLS 权重设计精巧**：用 $w_t = \omega / \|\boldsymbol{r}_t\|_2^2$ 消除残差噪声对输入的依赖，将非标准回归问题转化为标准次高斯噪声模型
4. **双探索机制新颖**：伪信息增益 bonus 在观察奖励前就能估计选择某臂对偏好估计的贡献，同时平衡局部-全局探索
5. **框架统一性好**：PRUCB-UP 可视为 PRUCB-HP 的特殊情形（偏好噪声为零），两者共享相同的算法骨架

## 局限性

1. **合成数据验证**：实验仅在合成数据上进行，缺少真实推荐系统或评分系统的验证
2. **线性偏好假设**：总体奖励定义为偏好和奖励的内积（线性模型），无法建模非线性用户满意度函数
3. **偏好独立性假设**：Assumption 3 要求偏好与奖励独立，在某些场景下不一定成立（例如价格敏感用户可能因高价产生偏好变化）
4. **偏好平稳性**：假设每个用户的偏好均值 $\bar{\boldsymbol{c}}^n$ 固定，不考虑偏好漂移
5. **规模扩展性**：WLS 估计器需维护 $D \times D$ 的 Gram 矩阵及其逆，在高维目标空间中计算成本较高
6. **用户切换协议**：实验中的 block 轮换协议较为特殊，与更一般的推荐场景可能有差异

## 相关工作

- **Pareto 前沿估计类 MO-MAB**：Pareto-UCB（Drugan & Nowe 2013）、Knowledge Gradient（Yahyaa et al. 2014）、Thompson Sampling（Yahyaa & Manderick 2015）、多目标上下文 bandit（Turgay et al. 2018; Lu et al. 2019）
- **标量化 MO-MAB**：GGI 优化（Busa-Fekete et al. 2017）、音乐平台多目标优化（Mehrotra et al. 2020）、Pareto regret 分析（Xu & Klabjan 2023）
- **字典序偏好 MO-MAB**：Hüyük & Tekin 2021（字典序引入）、Cheng et al. 2024（混合 Pareto-字典序）
- **线性 bandit 基础技术**：OFUL（Abbasi-Yadkori et al. 2011）、MOSS（Audibert & Bubeck 2009）、方差依赖 bandit（Zhou et al. 2021）

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4.5 | 首次理论化用户偏好下的 MO-MAB 定制优化 |
| 理论深度 | 5 | 完整的下界 + 上界分析，近最优遗憾保证 |
| 实验充分度 | 3 | 仅合成实验，但覆盖多种设置和消融 |
| 实用性 | 3.5 | 推荐/评分系统有应用前景，但线性假设限制较大 |
| 写作质量 | 4 | 结构清晰，动机图示直观，理论推导严谨 |
| **总分** | **4** | 理论贡献扎实的 MO-MAB 偏好定制化先驱工作 |

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](../../NeurIPS2025/reinforcement_learning/thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[AAAI 2026\] Scalable Multi-Objective and Meta Reinforcement Learning via Gradient Estimation](scalable_multi-objective_and_meta_reinforcement_learning_via_gradient_estimation.md)
- [\[AAAI 2026\] Object-Centric Latent Action Learning](object-centric_latent_action_learning.md)
- [\[AAAI 2026\] Object-Centric World Models for Causality-Aware Reinforcement Learning](object-centric_world_models_for_causality-aware_reinforcement_learning.md)

<!-- RELATED:END -->
