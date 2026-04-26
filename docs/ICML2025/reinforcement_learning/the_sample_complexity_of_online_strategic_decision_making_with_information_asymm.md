---
title: >-
  [论文解读] The Sample Complexity of Online Strategic Decision Making with Information Asymmetry and Knowledge Transportability
description: >-
  [ICML2025][信息不对称] 在信息不对称（代理拥有隐私类型和动作作为混淆变量）且需要跨分布知识迁移的在线强化学习场景中，提出基于非参数工具变量（NPIV）方法的模型算法 OPME，证明以 $\tilde{O}(1/\epsilon^2)$ 样本复杂度学得 $\epsilon$-最优策略，并匹配对应下界。
tags:
  - ICML2025
  - 信息不对称
  - 知识迁移
  - 工具变量
  - 因果识别
  - 样本复杂度
  - 主体-代理问题
  - 混淆变量
  - 在线学习
---

# The Sample Complexity of Online Strategic Decision Making with Information Asymmetry and Knowledge Transportability

**会议**: ICML2025  
**arXiv**: [2506.09940](https://arxiv.org/abs/2506.09940)  
**代码**: 无  
**领域**: reinforcement_learning  
**关键词**: 信息不对称, 知识迁移, 工具变量, 因果识别, 样本复杂度, 主体-代理问题, 混淆变量, 在线学习

## 一句话总结

在信息不对称（代理拥有隐私类型和动作作为混淆变量）且需要跨分布知识迁移的在线强化学习场景中，提出基于非参数工具变量（NPIV）方法的模型算法 OPME，证明以 $\tilde{O}(1/\epsilon^2)$ 样本复杂度学得 $\epsilon$-最优策略，并匹配对应下界。

## 研究背景与动机

### 信息不对称与多代理系统
多代理系统广泛存在于经济学、社会科学和机器人学中。在这些系统中，代理（agent）拥有 **私有类型** $t_h$ 和 **私有动作** $b_h$，它们对主体（principal）不可观测，构成混淆变量（confounder）。代理会根据私有信息策略性地选择动作以最大化自身收益，导致主体观测到的反馈 $e_h$ 被混淆。

### 知识迁移的挑战
在很多实际场景中，主体需要将从**源分布** $\mathcal{P}^s$（在线数据来源群体）学到的知识迁移到**目标分布** $\mathcal{P}^t$（实际服务群体），因为目标环境可能无法直接开展大规模实验。例如：将纽约的临床试验结论迁移到洛杉矶人群，或用 LLM 的实验数据指导面向人类的机制设计。

### 核心问题
传统 RL 方法假设数据是 i.i.d. 的且无混淆，无法直接适用。论文发问：

**能否用非 i.i.d. 动作在有混淆变量的环境中进行学习？**

**当源分布和目标分布不同时，能否设计近似最优算法，分布偏移如何影响样本复杂度？**

## 方法详解

### 在线战略交互模型

论文提出 **在线战略交互模型（Online Strategic Interaction Model）**，推广了 Yu et al. (2022) 的 strategic MDP 到在线设定。每一步 $h \in [H]$ 的交互流程：

1. 主体在状态 $s_h$ 采取动作 $a_h$
2. 代理的私有类型 $t_h \sim \mathcal{P}_h^s$ 被采样，代理策略性地选择 $b_h = \arg\max_b R_h^a(s_h, a_h, t_h, b)$
3. 主体收到被操纵的反馈 $e_h \sim F_h(\cdot | s_h, a_h, t_h)$（$t_h, b_h$ 不可观测）
4. 主体获得奖励 $r_h = R_h^*(s_h, a_h, e_h) + \xi_h$，其中 $\xi_h$ 是与 $t_h$ 关联的**内生噪声**
5. 状态转移 $s_{h+1} \sim P_h^*(\cdot | s_h, a_h, e_h)$

关键困难：$\xi_h$ 与 $e_h$ 都和隐私类型 $t_h$ 相关，导致 $\mathbb{E}[\xi_h | s_h, a_h, e_h] \neq 0$，即传统回归方法失效。

### 聚合模型与规划

在目标分布 $\mathcal{P}^t$ 下，定义聚合模型 $\bar{\mathcal{M}}^*$：

$$\bar{R}_h^*(s_h, a_h) = \mathbb{E}_{t \sim \mathcal{P}_h^t, e \sim F_h(\cdot|s_h,a_h,t)}[R_h^*(s_h, a_h, e)]$$

$$\bar{P}_h^*(\cdot|s_h, a_h) = \mathbb{E}_{t \sim \mathcal{P}_h^t, e \sim F_h(\cdot|s_h,a_h,t)}[P_h^*(\cdot|s_h, a_h, e)]$$

目标是学习聚合模型的最优策略 $\bar{\pi}^*$。

### 非参数工具变量（NPIV）因果识别

核心观察：$(s_h, a_h)$ 可作为 $(s_h, a_h, e_h)$ 的 **工具变量（instrumental variable）**，因为：

$$\mathbb{E}_{\mathcal{M}^*(\mathcal{P}^s)}[r_h - R_h^*(s_h, a_h, e_h) | s_h, a_h] = 0$$

该条件矩方程成立是因为 $\xi_h$ 虽依赖于 $t_h$，但在给定 $(s_h, a_h)$ 对 $t_h$ 积分后均值为零。

### OPME 算法（Optimistic Planning with Minimax Estimation）

算法遵循 **不确定性下的乐观原则**，核心步骤：

**Step 1 — 极小极大风险估计**：无法直接计算条件最小二乘（因条件期望在平方内），通过 Fenchel-Rockafellar 对偶引入判别器函数类 $\mathcal{F}_h$ 进行极小极大估计：

$$\hat{L}_h^k(R_h) = \max_{f_h \in \mathcal{F}_h} \hat{l}_h^k(R_h, f_h) - \frac{1}{2}\sum_{\tau=1}^k f_h^2(s_h^\tau, a_h^\tau)$$

其中 $\hat{l}_h^k(R, f) = \sum_{\tau=1}^k f(s_h^\tau, a_h^\tau)(R(s_h^\tau, a_h^\tau, e_h^\tau) - r_h^\tau)$。

对转移函数 $P_h^*$，引入额外判别器类 $\mathcal{G}$ 捕获所有候选模型的最优价值函数，类似构造风险函数。

**Step 2 — 构建置信集**：

$$\mathcal{R}_h^k = \{R_h \in \mathcal{R}_h : \hat{L}_h^k(R_h) \leq \beta_1\}, \quad \mathcal{P}_h^k = \{P_h \in \mathcal{P}_h : \hat{L}_h^k(P_h) \leq \beta_2\}$$

**Step 3 — 知识迁移与乐观规划**：用源分布数据估计 $R^*, P^*$，再通过已知的目标分布 $\mathcal{P}^t$ 和反馈操纵分布 $F$ 构建聚合模型的置信集 $\bar{\mathcal{C}}^k$，选择其中累积奖励最大的模型的最优策略进行探索。

> 简言之：**用 $\mathcal{P}^s$ 做估计，用 $\mathcal{P}^t$ 做探索！**

## 理论结果

### 主定理（样本复杂度）

| 量 | 含义 |
|---|---|
| $\epsilon$ | 最优性间隔 |
| $d_{V,h}$ | 分布 Eluder 维度（模型复杂度） |
| $\tau_h$ | 病态度量（ill-posedness），度量混淆变量带来的估计难度 |
| $C_h^f$ | 知识迁移乘性项，度量源/目标分布偏移的代价 |
| $B$ | 函数类的值域上界 |

**Theorem 5.4**：在可实现性假设下，OPME 算法学得 $\epsilon$-最优策略的样本复杂度为：

$$\tilde{O}\left(\sum_{h=1}^H B^2 d_{V,h} \tau_h C_h^f \log(|\mathcal{R} \times \mathcal{P} \times \mathcal{G} \times \mathcal{F}|/\delta) \cdot \epsilon^{-2}\right)$$

### 关键发现

- **$\epsilon^{-2}$ 依赖是最优的**：匹配 Domingues et al. (2021) 的下界（即便无混淆也需 $\tilde{O}(\epsilon^{-2})$）
- **线性 MDP 特例**：$d_{V,h} \lesssim \tilde{O}(d)$，复杂度为 $\tilde{O}(H d \tau C^f \epsilon^{-2})$
- **病态度量 $\tau_h$**：量化了从投影均方误差到真实均方误差的差距，体现混淆的不可避免代价
- **迁移乘性项 $C_h^f$**：当 $\mathcal{P}^s = \mathcal{P}^t$ 时 $C_h^f = 1$，退化为无迁移场景

## 亮点与洞察

- **首次在在线 RL 中统一处理信息不对称 + 知识迁移**：将离线 strategic MDP 推广到在线非 i.i.d. 设定，并引入分布偏移
- **因果视角的 RL 算法设计**：将 $(s_h, a_h)$ 作为工具变量处理内生噪声，优雅规避了混淆变量问题
- **紧样本复杂度**：$1/\epsilon^2$ 的速率在对数因子内匹配下界，分解结构清晰地展示了各因素（混淆 $\tau$、迁移 $C^f$、复杂度 $d_V$）的独立贡献
- **丰富的实际动机**：合同设计（contract design）和实验设计（用 LLM 替代人类实验）案例生动展示模型的应用价值
- **技术创新**：面对非 i.i.d. 数据，传统集中不等式失效，论文发展了新的快速鞅集中分析

## 局限与展望

- **需要已知目标分布 $\mathcal{P}^t$ 和反馈操纵分布 $F$**：这在实际中可能不易获得，虽然论文给出了合理性解释（如经济学中类型分布通常已知）
- **纯理论贡献，无实验验证**：算法涉及优化置信集上的极大化等计算上可能 intractable 的步骤
- **下界的参数依赖尚未完全刻画**：$\tau_h, C_h^f, d_{V,h}$ 各项的下界匹配性留作开放问题
- **未涉及部分可观测场景（POMDP）**：当状态也部分不可观测时，NPIV 是否仍然有效尚不清楚
- **Concentrability 假设较强**：要求数据分布均匀覆盖目标分布，限制了适用范围

## 相关工作与启发

- **Yu et al. (2022)**：离线 strategic MDP + i.i.d. 数据，本文的直接推广对象
- **Angrist & Imbens (1995); Newey & Powell (2003)**：非参数工具变量方法的经典来源
- **Jin et al. (2021); Ayoub et al. (2020)**：价值目标回归和一般函数逼近 RL 的探索方法
- **Pearl & Bareinboim (2011)**：因果推断中知识可迁移性的形式化
- **Chen & Zhang (2021); Liao et al. (2021)**：RL 中使用工具变量处理混淆数据的先驱工作
- **Myerson (1982)**：广义主体-代理问题的经济学基础

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次在在线 RL 中系统性处理信息不对称 + 知识迁移 + 非 i.i.d.
- 理论深度: ⭐⭐⭐⭐⭐ — 紧样本复杂度、新鞅集中分析、清晰的因子分解
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、示例丰富，但符号体系较重
- 实用性: ⭐⭐⭐ — 纯理论框架，计算可行性未讨论
- 综合: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](../../ICLR2026/reinforcement_learning/the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)
- [\[CVPR 2025\] Decision SpikeFormer: Spike-Driven Transformer for Decision Making](../../CVPR2025/reinforcement_learning/decision_spikeformer_spike-driven_transformer_for_decision_making.md)
- [\[ICML 2025\] Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)
- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](../../NeurIPS2025/reinforcement_learning/structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[ICML 2025\] Online Pre-Training for Offline-to-Online Reinforcement Learning](online_pre-training_for_offline-to-online_reinforcement_learning.md)

<!-- RELATED:END -->
