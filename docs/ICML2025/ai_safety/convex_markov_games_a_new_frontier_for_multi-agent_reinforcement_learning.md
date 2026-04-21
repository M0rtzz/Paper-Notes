---
title: >-
  [论文解读] Convex Markov Games: A New Frontier for Multi-Agent Reinforcement Learning
description: >-
  [ICML2025][AI安全][Convex Markov Game] 提出**凸 Markov 博弈 (cMG)** 框架，将单 agent 凸 MDP 推广到多 agent 设定，允许对占用度量 (occupancy measure) 施加一般凸偏好（如熵、KL 散度、公平性惩罚、安全约束），证明纯策略 Nash 均衡存在，并设计可微的投影梯度损失 (PGL) 算法逼近均衡。
tags:
  - ICML2025
  - AI安全
  - Convex Markov Game
  - Nash Equilibrium
  - Occupancy Measure
  - Multi-Agent RL
  - 凸优化
  - 安全性
  - 公平性
---

# Convex Markov Games: A New Frontier for Multi-Agent Reinforcement Learning

**会议**: ICML2025  
**arXiv**: [2410.16600](https://arxiv.org/abs/2410.16600)  
**代码**: 未开源  
**领域**: 多Agent强化学习 / 博弈论  
**关键词**: Convex Markov Game, Nash Equilibrium, Occupancy Measure, Multi-Agent RL, 凸优化, 安全性, 公平性

## 一句话总结

提出**凸 Markov 博弈 (cMG)** 框架，将单 agent 凸 MDP 推广到多 agent 设定，允许对占用度量 (occupancy measure) 施加一般凸偏好（如熵、KL 散度、公平性惩罚、安全约束），证明纯策略 Nash 均衡存在，并设计可微的投影梯度损失 (PGL) 算法逼近均衡。

## 研究背景与动机

- **MDP → 凸 MDP**: 单 agent RL 中，凸 MDP (cMDP) 允许目标为占用度量的一般凸函数（如最大化探索熵），而非仅线性回报累加。
- **Markov 博弈 (MG)**: 多 agent RL 的经典框架，但目标仍限于线性回报。
- **空缺**: 多 agent + 凸目标的交叉区域（n>1 × 凸损失）此前无正式定义，更无 Nash 均衡存在性证明。
- **现实需求**: 行为多样性（creativity）、模仿学习、公平性、安全约束等目标天然不满足时间步可加性，无法用标准 MG 建模。

| | 线性损失 | 凸损失 |
|---|---|---|
| n=1 | MDP | 凸 MDP |
| n>1 | Markov 博弈 | **凸 Markov 博弈 (本文)** |

## 方法详解

### 1. 凸 Markov 博弈定义

cMG 为六元组 $\mathcal{G} = \langle \mathcal{S}, \mathcal{A}, P, u, \gamma, \mu_0 \rangle$，关键区别在于效用函数 $u_i$ 是占用度量 $\mu_i$ 的**凹函数**（等价于最小化凸损失）：

$$u_i : \left(\prod_{j=1}^{n} \Delta^{\mathcal{S} \times \mathcal{A}_j}\right) \to \mathbb{R}$$

占用度量定义：

$$\mu^{\pi}(s, a) = (1 - \gamma) \sum_{t=0}^{\infty} \gamma^t \mathbb{P}(s_t = s, a_t = a \mid \mu_0, \pi, P)$$

状态占用可通过矩阵方程恢复：$\mu^s(\pi) = (1-\gamma)[I - \gamma P^{\pi}]^{-1} \mu_0$。

### 2. Nash 均衡存在性

- **混合策略 NE**（Proposition 1）：策略空间紧凸 + 效用连续 → Glicksberg 定理直接得出。
- **纯策略 NE**（Theorem 1，核心贡献）：最佳响应集在占用度量空间凸但在策略空间**非凸**（打破 Kakutani 不动点前提）。论文诉诸拓扑论证——证明最佳响应集是**可收缩的 (contractible)**，利用 Debreu (1952) 的更一般存在性定理完成证明。

### 3. 均衡计算：投影梯度损失 (PGL)

**可利用度（exploitability）** 定义：

$$\epsilon = \max_{i} \left[\max_{z \in \mathcal{M}_i(\pi_{-i})} u_i(z, \pi_{-i}) - u_i(\mu_i, \pi_{-i})\right]$$

直接最小化 exploitability 需求解 n 个凸规划，代价高。论文引入熵正则化后推导**上界**：

$$\epsilon_i(\pi) \leq \tau \log(|\mathcal{S}||\mathcal{A}_i|) + \sqrt{2} \|\Pi_{T\mathcal{U}_i}(\nabla_{\mu_i}^{i\tau})\|$$

其中 $\Pi_{T\mathcal{U}_i}$ 为占用度量可行集切空间投影算子：

$$\Pi_{T\mathcal{U}_i} = I - A^\top (A A^\top)^{-1} A$$

由此定义可微损失函数：

$$\mathcal{L}^{\tau}(\pi) = \sum_i \|\Pi_{T\mathcal{U}_i}(\nabla_{\mu_i}^{i\tau})\|^2$$

**PGL 算法**：在策略 logit 空间用 Adam 优化 $\mathcal{L}^{\tau}$，并逐步退火温度 $\tau \to 0$，兼取两视角优势——策略视角的独立简单投影 + 占用度量视角的凸性上界。

### 4. 四类应用场景

| 应用 | 效用函数形式 | 核心思想 |
|---|---|---|
| **创造性** | $u_i = r_i^\top \mu_i + \tau H(\mu_i)$ | 熵奖励鼓励多样行为 |
| **模仿** | $u_i = r_i^\top \mu_i - \tau \,d_{\text{KL}}(\mu_i \| \mu_i^{\text{ref}})$ | KL 正则化接近人类策略 |
| **公平性** | $u_i = r_i^\top \mu_i - (\text{状态频次差})^2$ | 惩罚访问频率不均 |
| **安全性** | $u_i = r_i^\top \mu_i - c \cdot \max(0, \mu_i(s_{\text{danger}}, a_{\text{fast}}) - 0.1)$ | 非光滑凸惩罚限制危险行为 |

## 实验关键数据

| 实验场景 | 方法 | Exploitability | 关键发现 |
|---|---|---|---|
| 多 agent 寻路 (pathfinding) | PGL | ε ≈ 1.7（≈7% utility） | 学会协调通过瓶颈走廊 |
| 重复囚徒困境 (IPD) | PGL | 收敛至 ≈0 | 发现互惠合作策略（utility=0.47 > DD 的 0.33） |
| 重复公共品博弈 (IPGG) | PGL | 收敛至 ≈0 | 找到条件贡献策略（utility=0.03 > 全不贡献的 0） |
| Bach-Stravinsky 公平性 | PGL | ε ≤ 2.5×10⁻⁵ | 60/40 公平出席，频次差 < 10⁻⁵ |
| 仓库机器人安全 | PGL (无安全损失) | ε ≤ 3.4×10⁻² | 快速动作占 69% |
| 仓库机器人安全 | PGL (有安全损失) | ε ≤ 1.0×10⁻³ | 快速动作降至 42%，安全行为提升 |
| IPD 模仿人类 | PGL+KL退火 | ε = 1.4×10⁻⁴ | 策略与人类近似，但 utility 更高(0.48 vs 0.46)，exploitability 降低 3 个数量级 |

基线对比：直接最小化 exploitability (cvxpylayers) 因数值不稳定**崩溃**；Round-Robin 收敛但找到更简单的状态无关均衡；sgamesolver 同样只找到底层 NFG 的状态无关 NE。PGL 独特地发现了**状态依赖的对称策略**。

## 亮点与洞察

1. **填补理论空白**: 首次正式定义 cMG 并证明纯策略 NE 存在，拓扑论证方法（可收缩性 → 不动点）绕过了 Bellman 方程和 Kakutani 定理均不可用的难点。
2. **统一框架**: 创造性、模仿、公平、安全四大应用在同一数学语言下统一处理。
3. **暂态模仿效应**: 高温退火阶段的熵奖励迫使 agent 探索合作状态，低温后虽奖励消失，合作行为被"锁定"——这是一种巧妙的均衡选择机制。
4. **实验发现**: 在 IPD 中发现的互惠策略与 tit-for-tat 结构相似但更稳健（exploitability 极低）；在仓库场景中，凸安全惩罚有效降低危险行为频率。
5. **可微端到端**: 损失函数、投影算子、占用度量映射均可微分，支持自动微分优化。

## 局限与展望

1. **中心化 + 需已知动态**: 当前算法假设完全了解转移概率，无法直接应用于 model-free / 去中心化场景。
2. **投影算子无偏估计困难**: 与 NFG 不同，cMG 的投影算子 $\Pi_{T\mathcal{U}_i}$ 依赖于其他玩家策略，难以构造无偏估计器。
3. **无收敛保证**: PGL 对非凸 exploitability 做梯度下降，只有经验有效，无理论收敛率。
4. **可扩展性待验证**: 实验仅限于小规模表格博弈（几十个状态），尚未扩展到深度 RL 场景。
5. **NE 计算本身是 PPAD-hard**: cMG 严格包含 MG，计算复杂度至少一样高，PGL 的退火启发式不保证找到全局最优。

## 相关工作与启发

- **凸 MDP** (Zhang et al., 2020; Zahavy et al., 2021): 本文的直接单 agent 前身。
- **Markov 博弈均衡计算** (Fink 1964; Eibelshäuser & Poensgen 2019): sgamesolver 的同伦方法启发了 PGL 的温度退火。
- **RLHF 与 Markov 博弈** (Wu et al., 2025): LLM 对齐被形式化为 Markov 博弈，cMG 有望提供更丰富的建模工具。
- **多 agent 探索** (Zahavy et al., 2023): 在国际象棋中用凸 MDP 发现创造性玩法，本文将其推广到多 agent 均衡。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 填补 cMDP 到多 agent 的理论空白，纯策略 NE 的拓扑存在性证明优雅
- 实验充分度: ⭐⭐⭐⭐ — 七个场景覆盖四类应用，但均为小规模表格实验
- 写作质量: ⭐⭐⭐⭐⭐ — 理论严谨、图示直观（Figure 1 的占用度量空间可视化尤佳）
- 价值: ⭐⭐⭐⭐⭐ — 为 MARL 开辟新方向，统一框架具有广泛应用潜力

<!-- RELATED:START -->

## 相关论文

- [Sample-Efficient Distributionally Robust Multi-Agent Reinforcement Learning via Online Interaction](../../ICLR2026/ai_safety/sample-efficient_distributionally_robust_multi-agent_reinforcement_learning_via_.md)
- [Adversarial Inception Backdoor Attacks against Reinforcement Learning](adversarial_inception_backdoor_attacks_against_reinforcement_learning.md)
- [Reverse Engineering Human Preferences with Reinforcement Learning](../../NeurIPS2025/ai_safety/reverse_engineering_human_preferences_with_reinforcement_learning.md)
- [Contextual Integrity in LLMs via Reasoning and Reinforcement Learning](../../NeurIPS2025/ai_safety/contextual_integrity_in_llms_via_reasoning_and_reinforcement_learning.md)
- [From Single to Societal: Analyzing Persona-Induced Bias in Multi-Agent Interactions](../../AAAI2026/ai_safety/from_single_to_societal_analyzing_persona-induced_bias_in_multi-agent_interactio.md)

<!-- RELATED:END -->
