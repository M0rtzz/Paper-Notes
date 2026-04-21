---
title: >-
  [论文解读] Hierarchical Reinforcement Learning with Targeted Causal Interventions
description: >-
  [ICML2025][层次强化学习] 提出 HRC 框架，将层次强化学习中的子目标关系建模为因果图，通过因果发现算法学习子目标结构，并基于因果效应优先级进行**定向干预**，显著降低长时域稀疏奖励任务的训练代价。
tags:
  - ICML2025
  - 层次强化学习
  - 因果发现
  - 子目标结构
  - 干预采样
  - 长时域稀疏奖励
---

# Hierarchical Reinforcement Learning with Targeted Causal Interventions

**会议**: ICML2025  
**arXiv**: [2507.04373](https://arxiv.org/abs/2507.04373)  
**代码**: [GitHub](https://github.com/sadegh16/HRC)  
**领域**: reinforcement_learning  
**关键词**: 层次强化学习, 因果发现, 子目标结构, 干预采样, 长时域稀疏奖励

## 一句话总结

提出 HRC 框架，将层次强化学习中的子目标关系建模为因果图，通过因果发现算法学习子目标结构，并基于因果效应优先级进行**定向干预**，显著降低长时域稀疏奖励任务的训练代价。

## 研究背景与动机

传统 RL 在长时域稀疏奖励任务中表现不佳，HRL 通过将任务分解为子目标层次来缓解该问题。核心挑战在于如何高效发现子目标间的层次结构并利用它来加速训练。

已有工作（Hu et al., 2022; Nguyen et al., 2024）尝试用因果发现算法推断子目标间的因果关系，但存在以下不足：

- 直接套用通用因果发现算法（Ke et al., 2019），未针对 HRL 场景做适配
- 从可控子目标集中**随机**选择进行探索，未利用因果效应做优先级排序
- 缺乏理论分析和性能保证

本文针对这些问题，提出了一个系统性的因果层次强化学习框架。

## 方法详解

### 整体框架：HRC (Hierarchical RL via Causality)

HRC 将环境中的"解锁因子"变量 $\mathcal{X} = \{X_1, \ldots, X_n\}$ 与子目标 $\Phi = \{g_1, \ldots, g_n\}$ 一一对应，子目标 $g_i$ 在 $X_i^t = 1$ 时达成。子目标间的因果关系用**子目标结构** $\mathscr{G}$（有向图）表示。

算法维护两个关键集合：

- **可控集 (CS)**：已学会如何达成的子目标
- **干预集 (IS)**：用于因果探索的子目标

### 算法流程（Algorithm 1）

1. **初始化**：训练根子目标（无父节点的子目标），加入 CS
2. **循环**（直到最终目标加入 IS）：
    - 从 CS 中选一个子目标 $g_{\text{sel}}$ → 加入 IS
    - **干预采样**：对 IS 中子目标做干预，收集轨迹数据 $D_I$
    - **因果发现**：推断子目标结构 $\hat{\mathscr{G}}$
    - **识别可达子目标** CCS：其所有父节点均在 IS 中
    - **训练可达子目标** → 加入 CS

### 定向干预策略（Targeted Strategy）

关键创新在于如何从 CS 中选择 $g_{\text{sel}}$，提出两个因果引导排序规则：

**规则1：因果效应排序 (Causal Effect Ranking)**

$$g_{\text{sel}, t} = \arg\max_{g_i \in CS_{t-1}} \widehat{ECE}_{t^*}^{\Delta}(\{g_i\}, \{\}, g_n)$$

选择对最终目标 $g_n$ 因果效应最大的子目标。当所有子目标为 AND 类型时，该规则仅将有路径通往最终目标的子目标加入 IS，达到最优。

**规则2：最短路径排序 (Shortest Path Ranking)**

借鉴 A* 搜索，选择组合代价 $\mathsf{f}(g_i) = \mathsf{g}(g_i) + \mathsf{h}(g_i)$ 最小的子目标。当子目标结构为 DAG 且全为 OR 类型时，该规则精确匹配最短路径，训练代价最小。

### 因果发现算法：SSD (Subgoal Structure Discovery)

将子目标变化建模为抽象结构因果模型 (A-SCM)：

$$X_i^{t+1} = \theta_i(\mathbf{X}^t) \oplus \epsilon_i^{t+1}$$

其中 AND 子目标 $\theta_i = \bigwedge_{g_j \in PA_{g_i}} X_j^t$，OR 子目标 $\theta_i = \bigvee_{g_j \in PA_{g_i}} X_j^t$。

通过最小化带稀疏正则的损失函数恢复父节点：

$$\mathcal{L}(\boldsymbol{\beta}) = \mathbb{E}[(\hat{X}_i^{t+1} - X_i^{t+1})^2] + \lambda \|\boldsymbol{\beta}\|_0$$

理论证明（Theorem 8.4）：存在 $\lambda > 0$ 使最优解 $\boldsymbol{\beta}^*$ 中正系数恰好对应 $X_i$ 的父节点。

## 实验关键数据

### 理论代价分析（Table 1）

| 图结构 | HRC_h（定向） | HRC_b（随机） |
|--------|--------------|--------------|
| 树 $G(n,b)$ | $O(\log^2(n) \cdot b)$ | $\Omega(n^2 b)$ |
| 半 Erdős–Rényi $G(n,p)$ | $O(n^{4/3+2c/3} \log n)$ | $\Omega(n^2)$ |

定向策略在树结构上实现**指数级**加速。

### 合成数据实验（Figure 5）

- 在半 Erdős–Rényi 图上，HRC_c 和 HRC_s 均大幅优于 HRC_b
- 在树结构上，因果效应规则与最短路径规则等价（仅一条路径到最终目标）
- 图越稀疏，定向策略的优势越大

### 2D-Minecraft 环境（Figure 6）

| 方法 | 收敛速度 |
|------|---------|
| HRC_h(SSD) | **最快** |
| HRC_b(SSD) | 次之 |
| CDHRL | 较慢 |
| HAC / HER / OHRL / PPO | 最慢 |

### 因果发现精度对比（Table 2）

| 方法 | SHD ↓ | 缺边 | 多边 |
|------|-------|------|------|
| SSD（本文）| **12.3** | 6.0 | **6.3** |
| SDI (Ke et al.) | 19.8 | 4.2 | 15.6 |

SSD 多边数远低于 SDI，总体 SHD 降低 38%。

## 亮点与洞察

- **因果视角的 HRL**：将子目标达成等价于因果干预（do 算子），建立了 HRL 与因果推断的优雅桥梁
- **理论保证**：首次为因果 HRL 提供训练代价复杂度分析，在树结构上从 $\Omega(n^2)$ 降至 $O(\log^2 n)$
- **定制化因果发现**：SSD 针对 HRL 的 AND/OR 子目标结构做专门设计，比通用算法更精确
- **双排序规则互补**：因果效应规则适合 AND 类型，最短路径规则适合 OR 类型，另有混合规则

## 局限与展望

- 资源变量假设为**离散二值**，连续/高维状态空间需额外的解耦表示学习
- 假设子目标一旦达成永不丢失（Assumption 4.2），不适用于可逆环境
- 因果发现只能恢复**可发现父节点**（Def 8.2），某些父子关系可能遗漏
- 仅在 2D-Minecraft 验证，缺少更复杂 3D 环境或机器人操作实验
- 实践中需预先定义资源环境变量集合 $\mathcal{X}$，自动发现仍是开放问题

## 相关工作与启发

- **CDHRL** (Hu et al., 2022)：首个因果 HRL 工作，用 SDI 做因果发现但未做优先级排序
- **Nguyen et al., 2024**：在状态-动作对上做因果发现，状态空间大时更困难
- **HAC** (Levy et al., 2017)：层次 Actor-Critic，无因果结构
- **HER** (Andrychowicz et al., 2017)：事后经验回放
- 启发：因果效应排序可推广到其他需要优先级探索的领域（如主动学习、实验设计）

## 评分

- 新颖性: ⭐⭐⭐⭐ (因果干预视角 + 定向探索策略 + 专用因果发现)
- 实验充分度: ⭐⭐⭐⭐ (理论验证 + 合成数据 + 2D-Minecraft，但缺少更多真实场景)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，craftsman 例子贯穿始终易于理解)
- 价值: ⭐⭐⭐⭐ (为因果 HRL 建立了理论基础，定向策略有实际加速效果)

<!-- RELATED:START -->

## 相关论文

- [Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning](graph-assisted_stitching_for_offline_hierarchical_reinforcement_learning.md)
- [Confounding Robust Deep Reinforcement Learning: A Causal Approach](../../NeurIPS2025/reinforcement_learning/confounding_robust_deep_reinforcement_learning_a_causal_approach.md)
- [Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/structural_information-based_hierarchical_diffusion_for_offline_reinforcement_le.md)
- [Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](divide_and_conquer_grounding_llms_as_efficient_decision-making_agents_via_offlin.md)
- [HCPO: Hierarchical Conductor-Based Policy Optimization in Multi-Agent Reinforcement Learning](../../AAAI2026/reinforcement_learning/hcpo_hierarchical_conductor-based_policy_optimization_in_multi-agent_reinforceme.md)

<!-- RELATED:END -->
