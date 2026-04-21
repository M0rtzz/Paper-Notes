---
title: >-
  [论文解读] Learning to Stop: Deep Learning for Mean Field Optimal Stopping
description: >-
  [机器人] 首次在离散时间有限状态空间下形式化并计算求解平均场最优停止（MFOS）问题，证明 MFOS 以 $O(1/N)$ 速率逼近多智能体最优停止（MAOS），并提出两种深度学习算法（直接法 DA 和动态规划法 DPP），在维度高达 300 的 6 个场景中验证有效性。
tags:
  - 机器人
---

# Learning to Stop: Deep Learning for Mean Field Optimal Stopping

****会议****: ICML 2025  
****arXiv****: [2410.08850](https://arxiv.org/abs/2410.08850)  
****作者****: Lorenzo Magnino, Yuchen Zhu, Mathieu Laurière (NYU Shanghai, Georgia Tech)  
****领域****: 优化与控制 / 机器学习  
****关键词****: 最优停止、平均场控制、多智能体、深度学习、动态规划  

## 一句话总结

首次在离散时间有限状态空间下形式化并计算求解平均场最优停止（MFOS）问题，证明 MFOS 以 $O(1/N)$ 速率逼近多智能体最优停止（MAOS），并提出两种深度学习算法（直接法 DA 和动态规划法 DPP），在维度高达 300 的 6 个场景中验证有效性。

## 研究背景与动机

最优停止（Optimal Stopping, OS）是优化领域的基础问题，核心是在不确定环境中决定何时停止一个随机过程以最小化代价。经典应用包括金融中的美式期权定价、求职问题（secretary problem）、机器人路径规划中的时空停止等。

传统 OS 研究主要聚焦于**单智能体**场景。然而现实中许多问题涉及**多智能体协作停止**（MAOS），例如：
- 无人机编队需要在不同时刻各自停止以匹配目标分布
- 多机器人搜索中每个机器人需决定何时停止探索

MAOS 的计算复杂度随智能体数 $N$ 急剧增长。平均场近似（Mean Field Approximation）是处理大规模多智能体系统的经典工具——当 $N \to \infty$ 时，智能体变为独立同分布，经验分布收敛到其自身的分布（混沌传播）。

**现有不足**：
- Talbi et al. (2023, 2024) 从纯理论角度研究了连续时空 MFOS，但其 PDE 方法在无穷维空间上不可解
- 深度学习方法仅用于**单智能体** OS（Becker et al. 2019; Herrera et al. 2023）
- **无任何工作**在离散时间有限空间下研究 MFOS，也没有计算方法

本文的核心动机是：**在离散时间有限空间框架下，建立 MFOS 的理论基础并首次提出可扩展的深度学习求解方法。**

## 方法详解

### 整体框架

论文构建了从有限智能体到平均场极限的完整理论-算法链：

```
N-agent MAOS → (N→∞) → MFOS → DPP → 深度学习求解
                         ↑                    ↓
                    O(1/N) 逼近保证      DA / DPP 两种算法
```

**问题定义**：$N$ 个智能体在有限状态空间 $\mathcal{X}$ 上运动，每个智能体在每步以概率 $p_n(x)$ 停止。停止时产生代价 $\Phi(x, \mu)$（依赖当前状态和群体分布）。目标是最小化社会代价：

$$J(p) = \sum_{m=0}^{T} \sum_{(x,a) \in \mathcal{S}} \nu_m^p(x,a) \cdot \Phi(x, \mu_m^p) \cdot a \cdot p_m(x)$$

其中 $\nu_m^p$ 是扩展状态 $(x, a)$ 的分布，$a \in \{0,1\}$ 标识智能体是否仍在运动。

### 关键理论贡献

**定理 2.2（ε-近似性）**：在 Lipschitz 条件下，MFOS 最优策略 $p^*$ 应用于 $N$-智能体问题的代价与 $N$-智能体最优策略的代价之差为 $O(1/N)$：

$$J_N(p^*, \ldots, p^*) - J_N(\hat{p}, \ldots, \hat{p}) = O(1/N)$$

**定理 3.1（动态规划原理）**：MFOS 的值函数满足 Bellman 方程：

$$V_n(\nu) = \inf_{h \in \mathcal{H}} \left[ \bar{\Phi}(\nu, h) + V_{n+1}(\bar{F}(\nu, h)) \right]$$

其中 $\bar{\Phi}(\nu, h) = \sum_{x} \nu(x,1) \Phi(x, \nu^X) h(x)$ 是单步代价。这是**首次**为离散时间 MFOS 建立 DPP。

### 两种深度学习算法

| 特性 | 直接法 (DA, Algorithm 1) | 动态规划法 (DPP, Algorithm 2) |
|------|------|------|
| **优化目标** | 直接最小化全程社会代价 $J(p)$ | 利用 DPP 逐时间步反向求解 |
| **网络输入** | $(x, \nu, t)$：状态+分布+时间 | $(x, \nu)$：状态+分布（无需时间） |
| **输出** | 停止概率 $p_n(x)$ | 单步停止概率 $h(x)$ |
| **训练方式** | 前向模拟完整轨迹，端到端优化 | 从 $T$ 到 $0$ 反向逐步训练 |
| **内存需求** | 随时间 $T$ 线性增长 | 恒定，与 $T$ 无关 |
| **收敛速度** | 较快（算力充足时） | 较慢但内存友好 |
| **适用场景** | 短时间/低维 | 长时间/高维 |

### 网络架构

- **状态编码**：one-hot 向量经 embedding 层
- **时间编码**（DA 专用）：正弦嵌入 → 2 层全连接 + SiLU
- **主干网络**：$k$ 个残差块，每块 4 层线性层 + SiLU，隐藏维度 $D$
- **输出层**：GroupNorm → 3 层全连接 + SiLU
- 1D 实验：$k=3, D=128$；2D 实验：$k=5, D=256$

### 损失函数

**DA 损失**：直接最小化社会代价
$$\mathcal{L}_{\text{DA}} = \sum_{n=0}^{T} \bar{\Phi}(\nu_n^p, p_n)$$

**DPP 损失**：对每个时间步 $n$，最小化
$$\mathcal{L}_n = \bar{\Phi}(\nu, h) + V_{n+1}(\bar{F}(\nu, h))$$
其中 $V_{n+1}$ 由后续时间步已训练的网络给出。

## 实验关键数据

### 主实验：6 个场景总览

| 实验 | 状态空间 $|\mathcal{X}|$ | 输入维度 $3|\mathcal{X}|$ | 时间 $T$ | 动力学 | 代价依赖 |
|------|------|------|------|------|------|
| Ex1: 趋向均匀 | 5 | 15 | 4 | 确定性右移 | 局部分布 $\mu(x)$ |
| Ex2: 掷骰子 | 6 | 18 | 5 | 随机（均匀） | 状态 $x$ |
| Ex3: 拥堵人群 | 6 | 18 | 4 | 随机+拥堵 | 状态 $x$ |
| Ex4: 分布代价 | 7 | 21 | 3 | 随机游走 | KL 散度到目标分布 |
| Ex5: 2D 均匀 | 25 | 75 | 4 | 确定性右移 | 局部分布 $\mu(x)$ |
| Ex6: 无人机编队 | 100 | **300** | **50** | 扩散+随机障碍 | 终端分布匹配 |

### 关键实验结果

**Example 1（趋向均匀）**：解析最优值 $V^* = \frac{T+2}{2(T+1)} = 0.6$，DA 和 DPP 均在数百次迭代内收敛到解析解。

**Example 2（掷骰子）**：
- 异步停止最优值 $V^* = 1.6525$，DA/DPP 均恢复此值
- 同步停止最优值 $\tilde{V}^* = 3.25$，远高于异步——说明异步停止类对于此类问题至关重要

**Example 3（拥堵）**：拥堵系数 $C_{\text{cong}} = 0.8$ 显著减缓群体运动。异步停止代价显著低于同步停止，验证了随机化停止时间的必要性。

**Example 6（无人机编队，维度 300）**：
- DPP 成功驱动初始分布匹配 "M"、"F"、"O"、"S" 字母形状目标分布
- 学到的策略对初始分布具有**鲁棒泛化性**——不同随机初始分布均能匹配目标
- DA 训练所需内存随 $T=50$ 大幅增加，DPP 内存恒定

### 消融与对比

| 对比维度 | DA | DPP |
|------|------|------|
| Ex1 收敛迭代数 | ~500 | ~1000（每时间步） |
| Ex6 可运行性 | 内存受限 | 稳定运行 |
| 学习率敏感性 | $10^{-4}$ 最稳定 | $10^{-4}$ 最稳定 |
| MFOS vs MAOS 逼近（Ex1）| $L_2$ 距离衰减 $\sim N^{-1/2}$ | — |

**超参数扫描**（Appendix F，Example 1）：学习率 $10^{-2}$ 训练不稳定，$10^{-3}$ 尚可，$10^{-4}$ 最稳定。训练超参数为 AdamW 优化器，batch size 128，最多 $10^4$ 次迭代。

### 计算资源

所有实验均在 RTX 4090 GPU（每个场景 ≤3 分钟）或 M2 MacBook Pro（≤7 分钟）上完成，展现了方法的高效可扩展性。

## 亮点与洞察

1. **理论-算法闭环**：从 MAOS 到 MFOS 的 $O(1/N)$ 逼近保证 → DPP → 两种深度学习算法，形成完整链路
2. **随机化停止时间的必要性**：通过简洁反例（Example 1 in Appendix）证明确定性停止策略在平均场设定下次优，必须引入随机化
3. **异步 vs 同步停止**：异步停止类在多个场景中显著优于同步，揭示了多智能体停止问题中个体化决策的重要性
4. **DPP 的内存优势**：恒定内存使得长时间高维问题可解，这是 DA 无法匹敌的
5. **策略的分布鲁棒性**：Example 6 中学到的停止规则对未见过的初始分布仍有效，展示了良好的泛化能力

## 局限性

- 未证明深度学习算法的收敛性（深度网络分析的固有困难）
- 同步 vs 异步停止的理论比较尚未完成
- 实验均为合成场景，缺少真实世界大规模验证
- 当前框架假设模型已知（转移概率已知），未扩展到 model-free/强化学习设定
- 仅考虑合作型多智能体，未涉及竞争型（Mean Field Game）设定

## 相关工作与启发

- **单智能体深度 OS**：Becker et al. (2019) 深度最优停止；Herrera et al. (2023) 随机化神经网络方法——本文将其扩展到平均场设定
- **平均场控制**：Carmona & Laurière (2023) 深度学习 MFC；Gu et al. (2023) 强化学习 MFC——本文将停止问题转化为 MFC 子类是概念性贡献
- **连续 MFOS**：Talbi et al. (2023, 2024) 建立了连续时空理论但不可计算——本文转向离散框架使计算成为可能
- **机器人应用**：Best et al. (2017, 2018) 机器人时空最优停止——本文的平均场方法为大规模机器人编队提供了可扩展方案

**启发**：将 MFOS 解释为 MFC 的特殊情形，为两个领域的交叉提供了新视角。该框架可扩展到更多涉及"何时停止"的多智能体问题，如分布式训练中的模型早停、多机器人探索终止条件等。

## 评分

| 维度 | 分数 (1-5) | 评价 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首次形式化并求解离散 MFOS，MFC 视角是概念贡献 |
| 理论深度 | ⭐⭐⭐⭐⭐ | 完整的逼近定理 + DPP 证明 |
| 实验充分性 | ⭐⭐⭐⭐ | 6 个场景覆盖从简单到高维，但缺少真实应用 |
| 实用价值 | ⭐⭐⭐ | 框架优雅但应用场景需进一步开发 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，理论与实验并重 |
| **总分** | **⭐⭐⭐⭐** | 理论扎实的开创性工作，为 MFOS 计算方法奠基 |

<!-- RELATED:START -->

## 相关论文

- [Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context](../../ACL2025/robotics/dice_idiomaticity.md)
- [Machine Learning from Explanations](machine_learning_from_explanations.md)
- [Understanding Prompt Tuning and In-Context Learning via Meta-Learning](../../NeurIPS2025/robotics/understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [BiAssemble: Learning Collaborative Affordance for Bimanual Geometric Assembly](biassemble_learning_collaborative_affordance_for_bimanual_geometric_assembly.md)
- [Certifiably Optimal Anisotropic Rotation Averaging](../../ICCV2025/robotics/certifiably_optimal_anisotropic_rotation_averaging.md)

<!-- RELATED:END -->
