---
title: >-
  [论文解读] Solving Continuous Mean Field Games: Deep Reinforcement Learning for Non-Stationary Dynamics
description: >-
  [NeurIPS 2025][平均场博弈] 提出DEDA-FP算法，首次在连续状态/动作空间的非平稳平均场博弈（MFG）中同时学习Nash均衡策略和种群分布，通过结合深度RL计算最优响应、监督学习表示平均策略、条件Normalizing Flow建模时变种群分布，实现了比现有方法快10倍以上的采样效率。
tags:
  - NeurIPS 2025
  - 平均场博弈
  - 深度强化学习
  - Fictitious Play
  - Normalizing Flow
  - Nash均衡
---

# Solving Continuous Mean Field Games: Deep Reinforcement Learning for Non-Stationary Dynamics

**会议**: NeurIPS 2025  
**arXiv**: [2510.22158](https://arxiv.org/abs/2510.22158)  
**代码**: 暂无  
**领域**: 强化学习  
**关键词**: 平均场博弈, 深度强化学习, Fictitious Play, Normalizing Flow, Nash均衡

## 一句话总结

提出DEDA-FP算法，首次在连续状态/动作空间的非平稳平均场博弈（MFG）中同时学习Nash均衡策略和种群分布，通过结合深度RL计算最优响应、监督学习表示平均策略、条件Normalizing Flow建模时变种群分布，实现了比现有方法快10倍以上的采样效率。

## 研究背景与动机

### 领域现状

平均场博弈（MFG）是建模大规模多智能体系统的有力框架：通过取智能体数量趋于无穷的极限，将多智能体问题简化为一个代表性智能体与种群分布之间的交互。MFG在经济学、金融、交通、通信等领域有广泛应用，且在这些领域中状态-动作空间自然是连续的，种群分布通常随时间演化（非平稳）。

### 现有痛点

现有MFG求解方法存在三大局限：

**离散限制**：大多数RL方法（Guo等2019; Elie等2020; Laurière等2022a）限于有限状态/动作空间

**平稳假设**：大部分方法（Perrin等2021; Angiuli等2023）假设种群分布不随时间变化

**密度不可得**：现有方法无法直接获取种群分布的概率密度值 $\mu(x)$，只能采样——这使得涉及局部密度依赖的MFG（如拥堵模型）无法精确求解

没有任何现有RL算法能够学习连续空间非平稳MFG的完整解（Nash均衡策略+种群分布）。

### 本文切入角度

核心思路是构建三组件方法：(1) 用DRL（SAC/PPO）计算最优响应；(2) 用监督学习逼近平均策略以获得Nash均衡策略；(3) 用时间条件Normalizing Flow建模非平稳种群分布，既能采样又能计算精确密度。整体嵌入在Fictitious Play（虚拟博弈）的迭代框架中。

## 方法详解

### 整体框架

DEDA-FP基于Fictitious Play(FP)算法的迭代框架，每轮迭代包含三个步骤：
1. 使用深度RL计算对当前平均策略的最优响应
2. 通过监督学习更新平均策略的神经网络表示
3. 训练条件Normalizing Flow学习平均策略诱导的时变种群分布

### 关键设计

1. **最优响应计算（DRL）**：使用SAC或PPO来求解：$\pi_k^* = \arg\max_\pi J_{\mu_0}^N(\pi, \bar{G}_{k-1})$，其中 $\bar{G}_{k-1}$ 是上一轮学到的种群分布模型。每次rollout时，$N-1$ 个虚拟智能体使用当前平均策略 $\bar{\pi}_{k-1}$ 来模拟种群行为。这是标准的DRL过程，但环境中的种群分布来自学到的生成模型而非经验采样。

2. **Nash均衡策略学习（监督学习）**：维护一个replay buffer $\mathcal{M}_{SL}$ 存储所有历史最优响应的 $(t, s, a)$ 三元组。训练一个策略网络 $\bar{\pi}^{\bar{\theta}}$ 最小化负对数似然：

$$\mathcal{L}_{\text{NLL}}(\bar{\theta}) = \mathbb{E}_{(t,s,a) \sim \mathcal{M}_{SL}}\left[-\log \bar{\pi}^{\bar{\theta}}(a|t,s)\right] = -\frac{1}{M}\sum_{i=1}^{M}\log \mathcal{N}(a_i; \mu_{\bar{\theta}}(s_i, t_i), \sigma_{\bar{\theta}}(s_i, t_i))$$

策略网络的输出为条件高斯分布的均值和标准差，同时以时间 $t$ 和状态 $s$ 为输入。这解决了"神经网络难以直接求平均"的问题。

3. **条件Normalizing Flow（CNF）建模种群分布**：使用Neural Spline Flow的自回归变体，以时间 $t$ 为条件变量。给定样本 $\mathbf{x}$ 和时间 $t$，密度通过变换公式计算：

$$p(\mathbf{x}|t) = p_0(f^{-1}(\mathbf{x}, t))\left|\det\left(\frac{\partial f^{-1}(\mathbf{x}, t)}{\partial \mathbf{x}}\right)\right|$$

训练目标为最大似然估计（等价于最小化NLL）。CNF的关键优势是**既能采样又能计算密度**，这对局部密度依赖的MFG至关重要。

### 收敛性分析

Theorem 1给出了exploitability的收敛界，它依赖三种误差源：
- **最优响应误差** $\epsilon_{br}^k$：DRL策略的次优性  
- **平均策略误差** $\epsilon_{sl}^k$：监督学习的拟合误差
- **分布误差** $\epsilon_{cnf}^k$：Normalizing Flow的密度估计误差

$$e_k^{\text{true}} < C_0 e_0^{\text{cnf}} + \frac{1}{k}\sum_{i=1}^{k-1}\left[(i+1)\epsilon_{br}^{i+1} + C_1(\epsilon_{sl}^{i+1} + \epsilon_{cnf}^{i+1}) + \frac{C_2}{i}\right]$$

## 实验关键数据

### Beach Bar问题（局部密度依赖）

| 算法 | Exploitability趋势 | 分布质量 | 能否处理 $\mu(x)$ |
|------|-------------------|---------|------------------|
| Algo 1（简单FP） | 收敛 | 粗糙 | 需高斯卷积近似 |
| Algo 2（+策略学习） | 收敛 | 中等 | 需高斯卷积近似 |
| DEDA-FP | 收敛，无性能损失 | 光滑集中 | **直接计算精确密度** |

### Linear-Quadratic模型

| 算法 | Exploitability | 策略线性性 | 分布集中度 |
|------|---------------|-----------|-----------|
| Algo 1 | ~0（快速收敛） | 未学到策略 | 基本 |
| Algo 2 | ~0（快速收敛） | 线性，符合理论 | 中等 |
| DEDA-FP | ~0（快速收敛） | 线性，符合理论 | **最优** |

### 4-Rooms探索（2D，熵最大化）

| 指标 | Algo 1 | Algo 2 | DEDA-FP |
|------|--------|--------|---------|
| Exploitability | 收敛 | 收敛 | 收敛，相当 |
| 采样5000条轨迹时间 | 基准 | ~基准 | **快10倍以上** |
| 分布表示质量 | 差（稀疏） | 中等 | **精确（8000样本）** |
| 能否直接用密度奖励 | 否 | 否 | **是** |

### 关键发现

- DEDA-FP在保持与基准方法相当的exploitability收敛性能的同时，提供了**质量显著更高**的种群分布表示
- CNF的采样效率比传统轨迹模拟方法快**10倍以上**
- 在局部密度依赖的问题中（Beach Bar, 4-Rooms），DEDA-FP**无需近似**即可直接使用密度 $\mu(x)$
- 学到的策略在LQ模型中呈现期望的线性特性，验证了方法的正确性

## 亮点与洞察

- **完整解决MFG**：首个同时学习Nash均衡策略和分布的DRL方法，填补了连续空间非平稳MFG的空白
- **CNF的精妙选择**：不同于GAN或扩散模型只能采样，Normalizing Flow同时提供采样和密度评估能力，完美契合MFG问题需求
- **模块化设计**：三个组件（DRL、监督学习、CNF）各司其职，可独立改进
- **理论保障**：误差传播分析明确了三种误差源对最终exploitability的贡献

## 局限与展望

- 理论分析尚不完整，特别是深度网络训练的精细分析缺失
- 仅考虑了标准MFG，未扩展到多种群、graphon博弈或含公共噪声的MFG
- exploitability的近似评估依赖于环境近似，评估本身不完全精确
- Normalizing Flow在高维状态空间中的表达能力和训练稳定性有待验证
- 未考虑真实世界应用中的实际部署问题

## 相关工作与启发

- 与Perrin等(2021)的对比：他们处理连续空间但限于平稳MFG且不学习Nash均衡策略
- 与Laurière等(2022a)的对比：他们学习Nash均衡策略但限于离散空间
- 与Zaman等(2020)的对比：他们处理非平稳连续空间但限于LQ模型
- Heinrich & Silver(2016)的Neural Fictitious Self-Play启发了使用监督学习来逼近平均策略的方案

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次统一解决连续空间非平稳MFG的三个难题
- 实验充分度: ⭐⭐⭐⭐ 三个递进复杂度的实验+附加金融应用，但高维实验缺乏
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法步骤递进展示（Algo 1→2→3）
- 价值: ⭐⭐⭐⭐ 打开了用DRL解决复杂MFG的大门，但通用性仍需更多验证
