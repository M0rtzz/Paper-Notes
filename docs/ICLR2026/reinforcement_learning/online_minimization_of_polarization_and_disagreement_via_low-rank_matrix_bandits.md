---
title: >-
  [论文解读] Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits
description: >-
  [ICLR 2026][观点极化] 将Friedkin-Johnsen观点动力学模型下极化+分歧最小化问题首次形式化为在线低秩矩阵bandit问题（OPD-Min），提出两阶段算法OPD-Min-ESTR通过子空间估计将维度从 $|V|^2$ 降至 $O(|V|)$，在合成和真实网络上显著优于全维度线性bandit基线。
tags:
  - ICLR 2026
  - 观点极化
  - Friedkin-Johnsen模型
  - 低秩矩阵bandit
  - 遗憾最小化
  - 社交网络干预
---

# Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits

**会议**: ICLR 2026  
**arXiv**: [2510.00803](https://arxiv.org/abs/2510.00803)  
**代码**: [GitHub](https://github.com/FedericoCinus/online-min-pol)  
**领域**: online learning, social network  
**关键词**: 观点极化, Friedkin-Johnsen模型, 低秩矩阵bandit, 遗憾最小化, 社交网络干预

## 一句话总结

将Friedkin-Johnsen观点动力学模型下极化+分歧最小化问题首次形式化为在线低秩矩阵bandit问题（OPD-Min），提出两阶段算法OPD-Min-ESTR通过子空间估计将维度从 $|V|^2$ 降至 $O(|V|)$，在合成和真实网络上显著优于全维度线性bandit基线。

## 研究背景与动机

**领域现状**：社交媒体可加剧观点极化和社会分裂。Friedkin-Johnsen (FJ) 模型是研究观点动力学的经典模型：每个agent的表达观点是其内在观点与邻居表达观点的加权平均，最终收敛到唯一均衡 $\bm{z}^* = (\mathbf{I} + \mathbf{L})^{-1}\bm{s}$，其中 $\mathbf{L}$ 是图Laplacian，$\bm{s}$ 是内在观点向量。

**现有痛点**：Musco et al. (2018)开创了在FJ均衡处最小化极化和分歧的研究，但假设完全知道所有agent的内在观点。现实中获取内在观点代价高昂，可能需要广泛的用户调查或行为分析，在隐私约束下甚至不可能。

**核心矛盾**：已有部分放松假设的工作——二值内在观点（Chen et al., 2018）、SDP上界优化（Chaitanya et al., 2024）、有限查询推理（Cinus et al., 2025）——但均未解决真正的在线设置：内在观点完全未知且不可查询，仅能通过每次干预后的标量反馈（极化+分歧值）来学习。

**本文方案**：将问题形式化为Online Polarization and Disagreement Minimization (OPD-Min)，建立社交媒体算法干预与多臂bandit理论的关键连接。利用目标函数的秩一结构 $f(\mathbf{X}) = \langle \bm{s}\bm{s}^\top, \mathbf{X} \rangle$，设计高效的低秩矩阵bandit算法。

## 方法详解

### 整体框架

在每个时间步 $t = 1, \ldots, T$：
1. 学习器从有限干预集 $\mathcal{L}$ 中选择一个图Laplacian $\mathbf{L}_t$，等价于选择forest matrix $\mathbf{X}_t = (\mathbf{I} + \mathbf{L}_t)^{-1}$
2. FJ动力学收敛至均衡，学习器观察含噪损失 $Y_t = \langle \mathbf{\Theta}^*, \mathbf{X}_t \rangle + \eta_t$，其中 $\mathbf{\Theta}^* = \bm{s}\bm{s}^\top$ 为秩一未知参数
3. 目标：最小化累积遗憾 $R_T = \sum_{t=1}^T [f(\mathbf{X}_t) - f(\mathbf{X}^*)]$

关键难点在于：直接线性化为 $|V|^2$ 维bandit会导致遗憾界为 $\tilde{O}(|V|^2\sqrt{T})$；现有低秩矩阵bandit假设可从连续空间（如高斯分布）采样，在forest matrix的离散结构化行动集上不适用。

### 关键设计1: 子空间估计（Stage 1）

**功能**：从 $T_1$ 轮探索中恢复未知参数矩阵 $\mathbf{\Theta}^*$ 的低维子空间。

**核心思路**：使用nuclear norm正则化最小二乘估计：

$$\widehat{\mathbf{\Theta}} = \arg\min_{\mathbf{\Theta}} \frac{1}{2T_1} \sum_{t=1}^{T_1} (Y_t - \langle \mathbf{X}_t, \mathbf{\Theta} \rangle)^2 + \lambda_{T_1} \|\mathbf{\Theta}\|_{\text{nuc}}$$

正则化参数 $\lambda_{T_1} = 2\sqrt{2\log(2|V|/\delta)/T_1}$。在Restricted Strong Convexity (RSC)条件下，估计误差满足：

$$\|\widehat{\mathbf{\Theta}} - \mathbf{\Theta}^*\|_F^2 \leq \frac{36\log(2|V|/\delta)}{\kappa^2 T_1}$$

**设计动机**：不同于已有工作假设"nice exploration distribution"，本文针对forest matrix集合直接证明RSC条件成立。RSC的曲率参数 $\kappa = \kappa_{\min}(\mathcal{X})$ 衡量行动集的多样性，通过Talagrand集中不等式和Rademacher过程控制统计偏差。

### 关键设计2: 降维线性Bandit（Stage 2）

**功能**：利用估计的子空间将问题降至 $O(|V|)$ 维。

**核心思路**：提取 $\widehat{\mathbf{\Theta}}$ 的top eigenvector $\hat{\bm{s}}$，构造正交基 $[\hat{\bm{s}}, \hat{\mathbf{S}}_\perp]$。对每个arm $\mathbf{X}$ 做旋转 $\mathbf{X}' = [\hat{\bm{s}}, \hat{\mathbf{S}}_\perp]^\top \mathbf{X} [\hat{\bm{s}}, \hat{\mathbf{S}}_\perp]$，仅保留第一行和第一列形成 $k = 2|V|-1$ 维特征向量 $\bm{x}_{\text{sub}}$。在此低维空间中运行标准线性bandit（如OFUL）。

**设计动机**：由于 $\mathbf{\Theta}^*$ 秩一，信号集中在 $\bm{s}$ 的方向上，正交补空间的投影误差可通过Davis-Kahan $\sin\theta$ 定理控制，随 $T_1$ 增大而衰减。

### 遗憾界

**Theorem 4.1**：在RSC条件下，最优选取 $T_1 \asymp \frac{1}{\|\bm{s}\|^2 \kappa} \sqrt{T \log(2|V|/\delta)}$，总遗憾为：

$$R_T = \widetilde{\mathcal{O}}\left(\max\left\{\frac{1}{\kappa}, \sqrt{|V|}\right\}\sqrt{|V| \cdot T}\right)$$

$\sqrt{T}$ 关于时间是最优的，$|V|$ 代替 $|V|^2$ 体现了降维的效果。

## 实验关键数据

### 主实验：合成网络上的累积遗憾

| 方法 | $|V|=8$ ER (regret/runtime) | $|V|=16$ ER (regret/runtime) | $|V|=8$ SBM | $|V|=16$ SBM |
|------|----------------------------|------------------------------|-------------|-------------|
| OPD-Min (本文) | 低遗憾 / **快** | 低遗憾 / **快** | 低遗憾 / **快** | 低遗憾 / **快** |
| Full-dim OFUL | 高遗憾 / 慢 | **显著**更高遗憾 / **极慢** | 高遗憾 / 慢 | 高遗憾 / 慢 |
| Oracle Subspace | 最低遗憾 / 快 | 最低遗憾 / 快 | 最低遗憾 / 快 | 最低遗憾 / 快 |

$|V|=16$ 时差距尤为明显：OFUL遗憾和运行时间均显著更差，OPD-Min紧密追踪oracle基线。

### 消融实验：RSC参数验证

| 图类型 | Arm regime | $|V|=32$ | $|V|=128$ | $|V|=1024$ |
|-------|-----------|----------|-----------|------------|
| ER | Diverse | 0.393 | 0.410 | 0.499 |
| ER | Local | $1.68 \times 10^{-5}$ | $1.49 \times 10^{-7}$ | $2.21 \times 10^{-7}$ |
| SBM | Diverse | 0.386 | 0.476 | 0.462 |
| SBM | Local | $2.97 \times 10^{-6}$ | $8.05 \times 10^{-7}$ | $1.05 \times 10^{-7}$ |

Diverse arm regime下 $\kappa$ 合理（≈0.4-0.5），Local regime因arm近共线导致 $\kappa$ 极小。

### 关键发现

- OPD-Min在所有网络拓扑和参数设置下均一致优于全维度OFUL，且运行时间优势随 $|V|$ 增大更显著
- 算法可扩展至 $|V|=1024$ 节点
- 在真实网络（Florentine families, Karate club, Les Misérables）上结果一致
- 在线算法可快速超越离线SDP基线（Chaitanya et al., 2024），仅约250次干预后即可找到更优策略
- 极化分布（bimodal opinions）下干预更容易识别和利用

## 亮点与洞察

- 首次将观点极化最小化与在线学习/bandit理论连接，开创全新研究方向
- 针对forest matrix的结构化行动集提供自洽的RSC分析，不依赖已有低秩bandit的连续采样假设
- 从 $|V|^2$ 到 $O(|V|)$ 的维度降低同时改善统计效率和计算效率
- 仅需标量反馈（全局极化+分歧值），无需观测个体观点，隐私友好

## 局限与展望

- RSC参数 $\kappa$ 在Local arm regime下极小，理论界可能过于悲观
- 仅考虑标量均衡反馈，更丰富的信号（如社区级极化）可改善性能
- 假设FJ动力学在干预之间收敛到均衡，忽略了非均衡状态的影响
- 内在观点随时间不变的假设在现实中可能过强
- 行动集限于无向图Laplacian，未覆盖有向图或其他干预形式

## 相关工作与启发

- **Musco et al. (2018)**：开创离线极化最小化，本文将其推广到在线设置
- **LowESTR**（Lu et al., 2021）和**G-ESTT**（Kang et al., 2022）：本文在其ESTR框架基础上针对OPD-Min的特殊结构做定制化分析
- **LowPopArt**（Jang et al., 2024）：基于optimal design的低秩bandit方法，但计算复杂度 $O(|V|^6)$ 在社交网络中不可行
- 对推荐系统干预设计的启发：可将类似框架应用于内容推荐中的极化缓解

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次建立观点动力学与在线bandit理论的连接，问题建模极为精巧
- 实验充分度: ⭐⭐⭐⭐ 合成+真实网络实验丰富，含RSC验证和可扩展性分析，但缺乏真实平台数据
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨完整，主文与附录组织清晰
- 价值: ⭐⭐⭐⭐ 开创性工作，但实际部署到社交平台仍有距离

<!-- RELATED:START -->

## 相关论文

- [Revisiting Matrix Sketching in Linear Bandits: Achieving Sublinear Regret via Dyadic Block Sketching](revisiting_matrix_sketching_in_linear_bandits_achieving_sublinear_regret_via_dya.md)
- [Beyond the Lower Bound: Bridging Regret Minimization and Best Arm Identification in Lexicographic Bandits](../../AAAI2026/reinforcement_learning/beyond_the_lower_bound_bridging_regret_minimization_and_best_arm_identification_.md)
- [Shift Before You Learn: Enabling Low-Rank Representations in Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/shift_before_you_learn_enabling_low-rank_representations_in_reinforcement_learni.md)
- [Single Index Bandits: Generalized Linear Contextual Bandits with Unknown Reward Functions](single_index_bandits_generalized_linear_contextual_bandits_with_unknown_reward_f.md)
- [AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)

<!-- RELATED:END -->
