---
title: >-
  [论文解读] Opinion Maximization in Social Networks by Modifying Internal Opinions
description: >-
  [NeurIPS 2025][LLM/NLP][opinion dynamics] 本文研究社交网络中通过修改 k 个关键节点的内部意见来最大化整体意见的优化问题，提出了两种基于采样的近似算法（随机游走和森林采样）以及一种基于异步更新的精确算法 MIS，后者在理论上保证收敛到最优解…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "opinion dynamics"
  - "social networks"
  - "opinion maximization"
  - "random walk"
  - "asynchronous algorithm"
  - "structural centrality"
---

# Opinion Maximization in Social Networks by Modifying Internal Opinions

**会议**: NeurIPS 2025  
**arXiv**: [2510.17226](https://arxiv.org/abs/2510.17226)  
**代码**: 待确认  
**领域**: LLM/NLP  
**关键词**: opinion dynamics, social networks, opinion maximization, random walk, asynchronous algorithm, structural centrality

## 一句话总结

本文研究社交网络中通过修改 k 个关键节点的内部意见来最大化整体意见的优化问题，提出了两种基于采样的近似算法（随机游走和森林采样）以及一种基于异步更新的精确算法 MIS，后者在理论上保证收敛到最优解，并在数千万节点的真实网络上展示了卓越的效率与精度。

## 研究背景与动机

社交网络中的舆论治理对公共健康运动、政治选举和商业营销至关重要：

**意见动力学模型**：Friedkin-Johnson (FJ) 模型是经典的意见演化框架。每个 agent $i$ 有内部意见 $s_i \in [0,1]$（固有立场）和抵抗系数 $\alpha_i \in (0,1]$（对邻居意见的抵抗程度）。意见在网络中迭代传播直至均衡：$\bm{z} = (\bm{I} - (\bm{I} - \bm{R})\bm{P})^{-1} \bm{R} \bm{s}$

**整体意见优化**：整体表达意见 $f(\bm{R}, \bm{s}) = \bm{1}^\top \bm{M} \bm{s} = \sum_v \rho_v s_v$ 是所有 agent 均衡意见之和，其中 $\rho_v$ 为节点 $v$ 的**结构中心性**——衡量节点内部意见对全局意见的影响力

**传统方法的瓶颈**：精确计算结构中心性需要矩阵求逆，复杂度 $O(n^3)$，对百万级网络不可行

**已有工作的局限**：此前的意见优化主要通过修改抵抗系数、调整表达意见或改变网络结构实现，**通过修改内部意见来最大化整体意见**这一问题在算法设计上尚未得到高效解决

**核心问题（OpinionMax）**：给定社交网络和整数 $k$，如何选择 $k$ 个节点并将其内部意见设为 1，使整体均衡意见最大化？

## 方法详解

### 整体框架

三种递进式算法：

1. **RWB**（随机游走采样）：通过吸收随机游走近似估计结构中心性
2. **Forest**（森林采样）：基于生成收敛森林的组合解释估计结构中心性
3. **MIS**（异步更新精确搜索）：两阶段确定性算法，先全局粗筛再局部精修，精确识别最优 k 节点集

### 关键设计

**结构中心性与潜在影响**：

整体意见可分解为 $f(\bm{R}, \bm{s}) = \sum_v \rho_v s_v$，其中结构中心性 $\rho_v = \sum_u \bm{M}(u,v)$ 衡量节点 $v$ 对所有其他节点均衡意见的贡献。将节点 $v$ 的内部意见从 $s_v$ 修改为 1 带来的增量为 $\Delta(v) = \rho_v (1 - s_v)$，称为**潜在影响**。因此 OpinionMax 等价于找到潜在影响最大的 $k$ 个节点。

**算法一：RWB（随机游走采样）**

- 核心思想：结构中心性等于从所有节点出发的吸收随机游走在该节点被吸收的概率之和（Lemma 1）
- 实现：模拟 $N$ 次吸收随机游走，起点均匀随机选取，统计每个节点的吸收频率乘以 $n$ 作为 $\hat{\rho}_v$
- 理论保证：$N = O(\frac{n}{\epsilon^2} \log n)$ 次游走可保证 $|\hat{\rho}_i - \rho_i| \geq \epsilon$ 的概率不超过 $1/n$
- 时间复杂度：$O(\frac{\alpha_{\max}(1-\alpha_{\min})}{\epsilon^2 \alpha_{\min}^2} \cdot n \log n)$
- 近优性保证（Corollary 1）：$\sum_{i \in \hat{T}} \rho_i(1-s_i) \geq \sum_{i \in T^*} \rho_i(1-s_i) - 2k\epsilon$

**算法二：Forest（森林采样）**

- 核心思想：通过 Neumann 级数和生成收敛森林的组合解释（Lemma 3），建立结构中心性与森林根节点分布的关系
- 实现：基于 Wilson 算法的扩展，用环消随机游走采样生成收敛森林
- 时间复杂度：$O(\frac{1}{\alpha_{\min}} l n)$，其中 $l$ 为采样次数

**算法三：MIS（精确异步搜索）**——核心贡献

**第一阶段：GlobalInfApprox（全局影响近似）**

维护残差向量 $\bm{r}_a$（初始为 $\bm{1}$）和估计向量 $\hat{\bm{\Delta}}$（初始为 $\bm{0}$）。对每个残差超过阈值 $\epsilon$ 的节点 $v$，执行三步异步操作：
- 将 $(1-\alpha_v)/d_v^+ \cdot r_a(v)$ 分发给每个出邻居
- 累积 $(1-s_v)\alpha_v \cdot r_a(v)$ 到自身的潜在影响估计
- 重置 $r_a(v) = 0$

关键保证（Lemma 5）：$(1-\epsilon)\Delta(v) \leq \hat{\Delta}(v) \leq \Delta(v)$

**第二阶段：TargetedNodeRefine（候选节点精修）**

利用 Lemma 4 的精确分解 $\Delta(v) - \hat{\Delta}(v) = (1-s(v)) \cdot \bm{e}_v^\top \bm{M}^\top \bm{r}_a$，对候选集 $C$ 中的节点逐个执行反向传播精修，逐步收紧误差界。

**合并判定**（Lemma 8）：根据估计值和误差界，将节点分为三类：
- 确定属于最优集的节点 $T$
- 可能属于最优集的候选节点 $C$
- 确定不属于的节点

通过逐步缩小 $\epsilon'$（从 1 到 $1/2n$ ...），候选集 $C$ 逐步缩小直至 $|T| = k$。

### 损失函数 / 理论复杂度

MIS 的时间复杂度上界（Theorem 3）：

$$O\left(\frac{d_{\max}^+ n}{\alpha_{\min}} \log \frac{1}{\epsilon} + \frac{m}{k_{\text{gap}} \cdot \alpha_{\min}}\right)$$

其中 $k_{\text{gap}}$ 为真实潜在影响向量中第 $k$ 和第 $k+1$ 大值的差距。

## 实验关键数据

### 主实验：运行时间对比（k=64，单位：秒）

| 数据集 (节点数) | RWB (Unif.) | Forest (Unif.) | MIS (Unif.) |
|----------------|------------|---------------|------------|
| DBLP (317K) | 120.61 | 66.02 | **0.28** |
| Google (876K) | 344.02 | 162.87 | **0.39** |
| YoutubeSnap (1.1M) | 896.66 | 196.53 | **1.03** |
| Pokec (1.6M) | 1037.87 | 467.81 | **3.72** |
| Flixster (2.5M) | 2342.12 | 326.52 | **2.59** |
| LiveJournal (4.8M) | 3275.20 | 1454.37 | **8.61** |
| Twitter (41.7M) | 超时 | 12115.80 | **279.73** |
| SinaWeibo (58.7M) | 超时 | 10538.27 | **156.83** |

MIS 比 RWB 快 2-3 个数量级，比 Forest 快 1-2 个数量级。在 Twitter（4170 万节点）上 MIS 仅需约 280 秒。

### 主实验：准确性对比

| 方法 | 意见值 | Precision | NDCG |
|------|-------|-----------|------|
| topRandom | 低 | 低 | 低 |
| topDegree | 中等 | 中等 | 中等 |
| topPageRank | 中等 | 中等 | 中等 |
| RWB | 高 | 有波动 | 高 |
| Forest | 高 | 有波动 | 高 |
| **MIS** | **最优** | **1.000** | **≈1.000** |

MIS 在所有数据集和所有指标上实现了完美精度（precision = 1），NDCG 接近 1。

### 消融实验

| 消融项 | 影响 |
|-------|------|
| 抵抗系数分布（均匀/正态/指数） | 指数分布（低 $\alpha_{\min}$）显著增加 RWB 和 MIS 运行时间 |
| k 值变化 (1→1024) | 所有方法在更大 k 时性能差异缩小，但 MIS 始终最优 |
| RWB 采样数 | 需大量采样（$O(n \log n / \epsilon^2)$）才能保证精度 |
| Forest 采样数 l=4000 | 在多数网络上表现良好，但对极端分布敏感 |

### 关键发现

1. **MIS 在效率和精度上全面碾压**：在所有 8 个真实网络上，MIS 的运行时间比最快的基线 Forest 快 40-60 倍，同时保证精确最优解
2. **采样方法面临精度-效率困境**：RWB 和 Forest 需要在采样量和精度之间权衡，MIS 通过确定性迭代完全避免了这一问题
3. **抵抗系数分布影响显著**：指数分布（$\alpha_{\min}$ 较小）导致随机游走长度增加，RWB 运行时间急剧增长（最高 10×），但 MIS 的相对优势更加突出
4. **异步更新 + 渐进精修是关键**：全局粗筛快速锁定候选集 + 局部精修逐步排除边界节点的两阶段策略，大幅减少了不必要的计算
5. **可扩展性优异**：MIS 在 5870 万节点的 SinaWeibo 上仅需约 157 秒，展示了真实超大规模网络的可行性

## 亮点与洞察

- **随机游走解释的精妙**：将结构中心性与吸收随机游走的吸收概率建立等价关系，为算法设计提供了直观的概率解释
- **异步更新的工程价值**：每个节点独立收敛、支持局部终止的异步更新范式，比全局同步方法更适合大规模网络
- **两阶段策略的通用性**：先粗后精的候选集缩减思路不仅适用于意见优化，还可推广到其他 top-k 选择问题
- **理论保证完整**：从近似界（Lemma 5）到精确收敛条件（Lemma 8），每个算法组件都有严格的数学证明
- **OpinionMax 与 OpinionMin 的等价性**：证明了最大化和最小化整体意见是等价问题，一个算法即可解决两个方向的实际应用

## 局限性

1. **对低抵抗系数敏感**：当 $\alpha_{\min}$ 很小时，MIS 的运行时间显著增加（Theorem 3 中的 $1/\alpha_{\min}$ 因子）
2. **边界节点相等时的终止问题**：当真实潜在影响的第 $k$ 和第 $k+1$ 值完全相等时（$k_{\text{gap}} = 0$），算法可能无法终止
3. **静态网络假设**：仅考虑固定网络结构，未处理动态变化的社交网络
4. **单一意见维度**：仅处理 $[0,1]$ 上的一维意见，未扩展到多维或多话题意见
5. **均匀修改策略**：将选中节点意见直接设为 1（或 0），未考虑中间值修改或预算约束下的最优修改量
6. **缺乏学习组件**：纯算法方法，未与图神经网络等端到端学习方法对比

## 相关工作与启发

- **Gionis et al. (2013)**：首次提出意见最大化问题但通过设定表达意见为 1 实现，本文通过修改内部意见提供了更实际的方案
- **Abebe et al. (2018, 2021)**：研究通过修改抵抗系数来优化意见，与本文修改内部意见的策略互补
- **Zhu et al. (2021)**：研究通过链接推荐最小化极化，关注网络结构修改而非节点属性修改
- **Wilson 算法扩展**：Forest 算法基于环消随机游走和生成收敛森林，展示了组合数学工具在网络算法中的应用价值
- **启发**：异步局部更新 + 全局一致性保证的算法范式对大图上的其他优化问题（如影响最大化、社区检测）具有借鉴意义

## 评分

- 新颖性: ⭐⭐⭐⭐ 异步精确算法 MIS 是主要贡献，随机游走和森林解释提供了新颖的理论视角
- 实验充分度: ⭐⭐⭐⭐⭐ 8 个真实大规模网络、3 种抵抗系数分布、5 种基线、3 种指标，评估极其全面
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，算法伪代码清晰，但公式密度较高
- 价值: ⭐⭐⭐⭐ 为社交网络舆论优化提供了实用的精确算法，5870 万节点的扩展性令人印象深刻

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] msf-CNN: Patch-based Multi-Stage Fusion with Convolutional Neural Networks for TinyML](msf-cnn_patch-based_multi-stage_fusion_with_convolutional_neural_networks_for_ti.md)
- [\[ICML 2025\] Position: LLM Social Simulations Are a Promising Research Method](../../ICML2025/llm_nlp/llm_social_simulations_are_a_promising_research_method.md)
- [\[ACL 2025\] SocialEval: Evaluating Social Intelligence of Large Language Models](../../ACL2025/llm_nlp/socialeval_evaluating_social_intelligence_of_large_language_models.md)
- [\[CVPR 2025\] STAA-SNN: Spatial-Temporal Attention Aggregator for Spiking Neural Networks](../../CVPR2025/llm_nlp/staa-snn_spatial-temporal_attention_aggregator_for_spiking_neural_networks.md)
- [\[ACL 2025\] Internal and External Impacts of Natural Language Processing Papers](../../ACL2025/llm_nlp/internal_and_external_impacts_of_natural_language_processing_papers.md)

</div>

<!-- RELATED:END -->
