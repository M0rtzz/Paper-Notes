---
title: >-
  [论文解读] MOBO-OSD: Batch Multi-Objective Bayesian Optimization via Orthogonal Search Directions
description: >-
  [NeurIPS 2025][优化/理论][多目标贝叶斯优化] 提出MOBO-OSD算法，通过在逼近的个体极小值凸包（CHIM）上定义正交搜索方向来生成多样化的Pareto最优解，结合Pareto前沿估计和批量选择策略，在合成与真实基准上持续超越SOTA多目标贝叶斯优化方法。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "多目标贝叶斯优化"
  - "Pareto前沿"
  - "正交搜索方向"
  - "超体积改进"
  - "批量优化"
---

# MOBO-OSD: Batch Multi-Objective Bayesian Optimization via Orthogonal Search Directions

**会议**: NeurIPS 2025  
**arXiv**: [2510.20872](https://arxiv.org/abs/2510.20872)  
**代码**: [GitHub](https://github.com/LamNgo1/mobo-osd)  
**领域**: 优化  
**关键词**: 多目标贝叶斯优化, Pareto前沿, 正交搜索方向, 超体积改进, 批量优化

## 一句话总结

提出MOBO-OSD算法，通过在逼近的个体极小值凸包（CHIM）上定义正交搜索方向来生成多样化的Pareto最优解，结合Pareto前沿估计和批量选择策略，在合成与真实基准上持续超越SOTA多目标贝叶斯优化方法。

## 研究背景与动机

多目标贝叶斯优化（MOBO）旨在有限评估预算下优化多个互相冲突的黑盒目标函数。现有方法面临三大挑战：

**多样性不足**: 许多基于标量化的方法（如ParEGO）无法捕获多样化的Pareto前沿，导致超体积指标次优

**可扩展性差**: DGEMO等方法无法处理超过3个目标（$M > 3$）；qEHVI在目标数增加时计算成本急剧上升

**批量优化复杂**: 在批量设置中需要建模未观测点之间跨目标的交互，现有方法处理不够优雅

本文的核心洞察源于经典的**Normal Boundary Intersection（NBI）方法**：目标空间边界与垂直于个体极小值凸包的向量的交点往往是Pareto最优解。NBI方法能生成分布均匀的Pareto解，但需要大量函数评估。作者将这一几何直觉迁移到有限预算的贝叶斯优化场景中。

## 方法详解

### 整体框架

MOBO-OSD的流程可概括为四步：
1. 构造近似CHIM并定义正交搜索方向（OSD）
2. 沿每个OSD求解约束优化子问题
3. 通过Pareto前沿估计（PFE）在现有解附近探索更多候选点
4. 使用超体积改进（HVI）和Kriging Believer批量选择评估点

### 关键设计

1. **近似CHIM构造**: 由于无法获得真实的个体极小值，作者用观测数据的理想点和天顶点构造 $M$ 个边界点 $\mathbf{p}_m$。第 $m$ 个边界点将理想点的第 $m$ 维替换为天顶值。近似CHIM为这些边界点的凸包 $\mathcal{U}(\boldsymbol{\beta}) = \sum_{m=1}^M \beta_m \mathbf{p}_m$。与直接使用已观测到的个体极小值相比，这种方法**避免了过早缩小搜索区域**。

2. **正交搜索方向与子问题**: OSD定义为经过近似CHIM上某点、沿法向量 $\mathbf{n}$ 方向的直线。子问题为约束优化：

    $(\mathbf{x}^{\text{OSD}}, \lambda^{\text{OSD}}) \in \max_{\mathbf{x} \in \mathcal{X}} \lambda \quad \text{s.t.} \quad \gamma(\mathbf{x}; \boldsymbol{\beta}, \mathbf{n}) \in \mathcal{Q}(\mathbf{x})$

   其中将目标值替换为GP后验均值 $\boldsymbol{\mu}(\mathbf{x})$，并将等式约束松弛为置信区间约束 $\mathcal{Q}(\mathbf{x})$。使用SLSQP求解器，多起点策略后通过超体积贡献选择最优解。

3. **Pareto前沿估计（PFE）**: 对每个OSD解 $\mathbf{x}^{\text{OSD}}$，利用一阶近似法计算局部探索空间 $\mathcal{T}$，在其中随机采样 $n_e$ 个额外候选点。这避免了求解过多密集子问题。

4. **批量选择策略**: 结合Kriging Believer（每选一个点后更新GP）和探索空间多样性约束——确保来自不同OSD的候选点差异不超过1。使用Riesz s-Energy方法在单位单纯形上生成均匀分布的权重向量 $\{\boldsymbol{\beta}\}$。

### 损失函数 / 训练策略

- 采集函数为超体积改进（HVI）：$\alpha_{\text{HVI}}(\mathbf{x}) = \text{HV}(\boldsymbol{\mu}(\mathbf{x}) \cup \mathcal{P}_f, \mathbf{r}) - \text{HV}(\mathcal{P}_f, \mathbf{r})$
- 每个目标函数独立建模一个GP代理模型
- 默认使用 $n_\beta = 20$ 个OSD方向

## 实验关键数据

### 主实验（顺序优化，batch size = 1）

| 基准问题 | 目标数 | MOBO-OSD | qEHVI | DGEMO | USeMO | NSGA-II |
|---|---|---|---|---|---|---|
| DTLZ2-M2 | 2 | **最优** | 次优 | 竞争力强 | 一般 | 较差 |
| DTLZ2-M3 | 3 | **最优** | 次优 | 竞争力强 | 一般 | 较差 |
| DTLZ2-M4 | 4 | **最优** | 次优 | 不支持 | 一般 | 较差 |
| ZDT1 | 2 | **最优** | 次优 | 竞争力强 | 一般 | 较差 |
| VLMOP2 | 2 | **最优** | 竞争力强 | 次优 | 一般 | 较差 |
| Speed Reducer | 2 | **最优** | 竞争力强 | 次优 | 一般 | 较差 |
| Car Side Design | 3 | **最优** | 次优 | 竞争力强 | 一般 | 较差 |
| Marine Design | 4 | **最优** | 次优 | 不支持 | 一般 | 较差 |
| Water Planning | 6 | **最优** | 次优 | 不支持 | 一般 | 较差 |

### 消融实验（PFE组件影响）

| 配置 | DTLZ2-M2 (HV) | VLMOP2 (HV) | Car Side Design (HV) |
|---|---|---|---|
| 无PFE ($n_\beta=20$) | 0.4041±0.0004 | 0.2713±0.0020 | 145.12±0.33 |
| 无PFE ($n_\beta=100$) | 0.4118±0.0001 | 0.2978±0.0011 | 154.32±0.27 |
| 无PFE ($n_\beta=200$) | 0.4142±0.0001 | 0.3076±0.0006 | 157.23±0.21 |
| 无PFE ($n_\beta=500$) | 0.4164±0.0001 | 0.3159±0.0004 | 160.28±0.21 |
| **默认（含PFE）** | **0.4217±0.0000** | **0.3383±0.0000** | **177.48±0.23** |

### 关键发现

- MOBO-OSD在所有9个基准问题、顺序和批量设置下均一致超越SOTA
- PFE组件显著提升效率：相当于额外增加20+倍的搜索方向
- 算法对 $n_\beta$ 参数鲁棒，$n_\beta \in \{10, 50, 100\}$ 性能稳定
- 可扩展至6个目标，而DGEMO仅支持≤3个目标

## 亮点与洞察

- 将NBI方法的几何直觉优雅地嵌入贝叶斯优化框架，同时克服了NBI需要大量评估的限制
- 近似CHIM的设计避免了过早收缩搜索空间的问题，这是一个精妙的工程选择
- 批量选择策略中探索空间多样性约束的设计简洁而有效
- OSD子问题的双目标选择步骤（最大化 $\lambda$ 与最小化偏离距离）体现了平衡探索与利用的思想

## 局限与展望

- 目前仅处理**无噪声观测**，实际应用中往往存在观测噪声
- 未与HVKG等最新方法全面比较（但文中提到了兼容性）
- OSD的正交方向依赖于法向量 $\mathbf{n}$ 的计算，对高度非凸Pareto前沿可能需要更灵活的方向策略
- 可考虑结合多保真度信息进一步降低计算成本

## 相关工作与启发

- NBI方法为MOBO领域提供了新视角，暗示经典多目标优化中的几何方法可被更多借鉴
- 与DGEMO同样使用PFE技术但搜索策略更系统化（有序OSD vs 随机搜索）
- 启发：其他经典优化中的几何分解方法是否也能被引入到BO框架中？

## 评分

- 新颖性: ⭐⭐⭐⭐ 将NBI引入MOBO是新颖的，但各组件（PFE、KB）是已有技术
- 实验充分度: ⭐⭐⭐⭐⭐ 9个基准、多种batch size、详尽消融，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观
- 价值: ⭐⭐⭐⭐ 为MOBO提供了强大且可扩展的新baseline

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Multi-Objective Bayesian Optimization via Adaptive ε-Constraints Decomposition](../../ICML2026/optimization/multi-objective_bayesian_optimization_via_adaptive_varepsilon-constraints_decomp.md)
- [\[NeurIPS 2025\] Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis](learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)
- [\[NeurIPS 2025\] In Search of Adam's Secret Sauce](in_search_of_adams_secret_sauce.md)
- [\[NeurIPS 2025\] Optimistic Online-to-Batch Conversions for Accelerated Convergence and Universality](optimistic_online-to-batch_conversions_for_accelerated_convergence_and_universal.md)
- [\[NeurIPS 2025\] Multi-head Transformers Provably Learn Symbolic Multi-step Reasoning via Gradient Descent](multi-head_transformers_provably_learn_symbolic_multi-step_reasoning_via_gradien.md)

</div>

<!-- RELATED:END -->
