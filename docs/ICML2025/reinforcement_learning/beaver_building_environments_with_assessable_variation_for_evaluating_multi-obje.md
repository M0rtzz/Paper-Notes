---
title: >-
  [论文解读] BEAVER: Building Environments with Assessable Variation for Evaluating Multi-Objective Reinforcement Learning
description: >-
  [ICML 2025][强化学习][多目标RL] 提出 BEAVER 基准——首个面向建筑能源管理的多目标上下文强化学习评估框架，通过参数化热动力学和气候区域构建可控环境变化，系统评估现有 MORL 算法的跨环境泛化能力。 领域现状：基于强化学习的建筑 HVAC 控制在单一仿真环境中取得了成功，能够有效优化温控和能耗。然而…
tags:
  - "ICML 2025"
  - "强化学习"
  - "多目标RL"
  - "建筑能源管理"
  - "MOC-MDP"
  - "泛化评估"
  - "上下文MDP"
  - "Pareto前沿"
---

# BEAVER: Building Environments with Assessable Variation for Evaluating Multi-Objective Reinforcement Learning

**会议**: ICML 2025  
**arXiv**: [2507.07769](https://arxiv.org/abs/2507.07769)  
**代码**: [https://github.com/chennnnnyize/BEAVER](https://github.com/chennnnnyize/BEAVER)  
**领域**: 强化学习  
**关键词**: 多目标RL, 建筑能源管理, MOC-MDP, 泛化评估, 上下文MDP, Pareto前沿

## 一句话总结

提出 BEAVER 基准——首个面向建筑能源管理的多目标上下文强化学习评估框架，通过参数化热动力学和气候区域构建可控环境变化，系统评估现有 MORL 算法的跨环境泛化能力。

## 研究背景与动机

**领域现状**：基于强化学习的建筑 HVAC 控制在单一仿真环境中取得了成功，能够有效优化温控和能耗。然而，现实中建筑运维面临一个核心难题：不同建筑在材料、结构、地理位置上差异巨大，导致控制器在一栋楼训练的策略难以直接迁移到另一栋楼。

**现有痛点**：(1) 现有 MORL 研究假设单一静态环境，忽略了底层动力学的变化；(2) 建筑 RL 基准缺乏对多目标（舒适度 vs 能耗 vs 碳排放）和跨环境泛化的标准化评估支持；(3) 实践中部署控制器的管理人员需要面对不同建筑和不同用户偏好，现有基准无法评估这种多维变化下的鲁棒性。

**核心矛盾**：建筑的热动力学参数（热阻、热容）因材料和气候而异，但现有方法将这些差异简单忽略，导致"在实验室有效、在现场失效"的困境。

**本文目标**：构建一个标准化、自动化的基准框架，系统评估 MORL 算法在建筑控制场景中的多目标权衡能力和跨环境泛化能力。

**切入角度**：将建筑 RL 问题形式化为多目标上下文 MDP（MOC-MDP），其中"上下文"包含两类可控变量——热对流参数（影响状态转移）和气候区域（影响外部输入），并基于物理原理（RC 热网络模型 + EnergyPlus 仿真）自动化构建变化的环境集。

**核心 idea**：MOC-MDP = MOMDP + 上下文参数化，将"环境变化"和"目标偏好"解耦为独立维度，实现系统化评估。

## 方法详解

### 整体框架

BEAVER 基准由三部分组成：(1) 基于物理原理的 RC 网络建筑热动力学环境；(2) 参数化上下文变量（$U_{\text{wall}}$ 热动力学 + 气候区域）；(3) 多目标奖励设计与标准化评估指标。框架支持自动化环境构建、MORL 算法集成和定量/可视化分析。

### 关键设计

1. **MOC-MDP 形式化**：在标准 MOMDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}_{1:n}, \Omega, f, \gamma \rangle$ 基础上引入上下文空间 $\mathcal{C}$，定义映射 $\boldsymbol{M}(c)$ 将上下文 $c \in \mathcal{C}$ 映射到具体的 MOMDP。上下文 $c$ 包括气候条件和建筑热动力学参数，不同 $c$ 值改变状态转移函数 $\mathcal{P}^c$。关键假设是上下文对 agent 不可观测——如果可观测就退化为增广状态空间的 MOMDP。目标偏好 $\omega \in \Omega$ 通过线性标量化 $f_\omega(\mathbf{r}) = \omega^\top \mathbf{r}$ 映射为标量效用。

2. **参数化建筑热动力学**：采用经典 RC 网络模型描述区域温度动态：$C_i \frac{dT_i}{dt} = \sum_{j \in \mathcal{N}(i)} \frac{T_j - T_i}{R_{i,j}} + Q_i^h + Q_i^a + Q_i^s$，其中 $C_i$ 为热容、$R_{i,j}$ 为热阻。通过采样 EnergyPlus 参考建筑的墙壁 U-factor（$U_{\text{wall}}$，单位 W/m²·°C），将其转换为每面墙的 $R_i, C_i$，实现热动力学参数的系统化变化。低 U-factor 意味着更好的隔热性能。

3. **多目标奖励设计与评估指标**：定义两类奖励——热舒适度 $\mathcal{R}_{\text{thermal}} = M - 0.05 \sum_i |T_i[t] - T_i^s[t]|$（偏离设定温度的惩罚）和能源成本 $\mathcal{R}_{\text{cost}} = M - 0.05 \sum_i c[t] |P_i[t]|$（电力消耗惩罚），并额外支持功率爬坡率奖励。评估采用三个互补指标：Hypervolume（HV，Pareto 前沿近似质量）、Expected Utility（EU，平均偏好效用）和 Sparsity（SP，解集密度）。

### 损失函数 / 训练策略

- 支持两种训练模式：Static-Train（单一固定环境训练）和 Dynamic-Train（每个 episode 从分布中采样 $U_{\text{wall}}$）
- 兼容 MORL-Generalization 框架，支持 C-MORL 等约束多目标优化算法
- 5 次独立运行取平均值

## 实验关键数据

### 主实验：热动力学变化下的泛化

| 指标 | 训练模式 | Dynamics 1 | Dynamics 2 | Dynamics 3 | Dynamics 4 | Dynamics 5 |
|------|---------|-----------|-----------|-----------|-----------|-----------|
| HV($10^7$)↑ | Static-Train | 9.35±0.04 | 9.38±0.06 | **8.49±0.05** | 9.76±0.05 | 9.27±0.05 |
| HV($10^7$)↑ | Dynamic-Train | 9.38±0.06 | 9.37±0.07 | **8.59±0.09** | 9.67±0.14 | 9.32±0.09 |
| EU($10^3$)↑ | Static-Train | 9.34±0.01 | 9.33±0.02 | **9.14±0.02** | 9.47±0.02 | 9.32±0.01 |
| SP($10^5$)↓ | Static-Train | 0.36±0.18 | 0.41±0.14 | **1.27±1.26** | 0.31±0.07 | 0.65±0.42 |

### 气候变化下的泛化（训练环境：Warm Marine）

| 指标 | Mixed Marine | Cool Marine | Warm Humid | Warm Dry | Hot Humid | Warm Marine* |
|------|-------------|-------------|-----------|---------|----------|-------------|
| HV($10^7$)↑ | 8.78±0.11 | 8.74±0.10 | 8.59±0.09 | 9.59±0.05 | 10.00±0.02 | **10.13±0.14** |
| EU($10^3$)↑ | 8.80±0.06 | 8.78±0.06 | 8.71±0.05 | 9.25±0.04 | 9.58±0.02 | **9.67±0.02** |
| SP($10^5$)↓ | 0.79±0.37 | 0.80±0.30 | **7.24±0.38** | 0.31±0.09 | 0.19±0.08 | 0.19±0.09 |

### 消融实验

| 对比维度 | 动态影响 | 气候影响 |
|---------|---------|---------|
| Dynamics 3 退化 | HV降至8.49 (vs 9.35基线, -9.2%) | — |
| Warm Humid退化 | — | SP从0.19飙升到7.24 (38x) |
| Dynamic-Train vs Static-Train | HV仅微升0.03-0.10 | — |

### 关键发现

- Dynamics 3 是最具挑战性的环境变化，导致 HV 和 EU 显著下降——说明某些热动力学配置对现有 MORL 构成严峻挑战
- Dynamic-Train 相比 Static-Train 改进微弱，表明当前的采样混合训练策略不足以增强鲁棒性
- 气候变化跨区域泛化非常不稳定，仅 Hot Humid 接近训练环境性能，其余均退化
- Pareto 前沿可视化显示两种训练模式在用户舒适度目标上表现较差

## 亮点与洞察

- **填补空白**：首个将 MOC-MDP 形式化应用于建筑 HVAC 控制的标准化基准，系统化地解耦了"环境变化"和"目标偏好"两个维度
- **物理驱动**：基于 EnergyPlus 参考建筑提取真实的 $U_{\text{wall}}$ 值，提高了评估的实际意义
- **暴露关键不足**：实验清晰揭示现有 MORL 方法缺乏跨环境鲁棒性，为后续算法设计提供明确方向

## 局限与展望

- 目前仅支持单区域建筑，多区域复杂建筑布局有待扩展
- MORL 基线方法仅 C-MORL，未覆盖 GPI-PD、PGMORL 等更多算法
- Dynamic-Train 的采样策略过于简单（均匀采样），课程学习或自适应采样可能更有效
- 未来可加入初始值分布、不同占用率等更多上下文维度

## 相关工作与启发

- 与 MORL-Generalization (Teoh et al. 2025) 互补：后者提供通用 MORL 基准，BEAVER 专注建筑控制的实际场景
- C-MORL (Liu et al. 2025) 在大量目标时表现良好，但跨环境泛化仍是瓶颈
- MPC 方法 (Ma et al. 2012) 的 RC 网络建模为 BEAVER 提供了物理基础

## 评分

⭐⭐⭐⭐ — 在建筑控制这一实际应用领域提供了规范的 MORL 基准，MOC-MDP 形式化优雅且实验揭示了有价值的负面结论。方法创新主要在 benchmark 设计而非算法突破，基线方法覆盖有限。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Craftium: An Extensible Framework for Creating Reinforcement Learning Environments](craftium_bridging_flexibility_and_efficiency_for_rich_3d_single-_and_multi-agent.md)
- [\[NeurIPS 2025\] Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach](../../NeurIPS2025/reinforcement_learning/multi-objective_reinforcement_learning_with_max-min_criterion_a_game-theoretic_a.md)
- [\[ICML 2025\] Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization](graph-supported_dynamic_algorithm_configuration_for_multi-objective_combinatoria.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](../../NeurIPS2025/reinforcement_learning/thompson_sampling_for_multi-objective_linear_contextual_bandit.md)
- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](../../NeurIPS2025/reinforcement_learning/forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)

</div>

<!-- RELATED:END -->
