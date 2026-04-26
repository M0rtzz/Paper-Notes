---
title: >-
  [论文解读] Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization
description: >-
  [ICML2025][动态算法配置] 提出 GS-MODAC，利用 GNN 将目标空间中的解映射为图结构来学习状态表征，结合 PPO 实现对多目标进化算法（MOEA）参数的动态配置，在调度和路由两类 NP-hard 组合优化问题上超越静态和已有 DRL 方法，并展现出跨问题规模和目标数量的泛化能力。
tags:
  - ICML2025
  - 动态算法配置
  - 图神经网络
  - 深度强化学习
  - 多目标进化算法
  - Pareto 优化
---

# Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization

**会议**: ICML2025  
**arXiv**: [2505.16471](https://arxiv.org/abs/2505.16471)  
**代码**: [GitHub](https://github.com/RobbertReijnen/GS-MODAC)  
**领域**: 算法配置 / 多目标组合优化  
**关键词**: 动态算法配置, 图神经网络, 深度强化学习, 多目标进化算法, Pareto 优化

## 一句话总结

提出 GS-MODAC，利用 GNN 将目标空间中的解映射为图结构来学习状态表征，结合 PPO 实现对多目标进化算法（MOEA）参数的动态配置，在调度和路由两类 NP-hard 组合优化问题上超越静态和已有 DRL 方法，并展现出跨问题规模和目标数量的泛化能力。

## 研究背景与动机

**核心问题：** 进化算法（EA）的超参数（交叉率、变异率等）对求解质量影响巨大，且最优参数在搜索过程中会随阶段变化。现有动态算法配置（DAC）方法主要针对连续优化问题设计，难以直接迁移到多目标组合优化（MOCO）。

**已有方法的不足：**

**静态配置方法**（irace、SMAC3）：在整个搜索过程中使用固定参数，无法适应搜索阶段的变化

**MADAC**（Xue et al., 2022）：虽然是 DRL-based DAC，但依赖人工设计的状态特征（精英解数量、解间距、超体积等），特征选择费时且可能次优；离散化动作空间限制了参数调控精度
3. 现有方法在大规模、多目标的组合优化问题上表现不佳，因为 COP 的解空间不光滑且目标值范围差异大

**本文动机：** 用 GNN 自动从目标空间的图结构中学习状态表征，替代人工特征工程；设计与问题实例无关的奖励函数，使模型跨不同规模和类型的问题泛化。

## 方法详解

### 整体框架

GS-MODAC 将动态算法配置建模为上下文马尔可夫决策过程（contextual MDP），核心思路是：每次搜索迭代中，将当前种群在目标空间上的分布转化为图，用 GNN 提取嵌入作为状态，DRL 智能体据此输出下一次迭代的 EA 参数。

### MDP 组件设计

**状态空间（States）：** 将目标空间映射为图结构：

- 每个节点对应一个解，节点特征为归一化的目标值
- 归一化基准：搜索过程中遇到的最优值 + 初始代的最差值
- 通过非支配排序将解划分到不同 Pareto 前沿层，同层解之间建边
- 附加特征向量：归一化的已用代数（反映剩余搜索预算）

该设计的关键优势是**状态维度与目标数量无关**，可自然扩展到任意数目的目标。

**动作空间（Actions）：** 连续值，归一化到 $[-1, 1]$，映射到 EA 参数范围。以 NSGA-II 为例：

- 交叉率 $\in [0.6, 1.0]$
- 变异率 $\in [0.0, 0.1]$

**奖励函数（Rewards）：** 基于超体积改进的渐进式奖励：

$$r_t = \begin{cases} \Delta_{\text{current}}^2 - \Delta_{\text{best}}^2 & \text{if } HV_{\text{current}} > HV_{\text{best}} \\ 0 & \text{otherwise} \end{cases}$$

其中改进比例定义为：

$$\Delta_{\text{current}} = \frac{HV_{\text{current}} - HV_{\text{initial}}}{HV_{\text{ideal}} - HV_{\text{initial}}} \times 100$$

通过平方差设计，**后期更大的超体积提升会获得更高奖励**，鼓励搜索后期的精细进化。理想超体积 $HV_{\text{ideal}}$ 通过一次高预算（如双倍）运行近似获得，使奖励与具体实例无关。

### 策略网络结构

- 2 层 GCN 提取图节点嵌入
- 全局均值池化将节点嵌入聚合为单一图嵌入
- 拼接搜索预算特征
- 线性层输出动作分布均值
- 使用 PPO 训练

### 适配的 MOEA

- **NSGA-II**：非支配排序 + 拥挤距离，主实验算法
- **MOPSO**：粒子群优化，验证方法通用性

## 实验关键数据

### 实验设置

- **问题**：柔性作业车间调度（FJSP，2/3/5 目标）和有容量车辆路由（CVRP，2 目标）
- **规模**：FJSP 5j5m / 10j5m / 25j5m；CVRP 100 / 200 / 500 客户
- **基线**：NSGA-II 默认参数、irace（静态）、SMAC3（静态）、MADAC（DRL）
- **指标**：超体积的均值（mean）、最大值（max）、标准差（std），100 实例 × 10 次运行

### 主要结果（超体积，加粗为最优）

| 问题 | 规模 | GS-MODAC 排名 | 关键发现 |
|------|------|--------------|----------|
| Bi-FJSP | 5j5m / 10j5m / 25j5m | **全部最优** | mean 和 max 均为最佳 |
| Tri-FJSP | 5j5m | 第二 | 接近 irace |
| Tri-FJSP | 10j5m / 25j5m | **最优** | 显著优于所有基线 |
| Penta-FJSP | 全部规模 | **全部最优** | 5 目标优势更明显 |
| Bi-CVRP | 100 / 200 / 500 | **最优或并列** | 与 MADAC 相当 |

### 泛化性实验

- 在小规模（如 5j5m）训练的模型可直接用于大规模（25j5m）实例，性能仍优于在目标规模上训练的静态方法
- 可迁移到训练中未见过的约束更强的问题变体
- 跨目标数量（2→3→5）也展现出良好泛化

### MOPSO 适配

在 MOPSO 上同样验证了 GS-MODAC 的有效性，证明方法不依赖于特定的 MOEA。

## 亮点与洞察

1. **图状态表征替代人工特征**：用 GNN 自动学习目标空间结构，消除手动设计状态特征的繁琐过程，且自然适配不同目标数量
2. **实例无关的奖励设计**：归一化超体积 + 渐进式平方差奖励，使单个模型可跨不同规模和类型的问题工作
3. **连续动作空间**：相比 MADAC 的离散动作，连续参数控制更精细
4. **目标数量越多优势越大**：Penta-FJSP（5 目标）上 GS-MODAC 领先幅度最大，说明图表征在高维目标空间中的优势
5. **强泛化能力**：跨规模、跨约束、跨目标数量三个维度均展现泛化性

## 局限与展望

1. **训练成本高**：大规模问题训练需数天（25j5m FJSP 最长 3 天），实际部署前需大量预训练
2. **仅验证了两类问题**：FJSP 和 CVRP，对其他 MOCO 问题（如背包、图着色等）的效果待验证
3. **理想超体积需预运行**：$HV_{\text{ideal}}$ 需要额外一次高预算运行来近似，增加了前期计算开销
4. **仅配置交叉率和变异率**：动作空间仅覆盖两个参数，对于参数更多的 EA（如带自适应机制的 CMA-ES）可能不够
5. **GCN 池化方式单一**：使用全局均值池化，可能丢失 Pareto 前沿的层次结构信息

## 相关工作与启发

- **MADAC**（Xue et al., 2022）：最直接的前驱工作，多智能体 DRL + 人工状态特征的 DAC
- **DAC-Bench**（Biedenkapp et al., 2020; Adriaensen et al., 2022）：DAC 的标准化框架和基准
- **irace / SMAC3**：经典静态算法配置方法，作为强竞争基线
- **GNN for CO**：与用 GNN 直接求解组合优化（如 Attention Model）不同，本文用 GNN 做元层控制

## 评分

- 新颖性: ⭐⭐⭐⭐ — 图状态表征 + 实例无关奖励的组合是新颖的，但整体仍是 DAC + GNN 的标准范式
- 实验充分度: ⭐⭐⭐⭐ — 两类问题、多种规模、泛化实验、MOPSO 适配、消融研究，较为全面
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、方法描述完整、图示直观
- 价值: ⭐⭐⭐⭐ — 为 MOEA 在组合优化中的自动配置提供了实用且可泛化的方案

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](../../NeurIPS2025/reinforcement_learning/sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[NeurIPS 2025\] DCcluster-Opt: Benchmarking Dynamic Multi-Objective Optimization for Geo-Distributed Data Center Workloads](../../NeurIPS2025/reinforcement_learning/dccluster-opt_benchmarking_dynamic_multi-objective_optimization_for_geo-distribu.md)
- [\[ICML 2025\] Meta-Black-Box-Optimization through Offline Q-function Learning (Q-Mamba)](meta-black-box-optimization_through_offline_q-function_learning.md)
- [\[ICML 2025\] Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning](graph-assisted_stitching_for_offline_hierarchical_reinforcement_learning.md)
- [\[ICML 2025\] Wasserstein Policy Optimization](wasserstein_policy_optimization.md)

<!-- RELATED:END -->
