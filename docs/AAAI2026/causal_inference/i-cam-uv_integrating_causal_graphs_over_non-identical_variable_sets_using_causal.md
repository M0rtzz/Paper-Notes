---
title: >-
  [论文解读] I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables
description: >-
  [AAAI 2026][因果推理][causal discovery] 提出 I-CAM-UV 方法，通过对多个变量集不同的 CAM-UV 因果图结果进行一致性约束枚举，恢复因未观测变量而丢失的因果关系，并设计基于不一致代价单调性的最优优先搜索算法高效求解。
tags:
  - "AAAI 2026"
  - "因果推理"
  - "causal discovery"
  - "causal additive model"
  - "unobserved variables"
  - "non-identical variable sets"
  - "graph integration"
  - "combinatorial search"
---

# I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables

**会议**: AAAI 2026  
**arXiv**: [2603.03207](https://arxiv.org/abs/2603.03207)  
**代码**: 未公开  
**领域**: 因果推理  
**关键词**: causal discovery, causal additive model, unobserved variables, non-identical variable sets, graph integration, combinatorial search

## 一句话总结

提出 I-CAM-UV 方法，通过对多个变量集不同的 CAM-UV 因果图结果进行一致性约束枚举，恢复因未观测变量而丢失的因果关系，并设计基于不一致代价单调性的最优优先搜索算法高效求解。

## 研究背景与动机

- **因果发现的现实需求**：从观测数据中发现因果关系是科学研究的基本任务，但随机对照试验往往因成本、伦理等原因不可行，因此观测数据的因果发现方法日益重要
- **多数据集场景普遍存在**：实际中常有多个数据集共享相同研究目标但观测到的变量集不同（如不同实验室测量不同指标），如何整合利用这些数据集是关键挑战
- **未观测混淆变量问题**：即使变量数较少（约10个），未观测的混淆变量仍然普遍存在，而大多数方法都假设无未观测混淆因子
- **简单重叠方法的局限**：对每个数据集单独估计因果图再简单叠加，会因未观测变量导致混淆而遗漏许多真实因果关系，且某些变量对在所有数据集中都未被同时观测
- **已有方法的不足**：ION/IOD/COmbINE 输出 PAG（含不确定信息），CD-MiNi 仅支持线性非高斯模型，无法处理非线性因果关系
- **CAM-UV 提供了新的机会**：CAM-UV 能识别因未观测变量导致的未观测因果路径(UCP)和未观测后门路径(UBP)信息，这为整合多数据集的因果图提供了结构一致性约束

## 方法详解

### 整体框架

给定 $m$ 个变量集不同的数据集 $V_1, \dots, V_m \subset V$，先对每个数据集运行 CAM-UV 得到混合图 $G_k = (V_k, A_k, N_k)$（含有向边和无向边），再通过 I-CAM-UV 枚举所有满足一致性约束的 DAG。核心流程：(1) 重叠所有有向边得到 $\hat{G}$；(2) 收集待定边集 $E = E_{\text{imp}} \cup E_{\text{uno}}$（因 UCP/UBP 未识别的边和从未被同时观测的变量对）；(3) 对 $E$ 中每条边赋方向或排除，枚举所有一致 DAG。

### 关键设计一：基于 UCP/UBP 的一致性定义

- **功能**：定义候选 DAG 与 CAM-UV 结果的"结构一致性"
- **核心思路**：对于每个数据集 $V_k$，已识别因果关系的变量对 $I_k$ 之间不应存在 UCP 或 UBP，而存在无向边的变量对 $N_k$ 之间必须存在 UCP 或 UBP。候选 DAG 需同时满足所有 $m$ 个数据集的一致性
- **设计动机**：CAM-UV 的无向边精确反映了未观测路径的存在性，利用这一结构信息可约束候选解空间。Theorem 1 证明在理想情况下（所有变量可覆盖、CAM-UV 无误差），真实因果图一定是一致的

### 关键设计二：不一致代价松弛

- **功能**：引入不一致代价 $C(\tilde{G})$ 将问题从精确一致性枚举松弛为近似一致性枚举
- **核心思路**：统计所有数据集上不满足一致性的变量对总数作为代价，枚举不一致代价 $\leq C^* + b$ 的所有 DAG，其中 $C^*$ 为最小代价，$b$ 为用户参数
- **设计动机**：实际中 CAM-UV 可能有估计误差（如错误的无向边或遗漏的有向边），严格一致性可能无解。松弛后方法仍可在有误差时恢复真实 DAG

### 关键设计三：基于单调性的最优优先搜索

- **功能**：设计高效的组合搜索算法枚举所有满足代价约束的 DAG
- **核心思路**：定义代价下界函数 $\tilde{C}(t, \tilde{G})$，证明其在搜索过程中具有单调性（Theorem 2），从而使用优先队列（堆）按代价升序搜索，当弹出的状态代价超过 $C^* + b$ 时提前终止
- **设计动机**：穷举所有 $3^{|E|}$ 种连接模式计算代价不可行。利用单调性，可大幅剪枝搜索空间。UCP/UBP 的存在性判定可通过 BFS 在 $O(|A|)$ 时间完成，保证每个搜索状态的评估高效

### 关键设计四：多项式时间 UCP/UBP 检测

- **功能**：设计 UCP 和 UBP 的快速检测子算法
- **核心思路**：对于 UCP，从 $v_i$ 出发做 BFS 看是否能到达 $v_j$ 的未观测父节点；对于 UBP，分别计算 $v_i$ 和 $v_j$ 的未观测父节点的祖先集 $S_i, S_j$，判断 $S_i \cap S_j \neq \emptyset$
- **设计动机**：搜索中每个状态都需反复判断 UCP/UBP 存在性，$O(|A|)$ 的检测确保整体搜索的时间效率

## 损失函数与训练

本文为非参数化方法，不涉及神经网络训练。整体流程为：(1) 对每个数据集独立运行 CAM-UV 估计算法（包含回归分析和独立性检验）；(2) 收集重叠图和待定边集；(3) 运行组合搜索枚举一致 DAG。核心优化目标为最小化不一致代价 $C(\tilde{G})$。

## 实验

### 实验设置

在 Erdős–Rényi 随机图模型上生成 100 个合成数据集（10 个变量、边概率 0.3、非线性函数），构建 $m \in \{2, 3\}$ 个变量集不同的子数据集，每个数据集有 $|U| \in \{3, 4\}$ 个未观测变量，每个数据集 1000 个样本。

### 表1：各方法在不同设置下的 F1 得分对比

| 方法 | $(m{=}2, \|U\|{=}3)$ | $(m{=}2, \|U\|{=}4)$ | $(m{=}3, \|U\|{=}3)$ | $(m{=}3, \|U\|{=}4)$ |
|------|---------|---------|---------|---------|
| I-CAM-UV | **最高 recall** | **最高 recall** | **最高 recall** | **最高 recall** |
| CAM-UV-OVL | F1 与 I-CAM-UV 接近 | F1 与 I-CAM-UV 接近 | 低于 I-CAM-UV | 低于 I-CAM-UV |
| PC-OVL | 较低 | 较低 | 较低 | 较低 |
| Imputation | 较低 | 较低 | 较低 | 较低 |
| CD-MiNi | 较低 | 较低 | 较低 | 较低 |

I-CAM-UV 在所有设置的 recall 上均显著优于所有竞争方法（包括观测和未观测变量对），但 precision 略低于 CAM-UV-OVL，整体 F1 两者接近。

### 表2：I-CAM-UV 枚举 DAG 数量与计算时间统计

| 指标 | $(m{=}2, \|U\|{=}3)$ | $(m{=}2, \|U\|{=}4)$ | $(m{=}3, \|U\|{=}3)$ | $(m{=}3, \|U\|{=}4)$ |
|------|---------|---------|---------|---------|
| 中位 DAG 数 | 较少 | 中等 | 中等 | 较多 |
| 中位枚举时间 | <1s | 数秒 | 数秒 | 数十秒 |
| 最大枚举时间 | 异常值可达数百秒 | 异常值更大 | — | — |

枚举 DAG 数量变化范围大（从 1 到上千），但精度分布呈单一聚类模式，表明大多数枚举 DAG 的质量相近。最优优先搜索在多数实例上计算时间可接受。

## 亮点

- **问题定义新颖**：首次将 CAM-UV 的 UCP/UBP 结构信息用于多变量集因果图整合，能发现简单重叠方法完全无法识别的因果关系（尤其是从未被同时观测的变量对之间的因果方向）
- **理论保证扎实**：证明了真实 DAG 在理想条件下的一致性（Theorem 1）、代价函数的单调性（Theorem 2），以及算法的正确性
- **算法设计精巧**：最优优先搜索+单调性剪枝+多项式时间 UCP/UBP 检测，将指数级搜索空间有效缩减
- **研究问题实际意义强**：多数据集变量不一致是真实科学研究中的常见场景

## 局限性

- 算法时间复杂度在最坏情况下仍为指数级 $O(3^{|E|})$，变量数较多或 $|E|$ 较大时可能不可行
- 精度高度依赖 CAM-UV 基础估计的准确性，CAM-UV 本身的理论完整性尚未完全建立
- 可能输出大量 DAG，人工逐一检查不现实，缺乏有效的压缩表示或排序机制
- 仅在合成数据上验证，缺乏真实数据集的实验验证
- 未与更多基于非线性模型的因果发现方法进行对比

## 相关工作

- **多数据集因果发现**：ION (Tillman et al. 2009)、IOD (Tillman & Spirtes 2011)、COmbINE (Triantafillou et al. 2010) 基于约束，输出 PAG 而非完整 DAG；CD-MiNi (Huang et al. 2020) 基于 LiNGAM 仅处理线性情况
- **因果加性模型**：CAM (Bühlmann et al. 2014) 假设非线性可加函数关系，CAM-UV (Maeda & Shimizu 2021) 扩展至允许未观测变量，Pham et al. (2025) 改进了 CAM-UV 的估计精度
- **函数模型方法**：LiNGAM (Shimizu et al. 2006) 处理线性非高斯情况，ANM (Hoyer et al. 2009) 处理一般性加性噪声模型

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 CAM-UV 的 UCP/UBP 信息用于多变量集整合是全新的思路
- 实验充分度: ⭐⭐⭐ — 合成数据实验设计合理但缺乏真实数据验证
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，示例丰富，算法描述规范
- 价值: ⭐⭐⭐⭐ — 解决了有实际意义的开放问题，但可扩展性待验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models](../../ICLR2026/causal_inference/distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)
- [\[ACL 2025\] CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation](../../ACL2025/causal_inference/causalrag_integrating_causal_graphs_into_retrieval-augmented_generation.md)
- [\[NeurIPS 2025\] Characterization and Learning of Causal Graphs from Hard Interventions](../../NeurIPS2025/causal_inference/characterization_and_learning_of_causal_graphs_from_hard_interventions.md)
- [\[AAAI 2026\] Learning Subgroups with Maximum Treatment Effects without Causal Heuristics](learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics.md)

</div>

<!-- RELATED:END -->
