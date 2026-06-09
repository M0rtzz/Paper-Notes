---
title: >-
  [论文解读] RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases
description: >-
  [NeurIPS 2025][关系数据库] 本文提出 RDB2G-Bench——首个评估关系数据库到图建模方法的基准框架，包含 5 个真实 RDB、12 个预测任务和约 5 万个预计算的图模型-性能对，并对 10 种自动图建模方法进行了系统比较。
tags:
  - "NeurIPS 2025"
  - "关系数据库"
  - "图建模"
  - "benchmark"
  - "图神经网络"
  - "自动建模"
---

# RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases

**会议**: NeurIPS 2025  
**arXiv**: [2506.01360](https://arxiv.org/abs/2506.01360)  
**代码**: [github.com/chlehdwon/RDB2G-Bench](https://github.com/chlehdwon/RDB2G-Bench)  
**领域**: LLM评测  
**关键词**: 关系数据库, 图建模, benchmark, 图神经网络, 自动建模

## 一句话总结

本文提出 RDB2G-Bench——首个评估关系数据库到图建模方法的基准框架，包含 5 个真实 RDB、12 个预测任务和约 5 万个预计算的图模型-性能对，并对 10 种自动图建模方法进行了系统比较。

## 研究背景与动机

关系数据库（RDB）在金融、医疗、电商等领域广泛使用。近年来，基于图的机器学习方法（将 RDB 表的行建模为节点、外键关系建模为边，然后应用 GNN）在 RDB 上的预测任务中展现了优势。

然而，RDB 到图的建模方式存在**巨大的设计空间**：
- 表的行可以建模为节点或边
- 可以选择性地包含/排除某些表和外键关系
- 不同的建模选择导致 GNN 性能差异巨大（最高可达 10%）

现有研究大多使用固定的启发式规则（如将所有行建模为节点），但这种方法并非最优。自动探索最优图建模的研究面临一个根本障碍：**评估成本极高**——每评估一个图模型都需要训练一个 GNN。

本文的核心动机是构建一个预计算的基准数据集，使研究者无需反复训练 GNN 即可评估图建模策略，从而加速该方向的研究。

## 方法详解

### 整体框架

RDB2G-Bench 的构建和使用流程：
1. **图模型设计空间定义**：选择哪些表和外键（Step 1）+ 如何表示行（节点/边）（Step 2）
2. **图模型生成**：基于约束条件枚举合法的图模型
3. **性能指标收集**：对每个图模型训练 GNN，收集训练/验证/测试性能、运行时间和参数量
4. **基准评估**：10 种建模方法在预计算数据集上的快速评估

### 关键设计

**图模型空间的约束条件**

合法的图模型必须满足：
1. 必须包含任务表（预测任务定义所在的表）
2. 所有选中的表必须通过路径连接到任务表（路径长度不超过 GNN 层数）
3. 将行建模为边的表必须恰好有 2 个外键，且其主键不被其他表引用

**十种自动建模方法**

分为三类：

*启发式方法：*
- **Random**：随机采样图模型，选最好的
- **AR2N（All-Rows-to-Nodes）**：将所有行建模为节点，包含所有表和外键——最常用的基线

*基于动作的搜索算法（4 种动作：add/remove FK边、row→edge/edge→row）：*
- **Greedy Forward (GF)**：从任务表出发贪心添加
- **Greedy Backward (GB)**：从完整图出发贪心删减
- **Greedy Local (GL)**：从随机图出发贪心搜索
- **Evolutionary Algorithm (EA)**：进化策略
- **Bayesian Optimization (BO)**：BANANAS 算法
- **Reinforcement Learning (RL)**：基于 RNN 的策略梯度

*LLM 方法：*
- **LLM**：用 Claude Sonnet-3.5 直接生成动作序列
- **LLM-CoT**：加入 Chain-of-Thought 提示设计

### 损失函数 / 训练策略

GNN 使用 Heterogeneous GraphSAGE（分类/回归任务）和 ID-GNN（推荐任务）。训练协议遵循 RelBench，固定 epoch 为 20，仅调学习率。每个实验重复 5-15 次以确保可靠性。

## 实验关键数据

### 主实验

**数据集概览**（图 2a）

| RDB | 任务名 | 任务类型 | 表数 | 图模型数 | 最优 | AR2N | 最差 |
|-----|-------|---------|------|---------|------|------|------|
| rel-avito | user-clicks | 分类(AUC%) | 8 | 944 | 67.93 | 64.66 | 60.89 |
| rel-avito | user-visits | 分类(AUC%) | 8 | 944 | 66.33 | 65.97 | 59.83 |
| rel-avito | ad-ctr | 回归(MAE↓) | 8 | 1304 | 0.039 | 0.040 | 0.044 |
| rel-event | user-repeat | 分类(AUC%) | 5 | 214 | 82.29 | 77.65 | 63.96 |
| rel-event | user-ignore | 分类(AUC%) | 5 | 214 | 82.82 | 82.22 | 74.29 |
| rel-f1 | driver-dnf | 分类(AUC%) | 9 | 722 | 74.56 | 73.14 | 67.40 |
| rel-f1 | driver-top3 | 分类(AUC%) | 9 | 722 | 81.88 | 78.11 | 75.37 |
| rel-f1 | driver-position | 回归(MAE↓) | 9 | 722 | 3.831 | 3.913 | 4.171 |
| rel-stack | post-related | 推荐(MAP%) | 7 | 7979 | 12.04 | 10.82 | 0.006 |
| rel-trial | study-outcome | 分类(AUC%) | 15 | 36863 | 70.91 | 68.09 | 62.85 |

**十种方法在 rel-f1 上的表现**（表 1，driver-top3 任务 AUC-ROC%）

| 方法 | Budget 1% | 2% | 3% | 5% |
|------|----------|-----|-----|-----|
| Best | 81.88 | 81.88 | 81.88 | 81.88 |
| AR2N | 78.11 | 78.11 | 78.11 | 78.11 |
| GF | **81.88** | **81.88** | **81.88** | **81.88** |
| GB | 80.15 | 80.56 | 80.56 | 80.56 |
| BO | 79.42 | 79.80 | 80.13 | 80.35 |
| RL | 79.04 | 79.39 | 79.44 | 80.08 |
| LLM | 80.34 | 80.54 | 80.54 | 80.61 |
| Random | 79.63 | 80.17 | 80.43 | 80.60 |

### 消融实验

**五个关键观察**

| 观察 | 核心发现 |
|------|---------|
| Obs 1 | 最优图模型比 AR2N 提升最高 10%，且通常更小更快 |
| Obs 2 | 将行建模为边（Row2Edge）的效果因任务而异，同一 RDB 上不同任务可能得出相反结论 |
| Obs 3 | 高性能图模型共享公共子结构（如特定外键关系和边建模策略） |
| Obs 4 | 不同任务需要不同图模型，跨任务 Spearman 相关通常 < 0.4 |
| Obs 5 | 图模型的有效性跨 GNN 架构泛化良好（Spearman 相关 > 0.7-0.8） |

**效率增益**

| 评估方式 | 总时间 | 加速比 |
|---------|--------|--------|
| On-the-fly（每次训练 GNN） | 850+ 小时 | 1× |
| RDB2G-Bench（预计算查询） | 2.20 小时 | **389×** |

### 关键发现

1. **"更多数据不一定更好"**：包含所有表和外键（AR2N）通常不是最优策略。选择性地使用更少但更相关的表能带来更好的性能和效率。
2. **Greedy Forward 在小预算下表现最优**：在 rel-f1 的 driver-top3 任务上仅用 1% budget 就找到了全局最优（81.88%）。
3. **复杂方法（RL、EA）在小预算下不稳定**：需要更多探索才能收敛。
4. **LLM 方法短期推理能力强但长期规划弱**：初始迭代快速提升，但随 budget 增加收益有限。
5. **简单方法（Greedy + Random）与复杂方法表现相当**：说明该领域有很大的改进空间。

## 亮点与洞察

- **首创性**：第一个专门评估 RDB-to-graph 建模策略的基准，填补了重要空白。
- **实用价值极高**：389× 的加速使大规模评估成为可能，预计算数据集直接可用。
- **发现具有实际指导意义**：
    - 不要盲目使用 AR2N——针对任务选择合适的图建模策略
    - Greedy Forward 是实用的默认选择
    - 图模型的有效性跨 GNN 泛化，说明图建模比 GNN 选择更重要
- **系统性的实验设计**：5 个真实 RDB × 12 个任务 × 50k 图模型，规模充分。

## 局限与展望

1. 图模型空间仅考虑"节点 vs 边"和"包含 vs 排除"两个维度，未涉及超边建模。
2. 仅使用 RelBench 提供的 5 个 RDB，覆盖的领域和 schema 复杂度有限。
3. 预计算数据集绑定于特定 GNN 架构和训练设置，其他设置下可能需要重新构建。
4. LLM 方法仅使用 Claude Sonnet-3.5，未与 GPT-4 等其他模型比较。
5. 未探索图模型与下游任务语义之间的理论关系。

## 相关工作与启发

- **RelBench**（Fey et al. 2024）：提供了 RDB 和预测任务，但所有方法使用同一图建模。RDB2G-Bench 在此基础上评估不同的图建模策略。
- **AutoG**（Chen et al. 2025）：利用 LLM 探索有效的图模型，但评估成本高。RDB2G-Bench 使其评估快速化。
- **RDBench、4DBInfer**：同类基准但关注不同 ML 方法在固定图建模下的表现。
- 启发：图建模选择对 GNN 性能的影响可能比 GNN 架构选择更大，值得更多关注。

## 评分

- **创新性**: ★★★★☆（首个专门评估图建模策略的基准，问题定义新颖）
- **实验规模**: ★★★★★（50k 图模型、12 任务、10 方法、10400 GPU 小时构建）
- **实用性**: ★★★★★（预计算数据集、开源代码、389× 加速）
- **清晰度**: ★★★★☆（结构清晰，5 个观察逐步展开，图表直观）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras](influx_a_benchmark_for_self-calibration_of_dynamic_intrinsics_of_video_cameras.md)
- [\[NeurIPS 2025\] FACE: Faithful Automatic Concept Extraction](face_faithful_automatic_concept_extraction.md)
- [\[NeurIPS 2025\] HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild](houselayout3d_a_benchmark_and_training-free_baseline_for_3d_layout_estimation_in.md)
- [\[NeurIPS 2025\] EvoBrain: Dynamic Multi-Channel EEG Graph Modeling for Time-Evolving Brain Networks](evobrain_dynamic_multi-channel_eeg_graph_modeling_for_time-evolving_brain_networ.md)
- [\[ICML 2026\] iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework](../../ICML2026/others/iworld-bench_a_benchmark_for_interactive_world_models_with_a_unified_action_gene.md)

</div>

<!-- RELATED:END -->
