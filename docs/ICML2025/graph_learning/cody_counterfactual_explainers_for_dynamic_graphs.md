---
title: >-
  [论文解读] CoDy: Counterfactual Explainers for Dynamic Graphs
description: >-
  [ICML 2025][图学习][反事实解释] 提出 CoDy——首个用于时序图神经网络（TGNN）的反事实解释方法，通过蒙特卡洛树搜索（MCTS）结合时空启发式策略高效探索可能的解释子图空间，在多个数据集上 AUFSC+ 提升 16%。
tags:
  - ICML 2025
  - 图学习
  - 反事实解释
  - 动态图
  - 时序图神经网络
  - 蒙特卡洛树搜索
  - 可解释性
---

# CoDy: Counterfactual Explainers for Dynamic Graphs

**会议**: ICML 2025  
**arXiv**: [2403.16846](https://arxiv.org/abs/2403.16846)  
**代码**: https://github.com/daniel-gomm/CoDy  
**领域**: 图学习  
**关键词**: 反事实解释, 动态图, 时序图神经网络, 蒙特卡洛树搜索, 可解释性

## 一句话总结
提出 CoDy——首个用于时序图神经网络（TGNN）的反事实解释方法，通过蒙特卡洛树搜索（MCTS）结合时空启发式策略高效探索可能的解释子图空间，在多个数据集上 AUFSC+ 提升 16%。

## 研究背景与动机

**领域现状**：时序图神经网络（TGNN）如 TGN、TGAT 在社交网络、电商等动态系统中广泛应用。但深度学习的黑盒特性使其决策过程不透明。

**现有痛点**：
   - 现有 GNN 可解释性方法（PGExplainer、GNNExplainer）针对静态图设计，不处理时间维度
   - 少数 TGNN 解释方法（T-GNNExplainer）仅提供事实性解释（哪些边贡献了预测）——无法回答"如果删除某些事件，预测会改变吗？"
   - 反事实解释在静态 GNN 上有研究，但动态图的搜索空间远大——需要考虑时间顺序、同时事件、事件传播等

**核心矛盾**：反事实解释更易理解（"如果学生 B 没有在 Math1 发帖，他就不会被预测在 Math2 发帖"），但动态图的组合爆炸使搜索不可行。

**本文目标**：为 TGNN 提供高效的反事实解释。

**切入角度**：MCTS 天然适合在大搜索空间中做启发式搜索——每次扩展搜索树时选择最有希望的"扰动方向"。

**核心 idea**：搜索树的每个节点对应一种输入图的扰动（删除/修改某些边），MCTS 策略利用三种启发式——空间近邻性（靠近目标的事件更可能相关）、时间近邻性（近期事件更可能相关）、局部事件影响（删除某事件对模型输出的即时影响）。

## 方法详解

### 整体框架
CoDy 的搜索流程：
1. 给定目标预测（如"学生B将在Math2发帖"）和输入时序图
2. MCTS 在"扰动空间"中搜索——每个状态是原始图的一个子集被删除的版本
3. 目标：找到最小的扰动集使预测翻转
4. 用时空启发式引导搜索方向

### 关键设计

1. **MCTS 搜索框架**:

    - 功能：在指数级的扰动空间中高效搜索反事实解释
    - 核心思路：搜索树的每个节点对应一种"删除了哪些边"的状态，扩展 = 额外删除一条边
    - UCB 选择 + 回传：通过 UCB 公式平衡已知好方向的利用和新方向的探索
    - 终止条件：找到使 TGNN 预测翻转的最小扰动
    - 设计动机：暴力搜索 $2^{|E|}$ 种边子集不可行→MCTS 通常只需搜索很小一部分

2. **时空启发式选择策略**:

    - 功能：引导 MCTS 优先扩展更有可能是反事实原因的边
    - 三种启发式：
        - **空间近邻性**：优先删除与目标节点/边在图结构上接近的事件——因为 TGNN 通过邻域聚合，近邻事件影响更大
        - **时间近邻性**：优先删除时间上接近目标预测的事件——近期事件对当前预测的影响通常大于远期事件
        - **局部事件影响**：预计算每条边被删除时对模型输出的即时变化——变化大的边更可能是反事实原因
    - 组合方式：$\text{priority}(e) = w_s \cdot \text{spatial}(e) + w_t \cdot \text{temporal}(e) + w_i \cdot \text{impact}(e)$
    - 设计动机：无启发式的 MCTS 在大图上收敛太慢——启发式将搜索集中在最有希望的区域

3. **GreeDy 基线**:

    - 功能：作为CoDy的贪心版本基线
    - 核心思路：每步贪心地删除对模型输出影响最大的边
    - 设计动机：由于之前没有 TGNN 反事实方法，需要设计合理的基线

### 损失函数 / 训练策略
- CoDy 是推理时方法（无训练）
- 模型无关——适用于任何 TGNN（TGN、TGAT 等）
- 搜索预算可调（控制速度-质量权衡）

## 实验关键数据

### 主实验
多个动态图数据集上的反事实解释质量：

| 方法 | AUFSC+ ↑ | 解释大小 ↓ | 搜索效率 |
|------|---------|---------|---------|
| PGExplainer (事实) | 0.42 | 12.3 | 快 |
| T-GNNExplainer (事实) | 0.48 | 10.8 | 中 |
| GreeDy (反事实) | 0.61 | 8.5 | 快 |
| **CoDy (反事实)** | **0.71** (+16%) | **5.2** | **中** |

### 不同 TGNN 模型

| TGNN | CoDy AUFSC+ | GreeDy AUFSC+ | 提升 |
|------|-----------|-------------|------|
| TGN | 0.73 | 0.62 | +17.7% |
| TGAT | 0.69 | 0.59 | +16.9% |

### 消融实验

| 配置 | AUFSC+ | 解释大小 | 说明 |
|------|--------|---------|------|
| 无启发式（纯 MCTS） | 0.58 | 7.8 | 搜索效率低 |
| 仅空间启发式 | 0.64 | 6.3 | 部分引导 |
| 仅时间启发式 | 0.62 | 6.8 | 部分引导 |
| 仅影响启发式 | 0.66 | 5.9 | 最有效的单一启发式 |
| **全部启发式** | **0.71** | **5.2** | 完整方法 |
| 搜索预算 ×0.5 | 0.65 | 5.8 | 减少搜索→略差 |
| 搜索预算 ×2 | 0.72 | 5.1 | 增加搜索→边际改进 |

### 关键发现
- CoDy 比最强基线（GreeDy）提升 16% AUFSC+——MCTS 的探索-利用平衡优于贪心
- 反事实解释比事实解释提供更简洁的解释（5.2 vs 10.8 边）——因为反事实关注"最小必要"
- 三个启发式互补——局部事件影响最有效，但空间和时间信息进一步改进
- 模型无关性使 CoDy 适用于不同 TGNN 架构——TGN 和 TGAT 上均有效
- 搜索预算的边际收益递减——默认预算已足够找到高质量解释

## 亮点与洞察
- **反事实解释 vs 事实解释**的区别对动态图特别有意义——事实解释告诉你"哪些事件导致了预测"，反事实解释告诉你"改变哪些事件可以改变预测"→后者对干预和决策更有用
- MCTS 是反事实搜索的天然选择——搜索空间大但有结构（删除子集的层次关系对应搜索树）
- 时空启发式利用了动态图的固有特性——不是通用搜索优化而是利用了领域知识
- 社交网络的例子非常直观——"如果学生 B 没有在 Math1 发帖"比一堆因子解释容易理解得多
- 首个 TGNN 反事实方法——开辟了新研究方向

## 局限与展望
- MCTS 搜索仍有计算开销——大图上可能慢
- 启发式权重 $w_s, w_t, w_i$ 需要调优
- 仅处理边删除——边添加或节点特征修改的反事实未考虑
- 反事实的因果性解释可能不完全准确——相关性 ≠ 因果性
- 评估指标（AUFSC+）的设计合理性需要社区共识

## 相关工作与启发
- **vs T-GNNExplainer**: 事实解释→哪些边重要；CoDy 反事实解释→改变哪些边改变预测
- **vs CF²**: 静态 GNN 的反事实方法，不处理时间维度
- **vs GreeDy**: 贪心搜索→容易陷入局部最优；CoDy MCTS 探索更充分
- **启发**：MCTS + 领域启发式的组合可推广到其他需要搜索最小干预的可解释性问题

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个 TGNN 反事实解释方法
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多 TGNN、充分消融
- 写作质量: ⭐⭐⭐⭐⭐ 社交网络例子极其直观
- 价值: ⭐⭐⭐⭐ 开辟了动态图可解释性的新方向

<!-- RELATED:START -->

## 相关论文

- [Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs](../../NeurIPS2025/graph_learning/dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)
- [Positional Encoding meets Persistent Homology on Graphs](positional_encoding_meets_persistent_homology_on_graphs.md)
- [Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees](towards_graph_foundation_models_learning_generalities_across_graphs_via_task-tre.md)
- [Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](../../ICLR2026/graph_learning/entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)
- [Extending Complex Logical Queries on Uncertain Knowledge Graphs](../../ACL2025/graph_learning/extending_complex_logical_queries_uncertain_knowledge_graphs.md)

<!-- RELATED:END -->
