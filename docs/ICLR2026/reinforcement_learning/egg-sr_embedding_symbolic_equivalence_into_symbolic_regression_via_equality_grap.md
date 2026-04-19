---
title: >-
  [论文解读] EGG-SR: Embedding Symbolic Equivalence into Symbolic Regression via Equality Graph
description: >-
  [ICLR 2026][符号回归] 提出 Egg-SR 统一框架，通过等价图（e-graph）将符号等价性嵌入 MCTS、DRL 和 LLM 三类符号回归方法中，分别实现子树剪枝、梯度方差降低和反馈提示增强。理论证明 Egg-MCTS 收紧遗憾界、Egg-DRL 降低梯度估计方差，实验验证一致提升表达式发现精度。
tags:
  - ICLR 2026
  - 符号回归
  - 符号等价
  - 等价图
  - 蒙特卡洛树搜索
  - 深度强化学习
---

# EGG-SR: Embedding Symbolic Equivalence into Symbolic Regression via Equality Graph

**会议**: ICLR 2026  
**arXiv**: [2511.05849](https://arxiv.org/abs/2511.05849)  
**代码**: [项目页面](https://nan-jiang-group.github.io/egg-sr)  
**领域**: 强化学习  
**关键词**: 符号回归, 符号等价, 等价图, 蒙特卡洛树搜索, 深度强化学习

## 一句话总结

提出 Egg-SR 统一框架，通过等价图（e-graph）将符号等价性嵌入 MCTS、DRL 和 LLM 三类符号回归方法中，分别实现子树剪枝、梯度方差降低和反馈提示增强。理论证明 Egg-MCTS 收紧遗憾界、Egg-DRL 降低梯度估计方差，实验验证一致提升表达式发现精度。

## 研究背景与动机

符号回归（Symbolic Regression）旨在从实验数据中发现闭合形式的物理规律，是 AI 驱动科学发现的重要任务。搜索空间随数据维度呈指数增长，计算挑战巨大。

一个有前景但未充分探索的方向是**符号等价性**：许多语法不同的表达式定义相同函数。例如：
- $\log(x_1^2 x_2^3)$
- $\log(x_1^2) + \log(x_2^3)$
- $2\log(x_1) + 3\log(x_2)$

这三者在数学上完全等价，但现有算法将它们视为不同输出，导致：

**冗余探索**：MCTS 会独立搜索等价子树，浪费计算

**训练缓慢**：DRL 无法利用等价序列共享奖励信号，梯度方差大

**反馈信息不足**：LLM 仅看到单一表达式形式，错失等价变体的信息

核心挑战：如何统一且可扩展地表示符号等价表达式并嵌入现代学习框架？显式枚举等价变体的数量随表达式长度指数增长，不可行。

## 方法详解

### 整体框架

Egg-SR 框架包含一个核心模块 **Egg**（Equality graph for Grammar-based expressions）和三个集成接口：

1. **Egg-MCTS**：共享等价路径的搜索统计量，剪枝冗余子树
2. **Egg-DRL**：聚合等价序列的概率，降低策略梯度方差
3. **Egg-LLM**：用等价表达式丰富反馈提示，引导下轮预测

### 关键设计一：Egg 模块——基于文法的等价图

**表达式表示**：使用上下文无关文法 $\langle V, \Sigma, R, S \rangle$ 表示符号表达式，其中 $V=\{A\}$ 为非终结符，$\Sigma$ 为终结符（变量+常数），$R$ 为产生式规则。表达式通过依次对最左非终结符应用规则构建。

**e-graph 数据结构**：
- 由等价类（e-class）组成，每个 e-class 包含一组等价的 e-node
- 每个 e-node 编码一个数学运算并引用子 e-class
- 共享公共子表达式，实现紧凑表示

**构建过程（等价饱和）**：
1. 用输入表达式的产生式序列初始化 e-graph
2. 对每条重写规则（如 $\log(ab) \leadsto \log(a) + \log(b)$），匹配 LHS 模式
3. 为匹配位置创建 RHS 对应的新 e-class 和 e-node
4. 合并等价 e-class
5. 迭代直到饱和或达到最大次数

**提取策略**：(1) 基于代价的提取——选择最简化的表达式；(2) 随机游走采样——生成一批等价表达式。

### 关键设计二：Egg-MCTS——等价感知的反向传播

标准 MCTS 包含选择（UCT）→ 扩展 → 模拟 → 反向传播四步。Egg-MCTS 的改进集中在**反向传播**步骤：

- 将当前路径（部分完成的表达式）转化为 e-graph 并饱和
- 从饱和 e-graph 中采样多个等价序列
- 检查搜索树中是否存在对应路径
- **同时更新所有等价路径**的奖励和访问计数

这改变了 UCT 公式的语义：$\mathtt{visits}(s)$ 不再计算特定节点的访问次数，而是计算**等价类中任意代表**的访问次数，类似于换位表（transposition table）。

**示例**：路径 1 对应 $\log(x_1 \times A)$，路径 2 对应 $\log(x_1) + \log(A)$，两者在 $\log(ab) \leadsto \log(a)+\log(b)$ 下等价。标准 MCTS 会独立探索两个子树，Egg-MCTS 共享统计量避免冗余。

### 关键设计三：Egg-DRL——等价感知的策略梯度

标准 DRL 使用序列解码器采样产生式序列 $\tau$，策略梯度估计器：

$$g(\theta) \approx \frac{1}{N}\sum_{i=1}^N (\mathtt{reward}(\tau_i) - b) \nabla_\theta \log p_\theta(\tau_i)$$

Egg-DRL 的改进：对每个采样序列 $\tau_i$ 构建 e-graph，采样 $K-1$ 个等价序列，使用等价感知估计器：

$$g_\mathtt{egg}(\theta) \approx \frac{1}{N}\sum_{i=1}^N (\mathtt{reward}(\tau_i) - b') \nabla_\theta \log\left[\sum_{k=1}^K p_\theta(\tau_i^{(k)})\right]$$

关键改变在于用等价序列的概率之和 $\sum_k p_\theta(\tau_i^{(k)})$ 替代单一序列概率。由于等价序列共享相同奖励，对同奖励序列取平均降低了组内变异性。

### 关键设计四：Egg-LLM——等价增强的反馈提示

LLM 符号回归的迭代过程：假设生成 → 数据评估 → 经验管理。Egg-LLM 在反馈环节将生成的 Python 表达式解析为符号表达式，转化为 e-graph 提取等价变体，将等价表达式汇总入反馈提示，使 LLM 观察到更丰富的函数等价形式。

### 理论保证

**定理 3.1（Egg-MCTS 遗憾界）**：令 $\kappa$ 为标准 MCTS 的近最优分支因子，$\kappa_\infty$ 为 Egg-MCTS 的分支因子，则 $\kappa_\infty \leq \kappa$，遗憾界从 $\tilde{\mathcal{O}}(n^{-\frac{\log(1/\gamma)}{\log\kappa}})$ 收紧为 $\tilde{\mathcal{O}}(n^{-\frac{\log(1/\gamma)}{\log\kappa_\infty}})$。

**定理 3.2（Egg-DRL 方差降低）**：Egg-DRL 的梯度估计器无偏且方差严格不大于标准 DRL：$\mathbb{V}\mathrm{ar}[g_\mathtt{egg}(\theta)] \leq \mathbb{V}\mathrm{ar}[g(\theta)]$。

## 实验关键数据

### 主实验：三角函数数据集 NMSE

| 方法 | (2,1,1) 无噪声 | (3,2,2) 无噪声 | (4,4,6) 无噪声 | (2,1,1) 有噪声 | (3,2,2) 有噪声 |
|------|--------------|--------------|--------------|--------------|--------------|
| MCTS | 0.006 | 0.033 | 0.144 | 0.015 | 0.007 |
| **Egg-MCTS** | **<1E-6** | **<1E-6** | **0.006** | **0.005** | **0.012** |
| DRL | 0.030 | 0.277 | 2.990 | 0.09 | 0.44 |
| **Egg-DRL** | **0.020** | **0.161** | **2.381** | **0.07** | **0.35** |

Egg-MCTS 在无噪声简单情况下将 NMSE 从 0.006/0.033 降至 <1E-6，提升数个数量级。

### 主实验：科学基准 LLM 对比

| 方法 | Oscillation I IID | Oscillation II OOD | Bacterial OOD | Stress-Strain OOD |
|------|-----------------|-------------------|-------------|-----------------|
| LLM-SR (GPT3.5) | <1E-6 | 3.81E-5 | 0.0264 | 0.0516 |
| **Egg-LLM (GPT3.5)** | <1E-6 | **<1E-6** | **0.0198** | **0.0419** |
| LLM-SR (Mistral) | <1E-6 | 0.0291 | 0.0037 | 0.0946 |
| **Egg-LLM (Mistral)** | **<1E-6** | **0.0114** | **0.0107** | **0.0754** |

### 效率分析

**空间效率**：对 $\log(x_1 \times \cdots \times x_n)$ 有 $2^{n-1}$ 个等价变体，数组存储指数增长，e-graph 通过子表达式共享实现显著压缩。

**时间效率**：Egg 模块引入的时间开销可忽略不计，相比 DRL 中系数拟合和参数更新的耗时，Egg 的 e-graph 构建和等价序列采样占比极小。

### 关键发现

- Egg 在三类算法（MCTS/DRL/LLM）中一致提升性能，验证了框架的统一性
- Egg-MCTS 维持更广更深的搜索树，探索更大更多样的搜索空间
- Egg-DRL 的梯度估计方差在训练过程中显著降低（图 3 右）
- 三角函数表达式因含丰富的等价变体（三角恒等式），受益最大

## 亮点与洞察

1. **统一框架**：一个 Egg 模块同时服务 MCTS、DRL、LLM 三类方法，接口一致
2. **理论+实验双重验证**：定理 3.1 和 3.2 分别证明了 MCTS 遗憾界收紧和 DRL 方差降低，实验完全印证
3. e-graph 的核心优势在于**共享子表达式**实现指数压缩，变量数 n=8 时内存优势超过 100 倍
4. 与 Egg 位操作无关的等价关系（如系数等价 in SymNet）指出了有趣的开放问题

## 局限与展望

1. 重写规则集需人工定义，目前覆盖对数、三角等恒等式，更复杂的数学恒等式有待扩展
2. 对不含丰富等价变体的表达式（如纯多项式），Egg 收益有限
3. 当前仅支持基于文法的表达式表示，基于 SymNet 的层状表示需要扩展
4. Egg 在推理阶段（如 LLM）如何最优利用仍是开放问题
5. e-graph 饱和过程的终止条件（最大迭代次数）需要调优

## 相关工作与启发

- e-graph 源自程序综合和编译优化领域，本文首次统一嵌入到符号回归的学习算法中
- 与 GP 中 e-graph 用于去重和简化不同，Egg-SR 利用等价变体进行等价感知学习
- MCTS 中的等价路径共享类似博弈中的换位表，但需处理符号等价的模式匹配
- DRL 中概率聚合的思路可推广到任何存在输出等价性的序列生成任务
- 知识引导科学发现是 SR 的重要方向，Egg-SR 与 AI-Feynman 的对称性约束、单位约束等正交

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (统一框架 + 理论保证，首次系统性嵌入符号等价)
- 实验充分度: ⭐⭐⭐⭐ (MCTS/DRL/LLM 三类方法覆盖，含效率分析)
- 写作质量: ⭐⭐⭐⭐⭐ (图示精美，理论-实验衔接紧密)
- 价值: ⭐⭐⭐⭐ (对符号回归领域有实质推进，可推广到其他等价性场景)

<!-- RELATED:START -->

## 相关论文

- [GraphOmni: A Comprehensive and Extensible Benchmark Framework for Large Language Models on Graph-theoretic Tasks](graphomni_a_comprehensive_and_extensible_benchmark_framework_for_large_language_.md)
- [Embedding Safety into RL: A New Take on Trust Region Methods](../../ICML2025/reinforcement_learning/embedding_safety_into_rl_a_new_take_on_trust_region_methods.md)
- [Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization](../../ICML2025/reinforcement_learning/graph-supported_dynamic_algorithm_configuration_for_multi-objective_combinatoria.md)
- [TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction](../../AAAI2026/reinforcement_learning/tadarag_task_adaptive_retrieval-augmented_generation_via_on-the-fly_knowledge_gr.md)
- [Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning](../../ICML2025/reinforcement_learning/graph-assisted_stitching_for_offline_hierarchical_reinforcement_learning.md)

<!-- RELATED:END -->
