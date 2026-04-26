---
title: >-
  [论文解读] Wider or Deeper: Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search
description: >-
  [NeurIPS 2025][LLM/NLP][MCTS] AB-MCTS 提出了一种自适应分支的蒙特卡洛树搜索框架，在搜索树的每个节点上动态决定是"变宽"（生成新候选答案）还是"变深"（利用反馈优化现有答案），通过贝叶斯后验更新平衡探索与利用，在编程和工程任务上超越了重复采样和标准 MCTS。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - MCTS
  - 推理时计算
  - 自适应分支
  - Thompson采样
  - 代码生成
---

# Wider or Deeper: Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search

**会议**: NeurIPS 2025  
**arXiv**: [2503.04412](https://arxiv.org/abs/2503.04412)  
**代码**: [GitHub](https://github.com/SakanaAI/treequest)  
**领域**: LLM推理 / 推理时计算扩展  
**关键词**: MCTS, 推理时计算, 自适应分支, Thompson采样, 代码生成

## 一句话总结

AB-MCTS 提出了一种自适应分支的蒙特卡洛树搜索框架，在搜索树的每个节点上动态决定是"变宽"（生成新候选答案）还是"变深"（利用反馈优化现有答案），通过贝叶斯后验更新平衡探索与利用，在编程和工程任务上超越了重复采样和标准 MCTS。

## 研究背景与动机

推理时计算扩展（Inference-Time Scaling）是提升 LLM 在复杂任务上表现的重要方向。现有方法分三类：(1) 后训练微调（如 o1）、(2) 奖励引导 CoT、(3) 多答案生成。其中多答案生成（如重复采样 Best-of-N）简单有效，但缺乏利用外部反馈进行改进的机制。

**核心矛盾**：
- 重复采样专注于"探索"——独立生成大量候选，但不利用反馈进行"利用"
- 标准 MCTS（如 LATS）支持多轮改进，但使用固定分支因子作为超参数，限制了对 LLM 多样化输出空间的利用
- 重复采样的成功表明，有效的推理时扩展需要充分利用 LLM 的广阔输出空间，固定宽度阻碍了这一点

**切入角度**：将无界分支引入 MCTS，让搜索树在每个节点自适应地决定是扩展新分支还是沿现有分支深入，统一探索和利用两种策略。

## 方法详解

### 整体框架

AB-MCTS 构建搜索树 $T$，每个非根节点对应一个 LLM 生成的答案。每次迭代包含三步：(1) 选择要扩展的节点；(2) 生成新子节点；(3) 分数回传。核心创新在于引入 **GEN 节点**——每个节点都有一个 GEN 子节点，代表"生成新的子答案"这一动作。当 GEN 节点被选中时，从其父节点扩展新分支。

### 关键设计

1. **GEN 节点与自适应分支**:

    - 每个节点 $N$ 的可选动作集为 $A_N = \{a_0, a_1, \ldots, a_{n_{\text{child}}}\}$
    - $a_0$ 对应 GEN 节点（"变宽"），$a_1 \ldots a_{n_{\text{child}}}$ 对应已有子节点（"变深"）
    - 选择 GEN 节点 = 生成新候选答案；选择已有子节点 = 深入改进
    - 与标准 MCTS 的关键区别：已扩展的节点可以再次扩展，分支因子理论上无界

2. **AB-MCTS-M（Mixed Model 变体）**:

    - 为每个节点拟合混合贝叶斯模型：$r_{N_{\text{new}}, a_j} = \alpha_j + \sigma_y \epsilon_{N_{\text{new}}}$
    - $\alpha_j = \mu_\alpha + \sigma_\alpha \epsilon_j$ 为组级截距，捕获基础答案的质量
    - GEN 节点视为新引入的组，其参数通过共享后验 $\mu_\alpha, \sigma_\alpha$ 从其他组推断
    - 使用 MCMC 从后验分布采样，通过 Thompson 采样决定选择哪个动作
    - 分数回传：新节点分数添加到其所有祖先的历史中

3. **AB-MCTS-A（Node Aggregation 变体）**:

    - 引入 CONT 节点聚合所有子节点，代表"继续改进现有答案"
    - 每个节点有 GEN 节点和 CONT 节点两个选项
    - 使用指数族分布 + 共轭先验进行高效后验更新
    - 两个子变体：高斯模型（无界分数）和 Beta 模型（[0,1] 分数）
    - 比 AB-MCTS-M 更轻量，无共享参数

### Thompson 采样选择策略

不使用 UCT 分数是因为 GEN 节点使问题根本不同于标准多臂赌博机——标准 MCTS 的臂是静态的，而 AB-MCTS 中 GEN 节点会动态生成新臂。Thompson 采样基于后验分布的采样决策天然适合这个场景，还支持并行扩展。

## 实验关键数据

### 主实验（多基准 × 多模型）

| 方法 | LiveCodeBench (GPT-4o) | CodeContest (GPT-4o) | ARC-AGI (GPT-4o) | 平均排名 |
|------|----------------------|---------------------|-------------------|---------|
| Repeated Sampling | 37.8 | 37.9 | **15.0** | 3.5 |
| Sequential Refinement | 37.8 | 30.1 | 8.7 | 5.5 |
| Standard MCTS | 36.7 | 37.5 | 9.0 | 4.2 |
| **AB-MCTS-M** | 38.9 | **40.6** | 12.3 | **2.3** |
| AB-MCTS-A (Gaussian) | **39.1** | 40.2 | 13.0 | 2.7 |
| AB-MCTS-A (Beta) | 38.7 | 40.4 | 14.0 | 2.7 |

### MLE-Bench（机器学习竞赛任务）

| 方法 | Nomad2018 | Spooky | Pizza | 平均排名 |
|------|-----------|--------|-------|---------|
| Repeated Sampling | 0.065 | 0.47 | 0.72 | 3.0 |
| Standard MCTS | 0.076 | 0.45 | 0.60 | 3.3 |
| **AB-MCTS-M** | **0.060** | **0.38** | **0.72** | **1.3** |

### 关键发现
- AB-MCTS 在 LiveCodeBench 和 CodeContest 上一致优于基线，平均排名最优（2.3）
- 在 ARC-AGI 上重复采样仍然强劲（需要广泛探索），但 AB-MCTS 达到了与之可比的性能
- 扩大预算到 512 时（ARC-AGI），AB-MCTS 的改进曲线持续上升，而重复采样趋于平缓
- 搜索树分析显示 AB-MCTS 倾向于生成更宽的树（自适应扩展），同时也能在有潜力的分支上深入
- 不同任务受益于不同的探索-利用权衡，AB-MCTS 能自适应调整

## 亮点与洞察

- **核心贡献是将无界分支引入 MCTS**：这看似简单但技术上非平凡，需要全新的统计模型和选择策略
- **GEN 节点的设计很优雅**：将"是否扩展新分支"转化为与"选择哪个子节点"统一的决策问题
- **贝叶斯后验更新的使用**：相比基于计数的 UCT，贝叶斯方法在数据稀疏时表现更好，适合 LLM 场景
- **两种变体各有所长**：M 变体通过共享参数利用跨组信息，A 变体更轻量拥有解析更新
- **与后训练方法正交**：可以与 o1 等 CoT 微调方法无缝结合

## 局限与展望

- 依赖可靠的外部评分函数（如测试用例），不是所有任务都有
- MCMC 采样（M 变体）在节点多时可能成为计算瓶颈
- 在 ARC-AGI 上未能显著超越重复采样，说明对纯探索型任务优势有限
- 仅在编程和 ML 工程任务上验证，对数学推理、创意写作等任务的适用性未知
- 生成预算以 API 调用次数衡量，未考虑不同调用的实际耗时差异

## 相关工作与启发

- **vs 重复采样**: 重复采样纯探索，不利用反馈；AB-MCTS 自适应平衡探索与利用
- **vs LATS/标准 MCTS**: 固定分支因子限制了对 LLM 输出多样性的利用；AB-MCTS 支持无界分支
- **vs Sequential Refinement**: 纯利用无探索，AB-MCTS 更灵活
- **vs Progressive Widening**: PW 是经典 MCTS 扩宽技术，但基于访问计数；AB-MCTS 使用贝叶斯统计模型

## 评分

- 新颖性: ⭐⭐⭐⭐ 将无界分支引入 MCTS 的理论和实践贡献有价值，但核心思想（自适应宽度 vs 深度）较直观
- 实验充分度: ⭐⭐⭐⭐ 四个基准 × 两个前沿模型 × 多种预算设置 × 搜索树分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，两种变体的推导严谨，图表专业
- 价值: ⭐⭐⭐⭐ 对 LLM 推理时计算扩展有直接意义，特别适合编程任务场景

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute](../../ICML2025/llm_nlp/best-route_adaptive_llm_routing_with_test-time_optimal_compute.md)
- [\[ACL 2025\] Dynamic Parallel Tree Search for Efficient LLM Reasoning](../../ACL2025/llm_nlp/dynamic_parallel_tree_search_for_efficient_llm_reasoning.md)
- [\[ACL 2025\] Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning](../../ACL2025/llm_nlp/boosting_llms_molecular_structure_elucidation_with_knowledge_enhanced_tree_searc.md)
- [\[NeurIPS 2025\] Don't Be Lazy: CompleteP Enables Compute-Efficient Deep Transformers](dont_be_lazy_completep_enables_compute-efficient_deep_transformers.md)
- [\[NeurIPS 2025\] Valid Inference with Imperfect Synthetic Data](valid_inference_with_imperfect_synthetic_data.md)

<!-- RELATED:END -->
