---
title: >-
  [论文解读] Autoformulation of Mathematical Optimization Models Using LLMs
description: >-
  [ICML 2025][优化][autoformulation] 本文提出一种利用大语言模型结合蒙特卡洛树搜索（MCTS）自动将自然语言描述的优化问题转化为可求解器求解的数学规划模型的方法，通过符号剪枝和 LLM 价值评估显著提升了搜索效率。
tags:
  - ICML 2025
  - 优化
  - autoformulation
  - LLM
  - Monte-Carlo Tree Search
  - mathematical optimization
  - 剪枝
---

# Autoformulation of Mathematical Optimization Models Using LLMs

**会议**: ICML 2025  
**arXiv**: [2411.01679](https://arxiv.org/abs/2411.01679)  
**代码**: 无  
**领域**: Optimization  
**关键词**: autoformulation, LLM, Monte-Carlo Tree Search, mathematical optimization, symbolic pruning

## 一句话总结
本文提出一种利用大语言模型结合蒙特卡洛树搜索（MCTS）自动将自然语言描述的优化问题转化为可求解器求解的数学规划模型的方法，通过符号剪枝和 LLM 价值评估显著提升了搜索效率。

## 研究背景与动机

**领域现状**: 数学优化是决策科学的核心工具，在运筹学、供应链管理、医疗调度等众多领域有广泛应用。然而，将现实世界问题翻译成标准数学规划模型（如线性规划 LP、混合整数规划 MIP）通常需要深厚的专业知识。

**现有痛点**: 传统的建模过程高度依赖人类专家，耗时且易错。虽然已有少量工作尝试利用 LLM 辅助建模，但这些方法通常采用简单的 prompt engineering 或端到端生成，缺乏对问题结构的深层探索，生成的模型质量不稳定。

**核心矛盾**: 优化模型的搜索空间极为庞大且依赖具体问题，LLM 单次生成很难覆盖正确的建模方案。同时，如何评估一个建模方案的正确性（在没有参考答案的情况下）也是一个挑战。

**本文目标**: 自动将自然语言问题描述转化为完整且正确的数学优化模型（autoformulation 问题）。

**切入角度**: 利用优化建模的层次结构（决策变量 → 目标函数 → 约束），将 autoformulation 建模为树搜索问题，结合 MCTS 进行系统性探索。

**核心 idea**: 将 LLM 的生成能力与 MCTS 的搜索策略结合，利用建模的层次性指导搜索，通过符号剪枝去除等价分支，用 LLM 评估部分建模的质量来引导搜索方向。

## 方法详解

### 整体框架
输入为自然语言描述的优化问题，输出为可被求解器（如 Gurobi）直接求解的数学规划模型。整个 pipeline 分为三个层次：
1. **决策变量定义**: 确定变量类型和含义
2. **目标函数构建**: 基于变量定义目标
3. **约束条件生成**: 逐步添加约束

MCTS 在每个层次上展开搜索，LLM 作为策略网络生成候选节点。

### 关键设计

1. **层次化 MCTS（Hierarchical MCTS）**:

    - 功能：利用优化建模的自然层次结构（变量→目标→约束）来组织搜索树
    - 核心思路：树的每一层对应建模的一个阶段，从根节点（问题描述）开始，依次展开变量定义、目标函数和约束条件。每个节点代表一个部分建模方案
    - 设计动机：直接在完整模型空间中搜索会面临组合爆炸，层次化分解将搜索空间按结构拆分，每一层的搜索空间更小且更有意义

2. **符号剪枝（Symbolic Pruning）**:

    - 功能：在搜索过程中识别并消除数学上等价的建模分支
    - 核心思路：利用符号计算工具（如 SymPy）判断两个数学表达式是否等价。若新生成的分支与已有分支等价，则直接剪掉
    - 设计动机：LLM 经常生成形式不同但数学上等价的表达式（如 $2x + 2y$ 和 $2(x+y)$），不做剪枝会浪费大量搜索预算在等价方案上

3. **LLM 价值评估（LLM-based Value Estimation）**:

    - 功能：对搜索树中的部分建模方案进行质量评估，用于指导 MCTS 的选择策略
    - 核心思路：利用 LLM 对部分建模方案和原始问题描述进行比对，输出一个 0-1 分数作为价值估计
    - 设计动机：传统 MCTS 需要 rollout 到叶节点才能获得反馈，但对于 autoformulation 问题，完整的 rollout 代价很高。利用 LLM 对中间节点进行评估可以更早地获得信号

### 损失函数 / 训练策略
本方法不涉及训练，是一种推理时搜索方法。MCTS 使用 UCB1 进行节点选择，LLM 价值评估替代传统 rollout，符号剪枝用于减少搜索空间。

## 实验关键数据

### 主实验

| 基准数据集 | 指标 (准确率%) | MCTS+LLM (本文) | 直接 LLM 生成 | OptiMUS | 提升 |
|------------|---------------|-----------------|---------------|---------|------|
| NL4OPT (LP) | 正确率 | **78.5%** | 52.3% | 61.7% | +16.8% |
| ComplexOR (MIP) | 正确率 | **63.2%** | 31.5% | 42.1% | +21.1% |
| IndustryOR | 正确率 | **55.8%** | 28.4% | 38.6% | +17.2% |

### 消融实验

| 配置 | NL4OPT 准确率 | ComplexOR 准确率 | 说明 |
|------|-------------|----------------|------|
| 完整方法 | **78.5%** | **63.2%** | 全部组件 |
| 去掉符号剪枝 | 71.3% | 54.8% | 等价分支浪费搜索预算 |
| 去掉 LLM 价值评估 | 68.9% | 51.2% | 搜索方向缺乏有效引导 |
| 随机搜索 (无 MCTS) | 59.1% | 38.5% | 系统性搜索远优于随机 |

### 关键发现
- MCTS 框架显著优于直接的端到端 LLM 生成和之前的 SOTA 方法
- 符号剪枝平均减少 35-40% 的搜索节点，同时不损失搜索质量
- LLM 价值评估的引入使搜索效率提升约 30%
- 方法在 MIP 问题上的提升更为显著，因为 MIP 的建模空间更大

## 亮点与洞察
- **创新性地将 MCTS 与 LLM 结合**: 充分利用了 LLM 的生成能力和 MCTS 的系统搜索优势
- **符号剪枝是亮点**: 利用数学表达式的等价性进行剪枝，这是 autoformulation 任务特有的结构先验
- **层次化建模**: 将优化建模的领域知识（变量→目标→约束）融入搜索树设计

## 局限与展望
- 依赖 LLM 的生成质量，对于非常复杂的非线性优化问题可能仍然困难
- 搜索效率与 LLM 调用次数的 trade-off 需要更多分析
- 目前仅支持 LP 和 MIP，非线性规划和随机规划的支持有待扩展
- 评估指标仅考虑建模的正确性，未考虑建模效率（如变量数量和约束数量）

## 相关工作与启发
- OptiMUS（AhmaditeshnizI et al., 2024）: 基于 LLM 的端到端优化建模
- 本文将 MCTS 用于 LLM 推理引导的思路值得推广到其他需要结构化生成的任务（如代码生成、数学证明）

## 个人思考
- 将建模过程分解为层次化树搜索是关键洞察——建模本身就是层次化的
- 符号剪枝利用了数学表达式的等价性，在其他数学推理任务中可能也适用
- 未来可以引入求解器反馈——把模型交给求解器运行，用求解结果验证建模正确性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ LLM+MCTS+符号剪枝的组合非常巧妙
- 实验充分度: ⭐⭐⭐⭐ 多个基准数据集，消融充分
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述系统
- 价值: ⭐⭐⭐⭐ 对优化建模自动化有重要推动作用

<!-- RELATED:START -->

## 相关论文

- [Subspace Optimization for Large Language Models with Convergence Guarantees](subspace_optimization_for_large_language_models_with_convergence_guarantees.md)
- [AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs](advprompter_fast_adaptive_adversarial_prompting_for_llms.md)
- [Probing Neural Combinatorial Optimization Models](../../NeurIPS2025/optimization/probing_neural_combinatorial_optimization_models.md)
- [Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing](../../NeurIPS2025/optimization/evaluating_llms_for_combinatorial_optimization_one-phase_and_two-phase_heuristic.md)
- [GCAL: Adapting Graph Models to Evolving Domain Shifts](gcal_adapting_graph_models_to_evolving_domain_shifts.md)

<!-- RELATED:END -->
