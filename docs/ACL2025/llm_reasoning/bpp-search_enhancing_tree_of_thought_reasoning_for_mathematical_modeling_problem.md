---
title: >-
  [论文解读] BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving
description: >-
  [ACL 2025][LLM推理][Tree-of-Thought] 提出 BPP-Search 算法，将 Beam Search、过程奖励模型 (PRM) 和 Pairwise Preference 机制整合到 Tree-of-Thought 框架中，用于运筹学数学建模问题的自动求解，在 StructuredOR 等数据集上以更少的推理步骤显著超越 CoT/SC/ToT 基线。
tags:
  - ACL 2025
  - LLM推理
  - Tree-of-Thought
  - 数学建模
  - 过程奖励模型
  - Beam Search
  - 运筹学
---

# BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving

**会议**: ACL 2025  
**arXiv**: [2411.17404](https://arxiv.org/abs/2411.17404)  
**代码**: https://github.com/LLM4OR/StructuredOR  
**领域**: LLM推理  
**关键词**: Tree-of-Thought, 数学建模, 过程奖励模型, Beam Search, 运筹学

## 一句话总结
提出 BPP-Search 算法，将 Beam Search、过程奖励模型 (PRM) 和 Pairwise Preference 机制整合到 Tree-of-Thought 框架中，用于运筹学数学建模问题的自动求解，在 StructuredOR 等数据集上以更少的推理步骤显著超越 CoT/SC/ToT 基线。

## 研究背景与动机
**领域现状**：LLM 在数学推理方面能力逐步增强，将自然语言问题自动转化为线性规划 (LP)/混合整数规划 (MIP) 模型是一个新兴方向。现有方法主要使用 CoT、SC 或 ToT 框架来引导模型进行多步推理。

**现有痛点**：现有开源运筹学数据集（如 NL4OPT、MAMO-ComplexLP）只标注了目标函数值，缺少建模过程（变量定义、约束设置等）的详细标注，这限制了基于过程监督的强化学习方法的应用。同时，ToT 框架虽然能生成多条推理路径，但缺乏有效的叶节点选择策略。

**核心矛盾**：CoT 只生成单条路径，容易出错；SC 无法验证中间步骤正确性；ToT 生成大量叶节点但不知道选哪个作为最终答案。PRM 可以给中间步骤打分来指导搜索，但 PRM 在回归任务上打分不够精确——结构高度相似但正确性不同的候选方案会得到接近的分数。

**本文目标** (a) 构建带有完整建模过程标注的运筹学数据集；(b) 在 ToT 框架中高效搜索并可靠选出最优建模方案。

**切入角度**：观察到 PRM 在最后一层 Beam Search 中难以区分高度相似的正确/错误候选，于是引入 Pairwise Preference 模型来做成对比较排序。

**核心 idea**：用 Beam Search + PRM 做树搜索剪枝加速，再用 Pairwise Preference 模型在最终候选中做成对排序选出最优答案。

## 方法详解

### 整体框架
输入是自然语言描述的运筹学问题，输出是完整的数学建模方案（集合→参数→变量→目标函数→约束）。整个推理过程构建为一棵 4 层的 Tree-of-Thought：第 1 层为问题，第 2 层为集合+参数，第 3 层为变量，第 4 层为目标函数+约束。每层最多 3 个子节点。搜索分两阶段：前面几层用 Beam Search + PRM 剪枝保留 top-k 候选；最后一层用 Pairwise Preference 模型做成对比较选出最优方案。

### 关键设计

1. **StructuredOR 数据集**:

    - 功能：提供带完整建模过程标注的运筹学数据集（124 个问题，覆盖物流、调度、网络等领域）
    - 核心思路：先用 GPT-4o 生成抽象模型的参数分布，再通过模拟生成实例数据，用 Gurobi 求解器验证可解性，最后用 GPT-4o 生成自然语言描述并改写 3 个语义等价版本
    - 设计动机：现有数据集只有目标值标注，无法用于训练 PRM 等过程监督模型。StructuredOR 提供了从集合、参数、变量到约束的全链路标注

2. **过程奖励模型 (PRM)**:

    - 功能：对 ToT 中每一层的中间推理结果打分，指导搜索方向
    - 核心思路：基于 Qwen2.5-Math-1.5B 做全参数微调的二分类任务。提取正确标签对应的 logit 值，经 sigmoid 得到分数 $S_{\text{PRM}} = \frac{1}{1+e^{-l_{prm}}}$
    - 设计动机：PRM 能评估中间步骤的正确性（不仅仅是最终结果），让小模型+验证器可以超越大模型直接推理的效果

3. **Pairwise Preference 模型**:

    - 功能：在最终候选队列中做成对比较，选出最优答案
    - 核心思路：同样基于 Qwen2.5-Math-1.5B 微调。对任意两个候选 (A, B)，计算偏好分数 $S_{PM}(A \succ B) = \frac{1}{1+e^{-l_{pm}}}$。对每个候选 A，其综合分数为与所有其他候选的成对偏好分数均值：$S_{PM}(A) = \frac{1}{n-1}\sum_{j \neq i} S(A \succ X_j)$
    - 设计动机：PRM 对结构高度相似的候选打分差异很小，无法可靠区分。成对比较能更好地捕捉细微差异，因为模型直接对比两个方案而非独立评分

4. **Random Greedy 搜索**:

    - 功能：作为 PRM 打分不精确时的替代搜索策略
    - 核心思路：保留与最高分差距在 threshold 之内的候选 $P(a_{\max}) - P(a_i) \leq \text{threshold}$，从中随机选择继续搜索
    - 设计动机：PRM 提供的是粗略的偏好排序而非精确分数，引入随机性可以缓解打分误差的影响

### 损失函数 / 训练策略
- PRM 训练：二分类交叉熵损失，标签来自 CoT/ToT 生成的正确/错误推理路径的手工标注
- Preference Model 训练：二分类交叉熵，训练数据为 (正确路径, 错误路径) 的成对组合，标签指示谁排在前面
- 两个模型都基于 Qwen2.5-Math-1.5B 全参数微调

## 实验关键数据

### 主实验

| 数据集 | 方法 | 正确率 | 推理步数 |
|--------|------|--------|---------|
| StructuredOR | CoT | 0.633 | 1 |
| StructuredOR | SC | 0.700 | 4 |
| StructuredOR | ToT-Rethink | 0.766 | 40 |
| StructuredOR | **BPP-Search** | **0.933** | **15** |
| Mamo-ComplexLP | CoT | 0.486 | 1 |
| Mamo-ComplexLP | **BPP-Search** | **0.722** | **21** |
| NL4OPT | CoT | 0.566 | 1 |
| NL4OPT | ToT-Rethink | 0.622 | 40 |
| NL4OPT | **BPP-Search** | **0.804** | **15** |

### 消融实验

| 配置 | StructuredOR | Mamo-ComplexLP | NL4OPT | 步数 |
|------|-------------|---------------|--------|------|
| Greedy + PRM | 0.733 | 0.555 | 0.699 | 9 |
| Random Greedy + PRM | 0.833 | 0.513 | 0.692 | 9 |
| Beam Search (W=2) + PRM | 0.800 | 0.652 | 0.783 | 15 |
| Beam Search (W=3) + PRM | 0.766 | 0.666 | 0.755 | 21 |
| BPP-Search (W=2) | **0.933** | 0.652 | **0.804** | 15 |
| BPP-Search (W=3) | 0.866 | **0.722** | 0.797 | 21 |

### 关键发现
- **Beam Search 宽度增大反而可能降低准确率**（如 StructuredOR 上 W=3 比 W=2 低 7%），说明 PRM 打分不精确导致更多候选引入更多噪声
- **Pairwise Preference 模型有效缓解了这个问题**：BPP-Search (W=2) 在 StructuredOR 上达到 93.3%，比纯 Beam Search + PRM 高 13.3%
- **BPP-Search 用 15 步达到的准确率超过 ToT-Fully-Traverse 的 40 步**，搜索效率大幅提升
- GPT-4o 在 ToT 框架下表现最优；小模型（如 Llama-3.2-11B）几乎无法完成此类数学建模任务

## 亮点与洞察
- **PRM + Pairwise Preference 的两阶段评估**很巧妙：PRM 做粗筛维持搜索效率，Preference 做精排处理最终候选的细微差异。这个"粗筛+精排"的范式可以迁移到其他需要从多候选中选优的推理任务
- **用手工标注代替 MCTS 生成 PRM 训练数据**：MCTS 需要大量 rollout 且存在 Reward Hacking 风险，手工标注虽然成本高但数据质量更可靠
- **Random Greedy 作为 PRM 不精确时的简单 baseline** 很实用——仅保留接近最高分的候选并随机选择，在某些情况下比标准 Greedy 更好

## 局限与展望
- **数据集规模偏小**：StructuredOR 只有 124 个问题，限制了结论的泛化性
- **树的宽度和深度受限于计算成本**：当前每层最多 3 个子节点、4 层深度，更大的树可能带来更好效果但代价过高
- **对 Policy Model 能力要求高**：小模型（如 Llama-3.2-11B）几乎无法在 ToT 框架下生成有效建模方案，方法的适用范围有限
- **PRM 和 Preference Model 都很小（1.5B）**：更大的验证器模型可能进一步提升性能
- 可以考虑将 Pairwise Preference 扩展到树的中间层而不仅仅是最后一层

## 相关工作与启发
- **vs CoT/SC**: CoT 单路径容易出错，SC 多路径但无中间验证。BPP-Search 通过树结构+过程奖励实现多路径探索+中间步骤评估
- **vs MCTS-based PRM**: MCTS 方法（如 Math-Shepherd）需要大量 rollout 生成训练数据，BPP-Search 使用手工标注的确定性数据，避免了 Reward Hacking
- **vs ToT-Rethink**: ToT-Rethink 将所有叶节点交给 LLM 重新评估，BPP-Search 用专门训练的小模型做评估，更可靠且计算成本更低

## 评分
- 新颖性: ⭐⭐⭐⭐ PRM + Pairwise Preference 的组合在 ToT 框架中较新，但各组件本身不算新颖
- 实验充分度: ⭐⭐⭐⭐ 三个数据集 + 多种搜索变体消融，但数据集偏小
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细
- 价值: ⭐⭐⭐ 主要面向运筹学数学建模这一较窄领域，但"粗筛+精排"的搜索范式有一定通用价值

<!-- RELATED:START -->

## 相关论文

- [Enhancing Mathematical Reasoning in LLMs by Stepwise Correction](enhancing_mathematical_reasoning_in_llms_by_stepwise_correction.md)
- [Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](safe_math_reasoning.md)
- [MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](mm-verify_enhancing_multimodal_reasoning_with_chain-of-thought_verification.md)
- [FineReason: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving](finereason_evaluating_and_improving_llms_deliberate_reasoning_through_reflective.md)
- [RPM-MCTS: Knowledge-Retrieval as Process Reward Model with Monte Carlo Tree Search for Code Generation](../../AAAI2026/llm_reasoning/rpm-mcts_knowledge-retrieval_as_process_reward_model_with_monte_carlo_tree_searc.md)

<!-- RELATED:END -->
