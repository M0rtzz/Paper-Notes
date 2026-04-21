---
title: >-
  [论文解读] Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving
description: >-
  [ICLR 2026][进化算法] 提出 HELIX 框架，将强化学习（GRPO）与进化算法（NSGA-II）结合用于开放式科学问题求解：RL 迭代优化策略，进化机制平衡解的质量与多样性，in-context learning 利用历史解指导探索，仅用 14B 模型在圆填充、机器学习任务等 20 个任务中超越 GPT-4o 流水线。
tags:
  - ICLR 2026
  - 进化算法
  - GRPO
  - 科学优化
  - NSGA-II
  - 上下文学习
---

# Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving

**会议**: ICLR 2026  
**arXiv**: [2603.07642](https://arxiv.org/abs/2603.07642)  
**代码**: 无（论文未提供）  
**领域**: 强化学习 / 科学发现  
**关键词**: 进化算法, GRPO, 科学优化, NSGA-II, 上下文学习

## 一句话总结

提出 HELIX 框架，将强化学习（GRPO）与进化算法（NSGA-II）结合用于开放式科学问题求解：RL 迭代优化策略，进化机制平衡解的质量与多样性，in-context learning 利用历史解指导探索，仅用 14B 模型在圆填充、机器学习任务等 20 个任务中超越 GPT-4o 流水线。

## 研究背景与动机

**领域现状**：用 LLM 解决复杂科学问题（符号回归、分子生成、数学优化）是热门方向。后训练方法（SFT/RLVR）在推理任务上有效，但面对开放式科学问题容易 entropy collapse 难以发现真正新颖的解。工作流方法（如 AlphaEvolve）把 LLM 嵌入进化流水线，但高度依赖任务特定设计。

**现有痛点**：(a) 纯 RL 方法无记忆——对同一问题的采样上下文固定，无法利用历史发现的好解；(b) 进化方法用的是固定预训练模型做变异，不更新模型参数，探索能力受限于预训练知识；(c) 两类方法都缺乏探索与利用的良好平衡。

**核心矛盾**：开放式科学问题三个特性——领域特异、解空间无界、无全局最优保证——要求同时具备：从经验中学习、平衡质量与多样性、站在巨人肩上探索。

**本文目标**：设计一个通用框架，让 LLM 能在无标准答案的科学优化问题中，通过 RL + 进化的协同迭代持续发现更优解。

**切入角度**：将"解"表示为代码，用 LLM 作变异/改进算子；用 RL（GRPO）更新策略参数使模型越来越会改进解；用 NSGA-II 在奖励-多样性 Pareto 前沿上筛选种群。

**核心 idea**：RL 教模型"如何改进解"，进化保证"探索多样方向"，in-context learning 让模型"站在已知好解的肩上"。

## 方法详解

### 整体框架

HELIX 三模块协同迭代：
- **RL 模块**：用 GRPO 更新 LLM 策略参数 $\theta$，使模型学会生成高奖励的代码修改
- **进化模块**：用 NSGA-II 从所有历史解中选择高奖励+高多样性的种群 $\mathcal{P}_t$
- **ICL 模块**：构建包含当前解+祖先解（lineage tree）的 prompt，让模型参考历史经验
- **迭代流程**：选种群 → 构造 prompt → LLM 生成修改 → 评估奖励 → 更新策略+种群 → 下一轮

### 关键设计

1. **GRPO 策略优化**:

    - 功能：基于奖励信号更新 LLM 参数，使其越来越擅长改进代码解
    - 核心思路：给定 prompt $q$ 和当前解 $s_t$，生成 G 个 rollout $\{a_j\}$，用 GRPO 的标准 clipped surrogate objective 训练，advantage 通过组内奖励归一化计算 $\hat{A}_{j,k} = \frac{R(s_t,a_j) - \text{mean}\{R\}}{\text{std}\{R\}}$
    - 设计动机：RL 让模型的变异能力不断提升，而非停留在预训练知识——这是与 AlphaEvolve 等纯工作流方法的根本区别

2. **NSGA-II 多目标种群选择**:

    - 功能：在奖励-多样性两个目标上选择 Pareto 最优种群
    - 核心思路：对每个解计算奖励 $R(s)$ 和多样性 $\text{Div}(s) = 1 - \frac{1}{k}\sum_{j \in \mathcal{N}_k(i)} \cos(E(s_i), E(s_j))$（用预训练embedding 的 KNN 余弦距离）。NSGA-II 做非支配排序 + 拥挤度筛选
    - 设计动机：防止 RL 的 entropy collapse——如果只按奖励选解，很快会收敛到局部最优。NSGA-II 保留多样的 Pareto 前沿使探索保持开放

3. **Lineage Tree In-context Learning**:

    - 功能：把当前解的"家族谱"（祖先解、它们的奖励和反馈）放入 prompt
    - 核心思路：$q = \text{ConstructPrompt}(\{p\} \cup \{s_t, R(s_t), F(s_t)\} \cup \{f^{(k)}(s_t), R(f^{(k)}(s_t)), F(f^{(k)}(s_t))\}_{1 \leq k < n})$，其中 $f^{(k)}$ 是第 k 代祖先
    - 设计动机："站在巨人肩上"——让模型看到这个解是如何一步步从 v0 改进到 v_n 的，理解改进方向

### 损失函数 / 训练策略

- GRPO objective with clipping $\epsilon$ and KL penalty $\beta$
- 多样性度量：用预训练 embedding 模型（而非原始代码文本）计算 KNN 余弦多样性
- 迭代训练：每轮生成新解 → 评估 → 更新种群 → 更新策略参数

## 实验关键数据

### 主实验

20 个任务 5 个类别的最佳结果对比：

| 任务类别 | 任务 | Task-Specific | GPT-4o+OpenEvolve | **HELIX (14B)** |
|----------|------|---------------|-------------------|-----------------|
| ML | Adult Income (F1↑) | 80.72 | 72.27 | **82.07** |
| ML | Bank Marketing (F1↑) | 76.32 | 78.54 | **80.65** |
| ML | Boston Housing (RMSE↓) | 3.258 | 2.937 | **1.747** |
| 圆填充 | Sum of Radii ↑ | - | - | **2.63598** |

HELIX 用 14B 模型在 ML 任务上超越 GPT-4o 流水线，F1 平均提升 5.95 分。

### 消融实验

| 配置 | 平均奖励 | 说明 |
|------|----------|------|
| Full HELIX | 最高 | RL + Evolution + ICL |
| w/o RL (只用进化) | 中等 | 模型不更新参数，变异能力固定 |
| w/o Evolution (只用RL) | 低 | entropy collapse，解多样性丧失 |
| w/o ICL (无历史) | 中等 | 无法利用祖先经验 |
| w/o Diversity in selection | 中低 | 只按奖励选解，快速收敛到局部最优 |

### 关键发现

- **RL 和进化缺一不可**：纯 RL 会 entropy collapse，纯进化用固定模型变异能力上限低。两者协同才能持续发现更优解
- 多样性度量用语义 embedding 比用代码文本更好——因为功能相同但代码风格不同的解应被视为"不多样"
- Lineage tree 的深度（祖先数量）对性能有显著影响——过短则缺乏上下文，过长则 prompt 太长
- 14B 模型 + HELIX 在多个任务上超越 GPT-4o + 精心设计的流水线，说明更新模型参数（RL）比增大模型更有效

## 亮点与洞察

- **RL + 进化的融合范式**：RL 负责"越来越会改进"，进化负责"不要只改进一个方向"。这种二元体系比单一的 RL 或单一的进化算法都更适合无界开放问题
- **Lineage tree 作为 ICL 上下文**：不是随机给几个好解做示例，而是展示一个解的完整进化历史——让模型理解"什么样的修改方向是有效的"
- **通用框架**：同一个 HELIX 框架处理 ML 任务、物理仿真、圆填充、函数最小化、符号回归等完全不同的问题类别，泛化性很强

## 局限与展望

- **代码评估可能不安全**：在物理仿真等任务中需要执行生成的代码来获取奖励，存在安全风险
- **仍需任务特定的评估函数**：虽然框架通用，但每个任务仍需定义准确的奖励函数 $R(s)$
- **训练计算量**：每轮迭代需要生成 + 评估 + RL 更新，在大模型上的训练开销不小
- **32B 模型才能处理物理任务**：对几何推理能力要求高的任务需要更大模型，14B 不够

## 相关工作与启发

- **vs AlphaEvolve/OpenEvolve**: 这些方法用固定预训练模型做变异，不更新模型参数。HELIX 加入 RL 使模型越来越会变异，是更本质的进步
- **vs 标准 RLVR (GRPO/DAPO)**: RLVR 不维护解的种群，每次独立采样。HELIX 通过进化种群保持多样性和历史记忆
- **对 AI for Science 的启发**：这种"RL 教能力 + 进化保多样 + ICL 给上下文"的三位一体思路可推广到蛋白质设计、催化剂发现、芯片设计等科学优化场景

## 评分

- 新颖性: ⭐⭐⭐⭐ RL+进化的结合思路有创意且执行到位，但各组件（GRPO、NSGA-II、ICL）都不是新的
- 实验充分度: ⭐⭐⭐⭐⭐ 5 类 20 个任务覆盖面广，消融详细
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，但公式较多
- 价值: ⭐⭐⭐⭐⭐ 为 AI 解决开放式科学问题提供了强大的通用框架

<!-- RELATED:START -->

## 相关论文

- [From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation](from_verifiable_dot_to_reward_chain_harnessing_verifiable_reference-based_reward.md)
- [$\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving](textbfre2_unlocking_llm_reasoning_via_reinforcement_learning_with_re-solving.md)
- [Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)
- [Solving Football by Exploiting Equilibrium Structure of 2p0s Differential Games with One-Sided Information](solving_football_by_exploiting_equilibrium_structure_of_2p0s_differential_games_.md)
- [ReMoT: Reinforcement Learning with Motion Contrast Triplets](remot_reinforcement_learning_with_motion_contrast_triplets.md)

<!-- RELATED:END -->
