---
title: >-
  [论文解读] Learning Interestingness in Automated Mathematical Theory Formation
description: >-
  [NeurIPS 2025][强化学习][自动理论形成] 提出 Fermat——一个将数学理论形成建模为 MDP 的强化学习环境，以及 EvoAbstract——一个带抽象学习的 LLM 驱动进化算法，用于自动合成数学对象的"兴趣度"度量函数，在初等数论和有限域上显著超越硬编码基线。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "自动理论形成"
  - "兴趣度学习"
  - "进化程序合成"
  - "LLM 驱动搜索"
---

# Learning Interestingness in Automated Mathematical Theory Formation

**会议**: NeurIPS 2025  
**arXiv**: [2511.14778](https://arxiv.org/abs/2511.14778)  
**代码**: [https://github.com/trishullab/Fermat](https://github.com/trishullab/Fermat)  
**领域**: 强化学习 / 自动数学发现  
**关键词**: 自动理论形成, 兴趣度学习, 强化学习, 进化程序合成, LLM 驱动搜索

## 一句话总结

提出 Fermat——一个将数学理论形成建模为 MDP 的强化学习环境，以及 EvoAbstract——一个带抽象学习的 LLM 驱动进化算法，用于自动合成数学对象的"兴趣度"度量函数，在初等数论和有限域上显著超越硬编码基线。

## 研究背景与动机

自动化数学发现是 AI 的长期梦想——从 1950 年代的 Logic Theorist 到近年的 AlphaProof。然而，现有工作主要聚焦于**解决预定义问题**（定理证明），而数学研究的本质是**开放式探索**过程：定义新概念、研究性质、提出猜想、证明或反驳。

两个核心挑战：

**缺乏完整的理论形成框架**：现有系统要么只做定理证明，要么只做猜想生成，没有一个统一框架支持定义新概念、提出猜想、证明定理的完整过程

**搜索引导问题**：数学探索的搜索空间组合爆炸，大部分路径导向平凡或无趣的数学。人类数学家靠直觉判断"什么值得研究"——即**兴趣度（interestingness）**，但之前的系统（如 HR）依赖硬编码的兴趣度度量

本文的切入：将兴趣度度量的发现建模为**学习问题**——用进化程序合成自动搜索最优的兴趣度函数，引导智能体发现有意义的数学理论。

## 方法详解

### 整体框架

**Fermat 环境**：将数学理论形成建模为 MDP $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R})$：
- **状态 $\mathcal{S}$**：知识图谱 $G=(V,E)$，节点为定义、猜想、定理，边为构造依赖
- **动作 $\mathcal{A}$**：定义产生动作（Exists, Specialize, Compose 等 9 种）、猜想产生动作（Implication, Equivalence 等 4 种）、证明动作（prove/disprove）
- **外在奖励 $\mathcal{R}_\mathcal{E}$**：发现预定义的真实数学实体集 $\mathcal{E}$ 中的实体得 1 分

**EvoAbstract**：学习最优兴趣度函数 $\mathcal{I}^* = \arg\max_\mathcal{I} \mathbb{E}_{\tau \sim \pi_\mathcal{I}}[\sum_t \gamma^t \mathcal{R}_\mathcal{E}]$

### 关键设计

1. **数学实体表示**：每个实体 $m$ 包含三个组件：

    - 符号定义 $m_{sym}$（Fermat DSL 中的形式化表达）
    - 计算解释 $m_{comp}$（可执行 Python 函数）
    - 缓存实例 $\mathcal{X}(m) = (\mathcal{X}^+(m), \mathcal{X}^-(m))$（正例和反例集合）

2. **EvoAbstract 进化搜索**：

    - **EvolutionStep**：使用 LLM $\mathcal{L}_{var}$（GPT-4o-mini）作为变异算子，输入高分父程序和模板，生成新的兴趣度函数候选
    - **Island 模型**：$k=4$ 个并行种群维护多样性
    - **PolicyEvaluationStep**：通过 episodic rollouts 在 Fermat 中评估候选程序（64 episodes, 60 秒超时），累积外在奖励作为适应度

3. **抽象学习（EvoAbstract 的核心创新）**：

    - **AbstractionStep**：每 8 轮进行一次，LLM $\mathcal{L}_{abs}$ 分析高分程序集，识别可复用的子程序（抽象），加入抽象库 $\text{Lib}_i$
    - 后续进化中，$\mathcal{L}_{var}$ 可使用抽象库中的组件构建更复杂的解决方案
    - 效果：促进模块化、组合式构建，引导搜索走向更有前景的程序空间区域
    - 例子：EvoAbstract 自动发现了 `compute_example_balance`（类似 applicability）、`calculate_rule_diversity_score`（规则多样性）等抽象

4. **产生规则**：

    - 定义产生：Exists（存在量化）、Specialize（特化变量）、Compose（组合）、ForAll（全称量化）、Match（等值匹配）、Negate（取反）、Size（计数）等
    - 猜想产生：Implication（蕴含）、Equivalence（等价）、Nonexistence（不存在性）、Exclusivity（排他性）
    - 后端证明器：Z3 SMT Solver

### 损失函数 / 训练策略

兴趣度函数作为内在奖励 $\mathcal{R}_\mathcal{I}(S,a,S') = \mathcal{I}(m_{new}, S')$，策略 $\pi_\mathcal{I}$ 根据兴趣度分数引导动作选择。优化目标是最大化累积外在奖励。每个配置运行 4 次取平均。

## 实验关键数据

### 主实验（累积外在奖励，越高越好）

| 方法 | succ_zero_eq | arithmetic_base | ff_27 |
|------|------|----------|------|
| Random | 4.68 | 4.44 | 2.33 |
| Comprehensibility (HR最佳) | 8.23 | 8.55 | 5.38 |
| Equal Weight (HR组合) | 6.57 | 5.93 | 3.93 |
| GPT-4o (one-shot) | 5.26 | 6.46 | 2.36 |
| FunSearch | 10.23 | 22.41 | **11.34** |
| **EvoAbstract** | 9.62 | **23.98** | 9.82 |

EvoAbstract 和 FunSearch 在所有起始配置上显著超越所有基线，其中 EvoAbstract 在 arithmetic_base 上最优（23.98 vs 22.41 发现实体）。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| FunSearch (无抽象) | 22.41 (arith) | 进化搜索已有效 |
| EvoAbstract (有抽象) | **23.98** (arith) | 抽象学习带来额外增益 |
| GPT-4o (one-shot, best) | 9.45 (arith) | 单次采样效果有限 |
| HR Applicability | 5.71 (arith) | 单一手工度量不足 |

### 关键发现
- **GPT-4o 直接生成的兴趣度函数效果很差**：甚至不如手工的 comprehensibility 度量——因为 LLM 倾向于奖励构造深度和连接度，导致关注初始但无关的实体
- **进化搜索极其有效**：FunSearch/EvoAbstract 发现的度量远优于所有静态基线
- **抽象学习有价值但有两面性**：在 arithmetic_base 上带来增益，但在 ff_27 和 succ_zero_eq 上可能产生"抽象锁定"效应，限制后期探索多样性
- **可重发现基础数学概念**：从 succ_zero_eq 出发发现加法、乘法、整除性；从 arithmetic_base 出发发现幂运算和素数概念
- EvoAbstract 生成的程序更模块化、可读性更好（使用多个有意义的抽象函数）

## 亮点与洞察
- **数学理论形成作为 RL 问题**的形式化非常优雅，将开放式数学发现转化为可量化评估的框架
- **兴趣度学习**抓住了自动数学发现的核心难题——不仅要能证明定理，更要知道什么值得证明
- EvoAbstract 的抽象学习机制类似于人类数学中提取"通用概念"的过程——从具体成功经验中抽象出可复用的模式
- 产生规则 + 符号证明器的设计使得所有发现都有严格保证

## 局限与展望
- 策略模板限制了动作空间暴露，限制了可发现的复杂数学对象
- "瓶颈实体"问题：关键概念（如素数）被发现时知识图谱已过大，阻碍后续有价值的操作
- 缺乏实体等价性检查导致表示冗余
- Z3 证明器在理论增长后变得计算不可行
- 未集成交互式定理证明器（如 Lean），限制了可探索的数学领域

## 相关工作与启发
- **vs HR [Colton 2000]**：HR 使用硬编码兴趣度度量，Fermat + EvoAbstract 自动学习度量
- **vs FunSearch [2024]**：EvoAbstract 在 FunSearch 基础上增加抽象学习，程序更模块化
- **vs AlphaProof**：AlphaProof 专注于竞赛级定理证明，Fermat 则关注理论的开放式探索
- **vs Minimo [2024]**：Minimo 限于初始公理定义上的猜想-证明博弈，Fermat 支持生成新定义

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将数学理论形成形式化为 RL，兴趣度学习的思路新颖独特
- 实验充分度: ⭐⭐⭐⭐ 多个起始配置、丰富基线对比、定性分析，但规模受限于 Z3 性能
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，但产生规则的技术细节较重
- 价值: ⭐⭐⭐⭐⭐ 为自动数学发现开辟了新方向，Fermat 框架有广泛的后续研究价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning](discover_automated_curricula_for_sparse-reward_reinforcement_learning.md)
- [\[NeurIPS 2025\] A Theory of Multi-Agent Generative Flow Networks](a_theory_of_multi-agent_generative_flow_networks.md)
- [\[ACL 2026\] Adaptive Instruction Composition for Automated LLM Red-Teaming](../../ACL2026/reinforcement_learning/adaptive_instruction_composition_for_automated_llm_red-teaming.md)
- [\[ICLR 2026\] ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning](../../ICLR2026/reinforcement_learning/arm-fm_automated_reward_machines_via_foundation_models_for_compositional_reinfor.md)
- [\[ICLR 2026\] Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](../../ICLR2026/reinforcement_learning/unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)

</div>

<!-- RELATED:END -->
