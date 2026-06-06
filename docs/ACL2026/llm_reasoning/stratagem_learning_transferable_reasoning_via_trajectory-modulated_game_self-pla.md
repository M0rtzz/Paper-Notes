---
title: >-
  [论文解读] Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play
description: >-
  [ACL2026][LLM推理][self-play] Stratagem 在文本游戏自博弈中不再只按输赢强化模型，而是用“抽象可迁移性”和“推理演化”两个轨迹级信号调制 advantage，使从游戏中学到的策略更能迁移到数学、通用推理和代码生成任务。
tags:
  - "ACL2026"
  - "LLM推理"
  - "self-play"
  - "迁移推理"
  - "轨迹优势调制"
  - "数学推理"
  - "代码生成"
---

# Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play

**会议**: ACL2026  
**arXiv**: [2604.17696](https://arxiv.org/abs/2604.17696)  
**代码**: https://github.com/ydyyyy/Stratagem  
**领域**: 代码智能 / LLM 推理 / 自博弈强化学习  
**关键词**: self-play、迁移推理、轨迹优势调制、数学推理、代码生成

## 一句话总结
Stratagem 在文本游戏自博弈中不再只按输赢强化模型，而是用“抽象可迁移性”和“推理演化”两个轨迹级信号调制 advantage，使从游戏中学到的策略更能迁移到数学、通用推理和代码生成任务。

## 研究背景与动机
**领域现状**：用游戏训练智能体是强化学习的经典路线，近期也被迁移到 LLM 训练中。SPIRAL 这类方法让语言模型在零和文本游戏中自博弈，通过终局胜负信号更新策略，希望游戏中的规划、概率判断和决策能力能迁移到数学与代码任务。

**现有痛点**：终局输赢只能告诉模型“这局赢了还是输了”，却无法区分赢局中哪些推理是可迁移的抽象策略，哪些只是游戏特定技巧。例如“国王压过皇后”这类规则记忆能赢牌局，但不一定帮助解数学题。

**核心矛盾**：自博弈产生的轨迹很丰富，但传统 advantage 只看 game outcome。模型可能强化了 domain-specific heuristic，而没有优先强化可迁移的枚举、反证、条件分解和概率推理。

**本文目标**：作者希望在不依赖人工推理数据的前提下，从游戏轨迹中筛出真正能迁移到下游任务的推理模式，并让训练信号偏向这些轨迹。

**切入角度**：论文把迁移失败归纳为两个原因：domain specificity 让模型停留在游戏语义，contextual stasis 让模型缺少随上下文推进而深化推理的能力。

**核心 idea**：在 SPIRAL 的 game advantage 上加入轨迹级调制，抽象程度高的轨迹保留更大 advantage，推理在多轮中持续深化的轨迹再获得额外奖励。

## 方法详解
Stratagem 的方法很克制：它没有重新设计游戏环境，也没有把数学题混入训练，而是修改自博弈轨迹的训练权重。给定一条游戏轨迹，原始 SPIRAL 根据终局胜负和角色 baseline 计算 game advantage；Stratagem 再用两个信号对它调制，得到用于策略梯度更新的 modified advantage。

### 整体框架
训练环境包含三个文本零和游戏：Tic-Tac-Toe、Kuhn Poker 和 Simple Negotiation。它们分别覆盖空间规划、概率推理和策略优化。两个玩家共享同一个 LLM policy，通过角色条件区分当前行动方。

每条 trajectory 包含多轮状态、模型回复、推理文本和动作。SPIRAL 只依据终局胜负计算 role-conditioned advantage；Stratagem 在这个基础上计算 Reasoning Transferability Coefficient 与 Reasoning Evolution Reward，并用公式 `A_mod = A_game * phi + beta * psi` 更新模型。

实验使用 Qwen3-4B-Base 作为基础模型，`phi` 和 `psi` 由 GPT-4 作为评估器打分，`beta` 默认设为 0.2。作者通过轨迹子采样控制评估成本，单次训练约需 2 张 A100 的 30 GPU-hours，GPT-4 评分成本约 100 美元。

### 关键设计
1. **Reasoning Transferability Coefficient**:

    - 功能：衡量一条游戏轨迹中的推理是否抽象、领域无关、可迁移。
    - 核心思路：`phi` 取 0、0.5 或 1。游戏规则词和固定套路占主导时，轨迹权重被压低；出现枚举情形、概率估计、目标约束分解等抽象推理时，轨迹保留更高 advantage。
    - 设计动机：输赢奖励会强化一切有助于赢游戏的行为，而迁移训练需要优先强化“为什么能赢”的推理结构。

2. **Reasoning Evolution Reward**:

    - 功能：奖励推理随对局进展而更新、深化和自洽。
    - 核心思路：`psi` 取 -1、0 或 +1，评价轨迹中的推理是否根据新状态调整策略、是否保持多步一致性、是否从浅层反应发展为更完整的计划。
    - 设计动机：数学和代码任务的上下文会在中间结果出现后不断变化，静态游戏规则训练如果不奖励“推理演化”，容易学到固定模板。

3. **轨迹优势调制训练**:

    - 功能：把上述两个轨迹质量信号转化为策略梯度中的训练权重。
    - 核心思路：`A_game * phi` 是乘法门控，让低迁移轨迹即使赢了也少影响参数；`beta * psi` 是加性奖励，让持续演化的推理轨迹在胜负之外获得额外信号。
    - 设计动机：乘法项解决“学错技巧”，加法项解决“只学静态反应”，两者共同把自博弈从赢游戏转向学推理。

### 损失函数 / 训练策略
训练仍然是 self-play policy gradient。不同之处是用于更新的 advantage 从 `A_game` 换成 `A_mod`。每一轮采样游戏轨迹后，系统计算玩家终局收益、角色 baseline、`phi` 与 `psi`，然后对轨迹中对应玩家的生成回复求对数概率梯度。

作者没有把 downstream benchmark 作为训练奖励，因此数学、通用推理和代码提升可以被解释为跨域迁移，而不是在目标任务上直接优化。这个设定让论文的核心论点更清晰：游戏训练能否迁移，取决于轨迹中被强化的推理类型。

## 实验关键数据

### 主实验
评估覆盖数学推理、通用推理和代码生成。所有评测采用 zero-shot prompting，代码任务使用 HumanEval pass@1。

| 模型 | MATH500 | AIME24 | AIME25 | AMC-23 | GPQA | MMLU-Pro | HumanEval |
|------|---------|--------|--------|--------|------|----------|-----------|
| Qwen3-4B-Base | 65.80 | 10.00 | 3.30 | 50.00 | 30.60 | 47.20 | 67.93 |
| SPIRAL | 71.00 | 10.00 | 6.70 | 45.00 | 36.41 | 53.93 | 77.44 |
| Stratagem | 76.00 | 20.00 | 13.30 | 60.00 | 38.23 | 57.83 | 77.93 |
| Stratagem vs Base | +10.20 | +10.00 | +10.00 | +10.00 | +7.63 | +10.63 | +10.00 |
| Stratagem vs SPIRAL | +5.00 | +10.00 | +6.60 | +15.00 | +1.82 | +3.90 | +0.49 |

最强提升集中在竞赛数学：AIME24 从 10.00 提到 20.00，AIME25 从 3.30 提到 13.30，AMC-23 从 50.00 提到 60.00。这与论文假设一致，复杂多步推理比知识回忆更依赖可迁移的策略结构。

### 消融实验

| 配置 | MATH500 | AIME24 | AIME25 | OlympiadBench | AMC-23 | GPQA | MMLU-Pro | HumanEval |
|------|---------|--------|--------|---------------|--------|------|----------|-----------|
| Stratagem full | 76.00 | 20.00 | 13.30 | 39.90 | 60.00 | 38.23 | 57.83 | 77.93 |
| w/o 推理演化奖励 | 74.60 | 13.30 | 10.00 | 39.30 | 52.50 | 37.22 | 56.92 | 77.80 |
| full - w/o | +1.40 | +6.70 | +3.30 | +0.60 | +7.50 | +1.01 | +0.91 | +0.13 |

| 人类评估维度 | Qwen3-4B-Base | SPIRAL | Stratagem w/o evolution | Stratagem |
|----------------|--------------|--------|--------------------------|-----------|
| 推理抽象性 | 2.48 | 3.24 | 3.82 | 4.06 |
| 推理推进性 | 2.32 | 3.08 | 3.36 | 4.18 |

### 关键发现
- `psi` 对 AIME24 和 AMC-23 的贡献最明显，说明“推理是否随着局面推进”对竞赛数学类任务尤其重要。
- 参数敏感性显示 `beta = 0.20` 是整体较优点；`beta` 太小接近去掉演化奖励，太大会让演化评分压过原始游戏目标，训练不稳定。
- 人类评估显示去掉 `psi` 后抽象性仍较高，但推进性明显下降，支持 `phi` 和 `psi` 分别对应两个不同能力维度。

## 亮点与洞察
- 论文抓住了 self-play 迁移的核心难点：不是游戏能不能产生数据，而是哪些游戏轨迹值得学。这个问题比单纯扩展游戏数量更基础。
- `phi` 和 `psi` 的组合很直观。一个筛掉领域特定技巧，一个奖励动态推理过程，刚好对应“会抽象”和“会推进”两种可迁移能力。
- 结果中代码生成提升不如数学显著，但 HumanEval 仍高于 SPIRAL，说明游戏里的结构化计划和约束满足可能对程序合成有弱迁移。
- 这篇论文提供了一个更一般的训练范式：当环境奖励只覆盖结果时，可以用轨迹级 meta-evaluator 给过程质量加权，而不必人工标注每一步。

## 局限与展望
- 轨迹质量评分依赖 GPT-4，这会带来成本、偏好和可复现性问题；如果评估器偏好某种推理风格，训练出来的模型也会继承这种偏差。
- 训练环境只有三个文本游戏，覆盖的推理类型仍有限。更复杂的工具使用、代码调试或多智能体协作环境是否同样有效，还需要验证。
- `phi` 和 `psi` 是离散档位，表达力有限。真实轨迹可能同时包含可迁移策略和游戏特定技巧，粗粒度评分会损失细节。
- 方法没有直接证明模型内部学到的表示确实从游戏语义抽象到了数学语义，当前证据主要来自下游分数和人工轨迹评分。

## 相关工作与启发
- **vs SPIRAL**: SPIRAL 只使用终局胜负更新策略，Stratagem 在同样 self-play 框架中加入轨迹质量调制，因此能解释为什么某些胜利轨迹更值得学习。
- **vs Absolute Zero / self-play reasoning**: 这类工作强调无外部数据的自生成训练，Stratagem 更关注奖励塑形，特别是从环境结果奖励中分离出可迁移推理信号。
- **vs RLHF / RLAIF**: RLHF 通常评价回答偏好，Stratagem 评价整条交互轨迹的抽象性和演化性，粒度更贴近多步推理训练。
- **启发**: 代码智能领域可以把单元测试通过率视为 `A_game`，再引入“解法抽象性”和“调试过程推进性”作为轨迹调制信号，避免模型只学到测试集特定投机策略。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 用 trajectory modulation 解释并改进游戏到推理任务的迁移，想法清楚且有启发。
- 实验充分度: ⭐⭐⭐⭐☆ 覆盖数学、通用推理、代码和人工评估，但游戏环境和基础模型规模仍偏有限。
- 写作质量: ⭐⭐⭐⭐☆ 问题定义和公式简洁，实验附录给出完整数值；部分 `phi`、`psi` 评分细节还可以更透明。
- 价值: ⭐⭐⭐⭐☆ 对自博弈训练和代码推理 RL 很有参考意义，尤其适合结果奖励稀疏但轨迹丰富的任务。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain](../../ICML2026/llm_reasoning/self-play_only_evolves_when_self-synthetic_pipeline_ensures_learnable_informatio.md)
- [\[ACL 2026\] HISR: Hindsight Information Modulated Segmental Process Rewards for Multi-turn Agentic Reinforcement Learning](hisr_hindsight_information_modulated_segmental_process_rewards_for_multi-turn_ag.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](../../AAAI2026/llm_reasoning/llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)
- [\[ICML 2025\] Improving Rationality in the Reasoning Process of Language Models through Self-playing Game](../../ICML2025/llm_reasoning/improving_rationality_in_the_reasoning_process_of_language_models_through_self-p.md)
- [\[AAAI 2026\] SERL: Self-Examining Reinforcement Learning on Open-Domain](../../AAAI2026/llm_reasoning/serl_self-examining_reinforcement_learning_on_open-domain.md)

</div>

<!-- RELATED:END -->
