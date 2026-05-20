---
title: >-
  [论文解读] Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play
description: >-
  [ACL2026][LLM推理][自博弈] STRATAGEM 在 SPIRAL 式游戏自博弈上加入“抽象可迁移性”和“跨轮推理演化”两个轨迹级奖励调制信号，让模型不只学会赢游戏，而是学到能迁移到数学、通用推理和代码生成的推理模式。
tags:
  - "ACL2026"
  - "LLM推理"
  - "自博弈"
  - "强化学习"
  - "可迁移推理"
  - "游戏训练"
  - "轨迹奖励"
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
- 价值: ⭐⭐⭐⭐☆ 对自博弈训练和代码推理 RL 很有参考意义，尤其适合结果奖励稀疏但轨迹丰富的任务。# STRATAGEM: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play

**会议**: ACL2026  
**arXiv**: [2604.17696](https://arxiv.org/abs/2604.17696)  
**代码**: https://github.com/ydyyyy/Stratagem  
**领域**: 代码智能 / LLM推理  
**关键词**: 自博弈, 强化学习, 可迁移推理, 游戏训练, 轨迹奖励  

## 一句话总结
STRATAGEM 在 SPIRAL 式游戏自博弈上加入“抽象可迁移性”和“跨轮推理演化”两个轨迹级奖励调制信号，让模型不只学会赢游戏，而是学到能迁移到数学、通用推理和代码生成的推理模式。

## 研究背景与动机
**领域现状**：用游戏训练智能体一直是强化学习的重要路线。对于 LLM，文本游戏也提供了一个有吸引力的训练环境：规则明确、奖励可验证、互动多轮，而且天然要求策略规划、概率推断和自适应决策。SPIRAL 这类方法已经证明，在零和文本游戏中做 self-play 可以提升语言模型的推理能力。

**现有痛点**：传统 self-play 主要依赖终局胜负奖励。胜负能告诉模型“这局赢了没有”，却不能区分胜利来自可迁移的抽象推理，还是来自具体游戏的经验性小技巧。例如，Kuhn Poker 中“King usually wins”可能能赢牌，但迁移不到数学；而“枚举所有情况、计算期望收益、选择最大值”才是能迁移的 reasoning pattern。

**核心矛盾**：游戏训练有稳定奖励，但奖励太粗；下游数学和代码需要的推理是抽象、结构化、随上下文逐步演化的，而游戏环境中的状态和规则往往相对静态，模型容易学到游戏语义绑定的启发式。

**本文目标**：作者把问题拆成两个障碍。第一是 domain specificity，即游戏中的推理模式被具体术语和规则绑定；第二是 contextual stasis，即游戏上下文相对固定，训练信号没有鼓励推理在多轮中逐步深化和适配。

**切入角度**：论文不改变 self-play 的主体框架，而是改造“轨迹优势”本身。终局胜负仍然保留，但每条 trajectory 的优势会根据其可迁移抽象程度被乘性放大或削弱，并根据推理是否随回合演化被加性奖励或惩罚。

**核心 idea**：把“赢了游戏”从唯一奖励变成基础奖励，再用 Reasoning Transferability Coefficient 和 Reasoning Evolution Reward 对 trajectory advantage 做调制，只强化那些既抽象又会随局势推进的推理轨迹。

## 方法详解
STRATAGEM 可以看成是对 SPIRAL 的一层奖励重写。SPIRAL 里，一条 game trajectory 的优势主要来自玩家在某个游戏和角色下的终局收益减去 EMA baseline。STRATAGEM 保留这个 game advantage，但对它做两种调制：一是用 $\varphi(\tau)$ 衡量这条轨迹里的推理是否足够抽象、结构化、原则化；二是用 $\psi(\tau)$ 衡量推理是否在多轮中逐步深化、根据新信息调整并保持逻辑连贯。

### 整体框架
训练时，系统从 TextArena 中采样一个零和文本游戏和一个玩家角色，让当前策略模型自博弈生成轨迹。轨迹结束后得到玩家收益 $R_p(\tau)$，并按照 SPIRAL 的方式计算 $A_{game}(\tau)=R_p(\tau)-b_{G,p}$。随后，评估器对整条轨迹打两个轨迹级分数：Reasoning Transferability Coefficient $\varphi(\tau)$ 和 Reasoning Evolution Reward $\psi(\tau)$。最终用于 policy gradient 的优势变成 $A_{mod}(\tau)=A_{game}(\tau)*\varphi(\tau)+\beta*\psi(\tau)$。训练数据来自三类游戏：Tic-Tac-Toe 负责空间策略，Kuhn Poker 负责概率推断，Simple Negotiation 负责策略优化。基础模型是 Qwen3-4B-Base，评估器使用 GPT-4，论文设置 $\beta=0.2$。

### 关键设计
1. **Reasoning Transferability Coefficient**:

    - 功能：判断一条游戏轨迹里的推理是否能迁移到数学、代码和一般决策任务。
    - 核心思路：$\varphi$ 由三个离散维度加权得到：Abstraction Level、Structural Clarity、Principle Orientation。每个维度取 0、0.5 或 1，权重分别是 0.35、0.35、0.30。高分轨迹通常使用“expected value”“case enumeration”“if-then chain”这类领域无关概念；低分轨迹则依赖“King beats Queen”“中心格很重要”这样的游戏专有经验。由于 $\varphi$ 是乘在 $A_{game}$ 上的，胜利但不抽象的轨迹会被削弱，胜利且抽象的轨迹保留完整优势。
    - 设计动机：单纯胜负奖励会把所有赢法混在一起。乘性调制的好处是，不否认游戏胜负的重要性，但要求胜利必须有“可迁移理由”才值得大幅强化。

2. **Reasoning Evolution Reward**:

    - 功能：奖励多轮推理从浅到深、随新状态调整并保持因果连贯的轨迹。
    - 核心思路：$\psi$ 由 Reasoning Deepening、Strategy Adaptation、Logical Coherence 三个维度组成，取值为 -1、0、+1，权重分别是 0.35、0.25、0.40。与 $\varphi$ 不同，$\psi$ 是加到优势上的项，形式是 $\beta*\psi$。如果轨迹一开始认真分析、后面退化成“照常行动”，或者每轮都像第一次行动一样重置上下文，就会被惩罚；如果后续推理能承接前面的观察并调整策略，就会获得额外奖励。
    - 设计动机：数学题和代码题的上下文不是静态的，前一步分解会改变后一步要处理的问题。用零中心的 $\psi$ 可以惩罚推理退化，并减少 policy gradient 中正偏置奖励带来的方差。

3. **轨迹优势调制训练流程**:

    - 功能：在不推翻 self-play 主循环的前提下，把可迁移推理质量注入 RL 更新。
    - 核心思路：每次迭代包括四步：采样游戏和角色，生成 self-play trajectory；计算终局收益和 SPIRAL 风格的 role-conditioned advantage；用 GPT-4 评估 $\varphi$ 和 $\psi$；用调制后的 $A_{mod}$ 更新策略。为了控制成本，只抽样一部分轨迹做完整 LLM 评价，其余轨迹分配 batch mean。论文报告 GPT-4 评分每次训练约增加 100 美元，而完整训练约需 2 张 A100 上的 30 GPU-hours。
    - 设计动机：如果把 $\varphi$ 和 $\psi$ 设计成 token-level reward，成本和噪声都会很高。轨迹级奖励虽然粗一些，但正好对应“推理模式是否可迁移”和“推理是否演化”这两个宏观属性。

### 损失函数 / 训练策略
STRATAGEM 使用 policy gradient 更新，核心变化是把原始 game advantage 换成调制后的 $A_{mod}$。评估分数由 prompt-based evaluator 给出，$\varphi$ 的三维提示关注抽象程度、结构清晰度和原则导向，$\psi$ 的提示关注推理深化、策略适配和逻辑连贯。训练后在数学、通用推理和代码生成任务上零样本评测，代码生成使用 HumanEval pass@1。

## 实验关键数据

### 主实验
完整结果显示，STRATAGEM 在 9 个下游 benchmark 中 8 个取得最好结果，尤其对竞赛级数学最明显。它不仅超过 Qwen3-4B-Base，也普遍超过 SPIRAL。

| 模型 | MATH500 | AIME24 | AIME25 | OlympiadBench | AMC-23 | Minerva | GPQA | MMLU-Pro | HumanEval |
|------|---------|--------|--------|---------------|--------|---------|------|----------|-----------|
| Qwen3-4B-Base | 65.80 | 10.00 | 3.30 | 33.30 | 50.00 | 24.30 | 30.60 | 47.20 | 67.93 |
| SPIRAL | 71.00 | 10.00 | 6.70 | 34.70 | 45.00 | 42.30 | 36.41 | 53.93 | 77.44 |
| STRATAGEM | 76.00 | 20.00 | 13.30 | 39.90 | 60.00 | 41.50 | 38.23 | 57.83 | 77.93 |
| 相对 Base 提升 | +10.20 | +10.00 | +10.00 | +6.60 | +10.00 | +17.20 | +7.63 | +10.63 | +10.00 |
| 相对 SPIRAL 提升 | +5.00 | +10.00 | +6.60 | +5.20 | +15.00 | -0.80 | +1.82 | +3.90 | +0.49 |

### 消融实验
去掉 $\psi$ 后，模型仍然有抽象推理奖励，但缺少“多轮推理演化”的训练信号。结果说明 $\psi$ 对需要长链步骤的数学题尤其关键。

| 配置 | MATH500 | AIME24 | AIME25 | OlympiadBench | AMC-23 | Minerva | GPQA | MMLU-Pro | HumanEval |
|------|---------|--------|--------|---------------|--------|---------|------|----------|-----------|
| STRATAGEM Full | 76.00 | 20.00 | 13.30 | 39.90 | 60.00 | 41.50 | 38.23 | 57.83 | 77.93 |
| STRATAGEM w/o $\psi$ | 74.60 | 13.30 | 10.00 | 39.30 | 52.50 | 42.60 | 37.22 | 56.92 | 77.80 |
| Full - w/o $\psi$ | +1.40 | +6.70 | +3.30 | +0.60 | +7.50 | -1.10 | +1.01 | +0.91 | +0.13 |

### 关键发现
- 最强收益集中在竞赛级数学：AIME24 从 10.00 提升到 20.00，AIME25 从 3.30 提升到 13.30，说明游戏轨迹里的抽象策略和多轮演化确实能迁移到长链推理。
- $\psi$ 不是锦上添花的小奖励。去掉它后 AMC-23 下降 7.50，AIME24 下降 6.70，说明“推理是否随状态推进”是 transfer 的关键变量。
- 代码生成也有收益，但幅度小于数学：HumanEval 从 Base 的 67.93 到 77.93，相对 SPIRAL 只高 0.49，说明 SPIRAL 已经学到了不少结构化程序推理，STRATAGEM 主要补足抽象和演化质量。

### 额外分析

| 分析项 | 结果 | 说明 |
|--------|------|------|
| 人类评价：Abstraction | STRATAGEM 4.06，SPIRAL 3.24，Base 2.48 | 轨迹更少依赖游戏专有词，更像可迁移推理 |
| 人类评价：Progression | STRATAGEM 4.18，SPIRAL 3.08，Base 2.32 | 多轮推理更会承接上一轮状态 |
| 评估器一致性 | GPT-4 vs Claude 的 $\varphi$ Cohen κ = 0.71，$\psi$ κ = 0.68 | 奖励信号不是单一评估器的任性偏好 |
| OOD 游戏胜率 | Snake 0.35 vs SPIRAL 0.15，Pig Dice 0.96 vs 0.76，Truth & Deception 0.80 vs 0.72 | 学到的不是原始三种游戏的局部策略 |

## 亮点与洞察
- 最巧妙的地方是把“赢游戏”拆成“结果好”和“为什么好”。这让 self-play 的可验证奖励继续发挥作用，同时避免模型把不可迁移的游戏小技巧当成通用推理能力。
- $\varphi$ 采用乘性调制很合理：如果一条轨迹输了，即使推理看起来抽象，也不应该被强推；如果赢了但靠的是游戏术语绑定的捷径，也不该得到完整优势。
- $\psi$ 把多轮轨迹看成一个动态推理对象，而不是一组独立动作。这个思想可以迁移到 agent 训练、代码调试、长文 QA 等任务：真正有用的信号不是单步答案，而是中间状态如何改变后续推理。
- 评估器一致性和人工评价是必要补强。否则这种 reward shaping 很容易被质疑成“GPT-4 喜欢的风格”，论文用 Claude、Gemini 和专家评分证明它捕捉到的至少是跨评估者可识别的轨迹属性。

## 局限与展望
- 训练环境仍然只有 TextArena 的三个文本游戏，覆盖空间、概率和谈判，但还不足以代表开放世界、多智能体协作或长周期任务。
- 主实验使用 Qwen3-4B，另补 Qwen3-4B-Instruct；更大模型、更多架构和不同预训练语料上的 scaling law 还没有系统验证。
- $\varphi$ 和 $\psi$ 依赖 GPT-4 评价，虽然成本相对训练可接受，但会引入外部 API、可复现性和潜在偏差问题。后续可以蒸馏本地 reward model。
- 论文主要展示 accuracy/pass@1，较少分析失败类型，例如哪些代码题仍然不受益、游戏轨迹中的哪些抽象模式会误迁移。

## 相关工作与启发
- **vs SPIRAL**: SPIRAL 用文本游戏 self-play 的终局胜负提升推理能力，STRATAGEM 进一步问“哪些胜利轨迹值得学”。优势是迁移信号更细，代价是需要额外 evaluator。
- **vs 普通 RLHF/RLAIF**: 传统偏好奖励多评估最终回答质量，STRATAGEM 评估整条 trajectory 的推理属性，更接近过程奖励，但又避免 token-level process reward 的标注成本。
- **vs LLM routing / ensemble**: 一些方法通过选择更强模型或组合模型提升推理，本文则试图让单个小模型从游戏中内化可迁移 reasoning pattern，目标更偏训练而非推理时调度。
- **对代码智能的启发**：代码生成常需要把需求分解成状态、约束和边界条件。如果用类似 $\varphi$ / $\psi$ 的轨迹奖励筛选代码调试过程，也许能奖励“抽象到算法原则”和“根据测试反馈演化”的推理，而不是只看最终 AC。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 用轨迹优势调制解释游戏推理迁移，思路清楚且比单纯 self-play 更细。
- 实验充分度: ⭐⭐⭐⭐☆ 数学、通用推理、代码、OOD 游戏、人类评价和消融都覆盖了，但大模型规模验证还偏少。
- 写作质量: ⭐⭐⭐⭐☆ 动机和两个 reward 设计讲得很顺，公式也不复杂；部分 evaluator 细节需要读附录才能完全复现。
- 价值: ⭐⭐⭐⭐☆ 对“如何从可验证环境学通用推理”很有启发，尤其适合后续迁移到 agent 和代码训练。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improving Rationality in the Reasoning Process of Language Models through Self-playing Game](../../ICML2025/llm_reasoning/improving_rationality_in_the_reasoning_process_of_language_models_through_self-p.md)
- [\[ACL 2026\] HISR: Hindsight Information Modulated Segmental Process Rewards for Multi-turn Agentic Reinforcement Learning](hisr_hindsight_information_modulated_segmental_process_rewards_for_multi-turn_ag.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](../../AAAI2026/llm_reasoning/llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)
- [\[AAAI 2026\] SERL: Self-Examining Reinforcement Learning on Open-Domain](../../AAAI2026/llm_reasoning/serl_self-examining_reinforcement_learning_on_open-domain.md)
- [\[ACL 2026\] Reliability-Aware Adaptive Self-Consistency for Efficient Sampling in LLM Reasoning](reliability-aware_adaptive_self-consistency_for_efficient_sampling_in_llm_reason.md)

</div>

<!-- RELATED:END -->
