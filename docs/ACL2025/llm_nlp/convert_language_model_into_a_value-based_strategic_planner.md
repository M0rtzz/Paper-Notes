---
title: >-
  [论文解读] Convert Language Model into a Value-based Strategic Planner
description: >-
  [ACL 2025][LLM/NLP][emotional support conversation] 提出 straQ* 框架，将 LLM 的 next-token prediction 转化为 next-strategy prediction，用 Bellman 方程训练 LLM 作为策略级 Q 网络，在情感支持对话（ESC）中根据长期回报规划最优支持策略，以即插即用的轻量规划器引导对话 LLM 生成高质量回复。
tags:
  - ACL 2025
  - LLM/NLP
  - emotional support conversation
  - Q-learning
  - strategic planning
  - 强化学习
  - dialogue
---

# Convert Language Model into a Value-based Strategic Planner

**会议**: ACL 2025  
**arXiv**: [2505.06987](https://arxiv.org/abs/2505.06987)  
**代码**: [https://github.com/suran662/StraQ](https://github.com/suran662/StraQ)  
**领域**: LLM/NLP  
**关键词**: emotional support conversation, Q-learning, strategic planning, reinforcement learning, dialogue

## 一句话总结

提出 straQ* 框架，将 LLM 的 next-token prediction 转化为 next-strategy prediction，用 Bellman 方程训练 LLM 作为策略级 Q 网络，在情感支持对话（ESC）中根据长期回报规划最优支持策略，以即插即用的轻量规划器引导对话 LLM 生成高质量回复。

## 研究背景与动机

**领域现状**：情感支持对话（Emotional Support Conversation, ESC）旨在通过有效对话缓解用户情绪困扰。ESC 理论将支持过程分为三个阶段：探索（Exploration）→ 安慰（Comforting）→ 行动（Action），支持者需要在对话中选择合适的策略（如提问、共情、提供建议等）并实现阶段的自然过渡。随着 LLM 发展，基于 LLM 的 ESC 方案取得了显著进展。

**现有痛点**：现有 LLM 方法大多关注即时回复质量，缺乏对长期支持策略的系统规划。具体表现为：(1) LLM 倾向于反复使用同一策略（如持续"复述与改写"），导致对话阶段转换不顺畅；(2) 没有从 MDP 状态模型角度建模 ESC，无法优化长期满意度；(3) 策略选择存在偏差，不同策略的使用频率严重不均衡。

**核心矛盾**：如何让 LLM 在多轮对话中基于长期回报（而非贪心地追求当前最优）选择支持策略？

**切入角度**：将 ESC 任务形式化为策略级 MDP，借鉴 Deep Q-Learning（DQN），用 LLM 的平均 token logit 作为 Q 值近似策略价值函数，通过 Bellman 方程微调 LLM 参数，将其转化为即插即用的策略规划器。

## 方法详解

### 策略级 MDP 定义

将 ESC 任务建模为五元组 MDP $(S, A, R, T, \gamma)$：

- **状态 $s$**：由对话背景描述、当前情绪、对话历史和当前用户发言组成，$s = \{desc, e, h, query\}$
- **动作 $a$**：ESConv 数据集定义的 8 种支持策略（提问、复述/改写、情感反映、自我表露、肯定/安慰、提供建议、提供信息、其他）
- **奖励 $r$**：用户即时满意度，由数据集标注或 GPT-4 评分获得
- **转移函数 $T$**：每轮对话后历史更新，用户产生新的发言和情绪

关键创新在于动作空间是「策略级」而非「token 级」，这使得 MDP 的动作空间小（仅 8 个策略）且具有明确语义，适合 Q-learning 高效求解。

### LLM 作为 Q 函数

核心思想是直接复用 LLM 架构作为 Q 网络，不引入额外的 value head：

1. 构造指令模板 $\mathcal{I}(s)$，将状态 $s$ 填入多选题格式的 prompt
2. 将指令与策略拼接 $\mathcal{I}(s) \oplus a$，输入 LLM
3. **Q 值定义**：取 LLM 输出中策略 token 的平均 log probability 作为 $Q_\theta(s, a)$
4. 推理时遍历所有 $K$ 个策略，选择 Q 值最大的策略：$a^\star = \arg\max_a \text{LLM}(\mathcal{I}(s) \oplus a)$

设计动机：平均 logit 自然反映 LLM 对该策略在当前上下文中的"信心"程度，无需额外网络参数。将指令设计为 MCQ（多选题）格式，让 LLM 选择策略编号，强化其对策略选择的理解能力。

### Bellman 方程训练

用 DQN 的 TD loss 替代传统交叉熵损失微调 LLM：

$$\mathcal{L}(\theta) = |r(s,a) + \gamma Q_\phi(s', a') - Q_\theta(s,a)|^2$$

- $\theta$：Q 网络（当前 LLM）参数
- $\phi$：目标 Q 网络参数，每 10 步从 $\theta$ 同步
- $\gamma = 0.85$：折扣因子
- 利用 Transformer 的因果掩码在整个序列上并行执行 Bellman 更新

训练将 next-token prediction 转化为 next-strategy prediction，loss 从 token 级交叉熵变为策略级 TD loss。

### 两种奖励机制

- **straQ\*-imit（模仿）**：数据集中的 $(s, a)$ 对赋予 $r = +1$，随机采样不同策略赋予 $r = -1$，正负比 1:1。直接模仿专家标注
- **straQ\*-distill（蒸馏）**：用 GPT-4 对每个 $(s, a)$ 打分 0-5 分作为奖励。蒸馏教师模型知识

### 推理流程

straQ* 作为即插即用规划器使用：(1) 规划器 LLM（1B）根据当前状态为所有策略计算 Q 值，选择最优策略；(2) 对话 LLM（8B）根据选定策略和上下文生成最终回复。规划器仅负责策略选择，不参与回复生成。

## 实验关键数据

### 域内评估（ESConv 数据集）

| 方法 | Acc ↑ | Q ↑ | B ↓ | B-2 ↑ | R-L ↑ |
|------|-------|-----|-----|-------|-------|
| LLaMA3-8B Direct | 11.80 | 10.26 | 1.61 | 3.47 | 10.64 |
| + Direct-Refine | 17.08 | 11.07 | 1.27 | 3.10 | 6.13 |
| + Self-Refine | 17.58 | 13.61 | 1.92 | 3.34 | 9.71 |
| + CoT | 15.32 | 10.38 | 1.69 | 3.16 | 10.50 |
| + FSM | 17.37 | 11.15 | 0.81 | 4.12 | 11.83 |
| **+ straQ\*-distill** | **41.22** | **38.95** | **0.57** | 3.89 | 11.80 |
| **+ straQ\*-imit** | **46.83** | **43.15** | 0.80 | **3.89** | **12.84** |
| LLaMA3-8B + SFT | 32.43 | 21.29 | 1.28 | 6.97 | 16.59 |
| + SFT + FSM | 28.83 | 18.36 | 1.32 | 7.57 | 17.42 |
| + SFT + straQ\*-distill | 41.22 | 38.95 | 0.57 | 7.01 | 16.93 |
| + SFT + straQ\*-imit | **46.83** | **43.15** | 0.80 | **7.63** | **17.30** |

straQ\*-imit 在策略准确率上达到 46.83%，比 Direct 的 11.80% 提升近 4 倍；策略偏差 B 从 1.61 降至 0.57-0.80。

### 域外泛化（EmpatheticDialogues）

| 方法 | B-2 | R-L | Dist-2 | CIDEr |
|------|-----|-----|--------|-------|
| Direct | 3.09 | 9.91 | 25.23 | 1.60 |
| + CoT | 2.91 | 9.79 | 32.65 | 1.37 |
| + FSM | 3.33 | 10.80 | 33.37 | 2.96 |
| **+ straQ\*-distill** | **4.49** | **12.93** | **46.53** | **8.36** |
| + straQ\*-imit | 4.27 | 12.66 | 46.80 | 8.11 |

在未见过的 EmpatheticDialogues 上，straQ\* 的 CIDEr 从 FSM 的 2.96 提升到 8.36，泛化能力显著。distill 版本在 OOD 上优于 imit，说明 GPT-4 蒸馏的知识更具通用性。

### 人类评价

| 方法 | Fluency | Emotion | Acceptance | Effectiveness | Sensitivity | Satisfaction |
|------|---------|---------|------------|---------------|-------------|-------------|
| 原始数据集 | 3.51 | 3.61 | 3.40 | 3.10 | 3.50 | 3.30 |
| LLaMA3-8B Direct | 2.95 | 3.00 | 2.60 | 2.40 | 2.70 | 2.60 |
| + FSM | 3.30 | 3.35 | 2.90 | 2.90 | 3.00 | 2.93 |
| + SFT + CoT | 3.67 | 3.61 | 3.22 | 3.67 | 3.56 | 3.45 |
| **+ straQ\*-distill** | **3.52** | **3.65** | **3.59** | **3.73** | **3.71** | **3.66** |
| + straQ\*-imit | 3.42 | 3.25 | 3.23 | 3.07 | 3.10 | 3.13 |

straQ\*-distill 的满意度 3.66 超过原始数据集标注的 3.30，也超过所有基线方法。在 Effectiveness 和 Sensitivity 维度上优势最明显。

### 消融实验

| 方法 | Acc ↑ | Q ↑ | B ↓ | B-2 ↑ | R-L ↑ |
|------|-------|-----|-----|-------|-------|
| w/ value head | 19.81 | 11.40 | 1.66 | 6.74 | 15.99 |
| auto-regressive (SFT) | 46.22 | 43.01 | 0.69 | 7.25 | 16.48 |
| **straQ\*-imit** | **46.83** | **43.15** | 0.80 | **7.63** | **17.03** |

用独立 value head 的准确率仅 19.81%，说明直接用平均 logit 作 Q 值优于额外分类头；straQ\* 在回复质量指标（B-2, R-L）上也超过纯 SFT。

### 策略价值分析

| 方法 | 平均奖励 (GPT-4) | 平均价值 |
|------|-----------------|---------|
| 原始数据集 | 3.01 | 252.09 |
| LLaMA3-8B Direct | 3.66 | 346.31 |
| straQ\*-distill | **3.99** | 424.78 |
| straQ\*-imit | 3.72 | **445.95** |

straQ\* 的长期价值（累积回报）显著高于基线，验证了 Q-learning 确实在优化长期回报。

## 亮点与洞察

- **极简但有效的 Q 值定义**：不引入额外网络，直接用 LLM 平均 logit 作 Q 值，训练后损失收敛，Q 值有效区分策略优劣
- **即插即用架构**：1B 规划器 + 8B 对话模型的解耦设计，规划器可搭配任意对话 LLM 使用
- **两种奖励互补**：imit 在自动指标/域内占优，distill 在人类评价/域外泛化占优，适用于不同场景
- **策略转移矩阵**验证了 straQ\* 学到了合理的 ESC 阶段转换（I→II→III），而 Direct 方法停留在第一阶段

## 局限与展望

- 仅在情感支持对话（ESConv + EmpatheticDialogues）验证，未扩展至协商、医疗咨询等其他策略导向对话场景
- 人类评价可能存在偏差——评估者是实习生而非真实求助者，样本量有限
- 规划器需遍历所有策略计算 Q 值，策略空间扩大时推理开销线性增长
- 当前 Q 值用平均 logit 近似缺乏严格理论保证，收敛性依赖经验验证

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性** ⭐⭐⭐⭐：将 DQN 嵌入 LLM 的方式新颖，策略级 MDP + 平均 logit 作 Q 值的设计简洁有效
- **实验充分度** ⭐⭐⭐⭐⭐：域内/域外测试、人类评价、消融、敏感性分析、case study 齐全
- **实用价值** ⭐⭐⭐⭐：即插即用的轻量规划器思路可迁移至其他需要长期策略规划的对话场景
- **写作质量** ⭐⭐⭐：论文整体清晰，但符号较多，部分推导可进一步精简
