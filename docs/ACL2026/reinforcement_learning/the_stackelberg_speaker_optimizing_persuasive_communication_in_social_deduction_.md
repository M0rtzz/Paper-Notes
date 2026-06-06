---
title: >-
  [论文解读] The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games
description: >-
  [ACL 2026][强化学习][说服性通信] 本文将社交推理游戏中的回合制对话建模为 Stackelberg 博弈，当前玩家作为 leader 通过度量下一玩家的响应分布来优化话语的说服力影响，使用 GRPO 训练 Refiner 模型在狼人杀、阿瓦隆等四个游戏基准上显著超越基线。
tags:
  - "ACL 2026"
  - "强化学习"
  - "说服性通信"
  - "社交推理游戏"
  - "Stackelberg博弈"
  - "GRPO"
  - "LLM智能体"
---

# The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games

**会议**: ACL 2026  
**arXiv**: [2510.09087](https://arxiv.org/abs/2510.09087)  
**代码**: [https://3dagentworld.github.io/leader_follower](https://3dagentworld.github.io/leader_follower)  
**领域**: 强化学习 / 社交推理游戏  
**关键词**: 说服性通信, 社交推理游戏, Stackelberg博弈, GRPO, LLM智能体

## 一句话总结

本文将社交推理游戏中的回合制对话建模为 Stackelberg 博弈，当前玩家作为 leader 通过度量下一玩家的响应分布来优化话语的说服力影响，使用 GRPO 训练 Refiner 模型在狼人杀、阿瓦隆等四个游戏基准上显著超越基线。

## 研究背景与动机

**领域现状**：LLM 智能体在社交推理游戏（SDGs）如狼人杀、阿瓦隆等中取得了显著进展。现有方法主要聚焦信息处理（推断其他玩家角色）和策略选择（选择最优行动）。

**现有痛点**：现有方法忽视了说服性通信的核心作用——在 SDGs 中，成功不仅取决于做出正确推断，更取决于说服他人按照自己的意图行动。现有 RL 方法（如 SLA、LSPO）将丰富的自然语言空间简化为有限动作分类问题，无法在连续语言空间中优化话语。

**核心矛盾**：SDGs 的核心挑战不是"知道什么是对的"，而是"让别人相信自己是对的"。说服性维度是游戏成功和真实人类交互的核心，但在当前研究中几乎未被触及。

**本文目标**：显式建模和优化社交推理游戏中的说服性通信，使智能体能主动引导对话流向有利结果。

**切入角度**：借用博弈论中的 Stackelberg 博弈框架——如果 leader 充分理解 follower 对不同行动的响应分布，就可以选择最大化自身效用的行动。在回合制对话中，当前说话者就是 leader。

**核心 idea**：训练一个 Refiner 模型将基础话语精炼为更具说服力的版本，奖励信号基于该话语对下一玩家响应概率分布的偏移量（增加期望响应概率、减少不期望响应概率）。

## 方法详解

### 整体框架

分三步：(1) 意图识别——API LLM 分析当前局势，生成期望/不期望的 follower 响应各 K=3 组；(2) 影响度量——API LLM 生成基础话语，Refiner 将其精炼为多个候选，Measurer 计算每个候选对 follower 响应分布的偏移作为奖励；(3) 策略优化——用 GRPO 优化 Refiner 使其最大化说服力影响。

### 关键设计

1. **Stackelberg 建模与意图识别**:

    - 功能：将每个说话回合建模为 leader-follower 交互，明确优化目标
    - 核心思路：当前玩家 $p_t$ 作为 leader，下一玩家 $p_{t+1}$ 作为 follower。leader 根据游戏规则 $\mathcal{R}$、游戏状态 $G_t$、对话历史 $D_t$ 和隐藏角色 $r_t$，用后端 LLM 生成 K=3 组期望响应 $\hat{u}_{t+1}^{+,(k)}$ 和不期望响应 $\hat{u}_{t+1}^{-,(k)}$
    - 设计动机：显式定义"什么是好的说服效果"，将模糊的说服目标转化为可度量的概率偏移

2. **说服力影响度量（Impact Measurement）**:

    - 功能：为每个候选话语计算其对 follower 行为的说服力奖励
    - 核心思路：用 Qwen2.5-72B 作为 Measurer 模拟 follower 的响应模式。对候选话语 $u_t^{(i)}$，奖励 $R(u_t^{(i)}) = \sum_k \log P_\mathcal{F}(\hat{u}_{t+1}^{+,(k)} | \text{ctx} \cup \{u_t^{(i)}\}) - \sum_k \log P_\mathcal{F}(\hat{u}_{t+1}^{-,(k)} | \text{ctx} \cup \{u_t^{(i)}\})$
    - 设计动机：直接在 follower 的概率空间中度量说服效果，比人工标注或启发式评估更客观

3. **GRPO 策略优化**:

    - 功能：训练 Refiner 在自然语言空间中优化话语的说服力
    - 核心思路：Qwen2.5-7B + LoRA (rank 16) 作为 Refiner，采样 n=8 个候选，用 GRPO 计算组内相对优势进行策略优化，KL 散度正则化防止偏离过远
    - 设计动机：GRPO 无需额外 critic 模型，直接利用批次内奖励分布计算相对优势

### 损失函数 / 训练策略

GRPO 目标函数：$\mathcal{J}(\theta) = \mathbb{E}_c[\frac{1}{n}\sum_i \mathcal{L}_i - \beta D_{KL}(\pi_\theta || \pi_{ref})]$，n=8, ε=0.2, β=0.04。每游戏 500 局自对弈，选 4000 实例训练。后端 LLM 随机选自 GPT-4o/Gemini-2.5-Flash/Claude-3.5-Haiku。学习率 $1 \times 10^{-6}$，4×A800 训练 3 epochs 约 50 小时。

## 实验关键数据

### 主实验

| 游戏 | 方法 | 总体胜率 |
|------|------|---------|
| 狼人杀 | LSPO | 38.6% |
| 狼人杀 | **Ours + LSPO** | **44.7%** |
| 阿瓦隆 | Strategist | 57.4% |
| 阿瓦隆 | **Ours + Strategist** | **61.3%** |
| ONUW | RL-ins. | 48.5% |
| ONUW | **Ours + RL-ins.** | **51.5%** |

### 消融实验

| 奖励变体 | 狼人杀 Avg | 阿瓦隆 Avg | ONUW Avg |
|---------|-----------|-----------|---------|
| ReAct (基线) | 49.0 | 44.0 | 48.0 |
| Pos-Only + ReAct | 64.0 | 58.0 | 60.0 |
| Neg-Only + ReAct | 49.0 | 46.0 | 47.0 |
| **Ours + ReAct** | **70.0** | **61.0** | **61.0** |

### 关键发现

- 正向奖励（增加期望响应概率）比负向奖励（减少不期望响应概率）贡献大得多
- Refiner 与强基线结合效果更好，说明方法是补充而非替代现有策略
- 在欺骗角色上提升尤为显著——狼人杀中狼人胜率从 79% 提升到 84.2%
- 方法成功泛化到 Sotopia 社交模拟环境，不限于 SDGs

## 亮点与洞察

- Stackelberg 博弈建模回合制对话非常自然——将说服力量化为"对方响应概率偏移"比直接优化胜率更精细
- 用独立大模型模拟 follower 响应分布巧妙绕过了 API LLM 无法获取概率的限制
- Refiner 作为"话语精炼器"的定位很实用——保留强 API LLM 语义理解，小模型做说服力增强

## 局限与展望

- Measurer 用固定大模型模拟 follower，实际对手行为可能不同
- 训练时使用完整信息（对手角色已知），推理时不可用
- 每个游戏需单独训练 checkpoint，跨游戏迁移未探索

## 相关工作与启发

- **vs SLA/LSPO**: 它们将语言简化为有限候选选择，本文直接在连续语言空间优化。Refiner 可叠加使用
- **vs Cicero**: Cicero 在外交游戏中寻全局均衡，本文用局部 Stackelberg 优化避免计算不可行

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Stackelberg 建模+说服力奖励的组合新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 三个 SDGs + Sotopia、多基线叠加、消融完整
- 写作质量: ⭐⭐⭐⭐ 理论清晰但部分公式密集
- 价值: ⭐⭐⭐⭐ 为 LLM 智能体说服性通信提供可行框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Learning in Structured Stackelberg Games](../../ICML2026/reinforcement_learning/learning_in_structured_stackelberg_games.md)
- [\[ICLR 2026\] Learning to Play Multi-Follower Bayesian Stackelberg Games](../../ICLR2026/reinforcement_learning/learning_to_play_multi-follower_bayesian_stackelberg_games.md)
- [\[ICLR 2026\] Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information](../../ICLR2026/reinforcement_learning/nearly-optimal_bandit_learning_in_stackelberg_games_with_side_information.md)
- [\[NeurIPS 2025\] Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis](../../NeurIPS2025/reinforcement_learning/learning_in_stackelberg_mean_field_games_a_non-asymptotic_analysis.md)
- [\[ACL 2026\] Savoir: Learning Social Savoir-Faire via Shapley-based Reward Attribution](savoir_learning_social_savoir-faire_via_shapley-based_reward_attribution.md)

</div>

<!-- RELATED:END -->
