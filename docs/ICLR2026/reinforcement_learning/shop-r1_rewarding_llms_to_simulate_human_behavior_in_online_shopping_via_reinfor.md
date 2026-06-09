---
title: >-
  [论文解读] Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning
description: >-
  [强化学习] 提出 Shop-R1 框架，通过分层奖励机制和难度感知缩放的强化学习方法，显著提升 LLM 模拟真实人类在线购物行为的能力，相比 SFT 基线精确匹配提升超过 65%。
tags:
  - "强化学习"
---

# Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2507.17842](https://arxiv.org/abs/2507.17842)
- **代码**: [https://damon-demon.github.io/shop-r1.html](https://damon-demon.github.io/shop-r1.html)
- **领域**: 强化学习
- **关键词**: LLM, reinforcement learning, human behavior simulation, online shopping, hierarchical reward, GRPO

## 一句话总结
提出 Shop-R1 框架，通过分层奖励机制和难度感知缩放的强化学习方法，显著提升 LLM 模拟真实人类在线购物行为的能力，相比 SFT 基线精确匹配提升超过 65%。

## 研究背景与动机
- LLM 在模拟人类网页行为方面展现潜力，但现有方法（零样本提示、SFT）效果仍不理想。
- **零样本提示**：缺乏个性化和适应性，准确率极低（0.32%）。
- **SFT 方法**：用 Claude 3.5 Sonnet 生成"推理-动作"训练数据进行微调，但性能受限于用于生成推理的模型能力天花板。
- **RL 的挑战**：直接用稀疏二值奖励做 RL 效果极差（1.01%），且容易出现 reward hacking——模型反复预测简单的 "terminate" 动作以获取容易的奖励。
- 核心问题：如何设计适合行为模拟（而非任务完成）的 RL 奖励？

## 方法详解

### 整体框架
Shop-R1 把"模拟人类购物"拆成两步预测：给定当前网页上下文和历史动作，模型先生成一段行为推理（rationale），再据此预测下一步动作——click、type_and_submit 或 terminate。难点在于真实推理没有 ground truth，且二值奖励会被模型钻空子反复输出 terminate，因此 Shop-R1 的核心贡献是一套层层递进、专为行为模拟设计的奖励函数。

### 关键设计

**1. 格式奖励：给后续奖励计算上保险**

行为模拟的奖励要拆解到推理和动作两个字段上分别打分，前提是模型输出能被可靠解析。Shop-R1 用一个二值的格式奖励守住这道门：当输出是合法 JSON 且同时含 rationale 和 action 两个 key 时给 0.5，否则给 0。这一步本身不评判内容好坏，只确保下游的自确定性奖励和分层动作奖励有结构化的输入可用，避免因格式崩坏而产生无意义的梯度。

**2. 自确定性奖励：在没有推理标签时也能监督推理质量**

真实用户的内心推理无从获取，无法用监督信号约束 rationale。Shop-R1 转而用模型对自身推理的"信心"作为代理：把推理 token 分布与均匀分布之间的平均 KL 散度作为奖励 $s(r_t | q_t) = \frac{1}{N|V|} \sum_{j=1}^{N} \sum_{i=1}^{|V|} p_{ij} \log\left(\frac{p_{ij}}{U_i}\right)$，其中 $N$ 为推理 token 数、$|V|$ 为词表大小、$U_i$ 为均匀分布。分布越偏离均匀（越尖锐），说明模型对该推理越确定、越自洽，奖励越高。这样即使缺少外部标签，也能把推理往"自信而一致"的方向推，而不是放任其退化成空泛文本。

**3. 分层奖励：让粗粒度和细粒度都拿到信号**

一个动作正确与否不是非黑即白：预测对动作类型只对了一半，还要看子动作属性（点击的按钮名、输入的文本）是否吻合。Shop-R1 因此把动作奖励按粒度分层累加——先判类型，再判属性，最后判文本相似度（用 ROUGE-L 衡量预测与真实字符串的重合），具体取值如下：

| 动作类型 | 类型奖励 | 子动作属性奖励 | 文本相似度奖励 |
|---------|---------|--------------|--------------|
| terminate | 0.3 | 无 | 无 |
| click | 0.3 | +0.2 (若 name ≠ ∅) | +DARS × ROUGE-L(name) |
| type_and_submit | 0.3 | +0.1 (name) + 0.1 (text) | +0.1×ROUGE-L(name) + DARS×ROUGE-L(text) |

这种逐层叠加的结构让模型即便没完全答对，也能因为类型对了、文本接近而拿到部分奖励，从而同时优化类型级和精确匹配两个目标，而非只盯着全对/全错的稀疏信号。

**4. 难度感知奖励缩放（DARS）：堵住 terminate 刷分的漏洞**

预测动作类型（三选一）远比预测一段按钮标签或搜索查询容易，如果各部分等权，模型会发现反复输出最简单的 terminate 就能稳拿类型奖励，于是 reward hacking。DARS（Difficulty-Aware Reward Scaling）对困难的长文本子动作所对应的奖励项乘上一个放大因子（默认 1000，见上表中文本相似度奖励的 DARS 系数），使得正确预测这些难部分的收益远高于躺着拿 terminate 的收益，从机制上消除了退化捷径。

### 损失函数 / 训练策略
总体 RL 目标在分层动作奖励之外，叠加自确定性项与对参考策略的 KL 正则：

$$\max_{\pi_\theta} \mathbb{E}_{r,a \sim \pi_\theta(q)} \left[ v(a) + \alpha s(r) - \beta \text{KL}(\pi_\theta(r,a|q) \| \pi_{\text{ref}}(r,a|q)) \right]$$

其中 $v(a)$ 为分层动作奖励，$\alpha=0.005$ 控制自确定性权重，$\beta=0.001$ 控制 KL 正则强度以防偏离 SFT 初始策略过远。训练分两阶段：先在 Claude 生成的 rationale-action 数据上冷启动 SFT（4 epochs，lr=2e-5）让模型学会基本格式与推理范式，再用 GRPO 做策略优化（500 步，lr=1e-7，batch=64，上下文长度 32K）让上述奖励真正塑造行为。

## 实验关键数据

### 主实验：不同方法性能对比

| 模型 (Qwen-2.5-3B) | 精确动作匹配 | 动作类型准确率 | 动作类型 F1 |
|---------------------|------------|--------------|-----------|
| Zero-shot | 0.32% | 15.33% | 16.15% |
| RL (Binary) | 1.01% | 6.17% | 9.92% |
| SFT | 16.76% | 22.25% | 24.52% |
| SFT + RL (Binary) | 16.55% | 23.74% | 28.07% |
| **Shop-R1 (Ours)** | **27.72%** | **36.40%** | **31.28%** |

> Shop-R1 相比 SFT 基线精确匹配提升 65%+，同时提升动作类型和细粒度匹配。

### 消融实验：跨模型规模

| 模型规模 | SFT | Shop-R1 | 提升 |
|---------|-----|---------|------|
| Qwen-2.5-0.5B | 9.90% | **27.72%** | +180% |
| Qwen-2.5-1.5B | 10.86% | **24.11%** | +122% |
| Qwen-2.5-3B | 16.76% | **27.72%** | +65% |

> 分层奖励机制在小模型上提升更为显著，0.5B 模型甚至达到 3B 模型同等性能。

### 关键发现
1. 稀疏二值 RL 奖励不足以引导行为模拟学习，甚至可能退化
2. 分层奖励同时提升粗粒度（类型级）和细粒度（精确匹配）性能
3. DARS 有效防止了 reward hacking（模型不再只预测 terminate）
4. 自确定性信号为无 ground truth 的推理提供有效监督
5. 框架可泛化到不同网页和 GUI 交互任务

## 亮点与洞察
- **首次将 RL 引入面向模拟的行为建模**，区别于现有面向任务完成的 Web Agent 研究
- **精心的奖励工程**：从格式→推理→动作类型→子动作，层层递进的奖励设计
- **DARS 机制**巧妙解决了 reward hacking 问题
- **0.5B 模型 = 3B 模型表现**：说明奖励设计比模型规模更重要

## 局限性
- 任务仍限定在特定电商环境（Shop-CART 数据集），泛化性需验证
- 推理质量仅通过自确定性间接评估，缺乏外部验证
- 模型预测仍与真实用户行为有较大差距（27.72% 精确匹配）
- 上下文长度限制（32K tokens）可能丢失长会话信息

## 相关工作
- **LLM 行为模拟**: ReAct (Yao et al., 2023), WebAgent (Gur et al., 2023), UX-Agent (Lu et al., 2025)
- **奖励设计**: RLHF (Ouyang et al., 2022), DPO (Rafailov et al., 2023), RLVR/DeepSeek-R1 (Guo et al., 2025)
- **购物导航 Agent**: WebArena (Zhou et al., 2023), 但关注任务完成而非行为模拟

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次将 RL 用于行为模拟（非任务完成），分层奖励设计有创意
- 理论深度: ⭐⭐⭐ — 主要是工程创新，理论贡献较少
- 实验充分性: ⭐⭐⭐⭐ — 多规模模型、消融、跨数据集验证
- 实用价值: ⭐⭐⭐⭐ — 对电商用户行为模拟有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[ICLR 2026\] Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals](unsupervised_learning_of_efficient_exploration_pre-training_adaptive_policies_vi.md)
- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)

</div>

<!-- RELATED:END -->
