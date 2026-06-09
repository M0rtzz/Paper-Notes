---
title: >-
  [论文解读] Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination
description: >-
  [AAAI 2026][强化学习][数据污染] 本文通过系统性的数据泄露审计揭示了Qwen2.5系列在MATH-500等标准数学基准上存在严重的数据污染问题，指出近期"虚假奖励也能提升数学推理"的发现是污染所致的虚假结论…
tags:
  - "AAAI 2026"
  - "强化学习"
  - "数据污染"
  - "LLM推理"
  - "虚假奖励"
  - "RLVR"
  - "数学推理评估"
---

# Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination

**会议**: AAAI 2026  
**arXiv**: [2507.10532](https://arxiv.org/abs/2507.10532)  
**代码**: [github](https://github.com/wumingqi/LLM-Math-Evaluation)  
**领域**: 强化学习  
**关键词**: 数据污染, LLM推理, 虚假奖励, RLVR, 数学推理评估

## 一句话总结

本文通过系统性的数据泄露审计揭示了Qwen2.5系列在MATH-500等标准数学基准上存在严重的数据污染问题，指出近期"虚假奖励也能提升数学推理"的发现是污染所致的虚假结论，并构建了完全无泄露的RandomCalculation基准验证只有正确奖励信号才能带来真实的推理提升。

## 研究背景与动机

近年来，强化学习（RL）在增强LLM数学推理能力方面取得了显著进展。特别是使用可验证奖励的强化学习（RLVR）——当预测答案等于真实答案时奖励为1，否则为0——因其不需要学习型奖励模型而受到广泛关注。

**令人困惑的现象**：多篇近期工作发现，即使使用随机奖励或错误奖励，也能在Qwen2.5-Math-7B上提升MATH-500的性能。更极端的是，只用1个标注样本（1-shot-RL）、甚至完全无标注（Absolute-Zero）都能获得显著提升。然而，这些"魔法"效果几乎只在Qwen2.5系列上观察到，不能迁移到Llama系列——这暗示了可能存在模型特定的问题。

**两个对立假说**：

**数据污染假说**：Qwen2.5的大规模预训练语料（高达36T tokens）包含了Internet上的GitHub仓库，其中存储了基准问题及其官方解答。虚假奖励仅仅是"唤醒"了记忆的答案

**强数学能力假说**：Qwen的预训练赋予了比Llama更强的数学能力，所以即使嘈杂的梯度更新也能帮助MATH-500。但如果真是如此，虚假奖励在干净基准上也应该有效

**本文的核心使命**：区分这两个假说，需要同时进行泄露审计和严格的分布外RLVR评估。

## 方法详解

### 整体框架

本文的研究框架分为四个递进阶段：
1. **回忆能力审计**：提出两个新指标检测Qwen在标准基准上的数据污染程度
2. **对照复现**：在MATH-500上复现虚假奖励的"成功"现象
3. **干净基准构建**：设计RandomCalculation生成器，生成完全无泄露的算术问题
4. **因果验证**：在干净基准上执行RLVR，证明虚假奖励失效

### 关键设计

#### 1. 数据污染检测：两个新指标

**Partial-Prompt Completion Rate（部分提示补全率）**：截断题目（保留前40%/60%/80%），让模型生成剩余部分，用ROUGE-L和EM评估与原题的匹配度。如果模型能准确补全被截断的题目，说明见过该题。

**Partial-Prompt Answer Accuracy（部分提示答案准确率）**：同样截断题目，检查模型生成的续写中是否包含正确答案。如果仅凭部分题目就能给出正确答案，高度指示数据泄露。

**惊人发现**：给定MATH-500前60%的题目内容，Qwen2.5-Math-7B的补全率达到54.60% EM，答案准确率达到53.6%。相比之下，Llama3.1-8B仅为3.8% EM和2.4%准确率。更关键的验证——在最新的LiveMathBench（2025年5月版）上，Qwen的补全率降至0.0%，与Llama一致，证实这是污染而非能力差异。

#### 2. RandomCalculation基准构建

设计自动生成器构建完全无泄露的算术表达式：

- **基础元素**：0到100的整数，及其衍生的分数、平方、立方
- **组合方式**：使用加减乘除四种运算，随机生成1到20步的数学表达式
- **数据集构成**：20个子数据集，每个包含1000个唯一问题
- **关键特性**：所有实例在Qwen发布后生成，保证零污染

**零样本测试验证无记忆**：Qwen2.5在RandomCalculation上的准确率随计算步数单调递减，没有任何记忆模式。

#### 3. 针对RandomCalculation的连续奖励函数

由于随机算术的真实答案常含高精度小数，标准二值RLVR无法提供有效正反馈。设计连续奖励函数：

$$r = 1 - \underbrace{0.5 \cdot \min(|a-b|, 1)}_{\text{绝对距离}} - \underbrace{0.5 \cdot \min\left(\frac{|a-b|}{|b| + \epsilon}, 1\right)}_{\text{相对距离}}$$

其中 $a$ 为模型输出，$b$ 为参考答案，$\epsilon = 10^{-6}$。

#### 4. 虚假奖励的梯度分析：利用偏差机制

论文从理论上解释了为什么虚假奖励在污染数据上"有效"。GRPO的梯度为：

$$\nabla_\theta J_{\text{CLIP}} = \nabla_\theta r_{i,t} \cdot G(r_{i,t})$$

对于高概率token（$\pi_{\text{old}} = 0.85$），上裁剪边界 $1.02 \cdot \pi_{\text{old}} = 1.02$ 超过概率上限1.0，因此梯度非负，高概率token持续被上调。由于数据污染，MATH-500的正确答案token本身就有高概率——所以随机奖励通过GRPO的利用偏差"检索"出这些记忆的答案。

对于中概率token（$\pi_{\text{old}} = 0.5$，RandomCalculation的典型值），裁剪边界 $[0.4, 0.6]$ 意味着随机奖励的梯度扰动大多被裁剪掉，$G(r_{i,t}) \approx 0$，所以无有意义的改进。

### 损失函数 / 训练策略

- **RLVR算法**：GRPO（Group Relative Policy Optimization）
- **虚假奖励类型**：
    - Random：以概率 $\gamma=0.5$ 给予奖励1
    - Inverted：翻转正确信号（$1 - \text{correct}$）
    - Mv-incorrect：使用多数投票产生的错误标签
- **训练配置**：学习率5e-7，温度1.0，每个prompt采样16个回复，batch size 128
- **硬件**：8×NVIDIA A800 80G GPUs

## 实验关键数据

### 主实验

**数据污染检测（Greedy w/o Template）**：

| 模型 | 数据集 | 80%提示EM | 60%提示EM | 40%提示EM |
|------|--------|----------|----------|----------|
| Qwen2.5-Math-7B | MATH-500 | **65.80%** | **54.60%** | **39.20%** |
| Qwen2.5-Math-7B | AMC | 55.42% | 42.17% | 36.14% |
| Qwen2.5-Math-7B | AIME2024 | 56.67% | 20.00% | 16.67% |
| Qwen2.5-Math-7B | AIME2025 | 16.67% | 0.00% | 0.00% |
| Qwen2.5-Math-7B | LiveMathBench | 5.00% | **0.00%** | **0.00%** |
| Llama3.1-8B | MATH-500 | 17.80% | 3.80% | 0.60% |

**RLVR在RandomCalculation上的效果**：

| 奖励类型 | MATH-500 (Qwen) | RandomCalc 5步 (Qwen) | RandomCalc 10步 (Qwen) |
|---------|----------------|---------------------|----------------------|
| 正确奖励 | 稳定提升 | 稳定提升，超越Max@16 | 稳定提升，超越Max@16 |
| 随机奖励 | 显著提升 | 不稳定，无可靠改进 | 无改进 |
| 反转奖励 | 轻微降低 | 快速崩溃 | 快速崩溃 |

### 消融实验

**RL前后response相似度（ROUGE-L）**：

| 数据集 | 正确奖励 | 随机奖励 | Mv-incorrect | 说明 |
|--------|---------|---------|-------------|------|
| MATH-500 | 0.555 | **0.601** | 0.563 | 虚假奖励≈检索记忆 |
| RandomCalc 5步 | 0.225 | 0.247 | 0.251 | 相似度低=真正推理 |
| RandomCalc 10步 | 0.193 | 0.251 | 0.279 | 步数越多记忆越少 |

**模板效应（Qwen Base模型）**：

| 配置 | Qwen2.5-7B准确率 | Qwen2.5-Math-7B准确率 | 说明 |
|------|----------------|---------------------|------|
| Greedy (w/o Template) | 高 | **72.20%** | 无模板时最高 |
| Greedy (w/ Template) | 显著降低 | 50.60% | 模板严重损害性能 |
| RLVR起点 (w/ Template) | ~35% | ~50% | 低估了真实起点 |

### 关键发现

1. **数据污染是Qwen虚假奖励"成功"的根本原因**：去污染后，虚假奖励的"魔法"消失
2. **只有正确奖励才能真正提升推理能力**：在RandomCalculation上，正确奖励使Qwen超越Max@16上界，而虚假奖励无法
3. **模板效应造成的起点错估**：Qwen Base模型在使用chat template后性能大幅下降，RLVR的"提升"部分来自于适应模板格式
4. **KL散度进一步确认**：MATH-500上RL前后的token-level KL散度显著低于RandomCalculation，说明在MATH-500上RL主要是在利用已有记忆
5. **Llama正确奖励也无法突破**：Llama在RandomCalculation上即使用正确奖励也无法超越Max@16，说明Qwen在数学中训练上确实更强——但这不是虚假奖励"成功"的原因

## 亮点与洞察

- **研究问题极其timely**：直击当前RL for LLM reasoning最火热的争议，具有高度的社区价值
- **检测方法简洁有效**：部分提示补全率是一种通用的数据污染检测方法，可广泛应用于其他基准
- **RandomCalculation设计巧妙**：可控难度、保证无泄露、支持连续奖励，是理想的RL评估工具
- **GRPO利用偏差的理论分析**：从梯度层面解释了为什么高概率token（记忆答案）在随机奖励下会被进一步增强
- **对社区的重要警示**：提醒研究者在评估RL方法时必须使用无污染基准，且测试多个模型系列

## 局限与展望

1. **计算资源限制**：仅测试了Qwen2.5和Llama3.1的子集，未覆盖更多模型家族（如Gemma、Mistral等）
2. **RandomCalculation较为简单**：仅涉及基础算术运算，不能全面代表数学推理能力（缺乏代数、几何、概率等）
3. **未区分污染类型**：未分析是完整题目泄露、类似题型泄露还是解法模式泄露
4. **RL方法覆盖有限**：仅测试了GRPO，未验证PPO、REINFORCE等其他RL算法是否有相同现象
5. **正向结论的泛化性**："正确奖励有效"的结论在更复杂的推理任务（如证明、代码）上是否成立需要进一步验证

## 相关工作与启发

- 直接回应了Spurious Rewards等工作的结论，提供了反面证据
- TTRL（Test-time RL）、Few-Shot-RL等方法在Qwen上的"成功"都可能受到数据污染的影响
- 启发：未来RL for reasoning研究应标配：(1) 多模型系列测试，(2) 无污染基准验证，(3) 数据泄露审计
- LiveMathBench、AIME2025等时间戳晚于预训练截止的基准更值得信赖

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次系统性揭示数据污染如何导致RL研究的虚假结论
- 实验充分度: ⭐⭐⭐⭐ — 检测、复现、构建、验证四阶段完整，梯度分析深入
- 写作质量: ⭐⭐⭐⭐⭐ — 论证逻辑严密，图表直观，结论明确
- 价值: ⭐⭐⭐⭐⭐ — 对当前RL for LLM reasoning社区具有重要的方法论警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Learning from Synthetic Data Improves Multi-hop Reasoning](../../ICLR2026/reinforcement_learning/learning_from_synthetic_data_improves_multi-hop_reasoning.md)
- [\[AAAI 2026\] Reasoning with Exploration: An Entropy Perspective](reasoning_with_exploration_an_entropy_perspective.md)
- [\[ICML 2026\] CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning](../../ICML2026/reinforcement_learning/cpmobius_iterative_coach-player_reasoning_for_data-free_reinforcement_learning.md)
- [\[AAAI 2026\] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)
- [\[ACL 2026\] LearnAlign: Data Selection for LLM Reinforcement Learning with Improved Gradient Alignment](../../ACL2026/reinforcement_learning/learnalign_data_selection_for_llm_reinforcement_learning_with_improved_gradient_.md)

</div>

<!-- RELATED:END -->
