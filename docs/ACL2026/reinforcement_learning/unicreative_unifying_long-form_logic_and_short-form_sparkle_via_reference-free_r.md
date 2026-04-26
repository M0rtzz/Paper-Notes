---
title: >-
  [论文解读] UniCreative: Unifying Long-form Logic and Short-form Sparkle via Reference-Free Reinforcement Learning
description: >-
  [ACL 2026][创意写作] 本文提出 UniCreative 框架，通过自适应约束偏好优化（ACPO）和自适应标准生成式奖励模型（AC-GenRM），在无需 SFT 和参考答案的条件下统一长文本（规划→写作）和短文本（直接生成）两种创意写作模式，模型涌现出自主区分任务类型的元认知能力。
tags:
  - ACL 2026
  - 创意写作
  - 无参考强化学习
  - 偏好优化
  - 生成式奖励模型
  - 元认知
---

# UniCreative: Unifying Long-form Logic and Short-form Sparkle via Reference-Free Reinforcement Learning

**会议**: ACL 2026  
**arXiv**: [2604.05517](https://arxiv.org/abs/2604.05517)  
**代码**: https://github.com/weixiaolong94-hub/UniCreative  
**领域**: 强化学习/创意写作  
**关键词**: 创意写作, 无参考强化学习, 偏好优化, 生成式奖励模型, 元认知

## 一句话总结

本文提出 UniCreative 框架，通过自适应约束偏好优化（ACPO）和自适应标准生成式奖励模型（AC-GenRM），在无需 SFT 和参考答案的条件下统一长文本（规划→写作）和短文本（直接生成）两种创意写作模式，模型涌现出自主区分任务类型的元认知能力。

## 研究背景与动机

**领域现状**：LLM 在通用文本生成上表现出色，但创意写作仍受到两个根本性挑战的困扰：长文本（如小说、剧本）需要全局结构连贯性，容易出现主题漂移和重复；短文本（如诗歌、祝福语、广告文案）需要语言表达的灵动感，但自回归模型倾向于高概率"安全"token，导致输出平庸。

**现有痛点**：(1) 长文本仅增大上下文窗口无法解决结构退化问题，需要显式规划；(2) 短文本若强加规划机制反而会过度限定，扼杀创作中的灵感火花；(3) 现有对齐方法（RLHF/DPO）严重依赖高质量标注数据和参考答案，在开放式创作任务中成本极高且难以扩展。

**核心矛盾**：长文本的"近视症"（缺乏全局规划导致结构崩塌）与短文本的"过度确定"（过多结构约束压制表达多样性）是两种对立的失败模式，不能用统一的生成策略解决。

**本文目标**：构建一个统一框架，让模型自主判断何时需要规划、何时应直接生成，且不依赖任何人工标注的完成文本。

**切入角度**：将规划视为"可动态调用的计算资源"而非固定前置步骤，通过强化学习让模型学会在两种模式间自适应切换。

**核心 idea**：跳过 SFT 阶段，直接通过无参考的强化学习（ACPO）训练模型，利用自适应标准的生成式奖励模型（AC-GenRM）提供细粒度偏好信号。

## 方法详解

### 整体框架

UniCreative 包含两个核心组件：(1) AC-GenRM——根据查询语义动态生成评估标准并进行去偏成对排序的生成式奖励模型；(2) ACPO——基于 GRPO 的策略优化算法，综合内容质量奖励、结构范式约束和长度正则化三个信号来训练模型。训练时模型对每个查询生成 G 个响应，AC-GenRM 通过自举对比产生相对奖励信号。

### 关键设计

1. **AC-GenRM（自适应标准生成式奖励模型）**:

    - 功能：为每个创作查询动态生成评估标准并提供去偏的成对偏好判断
    - 核心思路：分为两步——(a) 动态标准合成：给定查询 $x$，模型 $\pi_{critic}$ 自动生成针对该查询的评估维度（如恐怖故事侧重"情节反转"和"氛围"，贺卡侧重"温暖"和"简洁"）；(b) 去偏成对排序：用对称数据增强（50% 概率交换响应顺序）训练，消除位置偏差，使偏好信号严格对齐动态标准。
    - 设计动机：静态评估标准无法捕捉不同创作类型的质量差异（悬疑小说和情诗的好坏维度完全不同），而 LLM 评委的位置偏差是公认问题。

2. **三维奖励组合**:

    - 功能：引导模型同时学习内容质量、结构选择和长度控制
    - 核心思路：总奖励 $R_{total} = R_{rel} + R_{struct} + R_{len}$。$R_{rel}$ 通过自举对比获得（组内自我竞争，赢+2 输-2）；$R_{struct}$ 是范式感知惩罚（长文本不用规划或短文本使用规划时扣 $\beta_s=5.0$ 分）；$R_{len}$ 是非对称长度正则化（长文本惩罚过短，短文本惩罚过长，有上限 $\gamma=5.0$ 防止异常值产生过大梯度）。
    - 设计动机：单纯的内容质量奖励无法教会模型区分任务类型和控制输出长度，需要正交的结构约束和长度约束共同引导策略学习。

3. **ACPO 优化算法**:

    - 功能：在无 SFT、无参考答案的条件下直接优化策略
    - 核心思路：基于 GRPO（组相对策略优化），每个查询采样 G 个响应，计算组内归一化优势 $A_i$，使用裁剪重要性比率和 KL 散度约束进行策略更新。通过投影算子 $\phi$ 去除规划 token 后再送入 AC-GenRM 评估，确保奖励只反映最终生成内容的质量。
    - 设计动机：创意写作无唯一正确答案，传统 SFT 在此失效。GRPO 避免了训练不稳定的价值网络，特别适合长文本生成中的高方差梯度估计。

### 损失函数 / 训练策略

使用 GRPO 目标函数，最大化裁剪优势减去 KL 惩罚。训练 Qwen3 系列（1.7B、4B、8B）在 8 块 H800 GPU 上，直接从 thinking 模型开始 RL 训练，无需 SFT 中间步骤。

## 实验关键数据

### 主实验

| 模型 | WritingBench 均分 | Blessing 优秀率 |
|--------|------|------|
| Qwen3-8B-Thinking | 77.11 | 68.0% |
| Qwen3-8B-Thinking + RL | 82.42 | 93.6% |
| Qwen3-4B-Thinking + RL | 77.36 | 91.4% |
| Claude-Sonnet-3.7 | 78.48 | - |
| DeepSeek-R1-0528 | 83.22 | - |
| Claude-Sonnet-4.5 | - | 93.2% |

Qwen3-8B + RL 在 WritingBench 上接近 DeepSeek-R1-0528（83.22），Blessing 短文本上超越 Claude-Sonnet-4.5。

### 消融实验

| 配置 | WritingBench | Blessing | 说明 |
|------|---------|------|------|
| Qwen3-8B Base | 70.75 | 43.6% | 无思考无 RL |
| + Thinking | 77.11 | 68.0% | 思考模式 |
| + Thinking + RL | 82.42 | 93.6% | 完整方法，提升 +25.6% |

### 关键发现

- **RL 增益巨大**：仅 RL 训练（无 SFT）就能在 WritingBench 上带来 5-10 分提升，在 Blessing 上从 68% 提升到 93.6%。
- **AC-GenRM 对齐度高**：Qwen3-8B AC-GenRM 在 LitBench 上与专家判断的一致率达 0.807，在 Blessing 上达 0.994，超过 Claude-Sonnet-3.7（0.731）和 GPT-4.1（0.702）。
- **涌现元认知能力**：训练后的模型自主学会了在长文本任务中使用 Plan-then-Write 模式、在短文本任务中使用直接生成模式，无需显式的任务类型标签。
- **小模型也受益**：1.7B 模型通过 RL 从 64.2% 提升到 90.0%（Blessing），接近大模型水平。

## 亮点与洞察

- **无参考 RL 的可行性**：首次证明在创意写作领域可以完全跳过 SFT 阶段，仅用 RL + 自适应奖励模型就能达到或超越 SFT+RLHF 的效果，大幅降低数据标注成本。
- **结构范式惩罚的巧妙设计**：通过简单的二元惩罚（长文本不规划扣分/短文本规划扣分）就能教会模型自主选择认知模式，设计极其简洁但效果显著。
- **动态标准合成**：AC-GenRM 根据查询语义自动生成评估维度的思路，可推广到所有开放式生成任务的评估中。

## 局限与展望

- 评估仍依赖 LLM 评委（WritingBench 用 GPT-4o 评分），主观创作的自动评估本身是开放问题。
- 仅在 Qwen3 系列上验证，未测试在其他基座模型上的泛化性。
- 任务类型分类（长/短）较为粗粒度，中等长度任务（如邮件、报告）的最优策略不确定。
- 结构范式惩罚需要预定义任务类型标签（Long/Short），限制了完全自主的模式选择。

## 相关工作与启发

- **vs Writing-Zero/LongWriter-Zero**: 这些方法专注于长文本 RL 优化，UniCreative 统一了长短文本两种模式，且无需 SFT 阶段。
- **vs DPO/RLHF**: 传统方法依赖参考答案和标注偏好对，UniCreative 通过自举对比完全免除参考依赖。
- **vs GRPO (DeepSeek-R1)**: UniCreative 在 GRPO 基础上增加了结构范式约束和自适应长度正则化，使其适用于创意写作场景。

## 评分

- 新颖性: ⭐⭐⭐⭐ 统一长短文本创作的框架设计和无参考 RL 训练思路有新意
- 实验充分度: ⭐⭐⭐⭐ 多尺寸模型对比，WritingBench 和 Blessing 双基准评测充分
- 写作质量: ⭐⭐⭐⭐ 问题动机和方法设计清晰
- 价值: ⭐⭐⭐⭐ 为开放式创作任务的 RL 优化提供了实用范式

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form](../../ICLR2026/reinforcement_learning/safe_continuous-time_multi-agent_reinforcement_learning_via_epigraph_form.md)
- [\[ACL 2026\] Frame of Reference: Addressing the Challenges of Common Ground Representation in Dialogue](frame_of_reference_addressing_the_challenges_of_common_ground_representation_in_.md)
- [\[ICLR 2026\] LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](../../ICLR2026/reinforcement_learning/loongrl_rl_for_reasoning_long_contexts.md)
- [\[AAAI 2026\] Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning](../../AAAI2026/reinforcement_learning/do_it_for_her_first-order_temporal_logic_reward_specification_in_reinforcement_l.md)
- [\[ICLR 2026\] LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning](../../ICLR2026/reinforcement_learning/longwriter-zero_mastering_ultra-long_text_generation_via_reinforcement_learning.md)

<!-- RELATED:END -->
