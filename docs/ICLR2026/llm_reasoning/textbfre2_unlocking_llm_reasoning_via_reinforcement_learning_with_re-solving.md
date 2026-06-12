---
title: >-
  [论文解读] $\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving
description: >-
  [ICLR 2026][LLM推理][RLVR] 本文提出 Re² 方法，通过纯强化学习训练 LLM 学会在推理过程中主动放弃无效思维链并重新开始求解，将罕见的 redo 行为从 0.5% 提升至 30% 以上，在相同训练计算预算下显著超越标准 RLVR 方法。
tags:
  - "ICLR 2026"
  - "LLM推理"
  - "RLVR"
  - "思维链优化"
  - "重新求解"
  - "过度思考"
---

# $\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving

**会议**: ICLR 2026  
**arXiv**: [2603.07197](https://arxiv.org/abs/2603.07197)  
**代码**: 无  
**领域**: Reinforcement Learning  
**关键词**: RLVR, LLM推理, 思维链优化, 重新求解, 过度思考

## 一句话总结
本文提出 Re² 方法，通过纯强化学习训练 LLM 学会在推理过程中主动放弃无效思维链并重新开始求解，将罕见的 redo 行为从 0.5% 提升至 30% 以上，在相同训练计算预算下显著超越标准 RLVR 方法。

## 研究背景与动机
大语言模型的推理能力可通过带有可验证奖励的强化学习（RLVR）来提升，这类方法通过增加测试时计算量来改善表现。然而，即便经过充分的 RLVR 训练，模型在生成思维链（Chain-of-Thought, CoT）时仍然容易产生不必要且低质量的推理步骤，导致"过度思考"（overthinking）问题，在消耗大量 token 的同时反而降低了最终答案的质量。

核心观察是：**当 CoT 的初始方向或质量不佳时，模型往往无法到达正确答案**，即使模型为此生成了比初始 CoT 质量良好时多出数倍的 token。这揭示了一个关键问题——标准 RLVR 训练的模型缺乏"及时止损"和"重新开始"的能力，它们总是执着于完成已经走偏的推理路径。

本文的核心 idea：教会 LLM 在推理过程中灵活地放弃不productive的推理路径，并在必要时重新开始求解过程，而非总是固守到最终答案。

## 方法详解

### 整体框架
Re²（Reinforcement Learning with Re-solving）是一套纯强化学习方案，不依赖任何预先的监督微调（SFT）。给定一个数学或推理问题，模型自回归生成一条可能包含多次"推倒重来"的长推理链，再用可验证奖励检查最终答案对错，并据此更新策略。整个过程不引入新的网络模块，唯一被改变的是模型对"什么时候该放弃当前思路重新开始"的概率分布。

### 关键设计

**1. Re-solving 机制：让模型敢于在中途推倒重来。** 标准 RLVR 训练出的模型有一个隐蔽的脆弱点——一旦思维链（CoT）的开头方向走偏，后续即便再堆几倍的 token 也很难自我纠正回到正确答案。Re² 的对策不是局部修补，而是允许模型在感知到当前路径不productive 时直接放弃它、回到问题原点重新求解。这一灵感来自对 vanilla 模型 rollout 的统计观察：这类自发的 redo 行为虽然只占约 0.5% 的样本，却往往伴随更好的推理结果。Re² 把这个被埋没的少数行为当作可被放大的"金矿"，而不是去人工设计一套重新求解的格式模板。

**2. 纯 RL 放大稀有行为：用奖励信号自然引导，而非 SFT 模仿。** 既然有益的 redo 行为已经潜藏在模型分布里，Re² 就完全靠强化学习把它的概率拉高，省去了收集 SFT 数据并定义 re-solving 标注格式的繁琐环节。机制很直接：当某条 rollout 自发采用了重新求解并最终答对，可验证奖励就给它正向信号，这条轨迹上的 re-solving 行为在后续训练中被强化；答错则给负向信号。奖励函数只看最终答案是否正确，不对输出格式施加任何额外约束，因此模型可以自由决定在哪一步、是否要重新开始，由奖励信号把这一选择推向合适的时机。

**3. 渐进式行为涌现：redo 率从 0.5% 自然爬升到 30%+。** 训练从 vanilla 模型极低的 redo 率（约 0.5%）起步，随着 RL 持续进行，模型越来越频繁地调用重新求解策略，最终 redo 行为占比可超过 30%。值得注意的是这一爬升是自发的，无需专门设计 curriculum 或分阶段调度——奖励信号本身就足以让模型在反复试错中把这条原本罕见的路径变成常规手段。

### 损失函数 / 训练策略
Re² 复用标准 RLVR 训练框架，以可验证奖励作为唯一信号：答案正确给正向奖励，错误给负向奖励。整套训练的精髓在于"少即是多"——不增加任何关于 re-solving 的格式约束或辅助损失，只让答案正确性这一信号去间接塑造模型的重新求解习惯，从而把方法的复杂度压到最低，同时保持与现有 RLVR 流水线的兼容。

## 实验关键数据

### 主实验

| 数据集 | 指标 | Re² | 标准 RLVR | 提升 |
|--------|------|-----|-----------|------|
| 数学推理基准 | 准确率 | 显著优于基线 | 基线 | 大幅提升 |
| 同等训练计算预算 | Pass@1 | 更高 | 较低 | 一致性提升 |
| 测试时扩展 | 多样本采样 | 随样本数增加且表现持续提升 | 提升放缓 | 更好的 scaling 行为 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Vanilla 模型 | redo 率 ~0.5% | 基线中 redo 行为极为罕见 |
| Re² 训练后 | redo 率 >30% | 成功放大了 redo 行为 |
| 有 SFT 预训练 | 对比 | Re² 的纯 RL 方式效果更优 |
| 不同计算预算 | 收敛曲线 | Re² 在相同预算下性能更优 |

### 关键发现
- 当 CoT 初始方向不佳时，即使模型生成数倍于正常长度的 token，也难以纠正错误，证明了 re-solving 的必要性
- 纯 RL 方法足以将 redo 率从 0.5% 提升至 30%+，无需 SFT 数据
- Re² 在测试时表现出更好的 scaling 行为：随着采样数量增加，性能持续提升
- Re-solving 不仅提升了准确率，还提升了推理效率（减少了无效 token 生成）

## 亮点与洞察
- **简洁而有效的设计理念**: 不是设计更复杂的推理结构，而是赋予模型"重头再来"的能力，这与人类解题时的自然行为一致
- **纯 RL 训练无需 SFT 数据**: 证明了仅通过强化学习就能从模型中挖掘和放大有益的推理模式，这为未来的 LLM 训练提供了新的思路
- **对 overthinking 问题的深入分析**: 清晰地揭示了标准 RLVR 模型在 CoT 初始方向不佳时的脆弱性
- **测试时计算效率**: Re² 不仅提升了 pass@1，在需要多次采样的 pass@k 设置下也表现出色，说明该方法生成的多条推理路径更加多样化

## 局限与展望
- 论文主要关注数学推理任务，在代码生成、逻辑推理等其他推理领域的效果有待验证
- Re-solving 机制增加了模型的平均输出长度，在推理延迟敏感的场景中可能不够理想
- 何时触发 re-solving 的决策完全由模型隐式学习，缺乏显式的触发条件分析
- 对于简单问题，re-solving 机制可能带来不必要的计算开销
- 能否与更先进的 CoT 优化方法（如 tree-of-thought）结合使用值得探索

## 相关工作与启发
- **RLVR 方法系列**: 如 DeepSeek-R1 等工作通过可验证奖励提升 LLM 推理能力，Re² 在此基础上解决了 overthinking 问题
- **CoT 优化**: 与 self-reflection、backtracking 等方法不同，Re² 采用更彻底的"重新开始"策略而非局部修正
- **测试时计算优化**: Re² 在测试时的表现暗示了 re-solving 对样本多样性的正面影响，与 best-of-N 采样策略有协同效应
- **启发**: 在 RL 训练中，模型自身蕴含的罕见但有益的行为模式可以被有效放大，这一思路可能推广到其他领域

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning](stabilizing_policy_gradients_for_sample-efficient_reinforcement_learning_in_llm_.md)
- [\[ICML 2026\] When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning](../../ICML2026/llm_reasoning/when_to_re-plan_subgoal_persistence_in_hierarchical_latent_reasoning.md)
- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[AAAI 2026\] Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning](../../AAAI2026/llm_reasoning/well_begun_half_done_reinforcement_learning_with_prefix_optimization_for_llm_rea.md)
- [\[NeurIPS 2025\] Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning](../../NeurIPS2025/llm_reasoning/re-forc_adaptive_reward_prediction_for_efficient_chain-of-thought_reasoning.md)

</div>

<!-- RELATED:END -->
