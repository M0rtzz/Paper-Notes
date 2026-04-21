---
title: >-
  [论文解读] $\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving
description: >-
  [ICLR 2026][RLVR] 本文提出 Re² 方法，通过纯强化学习训练 LLM 学会在推理过程中主动放弃无效思维链并重新开始求解，将罕见的 redo 行为从 0.5% 提升至 30% 以上，在相同训练计算预算下显著超越标准 RLVR 方法。
tags:
  - ICLR 2026
  - RLVR
  - LLM推理
  - 思维链优化
  - 重新求解
  - 过度思考
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
Re²（Reinforcement Learning with Re-solving）采用纯强化学习方法，不需要任何预先的监督微调（SFT）。整体流程为：输入一个数学/推理问题 → 模型生成包含可能多次重新求解的长推理链 → 通过可验证奖励评估最终答案 → 强化学习更新策略。

### 关键设计
1. **Re-solving 机制**: Re² 的核心在于让模型学会在推理过程中插入"重新求解"标记。当模型感知到当前推理方向可能有误时，它可以选择放弃当前的推理路径，从问题的起点重新开始思考。这一机制的设计动机来自于对 vanilla 模型的观察——这些模型中偶尔（约 0.5%）会自发出现 redo 行为，而这些罕见的 redo 实例往往与更好的推理结果相关。

2. **纯 RL 训练策略**: 与需要先收集 SFT 数据的方法不同，Re² 完全通过强化学习来放大模型中已存在但极为罕见的 redo 行为。训练过程中，当模型在某个 rollout 中自发采用了 re-solving 策略并最终得到正确答案时，这一行为会获得正向奖励，从而在后续训练中被强化。这种方式避免了人工设计 re-solving 格式的繁琐过程。

3. **渐进式行为放大**: 训练从 vanilla 模型的极低 redo 率（~0.5%）开始，通过持续的 RL 训练，模型逐渐学会更频繁地使用 re-solving 策略。最终，redo 行为比例可提升至 30% 以上。这一渐进过程是自然发生的，不需要特别的 curriculum 设计。

### 损失函数 / 训练策略
Re² 使用标准的 RLVR 训练框架，采用可验证奖励（verifiable rewards）作为信号。奖励函数检查模型最终答案的正确性，正确则给予正向奖励，错误则给予负向奖励。关键在于 Re² 不对模型输出的格式施加额外约束——模型可以自由选择是否执行 re-solving，而奖励信号会自然地引导模型在合适的时机采用这一策略。

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

## 相关论文

- [Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning](../../AAAI2026/reinforcement_learning/well_begun_half_done_reinforcement_learning_with_prefix_optimization_for_llm_rea.md)
- [Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving](helix_evolutionary_reinforcement_learning_for_open-ended_scientific_problem_solv.md)
- [Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)
- [Trinity: An Evolved LLM Coordinator](trinity_an_evolved_llm_coordinator.md)
- [Learning from Synthetic Data Improves Multi-hop Reasoning](learning_from_synthetic_data_improves_multi-hop_reasoning.md)

<!-- RELATED:END -->
