---
title: >-
  [论文解读] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs
description: >-
  [ICLR 2026][LLM推理][测试时缩放] 提出 Plan-and-Budget 框架，通过将复杂查询分解为子问题并基于估计复杂度自适应分配 token 预算，实现推理 LLM 的高效测试时缩放——最高提升 70% 准确率、减少 39% token、E3 指标提升 193.8%。
tags:
  - "ICLR 2026"
  - "LLM推理"
  - "测试时缩放"
  - "推理效率"
  - "过度思考"
  - "token预算分配"
  - "推理LLM"
---

# Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs

**会议**: ICLR 2026  
**arXiv**: [2505.16122](https://arxiv.org/abs/2505.16122)  
**代码**: [github.com/junhongmit/P-and-B](https://github.com/junhongmit/P-and-B)  
**领域**: LLM推理  
**关键词**: 测试时缩放, 推理效率, 过度思考, token预算分配, 推理LLM

## 一句话总结

提出 Plan-and-Budget 框架，通过将复杂查询分解为子问题并基于估计复杂度自适应分配 token 预算，实现推理 LLM 的高效测试时缩放——最高提升 70% 准确率、减少 39% token、E3 指标提升 193.8%。

## 研究背景与动机

推理型大语言模型（如 DeepSeek-R1、QwQ）在数学推理、代码生成等复杂任务中取得了显著成功，但推理阶段的计算效率问题日益突出：

**Overthinking（过度思考）**：许多主流 LLM 即使面对简单查询也会生成冗长、离题的推理链。模型"想得太多"，产生了大量不必要的中间推理步骤，浪费了计算资源。

**固定预算的局面**：近期工作试图通过限制固定 token 预算来缓解 overthinking，但这种"一刀切"的策略会导致 **underthinking（思考不足）**——对于困难问题，固定预算可能不够用，导致推理不充分。

**问题难度异质性**：现实中的查询复杂度变化很大。一个简单的算术题和一个复杂的多步推理问题需要截然不同的计算资源，但现有方法缺乏合理的资源分配机制。

**缺乏理论基础**：关于如何最优地分配推理计算资源，缺乏形式化的理论框架。

作者通过经验分析发现，**推理效率低下通常源于不清晰的问题解决策略**——模型在没有明确计划的情况下就开始推理，容易偏离方向。

## 方法详解

### 整体框架

这篇论文要解决的是推理 LLM「想得没章法」的问题：模型在没有计划的情况下边想边偏，简单题想得过多（overthinking）、难题又被一刀切的固定预算卡住（underthinking）。Plan-and-Budget 的思路是先让模型把题目拆成有结构的子问题，再按每个子问题的难度分配 token，把计算花在刀刃上。

整个框架是模型无关的纯推理时方法，输入一道查询后分三步走：先 **Plan**，让 LLM 把原始查询分解成一串子问题，给出明确的解题计划；再 **Estimate**，逐个估计每个子问题的复杂度；最后 **Budget**，根据复杂度把总 token 预算自适应地摊到各子问题上，难的多给、简单的少给，最终拼成完整解答。

### 关键设计

**1. BAM 理论模型（Budget Allocation Model）：为「难题多想、简题少想」提供数学依据**

直觉上我们都知道该把算力倾斜给难题，但缺一个严格的论证。BAM 把推理过程形式化为一串带不确定性的子问题序列：每个子问题 $q_i$ 配一个不确定性参数 $u_i$，解决它所需的 token 数与 $u_i$ 成正比。在这个模型下论文分析了不同预算分配策略的最优性，并证明了**自适应分配优于均匀分配**——把更多预算给不确定性更高（更难）的子问题，总体效率最优。这就把后面 Budget 阶段的设计从经验直觉抬升成了有理论支撑的结论。

**2. E3 评估指标（Effective and Efficient Evaluation）：用一个数同时量正确性和省不省**

现有评估要么只看准确率、要么只数 token，无法刻画两者的权衡，于是论文定义了 E3 指标，把准确率和归一化后的 token 消耗放进同一个比值里：

$$E3 = \frac{\text{Accuracy}}{\text{Normalized Token Cost}}$$

一个模型若能用更短、更精准的推理链拿到同样的答案，分母变小、E3 就更高。它给社区提供了一把统一的尺子来衡量「又对又省」，也是论文里 +193.8% 提升所对应的指标。

**3. 子问题分解（Plan 阶段）：先定计划，再开始推理**

论文的经验分析指出，推理效率低下的根源往往是「没有清晰的解题策略」——模型一上来就想，容易越想越偏。Plan 阶段用 LLM 自身或一个轻量辅助模型，在正式推理前先把原始查询拆成多个子问题，分解粒度按原问题的结构自适应调整，每个子问题都是一个可独立求解的单元。有了这份计划当骨架，后续推理就有了方向，过度发散的 overthinking 被显著抑制。

**4. 自适应预算调度（Budget 阶段）：把 BAM 的最优分配落到 token 上**

这一步把 BAM 的理论结论变成可执行的调度：根据每个子问题的估计复杂度动态分配 token，复杂度可由子问题的长度、关键词、问题类型等特征估出，简单子问题给较少预算、复杂子问题给较多预算；总预算既可以设上限、也可以自动确定。具体实现中，模型在生成某个子问题的解时会被显式告知当前的 token 限制。正因为预算随难度走，这套调度同时躲开了固定大预算的浪费（overthinking）和固定小预算不够用（underthinking）两个坑。

### 损失函数 / 训练策略

Plan-and-Budget 是纯推理时方法，不需要任何训练或微调，全靠精心设计的 prompt 来引导 LLM 完成子问题分解和预算控制。它与模型架构无关，可直接套到任意推理型 LLM 上，也能和不同规模的模型自由组合。

## 实验关键数据

### 主实验

| 任务类型 | 模型 | 方法 | 准确率变化 | Token 变化 | E3 变化 |
|---------|------|------|-----------|-----------|---------|
| 数学推理 | DS-Qwen-32B | Plan-and-Budget | +70% ↑ | -39% ↓ | +193.8% ↑ |
| 数学推理 | DS-LLaMA-70B | Plan-and-Budget | 提升 | 减少 | 显著提升 |
| 复杂推理 | 多模型 | Plan-and-Budget | 一致提升 | 一致减少 | 全面改善 |

**跨模型规模的关键发现**：Plan-and-Budget 使较小模型（DS-Qwen-32B）的效率达到了较大模型（DS-LLaMA-70B）的水平，展现了无需重训练即可弥合性能差距的能力。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅 Plan（无预算控制） | 准确率提升，效率提升有限 | 分解本身就有帮助 |
| 仅 Budget（无分解） | 效率提升，准确率可能下降 | 缺乏结构化引导 |
| Plan + 均匀预算 | 中等改善 | 不如自适应分配 |
| Plan + 自适应预算 | 最优 | 完整框架效果最佳 |
| 不同分解粒度 | 中等粒度最优 | 过细增加开销，过粗失去意义 |

### 关键发现

1. **Plan 和 Budget 缺一不可**：计划分解解决了推理方向问题，预算分配解决了资源效率问题，二者协同效果最佳
2. **小模型+Plan-and-Budget ≈ 大模型**：框架可以有效弥补模型规模的差距
3. **自适应优于固定**：无论是固定的大预算还是固定的小预算，都不如自适应分配
4. **模型无关性**：框架在不同的推理 LLM 上均有效

## 亮点与洞察

1. **理论与实践的优美结合**：先建立 BAM 理论模型，从理论推导出自适应分配的最优性，再设计 Plan-and-Budget 框架——不是纯启发式的
2. **E3 指标的提出**：填补了推理效率评估的空白，为社区提供了统一的衡量标准
3. **Overthinking 的精确诊断**：通过经验分析定位到"缺乏策略"是 overthinking 的根源，而非模型能力不足
4. **小模型的效率提升路径**：用计算效率而非模型规模来提升性能，对资源有限的场景非常实用
5. **零训练成本**：纯 test-time 方法，开箱即用

## 局限与展望

1. **分解质量依赖 LLM 能力**：如果 LLM 自身的分解能力不足，Plan 阶段可能产生不合理的子问题，进而影响整体效果
2. **复杂度估计的准确性**：自适应预算分配的效果取决于复杂度估计的准确性，而这本身是一个困难问题
3. **额外的 prompt 开销**：Plan 阶段的分解和 Budget 阶段的引导需要额外的 prompt token，对于极短的简单查询可能得不偿失
4. **子问题间的依赖**：线性分解为独立子问题的假设可能过于简化——现实中子问题间可能有复杂的依赖关系
5. **与 RL-based 方法的结合**：Plan-and-Budget 可以与基于强化学习的推理优化方法结合，但论文未探索

## 相关工作与启发

- **Test-time scaling**：Self-Consistency、Tree-of-Thought、Best-of-N 等测试时方法——Plan-and-Budget 与这些方法互补
- **Overthinking 研究**：STILL、S1 等关注推理冗余的先驱工作
- **Budget-aware推理**：token 预算约束、early stopping 等相关技术
- **启发**：BAM 理论模型的思路可以推广到其他需要资源分配的场景（如多模态推理、多工具调用）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 理论模型有贡献，但 Plan-and-Budget 的工程实现较直观
- **实验充分度**: ⭐⭐⭐⭐⭐ — 多模型多任务全面评估，消融完善，结果有说服力
- **写作质量**: ⭐⭐⭐⭐ — 理论与实验结合良好
- **价值**: ⭐⭐⭐⭐⭐ — 解决了推理 LLM 的实际效率痛点，即插即用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](atts_asynchronous_test-time_scaling_via_conformal_prediction.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](../../ACL2026/llm_reasoning/efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[NeurIPS 2025\] LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](../../NeurIPS2025/llm_reasoning/limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)

</div>

<!-- RELATED:END -->
