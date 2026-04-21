---
title: >-
  [论文解读] The Challenge of Teaching Reasoning to LLMs Without RL or Distillation
description: >-
  [ICML 2025 (AI for Math Workshop)][Chain-of-Thought] 仅用 20 个来自推理模型 QwQ-32B-Preview 的长 CoT 样例轻量微调 Qwen2.5-32B 就能超越 72B 的数学指令模型，但用非推理模型或人工生成的 CoT 无法达到同等效果，表明推理 CoT 中存在难以复制的"潜在质量"。
tags:
  - ICML 2025 (AI for Math Workshop)
  - Chain-of-Thought
  - 推理能力
  - 微调
  - 长CoT
  - 人工标注
---

# The Challenge of Teaching Reasoning to LLMs Without RL or Distillation

**会议**: ICML 2025 (AI for Math Workshop)  
**arXiv**: [2507.09850](https://arxiv.org/abs/2507.09850)  
**代码**: 无（数据集已发布：nvidia/Nemotron-Math-HumanReasoning）  
**领域**: LLM推理 / 数据工程  
**关键词**: Chain-of-Thought, 推理能力, 微调, 长CoT, 人工标注

## 一句话总结

仅用 20 个来自推理模型 QwQ-32B-Preview 的长 CoT 样例轻量微调 Qwen2.5-32B 就能超越 72B 的数学指令模型，但用非推理模型或人工生成的 CoT 无法达到同等效果，表明推理 CoT 中存在难以复制的"潜在质量"。

## 研究背景与动机

**领域现状**：具备推理能力的语言模型（如 o1、DeepSeek-R1）通过生成长的 Chain-of-Thought（CoT）轨迹来解决复杂任务，取得了 SOTA 性能。现有方法主要通过两条路径获得推理能力：（1）RL 训练（如 GRPO、PPO），（2）从强模型蒸馏（distillation）。

**现有痛点**：RL 训练和蒸馏都有较高成本——RL 需要大量采样和训练计算，蒸馏需要访问强大的推理模型。一个自然的问题是：能否用更简单的方式——只用少量 CoT 示例或纯人工标注——来激活 base model 的推理能力？

**核心矛盾**：之前工作已经表明，即使是短 CoT prompting（无微调）也能改善推理。那为什么还需要 RL 或蒸馏？少量高质量 CoT 数据能否替代大规模 RL/蒸馏？如果不能，推理模型生成的 CoT 到底有什么"特殊之处"是人工无法复制的？

**本文目标**：探索用最简方式（少量 CoT + 轻量微调）激活推理能力的边界——推理模型的 CoT vs 非推理模型的 CoT vs 人工编写的 CoT，谁能激活推理？差距在哪？

**切入角度**：系统性实验研究——控制变量地比较不同 CoT 来源（推理模型、非推理模型、人工）对微调效果的影响，分析难度、多样性、长度等关键属性。

**核心 idea**：仅 20 个推理模型生成的长 CoT 样例就足以解锁 base model 的强推理能力，但这种效果来自推理 CoT 中某种难以复制的"潜在质量"——非推理模型和人工标注的 CoT 均无法匹配。

## 方法详解

### 整体框架

本文不是提出新算法，而是一项系统性的实验研究。核心流程：

1. **收集不同来源的 CoT 数据**：从 QwQ-32B-Preview（推理模型）、非推理模型、人工标注者获取长 CoT
2. **在 Qwen2.5-32B（base model）上进行轻量微调**：使用极少量（低至 20 个）CoT 样例
3. **在数学推理基准上评估**：分析不同 CoT 来源对最终推理能力的影响
4. **属性分析**：研究问题难度、多样性、答案长度等维度对效果的影响

### 关键设计

1. **推理模型 CoT 微调（Reasoning Model CoT）**:

    - **功能**：从 QwQ-32B-Preview 收集 20 个长 CoT 示例，用于微调 Qwen2.5-32B base model
    - **核心思路**：这些 CoT 包含深度思考、回溯、验证等推理过程，内容较长（通常数千 token）
    - **设计动机**：验证"少量高质量推理数据能否替代大规模训练"的假设。结果发现——20 个这样的样例就足以让 32B base model 超越 72B 的数学指令模型

2. **非推理模型 CoT（Non-Reasoning Model CoT）**:

    - **功能**：使用非推理模型（如标准 instruct 模型）生成的 CoT 进行相同的微调实验
    - **核心结果**：效果显著弱于推理模型 CoT，即使增加数据量和优化 prompt 也无法弥补差距
    - **设计动机**：排除"只要是长 CoT 就有效"的可能性，证明推理模型 CoT 的特殊性

3. **人工编写的 CoT（Human-Authored CoT）**:

    - **功能**：让人类标注者编写长 CoT，并通过 prompt engineering、多轮编辑（multi-pass editing）、结构化指导（structural guidance）来提升质量
    - **核心结果**：即使经过多轮精心编辑，人工 CoT 的微调效果仍不及推理模型 CoT
    - **设计动机**：探索人工标注能否作为推理数据的替代来源，以及理解推理 CoT 中人类难以模仿的成分

4. **CoT 属性分析**:

    - **功能**：系统分析影响微调效果的关键数据属性
    - **关键发现**：
        - **问题难度**：适当难度的问题比过易或过难的更有效
        - **多样性**：来自不同数学子领域的问题优于同质化问题
        - **答案长度**：较长的 CoT 通常效果更好，但并非越长越好
    - **设计动机**：为理解"什么是好的推理数据"提供定量分析框架

### 训练策略

- 基座模型：Qwen2.5-32B（base，非 instruct 版本）
- 微调方式：标准 SFT，极小数据量（20 条即可）
- 数据来源对比：QwQ-32B-Preview / 标准 instruct 模型 / 人工标注
- 评估：MATH、GSM8K、AIME 等数学推理基准

## 实验关键数据

### 主实验

| CoT 来源 | 数据量 | MATH | 对比 |
|----------|-------|------|------|
| QwQ-32B-Preview（推理模型） | 20 条 | 超越 Qwen2.5-Math-72B-Instruct | 基线 |
| 非推理模型 | 20 条 | 显著弱于推理模型 CoT | 差距明显 |
| 人工编写（初版） | 20 条 | 弱于推理模型 CoT | 不及预期 |
| 人工编写（多轮编辑） | 20 条 | 有所提升但仍不及推理模型 | 差距缩小但未消除 |
| Qwen2.5-Math-72B-Instruct | - | 基线 | 被 20 条推理 CoT 微调的 32B 超越 |

### 消融实验

| 属性变化 | 效果影响 | 说明 |
|---------|---------|------|
| 增加 CoT 数据量 | 边际递减 | 20 条以上收益减少 |
| 提升问题难度 | 适中最佳 | 太简单学不到，太难学不好 |
| 增加领域多样性 | 正面 | 覆盖更多数学子领域有帮助 |
| 增加 CoT 长度 | 正面但有上限 | 更长的推理过程通常更好 |
| 多轮人工编辑 | 有限提升 | 缩小但无法消除与推理模型 CoT 的差距 |
| 结构化指导 | 有限提升 | 格式优化的帮助有限 |

### 关键发现

1. **20 条数据的惊人效果**：仅用 20 个来自推理模型的长 CoT 样例微调 32B base model，就能超越专门训练的 72B 数学模型，这说明推理能力可能已经"潜伏"在 base model 中，只需少量高质量数据来激活
2. **推理 CoT 的不可替代性**：非推理模型和人工编写的 CoT 均无法匹配推理模型 CoT 的效果，即使数据量更大、经过精心编辑
3. **CoT 中的"潜在质量"**：推理模型生成的 CoT 包含某种难以人工复制的质量特征——可能是隐含的认知模式、错误纠正策略、或特定的推理节奏

## 亮点与洞察

- **颠覆性发现**：20 条数据就能激活强推理能力，挑战了"推理需要大规模 RL 训练"的主流观点
- **深层问题**：推理 CoT 中到底有什么"魔法"是人类标注者无法复制的？这个问题指向了对 LLM 推理机制更深层的理解
- **实用价值**：即使最终结论是"还是需要推理模型的 CoT"，20-shot 微调的效率也极高
- **数据集贡献**：发布了多轮次精修的人工 CoT 数据集（Nemotron-Math-HumanReasoning），为后续研究提供了宝贵资源

## 局限与展望

1. 仅在数学推理任务上验证，其他推理类型（代码、逻辑推理）的结论可能不同
2. 对"推理 CoT 中的潜在质量"缺乏更深入的机制解释——停留在现象层面
3. 人工标注的 CoT 质量受限于标注者的数学水平，专业数学研究者的标注可能更接近推理模型
4. 20 条样例的选择是否存在敏感性（sensitivity to specific examples）未充分探讨
5. 作为 Workshop 论文，实验规模和深度相比主会议论文有限

## 相关工作与启发

- **DeepSeek-R1-Zero/R1**：通过 RL 或 RL+蒸馏获得推理能力的代表工作
- **s1（Simple Test-Time Scaling）**：同样探索用少量数据激活推理能力的方向
- **STaR**：self-taught reasoning，通过自生成训练数据迭代提升推理
- **启发**：推理能力可能是 base model 的一种"涌现"潜能，少量正确的刺激就能唤醒——这对理解 LLM 内部推理机制提出了新问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 系统性地挑战了"推理需要大规模训练"的假设，发现 20-shot 微调足够
- 实验充分度: ⭐⭐⭐ 控制变量实验设计好，但作为 Workshop 论文深度有限
- 写作质量: ⭐⭐⭐⭐ 问题驱动的写作方式清晰，实验分析有条理
- 价值: ⭐⭐⭐⭐ 提出了重要的科学问题（推理 CoT 的不可替代性），对后续研究有启发

<!-- RELATED:START -->

## 相关论文

- [Zero-Shot Generalization of Vision-Based RL Without Data Augmentation](zero-shot_generalization_of_vision-based_rl_without_data_augmentation.md)
- [Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples](flow_of_reasoning_training_llms_for_divergent_reasoning_with_minimal_examples.md)
- [Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/router-r1_teaching_llms_multi-round_routing_and_aggregation_via_reinforcement_le.md)
- [VinePPO: Refining Credit Assignment in RL Training of LLMs](vineppo_refining_credit_assignment_in_rl_training_of_llms.md)
- [T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling](t1_advancing_language_model_reasoning_through_reinforcement_learning_and_inferen.md)

<!-- RELATED:END -->
