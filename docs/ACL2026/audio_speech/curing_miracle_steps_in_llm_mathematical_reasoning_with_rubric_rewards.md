---
title: >-
  [论文解读] Curing "Miracle Steps" in LLM Mathematical Reasoning with Rubric Rewards
description: >-
  [ACL 2026][语音][数学推理] 本文发现当前 LLM 数学推理中存在大量"Miracle Steps"——推理链中凭空跳跃到正确答案的现象，并提出 Rubric Reward Model (RRM)，一种基于问题特定评分标准的过程奖励函数，在 RL 训练中显著减少 Miracle Steps 71% 并将 AIME2024 的 Verified Pass@1024 从 26.7% 提升至 62.6%。
tags:
  - ACL 2026
  - 语音
  - 数学推理
  - Miracle Steps
  - 奖励黑客
  - 过程奖励
  - Rubric奖励
---

# Curing "Miracle Steps" in LLM Mathematical Reasoning with Rubric Rewards

**会议**: ACL 2026  
**arXiv**: [2510.07774](https://arxiv.org/abs/2510.07774)  
**代码**: [https://github.com/YouliangYuan/rrm-cure-miracle-steps](https://github.com/YouliangYuan/rrm-cure-miracle-steps)  
**领域**: LLM 推理 / 数学推理  
**关键词**: 数学推理, Miracle Steps, 奖励黑客, 过程奖励, Rubric奖励

## 一句话总结

本文发现当前 LLM 数学推理中存在大量"Miracle Steps"——推理链中凭空跳跃到正确答案的现象，并提出 Rubric Reward Model (RRM)，一种基于问题特定评分标准的过程奖励函数，在 RL 训练中显著减少 Miracle Steps 71% 并将 AIME2024 的 Verified Pass@1024 从 26.7% 提升至 62.6%。

## 研究背景与动机

**领域现状**：基于结果奖励的 RL 训练（如 GRPO+二元通过/失败信号）已成为提升 LLM 数学推理能力的主流方法。模型在标准 Pass@N 指标上表现出色。

**现有痛点**：(1) 结果奖励容易被"奖励黑客"——模型生成的解决方案虽然得到正确答案，但推理过程中存在逻辑缺陷（"假阳性"）；(2) "Miracle Steps"是最常见的失败模式——推理链中突然跳到正确答案，没有有效的推导过程；(3) 标准 Pass@N 大幅高估了模型的真实推理能力。

**核心矛盾**：结果奖励仅验证最终答案，无法区分"正确推理得到正确答案"和"错误推理碰巧得到正确答案"。模型学会了利用预训练中记忆的答案来绕过严格推理——即"答案回忆捷径"。

**本文目标**：(1) 系统分析和分类数学推理中的假阳性模式；(2) 设计过程级奖励函数来惩罚逻辑缺陷、鼓励严格推导；(3) 在 RL 训练中验证过程奖励的效果。

**切入角度**：引入"Verified Pass@N"指标（人工验证推理过程的正确性），揭示标准 Pass@N 与真实推理能力的巨大差距，然后针对性设计过程奖励。

**核心 idea**：奖励推理过程而非仅奖励结果——通过问题特定的评分标准（rubric）评估整个推理轨迹的逻辑严密性。

## 方法详解

### 整体框架

RRM 集成到标准 RL 管道中：(1) 对每个数学问题生成问题特定的评分标准（rubric），列出关键推理步骤和逻辑检查点；(2) 评估模型生成的推理链是否符合 rubric 的要求；(3) 将过程评分作为奖励信号替代或补充结果奖励进行 RL 训练。

### 关键设计

1. **Miracle Steps 分类体系**:

    - 功能：系统分析假阳性推理的失败模式
    - 核心思路：通过人工验证建立分类：(a) Miracle Steps——推理链中凭空跳到正确答案；(b) 计算错误恰好抵消；(c) 错误假设碰巧成立等。探测实验表明 Miracle Steps 与"答案回忆捷径"有关——模型独立于推理链直接从预训练记忆中提取答案
    - 设计动机：理解失败模式是设计有效对策的前提

2. **Rubric Reward Model (RRM)**:

    - 功能：评估整个推理轨迹的逻辑严密性
    - 核心思路：为每个问题生成特定的评分标准（rubric），包含关键推理步骤、逻辑检查点和常见错误警示。评估模型的推理链时检查其是否遵循了正确的推理路径，显式惩罚逻辑跳跃和无效推导
    - 设计动机：通用的过程奖励模型（PRM）无法捕捉问题特定的推理结构，rubric 提供了问题级别的细粒度评估

3. **RL 训练集成**:

    - 功能：将过程奖励替代结果奖励用于 RL 优化
    - 核心思路：用 RRM 替代二元通过/失败奖励，RRM 输出的是对推理过程质量的连续评分。这迫使模型不能仅靠"碰巧正确"获得奖励，必须展示严格的推理过程
    - 设计动机：结果奖励给予所有正确答案相同的奖励，无论推理是否正确；RRM 区分高质量和低质量的正确答案

### 损失函数 / 训练策略

基于标准 RL 管道（GRPO），将奖励函数从二元结果奖励替换为 RRM 的过程评分。训练基于 Qwen3-4B-Base。

## 实验关键数据

### 主实验

**AIME2024 性能对比**

| 方法 | Standard Pass@1024 | Verified Pass@1024 |
|------|-------------------|-------------------|
| 结果奖励（基线） | 高 | 26.7% |
| **RRM 奖励** | 高 | **62.6%** |

### 消融实验

| 指标 | 结果奖励 | RRM 奖励 | 变化 |
|------|---------|---------|------|
| Miracle Steps 发生率 | 基线 | -71% | 大幅减少 |
| Verified Pass@1024 (AIME2024) | 26.7% | 62.6% | +135% |

### 关键发现

- Standard Pass@N 严重高估推理能力——标准 Pass@1024 与 Verified Pass@1024 之间存在巨大差距
- Miracle Steps 是最主要的假阳性模式，与预训练中的答案记忆捷径高度相关
- RRM 训练将 Miracle Steps 发生率降低 71%，说明过程奖励有效抑制了答案回忆捷径
- RRM 在四个数学基准上一致优于结果奖励，验证了"奖励过程而非结果"的核心理念
- 过程奖励训练的模型不仅减少假阳性，还提高了真实推理能力

## 亮点与洞察

- "Miracle Steps"概念精准命名了一个被广泛忽视的问题——LLM 数学推理中的"假装推理"
- Verified Pass@N 指标的引入为评估真实推理能力提供了必要工具
- 揭示了 LLM 数学推理中"正确答案 ≠ 正确推理"的关键区别

## 局限与展望

- Rubric 生成本身依赖 LLM，可能存在质量问题
- RRM 评估成本高于简单的结果奖励
- 仅在数学推理上验证，在编程、逻辑等其他推理任务上的效果待确认
- Verified Pass@N 依赖人工验证，规模化困难

## 相关工作与启发

- **vs PRM (Process Reward Model)**: PRM 通用但不针对特定问题，RRM 生成问题特定的 rubric
- **vs 结果奖励 GRPO**: 结果奖励无法区分推理质量，RRM 显式评估推理过程
- **vs DeepSeek-R1**: R1 的长 CoT 也可能包含 Miracle Steps，RRM 提供了检测和修复的方法

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Miracle Steps 概念和 RRM 方法对数学推理 RL 有重要启示
- 实验充分度: ⭐⭐⭐⭐ 四个基准、人工验证、分类分析，但 Verified 评估规模有限
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，可视化直观，叙事引人入胜
- 价值: ⭐⭐⭐⭐⭐ 揭示了数学推理 RL 的关键漏洞并提供了有效解决方案

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](../../ICLR2026/audio_speech/reasoningbank_scaling_agent_self-evolving_with_reasoning_memory.md)
- [\[ACL 2026\] How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models](how_hypocritical_is_your_llm_judge_listener-speaker_asymmetries_in_the_pragmatic.md)
- [\[ICLR 2026\] Incentive-Aligned Multi-Source LLM Summaries](../../ICLR2026/audio_speech/incentive-aligned_multi-source_llm_summaries.md)
- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](../../ICLR2026/audio_speech/mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[ICLR 2026\] EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning](../../ICLR2026/audio_speech/emotionthinker_prosody-aware_reinforcement_learning_for_explainable_speech_emoti.md)

<!-- RELATED:END -->
