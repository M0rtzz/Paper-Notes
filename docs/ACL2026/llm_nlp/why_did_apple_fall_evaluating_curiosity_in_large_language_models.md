---
title: >-
  [论文解读] Why Did Apple Fall: Evaluating Curiosity in Large Language Models
description: >-
  [ACL 2026][LLM/NLP][好奇心] 本文提出首个系统评估 LLM 好奇心行为的心理学启发框架，结合问卷自评和行为实验发现 LLM 展现出好奇心般的行为模式但并非内在特质，并设计好奇心驱动的提问管道证明模拟好奇行为可提升下游推理性能。
tags:
  - ACL 2026
  - LLM/NLP
  - 好奇心
  - LLM行为评估
  - 心理学量表
  - 行为实验
  - 推理增强
---

# Why Did Apple Fall: Evaluating Curiosity in Large Language Models

**会议**: ACL 2026  
**arXiv**: [2510.20635](https://arxiv.org/abs/2510.20635)  
**代码**: [https://github.com/Yukijudaii1352/CuriosityEval](https://github.com/Yukijudaii1352/CuriosityEval)  
**领域**: LLM 评估 / 认知科学  
**关键词**: 好奇心, LLM行为评估, 心理学量表, 行为实验, 推理增强

## 一句话总结

本文提出首个系统评估 LLM 好奇心行为的心理学启发框架，结合问卷自评和行为实验发现 LLM 展现出好奇心般的行为模式但并非内在特质，并设计好奇心驱动的提问管道证明模拟好奇行为可提升下游推理性能。

## 研究背景与动机

**领域现状**：好奇心驱动的强化学习（如 i-MENTOR、CDE）通过内在奖励信号引导 LLM 探索，已在数学和编程任务上展现潜力。然而，这些方法是否真正反映了 LLM 的好奇心行为、心理学意义上的好奇心概念能否迁移到 LLM，尚不清楚。

**现有痛点**：(1) 未充分评估 LLM 是否能展现类似好奇心的行为特征；(2) 现有方法依赖熵或困惑度等统计信号，难以区分改进来自增强的监督信号还是真正的好奇行为；(3) 缺乏系统化的评估框架。

**核心矛盾**：好奇心驱动的 RL 方法假设 LLM 的好奇心可以被激发和增强，但我们甚至不知道 LLM 是否"拥有"好奇心。

**本文目标**：(1) 用心理学量表和行为实验系统评估 LLM 的好奇心行为；(2) 区分好奇心是内在特质还是行为模式；(3) 探索好奇行为能否提升下游性能。

**切入角度**：改编五维好奇心量表修订版（5DCR），将人类好奇心的三个维度（信息寻求、刺激寻求、社交好奇）分别设计问卷评估和行为任务，实现从"自我报告"到"行为验证"的闭环评估。

**核心 idea**：LLM 展现出好奇心般的行为模式，但这更像是拟合人类数据和安全约束的产物而非内在驱动力；不过，即使是纯行为层面的好奇模拟也能提升推理性能。

## 方法详解

### 整体框架

四阶段评估框架：(A) 建立好奇心分类体系（5DCR → 信息寻求/刺激寻求/社交好奇）；(B) 问卷自评——LLM 用 7 点量表回答 24 题；(C) 行为实验——每个维度设计对应的决策任务验证问卷结果；(D) 好奇心驱动学习——设计 CoQ 提问管道，测试好奇行为的功能价值。

### 关键设计

1. **问卷+行为的双重评估**:

    - 功能：分别从自省和行为层面评估好奇心
    - 核心思路：问卷层面用 5DCR 的 24 题，计算 Cohen's d（与人类的标准化差异）和 McDonald's Omega（内部一致性）。行为层面设计三个实验：信息寻求用缺字游戏（模型填字后是否选择看答案）、刺激寻求用潜水艇游戏（选确定/不确定窗口）、社交好奇用对话实验（与虚拟陌生人对话中的提问频率）
    - 设计动机：问卷自评可能受幻觉人格影响，行为实验提供更可靠的行为证据

2. **好奇心驱动的提问管道（CoQ）**:

    - 功能：测试好奇行为是否对推理有功能价值
    - 核心思路：设计三种 prompt——Vanilla CoT（标准思维链）、Refined CoT（含反思和回溯）、Curious CoQ（鼓励自问自答，如"如果...会怎样"、"为什么"、"怎么做"）。同时在 SFT+RLVR 管道中对比三种思维过程的训练效果
    - 设计动机：如果好奇行为有功能价值，那么即使 LLM 没有内在好奇心，模拟好奇策略也应该有帮助

3. **行为-内在特质区分**:

    - 功能：判断 LLM 的好奇心是行为模式还是内在特质
    - 核心思路：分析好奇行为在不同 prompt、不同上下文中的稳定性。如果是内在特质应具有跨上下文一致性，如果是行为模式则应对上下文高度敏感
    - 设计动机：这一区分对理解好奇心驱动 RL 的理论基础至关重要

### 损失函数 / 训练策略

SFT 阶段用标准语言建模损失，RLVR 阶段用 GRPO，仅使用格式奖励和正确性奖励（二元奖励）。

## 实验关键数据

### 主实验

**问卷自评（7 点量表，越高越好奇）**

| 模型 | 信息寻求 | 刺激寻求 | 社交好奇 |
|------|---------|---------|---------|
| GPT-4o | 6.58 | 4.71 | 6.25 |
| DeepSeek-V3.1 | 7.00 | 4.38 | 6.01 |
| Gemini-2.5 | 6.08 | 1.58 | 4.88 |
| 人类平均 | 5.03 | 4.93 | 4.86 |

### 消融实验

| 配置 | 推理任务性能 | 说明 |
|------|------------|------|
| Vanilla CoT | 基线 | 标准思维链 |
| Refined CoT | 提升 | 反思和回溯有帮助 |
| **Curious CoQ** | **最优** | 好奇提问进一步提升 |

### 关键发现

- LLM 展现出**不对称的好奇模式**：信息寻求维度很强但刺激寻求维度很弱，这与安全训练（RLHF）压制冒险行为一致
- 好奇行为**高度上下文敏感、跨 prompt 不稳定**——更像是拟合人类数据的产物而非内在特质
- 问卷自评和行为实验**大致一致**，说明心理学工具可以用于系统评估 LLM 行为
- **Curious CoQ 在下游任务上优于 Vanilla CoT 和 Refined CoT**——模拟好奇提问确实能产生更高质量的中间思维
- SFT+RLVR 管道中，CoQ 训练数据也优于 CoT 训练数据

## 亮点与洞察

- "LLM 有好奇行为但无好奇特质"的区分非常精准——为好奇心驱动 RL 的理论基础提供了重要澄清
- 三个行为实验的设计巧妙地从心理学范式改编：缺字游戏、潜水艇游戏、社交对话，每个都有明确的行为代理指标
- CoQ 的实用价值：即使好奇心不是内在特质，模拟好奇策略也能提升性能——这是一个实践层面的重要发现

## 局限与展望

- 行为实验的任务设计较简单，可能无法完全捕捉好奇心的复杂性
- CoQ 的效果可能部分来自更多的"思维量"而非好奇性本身——需要更精细的控制实验
- 仅在推理任务上评估 CoQ，创造性任务（好奇心可能更重要的场景）未覆盖
- 好奇心量表的文化偏向性（基于西方心理学模型）可能影响跨文化适用性

## 相关工作与启发

- **vs i-MENTOR/CDE**: 这些方法用内在奖励增强好奇心，本文用行为实验评估并用 prompt 工程利用好奇行为
- **vs 人格评估工作**: 先前工作评估 LLM 的人格特质（如大五人格），本文首次评估好奇心

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个系统评估 LLM 好奇心的工作，跨学科创新突出
- 实验充分度: ⭐⭐⭐⭐ 问卷+行为+应用三层评估，但行为实验可更复杂
- 写作质量: ⭐⭐⭐⭐⭐ 叙事引人入胜，从爱因斯坦名言到牛顿苹果，学术与可读性兼顾
- 价值: ⭐⭐⭐⭐⭐ 对好奇心驱动 RL 的理论基础和 LLM 行为理解有重要贡献

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models](muldimif_a_multi-dimensional_constraint_framework_for_evaluating_and_improving_i.md)
- [\[ACL 2025\] ELI-Why: Evaluating the Pedagogical Utility of Language Model Explanations](../../ACL2025/llm_nlp/eli-why_evaluating_the_pedagogical_utility_of_language_model_explanations.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)
- [\[ACL 2026\] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)

<!-- RELATED:END -->
