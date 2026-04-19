---
title: >-
  [论文解读] DICE-Bench: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues
description: >-
  [ACL 2025][LLM/NLP][function-calling] 提出 DICE-Bench，一个面向多轮多方对话场景的函数调用评测基准，包含 1607 条高质量对话实例和量化信息分散度的 DICE-Score 指标，揭示当前 LLM 在复杂对话中工具调用能力的不足。
tags:
  - ACL 2025
  - LLM/NLP
  - function-calling
  - benchmark
  - multi-party dialogue
  - multi-round
  - tool-use evaluation
---

# DICE-Bench: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues

**会议**: ACL 2025  
**arXiv**: [2506.22853](https://arxiv.org/abs/2506.22853)  
**代码**: [snuhcc/DICE-Bench](https://github.com/snuhcc/DICE-Bench)  
**领域**: LLM / NLP  
**关键词**: function-calling, benchmark, multi-party dialogue, multi-round, tool-use evaluation  

## 一句话总结

提出 DICE-Bench，一个面向多轮多方对话场景的函数调用评测基准，包含 1607 条高质量对话实例和量化信息分散度的 DICE-Score 指标，揭示当前 LLM 在复杂对话中工具调用能力的不足。

## 研究背景与动机

- **现有问题**：已有的函数调用基准（如 APIBench、ToolLLM 等）大多聚焦于单轮交互场景，即所有 API 参数都出现在单条用户指令中，忽视了真实群聊中信息分散在多轮多人对话中的复杂性。
- **现实需求**：在实际应用中，虚拟助手需要在群聊场景中追踪多人多轮对话，从分散的上下文中聚合信息来完成 API 调用（如根据群聊讨论结果预订酒店和机票）。
- **评估缺口**：缺乏一个能定量衡量"工具相关信息在对话中分散程度"的指标，使得难以系统性评估 LLM 在真实场景中的函数调用能力。
- **本文方案**：构建 DICE-Bench 基准并提出 DICE-Score 指标，通过多智能体仿真生成多轮多方对话数据，系统评估 19 个 LLM 的工具调用能力。

## 方法详解

### 整体框架

DICE-Bench 的数据构建包含三个阶段：(1) **工具图构建**：从 TaskBench 和 ToolEyes 收集 124 个工具节点和 270 条有向边，建模工具间依赖关系；(2) **场景配置**：通过 DFS 采样工具链，配置对话类型（说服协商型/咨询信息型/争论型）、参与人数（2-4人）和独立 persona；(3) **对话生成**：使用多智能体系统模拟对话，由编排器控制发言顺序，迭代生成 N 轮对话。

### 关键设计

- **DICE-Score 指标**：定量衡量工具相关信息在对话中的分散程度。公式为 $\text{DICE}(S,T) = \frac{\min(|S_{\neq 0}|, T) \cdot \sqrt{|S| \cdot T}}{\sum_{i \in S} \ln(1 + \alpha \times S_i)}$，其中 $S$ 为各轮次提及工具信息的计数向量，$T$ 为需识别的不同工具项总数，$\alpha = e^2$ 控制重复惩罚。高分表示信息更分散、任务更难。
- **三阶段验证流水线**：Stage 1 使用 G-Eval（GPT-4o）按 6 维标准自动过滤低质量对话；Stage 2 通过规则过滤（如拒绝回复检测）；Stage 3 由人工按对话质量、功能集成和现实适用性三个维度共 15 个子标准评分，淘汰低分实例。
- **工具图依赖建模**：工具间的有向边显式编码了"上一轮工具输出作为下一轮参数"的跨轮依赖关系，保证多轮场景的真实性。

### 损失函数/评估指标

- 使用 **Exact Match (EM)** 作为主要评估指标，要求 LLM 同时准确预测函数名和所有参数值。

## 实验

### 主实验结果

| 模型 | Round 1 | Round 2 | Round 3 | Round 4 | 平均 |
|------|---------|---------|---------|---------|------|
| GPT-4o | 74.12 | 61.00 | 61.65 | 59.18 | 63.99 |
| Gemini 2 Flash | 74.47 | 59.45 | 59.40 | 58.73 | 63.01 |
| Phi4-15B | 71.29 | 57.06 | 58.02 | 56.44 | 60.70 |
| GLM4-9B-Chat | 58.24 | 47.55 | 47.24 | 46.03 | 49.76 |
| Qwen2.5-32B | 67.76 | 56.76 | 57.23 | 55.92 | 59.42 |
| ToolAce-8B | 2.47 | 0.66 | 0.33 | 0.51 | 0.99 |

### 消融实验

| 分析维度 | 关键发现 |
|----------|----------|
| DICE-Score vs 性能 | Pearson 相关系数 r ≈ -0.984，DICE-Score 越高性能越差 |
| 人类评估对齐 | Round 1→4 人类准确率从 80.5% 降至 49.3%，与 DICE-Score 1.42→5.36 强烈负相关 |
| 对话类型影响 | 争论型 (Eristic) 对话因立场频繁切换导致 EM 显著更低 |
| 工具专用模型 | ToolAce-8B、CALM-8B 等专用模型表现远不如通用对话模型 |

### 关键发现

1. 所有模型性能随轮次增加显著下降，Round 4 相比 Round 1 平均下降约 15 个百分点，说明多轮信息聚合是当前 LLM 的主要瓶颈。
2. 开源 15B 级别的 Phi4 可与闭源 GPT-4o 相当（平均分 60.7 vs 64.0），Qwen 2.5 的 128K 上下文窗口有助于长对话场景。
3. 专门针对单轮函数调用微调的模型在多方对话场景下性能极差（ToolAce-8B 仅约 1%），暗示单轮训练数据无法迁移到多轮多方场景。

## 亮点

- 首个覆盖多轮 + 多方的函数调用基准，填补了现有评测的重要空白。
- DICE-Score 指标与人类表现高度负相关（r ≈ -0.984），具有良好的可解释性和有效性。
- 严格的三阶段过滤（自动 + 规则 + 人工），从 1800 条中筛选出 1607 条高质量实例。

## 局限性

- Round 4 对话长度可能超出部分模型的 4K token 限制，无法测试所有模型。
- 部分模型虽然内容语义正确但输出格式不符 JSON 规范导致被判定为错误。
- 多智能体编排器（GPT-4o）在动态分配发言顺序方面能力有限，倾向于重复模式的轮流发言。
- 仅覆盖日常生活场景，缺乏法律、金融、医疗等专业领域工具。

## 相关工作

- **函数调用基准**：APIBench、ToolAlpaca、ToolLLM、API-Bank、MetaTool、TaskBench 等均聚焦单轮或单用户场景。
- **交互式对话系统**：LLM 集成虚拟助手的多轮多方交互研究仍处于早期阶段。
- **对话类型理论**：基于 Walton & Krabbe 的七类对话分类框架构建场景多样性。

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 实用性 | 4 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| 总评 | 4.0 |

<!-- RELATED:START -->

## 相关论文

- [From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment](from_neurons_to_semantics_evaluating_cross-linguistic_alignment_capabilities_of_.md)
- [SocialEval: Evaluating Social Intelligence of Large Language Models](socialeval_evaluating_social_intelligence_of_large_language_models.md)
- [ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)
- [Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models](gapo_multi_objective_alignment.md)
- [Multi-Attribute Steering of Language Models via Targeted Intervention](multi_attribute_steering.md)

<!-- RELATED:END -->
