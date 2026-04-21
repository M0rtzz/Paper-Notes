---
title: >-
  [论文解读] SMART: Self-Aware Agent for Tool Overuse Mitigation
description: >-
  [ACL 2025][LLM Agent][工具过度使用] 揭示 LLM Agent 中的"工具过度使用"现象（≥30% 的工具调用是不必要的），提出 SMART 元认知范式，通过在推理步骤中显式标注"知识驱动 vs 工具依赖"来训练 Agent 的自感知能力，7B 模型匹配 GPT-4o 水平。
tags:
  - ACL 2025
  - LLM Agent
  - 工具过度使用
  - 元认知
  - 自感知Agent
  - 参数化知识
  - 工具调用优化
---

# SMART: Self-Aware Agent for Tool Overuse Mitigation

**会议**: ACL 2025  
**arXiv**: [2502.11435](https://arxiv.org/abs/2502.11435)  
**代码**: https://github.com/qiancheng0/Open-SMARTAgent (有)  
**领域**: LLM Agent  
**关键词**: 工具过度使用, 元认知, 自感知Agent, 参数化知识, 工具调用优化

## 一句话总结
揭示 LLM Agent 中的"工具过度使用"现象（≥30% 的工具调用是不必要的），提出 SMART 元认知范式，通过在推理步骤中显式标注"知识驱动 vs 工具依赖"来训练 Agent 的自感知能力，7B 模型匹配 GPT-4o 水平。

## 研究背景与动机

**领域现状**：LLM Agent 系统通常配备外部工具（计算器、搜索引擎、API等），但使用策略多依赖启发式或手工规则。

**现有痛点**：LLM 经常在自身参数化知识已能解决问题时仍调用工具——作者定义为"工具过度使用"（Tool Overuse），过度使用率 $\alpha \cdot \beta$（$\alpha$ 为无需工具的问题比例，$\beta$ 为其中不必要调用工具的比例），在现有系统中超过 30%。

**核心矛盾**：Agent 缺乏元认知能力——无法评估自己的参数化知识是否足以解决当前问题，默认倾向于"有工具就用"。

**本文目标**：如何让 Agent "知道自己知道什么"，从而在知识充足时不调工具、确实需要时才调？

**切入角度**：构建 SMART-ER 数据集，在每个推理步骤标注"这步用知识还是工具"及理由，通过 SFT 训练 Agent 的自感知。

**核心 idea**：显式的元认知训练——让模型学会在推理中判断每一步是否需要工具。

## 方法详解

### 整体框架
构建 SMART-ER 数据集（Math/Time/Intention 三领域）→ 标注每步推理为知识驱动或工具依赖（含理由）→ SFT 训练 SMARTAgent → 推理时自主决定何时调工具。

### 关键设计

1. **工具过度使用的量化定义**:

    - 功能：定义 $\text{ToolOveruse} = \alpha \cdot \beta$
    - 核心思路：$\alpha$ = 可不用工具就能解决的问题比例；$\beta$ = 这些问题中 Agent 仍调用工具的比例
    - 设计动机：首次量化这个被忽视的问题，发现多个 SOTA Agent 过度使用率超 30%

2. **SMART-ER 数据集构建**:

    - 功能：为三个领域（数学/时间/意图理解）构建带元认知标注的推理数据
    - 核心思路：每步标注为 K（知识驱动）或 T（工具依赖），附带 justification 说明为什么需要/不需要工具。人工审核 5% 样本，95% 通过率
    - 设计动机：显式标注迫使模型学习"何时用工具"的判断能力

3. **SMARTAgent 训练**:

    - 功能：在 SMART-ER 上对开源 LLM 做 SFT
    - 核心思路：标准监督微调，模型学会在推理链中嵌入元认知判断
    - 设计动机：轻量高效，7B 模型就能学到有效的自感知能力

## 实验关键数据

### 主实验

| 模型 | 工具调用次数 | 准确率 | 领域 |
|------|-----------|--------|------|
| Llama-3.1-8B baseline | 1.93 | 51% | Math |
| SMARTAgent 8B | 0.88 | 54.75% | Math |
| 宏平均改进 | -24% 调用 | +37.1% 准确率 | Overall |

### 关键发现
- **工具用少了反而准确率更高**：减少 24% 调用，准确率提升 37.1%——过度使用工具实际上损害了性能
- **7B 模型匹配 70B 和 GPT-4o**：说明元认知训练的性价比极高
- **泛化到 OOD 数据**：在 GSM8K 和 MINTQA 上同样有效

## 亮点与洞察
- **首次定义和量化"工具过度使用"**：一个被广泛忽视但影响重大的问题
- **"少即是多"的反直觉发现**：限制工具使用反而提升性能，因为避免了工具引入的额外噪声
- **元认知作为 Agent 核心能力**：未来 Agent 不仅需要"会用工具"，更需要"知道何时不用"

## 局限与展望
- 仅覆盖三个领域，更复杂的多工具协作场景未验证
- 数据标注依赖 GPT-4 + 人工审核，成本不低
- 未探索工具过度使用的根因分析（是训练数据偏差还是模型缺陷？）

## 相关工作与启发
- **vs ReAct/Toolformer**: 它们教模型"如何用工具"，SMART 教模型"何时不用工具"
- **vs 元认知研究**: 将心理学中的元认知概念应用到 LLM Agent 设计

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题定义新颖，"工具过度使用"是重要但被忽视的问题
- 实验充分度: ⭐⭐⭐⭐ 三个领域 + OOD 泛化验证
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰
- 价值: ⭐⭐⭐⭐ 对 Agent 系统设计有实际指导意义

<!-- RELATED:START -->

## 相关论文

- [Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement](gödel_agent_a_self-referential_agent_framework_for_recursive_self-improvement.md)
- [SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents](../../ICLR2026/llm_agent/simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_llm_agents.md)
- [Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger](meco_metacognition_tool_use.md)
- [Agentic Knowledgeable Self-Awareness](agentic_knowledgeable_self-awareness.md)
- [Play2Prompt: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play](play2prompt_zero-shot_tool_instruction_optimization_for_llm_agents_via_tool_play.md)

<!-- RELATED:END -->
