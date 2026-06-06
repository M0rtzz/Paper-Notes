---
title: >-
  [论文解读] EscapeBench: Towards Advancing Creative Intelligence of Language Model Agents
description: >-
  [ACL 2025 (Long Paper)][LLM/NLP][creative intelligence] 本文推出 EscapeBench——基于密室逃脱游戏的 LLM Agent 创意智能评测基准（36 个场景、3 个难度），揭示当前模型在创造性工具使用和隐式目标推断上的严重不足…
tags:
  - "ACL 2025 (Long Paper)"
  - "LLM/NLP"
  - "creative intelligence"
  - "escape room"
  - "agent benchmark"
  - "tool use"
  - "reasoning"
---

# EscapeBench: Towards Advancing Creative Intelligence of Language Model Agents

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2412.13549](https://arxiv.org/abs/2412.13549)  
**代码**: [https://github.com/qiancheng0/EscapeBench](https://github.com/qiancheng0/EscapeBench)  
**领域**: LLM/NLP  
**关键词**: creative intelligence, escape room, agent benchmark, tool use, reasoning

## 一句话总结

本文推出 EscapeBench——基于密室逃脱游戏的 LLM Agent 创意智能评测基准（36 个场景、3 个难度），揭示当前模型在创造性工具使用和隐式目标推断上的严重不足，并提出 EscapeAgent（Foresight + Reflection）将提示依赖降低近 50%。

## 研究背景与动机

**领域现状**：LLM Agent 在长程规划和推理方面取得了显著进展，涌现了大量评测基准——从 Web 操作到科学研究、从文本游戏到 Minecraft 沙盒。这些基准评估的主要是分析智能（推理能力）和实践智能（知识应用能力），但创造性智能的评估严重缺失。

**现有痛点**：现有 Agent 的训练侧重于记忆工具-任务的标准关联，严重忽视了对工具可供性（affordance）的深层探索和在非结构化场景下的适应能力。当面对需要"跳出常规思维"的场景时（如将木棍创造性地用作撬棒），模型往往束手无策。

**核心矛盾**：Sternberg 的三元智力理论将智力分为实践、分析和创造三类。现有基准覆盖了前两类，但创造性智能——在新颖环境中进行创新性思维和适应性问题求解的能力——仍是评估盲区，缺乏专门的测试环境。

**本文目标** 1）缺乏专门评估 LLM Agent 创意智能的基准环境；2）需要量化当前模型在创造性工具使用和隐式目标发现方面的能力水平；3）如何增强 Agent 的创造性推理能力。

**切入角度**：密室逃脱游戏天然要求创造性工具使用、隐式目标推断和超长推理链，非常适合作为创意智能评测场景。一局游戏人类可能需要约 1 小时完成，全知 Agent 也至少需要 100+ 步操作和 40+ 个关键步骤。

**核心 idea**：以密室逃脱游戏为载体构建创意智能评测基准，并通过 Foresight（前瞻性工具使用假设）和 Reflection（任务列表动态管理）增强 Agent 创造力。

## 方法详解

### 整体框架

EscapeBench 基于自定义游戏引擎，包含场景（Scenes，图结构连接）、物品（Items，可交互对象）和工具（Tools，可收集使用或合成）三类核心组件。Agent 通过五种动作与环境交互。EscapeAgent 在 BaseAgent（工作记忆 + CoT 推理）基础上集成 Foresight 和 Reflection 模块。

### 关键设计

1. **游戏引擎与评测场景设计**:

    - 功能：模拟需要创造性推理的密室逃脱环境
    - 核心思路：Scenes 以图结构相连表示物理空间连通性；Items 需要工具应用（Apply）、输入（Input）或点击（Click）来触发状态变化或效果；Tools 可被收集后 Apply 到 Items 或两个 Tools Craft 合成新工具。五种动作——Move(Scene)、Click(Item)、Apply(Tool, Item)、Input(str, Item)、Craft(Tool, Tool)。其中 Apply 和 Craft 最考验创造力，要求 Agent 创新性地使用或合成工具
    - 设计动机：36 个手工标注的高质量场景（3 个难度 × 每难度 12 场景，每场景 3 个描述清晰度版本），确保了多样性和可控的难度梯度。每局游戏 Oracle 最优解平均 107.83 步

2. **Foresight 模块（前瞻推理）**:

    - 功能：增强创造性工具使用能力
    - 核心思路：在两种情况下激活——发现新任务时，Agent 根据已有工具假设潜在的工具使用策略并评估可行性；收集新工具时，Agent 评估该工具对解决现有任务的适用性或与其他工具合成的可能性。若提出可执行的假设则进入 "Try Action" 状态直接尝试，否则保持 "Free Explore" 模式自由探索
    - 设计动机：让 Agent 在行动前显式推理工具的非常规用途，仿照人类"先假设再验证"的创造性决策过程，避免无目的地反复试错

3. **Reflection 模块（反思管理）**:

    - 功能：管理隐式目标，防止重复失败，提升行动效率
    - 核心思路：维护一个结构化任务列表，支持三种操作——New（添加新发现的未解决任务）、Update（记录尝试但失败的动作以避免重复）、Delete（任务完成后移除）。每条记录包含任务名称、目标物品和失败动作列表。在每次非 Move 动作后根据环境反馈触发更新
    - 设计动机：密室逃脱中目标是隐式的、渐进发现的，Agent 需要主动管理已知和未知的子目标列表，防止在相同错误上反复浪费步数

### 损失函数 / 训练策略

EscapeAgent 是一个无需训练的框架，依赖 prompting 和运行时推理。核心设定：BaseAgent 使用 CoT 推理和长度为 10 的工作记忆；连续 50 步无进展时系统自动提供提示（hint）确保游戏最终可完成；采样温度 $T=0$, $n=1$。

## 实验关键数据

### 主实验

| 模型 | Hints Used ↓ | Total Steps ↓ | Early Exit ↑ | Key Steps Hints % ↓ |
|------|-------------|---------------|--------------|---------------------|
| GPT-4o | 10.30 | 723.61 | 24.75% | 24.27% |
| Claude-3.5-Sonnet | 8.97 | 690.31 | 28.95% | 22.44% |
| Llama-3.1-70B | 14.53 | 982.42 | 19.00% | 33.29% |
| Qwen2.5-72B | 16.50 | 1102.50 | 12.46% | 32.02% |
| Llama-3.1-8B | 25.86 | 1543.30 | 10.10% | 56.00% |
| Qwen2.5-7B | 32.20 | 1950.42 | 6.52% | 54.43% |
| **人类平均** | **4.33** | **257.83** | **59.65%** | **12.28%** |

### EscapeAgent 效果

| 模型 + Agent | Hints Used ↓ | Steps ↓ | Hints 降幅 | Steps 降幅 |
|-------------|-------------|---------|-----------|-----------|
| GPT-4o BaseAgent | 10.30 | 723.61 | — | — |
| GPT-4o **EscapeAgent** | **5.03** | **452.75** | **-5.27** | **-270.86** |
| Llama-70B BaseAgent | 14.53 | 982.42 | — | — |
| Llama-70B **EscapeAgent** | **7.92** | **645.19** | **-6.61** | **-337.23** |
| Qwen-72B BaseAgent | 16.50 | 1102.50 | — | — |
| Qwen-72B **EscapeAgent** | **9.72** | **746.61** | **-6.78** | **-355.89** |

### 关键发现

- **创意智能严重不足**：最好的 Claude-3.5 也需要约 9 个提示才能完成（人类仅需约 4 个），行动步数是 Oracle 最优解的 6-7 倍，Early Exit 仅 29%（人类 60%）
- **Apply 和 Craft 是最大瓶颈**：提示需求分布分析显示，Apply（创造性工具使用）和 Craft（工具合成）类关键步骤的提示依赖最高，是模型最弱的能力维度
- **EscapeAgent 全面有效**：Foresight + Reflection 在所有测试模型上均大幅降低 hints 和 steps。GPT-4o 从 10.30 hints 降到 5.03（↓51%），Early Exit Progress 从 24.75% 提升到 47.03%
- 模型规模与创意能力正相关但差距远不如分析推理任务——即使是最大的闭源模型也远未达到人类水平
- 部分小模型（如 DeepSeek-67B、Yi-34B）使用 EscapeAgent 后 Tool Hints 反而略有增加，说明 Foresight 在弱模型上可能引入噪声假设

## 亮点与洞察

- **首个聚焦创意智能的 Agent 基准**：填补了 Sternberg 三元智力理论中创造性维度的评估空白。密室逃脱作为载体非常巧妙——自然包含创造性工具使用、隐式目标发现和超长推理链三大核心挑战
- **Foresight 的"先假设再行动"模式**：将创造性思维外显化为可验证的假设-行动循环，本质上是 hypothesis-driven reasoning。这种方法可迁移到机器人操作中的创造性工具使用、开放世界游戏 AI 等领域
- **精细的评估指标体系**：hints used、total steps、early exit progress、tool hints / key steps hints 等多维度指标，能细粒度刻画创意推理中不同能力维度的表现

## 局限与展望

- 36 个场景虽然手工标注质量高，但规模有限，可能不足以覆盖创意智能的所有维度
- 当前仅基于文本交互，如果加入视觉信息（如物品外观、材质纹理），可能对创造性工具使用推理更有帮助
- EscapeAgent 的 Foresight 仍依赖 LLM 自身的推理能力，对于完全超出训练分布的创造性场景仍可能失效
- 密室逃脱虽有趣但毕竟是游戏场景，与真实世界创造性问题求解之间的迁移性有待验证
- 模型在 Apply/Craft 类动作上的极端弱势暗示需要从根本上改变工具使用的训练范式，仅靠 prompting 可能不够

## 相关工作与启发

- **vs TextWorld/Zork**: 传统文本游戏评测语言理解和规划能力，但场景相对简单且目标明确；EscapeBench 的隐式目标和创造性工具使用要求更高阶的认知能力
- **vs Minecraft 基准**: Minecraft 测试空间推理和规划，但工具使用通常遵循标准的 crafting recipe 模式；EscapeBench 要求非常规工具使用（如木棍当撬棒）
- **vs AUT/TTCT 创造力心理测试**: 心理学创造力测试评估发散思维但脱离任务执行；EscapeBench 在实际交互任务中评估创造力，更具生态效度
- 该工作启发的核心问题：当前 Agent 的"创造力"瓶颈是出在模型能力上，还是框架设计上？EscapeAgent 的显著提升暗示框架设计很重要，但与人类的巨大差距说明模型能力本身也需要根本性提升

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首创性地提出创意智能评测问题并给出完整的环境+评估+方法方案
- 实验充分度: ⭐⭐⭐⭐⭐ 12 个模型对比、人类基线、EscapeAgent 在 8 个模型上的全面消融、错误分析
- 写作质量: ⭐⭐⭐⭐ 问题动机论述深入，三元智力理论的引用准确且有说服力
- 价值: ⭐⭐⭐⭐⭐ 开创了 Agent 创意智能评估的新方向，benchmark 和 agent 框架均有长期影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AgentGym: Evolving Large Language Model-based Agents across Diverse Environments](agentgym_evaluating_and_training_large_language_model-based_agents_across_divers.md)
- [\[ACL 2025\] SocialEval: Evaluating Social Intelligence of Large Language Models](socialeval_evaluating_social_intelligence_of_large_language_models.md)
- [\[NeurIPS 2025\] Writing in Symbiosis: Mapping Human Creative Agency in the AI Era](../../NeurIPS2025/llm_nlp/writing_in_symbiosis_mapping_human_creative_agency_in_the_ai_era.md)
- [\[ACL 2025\] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)
- [\[ACL 2025\] MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)

</div>

<!-- RELATED:END -->
