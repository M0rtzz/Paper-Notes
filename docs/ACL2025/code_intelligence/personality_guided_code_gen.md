---
title: >-
  [论文解读] Personality-Guided Code Generation Using Large Language Models
description: >-
  [ACL 2025][code generation] 用 GPT-4o 为每个编程任务动态生成适配的 MBTI 人格类型和详细描述，再让目标 LLM 以该人格角色扮演程序员生成代码，在 7 个 LLM × 4 个数据集的 28 个组合中 23 个取得 pass rate 提升，最高达 12.9%，关键因素是人格多样性而非某个特定人格。
tags:
  - ACL 2025
  - code generation
  - personality
  - MBTI
  - 提示学习
  - role-playing
---

# Personality-Guided Code Generation Using Large Language Models

**会议**: ACL 2025  
**arXiv**: [2411.00006](https://arxiv.org/abs/2411.00006)  
**代码**: [GitHub](https://github.com/IanWalls/Persona-Code)  
**领域**: Code Intelligence  
**关键词**: code generation, personality, MBTI, prompt engineering, role-playing

## 一句话总结

用 GPT-4o 为每个编程任务动态生成适配的 MBTI 人格类型和详细描述，再让目标 LLM 以该人格角色扮演程序员生成代码，在 7 个 LLM × 4 个数据集的 28 个组合中 23 个取得 pass rate 提升，最高达 12.9%，关键因素是人格多样性而非某个特定人格。

## 研究背景与动机

**领域现状**：LLM 驱动的代码生成已成为主流，常见做法是通过 prompt 让 LLM 扮演"程序员"角色来生成代码。同时，软件工程研究已充分证明，任务与开发者人格的匹配能提升开发质量，团队人格多样性与高质量软件交付正相关。

**现有痛点**：现有的 role-playing prompt 只是泛泛地让 LLM 当"程序员"，没有细化到具体的人格特质。所有编程任务使用同一种角色设定，忽视了不同任务可能适合不同思维方式的事实。

**核心矛盾**：人类软件开发中"任务-人格匹配"带来的质量提升是否可以迁移到 LLM 的代码生成中？如果可以，什么因素影响效果？

**切入角度**：借鉴 MBTI 人格框架，为每个编程任务自动生成适配的人格类型，让 LLM 以多样化的人格角色生成代码，并通过大规模实验验证效果和影响因素。

## 方法详解

### 整体框架

两阶段 pipeline：
1. **Personality Generation**：用 GPT-4o 分析编程任务描述，输出最适合的 MBTI 类型（16 种之一）及其针对该任务的详细人格描述
2. **Code Generation**：将生成的 MBTI 类型和详细描述嵌入 prompt，让目标 LLM 以该人格角色扮演程序员生成代码

评估方式：生成的代码需通过全部测试用例才算"pass"。每个 LLM 在每个数据集上运行 3 次取平均 pass rate。

### 关键设计

1. **任务适配的动态人格生成**:

    - 功能：GPT-4o 根据每个编程任务的具体特点，动态选择最适合的 MBTI 类型并生成定制描述
    - 核心思路：不同编程任务适合不同认知风格——逻辑密集型适合 Thinking 维度、算法设计适合 Intuition 维度、调试任务适合 Sensing 维度
    - 设计动机：统一固定任何一种 MBTI 类型的效果与不用人格引导几乎无差（Qwen-Long 上 16 种固定 MBTI 的 pass rate 为 65.7%-68.4%，而多样化适配达 80.8%），说明"多样性"才是关键

2. **Full Prompt 设计（详细人格描述）**:

    - 功能：prompt 中不仅指定 MBTI 四字母类型，还包含 GPT-4o 生成的针对该任务的详细人格描述
    - 核心思路：帮助 LLM 更深入地理解和扮演角色
    - 设计动机：Full Prompt 比仅标注 MBTI 类型（Short Prompt）在 7 个 LLM 上平均高 3.94%；使用通用模板描述（而非任务定制描述）的 Qwen-Long pass rate 仅 65.5%，远低于定制描述的 80.8%

3. **与其他 prompt 策略的正交组合**:

    - 功能：人格引导可与 CoT、few-shot 等策略叠加使用
    - 核心思路：人格引导影响"角色认知"，CoT 影响"推理过程"，二者正交互补
    - 设计动机：CoT + Personality 在 5/7 个 LLM 上优于单独任一策略，最大提升达 13.8%

### 损失函数 / 训练策略

本文为零训练的 prompt engineering 方法，不涉及模型训练。人格生成使用 GPT-4o，代码生成使用各目标 LLM 的默认设置。

## 实验关键数据

### 主实验

7 个 LLM × 4 个数据集的 pass rate 对比（Direct vs MBTI-guided）：

| LLM | MBPP Sanitized | MBPP+ | HumanEval+ | APPS | 平均提升 |
|-----|---------------|-------|------------|------|---------|
| GPT-4o | 78.2→84.3 (+6.1%) | 71.2→72.7 (+1.5%) | 84.8→82.9 (-1.9%) | 46.2→45.2 (-1.0%) | +1.2% |
| GPT-4o mini | 69.3→82.2 **(+12.9%)** | 69.4→71.7 (+2.3%) | 80.5→82.3 (+1.8%) | 34.6→37.2 (+2.6%) | +4.9% |
| Llama3.1 (70B) | 69.8→81.0 (+11.2%) | 66.7→69.2 (+2.5%) | 72.0→72.6 (+0.6%) | 18.4→25.2 (+6.8%) | +5.3% |
| Qwen-Long | 68.4→80.8 (+12.4%) | 67.7→71.2 (+3.5%) | 76.8→78.7 (+1.9%) | 10.2→18.2 (+8.0%) | **+6.5%** |
| DeepSeek-Coder V2 | 74.9→85.7 (+10.8%) | 71.4→72.2 (+0.8%) | 80.5→76.2 (-4.3%) | 39.4→34.4 (-5.0%) | +0.6% |
| Codestral (22B) | 64.2→73.8 (+9.6%) | 61.2→64.9 (+3.7%) | 75.6→76.8 (+1.2%) | 15.8→22.6 (+6.8%) | +5.3% |
| CodeLlama (13B) | 43.3→46.8 (+3.5%) | 42.4→52.4 (+10.0%) | 32.9→29.9 (-3.0%) | 1.4→6.4 (+5.0%) | +3.9% |

**整体**：28 个组合中 23 个提升，11 个超过 5%，5 个超过 10%。

### 消融实验

| 配置 | Qwen-Long MBPP Pass Rate | 说明 |
|------|--------------------------|------|
| Direct（无人格） | 68.4% | 基线 |
| 16 种固定 MBTI 类型 | 65.7%-68.4% | 与基线几乎无差 |
| Diverse MBTI (Full Prompt) | **80.8%** | +12.4%，多样性是核心 |
| Short Prompt（仅 MBTI 四字母） | 73.3% | 比 Full Prompt 低 7.5% |
| 通用模板描述 | 65.5% | 远不如任务定制描述 |
| MBTI vs Big Five 人格 | 80.8% vs 71.4% | MBTI 显著优于大五人格 |
| Personality alone | 84.3% (GPT-4o) | 优于 3-shot (77.3%) 和 CoT (77.0%) |
| CoT + Personality | 85.7% (GPT-4o) | 组合最优 |
| CoT + Personality (Qwen-Long) | **82.2%** | 比 Direct 提升 **13.8%** |

### 关键发现

- **中等性能模型获益最大**：GPT-4o（太强）和 CodeLlama（太弱）获益有限，Qwen-Long/Llama3.1/Codestral 获益最显著
- **中等难度数据集提升最大**：HumanEval+（78%任务被分配 INTJ，多样性低）和 APPS（90.6% 为 INTJ）效果不稳定
- **人格多样性是核心因素**：任何固定单一人格 ≈ 无人格引导；不同人格解决的任务有互补性（INTJ vs ISTJ 各有 4.5% 独有可解题目）
- **MBTI 优于 Big Five**：MBTI 的四维度（S/N、T/F、E/I、J/P）更贴合编程的认知需求；Big Five 仅 conscientiousness 与编程强相关
- **Full Prompt 一致优于 Short Prompt**：任务定制的详细描述比仅标注类型平均高 3.94%

## 亮点与洞察

- 将软件工程中的人格-任务匹配理论跨域迁移到 LLM prompt 设计，思路有启发性
- 核心发现是"多样性"而非"某个特定人格"带来提升——暗示 role-playing prompt 需要与任务特点动态匹配而非静态设定
- 方法极其轻量：零训练，仅在 prompt 前加一段人格描述，可即插即用
- CoT + Personality 组合策略展示了 prompt engineering 各维度正交互补的可能性

## 局限与展望

- 人格生成依赖 GPT-4o，引入额外 API 调用成本，每个任务增加一次 LLM 调用
- MBTI 在心理学界有争议，不是公认最科学的人格框架
- 仅在 Python 函数级代码生成上验证，未覆盖仓库级、多文件等复杂场景
- 为什么人格引导有效的深层机制不清楚——可能只是通过更丰富的 prompt 增加了角色扮演的上下文信息
- 无法计算理论上限：缺乏人格-任务匹配的 ground truth

## 相关工作与启发

- **vs 直接 role-playing**：简单的"你是程序员"在所有 7 个 LLM 上都不如人格引导，说明角色扮演需要细化
- **vs CoT**：人格引导在所有 7 个 LLM 上单独优于 CoT 和 3-shot，且可与 CoT 组合获得额外提升
- **vs 团队多样性研究 (Pieterse et al., 2018)**：验证了人格多样性在 LLM 场景下同样重要

## 评分

- 新颖性: ⭐⭐⭐ 将 MBTI 引入代码生成有趣但本质上是 prompt engineering 的变体
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个 LLM、4 个数据集、5 个 RQ、16 种固定 MBTI 对比、双人格框架对比，极为全面
- 写作质量: ⭐⭐⭐⭐ RQ 驱动的实证研究范式清晰，每个 RQ 都有明确结论
- 价值: ⭐⭐⭐ 实用性一般（不同场景效果不稳定），但"多样性是关键"的发现有启发价值

<!-- RELATED:START -->

## 相关论文

- [CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)
- [DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models](codereviewqa_the_code_review_comprehension_assessment_for_large_language_models.md)
- [GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](galla_graph_aligned_large_language_models.md)
- [Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models](tree_of_evolution_code_gen.md)

<!-- RELATED:END -->
