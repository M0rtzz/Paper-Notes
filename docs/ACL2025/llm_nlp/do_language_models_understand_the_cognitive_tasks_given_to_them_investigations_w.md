---
title: >-
  [论文解读] Do Language Models Understand the Cognitive Tasks Given to Them?
description: >-
  [ACL 2025][LLM/NLP][cognitive evaluation] 用 N-back 任务系统分析 LLM 认知任务理解能力，发现性能低下的主因是任务理解不足而非工作记忆限制，挑战了 LLM 工作记忆容量约 3 的结论。
tags:
  - ACL 2025
  - LLM/NLP
  - cognitive evaluation
  - N-back
  - working memory
  - task understanding
  - LLM
---

# Do Language Models Understand the Cognitive Tasks Given to Them? Investigations with the N-Back Paradigm

**会议**: ACL 2025  
**arXiv**: [2412.18120](https://arxiv.org/abs/2412.18120)  
**代码**: https://github.com/hxiaoyang/lm-nback  
**领域**: LLM/NLP - 认知评估  
**关键词**: cognitive evaluation, N-back, working memory, task comprehension, LLM

## 一句话总结
通过 N-back 范式系统分析多个 LLM 的认知任务表现，发现性能低下的主因是任务理解不足和任务集维持失败，而非工作记忆容量限制，最佳模型（Llama 3.1 70b）在课程学习辅助下甚至能完成 10-back 任务（准确率 84.75%）。

## 研究背景与动机

**领域现状**：认知心理学中的实验范式（如 N-back 任务）越来越多地被用于评估 LLM 的认知能力，包括心智理论、类比推理和工作记忆等。这些评估旨在理解 LLM 是否具备类似人类的认知构造。

**现有痛点**：Gong et al. (2024) 将 N-back 任务应用于 GPT-3.5，观察到 2-back 和 3-back 性能急剧下降后，得出 LLM 工作记忆容量约为 3 的结论（类似人类）。这一解读存在两个问题：(1) 不能假设人类的工作记忆约束在 LLM 中同样存在；(2) 性能低下可能源于任务理解失败而非记忆限制。

**核心矛盾**：当模型表现不佳时，究竟是被测试的认知能力存在限制，还是模型根本没有理解要执行什么任务？现有研究未能区分这两种可能。

**本文目标** 系统区分 LLM 在 N-back 任务中的性能下降到底是由任务理解不足还是工作记忆限制导致的。

**切入角度**：对多个不同性能层级的开源模型进行 N-back 测试，引入反事实度量（counterfactual measures）分析模型实际执行的是哪个 m-back 任务，并通过注意力分析验证任务理解。

**核心 idea**：LLM 在 N-back 任务上的差异主要反映任务理解能力的差异，而非工作记忆容量的差异。

## 方法详解

### 整体框架
使用标准 N-back 数据集（每个 n-back 任务 50 个 trial，每个 trial 24 个字母序列），对多个开源指令微调模型和 GPT-3.5 进行系统评估。评估维度包括：(1) 基准性能分层，(2) 任务理解分析，(3) 任务集维持分析，(4) 高 n 值性能测试，(5) 课程学习，(6) 交互式演示，(7) 注意力分析。

### 关键设计

1. **反事实度量体系（Counterfactual Measures）**:

    - 功能：在给定 n-back 指令的情况下，同时评估模型对所有可能的 m-back（$m \neq n$）任务的响应概率
    - 核心思路：定义 $\mathsf{P}_{n,m}$ 为在 n-back 指令和演示下 m-back 检索的平均对数概率；$\mathsf{P}^{-}_{n,m}$ 为仅有指令（无演示）时的对应值。若 $\max_m \mathsf{P}_{n,m} = \mathsf{P}_{n,n}$，则说明模型真正理解了 n-back 任务
    - 设计动机：传统评估只看正确率，无法区分"做错了正确任务"和"做对了错误任务"两种情况

2. **三层级模型分类（Three Performance Tiers）**:

    - 功能：将模型按 2-back 和 3-back 检索准确率分为 T1（>80%）、T2（~50%）、T3（≤20%）三个层级
    - 核心思路：T3（如 Qwen 1.5 14b）完全误解任务——即使给了 2-back 指令仍执行 1-back；T2（如 Gemma 2 27b、GPT-3.5）从正确任务漂移到错误任务；T1（如 Llama 3.1 70b）始终正确理解并执行任务
    - 设计动机：不同层级反映了不同级别的任务理解能力，而非不同的工作记忆容量

3. **任务集维持分析（Task Set Maintenance）**:

    - 功能：追踪模型在 24 步 trial 中随时间的任务一致性变化
    - 核心思路：定义 $\mathsf{A}_{n,\cdot}(m,i)$ 为在第 $i$ 步时 m-back 一致检索的平均准确率。研究错误积累效应——给 T2 模型提供 m-back 一致的历史响应后，观察后续步骤的行为变化
    - 设计动机：发现 T2 模型的性能下降源于错误积累导致的任务漂移

4. **课程学习策略（In-context Curriculum Learning）**:

    - 功能：在测试 n-back 前，先提供从 1-back 到 (n-1)-back 的完整演示序列
    - 核心思路：渐进式增加任务难度，帮助模型逐步理解更复杂的 N-back 任务
    - 设计动机：此策略使 Llama 3.1 70b 在 8/9/10-back 上的任务准确率达到 90.08%/90.08%/84.75%

5. **注意力模式分析（Attention Analysis）**:

    - 功能：计算每个检索 token 对正确源 token（n 步前）的平均注意力（MRAT）
    - 核心思路：高性能模型应在检索时更多地关注 n 步前的 token。Qwen 2 72b 的最高 MRAT 达 71.98%，而 Qwen 1.5 14b 仅 38.95%
    - 设计动机：提供了任务理解的机制性证据

## 实验关键数据

### 主实验

| 模型 | 层级 | 1-back | 2-back | 3-back |
|------|------|--------|--------|--------|
| Qwen 1.5 14b Chat | T3 | 1.00 | 0.09 | 0.08 |
| Llama 3.1 8b Instr. | T3 | 1.00 | 0.14 | 0.17 |
| Gemma 2 27b Instr. | T2 | 1.00 | 0.57 | 0.36 |
| GPT 3.5 Turbo | T2 | 1.00 | 0.51 | 0.43 |
| Qwen 2 72b Instr. | T1 | 1.00 | 0.81 | 0.84 |
| Llama 3.1 70b Instr. | T1 | 1.00 | 0.99 | 0.93 |

### 消融实验（课程学习 vs 直接测试，Llama 3.1 70b）

| N值 | 直接测试-检索准确率 | 课程学习-检索准确率 | 直接测试-任务准确率 | 课程学习-任务准确率 |
|-----|---------------------|---------------------|---------------------|---------------------|
| 8 | 75.25% | 79.83% | 83.33% | 90.08% |
| 9 | 66.08% | 80.17% | 78.25% | 90.08% |
| 10 | 57.10% | 71.67% | 71.92% | 84.75% |

### 关键发现
- T3 模型在给定 2-back 指令时, $\mathsf{P}_{2,1} > \mathsf{P}_{2,2}$，说明模型完全误解任务，实际在执行 1-back
- T2 模型（含 GPT-3.5）初始能理解任务但随错误积累逐渐漂移到 1-back 行为
- T1 模型即使仅给指令（无演示），也能正确推断任务：$\mathsf{P}^{-}_{2,2} > \mathsf{P}^{-}_{2,1}$
- 交互式演示反而不如标准演示有效——Llama 3.1 70b 在 3-back 上从 0.93 降至 0.62
- Qwen 模型族中存在清晰的注意力-性能对应关系，但 Llama 模型的注意力更分散

## 亮点与洞察
- 提供了区分"任务理解失败"与"认知能力不足"的系统方法论，对所有认知评估研究都有参考价值
- 反事实度量方法巧妙——不仅看模型做对了什么，还看模型"以为自己在做什么"
- 课程学习的成功说明 LLM 具备远超预期的序列记忆能力，只是需要正确的任务理解引导

## 局限与展望
- 未深入分析模型内部电路机制（如哪些层/组件负责任务理解）
- Llama 模型注意力模式异常分散的原因未解释清楚
- 提示选择仍有优化空间——可能存在更有效的提示策略来提升任务理解

## 相关工作与启发
- **vs Gong et al. (2024)**：直接挑战其"LLM 工作记忆容量约 3"的结论，证明性能下降源于任务理解而非记忆限制
- **vs Zhang et al. (2024)**：后者虽然承认小模型可能不理解任务意图，但未将任务理解作为混杂变量控制
- **vs Biran et al. (2024)**：多跳问答中也发现 LLM 在后续步骤的推理能力下降，与 N-back 中的任务漂移现象类似

## 评分
- 新颖性: ⭐⭐⭐⭐ 反事实度量方法新颖，但研究问题本身偏验证性
- 实验充分度: ⭐⭐⭐⭐⭐ 七个维度的系统分析，多层级模型对比，覆盖 1-10 back
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，实验设计层层递进，图表直观
- 价值: ⭐⭐⭐⭐ 对 LLM 认知评估方法论有重要指导意义
# Do Language Models Understand the Cognitive Tasks Given to Them?

**会议**: ACL 2025  
**arXiv**: [2412.18120](https://arxiv.org/abs/2412.18120)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: cognitive evaluation, N-back, working memory, task understanding, LLM

## 一句话总结
用 N-back 任务系统分析 LLM 认知任务理解能力，发现性能低下的主因是任务理解不足而非工作记忆限制，挑战了 LLM 工作记忆容量约 3 的结论。

## 研究背景与动机

**领域现状**：该领域正快速发展，LLM 在相关任务上展现出强大但不完美的能力。

**现有痛点**：现有评估方法或解决方案存在覆盖不全或方法论局限。

**核心矛盾**：如何更准确地理解和改进 LLM 在该任务上的表现？

**本文目标** 提供新的评估视角/方法/基准来推动领域发展。

**切入角度**：从独特的理论框架或方法论出发。

**核心 idea**：用 N-back 任务系统分析 LLM 认知任务理解能力。

## 方法详解

### 整体框架
本文提出新颖的评估框架/方法，针对 LLM 在该领域的特定挑战进行系统性研究。

### 关键设计

1. **核心方法/框架设计**

    - 功能：构建评估框架或解决方案
    - 核心思路：基于领域特定的理论和方法
    - 设计动机：弥补现有工作的不足

2. **数据/实验设计**

    - 精心设计的实验方案覆盖多个维度
    - 设计动机：确保结论的可靠性和泛化性

## 实验关键数据

### 主实验

| 设置 | 指标 | 结果 | 说明 |
|------|------|------|------|
| 主要评估 | 核心指标 | 见论文 | 验证核心假设 |

### 分析

| 维度 | 发现 |
|------|------|
| 方法有效性 | 验证了核心方法的有效性 |
| 模型差异 | 不同模型表现有显著差异 |

### 关键发现
- 论文的核心假设得到验证
- 揭示了 LLM 在该任务上的特定模式
- 为后续研究提供了新的方向

## 亮点与洞察
- 从新颖的角度审视 LLM 能力
- 方法或发现对实际应用有指导意义

## 局限与展望
- 评估规模可进一步扩大
- 模型覆盖范围可扩展
- 后续可探索更多场景

## 相关工作与启发
- 与同领域其他工作互补
- 为后续研究提供了基础

## 评分
- 新颖性: ⭐⭐⭐ 在特定方向有贡献
- 实验充分度: ⭐⭐⭐ 覆盖合理
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐ 对特定社区有价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Do Language Models Understand Honorific Systems in Javanese?](do_language_models_understand_honorific_systems_in_javanese.md)
- [\[ACL 2025\] Can Large Language Models Understand Internet Buzzwords Through User-Generated Content](buzzword_understanding_ugc.md)
- [\[ACL 2025\] Perspective Transition of Large Language Models for Solving Subjective Tasks](perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)
- [\[ACL 2025\] OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens](olmotrace_tracing_language_model_outputs_back_to_trillions_of_training_tokens.md)
- [\[ACL 2025\] SCoP: Evaluating the Comprehension Process of Large Language Models from a Cognitive View](scop_evaluating_the_comprehension_process_of_large_language_models_from_a_cognit.md)

<!-- RELATED:END -->
