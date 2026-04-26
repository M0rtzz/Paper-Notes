---
title: >-
  [论文解读] Iterative Formalization and Planning in Partially Observable Environments
description: >-
  [ACL 2026][LLM/NLP][部分可观测环境] 提出 PDDLego+ 框架，让 LLM 在部分可观测环境中迭代地生成和修正 PDDL（规划领域定义语言）表示，通过双层错误修复循环（solver error + simulation error）实现无需微调、无需示例的有效规划。
tags:
  - ACL 2026
  - LLM/NLP
  - 部分可观测环境
  - PDDL形式化
  - 迭代规划
  - LLM-as-Formalizer
  - 错误修复
---

# Iterative Formalization and Planning in Partially Observable Environments

**会议**: ACL 2026  
**arXiv**: [2505.13126](https://arxiv.org/abs/2505.13126)  
**代码**: [GitHub](https://github.com/zharry29/pddlego-plus)  
**领域**: LLM NLP / AI Planning  
**关键词**: 部分可观测环境, PDDL形式化, 迭代规划, LLM-as-Formalizer, 错误修复

## 一句话总结

提出 PDDLego+ 框架，让 LLM 在部分可观测环境中迭代地生成和修正 PDDL（规划领域定义语言）表示，通过双层错误修复循环（solver error + simulation error）实现无需微调、无需示例的有效规划。

## 研究背景与动机

**领域现状**：利用大语言模型进行规划是 AI 规划领域的热门方向，现有方法主要分为两类：LLM-as-planner（直接生成动作计划）和 LLM-as-formalizer（将环境形式化为 PDDL 再用传统求解器规划）。后者因可解释性好、可控性强而受到青睐，但绝大多数工作仅关注完全可观测环境。

**现有痛点**：真实世界中的规划场景（如机器人探索未知房间、网页代理操作）通常是部分可观测的——agent 只能看到局部观测，无法一次性生成完整计划。少数处理部分可观测环境的工作存在三个缺陷：(1) 假设部分规划表示已知（如预定义的 predicates 或 domain file）；(2) 仅使用一次性形式化而非迭代改进；(3) 依赖现有轨迹作为 in-context 示例。

**核心矛盾**：PDDL 等规划语言基于封闭世界假设——要求初始状态和目标的完整定义。这与部分可观测环境中信息逐步揭示的本质直接矛盾。

**本文目标**：设计一个无需微调、无需示例、无需预设 domain file 的框架，让 LLM 在部分可观测环境中通过迭代探索和错误修复，逐步构建完整的 PDDL 表示并完成规划任务。

**核心 idea**：将部分可观测问题分解为一系列完全可观测的子问题，每次基于当前观测生成局部 PDDL，用求解器规划并执行，根据新观测和错误反馈迭代更新。

## 方法详解

### 整体框架

PDDLego+ 的核心流程是一个"生成→求解→执行→更新"的迭代循环：(1) LLM 根据当前观测生成 Domain File（$\mathbb{DF}$，定义类型、谓词、动作）和 Problem File（$\mathbb{PF}$，定义对象、初始状态、目标）；(2) 形式化求解器（Fast Downward）搜索动作计划；(3) 在模拟环境中执行计划；(4) 根据新观测更新 PDDL 或根据错误修复 PDDL。与 PDDLego 不同，PDDLego+ 同时推断 DF 和 PF，不假设 domain file 已知。

### 关键设计

1. **双层错误修复循环（Two-Phase Error Refinement）**

    - 功能：处理 PDDL 生成过程中的两类错误
    - 核心思路：内层循环处理 solver error（PDDL 语法/语义错误导致求解器失败），外层循环处理 simulation error（计划在模拟器中执行失败）。形式化为：solver error 修复 $\mathrm{df}_i^{j,k+1}, \mathrm{pf}_i^{j,k+1} = \text{LLM}(\mathrm{err}_{\text{sol}}, \mathrm{df}_i^{j,k}, \mathrm{pf}_i^{j,k})$；simulation error 修复 $\mathrm{df}_i^{j+1}, \mathrm{pf}_i^{j+1} = \text{LLM}(\mathrm{err}_{\text{sim}}, \mathrm{df}_i^j, \mathrm{pf}_i^j)$
    - 设计动机：两类错误性质不同——solver error 是即时的语法/逻辑问题，simulation error 是更深层的语义问题（如缺少前置条件），需要分层处理

2. **目标分解与子目标预测（Goal Decomposition）**

    - 功能：将不可达的全局目标分解为当前可达的子目标
    - 核心思路：提供两种提示模板——simple prompt（粗略的目标分解指引）和 detailed prompt（提供 PDDL 目标模板如 `(:goal (at ?location))`，让 LLM 填充占位符）。每个时间步 LLM 预测一个局部可达的子目标
    - 设计动机：部分可观测环境中全局目标通常不可直接达成，必须通过探索逐步接近

3. **完整 Domain+Problem 推断**

    - 功能：同时从自然语言观测推断 DF 和 PF，而非假设 DF 已知
    - 核心思路：LLM 接收文本观测（如"你在厨房，东边有一扇关着的门"），生成完整的 PDDL 类型定义、谓词、动作语义（DF）以及对象实例、初始状态、目标（PF）
    - 设计动机：DF 推断是更困难的任务（类比于合成类和函数 vs 合成函数调用），但在真实场景中 DF 不可能预先给定

### 域知识复用

成功试验结束后产生的 DF 可作为"学到的域知识"复用于未来任务。实验使用 RAG 方式从历史成功试验中检索 DF，固定 DF 后仅让 LLM 预测 PF，某些模型上显著提升成功率。

## 实验关键数据

### 主实验

在 CoinCollector（导航任务）和 ALFWorld（物体操作任务）两个文本模拟环境上评估：

| 方法 | CoinCollector (o3-mini) | ALFWorld (o3-mini) |
|------|------------------------|-------------------|
| PlanGen (LLM-as-planner) | 52% | 5% |
| PDDLego (无修复) | 49% | 3% |
| PDDLego+ (本文) | **86%** | **38%** |

| 模型 | CoinCollector PlanGen/PDDLego+ | ALFWorld PlanGen/PDDLego+ |
|------|-------------------------------|--------------------------|
| DeepSeek-R1 | ~55% / ~75% | ~8% / ~25% |
| GPT-4.1 | ~60% / ~55% | ~3% / ~20% |
| o3-mini | 52% / 86% | 5% / 38% |
| o4-mini | ~65% / ~80% | ~10% / ~30% |

### 消融实验

- **复杂度鲁棒性**：CoinCollector 房间数从 3 增到 11 时，PDDLego+ 成功率保持稳定，PlanGen 和 PDDLego 逐渐下降
- **目标提示消融**：detailed prompt 优于 simple prompt，但 simple prompt 下 PDDLego+ 仍显著优于基线
- **域知识复用**：使用 RAG 检索的 DF，DeepSeek-R1 和 GPT-4.1 成功率提升，o3-mini 略有下降（已有足够强的 DF 生成能力）

### 关键发现

- PDDLego+ 在较复杂的 ALFWorld 上对所有模型都优于 PlanGen，说明形式化方法在复杂规划任务中的优势
- 大多数错误是 solver error（PDDL 语法问题），而非 simulation error，o3-mini 的错误修复率最高
- 错误分析显示主要瓶颈在 PF 的语义错误：幻觉事实、不可达目标、遗忘已观测信息

## 亮点与洞察

- **形式化方法在部分可观测环境中可行**：首次系统性证明 LLM-as-formalizer 在部分可观测环境中的有效性，打破了"PDDL 只能用于完全可观测环境"的认知
- **可解释性优势**：与 LLM-as-planner 不同，PDDLego+ 的每个失败都可以归因到具体的 PDDL 错误，支持因果错误分析
- **域知识可迁移**：成功试验产生的 DF 可复用，展示了形式化方法在知识积累方面的独特优势
- **推理模型的优势**：o3-mini 等推理模型在 PDDL 生成中显著优于常规模型，与 Huang & Zhang (2025) 的发现一致

## 局限与展望

- 依赖环境提供信息丰富的错误消息，在错误反馈模糊的环境中可能失效
- 需要针对特定环境设计提示词，泛化到未知领域的能力有限
- 需要高能力 LLM（如 o3-mini/DeepSeek-R1）且多次调用，计算成本高
- ALFWorld 上最高成功率仅 38%，仍有巨大提升空间
- PF 中的幻觉事实和遗忘问题是主要瓶颈，需要更好的世界状态维护机制

## 相关工作与启发

- **vs PDDLego (Zhang et al. 2024)**：PDDLego 假设 DF 已知且无错误修复，PDDLego+ 推断完整 DF+PF 并引入双层修复循环
- **vs PlanGen (LLM-as-planner)**：在简单任务上 PlanGen 有时更优（直接生成动作无需形式化），但在复杂任务（ALFWorld）上 PDDLego+ 全面领先
- **vs ReAct**：PDDLego+ 可视为 ReAct 的形式化升级版——用 PDDL 替代自然语言推理，获得形式保证

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次在部分可观测环境中实现完整的迭代 PDDL 形式化，双层修复循环设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 两个环境、四个模型、多维度分析和错误解剖，但 ALFWorld 成功率偏低
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法形式化完整，错误分析细致
- 价值: ⭐⭐⭐⭐ 为 LLM 驱动的形式化规划在真实场景中的应用提供了可行路径

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning](../../ICLR2026/llm_nlp/from_assumptions_to_actions_turning_llm_reasoning_into_uncertainty-aware_plannin.md)
- [\[ACL 2025\] LLM as a Broken Telephone: Iterative Generation Distorts Information](../../ACL2025/llm_nlp/llm_broken_telephone.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](../../ACL2025/llm_nlp/plangenllms_planning_survey.md)
- [\[ACL 2025\] On the Limit of Language Models as Planning Formalizers](../../ACL2025/llm_nlp/limit_llm_planning_formalizer.md)
- [\[ACL 2025\] AgentGym: Evolving Large Language Model-based Agents across Diverse Environments](../../ACL2025/llm_nlp/agentgym_evaluating_and_training_large_language_model-based_agents_across_divers.md)

<!-- RELATED:END -->
