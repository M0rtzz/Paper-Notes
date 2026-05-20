---
title: >-
  [论文解读] EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents
description: >-
  [ACL 2026][代码智能][软件工程Agent] 提出 EET——一种基于历史经验驱动的早停方法，在补丁生成和补丁选择阶段识别无效迭代并提前终止，将 SE Agent 总成本降低 19%-55%（平均 32%），同时几乎不损失任务性能（最多 0.2%）。
tags:
  - "ACL 2026"
  - "代码智能"
  - "软件工程Agent"
  - "成本优化"
  - "经验驱动"
  - "早停策略"
  - "SWE-bench"
---

# EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents

**会议**: ACL 2026  
**arXiv**: [2601.05777](https://arxiv.org/abs/2601.05777)  
**代码**: [GitHub](https://github.com/IanWalls/EET)  
**领域**: 代码智能  
**关键词**: 软件工程Agent, 成本优化, 经验驱动, 早停策略, SWE-bench

## 一句话总结

提出 EET——一种基于历史经验驱动的早停方法，在补丁生成和补丁选择阶段识别无效迭代并提前终止，将 SE Agent 总成本降低 19%-55%（平均 32%），同时几乎不损失任务性能（最多 0.2%）。

## 研究背景与动机

**领域现状**：基于 LLM 的软件工程 (SE) Agent 在自动化 issue 修复方面取得了显著进展，如 Agentless、Mini-SWE-Agent、Trae Agent 等在 SWE-bench 上表现出色。

**现有痛点**：SE Agent 的高昂货币成本是实际部署的主要障碍（53% 的开发者认为成本是使用障碍）。由于"token 雪球"效应，对话历史越来越长导致成本超线性增长；对难题或不可解问题的无效迭代进一步放大浪费。

**核心矛盾**：现有成本优化方法（如 turn-control）虽能降低成本，但会显著损害任务性能（平均下降 10.7%）。如何在大幅降成本的同时保持性能是核心挑战。

**本文目标**：提出一种通用的早停优化方法，可无缝集成到各类 SE Agent 中，在保持任务性能的前提下显著降低成本。

**切入角度**：借鉴经验丰富的开发者能直接定位解决方案而无需大量试错的直觉，用结构化历史经验指导 Agent 跳过冗余迭代。

**核心 idea**：将历史 issue 解决经验提炼为结构化知识（任务抽象 + 轨迹摘要 + 置信度评估），在新任务的补丁生成和选择阶段用于判断是否可提前终止。

## 方法详解

### 整体框架

EET 包含两大组件：(1) 经验生成——从历史 issue 解决记录中提炼结构化经验对象存入经验库；(2) 早停机制——在补丁生成阶段（里程碑检查点）和补丁选择阶段（置信度阈值）利用检索到的相关经验进行早停决策。

### 关键设计

1. **结构化经验表示与检索**:
    - 功能：将原始执行轨迹压缩为紧凑、可复用的经验对象
    - 核心思路：每个经验对象包含 task_description（issue 抽象）、execution_summary（轨迹摘要）、evaluation_result（均为 pass）、confidence 和 confidence_reason（质量评估）。仅保留成功解决的经验
    - 设计动机：原始轨迹噪声大、token 开销高；过度压缩又会丢失可用信号。结构化表示在信息密度和实用性之间取得平衡

2. **补丁生成阶段的里程碑早停**:
    - 功能：在单个补丁生成过程中识别无需继续迭代的时机
    - 核心思路：将代码修改和测试执行定义为里程碑，在每个里程碑后评估置信度分数，超过阈值 τ^gen 则提前终止
    - 设计动机：补丁质量的信号可能在代码修改后（结构对齐）或测试执行后（动态反馈）出现，双里程碑设计覆盖两种情况

3. **补丁选择阶段的双阈值早停**:
    - 功能：动态控制需要生成的候选补丁数量
    - 核心思路：每生成一个补丁，结合补丁内容、执行轨迹和历史经验评估置信度。高于 τ_upper^sel → 停止（补丁已足够好）；低于 τ_lower^sel → 停止（当前难以解决）
    - 设计动机：避免固定补丁数量带来的低效——简单问题不需要多个补丁，困难问题生成更多补丁也无益

### 损失函数 / 训练策略

EET 为推理时优化方法，不涉及训练。关键超参数包括：TF-IDF 相似度阈值 τ_sim、生成早停阈值 τ^gen、选择上下阈值 τ_upper^sel / τ_lower^sel，在 SWE-bench 的 100 个独立验证样本上调优。经验库由 SWE-bench Lite（去重后 207 题）生成。

## 实验关键数据

### 主实验

| Agent + 后端 | 解决率变化 | API 调用 | 输入 Token | 输出 Token | 总成本变化 |
|-------------|-----------|---------|-----------|-----------|-----------|
| Agentless + GPT-5-mini | +7.8% | -26.4% | -51.8% | -51.0% | **-55.1%** |
| Agentless + DeepSeek-V3.2 | +7.2% | -25.5% | -31.9% | -35.0% | -32.2% |
| Mini-SWE + GPT-5-mini | +1.0% | -7.9% | -13.7% | -3.7% | -19.4% |
| Mini-SWE + DeepSeek-V3.2 | +0.6% | -8.4% | -13.6% | -4.4% | -19.3% |
| Trae + GPT-5-mini | 0.0% | -29.9% | -30.4% | -28.0% | -28.2% |
| Trae + DeepSeek-V3.2 | -0.2% | -26.5% | -37.7% | -28.2% | -36.7% |
| **平均** | **+2.7%** | **-20.8%** | **-29.9%** | **-25.1%** | **-31.8%** |

### 消融实验

| 变体 (Trae + GPT-5-mini) | 解决率变化 | 总成本变化 |
|--------------------------|-----------|-----------|
| 完整 EET | 0.0% | -28.2% |
| 去掉经验注入 | -10.4% | -58.9% |
| 去掉早停机制 | +0.4% | +3.1% |

### 关键发现

- EET 平均为 11.3% 的 issue 实现了早停（8.6%-14.0%），这部分 issue 的成本节约最为显著
- 对 Agentless 的提升最大（解决率反而提高 7.2-7.8%），因为经验指导弥补了其固定流程的不足
- 与 Turn-control 对比：Turn-control 虽成本降低更多（-41.4%），但解决率大幅下降（-10.7%）
- LLM 的置信度分数具有良好校准性：置信度 >90 的补丁通过率 63.6%-92.6%，<40 的仅 8.7%-13.8%
- 跨仓库迁移实验表明经验捕获的是通用调试模式而非仓库特异性线索

## 亮点与洞察

- 方法极其通用，可即插即用地集成到不同范式的 SE Agent 中（固定流程/自主规划/生成+选择）
- "经验"概念设计精妙：不是简单的 RAG 检索原始轨迹，而是提炼为含置信度评估的结构化知识
- 双阈值设计覆盖了"够好了可以停"和"太难了也该停"两种场景，比单阈值更合理
- 消融实验清晰揭示了经验注入和早停机制的互补关系

## 局限与展望

- 依赖历史数据构建经验库，对全新领域存在冷启动问题
- 仅在 SWE-bench Verified 上评估，工业场景的泛化性有待验证
- 早停决策依赖 LLM 的置信度输出，不同模型的校准质量可能差异较大
- 目前聚焦 SE Agent，但设计哲学（经验驱动早停）是领域无关的，可推广到通用多步推理 Agent

## 相关工作与启发

- 与 RAG-based agent memory（如 MetaGPT、MemoryBank）的区别：EET 的经验专门服务于成本优化，而非提升性能
- Fan et al. 的"token 雪球"分析揭示了成本问题的根源，EET 从经验复用角度提供了解决方案
- 对 Agent 系统设计的启示：成本优化应作为一等公民考虑，而非性能的附属品

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性地用经验驱动早停解决 SE Agent 成本问题，视角新颖实用
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个 Agent × 2 个 LLM 后端，含基线对比、消融、跨仓库迁移分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述准确，实验设计全面

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Training Software Engineering Agents and Verifiers with SWE-Gym](../../ICML2025/code_intelligence/training_software_engineering_agents_and_verifiers_with_swe-gym.md)
- [\[ICLR 2026\] Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](../../ICLR2026/code_intelligence/ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)
- [\[NeurIPS 2025\] SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](../../NeurIPS2025/code_intelligence/swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?](koco-bench_can_large_language_models_leverage_domain_knowledge_in_software_devel.md)

</div>

<!-- RELATED:END -->
