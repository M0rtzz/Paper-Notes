---
title: >-
  [论文解读] AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation
description: >-
  [ACL 2026 Findings][LLM评测][Agent评估] 提出 AJ-Bench，首个系统评估 Agent-as-a-Judge 能力的基准，覆盖搜索、数据系统和 GUI 三个领域共 155 个任务和 516 条标注轨迹，实验表明 Agent-as-a-Judge 比 LLM-as-a-Judge 平均 F1 提升约 13 个百分点。
tags:
  - "ACL 2026 Findings"
  - "LLM评测"
  - "Agent评估"
  - "Agent-as-a-Judge"
  - "环境交互验证"
  - "轨迹评估"
  - "基准测试"
---

# AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation

**会议**: ACL 2026 Findings  
**arXiv**: [2604.18240](https://arxiv.org/abs/2604.18240)  
**代码**: [https://aj-bench.github.io/](https://aj-bench.github.io/)  
**领域**: 强化学习  
**关键词**: Agent评估, Agent-as-a-Judge, 环境交互验证, 轨迹评估, 基准测试

## 一句话总结
提出 AJ-Bench，首个系统评估 Agent-as-a-Judge 能力的基准，覆盖搜索、数据系统和 GUI 三个领域共 155 个任务和 516 条标注轨迹，实验表明 Agent-as-a-Judge 比 LLM-as-a-Judge 平均 F1 提升约 13 个百分点。

## 研究背景与动机

**领域现状**：随着 RL 持续扩展 LLM Agent 训练规模，可靠验证 Agent 在复杂环境中的行为变得越来越关键。当前主流验证方式包括规则验证器和 LLM-as-a-Judge，前者依赖预定义规则，后者基于文本表面信号做判断。

**现有痛点**：规则验证器难以泛化到复杂、开放场景（如科学假设验证、长文事实核查）。LLM-as-a-Judge 无法获取环境状态信息，只能基于轨迹文本做表面判断，容易产生错误评估。例如，判断一个 Agent 是否正确查询了某个数据库，仅看轨迹文本是不够的，需要实际检查数据库状态。

**核心矛盾**：验证 Agent 行为需要理解环境状态变化，但现有评估框架将 Judge 限制在"旁观者"角色，无法与环境交互获取验证证据。

**本文目标**：构建首个系统评估 Agent-as-a-Judge 能力的基准，量化 Agent 评估者在信息获取、状态验证和过程验证方面的能力。

**切入角度**：赋予评估者"代理"能力——让 Judge 能够与环境交互、使用工具获取超出轨迹文本的证据来做出更可靠的判断。

**核心 idea**：构建需要环境交互的验证任务，系统对比 Agent-as-a-Judge 和 LLM-as-a-Judge 的差异，揭示环境交互对评估可靠性的关键作用。

## 方法详解

### 整体框架

AJ-Bench 检验的是评估者能否像 agent 一样主动与环境交互、核实另一个 agent 是否真正完成了任务，而不是只读轨迹文本做表面判断。基准覆盖搜索、数据系统（文件系统+Postgres）、GUI（PPT/Word/Excel）三个领域共 155 个任务、516 条人工标注轨迹，并为 DS 与 GUI 领域重建了可交互的环境副本。评估时，Judge Agent 拿到任务描述与候选轨迹后，可调用 60 种工具从环境最终状态出发取证，最终输出"成功/失败"的二分类判断。

### 关键设计

**1. 三维度评估能力设计：用三个领域分别压测信息获取、状态验证、过程验证**

LLM-as-a-Judge 的盲区在于它只能看轨迹文本，碰到需要外部事实或环境状态才能判定的任务就会出错。AJ-Bench 把这种盲区拆成三类可量化的能力：搜索领域要求评估者通过外部检索核实轨迹里的事实性声明（信息获取）；数据系统领域要求评估者用工具检查环境当前状态是否符合预期，例如文件是否存在、数据库记录是否被正确写入（状态验证）；GUI 领域要求评估者核对关键动作与执行步骤，例如 PPT 幻灯片是否真的被改对（过程验证）。这三类正好覆盖了 Agent-as-a-Judge 相较 LLM-as-a-Judge 的全部增量优势。

**2. 多源轨迹收集与标注：用多模型来源和长度解耦防止 Judge 走捷径**

搜索域用 Gemini、Grok、Perplexity 生成轨迹，DS 域从 MCPMark 获取多模型轨迹并统一格式，GUI 域从 OSWorld 采集——多来源是为了避免单一模型的风格偏差被 Judge 当成判别特征。更关键的是刻意挑选"成功轨迹步骤多、失败轨迹步骤少"的样本，打破轨迹长度与成功率的天然相关性，否则 Judge 会偷懒地用长度当捷径。所有标签经规则验证加人工复审双重把关，保证正负样本质量。

**3. 环境重建与交互式评估：给评估者一个能真动手取证的真实环境**

DS 域在本地回放轨迹得到最终环境状态，GUI 域则在隔离的 AWS 实例上重建。Judge Agent 从这个最终状态出发，通过文件操作、数据库查询、GUI 检查等工具调用主动取证，而不是对着静态文本猜。这套可交互环境是整个基准最核心的基础设施——正是"能动手验证"把 Agent-as-a-Judge 和只会读文本的 LLM-as-a-Judge 区分开来。

### 损失函数 / 训练策略

AJ-Bench 是评估基准而非训练框架，主指标为 F1。搜索域在单条目级别聚合后计算 F1，DS 与 GUI 域在轨迹级别计算 F1，所有结果取三次运行平均值。

## 实验关键数据

### 主实验

| 模型 | 是否Agentic | 搜索域 F1 | DS域 F1 | GUI域 F1 | 整体 F1 |
|------|-----------|----------|---------|---------|--------|
| gemini-3-pro | ✗ | 77.0 | 74.5 | 74.2 | 75.1 |
| gpt-5 | ✗ | 73.4 | 60.9 | 52.8 | 61.0 |
| deepseek-v3.2 | ✗ | 63.3 | 63.3 | 66.1 | 64.5 |
| gpt-5-mini | ✓ | 70.8 | 67.4 | 76.8 | 72.4 |
| deepseek-v3.2 | ✓ | 77.3 | 72.7 | 80.5 | **77.3** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 交互轮次=5 vs 20 | F1: ~65 vs ~77 | 更多交互轮次持续提升性能 |
| 仅可访问性树 | GUI F1 变化不一 | PPT 任务足够，Word 不够 |
| 仅截图 | Word F1 最佳 | 不同任务最优模态不同 |
| 混合模态 | Excel F1 最佳 | 多模态并不总是更好 |

### 关键发现
- Agent-as-a-Judge 比同模型的 LLM-as-a-Judge 平均提升约 13 个百分点 F1，在 GUI 域提升最大（可达 31 个百分点）
- 弱模型 + 工具使用 > 强模型无工具：gpt-5-mini (agentic) 整体 F1 72.4 超越 gpt-5 (non-agentic) 的 61.0
- 增加推理努力不一定提升 Agent-as-a-Judge 性能：deepseek-v3.2 的 thinking 模式反而比无 thinking 差 0.23 F1
- 多模态输入不总是有益：混合输入可能引入噪声，不同子任务的最优模态不同

## 亮点与洞察
- "弱模型+工具 > 强模型无工具"是一个重要发现，说明 Agent-as-a-Judge 的价值不在于模型能力本身，而在于环境交互带来的信息增益
- 增加推理努力反而可能降低性能的发现说明，Agent-as-a-Judge 需要的不是更深的思考，而是更好的工具使用能力
- 任务域的设计（搜索/DS/GUI）分别对应信息获取/状态验证/过程验证三种核心能力，为后续研究提供了清晰的能力分类框架

## 局限与展望
- 大部分任务改编自已有基准而非从零构建，覆盖面有限
- 搜索域依赖外部网络环境，网络不稳定影响评估一致性
- 当前绝对性能仍有较大提升空间（最佳 F1 约 0.77），Agent-as-a-Judge 远未饱和
- 未来可扩展到更多领域（如科学验证、代码审查）并增大数据规模用于训练

## 相关工作与启发
- **vs RewardBench/RM-Bench**: 这些基准评估 LLM-as-a-Judge，不涉及环境交互；AJ-Bench 首次系统评估环境感知的 Agent-as-a-Judge
- **vs DevAI (Agent-as-a-Judge)**: DevAI 仅覆盖代码验证单一领域；AJ-Bench 覆盖三个领域且支持多模态
- **vs AgentRewardBench**: 评估 Judge 对 Agent 轨迹的判断但不提供环境交互能力

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统评估 Agent-as-a-Judge 的基准
- 实验充分度: ⭐⭐⭐⭐ 多模型对比、消融充分，但 agentic 模型数量受限
- 写作质量: ⭐⭐⭐⭐ 结构清晰，基准设计动机阐述充分
- 价值: ⭐⭐⭐⭐⭐ 填补了 Agent 评估基础设施的重要空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](../../ICLR2026/llm_evaluation/talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)
- [\[ACL 2026\] IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation](if-rewardbench_benchmarking_judge_models_for_instruction-following_evaluation.md)
- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home Agents](../../ICLR2026/llm_evaluation/simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_agents.md)
- [\[ACL 2026\] arXiv2Table: Toward Realistic Benchmarking and Evaluation for LLM-Based Literature-Review Table Generation](arxiv2table_toward_realistic_benchmarking_and_evaluation_for_llm-based_literatur.md)
- [\[ACL 2026\] Finch: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows](finch_benchmarking_finance_amp_accounting_across_spreadsheet-centric_enterprise_.md)

</div>

<!-- RELATED:END -->
