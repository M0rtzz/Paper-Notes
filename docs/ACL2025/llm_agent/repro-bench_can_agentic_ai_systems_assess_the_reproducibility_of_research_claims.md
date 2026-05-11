---
title: >-
  [论文解读] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?
description: >-
  [ACL 2025][LLM Agent] 本文提出 REPRO-Bench，一个包含 112 个社会科学论文实例的基准，用于评估 AI Agent 自动化评估论文可重复性的能力；现有最佳 Agent 准确率仅 21.4%（低于随机猜测的 25%），作者进一步开发的 REPRO-Agent 将准确率提升至 36.6%（71% 相对提升）。
tags:
  - ACL 2025
  - LLM Agent
---

# REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?

**会议**: ACL 2025  
**arXiv**: 无  
**代码**: 无  
**领域**: LLM智能体  

## 一句话总结

本文提出 REPRO-Bench，一个包含 112 个社会科学论文实例的基准，用于评估 AI Agent 自动化评估论文可重复性的能力；现有最佳 Agent 准确率仅 21.4%（低于随机猜测的 25%），作者进一步开发的 REPRO-Agent 将准确率提升至 36.6%（71% 相对提升）。

## 背景与动机

1. **社会科学可重复性危机严峻**：大规模重复实验表明，SSRP 平台上不到 40% 的论文被认为完全可重复，25% 的论文包含编码错误。
2. **人工评估成本极高**：347 名社会科学家参与重复 110 篇论文，Psychology Reproducibility Project 耗时超 5 年完成 100 项研究重复。
3. **AI Agent 展现自动化潜力**：随着 LLM 驱动的 Agent 在复杂任务上表现出色，自动化可重复性评估成为可能。
4. **现有 benchmark 存在三大不足**：(1) 仅关注代码执行而不检查结果与论文的一致性；(2) 过度简化真实场景（提供预处理上下文）；(3) 缺乏数据格式和编程语言的多样性。
5. **社会科学论文的特殊复杂性**：涉及多种编程语言（Stata、R、Python、MATLAB）和数据格式（.dta、.csv、.rda、.xlsx），需要跨领域知识整合。
6. **批判性推理能力的缺失**：现有 benchmark 未要求识别代码/数据不一致性，而这正是可重复性评估的核心。

## 方法详解

### REPRO-Bench 任务定义

每个任务实例包含三个输入：
1. **论文 PDF**：完整社会科学论文
2. **重复包**（reproduction package）：包含数据、代码和文档
3. **主要发现列表**：从重复报告中提取的需验证项（表格、图表、文本声明）

Agent 需根据评分标准输出可重复性分数（1-4 分）：
- **1 分**：主要发现不可重复
- **2 分**：代码中存在轻微不一致或错误（如变量编码问题），但不影响核心结论
- **3 分**：分析计算正确，但存在显示/报告层面的轻微问题（如四舍五入误差）
- **4 分**：主要发现完全可重复

### 数据收集

从 4 个来源收集 112 篇论文，遵循统一标准 $\mathcal{C}$（社会科学领域、有效 DOI、公开重复包、可信重复报告、重复时间 < 2 小时）：

| 来源 | 数量 | 特点 |
|------|------|------|
| Mass Reproducibility (Brodeur et al.) | 92 | 主要来源，大部分论文基本可重复 |
| I4R Discussion Paper Series | 11 | 包含发现关键重复性问题的论文 |
| Retraction Watch Database | 7 | 已撤回论文（包含数据/分析错误） |
| Twitter/X | 2 | 社交媒体上指出问题的论文 |

分数分布平衡：Score 1+2 共 56 篇 vs Score 3+4 共 56 篇。

### 数据统计

- 论文平均 29 页，重复包平均 4.2 GB、142 个文件
- 每篇论文平均 5 个主要发现（范围 1-19）
- 编程语言：63 篇 Stata、25 篇 R、15 篇多语言、2 篇 Python、1 篇 MATLAB、1 篇 Julia
- 数据格式：34 篇 .dta、11 篇 .csv、10 篇 .rda、51 篇多格式

论文特征（页数、文件数、语言/格式多样性等）与可重复性分数的 Spearman 相关系数均 $|\rho| < 0.1$，表明这些因素不影响可重复性。

### Agent 环境设计

- Agent 从包含 `paper.pdf` 和 `reproduction_package/` 的目录启动
- 预装所有必要软件（Stata、MATLAB、LaTeX）
- 可自由执行命令行操作和安装包
- 通过标准输出/错误流获取反馈
- API 费用上限：每个任务 $4

### 评测 Agent

| Agent | 类型 | 特点 |
|-------|------|------|
| AutoGPT | 通用 Agent | 长期规划、工具选择、行为反思 |
| CORE-Agent | 科学论文 Agent | 专为论文重复设计，含 VLM 工具 |
| SWE-Agent | 软件工程 Agent | 解决 GitHub Issues，含 ACI 接口 |

三个 Agent 均使用 gpt-4o-2024-05-13。

## 实验结果

### 主要性能对比

| Agent | 准确率 (%) | 可用率 (%) | 平均成本 ($) |
|-------|-----------|-----------|-------------|
| AutoGPT | 20.5 | 60.7 | 2.03 |
| CORE-Agent | **21.4** | 46.4 | 2.00 |
| SWE-Agent | 1.8 (调整后 10.7) | 1.8 (调整后 19.6) | 1.20 |
| REPRO-Agent | **36.6** | **92.9** | — |

最佳 Agent CORE-Agent 准确率仅 21.4%，甚至低于四选一随机猜测的 25%。REPRO-Agent 通过结构化模板、虚拟分数回退和少样本示例三大策略，实现 36.6% 准确率（71% 相对提升）和 92.9% 可用率。

### 按编程语言的准确率分析

- **R 任务优于 Stata 任务**：R 作为开源语言，LLM 对其有更好的知识覆盖
- **多语言任务表现最差**：Agent 难以保证跨语言一致执行
- **多格式数据不影响性能**：Agent 能有效利用数据加载器处理多种格式

### 按可重复性分数的表现

所有 Agent 在 Score 4（完全可重复）的论文上表现最好，在 Score 2 和 3 的细粒度判断上表现较差。Agent 倾向于生成二元结果而非深入调查不一致性来源。

### 失败原因分类（对错分类 Score 4 为 Score 1 的案例）

| 失败类型 | 描述 | 占比 |
|---------|------|------|
| Type 4：文件定位失败 | Agent 无法正确推断目录结构找到数据文件 | **最高** |
| Type 3：依赖安装失败 | 无法正确安装所需库 | 次高 |
| Type 2：终端输出遗漏 | Stata 错误信息存于日志而非终端，Agent 误判 | 中等 |
| Type 1：结果比较错误 | Agent 编写的比较脚本本身有误 | 较低 |

### 未识别不一致性的原因（Score 1 误分为 Score 4）

1. Agent 未严格遵循完整工作流——不到 42% 的案例同时包含代码检查和结果比较阶段
2. 代码检查时 Agent 常读取整个文件而非聚焦关键段落，长代码上下文下难以定位错误

## 亮点

- **填补真实评估 benchmark 空白**：首个要求 Agent 端到端评估论文可重复性的基准，不同于仅关注代码执行的已有 benchmark。
- **真实复杂度**：直接使用实际社会科学论文和重复包，涉及多语言（Stata/R/Python/MATLAB）和多格式数据。
- **系统化失败分析**：将 Agent 失败归纳为 4 类可操作性强的错误类型，为 Agent 改进提供明确方向。
- **REPRO-Agent 验证了分析价值**：基于经验分析的三大策略实现 71% 相对提升，证明针对性改进的有效性。
- **社会科学影响力**：法律专家确认 benchmark 捕获了社会科学论文的代表性模式，可促进更好的代码和数据管理。

## 局限性

- **准确率仍然很低**：即便 REPRO-Agent 的 36.6% 也远不足以实际应用，自动化可重复性评估仍需大幅改进。
- **缺乏任务实例变体**：未引入同一论文的多版本（含故意错误/修正），无法细粒度评估检测能力。
- **仅限社会科学领域**：未扩展到生物学等同样面临可重复性挑战的其他学科。
- **单一 LLM 后端**：三个 Agent 均使用 gpt-4o，未探索其他模型或更大上下文窗口的影响。
- **成本约束限制深度探索**：每个任务 $4 的 API 预算可能不足以支撑更深入的代码分析和结果比较。

## 评分与评价

- **新颖性** ⭐⭐⭐⭐：首个面向社会科学可重复性评估的 Agent benchmark，任务设计和数据来源构思独到。
- **技术深度** ⭐⭐⭐：benchmark 构建过程严谨，但 REPRO-Agent 本身的技术改进相对简单（模板+回退+少样本）。
- **实验充分度** ⭐⭐⭐⭐：三个代表性 Agent + 详尽的定性分析 + 失败原因分类 + 改进验证，分析链条完整。
- **写作质量** ⭐⭐⭐⭐：结构清晰，数据收集标准明确，统计分析到位。
- **实用价值** ⭐⭐⭐⭐⭐：直接面向社会科学界面临的真实问题，benchmark 公开可用，对 Agent 发展有重要指导意义。
- **综合评分** ⭐⭐⭐⭐（4/5）

## 相关引用

- Brodeur et al. (2024): 大规模社会科学可重复性实验
- SWE-Bench (Jimenez et al., 2024): 软件工程 Agent benchmark
- CORE-Bench (Siegel et al., 2024): 科学论文重复 benchmark
- AutoGPT (Gravitas, 2023): 通用自主 Agent

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_social_science_.md)
- [\[ACL 2025\] AndroidGen: Building an Android Language Agent under Data Scarcity](androidgen_agent_data_scarcity.md)
- [\[ACL 2025\] OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use](os_agents_a_survey_on_mllm-based_agents_for_general_computing_devices_use.md)
- [\[ACL 2025\] Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems](agentic_reward_modeling_integrating_human_preferences_with_verifiable_correctnes.md)
- [\[ACL 2025\] Can a Single Model Master Both Multi-turn Conversations and Tool Use? CoALM: A Unified Conversational Agentic Language Model](can_a_single_model_master_both_multi-turn_conversations_and_tool_use_coalm_a_uni.md)

</div>

<!-- RELATED:END -->
