---
title: >-
  [论文解读] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science?
description: >-
  [ACL2025][LLM Agent][AI Agent] 提出 REPRO-Bench，包含 112 个社会科学论文的可复现性评估任务，发现现有 AI Agent（最高准确率仅 21.4%）远不足以自动化该流程，并据此开发 REPRO-Agent 将准确率提升至 36.6%。
tags:
  - ACL2025
  - LLM Agent
  - AI Agent
  - 可复现性评估
  - 社会科学
  - benchmark
  - 代码执行
---

# REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science?

**会议**: ACL2025  
**arXiv**: [2507.18901](https://arxiv.org/abs/2507.18901)  
**代码**: [GitHub](https://github.com/uiuc-kang-lab/REPRO-Bench)  
**领域**: llm_agent  
**关键词**: AI Agent, 可复现性评估, 社会科学, benchmark, 代码执行

## 一句话总结
提出 REPRO-Bench，包含 112 个社会科学论文的可复现性评估任务，发现现有 AI Agent（最高准确率仅 21.4%）远不足以自动化该流程，并据此开发 REPRO-Agent 将准确率提升至 36.6%。

## 背景与动机
1. **可复现性危机严峻**：社会科学领域长期面临"可复现性危机"，SSRP 平台上不到 40% 的论文能达到完全可复现（Level 10），25% 含有编码错误。
2. **人工评估成本巨大**：Mass Reproduction 项目动员 347 位社会科学家花费数年才复现 110 篇论文，人力与时间成本不可持续。
3. **AI Agent 能力待验证**：LLM 驱动的 Agent（AutoGPT、SWE-Agent 等）在软件工程任务上表现突出，但能否处理社会科学特有的复杂复现流程尚未被系统评估。
4. **现有 Benchmark 假设过强**：SciCode 和 CORE-Bench 均假设论文完全可复现，无法评估 Agent 识别不一致性的能力。
5. **上下文过度简化**：已有基准提供预提取的具体步骤，而真实复现需要 Agent 从原始 PDF 和数据包中自主提取信息。
6. **语言与格式单一**：已有基准仅涉及单一编程语言（Python/R），而社会科学论文常混合 Stata、R、MATLAB 等多种语言和 .dta/.csv/.rda 等多种数据格式。

## 方法详解

### REPRO-Bench 构建
- **数据来源**：从 4 个来源收集 112 篇论文——(1) Mass Reproduction 92 篇、(2) I4R Discussion Papers 11 篇、(3) Retraction Watch 7 篇、(4) Twitter/X 2 篇，确保可复现与不可复现论文均衡覆盖。
- **评分体系**：4 级可复现性评分——1=主要发现不可复现，2=代码存在轻微不一致/错误，3=仅显示层面问题（如四舍五入），4=完全可复现。分布为 20/36/8/48，1+2 与 3+4 各 56 篇。
- **任务定义**：Agent 接收论文 PDF + 复现包（代码/数据/文档）+ 主要发现列表，需输出 1-4 整数评分。

### 评估的 Agent
- **AutoGPT**：通用型 Agent，具备长期规划、工具选择和反思能力。
- **CORE-Agent**：专为论文复现设计，含 VLM 工具，适配其 hard task 版本。
- **SWE-Agent**：软件工程 Agent，利用 Agent-Computer Interface (ACI) 执行调试。
- 三者均使用 GPT-4o（2024-05-13），每任务 API 消耗上限 $4。

### Agent 环境与评估协议
- **环境**：每个 Agent 起始目录含 paper.pdf 和 reproduction_package/ 子目录，user prompt 包含待复现的主要发现列表。
- **软件预装**：Stata、MATLAB、LaTeX 等社会科学常用工具均已预装，版本信息在任务描述中指定。
- **评估指标**：主指标为准确率（生成评分是否匹配 ground truth）；辅助指标为适用率（输出文件格式和位置是否正确）。
- **成功 Agent 工作流**（4 阶段）：Phase 1 理解环境（列目录、读论文、读 README）→ Phase 2 代码审查（检查不一致性）→ Phase 3 编辑并执行脚本 → Phase 4 比较执行结果与原始结果。

### REPRO-Agent 改进策略
基于对失败模式的系统分析，设计三项策略：(1) 基于成功案例构建结构化模板引导规划；(2) 引入 dummy score 作为兜底机制提升适用率；(3) 将常见错误类型作为 few-shot 示例增强上下文学习。

### Benchmark 统计特征
- 论文平均 29 页，复现包平均 4.2GB、142 个文件。
- 每篇论文平均 5 个主要发现（范围 1-19，标准差 4）。
- 编程语言：Stata 63 篇、R 25 篇、多语言 15 篇、MATLAB/Julia/Python 各少量。
- 数据格式：.dta 34 篇、.csv 11 篇、多格式 51 篇。
- Spearman 相关性分析（|ρ|<0.1）证实页数、发现数、包大小、语言/格式多样性均与可复现性评分无关。

## 实验关键数据

### 表1：Agent 整体性能与成本

| Agent | 准确率 (%) | 适用率 (%) | 平均成本 ($) |
|:--|:--:|:--:|:--:|
| AutoGPT | 20.5 | 60.7 | 2.03 |
| CORE-Agent | 21.4 | 46.4 | 2.00 |
| SWE-Agent | 1.8 (调整后 10.7) | 1.8 (调整后 19.6) | 1.20 |
| **REPRO-Agent** | **36.6** | **92.9** | - |

- 最优 Agent 准确率 21.4% 甚至低于随机猜测的 25%，凸显任务难度。
- REPRO-Agent 比 CORE-Agent 准确率相对提升 71%，适用率相对提升 53%。

### 表2：不同维度的性能分析

| 分析维度 | 关键发现 |
|:--|:--|
| 按评分 | Agent 在 Score 4（完全可复现）上显著更好，倾向给出二元判断 |
| 按语言 | R 代码任务准确率远高于 Stata，因 LLM 对 R 更熟悉 |
| 多语言 | 涉及多语言的任务准确率下降，Agent 难以跨语言一致执行 |
| 数据格式 | 单格式 vs 多格式准确率相近（54% vs 52%），格式多样性非瓶颈 |

### 失败原因分类
- **Type 1**（结果比较错误）：Agent 编写的比较脚本有误，将一致结果错判为不匹配。
- **Type 2**（输出捕获失败）：Stata 将错误信息写入 log 文件而非终端，Agent 未能读取。
- **Type 3**（库安装失败）：无法正确安装依赖。
- **Type 4**（文件定位失败，最常见）：复现包目录结构复杂，Agent 无法正确定位数据文件。

## 亮点
- 首个面向**社会科学可复现性评估**的端到端 Agent Benchmark，112 个任务实例贴近真实场景。
- 4 级评分体系细粒度覆盖不同可复现性层级，避免了简单的二元判断。
- 对 Agent 失败模式的**系统分类**（4 种类型）为后续改进提供了明确方向。
- REPRO-Agent 验证了"从失败分析到改进策略"的闭环有效性，71% 的相对提升令人信服。

## 局限与展望
- 每篇论文仅有一个任务版本，缺乏含故意错误/修正代码的变体来进一步测试 Agent 鲁棒性。
- 未探索更难场景（如遮蔽实验结果数据点，仅提供原始数据）。
- 仅覆盖社会科学领域，未扩展至生物学等同样面临可复现性问题的领域。
- REPRO-Agent 36.6% 的准确率距实际应用仍有很大差距。
- 标注过程中主要发现的提取虽经共识验证，但未报告 inter-annotator agreement 指标。

## 与相关工作的对比
- **SciCode**：将论文发现转化为编码问题，假设发现完全正确，仅涉及 Python——REPRO-Bench 无此假设且涵盖多语言。
- **CORE-Bench**：提供具体预定义步骤，以代码执行结果为 ground truth——REPRO-Bench 要求 Agent 自主从 PDF 提取并判断一致性。
- **SWE-Bench**：关注 GitHub issue 修复，代码仓库结构规范——REPRO-Bench 的复现包结构不规范，对 Agent 导航能力要求更高。
- **Mass Reproduction (Brodeur et al., 2024)**：REPRO-Bench 直接建立在该大规模人工复现工作之上，用其复现报告作为 ground truth。

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次系统评估 AI Agent 在社会科学可复现性评估上的能力，任务定义独特
- 实验充分度: ⭐⭐⭐⭐ — 3+1 个 Agent、多维度分析、失败原因分类完整，但缺少更多 LLM backbone 对比
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，数据收集与评分标准描述详尽
- 价值: ⭐⭐⭐⭐ — 填补了 AI 辅助可复现性评估的 Benchmark 空白，对社会科学研究流程有实际意义

<!-- RELATED:START -->

## 相关论文

- [REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)
- [LegalAgentBench: Evaluating LLM Agents in Legal Domain](legalagentbench_evaluating_llm_agents_in_legal_domain.md)
- [PaSa: An LLM Agent for Comprehensive Academic Paper Search](pasa_an_llm_agent_for_comprehensive_academic_paper_search.md)
- [ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use_benchmark.md)
- [GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents](guidebench_guideline_following.md)

<!-- RELATED:END -->
