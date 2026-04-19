---
title: >-
  [论文解读] VeriMaAS: Automated Multi-Agent Workflows for RTL Design
description: >-
  [NeurIPS 2025][RTL code generation] VeriMaAS 提出自动组合多 Agent 工作流的框架用于 RTL 代码生成，核心创新是将 HDL 工具的形式化验证反馈（Yosys 综合 + OpenSTA 时序分析）直接整合到工作流编排中，在 VeriThoughts 上 pass@1 提升 2-12%，且仅需数百样本做控制器调优，比全量微调训练数据少一个量级。
tags:
  - NeurIPS 2025
  - RTL code generation
  - multi-agent workflow
  - formal verification
  - HDL
  - EDA
---

# VeriMaAS: Automated Multi-Agent Workflows for RTL Design

**会议**: NeurIPS 2025  
**arXiv**: [2509.20182](https://arxiv.org/abs/2509.20182)  
**代码**: https://github.com/ (有，论文中提及)  
**领域**: 代码生成 / 硬件设计自动化  
**关键词**: RTL code generation, multi-agent workflow, formal verification, HDL, EDA

## 一句话总结

VeriMaAS 提出自动组合多 Agent 工作流的框架用于 RTL 代码生成，核心创新是将 HDL 工具的形式化验证反馈（Yosys 综合 + OpenSTA 时序分析）直接整合到工作流编排中，在 VeriThoughts 上 pass@1 提升 2-12%，且仅需数百样本做控制器调优，比全量微调训练数据少一个量级。

## 研究背景与动机

**RTL 代码生成的两难困境**。现有方法主要有两条路线：(1) 对 LLM 进行 RTL/HDL 数据微调，但需要大量 GPU 资源和数万级训练样本，且泛化性差；(2) 使用 frontier 推理模型（如 o4），虽然无需微调但推理成本极高。两条路都面临 HDL 领域数据远少于通用编程语言的根本限制。

**多 Agent 工作流的新范式**。近期 MaAS、AFlow 等工作证明了自动化多 Agent 工作流在 QA 和通用编程任务上的优势，但这些方法主要聚焦于"通用知识"领域（如维基百科问答、数学竞赛），在 RTL 设计这样的专业领域中，简单的 prompt 策略（如 Debate）难以直接生效。

**核心洞察与切入点**。RTL 设计的独特优势在于拥有成熟的形式化验证工具链——Yosys 综合器和 OpenSTA 时序分析器可以提供精确的对错判断。VeriMaAS 的核心 idea 是将这些验证反馈直接嵌入工作流编排过程，让 Agent 控制器根据设计的编译/综合结果动态调整推理策略，而非依赖 LLM 自身的代码判断能力。

## 方法详解

### 整体框架

VeriMaAS 的工作流为：给定 RTL 设计任务 → 自适应采样 Agent 算子生成 K=20 个候选 Verilog 设计 → 通过 Yosys 综合 + OpenSTA 时序/功耗分析进行验证 → 将验证日志/错误信息反馈给控制器 → 控制器决定是否升级到更复杂的推理算子或终止 → 返回候选设计池。

### 关键设计

1. **级联式 Agent 控制器（Cascading Controller）**:

    - 功能：根据任务难度自适应选择推理算子的复杂度
    - 核心思路：定义算子复杂度序列 I/O → CoT → ReAct → SelfRefine → Debate，每个阶段计算一个置信度分数 $s_c$（基于当前候选设计的验证失败率），若失败率超过阈值 $\tau_c$ 则升级到下一级算子
    - 设计动机：不同难度的 RTL 任务需要不同层级的推理——简单模块用 zero-shot 即可，复杂模块才需要多轮反思或辩论，级联设计避免对所有任务都使用最昂贵的策略

2. **形式化验证反馈闭环**:

    - 功能：为 Agent 提供精确的设计正确性信号
    - 核心思路：每个阶段的 K=20 个 Verilog 候选经过完整的 EDA 工具链评估——Yosys 做综合和面积估计、OpenSTA 做时序和静态功耗分析（使用 Skywater 130nm PDK），验证日志和错误信息直接反馈给控制器和后续推理
    - 设计动机：传统 LLM 代码生成依赖模型自身判断代码质量，但 HDL 验证需要精确的工具反馈；EDA 工具提供的二值化信号（通过/失败）比 LLM 的模糊自评价更可靠

3. **低成本控制器调优**:

    - 功能：以极低的数据需求完成控制器的策略学习
    - 核心思路：从 VeriThoughts 训练集随机采样 500 个数据点，对每个点运行 K=20 个候选并统计验证失败率，取 20/40/60/80 百分位作为各阶段阈值 $\mathcal{T}$；目标函数为 $\max_{\mathcal{T}} \mathbb{E}[U(\mathcal{T}) - \lambda \cdot C(\mathcal{T})]$，其中 U 为 pass@k，C 为 token 消耗
    - 设计动机：与需要数万样本的全量微调相比，仅需数百样本即可确定阈值，因为控制器只需学习"何时升级策略"而非"如何写 RTL 代码"

### 损失函数 / 训练策略

控制器调优目标为性能-成本的帕累托优化：$\max_{\mathcal{T}} \mathbb{E}_{(q,a)\sim D}[U(\mathcal{T};q,a,\mathbb{O}) - \lambda \cdot C(\mathcal{T};q,a,\mathbb{O})]$，其中 $\lambda=10^{-3}$。PPA 感知优化变体将成本项替换为 Yosys 报告的面积 $C=\text{Area}(\mathcal{T};q,a,\mathbb{O})$，实现功能正确性与物理设计指标的联合优化。

## 实验关键数据

### 主实验
| 模型+方法 | VeriThoughts pass@1 | VeriThoughts pass@10 | VerilogEval pass@1 | VerilogEval pass@10 |
|--------|------|------|----------|------|
| GPT-4o-mini (Instruct) | 80.64 | 90.87 | 50.26 | 61.02 |
| GPT-4o-mini + VeriMaAS | **83.09** (+2.45) | **92.85** (+1.98) | **52.05** (+1.79) | **64.02** (+3.00) |
| Qwen2.5-7B (Instruct) | 44.90 | 82.33 | 22.92 | 51.47 |
| RTLCoder-7B (微调) | - | - | 34.60 | 45.50 |
| Qwen2.5-7B + VeriMaAS | **56.62** (+11.72) | **86.29** (+3.96) | **29.10** (+6.18) | **56.45** (+4.98) |
| Qwen2.5-14B (Instruct) | 67.89 | 94.13 | 33.78 | 62.04 |
| VeriThoughts-14B (微调) | 78.50 | 92.10 | 43.70 | 55.14 |
| Qwen2.5-14B + VeriMaAS | **74.24** (+6.35) | **95.78** (+1.65) | **41.47** (+7.69) | **62.48** (+0.44) |
| Qwen3-14B (Reasoning) | 89.35 | 98.64 | 65.87 | 75.62 |
| Qwen3-14B + VeriMaAS | **92.16** (+2.81) | **98.75** (+0.11) | **66.96** (+1.09) | **75.71** (+0.09) |

### 消融实验
| 配置 (o4-mini 基线) | VeriThoughts pass@1 | Token 成本(k) | 说明 |
|------|---------|------|------|
| + CoT | 94.11 (+0.26) | 1.10 (1.09×) | 轻量级提升 |
| + ReAct | 91.96 (-1.89) | 1.70 (1.68×) | 反而下降 |
| + SelfRefine | 94.31 (+0.46) | 2.24 (2.22×) | 性能好但成本高 |
| + VeriMaAS | 94.09 (+0.24) | 1.21 (1.20×) | 性能接近最优，成本接近最低 |

### 关键发现
- 开源 LLM 上提升最大：Qwen2.5-7B pass@1 从 44.90→56.62（+11.72），超越微调的 RTLCoder-7B 在 pass@10 上的表现
- VeriMaAS 在 pass@10 上与微调基线持平或超越，同时 pass@1 也有提升——说明框架既提高了最佳候选质量也扩大了有效候选池
- 与单一 Agent 策略相比，VeriMaAS 在保持性能的同时 token 成本接近最低（仅 1.20× vs SelfRefine 的 2.22×）
- PPA 感知优化可在几乎不损失 pass@10 的前提下，将面积降低 9-23%、延迟降低 4-21%

## 亮点与洞察
- 形式化验证作为 Agent 反馈源的思路对整个硬件设计 AI 领域有普适价值——EDA 工具提供了软件编程所没有的精确验证信号
- 控制器调优只需数百样本（vs 微调的数万样本），大幅降低了方法的部署门槛
- 级联控制器实现了性能-成本的良好权衡，避免了"一刀切"使用最强策略的浪费
- PPA 感知优化展示了框架的灵活性——可以在不重新训练的情况下切换优化目标

## 局限与展望
- 本文为 NeurIPS ML for Systems Workshop 论文，实验规模相对有限
- 控制器的阈值基于百分位数的简单统计确定，更复杂的学习方法（如 RL）可能进一步提升
- 工作流搜索空间（5 个固定算子的级联序列）相对受限，自由组合可能发现更优策略
- 仅在 RTL 代码生成上验证，向其他 EDA 任务（布局布线、时序修复）的扩展有待探索

## 相关工作与启发
- 与 MaAS/AFlow 的区别：同属自动化多 Agent 工作流，但 VeriMaAS 引入了领域特定的形式化验证反馈，使得通用框架能适配专业硬件设计任务
- 与 RTLCoder/VeriThoughts 微调方法的区别：VeriMaAS 是正交的——可以叠加在任何基座模型上（包括微调模型），两者互补
- 启发：形式化验证反馈的思路可推广到任何有精确检验工具的代码生成领域（如 SQL 验证、数学证明检查）

## 评分
- 新颖性: ⭐⭐⭐⭐ 形式化验证+自动化Agent工作流的组合在RTL领域是首创
- 实验充分度: ⭐⭐⭐ Workshop论文规模，但覆盖了多个模型和两个benchmark
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，实验设计系统
- 价值: ⭐⭐⭐⭐ 对硬件设计自动化有实用参考，低监督成本使其易于部署

<!-- RELATED:START -->

## 相关论文

- [CARD: Towards Conditional Design of Multi-agent Topological Structures](../../ICLR2026/code_intelligence/card_towards_conditional_design_of_multi-agent_topological_structures.md)
- [A Self-Improving Coding Agent](a_selfimproving_coding_agent.md)
- [A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions](a_stochastic_differential_equation_framework_for_multi-objective_llm_interaction.md)
- [SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)
- [CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System](../../ACL2025/code_intelligence/compileagent_automated_real-world_repo-level_compilation_with_tool-integrated_ll.md)

<!-- RELATED:END -->
