---
title: >-
  [论文解读] AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage
description: >-
  [ACL 2026][多智能体][论文复现] AutoReproduce 提出了一个多智能体框架，通过"论文谱系"算法从引用文献中挖掘隐式领域知识，实现端到端的论文实验自动复现，在自建基准 ReproduceBench 上的代码执行率达 94.87%，性能差距仅 19.72%。
tags:
  - "ACL 2026"
  - "多智能体"
  - "论文复现"
  - "论文谱系"
  - "代码生成"
  - "科研自动化"
---

# AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage

**会议**: ACL 2026  
**arXiv**: [2505.20662](https://arxiv.org/abs/2505.20662)  
**代码**: [https://github.com/AI9Stars/AutoReproduce](https://github.com/AI9Stars/AutoReproduce)  
**领域**: LLM评测  
**关键词**: 论文复现, 论文谱系, 多智能体, 代码生成, 科研自动化

## 一句话总结

AutoReproduce 提出了一个多智能体框架，通过"论文谱系"算法从引用文献中挖掘隐式领域知识，实现端到端的论文实验自动复现，在自建基准 ReproduceBench 上的代码执行率达 94.87%，性能差距仅 19.72%。

## 研究背景与动机

**领域现状**：论文实验复现对加速科学进步至关重要，但随着方法日益复杂，复现需要深厚的领域专业知识和大量人力。LLM 已被用于论文分析、想法生成、环境配置等离散任务，但端到端的自动复现框架尚未出现。

**现有痛点**：(1) 论文中常常缺少关键实验细节——不同研究领域依赖于大量隐性知识（如特定模块架构、数据处理流程）；(2) 并行工作如 Paper2Code 仅生成代码而不考虑可执行性，无法验证复现的正确性；(3) 现有方法未系统利用引用文献中蕴含的领域惯例和实现实践。

**核心矛盾**：成功复现不仅需要理解论文本身的方法描述，还需要掌握论文未明确说明的领域常规实践——这些"默会知识"分散在引用文献和相关代码库中。

**本文目标**：(1) 从引用文献中系统挖掘隐式知识；(2) 构建端到端的可执行代码复现框架；(3) 建立包含执行验证的复现评估基准。

**切入角度**：提出"论文谱系"（Paper Lineage）算法，追溯引用文献和关联代码库，将历史研究中积累的实现惯例作为复现的知识来源。

**核心 idea**：论文复现 = 论文理解 + 领域知识挖掘 + 代码生成 + 执行验证，谱系算法通过引用链传递的隐性知识弥补了论文自身描述的不足。

## 方法详解

### 整体框架

AutoReproduce 分三个阶段：(1) 文献综述——研究代理对论文进行三级摘要（总体/方法/实验）；(2) 论文谱系——从引用中识别 top-k 相关论文，获取其代码库并提取关键文件；(3) 代码开发——研究代理和代码代理协作，经过数据获取、方法复现、实验执行三步生成可执行代码。

### 关键设计

1. **论文谱系算法**:

    - 功能：从引用文献中挖掘隐式领域知识和实现惯例
    - 核心思路：研究代理从源论文的引用中识别 top-k（默认 3）最相关论文，优先选择主实验部分的对比基线。通过 ArXiv API 获取论文并摘要，通过 GitHub API 克隆代码库。代码代理根据论文摘要和任务说明从代码库中选择性提取关键源文件，构建 ⟨summary, code⟩ 元组作为参考示例。无公开代码的论文仅使用其摘要作为知识来源
    - 设计动机：科研是累积过程，新方法建立在已有研究基础上，引用链中的代码库包含了未在论文中明确说明的实现标准

2. **三阶段代码开发**:

    - 功能：从数据处理到方法实现到实验运行的完整复现
    - 核心思路：(a) 数据获取——区分标准基准和自定义数据集，用 mini-batch 采样推断关键数据属性（tensor shape, dtype）；(b) 方法复现——代码代理生成实现代码，研究代理对比论文摘要验证并提供纠正反馈；(c) 实验执行——验证完整实验管道，使用 early-exit 机制快速验证。错误诊断和代码编辑解耦为两步
    - 设计动机：解耦错误分析和代码修改显著提高了调试成功率

3. **基于采样的单元测试**:

    - 功能：快速验证生成代码的可执行性
    - 核心思路：在方法复现阶段，代码代理通过 mini-batch 采样推断数据流属性，用 EDIT 命令进行精确的行级修改而非重新生成整个文件
    - 设计动机：减少 token 生成开销，避免全文件重生成的不稳定性

### 损失函数 / 训练策略

不涉及模型训练。使用 GPT-4o/Claude-3.5-Sonnet/o3-mini/Gemini-2.5-Pro 等 LLM 作为代理骨架。

## 实验关键数据

### 主实验

**ReproduceBench 评估**

| 方法 | LLM | Align-Score | Exec Rate | Perf Gap (↓) |
|------|-----|-------------|-----------|-------------|
| ChatDev | GPT-4o | 43.33 | 2.56% | 99.62% |
| Agent Lab | GPT-4o | 48.64 | 23.08% | 82.31% |
| PaperCoder | o3-mini | 60.26 | 17.94% | 89.23% |
| AutoReproduce | GPT-4o | 56.24 | **76.92%** | 41.77% |
| AutoReproduce | o3-mini | 75.21 | **92.31%** | 24.31% |
| AutoReproduce | Gemini-2.5-Pro | **77.56** | **94.87%** | **19.72%** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整 AutoReproduce | 最优 | 谱系 + 三阶段开发 |
| 无论文谱系 | 下降 | 缺少领域知识导致实现偏差 |
| 无单元测试 | Exec Rate 下降 | 可执行性验证缺失 |

### 关键发现

- AutoReproduce 的代码执行率（94.87%）远超所有基线（最高 23.08%），说明端到端的可执行性验证至关重要
- 论文谱系算法是关键贡献——移除后 Align-Score 和 Perf Gap 均显著下降
- Gemini-2.5-Pro 作为骨架 LLM 表现最优，但即使使用 GPT-4o，AutoReproduce 也大幅超越 PaperCoder
- 性能差距仍有 19.72%，说明完全自动化的高保真复现仍有挑战

## 亮点与洞察

- "论文谱系"的概念非常有洞察力——将科研的累积性质转化为可操作的知识挖掘算法
- 端到端可执行性的强调填补了现有工作（如 Paper2Code）的关键空白——不能执行的代码没有复现价值
- 解耦错误诊断和代码修改的策略是重要的工程洞察

## 局限与展望

- ReproduceBench 仅包含 13 篇论文，规模较小
- 依赖论文有公开代码库的引用，否则谱系算法退化为仅使用文本知识
- 性能差距仍有约 20%，高精度复现仍需人类参与
- 仅覆盖 AI 领域，扩展到其他学科需要额外适配

## 相关工作与启发

- **vs Paper2Code/PaperCoder**: 这些方法不考虑代码可执行性，AutoReproduce 强调端到端执行
- **vs Agent Laboratory**: Agent Lab 的执行率仅 23%，AutoReproduce 达到 95%

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 论文谱系算法和端到端复现框架都是重要创新
- 实验充分度: ⭐⭐⭐⭐ 多 LLM 对比和多基线覆盖，但基准规模小
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，流程图直观
- 价值: ⭐⭐⭐⭐⭐ 对科研自动化有重大推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)
- [\[ACL 2026\] Preference Estimation via Opponent Modeling in Multi-Agent Negotiation](preference_estimation_via_opponent_modeling_in_multi-agent_negotiation.md)
- [\[ACL 2026\] Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate](dialectic-med_mitigating_diagnostic_hallucinations_via_counterfactual_adversaria.md)
- [\[ACL 2026\] EvoSpark: Endogenous Interactive Agent Societies for Unified Long-Horizon Narrative Evolution](evospark_endogenous_interactive_agent_societies_for_unified_long-horizon_narrati.md)
- [\[ACL 2026\] Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)

</div>

<!-- RELATED:END -->
