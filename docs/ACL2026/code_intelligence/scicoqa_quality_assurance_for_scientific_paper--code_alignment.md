---
title: >-
  [论文解读] SciCoQA: Quality Assurance for Scientific Paper–Code Alignment
description: >-
  [ACL 2026][代码智能][论文代码差异检测] 本文提出 SciCoQA，首个用于检测科学论文与其代码实现之间差异的基准数据集，包含 635 个差异实例（92 个真实 + 543 个合成），评测 22 个 LLM 后发现最强模型仅能检测 46.7% 的真实差异，揭示了自动化科学质量保证中的关键能力缺口。
tags:
  - "ACL 2026"
  - "代码智能"
  - "论文代码差异检测"
  - "科学可复现性"
  - "跨模态验证"
  - "LLM评测"
  - "质量保证"
---

# SciCoQA: Quality Assurance for Scientific Paper–Code Alignment

**会议**: ACL 2026  
**arXiv**: [2601.12910](https://arxiv.org/abs/2601.12910)  
**代码**: [https://github.com/ukplab/scicoqa](https://github.com/ukplab/scicoqa)  
**领域**: 科学可复现性/论文-代码一致性验证  
**关键词**: 论文代码差异检测, 科学可复现性, 跨模态验证, LLM评测, 质量保证

## 一句话总结

本文提出 SciCoQA，首个用于检测科学论文与其代码实现之间差异的基准数据集，包含 635 个差异实例（92 个真实 + 543 个合成），评测 22 个 LLM 后发现最强模型仅能检测 46.7% 的真实差异，揭示了自动化科学质量保证中的关键能力缺口。

## 研究背景与动机

**领域现状**：科学可复现性危机持续困扰学术界，发布代码和数据已成为共识，但代码的可用性并不等于代码与论文描述的一致性。实际中，实现细节常常偏离论文描述——从"mathiness"（方程仅模拟技术深度而真正增益来自未记录的技巧）到评估指标的实现差异（如 BLEU 分数的不同实现导致科学比较无效）。

**现有痛点**：(1) 论文-代码不一致通常只在复现尝试中才被发现，浪费大量资源并侵蚀科学信任；(2) 审稿人已面临严重时间压力，进行细致的代码审查不切实际；(3) 随着"AI 科学家"等自动化系统（自动生成想法、代码和论文）的兴起，人工审查越来越不可行——一个自动生成的代码库可能完美运行且性能优异，但实现的方法与论文描述完全不同。

**核心矛盾**：科学产出正因自动化而指数级增长，但验证论文与代码忠实性的能力仍完全依赖人工。现有评估（如 PaperBench 的人工评分表、通用 LLM 评判或"代码能否运行"）都无法可靠检测论文-代码语义差异。

**本文目标**：构建首个论文-代码差异检测基准，系统评估 LLM 能否自动发现科学论文与代码之间的语义不一致。

**切入角度**：从可复现性社区的"自然发现"入手——利用 GitHub Issues 中用户报告的论文-代码差异和可复现性挑战赛的复现报告作为真实差异来源，并通过合成生成管道将数据扩展到 CS 以外的计算科学领域。

**核心 idea**：将论文-代码差异检测定义为跨模态验证任务（文本 vs 代码），构建包含差异类型分类体系（Difference/Paper Omission/Code Omission）和影响类别分类体系（Algorithm/Model/Loss/Evaluation/Data/Training）的结构化数据集，用以评估 LLM 的长上下文跨模态推理能力。

## 方法详解

### 整体框架

SciCoQA 的构建分为真实数据收集和合成数据生成两条路线。真实数据从 GitHub Issues（1890 个仓库、10636 个 issue 经自动+人工筛选得 59 个差异）和可复现性论文（171 篇复现报告经 GPT-5 提取+人工验证得 65 个差异）中获取，再经 Gemini 3.1 Pro 和 GPT-5 双重验证和标准化描述生成，共得 92 个真实差异。合成数据在 204 个仓库上用 GPT-5 生成代码修改，得 543 个差异，覆盖 CS 及物理、统计等领域。

### 关键设计

1. **论文-代码差异的严格定义与三类型分类**:
    - 功能：建立清晰的差异定义边界，排除无关噪声
    - 核心思路：差异被定义为"论文科学方法描述与代码实现之间的语义冲突，使代码无法忠实复现报告的方法"。分为三种类型——Difference（代码实现逻辑与论文不同，如 L1 vs L2 正则化）、Paper Omission（代码含论文未描述的关键组件）、Code Omission（论文描述的步骤在代码中缺失）。明确排除 bug（与论文描述无关）、可通过 CLI/配置文件解决的超参数差异、以及标准工程实践（如数值稳定性的噪声添加）。
    - 设计动机：没有清晰定义就无法构建一致的标注标准。排除工程细节和 bug 确保数据集聚焦于影响科学有效性的语义差异。

2. **六类别影响分类体系**:
    - 功能：描述差异影响了研究流水线的哪个环节
    - 核心思路：定义六个类别——Algorithm（步骤顺序/操作/核心逻辑）、Model（架构/权重初始化）、Loss（损失定义/项）、Evaluation（评估逻辑/指标）、Data（数据使用/预处理/增广）、Training（学习过程/调度/优化）。真实数据中 Algorithm（25%）和 Loss（24%）主导，合成数据中 Algorithm（26%）和 Model（21%）主导。
    - 设计动机：知道差异的类型（what）和影响范围（where）有助于理解不同差异的严重程度和检测难度。

3. **合成数据生成管道**:
    - 功能：将数据集从 CS/AI 扩展到物理、统计、定量生物学等计算科学领域
    - 核心思路：从 GitHub 爬取的链接 arXiv 论文且使用宽松许可证的仓库中采样 204 个，用 GPT-5 基于论文和代码生成 5 个代码 diff（按差异定义约束），每个仓库最多采样 3 个不操作同一文件且可精确匹配原始代码的修改。真实与合成数据的检测率相关性达 $r = 0.94$，验证了合成数据作为模型排名代理的可靠性。
    - 设计动机：真实差异自然稀少且局限于 CS/AI，合成管道解决了数据规模和领域覆盖的瓶颈，同时可用于生成不在模型预训练语料中的数据以对抗数据污染。

### 损失函数 / 训练策略

本文为评测基准论文，不涉及模型训练。评估使用 LLM-as-Judge（GPT-OSS 20B 推理模型），在人工标注的 1039 个样本上达到 F1 87.5%。模型被提示生成差异列表，然后解析为独立差异进行匹配评估。

## 实验关键数据

### 主实验

| 模型 | 真实数据召回率 | 合成数据召回率 | 平均召回率 |
|------|-------------|-------------|----------|
| Gemini 3.1 Pro | 46.7% | — | — |
| GPT-5 Mini | 46.7% | — | — |
| GPT-5 | — | 70.0% | — |
| Nemotron 49B | — | — | 23.9% |
| Qwen3 30B Coder | — | — | 23.5% |

| 模型 | Precision | Recall | F1 |
|------|-----------|--------|----|
| GPT-5 | 88.0 | 51.2 | 64.7 |
| Gemini 2.5 Pro | 94.6 | 41.1 | 57.3 |
| GPT-OSS 20B | 69.9 | 55.8 | 62.1 |

### 消融实验

| 输入条件 | 真实数据 | 合成数据 | 说明 |
|---------|---------|---------|------|
| 论文 + 代码 | 基线 | 基线 | 完整输入 |
| 仅代码 | -19.2pp（相对↓48.3%） | -16.3pp（相对↓30.8%） | 论文提供了必要的跨模态信号 |

### 关键发现

- **召回率是核心瓶颈**：最强模型仅检测 46.7% 的真实差异，精度（88-94.6%）远高于召回率，说明模型"找到的基本是对的，但漏掉了太多"。
- **Paper Omission 最难检测**：代码含论文未描述的组件时最难发现（GitHub 来源的差异 71.4% 是 Difference 所以更容易，复现报告来源 50% 是 Paper Omission 所以更难），因为无法从论文中找到锚点来对比。
- **长上下文严重降低性能**：随着论文+代码 token 数增加，检测率一致下降。中位数输入 56903 token，73/276 篇超 100k token。
- **数据污染影响显著**：模型在预训练截止日期之前发表的论文上表现更好，2025 年论文检测率最低，表明模型受益于预训练中看过的特定论文和代码。
- **论文是必需输入**：去掉论文后所有模型性能显著下降（真实数据相对下降 48.3%），确认任务的跨模态本质。
- **代码专用模型不占优**：GPT-5 Codex 反而不如 GPT-5 Mini，说明此任务需要代码理解和自然语言推理的双重能力，通用指令遵循更有用。

## 亮点与洞察

- **填补关键空白**：首次将论文-代码一致性检测形式化为可基准测试的 NLP 任务，在科学自动化时代具有极高的现实意义。
- **真实与合成的互补设计**：真实数据保证了现实性（来自实际用户报告和复现努力），合成数据解决了稀缺性和领域覆盖问题，两者检测率高度相关（$r=0.94$）验证了设计有效性。
- **"高精度低召回"的深刻洞察**：在验证场景中低召回率危害最大——漏检的差异会提供虚假安全感，而误报可由人工过滤。这对部署自动化验证系统具有重要警示意义。
- **数据污染的自然测试床**：按发表年份分析检测率的设计巧妙地揭示了数据污染问题，合成管道为持续生成未受污染数据提供了解决方案。

## 局限与展望

- 真实数据严重偏向 CS/AI，非 CS 领域仅有合成数据，错误分布可能与真实情况不同。
- 合成差异由 GPT-5 生成，而 GPT-5 也作为被评测模型之一，可能存在自我偏好偏差（去掉 GPT-5 后 $r$ 从 0.94 上升至 0.98）。
- 数据集规模较小（635 个差异），是质量与规模的权衡。
- 差异定义排除了 bug 和超参数问题，未覆盖研究代码中软件工程缺陷的完整谱系。
- 未来需扩大非 CS 领域真实数据的收集渠道，并开发专门的论文-代码验证模型。

## 相关工作与启发

- **vs Bianchi et al. (2025)**: 该工作检测论文文本内部的不一致，SciCoQA 扩展到论文与代码的跨模态不一致。
- **vs PaperBench**: PaperBench 用人工评分表验证代码实现的正确性，成本极高且不可扩展，SciCoQA 提供了可自动评测的基准。
- **vs CCI (Code-Comment Inconsistency)**: CCI 处理函数级别的代码-注释不一致，SciCoQA 要求对整篇论文和多文件代码库进行全局语义对齐，挑战性大得多。
- **vs ProcessBench/ErrorRadar**: 这些基准检测推理链中的错误，SciCoQA 检测论文描述与代码实现之间的跨模态语义差异。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次形式化论文-代码一致性验证任务，问题定义精准且极具现实意义
- 实验充分度: ⭐⭐⭐⭐⭐ 22 个模型、多维度分析（类型/来源/长度/年份/消融/精度验证），实验设计非常严谨
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机阐述深刻，差异定义严格，实验分析层层递进
- 价值: ⭐⭐⭐⭐⭐ 在科学自动化时代为验证论文-代码忠实性提供了基础设施级贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ICLR 2026\] Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](../../ICLR2026/code_intelligence/paper2code_automating_code_generation_from_scientific_papers_in_machine_learning.md)
- [\[ACL 2026\] QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions](qaq_bidirectional_semantic_coherence_for_selecting_high-quality_synthetic_code_i.md)
- [\[NeurIPS 2025\] Embedding Alignment in Code Generation for Audio](../../NeurIPS2025/code_intelligence/embedding_alignment_in_code_generation_for_audio.md)
- [\[NeurIPS 2025\] AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy](../../NeurIPS2025/code_intelligence/astrovisbench_a_code_benchmark_for_scientific_computing_and_visualization_in_ast.md)

</div>

<!-- RELATED:END -->
