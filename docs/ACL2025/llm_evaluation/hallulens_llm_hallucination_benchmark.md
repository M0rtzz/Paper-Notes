---
title: >-
  [论文解读] HalluLens: LLM Hallucination Benchmark
description: >-
  [ACL 2025][LLM评测][幻觉评估] 提出了 HalluLens 幻觉基准，明确区分幻觉与事实性，建立了外在幻觉（与训练数据不一致）和内在幻觉（与输入上下文不一致）的清晰分类体系，引入三个动态可重生成的外在幻觉评估任务，并全面分析了现有基准的局限性。
tags:
  - "ACL 2025"
  - "LLM评测"
  - "幻觉评估"
  - "外在幻觉"
  - "内在幻觉"
  - "动态测试集"
  - "benchmark"
---

# HalluLens: LLM Hallucination Benchmark

**会议**: ACL 2025  
**arXiv**: [2504.17550](https://arxiv.org/abs/2504.17550)  
**代码**: [github](https://github.com/facebookresearch/HalluLens)  
**领域**: LLM评测  
**关键词**: 幻觉评估, 外在幻觉, 内在幻觉, 动态测试集, benchmark

## 一句话总结

提出了 HalluLens 幻觉基准，明确区分幻觉与事实性，建立了外在幻觉（与训练数据不一致）和内在幻觉（与输入上下文不一致）的清晰分类体系，引入三个动态可重生成的外在幻觉评估任务，并全面分析了现有基准的局限性。

## 研究背景与动机

LLM 幻觉问题是制约其广泛应用的核心障碍，但现有研究存在严重的概念混淆和评估不足：

**定义不一致**：现有分类体系（Huang et al., 2023; Zhang et al., 2023）将幻觉与事实性（factuality）混为一谈。事实性关注的是生成内容是否符合现实世界知识，而幻觉应关注生成内容是否与模型训练数据或输入上下文一致。这两者需要不同的评估和缓解策略。

**外在幻觉被忽视**：现有基准主要关注内在幻觉（如文本摘要中的不忠实），而外在幻觉（生成内容与训练数据不一致）的评估几乎空白，随着 LLM 更多地基于任务指令生成自由文本，外在幻觉愈发重要。

**数据泄露导致基准饱和**：静态测试集容易被纳入后续模型的训练数据，导致基准分数虚高。TruthfulQA 等广泛使用的基准已出现严重污染。

**TruthfulQA 的问题**：约 25% 的被 MC1 评为错误的样本实际上可能是正确的；存在过时答案、主观题目、金标答案不准确等问题。

## 方法详解

### 整体框架

HalluLens 由两部分组成：(a) **新引入的外在幻觉评估**，包含三个动态生成的任务；(b) **整合的内在幻觉评估**，选取了三个未饱和的现有基准。

### 关键设计

1. **幻觉分类体系**：

    - **外在幻觉**（Extrinsic）：生成内容与训练数据不一致，模型试图填补知识盲区
    - **内在幻觉**（Intrinsic）：生成内容与输入上下文不一致，模型无法正确理解输入
    - **事实性**（排除）：需要外部知识源验证的正确性问题，不属于幻觉范畴
    - 关键区别：如果世界变化导致训练数据中的信息过时，模型按训练数据回答不算幻觉

2. **PreciseWikiQA 任务**：评估模型在短问答上的外在幻觉率

    - 从 GoodWiki 数据集（44,754 个高质量 Wikipedia 页面）动态生成 5,000 个问答对
    - 使用 harmonic centrality 控制难度（10 个等级），每级 500 页
    - 三个指标：虚假拒答率、非拒答时幻觉率、总体正确率
    - 自动生成的金标答案 97.2% 正确

3. **LongWiki 任务**：评估长文本生成中的外在幻觉

    - 动态生成 250 个段落级问题（难度 5-9 级，避免长尾知识）
    - 评估流程：声明提取 → 参考证据选择（Wikipedia 页面检索）→ 声明验证
    - 指标：Precision、Recall@32、F1@32

4. **NonExistentRefusal 任务**：评估模型面对不存在实体时的幻觉倾向

    - MixedEntities 子任务：混合真实动植物/药物名称生成不存在名称（8,000 样本）
    - GeneratedEntities 子任务：LLM 轮流生成虚构商业/事件/品牌名称（1,950 样本）
    - 指标：虚假接受率（越低越好）

### 损失函数 / 训练策略

本文是评估基准而非训练方法，核心设计原则：
- **动态测试集**：每次评估重新生成问题，防止数据泄露
- **可复现性**：使用 harmonic centrality 控制难度分布，确保不同版本测试集的结果稳定
- **自动化评估**：使用 LLaMA-3.1-70B-Instruct 作为评判器（拒答判断准确率 96.67%，正确性判断 95.56%）

## 实验关键数据

### 主实验

**PreciseWikiQA（13个模型）**：

| 模型 | 虚假拒答率 | 非拒答幻觉率 | 正确率 |
|------|----------|------------|-------|
| GPT-4o | 4.13% | 45.15% | **52.59%** |
| Llama-3.1-405B | 56.77% | **26.84%** | 31.62% |
| Llama-3.3-70B | 20.01% | 50.19% | 39.84% |
| Qwen2.5-7B | 13.85% | 85.22% | 12.73% |
| Mistral-7B | 7.77% | 81.19% | 17.34% |

**LongWiki**：

| 模型 | F1@32 | Precision | Recall@32 |
|------|-------|-----------|-----------|
| GPT-4o | **75.80** | **71.03** | **84.89** |
| Llama-3.1-405B | 61.98 | 56.94 | 74.44 |
| Qwen2.5-14B | 60.11 | 52.84 | 74.05 |

**NonExistentRefusal（虚假接受率，越低越好）**：

| 模型 | MixedEntities | GeneratedEntities | 平均 |
|------|-------------|-------------------|-----|
| Llama-3.1-405B | **11.48%** | **2.28%** | **6.88%** |
| Llama-3.1-8B | 19.78% | 6.58% | 13.18% |
| GPT-4o | 65.89% | 18.74% | 42.31% |
| Mistral-7B | 94.74% | 77.98% | 86.36% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 不同难度级别（PreciseWikiQA） | 长尾知识拒答率最高 | Llama/Claude 对长尾知识更倾向拒答 |
| 不同地点频率（NonExistent） | 中频地点幻觉最多 | 处于知识边界附近，模型不确定性最大 |
| TruthfulQA 误判分析 | ~25% 误判 | MC1 的 log概率求和方法存在严重缺陷 |
| 动态测试集稳定性 | <1.01% 标准差 | PreciseWikiQA 三次运行的模型排名一致 |

### 关键发现

1. **拒答与幻觉的权衡**：Llama-3.1-405B 幻觉率最低（26.84%）但拒答率最高（56.77%），GPT-4o 拒答少但幻觉偏高，二者在正确率上 GPT-4o 领先
2. **模型规模效应不统一**：同族大模型通常优于小模型，但跨族比较无此规律（如 Gemma-2-9B 与 Qwen2.5-14B 表现接近）
3. **Llama-3.3-70B 的变化**：相比 Llama-3.1-70B，拒答率大幅降低（52%→20%），但幻觉率上升（37%→50%），说明指令微调策略显著影响幻觉行为
4. **TruthfulQA 已不可靠**：发现其金标答案有误、评估方法（MC1 log概率）存在系统性偏差、时间敏感问题未更新

## 亮点与洞察

- **概念厘清的贡献巨大**：首次清晰区分幻觉与事实性，并论证两者需要不同的基准和缓解策略。这一概念框架对该领域的后续研究具有导向价值
- **动态测试集设计精巧**：通过 harmonic centrality 控制难度、动态生成问题、自动评估管线，在抗泄露与可复现性之间取得了良好平衡
- **对 TruthfulQA 的深入批判**：通过逐样本分析揭示了约 25% 的误判问题，为社区重新审视现有基准提供了实证基础
- **NonExistentRefusal 的设计巧妙**：测试模型是否知道"自己不知道什么"，直接触及幻觉的本质

## 局限与展望

1. **外在幻觉验证的假设**：假设 Wikipedia 在所有测试模型的训练数据中，但部分模型可能未完整覆盖
2. **内在幻觉缺乏动态测试集**：作者承认为内在幻觉创建动态测试集仍是开放问题
3. **评估范围有限**：仅覆盖文本幻觉，未涉及多模态幻觉
4. **评审模型偏差**：使用 LLaMA-3.1-70B 作为评审可能引入系统性偏差
5. **NonExistentRefusal 的领域偏差**：Gemma 模型拒绝回答所有药物相关问题，不区分存在与否

## 相关工作与启发

- **SimpleQA (Wang et al., 2024)**：事实性基准，通过修改指标可适配为外在幻觉基准
- **FaithEval (Ming et al., 2024)**：在噪声或反事实上下文中评估内在幻觉
- **ANAH 2.0 (Gu et al., 2024)**：在事实准确的输入上下文中评估内在幻觉
- 启发：评估基准的设计应明确"参考源"是什么（训练数据 vs 外部知识 vs 输入上下文），不同参考源对应不同类型的问题

## 评分

- 新颖性: ⭐⭐⭐⭐ 分类体系和外在幻觉评估任务设计新颖，对TruthfulQA的分析有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 13个模型、三个新任务、三个现有基准分析、稳定性验证，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 概念辨析清晰，图表丰富，论证严谨
- 价值: ⭐⭐⭐⭐⭐ 为幻觉研究建立了清晰的分类框架和新的评估标准，对领域发展有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining](tic-lm_a_web-scale_benchmark_for_time-continual_llm_pretraining.md)
- [\[ACL 2025\] RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis](realhitbench_a_comprehensive_realistic_hierarchical_table_benchmark_for_evaluati.md)
- [\[NeurIPS 2025\] Robust Hallucination Detection in LLMs via Adaptive Token Selection](../../NeurIPS2025/llm_evaluation/robust_hallucination_detection_in_llms_via_adaptive_token_selection.md)
- [\[ACL 2025\] Navigating Rifts in Human-LLM Grounding: Study and Benchmark](navigating_rifts_in_human-llm_grounding_study_and_benchmark.md)

</div>

<!-- RELATED:END -->
