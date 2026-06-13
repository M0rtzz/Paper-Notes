---
title: >-
  [论文解读] Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation
description: >-
  [ACL 2025][因果推理][反事实推理] 提出 COVER（COunterfactual VidEo Reasoning），一个多维度视频反事实推理 benchmark，将评估任务按抽象-具体和感知-认知两个维度分为四象限共 13 类任务，并通过将复杂问题分解为子问题（必要条件）来揭示——子问题准确率与反事实推理能力强相关，提升推理能力是改善视频理解鲁棒性的关键。
tags:
  - "ACL 2025"
  - "因果推理"
  - "反事实推理"
  - "视频QA"
  - "子问题分解"
  - "多模态评估"
  - "benchmark"
---

# Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation

**会议**: ACL 2025  
**arXiv**: [2503.10691](https://arxiv.org/abs/2503.10691)  
**代码**: [https://github.com/gongyifan-hash/COVER-Benchmark](https://github.com/gongyifan-hash/COVER-Benchmark)  
**领域**: 因果推理 / 视频理解  
**关键词**: 反事实推理, 视频QA, 子问题分解, 多模态评估, benchmark

## 一句话总结
提出 COVER（COunterfactual VidEo Reasoning），一个多维度视频反事实推理 benchmark，将评估任务按抽象-具体和感知-认知两个维度分为四象限共 13 类任务，并通过将复杂问题分解为子问题（必要条件）来揭示——子问题准确率与反事实推理能力强相关，提升推理能力是改善视频理解鲁棒性的关键。

## 研究背景与动机

**领域现状**：多模态大模型（MLLM）在视频理解上取得显著进展，相关 benchmark（Video-MME、MVBench 等）评估了时间推理、时空识别等能力。

**现有痛点**：(1) 现有 benchmark 很少评估反事实推理——即对"假设发生了不同的事情"的推断能力；(2) 已有反事实 benchmark（CRIPP-VQA、VITATECS）只关注特定子任务的鲁棒性，缺少从感知到认知、从具体到抽象的系统化评估；(3) 缺少对推理过程的细粒度分析机制。

**核心矛盾**：现有评估只看最终答案对错，无法诊断推理链中哪个环节出了问题——是感知出错还是认知推理出错？

**本文目标**：构建一个系统化、多维度的反事实视频推理 benchmark，支持子问题级别的细粒度评估。

**切入角度**：将反事实问题分解为多个子问题（必要条件），如果子问题答对但总结答错，说明推理整合能力不足。

**核心 idea**：子问题准确率是反事实推理性能的强预测因子——结构化推理能力直接决定视频理解的鲁棒性。

## 方法详解

### 整体框架
COVER 是一个 benchmark，不提出新模型。核心在任务分类体系 + 子问题分解机制。

### 关键设计

1. **四象限任务分类**:

    - **抽象×感知 (A&P)**：情绪识别（1类）
    - **具体×感知 (C&P)**：计数、颜色、方向、尺寸、形状、材质、位置（7类）
    - **具体×认知 (C&C)**：动作识别、物体识别（2类）
    - **抽象×认知 (A&C)**：动作预测、流程理解、社会关系（3类）
    - 设计动机：认知科学研究表明具体概念依赖多模态感知模拟，抽象概念主要通过语言符号操作表征。这一分类揭示模型在不同认知层次的能力差异

2. **子问题分解机制（核心创新）**:

    - 功能：每个反事实问题都分解为多个子问题，每个子问题是原问题正确推理的必要条件
    - 核心思路：如"倒放视频中男孩是否按顺序完成动作？"→ 子问题 Q1："倒放视频中第一个动作是什么？" Q2："最后一个动作是什么？" Q3："中间动作顺序如何？"
    - 设计动机：子问题将复杂推理拆分为可验证的步骤，能精确定位模型在推理链中的失败点

3. **三维评估指标**:

    - $\text{ori}_{acc}$：原始问题准确率（视频理解的基础能力）
    - $\text{cf}_{acc}$：反事实问题准确率（推理鲁棒性）
    - $\text{sub}_{acc}$：子问题准确率（推理过程质量）

### 数据构建
- 约 2800 个视频 + 约 12K-13K 个 QA 对，2923 个高质量反事实问题对
- 种子数据：146 个视频 + 150 个 QA（人工设计）→ GPT 扩展至每象限 720-760 对
- 8 名标注员验证 + 3 名专家交叉验证

## 实验关键数据

### 主实验（整体准确率）

| 模型 | ori_acc | cf_acc | sub_acc |
|------|---------|--------|---------|
| GPT-4o | 70.26 | 45.93 | 56.94 |
| Gemini 2.0 Flash | 77.18 | 46.90 | 62.92 |
| **InternVL2.5-78B** | 76.74 | **59.46** | **67.23** |
| LLaVA-Video-72B | 64.35 | 56.04 | 61.54 |
| InternVL2.5-8B | 74.31 | 57.75 | 61.65 |
| Qwen2-VL-7B | 71.83 | 46.90 | 58.40 |
| VILA-U-7B | 60.01 | 38.42 | 47.32 |

### 四象限分析（InternVL2.5-78B 为例）

| 象限 | ori_acc | cf_acc | sub_acc |
|------|---------|--------|---------|
| 抽象×认知 (A&C) | 72.88 | 59.60 | 57.67 |
| 具体×认知 (C&C) | 80.95 | 63.62 | 75.62 |
| 具体×感知 (C&P) | 75.99 | 58.25 | 63.65 |
| 抽象×感知 (A&P) | 76.86 | 56.20 | 70.07 |

### 关键发现
- **子问题准确率与反事实推理强相关**：sub_acc 高的模型 cf_acc 一定高，验证了"推理是鲁棒视频理解的基础"
- **反事实推理普遍困难**：所有模型 cf_acc 远低于 ori_acc（GPT-4o 差距 24.3%），说明假设情景推理是当前 MLLM 的显著短板
- **开源模型可超闭源**：InternVL2.5-78B（cf_acc 59.46%）超过 GPT-4o（45.93%），开源模型在结构化推理上潜力大
- **抽象×认知最难**：A&C 象限的 sub_acc 最低（如流程理解、动作预测），需要高阶因果推理
- **CoT 并不总是有帮助**：自动 CoT 在部分模型上甚至降低准确率，提示结构化 sub-question 比自由形式 CoT 更有效

## 亮点与洞察
- **子问题分解评估范式**：通过必要条件分解，可以精确诊断推理链的断裂点——是视觉感知错误还是逻辑推理错误。这个评估范式可推广到其他推理 benchmark
- **四象限分类的认知科学基础**：不是随意分类，而是基于认知科学的具体/抽象和感知/认知理论
- **"推理能力=泛化能力"的论点**：子问题准确率↑ → 反事实推理↑ → 视频理解鲁棒性↑，推理是通往泛化的关键路径

## 局限与展望
- 数据集部分由 GPT 扩展生成，可能引入基于 GPT 的偏见
- 2800 个视频规模适中，某些细分任务（如情绪识别）样本量较少
- 仅评估多选题 QA，未涵盖开放式生成任务
- 子问题的必要性假设可能不总成立——回答对所有子问题不一定保证最终答案正确

## 相关工作与启发
- **vs Video-MME**: Video-MME 评估一般视频理解，无反事实问题和子问题分解
- **vs CRIPP-VQA**: CRIPP-VQA 只关注物理属性的反事实，COVER 覆盖 13 类从感知到认知的任务
- **vs CoFCA**: CoFCA 处理图像而非视频，且无四象限分类

## 评分
- 新颖性: ⭐⭐⭐⭐ 子问题分解+四象限分类是系统化的新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 16 个模型、四象限分析、CoT 对比、帧采样分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，认知科学动机有说服力
- 价值: ⭐⭐⭐⭐ 为评估 MLLM 的反事实推理提供了标准化工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] RE-IMAGINE: Symbolic Benchmark Synthesis for Reasoning Evaluation](../../ICML2025/causal_inference/re-imagine_symbolic_benchmark_synthesis_for_reasoning_evaluation.md)
- [\[ACL 2025\] CoA-Reasoning: Explorations on Counterfactual Analysis in Physical Reasoning of LVLMs](coa-reasoning_explorations_on_counterfactual_analysis_in_physical_reasoning_of_l.md)
- [\[ACL 2025\] Causal Graph based Event Reasoning using Semantic Relation Experts](causal_graph_based_event_reasoning_using_semantic_relation_experts.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[ACL 2025\] Counterfactual Explanations for Aspect-Based Sentiment Analysis](counterfactual_explanations_for_aspect-based_sentiment_analysis.md)

</div>

<!-- RELATED:END -->
