---
title: >-
  [论文解读] Mis-prompt: Benchmarking Large Language Models for Proactive Error Handling
description: >-
  [ACL 2025][Proactive Error Handling] 提出 Mis-prompt 基准，包含 4 项评估任务、14 类错误分类体系和 14,969 条数据集，系统研究 LLM 在无显式错误处理指令时的**主动**纠错能力，发现当前 LLM 主动纠错能力严重不足，SFT 可显著提升。
tags:
  - ACL 2025
  - Proactive Error Handling
  - Error Detection
  - Error Correction
  - Error Guidance
  - benchmark
  - SFT
---

# Mis-prompt: Benchmarking Large Language Models for Proactive Error Handling

**会议**: ACL 2025  
**arXiv**: [2506.00064](https://arxiv.org/abs/2506.00064)  
**代码**: 未公开（数据集将公开）  
**领域**: LLM NLP / 错误处理与评估  
**关键词**: Proactive Error Handling, Error Detection, Error Correction, Error Guidance, benchmark, SFT

## 一句话总结

提出 Mis-prompt 基准，包含 4 项评估任务、14 类错误分类体系和 14,969 条数据集，系统研究 LLM 在无显式错误处理指令时的**主动**纠错能力，发现当前 LLM 主动纠错能力严重不足，SFT 可显著提升。

## 研究背景与动机

**领域现状**: LLM 在错误处理方面已取得显著进展，包括数学推理纠错 (EIC-Math, ErrorRadar)、语法检查 (GEC)、摘要事实核查 (SummEdits) 等。然而这些工作均为**被动**错误处理——依赖用户在 prompt 中给出**显式的错误处理指令**（如"请判断以下内容是否正确"）。

**核心问题**: 在真实场景中，用户通常**不会**提供显式的错误处理指令。例如用户问"拿破仑在 1815 年的滑铁卢战役中大获全胜后…"，这里隐含了历史错误（拿破仑战败），但用户没有要求 LLM 纠错。

**关键发现**: GPT-4o 面对含错误的用户输入时，无法主动识别错误，而是基于错误信息继续生成，严重损害系统的可信度和可靠性。

**与已有工作的区别**: 现有基准（BIG-Bench Mistake、ReaLMistake、Medec、ProcessBench等）均为被动模式且大多仅覆盖检测+识别，本文首次定义并评估**主动**错误处理，且涵盖检测、识别、纠正、引导四个维度。

## 方法详解

### 整体框架

Mis-prompt 框架包含三部分：(1) 错误分类体系——4 大类 14 小类；(2) 评估任务——检测/识别/纠正/引导四项任务；(3) 数据集——14,969 条带标注的错误 prompt 数据。

### 四项评估任务

1. **Error Detection（错误检测）**: 判断 prompt 是否包含错误，输出二元标签 y ∈ {True, False}
2. **Error Identification（错误识别）**: 评估模型能否尝试定位错误（y₁）以及是否正确定位（y₂），双标签评估
3. **Error Correction（错误纠正）**: 评估模型能否尝试纠正错误（y₁）以及纠正是否准确（y₂），双标签评估
4. **Error Guidance（错误引导）**: 评估模型能否提供有意义的建议帮助用户改进 prompt

### 错误分类体系

基于已有工作 (Pagnoni et al., Sourati et al., Orlovskiy et al., Masanti et al.) 构建：

| 一级分类 | 二级分类 | 数据量 |
|----------|----------|--------|
| 语言错误 | 语法错误 / 标点错误 / 拼写错误 | 3,135 |
| 不完整信息 | 说话人/人物 / 文本主题 / 地点 / 时间日期 | 4,164 |
| 事实错误 | 关系错误 / 实体错误 / 情境错误 | 3,109 |
| 逻辑错误 | 关联谬误 / 预设谬误 / 归纳缺陷 / 歧义谬误 | 4,288 |

### 数据构建方法

1. **数据生成**: 两种途径——(a) 转化已有数据集（FEVEROUS → 事实错误、CommonsenseQA → 逻辑错误、ROCStories → 叙事错误），通过 GPT-4o 将正确陈述转化为含错的 Wh-问句；(b) 直接生成——按二级分类让 GPT-4o 生成多样化的错误 prompt
2. **多样性设计原则**: 特殊疑问句+错误（非简单判断题）；错误信息嵌入从句；错误陈述+相关提问
3. **去重**: 使用 Sentence-BERT 计算余弦相似度，阈值 0.85 以上的合并
4. **质量控制**: 3 名研究生人工审核，Fleiss Kappa = 0.78（高度一致），最终质量评分 93.76%
5. **Ground-truth 生成**: GPT-4o 根据错误类别和错误 prompt 生成标准回答，含错误检测、解释、纠正和引导

### 评估方法

- 自动评估：GPT-4o 作为 judge 模型，F1 指标
- 人工评估：3 名研究生二次评估，Fleiss Kappa = 0.63，与自动评估差异仅 5.59%

## 实验

### 主实验结果（Table 3）

| 模型 | Det. | Att.Ident. | Acc.Ident. | Att.Corr. | Acc.Corr. | Guid. | Avg |
|------|------|------|------|------|------|------|------|
| Claude-3.5 | 63.98 | 67.53 | 63.01 | 36.48 | 30.23 | 43.73 | **50.83** |
| GPT-4o | 43.54 | 48.71 | 43.78 | 31.72 | 23.32 | 30.66 | 36.96 |
| LLaMA-3.3-70B | 57.78 | 59.23 | 53.50 | 39.67 | 30.17 | 37.40 | 46.29 |
| Qwen-2.5-32B | 51.11 | 54.91 | 50.63 | 34.20 | 27.21 | 41.39 | 43.24 |
| DeepSeek-V2-16B | 29.44 | 33.90 | 27.92 | 18.57 | 11.46 | 12.80 | 22.35 |

**关键发现**:
- 闭源模型整体优于开源模型，Claude-3.5 最佳（50.83%）
- 所有模型平均 F1 仅 37.53%，说明主动纠错能力严重不足
- 四项任务难度递增：检测 > 识别 > 纠正 > 引导
- 准确纠正（22.62%）是最难的任务

### 分类别分析（Table 4 - GPT-4o）

| 错误类别 | Det. | Acc.Ident. | Acc.Corr. | Guid. | Avg |
|----------|------|------|------|------|------|
| 事实错误 | 72.99 | 71.70 | 41.63 | 22.27 | **55.22** |
| 逻辑错误 | 41.49 | 43.04 | 30.91 | 18.36 | 38.74 |
| 不完整信息 | 40.58 | 41.32 | 7.83 | 48.43 | 32.53 |
| 语言错误 | 6.50 | 3.93 | 13.34 | 20.23 | **11.53** |

GPT-4o 最擅长发现事实错误（得益于丰富的知识库），但对语言错误近乎"视而不见"（F1 仅 6.5%），倾向于直接回答而忽略错误。

### SFT 提升效果（Table 5）

| 模型 | 方法 | Det. | Acc.Ident. | Acc.Corr. | Guid. | Avg |
|------|------|------|------|------|------|------|
| LLaMA-3.1-8B | zero-shot | 42.05 | 40.48 | 19.70 | 33.76 | 35.15 |
| LLaMA-3.1-8B | 3-shot | 81.99 | 69.09 | 40.72 | 82.43 | 72.15 |
| LLaMA-3.1-8B | CoT | 75.62 | 73.56 | 47.02 | 75.44 | 70.22 |
| LLaMA-3.1-8B | **SFT** | **90.16** | **80.02** | **62.86** | **84.77** | **81.77** |
| Qwen-2.5-32B | zero-shot | 51.11 | 50.63 | 27.21 | 41.39 | 43.24 |
| Qwen-2.5-32B | **SFT** | **97.88** | **88.43** | **70.86** | **93.17** | **89.55** |

SFT 带来 30-50 个百分点的提升，远超 few-shot 和 CoT 方法。Qwen-2.5-32B + SFT 达到 89.55% 的最佳结果。

### 缩放规律观察

- LLaMA 系列符合缩放定律：70B > 8B > 3B
- Qwen-2.5 系列出现反向缩放：32B > 72B，大模型不一定更好
- GPT-4o 在纠正/引导上表现低于预期，因其更倾向于直接回答用户问题

## 亮点与洞察

1. **问题定义新颖**: 首次明确区分主动 vs 被动错误处理，填补了评估空白。真实场景中用户很少会告诉模型"你的输入有错"
2. **分类体系全面**: 4 大类 14 小类错误分类覆盖广泛，数据量充足（近 15K 条）
3. **任务设计梯度合理**: 检测→识别→纠正→引导的梯度设计揭示了模型在不同复杂度下的能力断崖
4. **SFT 是银弹**: 实验有力证明 SFT 在主动纠错上远优于 ICL 和 CoT，表明主动纠错能力并非模型内禀的，需要显式训练
5. **有趣反例**: GPT-4o 几乎完全忽视语言错误（Det. 仅 6.5%），说明当前 LLM 的注意力分配机制在纠错场景中的局限

## 局限性

1. 仅限纯文本单轮对话，未覆盖多模态和多轮场景
2. F1 指标虽便于规模化评估，但可能无法全面捕捉评估的各个方面
3. 数据主要由 GPT-4o 生成，可能受到其特定偏见的影响
4. 未探索指令微调与 RLHF 结合的效果

## 相关工作

- **被动错误处理**: BIG-Bench Mistake（逻辑任务错误检测）、Medec（临床笔记纠错）、EIC-Math/ErrorRadar/ProcessBench（数学推理纠错）
- **LLM 错误检测评估**: ReaLMistake（多维度错误检测）、SummEdits（摘要事实核查）
- **错误处理改进**: LoRA 微调、Few-shot learning、CoT prompting

## 评分 ⭐⭐⭐⭐

- **创新性**: ⭐⭐⭐⭐ 主动 vs 被动纠错的区分简单而深刻，问题定义有实际意义
- **实验完备性**: ⭐⭐⭐⭐ 13 个模型 × 5 种方法，分类别分析透彻
- **数据集质量**: ⭐⭐⭐⭐ 分类体系完善，质量控制严格（93.76%通过率）
- **实用性**: ⭐⭐⭐⭐ 直接指导 LLM 安全性和可靠性的改进方向

<!-- RELATED:START -->

## 相关论文

- [CodeMEnv: Benchmarking Large Language Models on Code Migration](codemenv_benchmarking_large_language_models_on_code_migration.md)
- [AD-LLM: Benchmarking Large Language Models for Anomaly Detection](ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)
- [Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)
- [Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)

<!-- RELATED:END -->
