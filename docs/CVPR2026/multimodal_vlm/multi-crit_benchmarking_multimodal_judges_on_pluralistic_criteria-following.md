---
title: >-
  [论文解读] Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following
description: >-
  [CVPR 2026][多模态][LMM-as-Judge] 构建首个评估多模态 Judge 模型多准则遵循能力的基准 Multi-Crit，包含准则级人类标注和偏好冲突样本，配合 PAcc/TOS/CMR 三个新指标，全面评估 25 个 LMM 并揭示闭源最强模型在开放生成任务上仅 32.78% 的多准则一致性。
tags:
  - CVPR 2026
  - 多模态
  - LMM-as-Judge
  - 多准则评估
  - benchmark
  - 偏好冲突
  - 评估可靠性
---

# Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following

**会议**: CVPR 2026  
**arXiv**: [2511.21662](https://arxiv.org/abs/2511.21662)  
**代码**: https://multi-crit.github.io  
**领域**: 多模态VLM  
**关键词**: LMM-as-Judge, 多准则评估, benchmark, 偏好冲突, 评估可靠性

## 一句话总结

构建首个评估多模态 Judge 模型多准则遵循能力的基准 Multi-Crit，包含准则级人类标注和偏好冲突样本，配合 PAcc/TOS/CMR 三个新指标，全面评估 25 个 LMM 并揭示闭源最强模型在开放生成任务上仅 32.78% 的多准则一致性。

## 研究背景与动机

**领域现状**：LMM-as-a-Judge 范式被广泛用于自动评测和 RLHF 反馈。给定多模态 prompt、模型响应和预定义评估准则，Judge 模型输出偏好判断并附带文字理由。这一范式因可扩展性和灵活性被大量多模态 benchmark 采用，也有多项工作微调开源模型作为专用 Judge/Critic 来提供 AI 反馈。

**现有痛点**：现有多模态 Judge 基准（VL-Rewardbench、MM-RLHF Bench 等）仅提供单一总体偏好标签。这种粗粒度标注无法捕捉多维度评估的本质——两个回复会在不同准则间存在 trade-off，如一个回复简洁但有事实错误，另一个内容详尽但冗余。单一标签抹平了这些细节。

**核心矛盾**：Judge 模型的可靠性依赖两个要素：(1) 与人类判断一致；(2) 灵活遵循多样化的任务特定评估准则。现有工作关注前者但严重忽视后者。Judge 模型是否真正遵循了给定准则？面对准则间的偏好冲突时能否正确判断？这些关键问题未被系统研究。

**本文目标** (1) 如何构建包含多准则人工标注和准则间偏好冲突的评估数据？(2) 如何系统度量 Judge 模型的多准则遵循能力？

**切入角度**：多准则评估 + 冲突检测——让人类标注者独立标注每个准则下的偏好，天然暴露不同准则间的偏好冲突。

**核心 idea**：构建带准则级人工标注的挑战性基准 Multi-Crit，设计 PAcc/TOS/CMR 三个新指标，系统评估 25 个模型在多准则遵循上的表现与瓶颈。

## 方法详解

### 整体框架

Multi-Crit 将传统的 pairwise preference 评估扩展为多准则形式。传统基准的数据格式为 $(q, l_a, l_b, y)$，即一个 prompt 对应一个整体偏好标签 $y$。Multi-Crit 将其扩展为 $(q, l_a, l_b, \{(c_i, y_i)\}_{i=1}^{K_q})$，其中每个 $c_i$ 是一个评估准则，$y_i$ 是该准则下的偏好标签。这使得同一对回复可以在不同准则下有不同的偏好方向，从而捕捉准则间的冲突。

基准构建流程：多来源 prompt 收集 → 多模型响应生成与配对 → 三阶段过滤保留挑战性样本 → 准则级人工标注（9 名 CS PhD，289 小时） → 偏好聚合与质量验证 → 最终数据集。

### 关键设计

1. **数据构建管线（Data Curation Pipeline）**：

    - 功能：从多来源构建高质量、具有挑战性的多准则评估数据
    - 核心思路：Prompt 来自 8 个数据集覆盖开放生成（ImageInWords、DOCCI、WildVision-Bench/-Battle）和可验证推理（MathVerse、MM-K12、EMMA-mini、VisualPuzzles）两大场景；用 11 个高性能 LMM（含 GPT-4o、Gemini-2.5-Flash 等闭源和 Qwen2.5-VL、InternVL3 等开源）生成候选回复；构建跨模型对（两个不同模型）和同模型对（同一模型温度采样 5 次，选余弦距离最大的对）两种配对方式，共产生 3,538 个回复对
    - 设计动机：跨模型对捕捉模型间系统性差异，同模型对捕捉同一模型内的质量波动，两者互补保证评估全面性

2. **三阶段挑战性过滤（Three-Stage Filtering）**：

    - 功能：从 3,538 对中筛选出 707 对真正具有细粒度准则差异的挑战性样本
    - 核心思路：(1) **长度归一化**——排除长度比超出 [0.7, 1.4] 的回复对，避免长度偏见；(2) **推理正确性过滤**——对推理任务用 GPT-4o-mini 验证，仅保留双对或双错的样本（答案正确性本身是 trivial 信号）；(3) **集成难度过滤**——用三个强 Judge（GPT-4o、Gemini-2.5-Flash、Claude-3.7-Sonnet）做初始整体评估，三者一致的丢弃，仅保留存在分歧的挑战性样本
    - 设计动机：每一步都有针对性地去除"简单"样本——长度差异过大会导致 Judge 走捷径，答案本身就分对错的无需 Judge 评质量，强模型一致同意的说明差异太明显

3. **准则设计（Criteria Design）**：

    - 功能：定义评估的多个维度，覆盖多模态判断的核心能力
    - 核心思路：遵循三条原则——实用性（反映 Judge 常见使用场景）、特异性（准则间不重叠）、通用性（评估基本能力维度而非内容特定）。开放生成 5 准则：Completeness & Coverage、Visual Grounding & Details、Factuality / No Hallucination、Creativity & Expressiveness、Clarity & Coherence。可验证推理 5 准则：Visual Grounding、Logic Coherence & Consistency、Factuality / No Hallucination、Reflection & Exploration、Conciseness & Efficiency
    - 设计动机：多轮迭代精炼自现有 MLLM-as-a-Judge 基准的准则总结，确保准则间互补不冗余

4. **三个新评估指标（PAcc/TOS/CMR）**：

    - 功能：从不同维度度量 Judge 的多准则遵循能力
    - **PAcc (Pluralistic Adherence Accuracy)**：$\text{PAcc} = \frac{1}{|X|} \sum_{x \in X} \mathbb{I}[\bigwedge_{c \in C_x} \hat{y}_{x,c} = y_{x,c}]$——所有准则都判断正确才算该 prompt 通过，衡量多准则一致遵循能力
    - **TOS (Trade-Off Sensitivity)**：在存在准则冲突的样本上，Judge 是否至少能感知到不同准则应有不同偏好方向（只需存在一对冲突准则的预测方向不同即可），衡量灵活性而非精确度
    - **CMR (Conflict Matching Rate)**：在冲突准则对上，Judge 是否不仅检测到冲突而且解析方向与人类一致，是最严格的指标
    - 设计动机：PAcc 是整体性要求，TOS 检测 Judge 是否 criterion-agnostic（所有准则输出相同方向），CMR 细粒度检验冲突解析能力，三者从宽到严逐步刻画能力层次

### 标注流程与质量保证

标注团队为 9 名 CS PhD，均有多模态 AI 和 STEM 背景。先标注 20 个种子样本（10 开放 + 10 推理）进行小组讨论和校准，对齐理解后进入正式标注。每个样本分配 3 名标注者交叉验证，标注者每次只看一个准则，判定哪个回复更好（tie 限制在 10% 以下）并写简短理由。偏好聚合仅保留全体一致或两人一致且第三人为 tie 的样本；项目负责人人工审查文字理由，丢弃不一致或冗余的样本。最终标注耗时 289 小时，Cohen's $\kappa$ 达到开放任务 0.718 和推理任务 0.805，属于 substantial agreement。

## 实验关键数据

### 主实验：开放生成任务（Open-Ended Split）

| 模型 | PAcc(%) | CMR(%) | TOS(%) | 准则均值(%) |
|------|---------|--------|--------|------------|
| o4-mini | **32.78** | **43.11** | 64.56 | **69.67** |
| Claude-3.7-Sonnet | 31.77 | 42.32 | 64.08 | 67.37 |
| GPT-4o | 31.44 | 44.91 | **66.02** | 69.57 |
| o3 | 31.10 | 42.71 | 62.62 | 69.16 |
| GPT-5 | 29.77 | 38.52 | 62.62 | 68.51 |
| InternVL3.5-38B（开源最佳） | 30.43 | 33.73 | 64.08 | 65.10 |
| InternVL3-78B | 29.10 | 32.53 | 56.31 | 64.71 |
| MiMo-VL-7B | 29.10 | 39.52 | 65.53 | 63.37 |
| Qwen2.5-VL-72B | 28.43 | 35.53 | 60.68 | 63.84 |
| R1-Reward-7B（微调最佳） | 17.73 | 20.36 | 45.63 | 55.83 |
| Qwen2.5-VL-7B | 9.41 | 17.28 | 36.14 | 54.39 |

### 主实验：可验证推理任务（Reasoning Split）

| 模型 | PAcc(%) | CMR(%) | TOS(%) | 准则均值(%) |
|------|---------|--------|--------|------------|
| o4-mini | **53.17** | **65.84** | 83.49 | **80.85** |
| GPT-5 | 45.24 | 56.58 | 78.90 | 77.41 |
| o3 | 44.44 | 62.28 | 82.57 | 77.86 |
| GPT-4o | 41.27 | 55.16 | **84.40** | 69.79 |
| Gemini-2.5-Pro | 41.27 | 52.33 | 75.93 | 73.06 |
| InternVL3.5-38B（开源最佳） | 37.30 | 47.69 | 75.23 | 69.82 |
| MiMo-VL-7B | 37.30 | 41.99 | 71.56 | 66.30 |
| Qwen2.5-VL-72B | 32.54 | 45.91 | 77.06 | 64.48 |
| InternVL3-8B | 26.98 | 39.50 | 66.06 | 66.22 |
| R1-Reward-7B | 19.05 | 24.56 | 62.39 | 54.50 |

### 消融实验：Critic 微调对各准则的影响（开放生成）

| 模型 | Completeness | Grounding | Hallucination | Expressiveness | Clarity | Avg |
|------|-------------|-----------|---------------|----------------|---------|-----|
| Qwen2.5-VL-7B (base) | 56.12 | 51.70 | 48.20 | 64.12 | 51.82 | 54.39 |
| R1-Reward-7B | 59.29 | **60.71** | 49.72 | 55.44 | 53.98 | 55.83 |
| UnifiedReward-7B | 57.96 | **52.23** | 52.49 | 57.51 | 55.68 | 55.17 |
| LLaVA-Critic-R1-7B | 55.31 | **57.59** | 46.96 | 63.73 | 55.11 | 55.74 |

所有 Qwen-based 微调 Judge 均在 Visual Grounding 准则上有一致提升（51.70→52.23~60.71），但其他准则提升不一致甚至下降。

### 关键发现

- **多准则判断极其困难**：最强的 o4-mini 在开放生成上 PAcc 仅 32.78%，在推理上也仅 53.17%，表明即使 SOTA 模型也无法在所有准则上同时做出正确判断
- **开放任务比推理任务更难**：所有模型在开放生成上的表现显著低于推理任务，反映开放任务的主观性和对细粒度视觉感知的更高要求
- **没有模型全面领先**：o4-mini 在 Logic 和 Efficiency 上最强，但在 Hallucination 上被 o3 超越（84.21% vs 79.31%），在 Grounding 上被 Gemini-2.5-Pro 超越（79.01% vs 77.78%）
- **开源模型在冲突检测上差距更大**：CMR 从闭源到开源下降约 9.4 点（开放任务）和 18.1 点（推理任务），远超准则级准确率的 4-11 点差距
- **Critic 微调仅提升 Visual Grounding**：微调 Judge 在 Grounding 上一致改善，但在其他准则和冲突解析上提升有限甚至退步，因为训练信号是 holistic 偏好而非准则级
- **推理微调削弱 trade-off 识别**：GRPO 微调模型虽然推理能力提升，但 TOS 和 CMR 反而下降，说明 holistic accuracy reward 不利于准则间冲突感知
- **Test-time scaling 效果有限**：majority vote 对 o4-mini 有稳定提升（PAcc 32.78→37.12），但对其他模型效果不一致、方差大
- **闭源模型上限与人类一致性对齐**：闭源模型最强准则准确率与 Cohen's $\kappa$ 相关性 $r=0.73, p=0.024$，而开源模型仅 $r=0.36, p=0.344$

## 亮点与洞察

- 首个多准则多模态 Judge 基准，填补准则级评估空白，数据集中 68.9%（开放）和 86.5%（推理）的样本存在准则间偏好冲突
- PAcc/TOS/CMR 三个指标形成从宽到严的能力评估层次，揭示了单一准则准确率无法反映的系统性缺陷
- 289 小时高质量人工标注（Cohen's $\kappa$ 0.718/0.805），三阶段过滤确保样本具有细粒度准则差异
- "Critic 微调仅提升 Grounding"这一发现对构建更好的 Judge 训练方法有重要指导意义——需要准则级训练信号而非 holistic 偏好
- 闭源模型上限与人类标注者一致性高度相关，暗示下一步挑战是超越人类水平的评估对齐

## 局限与展望

- 仅支持 pairwise comparison 模式，pointwise scoring 的多准则评估值得探索
- 准则设计仍较通用，领域特定准则（医疗、法律、代码）需进一步扩展
- 标注成本高（289 小时为 9 人共计），规模化扩展需要半自动标注管线
- Tie 标注被限制在 10% 以下，可能丢失真正难以区分的边界样本
- 仅评估生成式 Judge，BT-style reward model 的多准则能力也应纳入研究
- 开源模型在所有指标上全面落后，亟需准则级 critic 训练数据和多准则 RLHF 方法

## 相关工作与启发

- **LMM-as-a-Judge**：GPT-4V 最早展示与人类一致的评估能力，后续 LLaVA-Critic、R1-Reward 等微调开源替代品，但训练信号是 holistic 偏好
- **Judge 基准**：MLLM-as-a-Judge 首先评估 LMM 作为 Judge 的能力，VL-Rewardbench、MM-RLHF Bench 扩展到多场景，但均为单一偏好标签
- **准则遵循**：文本 LLM 领域已有初步探索（嵌入准则级差异或从人类理由中总结准则），Multi-Crit 将其扩展至多模态并引入冲突检测
- **启发**：多准则 Judge 训练需要准则级标注数据和准则感知的 reward signal，而非仅靠 holistic preference

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个多准则多模态 Judge 基准，PAcc/TOS/CMR 三指标体系设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 25 个模型全面评估，含微调 Judge、reasoning fine-tuning、test-time scaling、人类上限分析等丰富 ablation
- 写作质量: ⭐⭐⭐⭐ 结构清晰、数据详实、准则定义严谨
- 价值: ⭐⭐⭐⭐ 揭示了当前 Judge 系统的系统性不足，尤其是 Critic 微调仅提升 Grounding 的发现对下一步研究有重要指导意义

<!-- RELATED:START -->

## 相关论文

- [CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning](crit_graph-based_automatic_data_synthesis_to_enhance_cross-modal_multi-hop_reaso.md)
- [GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning](graphvlm_benchmark_vlm_graph_learning.md)
- [MM-IFEngine: Towards Multimodal Instruction Following](../../ICCV2025/multimodal_vlm/mm-ifengine_towards_multimodal_instruction_following.md)
- [FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)
- [Benchmarking Vision-Language Models under Contradictory Virtual Content Attacks in Augmented Reality](benchmarking_vision-language_models_under_contradictory_virtual_content_attacks_.md)

<!-- RELATED:END -->
