---
title: >-
  [论文解读] ChartMuseum: 测试大型视觉语言模型的图表视觉推理能力
description: >-
  [NeurIPS 2025][多模态][图表理解] 提出ChartMuseum图表问答基准，包含1162个专家标注问题和184个来源的真实图表，首次系统区分视觉推理与文本推理能力，揭示当前最强模型Gemini-2.5-Pro仅63.0%而人类达93%，视觉推理性能比文本推理低35%-55%。
tags:
  - NeurIPS 2025
  - 多模态
  - 图表理解
  - 视觉推理
  - 基准测试
  - VLM评估
  - 图表问答
---

# ChartMuseum: 测试大型视觉语言模型的图表视觉推理能力

**会议**: NeurIPS 2025  
**arXiv**: [2505.13444](https://arxiv.org/abs/2505.13444)  
**代码**: https://chartmuseum-leaderboard.github.io  
**领域**: 多模态VLM  
**关键词**: 图表理解, 视觉推理, 基准测试, VLM评估, 图表问答

## 一句话总结

提出ChartMuseum图表问答基准，包含1162个专家标注问题和184个来源的真实图表，首次系统区分视觉推理与文本推理能力，揭示当前最强模型Gemini-2.5-Pro仅63.0%而人类达93%，视觉推理性能比文本推理低35%-55%。

## 研究背景与动机

- **现有图表QA基准过度依赖文本推理**: 在ChartQA上仅用提取的文本信息（不看图），Claude-3.7-Sonnet就能达到74.1%准确率（看图87.4%），说明大部分问题不需要真正的视觉推理
    - 而在ChartMuseum上，同样的纯文本方式只有15.2%（看图61.3%），差距达46%，说明ChartMuseum真正考察了视觉推理
- **前沿模型在已有基准上趋于饱和**: ChartQA上模型准确率集中在85%-90%之间，难以区分模型能力差异
- **视觉推理与文本推理的区分被忽视**: 图表理解涉及两类推理——直接从图形关系中推断（视觉推理）vs. 从提取的文本/数值中推断（文本推理），但现有工作未明确区分
- **合成数据案例研究揭示问题**: 作者用不含任何文字标注的合成图表测试，随着视觉复杂度（overlay/subplot数量n）增加，模型性能显著下降，而人类表现稳定

## 方法详解

### 整体框架

ChartMuseum是一个图表问答（Chart QA）基准数据集，由13名计算机科学研究者手工标注，包含1162个 (图像, 问题, 简短答案) 元组，图像来自928张独特的真实世界图表、184个不同网站来源。数据集划分为dev/test = 162/1000。

### 关键设计

1. **视觉推理vs文本推理区分**: 将图表理解中的推理明确分为两类：
    - **视觉推理**: 从图形关系中进行推断，用自然语言表达困难（如散点图中判断两变量的相关性）
    - **视觉提取**: 视觉推理的子类，通过视觉解读获取数值（如通过对比y轴刻度估计柱子的值）
    - **文本推理**: 对已提取信息进行逻辑/算术/比较运算，或直接从图表中读取文本标注
    - 这一区分表明现有基准严重偏向文本推理

2. **四类问题分类体系**:
    - **Textual Reasoning (123题)**: 几乎完全靠文本推理即可解答
    - **Visual Reasoning (510题)**: 主要需要视觉推理，占比最大
    - **Text/Visual Reasoning (234题)**: 文本或视觉推理均可解答
    - **Synthesis Reasoning (133题)**: 同时需要文本和视觉推理

3. **多阶段质量审核流程**:
    - 第一阶段：选取高质量图表
    - 第二阶段：手工创建问答对（不使用LLM辅助、不使用模板）
    - 第三阶段：独立审核者验证答案正确性
    - 第四阶段：与标注者讨论迭代优化
    - 每个样本平均耗时20分钟（标注10min + 审核5min + 迭代5min），总计约400小时
    - 标注规则：答案空间≥4选项、答案客观无歧义、排除why/how/描述性/复合问题

### 损失函数 / 训练策略

本文为基准测试论文，不涉及模型训练。评估使用LLM-as-a-Judge（GPT-4.1-mini）判断答案等价性，所有问题都有唯一确定答案，不使用容忍误差的近似匹配。

## 实验关键数据

### 主实验

| 模型 | Visual (510) | Synthesis (133) | Visual/Text (234) | Text (123) | Overall (1000) |
|---|---|---|---|---|---|
| **开源小模型** | | | | | |
| InternVL3-2B | 12.2 | 13.5 | 18.4 | 30.1 | 16.0 |
| Qwen2.5-VL-3B | 16.7 | 21.1 | 26.5 | 28.5 | 21.0 |
| **开源中型模型** | | | | | |
| Qwen2.5-VL-7B | 19.4 | 24.8 | 36.3 | 41.5 | 26.8 |
| InternVL3-8B | 23.5 | 24.8 | 32.9 | 42.3 | 28.2 |
| Bespoke-MiniChart-7B | 26.3 | 32.3 | 41.0 | 54.5 | 34.0 |
| **开源大模型** | | | | | |
| Qwen2.5-VL-32B | 29.0 | 36.1 | 46.2 | 62.6 | 38.1 |
| Pixtral-Large-124B | 31.6 | 36.1 | 40.6 | 65.9 | 38.5 |
| Qwen2.5-VL-72B | 30.4 | 35.3 | 42.3 | 68.3 | 38.5 |
| **闭源模型** | | | | | |
| Gemini-1.5-Flash | 22.7 | 30.8 | 36.3 | 56.1 | 31.1 |
| GPT-4o | 31.8 | 45.1 | 50.9 | 65.9 | 42.2 |
| GPT-4.1 | 37.1 | 53.4 | 54.3 | 78.9 | 48.4 |
| Claude-3.5-Sonnet | 45.7 | 53.4 | 61.5 | 78.0 | 54.4 |
| Claude-3.7-Sonnet | 50.6 | 55.6 | 69.2 | 88.6 | 60.3 |
| **推理模型** | | | | | |
| o3 (high) | 50.4 | 63.2 | 69.7 | 85.4 | 60.9 |
| o4-mini (high) | 51.2 | 66.2 | 68.4 | 86.2 | 61.5 |
| Claude-3.7-Sonnet (think) | 52.5 | 56.4 | 71.8 | 86.2 | 61.7 |
| **Gemini-2.5-Pro** | **53.3** | **64.7** | **70.1** | **87.8** | **63.0** |
| **人类** | **98.2** | — | — | — | **93.0** |

### 消融实验

**现有基准对比（文本提取实验）**:

| 数据集 | 仅文本提取 | 使用图像 |
|---|---|---|
| ChartQA | 74.1% | 87.4% |
| ChartMuseum | 15.2% | 61.3% |

ChartMuseum的文本提取vs图像差距达46%，远大于ChartQA的13%，证明ChartMuseum真正测试了视觉推理能力。

**视觉任务分类错误分析** (各采样50个错误实例):

| 错误类型 | Claude-3.7-Sonnet | Gemini-2.5-Pro |
|---|---|---|
| Symbol Selection | 34% | 28% |
| Visual Comparison | 28% | 26% |
| Trajectory Tracking | 14% | 12% |
| X/Y Value Identification | 6% | 28% |
| Strategy Error | 16% | 2% |
| Textual Reasoning Error | 6% | 2% |

### 关键发现

1. **闭源vs开源差距巨大**: 最佳开源模型Qwen2.5-VL-72B (38.5%) 与最佳闭源模型Gemini-2.5-Pro (63.0%) 差距达24.5%
2. **视觉推理远弱于文本推理**: 所有模型在Visual列的表现比Text列低35%-55%，如GPT-4.1在Text上78.9%但Visual仅37.1%（降41.8%），Qwen2.5-VL-72B从68.3%降至30.4%（降37.9%）
3. **推理模型提升有限**: Claude-3.7-Sonnet开启extended thinking仅提升1.4%（60.3%→61.7%），说明问题不在推理步骤长度而在基础视觉能力
4. **人类视觉推理近乎完美**: 人类在视觉推理子集上达98.2%（56/57正确），而最强模型仅53.3%
5. **专用模型仍有差距**: Bespoke-MiniChart-7B虽大幅超越同量级开源模型（34.0% vs 26.8%/28.2%），但仍远逊于闭源模型
6. **策略错误**: Claude-3.7-Sonnet有16%的错误属于策略错误——模型未能采用视觉推理"捷径"，转而尝试提取数值进行计算，导致答错

## 亮点与洞察

- **视觉推理与文本推理的形式化区分**是本文最重要的贡献，这一框架让我们能量化LVLM在两种能力上的不对称性
- **"提取即可答"实验**（Section 2.2）巧妙地证明了ChartQA等旧基准的局限性——不看图也能答对74%
- **四类视觉任务分类学**（Symbol Selection / Visual Comparison / Trajectory Tracking / X/Y Value Identification）为未来模型改进提供了具体方向
- **Strategy Error的发现特别有趣**: 模型过度依赖文本化推理策略，即使问题可以通过简单的视觉比较解决，模型也倾向于提取数值再计算——这揭示了当前LVLM的深层架构偏见
- 数据集标注完全由人类完成（不使用LLM生成问题），每个样本20分钟，总计400小时，质量控制严格

## 局限与展望

- 仅包含英文图表和问题，未覆盖多语言场景
- 仅评估短答案QA，未涵盖摘要生成、开放式回答等任务
- 不包含不可回答的问题（unanswerable questions）
- 数据集规模（1162题）相对不大，部分子类别样本较少
- 未提出改进模型视觉推理的具体方法（纯诊断性工作）
- 可扩展方向：基于发现的视觉推理弱点，设计针对性的训练数据或架构改进

## 相关工作与启发

- **图表QA基准演进**: FigureQA/DVQA（合成图表+模板问题）→ ChartQA（真实图表+人工问题）→ CharXiv/ChartQAPro（更复杂但来源有限或模型生成问题）→ ChartMuseum（多来源+纯人工+区分推理类型）
- **视觉推理困难的根源**: 视觉编码器瓶颈（Prismatic VLMs）、视觉特征解码错位、抽象视觉推理能力有限、难以识别可文字描述的特征
- **CoT对视觉推理效果有限**: 与数学/代码领域的显著提升不同，extended thinking在图表理解上几乎无效，呼应了"thinking makes humans worse"的发现
- **启发**: 未来LVLM需要从架构层面增强视觉推理能力，而非仅靠扩展推理链长度

## 评分

| 维度 | 分数 | 说明 |
|---|---|---|
| 问题重要性 | ⭐⭐⭐⭐⭐ | 揭示LVLM视觉推理的系统性缺陷，问题切中要害 |
| 方法创新性 | ⭐⭐⭐⭐ | 视觉vs文本推理的形式化区分和四类问题分类体系新颖 |
| 实验充分度 | ⭐⭐⭐⭐⭐ | 21个模型+人类基线，多维度分析，错误分类细致 |
| 写作质量 | ⭐⭐⭐⭐⭐ | 动机链条清晰：旧基准不够→合成实验验证→新基准→全面评估→错误分析 |
| 实用价值 | ⭐⭐⭐⭐ | 为LVLM视觉推理改进提供了诊断工具和具体方向 |
| 总分 | 4.6/5 | 高质量基准论文，问题界定精准、实验设计完整 |

<!-- RELATED:START -->

## 相关论文

- [To Think or Not To Think: A Study of Explicit Thinking in Rule-Based Visual Reinforcement Fine-Tuning](think_or_not_think_a_study_of_explicit_thinking_in_rule-based_visual_reinforceme.md)
- [CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models](choice_benchmarking_the_remote_sensing_capabilities_of_large_vision-language_mod.md)
- [Visual Structures Help Visual Reasoning: Addressing the Binding Problem in LVLMs](visual_structures_helps_visual_reasoning_addressing_the_binding_problem_in_vlms.md)
- [Recognition through Reasoning: Reinforcing Image Geo-localization with Large Vision-Language Models](recognition_through_reasoning_reinforcing_image_geo-localization_with_large_visi.md)
- [FineGRAIN: Evaluating Failure Modes of Text-to-Image Models with Vision Language Model Judges](finegrain_evaluating_failure_modes_of_text-to-image_models_with_vision_language_.md)

<!-- RELATED:END -->
