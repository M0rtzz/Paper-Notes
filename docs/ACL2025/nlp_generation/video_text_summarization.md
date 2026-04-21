---
title: >-
  [论文解读] What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations
description: >-
  [ACL 2025][文本生成][video summarization] 提出VISTA数据集——18,599个AI会议演讲视频与论文摘要配对，并引入plan-based摘要框架，通过生成中间问题序列引导科学视频的结构化摘要生成，显著提升事实一致性。
tags:
  - ACL 2025
  - 文本生成
  - video summarization
  - scientific presentations
  - plan-based generation
  - 多模态
  - dataset
---

# What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations

**会议**: ACL 2025  
**arXiv**: [2502.08279](https://arxiv.org/abs/2502.08279)  
**代码**: [dongqi.me/projects/VISTA](https://dongqi.me/projects/VISTA)  
**领域**: 视频文本摘要  
**关键词**: video summarization, scientific presentations, plan-based generation, multimodal, dataset  

## 一句话总结

提出VISTA数据集——18,599个AI会议演讲视频与论文摘要配对，并引入plan-based摘要框架，通过生成中间问题序列引导科学视频的结构化摘要生成，显著提升事实一致性。

## 研究背景与动机

**问题定义：** 将录制的学术演讲视频转化为简洁准确的文本摘要（即视频到文本摘要）是多模态学习中日益重要的挑战。现有数据集主要针对通用内容（YouTube、电影、新闻），缺乏针对学术科学视频的专门数据集和方法。

**现有方法局限：** 大型多模态模型（LMM）在科学场景中表现下降，特别是在处理技术术语和科学视觉元素（图表）时。端到端的摘要生成方法难以捕捉科学摘要的固有结构（背景-方法-结果-结论），导致生成内容缺乏条理性和事实准确性。

**核心动机：** 科学摘要通常遵循相对固定的结构，适合采用结构化生成策略。通过引入中间计划（plan）显式建模摘要的潜在结构，可以更好地引导摘要生成过程。

## 方法详解

### 整体框架

系统分为两个独立训练的模块：
1. **Plan Generation (PG) 模块：** 输入视频 $v$，生成计划 $p = \{q_1, q_2, \ldots, q_m\}$（一系列问题序列）
2. **Summary Generation (SG) 模块：** 输入视频与计划的拼接 $[v; p]$，生成最终摘要 $s$

推理时先用PG预测计划 $\hat{p}$，再将 $[v; \hat{p}]$ 送入SG生成摘要。学习目标从 $P(s|v)$ 扩展为 $P(s|v,p)$。

### 关键设计

1. **VISTA数据集构建：** 从ACL Anthology（ACL、EMNLP、NAACL、EACL等）和ML会议（ICML、NeurIPS）收集2020-2024年的演讲视频与论文摘要配对。排除教程/特邀报告和过短/过长视频（<1分钟或>30分钟），最终得到18,599个样本。平均视频6.76分钟、16.36个镜头；平均摘要192.6个token、7.19个句子

2. **基于QUD理论的Plan生成：** 受Question Under Discussion理论启发，假设摘要中的每个句子可视为对特定问题的回答。使用GPT-o1根据参考摘要句子和前序上下文生成银标准（silver-standard）的问题序列，问题顺序与原始摘要句子顺序一致，保证计划的连贯性

3. **全面的基准评估体系：** 涵盖文本→文本（LLaMA-3.1 + transcript/OCR）、音频→文本（Qwen2-Audio）、视频→文本（多种LMM）三类模态基线，以及零样本、QLoRA、全量微调三种训练设置

### 损失函数

PG和SG模块均采用标准自回归语言建模损失。PG在 $(v, p)$ 对上训练，SG在 $([v;p], s)$ 元组上训练。两个模块共享相同骨干但独立训练。

## 实验

### 主实验结果：零样本评估（摘选）

| 方法 | 开源 | R1 | R2 | RLsum | BLEU | BERTscore | FactVC |
|------|------|------|------|-------|------|-----------|--------|
| LLaMA-3.1 (transcript) | ✓ | 23.68 | 4.22 | 21.39 | 2.70 | 80.93 | 34.32 |
| Claude 3.5 Sonnet | ✗ | 27.71 | 5.59 | 24.14 | 3.14 | 82.57 | 50.11 |
| GPT-o1 | ✗ | 27.90 | 5.69 | 24.37 | 4.38 | 82.63 | 51.36 |
| mPLUG-Owl3 | ✓ | 25.57 | 4.82 | 22.84 | 2.99 | 81.39 | 42.07 |
| **Plan-mPLUG-Owl3** | ✓ | **25.62** | **4.95** | **22.97** | **3.14** | **81.45** | **47.37** |

### QLoRA微调结果（摘选）

| 方法 | R1 | R2 | RLsum | BLEU | BERTscore | FactVC |
|------|------|------|-------|------|-----------|--------|
| mPLUG-Owl3 | 33.40 | 12.82 | 30.66 | 8.29 | 83.49 | 70.08 |
| **Plan-mPLUG-Owl3** | **33.52** | **13.01** | **31.10** | 8.33 | **83.53** | **73.11** |
| LLaVA-NeXT-Interleave | 33.37 | 12.77 | 30.56 | 8.30 | 83.47 | 66.14 |

### 关键发现

1. **Plan-based方法一致提升：** 在零样本和微调设置下，Plan-mPLUG-Owl3均显著超越端到端mPLUG-Owl3，特别是事实一致性FactVC提升最为显著（零样本：42.07→47.37；QLoRA：70.08→73.11）
2. **领域微调效果显著：** QLoRA微调后R1从~25提升至~33，R2从~5提升至~13，表明领域内数据对科学视频摘要至关重要
3. **视频模型优于纯文本/音频模型：** 在相同条件下，视频LMM普遍优于仅使用转录文本或音频的模型，说明视觉信息（幻灯片、图表）对科学演讲理解有价值
4. **闭源模型领先但差距可缩小：** 零样本下GPT-o1/Gemini 2.0显著领先开源模型，但微调后开源模型大幅缩小差距
5. **模型与人类差距仍然显著：** 即使最佳模型在多个指标上仍明显低于人类水平

## 亮点

- 首个大规模科学演讲视频摘要数据集，18,599个样本覆盖ACL/EMNLP/NAACL/ICML/NeurIPS等顶会
- Plan-based框架巧妙利用科学摘要的结构化特性，通过QUD理论将摘要结构显式建模为问题序列
- 全面的实验设计：覆盖三种模态输入、三种训练设置、十余个模型，提供了详尽的基准
- 包含人类评估和错误分析，深入诊断模型生成摘要的关键问题

## 局限性

- 仅覆盖计算语言学和机器学习领域的会议，未扩展到其他学科（生物、物理等）
- Plan生成依赖GPT-o1生成的银标准问题，质量上限受限于GPT-o1的理解能力
- 以论文摘要作为视频摘要的代理（proxy），两者在信息侧重点上可能不完全一致
- 数据仅覆盖英语，未考虑多语言科学演讲场景

## 相关工作

- **视频文本摘要数据集：** VideoXum (Lin et al., 2023)、MMSum (Qiu et al., 2024)、Instruct-V2Xum (Hua et al., 2024)
- **科学文本摘要：** TalkSumm (Lev et al., 2019) 学术视频转录摘要、ACLSum (Takeshita et al., 2024) ACL论文摘要
- **Plan-based摘要：** Narayan et al. (2021, 2023) 基于实体/QA的计划框架、Blueprint models (Liu et al., 2023) 视觉叙事的蓝图框架
- **科学多模态：** M3AV (Chen et al., 2024) 支持ASR/TTS/幻灯片脚本生成

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [DTCRS: Dynamic Tree Construction for Recursive Summarization](dtcrs_dynamic_tree_construction_for_recursive_summarization.md)
- [Context-Aware Hierarchical Merging for Long Document Summarization](context-aware_hierarchical_merging_for_long_document_summarization.md)
- [An Empirical Study of Many-to-Many Summarization with Large Language Models](an_empirical_study_of_manytomany_summarization.md)
- [Theme-Explanation Structure for Table Summarization Using Large Language Models](theme-explanation_structure_for_table_summarization_using_large_language_models_.md)
- [PerSphere: A Comprehensive Framework for Multi-Faceted Perspective Retrieval and Summarization](persphere_a_comprehensive_framework_for_multi-faceted_perspective_retrieval_and_.md)

<!-- RELATED:END -->
