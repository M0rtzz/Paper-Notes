---
title: >-
  [论文解读] M3FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset
description: >-
  [ACL 2025][多语言/翻译][金融会议] 构建了 M3FinMeeting——首个面向金融会议的多语言（中英日）、多行业、多任务评测基准，包含 600 场真实金融会议的摘要、QA 对抽取和问答三项任务，揭示了当前最先进 LLM 在金融会议理解上仍有显著提升空间。 现有金融 NLP 基准（FinQA、ConvFinQA…
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "金融会议"
  - "多语言基准"
  - "长上下文理解"
  - "摘要"
  - "问答"
---

# M3FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset

**会议**: ACL 2025  
**arXiv**: [2506.02510](https://arxiv.org/abs/2506.02510)  
**代码**: [有](https://github.com/aliyun/qwen-dianjin)  
**领域**: 多语言翻译  
**关键词**: 金融会议, 多语言基准, 长上下文理解, 摘要, 问答

## 一句话总结

构建了 M3FinMeeting——首个面向金融会议的多语言（中英日）、多行业、多任务评测基准，包含 600 场真实金融会议的摘要、QA 对抽取和问答三项任务，揭示了当前最先进 LLM 在金融会议理解上仍有显著提升空间。

## 研究背景与动机

现有金融 NLP 基准（FinQA、ConvFinQA、CFLUE 等）存在三大局限：

**数据来源单一**：主要依赖新闻文章、财报、公告，缺少真实金融会议内容。金融会议具有对话性、实时性和策略讨论等独特特征，现有数据无法覆盖

**语言单一**：几乎都局限于英文或中文

**缺乏长上下文挑战**：金融会议通常持续 1-2 小时，转录文本常超过 10K tokens，对 LLM 的长上下文处理能力是真正的考验

M3FinMeeting 旨在填补这些空白，评估 LLM 在真实金融会议理解中的综合能力。

## 方法详解

### 整体框架

M3FinMeeting 是一个评测基准数据集，核心设计围绕"三个多"展开：

- **多语言**：英语（100 场）、中文（400 场）、日语（100 场），共 600 场会议
- **多行业**：覆盖 GICS 标准下全部 11 个行业板块（通信、科技、金融、能源等）
- **多任务**：摘要生成、QA 对抽取、问答三项任务

### 关键设计

#### 1. 数据收集与标注

**功能**：从合作金融机构获取真实会议音频，经 ASR 转录后人工校正。

**核心流程**：
- 采集标准：时效性（近年会议）、长度（优先长音频）、行业覆盖性、权威性
- 使用 Whisper 进行语音转文本，再由标注员逐段校正
- 平均每场会议约 1 小时，英文平均 10,086 tokens、中文约 11,740 tokens、日文约 13,284 tokens
- 严格排除敏感信息和个人身份信息

**设计动机**：直接使用真实金融会议而非合成数据，确保基准反映真实世界的挑战。

#### 2. 三项评测任务

**摘要生成**：LLM 需隐式识别文档中的不同主题段落，为每段生成摘要，再拼接为完整文档摘要。评估使用段级 P/R/F1（基于 cosine 相似度 ≥ 0.75 对齐）+ GPT-4-Judge 打分（覆盖度、冗余度、可读性、准确性、一致性五维度，0-100 分）。

**QA 对抽取**：从会议全文中识别有意义的问题及其对应答案。需要 LLM 理解对话结构，区分有意义问题和无意义插话，正确配对多轮问答。

**问答**：给定会议全文和一组预设问题，LLM 需在长上下文中定位证据并生成答案。将相关问题合并为一个 prompt，模拟写报告/综述的实际场景。

#### 3. 评估体系

**功能**：多层次评估，兼顾自动指标和人工判断。

- 段级 Precision/Recall/F1：基于嵌入相似度对齐生成与参考摘要
- GPT-4-Judge：五维度 0-100 打分，与 Qwen-plus-Judge 交叉验证
- 人工评估 + Fleiss' Kappa：验证 LLM 评估与人类一致性

### 损失函数 / 训练策略

本文是评测基准，不涉及模型训练。评测采用零样本设置。

## 实验关键数据

### 主实验（表格）

**三任务综合评估（GPT-4-Judge 分数）**：

| 模型 | 摘要 | QA 对抽取 | 问答 | 综合 |
|------|------|-----------|------|------|
| GPT-3.5-turbo | 44.56 | 31.13 | 42.78 | 39.55 |
| LLaMA3.1-8B | 52.01 | 44.64 | 40.01 | 45.76 |
| GLM4-9B-Chat | 67.71 | 46.06 | 67.72 | 60.76 |
| Qwen2-7B | 73.59 | 37.33 | 69.99 | 60.71 |
| GPT-4o | 73.61 | 66.85 | 71.79 | 70.66 |
| Qwen2-72B | 74.17 | 60.85 | 73.50 | 69.66 |
| **Qwen2.5-72B** | **74.51** | **68.03** | **74.81** | **72.54** |

**QA 对抽取的 F1 分数极低**：最好的 Qwen2.5-72B 也仅 38.41%，说明从长对话中自动抽取高质量 QA 对极其困难。

### 消融实验（表格）

**RAG 对问答性能的影响（Qwen2.5-72B，GPT-4-Judge）**：

| 方法 | <5K | 5-10K | 10-15K | 15-20K | >20K |
|------|-----|-------|--------|--------|------|
| Baseline 1 (一次回答全部) | 中 | 高 | 高 | 最好 | 最好 |
| Baseline 2 (逐个回答) | 中 | 中 | 高 | 次好 | 次好 |
| RAG (top 5) | 好 | 好 | 中 | 差 | 差 |
| RAG (top 1) | 差 | 差 | 差 | 最差 | 最差 |

关键发现：**在长文档（>10K tokens）上，完整上下文优于 RAG**，这与直觉相反。

### 关键发现

1. **Qwen2.5-72B 综合最优**，但仍仅 72.54 分（满分 100），表明巨大提升空间
2. **摘要任务**：段级 F1 不到 30%，LLM 在隐式文档分段上表现很差
3. **QA 抽取最难**：即便最好的模型召回率也不到 50%，漏掉了超过一半的关键问题
4. **语言效果**：多数模型在日语上最好，中英无明显差距；可能因为日语指令遵循更好
5. **行业差异**：通信、消费者可选和 IT 板块表现较好，不同行业差异在弱模型上更显著
6. **长度影响**：GPT-3.5 在 >15K tokens 时急剧退化（因 16K 窗口限制），Qwen2.5-72B 和 GPT-4o 在长文档上稳定
7. **LLM 评估可靠**：GPT-4-Judge 与 Qwen-plus-Judge 趋势一致，与 5 位人类标注者的 Fleiss' Kappa = 0.701

## 亮点与洞察

- **首个金融会议专属基准**：填补了会议场景的空白，与新闻/报告数据有本质区别
- **RAG 在长上下文反而不如直接全文输入**：这对 RAG 应用的实践有重要参考价值
- **多维度评估体系完善**：自动+LLM-Judge+人工，交叉验证消除偏差
- **揭示 QA 对抽取为最难任务**：这对未来金融 NLP 研究方向有指导意义

## 局限与展望

1. 标注成本极高（1-2 小时音频 + 10K+ tokens 文本需专业分析师标注）
2. 问答任务仅使用抽取出的 QA 对，未评估需深度推理的开放性问题
3. 中文会议数量（400）远多于英文和日文（各 100），数据不均衡
4. 仅评估了 7 个 LLM，未覆盖更多开源模型（如 Mistral、DeepSeek 等）
5. ASR 误差即使经人工校正仍可能存在，对下游任务有潜在影响

## 相关工作与启发

- 与 ECTSum（财报电话会议摘要）互补，但 M3FinMeeting 语言更多、行业更广、任务更全
- 借鉴 LongBench、RULER 等长上下文评测设计，但聚焦金融垂直领域
- GPT-4-Judge 评估方法已成为主流，本文增加了 Qwen-plus 交叉验证和人工 Kappa 计算，更具说服力

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个多语言多行业金融会议理解基准，场景定义有价值
- **实验充分度**: ⭐⭐⭐⭐ — 7 个模型、3 个任务、多语言/多行业/多长度分析，RAG 对比，人工评估
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，统计表格详实，评估方法论完善
- **价值**: ⭐⭐⭐⭐ — 对金融 NLP 社区有明确贡献，数据集和发现（如 RAG 在长文档反而劣势）有实际参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] M-MAD: Multidimensional Multi-Agent Debate for Advanced Machine Translation Evaluation](m-mad_multidimensional_multi-agent_debate_for_advanced_machine_translation_evalu.md)
- [\[ACL 2025\] Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding](code-switching_red-teaming_llm_evaluation_for_safety_and_multilingual_understand.md)
- [\[ACL 2025\] Multi-perspective Alignment for Increasing Naturalness in Neural Machine Translation](multi-perspective_alignment_for_increasing_naturalness_in_neural_machine_transla.md)
- [\[ACL 2025\] CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution](cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)
- [\[ACL 2025\] EXECUTE: A Multilingual Benchmark for LLM Token Understanding](execute_a_multilingual_benchmark_for_llm_token_understanding.md)

</div>

<!-- RELATED:END -->
