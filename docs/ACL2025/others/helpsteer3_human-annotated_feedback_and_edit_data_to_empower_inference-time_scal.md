---
title: >-
  [论文解读] HelpSteer3: Human-Annotated Feedback and Edit Data to Empower Inference-Time Scaling
description: >-
  [ACL 2025][推理时扩展] NVIDIA 发布 HelpSteer3 数据集（7000+标注员、80+国家），训练专用的 Feedback 和 Edit 模型，在推理时通过"初始响应→反馈→编辑"循环实现开放域通用任务的推理时扩展，基于 Llama 3 系列 70B 模型在 Arena Hard 上达到 92.7 分，超越 OpenAI o1-preview (90.4) 和 DeepSeek R1 (92.3)。
tags:
  - ACL 2025
  - 推理时扩展
  - 人工反馈
  - 反馈-编辑模型
  - 开放域任务
  - RLHF
  - 数据集
---

# HelpSteer3: Human-Annotated Feedback and Edit Data to Empower Inference-Time Scaling

**会议**: ACL 2025  
**arXiv**: [2503.04378](https://arxiv.org/abs/2503.04378)  
**代码**: [有](https://huggingface.co/datasets/nvidia/HelpSteer3)  
**机构**: NVIDIA
**领域**: Others / LLM对齐与推理时扩展  
**关键词**: 推理时扩展, 人工反馈, 反馈-编辑模型, 开放域任务, RLHF, 数据集

## 一句话总结

NVIDIA 发布 HelpSteer3 数据集（7000+标注员、80+国家），训练专用的 Feedback 和 Edit 模型，在推理时通过"初始响应→反馈→编辑"循环实现开放域通用任务的推理时扩展，基于 Llama 3 系列 70B 模型在 Arena Hard 上达到 92.7 分，超越 OpenAI o1-preview (90.4) 和 DeepSeek R1 (92.3)。

## 研究背景与动机

- **推理时扩展的局限**：当前主流推理时扩展技术（如 DeepSeek R1、OpenAI o1）依赖"先思考再回答"范式，核心要求是任务答案**可验证**（如数学、编程、逻辑推理），无法推广到开放域通用任务
- **RLHF 反馈形式不够丰富**：传统 RLHF 仅使用偏好对比（A vs B）或固定维度评分（如正确性、创造性等），缺乏具体指出"哪里有问题、如何改进"的**自然语言反馈**
- **自我反馈效果有限**：直接提示指令模型进行自我反馈和自我编辑，在简单任务上可能有效，但在**高难度任务**（如 Arena Hard 中的复杂编程问题）上几乎无提升甚至退化
- **人类改进范式的启发**：人类在写论文、写代码、做重大决策时，都遵循"初稿→征求反馈→修改"的循环。这种丰富的反馈-改进机制尚未被 LLM 充分利用
- **核心问题**：能否训练专用模型来模仿人类的反馈与编辑能力，使开放域任务也能实现有效的推理时扩展？

## 方法详解

### 整体框架：Feedback-Edit 推理时扩展系统

系统由三个独立模型组成：
1. **初始响应模型**（如 Llama-3.1-Nemotron-70B-Instruct）：生成初始回答
2. **Feedback 模型**：针对初始响应生成详细的自然语言反馈，指出不足并建议改进方向
3. **Edit 模型**：根据反馈对初始响应进行编辑改进

### 数据集构建：HelpSteer3

#### 数据收集流程
- **提示来源**：从 ShareGPT 和 WildChat 抽样，覆盖 General、STEM、Coding、Multilingual 四大类别
- **响应生成**：用 16+ 个不同模型（Nemotron 340B、Mistral Large 2、Gemma 2 等）生成响应，刻意包含不同能力水平的模型以增强泛化性
- **反馈标注**：7000+ 标注员为每个响应提供 3-5 条自然语言反馈（50-250 词），以"The response is {not/slightly/partially/mostly/perfectly} helpful"开头，聚焦整体帮助性评价
- **响应编辑**：将反馈汇总后交给独立标注员池进行响应编辑，仅使用三位最一致标注员的反馈

#### 三个训练数据集
1. **Feedback Demonstration**（81,642 条）：教模型如何生成反馈
2. **Edit Demonstration**（14,461 条）：教模型如何根据反馈编辑响应，包含所有反馈排列以学习顺序无关性
3. **Edit Preference**（3,274 对）：区分好编辑 vs 差编辑（不遵循反馈的编辑/直接复制原文的编辑）

### 模型训练

基于 Llama-3.3-70B-Instruct 初始化：
- **Feedback SFT**：在 Feedback Demonstration 上微调 1 epoch
- **Edit SFT**：在 Edit Demonstration 上微调 1 epoch
- **Edit RM**：在 Edit Preference 上训练 Bradley-Terry 奖励模型，设计每个 batch 同时包含 (差编辑, 好编辑) 和 (不编辑, 好编辑) 对
- **Edit RL**：使用 REINFORCE Leave One Out (RLOO) 在 Edit RM 指导下进一步优化 Edit 模型。RL 训练解决了 SFT 模型约 30% 概率直接复制原响应的问题

### 推理时多维扩展

四个可扩展维度：
- **初始响应数**：每个 prompt 生成多个初始响应（Best-of-N），通过奖励模型选优
- **有效反馈数**：生成更多反馈并根据建设性批评关键词重排序，筛选出有效反馈
- **编辑响应数**：对同一组反馈生成多个编辑版本，选奖励最高的
- **多维联合扩展**：同时扩展多个维度实现最优性能

## 实验

### 实验设置
- **评估指标**：AlpacaEval 2.0 LC（简单）、GPT-4-Turbo MT Bench（中等）、Arena Hard（困难）
- **基准模型**：Llama-3.1-Nemotron-70B-Instruct、Llama-3.3-70B-Instruct
- **外部对比**：Llama-3.1-405B-Instruct、Claude-3.5-Sonnet、GPT-4o、OpenAI o1-preview、DeepSeek R1

### 主实验结果

| 模型 | MT Bench | AlpacaEval LC | Arena Hard |
|------|----------|---------------|------------|
| Nemotron-70B-Instruct | 8.98 | 57.6 | 85.0 |
| + Feedback + Edit | **9.16** | **62.8** | **87.0** |
| Llama-3.3-70B-Instruct | 8.29 | 35.0 | 62.4 |
| + Feedback + Edit | **9.07** | **36.9** | **74.8** |
| GPT-4o-2024-05-13 | 8.74 | 57.5 | 79.3 |
| Claude-3-5-Sonnet | 8.81 | 52.4 | 79.2 |

Feedback-Edit 系统在所有三个指标上显著提升基础模型表现，且长度增幅可控。

### 消融实验

| 设置 | MT Bench | AlpacaEval LC | Arena Hard |
|------|----------|---------------|------------|
| Nemotron-70B 基线 | 8.98 | 57.6 | 85.0 |
| + Self-Feedback + Self-Edit | 9.11 | 64.6 | **84.6** ↓ |
| + Feedback + Self-Edit | 8.94 | 66.2 | 85.4 |
| + Feedback + Edit w/o RL | 9.12 | 64.4 | 86.4 |
| + Edit w/o Feedback | 9.14 | 67.4 | **84.5** ↓ |
| + Feedback + Edit (完整) | **9.16** | 62.8 | **87.0** |

**关键发现**：
- 自我反馈（Self-Feedback）在简单任务有效但在困难任务退化，证明了训练专用模型的必要性
- 移除 RL 后 Edit 模型有 ~30% 概率直接复制原响应不做修改
- 移除 Feedback 后 Arena Hard 反而**低于基线**（84.5 vs 85.0），说明反馈对困难任务至关重要

### 推理时扩展效果

最优配置（8 个初始响应 × 16 条有效反馈 + Nemotron-70B-Select 选择器）：
- Arena Hard: **92.7**，超越 OpenAI o1-preview (90.4) 和 DeepSeek R1 (92.3)
- 仅需约 16x 的 token 生成量（与 Best-of-16 相当），但效果显著优于纯 Best-of-N (88.5)

### 蒸馏实验

| 模型 | AlpacaEval LC | Arena Hard |
|------|---------------|------------|
| Llama-3.1-8B + Distill | 41.5 | 55.5 |
| Llama-3.3-70B + Distill | **61.6** | **88.8** |
| Nemotron-70B + Distill | 61.3 | 88.4 |

蒸馏数据可大幅提升基础模型零样本性能（Llama-3.3-70B Arena Hard: 62.4 → 88.8），适合延迟敏感场景。

## 亮点

- **新颖的推理时扩展范式**：将"反馈-编辑"人类协作模式系统化为 LLM 推理时扩展方法，是首个在开放域通用任务上达到 SOTA 的推理时扩展方案
- **大规模高质量数据集**：7000+ 标注员、80+ 国家、14 种编程语言、13 种自然语言；数据 CC-BY-4.0 开源
- **系统可分解部署**：Feedback/Edit 模型可分别部署在不同计算资源上，采样可并行化，总延迟仅约 2x 贪心生成——远低于 DeepSeek R1 等需要顺序生成大量思考 token 的方案
- **消融设计严谨**：通过 Self-Feedback、Self-Edit、移除 RL、移除 Feedback 等对照实验，清晰量化每个组件的贡献
- **蒸馏可行性**：验证反馈-编辑系统产生的数据可用于蒸馏，适合不同延迟需求

## 局限性

- **计算成本**：最优配置需要生成大量反馈并重排序过滤，采样和选择过程仍有优化空间（如约束解码减少无效反馈）
- **数据时效性**：提示来源于 2023-2024 年的 ShareGPT/WildChat，可能无法代表当前最复杂的用户查询
- **响应长度限制**：跳过了需要 2000+ 词响应或 4000+ 词输入的提示，限制了在长文本场景的适用性
- **仅在 70B 规模验证**：未探索更大（如 405B）或更小（仅蒸馏到 8B）模型的完整 Feedback-Edit 系统效果
- **Edit Preference 数据量偏小**（3,274 对）：可能限制 Edit RM 的泛化能力，且仅覆盖 General/STEM 子集

## 相关工作

- **RLHF 与偏好建模**：Ouyang et al. (2022)、HelpSteer2 (Wang et al., 2024)、UltraFeedback (Cui et al., 2023) 等使用评分或偏好对，本文扩展到自然语言反馈
- **推理时扩展（思考范式）**：OpenAI o1、DeepSeek R1、QwQ 等通过训练模型产生思考链来扩展，受限于可验证任务
- **自我纠正与自我改进**：Self-Refine (Madaan et al., 2023)、Self-Debug (Chen et al., 2023)，本文证明通用模型自我纠正在困难任务上无效
- **批评模型**：CritiqueLLM (Ke et al., 2024)、Critique-out-Loud (Ankner et al., 2024)、Shepherd (Wang et al., 2023) 等训练批评模型，本文进一步加入编辑组件形成闭环
- **Aligner (Ji et al., 2024)**：训练无反馈的编辑模型，本文证明在困难任务上反馈引导的编辑显著优于无反馈编辑

## 评分

⭐⭐⭐⭐ — 在开放域通用任务推理时扩展方面建立了新范式，数据集规模和实验设计扎实，消融分析清晰有力。Arena Hard 上超越 o1-preview 和 DeepSeek R1 具有说服力。局限在于计算成本优化和数据规模的进一步扩展空间。

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Learning to Reason from Feedback at Test-Time](learning_to_reason_from_feedback_at_test-time.md)
- [\[ACL 2025\] TestNUC: Enhancing Test-Time Computing Approaches and Scaling through Neighboring Unlabeled Data Consistency](testnuc_enhancing_test-time_computing_approaches_and_scaling_through_neighboring.md)
- [\[ACL 2025\] A Little Human Data Goes A Long Way](a_little_human_data_goes_a_long_way.md)
- [\[ACL 2025\] Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback](hybrid_preferences_learning_to_route_instances_for_human_vs_ai_feedback.md)
- [\[ACL 2025\] One for All: Update Parameterized Knowledge Across Multiple Models with Once Edit](one_for_all_update_parameterized_knowledge_across_multiple_models_with_once_edit.md)

<!-- RELATED:END -->
