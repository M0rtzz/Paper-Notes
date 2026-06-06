---
title: >-
  [论文解读] LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?
description: >-
  [ACL 2025][LLM/NLP][场景图] 提出 TSG Bench 基准，系统评估 11 个 LLM 在场景图理解和生成任务上的能力，揭示 LLM 在场景图生成（尤其是多动作分解）方面存在显著瓶颈。
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "场景图"
  - "基准测试"
  - "大语言模型"
  - "结构化表示"
  - "多模态推理"
---

# LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?

**会议**: ACL 2025  
**arXiv**: [2505.19510](https://arxiv.org/abs/2505.19510)  
**代码**: [GitHub](https://github.com/docworlds/tsg-bench)  
**领域**: LLM/NLP  
**关键词**: 场景图, 基准测试, 大语言模型, 结构化表示, 多模态推理

## 一句话总结

提出 TSG Bench 基准，系统评估 11 个 LLM 在场景图理解和生成任务上的能力，揭示 LLM 在场景图生成（尤其是多动作分解）方面存在显著瓶颈。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：大语言模型在文本任务上表现出色，但将其能力拓展到需要空间和时间推理的多模态环境时面临挑战。场景图（Scene Graph）是一种结构化表示，编码了场景中实体、属性和关系，广泛应用于具身 AI、机器人、3D 环境建模等领域。

然而，LLM 对场景图的利用能力缺乏系统性评估。已有工作主要聚焦于图像-场景图对，而文本-场景图的研究有限。现有基准如 FACTUAL 仅关注静态场景，不适用于动态、真实世界场景。关键问题包括：LLM 是否真正理解场景图的空间和语义结构？在处理长上下文和复杂三元组时是否会出现误解？

本文引入 TSG Bench，旨在系统评估 LLM 在场景图**理解**和**生成**两个维度上的能力，填补该领域的评估空白。

## 方法详解

### 整体框架

TSG Bench 基于叙事文本和对应的动态场景图序列构建，包含 120 个真实世界场景、2,041 个描述和 4,289 个场景图。基准设计了四个评估任务：两个理解任务（SGQA、SGDS）和两个生成任务（SA-SGG、MA-SGG）。

数据表示上，叙事 $D = (d_1, \dots, d_n)$ 由多个连贯的自然语言描述组成，每个描述 $d_i$ 对应一组场景图 $G_i = (G_{i1}, \dots, G_{ik})$，其中 $k \in [1, 8]$ 取决于描述的复杂度。每个场景图 $G_{ij} = (V_{ij}, E_{ij})$ 包含节点（person, action, object, hand 四类）和边（verb, dobj, preposition 三类）。

### 关键设计

**1. 场景图理解任务：SGQA + SGDS**

- **功能**: 评估 LLM 对场景图的推理和解读能力
- **核心思路**: SGQA 要求模型基于场景图序列进行多跳推理回答问题；SGDS 是选择题形式，要求模型在给定上下文和场景图后选出正确描述
- **设计动机**: 理解任务需要跨多个三元组的逻辑/时序连接，模拟实际应用中的上下文推理需求

**2. 场景图生成任务：SA-SGG + MA-SGG**

- **功能**: 评估 LLM 从文本生成结构化场景图的能力
- **核心思路**: SA-SGG 针对单动作描述生成场景图；MA-SGG 针对包含多个动作的复杂描述，需要先将描述分解为多个离散动作再分别生成场景图
- **设计动机**: 生成任务要求模型从自然语言中解析语义相似元素并构建三元组，MA-SGG 额外考验动作分解和排序能力

**3. 数据构建：Human-in-the-loop 多轮流程**

- **功能**: 保证基准数据的质量和多样性
- **核心思路**: 基于 EASG 数据集，通过 LLM 生成 + 人工审核的多轮流程构建叙事和场景图，包括句子生成、图生成、人工校验、同义改写和句子合并等步骤
- **设计动机**: 纯 LLM 生成的场景图往往不完整，需要人工逐元素检查和精修

### 损失函数 / 训练策略

本文为评估基准，不涉及模型训练。评估采用 zero-shot 设置，使用 Exact Match（SGQA）、Accuracy（SGDS）和 Precision/Recall/F1（SGG 任务）作为指标。

## 实验关键数据

### 主实验

| 模型 | SGDS Acc | SGQA EM | SA-SGG F1 | MA-SGG F1 |
|------|----------|---------|-----------|-----------|
| Human | 98.33 | 88.00 | 82.50 | 75.60 |
| Claude-3.5-Sonnet | 98.40 | 90.60 | 68.43 | 58.80 |
| GPT-4o | 96.40 | 84.80 | 59.23 | 43.99 |
| LLaMA-3.3-70B | 97.60 | 84.60 | 33.37 | 28.92 |
| Qwen-2.5-72B | 96.80 | 81.40 | 54.42 | 36.78 |
| DeepSeek-V3 | 96.40 | 79.60 | 54.45 | 39.34 |
| Qwen-2.5-7B | 93.60 | 73.40 | 9.39 | 6.34 |

### 消融实验

| 方法 | SGQA | SA-SGG F1 | MA-SGG F1 |
|------|------|-----------|-----------|
| Claude-3.5-Sonnet (zero-shot) | 90.60 | 68.43 | 58.80 |
| + CoT | 94.00 | 69.57 | 64.36 |
| + 10-shot | 92.00 | 75.29 | 71.75 |
| GPT-4o (zero-shot) | 84.80 | 59.23 | 43.99 |
| + CoT | 90.00 | 67.13 | 44.79 |
| + 10-shot | 84.40 | 65.78 | 57.40 |

### 关键发现

1. **理解 vs 生成的性能鸿沟**: LLM 在理解任务（SGDS ≥90%）表现优异，但在生成任务上与人类差距巨大（MA-SGG 最高仅 58.80 vs 人类 75.60）
2. **动作分解是核心瓶颈**: 多动作场景图生成中，模型难以正确分解隐式和重复动作
3. **10-shot ICL 对生成任务最有效**: 在 MA-SGG 上，Claude 使用 10-shot 可将 F1 从 58.80 提升至 71.75
4. **小模型在生成任务上几乎失败**: Qwen-2.5-7B 和 Mistral-7B 在 MA-SGG 上 F1 < 12
5. **LLM 存在场景图幻觉问题**: 较小模型生成了大量不在预定义集合 $L$ 中的元素

## 亮点与洞察

- 首个系统评估 LLM 在文本-场景图理解和生成能力的基准，填补了重要空白
- 通过分解实验（节点生成、边生成、动作分解）精确定位了性能瓶颈
- 发现 LLM 对隐式动作和重复动作的处理能力严重不足，揭示了数值感知的薄弱环节
- 场景图纠错实验发现，提供错误类型后模型修正能力显著提升（Claude 从 60.03 到 88.28）

## 局限与展望

- 当前基准仅涉及单个演员、不包含对象属性（颜色、大小等）
- 动作分解的复杂度有限（最多 8 个子场景图），真实场景可能更复杂
- 仅评估了文本模态，未来可结合 VLM 从图像/视频提取叙事并生成场景图
- 评估指标基于三元组完全匹配，可能忽略语义相近但表达不同的正确输出

## 相关工作与启发

- **场景图表示**: 在 3D 重建、交互游戏、机器人导航等任务中广泛应用，LLM 与场景图的结合正在成为新方向
- **启发**: LLM 在结构化输出生成方面的能力远不如理解能力，这对于依赖 LLM 理解环境的具身 AI 系统具有重要警示意义

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个文本-场景图的 LLM 评估基准，问题定义清晰
- **实验充分度**: ⭐⭐⭐⭐⭐ 11 个 LLM + 多种 prompting 策略 + 细粒度分析 + 纠错/幻觉实验
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，任务定义规范，图表丰富
- **价值**: ⭐⭐⭐⭐ 为场景图相关的 LLM 应用研究提供了重要的评估工具和发现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AAD-LLM: Neural Attention-Driven Auditory Scene Understanding](aad-llm_neural_attention-driven_auditory_scene_understanding.md)
- [\[ACL 2025\] Can Large Language Models Understand Internet Buzzwords Through User-Generated Content](buzzword_understanding_ugc.md)
- [\[ACL 2025\] Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?](graph_descriptive_order_llm.md)
- [\[ACL 2025\] Past Meets Present: Creating Historical Analogy with Large Language Models](past_meets_present_creating_historical_analogy_with_large_language_models.md)
- [\[ACL 2025\] Can Large Language Models Accurately Generate Answer Keys for Health-related Questions?](can_large_language_models_accurately_generate_answer_keys_for_health-related_que.md)

</div>

<!-- RELATED:END -->
