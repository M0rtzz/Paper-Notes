---
title: >-
  [论文解读] ATGen: A Framework for Active Text Generation
description: >-
  [ACL 2025][文本生成][主动学习] 提出ATGen——首个系统化的NLG主动学习框架，集成SOTA AL策略、人工/LLM标注界面、PEFT高效训练和vLLM推理优化，在TriviaQA/GSM8K等4个NLG任务上验证主动学习可将标注成本降低2-4倍。
tags:
  - ACL 2025
  - 文本生成
  - 主动学习
  - NLG
  - 标注效率
  - LLM标注
  - 框架
---

# ATGen: A Framework for Active Text Generation

**会议**: ACL 2025  
**arXiv**: [2506.23342](https://arxiv.org/abs/2506.23342)  
**代码**: [GitHub](https://github.com/Aktsvigun/atgen)  
**领域**: NLP生成 / 主动学习  
**关键词**: 主动学习, NLG, 标注效率, LLM标注, 框架  

## 一句话总结

提出ATGen——首个系统化的NLG主动学习框架，集成SOTA AL策略、人工/LLM标注界面、PEFT高效训练和vLLM推理优化，在TriviaQA/GSM8K等4个NLG任务上验证主动学习可将标注成本降低2-4倍。

## 研究背景与动机

- **领域现状**：NLG任务（摘要、问答、推理）快速发展，但领域特定任务仍需高质量标注数据。LLM标注可部分替代人工但成本高昂。
- **现有痛点**：(1)现有AL框架只支持分类/序列标注，不支持NLG；(2)NLG的AL策略评估缺乏统一平台；(3)现代LLM的AL需要PEFT和高效推理支持。
- **核心矛盾**：NLG任务的标注成本高 vs 缺乏减少标注的系统化工具。
- **本文目标**：构建统一的NLG主动学习框架，降低标注成本。
- **切入角度**：集成策略+标注+训练+评估的全栈框架。
- **核心 idea**：将AL系统化应用于NLG，降低人工和LLM API标注成本。

## 方法详解

### 整体框架

ATGen提供：(1)AL策略集合（HUDS/HADAS/Facility Location等）；(2)Web GUI人工标注；(3)LLM自动标注（支持OpenAI/Anthropic/本地模型）；(4)LoRA/QLoRA高效训练；(5)vLLM/SGLang推理加速；(6)Benchmarking脚本。

### 关键设计

**设计1：NLG专用AL策略集成**
- **功能**：实现并统一接口化所有SOTA NLG AL策略。
- **核心思路**：包括HUDS（不确定性+度量学习）、HADAS（幻觉感知）、Facility Location（子模函数）、BLEUVar、IDDS等策略。
- **设计动机**：分类任务的AL策略（如least confidence）在NLG中表现不佳，需要专门评估。

**设计2：双模标注支持**
- **功能**：同时支持人工标注和LLM标注两种模式。
- **核心思路**：人工模式推荐用ED（实验设计）策略一次性选择后标注；LLM模式支持OpenAI batch API（便宜50%）。
- **设计动机**：人工标注受AL迭代延迟影响大，ED策略消除了重训和查询的等待时间。

**设计3：高效训练推理集成**
- **功能**：支持LoRA/QLoRA/DoRA + vLLM/SGLang/Unsloth。
- **核心思路**：AL循环需要多次微调和推理，PEFT和高效推理框架使大模型AL成为可能。
- **设计动机**：大模型的AL若无高效训练推理则完全不可行。

### 损失函数/训练策略

各AL策略有不同的查询分数计算方式。训练使用标准causal LM损失+PEFT。评估用EM/F1/ROUGE-2/AlignScore。

## 实验关键数据

### 主实验

**TriviaQA（人工标注模拟，Qwen3-1.7B）**

| 策略 | 4%数据时EM | 12%数据时EM |
|------|-----------|------------|
| Random | ~30 | ~42 |
| HUDS | ~42 | ~48 |
| HADAS | ~40 | ~46 |
| Facility Location | ~38 | ~45 |

### 消融实验

| 维度 | 发现 |
|------|------|
| 人工 vs LLM标注 | LLM标注在GSM8K上整体质量下降几个百分点 |
| ED vs AL | ED在标注延迟敏感场景更优 |
| 不同acquisition模型 | Qwen3-1.7B效果良好 |

### 关键发现

1. HUDS、HADAS和Facility Location三个策略在多个任务上一致地显著超越随机采样。
2. AL在LLM标注场景下同样有效，可减少2-4倍API调用成本。
3. 数学推理任务中DeepSeek-R1标注仍有错误累积，说明领域专业任务仍需人工标注。

## 亮点与洞察

1. 首个完整的NLG AL框架，填补了重要工具空白。
2. 双模标注（人工+LLM）的设计切合当前AI辅助标注的趋势。
3. 开源MIT许可，社区友好。

## 局限与展望

1. 未研究AL引入的数据分布偏差问题。
2. 大规模LLM的AL计算开销仍然显著。
3. 评估集中在英文任务，多语言场景未覆盖。

## 相关工作与启发

- ALToolbox只支持分类/信息抽取，ATGen扩展到NLG。
- 启发：在LLM时代AL仍有价值——减少API成本而非仅减少人工。

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ★★★☆☆ |
| 实用性 | ★★★★★ |
| 实验充分性 | ★★★★☆ |
| 写作清晰度 | ★★★★☆ |
