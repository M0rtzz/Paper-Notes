---
title: >-
  [论文解读] SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models
description: >-
  [ICLR 2026][LLM推理][benchmark] 提出SealQA挑战基准（含Seal-0/Seal-Hard/LongSeal三种变体），每道题均经NLP研究者精心设计以触发歧义/冲突/噪声搜索结果，GPT-5最高仅43.2%准确率，揭示test-time scaling在噪声检索下不产生可靠增益。
tags:
  - ICLR 2026
  - LLM推理
  - benchmark
  - search-augmented LLM
  - RAG
  - noisy retrieval
  - test-time scaling
  - knowledge conflict
---

# SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models

**会议**: ICLR 2026  
**arXiv**: [2506.01062](https://arxiv.org/abs/2506.01062)  
**代码**: [HuggingFace](https://huggingface.co/datasets/vtllms/sealqa)  
**领域**: llm_reasoning  
**关键词**: benchmark, search-augmented LLM, RAG, noisy retrieval, test-time scaling, knowledge conflict

## 一句话总结

提出SealQA挑战基准（含Seal-0/Seal-Hard/LongSeal三种变体），每道题均经NLP研究者精心设计以触发歧义/冲突/噪声搜索结果，GPT-5最高仅43.2%准确率，揭示test-time scaling在噪声检索下不产生可靠增益。

## 研究背景与动机

**领域现状**：LLM已进入test-time scaling新范式，推理模型可分解问题、决定何时搜索、融合检索内容到推理路径中。前沿模型在MMLU等传统基准上已超过90%准确率，现有评估趋于饱和。

**现有痛点**：多数搜索增强LLM评估聚焦短事实性查询，top-ranked结果即可直接回答，仅需浅层理解。这无法反映真实搜索的混乱本质——返回的文档可能过时、误导或表面相关但实际无用。

**核心矛盾**：真实信息检索需要深层推理来过滤不一致信息、调和矛盾、识别可信信号，但现有基准无法模拟这些挑战。部分原因在于此类数据集难以大规模策划和验证。

**本文方案**：提出SealQA，一个小而极具挑战性的基准，每道题由NLP研究者精心设计，经多轮严格审核，专门触发歧义/冲突/噪声搜索结果。包含三种变体覆盖不同维度的搜索增强推理挑战。

## 方法详解

### 整体框架

SealQA包含三种变体：
1. **Seal-0**（111题）：核心集，每道题在GPT-4o、GPT-4.1等多个前沿模型10-15次尝试中准确率均为0%
2. **Seal-Hard**（254题）：包含Seal-0及其他未达严格零准确率阈值但仍极具挑战性的问题
3. **LongSeal**（254题）：needle-in-a-haystack变体，每题配一个gold文档和最多50个hard negative，测试长上下文多文档推理

问题横跨5类：高级推理 $\mathcal{Q}_1$（72.4%）、实体/事件消歧 $\mathcal{Q}_2$（58.3%）、时间追踪 $\mathcal{Q}_3$（13.7%）、跨语言推理 $\mathcal{Q}_4$（5.5%）、虚假前提检测 $\mathcal{Q}_5$（4.3%）。

### 关键设计1: 对抗性数据收集流程

**功能**：确保每道题对前沿LLM构成实质挑战。

**核心思路**：每道题由NLP研究者编写，经过严格的多轮审核流程——首先由2+名研究生级审核者审查，再经专家批准。对Seal-0，每道题迭代精炼直至GPT-4o、GPT-4.1等多个模型在10-15次尝试中全部失败。每道题平均开发时间超过1小时（约45分钟起草+额外审核修订时间），6名NLP研究者历时8个月。

**设计动机**：通过对抗性收集避免数据污染问题，确保基准难度随时间保持有效。小规模基准降低API评估成本，允许更频繁更新。

### 关键设计2: LongSeal多文档推理构建

**功能**：测试模型在大量干扰文档中识别和利用相关证据的能力。

**核心思路**：每道Seal-Hard题配备一组检索文档——1个gold文档（来自标注者提供的网页）和最多50个hard negative。hard negative通过Google检索top-10网页、限制2023年前内容的额外10页、以及GPT-4o-mini生成的3个语义相关查询获取。使用GPT-4o-mini过滤可能推断出正确答案的negative。

**设计动机**：测试在噪声检索条件下的长上下文推理，考察位置偏差和相关性建模能力。

### 评估协议

采用GPT-4o-mini自动评分器（改编自SimpleQA），取问题、预测答案和参考答案作为输入，标记"correct"/"incorrect"/"not attempted"。人工评估100个答案，与自动评分器一致率达98%。

## 实验关键数据

### 主实验

| 模型 | Seal-0 (w/o search) | Seal-0 (w/ search) | Seal-Hard (w/o search) | Seal-Hard (w/ search) |
|------|---------------------|--------------------|-----------------------|----------------------|
| GPT-4o | 0.0% | 0.0%† | 11.8% | 15.0%† |
| GPT-4.1 | 0.0% | 0.0%† | 15.0% | 20.5%† |
| o3-mini-high | 3.6% | 1.8% | 12.6% | 14.2% |
| o3-high | - | 14.4%† | - | 32.7%† |
| GPT-5-high | 15.3% | **43.2%**† | 37.8% | **63.8%**† |
| DeepSeek-R1-671B | 5.4% | 1.8% | **22.4%** | 11.0% |
| Qwen3-235B | 0.0% | 5.4% | 4.3% | 11.4% |
| Llama-4-Scout | 0.0% | 0.0% | 5.9% | 5.9% |

†使用ChatGPT内置搜索；其余使用FreshPrompt。

### 消融实验：Test-time Scaling效果

| 模型 | Low Effort | Medium Effort | High Effort |
|------|-----------|---------------|-------------|
| o3-mini (Seal-0) | 1.8% | 2.7% | 1.8% |
| o4-mini (Seal-0) | **6.3%** | 5.4% | 4.5% |
| o3 (Seal-0) | 11.7% | **17.1%** | 14.4% |

增加test-time计算不产生可靠增益，性能经常平台化甚至下降。

### 关键发现

- **高级推理模型对噪声极度敏感**：DeepSeek-R1使用FreshPrompt后Seal-Hard准确率从22.4%降至11.0%，在never-changing问题上下降17.7%
- **搜索可能有害**：GPT-4.1-mini使用内置搜索后准确率从13.8%降至11.8%
- **人类显著优于模型**：人类在50题Seal-Hard子集上开放搜索平均38.8%、oracle模式50.4%，最佳人类64.0%/72.0%
- **LongSeal中干扰越多性能越差**：GPT-4.1-mini在k=12时32.7% → k=30时29.5%；即使仅提供gold文档（无干扰），GPT-4.1仅48.0%准确率
- **不存在经典位置偏差**：新模型已缓解"lost-in-the-middle"效应，但识别相关文档仍是核心困难

## 亮点与洞察

- 极具创新性的对抗性基准构建方法，确保每道题对当前最强模型构成实质挑战
- 揭示了test-time scaling的局限——在噪声检索下更多推理可能放大虚假信息
- 证明了内置搜索训练（如ChatGPT）比检索式提示方法（FreshPrompt）更有效
- 动态版本化基准设计，承诺定期更新答案以反映最新知识

## 局限与展望

- 数据集规模小（Seal-0仅111题），统计显著性可能受限
- 答案随时间变化需持续维护，长期可持续性存疑
- 评估仅覆盖英文问题，跨语言推理类别较少（5.5%）
- 仅关注事实性问答，未覆盖更复杂的推理类型（如数学证明、代码生成）

## 相关工作与启发

- **SimpleQA**（Wei et al., 2024）：SealQA在其对抗性收集理念上进一步发展，将难度从"GPT-4失败"提升至"所有前沿模型多次尝试均失败"
- **FreshLLMs**（Vu et al., 2024）：SealQA的时间敏感性分类和FreshPrompt方法直接来源于此
- **BrowseComp**（Wei et al., 2025）：互补的浏览能力评估，SealQA更聚焦推理而非信息获取
- 对RAG系统设计的启发：naive检索整合可能放大噪声，需要更鲁棒的证据筛选和冲突解决机制

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个专门针对噪声/冲突检索结果设计的对抗性搜索增强基准，填补重要空白
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖20+模型，含人类评估、多维度消融（问题类型/时间/搜索方法/test-time scaling）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，但部分表格信息密度较高
- 价值: ⭐⭐⭐⭐⭐ 揭示了当前最强LLM在真实搜索场景下的根本性局限，对RAG系统设计有重要指导意义

<!-- RELATED:START -->

## 相关论文

- [AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)
- [Efficient Test-Time Scaling for Small Vision-Language Models](efficient_test-time_scaling_for_small_vision-language_models.md)
- [Native Reasoning Models: Training Language Models to Reason on Unverifiable Data](native_reasoning_models_training_language_models_to_reason_on_unverifiable_data.md)
- [Conflict-Aware Fusion: Resolving Logic Inertia in Large Language Models via Structured Cognitive Priors](conflict-aware_fusion_resolving_logic_inertia_in_large_language_models_via_struc.md)
- [AIMCoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](aimcot_active_information-driven_multimodal_chain-of-thought_for_vision-language.md)

<!-- RELATED:END -->
