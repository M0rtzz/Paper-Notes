---
title: >-
  [论文解读] LTD-Bench: Evaluating Large Language Models by Letting Them Draw
description: >-
  [NeurIPS 2025][LLM评测][LLM评估] LTD-Bench 通过让 LLM 画画（生成点阵或代码绘图）来评估其空间推理能力，将抽象的评分指标转化为直观可视的输出，揭示了当前先进 LLM 在建立语言与空间概念双向映射方面的严重不足。 当前 LLM 评估范式存在一个关键盲区：依赖于不透明的数值指标…
tags:
  - "NeurIPS 2025"
  - "LLM评测"
  - "LLM评估"
  - "空间推理"
  - "视觉生成"
  - "benchmark"
  - "绘图能力"
---

# LTD-Bench: Evaluating Large Language Models by Letting Them Draw

**会议**: NeurIPS 2025  
**arXiv**: [2511.02347](https://arxiv.org/abs/2511.02347)  
**代码**: [walktaster/LTD-Bench](https://github.com/walktaster/LTD-Bench)  
**领域**: LLM评测  
**关键词**: LLM评估, 空间推理, 视觉生成, benchmark, 绘图能力

## 一句话总结

LTD-Bench 通过让 LLM 画画（生成点阵或代码绘图）来评估其空间推理能力，将抽象的评分指标转化为直观可视的输出，揭示了当前先进 LLM 在建立语言与空间概念双向映射方面的严重不足。

## 研究背景与动机

当前 LLM 评估范式存在一个关键盲区：依赖于不透明的数值指标，掩盖了模型在空间推理方面的根本局限性。当模型在某基准上获得 85% 的分数时，这个数字到底揭示了什么具体的能力和局限？传统评估主要集中在语言理解（MMLU）、数学推理（GSM8K）和代码生成（HumanEval）等符号操作能力上，但：

**空间推理评估的缺失**：没有系统性的基准评估 LLM 的空间感知和想象能力

**评估结果不直观**：数值分数无法直观呈现模型能做什么、不能做什么

**物理世界理解的重要性**：LLM 越来越多地被部署于机器人、自动驾驶和设计工具等需要空间推理的领域

**认知科学的支持**：先天性盲人的研究表明空间认知可以通过非视觉模态（语言描述等）建立，这意味着纯文本 LLM 也应当具备空间理解能力

## 方法详解

### 整体框架

LTD-Bench 包含 **183 个任务**，设计了三个核心原则：

1. **视觉可解释性**：所有生成任务输出渲染为图像，直观展示模型能力
2. **双路径评估**：同时评估生成（空间想象）和识别（空间感知）两个方向
3. **渐进复杂度**：三个难度级别系统定位模型能力阈值

### 关键设计

**Easy 级别：离散网格空间理解**
- 生成任务：给定描述（如"画字母 H，3×3 的 0-1 矩阵"），模型输出点阵矩阵
- 识别任务：给定点阵矩阵，识别其代表的字符
- 共 86 题（生成 50 + 识别 36）

**Normal 级别：连续坐标空间的曲线组合**
- 生成任务：生成 Python 代码用曲线绘制指定字符（禁止使用文本渲染函数）
- 识别任务：给定绘制字符的 Python 代码，识别其绘制的字符
- 共 72 题（生成 36 + 识别 36）

**Hard 级别：真实世界物体绘制**
- 生成任务：开放式指令绘制复杂真实物体（如"画一只尖耳朵、长胡须、圆眼睛的猫"）
- 仅生成任务，采用 GPT-4.1 自动评估，评分 0.0~1.0
- 共 25 题

| 级别 | 生成 | 识别 | 总计 |
|------|------|------|------|
| Easy | 50 | 36 | 86 |
| Normal | 36 | 36 | 72 |
| Hard | 25 | - | 25 |
| **总计** | **111** | **72** | **183** |

### 损失函数 / 训练策略

本工作为评估基准，不涉及训练。评估策略：
- Easy 和 Normal 级别的生成任务：人工评估 + GPT-4.1 自动评估双轨制
- Easy 和 Normal 级别的识别任务：直接与标准答案比对计算准确率
- Hard 级别生成任务：仅使用 GPT-4.1 评估（因开放式输出主观性强）
- 代码执行失败的生成任务直接判 0 分

## 实验关键数据

### 主实验

7 个先进 LLM 在 LTD-Bench 上的综合表现：

| 模型 | Easy 生成 | Easy 识别 | Normal 生成 | Normal 识别 | Hard 生成 | 平均 |
|------|-----------|-----------|-------------|-------------|-----------|------|
| Deepseek-r1 | 82.0 | 69.4 | 65.3 | 77.8 | 63.2 | **71.5** |
| GPT-4.1-mini | 85.0 | 38.9 | 70.8 | 55.6 | 71.6 | 64.4 |
| Deepseek-v3 | 72.0 | 36.1 | 54.2 | 63.9 | 66.4 | 58.5 |
| GPT-4o | 81.0 | 41.7 | 45.8 | 44.4 | 48.0 | 52.2 |
| QwQ-32B | 65.0 | 36.1 | 38.9 | 58.3 | 42.0 | 48.1 |
| Qwen2.5-72B | 56.0 | 13.9 | 18.1 | 25.0 | 40.8 | 30.8 |
| Llama3.3-70B | 46.0 | 11.1 | 23.6 | 19.4 | 35.2 | 27.1 |

按生成 vs 识别能力分拆：

| 模型 | 生成 | 识别 | 平均 |
|------|------|------|------|
| Deepseek-r1 | 72.9 | **73.6** | **73.2** |
| GPT-4.1-mini | **77.5** | 47.2 | 62.3 |
| Deepseek-v3 | 64.7 | 50.0 | 57.4 |
| GPT-4o | 62.1 | 43.1 | 52.6 |

### 消融实验

深度推理蒸馏对 Llama3.3-70B 的影响：

| 模型 | 生成 | 识别 | 平均 |
|------|------|------|------|
| Llama3.3-70B-Instruct | 36.6 | 15.3 | 26.0 |
| Deepseek-r1-distill-Llama3.3-70B | 33.7 | 33.3 | 33.5 |
| 变化Δ | ↓2.9 | **↑18.1** | ↑7.6 |

深度推理蒸馏显著提升识别能力（+18%），但生成能力反而略降（-2.9%），说明推理增强对感知有帮助但对想象无效甚至有harm。

### 关键发现

1. **LLM 空间推理整体较弱**：仅 Deepseek-r1 平均超过 70%，多数模型低于 60%；人类专家在 Easy/Normal 任务上接近满分
2. **深度推理提升识别但非生成**：推理能力增强空间感知，但空间想象需要不同的能力基础
3. **多模态 LLM 无明显优势**：GPT-4.1-mini 和 GPT-4o 等多模态模型在纯文本空间任务上并未一致优于纯文本模型
4. **模型相似性分析**：Qwen2.5 系列两个模型生成的图像风格高度相似（22 个样本中 12 个更相似），提供了模型相似性评估的新视角

## 亮点与洞察

1. **评估范式创新**：将 LLM 评估从抽象数字转化为直观可视输出，非专业人士也能直观感受模型能力边界，这是非常有价值的评估理念
2. **双路径+渐进设计**：同时测试语言→空间（生成）和空间→语言（识别）两个方向的映射，配合三级难度，提供了全面的空间认知能力画像
3. **深度推理与空间想象的分离**：发现推理能力与空间想象能力是正交的，这对理解 LLM 的认知结构很有启发
4. **模型风格相似性**：Hard 级别生成任务的视觉输出可用于分析模型间的风格相似性，是传统评估难以捕捉的维度

## 局限与展望

1. **数据集规模较小**：仅 183 个任务，可能不足以得出统计上稳健的结论
2. **评估维度单一**：仅关注空间感知和想象，未涵盖其他空间推理能力（如空间变换、3D 推理等）
3. **GPT-4.1 评估的可靠性**：Hard 级别完全依赖 GPT-4.1 评估，引入了评估模型自身的偏差
4. **模型相似性分析初步**：仅基于风格视觉比较，缺乏系统化的定量方法
5. **未深入探究失败原因**：没有分析模型在哪些具体的空间推理子能力上失败（如旋转、比例、拓扑关系等）

## 相关工作与启发

- **现有 LLM 基准**：MMLU、GSM8K、HumanEval 等侧重符号操作，缺乏空间评估
- **空间认知神经科学**：Striem-Amit et al. (2018) 等研究表明空间认知不依赖视觉经验，为文本 LLM 应具备空间能力提供了理论基础
- **ARC Prize**：Chollet et al. (2024) 的抽象推理挑战与本文关注点相关但方法不同
- **启发**：类似的"让模型做可视化任务"思路可扩展到物理模拟、3D 场景构建等更复杂的空间推理评估；"从输出可视化反推能力"的评估范式可迁移到其他需要结构化输出的领域

## 评分

- 新颖性: ⭐⭐⭐⭐ 让 LLM 画画的评估思路非常新颖，将抽象分数转化为直观视觉输出是很好的创意
- 实验充分度: ⭐⭐⭐ 测试了 7 个模型，但数据集仅 183 题，统计显著性有限
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，动机阐述到位，视觉示例很有说服力
- 价值: ⭐⭐⭐⭐ 揭示了 LLM 空间推理的重要能力差距，为未来评估方向提供了参考

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](../../ACL2025/llm_evaluation/batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](../../ACL2025/llm_evaluation/wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)
- [\[NeurIPS 2025\] Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)
- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](../../ACL2025/llm_evaluation/seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)

</div>

<!-- RELATED:END -->
