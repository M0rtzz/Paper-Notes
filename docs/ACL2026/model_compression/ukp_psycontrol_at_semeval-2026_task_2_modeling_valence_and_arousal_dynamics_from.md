---
title: >-
  [论文解读] UKP_Psycontrol at SemEval-2026 Task 2: Modeling Valence and Arousal Dynamics from Text
description: >-
  [ACL 2026][模型压缩][情感评估] UKP_Psycontrol 在 SemEval-2026 Task 2 上取得双项第一，通过结合 LLM 提示、Ising 交互的 MaxEnt 模型和神经回归模型，发现 LLM 擅长捕捉静态情感信号而短期情感变化更多由近期数值轨迹而非文本语义解释。
tags:
  - "ACL 2026"
  - "模型压缩"
  - "情感评估"
  - "纵向分析"
  - "效价-唤醒度"
  - "LLM提示"
  - "MaxEnt模型"
---

# UKP_Psycontrol at SemEval-2026 Task 2: Modeling Valence and Arousal Dynamics from Text

**会议**: ACL 2026  
**arXiv**: [2604.21534](https://arxiv.org/abs/2604.21534)  
**代码**: [GitHub](https://github.com/)  
**领域**: 情感计算 / 纵向情感建模  
**关键词**: 情感评估, 纵向分析, 效价-唤醒度, LLM提示, MaxEnt模型

## 一句话总结

UKP_Psycontrol 在 SemEval-2026 Task 2 上取得双项第一，通过结合 LLM 提示、Ising 交互的 MaxEnt 模型和神经回归模型，发现 LLM 擅长捕捉静态情感信号而短期情感变化更多由近期数值轨迹而非文本语义解释。

## 研究背景与动机

**领域现状**：计算情感分析中，情感通常用效价（valence，正负向）和唤醒度（arousal，激活程度）两个连续维度表示。大多数 NLP 研究使用社交媒体或评论数据，由外部评估者标注或通过情感代理近似，仅能间接获取内部情感状态。

**现有痛点**：SemEval-2026 Task 2 提出了新挑战——需要对时间序列排列的自我报告文本进行纵向情感评估和预测。数据来自美国服务业工人数年间的日志，包含自由散文和情感词汇列表，配合自评的效价和唤醒度。这要求模型不仅理解单个文本的情感，还要捕捉情感随时间的动态变化。

**核心矛盾**：文本语义与数值情感轨迹对情感预测的贡献可能不同——LLM 擅长理解文本含义，但短期情感波动可能更多反映个人的情感惯性而非新的文本信息。

**本文目标**：(1) 评估静态情感识别（Subtask 1）；(2) 预测未来情感变化（Subtask 2A）；(3) 理解文本语义 vs 数值轨迹在不同任务中的相对重要性。

**切入角度**：三种互补方法的组合——LLM 提示（利用语言理解）、MaxEnt+Ising（利用概率图模型的结构化依赖建模）、神经回归（利用数值轨迹和用户嵌入）。

**核心 idea**：静态情感评估靠 LLM 文本理解，动态情感预测靠数值轨迹的短期惯性，两者各有所长。

## 方法详解

### 整体框架

系统包含三个模块：(1) LLM 提示模块——在用户感知和用户无关设置下用 GPT-5 预测效价和唤醒度；(2) MaxEnt+Ising 模块——用最大熵模型和 Ising 交互建模情感状态的结构化依赖；(3) 神经回归模块——用滑动窗口的近期情感轨迹 + RoBERTa 文本嵌入 + 可训练用户嵌入预测下一步情感变化。

### 关键设计

1. **LLM 提示策略**:

    - 功能：利用 LLM 的语言理解能力预测文本中的情感
    - 核心思路：区分用户感知（使用同一用户的历史示例作为 few-shot）和用户无关（使用标签平衡的随机示例）两种设置。比较两种输出格式：文本情感标签（映射到数值）vs 直接数值预测，发现文本标签更稳定。引入滑动窗口动态更新策略（最近 N=15 条），但发现预测误差会累积传播，固定示例反而更好。散文和情感词汇分开处理
    - 设计动机：利用 LLM 与人类效价/唤醒度评分的强相关性，用户感知设置能捕捉个体表达模式

2. **MaxEnt+Ising 结构化模型**:

    - 功能：用概率图模型捕捉情感状态间的结构化依赖
    - 核心思路：定义能量函数 $E(\mathbf{x}) = -\mathbf{x}^\top \mathbf{h} - \frac{1}{2}\mathbf{x}^\top \mathbf{J}\mathbf{x}$，其中 $\mathbf{h}$ 建模线性效应，$\mathbf{J}$ 捕捉成对交互。情感变量用 one-hot 编码，语义信息通过自编码器压缩为二值向量。由于状态空间有界，可精确计算配分函数 $Z$，实现最大似然训练。推理时通过条件期望解码，产生与相关性评估指标对齐的连续预测
    - 设计动机：心理学理论认为心理状态在潜在能量景观上演化并遵循玻尔兹曼分布

3. **神经回归模型（Subtask 2A）**:

    - 功能：用近期情感轨迹预测下一步情感变化
    - 核心思路：输入包括滑动窗口（1-4 步）内的近期文本嵌入（RoBERTa mean-pooling）、当前效价/唤醒度、前一步状态变化、可训练用户嵌入。比较三种设置：(a) 无文本基线（仅数值特征+用户嵌入）、(b) 文本增强、(c) 语义聚类表示
    - 设计动机：假设短期情感变化主要由个人情感惯性驱动，用户嵌入捕捉个体差异

### 损失函数 / 训练策略

MaxEnt 模型用最大似然训练，神经回归模型用标准回归损失。LLM 提示无需训练。最终提交组合了各模块的最优配置。

## 实验关键数据

### 主实验

**Subtask 1（纵向情感评估）— 测试集**

| 方法 | Valence r_composite | Arousal r_composite |
|------|-------------------|-------------------|
| Baseline linear(BERT) | 0.557 | 0.299 |
| MaxEnt Ising | 0.589 | 0.327 |
| **LLM-based (提交)** | **0.667** | **0.554** |

**Subtask 2A（情感变化预测）— 测试集**

| 方法 | Valence r | Arousal r |
|------|-----------|-----------|
| Baseline linear(prev) | 0.520 | 0.609 |
| MaxEnt Ising | — | — |
| **Neural Regression (提交)** | **0.675** | **0.683** |

### 关键发现

- 用户感知提示仅比用户无关提示略好，表明标签平衡的随机示例已能近似大部分用户特异性信息
- 增加 shot 数改善效价相关性（10→20 shot: 0.617→0.661），但对唤醒度无类似效果
- 文本情感标签预测优于直接数值预测，说明自然语言描述更符合 LLM 预训练分布
- **关键发现**：在 Subtask 2A 中，无文本基线（仅数值轨迹+用户嵌入）的表现与文本增强模型相当，说明短期情感变化更多由近期数值状态而非文本语义解释
- 动态更新策略（滑动窗口替换）不如固定 shot，因为预测误差累积传播

## 亮点与洞察

- "短期情感变化由数值轨迹而非文本语义驱动"这一发现对情感计算领域有重要启示——可能意味着情感有自身的时间序列惯性，文本只是快照而非驱动力
- MaxEnt+Ising 模型将心理学理论（能量景观假说）引入 NLP 任务，提供了可解释的概率框架
- 三种方法的互补组合策略值得借鉴：LLM 处理语义理解，概率模型处理结构化依赖，神经网络处理数值序列

## 局限与展望

- 数据量较小（2764 条目，137 用户），限制了深度学习方法的潜力
- 数据质量问题突出：92% 用户仅参与一个时间段，部分用户效价/唤醒度始终不变
- MaxEnt 模型的二值化语义表示可能丢失情感细微差别
- 仅在英文服务业工人群体上验证，文化差异可能影响情感表达模式

## 相关工作与启发

- **vs 纯 LLM 方法**: LLM 在静态评估上强大但在动态预测上不够，需要数值轨迹建模的补充
- **vs 传统 BERT 基线**: 结合多种方法（LLM+MaxEnt+神经回归）的系统在两个子任务上均获第一

## 评分

- 新颖性: ⭐⭐⭐⭐ 三种方法的组合策略和 MaxEnt+Ising 的应用有新意，但各组件单独不算新颖
- 实验充分度: ⭐⭐⭐⭐ 详细的消融和对比，但受限于共享任务的数据规模
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细
- 价值: ⭐⭐⭐⭐ "文本 vs 数值轨迹"的发现对情感计算研究有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination](tell-tale_task_efficient_llms_with_task_aware_layer_elimination.md)
- [\[ACL 2026\] Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics](why_steering_works_toward_a_unified_view_of_language_model_parameter_dynamics.md)
- [\[ACL 2026\] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs](task-stratified_knowledge_scaling_laws_for_post-training_quantized_large_languag.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[ICML 2026\] Proxy Compression for Language Modeling](../../ICML2026/model_compression/proxy_compression_for_language_modeling.md)

</div>

<!-- RELATED:END -->
