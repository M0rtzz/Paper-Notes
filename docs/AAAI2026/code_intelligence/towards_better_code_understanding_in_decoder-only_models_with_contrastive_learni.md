---
title: >-
  [论文解读] Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning
description: >-
  [AAAI 2026][对比学习] 本文提出CL4D框架，通过对比学习对预训练的decoder-only代码生成模型进行继续预训练，使其能够有效提取代码表示并在代码搜索和克隆检测等理解任务上达到甚至超越同规模encoder-only模型的性能。
tags:
  - AAAI 2026
  - 对比学习
  - Decoder-Only模型
  - 代码理解
  - 代码搜索
  - 克隆检测
---

# Towards Better Code Understanding in Decoder-Only Models with Contrastive Learning

**会议**: AAAI 2026  
**arXiv**: [2406.12326](https://arxiv.org/abs/2406.12326)  
**代码**: https://github.com/JiayiLin1024/CL4D (有)  
**领域**: Self-Supervised Learning / Code Understanding  
**关键词**: 对比学习, Decoder-Only模型, 代码理解, 代码搜索, 克隆检测

## 一句话总结
本文提出CL4D框架，通过对比学习对预训练的decoder-only代码生成模型进行继续预训练，使其能够有效提取代码表示并在代码搜索和克隆检测等理解任务上达到甚至超越同规模encoder-only模型的性能。

## 研究背景与动机
近年来，大规模decoder-only代码生成模型（如CodeGen、StarCoder、Code Llama等）在代码生成任务上取得了显著成果，模型规模已达到70B参数级别。然而，这些模型在代码理解任务（如代码搜索、克隆检测）上的表现却明显不如专门设计的encoder-only模型（如CodeBERT、UniXcoder、CodeSage），后者最大仅1.5B参数。

核心矛盾在于：decoder-only模型拥有更多参数、看过更多代码数据，但由于自回归训练目标和单向注意力机制的限制，其表示能力在理解任务上受到制约。而从头训练一个同等规模的encoder-only模型，计算成本又极其昂贵。

本文的切入角度是：**能否在不重新训练的情况下，复用已有的decoder-only代码生成模型的知识，通过高效的继续预训练来提升其代码理解能力？** 核心idea是利用对比学习（CL4D）来弥补decoder-only架构在表示学习方面的不足。

## 方法详解

### 整体框架
CL4D是一个针对decoder-only模型的对比学习继续预训练框架。整体pipeline包括：(1) 从The Stack数据集构建跨语言的(query, code)训练对；(2) 采用双编码器架构（共享权重的两个decoder模块）编码输入；(3) 通过对比学习（含in-batch负样本和hard negatives）训练模型学习判别性表示。

### 关键设计
1. **数据构建**:

    - 功能：从The Stack数据集中抽取6种编程语言（Python、Java、Go、PHP、JavaScript、Ruby）的代码
    - 核心思路：使用Tree-Sitter提取函数文档字符串的第一行作为query，对应函数作为code，构建双模态(query, code)对
    - 设计动机：增强模型区分自然语言和编程语言的能力，利用预训练模型已有的代码知识，仅需数百万样本进行继续预训练

2. **代码表示提取方法**:

    - 功能：探索从decoder-only模型中提取代码表示的最优方式
    - 核心思路：研究了两种方法——(1) 使用最后一个token的embedding；(2) 使用所有token embedding的平均值。同时研究了left padding和right padding的影响
    - 设计动机：与encoder-only模型使用[CLS] token不同，decoder-only模型的单向注意力机制决定了只有最后一个token能聚合全部信息。实验发现**right padding + 平均所有token embedding**是最优策略

3. **对比学习训练**:

    - 功能：通过对比学习增强decoder-only模型的表示能力
    - 核心思路：随机采样n个(query, code)对组成batch，以配对code为正样本，batch内其他code为负样本。额外引入hard negatives（在表示空间中与query相近但语义不同的代码片段，通过UniXcoder预先筛选）
    - 设计动机：直接使用next-token prediction训练目标限制了表示能力，对比学习可以拉近语义相似样本、推远语义不同样本，同时允许batch内混合多种编程语言，学习统一的语义空间

### 损失函数 / 训练策略
损失函数为InfoNCE变体，包含in-batch负样本和hard negatives：

$$\mathcal{L} = -\log\frac{\exp(s(q,c^+)/\tau)}{\sum_{i=1}^n \exp(s(q,c_i)/\tau) + \exp(s(q,c^h)/\tau)}$$

其中温度系数 $\tau = 0.05$, $s(q,c)$ 为余弦相似度。训练使用AdamW优化器，学习率2e-5，batch size 64，在8×A100 GPU上训练2个epoch。最长训练时间（phi-1）约3天。

## 实验关键数据

### 主实验

| 模型 | CSN (MRR) | CoSQA (MRR) | POJ-104 (MAP) |
|------|-----------|-------------|---------------|
| UniXcoder (125M, encoder) | 74.40 | 70.1 | 89.56 |
| CodeSage (1.3B, encoder) | 75.80 | 68.0 | 87.70 |
| CodeGPT (125M) + CL4D | 70.20 | 69.0 | 87.96 |
| CodeGen (350M) + CL4D | 73.30 | 71.5 | 89.68 |
| phi-1 (1.3B) + CL4D | 75.18 | 72.8 | 92.72 |
| DeepSeek-Coder (1.3B) + CL4D | **77.57** | **71.9** | 89.71 |

CL4D使decoder-only模型在大多数任务上超越同等规模的encoder-only模型约2%。

### 消融实验

| 配置 | CSN (MRR) | CoSQA (MRR) | POJ-104 (MAP) |
|------|-----------|-------------|---------------|
| CL4D (完整) | 72.00 | 51.20 | 45.84 |
| - Hard Negative | 70.80 | 50.40 | 44.65 |
| - In-Batch Negative | 1.42 | 0.45 | 13.20 |

去除in-batch negatives后性能骤降，说明对比学习是提升的关键。Hard negatives额外贡献约1.5%提升。

### 关键发现
- **Zero-shot性能大幅提升**：CL4D使decoder-only模型零样本性能提升40%-76%（最大提升75.90%），甚至在CSN上匹配encoder-only模型的fine-tuned结果
- **模型规模效应**：更大的decoder-only模型在CL4D后获得更好的理解性能，表明scale law在代码理解上同样适用
- **表示空间可视化**：t-SNE可视化表明，CL4D显著改善了语义相似代码在表示空间中的聚集程度，原始decoder-only模型的表示极度分散

## 亮点与洞察
- 首次系统性地探索了如何将decoder-only代码生成模型适配到代码理解任务，填补了生成和理解之间的鸿沟
- 发现right padding + average pooling是decoder-only模型提取代码表示的最优策略，这一发现具有广泛适用性
- 证明了decoder-only架构有潜力统一代码理解和生成任务，无需维护两套独立模型
- 低成本方案：仅需在小规模代码语料上继续预训练数天，即可大幅提升理解能力

## 局限与展望
- 仅在代码搜索和克隆检测两个任务上验证，缺少代码摘要、缺陷检测、代码翻译等更多理解任务的评估
- Hard negative的构建依赖UniXcoder预先计算，引入了额外的预处理开销，且hard negative的质量受限于UniXcoder本身的表示能力
- 未探索更高效的参数微调方法（如LoRA、Adapter），全参数继续预训练在8×A100上仍需数天
- 未在更大规模的decoder-only模型（如7B、13B、70B）上验证，无法确认scaling trend是否持续
- 训练数据仅覆盖6种编程语言，对低资源语言的泛化性未评估
- 未研究continued pre-training后模型的生成性能是否受到影响（理解提升是否以牺牲生成为代价）

## 相关工作与启发
- 与CodeSage类似使用对比学习提升代码表示，但本文聚焦在decoder-only架构的适配上
- 方法论可推广到其他领域：任何已有强大生成模型但理解能力不足的场景（如文本、蛋白质序列）都可尝试类似的对比学习适配策略
- 为构建统一的代码大模型（同时支持生成和理解）提供了一条可行路径
- 与CoLSBERT的scaling law研究互补：CoLSBERT证明encoder-only扩大有效，CL4D证明可以直接复用已有的大规模decoder-only模型

## 评分
- 新颖性: ⭐⭐⭐⭐ (idea清晰但方法相对直接，对比学习本身不新)
- 实验充分度: ⭐⭐⭐⭐ (多模型、多任务、多消融，但任务种类有限)
- 写作质量: ⭐⭐⭐⭐ (结构清晰、论述充分、图表丰富)
- 价值: ⭐⭐⭐⭐ (为decoder-only模型统一生成与理解铺路，实用性强)

<!-- RELATED:START -->

## 相关论文

- [ReCode: Updating Code API Knowledge with Reinforcement Learning](recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion](equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat.md)
- [SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)
- [DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation](diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac.md)
- [Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction](extracting_events_like_code_a_multi-agent_programming_framework_for_zero-shot_ev.md)

<!-- RELATED:END -->
