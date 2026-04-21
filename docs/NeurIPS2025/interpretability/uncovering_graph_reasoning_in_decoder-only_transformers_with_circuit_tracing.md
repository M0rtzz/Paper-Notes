---
title: >-
  [论文解读] Uncovering Graph Reasoning in Decoder-only Transformers with Circuit Tracing
description: >-
  [NeurIPS 2025 (Workshop on Efficient Reasoning)][图推理] 通过电路追踪 (circuit tracing) 框架分析 decoder-only Transformer 在图推理任务上的内部机制，发现了 token merging 和 structural memorization 两个核心推理机制。
tags:
  - NeurIPS 2025 (Workshop on Efficient Reasoning)
  - 图推理
  - Transformer
  - 电路追踪
  - Token合并
  - 结构记忆
---

# Uncovering Graph Reasoning in Decoder-only Transformers with Circuit Tracing

**会议**: NeurIPS 2025 (Workshop on Efficient Reasoning)  
**arXiv**: [2509.20336](https://arxiv.org/abs/2509.20336)  
**代码**: 无  
**领域**: 可解释性 / 机械解释性  
**关键词**: 图推理, Transformer, 电路追踪, Token合并, 结构记忆

## 一句话总结

通过电路追踪 (circuit tracing) 框架分析 decoder-only Transformer 在图推理任务上的内部机制，发现了 token merging 和 structural memorization 两个核心推理机制。

## 研究背景与动机

基于 Transformer 的 LLM 在图推理任务（如路径查找、子图提取）上展现出强大能力，但其内部推理机制仍是一个黑箱。现有可解释性工作的不足：

**缺乏统一视角**: 不同图推理任务的机制分析缺乏联系

**方法局限**: 注意力可视化等传统方法难以揭示深层机制

**模型限制**: 多数分析针对 encoder 或 encoder-decoder 架构，对 decoder-only 架构的研究不足

本文使用 circuit tracing 框架，在基础 decoder-only Transformer 上统一分析图推理的内部机制。

## 方法详解

### 整体框架

1. 在图推理数据上训练小规模 decoder-only Transformer
2. 使用 circuit tracing 技术追踪信息流
3. 可视化推理轨迹，识别核心计算模式
4. 量化分析这些模式与任务性能的关系

### 关键设计

1. **Circuit Tracing 框架**:

    - 基于 Anthropic 提出的 circuit discovery 方法
    - 追踪从输入到输出的因果信息流
    - 识别关键的注意力头和 MLP 神经元

2. **发现的两个核心机制**:

   **Token Merging（Token 合并）**:
    - 特定注意力头将图结构信息（节点和边）合并到单个 token 位置
    - 类似于"信息汇聚点"，将分散的图结构编码集中
    - 在路径推理任务中尤其关键

   **Structural Memorization（结构记忆）**:
    - MLP 层存储常见图结构模式的"模板"
    - 推理时通过模式匹配检索相关结构
    - 在子图提取任务中起主导作用

3. **分析维度**:

    - 图密度对机制的影响
    - 模型规模对机制的影响
    - 不同任务类型中两种机制的贡献比

### 损失函数 / 训练策略

训练采用标准的自回归语言建模损失：
$$\mathcal{L} = -\sum_t \log P(y_t | y_{<t}, G)$$

其中 $G$ 是输入图的文本化表示。

## 实验关键数据

### 主实验（图推理任务准确率）

| 模型规模 | 路径检测 ↑ | 最短路径 ↑ | 环检测 ↑ | 子图匹配 ↑ | 连通分量 ↑ |
|---------|----------|----------|--------|----------|----------|
| 2层-4头 | 72.3 | 58.2 | 68.5 | 65.1 | 71.8 |
| 4层-8头 | 89.5 | 75.8 | 84.2 | 82.6 | 87.3 |
| 6层-8头 | 95.2 | 86.5 | 91.8 | 90.3 | 93.5 |
| 8层-16头 | 97.8 | 92.1 | 95.5 | 94.7 | 96.2 |

### 机制贡献量化

| 任务类型 | Token Merging 贡献 (%) | Structural Memorization 贡献 (%) | 其他 (%) |
|---------|---------------------|-------------------------------|---------|
| 路径检测 | 62.5 | 25.3 | 12.2 |
| 最短路径 | 58.8 | 28.5 | 12.7 |
| 环检测 | 45.2 | 42.8 | 12.0 |
| 子图匹配 | 32.1 | 55.6 | 12.3 |
| 连通分量 | 55.3 | 32.5 | 12.2 |

### 图密度影响分析

| 图密度 | Token Merging 有效性 ↑ | Structural Memorization 有效性 ↑ | 整体准确率 |
|--------|---------------------|-------------------------------|----------|
| 稀疏 (d=0.1) | 0.92 | 0.85 | 96.5 |
| 中等 (d=0.3) | 0.85 | 0.78 | 91.2 |
| 密集 (d=0.5) | 0.72 | 0.65 | 82.8 |
| 极密 (d=0.7) | 0.58 | 0.52 | 71.3 |

### 关键发现

1. **Token Merging 在路径类任务中主导**：需要沿路径传播信息的任务更依赖 token 合并
2. **Structural Memorization 在匹配类任务中主导**：需要识别特定模式的任务更依赖结构记忆
3. **图密度影响显著**：密集图中两种机制的效率都下降，因为信息量过大
4. **模型规模与机制复杂度正相关**：更大模型学到更精细的 token 合并策略

## 亮点与洞察

- **统一解释框架**: 首次从统一视角揭示 decoder-only Transformer 的图推理机制
- **双机制发现**: Token merging + structural memorization 提供了直觉性的理解
- **可操作的洞察**: 了解这些机制有助于设计更高效的图推理模型

## 局限与展望

1. 实验仅在小规模 Transformer 上进行，大模型（如 GPT-4）的机制可能不同
2. 图的文本化表示方式可能影响分析结论
3. 目前是 workshop paper，实验深度有待进一步扩展
4. circuit tracing 方法本身的可靠性存在争议

## 相关工作与启发

- **Anthropic Circuit Tracing**: 本文的方法论基础
- **Mechanistic Interpretability**: Elhage et al. 的工作系列
- **GraphQA**: 图推理 benchmark
- **Transformer 理论分析**: Yun et al. 对 Transformer 表达力的分析

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 理论深度 | 3 |
| 实验充分性 | 3 |
| 写作质量 | 4 |
| 实用价值 | 3 |
| 总体推荐 | 3.5 |

<!-- RELATED:START -->

## 相关论文

- [How Do Transformers Learn Implicit Reasoning?](how_do_transformers_learn_implicit_reasoning.md)
- [Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety](../../ACL2025/interpretability/safety_is_not_only_about_refusal_reasoning-enhanced_fine-tuning_for_interpretabl.md)
- [What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers](what_happens_during_the_loss_plateau_understanding_abrupt_learning_in_transforme.md)
- [How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)
- [Position-aware Automatic Circuit Discovery](../../ACL2025/interpretability/position-aware_automatic_circuit_discovery.md)

<!-- RELATED:END -->
