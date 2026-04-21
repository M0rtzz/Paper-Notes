---
title: >-
  [论文解读] Hierarchical Attention Generates Better Proofs
description: >-
  [ACL 2025][形式化定理证明] 提出 Hierarchical Attention 正则化方法，通过建立五层语义层次结构来引导 LLM 的注意力机制，使其与数学推理的自然信息流对齐，在 miniF2F 和 ProofNet 上分别提升证明成功率 2.05% 和 1.69%，同时降低证明复杂度 23.81% 和 16.50%。
tags:
  - ACL 2025
  - 形式化定理证明
  - 层次化注意力
  - Lean
  - 数学推理
  - 正则化
---

# Hierarchical Attention Generates Better Proofs

**会议**: ACL 2025  
**arXiv**: [2504.19188](https://arxiv.org/abs/2504.19188)  
**代码**: [https://github.com/Car-pe/HAGBP](https://github.com/Car-pe/HAGBP)  
**领域**: 其他  
**关键词**: 形式化定理证明, 层次化注意力, Lean, 数学推理, 正则化

## 一句话总结

提出 Hierarchical Attention 正则化方法，通过建立五层语义层次结构来引导 LLM 的注意力机制，使其与数学推理的自然信息流对齐，在 miniF2F 和 ProofNet 上分别提升证明成功率 2.05% 和 1.69%，同时降低证明复杂度 23.81% 和 16.50%。

## 研究背景与动机

形式化定理证明是 AI 与数学交叉领域的重要研究方向。Lean、Coq、Isabelle 等证明助手已成为探索这一方向的关键平台。LLM 在生成证明方面展现了潜力，但仍面临核心挑战：

**序列处理 vs 结构推理的鸿沟**：LLM 本质上处理的是平坦的 token 序列，缺乏对形式语义的显式理解。然而数学证明具有内在的层次化结构——概念之间存在依赖关系和组合关系。

**注意力机制的盲区**：标准 Transformer 的注意力机制允许任意 token 间的自由交互，无法反映数学推理中"底层概念支撑上层定理"的自然偏序关系。

**证明复杂度问题**：LLM 经常生成不必要冗长的证明，或在困难问题上完全失败。

核心论点是：数学推理遵循自然的层次化信息流，应将这种结构先验注入到模型的注意力模式中。

## 方法详解

### 整体框架

方法分两步：首先从输入中提取层次化的信息流模式（Extract Flow Pattern），然后通过专门的损失函数引导模型的注意力遵循这些层次化约束（Hierarchical Attention Loss）。

### 关键设计

1. **五层语义层次结构**：基于 Lean 语言的强类型特性，将 token 分为五个自然层级：

    - **Level 0（Context 层）**：包含背景信息、辅助概念和一般知识
    - **Level 1（Case 层）**：模式匹配和分支分析
    - **Level 2（Type 层）**：类型声明和定义
    - **Level 3（Instance 层）**：实例声明和具体示例
    - **Level 4（Goal 层）**：核心定理或命题（最高层）
   
   这些层级遵循偏序关系：$context \prec case \prec type \prec instance \prec goal$。

2. **三种信息流类型**：基于两个 token $t_i, t_j$ 的层级关系定义信息流：

    - **Unrestricted（无限制）**：$level(t_i) = level(t_j)$，同层 token 自由交互
    - **Guided（引导）**：$level(t_i) < level(t_j)$，低层向高层传递信息（鼓励）
    - **Limited（限制）**：$level(t_i) > level(t_j)$，高层向低层传递信息（惩罚）
   
   核心思想：高层概念（如 Goal）应从低层概念（如 Context、Type）获取信息来构建推理，反向则应受到限制。

3. **层次化注意力掩码**：构造二值掩码 $M_{ij}$，当 $level(t_i) \leq level(t_j)$ 时 $M_{ij} = 1$（允许），否则 $M_{ij} = 0$（限制）。

4. **逐层适应因子**：$\alpha_l = 1 - l/L$，在较浅的 Transformer 层强制更严格的层次约束，在较深的层允许更灵活的注意力模式。这反映了一个直觉：浅层负责编码基本结构，深层需要跨层级的复杂推理。

### 损失函数 / 训练策略

流损失（Flow Loss）惩罚违反层次约束的注意力模式：

$$\mathcal{L}_{flow} = \frac{1}{|T|} \sum_{l=1}^{L} \alpha_l \cdot \sum_{i,j} \text{ReLU}(att_l(t_i, t_j) \cdot (1 - M_{ij}))$$

最终训练目标为标准交叉熵损失与流损失的加权和：

$$\mathcal{L} = \mathcal{L}_{LM} + \lambda \mathcal{L}_{flow}$$

其中 $\lambda$ 控制层次约束的强度。基础模型为 Pythia-2.8B，在 LeanDojo Benchmark 4 上微调 3 个 epoch。评估协议包含 best-first search（BFS）和 single-pass sampling（SPS）两种策略。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 (BFS, K=64) | Baseline (BFS, K=64) | 提升 |
|--------|------|------|----------|------|
| miniF2F test | Pass率 | **31.56%** | 29.51% | +2.05% |
| miniF2F valid | Pass率 | **34.02%** | 31.56% | +2.46% |
| ProofNet test | Pass率 | **15.25%** | 13.56% | +1.69% |
| ProofNet valid | Pass率 | **11.86%** | 10.17% | +1.69% |
| miniF2F test | 证明复杂度比 $R_{avg}$ | **0.76** | 1.00 | 减少23.81% |
| ProofNet test | 证明复杂度比 $R_{avg}$ | **0.84** | 1.00 | 减少16.50% |

Single-pass sampling 改进更显著：miniF2F test 从 23.36% → 27.87%（+4.51%），miniF2F valid 从 21.72% → 26.64%（+4.92%）。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 注意力分布：约束层 vs 非约束层 | Limited flow 从 5.5-27.8% → ≈0% | 约束成功实施 |
| 非约束层中 Limited flow | 仅 0.5-3.2% | 模型已内化层次结构 |
| Goal 层 Guided flow | 约束层 68.7%，非约束层 84.5% | 高层有效整合低层信息 |
| Type/Instance 层 Guided flow | 77.7%/71.0%（约束层） | 中间层也维持层次传播 |

### 关键发现

- BFS 策略一致优于 SPS，本文方法在 BFS 基础上进一步提升。
- 性能增益在计算预算 K ≥ 16 时更加显著，说明方法在充足搜索资源下优势更大。
- 注意力模式分析揭示了重要现象：即使在 $\alpha_l = 0$（无约束）的深层中，模型的注意力模式仍自发保持层次结构，说明模型已经内化了数学推理的层次化模式。
- 证明复杂度的降低幅度（23.81%）比 Pass 率提升（2.05%）更显著，说明方法的核心优势在于生成更简洁的证明。

## 亮点与洞察

- **核心洞察深刻**：将数学推理的层次化本质与 Transformer 注意力机制显式对齐，是一种优雅的结构先验注入方式。
- **轻量化设计**：仅添加一个正则化项，无需修改模型架构，易于与其他方法结合。
- **逐层适应因子的巧妙设计**：$\alpha_l = 1 - l/L$ 在浅层约束结构、深层释放灵活性，符合 Transformer 的分层表示特性。
- **"内化"现象意义重大**：非约束层自发维持层次结构的发现说明，适当的训练信号能让模型真正学会结构化推理，而非仅被动遵循约束。

## 局限与展望

1. 层次定义针对 Lean 语言的语义，迁移到 Coq 或 Isabelle 等其他证明语言时需要重新适配。
2. 固定的五层层次结构可能限制动态推理模式——某些复杂证明可能需要跨层级的灵活信息流。
3. 数据限制使得无法在更先进的模型（如 DeepSeek-Prover、InternLM-Math）上评估。
4. 层次化解析依赖字符串模式匹配，可能对复杂的嵌套结构不够鲁棒。
5. 未来可探索自适应层次结构和跨领域泛化能力。

## 相关工作与启发

- 与之前将数学公式解析为树或图的方法（Wang et al. 2017; Paliwal et al. 2020）相比，本文不依赖精心设计的规则或程序生成的数据，而是通过软约束让模型学习结构。
- 基线方法 LLMSTEP（Welleck & Saha, 2023）提供了完整的模型、数据和超参数，保证了对比分析的可复现性。
- 该工作为如何在 LLM 中注入领域特定结构先验提供了可推广的思路。类似的方法可应用于代码生成、逻辑推理等其他结构化任务。

## 评分

- 新颖性: ⭐⭐⭐⭐ 层次化注意力正则化在定理证明中的应用新颖，core insight 深刻
- 实验充分度: ⭐⭐⭐⭐ 覆盖 miniF2F 和 ProofNet 两个基准，BFS/SPS 两种策略，注意力可视化充分
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、框架图优美、数学表述简洁严谨
- 价值: ⭐⭐⭐⭐ 方法轻量可推广，但绝对性能提升幅度有限，需更大规模验证

<!-- RELATED:START -->

## 相关论文

- [Better Embeddings with Coupled Adam](better_embeddings_with_coupled_adam.md)
- [Towards Better Evaluation for Generated Patent Claims](patclaimeval_patent_evaluation.md)
- [The Hidden Attention of Mamba Models](the_hidden_attention_of_mamba_models.md)
- [Segment-Based Attention Masking for GPTs](segment-based_attention_masking_for_gpts.md)
- [Hierarchical Memory Organization for Wikipedia Generation](hierarchical_memory_wikipedia_gen.md)

<!-- RELATED:END -->
