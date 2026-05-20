---
title: >-
  [论文解读] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures
description: >-
  [ACL 2026][可解释性][内在可解释性] 系统综述了 LLM 内在可解释性的最新进展，将现有方法分为五大设计范式（功能透明性、概念对齐、表征可分解性、显式模块化、潜在稀疏归纳），并讨论了开放挑战和未来方向。
tags:
  - "ACL 2026"
  - "可解释性"
  - "内在可解释性"
  - "大语言模型"
  - "设计范式分类"
  - "模块化架构"
  - "稀疏归纳"
---

# Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures

**会议**: ACL 2026  
**arXiv**: [2604.16042](https://arxiv.org/abs/2604.16042)  
**代码**: 无  
**领域**: 可解释性  
**关键词**: 内在可解释性, 大语言模型, 设计范式分类, 模块化架构, 稀疏归纳

## 一句话总结

系统综述了 LLM 内在可解释性的最新进展，将现有方法分为五大设计范式（功能透明性、概念对齐、表征可分解性、显式模块化、潜在稀疏归纳），并讨论了开放挑战和未来方向。

## 研究背景与动机

**领域现状**：大语言模型在各类 NLP 任务上取得了显著成功，但其内部机制的不透明性（黑盒特性）阻碍了可信部署，尤其在医疗、法律等高风险领域。现有可解释 AI 综述主要聚焦于事后解释方法（post-hoc），如 LIME、SHAP、稀疏自编码器、因果干预等。

**现有痛点**：事后解释方法通过外部近似来解释已训练好的模型，存在"保真度鸿沟"——解释和模型的真实计算之间存在根本性偏差。即使是因果干预方法（如 ROME），虽然局部保真度更强，但其解释粒度过细，难以聚合为对模型整体行为的连贯理解。

**核心矛盾**：历史上，内在可解释的模型（如线性模型、决策树）在表达能力上远不及黑盒大模型，导致"可解释性 vs 性能"被视为不可调和的 trade-off。但近期研究表明，通过将模块化、稀疏性、解纠缠等归纳偏置嵌入现代架构，这一 trade-off 正在被打破。

**本文目标**：为内在可解释性方法提供统一的分类框架，系统梳理设计原则，明确各方法的优劣和适用场景，并指出未来研究方向。

**切入角度**：不同于事后解释综述从"工具"出发，本文从"设计原则"出发，关注如何从架构和训练过程中构建透明性。

**核心 idea**：将内在可解释性方法组织为五大设计范式，每个范式代表一种不同的"透明性来源"。

## 方法详解

### 整体框架

本文提出的分类体系包含五大设计范式，从不同层面为 LLM 引入可解释性：

### 关键设计

1. **功能透明性（Functional Transparency）**:

    - 功能：确保模型的每个计算步骤本身可解释
    - 核心思路：包括广义加性模型（GAMs）及其扩展（GA2M、EBMs、GAMI-Net），通过可加性约束使每个特征的贡献可视化；自解释神经网络（SENN）分解为基础概念和相关性分数；B-cos 网络通过权重-输入对齐变换产生线性解释；Kolmogorov-Arnold Networks (KANs) 使用可学习样条函数替代固定激活函数
    - 设计动机：最直接的可解释性来源——如果计算本身是透明的，就不需要外部解释工具。局限在于可加性约束限制了建模能力，KANs 在大规模 LLM 上的适用性尚未验证

2. **概念对齐（Concept Alignment）**:

    - 功能：将模型内部表征与人类可理解的概念对齐
    - 核心思路：概念瓶颈模型（CBMs）在中间层强制预测人类定义的概念，然后基于概念做最终预测；CB-LLM 将此扩展到 LLM，通过混合瓶颈 + 对抗训练保留性能；Label-free CBM 借助 CLIP 自动发现概念；Codebook Features 通过向量量化实现离散化的概念编码
    - 设计动机：概念是人类思维的基本单位，将模型表征与概念对齐可以产生最自然的解释。但概念定义需要领域专家，且残差通道（hybrid CBMs）可能泄露信息绕过瓶颈

3. **表征可分解性（Representational Decomposability）**:

    - 功能：使模型表征可分解为独立的、可解释的组件
    - 核心思路：Backpack 语言模型为每个词学习多个"含义向量"（sense vectors），通过上下文权重加权组合；CoCoMix 在训练中预测连续概念并混合到表征中，保持了概念级别的可追溯性
    - 设计动机：不改变整体架构，只在表征层面引入分解结构。Backpack 的优势在于可以追踪每个词的哪个含义被激活，但推理开销较大

### 损失函数 / 训练策略

本文为综述，不涉及具体训练。但总结了各范式的训练成本特征：功能透明性和概念对齐方法训练成本低-中，显式模块化（MoE）方法成本中-高，潜在稀疏归纳（如 $L_0$ 正则化）成本极高。

## 实验关键数据

### 主实验

综合对比表（节选自 Table 1）：

| 方法类别 | 代表方法 | 可解释性来源 | 训练成本 | 推理成本 | 性能影响 |
|---------|---------|------------|---------|---------|---------|
| 功能透明性 | KANs, B-cos LMs | 形状函数/线性解释 | 中-高 | 中-高 | ≈ 基线 |
| 概念对齐 | CB-LLM, CBMs | 概念分数 | 高 | 低 | ↓ 或 ≈ |
| 表征可分解 | Backpack, CoCoMix | 含义向量/连续概念 | 中 | 高 | ↓ 或 ≈ |
| 显式模块化 | MoE-X, MONET | 稀疏专家/单义专家 | 低-高 | 低-中 | ≈ 或 ↑ |
| 稀疏归纳 | Weight-Sparse, GLU | 稀疏电路/激活路径 | 极高/低 | 低 | ↓ 或 ≈ |

### 消融实验

各范式可解释性-性能 trade-off 对比：

| 范式 | 保真度 | 粒度 | 可扩展性 | 性能保持 |
|------|-------|------|---------|---------|
| 功能透明性 | 最高 | 特征级 | 差 | 中 |
| 概念对齐 | 高 | 概念级 | 中 | 中 |
| 表征可分解 | 中 | 词/概念级 | 中 | 中 |
| 显式模块化 | 中 | 专家/路由级 | 好 | 好 |
| 稀疏归纳 | 中-高 | 电路/神经元级 | 好 | 中 |

### 关键发现

- 显式模块化（MoE 类方法）在可扩展性和性能保持方面最有优势，是目前最有前景的范式
- 功能透明性方法保真度最高但可扩展性最差，难以直接应用于数十亿参数的 LLM
- 概念对齐方法依赖人工概念定义，CB-LLM 开始探索自动概念发现但仍处于早期
- $L_0$ 正则化产生的权重稀疏模型虽然电路可解释，但训练成本极高（约 3x 标准训练）
- GLU/SwiGLU 是"免费"的稀疏归纳——几乎所有现代 LLM 已在使用，但其可解释性潜力尚未被充分挖掘

## 亮点与洞察

- **五范式分类框架**非常清晰实用——将分散的文献统一在共同的设计原则下，便于研究者定位自己的工作和发现研究空白
- **"可解释性不一定牺牲性能"的论证**有力——MoE-X、B-cos LMs 等方法表明，精心设计的归纳偏置可以在保持性能的同时提供可解释性
- **跨范式组合的潜力**被明确指出——例如将概念对齐与显式模块化结合（概念瓶颈 + MoE），或表征可分解与稀疏归纳结合，开辟了广阔的研究空间

## 局限与展望

- 大部分内在可解释方法仅在中小规模模型上验证，是否能扩展到百亿/千亿参数 LLM 尚不确定
- 缺乏统一的可解释性评估指标——不同方法的"可解释性"定义和衡量标准不一致
- 对多模态大模型的内在可解释性研究几乎空白
- 未来方向包括：可解释性与安全对齐的结合、可解释的推理链路追踪、动态稀疏激活的可解释性分析

## 相关工作与启发

- **vs 事后解释综述（Madsen et al., 2022; Zhao et al., 2024）**: 这些综述聚焦于分析已训练模型的工具（如探针、注意力可视化），本文关注从设计层面构建透明性
- **vs 机制可解释性（Sharkey et al., 2025）**: 机制可解释性是事后方法中最接近内在可解释的方向，但仍是"逆向工程"而非"正向设计"

## 评分

- 新颖性: ⭐⭐⭐⭐ 五范式分类框架是新贡献，但综述本身不提出新方法
- 实验充分度: ⭐⭐⭐ 综述论文，无原创实验，但 Table 1 的对比整理很有参考价值
- 写作质量: ⭐⭐⭐⭐⭐ 分类清晰，覆盖全面，适合作为该领域的入门指南
- 价值: ⭐⭐⭐⭐ 为快速增长的内在可解释性领域提供了急需的结构化框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)
- [\[ACL 2026\] From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context Language Models](from_interpretability_to_performance_optimizing_retrieval_heads_for_long-context.md)

</div>

<!-- RELATED:END -->
