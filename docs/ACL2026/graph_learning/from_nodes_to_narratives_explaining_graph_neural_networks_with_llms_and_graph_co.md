---
title: >-
  [论文解读] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context
description: >-
  [ACL 2026][图学习][图神经网络] 本文提出 Gspell，一个轻量级后验解释框架，通过将 GNN 节点嵌入投影到 LLM 嵌入空间并构建混合提示（软提示+文本），使 LLM 能够直接推理 GNN 内部表示并生成自然语言解释和解释子图，在文本属性图（TAG）上实现了忠实性与可解释性的良好平衡。
tags:
  - ACL 2026
  - 图学习
  - 图神经网络
  - LLM 解释器
  - 软提示
  - 文本属性图
  - 自然语言解释
---

# From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context

**会议**: ACL 2026  
**arXiv**: [2508.07117](https://arxiv.org/abs/2508.07117)  
**代码**: 无  
**领域**: 图学习 / 可解释性  
**关键词**: GNN 可解释性, LLM 解释器, 软提示, 文本属性图, 自然语言解释

## 一句话总结

本文提出 Gspell，一个轻量级后验解释框架，通过将 GNN 节点嵌入投影到 LLM 嵌入空间并构建混合提示（软提示+文本），使 LLM 能够直接推理 GNN 内部表示并生成自然语言解释和解释子图，在文本属性图（TAG）上实现了忠实性与可解释性的良好平衡。

## 研究背景与动机

**领域现状**：GNN 在医疗、药物设计、推荐系统等高风险领域广泛应用，其预测的可信度至关重要。现有 GNN 解释方法（如 GNNExplainer、PGExplainer）主要输出子图掩码或特征重要性分数，但在文本属性图（TAG）上表现不佳且不够人类可读。同时，LLM 与 GNN 的整合主要集中在提升 GNN 任务性能，而非解释 GNN 预测。

**现有痛点**：(1) 现有 LLM-GNN 解释框架依赖刚性模板将 GNN 解释器输出与 LLM 输入对齐，需要手工评分或额外训练；(2) 现有方法未直接利用 GNN 内部表示，导致解释泛化或不忠于 GNN 实际工作方式；(3) 调用外部 GNN 解释器可能偏置 LLM 的推理——当解释器本身有噪声时，会误导 LLM 的判断。

**核心矛盾**：GNN 的嵌入空间与 LLM 的 token 空间存在根本性的不对齐——如何让 LLM "看到"并理解 GNN 的内部表示，而不是仅依赖外部解释器的二手信息？

**本文目标**：设计一个无需外部解释器、直接将 GNN 内部表示注入 LLM 的后验解释框架，生成忠实且可解释的自然语言解释。

**切入角度**：类比多模态对齐（如 CLIP 对齐图像和文本嵌入），将 GNN 嵌入投影为 LLM 的软提示 token，使 LLM 既能利用 GNN 学到的结构信息，又能发挥自身的语言推理能力。

**核心 idea**：训练一个投影器将 GNN 节点嵌入映射到 LLM token 空间，构建交错软提示与文本的混合提示，让 LLM 直接从 GNN 表示生成自然语言解释，并从中提取解释子图。

## 方法详解

### 整体框架

Gspell 包含三个步骤：(1) **投影器训练**——通过上下文对齐损失和对比损失将 GNN 嵌入映射到 LLM token 空间；(2) **混合提示构建**——将投影后的嵌入作为软提示与节点文本描述交错排列；(3) **解释生成**——LLM 基于混合提示生成自然语言解释，并从中提取支持/反对/中立节点，构建解释子图。

### 关键设计

1. **GNN-LLM 嵌入投影器（Embedding Projector）**:

    - 功能：将 GNN 节点嵌入对齐到 LLM 的 token 嵌入空间
    - 核心思路：投影器 $\Pi: \mathbb{R}^m \to \mathbb{R}^{k \times h}$ 将 GNN 嵌入 $f_\Phi(v)$ 映射为 $k$ 个软提示 token。优化两个损失：(a) 上下文对齐损失——鼓励平均软 token 表示与节点文本的 LLM 嵌入对齐（余弦相似度）；(b) 对比损失——保持 GNN 嵌入的相似性结构在投影空间中不变（KL 散度最小化）
    - 设计动机：GNN 嵌入和 LLM token 处于完全不同的空间，直接注入会导致语义失配；双重损失确保投影既保留 GNN 的结构信息又与 LLM 的语义空间对齐

2. **混合提示构建（Hybrid Prompt）**:

    - 功能：将 GNN 结构信息和文本信息以统一格式输入 LLM
    - 核心思路：对目标节点 $v$，构建其 GNN 计算树 $\mathcal{T}^{\phi}_v$（深度为 GNN 层数 $L$ 的树），将计算树中每个节点的软提示嵌入与文本描述交错排列。LLM 输入序列为：系统提示 → 目标节点软提示+文本 → 计算树节点（各自的软提示+文本）→ 查询指令
    - 设计动机：将 GNN 嵌入视为 LLM 的"原生 token"，使 LLM 能同时推理 GNN 的结构表示和节点文本特征，而非仅依赖文本

3. **解释子图提取与幻觉缓解**:

    - 功能：从 LLM 生成的自然语言解释中提取结构化的解释子图
    - 核心思路：LLM 预测计算树中每个节点是支持（+1）、反对（-1）还是中立（0）目标节点的分类。支持节点集 $S^+_v$ 构成解释子图。通过两种机制缓解幻觉：(a) 利用 GNN 嵌入约束 LLM 推理；(b) 后处理验证确保引用的节点确实存在于计算树中
    - 设计动机：纯文本解释不够结构化，子图解释不够可读——结合两者提供完整的解释

### 损失函数 / 训练策略

投影器训练损失 $\mathcal{L} = \beta \mathcal{L}_{context} + (1-\beta) \mathcal{L}_{contrast}$。GNN 和 LLM 均冻结，仅训练投影器。推理时无需额外微调，即插即用。

## 实验关键数据

### 主实验

**节点分类解释质量（Cora 数据集）**

| 方法 | Fidelity+ ↑ | Fidelity- ↓ | Sparsity ↑ | Insightfulness ↑ |
|------|------------|------------|-----------|-----------------|
| GNNExplainer | 0.12 | 0.08 | 0.65 | — |
| PGExplainer | 0.15 | 0.10 | 0.70 | — |
| GraphLLM | 0.18 | 0.12 | 0.55 | 2.8 |
| **Gspell** | **0.22** | **0.06** | **0.72** | **3.5** |

### 消融实验

| 配置 | Fidelity+ | Sparsity | Insightfulness |
|------|-----------|----------|----------------|
| 无软提示（纯文本） | 0.14 | 0.68 | 2.9 |
| 无对比损失 | 0.18 | 0.70 | 3.2 |
| 无上下文对齐 | 0.16 | 0.69 | 3.0 |
| **完整 Gspell** | **0.22** | **0.72** | **3.5** |

### 关键发现

- 软提示的引入显著提升忠实度（+0.08 Fidelity+），证明 GNN 内部表示为解释提供了传统文本输入无法捕获的信息
- 双重损失设计中两个组件互补——上下文对齐保证语义一致性，对比损失保持结构信息
- Gspell 在 insightfulness（由人类评估的洞察力指标）上大幅领先，证明自然语言解释比子图掩码更易于人类理解
- 即插即用的特性（无需微调 GNN 或 LLM）使其适用于已部署的模型

## 亮点与洞察

- "绕过传统 GNN 解释器，让 LLM 直接解读 GNN 内部表示"的思路简洁有力——减少了中间环节的信息损失和偏差
- 投影器训练中对比损失的设计巧妙——不仅要求单个嵌入对齐，还要求嵌入间的相对关系在两个空间中一致
- 混合提示的交错设计使 LLM 能同时"看到"每个节点的数值表示和文本描述，形成多视角推理

## 局限与展望

- 仅在节点分类任务上验证，未探索图级别分类或链接预测的解释
- 投影器质量依赖于 GNN 嵌入的可分性——当 GNN 嵌入空间结构混乱时，投影效果可能打折
- 解释子图的提取依赖 LLM 的输出解析，可能受 LLM 格式遵从能力影响
- 计算树的构建需要知道 GNN 架构的层数，限制了对黑盒 GNN 的适用性

## 相关工作与启发

- **vs GNNExplainer**: GNNExplainer 通过优化掩码生成子图解释，但不生成自然语言；Gspell 同时提供子图和自然语言解释
- **vs Pan et al. (2024)**: 先用外部解释器生成伪标签再微调 LLM，有外部解释器偏差；Gspell 直接从 GNN 嵌入推理
- **vs He et al. (2024b)**: 生成反事实解释但通过自编码器中转；Gspell 通过投影器直接桥接
- **vs 多模态对齐**: 类比 CLIP/LLaVA 的视觉-语言对齐，Gspell 实现了 GNN-语言对齐

## 评分

- 新颖性: ⭐⭐⭐⭐ 将多模态对齐思路应用于 GNN 可解释性是新颖的，但投影器设计相对标准
- 实验充分度: ⭐⭐⭐ 在真实 TAG 数据集上评估，但缺少更多数据集和大规模 GNN 的验证
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，框架设计动机充分
- 价值: ⭐⭐⭐⭐ 为 GNN 可解释性提供了新方向，实用性高（即插即用）

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks](../../ICLR2026/graph_learning/logicxgnn_grounded_logical_rules_for_explaining_graph_neural_networks.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ACL 2025\] GraphNarrator: Generating Textual Explanations for Graph Neural Networks](../../ACL2025/graph_learning/graphnarrator.md)
- [\[ACL 2026\] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs](llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp.md)

<!-- RELATED:END -->
