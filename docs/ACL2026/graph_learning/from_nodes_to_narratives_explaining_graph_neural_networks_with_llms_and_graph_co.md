---
title: >-
  [论文解读] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context
description: >-
  [ACL 2026][图学习][图神经网络] 本文提出 Gspell，一个轻量级后验解释框架，通过将 GNN 节点嵌入投影到 LLM 嵌入空间并构建混合提示（软提示+文本），使 LLM 能够直接推理 GNN 内部表示并生成自然语言解释和解释子图，在文本属性图（TAG）上实现了忠实性与可解释性的良好平衡。
tags:
  - "ACL 2026"
  - "图学习"
  - "图神经网络"
  - "LLM 解释器"
  - "软提示"
  - "文本属性图"
  - "自然语言解释"
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

Gspell 是一个即插即用的后验解释框架：它要解决的是「GNN 的嵌入空间与 LLM 的 token 空间根本不对齐，LLM 看不懂 GNN 在想什么」这一矛盾。做法是先训一个投影器把冻结 GNN 的节点嵌入映射进 LLM 的 token 空间，再把投影后的软提示与节点文本交错拼成混合提示喂给冻结的 LLM，让 LLM 直接读 GNN 的内部表示、生成自然语言解释，并顺带判定计算树里每个节点是支持还是反对目标预测，从而抽出一张解释子图。整条链路里 GNN 与 LLM 都不动，只有中间那层投影器需要学。

### 关键设计

**1. GNN-LLM 嵌入投影器：把 GNN 表示翻译成 LLM 听得懂的「原生 token」。**

GNN 嵌入和 LLM token 处在完全不同的空间，直接把数值嵌入塞进 prompt 只会语义失配，所以 Gspell 用投影器 $\Pi:\mathbb{R}^m \to \mathbb{R}^{k\times h}$ 把单个 GNN 嵌入 $f_\Phi(v)$ 展开成 $k$ 个软提示 token。训练时同时压两个损失来保证「既忠于 GNN、又对得上 LLM」：上下文对齐损失让 $k$ 个软 token 的平均表示在余弦相似度上靠拢节点文本的 LLM 嵌入，负责语义这一头；对比损失则用 KL 散度约束投影前后 GNN 嵌入之间的相似性结构保持不变，负责结构这一头。两个损失一拉一稳，投影出的软提示才既保留了 GNN 学到的邻域结构，又落进了 LLM 能解析的语义空间。

**2. 混合提示构建：让 LLM 同时「看见」结构表示和节点文本。**

光有软提示还不够，LLM 需要在正确的图结构语境里读这些 token。Gspell 对目标节点 $v$ 先展开它的 GNN 计算树 $\mathcal{T}^{\phi}_v$——深度等于 GNN 层数 $L$ 的那棵消息传递树，再把树上每个节点的软提示嵌入与其文本描述交错排成一条序列：系统提示 → 目标节点（软提示 + 文本）→ 计算树各节点（各自软提示 + 文本）→ 查询指令。这样 LLM 拿到的不再是抽象的二手解释，而是每个节点同时带着「数值结构表示」和「自然语言描述」的双视角输入，能照着 GNN 真实的感受野去推理，而非仅凭文本猜测。

**3. 解释子图提取与幻觉缓解：把自由文本落回结构化子图。**

纯文本解释不够结构化、纯子图掩码又不够可读，Gspell 让 LLM 在生成解释的同时给计算树里每个节点打一个三分类标签——支持（+1）、反对（-1）或中立（0），所有支持节点构成的集合 $S^+_v$ 即为解释子图，两种表达就此合二为一。为防 LLM 凭空捏造证据，框架用两道闸缓解幻觉：一是用 GNN 嵌入约束 LLM 的推理，让结论锚定在真实表示上；二是做后处理验证，确保解释里引用的节点确实存在于计算树中，杜绝引用不存在的邻居。

### 损失函数 / 训练策略

投影器的训练损失是 $\mathcal{L} = \beta \mathcal{L}_{context} + (1-\beta) \mathcal{L}_{contrast}$ 的加权组合；GNN 与 LLM 全程冻结，只训这一层投影器，推理时不需任何额外微调，可直接套到已部署模型上。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks](../../ICLR2026/graph_learning/logicxgnn_grounded_logical_rules_for_explaining_graph_neural_networks.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ACL 2025\] GraphNarrator: Generating Textual Explanations for Graph Neural Networks](../../ACL2025/graph_learning/graphnarrator.md)
- [\[ACL 2026\] Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks](evaluating_llms_on_large-scale_graph_property_estimation_via_random_walks.md)

</div>

<!-- RELATED:END -->
