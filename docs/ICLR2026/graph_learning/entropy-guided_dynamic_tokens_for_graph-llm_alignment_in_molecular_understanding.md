---
title: >-
  [论文解读] Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding
description: >-
  [ICLR 2026][图学习][Graph-LLM对齐] 提出 EDT-Former（Entropy-guided Dynamic Token Transformer），通过熵引导的动态token生成机制，在冻结图编码器和LLM之间建立高效对齐…
tags:
  - "ICLR 2026"
  - "图学习"
  - "Graph-LLM对齐"
  - "动态Token"
  - "分子图"
  - "Q-Former"
  - "熵引导"
---

# Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding

**会议**: ICLR 2026  
**arXiv**: [2602.02742](https://arxiv.org/abs/2602.02742)  
**代码**: 无  
**领域**: 图学习 / 分子理解  
**关键词**: Graph-LLM对齐, 动态Token, 分子图, Q-Former, 熵引导

## 一句话总结
提出 EDT-Former（Entropy-guided Dynamic Token Transformer），通过熵引导的动态token生成机制，在冻结图编码器和LLM之间建立高效对齐，无需微调LLM主干网络即在分子问答、分子指令和属性预测等多个基准上达到SOTA。

## 研究背景与动机
分子理解是科学发现（药物设计、材料发现等）的核心环节，而大语言模型（LLM）在处理分子图结构方面存在天然的困难——LLM擅长处理序列文本，但分子是图结构数据，包含原子连接关系、立体化学信息和子结构上下文。

现有的图-LLM桥接方案主要借鉴视觉-语言领域的Q-Former架构，使用固定长度的静态查询token来压缩图信息。但这种方案存在三个核心问题：

**静态token的信息损失**：固定长度的token序列无法根据分子复杂度自适应调整，简单分子可能被过度表示，复杂分子则信息不足。Q-Former最初为视觉任务设计，图结构数据的拓扑信息、立体化学特性无法被有效捕获

**忽略立体化学和子结构**：分子的三维构型和功能基团是理解化学性质的关键，但现有的固定token方法难以保留这些局部和全局特征

**昂贵的LLM微调**：大多数方法需要对LLM主干网络进行微调，计算成本高且泛化性受限

核心idea是：利用信息熵来自适应地确定每个分子需要多少个token以及这些token应关注分子的哪些部分，实现动态的、内容感知的图到文本表征转换。

## 方法详解

### 整体框架
EDT-Former的pipeline包含三个核心组件：
- **图编码器**（冻结）：将分子图编码为节点级别的表征
- **EDT-Former连接器**：将图表征转化为动态数量和内容的token序列
- **LLM**（冻结主干，仅微调嵌入层）：接收token序列和文本指令，产生输出

整体目标是在**不微调LLM主干**的前提下，仅通过EDT-Former连接器和嵌入层的轻量微调，实现分子图与语言的有效对齐。

### 关键设计

1. **熵引导的动态Token生成（Entropy-Guided Dynamic Tokenization）**: 这是方法的核心创新。与Q-Former使用固定数量的静态查询不同，EDT-Former根据分子图的信息分布动态决定token的数量和关注点。具体做法是：

    - 首先计算图编码器输出的节点表征的信息熵分布，熵值高的区域代表信息丰富/不确定性大的分子片段
    - 根据熵分布将分子图划分为多个"分子补丁"（molecular patches），每个补丁对应分子中一个信息密集的子结构
    - 为每个补丁生成一个或多个动态token，高熵区域分配更多token以保留更多信息
    - 这样，复杂分子自动获得更多token表示，简单分子使用更少token，实现了计算效率和表示质量的平衡

2. **分子补丁对齐（Molecular Patch Alignment）**: EDT-Former生成的动态token与信息丰富的分子补丁对齐，确保每个token都承载有意义的结构化学信息。这一设计的动机是：在视觉领域，ViT的补丁包含空间上连续的像素信息；类比地，分子补丁应包含拓扑上连续且化学上有意义的原子团信息。通过对齐机制，token既保留了局部子结构特征（如功能基团），又通过注意力机制整合了全局结构信息（如分子骨架拓扑）。

3. **高效对齐训练策略**: EDT-Former采用了参数高效的训练方案：

    - 图编码器完全冻结，保留预训练的通用分子表征能力
    - LLM主干完全冻结，保留语言理解和生成能力
    - 仅训练EDT-Former连接器和LLM的嵌入层
    - 这种设计大幅降低了计算成本（相比全量微调），同时保持了模型的泛化能力
    - 训练目标包括对齐损失（使token表征与LLM嵌入空间对齐）和任务损失（如问答准确率、属性预测误差）

### 损失函数 / 训练策略
训练采用多阶段策略：先进行图-文本对齐预训练（使token表征与文本嵌入空间对齐），再在下游任务上微调。损失函数包含对比学习损失（拉近匹配的分子-文本对，推远不匹配的对）和生成损失（如交叉熵用于问答任务）。关键点是整个训练过程的参数量仅为EDT-Former连接器和嵌入层，大幅减少了可训练参数。

## 实验关键数据

### 主实验
EDT-Former在四类分子理解基准上进行了评估，在所有基准上均达到或超越SOTA：

| 基准数据集 | 任务类型 | EDT-Former | 之前SOTA | 核心发现 |
|-----------|---------|-----------|---------|---------|
| MoleculeQA | 分子问答 | SOTA | Q-Former variants | 动态token显著优于静态token |
| Mol-Instructions | 分子指令跟随 | SOTA | 需要LLM微调的方法 | 无需微调LLM即超越需微调的方法 |
| TDC | 属性预测 | SOTA | 图模型+LLM微调 | 在多个子任务上一致领先 |
| MoleculeNet | 属性预测 | SOTA | 传统图神经网络 | 特别在低数据量场景优势明显 |

### 消融实验

| 配置 | 关键指标变化 | 说明 |
|------|------------|------|
| 固定token vs 动态token | 动态token显著更优 | 验证了自适应token生成的必要性 |
| 有熵引导 vs 无熵引导 | 有引导更优 | 熵信号有效指导了token分配 |
| 冻结LLM vs 微调LLM | 冻结LLM效果可比 | 说明EDT-Former的对齐质量足够高 |
| 不同图编码器 | EDT-Former在多种编码器上有效 | 框架具有通用性 |

### 关键发现
- 熵引导的动态token在所有任务上一致优于固定长度token，证明了自适应表征长度的重要性
- EDT-Former无需微调LLM主干即超越需要全量微调的方法，展示了高效图-语言对齐的可行性
- 在分子属性预测这种需要精确数值理解的任务上，EDT-Former也表现出色，说明动态token有效保留了分子的定量化学信息
- 仅微调嵌入层是一个关键设计选择——完全不微调LLM的任何部分效果较差，但微调嵌入层即可大幅弥补差距

## 亮点与洞察
- **从视觉到分子的适配思路**：巧妙地将视觉语言领域的Q-Former范式引入分子理解，同时解决了直接搬用的缺陷（静态token、忽略拓扑结构）
- **熵作为信息分配信号**：使用信息熵来决定token数量和分配是一个优雅的设计——高熵区域确实需要更精细的表征
- **冻结主干+轻量连接器的范式**：EDT-Former进一步验证了"冻结大模型+训练小型连接器"这一高效范式在分子领域的有效性
- **动态长度表征的一般性价值**：动态token的思想可推广到其他图-语言任务（如蛋白质理解、材料设计等）

## 局限与展望
- 当前仅验证了2D分子图的场景，对3D分子构象（如蛋白质折叠构型）的处理能力未探索
- 动态token数量的上限和下限如何设定可能影响效率和效果的平衡，需要进一步的灵敏度分析
- 嵌入层的微调虽然参数量小，但仍然需要足够的对齐数据，在低资源化学领域可能受限
- 与最新的分子大模型（如Galactica、Mol-GPT等端到端模型）的对比不够充分
- 熵引导的token生成引入了额外的计算步骤，对于大规模分子筛选场景的推理效率影响需要评估

## 相关工作与启发
- **BLIP-2 / Q-Former**: EDT-Former的设计灵感直接来源于视觉-语言领域的Q-Former，但通过动态化和熵引导解决了图结构数据的特殊挑战
- **MolCA、MoMu等分子-语言方法**: 这些工作建立了分子-语言对齐的基础，EDT-Former在此基础上通过动态token实现了更细粒度的对齐
- **GNN + LLM联合框架**: EDT-Former属于"图编码器+连接器+LLM"范式的一员，其核心贡献在于连接器的设计
- 本文的方法启发我们：在多模态对齐中，连接器的设计（尤其是动态vs静态、内容感知vs固定）可能比增大模型规模更重要

## 评分
- 新颖性: ⭐⭐⭐⭐ （熵引导动态token是一个有效的创新点，但整体框架仍是Q-Former变体）
- 实验充分度: ⭐⭐⭐⭐⭐ （四类基准，全面的消融实验）
- 写作质量: ⭐⭐⭐⭐ （动机清晰，实验设计合理）
- 价值: ⭐⭐⭐⭐ （为分子理解的多模态方法提供了新的高效范式）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] MolLangBench: A Comprehensive Benchmark for Language-Prompted Molecular Structure Recognition, Editing, and Generation](mollangbench_a_comprehensive_benchmark_for_language-prompted_molecular_structure.md)
- [\[NeurIPS 2025\] ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG](../../NeurIPS2025/graph_learning/remindrag_low-cost_llm-guided_knowledge_graph_traversal_for_efficient_rag.md)
- [\[AAAI 2026\] Hyperbolic Continuous Structural Entropy for Hierarchical Clustering](../../AAAI2026/graph_learning/hyperbolic_continuous_structural_entropy_for_hierarchical_clustering.md)
- [\[ICLR 2026\] RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation](ras_retrieval-and-structuring_for_knowledge-intensive_llm_generation.md)
- [\[ICLR 2026\] Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)

</div>

<!-- RELATED:END -->
