---
title: >-
  [论文解读] From Sequence to Structure: Uncovering Substructure Reasoning in Transformers
description: >-
  [NeurIPS 2025][图学习][Transformer] 本文通过实证和理论分析揭示 decoder-only Transformer 如何从文本序列中理解图结构，提出"诱导子图过滤"（ISF）解释子结构逐层识别机制，并扩展到 LLM 验证一致性、复合图推理（Thinking-in-Substructures）和属性图（分子图）子结构提取。
tags:
  - "NeurIPS 2025"
  - "图学习"
  - "Transformer"
  - "子结构提取"
  - "图推理"
  - "诱导子图过滤"
  - "LLM图理解"
---

# From Sequence to Structure: Uncovering Substructure Reasoning in Transformers

**会议**: NeurIPS 2025  
**arXiv**: [2507.10435](https://arxiv.org/abs/2507.10435)  
**代码**: [https://github.com/DDigimon/From_Sequence_to_Structure](https://github.com/DDigimon/From_Sequence_to_Structure)  
**领域**: 图学习  
**关键词**: Transformer可解释性, 子结构提取, 图推理, 诱导子图过滤, LLM图理解

## 一句话总结
本文通过实证和理论分析揭示 decoder-only Transformer 如何从文本序列中理解图结构，提出"诱导子图过滤"（ISF）解释子结构逐层识别机制，并扩展到 LLM 验证一致性、复合图推理（Thinking-in-Substructures）和属性图（分子图）子结构提取。

## 研究背景与动机

**领域现状**：LLM 已被证明能解决图推理任务——即使图结构嵌入在文本描述中，LLM 仍能识别节点连接、检测模式、比较子图。但 Transformer 本质上是序列模型，不具备原生的图结构理解能力。

**现有痛点**：现有对 Transformer 图推理能力的理解集中在**最短路径**任务上（如 SLN 的谱导航或 ALPINE 的路径组合），但这些都是线性结构。真实图包含大量非线性子结构（环、树、motif），现有理论无法解释 Transformer 如何处理这些。

**核心矛盾**：Transformer 是序列处理架构，但需要理解非欧几何结构（图），从路径推广到一般子结构的理论空白。

**本文目标**：(a) Transformer 如何逐层识别子结构？(b) 输入格式（邻接表 vs 边列表）和查询提示如何影响识别？(c) 这些发现在 LLM 中是否一致？(d) 能否扩展到更复杂的图推理？

**切入角度**：从子结构提取任务入手（比路径更一般），通过可视化 + 理论分析揭示 Transformer 的内部机制。

**核心 idea**：Transformer 通过逐层的"诱导子图过滤"（ISF）过程渐进式识别子结构，且这一机制在从头训练的 Transformer 和 LLM 中一致存在。

## 方法详解

### 整体框架
输入是图的文本表示 $\text{Encoder}_G(G)$ + 查询提示 $\text{Encoder}_T(T)$ 的拼接序列 $Q$，Transformer 输出子结构节点集合 $\text{Encoder}_A(\hat{G})$。研究分三个层面：
1. **内部机制分析**（Section 3.1）：ISF 过程
2. **输入查询影响**（Section 3.2）：图表示格式和提示类型
3. **扩展应用**（Section 4-5）：LLM 验证、复合图推理、属性图

### 关键设计

1. **诱导子图过滤（Induced Substructure Filtration, ISF）**:

    - 功能：解释 Transformer 如何跨层逐步识别子结构。
    - 核心思路：定义 $k$-Node $m$-Filtration $\mathcal{F}(V')=(V'_1,\dots,V'_m)$，其中 $\emptyset\neq V'_1\subseteq\cdots\subseteq V'_m=V'$，对应诱导子图序列 $(G'_1,\dots,G'_m)$。每个 Transformer 层对应一步过滤——从空集开始，逐步聚合节点直到识别完整子结构。
    - 理论支撑：**Theorem 3.3**（渐进识别表达力）证明具有 $m+2$ 层、$O(n^k)$ 隐藏维度的 log-precision Transformer 可在第 $i+2$ 层输出第 $i$ 步过滤的子图同构指示张量 $\mathcal{T}(G,G'[V'_i])$。**Theorem 3.5** 证明在唯一子结构假设下，常数深度 Transformer 可输出正确的 $k$-tuple 匹配。
    - 可视化验证：t-SNE 投影显示——在 layer 2 中共享部分节点的子结构开始聚类，layer 3 中共享更多节点的进一步靠近，最终层完全分离。答案在生成步骤前就已确定。

2. **多子结构同时检测**:

    - **Single-Shape-Multi-Num**：图中有多个同形状子结构。Theorem 3.8 证明常数深度 Transformer 可完成此任务，实验中三角形/正方形检测 >85% 准确率，子结构数量对性能影响小。
    - **Multi-Shape-Single-Num**：图中有不同形状子结构。将不同形状训练数据混合，Transformer 由查询提示引导识别特定形状。Theorem 3.10 证明对 $|V'|=k'\leq k$ 的任意目标子图均可完成。有趣发现：简单子结构（如三角形）在浅层（layer 3）就被识别，复杂子结构推迟到更深层。

3. **输入查询格式分析**:

    - **图表示**：邻接表（AL）和边列表（EL）理论上等价（都可转化为邻接矩阵 $A(G)$ 的向量化 $\text{vec}(A(G))$），但 EL 需要更多 token（全连通图 $3(|V|^2-|V|)$ vs AL 的 $2|V|^2-|V|$），因此实际需要更多层。
    - **查询提示**：术语型（"triangle"）和拓扑型（"A:BC,B:C"）均可工作，但术语型更好（>85% vs ~80%）。扰动实验发现 Transformer 并非完整理解拓扑描述，而是抽象为关键 token 序列来判断子结构类型。

4. **LLM 一致性验证（Section 4）**:

    - 在 fine-tuned LLaMA 3.1-8B-Instruct 上验证 ISF：t-SNE 可视化显示类似的逐层聚类行为，ARI/NMI 指标随层深度增加。
    - 差异：LLM 会生成解释性内容（23% 包含 Python 代码），导致末尾层 ARI/NMI 略降。

5. **Thinking-in-Substructures (Tins, Section 5.1)**:

    - 功能：将复杂子结构分解为更简单的组成部分进行推理。
    - 核心思路：输出重新格式化为 $\text{ANS}_{\text{Tins}}=(\{P_1\},\{P_2\},\dots,\{P_t\},\text{<ANS>},\text{ANS}(\hat{G}))$，其中 $\{P_i\}$ 是分解后的子结构。
    - 效果：将表达力需求从 $O(n^k)$ 降到 $O(n^q)$（$q<k$），100K 训练样本下性能提升 ~10%，diagonal 结构在 3 层时提升 46%。

6. **属性图扩展（Section 5.2）**:

    - 将节点特征（如原子类型）编入邻接表 $\text{AL}_f(G)$
    - 在分子图上测试功能团提取：Hydroxyl 92.07%、Carboxyl 91.59%、Benzene 72.45%、混合 89.46%

## 实验关键数据

### 主实验（单子结构提取准确率）

| 训练量 | 层数 | Triangle | Square | Pentagon | House | Diamond |
|--------|------|----------|--------|----------|-------|---------|
| 100K | 2 | 0.530 | 0.194 | 0.363 | 0.371 | 0.066 |
| 100K | 3 | 0.966 | 0.399 | 0.564 | 0.560 | 0.164 |
| 300K | 4 | 0.995 | 0.940 | 0.863 | 0.839 | 0.877 |
| 400K | 5 | **0.998** | **0.968** | **0.892** | **0.853** | **0.931** |

### Thinking-in-Substructures 消融

| 子结构 | 直接预测(4层) | Tins(4层) | 直接预测(3层) | Tins(3层) |
|--------|--------------|-----------|--------------|-----------|
| Diagonal | 0.631 | **0.865** | 0.200 | **0.661** |
| Diamond | 0.476 | **0.779** | 0.129 | **0.434** |
| House | 0.589 | **0.807** | 0.564 | **0.668** |
| Complex | 0.118 | **0.227** | 0.121 | **0.212** |

### 关键发现
- **层数与子结构大小正相关**：3 节点子结构（三角形）3 层即可 99%，4 节点需 4 层达 85%+，5 节点需 5 层。这与 ISF 理论（$m+2$ 层）一致。
- **数据量同样重要**：Pentagon 从 100K→400K 训练数据准确率从 36% 升到 89%，说明复杂子结构需要更多样本。
- **AL 格式优于 EL**：在相同层数下 AL 准确率更高，因为 token 更紧凑，信息密度更大。
- **术语提示优于拓扑提示**：Transformer 倾向于将子结构概念抽象为 token 序列，而非完整理解拓扑描述。
- **Tins 显著提升有限训练数据下的性能**：通过分解复杂结构，降低了对模型容量和数据量的需求。

## 亮点与洞察
- **ISF 理论框架**提供了理解 Transformer 图推理的统一视角：过滤过程（filtration）这一拓扑学概念优雅地刻画了逐层信息聚合行为。该框架不仅解释路径（已有工作），也解释环、树等一般子结构。
- **"Transformer 并不真正理解拓扑，而是抽象为 token 模式"**这个发现很有洞察力：扰动实验表明 Transformer 依赖特定 token cue 而非完整结构理解，暗示了 LLM 图推理能力的边界。
- **Tins（中间分解思维）**是一种巧妙的 CoT 变体：将复杂图结构分解为简单组件，降低了 $O(n^k) \to O(n^q)$ 的表达力需求。这个思路可推广到其他组合优化任务。
- 在 LLaMA 上的验证证实 ISF 不是小模型的特殊现象，而是 decoder-only 架构的固有行为。

## 局限与展望
- 实验中的 Transformer 是轻量 GPT-2 变体（384 hidden dim），虽然 LLaMA 验证了一致性，但更大规模模型上的定量分析缺失。
- 合成图数据（4-16 节点）与真实知识图谱/社交网络的规模和结构差距大，ISF 在大规模图上是否仍然成立需要验证。
- Theorem 3.3 要求 $O(n^k)$ 隐藏维度，对大图不现实（$n=100, k=5$ 需 $10^{10}$ 维），实际中如何绕过这个理论瓶颈未讨论。
- 属性图实验仅在分子图上做了初步验证，更丰富的节点/边特征类型（如连续值、多模态）未涉及。
- Tins 需要人工定义分解方式，自动化分解策略是自然的后续方向。

## 相关工作与启发
- **vs ALPINE**: ALPINE 证明 GPT 层可嵌入邻接矩阵和可达性矩阵，但局限于路径任务。本文 ISF 推广到任意子结构，是 ALPINE 的直接扩展。
- **vs SLN (Spectral Line-Graph Navigation)**: SLN 发现 Transformer 在最短路径任务中学到谱嵌入而非 Dijkstra 规则。ISF 提供了更一般的解释：子结构识别通过过滤而非谱方法。
- **vs GraphPatt**: GraphPatt 评估 LLM 的图模式理解能力，报告了术语提示优于拓扑提示。本文从机制层面解释了这一现象——Transformer 将子结构抽象为 token 序列。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ ISF 框架首次从拓扑学角度统一解释 Transformer 的子结构推理机制，理论和实验互补
- 实验充分度: ⭐⭐⭐⭐ 涵盖单/多子结构、不同图表示、LLM验证、Tins、属性图，但规模有限
- 写作质量: ⭐⭐⭐⭐ 逻辑链清晰（单→多→LLM→扩展），理论定义严谨；但内容量大导致部分细节需查附录
- 价值: ⭐⭐⭐⭐⭐ 对"LLM如何理解图结构"这一基础问题有重要推进，ISF可指导未来图推理方法设计

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Relieving the Over-Aggregating Effect in Graph Transformers](relieving_the_over-aggregating_effect_in_graph_transformers.md)
- [\[NeurIPS 2025\] Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework](unifying_and_enhancing_graph_transformers_via_a_hierarchical_mask_framework.md)
- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [\[CVPR 2025\] NN-Former: Rethinking Graph Structure in Neural Architecture Representation](../../CVPR2025/graph_learning/nn-former_rethinking_graph_structure_in_neural_architecture_representation.md)
- [\[ICML 2025\] Banyan: Improved Representation Learning with Explicit Structure](../../ICML2025/graph_learning/banyan_improved_representation_learning_with_explicit_structure.md)

</div>

<!-- RELATED:END -->
