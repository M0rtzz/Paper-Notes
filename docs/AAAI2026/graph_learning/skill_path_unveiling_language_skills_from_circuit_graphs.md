---
title: >-
  [论文解读] Skill Path: Unveiling Language Skills from Circuit Graphs
description: >-
  [AAAI2026][图学习][机制可解释性] 提出 Skill Path 概念及三步框架（分解-剪枝-因果中介），从电路图中提取语言模型特定技能的线性路径，定量验证了技能的分层性（Stratification）和包容性（Inclusiveness）两大猜想。
tags:
  - "AAAI2026"
  - "图学习"
  - "机制可解释性"
  - "电路发现"
  - "skill path"
  - "causal mediation"
  - "语言模型"
---

# Skill Path: Unveiling Language Skills from Circuit Graphs

**会议**: AAAI2026 Oral  
**arXiv**: [2410.01334](https://arxiv.org/abs/2410.01334)  
**代码**: [GitHub](https://github.com/Zodiark-ch/Language-Skill-of-LLMs)  
**领域**: 因果推理  
**关键词**: 机制可解释性, 电路发现, skill path, causal mediation, 语言模型

## 一句话总结

提出 Skill Path 概念及三步框架（分解-剪枝-因果中介），从电路图中提取语言模型特定技能的线性路径，定量验证了技能的分层性（Stratification）和包容性（Inclusiveness）两大猜想。

## 背景与动机

Circuit Discovery 是理解语言模型内部机制的核心方法，通过剪枝计算图中低效边/节点来获取保持输出忠实度的电路图。然而现有电路图存在两个关键限制：

1. **技能混淆**：从真实数据集获得的电路图不可避免地包含与目标技能无关的其他技能效应（如 induction 数据集中可能混入 arithmetic 或 multi-choice 技能的路径）
2. **原子消融问题**：现有方法使用边或节点级别的消融，忽略了协同组件之间的因果依赖关系；同时 MLP 的非线性导致计算图无法完全线性化，难以隔离包含多个组件的路径效应

这两个限制使得现有电路图无法精确定位单一目标技能的机制。

## 核心问题

如何从粗粒度的电路图中提取细粒度的"技能路径"（Skill Path）——一种由线性组件链构成的紧凑子电路，能精确反映目标技能在电路中的位置，并排除其他技能效应和噪声的干扰？

## 方法详解

### 整体框架：三步法

#### Step 1: Decomposition（分解）

核心创新在于对 Transformer 的 MLP 层进行完全线性分解。由于 MLP 输入包含残差流 $X$ 和注意力输出 $\text{attn}(X)$，非线性激活使得 $\text{mlp}(\text{attn}(X)+X)$ 无法直接分解。作者引入**补偿组件（Compensation Component）**：

$$\text{mlp}(\text{attn}(X)+X) = \text{mlp}(\text{attn}(X)) + \text{mlp}(X) + \text{Cps}(X)$$

基于此，每层 Transformer 被解耦为 29 个组件（以 GPT2-small 为例，12层×12头）：

- $C^0$: Self（残差流直通）
- $C^{1-12}$: Attention（12个注意力头）
- $C^{13}$: MLP（纯 MLP 路径）
- $C^{14-25}$: Attention+MLP（注意力经过 MLP，12个头各一个）
- $C^{26-27}$: Compensation（补偿组件，捕获非线性协同效应）
- $C^{28}$: Bias（偏置项）

其中 $C^{1-25}$ 为**功能组件（Functional Component）**，每个都可表示为 $C = f(X) \cdot W$ 的形式，$f(X)$ 为非线性激活函数，$W$ 为与输入无关的参数矩阵（可视为训练学到的"记忆"）。

关键性质：这种分解是**无损的**，即 $LM_l(X) = \sum_{i=0}^{28} C^i$，能精确恢复原始模型输出。

#### Step 2: Pruning（剪枝）

基于分解后的计算图 $\mathcal{G} = \{\mathcal{C}, \mathcal{E}\}$，使用现有剪枝策略（默认 ACDC）进行 interchange ablation，剔除对输出 KL 散度影响小的组件。为支持后续因果分析，不取整个数据集的平均值，而是每 10 个样本为单位生成一组电路图集合。

#### Step 3: Post-pruning Causal Mediation（后剪枝因果中介）

这是排除混淆效应的关键步骤。将电路图中的路径分为三类效应：

- **Skill Effect**：目标技能对输出的贡献（论文关注重点）
- **Background Effect**：输入文本中其他潜在技能的影响（如 ICL 样本中混入的 induction 效应）
- **Self Effect**：仅由最后一个 token 的记忆产生的影响（类似 bi-gram 模型）

对每个输入样本 $s$ 构造两种扰动文本 $s_{\text{Bkg}}$（背景文本）和 $s_{\text{Self}}$（自身文本），分别剪枝得到三个电路图 $\mathcal{G}^*_{\text{Ori}}$、$\mathcal{G}^*_{\text{Bkg}}$、$\mathcal{G}^*_{\text{Self}}$。技能路径的判定标准：

$$\text{Eff}_{\text{Skill}}(\text{Path}^i) = \frac{N_{\text{Path}^i \in \mathcal{G}^*_{\text{Ori}},\; \text{Path}^i \notin \mathcal{G}^*_{\text{Bkg}},\; \text{Path}^i \notin \mathcal{G}^*_{\text{Self}}}}{N_{\text{all}}}$$

即：只在原始文本电路图中出现、而在背景和自身扰动电路图中均不出现的路径，才被认定为目标技能路径。最终以阈值 $\delta$（最优 0.6-0.7）筛选形成技能图 $\mathcal{G}^S$。

### 研究的三种技能

按复杂度递进：

1. **Previous Token Skill (PVT)**：从前一个 token 接收信息
2. **Induction Skill (IDT)**：识别前缀匹配模式并复制重复 token 序列（"A B ... A" → 预测 B）
3. **ICL Skill**：识别并复制 demonstration 中的模式进行推理

## 实验关键数据

### Path Ablation 验证

移除技能路径后，所有类型样本的准确率几乎降为 0（表明路径确实承载了目标技能）：

| 移除方式 | PVT准确率 | IDT准确率 | ICL1准确率 |
|---------|----------|----------|-----------|
| 完整电路 | 1.00 | 1.00 | 1.00 |
| 随机移除50条路径 | 0.46 | 0.58 | 0.61 |
| 随机移除500条路径 | 0.23 | 0.29 | 0.23 |
| 移除PVT技能路径 | **0.01** | 0.08 | 0.01 |
| 移除IDT技能路径 | 0.00 | **0.00** | 0.00 |
| 移除ICL技能路径 | — | — | **0.00** |

### 分层性（Stratification）定量结果

- PVT 关键接收器位于 1-2 层（最简单技能在最浅层）
- IDT 关键接收器位于 2-6 层（中等复杂度在中层）
- ICL 关键接收器分布在几乎所有层（最复杂技能跨越全部层）

### 包容性（Inclusiveness）定量结果

使用路径重叠率 $\text{ovlp}(A,B)$ 衡量，技能路径的包容关系显著：

| 对比 | 电路图重叠率 | 技能路径重叠率 |
|------|-----------|-------------|
| ovlp(IDT, PVT) | 0.19 | **0.74** |
| ovlp(ICL1, PVT) | 0.06 | **0.81** |
| ovlp(ICL1, IDT) | 0.17 | **0.63** |

传统电路图几乎看不出包容关系，而技能路径清晰展示了 74% 的 IDT 边存在于 PVT 中、81% 的 ICL1 边存在于 PVT 中。

### 跨剪枝策略鲁棒性

框架兼容 ACDC、E-pruning、EAP、DiscoGP、Scrubbing 等主流剪枝方法，所有组合下技能路径的包容性指标均显著高于电路图。

## 亮点

1. **理论贡献扎实**：提出 Transformer 计算图的完全无损线性分解，引入补偿组件解决 MLP 非线性耦合问题
2. **因果推断思路巧妙**：通过构造背景扰动和自身扰动文本，利用反事实和干预技术剥离混淆效应，是标准因果推断方法在机制可解释性中的优雅应用
3. **首次定量验证**：将"简单技能在浅层、复杂技能在深层"以及"复杂技能包含简单技能"从定性猜想提升为定量证据
4. **框架通用性**：三步框架与具体剪枝策略解耦，可即插即用替换不同的 circuit discovery 方法

## 局限与展望

1. **模型规模受限**：仅在 GPT2-small（12层12头）上验证，更大规模模型的适用性未知
2. **技能路径无法保持输出忠实度**：因为只包含技能效应（不含背景和自身效应），无法完整恢复原始输出，限制了传统电路评估指标的使用
3. **补偿组件的处理**：假设补偿组件的所有边始终存在，这是一个简化假设，可能遗漏部分信息
4. **技能定义依赖人工**：三种技能及其层级关系来源于先验知识，框架本身不能自动发现新技能
5. **扩展到更多技能**：目前只验证了三种渐进式技能，对于更复杂或并行的技能关系尚未探索

## 与相关工作的对比

| 维度 | 传统 Circuit Discovery | 本文 Skill Path |
|------|----------------------|----------------|
| 粒度 | 边/节点级电路图 | 路径级（线性组件链） |
| 目标 | 保持输出忠实度 | 隔离特定技能机制 |
| 混淆处理 | 无 | 因果中介去除背景/自身效应 |
| MLP分解 | 未完全线性化 | 引入补偿组件实现无损线性分解 |
| 技能覆盖 | 局部（特定输入的实例化） | 全局（完整技能路径） |
| 包容性检测 | 重叠率低（~0.17） | 重叠率高（~0.74） |

与 IOI 电路分析等工作的关键区别：现有方法发现的是特定输入样本下的电路实例（如某个 induction head），而本文发现的是跨大量样本的完整技能路径。

## 启发与关联

- **技能层级结构对模型压缩的启示**：如果复杂技能建立在简单技能之上，那么剪枝时需要优先保护底层技能路径
- **与 Probing 方法的互补**：Probing 探测"某层是否编码了某种信息"，而 Skill Path 揭示"信息如何在层间流动以实现某种技能"
- **模型编辑的精准定位**：技能路径提供了比电路图更精确的干预目标，有望用于知识编辑和技能选择性删除
- **可解释 scaling**：分层性和包容性是否在更大模型中仍然成立？这对理解 emergent abilities 有重要意义

## 评分
- 新颖性: ⭐⭐⭐⭐ （Skill Path 概念和三步框架均为首创）
- 实验充分度: ⭐⭐⭐⭐ （消融、可视化、跨策略对比全面，但限于 GPT2-small）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，图表丰富，但公式密度较高）
- 价值: ⭐⭐⭐⭐ （首次定量验证技能分层/包容猜想，对 mechanistic interpretability 领域有推动意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation](enhancing_logical_expressiveness_in_graph_neural_networks_via_path-neighbor_aggr.md)
- [\[AAAI 2026\] MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training](mug_meta-path-aware_universal_heterogeneous_graph_pre-training.md)
- [\[CVPR 2026\] R2G: A Multi-View Circuit Graph Benchmark Suite from RTL to GDSII](../../CVPR2026/graph_learning/r2g_multi_view_circuit_graph_benchmark_suite_from_rtl_to_gdsii.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](../../NeurIPS2025/graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[AAAI 2026\] RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)

</div>

<!-- RELATED:END -->
