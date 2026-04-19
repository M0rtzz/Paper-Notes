---
title: >-
  [论文解读] On the Expressive Power of GNNs for Boolean Satisfiability
description: >-
  [ICLR 2026][图学习][图神经网络表达力] 从 Weisfeiler-Leman (WL) 测试角度严格证明了完整的 WL 层级无法区分可满足与不可满足的 3-SAT 实例，揭示了 GNN 用于 SAT 求解的理论表达力极限，同时识别出平面 SAT 和随机 SAT 等 GNN 可成功区分的正面实例族。
tags:
  - ICLR 2026
  - 图学习
  - 图神经网络表达力
  - 布尔可满足性
  - Weisfeiler-Leman 测试
  - SAT 求解
  - 理论分析
---

# On the Expressive Power of GNNs for Boolean Satisfiability

**会议**: ICLR 2026  
**arXiv**: [2602.08745](https://arxiv.org/abs/2602.08745)  
**代码**: [GitHub](https://github.com/sakupeltonen/sat-expressivity)  
**领域**: 图学习  
**关键词**: 图神经网络表达力, 布尔可满足性, Weisfeiler-Leman 测试, SAT 求解, 理论分析

## 一句话总结

从 Weisfeiler-Leman (WL) 测试角度严格证明了完整的 WL 层级无法区分可满足与不可满足的 3-SAT 实例，揭示了 GNN 用于 SAT 求解的理论表达力极限，同时识别出平面 SAT 和随机 SAT 等 GNN 可成功区分的正面实例族。

## 研究背景与动机

布尔可满足性问题（SAT）是经典 NP 完全问题。近年来 GNN 成为学习式 SAT 求解的主流架构（如 NeuroSAT、QuerySAT），因为 CNF 公式天然表示为文字-子句二部图（LCG）。然而 GNN 表达力受 WL 测试约束——WL 等价的图输入必然得到相同输出。

**核心问题**：GNN 是否有足够的表达力来推理可满足性？

现有工作缺乏从表达力角度对 GNN 在 SAT 任务上的系统分析。尽管有人可能直觉认为"SAT 是 NP-hard，所以 GNN 必然不够强"，但表达力与计算复杂度是正交概念——例如平面 SAT 也是 NP 完全的，但 4-WL 可以识别所有平面图。

## 方法详解

### 整体框架

本文是理论驱动的工作，核心贡献是一系列关于 WL 层级与 SAT 公式可区分性的定理和构造，辅以实验验证。研究路线：

1. 定义 SAT 公式的图表示（LCN: Literal-Clause Graph with Negation connections）
2. 构造 WL 不可区分的 SAT 实例族
3. 分析特殊 SAT 家族（正则、平面、随机）的可区分性
4. 在随机实例和竞赛实例上验证 WL 表达力

### 关键设计

#### 图表示：LCN（文字-子句图 + 否定连接）

标准的 LCG 是文字与子句之间的二部图。本文强调必须加入文字与其否定之间的边（称为 LCN），因为：
- 去掉节点标签后（保持排列不变性），仅靠 LCG 会丢失信息
- LCN 表示是充分的：无标签 LCN 可唯一确定 SAT 公式（同构意义下）

#### 定理 5.3（主定理）：k-WL 不可区分的 SAT 实例

**构造思路**：基于 Cai-Fürer-Immerman (1992) 的经典构造。

给定图 $G$，构造公式 $f_G$ 编码"$G$ 的边是否存在偶数定向"（每个节点出度为偶数的定向）。此问题的答案取决于边数奇偶性。然后构造"扭曲"公式 $\tilde{f}_G$：其中一条边变为双向。这样 $f_G$ 和 $\tilde{f}_G$ 恰好一个可满足、一个不可满足，但对应的 LCN 在 $n$-WL 下不可区分。

$$\exists\, f, \tilde{f} \text{ (3-SAT)}: |f| = O(n) \text{ 变量}, f \text{ 可满足}, \tilde{f} \text{ 不可满足}, \text{LCN}(f) \equiv_{n\text{-WL}} \text{LCN}(\tilde{f})$$

这一构造与 Tseitin 公式（resolution 难实例）高度相关——两者都涉及全局不一致性无法被局部推理检测出来。

#### 对顺序求解器的影响（引理 5.4）

很多 GNN-SAT 求解器（如 QuerySAT）通过逐步设置变量来求解。引理 5.4 表明：即使已设置 $\lfloor k/2 \rfloor - 1$ 个变量，如果两个公式在 $k$-WL 下不可区分，剩余公式仍然不可区分。

**推论 5.5**：WL-powerful GNN 即使在设置了 $\Theta(n)$ 个变量后，仍然无法区分可满足与不可满足的残余公式——这对重启策略学习有重要影响。

#### 3-正则 SAT：WL 完全无用的 NP 完全问题

定义 $k$-正则 SAT：每个文字恰好出现在 $k$ 个子句中，每个子句含 $k$ 个文字。

- **定理 5.1**：3-正则 SAT 是 NP-完全的
- **观察 5.2**：WL 测试无法区分任何两个同变量数的 3-正则 SAT 公式

#### 正面结果：可区分的 SAT 家族

**平面 SAT（定理 6.1）**：4-WL 可以区分所有平面 SAT 实例（因为平面图被 4-WL 完全识别）

**随机 SAT（定理 6.3）**：从均匀随机文字关联图提取的 CNF 公式，以至少 $1 - n^{-1/7}$ 的概率被 WL 识别

### 损失函数 / 训练策略

本文是纯理论工作，不涉及模型训练。实验部分通过以下方式验证 WL 表达力：
- 对可满足公式运行 $r$ 轮 WL，对等价类中的文字加等值约束
- 检查加约束后的公式 $f_r$ 是否仍可满足
- $r_{\text{crit}}$ 为最小的使 $f_r$ 可满足的轮数

## 实验关键数据

### 主实验：随机实例（G4SAT 基准）

| 家族 | 难度 | $r_{\text{crit}}$ | $r_{\text{converge}}$ | 变量数 | 数量 |
|------|------|---------|-------------|--------|------|
| 3-SAT | easy | 2.97±0.18 | 3.68±0.47 | 26±9 | 1000 |
| 3-SAT | medium | 3.00±0.04 | 3.92±0.28 | 119±47 | 1000 |
| 3-SAT | hard++ | 3.08±0.28 | 4.00±0.00 | 5001±62 | 25 |
| k-clique | easy | 4.12±0.73 | 6.26±0.83 | 33±13 | 960 |

随机实例通常只需 3-4 轮 WL 即可区分文字，约 40% 的公式在收敛时所有文字已获得唯一标识。

### SAT 竞赛实例

| 家族 | $r_{\text{crit}}$ | $r_{\text{converge}}$ | 变量数 | 数量 |
|------|---------|-------------|--------|------|
| argumentation | 2.94±0.44 | 4.31±0.87 | 1266±625 | 16 |
| circuit-multiplier | **unsat** | 7.18±0.40 | 1075±50 | 11 |
| cryptography | 15.74±14.67 | 17.63±14.34 | 41510±29705 | 19 |
| heule-nol | **unsat** | 8.60±1.26 | 1419±0 | 10 |
| maxsat-optimum | 26.64±3.64 | 29.27±2.49 | 22157±5623 | 11 |

448 个竞赛实例中仅 234 个可用 WL 求解；69 个家族中 38 个包含 WL 无法区分的实例。

### 消融实验

**3-SAT 的 WL 收敛模式分析**：
- 第 1 轮：每个文字看到自己的度数
- 第 2 轮：无法细分（因为所有邻居子句度数相同 = 3）
- 第 3 轮：文字观察到共享子句中其他文字的度数，实现有效区分
- 第 4 轮：WL 通常收敛

### 关键发现

1. **随机 vs 工业实例差异巨大**：随机实例对 WL 几乎无难度，但工业实例经常超出 WL 表达力
2. **正则结构是致命的**：heule-nol 家族编码网格着色问题，正则结构使 WL 完全失效
3. **与 resolution 复杂度的联系**：本文构造的不可区分实例与 Tseitin 公式（resolution 难实例）结构相似，局部推理无法检测全局不一致性
4. k-clique 和 k-vercov 家族有少量实例（2.0% 和 0.3%）WL 无法求解，源于底层图的对称性

## 亮点与洞察

- **里程碑式的理论结果**：首次证明完整 WL 层级对 SAT 的表达力极限，澄清了"NP-hard = WL 不可区分"的常见误解（定理 6.1 反例）
- **Tseitin-CFI 联系**：发现 Cai-Fürer-Immerman 构造与 Tseitin 公式的深层联系，为 WL 测试和证明复杂度之间建立了新桥梁
- **实践指导清晰**：GNN 适合处理随机 SAT，但对工业 SAT 需要超越 WL 的架构（如对称性破缺、高阶 GNN）

## 局限与展望

1. 理论结果基于最坏情况构造，实际中稀疏/结构化实例可能有更好行为
2. 实验仅测试 WL 等级的表达力是否充分，未考虑泛化性从数据学习的能力
3. 未探索具体的超越 WL 的架构方案（如 k-GNN、随机特征增强）在 SAT 上的效果
4. 工业实例的实验受限于 10MB 大小限制，更大规模实例的行为未知

## 相关工作与启发

- **NeuroSAT / QuerySAT / SATformer**：端到端学习式 SAT 求解器，均受本文理论限制
- **Cai-Fürer-Immerman (1992)**：本文将其经典不可区分图构造推广到 SAT 领域
- **GNN 表达力文献**（GIN, k-GNN, 高阶 WL）：本文为这些工作在 SAT 应用上提供了具体的正/负面结果
- 对未来学习式 SAT 求解器设计有重要启示：需要针对工业实例的结构特性设计专门的表达力增强策略

## 评分

- 新颖性：★★★★★ — 首个系统性的 GNN-SAT 表达力理论分析
- 技术深度：★★★★★ — 证明严格、构造巧妙，与代数/复杂性理论紧密联系
- 实验充分度：★★★★☆ — 随机 + 竞赛实例覆盖全面，验证了理论预测
- 写作质量：★★★★☆ — 结构清晰，理论与实践交织良好

<!-- RELATED:START -->

## 相关论文

- [Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids](../../NeurIPS2025/graph_learning/graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)
- [Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)
- [The Underappreciated Power of Vision Models for Graph Structural Understanding](../../NeurIPS2025/graph_learning/the_underappreciated_power_of_vision_models_for_graph_structural_understanding.md)
- [Logical Characterizations of GNNs with Mean Aggregation](../../AAAI2026/graph_learning/logical_characterizations_of_gnns_with_mean_aggregation.md)
- [Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation](../../AAAI2026/graph_learning/enhancing_logical_expressiveness_in_graph_neural_networks_via_path-neighbor_aggr.md)

<!-- RELATED:END -->
