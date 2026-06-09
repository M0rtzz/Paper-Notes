---
title: >-
  [论文解读] On the Expressive Power of GNNs for Boolean Satisfiability
description: >-
  [ICLR 2026][图学习][图神经网络表达力] 从 Weisfeiler-Leman (WL) 测试角度严格证明了完整的 WL 层级无法区分可满足与不可满足的 3-SAT 实例，揭示了 GNN 用于 SAT 求解的理论表达力极限，同时识别出平面 SAT 和随机 SAT 等 GNN 可成功区分的正面实例族…
tags:
  - "ICLR 2026"
  - "图学习"
  - "图神经网络表达力"
  - "布尔可满足性"
  - "Weisfeiler-Leman 测试"
  - "SAT 求解"
  - "理论分析"
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

**1. LCN 图表示：把否定连接补回来，否则图本身就丢信息**

学习式求解器普遍把 CNF 公式画成 LCG——文字与子句之间的二部图。本文指出这个表示其实是有漏洞的：一旦去掉节点标签来保持 GNN 需要的排列不变性，仅凭 LCG 就无法分辨一个文字和它的否定，信息从源头就丢了。补救办法是显式加入文字与其否定之间的边，得到 LCN（Literal-Clause Graph with Negation connections）。加上这层边后表示变得充分——无标签的 LCN 在同构意义下能唯一确定原 SAT 公式，后续所有关于"WL 能否区分两个公式"的讨论才站得住脚。

**2. 主定理（定理 5.3）：存在 WL 永远分不开的"一可满足一不可满足"实例对**

这是全文的负面核心。要证明 WL 表达力不够，就得造出两个公式：一个可满足、一个不可满足，但它们的 LCN 在任意阶 WL 下都等价。构造借用了 Cai-Fürer-Immerman (1992) 的经典手法：给定图 $G$，先写公式 $f_G$ 编码"$G$ 是否存在每个节点出度为偶数的定向"，这个问题的答案只取决于边数奇偶性这一全局量；再造一个"扭曲"版本 $\tilde{f}_G$，把其中一条边改成双向，于是 $f_G$ 与 $\tilde{f}_G$ 恰好一个可满足、一个不可满足。关键在于这点扭曲是局部不可见的，二者的 LCN 在 $n$-WL 下仍然不可区分：

$$\exists\, f, \tilde{f} \text{ (3-SAT)}: |f| = O(n) \text{ 变量}, f \text{ 可满足}, \tilde{f} \text{ 不可满足}, \text{LCN}(f) \equiv_{n\text{-WL}} \text{LCN}(\tilde{f})$$

这与 Tseitin 公式（resolution 的经典难实例）同出一脉——全局的不一致性无法靠局部消息传递察觉，正是 WL 失效的根源。

**3. 对顺序求解器的连锁影响（引理 5.4 / 推论 5.5）：边解边判也救不回来**

很多 GNN-SAT 求解器（如 QuerySAT）不是一次出答案，而是逐步设置变量、边解边判。一个自然的指望是：固定掉一部分变量后，残余公式或许就能被 WL 区分了。引理 5.4 否掉了这个指望——即便已经设置了 $\lfloor k/2 \rfloor - 1$ 个变量，只要两个公式原本在 $k$-WL 下不可区分，它们的残余公式依然不可区分。推论 5.5 把结论推到极致：哪怕设置了 $\Theta(n)$ 个变量，WL-powerful 的 GNN 仍分不开可满足与不可满足的残余公式。这意味着靠"试探—回退—重启"的学习策略也无法绕过表达力天花板。

**4. 3-正则 SAT：一个 WL 彻底失效却仍是 NP 完全的家族**

定义 $k$-正则 SAT：每个文字恰好出现在 $k$ 个子句中、每个子句恰含 $k$ 个文字。这种高度对称的结构对 WL 是毁灭性的——定理 5.1 证明 3-正则 SAT 仍是 NP-完全的，而观察 5.2 指出 WL 测试无法区分任何两个变量数相同的 3-正则 SAT 公式（所有节点局部邻域长得一模一样，消息传递得不到任何区分信号）。这给出了一整族"难且 WL 看不见"的实例，比单个构造更有说服力。

**5. 正面结果：哪些 SAT 家族 WL 反而能分得开**

负面结果之外，本文也划出 GNN 仍然有戏的区域，正面回击"NP-hard 就一定 WL 不可区分"的直觉。平面 SAT（定理 6.1）：因为平面图能被 4-WL 完全识别，所有平面 SAT 实例都可被 4-WL 区分——而平面 SAT 本身仍是 NP 完全的，说明表达力与计算复杂度确实正交。随机 SAT（定理 6.3）：从均匀随机文字关联图提取的 CNF 公式，以至少 $1 - n^{-1/7}$ 的概率能被 WL 识别，这也解释了后面实验里随机实例几乎对 WL 毫无难度的现象。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](../../ICML2026/graph_learning/full-spectrum_graph_neural_network_expressive_and_scalable.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids](../../NeurIPS2025/graph_learning/graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)
- [\[ICLR 2026\] Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)
- [\[NeurIPS 2025\] The Underappreciated Power of Vision Models for Graph Structural Understanding](../../NeurIPS2025/graph_learning/the_underappreciated_power_of_vision_models_for_graph_structural_understanding.md)
- [\[AAAI 2026\] Logical Characterizations of GNNs with Mean Aggregation](../../AAAI2026/graph_learning/logical_characterizations_of_gnns_with_mean_aggregation.md)

</div>

<!-- RELATED:END -->
