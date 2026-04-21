---
title: >-
  [论文解读] The Expressive Limits of Diagonal SSMs for State-Tracking
description: >-
  [ICLR 2026][视频理解][状态空间模型] 建立了输入依赖复数对角（DCD）SSM 在群状态追踪任务上的完整表达能力刻画：单层不能追踪任何非阿贝尔群，$k$ 层能追踪群 $G$ 当且仅当 $G$ 存在长度为 $k$ 的子正规链且因子均为阿贝尔群——精确定义了深度对表达能力的严格提升，同时实验揭示表达能力与可学习性之间的显著 gap。
tags:
  - ICLR 2026
  - 视频理解
  - 状态空间模型
  - 表达能力
  - 群状态追踪
  - 对角SSM
  - 可解群
  - Mamba
---

# The Expressive Limits of Diagonal SSMs for State-Tracking

**会议**: ICLR 2026  
**arXiv**: [2603.01959](https://arxiv.org/abs/2603.01959)  
**领域**: 序列建模 / 理论  
**关键词**: 状态空间模型, 表达能力, 群状态追踪, 对角SSM, 可解群, Mamba

## 一句话总结

建立了输入依赖复数对角（DCD）SSM 在群状态追踪任务上的完整表达能力刻画：单层不能追踪任何非阿贝尔群，$k$ 层能追踪群 $G$ 当且仅当 $G$ 存在长度为 $k$ 的子正规链且因子均为阿贝尔群——精确定义了深度对表达能力的严格提升，同时实验揭示表达能力与可学习性之间的显著 gap。

## 研究背景与动机

**领域现状**：状态空间模型（SSM），如 Mamba、S4D 等，已作为 Transformer 的高效替代品在长序列建模任务上取得了强大的经验性能。它们通过线性递归和对角化状态转移矩阵实现了 $O(\log T)$ 的并行化。然而，SSM 的理论表达能力理解仍然有限，特别是在状态追踪（state-tracking）这类任务上。

**现有痛点**：
1. SSM最初被期望在状态追踪上优于Transformer（因为有显式状态表示），但实验表明SSM同样失败
2. Merrill et al. 证明SSM属于 $\mathsf{TC}^0$ 复杂性类，推测不能追踪非可解群（如 $S_5$），但这依赖于复杂性理论中的未解猜想
3. Sarrof et al. 证明 Mamba（非负对角矩阵）不能解决奇偶问题（最简单的非平凡群 $C_2$），但对角 SSM 的精确能力边界仍然未知
4. 单层 vs 多层的表达能力差异在理论上不清楚——深度是否带来严格益处？

**核心矛盾**：对角化是 SSM 高效并行化的基础，但也限制了状态转移矩阵的表达能力。如何在保持效率的同时理解其表达极限？

**本文方案**：从代数群论的视角出发，精确刻画 $k$ 层 DCD SSM 能追踪哪些群——给出不依赖任何猜想的充要条件。

## 方法详解

### 整体框架

本文研究的对象是输入依赖复数对角（DCD）SSM，其状态递归为：

$$h_t = A(x_t) h_{t-1} + b(x_t)$$

其中 $A(x_t) \in \mathbb{C}^{d \times d}$ 是对角矩阵，$b(x_t) \in \mathbb{C}^d$ 是输入向量。输出通过解码器 $y_t = \text{dec}(h_t, x_t)$ 产生。假设 $A$、$b$、$\text{dec}$ 均为通用函数逼近器。

群状态追踪任务定义：给定群 $G$ 和输入序列 $x_1, \ldots, x_T \in G$，输出 $y_t = x_1 \cdot x_2 \cdots x_t$，即累积群乘积。

### 关键设计1：单层 DCD SSM 的不可能性证明

**核心定理（Theorem 1）**：单层 DCD SSM 在有限精度下能追踪群 $G$ 当且仅当 $G$ 是阿贝尔群。

证明思路分三步：

1. **消除无用坐标（Lemma 1）**：如果某个状态坐标在某输入下收缩（$|\lambda(x)_j| < 1$）、扩张（$|\lambda(x)_j| > 1$）或漂移（$|\lambda(x)_j| = 1$ 且 $b(x)_j \neq 0$），则可以构造另一个SSM将该坐标设为固定值而不影响追踪能力。

2. **旋转中心一致性（Lemma 2）**：中性旋转 $h \mapsto \lambda(h - c_1) + c_1$ 和共轭旋转 $h \mapsto \lambda^*(h - c_2) + c_2$ 的复合（当 $c_1 \neq c_2$ 时）是一个非零平移——这会导致状态发散。

3. **收敛性要求（Lemma 3）**：对于有限群追踪，所有状态坐标必须共享同一旋转中心。结合对角矩阵的交换性，状态更新必然是可交换的，即群必须是阿贝尔群。

### 关键设计2：多层 DCD SSM 的充要条件

**核心定理（Theorem 2）**：$k$ 层 DCD SSM 在有限精度下能追踪群 $G$ 当且仅当 $G$ 存在长度为 $k$ 的子正规链：

$$\{e\} = G_0 \trianglelefteq G_1 \trianglelefteq \cdots \trianglelefteq G_k = G$$

且每个因子 $G_{i+1}/G_i$ 是阿贝尔群。这恰好是可解群的标准定义——但精细化为"需要**恰好** $k$ 层"。

**构造性证明**（充分性）：以 $S_3$ 为例，将其分解为半直积 $C_3 \rtimes C_2$。第一层追踪 $C_2$（奇偶性），第二层以第一层状态为条件追踪 $C_3$（模3旋转方向取决于奇偶状态）：

- 第一层 AUSSM：$\Lambda((0,\beta)) = 0$，$\Lambda((1,\beta)) = \pi$
- 第二层 AUSSM：$\Lambda^{(2)}((1,\alpha,\beta)) = \frac{2\pi}{3}\beta$，$\Lambda^{(2)}((-1,\alpha,\beta)) = -\frac{2\pi}{3}\beta$

### 关键设计3：与现有 SSM 变体的关系

| 模型 | 转移矩阵 $A(x)$ | 特征值范围 | 能追踪的群 |
|------|-----------------|-----------|-----------|
| S4/S4D | $\Lambda$（固定对角） | $\mathbb{C}$ | 无（时间不变） |
| Mamba | $\exp(\Delta(x) \odot \Lambda)$ | $(0, 1]$（非负实数） | 无群（不含负值） |
| Negative Mamba | $2\exp(\Delta(x) \odot \Lambda) - I$ | $(-1, 1]$ | 仅 $C_2$（单层） |
| AUSSM | $\exp(i\Delta(x) \odot \Lambda(x))$ | 单位圆 $|z|=1$ | 所有阿贝尔群（单层） |

## 实验关键数据

### 主实验：状态追踪任务（最长泛化序列长度，训练长度≤60）

| 群       | 阿贝尔 | 可解 | Mamba | Neg Mamba | Simple AUSSM | AUSSM | RNN  |
|----------|--------|------|-------|-----------|-------------|-------|------|
| $C_2$    | ✓      | ✓    | ✘     | 1000      | 160         | 1000  | 1000 |
| $C_6$    | ✓      | ✓    | ✘     | ✘         | 240         | 940   | 1000 |
| $C_{24}$ | ✓      | ✓    | ✘     | ✘         | 240         | 260   | 1000 |
| $C_{60}$ | ✓      | ✓    | ✘     | ✘         | 300         | 240   | ✘    |
| $S_3$    | ✗      | ✓    | ✘     | ✘         | ✘           | ✘     | 1000 |
| $A_4$    | ✗      | ✓    | ✘     | ✘         | ✘           | ✘     | 1000 |
| $A_5$    | ✗      | ✗    | ✘     | ✘         | ✘           | ✘     | ✘    |

> ✘ 表示模型无法泛化到长度≥100的序列。报告的是准确率>90%的最长序列长度。

### 消融实验：单层 vs 双层

| 群              | 模型           | 单层  | 双层  | 理论预期 |
|-----------------|---------------|-------|-------|----------|
| $C_2$           | AUSSM         | 1000  | 200   | 均可     |
| $C_2 \times C_4$| Neg Mamba     | ✘     | 360   | 双层可   |
| $S_3$           | AUSSM         | ✘     | ✘     | 双层可（gap!） |
| $A_4$           | AUSSM         | ✘     | ✘     | 双层可（gap!） |

### 核心发现

- **阿贝尔群**：AUSSM 和 Simple AUSSM 的单层理论掌控所有阿贝尔群，实验中梯度优化成功找到泛化解
- **非阿贝尔群**：双层 AUSSM 理论上可追踪 $S_3$ 和 $A_4$，但实验中标准训练**从未成功**——表达能力 ≠ 可学习性
- **初始化敏感性**：将 AUSSM 初始化到 $S_3$ 解析解附近后，训练成功并泛化到4倍长度——解存在但 loss landscape 中难以找到
- **RNN 无理论限制**：RNN 可追踪所有可解群，但在大群（$C_{60}$）上也遇到优化困难

## 亮点与洞察

- **精确的代数刻画**：给出充要条件而非"大概能/不能"——$k$ 层严格对应子正规链长度为 $k$ 的群
- **深度的严格益处**：$k$ 层严格 > $(k-1)$ 层——非阿贝尔群需要深度，且需要的层数有精确的群论含义
- **表达 ≠ 学习的重要区分**：理论可表达但 SGD 学不到——提醒社区不能仅看理论表达力，优化瓶颈同样关键
- **对 Mamba 设计的直接指导**：Mamba 的非负特征值约束是根本限制；允许负值/复值特征值是解锁更强追踪能力的必要条件

## 局限与展望

- 结果限于对角 SSM，分块对角（如 $2 \times 2$）可能将表达力扩展到 $\mathsf{NC}^1$
- 可学习性 gap 的根本原因未解——loss landscape 的几何结构（平坦区域/鞍点/孤立极小值）需要进一步研究
- 目前仅讨论群状态追踪，实际序列任务更复杂——理论结论对实际任务的预测力有待验证
- 未探索混合架构（如 AUSSM+Mamba 交替层）在非阿贝尔群上的可学习性
- 扩展到半群和幺半群的讨论仅为初步，完整理论有待建立

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [GG-SSMs: Graph-Generating State Space Models](../../CVPR2025/video_understanding/gg-ssms_graph-generating_state_space_models.md)
- [Mamba-3: Improved Sequence Modeling using State Space Principles](mamba-3_improved_sequence_modeling_using_state_space_principles.md)
- [Parity Requires Unified Input Dependence and Negative Eigenvalues in SSMs](../../ICML2025/video_understanding/parity_requires_unified_input_dependence_and_negative_eigenvalues_in_ssms.md)
- [Fixed-Point RNNs: Interpolating from Diagonal to Dense](../../NeurIPS2025/video_understanding/fixed-point_rnns_interpolating_from_diagonal_to_dense.md)
- [MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](../../CVPR2025/video_understanding/mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)

<!-- RELATED:END -->
