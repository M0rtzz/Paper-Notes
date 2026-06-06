---
title: >-
  [论文解读] Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models
description: >-
  [ICML 2026][时间序列][连续时间动态图] CTDG-SSM 首次通过**拓扑感知 HiPPO 投影**和状态空间模型，同时捕捉动态图中的多跳长距离空间依赖（LRS）和长距离时间依赖（LRT），在链接预测 / 节点分类等任务上超越 SOTA 且参数量仅为竞争方法的 1/10。
tags:
  - "ICML 2026"
  - "时间序列"
  - "连续时间动态图"
  - "状态空间模型"
  - "长距离依赖"
  - "时空表示学习"
---

# Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models

**会议**: ICML 2026  
**arXiv**: [2606.04672](https://arxiv.org/abs/2606.04672)  
**代码**: 待确认  
**领域**: 时间序列 / 图学习 / 动态图  
**关键词**: 连续时间动态图, 状态空间模型, 长距离依赖, 时空表示学习

## 一句话总结
CTDG-SSM 首次通过**拓扑感知 HiPPO 投影**和状态空间模型，同时捕捉动态图中的多跳长距离空间依赖（LRS）和长距离时间依赖（LRT），在链接预测 / 节点分类等任务上超越 SOTA 且参数量仅为竞争方法的 1/10。

## 研究背景与动机

**领域现状**：连续时间动态图（CTDG）提供了建模演化关系数据的强大框架。现有方法主要分两类——事件驱动模型（TGAT、TGN）计算高效但难保留长时间尺度的历史信息（LRT 能力弱）；序列模型变体（DyGFormer、DyGmamba）能捕捉 LRT 但预处理时把注意力限制在 1-hop 邻域，丧失多跳全局空间结构信息（LRS 能力弱）。

**现有痛点**：没有现有方法能同时保持 LRS 和 LRT——这在金融欺诈检测等实际应用中很关键（洗钱通常通过长交易链传播而非孤立的局部交互）。

**核心矛盾**："空间-时间权衡"困境——要么为了捕捉 LRT 而打破图结构，要么为了利用图结构而限制时间感受野。

**本文目标**：开发统一的时空状态空间框架，在同一框架内既能压缩历史事件信息到紧凑内存（LRT），又能通过图多项式滤波器聚合多跳邻域信息（LRS）。

**切入角度**：从古典 HiPPO（High-order Polynomial Projection Operator）扩展到图数据。关键观察是可通过投影古典 HiPPO 系数到拉普拉斯矩阵多项式的逆，得到同时编码时间与拓扑动态的状态空间模型。

**核心 idea**：拓扑感知的高阶多项式投影（CTT-HiPPO）替代简单的序列内存机制——内存系数既受时间演化影响也受图结构约束；通过零阶保持离散化实现高效计算。

## 方法详解

### 整体框架
1. 输入：连续时间事件流 $\{(u, v, t_i)\}$。
2. 子图采样：每批次构造批级拉普拉斯矩阵 $L_B[k]$，采样每个节点 $N_u$ 个最近邻。
3. 节点特征编码：节点静态嵌入 + 动态邻域特征 + 边属性 + 时间编码，通过 2 层编码器投影到 $d$ 维隐空间。
4. CTDG-SSM 层：多层堆积，每层 RMSNorm → CTDG-SSM 递推 → GeLU → 残差连接。
5. 输出：最后一层隐状态 + 静态嵌入聚合，通过 MLP 解码器生成链接预测分数或节点分类概率。

### 关键设计

1. **拓扑感知高阶多项式投影（CTT-HiPPO）**:

    - 功能：为 CTDG 设计内存压缩机制，联合编码时间与图拓扑信息。
    - 核心思路：在时间窗口 $[0, \tau]$ 上把 $i$ 维节点特征建模为 $X_{:, i}(t) = p(L_\tau) H_\tau^{(i)} g(t) + r_i(t)$，$g(t)$ 是正交多项式基，$p(L_\tau)$ 是拉普拉斯矩阵多项式（图滤波器）。一阶最优性条件给出 $H_\tau^{(i)} = p(L_\tau)^{-1} H_\tau^{(i), \text{HiPPO}}$——古典 HiPPO 系数通过逆多项式滤波器的投影。
    - 设计动机：既继承 HiPPO 对时间序列的最优压缩特性，又通过 $p(L_\tau)$ 注入图拓扑约束；$K$ 阶多项式滤波器可聚合 $K$ 跳邻域，自动实现多跳聚合；图结构演变时拉普拉斯矩阵随时间变化，滤波器动态适应。

2. **连续时间状态空间模型（CTDG-SSM）**:

    - 功能：对 CTT-HiPPO 系数矩阵 $H_s$ 的时间演化建模。
    - 核心思路：证明内存系数演化满足微分方程 $\frac{d H_s}{d s} = -\frac{H_s A^\top}{M(s)} - p(L_s)^{-1} \frac{d p(L_s)}{d s} H_s + \frac{p(L_s)^{-1} X(s) B^\top}{M(s)}$。第一项时间记忆衰减，第二项图拓扑变化的修正，第三项整合新观测。零阶保持（ZOH）离散化得递推 $H[k+1] = \bar{A}_{L[k]} H[k] \bar{A} + \bar{B}(L[k], X[k])$。
    - 设计动机：统一将图拓扑变化与时间演化融入单个微分方程；当 $p(L_\tau) = I$（无图）时退化为经典 SSM，当图结构固定时退化为分段常数 SSM。

3. **高效离散实现 + 鲁棒性保证**:

    - 功能：实现可扩展高效的训练与推理；理论上保证图扰动下的稳定性和节点排列等变性。
    - 核心思路：批次级子图采样避免稠密图 Laplacian 计算，只需 $N_B \times N_B$ 操作。残差连接 + RMSNorm 借鉴 Mamba 等现代架构。证明拉普拉斯矩阵受扰动 $\|\Delta L\|_2 \leq \epsilon$ 时 CTT-HiPPO 系数相对误差以 $\epsilon$ 一阶线性界，并保证节点排列等变性。
    - 设计动机：避免全图运算开销，通过邻域采样既保留多跳信息又控制计算复杂度；理论保证使方法在图噪声下稳定。

## 实验关键数据

### 主实验（动态链接预测，AUC ROC）

| 数据集 | JODIE | TGN | DyGmamba | **CTDG-SSM** |
|--------|-------|-----|----------|-------------|
| LastFM | 70.89 | 76.64 | 93.31 | **93.79** |
| Enron | 87.77 | 88.72 | 93.34 | **94.98** |
| MOOC | 84.50 | 91.91 | 89.58 | **99.00** |
| Reddit | 98.29 | 98.61 | 99.27 | **99.48** |
| **Avg. Rank** | 7.93 | 4.57 | 2.00 | **1.86** |

CTDG-SSM 在 LRT 基准（LastFM、MOOC、Enron）上显著领先，MOOC 上相对 DyGmamba 提升 9.4%。

### 消融实验（序列分类，长距离依赖测试）

| 变体 | n=3 | n=9 | n=15 | n=20 | 平均 |
|------|-----|-----|------|------|------|
| TU-SSM（无拓扑项） | 47.0 | 50.7 | 52.3 | 54.5 | 51.1% |
| CTDG-SSM (FO，1 阶) | 100.0 | 97.1 | 97.4 | 97.1 | 97.9% |
| **CTDG-SSM (SO，2 阶)** | **100.0** | **98.1** | **97.8** | **98.6** | **98.6%** |

### 效率对比

| 指标 | CTDG-SSM | DyGmamba | DyGFormer |
|------|----------|----------|-----------|
| 参数量（相对） | 1× | ~10× | ~8× |
| LastFM 训练时间 / epoch | 4.45 分 | 28.45 分 | 47.00 分 |
| GPU 内存 | 1.15 GB | 4.17 GB | 7.57 GB |

### 关键发现
- 去掉拓扑项（TU-SSM）性能从 98% 崩到 51%，证实结构化记忆更新的关键作用。
- 二阶多项式相比一阶在长序列上显著改进（n=20 从 97.13% → 98.60%）。
- 参数量是竞争方法 1/10，训练速度 6.4× 快，GPU 内存 3.6× 少。

## 亮点与洞察
- **理论深度**：从古典 HiPPO 推导拓扑感知变体，巧妙将图滤波注入时间记忆；推导比直接设计更有原理性。
- **空间-时间统一框架**：而非"先时间后空间"流水线，通过微分方程自然耦合两者于内存动态中。
- **参数高效性**：仅靠多项式滤波系数 + 状态转移矩阵达 SOTA，参数量竞争方法 1/10，在模型压缩和边缘部署上有实际价值。
- **可迁移设计**：CTT-HiPPO 思路（通过逆滤波器投影）可推广到其他需要联合时空建模的任务。

## 局限与展望
- 子图采样策略固定（$N_u$ 最近邻），未来可探索学习式采样。
- 零阶保持假设对高频图变化可能不精确，可尝试更高阶离散化。
- 可解释性缺失——对"学到的滤波器在表达什么"缺少可视化分析。
- 未涵盖事件时间间隔差异极大的场景（传感器数据稀疏 + 突发）。

## 相关工作与启发
- **vs DyGmamba**：后者只捕捉 LRT，结构受限；CTDG-SSM 通过拓扑项同时捕捉 LRS。
- **vs GraphSSM**：面向离散图、假设固定结构；CTDG-SSM 适配连续事件流与动态拓扑。
- **vs Transformer 变体（DyGFormer）**：用注意力但限制 1-hop；CTDG-SSM 用图滤波器隐式聚合多跳，计算量更低。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  首次从状态空间视角统一 LRT 与 LRS，理论推导严谨。
- 实验充分度: ⭐⭐⭐⭐  三类任务充分验证，消融深入，效率对比全面；可解释性和极端场景测试还有余地。
- 写作质量: ⭐⭐⭐⭐⭐  逻辑清晰，数学严格，问题动机 → 理论 → 实验完整闭环。
- 价值: ⭐⭐⭐⭐⭐  解决重要 CTDG 建模问题，参数高效率对实际部署有价值，理论可迁移性强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences](fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an.md)
- [\[ICML 2026\] HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models](hippo_zoo_explicit_memory_mechanisms_for_interpretable_state_space_models.md)
- [\[AAAI 2026\] LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs](../../AAAI2026/time_series/loretta_a_low_resource_framework_to_poison_continuous_time_dynamic_graphs.md)
- [\[NeurIPS 2025\] WaLRUS: Wavelets for Long-range Representation Using SSMs](../../NeurIPS2025/time_series/walrus_wavelets_for_long-range_representation_using_ssms.md)
- [\[ICML 2026\] Nested Spatio-Temporal Time Series Forecasting](nested_spatio-temporal_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
