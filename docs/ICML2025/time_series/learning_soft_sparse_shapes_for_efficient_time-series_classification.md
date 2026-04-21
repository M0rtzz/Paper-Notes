---
title: >-
  [论文解读] Learning Soft Sparse Shapes for Efficient Time-Series Classification
description: >-
  [ICML2025][时间序列][shapelet] 提出 SoftShape 模型，用基于贡献分数的软稀疏化替代传统硬筛选 shapelet 的方式，结合 MoE 驱动的 intra-shape 和 shared expert 的 inter-shape 双模式时序模式学习，在 128 个 UCR 数据集上取得 SOTA 分类精度。
tags:
  - ICML2025
  - 时间序列
  - shapelet
  - 时间序列分类
  - 软稀疏化
  - Mixture-of-Experts
  - 可解释性
---

# Learning Soft Sparse Shapes for Efficient Time-Series Classification

**会议**: ICML2025  
**arXiv**: [2505.06892](https://arxiv.org/abs/2505.06892)  
**代码**: [GitHub](https://github.com/qianlima-lab/SoftShape)  
**领域**: time_series  
**关键词**: shapelet, 时间序列分类, 软稀疏化, Mixture-of-Experts, 可解释性

## 一句话总结
提出 SoftShape 模型，用基于贡献分数的软稀疏化替代传统硬筛选 shapelet 的方式，结合 MoE 驱动的 intra-shape 和 shared expert 的 inter-shape 双模式时序模式学习，在 128 个 UCR 数据集上取得 SOTA 分类精度。

## 研究背景与动机

Shapelet（判别性子序列）是时间序列分类（TSC）中兼具高准确率与可解释性的经典范式。然而：

**计算瓶颈**：暴力搜索所有候选子序列代价极高，需要对不同位置和长度的子序列逐一评估。

**硬筛选信息损失**：现有方法（如 shapelet transform）以硬方式丢弃大量子序列，可能遗漏对分类有益的时序模式。

**贡献度差异被忽视**：传统方法未区分不同 shapelet 对分类的贡献程度，导致部分"次要但有用"的子序列被直接抛弃。

本文的核心动机是：能否用**软方式**保留所有子序列信息，同时通过稀疏化减少计算开销？

## 方法详解

### 整体架构

SoftShape 由三个核心模块组成：Shape Embedding → Soft Shape Sparsification → Soft Shape Learning Block，堆叠 $L$ 层后接线性分类器。

### 1. Shape Embedding（形状嵌入）

使用 1D CNN 将输入时间序列 $\mathcal{X}_n = \{x_1, \ldots, x_T\}$ 转换为 $M = \frac{T-m}{q}+1$ 个重叠子序列嵌入：

$$\hat{\mathcal{S}}_{n,p}^{m} = \sum_{i=0}^{m-1} \mathbf{W}_i \mathcal{S}_{n,p}^{m}, \quad p = 0, q, 2q, \ldots, T-m$$

其中 $m$ 为窗口大小，$q$ 为步长，并加入可学习位置编码捕捉时序依赖。

### 2. Soft Shape Sparsification（软形状稀疏化）

**核心创新**：用门控注意力机制（gated attention）为每个 shape 嵌入计算分类贡献分数：

$$\alpha(\hat{\mathcal{S}}_{n,p}^{m}) = \sigma\left(\mathbf{W}_2 \tanh(\mathbf{W}_1 \hat{\mathcal{S}}_{n,p}^{m} + \mathbf{b}_1) + \mathbf{b}_2\right)$$

- **高分 shape**（Top-$\eta$ 比例）：乘以贡献分数得到软表示 $\tilde{\mathcal{S}}_{n,p}^{m} = \alpha \cdot \hat{\mathcal{S}}_{n,p}^{m}$
- **低分 shape**（剩余部分）：加权融合为单个嵌入 $\tilde{\mathcal{S}}_{\text{fused}}^{m} = \sum_{p \in \mathcal{E}} \alpha \cdot \hat{\mathcal{S}}_{n,p}^{m}$

这样既保留了所有子序列的信息（区别于硬丢弃），又减少了后续模块的输入规模。

### 3. Soft Shape Learning Block（软形状学习块）

#### 3.1 Intra-Shape Learning（形状内学习）——MoE

利用 MoE 路由器激活少量类别特定专家网络，为每个软 shape 嵌入学习局部时序模式：

$$G(\tilde{\mathcal{S}}_{n,p}^{m}) = \text{TOP}_k\left(\text{softmax}(\mathbf{W}_t \tilde{\mathcal{S}}_{n,p}^{m})\right)$$

每个专家为轻量级 MLP，输出为：

$$h_e(\theta, \tilde{\mathcal{S}}_{n,p}^{m}) = \hat{G}_e(\tilde{\mathcal{S}}_{n,p}^{m}) \cdot \text{GeLU}(\mathbf{W}_e \tilde{\mathcal{S}}_{n,p}^{m} + \mathbf{b}_e)$$

通过 importance loss $\mathcal{L}_{\text{imp}}$ 和 load balancing loss $\mathcal{L}_{\text{load}}$ 缓解专家不平衡问题。

#### 3.2 Inter-Shape Learning（形状间学习）——Shared Expert

将稀疏化后的 shape 嵌入转置为序列 $\mathcal{Q}_n^m = (\tilde{\mathcal{S}}_n^m)^T$，送入基于 Inception 模块的共享专家（三个不同 kernel size 的 1D CNN），学习跨 shape 的全局时序模式。

### 4. 训练目标

总损失函数为分类交叉熵损失与专家平衡损失的加权和：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ce}} + \lambda(\mathcal{L}_{\text{imp}} + \mathcal{L}_{\text{load}})$$

采用 conjunctive pooling 聚合最终预测。

## 实验关键数据

### 主实验：128 UCR 数据集

| 方法 | Avg. Acc | Avg. Rank | Win |
|------|----------|-----------|-----|
| InceptionTime | 0.9181 | 4.05 | 29 |
| TSLANet | 0.9205 | 3.68 | 31 |
| MR-H (Non-DL) | 0.8972 | 5.51 | 29 |
| RDST (Non-DL) | 0.8897 | 6.41 | 23 |
| **SoftShape** | **0.9334** | **2.72** | **53** |

- SoftShape 平均精度 93.34%，比第二名 TSLANet 高 1.3 个百分点
- 在 53/128 个数据集上取得最佳，远超其他方法
- 所有 P-value < 0.05，差异具有统计显著性

### 消融实验

| 变体 | Avg. Acc | Avg. Rank |
|------|----------|-----------|
| w/o Soft Sparse | 0.9123 | 3.04 |
| w/o Intra | 0.9245 | 2.75 |
| w/o Inter | 0.9022 | 3.74 |
| w/o Intra & Inter | 0.8696 | 5.02 |
| **SoftShape** | **0.9334** | **2.04** |

- Inter-shape 模块贡献最大（移除后精度下降 3.1%）
- 各模块均不可缺少

### 稀疏率分析（18 UCR 数据集）

| 稀疏率 | Avg. Acc | P-value |
|--------|----------|---------|
| 0% | 0.9461 | - |
| 10% | 0.9469 | 0.297 |
| 50% | 0.9453 | 0.346 |
| 70% | 0.9323 | 0.009 |
| 90% | 0.9261 | 0.0004 |

- 稀疏率 ≤50% 时精度无显著下降（P > 0.05），默认取 50% 兼顾效率

## 亮点与洞察

1. **软稀疏化思想精妙**：不丢弃任何子序列，而是将低贡献 shape 融合为单一表示，既减少计算量又保留信息完整性
2. **双路学习设计互补**：MoE 捕捉类别特定的局部模式 + Shared Expert 捕捉全局跨形状模式，局部-全局结合
3. **可解释性内生**：注意力分数 $\alpha$ 本身就是各子序列对分类贡献的可视化指标，无需额外解释模块
4. **实验扎实**：128 个 UCR 数据集 + 19 个 baseline + 完整消融 + 统计检验
5. **稀疏率 50% 性能无损**：说明大量子序列确实是冗余的，但不应硬丢弃

## 局限与展望

1. **仅支持单变量时间序列**：未验证多变量场景，实际应用中多变量更常见
2. **效率分析不够充分**：虽然声称提升效率，但缺乏详细的 FLOPs/推理速度对比
3. **数据集集中在 UCR**：UCR 数据集长度和复杂度有限，缺少长序列/大规模数据集的验证
4. **MoE 专家数等于类别数**：当类别数极多时，专家数量可能爆炸
5. **稀疏率 $\eta$ 需手动设定**：未探索自适应稀疏率策略

## 相关工作与启发

- **Shapelet 领域**：从硬筛选（Ye & Keogh 2009）→ shapelet transform（Hills 2014）→ 本文的软稀疏化，是该方向的自然演进
- **MoE 在时序中的应用**：不同于 forecasting 中用 MoE 处理多域混合数据，本文将 MoE 用于 shape 级别的类别特定模式学习
- **Patch tokenization**：借鉴 PatchTST 的 patch 思想，但将 patch 重新定义为 shape，引入贡献度加权

## 评分
- 新颖性: ⭐⭐⭐⭐ — 软稀疏化 + MoE-shape 双路学习的组合有新意
- 实验充分度: ⭐⭐⭐⭐⭐ — 128 数据集 + 19 baseline + 消融 + 统计检验非常完整
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式推导完整，图示直观
- 价值: ⭐⭐⭐⭐ — 为 shapelet 方法提供了新范式，但仅限单变量场景限制了实际影响力

<!-- RELATED:START -->

## 相关论文

- [A Unified Shape-Aware Foundation Model for Time Series Classification](../../AAAI2026/time_series/a_unified_shape-aware_foundation_model_for_time_series_class.md)
- [MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series](../../NeurIPS2025/time_series/maestro_adaptive_sparse_attention_and_robust_learning_for_multimodal_dynamic_tim.md)
- [TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting](temporal_query_network_for_efficient_multivariate_time_series_forecasting.md)
- [LETS-C: Leveraging Text Embedding for Time Series Classification](../../ACL2025/time_series/lets-c_leveraging_text_embedding_for_time_series_classification.md)
- [TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)

<!-- RELATED:END -->
