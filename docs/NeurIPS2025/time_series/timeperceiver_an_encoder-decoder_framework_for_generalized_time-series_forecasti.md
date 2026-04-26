---
title: >-
  [论文解读] TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting
description: >-
  [NeurIPS 2025][时间序列][时间序列预测] 提出 TimePerceiver 统一编码器-解码器框架，通过广义化预测任务（同时包含外推、插值和填补）以及潜在瓶颈编码器 + 查询式解码器设计，在 8 个标准基准上取得全面 SOTA。
tags:
  - NeurIPS 2025
  - 时间序列
  - 时间序列预测
  - 编码器-解码器
  - 潜在瓶颈
  - 广义预测公式
  - 交叉注意力
---

# TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting

**会议**: NeurIPS 2025  
**arXiv**: [2512.22550](https://arxiv.org/abs/2512.22550)  
**代码**: [GitHub](https://github.com/efficient-learning-lab/TimePerceiver)  
**领域**: 时间序列  
**关键词**: 时间序列预测, 编码器-解码器, 潜在瓶颈, 广义预测公式, 交叉注意力

## 一句话总结

提出 TimePerceiver 统一编码器-解码器框架，通过广义化预测任务（同时包含外推、插值和填补）以及潜在瓶颈编码器 + 查询式解码器设计，在 8 个标准基准上取得全面 SOTA。

## 研究背景与动机

时间序列预测领域近年来涌现了大量新架构（Transformer、CNN、MLP、SSM），但这些工作**过度聚焦于编码器设计**，而忽略了两个同样重要的方面：

**解码策略粗糙**：大多数方法直接用线性投影从编码表示预测未来值，难以充分捕获复杂的时序结构

**训练策略与架构脱节**：受 BERT 启发的预训练-微调（mask-and-reconstruct → forecasting）两阶段训练存在目标不对齐问题——预训练学的是重建，但最终目标是预测

此外，通道独立模型（如 PatchTST）简单鲁棒但忽略了跨通道交互，而通道依赖模型（如 iTransformer、CARD）虽然建模了交互但计算成本高且效果不稳定。

**核心创新思路**：将标准预测任务（从过去连续观测预测未来连续值）**广义化**为在时间轴上的任意位置进行预测（外推+插值+填补），使训练过程自然对齐预测目标，消除两阶段训练的需求。

## 方法详解

### 整体框架

TimePerceiver 由三部分组成：(1) 基于 Patch 的嵌入构建；(2) 带有潜在瓶颈的编码器，联合建模时序和跨通道依赖；(3) 基于查询的解码器，根据目标时间戳选择性检索相关信息。

### 关键设计

1. **广义预测公式**：将标准预测 $f_\theta(\mathbf{X}_{\text{past}}) \to \mathbf{X}_{\text{future}}$ 推广为对任意时间索引子集的预测。给定输入索引 $\mathcal{I}$ 和目标索引 $\mathcal{J} = \{1,...,T\} \setminus \mathcal{I}$：

   $$\hat{\mathbf{X}}_{\mathcal{J}} = g_\theta(\mathbf{X}_{\mathcal{I}}, \mathcal{I}, \mathcal{J})$$

   训练时随机采样输入-目标分割，涵盖外推（预测未来）、插值（预测中间缺失）和填补（预测过去缺失）三种任务。标准预测只是这个广义公式的特例。这使模型在单轮端到端训练中就能学到深层的时序动态理解。

2. **潜在瓶颈编码器**：引入 $M$ 个可学习的潜在 token $\mathbf{Z}^{(0)} \in \mathbb{R}^{M \times D}$（$M \ll C|\mathcal{I}_{\text{patch}}|$），通过三步瓶颈过程编码输入：

    - **压缩**：潜在 token 通过交叉注意力从输入 token 收集上下文信息
     $\mathbf{Z}^{(1)} = \text{AttnBlock}(\mathbf{Z}^{(0)}, \mathbf{H}^{(0)})$
    - **精炼**：$K$ 层自注意力在潜在空间内相互作用
     $\mathbf{Z}^{(k+1)} = \text{AttnBlock}(\mathbf{Z}^{(k)}, \mathbf{Z}^{(k)})$
    - **回传**：更新后的潜在 token 反向增强输入 token
     $\mathbf{H}^{(1)} = \text{AttnBlock}(\mathbf{H}^{(0)}, \mathbf{Z}^{(K+1)})$

   复杂度从全注意力的 $\mathcal{O}(N^2)$ 降为 $\mathcal{O}(NM)$，同时选择性保留时序和跨通道关键模式。

3. **查询式解码器**：利用目标 patch 对应的位置嵌入（时间位置 + 通道位置）构建查询 $\mathbf{Q}^{(0)}$，通过交叉注意力从编码输出中检索相关信息：

   $$\mathbf{Q}^{(1)} = \text{AttnBlock}(\mathbf{Q}^{(0)}, \mathbf{H}^{(1)})$$

   最终通过线性投影 $\hat{\mathbf{X}}_{\mathcal{P}_j, c} = \mathbf{Q}^{(1)}_{c,j} \mathbf{W}_{\text{output}}$ 生成预测值。这种设计自然适配广义预测公式——无论目标位置在哪，解码器都能通过位置查询获取正确的上下文。

### 损失函数 / 训练策略

端到端训练，使用 MSE 损失：

$$\mathcal{L} = \frac{1}{|\mathcal{J}|C} \sum_{j \in \mathcal{J}} \|\hat{\mathbf{x}}_j - \mathbf{x}_j\|_2^2$$

训练时随机采样输入-目标索引分割，无需预训练阶段。输入长度从 $\{96, 384, 768\}$ 中变化以增强泛化性。

## 实验关键数据

### 主实验（8 个数据集，MSE 平均值，$L$ 在 96/384/768 上平均）

| 数据集 | TimePerceiver | DeformableTST | CARD | PatchTST | iTransformer | 提升(vs次优) |
|-------|-------------|--------------|------|----------|------------|-----------|
| Weather | **0.227** | 0.233 | 0.247 | 0.236 | 0.244 | -2.6% |
| Solar | **0.198** | 0.199 | 0.228 | 0.234 | 0.214 | -0.5% |
| Electricity | **0.161** | 0.169 | 0.174 | 0.177 | 0.175 | -4.7% |
| Traffic | **0.407** | 0.410 | 0.426 | 0.430 | 0.424 | -0.7% |
| ETTh1 | **0.410** | 0.413 | 0.430 | 0.438 | 0.461 | -0.7% |
| ETTh2 | **0.344** | 0.336 | 0.355 | 0.356 | 0.390 | — |
| ETTm1 | **0.347** | 0.358 | 0.368 | 0.365 | 0.386 | -3.1% |
| ETTm2 | **0.261** | 0.267 | 0.268 | 0.273 | 0.281 | -2.2% |
| **Rank** | **1.375** | 2.525 | 4.975 | 5.450 | 6.475 | — |

***80 个指标中取得 55 个最优、17 个次优。***

### 消融实验

| 公式/编码器/PE策略 | ETTh1 MSE | ETTm1 MSE | Solar MSE | ECL MSE |
|----------------|----------|----------|----------|---------|
| 标准公式 + 潜在瓶颈 | 0.420 | 0.355 | 0.194 | 0.169 |
| **广义公式 + 潜在瓶颈** | **0.404** | **0.338** | **0.182** | **0.157** |
| 广义公式 + 全自注意力 | 0.425 | 0.353 | 0.192 | 0.161 |
| 广义公式 + 解耦自注意力 | 0.423 | 0.356 | 0.189 | 0.158 |
| 广义公式 + 不共享 PE | 0.423 | 0.342 | 0.193 | 0.163 |

### 关键发现

1. **广义公式比标准公式全面更优**：平均 MSE 改善 5.0%，MAE 改善 3.4%，说明暴露于更多样的时序推理任务有助于泛化
2. **潜在瓶颈优于全注意力**：瓶颈不仅计算高效，还通过信息压缩迫使模型学习更本质的模式
3. **广义公式具有通用性**：应用于 PatchTST 编码器 + 查询解码器的组合同样带来提升（ETTh1 MSE: 0.423→0.415）
4. **通道共享 PE 优于不共享**：共享位置编码使模型更好地利用跨通道的位置信息

## 亮点与洞察

- **视角转换**：不再只关注"更好的编码器"，而是从训练目标和解码器设计角度系统思考预测问题
- **广义公式的优雅性**：通过随机采样输入-目标分割，将预训练和预测训练统一为一个过程，消除了两阶段训练的复杂性
- **瓶颈机制的双重作用**：既降低计算成本又起了类似正则化的作用，提升泛化

## 局限与展望

1. 广义公式的随机采样策略可能需要更多训练 epoch 收敛
2. 查询解码器增加了额外的交叉注意力计算，比纯线性投影慢
3. 目前仅在固定 patch 大小设置下评估，未探索自适应 patch 策略
4. 在通道极多（如 Traffic 862 通道）时瓶颈大小选择对结果影响较大

## 相关工作与启发

TimePerceiver 的名字和架构灵感来自 Perceiver 系列（DeepMind），将潜在瓶颈思想引入时序领域。广义预测公式可视为 BERT 式预训练与预测任务的统一，为时序基础模型的训练范式提供了新思路。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 广义预测公式 + 瓶颈编码器 + 查询解码器的统一框架非常新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ 8 数据集、多输入长度、详尽消融、与 9 个基线全面比较
- **写作质量**: ⭐⭐⭐⭐⭐ 动机论述充分、公式清晰、图示直观
- **价值**: ⭐⭐⭐⭐⭐ 在竞争激烈的时序预测领域建立了新 SOTA，思路可推广

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics](ioncast_a_deep_learning_framework_for_forecasting_ionospheric_dynamics.md)
- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](selective_learning_for_deep_time_series_forecasting.md)
- [\[NeurIPS 2025\] AERO: A Redirection-Based Optimization Framework Inspired by Judo for Robust Probabilistic Forecasting](aero_a_redirection-based_optimization_framework_inspired_by_judo_for_robust_prob.md)
- [\[CVPR 2025\] DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification](../../CVPR2025/time_series/dejavid_encoder-agnostic_learned_temporal_matching_for_video_classification.md)
- [\[ICLR 2026\] Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](../../ICLR2026/time_series/towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)

<!-- RELATED:END -->
