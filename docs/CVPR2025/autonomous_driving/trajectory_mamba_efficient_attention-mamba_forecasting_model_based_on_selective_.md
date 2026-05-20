---
title: >-
  [论文解读] Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM
description: >-
  [CVPR 2025][自动驾驶][轨迹预测] 提出 Trajectory Mamba（Tamba），基于选择性状态空间模型重新设计自注意力机制，实现线性时间复杂度的轨迹预测，同时通过联合 polyline 编码策略和跨状态空间解码器保持预测精度，参数减少 40%+ 且 FLOPs 降低 4 倍。
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "轨迹预测"
  - "状态空间模型"
  - "Mamba"
  - "注意力机制"
---

# Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM

**会议**: CVPR 2025  
**arXiv**: [2503.10898](https://arxiv.org/abs/2503.10898)  
**代码**: [GitHub](https://github.com/YiZhou-H/Trajectory-Mamba-CVPR)  
**领域**: 自动驾驶  
**关键词**: 轨迹预测, 状态空间模型, Mamba, 注意力机制, 自动驾驶

## 一句话总结

提出 Trajectory Mamba（Tamba），基于选择性状态空间模型重新设计自注意力机制，实现线性时间复杂度的轨迹预测，同时通过联合 polyline 编码策略和跨状态空间解码器保持预测精度，参数减少 40%+ 且 FLOPs 降低 4 倍。

## 研究背景与动机

自动驾驶中的运动预测需要根据历史轨迹预测未来车辆运动，对实时性和准确性有双重要求。基于注意力机制的 Transformer 模型在预测精度上表现优异，但计算复杂度随目标数量二次增长，在高动态场景中成为瓶颈。

LSTM 等方法虽然更高效，但预测精度无法匹敌 Transformer。稀疏上下文编码技术（如 QCNet）提升了效率，但注意力机制的内在设计约束仍限制了递归推理的效率。

**核心问题**: 如何在保持 Transformer 级别预测精度的同时，实现线性时间复杂度？状态空间模型（SSM/Mamba）提供了一种可能。

此外，现有方法在融合异构数据（静态场景 vs 动态智能体）时采用统一的水平分类方式，未充分考虑场景元素间的深层关联（如行人和交通灯对车辆的联合约束）。

## 方法详解

### 整体框架

Tamba 采用三编码器-一解码器的多模态架构。三个并行 Tamba 编码器分别处理：(I) 时空注意力（每个时间步内所有元素间交互）；(II) 场景注意力（所有智能体与场景元素交互）；(III) 交通注意力（行人+交通灯对其他动态智能体的影响）。编码器输出拼接后输入 Cross-Tamba 解码器，生成 $K$ 条候选轨迹。二次解码融合场景上下文进行轨迹精炼，最终通过 RNN 分配轨迹置信度分数。

### 关键设计1: 联合 Polyline 编码策略

**功能**: 更有效地融合异构交通数据，增强预测精度。

**核心思路**: 将强信息关联的 polyline 类型使用共享嵌入器联合编码。具体地，行人轨迹和交通灯信号使用共享嵌入器联合编码：$\mathcal{P}_{\text{joint}} = \text{Fusion}(\text{Embed}(\mathcal{P}_{\text{pedestrian}}), \text{Embed}(\mathcal{P}_{\text{traffic}}))$。车道线、交通标志等特征差异大的类型使用独立嵌入器。

**设计动机**: 行人和交通灯对车辆施加相似的行为约束（如行人优先规则即使无交通灯也有效）。联合编码通过共享嵌入器促进特征共享，使这些约束因素能协同影响车辆轨迹推理。此设计减少了模型复杂度同时增强了语义理解。

### 关键设计2: 选择性 SSM 替代多头注意力

**功能**: 将自注意力机制的二次复杂度降低为线性。

**核心思路**: 使用 Mamba 的选择性状态空间模型替代标准多头注意力。SSM 通过输入依赖的参数化选择性地保留或丢弃序列信息：$\mathbf{h}_{t+1} = A(\mathcal{P}_t)\mathbf{h}_t + B(\mathcal{P}_t)\mathbf{u}_t$，$\mathbf{y}_t = C(\mathcal{P}_t)\mathbf{h}_t + D(\mathcal{P}_t)\mathbf{u}_t$。矩阵 $A$, $B$, $C$, $D$ 均为输入 $\mathcal{P}_t$ 的函数，整体复杂度为 $O(T \cdot n^2)$（$T$ 为序列长度，$n$ 为状态维度），对序列长度线性。

**设计动机**: 轨迹预测是递归性任务，每步预测作为下一步输入，Transformer 的二次复杂度在递归中逐步累积。SSM 天然支持序列处理且复杂度线性，同时选择性机制使模型能聚焦关键信息。一维卷积预处理提取局部短程特征，确保在进入 SSM 前已进行初步筛选。

### 关键设计3: Cross-Tamba 解码器

**功能**: 解决"一对多"轨迹生成问题，让所有目标共享统一场景表示。

**核心思路**: 借鉴 DETR 使用 $K$ 个独立查询向量表示不同候选轨迹。用 SSM 替代交叉注意力中的多头注意力：查询向量 $Q$ 与编码器输出的 Key/Value 分别交互，Key/Value 融合了当前编码特征和上一递归步的推理结果。所有目标智能体共享单一静态场景表示。预测权重推理模块使用 RNN 生成轨迹置信度。

**设计动机**: 解码器与编码器采用不同架构设计，编码器侧重帧内特征提取（SSM 替代自注意力），解码器侧重跨模态交互（SSM 替代交叉注意力），在效率和精度间取得平衡。RNN 提供跨轨迹约束确保整体分布合理。

### 损失函数

三部分联合损失：$L_{\text{total}} = L_{\text{proposal}} + L_{\text{refine}} + \lambda L_{\text{cls}}$。$L_{\text{proposal}}$ 为候选轨迹的 MSE 损失；$L_{\text{refine}}$ 采用 Laplace 混合分布的负对数似然，使用"赢家通吃"策略仅回传最佳轨迹；$L_{\text{cls}}$ 为混合系数的分类损失。

## 实验关键数据

### Argoverse 2 主实验结果

| 方法 | b-minFDE6↓ | minADE6↓ | minFDE6↓ | MR6↓ | #Params(M)↓ | FLOPs(G)↓ |
|------|-----------|---------|---------|------|------------|----------|
| MTR | 1.98 | 0.73 | 1.44 | 0.15 | 65.78 | — |
| GANet | 1.96 | 0.72 | 1.34 | 0.17 | 61.73 | — |
| QML | 1.95 | 0.69 | 1.39 | 0.19 | 9.39 | — |
| **Tamba** | **SOTA** | **SOTA** | **SOTA** | — | **~5.6** | **~4x fewer** |

### 效率对比

| 指标 | Tamba vs 现有方法 |
|------|-----------------|
| 参数量 | **减少 40%+** |
| FLOPs | **减少 4x** |
| 推理速度 | **SOTA** |

### 关键发现

1. **效率与精度双赢**: Tamba 参数减少 40% 以上、FLOPs 降低 4 倍，同时预测精度超越绝大多数方法。
2. **联合编码有效**: 行人-交通灯联合编码提升了车辆轨迹预测精度，验证了交通规则约束的建模价值。
3. **两个数据集一致性**: 在 Argoverse 1 和 2 上均取得效率 SOTA，证明方法的通用性。
4. **SSM 替代注意力可行**: 首次在轨迹预测领域验证 Mamba/SSM 能有效替代 Transformer，开创了新的技术路线。

## 亮点与洞察

- **Mamba 在轨迹预测的首次系统应用**: 从编码器到解码器全面使用 SSM，不是简单替换，而是针对任务特点重新设计。
- **编解码器异构设计**: 编码器用 SSM 替代自注意力关注帧内提取，解码器用 SSM 替代交叉注意力关注跨模态推理。
- **联合编码的交通语义洞察**: 识别到行人和交通灯作为"交通控制元素"对车辆有协同约束效应。

## 局限与展望

- **SSM 的表达能力上界**: SSM 在捕捉极其复杂的长距离依赖时可能不如全注意力 Transformer。
- **行人-交通灯假设**: 联合编码基于领域知识的手动设计，可能不适用于所有交通场景。
- **未验证更大规模场景**: 如何在超大规模场景（100+ agents）中表现有待验证。
- 未来可探索自适应聚合策略、端到端的 polyline 分组学习。

## 相关工作与启发

- **QCNet/MTR**: 基于 Transformer 的轨迹预测 SOTA，Tamba 以更少参数达到相当性能。
- **Motion Mamba**: 人体运动预测中的 SSM 应用，Tamba 扩展到更复杂的多智能体交通场景。
- **启发**: SSM/Mamba 在时序递归任务中的线性复杂度优势值得在更多自动驾驶子任务中探索。

## 评分

⭐⭐⭐⭐ — 首次系统性地将 Mamba 引入轨迹预测，4x FLOPs 减少和 40%+ 参数减少的效率提升令人印象深刻。联合编码设计有独到的领域洞察。在精度微小不足的情况下实现巨大效率提升，权衡合理。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] CAWM-Mamba: A Unified Model for Infrared-Visible Image Fusion and Compound Adverse Weather Restoration](cawm-mamba_a_unified_model_for_infrared-visible_image_fusion_and_compound_advers.md)
- [\[CVPR 2025\] WeatherGen: A Unified Diverse Weather Generator for LiDAR Point Clouds via Spider Mamba Diffusion](weathergen_a_unified_diverse_weather_generator_for_lidar_point_clouds_via_spider.md)
- [\[ICCV 2025\] Future-Aware Interaction Network For Motion Forecasting](../../ICCV2025/autonomous_driving/future-aware_interaction_network_for_motion_forecasting.md)
- [\[CVPR 2025\] Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting](spatiotemporal_decoupling_for_efficient_vision-based_occupancy_forecasting.md)
- [\[CVPR 2025\] Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning](tra-moe_learning_trajectory_prediction_model_from_multiple_domains_for_adaptive_.md)

</div>

<!-- RELATED:END -->
