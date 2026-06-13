---
title: >-
  [论文解读] Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation
description: >-
  [ICML 2025][模型压缩][量化] 首次系统分析量化引入的离散化噪声如何破坏模型融合效果，提出 HDRQ（Hessian and Distance Regularizing Quantization）——通过 Hessian 正则化平坦化损失曲面 + 距离正则化保持量化模型间权重对齐 + 噪声采样舍入消除舍入歧义，使量化模型在多目标域适应融合中性能接近全精度水平。
tags:
  - "ICML 2025"
  - "模型压缩"
  - "量化"
  - "model merging"
  - "域适应"
  - "error barrier"
  - "Hessian regularization"
---

# Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation

**会议**: ICML 2025  
**作者**: Juncheol Shin, Minsang Seok, Seonggon Kim, Eunhyeok Park  
**arXiv**: [2505.23651](https://arxiv.org/abs/2505.23651)  
**代码**: 未公开  
**领域**: 模型压缩, 域适应  
**关键词**: post-training quantization, model merging, multi-target domain adaptation, error barrier, Hessian regularization  

## 一句话总结

首次系统分析量化引入的离散化噪声如何破坏模型融合效果，提出 HDRQ（Hessian and Distance Regularizing Quantization）——通过 Hessian 正则化平坦化损失曲面 + 距离正则化保持量化模型间权重对齐 + 噪声采样舍入消除舍入歧义，使量化模型在多目标域适应融合中性能接近全精度水平。

## 研究背景与动机

**领域现状**：模型融合（model merging）通过简单权重平均将多个针对不同目标域微调的模型合并为一个通用模型，实现无需额外训练的多目标域适应（MTDA）。与此同时，量化（quantization）是边缘设备部署的必要步骤，通过降低权重精度减少内存和计算开销。

**现有痛点**：量化和融合各自都有成熟的研究，但二者的交互完全被忽视。量化引入的离散化效应（discretization）会破坏权重的连续性，使得多个量化模型之间的权重插值路径穿越损失景观的高地（高误差屏障），导致融合后性能严重退化——甚至低于单个量化模型的性能。

**核心矛盾**：传统 PTQ 方法只优化单模型重建误差，不考虑量化后模型之间的融合兼容性。量化使模型权重偏离原始位置，同时增大了不同量化模型间的权重距离，双重因素导致插值路径上的误差屏障急剧升高。

**切入角度**：从误差屏障（error barrier）理论出发，将量化噪声显式引入误差屏障分析框架，推导出影响融合质量的两个关键因素：(1) 损失曲面的曲率（Hessian 敏感度）和 (2) 量化模型间的权重距离。

**核心 idea**：在后训练量化过程中加入面向融合的正则化约束，使量化模型"天生"适合被融合。

## 方法详解

### 整体框架

HDRQ 是一个修改后的 PTQ 流程：输入预训练源模型和多个目标域的少量校准数据，输出多个"融合友好"的量化模型。核心改变在于优化目标：在标准重建损失之上添加 Hessian 正则化项和距离正则化项，并用噪声采样代替传统的确定性舍入。

### 关键设计

1. **Hessian 正则化（敏感度控制）**:

    - 功能：平坦化量化模型附近的损失曲面，降低权重微扰对输出的影响
    - 核心思路：误差屏障的高度与权重偏移量和局部 Hessian 矩阵特征值成正比。对 Hessian trace 大的参数施加更强的正则化惩罚，迫使量化器在高敏感位置保持更高精度。实际使用 Fisher 信息矩阵近似 Hessian 对角线以降低计算成本
    - 设计动机：平坦的损失景观意味着权重插值路径上不会出现尖锐的损失跳跃，这是成功融合的必要条件

2. **距离正则化（权重对齐）**:

    - 功能：约束量化模型与预训练源模型之间的权重距离
    - 核心思路：最小化每个量化模型与源预训练模型之间的 L2 距离。由于所有目标域模型都从同一源模型出发微调，保持与源模型的距离就间接保持了量化模型之间的紧聚集
    - 设计动机：模型融合本质上是权重空间的线性插值，插值路径越短、端点越近，路径穿越高损失区域的概率越低

3. **噪声采样舍入（消除舍入歧义）**:

    - 功能：解决传统 round-to-nearest 在决策边界处的不稳定性
    - 核心思路：将量化噪声建模为加性噪声（additive noise），引入受控高斯噪声模拟量化粒度，使得正则化约束可以通过梯度反传优化。这种方式避免了 STE（直通估计器）的近似误差，在 PTQ 的少量校准数据条件下更稳定
    - 设计动机：确定性舍入在量化边界附近高度不稳定，微小的权重变化可能导致舍入方向翻转，进而在融合中引入不可预测的误差

### 损失函数

总优化目标为重建损失 + Hessian 正则化 + 距离正则化三项加权和：$\mathcal{L} = \mathcal{L}_{recon} + \lambda_H \cdot \mathcal{L}_{Hessian} + \lambda_D \cdot \mathcal{L}_{dist}$，其中 $\mathcal{L}_{recon}$ 为标准层级输出重建损失，$\mathcal{L}_{Hessian}$ 按 Hessian 对角线加权的敏感度惩罚，$\mathcal{L}_{dist}$ 为量化权重到源模型权重的 L2 距离。

## 实验

### 主实验——语义分割多目标域适应

| 方法 | 位宽 | 单模型 mIoU | 2 域融合 mIoU | 3 域融合 mIoU |
|------|------|-----------|-------------|-------------|
| 全精度（上界） | 32-bit | 64.23 | 65.41 | 66.12 |
| 传统 PTQ（BRECQ） | 4-bit | 63.15 | 60.82 | 59.54 |
| **HDRQ（本文）** | 4-bit | 63.58 | **65.03** | **65.87** |
| 改进幅度 | - | +0.43 | **+4.21** | **+6.33** |

### 消融实验——误差屏障与正则化贡献

| 配置 | 最大屏障高度 | 插值损失方差 | 融合 mIoU |
|------|-----------|-----------|----------|
| PTQ 无正则化 | 8.34 | 2.81 | 60.82 |
| + Hessian 正则化 | 5.21 | 1.45 | 63.17 |
| + 距离正则化 | 4.87 | 1.22 | 63.62 |
| + 两者 + 噪声舍入（完整 HDRQ） | **3.12** | **0.93** | **65.03** |

### 关键发现

- 传统 PTQ 融合后 mIoU 甚至低于单模型（60.82 < 63.15），说明量化严重破坏了融合特性；HDRQ 几乎完全恢复融合增益（65.03 vs 全精度 65.41）
- 误差屏障高度从 8.34 降至 3.12（-62.6%），直接验证了理论分析的有效性
- 4-bit 场景改进幅度（+4.21）远大于 8-bit（约+1.9），说明量化越激进、HDRQ 的价值越大
- 计算开销约为标准 PTQ 的 1.5 倍（Hessian 可用 Fisher 近似），远低于 QAT

## 亮点与洞察

- 首次定义并解决"量化-融合兼容性"问题，切中实际部署中量化+多任务的核心痛点
- 理论驱动设计：从误差屏障分析推导出正则化项的具体形式，而非凭直觉添加约束
- PTQ 级别的方法（无需全数据集训练），可插入任何现有 PTQ 流程，实用性强
- 在仅加 1.5 倍计算开销的情况下，4-bit 融合性能提升超过 4 mIoU——在语义分割中非常显著

## 局限性

- Hessian 对角线近似的精度在超大模型（100B+）上未经验证，计算可行性存疑
- 距离正则化假设所有目标域模型从同一源模型出发，不适用于异构初始化场景
- 超参 $\lambda_H$ 和 $\lambda_D$ 需要手动调节，缺乏自适应策略
- 实验主要在语义分割任务上验证，未涵盖 NLP 或生成模型的融合场景
- 融合策略固定为简单权重平均，未探索 Task Arithmetic 或 TIES-MERGING 等高级融合方法

## 相关工作与启发

- **PTQ 演进**：从 Nagel 2020 的层级重建到 Li 2021 的块级重建再到本文的融合感知重建，PTQ 的优化目标越来越"全局化"
- **模型融合**：Li et al. 2024 发现同源微调模型可通过权重平均实现 MTDA，但未考虑量化；本文填补了这个关键缺口
- **启发**：类似的"下游任务感知量化"思想可扩展到安全性约束（量化后模型是否保持公平性/对齐特性）

## 评分

| 维度 | 分数 | 理由 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ | 首次发现并解决量化-融合兼容性问题 |
| 技术深度 | ⭐⭐⭐⭐ | 误差屏障理论分析扎实，正则化推导有理论支撑 |
| 实验完整度 | ⭐⭐⭐⭐ | 多位宽、多域数量消融充分，但任务类型较单一 |
| 写作质量 | ⭐⭐⭐⭐ | 理论部分清晰，动机阐述充分 |
| 实用性 | ⭐⭐⭐⭐⭐ | PTQ 级复杂度+显著融合提升，直接可用于部署 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] BoA: Attention-aware Post-training Quantization without Backpropagation](boa_attention-aware_post-training_quantization_without_backpropagation.md)
- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](../../NeurIPS2025/model_compression/quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)
- [\[CVPR 2025\] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](../../CVPR2025/model_compression/q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)
- [\[ICML 2025\] WildChat-50m: A Deep Dive Into the Role of Synthetic Data in Post-Training](wildchat-50m_a_deep_dive_into_the_role_of_synthetic_data_in_post-training.md)
- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](../../AAAI2026/model_compression/post_training_quantization_for_efficient_dataset_condensation.md)

</div>

<!-- RELATED:END -->
