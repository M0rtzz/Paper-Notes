---
title: >-
  [论文解读] Batch Loss Score for Dynamic Data Pruning
description: >-
  [CVPR 2026][模型压缩][剪枝] 提出 Batch Loss Score (BLS)，一种仅用均值 batch loss（而非难以获取的逐样本 loss）来估计样本重要性的方法，通过 EMA 低通滤波的信号处理视角提供理论保证，仅需 3 行代码即可集成到现有动态剪枝框架中。
tags:
  - "CVPR 2026"
  - "模型压缩"
  - "剪枝"
  - "batch loss"
  - "EMA"
  - "training efficiency"
  - "sample importance"
---

# Batch Loss Score for Dynamic Data Pruning

**会议**: CVPR 2026  
**arXiv**: [2604.04681](https://arxiv.org/abs/2604.04681)  
**代码**: [https://github.com/mrazhou/BLS](https://github.com/mrazhou/BLS)  
**领域**: 训练效率 / 数据剪枝  
**关键词**: dynamic data pruning, batch loss, EMA, training efficiency, sample importance

## 一句话总结

提出 Batch Loss Score (BLS)，一种仅用均值 batch loss（而非难以获取的逐样本 loss）来估计样本重要性的方法，通过 EMA 低通滤波的信号处理视角提供理论保证，仅需 3 行代码即可集成到现有动态剪枝框架中。

## 研究背景与动机

动态数据剪枝通过跳过不太信息化的样本来加速深度学习训练。逐样本 loss 是最直观的重要性度量，但在实践中获取它面临重大障碍：标准训练管线高度优化于计算均值 batch loss，从聚合后的损失恢复个体 loss 并非易事。对于复杂目标函数（如多组件检测 loss），定义和分离逐样本标量需要深度的任务特定知识和代码修改。

BLS 的核心洞察：虽然逐样本 loss 难以获取，但均值 batch loss 是无处不在的。通过为每个样本维护一个 EMA 分数（仅在该样本出现在当前 batch 时更新），可以间接推断样本重要性。

## 方法详解

### 整体框架

每个样本关联一个分数 s_i(t)，当样本 i 出现在 batch B_t 中时用 EMA 更新：s_i(t) = α·s_i(t-1) + (1-α)·L(B_t,t)。BLS 作为透明代理替换现有框架中的逐样本 loss。

### 关键设计

1. **信号分解与滤波**：从单个样本视角，均值 batch loss = 缩放信号（该样本的 1/B · l_i(t)）+ 批组成噪声（其他 B-1 个样本的 loss 贡献）。EMA 作为一阶 IIR 低通滤波器，衰减高频批组成噪声，保留低频的持久 loss 趋势。

2. **频率分离假设**：批组成噪声的高频波动（每步随机抽样导致）远高于缩放逐样本 loss 的演变频率（模型参数缓慢更新导致），使低通滤波有效。

3. **无缝代理集成**：BLS 作为逐样本 loss 的即插即用替代品，下游剪枝算法完全不感知分数来源，无需修改核心调度逻辑或超参数。3 行代码注入 vs InfoBatch 的 33+ 行侵入式修改。

### 损失函数 / 训练策略

BLS 本身不改变训练损失，仅影响样本选择。EMA 衰减因子 α 控制滤波特性：α 越大噪声抑制越强但响应越慢。

### 理论保证

从信号分解角度，单个样本 $i$ 所在 batch 的均值 loss 可分解为缩放信号 $\frac{1}{B} l_i(t)$ 和批组成噪声 $\frac{1}{B}\sum_{j \neq i} l_j(t)$。频率分离假设指出批组成噪声的高频波动远高于缩放逐样本 loss 的演变频率。EMA 作为一阶 IIR 低通滤波器 $H_\alpha$，其脉冲响应为 $h[n] = (1-\alpha)\alpha^n u[n]$，频率响应 $|H(e^{j\omega})| = \frac{1-\alpha}{\sqrt{1-2\alpha\cos(\omega)+\alpha^2}}$ 在 $\omega=0$ 处最大，滤除高频噪声保留低频趋势。

## 实验关键数据

### 主实验

| 数据集/任务 | 方法 | 剪枝率 | 性能 | 说明 |
|------------|------|--------|------|------|
| ToCa (3M, 零样本字幕) | BLS-SeTa | 32% | CIDEr 71.2 | ≈ SeTa 71.5 |
| MJ+ST (15M, 文字识别) | BLS-SeTa | 33% | IIIT5k 96.2% | ≈ Full 96.1% |
| CIFAR10 | BLS-InfoBatch | 30% | 95.5% | ≈ Full 95.6% |

BLS 作为 InfoBatch 和 SeTa 两种剩下框架的透明代理，仅需 3 行代码注入（vs InfoBatch 33+ 行侵入式修改）。
下游剪枝算法完全不感知分数来源，无需修改核心调度逻辑或超参数。

### 关键发现

- BLS 在 14 个数据集、11 个任务、18 个模型上验证，可无损剪枝 20%-50% 的样本
- 作为代理替换逐样本 loss 后，性能与原始方法相当甚至更优
- 特别适合复杂场景（多组件 loss、大规模数据）中逐样本 loss 难以获取的情况
- BLS 初始化为第一个 batch 的均值 loss，之后仅在样本出现在当前 batch 时更新

## 亮点与洞察

- 从信号处理角度（低通滤波）为 BLS 提供了严格的理论保证
- 3 行代码的极简实现降低了使用门槛
- 解耦了"样本评分"和"样本选择"，使其可与任何基于 loss 的剪枝策略组合
- 频率分离假设直觉清晰且有实验验证

## 局限与展望

- EMA α 需要根据任务调优
- 在训练极早期（分数未充分积累时）可能不够准确

## 评分

- 新颖性：⭐⭐⭐⭐ — 用 batch loss 代理逐样本 loss 思路新颖
- 技术深度：⭐⭐⭐⭐⭐ — 信号处理理论分析严谨
- 实验充分度：⭐⭐⭐⭐⭐ — 14 数据集 11 任务 18 模型
- 实用价值：⭐⭐⭐⭐⭐ — 3行代码，极高实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Beyond Loss Values: Robust Dynamic Pruning via Loss Trajectory Alignment](beyond_loss_values_robust_dynamic_pruning_via_loss_trajectory_alignment.md)
- [\[CVPR 2026\] HeSS: Head Sensitivity Score for Sparsity Redistribution in VGGT](hess_head_sensitivity_score_for_sparsity_redistribution_in_vggt.md)
- [\[CVPR 2026\] PPCL: Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](ppcl_pluggable_pruning_dit_distillation.md)
- [\[CVPR 2026\] SODA: Sensitivity-Oriented Dynamic Acceleration for Diffusion Transformer](soda_sensitivity-oriented_dynamic_acceleration_for_diffusion_transformer.md)
- [\[CVPR 2026\] Fixed Anchors Are Not Enough: Dynamic Retrieval and Persistent Homology for Dataset Distillation](fixed_anchors_are_not_enough_dynamic_retrieval_and_persistent_homology_for_datas.md)

</div>

<!-- RELATED:END -->
