---
title: >-
  [论文解读] L-SWAG: Layer-Sample Wise Activation with Gradients Information for Zero-Shot NAS
description: >-
  [CVPR 2025][零代价NAS] 提出L-SWAG零代价代理，利用层级和样本级激活-梯度信息实现Vision Transformer的零样本神经架构搜索
tags:
  - CVPR 2025
  - 神经架构搜索
  - 零代价代理
  - Vision Transformer
  - NAS
---

# L-SWAG: Layer-Sample Wise Activation with Gradients Information for Zero-Shot NAS on Vision Transformers

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: 待确认  
**领域**: 零代价NAS  
**关键词**: 零代价代理, 神经架构搜索, ViT, 激活值, 梯度信息

## 一句话总结

本文提出L-SWAG（Layer-Sample Wise Activation with Gradients），一种新型通用零代价代理，通过结合层级和样本级的激活值与梯度信息来评估网络架构质量，首次将零代价NAS系统性地扩展到Vision Transformer搜索空间，并在Autoformer搜索空间的6个任务上建立了新的benchmark。

## 研究背景与动机

**领域现状**：神经架构搜索（NAS）旨在自动寻找最优网络结构，但传统NAS方法（多次/一次训练）计算开销巨大。零代价NAS（ZC-NAS）通过设计零代价代理（zero-cost proxy）在不训练的情况下预测架构性能，极大提升搜索效率。

**现有痛点**：(1) 现有SOTA零代价代理（如NASWOT、SynFlow、ZenNAS等）主要针对CNN搜索空间（如NAS-Bench-201）设计和验证，在ViT搜索空间上的表现未知。(2) 随着LLM推动Transformer架构成为主流，ViT架构搜索变得越来越重要，但缺乏系统性的ViT零代价搜索benchmark。(3) 现有代理要么只用激活值（如NASWOT的核函数），要么只用梯度（如SynFlow），未能有效结合二者的互补信息。

**核心矛盾**：零代价代理需要在极短时间内（单次前向/反向传播）准确估计一个架构的潜力，但ViT架构中注意力机制的复杂性使得传统代理的假设（如ReLU激活、卷积层级结构）不再成立。

**本文目标** 如何设计一个通用的零代价代理，使其在CNN和ViT搜索空间上都表现优异？

**切入角度**：从信息论角度出发，同时捕捉网络各层的激活值多样性和梯度信号质量，用层级和样本级的统计量来评估架构。

**核心 idea**：在每一层计算激活值与梯度的交互统计量，聚合样本级和层级信息形成架构的综合评分。

## 方法详解

### 整体框架

L-SWAG的计算流程：(1) 对候选架构的随机初始化参数，输入一小批数据做一次前向传播和反向传播。(2) 在每一层提取激活值和梯度张量，计算层级统计量。(3) 跨样本和跨层聚合统计量，得到单一标量评分。(4) 按评分对候选架构排序，选择最优架构。

### 关键设计

1. **层-样本级激活与梯度交互（Layer-Sample Wise Activation-Gradient Statistics）**：
    - 功能：捕捉每一层对输入数据的区分能力和梯度流质量
    - 核心思路：对网络的第$l$层，收集激活值矩阵 $A^l \in \mathbb{R}^{B \times D_l}$（B个样本，$D_l$维特征）和对应梯度矩阵 $G^l$。计算激活值的样本间相关性矩阵 $K_A^l = A^l (A^l)^T$（类似NASWOT的核矩阵），同时计算梯度的样本间相关性矩阵 $K_G^l = G^l (G^l)^T$。L-SWAG将二者结合，如计算 $\text{score}^l = f(K_A^l, K_G^l)$，其中$f$可以是矩阵元素积的统计量
    - 设计动机：激活值反映网络提取特征的多样性（好的架构应使不同输入的表示差异大），梯度反映训练信号的有效传播（好的架构梯度流应畅通且有区分性）

2. **跨层聚合策略**：
    - 功能：将各层的评分合成为全局架构质量评估
    - 核心思路：对所有可搜索层的评分进行加权聚合，权重可以是均匀的或基于层深度的递增/递减权重。最终评分 $S = \sum_l w_l \cdot \text{score}^l$。此外，还考虑了对不同层类型（注意力层vs FFN层）使用不同的聚合策略
    - 设计动机：ViT中不同层的功能差异显著，浅层注重局部特征、深层关注全局语义，应针对性加权

3. **Autoformer搜索空间Benchmark**：
    - 功能：首个系统性的ViT零代价NAS评测平台
    - 核心思路：基于Autoformer搜索空间（搜索embed_dim、depth、num_heads、mlp_ratio等），在ImageNet分类、COCO检测等6个下游任务上评测各零代价代理的排序相关性（Spearman/Kendall $\tau$），建立完整的benchmark
    - 设计动机：缺乏ViT零代价NAS的标准评测，阻碍了该领域的发展

## 实验关键数据

### 关键发现

- L-SWAG在Autoformer搜索空间的6个任务上均取得最佳或接近最佳的排序相关性
- 在传统CNN搜索空间（NAS-Bench-201、NAS-Bench-101）上也表现优异，证明通用性
- 现有SOTA代理（NASWOT、SynFlow等）在ViT搜索空间上表现显著下降
- 同时使用激活值和梯度比单独使用任一者提升约15-20%的排序相关性
- 搜索效率极高，单架构评估时间<0.5秒，整个搜索过程<10分钟

## 亮点与洞察

- **填补ViT零代价NAS空白**：系统性地将ZC-NAS扩展到ViT领域并提供benchmark
- **设计兼顾通用性**：在CNN和ViT空间均表现良好，不依赖特定架构假设
- **信息互补**：激活值+梯度的结合比单一信号更全面

## 局限与展望

- 评估仍依赖一小批数据（~64个样本），数据选择可能引入噪声
- 在超大规模搜索空间（如组合空间>10^10种架构）中的扩展性待验证
- 超参数（聚合权重、统计量选择）需要在验证集上调优
- 未来可探索与NAS搜索策略（进化算法、强化学习）的深度结合
