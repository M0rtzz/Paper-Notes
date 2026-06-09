---
title: >-
  [论文解读] PQ-SAM: Post-training Quantization for Segment Anything Model
description: >-
  [ECCV 2024][模型压缩][训练后量化] 本文提出PQ-SAM，首个专为Segment Anything Model定制的训练后量化方法，通过分组激活分布变换(GADT)和两阶段异常值层次聚类(OHC)方案解决SAM的高度不对称激活分布和有害异常值问题，将4-bit量化的SAM推进到可用水平。
tags:
  - "ECCV 2024"
  - "模型压缩"
  - "训练后量化"
  - "SAM"
  - "激活分布变换"
  - "异常值聚类"
  - "低比特量化"
---

# PQ-SAM: Post-training Quantization for Segment Anything Model

**会议**: ECCV 2024  
**arXiv**: N/A  
**代码**: 无  
**领域**: 模型压缩 / 量化  
**关键词**: 训练后量化, SAM, 激活分布变换, 异常值聚类, 低比特量化

## 一句话总结
本文提出PQ-SAM，首个专为Segment Anything Model定制的训练后量化方法，通过分组激活分布变换(GADT)和两阶段异常值层次聚类(OHC)方案解决SAM的高度不对称激活分布和有害异常值问题，将4-bit量化的SAM推进到可用水平。

## 研究背景与动机

**领域现状**：Segment Anything Model (SAM)是一个基于prompt引导的视觉基础模型，能够分割任意目标，展现了强大的零样本泛化能力。然而SAM拥有数十亿参数，计算开销巨大，难以在资源受限的边缘设备上部署。训练后量化(PTQ)是一种有效的快速部署方案，无需重新训练即可将模型压缩为低比特表示。

**现有痛点**：SAM经过数十亿规模的预训练，产生了高度不对称的激活分布，并在过多通道中存在有害的异常值(outlier)。这些特征导致现有PTQ方法在低比特（如4-bit）量化SAM时性能急剧下降。具体来说，不同通道间的激活值范围差异极大——某些通道的值可能是其他通道的100倍以上——使得统一量化参数无法同时适应所有通道。

**核心矛盾**：PTQ方法需要在量化精度和模型大小之间取得平衡。对于SAM这样的大模型，异常值的存在使得量化步长必须覆盖极端值，导致正常值的量化精度严重受损。现有方法要么逐通道量化（计算复杂），要么简单截断异常值（丢失重要信息），都不能很好地处理SAM特有的激活分布。

**本文目标** (1) 如何有效处理SAM激活中的极端异常值；(2) 如何在保持模型精度的同时实现张量级低比特量化；(3) 如何降低逐通道量化的参数优化难度。

**切入角度**：作者发现SAM的异常值呈现层次分布特征——少数通道有极端异常值，较多通道有中等异常值，大部分通道值正常。基于此观察，提出分层次处理异常值并通过分组策略降低优化复杂度。

**核心 idea**：通过两阶段异常值层次聚类来识别和处理不同级别的异常值，再利用分组机制对相似分布的通道统一学习缩放和偏移参数，将激活分布变换为量化友好的形式。

## 方法详解

### 整体框架
PQ-SAM在标准PTQ流程基础上增加了分组激活分布变换(GADT)模块。输入是预训练好的SAM模型和少量校准数据，输出是量化后的低比特SAM模型。核心流程为：首先通过异常值层次聚类(OHC)分析每层激活的分布特征，识别并分级处理异常值；然后将通道分组并学习每组的缩放和偏移参数，使激活分布更适合量化；最后联合优化变换参数和量化步长。

### 关键设计

1. **两阶段异常值层次聚类(Outlier Hierarchical Clustering, OHC)**:

    - 功能：识别激活中不同级别的异常值通道，并进行分级处理
    - 核心思路：第一阶段识别并截断极端异常值。对所有通道的激活范围进行统计，找到那些范围远超平均水平的"极端"通道（如超过均值3个标准差），对这些通道进行截断处理，将其范围缩小到合理区间。这一步大幅降低了通道间的尺度差异。第二阶段进行迭代聚类分组。在截断极端值后，剩余通道仍有分布差异，OHC根据通道激活的统计特征（均值、方差、范围）进行层次聚类，将分布相似的通道归为一组。每组内的通道可以共享量化参数，减少了需要优化的参数数量
    - 设计动机：直接对所有通道统一量化会因异常值"拉大"量化范围导致精度下降；逐通道独立量化参数又太多难以优化。分层处理先解决极端情况，再分组处理中等差异，兼顾了精度和效率

2. **分组激活分布变换(Grouped Activation Distribution Transformation, GADT)**:

    - 功能：学习通道级的缩放(scale)和偏移(shift)参数，将非对称的激活分布变换为量化友好的对称分布
    - 核心思路：对OHC产生的每个通道组，学习一组共享的缩放因子 $s_g$ 和偏移量 $z_g$。变换公式为 $\hat{a}_c = s_g \cdot a_c + z_g$，其中 $c$ 属于第 $g$ 组。缩放因子将不同组的值范围归一化到相近的尺度，偏移量将分布中心对齐到零点附近。由于组内通道分布相似，共享参数不会损失太多精度，但大幅减少了可学习参数的数量（从每通道2个参数降为每组2个参数）
    - 设计动机：减少可学习参数数量能显著降低优化难度，避免过拟合少量校准数据。同时分组策略保证了变换的有效性，因为组内通道确实具有相似的分布特征

3. **联合优化策略**:

    - 功能：同时优化分布变换参数和量化步长，获得全局最优解
    - 核心思路：将GADT的缩放/偏移参数与量化器的步长参数(step size)放在同一个优化目标下联合训练。损失函数为量化前后输出的均方误差(MSE)。使用少量校准数据（通常32-128张图片），通过梯度下降迭代优化所有参数。优化过程采用逐块(block-wise)策略，即每次只优化模型中一个Transformer块的参数，前面块的量化结果作为后续块的输入
    - 设计动机：分开优化变换参数和量化参数可能陷入局部最优，联合优化允许两者相互适应，找到更好的量化配置

### 损失函数 / 训练策略
使用重建误差作为损失函数：$\mathcal{L} = \|f(\hat{W}, \hat{A}) - f(W, A)\|^2$，即量化后的块输出与原始浮点块输出之间的MSE。采用AdamW优化器，学习率1e-4，在校准集上迭代10000步。

## 实验关键数据

### 主实验

| 数据集 | 比特位 | PQ-SAM (mIoU) | PTQ4ViT (mIoU) | 全精度 |
|--------|--------|--------------|----------------|--------|
| COCO (zero-shot) | W4A4 | **62.3** | 48.7 | 67.1 |
| COCO (zero-shot) | W6A6 | **66.2** | 63.8 | 67.1 |
| LVIS (zero-shot) | W4A4 | **58.1** | 41.2 | 63.5 |
| LVIS (zero-shot) | W6A6 | **62.7** | 59.4 | 63.5 |
| ADE20K (zero-shot) | W4A4 | **55.8** | 39.6 | 60.2 |

### 消融实验

| 配置 | mIoU (4-bit) | 说明 |
|------|-------------|------|
| Full PQ-SAM | **62.3** | 完整方法 |
| w/o OHC Stage-1 | 55.1 | 不处理极端异常值，掉7.2% |
| w/o OHC Stage-2 | 59.4 | 不分组，每通道独立优化，掉2.9% |
| w/o GADT (仅截断) | 52.8 | 不做分布变换，掉9.5% |
| 均匀分组 (无聚类) | 57.6 | 不用OHC聚类，随机分组，掉4.7% |

### 关键发现
- OHC的第一阶段（极端异常值截断）贡献最大，说明SAM中少数极端异常通道是量化的主要障碍
- 分组策略比逐通道优化效果更好，说明减少参数量有助于避免在少量校准数据上过拟合
- PQ-SAM在9个零样本数据集上一致超越现有PTQ方法，验证了方法的泛化性
- 4-bit量化首次达到可用水平（全精度的93%），此前4-bit SAM性能不到全精度的75%

## 亮点与洞察
- **层次化处理异常值**是一个通用且有效的策略。不同于暴力截断或忽略异常值，OHC通过分层处理保留了异常值中的有用信息，同时消除了对量化的负面影响。这一思路可以迁移到其他大规模预训练模型的量化中
- **分组优化减参**的策略巧妙利用了通道分布的聚类特性，在精度和可优化性之间找到了很好的平衡点
- 对SAM激活分布的深入分析（异常值分布的层次性特征）本身就是有价值的发现，有助于理解大规模预训练对模型内部表示的影响

## 局限与展望
- 论文主要关注SAM-ViT-H，对更小版本（SAM-ViT-B）的效果未充分验证
- 校准数据的选择对结果可能有影响，但论文对此讨论不足
- 仅针对权重和激活的量化，未考虑注意力矩阵的量化
- 可以尝试将OHC与GPTQ等更先进的权重量化方法结合
- 混合精度量化（对异常值通道保持高精度）可能是一个值得探索的方向

## 相关工作与启发
- **vs PTQ4ViT**: PTQ4ViT为ViT设计了twin均匀量化器，但没有针对SAM极端异常值分布的处理。PQ-SAM的OHC+GADT专门解决了这一问题
- **vs SmoothQuant**: SmoothQuant通过将量化难度从激活转移到权重来处理异常值，而PQ-SAM直接变换激活分布本身，两种思路互补
- **vs GPTQ**: GPTQ专注于权重量化，PQ-SAM专注于激活量化，二者可以组合使用

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个SAM专用PTQ方法，OHC+GADT组合新颖
- 实验充分度: ⭐⭐⭐⭐ 9个零样本数据集验证，消融全面
- 写作质量: ⭐⭐⭐⭐ 问题分析深入，方法描述清晰
- 价值: ⭐⭐⭐⭐ 将4-bit SAM推进到可用水平，有实际部署价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] MetaAug: Meta-Data Augmentation for Post-Training Quantization](metaaug_meta-data_augmentation_for_post-training_quantization.md)
- [\[ECCV 2024\] AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer](adalog_post-training_quantization_for_vision_transformers_with_adaptive_logarith.md)
- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](../../AAAI2026/model_compression/post_training_quantization_for_efficient_dataset_condensation.md)
- [\[ICML 2025\] BoA: Attention-aware Post-training Quantization without Backpropagation](../../ICML2025/model_compression/boa_attention-aware_post-training_quantization_without_backpropagation.md)
- [\[CVPR 2025\] QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge](../../CVPR2025/model_compression/quartdepth_post-training_quantization_for_real-time_depth_estimation_on_the_edge.md)

</div>

<!-- RELATED:END -->
