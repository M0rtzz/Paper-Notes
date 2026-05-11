---
title: >-
  [论文解读] LEMMA: Laplacian Pyramids for Efficient Marine Semantic Segmentation
description: >-
  [CVPR 2026][图像分割][轻量化语义分割] 提出LEMMA，一种基于拉普拉斯金字塔的轻量级海洋语义分割模型，通过金字塔分解提取边缘信息来替代深层特征计算，在参数量减少71倍的条件下实现了SOTA级别的分割精度（MaSTr1325上98.97% mIoU）。
tags:
  - CVPR 2026
  - 图像分割
  - 轻量化语义分割
  - 拉普拉斯金字塔
  - 海洋语义分割
  - 边缘检测
  - 无人水面艇
---

# LEMMA: Laplacian Pyramids for Efficient Marine Semantic Segmentation

**会议**: CVPR 2026  
**arXiv**: [2603.25689](https://arxiv.org/abs/2603.25689)  
**代码**: 无  
**领域**: 语义分割  
**关键词**: 轻量化语义分割, 拉普拉斯金字塔, 海洋语义分割, 边缘检测, 无人水面艇

## 一句话总结

提出LEMMA，一种基于拉普拉斯金字塔的轻量级海洋语义分割模型，通过金字塔分解提取边缘信息来替代深层特征计算，在参数量减少71倍的条件下实现了SOTA级别的分割精度（MaSTr1325上98.97% mIoU）。

## 研究背景与动机

海洋场景的语义分割对于无人水面艇(USV)自主导航和沿海地球观测（如油污检测）至关重要。然而，现有的语义分割方法（如WaSR-T、DeepLabv3等）通常依赖深层CNN或Transformer架构，拥有数千万甚至上亿的参数量和极高的计算开销，难以在无人机、USV等资源受限的边缘设备上实时运行。

核心矛盾在于：海洋场景需要高精度分割（水面反射、薄油膜等低对比度区域），但部署平台（无人机/USV）的算力极为有限。现有方法在精度和效率之间无法兼顾——WaSR-T虽然达到99.80% mIoU，但需要71.4M参数和133.8 GFLOPs。

本文的切入角度是利用拉普拉斯金字塔分解天然提供的边缘信息。金字塔的各层包含了不同分辨率下的边缘细节，这些信息可以在特征提取的早期阶段就被注入，从而避免在深层网络中进行昂贵的特征图计算。核心idea：用拉普拉斯金字塔的边缘先验替代深层特征提取，实现轻量化与高精度的兼得。

## 方法详解

### 整体框架

LEMMA将输入图像分解为深度为3的拉普拉斯金字塔（$L_1$、$L_2$、$L_3$），然后通过三个分支分别处理不同尺度的特征：Low-level Feature Branch (LFB)处理最低分辨率的$L_3$，Middle-level Feature Branch (MFB)融合$L_2$与LFB的输出，High-level Feature Branch (HFB)融合$L_1$与前两个分支的特征并生成最终分割掩码。各分支内部使用残差块链和卷积层进行特征提取，通过级联拼接和转置卷积实现跨尺度信息融合。

### 关键设计

1. **拉普拉斯金字塔分解**:
    - 功能：将图像分解为不同分辨率的边缘信息表示
    - 核心思路：金字塔的每一层天然包含该分辨率下的高频边缘细节，一次分解即可获得多尺度边缘表征
    - 设计动机：边缘信息是海洋场景中区分水面、障碍物、油污等的关键线索，利用金字塔可避免深层网络逐步学习边缘特征的高昂开销

2. **三分支残差处理架构 (LFB/MFB/HFB)**:
    - 功能：对金字塔各层特征进行不同深度的精炼和融合
    - 核心思路：LFB处理最低分辨率特征（64通道），MFB融合中低层信息，HFB在最高分辨率上仅用16通道轻量处理即可完成掩码重建
    - 设计动机：HFB使用16通道而非64通道，大幅降低高分辨率特征图上的GFLOPs；通过级联拼接保留各层的原始信息，避免信息丢失

3. **可配置的残差块链**:
    - 功能：控制每个分支的特征提取深度
    - 核心思路：每个分支内嵌入可配置数量（NRBL/NRBM/NRBH）的残差块，每个块由conv-LeakyReLU-conv加残差连接构成
    - 设计动机：通过消融实验找到不同数据集的最优配置（MaSTr1325: 7/7/1, Oil Spill: 6/7/4），使参数与精度达到最佳平衡

### 损失函数 / 训练策略

- 使用Focal Loss作为损失函数，在两个数据集上均优于Dice Loss和CE+Dice组合
- 使用Adam优化器，batch size为8，训练300个epoch
- 在NVIDIA TESLA P100上训练，推理使用NVIDIA 2080和Intel 4-core XEON CPU

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文(LEMMA) | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| MaSTr1325 | mIoU | 98.97% | 99.91% (BEMRF-Net) | -0.94%（但参数少71x） |
| MaSTr1325 | 参数量 | 1.07M | 71.4M (WaSR-T) | 减少66.7x |
| MaSTr1325 | GFLOPs | 17.83 | 156.0 (BEMRF-Net) | 减少88.5% |
| MaSTr1325 | 推理时间 | 7.3ms | 47.55ms (DeepLabv3) | 减少84.65% |
| Oil Spill | mIoU | 93.42% | 92.66% (R-GSSNet) | +0.76% |
| Oil Spill | 参数量 | 1.01M | 62.6M (R-Segformer) | 减少62x |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 残差块 7/7/1 (MaSTr1325) | mIoU 98.96% | 最优配置，增加HFB块数反而降低性能 |
| 残差块 6/7/4 (Oil Spill) | mIoU 93.42% | 最优配置 |
| Focal Loss vs Dice Loss | 98.97% vs 98.72% | Focal Loss在两个数据集上均最优 |
| Focal Loss vs CE+Dice | 98.97% vs 98.86% | 验证Focal Loss的优势 |

### 关键发现

- LEMMA在参数量仅1M左右的情况下，可以与拥有数千万参数的模型（如WaSR-T的71.4M）性能相当
- 模型在USV地面视角（MaSTr1325）和无人机航拍视角（Oil Spill）两种截然不同的视角下均表现优异，展示了跨平台鲁棒性
- HFB使用16个通道就足够完成高分辨率掩码重建，这是降低计算量的关键设计
- 拉普拉斯金字塔能隐式抑制低频光照漂移（如太阳眩光、水面反射）

## 亮点与洞察

- 将传统图像处理技术（拉普拉斯金字塔）与深度学习残差网络巧妙结合，用物理先验减少学习负担
- 极致的轻量化：1M参数即可达到接近SOTA的精度，适合在无人机/USV等资源受限设备上实时部署
- 跨平台通用性好：同一个架构既适用于地面USV障碍物检测，也适用于航拍油污分割
- 不需要ImageNet预训练，从头训练即可达到高性能

## 局限与展望

- 反射/波浪/眩光等环境因素会影响拉普拉斯金字塔的质量，导致失败（论文展示了反射导致的失败案例）
- 当前使用固定金字塔层数和静态残差块配置，未来可探索自适应金字塔深度分配
- 数据集规模有限（MaSTr1325仅1325张，Oil Spill仅847张），难以验证在大规模场景下的泛化能力
- 与WaSR-T等最强模型在精度上仍有约1%的差距

## 相关工作与启发

- **vs WaSR-T**: WaSR-T使用Transformer达到99.80% mIoU，但需要71.4M参数；LEMMA在1.07M参数下达98.97%，效率提升数十倍
- **vs DeepLabv3**: DeepLabv3达97.67% mIoU需48M参数和123 GFLOPs；LEMMA以1/45参数量超越其性能
- **vs LETNet**: 同为轻量模型，LETNet 83.18% mIoU，LEMMA用相近参数量（1.07M vs 0.94M）提升了近16个百分点
- **启发**: 传统CV技术（金字塔、边缘检测）与深度学习结合可以在特定领域实现极致轻量化

## 评分

- 新颖性: ⭐⭐⭐ 拉普拉斯金字塔用于分割不算全新，但在海洋场景的落地和三分支设计有新意
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、大量baselines对比、详细消融实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机明确，实验分析充分
- 价值: ⭐⭐⭐⭐ 对边缘设备部署的海洋分割有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] HFP-SAM: Hierarchical Frequency Prompted SAM for Efficient Marine Animal Segmentation](../../CVPR2025/segmentation/hfp-sam_hierarchical_frequency_prompted_sam_for_efficient_marine_animal_segmenta.md)
- [\[CVPR 2026\] MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention](mixercseg_an_efficient_mixer_architecture_for_crack_segmentation_via_decoupled_m.md)
- [\[CVPR 2026\] MPM: Mutual Pair Merging for Efficient Vision Transformers](mpm_mutual_pair_merging_for_efficient_vision_transformers.md)
- [\[CVPR 2026\] Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training](data_warmup_complexity-aware_curricula_for_efficient_diffusion_training.md)
- [\[CVPR 2026\] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgb-d_scene_understanding_via_multi-task_adaptive_learning_and_cross-d.md)

</div>

<!-- RELATED:END -->
