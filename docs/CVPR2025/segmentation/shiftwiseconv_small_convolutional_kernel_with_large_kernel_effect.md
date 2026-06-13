---
title: >-
  [论文解读] ShiftwiseConv: Small Convolutional Kernel with Large Kernel Effect
description: >-
  [CVPR 2025][语义分割][大卷积核] 本文揭示大卷积核的有效性可解耦为"特定粒度的特征提取"和"多路径特征融合"两个因素，据此提出ShiftwiseConv（SW Conv）——一个使用标准3×3卷积通过空间移位和多路径连接来模拟大卷积核效果的即插即用CNN模块，在分类、检测、分割等任务上超越了SLaK和UniRepLKNet等大核CNN以及多种Transformer架构。
tags:
  - "CVPR 2025"
  - "语义分割"
  - "大卷积核"
  - "小卷积替代"
  - "移位操作"
  - "CNN架构设计"
  - "长距离依赖"
---

# ShiftwiseConv: Small Convolutional Kernel with Large Kernel Effect

**会议**: CVPR 2025  
**arXiv**: [2401.12736](https://arxiv.org/abs/2401.12736)  
**代码**: [https://github.com/lidc54/shift-wiseConv](https://github.com/lidc54/shift-wiseConv)  
**领域**: 图像分割  
**关键词**: 大卷积核, 小卷积替代, 移位操作, CNN架构设计, 长距离依赖

## 一句话总结

本文揭示大卷积核的有效性可解耦为"特定粒度的特征提取"和"多路径特征融合"两个因素，据此提出ShiftwiseConv（SW Conv）——一个使用标准3×3卷积通过空间移位和多路径连接来模拟大卷积核效果的即插即用CNN模块，在分类、检测、分割等任务上超越了SLaK和UniRepLKNet等大核CNN以及多种Transformer架构。

## 研究背景与动机

Vision Transformer（ViT）因其优越的长距离建模能力在多项视觉任务上超越了传统CNN。为了吸收ViT的优势，ConvNeXt等工作通过增大卷积核来增强长距离依赖建模，后续RepLKNet（31×31）、SLaK（51×5）、UniRepLKNet（13×13）进一步推高了大核CNN的性能。

然而，研究者观察到一个关键现象：**简单增大卷积核尺寸带来的性能收益已出现边际递减甚至停滞**。要继续提升性能需要大量精心设计的trick。这暗示大核的有效性不仅仅来自"更大的感受野"。

本文受人眼视网膜结构启发——光感受器细胞通过多种路径将视觉信号传递给神经节细胞——提出了一个新视角：**大核卷积的关键因素可解耦为两个独立组件：(1) 以特定粒度提取基础特征，(2) 通过多路径连接融合特征**。基于这一洞察，作者证明了标准3×3卷积配合空间移位和多路径融合可以完全替代大核卷积并进一步提升性能。

## 方法详解

### 整体框架

ShiftwiseConv建立在SLaK架构之上，将其M×N的条形大卷积替换为多个N×N（默认3×3）的分组卷积，通过空间移位操作将这些小卷积的输出在空间维度排列叠加，模拟大卷积的感受野。辅以多路径特征融合（多edge）、重参数化（Rep）和粗粒度剪枝策略，形成完整的即插即用模块。

### 关键设计

1. **空间堆叠替代实验（小卷积→大卷积等价）**:
    - 功能：证明M×N大卷积可以用多个N×N小卷积在空间上堆叠来等价替代
    - 核心思路：将M×N条形卷积替换为分组卷积（组数=输入通道C，输出通道=⌈M/N⌉），对每个小卷积输出按序号施加空间偏移，使其覆盖大卷积的对应位置。使用SLaK预训练参数，替代后推理精度不变（82.5%）
    - 设计动机：这个关键实验直接验证了大核卷积可以被"小核+偏移"等价替代，为后续设计提供了基础

2. **多路径特征移位融合（Multi-edge Feature Shift）**:
    - 功能：提升特征图的利用率，模拟多样化的长距离连接
    - 核心思路：分析发现单一移位路径（edge）下特征图利用率低且可预测。通过引入多条边（edge），每条边采用不同的通道顺序映射（打乱通道排列），显著提高覆盖率。默认使用4条edge。当edge数增加且通道顺序随机化时，利用率从约35%提升到接近100%
    - 设计动机：类比视网膜中光感受器通过多种路径连接到神经节细胞，多路径融合增加了特征交互的多样性

3. **消除冗余与参数优化**:
    - 功能：精简架构，减少参数量同时保持或提升性能
    - 核心思路：
        - 将SLaK的两个条形卷积分支合并为共享卷积输出（反向偏移保持差异性），参数减半
        - 使用Ghost-like方法：设比例G让部分通道直接跳过大卷积（G=0.23抵消SLaK的1.3倍宽度扩展）
        - 将卷积核从5×5进一步缩小到3×3（#7实验精度反而略有提升81.44% vs 81.34%，说明更细粒度更有益）
        - 重参数化分支的BN从卷积后移到移位操作后（因移位分支间差异大于Rep分支间差异）
    - 设计动机：大核卷积中边缘position的滑动窗口大量覆盖padding区域，去除冗余可同时减少参数和改善性能

### 损失函数 / 训练策略

- 继承SLaK的训练设定：先120 epochs探索超参数趋势，再300 epochs完整训练
- 使用粗粒度剪枝（prune-and-grow策略）：对filter级别按参数绝对值总和排序剪枝，而非SLaK的细粒度逐元素剪枝
- 稀疏掩码共享频率：减少不同Rep分支间掩码同步频率可提升性能（允许探索不同filter组合）
- 架构超参数采用UniRepLKNet的深度优先策略（[3,3,18,3]块配置），优于SLaK的宽度优先策略

## 实验关键数据

### 主实验

ImageNet-1K分类：

| 方法 | 类型 | Params(M) | FLOPs(G) | Top-1 Acc(%) |
|------|------|-----------|----------|-------------|
| SW-tiny | CNN | 31 | 5.0 | **83.4** |
| UniRepLKNet-T | CNN | 31 | 4.9 | 83.2 |
| SLaK-T | CNN | 30 | 5.0 | 82.5 |
| SwinV2-T | Transformer | 28 | 6 | 81.8 |
| SW-small | CNN | 56 | 9.4 | **83.9** |
| UniRepLKNet-S | CNN | 56 | 9.1 | 83.9 |
| SLaK-S | CNN | 55 | 9.8 | 83.8 |

COCO目标检测（Cascade Mask R-CNN）：

| 方法 | Params(M) | AP^box | AP^mask |
|------|-----------|--------|---------|
| SW-tiny | 87 | **52.21** | **45.19** |
| UniRepLKNet-T | 89 | 51.8 | 44.9 |
| SLaK-T | - | 51.3 | 44.3 |

ADE20K语义分割（UPerNet）：

| 方法 | Params(M) | mIoU(SS) | mIoU(MS) |
|------|-----------|----------|----------|
| SW-tiny | 62 | **49.22** | **50.06** |
| UniRepLKNet-T | 61 | 48.6 | 49.1 |
| SLaK-T | 65 | 47.6 | - |
| SW-small | 88 | 49.79 | 50.83 |
| UniRepLKNet-S | 86 | **50.5** | **51.0** |

### 消融实验

从SLaK到SW的渐进演化（120 epochs, ImageNet-1K）：

| 配置 | Acc(%) | 说明 |
|------|--------|------|
| #0 SLaK-tiny | 81.6 | 基线(51×5大核) |
| #1 laid-out(train) | 82.27 | 小核空间堆叠替代 |
| #3 SW-(pad=N//2) | 81.26 | 合并分支+统一padding |
| #4 Rep×2 | 81.52 | 加入重参数化 |
| #7 N=5→3 | 81.44 | 3×3核(更细粒度) |
| #11 rep2 E4 | 81.82 | 2Rep+4Edge |
| #15 rep2 E4 mean | 81.94 | 优化初始稀疏度 |
| #19 architecture | 82.25 | UniRepLKNet架构超参 |
| #20 +SE | 82.27 | 加SE模块 |

### 关键发现

- 3×3卷积不仅能替代大核，更细粒度还略有优势（#6 81.34% → #7 81.44%）
- 多路径融合（多edge）比单路径带来显著提升，但多Rep分支的边际收益与多edge有重叠效果
- 深度优先策略（UniRepLKNet风格）优于宽度优先策略（SLaK风格），验证了VGG的设计哲学
- 数据驱动的稀疏分析显示：更深的层倾向于剪枝更多filter（信息传递为主），每阶段最后一层剪枝最多（阶段过渡）

## 亮点与洞察

- **元洞察：大核的真正因素**：将大核解耦为"粒度提取"和"多路径融合"，这个视角突破了对大核尺寸的固有认知，为CNN设计提供新方向
- **VGG回归**：用3×3小核替代大核的结论与VGG的设计哲学一脉相承，但在现代CNN架构中赋予了新含义——不是简单堆叠深度，而是空间堆叠+多路径
- **即插即用设计**：SW Conv可直接替换现有架构中的大核卷积，无需修改整体架构
- **数据驱动的稀疏结构分析**：通过分析粗粒度剪枝后的模式，发现层间稀疏度的规律，可指导未来架构设计
- **系统化的演化实验**：从#0到#20的渐进实验方法论值得学习，每一步的改变和效果都清晰可追踪

## 局限与展望

- 仅展示了tiny和small规模的结果，scaling到base/large模型时的超参数搜索成本较高
- 等价大核尺寸继承自SLaK（51×3），这些尺寸是否对SW也最优需要进一步探索
- 推理速度虽通过重参数化优化，但多edge的内存IO开销需关注
- 粗粒度剪枝在通用硬件上更友好，但剪枝率与模型架构的交互关系还需深入分析
- 未来可探索自适应的edge数量和连接模式，以及与注意力机制的更深入融合

## 相关工作与启发

- **vs SLaK**: SLaK使用51×5条形大核+细粒度稀疏，本文证明可用3×3核+空间移位达到更好效果，参数量几乎减半
- **vs UniRepLKNet**: UniRepLKNet使用13×13膨胀卷积+SE，本文的SW-tiny在相同架构下以3×3核超越其精度，且提供了对大核设计更本质的理解
- **vs VAN（Visual Attention Network）**: VAN通过多层堆叠小核模拟大核，是深度维度的堆叠；SW是宽度（空间）维度的堆叠，更高效地扩展有效感受野

## 评分

- 新颖性: ⭐⭐⭐⭐ 从解耦视角理解大核效应并用小核重构，观点新颖且有说服力
- 实验充分度: ⭐⭐⭐⭐ 覆盖分类/检测/分割/3D检测，渐进消融非常详细
- 写作质量: ⭐⭐⭐ 逻辑清晰但实验编号系统较复杂，图表较密集
- 价值: ⭐⭐⭐⭐ 为CNN架构设计提供了新范式，3×3替代大核的结论有重要启示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Scale Efficient Training for Large Datasets](scale_efficient_training_for_large_datasets.md)
- [\[CVPR 2025\] F-LMM: Grounding Frozen Large Multimodal Models](f-lmm_grounding_frozen_large_multimodal_models.md)
- [\[CVPR 2025\] StoryGPT-V: Large Language Models as Consistent Story Visualizers](storygpt-v_large_language_models_as_consistent_story_visualizers.md)
- [\[NeurIPS 2025\] FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning](../../NeurIPS2025/segmentation/finers_fine-grained_reasoning_and_segmentation_of_small_objects_with_reinforceme.md)
- [\[NeurIPS 2025\] Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning](../../NeurIPS2025/segmentation/fast_and_fluent_diffusion_language_models_via_convolutional_decoding_and_rejecti.md)

</div>

<!-- RELATED:END -->
