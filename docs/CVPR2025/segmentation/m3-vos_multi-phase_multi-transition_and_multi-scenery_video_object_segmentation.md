---
title: >-
  [论文解读] M3-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation
description: >-
  [CVPR 2025][图像分割][视频目标分割] 本文引入"物相"（Phase）概念到视频目标分割任务中，构建了包含479个视频、205K掩码、覆盖6种相态和23种相变的M3-VOS基准，并提出即插即用的ReVOS方法通过逆向传播精炼来改善相变物体的分割性能。
tags:
  - CVPR 2025
  - 图像分割
  - 视频目标分割
  - 物相转变
  - 逆向传播
  - 多相态
  - 基准数据集
---

# M3-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation

**会议**: CVPR 2025  
**arXiv**: [2412.13803](https://arxiv.org/abs/2412.13803)  
**代码**: https://zixuan-chen.github.io/M-cube-VOS.github.io/  
**领域**: 视频分割 / 视频目标分割  
**关键词**: 视频目标分割, 物相转变, 逆向传播, 多相态, 基准数据集

## 一句话总结
本文引入"物相"（Phase）概念到视频目标分割任务中，构建了包含479个视频、205K掩码、覆盖6种相态和23种相变的M3-VOS基准，并提出即插即用的ReVOS方法通过逆向传播精炼来改善相变物体的分割性能。

## 研究背景与动机

**领域现状**：视频目标分割（VOS）已经取得了显著进展，DAVIS、YouTubeVOS等基准推动了XMem、Cutie、SAM2等高性能方法的发展。主流方法基于外观匹配和记忆库机制，在标准场景中表现优异。

**现有痛点**：现有的VOS基准和方法几乎都集中在单一相态（通常是固态）的物体上，对物体发生相变时的分割能力严重不足。例如，冰融化为水、干冰升华为气体、液体沸腾蒸发等过程中，物体的外观、形态和边界会发生剧烈变化。基于外观匹配的方法在这些场景下会严重失效——它们无法追踪一个形态完全改变的物体。VOST和VSCOS虽然关注了物体外观变化，但仍局限于单一相态内的变化。

**核心矛盾**：物体的相变是自然界中极其常见的现象（烹饪、工业制造、实验室操作等），但计算机视觉社区几乎完全忽略了对物相转变的理解。根本原因在于：(1) 缺乏覆盖多种相变过程的标注数据集；(2) 当前基于外观先验的模型架构本身就不适合处理外观发生根本性变化的场景。

**本文目标**：(1) 系统性地定义物相分类和相变类型；(2) 构建覆盖多相态、多相变、多场景的VOS基准；(3) 提出改善相变物体分割的方法。

**切入角度**：作者从物理学/化学中借鉴了"相态"概念，但采用宏观视觉特征（而非微观分子间距）来定义相态。一个关键观察是：物体在相变过程中，其"无序度"（entropy/disorder）往往是递增的——视频前半部分的mask比后半部分更规则。如果从视频末端反向传播，则是一个从高无序到低无序的过程，可能更容易分割。

**核心 idea**：利用相变过程中无序度递增的物理先验，提出逆向传播（reverse propagation）机制来精炼前向传播的分割结果。

## 方法详解

### 整体框架
M3-VOS包含数据集和方法两部分。数据集部分系统定义了固态（颗粒/刚体/柔体）、液态（粘性/非粘性）、气溶胶/气体三大相态、6个子类、23种相变类型，收集了479个高分辨率视频并提供30fps的密集掩码标注。方法部分提出ReVOS——一个即插即用的框架，可以叠加在任何基于掩码传播的VOS骨干网络（如Cutie）之上。ReVOS在前向传播完成后，从视频末端反向传播预测掩码，再通过Readout Fusion模块融合正反两个方向的特征来生成最终掩码。

### 关键设计

1. **相态分类与相变体系（Phase Taxonomy）**:

    - 功能：为VOS任务提供系统性的物体状态描述框架
    - 核心思路：从宏观视觉特征出发，将日常物体分为三大相态：固态（进一步分为颗粒状、刚体、柔体）、液态（粘性/非粘性）和气溶胶/气体。相变分为相内变换（intra-phase，如液体流动、固体断裂、气体扩散）和跨相变换（cross-phase，如凝固、融化、升华、蒸发、溶解等10种）。共计23种具体相变类型
    - 设计动机：现有VOS基准缺乏对物体相态属性的描述，无法评估模型对相变的理解能力。系统化的分类为基准设计和性能分析提供了结构化框架

2. **逆向传播机制（Reverse Memory）**:

    - 功能：利用信息熵递减的逆向过程来改善分割精度
    - 核心思路：作者用LBP（局部二值模式）计算掩码的纹理熵 $h_{LBP}$，发现多数VOS数据集中视频后半部分的掩码熵高于前半部分（如M3-VOS中4.72 vs 4.68），验证了"正向传播越来越难、逆向传播越来越简单"的假设。在实现上，完成正向传播后，取最后一帧的预测掩码作为起点，在一个大小为 $T$ 的滑动窗口内执行逆向传播，维护一个专门的逆向working memory来存储高分辨率特征
    - 设计动机：正向传播过程中，物体发生相变导致外观剧变，使得掩码预测逐步退化。逆向传播则经历了一个熵减过程，可以补偿前向传播的信息损失

3. **Booster模块**:

    - 功能：增强前向传播最后一帧掩码的覆盖范围，为逆向传播提供更好的起点
    - 核心思路：在前向传播过程中，通过引入一个放大因子 $\alpha$ 对decoded logits进行缩放，公式为 $M = \sigma(\alpha \cdot X_{decode})$，其中 $\sigma$ 是sigmoid函数。这相当于降低了mask预测的阈值，使更多可能属于目标的区域被包含进来，减少漏检
    - 设计动机：如果前向传播在最后一帧已经严重丢失了目标信息，那么逆向传播也无法恢复。Booster通过放宽检测标准确保最后一帧掩码覆盖尽可能完整，即使引入少量假阳性也好过丢失大片目标区域

### 损失函数 / 训练策略
冻结Cutie骨干网络的所有参数，仅训练Readout Fusion模块。使用AdamW优化器，学习率1e-5，总共训练75K迭代，在60K和67.5K时学习率衰减10倍。在4张A100 GPU上训练约10小时。训练数据包含YouTubeVOS、DAVIS、BURST、OVIS和MOSE五个数据集。

## 实验关键数据

### 主实验

| 方法 | M3-VOS Full $\mathcal{J}$ | M3-VOS Core $\mathcal{J}$ | VOST $\mathcal{J}$ | DAVIS'17 $\mathcal{J}$ |
|------|--------------------------|--------------------------|-------------------|---------------------|
| DeAOT | 72.5 | 65.2 | 82.7 | 86.2 |
| XMem | 70.4 | 61.5 | 82.9 | 85.4 |
| SAM2 | 69.5 | 57.8 | 85.5 | 85.2 |
| Cutie-base | 74.6 | 64.6 | 85.6 | 86.8 |
| **ReVOS (Ours)** | **75.6** | **66.5** | **86.0** | 86.8 |

### 消融实验

| 配置 | $\mathcal{J}$ | $\mathcal{J}_{tr}$ | $\mathcal{J}_{cc}$ | 说明 |
|------|------|---------|---------|------|
| 仅前向 (Cutie) | 74.7 | 64.6 | 64.5 | 基线 |
| 仅逆向 | 75.4 | 66.4 | 64.9 | 逆向单独已有提升 |
| +Readout Fusion | 75.7 | 66.4 | 65.3 | 融合正反向更优 |
| w/o Booster | 74.2 | 64.2 | 63.9 | 去掉Booster反而比基线差 |
| +Booster | 75.7 | 66.4 | 65.3 | Booster贡献关键 |

### 关键发现
- 所有现有方法在M3-VOS上的性能显著低于在DAVIS等传统基准上的表现，证明相变物体分割是一个真正未解决的挑战
- M3-VOS Core子集（平衡分布）上所有方法表现更差，说明现有模型存在scene bias和phase bias
- ReVOS的提升在跨相变(cross-phase)场景上最为显著，证明逆向传播对处理剧烈外观变化最有效
- 纯逆向传播已经优于纯前向传播，但融合两者可以取得最佳效果
- Booster模块至关重要——没有它，逆向传播的起点掩码不完整，反而拖累整体性能
- 逆向间隔 $L=60$ 时性能最好且FPS最高（21.6），因为跳步可以覆盖更大时间范围同时减少计算量

## 亮点与洞察
- **物相概念引入VOS**是一个非常新颖的切入角度，从物理学视角重新审视了VOS的难点。这一分类体系具有普适性，可以指导未来数据集和方法的设计
- **逆向传播的熵减动机**来自物理直觉但用实验验证了其有效性。这一思路可以迁移到其他序列预测任务——当正向过程的难度递增时，可以尝试逆向过程来补偿
- **即插即用设计**使ReVOS可以叠加在任何掩码传播骨干上，实用性很强

## 局限与展望
- 论文承认标注偏差不可完全消除，尤其是对于边界模糊的气态/液态物体
- ReVOS对小物体的改善不明显，甚至在某些情况下略有退化——因为小物体的Booster可能引入过多假阳性
- 逆向传播需要额外的计算开销，FPS从Cutie的30+降低到15-22
- 当前所有模型在M3-VOS上的性能都不高，未来可以考虑引入多模态大语言模型中的物理知识来辅助理解相变过程
- 数据集规模（479视频）相对有限，扩大数据集并纳入更多极端相变场景（如爆炸、化学反应）是有价值的方向

## 相关工作与启发
- **vs VOST**: VOST关注物体外观变化（如物体被使用后的状态变化），但仅限于固态物体，不涉及跨相变。M3-VOS在物相维度上是VOST的超集
- **vs SAM2**: SAM2在VOST上表现最好但在M3-VOS上反而不如Cutie，说明SAM2虽然数据量大但训练分布偏向固态物体，对液态/气态的泛化能力不足
- **vs DyeNet（双向传播）**: DyeNet需要高置信度掩码才能进行逆向传播，ReVOS通过Booster模块放宽了这一要求，且作为即插即用模块更加灵活

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 物相概念引入VOS是全新的视角，熵减逆向传播动机优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 数据集详尽，实验覆盖多个基准，消融全面，含详细的相变类型分析
- 写作质量: ⭐⭐⭐⭐ 概念定义清晰，但论文较长，部分内容略冗余
- 价值: ⭐⭐⭐⭐⭐ M3-VOS填补了VOS在物相变化方面的空白，ReVOS提供了有效的baseline

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] MammAlps: A Multi-view Video Behavior Monitoring Dataset of Wild Mammals in the Swiss Alps](mammalps_a_multi-view_video_behavior_monitoring_dataset_of_wild_mammals_in_the_s.md)
- [\[ICML 2025\] unMORE: Unsupervised Multi-Object Segmentation via Center-Boundary Reasoning](../../ICML2025/segmentation/unmore_unsupervised_multi-object_segmentation_via_center-boundary_reasoning.md)
- [\[ACL 2025\] Pixel-Level Reasoning Segmentation via Multi-turn Conversations](../../ACL2025/segmentation/pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)
- [\[ECCV 2024\] OLAF: A Plug-and-Play Framework for Enhanced Multi-object Multi-part Scene Parsing](../../ECCV2024/segmentation/olaf_a_plug-and-play_framework_for_enhanced_multi-object_multi-part_scene_parsin.md)
- [\[CVPR 2025\] MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation](mv-ssm_multi-view_state_space_modeling_for_3d_human_pose_estimation.md)

<!-- RELATED:END -->
