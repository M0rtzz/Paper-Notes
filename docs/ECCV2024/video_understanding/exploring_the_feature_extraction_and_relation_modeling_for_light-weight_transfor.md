---
title: >-
  [论文解读] Exploring the Feature Extraction and Relation Modeling For Light-Weight Transformer Tracking
description: >-
  [ECCV 2024][视频理解][轻量级跟踪] 本文提出FERMT（Feature Extraction and Relation Modeling Tracker），通过将one-stream tracker中的注意力机制分解为四个功能不同的子模块——浅层专注特征提取、深层专注关系建模——并引入双注意力单元进行特征预处理，在GOT-10k上以69.6%的AO分数超越领先实时跟踪器5.6%，同时CPU速度提升54%。
tags:
  - ECCV 2024
  - 视频理解
  - 轻量级跟踪
  - Transformer
  - 特征提取
  - 关系建模
  - 注意力分解
---

# Exploring the Feature Extraction and Relation Modeling For Light-Weight Transformer Tracking

**会议**: ECCV 2024  
**arXiv**: 无  
**代码**: [GitHub](https://github.com/KarlesZheng/FERMT)  
**领域**: 目标跟踪  
**关键词**: 轻量级跟踪, Transformer, 特征提取, 关系建模, 注意力分解

## 一句话总结

本文提出FERMT（Feature Extraction and Relation Modeling Tracker），通过将one-stream tracker中的注意力机制分解为四个功能不同的子模块——浅层专注特征提取、深层专注关系建模——并引入双注意力单元进行特征预处理，在GOT-10k上以69.6%的AO分数超越领先实时跟踪器5.6%，同时CPU速度提升54%。

## 研究背景与动机

1. **领域现状**: 基于Transformer的轻量级目标跟踪器近年来在多个基准测试中树立了新标准，以其效率和有效性兼备的优势受到广泛关注。one-stream架构（将模板和搜索区域作为统一输入处理）逐渐成为轻量级跟踪的主流范式，避免了two-stream架构中冗余的特征交互。

2. **现有痛点**: 当前大多数轻量级跟踪器直接复用现有的目标检测网络架构作为backbone，没有针对目标跟踪这一特定任务优化网络结构。目标跟踪与目标检测在任务需求上存在根本差异——跟踪需要同时处理模板匹配（关系建模）和目标外观编码（特征提取），而检测只关注后者。这种忽视导致了体系结构设计上的次优。

3. **核心矛盾**: 在one-stream跟踪器中，每层的自注意力机制同时承担特征提取和模板-搜索关系建模两个功能，但这两个功能在网络不同深度的重要性是不同的。浅层应侧重于提取局部低级特征，深层应侧重于建模模板与搜索区域之间的语义关联。统一的注意力机制无法有效分配计算资源。

4. **本文目标**: 设计一个专门针对目标跟踪任务优化的轻量级backbone，在不同网络深度对特征提取和关系建模功能进行差异化设计，实现速度和精度的双重提升。

5. **切入角度**: 将标准注意力机制分解为四个子模块，根据网络深度差异化配置：浅层使用高效的局部特征提取模块，深层使用注重全局交互的关系建模模块。同时引入双注意力单元（Dual Attention Unit）进行通道级和空间级的特征预处理。

6. **核心 idea**: 对one-stream跟踪器的注意力机制按功能分解、按深度差异化配置，让网络每一层都做最适合它的工作。

## 方法详解

### 整体框架

FERMT是一个端到端的one-stream轻量级跟踪器。输入为模板图像和搜索区域的拼接序列，输出为目标的边界框预测。网络由多层堆叠的功能分解Transformer块组成，前若干层使用特征提取优化模块，后若干层使用关系建模优化模块。在进入主干网络之前，输入特征首先经过Dual Attention Unit进行全局通道和空间注意力预处理。

### 关键设计

1. **注意力机制四模块分解**: 作者将标准的Multi-Head Self-Attention分解为四个功能子模块：（a）模板自注意力——模板token之间的特征交互；（b）搜索区域自注意力——搜索区域token之间的特征交互；（c）模板到搜索的交叉注意力——将模板信息注入搜索区域；（d）搜索到模板的交叉注意力——将搜索区域信息反馈给模板。基于这种分解，浅层仅保留（a）和（b）两个自注意力模块（专注特征提取），而深层则四个模块全部启用（加入关系建模）。这种设计的关键洞察是：浅层网络处理的是低级视觉特征，此时模板和搜索区域之间的交叉注意力不仅计算浪费，还可能引入噪声；而深层网络拥有足够的语义信息来进行有意义的关系建模。

2. **Dual Attention Unit (DAU)**: 在backbone之前引入的特征预处理模块，由通道注意力和空间注意力两个分支组成。通道注意力通过全局平均池化和全连接层学习通道间的依赖关系，对不同特征通道进行自适应加权；空间注意力则通过对特征图进行空间维度的注意力计算，增强目标区域的特征响应。DAU的目的是在特征进入主体Transformer之前就建立全局的通道交互和空间先验，为后续的特征提取和关系建模提供更丰富的输入表示。

3. **轻量化backbone设计**: 整体网络结构在保持高精度的同时注重轻量化。通过将计算密集的交叉注意力操作限制在深层，大幅减少了浅层的计算开销。同时，使用高效的depth-wise卷积和group normalization替代传统的全连接和batch normalization，进一步降低参数量和计算复杂度。这些设计使得FERMT在CPU上的跟踪速度比同等精度的其他方法快54%。

### 损失函数 / 训练策略

- 跟踪头使用标准的目标检测损失组合：L1回归损失 + GIoU损失用于边界框回归
- 分类分支使用focal loss处理正负样本不均衡问题
- 训练数据包括LaSOT, GOT-10k, COCO, TrackingNet等常用跟踪数据集
- 使用AdamW优化器，配合余弦退火学习率调度
- 数据增强包括随机裁剪、水平翻转、颜色抖动等

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文FERMT | 之前最佳实时跟踪器 | 提升 |
|--------|------|-----------|-------------------|------|
| GOT-10k | AO | **69.6%** | 64.0% | +5.6% |
| GOT-10k | SR₇₅ | - | - | 显著提升 |
| LaSOT | AUC | - | - | 竞争力 |
| TrackingNet | AUC | - | - | 竞争力 |

### CPU跟踪速度对比

| 方法 | CPU速度 | 相对提升 |
|------|---------|---------|
| 之前领先实时跟踪器 | 基线 | - |
| FERMT | **+54%** | CPU速度提升54% |

### 消融实验

| 配置 | GOT-10k AO | 说明 |
|------|-----------|------|
| 全层统一注意力 | 基线 | 标准one-stream设计 |
| + 注意力分解(浅层仅自注意力) | 显著提升 | 验证了功能分解的有效性 |
| + Dual Attention Unit | 进一步提升 | DAU的特征预处理提供了有益的先验 |
| 去除DAU通道注意力 | 下降 | 通道注意力对特征表示很重要 |
| 去除DAU空间注意力 | 下降 | 空间注意力对目标定位有帮助 |

### 关键发现

- 浅层使用轻量的自注意力（只做特征提取）、深层引入交叉注意力（做关系建模）的分解策略显著优于全层统一注意力
- DAU的两个分支（通道和空间）都对最终性能有独立贡献
- 在精度大幅提升的同时，CPU速度不降反升，证明了功能分解减少冗余计算的效果
- 网络深度与功能分配的对应关系（浅层→特征提取，深层→关系建模）被消融实验充分验证

## 亮点与洞察

1. **任务驱动的架构设计**: 不是简单地缩小现有检测架构，而是从跟踪任务的本质需求出发重新思考每一层应做什么，这种设计哲学值得借鉴
2. **功能分解的简洁性**: 将注意力分为四个子模块、按深度配置的方法概念简单但效果显著，体现了"简单有效"的设计美学
3. **速度-精度帕累托前沿**: 在轻量级跟踪领域同时推进精度和速度两个维度非常难得，说明冗余计算的消除带来了真实的效率提升
4. **网络深度与功能对齐的洞察**: 浅层提取特征、深层建模关系的发现对其他视觉任务的网络设计也有参考价值

## 局限与展望

1. 消融实验中各层深度划分的选择（几层做特征提取、几层做关系建模）可能依赖于手动调优，自动搜索最优划分方案是改进方向
2. 目前主要针对RGB单模态跟踪，多模态（RGB-D, RGB-T）场景下的适用性未验证
3. 长期跟踪中的模板更新策略未重点讨论，这对实际部署至关重要
4. 实验对比主要集中在实时/轻量级跟踪器之间，与大模型跟踪器（如OSTrack-384等）的差距需明确量化
5. 在遮挡、快速运动等极端场景下的鲁棒性分析可以更深入

## 相关工作与启发

- **OSTrack (Ye et al., 2022)**: one-stream跟踪的代表工作，统一处理模板和搜索区域，FERMT在其基础上进一步优化了注意力机制的功能分配
- **HiT (Kang et al., 2023)**: 轻量级跟踪器的最新进展，但仍基于通用backbone
- **MixFormer (Cui et al., 2022)**: 混合注意力跟踪，关注模板-搜索的交互但未考虑深度差异化
- 启发：在其他视觉任务（如视频目标分割、动作识别）中也可以考虑按网络深度差异化分配不同功能模块

## 评分

- **新颖性**: ⭐⭐⭐⭐ 对注意力机制进行功能分解并按深度差异化配置的思路在跟踪领域是新颖的
- **实验充分度**: ⭐⭐⭐⭐ 消融实验详尽，在多个基准上进行了验证，速度对比有说服力
- **写作质量**: ⭐⭐⭐⭐ 动机分析清晰，设计逻辑循证有据
- **价值**: ⭐⭐⭐⭐ 为轻量级跟踪器的backbone设计提供了新思路，在实时跟踪领域具有很强的实用价值

<!-- RELATED:START -->

## 相关论文

- [Towards Efficient General Feature Prediction in Masked Skeleton Modeling](../../ICCV2025/video_understanding/towards_efficient_general_feature_prediction_in_masked_skeleton_modeling.md)
- [Elysium: Exploring Object-Level Perception in Videos via MLLM](elysium_exploring_objectlevel_perception_in_videos_via_mllm.md)
- [General Compression Framework for Efficient Transformer Object Tracking](../../ICCV2025/video_understanding/general_compression_framework_for_efficient_transformer_object_tracking.md)
- [Data Collection-Free Masked Video Modeling](data_collection-free_masked_video_modeling.md)
- [Local All-Pair Correspondence for Point Tracking](local_all-pair_correspondence_for_point_tracking.md)

<!-- RELATED:END -->
