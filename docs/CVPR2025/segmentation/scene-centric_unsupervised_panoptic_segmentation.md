---
title: >-
  [论文解读] Scene-Centric Unsupervised Panoptic Segmentation
description: >-
  [CVPR 2025][图像分割][无监督全景分割] CUPS 是首个直接在场景中心图像（如自动驾驶场景）上训练的无监督全景分割方法，通过融合自监督视觉特征、立体深度和光流运动线索生成高质量伪标签，在 Cityscapes 上的 PQ 超越此前 SOTA U2Seg 9.4 个点。
tags:
  - CVPR 2025
  - 图像分割
  - 无监督全景分割
  - 场景中心
  - 伪标签
  - 运动分割
  - 深度引导
---

# Scene-Centric Unsupervised Panoptic Segmentation

**会议**: CVPR 2025  
**arXiv**: [2504.01955](https://arxiv.org/abs/2504.01955)  
**代码**: https://visinf.github.io/cups  
**领域**: 分割  
**关键词**: 无监督全景分割, 场景中心, 伪标签, 运动分割, 深度引导

## 一句话总结

CUPS 是首个直接在场景中心图像（如自动驾驶场景）上训练的无监督全景分割方法，通过融合自监督视觉特征、立体深度和光流运动线索生成高质量伪标签，在 Cityscapes 上的 PQ 超越此前 SOTA U2Seg 9.4 个点。

## 研究背景与动机

**领域现状**：全景分割将语义分割和实例分割统一为一个任务，目前主流方法依赖大量像素级标注。近年来，自监督表示学习（如 DINO）的进步推动了无监督分割的发展——STEGO 解决无监督语义分割，CutLER 解决无监督实例分割。

**现有痛点**：唯一的无监督全景分割方法 U2Seg 存在三大缺陷：(1) 依赖 MaskCut 生成实例伪标签，但 MaskCut 假设输入为「物体中心」图像（前景物体大且清晰），在复杂场景图像上表现极差（mask precision 仅 6.5%）；(2) 无法直接在场景中心目标数据集上训练，被迫使用大量伪类别绕开 thing/stuff 区分；(3) 使用低分辨率 STEGO 语义预测，在高分辨率场景数据上效果有限。

**核心矛盾**：现有无监督方法从「物体中心」数据出发，但真实应用（如自动驾驶）面对的是复杂的「场景中心」图像——物体小而密集、背景复杂、thing 和 stuff 混杂。物体中心假设与场景中心现实之间存在根本性 gap。

**本文目标**：构建首个能直接在场景中心图像上训练的无监督全景分割方法，同时解决高分辨率语义预测和运动物体实例发现两大关键问题。

**切入角度**：作者从格式塔心理学的感知组织原则出发——人类自然地基于相似性、不变性和共同命运来分组视觉元素。除了视觉外观线索外，深度提供空间三维信息，运动提供"物体是能移动的实体"这一物理先验，两者共同帮助在复杂场景中消歧。

**核心 idea**：利用立体视频提取深度引导语义信息 + 运动引导实例信息，融合生成高质量全景伪标签，再用伪标签引导和自训练策略训练全景网络。

## 方法详解

### 整体框架

CUPS 包含三个阶段：(1) 伪标签生成——从立体视频帧中提取场景流和深度，通过运动分割获得实例伪标签、通过深度增强推理获得语义伪标签，再融合为全景伪标签；(2) 全景引导——用伪标签配合 DropLoss 和 self-enhanced copy-paste 增强训练全景网络；(3) 全景自训练——用动量网络的多尺度/多翻转集成预测生成自标签进一步提升性能。输入为立体视频对（仅伪标签阶段），推理时仅需单目图像。

### 关键设计

1. **基于运动的实例伪标签生成**:

    - 功能：从立体视频中发现运动物体并生成高精度实例 mask
    - 核心思路：使用无监督光流模型 SMURF 估计前后向光流和视差，计算场景流 $\mathbf{F}$ 和一致性 mask $\mathbf{O}$。将场景流输入 SF2SE3 进行 SE(3) 刚体运动聚类。为解决 SF2SE3 随机初始化导致的不一致问题，运行 n 次取得 m 个潜在重叠 mask，通过一致性评分 $c_i$ 过滤掉在少于 80% 运行中出现的 mask，最后用矩阵 NMS 消除重叠，分离连通域得到最终高精度运动物体 mask
    - 设计动机：运动线索天然提供 thing 类别的物理先验（能移动的就是物体），比纯视觉特征在复杂场景中更可靠地发现实例；集成过滤策略有效提高伪标签精度（mask precision 从 MaskCut 的 6.5% 提升到 59.6%）

2. **深度引导语义伪标签生成**:

    - 功能：在高分辨率场景图像上生成高质量语义分割伪标签
    - 核心思路：基于 DepthG 蒸馏 DINO 特征获取低维语义表示，然后进行深度引导多分辨率融合推理。具体地，获取低分辨率预测 $\mathbf{P}^{low}$（全图缩放输入）和高分辨率预测 $\mathbf{P}^{high}$（滑动窗口拼接），用深度 $D$ 计算逐像素权重 $\alpha_{h,w} = (D_{h,w}+1)^{-1}$，按 $\mathbf{P}^* = \alpha \odot \mathbf{P}^{low} + (1-\alpha) \odot \mathbf{P}^{high}$ 融合。近处像素（深度小）以低分辨率预测为主，远处像素（深度大）以高分辨率预测为主
    - 设计动机：自监督模型在低分辨率上提取的特征对近处大物体更可靠，但对远处小物体需要高分辨率细节。深度天然编码了物体与相机的距离关系，作为分辨率选择的桥梁，巧妙解决了 SSL 特征分辨率受限的问题

3. **全景伪标签融合与 Thing/Stuff 自动区分**:

    - 功能：将语义和实例伪标签融合为统一的全景伪标签，同时自动区分 thing 和 stuff 类别
    - 核心思路：统计每个伪语义类别在实例 mask 内的像素占比与其全局出现频率的比值，高于阈值 $\psi^{ts}$ 的标为 thing 类，低于的标为 stuff 类。然后将实例 mask 内最频繁的语义 ID 赋给该实例；无实例 mask 的 thing 区域标为 ignore
    - 设计动机：U2Seg 因无法直接区分 thing/stuff 而使用大量伪类别绕弯，本文利用运动物体 mask 天然对应 thing 类别这一先验，通过简单统计实现自动判别

### 损失函数 / 训练策略

- **Stage 2 (引导学习)**：使用 DropLoss 只监督与伪 mask 有足够 IoU 重叠的 thing 预测（不惩罚无对应伪标签的预测，允许网络自行发现静止物体）+ 标准交叉熵损失用于语义（忽略 ignore 像素）+ Self-enhanced copy-paste 增强（将训练中逐渐可靠的模型预测 paste 到训练图像中）。4000 步 AdamW 优化
- **Stage 3 (自训练)**：维护 EMA 动量网络，对输入做翻转和多尺度增强生成集成预测，通过实例置信度阈值 $\gamma$ 和类别依赖语义阈值 $\zeta_k$ 过滤后得到自标签。仅训练 heads、冻结 norm 层。1500 步 AdamW 优化

## 实验关键数据

### 主实验

| 数据集 | 指标 | CUPS | U2Seg (prev SOTA) | 提升 |
|--------|------|------|----------|------|
| Cityscapes val | PQ | 27.8 | 18.4 | +9.4 |
| Cityscapes val | SQ | 57.4 | 55.8 | +1.6 |
| Cityscapes val | RQ | 35.2 | 22.7 | +12.5 |
| Cityscapes val | PQ_Th | 17.7 | 10.2 | +7.5 |
| Cityscapes val | PQ_St | 35.1 | 24.3 | +10.8 |
| KITTI | PQ | 25.5 | 20.6 | +4.9 |
| BDD | PQ | 19.9 | 15.8 | +4.1 |
| Waymo | PQ | 26.4 | 19.8 | +6.6 |
| MOTS (OOD) | PQ | 67.8 | 50.7 | +17.1 |

在无监督语义分割上，CUPS 的 mIoU 达到 26.8%，Acc 达到 83.2%，同样超越所有先前方法。

### 消融实验

| 配置 | PQ | 说明 |
|------|------|------|
| Full CUPS | 27.8 | 完整模型 |
| w/o 深度引导推理 | ~24 | 去掉深度自适应融合后语义质量下降 |
| w/o 运动实例 mask | ~20 | 没有运动线索无法获得可靠 thing 实例 |
| w/o 自训练 (Stage 3) | ~24 | 自训练带来显著提升 |
| w/o self-enhanced copy-paste | ~25 | copy-paste 增强帮助发现静止物体 |

### 关键发现

- 运动线索是 CUPS 最关键的贡献：用 MaskCut 替换运动分割后 mask precision 从 59.6% 骤降至 6.5%，证明物体中心方法在场景中心数据上完全失效
- 深度引导推理有效提升高分辨率场景的语义分割精度，尤其对远处小物体
- CUPS 在 OOD 数据集 MOTS 上泛化极好（PQ 67.8 vs U2Seg 50.7），说明运动+深度先验比纯视觉更具鲁棒迁移能力
- 与监督方法的差距从 U2Seg 时的 43.9 PQ gap 缩小为 34.5，减少了约 20%

## 亮点与洞察

- **深度引导分辨率自适应推理**是一个非常优雅的设计——利用深度作为"分辨率路由器"，自动让近处用低分辨率全局特征、远处用高分辨率局部特征，不需要额外参数就解决了 SSL 特征的分辨率瓶颈
- **将格式塔原则系统化地引入无监督分割**是一个有启发性的视角。运动=共同命运、深度=空间邻近和相似性，这些线索的组合可以推广到其他无监督场景理解任务
- **集成过滤策略**简单但极其有效——多次运行随机算法后通过一致性评分过滤，可以迁移到任何包含随机初始化的感知模块

## 局限与展望

- 依赖立体视频数据生成伪标签，限制了应用场景——单目视频或静态图像数据集无法使用此方法
- 运动线索只能发现"移动的"物体，静止的 thing（如停着的车）仍然依赖 bootstrapping 和自训练来发现，覆盖率有限
- 27 个伪类别是人为设定的超参数，换到不同数据集需要调整
- 改进方向：结合单目深度估计（如 DepthAnything）和视频光流替代立体视频依赖；利用 foundation model（如 SAM、DINOv2）进一步提升伪标签质量

## 相关工作与启发

- **vs U2Seg**: U2Seg 依赖 CutLER 的 MaskCut 生成实例 mask + STEGO 语义，两者都假设物体中心且低分辨率，无法处理场景中心数据。CUPS 用运动+深度完全绕过了这些限制，证明物理线索在场景理解中比纯视觉更鲁棒
- **vs CutLER/MaskCut**: MaskCut 在场景图像上的 precision 仅 6.5%（因为 Normalized Cut 在复杂场景中关注语义区域而非实例），CUPS 的运动分割达到 59.6%，差了 9 倍
- **vs DepthG**: CUPS 的语义模块基于 DepthG 但增加了深度引导多分辨率推理，在 Cityscapes 上 mIoU 从 23.1 提升到 26.8

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将运动+深度线索系统性地用于无监督全景分割，是一个重要的范式突破
- 实验充分度: ⭐⭐⭐⭐⭐ 6 个数据集评估 + 跨域泛化 + 子任务评估 + 消融实验非常完善
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，从动机到方法到实验环环相扣
- 价值: ⭐⭐⭐⭐ 将无监督全景分割推向实际可用的方向，特别是自动驾驶场景

<!-- RELATED:START -->

## 相关论文

- [Hierarchical Compact Clustering Attention (COCA) for Unsupervised Object-Centric Learning](hierarchical_compact_clustering_attention_coca_for_unsupervised_object-centric_l.md)
- [Revisiting Audio-Visual Segmentation with Vision-Centric Transformer](revisiting_audio-visual_segmentation_with_vision-centric_transformer.md)
- [DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime](../../CVPR2026/segmentation/dsflash_comprehensive_panoptic_scene_graph_generation_in_realtime.md)
- [Towards Generalizable Scene Change Detection](towards_generalizable_scene_change_detection.md)
- [SPADE: Spatial-Aware Denoising Network for Open-vocabulary Panoptic Scene Graph Generation](../../ICCV2025/segmentation/spade_spatial-aware_denoising_network_for_open-vocabulary_panoptic_scene_graph_g.md)

<!-- RELATED:END -->
