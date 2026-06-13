---
title: >-
  [论文解读] GSOT3D: Towards Generic 3D Single Object Tracking in the Wild
description: >-
  [ICCV 2025][3D视觉][3D单目标跟踪] 提出 GSOT3D，目前最大的通用3D单目标跟踪基准，包含620个多模态序列（点云+RGB+深度）覆盖54类物体，支持PC/RGB-PC/RGB-D三种3D跟踪任务，并提出渐进式时空跟踪器PROT3D以9DoF包围盒实现最优性能。 3D单目标跟踪(SOT)在智能驾驶、移动…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D单目标跟踪"
  - "通用跟踪"
  - "点云"
  - "多模态"
  - "基准数据集"
  - "9DoF"
---

# GSOT3D: Towards Generic 3D Single Object Tracking in the Wild

**会议**: ICCV 2025  
**arXiv**: [2412.02129](https://arxiv.org/abs/2412.02129)  
**代码**: [ailovejinx/GSOT3D](https://github.com/ailovejinx/GSOT3D)  
**领域**: 3D视觉  
**关键词**: 3D单目标跟踪, 通用跟踪, 点云, 多模态, 基准数据集, 9DoF

## 一句话总结

提出 GSOT3D，目前最大的通用3D单目标跟踪基准，包含620个多模态序列（点云+RGB+深度）覆盖54类物体，支持PC/RGB-PC/RGB-D三种3D跟踪任务，并提出渐进式时空跟踪器PROT3D以9DoF包围盒实现最优性能。

## 研究背景与动机

3D单目标跟踪(SOT)在智能驾驶、移动机器人和导航中有关键应用，但现有基准严重制约了通用3D跟踪的发展：

**目标类别极度有限**：KITTI仅包含8类，NuScenes包含23类，均局限于自动驾驶场景中的车辆和行人。这使得训练出的跟踪器缺乏泛化到日常物体的能力。

**场景单一**：现有数据集仅来自交通场景，无法覆盖室内/户外的多样环境（办公室、公园、家庭等），不适合通用跟踪器的训练和评估。

**自由度受限**：KITTI和NuScenes仅使用7DoF包围盒（4D姿态+3D尺寸），不包含完整的6D姿态（3D平移+3D旋转），无法精确描述任意姿态的物体。

**RGB-D 3D跟踪数据不足**：Track-it-in-3D虽然提供9DoF标注，但仅有300个序列36K帧，规模不足以训练深度3D跟踪器。

**核心洞察**：要推动通用3D跟踪，需要一个涵盖丰富类别、多样场景、多种模态且具有精密9DoF标注的大规模基准。GSOT3D以620序列、54类、123K帧填补了这一空白，并首次同时支持PC、RGB-PC、RGB-D三种3D跟踪任务。

## 方法详解

### GSOT3D基准构建

**数据采集平台**：基于Clearpath Husky A200移动机器人平台，搭载64线激光雷达(Ouster OS-64)、深度相机(OAK D-Pro)和RGB相机(FLIR)，传感器校准同步，20fps输出。

**类别设计**：10个元类（家具、人类、车辆、家居用品、办公用品、食物、动物、运动器材、玩具、杂项）下细分54个子类，覆盖日常生活中适合3D跟踪的物体。

**标注质量保证**：
- 每帧手动标注最紧凑的9DoF 3D包围盒（3D平移+3D旋转+3D尺寸）
- 多轮专家检查+标注者修正的迭代流程，确保标注精度
- 标注7种序列属性：不可见性(INV)、形变(DEF)、快速运动(FM)、旋转(ROT)、尺度变化(SV)、相似干扰物(SD)、稀疏性(SPA)

**数据集规模对比**：

| 基准 | 序列数 | 帧数 | 类别数 | 场景 | 支持的任务 |
|------|--------|------|--------|------|-----------|
| KITTI | 21 | 15K | 8 | 户外 | PC, RGB-PC |
| NuScenes | 1000 | 40K | 23 | 户外 | PC, RGB-PC |
| Track-it-in-3D | 300 | 36K | 44 | 室内外 | RGB-D |
| **GSOT3D** | **620** | **123K** | **54** | **室内外** | **PC, RGB-PC, RGB-D** |

**评估协议**：使用类别平均重叠度(mAO)和类别平均成功率(mSR@0.5/0.75)，基于3D IoU计算，不使用precision指标（因为中心点距离无法评估9DoF包围盒的尺寸和角度精度）。

### PROT3D跟踪器

PROT3D是一个面向3D-SOT_PC的类别无关跟踪器，核心是渐进式时空网络。

**整体流程**：
1. 使用共享骨干 $\Phi(\cdot)$ 提取当前帧特征 $\mathbf{x}^1_t$ 和历史K帧特征，拼接为记忆特征 $\mathbf{H}_{t-1}$
2. 通过多阶段渐进架构逐步精化搜索区域特征

**每阶段处理**（对阶段 $i$）：
- 时空Transformer融合：$\mathbf{F}^i_t = \text{SPT}(\mathbf{x}^i_t, \mathbf{H}_{t-1})$，包含cross-attention和self-attention
- MLP定位：$R^i_t = [C^i_t, M^i_t, S^i_t]$（目标中心、目标性mask、proposal分数）
- FPS采样+特征变换：$\mathbf{x}^{i+1}_t = \text{FTB}(\bar{C}^i_t, M^i_t) + \text{Conv1D}(S^i_t)$
- 精化后的特征传入下一阶段继续细化

**最终定位**：最后阶段输出后通过MLP预测9DoF包围盒参数和目标性得分：

$$\mathcal{R}_t = \text{MLP}(\mathbf{x}^{N+1}_t), \quad b_t = \mathcal{B}_t(h), \quad h = \arg\max_d \mathcal{S}(d)$$

**关键设计**：
- 渐进式特征精化：每阶段的搜索区域特征通过编码目标信息变得更具判别力
- 9DoF包围盒预测：预测中心平移偏移、角度偏移和尺寸偏移
- 多帧记忆：利用前K帧的时序信息增强鲁棒性

## 实验

### 主实验：3D-SOT_PC整体性能

| 跟踪器 | mAO(%) | mSR50(%) | mSR75(%) |
|--------|--------|----------|----------|
| P2B | 9.79 | 8.59 | 1.75 |
| BAT | 6.56 | 3.54 | 0.88 |
| M2-Track | 20.26 | 14.34 | 1.88 |
| MBPTrack | 20.54 | 16.55 | 2.57 |
| M3SOT | 17.40 | 12.47 | 1.74 |
| **PROT3D** | **21.97** | **19.76** | **5.22** |

**关键发现**：PROT3D在所有指标上领先，特别是mSR75提升最为显著（+2.65% vs MBPTrack），表明渐进式精化对高精度定位尤为有效。所有现有跟踪器在GSOT3D上性能大幅下降，证明通用3D跟踪的巨大挑战。

### 消融实验

| 消融项 | mAO(%) | mSR50(%) | mSR75(%) |
|--------|--------|----------|----------|
| 1阶段+7DoF | 19.86 | 15.16 | 2.36 |
| 1阶段+9DoF | 20.03 | 15.46 | 3.29 |
| **2阶段+9DoF (PROT3D)** | **21.97** | **19.76** | **5.22** |
| 3阶段+9DoF | 21.58 | 19.61 | 5.19 |
| 记忆K=2 | 21.37 | 19.52 | 5.32 |
| **记忆K=3** | **21.97** | **19.76** | **5.22** |
| 记忆K=4 | 21.84 | 19.69 | 5.17 |

**关键发现**：
- 9DoF vs 7DoF：mSR75从2.36%提升至3.29%（+39.4%），更精确的姿态估计显著提升高阈值成功率
- 渐进式架构（2阶段）将mSR50从15.46%提升至19.76%（+27.8%），但3阶段出现轻微过拟合
- 记忆大小K=3最优，过多历史帧可能引入噪声

### 与KITTI的对比

| 跟踪器 | KITTI mAO(%) | GSOT3D mAO(%) | 下降幅度 |
|--------|-------------|---------------|---------|
| MBPTrack | 71.95 | 20.54 | -71.4% |
| M2-Track | 67.71 | 20.26 | -70.1% |
| CXTrack | 70.18 | 14.29 | -79.6% |

所有跟踪器在GSOT3D上性能暴跌70-80%，凸显了从少类别交通场景到多类别通用场景的巨大困难。

## 亮点与洞察

1. **基准的全面性与前瞻性**：GSOT3D首次在单一基准中同时支持PC/RGB-PC/RGB-D三种3D跟踪任务，54类覆盖多样物体，9DoF标注更精确
2. **渐进式精化的简洁有效**：PROT3D的多阶段级联架构逐步编码目标信息，使搜索区域特征越来越具判别力，思路简单但效果显著
3. **暴露通用3D跟踪的差距**：所有SOTA跟踪器在GSOT3D上表现极差（mAO仅6-22%），远低于KITTI上的60-72%，凸显了适合通用3D跟踪的多样化数据的重要性
4. **数据驱动的提升证据**：在GSOT3D上重训练可显著提升性能（如P2B从2.81%→9.79%的mAO），证明多样化训练数据的关键作用

## 局限性

- 实验主要聚焦PC模态跟踪，尚未充分探索RGB-PC和RGB-D多模态跟踪
- 序列长度相对较短（平均198帧），不适合长期跟踪研究
- 数据集规模（620序列）虽然比现有3D SOT基准大，但远小于2D跟踪基准（数千至数万视频）
- PROT3D仅限于点云模态，未利用RGB和深度信息的互补性

## 相关工作

- **3D SOT基准**：KITTI（8类交通场景），NuScenes（23类交通场景），Track-it-in-3D（44类RGB-D, 300序列）
- **通用2D跟踪基准**：GOT-10K、LaSOT、TrackingNet等大规模多类别数据集
- **3D跟踪算法**：P2B、BAT、M2-Track、CXTrack、MBPTrack等基于点云的Siamese/Transformer跟踪器

## 评分

- **创新性**: ⭐⭐⭐⭐ — 基准的全面性是主要贡献，PROT3D的渐进式架构有亮点但非高度原创
- **技术质量**: ⭐⭐⭐⭐ — 数据采集和标注流程严谨，但仅评估PC模态略显不足
- **实验充分度**: ⭐⭐⭐⭐ — 8个基线 + 详细消融，属性分析和跨基准对比有说服力
- **表达清晰度**: ⭐⭐⭐⭐ — 结构清晰，图示丰富
- **综合评分**: 7.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] 3D Single-Object Tracking in Point Clouds with High Temporal Variation](../../ECCV2024/3d_vision/3d_single-object_tracking_in_point_clouds_with_high_temporal_variation.md)
- [\[ICCV 2025\] Multi-View 3D Point Tracking](multi-view_3d_point_tracking.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[ICCV 2025\] WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images](wildseg3d_segment_any_3d_objects_in_the_wild_from_2d_images.md)
- [\[CVPR 2025\] HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos](../../CVPR2025/3d_vision/hot3d_hand_and_object_tracking_in_3d_from_egocentric_multi-view_videos.md)

</div>

<!-- RELATED:END -->
