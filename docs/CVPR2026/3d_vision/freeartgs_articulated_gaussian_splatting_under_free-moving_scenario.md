---
title: >-
  [论文解读] FreeArtGS: Articulated Gaussian Splatting Under Free-Moving Scenario
description: >-
  [CVPR 2026][3D视觉][铰接物体重建] FreeArtGS 提出在"自由移动场景"(物体姿态和关节状态同时任意变化)下从单目RGB-D视频重建铰接物体的方法，通过运动驱动的部件分割、鲁棒关节估计和端到端3DGS优化的三阶段流程，在自建FreeArt-21基准和现有数据集上远超所有基线。
tags:
  - CVPR 2026
  - 3D视觉
  - 铰接物体重建
  - 高斯溅射
  - 自由移动
  - 关节估计
  - 运动分割
---

# FreeArtGS: Articulated Gaussian Splatting Under Free-Moving Scenario

**会议**: CVPR 2026  
**arXiv**: [2603.22102](https://arxiv.org/abs/2603.22102)  
**代码**: https://freeartgs.github.io/  
**领域**: 3D视觉  
**关键词**: 铰接物体重建, 高斯溅射, 自由移动, 关节估计, 运动分割

## 一句话总结
FreeArtGS 提出在"自由移动场景"(物体姿态和关节状态同时任意变化)下从单目RGB-D视频重建铰接物体的方法，通过运动驱动的部件分割、鲁棒关节估计和端到端3DGS优化的三阶段流程，在自建FreeArt-21基准和现有数据集上远超所有基线。

## 研究背景与动机

1. **领域现状**：铰接物体重建是3D视觉的重要问题，对增强现实和机器人仿真有重要价值。现有方法主要分三个方向：(a) 基于基础模型的单图生成，泛化性差；(b) 从两个铰接状态的固定多视角相机重建，需要对齐两状态的轴；(c) 从单目视频重建，假设有固定不动的基底部件。
2. **现有痛点**：单图生成缺乏后优化难以泛化；多视角双状态方法轴对齐困难限制实用性；单目视频方法的"静态基底"假设在实际操作中常被违反(如使用剪刀、钳子时两个部件都在动)，且覆盖不完整。
3. **核心矛盾**：真实场景中铰接物体常被自由操纵——物体姿态和关节状态同时变化，没有固定基底参考。现有方法无法处理这种最自然的使用场景。
4. **本文目标** 在自由移动场景下，仅从单目RGB-D视频重建铰接物体的完整外观、几何和关节参数。
5. **切入角度**：将密集2D点跟踪先验与3DGS优化结合——用点跟踪提供运动线索驱动部件分割，用优化提供最终的高精度重建。
6. **核心 idea**：用点跟踪+特征先验做自由运动部件分割，用相对变换估计关节类型和轴向，再端到端优化3DGS同时精修外观、几何和关节。

## 方法详解

### 整体框架
输入：单目RGB-D视频 + 前景掩码(SAM生成)。输出：两个部件的规范高斯 $\mathcal{G}_c^0, \mathcal{G}_c^1$ 和关节参数 $\mathcal{J}$。流程分三个模块：(1) 自由移动部件分割——从运动中分割两个刚性部件；(2) 关节估计——从部件到相机的变换推断关节类型和轴；(3) 端到端优化——联合优化外观、几何、相机和铰接参数。

### 关键设计

1. **Free-moving Part Segmentation (自由移动部件分割)**:

    - 功能：从自由移动视频中将铰接物体分割为两个刚性部件
    - 核心思路：核心假设是短时间窗口内每个部件的运动可近似为独立的刚体变换。使用AllTracker提供像素级2D轨迹，结合深度提升为3D轨迹。用DINOv3特征初始化每点的部件权重 $w_{t,p} \in [0,1]$。在滑动窗口(8帧)内优化两个刚体变换 $T^0, T^1$ 和软部件权重。主损失函数为Huber损失，以相对运动误差度量每点属于哪个部件。关键正则化包括：熵损失鼓励接近二值的分配、基于特征空间邻居图的平滑损失保证空间一致性、与初始化权重的BCE损失防止偏离语义先验。
    - 设计动机：不假设任何部件静止，纯粹从相对运动差异分割部件。特征空间正则化防止不稳定的点跟踪结果导致次优解。

2. **Joint Estimation (关节估计)**:

    - 功能：从部件变换序列中推断关节类型(旋转/平移)和轴参数
    - 核心思路：首先用现成位姿估计器为每帧标定每个部件到相机的变换 $E_i^k \in SE(3)$，分别重建两个部件的3DGS并优化位姿。将两个部件校准到统一坐标系(以运动量最小的部件为参考)。从相对变换序列 $\{T_i\}$ 判断关节类型——小旋转跨度+强线性=平移关节，否则为旋转关节。旋转关节从成对相对旋转求闭合解旋转轴，平移关节用PCA求平移轴。两项鲁棒性设计：(a) 使用相邻帧间成对相对变换 $T_{i \to (i+1)}$ 而非绝对变换；(b) 以2σ阈值滤除异常值变换。
    - 设计动机：直接估计绝对变换 $T_i$ 受点跟踪噪声影响大，成对相对变换对噪声更鲁棒。异常值过滤进一步提升稳定性。

3. **End-to-end Optimization (端到端优化)**:

    - 功能：联合精修外观、几何、相机位姿和铰接参数
    - 核心思路：将关节参数化为旋转关节($u, o, \theta_i$)或平移关节($u, d_i$)。引入Blended Rendering——对规范高斯施加刚体变换后，按部件权重 $w \in [0,1]$ 进行alpha混合渲染：$\mathcal{G}_i = w(\mathcal{G}_c \circ I) \cup (1-w)(\mathcal{G}_c \circ \mathcal{J}_i)$。监督信号包括RGB(L1+SSIM)、深度(L1)和前景掩码(L1)。总损失为 $\mathcal{L}_{E2E} = \sum_i (\mathcal{L}_{rgb}^i + \lambda_{depth}\mathcal{L}_{depth}^i + \lambda_{mask}\mathcal{L}_{mask}^i)$。
    - 设计动机：前两个模块提供粗略但合理的初始化，端到端优化利用可微渲染约束紧密耦合外观和运动学，修正粗关节中的小偏差。Blended Rendering允许部件归属在优化中被细粒度调整。

### 损失函数 / 训练策略
部件分割阶段：$\mathcal{L} = 200\mathcal{L}_{main} + 10\mathcal{L}_{smooth} + 0.01\mathcal{L}_{ent} + 5\mathcal{L}_{init}$，每个帧对100次迭代。部件重建和端到端优化各30000次迭代，基于NeRFStudio实现。全流程约25分钟(100帧640×360视频，RTX 4090)。

## 实验关键数据

### 主实验 (FreeArt-21, 旋转关节)

| 方法 | Axis↓ (deg) | Position↓ (cm) | State↓ (deg) | CD-w↓ (cm) | CD-m↓ (cm) | PSNR↑ (dB) |
|------|-------------|----------------|--------------|------------|------------|------------|
| ArticulateAnything | 42.00 | 59.38 | - | - | - | - |
| Video2Articulation | 20.00 | 16.31 | 27.37 | 2.29 | 10.74 | - |
| **FreeArtGS** | **1.04** | **0.29** | **1.43** | **0.14** | **0.28** | **24.02** |

### 消融实验 (FreeArt-21, 旋转关节)

| 配置 | Axis↓ | Position↓ | State↓ | CD-w↓ | PSNR↑ |
|------|-------|-----------|--------|-------|-------|
| Full model | 1.04 | 0.29 | 1.43 | 0.14 | 24.02 |
| w/o Smooth Loss | 28.01 | 17.73 | 18.74 | 5.72 | 10.60 |
| w/o Init Loss | 9.35 | 19.58 | 14.64 | 0.75 | 13.07 |
| w/o Noise Resistance | 4.75 | 2.22 | 1.30 | 0.17 | 22.65 |
| w/o Blended Rendering | 1.72 | 1.88 | 1.88 | 0.12 | 22.23 |

### 关键发现
- FreeArtGS在关节轴精度上相比Video2Articulation提升约20倍(1.04° vs 20.00°)，位置精度提升56倍
- **Smooth Loss贡献最大**：移除后轴误差从1.04°暴增至28.01°，证明点跟踪不稳定性必须通过特征空间正则化缓解
- **Init Loss也很关键**：移除后位置误差从0.29cm增至19.58cm，DINOv3特征提供的语义先验对正确分区至关重要
- Noise Resistance(异常值过滤)对关节估计的鲁棒性有明显帮助
- Blended Rendering提升约2dB PSNR，同时保持关节精度
- 在Video2Articulation-S数据集(静态基底设定)上也超越所有方法，证明通用性
- 真实世界6个物体平均轴误差2.73°，几何CD 2.48cm

## 亮点与洞察
- **问题定义的价值**：首次提出并定义"自由移动场景"的铰接物体重建问题，这是最自然的使用场景，比现有假设(静态基底、多视角双状态)更实用
- **先验+优化的组合策略**：用现成模型(AllTracker, DINOv3, SAM)提供初始化先验，用优化提供最终精度。单独使用任一都不够——先验有噪声，纯优化难初始化
- **FreeArt-21基准的构建**：用VR系统遥操作PartNet-Mobility物体在Sapien中生成自由移动数据，覆盖7类21个物体(5旋转+2平移关节)，填补了领域空白
- **全流程25分钟**：100帧视频处理仅需25分钟(分割6min+关节估计1min+端到端优化18min)，实用性强

## 局限与展望
- 当前假设物体仅有两个刚性部件，无法处理多部件铰接结构(如机械臂)；可通过顺序捕获每个运动部件扩展
- 依赖多个现成模型(AllTracker、DINOv3、SAM、位姿估计器)，级联误差可能在复杂场景中放大；理想方案是开发统一前馈模型
- 需要RGB-D输入(深度信息)，纯RGB视频目前因连续视频深度预测精度不足无法支持
- 手持操作时手部遮挡虽有一定鲁棒性，但严重遮挡仍可能导致失败

## 相关工作与启发
- **vs Video2Articulation**: V2A依赖预训练前馈重建模型(Monst3R)预测动态，在自由移动场景下失败；FreeArtGS用优化方法从运动差异分割部件
- **vs ArticulateAnything**: AA使用VLM推理生成URDF，受幻觉影响在大多情况下预测错误的轴；FreeArtGS通过几何优化获得精确关节
- **vs RSRD**: RSRD假设每个部件运动模式独特，不适用于铰接物体(部件运动由关节约束关联)；在V2A-S上所有指标最差
- **vs 动态重建方法**: 前馈动态重建(如Monst3R)无法恢复自由移动场景的精确运动；FreeArtGS将前馈先验与优化结合

## 评分
- 新颖性: ⭐⭐⭐⭐ 自由移动设定是全新的问题定义，方法虽然是多个已有技术(3DGS、点跟踪、刚体拟合)的组合但组合方式非平凡
- 实验充分度: ⭐⭐⭐⭐⭐ 自建基准+现有数据集+真实物体三重验证，详尽消融，指标全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法条理分明，各模块设计动机交代充分
- 价值: ⭐⭐⭐⭐⭐ 新问题+新基准+强结果，对数字孪生和机器人学习有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Sky2Ground: A Benchmark for Site Modeling under Varying Altitude](sky2ground_a_benchmark_for_site_modeling_under_varying_altitude.md)
- [\[CVPR 2026\] E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)
- [\[NeurIPS 2025\] OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects](../../NeurIPS2025/3d_vision/onlinesplatter_pose-free_online_3d_reconstruction_for_free-moving_objects.md)
- [\[CVPR 2026\] Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty](rethinking_pose_refinement_in_3d_gaussian_splatting_under_pose_prior_and_geometr.md)
- [\[CVPR 2026\] Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction](neural_gabor_splatting.md)

<!-- RELATED:END -->
