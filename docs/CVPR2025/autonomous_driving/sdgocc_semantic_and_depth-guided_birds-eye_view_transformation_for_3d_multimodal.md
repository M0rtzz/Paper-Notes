---
title: >-
  [论文解读] SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction
description: >-
  [CVPR 2025][自动驾驶][3D占用预测] 本文提出 SDG-OCC，一个多模态 3D 语义占用预测框架，通过语义和深度引导的视图变换（利用 LiDAR 深度和图像语义分割掩码构建虚拟点）替代传统 LSS 管线，结合融合到占用驱动的主动蒸馏模块，在 Occ3D-nuScenes 上取得 SOTA 并保持实时推理速度。
tags:
  - CVPR 2025
  - 自动驾驶
  - 3D占用预测
  - 多模态融合
  - BEV变换
  - 知识蒸馏
  - LiDAR-Camera融合
---

# SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction

**会议**: CVPR 2025  
**arXiv**: [2507.17083](https://arxiv.org/abs/2507.17083)  
**代码**: https://github.com/DzpLab/SDGOCC  
**领域**: 自动驾驶 / 3D占用预测  
**关键词**: 3D占用预测, 多模态融合, BEV变换, 知识蒸馏, LiDAR-Camera融合

## 一句话总结

本文提出 SDG-OCC，一个多模态 3D 语义占用预测框架，通过语义和深度引导的视图变换（利用 LiDAR 深度和图像语义分割掩码构建虚拟点）替代传统 LSS 管线，结合融合到占用驱动的主动蒸馏模块，在 Occ3D-nuScenes 上取得 SOTA 并保持实时推理速度。

## 研究背景与动机

**领域现状**：3D 语义占用预测同时估计场景体素的几何结构和语义类别，相比 3D 目标检测和语义分割，能更全面地建模环境。多模态方法利用 LiDAR 和相机数据的互补性：相机提供丰富的语义信息但缺乏精确深度，LiDAR 提供精确深度但数据稀疏。

**现有痛点**：（1）当前轻量级方法主要依赖 LSS（Lift-Splat-Shoot）管线进行 2D-3D 视图变换，但 LSS 的深度估计不准确，BEV 表征极度稀疏——只有不到 50% 的 BEV 网格能接收到有效图像特征；（2）减小深度间隔虽可提高准确率但大幅增加计算开销；（3）现有融合方法直接拼接 LiDAR 和图像 BEV 特征，但由于外参偏差导致的特征未对齐问题，融合效果不佳；（4）多数多模态方法计算量巨大，无法实时运行。

**核心矛盾**：LSS 管线为每个像素在预定义深度范围内构建大量虚拟点，但大部分虚拟点落入空白 BEV 网格，产生冗余计算且 BEV 利用率低；而 LiDAR 可以提供精确的深度先验却未被充分利用。

**本文目标**：设计一个多模态占用预测框架，在提升准确率的同时保持实时推理速度，关键是改进 2D-3D 视图变换和多模态融合方式。

**切入角度**：利用 LiDAR 点云的稀疏深度信息作为先验，在同一语义类别内扩散深度值，然后通过双线性离散化生成高精度虚拟点种子，大幅减少冗余虚拟点数量的同时提升 BEV 特征质量。

**核心 idea**：用语义分割掩码引导 LiDAR 深度扩散生成半密集深度图，结合双向线性增减离散化构建精确虚拟点，替代 LSS 的盲目深度采样，同时通过融合-占用驱动的主动蒸馏将多模态知识单向蒸馏到图像特征中。

## 方法详解

### 整体框架

输入多帧多视角图像和对应点云，分别通过图像 backbone 和 LiDAR backbone 提取特征。图像特征经过多任务头生成语义掩码和深度分布，结合 LiDAR 深度图通过 SDG 视图变换生成图像 BEV 特征 $F_{bev}^C$。点云特征通过体素化和 SPVCNN 编码后压缩为 LiDAR BEV 特征 $F_{bev}^L$。两种 BEV 特征通过动态邻域特征融合和占用驱动的主动蒸馏整合后，送入占用预测头生成最终输出。

### 关键设计

1. **SDG 视图变换 (Semantic and Depth-Guided View Transformation)**:

    - 功能：替代传统 LSS 管线，利用 LiDAR 深度和语义信息构建更精确、更高效的 BEV 特征
    - 核心思路：首先通过多任务头生成语义分割掩码和深度分布。然后将 LiDAR 点云投影到图像上获取稀疏深度图，在同一语义类别的区域内进行局部深度扩散（半径为 $r$ 的圆内，按 $D_{\text{temp}}(i,j) = \frac{\sum_{(p,q)\in N(i,j)} D(p,q) \cdot \mathbb{I}[M(p,q)=M(i,j)]}{\sum_{(p,q)\in N(i,j)} \mathbb{I}[M(p,q)=M(i,j)]}$ 计算），生成半密集深度图。接着通过双向线性增减离散化生成精确虚拟点种子。最后将图像纹理特征 $F_t$ 和深度分布权重 $D_w$ 通过外积 $F_t \otimes D_w$ 得到每个虚拟点的特征，经 BEV pooling 生成图像 BEV 特征
    - 设计动机：LSS 为每个像素产生大量深度假设点，但大部分无效且 BEV 利用率低（<50%）；利用 LiDAR 深度先验和语义掩码引导可以大幅减少冗余虚拟点，同时提高深度估计精度

2. **动态邻域特征融合 (Dynamic Neighborhood Feature Fusion)**:

    - 功能：解决 LiDAR 和图像 BEV 特征因外参偏差导致的空间未对齐问题
    - 核心思路：图像特征作为 source（query），LiDAR 特征作为 cross（key/value），使用邻域注意力在对应像素的局部 patch 内提取特征：$F_{\text{neighbor}} = \sigma(\frac{Q_s^i \cdot (K_c^{n(i)})^T + B(i, n(i))}{\sqrt{v}}) \cdot V_c^i$，其中 $n(i)$ 是大小为 $k$ 的邻域。然后通过门控注意力动态调整融合权重：$F_{bev}^{fuse} = \sigma(\text{Conv}(f_{\text{Avg}}(F_{\text{neighbor}}))) \cdot F_{\text{neighbor}}$
    - 设计动机：简单的通道拼接无法处理 LiDAR 和图像 BEV 特征的空间未对齐问题；邻域注意力+门控机制可以隐式处理投影偏差并动态调整融合权重

3. **占用驱动的主动蒸馏 (Occupancy-Driven Active Distillation)**:

    - 功能：将多模态融合特征的知识单向蒸馏到纯图像特征中，实现更快推理
    - 核心思路：将空间分为 Active Region（AR，LiDAR 和图像特征均有占用）和 Inactive Region（IR，仅 LiDAR 特征有占用）。由于 AR 通常远大于 IR，使用自适应缩放 $W_{I,i,j} = \alpha$（AR 区域）或 $\rho \times \beta$（IR 区域），其中 $\rho = N_{AR}/N_{IR}$ 防止蒸馏过度偏向 AR。蒸馏损失为 $L_{\text{distill}} = \sum W_{i,j}(F_{bev}^{fuse} - F_{bev}^C)^2$。训练时带融合和蒸馏（SDG-KL），推理时只用图像分支
    - 设计动机：融合模型（SDG-Fusion）精度高但推理需要同时处理 LiDAR 和图像数据；通过主动蒸馏，SDG-KL 在推理时只需图像输入就能获得接近融合的性能，实现实时推理

### 损失函数 / 训练策略

- SDG-Fusion：分类损失（占用预测头输出的交叉熵损失）
- SDG-KL：分类损失 + 蒸馏损失（区域加权 MSE）
- 语义分割辅助任务为多任务头提供监督信号
- 深度头和语义头通过门控注意力互补跨任务信息

## 实验关键数据

### 主实验

Occ3D-nuScenes 验证集：

| 方法 | 输入 | Backbone | mIoU | 推理时间(ms) |
|------|------|----------|------|-------------|
| FlashOcc | C | Swin-B | 43.52 | 909 |
| COTR | C | Swin-B | 46.2 | 840 |
| OCCFusion | C+L | R-101 | 46.79 | - |
| RadOcc-LC | C+L | Swin-B | 49.38 | 3333 |
| **SDG-KL** | C+L | R-50 | **50.16** | **83** |
| **SDG-Fusion** | C+L | R-50 | **51.66** | 133 |

SDG-Fusion 以 R-50 backbone 在 133ms 推理时间下达到 51.66 mIoU，SDG-KL 在 83ms 下达到 50.16 mIoU，均优于使用更大 backbone 的方法。

### 消融实验

视图变换对比（从论文 Fig.2 可见）：
- LSS 的 BEV 特征图中有效像素<50%，高度稀疏
- SDG 视图变换生成的 BEV 特征密度和分布明显接近 Ground Truth

融合方式消融验证了邻域注意力+门控机制优于简单拼接，主动蒸馏的区域加权策略有效平衡了 AR 和 IR 区域的知识迁移。

### 关键发现

- 使用 4x 下采样（而非更高倍率）的特征进行视图变换效果最佳，因为更高下采样会增加像素的语义和深度歧义
- SDG 视图变换不仅提升了 BEV 特征质量，还减少了虚拟点数量从而加速推理
- SDG-KL 通过蒸馏在仅损失约 1.5 mIoU 的代价下将推理速度提升 37%（133ms → 83ms）
- 在更具挑战性的 SurroundOcc-nuScenes 数据集上也展现了可比性能

## 亮点与洞察

- 精准定位了 LSS 管线的核心瓶颈——虚拟点冗余和 BEV 稀疏，并给出了优雅的解决方案
- 利用语义分割掩码引导深度扩散的思路很巧妙——同一语义类别通常具有连续的深度值
- SDG-Fusion / SDG-KL 的双模式设计非常实用——高精度场景用融合模式，实时场景用蒸馏模式
- 在 R-50 backbone 下超越了使用 Swin-B 的方法，说明框架设计的有效性
- 主动蒸馏的 AR/IR 区域自适应加权是一个值得借鉴的细节设计

## 局限与展望

- 语义分割辅助任务为视图变换引入了额外的标注需求（虽然 LiDAR 已有语义标签）
- 深度扩散的扩散半径 $r$ 需要手动设定，不同场景可能需要调整
- 未探索时序信息的利用，多帧融合可能进一步提升性能
- SDG-KL 的推理只使用图像分支，但训练仍需要 LiDAR 数据
- 在极端遮挡或恶劣天气条件下的鲁棒性未做充分验证

## 相关工作与启发

- 与 BEVFusion 系列方法相比，SDG-OCC 通过改进视图变换从源头提升 BEV 质量，而非仅在 BEV 空间做融合
- FlashOcc 和 FastOcc 聚焦纯视觉的轻量化，SDG-KL 通过蒸馏达到类似速度但精度更高
- CO-Occ 使用 KNN 搜索识别共现体素，SDG-OCC 的邻域注意力融合更加灵活
- 对 BEV 感知方向的视图变换改进和多模态融合策略有参考价值

## 评分

- **新颖性**: 7/10 — 视图变换改进和蒸馏策略在各自领域有创新，但整体属于增量改进
- **实验充分度**: 8/10 — 多数据集验证，对比全面，可视化直观
- **写作质量**: 7/10 — 方法描述清晰但符号较多，部分公式排版不够清晰
- **价值**: 8/10 — 实时性能和精度的平衡对自动驾驶部署有实际意义

<!-- RELATED:START -->

## 相关论文

- [Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)
- [ProtoOcc: 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation](3d_occupancy_prediction_with_low-resolution_queries_via_prototype-aware_view_tra.md)
- [GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction](gaussianformer-2_probabilistic_gaussian_superposition_for_efficient_3d_occupancy.md)
- [GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction](gaussianworld_gaussian_world_model_for_streaming_3d_occupancy_prediction.md)
- [OccMamba: Semantic Occupancy Prediction with State Space Models](occmamba_semantic_occupancy_prediction_with_state_space_models.md)

<!-- RELATED:END -->
