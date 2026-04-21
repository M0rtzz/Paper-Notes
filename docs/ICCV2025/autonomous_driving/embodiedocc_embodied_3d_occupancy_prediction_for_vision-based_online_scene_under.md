---
title: >-
  [论文解读] EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding
description: >-
  [ICCV 2025][自动驾驶][3D occupancy prediction] 提出EmbodiedOcc框架，利用3D语义高斯作为全局记忆，通过逐步探索和局部更新实现基于单目视觉输入的在线室内场景三维占据预测。
tags:
  - ICCV 2025
  - 自动驾驶
  - 3D occupancy prediction
  - embodied perception
  - 3D Gaussian
  - 场景理解
  - indoor scene
---

# EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding

**会议**: ICCV 2025  
**arXiv**: [2412.04380](https://arxiv.org/abs/2412.04380)  
**代码**: [https://github.com/YkiWu/EmbodiedOcc](https://github.com/YkiWu/EmbodiedOcc)  
**领域**: autonomous_driving  
**关键词**: 3D occupancy prediction, embodied perception, 3D Gaussian, online scene understanding, indoor scene

## 一句话总结

提出EmbodiedOcc框架，利用3D语义高斯作为全局记忆，通过逐步探索和局部更新实现基于单目视觉输入的在线室内场景三维占据预测。

## 研究背景与动机

三维占据预测为智能体提供对周围环境的全面理解，已成为3D感知的核心任务。现有方法主要聚焦于离线感知或少数视角的3D占据预测，无法应用于需要渐进式探索场景的具身智能体。尤其在室内场景中，通常需要多次遍历以获得房间级别的全局理解，而非局部视锥范围内的单次预测。

作者观察到：(1) 室内3D占据预测需要从流式单目RGB输入中在线更新全局场景理解；(2) 人类感知新环境的方式是通过具身探索逐步积累知识；(3) 现有户外场景方法（如TPVFormer、SurroundOcc）直接迁移到室内场景效果不佳，因为它们关注粗糙布局而非精细结构。

## 方法详解

### 整体框架

EmbodiedOcc采用两阶段训练流程：首先训练局部精炼模块用于单帧视锥内的占据预测，然后利用训练好的局部模块训练包含高斯记忆的全局在线框架。框架在世界坐标系中用均匀分布的3D语义高斯初始化场景，随后在每个时间步中根据智能体的新观测更新当前视锥内的高斯，最终通过Gaussian-to-voxel splatting获得全局3D占据。

### 关键设计

1. **局部精炼模块 (Local Refinement Module)**: 使用16200个3D语义高斯表示当前视锥。每个高斯由均值$\mathbf{m}$、尺度$\mathbf{s}$、旋转四元数$\mathbf{r}$、不透明度$\mathbf{o}$和语义logits $\mathbf{c}$组成。通过embedding层将高斯向量提升为高维特征，然后通过3D稀疏卷积进行高斯间交互，并使用可变形注意力整合图像特征进行精炼。设计动机：相比体素化表示，高斯具有更好的灵活性，适合局部-全局交互。

2. **深度感知分支 (Depth-Aware Branch)**: 利用DepthAnything-V2预测深度图$D_{metric}$，将每个高斯的均值投影到图像坐标系获取采样深度值$d$，与高斯在相机坐标系中的z分量一起输入3层MLP，生成深度感知特征$\mathbf{Q}_{depth}$并加到原始特征上。公式：$\mathbf{Q}_{depth} = \mathcal{M}_{depthaware}(D_{metric}(u,v), z)$，$\hat{\mathbf{Q}}_i = \mathbf{Q}_i + \mathbf{Q}_i^{depth}$。设计动机：深度信息不仅影响高斯均值，还能促进其他属性的更新（如语义、不透明度）。

3. **高斯记忆与置信度更新 (Gaussian Memory with Confidence Refinement)**: 在世界坐标系中维护显式全局高斯记忆。每个高斯附带标签$\gamma \in \{0,1\}$指示是否被更新过。对已更新的高斯（$\gamma=1$），设置置信度$\theta$来权衡记忆信息和当前输入：$\Delta\mathbf{G}_{online} = (1-\theta)\Delta\mathbf{G}$。在前两层精炼中$\theta=0$（冻结），最后一层$\theta=0.5$。设计动机：模拟人类重访已知场景时的微调行为，避免破坏先前良好的预测。

### 损失函数 / 训练策略

训练使用四种损失函数的加权组合：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{focal} + \mathcal{L}_{lov} + \mathcal{L}_{scal}^{geo} + \mathcal{L}_{scal}^{sem}$$

- Focal loss ($\mathcal{L}_{focal}$)：处理类别不平衡
- Lovász-softmax loss ($\mathcal{L}_{lov}$)：直接优化IoU
- Scene-class affinity loss（几何和语义两个版本）

局部模块先在Occ-ScanNet上训练10个epoch，然后EmbodiedOcc在EmbodiedOcc-ScanNet上训练5个epoch。使用AdamW优化器，学习率warm-up后最大值2e-4，余弦衰减。

## 实验关键数据

### 主实验

**局部占据预测 (Occ-ScanNet)**：

| 方法 | 输入 | IoU | mIoU |
|------|------|-----|------|
| TPVFormer | RGB | 33.39 | 24.94 |
| GaussianFormer | RGB | 40.91 | 29.93 |
| MonoScene | RGB | 41.60 | 24.62 |
| SurroundOcc | RGB | 42.52 | 30.83 |
| **EmbodiedOcc** | **RGB** | **53.55** | **45.15** |

**具身占据预测 (EmbodiedOcc-ScanNet)**：

| 方法 | IoU | mIoU |
|------|-----|------|
| TPVFormer | 35.88 | 25.70 |
| GaussianFormer | 38.02 | 27.36 |
| SplicingOcc (局部拼接) | 49.01 | 40.74 |
| **EmbodiedOcc** | **51.52** | **42.53** |

### 消融实验

**模型设计分析**：

| 方法 | Gaussian | 结构感知 | 记忆 | 局部IoU/mIoU | 具身IoU/mIoU |
|------|----------|---------|------|-------------|-------------|
| Voxel版本 | ✗ | ✓ | ✓ | 47.50/38.12 | 37.53/26.99 |
| 无记忆 | ✓ | ✓ | ✗ | 53.55/45.15 | 49.01/40.74 |
| **完整模型** | **✓** | **✓** | **✓** | **53.55/45.15** | **51.52/42.53** |

**深度分支消融**：

| 分支类型 | 局部IoU/mIoU | 具身IoU/mIoU |
|---------|-------------|-------------|
| 无深度分支 | 48.15/40.07 | 37.52/30.73 |
| Naive深度分支 | 50.32/42.73 | - |
| **深度感知分支(DAv2)** | **53.93/46.20** | **50.78/41.45** |

### 关键发现

- Look-Back评估验证了连续更新的有效性：重温已探索区域后，K=5时mIoU从40.03提升至40.98
- 体素版本在局部效果尚可但全局崩塌，验证了高斯表示在局部-全局交互中的优势
- 中等程度的置信度更新（$\theta=0.5$）效果最佳
- 运行时分析：每帧更新约114ms，其中图像骨干网络61ms、深度估计35ms

## 亮点与洞察

1. **任务定义创新**：首次形式化"具身3D占据预测"任务，填补了从离线局部预测到在线全局感知的空白
2. **高斯记忆机制**：用显式3D高斯维护全局记忆，既保持了结构灵活性又支持高效的局部-全局信息融合
3. **与人类认知的对齐**：置信度机制模拟了人类重访场景时的认知过程——对已知区域微调，对新区域重建
4. **基准数据集构建**：重组了EmbodiedOcc-ScanNet基准，包含537/137个训练/验证场景

## 局限与展望

1. 仅在室内场景验证，未扩展到更大规模的户外环境
2. 停止机制较为简单（基于已更新高斯比例的阈值），缺乏对探索策略的学习
3. 依赖预训练深度估计模型（DepthAnything-V2），其误差会传播
4. 帧级运行时约114ms，距实时应用仍有差距
5. 可考虑将显式高斯记忆与语言模型结合，支持基于语义的场景查询

## 相关工作与启发

- **GaussianFormer**: 首次将3D高斯应用于户外3D占据预测，本文将其扩展到室内具身场景
- **Online3D**: 在线3D感知的先驱，但依赖RGB-D输入且主要针对点分割和检测
- **MonoScene/ISO**: 单目3D占据预测的代表工作，本文在此基础上引入全局在线框架

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次定义具身3D占据预测任务，高斯记忆设计自然优雅
- **实验充分度**: ⭐⭐⭐⭐ 多维度消融实验全面，Look-Back评估设计巧妙
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，方法描述详实，图表精心设计
- **价值**: ⭐⭐⭐⭐ 为具身智能的场景理解提供了实用方案，开源代码

<!-- RELATED:START -->

## 相关论文

- [Semantic Causality-Aware Vision-Based 3D Occupancy Prediction](semantic_causality-aware_vision-based_3d_occupancy_prediction.md)
- [GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)
- [Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)
- [GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction](../../ECCV2024/autonomous_driving/gaussianformer_scene_as_gaussians_for_vision-based_3d_semantic_occupancy_predict.md)
- [Online Video Understanding: OVBench and VideoChat-Online](../../CVPR2025/autonomous_driving/online_video_understanding_ovbench_and_videochat-online.md)

<!-- RELATED:END -->
