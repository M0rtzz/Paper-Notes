---
title: >-
  [论文解读] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation
description: >-
  [ICCV 2025][视频理解][motion transfer] 提出 MotionShot，一个无需训练的运动迁移框架，通过高层语义对齐和低层形态对齐的两级运动对齐策略，实现在外观和结构差异显著的任意参考-目标物体对之间的高保真运动迁移。
tags:
  - ICCV 2025
  - 视频理解
  - motion transfer
  - text-to-video
  - training-free
  - TPS warping
  - 注意力机制
---

# MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation

**会议**: ICCV 2025  
**arXiv**: [2507.16310](https://arxiv.org/abs/2507.16310)  
**代码**: [项目页面](https://motionshot.github.io/)  
**领域**: video_understanding  
**关键词**: motion transfer, text-to-video, training-free, TPS warping, temporal attention guidance

## 一句话总结

提出 MotionShot，一个无需训练的运动迁移框架，通过高层语义对齐和低层形态对齐的两级运动对齐策略，实现在外观和结构差异显著的任意参考-目标物体对之间的高保真运动迁移。

## 研究背景与动机

**运动迁移的挑战**：
- 现有方法大多只能处理外观相似的参考-目标对（如人→人、动物→同类动物）
- 当参考物体和目标物体存在显著外观/结构差异时（如动漫人物→维尼熊），运动迁移效果急剧下降

**现有方法的局限**：

**关键点序列方法**：需要为每类物体预定义关键点，无法泛化到任意物体

**时空特征方法**：运动和外观在潜在表示中纠缠，导致参考外观泄漏

**深度/边缘/光流条件**：不考虑区域级语义对应和像素级结构对应，面对大差异物体对失败

**基于注意力的方法**：运动与结构高度耦合，目标与参考差异大时运动不兼容

**核心问题**：如何在保持目标物体外观的同时，准确迁移参考物体的运动模式？

## 方法详解

### 整体框架

MotionShot 基于 AnimateDiff 视频生成框架，包含三个主要阶段：
1. **语义运动对齐**：建立参考-目标物体间的高层语义对应
2. **形态运动对齐**：通过 TPS 变换实现低层结构映射
3. **注意力引导生成**：用变形后的参考帧引导视频生成

### 关键设计

1. **语义运动对齐（Semantic Motion Alignment）**：

    - **假目标生成**：使用 ControlNet-segmentation 模型，输入退化的参考物体分割图（粗略初始姿态提示）+ 文本提示，生成与参考物体初始姿态相近的假目标物体。ControlNet 条件权重设为 0.6，确保文本提示主导
    - **结构感知关键点采样**：在参考物体上采样 $m=30$ 个关键点，包括均匀轮廓采样（间隔 $d=200$）和泊松盘内部采样，确保关键点分散分布在物体各区域
    - **语义特征匹配**：融合 Stable Diffusion 特征（低层空间信息）和 DINOv2 特征（高层语义信息），通过 $L_2$ 距离计算相似度：$\text{Sim}(i,j) = -\|f_\text{tar}^s(i) - f_\text{ref}^s(j)\|_2$
    - 设计动机：SD 特征提供精细空间细节但在歧义区域易出错，DINO 捕获高层语义但可能遗漏细节，融合互补

2. **形态运动对齐（Morphological Motion Alignment）**：

    - **目标关键点序列构建**：使用 CoTracker3 跟踪参考关键点，通过全局运动（椭圆旋转和平移）和局部运动（极坐标相对偏移）迁移到目标空间
    - 全局运动：$K_\text{tar}^t = \mathcal{S}(\mathcal{R}(K_\text{tar}^0, \Delta\Theta^t), \Delta O^t)$
    - 局部运动：将关键点偏移分解为径向缩放和极角偏移
    - **TPS 形状变形**：利用 TPS（Thin Plate Spline）变换将参考帧变形为目标形状
    $\mathcal{T}^t(p) = A^t\begin{bmatrix}p\\1\end{bmatrix} + \sum_{i=1}^m w^{t,i}\mathcal{U}(\|\mathbf{K}_\text{tar}^{t,i}-p\|^2)$
    - 设计动机：点级引导缺乏连续性，破坏时间注意力；TPS 变形提供连续的形状映射

3. **注意力引导视频生成（Attention-guided Generation）**：

    - 对变形后的参考帧进行单步加噪-去噪，提取时间注意力图 $A_\text{ref}^\tau$
    - 选取 top-k（$k=1$）稀疏控制掩码，减少噪声
    - 定义能量函数：$g = \|M^\tau \cdot (A_\text{ref}^\tau - A_\text{gen}^t)\|_2^2$
    - 通过 score-based guidance 引导扩散采样：$\hat{\epsilon}_\theta = \epsilon_\theta(z_t, \text{text}, t) - \lambda\nabla_{z_t}g$
    - 使用 DDIM 采样器，300 步中前 180 步应用引导
    - 设计动机：由于参考帧已被变形为目标形状，时间注意力中的运动信息自然与目标结构对齐

### 损失函数 / 训练策略

MotionShot 是**完全无训练**的框架：
- 不需要额外的训练数据或微调
- 基于预训练的 AnimateDiff、ControlNet、Stable Diffusion、DINOv2、CoTracker3
- 所有对齐通过特征匹配和几何变换实现

## 实验关键数据

### 主实验 (表格)

**定量比较（CLIP Scores + 用户研究）**：

| 方法 | Text Align↑ | Temporal Consist↑ | Motion Preserv↑ | Appear Diversity↑ | User-Text↑ | User-Temporal↑ |
|------|------------|-------------------|-----------------|-------------------|-----------|---------------|
| VideoComposer | 26.54 | 95.95 | 3.00 | 2.72 | 2.79 | 2.82 |
| Gen-1 | 22.79 | 97.67 | 2.87 | 2.71 | 2.75 | 2.87 |
| VMC | 26.77 | 97.72 | 2.80 | 2.78 | 2.78 | 2.87 |
| Tune-A-Video | 26.60 | 95.99 | 2.86 | 2.78 | 2.88 | 2.86 |
| Control-A-Video | 24.87 | 95.54 | 2.94 | 2.66 | 2.40 | 2.92 |
| MotionClone | 26.41 | 97.48 | 2.90 | 2.50 | 2.80 | 2.82 |
| **MotionShot** | **26.95** | **97.81** | **4.95** | **4.95** | **4.94** | **4.90** |

- 用户研究中 MotionShot 在所有四个维度上均接近满分（5分制），远超其他方法

### 消融实验 (表格)

**关键点数量消融**：

| 关键点数 $m$ | 轮廓点 | 内部点 | 效果 |
|-------------|--------|--------|------|
| 10 | 8 | 2 | TPS 变形失败，无法匹配目标形状 |
| **30** | **24** | **6** | **合理变形，最佳效果** |
| 60 | 48 | 12 | 过拟合，变形结果不自然 |

**语义特征匹配方法比较**：

| 方法 | 效果描述 |
|------|---------|
| X-Pose 关键点检测器 | 仅预测 17 点，分布不均，外观不匹配 |
| 仅 SD 特征 | 精细空间细节，但歧义区域易错（如尾巴） |
| 仅 DINO 特征 | 高层语义好，但遗漏细节（如腿部） |
| **SD + DINO 融合** | **平衡精细与高层精度，效果最优** |

**形状重定向方法比较**：

| 方法 | 问题 |
|------|------|
| 原始序列不变形 | 运动-形状不匹配，生成物体变形 |
| 简单缩放 | 尺寸一致但拓扑扭曲（如腿部错位） |
| **关键点 TPS 变形** | **运动精度和结构一致性俱佳** |

### 关键发现

- 用户研究中运动保留和外观多样性得分接近满分（4.95/5），远超第二名（~3.0/5）
- 纯无训练方法在运动迁移质量上大幅超越需要训练的方法
- SD+DINO 特征融合对语义匹配至关重要，单独使用任一特征均不够
- TPS 变形是关键步骤，点级引导或简单缩放都会导致形状问题
- 30 个关键点是最佳数量，太少变形不充分，太多过拟合

## 亮点与洞察

1. **两级对齐框架**：首次显式建模高层语义对齐和低层形态对齐，解决了任意物体对运动迁移的核心难题
2. **完全无训练**：不需要任何额外训练或微调，组合现有预训练模型实现强大功能
3. **假目标生成策略**：利用退化分割图引导生成初始姿态一致的假目标，巧妙解决初始姿态不匹配问题
4. **结构感知关键点采样**：均匀轮廓+泊松盘内部的采样策略确保关键点在物体各区域的均匀覆盖
5. **运动分解**：全局旋转+平移 + 局部极坐标偏移的运动分解方式，细粒度地迁移复杂运动

## 局限与展望

- 当参考物体和目标物体完全没有语义相似性时（如飞机→花朵），方法失效
- 语义对应依赖 SD 和 DINO 的特征质量，对于远离预训练分布的物体可能不可靠
- 基于 AnimateDiff 框架，视频质量和长度受底层模型限制
- 300 步采样 + 前 180 步引导，推理效率有待提升
- 仅支持单物体运动迁移，多物体交互的运动迁移未探索

## 相关工作与启发

- 两级对齐（语义+形态）的思路可推广到其他需要跨域对应的任务（如风格迁移、动画）
- TPS 变形作为连续的空间映射工具，比离散点级引导更适合注意力引导方案
- 无训练方法通过巧妙组合预训练模型，可以解决复杂的视觉生成问题

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 两级运动对齐框架是首创，解决了运动迁移中长期存在的任意物体对难题
- **实验充分度**: ⭐⭐⭐⭐ 定量+用户研究+详细消融，但缺少更多定量指标（自动化运动保真度评估）
- **写作质量**: ⭐⭐⭐⭐ 方法描述清晰，图示丰富，动机论述到位
- **价值**: ⭐⭐⭐⭐⭐ 用户研究中几乎满分的表现证明了方法的实际价值，无训练设计降低使用门槛

<!-- RELATED:START -->

## 相关论文

- [DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization](dualreal_adaptive_joint_training_for_lossless_identity-motion_fusion_in_video_cu.md)
- [EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation](free-form_motion_control_controlling_the_6d_poses_of_camera_and_objects_in_video.md)
- [VMBench: A Benchmark for Perception-Aligned Video Motion Generation](vmbench_a_benchmark_for_perception-aligned_video_motion_generation.md)
- [Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)

<!-- RELATED:END -->
