---
title: >-
  [论文解读] RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion
description: >-
  [CVPR 2025][自动驾驶][雷达-相机融合] 提出基于查询(query-based)的雷达-相机融合框架 RaCFormer，通过同时从图像视角和 BEV 视角采样特征，结合圆形查询初始化、雷达感知深度预测和隐式动态捕获模块，在 nuScenes 上达到 64.9% mAP 和 70.2% NDS。
tags:
  - CVPR 2025
  - 自动驾驶
  - 雷达-相机融合
  - 3D目标检测
  - BEV感知
  - 查询机制
---

# RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion

**会议**: CVPR 2025  
**arXiv**: [2412.12725](https://arxiv.org/abs/2412.12725)  
**代码**: [GitHub](https://github.com/cxmomo/RaCFormer)  
**领域**: 自动驾驶  
**关键词**: 雷达-相机融合, 3D目标检测, BEV感知, 查询机制

## 一句话总结

提出基于查询(query-based)的雷达-相机融合框架 RaCFormer，通过同时从图像视角和 BEV 视角采样特征，结合圆形查询初始化、雷达感知深度预测和隐式动态捕获模块，在 nuScenes 上达到 64.9% mAP 和 70.2% NDS。

## 研究背景与动机

当前主流的雷达-相机融合 3D 检测方法主要采用 BEV 融合范式：将图像和雷达特征各自转换到 BEV 空间后进行融合（拼接或交叉注意力）。然而，这一范式存在两个核心瓶颈：

1. **相机 BEV 特征失真**：图像到 BEV 的转换依赖深度估计，深度不准确导致 BEV 特征中存在未对齐的视觉内容
2. **雷达 BEV 特征稀疏**：毫米波雷达空间分辨率有限，生成的 BEV 特征极为稀疏
3. **视角信息浪费**：原始透视图图像特征语义丰富且不存在失真，但 BEV 融合范式未能利用这些特征

关键观察是：query-based 方法可以将 3D 空间初始化的目标查询作为媒介，从任意投影视角（透视图和 BEV）自适应地采样特征，从而绕过特征密度差异和失真问题。这促使作者提出 RaCFormer，一个跨视角、跨模态的查询融合框架。

## 方法详解

### 整体框架

RaCFormer 包含六个核心模块：图像编码器、Pillar 编码器、雷达感知深度头、LSS 视角转换模块、隐式动态捕获器和 Transformer 解码器。图像编码器提取多帧多相机特征；Pillar 编码器处理雷达点云并展平到 BEV；深度头利用雷达增强深度估计；LSS 模块生成相机 BEV 特征；隐式动态捕获器捕获雷达 BEV 中的时序运动元素；Transformer 解码器以圆形分布初始化查询，通过 $L$ 层逐步细化查询，每层使用射线采样模块从 BEV 和图像视角提取特征。

### 关键设计一：线性递增圆形查询分布

**功能**：优化目标查询在 3D 空间的初始化分布

**核心思路**：传统径向分布（RayFormer）沿相机射线均匀放置查询，导致远距离区域查询稀疏。RaCFormer 采用同心圆分布，将查询放置在 $k$ 个同心圆上，最内圈 $n$ 个查询，每向外一圈增加 $\alpha$ 倍，第 $k$ 圈有 $\alpha^{k-1} \times n$ 个查询。总查询数为：

$$N = \frac{\alpha^k - 1}{\alpha - 1} \times n, \quad \alpha \neq 1$$

**设计动机**：远距离需要更多查询覆盖更大面积，线性递增策略使查询密度随距离合理增长，与传感器投影原理匹配。当 $\alpha = 1$ 时退化为径向分布。

### 关键设计二：雷达感知深度预测

**功能**：利用雷达深度信息增强图像到 BEV 的视角转换精度

**核心思路**：常规汽车雷达垂直角分辨率低，导致 $z$ 坐标估计误差大。RaCFormer 将所有雷达点的 $z_r$ 设为 1，投影到图像平面后将每个投影点的垂直坐标扩展到图像全高 $H$，并离散化深度值。同时嵌入 RCS 属性和像素位置信息，与下采样的图像特征 $C4$ 拼接后输入深度头。

**设计动机**：强制设置高度为常数可最大化雷达点落入图像视野的数量；结合 RCS 和位置嵌入提供更全面的雷达感知特征，从而生成更精确的深度概率分布 $D'$。

### 关键设计三：隐式动态捕获器（Implicit Dynamic Catcher）

**功能**：从多帧雷达 BEV 特征中捕获时序运动元素

**核心思路**：利用毫米波雷达的多普勒效应进行运动物体速度测量。采用 ConvGRU 跨连续帧累积隐藏状态：

$$h_t = \text{ConvGRU}(x_t, h_{t-1})$$
$$x'_t = \text{Conv2D}(h_t \oplus x_t)$$

其中 $x_t$ 为第 $t$ 帧 BEV 特征，$h_{t-1}$ 为上一帧隐藏状态。

**设计动机**：ConvGRU 擅长处理序列数据并捕获空间层级关系，通过累积多帧雷达 BEV 隐藏状态，隐式建模运动目标的时序动态，增强运动感知能力。

### 损失函数

采用标准 3D 目标检测损失，包括分类损失和回归损失（位置、尺寸、朝向、速度、属性），通过分类头和回归头解析细化后的查询。

## 实验关键数据

### 主实验：nuScenes 验证集

| 方法 | 输入 | Backbone | mAP↑ | NDS↑ | mATE↓ | mAVE↓ |
|------|------|----------|------|------|-------|-------|
| StreamPETR | C | ResNet50 | 45.0 | 55.0 | 0.613 | 0.265 |
| RCBEVDet | C+R | ResNet50 | 45.3 | 56.8 | 0.486 | 0.220 |
| HyDRa | C+R | ResNet50 | 49.4 | 58.5 | 0.463 | 0.227 |
| **RaCFormer** | **C+R** | **ResNet50** | **54.1** | **61.3** | **0.478** | **0.208** |
| HyDRa | C+R | ResNet101 | 53.6 | 61.7 | 0.416 | 0.231 |
| **RaCFormer** | **C+R** | **ResNet101** | **57.3** | **63.0** | **0.476** | **0.213** |

### nuScenes 测试集

RaCFormer 达到 **64.9% mAP** 和 **70.2% NDS**，超越所有雷达-相机融合方法。

### 消融实验

| 配置 | mAP | NDS |
|------|-----|-----|
| 基线（BEV融合） | 49.4 | 58.5 |
| + 查询融合 | 51.2 | 59.8 |
| + 圆形查询初始化 | 52.5 | 60.4 |
| + 雷达感知深度头 | 53.3 | 60.9 |
| + 隐式动态捕获器 | 54.1 | 61.3 |

### 关键发现

- 查询融合范式比 BEV 融合范式高出约 **+4.7 mAP**，验证了跨视角采样的优势
- 圆形查询初始化较径向分布带来 **+1.3 mAP** 提升，远距离检测改善明显
- 在 VoD 数据集上，RaCFormer 以 **54.4% mAP** 取得全注释区域第一名

## 亮点与洞察

1. **核心洞察**：跳出 BEV 融合的固有范式，回归查询机制从任意视角采样特征，巧妙规避特征密度不均和深度估计失真
2. **圆形查询分布**：简单但有效的几何先验，线性递增的查询密度自然适配自动驾驶场景中远近目标分布不均的特点
3. **雷达深度感知**：通过设置默认高度并扩展到全图高度的预处理策略，创造性地解决了车载雷达垂直角分辨率不足的问题

## 局限与展望

- 方法依赖目标 3D 标注的高质量跟踪数据（tracklet），对于极度遮挡场景可能受限
- 圆形查询分布的超参数 $\alpha$ 和 $k$ 需要针对不同场景调整
- 未来可探索动态调整查询分布的自适应策略或与 4D 雷达融合的扩展

## 相关工作与启发

- **RayFormer**：径向射线查询初始化的先驱，RaCFormer 在此基础上改进为距离自适应的圆形分布
- **SparseBEV**：全稀疏检测框架，RaCFormer 借鉴其尺度自适应自注意力机制
- **HyDRa**：最强基线之一，采用混合方法融合透视图和 BEV 特征，但仍受限于 BEV 融合范式

## 评分

⭐⭐⭐⭐ — 方法设计动机清晰，圆形查询分布和雷达感知深度预测都针对实际问题给出了简洁有效的解决方案。nuScenes 上 +4.7 mAP 的提升显著，VoD 第一名成绩过硬。但核心创新偏向工程组合而非根本性突破。

<!-- RELATED:START -->

## 相关论文

- [R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](../../CVPR2026/autonomous_driving/r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)
- [TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-Stage Fusion](tacodepth_towards_efficient_radar-camera_depth_estimation_with_one-stage_fusion.md)
- [CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](../../ICCV2025/autonomous_driving/cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection](v2x-r_cooperative_lidar-4d_radar_fusion_with_denoising_diffusion_for_3d_object_d.md)
- [PAP: A Prediction-as-Perception Framework for 3D Object Detection](a_prediction-as-perception_framework_for_3d_object_detection.md)

<!-- RELATED:END -->
