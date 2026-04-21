---
title: >-
  [论文解读] Segment Anything, Even Occluded
description: >-
  [CVPR 2025][自动驾驶][遮挡分割] 提出 SAMEO，将 EfficientSAM 适配为遮挡物体的 amodal 分割解码器，结合新构建的 300K 图像 Amodal-LVIS 数据集，实现了在 COCOA-cls 和 D2SA 上超越监督方法的零样本 amodal 分割性能。
tags:
  - CVPR 2025
  - 自动驾驶
  - 遮挡分割
  - SAM适配
  - 合成数据集
  - 零样本泛化
  - 基础模型
---

# Segment Anything, Even Occluded

**会议**: CVPR 2025  
**arXiv**: [2503.06261](https://arxiv.org/abs/2503.06261)  
**代码**: 无  
**领域**: autonomous_driving  
**关键词**: 遮挡分割, SAM适配, 合成数据集, 零样本泛化, 基础模型

## 一句话总结

提出 SAMEO，将 EfficientSAM 适配为遮挡物体的 amodal 分割解码器，结合新构建的 300K 图像 Amodal-LVIS 数据集，实现了在 COCOA-cls 和 D2SA 上超越监督方法的零样本 amodal 分割性能。

## 研究背景与动机

Amodal 实例分割旨在预测物体的完整形状（包括被遮挡的部分），在自动驾驶、机器人操作和场景理解中有重要应用。现有方法存在以下不足：

1. **灵活性不足**：现有方法需要联合训练前端检测器和 mask 解码器，无法利用已有的强力预训练检测器
2. **数据集规模有限**：现有 amodal 数据集图像数量少，标注质量参差不齐
3. **标注偏差**：很多数据集包含大量无意义标注（如墙壁、地板），对场景理解贡献有限
4. **合成数据问题**：自动生成的数据集存在不一致和错误的实例标注

SAM 等基础模型在 modal 分割上表现优异，但无法直接处理遮挡区域。本文的核心思路是：将 SAM 的能力扩展至 amodal 分割，同时保持其零样本泛化能力。

## 方法详解

### 整体框架

SAMEO 基于 EfficientSAM 架构，保留轻量级图像编码器 $\mathcal{E}$、prompt 编码器 $\mathcal{P}$ 和双交叉注意力 mask 解码器 $\mathcal{D}$。给定图像 $I$ 和框提示 $B$，预测 amodal mask $\hat{M}$ 和 IoU 估计 $\hat{\rho}$：

$$\hat{M}, \hat{\rho} = \mathcal{D}(\mathcal{E}(I), \mathcal{P}(B))$$

推理时可灵活接入不同前端检测器（如 AISFormer、RTMDet），检测框作为 prompt 输入 SAMEO 生成 amodal mask。

### 关键设计

**1. 仅微调 Mask 解码器的训练策略**

- **功能**：在保持图像编码器和 prompt 编码器权重不变的情况下，仅微调 mask 解码器来适配 amodal 分割
- **核心思路**：训练时随机以等概率选择 modal 或 amodal ground-truth 框作为 prompt，使模型同时学习两种提示下的 amodal 预测能力
- **设计动机**：保持编码器的预训练表征能力，避免在有限 amodal 数据上过拟合，同时通过随机 prompt 策略提高对不同前端检测器的兼容性

**2. Amodal-LVIS 大规模合成数据集**

- **功能**：提供 300K 图像的配对训练数据，每张图像包含遮挡和非遮挡版本的实例标注
- **核心思路**：从 LVIS/LVVIS 中收集完整无遮挡物体，随机配对生成合成遮挡，并采用双标注机制（同时保留遮挡和原始版本）
- **设计动机**：仅训练遮挡实例会导致模型过度预测背景为被遮挡物体（over-prediction bias），双标注机制有效防止遮挡偏差

**3. 综合数据集清洗与收集**

- **功能**：构建 1M 图像、2M 实例标注的综合训练集
- **核心思路**：对 DYCE、MP3D-amodal 过滤无意义建筑元素，对 WALT 设置遮挡阈值过滤不自然遮挡，对 COCOA 等过滤"stuff"类标注
- **设计动机**：现有数据集存在标注噪声和无关物体，系统性清洗确保训练数据质量

### 损失函数

训练损失结合 Dice loss、Focal loss 和 IoU 估计 L1 loss：

$$\mathcal{L} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{Focal}} + \lambda \mathcal{L}_{\text{IoU}}$$

其中 $\lambda = 0.05$，Focal loss 中 $\gamma = 2$。IoU 预测用于推理时精细化前端检测器的置信度：$\hat{\rho}_{\text{ref}} = \hat{\rho}_{\text{front}} \times \hat{\rho}_{\text{ours}}$。

## 实验关键数据

### 主实验：不同前端检测器下的性能对比（COCOA-cls / D2SA）

| 方法 | COCOA-cls AP | COCOA-cls AP50 | D2SA AP | D2SA AP50 |
|------|-------------|---------------|---------|----------|
| AISFormer | 40.6 | 70.5 | 66.3 | 89.9 |
| RTMDet* | 49.8 | 71.2 | 59.7 | 81.3 |
| AISFormer + SAMEO | **54.3** | **74.0** | **79.8** | **92.7** |
| RTMDet* + SAMEO | **55.3** | **75.2** | 72.7 | 85.8 |
| ConvNeXt-V2* + SAMEO | 54.1 | 73.1 | **80.8** | **94.0** |

### 零样本性能对比

| 方法 | COCOA-cls AP | D2SA AP |
|------|-------------|---------|
| AISFormer (supervised) | 40.6 | 66.3 |
| RTMDet* + SAMEO† (zero-shot) | **54.4** | 68.4 |
| CO-DETR* + SAMEO† (zero-shot) | 54.0 | **75.0** |

### 消融实验

| 消融项 | AP | AP50 | AP75 |
|--------|-----|------|------|
| 无 IoU 预测 | 52.4 | 73.2 | 57.8 |
| 有 IoU 预测 | **54.3** | **74.0** | **59.7** |
| 仅 amodal prompt | 53.0 | 72.9 | 58.0 |
| 仅 modal prompt | 53.7 | 73.3 | 59.3 |
| 随机 prompt | **54.2** | **73.5** | **59.5** |

### 关键发现

- SAMEO 零样本性能超越 AISFormer 监督方法，COCOA-cls 上 AP 提升高达 13.8 点
- 随机选择 modal/amodal prompt 训练效果最好，泛化性最强
- 仅训练遮挡数据会产生 over-prediction，双标注机制有效缓解

## 亮点与洞察

1. **解耦设计思路优秀**：将 amodal 分割解耦为「前端检测 + SAMEO 解码」，实现即插即用，任意检测器均可升级为 amodal 分割
2. **数据工程驱动的零样本能力**：通过大规模数据收集和清洗实现零样本性能超越监督方法，说明数据质量和规模的重要性
3. **双标注机制**：发现仅用遮挡数据训练的 over-prediction 问题并提出简洁有效的解决方案

## 局限与展望

- 模型本身不进行检测，仍依赖前端检测器的质量
- Amodal-LVIS 的合成遮挡可能不完全反映真实世界的复杂遮挡模式
- 未来可探索端到端的 amodal 分割方案，或将方法推广到视频 amodal 分割

## 相关工作与启发

- **SAM/EfficientSAM**：验证了基础分割模型可以通过仅微调解码器适配到新任务
- **AISFormer**：当前 SOTA amodal 方法，但灵活性不足
- **pix2gestalt**：提供大规模合成 amodal 数据，但存在标注不完整问题

## 评分

⭐⭐⭐⭐ — 方法简洁优雅，数据工程扎实。将基础模型成功适配到 amodal 分割且实现零样本超越监督方法是核心贡献。框架的即插即用特性实用价值高。

<!-- RELATED:START -->

## 相关论文

- [SAM4D: Segment Anything in Camera and LiDAR Streams](../../ICCV2025/autonomous_driving/sam4d_segment_anything_in_camera_and_lidar_streams.md)
- [Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation](prompting_depth_anything_for_4k_resolution_accurate_metric_depth_estimation.md)
- [Detect Anything 3D in the Wild](../../ICCV2025/autonomous_driving/detect_anything_3d_in_the_wild.md)
- [Cubify Anything: Scaling Indoor 3D Object Detection](cubify_anything_scaling_indoor_3d_object_detection.md)
- [Towards In-the-Wild 3D Plane Reconstruction from a Single Image](towards_in-the-wild_3d_plane_reconstruction_from_a_single_image.md)

<!-- RELATED:END -->
