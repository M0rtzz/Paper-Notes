---
title: "Cubify Anything: Scaling Indoor 3D Object Detection"
description: "提出Cubify Anything，使用长方体表示扩展室内3D物体检测的规模和泛化能力"
tags:
  - CVPR2025
  - 自动驾驶
  - 3D 目标检测
  - Indoor Scene Understanding
  - Cuboid Representation
---

# Cubify Anything: Scaling Indoor 3D Object Detection

**会议**: CVPR 2025  
**arXiv**: 2412.04458  
**主题**: 室内3D物体检测 / 长方体表示  

## 研究背景与动机

室内3D物体检测是场景理解的基础任务，对增强现实、机器人导航和智能家居等应用至关重要。该任务要求从3D扫描（点云或RGB-D图像）中定位并识别室内环境中的所有物体。

**核心挑战：**

**类别多样性**：室内场景包含大量不同类别的物体——家具、电器、装饰品、日用品等。现有数据集通常只标注有限类别，限制了模型的实用性

**场景复杂性**：室内环境中物体密集摆放，存在大量遮挡、接触和嵌套关系。例如桌上的物品、架子上的书籍等

**尺度差异巨大**：从小型物品（杯子、遥控器）到大型家具（沙发、衣柜），物体尺度跨越多个数量级

**标注成本高昂**：3D边界框标注需要专业工具和人力，比2D标注昂贵许多，限制了数据集的规模

**泛化能力不足**：现有方法通常在特定数据集上训练和评估，跨场景、跨数据集的泛化能力较弱

本文提出Cubify Anything，旨在通过长方体（cuboid）表示和可扩展的训练策略来提升室内3D物体检测的规模和泛化能力。

## 方法详解

### 长方体表示

本文选择长方体作为物体的基础3D表示。每个物体用一个带朝向的3D长方体来描述：

$$\text{Cuboid} = (x, y, z, w, h, l, \theta)$$

| 参数 | 含义 |
|------|------|
| $(x, y, z)$ | 长方体中心的3D坐标 |
| $(w, h, l)$ | 宽度、高度、长度 |
| $\theta$ | 绕竖直轴的朝向角 |

长方体表示虽然简单，但对于室内物体而言是足够的——大多数室内物体（如桌椅、柜子）可以很好地用长方体近似。

### 可扩展的检测框架

Cubify Anything的检测框架设计考虑了可扩展性：

**多模态输入支持：**

- 点云输入：直接从3D扫描获取
- RGB-D输入：通过深度图像生成点云
- 纯RGB输入：通过深度估计网络获取伪深度

**特征提取：**

使用3D骨干网络对输入的3D数据进行特征编码。骨干网络采用稀疏卷积来高效处理大规模点云数据。

**检测头设计：**

采用anchor-free的检测头，直接回归每个候选位置的长方体参数和类别概率。

### 数据扩展策略

为了提升模型的泛化能力，Cubify Anything采用了多种数据扩展策略：

| 策略 | 描述 | 目的 |
|------|------|------|
| 多数据集联合训练 | 在多个室内数据集上同时训练 | 扩大训练数据的多样性 |
| 类别统一 | 将不同数据集的类别映射到统一的类别体系 | 解决类别定义不一致的问题 |
| 伪标注生成 | 利用预训练模型为无标注数据生成伪标注 | 利用大量未标注的3D扫描数据 |
| 数据增强 | 3D空间中的旋转、缩放、翻转等 | 提升模型的鲁棒性 |

### 推理流程

推理时，模型接受3D扫描作为输入，经过特征提取和检测头后，输出场景中所有物体的长方体参数和类别。后处理步骤包括非极大值抑制（NMS）和置信度阈值筛选。

## 实验结果

### 主要评估

Cubify Anything在多个室内3D检测基准上进行了评估：

- 在标准数据集上取得了有竞争力的检测性能
- 多数据集联合训练策略有效提升了模型的跨场景泛化能力
- 相比现有方法，在物体类别覆盖范围上有显著优势

### 泛化能力分析

| 评估场景 | 表现 |
|----------|------|
| 训练集内场景 | 与专用模型性能相当 |
| 跨数据集测试 | 显著优于单数据集训练的模型 |
| 新类别物体 | 表现出一定的开放词汇检测能力 |
| 复杂场景 | 在高度杂乱的场景中保持鲁棒 |

### 消融实验

- 多数据集联合训练是提升泛化能力的关键因素
- 伪标注生成策略有效地利用了未标注数据
- 检测头的anchor-free设计优于anchor-based设计，尤其在处理不同尺度的物体时

## 总结与展望

Cubify Anything通过长方体表示和可扩展的训练策略，推动了室内3D物体检测的规模化发展。多数据集联合训练和伪标注生成策略有效提升了模型的泛化能力和类别覆盖范围。未来可以探索更精细的物体表示（如实例分割与重建），以及结合语言模型实现开放词汇的3D物体检测。

<!-- RELATED:START -->

## 相关论文

- [Approaching Outside: Scaling Unsupervised 3D Object Detection from 2D Scene](../../ECCV2024/autonomous_driving/approaching_outside_scaling_unsupervised_3d_object_detection_from_2d_scene.md)
- [PAP: A Prediction-as-Perception Framework for 3D Object Detection](a_prediction-as-perception_framework_for_3d_object_detection.md)
- [Detect Anything 3D in the Wild](../../ICCV2025/autonomous_driving/detect_anything_3d_in_the_wild.md)
- [RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)
- [Segment Anything, Even Occluded](segment_anything_even_occluded.md)

<!-- RELATED:END -->
