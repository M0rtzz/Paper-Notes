---
title: >-
  [论文解读] SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection
description: >-
  [ICLR 2026][目标检测][旋转目标检测] 提出 SPWOOD 框架统一处理稀疏标注和弱标注（HBox/Point）的旋转目标检测问题，通过自适应旋转目标检测器(SAOD)和空间布局学习策略，在 DOTA 基准上以混合标注（RBox:HBox:Point=1:1:1）达到接近全监督的性能。
tags:
  - "ICLR 2026"
  - "目标检测"
  - "旋转目标检测"
  - "弱监督"
  - "稀疏标注"
  - "半监督"
  - "遥感"
---

# SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection

**会议**: ICLR 2026  
**arXiv**: [2602.03634](https://arxiv.org/abs/2602.03634)  
**代码**: 无  
**领域**: 目标检测 / 遥感  
**关键词**: 旋转目标检测, 弱监督, 稀疏标注, 半监督, 遥感

## 一句话总结
提出 SPWOOD 框架统一处理稀疏标注和弱标注（HBox/Point）的旋转目标检测问题，通过自适应旋转目标检测器(SAOD)和空间布局学习策略，在 DOTA 基准上以混合标注（RBox:HBox:Point=1:1:1）达到接近全监督的性能。

## 研究背景与动机

**领域现状**：旋转目标检测（OOD）在遥感等领域至关重要，但精确的旋转框（RBox）标注成本极高——需要标注中心点、宽高和旋转角度。

**现有痛点**：现有降低标注成本的方法要么只处理弱标注（如用水平框 HBox 或点标注 Point 替代 RBox），要么只处理稀疏标注（只标注部分实例），但实际场景中两种问题同时存在。

**核心矛盾**：稀疏标注（不是所有实例都标了）和弱标注（标了但不精确）各自都会导致严重的训练信号缺失，二者叠加使问题更加困难——未标注实例可能被当作负例训练，弱标注可能引导错误的角度学习。

**本文目标** 在同时存在稀疏和弱标注的极端低成本设置下，如何训练高质量的旋转目标检测器？

**切入角度**：设计统一框架同时从三种不同质量的标注（RBox、HBox、Point）中学习，并通过自我训练挖掘未标注实例。

**核心 idea**：通过自适应旋转检测器统一处理精确/弱/无标注三种信号，结合空间布局学习和角度一致性约束来恢复旋转信息。

## 方法详解

### 整体框架
SPWOOD 在 teacher-student 自训练框架基础上，引入三个关键组件：(1) SAOD 基线检测器处理 RBox 和通过弱标注恢复的伪 RBox；(2) 角度学习模块从图像增强的角度一致性中学习旋转；(3) 空间布局学习从 Voronoi 分水岭分析中恢复点标注的尺度信息。

### 关键设计

1. **自适应旋转目标检测器 (SAOD)**:

    - 做什么：统一处理不同质量标注的检测器基线
    - 核心思路：对 RBox 标注使用标准旋转检测损失；对 HBox 标注，从水平框推断可能的旋转范围，生成伪 RBox 用于训练；对 Point 标注，仅使用中心点位置信息，尺度通过空间布局学习恢复。Teacher 模型在强标注上训练后对未标注区域生成伪标签
    - 设计动机：不同标注类型提供不同维度的信息，需要差异化处理而非统一忽略

2. **角度学习 (Angle Learning)**:

    - 做什么：从标注不含角度信息的 HBox/Point 中恢复旋转角度
    - 核心思路：利用图像增强的几何一致性——对同一图像进行翻转或旋转增强（旋转角 $\mathcal{R}$），要求检测器预测的角度满足 $\theta_{rot} = \theta + \mathcal{R}$。损失为 $\mathcal{L}_{Ang}^s = \text{SmoothL1}(\theta_{flp} - \theta_{flp}^{gt}) + \text{SmoothL1}(\theta_{rot} - \theta - \mathcal{R})$
    - 设计动机：角度是旋转检测中最难从弱标注恢复的信息，但几何变换提供了自然的自监督信号

3. **空间布局学习 (Spatial Layout Learning)**:

    - 做什么：从仅有点标注的实例中恢复目标的宽度和高度
    - 核心思路：对每个点标注构建 Voronoi 图将图像分割为每个点对应一个区域，然后用分水岭算法在每个区域内基于像素相似性获得像素级分类。将分水岭结果旋转对齐后得到宽高回归目标。Voronoi 分水岭损失为 $\mathcal{L}_W^s = L_{GWD}(\text{pred}, \text{watershed\_target})$
    - 设计动机：Voronoi 图天然将相邻目标分离，分水岭提供了基于外观的尺度估计

### 损失函数 / 训练策略
总损失 = 检测损失（标准 RBox 回归+分类）+ 角度一致性损失 $\mathcal{L}_{Ang}$ + 空间布局损失 $\mathcal{L}_W$ + 高斯重叠损失 $\mathcal{L}_O$（约束预测框不重叠）。Teacher-student 框架使用 EMA 更新。

## 实验关键数据

### 主实验

| 方法 | 类型 | 10%稀疏·10%部分 | 20%·20% | 30%·20% |
|------|------|-----------------|---------|---------|
| H2RBox-v2 | 弱监督(HBox) | 30.6 | 42.7 | 49.2 |
| MCL | 半监督(RBox) | 31.7 | 44.5 | 47.8 |
| PWOOD | 部分弱监督(RBox) | 38.0 | 51.9 | 55.2 |
| RSST | 稀疏监督(RBox) | 43.4 | 52.3 | 56.6 |
| **SPWOOD (RBox)** | **稀疏+弱** | **48.5** | **57.8** | **60.3** |
| SPWOOD (HBox) | 稀疏+弱 | 45.5 | 54.0 | 56.5 |
| SPWOOD (R:H:P=1:1:1) | 混合 | 42.4 | 53.0 | 54.8 |

### 消融实验

| 配置 | mAP (10%·10%) | 说明 |
|------|--------------|------|
| SPWOOD 完整 | 48.5 | 所有组件 |
| 无角度学习 | ~43 | 弱标注角度不准 |
| 无空间布局 | ~44 | 点标注尺度恢复差 |
| 无 teacher-student | ~40 | 未标注实例浪费 |

### 关键发现
- SPWOOD (RBox) 在所有稀疏-部分比例下都显著超越现有方法，最大提升 5+ mAP
- 即使使用混合标注 (R:H:P=1:1:1)，仍能达到接近全 RBox 稀疏监督的性能
- 角度学习对弱标注场景贡献最大
- 空间布局学习在极稀疏设置下更为关键

## 亮点与洞察
- **统一框架处理多种标注类型**：不同标注提供不同质量的信息，SPWOOD 优雅地整合三种信号源
- **几何一致性的巧妙利用**：通过图像增强的角度约束来自监督学习角度信息，不需要任何角度标注

## 局限与展望
- Voronoi 分水岭在密集目标场景中可能失效
- 角度学习假设增强变换已知，不适用于自然场景中的未知视角变化
- 仅在 DOTA 遥感数据上评估，对自然图像的旋转检测效果未知

## 相关工作与启发
- **vs Point2RBox**: 仅从点标注恢复旋转框，不处理稀疏标注问题
- **vs PWOOD**: 处理部分弱监督但不处理稀疏（假设所有实例至少有弱标注）

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次统一处理稀疏+弱标注的旋转检测
- 实验充分度: ⭐⭐⭐⭐ 多种标注比例、多种方法对比
- 写作质量: ⭐⭐⭐ 方法描述清晰但公式较密
- 价值: ⭐⭐⭐⭐ 对低成本遥感检测有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Fourier Angle Alignment for Oriented Object Detection in Remote Sensing](../../CVPR2026/object_detection/fourier_angle_alignment_for_oriented_object_detection_in_remote_sensing.md)
- [\[ICLR 2026\] Long-Context Generalization with Sparse Attention](long-context_generalization_with_sparse_attention.md)
- [\[AAAI 2026\] TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding](../../AAAI2026/object_detection/tubermc_tube-conditioned_reconstruction_with_mutual_constraints_for_weakly-super.md)
- [\[ECCV 2024\] Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation](../../ECCV2024/object_detection/projecting_points_to_axes_oriented_object_detection_via_point-axis_representatio.md)
- [\[CVPR 2025\] SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection](../../CVPR2025/object_detection/simltd_simple_supervised_and_semi-supervised_long-tailed_object_detection.md)

</div>

<!-- RELATED:END -->
