---
title: >-
  [论文解读] Shape-Guided Configuration-Aware Learning for Endoscopic-Image-Based Pose Estimation of Flexible Robotic Instruments
description: >-
  [ECCV 2024][医学图像][柔性机器人] 利用柔性机器人的3D形状先验引导图像特征学习，通过部件级几何表示提取和动态形状变形机制，实现了高精度的内窥镜图像柔性机器人位姿估计，在外部朝向和内部弯曲角度估计上显著超越了关键点、骨架和直接回归等基线方法。
tags:
  - ECCV 2024
  - 医学图像
  - 柔性机器人
  - 位姿估计
  - 3D形状先验
  - 内窥镜手术
  - 形状引导学习
---

# Shape-Guided Configuration-Aware Learning for Endoscopic-Image-Based Pose Estimation of Flexible Robotic Instruments

**会议**: ECCV 2024  
**机构**: 香港中文大学 / Agilis Robotics / 港大
**代码**: https://github.com/Yiyao-Ma/PoseFlex  
**领域**: 医学图像  
**关键词**: 柔性机器人, 位姿估计, 3D形状先验, 内窥镜手术, 形状引导学习

## 一句话总结

利用柔性机器人的3D形状先验引导图像特征学习，通过部件级几何表示提取和动态形状变形机制，实现了高精度的内窥镜图像柔性机器人位姿估计，在外部朝向和内部弯曲角度估计上显著超越了关键点、骨架和直接回归等基线方法。

## 研究背景与动机

**领域现状**：柔性机器人在腔内手术（如胃肠道内窥镜手术）中日益重要，精确估计其外部朝向（roll、pitch、yaw等三个角度）和内部弯曲角度对于术中导航和控制至关重要。传统方法包括基于传感器的方法和基于图像的方法两大类。

**现有痛点**：基于传感器的方法（如电磁追踪、光纤传感器）受限于成本高、环境约束（电磁干扰）、与机器人集成困难等问题。基于图像的方法虽然不需要额外传感器，但现有方案在处理柔性机器人的形状复杂性时表现不佳。具体来说，关键点检测方法难以在高自由度柔性结构上定位稳定的关键点；骨架提取方法无法处理形状剧烈变化的情况；直接回归方法缺乏有效建模形状变化的机制。

**核心矛盾**：柔性机器人的形状随弯曲角度动态变化，图像中的2D表征难以捕捉其复杂的3D几何关系。单纯依赖2D图像特征无法充分理解机器人的3D空间状态。

**本文目标** (1) 如何在图像特征学习中引入有效的3D几何信息？(2) 如何处理形状先验与实际图像中机器人形状的差异？(3) 如何利用初始估计进一步精细化位姿预测？

**切入角度**：受2D-3D联合表示学习近期进展的启发，作者观察到柔性机器人虽然形状可变，但其几何结构在部件级别具有稳定的先验知识（如圆柱体段的组合）。如果能将这种3D先验注入到图像特征中，就可以弥合2D观测与3D状态之间的gap。

**核心 idea**：用3D形状先验的部件级几何特征去查询和增强图像特征表示，并通过动态变形机制让形状先验自适应匹配实际观测，从而提升柔性机器人位姿估计精度。

## 方法详解

### 整体框架

PoseFlex 采用两阶段框架：第一阶段 **PoseEst.** 进行形状引导的位姿估计；第二阶段 **PoseRefine.** 基于初始位姿进行形状变形与位姿精细化。输入为单张内窥镜图像以及预定义的柔性机器人3D形状先验模型，输出为四个位姿参数（roll、pitch、yaw以及弯曲角度）。

在PoseEst.阶段，系统从3D形状先验中提取部件级几何表示，通过cross-attention机制将这些表示与图像特征对齐，然后用增强后的特征进行位姿回归。在PoseRefine.阶段，根据PoseEst.预测的初始位姿，对形状先验进行骨架曲线建模和圆柱体实例化来动态变形，再用变形后的形状重新引导图像特征提取，从而实现更精确的位姿估计。

### 关键设计

1. **部件级3D形状表示提取（Part-level Shape Representation）**:

    - 功能：从预定义的3D形状先验中提取具有配置感知能力的几何特征
    - 核心思路：首先根据柔性机器人的配置信息（如段数、段长度等），将3D形状先验的点云按部件（如各段圆柱体、关节等）进行标注分割。然后使用PointNet++风格的编码器提取每个部件的几何特征。这样每个部件的特征编码了该部件的局部几何信息，同时通过部件标签引入了全局配置先验。相比全局特征，部件级特征能捕获各段机器人的局部形态差异。
    - 设计动机：柔性机器人由多个可弯曲的段组成，不同段的形变模式不同，部件级表示比全局表示能更精细地描述机器人的3D结构。消融实验证明移除配置信息会导致精度持续下降。

2. **形状引导的图像特征增强（Shape-guided Image Feature Enhancement）**:

    - 功能：利用3D形状先验的部件级特征来查询并增强图像中的机器人表示
    - 核心思路：将3D形状的部件级特征作为query，图像backbone提取的特征图作为key和value，通过cross-attention机制进行交互。具体来说，每个部件的shape token会在图像特征图上寻找与自己对应的区域进行attention聚合。这样shape token吸收了对应图像区域的视觉信息，而图像特征则获得了3D几何的结构化指导。增强后的特征被送入位姿回归头预测位姿参数。
    - 设计动机：直接用图像特征回归位姿缺乏3D几何约束，而用3D形状作为结构化query可以让网络"知道应该关注哪些区域"，类似于DETR中用learnable query检测目标的思路。

3. **配置感知的动态形状变形（Configuration-aware Shape Deformation）**:

    - 功能：基于初始位姿估计，将静态的形状先验变形为与图像中实际机器人形状更接近的版本
    - 核心思路：给定PoseEst.阶段预测的初始位姿参数，首先通过弯曲角度参数化一条3D骨架曲线（基于等曲率假设的圆弧模型），然后沿骨架曲线实例化圆柱体来重建机器人的3D模型。变形后的形状先验与实际图像中的机器人更加匹配，用其提取的新一轮部件级特征可以提供更精准的形状引导，从而进一步精细化位姿估计。
    - 设计动机：最初的形状先验是静态的"默认"姿态，与实际弯曲后的机器人形状差异较大。如果不做变形，shape token查询到的图像区域可能与机器人各部件不匹配，影响引导效果。动态变形弥合了先验形状与实际形状间的gap。

### 损失函数 / 训练策略

位姿预测采用概率模型，使用Matrix Fisher分布来建模旋转矩阵的不确定性。损失函数结合了NLL损失（negative log-likelihood）用于旋转参数和L1损失用于弯曲角度。同时输出不确定性估计，高不确定性的预测倾向于具有更大的误差，可作为位姿质量的可靠指标。

## 实验关键数据

### 主实验

使用为腔内手术设计的通用柔性机器人平台进行评估，包含不同光照、遮挡和运动模糊条件。

| 方法 | Roll(Mean°) | Pitch(Mean°) | Yaw(Mean°) | Bend(Mean°) | Acc5° | Acc10° |
|------|------------|--------------|------------|-------------|-------|--------|
| Keypoint (KP) | 较高 | 较高 | 较高 | 较高 | 低 | 低 |
| Skeleton (SKL) | 较高 | 较高 | 较高 | 较高 | 低 | 低 |
| Direct Regression (DR) | 中等 | 中等 | 中等 | 中等 | 中等 | 中等 |
| SimPS | 中等 | 中等 | 中等 | 中等 | 中等 | 中等 |
| **PoseEst. (Ours)** | **更低** | **更低** | **更低** | **更低** | **更高** | **更高** |
| **PoseRefine. (Ours)** | **最低** | **最低** | **最低** | **最低** | **最高** | **最高** |

PoseEst.显著优于所有基线方法，PoseRefine.在PoseEst.基础上进一步提升精度。

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| Full model (PoseEst.) | 最佳 | 完整形状引导 |
| w/o shape guidance | 精度下降显著 | 去掉3D形状引导后各角度误差均增大 |
| w/o configuration info | 精度下降 | 去掉部件标签后引导精度降低 |
| 用Depth替代Shape | 精度更差 | DPT预测的深度噪声导致严重形状失真 |
| PoseRefine on baselines | 均有提升 | 精细化模块可泛化到其他基线方法 |

### 关键发现

- 形状引导是最大的性能贡献者，移除后所有位姿参数的误差都显著增大
- 基于深度预测的3D信息（DPT等）无法替代显式形状先验，因为单目深度估计在内窥镜场景中噪声严重
- PoseRefine.不仅能改善自己的初始预测，还能有效精细化其他baseline方法的预测结果，展现了良好的泛化性
- 在不同臂粗细、臂长度和段数的机器人配置下，方法均能平稳适应
- 在亮度异常、遮挡和运动模糊等手术常见挑战场景下，方法保持了可靠性能

## 亮点与洞察

- **3D形状先验作为结构化Query的思路非常巧妙**。不同于DETR中的可学习query，这里用具有物理意义的3D几何作为query，既引入了先验知识又保留了可解释性。这种"用3D模型去查询2D图像"的范式可以迁移到其他需要2D-3D对齐的任务。
- **两阶段"粗估计-变形-精细化"的迭代策略**设计得很自然。用初始位姿来变形先验模型，让第二轮引导更精确，形成正向循环。这种coarse-to-fine with geometric feedback的思路可以用在更多涉及可变形物体的位姿估计场景。
- **概率性位姿模型同时输出估计值和不确定性**，不确定性与实际误差高度相关，为临床应用提供了可靠性指标。

## 局限与展望

- 当前方法需要预先定义机器人的3D形状先验模型，对于未知构型的机器人需要重新建模
- 等曲率假设在复杂多段弯曲场景下可能不够精确
- 只在单一类型的柔性手术机器人上验证，对导管、内窥镜等其他柔性器械的泛化性有待验证
- 未利用时序信息，结合视频序列的时序一致性约束可能进一步提升精度
- 实时性能未详细报告，临床部署需要考虑推理速度约束

## 相关工作与启发

- **vs Keypoint-based methods**: 关键点方法在刚性物体上效果好，但柔性机器人的高自由度变形使得稳定关键点难以定位。本文通过部件级3D先验直接绕开了关键点检测的难题。
- **vs SimPS (基于仿真的Photometric Stereo)**: SimPS用仿真渲染辅助训练，但缺乏有效的形状变化建模机制。本文的形状引导方案提供了更强的几何约束。
- **vs Depth-based 3D methods**: 用预训练深度估计模型获取3D信息的方案效果不如显式形状先验，说明在特殊成像条件下（内窥镜低纹理、光照不均），领域专用的先验知识比通用预训练模型更有效。

## 评分

- 新颖性: ⭐⭐⭐⭐ 3D形状先验引导的框架设计新颖，部件级查询和动态变形机制有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 消融实验全面，包含多种配置变化、环境挑战和交叉验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，项目主页做得详细
- 价值: ⭐⭐⭐⭐ 对手术机器人位姿估计有实用价值，框架思路可迁移

<!-- RELATED:START -->

## 相关论文

- [Benchmarking Endoscopic Surgical Image Restoration and Beyond](../../CVPR2026/medical_imaging/benchmarking_endoscopic_surgical_image_restoration_and_beyond.md)
- [A Rotation-Invariant Texture ViT for Fine-Grained Recognition of Esophageal Cancer Endoscopic Ultrasound Images](a_rotation-invariant_texture_vit_for_fine-grained_recognition_of_esophageal_canc.md)
- [Pathology-knowledge Enhanced Multi-instance Prompt Learning for Few-shot Whole Slide Image Classification](pathology-knowledge_enhanced_multi-instance_prompt_learning_for_few-shot_whole_s.md)
- [Unlocking Positive Transfer in Incrementally Learning Surgical Instruments: A Self-reflection Hierarchical Prompt Framework](../../CVPR2026/medical_imaging/unlocking_positive_transfer_in_incrementally_learning_surgical_instruments_a_sel.md)
- [Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](../../CVPR2026/medical_imaging/learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)

<!-- RELATED:END -->
