---
title: >-
  [论文解读] Solving Instance Detection from an Open-World Perspective
description: >-
  [CVPR 2025][机器人][实例检测] 从开放世界视角出发，通过度量学习适配基础模型特征、干扰物采样和NeRF新视角合成三种策略，显著提升实例检测中的实例级特征匹配性能，在CID和NID两种设定下均大幅超越前人方法。
tags:
  - CVPR 2025
  - 机器人
  - 实例检测
  - 开放世界
  - 基础模型适配
  - 度量学习
  - NeRF数据增强
---

# Solving Instance Detection from an Open-World Perspective

**会议**: CVPR 2025  
**arXiv**: [2503.00359](https://arxiv.org/abs/2503.00359)  
**代码**: [项目主页](https://shenqq377.github.io/IDOW)  
**领域**: Robotics / Object Detection  
**关键词**: 实例检测, 开放世界, 基础模型适配, 度量学习, NeRF数据增强

## 一句话总结

从开放世界视角出发，通过度量学习适配基础模型特征、干扰物采样和NeRF新视角合成三种策略，显著提升实例检测中的实例级特征匹配性能，在CID和NID两种设定下均大幅超越前人方法。

## 研究背景与动机

实例检测（InsDet）旨在根据给定的视觉参考图像，在新场景中定位特定物体实例。其开放世界特性带来核心挑战：(1) 测试场景在训练时完全未知；(2) 视觉参考与检测提案之间存在域差距（如遮挡、光照变化）。

现有方法在不同维度部分利用了开放世界信息：CPL采样随机背景图像合成训练数据、VoxDet利用外部3D数据集学习体素表示、OTS-FM直接使用SAM和DINOv2等现成基础模型。然而，关键发现是：虽然预训练的开放世界检测器能获得很高的召回率，但基础模型（如DINOv2）并未针对实例级特征匹配进行优化——直接使用FM进行InsDet是次优的。

本文的核心动机是：全面利用开放世界中可获取的数据和基础模型，通过度量学习适配FM以获得更具区分力的实例级特征表示。

## 方法详解

### 整体框架

IDOW方法包含两个阶段：(1) 使用预训练开放世界检测器（如SAM或GroundingDINO）进行提案检测，获取高召回率；(2) 通过度量学习适配基础模型DINOv2，使其产生更适合实例级匹配的特征，并利用干扰物采样和NeRF新视角合成两种数据增强技术提升FM适配效果。最终通过稳定匹配算法将提案与视觉参考配对。

### 关键设计1：基于度量学习的基础模型适配

- **功能**: 将通用视觉基础模型的特征空间调整为适合实例级匹配的度量空间
- **核心思路**: 构建三元组$(I_a, I_p, I_n)$训练数据，使用三元组损失$\ell = [d(f_\theta(I_a), f_\theta(I_p)) - d(f_\theta(I_a), f_\theta(I_n)) + \alpha]_+$微调DINOv2，使同一实例的特征更近、不同实例更远。采用批级别困难负例挖掘策略
- **设计动机**: DINOv2虽然提供了优秀的通用视觉特征，但未针对实例级别的精确匹配优化。通过度量学习适配可以显著提升匹配精度（+5-7 AP）

### 关键设计2：干扰物采样（Distractor Sampling）

- **功能**: 从开放世界图像中采样通用负样本，定义开放空间边界
- **核心思路**: 在随机背景图像上运行SAM生成物体候选分割，将这些分割作为universal negative data加入度量学习的三元组训练。这些干扰物帮助特征更好地区分有意义的物体实例与背景杂物
- **设计动机**: 仅使用已知实例作为负例不足以覆盖测试时可能遇到的各种干扰物体，引入开放世界采样的干扰物可以增强特征的鲁棒性和区分能力

### 关键设计3：NeRF新视角合成

- **功能**: 为每个物体实例生成更多视角的合成参考图像，增强视觉参考的多样性
- **核心思路**: 对每个物体实例训练一个Zip-NeRF，利用COLMAP估计的相机位姿从新视角渲染合成图像。合成图像不仅用于训练中的数据增强，还在测试时作为额外的视觉参考存储
- **设计动机**: 在CID设定下视觉参考图像数量有限，通过NeRF合成可以大幅增加视角多样性，帮助FM学到更鲁棒的视角不变特征。特别创新之处在于合成图像也在测试阶段使用

### 损失函数

使用标准的三元组损失（Triplet Loss），距离度量采用反余弦相似度，配合批级别困难负例挖掘。最终匹配使用稳定匹配算法并设置相似度阈值（0.4）过滤低置信度匹配。

## 实验关键数据

### 主实验：CID设定在HR-InsDet数据集上

| 方法 | AP | AP50 | AP75 |
|------|-----|------|------|
| CPL_DINO | 27.99 | 39.62 | 32.19 |
| OTS-FM_SAM | 41.61 | 49.10 | 45.95 |
| OTS-FM_GroundingDINO | 51.68 | 62.50 | 56.78 |
| **IDOW_SAM** | **48.75** | **57.59** | **54.06** |
| **IDOW_GroundingDINO** | **57.01** | **69.33** | **62.84** |

### 消融实验：各组件贡献

| 配置 | AP |
|------|-----|
| OTS-FM (baseline) | 41.61 |
| + 度量学习适配 | ~46 |
| + 干扰物采样 | ~47 |
| + NeRF新视角合成 | 48.75 |

### 关键发现

- FM适配带来5-7 AP的显著提升，证明通用FM特征在实例级匹配上的优化空间很大
- 使用更强的开放世界检测器（GroundingDINO vs SAM）始终带来8+ AP提升
- NeRF合成的图像在测试阶段也有用——将合成参考加入匹配可进一步提升性能
- IDOW在CID和NID两种设定下均取得最优，超越前人10+ AP

## 亮点与洞察

1. **开放世界视角统一现有方法**: 论文将背景采样、外部数据利用、FM使用统一在"开放世界信息利用"的框架下理解，提供了清晰的方法论视角
2. **FM适配 vs 直接使用FM**: 实验证明即使最强FM也需要针对特定任务适配，直接使用是次优的
3. **NeRF不仅用于训练数据增强也用于测试**: 将NeRF合成的参考图像存储用于测试匹配是实用的创新

## 局限与展望

- 需要为每个物体单独训练NeRF，计算开销较大
- 依赖COLMAP进行参考图像的位姿估计，对少量参考可能不稳定
- 方法在小物体检测上仍有提升空间（AP_small仅35.25）

## 相关工作与启发

- 实例检测与开放词汇检测的区别在于需要实例级而非类别级匹配
- 度量学习适配FM的思路可推广到其他需要精细匹配的任务（如ReID、细粒度检索）

## 评分

⭐⭐⭐⭐ — 方法简洁有效，开放世界视角提供了统一理解框架。10+ AP的性能提升令人信服。NeRF在测试时也使用是巧妙的设计。但方法新颖性主要在组合层面。

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](../../NeurIPS2025/robotics/mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)
- [\[NeurIPS 2025\] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World](../../NeurIPS2025/robotics/c-nav_towards_self-evolving_continual_object_navigation_in_open_world.md)
- [\[ICML 2025\] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](../../ICML2025/robotics/founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci.md)
- [\[CVPR 2026\] IGen: Scalable Data Generation for Robot Learning from Open-World Images](../../CVPR2026/robotics/igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)
- [\[ICCV 2025\] Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective](../../ICCV2025/robotics/beyond_losses_reweighting_empowering_multi-task_learning_via_the_generalization_.md)

<!-- RELATED:END -->
