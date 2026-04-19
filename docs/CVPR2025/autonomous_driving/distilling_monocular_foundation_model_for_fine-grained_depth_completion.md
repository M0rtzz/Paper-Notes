---
title: >-
  [论文解读] Distilling Monocular Foundation Model for Fine-grained Depth Completion
description: >-
  [CVPR 2025][自动驾驶][深度补全] 本文提出DMD3C，一个两阶段知识蒸馏框架，将单目深度基础模型（如Depth Anything V2）的几何知识迁移到深度补全网络，第一阶段通过合成训练数据进行预训练，第二阶段通过尺度-偏移不变损失（SSI Loss）在真实数据上微调，在KITTI深度补全排行榜上取得第一名。
tags:
  - CVPR 2025
  - 自动驾驶
  - 深度补全
  - 知识蒸馏
  - 单目深度估计
  - 基础模型
  - 尺度不变损失
---

# Distilling Monocular Foundation Model for Fine-grained Depth Completion

**会议**: CVPR 2025  
**arXiv**: [2503.16970](https://arxiv.org/abs/2503.16970)  
**代码**: [Sharpiless/DMD3C](https://github.com/Sharpiless/DMD3C)  
**领域**: 自动驾驶  
**关键词**: 深度补全, 知识蒸馏, 单目深度估计, 基础模型, 尺度不变损失

## 一句话总结
本文提出DMD3C，一个两阶段知识蒸馏框架，将单目深度基础模型（如Depth Anything V2）的几何知识迁移到深度补全网络，第一阶段通过合成训练数据进行预训练，第二阶段通过尺度-偏移不变损失（SSI Loss）在真实数据上微调，在KITTI深度补全排行榜上取得第一名。

## 研究背景与动机

深度补全任务的目标是从稀疏LiDAR输入预测稠密深度图，在自动驾驶、机器人和增强现实等领域有广泛应用。当前方法的核心挑战在于：训练数据的稀疏性严重限制了模型学习细粒度几何特征的能力。以KITTI数据集为例，LiDAR稀疏深度仅覆盖约5%的图像像素，经过多帧融合等后处理，ground truth的覆盖率也仅约20%，远距离和动态物体区域通常无法获得标注。

另一方面，单目深度估计基础模型（如Depth Anything V2、MiDaS）已经能够从单张RGB图像预测出包含丰富几何细节和相对深度关系的稠密深度图。但这些模型存在固有的尺度模糊性（scale ambiguity），无法直接提供真实世界的度量深度。

本文的核心思路是：利用单目基础模型提供稠密监督信号来弥补稀疏标注的不足，同时通过精心设计的两阶段蒸馏策略解决尺度模糊性问题。

## 方法详解

### 整体框架
DMD3C采用两阶段蒸馏框架：第一阶段通过数据生成策略从无标签自然图像预训练深度补全模型，学习多样化的几何先验；第二阶段在有标签的真实数据集上微调，利用SSI Loss结合稀疏监督和单目深度蒸馏，学习真实世界尺度。

### 关键设计

1. **第一阶段：数据生成与预训练**:

    - 目标：从海量无标签自然图像中学习几何特征，无需任何ground truth深度
    - 数据来源：混合使用COCO（118K）、Google Landmarks（117K）、nuScenes（93K）、Cityscapes（20K）、DAVIS（10K）等约36万张多样化图像
    - 数据生成流程：
      (a) 用Depth Anything V2预测每张图像的单目深度图
      (b) 随机采样相机内参矩阵K
      (c) 将深度图反投影为3D点云
      (d) 通过泊松表面重建生成连续3D网格
      (e) 模拟LiDAR射线扫描过程生成稀疏深度点
    - 预训练时使用L1 loss，以合成稠密深度为监督
    - 关键创新：通过网格重建和射线模拟，生成的稀疏深度与真实LiDAR扫描模式高度一致

2. **第二阶段：带SSI Loss的微调**:

    - 在有稀疏ground truth的真实数据集（KITTI、NYUv2）上微调
    - 稀疏L1监督：用稀疏ground truth提供真实世界尺度
    - 尺度-偏移不变损失（SSI Loss）：解决单目深度的尺度模糊性
        - 目标：找到最优的尺度参数s和偏移参数b，使 |D_f - (s·D_m + b)| 最小
        - 通过对齐预测深度和单目深度的全局尺度和偏移，学习保持相对深度结构
    - 梯度匹配正则项：保持深度边缘的锐利度，在K=4个尺度级别上计算
    - 最终损失 = L_sup + L_SSIL + L_reg

3. **网络架构compatibility**:

    - 方法是训练策略级别的创新，不依赖特定网络架构
    - 主实验使用BP-Net（基于ResNet块），同时验证了在LRRU和CFormer上的适用性
    - 所有架构均获得一致的性能提升

### 损失函数 / 训练策略
- 第一阶段：L1预训练损失，纯合成数据，从零训练600K iterations
- 第二阶段：L_sup（稀疏L1）+ L_SSIL（尺度不变蒸馏）+ L_reg（梯度正则），微调300K iterations
- 优化器：AdamW，weight decay 0.05，梯度裁剪阈值0.1
- 使用EMA（decay=0.9999）稳定训练
- 学习率策略：OneCycle
- 4块A100 GPU，batch size 16

## 实验关键数据

### 主实验
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| KITTI | RMSE (mm) ↓ | 678.12 | 684.90 (BP-Net) | -6.78 |
| KITTI | MAE (mm) ↓ | 194.46 | 187.95 (ImprovingDC) | - |
| KITTI | iRMSE ↓ | 1.82 | 1.82 (BP-Net/TPVD) | 持平 |
| NYUv2 | RMSE ↓ | 0.085 | 0.086 (TPVD) | -0.001 |
| NYUv2 | δ1.25 ↑ | 99.7 | 99.7 (TPVD) | 持平 |

### 消融实验
| 配置 | RMSE ↓ | 说明 |
|------|---------|------|
| DMD3C（完整） | 678.12 | 两阶段蒸馏 |
| w/o Pre-train | 682.34 (+4.22) | 移除第一阶段预训练 |
| w/o SSI Loss | 684.54 (+6.42) | 移除SSI Loss |
| BP-Net baseline | 684.90 (+6.78) | 原始BP-Net |
| LRRU baseline | 696.51 | 原始LRRU |
| LRRU + Ours | 693.17 (-3.34) | 通用性验证 |
| CFormer baseline | 764.87 | 原始CFormer |
| CFormer + Ours | 760.29 (-4.58) | 通用性验证 |

### 跨域零样本泛化
| 数据集 | 本文 RMSE | BP-Net RMSE | 提升 |
|--------|-----------|-------------|------|
| ScanNet | 0.101 | 0.122 | -17.2% |
| DDAD | 7.766 | 8.903 | -12.8% |
| VOID1500 | 0.676 | 0.704 | -4.0% |

### 关键发现
- 预训练和SSI Loss各自贡献约4-6mm的RMSE改善，两者互补
- 方法对不同网络架构具有普适性，BP-Net、LRRU、CFormer均获益
- 零样本跨域性能显著优于baseline，特别在DDAD上RMSE降低12.8%，表明预训练显著增强了泛化能力
- 在无ground truth的区域也能生成高质量深度，3D点云可视化更完整、更连贯

## 亮点与洞察
- 思路简洁优雅：用已有的单目基础模型作为"知识教师"来补充稀疏标注的不足
- 第一阶段的数据生成管线设计巧妙：通过网格重建+射线模拟生成合成LiDAR，避免了domain gap
- SSI Loss的设计解决了单目深度尺度模糊这一根本性问题，使得稠密蒸馏在真实场景中可行
- 方法是训练策略层面的创新，可以即插即用地提升任意深度补全架构
- KITTI排行榜第一的成绩具有很强的说服力

## 局限与展望
- RMSE指标虽然排名第一，但MAE并非最优，说明在某些区域可能仍有较大误差
- 第一阶段预训练需要600K iterations，计算成本较高
- 单目基础模型本身的质量直接影响蒸馏效果，对于基础模型表现不佳的场景（如高反射、透明物体），蒸馏效果可能受限
- 仅在KITTI和NYUv2上做了定量评估，其他深度补全benchmark（如Waymo）未涉及
- 梯度匹配正则项的超参数（尺度级别数K=4）缺乏敏感性分析
- 动态场景中快速运动物体的深度补全效果未专门分析

## 相关工作与启发
- 与TPVD、ImprovingDC等KITTI榜单上的竞争方法相比，本文通过引入外部知识（单目基础模型）获得了决定性优势
- SSI Loss的思想来源于G2-MonoDepth和MiDaS，但本文在深度补全任务中首次系统性地利用它
- 预训练策略受Depth Anything系列的启发，但创新地通过网格重建和LiDAR模拟生成了适合深度补全的训练数据
- 该蒸馏框架的思想可以推广到其他稀疏监督任务（如稀疏到稠密的光流估计、点云补全等）

## 评分
- 新颖性: ⭐⭐⭐⭐ 两阶段蒸馏框架设计合理，SSI Loss在深度补全中的应用有新意，但核心组件多为已有技术的组合
- 实验充分度: ⭐⭐⭐⭐ 多数据集评估、多架构验证、零样本泛化实验都很全面，但仅限两个主基准
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但数学符号在某些地方略显冗余
- 价值: ⭐⭐⭐⭐⭐ KITTI第一名是最有力的证明，且方法的通用性和即插即用特性极具实用价值

<!-- RELATED:START -->

## 相关论文

- [Distilling Multi-modal Large Language Models for Autonomous Driving](distilling_multi-modal_large_language_models_for_autonomous_driving.md)
- [PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency](../../CVPR2026/autonomous_driving/ptc-depth_pose-refined_monocular_depth_estimation_with_temporal_consistency.md)
- [Fine-Grained Representation for Lane Topology Reasoning](../../AAAI2026/autonomous_driving/fine-grained_representation_for_lane_topology_reasoning.md)
- [HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning](../../NeurIPS2025/autonomous_driving/holollm_multisensory_foundation_model_for_language-grounded_human_sensing_and_re.md)
- [Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation](helvipad_a_real-world_dataset_for_omnidirectional_stereo_depth_estimation.md)

<!-- RELATED:END -->
