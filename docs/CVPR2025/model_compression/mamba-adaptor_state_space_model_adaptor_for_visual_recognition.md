---
title: >-
  [论文解读] Mamba-Adaptor: State Space Model Adaptor for Visual Recognition
description: >-
  [CVPR 2025][模型压缩][Mamba适配器] 提出 Mamba-Adaptor，通过两个模块增强 Vision Mamba/SSM：Adaptor-T（时序）用可学习记忆选择机制保留关键历史状态，Adaptor-S（空间）用多尺度空心深度卷积增强空间局部性，在 ImageNet 上 83.0% Top-1（Mamba-Adaptor-b2），检测/分割+迁移学习全面提升。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "Mamba适配器"
  - "状态空间模型"
  - "可学习记忆选择"
  - "多尺度空间卷积"
  - "迁移学习"
---

# Mamba-Adaptor: State Space Model Adaptor for Visual Recognition

**会议**: CVPR 2025  
**arXiv**: [2505.12685](https://arxiv.org/abs/2505.12685)  
**代码**: 无  
**领域**: 模型压缩 / 高效适配  
**关键词**: Mamba适配器, 状态空间模型, 可学习记忆选择, 多尺度空间卷积, 迁移学习

## 一句话总结

提出 Mamba-Adaptor，通过两个模块增强 Vision Mamba/SSM：Adaptor-T（时序）用可学习记忆选择机制保留关键历史状态，Adaptor-S（空间）用多尺度空心深度卷积增强空间局部性，在 ImageNet 上 83.0% Top-1（Mamba-Adaptor-b2），检测/分割+迁移学习全面提升。

## 研究背景与动机

### 领域现状

**领域现状**：Vision Mamba 等 SSM 模型以线性复杂度处理长序列，但两个固有缺陷限制了其视觉性能：（1）固定的状态衰减机制导致重要历史信息被遗忘；（2）1D 序列处理忽略了图像的 2D 空间结构。

**现有痛点**：SSM 中选择性状态衰减（$\Delta, A, B$ 参数）是数据驱动的，但没有显式机制保护关键历史状态不被衰减。远距离 token 的影响随时间指数衰减，即使它们包含重要信息。

**核心矛盾**：SSM 的线性效率来源于递推结构（只保留隐状态），但这与保留丰富历史信息相矛盾。

**切入角度**：在 SSM 的隐状态上加可学习的记忆选择层——用线性层预测 K 个关键状态的坐标并保留，多序列聚合不同尺度的时序信息。

**核心 idea**：可学习记忆选择（时序）+ 多尺度空心卷积（空间）= SSM 的轻量增强适配器。

## 方法详解

### 关键设计

1. **Adaptor-T（时序增强）**:

    - 功能：在 SSM 状态中保留关键历史信息
    - 核心思路：线性预测层从当前隐状态中选择 K 个关键坐标，提取对应的状态值保留。多序列（S 个）各自维护不同粒度的记忆窗口，聚合后注入回 SSM
    - 设计动机：消融显示可学习选择比静态选择好 +0.3% ImageNet，多尺度比单尺度好 +0.2-0.7%

2. **Adaptor-S（空间增强）**:

    - 功能：恢复 SSM 丢失的 2D 空间局部性
    - 核心思路：多尺度空心（dilated）深度卷积在不同感受野上提取局部空间特征，与 SSM 全局特征融合
    - 设计动机：SSM 将 2D 图像展平为 1D 序列，破坏了局部空间关系

### 损失函数 / 训练策略

标准分类/检测损失。迁移学习时权重共享减少 94% 参数。Adaptor 增加 <7% FLOPs。

## 实验关键数据

### 主实验

| 模型 | ImageNet Top-1 | COCO Box AP |
|------|---------------|-------------|
| VMamba-T | 82.6% | 45.3% |
| Swin-T | 81.3% | - |
| **Mamba-Adaptor-b2** | **83.0%** | **49.1%** |

### 消融实验

| 配置 | ImageNet | 说明 |
|------|----------|------|
| 静态选择 | 82.7% | — |
| **可学习选择** | **83.0%** | +0.3% |
| 单尺度 | 82.3% | — |
| **多尺度** | **83.0%** | +0.7% |

### 关键发现
- 时序和空间增强各自贡献约 0.3%，合计 0.4%（有重叠）
- 迁移学习仅 9.25% 参数即达全量微调 99% 性能
- COCO 检测 +3.8% AP（49.1 vs 45.3），说明空间局部性对检测尤为重要

## 亮点与洞察
- **为 SSM 的两个根本缺陷提供了轻量解法**——不改变 SSM 核心架构，只加适配器
- **迁移学习的高效性**——94% 参数节省且性能接近全量微调

## 局限与展望
- 大模型规模未探索
- 仅适用于 Mamba/SSM 变体
- 计算开销虽小但仍增加 7%

## 评分
- 新颖性: ⭐⭐⭐ 可学习记忆选择新颖，但整体框架偏工程
- 实验充分度: ⭐⭐⭐⭐ 分类/检测/分割/迁移多任务
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐ 增量性提升，对 SSM 社区有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality](efficientvim_efficient_vision_mamba_with_hidden_state_mixer_based_state_space_du.md)
- [\[CVPR 2025\] MobileMamba: Lightweight Multi-Receptive Visual Mamba Network](mobilemamba_lightweight_multi-receptive_visual_mamba_network.md)
- [\[CVPR 2025\] MambaIC: State Space Models for High-Performance Learned Image Compression](mambaic_state_space_models_for_high-performance_learned_image_compression.md)
- [\[CVPR 2025\] JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba](jamma_ultra-lightweight_local_feature_matching_with_joint_mamba.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)

</div>

<!-- RELATED:END -->
