---
title: >-
  [论文解读] Continuous 3D Perception Model with Persistent State
description: >-
  [CVPR 2025][3D视觉][在线3D重建] 提出CUT3R（Continuous Updating Transformer for 3D Reconstruction），一个维持持续内部状态的循环模型，能从图像流中在线、增量地进行度量级3D重建、相机位姿估计，并能推断未观测区域的3D结构。
tags:
  - CVPR 2025
  - 3D视觉
  - 在线3D重建
  - 持续感知
  - 循环状态
  - 点图预测
  - 动态场景
---

# Continuous 3D Perception Model with Persistent State

**会议**: CVPR 2025  
**arXiv**: [2501.12387](https://arxiv.org/abs/2501.12387)  
**代码**: [cut3r.github.io](https://cut3r.github.io/)  
**领域**: 3D视觉 / 3D重建  
**关键词**: 在线3D重建, 持续感知, 循环状态, 点图预测, 动态场景

## 一句话总结

提出CUT3R（Continuous Updating Transformer for 3D Reconstruction），一个维持持续内部状态的循环模型，能从图像流中在线、增量地进行度量级3D重建、相机位姿估计，并能推断未观测区域的3D结构。

## 研究背景与动机

### 领域现状

**领域现状**：传统3D重建方法（SfM、SLAM、NeRF等）从头开始处理每个场景（tabula rasa），难以应对稀疏观测、退化运动等情况

### 现有痛点

**现有痛点**：DUSt3R等学习方法仅能处理图像对，扩展到多视图需要耗时的全局对齐，且无法在线更新

### 核心矛盾

**核心矛盾**：Spann3R虽支持连续重建，但其空间记忆仅作为缓存，不能推断未观测区域

### 解决思路

**解决思路**：人类是在线视觉学习者：持续处理视觉流、累积观测、利用先验知识推断遮挡区域

### 补充说明

**补充说明**：需要一个统一框架：(1)从少量观测重建3D，(2)随新观测持续细化，(3)推断未观测区域

## 方法详解

### 整体框架

CUT3R使用ViT编码器将每帧图像编码为visual tokens，这些tokens与一组持续维护的状态tokens进行双向交互——状态更新（将新信息融入状态）和状态读出（从状态中检索历史上下文），产生世界坐标系和相机坐标系下的稠密点图及相机位姿。

### 关键设计

1. **持续状态机制（Persistent State）**:
    - 功能：以固定大小的token集合编码对场景的累积理解
    - 核心思路：状态由768个768维可学习token组成，初始化共享。图像tokens通过两个互联Transformer解码器与状态双向交互：状态更新融入新观测，状态读出提取历史上下文
    - 设计动机：固定大小的压缩状态不仅缓存已观测内容，还能编码未观测区域的推断，类似人类对环境的心理模型

2. **虚拟视角查询（Unseen View Query）**:
    - 功能：在不添加新观测的情况下，从状态中推断未见区域的3D结构和颜色
    - 核心思路：将虚拟相机的内参/外参编码为raymap（6通道图像，每像素编码光线原点和方向），经ViT编码后与状态交互读出（不更新状态），预测对应的点图和颜色
    - 设计动机：类比MAE中的patch补全，此处是图像级别的补全，利用状态中编码的场景全局上下文

3. **冗余预测与部分标注训练**:
    - 功能：预测自身坐标系和世界坐标系两份点图及6-DoF位姿
    - 核心思路：$\hat{X}_t^{\text{self}}, \hat{X}_t^{\text{world}}, \hat{P}_t$ 分别由 HeadSelf（DPT）、HeadWorld（DPT+pose token）、HeadPose（MLP）输出
    - 设计动机：看似冗余但简化训练——每个输出独立监督，且支持在仅有位姿或仅有单视图深度的数据集上训练

### 损失函数 / 训练策略

- **3D回归损失**：置信度感知的点图回归 $\mathcal{L}_{conf} = \sum (c \cdot \|\hat{x}/\hat{s} - x/s\|_2 - \alpha \log c)$
- **位姿损失**：四元数和平移的L2损失
- **颜色损失**：raymap查询时的MSE重建损失
- **课程训练**：4阶段——(1)4视图静态场景224×224，(2)加入动态场景和部分标注，(3)高分辨率（最大边512），(4)冻结编码器在4-64视图长序列上训练
- 32个训练数据集覆盖：合成/真实、静态/动态、室内/室外

## 实验关键数据

### 主实验（单帧深度）

| 数据集 | 方法 | Abs Rel ↓ | δ<1.25 ↑ |
|--------|------|-----------|----------|
| Bonn | DUSt3R | 0.141 | 82.5 |
| Bonn | MonST3R | 0.076 | 93.9 |
| Bonn | **CUT3R** | **0.063** | **96.2** |
| NYU-v2 | DUSt3R | 0.080 | 90.7 |
| NYU-v2 | **CUT3R** | **0.086** | **90.9** |
| KITTI | MASt3R | 0.079 | 94.7 |
| KITTI | **CUT3R** | 0.092 | 91.3 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无状态（仅图像对） | 多视图不一致 | 退化为DUSt3R式成对预测 |
| 无raymap查询 | 无法推断未见区域 | 缺失场景先验推断能力 |
| 短序列训练（无第4阶段） | 长序列性能下降 | 长上下文推理不足 |

### 关键发现

- 在线处理速度16.58 FPS（A100, 512×144），显著快于需要全局对齐的方法
- 在Bonn和NYU-v2上单帧深度估计达到SOTA
- 视频深度估计在多个数据集上与需要全局对齐的方法竞争甚至超越
- 能无缝处理静态和动态场景（包括运动的人/物体）
- 支持度量尺度预测（不同于DUSt3R的相对尺度）

## 亮点与洞察

- 将3D重建建模为"持续感知"问题，固定大小的状态token是优雅的设计
- 虚拟视角查询类比MAE的图像级补全，概念上简洁且有效
- 灵活性极强：统一处理视频流、无序照片集、静态/动态场景
- 冗余输出设计看似浪费实则精妙，使训练数据利用率最大化
- 从DUSt3R的成对→全局对齐范式升级为在线循环范式，是3D重建领域的重要推进

## 局限与展望

- 固定大小状态token可能限制对超大场景的表达能力
- 训练需要8×A100，计算成本高
- 动态场景的重建质量仍不如专门的动态SLAM方法
- 虚拟视角查询的质量依赖于状态中积累的信息量
- 可探索自适应状态大小或层次化状态表示

## 相关工作与启发

- 直接继承DUSt3R/MASt3R的点图预测范式，但扩展为在线循环架构
- 与Spann3R同期但更进一步：状态不仅缓存观测还推断未见区域
- 与MonST3R互补：前者在线处理任意长度序列，后者扩展到动态场景
- 启发：3D感知可建模为"阅读"般的持续过程，不断更新心理模型

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 持续状态+虚拟查询的统一框架在3D重建领域极具创新性
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖深度、位姿、重建等多任务，多数据集全面评测
- 写作质量: ⭐⭐⭐⭐⭐ 叙事流畅，类比清晰（人类视觉、MAE），图表精美
- 价值: ⭐⭐⭐⭐⭐ 统一框架解决多种3D任务，在线特性有广泛应用前景

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] PartRM: Modeling Part-Level Dynamics with Large Cross-State Reconstruction Model](partrm_modeling_part-level_dynamics_with_large_cross-state_reconstruction_model.md)
- [\[CVPR 2025\] Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes](mesh_mamba_a_unified_state_space_model_for_saliency_prediction_in_non-textured_a.md)
- [\[CVPR 2025\] SphereUFormer: A U-Shaped Transformer for Spherical 360 Perception](sphereuformer_a_u-shaped_transformer_for_spherical_360_perception.md)
- [\[CVPR 2025\] EventFly: Event Camera Perception from Ground to the Sky](eventfly_event_camera_perception_from_ground_to_the_sky.md)
- [\[CVPR 2025\] DSPNet: Dual-vision Scene Perception for Robust 3D Question Answering](dspnet_dual-vision_scene_perception_for_robust_3d_question_answering.md)

<!-- RELATED:END -->
