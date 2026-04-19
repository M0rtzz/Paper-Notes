---
title: >-
  [论文解读] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision
description: >-
  [ICCV 2025][3D视觉][自我中心视觉] 提出 Fish2Mesh，一种鱼眼感知的 Transformer 模型，通过新颖的自我中心位置编码（EPE）将等距柱状投影的3D球面信息嵌入 Swin Transformer，实现从头戴式鱼眼相机图像精确恢复3D人体网格。
tags:
  - ICCV 2025
  - 3D视觉
  - 自我中心视觉
  - 鱼眼相机
  - 人体网格恢复
  - Transformer
  - 位置编码
  - SMPL
---

# Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision

**会议**: ICCV 2025  
**arXiv**: [2503.06089](https://arxiv.org/abs/2503.06089)  
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: 自我中心视觉, 鱼眼相机, 人体网格恢复, Swin Transformer, 位置编码, SMPL

## 一句话总结

提出 Fish2Mesh，一种鱼眼感知的 Transformer 模型，通过新颖的自我中心位置编码（EPE）将等距柱状投影的3D球面信息嵌入 Swin Transformer，实现从头戴式鱼眼相机图像精确恢复3D人体网格。

## 研究背景与动机

自我中心人体估计（Egocentric HMR）面临三大独特挑战：

**数据稀缺**：头戴式鱼眼相机的标注数据极少，难以大规模获取

**鱼眼畸变**：鱼眼镜头在图像边缘引入严重的非线性畸变，传统位置编码无法处理

**自遮挡**：头戴视角下手臂经常遮挡躯干，腿部和脚部可能完全不在视野内

此前 SOTA 方法 EgoHMR 采用扩散模型进行3D网格恢复，但其变分特性不适用于实时XR系统等高精度需求场景。更关键的是，EgoHMR **完全忽略了鱼眼图像畸变的处理**，这会降低网格恢复质量。

Fish2Mesh 的核心思想：**不将鱼眼畸变视为需要校正的异常，而是通过参数化位置编码直接编码等距柱状投影的3D球面信息**，让模型原生理解鱼眼几何。

## 方法详解

### 整体框架

Fish2Mesh 架构由三部分组成：
1. **自我中心位置编码（EPE）**：生成编码鱼眼球面几何的位置嵌入表
2. **Swin Transformer blocks**：层次化特征提取
3. **多任务头**：SMPL参数回归、相机变换预测、全局朝向估计

### 关键设计

**等距柱状投影与3D球面坐标恢复**：首先将180度鱼眼图像投影到2D平面，然后反投影恢复3D空间坐标（x_3D, y_3D, z_3D），将离散化的3D坐标作为可学习位置嵌入表的索引，注入 Swin Transformer。这样每个像素都带有其对应的3D球面坐标信息。

**移除相对位置偏置**：由于 EPE 已经包含了完整的位置信息，因此移除 Swin Transformer 中的相对位置偏置参数，避免冗余。

**弱监督数据增强**：使用预训练 4D-Human 模型和第三人称相机视角提供弱监督标签。设计基于提示的数据采集系统，引导参与者执行自然动作（如"搅大锅"），产生包含真实头部运动和自遮挡的多样化运动，替代传统的脚本式动作采集。

### 损失函数

多任务组合损失包含四个部分：
- SMPL 形状参数和姿态参数的 L2 损失
- 全局朝向 L1 损失
- 辅助3D关节 L1 损失
- 辅助2D投影关节 L1 损失

3D-2D一致性约束确保模型同时优化3D重建和2D投影对齐。

## 实验关键数据

### 主实验表格

**三个数据集上的 PA-MPJPE / PA-MPVPE（mm，越低越好）**：

| 模型 | ECHP PA-MPJPE | ECHP PA-MPVPE | Ego4D PA-MPJPE | Ego4D PA-MPVPE | Our PA-MPJPE | Our PA-MPVPE |
|------|--------------|--------------|----------------|----------------|-------------|-------------|
| 4D-Human | 90.04 | 129.85 | 120.31 | 132.83 | 98.61 | 120.30 |
| FisheyeViT | 94.18 | - | 91.98 | - | 93.55 | - |
| EgoHMR | 64.11 | 79.03 | 114.42 | 129.00 | 127.55 | 144.18 |
| **Fish2Mesh** | **57.67** | **75.32** | **41.93** | **54.76** | **37.24** | **51.58** |

在 Ego4D 数据集上，Fish2Mesh 的 PA-MPJPE（41.93mm）比 EgoHMR（114.42mm）降低了 **63.4%**。

### 消融表格

| 消融设置 | ECHP PA-MPJPE | Ego4D PA-MPJPE | Our PA-MPJPE |
|---------|--------------|----------------|-------------|
| 去除 EPE | 92.45 | 90.11 | 87.73 |
| 去除自定义数据集 | 71.45 | 65.56 | 60.45 |
| **完整 Fish2Mesh** | **57.67** | **41.93** | **37.24** |

EPE 的影响最为显著：移除后 ECHP 上 PA-MPJPE 从 57.67mm 增至 92.45mm（+60.3%），证明鱼眼感知的位置编码对于正确解读鱼眼像素信息至关重要。

### 关键发现

1. Fish2Mesh 在所有三个数据集上全面超越 SOTA（EgoHMR），尤其在 Ego4D 上提升巨大
2. EPE 是最关键的组件，移除后性能大幅下降（ECHP: 57.67 -> 92.45 mm）
3. 提示式数据扩展也有显著贡献（ECHP: 57.67 -> 71.45 mm）
4. 定性结果显示 Fish2Mesh 对下半身（坐姿、蹲姿）的估计明显优于其他方法

## 亮点与洞察

- **将鱼眼畸变从缺陷转化为特征**：不去校正畸变而是直接编码球面几何信息，这一范式转变思路新颖
- 等距柱状投影到3D球面坐标到可学习位置嵌入的链条设计简洁有效
- 提示式数据采集（"搅大锅"而非"原地走"）显著提高数据多样性和模型鲁棒性，成本低效果好
- 使用 Procrustes 对齐指标（PA-MPJPE）作为主评估指标避免了相机坐标系差异的影响

## 局限性

- 模型从头训练，未利用大规模预训练权重，泛化能力可能受限
- 仅评估了头戴下视鱼眼场景，对其他鱼眼配置（侧向、胸前安装）未有验证
- 弱监督依赖 4D-Human 的预测质量作为伪标签，可能引入系统偏差
- 未与更新的 HMR 2.0 等方法比较
- 推理速度和模型参数量未报告

## 相关工作与启发

- **EgoHMR** 是最直接的对比方法，Fish2Mesh 通过确定性回归替代扩散采样，更适合实时应用
- **FisheyeViT** 的 patch 策略计算开销大，EPE 提供了更高效的鱼眼处理方案
- 等距柱状投影的位置编码思路可推广到全景图像理解、自动驾驶环视相机等场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ — EPE 设计新颖，将球面几何编入 Transformer 的思路有启发性
- **实验充分度**: ⭐⭐⭐ — 三个数据集+消融，但缺少推理速度对比和更多 baseline
- **写作质量**: ⭐⭐⭐ — 整体清晰但数学符号较多，可读性一般
- **价值**: ⭐⭐⭐⭐ — 对自我中心人体估计领域有实际贡献，EPE 概念可广泛适用

## 亮点与洞察

## 局限与展望

## 相关工作与启发

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)
- [HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery](../../CVPR2025/3d_vision/heatformer_a_neural_optimizer_for_multiview_human_mesh_recovery.md)
- [MEGA: Masked Generative Autoencoder for Human Mesh Recovery](../../CVPR2025/3d_vision/mega_masked_generative_autoencoder_for_human_mesh_recovery.md)
- [Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimat.md)
- [Global-to-Pixel Regression for Human Mesh Recovery](../../ECCV2024/3d_vision/global-to-pixel_regression_for_human_mesh_recovery.md)

<!-- RELATED:END -->
