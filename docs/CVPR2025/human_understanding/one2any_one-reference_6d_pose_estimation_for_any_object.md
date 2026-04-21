---
title: >-
  [论文解读] One2Any: One-Reference 6D Pose Estimation for Any Object
description: >-
  [CVPR 2025][人体理解][单参考] 提出 One2Any，仅需单张参考图像即可估计任意新物体的 6D 位姿——用参考物体坐标（ROC，以参考相机帧为基准而非规范坐标）编码参考姿态，通过 VQVAE+U-Net 条件生成密集 ROC 图，再用 Umeyama 算法恢复位姿，在 YCB-Video 上 93.7% ADD-S AUC，推理仅 0.09 秒。
tags:
  - CVPR 2025
  - 人体理解
  - 单参考
  - 6D位姿
  - 条件生成
  - 参考物体坐标
  - VQVAE
---

# One2Any: One-Reference 6D Pose Estimation for Any Object

**会议**: CVPR 2025  
**arXiv**: [2505.04109](https://arxiv.org/abs/2505.04109)  
**代码**: https://github.com/lmy1001/One2Any (有)  
**领域**: 人体理解 / 6D 位姿估计  
**关键词**: 单参考, 6D位姿, 条件生成, 参考物体坐标, VQVAE

## 一句话总结

提出 One2Any，仅需单张参考图像即可估计任意新物体的 6D 位姿——用参考物体坐标（ROC，以参考相机帧为基准而非规范坐标）编码参考姿态，通过 VQVAE+U-Net 条件生成密集 ROC 图，再用 Umeyama 算法恢复位姿，在 YCB-Video 上 93.7% ADD-S AUC，推理仅 0.09 秒。

## 研究背景与动机

**领域现状**：6D 物体位姿估计是机器人抓取和 AR 的关键。传统方法需要精确 CAD 模型或多视角参考图。近年的"无 CAD/少参考"方法（如 FoundationPose、Oryon）仍需要多视角几何或昂贵的在线推理。

**现有痛点**：（1）FoundationPose 推理 1 秒/帧（11× 慢于 One2Any）；（2）Oryon 需要视频序列而非单张参考；（3）NOCS（归一化物体坐标空间）要求物体有规范坐标定义——对新颖物体不可行。

**核心矛盾**：单张参考图信息极其有限（只有一个视角），但 6D 位姿需要理解物体的完整几何。

**切入角度**：放弃 NOCS 的规范坐标假设，改用参考相机帧作为坐标系——ROC 只需要参考图像本身就能定义，不需要任何关于物体几何的先验知识。

**核心 idea**：参考物体坐标（ROC）替代 NOCS + 条件生成密集坐标图 = 单参考任意物体 6D 位姿。

## 方法详解

### 关键设计

1. **参考物体坐标（ROC）**:

    - 功能：以参考相机帧为坐标系定义物体表面坐标
    - 核心思路：从参考图像的深度图和掩码生成物体的 3D 点云，直接用参考相机坐标系下的坐标作为 ROC。无需物体 CAD 模型或规范坐标
    - 设计动机：消融显示 ROC 比直接预测旋转/平移高 6.5%（91.2% vs 84.7% ADD-S）

2. **ROPE 编码器 + OPD 解码器**:

    - 功能：从参考图生成查询图的密集 ROC 图
    - 核心思路：ROPE 编码器将参考图的 RGB+ROC+掩码编码为物体表示。OPD 解码器基于预训练 VQVAE + U-Net，以 ROPE 特征为条件通过交叉注意力生成查询图的 ROC 图
    - 设计动机：条件生成比特征匹配更适合处理大视角差——可以"想象"未见过的物体表面

3. **Umeyama 位姿恢复**:

    - 功能：从预测的 ROC 图和查询深度图恢复 6D 位姿
    - 核心思路：将预测的 ROC 3D 点与查询图的实际 3D 点做 Umeyama 对齐
    - 设计动机：经典几何方法，鲁棒且高效

### 损失函数 / 训练策略

Smooth L1 损失：$\mathcal{L} = \frac{1}{N}\sum_{i,j} Q_M(i,j) E(i,j)$，$\beta=0.1$。推理 0.09 秒/帧。

## 实验关键数据

### 主实验

| 数据集 | One2Any | Oryon | FoundationPose |
|--------|---------|-------|---------------|
| YCB ADD-S AUC | **93.7%** | 13.3% | 92.7% |
| Real275 AR | **54.9%** | 46.5% | - |
| 推理时间 | **0.09s** | 0.95s | 1.0s |

### 消融实验

| 配置 | ADD-S AUC |
|------|-----------|
| 直接预测旋转/平移 | 84.7% |
| **ROC 表示** | **91.2%** |
| RGB+Depth 输入 | 90.0% |
| **RGB+ROC+Mask 输入** | **91.2%** |

### 关键发现
- **ROC 替代 NOCS 是关键**：无需规范坐标，参考帧即可定义
- **极快推理**：0.09 秒，适合实时机器人应用
- **Oryon 在单参考下崩溃**：13.3% vs 93.7%——专为多视角设计的方法迁移失败

## 亮点与洞察
- **ROC 的简洁优雅**——不需要任何物体先验知识，参考图自身就是坐标系定义
- **生成式 vs 判别式**——用条件生成而非特征匹配处理大视角差，更鲁棒

## 局限与展望
- 需要 GT 深度和掩码
- 无纹理物体表现差（LINEMOD ape 仅 33.1%）
- 参考视角质量影响性能

## 评分
- 新颖性: ⭐⭐⭐⭐ ROC 替代 NOCS 的概念简洁有力
- 实验充分度: ⭐⭐⭐⭐⭐ Real275/YCB/LINEMOD/Toyota 多数据集
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 实用的单参考 6D 位姿方案

<!-- RELATED:START -->

## 相关论文

- [RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](../../ICCV2025/human_understanding/raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [CoordAR: One-Reference 6D Pose Estimation of Novel Objects via Autoregressive Coordinate Map Generation](../../AAAI2026/human_understanding/coordar_one-reference_6d_pose_estimation_of_novel_objects_via_autoregressive_coo.md)
- [GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation](unihope_a_unified_approach_for_hand-only_and_hand-object_pose_estimation.md)
- [CRISP: Object Pose and Shape Estimation with Test-Time Adaptation](crisp_object_pose_and_shape_estimation_with_test-time_adaptation.md)

<!-- RELATED:END -->
