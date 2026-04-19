---
title: >-
  [论文解读] Any6D: Model-free 6D Pose Estimation of Novel Objects
description: >-
  [CVPR 2025][目标检测][位姿估计] 提出 Any6D 框架，仅从单张 RGB-D 锚点图像即可估计未知物体的 6D 位姿和尺寸，通过 InstantMesh 3D 重建 + 朝向包围盒粗对齐 + 联合尺寸-位姿精细化，在 HO3D 上 ADD-S 达 98.7% 远超 GEDI 的 71.9%。
tags:
  - CVPR 2025
  - 目标检测
  - 位姿估计
  - model-free
  - single anchor
  - InstantMesh
  - FoundationPose
  - render-and-compare
---

# Any6D: Model-free 6D Pose Estimation of Novel Objects

**会议**: CVPR 2025  
**arXiv**: [2503.18673](https://arxiv.org/abs/2503.18673)  
**代码**: [项目页面](https://taeyeop.com/any6d)  
**领域**: 目标检测 / 6D位姿估计  
**关键词**: 6D pose estimation, model-free, single anchor, InstantMesh, FoundationPose, render-and-compare

## 一句话总结
提出 Any6D 框架，仅从单张 RGB-D 锚点图像即可估计未知物体的 6D 位姿和尺寸，通过 InstantMesh 3D 重建 + 朝向包围盒粗对齐 + 联合尺寸-位姿精细化，在 HO3D 上 ADD-S 达 98.7% 远超 GEDI 的 71.9%。

## 研究背景与动机

**领域现状**：6D 物体位姿估计在机器人操作和增强现实中至关重要。现有方法可分为实例级（需精确 CAD 模型）、类别级（需类别先验）和类别无关方法。

**现有痛点**：
   - 基于 CAD 模型的方法需精确纹理化 3D 模型，获取成本高
   - 多视角方法（Gen6D、OnePose、FoundationPose model-free 模式）需多张参考图或视频序列
   - 单视角匹配方法（Oryon、LoFTR）在遮挡或非重叠视角时性能急剧下降

**核心矛盾**：实际机器人场景中，机器人在新环境遇到未知物体时，无法获取 CAD 模型或多视角图像，现有方法均无法有效处理。

**切入角度**：利用图像到3D生成模型（InstantMesh）从单张图重建完整 3D 形状，结合深度信息估计度量尺度，实现完整的 full-to-partial 匹配。

**核心 idea**：单张 RGB-D → InstantMesh 重建归一化 3D → 朝向包围盒粗对齐 → FoundationPose 联合尺寸-位姿精细化 → render-and-compare 选最优假设。

## 方法详解

### 整体框架
给定锚点图像 $I_A$（RGB-D）和查询图像 $I_Q$（RGB-D），目标是估计相对位姿 $\mathbf{T}_{A \to Q} \in SE(3)$。
方法分两步：
1. 从锚点图像重建归一化形状 $O_N$，通过 Object Alignment 估计度量尺度形状 $O_M$ 和锚点位姿 $T_{O_M \to A}$
2. 用度量尺度形状和查询图像估计 $T_{O_M \to Q}$，最终 $\mathbf{T}_{A \to Q} = (T_{O_M \to A})^{-1} \cdot T_{O_M \to Q}$

### 关键设计

1. **3D 形状重建（InstantMesh）**

    - 功能：从锚点图像的 RGB 生成归一化 3D 网格 $O_N$（范围 [-1,1]）
    - 核心限制：生成的形状没有度量尺度，无法直接用于位姿估计
    - 优势：相比 NeRF 或部分视角重建，能生成完整形状，支持 full-to-partial 匹配

2. **粗对齐（Coarse Object Alignment）**

    - 功能：估计初始物体尺寸 $s \in \mathbb{R}^3$ 和粗略位姿
    - 核心思路：使用**朝向包围盒（Oriented Bounding Box）**确定物体中心
    - 为什么不用其他中心估计：
        - 点云均值：部分可见时不可靠
        - 轴对齐包围盒：部分遮挡下中心偏移
    - 操作流程：采样不同旋转角度，计算 $I_A$ 和 $O_N$ 的包围盒 IoU，选 IoU 最高的旋转+缩放组合

3. **精对齐（Fine Object Alignment）**

    - 功能：联合精化尺寸和位姿
    - 基于 FoundationPose 的扩展：
        - 原 FoundationPose 仅在 $SO(3)$ 中采样位姿假设
        - Any6D 额外采样尺寸 $\Delta s \in [0.6, 1.4]$
    - 三模块交替迭代：位姿估计 → 尺寸估计 → 轴对齐
    - Render-and-Compare 选择最优：位姿排序网络 + 自注意力全局评分

4. **位姿选择（Pose Selection）**

    - 两级策略：先用位姿排序网络比较渲染图与裁剪观测，再用自注意力融合所有假设嵌入，输出最终分数

### 训练策略
- 无需额外训练：利用预训练的 InstantMesh 和 FoundationPose
- 在线推理时进行优化式对齐

## 实验关键数据

### 主实验（HO3D 数据集）
| 方法 | 输入模态 | ADD-S↑ | ADD↑ | AR↑ |
|------|----------|--------|------|------|
| Oryon | RGB-D+Language | 23.0 | 0.0 | 1.0 |
| LoFTR | RGB-D | 29.5 | 2.3 | 3.2 |
| GEDI | Depth | 71.9 | 9.7 | 7.4 |
| **Any6D (Ours)** | **RGB-D** | **98.7** | **40.4** | **38.3** |

### 其他数据集
| 数据集 | ADD-S↑ | ADD↑ | AR↑ |
|--------|--------|------|------|
| YCBINEOAT | 89.3 | 45.6 | 37.5 |
| Toyota-Light (ADD(-S)) | 32.2 | AR: 43.3 | MSSD: 55.8 |
| REAL275 (ADD(-S)) | 53.5 | AR: 51.0 | MSPD: 65.3 |
| LM-O (vs GigaPose) | AR: 28.6 | MSPD: 36.1 | VSD: 17.6 |

### 消融实验（HO3D 数据集）
| 配置 | ADD-S↑ | ADD↑ | AR↑ | CD↓ |
|------|--------|------|------|------|
| Baseline (NeRF 部分视角) | 28.6 | 0.0 | 0.2 | 1.02 |
| (1) 无任何对齐 | 0.0 | 0.0 | 0.0 | 1.47 |
| (2) 无粗尺寸，有精化+轴对齐 | 98.0 | 25.5 | 26.8 | 0.53 |
| (3) 有粗尺寸，无精化 | 83.7 | 26.6 | 22.5 | 0.92 |
| (4) 有粗尺寸+精化，无轴对齐 | 92.3 | 23.6 | 24.9 | 0.66 |
| **Full (Ours)** | **98.7** | **40.4** | **38.3** | **0.49** |

### 关键发现
- 粗尺寸估计是基础，缺失则完全失败（配置1）
- 轴对齐对 ADD 和 AR 提升显著（+14.9 AR）
- 尺寸精化避免 XYZ 比例畸变

## 亮点与洞察
- **单张 RGB-D 即可**：无需 CAD 模型、多视角图像或视频序列
- **朝向包围盒**中心估计简单有效，解决部分可见性问题
- **Full-to-partial 匹配**：完整重建消除了部分匹配的歧义
- 在手部遮挡（HO3D）和机器人抓取（YCBINEOAT）场景下均显著领先

## 局限与展望
- 依赖 InstantMesh 的重建质量，初始 3D 形状不准确时性能下降
- 当前不包含形状更新/优化步骤
- 推理速度受 InstantMesh 限制

## 评分
- 新颖性: ⭐⭐⭐⭐ InstantMesh+FoundationPose 联合估计尺寸位姿
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集+详细消融
- 写作质量: ⭐⭐⭐⭐ 动机清晰框架完整
- 价值: ⭐⭐⭐⭐⭐ 对机器人操作有重大实用价值

<!-- RELATED:START -->

## 相关论文

- [VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions](../../ICCV2025/object_detection/voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)
- [Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](../../CVPR2026/object_detection/show_dont_tell_detecting_novel_objects_by_watching.md)
- [Robust Long-term Test-Time Adaptation for 3D Human Pose Estimation through Motion Discretization](../../AAAI2026/object_detection/robust_long-term_test-time_adaptation_for_3d_human_pose_estimation_through_motio.md)
- [MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)
- [PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space](../../NeurIPS2025/object_detection/pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio.md)

<!-- RELATED:END -->
