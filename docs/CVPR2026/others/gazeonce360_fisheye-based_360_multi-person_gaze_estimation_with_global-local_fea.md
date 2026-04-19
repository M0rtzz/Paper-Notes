---
title: >-
  [论文解读] GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion
description: >-
  [CVPR 2026][视线估计] 本文提出 GazeOnce360，一个端到端的双分辨率 CNN 模型，用于从单个朝上放置的桌面鱼眼相机进行 360° 多人视线方向估计，同时构建了首个面向该场景的大规模合成数据集 MPSGaze360，在精度和速度两方面均大幅超越现有多阶段方法 GAM360。
tags:
  - CVPR 2026
  - 视线估计
  - 鱼眼相机
  - 多人场景
  - 双分辨率融合
  - 合成数据
---

# GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion

**会议**: CVPR 2026  
**arXiv**: [2603.17161](https://arxiv.org/abs/2603.17161)  
**代码**: https://caizhuojiang.github.io/GazeOnce360/ (Project Page)  
**领域**: 其他  
**关键词**: 视线估计, 鱼眼相机, 多人场景, 双分辨率融合, 合成数据

## 一句话总结
本文提出 GazeOnce360，一个端到端的双分辨率 CNN 模型，用于从单个朝上放置的桌面鱼眼相机进行 360° 多人视线方向估计，同时构建了首个面向该场景的大规模合成数据集 MPSGaze360，在精度和速度两方面均大幅超越现有多阶段方法 GAM360。

## 研究背景与动机
视线估计在人机交互、协作分析、VR 等领域有广泛应用。现有单人视线估计方法已较为成熟（MPIIGaze、ETH-XGaze 等数据集驱动），但真实场景往往涉及多人。

现有多人视线估计的痛点：(1) **前向相机视场角有限**，需要多台同步设备才能覆盖全部方向；(2) 已有尝试（如 GAM360）使用鱼眼相机，但采用**多阶段流水线**（检测人脸 → 透视投影 → 单独估计），计算昂贵、容易累积误差，且全景拼接可能在边界处裂开人脸导致检测遗漏。

本文的切入点是：朝上放置的鱼眼相机天然覆盖 360° 全景，一台设备即可捕获所有方向的人。但鱼眼图像存在**严重的几何畸变**和**透视变化**，且缺乏公开的多人朝上鱼眼视线数据集。于是作者从数据（合成数据集）和模型（端到端双分辨率架构）两方面同时解决这些问题。

## 方法详解

### 整体框架
GazeOnce360 是一个 anchor-based 的检测+回归网络。输入鱼眼图像经全局低分辨率分支提取上下文并检测人脸框，对每个检测到的人脸区域用高分辨率局部分支提取精细眼部特征，然后通过交叉注意力融合两路特征，最终通过多任务头预测人脸框、头部姿态、视线方向、人脸/眼部关键点等。训练时用 GT 框裁剪，测试时用预测框。

### 关键设计

1. **旋转卷积（Rotational Convolution）**:

    - 功能：替代标准卷积来处理鱼眼图像的旋转畸变
    - 核心思路：将卷积核复制为四个正交旋转版本，对各方向的特征响应进行加权平均，增强网络的旋转不变性
    - 设计动机：鱼眼相机从下往上拍摄，不同方向的人脸呈现大幅度旋转。标准 CNN 只具有平移不变性，无法处理这种旋转。旋转卷积集成在 FPN 的顶层，让高层特征具备旋转适应能力。实验表明它比可变形卷积（DCN）更适合鱼眼畸变（10.39° vs 11.05°）

2. **双分辨率特征融合（Dual-Resolution Feature Fusion）**:

    - 功能：并行处理低分辨率全局图和高分辨率人脸裁剪，用交叉注意力融合
    - 核心思路：全局分支在 512×512 低分辨率下提取空间布局信息，局部分支在 1024×1024 下对裁剪的人脸区域提取精细眼部特征。融合采用交叉注意力：$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\frac{\mathbf{QK}^T}{\sqrt{d_k}})\mathbf{V}$，其中全局特征为 Query、局部人脸特征为 Key/Value，并通过空间掩码限制注意力在对应人脸区域
    - 设计动机：视线估计需要极其精细的眼部特征（虹膜、瞳孔位置），但全分辨率处理整幅鱼眼图效率低下（大部分背景像素无用）。双分辨率方案在保持精度的同时提速 22%

3. **多任务监督与合成数据集 MPSGaze360**:

    - 功能：利用 Unreal Engine 5 + MetaHuman 生成大规模合成训练数据，并用人脸、眼部关键点作为辅助监督
    - 核心思路：数据集包含 23,496 张鱼眼图像，每帧 1-7 人，69 种不同人物模型。五个正交透视视角渲染后投影为等距鱼眼图像。标注包括 3D 视线向量、2D 人脸/眼部关键点、人脸框、3D 头部姿态等
    - 设计动机：真实世界中从下向上的鱼眼视角几乎不可能获取精确的多人视线标注（尤其是瞳孔中心等眼部关键点）。合成数据可提供像素级精确标注，且已有研究（GazeGene）证明合成 → 真实的泛化性

### 损失函数 / 训练策略
多任务联合损失：$\mathcal{L} = \lambda_1\mathcal{L}_c + \lambda_2\mathcal{L}_b + \lambda_3\mathcal{L}_d + \lambda_4\mathcal{L}_h + \lambda_5\mathcal{L}_g + \lambda_6\mathcal{L}_{fl} + \lambda_7\mathcal{L}_{el}$，其中 $\mathcal{L}_c$ 为平衡交叉熵分类损失，其余均为 Smooth L1 损失。训练 150 epochs，Adam 优化器，初始 lr=$10^{-3}$，在 30 和 100 epoch 衰减。

## 实验关键数据

### 主实验
| 方法 | 视线误差(°) ↓ | 调整后视线误差(°) ↓ | FPS ↑ |
|------|-------------|-------------------|-------|
| GAM360 (多阶段) | 18.96 | 18.76 | 4.23 |
| **GazeOnce360** | **10.39** | **9.99** | **16.23** |
| 提升 | -8.57 | -8.77 | +12.00 |

### 消融实验
| 配置 | 精度↑ | 召回↑ | 视线误差(°)↓ | FPS↑ | 说明 |
|------|-------|-------|-------------|------|------|
| 基线(无RotConv,无关键点) | 0.984 | 0.993 | 12.14 | — | 基线 |
| +RotConv | 0.992 | 0.993 | 11.14 | — | 旋转不变性降低1° |
| +RotConv+眼部关键点 | 0.994 | 0.994 | **8.89** | — | 眼部监督贡献最大 |
| 单分辨率(512) | 0.996 | 0.992 | 16.50 | 20.49 | 低分辨率精度差 |
| 单分辨率(1024) | 0.998 | 0.993 | 8.945 | 13.30 | 高精度但慢 |
| 双分辨率(512+1024) | 0.999 | 0.993 | 8.968 | **16.23** | 精度≈高分辨率,速度+22% |
| RotConv vs DCN | — | — | 10.39 vs 11.05 | — | RotConv 优于 DCN |

### 关键发现
- 眼部关键点监督是提升视线精度的最大贡献因素（12.14° → 8.89°，降低了 26.8%）
- 旋转卷积明确优于可变形卷积，说明鱼眼畸变的核心问题是旋转而非一般的空间变形
- 模型在跨场景+跨身份设置下视线误差仅从 8.945° 升至 10.39°，泛化性良好
- 纯合成数据训练的模型在真实鱼眼图像上也能产生合理的视线预测

## 亮点与洞察
- 问题定义有价值：单台桌面鱼眼相机实现 360° 多人视线估计，在智能会议室、服务机器人等场景有明确应用前景
- 端到端方案 vs 多阶段流水线：精度提升近 2 倍，速度提升近 4 倍
- 合成数据生成管线设计完善，从 UE5 MetaHuman 到鱼眼投影模型，可复现可扩展

## 局限与展望
- 目前仅在合成数据上训练和评估，真实场景的定量结果缺失（只有定性可视化）
- MPSGaze360 规模相对有限（23K 图像，69 种人物），多样性可能不足以覆盖真实场景的复杂性
- 鱼眼投影只采用了等距模型，实际部署中不同镜头的畸变模型差异较大
- 远距离人物的眼部分辨率仍然很低，高分辨率裁剪的效果可能打折扣

## 相关工作与启发
- GazeOnce（前置相机多人视线估计）的鱼眼扩展版，继承了 anchor-based 多任务设计
- 旋转卷积在鱼眼感知中的成功可迁移到其他鱼眼任务（如鱼眼检测、分割）
- Sim-to-Real 策略的可行性得到了验证，未来可通过域适应或域随机化进一步提升泛化

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题定义新颖，但各技术组件（旋转卷积、双分辨率、合成数据）均非首创
- 实验充分度: ⭐⭐⭐ 仅合成数据评估，缺少真实数据定量对比和更多基线方法
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示丰富，数据集生成流程讲述详尽
- 价值: ⭐⭐⭐⭐ 鱼眼视线估计的首个端到端方案，有明确的实用场景

<!-- RELATED:START -->

## 相关论文

- [Gaze Target Detection Based on Head-Local-Global Coordination](../../ECCV2024/others/gaze_target_detection_based_on_head-local-global_coordination.md)
- [De-confounded Gaze Estimation](../../ECCV2024/others/de-confounded_gaze_estimation.md)
- [Local Guidance for Configuration-Based Multi-Agent Pathfinding](../../AAAI2026/others/local_guidance_for_configuration-based_multi-agent_pathfinding.md)
- [AssistMimic: Physics-Grounded Humanoid Assistance via Multi-Agent RL](assistmimic_physics_grounded_humanoid_assistance.md)
- [HPFF: Hierarchical Locally Supervised Learning with Patch Feature Fusion](../../ECCV2024/others/hpff_hierarchical_locally_supervised_learning_with_patch_feature_fusion.md)

<!-- RELATED:END -->
