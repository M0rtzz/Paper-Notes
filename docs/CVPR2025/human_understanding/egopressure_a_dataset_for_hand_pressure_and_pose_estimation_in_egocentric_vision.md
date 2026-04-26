---
title: >-
  [论文解读] EgoPressure: A Dataset for Hand Pressure and Pose Estimation in Egocentric Vision
description: >-
  [CVPR 2025][人体理解][手部姿态估计] EgoPressure 提出首个第一人称视角的手部触觉压力和姿态数据集，包含 21 名参与者 5 小时的 RGB-D 交互数据、基于多视角优化的高保真 MANO 手部网格标注和压力传感器的真实压力映射，并建立了从 RGB 图像估计手部压力和姿态的基准模型。
tags:
  - CVPR 2025
  - 人体理解
  - 手部姿态估计
  - 压力感知
  - 第一人称视角
  - 多视角标注
  - 手-物交互
---

# EgoPressure: A Dataset for Hand Pressure and Pose Estimation in Egocentric Vision

**会议**: CVPR 2025  
**arXiv**: [2409.02224](https://arxiv.org/abs/2409.02224)  
**代码**: https://yiming-zhao.github.io/EgoPressure/  
**领域**: 人体理解  
**关键词**: 手部姿态估计, 压力感知, 第一人称视角, 多视角标注, 手-物交互

## 一句话总结

EgoPressure 提出首个第一人称视角的手部触觉压力和姿态数据集，包含 21 名参与者 5 小时的 RGB-D 交互数据、基于多视角优化的高保真 MANO 手部网格标注和压力传感器的真实压力映射，并建立了从 RGB 图像估计手部压力和姿态的基准模型。

## 研究背景与动机

**领域现状**：理解人手与物体交互中的触觉接触和压力信息对 AR/VR 和机器人操作至关重要。现有压力估计方法主要依赖传感手套或机器人触觉传感器来获取压力标签，但这些设备会干扰自然触觉反馈。基于视觉的方法无需手部穿戴设备，更易部署在智能眼镜等设备上。PressureVision 数据集是少数提供真实压力标注的视觉数据集，但仅限于静态外部相机视角。

**现有痛点**：现有数据集存在三个关键缺陷。第一，缺乏第一人称视角——智能眼镜和 VR 头显的相机都是第一人称的，但现有压力数据集仅包含静态外部相机视角，模型无法泛化到第一人称场景。第二，缺乏精确的手部姿态标注——大多数手-物交互数据集的接触信息都是从粗糙的手部姿态和物体网格的交叉推断的，不够准确。第三，压力信息缺乏3D定位——现有方法预测的是 2D 图像平面上的压力投影，无法确定压力在手部 3D 表面的具体位置。

**核心矛盾**：视觉压力估计需要大量带压力标签的训练数据，但精确的压力标注需要昂贵的传感设备；同时第一人称视角下手部自遮挡严重，单视角的手部姿态估计精度不足以支持精确的压力-手部映射。

**本文目标** (1) 构建首个包含第一人称视角、真实压力标签和精确手部姿态/网格标注的数据集；(2) 开发多视角手部姿态标注方法，不需要任何手部标记(marker)；(3) 建立手部压力和姿态联合估计的基准。

**切入角度**：作者设计了一个包含 8 台 Azure Kinect RGB-D 相机（7 台静态 + 1 台头戴）和 Sensel Morph 压力触摸板的采集平台。通过多视角可微渲染优化获得精确的 MANO 手部网格，然后通过虚拟正交相机将压力逆投影到手部网格的 UV 纹理图上。

**核心 idea**：构建首个第一人称压力数据集，通过多视角可微渲染优化获得高保真手部网格标注，并将压力映射为手部 UV 纹理，实现 3D 空间中的压力定位。

## 方法详解

### 整体框架

EgoPressure 的构建流程包括三个阶段：(1) 数据采集——21 名参与者在 8 相机+压力传感器平台上执行 31 种手势；(2) 手部姿态标注——使用 HaMeR 初始化 + 多视角可微渲染优化获得精确的 MANO 手部网格；(3) 压力映射——通过虚拟正交相机将触摸板捕获的压力数据逆投影到手部网格的 UV 纹理图上。基准评估建立了三类模型：仅 RGB、RGB+手部姿态、以及联合估计手部网格+压力的 PressureFormer。

### 关键设计

1. **多视角无标记手部姿态标注 (Multi-view Marker-less Annotation)**:

    - 功能：从 7 个静态相机视角获得亚毫米级精度的 MANO 手部网格标注
    - 核心思路：首先用 HaMeR 在每个视角独立估计初始姿态，通过三角化 7 个视角的根关节解决尺度-平移模糊。然后执行两阶段优化：(1) **姿态优化**——固定手部形状参数 $\beta$，优化姿态 $\theta$ 和平移 $t$，目标函数 $\mathcal{L}_{\text{pose}} = \mathcal{L}_{\mathcal{R}} + \mathcal{L}_{\text{insec}}$，其中 $\mathcal{L}_{\mathcal{R}}$ 是跨所有视角的可微渲染损失，$\mathcal{L}_{\text{insec}}$ 防止网格自交叉；(2) **形状精修**——固定姿态，引入顶点位移 $D_{\text{vert}}$ 沿法线方向偏移每个顶点，加入几何正则化和虚拟渲染损失
    - 设计动机：单视角手部姿态估计在自遮挡下精度不足，多视角联合优化可从不同角度互补观测。顶点位移允许模型捕获 MANO 参数模型无法表示的软组织变形

2. **虚拟渲染压力映射 (Virtual Render for Contact and Pressure)**:

    - 功能：将触摸板平面上的 2D 压力数据映射为手部网格 UV 空间中的 3D 压力纹理
    - 核心思路：在触摸板下方放置一个虚拟正交相机，朝上对齐渲染面与触摸板表面。优化目标包含两项：(1) 渲染压力与真实压力的 MSE 损失；(2) 接触区域内手部网格到触摸板表面的深度一致性约束。损失函数为: $\mathcal{L}_{\mathcal{R}^v} = \text{MSE}(\mathcal{R}^v_P(\Theta^*, \mathcal{T}_P), P_{\text{gt}}) + |\mathbb{I}(P_{\text{gt}}>0) \odot (\mathcal{R}^v_D(\Theta^*)[z] - Z_{v2p})|_1$
    - 设计动机：传统压力数据集只提供平面压力图，无法知道手部哪个部位施加了多大压力。通过虚拟相机将压力"映射"回手部 UV 纹理，可以在 3D 空间中精确定位每个手部区域的压力，为机器人抓取等应用提供更有价值的信息

3. **PressureFormer 基准模型**:

    - 功能：从单张 RGB 图像联合估计手部网格和压力 UV 图
    - 核心思路：基于 HaMeR 架构扩展，利用 HaMeR 的 ViT 提取图像特征 token。将手部顶点 $V_{\text{hand}}$ 作为 Transformer 解码器的输入 token，对 ViT 特征做交叉注意力。每个输出 token 是一个 $D$ 维特征向量，映射到 MANO 的 UV 坐标系得到 UV 压力特征图，经过两层卷积插值后预测量化压力 UV 图 $U_{\text{pred}}$。通过可微渲染器将压力投影回图像平面计算图像空间的压力损失
    - 设计动机：直接在 UV 空间预测压力比在 2D 图像平面预测更有意义——UV 空间与手部表面一一对应，且对相机视角不变

### 损失函数 / 训练策略

PressureFormer 的损失：$\mathcal{L}_{\mathcal{PF}} = w_1 \mathcal{L}_c + w_2 \mathcal{L}_p$，其中 $\mathcal{L}_c$ 是 UV 空间的粗压力分类损失，$\mathcal{L}_p$ 是通过可微渲染投影到图像平面后的压力损失，均使用交叉熵。训练时对手部中心裁剪图像做数据增强（平移、缩放、旋转）。

## 实验关键数据

### 主实验

不同输入模态的压力估计对比（全数据集 21 人）：

| 模型 | 训练视角 | 测试视角 | 输入模态 | Contact IoU↑ | Vol. IoU↑ | MAE(Pa)↓ |
|------|---------|---------|---------|-------------|-----------|----------|
| PressureVisionNet | Ego | Ego | RGB | 55.73 | 38.64 | 53.60 |
| + HaMeR 姿态 | Ego | Ego | RGB+pred pose | 56.25 | 40.52 | 55.23 |
| + GT 姿态 | Ego | Ego | RGB+GT pose | 58.80 | 41.39 | 53.79 |
| PressureVisionNet | Exo(2345) | Exo(2345) | RGB | 62.11 | 44.73 | 43.15 |
| + GT 姿态 | Exo(2345) | Exo(2345) | RGB+GT pose | 64.39 | 47.58 | 41.72 |
| PressureVisionNet | Exo(2345) | Exo(167) | RGB | 36.82 | 25.05 | 62.22 |
| + GT 姿态 | Exo(2345) | Exo(167) | RGB+GT pose | 43.04 | 31.39 | 49.45 |

### 消融实验

| 模型 | Contact IoU↑ | Vol. IoU↑ | MAE(kPa)↓ | 说明 |
|------|-------------|-----------|-----------|------|
| PressureVisionNet | 55.73 | 38.64 | 53.60 | 仅 RGB 基线 |
| + HaMeR pose (2.5D) | 56.25 | 40.52 | 55.23 | 预测手部姿态作为额外通道 |
| + GT pose | 58.80 | 41.39 | 53.79 | GT 手部姿态的上界 |
| PressureFormer | **最优** | 可比 | 可比 | UV 空间压力预测 |

### 关键发现
- 手部姿态信息对压力估计确实有帮助：GT pose 在 Vol. IoU 上提升 5%+（外部视角）和 2.75%（第一人称）
- 手部姿态还显著提升了模型的跨视角泛化能力：未见视角上 RGB+GT pose 的 Contact IoU 从 36.82 提升到 43.04（+6.22）
- 即使使用预测的 HaMeR 姿态（非 GT），也能获得明显提升，说明当前 SOTA 手部估计器的精度已足够收益
- PressureFormer 首次实现了从单张 RGB 联合估计手部网格和 3D 压力分布，在 Contact IoU 上取得最优
- 约 45.1% 的帧有手-垫接触，压力阈值 0.5 kPa 用于过滤扩散噪声读数

## 亮点与洞察
- **多视角+压力传感器的采集平台设计**非常系统——8 台同步 Kinect + Sensel Morph + 红外 PTP 时间同步 + 四种桌面纹理，覆盖了数据质量的各个维度
- **将压力映射为 UV 纹理的思路**开创性地将 2D 传感数据提升到 3D 空间——不再是"图像上哪里有压力"，而是"手的哪个部位施加了多大压力"。这对机器人学习人类抓取非常有价值
- **手部姿态与压力的互补性**是一个重要的实验发现——压力提供了"力度"信息，姿态提供了"位置"信息，两者结合才能完整理解手-物交互。这为未来的多模态交互理解指明了方向

## 局限与展望
- 仅限平面触摸板交互场景，无法覆盖抓取3D物体时的压力分布——需要开发曲面压力传感器
- 21 名参与者的数据多样性有限，手型变化范围较窄（MANO β 分布集中），性别比例不平衡（6女15男）
- 标注方法依赖7台静态相机，采集平台复杂昂贵，难以大规模部署和扩展
- PressureFormer 的压力量化为离散类别，失去了连续压力值的精度；且依赖 HaMeR 的手部估计质量作为上界
- 所有手势都在受控实验室环境下执行，与真实使用场景（厨房、办公桌等）有较大差距

## 相关工作与启发
- **vs PressureVision**: PressureVision 是首个视觉压力数据集，但限于外部静态相机且无手部姿态标注。EgoPressure 增加了第一人称视角和高精度手部网格，是 PressureVision 的自然且重要的扩展
- **vs ContactPose**: ContactPose 通过热成像获得接触图但无压力强度信息，且接触标注基于热残留而非实时测量。EgoPressure 提供实时高分辨率压力读数
- **vs ARCTIC/GRAB**: 这些数据集的接触信息从手-物网格交叉推断，精度受限于姿态估计误差。EgoPressure 使用压力传感器获取真实接触和压力

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个第一人称压力数据集，标注方法和 UV 压力映射有创新
- 实验充分度: ⭐⭐⭐⭐ 多种基线模型 + 跨视角泛化实验 + 联合估计模型，覆盖全面
- 写作质量: ⭐⭐⭐⭐⭐ 数据集描述详尽，采集细节透明，方法可复现性强
- 价值: ⭐⭐⭐⭐⭐ 填补了第一人称压力数据集的空白，对 AR/VR 手部交互和机器人操作研究有重要推动作用

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation](unihope_a_unified_approach_for_hand-only_and_hand-object_pose_estimation.md)
- [\[CVPR 2025\] PoseBH: Prototypical Multi-Dataset Training Beyond Human Pose Estimation](posebh_prototypical_multi-dataset_training_beyond_human_pose_estimation.md)
- [\[CVPR 2025\] Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation](analyzing_the_synthetic-to-real_domain_gap_in_3d_hand_pose_estimation.md)
- [\[ECCV 2024\] 3D Hand Pose Estimation in Everyday Egocentric Images](../../ECCV2024/human_understanding/3d_hand_pose_estimation_in_everyday_egocentric_images.md)
- [\[CVPR 2025\] GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities](gigahands_a_massive_annotated_dataset_of_bimanual_hand_activities.md)

<!-- RELATED:END -->
