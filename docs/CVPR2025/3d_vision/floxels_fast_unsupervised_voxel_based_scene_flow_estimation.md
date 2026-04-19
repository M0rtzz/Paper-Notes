---
title: >-
  [论文解读] Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation
description: >-
  [CVPR 2025][3D视觉][场景流估计] 提出 Floxels，用简单的体素网格替代 MLP 作为场景流的隐式表示，结合多帧距离变换损失和聚类一致性约束，在 Argoverse 2 基准上取得仅次于 EulerFlow 的无监督方法第二名，同时将运行时间从一天缩短到10分钟（60-140倍加速）。
tags:
  - CVPR 2025
  - 3D视觉
  - 场景流估计
  - 体素网格
  - 无监督优化
  - 点云
  - 自动驾驶
---

# Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation

**会议**: CVPR 2025  
**arXiv**: [2503.04718](https://arxiv.org/abs/2503.04718)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 场景流估计, 体素网格, 无监督优化, 点云, 自动驾驶

## 一句话总结
提出 Floxels，用简单的体素网格替代 MLP 作为场景流的隐式表示，结合多帧距离变换损失和聚类一致性约束，在 Argoverse 2 基准上取得仅次于 EulerFlow 的无监督方法第二名，同时将运行时间从一天缩短到10分钟（60-140倍加速）。

## 研究背景与动机

1. **领域现状**：场景流估计（从连续LiDAR扫描估计3D运动场）主要有两类方法：(1) 监督方法快速但需大量标注数据且受域漂移影响；(2) 基于优化的无监督方法（如 NSFP、EulerFlow）不受域漂移影响但计算开销大。
2. **现有痛点**：EulerFlow 效果极好但需约24小时处理一个序列（V100上），实际不可用；NSFP/FNSF 虽然轻量但质量差，存在"风车伪影"（空区域预测错误光流）、遮挡阴影区域偏向近点匹配、以及因 MLP 收敛慢导致的精度不足。
3. **核心矛盾**：MLP 作为隐式表示收敛慢且有"均匀运动"偏差——梯度传播影响全局参数，导致局部细节难以捕捉；同时仅用两帧点云对时，遮挡变化导致错误的点对应。
4. **本文目标** (1) 如何加速收敛同时保持高质量？(2) 如何解决遮挡导致的错误对应？(3) 如何消除风车伪影和阴影区域错误？
5. **切入角度**：受 NeRF 领域中体素网格替代 MLP 可大幅加速收敛的启发，将同样的思路应用到场景流估计；扩展到多帧以处理遮挡；用聚类一致性鼓励邻近点有相似运动。
6. **核心 idea**：用体素网格显式参数化3D运动场替代 MLP，结合多帧距离变换约束和聚类一致性损失，以极低的计算代价获得接近 EulerFlow 的场景流质量。

## 方法详解

### 整体框架
输入为多帧LiDAR点云序列（默认5帧），输出为参考帧每个点的3D运动向量。方法首先构建覆盖场景的3D体素网格，每个网格顶点存储可学习的3D流向量；每个点的流通过三线性插值从邻近顶点获得。然后对参考帧的点施加流后，通过预计算的多帧距离变换评估匹配误差。同时用 DBSCAN 聚类鼓励同一簇内点有一致的运动。整个优化通过梯度下降进行，最多500个epoch，带早停。

### 关键设计

1. **体素网格替代 MLP**:

    - 功能：作为3D场景流场的显式参数化表示
    - 核心思路：构建覆盖场景的3D网格，每个顶点学习3D流向量 $f_t \in \mathbb{R}^3$，任意点的流通过三线性插值得到。与 MLP 相比，体素网格的梯度只影响邻近网格点，提供天然的局部平滑性正则化。默认体素大小 0.5 米，学习率 0.05。
    - 设计动机：MLP 收敛慢是因为每个点的梯度传播到所有参数；体素网格使梯度局部化，收敛快且自然消除了风车伪影——空区域没有梯度，流保持初始化的零值。

2. **多帧距离变换损失 (Multi-Scan DT Loss)**:

    - 功能：利用多个相邻帧约束流估计，解决遮挡导致的错误对应
    - 核心思路：假设恒速运动，将参考帧点按流投影到前后各 $m$ 帧的预计算距离变换中评估距离：$\ell_d = \sum_{t \neq 0} \frac{\lambda(t)}{N} D(\mathcal{S}_0 + f_0 \Delta_t, S_t)$，其中 $\lambda(t) = 1/t^2$ 使远帧权重衰减，距离变换中超过5米的值被截断以增强鲁棒性。仅需预计算 $2(m-1)$ 个距离变换。
    - 设计动机：两帧模式下，如果目标点在第二帧被遮挡就无法正确匹配；多帧模式下，即使某帧缺失对应点，相邻帧中很可能存在，从而提供正确的约束信号。

3. **聚类一致性损失 (Cluster Consistency Loss)**:

    - 功能：鼓励空间邻近点具有相似运动，防止动态物体与静态物体间的错误关联
    - 核心思路：用 DBSCAN（$\epsilon=0.5$, min_points=4）对参考帧点云聚类，确保过分割（每个物体可能对应多个簇），然后约束同一簇内点的流一致：$\ell_C = \frac{1}{N}\sum_i ||f_t^i - f_{C_i}||_2$，其中 $f_{C_i}$ 是簇 $C_i$ 的平均流。
    - 设计动机：过分割是安全的（不会造成灾难性错误），而将两个不同运动的物体并入同一簇会严重错误。这个约束有效对抗了动态物体靠近静态物体时的错误匹配。

### 损失函数 / 训练策略
- 最终损失：$\ell = \lambda_d \ell_d + (2m-1)(\lambda_C \ell_C + \lambda_\gamma \gamma)$
- $\gamma = ||f_t||_2$ 是流大小正则化，抑制静态区域的噪声流
- $(2m-1)$ 因子使聚类和正则化项随帧数线性缩放，与距离变换项保持平衡
- Adam优化器，学习率0.05，最多500 epoch，早停耐心250步

## 实验关键数据

### 主实验（Argoverse 2 场景流挑战赛测试集）

| 方法 | 类型 | mdnEPE↓ | 运行时间 |
|------|------|---------|---------|
| NSFP | 无监督优化 | ~0.55 | ~63s/帧 |
| FNSF | 无监督优化 | ~0.50 | ~21s/帧 |
| EulerFlow | 无监督优化 | **最优** | ~24小时/序列 |
| Flow4D | 监督 | 接近最优 | 快 |
| **Floxels (13帧)** | **无监督优化** | **第二名** | **~24分/序列** |

### 消融实验（nuScenes mini，动态点）

| 配置 | EPE↓ | Acc5↑ | Acc10↑ | 说明 |
|------|------|-------|--------|------|
| Floxels (完整) | **0.085** | **0.537** | **0.833** | 完整模型 |
| - flow norm | 0.084 | 0.528 | 0.833 | 流正则化影响小 |
| - cluster loss | 0.201 | 0.133 | 0.413 | 去掉聚类损失，暴跌 |
| - cluster + flow norm | 0.206 | 0.123 | 0.401 | 两者都去掉 |

| 帧数 | EPE↓ | Acc5↑ | 时间(s)↓ |
|------|------|-------|---------|
| 3帧 | 0.095 | 0.468 | 2.47 |
| 5帧 | 0.085 | 0.537 | 3.52 |
| 9帧 | 0.078 | 0.516 | 5.69 |
| 11帧 | 0.076 | 0.486 | 6.72 |

### 关键发现
- **聚类一致性损失贡献压倒性**：去掉后 EPE 从 0.085 飙升到 0.201，Acc5 从 0.537 暴跌到 0.133。这是最关键的组件。
- **体素网格彻底消除风车伪影**：MLP 在无物体的空区域预测错误流（风车形状），体素网格因空区域无梯度信号而自然保持零流。
- **速度 vs 质量权衡极佳**：Floxels 5帧仅需3.5秒/帧，比 FNSF 快4.7倍、比 NSFP 快14.4倍，且精度远超两者。相比 EulerFlow 加速60-140倍，质量仅次于它。
- **体素大小鲁棒**：0.3m-2.0m 范围内 EPE 变化很小，仅行人类别在2.0m时略有下降。

## 亮点与洞察
- **从 NeRF 到场景流的类比迁移**：NeRF 中体素网格替代 MLP（Plenoxels/DVGO）加速收敛的洞察被成功迁移到场景流估计——同样的"显式 vs 隐式"取舍在不同任务中重现，说明这是一个通用的范式。
- **过分割比欠分割安全**：聚类的过分割策略是一个重要的工程洞察——合并不同运动物体到同一簇会灾难性失败，但过分割最多损失一些局部一致性。
- **不需要时间条件化 MLP**：EulerFlow 的时间条件化 MLP 学习跨时间步的流变化但极慢。Floxels 证明简单的恒速假设+多帧约束就足够好，大幅简化了问题。

## 局限与展望
- **恒速假设**：多帧损失假设恒速运动，对加速/减速/转弯等变速场景可能不够精确。
- **体素分辨率限制**：0.5m 体素大小无法捕捉很小物体的精细运动（如行人手臂摆动）。
- **仅适用于LiDAR**：框架设计针对LiDAR点云，不直接适用于RGB场景流。
- **可改进**：引入多分辨率体素网格处理不同尺度运动；用可变形体素适应非均匀场景分布；扩展到图像域的场景流。

## 相关工作与启发
- **vs EulerFlow**：EulerFlow 用时间条件化 MLP + 全序列 Euler 积分获得最好精度，但需24小时。Floxels 用体素网格+5帧恒速假设在精度接近的同时加速60-140倍，展示了"足够好的简单方法"的价值。
- **vs NSFP/FNSF**：NSFP 用 MLP 表示场景流，FNSF 用距离变换加速。Floxels 分析了它们的失败模式（风车伪影、阴影区域偏差、近点偏差），并用体素网格+聚类+多帧逐一解决。
- **vs DifFlow3D (监督)**：监督方法快但需训练数据且有域漂移问题。在 Argoverse 2 上 Floxels（无监督）性能接近甚至超过 DifFlow3D，且不需要任何训练。

## 评分
- 新颖性: ⭐⭐⭐⭐ 体素替代MLP的思路虽源自NeRF但在场景流中的应用和分析很有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多基线、详细消融、计算效率分析、失败案例可视化
- 写作质量: ⭐⭐⭐⭐ 问题分析深入，图示清晰，故事完整
- 价值: ⭐⭐⭐⭐⭐ 实用性极强——在自动驾驶等对延时敏感的场景中，Floxels 是目前无监督方法的最佳选择

<!-- RELATED:START -->

## 相关论文

- [Flux4D: Flow-based Unsupervised 4D Reconstruction](../../NeurIPS2025/3d_vision/flux4d_flow-based_unsupervised_4d_reconstruction.md)
- [P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning](p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)
- [milliFlow: Scene Flow Estimation on mmWave Radar Point Cloud for Human Motion Sensing](../../ECCV2024/3d_vision/milliflow_scene_flow_estimation_on_mmwave_radar_point_cloud_for_human_motion_sen.md)
- [Flow-NeRF: Joint Learning of Geometry, Poses, and Dense Flow within Unified Neural Representations](flow-nerf_joint_learning_of_geometry_poses_and_dense_flow_within_unified_neural_.md)
- [PCSTracker: Long-Term Scene Flow Estimation for Point Cloud Sequences](../../CVPR2026/3d_vision/pcstracker_long-term_scene_flow_estimation_for_point_cloud_sequences.md)

<!-- RELATED:END -->
