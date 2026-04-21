---
title: >-
  [论文解读] RC-AutoCalib: An End-to-End Radar-Camera Automatic Calibration Network
description: >-
  [CVPR 2025][自动驾驶][雷达-相机标定] 提出 RC-AutoCalib，首个针对 3D 雷达和相机的端到端在线自动几何标定方法，通过双视角（前视+鸟瞰）特征表示、选择性融合机制和噪声抗性匹配器，有效解决雷达数据稀疏和高度不确定性问题，在 nuScenes 数据集上大幅超越现有 LiDAR-相机标定方法。
tags:
  - CVPR 2025
  - 自动驾驶
  - 雷达-相机标定
  - 在线自标定
  - 双视角表示
  - 特征匹配
  - 噪声抗性
---

# RC-AutoCalib: An End-to-End Radar-Camera Automatic Calibration Network

**会议**: CVPR 2025  
**arXiv**: [2505.22427](https://arxiv.org/abs/2505.22427)  
**代码**: https://github.com/nycu-acm/RC-AutoCalib  
**领域**: 自动驾驶 / 传感器标定  
**关键词**: 雷达-相机标定, 在线自标定, 双视角表示, 特征匹配, 噪声抗性

## 一句话总结

提出 RC-AutoCalib，首个针对 3D 雷达和相机的端到端在线自动几何标定方法，通过双视角（前视+鸟瞰）特征表示、选择性融合机制和噪声抗性匹配器，有效解决雷达数据稀疏和高度不确定性问题，在 nuScenes 数据集上大幅超越现有 LiDAR-相机标定方法。

## 研究背景与动机

**领域现状**：雷达和相机因成本低廉和全天候工作能力，在 ADAS 系统中越来越受欢迎。传感器间的精确标定是多模态融合的基础。现有标定方法主要分为离线标定（需要棋盘格等标定物，耗时费力且不能应对运行中的传感器偏移）和在线标定（利用自然场景特征，更灵活适应动态变化）。

**现有痛点**：(1) 雷达-相机在线自标定几乎没有被探索过，仅有 Schöller 等人用深度学习做旋转标定、未涉及平移标定；(2) 虽然 LiDAR-相机在线标定（如 LCCNet, CalibDepth）已有成熟方案，但雷达数据有两个独特难题：**稀疏性**（点数远少于 LiDAR）和 **高度不确定性**（雷达在高度维度测量精度极差，导致投影到前视图时深度值包含大量噪声）；(3) 现有方法主要从单一前视图视角提取特征，而雷达投影到前视图后点更稀疏且充满噪声。

**核心矛盾**：雷达数据的稀疏性和高度不确定性使得传统 LiDAR-相机标定方案直接用于雷达-相机标定效果很差。

**本文目标**：设计一个能有效处理稀疏和噪声雷达数据的端到端在线自标定网络，同时估计 6-DoF 的旋转和平移参数。

**切入角度**：鸟瞰图（BEV）不受高度不确定性影响（因为 BEV 只用 X 和 Z 坐标），可以提供对高度噪声鲁棒的特征。因此用双视角互补——前视图保留丰富语义但受噪声影响，BEV 提供稳定几何但损失语义。

**核心 idea**：双视角表示 + 基于注意力的选择性融合 + 显式特征匹配监督（通过噪声抗性匹配器提供更干净的匹配 GT）。

## 方法详解

### 整体框架

输入为 RGB 图像和雷达点云以及初始标定参数 $T_{init}$。数据变换模块将它们转换为四种表示：前视图深度图（相机估计 + 雷达投影）和 BEV 图（伪 BEV 图像 + 雷达 BEV 投影）。经特征提取（ResNet）后进入特征匹配模块（含多模态交叉注意力和显式匹配监督），再通过选择性融合机制合并双视角特征，最后由回归头（LSTM 序列解码器）预测旋转和平移向量。支持迭代精化：预测的 $T_{pred}^i$ 更新 $T_{init}$ 后重新输入网络。

### 关键设计

1. **双视角（Dual-Perspective）特征表示**:

    - 功能：从前视图和鸟瞰图两个互补视角提取雷达和相机特征
    - 核心思路：**雷达数据**：用初始标定参数将雷达 3D 点 $P_r$ 变换到相机坐标系，分别投影到前视图（记录深度 $Z_r^c$）和 BEV（记录高度 $Y_r^c$）。**相机数据**：用 DepthAnything+ZoeDepth 从 RGB 图估计度量深度作为前视图特征；将深度图反投影为伪点云再投影到 BEV 生成伪 BEV 图像。最终得到四对特征图：$I_R^{FV}, I_I^{FV}$（前视图）和 $I_R^{BEV}, I_I^{BEV}$（BEV）
    - 设计动机：BEV 视角中雷达数据不受高度不确定性影响（只用 X, Z 坐标），提供更稳定的几何特征；前视图保留丰富的语义和结构信息但受高度噪声污染。两者互补

2. **多模态交叉注意力 + 显式特征匹配监督**:

    - 功能：在每个视角内显式建立雷达和相机特征之间的对应关系
    - 核心思路：交叉注意力（MCA）让雷达和相机特征互相关注，计算注意力分数 $a_{IR} = K_I^\top K_R$，得到 attended 特征 $m_{I\leftarrow R}$ 和 $m_{R\leftarrow I}$。在此基础上用 Residual Conv Block 聚合为统一特征 $F_{view}$。训练时额外设置匹配分支：通过 softmax 归一化的相似度矩阵 $S$ 和可匹配性分数 $\sigma_*$ 计算赋值矩阵 $P$，与 GT 匹配矩阵 $\mathcal{M}$ 做匹配损失监督
    - 设计动机：之前方法只用拼接+卷积做隐式匹配，仅靠最终标定损失间接监督，无法明确学到对应点对。显式匹配监督让网络真正理解雷达和图像之间的几何对应

3. **噪声抗性匹配器（Noise-Resistant Matcher）**:

    - 功能：在前视图匹配中过滤因高度不确定性导致的不可靠雷达点，提供更干净的匹配 GT
    - 核心思路：利用 LiDAR 数据（仅训练时使用）识别不可靠雷达点。对每个雷达 3D 点构建自适应 3D 包围盒 $B$（高度 $h_B$、宽度 $w_B$、深度 $d_B$ 根据雷达的仰角 $\phi$、方位角 $\theta$、距离 $R$ 和允许误差 $\delta$ 自适应计算），若盒内 LiDAR 点数超过阈值 $\tau$ 则认为该雷达点可靠，否则从匹配 GT $\mathcal{M}$ 中剔除
    - 设计动机：由于雷达高度测量精度差，远离雷达平面的反射信号会产生不可靠的 3D 位置。直接用这些噪声点作为匹配 GT 会误导网络学习

### 损失函数 / 训练策略

总损失 $L_{total} = L_{calib} + \beta L_{matching}$。匹配损失 $L_{matching} = L_{M_{bev}} + L_{M_{fv}}$，每个视角的匹配损失包括正例损失（匹配对的 log 似然）和负例损失（非匹配点的 no-matchable 分数）。标定损失 $L_{calib}$ 采用 CalibDepth 的迭代标定损失。使用 nuScenes 数据集，12610 样本训练、1628 验证、1623 测试，深度范围 0-200m，输入分辨率 400×192。回归头用 LSTM 序列解码器进行多步自回归预测。

## 实验关键数据

### 主实验

误标定范围 R1 (±10°, ±0.25m):

| 方法 | 旋转误差(°) Mean | Roll | Pitch | Yaw | 平移误差(cm) Mean | X | Y | Z |
|------|----------|------|-------|-----|----------|---|---|---|
| LCCNet-1 | 1.603 | 0.123 | 3.130 | 1.556 | 16.531 | 22.99 | 17.65 | 8.95 |
| CalibDepth | 0.807 | 0.390 | 0.345 | 1.686 | 12.608 | 12.86 | 12.25 | 12.72 |
| **Ours** | **0.427** | **0.130** | **0.198** | **0.953** | **9.498** | 12.56 | **3.30** | 12.64 |

误标定范围 R2 (±20°, ±1.5m):

| 方法 | 旋转误差(°) Mean | 平移误差(cm) Mean |
|------|----------|----------|
| CalibDepth | 1.686 | 55.380 |
| **Ours** | **0.852** | **47.537** |

### 消融实验

| FV | BEV | SF | MCA | EMS | NR | Rot Mean(°) | Trans Mean(cm) |
|----|-----|----|----|-----|----|----|-----|
| ✓ | | | | | | 0.657 | 12.602 |
| | ✓ | | | | | 0.689 | 12.605 |
| ✓ | ✓ | | | | | 0.575 | 12.315 |
| ✓ | ✓ | ✓ | | | | 0.529 | 11.842 |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **0.427** | **9.498** |

### 关键发现

- 双视角组合（FV+BEV）比单一视角分别降低旋转误差 12.5% 和 16.5%
- 选择性融合（SF）在双视角基础上进一步降低旋转误差 8%，说明自适应选择比简单合并更有效
- 显式匹配监督（EMS）对平移标定贡献最大，Y 方向平移误差从 9.98cm 降至 3.30cm
- 噪声抗性匹配器（NR）有效过滤了前视图中的噪声匹配对，进一步提升精度
- 在大误标定范围（R2）下优势更明显，旋转误差仅 0.852°，远低于 CalibDepth 的 1.686°

## 亮点与洞察

- **首个雷达-相机在线自标定的完整方案**：不仅做了旋转还做了平移标定，填补了领域空白，并且超越了 LiDAR-相机方法（这意味着用更便宜的雷达也能获得精度可比甚至更好的标定）
- **BEV 视角规避高度不确定性的洞察**：非常聪明地利用了雷达在 X-Z 平面上精度高但 Y 轴精度差的特性，BEV 视角完全避开了问题维度
- **自适应 3D 包围盒的设计**：根据每个雷达点的角度和距离动态调整包围盒大小来判断可靠性，比简单阈值更物理合理
- **训练时用 LiDAR 辅助、推理时不需要 LiDAR**：巧妙地利用了 nuScenes 同时有 LiDAR 的优势来生成更干净的训练数据

## 局限与展望

- 训练时依赖 LiDAR 数据来构建噪声抗性匹配器的 GT，限制了在纯雷达-相机系统上的训练可能
- 使用 DepthAnything+ZoeDepth 做深度估计是固定的预处理步骤，估计误差会传播到后续模块
- 迭代精化的次数需要手动设定，自适应终止策略是一个有趣方向
- 目前只在 nuScenes 数据集上验证，在更多驾驶数据集（如 Waymo、ONCE）和极端天气条件下的鲁棒性有待验证

## 相关工作与启发

- **vs CalibDepth (LiDAR-Camera)**: 当前 SOTA LiDAR-相机方法，RC-AutoCalib 在用雷达替代 LiDAR 的情况下取得了更好的旋转精度和可比的平移精度
- **vs LCCNet**: 早期 LiDAR-相机方法，使用 cost volume 做特征匹配但缺乏显式匹配监督，在雷达场景下效果较差
- **vs Schöller et al.**: 唯一的前人雷达-相机深度学习标定工作，但仅处理旋转且使用的是固定交通雷达而非车载雷达
- 双视角思路可迁移到其他涉及稀疏 3D 数据和 2D 图像对齐的场景（如 ToF 相机标定）

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个完整的雷达-相机在线自标定方案，双视角+噪声抗性匹配器设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 消融充分，但仅一个数据集
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻，方法描述详细
- 价值: ⭐⭐⭐⭐ 对自动驾驶雷达-相机融合系统有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](diffusiondrive_truncated_diffusion_model_for_end-to-end_autonomous_driving.md)
- [SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving](solve_synergy_of_language-vision_and_end-to-end_networks_for_autonomous_driving.md)
- [TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-Stage Fusion](tacodepth_towards_efficient_radar-camera_depth_estimation_with_one-stage_fusion.md)
- [RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)
- [World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](../../ICCV2025/autonomous_driving/world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)

<!-- RELATED:END -->
