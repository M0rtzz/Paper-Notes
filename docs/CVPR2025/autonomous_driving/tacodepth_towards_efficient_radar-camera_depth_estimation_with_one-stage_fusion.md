---
title: >-
  [论文解读] TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-Stage Fusion
description: >-
  [CVPR 2025][自动驾驶][深度估计] TacoDepth 提出首个单阶段雷达-相机融合深度估计框架，通过基于图的雷达结构提取器和金字塔式雷达融合模块，绕过了中间准稠密深度图的需求，在精度提升 12.8% 的同时速度提升 91.8%，达到实时 37+ FPS。 领域现状：雷达-相机深度估计是自动驾驶和机器人三维感知的…
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "深度估计"
  - "雷达-相机融合"
  - "单阶段融合"
  - "图神经网络"
  - "实时处理"
---

# TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-Stage Fusion

**会议**: CVPR 2025  
**arXiv**: [2504.11773](https://arxiv.org/abs/2504.11773)  
**代码**: [https://github.com/RaymondWang987/TacoDepth](https://github.com/RaymondWang987/TacoDepth)  
**领域**: 自动驾驶  
**关键词**: 深度估计, 雷达-相机融合, 单阶段融合, 图神经网络, 实时处理

## 一句话总结

TacoDepth 提出首个单阶段雷达-相机融合深度估计框架，通过基于图的雷达结构提取器和金字塔式雷达融合模块，绕过了中间准稠密深度图的需求，在精度提升 12.8% 的同时速度提升 91.8%，达到实时 37+ FPS。

## 研究背景与动机

**领域现状**：雷达-相机深度估计是自动驾驶和机器人三维感知的关键任务。相比激光雷达，毫米波雷达成本低、功耗小、全天候可靠，但其点云极度稀疏（比激光雷达稀 1000 倍）且噪声大。

**现有痛点**：为克服雷达稀疏性，主流方法（Singh et al., RC-PDA, RadarCam-Depth 等）采用复杂的多阶段框架——先从稀疏雷达预测中间"准稠密深度"，再基于此预测最终稠密深度。问题有三：（1）多阶段推理效率低，RadarCam-Depth 的四阶段流程达 358ms；（2）中间准稠密深度仍然很稀疏且有噪声，有时在恶劣光照下几乎无有效深度值；（3）缺陷的中间结果导致最终深度出现结构断裂、细节模糊和明显伪影。

**核心矛盾**：雷达点云极度稀疏→需要中间跳板→多阶段带来效率和鲁棒性问题。能否一步到位，从稀疏雷达直接到稠密深度？

**本文目标**：实现单阶段的雷达-相机深度估计，同时提升效率和精度，并支持独立推理和插件式推理两种模式。

**切入角度**：之前方法只从雷达点的单点坐标提取特征，忽略了点云的几何结构和拓扑信息。点云的图结构（点间距离/关系）比单点坐标更信息丰富且对噪声更鲁棒。

**核心 idea**：用 GNN 提取雷达点云的图结构特征，通过金字塔式逐层融合将雷达结构信息与图像的多尺度语义特征高效整合，实现单阶段融合。

## 方法详解

### 整体框架

输入为单张 RGB 图像 $I \in \mathbb{R}^{H \times W \times 3}$ 和雷达点云 $P \in \mathbb{R}^{K \times 3}$（K 个点的三维坐标），输出为稠密度量深度图 $D \in \mathbb{R}^{H \times W}$。框架分两步：（1）图基雷达结构提取器从点云中抽取节点特征和边特征的层级表示；（2）金字塔式雷达融合模块将雷达特征与 ResNet-18 提取的图像多尺度特征逐层融合；最后解码器输出深度图。

### 关键设计

1. **图基雷达结构提取器（Graph-based Radar Structure Extractor, GE）**:

    - 功能：从稀疏雷达点云中提取丰富的几何结构和拓扑特征，替代之前方法中简单的 MLP 逐点特征提取
    - 核心思路：将雷达点视为图节点，构建邻接矩阵表示边。采用轻量级 GNN 架构（PCA-GM），在 $L=3$ 层中逐层更新节点特征 $N_l$ 和聚合边特征 $E_l$。浅层捕获点坐标，深层捕获全局拓扑结构。输出每层的节点特征和边特征供后续融合
    - 设计动机：Singh et al. 用 MLP 从 3D 坐标提取 32,256 维特征，引入大量冗余和噪声；GNN 可以建模点间几何关系（如距离、方向），这些关系对噪声和离群点更鲁棒，且为单阶段融合提供了比单点坐标更有效的信息

2. **金字塔式雷达融合模块（Pyramid-based Radar Fusion, PF）**:

    - 功能：将雷达图结构特征与图像多尺度特征层级式融合，实现高效的跨模态关联
    - 核心思路：GNN 第 $l$ 层的节点特征 $N_l$ 与图像特征 $F_{2l-1}$ 融合，边特征 $E_l$ 与 $F_{2l}$ 融合。每层内使用"雷达中心 Flash Attention"建立跨模态对应关系：对每个雷达点 $p$，基于其水平坐标 $x_p$ 定义一个窗口区域 $[x_p - a_l, x_p + a_l]$，只在该区域内的图像像素和雷达点之间计算 attention。查询来自图像特征，键值来自雷达边特征：$F'_{2l}[m] = \text{softmax}\frac{W_q \hat{F}_{2l}[m] (W_k \hat{E}_l)^T}{\sqrt{C_l}} W_v \hat{E}_l$
    - 设计动机：全图 attention 计算量过大且无关像素干扰多；利用雷达水平坐标较准确的特性（相比高度坐标），以水平位置为中心划定局部区域做 Flash Attention，大幅降低计算成本。浅层融合细节+坐标，深层融合语义+结构，形成互补

3. **灵活推理模式（Independent + Plug-in）**:

    - 功能：支持独立推理（无需外部深度模型，37+ FPS 实时）和插件式推理（利用预训练深度模型提升精度）
    - 核心思路：增加一个可选输入分支处理初始相对深度 $D^*$。独立模式下 $D^* = 0$；插件模式下接入 DPT、MiDaS、Depth-Anything-v2 等的相对深度输出。训练时每个 epoch 随机一半数据用相对深度、一半用零输入，同时优化两种模式：$D = \mathcal{T}_\theta(I, P | D^*)$
    - 设计动机：独立模型高效但精度有限，插件模型精度高但增加延迟。统一框架让用户按需选择，且可无缝受益于更强的深度预测器

### 损失函数 / 训练策略

训练损失为 L1 损失的组合：$\ell_{L_1} = \frac{1}{|\Omega_{gt}|}\sum|D - D_{gt}| + \frac{\lambda}{|\Omega_{acc}|}\sum|D - D_{acc}|$，其中 $D_{gt}$ 为激光雷达深度，$D_{acc}$ 为重投影累积深度，$\lambda=1$。使用 Adam 优化器，2 张 A6000 训练 50 个 epoch，初始学习率 1e-4，每 10 个 epoch 衰减 1e-5。

## 实验关键数据

### 主实验

nuScenes 0-70m 范围（独立模式）：

| 方法 | MAE↓ | RMSE↓ | 推理时间(ms)↓ |
|------|------|-------|-------------|
| **TacoDepth** | **1712.6** | **3960.5** | **26.7** |
| Li et al. (ECCV'24) | 1822.9 | 4303.6 | 67.6 |
| Singh et al. (CVPR'23) | 2073.2 | 4590.7 | 94.2 |
| CaFNet (IROS'24) | 2010.3 | 4493.1 | 103.9 |

nuScenes 0-70m 范围（插件模式，DPT-Hybrid）：

| 方法 | MAE↓ | RMSE↓ | 推理时间(ms)↓ |
|------|------|-------|-------------|
| **TacoDepth** | **1347.1** | **3152.8** | **29.3** |
| RadarCam-Depth | 1587.9 | 3662.5 | 358.3 |

### 消融实验

| 配置 | MAE↓ | RMSE↓ |
|------|------|-------|
| RGB Baseline (无雷达) | 2474.3 | 5402.1 |
| Singh et al. (MLP+两阶段) | 2073.2 | 4590.7 |
| TacoDepth w/o GE (MLP+PF) | 1815.6 | 4189.8 |
| **TacoDepth (GE+PF)** | **1712.6** | **3960.5** |

模型效率对比：

| 方法 | 参数量(M)↓ | FLOPs(G)↓ | 时间(ms)↓ |
|------|-----------|-----------|----------|
| **TacoDepth 独立** | **13.47** | **139.30** | **26.7** |
| Singh et al. | 22.81 | 502.09 | 94.2 |
| RadarCam-Depth | 33.26 | 619.02 | 358.3 |

### 关键发现

- PF（金字塔融合）贡献最大：仅换用 PF 不换 GE（用 MLP），MAE 就从 2073 降到 1816（降 12.4%），说明单阶段融合策略本身的价值
- GE 在 PF 基础上进一步降 MAE 5.7%，验证了图结构特征比逐点特征更有效
- GNN 层数 $L=3$ 最优，更深（$L=4$）反而因雷达过于稀疏无法获益
- 夜间场景提升更大（MAE 降 29.1% vs 白天 12.9%），证明单阶段方法在恶劣条件下更鲁棒——多阶段方法的中间深度在夜间几乎失效
- 插件模式搭配更强的深度预测器（Depth-Anything-v2）持续提升：MAE 从 983→730，展示了良好的兼容性

## 亮点与洞察

- **单阶段 vs 多阶段的范式突破**：直觉上稀疏雷达→稠密深度需要中间跳板，但本文证明用更好的特征提取和融合策略可以跳过中间步骤，这种"去中间化"的思路值得在其他稀疏到稠密任务中探索
- **雷达中心 Flash Attention**：利用雷达水平坐标准确这一领域先验，用极低成本的局部注意力替代全局注意力，既高效又避免了无关区域干扰。这个设计可迁移到任何稀疏传感器与稠密传感器融合的场景
- **双模式训练策略**：训练时随机给一半数据初始深度、一半给零，一个模型同时学会两种推理模式——设计简洁但效果很好

## 局限与展望

- 作者承认目前只提供了一种高效实现，更多技术可探索
- 仅在 nuScenes（3D 雷达）和 ZJU-4DRadarCam（4D 雷达）上验证，泛化性有待更多场景测试
- 图像编码器使用 ResNet-18，升级为 ViT 等更强骨干可能进一步提升
- GNN 对点云的建图策略（全连接邻接矩阵）在点数增多时可能遇到扩展性问题

## 相关工作与启发

- **vs Singh et al. (CVPR'23)**: 代表性的两阶段独立方法，用 MLP 提取雷达特征+门控融合；本文用 GNN+金字塔融合，参数少 41%、FLOPs 少 72%，精度更高
- **vs RadarCam-Depth (ICRA'24)**: 四阶段插件方法（初始深度→全局尺度对齐→准稠密深度→尺度学习器）；本文单阶段速度快 92%，精度高 12%
- **vs Depth-Anything-v2**: 纯视觉相对深度模型；本文可作为插件将其相对深度转化为准确的度量深度

## 评分

- 新颖性: ⭐⭐⭐⭐ 单阶段融合范式和雷达中心注意力设计新颖，但整体组件（GNN/Flash Attention/金字塔融合）是已有技术的巧妙组合
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集、多种对比、消融详尽、效率分析全面、日夜场景分别评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰、图表信息量大，动机和痛点阐述到位
- 价值: ⭐⭐⭐⭐ 实时雷达-相机融合深度估计对自动驾驶实用性强，开源代码增加了可复现性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)
- [\[CVPR 2025\] Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation](prompting_depth_anything_for_4k_resolution_accurate_metric_depth_estimation.md)
- [\[CVPR 2026\] R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](../../CVPR2026/autonomous_driving/r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)
- [\[CVPR 2025\] Toward Real-World BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting](toward_real-world_bev_perception_depth_uncertainty_estimation_via_gaussian_splat.md)
- [\[CVPR 2025\] RC-AutoCalib: An End-to-End Radar-Camera Automatic Calibration Network](rc-autocalib_an_end-to-end_radar-camera_automatic_calibration_network.md)

</div>

<!-- RELATED:END -->
