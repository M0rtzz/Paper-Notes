---
title: >-
  [论文解读] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots
description: >-
  [CVPR2026][自动驾驶][全景占据预测] 提出首个面向四足机器人的全景多模态语义占据预测数据集 PanoMMOcc 及框架 VoxelHound，通过垂直抖动补偿（VJC）和多模态信息提示融合（MIPF）模块，在全景 RGB+热成像+偏振+LiDAR 四模态下达到 23.34% mIoU，超越已有方法 +4.16%。
tags:
  - CVPR2026
  - 自动驾驶
  - 全景占据预测
  - 多模态融合
  - 四足机器人
  - 语义占据
  - BEV感知
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots

**会议**: CVPR2026  
**arXiv**: [2603.13108](https://arxiv.org/abs/2603.13108)  
**代码**: [PanoMMOcc](https://github.com/) (即将公开)  
**领域**: autonomous_driving  
**关键词**: 全景占据预测, 多模态融合, 四足机器人, 语义占据, BEV感知

## 一句话总结

提出首个面向四足机器人的全景多模态语义占据预测数据集 PanoMMOcc 及框架 VoxelHound，通过垂直抖动补偿（VJC）和多模态信息提示融合（MIPF）模块，在全景 RGB+热成像+偏振+LiDAR 四模态下达到 23.34% mIoU，超越已有方法 +4.16%。

## 背景与动机

1. **全景感知需求**：全景图像提供 360° 无盲区视觉覆盖，对动态非结构化场景中的移动智能体至关重要，但现有占据预测方法主要面向自动驾驶的多目标窄 FoV 相机
2. **四足平台挑战**：四足机器人相比轮式平台存在低视角、频繁自遮挡、步态动力学导致的强烈自运动等固有挑战，现有方法未考虑这些特殊性
3. **单一 RGB 局限**：仅依赖 RGB 模态在光照变化、低纹理区域和远距离感知场景下鲁棒性不足，需要多模态传感融合
4. **数据集缺口**：现有全景数据集聚焦 2D 视觉任务缺乏 3D 占据标注；现有占据基准面向自动驾驶场景，未涉及全景成像或四足平台
5. **融合策略不足**：现有多模态融合常用简单拼接/加法，对异构传感器贡献无差别对待，导致几何一致性被稀释、模态间干扰
6. **步态抖动问题**：四足行走引起垂直方向的体态振荡，导致采集图像的空间特征错位，降低 BEV 变换的稳定性

## 方法详解

### 整体框架 VoxelHound

输入四种模态：全景 RGB 图像 $\mathcal{I}^{pal}$、热成像 $\mathcal{I}^{th}$、偏振图像 $\mathcal{I}^{pol}$、LiDAR 点云 $\mathcal{P}$。各模态经独立编码器提取特征，投影到 BEV 空间进行统一融合，最终输出 $\mathbf{O} \in \mathbb{R}^{X \times Y \times Z}$ 的三维语义占据预测。

### 多模态融合网络

- **Camera 分支**：三种图像模态各经 ResNet-18 提取多尺度特征，FPN 聚合后通过 2D→BEV 视角变换得到各模态 BEV 特征 $\mathbf{F}_c^m \in \mathbb{R}^{C_m \times H \times W}$
- **LiDAR 分支**：点云体素化后经稀疏 3D 卷积（stride=8）提取层级几何特征，压扁为 BEV 特征 $\mathbf{F}_l \in \mathbb{R}^{C_l \times H \times W}$
- **Fusion 分支**：融合后经 SECOND-FPN BEV 编码器增强多尺度空间表示，占据头将 BEV 通道 reshape 为垂直 bin，输出 64×64×16 的体素预测

### 垂直抖动补偿模块（VJC）

插入在图像编码器与 BEV 视角变换之间：
1. 沿宽度维度均值池化获取垂直结构 $\mathbf{F}_v \in \mathbb{R}^{C \times H}$
2. 两层 Conv1D + ReLU 编码垂直特征
3. 自适应平均池化 + 线性层预测全局垂直偏移量 $\Delta h$
4. 归一化为网格坐标偏移，构造位移采样网格
5. 双线性网格采样得到补偿后特征 $\mathbf{F}_{comp}$

### 多模态信息提示融合模块（MIPF）

采用"几何主导+语义补充"的非对称融合原则：
1. 各模态经 1×1 卷积投影到共享嵌入空间
2. 对图像模态全局平均池化 + MLP 生成紧凑语义提示 $\mathbf{p}_m$
3. LiDAR BEV 特征作为 Query，模态提示作为 Key/Value 进行提示注意力
4. 残差调制：$\mathbf{F}_f = \tilde{\mathbf{F}}_l + \sigma(\gamma(\mathbf{F}_{attn})) \odot \tilde{\mathbf{F}}_l$，确保几何结构为主表示基础

### 损失函数

$$\mathcal{L}_{occ} = \mathcal{L}_{ce} + \mathcal{L}_{ls} + \mathcal{L}_{scal}^{geo} + \mathcal{L}_{scal}^{sem}$$

包含交叉熵损失、Lovász-Softmax 损失、几何/语义亲和力损失。

## 实验关键数据

### 数据集 PanoMMOcc

- 四足机器人 Unitree Go2 搭载全景相机（360°×70° FoV, 2048×2048）、MID360 LiDAR、热成像（640×512）、偏振相机（1224×1024）
- 54 个序列 @10Hz、每序列 40 秒，共 21,600 帧；42 序列标注、12 类语义
- 体素空间: 64×64×16，分辨率 0.4m，空间范围 [-12.8, 12.8]m (xy) × [-2.4, 4.0]m (z)
- 训练 30 序列 / 测试 12 序列，覆盖校园、城市、住宅、绿地、乡村、森林六类场景

### 主实验结果 (mIoU %)

| 方法 | 模态 | mIoU |
|------|------|------|
| MonoScene | C | 8.94 |
| EFFOcc-C | C | 4.47 |
| EFFOcc-L | L | 18.77 |
| EFFOcc-T | C+L | 19.18 |
| C-CONet | C | 3.79 |
| M-CONet | C+L | 4.68 |
| **VoxelHound** | C | 5.79 |
| **VoxelHound** | C+T+P | 6.14 |
| **VoxelHound** | C+L | 22.87 |
| **VoxelHound** | **C+L+T+P** | **23.34** |

- 全模态 VoxelHound 超越最强竞争方法 EFFOcc-T **+4.16% mIoU**
- 相比纯相机 MonoScene **+14.40% mIoU**
- 热成像+偏振在夜间提升显著：夜间 C→C+T+P mIoU 从 3.52% 升至 4.07%

### 消融实验

| VJC | MIPF | mIoU |
|-----|------|------|
| ✗ | ✗ | 22.74 |
| ✓ | ✗ | 22.92 |
| ✗ | ✓ | 23.14 |
| ✓ | ✓ | **23.34** |

- VJC 贡献 +0.18%，MIPF 贡献 +0.40%，二者组合额外 +0.20%
- VJC 最优隐通道维度 $C_{hd}=64$，MIPF 最优 $C_{pd}=8, C_{nh}=8$

## 亮点

1. **首创性**：首个面向四足机器人的全景多模态占据预测数据集和框架，填补了全景占据+四足平台的空白
2. **VJC 模块设计精巧**：仅通过轻量 1D 卷积估计垂直偏移量进行网格采样补偿，参数开销极小（+0.04M）
3. **MIPF 非对称融合**：将图像模态压缩为紧凑提示避免密集空间交叉注意力，"几何主导+语义补充"的设计符合传感器特性
4. **四模态传感**：融合全景 RGB、热成像、偏振、LiDAR 四种互补模态，首次引入偏振成像到占据预测任务

## 局限与展望

1. **绝对性能仍偏低**：最优 mIoU 仅 23.34%，部分类别（bicycle=0.00%, pedestrian=0.00%）完全无法检测，小目标感知严重不足
2. **VJC 提升有限**：单独 VJC 仅带来 +0.18% 提升，对步态抖动的补偿效果有限，可能需要更复杂的时序建模
3. **数据规模较小**：21.6k 帧的数据量相比 nuScenes 等驾驶数据集偏少，泛化能力有待验证
4. **仅评估自建数据集**：VoxelHound 未在已有公共占据基准上验证，跨数据集泛化未知
5. **夜间全模态反而下降**：夜间 C+L+T+P (18.68%) 低于 C+L (19.17%)，暗示热/偏振在特定条件下引入噪声

## 与相关工作的对比

- **vs MonoScene/EFFOcc/CONet**：均为自动驾驶场景设计，使用针孔相机多目方案，不适配全景成像+四足平台的场景；VoxelHound 在四足场景下全模态显著领先
- **vs QuadOcc**：同为四足平台全景占据数据集，但 QuadOcc 仅含 RGB 单模态、6 类语义、体素 64×64×8；PanoMMOcc 扩展为四模态、12 类、64×64×16
- **现有多模态融合**：多数方案用 Camera+LiDAR 简单拼接/对称融合；MIPF 的非对称提示注意力更好利用了模态间互补性
- **全景感知领域**：已有工作主要做 2D 语义分割/BEV 映射，本文首次将全景视觉扩展到 3D 占据预测

## 评分

- 新颖性: ⭐⭐⭐⭐ (首个四足全景多模态占据数据集+框架，VJC/MIPF 设计合理)
- 实验充分度: ⭐⭐⭐ (消融完整但绝对性能偏低，缺少跨数据集实验)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，图表丰富)
- 价值: ⭐⭐⭐⭐ (开创四足机器人全景占据方向，数据集和基准有长期价值)

<!-- RELATED:START -->

## 相关论文

- [OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera](oneocc_semantic_occupancy_prediction_for_legged_robots_with_a_single_panoramic_c.md)
- [An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)
- [CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection](coin3d_revisiting_configuration-invariant_multi-camera_3d_object_detection.md)
- [TT-Occ: Test-Time 3D Occupancy Prediction](test-time_3d_occupancy_prediction.md)
- [Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization](learning_geometric_and_photometric_features_from_panoramic_lidar_scans_for_outdo.md)

<!-- RELATED:END -->
