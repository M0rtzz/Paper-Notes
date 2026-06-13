---
title: >-
  [论文解读] Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method
description: >-
  [CVPR 2025][自动驾驶][道路图提取] 本文构建了一个覆盖全球的大规模卫星道路图提取数据集 Global-Scale（比现有最大公开数据集大约20倍），并提出 SAM-Road++ 方法，通过节点引导重采样策略解决训练与推理不匹配问题，同时引入"延长线"策略缓解遮挡导致的道路断裂，在多个数据集上取得了 SOTA 效果。
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "道路图提取"
  - "卫星图像"
  - "全球数据集"
  - "图连接预测"
  - "SAM"
---

# Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method

**会议**: CVPR 2025  
**arXiv**: [2411.16733](https://arxiv.org/abs/2411.16733)  
**代码**: [https://github.com/earth-insights/samroadplus](https://github.com/earth-insights/samroadplus)  
**领域**: 自动驾驶/遥感  
**关键词**: 道路图提取, 卫星图像, 全球数据集, 图连接预测, SAM

## 一句话总结

本文构建了一个覆盖全球的大规模卫星道路图提取数据集 Global-Scale（比现有最大公开数据集大约20倍），并提出 SAM-Road++ 方法，通过节点引导重采样策略解决训练与推理不匹配问题，同时引入"延长线"策略缓解遮挡导致的道路断裂，在多个数据集上取得了 SOTA 效果。

## 研究背景与动机

**领域现状**：道路图提取是自动驾驶和导航系统中的关键任务。现有端到端方法分为迭代式（RNGDet 系列，逐点生成道路图）和全局式（SAM-Road，全局预测道路图） 两大类。迭代方法容易积累误差且计算开销大，全局方法效率更高但仍存在设计缺陷。

**现有痛点**：SAM-Road 是全局方法的代表，其训练阶段使用 ground truth 节点训练连接分类器，但推理阶段使用从预测分割掩码中提取的节点作为输入。这导致训练与推理阶段的输入分布存在根本性不匹配（mismatch），分类器在推理时表现受限。此外，卫星图像中树木、建筑阴影等遮挡物导致道路连通性判断困难，是一个被忽视的瓶颈。

**核心矛盾**：全局方法中分割阶段和连接预测阶段的解耦设计导致训练-推理不一致；数据层面，现有图标注数据集（City-Scale 仅180张图像、SpaceNet 仅 $400 \times 400$ 小图）规模太小且场景单一，造成模型泛化能力不足和评估不可靠。

**本文目标**：(1) 构建一个数据量大、场景多样的全球道路图提取基准；(2) 解决训练-推理不匹配问题；(3) 缓解遮挡场景下的道路断裂。

**切入角度**：作者发现 SAM-Road 的分类器训练时使用 GT 节点但推理时使用预测节点，如果能让训练时的节点分布更接近推理时的节点分布，就能提升一致性。对于遮挡问题，作者假设如果一条直线道路两端存在道路线，那么中间被遮挡的部分也大概率存在道路。

**核心 idea**：通过节点引导重采样（node-guided resampling）让训练阶段的节点从预测掩码中获取最高概率坐标来替代 GT 节点坐标，同时利用延长线策略为分类器提供额外上下文信息以识别遮挡场景。

## 方法详解

### 整体框架

SAM-Road++ 以 SAM 编码器-解码器为骨干，输入卫星图像，首先生成道路分割掩码和关键点掩码。训练阶段通过节点引导重采样从 GT 与预测掩码中获取节点对，推理阶段通过 NMS 从掩码中选择节点。连接分类器基于节点特征和延长线信息预测节点间是否存在道路。总损失为 $\mathcal{L}_{mask}$（道路分割）和 $\mathcal{L}_{topo}$（拓扑连接）的二元交叉熵之和。

### 关键设计

1. **节点引导重采样（Node-guided Resampling）**:

    - 功能：对齐训练和推理阶段分类器的输入分布
    - 核心思路：训练时先从 GT 采样 $N$ 个源节点，确定每个源节点距离 $R$ 内的目标节点及连接关系。然后保持源节点不变，以每个目标节点为中心、半径 $r$ 范围内在预测掩码上找到概率最大的点作为新目标节点。这样重采样节点既保留了 GT 的连接信息，位置又更接近推理时 NMS 选出的节点。为保证多样性，倾向于采样度属性更稀有的节点。
    - 设计动机：直接在训练中使用推理流程（NMS）无法获得监督用的连接信息，因此采用折中策略——保留 GT 的拓扑信息但让坐标向预测掩码对齐，同时充分利用第一阶段的分割经验。

2. **延长线策略（Extended-line Strategy）**:

    - 功能：为连接分类器提供遮挡场景下的额外上下文信息
    - 核心思路：对于需要判断连通性的一对节点，除了提取它们各自的节点特征外，还在两个节点连线的两端延长线上均匀采样 $n=15$ 次获取掩码值，在两节点之间的连线上均匀采样 $m=20$ 次。延长线长度设为 8 像素、宽度 3 像素以模拟道路。这些额外采样值作为分类器的附加输入。
    - 设计动机：基于道路的延展性假设——如果两个相邻节点对应的延长方向上都明确存在道路，即使中间被树木或建筑阴影遮挡，也很可能存在连接道路。这让模型能"看到"被遮挡区域前后的道路信息。

3. **Global-Scale 数据集**:

    - 功能：提供大规模、多样化的全球道路图提取基准
    - 核心思路：从 Google Earth 手动选取全球各大洲（除南极洲）的城市、乡村、山区等多种地形的经纬度坐标，从 Google Static Map API 获取 $2048 \times 2048$、空间分辨率 1m/pixel 的卫星图像，从 OpenStreetMap 获取对应道路图标注。包含 3468 张图像（训练 2375、验证 339、域内测试 624、域外测试 130 张来自未见过的城市）。
    - 设计动机：现有数据集仅覆盖城市场景且规模有限，无法支撑鲁棒的模型评估和泛化能力训练。域外测试集设计可评估模型在未见区域的预测能力。

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{mask} + \mathcal{L}_{topo}$，均为二元交叉熵损失。$\mathcal{L}_{mask}$ 监督道路分割，$\mathcal{L}_{topo}$ 监督节点间拓扑连接。使用 Adam 优化器，学习率 0.001，Global-Scale 训练 150 个 epoch，其余数据集训练至验证指标稳定。

## 实验关键数据

### 主实验

| 数据集 | 方法 | F1 | Precision | Recall | APLS |
|--------|------|-----|-----------|--------|------|
| City-Scale | SAM-Road | 77.23 | 90.47 | 67.69 | 68.37 |
| City-Scale | **SAM-Road++** | **80.01** | 88.39 | **73.39** | 68.34 |
| City-Scale | SAM-Road++* | **80.66** | 89.08 | **74.07** | **69.55** |
| SpaceNet | SAM-Road | 80.52 | 93.03 | 70.97 | 71.64 |
| SpaceNet | **SAM-Road++** | **81.57** | **93.68** | 72.23 | **73.44** |
| Global-Scale (ID) | SAM-Road | 59.80 | **91.93** | 45.64 | 59.08 |
| Global-Scale (ID) | **SAM-Road++** | **62.33** | 88.95 | **49.27** | **62.19** |
| Global-Scale (OOD) | SAM-Road | 46.64 | 84.54 | 33.81 | 40.51 |
| Global-Scale (OOD) | **SAM-Road++** | **48.34** | 82.21 | **36.04** | **43.17** |

*表示在 Global-Scale 上预训练后微调。

### 消融实验

| Extended-line | Node-guided Resampling | APLS | F1 |
|:---:|:---:|------|------|
| ✗ | ✗ | 71.64 | 80.52 |
| ✗ | ✓ | 71.90 | 81.77 |
| ✓ | ✗ | 73.22 | 80.89 |
| ✓ | ✓ | **73.44** | **81.57** |

### 关键发现

- 节点引导重采样对 F1 指标贡献更大（+1.25），说明它有效改善了拓扑预测的整体准确性；延长线策略对 APLS 贡献更大（+1.58），说明它帮助模型预测更准确的道路路径长度。
- 在 Global-Scale 数据集上预训练后在 City-Scale 和 SpaceNet 上微调可进一步提升性能，验证了大规模数据集的价值。
- 所有模型在 Global-Scale 上的性能均低于小数据集，证明该数据集确实更具挑战性。
- 域外测试集上性能下降明显，但 SAM-Road++ 仍显著领先其他方法，泛化能力更强。

## 亮点与洞察

- **训练-推理对齐思路**的通用性：节点引导重采样本质上是在训练阶段模拟推理环境，这种思路可迁移到任何存在训练-推理不匹配的两阶段pipeline（如检测+分类框架）。
- **延长线策略的简洁高效**：不需要引入额外模型或复杂架构，仅通过采样连线延长方向上的掩码信息就能缓解遮挡问题。这种利用道路几何先验（延展性）的设计思路值得借鉴。
- 域外测试集的设计使评估更贴近真实应用场景，为遥感领域建立了更可靠的基准。

## 局限与展望

- Precision 在某些场景下略低于原始 SAM-Road，因为重采样节点可能偏离 GT 导致预测出 GT 中不存在的道路。
- City-Scale 测试集仅 27 张图像，单张图像的异常表现对指标影响过大，评估不够稳定。
- 数据集标注依赖 OpenStreetMap，在经济欠发达地区标注不完整。
- 未来可探索更自适应的重采样半径 $r$ 和延长线长度，以及将方法扩展到更细粒度的道路属性（车道数、道路类型）提取。

## 相关工作与启发

- **vs SAM-Road**: SAM-Road 首次将 SAM 引入道路图提取领域实现全局预测，但两阶段解耦导致训练-推理不匹配。SAM-Road++ 通过重采样策略耦合了两个阶段，同时保持了推理效率不受影响。
- **vs RNGDet++**: 迭代式方法虽然精度高但速度慢且误差累积，SAM-Road++ 作为全局方法在 F1 上全面超越且效率更高。
- **vs Sat2Graph**: 早期全局方法依赖图编码张量和复杂后处理，SAM-Road++ 简化了流程并取得更好效果。

## 评分

- 新颖性: ⭐⭐⭐⭐ 训练-推理对齐和延长线策略虽然思路清晰但并非颠覆性创新，数据集贡献是亮点
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、域内外测试、消融实验完整
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图示丰富
- 价值: ⭐⭐⭐⭐ 数据集和方法均有实用价值，对遥感社区贡献明显

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LiSu: A Dataset and Method for LiDAR Surface Normal Estimation](lisu_a_dataset_and_method_for_lidar_surface_normal_estimation.md)
- [\[CVPR 2025\] ClimbingCap: Multi-Modal Dataset and Method for Rock Climbing in World Coordinate](climbingcap_multi-modal_dataset_and_method_for_rock_climbing_in_world_.md)
- [\[CVPR 2026\] SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving](../../CVPR2026/autonomous_driving/searchad_large-scale_rare_image_retrieval_dataset_for_autonomous_driving.md)
- [\[CVPR 2025\] GLane3D: Detecting Lanes with Graph of 3D Keypoints](glane3d_detecting_lanes_with_graph_of_3d_keypoints.md)
- [\[NeurIPS 2025\] ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset](../../NeurIPS2025/autonomous_driving/chronograph_a_real-world_graph-based_multivariate_time_series_dataset.md)

</div>

<!-- RELATED:END -->
