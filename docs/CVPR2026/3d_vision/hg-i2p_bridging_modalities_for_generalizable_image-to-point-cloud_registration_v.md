---
title: >-
  [论文解读] Hg-I2P: Bridging Modalities for Generalizable Image-to-Point-Cloud Registration via Heterogeneous Graphs
description: >-
  [CVPR 2026][3D视觉][图像-点云配准] Hg-I2P 引入异构图（Heterogeneous Graph）来统一建模 2D 图像区域和 3D 点云区域之间的关系，通过多路径邻接关系挖掘学习跨模态边、基于异构边的特征适配和基于图的投影一致性剪枝，在六个室内外跨域基准上实现了最优的泛化能力和精度。
tags:
  - CVPR 2026
  - 3D视觉
  - 图像-点云配准
  - 异构图
  - 跨模态特征适配
  - 对应关系剪枝
  - 跨域泛化
---

# Hg-I2P: Bridging Modalities for Generalizable Image-to-Point-Cloud Registration via Heterogeneous Graphs

**会议**: CVPR 2026  
**arXiv**: [2603.27969](https://arxiv.org/abs/2603.27969)  
**代码**: https://github.com/anpei96/hg-i2p-demo  
**领域**: 3D视觉  
**关键词**: 图像-点云配准, 异构图, 跨模态特征适配, 对应关系剪枝, 跨域泛化

## 一句话总结

Hg-I2P 引入异构图（Heterogeneous Graph）来统一建模 2D 图像区域和 3D 点云区域之间的关系，通过多路径邻接关系挖掘学习跨模态边、基于异构边的特征适配和基于图的投影一致性剪枝，在六个室内外跨域基准上实现了最优的泛化能力和精度。

## 研究背景与动机

1. **领域现状**：图像到点云（I2P）配准旨在建立 2D 像素与 3D 点之间的对应关系，是视觉定位、导航和 3D 重建的基石。近年来基于学习的方法通过改进骨干网络、匹配策略和损失函数取得了进展，如 MATR 采用 coarse-to-fine 匹配、CoFiI2P 引入补丁级匹配。

2. **现有痛点**：现有方法在训练域内表现良好，但在未见过的场景中性能严重下降。核心原因是 2D 图像特征（基于外观）和 3D 点云特征（基于几何）分布差异巨大——即使是正确的对应关系，特征相似度也可能很低，神经网络难以区分正确匹配。

3. **核心矛盾**：现有改进要么只做特征精炼（缺乏显式跨模态推理），要么只做对应关系剪枝（依赖深度预测或手工启发式），两者割裂处理无法系统性解决泛化问题。虽然视觉基础模型（SAM、DepthAnything 等）能帮助桥接模态差距，但缺乏统一框架来同时利用特征精炼和对应关系剪枝。

4. **本文目标** (1) 如何构建一个统一结构，同时支持跨模态特征精炼和对应关系剪枝？(2) 如何有效学习 2D-3D 区域之间的跨模态映射关系？(3) 如何利用图结构中的一致性信息过滤错误匹配？

5. **切入角度**：用 2D/3D SAM 将图像和点云分割为区域，构建异构图来建模区域间的关系。图的异构边（I2P 边）定义了一种 2D-3D 区域映射，既能指导特征精炼（沿边进行跨模态消息传递），又能支持对应关系剪枝（通过图内的投影一致性检查）。

6. **核心 idea**：用异构图统一建模 2D-3D 区域关系，在同一图结构上同时进行跨模态特征适配和对应关系剪枝，实现强泛化的 I2P 配准。

## 方法详解

### 整体框架

输入一张 RGB 图像和一个彩色点云，先用 2D/3D SAM 分割为 $M$ 个 2D 区域和 $N$ 个 3D 区域，构成异构图 $\mathcal{G}_H = (\mathcal{V}_H, \mathcal{E}_H)$。然后通过三个核心模块：(1) **MP-mining** 学习异构边 $\mathcal{E}_{I2P}$；(2) **HE-adapting** 沿异构边进行跨模态特征消息传递以精炼特征；(3) **HC-pruning** 基于图的投影一致性剪枝错误对应关系。最终用精炼特征进行特征级匹配得到 2D-3D 对应关系，经 RANSAC-PnP 估计位姿。

### 关键设计

1. **异构图定义与构建**:

    - 功能：为 2D 图像和 3D 点云建立统一的关系表示结构
    - 核心思路：顶点 $\mathcal{V}_H = \mathcal{V}_I \cup \mathcal{V}_P$ 分别对应 $M$ 个 2D 区域和 $N$ 个 3D 区域，每个顶点有一个 $c$ 维特征向量（区域内特征的平均池化）。边分三类：(a) 同构 2D-2D 边 $\mathbf{E}_{I2I}$：基于 2D 区域特征距离 $e^{-\alpha\|\mathbf{v}_i^I - \mathbf{v}_j^I\|_2^2}$；(b) 同构 3D-3D 边 $\mathbf{E}_{P2P}$：基于 3D 区域特征距离；(c) 异构 2D-3D 边 $\mathbf{E}_{I2P}$：理论上应由 GT 位姿下 3D 区域投影到 2D 的 IoU 定义，但推理时 GT 位姿不可用，所以需要被学习
    - 设计动机：不同于以往方法孤立处理 2D 和 3D 特征，异构图提供了一个统一框架来联合建模模态内和模态间关系，使得后续的特征精炼和对应关系剪枝可以在同一结构内完成

2. **MP-mining（多路径邻接关系挖掘）**:

    - 功能：在推理时（没有 GT 位姿）学习异构边 $\mathcal{E}_{I2P}$
    - 核心思路：利用已知的同构边 $\mathbf{E}_{I2I}$、$\mathbf{E}_{P2P}$，通过三条路径挖掘 2D-3D 邻接关系：$\mathbf{E}_{I2P}^1 = \mathbf{E}_{I2I}\tilde{\mathbf{E}}_{I2P}$（经 2D 邻居中转），$\mathbf{E}_{I2P}^2 = \tilde{\mathbf{E}}_{I2P}\mathbf{E}_{P2P}$（经 3D 邻居中转），$\mathbf{E}_{I2P}^3 = \mathbf{E}_{I2I}\tilde{\mathbf{E}}_{I2P}\mathbf{E}_{P2P}$（经两步中转）。将三个矩阵拼接后通过 attention 层预测最终 $\hat{\mathbf{E}}_{I2P}$
    - 设计动机：从贝叶斯推理角度看，多路径邻接关系捕捉了 2D-3D 区域之间的间接因果关系——即使直接的初始匹配 $\tilde{\mathbf{E}}_{I2P}$ 不准确，通过同类区域的邻接传递可以修正估计

3. **HE-adapting（异构边引导的特征适配）**:

    - 功能：利用学到的异构边进行跨模态消息传递，精炼 2D 和 3D 特征以增强跨模态匹配能力
    - 核心思路：分两步——(a) **消息生成**：对每个 2D 区域 $\mathcal{I}_i$，从其通过 $\mathcal{E}_{I2P}$ 连接的 3D 区域邻居中加权聚合特征得到跨模态消息 $\bar{\mathbf{m}}_i^I$，通过交叉注意力学习 2D 区域特征与跨模态消息的相关性；(b) **消息交互**：将原始区域特征与消息特征沿通道拼接，通过自注意力融合后按比例 $\beta$ 与原始特征加权组合得到适配后特征。3D 端对称执行相同操作
    - 设计动机：HE-adapting 实现了基于图结构的跨模态信息流动——2D 特征"看到"了匹配的 3D 几何信息，3D 特征"看到"了匹配的 2D 外观信息，从而缩小了模态差距并提高了跨域泛化能力

4. **HC-pruning（基于图的投影一致性剪枝）**:

    - 功能：过滤特征匹配产生的错误对应关系
    - 核心思路：先用 RANSAC-PnP 从精炼特征的匹配结果估计初始位姿 $\tilde{\mathbf{T}}$，然后用两个互补的剪枝标准：(a) 基于 $\mathcal{E}_{I2P}$ 邻接和重投影距离 $\delta_{\text{rej}}$；(b) 基于图投影导出的相对位置向量的余弦相似度。满足至少一个标准的对应关系被保留为内点
    - 设计动机：双标准设计能有效处理位姿估计噪声或边学习不完美导致的假匹配——两个标准互补，一个基于局部距离约束，一个基于全局方向一致性

### 损失函数

$L_{\text{Hg-I2P}} = L_{\text{corr}} + \lambda_1 \|\hat{\mathbf{E}}_{I2P}[\text{mask}] - \mathbf{E}_{I2P}[\text{mask}]\|_2^2$

其中 $L_{\text{corr}}$ 是标准的 circle loss 对应关系损失，第二项监督异构边的学习（只在有效的非零位置上计算）。

## 实验关键数据

### 主实验

在 7-Scenes 数据集上的跨场景 I2P 配准（从一个场景训练，在其他场景测试）：

| 方法 | IR (C→) AVG | RR (C→) AVG | IR (K→) AVG | RR (K→) AVG |
|------|------------|------------|------------|------------|
| MATR | 0.387 | 0.478 | 0.537 | 0.706 |
| Top-I2P | 0.433 | 0.628 | 0.596 | 0.785 |
| MinCD | 0.445 | 0.592 | 0.568 | 0.814 |
| Hg-I2P† (无 HC-pruning) | 0.472 | 0.642 | 0.618 | 0.802 |
| **Hg-I2P (Ours)** | **0.581** | **0.667** | **0.688** | **0.853** |

在 RGBD-V2、ScanNet 等跨数据集设置上也有显著提升。

### 消融实验

| 配置 | IR AVG | RR AVG | 说明 |
|------|--------|--------|------|
| Hg-I2P (完整) | 0.581 | 0.667 | 完整模型 |
| Hg-I2P† (w/o HC-pruning) | 0.472 | 0.642 | 去掉 HC-pruning，IR 下降 18.8% |
| 基线 MATR | 0.387 | 0.478 | 基线方法 |

HC-pruning 的加入带来了约 23% 的 IR 提升和 4% 的 RR 提升，说明图基的对应关系剪枝对精确配准至关重要。

### 关键发现

- 异构图的统一框架显著优于仅做特征精炼或仅做对应关系剪枝的方法
- 在跨域（训练和测试来自不同数据集）设置下优势更为明显，验证了方法的泛化能力
- MP-mining 学到的异构边对后续的 HE-adapting 和 HC-pruning 都至关重要——准确的 2D-3D 区域映射是整个系统的基础
- 与同样使用 SAM 的先前工作相比（如 An et al.），Hg-I2P 通过图结构系统性地利用了边信息和投影约束

## 亮点与洞察

- **异构图作为统一框架**：将特征精炼和对应关系剪枝这两个传统上独立处理的问题统一到一个图结构中，优雅地避免了碎片化处理的弊端。图的边同时服务于特征传播（HE-adapting）和几何验证（HC-pruning），一举两得
- **多路径邻接关系挖掘的贝叶斯解释**：将同构边看作条件概率，多路径乘积对应贝叶斯推理的边际化，这个视角为图上的关系学习提供了优雅的理论基础
- **粗到细的跨模态消息传递**：消息先在区域级别（粗）聚合，再在像素/点级别（细）与原始特征交互，兼顾了效率和精度

## 局限与展望

- 依赖 2D/3D SAM 的分割质量——如果 SAM 在某些场景（如纹理缺失区域）分割不佳，异构图的构建质量会下降
- HC-pruning 需要先用 RANSAC-PnP 估计初始位姿，如果初始匹配质量太差，可能导致错误的位姿估计进而影响后续剪枝
- 图的顶点数量（$M + N$）取决于 SAM 的分割粒度，可能需要针对不同场景调整
- 文中未报告运行时间，SAM + 图构建 + 消息传递的总开销是否适合实时应用值得关注

## 相关工作与启发

- **vs MinCD (Bie et al.)**: MinCD 用 DepthAnything 将 I2P 转化为 3D-3D 配准，但预测深度缺乏真实尺度需要额外对齐。Hg-I2P 直接在 2D-3D 空间操作，避免了深度尺度不准确的问题
- **vs An et al. (2024)**: 同样使用 SAM，但只用于对齐物体对提取对应关系。Hg-I2P 更进一步地定义了异构图结构，系统性利用边信息进行特征适配和剪枝
- **vs MATR**: MATR 用 coarse-to-fine 匹配但缺乏跨模态推理，Hg-I2P 通过图消息传递显式引入了跨模态信息流

## 评分

- 新颖性: ⭐⭐⭐⭐ 异构图用于 I2P 配准是新的视角，MP-mining 和 HE-adapting 设计精巧
- 实验充分度: ⭐⭐⭐⭐ 六个数据集的跨域实验覆盖全面
- 写作质量: ⭐⭐⭐⭐ 图表清晰，公式推导详尽，但论文偏长
- 价值: ⭐⭐⭐⭐ 为 I2P 配准的泛化问题提供了系统性解决方案

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration](cmhanet_a_crossmodal_hybrid_attention_network_for.md)
- [\[ICCV 2025\] CA-I2P: Channel-Adaptive Registration Network with Global Optimal Selection](../../ICCV2025/3d_vision/ca-i2p_channel-adaptive_registration_network_with_global_optimal_selection.md)
- [\[ICCV 2025\] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](../../ICCV2025/3d_vision/buffer-x_towards_zero-shot_point_cloud_registration_in_diverse_scenes.md)
- [\[ICCV 2025\] TurboReg: TurboClique for Robust and Efficient Point Cloud Registration](../../ICCV2025/3d_vision/turboreg_turboclique_for_robust_and_efficient_point_cloud_registration.md)
- [\[CVPR 2025\] ColabSfM: Collaborative Structure-from-Motion by Point Cloud Registration](../../CVPR2025/3d_vision/colabsfm_collaborative_structure-from-motion_by_point_cloud_registration.md)

<!-- RELATED:END -->
