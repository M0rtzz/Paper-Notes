---
title: >-
  [论文解读] UniM2AE: Multi-modal Masked Autoencoders with Unified 3D Representation for 3D Perception in Autonomous Driving
description: >-
  [ECCV 2024][自动驾驶][多模态掩码自编码器] 本文提出 UniM2AE，一个多模态自监督预训练框架，通过将图像和 LiDAR 点云特征统一投影到 3D 体素空间（比 BEV 多保留高度维度），并设计 Multi-modal 3D Interactive Module（MMIM）进行高效跨模态交互，实现了比独立预训练和简单拼接的前序方法更强的 3D 检测（+1.2% NDS）和 BEV 分割（+6.5% mIoU）提升。
tags:
  - ECCV 2024
  - 自动驾驶
  - 多模态掩码自编码器
  - 自监督预训练
  - 3D体素空间
  - LiDAR-相机融合
  - BEV感知
---

# UniM2AE: Multi-modal Masked Autoencoders with Unified 3D Representation for 3D Perception in Autonomous Driving

**会议**: ECCV 2024  
**arXiv**: [2308.10421](https://arxiv.org/abs/2308.10421)  
**代码**: https://github.com/hollow-503/UniM2AE  
**领域**: 自动驾驶  
**关键词**: 多模态掩码自编码器, 自监督预训练, 3D体素空间, LiDAR-相机融合, BEV感知

## 一句话总结
本文提出 UniM2AE，一个多模态自监督预训练框架，通过将图像和 LiDAR 点云特征统一投影到 3D 体素空间（比 BEV 多保留高度维度），并设计 Multi-modal 3D Interactive Module（MMIM）进行高效跨模态交互，实现了比独立预训练和简单拼接的前序方法更强的 3D 检测（+1.2% NDS）和 BEV 分割（+6.5% mIoU）提升。

## 研究背景与动机

**领域现状**：Masked Autoencoders（MAE）在 2D 视觉和 3D 感知任务中展现出强大的自监督预训练能力。在自动驾驶场景中，多传感器融合（LiDAR + Camera）是获取丰富环境感知的标准做法。已有方法如 GreenMIM（图像）和 Voxel-MAE（LiDAR）分别对单模态做 MAE 预训练，但缺乏对多模态联合预训练的有效解决方案。

**现有痛点**：多模态 MAE 面临的核心挑战在于两种模态之间的巨大差异：图像提供稠密的 2D 语义信息，LiDAR 提供稀疏的 3D 几何信息，两者在数据密度、空间维度和信息类型上截然不同。现有尝试（如 PiMAE）将 LiDAR 投影到图像平面进行对齐，但这会引入严重的几何畸变——物理距离远的点在像素坐标上可能相邻，且大量 LiDAR 点超出相机视野导致无法投影。反过来，从相机到 LiDAR 的投影也因 LiDAR 稀疏性导致大量相机特征丢失。

**核心矛盾**：多模态特征融合需要一个共同的表示空间，但现有的表示空间（图像平面或 LiDAR 坐标系）都存在固有缺陷，要么丢失 3D 几何信息，要么丢失稠密语义信息。BEV 表示是一个折中，但它压缩了高度维度，对高低不同的物体（如交通灯 vs 车辆）无法精确表示。

**本文目标** （1）如何设计一个统一的多模态表示空间，既保留图像语义又保留 LiDAR 几何？（2）如何在这个空间中高效实现跨模态交互？（3）预训练得到的特征能否有效迁移到多种下游任务？

**切入角度**：作者提出将 BEV 空间沿高度轴（z轴）扩展为 3D 体素空间。这个扩展看似简单，但带来两个关键优势：（1）保留了物体的高度信息，避免了 BEV 的信息压缩损失；（2）3D 体素空间天然与两种模态对齐——LiDAR 点直接通过坐标映射，图像特征通过已知的相机内外参做空间交叉注意力投影。

**核心 idea**：将多模态特征投影到扩展了高度维的 3D 体素空间进行统一表示和交互，实现保留信息更完整的多模态 MAE 预训练。

## 方法详解

### 整体框架
UniM2AE 采用双分支编码器-单融合模块-双分支解码器的对称架构。LiDAR 分支将点云体素化后用 SST 编码器提取特征 $F_V$；Camera 分支将多视角图像分 patch 后用 Swin-T 编码器提取特征 $F_I$。两分支分别随机遮掩输入（LiDAR 70%、Camera 75%）。编码后的特征通过 Token-Volume 投影进入统一的 3D 体素空间，经 MMIM 模块交互融合后，通过 Volume-Token 逆投影回各自模态空间，最终由模态特定解码器重建原始输入。

### 关键设计

1. **统一 3D 体素表示空间（Unified 3D Volume Space）**:

    - 功能：提供跨模态特征对齐和融合的统一空间，保留完整的空间信息
    - 核心思路：定义感知范围为 x/y 轴 [-50m, 50m]、z 轴 [-5m, 3m]，将空间离散化为体素网格。对于 LiDAR，将体素特征直接根据自车坐标系中的位置映射到 3D 体素空间，得到 $F_V^{vol}$。对于图像，使用 2D-3D 空间交叉注意力（Spatial Cross-Attention），通过相机内外参将 3D 体素查询点投影到 2D 图像视图上，采样对应的图像特征，公式为 $F_I^{vol} = \frac{1}{|\mathcal{V}_{hit}|} \sum_{i \in \mathcal{V}_{hit}} \sum_j \text{DeformAttn}(Q_{vol}, \mathcal{P}(p,i,j), F_I^i)$。关键在于 z 轴的扩展——默认使用 2 层高度分辨率，在精度和效率之间取得平衡
    - 设计动机：与 BEV 相比，3D 体素空间保留了高度信息，使得交通灯、行人头部等不同高度的物体能被精确表示。更重要的是，3D 体素可以直接反投影回原始模态用于重建——这是 MAE 框架的基本要求

2. **多模态 3D 交互模块（Multi-modal 3D Interaction Module, MMIM）**:

    - 功能：在统一的 3D 体素空间中高效完成跨模态特征交互
    - 核心思路：MMIM 由 $L=3$ 个堆叠的 3D 可变形自注意力块组成。首先将 LiDAR 体素特征 $F_V^{vol}$ 和图像体素特征 $F_I^{vol}$ 沿通道维度拼接，reshape 为 $F_c^{vol} \in \mathbb{R}^{HWZ \times 2C}$。然后送入 3D 可变形自注意力模块进行交互：$F_c' = \sum_m W_m \sum_k A_{mk} \cdot W_m' F_c^{vol}(p_{vol} + \Delta p_k^{vol})$，其中 $\Delta p_k^{vol}$ 是学习到的采样偏移，$A_{mk}$ 是注意力权重。交互后将 $F_c'$ 沿通道维度拆分，得到融合后的模态特定 3D 特征 $(F_V', F_I')$
    - 设计动机：使用可变形自注意力而非标准自注意力，是因为 3D 体素空间中的 token 序列长度 $H \times W \times Z$ 可能很大，标准自注意力计算成本过高。可变形注意力自适应地聚焦于最显著的空间位置，且计算复杂度仅随采样点数 $K$ 线性增长。此外，MMIM 的预训练权重可以直接迁移到下游融合任务中

3. **双模态重建目标（Dual-modal Reconstruction Targets）**:

    - 功能：为 MAE 预训练提供多粒度的监督信号
    - 核心思路：融合后的特征通过 Volume-Token 逆投影回各自模态空间：LiDAR 分支直接在自车坐标中采样体素特征 $F_V^{sp}$；图像分支通过相机投影函数 $T_{proj}$ 将 3D 体素特征映射到 2D 像素坐标，得到 $F_I^{sp}$。三个重建目标分别为：（a）LiDAR 点数重建——预测每个体素内的点数，用 Chamfer Distance 监督 $\mathcal{L}_c$；（b）体素占用预测——预测体素是否为空，用 BCE 损失 $\mathcal{L}_{occ}$；（c）图像像素重建——预测被遮掩区域的原始像素，用 MSE 损失 $\mathcal{L}_{img}$
    - 设计动机：多重建目标从不同角度促进跨模态特征学习。LiDAR 重建迫使图像特征包含几何信息，图像重建迫使 LiDAR 特征包含语义信息，从而实现真正的跨模态增强

### 损失函数 / 训练策略
总预训练损失 $\mathcal{L} = \mathcal{L}_{voxel} + \mathcal{L}_{img} = (\mathcal{L}_c + \mathcal{L}_{occ}) + \mathcal{L}_{MSE}$。预训练 200 epochs，8 GPU，基础学习率 2.5e-5。下游微调时移除解码器，仅使用编码器和可选的 MMIM 模块。

## 实验关键数据

### 主实验

| 方法 | 模态 | NDS↑ | mAP↑ | 提升(vs Random) |
|--------|------|------|----------|------|
| BEVFusion-SST (Random) | C+L | 67.4 | 63.6 | - |
| MIM + Voxel-MAE | C+L | 67.7 | 63.7 | +0.3/+0.1 |
| PiMAE | C+L | 67.9 | 63.9 | +0.5/+0.3 |
| **UniM2AE** | C+L | **68.1** | **64.3** | **+0.7/+0.7** |
| BEVFusion-SST + MMIM† | C+L | **72.7** | **69.7** | -- |

### 消融实验

| 配置 | mAP | NDS | 说明 |
|------|---------|------|------|
| 从头训练 (无预训练) | 59.0 | 61.8 | 基线 |
| 仅 Camera 预训练 | 59.7 | 62.6 | 图像先验有限 |
| 仅 LiDAR 预训练 | 60.1 | 62.6 | 几何先验有限 |
| Camera + LiDAR 独立预训练 | 60.7 | 63.1 | 简单合并两个预训练结果 |
| UniM2AE (BEV 交互) | 62.0 | 64.3 | BEV 压缩了高度信息 |
| UniM2AE (3D Volume 交互) | **62.8** | **65.2** | 3D 体素保留高度信息 |

### 关键发现
- **联合预训练 vs 独立预训练**：UniM2AE 比独立预训练+合并的方案高 2.1 NDS，说明统一空间中的联合学习发挥了互补优势
- **3D 体素 vs BEV**：用 3D 体素空间替代 BEV 带来 0.9 NDS 的提升，高度信息对精确位姿和检测很重要
- **z 轴层数**：2 层高度分辨率即可达到最佳平衡（更多层增加计算但提升微弱，因为道路场景中物体高度分布有限）
- **掩码比例**：LiDAR 70% + Camera 75% 是最优组合；过高或过低都会降低性能
- **数据效率**：在仅 20% 标注数据时，UniM2AE 相对从头训练的提升最显著（+4.4 mAP），验证了自监督预训练在标注稀缺时的价值
- BEV 分割任务中带 MMIM 的 UniM2AE 比 X-Align 高 2.1 mIoU，体现了跨任务的强迁移能力

## 亮点与洞察
- **3D 体素空间作为统一表示**是一个简单但深刻的设计：它解决了 BEV 丢失高度信息的根本问题，同时天然支持双向投影（编码和解码都需要），使得 MAE 框架可以自然运作。这个设计可以推广到任何多模态 3D 理解任务
- **MMIM 的可迁移性**是一个实用亮点——预训练 MMIM 可以直接插入下游融合模型，带来显著提升（+5.1 NDS），省去了融合模块从头训练的成本
- 实验中 PiMAE 在 20% 数据时仅提升 1.1 NDS（因为图像平面投影丢失了深度信息），而 UniM2AE 提升 4.4 NDS，凸显了表示空间选择的重要性

## 局限与展望
- 使用随机掩码策略，未考虑两种模态输入之间的关联性——互补掩码（遮掩一个模态的某区域、保留另一个模态的对应区域）可能迫使跨模态学习更充分
- 忽略了时序连续性——nuScenes 中相邻帧高度相似，当前方法在冗余数据上重复预训练，降低了效率。引入时序掩码或帧间对比学习可能是改进方向
- 体素分辨率固定（0.15m），对于小物体（行人、自行车）的表示精度可能不足
- 仅在 nuScenes 上验证，未在 Waymo 或 Argoverse 等更大规模数据集上测试
- 预训练成本较高（200 epochs，8 GPU），与现代大规模预训练相比规模仍然有限

## 相关工作与启发
- **vs Voxel-MAE**: 仅对 LiDAR 做自监督预训练，忽略了图像信息；UniM2AE 通过统一空间将两者融合
- **vs GreenMIM**: 仅对图像做 MAE 预训练，且工作在 2D 空间；UniM2AE 将其扩展到 3D 多模态
- **vs PiMAE**: 在图像平面上对齐 LiDAR 和图像，引入了几何畸变；UniM2AE 在 3D 体素空间对齐，信息损失更小
- **vs BEVFusion**: BEVFusion 在 BEV 空间融合，UniM2AE 用体素空间保留更多信息；两者互补——UniM2AE 的预训练权重可以初始化 BEVFusion

## 评分
- 新颖性: ⭐⭐⭐⭐ 3D 体素统一空间 + 多模态 MAE 的组合是首次在自动驾驶中实现
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据比例、多下游任务、充分消融
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述系统
- 价值: ⭐⭐⭐⭐ 为多模态自监督预训练提供了有效的统一框架

<!-- RELATED:START -->

## 相关论文

- [OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](occgen_generative_multimodal_3d_occupancy_prediction_for_aut.md)
- [GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection](graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)
- [4D Contrastive Superflows are Dense 3D Representation Learners](4d_contrastive_superflows_are_dense_3d_representation_learners.md)
- [OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)
- [Fully Sparse 3D Occupancy Prediction](fully_sparse_3d_occupancy_prediction.md)

<!-- RELATED:END -->
