---
title: >-
  [论文解读] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving
description: >-
  [CVPR 2026][自动驾驶][3D占用预测] Dr.Occ 提出深度引导与区域引导的统一 3D 占用预测框架，通过 D2-VFormer 利用 MoGe-2 的高质量深度先验实现精确的 2D→3D 几何映射，并通过 R/R2-EFormer 借鉴 MoE/MoR 思想自适应分配区域专家处理空间语义各向异性，在 BEVDet4D 基线上提升 7.43% mIoU。
tags:
  - CVPR 2026
  - 自动驾驶
  - 3D占用预测
  - 深度引导
  - 区域专家
  - MoE
  - 视觉感知
---

# Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving

**会议**: CVPR 2026  
**arXiv**: [2603.01007](https://arxiv.org/abs/2603.01007)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 3D占用预测, 深度引导, 区域专家, MoE, 视觉感知

## 一句话总结

Dr.Occ 提出深度引导与区域引导的统一 3D 占用预测框架，通过 D2-VFormer 利用 MoGe-2 的高质量深度先验实现精确的 2D→3D 几何映射，并通过 R/R2-EFormer 借鉴 MoE/MoR 思想自适应分配区域专家处理空间语义各向异性，在 BEVDet4D 基线上提升 7.43% mIoU。

## 研究背景与动机

3D 语义占用预测是自动驾驶感知的核心任务，需要从环视相机图像重建精确的体素级语义场景。现有方法面临两大挑战：

**几何失配(Geometric Misalignment)**：现有视图变换方法（前向/后向/双向投影）依赖低分辨率、噪声深度估计进行 2D→3D 特征映射，导致投影误差和特征错位

**空间类别严重不平衡(Spatial Class Imbalance)**：不同语义类别在 3D 空间中展现强烈的位置偏好——行人集中在路边，车辆聚集在路中心，建筑和植被出现在较高位置。这种空间各向异性使得统一模型难以均衡学习

**关键观察**：
- 随着大视觉模型（如 MoGe-2）的发展，高质量像素级深度估计已可用，但简单拼接深度图或转换为伪点云效果不佳——直接用于前向投影反而降低性能
- 约 90% 的占用体素是空的，直接拟合所有体素效率低下。因此利用深度生成几何感知占用掩码来聚焦非空体素更为高效

## 方法详解

### 整体框架

Dr.Occ 在典型的占用预测管线基础上做了两个关键改进：

1. **D2-VFormer（Depth-guided 2D-to-3D View Transformer）**：利用 MoGe-2 深度先验构建几何感知占用掩码，引导高效精确的体素特征构建
2. **R/R2-EFormer（Region-guided Expert Transformer）**：基于 MoE/MoR 思想，根据距离和高度划分空间区域，自适应分配专家处理不同区域的语义分布

### 关键设计

#### 1. 深度引导双投影视图变换器 (D2-VFormer)

**深度先验获取**：使用 MoGe-2 同时提取深度特征 $\mathbf{F}^{(D)}$ 和深度图 $\{\mathbf{D}_i\}$。通过相机投影将深度转为伪点云 $\mathcal{P}$，然后体素化生成几何感知占用掩码 $M(\mathbf{v})$：

$$M(\mathbf{v}) = \begin{cases} 1, & \mathbf{v} \in \text{Voxelize}(\mathcal{P}, r) \\ 0, & \text{otherwise} \end{cases}$$

**三阶段渐进细化**：

**Stage 1：前向投影+下采样**。沿用 BEVStereo 将 2D 特征提升到体素空间（约 30% 占用），然后将体素特征和几何掩码各下采样 $\lambda$ 倍，获得：(1) 计算效率提升 (2) 粗体素分辨率天然容忍像素级深度误差

**Stage 2：后向投影稠密化**。用可变形交叉注意力(DCA)融合多视角图像特征，恢复几何完整性：

$$\mathbf{F}_{dense} = \text{DCA}(\mathbf{F}_{down}, \mathbf{F}^{(I)})$$

**Stage 3：深度引导非空体素精化**。在几何掩码指引下进行选择性两步精化：

- **几何精化**：仅对占用体素融合深度特征 $\mathbf{F}^{(D)}$，非占用体素赋予可学习空嵌入 $\mathbf{e}_{empty}$
- **语义增强**：对占用体素融合多视角图像特征

$$\mathbf{F}_{out} = \text{DCA}(\mathbf{F}_{geo}, \mathbf{F}^{(I)}; \mathcal{M}_{down})$$

核心思想：将计算资源集中在约 10% 的语义有意义体素上，避免在 90% 空体素上浪费计算。

#### 2. 区域引导专家 Transformer (R-EFormer)

**空间各向异性分析**：统计发现不同语义类别在高度和距离维度上分布差异显著：
- 路面集中在低高度、近距离
- 植被/建筑占据较高位置、中等距离
- 动态目标仅出现在窄空间带

**区域划分**：沿距离（近 0-10m / 中 10-30m / 远 ≥30m）和高度（低 -1~0.2m / 中 0.2~2.2m / 高 2.2~5.4m）将 3D 空间划分为 3×3=9 个区域 $\mathcal{R}_m$，每个区域分配专属专家 $E_m$。

**路由与专家选择**：

$$s_m = \text{Router}(\mathbf{F}_{out}), \quad \mathcal{S} = \text{TopK}(\{s_m\}_{m=1}^M, K)$$

每个专家应用同样的 DCA 模块但限制在对应区域的二值掩码 $\mathcal{M}_m$ 内：

$$\mathbf{F}_{final} = \sum_{m \in \mathcal{S}} w_m \cdot E_m(\mathbf{F}_{out}, \mathbf{F}^{(I)}; \mathcal{M}_m)$$

#### 3. 区域引导递归专家 Transformer (R2-EFormer)

R-EFormer 需要手动定义区域，存在超参数敏感性。R2-EFormer 借鉴 Mixture-of-Recursions (MoR) 思想，用单个专家递归迭代 n 次，每次迭代通过路由器逐步聚焦更小的显著区域：

$$\mathcal{M}^{(t)} = \begin{cases} \Omega, & t=1 \\ \text{TopK}(\mathcal{R}^{(t)}(\mathbf{F}^{(t-1)}, \mathcal{M}^{(t-1)}), k_t), & t>1 \end{cases}$$

覆盖比例逐步递减（100% → 75% → 50%），确保 $\mathcal{M}^{(t)} \subset \mathcal{M}^{(t-1)}$。

R2-EFormer 三大优势：
1. 单递归专家减少参数量
2. 自适应发现区域、减少手动定义的超参数敏感性
3. 逐步聚焦高置信区域，增强语义预测

### 损失函数 / 训练策略

- 标准占用预测损失（focal loss 分类 + scene-class CE）
- 图像编码器：ResNet-50；深度估计：moge-2-vits-normal
- 体素分辨率 0.4m，空间范围 80m×80m×6.4m，分辨率 200×200×16
- 前向投影特征下采样 1/16
- R-EFormer 多头注意力 8 头，$N_{ref}$=4 参考点
- AdamW：lr=$1 \times 10^{-4}$，weight decay=$1 \times 10^{-2}$，batch size 16（8×L20 GPU），24 epochs

## 实验关键数据

### 主实验

**表1：Occ3D-nuScenes mIoU(%) 对比**

| 方法 | Backbone | mIoU (%) |
|------|---------|:---:|
| BEVFormer | R101 | 26.9 |
| SparseOcc | R50 | 30.9 |
| BEVDet4D* | R50 | 36.0 |
| FlashOcc* | R50 | 37.8 |
| FB-Occ* | R50 | 39.1 |
| ViewFormer* | R50 | 41.9 |
| COTR* | R50 | 43.1 |
| **BEVDet4D+Dr.Occ*** | **R50** | **43.4** |
| **COTR+Dr.Occ*** | **R50** | **44.1** |

Dr.Occ 在 BEVDet4D 基线上提升 **+7.43% mIoU** 和 **+3.09% IoU**，集成到 SOTA 方法 COTR 后进一步提升 **+1.0% mIoU**。

**前景类别显著提升**：bicycle +20.4%、motorcycle +6.9%、pedestrian +13.4%、traffic cone +9.5%，验证了区域专家对稀有类的增益。

### 消融实验

**表2：各组件消融**

| D2-VFormer | R-EFormer | R2-EFormer | IoU (%) | mIoU (%) |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✗ | ✗ | 70.36 | 36.01 |
| ✓ | ✗ | ✗ | 71.29 | 41.45 |
| ✓ | ✓ | ✗ | **73.45** | 43.03 |
| ✓ | ✗ | ✓ | 72.87 | **43.43** |

- D2-VFormer 单独贡献 +5.44% mIoU——几何对齐是最大增益来源
- R-EFormer 在 D2-VFormer 基础上再加 +1.58% mIoU
- R2-EFormer IoU 略低但 mIoU 最高（43.43%），因为递归细化更善于处理稀有类别

### 关键发现

1. 高质量深度先验（MoGe-2）的有效利用方式不是直接投影，而是生成占用掩码引导模型聚焦——这一反直觉发现至关重要
2. 90% 体素为空的观察直接指导了计算效率优化策略
3. 语义类别的空间各向异性是一个被忽视但重要的问题，区域专家提供了有效解决方案
4. R2-EFormer 的自适应递归优于手动区域定义，在夜间等困难场景表现突出

## 亮点与洞察

1. **深度先验利用方式创新**：不是暴力使用深度图做投影（反而降低性能），而是巧妙地转化为占用掩码引导注意力——体现了对问题本质的深入理解
2. **MoE/MoR 在 3D 感知中的首次应用**：将区域感知与专家混合自然结合，为占用预测中的长尾问题提供了新思路
3. **即插即用设计**：D2-VFormer 和 R/R2-EFormer 可独立集成到不同基线，COTR+Dr.Occ 验证了泛化性
4. **空间各向异性的统计分析**：Figure 4 的可视化直观展示了语义类别在高度/距离维度的分布差异，为区域划分提供了数据支撑

## 局限与展望

1. MoGe-2 深度估计器未微调，可能存在域间差距；联合微调或换用驾驶域专用深度估计器可能进一步提升
2. R-EFormer 的区域划分依赖手动定义的高度/距离阈值，泛化性有限；R2-EFormer 解决了此问题但增加了递归计算
3. 仅在 nuScenes/Occ3D 评测，未验证 Waymo 等更大规模数据集
4. D2-VFormer 的三阶段渐进细化增加了模型复杂度，实时性评估不充分
5. 未探索时序信息的更深度利用（如视频级深度估计一致性）

## 相关工作与启发

- **BEVDet4D/BEVStereo**：前向投影基线，Dr.Occ 的几何增强在其上效果最显著
- **COTR**：双向投影 SOTA 方法，Dr.Occ 模块进一步提升其性能
- **MoGe/MoGe-2**：大规模视觉模型的深度估计泛化能力为 3D 感知提供新机遇
- **MoE/MoR**：从 NLP 领域引入的高效模型扩展范式，在 3D 视觉中有广阔应用空间
- 深度引导的占用掩码思想可推广到其他需要 2D→3D 映射的任务（如 3D 目标检测、场景重建）

## 评分

| 维度 | 分数 (1-5) |
|------|:---:|
| 创新性 | 4 |
| 技术深度 | 5 |
| 实验充分度 | 4 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| 总评 | 4.2 |

<!-- RELATED:START -->

## 相关论文

- [ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction](proood_prototype-guided_out-of-distribution_3d_occupancy_prediction.md)
- [M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)
- [OccAny: Generalized Unconstrained Urban 3D Occupancy](occany_generalized_unconstrained_urban_3d_occupancy.md)
- [An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)
- [KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System](knowval_a_knowledge-augmented_and_value-guided_autonomous_driving_system.md)

<!-- RELATED:END -->
