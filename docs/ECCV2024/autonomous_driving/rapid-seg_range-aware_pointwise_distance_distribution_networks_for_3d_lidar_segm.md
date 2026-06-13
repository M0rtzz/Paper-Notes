---
title: >-
  [论文解读] RAPiD-Seg: Range-Aware Pointwise Distance Distribution Networks for 3D LiDAR Segmentation
description: >-
  [ECCV 2024][自动驾驶][LiDAR语义分割] 本文提出 RAPiD 特征（Range-Aware Pointwise Distance Distribution），一种对刚体变换不变且适应点密度变化的 LiDAR 点云局部几何特征，配合双层嵌套自编码器和通道注意力融合，在 SemanticKITTI（76.1 mIoU）和 nuScenes（83.6 mIoU）上达到 SOTA 分割性能。
tags:
  - "ECCV 2024"
  - "自动驾驶"
  - "LiDAR语义分割"
  - "等距不变性"
  - "距离分布特征"
  - "自编码器嵌入"
  - "注意力融合"
---

# RAPiD-Seg: Range-Aware Pointwise Distance Distribution Networks for 3D LiDAR Segmentation

**会议**: ECCV 2024  
**arXiv**: [2407.10159](https://arxiv.org/abs/2407.10159)  
**代码**: [https://github.com/l1997i/rapid_seg](https://github.com/l1997i/rapid_seg)  
**领域**: 自动驾驶  
**关键词**: LiDAR语义分割, 等距不变性, 距离分布特征, 自编码器嵌入, 注意力融合

## 一句话总结

本文提出 RAPiD 特征（Range-Aware Pointwise Distance Distribution），一种对刚体变换不变且适应点密度变化的 LiDAR 点云局部几何特征，配合双层嵌套自编码器和通道注意力融合，在 SemanticKITTI（76.1 mIoU）和 nuScenes（83.6 mIoU）上达到 SOTA 分割性能。

## 研究背景与动机

3D LiDAR 语义分割是自动驾驶场景理解的基础任务。现有方法主要依赖坐标和反射强度作为输入特征，存在两个核心问题：(1) 对刚体变换（旋转、平移）缺乏不变性；(2) 在点云稀疏或遮挡区域表现不佳。尽管数据增强可以部分缓解变换问题，但无法保证覆盖所有可能的变换。

关键矛盾在于：需要一种特征同时满足"捕获局部几何结构"、"刚体变换不变"和"适应 LiDAR 噪声环境"三个要求，但现有方法难以全部满足。

本文的灵感来自晶体学中的 Pointwise Distance Distribution（PDD），这是一种通过计算点与邻近点距离来表示局部结构的等距不变量。但 PDD 直接用于大规模 LiDAR 点云存在维度过高、计算量大、忽略局部性等问题。核心 idea 是将 PDD 适配为 LiDAR 场景的 RAPiD 特征，利用 LiDAR 的环状结构和语义类别进行局部化，并引入4D距离（3D几何+反射率）增强语义判别力。

## 方法详解

### 整体框架

输入点云产生三类特征：坐标特征 $F_C$、反射强度特征 $F_I$、RAPiD 特征 $F_R$。$F_C \oplus F_I$ 通过 VSA 体素编码器得到体素表征，$F_R$ 通过 RAPiD AutoEncoder 压缩为低维体素嵌入。两者经通道注意力融合后送入 backbone（Minkowski-UNet34）进行分割预测。

### 关键设计

1. **RAPiD 特征**: 给定点 $\bm{p}_j$ 及其 k 个局部邻域点，计算4D距离矩阵：$\bm{\rho}_{j,l} = \|[\bm{p}_j - \bm{p}_{j,l}, g(r_j) - g(r_{j,l})]\|_2$，其中 $g(\cdot)$ 将反射率映射到与欧氏距离相同的数值范围。按行和列排序后得到 $u \times k$ 的 RAPiD 矩阵。设计动机：(a) 距离在刚体变换下不变，确保特征的等距不变性；(b) 4D距离融入反射率差异，不同表面材质会产生不同的距离分布，增强类间区分度；(c) 按距离范围使用不同的 $k$ 值（$k_{close}, k_{mid}, k_{far}$），适应 LiDAR 点密度的远近变化。

2. **Intra-Ring RAPiD (R-RAPiD) 和 Intra-Class RAPiD (C-RAPiD)**: R-RAPiD 将 RoI 限制在同一激光束环内，利用 LiDAR 各向同性辐射的固有结构减小计算开销。通过 beam ID 将点云分割为 B 个环，每个环内独立计算 RAPiD。C-RAPiD 将 RoI 限制在同一语义类别内，利用语义标签增强类内特征一致性。训练时使用 GT 标签，测试时使用预训练 R-RAPiD-Seg 生成的伪标签。两者互补：R-RAPiD 不依赖标签、通用性强，C-RAPiD 加强类内嵌入保真度。

3. **双层嵌套 RAPiD AutoEncoder**: 外层 VSA AE 将高维逐点特征压缩为体素表征 $H^v \in \mathbb{R}^{c \times l \times d}$，使用 scatter-sum 聚合和 cross-attention 交互。内层 AE 在体素维度上进一步降维（$d \to d'$），使用卷积层和 ConvFFN 促进体素间信息交换。关键创新是 class-aware 对比损失 $\mathcal{L}_{contr}$，最大化不同类别嵌入间距离、最小化同类嵌入间距离，解决 AE 嵌入非唯一性问题。

4. **通道注意力融合（FuAtten）**: 将坐标嵌入 $E_C$、反射率嵌入 $E_I$ 和 RAPiD 嵌入 $E_R$ 拼接后，通过 squeeze-excitation 机制生成通道级权重 $\bm{a}_z = \sigma(\mathbf{W}_2 \delta(\mathbf{W}_1 \mathbf{z}))$，自适应加权各通道特征。避免了简单拼接导致的维度爆炸和训练偏向问题。

### 损失函数 / 训练策略

AE 总损失 $\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda \mathcal{L}_{contr}$，其中 $\mathcal{L}_{recon}$ 为 MSE 重建损失。

采用两阶段训练：第一阶段独立训练 RAPiD AE；第二阶段冻结 AE 参数后整合到完整网络中训练分割。两种架构变体：R-RAPiD-Seg（轻量版，仅用 R-RAPiD）和 C-RAPiD-Seg（性能版，同时用 R-RAPiD 和 C-RAPiD）。

4×A100 GPU，学习率 1e-3，SGD 优化器，cosine schedule，2个 epoch warmup，共100 epochs。推理时间约 105ms/帧。

## 实验关键数据

### 主实验

| 数据集 | 指标 | RAPiD-Seg | 之前SOTA | 提升 |
|--------|------|-----------|----------|------|
| SemanticKITTI test | mIoU | **76.1** | UniSeg 75.2 (多模态) | +0.9 |
| nuScenes test | mIoU | **83.6** | UniSeg 83.5 (多模态) | +0.1 |

SemanticKITTI test 细分（vs 同为 LiDAR-only 方法）：

| 方法 | mIoU | truck | o.veh | park | o.gro |
|------|------|-------|-------|------|-------|
| PCSeg | 72.9 | 58.6 | 68.6 | 71.5 | 36.9 |
| RangeFormer | 73.3 | 59.9 | 66.2 | 73.0 | 42.4 |
| **RAPiD-Seg** | **76.1** | **72.5** | **80.7** | **78.2** | **46.0** |

nuScenes test（vs 多模态方法）：

| 方法 | 模态 | mIoU | truck | trail | const |
|------|------|------|-------|-------|-------|
| UniSeg | L+C | 83.5 | 76.7 | 86.3 | 80.5 |
| LidarMultiNet | L+C | 81.4 | 74.8 | 86.9 | 71.5 |
| **RAPiD-Seg** | **L** | **83.6** | **79.0** | **88.5** | **84.6** |

### 消融实验

SemanticKITTI val 逐步添加组件：

| 配置 | mIoU | Δ | 说明 |
|------|------|---|------|
| Baseline | 70.04 | - | 无 RAPiD |
| + Geometric RAPiD | 71.21 | +1.17 | 3D距离有效 |
| + Reflectivity | 71.93 | +1.89 | 反射率融入提升大 |
| + RAPiD Embedding (AE) | 72.15 | +2.11 | AE 压缩有效 |
| + Attention 融合 | 72.32 | +2.28 | 注意力优于拼接 |
| + 全部（R+C-RAPiD） | **73.02** | **+2.98** | 所有组件协同最优 |

RAPiD vs PDD 对比：

| 方法 | mIoU | truck | o.veh | park |
|------|------|-------|-------|------|
| Baseline | 70.0 | 59.8 | 70.3 | 69.2 |
| PDD（原始） | 66.2 | 40.3 | 65.8 | 67.5 |
| **RAPiD+R** | **73.0** | **70.4** | **78.5** | **75.8** |

### 关键发现

- 直接使用 PDD 特征反而比 baseline 差 3.8 mIoU，说明原始 PDD 不适合 LiDAR 场景；RAPiD 的 range-aware 设计和反射率融入是关键
- 单模态（仅 LiDAR）方法超过了多模态方法（含相机），说明 RAPiD 提取的局部几何特征比 RGB 信息更有效
- 刚体目标类别（truck、o.veh、park 等）提升最大，符合 RAPiD 对刚体变换不变性的设计初衷
- R-RAPiD-Seg 已有 2.3 mIoU 提升，C-RAPiD-Seg 进一步提升到 3.0 mIoU，说明类内特征约束有效

## 亮点与洞察

- **从晶体学到自动驾驶的跨领域迁移**：PDD → RAPiD 的适配是一个漂亮的科研范例，但需要非平凡的改造（range-aware、反射率4D距离、LiDAR 环结构）
- **4D距离的设计**：将反射率映射到与几何距离相同的量纲后融入距离计算，优雅且有效
- **双层嵌套 AE 的必要性**：外层处理点→体素的不规则转换，内层处理高维→低维的压缩，分工明确
- **类感知对比损失**：解决 AE 嵌入非唯一性的关键，使同类特征聚拢、异类特征分开

## 局限与展望

- C-RAPiD 依赖语义标签或伪标签，伪标签质量直接影响性能
- RAPiD 特征需要预计算，增加了数据处理流程的复杂性
- 在极度稀疏的远距离区域，邻域点数可能不足以计算有意义的 RAPiD
- AE 的两阶段训练增加了训练复杂度
- 仅验证了单帧分割，未探索多帧时序融合

## 相关工作与启发

- **PDD（Widdowson & Kurlin）**: 晶体学中的等距不变量，是 RAPiD 的理论基础
- **Cylinder3D**: 针对 LiDAR 点云分布的柱面分区，是常见 backbone
- **SPVCNN**: 点-体素混合方法，为 RAPiD-Seg 提供了 backbone 参考
- 启发：跨领域的数学不变量可能在新领域有意想不到的应用，但需要针对性适配

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 从晶体学引入 PDD 到 LiDAR 分割，4D距离和 range-aware 设计新颖独特
- 实验充分度: ⭐⭐⭐⭐⭐ — 两个主流数据集 SOTA，消融极其详细，PDD vs RAPiD 对比令人信服
- 写作质量: ⭐⭐⭐⭐ — 数学公式严谨，但信息密度大，阅读难度较高
- 价值: ⭐⭐⭐⭐ — 提供了一种新的点云特征设计范式，可启发后续工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] ItTakesTwo: Leveraging Peer Representations for Semi-supervised LiDAR Semantic Segmentation](ittakestwo_leveraging_peer_representations_for_semi-supervised_lidar_semantic_se.md)
- [\[ECCV 2024\] SFPNet: Sparse Focal Point Network for Semantic Segmentation on General LiDAR Point Clouds](sfpnet_sparse_focal_point_network_for_semantic_segmentation_on_general_lidar_poi.md)
- [\[ICLR 2026\] Adaptive Augmentation-Aware Latent Learning for Robust LiDAR Semantic Segmentation](../../ICLR2026/autonomous_driving/adaptive_augmentation-aware_latent_learning_for_robust_lidar_semantic_segmentati.md)
- [\[ECCV 2024\] Rethinking Data Augmentation for Robust LiDAR Semantic Segmentation in Adverse Weather](rethinking_data_augmentation_for_robust_lidar_semantic_segmentation_in_adverse_w.md)
- [\[CVPR 2026\] Neural Distribution Prior for LiDAR Out-of-Distribution Detection](../../CVPR2026/autonomous_driving/neural_distribution_prior_for_lidar_ood_detection.md)

</div>

<!-- RELATED:END -->
