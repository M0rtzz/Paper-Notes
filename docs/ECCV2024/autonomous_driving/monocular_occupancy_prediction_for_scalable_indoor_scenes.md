---
title: >-
  [论文解读] Monocular Occupancy Prediction for Scalable Indoor Scenes
description: >-
  [ECCV 2024][自动驾驶][3D 占用预测] 提出 ISO（Indoor Scene Occupancy）方法，通过预训练深度模型和 D-FLoSP（双特征视线投影）模块实现室内场景的单目 3D 占用预测，并构建了规模比 NYUv2 大 40 倍的 Occ-ScanNet 基准数据集。
tags:
  - ECCV 2024
  - 自动驾驶
  - 3D 占用预测
  - 单目视觉
  - 室内场景
  - 深度估计
  - 语义场景补全
---

# Monocular Occupancy Prediction for Scalable Indoor Scenes

**会议**: ECCV 2024  
**arXiv**: [2407.11730](https://arxiv.org/abs/2407.11730)  
**代码**: [有](https://github.com/hongxiaoy/ISO.git)  
**领域**: 自动驾驶 / 室内场景理解  
**关键词**: 3D 占用预测, 单目视觉, 室内场景, 深度估计, 语义场景补全

## 一句话总结

提出 ISO（Indoor Scene Occupancy）方法，通过预训练深度模型和 D-FLoSP（双特征视线投影）模块实现室内场景的单目 3D 占用预测，并构建了规模比 NYUv2 大 40 倍的 Occ-ScanNet 基准数据集。

## 研究背景与动机

**室内 vs 室外的关键差异**：3D 占用预测在户外自动驾驶场景已取得显著进展（TPVFormer、FB-OCC 等），但室内场景研究仍然不足。室内场景有两大核心挑战区别于户外：

**场景尺度复杂性**：室内房间尺寸差异巨大——从宽敞的客厅到狭窄的厨房，不同于户外驾驶场景通常聚焦于固定 3D 空间。这对深度预测的精度提出了更高要求。

**物体复杂性**：室内物体密度更高、种类更多、尺寸变化更大。户外的车辆和行人在各自类别内尺寸相对一致且彼此间距较大，而室内家具的尺寸变化显著且紧密排列。

**现有方法的不足**：

- **MonoScene**：提出 FLoSP（视线投影）将 2D 特征提升到 3D，但共享的 2D 特征沿射线投影导致深度歧义——不同深度的体素获得相同特征，无法区分前后物体。
- **NDC-Scene**：设计了深度自适应双解码器，性能有所提升但仍有限。
- **数据集瓶颈**：现有室内场景工作主要使用 NYUv2 数据集，仅有 795/654 个训练/测试样本，规模太小，限制了模型的可扩展性研究。

**ISO 的核心思路**：
- 利用强大的预训练深度模型（Depth-Anything）获取精确的度量深度
- 设计 D-FLoSP 模块将深度信息与 2D 特征联合投影到 3D 空间，解决 FLoSP 的深度歧义
- 引入多尺度特征融合以适应室内物体的尺寸变化
- 构建大规模 Occ-ScanNet 基准推动该方向研究

## 方法详解

### 整体框架

ISO 的整体架构基于 2D UNet + 3D UNet，核心设计集中在 2D→3D 的特征变换上：

1. **2D 特征提取**：EfficientNet-B7 编码器 + 2D UNet 解码器得到多尺度 2D 特征图
2. **深度分支**：预训练 Depth-Anything 估计粗略度量深度 → DepthNet 精化为深度分布
3. **D-FLoSP 模块**：将深度分布和 2D 特征联合投影到 3D 体素空间
4. **3D UNet**：处理 3D 体素特征并输出占用预测

### 关键设计

1. **深度分支（Depth Branch）**

   **功能**：从单张 RGB 图像估计精确的像素级深度信息，为 2D→3D 特征变换提供关键的空间定位。

   **核心思路**：分为粗估计和精化两步：

   粗估计直接使用冻结的 Depth-Anything 模型：
   $$\mathbf{D}^{\text{metric}} = \mathbf{N}_{\text{depth}}(\mathbf{I}^{\text{rgb}}) \in \mathbb{R}^{1 \times H \times W}$$

   精化阶段将粗深度与图像特征拼接，通过 DepthNet 生成深度分布：
   $$\mathbf{D}^{\text{dist}}_{\text{s=1}} = \mathbf{F}_{\text{depth}}(\text{Concat}(\mathbf{D}^{\text{metric}}, \mathbf{X}^{\text{2d}}_{\text{s=1}})) \in \mathbb{R}^{N_{\text{bins}} \times H \times W}$$

   深度分布用 BCE 损失监督，GT 深度转为 one-hot 向量。

   **设计动机**：直接从零学习深度非常困难，而预训练深度模型（如 Depth-Anything）已经能估计度量级深度。但预训练深度不够精确（对高 mIoU 不足），因此设计了可学习的精化策略——将深度先验与图像特征融合，学习残差式的深度修正。

2. **D-FLoSP（双特征视线投影）模块**

   **功能**：将 2D 图像特征和深度分布联合提升到 3D 体素空间，解决原始 FLoSP 的深度歧义问题。

   **核心思路**：对每个 3D 体素中心 $x^c$，分别执行两条投影路径：

   路径 1 — 深度分布投影：将 2D 深度分布投影到 3D，得到每个体素的深度概率
   $$\mathbf{D}^{\text{3d}}_{\text{s}=k} = \mathbf{\Phi}^{\text{3d}}_{\rho(x^c)}(\mathbf{D}^{\text{dist}}_{\text{s}=k})$$

   路径 2 — 特征投影：类似原始 FLoSP，将 2D 特征投影到 3D
   $$\mathbf{X}^{\text{3d}}_{\text{s}=k} = \mathbf{\Phi}^{\text{2d}}_{\rho(x^c)}(\mathbf{X}^{\text{2d}}_{\text{s}=k})$$

   最终对多尺度进行逐元素乘法加权求和：
   $$\mathbf{X}^{\text{3d}} = \sum_{s \in \{1,2,4,8\}} \mathbf{X}^{\text{3d}}_{\text{s}=k} \odot \mathbf{D}^{\text{3d}}_{\text{s}=k}$$

   **设计动机**：原始 FLoSP 仅考虑射线投影，同一射线上所有体素获得相同的 2D 特征，导致深度歧义严重。D-FLoSP 引入深度分布作为权重，使得靠近真实深度的体素获得更高权重，远离的体素权重趋近于零，有效解决了歧义问题。"Dual" 指特征和深度两条并行的投影路径。

3. **多尺度深度融合**

   **功能**：利用多尺度（1/1, 1/2, 1/4, 1/8）的深度分布分别加权对应尺度的 3D 特征。

   **核心思路**：从 scale 1 的深度分布下采样得到其他尺度的深度分布，每个尺度的 3D 特征用对应尺度的深度分布加权，然后求和。

   **设计动机**：室内场景中不同物体尺寸差异大（天花板 vs 小物件），不同尺度的特征对不同大小物体的重要性不同。多尺度深度加权可以让模型在每个尺度上独立判断深度权重，而非用统一的深度分布"一刀切"地处理所有特征。

### 损失函数 / 训练策略

- **深度损失** $\mathcal{L}_{\text{depth}}$：BCE 损失，监督深度分布预测
- **占用损失**：遵循 MonoScene 的损失设计
- 2D 编码器使用预训练 EfficientNet-B7，Depth-Anything 模型冻结
- NYUv2 训练 30 epochs，学习率 5e-6（DepthNet）/ 1e-4（其他），epoch 20 时降 10 倍

## 实验关键数据

### 主实验

**NYUv2 数据集性能**（体素分辨率 60×36×60）：

| 方法 | 输入 | IoU | ceil | floor | wall | window | chair | bed | sofa | table | tvs | furn | obj | mIoU |
|------|------|-----|------|-------|------|--------|-------|-----|------|-------|-----|------|-----|------|
| LMSCNet | occ | 33.93 | 4.49 | 88.41 | 4.63 | 0.25 | 3.94 | 32.03 | 15.44 | 6.57 | 0.02 | 14.51 | 4.39 | 15.88 |
| MonoScene | rgb | 42.51 | 8.89 | 93.50 | 12.06 | 12.57 | 13.72 | 48.19 | 36.11 | 15.13 | 15.22 | 27.96 | 12.94 | 26.94 |
| NDC-Scene | rgb | 44.17 | 12.02 | 93.51 | 13.11 | 13.77 | 15.83 | 49.57 | 39.87 | 17.17 | 24.57 | 31.00 | 14.96 | 29.03 |
| **ISO** | **rgb** | **47.11** | **14.21** | 93.47 | **15.89** | **15.14** | **18.35** | **50.01** | **40.82** | **18.25** | **25.90** | **34.08** | **17.67** | **31.25** |

ISO 在 mIoU 上达到 31.25%，相比 NDC-Scene（29.03%）提升 **+2.22%**，IoU 从 44.17% 提升到 47.11%。

**Occ-ScanNet 数据集性能**（体素分辨率 60×60×36）：

| 方法 | IoU | mIoU |
|------|-----|------|
| MonoScene* | 41.60 | 24.62 |
| **ISO** | **42.16** | **28.71** |

在大规模 Occ-ScanNet 上，ISO 的 mIoU 从 24.62% 提升到 28.71%（**+4.09%**）。

### 消融实验

**深度融合方式消融**（NYUv2 + Occ-ScanNet-mini）：

| 配置 | NYUv2 IoU | NYUv2 mIoU | Occ-mini IoU | Occ-mini mIoU |
|------|-----------|------------|--------------|---------------|
| baseline | 42.27 | 27.13 | 50.94 | 38.95 |
| + BEV-depth | 42.67 | 27.14 | 51.58 | 38.48 |
| + voxel-depth | **47.11** | **31.25** | 51.03 | **39.08** |

Voxel-depth 融合远优于 BEV-depth（室内场景高度信息比鸟瞰图更重要）。

**深度模型消融**（NYUv2）：

| 深度方法 | learned | multi-scale | IoU | mIoU |
|---------|---------|-------------|-----|------|
| GT 深度 | ✓ | — | 53.98 | 34.47 |
| ZoeDepth | ✓ | ✓ | 45.24 | 29.40 |
| DepthAnything | ✓ | ✓ | 46.94 | 31.02 |
| DepthAnything | ✓ | ✗ | **47.11** | **31.25** |

DepthAnything 优于 ZoeDepth，且可学习的深度精化策略进一步提升性能。

### 关键发现

- **体素深度优于 BEV 深度**：BEV 方法在天花板/地板上表现好但对家具差，因为室内高度信息比水平投影更重要
- **多尺度深度并非总是更好**：在 DepthAnything + learned 设置下，单尺度（47.11/31.25）反而略优于多尺度（46.94/31.02），可能是因为多尺度引入了冗余
- **数据规模效应显著**：10% 样本训练 IoU=21.79/mIoU=9.57，100% 样本 IoU=42.16/mIoU=28.71，说明室内占用预测亟需大规模数据
- **GT 深度的上界**：使用 GT 深度时到 mIoU=34.47，说明深度精度仍有较大提升空间

## 亮点与洞察

1. **Occ-ScanNet 基准贡献突出**：45755/19764 样本，比 NYUv2 大 40 倍，配合完整的标注流程和质量控制，有望成为室内占用预测的标准基准
2. **D-FLoSP 设计优雅**：用深度分布加权 FLoSP 特征，以最小的架构改动解决了严重的深度歧义问题
3. **复用预训练深度模型的策略**：冻结 Depth-Anything + 可学习精化的组合，平衡了预训练知识和任务特定学习

## 局限与展望

- **语义学习困难**：类别不平衡导致小物体（如电视、窗户）的预测精度仍然较低
- **仅 11 个语义类别**：未涵盖真实场景的类别多样性
- **单目限制**：尽管利用了深度先验，但单目输入在大遮挡场景下仍有根本局限
- 可探索结合基础分割模型（如 SAM）为占用预测提供语义先验
- 数据集可扩展到更多场景和更多语义类别

## 相关工作与启发

- **MonoScene** 开创了纯视觉单目 SSC，FLoSP 是其核心创新，ISO 在此基础上引入深度信息改进
- **BEVDepth/LSS** 的 2D→3D lift 思想被适配到室内场景，但需从 BEV 改为 voxel 深度
- 室内占用预测可能受益于 3D 重建（NeRF/3DGS）的深度先验迁移

## 评分

- **新颖性**: ⭐⭐⭐ — D-FLoSP 是对 FLoSP 的合理改进但增量有限，Occ-ScanNet 基准的贡献更为突出
- **实验充分度**: ⭐⭐⭐⭐ — 两个数据集验证 + 详细消融（深度融合/深度模型/多尺度/数据规模），分析充分
- **写作质量**: ⭐⭐⭐⭐ — 问题分析清晰，室内 vs 户外的差异论述有说服力
- **价值**: ⭐⭐⭐⭐ — Occ-ScanNet 数据集的长期价值较高，方法为室内 3D 感知提供了强基线

<!-- RELATED:START -->

## 相关论文

- [Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)](../../CVPR2026/autonomous_driving/monocular_open_vocabulary_occupancy_prediction_for_indoor_scenes.md)
- [UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction](unitraj_a_unified_framework_for_scalable_vehicle_trajectory_prediction.md)
- [Fully Sparse 3D Occupancy Prediction](fully_sparse_3d_occupancy_prediction.md)
- [OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](occgen_generative_multimodal_3d_occupancy_prediction_for_aut.md)
- [GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction](gaussianformer_scene_as_gaussians_for_vision-based_3d_semantic_occupancy_predict.md)

<!-- RELATED:END -->
