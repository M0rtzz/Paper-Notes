---
title: >-
  [论文解读] Fully Sparse 3D Occupancy Prediction
description: >-
  [ECCV 2024][自动驾驶][3D Occupancy] 提出 SparseOcc，首个完全稀疏的 3D 占用预测网络，通过稀疏体素解码器和掩码引导的 Mask Transformer 实现高效占用预测，并设计了 RayIoU 评价指标解决传统 mIoU 的深度方向不一致惩罚问题。
tags:
  - ECCV 2024
  - 自动驾驶
  - 3D Occupancy
  - 稀疏网络
  - Transformer
  - 评价指标
  - BEV感知
---

# Fully Sparse 3D Occupancy Prediction

**会议**: ECCV 2024  
**arXiv**: [2312.17118](https://arxiv.org/abs/2312.17118)  
**代码**: [GitHub](https://github.com/MCG-NJU/SparseOcc)  
**领域**: 自动驾驶  
**关键词**: 3D Occupancy, 稀疏网络, Mask Transformer, 评价指标, BEV感知

## 一句话总结

提出 SparseOcc，首个完全稀疏的 3D 占用预测网络，通过稀疏体素解码器和掩码引导的 Mask Transformer 实现高效占用预测，并设计了 RayIoU 评价指标解决传统 mIoU 的深度方向不一致惩罚问题。

## 研究背景与动机

3D 占用预测是自动驾驶中的关键任务，为场景提供比 3D 目标检测更精细的几何描述。但现有方法存在严重效率瓶颈：

**密集表示冗余**：现有方法（BEVFormer-Occ、SurroundOcc 等）构建密集的 3D 体积（如 200x200x16），但统计显示超过 90% 的体素是空的

**计算开销巨大**：密集方法仅能达到 2-3 FPS（A100），远不能满足实时需求

**伪稀疏方法**：VoxFormer 等虽然引入了稀疏 query，但仍包含 sparse-to-dense 模块（如 MAE），本质上不是完全稀疏

**评价指标有缺陷**：传统 voxel-level mIoU 存在深度方向不一致惩罚，模型可通过预测更厚的表面作弊

本文从两个角度同时推进：(1) 架构层面提出完全稀疏的设计；(2) 评价层面设计更合理的 RayIoU 指标。

## 方法详解

### 整体框架

SparseOcc 包含三个模块：
1. **图像编码器**：Backbone + FPN 提取多视图 2D 特征
2. **稀疏体素解码器**：从粗到细重建场景的稀疏 3D 几何结构
3. **Mask Transformer**：用稀疏 query 预测每个语义/实例段的掩码和标签

整个流程没有任何密集设计——不依赖密集 3D 特征、没有 sparse-to-dense 模块、没有全局注意力。

### 关键设计

#### 稀疏体素解码器（Sparse Voxel Decoder）

采用从粗到细的流水线结构，包含 3 层：

- **初始化**：在 3D 空间均匀分布一组粗体素 query（如 25x25）
- **每层操作**：(1) 2x 上采样（每个体素分裂为 8 个）(2) 估计每个体素的占用得分 (3) 通过 top-k 剪枝移除空体素
- **特征获取**：对每个体素 query，使用自注意力聚合局部/全局特征，然后通过线性层生成 3D 采样偏移，投影到多视图图像空间获取特征

关键设计参数：k=32000（仅占总体素 640000 的 5%），即 5% 稀疏度即足以覆盖场景。

**时序建模**：利用全局采样参考点的灵活性，将它们变换到历史时间戳采样历史多视图特征，无需密集特征的 warp 操作。

**监督**：每层使用加权 BCE Loss，不同类别的权重按占比反比分配：w_c = sum(M_i) / M_c。

#### Mask Transformer

受 Mask2Former 启发，使用 N 个稀疏语义/实例 query：

- **掩码引导稀疏采样（Mask-guided Sparse Sampling）**：在上一层预测的掩码内随机选择 3D 点，投影到多视图图像获取特征。相比密集交叉注意力，推理速度提升 50% 且性能更优
- **预测**：query embedding 经线性分类器预测类别标签；经 MLP 转为 mask embedding，与稀疏体素 embedding 做点积得到掩码预测
- **预测空间限制**：掩码预测仅在稀疏体素解码器输出的稀疏空间内，而非全场景

**损失函数**：使用匈牙利匹配后，Focal Loss（分类）+ DICE Loss + BCE Mask Loss + Occ Loss（体素解码器）。

#### RayIoU 评价指标

针对传统 voxel-level mIoU 的三大问题设计：

**问题分析**：
1. 密集方法通过预测更厚的表面可以作弊——BEVFormer 使用 visible mask 训练后 mIoU 虚增 5-15
2. 薄表面预测过于严格——偏差一个体素就导致 IoU 为零
3. Visible mask 仅考虑当前时刻可见区域，忽略场景补全能力

**RayIoU 方案**：模拟 LiDAR 行为，向预测的 3D 占用体积发射查询射线：
- 计算射线到第一个占用体素的距离和类别
- 真正例条件：类别一致 + 深度 L1 误差 < 阈值（1m/2m/4m）
- 最终指标取三个阈值的平均
- 射线来源于 8 个 LiDAR 位置（支持时间投射评估场景补全能力）
- 射线重采样以平衡不同距离的权重

### 损失函数 / 训练策略

- **优化器**：AdamW，全局 batch size 8，初始学习率 2e-4，余弦退火
- **训练**：24 epochs，ResNet-50 backbone
- **Mask Transformer**：3 层共享权重
- **损失**：L = L_focal + L_mask + L_dice + L_occ

## 实验关键数据

### 主实验（Occ3D-nuScenes val）

| 方法 | Backbone | 帧数 | RayIoU | FPS |
|------|----------|------|--------|-----|
| BEVFormer (4f) | R101 | 4 | 32.4 | 3.0 |
| BEVDet-Occ (2f) | R50 | 2 | 29.6 | 2.6 |
| BEVDet-Occ-Long (8f) | R50 | 8 | 32.6 | 0.8 |
| FB-Occ (16f) | R50 | 16 | 33.5 | 10.3 |
| **SparseOcc (8f)** | R50 | 8 | **34.0** | **17.3** |
| SparseOcc (16f) | R50 | 16 | 35.1 | 12.5 |
| SparseOcc (16f, 48ep) | R50 | 48ep | 36.1 | 12.5 |

SparseOcc 在弱设置（R50, 8 frames, 704x256）下超越 FB-Occ（挑战赛冠军）+1.6 RayIoU，速度快 1.7x。

### 消融实验

**稀疏 vs 密集体素解码器**：

| 解码器类型 | RayIoU | FPS |
|-----------|--------|-----|
| 密集 coarse-to-fine | 29.9 | 6.3 |
| 密集 patch-based | 25.8 | 7.8 |
| **稀疏 coarse-to-fine** | **29.9** | **24.0** |

稀疏解码器速度快 4x，性能不降。

**Mask Transformer + 稀疏采样**：

| Mask Transformer | 交叉注意力 | RayIoU | FPS |
|-----------------|-----------|--------|-----|
| 无 | - | 27.0 | 29.0 |
| 有 | 密集 | 28.7 | 16.2 |
| 有 | 稀疏+掩码引导 | **29.2** | **24.0** |

**体素稀疏度**：k=32000（5%）时最优；继续增加密度反而引入噪声降低精度。

**时序帧数**：性能随帧数增加持续提升至 12 帧后饱和，但推理速度线性下降。

### 关键发现

1. 仅 5% 的体素即足以覆盖场景几何——密集表示存在大量冗余
2. 掩码引导的稀疏采样比密集交叉注意力更强且更快
3. RayIoU 成功避免了厚表面作弊问题，提供了更公平的评价
4. 阈值剪枝与 top-k 剪枝性能相当，具有更好的场景泛化能力

## 亮点与洞察

- **第一个完全稀疏占用网络**：从架构上彻底消除密集设计，实时推理 17.3 FPS
- **RayIoU 指标贡献独立于模型**：解决了社区共同面对的评价问题，被后续工作广泛采用
- **与 Mask2Former 的巧妙结合**：将 2D 分割中的 mask transformer 范式迁移到 3D 稀疏占用
- **5% sparsity 的实证发现**：为后续稀疏方法提供了关键的设计指导

## 局限与展望

1. top-k 剪枝是数据集相关的超参数，阈值剪枝虽可泛化但需要调参
2. 稀疏体素解码器的早期剪枝可能丢失某些小目标（如行人、交通锥）
3. 时序建模效率随帧数线性下降，需要更高效的时序融合方案
4. 当前仅使用语义 query，实例级占用（全景占用）的验证不够充分

## 相关工作与启发

- **SparseBEV**：完全稀疏的 3D 检测方法，SparseOcc 将其从 pillar query 扩展到 3D voxel query
- **Mask2Former / Mask3D**：mask transformer 的 2D/3D 分割方法，提供了掩码预测范式
- **FB-Occ**：CVPR 2023 占用挑战赛冠军，复杂但密集的设计被 SparseOcc 以更简洁架构超越
- 稀疏思路可进一步推广至 4D 占用预测和端到端驾驶

## 评分

- **创新性**: 5/5 - 首个完全稀疏占用网络 + RayIoU 两大贡献
- **实验质量**: 4/5 - 消融充分，但缺少更多 backbone 和数据集验证
- **写作质量**: 5/5 - 问题分析透彻，RayIoU 的动机论述精彩
- **实用价值**: 5/5 - 实时推理 + 开源代码，实用性强

<!-- RELATED:START -->

## 相关论文

- [GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction](gaussianformer_scene_as_gaussians_for_vision-based_3d_semantic_occupancy_predict.md)
- [OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)
- [OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](occgen_generative_multimodal_3d_occupancy_prediction_for_aut.md)
- [SparseAlign: A Fully Sparse Framework for Cooperative Object Detection](../../CVPR2025/autonomous_driving/sparsealign_a_fully_sparse_framework_for_cooperative_object_detection.md)
- [Monocular Occupancy Prediction for Scalable Indoor Scenes](monocular_occupancy_prediction_for_scalable_indoor_scenes.md)

<!-- RELATED:END -->
