---
title: >-
  [论文解读] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation
description: >-
  [3D视觉] 提出 Easy3D，一种简洁高效的 3D 交互式实例分割方法，结合体素稀疏编码器、轻量 Transformer 解码器和隐式点击融合策略，在域内和域外数据集上一致性地超越 SOTA，并首次将学习的负嵌入 (learned negative embedding) 成功应用于隐式点击融合。
tags:
  - 3D视觉
---

# Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2504.11024](https://arxiv.org/abs/2504.11024)
- **代码**: [simonelli-andrea.github.io/easy3d](https://simonelli-andrea.github.io/easy3d)
- **领域**: 3D Vision / 3D 交互式分割
- **关键词**: 3D交互式实例分割, 体素编码器, 隐式点击融合, 负嵌入, 跨域泛化
- **作者**: Andrea Simonelli, Norman Müller, Peter Kontschieder (Meta Reality Labs Zürich)

## 一句话总结

提出 Easy3D，一种简洁高效的 3D 交互式实例分割方法，结合体素稀疏编码器、轻量 Transformer 解码器和隐式点击融合策略，在域内和域外数据集上一致性地超越 SOTA，并首次将学习的负嵌入 (learned negative embedding) 成功应用于隐式点击融合。

## 研究背景与动机

随着 3D 数字环境（通过 NeRF、3DGS、激光扫描等获取）的普及，3D 交互式分割的需求日益增长。用户可以通过 3D 点击定义目标物体，系统生成分割掩码，用户可以通过正/负点击迭代修正结果。

现有方法存在两大局限：

**AGILE3D**：使用体素编码器 + 显式点击融合（per-click mask 的 max 操作），在域外数据上泛化差

**Point-SAM**：使用 ViT 编码器 + 隐式点击融合，参数量大、效率低，且 point-group 表示缺乏通用性

核心观察：体素表示的域无关性（domain-independent）和隐式点击融合的强泛化性可以结合取长补短。

## 方法详解

### 整体架构

输入：3D 场景点云 $S_P \in \mathbb{R}^{N_P \times 6}$（每点 3D 坐标 + 颜色）和用户点击集 $C = \{c_1, ..., c_{N_C}\} \in \mathbb{R}^{N_C \times 4}$（每点击 3D 坐标 + 正/负标签）

流程：
1. **场景预处理**：将点云体素化为 $S_V$，分辨率 $V_S = 5$cm
2. **场景编码**：稀疏 U-Net 编码体素化场景 → 场景嵌入 $S_E$
3. **点击编码**：位置编码 + 标签编码 → 点击嵌入 $C_E$
4. **解码**：Two-Way Transformer 双向注意力交互更新 $S_E$ 和 $C_E$
5. **点击融合**：隐式融合得到分割掩码 $M_V$
6. **后处理**：映射回原始点云获得 $M_P$

### 体素化 vs Point-Group

体素化的优势：
- 降维同时保持显式、通用的度量分辨率
- 可利用高效稀疏卷积库（SpConv）
- 域无关表示，对物体类型/密度变化更鲁棒
- Point-SAM 的 point-group 依赖于特定数据集的几何和密度分布

### 隐式 vs 显式点击融合（核心对比）

**显式融合**（AGILE3D）：
- 每个点击预测独立掩码 → max 操作合并
- 解码器不感知正/负标签（仅在 max 操作中区分）
- 单个点击掩码在域外数据上置信度低，融合效果差

**隐式融合**（Easy3D/SAM）：
- 引入学习的 Output Embedding，在解码器中与所有点击嵌入一起参与注意力
- 解码器感知正/负标签，通过注意力机制综合所有点击信息
- 最终掩码 = 更新后 Output Embedding 与场景嵌入的点积

### 学习负嵌入 (Learned Negative Embedding) — 首次用于隐式融合

引入第二个学习的输出嵌入（负输出嵌入），类似 AGILE3D 中的无位置负点击：
- 在训练中自动学习哪些场景部分通常是背景
- 最终掩码 = 正输出嵌入掩码 > 负输出嵌入掩码的区域
- 即使只有 1 个用户点击也能有效排除背景

### 训练

- 模拟用户交互：自动选择点击（首次在物体中心，后续在最大错误区域中心）
- 迭代 $N_C = 10$ 次点击
- 损失：DICE + Cross Entropy，等权，每轮累积后一次反传
- 训练 1k epochs，PyTorch + SpConv，lr=1e-4，多项式衰减

## 实验关键数据

### 主实验：跨数据集交互式分割（Table 1，均仅在 ScanNet40 训练）

| 测试集 | 方法 | IoU@1 | IoU@2 | IoU@3 | IoU@5 | IoU@10 |
|--------|------|-------|-------|-------|-------|--------|
| ScanNet40 | AGILE3D | 63.0 | 70.6 | 75.1 | 79.7 | 83.5 |
| ScanNet40 | **Easy3D** | **68.2** | **74.6** | **77.3** | 79.6 | 81.7 |
| S3DIS | AGILE3D | 58.5 | 70.7 | 77.4 | 83.6 | 88.3 |
| S3DIS | Point-SAM | 38.8 | — | 67.1 | 72.2 | 80.6 |
| S3DIS | **Easy3D** | **65.7** | **76.0** | **80.8** | **84.9** | 87.8 |
| KITTI-360 | AGILE3D | 34.8 | 40.7 | 42.7 | 44.4 | 49.6 |
| KITTI-360 | Point-SAM | 44.0 | — | 67.1 | 72.2 | 80.8 |
| KITTI-360 | **Easy3D** | **46.3** | **58.7** | **66.7** | **76.2** | **83.6** |

在最具挑战的域外数据集 KITTI-360 上，Easy3D 的 IoU@10 达到 83.6，比 AGILE3D 高 **+34**。

### 消融：点击融合策略 + 负嵌入（Table 3）

| 测试集 | 融合 | 负嵌入 | IoU@1 | IoU@3 | IoU@10 |
|--------|------|--------|-------|-------|--------|
| ScanNet40 | 显式 | ✗ | 59.6 | 73.2 | 82.6 |
| ScanNet40 | 显式 | ✓ | 62.7 | 75.2 | 83.6 |
| ScanNet40 | 隐式 | ✗ | 66.4 | 76.3 | 81.2 |
| ScanNet40 | **隐式** | **✓** | **68.2** | **77.3** | **81.7** |
| KITTI-360 | 显式 | ✗ | 31.0 | 40.0 | 46.3 |
| KITTI-360 | 显式 | ✓ | 34.5 | 42.6 | 48.2 |
| KITTI-360 | 隐式 | ✗ | 44.9 | 65.7 | 83.2 |
| KITTI-360 | **隐式** | **✓** | **46.3** | **66.7** | **83.6** |

### 与非交互方法对比（Table 2，仅 ScanNet20 训练）

| 设置 | 方法 | mAP | AP50 | AP25 |
|------|------|-----|------|------|
| ScanNet20→ScanNet20 | Mask3D | 51.5 | 77.0 | 90.2 |
| ScanNet20→ScanNet20 | AGILE3D | 53.5 | 75.6 | 91.3 |
| ScanNet20→ScanNet20 | **Easy3D** | **56.1** | **79.5** | **93.1** |
| ScanNet20→ScanNet40 | Mask3D | 5.3 | 13.1 | 24.7 |
| ScanNet20→ScanNet40 | AGILE3D | 24.8 | 45.7 | 72.4 |
| ScanNet20→ScanNet40 | **Easy3D** | **39.2** | **64.6** | **85.5** |

在 unseen 类上优势极为明显（mAP 39.2 vs 24.8）。

### 关键发现

1. **隐式融合在域外泛化上远超显式**：KITTI-360 上 IoU@10 相差 +73%（83.6 vs 48.2）
2. **显式融合在域内 + 多点击时有微弱优势**：因为每个点击的掩码在已知域内置信度高
3. **负嵌入一致性提升所有设置**：无论隐式/显式融合，无论域内/域外
4. **体素>点组**：域无关的体素表示带来更稳定的跨域表现
5. **甚至适用于 Gaussian Splatting 场景**：GS-ScanNet40 上优势显著

## 亮点与洞察

- **"Easy" 的名字恰如其分**：简洁地组合已有组件（体素编码器 + 隐式融合 + 负嵌入），但效果出众
- **首次系统分析隐式 vs 显式融合**：清晰揭示了两种策略的优劣场景
- **VR 应用演示**：在消费级头显上实现实时 3D 交互式分割 + 物体操控，实用性强
- 无需预训练，从头训练 1k epochs 即可

## 局限性

- 隐式融合在域内 + 大量点击时略逊于显式融合（差距很小）
- 体素分辨率 5cm 固定，极精细/极大场景可能受限
- 目前仅支持单物体分割场景
- 缺乏对 ScanNet++ 上更多分析

## 相关工作与启发

- **SAM 到 3D 的自然延伸**：将 SAM 的隐式融合范式成功迁移到 3D
- **体素稀疏卷积的持续价值**：在 Transformer 盛行的时代，体素 + 稀疏卷积仍是 3D 处理的高效基石
- **对 3DGS 场景理解的支持**：Easy3D 可直接嵌入 Gaussian Splatting 渲染管线

## 评分 ⭐⭐⭐⭐

方法简洁有效，工程设计优良。消融实验清晰展示了各组件的贡献。跨域泛化能力非常出色，尤其在 KITTI-360 和 GS 场景上。VR 应用演示增加了实用价值。

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates](supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)
- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)
- [\[ICCV 2025\] CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation](cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs](gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)

<!-- RELATED:END -->
