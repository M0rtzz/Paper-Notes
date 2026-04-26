---
title: >-
  [论文解读] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction
description: >-
  [CVPR 2026][自动驾驶][全向占用预测] 首个纯视觉、端到端的全向开放词汇占用预测框架 O3N，通过极坐标螺旋 Mamba (PsM)、占用代价聚合 (OCA) 和自然模态对齐 (NMA) 三个核心模块，在 360° 全景图像输入下实现了超越闭集监督方法的开放词汇 3D 占用预测性能。
tags:
  - CVPR 2026
  - 自动驾驶
  - 全向占用预测
  - 开放词汇
  - Mamba
  - 对比学习
  - 全景感知
---

# O3N: Omnidirectional Open-Vocabulary Occupancy Prediction

**会议**: CVPR 2026  
**arXiv**: [2603.12144](https://arxiv.org/abs/2603.12144)  
**代码**: 即将开源  
**领域**: 自动驾驶 / 3D场景理解  
**关键词**: 全向占用预测, 开放词汇, Mamba, 对比学习, 全景感知

## 一句话总结

首个纯视觉、端到端的全向开放词汇占用预测框架 O3N，通过极坐标螺旋 Mamba (PsM)、占用代价聚合 (OCA) 和自然模态对齐 (NMA) 三个核心模块，在 360° 全景图像输入下实现了超越闭集监督方法的开放词汇 3D 占用预测性能。

## 研究背景与动机

**领域现状**：3D 语义占用预测已成为自动驾驶和具身智能的核心感知任务，现有方法如 MonoScene、VoxFormer、SGN 等在闭集设定下取得了显著进展。同时全景/全向图像因其单帧 360° 覆盖的优势，正被越来越多地用于具身智能体的场景理解。

**现有痛点**：(1) 现有 3D 占用预测方法局限于有限视角输入和预定义训练类别分布，难以应用于需要全面安全感知的开放世界场景；(2) 等距柱投影 (ERP) 引入的严重几何畸变和非均匀采样使得远距区域在图像中占比极小；(3) "像素-体素-文本"三元特征对齐在不均匀数据分布下容易过拟合，导致新类语义对齐失败。

**核心矛盾**：全向图像的几何畸变特性与开放词汇语义对齐的精度需求之间存在根本冲突——ERP 投影使远处区域像素稀疏，加剧了跨模态特征对齐中的过拟合风险。

**本文目标**：在全向视觉输入下，如何同时解决 360° 空间连续性建模、开放词汇语义泛化和跨模态特征对齐三大挑战。

**切入角度**：(1) 利用极坐标螺旋扫描适配全景几何；(2) 构建体素-文本代价体积替代直接特征对齐；(3) 通过无梯度随机游走弥合模态差距。

**核心 idea**：将全向感知的几何特性融入体素表示、代价聚合和模态对齐的全流程设计，实现首个全向开放词汇占用预测框架。

## 方法详解

### 整体框架

O3N 以全景等距柱投影 (ERP) 图像为输入，经过四个核心阶段：(1) 视觉特征提取器提取全向图像特征；(2) 2D-to-3D 视角变换生成立方体和柱坐标两种体素表示；(3) 增强了 PsM 模块的 3D 解码器学习细粒度空间几何和语义信息；(4) 占用预测头输出最终结果。在开放词汇路径上，OCA 和 NMA 模块保证"像素-体素-文本"三元语义一致性。

### 关键设计

1. **极坐标螺旋 Mamba (PsM) 模块**:
    - 功能：捕获全向图像固有空间结构中的长程依赖，解决柱坐标体素在极点附近的数据不连续性问题
    - 核心思路：采用双分支架构，将柱坐标体素 $\mathbf{V}_p \in \mathbb{R}^{C \times R \times P \times Z}$ 压缩为 BEV 特征 $\mathbf{B}_p \in \mathbb{R}^{C \times R \times P}$，然后以螺旋路径从极点向外逐步扫描（P-SMamba），与全景成像中从近到远的信息密度变化天然对齐
    - 坐标融合：在每层将极坐标体素重采样到笛卡尔空间，通过 $\mathbf{V}_f^i = \mathbf{V}_c^i + \Phi_{\rho(c)}(\mathbf{V}_p^i)$ 聚合两种坐标系的互补优势
    - 设计动机：标准 3D 卷积无法处理柱坐标极点不连续性，Transformer 计算开销过大，Mamba 的线性复杂度和长序列建模能力恰好适配此需求

2. **占用代价聚合 (OCA) 模块**:
    - 功能：构建体素-文本代价体积并进行空间和类别维度的聚合，避免直接离散特征对齐导致的过拟合
    - 核心思路：计算体素嵌入 $\mathbf{V}$ 与文本嵌入 $\mathbf{T}$ 之间的余弦相似度作为占用代价 $C(i,l) = \frac{V_i \cdot T_l}{\|V_i\| \|T_l\|}$，生成粗糙的 3D 语义掩码；然后通过 ASPP 进行空间聚合，通过线性 Transformer 进行类别聚合
    - 监督策略：使用场景亲和度损失 $\mathcal{L}_{oca}$ 捕获体素间语义关联，包含 Precision、Recall 和 Specificity 三个分量，训练时仅计算基类体素的损失
    - 设计动机：直接交叉熵监督会导致孤立的体素语义映射，削弱泛化能力；代价聚合保留了语义关系的连续性

3. **自然模态对齐 (NMA)**:
    - 功能：无梯度地弥合文本嵌入与语义原型之间的模态差距，防止对已见语义的过度依赖
    - 核心思路：通过 EMA 更新基类原型 $\mathbf{P}_t^b = \alpha \cdot \mathbf{P}_{t-1}^b + (1-\alpha) \cdot \frac{1}{|\Omega_b|}\sum_{i \in \Omega_b} \mathbf{f}_{seg}(i)$，然后基于随机游走迭代聚合文本嵌入与原型
    - 收敛形式：利用 Neumann 级数推导闭式解 $\mathbf{T}_t^\infty = (1-\beta)(\mathbf{I} - \beta^2 \mathcal{A})^{-1}(\beta \mathcal{S} \mathbf{P}_t^0 + \mathbf{T}_t^0)$
    - 设计动机：基于学习的对齐策略会过度拟合已见语义分布，无梯度方式可保持对无限语义的理解能力

### 损失函数 / 训练策略

总损失由三部分组成：$\mathcal{L} = \mathcal{L}_{occ} + \mathcal{L}_{vox-pix} + \mathcal{L}_{oca}$

- $\mathcal{L}_{occ}$：来自 MonoScene 的语义占用监督，包含交叉熵损失、语义/几何场景亲和度损失和 frustum proportion 损失
- $\mathcal{L}_{vox-pix}$：来自 OVO 的体素-像素特征对齐损失
- $\mathcal{L}_{oca}$：占用代价聚合的场景亲和度损失（仅计算基类体素）

推理时基类使用占用头直接预测，新类通过体素嵌入与文本嵌入的相似度结合 OCA 预测概率进行预测。

## 实验关键数据

### 主实验 (QuadOcc 验证集)

| 方法 | 输入 | mIoU | Novel mIoU | Base mIoU |
|------|------|------|------------|-----------|
| MonoScene (全监督) | C | 19.19 | 25.56 | 12.82 |
| OneOcc (全监督) | C | 20.56 | 27.53 | 13.59 |
| OVO (开放词汇) | C | 14.33 | 18.15 | 10.52 |
| **O3N (Ours)** | C | **16.54** | **21.16** | **11.92** |
| O3N 增量 vs OVO | - | +2.21 | +3.01 | +1.40 |

### 消融实验 (QuadOcc)

| 配置 | mIoU | Novel mIoU | Base mIoU |
|------|------|------------|-----------|
| Baseline (OVO) | 14.33 | 18.15 | 10.52 |
| + PsM | 15.21 | 19.43 | 11.00 |
| + PsM + OCA | 15.89 | 20.31 | 11.48 |
| + PsM + OCA + NMA (Full) | **16.54** | **21.16** | **11.92** |

### 关键发现

- O3N 在开放词汇设定下（mIoU 16.54）甚至超过了部分全监督方法（如 SSCNet 14.60、LMSCNet 18.44）
- 新类别 mIoU 从 18.15 提升至 21.16（+3.01），验证了开放词汇能力的显著增强
- 框架具有通用性：在 SGN-S 骨干上同样实现了从 13.81 到 15.52 的提升
- 在 Human360Occ 数据集上同样取得 SOTA，验证了跨场景泛化能力

## 亮点与洞察

- **首创性**：首次定义并解决全向开放词汇占用预测任务，将全景感知与开放语义预测统一
- **几何感知设计**：PsM 的螺旋扫描路径与全景成像的信息密度分布天然匹配，是一种优雅的 inductive bias
- **理论优美**：NMA 通过随机游走推导出闭式解，既避免了梯度优化的过拟合风险，又保证了收敛
- **模块化通用性**：O3N 可插拔地应用于 MonoScene、SGN 等多种占用预测架构

## 局限与展望

- 新类别的绝对性能仍较低（vehicle 仅 0.52 mIoU），极端稀有类的识别仍具挑战性
- 基类性能在开放词汇设定下有所下降（vs 全监督方法），说明开放词汇能力与闭集精度之间仍存在 trade-off
- 仅在全景数据集上验证，未探索对多视角相机输入的扩展
- 计算开销分析不充分，实际部署的实时性需要进一步评估

## 相关工作与启发

- **OVO** (Tan et al., 2023)：开放词汇占用预测的先驱，本文在其基础上引入全向感知和代价聚合改进
- **CAT-Seg** (Cho et al., 2024)：2D 开放词汇分割中的代价聚合思想被本文扩展到 3D 体素空间
- **OneOcc** (Shi et al., 2025)：全向全监督占用预测的 SOTA，为本文提供了基线和数据集
- **启发**：代价聚合替代直接特征对齐这一思路值得在其他跨模态任务中推广

## 评分 (⭐星级)

| 维度 | 评分 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐ |
| **综合** | **⭐⭐⭐⭐** |

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)](monocular_open_vocabulary_occupancy_prediction_for_indoor_scenes.md)
- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[CVPR 2026\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction.md)
- [\[CVPR 2026\] ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction](proood_prototype-guided_out-of-distribution_3d_occupancy_prediction.md)
- [\[CVPR 2026\] An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)

<!-- RELATED:END -->
