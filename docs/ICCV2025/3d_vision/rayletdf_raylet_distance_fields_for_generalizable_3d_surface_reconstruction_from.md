---
title: >-
  [论文解读] RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians
description: >-
  [ICCV 2025][3D视觉][Raylet Distance Field] 提出 RayletDF，一种基于"raylet"（光线片段）距离场的泛化3D表面重建方法，通过raylet特征提取器、距离场预测器和多raylet混合器三个模块，从点云或3D高斯直接预测表面点，在未见数据集上实现单次前向传播的高精度跨数据集泛化。
tags:
  - ICCV 2025
  - 3D视觉
  - Raylet Distance Field
  - Surface Reconstruction
  - 点云
  - 3DGS
  - Generalization
---

# RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians

**会议**: ICCV 2025  
**arXiv**: [2508.09830](https://arxiv.org/abs/2508.09830)  
**代码**: [https://github.com/vLAR-group/RayletDF](https://github.com/vLAR-group/RayletDF)  
**领域**: 3D视觉 / 表面重建 / 泛化表示  
**关键词**: Raylet Distance Field, Surface Reconstruction, Point Cloud, 3DGS, Generalization  

## 一句话总结
提出 RayletDF，一种基于"raylet"（光线片段）距离场的泛化3D表面重建方法，通过raylet特征提取器、距离场预测器和多raylet混合器三个模块，从点云或3D高斯直接预测表面点，在未见数据集上实现单次前向传播的高精度跨数据集泛化。

## 研究背景与动机

从RGB/D图像或点云恢复3D表面是混合现实、具身AI等应用的基础需求。现有方法各有局限：

- **坐标基方法**（OF、SDF、NeRF）：需要密集采样+网络评估才能获取显式表面，计算量大
- **3DGS**：实时渲染RGB但深度质量差，无法捕捉精细表面几何
- **射线基方法**（DRDF、PRIF、RayDF）：效率高但受限于Plucker/球面坐标的射线参数化，仅能表示物体级表面，且需逐场景训练

**核心洞察**：现有射线方法使用完整光线作为输入，无法捕捉局部几何模式。如果改用光线的**局部片段（raylet）**——起始点位于表面附近的单位光线段——则可以聚焦于精细的局部表面模式，而这些局部模式在不同形状间是**可泛化**的。

## 方法详解

### 核心概念：Raylet和Raylet Distance

- **Raylet** $\mathbf{l}$：光线的单位段，起始点采样在形状表面附近，参数化为起始点xyz + 方向单位向量，共6维
- **Raylet Distance** $d_l$：表面命中点与raylet起始点之间的有符号距离。正值表示命中点在起始点前方，负值表示在后方
- 关键优势：一条光线可在表面两侧采样多个raylet，每个raylet聚焦于局部表面模式

### 三模块流水线

**模块1：Raylet特征提取器**
- 用SparseConv对输入场景（点云或3D高斯）提取逐点特征 $\mathbf{F} \in \mathbb{R}^{N \times 32}$
- 对查询raylet $\mathbf{l}$，通过KNN找到最近的K个点，聚合邻域信息：
$$\hat{\mathbf{f}}_l^k = \left(\mathbf{p}_l^k \oplus \frac{(\mathbf{p}_l^k - \mathbf{p}_l)}{\|\mathbf{p}_l^k - \mathbf{p}_l\|} \oplus \|\mathbf{p}_l^k - \mathbf{p}_l\|\right) \oplus \mathbf{f}_l^k$$
- 关键洞察：提取的特征保留了表面附近的局部几何模式，使学到的表示可跨场景泛化

**模块2：Raylet距离场预测器**
- 8层MLP（每层256隐藏单元），输入raylet位置+方向+特征，输出距离值和置信度分数：
$$(d_l, s_l) = MLPs(\mathbf{p}_l \oplus \mathbf{u}_l \oplus \mathbf{f}_l)$$
- 无需沿光线密集采样坐标，一次预测即得表面距离

**模块3：多Raylet混合器**
- 沿同一光线采样T个raylet（相同方向，不同起始点），并行预测距离
- 通过softmax加权融合多个预测：
$$D = \sum_{t=1}^T \hat{s}_{l_t}\left(\|\mathbf{p}_{cam} - \mathbf{p}_{l_t}\| + d_{l_T}\right), \quad \hat{s}_{l_t} = \frac{e^{s_{l_t}}}{\sum_{t=1}^T e^{s_{l_t}}}$$
- 多raylet融合提升了泛化性和鲁棒性

### Raylet采样策略
- 对点云：为每个点构建虚拟球（半径=最近点距离），场景表面由所有虚拟球的并集界定
- 光线穿过的虚拟球交点投影到光线上，选择垂直距离最短的Top-T个交点作为raylet起始点
- 对3DGS：计算光线与高斯的交点，按alpha blending贡献选择Top-T个

### 训练损失
- 使用 $\ell_1$ 损失监督预测距离D，GT从深度图转换

## 实验

### 主实验：跨数据集泛化（训练在ARKitScenes）

| 方法 | 类型 | ARKitScene ADE↓ | ScanNet/++ ADE↓ | MultiScan ADE↓ |
|------|------|----------------|----------------|---------------|
| 3DGS | 逐场景 | 0.268 | 0.321 | 0.431 |
| PGSR | 逐场景 | 0.219 | 0.202 | 0.315 |
| DepthAnythingV2 | 对齐 | 0.206 | 0.168 | 0.228 |
| Pointersect | 泛化 | 0.286 | 0.366 | 0.266 |
| RayDF | 泛化 | 0.183 | 0.227 | 0.326 |
| **RayletDF** | **泛化** | **0.115** | **0.175** | **0.216** |

### 消融：关键组件影响（δ指标）

| 消融项 | ARKit δ↑ | ScanNet++ δ↑ |
|--------|----------|-------------|
| RayletDF完整 | **0.928** | **0.894** |
| 无多Raylet混合 | 0.908 | 0.847 |
| 无置信度分数 | 0.921 | 0.882 |
| K=4 (vs K=16) | 0.916 | 0.870 |

**关键发现**：
- RayletDF在ARKitScene上ADE降低37%（0.183→0.115 vs RayDF），跨数据集泛化尤其突出
- 即使训练在ARKit上，在完全未见的ScanNet++和MultiScan上仍显著优于所有泛化基线
- 多Raylet混合是关键（去除后δ下降2%），置信度权重进一步提升精度
- 同时支持从点云和3DGS两种输入形式进行重建
- 将发布7770个3D场景的高斯数据（ScanNet/++、ARKit、MultiScan）

## 亮点与洞察
1. **Raylet概念精妙**：将射线分解为聚焦局部模式的片段，是实现泛化的关键——局部几何模式在不同场景间共享
2. **免密集采样**：相比SDF/OF需沿光线密集采样坐标，RayletDF一次预测即得表面距离
3. **统一输入**：同一流水线无缝处理点云和3DGS两种输入
4. **法线的闭式推导**：射线基公式允许直接推导法向量，无需额外网络

## 局限性
- SparseConv骨干需要体素化，内存消耗随场景规模增长
- 在远离点云表面的查询光线上无法产生预测（被丢弃）
- 跨数据集泛化在MultiScan上仍有一定精度差距
- 未探索利用法线作为额外正则化或离群点过滤

## 相关工作
- 坐标基方法：OF、SDF、UDF需密集采样
- 射线基方法：RayDF、PRIF仅处理物体级形状
- 深度估计：DepthAnythingV2高质量但缺乏跨帧一致性

## 评分
- **创新性**: ★★★★★ — Raylet距离场是优雅且有效的新表示
- **实用性**: ★★★★☆ — 泛化重建在下游应用（AR/机器人）中价值大
- **实验**: ★★★★★ — 跨多数据集评估全面，消融详尽
- **写作**: ★★★★☆ — 概念阐述清晰，图示直观

<!-- RELATED:START -->

## 相关论文

- [SVG-Head: Hybrid Surface-Volumetric Gaussians for High-Fidelity Head Reconstruction and Real-Time Editing](svg-head_hybrid_surface-volumetric_gaussians_for_high-fidelity_head_reconstructi.md)
- [GAP: Gaussianize Any Point Clouds with Text Guidance](gap_gaussianize_any_point_clouds_with_text_guidance.md)
- [Implicit Filtering for Learning Neural Signed Distance Functions from 3D Point Clouds](../../ECCV2024/3d_vision/implicit_filtering_for_learning_neural_signed_distance_functions_from_3d_point_c.md)
- [Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction](timeformer_capturing_temporal_relationships_of_deformable_3d_gaussians_for_robus.md)

<!-- RELATED:END -->
