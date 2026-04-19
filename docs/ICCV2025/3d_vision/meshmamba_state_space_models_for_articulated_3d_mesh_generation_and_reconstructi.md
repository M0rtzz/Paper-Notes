---
title: >-
  [论文解读] MeshMamba: State Space Models for Articulated 3D Mesh Generation and Reconstruction
description: >-
  [ICCV2025][3D视觉][Mamba] MeshMamba 提出基于 Mamba 状态空间模型的 3D 关节体网格生成与重建方法，通过设计基于身体部位 UV 图和模板网格坐标的顶点序列化技术，实现了万级顶点网格的高效生成和重建，速度比 Transformer 快 6-9 倍。
tags:
  - ICCV2025
  - 3D视觉
  - Mamba
  - state space model
  - 3D mesh generation
  - human mesh recovery
  - vertex serialization
---

# MeshMamba: State Space Models for Articulated 3D Mesh Generation and Reconstruction

**会议**: ICCV2025  
**arXiv**: [2507.15212](https://arxiv.org/abs/2507.15212)  
**代码**: -  
**领域**: 3d_vision  
**关键词**: Mamba, state space model, 3D mesh generation, human mesh recovery, vertex serialization  

## 一句话总结

MeshMamba 提出基于 Mamba 状态空间模型的 3D 关节体网格生成与重建方法，通过设计基于身体部位 UV 图和模板网格坐标的顶点序列化技术，实现了万级顶点网格的高效生成和重建，速度比 Transformer 快 6-9 倍。

## 研究背景与动机

- **问题定义**：生成和重建多样体型、姿态的 3D 关节体网格模型是计算机视觉和图形学的关键问题
- **现有方法局限**：
    - **参数化方法**（SMPL/SMPL-X）：表示紧凑但受限于预定义人体模型，难以捕捉复杂变形（如衣服）
    - **基于顶点的方法**：直接操作网格顶点，更灵活，但 Transformer 架构面临 $O(n^2)$ 复杂度瓶颈
    - 现有顶点级 Transformer 方法通常只能处理约 500 个顶点的粗分辨率网格，需额外上采样，丢失局部几何细节
    - 因此当前方法局限于仅身体姿态重建，无法包含手部姿态和面部表情
- **动机**：Mamba SSM 具有近线性复杂度和高效推理速度，是处理大量顶点 token 的理想选择。核心挑战在于如何将网格顶点序列化为 Mamba 易处理的 1D 序列

## 方法详解

### 整体框架

MeshMamba 由三个核心组件构成：
1. **顶点序列化**（Vertex Serialization）：将网格顶点序列化为有序 1D 序列
2. **MambaDiff3D**：基于 MeshMamba 的去噪扩散模型，用于 3D 关节体网格生成
3. **Mamba-HMR**：基于 MeshMamba 的人体网格重建模型，从单张图像恢复 3D 人体

### 关键设计一：顶点序列化

与 Transformer 不同，Mamba 需要有序输入序列。现有 3D 点云的 z-order 和 Hilbert 曲线排序不适用于从随机噪声或图像开始的网格生成任务。

**两种序列化策略**：
1. **基于 DensePose IUV 图**：按 I 分割图排序（24 个身体部位），部位内按 U、V 图排序
2. **基于模板网格 3D 坐标**：按 T-pose 模板的坐标轴排序（如先按 x 轴，相同则按 y，再按 z）

**多策略组合**：生成 6 种序列化变体（"xyz"、"-xyz"、"yzx"等），在不同 Mamba 层间交替使用。实验发现使用 **两种策略的组合** 最佳平衡效率和质量。除一层外的所有层使用一种策略，剩余一层使用另一种。

### 关键设计二：MambaDiff3D

**网络结构**：受 U-ViT 启发，$L+1$ 层 Mamba 块 + 输入/输出 MLP。前半 $L/2$ 块（浅层）、中间块、后半 $L/2$ 块（深层），浅层与深层间有跳跃连接。时间嵌入通过加法注入每个 Mamba 块。

**训练损失**（v-prediction 参数化 + 余弦方差调度器）：

$$L = \mathbb{E}_{t, \mathbf{x}_0, \epsilon} \; w_t \|\epsilon - \epsilon_\theta(\mathbf{x}_t, t)\|_2^2, \quad w_t = e^{-\lambda_t / 2}$$

**采样**：使用 DDIM 采样器，扩散步数 $T=1000$，采样步数可选 50/100/250。

**顶点+法线联合生成**：针对顶点生成易有局部噪声、Jacobian 生成易全局畸变的问题，同时生成顶点位置和法向量，通过求解 Poisson 系统将二者融合，实现平滑重建的同时保持表面细节。

### 关键设计三：Mamba-HMR

替换 Mesh Transformer 中的 Transformer 块为 MeshMamba 块。将 CNN 图像特征作为输入，包含关节查询和顶点查询 + 位置编码。

**关键优势**：直接输出全分辨率网格（10475 顶点），无需上采样器，大幅减少模型参数。

**训练损失**：

$$L = \lambda_{3D}^V L^V + \lambda_{3D}^J (L_{3D}^J + L_{reg3D}^J) + \lambda_{2D}^J (L_{2D}^J + L_{reg2D}^J) + \lambda_{edge} L_{edge} + \lambda_{lap} L_{lap} + \lambda_{normal} L_{normal}$$

其中 $L_{edge}$、$L_{lap}$、$L_{normal}$ 分别为边长度、Laplacian、法向量的局部几何正则化损失，对保持稠密网格的局部形状至关重要。

## 实验关键数据

### 主实验：3D 人体生成

| 方法 | 训练集 | 1-NNA [%] ↓ | FID ↓ | APD ↑ |
|------|--------|-------------|-------|-------|
| Pose-NDF | AMASS | 92.0 | 3.92 | 37.81 |
| NRDF | AMASS | 81.6 | 0.64 | 23.12 |
| VPoser | AMASS | 60.7 | 0.05 | 14.68 |
| DiffSurf | SURREAL | 54.4 | - | - |
| **MambaDiff3D** | SURREAL | **53.1** | 0.32 | 23.01 |
| **MambaDiff3D** | AMASS | 55.1 | **0.22** | **23.8** |

**全身人体网格重建（UBody 数据集）**：

| 方法 | PA-MVE All ↓ | PA-MVE Hands ↓ | PA-MVE Face ↓ | MVE All ↓ | FPS |
|------|-------------|----------------|---------------|-----------|-----|
| SMPLer-X-L† | 31.9 | 10.3 | 2.8 | 57.4 | 24 |
| AiOS | 32.5 | 7.3 | 2.8 | 58.6 | - |
| Multi-HMR-B | 31.4 | 9.8 | 6.1 | 65.1 | 23 |
| **Mamba-HMR†** | **25.9** | **9.7** | **2.1** | **51.7** | 22 |

### 消融实验

**网络块类型对比**（1-NNA ↓）：

| 块类型 | 1-NNA [%] ↓ |
|--------|-------------|
| MLP | 73.7 |
| GNN | 74.2 |
| Transformer | 53.6 |
| **Mamba** | **53.1** |

**序列化策略对比**：

| 序列化方式 | 1-NNA [%] ↓ |
|------------|-------------|
| SMPL connectivity ×1 | 60.0 |
| part-IUV ×1 | 54.4 |
| part-IUV ×2 | 53.7 |
| SMPL ×1 + XYZ ×1 | **53.1** |

### 关键发现

1. **推理速度优势**：在 A100 上生成 10475 顶点网格，Mamba 需 4.5s（250 步），Transformer 需 28.1s，**快 6 倍**；V100 上快 9 倍
2. 训练时间 Mamba 约 18 min/epoch vs Transformer 约 100 min/epoch（6890 顶点）
3. 随机排序的序列化完全无法训练，验证了有序序列化的必要性
4. 两种策略组合最佳平衡效率和质量，增加更多策略提升有限但增加推理时间
5. 顶点+法向量联合生成优于纯顶点或纯 Jacobian 生成

## 亮点与洞察

1. **首次将 Mamba 引入 3D 网格生成与重建**：巧妙利用 SSM 的近线性复杂度解决 Transformer 在大量顶点上的效率瓶颈
2. **序列化设计精巧**：利用人体身体部位结构和模板网格的先验知识，确保序列化尊重关节体形状的结构
3. **可扩展性突破**：首次实现 10000+ 顶点的稠密网格生成，能捕捉衣服变形和抓握手势
4. **端到端全分辨率输出**：Mamba-HMR 无需下采样-上采样，减少信息损失
5. **梯度域网格表示创新**：在非端到端框架中结合位置和法线生成，通过 Poisson 系统实现平滑重建

## 局限性

1. 仅限于固定拓扑结构的贴身衣物，未能处理宽松衣物的拓扑变化
2. 对未在训练集中出现的新数据集泛化能力有限
3. 非端到端生成流程（生成 + Poisson 重建分离），可能限制整体优化
4. Mamba-HMR 在手部和面部的重建精度仍有改进空间

## 相关工作与启发

- **Mamba 在视觉中的应用**：ViM（图像）→ DiS（图像生成）→ 3D 点云分析 → **本文首次用于 3D 网格**
- **Transformer vs Mamba 的权衡**：当 token 数量大时（>5000），Mamba 的效率优势显著，这对需要高分辨率输出的任务很有启发
- **序列化策略的通用思路**：利用领域先验知识（身体部位结构）进行序列化排序，可推广到其他需要将结构化数据喂入序列模型的场景
- **梯度域方法在生成中的应用**：Poisson 重建结合生成模型是一个值得深入探索的方向

## 评分 ⭐⭐⭐⭐

首次将 Mamba 引入 3D 网格生成与重建，方法新颖，序列化设计合理。效率提升显著（6-9 倍速度提升），在生成和重建任务上均达到 SOTA。实验充分，消融分析细致。局限在于拓扑固定和泛化能力，但作为开创性工作完成度高。

<!-- RELATED:START -->

## 相关论文

- [Global-Aware Monocular Semantic Scene Completion with State Space Models](global-aware_monocular_semantic_scene_completion_with_state_space_models.md)
- [UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling](ust-ssm_unified_spatio-temporal_state_space_models_for_point_cloud_video_modelin.md)
- [Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes](../../CVPR2025/3d_vision/mesh_mamba_a_unified_state_space_model_for_saliency_prediction_in_non-textured_a.md)
- [Nautilus: Locality-aware Autoencoder for Scalable Mesh Generation](nautilus_locality-aware_autoencoder_for_scalable_mesh_generation.md)
- [Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)

<!-- RELATED:END -->
