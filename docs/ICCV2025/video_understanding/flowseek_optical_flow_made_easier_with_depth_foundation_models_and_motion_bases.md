---
title: >-
  [论文解读] FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases
description: >-
  [ICCV 2025][视频理解][光流估计] FlowSeek 将深度基础模型（Depth Anything V2）的先验知识和经典的低维运动参数化（motion bases）融入光流网络，在仅使用单张消费级 GPU 训练的条件下即可实现 SOTA 的跨数据集泛化性能。
tags:
  - ICCV 2025
  - 视频理解
  - 光流估计
  - 深度基础模型
  - 运动基
  - 低资源训练
  - 跨数据集泛化
---

# FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases

**会议**: ICCV 2025  
**arXiv**: [2509.05297](https://arxiv.org/abs/2509.05297)  
**代码**: [https://flowseek25.github.io/](https://flowseek25.github.io/)  
**领域**: 视频理解 / 光流估计  
**关键词**: 光流估计, 深度基础模型, 运动基, 低资源训练, 跨数据集泛化

## 一句话总结

FlowSeek 将深度基础模型（Depth Anything V2）的先验知识和经典的低维运动参数化（motion bases）融入光流网络，在仅使用单张消费级 GPU 训练的条件下即可实现 SOTA 的跨数据集泛化性能。

## 研究背景与动机

光流估计是计算机视觉的经典问题，近年来以 RAFT 为代表的迭代式深度网络取得了显著进步。然而，当前 SOTA 方法（如 SEA-RAFT、FlowFormer 等）的一大共同依赖在于：**需要大量高端 GPU 进行训练**——FlowFormer 使用 4×V100，SEA-RAFT 使用 8×3090。

这种对硬件的依赖导致两个核心问题：

**学术不平等**：仅拥有有限 GPU 资源的研究组无法复现和竞争

**方法论停滞**：过度依赖硬件暴力推进精度，可能忽视更本质的方法创新

作者受 DeepSeek（NLP 领域低资源训练范例）启发，认为**复用已有视觉基础模型的知识比从头训练更高效**。光流和深度在几何上密切相关——给定固定运动，像素在图像上的位移与其 3D 点的逆深度成正比。因此，深度基础模型中蕴含的丰富几何先验可以被"移植"到光流任务中。

**核心 idea**：将 Depth Anything V2 的特征和深度输出通过经典的 6-DOF 运动基（motion bases）转化为光流先验，注入 SEA-RAFT 框架中，以极低的训练成本获得 SOTA 精度。

## 方法详解

### 整体框架

FlowSeek 的架构建立在 SEA-RAFT 骨干之上，融合三个层面的技术：
1. **现代光流网络设计**（SEA-RAFT 的迭代精炼框架）
2. **深度基础模型**（Depth Anything V2，冻结参数）
3. **经典运动参数化**（来自 30 年前的低维运动基理论）

### 关键设计

#### 1. 深度基础模型特征注入

给定图像对 $\mathbf{I}_0, \mathbf{I}_1$，FlowSeek 用冻结的 Depth Anything V2 提取：
- **逆深度图** $\mathbf{D}_0, \mathbf{D}_1$
- **解码器末端特征** $\mathbf{\Phi}_0, \mathbf{\Phi}_1$（与深度高度相关的中间表示）

这些特征经 BottNeck 网络（三层 3×3 卷积，步长 2）下采样至 1/8 分辨率后，与原始特征提取器的输出拼接，形成增强特征：

$$\mathbf{F}_0^{\mathbf{\Phi}} = \text{FeatNet}(\mathbf{I}_0) \oplus \text{BottNeck}(\mathbf{\Phi}_0)$$

增强特征用于构建 4D 相关体（correlation volume），提升匹配质量。

#### 2. 低维运动基（Motion Bases）

这是 FlowSeek 最核心的创新点。经典理论指出，对于已知深度的静态场景，光流可以分解为 6 个基向量的线性组合（对应 3D 运动的 6 个自由度）。

作者将原始 6 基扩展为 **8 个无需焦距的基**：

$$\mathcal{B}_{\text{motion}} = \{\Delta_{\mathbf{T}x}, \Delta_{\mathbf{T}y}, \Delta_{\mathbf{T}z}, \Delta_{\mathbf{R}^1 x}, \Delta_{\mathbf{R}^2 x}, \Delta_{\mathbf{R}^1 y}, \Delta_{\mathbf{R}^2 y}, \Delta_{\mathbf{R} z}\}$$

其中平移基依赖深度 $\mathbf{D}_0$，旋转基仅依赖像素坐标。关键技巧在于通过拆分旋转基并假设 $f_x = f_y$，消除了对相机焦距的依赖。

这些基通过一个 **BasesNet** 网络提取特征，并与上下文特征拼接，为迭代精炼提供初始运动先验。

**设计动机**：虽然运动基理论仅对刚体运动成立，但它提供的初始猜测可大幅降低迭代精炼的负担，使模型更快收敛到正确光流。

#### 3. 上下文增强

深度图 $\mathbf{D}_0, \mathbf{D}_1$ 也可以送入 ContextNet，与图像一起提取更强的上下文特征：

$$\mathbf{C}, \mathbf{H}^0 = \text{ContexNet}(\mathbf{I}_0 \oplus \mathbf{D}_0 \oplus \mathbf{I}_1 \oplus \mathbf{D}_1)$$

### 损失函数 / 训练策略

沿用 SEA-RAFT 的混合 Laplace 分布建模。每次迭代的光流更新参数化为两个 Laplace 分布的混合，通过负对数似然最小化进行监督：

$$\mathcal{L}_{\mathcal{F}} = \sum_{j=0}^{\text{iters}} \gamma^{N-j}(-\log \mathcal{F}^j)$$

所有模型均在**单张 RTX 3090 GPU** 上训练，batch size 为 4-6。

## 实验关键数据

### 主实验

| 方法 | Extra Data | Sintel Clean↓ | Sintel Final↓ | KITTI Fl-EPE↓ | KITTI Fl-all↓ |
|------|-----------|--------------|--------------|--------------|--------------|
| RAFT | - | 1.43 | 2.71 | 5.04 | 17.4 |
| FlowFormer | - | 1.01 | 2.40 | 4.09 | 14.7 |
| SEA-RAFT (L) | - | 1.19 | 4.11 | 3.62 | 12.9 |
| FlowSeek (S) | - | 1.04 | 2.43 | 3.36 | 11.5 |
| FlowSeek (L) | - | 1.07 | 2.21 | 3.82 | 12.5 |
| SEA-RAFT (L) | Tartan | 1.23 | 3.37 | 3.73 | 12.7 |
| **FlowSeek (L)** | **Tartan** | **1.03** | **2.18** | **3.31** | **11.2** |

FlowSeek (L) 在 Sintel Final 上相对 SEA-RAFT (L) 提升 **35%**，在 KITTI 上提升 **12%**，且训练 batch size 仅为对手的 1/8。

### 消融实验

| 先验组合 | Φ | D | BaseNet | TartanAir EPE↓ | KITTI Fl-All↓ |
|---------|---|---|---------|---------------|--------------|
| SEA-RAFT (S) 基线 | - | - | - | 1.38 | 6.31 |
| 仅特征 Φ | ✓ | - | - | 1.30 | 5.69 |
| 仅深度 D | - | ✓ | - | 1.15 | 4.67 |
| **仅 BaseNet** | - | - | ✓ | **1.04** | **4.23** |
| **Φ + BaseNet（最优）** | ✓ | - | ✓ | **1.03** | **4.16** |
| 三者全用 | ✓ | ✓ | ✓ | 1.03 | 4.19 |

关键发现：BaseNet（运动基）是提升性能最关键的组件，单独使用即可将 EPE 从 1.38 降至 1.04。

### 关键发现

1. **运动基是核心**：BasesNet 对精度贡献最大，证明将深度先验转化为运动基比直接使用深度图更有效
2. **深度模型越新越好**：从 DPT → Depth Anything V1 → V2，性能单调提升
3. **通用性强**：在 CRAFT、FlowFormer 等不同骨干上加入 BaseNet 均有显著提升
4. **LayeredFlow 优势突出**：在含透明/反射表面的 LayeredFlow 数据集上，FlowSeek (L) 的 EPE 比 SEA-RAFT (L) 降低超过 2 像素

## 亮点与洞察

- **"以小博大"的范式**：复用已有基础模型知识，避免从头训练，实现了 1 GPU = 8 GPU 的竞争力
- **经典与现代的桥接**：30 年前的运动基理论 + 最新的深度基础模型，两者跨时代协同
- **设计空间的系统探索**：对先验组合、模型规模、深度模型选择、骨干兼容性的全面消融非常扎实

## 局限与展望

- 依赖预训练深度模型（其本身可能需要大量算力训练），某种程度上"转移"了计算开销
- 运动基理论仅对刚体运动成立，对剧烈非刚体运动场景效果有限
- 训练数据仍是瓶颈——更好的合成数据可能进一步提升效果

## 相关工作与启发

- **SEA-RAFT** [Wang et al., 2024]：本文直接构建于此，保留其迭代精炼和混合 Laplace 损失
- **Depth Anything V2** [Yang et al., 2024]：作为深度先验的来源，冻结使用
- **Stereo Anywhere** [Bartolomei et al., 2025]：类似思路将深度基础模型用于立体匹配
- 启发：其他几何视觉任务（场景流、位姿估计）也可沿用这种"基础模型先验 + 经典参数化"的范式

## 评分

- 新颖性：⭐⭐⭐⭐ — 运动基 + 深度基础模型的组合思路独特
- 技术深度：⭐⭐⭐⭐ — 消融全面、设计考量充分
- 实验充分度：⭐⭐⭐⭐⭐ — 多数据集、多骨干、多模型规模的系统评估
- 实用性：⭐⭐⭐⭐⭐ — 单 GPU 训练，极具实用价值

<!-- RELATED:START -->

## 相关论文

- [Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow](flow4agent_long-form_video_understanding_via_motion_prior_from_optical_flow.md)
- [Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View](prior-flow_enhancing_primitive_panoramic_optical_flow_with_o.md)
- [MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [Estimating 2D Camera Motion with Hybrid Motion Basis](estimating_2d_camera_motion_with_hybrid_motion_basis.md)

<!-- RELATED:END -->
