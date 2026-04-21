---
title: >-
  [论文解读] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow
description: >-
  [ICCV 2025][自动驾驶][占用网格估计] 提出 GaussianFlowOcc，用稀疏 3D Gaussian 分布替代密集体素网格进行占用估计，通过 Gaussian Transformer 高效建模场景，引入 Temporal Module 估计每个 Gaussian 的 3D 时序流处理动态物体，在 nuScenes 上以弱监督方式大幅超越现有方法（mIoU 提升 51%+），推理速度快 50 倍。
tags:
  - ICCV 2025
  - 自动驾驶
  - 占用网格估计
  - 3D Gaussian Splatting
  - 弱监督
  - 时序流
  - 稀疏表示
---

# GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow

**会议**: ICCV 2025  
**arXiv**: [2502.17288](https://arxiv.org/abs/2502.17288)  
**代码**: [GitHub](https://github.com/boschresearch/GaussianFlowOcc)  
**领域**: 自动驾驶  
**关键词**: 占用网格估计, 3D Gaussian Splatting, 弱监督, 时序流, 稀疏表示

## 一句话总结

提出 GaussianFlowOcc，用稀疏 3D Gaussian 分布替代密集体素网格进行占用估计，通过 Gaussian Transformer 高效建模场景，引入 Temporal Module 估计每个 Gaussian 的 3D 时序流处理动态物体，在 nuScenes 上以弱监督方式大幅超越现有方法（mIoU 提升 51%+），推理速度快 50 倍。

## 研究背景与动机

3D 语义占用估计是自动驾驶的核心任务，为车辆周围环境提供密集的体素级理解。但现有方法面临三个关键限制：

**密集 3D 标注依赖**：大多数占用估计模型需要昂贵的 3D 体素 ground truth 标注，这些标注通常来自 LiDAR 积累+人工校正，获取成本极高且难以大规模扩展。

**密集体素表示的低效性**：传统密集 3D 体素网格浪费大量计算资源在"空"体素上——真实驾驶场景中大部分 3D 空间是空区域。3D 卷积操作进一步加剧计算负担。

**弱监督方法忽视场景动态**：现有自/弱监督方法（如 SelfOcc, OccNeRF）通过时序渲染一致性训练，但未处理动态物体运动导致的时序不一致——运动物体在相邻帧位置不同，直接渲染会产生错误监督信号。

GaussianFlowOcc 的核心洞察：用稀疏 Gaussian 替代密集体素，不仅节省计算，还天然支持 Gaussian Splatting 进行高效 2D 渲染训练；同时通过学习每个 Gaussian 的 3D flow 来显式建模场景动态，解决弱监督中的时序不一致问题。

## 方法详解

### 整体框架

模型输入为多视角相机图像，经图像编码器提取特征后，Gaussian Transformer 迭代地将初始 Gaussian 更新为最终的 3D 场景表示。Gaussian Heads 预测每个 Gaussian 的属性（不透明度、尺度、旋转、语义），Temporal Module 估计到相邻帧的 3D flow。通过 Gaussian Splatting 渲染深度和语义图，用 2D 伪标签（GroundedSAM 语义 + Metric3D 深度）训练。

场景表示为 $N$ 个 Gaussian 的集合 $\mathcal{G} = \{G_1, \ldots, G_N\}$，每个 $G_i = (\mu, \sigma, s, r, c)$ 包含位置、不透明度、尺度、旋转和语义 logits。

### 关键设计

1. **Gaussian Transformer（高效 3D 场景建模）**：这是架构的核心创新。包含 $B$ 个迭代块，每个块依次执行：

    - **位置编码**：将前一块的 Gaussian 位置 $\mathcal{G}_\mu^{b-1}$ 通过 MLP 编码并加到特征上
    - **GICA（Gaussian-Image Cross-Attention）**：采用可变形交叉注意力（deformable cross-attention），将 Gaussian 位置投影到图像特征图上采样局部信息
    - **ISA（Induced Self-Attention）**：受 Set Transformer 启发，引入 $M$ 个可学习的诱导点 $P \in \mathbb{R}^{M \times D}$（$M \ll N$），将二次复杂度 $\mathcal{O}(N^2)$ 降低为 $\mathcal{O}(MN)$：
    $H = \text{MHA}(P, \mathcal{G}_f, \mathcal{G}_f), \quad \text{ISA}(\mathcal{G}_f) = \text{MHA}(\mathcal{G}_f, H, H)$
   这使得模型能处理 $N=10000$ 个 Gaussian，而标准注意力在 $N=5000$ 时就需 50GB 显存
    - **ITA（Induced Temporal Attention）**：类似 ISA 但用于时序信息传播，诱导点先聚合前一帧 Gaussian 特征，再让当前帧与之交互
    - **Gaussian Rectification**：MLP 估计位置残差 $\Delta\mathcal{G}_\mu^b$ 更新 Gaussian 位置

2. **Temporal Module（3D 时序流估计）**：解决弱监督中动态物体的关键问题。为每个 Gaussian 估计到每个时间步的 3D 位移：

$$\vec{v}(t) = \text{MLP}_v(\mathcal{G}_f \oplus \Psi(t))$$

其中 $\Psi \in \mathbb{R}^{2T \times D}$ 是每个时间步的可学习 time token。估计的偏移量加到 Gaussian 位置上，再用 Gaussian Splatting 渲染到对应时序帧。该模块**不需要额外损失或 ground truth 运动数据**——只有正确估计运动时，动态物体才能在时序帧中被正确渲染，从而通过已有渲染损失隐式学习。

3. **Temporal Gaussian Splatting（增强时序监督）**：在训练时加载相邻 $T$ 帧的相机参数和标签，将当前估计的 Gaussian（经 Temporal Module 位移校正后）渲染到这些时序视角，计算额外的 2D 渲染损失。这有效增加了视角重叠，弥补自动驾驶多相机视锥重叠小的问题。

### 损失函数 / 训练策略

- 渲染损失：深度 MSE 损失 $\mathcal{L}_{depth}$ + 语义二值交叉熵损失 $\mathcal{L}_{seg}$
- 2D 伪标签由 GroundedSAM（语义分割）和 Metric3D（深度估计）生成
- 时序 Gaussian Splatting horizon $T=6$
- ResNet-50 backbone，图像分辨率 $256 \times 704$
- $N=10000$ Gaussians，3个 Transformer 块，$M=500$ 诱导点
- 18 epochs 训练，4 块 A100 GPU
- 体素化为后处理步骤，仅用于 benchmark 评估

## 实验关键数据

### 主实验

**Occ3D-nuScenes 弱监督占用估计**

| 方法 | Backbone | mIoU↑ | IoU↑ | RayIoU↑ | FPS↑ |
|------|----------|-------|------|---------|------|
| SelfOcc | R50 | 10.54 | 45.01 | - | 1.15 |
| OccNeRF | R101 | 10.81 | 22.81 | - | 1.27 |
| GaussianOcc | R101 | 11.26 | - | 11.85 | 5.57 |
| GaussTR | 2×ViT | 13.26 | 45.19 | - | 0.20 |
| **GaussianFlowOcc** | **R50** | **17.08** | **46.91** | **16.47** | **10.2** |

相对改进：mIoU 比 GaussTR 提升 29%，比体素方法提升 51%+；推理速度比 GaussTR 快 50 倍。

### 消融实验

**Temporal Module 的影响**

| 配置 | mIoU | RayIoU | 说明 |
|------|------|--------|------|
| 无 Temporal Module | 14.18 | 14.46 | 时序不一致未处理 |
| **有 Temporal Module** | **17.08** | **16.47** | **+20% mIoU 提升** |

**注意力机制消融**

| Self-Attention | Temporal Attention | mIoU |
|---------------|-------------------|------|
| ✗ | ✗ | 13.81 |
| ✓ | ✗ | 14.60 |
| ✗ | ✓ | 14.47 |
| ✓ | ✓ | **17.08** |

**Gaussian 属性消融**

| Opacity | Scale | Rotation | mIoU | 说明 |
|---------|-------|----------|------|------|
| ✗ | ✗ | ✗ | 9.48 | 仅位置 |
| ✓ | ✗ | ✗ | 13.12 | +不透明度 |
| ✓ | ✓ | ✗ | 14.98 | +尺度 |
| ✓ | ✓ | ✓ | **17.08** | 全部属性 |

### 关键发现

- Temporal Module 带来 20% mIoU 提升，确认动态物体运动补偿是弱监督占用估计的关键瓶颈
- ISA 使模型能使用 10000 个 Gaussian，而标准注意力在 5000+ 时就导致显存爆炸/训练不收敛
- 轻量级 ResNet-50 backbone 即可超越使用 2×ViT-L backbone 的 GaussTR
- 3D Gaussian 表示对薄/扁平物体（交通标志、杆、行人）建模优势明显，不受体素分辨率限制
- 时序 horizon $T=6$ 是最优值，$T>8$ 时模型发散
- 不透明度是最重要的单一 Gaussian 属性（+3.64 mIoU）

## 亮点与洞察

- **稀疏替代密集的范式转变**：从密集体素到稀疏 Gaussian 的转变不仅是效率提升，更从根本上改变了场景理解的表示方式，Gaussian 的连续参数天然适合表达细粒度几何
- **巧妙的自监督动态建模**：Temporal Module 通过渲染一致性隐式学习运动，无需 GT flow 标注，是解决弱监督时序不一致的优雅方案
- **Induced Attention 的实用性**：将 Set Transformer 思想引入 3D 场景建模，使大规模 Gaussian 处理成为可能（$\mathcal{O}(MN)$ vs $\mathcal{O}(N^2)$）
- **推理极快**：10.2 FPS 比 GaussTR（0.2 FPS）快 50 倍，达到实用部署水平

## 局限与展望

- 2D 伪标签质量直接影响性能上限，基础模型（GroundedSAM、Metric3D）的误差会传播
- 时序 horizon 超过 8 时训练不稳定，限制了远距离时序信息的利用
- $N=10000$ 以上增加 Gaussian 数量不再带来提升，可能需要自适应密度控制
- 当前仅使用单帧输入，多帧输入融合可能进一步提升
- 与全监督方法仍有差距，可探索结合少量 3D 标注的半监督方案

## 相关工作与启发

- 与 GaussianFormer 和 GaussianWorld 共享稀疏 Gaussian 表示的思路，但首次引入时序流建模和弱监督训练
- Induced Attention 来自 Set Transformer，成功证明了其在 3D 视觉任务中的可行性
- Temporal Gaussian Splatting 的思路可推广到其他时序密集预测任务
- 为弱监督 3D 场景理解方法如何处理动态场景提供了标准方案

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 稀疏 Gaussian 占用估计+时序流+弱监督的完整框架，多项首创
- **实验充分度**: ⭐⭐⭐⭐ — nuScenes 上全面评估，消融极为详尽（注意力/属性/时序/参数量），但仅一个数据集
- **写作质量**: ⭐⭐⭐⭐⭐ — 论文结构清晰，每个贡献的实验支撑到位，图表质量高
- **价值**: ⭐⭐⭐⭐⭐ — 推理效率和弱监督特性使其具有极高实用价值，对自动驾驶感知系统架构设计有深远影响

<!-- RELATED:START -->

## 相关论文

- [6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](6dopegs_online_6d_object_pose_estimation_using_gaussian_spla.md)
- [AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)
- [GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [Self-Supervised Sparse Sensor Fusion for Long Range Perception](self-supervised_sparse_sensor_fusion_for_long_range_perception.md)
- [3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](3d_gaussian_splatting_driven_multiview_robust_physical_adver.md)

<!-- RELATED:END -->
