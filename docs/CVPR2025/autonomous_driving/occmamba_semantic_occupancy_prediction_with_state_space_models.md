---
title: >-
  [论文解读] OccMamba: Semantic Occupancy Prediction with State Space Models
description: >-
  [CVPR 2025][自动驾驶][语义占据] OccMamba 把 SSM/Mamba 引入户外语义占据预测，用 height-prioritized 2D Hilbert 展开把 3D voxel 序列化为 1D 序列，再用层次化 Mamba + 局部上下文处理器同时建模全局和局部，在 OpenOccupancy/SemanticKITTI/SemanticPOSS 上全面 SOTA，且显存远小于 transformer 方案。
tags:
  - CVPR 2025
  - 自动驾驶
  - 语义占据
  - Mamba
  - Hilbert 曲线
  - LiDAR-Camera 融合
  - 长程建模
---

# OccMamba: Semantic Occupancy Prediction with State Space Models

**会议**: CVPR 2025  
**arXiv**: [2408.09859](https://arxiv.org/abs/2408.09859)  
**代码**: https://github.com/USTCLH/OccMamba  
**领域**: 自动驾驶 / 3D 感知 / 语义占据预测  
**关键词**: 语义占据、Mamba、Hilbert 曲线、LiDAR-Camera 融合、长程建模

## 一句话总结
OccMamba 把 SSM/Mamba 引入户外语义占据预测，用 height-prioritized 2D Hilbert 展开把 3D voxel 序列化为 1D 序列，再用层次化 Mamba + 局部上下文处理器同时建模全局和局部，在 OpenOccupancy/SemanticKITTI/SemanticPOSS 上全面 SOTA，且显存远小于 transformer 方案。

## 研究背景与动机
1. **领域现状**：语义占据预测要为大规模 3D 体素 (上百万个 voxel) 输出占据 + 类别，是 AD/AR/机器人核心感知任务。
2. **现有痛点**：
    - 单模态 (LiDAR/Camera) 信息不足；多模态 CNN (M-CONet) 无法捕全局；
    - Transformer 类方法 (OccFormer/OccNet) 复杂度 $\mathcal{O}(N^2)$，体素数大时显存爆炸，需要 deformable attn / projection 牺牲精度。
3. **核心矛盾**：百万体素既需要全局建模，又需要可承受的算力。
4. **本文目标**：在线性复杂度内同时获得全局 + 局部 + 跨模态融合能力。
5. **切入角度**：Mamba (SSM) 已在 NLP / 点云中验证可线性复杂度 + 全局建模；但 Mamba 是 1D 序列模型，把 3D voxel 拍平到 1D 时空间邻接关系会被破坏。
6. **核心 idea**：设计专门面向"水平大、垂直短"的驾驶场景体素的展开策略——先在 z 轴优先排列，再在 xy 平面用 2D Hilbert 曲线连接，最大程度保留空间邻接性。

## 方法详解

### 整体框架
基于 M-CONet 拓扑：
- **多模态编码**：LiDAR 经稀疏卷积得 $\mathbf{V}_\mathcal{L}$，多视角图像经 ResNet+FPN+View Transformer 投影到 voxel 得 $\mathbf{V}_\mathcal{C}$，二者沿通道 concat 得 $\mathbf{V}_\mathcal{F}$。
- **OccMamba Encoder**：层次 Mamba 模块 (encoder-decoder + skip) → 局部上下文处理器 → 输出 $\mathbf{V}_\mathcal{P}$。
- **Occupancy Head**：coarse-to-fine 上采样 + MLP 预测每 voxel 类别。

### 关键设计

1. **Height-Prioritized 2D Hilbert 展开**

    - 功能：把 3D voxel 高质量地序列化成 1D，让相邻体素在序列中尽可能临近。
    - 核心思路：把 (x,y,z) 拆成 (xy 平面) + (z 轴)；从 z=0 出发按 z 轴生成"垂直串"，再按 2D Hilbert 曲线遍历 xy 平面把这些垂直串串联起来。形式上 $\mathbf{V} = \mathcal{R}_{1D\to 3D}(\mathcal{R}_{3D\to 1D}(\mathbf{V}))$。
    - 设计动机：驾驶场景 z 轴尺寸远小于 xy (扁平结构)，且高度信息蕴含强类别先验 (路面/植被/车辆通常对应不同 z 段)；优先在 z 轴聚集再 Hilbert 在 xy 上滑动，比 (a) XYZ 顺序、(b) ZXY 顺序、(c) 3D Hilbert 都更贴合该任务的几何先验，相邻体素的"序列距离"最短。

2. **Hierarchical Mamba Module (Encoder-Decoder)**

    - 功能：在多分辨率上聚合全局上下文。
    - 核心思路：每 group 包含 2 个 Mamba block + 下采样；decoder 对称含 Mamba block + 上采样 + skip connection。每个 Mamba block 内部为 LN→Linear→Conv1d→Silu→Selective SSM→门控 → Linear，输入序列长度 $L$ 复杂度 $\mathcal{O}(L)$。每个 Mamba block 前后都做 reorder/de-reorder 保持 voxel 拓扑。
    - 设计动机：占据预测要在大尺度 (整个场景) 和小尺度 (单车辆) 都准确，分层下采样 + 跳连同时获取多尺度信息。

3. **Local Context Processor**

    - 功能：补充 Mamba 全局建模容易忽略的局部细节。
    - 核心思路：把全局 Mamba 输出 $\mathbf{V}_\mathcal{M}$ 沿 xy 平面用多种窗口尺寸 $\{w_i\}$ + 步长 $\{s_i\}$ 切 patch，对每个 patch 单独跑 Mamba blocks；不同窗口的输出按通道 concat 后用 1×1×1 3D 卷积压缩。
    - 设计动机：不同尺寸的局部窗口对应不同物体尺度 (行人/车辆/卡车)，类似 multi-scale local attention。

### 损失函数 / 训练策略
- 标准的语义占据交叉熵 + Lovasz-Softmax + scale-invariant loss (与 M-CONet 一致)。
- 训练只动新增模块，重用 LiDAR/camera backbone 权重。

## 实验关键数据

### 主实验
**OpenOccupancy 验证集** (相机+LiDAR)：
| 方法 | IoU | mIoU |
|------|-----|------|
| MonoScene (C) | 18.4 | 6.9 |
| M-CONet (C+L) | ~30 | ~20 |
| Co-Occ (C+L, 之前 SOTA) | 31.2 | 22.5 |
| **OccMamba (C+L)** | **36.3** | **26.8** |

提升 **+5.1 IoU / +4.3 mIoU**。

**SemanticKITTI** (单模态/多模态)、**SemanticPOSS**：均创新 SOTA。

**显存比较** (Fig.1b)：体素数从 256³ 增长时，OccMamba 的 GPU 内存呈线性增长，transformer-based 方法呈二次增长，到 512×512×40 已 OOM 而 OccMamba 仍可训练。

### 消融实验
| 配置 | mIoU (OpenOccupancy) |
|------|-----|
| 用 XYZ 顺序展开 | 24.x |
| 用 ZXY 顺序 | 25.x |
| 3D Hilbert 曲线 | 25.5 |
| **Height-prioritized 2D Hilbert (本文)** | **26.8** |
| 去掉 Hierarchical Mamba | 显著下降 |
| 去掉 Local Context Processor | -1.x mIoU |

### 关键发现
- 展开策略对 Mamba 效果至关重要，差的展开方式会让 mIoU 掉 1-2 个点。
- 高度信息的 SSM 顺序 (z 优先) 在驾驶场景中最重要——因为类别 (路面/天空/车顶) 与 z 高度强相关。
- Mamba 让 OccMamba 可以**无压缩**地处理 voxel，相比 deformable attention 不需要选关键点，因此对遮挡推理更鲁棒。

## 亮点与洞察
- **第一个 outdoor 占据预测 Mamba 网络**，展示 SSM 在 3D 大规模感知任务上的可行性。
- **任务先验 driven 的展开策略**：不是简单照搬 3D Hilbert，而是深入分析场景几何 (扁平、z 类别强相关) 设计专用展开——这种"针对任务调整 SSM 序列化"的思想可迁移到任意 3D 任务 (语义分割、检测、补全)。
- **线性复杂度让无压缩多模态融合成为可能**：之前因显存问题不得不做 BEV/range 压缩，OccMamba 直接处理 dense voxel，遮挡场景的恢复更准。
- **Local Context Processor 是 Mamba 的有效补丁**：global SSM 在密集体素下仍倾向于忽略小范围细节，加 patch-Mamba 是一个简洁补救。

## 局限与展望
- 展开策略仍是手工设计，能否端到端学习展开顺序是开放问题。
- Mamba 顺序敏感、双向扫描代价较高；只用单向扫描可能漏向后依赖。
- 未在动态时序占据 (4D occupancy) 上扩展。
- 大模型规模未充分探索，仅在中等规模上验证。
- 改进方向：在 z 优先 + xy Hilbert 之外引入 learnable permutation 或 random shuffle augmentation，提升对未见城市场景的泛化。

## 相关工作与启发
- **vs Co-Occ (CVPR 24)**：Co-Occ 用 transformer + projection，OccMamba 用 SSM + 直接体素，IoU/mIoU 大幅领先。
- **vs PointMamba**：PointMamba 用 3D Hilbert 处理点云，OccMamba 针对 dense voxel 提出 z 优先展开，更贴合 occupancy。
- **vs OccFormer/OccNet**：transformer-based 同行，本文是显存与精度的双重 Pareto 改进。
- 启发：任何"维度大但分布不均"的 3D 任务都可借鉴"按物理优先方向 + Hilbert 串联"的展开思想。

## 评分
- 新颖性: ⭐⭐⭐⭐ Mamba 用于 occupancy 是新的且 reorder 设计有针对性
- 实验充分度: ⭐⭐⭐⭐ 三个 benchmark + 显存对比 + 充分消融
- 写作质量: ⭐⭐⭐⭐ 公式与图示清楚，Hilbert 展开图很直观
- 价值: ⭐⭐⭐⭐ 显著降低占据预测的显存门槛，可作为新一代 baseline

<!-- RELATED:START -->

## 相关论文

- [SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction](sdgocc_semantic_and_depth-guided_birds-eye_view_transformation_for_3d_multimodal.md)
- [O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)
- [Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)
- [M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)
- [GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction](rethinking_temporal_fusion_with_a_unified_gradient_descent_view_for_3d_semantic_.md)

<!-- RELATED:END -->
