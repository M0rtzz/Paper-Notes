---
title: >-
  [论文解读] PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes
description: >-
  [CVPR 2025][预训练][平面重建] 本文提出 PlanarSplatting，通过直接优化可学习的 3D 矩形平面基元，利用新设计的矩形 splatting 函数将平面可微地渲染为深度和法线图，仅需 3 分钟即可从多视角图像重建精确的室内平面场景，无需任何平面标注。 领域现状：室内平面 3D 重建是计算机视觉中的经…
tags:
  - "CVPR 2025"
  - "预训练"
  - "平面重建"
  - "可微渲染"
  - "3D平面基元"
  - "室内场景"
  - "高斯渲染"
---

# PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes

**会议**: CVPR 2025  
**arXiv**: [2412.03451](https://arxiv.org/abs/2412.03451)  
**代码**: 无（CUDA 实现将公开）  
**领域**: 视频理解  
**关键词**: 平面重建, 可微渲染, 3D平面基元, 室内场景, 高斯渲染

## 一句话总结

本文提出 PlanarSplatting，通过直接优化可学习的 3D 矩形平面基元，利用新设计的矩形 splatting 函数将平面可微地渲染为深度和法线图，仅需 3 分钟即可从多视角图像重建精确的室内平面场景，无需任何平面标注。

## 研究背景与动机

**领域现状**：室内平面 3D 重建是计算机视觉中的经典问题。传统方法从已有的点云/网格拟合平面，学习方法如 PlanarRecon 和 AirPlanes 需要在大量 2D/3D 平面标注上训练检测、匹配、追踪的 pipeline。

**现有痛点**：（1）PlanarRecon 等方法将平面当作视觉特征处理，但平面是区域性的而非局部的，结果通常粗糙且丢失细节；（2）现有学习方法依赖 2D/3D 平面标注，但大规模平面标注获取困难，限制了性能和扩展性；（3）传统两步法（先重建网格，再 RANSAC 拟合平面）速度慢且依赖中间表示的质量。

**核心矛盾**：要获得准确完整的平面重建，传统方法需要先有高质量的 3D 几何（点云/网格），而学习方法需要大量平面标注。两者都不够高效。

**本文目标**：设计一种无需平面标注、直接从多视角图像优化 3D 平面基元的方法，兼顾速度和精度。

**切入角度**：借鉴 3DGS 的可微渲染思路——将 3D 基元通过可微 splatting 渲染到 2D，用梯度下降优化。但 3D 高斯的圆形/椭圆形不适合矩形平面。

**核心 idea**：设计矩形 3D 平面基元（中心、旋转、双向半径）和配套的矩形 splatting 函数（基于 Sigmoid 而非高斯），利用单目深度/法线基础模型作为伪监督，直接优化平面基元来拟合场景几何。

## 方法详解

### 整体框架

输入为多视角位姿图像，用单目深度模型（Metric3Dv2）初始化约 2000 个 3D 矩形平面基元。通过可微平面渲染将基元 splatting 为深度图和法线图，以 Metric3Dv2 预测的深度和 Omnidata 预测的法线作为伪标签监督优化。5000 次迭代后合并相似平面得到最终平面实例。

### 关键设计

1. **可学习矩形平面基元表示**:

    - 功能：用显式参数表达 3D 矩形平面，支持位置、朝向、形状的梯度优化
    - 核心思路：每个平面基元 $\pi$ 由三组参数定义：中心 $\mathbf{p}_\pi \in \mathbb{R}^3$、旋转四元数 $\mathbf{q}_\pi \in \mathbb{R}^4$、双向半径 $\mathbf{r}_\pi = \{r^{x+}, r^{x-}, r^{y+}, r^{y-}\} \in \mathbb{R}_+^4$。双向半径设计允许矩形中心不对称，提供更灵活的形状拟合能力。法线由旋转自动确定 $\mathbf{n}_\pi = \mathbf{R}(\mathbf{q}_\pi)[0,0,1]^\top$。
    - 设计动机：3DGS 的高斯基元是圆形/椭圆形，边界模糊，不适合表达矩形墙壁、地板等平面。双向半径使一个基元就能表达 L 形等非对称形状，减少所需基元数量。

2. **矩形 Plane Splatting 函数**:

    - 功能：将 3D 矩形平面基元可微地投影到像素空间，计算精确的矩形边界权重
    - 核心思路：先计算光线与平面的交点 $\mathbf{x}_\pi^\mathbf{r}$，再将交点投影到平面局部坐标系得到 $\mathcal{P}_X, \mathcal{P}_Y$。权重用 Sigmoid 函数计算：$w_X = 2\sigma(5\lambda(r^{x+} - |\mathcal{P}_X|))$（当 $\mathcal{P}_X > 0$），其中 $\lambda$ 随迭代指数增长（最大 300），使得权重从平滑过渡逐渐逼近矩形的硬边界。最终权重取 $w = \min(w_X, w_Y)$。
    - 设计动机：如果用高斯 splatting 函数，平面边界会模糊（如图 4/5 所示），导致相邻平面交界处质量下降。Sigmoid 函数随 $\lambda$ 增大可以精确逼近矩形边界，且 $\lambda$ 渐进增大的策略保证了优化初期的平滑梯度和后期的锐利边界。

3. **平面分裂与合并策略**:

    - 功能：自适应调整平面数量以更好拟合场景
    - 核心思路：**分裂**：每 1000 次迭代检查平面半径梯度，若 X 方向平均梯度 > 0.2 则沿 Y 轴分裂（反之亦然），将一个大平面拆成两个较小的平面。**合并**：优化完成后，法线角度误差 < 25° 且到场景中心偏移距离误差 < 0.1cm 的相邻平面合并为同一平面实例。
    - 设计动机：初始化时只有约 2000 个平面基元可能不够，分裂操作可以在需要更多细节的区域增加基元。合并操作将属于同一物理平面的多个基元聚合为一个平面实例，得到紧凑的最终表示。

### 损失函数 / 训练策略

渲染损失 $\mathcal{L}_{\text{render}}$ 包含三项：法线余弦损失（$\alpha_1=5.0$）、法线 L1 损失（$\alpha_1=5.0$）、深度 L1 损失（$\alpha_2=1.0$）。用 Adam 优化器训练 5000 次迭代。深度伪标签来自 Metric3Dv2，法线伪标签来自 Omnidata。CUDA 实现使得整个优化在 3 分钟内完成。

## 实验关键数据

### 主实验

ScanNetV2 数据集（100 场景）平面重建质量：

| 方法 | 需标注 | Chamfer↓ | F-score↑ | SC↑ | Planar Chamfer↓ |
|---|---|---|---|---|---|
| PlanarRecon | ✓ | 9.89 | 43.47 | 0.405 | 17.53 |
| AirPlanes | ✓ | 5.30 | 64.92 | 0.568 | 8.37 |
| 2DGS+RANSAC | ✗ | 14.15 | 31.33 | 0.257 | 27.40 |
| SR+RANSAC | ✗ | 5.40 | 65.45 | 0.515 | 9.78 |
| **PlanarSplatting** | **✗** | **4.83** | **68.85** | **0.532** | **9.20** |

ScanNet++ 数据集（30 场景）：

| 方法 | Chamfer↓ | F-score↑ | Planar Chamfer↓ |
|---|---|---|---|
| PlanarRecon | 17.85 | 31.10 | 26.90 |
| AirPlanes | 13.75 | 32.58 | 20.37 |
| **PlanarSplatting** | **9.33** | **47.04** | **14.75** |

### 消融实验

Splatting 函数对比：

| 配置 | Chamfer↓ | F-score↑ | 说明 |
|---|---|---|---|
| 高斯 Splatting | 较差 | 较差 | 边界模糊导致平面拟合不精确 |
| 矩形 Plane Splatting | **4.83** | **68.85** | 锐利边界精确拟合矩形表面 |

### 关键发现

- PlanarSplatting 在无标注条件下超越了需要标注的 PlanarRecon（Chamfer 4.83 vs 9.89），甚至在几何指标上接近 AirPlanes
- 在 ScanNet++ 上大幅领先所有方法（F-score 47.04 vs 之前最好 35.93）
- 矩形 splatting 函数对比高斯 splatting 有明显优势，证实了形状先验的重要性
- PlanarSplatting 可以作为 3DGS/2DGS 的初始化器——用其重建结果初始化高斯后，渲染质量显著提升且训练时间大幅缩短
- 仅用 3 分钟完成优化，比学习方法的推理速度和两步法的重建速度都快

## 亮点与洞察

1. **"形状先验即效率"**：用矩形基元替代高斯基元，从表示层面引入了室内场景的平面先验，大幅减少了所需基元数量和优化时间
2. **Sigmoid 渐进逼近矩形的设计**：$\lambda$ 指数增长的策略巧妙平衡了优化稳定性和边界锐利度，这种"从软到硬"的策略可迁移到其他需要学习硬边界的场景
3. **与 GS 方法的协同**：作为 3DGS 的几何初始化器使用，展示了显式几何重建和神经渲染的互补价值

## 局限与展望

- 仅处理平面场景，对曲面物体（如椅子、家具）无法表达
- 依赖单目深度/法线基础模型的质量，基础模型的误差会传播到平面重建
- 平面合并使用简单的阈值规则，复杂拓扑关系中可能出现过合并或欠合并
- 未来可以将矩形基元扩展为圆弧或 NURBS 基元来处理曲面

## 相关工作与启发

- **vs PlanarRecon**：PlanarRecon 端到端学习但需要标注和大量训练数据；PlanarSplatting 免训练、免标注，且结果更好
- **vs AirPlanes**：AirPlanes 先重建 mesh 再提取平面，是两步法；PlanarSplatting 直接优化平面基元，更高效
- **vs 3DGS/2DGS**：高斯基元不含形状先验；PlanarSplatting 的矩形基元天然适配室内平面，且可作为 GS 的初始化加速后续渲染优化
- 启发：将场景的结构先验（平面、对称性等）编码到基元设计中，可以用更少的参数获得更好的重建

## 评分

- 新颖性: 8/10 — 矩形平面基元+Sigmoid splatting 的设计新颖，将 GS 思路扩展到结构化重建
- 实验充分度: 8/10 — 两大数据集数百场景，对比全面，但缺少时间效率的详细分析
- 写作质量: 8/10 — 方法描述清晰，图示直观，公式推导完整
- 价值: 8/10 — 3 分钟高质量平面重建在实际应用中有很大价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Law of Data Reconstruction for Random Features (and Beyond)](../../ICLR2026/llm_pretraining/a_law_of_data_reconstruction_for_random_features_and_beyond.md)
- [\[CVPR 2025\] Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction](improving_autoregressive_visual_generation_with_cluster-oriented_token_predictio.md)
- [\[CVPR 2025\] ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)
- [\[CVPR 2025\] Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](bridging_the_vision-brain_gap_with_an_uncertainty-aware_blur_prior.md)
- [\[CVPR 2025\] Robust Message Embedding via Attention Flow-Based Steganography](robust_message_embedding_via_attention_flow-based_steganography.md)

</div>

<!-- RELATED:END -->
