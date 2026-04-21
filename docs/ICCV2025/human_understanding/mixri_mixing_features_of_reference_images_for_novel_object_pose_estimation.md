---
title: >-
  [论文解读] MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation
description: >-
  [ICCV 2025][人体理解][新物体位姿估计] 提出 MixRI，一个仅需 12 张参考图像和 5.3M 参数的轻量级网络，通过多视角特征融合策略直接建立多参考图与查询图之间的 2D-3D 对应关系，在 BOP 挑战的 7 个核心数据集上实现了与需要数百张参考图的方法相当的位姿估计性能。
tags:
  - ICCV 2025
  - 人体理解
  - 新物体位姿估计
  - 特征匹配
  - 多视角融合
  - 轻量级网络
  - 边缘AI
---

# MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation

**会议**: ICCV 2025  
**arXiv**: [2601.06883](https://arxiv.org/abs/2601.06883)  
**代码**: [项目主页](https://npucvr.github.io/MixRI/)  
**领域**: 人体理解  
**关键词**: 新物体位姿估计, 特征匹配, 多视角融合, 轻量级网络, 边缘AI

## 一句话总结

提出 MixRI，一个仅需 12 张参考图像和 5.3M 参数的轻量级网络，通过多视角特征融合策略直接建立多参考图与查询图之间的 2D-3D 对应关系，在 BOP 挑战的 7 个核心数据集上实现了与需要数百张参考图的方法相当的位姿估计性能。

## 研究背景与动机

六自由度（6DoF）物体位姿估计是机器人操作、增强现实等具身 AI 应用的基础。传统方法需要在训练阶段见过目标物体并生成大量合成数据，部署繁琐。**基于 CAD 模型的新物体位姿估计**应运而生：训练时不知道测试时要推理的具体物体，测试时利用 CAD 渲染的参考图像来估计位姿。

现有方法普遍面临三个问题，限制了其在实际嵌入式 AI 中的应用：

**参考图像数量过多**：FoundPose 需要 798 张、MegaPose 需要 520 张、GigaPose 需要 162 张。每个物体都需要渲染和存储大量参考图像，严重限制了可同时处理的物体数量。更多参考图还意味着更多的特征预提取和缓存需求。

**网络参数庞大**：GigaPose 有 316.3M 参数，FoundPose 有 302.9M 参数，对内存有限的边缘设备不友好。

**推理速度慢**：现有方法普遍采用**两阶段流水线**——先从大量参考图中检索最近视角（模板匹配），再与选中的参考图做精细匹配。检索阶段本身就增加了延迟。

更关键的问题是：当参考图像数量大幅减少时，**最近参考图与查询图之间可能存在大视角差异**（宽基线问题），使得传统的两阶段方法急剧性能下降。作者观察到两个关键点：(1) 多视角几何能提供比单张图更丰富的信息；(2) 参考图稀疏时遮挡更常见，更需集成多视角信息。因此，**应该跳过视角检索阶段，直接利用所有参考图的信息**。

## 方法详解

### 整体框架

MixRI 是一个**纯特征匹配**的单阶段方法。给定查询图像和少量参考图像（12张），网络的目标是：对于 3D 物体表面上采样的每个点 $\mathbf{p}_k$，确定其在查询图上的投影坐标 $\mathbf{u}_{0,k}$ 和遮挡标志 $O_{0,k}$。建立足够的 2D-3D 对应后，使用 RANSAC + PnP 求解 6DoF 位姿。流程：共享编码器提取特征 → 双注意力特征混合器融合多视角信息 → 构建 cost volume → 两个预测头分别输出坐标和遮挡标志。

### 关键设计

1. **跨参考图对应关系建立**：在训练前，利用已知的参考图位姿和深度，对每个参考图上采样的 2D 点反投影到 3D 物体坐标，再投影到其他参考图上，得到同一 3D 点在所有参考图上的 2D 投影坐标 $\{\mathbf{u}_{i,k}\}$ 及其遮挡标志 $O_{i,k}$。遮挡判定类似 Z-buffer：若投影深度与实际深度差异超过阈值 $\tau$，则标为遮挡。

$$\mathbf{p}_k = d_{i,k} \mathbf{T}_i^{-1} \mathbf{K}_i^{-1} \tilde{\mathbf{u}}_{i,k}$$

这为特征融合提供了精确的空间对齐基础。

2. **双注意力特征混合器（Dual-Attention Based Feature Mixer）**：这是 MixRI 的核心创新。它包含三个模块交替迭代：

    - **SAP（点间自注意力）**：在同一参考图的 $N$ 个采样点之间做注意力，整合空间位置信息
    - **SAF（帧间自注意力）**：在 $S$ 个参考帧之间做注意力，并与可学习的融合 token $\bar{\mathbf{F}}$ 交互，实现跨视角信息聚合
    - **MARQ（参考-查询混合注意力）**：在融合后的参考 token 和查询图特征之间做交叉注意力，增强查询图特征

   整体用公式表示为：
   $$\hat{\mathbf{F}} = \text{Self}_N(g_N(\hat{\mathbf{F}}), \mathbf{O}), \quad \hat{\mathbf{F}}, \bar{\mathbf{F}} = \text{Self}_S(g_S(\hat{\mathbf{F}}), \bar{\mathbf{F}}, \mathbf{O}), \quad \bar{\mathbf{F}}, \mathbf{F}_0 = \text{Mix}(\bar{\mathbf{F}}, \mathbf{F}_0, \mathbf{O})$$

   遮挡掩码 $\mathbf{O}$ 用于 masked attention，避免融合被遮挡点的错误特征。迭代 $n_0$ 次后得到最终融合结果。

3. **遮挡感知的 Cost Volume 与双头预测**：融合后的参考 token $\bar{\mathbf{F}}$ 与查询图特征 $\tilde{\mathbf{F}}_0$ 构建 cost volume $\mathbf{C} \in \mathbb{R}^{H' \times W' \times N}$，经 Conv3D 骨干处理后，两个独立的预测头分别输出：(a) 2D 热力图（通过 spatial soft argmax 转为坐标）；(b) 遮挡概率。仅使用预测为可见的点参与 PnP 求解。

   编码器使用 ResNet-like 骨干（共享权重），并集成了旋转不变特征模块，使提取的特征不受物体姿态影响。

### 损失函数 / 训练策略

- **遮挡损失**：$L_{occ} = \text{BCE}(O_{gt,k}, O_{0,k})$
- **定位损失**：$L_{loc} = \text{Huber}(\mathbf{U}_{gt,k}, \mathbf{U}_{0,k}) \cdot \mathbf{1}\{O_{gt,k}=0\}$（仅对可见点计算）
- **总损失**：$L = L_{occ} + 100 \cdot L_{loc}$
- 仅使用 GSO-Dataset 的合成图像训练，在真实场景 BOP 数据集上测试
- 使用 CNOS 作为通用目标检测器（可替换为其他轻量级方法）

## 实验关键数据

### 主实验

**BOP 挑战 7 核心数据集（Average Recall ↑）**

| 方法 | 参数量 | 参考图数 | LM-O | IC-BIN | HB | YCB-V | MEAN | 推理时间 |
|------|--------|----------|------|--------|-----|-------|------|----------|
| MegaPose | 21.6M | 520 | 22.9 | 15.2 | 25.1 | 28.1 | 20.8 | 15.5s |
| GigaPose | 316.3M | 162 | 29.9 | 23.1 | 34.8 | 29.0 | 27.6 | 0.8s |
| FoundPose | 302.9M | 798 | 39.6 | 23.9 | 50.8 | 45.2 | 37.2 | 1.6s |
| **MixRI (12)** | **5.3M** | **12** | 27.0 | **29.7** | 44.9 | **52.8** | 31.4 | **0.5s** |
| **MixRI (24)** | **5.3M** | **24** | 30.4 | **30.8** | **50.2** | **54.6** | 34.1 | 0.7s |

MixRI 使用 33× 少的参考图、57× 少的参数和 3× 快的速度达到接近 SOTA 的性能。

### 消融实验

| 配置 | YCB-V | LM-O | TUD-L | MEAN | 说明 |
|------|-------|------|-------|------|------|
| 无 SAP + 无 MARQ | 2.9 | 2.8 | 2.2 | 2.6 | 基础匹配 |
| 有 SAP | 19.2 | 7.8 | 8.7 | 11.9 | +9.3% AR |
| 有 MARQ | 33.2 | 13.4 | 21.9 | 22.8 | +20.2% AR |
| **有 SAP + MARQ** | **54.6** | **30.4** | **33.6** | **39.5** | **+36.9% AR** |

**有限参考图数量对比（MixRI vs GigaPose）**

| 参考图数 | GigaPose MEAN | MixRI MEAN |
|----------|---------------|------------|
| 4 | 9.1 | 17.9 |
| 6 | 11.0 | 31.4 |
| 12 | 14.7 | 36.4 |
| 24 | 18.3 | 39.5 |

### 关键发现

- GigaPose 在参考图减少时性能急剧下降（两阶段方法的固有弱点），而 MixRI 天然适应少参考图场景
- MARQ（参考-查询混合注意力）贡献最大（+20.2%），表明查询图与参考图的交叉注意力对匹配至关重要
- SAP 和 MARQ 存在强协同效应：单独使用各贡献 9.3% 和 20.2%，组合后达到 36.9%
- 仅 60 个对应点即可在 YCB-V 上达到 45% AR，展现方法的鲁棒性
- ITODD 数据集表现不佳（灰度图 + 强反射 + 弱纹理），暴露了 RGB 训练在灰度场景的局限性

## 亮点与洞察

- **"少即是多"哲学**的成功验证——不是更多参考图更好，而是更智能的信息融合更重要
- 从两阶段（检索+匹配）到单阶段（直接全图融合匹配）的范式转换是关键创新
- 5.3M 参数量极其轻量（比 GigaPose 小 60×），对边缘设备部署非常友好
- 遮挡检测作为网络的自然输出（而非后处理），简化了流水线
- 将点跟踪和特征匹配的思路统一——同一 3D 点在多张参考图中的投影类似于跟踪轨迹

## 局限与展望

- 在灰度图像（ITODD）上表现不佳，需要考虑颜色模式的泛化性
- 对称物体的处理没有专门设计，可能在高度对称的工业零件上受限
- 12 张参考图的均匀采样策略可能不是最优的——自适应视角选择策略值得探索
- 未与 refinement 方法结合（如 MegaPose 的 coarse-to-fine），进一步提升精度
- 在极端遮挡场景（IC-BIN）上虽然表现好，但 ITODD 的弱纹理仍是挑战

## 相关工作与启发

- **GigaPose** 和 **FoundPose** 是最直接的对比——代表了检索+匹配范式的极致
- **LoFTR** 等检测器无关匹配方法启发了端到端匹配设计
- **TAP-Net** 等点跟踪方法的遮挡建模启发了多视角对应关系设计
- 启发：轻量级多视角融合策略可推广到抓取姿态估计、AR 中的物体放置等下游任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 单阶段多视角融合匹配的范式转换是重要创新
- **实验充分度**: ⭐⭐⭐⭐⭐ BOP 挑战 7 个数据集、详细的参考图数量对比、完整消融
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，方法描述规范，气泡图对比直观
- **价值**: ⭐⭐⭐⭐⭐ 对边缘 AI 部署有极高实用价值，参数量和推理速度优势显著

<!-- RELATED:START -->

## 相关论文

- [COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](../../CVPR2026/human_understanding/cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)
- [One2Any: One-Reference 6D Pose Estimation for Any Object](../../CVPR2025/human_understanding/one2any_one-reference_6d_pose_estimation_for_any_object.md)
- [Co-op: Correspondence-Based Novel Object Pose Estimation](../../CVPR2025/human_understanding/co-op_correspondence-based_novel_object_pose_estimation.md)
- [FoundPose: Unseen Object Pose Estimation with Foundation Features](../../ECCV2024/human_understanding/foundpose_unseen_object_pose_estimation_with_foundation_features.md)
- [CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation](cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)

<!-- RELATED:END -->
