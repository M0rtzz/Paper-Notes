---
title: >-
  [论文解读] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning
description: >-
  [ICCV 2025][3D视觉][点云表示学习] 提出 StruMamba3D，通过为 SSM 的隐含状态赋予空间位置属性（空间状态）来维护 3D 点的邻接关系，并引入序列长度自适应策略解决预训练与下游任务之间的序列长度差异问题，在 ScanObjectNN 最难分割上达到 92.75% 准确率，ModelNet40 达到 95.1%，均为单模态 SOTA。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "点云表示学习"
  - "状态空间模型"
  - "Mamba"
  - "自监督学习"
  - "结构建模"
---

# StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning

**会议**: ICCV 2025  
**arXiv**: [2506.21541](https://arxiv.org/abs/2506.21541)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 点云表示学习, 状态空间模型, Mamba, 自监督学习, 结构建模

## 一句话总结

提出 StruMamba3D，通过为 SSM 的隐含状态赋予空间位置属性（空间状态）来维护 3D 点的邻接关系，并引入序列长度自适应策略解决预训练与下游任务之间的序列长度差异问题，在 ScanObjectNN 最难分割上达到 92.75% 准确率，ModelNet40 达到 95.1%，均为单模态 SOTA。

## 研究背景与动机

### 问题定义

点云表示学习旨在从无序、稀疏的点云中提取几何和语义特征，支持分类、分割、检测等下游任务。自监督预训练方法通过在大量无标注数据上学习通用表示，显著提升了下游性能。

### 已有方法的不足

现有 Mamba 点云方法（PointMamba、Mamba3D、PCM）存在两个根本问题：

**3D 点邻接关系被破坏**：将 3D 点序列化为 1D 序列时，空间上相邻的点在序列中不一定相邻。与文本和图像不同，点云**天然缺乏上下文信息**，依赖局部结构作为基本特征单元。序列化造成的空间扭曲阻碍了 Mamba 建模细粒度结构信息。

**下游任务中长序列记忆不足**：Mamba-based 方法使用掩码点建模（MPM）进行预训练（输入短序列），但下游任务输入完整序列（更长）。Mamba 的选择机制在预训练时学会了较高频率的状态更新，当序列长度增加时，频繁更新导致模型难以保持长程记忆，影响长距离语义建模。

### 核心动机

**核心问题**：能否通过 SSM 的隐含状态来建模 3D 点之间的空间关系？如果给隐含状态赋予位置属性，让它们作为局部结构的代理，就可以在 SSM 处理过程中维护点之间的空间依赖。同时，如果模型能根据序列长度调整状态更新频率，就能在更长的输入中保持长程记忆。

## 方法详解

### 整体框架

给定原始点云 $P_{raw} \in \mathbb{R}^{N_0 \times 3}$，通过 FPS 和 KNN 获取输入点 $P_x$，用 PointNet 提取 token 嵌入 $F_x \in \mathbb{R}^{N \times D}$。同时初始化空间状态 $P_h \in \mathbb{R}^{M \times 3}$（$M=16$）。输入 token 和空间状态通过多个结构化 SSM 块处理，空间状态在块之间级联传递。预训练使用 MPM 任务 + 空间状态一致性损失。

### 关键设计

#### 1. **空间状态初始化与状态级更新**

- **功能**：给 SSM 的隐含状态赋予 3D 位置属性，使其能代表点云的局部结构区域
- **核心思路**：

  **空间状态初始化**：通过 FPS + KNN 将原始点云分成 $M$ 组 $\{\mathcal{G}_m\}_{m=1}^M$，计算质心作为状态位置：
  $$P_h^m = \frac{1}{|\mathcal{G}_m|} \sum_{P_i \in \mathcal{G}_m} P_i$$
  然后用线性映射将位置编码为状态特征 $F_h = \phi_h(P_h) \in \mathbb{R}^{M \times D}$。

  **状态级更新**：修改 SSM 的状态更新方程，显式建模输入点与空间状态之间的空间关系。首先计算相对偏移 $\triangle P_i^m = P_x^i - P_h^m$，然后将偏移信息融入 SSM 参数：
  $$(\mathbf{B}_i^m, \mathbf{C}_i^m) = \phi(x_i) + \text{MLP}(\triangle P_i^m)$$
  这使得空间状态可以选择性地利用同一区域内的点更新特征，而输入点可以从状态获取局部结构信息。

- **设计动机**：原始 Mamba 的隐含状态不包含几何信息，无法建模点云的局部结构。通过赋予位置属性，让每个状态负责特定空间区域，同时将空间关系融入 SSM 参数生成过程，实现"结构感知"的状态选择和传播。

#### 2. **结构化 SSM 块**

- **功能**：完整的结构化 SSM 处理单元，包含双向扫描和轻量卷积
- **核心思路**：

  **双向结构 SSM**：采用前向和后向两个方向的结构化 SSM，共享空间状态 $\hat{F}_h$ 作为初始状态，最终通过线性层融合：
  $$F'_x, F'_h = \phi_o(\text{SSM}_f(\hat{F}_x, \hat{F}_h) + \text{SSM}_b(\hat{F}_x, \hat{F}_h))$$

  **轻量卷积模块**：解决空间状态之间缺乏直接交互的问题。对每个空间状态 $P_h^m$，找到其 $k$ 近邻 $\mathcal{N}(m)$，基于相对位置生成注意力权重并加权聚合邻居特征：
  $$w_h^{mj} = \text{softmax}(\phi_w(\triangle P_h^{mj}, P_h^m))$$
  $$\hat{F}_h^m = \phi_c(\sum_{j \in \mathcal{N}(m)} w_h^{mj} F_h^{mj})$$
  同样对输入点也应用轻量卷积，替代 Mamba 中的因果 1D 卷积。

- **设计动机**：单向扫描无法在单次前向中实现双向信息交换，双向机制弥补这一缺陷。空间状态在标准 SSM 中是孤立的，轻量卷积扩展其感受野，捕获全局语义。

#### 3. **序列长度自适应策略**

- **功能**：解决预训练（短序列）和下游任务（长序列）之间的序列长度差异问题
- **核心思路**：

  **自适应状态更新机制**：引入可学习参数 $\tau$ 调节总采样时间，使不同序列长度下保持一致的总采样时间 $\Delta_{all} = \tau$：
  $$\Delta_i = \frac{\tau \times \Delta_i}{\sum_{i=1}^N \Delta_i}$$

  **空间状态一致性损失**：使用教师-学生框架。教师模型用完整 token 更新空间状态（输出 $F_h'^f$），学生模型用可见 token 更新（输出 $F_h'^v$），强制一致性：
  $$\mathcal{L}_{ssc} = \text{Smooth L1}(F_h'^v, F_h'^f)$$
  最终预训练总损失：$\mathcal{L}_{total} = \mathcal{L}_{cd} + \lambda \times \mathcal{L}_{ssc}$

- **设计动机**：由于 $\Delta$ 控制状态更新频率，长序列需要更低的更新频率来保持长程记忆。同时，状态一致性损失确保模型即使只看到部分点也能推断完整结构。

### 损失函数 / 训练策略

- **预训练损失**：$\mathcal{L}_{total} = \mathcal{L}_{cd} + 2 \times \mathcal{L}_{ssc}$
- 在 ShapeNet 上预训练，52472 个 3D 模型，55 个类别
- 每个点云 1024 点，分 64 个 patch（每 patch 32 点），掩码率 0.6
- 结构化 SSM 块 12 层，特征维度 384，空间状态数 $M=16$
- 教师模型通过 EMA 更新

## 实验关键数据

### 主实验

**ScanObjectNN 分类**（真实世界数据集，最难分割 PB-T50-RS）：

| 方法 | 骨干 | 参数量(M) | OBJ-BG | OBJ-ONLY | PB-T50-RS |
|------|------|----------|--------|----------|-----------|
| PointMAE† | Transformer | 22.1 | 92.77 | 91.22 | 89.04 |
| PointGPT-S† | Transformer | 29.2 | 93.39 | 92.43 | 89.17 |
| PointMamba† | Mamba | 12.3 | 94.32 | 92.60 | 89.31 |
| Mamba3D† | Mamba | 16.9 | 93.12 | 92.08 | 92.05 |
| **StruMamba3D†** | **Structural SSM** | **15.8** | **95.18** | **93.63** | **92.75** |

**ModelNet40 分类**：

| 方法 | w/o Voting | w/ Voting |
|------|-----------|----------|
| Mamba3D† | 94.7 | 95.1 |
| **StruMamba3D†** | **95.1** | **95.4** |

**ShapeNetPart 部件分割**（单尺度模型）：

| 方法 | mIoU_c | mIoU_i |
|------|--------|--------|
| PointMamba | 84.4 | 86.2 |
| Mamba3D | 83.6 | 85.6 |
| **StruMamba3D** | **85.0** | **86.7** |

### 消融实验

**各模块效果**：

| 结构 SSM 块 | 长度自适应策略 | ScanObjectNN | ModelNet40 | ShapeNetPart(mIoU_c) |
|-----------|------------|-------------|-----------|---------------------|
| ✗ | ✗ | 87.23 | 91.86 | 81.56 |
| ✓ | ✗ | 92.09 | 94.45 | 84.49 |
| ✓ | ✓ | **92.75** | **95.06** | **84.96** |

**结构 SSM 内部组件**：

| 方法 | ScanNN | MN40 | SNPart |
|------|--------|------|--------|
| baseline（标准 Mamba） | 88.24 | 92.50 | 82.08 |
| + Structural SSM | 91.78 | 93.92 | 84.15 |
| + 空间状态轻量卷积 | 92.22 | 94.65 | 84.62 |
| + 输入点轻量卷积 | 92.40 | 94.81 | 84.77 |
| + 双向扫描 | **92.75** | **95.06** | **84.96** |

**SSM 参数来源消融**：

| $\phi(x)$ | 空间状态 | $\text{MLP}(\triangle P)$ | ScanNN | MN40 |
|-----------|--------|------------------------|--------|------|
| ✓ | ✗ | ✗ | 90.94 | 93.84 |
| ✓ | ✓ | ✗ | 91.33 | 94.12 |
| ✓ | ✓ | ✓ | **92.75** | **95.06** |
| ✗ | ✓ | ✓ | 91.12 | 94.25 |

### 关键发现

1. **结构 SSM 块贡献最大**：相比标准 Mamba baseline，在 ScanObjectNN 上提升 4.86%，证明通过隐含状态建模结构信息的有效性
2. **空间关系 + 输入特征缺一不可**：仅用空间关系（91.12%）或仅用输入特征（90.94%）都不如两者结合（92.75%）
3. **长度自适应策略的两个组件互补**：单独使用自适应更新机制或一致性损失仅有微小提升，组合使用才有显著效果
4. **超越跨模态方法**：StruMamba3D 仅使用单模态信息即超越大部分使用跨模态（图像+文本）信息的方法
5. **Mamba3D 在部件分割上失败**：没有序列化策略的 Mamba3D 甚至不如 PointMAE，验证了结构信息对精细任务的关键性

## 亮点与洞察

1. **核心洞察独到**：将 SSM 的隐含状态从"黑盒记忆"重新定义为"空间代理"，通过赋予位置属性来建模 3D 结构——这是首次从状态空间角度解决点云结构建模问题
2. **解决了 Mamba 点云方法的根本矛盾**：序列化 ↔ 3D 邻接关系不可兼得，空间状态提供了"第三条路"
3. **长度自适应策略的洞察深刻**：识别出预训练-下游序列长度差异对 Mamba 选择机制的影响，这个问题在之前的工作中被忽视
4. **轻量卷积的精妙设计**：受图卷积启发，用于空间状态之间的交互，既保持了 SSM 的线性复杂度又增强了全局感知

## 局限与展望

1. **空间状态数量固定**：$M=16$ 是手动设置的，可能不适合所有场景；自适应确定状态数是改进方向
2. **仅验证单尺度架构**：多尺度设计（如 PointM2AE）可能进一步提升部件分割等精细任务性能
3. **未探索更大规模模型**：当前参数量为 15.8M，扩展到更大规模的效果未知
4. **空间状态位置固定**：状态位置由预处理确定，不随层级更新——可学习的位置更新可能更灵活

## 相关工作与启发

- 与 PointMamba 的关系：PointMamba 用空间填充曲线序列化但仍破坏邻接关系，StruMamba3D 通过空间状态绕过此问题
- 与 Mamba3D 的对比：Mamba3D 用双向扫描 + 局部 norm pooling 但在分割任务上效果差（83.6 mIoU_c），StruMamba3D (85.0) 显著领先
- DGCNN 的图卷积思路被借鉴到轻量卷积模块设计

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次从 SSM 状态空间角度解决点云结构建模问题，空间状态概念原创性强
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4 个下游任务 + 详尽的消融，每个设计选择都有消融验证
- **写作质量**: ⭐⭐⭐⭐ — 问题分析清晰，但公式较多，部分地方阅读难度较高
- **价值**: ⭐⭐⭐⭐ — 为 Mamba 在 3D 领域的应用提供了重要的结构建模范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views](towards_more_diverse_and_challenging_pre-training_for_point_cloud_learning_self-.md)
- [\[CVPR 2025\] Spectral Informed Mamba for Robust Point Cloud Processing](../../CVPR2025/3d_vision/spectral_informed_mamba_for_robust_point_cloud_processing.md)
- [\[CVPR 2025\] PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter](../../CVPR2025/3d_vision/pma_towards_parameter-efficient_point_cloud_understanding_via_point_mamba_adapte.md)
- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)

</div>

<!-- RELATED:END -->
