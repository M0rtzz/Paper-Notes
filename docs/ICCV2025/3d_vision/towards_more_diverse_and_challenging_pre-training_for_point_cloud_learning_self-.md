---
title: >-
  [论文解读] Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views
description: >-
  [ICCV 2025][3D视觉][点云自监督学习] 提出Point-PQAE，首个将跨视图重建（Cross Reconstruction）引入3D生成式自监督学习的框架，通过点云裁剪机制生成解耦视图、设计视图相对位置编码（VRPE）和位置查询模块，使预训练更具挑战性和信息量，在ScanObjectNN上以Mlp-Linear协议平均超越Point-MAE 6.7%。
tags:
  - ICCV 2025
  - 3D视觉
  - 点云自监督学习
  - 跨视图重建
  - 解耦视图
  - 位置查询
  - 预训练
---

# Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views

**会议**: ICCV 2025  
**arXiv**: [2509.01250](https://arxiv.org/abs/2509.01250)  
**代码**: [GitHub](https://github.com/aHapBean/Point-PQAE)  
**领域**: 3D视觉  
**关键词**: 点云自监督学习, 跨视图重建, 解耦视图, 位置查询, 预训练

## 一句话总结

提出Point-PQAE，首个将跨视图重建（Cross Reconstruction）引入3D生成式自监督学习的框架，通过点云裁剪机制生成解耦视图、设计视图相对位置编码（VRPE）和位置查询模块，使预训练更具挑战性和信息量，在ScanObjectNN上以Mlp-Linear协议平均超越Point-MAE 6.7%。

## 研究背景与动机

点云自监督学习因其无需人工标注的特性而受到广泛关注。现有生成式方法（以Point-MAE为代表）主要采用**自重建**（Self-Reconstruction）范式：在单一视图中mask部分点，从可见部分重建被mask的部分。

**核心观察与动机**：

**两视图学习更优**：在2D自监督学习中，两视图范式（如SimCLR、MoCo）已被证明比单视图更有效，因为两视图引入更多方差和信息

**自重建太简单**：单视图的mask-and-reconstruct任务难度有限，模型可能走捷径而非学到深层语义

**跨重建更有挑战**：从一个视图重建另一个视图（Cross Reconstruction）比自重建困难得多，因为需要同时学习两视图间的相对位置关系和视图内的空间信息

**关键挑战**：
- 如何在点云数据中构造两个视图？（不像图像有成熟的Random Crop）
- 如何实现两个解耦视图之间的跨重建？（需要建立跨视图的空间对应关系）

## 方法详解

### 整体框架

Point-PQAE包含三个核心模块：
1. **解耦视图生成**：点云裁剪 → 独立归一化 → 随机旋转 → 两个解耦视图
2. **VRPE生成**：记录裁剪几何中心 → 计算相对位置 → 固定正弦位置编码
3. **位置查询模块**：以VRPE为Query、视图隐表示为Key/Value做cross-attention → decoder重建另一视图

### 关键设计

1. **点云裁剪机制（Point Cloud Crop）**：

   首次为3D自监督学习设计裁剪机制。与2D不同，3D空间中同样大小的立方体内点数可能差异很大。设计思路：
   
   - 随机采样裁剪比例 $r_1, r_2 \in [r_m, 1]$（$r_m = 0.6$）
   - 随机选择两个中心点 $\mathbf{C_1}, \mathbf{C_2}$
   - 对每个中心点选择最近的 $r_i \times p$ 个点构成视图 $\mathbf{X}_1, \mathbf{X}_2$
   - 记录各视图的几何中心 $\mathbf{L}_1, \mathbf{L}_2$
   - **独立归一化**：各视图以自身几何中心为原点进行min-max归一化
   - **随机旋转增强**：对各视图独立旋转，进一步解耦坐标系
   
   独立归一化中心 + 随机旋转使两视图的坐标系完全解耦，破坏了固定的空间关系。

2. **视图相对位置编码（View-Relative Positional Embedding, VRPE）**：

   解耦视图的重建关键在于告知模型两视图之间的相对位置关系。
   
   - 定义视图间相对位置：$\mathbf{RL}_{1 \to 2} = \mathbf{L}_1 - \mathbf{L}_2$
   - 组合patch级相对位置：$\mathbf{RP}_{1 \to 2} = \text{Concat}(\mathbf{G}_2, \mathbf{RL}_{1 \to 2}) \in \mathbb{R}^{n \times 6}$
   - 使用**固定正弦编码**（而非可学习编码）将6维相对位置映射为 $D$ 维：
   
   $$\mathbf{VRPE}_{1 \to 2}^i = [\sin(\frac{\mathbf{RP}^i}{e^{2/D_{12}}}), \cos(\frac{\mathbf{RP}^i}{e^{2/D_{12}}}), \ldots]$$
   
   固定编码避免了可学习编码损害相对位置的精确表达（消融实验证实）。

3. **位置查询模块（Positional Query Block）**：

   核心思想：用VRPE作为Query查询另一视图的隐表示，实现跨视图信息提取。
   
   假设视图1的隐表示 $\mathbf{H}_1$ 已包含其内在特征和全局信息，结合 $\mathbf{VRPE}_{1 \to 2}$，就能重建视图2：
   
   $$\mathbf{T}_2 = \text{Softmax}(\frac{\mathbf{Q}_{\mathbf{VRPE}_{1 \to 2}} \mathbf{K}_{\mathbf{H}_1}}{\sqrt{D}}) \mathbf{V}_{\mathbf{H}_1}$$
   
   重建是对称的（双向跨重建），最终损失使用 $\ell_2$ Chamfer Distance。

### 损失函数 / 训练策略

$$\mathcal{L}_{cross} = \mathcal{L}_{2 \to 1} + \mathcal{L}_{1 \to 2}$$

每项为标准Chamfer Distance：

$$\mathcal{L}_{2 \to 1} = \frac{1}{|\mathbf{P}_{pred}^1|} \sum_{a \in \mathbf{P}_{pred}^1} \min_{b \in \mathbf{P}_1} \|a - b\|_2^2 + \frac{1}{|\mathbf{P}_1|} \sum_{b \in \mathbf{P}_1} \min_{a \in \mathbf{P}_{pred}^1} \|a - b\|_2^2$$

- 预训练在ShapeNet（~51K模型）上进行300 epochs
- 编码器12层Transformer blocks + 解码器4层，隐藏维度384，6头注意力
- AdamW优化器，lr=0.0005，weight decay=0.05，cosine衰减
- 每个点云采样1024点，裁剪为64 patches × 32 points

## 实验关键数据

### 主实验（ScanObjectNN分类，真实世界3D物体）

| 方法 | 参数量 | OBJ-BG | OBJ-ONLY | PB-T50-RS | 协议 |
|------|--------|--------|----------|-----------|------|
| Point-MAE† | 22.1M | 92.6 | 91.9 | 88.4 | Full |
| **Point-PQAE** | 22.1M | **95.0** | **93.6** | **89.6** | Full |
| Point-MAE† | 22.1M | 82.8 | 83.2 | 74.1 | Mlp-Linear |
| **Point-PQAE** | 22.1M | **89.3** | **90.2** | **80.8** | Mlp-Linear |
| Point-MAE† | 22.1M | 85.8 | 85.5 | 80.4 | Mlp-3 |
| **Point-PQAE** | 22.1M | **90.7** | **90.9** | **83.3** | Mlp-3 |

在最严格的冻结评估协议Mlp-Linear下，三个变体平均超越Point-MAE **6.7%**（82.8→89.3, 83.2→90.2, 74.1→80.8），证明跨重建学到了更高质量的表示。

### 消融实验（ScanObjectNN-PB-T50-RS，Full协议）

| 配置 | OBJ-BG | OBJ-ONLY | PB-T50-RS | 说明 |
|------|--------|----------|-----------|------|
| Self-recon（Point-MAE） | 92.6 | 91.9 | 88.4 | 自重建基线 |
| Cross-recon（无VRPE） | - | - | ~85 | 无位置信息跨重建失败 |
| Cross-recon + 可学习PE | - | - | ~88 | 可学习编码不够精确 |
| Cross-recon + VRPE（本文） | **95.0** | **93.6** | **89.6** | 固定正弦编码最优 |

### Few-shot学习（ModelNet40）

| 方法 | 5-way 10-shot | 5-way 20-shot | 10-way 10-shot | 10-way 20-shot |
|------|:---:|:---:|:---:|:---:|
| Point-MAE† | 96.4±2.8 | 97.8±2.0 | 92.5±4.4 | 95.2±3.9 |
| **Point-PQAE** | **96.9±3.2** | **98.9±1.0** | **94.1±4.2** | **96.3±2.7** |
| ReCon（跨模态） | 97.3±1.9 | 98.9±1.2 | 93.3±3.9 | 95.8±3.0 |

Point-PQAE在不使用跨模态教师的情况下，与使用强预训练教师的ReCon性能相当甚至更优。

### 关键发现

1. **跨重建显著优于自重建**：相同参数量下，Mlp-Linear协议平均提升6.7%，说明更有挑战的预训练任务确实学到了更好的表示
2. **VRPE是跨重建成功的关键**：没有位置信息，跨重建性能急剧下降；固定正弦编码优于可学习编码
3. **泛化到不同裁剪比例**：虽然预训练使用 $r_m = 0.6$，但模型对其他裁剪比例泛化良好
4. **单模态方法首次逼近跨模态方法**：在Mlp-Linear/Mlp-3协议下接近使用2D教师的ACT和ReCon

## 亮点与洞察

- **范式创新**：从自重建到跨重建的转变是3D生成式自监督学习的重要范式突破
- **首创点云裁剪机制**：基于最近邻的裁剪策略简洁高效，避免了3D空间中立方体裁剪的不均匀问题
- **VRPE设计精巧**：将视图间和patch级别的位置关系编码为固定正弦信号，既精确又高效
- **即插即用的PQ模块**：位置查询模块可方便地应用到蒸馏等其他任务

## 局限与展望

- 预训练仅使用合成数据ShapeNet，对真实场景点云（如室外LiDAR）的效果待验证
- 跨重建的额外计算开销（需要编码两个视图 + cross-attention）
- 裁剪比例和视图数量等超参数对不同数据集可能需要调整
- 未与最新的3D自监督方法（如基于MAE的更多变体）进行全面对比

## 相关工作与启发

- Point-MAE的自重建范式是本文的直接基线，本文从"增加任务难度"角度进行改进
- 2D领域SimCLR/MoCo的两视图学习理念被成功迁移到3D生成式学习
- VRPE的设计可启发其他需要建模两组3D点之间空间关系的任务

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 跨重建范式在3D自监督学习中首次提出，点云裁剪机制和VRPE都是原创设计
- **实验充分度**: ⭐⭐⭐⭐ 分类/few-shot/分割多任务验证，三种评估协议，消融研究到位
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，与自重建的对比直观，公式推导完整
- **价值**: ⭐⭐⭐⭐ 为3D自监督学习提供了新范式，冻结评估提升显著，实际指导意义强

<!-- RELATED:START -->

## 相关论文

- [StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)
- [4D Visual Pre-training for Robot Learning](4d_visual_pre-training_for_robot_learning.md)
- [DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning](dapmae_domainadaptive_point_cloud_masked_autoencoder_for_eff.md)
- [BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](buffer-x_towards_zero-shot_point_cloud_registration_in_diverse_scenes.md)
- [E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training](../../CVPR2026/3d_vision/e-rayzer_self-supervised_3d_reconstruction_as_spatial_visual_pre-training.md)

<!-- RELATED:END -->
