---
title: >-
  [论文解读] Particulate: Feed-Forward 3D Object Articulation
description: >-
  [CVPR 2026][3D视觉][铰接物体] Particulate 提出了一个前馈式模型，给定静态 3D 网格即可在数秒内推断出完整的铰接结构（部件分割、运动学树、运动约束），基于 Part Articulation Transformer 在公开数据集上端到端训练，显著优于需要逐物体优化的现有方法，并能与 3D 生成模型结合实现从单张图像到铰接 3D 物体的生成。
tags:
  - CVPR 2026
  - 3D视觉
  - 铰接物体
  - 3D部件分割
  - 运动约束预测
  - 前馈推理
  - Transformer
---

# Particulate: Feed-Forward 3D Object Articulation

**会议**: CVPR 2026  
**arXiv**: [2512.11798](https://arxiv.org/abs/2512.11798)  
**代码**: https://ruiningli.com/particulate  
**领域**: 3D视觉  
**关键词**: 铰接物体, 3D部件分割, 运动约束预测, 前馈推理, Transformer

## 一句话总结
Particulate 提出了一个前馈式模型，给定静态 3D 网格即可在数秒内推断出完整的铰接结构（部件分割、运动学树、运动约束），基于 Part Articulation Transformer 在公开数据集上端到端训练，显著优于需要逐物体优化的现有方法，并能与 3D 生成模型结合实现从单张图像到铰接 3D 物体的生成。

## 研究背景与动机

1. **领域现状**：大多数现实物体不仅有形状，还有运动能力（如柜门旋转、抽屉滑动）。理解物体的铰接结构对于机器人操作、游戏仿真、数字孪生至关重要。现有方法要么依赖规则式程序生成难以覆盖长尾物体，要么需要逐物体多视角优化耗时极长（10-20分钟以上）。

2. **现有痛点**：学习式方法分三类——(a) 3D 部件分割方法只预测语义分割不建模铰接关系；(b) 3D 铰接物体生成方法仅覆盖少数类别且假设运动学结构已知；(c) 基于 VLM 的方法（如 Articulate AnyMesh）虽然泛化性好但需要逐物体优化十几分钟且无法处理内部/遮挡部件。

3. **核心矛盾**：如何在保持泛化性的同时实现快速前馈推理，且能处理内部不可见部件？

4. **本文目标** 从静态 3D 网格直接前馈式预测完整铰接结构（部件分割 + 运动学树 + 运动参数），支持多关节、多类别、AI 生成的 3D 资产。

5. **切入角度**：利用 Transformer 的灵活性和可扩展性，在大规模多类别铰接数据集上端到端训练，用 learnable part query 和多头解码器分别预测各铰接属性。

6. **核心 idea**：用标准 Transformer + learnable part queries 在点云上做端到端训练，一次前馈推理即可预测所有铰接属性，包括部件分割、运动学树和运动约束。

## 方法详解

### 整体框架
输入为 3D 网格（转换为点云 $\mathcal{P}$），输出为完整铰接结构 $\mathcal{A} = (P, S, K, M)$：部件数 $P$、面到部件的分割映射 $S$、运动学树 $K$、运动约束 $M$（运动类型、方向、旋转轴、运动范围）。模型由 Transformer backbone + 多个专用解码头组成，端到端训练后前馈推理仅需约 10 秒。

### 关键设计

1. **Part Articulation Transformer**:

    - 功能：从点云中提取 point token 和 part token 的潜在表示
    - 核心思路：每个点 $\mathbf{p}_i$ 通过坐标、法向量和 PartField 语义特征三个 MLP 分别编码后相加得到 point token $\tilde{\mathbf{p}}_i$。初始化 $P_{max}$ 个可学习的 part query $\mathcal{Q}$（远大于实际部件数）。Backbone 由 $B=8$ 个 attention block 组成，每个 block 交替执行 query 自注意力和 query-to-point 交叉注意力。为节省内存，不在 point token 之间做自注意力（因为点数量 $N \gg P_{max}$）。
    - 设计动机：用 DETR 风格的 part query 解决部件数量未知的问题，Transformer 的注意力机制能灵活捕捉部件间和点与部件间的关系。PartField 特征引入了 2D 语义部件先验，增强对新类别的泛化。

2. **多头解码器（Decoder Heads）**:

    - 功能：从 point/part token 分别解码各铰接属性
    - 核心思路：**部件分割**用 MLP $h_S(\tilde{\mathbf{p}}_i, \tilde{\mathbf{q}}_j)$ 预测 $N \times P_{max}$ 的 logit 矩阵。**运动学树**用 MLP $h_K(\tilde{\mathbf{q}}_i, \tilde{\mathbf{q}}_j)$ 预测 $P_{max} \times P_{max}$ 的父子关系概率矩阵，推理时用 Edmonds 算法提取最大生成树。**运动类型/范围/棱柱方向**分别用独立 MLP 从 part token 预测。
    - 设计动机：将铰接结构分解为多个可独立预测的属性，每个用专门的 MLP 解码，简化学习问题。

3. **旋转轴的过参数化预测（Over-parameterized Revolute Axes）**:

    - 功能：准确预测旋转轴的方向和位置
    - 核心思路：旋转轴方向 $\tilde{\mathbf{d}}_{ra}^i$ 直接由 MLP 从 part token 预测并归一化。但轴的位置不用 MLP 直接预测（容易过拟合），而是让每个属于该部件的 3D 点投票：每个点通过 MLP $h_{cp}(\tilde{\mathbf{p}}_j, \tilde{\mathbf{q}}_i)$ 预测其到旋转轴的正交投影点，推理时取所有投票的中位数作为轴位置。
    - 设计动机：旋转轴方向通常是轴对齐的相对容易学习，但位置预测需要高精度。用点级投票的过参数化方式，利用空间先验获得更鲁棒的轴位置估计，同时中位数聚合对离群值鲁棒。

### 损失函数 / 训练策略

多任务损失 $\mathcal{L} = \mathcal{L}_S + \mathcal{L}_K + \mathcal{L}_M$。部件分割用交叉熵损失，运动学树用二值交叉熵。运动约束损失包括运动类型的交叉熵、棱柱/旋转范围和方向的 L1 损失、旋转轴方向和位置的 L1 损失。训练时用 Hungarian 算法将 $P_{max}$ 个预测 part query 与 $P$ 个 GT 部件匹配（类似 DETR）。训练数据来自 PartNet-Mobility（3800 个物体, 50 类）和 GRScenes，每次迭代随机采样铰接状态并在线计算 PartField 特征。使用 AdamW 优化器，全局 batch size 128，在 8 张 H100 上训练 100K 迭代。

## 实验关键数据

### 主实验（铰接部件分割）

| 方法 | Lightwheel gIoU↑ | Lightwheel PC↓ | PartNet gIoU↑ | PartNet PC↓ |
|-----|------------------|----------------|---------------|-------------|
| Naive Baseline | 0.018 | 0.285 | 0.296 | 0.210 |
| PartField† | 0.079 | 0.106 | 0.183 | 0.123 |
| SINGAPO (1@10)† | -0.050 | 0.221 | 0.271 | 0.117 |
| Articulate AnyMesh† | 0.172 | 0.190 | 0.383 | 0.104 |
| **Particulate†** | **0.332** | **0.168** | **0.880** | **0.003** |

†: 使用 mesh 连通分量优化

### 铰接运动预测（全铰接几何比较）

| 方法 | Lightwheel gIoU↑ | Lightwheel OC↓ | PartNet gIoU↑ | PartNet OC↓ |
|-----|------------------|----------------|---------------|-------------|
| SINGAPO (1@10)† | -0.056 | 0.019 | 0.264 | 0.041 |
| Articulate AnyMesh† | 0.158 | 0.010 | 0.378 | 0.022 |
| **Particulate†** | **0.305** | **0.009** | **0.843** | **0.003** |

### 消融实验

| 配置 | gIoU↑ | 说明 |
|------|-------|------|
| Full model | 0.332 | 完整模型（Lightwheel, 带 connectivity） |
| w/o PartField features | 较低 | 去掉语义特征后泛化性下降 |
| w/o connected comp. refinement | 0.183 | 不用 mesh 连通分量优化后大幅下降 |
| w/o over-parameterized axis | 较低 | 直接预测轴位置导致偏移 |

### 关键发现
- Particulate 在 PartNet-Mobility 上 gIoU 达到 0.880，远超第二名 Articulate AnyMesh 的 0.383
- 在更具挑战性的 Lightwheel 数据集上优势依然明显（0.332 vs 0.172）
- PartField 和 P3SAM 预测的是语义部件，与铰接部件定义不同，导致预测不匹配
- 基于 VLM 的方法（Articulate AnyMesh）无法处理内部不可见部件（如微波炉内部托盘）
- Particulate 能很好泛化到 AI 生成的 3D 资产（Hunyuan3D 生成的物体）

## 亮点与洞察
- **旋转轴的过参数化投票机制非常精巧**：让每个点投票轴位置再取中位数，巧妙利用了"轴位置必须在所有点的正交投影平面上一致"这个几何约束，避免了直接回归的过拟合问题
- **DETR 风格的 part query 适配铰接预测**：用可学习的 part query 优雅地处理"部件数量未知"的问题，且能同时预测部件间的运动学关系
- **数据增强策略可迁移**：每次迭代随机采样不同铰接状态训练，等价于做了大量数据增强，使模型能理解各种姿态

## 局限与展望
- $P_{max}=16$ 限制了能处理的最大部件数，对于非常复杂的铰接物体（如机器人手臂）可能不够
- 仅考虑部分刚性铰接（revolute/prismatic），不支持柔性形变
- 训练数据仅 3800 个物体，规模较小，扩大数据可能进一步提升泛化
- 推理时需要 PartField 特征计算，增加了一定开销
- 新引入的 Lightwheel 基准仅 243 个物体，规模有限

## 相关工作与启发
- **vs SINGAPO**: SINGAPO 用 part retrieval 组装铰接物体，受限于 part 库覆盖范围且仅训练少数类别。Particulate 直接端到端预测，不依赖检索
- **vs Articulate AnyMesh**: 后者用 VLM 推理铰接，通用性好但需 15min/物体且无法处理内部部件。Particulate 10s 完成且能处理内部结构
- **vs PartField**: PartField 做语义分割而非铰接分割，二者定义不同。Particulate 将 PartField 作为输入特征使用，取长补短

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个从静态 3D mesh 前馈预测完整铰接结构的方法，旋转轴过参数化设计有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集、详细消融、新评估协议、丰富可视化，比较非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 形式化定义清晰，方法描述详尽，Related Work 表格总结到位
- 价值: ⭐⭐⭐⭐ 在 3D 铰接理解这个实用场景有重要意义，结合 3D 生成模型可实现端到端物体创建

<!-- RELATED:START -->

## 相关论文

- [PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery](panovggt_feed-forward_3d_reconstruction_from_panoramic_imagery.md)
- [Speed3R: Sparse Feed-forward 3D Reconstruction Models](speed3r_sparse_feed-forward_3d_reconstruction_models.md)
- [VGG-T3: Offline Feed-Forward 3D Reconstruction at Scale](vgg-t3_offline_feed-forward_3d_reconstruction_at_scale.md)
- [PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis](physgm_large_physical_gaussian_4d_synthesis.md)
- [AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)

<!-- RELATED:END -->
