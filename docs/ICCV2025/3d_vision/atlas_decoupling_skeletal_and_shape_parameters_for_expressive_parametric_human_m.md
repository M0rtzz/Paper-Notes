---
title: >-
  [论文解读] ATLAS: Decoupling Skeletal and Shape Parameters for Expressive Parametric Human Modeling
description: >-
  [ICCV 2025][3D视觉][Parametric Body Model] 提出 ATLAS 参数化人体模型，通过显式解耦外部表面形状与内部骨骼参数，结合稀疏非线性姿态校正，在 60 万高分辨率扫描上训练，实现比 SMPL-X 更精确可控的人体建模。
tags:
  - ICCV 2025
  - 3D视觉
  - Parametric Body Model
  - Skeleton Decoupling
  - Pose Correctives
  - SMPL
  - Human Mesh
---

# ATLAS: Decoupling Skeletal and Shape Parameters for Expressive Parametric Human Modeling

**会议**: ICCV 2025  
**arXiv**: [2508.15767](https://arxiv.org/abs/2508.15767)  
**代码**: https://jindapark.github.io/projects/atlas  
**领域**: 3D Vision / Human Body Modeling  
**关键词**: Parametric Body Model, Skeleton Decoupling, Pose Correctives, SMPL, Human Mesh

## 一句话总结

提出 ATLAS 参数化人体模型，通过显式解耦外部表面形状与内部骨骼参数，结合稀疏非线性姿态校正，在 60 万高分辨率扫描上训练，实现比 SMPL-X 更精确可控的人体建模。

## 研究背景与动机

参数化人体模型（如 SMPL/SMPL-X）是 3D 人体理解的基础。当前主流方法遵循**以顶点为中心**的范式：先用线性基函数定制表面顶点，然后从表面顶点**回归**内部骨骼关节位置，最后用 LBS 驱动。这引入了三个核心问题：

**关节-顶点耦合**：从表面推导关节导致不正确的关联——SMPL-X 的骨骼关节是不对称的（肘部、脊柱、脚部），且脊柱会随软组织变化而左右偏移

**控制困难**：修改肩宽必须调整多个形状分量，同时不可避免地影响软组织

**关键点拟合引入幻觉**：用关键点拟合时会产生无依据的软组织偏差

ATLAS 的核心 idea：**将形状（软组织）和骨骼显式解耦**，骨骼参数独立于表面形状，可分别控制。

## 方法详解

### 整体框架

ATLAS 分两步生成有形状和姿态的人体网格：
1. **表面定制**：在固定模板骨骼的 A-pose 下定制表面顶点（软组织属性）
2. **骨骼定制 + 姿态驱动**：通过 76 个可控骨骼属性修改内部骨骼，然后用 LBS 同时缩放和驱动姿态

### 关键设计

1. **解耦的表面与骨骼基函数**

    - 表面定制：$\tilde{X}(\beta^s, \beta^f, \theta) = \bar{X} + \mathcal{B}^s(\beta^s, \mathcal{S}) + \mathcal{B}^f(\beta^f, \mathcal{F}) + \mathcal{B}^p(\theta, \mathcal{P})$
    - 此时网格仍对齐到固定模板骨骼，未被姿态驱动
    - 骨骼定制：$\ell = \sigma \oplus t$，包含 15 个体部大小缩放和 61 个骨长度参数
    - 骨骼也有独立的基函数 $\mathcal{B}^k(\beta^k)$ 捕捉常见变异
    - **关键**：关节位置仅由 $\beta^k$ 和 $\theta$ 决定，与顶点形状参数 $\beta^s$ 完全无关

2. **可控骨骼属性（76 维）**

    - 15 个缩放属性：整体身体大小、头、手、脚、各手指
    - 61 个骨长度属性：脊柱、颈部、上下臂、上下腿、手指等
    - 每个属性可独立调整，支持精细定制（如仅加宽肩膀、仅加长手臂）

3. **稀疏非线性姿态校正**

    - 结合两种方案的优势：稀疏线性（STAR，避免虚假相关）+ 密集非线性（GHUM，表达力强）
    - 首先用轻量 MLP 的非线性编码关节 $j$ 及其运动学邻居的姿态角：
    $\text{Non-Linear}_j(\theta) = \text{MLP}(\{R_{6d}(\theta_a) - R_{6d}(\vec{0}) \mid a \in n(j)\})$
    - 然后通过稀疏掩码限制每个关节影响的顶点范围：
    $\mathcal{B}^p_j = \phi(A_j) \odot (P_j \times \text{Non-Linear}_j(\theta))$
    - 掩码 $A_j$ 用归一化测地距离初始化，配合 L1 正则化训练后保持稀疏

4. **单图像拟合管线**

    - 解耦拟合：骨骼参数 $\beta^k$ 仅用关键点+深度项优化，表面参数 $\beta^s$ 仅用轮廓 mask 项优化
    - 利用 Sapiens 的深度预测和前景 mask
    - VAE 姿态先验（在 60 万帧上训练）+ PCA 手部先验
    - 表情拟合采用先对齐再最小化的改进方案

### 损失函数 / 训练策略

训练数据解耦处理：
- 第一步：仅用骨骼参数和姿态优化配准（用三角化关键点正则化关节）
- 第二步：优化表面形状以建模软组织
- 然后用自编码器分别训练骨骼和表面空间

拟合目标函数：
$$E(\beta^s, \beta^f, \beta^k, \theta) = E_{data} + E_{\theta_{body}} + E_{\theta_{hand}} + E_{\beta^s} + E_{\beta^f} + E_{\beta^k}$$

## 实验关键数据

### 主实验（3DBodyTex 拟合）

| 方法 | 32 分量顶点误差 (mm) | 特点 |
|------|---------------------|------|
| SMPL | ~5.5 | 1.8k 扫描训练 |
| STAR | ~5.2 | 稀疏校正 |
| SMPL-X | ~4.8 | 含手脸 |
| SUPR | ~4.6 | 联邦数据集 |
| **ATLAS** | **~3.8** (↓21.6%) | 解耦骨骼+形状 |

在 Goliath-Test（100 个未见扫描）上：ATLAS 2.34mm vs SMPL-X 2.78mm。

单图像拟合（Goliath-Test 200 扫描）：

| 方法 | 顶点误差 (mm) | 关节误差 (mm) |
|------|-------------|-------------|
| SMPLify-X | 87.7 | 73.2 |
| **ATLAS** | **55.4** | **53.7** |

### 消融实验

| 配置 | 拟合误差 (mm) | 说明 |
|------|-------------|------|
| 线性姿态校正 | 1.82 | 传统方案 |
| **非线性姿态校正** | **1.61** | 关节区域改善显著 |
| 无相对深度 | 60.7 | 深度项有效 |
| 无深度+无 mask | 61.8 | 两项均有贡献 |

运行时性能：

| 方法 | 顶点数 | 推理时间 (ms) |
|------|--------|-------------|
| SMPL-X | 10475 | 3.74 |
| ATLAS (SMPL 拓扑) | 6890 | 2.39 |
| ATLAS (高分辨率) | 115834 | 5.37 |

### 关键发现

- 解耦使得更少分量即可达到更低误差：32 分量时 ATLAS 比 SMPL-X 低 21.6%
- 非线性校正在肩部、肘部等复杂关节处改善最显著
- 骨骼属性修改精确保持原始表面细节，反之亦然
- ATLAS 基于 CUDA 优化实现，同分辨率下比 SMPL-X 快 34%

## 亮点与洞察

- **解耦是关键**：将骨骼从顶点回归改为独立参数化，从根本上消除了虚假耦合
- **数据规模惊人**：60 万高分辨率扫描（240 相机）、1.5 万身份、157 种姿态，远超此前所有工作
- **稀疏非线性校正**的设计精巧：测地距离初始化 + L1 正则保证稀疏性，MLP 邻居编码保证非线性表达力
- 解耦拟合管线（骨骼用关键点、形状用轮廓）的设计合理，避免了关键点拟合导致的软组织幻觉

## 局限与展望

- 1.5 万受试者仍无法覆盖人类体型的全部变异
- 高分辨率扫描采集和处理成本高昂，难以进一步扩展
- 未涉及衣物建模
- 手部和面部表情重定向自 FLAME/MANO，非端到端学习

## 相关工作与启发

- 对 SMPL 系列的**根本性改进**：不是修修补补式的改良，而是从建模范式上改变
- BLSM 和 SKEL 也尝试解耦，但各有局限（BLSM 无姿态校正、SKEL 继承 SMPL 形状空间）
- 稀疏+非线性的组合思路可推广到面部、手部等其他部位的参数化建模
- Meta 出品，数据优势明显，工程完成度高

## 评分

- 新颖性: ⭐⭐⭐⭐ 骨骼-形状解耦不全新但工程化到位
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多指标、多消融、单图管线全面评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 有潜力成为下一代标准人体模型

<!-- RELATED:START -->

## 相关论文

- [AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)
- [Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision](fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)
- [NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement](neuraleaf_neural_parametric_leaf_models_with_shape_and_deformation_disentangleme.md)
- [SceneMI: Motion In-betweening for Modeling Human-Scene Interactions](scenemi_motion_in-betweening_for_modeling_human-scene_interaction.md)
- [Sequential Gaussian Avatars with Hierarchical Motion Context](sequential_gaussian_avatars_with_hierarchical_motion_context.md)

<!-- RELATED:END -->
