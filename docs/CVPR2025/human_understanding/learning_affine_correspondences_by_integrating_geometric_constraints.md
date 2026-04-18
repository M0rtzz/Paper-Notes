---
title: >-
  [论文解读] Learning Affine Correspondences by Integrating Geometric Constraints
description: >-
  [CVPR 2025][image matching] 提出一种融合稠密匹配与几何约束的仿射对应估计框架，通过 Sampson 距离损失监督点匹配、仿射 Sampson 距离损失约束局部仿射变换，在图像匹配和相对位姿估计上超越 SOTA。
tags:
  - CVPR 2025
  - affine correspondence
  - image matching
  - geometric constraints
  - pose estimation
  - dense matching
---

# Learning Affine Correspondences by Integrating Geometric Constraints

**会议**: CVPR 2025  
**arXiv**: [2504.04834](https://arxiv.org/abs/2504.04834)  
**代码**: [GitHub](https://github.com/stilcrad/DenseAffine)  
**领域**: human_understanding  
**关键词**: affine correspondence, dense matching, epipolar geometry, Sampson distance, pose estimation

## 一句话总结

提出一种融合稠密匹配与几何约束的仿射对应估计新框架（DenseAffine），采用两阶段解耦训练：先用 Sampson 距离损失训练稠密点匹配器，再冻结匹配器、用仿射 Sampson 距离损失训练局部仿射变换提取器，在 HPatches 匹配和 MegaDepth 位姿估计上均取得 SOTA。

## 研究背景与动机

**领域现状**: 仿射对应 (Affine Correspondences, ACs) 因包含局部仿射变换（尺度、旋转、形变信息），比纯点对应提供更丰富的几何信息，能显著加速并提升单应性估计、本质矩阵估计等任务的精度。

**现有方案的局限**:
1. **传统检测器方法** (MSER, Hessian-Affine, AffNet): 依赖稀疏关键点检测，在弱纹理、重复纹理区域表现差
2. **合成视图方法** (ASIFT): 依赖检测器且计算代价高
3. **稠密匹配方法** (DKM, RoMa): 能提供精确点对应，但**缺乏局部仿射变换估计**
4. 已有方法未充分利用**对极几何约束**来监督仿射变换学习

**核心动机**: 将稠密匹配的高精度点对应能力与几何约束（对极约束）的有效监督结合，构建从点匹配到完整仿射对应的统一框架。

## 方法详解

### 整体框架

两阶段级联:
1. **第一子网络 (Feature Matching Module)**: 稠密匹配提取精确点对应
2. **第二子网络 (Local Affine Transformation Estimation Module)**: 估计每对匹配点的局部仿射变换

训练采用**解耦策略** (Decoupled Training): 先训练匹配子网络至收敛 → 冻结 → 再训练仿射子网络。

### 关键设计

#### 1. 稠密特征匹配模块

基于 DKM 的稠密匹配范式：
- ResNet-50 编码器提取多尺度特征（粗-细两级）
- 全局 Gaussian 过程回归生成粗匹配 → 细化器迭代精化
- 创新点：加入**基于 Sampson 距离的对极约束损失** $L_{pc}$，使网络学习更多几何信息
- 最终从稠密 warp + certainty 中采样点对应和 32×32 patches

#### 2. 局部仿射变换估计模块

仿射矩阵分解为三个组件（参照 AffNet）：
- **尺度 & 方向**: 两个独立全连接网络 $E_{o,s}$ 分别预测每个 patch 的方向 $O$ 和尺度 $S$，采用**概率协变损失**（离散化角度/尺度后转分类）
- **残差形变**: 独立网络 $E_{aff}$ 回归残差仿射形状矩阵 $\bm{A}''$（det=1 约束）
- 最终仿射对应 = 合成 $(P_i^A, P_i^B, O_i^{A\to B}, S_i^{A\to B}, A_i'')$

#### 3. 仿射 Sampson 距离约束损失

核心创新 — 基于对极约束的仿射损失函数：
- 利用仿射对应与基础矩阵 $\bm{F}$ 的几何关系：$(\bm{F}^T\bm{p}_1)_{(1:2)} + (\bm{A}^T\bm{F}\bm{p}_2)_{(1:2)} = 0$
- 构造仿射 Sampson 距离 $SD_A(E_{AC})$ 衡量预测仿射变换与对极几何的一致性
- 训练目标：$L_{aff} = -\frac{1}{N}\sum_i SD_A(E_{AC}^i)$

此损失无需仿射变换的 GT 标注，仅需基础矩阵（可从深度/位姿计算）。

### 损失函数

**匹配子网络**: $L_m = \sum_l L_{warp} + \lambda L_{conf} + \gamma L_{pc}$
- $L_{warp}$: 预测 warp 与 GT 的 L2 距离
- $L_{conf}$: 置信度的二元交叉熵
- $L_{pc}$: Sampson 距离对极约束损失

**仿射子网络**: $L_{ext} = L_{aff} + L_{ori} + L_{sca}$
- $L_{aff}$: 仿射 Sampson 距离损失
- $L_{ori}, L_{sca}$: 方向和尺度的概率协变损失

## 实验关键数据

### 主实验表

**HPatches 图像匹配 (MMAscore)**:

| 方法 | Overall | Illumination | Viewpoint |
|------|---------|-------------|-----------|
| SuperPoint | 0.658 | 0.715 | 0.606 |
| DISK | 0.763 | 0.813 | 0.716 |
| PoSFeat | 0.775 | 0.826 | 0.728 |
| DKM | 0.819 | 0.869 | 0.772 |
| RoMa | 0.843 | 0.901 | 0.789 |
| **Ours** | **0.851** | **0.908** | **0.798** |

**MegaDepth 相对位姿估计 (AUC)**:

| 方法 | @5° | @10° | @20° |
|------|-----|------|------|
| LoFTR | 52.8 | 69.2 | 81.2 |
| ASpanFormer | 55.3 | 71.5 | 83.1 |
| DKM | 62.1 | 76.7 | 86.4 |
| RoMa | 63.4 | 77.8 | 87.2 |
| **Ours (AC+GC-RANSAC)** | **64.7** | **78.3** | **87.6** |

### 消融实验表

**仿射帧精度 (HPatches)**:

| 方法 | Euclidean Distance ↓ | Cosine Similarity ↑ |
|------|---------------------|---------------------|
| VLFeat | 0.202 | 0.988 |
| AffNet | 0.264 | 0.973 |
| ASIFT | 0.329 | 0.894 |
| **Ours** | **0.123** | **0.994** |

仿射矩阵精度相比最优基线 VLFeat 降低了 **39%** 的欧氏距离。

### 关键发现

- 稠密匹配 + 几何约束的组合显著优于传统稀疏检测器方案
- 仿射 Sampson 距离损失是关键创新，它让网络在**无需仿射 GT** 的条件下学习几何一致的仿射变换
- 解耦训练比联合训练收敛更快、内存占用更低、最终性能更好
- 利用仿射对应进行 GC-RANSAC 位姿估计，比纯点对应 + RANSAC 更准确

## 亮点与洞察

1. **巧妙的无标注监督**: 仿射 Sampson 距离损失仅需基础矩阵而非仿射变换 GT，极大降低了标注需求
2. **稠密匹配的新应用**: 首次将稠密 warp 扩展到完整仿射对应估计，突破了稀疏检测器的瓶颈
3. **解耦训练策略**: 避免了弱监督导致的损失歧义，实际效果优于端到端训练
4. **实际应用价值**: 仿射对应可以用 2AC 求解单应性（而非 4PC）、1AC 求解本质矩阵（而非 5PC），大幅提升 RANSAC 效率

## 局限与展望

1. 分类领域标注为 human_understanding 似乎有误，更适合归类为 **image_matching / geometric estimation**
2. Patch 大小固定为 32×32，**自适应 patch 尺寸**可能提升大变形场景表现
3. 当前仅测试了刚性场景（平面/一般场景），对**非刚性变形**的处理能力未知
4. 尺度/方向估计依赖离散化分类，**连续回归**方案可能更精确

## 相关工作与启发

- **DKM/RoMa**: 稠密匹配的 SOTA，本文在其基础上扩展了仿射估计维度
- **AffNet** (Mishkin et al.): 学习仿射协变区域的先驱，本文在精度上大幅超越
- **GC-RANSAC** (Barath et al.): 图割 RANSAC，与仿射对应配合可进一步提升位姿估计
- **启发**: 对极几何约束作为无标注的监督信号，可推广到其他需要学习几何变换的任务（如光流、场景流估计）

## 评分

⭐⭐⭐⭐ — 方法设计合理、几何约束损失新颖实用、在匹配和位姿任务上均 SOTA；解耦训练策略简单有效。
