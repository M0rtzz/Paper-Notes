---
title: >-
  [论文解读] FoundPose: Unseen Object Pose Estimation with Foundation Features
description: >-
  [ECCV 2024][人体理解][6D位姿估计] FoundPose 利用冻结的 DINOv2 基础模型提取 patch 描述子，通过 bag-of-words 模板检索和 kNN 匹配建立 2D-3D 对应关系，无需任何任务特定训练即可实现未见物体的 6D 位姿估计，在 BOP 基准上显著超越现有 RGB 方法。
tags:
  - ECCV 2024
  - 人体理解
  - 6D位姿估计
  - DINOv2
  - 基础模型
  - 无训练
  - 模板匹配
---

# FoundPose: Unseen Object Pose Estimation with Foundation Features

**会议**: ECCV 2024  
**arXiv**: [2311.18809](https://arxiv.org/abs/2311.18809)  
**代码**: 有 (evinpinar.github.io/foundpose)  
**领域**: 人体理解 / 物体姿态估计  
**关键词**: 6D位姿估计, DINOv2, 基础模型, 无训练, 模板匹配

## 一句话总结

FoundPose 利用冻结的 DINOv2 基础模型提取 patch 描述子，通过 bag-of-words 模板检索和 kNN 匹配建立 2D-3D 对应关系，无需任何任务特定训练即可实现未见物体的 6D 位姿估计，在 BOP 基准上显著超越现有 RGB 方法。

## 研究背景与动机

### 问题定义
6D 物体位姿估计（3D旋转 + 3D平移）是空间 AI 的核心问题，在机器人操作和混合现实中有重要应用。本文聚焦于**基于模型的未见物体位姿估计**：假设有 3D 网格模型可用，但没有足够预算进行大规模数据集渲染和网络训练。

### 现有方法的不足

**早期经典方法**（手工特征匹配、模板匹配）：可处理未见物体，但精度有限

**深度学习方法**（MegaPose、GigaPose、GenFlow）：精度高但需要在海量任务特定数据（数百万张合成图像、数千个不同物体）上预训练，数据集生成代价大

**现有无训练方法**（ZS6D、PoMZ）：ZS6D 使用最后一层 ViT 特征建立对应关系，精度显著低于 FoundPose；PoMZ 需要 RGB-D 输入

### 核心动机
视觉基础模型（如 DINOv2）经自监督训练后具有强大的泛化能力。作者发现 DINOv2 **中间层**的 patch 描述子同时携带**位置信息**和**语义信息**，能够跨越合成-真实域差距建立可靠的 2D-2D 对应关系。这是一个关键洞察：浅层特征偏位置信息，深层特征偏语义信息，中间层（第18层）提供了最佳平衡——当语义信息因物体对称性或缺少纹理而模糊时，位置信息能确保几何一致的匹配。

## 方法详解

### 整体框架

FoundPose 采用经典的"渲染-匹配-求解"范式，但以 DINOv2 特征替代传统手工特征：

1. **离线建模阶段**：渲染物体 3D 模型的 RGB-D 模板，提取 DINOv2 patch 描述子并注册到 3D 空间
2. **在线推理阶段**：裁剪查询图像 → bag-of-words 模板检索 → patch 匹配建立 2D-3D 对应 → PnP-RANSAC 求解位姿 → 特征度量精细化

### 关键设计

#### 1. **模板表示与 3D 注册**

给定纹理映射的 3D 物体模型，渲染 $n$ 个 RGB-D 模板，覆盖 $SO(3)$ 旋转群。对每个模板提取 $14 \times 14$ 像素的非重叠 patch 描述子 $\{\mathbf{p}_{t,i}\}_{i=1}^{m}$，经 PCA 降维到 $d$ 维：

$$\mathbf{p}_{t,i} = \phi_d(\mathbf{p}'_{t,i}), \quad \phi_d: \mathbb{R}^r \mapsto \mathbb{R}^d$$

每个 patch 描述子关联一个 3D 位置 $\mathbf{x}_j$（通过深度通道和相机内参从 2D 投影到 3D），形成模板表示 $T_t = \{(\mathbf{p}_{t,j}, \mathbf{x}_j) | j \in M\}$。

**设计动机**：将 DINOv2 描述子注册到 3D 空间，使得 2D patch 匹配可直接转化为 2D-3D 对应关系，无需额外的深度估计网络。

#### 2. **Bag-of-Words 模板检索**

为高效地从数百个模板中找到与查询最相似的少数几个，将 DINOv2 patch 描述子集成到经典的 bag-of-words 框架中：

- 对所有模板的 patch 描述子进行 $k$-means 聚类，获得视觉词汇表
- 每个模板用 TF-IDF 加权的 bag-of-words 向量表示：$b_i = (n_{i,t}/n_t) \log(N/n_i)$
- 推理时计算裁剪图像的 bag-of-words 向量，用余弦相似度检索 $h$ 个最相似模板

**核心优势**：
- 比 MegaPose 的 render-and-compare 粗估计快 **15倍**
- 仅需 **800 个模板**（vs MegaPose 的 90K+），内存占用低 **25倍**
- 余弦相似度对遮挡鲁棒——被遮挡物体的可见部分仍贡献有效视觉词汇
- 软分配（soft-assign）减轻量化误差

#### 3. **基于中间层 DINOv2 的 Patch 匹配**

对检索到的每个模板，通过 kNN 搜索将查询 crop 的 patch 描述子匹配到模板描述子，建立 2D-3D 对应关系 $C_t = \{(\mathbf{u}_i, \mathbf{x}_i)\}_{i=1}^{m}$。

**关键发现**：使用 DINOv2 ViT-L 第 **18 层**（共 23 层）的描述子效果最佳。浅层到深层，特征从位置信息主导逐渐变为语义信息主导：
- 第 13 层：物体不同侧面有明显不同颜色（强位置信息）
- 第 23 层：整个物体趋于同一颜色（强语义信息）
- 第 18 层：在对称/无纹理物体上，位置信息能**消歧**语义模糊的对应关系

#### 4. **特征度量精细化 (Featuremetric Refinement)**

粗位姿通过 PnP-RANSAC 从 2D-3D 对应关系估计。由于 patch 较大（14×14 px），2D 中心与 3D 投影之间存在偏差。通过 Levenberg-Marquardt 优化最小化特征度量误差进行精细化：

$$(\mathbf{R}_r, \mathbf{t}_r) = \arg\min_{(\mathbf{R}, \mathbf{t})} \sum_{(\mathbf{p}_i, \mathbf{x}_i) \in T_t} \rho\left(\mathbf{p}_i - \mathbf{F}_q\left(\pi(\mathbf{R}\mathbf{x}_i + \mathbf{t})/s\right)\right)$$

其中 $\rho$ 是 Barron 鲁棒损失函数，$\mathbf{F}_q$ 是查询特征图（双线性插值采样），$\pi$ 是 2D 投影函数。

**设计动机**：类似经典光度对齐，但在 DINOv2 特征空间操作，使模板 patch 对齐到查询图中的最优位置，有效补偿粗采样导致的偏差。

### 损失函数 / 训练策略

FoundPose **不需要任何训练**——完全依赖冻结的 DINOv2 权重。离线建模阶段仅需：
- 渲染 800 个模板（约 25° 角间距）
- 提取 patch 描述子并计算 PCA（从所有模板的有效 patch 描述子计算 top-256 分量）
- 计算 k-means 聚类中心（2048 个视觉词汇）
- 整个建模过程 < 5 分钟（1 GPU）

## 实验关键数据

### 主实验

在 BOP 基准 7 个数据集上的 Average Recall (AR) 对比（**无精细化**情况）：

| 方法 | LM-O | T-LESS | TUD-L | IC-BIN | ITODD | HB | YCB-V | 平均 AR | 需要训练 |
|------|------|--------|-------|--------|-------|----|-------|---------|---------|
| **FoundPose** | **39.6** | **33.8** | **46.7** | **23.9** | **20.4** | **50.8** | **45.2** | **37.2** | ✗ |
| GigaPose | 29.9 | 27.3 | 30.2 | 23.1 | 18.8 | 34.8 | 29.0 | 27.6 | ✓ |
| GenFlow | 25.0 | 21.5 | 30.0 | 16.8 | 15.4 | 28.3 | 27.7 | 23.5 | ✓ |
| MegaPose | 22.9 | 17.7 | 25.8 | 15.2 | 10.8 | 25.1 | 28.1 | 20.8 | ✓ |

加上 MegaPose refiner（5 假设精细化）：

| 方法 | 平均 AR | 时间(s) |
|------|---------|---------|
| **FoundPose + Feat. + MegaPose** | **59.6** | 20.5 |
| GigaPose + MegaPose | 57.9 | 7.3 |
| GenFlow + GenFlow | 57.1 | 20.9 |
| MegaPose + MegaPose | 54.7 | 47.4 |

### 消融实验

不同 backbone/层级对粗位姿估计精度的影响：

| 特征提取器 | LM-O | T-LESS | TUD-L | 平均 AR | 说明 |
|-----------|------|--------|-------|---------|------|
| DINOv2 ViT-L layer 18 | 39.6 | 33.8 | 46.7 | 37.2 | 最佳中间层 |
| DINOv2 ViT-L layer 23 | 23.2 | 22.8 | 31.2 | 23.5 | 最后层，语义过强 |
| DINOv2 ViT-S layer 9 | 34.0 | 31.6 | 42.7 | 34.0 | 小模型中间层 |
| SAM ViT-L layer 23 | 2.2 | 12.8 | 9.2 | 10.7 | 不适合位姿估计 |
| Dense SIFT | 3.2 | 2.6 | 6.5 | 7.6 | 经典手工特征 |

模板检索方式消融：

| 检索方式 | 平均 AR | 说明 |
|---------|---------|------|
| Bag-of-Words (本文) | 37.2 | 鲁棒、高效 |
| CLS token 匹配 | 18.2 | 对遮挡敏感 |
| CLS token + 黑色背景 | 25.1 | 有改进但仍不足 |
| Top 模板位姿直接使用 | 17.7 | 检索质量验证 |

### 关键发现

1. **DINOv2 中间层描述子是关键**：第 18 层 vs 第 23 层带来 +13.7 AR 的巨大提升，证明位置-语义平衡的重要性
2. **无训练超越有训练**：FoundPose 无需任何训练即超过 GigaPose（+9.6 AR）、GenFlow（+13.7 AR）、MegaPose（+16.4 AR）
3. **特征度量精细化有效**：+5 AR（单假设），与 MegaPose refiner 互补
4. **内存效率极高**：每物体仅 234MB（vs OSOP/Nguyen 的 5.6GB，低 25 倍）
5. **分割质量是瓶颈**：使用 GT mask 时 AR 提升 +6~19 分

## 亮点与洞察

1. **范式启发**：证明了冻结基础模型 + 经典几何方法的组合可以超越端到端训练方法，为"不训练也能用"提供了强有力的证据
2. **中间层特征的价值**：不同于通常使用最后一层特征的做法，发现中间层特征在位姿估计中更优，因为它兼具位置感知和语义理解能力
3. **Bag-of-Words 的回归**：2003 年的经典文本检索技术在 2024 年与视觉基础模型结合后焕发新生
4. **合成-真实域泛化**：DINOv2 的自监督训练使其 patch 描述子天然跨域一致，无需显式的域适应

## 局限与展望

1. FoundPose 依赖 CNOS 提供的分割 mask，mask 质量是主要误差来源
2. 仅支持 RGB 输入，未利用深度信息（与 FoundationPose 等 RGB-D 方法无法直接比较）
3. Patch 粒度较粗（14×14 px），精确定位依赖精细化步骤
4. 对高度对称物体（如圆柱体）仍存在挑战
5. 未探索更高效的基础模型（如蒸馏模型）对速度的提升潜力

## 相关工作与启发

- **DINOv2** 的特征层分析可迁移到其他需要精确空间对应的任务（如光流、深度估计）
- Bag-of-Words + 基础模型特征的组合思路可用于其他检索任务
- 特征度量对齐框架可扩展到动态物体跟踪场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将基础模型特征与经典几何方法巧妙结合，中间层特征的发现具有启发性
- **实验充分度**: ⭐⭐⭐⭐⭐ — 7 个数据集全面评估，丰富的消融实验涵盖特征层级、检索方式、各种参数
- **写作质量**: ⭐⭐⭐⭐⭐ — 逻辑清晰，动机充分，可视化出色（特征 PCA 可视化非常直观）
- **实用价值**: ⭐⭐⭐⭐⭐ — 无需训练、低内存、高精度，实际部署价值极高

<!-- RELATED:START -->

## 相关论文

- [GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence](gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)
- [RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](../../ICCV2025/human_understanding/raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [U-COPE: Taking a Further Step to Universal 9D Category-Level Object Pose Estimation](u-cope_taking_a_further_step_to_universal_9d_category-level_object_pose_estimati.md)
- [LaPose: Laplacian Mixture Shape Modeling for RGB-Based Category-Level Object Pose Estimation](lapose_laplacian_mixture_shape_modeling_for_rgb-based_category-level_object_pose.md)
- [MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation](../../ICCV2025/human_understanding/mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)

<!-- RELATED:END -->
