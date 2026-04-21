---
title: >-
  [论文解读] SceneGraphLoc: Cross-Modal Coarse Visual Localization on 3D Scene Graphs
description: >-
  [ECCV 2024][3D视觉][粗定位] 提出 SceneGraphLoc，将查询图像在由多模态 3D 场景图组成的参考地图中进行粗定位，在不依赖大规模图像数据库的前提下，实现了接近 SOTA 图像级方法的定位精度，同时存储需求降低三个数量级。
tags:
  - ECCV 2024
  - 3D视觉
  - 粗定位
  - 3D场景图
  - 跨模态
  - 对比学习
  - 场景检索
---

# SceneGraphLoc: Cross-Modal Coarse Visual Localization on 3D Scene Graphs

**会议**: ECCV 2024  
**arXiv**: [2404.00469](https://arxiv.org/abs/2404.00469)  
**代码**: 有 (scenegraphloc.github.io)  
**领域**: 3D视觉  
**关键词**: 粗定位, 3D场景图, 跨模态, 对比学习, 场景检索

## 一句话总结

提出 SceneGraphLoc，将查询图像在由多模态 3D 场景图组成的参考地图中进行粗定位，在不依赖大规模图像数据库的前提下，实现了接近 SOTA 图像级方法的定位精度，同时存储需求降低三个数量级。

## 研究背景与动机

粗视觉定位（place recognition）是计算机视觉和机器人中的基础任务，通常被建模为图像检索问题：将待定位图像与大规模位姿图像数据库进行比对。然而，当前最优方法（如 AnyLoc）严重依赖庞大的图像数据库，不仅存储开销大，查询速度也慢。跨模态方法虽然尝试桥接不同数据类型，但通常仅限于两种模态之间的匹配（如图像-点云），应用范围受限。

本文提出了一个全新的问题设定：在由 3D 场景图组成的多模态参考地图中定位查询图像。3D 场景图集成了点云、图像、语义类别、物体属性和关系等多种模态，是一种轻量高效的场景表示方式。这一设定的核心优势在于：场景图一旦构建完成，每个节点只需存储固定大小的嵌入向量，不再需要保存原始图像数据库。

## 方法详解

### 整体框架

SceneGraphLoc 包含两个并行的嵌入生成分支：

1. **场景图节点嵌入**：为场景图中的每个节点（物体实例）生成固定维度的嵌入 $e_v \in \mathbb{R}^D$，融合点云、图像、结构、属性和关系五种模态信息。
2. **查询图像嵌入**：将查询图像切分为规则的 patch，为每个 patch 生成嵌入 $e_q \in \mathbb{R}^D$，表示该 patch 可见的物体。

训练目标是通过对比学习使正样本对（同一物体的 patch 和节点）的嵌入距离趋近于 0，负样本对的距离远离 0。推理时，通过最近邻匹配和相似度打分完成场景检索。

### 关键设计

**多模态节点嵌入**：每个场景图节点融合五种模态：
- **点云嵌入** $e_v^{\mathcal{P}}$：使用 PointNet 从物体级点云中提取几何特征。
- **图像嵌入** $e_v^{\mathcal{I}}$：选取每个物体可见性最大的 top-$K_{view}=10$ 张图像，通过多级 bounding box 裁剪 + DINOv2 提取多层级特征，再用 Transformer encoder 融合多视角信息。
- **结构嵌入** $e_v^{\mathcal{S}}$：用 GAT（图注意力网络）编码物体间的相对位置关系。
- **属性嵌入** $e_v^{\mathcal{A}}$ 和 **关系嵌入** $e_v^{\mathcal{R}}$：分别用词袋特征 + 前馈网络编码。

五种模态通过 softmax 注意力加权拼接后，经两层 MLP 映射到统一维度 $D$：

$$e_v = \text{MLP}\left(\bigoplus_{k \in \mathcal{K}} \frac{\exp(w_k)}{\sum_j \exp(w_j)} e_v^k\right)$$

**查询图像嵌入**：使用 DINOv2 作为 backbone 提取 patch 级特征，再经过 4 层 CNN 残差块 + 3 层 MLP 映射到维度 $D$。相比直接使用全景分割（容易过/欠分割），基于 patch 的策略更加鲁棒。

**场景图-图像相似度**：对于每个候选场景图 $\mathcal{G}_i$，计算所有 patch 与其最近邻节点相似度的平均值：

$$s(\mathcal{G}_i, I) = \frac{1}{|\mathcal{Q}_I|} \sum_{q \in \mathcal{Q}_I} [1 - \delta(e_q, \text{NN}(q))]$$

### 损失函数 / 训练策略

采用双向 N-pair 对比损失，包含静态损失和时序损失：

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{static}} + (1 - \alpha) \cdot \mathcal{L}_{\text{temp}}$$

- **静态损失**：查询图像与同时刻场景图构成正样本对。
- **时序损失**：利用 3RScan 中同一房间不同时间的扫描（物体可能移动、光照变化），增强对时序变化的鲁棒性。

负样本来源包括同一图像中看到不同物体的 patch（图像侧负样本）和不同场景的节点（场景图侧负样本），双向计算使嵌入空间更具判别力。

## 实验关键数据

### 主实验（表格）

在 3RScan 数据集上的 Recall@K 结果（50 场景选择，时序场景）：

| 方法 | 地图模态 | $R^t$@1 | $R^t$@3 | $R^t$@5 | 存储(MB) |
|------|---------|---------|---------|---------|---------|
| LidarCLIP | 点云 | 14.1 | 10.3 | 15.6 | 1000.4 |
| LIP-Loc | 点云 | 12.3 | 18.6 | 15.2 | 1001.0 |
| OpenMask3D | 点云+图像 | 21.1 | 38.1 | 48.0 | 1020.1 |
| **SceneGraphLoc (无图像)** | 点云+其他 | **28.2** | **46.2** | **56.4** | **1005.4** |
| **SceneGraphLoc (含图像)** | 全模态 | **69.3** | **78.6** | **84.4** | **1005.4** |
| CVNet | 图像 | 66.5 | 77.0 | 81.7 | 1239.1 |
| AnyLoc | 图像 | 80.6 | 87.4 | 90.0 | 5720.3 |

### 消融实验（表格）

3RScan 验证集上模态消融（10 场景选择）：

| 点云 | 图像 | 属性 | 结构 | 关系 | R@1 (DINOv2) | $R^t$@1 (DINOv2) |
|------|------|------|------|------|-------------|-----------------|
| ✓ | | | | | 45.2 | 43.9 |
| ✓ | | ✓ | | | 56.3 | 54.8 |
| ✓ | | ✓ | ✓ | | 58.4 | 56.5 |
| ✓ | | ✓ | ✓ | ✓ | 63.7 | 62.7 |
| ✓ | ✓ | | | | - | 80.2 |
| ✓ | ✓ | ✓ | ✓ | ✓ | - | 88.5 |

### 关键发现

1. **跨模态大幅领先**：即使不使用图像模态，SceneGraphLoc 也显著优于其他跨模态方法（LidarCLIP、LIP-Loc、OpenMask3D）。
2. **接近图像方法的精度，存储减少 1000 倍**：含图像版本的 SceneGraphLoc 在 3RScan 上 $R^t$@1 达 69.3%（vs AnyLoc 80.6%），但存储仅 1005 MB（vs AnyLoc 5720 MB）。
3. **每种模态都有贡献**：从仅点云到全模态，R@1 从 45.2% 提升到 63.7%（不含图像）和 88.5%（含图像）。
4. **DINOv2 >> GCVit**：使用 DINOv2 提取图像特征的性能显著优于 GCVit。
5. **推理速度优势**：SceneGraphLoc 从 50 个场景中检索仅需约 1 ms，而 AnyLoc 需要 1826 ms。

## 亮点与洞察

- **新问题设定**：首次提出在多模态 3D 场景图中定位查询图像的任务，是一个非常有前景的轻量化定位范式。
- **知识蒸馏式设计**：映射阶段将多模态信息蒸馏到固定大小嵌入中，推理时无需访问原始模态数据，实现了存储与速度的双重优势。
- **时序鲁棒性**：利用同一场景不同时间的扫描作为正样本，使模型对环境变化具有鲁棒性。

## 局限与展望

- 当查询图像中可见物体种类少（如主要是墙面）时，定位容易失败，因为缺乏足够的判别性信息。
- 在 ScanNet 上使用 SceneGraphFusion 预测的场景图时，由于实例分割不精确和缺少属性标注，性能有所下降。
- 与纯图像方法（AnyLoc）仍有一定精度差距，特别是在场景数量较多时。
- 可以考虑引入文本描述、楼层平面图等更多模态来进一步提升性能。

## 相关工作与启发

- 场景图在具身智能、SLAM、任务规划等领域有广泛应用，本文证明了它在定位任务中的巨大潜力。
- 对比学习 + 多模态融合是跨模态匹配的有效范式，可推广到其他跨模态检索任务。
- DINOv2 的 patch 特征在物体级匹配任务中表现优异，值得在更多视觉定位场景中探索。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 全新的问题设定，首次将场景图用于粗视觉定位
- **技术质量**: ⭐⭐⭐⭐ — 多模态融合设计合理，对比学习框架完善
- **实验充分度**: ⭐⭐⭐⭐ — 两个数据集+多种 baseline+详细消融
- **实用性**: ⭐⭐⭐⭐⭐ — 存储减少 1000 倍，对机器人/AR 部署极具价值
- **总体推荐**: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding](sceneverse_scaling_3d_vision-language_learning_for_grounded_scene_understanding.md)
- [The NeRFect Match: Exploring NeRF Features for Visual Localization](the_nerfect_match_exploring_nerf_features_for_visual_localization.md)
- [Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)

<!-- RELATED:END -->
