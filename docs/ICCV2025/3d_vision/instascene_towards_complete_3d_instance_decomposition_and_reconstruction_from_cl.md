---
title: >-
  [论文解读] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes
description: >-
  [3D视觉] InstaScene 提出统一的杂乱场景实例分解与完整重建框架，通过追踪高斯光栅化构建空间对比学习实现精准实例分割，并设计 in-situ 生成管线利用已知观测和几何线索引导 3D 生成模型重建完整物体。
tags:
  - 3D视觉
---

# InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2507.08416](https://arxiv.org/abs/2507.08416)
- **代码**: [项目页面](https://zju3dv.github.io/instascene/)
- **领域**: 3D视觉
- **关键词**: 3D实例分割, 场景分解, 完整重建, 高斯Splatting, 对比学习, 3D生成先验

## 一句话总结
InstaScene 提出统一的杂乱场景实例分解与完整重建框架，通过追踪高斯光栅化构建空间对比学习实现精准实例分割，并设计 in-situ 生成管线利用已知观测和几何线索引导 3D 生成模型重建完整物体。

## 研究背景与动机

人类能够自然地在杂乱环境中识别并脑补被遮挡的物体，但赋予机器人同样的感知能力仍极具挑战：

**整体建模 vs 实例理解**：通用 3D 重建方法（NeRF/3DGS）将场景视为整体，无法进行实例级交互

**分割但不完整**：开放集场景理解方法（OpenScene/LangSplat）能查询和分割物体，但无法恢复完整几何

**生成但不对齐**：类别特定的生成方法能预测完整形状，但与真实场景的尺寸、外观不一致

**2D 先验噪声**：现有方法依赖 2D 分割掩码提升到 3D，但杂乱场景中掩码噪声大、跨视角不一致

**实际需求**：机器人操作、场景编辑、仿真等下游任务需要完整的实例级 3D 模型

## 方法详解

### 整体框架

InstaScene 以重建好的 2D Gaussian Splatting 场景为输入，包含三个阶段：

1. **掩码聚类与过滤**：追踪高斯光栅化进行跨视角 2D 掩码匹配
2. **空间对比学习**：联合 2D/3D 掩码监督训练实例特征场
3. **In-situ 生成**：对分解出的不完整物体，利用生成模型补全

### 空间高斯追踪器 (Spatial Gaussian Tracker)

核心问题：如何将多视角 2D 分割掩码关联到同一个 3D 实例？

方法：利用高斯光栅化的可追踪性。对于每个视图 $I_i$ 的掩码 $m_{i,j}$，收集光栅化时对该掩码区域贡献显著（透射率 > 0.5）的高斯点集合，构成空间追踪器 $P_{i,j}$。

跨视角匹配通过视图共识率判断：

$$\mathcal{C}(P_{i,j}, P_{k,l}) = \frac{N_{contain}(P_{i,j}, P_{k,l})}{N_{vis}(P_{i,j}, P_{k,l})}$$

当 $\mathcal{C}$ 超过 0.9 时，认为两个掩码属于同一实例。此外，检测并过滤欠分割掩码（一个掩码追踪器同时与同帧多个追踪器相交）。

### 空间对比学习

动机：3D 掩码足够鲁棒但稀疏（DBSCAN 过滤了有意义的点），2D 掩码更密集但含噪声。两者互补引导。

为每个高斯附加 16 维特征 $f_i^{3d}$，通过光栅化渲染到像素级特征 $\mathbf{F}$。对比学习损失：

$$\mathcal{L}_{CF}(\mathcal{F}) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{|\{f_i\}|}\log\frac{\exp(f_i^j \cdot \bar{f}_i / \phi_i)}{\sum_{k=1}^{N}\exp(f_i^j \cdot \bar{f}_k / \phi_k)}$$

总训练损失融合三个层面的监督：

$$\mathcal{L}_{\mathcal{F}} = \lambda_1 \mathcal{L}_{CF}(\mathbf{F}_i) + \lambda_2 \mathcal{L}_{CF}(\bar{\mathbf{F}}_i) + \lambda_3 \mathcal{L}_{CF}(\mathbf{f}_i^{3d})$$

- $\mathbf{F}_i$：单视图内 2D 掩码特征对比
- $\bar{\mathbf{F}}_i$：相邻视图间跨视角掩码特征对比
- $\mathbf{f}_i^{3d}$：3D 高斯点特征对比（由聚类后的 3D 掩码监督）

分割时，计算每个高斯特征与粗略 3D 实例特征均值的余弦相似度，阈值 $\tau_{seg} = 0.9$。

### In-Situ 生成

目标：重建的完整物体不仅几何完整，还需与真实场景在外观和尺度上对齐。

**全条件扩散**：利用所有已知信息控制 3D 扩散模型（MVDFusion），交替使用多个最优视角作为条件：

$$\bar{\epsilon}_\theta^n = \frac{1}{N_k}\sum_{k=1}^{N_k}\epsilon_\theta^n(x_t^n, y^k, \hat{\pi}_n^k)$$

**几何感知特征扭曲**：在扩散每步迭代中，将已知视图的含噪潜在特征通过渲染深度投影到目标视图可见像素，使用 2DGS 融合网格的表面法线过滤背面投影，强制已知区域的一致性。

**遮挡感知视点选择**：
1. 围绕分割物体设置 16 个视点
2. 选择场景遮挡最少的视点作为生成条件
3. 被遮挡视点的内容由生成模型补充
4. 用原始观测 + 生成视图联合微调物体 2DGS

## 实验关键数据

### 主实验：3D 实例分割 (LERF-Mask 数据集)

| 方法 | Figurines | Teatime | Kitchen | 平均 mIoU (%) |
|------|-----------|---------|---------|---------------|
| LangSplat | 58.1 | 73.0 | 50.7 | 60.6 |
| GSGrouping | 59.0 | 72.3 | 43.1 | 58.1 |
| **InstaScene** | **85.7** | **93.7** | **77.3** | **85.6** |

InstaScene 在平均 mIoU 上比最优基线提升 **25+ 个百分点**，在杂乱 Kitchen 场景优势尤其突出。

### 消融实验：空间对比学习组件

| 模型配置 | Figurines | Teatime | Kitchen | 平均 |
|----------|-----------|---------|---------|------|
| 仅噪声 2D 掩码 | 80.3 | 90.1 | 71.2 | 80.5 |
| 仅 3D 掩码 | 81.5 | 88.5 | 67.0 | 79.0 |
| + 过滤后 2D 掩码 | 83.9 | 91.4 | 75.4 | 83.6 |
| + 跨视角 2D 掩码 (Full) | **85.7** | **93.7** | **77.3** | **85.6** |

### In-Situ 生成定量对比 (Replica-CAD 数据集)

| 方法 | PSNR(已知)↑ | PSNR(未知)↑ | CD↓ | F1↑ | Vol-IoU↑ |
|------|-------------|-------------|-----|-----|----------|
| MVDFusion (单视角) | 17.19 | 17.46 | 0.081 | 0.150 | 0.531 |
| InstantMesh (单视角) | 23.05 | 22.83 | 0.045 | 0.382 | 0.570 |
| SpaRP (多视角) | 25.09 | 23.03 | 0.037 | 0.406 | 0.590 |
| **InstaScene** | **32.57** | **29.02** | **0.016** | **0.767** | **0.716** |

InstaScene 在已知区域渲染质量接近原始 2DGS（31.67 dB），在未知区域也远超所有基线。

### 关键发现

1. **互补引导有效**：2D 掩码提供密度，3D 掩码提供鲁棒性，两者联合显著优于单独使用
2. **追踪光栅化关键**：利用高斯光栅化的可追踪性构建跨视角匹配，比视频追踪和 CLIP 特征更可靠
3. **In-situ vs 通用生成**：通用 Image-to-3D 方法在杂乱场景中出现把手断裂、尺度失配等严重问题
4. **几何扭曲必要**：仅交替视角条件仍有 floaters 和不一致性，加入几何特征扭曲后显著改善

## 亮点与洞察

1. **问题定义新颖**：首次系统性地将场景分解和完整重建统一到一个框架，填补了感知→完整建模的 gap
2. **追踪光栅化的巧妙利用**：高斯 Splatting 的光栅化过程本身包含丰富的空间关联信息，被自然地用于跨视角掩码匹配
3. **渐进式信息聚合**：分割阶段产出的空间先验（几何、视角、掩码）直接指导后续生成模块
4. **对真实杂乱场景的鲁棒性**：在 ZipNeRF 数据集的复杂真实场景中也展示了良好效果
5. **应用前景**：分解出的完整物体可直接用于场景操作（如搬动婴儿车），为机器人操作提供基础

## 局限性

1. **不支持动态物体**：当前框架假设静态场景
2. **透明/高反射物体**：无法处理透明或高反射表面的分解和重建
3. **生成模型域差**：In-situ 生成的质量受限于 3D 生成模型的训练数据域
4. **计算成本**：需要先完成场景重建 → 特征场训练 → 逐物体生成，整体流程较长

## 相关工作与启发

- **LangSplat/GSGrouping**：将语义特征蒸馏到高斯点中，但在杂乱场景中分辨力不足
- **DP-Recon**（并发工作）：也利用生成先验改善稀疏/遮挡区域，但先补几何再补纹理分两步
- **MaskClustering/SAI3D**：利用重投影空间一致性进行 3D 分割，InstaScene 在此基础上加入光栅化追踪
- **启发**：分割和生成不应孤立——分割提供的空间先验是控制生成质量的关键

## 评分

⭐⭐⭐⭐ (4/5)

问题定义具有前瞻性，空间对比学习和 in-situ 生成的设计都很有说服力，实验对比充分且提升显著。局限在于计算成本高、不支持动态/透明场景。是杂乱场景理解领域的重要推进。

<!-- RELATED:START -->

## 相关论文

- [Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion](disentangling_instance_and_scene_contexts_for_3d_semantic_scene_completion.md)
- [CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation](cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)
- [PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)
- [Relative Illumination Fields: Learning Medium and Light Independent Underwater Scenes](relative_illumination_fields_learning_medium_and_light_independent_underwater_sc.md)
- [Discretized Gaussian Representation for Tomographic Reconstruction](discretized_gaussian_representation_for_tomographic_reconstruction.md)

<!-- RELATED:END -->
