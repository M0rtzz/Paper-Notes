---
title: >-
  [论文解读] Towards 3D Objectness Learning in an Open World
description: >-
  [NeurIPS 2025][3D视觉][3D objectness] 提出 OP3Det，一个无需文本提示的类无关开放世界 3D 检测器，通过 2D 基础模型进行 3D 物体发现，并设计跨模态混合专家（MoE）动态融合点云与图像特征，大幅提升新类别物体的召回率。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 3D objectness
  - open-world detection
  - class-agnostic
  - 跨模态
  - SAM
---

# Towards 3D Objectness Learning in an Open World

**会议**: NeurIPS 2025  
**arXiv**: [2510.17686](https://arxiv.org/abs/2510.17686)  
**代码**: https://github.com/op3det (有)  
**领域**: 3D Vision / Open-World Detection  
**关键词**: 3D objectness, open-world detection, class-agnostic, cross-modal MoE, SAM

## 一句话总结

提出 OP3Det，一个无需文本提示的类无关开放世界 3D 检测器，通过 2D 基础模型进行 3D 物体发现，并设计跨模态混合专家（MoE）动态融合点云与图像特征，大幅提升新类别物体的召回率。

## 研究背景与动机

3D 感知系统（自动驾驶、机器人等）面临的核心挑战是：现实世界中物体类别不断变化，系统需要能够定位"所有"物体，而非仅限于训练时见过的类别。

**现有痛点**：

**封闭集 3D 检测器**：只能识别训练时预定义的类别，面对新类别完全失效

**开放词汇 3D 检测器**：依赖人工设计的文本提示进行检测，当词汇不完整或与场景不匹配时，仍无法检测所有物体，导致新类别召回率低

**3D 数据稀缺**：3D 点云数据在规模和标注类别上都极为有限，远不如 2D 领域丰富

**核心矛盾**：如何在 3D 标注类别极度有限的情况下，学习到通用的"3D 物体性"（objectness），使检测器能够发现任意类别的物体？

**切入角度**：既然 2D 领域有大量预训练基础模型（如 SAM）具备强大的零样本泛化能力，那么可以将 2D 模型的零样本能力迁移到 3D 领域，用于学习开放世界的 3D objectness。核心 idea 是：(1) 利用 SAM 进行类无关的 3D 物体发现来扩充训练数据，(2) 设计跨模态 MoE 动态融合多模态特征来学习通用的 3D objectness。

## 方法详解

### 整体框架

OP3Det 采用两阶段设计：
1. **3D 物体发现**（训练前）：使用 SAM 在 RGB 图像上提取类无关掩码 → 多尺度点采样去噪 → 类无关 2D 检测器后处理 → 投影到 3D 空间获取新的 3D 框
2. **跨模态 MoE 训练**（训练中）：体素化点云特征 $F_P$ + 图像特征 $F_I'$ + 多模态拼接特征 $F_M$ → 自注意力编码 → 多模态路由器分配权重 → 模态专家加权融合 → 检测头

### 关键设计

1. **多尺度点采样策略（Multi-scale Point Sampling）**：

    - **功能**：解决 SAM 输出碎片化掩码的问题
    - **核心思路**：SAM 使用 64×64 均匀网格点提示来生成掩码，但输出往往是碎片或物体局部。因此，先根据 IoU 分数和自监督模型注意力值选择最可能属于物体的源点 $(x_s, y_s)$，然后过滤掉 3D 距离超过阈值 $\delta$ 的邻近点，保证局部几何一致性
    - **多尺度融合**：使用 $\delta = (0.2, 0.5, 1, 2)$ 四种尺度分别采样，通过 NMS 合并结果，最后经过类无关 2D 检测器进一步过滤噪声
    - **设计动机**：单一尺度要么过滤不充分（小 $\delta$），要么排除有用物体（大 $\delta$），多尺度结合取长补短

2. **跨模态混合专家（Cross-Modal MoE）**：

    - **功能**：解决开放世界下多模态融合的问题——简单融合（拼接/相加）反而损害性能
    - **核心思路**：先用自注意力分别编码三种特征：$\mathcal{F}_P = \text{SelfAttn}(F_P)$、$\mathcal{F}_I = \text{SelfAttn}(F_I')$、$\mathcal{F}_M = \text{SelfAttn}(F_M)$。然后由多模态路由器 $\mathcal{R}$ 基于多模态特征计算路由概率 $(p_P, p_I, p_M) = \mathcal{R}(\mathcal{F}_M)$，最后通过三个模态专家加权融合：$\mathcal{F} = \sum_{i \in (P,I,M)} p_i \cdot \mathcal{E}_i(\mathcal{F}_i)$
    - **设计动机**：开放世界的类无关二分类中，几何信息（点云）和语义信息（图像）的重要性随场景变化。路由器让模型自适应决定依赖哪种模态，避免跨模态噪声干扰

3. **3D 物体发现的 2D→3D 投影**：

    - **功能**：将 2D 框映射到 3D 空间
    - **核心思路**：通过相机内参 $K$ 和外参 $R_t$ 将 3D 点投影到 2D 空间，找到 2D 框内的点，然后聚类获得 3D 框
    - **后处理**：SAM 的 IoU 预测分数与类无关 2D 检测器的 objectness 分数相乘，用 0.6 阈值过滤低质量发现

### 损失函数 / 训练策略

- 分类损失采用类无关的二元分类损失（前景/背景）
- 其他损失沿用 OV-Uni3DETR 的设计
- 使用 ResNet50 + FPN 作为图像特征提取器，Sparse 3D ResNet 作为体素特征提取器
- 训练使用 AdamW 优化器
- 推理时不需要 SAM 或任何额外模块，直接在点云-图像对上运行

## 实验关键数据

### 主实验

**跨类别泛化（SUN RGB-D & ScanNet）**：

| 方法 | 数据集 | AR_novel | AR_all | AR_base | AP_all |
|------|--------|----------|--------|---------|--------|
| FCAF3D (closed) | SUN RGB-D | 65.3 | 86.5 | 92.7 | 62.0 |
| OV-Uni3DETR (open-vocab) | SUN RGB-D | 62.8 | 82.5 | 88.8 | 57.4 |
| **OP3Det (ours)** | **SUN RGB-D** | **78.8** | **89.7** | **93.1** | **65.4** |
| FCAF3D (closed) | ScanNet | 61.7 | 71.3 | 83.2 | 24.7 |
| OV-Uni3DETR (open-vocab) | ScanNet | 67.6 | 71.6 | 76.5 | 25.9 |
| **OP3Det (ours)** | **ScanNet** | **79.9** | **83.2** | **87.3** | **28.6** |

OP3Det 在新类别上分别提升 13.5%（vs FCAF3D）和 16.0%（vs OV-Uni3DETR）。

**跨数据集泛化**：

| 设置 | 方法 | AR25 | AP25 |
|------|------|------|------|
| ScanNet→SUN RGB-D | FCAF3D | 59.3 | 17.9 |
| ScanNet→SUN RGB-D | **OP3Det** | **73.1** | **22.3** |
| SUN RGB-D→ScanNet | FCAF3D | 47.7 | 12.9 |
| SUN RGB-D→ScanNet | **OP3Det** | **77.9** | **21.2** |

跨数据集场景下提升高达 30% AR25。

### 消融实验

| SAM | 多尺度采样 | CM-MoE | AR_novel | AR_all |
|-----|-----------|--------|----------|--------|
| ✗ | ✗ | ✗ | 54.2 | 84.0 |
| ✓ | ✗ | ✗ | 50.0 | 74.1 |
| ✓ | ✓ | ✗ | 69.2 | 87.9 |
| ✓ | ✓ | ✓ | **78.8** | **89.7** |

- 单独加 SAM 反而下降（碎片掩码引入噪声）
- 多尺度采样使 AR_novel 从 50.0 → 69.2（+19.2%）
- CM-MoE 进一步提升 AR_novel 到 78.8（+9.6%）

| 融合方式 | AR_novel | AR_all |
|---------|----------|--------|
| 仅点云 | 69.2 | 87.9 |
| 特征相加 | 65.4 | 85.6 |
| 特征拼接 | 66.0 | 85.8 |
| **CM-MoE** | **78.8** | **89.7** |

简单融合反而不如单模态，CM-MoE 才能有效利用多模态互补信息。

### 关键发现

- 在类无关设置下，简单将多模态特征拼接/相加会导致 RGB 特征干扰 3D 几何线索
- SAM 的碎片化输出需要精心设计的后处理才能有效用于 3D 场景
- 该方法可直接扩展到室外场景（KITTI）和类特定检测，具有良好的通用性

## 亮点与洞察

1. **问题定义新颖**：首次正式定义并解决"类无关开放世界 3D 物体检测"问题
2. **2D→3D 迁移思路**：巧妙利用 2D 基础模型的零样本能力弥补 3D 数据的不足
3. **多尺度点采样**：有效解决 SAM 碎片化输出问题，是 SAM 应用于 3D 场景的关键技术贡献
4. **动态路由的 MoE**：解决了开放世界下多模态融合的退化问题，是方法论层面的创新

## 局限与展望

- SAM 的推理成本较高，训练前的物体发现阶段耗时
- 室外场景（如 KITTI）中前景稀疏、背景干扰大，提升幅度相对有限
- 类无关设置下 AP 指标无法按类别分别计算，评估粒度受限
- 未探索更大规模的 2D 基础模型（如 SAM 2）或更多模态（如深度估计）

## 相关工作与启发

- **SAM 在 3D 中的应用**：SAM3D、OpenMask3D 等用 SAM 做 3D 分割，但直接使用会引入噪声。本文的多尺度点采样策略对 SAM→3D 的迁移有重要参考价值
- **多模态融合**：BEVFusion、SparseFusion 等方法重在融合完整性，本文指出在开放世界下应该**动态选择**模态而非一味融合
- **启发**：在数据稀缺的 3D 任务中，利用 2D 基础模型做自动标注是一个有前景的方向

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次定义类无关开放世界 3D 检测问题，框架设计有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 跨类别+跨数据集+跨场景+消融，实验非常全面
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法描述完整
- 价值: ⭐⭐⭐⭐ 对开放世界 3D 感知有重要推动作用

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] EA3D: Online Open-World 3D Object Extraction from Streaming Videos](ea3d_online_open-world_3d_object_extraction_from_streaming_videos.md)
- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[CVPR 2025\] Open-World Amodal Appearance Completion](../../CVPR2025/3d_vision/open-world_amodal_appearance_completion.md)
- [\[CVPR 2025\] Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces](../../CVPR2025/3d_vision/open-vocabulary_functional_3d_scene_graphs_for_real-world_indoor_spaces.md)
- [\[CVPR 2025\] DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos](../../CVPR2025/3d_vision/depthcrafter_generating_consistent_long_depth_sequences_for_open-world_videos.md)

<!-- RELATED:END -->
