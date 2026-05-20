---
title: >-
  [论文解读] Structure-Aware Correspondence Learning for Relative Pose Estimation
description: >-
  [CVPR 2025][人体理解][相对位姿估计] 提出结构感知对应学习方法(SAC-Pose)，通过学习能代表物体结构的关键点，并基于图像间结构感知特征直接回归3D-3D对应关系（无需显式特征匹配），显著提升未见类别物体的相对位姿估计精度。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "相对位姿估计"
  - "结构感知关键点"
  - "3D对应"
  - "无特征匹配"
  - "SVD求解"
---

# Structure-Aware Correspondence Learning for Relative Pose Estimation

**会议**: CVPR 2025  
**arXiv**: [2503.18671](https://arxiv.org/abs/2503.18671)  
**代码**: [https://github.com/Cyhhzo02/SAC-Pose-code](https://github.com/Cyhhzo02/SAC-Pose-code)  
**领域**: 人体姿态/3D视觉  
**关键词**: 相对位姿估计, 结构感知关键点, 3D对应, 无特征匹配, SVD求解

## 一句话总结

提出结构感知对应学习方法(SAC-Pose)，通过学习能代表物体结构的关键点，并基于图像间结构感知特征直接回归3D-3D对应关系（无需显式特征匹配），显著提升未见类别物体的相对位姿估计精度。

## 研究背景与动机

1. **领域现状**：相对位姿估计旨在从一对图像估计物体的相对旋转，对于实现物体无关的位姿估计有重要价值。主流方法分三类：2D对应（SuperGlue/LoFTR）、假设验证（RelPose/RelPose++）、3D对应（DVMNet）。
2. **现有痛点**：2D方法在大视角差/小重叠区域下匹配失败；假设验证方法依赖离散采样，计算成本高且无法建模连续位姿空间；3D方法将2D特征提升到3D体素后做密集匹配，但不可见区域的3D特征推断不可靠，且立方级复杂度很高。
3. **核心矛盾**：现有3D对应方法依赖显式特征匹配，但从单视图2D表面特征推断不可见区域的3D特征本身就不可靠，导致匹配错误。
4. **本文目标**：设计一种无需显式特征匹配就能建立可靠3D-3D对应关系的方法。
5. **切入角度**：模仿人类"拼装能力"——人看到手提箱的前面和后面，即使重叠区域很小，也能通过结构信息（形状、把手位置、颜色pattern）推断拼合方式。
6. **核心 idea**：用一组结构化关键点表示物体结构，再通过结构感知特征交互直接回归3D对应坐标，绕过匹配步骤。

## 方法详解

### 整体框架

输入query和reference两张图像 → 共享特征提取器 + 对称注意力 → 结构感知关键点提取（独立提取） → 结构感知对应估计（自/交叉注意力） → 2D关键点提升至3D + 回归参考坐标系中的3D坐标 → wSVD求解相对旋转。

### 关键设计

1. **结构感知关键点提取模块 (SA-KPE)**

    - 功能：从特征图中自适应地选取能代表物体结构的稀疏关键点
    - 核心思路：初始化一组可学习query $\mathbf{Q} \in \mathbb{R}^{N_{kpt} \times C}$，通过交叉注意力与图像特征交互得到图像自适应的关键点检测器 $\tilde{\mathbf{Q}}_q$。计算检测器与特征的相似度生成热力图 $\mathbf{H}_q = \text{softmax}(\tilde{\mathbf{Q}}_q \cdot \mathbf{F}_q'^{\top})$，再通过热力图加权平均得到关键点坐标和特征。为防止关键点聚集，设计图像重建损失（从关键点特征+坐标重建前景图像），包含L2像素损失+VGG感知损失。
    - 设计动机：密集像素特征引入背景噪声且计算贵，随机采样缺乏一致性；关键点方式以更少计算量（50.05G vs 55.26G MACs）实现更好性能（mAE 14.2° vs 15.52°）。图像重建约束迫使关键点分散到语义丰富区域。

2. **结构感知对应估计模块 (SA-CE)**

    - 功能：从关键点特征中提取结构感知特征，用于3D对应关系回归
    - 核心思路：先用带ROPE位置编码的自注意力聚合同一图像内的关键点结构信息 $\tilde{\mathbf{F}}_{kpt,q} = \text{MHSA}(\mathbf{F}_{kpt,q} \circledast R(\mathbf{X}_{kpt,q}))$；再用交叉注意力聚合参考图像的关键点特征。有了结构感知特征后，用MLP回归伪深度 $d_{i,q}$ 将2D关键点提升到query坐标系的3D空间 $\mathbf{x}^{(\mathcal{Q})}_{i,q}$，再用另一个MLP回归其在参考坐标系中的3D坐标 $\mathbf{x}^{(\mathcal{R})}_{i,q}$ 及置信度 $c_i$。
    - 设计动机：自注意力+ROPE让每个关键点感知自身在物体结构中的相对位置（图内结构），交叉注意力让关键点理解两个视图的互补结构信息（图间结构），二者结合使网络能"脑补"两个部分如何拼合。

3. **基于wSVD的端到端位姿求解**

    - 功能：从3D-3D对应关系求解最优旋转矩阵
    - 核心思路：计算加权协方差矩阵 $\mathbf{H} = \sum_i c_i \mathbf{x}^{(\mathcal{Q})}_{i,q}(\mathbf{x}^{(\mathcal{R})}_{i,q})^\top$，对其做SVD分解 $\mathbf{H}=\mathbf{U}\Sigma\mathbf{V}^\top$，则最优旋转 $\Delta\mathbf{R}=\mathbf{V}\mathbf{U}^\top$。置信度 $c_i$ 由网络预测，自动降低不可靠对应的权重。
    - 设计动机：wSVD提供了从对应关系到旋转的可微闭式解，保证端到端训练。置信度权重让模型自动识别哪些对应可信。

### 损失函数 / 训练策略

总损失 $\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{pts} + \lambda_2 \mathcal{L}_{rec} + \lambda_3 \mathcal{L}_{rot} + \lambda_4 \mathcal{L}_{mask}$。其中 $\mathcal{L}_{pts}$ 使用对称stop-gradient约束3D坐标预测精度（含置信度加权）；$\mathcal{L}_{rot}$ 用L1损失对齐6D旋转表示；训练时query和reference对称使用以增加数据效率。

## 实验关键数据

### 主实验

| 方法 | 类型 | CO3D mAE↓ | CO3D Acc@15°↑ | Objaverse mAE↓ | LineMOD mAE↓ |
|------|------|-----------|-------------|---------------|-------------|
| SuperGlue | 2D | 67.2° | 37.7% | 102.4° | 64.8° |
| LoFTR | 2D | 77.5° | 33.1% | 134.1° | 84.5° |
| DVMNet | 3D | 19.9° | 62.3% | 20.2° | 36.8° |
| **Ours** | 3D | **14.2°** | **80.2%** | **15.3°** | **27.2°** |

### 消融实验

| 配置 | mAE↓ | Acc@30°↑ | Acc@15°↑ | MACs(G) |
|------|------|---------|---------|---------|
| Dense features | 15.52 | 92.65 | 78.20 | 55.26 |
| Random points | 20.15 | 88.34 | 68.99 | 49.59 |
| **Keypoint (ours)** | **14.2** | **93.6** | **80.2** | 50.05 |
| w/o Self-Attn | 17.79 | 90.22 | 72.48 | - |
| w/o Cross-Attn | 18.53 | 89.38 | 70.99 | - |

### 关键发现

- 相比DVMNet，CO3D上mAE降低约6°（19.9→14.2），Acc@15°提升近18个百分点，证明直接回归对应优于密集3D匹配
- 去掉自注意力(+3.6° mAE)和交叉注意力(+4.3° mAE)后性能均显著下降，说明图内和图间结构信息都至关重要
- 关键点方法比密集特征少10%计算量却性能更优，说明结构化稀疏表示比密集表示更高效

## 亮点与洞察

- **"绕过匹配直接回归"的思路很聪明**：避免了在特征空间做显式匹配（容易受不可见区域影响），改为让网络直接学从结构特征到3D坐标的映射。这种思路可迁移到点云配准等领域。
- **图像重建作为关键点分散约束**：用重建目标驱动关键点覆盖语义丰富区域，比直接加分散正则更自然且有效。
- **对称训练策略**：query和reference互换使用，相当于数据翻倍，简单但有效。

## 局限与展望

- 目前只估计旋转（3DoF），平移部分假设可由2D检测解决，但这在实际应用中不一定成立
- 关键点数量 $N_{kpt}$ 需要预设，对不同复杂度的物体可能需要自适应调整
- 只在物体级别测试，未验证在场景级别（大规模、多物体）的效果
- 图像重建模块增加了训练开销，推理时可去掉但训练需要额外计算

## 相关工作与启发

- **vs DVMNet**: DVMNet做3D体素密集匹配，本文用稀疏关键点直接回归，计算更少效果更好
- **vs LoFTR/SuperGlue**: 这些2D方法在大视角差下失败，本文通过3D对应回归天然支持大视角
- 3D关键点+结构感知的思路可以扩展到多视图重建、6DoF物体姿态估计等任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 结构感知关键点+直接回归对应的组合新颖
- 实验充分度: ⭐⭐⭐⭐ 三个数据集+详细消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，人类拼装类比很直观
- 价值: ⭐⭐⭐⭐ 在相对位姿估计领域提供了新的SOTA方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Co-op: Correspondence-based Novel Object Pose Estimation](co-op_correspondence-based_novel_object_pose_estimation.md)
- [\[CVPR 2025\] Probabilistic Prompt Distribution Learning for Animal Pose Estimation](probabilistic_prompt_distribution_learning_for_animal_pose_estimation.md)
- [\[CVPR 2026\] COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](../../CVPR2026/human_understanding/cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)
- [\[ECCV 2024\] Pose-Aware Self-Supervised Learning with Viewpoint Trajectory Regularization](../../ECCV2024/human_understanding/pose-aware_self-supervised_learning_with_viewpoint_trajectory_regularization.md)
- [\[ECCV 2024\] GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence](../../ECCV2024/human_understanding/gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)

</div>

<!-- RELATED:END -->
