---
title: >-
  [论文解读] Co-op: Correspondence-based Novel Object Pose Estimation
description: >-
  [CVPR 2025][人体理解][6DoF位姿估计] 本文提出 Co-op，一个基于对应关系的新物体6DoF位姿估计框架，在粗估计阶段用混合表示（patch级分类+偏移回归）仅42个模板即可快速准确估计初始位姿，在精细化阶段用概率流回归+可微PnP端到端优化，在BOP Challenge七个核心数据集上大幅超越现有方法。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "6DoF位姿估计"
  - "新物体泛化"
  - "对应关系匹配"
  - "混合表示"
  - "概率流回归"
---

# Co-op: Correspondence-based Novel Object Pose Estimation

**会议**: CVPR 2025  
**arXiv**: [2503.17731](https://arxiv.org/abs/2503.17731)  
**代码**: 有（推断，NAVER LABS）  
**领域**: 人体理解  
**关键词**: 6DoF位姿估计, 新物体泛化, 对应关系匹配, 混合表示, 概率流回归

## 一句话总结
本文提出 Co-op，一个基于对应关系的新物体6DoF位姿估计框架，在粗估计阶段用混合表示（patch级分类+偏移回归）仅42个模板即可快速准确估计初始位姿，在精细化阶段用概率流回归+可微PnP端到端优化，在BOP Challenge七个核心数据集上大幅超越现有方法。

## 研究背景与动机

**领域现状**：6DoF物体位姿估计在机器人抓取、增强现实等场景中至关重要。传统方法需要为每个新物体重新训练，实用性受限。基于模型的新物体位姿估计方法利用3D CAD模型，通过模板匹配或特征匹配实现泛化，如 MegaPose、GenFlow 等。

**现有痛点**：(1) 模板匹配方法（如 MegaPose）需要大量模板进行穷举比较，计算开销大；(2) 基于 DINOv2 的特征匹配方法（如 FoundPose、GigaPose）本质上是 detect-and-describe 框架，依赖分割mask作为特征检测器，对噪声mask不够鲁棒；(3) Render-and-compare精细化方法中的流回归容易受不准确光流的影响，RANSAC对异常值分布敏感。

**核心矛盾**：高精度位姿估计需要密集的对应关系和可靠的置信度估计，但现有方法要么粗估计阶段效率低（大量模板），要么精细化阶段鲁棒性差（对离群光流处理不足）。

**本文目标**：设计一个高效且鲁棒的两阶段位姿估计框架，在两个阶段都基于对应关系，以少量模板实现快速准确的粗估计和精确的精细化。

**切入角度**：将位姿估计重新定义为两个图像间的对应关系查找问题。在粗估计阶段用分类（离散化）+回归（连续化）的混合表示，在精细化阶段用概率流建模学习对应关系的不确定性。

**核心 idea**：用 patch 级分类确定大致对应区域（鲁棒），用偏移回归在 patch 内精确定位（精确），两者结合实现少模板下的高精度粗估计；精细化阶段通过学习流的 Laplace 分布参数获得可靠的概率置信度。

## 方法详解

### 整体框架
给定裁剪后的查询图像和物体CAD模型，Co-op 分两阶段：(1) **粗估计**——从42个预渲染模板中找到最匹配的，估计半密集对应关系，用 EPnP+RANSAC 求解初始位姿；(2) **精细化**——根据初始位姿渲染图像，估计查询与渲染图之间的概率密集光流和置信度，通过可微PnP求解精确位姿。两阶段共享 ViT 编码器 + Transformer 解码器的结构。

### 关键设计

1. **混合表示（Hybrid Representation）——粗估计**:

    - 功能：用最少模板（42个）实现快速且准确的初始位姿估计
    - 核心思路：将查询图像和模板都通过 ViT 编码器（降采样16倍）得到特征图，对每个查询patch预测：(1) 分类张量 $\mathcal{C} \in \mathbb{R}^{H/16 \times W/16 \times K}$，表示与模板哪个patch匹配（K=H/16×W/16+1个类，最后一类代表无匹配/遮挡）；(2) 偏移量 $\mathcal{U} \in \mathbb{R}^{H/16 \times W/16 \times 2}$，在匹配的模板patch内进一步精确定位（范围[-0.5, 0.5]）。最终对应位置为 $\mathcal{M}^T_{i,j} = (\text{patch中心} + \mathcal{U}_{i,j}) \times 16$
    - 设计动机：直接回归连续坐标对域迁移不鲁棒，纯分类精度受限于patch分辨率。混合表示结合了分类的鲁棒性（学低层信息，抗域迁移）和回归的精度（sub-patch定位），使得42个模板就能达到之前需要数百模板的效果

2. **概率流回归（Probabilistic Flow Regression）——精细化**:

    - 功能：通过 render-and-compare 精确修正初始位姿
    - 核心思路：在精细化模型后加 DPT 模块实现像素级预测。不同于传统流回归只输出均值，本方法将流建模为 Laplace 分布 $p(Y|\mathcal{I}_Q, \mathcal{I}_R; \theta)$，同时预测均值 $\mu$ 和尺度参数 $b$。流概率 $P_R = P(\|y - \mu\|_1 < R) = 1 - \exp(-R/b)$ 提供了可解释的精度度量。最终的流置信度 $\mathcal{W}$ 由确信度（无遮挡概率）× 敏感度（姿态信息丰富度）× 流概率三者逐元素相乘得到
    - 设计动机：GenFlow 只学置信度不提升流本身的精度，PFA 依赖RANSAC排除不准的流但对异常值分布敏感。概率流让模型同时提高流精度和不确定性估计，通过端到端训练获得最优的6D位姿

3. **可微 PnP 端到端训练**:

    - 功能：将流和置信度直接转化为位姿更新，实现端到端梯度传播
    - 核心思路：使用基于 Levenberg-Marquardt 的可微 PnP 求解器，给定渲染深度图将2D对应转换为3D点，结合置信度 $\mathcal{W}$ 加权后求解 $\mathbf{P}_{\text{refined}}$。整个过程可微分，6D位姿损失梯度直接传回流预测和置信度网络
    - 设计动机：与RANSAC不同，端到端的可微PnP让网络学习什么样的流和置信度组合能产生最好的位姿

### 损失函数 / 训练策略
粗估计阶段：分类用交叉熵损失，偏移回归用 L1 损失。精细化阶段：Laplace负对数似然训练流概率，确信度用二元交叉熵（遮挡vs可见），敏感度通过6D位姿损失端到端学习。可选的 Pose Selection 模块进一步提升精度。

## 实验关键数据

### 主实验

| 方法 | BOP 平均AR | 模板数 | 速度 |
|------|----------|--------|------|
| Co-op | **SOTA** | 42 | 快 |
| MegaPose | 次优 | 576 | 慢 |
| GenFlow | 竞争力 | 多 | 慢 |
| GigaPose | 竞争力 | 42 | 快 |
| FoundPose | 竞争力 | 多 | 中 |

### 消融实验

| 配置 | 关键指标变化 |
|------|-------------|
| 仅分类（无偏移） | 精度受限于patch分辨率(16×16) |
| 仅回归（无分类） | 对域迁移不鲁棒，大误差增多 |
| 混合表示 | 结合两者优势，42模板即达最优 |
| 确定性流（无概率） | 置信度估计不可靠，精细化效果下降 |
| 概率流 + 可微PnP | 最佳，端到端优化显著提升精度 |

### 关键发现
- 混合表示是关键创新：将分类的鲁棒性和回归的精度结合，使42个模板胜过其他方法数百个模板
- 概率流比确定性流+RANSAC更有效：学习不确定性让网络自动聚焦可靠区域
- 端到端可微PnP比独立RANSAC后处理性能更好
- 在BOP Challenge所有七个核心数据集上达到SOTA，特别在遮挡和无纹理物体上优势明显

## 亮点与洞察
- 将对应关系查找贯穿两个阶段的设计非常统一，使模型学习低层几何和结构信息，天然抗域迁移
- 混合表示的"先粗后精"思路（分类锚定区域→偏移精确定位）可迁移到其他需要鲁棒密集匹配的任务（如视觉定位、姿态迁移）
- 流置信度的三因子分解（确信度×敏感度×流概率）具有很好的可解释性，每个因子都有明确的物理含义

## 局限与展望
- 依赖 CNOS 或 SAM-6D 提供的目标检测框，检测失败时整个管线受限
- 仅使用 RGB 输入，缺乏深度信息可能在对称物体上产生歧义
- 在极端遮挡（>80%）场景下，可用的对应关系过少，性能可能显著下降
- 可以探索学习更多模板 viewpoint 的自适应选择策略

## 相关工作与启发
- **vs MegaPose**: 需576个模板穷举比较，计算代价大。Co-op用42个模板+混合表示达到更好效果
- **vs GigaPose**: 同样用少模板，但依赖 DINOv2 特征的 detect-and-describe 框架，对分割mask噪声敏感。Co-op的detector-free方法更鲁棒
- **vs GenFlow**: 使用确定性流+RANSAC做精细化。Co-op的概率流+可微PnP提供更精确和可靠的估计

## 评分
- 新颖性: ⭐⭐⭐⭐ 混合表示和概率流的设计新颖实用
- 实验充分度: ⭐⭐⭐⭐⭐ BOP七个核心数据集全面评测，消融充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示丰富
- 价值: ⭐⭐⭐⭐⭐ 对机器人抓取和增强现实有直接应用价值，SOTA结果

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Structure-Aware Correspondence Learning for Relative Pose Estimation](structure-aware_correspondence_learning_for_relative_pose_estimation.md)
- [\[CVPR 2025\] Any6D: Model-free 6D Pose Estimation of Novel Objects](any6d_model-free_6d_pose_estimation_of_novel_objects.md)
- [\[ICCV 2025\] MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation](../../ICCV2025/human_understanding/mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)
- [\[CVPR 2026\] COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](../../CVPR2026/human_understanding/cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)

</div>

<!-- RELATED:END -->
