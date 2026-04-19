---
title: >-
  [论文解读] Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding
description: >-
  [ICCV 2025][目标检测][视频时序定位] 提出 SDST（Sparse-Dense Side-Tuner），首个无锚框（anchor-free）的 Side-Tuning 架构，通过稀疏-稠密双流设计同时处理时刻检索（MR）和高光检测（HD），并提出 Reference-based Deformable Self-Attention（RDSA）解决可变形注意力的上下文缺失问题，在 QVHighlights、TACoS、Charades-STA 上取得 SOTA 或高度竞争性结果，同时将可训练参数量减少至现有 SOTA 的 27%。
tags:
  - ICCV 2025
  - 目标检测
  - 视频时序定位
  - Side-Tuning
  - 参数高效微调
  - 可变形注意力
  - InternVideo2
---

# Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding

**会议**: ICCV 2025  
**arXiv**: [2507.07744](https://arxiv.org/abs/2507.07744)  
**代码**: [GitHub](https://github.com/davidpujol/SDST)  
**领域**: 目标检测  
**关键词**: 视频时序定位, Side-Tuning, 参数高效微调, 可变形注意力, InternVideo2

## 一句话总结

提出 SDST（Sparse-Dense Side-Tuner），首个无锚框（anchor-free）的 Side-Tuning 架构，通过稀疏-稠密双流设计同时处理时刻检索（MR）和高光检测（HD），并提出 Reference-based Deformable Self-Attention（RDSA）解决可变形注意力的上下文缺失问题，在 QVHighlights、TACoS、Charades-STA 上取得 SOTA 或高度竞争性结果，同时将可训练参数量减少至现有 SOTA 的 27%。

## 研究背景与动机

视频时序定位（VTG）任务需要根据文本查询在视频中定位特定时刻（MR）并检测高光片段（HD）。现有方法面临以下关键问题：

**冻结特征的局限性**：大多数方法仅使用冻结预训练骨干网络（如 CLIP）的最后一层特征，当预训练分布与下游任务之间存在较大偏移时，性能显著下降。这在 VTG 中尤为突出——用图像域骨干处理视频域任务。

**全量微调不可行**：完整微调大型视觉-语言模型计算代价过高。虽然参数高效微调（PEFT）方法（如 Prompt、Adapter）能减少可训练参数，但仍需通过整个骨干进行反向传播，内存消耗大。

**现有 Side-Tuning 方法的缺陷**：R2-Tuning 是首个用于 VTG 的 Side-Tuning 方法，但它采用基于锚框的设计，从帧级精细化的角度处理问题，忽略了 MR 任务的内在稀疏性。实验表明，这种基于锚框的方法在 MR 任务上表现不佳。

**可变形注意力的上下文限制**：现有无锚框方法（如 DETR 系列）的核心——可变形注意力模块——在交叉注意力场景中存在隐含的上下文缺失问题。由于查询和键来自不同空间，CNN 偏移预测器无法为查询提供键/值空间的上下文信息，导致偏移量坍缩到初始值附近。

## 方法详解

### 整体框架

SDST 是一个双流 Side-Tuning 架构，附着在冻结的 InternVideo2-1B 骨干网络的最后 $K$ 个中间层上。输入视频和文本首先通过 InternVideo2 提取 $K$ 个中间视觉-文本表示，然后通过权重共享的 SDST 层递归精细化两个流：

- **稠密流（Dense Stream $\mathcal{D}$）**：精细化帧级嵌入，适用于 HD 任务
- **稀疏流（Sparse Stream $\mathcal{S}$）**：精细化循环解码器查询（recurrent decoder queries），适用于 MR 任务

整个递归过程可表示为：

$$\mathbf{D}^{\ell+1}, \mathbf{R}^{\ell+1}, \mathbf{H}^{\ell+1} = SDST(\mathbf{D}^{\ell}, \mathbf{R}^{\ell}, \mathbf{H}^{\ell}, \tilde{\mathbf{V}}^{\ell}, \tilde{\mathbf{T}}^{\ell})$$

### 关键设计

1. **稠密学习流（Dense Learning Stream）**:

    - 功能：逐层精细化帧级稠密嵌入 $\mathbf{D}^{\ell}$，融合多模态信息并建模时序关系
    - 核心思路：首先将视觉和文本的中间特征投影到共享 $F$ 维空间，通过加权求和将视觉信息融入稠密嵌入：$\mathbf{D}^{\ell} := \beta^{\ell}\mathbf{D}^{\ell} + (1-\beta^{\ell})\mathbf{V}^{\ell}$，其中 $\beta^{\ell}$ 是零初始化的层级参数。然后依次使用交叉注意力注入文本信息、自注意力建模时序关系：$\mathbf{D}^{\ell+1} = PFFN(SA(CA(\mathbf{D}^{\ell}, \mathbf{T}^{\ell}, \mathbf{T}^{\ell})))$
    - 设计动机：帧级嵌入天然适合为每帧预测显著性分数（HD 任务），同时作为稀疏流的基础信号

2. **稀疏学习流（Sparse Learning Stream）**:

    - 功能：精细化循环解码器查询——包括中心-宽度参考 $\mathbf{R}^{\ell} \in \mathbb{R}^{M \times 2}$ 和对应的隐嵌入 $\mathbf{H}^{\ell} \in \mathbb{R}^{M \times F}$
    - 核心思路：首先通过 CA 和 SA 注入文本信息并促进不同时刻提议之间的信息流动，然后通过 RDSA 将视频模态信息注入查询：$\mathbf{H}^{\ell} = PFFN(RDSA(\mathbf{R}^{\ell}, \mathbf{H}^{\ell}, \mathbf{D}^{\ell+1}))$
    - 设计动机：MR 是高度稀疏的任务（视频中可能只有极少量 GT 动作），基于 DETR 的无锚框架构在稀疏检测任务中已被证明优于基于锚框的方法

3. **Reference-based Deformable Self-Attention（RDSA）**:

    - 功能：替代标准可变形交叉注意力，解决查询缺乏键/值空间上下文信息的问题
    - 核心思路：不使用可学习查询 $\mathbf{H}^{\ell}$ 来预测偏移和注意力分数，而是从稠密嵌入中提取三个关键动作嵌入——左端(l)、中心(c)、右端(r)——作为新查询：$\hat{\mathbf{Q}} = \hat{\mathbf{X}}_{\mathcal{Q}} \mathbf{W}_{\mathcal{Q}}^{def}$，其中 $\hat{\mathbf{X}}_{\mathcal{Q}} = CNN(\mathbf{D}^{\ell})[l, c, r] \in \mathbb{R}^{M \times 3F}$。这样查询和键都来自相同的潜在空间（稠密嵌入），自然地解决了交叉注意力中的上下文缺失问题
    - 设计动机：标准可变形 CA 中 $\mathbf{X}_{\mathcal{Q}} \neq \mathbf{X}_{\mathcal{K}}$，CNN 偏移预测器无法获取键空间的上下文信息，导致偏移量坍缩在初始值附近，且无法选取当前估计边界之外的键——这对于将短动作精细化为长动作至关重要

4. **InternVideo2 中间表示提取**:

    - 功能：解决从 InternVideo2 骨干中提取中间视觉表示的池化挑战
    - 核心思路：复用 InternVideo2 最后一层的冻结 AdaptivePool 模块来池化所有中间层的时空 token：$\tilde{\mathbf{V}}^{\ell} = AdaptivePool(\hat{\mathbf{V}}^{\ell})$
    - 设计动机：CLS 池化不适用于 InternVideo2 的中间层（限制空间聚合能力），而为每层训练独立的 AdaptivePool 需要全量反向传播，计算不可行。复用冻结模块既利用了其增强的多模态对齐能力，又无需额外的反向传播和内存开销

### 损失函数 / 训练策略

总损失由三部分组成：$\mathcal{L} = \lambda_5 \mathcal{L}_{HD} + \lambda_6 \mathcal{L}_{MR} + \lambda_7 \mathcal{L}_{align}$

- **HD 损失**：InfoNCE 损失，基于稠密嵌入与池化文本表示的余弦相似度
- **MR 损失**：使用匈牙利匹配后，包含 FocalLoss（分类）、L1 + IoU（边界回归）、L1（actionness 分数），关键是跨所有精细化层级优化以促进收敛
- **对齐损失**：基于 SampledNCE 的视觉-文本对齐，沿 batch 和中间层两个维度应用

## 实验关键数据

### 主实验

| 数据集 | 指标 | SDST (本文) | SG-DETR | R2-Tuning | 参数量 |
|--------|------|------------|---------|-----------|--------|
| QVHighlights val | R1@0.5 | 73.68 | - | 68.71 | 4.1M vs 15M vs 2.7M |
| QVHighlights val | R1@0.7 | 60.90 | - | 52.06 | - |
| QVHighlights val | mAP Avg | 55.60 | 55.64 | 47.59 | - |
| QVHighlights val | HD mAP | 44.00 | 43.91 | 40.59 | - |
| QVHighlights val | HD HIT@1 | 72.00 | 71.47 | 64.32 | - |
| Charades-STA | R@0.7 | **52.6** | 49.5 | 37.0 | - |
| Charades-STA | mIoU | **61.2** | 59.1 | 50.9 | - |
| TACoS | R@0.7 | **32.3** | 29.9 | 25.1 | - |
| TACoS | mIoU | **42.2** | 40.9 | 35.9 | - |

SDST 仅使用 SG-DETR 27% 的参数量即达到竞争甚至超越的性能。

### 消融实验

| 配置 | MR mAP | HD HIT@1 | 说明 |
|------|--------|----------|------|
| CLS 池化 | 50.53 | 64.26 | 最差，限制空间聚合 |
| 平均池化 | 53.44 | 69.68 | 部分缓解但仍不足 |
| AdaptivePool（本文）| **55.60** | **72.00** | 复用冻结池化模块最优 |
| 标准 CA | 42.72 | 69.87 | 严重性能下降 |
| Def. CA | 54.27 | 70.58 | 基线可变形 CA |
| Def. CA + PureInit | 52.92 | 70.32 | 初始化反而有害 |
| RDSA（本文）| **55.60** | **72.00** | 比 Def. CA 提升 1.33 mAP |

### 关键发现

- RDSA 学会了查看当前动作边界之外的帧（偏移量 < -1 或 > +1），这对精细化长动作至关重要
- 中间特征的有用性并非如此前文献所示那样简单；仅使用最后层特征做多步精细化在 K=2,3 时甚至优于中间特征
- 从更浅层采样特征存在"深度-池化权衡"——浅层特征提供互补信息，但分布偏移使冻结 AdaptivePool 效果下降

## 亮点与洞察

- 首次将 Side-Tuning 与无锚框 DETR 架构结合，为稀疏-稠密多任务学习提供了优雅解决方案
- RDSA 深入分析了可变形注意力在 CA 和 SA 场景下的本质区别——SA 中 CNN 偏移预测器天然获得局部上下文，而 CA 中则完全"盲目"
- 首次成功将 InternVideo2 集成到 Side-Tuning 框架中，解决了关键的 token 池化难题
- 对"中间特征 vs 多步精细化"的深入实验分析提供了新的理解视角

## 局限与展望

- 依赖预提取的 InternVideo2 特征，未实现端到端训练
- AdaptivePool 复用的depth-pooling trade-off 限制了利用更浅层特征
- 仅在 VTG 相关数据集上验证，未拓展到更广泛的视频理解任务
- Side-Tuning 范式对骨干网络的选择敏感，从 CLIP 切换到 InternVideo2 需要非平凡的工程挑战

## 相关工作与启发

- R2-Tuning 作为首个 VTG 的 ST 方法，其基于锚框的设计是本文的主要对比对象
- RDSA 的思路可推广至其他使用可变形交叉注意力的任务（如目标检测中的 Deformable DETR）
- 特征池化策略的选择对 ST 性能至关重要，这一发现对其他视频理解任务也有借鉴意义

## 评分

- 新颖性: ⭐⭐⭐⭐ 双流 ST 架构和 RDSA 是有意义的贡献，但整体框架建立在已有组件上
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集全面评估，丰富的消融实验，深入的可变形注意力分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机论述充分，但公式密度较高
- 价值: ⭐⭐⭐⭐ 为参数高效的视频时序定位提供了实用的解决方案，减少 73% 参数量达到 SOTA

<!-- RELATED:START -->

## 相关论文

- [Sim-DETR: Unlock DETR for Temporal Sentence Grounding](sim-detr_unlock_detr_for_temporal_sentence_grounding.md)
- [HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling](../../CVPR2026/object_detection/hieramamba_video_temporal_grounding_via_hierarchical_anchor-mamba_pooling.md)
- [TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding](../../AAAI2026/object_detection/tubermc_tube-conditioned_reconstruction_with_mutual_constraints_for_weakly-super.md)
- [ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning](chartpoint_guiding_mllms_with_grounding_reflection_for_chart_reasoning.md)
- [ViewSRD: 3D Visual Grounding via Structured Multi-View Decomposition](viewsrd_3d_visual_grounding_via_structured_multi-view_decomposition.md)

<!-- RELATED:END -->
