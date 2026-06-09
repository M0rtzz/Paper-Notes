---
title: >-
  [论文解读] Interpretable Image Classification via Non-parametric Part Prototype Learning
description: >-
  [CVPR 2025][可解释性][可解释分类] 本文提出一种基于非参数原型学习的可解释图像分类框架，通过对自监督ViT特征进行最优传输聚类来发现语义上不同的物体部件原型，解决了现有ProtoPNet方法中原型重复冗余的问题…
tags:
  - "CVPR 2025"
  - "可解释性"
  - "可解释分类"
  - "原型学习"
  - "非参数方法"
  - "自监督ViT"
  - "部件发现"
---

# Interpretable Image Classification via Non-parametric Part Prototype Learning

**会议**: CVPR 2025  
**arXiv**: [2503.10247](https://arxiv.org/abs/2503.10247)  
**代码**: [https://github.com/zijizhu/proto-non-param](https://github.com/zijizhu/proto-non-param)  
**领域**: 可解释性  
**关键词**: 可解释分类、原型学习、非参数方法、自监督ViT、部件发现

## 一句话总结

本文提出一种基于非参数原型学习的可解释图像分类框架，通过对自监督ViT特征进行最优传输聚类来发现语义上不同的物体部件原型，解决了现有ProtoPNet方法中原型重复冗余的问题，同时引入了Distinctiveness和Comprehensiveness两个新指标来量化解释质量。

## 研究背景与动机

Prototypical Part Networks（ProtoPNet）通过学习代表物体部件的原型来模拟人类视觉推理过程，实现可解释的图像分类。然而，现有方法将原型作为**可学习参数**训练，往往导致同一类别的多个原型关注**相同的物体区域**——例如4个最活跃的原型中有3个都关注鸟的翅膀，忽略了腿和尾巴。这导致解释既不全面也不多样。核心矛盾在于：参数化原型的学习过程难以强制不同原型关注不同语义部件。本文的切入角度：借鉴语义分割中非参数原型的思想，通过**聚类**方式发现原型。每个原型作为其对应聚类的经验均值，天然保证了不同原型代表不同的语义概念。结合自监督ViT（DINO）的强部件表示能力，可以无需部件级标注即可发现多样化的物体部件。

## 方法详解

### 整体框架

框架分为两个训练阶段：**阶段一**——在冻结的自监督ViT骨干上，通过最优传输聚类发现每个类别的非参数部件原型；**阶段二**——固定原型，通过Block Expansion微调骨干网络，使用交叉熵损失和patch-原型对比损失来适应目标域并提升分类精度。推理时，通过计算输入图像patch与原型的相似度来生成可解释的分类决策。

### 关键设计

1. **非参数原型发现（最优传输聚类）**:
    - 功能：为每个类别学习 $K$ 个语义上不同的部件原型
    - 核心思路：将一个batch中同一类别的所有前景patch token视为一个"特征包"，计算它们与当前原型的相似度矩阵 $\mathbf{S}^c \in \mathbb{R}^{N \times K}$，然后通过最优传输（OT）求解assignment矩阵 $\mathbf{A}^c$，使得patch-原型的总相似度最大化：$\max_{\mathbf{A}^c} \text{Tr}((\mathbf{A}^c)^\top \mathbf{S}^c)$。关键约束是**唯一分配**（每个patch只分配给一个原型）和**均分**（每个原型获得大致相等数量的patch），通过Sinkhorn-Knopp算法高效求解。原型通过动量更新为其聚类均值：$\mathbf{P}_k^c \leftarrow \mathbf{P}_k^c + (1-\beta)\bar{\mathbf{f}}^{c,k}$
    - 设计动机：均分约束强制每个原型必须代表不同的语义区域，从根本上避免了多个原型关注同一部位的问题。相比参数化方法的正则化"引导"，非参数方法的"约束"更直接有效

2. **Block Expansion微调策略**:
    - 功能：在不破坏DINO特征空间结构的前提下适应目标域
    - 核心思路：在ViT骨干的每 $m$ 个transformer block后插入一个**恒等初始化block**（attention投影层和第二MLP层权重设为零），使得初始输出不变。训练时冻结原始权重和原型，只训练新增block和调制向量 $w$
    - 设计动机：直接微调整个ViT会破坏DINO学到的良好语义特征空间，而Block Expansion提供了一种高效的适配方式

3. **Patch-Prototype对比损失（$\mathcal{L}_{\text{PPC}}$）**:
    - 功能：在微调阶段将特征锚定到已学习的原型上
    - 核心思路：类内对比损失——将每个patch"拉向"其分配的原型，"推离"同类的其他原型：$\mathcal{L}_{\text{PPC}} = \sum_c \frac{\exp(\mathbf{f}_i^{c\top} \mathbf{P}_{k_i}^c)}{\exp(\mathbf{f}_i^{c\top} \mathbf{P}_{k_i}^c) + \sum_j \exp(\mathbf{f}_i^{c\top} \mathbf{P}_{k_j}^c)}$
    - 设计动机：纯交叉熵微调虽提升分类精度，但会降低一致性和稳定性。PPC损失引导特征空间在适应目标域的同时保持与原型的对应关系

### 损失函数 / 训练策略

- 阶段一：非参数更新，无传统损失，使用 $\beta=0.99$ 的动量更新，$\kappa=0.05$ 的Sinkhorn平滑参数
- 阶段二：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ce}} + \lambda_{\text{PPC}} \mathcal{L}_{\text{PPC}}$，$\lambda_{\text{PPC}}=0.8$
- 阶段一训练1个epoch，阶段二训练5个epoch，Adam优化器
- 骨干学习率0.0001，调制向量学习率 $1 \times 10^{-6}$

## 实验关键数据

### 主实验（CUB-200-2011，DINOv2 ViT-B/14，K=5）

| 方法 | 一致性(%) | 稳定性(%) | **独特性(%)** | 分类精度(%) |
|------|----------|----------|-------------|------------|
| ProtoPNet | 1.30 | 63.84 | 11.99 | 88.16 |
| Deformable ProtoPNet | 2.83 | 80.43 | 10.34 | 88.14 |
| ProtoPool | 29.27 | 57.49 | 89.62 | 89.90 |
| TesNet | 58.60 | 74.60 | 81.93 | 90.35 |
| EvalProtoPNet | 67.80 | 74.40 | 72.12 | 90.06 |
| **本文** | **66.40** | **82.97** | **94.45** | **90.82** |

### 消融实验

| 配置 | 一致性 | 稳定性 | 独特性 | 分类精度 |
|------|--------|--------|--------|---------|
| 无约束聚类 | 4.09 | 75.50 | 49.47 | 9.25 |
| 无前景提取 | 28.40 | 82.33 | 92.69 | 64.29 |
| 无特征微调（仅阶段一） | 63.74 | 83.37 | 96.07 | 77.96 |
| 无PPC损失（仅CE） | 55.61 | 77.67 | 80.41 | 90.78 |
| **完整模型** | **66.40** | **82.97** | **96.74** | **91.02** |

### 关键发现

- **独特性得分大幅领先**：在K=5时，本文方法(94.45%)比最优基线TesNet(81.93%)高出12.52%，说明原型确实关注了不同的物体部件
- **阶段一已有不错分类能力**：仅用冻结DINO特征的非参数原型即可达到77.96%精度，验证了自监督ViT特征空间的语义质量
- **约束缺失导致崩溃**：去掉均分和唯一分配约束后分类精度降至9.25%（接近随机），说明OT约束是方法的核心
- **PPC损失对微调至关重要**：无PPC时独特性从96.74%降至80.41%，说明纯CE微调会破坏特征与原型的对应关系
- **概念学习适配**：将框架适配到Concept Bottleneck Model后，在CUB上取得47.04%的概念可信赖度和88.18%的分类精度

## 亮点与洞察

- **非参数 vs 参数化的根本优势**：参数化原型靠正则化"引导"分散，但没有硬约束保证；非参数方法通过均分约束**强制**每个原型对应不同的语义区域
- **DINO特征的开箱即用能力**：自监督ViT的patch token已经编码了丰富的部件语义，用于原型学习是自然的选择
- **两个新指标填补评估空白**：Distinctiveness Score量化原型间重叠，Comprehensiveness Score量化前景覆盖率，使可解释性可量化比较

## 局限与展望

- 依赖前景提取模块的质量——对背景复杂的场景可能不适用
- 均分约束假设各部件的patch数量大致相等，对不均匀部件分布可能不optimal
- 每个类别的原型数K需要预设，自适应确定K是一个开放问题
- 仅在细粒度分类数据集上验证，通用图像分类场景未测试

## 相关工作与启发

- 受ProtoSeg和Deep Nearest Centroids等非参数原型方法启发，将分割领域的思路迁移到可解释分类
- Block Expansion微调策略来自参数高效微调领域
- Sinkhorn-Knopp算法在SwAV自监督学习中也被用于均匀分配，这里用于部件原型分配

## 评分

- 新颖性: ⭐⭐⭐⭐ 非参数原型+OT聚类的思路在ProtoPNet框架中是新的，但各组件（DINO、Sinkhorn、Block Expansion）均有先例
- 实验充分度: ⭐⭐⭐⭐ 消融全面，引入了新的定量指标，但数据集限于细粒度分类（3个）
- 写作质量: ⭐⭐⭐⭐ 方法描述清楚，但阶段一和阶段二的衔接可以更紧凑
- 价值: ⭐⭐⭐⭐ 在可解释性上有显著突破（独特性+12%），对可解释AI社区有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Why Does It Look There? Structured Explanations for Image Classification](why_does_it_look_there_structured_explanations_for_image_classification.md)
- [\[CVPR 2026\] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](../../CVPR2026/interpretability/dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)
- [\[CVPR 2025\] On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_image-in-image_steganography.md)
- [\[ICML 2026\] Accurate Evaluation of Quickest Changepoint Detectors via Non-parametric Survival Analysis](../../ICML2026/interpretability/accurate_evaluation_of_quickest_changepoint_detectors_via_non-parametric_surviva.md)
- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)

</div>

<!-- RELATED:END -->
