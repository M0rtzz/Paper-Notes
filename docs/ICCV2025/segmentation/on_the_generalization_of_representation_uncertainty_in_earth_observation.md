---
title: >-
  [论文解读] On the Generalization of Representation Uncertainty in Earth Observation
description: >-
  [ICCV 2025][语义分割][地球观测] 系统研究了预训练表示不确定性在地球观测（EO）领域的泛化能力，发现 EO 预训练的不确定性在不同地理位置、EO 任务和目标粒度上具备强泛化能力，同时对地面采样距离（GSD）高度敏感。 地球观测（EO）应用的关键性要求深度学习模型具有可信赖性，不确定性估计是提供预测置信度的重要手…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "地球观测"
  - "表示不确定性"
  - "零样本迁移"
  - "不确定性泛化"
---

# On the Generalization of Representation Uncertainty in Earth Observation

**会议**: ICCV 2025  
**arXiv**: [2503.07082](https://arxiv.org/abs/2503.07082)  
**代码**: [GitHub](https://github.com/Orion-AI-Lab/EOUncertaintyGeneralization)  
**领域**: 图像分割  
**关键词**: 地球观测, 表示不确定性, 零样本迁移, 语义分割, 不确定性泛化

## 一句话总结

系统研究了预训练表示不确定性在地球观测（EO）领域的泛化能力，发现 EO 预训练的不确定性在不同地理位置、EO 任务和目标粒度上具备强泛化能力，同时对地面采样距离（GSD）高度敏感。

## 研究背景与动机

地球观测（EO）应用的关键性要求深度学习模型具有可信赖性，不确定性估计是提供预测置信度的重要手段。然而，标准的不确定性感知方法（如贝叶斯神经网络、集成学习等）引入了高额的建模复杂度和计算开销。

近年来，计算机视觉领域提出了**预训练表示不确定性**的概念——在大规模预训练模型的表示空间上学习不确定性，实现零样本不确定性迁移。这对 EO 领域极具潜力，但存在以下挑战：

**EO 数据的独特性**：EO 图像具有多尺度、多分辨率、多光谱特性，缺乏明确的"背景"概念，所有场景元素都有语义意义

**域间差异**：自然图像预训练模型能否直接迁移到 EO？EO 内部不同传感器/任务之间能否迁移？

**缺乏评估框架**：现有评估仅限于单标签分类（Recall@1），EO 任务多为多标签分类或语义分割，需要新的评估指标

本文的研究动机是填补 EO 领域中预训练表示不确定性研究的空白。通过定义影响 EO 图像的四个**语义因子（Semantic Factors）**——GSD、兴趣域、目标粒度、空间排列，系统评估不确定性的泛化特性。

## 方法详解

### 整体框架

采用 Kirchhof et al. 提出的框架：在预训练冻结的表示空间上训练不确定性模块（MLP），以预训练任务损失作为偶然不确定性（aleatoric uncertainty）的代理。为每个输入 $x$ 估计标量不确定性值 $u(x)$。核心思路是：不确定性模块完全在表示空间上工作，不修改骨干网络，实现轻量级且可迁移的不确定性估计。

### 关键设计

1. **四大语义因子（Semantic Factors）**：定义了影响 EO 图像不确定性泛化的四个关键维度。SF1: 地面采样距离（GSD），决定空间分辨率和物体可检测性。SF2: 兴趣域，包括地理、时间、主题/环境因素差异。SF3: 目标粒度，从细粒度（<1m，如树种分类）到粗粒度（>1km，如森林检测）。SF4: 空间排列，物体的空间分布揭示场景形成过程。这四个因子构成了系统评估不确定性泛化的分析框架。

2. **多标签和分割的 Label Agreement@1 指标族**：现有 Recall@1 仅适用于单标签任务。本文提出 LA@1 指标族扩展到 EO 场景：多标签分类中包括 One-LA@1（共享至少一个类别）、All-LA@1（包含所有类别）、%-LA@1（类别匹配比例）。语义分割中包括 All-LA@1（上下文指标）、Patches-p-LA@1（空间补丁级相似度）、PD-LA@1（类别分布距离）、Patches-p-PD-LA@1（结合空间与上下文的分布距离）。并使用 CPA（Coefficient of Predictive Ability）替代 AUROC 来评估连续值指标。

3. **空间不确定性估计**：利用 ViT 的序列到序列特性，将嵌入 token 序列的每个 token 独立送入不确定性模块，计算每个 patch 的不确定性值，实现逐区域的空间不确定性图（每个 token 对应 $p \times p$ 像素区域）。

### 损失函数 / 训练策略

不确定性模块使用基于排名的损失来预测上游任务损失，避免下游任务中的损失缩放问题。具体来说，在大规模 EO 数据集（BigEarthNet 和 Flair）上训练 ViT-Tiny/Small/Base/Large 的四种变体，冻结其表示后训练上方的不确定性模块。训练使用 RGB 波段以确保与 ImageNet 预训练模型的公平对比。

## 实验关键数据

### 主实验

**EO 预训练 vs ImageNet 预训练的不确定性泛化（ViT-Large，多数据集）**

| 预训练数据 | MLRSNet %-LA-CPA | Woody Patches-LA-CPA | MARIDA %-LA-CPA |
|-----------|------------------|----------------------|-----------------|
| ImageNet | <0.50（低于随机基线） | <0.50 | <0.50 |
| BigEarthNet | >0.55 | >0.55 | >0.55 |
| **Flair** | >0.55 | **最优** | >0.55 |

EO 预训练在几乎所有指标和模型上一致优于 ImageNet 预训练和随机基线。ImageNet 预训练在 EO 数据集上甚至低于随机猜测基线，说明域差异严重影响不确定性泛化。

**丢弃测试（Discard Test）** - 零样本不确定性与下游损失的对齐

| 数据集 | BigEarthNet 预训练 MF | Flair 预训练 MF | ImageNet 预训练 MF |
|--------|---------------------|----------------|-------------------|
| MLRSNet | 高 ↓ | 高 ↓ | 低/不降 |
| Woody | 高 ↓ | 高 ↓ | 低/不降 |
| MARIDA | 高 ↓ | 高 ↓ | 低/不降 |

（MF = Monotonicity Fraction，衡量移除高不确定性样本后损失下降的频率）

### 消融实验

**GSD 对不确定性泛化的影响（ViT-Large，BigEarthNet 不同分辨率预训练）**

| 预训练分辨率 | 高分辨率推理指标 | 低分辨率推理指标 | 说明 |
|------------|---------------|---------------|------|
| 120×120（原始） | **最优** | 良好 | 原始分辨率预训练对高分辨率推理最佳 |
| 60×60 | 次优 | 足够好 | 降低预训练分辨率仍可工作 |
| 30×30 | 下降 | 足够好 | 适用于低分辨率推理 |
| 16×16 | 严重下降 | 下降 | 极端降采样全面退化 |

**目标粒度消融（BigEarthNet-5 vs BigEarthNet）**

| 模型 | BigEarthNet-5 (5类) | BigEarthNet (19类) |
|------|-------------------|-------------------|
| ViT-Tiny/Small | 一致优于 19 类版 | 基线 |
| ViT-Base/Large | 可比 | 基线 |

粗标签强调基础语义模式而非细微差异，足以维持泛化能力。

### 关键发现

- **域差异是不确定性泛化的最大障碍**：ImageNet 预训练的不确定性模块在 EO 数据上完全失效，但其编码器产生的表示与 EO 编码器质量相当。不确定性退化完全来自不确定性模块，而非编码器
- **EO 预训练不确定性对地理漂移鲁棒**：在欧洲数据上预训练的模型可泛化到智利和新西兰
- **GSD 对齐关键**：上下游分辨率匹配时不确定性估计最优
- **空间不确定性有前景**：利用 ViT token 级不确定性可生成空间不确定性图，雪覆盖/云覆盖区域获得更高不确定性

## 亮点与洞察

- **开创性**：首次系统研究 EO 领域的预训练表示不确定性泛化问题
- **评估框架完整**：提出了适用于多标签分类和语义分割的表示空间不确定性评估指标族（LA@1 + CPA）
- **实用意义大**：证明轻量级不确定性模块（MLP）即可提供可靠的零样本不确定性估计
- **空间不确定性**：巧妙利用 ViT 的 patch token 结构实现逐区域不确定性，无需额外训练

## 局限与展望

- 不确定性仅在图像级任务表示上学习，未在密集预测任务上直接训练
- 空间不确定性估计分辨率受限于 ViT patch 大小（16×16）
- 未探索多光谱/SAR 数据的系统评估（仅在附录中简要涉及）
- 当图像不确定性聚合为单一值时可能不可靠（如 Flair 中平均损失 vs 最大像素损失的差异）
- 未来方向：构建通用 EO 不确定性基础模型

## 相关工作与启发

- Kirchhof et al. 提出了预训练视觉不确定性框架并在自然图像上验证，本文将其推广到 EO 领域
- Scale-MAE、CROMA、DOFA 等 EO 基础模型为大规模表示学习奠定了基础
- 不确定性估计在 EO 领域存在被严重低估的空间，本文的评估框架可推广到更多 vision 领域

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次将表示不确定性泛化问题引入 EO 领域，定义了语义因子分析框架
- **实验充分度**: ⭐⭐⭐⭐⭐ 8 个数据集、4 种 ViT 变体、多维度指标、丢弃测试、噪声测试全面覆盖
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，分析深入，但指标体系较复杂
- **价值**: ⭐⭐⭐⭐ 为 EO 社区提供了不确定性评估的标准框架和关键洞察

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation](../../CVPR2026/segmentation/brewing_stronger_features_dual-teacher_distillation_for_multispectral_earth_obse.md)
- [\[ICCV 2025\] Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation](exploring_probabilistic_modeling_beyond_domain_generalization_for_semantic_segme.md)
- [\[ICCV 2025\] Region-based Cluster Discrimination for Visual Representation Learning](region-based_cluster_discrimination_for_visual_representation_learning.md)
- [\[NeurIPS 2025\] Torch-Uncertainty: A Deep Learning Framework for Uncertainty Quantification](../../NeurIPS2025/segmentation/torch-uncertainty_a_deep_learning_framework_for_uncertainty_quantification.md)
- [\[ICCV 2025\] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation](exploiting_domain_properties_in_language-driven_domain_generalization_for_semant.md)

</div>

<!-- RELATED:END -->
