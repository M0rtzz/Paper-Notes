---
title: >-
  [论文解读] Distilling Long-tailed Datasets
description: >-
  [CVPR 2025][模型压缩][数据集蒸馏] 首次系统研究长尾数据集蒸馏问题，发现现有方法在长尾场景下严重退化（甚至不如随机选择），提出Distribution-agnostic Matching（DAM）和Expert Decoupling（ED）两个策略，在CIFAR-10/100-LT和Tiny-ImageNet-LT上大幅超越现有方法（如在imbalance factor=100时超越DATM 19.7%）。
tags:
  - CVPR 2025
  - 模型压缩
  - 数据集蒸馏
  - 长尾分布
  - 轨迹匹配
  - 分布无关匹配
  - 专家解耦
---

# Distilling Long-tailed Datasets

**会议**: CVPR 2025  
**arXiv**: [2408.14506](https://arxiv.org/abs/2408.14506)  
**代码**: https://github.com/ichbill/LTDD  
**领域**: 模型压缩  
**关键词**: 数据集蒸馏、长尾分布、轨迹匹配、分布无关匹配、专家解耦

## 一句话总结
首次系统研究长尾数据集蒸馏问题，发现现有方法在长尾场景下严重退化（甚至不如随机选择），提出Distribution-agnostic Matching（DAM）和Expert Decoupling（ED）两个策略，在CIFAR-10/100-LT和Tiny-ImageNet-LT上大幅超越现有方法（如在imbalance factor=100时超越DATM 19.7%）。

## 研究背景与动机

**领域现状**：数据集蒸馏在均衡数据集（CIFAR-10/100、ImageNet）上取得了很好效果，但现实场景中的数据通常呈长尾分布（如医学图像），少数头部类有大量样本，而多数尾部类样本很少。

**现有痛点**：将现有DD方法直接应用于长尾数据集时性能急剧下降——DATM在imbalance factor=200时仅40.1%准确率，甚至不如随机选择的49.9%。问题根源有两个：（1）专家在长尾数据上训练导致偏向头部类的偏置梯度，蒸馏过程中这种偏置被传递到合成数据集，使尾部类图像包含的有用信息远少于头部类；（2）长尾专家对尾部类的预测不可靠（置信度仅0.38 vs 头部类0.97），导致软标签初始化质量差。

**核心矛盾**：蒸馏过程中student在均衡合成数据上训练，但需要匹配在长尾数据上训练的expert，两者权重分布根本不同，导致bi-level优化目标中产生冲突。

**本文目标** 如何从长尾分布的训练数据中蒸馏出均衡且信息丰富的合成数据集。

**切入角度**：从两方面入手——让student"模拟"长尾训练行为来弥合与expert的分布差距，以及用解耦训练的专家提供更可靠的监督信号。

**核心 idea**：用长尾分布感知的损失函数让student的权重分布自动模仿长尾expert，同时用解耦的representation expert和classifier expert分别提供backbone轨迹和分类器轨迹的监督。

## 方法详解

### 整体框架
基于轨迹匹配框架（如DATM），引入两个改进：（1）DAM修改student内循环的训练损失，使student在均衡合成数据上也能产生类似长尾训练的权重分布；（2）ED训练两种解耦专家——representation expert（在长尾数据上训练整个模型）和classifier expert（冻结backbone后在均衡数据上微调分类器），分别匹配backbone层和classifier层的轨迹。

### 关键设计

1. **Distribution-agnostic Matching (DAM)**:

    - 功能：消除student和expert之间因数据分布差异导致的权重分布不匹配
    - 核心思路：在student内循环训练时，使用改良的长尾分布损失$\mathcal{L}^c$代替标准交叉熵。该损失借鉴Balanced Softmax，对每个类按原始长尾数据集中的样本数加权：头部类给更大权重，尾部类给更小权重。这使得student在均衡合成数据上训练时，权重分布自然趋向长尾专家的分布，减小了两者的轨迹距离。关键公式中用$-\lambda \log(s_{y_i})$调整logits，$g(s_{y_i})$归一化类别权重
    - 设计动机：直接匹配偏置expert会把偏置传递给合成数据；DAM让student主动"模拟"偏置，使偏置停留在student权重中而不污染合成数据

2. **Expert Decoupling (ED)**:

    - 功能：提供更准确的蒸馏监督和软标签初始化
    - 核心思路：分两阶段训练专家——representation expert在原始长尾数据上训练整个模型学习特征表示；classifier expert冻结backbone后在过采样均衡数据上微调分类器。蒸馏时联合匹配：用representation expert的backbone层轨迹指导student backbone，用classifier expert的classifier层轨迹指导student classifier。软标签用classifier expert生成，因其对尾部类也有高置信度
    - 设计动机：representation learning阶段长尾数据反而有利于学习好的特征（更多样的头部类数据不会伤害表示质量），但分类器会严重偏置。解耦后各取所长

3. **联合轨迹匹配**:

    - 功能：同时优化backbone和classifier的匹配
    - 核心思路：总损失$\mathcal{L} = \lambda_{\text{rep}} \mathcal{L}_{\text{match}}(\text{backbone}) + \lambda_{\text{cls}} \mathcal{L}_{\text{match}}(\text{classifier})$，其中backbone匹配用representation expert，classifier匹配用classifier expert
    - 设计动机：单独匹配某一个expert都不够好，因为任何一个expert都有偏置或局限

### 损失函数 / 训练策略
DAM使用长尾分布加权的softmax损失（灵感来自Balanced Softmax）；轨迹匹配使用标准的归一化L2距离。λ用于控制logit调整的平滑程度，$\lambda_{\text{rep}}$和$\lambda_{\text{cls}}$控制两个匹配损失的权重比。

## 实验关键数据

### 主实验

| 数据集 | IF | IPC | Ours | DATM | MTT | Random |
|--------|-----|-----|------|------|-----|--------|
| CIFAR-10-LT | 10 | 50 | 70.5% | 66.7% | 62.0% | 51.9% |
| CIFAR-10-LT | 100 | 50 | 64.0% | 44.3% | 47.8% | 52.6% |
| CIFAR-10-LT | 200 | 50 | 62.3% | 40.1% | 23.9% | 49.9% |
| CIFAR-100-LT | 10 | 50 | 34.8% | - | - | 32.1% |
| Tiny-ImageNet-LT | 10 | 50 | - | - | - | 20.6% |

*IF=Imbalance Factor；IF越大越不均衡。IF=200时DATM和MTT严重崩溃（甚至不如随机），而Ours仍保持62.3%。

### 消融实验

| 配置 | CIFAR-10-LT IF=100 IPC=50 | 说明 |
|------|--------------------------|------|
| DAM + ED (Full) | 64.0% | 完整方法 |
| 仅DAM | 57.1% | 缺少ED掉6.9% |
| 仅ED | 54.8% | 缺少DAM掉9.2% |
| Baseline (DATM) | 44.3% | 无任何改进 |

### 关键发现
- 随着不均衡程度增加（IF从10→200），现有方法性能持续下降，但本文方法降幅远小于其他方法，展现了强鲁棒性
- DAM和ED都是必要的——DAM解决分布偏置问题（+12.8% vs baseline），ED解决弱监督问题（+10.5% vs baseline），组合后效果最优
- 可视化显示：使用本方法后分类器权重分布均衡（所有类的权重范数接近），而baseline的分类器权重严重偏向头部类
- 首次证明数据集蒸馏在长尾场景下可以实现接近无损压缩（某些IF和IPC设置下接近全数据集训练性能）

## 亮点与洞察
- **开创性问题定义**：长尾数据集蒸馏是一个被完全忽略但极其实际的问题（医学、自动驾驶数据天然长尾），本文首次系统化定义并解决
- **反直觉的DAM设计**：直觉上student应该学"正确"的均衡分布，但DAM反而让student去模拟长尾偏置，把偏置控制在student权重中而非合成数据中。这种"以毒攻毒"的思路非常巧妙
- **解耦专家的互补使用**：不是简单用一个更好的专家，而是让两种专家各负责最擅长的部分（表示 vs 分类），这种分工思路有广泛迁移价值

## 局限与展望
- 需要知道原始数据集的类别分布来构造DAM损失，如果分布未知需要额外估计
- Expert Decoupling需要训练两组专家轨迹，计算成本翻倍
- 仅在CIFAR和Tiny-ImageNet级别验证，缺少ImageNet-1K规模的长尾实验
- 当imbalance factor极端且同时IPC极低时，某些尾部类可能原始样本数少于IPC，此边界情况未充分讨论

## 相关工作与启发
- **vs DATM**: DATM在均衡数据上是SOTA，但在IF=200时崩溃到40.1%；本文通过DAM+ED在同一设置取得62.3%，提升22.2个百分点
- **vs 长尾学习方法**: Balanced Softmax等长尾学习技巧被创造性地整合到蒸馏框架中，不是简单地在训练时用，而是在student内循环中用来弥合分布差距
- **vs IDM**: IDM在长尾场景下相对稳定（分布匹配类方法比轨迹匹配更鲁棒），但本文仍超越IDM约10%

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次定义并解决长尾DD问题，DAM设计反直觉且有效
- 实验充分度: ⭐⭐⭐⭐ 四个数据集、多IF和IPC设置、详细消融，但缺大规模实验
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻，图示清晰（特别是Figure 3和5）
- 价值: ⭐⭐⭐⭐⭐ 填补了DD领域在长尾场景的空白，实际应用价值极高

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation](../../NeurIPS2025/model_compression/rectifying_soft-label_entangled_bias_in_long-tailed_dataset_distillation.md)
- [\[AAAI 2026\] Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling](../../AAAI2026/model_compression/rethinking_long-tailed_dataset_distillation_a_uni-level_framework_with_unbiased_.md)
- [\[CVPR 2026\] Distilling Balanced Knowledge from a Biased Teacher](../../CVPR2026/model_compression/distilling_balanced_knowledge_from_a_biased_teacher.md)
- [\[ACL 2025\] Towards the Law of Capacity Gap in Distilling Language Models](../../ACL2025/model_compression/law_of_capacity_gap_distilling_language_models.md)
- [\[ICML 2025\] Distilling Tool Knowledge into Language Models via Back-Translated Traces](../../ICML2025/model_compression/distilling_tool_knowledge_into_language_models_via_back-translated_traces.md)

<!-- RELATED:END -->
