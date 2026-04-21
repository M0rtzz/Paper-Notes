---
title: >-
  [论文解读] A Unified Framework for Heterogeneous Semi-supervised Learning
description: >-
  [CVPR 2025][异构半监督学习] 提出异构半监督学习(HSSL)新问题设定——标记数据和无标记数据来自不同分布的域，目标是训练能在两个域上都泛化的模型；通过将C类问题扩展为2C类分类（每个域的同一语义类视为不同类），结合WMA伪标签、跨域原型对齐和渐进式跨域Mixup三个组件统一解决。
tags:
  - CVPR 2025
  - 异构半监督学习
  - 跨域伪标签
  - 原型对齐
  - 渐进式Mixup
  - 2C类分类
---

# A Unified Framework for Heterogeneous Semi-supervised Learning

**会议**: CVPR 2025  
**arXiv**: [2503.00286](https://arxiv.org/abs/2503.00286)  
**代码**: 无  
**领域**: 半监督学习 / 域适应  
**关键词**: 异构半监督学习, 跨域伪标签, 原型对齐, 渐进式Mixup, 2C类分类

## 一句话总结

提出异构半监督学习(HSSL)新问题设定——标记数据和无标记数据来自不同分布的域，目标是训练能在两个域上都泛化的模型；通过将C类问题扩展为2C类分类（每个域的同一语义类视为不同类），结合WMA伪标签、跨域原型对齐和渐进式跨域Mixup三个组件统一解决。

## 研究背景与动机

**领域现状**：传统半监督学习(SSL)假设标记和无标记数据同分布，无监督域适应(UDA)只关注目标域性能。两者都不适用于标记和无标记数据来自不同域、且需在两个域上都泛化的场景。

**现有痛点**：医学影像、航拍图像等实际场景中，标记数据来自高端设备/研究医院，无标记数据来自多种农村诊所的不同设备——分布差异显著。标准SSL方法忽略分布差异会失效，UDA只优化目标域会牺牲源域性能。

**核心矛盾**：HSSL同时面临三大挑战：(1) 域间特征分布不同 $p_L(x|y) \neq p_U(x|y)$；(2) 域间标签分布不同 $p_L(y) \neq p_U(y)$；(3) 测试集来自两个域的混合。

**核心idea**：将C个语义类扩展为2C个细粒度类（前C个对应标记域，后C个对应无标记域），用统一的2C类分类器自然处理域差异，辅以三个组件促进跨域知识迁移。

## 方法详解

### 整体框架

预训练C类模型 → 为无标记数据生成初始伪标签 → 扩展为2C类模型 → 联合训练：监督损失(标记集) + WMA伪标签损失(无标记集) + 跨域原型对齐损失 + 渐进式Mixup损失。

### 关键设计

1. **加权移动平均(WMA)伪标签**：

    - 功能：自适应更新无标记数据的伪标签
    - 核心思路：$\hat{y}_i^t = \beta \hat{y}_i^{t-1} + (1-\beta) h(f(x_i^u))$，其中 $\beta=0.8$。仅使用置信度超过阈值 $\epsilon$ 的伪标签参与训练
    - 设计动机：由于跨域差距，初始伪标签噪声大。WMA提供平滑自适应的标签更新，避免振荡

2. **跨域语义原型对齐**：

    - 功能：对齐两个域中相同语义类的原型向量
    - 核心思路：计算标记域第k类原型 $p_k$ 和无标记域第k类原型 $p_{C+k}$，用对称对比损失拉近对应类对、推开非对应类对：$\mathcal{L}_{pa} = -\sum_{k=1}^{C} [\log \frac{\exp(\cos(p_k, p_{C+k})/\tau)}{\sum_{k'\neq k} \exp(\cos(p_k, p_{C+k'})/\tau)} + ...]$
    - 设计动机：利用两个域共享的语义类关系促进知识迁移

3. **渐进式跨域Mixup**：

    - 功能：生成跨域合成样本桥接两个域
    - 核心思路：$x^m = \lambda x^u + (1-\lambda) x^l$，关键在于渐进式调度——$\lambda \sim \psi(t) \times \text{Beta}(\alpha, \alpha)$，其中 $\psi(t) = 0.5 + t/(2T)$。训练初期 $\lambda \in [0, 0.5)$ 侧重标记域，后期逐渐扩展到 $[0, 1]$ 平等融合
    - 设计动机：训练初期伪标签不可靠，应偏向标记域；随训练进展伪标签质量改善，逐步增加无标记域权重

### 总损失

$\mathcal{L}_{total} = \mathcal{L}_{cl}^L + \lambda_{pl} \mathcal{L}_{pl}^U + \lambda_{pa} \mathcal{L}_{pa} + \lambda_{Mixup} \mathcal{L}_{Mixup}$

## 实验关键数据

### 主实验：Office-31 (ResNet-50)

| 方法 | W/A | A/W | D/A | D/W | W/D | 平均 |
|------|-----|-----|-----|-----|-----|------|
| Supervised | 68.6 | 82.8 | 35.5 | 96.9 | 98.2 | 77.8 |
| FixMatch (SSL) | 69.1 | 83.4 | 53.7 | 98.1 | 98.2 | 81.5 |
| SimMatch (SSL) | 71.1 | 84.1 | 68.6 | 96.8 | 98.8 | 84.3 |
| MCC+Sup (UDA) | 71.5 | 88.8 | 67.6 | 81.7 | 99.5 | 83.0 |
| BiAdopt | 70.2 | 85.0 | 67.1 | 94.2 | 98.5 | 82.0 |
| **Uni-HSSL** | **73.1** | **90.2** | **72.1** | **100** | **100** | **87.5** |

### 消融实验（Office-Home平均精度，附录）

消融验证了WMA伪标签、原型对齐和渐进Mixup三个组件的贡献。

### 关键发现

- Uni-HSSL在Office-31上平均87.5%，比最强SSL方法SimMatch高3.2%，比最强UDA方法MCC+Sup高4.5%
- 在难度最大的D→A任务（DSLR→Amazon，域差异最大）上提升最显著：72.1% vs SimMatch 68.6%
- 2C类分类策略有效：无需额外域分类器即可自然区分样本来源
- 在VisDA（大规模合成→真实）和ISIC-2019（医学图像）上同样领先
- UDA方法（CDAN+Sup）在某些任务上反而不如Supervised——因为UDA只优化目标域会损害源域性能

## 亮点与洞察

- **问题定义的价值**：HSSL桥接SSL和UDA两个社区，更贴近真实场景——数据来源多样化是常态
- **2C类扩展的优雅性**：简单但有效——把"同一概念在不同域"视为不同类，让模型自然学习域内和域间的特征差异，无需复杂的域分类器
- **渐进式Mixup的课程学习思想**：体现了"先易后难"的课程学习原则——先信赖标记数据，逐步信赖伪标签

## 局限与展望

- 假设两个域共享完全相同的C个类，无法处理类集不完全重叠的情况
- 2C类分类在类数大时可能导致分类器容量瓶颈
- 原型对齐依赖可靠伪标签，训练初期原型质量可能不佳
- 仅在图像分类上验证，未扩展到检测/分割等密集预测任务

## 相关工作与启发

- **vs BiAdopt**：BiAdopt也处理异构SSL但用独立组件分别处理不同域，而Uni-HSSL用统一2C类框架端到端训练
- **vs FixMatch/FlexMatch**：标准SSL方法假设同分布，在域差距大时效果有限（D→A仅53.7%），Uni-HSSL的跨域组件显著缓解
- **vs UDA**：UDA只优化目标域，在HSSL设定下源域性能会下降

## 评分

- 新颖性: ⭐⭐⭐⭐ 问题定义有价值，2C类扩展简洁有效，但各组件（伪标签、原型对齐、Mixup）均为已有技术的组合
- 实验充分度: ⭐⭐⭐⭐ 4个数据集、多种baseline对比，但缺乏详细消融和大规模数据验证
- 写作质量: ⭐⭐⭐⭐ 问题形式化清晰，方法描述完整，但符号较多
- 价值: ⭐⭐⭐⭐ 为异构数据场景的SSL提供了统一解决方案，有广泛应用前景

<!-- RELATED:START -->

## 相关论文

- [Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning](../../ICML2025/llm_pretraining/density_ratio_estimation-based_bayesian_optimization_with_semi-supervised_learni.md)
- [A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning](../../ICML2025/llm_pretraining/a_square_peg_in_a_square_hole_meta-expert_for_long-tailed_semi-supervised_learni.md)
- [An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling](../../ACL2025/llm_pretraining/dual_stage_curriculum_learning_sequence_labeling.md)
- [ConstStyle: Robust Domain Generalization with Unified Style Transformation](../../ICCV2025/llm_pretraining/conststyle_robust_domain_generalization_with_unified_style_transformation.md)
- [MoMa: A Simple Modular Deep Learning Framework for Material Property Prediction](../../ICLR2026/llm_pretraining/moma_a_modular_deep_learning_framework_for_material_property_prediction.md)

<!-- RELATED:END -->
