---
title: >-
  [论文解读] Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation
description: >-
  [CVPR 2025][模型压缩][数据集蒸馏] 提出CCFS方法，通过课程学习框架渐进式地从原始数据集中选择合适的真实样本补充蒸馏数据，解决高IPC场景下蒸馏数据与真实数据的不兼容问题，在CIFAR-10/100和Tiny-ImageNet上大幅超越SOTA（最高+6.6%）。
tags:
  - CVPR 2025
  - 模型压缩
  - 数据集蒸馏
  - 课程学习
  - 高IPC
  - 真实数据选择
  - 粗到细
---

# Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation

**会议**: CVPR 2025  
**arXiv**: [2503.18872](https://arxiv.org/abs/2503.18872)  
**代码**: https://github.com/CYDaaa30/CCFS  
**领域**: 模型压缩  
**关键词**: 数据集蒸馏、课程学习、高IPC、真实数据选择、粗到细

## 一句话总结
提出CCFS方法，通过课程学习框架渐进式地从原始数据集中选择合适的真实样本补充蒸馏数据，解决高IPC场景下蒸馏数据与真实数据的不兼容问题，在CIFAR-10/100和Tiny-ImageNet上大幅超越SOTA（最高+6.6%）。

## 研究背景与动机

**领域现状**：数据集蒸馏旨在将大规模训练数据压缩为少量合成图像，使模型训练后能达到接近全数据集的性能。现有方法在极低IPC（如1或5张/类）时表现优秀，但随着IPC增大，性能急剧下降，甚至不如随机选择。

**现有痛点**：最新的组合范式方法SelMatch提出将蒸馏数据$\mathcal{D}_{\text{distill}}$和真实数据$\mathcal{D}_{\text{real}}$合并构建合成数据集，虽然取得了SOTA，但其真实数据选择存在两个缺陷：（1）一次性固定选择，可能选到不合适的真实图像；（2）$\mathcal{D}_{\text{real}}$的选择独立于$\mathcal{D}_{\text{distill}}$，两者缺乏互补性。

**核心矛盾**：SelMatch在蒸馏前选择真实数据，但蒸馏过程会改变$\mathcal{D}_{\text{distill}}$的特征分布，导致预先选择的真实数据与蒸馏后的数据不兼容。作者通过三组对比实验验证了这一点——先蒸馏再选择比先选择再蒸馏平均高1.7%。

**本文目标** 如何在组合范式中渐进式地选出与当前蒸馏数据最互补的真实样本，解决不兼容问题。

**切入角度**：将真实数据选择建模为课程学习问题，从简到难逐步引入真实样本。利用filter模型在当前合成数据上训练后识别其"盲区"（被错分的样本），从中选择最简单的作为补充。

**核心 idea**：先蒸馏再用课程学习框架多轮选择真实数据，每轮通过"粗筛错分样本+细选最简单的"策略获取最优补充。

## 方法详解

### 整体框架
CCFS分为两个阶段：首先用现有DD方法（如CDA）蒸馏得到$\mathcal{D}_{\text{distill}}$，然后通过J轮课程学习逐步从原始数据集中选择真实样本加入合成数据集。每轮课程中，先在当前合成数据集上训练一个filter模型，然后用粗到细策略选择最优真实样本，最终合并得到$\mathcal{S} = \mathcal{D}_{\text{distill}} \cup \mathcal{D}_{\text{real}}^{1} \cup ... \cup \mathcal{D}_{\text{real}}^{J}$。

### 关键设计

1. **课程选择框架（Curriculum Selection Framework）**:

    - 功能：将真实数据选择过程结构化为多轮渐进式选择
    - 核心思路：初始合成集$\mathcal{S}_0 = \mathcal{D}_{\text{distill}}$，每轮选择$k_j = \lfloor \text{IPC} \times (1-\alpha) / J \rfloor$个真实样本/类加入，排除已选样本。关键在于先蒸馏再选择，确保选择基于最终蒸馏结果而非初始化数据
    - 设计动机：一次性选择无法适应蒸馏后数据的变化，多轮选择能随着合成数据集的丰富逐步提高选择难度，实现从简到难的渐进覆盖

2. **粗到细选择策略（Coarse-to-Fine Selection）**:

    - 功能：在每轮课程中精准找到当前合成数据集最需要的补充样本
    - 核心思路：分两步——粗筛阶段在当前$\mathcal{S}$上训练filter模型，用它评估原始训练集，过滤掉被正确分类的样本（这些特征已被合成集覆盖）；细选阶段对剩余错分样本按预计算的Forgetting分数升序排列，选最简单的$k$个/类。选最简单的错分样本是因为它们是"模型刚好学不到但又不太难的"特征，最具训练价值
    - 设计动机：正确分类的样本说明其特征已被合成集覆盖，不需要重复引入；错分样本中选最简单的，避免引入过于复杂的特征干扰训练

3. **难度分数（Difficulty Scores）**:

    - 功能：为细选阶段提供全局样本难度排序
    - 核心思路：使用预计算的Forgetting分数衡量样本难度。实验对比了Forgetting、C-score和Logits三种难度度量，Forgetting在所有数据集上表现最好（比Logits高2.8%在CIFAR-100上）
    - 设计动机：需要一个与当前filter无关的全局难度指标来稳定指导选择，Forgetting分数基于训练过程中样本被遗忘的次数，能较好反映样本的内在复杂度

### 损失函数 / 训练策略
蒸馏阶段使用CDA方法的MTT匹配损失。课程选择阶段不涉及额外损失函数，filter模型使用标准交叉熵训练。默认3轮课程，蒸馏比例$\alpha$在不同IPC下需搜索最优值。

## 实验关键数据

### 主实验

| 数据集 | IPC (压缩比) | CCFS | SelMatch | CDA | 提升(vs SelMatch) |
|--------|-------------|------|----------|-----|-------------------|
| CIFAR-10 | 500 (10%) | 92.5% | 85.9% | 84.4% | +6.6% |
| CIFAR-100 | 50 (10%) | 71.5% | 54.5% | 59.7% | +5.8%* |
| Tiny-ImageNet | 100 (20%) | 60.2% | 50.4% | 52.4% | +3.4%* |

*注：CIFAR-100超越当时最好的CUDD的65.7%达到71.5%（+5.8%）；Tiny-ImageNet接近全数据集训练（60.5%），仅差0.3%。

### 消融实验

| 配置 | CIFAR-100 IPC=50 | 说明 |
|------|-----------------|------|
| 错分+选最简单 | 71.5% | 完整CCFS策略 |
| 错分+随机选 | 70.1% | 不做细选掉1.4% |
| 错分+选最难 | 65.0% | 选太难的反而有害 |
| 正确+选最简单 | 66.8% | 粗筛方向错了 |
| 1轮课程 | 67.9% | 少了课程渐进 |
| 3轮课程 | 71.5% | 最佳平衡点 |
| Forgetting分数 | 71.5% | 最佳难度度量 |
| Logits分数 | 68.7% | 粗糙度量掉2.8% |

### 关键发现
- 粗到细策略中，"选错分样本中最简单的"组合显著优于其他5种组合，说明模型刚好学不到的简单特征最有补充价值
- 课程轮数从1增到3带来显著提升（67.9%→71.5%），但4-5轮增益接近饱和
- 跨架构泛化性强：用ResNet-18做filter选出的数据，在ResNet-50/101、DenseNet-121、RegNet上都有效
- 随着课程推进，选出的真实样本难度单调递增（可视化可见从简单背景→复杂姿态/遮挡）

## 亮点与洞察
- **先蒸馏后选择的逆向思维**：打破了现有方法"先选真实数据→再蒸馏"的流程，通过先蒸馏再根据结果选择真实数据，建立了两者之间的强关联。这个"先做完A再根据A选B"的思路可以迁移到很多需要两组数据互补的场景
- **以错分样本作为"需求检测器"**：用在合成集上训练的模型去识别它"学不到什么"，精准定位数据集的短板，这个思路简洁有效
- **课程框架的自适应正反馈**：合成集越丰富 → filter越强 → 错分样本越难 → 自然引入更难样本，形成良性循环，无需手动设计课程难度曲线

## 局限与展望
- 需要预计算全数据集的Forgetting分数，增加了前置计算成本
- 每轮课程需要从头训练filter模型，J轮课程意味着J次额外全模型训练
- 蒸馏比例$\alpha$需要针对每个IPC搜索最优值，增加超参数调节负担
- 仅在CIFAR-10/100和Tiny-ImageNet上验证，缺少ImageNet-1K等大规模数据集的实验
- 选择"最简单的错分样本"可能在某些数据分布下不是最优策略（如类别不平衡场景）

## 相关工作与启发
- **vs SelMatch**: SelMatch用一次性滑窗选择真实数据，选择独立于蒸馏结果；CCFS先蒸馏再多轮选择，完全消除了不兼容问题，提升巨大（+5.8%在CIFAR-100）
- **vs DATM**: DATM从优化角度入手，用不同训练阶段的轨迹生成多样性数据；CCFS从数据构成角度入手，两者思路正交，理论上可组合
- **vs CUDD**: CUDD也用课程扩展蒸馏集，但CCFS的粗到细选择策略更精细，在CIFAR-100上超越CUDD的65.7%达到71.5%

## 评分
- 新颖性: ⭐⭐⭐⭐ 课程+粗到细选择的组合有创新，但核心思想（先蒸馏再选真实数据）在分析实验中已经被发现
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、多个IPC设置、跨架构、消融全面、可视化清晰
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻，从分析实验自然过渡到方法设计，逻辑清晰
- 价值: ⭐⭐⭐⭐ 在高IPC数据集蒸馏这个实际重要场景取得接近全数据集性能，实用价值高

<!-- RELATED:START -->

## 相关论文

- [HierAmp: Coarse-to-Fine Autoregressive Amplification for Generative Dataset Distillation](../../CVPR2026/model_compression/hieramp_coarse-to-fine_autoregressive_amplification_for_generative_dataset_disti.md)
- [Enhancing Dataset Distillation via Non-Critical Region Refinement](enhancing_dataset_distillation_via_non-critical_region_refinement.md)
- [Dataset Distillation with Neural Characteristic Function: A Minmax Perspective](dataset_distillation_with_neural_characteristic_function_a_minmax_perspective.md)
- [Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios](emphasizing_discriminative_features_for_dataset_distillation_in_complex_scenario.md)
- [DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation](delt_a_simple_diversity-driven_earlylate_training_for_dataset_distillation.md)

<!-- RELATED:END -->
