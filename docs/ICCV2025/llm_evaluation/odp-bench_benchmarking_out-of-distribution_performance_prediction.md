---
title: >-
  [论文解读] ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction
description: >-
  [ICCV 2025][OOD性能预测] 构建了首个全面的OOD性能预测基准ODP-Bench，涵盖29个OOD数据集、10种预测算法和1,444个预训练模型，揭示现有算法在合成corruption上表现较好但在自然分布偏移上普遍失效的关键发现。
tags:
  - ICCV 2025
  - OOD性能预测
  - 分布偏移
  - 基准评测
  - 鲁棒性评估
  - 模型选择
---

# ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction

**会议**: ICCV 2025  
**arXiv**: [2510.27263](https://arxiv.org/abs/2510.27263)  
**代码**: [https://github.com/h-yu16/Performance_Prediction/](https://github.com/h-yu16/Performance_Prediction/) (有)  
**领域**: 其他/OOD性能预测  
**关键词**: OOD性能预测, 分布偏移, 基准评测, 鲁棒性评估, 模型选择  

## 一句话总结
构建了首个全面的OOD性能预测基准ODP-Bench，涵盖29个OOD数据集、10种预测算法和1,444个预训练模型，揭示现有算法在合成corruption上表现较好但在自然分布偏移上普遍失效的关键发现。

## 研究背景与动机

**领域现状**：OOD性能预测的目标是预测已训练模型在无标签OOD测试集上的性能表现，从而在风险敏感场景（自动驾驶、医疗影像）中更安全地使用现有模型。近年来已有多种方法从模型置信度、分布差异、模型一致性等角度来预测OOD性能。

**现有痛点**：
   - **评测不一致**：不同文献的评测协议差异巨大，包括训练模型、测试数据集和评价指标均不统一
   - **覆盖不足**：大多数工作仅使用有限的OOD数据集，主要集中在合成corruption偏移上，很少涵盖domain generalization和subpopulation shift等重要OOD场景
   - **分布偏移类型单一**：很少涉及camera location、image background、demographic attributes等真实世界分布偏移

**核心矛盾**：在不公平的评测条件下，无法准确判断各算法的能力边界和适用范围。尽管OOD泛化算法（不变学习、domain generalization等）取得了进展，但实验表明没有算法能显著提升OOD性能，因此直接预测现有模型的OOD性能并据此做模型选择变得更为重要。

**本文目标**：建立统一、全面、公平的OOD性能预测基准，使不同算法可以在相同条件下比较，并深入分析现有算法的能力边界。

**切入角度**：从数据集覆盖面、分布偏移多样性、预训练模型数量三个维度同时扩展，提供开箱即用的1,444个模型作为testbench，避免重复训练。

**核心 idea**：通过统一的大规模基准暴露现有OOD性能预测算法在自然分布偏移上的根本局限性。

## 方法详解

### 整体框架
ODP-Bench由三部分组成：(1) 涵盖多种分布偏移类型的29个OOD数据集；(2) 1,444个不同架构、初始化和训练算法的预训练模型作为testbench；(3) 10种实用的性能预测算法。给定训练好的模型 $f_{\theta_0}$、有标签验证集 $\{x_i^{va}, y_i^{va}\}_{i=1}^{n_{va}}$ 和无标签OOD测试集 $\{x_i^{te}\}_{i=1}^{n_{te}}$，目标是预测模型在测试集上的性能或计算一个与真实性能正相关的代理分数。

### 关键设计

1. **数据集设计（29个OOD数据集）**：

    - **合成偏移**：CIFAR-10-C, CIFAR-100-C, ImageNet-C, TinyImageNet-C（corruption类型）
    - **风格偏移**：ImageNet-S, ImageNet-R, PACS（style类型）
    - **背景偏移**：NICO++, Waterbirds（background类型）
    - **数据采集偏移**：CIFAR-10.1/10.2, CINIC-10, STL-10, ImageNet-V2, VLCS
    - **相机位置偏移**：iWildCam, TerraInc, ObjectNet
    - **人口统计偏移**：CelebA, CivilComments, CheXpert
    - **其他**：FMoW(时间+地区), RxRx1(批次效应), Amazon, DomainNet, OfficeHome

2. **模型训练策略**：

    - ImageNet变体：直接使用Torchvision的109个开源模型
    - CIFAR变体：从头训练，每个架构3个随机种子（CIFAR-10共57个，CIFAR-100共108个）
    - WILDS：从ImageNet预训练权重初始化，30种架构各1个模型
    - Domain Generalization/Subpopulation Shift：使用supervised/MoCo/CLIP预训练权重，ResNet-50和ViT-B/16两种骨架，leave-one-domain/group-out，每个设置5个随机种子

3. **评价指标——Spearman秩相关**：
    $\rho = 1 - \frac{6\sum_{i=1}^{n}(R(\hat{S}_i) - R(Acc_i))^2}{n(n^2-1)}$
   其中 $\hat{S}_i$ 是预测分数，$Acc_i$ 是真实精度。选择此指标而非 $R^2$ 是因为后者对异常值敏感且无法处理非线性相关。本文跨架构计算指标（而非以往的同架构内计算），更具挑战性且更接近实际。

4. **10种预测算法**：涵盖基于置信度（ATC, DoC）、基于分布差异（COT, COTT）、基于特征（Nuclear Norm, MaNo, Dispersion, MDE）、基于数据增强（NI）、基于模型一致性（Agreement）等多个方向。

## 实验

### 主实验：29个OOD数据集 × 10种算法

| 数据集类型 | 代表数据集 | 平均 $\rho$ | 有效算法数($\rho>0.7$) |
|:---|:---|:---|:---|
| 合成Corruption | CIFAR-10-C | 0.746 | 9/10 |
| 合成Corruption | CIFAR-100-C | 0.712 | 9/10 |
| 自然偏移-风格 | ImageNet-S | 0.559 | 4/10 |
| 自然偏移-背景 | NICO++ | 0.705 | 9/10 |
| 自然偏移-类别细粒度 | DomainNet | 0.451 | 2/10 |
| 自然偏移-相机位置 | iWildCam | 0.328 | 0/10 |
| 自然偏移-对抗 | ImageNet-A | 0.270 | 1/10 |

**关键发现1**：合成corruption数据集上大多数算法的秩相关都>0.7（9/10有效），但在自然分布偏移上有效算法数急剧下降（iWildCam上0/10有效）。

**关键发现2**：Agreement和ATC是整体最稳定的算法。Agreement在多个数据集上达到>0.9的秩相关（CIFAR-10-C: 0.991, ImageNet-V2: 0.996），ATC在15/29个数据集上排名前三。

### 消融/分析实验

| 分析维度 | 发现 |
|:---|:---|
| 预训练权重 | CLIP预训练导致模型性能更集中、预测更难；MoCo与supervised差异不大 |
| 模型架构 | 同架构内 $\rho$ 通常很高（>0.95），但跨架构时性能下降明显 |
| $R^2$ vs $\rho$ | $R^2$ 指标对异常值更敏感，部分数据集上 $R^2<0$ 但 $\rho>0.5$ |
| 偏移类型 | Camera location和adversarial偏移最难预测，corruption最容易 |

### 关键发现总结
1. 现有算法在合成corruption上有效但在自然偏移上普遍失效——这是OOD性能预测领域的最核心挑战
2. 没有单一算法能在所有偏移类型上都表现良好
3. 跨架构的性能预测比同架构内更具挑战性
4. CLIP预训练权重使得性能预测更困难

## 亮点与洞察

1. **大规模统一基准**：首次将domain generalization、subpopulation shift等OOD子领域的数据集纳入性能预测基准，29个数据集覆盖7种分布偏移类型
2. **可复用testbench**：提供1,444个预训练模型，未来研究者无需重复训练即可公平比较
3. **暴露关键盲区**：清晰地展示了现有算法在"合成偏移易、自然偏移难"的能力边界，为未来研究指明方向
4. **跨架构评估的重要性**：指出以往同架构评估可能高估了算法的实用性

## 局限性

1. 目前仅关注分类任务，未涉及检测、分割等任务的OOD性能预测
2. 自然分布偏移上所有算法表现都不理想，但未提出改进方案
3. 未分析不同偏移类型之间的关联性或可复合偏移的情况
4. 1,444个模型仍以CNN为主（ResNet, VGG等），大模型（如ViT-L, Swin）的覆盖有限

## 相关工作

- **OOD泛化**：不变学习（IRM, IRMv1）、DRO、domain generalization（SWAD, CORAL）、stable learning等方向
- **OOD性能预测**：基于模型置信度（ATC, DoC）、分布差异（COT）、模型一致性（Agreement-on-the-line）、特征分离度（Dispersion Score）等
- **相关基准**：WILDS, DomainBed, MetaShift等提供了部分OOD数据集，但没有专门针对性能预测的综合基准

## 评分
- 创新性：★★★☆☆（benchmark论文，创新主要在于系统性整合）
- 实验充分度：★★★★★（29个数据集、10种算法、1,444个模型）
- 实用价值：★★★★★（提供开源代码和模型testbench，直接可用）
- 写作质量：★★★★☆（结构清晰，分析深入）

<!-- RELATED:START -->

## 相关论文

- [DisCoPatch: Taming Adversarially-driven Batch Statistics for Improved Out-of-Distribution Detection](discopatch_taming_adversarially-driven_batch_statistics_for_improved_out-of-dist.md)
- [Gradient-Regularized Out-of-Distribution Detection](../../ECCV2024/llm_evaluation/gradient-regularized_out-of-distribution_detection.md)
- [OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](../../CVPR2025/llm_evaluation/oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)
- [R²-Bench: Benchmarking the Robustness of Referring Perception Models under Perturbations](../../ECCV2024/llm_evaluation/r2-bench_benchmarking_the_robustness_of_referring_perception_models_under_pertur.md)
- [CalibraEval: Calibrating Prediction Distribution to Mitigate Selection Bias in LLMs-as-Judges](../../ACL2025/llm_evaluation/calibraeval_calibrating_prediction_distribution_to_mitigate_selection_bias_in_ll.md)

<!-- RELATED:END -->
