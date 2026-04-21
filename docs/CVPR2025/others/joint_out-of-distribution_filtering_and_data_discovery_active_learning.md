---
title: >-
  [论文解读] Joint Out-of-Distribution Filtering and Data Discovery Active Learning
description: >-
  [CVPR 2025][active learning] 提出 Open-Set Discovery Active Learning (OSDAL) 场景，并设计 Joda 算法，通过训练-过滤-选择三阶段流程，用单一模型同时过滤 OOD 数据和发现新类别，无需额外辅助模型，在 18 个配置上持续达到最高准确率。
tags:
  - CVPR 2025
  - active learning
  - OOD detection
  - category discovery
  - open-set
  - energy score
  - SISOMe
---

# Joint Out-of-Distribution Filtering and Data Discovery Active Learning

**会议**: CVPR 2025  
**arXiv**: [2503.02491](https://arxiv.org/abs/2503.02491)  
**代码**: [项目页](https://www.cs.cit.tum.de/daml/joda/)  
**领域**: others  
**关键词**: active learning, OOD detection, category discovery, open-set, energy score, SISOMe

## 一句话总结

提出 Open-Set Discovery Active Learning (OSDAL) 场景，并设计 Joda 算法，通过训练-过滤-选择三阶段流程，用单一模型同时过滤 OOD 数据和发现新类别，无需额外辅助模型，在 18 个配置上持续达到最高准确率。

## 研究背景与动机

**领域现状**: 主动学习（AL）通过策略性选择样本标注来降低成本。现实场景中，未标注池存在两个挑战：(1) OOD 数据污染（open-set AL 已有研究），(2) 未知的新类别需要发现（category discovery 有独立研究）。

**现有痛点**:
- 现有 open-set AL 方法（LfOSA、CCAL、MQNet、Pal）需要 1-2 个额外辅助模型，且多数需要访问未标注池训练
- 这些方法仅处理 OOD 过滤，不考虑新类别发现
- AGD 考虑了类别发现但不处理 OOD 数据
- 两个问题在现实中（如自动驾驶）同时存在，但此前无统一方案

**核心矛盾**: 新类别数据是 near-OOD（与 InD 语义相近），OOD 数据是 far-OOD。现有方法粒度不够，无法在三类数据（已知 InD、可发现新类、不相关 OOD）间做细粒度区分。

**本文目标**: 在主动学习中同时实现 OOD 数据过滤和新类别发现，且不增加额外模型复杂度。

## 方法详解

### 整体框架

Joda 包含三个阶段：
1. **训练阶段 (I)**: 用 CrossEntropy + Outlier Exposure 联合训练单一分类模型
2. **过滤阶段 (II)**: 基于 energy score 和 ROC 分析确定阈值，过滤 OOD 样本
3. **选择阶段 (III)**: 用 SISOMe 指标 + 类别平衡因子选择最有价值样本

### 关键设计

**1. 深度耦合的训练损失**
- **功能**: 将标注池中的 InD 样本和意外标注到的 OOD 样本分别处理：$\mathcal{L}(b) = \mathcal{L}_{CE}(b_{InD}) + \lambda_{OE} \cdot \mathcal{L}_{OE}(b_{OOD})$。
- **核心思路**: Outlier Exposure 损失正则化模型对 OOD 样本预测均匀分布，这使得 energy score 在 InD/OOD 间产生清晰分界。初始标注池无 OOD 数据时退化为标准 CE。
- **设计动机**: 不同于外部训练单独的 OOD 检测器，OE 直接融入任务模型训练，构建了区分三类数据的共享特征空间。

**2. Energy-based OOD 过滤**
- **功能**: 计算 $E(x) = -\log \sum_{i=1}^{c} \exp(f(x)_i)$，在标注池上做 ROC 分析，找 Youden's J 最优阈值 $t_{opt}$ 进行过滤。
- **核心思路**: OE 训练后 OOD 样本的 logits 趋于均匀（energy 低），而 InD 和新类别样本的 energy 较高。新类别作为 near-OOD 不受 OE 正则化影响，energy 分布介于 InD 和 far-OOD 之间。
- **设计动机**: 利用 OE 训练的副产品自然实现 OOD 过滤，无需额外模型。Youden's J 自适应选阈值。

**3. SISOMe 选择 + 类别平衡**
- **功能**: 用 SISOMe 指标综合 energy score 和特征空间内/外距离比，再加类别平衡因子 $b_f(c) = -\sigma_{\hat{m}_l} (\frac{n(c) \cdot C}{|L|} - 1)$。
- **核心思路**: SISOMe 的自平衡机制在能量和多样性之间权衡；新类别因未被 OE 正则化而天然具有较高的选择分数（类似 near-OOD）；类别平衡因子利用伪标签估计类分布，偏向欠采样类。
- **设计动机**: 闭环设计——训练中的 OE 损失、过滤中的 energy、选择中的 SISOMe 都围绕同一个 energy/logit 空间，三个阶段深度耦合。

### 损失函数 / 训练策略

- 损失: $\mathcal{L} = \mathcal{L}_{CE}(b_{InD}) + 0.5 \cdot \mathcal{L}_{OE}(b_{OOD})$，$\lambda_{OE}=0.5$
- 模型: ResNet-18，无额外辅助模型
- 第一个 AL cycle 中标注池纯净，跳过过滤步骤
- 发现的新类别样本数达到阈值 $t_e$ 后并入已知类训练

## 实验关键数据

### 主实验

在 CIFAR-10、CIFAR-100、TinyImageNet 上测试，OOD 数据源包括 Random、MNIST、Places365、ImageNetC-800。

| 方法 | 额外模型 | CIFAR-100 Acc↑ | TinyImageNet Acc↑ | 选择精度↑ |
|---|---|---|---|---|
| LfOSA | 1 | ~42% | ~32% | ~0.95 |
| CCAL | 2 | ~40% | ~30% | ~0.85 |
| MQNet | 2(+1) | ~38% | ~28% | ~0.80 |
| Pal | 2 | ~41% | ~31% | ~0.90 |
| Badge | 0 | ~39% | OOM | ~0.75 |
| **Joda** | **0** | **~46%** | **~35%** | **~1.0** |

### 消融实验

| 设置 | 准确率 | 类发现 | 选择精度 |
|---|---|---|---|
| Full Joda | 最高 | 最快 | ~1.0 |
| w/o OE（无 Outlier Exposure） | 显著下降 | 下降 | 显著下降 |
| w/o 过滤（无 energy 过滤） | 显著下降 | 下降 | 显著下降 |
| Energy Exposure 替代 OE | 下降 | 下降 | 下降 |
| 不同 $\lambda$ (0.1-1.0) | 鲁棒 | 鲁棒 | 鲁棒 |

### 关键发现

1. **三阶段深度耦合是关键**: 去掉任何组件（OE、过滤、SISOMe）都导致性能显著下降。
2. **选择精度接近完美**: Joda 在 8/10 个配置中达到接近 1.0 的选择精度（几乎不选 OOD）。
3. **类发现最快**: 在所有配置中 Joda 发现新类别的速度最快。
4. **零额外模型**: 相比竞争方法需要 1-2 个额外模型，Joda 仅用单一分类模型。
5. **超参数鲁棒**: $\lambda_{OE}$ 在 0.1-1.0 范围内性能变化很小。

## 亮点与洞察

- OSDAL 场景定义有价值：首次形式化了 AL + OOD 过滤 + 类别发现的联合问题
- "做减法"的设计哲学：相比现有方法用 2+ 个额外模型，Joda 用 0 个额外模型反而效果最好
- 三阶段深度耦合的闭环设计：训练/过滤/选择共享同一特征空间和 energy 指标
- OE 损失的巧妙利用：不仅做 OOD 检测，还天然区分了 near-OOD（新类别）和 far-OOD

## 局限与展望

- 仅在图像分类任务上验证，未扩展到目标检测、语义分割等更复杂任务
- 新类别的阈值 $t_e$ 需要预先设定
- 当 OOD 数据与 InD 数据有语义重叠时（如 Places365 与 TinyImageNet），过滤精度略降
- 未与更新的基于 foundation model 的 OOD 检测方法对比
- 可探索将 Joda 扩展到 stream-based AL 场景

## 相关工作与启发

- **SISOMe (Schmidt et al.)**: Joda 的选择指标核心，结合能量和特征空间距离
- **Outlier Exposure (Hendrycks & Gimpel)**: OOD 检测经典方法，Joda 巧妙将其融入 AL 训练循环
- **LfOSA / CCAL / MQNet**: Open-set AL baselines，都依赖额外模型
- **AGD (Ma et al.)**: 仅关注新类别发现的 AL，不处理 OOD 过滤

## 评分

⭐⭐⭐⭐ — 问题定义有实际价值，方法简洁优雅（0 额外模型、0 额外数据），实验覆盖全面（18 配置 3 指标）。在实际部署场景（自动驾驶、机器人感知）中有直接应用价值。

<!-- RELATED:START -->

## 相关论文

- [H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection](h2st_hierarchical_two-sample_tests_for_continual_out-of-distribution_detection.md)
- [Open Set Label Shift with Test Time Out-of-Distribution Reference](open_set_label_shift_with_test_time_out-of-distribution_reference.md)
- [Instance-wise Supervision-level Optimization in Active Learning](instance-wise_supervision-level_optimization_in_active_learning.md)
- [Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](../../NeurIPS2025/others/redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [Harnessing Feature Resonance under Arbitrary Target Alignment for Out-of-Distribution Node Detection](../../NeurIPS2025/others/harnessing_feature_resonance_under_arbitrary_target_alignment_for_out-of-distrib.md)

<!-- RELATED:END -->
