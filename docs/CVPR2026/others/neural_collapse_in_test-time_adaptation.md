---
title: >-
  [论文解读] Neural Collapse in Test-Time Adaptation
description: >-
  [CVPR 2026][神经坍缩] 将神经坍缩 (Neural Collapse) 理论从类级别扩展到样本级别，发现了NC3+现象（样本特征嵌入与对应分类器权重对齐），基于此揭示了分布偏移下性能退化的根本原因是样本级特征-分类器错位，并提出NCTTA方法通过几何邻近度与预测置信度的混合目标引导特征重新对齐，在ImageNet-C上比Tent提升14.52%。
tags:
  - CVPR 2026
  - 神经坍缩
  - 测试时自适应
  - 分布外鲁棒性
  - 特征-分类器对齐
  - 混合目标
---

# Neural Collapse in Test-Time Adaptation

**会议**: CVPR 2026  
**arXiv**: [2512.10421](https://arxiv.org/abs/2512.10421)  
**代码**: https://github.com/Cevaaa/NCTTA  
**领域**: 其他 (分布外泛化/测试时自适应)  
**关键词**: 神经坍缩, 测试时自适应, 分布外鲁棒性, 特征-分类器对齐, 混合目标

## 一句话总结
将神经坍缩 (Neural Collapse) 理论从类级别扩展到样本级别，发现了NC3+现象（样本特征嵌入与对应分类器权重对齐），基于此揭示了分布偏移下性能退化的根本原因是样本级特征-分类器错位，并提出NCTTA方法通过几何邻近度与预测置信度的混合目标引导特征重新对齐，在ImageNet-C上比Tent提升14.52%。

## 研究背景与动机

1. **领域现状**：测试时自适应 (TTA) 已成为应对分布偏移的实用方案，主要方法包括：基于原型的方法（SHOT、T3A）、一致性正则化方法（MEMO、CoTTA）、归一化层方法（NOTE、SAR）和熵最小化方法（Tent、EATA、DeYO）。

2. **现有痛点**：虽然这些方法通过算法优化在推理时取得了不错效果，但普遍缺乏对分布偏移下模型退化根本原因的理论理解，更多是"知其然不知其所以然"。

3. **核心矛盾**：Neural Collapse (NC) 理论揭示了训练后DNN的优雅几何结构（类均值↔分类器权重对齐），但其分析依赖类标签和全训练集来计算类均值——这在TTA场景中不可行（只有无标签的小batch测试数据）。

4. **本文目标**
    - 将NC理论扩展到样本级别，使其适用于TTA场景
    - 从NC视角解释分布偏移下的性能退化原因
    - 提出理论驱动的TTA方法

5. **切入角度**：既然NC3说"类均值与分类器权重对齐"，那么在TPT后期，类内方差趋近零（NC1），意味着每个样本的特征也该与对应分类器权重对齐——这就是NC3+。

6. **核心 idea**：性能退化 = 样本特征偏离了正确的分类器权重，因此TTA的核心任务是重新对齐，但伪标签不可靠，需用几何邻近度+预测置信度的混合目标替代。

## 方法详解

### 整体框架
NCTTA在测试时通过混合目标引导的对比对齐机制更新模型参数。输入是无标签测试mini-batch，输出是自适应后的模型预测。核心处理流程：(1) 计算样本特征与所有分类器权重的FCA距离；(2) 用FCA距离+预测置信度构建混合目标；(3) 选出最可能正确的top-k类作为正样本，其余为负样本；(4) 用NC引导的对齐损失拉近正样本、推远负样本。

### 关键设计

1. **NC3+：样本级对齐坍缩**:

    - 功能：将NC理论从类级别扩展到样本级别，为TTA提供理论基础。
    - 核心思路：定义FCA距离 $d_{ij} = \|\frac{\mathbf{h}_i}{\|\mathbf{h}_i\|_2} - \frac{w_j}{\|w_j\|_2}\|_2$ 作为样本特征嵌入 $\mathbf{h}_i$ 与第 $j$ 类分类器权重 $w_j$ 之间的归一化欧几里得距离。理论证明在交叉熵损失下，Ground-truth FCA距离 $d_{iy_i}$ 单调递减并趋于零。在ImageNet-100上用多种backbone验证：训练过程中G-FCA距离持续下降。
    - 设计动机：NC3需要类均值（依赖完整标注数据），在TTA中不可用。NC3+只需要单个样本的特征和分类器权重，完美适配TTA场景。

2. **基于NC3+的性能退化解释**:

    - 功能：从FCA距离角度解释OOD数据为什么会被错分。
    - 核心思路：对OOD数据分析发现两种距离分布的变化：正确分类样本的G-FCA距离 $d_{iy_i}^{\text{correct}}$ 较小（特征仍对齐正确权重），但错分样本的G-FCA距离 $d_{iy_i}^{\text{wrong}}$ 显著增大（特征偏离正确权重），而其P-FCA距离 $d_{i\hat{y}_i}^{\text{wrong}}$ 变小（特征漂移到错误权重附近）。随着corruption severity增加，这个gap持续扩大。
    - 设计动机：建立了"特征-分类器错位"与"性能退化"之间的定量联系，指明TTA的核心任务是重新对齐。

3. **NCTTA：混合目标引导的对比对齐**:

    - 功能：在TTA阶段显式引导特征嵌入与正确分类器权重重新对齐。
    - 核心思路：由于伪标签不可靠（特征已偏移），不能直接用伪标签 $\hat{y}_i$ 指定对齐目标。NCTTA构建混合目标 $\widetilde{\mathbf{y}}_i = (1-\alpha)\hat{d}_i + \alpha p_i$，其中 $\hat{d}_i$ 是FCA距离的softmax归一化（几何邻近度），$p_i$ 是预测概率（置信度），$\alpha$ 平衡两者。根据 $\widetilde{\mathbf{y}}_i$ 排序选top-k类作为正样本集 $\mathcal{T}_i$，用NC引导的对齐损失 $\mathcal{L}_{\text{NC}}$ 拉近正样本、推远负样本。还引入动态权重 $\lambda_i$，综合熵指标和P-FCA距离控制每个样本的损失权重。
    - 设计动机：纯伪标签（$\alpha=1, k=1$）在严重偏移下错误率高；纯几何邻近度又可能被异常特征误导。混合方案比两者都更鲁棒，且top-k而非top-1进一步增加了容错空间。

### 损失函数 / 训练策略
最终损失为 $\mathcal{L}_{\text{total}}(x_i) = \lambda_i \cdot \mathbb{I}_{x_i \in S_{\text{ENT}}} \cdot (\mathcal{L}_{\text{ENT}}(x_i) + \mathcal{L}_{\text{NC}}(x_i))$，其中 $S_{\text{ENT}}$ 是熵过滤后的样本集（排除高熵预测），$\mathcal{L}_{\text{ENT}}$ 是标准的熵最小化损失，$\mathcal{L}_{\text{NC}}$ 可实例化为InfoNCE/L2/Triplet三种形式。

## 实验关键数据

### 主实验

| 方法 | CIFAR-10-C Avg (ResNet50) | ImageNet-C Avg (ViT-B/16) |
|------|--------------------------|--------------------------|
| no_adapt | 57.39 | 38.88 |
| Tent | 75.19 | 51.87 |
| EATA | 74.04 | 63.91 |
| SAR | 74.67 | 53.97 |
| NOTE | 71.03 | 39.15 |
| MEMO | 68.85 | 45.38 |
| DeYO | 76.65 | 63.49 |
| **NCTTA** | **78.16** | **66.46** |

NCTTA在ImageNet-C上比Tent提升14.59%，比DeYO提升2.97%。

### 消融实验

| $\mathcal{L}_{\text{NC}}$ 形式 | ImageNet-C Contrast (Sev-5) |
|------|------|
| InfoNCE-style | **最优** |
| L2-style | 略低 |
| Triplet-style | 最低 |

| $\alpha$ | $k=1$ | $k=3$ | $k=5$ | 说明 |
|------|-------|-------|-------|------|
| 0.0 (纯几何) | 较低 | 中等 | 中等 | 纯FCA距离不够 |
| 0.5 (混合) | 中等 | **最优** | 中等 | 平衡几何和置信度 |
| 1.0 (纯置信度) | 最低 | 低 | 低 | 纯伪标签不可靠 |

### 关键发现
- NCTTA在几乎所有corruption类型上都是最好或次好的，展示了很强的泛化性。
- InfoNCE-style损失最有效，可能因为其对比学习的梯度更有信息量。
- $\alpha=0.5, k=3$ 是最佳配置，说明几何和置信度的平衡以及适度的top-k范围最重要。
- 在Waterbirds数据集上最差组准确率从70.87%(no_adapt)/75.65%(DeYO)提升至76.56%，说明对子群偏移也有效。
- PACS跨域实验中也取得了最好的平均结果。

## 亮点与洞察
- **NC理论与TTA的桥接非常自然**：NC3+是NC3在满足NC1（类内方差→0）情况下的自然推论，但之前无人明确指出并加以利用。这个样本级视角完美适配了TTA只有无标签小batch的场景限制。
- **混合目标设计精巧**：用几何邻近度"校正"不可靠的伪标签是很好的思路。在严重偏移下伪标签错误率高，但几何上的近邻关系仍保持一定可靠性，两者互补。
- **理论→方法→实验的完整链条**：从NC3+理论发现→性能退化解释→方法设计→实验验证，逻辑链非常清晰完整，是理论驱动方法设计的好范例。

## 局限与展望
- NC3+的理论证明假设交叉熵损失和标准的TPT条件，对其他损失函数（如对比学习预训练的模型）的适用性未讨论。
- 目前NCTTA需要遍历所有K个类的分类器权重计算FCA距离，对类别数很大的任务（如ImageNet-21K）可能有计算开销。
- 连续域适应（continual TTA）场景下模型参数不断更新，分类器权重也在变化，NC3+的假设是否仍成立需要进一步分析。
- 未考虑标签空间偏移（open-set TTA）的情况。

## 相关工作与启发
- **vs Tent**：Tent仅做熵最小化，没有利用特征-分类器的几何结构。NCTTA在ImageNet-C上超越Tent 14.59%，说明几何引导的对齐比纯熵最小化更有效。
- **vs DeYO**：DeYO通过更精细的样本筛选提升性能，但仍缺乏对齐机制。NCTTA进一步提升2.97%。
- **vs EATA**：EATA也做熵过滤，但缺少NC引导的对齐。在CIFAR-10-C上NCTTA超越EATA 4.12%。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ NC3+是新发现，理论到方法的桥接非常优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集多backbone验证，消融详尽
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，可视化直观
- 价值: ⭐⭐⭐⭐ 为TTA领域提供了新的理论视角和实用方法

<!-- RELATED:START -->

## 相关论文

- [SPACE: SPike-Aware Consistency Enhancement for Test-Time Adaptation in Spiking Neural Networks](../../NeurIPS2025/others/space_spike-aware_consistency_enhancement_for_test-time_adaptation_in_spiking_ne.md)
- [MemBN: Robust Test-Time Adaptation via Batch Norm with Statistics Memory](../../ECCV2024/others/membn_robust_test-time_adaptation_via_batch_norm_with_statistics_memory.md)
- [Effortless Active Labeling for Long-Term Test-Time Adaptation](../../CVPR2025/others/effortless_active_labeling_for_long-term_test-time_adaptation.md)
- [Beyond Entropy: Region Confidence Proxy for Wild Test-Time Adaptation](../../ICML2025/others/beyond_entropy_region_confidence_proxy_for_wild_test-time_adaptation.md)
- [Optimizer Choice Matters for the Emergence of Neural Collapse](../../ICLR2026/others/optimizer_choice_matters_for_the_emergence_of_neural_collapse.md)

<!-- RELATED:END -->
