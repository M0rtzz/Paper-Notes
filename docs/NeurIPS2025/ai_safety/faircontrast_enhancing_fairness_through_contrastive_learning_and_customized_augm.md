---
title: >-
  [论文解读] FairContrast: Enhancing Fairness through Contrastive Learning and Customized Augmentation
description: >-
  [NeurIPS 2025][AI安全][公平性] FairContrast 提出一种面向表格数据的公平对比学习框架，通过策略性的正对样本选择（将优势组有利结果样本与对应弱势组样本配对），结合有监督或自监督对比损失与交叉熵损失的端到端训练，在不引入额外公平约束损失的前提下显著降低了预测偏差，且精度损失极小。
tags:
  - NeurIPS 2025
  - AI安全
  - 公平性
  - 对比学习
  - 表格数据
  - 表示学习
  - 人口统计均等
---

# FairContrast: Enhancing Fairness through Contrastive Learning and Customized Augmentation

**会议**: NeurIPS 2025  
**arXiv**: [2510.02017](https://arxiv.org/abs/2510.02017)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 公平性, 对比学习, 表格数据, 表示学习, 人口统计均等

## 一句话总结

FairContrast 提出一种面向表格数据的公平对比学习框架，通过策略性的正对样本选择（将优势组有利结果样本与对应弱势组样本配对），结合有监督或自监督对比损失与交叉熵损失的端到端训练，在不引入额外公平约束损失的前提下显著降低了预测偏差，且精度损失极小。

## 研究背景与动机

**领域现状**：深度学习模型在实际应用中容易继承数据中的社会偏见，导致对特定敏感属性群体的不公平对待。经典案例包括再犯预测模型对特定种族的偏见、招聘模型对女性候选人的歧视等。表示学习特别是对比学习已在 CV/NLP 领域展现出优越的鲁棒性和泛化能力。

**现有方法问题**：
   - 表格数据的对比学习 SOTA（VIME、SCARF）未考虑公平性，其学习到的表示中仍包含敏感属性偏见
   - 视觉领域的公平对比学习方法（如 CCL）将高斯噪声作为增强策略用于表格数据，对离散和分类特征语义不合理
   - 已有公平表示学习方法（FCRL、CVIB、对抗遗忘）通常需要额外的公平约束模块或对抗训练，增加了模型复杂度

**核心动机**：能否仅通过精心设计对比学习的配对策略，将公平性自然嵌入标准对比学习过程，而无需额外的公平约束损失？

## 方法详解

### 整体框架

FairContrast 的核心思路：
1. 输入经过编码器 z = Enc(x) 映射到表示空间
2. 通过定制的正对采样策略选择正对样本
3. 使用对比损失（有监督或自监督）加二元交叉熵损失进行端到端训练
4. 学习到的公平表示可用于任意下游任务

### 关键设计

**1. 正对采样策略（核心创新）**

对于每个锚点样本，正对选择规则如下：

- **非优势组或不利结果**：与同类、同敏感属性的样本配对。例如低收入女性与另一低收入女性配对。这保持了子组内的特征相似性。
- **优势组且有利结果**：与弱势组中同样有利结果的样本配对。例如高收入男性与高收入女性配对。

核心直觉：将优势组的有利结果样本拉向弱势组的有利结果样本，使模型不再依赖敏感属性做出有利预测，与机会均等公平准则对齐。

**负对策略**：
- 有监督：不同类别的样本互为负对
- 自监督：batch 内除正对外所有其他样本均为负对

**2. 自监督对比损失（InfoNCE）**

L_self = -(1/N) * sum_i log[exp(sim(z_i, z_j)/tau) / sum_{k!=i} exp(sim(z_i, z_k)/tau)]

其中 sim 为余弦相似度，tau 为温度参数。

**3. 有监督对比损失**

L_sup 在 InfoNCE 基础上扩展为多正对：对于锚点 i，所有同类样本 P(i) 都作为正对，不同类样本作为负对。这使得所有属于同一类的样本在嵌入空间中更加紧凑。

### 损失函数 / 训练策略

**总损失**：L_total = alpha * L_BCE + L_SCL

alpha 控制分类损失与对比损失的平衡。消融实验显示 alpha > 1 时公平-准确权衡趋于稳定。

**理论分析（互信息分解）**：

在配对马尔可夫性假设下，正对之间的互信息可分解为：

I(Z; Z+) = I(Z; Y) + (1-pi) * I(Z; S|Y)

其中 pi = Pr[S+ != S | Y=1] 为交叉组配对概率。

**关键定理（InfoNCE 等价信息瓶颈）**：

argmin_theta L_NCE(theta) = argmax_theta { I(Z;Y) - lambda * I(Z; S|Y) }

其中 lambda = 1-pi。这意味着仅通过配对策略，标准对比学习就自然转化为信息瓶颈：保留标签相关信息 I(Z;Y)，压制条件敏感信息 I(Z;S|Y)。权衡系数完全由数据驱动，无需额外调参。

**训练配置**：Adam 优化器，lr=0.001，tau=1，100 epoch，NVIDIA RTX 3090 GPU。

## 实验关键数据

### 主实验

三个公平性基准数据集上的准确率和人口统计均等（DP，越低越公平）：

| 数据集 | 模型 | Accuracy | DP |
|--------|------|----------|-----|
| Adult | Unfair MLP | 84.5 | 0.1855 |
| | FCRL | 83.29 | 0.1600 |
| | CVIB | 81.28 | 0.1350 |
| | SCARF | 82.13 | 0.1848 |
| | VIME-self | 84.47 | 0.1779 |
| | FairContrast-sup | 84.4 | 0.0255 |
| | FairContrast-unsup | 84.4 | 0.1201 |
| German | Unfair MLP | 78.5 | 0.3125 |
| | CVIB | 69.5 | 0.0244 |
| | VIME-self | 78.0 | 0.0482 |
| | FairContrast-sup | 78.0 | 0.0099 |
| Health | Unfair MLP | 84.64 | 0.6468 |
| | FCRL | 78.27 | 0.4407 |
| | VIME-self | 84.22 | 0.6192 |
| | FairContrast-sup | 84.3 | 0.4135 |

核心发现：
- Adult 数据集上有监督版 DP 仅 0.0255（vs Unfair MLP 的 0.1855），降低约 86%，同时准确率几乎不变
- German 数据集上 DP 降至 0.0099（接近完美公平），且保持 78% 准确率
- SCARF/VIME 等不考虑公平的对比学习方法偏见水平与 Unfair MLP 相当

### 消融实验

**损失权重 alpha 消融**（Adult 数据集）：
- 使用 AOC（公平-准确 Pareto 曲线下面积）评价
- alpha > 1 时有监督和自监督设置均趋于稳定
- 较小的 alpha 使对比损失主导，可能过度压制信息；较大 alpha 使分类损失主导

**反事实数据增强对比**：
- 反事实增强（翻转敏感属性）也可集成到框架中
- 但效果不如 FairContrast 的策略性配对（Adult 上 DP=0.1639 vs 0.0255）

### 关键发现

- 仅通过精心设计的配对策略就能实现显著的去偏效果，无需对抗训练或额外约束
- 有监督版本一致优于自监督版本，表明标签信息对公平表示学习至关重要
- VIME/SCARF 在公平性上与未去偏的 MLP 表现类似，说明通用对比学习不自动提升公平性
- 理论上证明了公平-准确权衡系数 lambda = 1-pi 完全由数据分布决定

## 亮点与洞察

1. **极简优雅的设计**：不引入对抗器、不加公平约束损失、不修改架构，仅通过配对策略就将标准对比学习转化为公平信息瓶颈，用最少的改动实现最大的效果
2. **理论与实践完美对应**：互信息分解精确揭示了配对策略如何自动权衡标签信息保留和敏感信息压制
3. **通用性强**：学习到的公平表示可直接用于任意下游任务，具有跨任务迁移潜力
4. **Adult 数据集上 86% 的 DP 降低几乎不牺牲精度**，展示了公平与准确不必矛盾的理念

## 局限与展望

1. 仅考虑组公平（人口统计均等），未涉及个体公平
2. 当前关注单一公平指标 DP，而实际中多种公平定义可能相互冲突
3. 仅在表格数据上验证，图像/文本/多模态场景的扩展尚未探索
4. 正对策略要求已知敏感属性的标签，在敏感属性不可观测时不适用
5. 实验数据集规模较小（Adult 48K, German 1K, Health 50K），大规模场景验证缺失

## 相关工作与启发

- **SupCon**（Khosla et al., NeurIPS 2020）：有监督对比损失的基础
- **SCARF**（Bahri et al., 2021）：表格数据对比学习 SOTA，但不考虑公平
- **FCRL**（Gupta et al., AAAI 2021）：基于互信息估计的公平表示学习
- **CCL**（Park et al.）：条件对比学习处理公平，但增强策略不适合表格数据
- 启示：许多公平性问题可通过数据层面的配对和采样策略解决，而非复杂的模型层面约束

## 评分

4/5 优秀。方法设计简洁优雅、理论分析完整、实验充分对比。核心思想即配对策略等价信息瓶颈是令人印象深刻的洞察。主要遗憾是仅验证了表格数据和组公平场景。

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Mitigating Disparate Impact of Differentially Private Learning through Bounded Adaptive Clipping](mitigating_disparate_impact_of_differentially_private_learning_through_bounded_a.md)
- [\[ICCV 2025\] Backdooring Self-Supervised Contrastive Learning by Noisy Alignment](../../ICCV2025/ai_safety/backdooring_self-supervised_contrastive_learning_by_noisy_alignment.md)
- [\[CVPR 2025\] A Simple Data Augmentation for Feature Distribution Skewed Federated Learning](../../CVPR2025/ai_safety/a_simple_data_augmentation_for_feature_distribution_skewed_federated_learning.md)
- [\[NeurIPS 2025\] Geo-Sign: Hyperbolic Contrastive Regularisation for Geometrically Aware Sign Language Translation](geo-sign_hyperbolic_contrastive_regularisation_for_geometrically_aware_sign_lang.md)
- [\[ICLR 2026\] Less is More: Towards Simple Graph Contrastive Learning](../../ICLR2026/ai_safety/less_is_more_towards_simple_graph_contrastive_learning.md)

<!-- RELATED:END -->
