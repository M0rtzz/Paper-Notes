---
title: >-
  [论文解读] ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data
description: >-
  [NeurIPS 2025][预训练][零样本学习] ZEUS 是首个面向表格数据的零样本聚类方法，通过在合成数据集上预训练一个 Transformer 编码器来学习可泛化的表示，使得新数据集无需任何额外训练或调参即可在单次前向传播中完成高质量聚类。
tags:
  - "NeurIPS 2025"
  - "预训练"
  - "零样本学习"
  - "表格数据"
  - "聚类"
  - "Transformer"
  - "Prior-data Fitted Networks"
---

# ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data

**会议**: NeurIPS 2025  
**arXiv**: [2505.10704](https://arxiv.org/abs/2505.10704)  
**代码**: [GitHub](https://github.com/gmum/zeus)  
**领域**: 表格数据聚类 / 无监督学习  
**关键词**: 零样本学习, 表格数据, 聚类, Transformer, Prior-data Fitted Networks

## 一句话总结

ZEUS 是首个面向表格数据的零样本聚类方法，通过在合成数据集上预训练一个 Transformer 编码器来学习可泛化的表示，使得新数据集无需任何额外训练或调参即可在单次前向传播中完成高质量聚类。

## 研究背景与动机

表格数据聚类一直是无监督学习中的核心难题。与图像数据不同，表格数据缺乏天然的空间或语义结构，不同数据集之间的"相似性"定义差异巨大，导致聚类方法难以跨数据集泛化。

现有方法存在两大痛点：

**经典方法（如 k-means、GMM）** 依赖预定义的距离度量，无法捕捉复杂非线性关系，在高维异构数据上表现有限

**深度学习方法（如 DEC、IDEC）** 虽然能学习更丰富的表示，但对超参数极度敏感，且在无监督场景下缺乏标签信号来指导调参，导致性能不稳定且需要逐数据集训练

ZEUS 的核心思路来源于 Prior-data Fitted Networks (PFNs) 框架的启发：既然 TabPFN 已在监督任务上展示了上下文学习的潜力，能否将其扩展到无监督领域？但关键挑战在于：（1）如何生成具有清晰但非平凡聚类结构的合成训练数据；（2）如何在没有标签的情况下编码聚类先验知识。

## 方法详解

### 整体框架

ZEUS 的工作流程分三阶段：
- **合成数据生成**：从隐变量模型采样大量具有已知聚类结构的合成数据集
- **预训练**：在合成数据上训练 Transformer 编码器，使其学会将数据映射到便于聚类的表示空间
- **推理**：对新数据集直接进行一次前向传播得到表示，再用简单的 k-means 完成聚类

### 关键设计

1. **概率聚类表示学习**：ZEUS 不直接输出聚类标签，而是训练编码器 $f_\theta$ 将输入 $\mathbf{x}$ 映射到表示向量 $z(x_i) = f_\theta(\mathbf{x})_i \in \mathbb{R}^D$。借鉴 GMM 的思想，在表示空间中定义 $K$ 个聚类中心 $c_k$，通过 softmax 将距离转化为聚类隶属概率：
    $p_k(x_i) = \frac{\hat{\pi}_k \exp(\alpha_k(x_i))}{\sum_{j=1}^K \hat{\pi}_j \exp(\alpha_j(x_i))}, \quad \alpha_k(x_i) = -\|z(x_i) - \hat{c}_k\|^2$
   其中 $\hat{\pi}_k$ 是从训练标签估计的类先验，$\hat{c}_k$ 是同类样本表示的均值。训练目标是最大化正确聚类赋值的对数似然 $\mathcal{L}_{prob} = -\sum_i \log p_{y_i}(x_i)$。这样设计巧妙地将无监督聚类问题在预训练阶段转化为有监督分类问题。

2. **合成数据生成先验**：ZEUS 从隐变量模型采样训练数据，每个数据集是 $K$（2到10）个分量的混合分布。具体包含三种先验：

    - **高斯混合**：多元高斯分布，组件间有充分分离约束
    - **神经网络变换**：将高斯样本通过随机 ResNet 进行非线性变换，产生更真实的非高斯聚类形状。选用 ResNet 是因为其可以定义可逆变换，保证聚类结构在输出中得以保留
    - **分类特征**：以 one-hot 编码形式追加的离散特征
   
   这种设计的核心理念是让模型学会"反演数据生成过程"，而非像 k-means 或 DEC 那样优化任意的启发式目标。

3. **正则化机制**：除主损失 $\mathcal{L}_{prob}$ 外，引入两个正则项以提升推理质量：

    - **点集中正则 $\mathcal{L}_{cp}$**：最小化同类样本表示到聚类中心的距离，增强类内紧凑性，类似 k-means 的类内平方和目标
    $\mathcal{L}_{cp} = \sum_k \sum_{i:y_i=k} \alpha_k(x_i)$
    - **中心分离正则 $\mathcal{L}_{sep}$**：最大化不同聚类中心间的距离（带截断阈值 $T$ 防止无限远推），避免中心坍塌
    $\mathcal{L}_{sep} = -\sum_{k=1}^K \sum_{j=k+1}^K \min(\|\hat{c}_k - \hat{c}_j\|^2, T)$

### 损失函数 / 训练策略

最终损失为三项加权组合：
$$\mathcal{L} = \mathcal{L}_{prob} + \lambda_{cp}\mathcal{L}_{cp} + \lambda_{sep}\mathcal{L}_{sep}$$
其中 $\lambda_{cp} = \lambda_{sep} = 1$。模型采用 12 层注意力块、6 头、512 维的 Transformer 架构（类似 TabPFN）。使用 Adam 优化器，学习率 2e-5，余弦调度器带 warm-up。预训练时高斯和变换数据各占 50%，每 epoch 采样 1000 个数据集。推理时先标准化数值特征到 $[-1,1]$，分类特征 one-hot 编码，超过 30 维用 PCA 降维，最后对归一化后的 Transformer 输出执行 k-means。

## 实验关键数据

### 主实验

评估指标为 ARI（×100），在 34 个 OpenML 真实数据集、20 个合成高斯数据集、20 个合成变换数据集上对比 11 种方法。

| 数据集组 | ZEUS | DEC | k-means | GMM | TabPFN | 最佳基线 |
|----------|------|-----|---------|-----|--------|---------|
| Real (OpenML) | **57.43** | 55.93 | 55.54 | 48.49 | 31.32 | 55.93 (DEC) |
| Syn. Gauss. | 89.03 | 89.35 | **89.90** | 76.93 | 55.97 | 89.90 (KM) |
| Syn. Transf. | **86.33** | 79.94 | 75.04 | 75.88 | 15.66 | 79.94 (DEC) |

| 数据集组 | ZEUS 平均排名 | DEC 排名 | KM 排名 | 最差方法 |
|----------|-------------|----------|---------|---------|
| Real | **4.13** | 4.69 | 4.69 | SCARF (8.85) |
| Syn. Gauss. | 2.92 | 3.23 | **2.65** | SCARF (11.00) |
| Syn. Transf. | **2.20** | 3.20 | 4.80 | SCARF (11.00) |

### 消融实验

| 损失组合 | Real ARI | Syn. Gauss. ARI | Syn. Transf. ARI |
|---------|---------|----------------|-----------------|
| $\mathcal{L}_{prob}$ only | 44.80 | 83.37 | 79.85 |
| $+\mathcal{L}_{sep}$ | 51.60 | 81.88 | 79.29 |
| $+\mathcal{L}_{cp}$ | 48.65 | **90.59** | **88.58** |
| $+\mathcal{L}_{sep}+\mathcal{L}_{cp}$ (完整) | **57.43** | 89.03 | 86.33 |

数据先验消融：

| 先验组合 | Real | Syn. Gauss. | Syn. Transf. |
|---------|------|-------------|-------------|
| Gauss. + Cat. | 40.59 | **92.61** | 73.34 |
| NN-transf. + Cat. | 50.90 | 89.90 | **87.04** |
| Gauss. + NN-transf. | 52.00 | 75.25 | 71.29 |
| 全部三种 (ZEUS) | **57.43** | 89.03 | 86.33 |

### 关键发现

- ZEUS 在最具挑战性的真实数据和合成变换数据上均排名第一，在简单高斯数据上排前三
- $\mathcal{L}_{cp}$ 对合成数据至关重要（+7pp），$\mathcal{L}_{sep}$ 对真实数据很重要（+9pp），两者结合在真实数据上提升最大
- 三种先验的组合对真实数据泛化至关重要：仅用高斯先验在真实数据上 ARI 仅 40.59
- ZEUS 的推理时间几乎恒定，仅略慢于基础 k-means，远快于深度聚类方法
- Brier score 评估显示 ZEUS 的概率赋值校准质量优秀，在合成变换数据上显著领先

## 亮点与洞察

- **范式创新**：将聚类视为"反演数据生成过程"而非优化启发式目标，这一思路与生成模型的理念一致
- **理论扎实**：证明了 ZEUS 符合 PFN 框架，隐式实现了近似贝叶斯推断
- **实用性强**：即插即用、单次前向传播、无须调参，真正解决了深度聚类方法易用性差的痛点
- 巧妙利用了预训练阶段的已知标签来估计中心和先验，规避了无监督场景的调参困难

## 局限与展望

- 继承 TabPFN 的限制：输入特征数上限 30（超过需 PCA），样本数上限 2000
- 聚类质量强依赖合成数据先验的设计，当真实数据分布与先验差距大时可能退化
- 模型本身不做聚类，只提供表示空间，仍需外接 k-means 等算法
- 未来可探索更丰富的数据生成先验、扩展到更大规模数据集（参考 TabPFN v2 思路）

## 相关工作与启发

- 可与 TabPFN v2 结合，突破规模限制
- 先验设计的思路可扩展到异常检测、缺失值补全等其他无监督表格任务
- 对于领域特定数据，可通过定制先验分布进一步提升性能

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个零样本表格数据聚类方法，范式创新明确
- **实验充分度**: ⭐⭐⭐⭐ 34 个真实数据集 + 40 个合成数据集，消融全面，但大规模场景未验证
- **写作质量**: ⭐⭐⭐⭐⭐ 逻辑清晰，理论推导严谨，实验组织良好
- **价值**: ⭐⭐⭐⭐ 在表格数据聚类领域具有标杆意义，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](climb_class-imbalanced_learning_benchmark_on_tabular_data.md)
- [\[ACL 2025\] AutoDS: Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts](../../ACL2025/llm_pretraining/autonomous_data_selection_with_zero-shot_generative_classifiers_for_mathematical.md)
- [\[ECCV 2024\] Prompting Language-Informed Distribution for Compositional Zero-Shot Learning](../../ECCV2024/llm_pretraining/prompting_language-informed_distribution_for_compositional_zero-shot_learning.md)
- [\[ACL 2025\] Unsupervised Morphological Tree Tokenizer](../../ACL2025/llm_pretraining/unsupervised_morphological_tree_tokenizer.md)
- [\[AAAI 2026\] GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval](../../AAAI2026/llm_pretraining/granalign_granularity-aware_alignment_framework_for_zero-shot_video_moment_retri.md)

</div>

<!-- RELATED:END -->
