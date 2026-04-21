---
title: >-
  [论文解读] Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing
description: >-
  [ICCV 2025][AI安全][联邦学习] 提出Client2Vec机制，在联邦学习训练前利用CLIP编码器和分布偏移感知索引生成网络（DSA-IGN）为每个客户端生成包含标签和特征分布信息的索引向量，进而改善客户端采样、模型聚合和本地训练三个关键阶段。
tags:
  - ICCV 2025
  - AI安全
  - 联邦学习
  - 分布偏移
  - 客户端索引
  - CLIP
  - 非IID数据
---

# Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing

**会议**: ICCV 2025  
**arXiv**: [2405.16233](https://arxiv.org/abs/2405.16233)  
**代码**: https://github.com/LINs-lab/client2vec  
**领域**: AI安全/联邦学习  
**关键词**: 联邦学习, 分布偏移, 客户端索引, CLIP, 非IID数据

## 一句话总结

提出Client2Vec机制，在联邦学习训练前利用CLIP编码器和分布偏移感知索引生成网络（DSA-IGN）为每个客户端生成包含标签和特征分布信息的索引向量，进而改善客户端采样、模型聚合和本地训练三个关键阶段。

## 研究背景与动机

联邦学习（FL）的核心挑战在于客户端之间的数据分布异构性（non-IID）。现有方法主要在**训练过程中**进行优化：改进客户端采样策略、模型聚合权重或本地训练目标等。但很少有工作从训练**之前**入手来缓解该问题。

已有的预训练阶段工作（如数据集蒸馏FedFed、合成伪数据VHL）存在额外计算成本高、适用场景有限、与训练流程不兼容等缺点。受NLP中Word2Vec和域泛化中域索引的启发，作者提出了一个关键问题：能否在训练前为每个客户端生成一个"身份向量"，编码其本地数据分布信息，从而在整个训练流程中提供辅助？

Client2Vec的三大优势：(1) 索引生成与FL训练解耦，减轻训练负担；(2) 每个客户端仅需一个索引向量，高效简洁；(3) 可增强FL训练的所有阶段（采样、聚合、本地训练）。

## 方法详解

### 整体框架

分两个阶段：(1) 训练前——通过DSA-IGN网络为每个客户端生成索引向量 $\boldsymbol{\beta}_i = [\boldsymbol{\beta}_i^f; \boldsymbol{\beta}_i^l]$，包含特征索引和标签索引；(2) 训练中——利用生成的索引改进三个案例：客户端采样、模型聚合和本地训练。

### 关键设计

1. **CLIP编码与索引定义**: 利用预训练CLIP模型将原始数据 $(x_{i,j}, y_{i,j})$ 编码为图像嵌入 $\mathbf{D}_{i,j}$（包含标签和客户端特定信息）和标签嵌入 $\mathbf{L}_{i,j}$（仅包含标签信息）。样本标签索引直接设为 $\mathbf{u}_{i,j}^l = \mathbf{L}_{i,j}$；样本特征索引 $\mathbf{u}_{i,j}^f$ 需从 $\mathbf{D}_{i,j}$ 中分离出与标签无关的客户端特定信息。客户端索引为所有样本索引的均值：$\boldsymbol{\beta}_i = \frac{1}{N_i}\sum_{j=1}^{N_i}\mathbf{u}_{i,j}$。这一设计的核心理念是：特征索引应编码客户端特有的分布特征（如风格、背景等），而非与分类相关的标签信息。

2. **分布偏移感知索引生成网络（DSA-IGN）**: 使用三层Transformer编码器将 $\mathbf{D}_{i,j}$ 分解为数据编码 $\mathbf{z}_{i,j}$（与标签相关）和特征索引 $\mathbf{u}_{i,j}^f$（与标签无关）。训练目标包含四个损失：(a) $\mathcal{L}_{\text{sim}}$——对齐 $\mathbf{z}_{i,j}$ 和标签嵌入，确保标签敏感性；(b) $\mathcal{L}_{\text{orth}}$——保证 $\mathbf{u}_{i,j}^f$ 和 $\mathbf{z}_{i,j}$ 正交独立；(c) $\mathcal{L}_{\text{recon}}$——拼接 $\mathbf{u}_{i,j}^f$ 和 $\mathbf{z}_{i,j}$ 重建 $\mathbf{D}_{i,j}$，保留完整信息；(d) $\mathcal{L}_{\text{div}}$——类似SimCLR的负对损失，促进不同样本的 $\mathbf{u}_{i,j}^f$ 多样性，避免训练坍塌。支持Global（上传128样本到服务器集中训练）和Federated（通过FedAvg联邦训练）两种策略。

3. **三个应用案例**:

    - **案例1（客户端采样）**: 基于贪心策略，让第 $t$ 轮采样的客户端与第 $t-1$ 轮相似。采样概率 $p_i^t = \frac{\exp(S(\boldsymbol{\beta}_i, \mathcal{C}^{t-1})/\tau)}{\sum_j \exp(S(\boldsymbol{\beta}_j, \mathcal{C}^{t-1})/\tau)}$，其中相似度函数 $S$ 同时考虑特征索引和标签索引的余弦相似度。
    - **案例2（模型聚合）**: 基于MWU算法，为相似度更高的客户端分配更大的聚合权重。求解优化问题得到 $p_{i,g}^t \propto q_i^t \exp(\frac{1}{\lambda_1}\sum_{\tau=1}^t \gamma^{t-\tau} S(\beta_i, \mathcal{C}^\tau))$，其中包含利润项（相似度）、熵项（正则化）和归一化约束。
    - **案例3（本地训练）**: 通过投影层将本地特征映射到与 $\boldsymbol{\beta}_i^f$ 同维空间，添加正交损失 $\mathcal{L}_{\text{orth}} = \|\mathbf{z}_P \mathbf{B}^f\|_1$ 鼓励本地特征学习与客户端特定信息正交的表征，加上蒸馏损失保留原始特征的信息量。

### 损失函数 / 训练策略

DSA-IGN的总损失为 $\mathcal{L} = \mathcal{L}_{\text{div}} + \mathcal{L}_{\text{sim}} + \mathcal{L}_{\text{orth}} + \mathcal{L}_{\text{recon}}$。FL训练阶段，本地训练损失为 $\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{orth}} + \mathcal{L}_{\text{dist}}$，在标准分类损失基础上添加正交约束和知识蒸馏。

## 实验关键数据

### 主实验

| 数据集(模型) | FL算法 | 原始 | +采样+聚合+本地训练(Global) | 最大提升 |
|-------------|--------|------|-------------------------|---------|
| Shakespeare(LSTM) | FedAvg | 49.93 | **50.51** | +0.58 |
| CIFAR10(ResNet18) | FedAvg | 42.24 | **59.29** | +17.05 |
| CIFAR10(ResNet18) | FedAvgM | 42.56 | **69.37** | +26.81 |
| CIFAR10(ResNet18) | FedDyn | 37.22 | **70.59** | +33.37 |
| DomainNet(MobileNetV2) | FedAvg | 46.31 | **57.43** | +11.12 |
| DomainNet(MobileNetV2) | Moon | 50.56 | **60.48** | +9.92 |

在CIFAR10上提升最为显著（最高+33.37%），在DomainNet上也有10%+的提升，说明Client2Vec对标签偏移和特征偏移均有效。

### 消融实验

| 配置 | CIFAR10(FedAvg) | DomainNet(FedAvg) | 说明 |
|------|----------------|------------------|------|
| 原始 | 42.24 | 46.31 | 基线 |
| +采样(i) | 44.60 | 50.78 | 采样改进约2-4% |
| +采样+聚合(i+ii) | 44.10 | 53.83 | 聚合进一步提升 |
| +全部(i+ii+iii) | **59.29** | **56.43** | 本地训练贡献最大 |

### 关键发现

- 三个案例的改进是**递进累加**的，且本地训练（案例3）贡献最大，说明消除本地特征中的客户端特定信息对模型泛化至关重要。
- DomainNet上的可视化显示：同一特征域的客户端索引相似度接近1.0，不同域之间距离大，验证了索引向量有效编码了分布信息。
- 域间相似度与人类直觉一致：Real域与Clipart、Painting、Sketch更近，与Infograph、Quickdraw更远。
- Global和Federated两种训练策略都能生成有意义的索引，Global策略的域边界更清晰。

## 亮点与洞察

- 将"训练前分析"与"训练中优化"解耦的思路值得借鉴：生成一次索引，全流程受益。
- 利用CLIP的跨模态对齐能力巧妙解决了"如何将标签和图像映射到同一空间"的问题。
- 正交约束的使用贯穿始终（索引生成阶段分离特征索引和数据编码，本地训练阶段分离本地特征和客户端特征索引），体现了一致的设计哲学。

## 局限与展望

- 依赖CLIP预训练模型，对于CLIP覆盖不好的领域（如医学影像）效果可能打折。
- Global策略需要上传部分数据嵌入到服务器，虽然是CLIP特征而非原始数据，但隐私风险仍需评估。
- 在NLP任务（Shakespeare）上提升较小（<1%），说明对于分布偏移不严重的场景收益有限。
- 索引维度 $d_i$ 的选择和DSA-IGN的训练epoch数需要针对不同场景调优。

## 相关工作与启发

- 延伸了VDI（变分域索引）的思路到联邦学习场景，同时解决了VDI在FL中的通信成本、隐私和标签偏移忽略问题。
- 与FedBR、VHL等数据共享方法相比，Client2Vec的通信开销更小（仅需索引向量）。
- MWU算法在模型聚合中的应用给出了理论优美的权重推导。

## 评分

- 新颖性: ⭐⭐⭐⭐ 训练前索引生成的思路新颖，三个案例应用覆盖全面
- 实验充分度: ⭐⭐⭐⭐ 三个数据集×多个基线算法×两种训练策略，实验矩阵完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，定义严谨
- 价值: ⭐⭐⭐ 实际提升在不同场景差异较大，NLP场景收益有限

<!-- RELATED:START -->

## 相关论文

- [FLUX: Efficient Descriptor-Driven Clustered Federated Learning under Arbitrary Distribution Shifts](../../NeurIPS2025/ai_safety/flux_efficient_descriptor-driven_clustered_federated_learning_under_arbitrary_di.md)
- [FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors](../../CVPR2025/ai_safety/fedawa_adaptive_optimization_of_aggregation_weights_in_federated_learning_using_.md)
- [A Simple Data Augmentation for Feature Distribution Skewed Federated Learning](../../CVPR2025/ai_safety/a_simple_data_augmentation_for_feature_distribution_skewed_federated_learning.md)
- [MARS: A Malignity-Aware Backdoor Defense in Federated Learning](../../NeurIPS2025/ai_safety/mars_a_malignity-aware_backdoor_defense_in_federated_learning.md)
- [Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning](../../CVPR2025/ai_safety/geometric_knowledge-guided_localized_global_distribution_alignment_for_federated.md)

<!-- RELATED:END -->
