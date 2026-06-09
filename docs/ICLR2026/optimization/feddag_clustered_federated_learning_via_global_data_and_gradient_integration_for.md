---
title: >-
  [论文解读] FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments
description: >-
  [ICLR 2026][优化/理论][聚类联邦学习] 提出 FedDAG 聚类联邦学习框架，通过融合数据和梯度信息进行加权类别级相似度计算来实现更准确的客户端聚类，并通过双编码器架构实现跨集群特征迁移，在多种异构性设置下一致超越现有基线。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "聚类联邦学习"
  - "数据异构性"
  - "双编码器架构"
  - "跨集群知识共享"
  - "自适应聚类"
---

# FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments

**会议**: ICLR 2026  
**arXiv**: [2602.23504](https://arxiv.org/abs/2602.23504)  
**代码**: [https://tinyurl.com/2rbkb3zu](https://tinyurl.com/2rbkb3zu)  
**领域**: 优化 / 联邦学习  
**关键词**: 聚类联邦学习, 数据异构性, 双编码器架构, 跨集群知识共享, 自适应聚类

## 一句话总结
提出 FedDAG 聚类联邦学习框架，通过融合数据和梯度信息进行加权类别级相似度计算来实现更准确的客户端聚类，并通过双编码器架构实现跨集群特征迁移，在多种异构性设置下一致超越现有基线。

## 研究背景与动机

**领域现状**：联邦学习（FL）通过协作训练模型而不共享数据，但客户端数据异构性（non-IID）会导致收敛慢和精度不佳。聚类 FL 通过将相似客户端分组来应对，每个集群训练自己的模型。

**现有痛点**：现有聚类 FL 方法存在四大限制：1) 仅使用数据或梯度单一信号计算相似度，不够全面；2) 知识共享限制在同一集群内，无法利用跨集群的多样化表征；3) 主要处理标签偏斜，忽视概念漂移和数量偏移；4) 需要预先指定集群数量。

**核心矛盾**：数据相似度和梯度相似度各有盲区——高维数据中梯度相似度可能产生误判，而数据相似度忽视概念漂移。单独使用任一信号都无法准确刻画客户端间的真实相似性。

**本文目标** 如何综合利用数据和梯度信息动态聚类客户端，同时允许集群间的表征共享？

**切入角度**：将相似度计算细化到类别级（class-wise），为数据和梯度信号自动学习权重，并用双编码器架构实现跨集群特征迁移。

**核心 idea**：通过类别级加权融合数据和梯度相似度进行更精准的客户端聚类，并用双编码器架构在保持集群特化的同时实现跨集群知识共享。

## 方法详解

### 整体框架
FedDAG 把聚类联邦学习拆成两段串起来跑：先融合数据和梯度信息算出客户端之间的加权相似度矩阵，用层次聚类配上联邦感知指标自动定下集群划分；再让每个集群模型带一对主副编码器，主编码器吃本集群数据、副编码器吃互补集群梯度，拼接特征后训分类器，从而在保持集群特化的同时把跨集群的互补表征引进来。

### 关键设计

**1. 加权类别级相似度：让数据和梯度互相补盲区。** 聚类的成败全看相似度矩阵准不准，而单看数据子空间会漏掉概念漂移，单看梯度在高维下又容易误判。FedDAG 把 PACFL 的子空间比较细化到类别级——只在同一类别内部比较数据子空间而不是拿整体子空间硬比，这样"同一个标签在不同客户端含义不同"的概念漂移就能自然暴露出来。在此基础上，每个客户端学一个权重 $w_i$ 来调和两路信号，最终相似度写成 $S_{ij} = w_i \cdot S_{ij}^{data} + (1 - w_i) \cdot S_{ij}^{grad}$；$w_i$ 通过最小化一个基于熵的损失来优化，目的是把邻接矩阵推向更锐利、更接近二值的形态，让该相似的更相似、该分开的更分开。融合而非二选一，等于让每个客户端自适应地挑出对自己最有信息量的那路信号。

**2. 双编码器架构：把集群特化和跨集群互补分到两条通道。** 现有方法要么把知识共享锁死在集群内部、白白浪费别的集群的多样表征，要么用软聚类让客户端混搭多个集群模型、结果把噪声也混了进来。FedDAG 给每个集群模型配主编码器 $\phi^{(1)}$ 和副编码器 $\phi^{(2)}$：主编码器用本集群客户端的聚合梯度更新参数 $\Theta_z^{1f}$，专注学集群内的特化特征；副编码器用互补集群的梯度更新 $\Theta_z^{2f}$，负责引进外部视角。两路输出在特征维度上拼接后再送进分类器，$F_z(\cdot) = \psi(\phi^{(1)}(\cdot; \Theta_z^{1f}), \phi^{(2)}(\cdot; \Theta_z^{2f}); \Theta_z^c)$。结构上的分离保证了跨集群知识是"并列引入"而不是"混合污染"，集群既能保住自己的特化，又能借到别人的互补信息。

**3. 联邦感知自适应聚类：不用人工预设集群数。** 实际场景里根本没法提前知道该分几个集群，而层次聚类在联邦设定下又特别容易过度分裂、切出一堆只有几个客户端的退化集群。FedDAG 让层次聚类先生成不同粒度的候选划分，再用一个新提出的联邦感知指标逐一打分：这个指标一边奖励集群内部的紧凑度，一边惩罚过度分裂出来的小集群，最后选得分最高的划分定案，从而在"分得够细"和"别分得太碎"之间自动找到平衡。

训练侧整体仍是标准配置：交叉熵损失在每个集群内聚合，相似度权重的优化靠前面提到的熵正则把邻接矩阵往二值推。通信上对传输的梯度做压缩，且每个客户端每轮至多只需算一个模型的梯度，把跨集群共享的开销压到可接受范围。

## 实验关键数据

### 主实验

| 算法 | 技术 | CIFAR-10 | FMNIST |
|------|------|----------|--------|
| PACFL | 数据 (D) | 90.45±0.30 | 94.41±0.31 |
| CFL | 梯度 (G) | 72.80±0.66 | 86.97±0.23 |
| IFCA | 梯度 (G) | 89.68±0.17 | 94.03±0.09 |
| **FedDAG** | **D+G+全局特征共享** | **94.53±0.12** | **96.82±0.18** |

### 消融实验

| 配置 | CIFAR-10 | 说明 |
|------|----------|------|
| FedDAG (完整) | 94.53 | 完整框架 |
| 仅数据相似度 | ~91.0 | 退化为 PACFL++ |
| 仅梯度相似度 | ~88.5 | 退化为改进版 CFL |
| 无双编码器 | ~92.0 | 无跨集群特征 |
| 无自适应聚类数 | ~93.0 | 使用预设聚类数 |

### 关键发现
- FedDAG 在 CIFAR-10 上比最强基线 PACFL 高出 4+ 个百分点
- 数据和梯度信号的融合一致性优于单一信号，尤其在概念漂移场景下
- 双编码器架构相比单编码器带来 2-3% 的提升，跨集群知识共享确实有价值
- 在标签偏斜、特征偏斜、概念漂移和数量偏移四种异构类型下都有效

## 亮点与洞察
- **类别级相似度计算**：将相似度细化到类别维度是处理概念漂移的自然方式，比整体子空间比较更鲁棒
- **双编码器的职责分离设计**：主副编码器各自专注不同信号来源，避免了软聚类方法中的噪声混合问题

## 局限与展望
- 双编码器增加了模型参数和计算开销
- 类别级比较在类别数很多时计算成本增长
- 依赖客户端上传少量信息进行相似度计算，虽然压缩但仍有隐私风险
- 未在真实联邦场景（如跨设备 FL）中测试

## 相关工作与启发
- **vs PACFL**: PACFL 用主角度比较整体子空间，FedDAG 改为类别级比较+加权融合，更全面
- **vs FedSoft/FedRC**: 它们通过软聚类让客户端混合多个集群模型，可能引入噪声；FedDAG 的双编码器在结构上分离了两个信号来源

## 评分
- 新颖性: ⭐⭐⭐ 类别级融合和双编码器是合理但增量式创新
- 实验充分度: ⭐⭐⭐⭐ 四种异构类型的评估较全面
- 写作质量: ⭐⭐⭐ 内容充实但结构略显复杂
- 价值: ⭐⭐⭐ 对聚类 FL 有实际改进，但场景较为特定

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Incentives in Federated Learning with Heterogeneous Agents](incentives_in_federated_learning_with_heterogeneous_agents.md)
- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](../../AAAI2026/optimization/smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [\[ICML 2026\] Adaptive Estimation and Inference in Semi-parametric Heterogeneous Clustered Multitask Learning via Neyman Orthogonality](../../ICML2026/optimization/adaptive_estimation_and_inference_in_semi-parametric_heterogeneous_clustered_mul.md)
- [\[ICCV 2025\] Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](../../ICCV2025/optimization/federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)
- [\[ICML 2026\] Learning Locally, Revising Globally: Global Reviser for Federated Learning with Noisy Labels](../../ICML2026/optimization/learning_locally_revising_globally_global_reviser_for_federated_learning_with_no.md)

</div>

<!-- RELATED:END -->
