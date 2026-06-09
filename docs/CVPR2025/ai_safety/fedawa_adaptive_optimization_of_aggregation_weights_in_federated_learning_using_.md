---
title: >-
  [论文解读] FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors
description: >-
  [CVPR 2025][AI安全][联邦学习] 提出 FedAWA，受任务算术（task arithmetic）启发，用客户端向量（本地参数与全局参数的差值）来自适应优化联邦学习中的聚合权重——与全局优化方向一致的客户端获得更高权重，在 non-IID 场景下稳定提升 FedAvg 1-4 个点。
tags:
  - "CVPR 2025"
  - "AI安全"
  - "联邦学习"
  - "聚合权重优化"
  - "客户端向量"
  - "任务算术"
  - "非IID数据"
---

# FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors

**会议**: CVPR 2025  
**arXiv**: [2503.15842](https://arxiv.org/abs/2503.15842)  
**代码**: [https://github.com/ChanglongShi/FedAWA](https://github.com/ChanglongShi/FedAWA)  
**领域**: AI安全 / 联邦学习  
**关键词**: 联邦学习, 聚合权重优化, 客户端向量, 任务算术, 非IID数据

## 一句话总结

提出 FedAWA，受任务算术（task arithmetic）启发，用客户端向量（本地参数与全局参数的差值）来自适应优化联邦学习中的聚合权重——与全局优化方向一致的客户端获得更高权重，在 non-IID 场景下稳定提升 FedAvg 1-4 个点。

## 研究背景与动机

### 领域现状

**领域现状**：联邦学习（FL）通过在多客户端间聚合模型参数来训练全局模型。FedAvg 按数据量加权聚合，但在 non-IID 场景下不同客户端的更新方向可能互相冲突，导致全局模型收敛不稳定。

**现有痛点**：现有自适应聚合方法（如 FedLAW、L-DAWA）要么计算量大（10+ 秒/轮），要么需要额外的验证数据。缺乏一种既轻量又有效的聚合权重优化方案。

**核心矛盾**：等权或数据量加权聚合忽略了客户端更新的"质量"——在 non-IID 下某些客户端的更新方向是有害的（偏离全局最优），应该降低其权重。

**切入角度**：模型合并领域的任务算术理论表明，参数差值向量（task vector）包含了任务特定知识。将其迁移到 FL——客户端向量 $\tau_k = \theta_k - \theta_g$ 反映了本地数据的特征，可以用来衡量更新的"有用性"。

**核心 idea**：用客户端向量与全局聚合向量的对齐度优化聚合权重 = 更一致的全局更新方向。

## 方法详解

### 关键设计

1. **客户端向量驱动的权重优化**:

    - 功能：自适应地为每个客户端分配聚合权重
    - 核心思路：定义客户端向量 $\tau_k^t = \theta_k^t - \theta_g^t$，全局聚合向量 $\tau_g^t = \sum_k \lambda_k \tau_k^t$。优化目标：$\min_\lambda \sum_k \lambda_k \|\tau_k^t - \tau_g^t\|_2 + d(\sum_k \lambda_k \theta_k^t, \theta_g^t)$，约束 $\|\lambda\|_1 = 1$。第一项鼓励选择与全局方向一致的客户端，第二项约束聚合后不偏离太远
    - 设计动机：Figure 2 验证客户端向量的差异确实反映了数据分布差异，全局向量比单个客户端向量更接近"理想"向量

2. **逐层变体 FedAWA-L**:

    - 功能：为每层独立优化权重，更细粒度的控制
    - 核心思路：对每层 $l$ 独立求解 $\lambda_l^t$，不同层可以有不同的最优权重组合
    - 设计动机：不同层学习不同层次的特征，non-IID 对浅层（局部特征）和深层（语义特征）的影响不同

### 损失函数 / 训练策略

聚合权重通过约束优化求解，使用 1-余弦相似度作为距离函数。客户端本地用标准 SGD + 交叉熵训练。聚合时间仅 0.82 秒（vs L-DAWA 2.52 秒，FedLAW 10.11 秒）。

## 实验关键数据

### 主实验

CIFAR-10 Top-1 准确率（%）：

| 方法 | IID (α=100) | non-IID (α=0.5) |
|------|-------------|-----------------|
| FedAvg | 76.01 | 74.47 |
| FedProx | 76.47 | 73.85 |
| **FedAWA** | **80.10** | **75.65** |
| **FedAWA-L** | **79.70** | 74.90 |

### 消融实验

| 配置 | 效果 |
|------|------|
| 与 FedDisco 结合 | 额外提升，证明即插即用兼容性 |
| K=10/30/50 客户端 | 一致提升，参数鲁棒 |
| E=1/5/10 本地轮次 | 一致提升 |

### 关键发现
- **IID 下提升也显著**：80.10 vs 76.01，说明即使数据分布相同，等权聚合也不是最优的
- **计算开销极低**：0.82 秒/轮，仅比 FedAvg 多一个优化步
- **跨架构通用**：CNN/ResNet/WRN/DenseNet/ViT 都有效

## 亮点与洞察
- **Task arithmetic 的联邦学习迁移**——客户端向量 = 任务向量的 FL 版本，这个类比简洁且有效
- **即插即用设计**——可与任何现有 FL 方法结合，不改变本地训练过程

## 局限与展望
- 需要存储和传输客户端级参数向量，通信开销增加
- FedAWA-L 聚合时间 15.21 秒 vs 全局版 0.82 秒
- 缺少收敛性理论分析
- 客户端向量可能间接泄露模型更新信息

## 评分
- 新颖性: ⭐⭐⭐⭐ Task arithmetic → FL 的迁移简洁有效
- 实验充分度: ⭐⭐⭐⭐ 多数据集多架构多配置
- 写作质量: ⭐⭐⭐⭐ 清晰完整
- 价值: ⭐⭐⭐ 提升幅度中等但方法轻量实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing](../../ICCV2025/ai_safety/client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)
- [\[CVPR 2025\] A Simple Data Augmentation for Feature Distribution Skewed Federated Learning](a_simple_data_augmentation_for_feature_distribution_skewed_federated_learning.md)
- [\[ICCV 2025\] LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement](../../ICCV2025/ai_safety/lora-fair_federated_lora_fine-tuning_with_aggregation_and_initialization_refinem.md)
- [\[CVPR 2025\] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)
- [\[CVPR 2025\] Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning](geometric_knowledge-guided_localized_global_distribution_alignment_for_federated.md)

</div>

<!-- RELATED:END -->
