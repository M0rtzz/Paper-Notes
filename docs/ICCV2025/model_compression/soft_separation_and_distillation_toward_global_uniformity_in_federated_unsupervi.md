---
title: >-
  [论文解读] Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning
description: >-
  [ICCV 2025][模型压缩][联邦无监督学习] 提出 SSD（Soft Separation and Distillation）框架，通过维度缩放正则化（DSR）和投影器蒸馏（PD）两个模块，改善联邦无监督学习中跨客户端的全局表征均匀性（inter-client uniformity），在不增加通信开销的前提下显著提升表征质量和下游任务性能。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "联邦无监督学习"
  - "表征均匀性"
  - "对比学习"
  - "知识蒸馏"
  - "非IID数据"
---

# Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning

**会议**: ICCV 2025  
**arXiv**: [2508.01251](https://arxiv.org/abs/2508.01251)  
**代码**: [项目页面](https://ssd-uniformity.github.io/)  
**领域**: 模型压缩 / 联邦学习  
**关键词**: 联邦无监督学习, 表征均匀性, 对比学习, 知识蒸馏, 非IID数据

## 一句话总结

提出 SSD（Soft Separation and Distillation）框架，通过维度缩放正则化（DSR）和投影器蒸馏（PD）两个模块，改善联邦无监督学习中跨客户端的全局表征均匀性（inter-client uniformity），在不增加通信开销的前提下显著提升表征质量和下游任务性能。

## 研究背景与动机

联邦无监督学习（FUL）旨在分布式且无标签的数据环境下学习高质量表征。表征质量主要由两个指标决定：**对齐性**（alignment，相似样本表征接近）和**均匀性**（uniformity，表征均匀分布在超球面上）。

现有 FUL 方法存在一个核心挑战：虽然各客户端能在本地数据上实现较好的**局部均匀性**（intra-client uniformity），但在模型聚合后无法保证**全局均匀性**（inter-client uniformity）。原因有二：
1. 非IID数据分布导致各客户端更新方向发散
2. 联邦学习的去中心化特性使服务器无法直接在跨客户端数据上施加均匀性约束

现有工作主要从两个方向尝试解决：一类是添加近端约束（FedProx、FedX等），限制本地更新偏离全局模型；另一类是缓解维度坍缩（FedDecorr、FedU2等），提升本地表征均匀性。然而这些方法都没有显式地解决跨客户端均匀性问题。

## 方法详解

### 整体框架

SSD 基于标准的联邦学习框架（FedAvg），在每个客户端的本地训练目标中引入两个额外的正则化项。整体训练流程为：
1. 服务器为每个客户端分配唯一的维度缩放向量 $\mathbf{d}_k$
2. 客户端在本地数据上进行自监督训练，优化组合损失
3. 服务器聚合所有客户端模型参数
4. 迭代重复步骤2-3直至收敛

总损失为：$\mathcal{L}^k = \mathcal{L}_{\text{align}}^k + \beta \mathcal{L}_{\text{uniform}}^k + \gamma \mathcal{L}_{\text{DSR}}^k + \delta \mathcal{L}_{\text{distill}}^k$

### 关键设计

1. **维度缩放正则化（DSR）**: 核心思想是鼓励不同客户端的嵌入在表征空间中朝不同方向扩展，从而在模型聚合时减少干扰。为每个客户端 $k$ 定义一个缩放向量 $\mathbf{d}_k \in \mathbb{R}^d$，其中 $\mathcal{S}_k$ 维度被缩放因子 $\alpha$ 拉大，不同客户端的缩放维度不重叠（$\mathcal{S}_i \cap \mathcal{S}_j = \emptyset$）。DSR 损失拉近嵌入与其缩放版本的距离：$\mathcal{L}_{\text{DSR}}^k = \mathbb{E}[\|\mathbf{z} - \text{stopgrad}(\mathbf{z} \odot \mathbf{d}_k)\|_2^2]$。这种"软分离"不同于将每个客户端限制在独立子空间的"硬分离"，允许维度共享同时引导方向差异，保持表征灵活性。

2. **投影器蒸馏（PD）**: 实验观察到 DSR 的正则化效果主要体现在 embedding 层面（投影器输出），而不能有效传递到 representation 层面（编码器输出），因为投影器会吸收大部分优化效果。但直接移除投影器会降低性能。PD 通过最小化表征和嵌入之间的 KL 散度来桥接这一差距：$\mathcal{L}_{\text{distill}}^k = \mathbb{E}[D_{\text{KL}}(\sigma(\mathbf{h}) \| \sigma(\mathbf{z}))]$。这使编码器能内化 embedding 空间中学到的有益结构。

3. **全局均匀性分析**: 论文形式化地将均匀性损失分解为 intra-client 和 inter-client 两部分。在联邦学习中，各客户端只能优化 intra-client 项；在 non-IID 设置下，优化 intra-client 均匀性并不能保证 inter-client 均匀性。SSD 通过鼓励不同客户端表征在超球面上占据不同区域来间接优化 inter-client 均匀性。

### 损失函数 / 训练策略

- $\beta = 1.0, \gamma = 1.0, \delta = 0.1$，缩放因子 $\alpha = 10.0$
- 每个客户端的缩放维度集合大小约为 $\lfloor d/K \rfloor$
- 编码器采用 ResNet-18，投影器为两层线性网络
- 无额外通信开销：只需在初始阶段传输轻量级权重向量 $\{\mathbf{d}_k\}$

## 实验关键数据

### 主实验

| 方法 | CIFAR10 Cross-Silo LP | CIFAR10 Cross-Silo FT1% | CIFAR10 Cross-Silo FT10% | CIFAR100 Cross-Silo LP | CIFAR100 Cross-Device LP |
|------|------|------|------|------|------|
| FedAlignUniform | 80.84 | 69.99 | 81.00 | 57.25 | 43.03 |
| FedX | 78.40 | 66.78 | 80.01 | 57.34 | 43.07 |
| FedDecorr | 80.13 | 69.09 | 80.33 | 57.25 | 44.74 |
| FedU2 | 81.01 | 69.62 | 81.01 | 57.40 | 42.90 |
| **SSD** | **81.32** | **70.74** | **81.67** | **57.38** | **45.21** |

SSD 在 cross-silo（K=10）和 cross-device（K=50）设置下均取得最佳性能。

### 消融实验

| 配置 | LP | FT 1% | FT 10% | $-\mathcal{L}_{\text{uniform}}$ (↑) |
|------|------|------|------|------|
| FedAlignUniform | 80.84 | 69.99 | 81.00 | 3.79 |
| + PD | 80.74 | 69.78 | 80.71 | 3.80 |
| + DSR | 81.05 | 69.77 | 81.15 | 3.81 |
| + DSR + PD (SSD) | **81.32** | **70.74** | **81.67** | **3.84** |

- 单独加 PD 几乎无效果
- 单独加 DSR 有轻微提升
- DSR + PD 组合产生显著的协同效应，均匀性提升最大

### 关键发现

1. **OOD泛化**: 在 CIFAR100→CIFAR10 和 TinyImageNet→CIFAR10 迁移实验中，SSD 均取得最高 LP 准确率（78.48% 和 80.00%）和最佳 effective rank（86.95 和 98.45）
2. **软分离 vs 硬分离**: 硬分离（HSD，完全独立子空间）虽然获得最高均匀性，但严重损害对齐性，导致下游性能差；SSD 在两者间取得平衡
3. **鲁棒性**: 不同 $\alpha$ 值和不同维度选择下性能稳定，说明方法对超参数不敏感
4. **投影器的重要性**: 移除投影器后加 DSR 可提升均匀性（3.72→3.77），但总体性能仍低于保留投影器的方案（76.14 vs 81.05）

## 亮点与洞察

- **问题定义精准**: 将联邦无监督学习中的均匀性问题分解为 intra-client 和 inter-client 两个层面，首次形式化了 inter-client 均匀性的重要性
- **设计简洁有效**: DSR 只需在初始阶段分配维度缩放向量，无额外通信开销，无隐私泄露风险
- **DSR+PD 的协同效应**: 单独使用效果有限，但组合使用时 PD 能将 DSR 在 embedding 层面的优化效果有效传递到 representation 层面
- **"软"与"硬"的权衡分析**: 提供了关于表征空间分离程度与下游性能关系的深入分析

## 局限与展望

- 仅在 CIFAR-10/100 和 TinyImageNet 上验证，缺乏大规模数据集（如 ImageNet）实验
- 缩放维度数量为 $\lfloor d/K \rfloor$，当客户端数 $K$ 很大时每个客户端分配的维度可能过少
- 仅考虑了 cross-silo 和 cross-device 两种标准设置，未探索更复杂的联邦场景
- PD 使用 KL 散度，是否有更好的蒸馏方式值得探索

## 相关工作与启发

- 继承了 FedAvg 框架的简洁性，在不改变聚合方式的前提下通过本地训练目标改善全局表征质量
- 与 FedDecorr（解相关）和 FedU2（球面高斯正则化）相比，SSD 直接针对 inter-client 均匀性，是更直接的优化路径
- 投影器蒸馏的思路可以推广到其他需要在投影空间优化但目标是 representation 质量的场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次明确提出并解决 inter-client uniformity 问题，DSR 设计新颖
- **实验充分度**: ⭐⭐⭐⭐ 多设置、消融、鲁棒性、OOD 泛化等实验全面，但数据集规模偏小
- **写作质量**: ⭐⭐⭐⭐⭐ 动机清晰、公式推导完整、图示直观
- **价值**: ⭐⭐⭐⭐ 为联邦无监督学习提供了新视角，方法简洁实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification](competitive_distillation_a_simple_learning_strategy_for_improving_visual_classif.md)
- [\[ECCV 2024\] Simple Unsupervised Knowledge Distillation With Space Similarity](../../ECCV2024/model_compression/simple_unsupervised_knowledge_distillation_with_space_similarity.md)
- [\[NeurIPS 2025\] Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation](../../NeurIPS2025/model_compression/rectifying_soft-label_entangled_bias_in_long-tailed_dataset_distillation.md)
- [\[ICML 2026\] FedSDR: Federated Self-Distillation with Rectification](../../ICML2026/model_compression/fedsdr_federated_self-distillation_with_rectification.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](knowledge_distillation_with_refined_logits.md)

</div>

<!-- RELATED:END -->
