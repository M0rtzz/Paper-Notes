---
title: >-
  [论文解读] FEAT: Federated Geometry-Aware Correction for Exemplar Replay under Continual Dynamic Heterogeneity
description: >-
  [CVPR 2026][federated continual learning] 提出 FEAT 方法解决联邦持续学习中回放样本利用不足的问题，通过几何结构对齐（基于 ETF 原型的角度蒸馏）和能量基几何校正（推理时去偏）缓解跨客户端异构和任务级数据不平衡。
tags:
  - CVPR 2026
  - federated continual learning
  - exemplar replay
  - equiangular tight frame
  - geometric correction
  - class imbalance
---

# FEAT: Federated Geometry-Aware Correction for Exemplar Replay under Continual Dynamic Heterogeneity

**会议**: CVPR 2026  
**arXiv**: [2604.08617](https://arxiv.org/abs/2604.08617)  
**代码**: 无  
**领域**: 其他  
**关键词**: federated continual learning, exemplar replay, equiangular tight frame, geometric correction, class imbalance

## 一句话总结

提出 FEAT 方法解决联邦持续学习中回放样本利用不足的问题，通过几何结构对齐（基于 ETF 原型的角度蒸馏）和能量基几何校正（推理时去偏）缓解跨客户端异构和任务级数据不平衡。

## 研究背景与动机

联邦持续学习（FCL）中，回放样本是缓解灾难性遗忘的主流策略。现有研究主要关注如何选择代表性样本（如 Re-Fed、FedCBDR），但忽略了如何有效利用这些有限样本。回放引入两个持续挑战：(1) 回放数据加剧跨客户端异构性；(2) 历史任务（尾类）与当前任务（头类）之间存在严重分布不平衡，导致尾类特征向头类方向漂移。

ETF 分类器虽然鼓励全局一致的类方向，但在持续动态异构下，尾类的跨客户端特征对齐仍明显弱于头类。

## 方法详解

### 整体框架

FEAT 包含两个正交于样本选择策略的模块，可无缝组合 Re-Fed+、FedCBDR 等现有回放方法。

### 关键设计

1. **几何结构对齐 (GSA)**：构建 batch 级别的特征余弦相似度矩阵 M_F 和对应 ETF 原型的相似度矩阵 M_P，通过行 softmax 归一化后计算 KL 散度。采用类平衡聚合——对每个类的样本独立平均后再跨类平均，确保尾类获得足够的几何监督。

2. **能量基几何校正 (EGC)**：推理时将特征投影到头类和尾类 ETF 子空间，计算各自的归一化能量。训练时通过 EMA 收集尾类样本的能量统计作为先验。推理时从特征中移除偏向头类子空间的分量，减少对多数类的过度自信，提升对少数类的敏感度。

3. **ETF 子空间分割**：将当前任务类别视为头类、历史任务类别视为尾类，利用 ETF 原型构建正交投影算子，分别计算头类和尾类子空间的能量。

### 损失函数 / 训练策略

L = L_CLS + λ · L_GSA。L_CLS 使用 ETF 原型与特征的相似度作为 logits 的交叉熵。每轮通信后服务器聚合模型参数和全局能量统计。EGC 仅在推理时应用，不增加训练成本。

## 实验关键数据

### 主实验

| 数据集 | 异构度 | FEAT | 之前SOTA | 提升 |
|--------|--------|------|---------|------|
| CIFAR-100 (α=0.1) | 高 | 最优 | 多种方法 | Top-1 一致提升 |
| Tiny-ImageNet | 中 | 最优 | 多种方法 | 一致提升 |
| Mini-ImageNet | 低 | 最优 | 多种方法 | 一致提升 |

### 消融实验

| 配置 | Top-1 准确率 | 说明 |
|------|-----------|------|
| Baseline (无 FEAT) | 较低 | 尾类漂移严重 |
| + GSA | 提升 | 跨客户端对齐改善 |
| + EGC | 进一步提升 | 推理去偏有效 |
| + 两者组合 | 最优 | 互补效果 |

### 关键发现

- GSA 有效改善尾类的跨客户端特征一致性
- EGC 的推理时去偏在不增加训练成本的前提下显著提升尾类准确率
- FEAT 正交于样本选择策略，与 Re-Fed+、FedCBDR 组合均有提升

## 亮点与洞察

- 关注回放样本"如何用"而非"如何选"，填补研究空白
- GSA 的类平衡 KL 蒸馏确保尾类获得公平的对齐监督
- EGC 作为推理时后处理零额外训练成本，实用性强
- 方法与回放策略正交的设计使其具有广泛适用性

## 局限与展望

- ETF 原型数量随类别数增长，可能面临高维空间的挑战
- EGC 的能量统计依赖于训练时收集的先验，分布漂移时可能不准确

## 评分

- 新颖性：⭐⭐⭐⭐ — 关注回放利用而非选择的新视角
- 技术深度：⭐⭐⭐⭐ — ETF+角度蒸馏+能量校正设计完整
- 实验充分度：⭐⭐⭐⭐ — 三数据集多异构度验证
- 实用价值：⭐⭐⭐⭐ — 即插即用，推理去偏零成本

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Expandable and Differentiable Dual Memories with Orthogonal Regularization for Exemplar-free Continual Learning](../../AAAI2026/others/expandable_and_differentiable_dual_memories_with_orthogonal_regularization_for_e.md)
- [\[CVPR 2026\] Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis](deconstructing_the_failure_of_ideal_noise_correction_a_three-pillar_diagnosis.md)
- [\[ICLR 2026\] Federated ADMM from Bayesian Duality](../../ICLR2026/others/federated_admm_from_bayesian_duality.md)
- [\[CVPR 2026\] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](polishing_the_sky_widefield_and_highdynamic_range.md)
- [\[CVPR 2025\] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](../../CVPR2025/others/subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)

</div>

<!-- RELATED:END -->
