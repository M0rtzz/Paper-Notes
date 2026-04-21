---
title: >-
  [论文解读] Robust Watermarking on Gradient Boosting Decision Trees
description: >-
  [AI安全] 提出首个针对 GBDT 模型的鲁棒水印框架，通过 in-place 微调嵌入水印，设计了四种嵌入策略（Wrong Prediction Flip、Outlier Flip、Cluster Center Flip、Confidence Flip），实现高嵌入成功率、低精度损失和强抗微调鲁棒性。
tags:
  - AI安全
---

# Robust Watermarking on Gradient Boosting Decision Trees

- **会议**: AAAI 2026
- **arXiv**: [2511.09822](https://arxiv.org/abs/2511.09822)
- **代码**: [jc4303/gbdt_watermarking](https://github.com/jc4303/gbdt_watermarking)
- **领域**: ai_safety
- **关键词**: 水印, 梯度提升决策树, 知识产权保护, 模型安全, in-place微调

## 一句话总结

提出首个针对 GBDT 模型的鲁棒水印框架，通过 in-place 微调嵌入水印，设计了四种嵌入策略（Wrong Prediction Flip、Outlier Flip、Cluster Center Flip、Confidence Flip），实现高嵌入成功率、低精度损失和强抗微调鲁棒性。

## 背景与动机

- **GBDT 广泛使用**: 梯度提升决策树在结构化数据上表现优异，广泛应用于工业界和学术界，包括隐私敏感和医疗健康领域
- **水印研究缺失**: 神经网络的水印技术已有大量研究，但 GBDT 模型的水印方法严重不足
- **GBDT 水印的困难**:
    - 树是顺序构建的，每棵树依赖先前预测的梯度，修改已有树会引起级联破坏
    - 树模型不可微分，神经网络的水印方法无法直接迁移
    - 随机森林的直接树修改方法不适用于梯度提升模型，因为树之间不独立
- **现有工作局限**: Zhao et al. (KDD 2022) 针对提升树的水印方法仅关注脆弱完整性认证（弱水印），而非鲁棒嵌入

## 方法详解

### 1. In-place 更新机制

传统 GBDT 微调通过添加新树实现（如 XGBoost），但新增的树可以通过剪枝低贡献树轻松移除。本文提出 **in-place 更新**，直接修改已有树的内部参数而非添加新树，使水印更深度嵌入。

核心流程（Algorithm 1）：
- 对每轮 boosting 迭代 $m$ 和每个类别 $k$，计算伪残差构建微调数据集：

$$\mathcal{D}_{\text{fine}}' = \{(\mathbf{x}_i, r_{i,k} - p_{i,k})\}$$

- 计算新的梯度 $g_{i,k}'$ 和 Hessian $h_{i,k}'$
- 对树中每个非终端节点（深度优先遍历），重新计算增益和最优分裂 $S'$
- 若新分裂 $S' \neq S$，则重训该子树；否则仅更新受影响的叶节点预测值

### 2. 水印嵌入框架

给定候选数据集 $\mathcal{D}_{\text{cand}}$，识别候选样本集 $\mathcal{C}$，从中选择子集 $\mathcal{W} \subset \mathcal{C}$（大小为 $k$）进行水印嵌入。每个样本编码一位信息：修改标签为 1，保持原标签为 0。

水印标签设定为最自信的错误预测（排除真实标签和模型原始预测）：

$$y_i^{\text{wm}} = \underset{c \neq y_i,\; c \neq \hat{y}_i}{\text{argmax}}\; F_c(\mathbf{x}_i)$$

### 3. 四种水印嵌入策略

**Wrong Prediction Flip（错误预测翻转）**:
- 从 $\mathcal{D}_{\text{cand}}$ 中选择模型初始预测错误的样本，取置信度最低的 $n$ 个作为候选
- 水印标签设为第二高概率的错误类别（而非原始错误预测），避免与无关模型的"困难样本"混淆
- 优势：嵌入发生在本已易错的区域，对整体精度影响最小
- 局限：依赖错误预测数量，GBDT 对训练集通常很准确，候选不足

**Outlier Flip（离群点翻转）**:
- 选择特征空间中距离所有聚类中心最远的 $n$ 个正确预测样本：

$$\mathcal{C} = \left\{\underset{\mathbf{x}_i \in \mathcal{D}}{\text{argmax}_n}\; \min_{j \in \{1,\dots,m\}} \|\mathbf{x}_i - \boldsymbol{\mu}_j\| \right\}$$

- 使用 k-Means 聚类，选择使轮廓系数最大的聚类数 $m$
- 在稀疏区域嵌入水印，限制精度影响并增强抗微调鲁棒性

**Cluster Center Flip（聚类中心翻转）**:
- 对数据做聚类，选择每个聚类中最接近质心的样本作为水印候选
- 同时选取其 $l$ 个最近邻保持原始正确标签——形成局部"空洞"
- 通过正确标签邻居锚定决策边界，最小化对全局精度的影响
- 为对抗邻居的反向压力，将质心样本在微调数据中复制一次

**Confidence Flip（置信度翻转）**:
- 选择模型正确预测但置信度最低的 $n$ 个样本：

$$\mathcal{C} = \underset{\mathbf{x}_i \in \mathcal{D}}{\text{argmin}_n}\; F_{y_i}(\mathbf{x}_i)$$

- 这些样本位于决策边界附近，标签更容易翻转
- 嵌入对高置信区域影响最小，鲁棒性较好

### 4. 候选选择策略

从候选集 $\mathcal{C}$ 中最终选出 $k$ 个水印样本，提出两种策略：

- **最低置信度选择**: 选预测置信度最低的 $k$ 个样本，位于决策边界处更易嵌入
- **最大距离选择**: 最大化水印样本间的空间距离，类似 maximum diversity problem（NP-hard），采用贪心近似解

## 实验

### 实验设置

- **数据集**: Avila、Image Segmentation、Letter Recognition、optdigits、pendigits、Wine Quality
- **场景**: $\mathcal{D}_{\text{cand}} = \mathcal{D}_{\text{train}}$（内部水印）和 $\mathcal{D}_{\text{cand}} \neq \mathcal{D}_{\text{train}}$（事后水印）
- **水印比例**: $|\mathcal{W}|/|\mathcal{D}_{\text{train}}| \in \{0.001, 0.01, 0.1\}$
- **评估指标**: 嵌入成功率 $\mathcal{A}_{\text{wm}}$、调整后模型精度 $\mathcal{A}_{\text{model}}' = \mathcal{A}_{\text{model}} \cdot \mathcal{A}_{\text{wm}}$、微调鲁棒性

### 水印嵌入成功率（Table 1, $\mathcal{D}_{\text{cand}} = \mathcal{D}_{\text{train}}$）

| 方法 | ratio=0.001 | ratio=0.01 | ratio=0.1 |
|------|-------------|------------|-----------|
| Cluster (Conf) | 0.792 | 0.980 | 0.999 |
| Outlier (Conf) | 0.896 | 0.953 | 0.999 |
| Conf. (Conf) | 0.771 | 0.951 | 0.999 |
| Random (Conf) | 0.694 | 0.819 | 0.982 |

所有提出的方法平均成功率显著高于随机基线，尤其在较大水印比例下接近 100%。

### 调整后模型精度（Table 3, $\mathcal{D}_{\text{cand}} = \mathcal{D}_{\text{train}}$）

| 方法 | ratio=0.001 | ratio=0.01 | ratio=0.1 |
|------|-------------|------------|-----------|
| Cluster (Conf) | 0.699 | 0.880 | 0.872 |
| Outlier (Conf) | 0.802 | 0.854 | 0.869 |
| Conf. (Conf) | 0.681 | 0.854 | 0.880 |
| Random (Conf) | 0.603 | 0.729 | 0.877 |

Cluster Flip 和 Confidence Flip 在保持模型精度方面表现竞争力强，均优于随机基线。

### 微调鲁棒性（Table 5, $\mathcal{D}_{\text{cand}} = \mathcal{D}_{\text{train}}$）

| 方法 | ratio=0.001 | ratio=0.01 | ratio=0.1 |
|------|-------------|------------|-----------|
| Cluster (Conf) | 0.875 | 0.958 | 0.962 |
| Conf. (Conf) | 0.833 | 0.968 | 0.986 |
| Conf. (Dist) | 0.833 | 0.976 | 0.989 |
| Random (Conf) | 0.778 | 0.865 | 0.923 |

Confidence Flip 在鲁棒性方面整体略优，水印在进一步微调后仍能保持高检测率。

## 主要发现

1. **In-place 微调是关键**: 通过直接修改现有树结构而非添加新树，避免了水印被简单剪枝移除
2. **四种策略各有适用场景**: Wrong Prediction Flip 成功率最高但候选受限；Cluster Center Flip 精度保持最好；Confidence Flip 鲁棒性最强；Outlier Flip 在分布相似时表现稳定
3. **水印比例越大越稳定**: ratio=0.1 时各方法成功率和鲁棒性均接近满分
4. **候选数据来源影响效果**: 使用独立数据集 ($\mathcal{D}_{\text{cand}} \neq \mathcal{D}_{\text{train}}$) 避免梯度冲突，但内部数据集通过复制因子也可获得良好效果

## 亮点

- **首创性**: 首个针对 GBDT 的鲁棒水印框架，填补了树模型知识产权保护的重要空白
- **系统化设计**: 四种嵌入策略 + 两种候选选择策略形成完整的方法矩阵，不同场景有针对性方案
- **实用性强**: 支持内部水印和事后水印两种场景，适用于模型发布后的第三方保护
- **理论分析扎实**: 对梯度方向的分析表明了水印嵌入的理论约束，增强了方法的可解释性

## 局限性

- **仅验证分类任务**: 未探索回归任务或其他 GBDT 应用场景
- **聚类参数敏感**: Outlier Flip 和 Cluster Center Flip 的效果依赖聚类质量和参数选择
- **分布假设**: Outlier Flip 假设微调数据和候选数据分布相似，现实中不一定成立
- **Wrong Prediction Flip 受限**: 强模型几乎没有错误预测，极大限制了该策略的适用性
- **未讨论计算开销**: 缺少对 in-place 更新相比标准微调的时间/空间复杂度分析
- **对抗性攻击未考虑**: 仅评估了常规微调的鲁棒性，未考虑针对性的水印移除攻击

## 相关工作

- **神经网络水印**: Adi et al. (USENIX 2018) 提出后门水印，Uchida et al. (2017) 嵌入权重正则化水印
- **树模型水印**: Calzavara et al. (EDBT 2025) 针对随机森林直接修改树结构；Zhao et al. (KDD 2022) 提出提升树脆弱水印
- **GBDT 框架**: XGBoost (Chen & Guestrin, KDD 2016)、LightGBM (Ke et al., NeurIPS 2017)
- **鲁棒水印**: Pagnotta et al. (ACSAC 2024)、Yan et al. (USENIX 2023) 关注抗修改水印

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将鲁棒水印概念引入 GBDT，problem formulation 有开创意义
- **技术深度**: ⭐⭐⭐ — 四种策略设计合理但技术门槛不高，in-place 更新是关键创新
- **实验充分性**: ⭐⭐⭐⭐ — 多数据集、多比例、多场景的系统对比，但缺少与其他潜在方法的比较
- **实用价值**: ⭐⭐⭐⭐ — 直接解决 GBDT 模型的 IP 保护需求，工业界和法律场景有实际意义
- **总体推荐**: ⭐⭐⭐⭐ — 填补重要空白的 solid work，方法虽不复杂但系统完善

<!-- RELATED:START -->

## 相关论文

- [ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)
- [Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)

<!-- RELATED:END -->
