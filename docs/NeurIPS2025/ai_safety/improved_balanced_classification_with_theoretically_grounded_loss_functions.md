---
title: >-
  [论文解读] Improved Balanced Classification with Theoretically Grounded Loss Functions
description: >-
  [NeurIPS 2025][AI安全][平衡分类损失] 提出两个理论驱动的代理损失函数族——广义Logit调整(GLA)损失和广义类感知加权(GCA)损失，为类别不平衡下的多类分类提供更强的理论保证和实证性能。
tags:
  - NeurIPS 2025
  - AI安全
  - 平衡分类损失
  - 代理损失
  - H-一致性
  - Logit调整
  - 类感知加权
---

# Improved Balanced Classification with Theoretically Grounded Loss Functions

**会议**: NeurIPS 2025  
**arXiv**: [2512.23947](https://arxiv.org/abs/2512.23947)  
**作者**: Corinna Cortes, Mehryar Mohri, Yutao Zhong
**代码**: 无  
**领域**: 机器学习理论 / 类别不平衡分类  
**关键词**: 平衡分类损失, 代理损失, H-一致性, Logit调整, 类感知加权

## 一句话总结

提出两个理论驱动的代理损失函数族——广义Logit调整(GLA)损失和广义类感知加权(GCA)损失，为类别不平衡下的多类分类提供更强的理论保证和实证性能。

## 研究背景与动机

在多类分类中，类别不平衡是一个普遍存在的问题。平衡分类损失通过对所有类别赋予相同重要性来促进公平性，确保少数类不被忽视。然而，直接最小化平衡分类损失通常是不可解的，因此设计有效的代理损失成为核心问题。

现有代理损失方法存在以下局限：

**标准类加权损失**：简单地按类频率的倒数缩放损失，理论保证有限

**Logit调整(LA)损失**：根据类先验概率移动logit，在标准交叉熵家族中有效，但仅对完备（无界）假设集具有H-一致性

**理论保证不足**：现有方法的H-一致性界依赖类概率最小值 $p_{\min}$ 的倒数，在高度不平衡设置下退化严重

本文的核心动机是：能否设计出在更广泛条件下具有更强理论保证的代理损失，同时保持良好的实证性能？

## 方法详解

### 整体框架

本文提出两个代理损失函数族，均扩展到广义交叉熵损失家族：

1. **GLA (Generalized Logit-Adjusted)**: 将Logit调整推广到广义交叉熵家族
2. **GCA (Generalized Class-Aware weighted)**: 引入类别相关的置信度边距，扩展标准类加权损失

### 关键设计

#### 1. 广义Logit调整损失 (GLA)

标准LA损失通过对每个类别 $c$ 添加偏移量 $\log p_c$ 来调整logit：

$$\ell_{\text{LA}}(h, x, y) = -\log \frac{e^{h_y(x) + \log p_y}}{\sum_c e^{h_c(x) + \log p_c}}$$

GLA将此推广到广义交叉熵损失家族 $\Phi$，允许使用更一般的凸函数 $\Phi$ 替代 $\log$：

$$\ell_{\text{GLA}}(h, x, y) = \Phi\left(\frac{e^{h_y(x) + \tau_y}}{\sum_c e^{h_c(x) + \tau_c}}\right)$$

其中偏移量 $\tau_c$ 可以取不同于 $\log p_c$ 的值，并可在广义交叉熵框架下统一分析。

**理论性质**：
- Bayes一致性：GLA损失是Bayes一致的
- H-一致性：仅对完备（无界）假设集成立
- H-一致性界：依赖 $1/p_{\min}$，在不平衡设置下不够紧

#### 2. 广义类感知加权损失 (GCA)

GCA损失在标准类加权损失基础上引入两个关键创新：

**a. 类别相关的置信度边距**：为每个类别设定不同的置信度边距 $m_c$，而非统一阈值：

$$\ell_{\text{GCA}}(h, x, y) = \frac{1}{p_y} \Phi\left(\frac{e^{h_y(x) - m_y}}{\sum_c e^{h_c(x) - m_c}}\right)$$

这些边距可以通过理论分析校准，为少数类设置更宽松的边距，为多数类设置更严格的边距。

**b. 广义交叉熵扩展**：与GLA类似，扩展到更广泛的凸函数族。

**理论性质**：
- H-一致性：对任何有界或完备的假设集成立（比GLA更广泛）
- H-一致性界：依赖 $1/\sqrt{p_{\min}}$，比GLA的 $1/p_{\min}$ 更优
- 在高度不平衡设置下提供显著更强的理论保证

#### 3. 理论分析框架

核心理论工具是H-一致性界(H-consistency bounds)，用于度量代理损失最小化器与平衡分类误差之间的差距：

| 损失类型 | Bayes一致性 | H-一致性条件 | 界的依赖关系 |
|---------|-----------|-------------|------------|
| 标准类加权 | 是 | 有界/完备 | 基线 |
| LA (原始) | 是 | 仅完备 | $1/p_{\min}$ |
| GLA (本文) | 是 | 仅完备 | $\geq 1/p_{\min}$ |
| GCA (本文) | 是 | 有界/完备 | $1/\sqrt{p_{\min}}$ |

### 损失函数 / 训练策略

GCA损失的边距校准策略：
- 根据类别频率 $p_c$ 设置边距 $m_c$
- 少数类获得更大边距（降低置信度要求）
- 边距选择需保证H-一致性界最优
- 可通过交叉验证在验证集上微调

## 实验关键数据

### 主实验

#### 标准不平衡分类基准

| 方法 | CIFAR-10-LT (IF=100) | CIFAR-100-LT (IF=100) | ImageNet-LT | 理论保证 |
|------|---------------------|----------------------|-------------|---------|
| 标准类加权 | 基线 | 基线 | 基线 | 有界/完备 |
| LA (原始) | 高于基线 | 高于基线 | 高于基线 | 仅完备 |
| **GLA (本文)** | **通常最优** | **通常最优** | **通常最优** | 仅完备 |
| **GCA (本文)** | 接近最优 | 接近最优 | 接近最优 | 有界/完备 |

其中 IF = Imbalance Factor，表示最大类与最小类的样本数之比。

#### 高度不平衡设置下的表现

| 方法 | IF=10 | IF=50 | IF=100 | IF=200 |
|------|-------|-------|--------|--------|
| 标准类加权 | 基线 | 基线 | 基线 | 基线 |
| LA 损失 | +较小 | +中等 | +中等 | +中等 |
| **GLA** | +较大 | +较大 | **最优** | 接近最优 |
| **GCA** | +中等 | +较大 | 接近最优 | **最优** |

关键观察：GLA在常见基准上通常略优，而GCA在高度不平衡设置(IF≥100)下展现出边际优势，这与理论分析一致——GCA的 $1/\sqrt{p_{\min}}$ 界在极端不平衡时更有利。

### 消融实验

#### 边距校准的影响

| GCA变体 | 无边距 | 统一边距 | 校准边距(理论) | 校准边距(验证集) |
|---------|-------|---------|--------------|----------------|
| 平衡准确率 | 基线 | +小 | +中等 | **+最大** |

#### 广义交叉熵函数 $\Phi$ 的选择

| $\Phi$ 选择 | GLA性能 | GCA性能 | 特点 |
|-------------|---------|---------|------|
| 标准log | 基线 | 基线 | 经典交叉熵 |
| 多项式 | 略高 | 略高 | 平滑梯度 |
| 指数 | 类似 | 类似 | 强调困难样本 |

### 关键发现

1. **GLA vs GCA的互补性**：GLA在常见基准上略优，GCA在极端不平衡下更好
2. **理论与实证的一致性**：H-一致性界的紧致程度与实际性能差异相对应
3. **类加权损失的基线强度**：简单类加权已是强基线，但GLA/GCA能进一步提升
4. **边距校准的重要性**：GCA的性能很大程度上依赖于正确的边距设置

## 亮点与洞察

1. **理论严谨性**：来自Google Research的团队(Cortes是SVM的共同发明人)，理论分析完备
2. **H-一致性界的改进**：GCA的 $1/\sqrt{p_{\min}}$ 界相比LA的 $1/p_{\min}$ 是根本性的改进
3. **实用贡献**：GLA和GCA可作为现有方法的直接替换(drop-in replacement)
4. **假设集的重要性**：论文揭示了Loss一致性分析中假设集(有界vs完备)的关键区别
5. **不平衡程度的自适应**：GCA通过边距校准自适应不同的不平衡程度

## 局限与展望

1. **长尾识别场景**：未测试在极端长尾场景(>1000类)下的表现
2. **与其他长尾方法的结合**：GLA/GCA能否与解耦训练、数据增强等方法互补
3. **大模型微调**：在预训练+微调范式下的表现尚未验证
4. **计算开销**：边距校准增加了超参数调优的成本
5. **理论到实践的差距**：理论最优的 $\Phi$ 选择与实际最优不完全一致

## 相关工作与启发

- **Logit Adjustment** (Menon et al., 2021)：LA损失的原始提出，本文将其推广
- **Class-Balanced Loss** (Cui et al., 2019)：经典类加权损失，本文的GCA是其理论增强版
- **H-consistency bounds** (Awasthi et al., 2022)：核心理论工具，本文在不平衡分类场景下做了深入应用
- **Focal Loss** (Lin et al., 2017)：另一种处理不平衡的方法，与本文方法正交
- **前序工作** (Mao, Mohri, Zhong, 2023-2024)：同一团队在多类抽象等问题上的理论工作

## 评分

- **新颖性**: ★★★★☆ — GCA的边距设计和更强理论保证
- **理论深度**: ★★★★★ — H-一致性分析严谨完备
- **实验充分度**: ★★★★☆ — 多尺度不平衡、多数据集验证
- **实用价值**: ★★★★☆ — 可直接替换现有损失函数
- **写作质量**: ★★★★★ — 来自理论社区顶级团队，行文规范

<!-- RELATED:START -->

## 相关论文

- [Enhancing Graph Classification Robustness with Singular Pooling](enhancing_graph_classification_robustness_with_singular_pooling.md)
- [Robust Graph Condensation via Classification Complexity Mitigation](robust_graph_condensation_via_classification_complexity_mitigation.md)
- [Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)
- [Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)
- [CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing](cryptomoe_privacy-preserving_and_scalable_mixture_of_experts_inference_via_balan.md)

<!-- RELATED:END -->
