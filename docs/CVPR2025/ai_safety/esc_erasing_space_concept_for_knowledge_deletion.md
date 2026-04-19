---
title: >-
  [论文解读] ESC: Erasing Space Concept for Knowledge Deletion
description: >-
  [CVPR 2025][AI安全][知识删除] 提出 ESC（Erasing Space Concept），通过 SVD 分解待遗忘数据的特征空间并移除主成分方向，实现训练无关的特征级知识删除，首次定义了"知识删除"（Knowledge Deletion）任务并提出 Knowledge Retention Score 评估特征级遗忘效果。
tags:
  - CVPR 2025
  - AI安全
  - 知识删除
  - 机器遗忘
  - SVD子空间
  - 特征级隐私
  - 训练无关方法
---

# ESC: Erasing Space Concept for Knowledge Deletion

**会议**: CVPR 2025  
**arXiv**: [2504.02199](https://arxiv.org/abs/2504.02199)  
**代码**: https://github.com/KU-VGI/ESC (有)  
**领域**: AI安全 / 机器遗忘  
**关键词**: 知识删除, 机器遗忘, SVD子空间, 特征级隐私, 训练无关方法

## 一句话总结

提出 ESC（Erasing Space Concept），通过 SVD 分解待遗忘数据的特征空间并移除主成分方向，实现训练无关的特征级知识删除，首次定义了"知识删除"（Knowledge Deletion）任务并提出 Knowledge Retention Score 评估特征级遗忘效果。

## 研究背景与动机

**领域现状**：机器遗忘（Machine Unlearning, MU）旨在从已训练模型中移除特定数据的影响。现有方法（如 Negative Gradient、Random Label、SalUn 等）通过端到端训练修改模型权重来实现遗忘。

**现有痛点**：现有 MU 方法存在严重的特征级知识残留问题——虽然分类头被有效修改导致预测改变，但特征提取器中的知识几乎未被触及。实验显示只需在"已遗忘"模型的冻结特征上训练一个新的线性探测器，就能恢复大量"已删除"知识（如 Figure 1 中线性探测恢复率接近原始模型）。

**核心矛盾**：现有方法使用基于 logit 的损失函数进行端到端训练，模型会找到"走捷径"的方式——只修改分类头就足以最小化 logit 上的遗忘损失，导致特征提取器中的知识完好无损。

**本文目标** 实现特征级别的知识删除，确保即使通过线性探测等方法也无法从特征中恢复被删除的知识。

**切入角度**：直接在特征空间中操作——用 SVD 找到待遗忘数据的主方向，然后投影到剩余子空间中消除这些方向的激活。无需训练即可完成。

**核心 idea**：用 SVD 分解待遗忘数据的特征矩阵，移除前 p% 主成分方向 = 训练无关的特征级知识删除。

## 方法详解

### 整体框架

将分类模型分为特征提取器 $h_\psi$ 和分类头 $g_\phi$。将待遗忘数据 $\mathcal{D}_f$ 通过 $h_\psi$ 得到特征矩阵 $Z_f$，对 $Z_f$ 做 SVD 得到主方向 $U$，移除前 $k$ 个主方向得到 $U_P$。推理时用 $U_P U_P^\top$ 投影所有特征。

### 关键设计

1. **ESC（训练无关版本）**:

    - 功能：无需训练即可删除特征空间中的遗忘知识
    - 核心思路：对待遗忘数据的特征矩阵 $Z_f = U \Sigma V^\top$ 做 SVD，移除前 $k = \frac{d}{100} \cdot p$ 个主成分方向得到 $U_P = U[k:]$。推理时特征投影为 $h_{\psi_P}(x) = U_P U_P^\top h_\psi(x)$。被移除的主方向正是待遗忘数据中方差最大的方向，包含最多判别信息
    - 设计动机：Figure 3 的 toy 实验显示，移除主成分后待遗忘类的特征与原始特征的余弦相似度从 >0.5 降到 <0.35，同时其他类特征几乎不受影响

2. **ESC-T（带训练版本）**:

    - 功能：通过可学习掩码实现更细粒度的知识删除，平衡遗忘与保留
    - 核心思路：不直接移除整个主方向，而是为每个主方向引入可学习掩码 $M_0$（初始化为 1）。用 Penalized Cross-Entropy Loss 训练掩码——当模型对遗忘数据预测正确时施加惩罚，驱动掩码关闭对应元素。最终得到精化主方向 $U_R$
    - 设计动机：ESC 的硬剪裁可能过度遗忘（移除整个方向而非方向中的关键元素），ESC-T 通过逐元素掩码实现"精准手术"式删除

3. **Knowledge Retention Score (KR)**:

    - 功能：评估特征级别的知识残留程度
    - 核心思路：冻结遗忘后模型的特征提取器，只训练新的线性探测器，测量对遗忘数据和保留数据的分类准确率。如果线性探测能恢复遗忘数据的高准确率，说明特征级知识仍然存在
    - 设计动机：现有评估（准确率、MIA）只关注输出层，无法检测特征中的残留知识

### 损失函数 / 训练策略

ESC 完全无需训练。ESC-T 使用 Penalized Cross-Entropy Loss：$\mathcal{L}_{PCE} = -\sum_c \hat{y}_c \log(1 - p_c)$，当模型对遗忘类预测正确时产生高损失，驱动掩码关闭相关特征。只训练掩码参数，backbone 冻结。

## 实验关键数据

### 主实验

CIFAR-10 知识删除对比（All-CNN）：

| 方法 | $D_f$↓ | $D_r$↑ | $D_{ft}$↓ | HM↑ | MIA | KR-$D_f$↓ |
|------|--------|--------|-----------|-----|-----|-----------|
| Original | 98.42 | 98.29 | 85.90 | 3.11 | 57.68 | 98.40 |
| Retrain | 0.00 | 96.93 | 0.00 | 98.44 | 50.06 | 41.28 |
| SalUn | 0.00 | 98.86 | 0.00 | 99.43 | 56.42 | **62.03** |
| **ESC** | 9.46 | 96.52 | 10.73 | 93.43 | 53.02 | **10.21** |
| **ESC-T** | **0.00** | **97.23** | **0.00** | **98.60** | 56.72 | **14.62** |

关键发现：SalUn 在输出层实现完美遗忘（$D_f$=0）但 KR 高达 62%（特征级知识仍在！），ESC/ESC-T 的 KR 降到 10-15%。

### 消融实验

| 配置 | 遗忘效果 | 保留效果 | 说明 |
|------|---------|---------|------|
| p=10% | 部分遗忘 | 保留很好 | 删除不够彻底 |
| p=30% | 良好遗忘 | 保留良好 | 最佳平衡点 |
| p=50% | 完全遗忘 | 保留下降 | 过度遗忘 |
| ESC-T | 完全遗忘 | 保留最优 | 可学习掩码精确控制 |

### 关键发现
- **现有 MU 方法在特征级别失败**：线性探测恢复率高达 80-96%，说明分类头面知识被删除但特征提取器中的知识完好
- **ESC 实现零训练的特征级删除**：仅用 SVD + 投影就能将 KR 从 98% 降到 10%，整个过程无需梯度计算
- **ESC-T 精化效果更好**：可学习掩码让遗忘和保留的平衡更优，HM 分数接近 Retrain 的理想值
- **适用于人脸场景**：在 CelebA-HQ 等人脸数据集上也有效，满足真实世界的隐私删除需求

## 亮点与洞察

- **揭露了现有机器遗忘的"虚假安全"**：通过 KR 指标证明了大量遗忘方法实际上只改变了分类头而非真正删除知识——这对整个机器遗忘社区是一个重要警示
- **SVD 的优雅应用**：用主成分方向代表"概念空间"，移除主方向等于移除概念——这个抽象既直觉合理又数学严谨
- **训练无关的速度优势**：ESC 只需一次 SVD 分解（秒级），比任何基于梯度的方法都快几个数量级

## 局限与展望

- **主成分方向可能在类间共享**：如果待遗忘类和保留类共享某些主方向（如背景特征），移除可能导致保留类性能下降
- **固定剪裁比例 p 需要调参**：不同数据集和模型需要不同的 p 值
- **仅在分类任务上验证**：生成模型（如扩散模型）的特征级遗忘是更大挑战
- **KR 指标依赖线性探测**：非线性探测可能恢复更多信息

## 相关工作与启发

- **vs SalUn**: SalUn 用显著性图引导梯度，在输出层实现完美遗忘但 KR 高达 62%。ESC 从特征空间直接操作，KR 降到 10%
- **vs ℓ1-sparse**: ℓ1 稀疏方法也能降低 $D_f$ 到 0，但保留准确率下降较大（89.95% vs ESC-T 97.23%）
- **vs Retrain**: ESC-T的 KR 虽低于 Retrain（14.6% vs 41.3%）但 HM 分数相近，说明在特征级甚至比从头训练删除得更彻底

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次定义特征级知识删除问题，KR 指标揭示了领域盲点
- 实验充分度: ⭐⭐⭐⭐ 多数据集多模型，KR 分析深入，但缺少大规模实验
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，但符号繁多
- 价值: ⭐⭐⭐⭐⭐ 对机器遗忘社区有方向性影响，KR 指标可能成为标准评估工具

<!-- RELATED:START -->

## 相关论文

- [Causally Reliable Concept Bottleneck Models](../../NeurIPS2025/ai_safety/causally_reliable_concept_bottleneck_models.md)
- [Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning](geometric_knowledge-guided_localized_global_distribution_alignment_for_federated.md)
- [Activation Space Interventions Can Be Transferred Between Large Language Models](../../ICML2025/ai_safety/activation_space_interventions_can_be_transferred_between_large_language_models.md)
- [Preserving Task-Relevant Information Under Linear Concept Removal](../../NeurIPS2025/ai_safety/preserving_task-relevant_information_under_linear_concept_removal.md)
- [The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples](../../NeurIPS2025/ai_safety/the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)

<!-- RELATED:END -->
