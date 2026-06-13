---
title: >-
  [论文解读] Failure Cases Are Better Learned But Boundary Says Sorry: Facilitating Smooth Perception Change for Accuracy-Robustness Trade-Off in Adversarial Training
description: >-
  [ICCV 2025][AI安全][对抗训练] 揭示了对抗训练中一个反直觉现象——失败样本的模型感知变化反而比成功样本更小（即被"过度学习"），据此提出 Robust Perception Adversarial Training (RPAT)，通过鼓励感知随扰动平滑变化来缓解准确率-鲁棒性权衡问题。
tags:
  - "ICCV 2025"
  - "AI安全"
  - "对抗训练"
  - "准确率-鲁棒性权衡"
  - "感知一致性"
  - "决策边界"
  - "鲁棒感知"
---

# Failure Cases Are Better Learned But Boundary Says Sorry: Facilitating Smooth Perception Change for Accuracy-Robustness Trade-Off in Adversarial Training

**会议**: ICCV 2025  
**arXiv**: [2508.02186](https://arxiv.org/abs/2508.02186)  
**代码**: [https://github.com/FlaAI/RPAT](https://github.com/FlaAI/RPAT)  
**领域**: 其他  
**关键词**: 对抗训练, 准确率-鲁棒性权衡, 感知一致性, 决策边界, 鲁棒感知

## 一句话总结

揭示了对抗训练中一个反直觉现象——失败样本的模型感知变化反而比成功样本更小（即被"过度学习"），据此提出 Robust Perception Adversarial Training (RPAT)，通过鼓励感知随扰动平滑变化来缓解准确率-鲁棒性权衡问题。

## 研究背景与动机

对抗训练（AT）是训练鲁棒 DNN 最有效的方法，但存在固有的**准确率-鲁棒性权衡**：提升对抗鲁棒性往往以降低干净准确率为代价。

**已有共识**：AT 导致过于复杂的决策边界是权衡问题的根源，而现有工作普遍将此归因于模型对**困难对抗样本学习不足**，因此提出各种策略加强对困难样本的学习（如数据重加权 GAIRAT/MAIL、自适应扰动半径 MMA、非 one-hot 监督 TE/SOVR 等）。

**本文新发现**：通过概念验证实验，作者首次揭示一个反直觉事实——在 PGD-AT 训练的鲁棒模型中，防御失败样本（failure cases）的感知变化（良性样本 vs 对抗样本的 logits MSE）反而比成功样本更小。这意味着失败样本在当前 AT 目标下已被**过度学习**，而非学习不足。真正的问题在于决策边界的位置不当——过度追求感知一致性迫使模型将扰动视为噪声，忽略了扰动中可用于合理推动决策边界的信息。

## 方法详解

### 整体框架

在传统 AT 目标之上新增 Robust Perception 正则项，鼓励模型感知随输入扰动平滑线性变化，而非追求感知"不变"。整体 RPAT 目标结合了对抗样本的交叉熵损失和感知平滑正则化。

### 关键设计

1. **Robust Perception 目标定义**：对良性样本 $\mathbf{x}$ 及其对抗样本 $\mathbf{x}'$，对任意插值 $\alpha \in [0,1]$，要求：
    $\|h_{\bm{\theta}}(\mathbf{x}+\alpha \cdot \Delta) - h_{\bm{\theta}}(\mathbf{x})\| = \alpha \cdot \|h_{\bm{\theta}}(\mathbf{x}') - h_{\bm{\theta}}(\mathbf{x})\|$
   即模型感知应与扰动强度成正比地线性变化。

2. **理论支撑**（两个定理）：

    - *Theorem 1*（局部线性性）：Robust Perception 约束使 Hessian 矩阵的二次型趋于零 $\Delta^\top H_{h_\theta}(\mathbf{x}) \Delta \to 0$，即抑制高阶非线性效应，使感知主要沿扰动的线性项变化
    - *Theorem 2*（Lipschitz 正则化）：保证 Jacobian 在扰动方向上变化受限，全局 Lipschitz 常数增量被约束为微量 $\gamma$，从而平滑决策边界

3. **RPAT 损失函数**：使用 logits 作为模型感知表示，在良性样本 $\mathbf{x}$、插值样本 $\tilde{\mathbf{x}} = \mathbf{x} + \alpha \cdot \Delta$、对抗样本 $\hat{\mathbf{x}}'$ 之间施加 MSE 正则化：
    $\mathcal{L}^{\text{RPAT}} = \frac{1}{n}\sum_{i=1}^n \left(\mathcal{L}^{\text{CE}}(\mathbf{p}(\hat{\mathbf{x}}_i', \bm{\theta}), y_i) + \lambda \cdot \mathcal{L}^{\text{MSE}}\left(\frac{\mathbf{z}(\tilde{\mathbf{x}}_i) - \mathbf{z}(\mathbf{x}_i)}{\alpha} \bigg\| \frac{\mathbf{z}(\hat{\mathbf{x}}_i') - \mathbf{z}(\tilde{\mathbf{x}}_i)}{1-\alpha}\right)\right)$

### 损失函数 / 训练策略

- RPAT 可灵活叠加到任意 AT 基线上（PGD-AT、TRADES、MART、Consistency-AT）
- RPAT++ 将 RPAT 集成到当前 SOTA 方法 ReBAT 中
- 正则权重 $\lambda$ 和插值系数 $\alpha$ 是两个超参数
- 不显式区分成功/失败样本——成功样本本身也应满足 Robust Perception

## 实验关键数据

### 主实验 (表格)

RPAT 在 4 个 AT 基线上的提升（ResNet-18, $\ell_\infty$）：

| 数据集 | 方法 | Clean | PGD-20 | AA | Mean | NRR |
|--------|------|-------|--------|----|----- |-----|
| CIFAR-10 | PGD-AT | 82.92 | 50.61 | 46.74 | 64.83 | 59.78 |
| CIFAR-10 | +RPAT | **83.20** | **51.29** | **48.00** | **65.60** | **60.88** |
| CIFAR-10 | Consistency-AT | 83.42 | 51.96 | 47.72 | 65.57 | 60.71 |
| CIFAR-10 | +RPAT | **84.12** | **52.33** | **48.98** | **66.55** | **61.91** |
| CIFAR-100 | PGD-AT | 56.56 | 28.80 | 25.02 | 40.79 | 34.69 |
| CIFAR-100 | +RPAT | **58.22** | **29.16** | **24.88** | **41.55** | **34.86** |

RPAT++ vs 12 个 SOTA 方法（PreActResNet-18）：

| 方法 | Clean | AA | Mean | NRR |
|------|-------|-----|------|-----|
| ReBAT (NeurIPS'23) | 82.09 | 50.72 | 66.41 | 62.70 |
| **RPAT++ (Ours)** | **82.63** | **51.00** | **66.82** | **63.07** |
| CIFAR-100 ReBAT | 56.13 | 27.60 | 41.87 | 37.00 |
| CIFAR-100 **RPAT++** | **56.84** | **27.68** | **42.26** | **37.23** |

### 消融实验 (表格)

不同正则化度量的消融（$\ell_\infty$, 来自正文 Section 5.3 描述）：

| 度量 | 适用性 |
|------|--------|
| MSE | 基础版本，稳定有效 |
| KL | 也被证明有效 |
| Cosine | 可行替代 |

插值系数 $\alpha$ 和权重 $\lambda$ 的消融实验验证了方法对超参数的鲁棒性。

### 关键发现

- RPAT 在**所有** 4 个基线 × 3 个数据集 × 2 种范数设置下均带来一致提升
- 同时提升干净准确率和对抗鲁棒性（而非此消彼长）
- RPAT++ 在 CIFAR-10 和 CIFAR-100 上的 $\ell_\infty$ 设置超越了当前 12 个 SOTA（含 AWP、KD+SWA、GAIRAT、TE 等）
- 在 Tiny-ImageNet 上同样有效，说明方法可扩展到大规模数据集

## 亮点与洞察

- **核心贡献是认知更新**：推翻了"失败样本学习不足"的广泛共识，提出"过度学习导致决策边界位置不当"的新视角
- **概念验证实验设计精巧**：通过对比 clean training / random perturbation / PGD-AT 三种场景下成功/失败样本的感知变化，清晰展示了反直觉现象
- **理论完备**：从局部线性性和 Lipschitz 正则化两个角度证明了 Robust Perception 的合理性
- **即插即用**：RPAT 正则项可直接叠加到任意现有 AT 方法上

## 局限与展望

- 超参数 $\alpha$ 和 $\lambda$ 的选择仍需针对不同数据集调优
- 实验主要在相对小规模的数据集（CIFAR-10/100, Tiny-ImageNet）上，缺乏 ImageNet 规模验证
- 仅关注分类任务，对检测/分割等下游任务的适用性未验证
- "过度学习"现象的深层原因可以进一步分析
- 插值点只用了一个 $\alpha$，多点采样可能更精确但计算开销大

## 相关工作与启发

- TRADES 首先考虑平衡干净准确率与鲁棒性
- AWP 通过权重扰动平滑损失景观，ReBAT 将 AT 视为动态博弈
- GAIRAT/MAIL 聚焦于对困难样本重加权
- 本文的视角转换（从样本学习不足→决策边界位置不当）可能启发 AT 领域新方向

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次揭示 AT 中失败样本的"过度学习"现象，提供了完全不同于主流的新视角
- 实验充分度: ⭐⭐⭐⭐ 3 个数据集、3 种架构、4 个基线、12 个 SOTA、两种攻击范数
- 写作质量: ⭐⭐⭐⭐⭐ 故事讲述极其流畅，从发现→分析→解决的逻辑链完美
- 价值: ⭐⭐⭐⭐ 对 AT 社区有重要启发，且方法实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] On the Complexity-Faithfulness Trade-off of Gradient-Based Explanations](on_the_complexity-faithfulness_trade-off_of_gradient-based_explanations.md)
- [\[AAAI 2026\] Forest vs Tree: The (N, K) Trade-off in Reproducible ML Evaluation](../../AAAI2026/others/forest_vs_tree_the_n_k_trade-off_in_reproducible_ml_evaluation.md)
- [\[CVPR 2025\] TAET: Two-Stage Adversarial Equalization Training on Long-Tailed Distributions](../../CVPR2025/others/taet_two-stage_adversarial_equalization_training_on_long-tailed_distributions.md)
- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [\[ICCV 2025\] FixTalk: Taming Identity Leakage for High-Quality Talking Head Generation in Extreme Cases](fixtalk_taming_identity_leakage_for_high-quality_talking_head_generation_in_extr.md)

</div>

<!-- RELATED:END -->
