---
title: >-
  [论文解读] Retraining with Predicted Hard Labels Provably Increases Model Accuracy
description: >-
  [ICML 2025][AI安全][label noise] 在噪声标签下，用模型自身预测的硬标签（0/1标签）对训练集重新标注并重训练，可以**理论上可证明地**提升模型准确率；进一步提出 consensus-based retraining（仅对预测标签与给定标签一致的样本重训练），在 label DP 场景下无额外隐私代价即可大幅提升性能。
tags:
  - "ICML 2025"
  - "AI安全"
  - "label noise"
  - "label differential privacy"
  - "retraining"
  - "noisy labels"
  - "self-training"
  - "theoretical analysis"
---

# Retraining with Predicted Hard Labels Provably Increases Model Accuracy

**会议**: ICML 2025  
**arXiv**: [2406.11206](https://arxiv.org/abs/2406.11206)  
**代码**: 无  
**领域**: AI安全  
**关键词**: label noise, label differential privacy, retraining, noisy labels, self-training, theoretical analysis  

## 一句话总结

在噪声标签下，用模型自身预测的硬标签（0/1标签）对训练集重新标注并重训练，可以**理论上可证明地**提升模型准确率；进一步提出 consensus-based retraining（仅对预测标签与给定标签一致的样本重训练），在 label DP 场景下无额外隐私代价即可大幅提升性能。

## 研究背景与动机

标签差分隐私（Label DP）要求在监督学习中保护标签隐私，通常通过向标签注入噪声实现。然而注入噪声后标签质量大幅下降，导致模型性能显著退化。已有大量关于自训练（self-training）和自蒸馏（self-distillation）利用模型自身预测的工作，但缺乏理论分析证明"用模型自身预测的**硬标签**重训练能在标签噪声下提升准确率"。

作者的核心观察是：当类别间有足够的分离度时，模型虽然在噪声标签上训练，但对远离决策边界的样本仍能做出正确预测。因此模型在训练集上的预测准确率可以显著高于给定噪声标签的准确率，用这些更准确的预测标签重训练就能提升性能。文中 Figure 1 直观展示了这一点——类别分离度大时重训练有效（测试准确率从 89% 提升到 97.67%），分离度小时则无改善。

## 方法详解

### 整体框架

方法分为两个层次：

1. **Full Retraining（完全重训练）**：用初始模型 $\hat{\boldsymbol{\theta}}_0$ 对所有 $n$ 个训练样本预测硬标签 $\tilde{y}_j = \text{sign}(\langle \boldsymbol{x}_j, \hat{\boldsymbol{\theta}}_0 \rangle)$，然后用 $\{(\boldsymbol{x}_j, \tilde{y}_j)\}_{j=1}^n$ 重新训练模型。

2. **Consensus-based Retraining（共识重训练）**：定义共识集 $\mathcal{S}_{\text{cons}} = \{j \mid \tilde{y}_j = \hat{y}_j\}$，即预测标签与给定噪声标签一致的样本子集。仅用共识集 $\{(\boldsymbol{x}_j, \tilde{y}_j)\}_{j \in \mathcal{S}_{\text{cons}}}$ 重训练。

核心直觉：共识集虽然更小，但标签准确率远高于全集。例如 CIFAR-100 在 $\epsilon=3$ 时，共识集仅占全集 11%，但其标签准确率高达 76.09%，而全集预测标签准确率仅 24.90%。

### 关键设计一：理论分析框架——带正 margin 的高斯混合模型

理论分析在线性可分二分类设定下展开。数据生成模型为带正 margin 的高斯混合模型：

$$\boldsymbol{x} = y(1+u)\boldsymbol{\mu} + \boldsymbol{\Sigma}^{1/2}\boldsymbol{z}$$

其中 $y \in \{+1, -1\}$ 是真实标签，$u > 0$ 是 sub-gaussian 随机变量（保证正 margin），$\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_d)$，$\gamma = \|\boldsymbol{\mu}\|_{\ell_2}$ 表示类间分离度。给定的噪声标签以概率 $p < 1/2$ 独立翻转：$\hat{y}_i = y_i$（概率 $1-p$）或 $\hat{y}_i = -y_i$（概率 $p$）。

初始分类器为 $\hat{\boldsymbol{\theta}}_0 = \frac{1}{n}\sum_i \hat{y}_i \boldsymbol{x}_i$，重训练分类器为 $\hat{\boldsymbol{\theta}}_1 = \frac{1}{n}\sum_i \tilde{y}_i \boldsymbol{x}_i$。

### 关键设计二：重训练提升准确率的理论保证

**核心定理（Remark 4.10）**：当 $p$ 接近 $1/2$（噪声很大），且样本数满足：

$$\frac{\lambda_{\min}^2 d}{\gamma^4(1-2p)^2} \log\frac{\lambda_{\min}^2 d}{\gamma^4(1-2p)^2} \lesssim n \lesssim \frac{\lambda_{\min}^2 d^2}{\gamma^4(1-2p)^2}$$

则重训练分类器的准确率严格优于初始分类器，即 $\text{acc}(\hat{\boldsymbol{\theta}}_1) > \text{acc}(\hat{\boldsymbol{\theta}}_0)$。

关键机理：初始分类器误差下界中指数项含 $(1-2p)^2$，而重训练误差上界中指数项的有效噪声率 $q' = \exp(-\frac{n(1-2p)\gamma^2}{40\lambda_{\max}})$，当 $n$ 足够大时 $q' \ll p$，因此重训练的有效噪声远小于原始噪声。直观理解：模型预测标签的准确率高于给定噪声标签的准确率（尤其对远离边界的样本），因此重训练等价于在更低噪声下学习。

证明的技术挑战在于预测标签 $\tilde{y}_i$ 依赖于整个训练集（非独立），且噪声是非均匀的、样本相关的。作者通过构造"dummy labels"来解耦依赖关系。

### 关键设计三：Label DP 下的零额外隐私代价

由于重训练仅使用模型自身的预测标签和已发布的噪声标签，不访问任何原始真实标签，因此是对 label DP 机制的**后处理**（post-processing），根据 DP 的后处理性质，不产生额外隐私代价。这意味着 consensus-based retraining 可以叠加在任意 label DP 算法之上。

## 实验关键数据

### 表1：CIFAR-10 测试集准确率（ResNet-18）

| $\epsilon$ | Baseline | Full RT | Consensus RT |
|:---:|:---:|:---:|:---:|
| 1 | 57.78 ± 1.13 | 60.07 ± 0.63 | **63.84 ± 0.56** |
| 2 | 79.06 ± 0.59 | 81.34 ± 0.40 | **83.31 ± 0.28** |
| 3 | 85.18 ± 0.50 | 86.67 ± 0.28 | **87.67 ± 0.28** |

### 表2：CIFAR-100 测试集准确率（ResNet-18）

| $\epsilon$ | Baseline | Full RT | Consensus RT |
|:---:|:---:|:---:|:---:|
| 3 | 23.53 ± 1.01 | 24.42 ± 1.22 | **29.98 ± 1.11** |
| 4 | 44.53 ± 0.81 | 46.99 ± 0.66 | **51.30 ± 0.98** |
| 5 | 55.75 ± 0.36 | 56.98 ± 0.43 | **59.47 ± 0.26** |

### 表3：共识集的标签过滤效果（CIFAR-100）

| $\epsilon$ | 预测标签(全集)准确率 | 给定标签(全集)准确率 | 预测标签(共识集)准确率 |
|:---:|:---:|:---:|:---:|
| 3 | 24.90% | 22.35% | **76.09%** |
| 4 | 50.85% | 46.32% | **91.59%** |
| 5 | 66.51% | 68.09% | **94.83%** |

### 表4：AG News Subset（小型 BERT）

| $\epsilon$ | Baseline | Full RT | Consensus RT |
|:---:|:---:|:---:|:---:|
| 0.3 | 54.54 ± 0.97 | 60.03 ± 2.90 | **65.91 ± 1.93** |
| 0.5 | 69.21 ± 0.31 | 75.63 ± 1.08 | **80.95 ± 1.47** |
| 0.8 | 79.10 ± 1.43 | 82.19 ± 1.54 | **84.26 ± 1.03** |

AG News 上 $\epsilon=0.5$ 时 consensus RT 比 baseline 提升 11.7%，共识集仅占训练集 32%。

## 关键发现

1. **重训练在理论上可证明地提升准确率**：在线性可分设定下，首次证明用预测硬标签重训练可降低 population error，且噪声越大（$p$ 越接近 $1/2$）或分离度越大（$\gamma$ 越大）时收益越明显。
2. **共识集是极其高效的过滤机制**：共识集虽然只占训练集的小比例（CIFAR-100 在 $\epsilon=3$ 时仅 11%），但其标签准确率可从约 25% 跃升至 76%，这是 consensus RT 远优于 full RT 的根本原因。
3. **对噪声鲁棒方法仍有增益**：即使在初始训练已使用 forward correction 或 symmetric CE 等噪声鲁棒技术后，consensus RT 仍能进一步提升性能。
4. **跨模态、跨架构通用**：方法在视觉（CIFAR-10/100、DomainNet / ResNet-18/34/50）和 NLP（AG News / BERT）上均有效。

## 亮点与洞察

- **极简但有效**：consensus-based retraining 不需要任何额外的无监督学习或半监督方法，仅靠模型自身预测与给定标签的一致性筛选，就能获得显著提升。这种"less is more"的思想值得关注。
- **理论与实践的桥梁**：线性分析虽然简化，但准确捕捉了核心机制——有效噪声率从 $p$ 降低到 $q' \approx \exp(-\Theta(n))$，这个指数级降噪是重训练有效的关键。
- **共识集的"双重过滤"效应**：模型在远离边界的样本上预测更准确，而共识筛选进一步滤除了边界附近的不确定样本；两重过滤叠加产生一个小而纯净的子集。
- **与 self-training 的关键区别**：self-training 在半监督设定下基于模型置信度选样本；本文在全监督噪声标签设定下基于预测-给定标签一致性选样本，后者效果更好（附录J验证）。

## 局限性

1. **理论仅覆盖 full retraining**：共识重训练虽然实验效果最好，但目前缺乏理论分析。
2. **理论限于均匀标签噪声**：实际中标签噪声往往是类别相关（instance-dependent）的，现有理论未覆盖。
3. **理论限于线性可分设定**：对深度网络、非线性分类的泛化尚无理论保证。
4. **样本数上界约束**：$n \lesssim d^2/(1-2p)^2$ 的上界可能是分析的 artifact，但在当前证明框架下无法去除。
5. **未在大规模模型和数据集上验证**：实验规模有限（ResNet-18/34，CIFAR-10/100），未覆盖大模型场景。

## 相关工作与启发

- **Self-training** (Scudder 1965; Lee et al. 2013)：半监督设定下用模型预测标签迭代训练，但现有理论不涉及噪声标签场景。本文是全监督噪声标签设定下的首个理论结果。
- **Self-distillation** (Furlanello et al. 2018; Das & Sanghavi 2023)：用教师模型的**软标签**训练学生模型。本文关注**硬标签**且不使用温度参数，分析更直接。
- **Label DP** (Ghazi et al. 2021)：标签隐私保护的核心方法。本文的 consensus RT 可作为任意 label DP 算法的即插即用后处理模块。
- **启发**：这种"预测-给定一致性"的筛选思想可以推广到其他噪声学习场景（如众包标注、弱监督学习），且不限于 DP 场景。

## 评分

| 维度 | 分数 | 说明 |
|:---|:---:|:---|
| 新颖性 | 7 | 首次理论证明硬标签重训练在噪声下有效；consensus filtering 虽直觉自然但此前未被理论或系统实验验证 |
| 技术深度 | 8 | 处理预测标签的非独立、非均匀噪声的理论分析有较高难度；dummy labels 解耦技巧优雅 |
| 实验充分性 | 7 | 覆盖视觉和 NLP、多种架构和 DP 参数，但规模有限，缺少大模型实验 |
| 表达清晰度 | 8 | 理论与实验组织清晰，直觉解释到位（Figure 1 很好），核心观察一目了然 |
| 实用价值 | 8 | 方法极简、无额外隐私代价、可即插即用，对 label DP 实践有直接推动意义 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Private Model Personalization Revisited](private_model_personalization_revisited.md)
- [\[ICML 2025\] Understanding Model Ensemble in Transferable Adversarial Attack](understanding_model_ensemble_in_transferable_adversarial_attack.md)
- [\[ICML 2025\] Rethinking the Bias of Foundation Model under Long-tailed Distribution](rethinking_the_bias_of_foundation_model_under_long-tailed_distribution.md)
- [\[ICML 2025\] Relative Error Fair Clustering in the Weak-Strong Oracle Model](relative_error_fair_clustering_in_the_weak-strong_oracle_model.md)
- [\[ICML 2025\] De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks](de-antifake_rethinking_the_protective_perturbations_against_voice_cloning_attack.md)

</div>

<!-- RELATED:END -->
