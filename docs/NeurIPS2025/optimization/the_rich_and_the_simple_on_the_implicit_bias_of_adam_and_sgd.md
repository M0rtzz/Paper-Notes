---
title: >-
  [论文解读] The Rich and the Simple: On the Implicit Bias of Adam and SGD
description: >-
  [NeurIPS 2025][优化][隐式偏置] 本文理论和实验证明，SGD训练的神经网络倾向于学习简单线性特征（简单性偏置），而Adam训练则产生更丰富的非线性特征，使模型更接近贝叶斯最优预测器，在分布偏移下泛化更好。
tags:
  - NeurIPS 2025
  - 优化
  - 隐式偏置
  - Adam
  - SGD
  - 简单性偏置
  - 特征学习
---

# The Rich and the Simple: On the Implicit Bias of Adam and SGD

**会议**: NeurIPS 2025  
**arXiv**: [2505.24022](https://arxiv.org/abs/2505.24022)  
**代码**: 暂无  
**领域**: 优化  
**关键词**: 隐式偏置, Adam, SGD, 简单性偏置, 特征学习

## 一句话总结

本文理论和实验证明，SGD训练的神经网络倾向于学习简单线性特征（简单性偏置），而Adam训练则产生更丰富的非线性特征，使模型更接近贝叶斯最优预测器，在分布偏移下泛化更好。

## 研究背景与动机

现代神经网络严重过参数化，训练过程中存在大量全局最优解。不同的优化算法会偏向不同的最优解，即所谓的"隐式偏置"(implicit bias)。已有大量工作研究了GD的隐式偏置，但对Adam——深度学习中最广泛使用的优化器——的隐式偏置理解仍然有限。

在实际应用中，SGD训练的神经网络已知会表现出**简单性偏置**(simplicity bias)：即倾向于找到简单的解，例如依赖数据的低维投影做预测。当数据中存在虚假特征(spurious features)时，简单性偏置会导致模型过度依赖这些简单但不具因果性的特征，从而在分布偏移下泛化效果差。

核心问题是：**Adam是否更能抵抗简单性偏置？如果是，这种差异的理论基础是什么？**

作者的切入角度是：在一个精心设计的高斯混合数据集上，分析两层ReLU网络分别用GD和Adam(含signGD)训练时的population gradient，推导出每个神经元的渐近收敛方向，从而精确刻画两种优化器学到的决策边界的形状差异。

## 方法详解

### 整体框架

研究采用两层齐次ReLU网络 $f(\mathbf{W};\mathbf{x}) = \mathbf{a}\sigma(\mathbf{W}\mathbf{x})$，其中最后一层固定，只训练第一层权重 $\mathbf{W}$。在一个特殊设计的高斯混合数据集上，分别分析GD和Adam的population gradient更新方向。

### 关键设计

1. **合成数据集构造**: 数据分为两类，第一维特征 $x_1$ 对两类都有区分力（线性可分），第二维 $x_2$ 只对正类的两个子簇有区分力（需要非线性边界）。贝叶斯最优预测器是分段线性的（非线性），这意味着只使用 $x_1$ 做线性预测是次优的。参数 $\omega$ 控制最优边界的非线性程度，$\kappa$ 控制各向异性。数据通过实现可达性假设(Assumption 1)确保最优边界过原点。

2. **Population Gradient闭式推导 (Proposition 2)**: 在correlation loss下，推导出每个神经元梯度的闭式表达式，涉及正态分布的PDF $\phi$ 和CDF $\Phi$。梯度包含三个方向 $\bar{\mu}_+, \bar{\mu}_-, \bar{\mu}_0$ 的加权组合，这些方向对应不同类别簇的归一化均值。

3. **GD的简单性偏置 (Theorem 1)**: 在gradient flow（连续时间GD）下，证明所有神经元的方向渐近收敛到 $a_k[1,0]$，即每个神经元只关注第一维，学到的决策边界是线性的。证明核心步骤是构造角度 $\theta_{k,t}$（神经元与 $[1,0]$ 方向的夹角），证明 $a_k \frac{d\cos\theta_{k,t}}{dt} > C\frac{(\sin\theta_{k,t})^2}{\|\mathbf{w}_{k,t}\|}$，从而 $\sin\theta_{k,t} \to 0$。

4. **Adam/signGD的丰富特征学习 (Theorem 2)**: 对于 $\beta_1=\beta_2=0$ 的Adam（即signGD），证明正类神经元($a_k>0$)根据初始化角度分别收敛到 $\frac{1}{\sqrt{2}}[1,1]$ 或 $\frac{1}{\sqrt{2}}[1,-1]$，负类神经元收敛到 $[-1,0]$。关键区别在于signGD的更新方向是梯度的符号 $[\text{sign}(a_k), \text{sign}(a_k\sin\theta_{k,t})]$，每一步在两个维度上都有非零更新，因此不会丢失第二维信息。

5. **Toy Data下的完整分析 (Theorem 4)**: 在 $\sigma \to 0$ 的简化设定下，完整刻画了GD、signGD和带动量Adam三种算法学到的神经元方向分布。带动量的Adam($\beta_1=\beta_2 \approx 1$)学到6种方向（包括一个额外的 $\frac{1}{\sqrt{s^2+1}}[s,\pm 1]$ 方向），产生更非线性的边界。

### 测试误差优势 (Theorem 3)

在 $\omega=\Theta(1)$ 的条件下，Adam学到的分段线性预测器的测试误差严格低于GD学到的线性预测器，从理论上证明了Adam在分布内和某些分布偏移下都能获得更好的泛化。

## 实验关键数据

### 主实验：子群体鲁棒性数据集 (Fig. 2 & Table 13)

| 数据集 | 指标(Worst-Group Acc.) | Adam | SGD | 提升 |
|---|---|---|---|---|
| Waterbirds | Worst-Group Acc. | ~93% | ~85% | +8% |
| CelebA | Worst-Group Acc. | ~85% | ~47% | +38% |
| MultiNLI | Worst-Group Acc. | ~73% | ~68% | +5% |
| CivilComments | Worst-Group Acc. | ~70% | ~60% | +10% |

在四个标准虚假相关基准数据集上，Adam在worst-group accuracy上全面超越SGD，特别是在CelebA上提升幅度巨大。

### Dominoes数据集实验 (Table 3)

| 配置 | Original Acc. | Core-Only Acc. | Decoded Acc. |
|---|---|---|---|
| SGD | 0.81±0.38 | 1.66±1.79 | 71.04±0.63 |
| Adam | **14.17±3.15** | **20.63±5.75** | **84.66±0.18** |

在MNIST-CIFAR数据集（95%虚假相关）上，Adam的原始worst-group准确率比SGD高17倍，decoded准确率高13.6个点，说明Adam确实学到了更多核心特征。

### Boolean Features数据集 (Table 4)

| 配置 | Test Acc. | Decoded Core Corr. | Decoded Spurious Corr. |
|---|---|---|---|
| SGD | 89.58±1.92 | 0.51±0.08 | 0.78±0.08 |
| Adam | **97.87±0.69** | **0.87±0.03** | **0.36±0.06** |

Adam显著提高了核心特征相关性(+0.36)，降低了虚假特征相关性(-0.42)。

### 关键发现

- GD的所有神经元都对齐到同一方向 $[\pm 1, 0]$，导致线性决策边界，只使用第一维特征
- Adam的神经元收敛到多个方向，产生分段线性决策边界，使用两个信号维度
- 在有限样本、logistic loss、Adam带动量等更实际的设定下，这些理论发现同样成立
- Adam的margin分布整体比SGD更大（Fig. 5）

## 亮点与洞察

- 提出了一个简单但富有洞察力的高斯混合数据设定，精确展示了GD和Adam隐式偏置的本质差异
- 挑战了"SGD在图像数据上通常优于Adam"的普遍共识——在存在虚假相关时，Adam因学习更丰富的特征而泛化更好
- signGD的"等尺度"更新机制（取符号）是避免简单性偏置的关键：它不会因为某个维度梯度小而忽略它

## 局限与展望

- 理论分析限于两层ReLU网络、固定外层权重、correlation loss等简化设定
- 简单性偏置并非总是有害的——对于分布内泛化，简单特征可能更好
- 没有分析AdamW（带权重衰减的Adam）的隐式偏置
- 更深网络和更复杂架构（如Transformer）上的分析待探索

## 相关工作与启发

- 与Kalimeris et al. (2019)「SGD先学线性再学非线性」和Shah et al. (2020)「简单性偏置」形成直接对比
- Kunstner et al. (2024) 从类别不平衡角度解释Adam的优势，本文从特征学习角度提供了互补的解释
- 为实践中选择优化器提供了理论指导：当数据可能存在虚假相关时，优先使用Adam

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统理论分析Adam vs GD在非线性模型上的隐式偏置差异
- 实验充分度: ⭐⭐⭐⭐ 合成+6个真实数据集，全面验证理论
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，理论-实验结合紧密
- 价值: ⭐⭐⭐⭐⭐ 对理解Adam的优势提供了深刻洞察，有直接的实践指导意义

<!-- RELATED:START -->

## 相关论文

- [The Implicit Bias of Structured State Space Models Can Be Poisoned With Clean Labels](the_implicit_bias_of_structured_state_space_models_can_be_poisoned_with_clean_la.md)
- [Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data](implicit_bias_of_spectral_descent_and_muon_on_multiclass_separable_data.md)
- [A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias](a_unified_stability_analysis_of_sam_vs_sgd_role_of_data_cohe.md)
- [In Search of Adam's Secret Sauce](in_search_of_adams_secret_sauce.md)
- [Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful](small_batch_size_training_for_language_models_when_vanilla_sgd_works_and_why_gra.md)

<!-- RELATED:END -->
