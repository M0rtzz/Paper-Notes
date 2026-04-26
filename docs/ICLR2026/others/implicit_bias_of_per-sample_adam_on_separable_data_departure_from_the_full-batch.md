---
title: >-
  [论文解读] Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime
description: >-
  [ICLR 2026][Adam] 首次证明mini-batch Adam的隐式偏差与full-batch不同：构造数据集使单样本Adam收敛到 $\ell_2$ 最大间隔分类器（而full-batch Adam收敛到 $\ell_\infty$），并通过AdamProxy刻画一般数据集上的数据自适应Mahalanobis范数间隔最大化行为。
tags:
  - ICLR 2026
  - Adam
  - 隐式偏差
  - 最大间隔
  - Mini-batch
  - Mahalanobis范数
---

# Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime

**会议**: ICLR 2026  
**arXiv**: [2510.26303](https://arxiv.org/abs/2510.26303)  
**代码**: 无  
**领域**: 优化理论  
**关键词**: Adam, 隐式偏差, 最大间隔, Mini-batch, Mahalanobis范数

## 一句话总结
首次证明mini-batch Adam的隐式偏差与full-batch不同：构造数据集使单样本Adam收敛到 $\ell_2$ 最大间隔分类器（而full-batch Adam收敛到 $\ell_\infty$），并通过AdamProxy刻画一般数据集上的数据自适应Mahalanobis范数间隔最大化行为。

## 研究背景与动机

**领域现状**：优化算法的隐式偏差决定了过参数化模型中哪个全局最优被选择。GD收敛到 $\ell_2$ 最大间隔解，full-batch Adam收敛到 $\ell_\infty$ 最大间隔解。SGD不改变GD的偏差（任何batch size都收敛到 $\ell_2$）。

**现有痛点**：现有Adam隐式偏差分析局限于full-batch设置。实际训练使用mini-batch，但不清楚mini-batch是否改变Adam的 $\ell_\infty$ 偏差。直觉上SGD不变→Adam也不变？

**核心矛盾**：实验发现mini-batch Adam（batch=1）在高斯数据上收敛到的方向与full-batch不同，更接近 $\ell_2$ 最大间隔！这与SGD的行为形成鲜明对比。

**切入角度**：通过分析单样本Adam（Inc-Adam）的epoch-wise更新的渐近形式，揭示预条件器跟踪的是单样本梯度平方的加权和（而非full-batch梯度的平方），导致自适应性质改变。

## 方法详解

### 整体框架
先在结构化数据(SR data)上证明Inc-Adam → $\ell_2$ 间隔；再对一般数据集引入AdamProxy代理算法（$\beta_2 \to 1$极限）→刻画为数据自适应Mahalanobis范数间隔最大化。

### 关键设计

1. **Epoch-wise近似 (Proposition 2.5)**:

    - Inc-Adam的epoch更新近似为 $w_{r+1}^0 - w_r^0 \approx -\eta \sum_i \frac{\sum_j \beta_1^{(i,j)} \nabla \mathcal{L}_j(w)}{\sqrt{\sum_j \beta_2^{(i,j)} \nabla \mathcal{L}_j(w)^2}}$
    - vs Full-batch Adam近似为SignGD: $w_{t+1} - w_t \approx -\eta \cdot \text{sign}(\nabla \mathcal{L}(w))$
    - 关键差异：Inc-Adam的预条件器是**单样本梯度平方的加权和**，不等于full-batch梯度的平方

2. **Scaled Rademacher数据上的精确结果 (Theorem 3.3)**:

    - SR数据: 每个样本各坐标绝对值相等（如 $x_i = (a_i, \pm a_i, \pm a_i, \pm a_i)$）
    - 此时Inc-Adam的坐标自适应性被消除→退化为加权归一化GD→收敛到 $\ell_2$ 最大间隔
    - 与full-batch Adam的 $\ell_\infty$ 偏差形成**极端对比**

3. **AdamProxy (一般数据集)**:

    - 取 $\beta_2 \to 1$ 极限得到简化更新：$\delta_t = \frac{\nabla \mathcal{L}(w)}{\sqrt{\sum_i \nabla \mathcal{L}_i(w)^2}}$
    - 收敛方向是Mahalanobis范数间隔最大化：$\max \min_i \frac{x_i^\top w}{\|w\|_M}$
    - 协方差矩阵 $M$ 由数据决定的对偶不动点方程确定

4. **Signum的不变性 (对比)**:

    - Signum(带动量的SignSGD)在任何batch size下仍收敛到 $\ell_\infty$
    - 原因：sign操作消除了预条件器中单样本vs全批的差异

## 实验关键数据

### SR数据验证

| 方法 | batch=full | batch=1 |
|------|-----------|---------|
| Adam | $\ell_\infty$ 间隔 | **$\ell_2$ 间隔** |
| SGD | $\ell_2$ 间隔 | $\ell_2$ 间隔 |
| Signum | $\ell_\infty$ 间隔 | $\ell_\infty$ 间隔 |

### 高斯数据验证

| 方法 | 与 $\ell_2$ 余弦 | 与 $\ell_\infty$ 余弦 |
|------|----------------|-------------------|
| Full-batch Adam | 低 | **1.0** |
| Inc-Adam | 高 | 低 |
| Adam (batch=1, replacement) | 高 | 低 |
| Adam (batch=1, reshuffling) | 高 | 低 |

### 关键发现
- Inc-Adam与有替换/reshuffling的batch=1 Adam行为一致→Inc-Adam是好的理论代理
- 对任何 $\beta_1 \leq \beta_2$，SR数据上结论成立→偏差变化不是特定超参的问题
- AdamProxy的Mahalanobis范数在某些数据上退化为 $\ell_2$、另一些退化为 $\ell_\infty$

## 亮点与洞察
- **反常识发现**：SGD的隐式偏差不依赖batch size→自然推测Adam也不依赖→但事实相反。这揭示了自适应方法的预条件器对采样方式敏感的本质。
- **预条件器的核心差异**：$\sum_i (\nabla \mathcal{L}_i)^2 \neq (\sum_i \nabla \mathcal{L}_i)^2$——单样本梯度的平方之和不等于全批梯度的平方。这个简单的数学事实导致了完全不同的隐式偏差。
- **Signum的鲁棒性**：sign操作使Signum对采样方式免疫——这可能部分解释了Signum/SignSGD在某些场景下的稳定性。

## 局限与展望
- AdamProxy分析需要假设方向收敛存在（Assumption 4.4）
- 仅分析了batch=1的极端情况，中间batch size的行为开放
- 仅限于线性分类+可分数据，深层网络的情况更复杂
- $\beta_2 \to 1$的极限可能与实践中 $\beta_2=0.999$ 有偏差

## 相关工作与启发
- **vs Zhang等人(2024)**: 他们证明full-batch Adam → $\ell_\infty$，本文证明mini-batch可以不同
- **vs Soudry等人(2018)**: GD → $\ell_2$ 不受batch size影响，但Adam受影响——自适应方法本质不同
- **启示**: 实践中Adam的行为可能同时受数据结构和batch size影响——不能简单从full-batch理论推断

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次发现并理论化Adam的batch-dependent隐式偏差
- 实验充分度: ⭐⭐⭐⭐ 结构化+随机数据验证，多种采样方案对比
- 写作质量: ⭐⭐⭐⭐ 数学严谨，直觉解释到位
- 价值: ⭐⭐⭐⭐⭐ 对理解Adam在实际训练中的行为有根本意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race](../../ACL2025/others/aligned_but_blind_implicit_bias.md)
- [\[ECCV 2024\] Rethinking Data Bias: Dataset Copyright Protection via Embedding Class-Wise Hidden Bias](../../ECCV2024/others/rethinking_data_bias_dataset_copyright_protection_via_embedding_class-wise_hidde.md)
- [\[ACL 2025\] Explicit and Implicit Data Augmentation for Social Event Detection](../../ACL2025/others/explicit_and_implicit_data_augmentation_for_social_event_detection.md)
- [\[ACL 2025\] Better Embeddings with Coupled Adam](../../ACL2025/others/better_embeddings_with_coupled_adam.md)
- [\[ECCV 2024\] Superpixel-Informed Implicit Neural Representation for Multi-Dimensional Data](../../ECCV2024/others/superpixel-informed_implicit_neural_representation_for_multi-dimensional_data.md)

<!-- RELATED:END -->
