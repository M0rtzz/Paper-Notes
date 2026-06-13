---
title: >-
  [论文解读] Matrix-Free Two-to-Infinity and One-to-Two Norms Estimation
description: >-
  [AAAI 2026][AI安全][matrix norm estimation] 提出 TwINEst 和 TwINEst++ 两种基于 Hutchinson 对角估计器的随机算法，用于在无矩阵 (matrix-free) 设定下高效估计 $\|A\|_{2\to\infty}$ 和 $\|A\|_{1\to 2}$ 范数，并提供了 oracle 复杂度理论保证，在 DNN 的 Jacobian 正则化（图像分类对抗鲁棒性）和推荐系统对抗攻击防御中展现了显著优势。
tags:
  - "AAAI 2026"
  - "AI安全"
  - "matrix norm estimation"
  - "two-to-infinity norm"
  - "Hutchinson estimator"
  - "Jacobian regularization"
  - "adversarial robustness"
---

# Matrix-Free Two-to-Infinity and One-to-Two Norms Estimation

**会议**: AAAI 2026  
**arXiv**: [2508.04444](https://arxiv.org/abs/2508.04444)  
**代码**: [github](https://github.com/fallnlove/TwoToInfinity)  
**领域**: AI Safety / Randomized Linear Algebra  
**关键词**: matrix norm estimation, two-to-infinity norm, Hutchinson estimator, Jacobian regularization, adversarial robustness

## 一句话总结

提出 TwINEst 和 TwINEst++ 两种基于 Hutchinson 对角估计器的随机算法，用于在无矩阵 (matrix-free) 设定下高效估计 $\|A\|_{2\to\infty}$ 和 $\|A\|_{1\to 2}$ 范数，并提供了 oracle 复杂度理论保证，在 DNN 的 Jacobian 正则化（图像分类对抗鲁棒性）和推荐系统对抗攻击防御中展现了显著优势。

## 研究背景与动机

在现代机器学习中，许多重要矩阵（如深度神经网络的 Jacobian）过大无法显式构建，但支持通过自动微分高效计算矩阵-向量乘积。这催生了 matrix-free 设定下的矩阵特性估计需求。$\|A\|_{2\to\infty}$ 范数等于矩阵各行 $\ell_2$ 范数的最大值，相比谱范数和 Frobenius 范数，它提供了更精细的逐行控制，特别适用于"高瘦矩阵"（如图像分类器的 Jacobian，输入维度远大于输出类别数）。

现有方法（自适应幂法）缺乏理论收敛保证，论文通过反例证明其在简单对角矩阵上就可能以 29.5% 的概率发散。核心挑战在于：如何仅通过矩阵-向量乘积随机估计 $\|A\|_{2\to\infty}$，同时提供可证明的 oracle 复杂度保证。

## 方法详解

### 整体框架

核心观察：$\|A\|_{2\to\infty}^2 = \max_{i \in [d]} \text{diag}(AA^\top)_i$，即 $\|A\|_{2\to\infty}$ 等价于 $AA^\top$ 对角线元素最大值的平方根。因此可以用 Hutchinson 对角估计器估计 $AA^\top$ 的对角线，找到最大行索引后精确计算该行的范数。

### 关键设计

**1. TwINEst 算法**

算法步骤：(a) 采样 $m$ 个 Rademacher 随机向量 $X^1, \dots, X^m \in \{-1,1\}^d$；(b) 对每个 $X^i$ 计算 $t_i = X^i \odot AA^\top X^i$；(c) 取平均得到对角线估计 $D = \frac{1}{m}\sum_i t_i$；(d) 找到最大索引 $j = \arg\max_i D_i$；(e) 精确计算 $L = \|A^\top e_j\|_2$。

关键技巧是 step (e)：不直接取 $\sqrt{\max_i D_i}$（方差大），而是用估计值找到候选行索引后精确计算该行范数，消除一层随机性。Oracle 复杂度：

$$m > \frac{8\log(2d/\delta)}{\Delta^2} \|AA^\top - \text{diag}(AA^\top)\|_{2\to\infty}^2$$

保证以 $1-\delta$ 概率返回精确值 $\|A\|_{2\to\infty}$，其中 $\Delta$ 是最大行范数与第二大行范数的平方差。

**2. TwINEst++ 算法（方差缩减版）**

借鉴 Hutch++ 的思想，将 $AA^\top$ 分解为低秩近似和残差：

$$AA^\top = \underbrace{AA^\top P}_{\text{低秩部分}} + \underbrace{AA^\top(I-P)}_{\text{残差}}$$

其中 $P = QQ^\top$ 是通过 QR 分解 $AA^\top S$（$S$ 为 Rademacher 矩阵）得到的正交投影。低秩部分的对角线精确计算，残差部分用 Hutchinson 估计。Oracle 复杂度改善至：

$$m > c \cdot \left(\frac{\sqrt{\log(2/\delta)}}{\Delta} \|A\|_F^2 + \log(1/\delta)\right)$$

从 $O(1/\Delta^2)$ 改善到 $O(1/\Delta)$，低秩矩阵上效果尤为显著。

**3. 自适应幂法的发散反例**

论文构造了一个简单反例：$A = \text{diag}(2, 1)$。初始随机向量经 $A$ 变换后，有 $\geq 29.5\%$ 概率第二分量绝对值大于第一分量，导致 dual$_\infty$ 操作选错行，此后所有迭代锁定在错误行上，最终输出 1 而非正确答案 2。

### 损失函数 / 训练策略

在深度学习应用中，使用 $\|A\|_{2\to\infty}$ 作为 Jacobian 的正则项：

$$\mathcal{L}(x,y) = \mathcal{L}_{\text{CE}}(f(x), y) + \lambda \cdot \|J_f(x)\|_{2\to\infty}^2$$

由于每次计算正则项代价等于一次反向传播，实际采用每 $k$ 次迭代更新一次正则项的策略。在推荐系统中，替换 UltraGCN 的权重衰减项为 $\|\hat{R}\|_{2\to\infty}^2$，用 TwINEst 近似计算。

## 实验关键数据

### 主实验

图像分类 Jacobian 正则化（WideResNet-16-10，3次试验平均）：

| 正则化方法 | CIFAR-100 Acc ↑ | FGSM ↑ | PGD ↑ | S.Rank ↓ | TinyImageNet Acc ↑ | FGSM ↑ | PGD ↑ | S.Rank ↓ |
|-----------|----------------|--------|-------|----------|-------------------|--------|-------|----------|
| 无正则化 | 75.5±0.2 | 24.4±0.6 | 11.7±0.3 | 32.0±1.1 | 57.8±1.3 | 30.4±0.3 | 20.2±0.1 | 30.9±4.3 |
| Frobenius | 75.7±0.5 | 23.5±0.2 | 13.3±0.2 | 31.6±0.2 | 58.6±0.3 | 31.1±0.2 | 20.7±0.5 | 27.8±0.9 |
| Spectral | 75.7±0.3 | 23.3±0.7 | 11.3±0.4 | 32.0±1.0 | 57.4±0.8 | 30.0±1.1 | 20.0±0.5 | 28.2±0.3 |
| Infinity | 75.8±0.4 | 23.7±0.7 | 11.1±0.2 | 30.7±1.2 | 57.1±0.7 | 29.6±1.4 | 19.7±1.0 | 28.8±0.9 |
| **Two-to-Infinity** | **77.3±0.1** | **26.9±0.5** | **14.5±0.5** | **18.3±0.8** | **59.6±0.9** | **31.0±1.0** | **23.4±0.7** | **24.9±0.3** |

推荐系统对抗鲁棒性（UltraGCN，NDCG@10）：

| 攻击强度 | 数据集 | Weight Decay | Factor $\|·\|_{2\to\infty}$ | Score $\|\hat{R}\|_{2\to\infty}$ (ours) |
|---------|-------|-------------|----------------------------|-----------------------------------------|
| 无攻击 | MovieLens-1M | ~0.35 | ~0.35 | ~0.35 |
| 中等攻击 | MovieLens-1M | ~0.28 | ~0.30 | ~0.32 |
| 强攻击 | MovieLens-1M | ~0.22 | ~0.25 | ~0.28 |

### 消融实验

合成矩阵上 $\|A\|_{2\to\infty}$ 估计精度对比（5000×5000 高斯矩阵，500次试验平均）：

| 方法 | $\Delta=10^{-2}$ (400次乘法相对误差) | $\Delta=10^{-1}$ (400次乘法相对误差) |
|-----|--------------------------------------|--------------------------------------|
| 自适应幂法 | ~0.15 (不收敛) | ~0.15 (不收敛) |
| Rademacher 平均 | ~0.02 | ~0.005 |
| TwINEst | ~0.005 | ~0.0 (已收敛) |
| TwINEst++ | ~0.002 | ~0.0 (最快收敛) |

WideResNet Jacobian 矩阵（$3072 \times 100$）上的收敛：TwINEst++ 约在 100 次矩阵-向量乘积时达到极低误差，与 Jacobian 的低秩特性一致。自适应幂法在 500 次乘积后仍未收敛。

### 关键发现

- CIFAR-100 上 $\|·\|_{2\to\infty}$ 正则化将测试准确率提升 1.8%（77.3 vs 75.5），PGD 对抗准确率提升 2.8%，stable rank 降低 42.8%（18.3 vs 32.0）
- 其他范数正则化（Frobenius、谱、$\infty$）改善有限甚至无效，凸显了 $\|·\|_{2\to\infty}$ 对 Jacobian 逐行控制的优势
- 推荐系统中 $\|\hat{R}\|_{2\to\infty}^2$ 正则化在所有三个数据集上均提升了中高强度攻击下的鲁棒性
- TwINEst++ 在低秩矩阵上显著优于 TwINEst，符合理论预期

## 亮点与洞察

- 反例证明自适应幂法可能发散是强有力的 motivation，展示了现有方法的根本缺陷
- TwINEst 的"先粗估再精算"策略（用 Hutchinson 找候选行，再精确计算行范数）极其简洁高效
- $O(1/\Delta^2) \to O(1/\Delta)$ 的复杂度改进对于 $\Delta$ 小的难实例意义重大
- $\|·\|_{2\to\infty}$ 范数正则化在图像分类上的效果出人意料地好，为 Jacobian 正则化提供了新的选择

## 局限与展望

- oracle 复杂度下界尚未建立，TwINEst/TwINEst++ 的最优性未知
- 深度学习实验仅在 WideResNet-16-10 上验证，缺乏对更多架构（ResNet、ViT 等）的评估
- 推荐系统实验中对抗攻击设定较简单（单步 FGSM 式攻击），更强的攻击下效果有待验证
- 正则化项的计算仍有一定开销（等价于若干次反向传播），虽然可以间隔 $k$ 步更新但引入了超参数

## 相关工作与启发

- **vs 自适应幂法 (Higham/Roth)**: 幂法缺乏理论保证且可能发散，TwINEst 提供了 oracle 复杂度保证且实验中稳定收敛
- **vs Frobenius/Spectral 范数正则化**: 这些经典正则化不区分 Jacobian 各行的差异，$\|·\|_{2\to\infty}$ 关注最大行范数提供更精细控制，在高瘦矩阵上尤其有效

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个有理论保证的 $\|·\|_{2\to\infty}$ matrix-free 估计算法，应用于 DNN 正则化的思路新颖
- 实验充分度: ⭐⭐⭐⭐ 合成实验+DNN分类+推荐系统三大场景覆盖，理论与实验互相验证
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨清晰，反例构造精巧，实验展示充分
- 价值: ⭐⭐⭐⭐ 算法简洁实用，理论扎实，在 DNN 训练和推荐系统中有落地价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] EFX and PO Allocation Exists for Two Types of Goods](efx_and_po_allocation_exists_for_two_types_of_goods.md)
- [\[ICLR 2026\] Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching](../../ICLR2026/ai_safety/beyond_match_maximization_and_fairness_retention-optimized_two-sided_matching.md)
- [\[CVPR 2026\] One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control](../../CVPR2026/ai_safety/one-to-more_high-fidelity_training-free_anomaly_generation_with_attention_control.md)
- [\[CVPR 2025\] H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection](../../CVPR2025/ai_safety/h2st_hierarchical_two-sample_tests_for_continual_out-of-distribution_detection.md)
- [\[NeurIPS 2025\] PubSub-VFL: Towards Efficient Two-Party Split Learning in Heterogeneous Environments via Publisher/Subscriber Architecture](../../NeurIPS2025/ai_safety/pubsub-vfl_towards_efficient_two-party_split_learning_in_heterogeneous_environme.md)

</div>

<!-- RELATED:END -->
