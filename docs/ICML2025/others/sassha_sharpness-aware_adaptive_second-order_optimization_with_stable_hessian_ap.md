---
title: >-
  [论文解读] Sassha: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation
description: >-
  [ICML2025][二阶优化] 提出 Sassha 优化器，将 sharpness-aware minimization（SAM）引入二阶优化框架，通过稳定 Hessian 近似和 lazy 更新策略，使二阶方法首次在泛化性能上全面超越 SGD、AdamW 和 SAM 等一阶方法。
tags:
  - ICML2025
  - 二阶优化
  - sharpness-aware minimization
  - Hessian 近似
  - 泛化
  - 损失景观
---

# Sassha: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation

**会议**: ICML2025  
**arXiv**: [2502.18153](https://arxiv.org/abs/2502.18153)  
**代码**: [GitHub](https://github.com/LOG-postech/Sassha)  
**领域**: 优化  
**关键词**: 二阶优化, sharpness-aware minimization, Hessian 近似, 泛化, 损失景观

## 一句话总结

提出 Sassha 优化器，将 sharpness-aware minimization（SAM）引入二阶优化框架，通过稳定 Hessian 近似和 lazy 更新策略，使二阶方法首次在泛化性能上全面超越 SGD、AdamW 和 SAM 等一阶方法。

## 研究背景与动机

近似二阶优化方法（如 AdaHessian、Sophia-H、Shampoo）借助曲率信息加速收敛，但在泛化性能上往往不如简单的 SGD。本文从**损失景观平坦度**的视角解释了这一现象：

- 作者在 ResNet-32 / CIFAR-100 上测量了多种 sharpness 指标（Hessian 最大特征值 $\lambda_{\max}(H)$、迹 $\text{tr}(H)$、梯度方向 sharpness $\delta L_{\text{grad}}$、平均 sharpness $\delta L_{\text{avg}}$），发现现有二阶方法收敛到的极小值比 SGD **尖锐数个数量级**
- 例如 Shampoo 的 $\lambda_{\max}(H)$ 高达 436374，而 SGD 仅为 265；Sassha 更低至 107
- 这种 sharp minima 与泛化能力的负相关已被大量理论和实证研究支持

**核心问题**：能否显式地减少二阶方法解的 sharpness，从而释放其泛化潜力？

## 方法详解

### 1. Sharpness-aware 二阶优化框架

Sassha 求解如下 min-max 问题：

$$\min_{x} \max_{\|\epsilon\|_2 \le \rho} f(x + \epsilon)$$

类似 SAM，通过一阶近似求解内层最大化获得扰动方向：

$$\epsilon_t^\star = \rho \frac{\nabla f(x_t)}{\|\nabla f(x_t)\|_2}$$

然后对扰动后的目标函数 $\tilde{f}_t(x) = f(x + \epsilon_t^\star)$ 做二阶 Taylor 展开，得到更新规则：

$$x_{t+1} = x_t - H(x_t + \epsilon_t^\star)^{-1} \nabla f(x_t + \epsilon_t^\star)$$

### 2. 稳定 Hessian 近似（Stable Hessian Approximation）

sharpness 最小化会使 Hessian 条目趋近零，导致对角 Hessian 估计的数值不稳定。Sassha 采用三项关键设计：

- **平方根缩放**：使用 $|{\hat{H}}|^{1/2}$ 替代原始 Hessian，平滑放大近零条目（当 $0 < h < 1$ 时 $\sqrt{h} > h$），保留各维度相对尺度，无需额外超参（对比 damping/clipping）
- **绝对值处理**：$|\hat{H}| = \sum_i |\hat{H}_{ii}| \mathbf{e}_i \mathbf{e}_i^\top$，翻转负曲率方向的符号以避免收敛到鞍点
- **指数移动平均**：对 Hessian 对角做 EMA 平滑，减少随机估计噪声

### 3. Lazy Hessian 更新

每 $k$ 步才重新计算 Hessian（默认 $k=10$），大幅节省计算开销：

$$D_t = \begin{cases} \beta_2 D_{t-1} + (1-\beta_2)|\hat{H}(x_t+\epsilon_t^\star)| & \text{if } t \bmod k = 1 \\ D_{t-1} & \text{otherwise}\end{cases}$$

关键发现：Sassha 对 lazy 更新的鲁棒性远优于其他二阶方法，因为 sharpness 最小化使优化轨迹经过低曲率变化区域，过去的 Hessian 在更长步数内仍保持有效。

### 4. 完整更新规则

$$x_{t+1} = x_t - \eta_t \bar{D}_t^{-1} \bar{m}_t - \eta_t \lambda x_t$$

其中 $\bar{m}_t$ 是扰动点梯度的偏差校正一阶矩，$\bar{D}_t = \sqrt{D_t/(1-\beta_2^t)}$ 是偏差校正的 Hessian 平方根。

## 实验关键数据

### 图像分类（验证准确率 %）

| 方法 | CIFAR-10 ResNet-20 | CIFAR-100 ResNet-32 | CIFAR-100 WRN-28-10 | ImageNet ResNet-50 | ImageNet ViT-s-32 |
|---|---|---|---|---|---|
| SGD | 92.03 | 69.32 | 80.06 | 75.58 | 62.90 |
| AdamW | 92.04 | 68.78 | 79.09 | 75.38 | 66.46 |
| SAM_SGD | 92.85 | 71.99 | 83.14 | 76.36 | 64.54 |
| SAM_AdamW | 92.77 | 71.15 | 82.88 | 76.35 | 68.31 |
| AdaHessian | 92.00 | 68.06 | 76.92 | 73.64 | 66.42 |
| Sophia-H | 91.81 | 67.76 | 79.35 | 72.06 | 62.44 |
| Shampoo | 88.55 | 64.08 | 74.06 | — | — |
| **Sassha** | **92.98** | **72.14** | **83.54** | **76.43** | **69.20** |

### 语言模型预训练（GPT1-mini, Wikitext-2 困惑度 ↓）

| 方法 | Perplexity |
|---|---|
| AdamW | 175.06 |
| SAM_AdamW | 158.06 |
| AdaHessian | 407.69 |
| Sophia-H | 157.60 |
| **Sassha** | **122.40** |

### Sharpness 对比（ResNet-32 CIFAR-100）

| 方法 | $\lambda_{\max}(H)$ | $\text{tr}(H) \times 10^3$ | 验证准确率 |
|---|---|---|---|
| SGD | 265 | 7.29 | 69.32% |
| AdaHessian | 11992 | 46.94 | 68.06% |
| Sophia-H | 22797 | 68.15 | 67.76% |
| Shampoo | 436374 | 6823 | 64.08% |
| **Sassha** | **107** | **1.87** | **72.14%** |

## 亮点与洞察

1. **首次诊断出二阶方法泛化差的 sharpness 根因**：用四种指标全面量化，解释了困扰社区多年的 "二阶方法泛化差" 现象
2. **SAM + 二阶的巧妙融合**：不是简单堆叠，而是发现 sharpness 最小化会破坏 Hessian 估计稳定性，并提出平方根缩放这一无超参的优雅解决方案
3. **Lazy Hessian 的意外收益**：sharpness 最小化天然使 Hessian 沿轨迹变化更缓慢，从而合理化了 lazy 更新策略，形成效率与性能的正向循环
4. **全面超越一阶基线**：Sassha 在 6 个视觉任务 + 语言预训练上均优于 SGD、AdamW 和 SAM，是二阶方法首次实现这一成就
5. **理论支撑完备**：收敛性证明 + 线性稳定性分析解释了 Sassha 为何偏好平坦极小值

## 局限与展望

1. **计算开销仍高于一阶方法**：即使 $k=10$ 的 lazy 更新，每 10 步仍需额外的 Hessian-vector product 反向传播，wall-clock 时间约为 SGD 的 1.1-1.2 倍
2. **对角 Hessian 近似的固有局限**：忽略了曲率的非对角结构信息，在高度非对角的损失景观中可能不够精确
3. **超参选择**：$\rho$（扰动半径）和 $k$（Hessian 更新间隔）的交互效应未充分探索
4. **大规模验证不足**：语言模型实验仅在 GPT1-mini 和 SqueezeBERT 上验证，缺乏在十亿级参数模型上的评估
5. **收敛分析限于凸情形**：理论结果假设二次可微和凸性，与深度学习的非凸实际存在差距

## 评分

- 新颖性: ⭐⭐⭐⭐ — SAM 与二阶优化的结合方式以及稳定性设计新颖且有洞察力
- 实验充分度: ⭐⭐⭐⭐ — 视觉 + 语言、CNN + ViT + Transformer，多种 sharpness 指标全面覆盖
- 写作质量: ⭐⭐⭐⭐⭐ — 动机清晰、逻辑严密、理论与实验互相印证
- 价值: ⭐⭐⭐⭐ — 为二阶优化在深度学习中的实用化打开了新的方向

<!-- RELATED:START -->

## 相关论文

- [ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](../../CVPR2025/others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)
- [Stable Matching with Ties: Approximation Ratios and Learning](../../NeurIPS2025/others/stable_matching_with_ties_approximation_ratios_and_learning.md)
- [Overfitting in Adaptive Robust Optimization](../../NeurIPS2025/others/overfitting_in_adaptive_robust_optimization.md)
- [TANGO: Clustering with Typicality-Aware Nonlocal Mode-Seeking and Graph-Cut Optimization](tango_clustering_with_typicality-aware_nonlocal_mode-seeking_and_graph-cut_optim.md)
- [Time-Aware World Model for Adaptive Prediction and Control](time-aware_world_model_for_adaptive_prediction_and_control.md)

<!-- RELATED:END -->
