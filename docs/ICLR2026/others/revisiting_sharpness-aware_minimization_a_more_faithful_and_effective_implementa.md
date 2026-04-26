---
title: >-
  [论文解读] Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation
description: >-
   对 SAM 的底层机制提出新的直觉解释——扰动点梯度近似局部最大值方向，并揭示其不精确性及多步退化问题，进而提出 XSAM 通过显式估计最大值方向实现更忠实更有效的锐度感知最小化。
tags:

---

# Revisiting Sharpness-Aware Minimization: A More Faithful and Effective Implementation

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2603.10048](https://arxiv.org/abs/2603.10048)
- **代码**: [https://github.com/Cccjl219/XSAM](https://github.com/Cccjl219/XSAM)
- **领域**: others
- **关键词**: sharpness-aware minimization, SAM, optimization, generalization, flat minima

## 一句话总结
对 SAM 的底层机制提出新的直觉解释——扰动点梯度近似局部最大值方向，并揭示其不精确性及多步退化问题，进而提出 XSAM 通过显式估计最大值方向实现更忠实更有效的锐度感知最小化。

## 研究背景与动机
- **SAM** 通过最小化 $\rho$-邻域内的最大损失来促进平坦极小值和更好泛化，但其实际实现是在扰动点的梯度应用于当前参数——这个"错位梯度"为什么有效，一直缺乏直觉理解。
- **常见误解**：在估计最大值点计算的梯度并不直接最小化邻域内最大损失——关键在于梯度计算位置和应用位置不同。
- **多步 SAM 的困惑**：理论上更多步应该更好逼近最大值，但实际多步 SAM 性能却不升反降。

## 方法详解

### 核心洞察（通过可视化发现）
1. **更好的近似**（图 1a）：单步扰动点梯度 $g_1@\vartheta_0$ 比局部梯度 $g_0$ 更好地近似了从当前参数到邻域最大值的方向
2. **不精确性**：近似往往不准确，且在训练过程中变化大
3. **多步退化**（图 1b）：$g_k@\vartheta_0$ 可能比 $g_1@\vartheta_0$ 更差地指向最大值方向

### 理论确认
**命题 1**：在二阶近似下，对于足够大的 $\rho_m$：
1. $L(\vartheta_0 + \rho_m \frac{g_1}{\|g_1\|}) > L(\vartheta_0 + \rho_m \frac{g_0}{\|g_0\|})$ （SAM 梯度确实更好近似最大值方向）
2. 存在 $\alpha$ 使得 $g_\alpha = \alpha g_1 + (1-\alpha) g_0$ 比 $g_1$ 更好（SAM 梯度仍非最优）

### XSAM 方法
在 $v_0$（当前参数到扰动点方向）和 $v_1$（扰动点梯度方向）张成的 2D 超平面上显式搜索最大值方向：

$$v_0 = \frac{\vartheta_k - \vartheta_0}{\|\vartheta_k - \vartheta_0\|}, \quad v_1 = \frac{g_k}{\|g_k\|}$$

使用球面线性插值生成候选方向：
$$v(\alpha) = \frac{\sin((1-\alpha)\psi)}{\sin(\psi)} v_0 + \frac{\sin(\alpha\psi)}{\sin(\psi)} v_1$$

显式寻找最优 $\alpha^*$：
$$\alpha^* = \arg\max_{\alpha \in [0, a]} L(\vartheta_0 + \rho_m \cdot v(\alpha))$$

参数更新：
$$\theta_{t+1} = \theta_t - \eta_t \cdot v(\alpha^*) \cdot \|g_k\|$$

### 关键设计优势
1. 搜索空间包含已知最高损失点（$v_1$ 指向的方向）
2. 统一处理单步和多步设置
3. $\alpha^*$ 在训练过程中缓慢变化（图 2），只需每 epoch 更新一次 → 计算开销极小

### 计算开销
每次 $\alpha^*$ 更新需 20-40 个前向传播，按每 epoch 首个迭代更新，整体额外计算 < 3%。

## 实验关键数据

### 主实验：单步设置分类任务

| 数据集/模型 | SGD | SAM | GSAM | WSAM | **XSAM** |
|-----------|-----|-----|------|------|---------|
| CIFAR-10/ResNet-18 | 95.3 | 96.0 | 96.0 | 96.1 | **96.3** |
| CIFAR-100/ResNet-18 | 78.0 | 79.5 | 79.8 | 79.8 | **80.3** |
| CIFAR-100/DenseNet-121 | 79.5 | 81.0 | 81.2 | 81.2 | **81.6** |
| Tiny-ImageNet/ResNet-18 | 64.5 | 66.0 | 66.2 | 66.3 | **66.8** |

> XSAM 在所有模型-数据集组合上一致优于 SAM 及其变体。

### 消融实验：多步设置

| 方法 | 1步 | 2步 | 5步 | 10步 |
|-----|-----|-----|-----|------|
| SAM | 79.5 | 79.2 | 78.8 | 78.3 |
| **XSAM** | **80.3** | **80.5** | **80.6** | **80.7** |

> SAM 性能随步数增加而下降，XSAM 则持续改善——验证了多步退化现象及 XSAM 的修复。

### 训练时间对比（小时/200epochs）

| 模型/数据集 | SAM | XSAM | 额外开销 |
|-----------|-----|------|---------|
| VGG-11/CIFAR-10 | 0.93 | 0.96 | +3.2% |
| ResNet-18/CIFAR-100 | 2.40 | 2.43 | +1.3% |
| DenseNet-121/CIFAR-100 | 8.05 | 8.07 | +0.2% |

> XSAM 几乎不增加额外计算时间。

### 关键发现
1. SAM 梯度确实比 SGD 梯度更好近似最大值方向，但仍不准确
2. 多步 SAM 退化是因为 $g_k$ 的方向信息在远离 $\vartheta_0$ 后失真
3. $\alpha^*$ 训练中稳定，epoch-wise 更新即可
4. XSAM 与 ASAM 组合可进一步提升性能

## 亮点与洞察
- **直觉解释填空白**：首次给出 SAM "错位梯度" 为什么有效的直觉和视觉解释
- **多步退化难题**：优雅解释了困惑社区的现象——为什么更多上升步数不等于更好
- **极小开销的改进**：每 epoch 20-40 个前向传播，开销 < 3%
- **统一框架**：单步和多步 SAM 的统一改进方案

## 局限性
- 搜索限制在 2D 超平面内，可能遗漏高维空间中的真正最大值方向
- 假设最大值在邻域边界上，对复杂损失面可能不成立
- $\rho_m$ 超参引入，与 SAM 的 $\rho$ 含义不同
- 对超大规模模型（如 LLM）的效果未验证

## 相关工作
- **SAM 变体**: ASAM (Kwon et al., 2021) 自适应扰动；GSAM (Zhuang et al., 2022) 局部梯度正交分量
- **WSAM** (Yue et al., 2023) 和 Zhao et al. (2022a) 也用 $g_0, g_1$ 线性组合，但权重固定
- **SAM 理论**: Wen et al. (2023), Bartlett et al. (2023) 研究隐式偏差
- **多步 SAM**: Foret et al. (2020) 原始论文已提出但效果不佳

## 评分
- 新颖性: ⭐⭐⭐⭐ — 新的直觉解释 + 多步退化解释 + 统一方法
- 理论深度: ⭐⭐⭐⭐ — 二阶近似下的理论确认,直觉与形式分析结合
- 实验充分性: ⭐⭐⭐⭐ — 多模型多数据集、多步消融、计算开销分析
- 实用价值: ⭐⭐⭐⭐ — 即插即用替换 SAM，几乎无额外开销

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](../../CVPR2026/others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)
- [\[ACL 2025\] Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale](../../ACL2025/others/verbosity-aware_rationale_reduction_effective_reduction_of_redundant_rationale_v.md)
- [\[ICLR 2026\] A Representer Theorem for Hawkes Processes via Penalized Least Squares Minimization](a_representer_theorem_for_hawkes_processes_via_penalized_least_squares_minimizat.md)
- [\[ICLR 2026\] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)
- [\[ICML 2025\] Sassha: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation](../../ICML2025/others/sassha_sharpness-aware_adaptive_second-order_optimization_with_stable_hessian_ap.md)

<!-- RELATED:END -->
