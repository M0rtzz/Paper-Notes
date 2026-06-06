---
title: >-
  [论文解读] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization
description: >-
  [ICML 2026][优化/理论][Stochastic Polyak step size] SPSsafe 把 Stochastic Polyak Step Size (SPS) 扩展到非光滑随机优化——既不需要 interpolation 假设也不需要知道最优值…
tags:
  - "ICML 2026"
  - "优化/理论"
  - "Stochastic Polyak step size"
  - "非光滑优化"
  - "随机次梯度"
  - "动量"
  - "鲁棒训练"
---

# SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization

**会议**: ICML 2026  
**arXiv**: [2512.02342](https://arxiv.org/abs/2512.02342)  
**代码**: 待确认  
**领域**: 优化 / 自适应步长 / 非光滑优化  
**关键词**: Stochastic Polyak step size, 非光滑优化, 随机次梯度, 动量, 鲁棒训练

## 一句话总结
SPSsafe 把 Stochastic Polyak Step Size (SPS) 扩展到非光滑随机优化——既不需要 interpolation 假设也不需要知道最优值，配合动量（IMA = SHB 等价形式）仍保有严格收敛保证；在 DNN 训练上比已有自适应方法（AdaGrad、Adam、DecSPS 等）更稳健，且梯度范数不塌缩到近零（抗梯度消失）。

## 研究背景与动机

**领域现状**：自适应优化是现代 ML 标配——AdaGrad、Adam 已是 default；Polyak step size 路线近年复兴（Loizou 2021 等），用函数值而非纯梯度决定步长，在 overparameterized 模型上表现强；已扩展到 SGD、Mirror Descent、Local SGD、SHB 等。

**现有痛点**：（1）现有 SPS 分析几乎都假设凸 + 光滑（smooth）；非光滑 regime（DNN 的 ReLU、L1 正则、ranking loss 等很常见）保证稀缺；（2）非光滑下已有 SPS 变体要么需 interpolation 假设（要求某个 $x^* \in X^*$ 使 $f_i(x^*) = f_i^*$ for all $i$，过强）要么需要知道 $f_i(x^*)$（实际不可用）；（3）动量 + SPS 在非光滑下更加 underexplored。

**核心矛盾**：要 Polyak 步长的 adaptive 优势 + 要非光滑下的收敛保证 + 不能假设 interpolation 也不能要 oracle 知最优值——三者同时满足的 SPS 变体不存在。

**本文目标**：（1）为 stochastic subgradient method (SSM) 提出 SPSsafe，非光滑凸优化下严格收敛、不依赖 interpolation 或最优值；（2）将动量（IMA 等价 SHB）加入并保持理论；（3）实证在 DNN 训练上稳健，且不出现梯度消失。

**切入角度**：经典 SPS 在非光滑下失效的根因——subgradient 可能跳变，$f_i(x_t) - f_i^*$ 是好的"distance to opt"代理但 $\|g_t^i\|^2$ 在非光滑下没保证；本文加 safeguard——把步长 cap 在一个固定上界 $\gamma_{\max}$，保证步长不爆但保留 Polyak 的 adaptive 性质。

**核心 idea**：safeguard $\gamma_t = \min\{(f_i(x_t) - \ell_i^*)_+/(\|g_t^i\|^2 + \epsilon), \gamma_{\max}\}$，其中 $\ell_i^* \leq \inf f_i$ 是已知下界（实际中常是 0 for nonnegative loss）；加 IMA 动量框架保持理论。

## 方法详解

### 整体框架

考虑 $\min_x f(x) = \tfrac{1}{n}\sum_i f_i(x)$，$f_i$ 凸 Lipschitz 非光滑、下界 $\ell_i^*$（如交叉熵 = 0）。

两个算法：
- **SSM with SPSsafe**：$x_{t+1} = x_t - \gamma_t g_t^i$，$g_t^i \in \partial f_i(x_t)$
- **IMA with SPSsafe**（动量，等价 SHB）：$z_{t+1} = z_t - \eta_t g_t^i$，$x_{t+1} = \tfrac{\lambda_{t+1}}{\lambda_{t+1}+1} x_t + \tfrac{1}{\lambda_{t+1}+1} z_{t+1}$

SPSsafe 步长：$\gamma_t = \min\left\{\frac{(f_i(x_t) - \ell_i^*)_+}{\|g_t^i\|^2 + \epsilon}, \gamma_{\max}\right\}$

### 关键设计

1. **Safeguard 上限 $\gamma_{\max}$（核心创新）**:

    - 功能：把 Polyak 步长 cap 在固定上界，避免非光滑下步长爆炸
    - 核心思路：非光滑下 $\|g_t^i\|^2$ 可能很小但 $f_i(x_t) - \ell_i^*$ 不一定小，naive Polyak 步长可能任意大；加 $\gamma_{\max}$ cap 保步长有界，从而保收敛分析里的 Hoeffding 类 bound 可用
    - 设计动机：classical SPS 在光滑下自然有界（gradient Lipschitz 保 $\|g\|^2$ 不会太小），非光滑下没这保证；safeguard 是 minimal 修复保理论

2. **$\ell_i^*$ 作下界替代 $f_i^*$**:

    - 功能：不需要 oracle 知最优 $f_i^*$，用已知下界即可
    - 核心思路：$\ell_i^* \leq \inf f_i$ 通常已知（非负损失 = 0），代入 $\gamma_t = (f_i(x_t) - \ell_i^*)_+/\|g_t^i\|^2$；用 $(\cdot)_+$ 截断避免负值
    - 设计动机：之前 SPS* 需 $f_i(x^*)$（oracle 不可得）；本文用下界相当于"已知 SPS"，不需要任何额外信息

3. **IMA 动量框架等价 SHB**:

    - 功能：把动量加入 SPSsafe 而不破坏理论
    - 核心思路：IMA 双序列（$z$ + $x$ 平均）等价 SHB 的 $x_{t+1} = x_t - \hat\gamma_t g_t^i + \beta(x_t - x_{t-1})$；用 $\hat\gamma_t = \gamma_t$ SPSsafe 步长；分析保持因为 IMA 形式更适合 Lyapunov 类分析
    - 设计动机：动量在实践中很有用但理论分析常很难；IMA 等价形式提供了 cleaner 分析框架

## 实验关键数据

### 凸非光滑基准

| 方法 | Logistic + L1 | SVM hinge loss | ranking loss |
|------|----------|----------|----------|
| AdaGrad | 0.083 | 0.142 | 0.295 |
| Adam | 0.076 | 0.135 | 0.281 |
| DecSPS | 0.082 | 0.148 | 0.302 |
| **SPSsafe (SSM)** | **0.069** | **0.121** | **0.258** |
| **SPSsafe (IMA)** | **0.063** | **0.114** | **0.247** |

3 个非光滑凸基准上 SPSsafe 一致领先，IMA 版（带动量）最优。

### DNN 训练（CIFAR-10 + ResNet-18）

| 方法 | 测试准确率 | 训练稳定性 |
|------|--------|--------|
| SGD + 调好的 LR | 94.7 | 中（需精调）|
| Adam | 93.8 | 高 |
| AdaGrad | 92.5 | 中 |
| **SPSsafe** | **94.5** | **高（不需调）** |
| **SPSsafe + IMA** | **95.1** | **高** |

SPSsafe + IMA 在 ResNet 上达 95.1%（不需手调 LR），与精调 SGD 持平甚至略胜。

### 梯度范数追踪（抗梯度消失）

| 方法 | 训练后期梯度范数 | 梯度消失? |
|------|-------------|----------|
| Adam | 趋近 0 | ✓ 严重 |
| AdaGrad | 趋近 0 | ✓ 严重 |
| **SPSsafe** | 保持 $\sim 10^{-2}$ | ✗ 无 |

SPSsafe 训练后期梯度范数不塌缩——意味着模型仍在 active learning，对继续微调或对抗鲁棒性友好。

### 关键发现
- **safeguard 是非光滑下保理论的关键**：去掉 safeguard，Polyak 步长在非光滑下可能爆炸或发散
- **跨凸非光滑 + DNN 一致**：3 个凸基准 + ResNet 都 SOTA
- **抗梯度消失是意外收益**：SPSsafe 训练后期梯度不塌，对 fine-tune 和 adversarial training 友好
- **不需要 LR 调参**：Polyak 风格 step size 天然 adaptive，省去 LR sweep

## 亮点与洞察
- **safeguard 是 minimum-viable fix**：非光滑下 SPS 失效的最简洁修复，理论与实践都不复杂
- **不需要 interpolation 或 oracle 是关键贡献**：之前 SPS 非光滑分析都需强假设；本文消除这些让方法真正实用
- **抗梯度消失的副作用**：传统 adaptive optimizer 都让后期梯度变小（这是 noise floor 副作用），SPSsafe 不出现——意味着可能让深网更可训
- **IMA 等价 SHB 的分析便利**：把动量分析变 cleaner，这种"双序列等价"trick 在动量分析里非常有用

## 局限性 / 可改进方向
- $\gamma_{\max}$ 和 $\ell_i^*$ 仍需手设；自适应 $\gamma_{\max}$ 可能更鲁棒
- 仅证凸非光滑收敛；非凸下保证未严格给（实验 work 但理论 open）
- 在大规模 LLM/Transformer 训练上未充分验证；后期可能需要与 Adam 风格 second-moment 估计结合
- 计算 $f_i(x_t)$ 每步要 forward，与传统 SGD 同复杂度但比纯 gradient method 稍贵
- 没探索分布式 / 通信压缩场景

## 相关工作与启发
- **vs Loizou 2021 (classical SPS)**：classical SPS 需 interpolation；本文不需
- **vs Garrigos 2023 (SPS*)**：SPS* 需 $f_i(x^*)$ oracle；本文用下界
- **vs AdaGrad / Adam**：adaptive second-moment 路线；SPS 用函数值，互补思路
- **vs DecSPS, SPSL, Oikonomou-Loizou 2025**：那些有动量但仍依赖部分假设；本文是首个非光滑下完整理论 + 动量版本
- **启发**：safeguard 思想（用上界 cap adaptive 量）可推广到其他 adaptive 算法的非光滑扩展；下界 $\ell_i^*$ 替 oracle 的技巧对所有 Polyak-style 方法适用

## 评分
- 新颖性: ⭐⭐⭐⭐ safeguard 简单但首次让 SPS 在非光滑下严格保证；动量 + 非光滑组合也是新
- 实验充分度: ⭐⭐⭐⭐⭐ 多凸非光滑基准 + DNN 训练 + 梯度范数分析
- 写作质量: ⭐⭐⭐⭐ 算法清晰，理论与实验对应；公式推导扎实
- 价值: ⭐⭐⭐⭐ 非光滑优化在 ML 普遍（ReLU、稀疏正则）；SPSsafe 提供 plug-in adaptive 方案，免调 LR

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler](adaptive_sharpness-aware_minimization_with_a_polyak-type_step_size_a_theory-grou.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[ICLR 2026\] Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization](../../ICLR2026/optimization/faster_gradient_methods_for_highly-smooth_stochastic_bilevel_optimization.md)
- [\[ICML 2026\] On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization](on_the_provable_suboptimality_of_momentum_sgd_in_nonstationary_stochastic_optimi.md)
- [\[ICML 2026\] Bayesian Gated Non-Negative Contrastive Learning](bayesian_gated_non-negative_contrastive_learning.md)

</div>

<!-- RELATED:END -->
