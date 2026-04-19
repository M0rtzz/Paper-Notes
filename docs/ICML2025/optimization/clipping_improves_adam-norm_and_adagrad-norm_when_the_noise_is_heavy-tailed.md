---
title: >-
  [论文解读] Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed
description: >-
  [ICML 2025][优化][梯度裁剪] 证明了 AdaGrad/Adam 在重尾噪声下的高概率收敛可能很差（依赖置信水平的多项式），并证明梯度裁剪可以修复这个问题——Clip-AdaGrad-Norm 和 Clip-Adam-Norm 在重尾噪声下实现了对置信水平的对数多项式依赖的高概率收敛界，扩展到延迟步长版本。
tags:
  - ICML 2025
  - 优化
  - 梯度裁剪
  - Adam
  - AdaGrad
  - 重尾噪声
  - 高概率收敛
---

# Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed

**会议**: ICML 2025  
**arXiv**: [2406.04443](https://arxiv.org/abs/2406.04443)  
**代码**: 无  
**领域**: 优化  
**关键词**: 梯度裁剪, Adam, AdaGrad, 重尾噪声, 高概率收敛

## 一句话总结
证明了 AdaGrad/Adam 在重尾噪声下的高概率收敛可能很差（依赖置信水平的多项式），并证明梯度裁剪可以修复这个问题——Clip-AdaGrad-Norm 和 Clip-Adam-Norm 在重尾噪声下实现了对置信水平的对数多项式依赖的高概率收敛界，扩展到延迟步长版本。

## 研究背景与动机

**领域现状**：自适应步长方法（AdaGrad、Adam）是深度学习尤其是 LLM 训练的核心优化器。梯度裁剪（gradient clipping）也被广泛使用，特别是在 BERT/GPT 训练中。

**现有痛点**：
   - Zhang et al. (2020) 发现 BERT 预训练中的梯度噪声是重尾分布（$\alpha$-th矩有界，$\alpha \in (1,2]$），此时 SGD 可能发散
   - Clip-SGD 在重尾噪声下有可证收敛性，Adam 在实践中表现类似——但 Adam 的高概率收敛的理论保证缺失
   - 一些作者声称"Adam 内在包含裁剪效果"（因为自适应步长除以梯度范数类似于裁剪）——但这是猜测而非证明
   - 实践中 Adam+clipping 经常同时使用（如 BERT 微调），但到底是否需要 clipping 缺乏理论指导

**核心矛盾**：Adam 和 Clip-SGD 看似相似，但 Adam 是否真的不需要额外裁剪？

**本文目标**：严格回答"AdaGrad/Adam 在重尾噪声下是否需要梯度裁剪"。

**切入角度**：构造反例证明 AdaGrad/Adam 在重尾噪声下的高概率收敛确实很差（负面结果），然后证明加入裁剪后收敛显著改善（正面结果）。

**核心 idea**：Adam 的"内在裁剪"是不够的——自适应步长虽然缩放了梯度，但不截断极端值，而梯度裁剪显式截断极端值后使收敛的尾部概率呈对数衰减而非多项式衰减。

## 方法详解

### 整体框架
两部分结果：
1. **负面结果**：构造问题实例，证明 AdaGrad-Norm 和 Adam-Norm 在重尾噪声下的高概率收敛界至少有对置信水平 $\delta$ 的多项式依赖 $O(1/\delta^p)$
2. **正面结果**：证明 Clip-AdaGrad-Norm 和 Clip-Adam-Norm 有对置信水平的poly-log依赖 $O(\log^q(1/\delta))$

### 关键设计

1. **负面结果：AdaGrad/Adam 可能很差**:

    - 功能：证明未裁剪的 AdaGrad/Adam 在重尾噪声下的高概率收敛不够好
    - 核心论证：
        - 构造一维凸优化问题，噪声分布为 $\alpha$-stable distribution（满足 $\mathbb{E}[|\xi|^\alpha] < \infty$ 但 $\mathbb{E}[|\xi|^2] = \infty$）
        - 在此问题上，AdaGrad-Norm 的误差的高概率界为 $O(1/\delta^{2/\alpha - 1})$——随 $\alpha \to 1$（噪声越重尾），界越差
        - 关键洞察：AdaGrad 的分母 $\sqrt{\sum g_i^2}$ 虽然缩放了大梯度，但不截断——当单个极端噪声使 $g_t$ 非常大时，$g_t / \sqrt{\sum_{i \leq t} g_i^2}$ 的更新步长可能仍然过大
    - 设计动机：反驳"Adam ≈ 隐式裁剪"的流行观点——缩放 ≠ 截断

2. **正面结果：裁剪修复问题**:

    - 功能：证明 Clip-AdaGrad-Norm 和 Clip-Adam-Norm 有 polylog 高概率收敛
    - 核心思路：
        - 裁剪操作 $\text{clip}(g_t, c_t) = g_t \cdot \min(1, c_t/\|g_t\|)$，其中 $c_t$ 是时变裁剪阈值
        - 裁剪后的梯度有界→自适应步长的分析变得更好控制
        - 对凸问题：$\mathbb{E}[f(\bar{x}_T) - f(x^*)] \leq O(T^{-1/2} \cdot \log^q(T/\delta))$
        - 对非凸问题：$\min_t \|\nabla f(x_t)\|^2 \leq O(T^{-1/4} \cdot \log^q(T/\delta))$
    - 关键新颖性：处理了 Adam 特有的动量和偏差修正——不是简单地将 Clip-SGD 的结果搬过来
    - 扩展到延迟步长版本：实际分布式训练中步长有延迟→证明延迟版本也有类似保证

3. **延迟步长分析**:

    - 功能：处理分布式训练中梯度计算和步长更新不同步的问题
    - 核心思路：延迟 $d$ 步意味着使用的步长基于 $d$ 步之前的梯度历史——需要额外处理这种不一致
    - 结果：延迟引入 $O(d^2)$ 的额外误差项但不影响对 $\delta$ 的 polylog 依赖

### 损失函数 / 训练策略
- 理论分析为主
- 凸+非凸设置
- 重尾噪声假设：$\mathbb{E}[\|g_t - \nabla f(x_t)\|^\alpha] \leq \sigma^\alpha$，$\alpha \in (1, 2]$
- 裁剪阈值的选择：$c_t = O(\sigma \cdot t^{1/(2\alpha)})$（理论指导的时变阈值）

## 实验关键数据

### 合成问题验证
一维凸问题，$\alpha$-stable 噪声：

| 方法 | 高概率收敛（$\delta = 0.01$）| 重尾程度 $\alpha=1.5$ |
|------|----------------------|---------------------|
| SGD | 发散 | 确认理论 |
| AdaGrad-Norm | 慢收敛 | 确认 poly 依赖 |
| Clip-SGD | 收敛 | polylog 依赖 |
| **Clip-AdaGrad-Norm** | **快收敛** | **polylog 依赖** |

### BERT 微调实验
| 优化器 | 最终性能（验证集 Acc） | 训练稳定性 |
|--------|----------------|----------|
| AdamW（无裁剪） | 89.2% | 不稳定（偶尔跳变） |
| AdamW + gradient norm clip | 89.8% | 稳定 |
| **Clip-Adam-Norm** | **90.1%** | **最稳定** |

### 消融实验
| 配置 | 高概率误差界类型 |
|------|-------------|
| AdaGrad（无裁剪） | $O(1/\delta^{p})$ 多项式——差 |
| 裁剪阈值太大 | 接近无裁剪——差 |
| 裁剪阈值太小 | 偏差大——差 |
| **理论指导的裁剪阈值** | **$O(\log^q(1/\delta))$ polylog——好** |

### 关键发现
- Adam 并不"内在包含裁剪"——在重尾噪声下确实需要显式梯度裁剪
- 裁剪将高概率收敛从多项式依赖改善到对数多项式依赖——这是质的飞跃
- BERT 微调中的实践进一步验证了理论——裁剪不仅在理论上需要，实践中也确实有帮助
- 延迟步长的分析使理论适用于实际的分布式训练场景
- 裁剪阈值的选择至关重要——需要根据噪声矩的阶 $\alpha$ 来设定

## 亮点与洞察
- **"缩放 ≠ 截断"**——一个简单但深刻的区分，澄清了Adam与Clip-SGD的本质差异
- 负面结果（AdaGrad/Adam 可能很差）的证明本身就是重要贡献——制止了不加裁剪使用Adam的风险猜测
- 理论结果直接解释了实践中为什么 BERT/GPT 训练总是使用 gradient clipping——这不是迷信而是数学必要性
- 延迟步长的扩展使理论对分布式 LLM 训练有实际指导价值
- 对优化器设计者和 LLM 训练工程师都有直接影响

## 局限与展望
- 负面结果基于特定构造的问题实例——在"典型"DL问题上 Adam 的实际表现可能不那么差
- 裁剪阈值的理论最优选择需要知道 $\alpha$——实践中 $\alpha$ 未知
- 仅分析了 Adam-Norm（一维步长），完整的 coordinate-wise Adam 待分析
- 重尾噪声的假设在所有 DL 问题中是否成立不确定

## 相关工作与启发
- **vs Zhang et al. (2020)**: 发现重尾噪声现象但仅分析 Clip-SGD；本文扩展到 Adam/AdaGrad
- **vs Faw et al. (2023)**: 分析 Adam 在广义光滑条件下的收敛，但不处理重尾噪声
- **vs Gorbunov et al. (2020)**: 分析 Clip-SGD 的高概率收敛；本文将裁剪与自适应步长结合
- **启发**：重尾噪声可能是"Adam + clipping 在 LLM 训练中是标配"的理论根源

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次严格证明Adam需要裁剪+提供裁剪后的tight界
- 实验充分度: ⭐⭐⭐ 合成+BERT验证，但实验规模有限
- 写作质量: ⭐⭐⭐⭐ 问题定义精准，正负结果并列
- 价值: ⭐⭐⭐⭐⭐ 对LLM训练实践有直接理论指导

<!-- RELATED:START -->

## 相关论文

- [Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [A Unified View on Learning Unnormalized Distributions via Noise-Contrastive Estimation](a_unified_view_on_learning_unnormalized_distributions_via_noise-contrastive_esti.md)
- [In Search of Adam's Secret Sauce](../../NeurIPS2025/optimization/in_search_of_adams_secret_sauce.md)
- [The Rich and the Simple: On the Implicit Bias of Adam and SGD](../../NeurIPS2025/optimization/the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](../../NeurIPS2025/optimization/understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)

<!-- RELATED:END -->
