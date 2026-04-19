---
title: >-
  [论文解读] In Search of Adam's Secret Sauce
description: >-
  [NeurIPS 2025][优化][Adam] 本文通过训练 1500+ 语言模型的大规模实验发现：(1) Signum 虽能缩小 96% 的 SGD-Adam 差距，但仍比 Adam 慢 25%；(2) 设 $\beta_1 = \beta_2$ 是 Adam 的近最优简化；(3) 在 $\beta_1 = \beta_2 = \beta$ 下 Adam 可被重新解读为基于在线高斯变分推断估计梯度均值和方差的信噪比自适应 Signum。
tags:
  - NeurIPS 2025
  - 优化
  - Adam
  - Signum
  - implicit bias
  - variational inference
  - signal-to-noise ratio
  - language modeling
---

# In Search of Adam's Secret Sauce

**会议**: NeurIPS 2025  
**arXiv**: [2505.21829](https://arxiv.org/abs/2505.21829)  
**代码**: [GitHub](https://github.com/aorvieto/SecretSauce)  
**领域**: optimization / 语言模型训练  
**关键词**: Adam, Signum, implicit bias, variational inference, signal-to-noise ratio, language modeling

## 一句话总结
本文通过训练 1500+ 语言模型的大规模实验发现：(1) Signum 虽能缩小 96% 的 SGD-Adam 差距，但仍比 Adam 慢 25%；(2) 设 $\beta_1 = \beta_2$ 是 Adam 的近最优简化；(3) 在 $\beta_1 = \beta_2 = \beta$ 下 Adam 可被重新解读为基于在线高斯变分推断估计梯度均值和方差的信噪比自适应 Signum。

## 研究背景与动机
**Adam 的不可替代性**：尽管 Muon、Scion、SOAP 等新优化器在特定场景下超越 Adam，但它们仍然依赖 Adam 来更新嵌入层、LM head 和归一化层参数。Adam 的核心优势尚未被完全理解。

**Adam ≈ Signum？**：近期研究指出 Adam 与 SignSGD with momentum (Signum) 有密切联系。但在 160M 参数规模上，精心调优的 Signum 仍导致 **25% 的有效减速**——相同困惑度需要多 25% 的训练 budget。

**核心问题**：Adam 相比 Signum、RMSprop、SGD 等简化变体的"secret sauce"究竟是什么？

**方法论**：投入约 10,000 A100 GPU 小时，系统地消融所有超参数（包括为每个学习率独立调优动量参数），提供可复现的全面基准。

## 方法详解

### 大规模基准实验（§3）

**基本设定**：160M 参数 Transformer LM，SlimPajama 数据集，Chinchilla optimal 训练。

**运行规模**：
- SGD: 131 runs（调 weight decay, gradient clipping, momentum, lr）
- RMSprop: 48 runs
- Signum: 70 runs（调 clipping, momentum, lr）
- Adam: 200 runs（调 $\beta_1$, $\beta_2$, lr）

### Takeaway 1: Signum 缩小差距但仍不够

| 优化器 | 验证困惑度 |
|--------|-----------|
| Adam | **21.86** ± 0.21 |
| Signum | 23.23 ± 0.16 |
| RMSprop | 27.04 ± 0.34 |
| SGD+Cclip | 33.40 ± 0.39 |
| SignSGD | 36.78 ± 0.57 |
| SGD+Gclip | 37.76 ± 0.61 |
| SGD | 53.62 ± 5.14 |

Signum 缩小了 96% 的 SGD-Adam 差距，但仍有 1.37 困惑度点的差距，对应 25% 的训练效率损失。

### Takeaway 2: $\beta_1 = \beta_2$ 是近最优选择
在 200 个 Adam 配置中，对每个 $\beta_1$ 值，最优 $\beta_2$ 总是接近 $\beta_1$。$\beta_1 = \beta_2$ 的性能退化不超过 0.3 困惑度点（远小于与 Signum 的 1.37 差距）。

### 理论解读（§4）：变分推断视角

设 $\beta_1 = \beta_2 = \beta$，$\epsilon = 0$（实验证明 $\epsilon$ 在 $[10^{-6}, 10^{-15}]$ 间无显著影响）：

**Proposition 1**：Adam 更新方向可改写为：

$$d_k = \frac{m_k}{\sqrt{m_k^2 + \beta \cdot \text{EMA}_\beta[(m_{k-1} - g_k)^2]}}$$

分母中 $\beta \cdot \text{EMA}_\beta[(m_{k-1} - g_k)^2]$ 正是梯度方差的在线估计。

**Theorem 4.1（变分推断解读）**：Adam 的两个动量缓冲区对应求解以下 KL 正则化最大似然问题的闭合解：

$$\min_{m, \sigma^2 \geq 0} -\log p(g_{k+1} | m, \sigma^2) + \frac{1}{\lambda} \text{KL}(\mathcal{N}(m_k, \sigma_k^2) \| \mathcal{N}(m, \sigma^2))$$

其中 $\beta = 1/(1+\lambda)$，解为：
- $m_{k+1} = \beta m_k + (1-\beta) g_{k+1} = \text{EMA}_\beta[g_{k+1}]$
- $\sigma_{k+1}^2 = \beta \sigma_k^2 + \beta(1-\beta)(m_k - g_{k+1})^2 = \beta \cdot \text{EMA}_\beta[(m_k - g_{k+1})^2]$

**信噪比（SNR）自适应信任域**：Adam 可视为信任域大小随 SNR 自适应的 Signum：

$$d_k = \frac{\text{sign}(m_k)}{\sqrt{1 + \sigma_k^2 / m_k^2}}$$

- SNR 高（$\sigma_k^2 / m_k^2$ 小）→ 步长 ≈ 1 → 接近 Signum
- SNR 低 → 步长被压缩 → 保守更新

**唯一性（Proposition, §C.2）**：Adam 方向能表示为 $m_k / \sqrt{m_k^2 + \gamma \cdot \text{EMA}_\tau[(a m_{k-1} - b g_k)^2]}$ **当且仅当** $\beta_1 = \beta_2$。即只有等 beta 时分母才有精确的方差估计解读。

### 混淆因素排除（§3.3）

| 混淆因素 | 结论 |
|----------|------|
| $\epsilon$ 值 | $10^{-6}$ 到 $10^{-15}$ 间无显著差异；Signum 加 $\epsilon$ mollifier 也无改善 |
| 动量初始化 | 零初始化 vs 梯度初始化无实质区别 |
| Bias correction | 最终验证困惑度几乎不变 |

## 实验关键数据

### 不同设定下 $\beta_1 = \beta_2$ 的鲁棒性

| 消融维度 | $\beta_1 = \beta_2$ 表现 |
|----------|------------------------|
| 不同 batch size (128, 256, 512) | 始终近最优 |
| 不同序列长度 (512, 2048) | 始终近最优 |
| 不同数据 (SlimPajama, Fineweb) | 始终近最优 |
| 无 weight decay | 始终近最优 |
| 2× token budget | 始终近最优 |
| 410M 参数模型 (44 runs) | 始终近最优 |

### 标准 $(\beta_1, \beta_2) = (0.9, 0.95)$ vs 等 beta

| 模型规模 | $(0.9, 0.95)$ | 等 beta 最优 |
|----------|---------------|-------------|
| 160M | 较好 | 相当或更好 |
| 410M | 次优（Figure 5 显示） | 更好 |

### 二次函数验证（§5）

| 设定 | SGD | Signum | Adam ($\beta_1=\beta_2$) |
|------|-----|--------|-------------------------|
| 同质 Hessian | 慢 | 快 | 快（≈ Signum） |
| 异质 Hessian（模拟 Transformer） | 很慢 | 较快 | **最快** |

关键：在异质景观中，Adam 的方差项 $\sigma_k^2$ 在不同参数块上有不同量级，实现了块级自适应——这是固定 mollifier 无法替代的。

## 亮点与洞察
- **Adam = SNR 自适应 Signum**：这是对 Balles & Hennig (2018) 观察的精确化——他们无法证明分母是方差，因为当时 $\beta_1 \neq \beta_2$；本文通过 $\beta_1 = \beta_2$ 的简化使这一联系精确成立
- **变分推断视角的意外优雅**：Adam 的两个 EMA 缓冲区恰好是高斯变分推断的在线解，regularization 参数 $\lambda$ 恰好对应 EMA 系数 $\beta$
- **固定 $\epsilon$ 不能替代自适应方差**：实验和二次函数示例都表明，Signum + 常数 mollifier 达不到 Adam 的效果，必须是数据驱动的自适应方差项
- **$\beta_1 = \beta_2 = 0.95$ 可作为 LM 训练默认**：已被 Zhao et al., Shah et al. 等独立采用

## 局限与展望
- **仅限 160M 和 410M 规模**：虽然数量惊人（1500+ runs），但未在 1B+ 模型上验证
- **超参数网格固定**：虽然展示了在网格内均处于最优而非边界，但不同网格可能导致不同结论
- **小 batch size 下 $\beta_1 = \beta_2$ 可能稍有偏移**：Figure 3 在 batch size 128 时暗示了这一点
- **Theorem 4.1 解释了 Adam 的结构但未解释为何这种结构有效**：为何均值/方差应被安排为商的形式？
- **未考虑 weight decay 对理论的影响**：AdamW 中 weight decay 与 Adam 更新的交互未被理论覆盖

## 评分
- 新颖性: ⭐⭐⭐⭐ $\beta_1 = \beta_2$ 的发现看似简单但影响深远，变分推断解读优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 1500+ runs，约 10K A100 GPU 小时，覆盖大量消融维度
- 理论深度: ⭐⭐⭐⭐ Proposition 1 + Theorem 4.1 的推导严密，唯一性证明有力
- 写作质量: ⭐⭐⭐⭐⭐ 故事线清晰（实验发现 → 简化 → 理论解释 → 验证），Figure 精心设计
- 实用价值: ⭐⭐⭐⭐⭐ 直接可用的调参建议：$\beta_1 = \beta_2$ 让 Adam 变成单参数优化器

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

## 相关论文

- [The Rich and the Simple: On the Implicit Bias of Adam and SGD](the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [MOBO-OSD: Batch Multi-Objective Bayesian Optimization via Orthogonal Search Directions](mobo-osd_batch_multi-objective_bayesian_optimization_via_orthogonal_search_direc.md)
- [Understanding Adam Requires Better Rotation Dependent Assumptions](understanding_adam_requires_better_rotation_dependent_assumptions.md)
- [Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed](../../ICML2025/optimization/clipping_improves_adam-norm_and_adagrad-norm_when_the_noise_is_heavy-tailed.md)

<!-- RELATED:END -->
