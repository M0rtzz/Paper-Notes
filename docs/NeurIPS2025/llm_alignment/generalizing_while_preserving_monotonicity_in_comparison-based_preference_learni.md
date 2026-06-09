---
title: >-
  [论文解读] Generalizing while Preserving Monotonicity in Comparison-based Preference Learning Models
description: >-
  [NeurIPS 2025][LLM对齐][preference learning] 提出 **Linear GBT with Diffusion Prior**，一类在保证**单调性**（偏好比较后被偏好方的分数不会反常下降）的同时能**泛化到未比较数据**的偏好学习模型…
tags:
  - "NeurIPS 2025"
  - "LLM对齐"
  - "preference learning"
  - "monotonicity"
  - "Bradley-Terry"
  - "扩散模型"
  - "alignment"
---

# Generalizing while Preserving Monotonicity in Comparison-based Preference Learning Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.08616](https://arxiv.org/abs/2506.08616)  
**代码**: [github.com/pevab/gbtlab2](https://github.com/pevab/gbtlab2)  
**领域**: LLM对齐  
**关键词**: preference learning, monotonicity, Bradley-Terry, diffusion prior, alignment

## 一句话总结

提出 **Linear GBT with Diffusion Prior**，一类在保证**单调性**（偏好比较后被偏好方的分数不会反常下降）的同时能**泛化到未比较数据**的偏好学习模型，首次正面回答了"泛化与单调性能否兼得"的核心问题。

## 研究背景与动机

偏好学习（如 RLHF、DPO）在 LLM 对齐中至关重要，但这些算法存在一个反直觉的"bug"：**当你告诉模型"A 优于 B"时，A 的分数可能反而下降**。

用一个极简例子说明：线性模型 $\beta \in \mathbb{R}^2$，两个 item 嵌入 $x_a=(1,0)$, $x_b=(2,0)$。比较 $a \succ b$ 会推动 $\beta_1$ 下降（因为 $x_{a1} < x_{b1}$），但 $a$ 的分数 $\beta^T x_a = \beta_1$ 因此也下降。更关键的是，第三个 item $c$ 嵌入 $(0,1)$ 的分数完全不受影响。

**核心矛盾**：
- 经典 **(Generalized) Bradley-Terry** 模型能保证单调性，但无法泛化——未被比较的 item 永远得分为 0
- 带嵌入的线性/非线性模型（包括 RLHF/DPO）能泛化，但**无法保证单调性**

**研究问题**：能否设计一个既能泛化又保证单调性的偏好学习算法？

## 方法详解

### 整体框架

**Linear GBT with Diffusion Prior** 在经典 GBT 基础上做两个扩展：

1. **嵌入 (Embeddings)**：每个备选方案 $a$ 有嵌入 $x_a \in \mathbb{R}^D$，分数建模为线性函数 $\theta_a(\beta) = x_a^T \beta$
2. **扩散先验 (Diffusion Prior)**：正则化项引入 Laplacian 矩阵 $L$ 编码备选方案之间的先验相似度

### 关键设计

**损失函数**：

$$\mathcal{L}(\beta|\mathbf{D}) = \underbrace{\frac{1}{2\sigma^2}\sum_d \beta_d^2 + \frac{1}{2}\sum_{ab}\theta_a(\beta)L_{ab}\theta_b(\beta)}_{\text{正则化 } \mathcal{R}(\beta)} + \sum_{(a,b,r)\in\mathbf{D}} \Phi_f(x_{a\ominus b}^T\beta) - r \cdot x_{a\ominus b}^T\beta$$

其中 $\Phi_f(\theta) = \log\int_{\mathfrak{R}} e^{r\theta} df(r)$ 是 root law $f$ 的累积生成函数。Laplacian 正则化项 $\sum_{ab} \theta_a L_{ab} \theta_b = \frac{1}{2}\sum_{a \neq b}|L_{ab}|(\theta_a - \theta_b)^2$ 鼓励相似备选方案具有相似分数。

**单调性保证的核心定理链**：

1. **Good Embedding**：嵌入 $x$ 是 good 的当且仅当对所有 Laplacian 矩阵 $Y$ 和所有 $(a,b)$：$e_a^T(I + XY)^{-1}X e_{a\ominus b} \geq 0$

2. **Diffusion Embedding**：嵌入 $x$ 是 diffusion embedding 当且仅当 Gram 矩阵 $X_\lambda = x^T x + \lambda I$ 的逆 $X_\lambda^{-1}$ 对所有 $\lambda > 0$ 都是 super-Laplacian 矩阵

3. **推导链**：Diffusion Embedding → Good Embedding → 单调性

**Theorem 1 (主定理)**：对任意 root law $f$、$\sigma > 0$、diffusion embedding $x$ 和 Laplacian $L$，$\text{GBT}_{f,\sigma,x,L}$ 是单调的。

### 损失函数 / 训练策略

**微分分析框架**：引入 Smoothed Loss $\mathcal{L}_\lambda$，利用积分表达式：

$$\theta^*(o(\mathbf{D})) - \theta^*(\mathbf{D}) = \int_0^1 \frac{d\theta_\lambda^*}{d\lambda}(\mathbf{D}, o) d\lambda$$

对 update 操作得到导数公式：

$$\frac{d\theta_{\lambda a}^*}{d\lambda}\bigg|_{\lambda=\mu} = (r-s) \cdot e_a^T(I + X(L+H))^{-1}X e_{a\ominus b}$$

对 append 操作得到类似公式，多出一个 $\Phi_f''$ 项。单调性归结为验证矩阵 $(I + X(L+\tilde{H}))^{-1}X$ 在 $e_a^T \cdot e_{a\ominus b}$ 方向上非负。

**One-Hot Encoding 特例 (Theorem 2)**：类别 one-hot 编码是 diffusion embedding，分数分解为 $\theta_a = \gamma_{d(a)} + s^2 \cdot \alpha_a$（类别分数 + 残差）。

## 实验关键数据

### 主实验

**随机嵌入的 Goodness 概率**：
- 高斯 i.i.d. 嵌入 $x$：$D/A$ 大时概率高，$A/D$ 大时急剧下降
- 拼接 $[I, x]^T$ 后：goodness 概率显著提升

**nMSE 对比**（$A=25$, $N=500$, uniform root law, 100 seeds）：

| 嵌入方式 | nMSE 表现 |
|---------|-----------|
| $[I, x]^T$（完整嵌入） | **最优**——融合类别和特征信息 |
| $I$（经典 GBT） | 小 $D$ 时好，大 $D$ 时差 |
| $x$（仅特征） | 小 $D$ 时差，大 $D$ 时好 |

### 消融实验

**数据效率**（$A=20$, $D=10$, one-hot 编码, 1000 seeds）：
- GBT with one-hot encoding 所需比较数量约为经典 GBT 的 1/2~1/3 即可达到相同 nMSE
- 说明结构化嵌入可**大幅减少数据需求**

### 关键发现

1. 随机嵌入大概率**不满足** goodness 条件，单调性远非自动保证
2. 拼接单位矩阵是简单有效的修复策略（理论支撑 Proposition 9）
3. 完整嵌入模型 $[I,x]^T$ 同时利用 GBT 和特征学习的优势，在不同 $D/A$ 比下始终表现最好
4. One-hot 结构化嵌入在数据稀缺时显著降低估计误差

## 亮点与洞察

- **首次正面回答**了泛化与单调性能否兼得的问题（答案是肯定的）
- **Diffusion embedding** 概念优美——将比较视为"热泵"，分数传播类似热扩散
- 将代数条件与图论（Laplacian）和物理（扩散动力学）联系起来，理论深度出色
- **实用意义**：对社交媒体内容评分（如 Tournesol 平台）、推荐系统等直接适用
- 明确指出 RLHF/DPO 等主流方法缺乏单调性保证，有助于引发社区对这一问题的重视
- 一个简单的修复方案：将嵌入与单位矩阵拼接即可大幅提升 goodness 概率

## 局限与展望

- 理论保证**仅限于 diffusion embedding**，更一般的嵌入类（如神经网络生成的）无法保证
- 仅考虑**线性模型**，而实际 RLHF/DPO 使用非线性模型（Transformer）
- 实验规模较小（合成数据为主），真实大规模偏好数据验证不足
- good embedding 的判定条件对实践者来说不够直观
- 未讨论如何将理论扩展到 DPO / RLHF 等基于梯度的非线性优化框架
- Goodness 判定为 NP-hard 或不可高效判定的可能性未讨论

## 相关工作与启发

- 经典 GBT (Fageot et al., AAAI 2024) 证明了单调性但不能泛化，本文是其自然延伸
- Bareilles et al. (2025) 证明非线性模型只有弱单调性（局部 pairwise），强化了线性+diffusion 的动机
- Chen et al. (NeurIPS 2024) 实证发现 DPO 排序不一致性，与本文理论互相印证
- 对 Tournesol 等协作评分平台有直接应用价值：保证用户偏好不会被"负反馈"
- Super-Laplacian 矩阵和图扩散的数学工具有潜力扩展到更复杂的社会选择问题

## 评分

- **创新性**：⭐⭐⭐⭐⭐ — 首次解决泛化+单调性兼得问题
- **理论深度**：⭐⭐⭐⭐⭐ — 完整的证明链，从 diffusion embedding 到单调性
- **实验充分度**：⭐⭐⭐ — 合成实验为主，真实数据验证较少
- **实用性**：⭐⭐⭐⭐ — 对社会选择和推荐系统有直接指导
- **综合评价**：8.0/10

## 与相关工作的对比

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Alignment of Large Language Models with Constrained Learning](alignment_of_large_language_models_with_constrained_learning.md)
- [\[NeurIPS 2025\] Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](reinforcement_learning_finetunes_small_subnetworks_in_large_language_models.md)
- [\[NeurIPS 2025\] Preference Learning with Lie Detectors can Induce Honesty or Evasion](preference_learning_with_lie_detectors_can_induce_honesty_or_evasion.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)
- [\[NeurIPS 2025\] LASeR: Learning to Adaptively Select Reward Models with Multi-Armed Bandits](laser_learning_to_adaptively_select_reward_models_with_multi-armed_bandits.md)

</div>

<!-- RELATED:END -->
