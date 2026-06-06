---
title: >-
  [论文解读] Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization
description: >-
  [预训练] 从 KL 散度最小化角度重新解释 Shampoo 和 SOAP 的结构化二阶矩估计，揭示其固有局限，并提出 KL-Shampoo 和 KL-SOAP 两种实用方案，在无需 Adam grafting 的情况下匹配或超越原始方法。
tags:
  - "预训练"
---

# Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization

## 论文信息
- **会议**: ICLR 2026
- **arXiv**: [2509.03378](https://arxiv.org/abs/2509.03378)
- **代码**: [https://github.com/yorkerlin/KL-Methods](https://github.com/yorkerlin/KL-Methods)
- **领域**: LLM预训练
- **关键词**: Shampoo, SOAP, KL 散度, Kronecker 结构, 二阶优化, 协方差估计

## 一句话总结
从 KL 散度最小化角度重新解释 Shampoo 和 SOAP 的结构化二阶矩估计，揭示其固有局限，并提出 KL-Shampoo 和 KL-SOAP 两种实用方案，在无需 Adam grafting 的情况下匹配或超越原始方法。

## 研究背景与动机

### 核心问题
Shampoo 及其高效变体 SOAP 使用 Kronecker 结构的二阶矩估计进行预条件优化。然而：
1. Shampoo 通常需要与 Adam 的 step-size grafting 才能达到竞争力
2. SOAP 通过在 Shampoo 特征基下运行 Adam 缓解了这个问题，但引入额外内存开销
3. 先前分析主要基于 Frobenius 范数，忽略了 SPD（对称正定）约束

### 为什么选 KL 散度？
1. KL 散度天然尊重 SPD 约束（Frobenius 范数不尊重）
2. 在拟牛顿方法（BFGS、DFP）中，KL 提供统一解释框架
3. SPD 矩阵的各项不具等价角色，Frobenius 范数等价对待所有项
4. KL 可自然扩展到张量值情况

## 方法详解

### Shampoo 的 KL 解释

**Claim 1**：Shampoo（$p=1/2$）的估计规则可作为 KL 最小化的最优解：

$$\min_{\boldsymbol{S}_a} \text{KL}(\mathbb{E}[\boldsymbol{g}\boldsymbol{g}^\top], \boldsymbol{S})$$

其中 $\boldsymbol{S} = (1/d_b \boldsymbol{S}_a) \otimes \boldsymbol{I}_b$，最优解 $\boldsymbol{S}_a^* = \mathbb{E}[\boldsymbol{G}\boldsymbol{G}^\top]$。

**关键局限**：Shampoo 的单侧方法不能充分解决两因子联合学习的 KL 问题。

### KL-Shampoo：理想化方案

**Claim 2**：KL 联合最小化 $\min_{\boldsymbol{S}_a, \boldsymbol{S}_b} \text{KL}(\mathbb{E}[\boldsymbol{g}\boldsymbol{g}^\top], \boldsymbol{S})$ 的最优解满足：

$$\boldsymbol{S}_a^* = \frac{1}{d_b}\mathbb{E}[\boldsymbol{G}(\boldsymbol{S}_b^*)^{-1}\boldsymbol{G}^\top], \quad \boldsymbol{S}_b^* = \frac{1}{d_a}\mathbb{E}[\boldsymbol{G}^\top(\boldsymbol{S}_a^*)^{-1}\boldsymbol{G}]$$

**统计等价**：KL-Shampoo = 零均值矩阵高斯的最大似然估计 = 矩阵高斯白化。

### EMA 实现方案

近似上述条件的 EMA 更新：
$$\boldsymbol{S}_a \leftarrow (1-\beta_2)\boldsymbol{S}_a + \frac{\beta_2}{d_b}\boldsymbol{G}\boldsymbol{S}_b^{-1}\boldsymbol{G}^\top$$

**Claim 3**：此 EMA 方案是 KL 最小化的随机近端梯度步。

### 高效实现：QR 分解 + EMA 特征值

核心技术贡献：
1. **用 QR 替代特征分解**：SOAP 级别的迭代运行时间
2. **EMA 特征值估计**：使用过时特征基时的校正方案

$$\begin{pmatrix}\boldsymbol{\lambda}_a \\ \boldsymbol{\lambda}_b\end{pmatrix} \leftarrow (1-\beta_2)\begin{pmatrix}\boldsymbol{\lambda}_a \\ \boldsymbol{\lambda}_b\end{pmatrix} + \beta_2\begin{pmatrix}\text{diag}(\boldsymbol{Q}_a^\top \Delta_a \boldsymbol{Q}_a) \\ \text{diag}(\boldsymbol{Q}_b^\top \Delta_b \boldsymbol{Q}_b)\end{pmatrix}$$

### 统一框架：散度-投影视角

| 方法 | 散度 | 预条件结构 | 估计方案 |
|------|------|-----------|---------|
| KL-Shampoo | KL | 稠密 Kronecker | 最大似然 |
| Adafactor | von Neumann | 对角 Kronecker | 矩阵矩匹配 |
| F-Shampoo | Frobenius | 稠密 Kronecker | SVD-based |

### 内存比较

| 方法 | Kronecker | 特征基 | 特征值 | Adam 2nd | 额外开销 |
|------|-----------|--------|--------|----------|---------|
| Shampoo | $d_a^2+d_b^2$ | $d_a^2+d_b^2$ | $d_a+d_b$ | **$d_a d_b$ (grafting)** | 有 |
| SOAP | $d_a^2+d_b^2$ | $d_a^2+d_b^2$ | N/A | **$d_a d_b$ (eigenbasis)** | 有 |
| **KL-Shampoo** | $d_a^2+d_b^2$ | $d_a^2+d_b^2$ | $d_a+d_b$ | **无** | **无** |

## 实验

### 语言模型预训练

使用 150 次随机搜索的公平对比：

| 模型 | KL-Shampoo | SOAP | Shampoo+grafting | Shampoo (无 grafting) |
|------|-----------|------|-----------------|---------------------|
| NanoGPT (123M) | **最低 loss** | 次优 | 第三 | 较差 |
| NanoRWKV7 (162M) | **最低 loss** | 次优 | 中等 | 完全失败 |
| Llama (134M) | **最低 loss** | 次优 | - | - |
| NanoMoE (227M, 3D tensors) | **最低 loss** | 次优 | - | - |

### 关键发现

1. **KL-Shampoo 一致优于 SOAP**：在所有 4 个模型上——出乎意料
2. **KL-Shampoo 无需 grafting**：Shampoo (p=1/2) 不用 grafting 在 RWKV7 的 150 次运行中全部失败
3. **KL-Shampoo 优于 KL-SOAP**：核心原因——在最优特征基下梯度已被 Kronecker 对角化，额外的 Adam 修正是多余的
4. **EMA 特征值方案至关重要**：瞬时估计在使用过时特征基时严重退化
5. **VN-Shampoo (trace scaling) + EMA 方案也能超越 SOAP**

## 亮点

1. **深刻的理论洞察**：KL 视角统一了 Shampoo、SOAP、Adafactor 的解释
2. **实用改进**：消除 Adam 依赖，减少内存，保持 SOAP 级运行时间
3. **KL-Shampoo > KL-SOAP 的解释**：在最优特征基下，矩阵高斯白化已满足，无需进一步对角修正
4. **张量自然扩展**：KL 框架直接支持 3D+ 权重，无需 reshape

## 局限性

1. KL-Shampoo 的 EMA 方案引入了额外的矩阵乘法（$\boldsymbol{G}\boldsymbol{S}_b^{-1}\boldsymbol{G}^\top$）
2. 理论分析假设零均值高斯，实际梯度分布可能不满足
3. 实验主要在 100-200M 规模模型上验证，未测试数十亿参数模型
4. QR 分解在 PyTorch 中不支持半精度，需要精度转换

## 相关工作
- **Shampoo**: Gupta et al. (2018) — 原始 Kronecker 预条件器
- **SOAP**: Vyas et al. (2025a) — 在 Shampoo 特征基上运行 Adam
- **拟牛顿方法**: BFGS/DFP — KL 散度的经典应用
- **二阶优化**: K-FAC, EKFAC — Fisher 信息矩阵近似

## 评分
- **创新性**: ⭐⭐⭐⭐⭐ — KL 视角提供了深刻且统一的新理解
- **实验充分性**: ⭐⭐⭐⭐ — 150 次随机搜索的公平对比很有说服力
- **写作质量**: ⭐⭐⭐⭐ — 数学严谨，但篇幅较长
- **实用性**: ⭐⭐⭐⭐⭐ — 实际改进显著，内存减少且性能更好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Reducing Class-Wise Performance Disparity via Margin Regularization](reducing_class-wise_performance_disparity_via_margin_regularization.md)
- [\[ACL 2025\] FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling](../../ACL2025/llm_pretraining/fr_spec_speculative_sampling.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[ICLR 2026\] Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors](understanding_the_emergence_of_seemingly_useless_features_in_next-token_predicto.md)
- [\[CVPR 2025\] Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction](../../CVPR2025/llm_pretraining/improving_autoregressive_visual_generation_with_cluster-oriented_token_predictio.md)

</div>

<!-- RELATED:END -->
