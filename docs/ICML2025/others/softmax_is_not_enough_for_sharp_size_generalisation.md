---
title: >-
  [论文解读] Softmax is not Enough (for Sharp Size Generalisation)
description: >-
  [ICML2025][softmax] 本文从理论上证明了 softmax 注意力在输入规模增大时**必然发生系数分散（dispersion）**，无法保持对少量关键元素的尖锐聚焦，并提出自适应温度（adaptive temperature）作为缓解手段。
tags:
  - ICML2025
  - softmax
  - 注意力分散
  - 尺寸泛化
  - 自适应温度
  - sharp function
  - Transformer
---

# Softmax is not Enough (for Sharp Size Generalisation)

**会议**: ICML2025  
**arXiv**: [2410.01104](https://arxiv.org/abs/2410.01104)  
**代码**: 无官方开源（论文提供了 JAX 实现片段）  
**领域**: Transformer泛化 / 注意力机制理论  
**关键词**: softmax, 注意力分散, 尺寸泛化, 自适应温度, sharp function, Transformer理论

## 一句话总结

本文从理论上证明了 softmax 注意力在输入规模增大时**必然发生系数分散（dispersion）**，无法保持对少量关键元素的尖锐聚焦，并提出自适应温度（adaptive temperature）作为缓解手段。

## 研究背景与动机

Transformer 中的 softmax 注意力被广泛认为能学到"电路"（circuits），在多样化输入上一致地执行尖锐（sharp）计算——例如找最大值时将全部注意力集中到一个 token 上。机械可解释性（mechanistic interpretability）研究已发现了 induction heads、comparator heads、retrieval heads 等多种尖锐注意力模式。

然而一个关键问题是：**这些尖锐行为能否在更大规模输入上泛化？** 实验表明，即使 LLM 在分布内能正确找到最大值，当输入序列变长时准确率会迅速下降。本文的核心动机正是从数学上**解释并证明**这一现象不可避免。

**Sharp 函数定义**：若一个函数的输出仅依赖于常数个输入（如 $\max$ 只依赖 1 个），则称其为 sharp 的。与之对比，$\text{average}$ 依赖全部 $n$ 个输入，不是 sharp 的。

## 方法详解

### 核心理论：Softmax 必然分散

**Lemma 2.1（softmax 系数分散）**：设 $n$ 个 logit $e_k^{(n)}$ 满足有界条件 $m \le e_k^{(n)} \le M$，温度 $\theta > 0$，则随着 $n \to \infty$：

$$\text{softmax}_\theta(\mathbf{e}^{(n)})_k = \Theta\left(\frac{1}{n}\right)$$

**证明思路**：利用 logit 上下界可得注意力系数的双侧界：

$$\frac{1}{n} \exp\left(-\frac{\delta}{\theta}\right) \le \alpha_k^{(n)} \le \frac{1}{n} \exp\left(\frac{\delta}{\theta}\right)$$

其中 $\delta = M - m$ 为 logit 的 spread。由于 $\delta, \theta$ 均为常数，注意力系数以 $\Theta(1/n)$ 的速率衰减至零。

**Theorem 2.2（Transformer 中 softmax 必然分散）**：对于使用有限词表（$|\mathcal{X}| < |\mathbb{N}|$）的 Transformer，由 MLP + softmax 自注意力层组成，无论 BERT 还是 GPT 风格，所有全局注意力头的系数在输入 token 数足够大时**必然趋于均匀分布**。

证明的关键洞察：有限词表 → 输入空间紧致 → 连续函数（MLP）将紧集映射到紧集 → 凸组合（注意力）保持紧性 → logit 有界 → 直接套用 Lemma 2.1。

**Proposition 3.1（尖锐性需要大权重）**：logit spread 满足：

$$\delta \le 2\sigma_{\max}^{(Q)} \sigma_{\max}^{(K)} \|\mathbf{y}\| \max_i \|\mathbf{x}_i\|$$

即注意力头要保持尖锐，必须**增大 $\mathbf{Q}, \mathbf{K}$ 矩阵的最大奇异值**，但这会导致过拟合和错误放大。

### 自适应温度（Adaptive Temperature）

直接设温度 $\theta = 0$（hard attention）在训练时难以收敛。作者提出一种**推理时自适应温度**方案：

1. 先计算原始 softmax 概率 $p = \text{softmax}(\mathbf{e})$
2. 计算 Shannon 熵 $H = -\sum_i p_i \log p_i$
3. 用一个 degree-4 多项式拟合 $\beta = 1/\theta$ 关于熵 $H$ 的关系：$\beta \approx -0.037H^4 + 0.481H^3 - 2.3H^2 + 4.917H - 1.791$
4. 仅在 $H > 0.5$ 且 $\beta > 1$ 时应用修正（不增加熵）
5. 返回 $\text{softmax}(\mathbf{e} \cdot \beta)$

该方法与 Flash Attention 兼容——熵修正计算可完全流式化，保持 $O(n)$ 内存。

## 实验关键数据

### Max Retrieval（找最大值任务）

训练在 $\le 16$ 个元素上，测试不同规模。自适应温度为**纯推理时调整**，不修改模型参数：

| 输入规模 | 16 (ID) | 64 | 256 | 1024 | 4096 | 16384 |
|---------|---------|-----|------|------|------|-------|
| Baseline | **98.6%** | 94.3% | 81.3% | 53.8% | 22.6% | 12.4% |
| Adaptive θ | **98.6%** | **94.5%** | **82.1%** | **57.7%** | **24.9%** | **14.0%** |
| p-value | 0.4 | 0.002 | 2e-4 | 1e-4 | 0.02 | 4e-3 |

OOD 规模上均有统计显著提升（paired t-test），且规模越大改善越显著（+3.9% @ 1024）。

### CLRS-Text 算法推理基准

在 Gemma 2B 微调实验中，将所有注意力头中的 softmax 替换为 adaptive temperature softmax（训练+推理时同时应用）。在 30 个算法任务的**绝大多数**上，自适应温度版本表现更优，尤其在 OOD 的更大问题规模上。

少数例外（Heapsort、MST Kruskal、Bubble Sort）可能因其占用的上下文窗口远超多项式拟合的训练范围。

### 实际 Transformer 中的 logit spread δ

| 模型 | δ 范围 | δ 均值 |
|------|--------|--------|
| Gemma 2B | [2.28, 14.78] | 5.69 ± 2.05 |
| Gemma 7B | [0.09, 32.74] | 5.82 ± 2.61 |

δ 远小于浮点数理论极限，说明实际模型中分散效应发生得比理论最坏情况更快。

## 亮点与洞察

- **理论贡献扎实**：首次严格证明 softmax 在 Transformer 中的注意力分散不可避免，解释了长度泛化失败的根本原因
- **分析优雅**：从紧致性 → 有界性 → 分散性的证明链条简洁有力
- **实用价值**：自适应温度作为 drop-in replacement，零成本应用于推理时
- **视野开阔**：论文结尾系统讨论了逃避分散定理的多种架构方向（unnormalized attention、sigmoid attention、selective attention、hard/local attention、不连续 MLP 等）
- **与 Flash Attention 兼容**：自适应温度可流式计算，不破坏高效注意力实现

## 局限与展望

1. **自适应温度是 ad-hoc 方法**：作者明确承认它并不逃脱理论结论，仅延缓分散
2. **多项式拟合的泛化性存疑**：degree-4 多项式基于特定任务拟合，迁移到不同场景（如 CLRS-Text）时需要重新适配或训练时引入
3. **CLRS-Text 需训练时引入**：不像 max retrieval 可以纯推理时使用，在复杂任务上需要重新微调
4. **未讨论 RoPE/ALiBi 等位置编码的影响**：现代 LLM 的位置编码可能与分散效应交互
5. **未验证在真实 LLM 推理任务**（如数学推理、代码生成）上的效果
6. **仅覆盖 softmax attention**：对于 linear attention、state space model 等不使用 softmax 的架构不适用（但这类架构本身可能不存在此问题）

## 相关工作与启发

- **机械可解释性**：Olsson et al. 2022 (induction heads)、Wang et al. 2022 (IOI)、Wu et al. 2024 (retrieval heads) — 本文从根本上质疑这些 circuits 的 OOD 鲁棒性
- **替代注意力机制**：Sigmoid Attention (Ramapuram et al. 2024)、Selective Attention (Leviathan et al. 2025)、Differential Transformer (Ye et al. 2025) — 可能天然避免分散
- **长度泛化**：Anil et al. 2022、Chiang & Cholak 2022 — 本文提供了长度泛化失败的数学解释
- **CLRS-Text**：Markeeva et al. 2024 — 算法推理基准

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次严格证明 softmax 分散定理，视角独特
- 实验充分度: ⭐⭐⭐⭐ — max retrieval 很干净，CLRS-Text 覆盖面广，但缺少真实LLM场景验证
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导清晰，行文优雅，从动机到结论一气呵成
- 价值: ⭐⭐⭐⭐⭐ — 对 Transformer 架构设计和长度泛化研究有深远启示

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Residual Matrix Transformers: Scaling the Size of the Residual Stream](residual_matrix_transformers_scaling_the_size_of_the_residual_stream.md)
- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](../../NeurIPS2025/others/one_sample_is_enough_to_make_conformal_prediction_robust.md)
- [\[ACL 2025\] Do not Abstain! Identify and Solve the Uncertainty](../../ACL2025/others/do_not_abstain_identify_and_solve_the_uncertainty.md)
- [\[NeurIPS 2025\] Coreset for Robust Geometric Median: Eliminating Size Dependency on Outliers](../../NeurIPS2025/others/coreset_for_robust_geometric_median_eliminating_size_dependency_on_outliers.md)
- [\[ACL 2025\] Limited Generalizability in Argument Mining: State-Of-The-Art Models Learn Datasets, Not Arguments](../../ACL2025/others/limited_generalizability_in_argument_mining_state-of-the-art_models_learn_datase.md)

<!-- RELATED:END -->
