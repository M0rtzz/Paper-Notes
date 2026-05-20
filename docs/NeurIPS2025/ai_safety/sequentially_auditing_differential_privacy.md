---
title: >-
  [论文解读] Sequentially Auditing Differential Privacy
description: >-
  [NeurIPS 2025][AI安全][differential privacy] 提出基于序贯假设检验和核 MMD 统计量的差分隐私审计框架，可以在流式处理机制输出时随时有效地检测隐私违规，将所需样本量从现有方法的 50K 降低到数百个，并能在不到一次完整训练的过程中识别 DP-SGD 的隐私违规。
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "differential privacy"
  - "Sequential Testing"
  - "MMD"
  - "E-values"
  - "DP-SGD Auditing"
---

# Sequentially Auditing Differential Privacy

**会议**: NeurIPS 2025  
**arXiv**: [2509.07055](https://arxiv.org/abs/2509.07055)  
**代码**: [有](https://github.com/google-research/google-research/tree/master/dp_sequential_test)  
**领域**: AI Safety  
**关键词**: differential privacy, Sequential Testing, MMD, E-values, DP-SGD Auditing

## 一句话总结

提出基于序贯假设检验和核 MMD 统计量的差分隐私审计框架，可以在流式处理机制输出时随时有效地检测隐私违规，将所需样本量从现有方法的 50K 降低到数百个，并能在不到一次完整训练的过程中识别 DP-SGD 的隐私违规。

## 研究背景与动机

差分隐私（DP）已被工业界和政府机构广泛采用，但理论保证的实现依赖于正确的算法设计和正确的实现。微妙的代码 bug（如缺失常数、错误采样、固定种子）可能完全破坏隐私保障。此外，理论隐私参数 $\varepsilon$ 和 $\delta$ 通常是最坏情况上界，经验审计对于评估实际隐私泄露至关重要。

现有隐私审计方法有两大局限：

**参数化检验**：统计功效高但依赖强假设（如需要知道机制内部信息和输出分布）

**黑盒检验**：假设少但需要大量样本（通常 50K+），对于计算昂贵的算法如 DP-SGD 不可行——每个样本可能需要完整训练一个模型

核心问题在于现有方法都是**批处理**的：需要预先确定样本数量 $n$，$n$ 太大浪费计算、$n$ 太小结论不确定，且样本通常不能重用。

论文的突破在于引入**序贯检验**：样本逐个处理，检验统计量持续累积，一旦达到显著性水平即可停止——自动适应问题的未知复杂度，无需预先确定样本量。

## 方法详解

### 整体框架

审计 DP 分为两步：(1) 找到使 $\mathcal{A}(S)$ 和 $\mathcal{A}(S')$ 差异最大的相邻数据集对；(2) 检验隐私保证是否对该对成立。本文聚焦第二步。

核心思路是构造一个随机过程 $\{\mathcal{K}_t\}$，在零假设（满足 DP）下是超鞅（期望有界），在备择假设（违反 DP）下指数增长，从而高效检测违规。

### 关键设计

1. **MMD 与 Hockey-Stick 散度的新联系（定理 3.1）**：

    - 近似 DP 的定义基于 Hockey-Stick 散度 $D_{e^\varepsilon}(P||Q) \leq \delta$
    - Hockey-Stick 散度的似然比不可知时难以直接检验
    - 论文建立了更紧的 MMD 上界：若 $\mathcal{A}$ 是 $(\varepsilon, \delta)$-DP，则 $\text{MMD}(\mathcal{A}(S), \mathcal{A}(S')) \leq \sqrt{2}\left(1 - \frac{2(1-\delta)}{1+e^\varepsilon}\right)$
    - 该界比先前工作严格更紧，且不会在 $\varepsilon \to \infty$ 时变为空洞（趋近 $\sqrt{2}$ 而非 $\infty$）
    - 这意味着可以用 MMD 检验代替 Hockey-Stick 检验

2. **序贯检验统计量构建（定理 3.2）**：

    - 定义阈值 $\tau(\varepsilon, \delta) = \sqrt{2}(1 - \frac{2(1-\delta)}{1+e^\varepsilon})$
    - 假设设定：$H_0: \text{MMD} \leq \tau$ vs $H_1: \text{MMD} > \tau$
    - 构造乘积过程 $\mathcal{K}_t^*(\lambda) = \prod_{i=1}^t (1 + \lambda[f^*(X_i) - f^*(Y_i) - \tau])$
    - $H_0$ 下对适当 $\lambda$ 是非负超鞅，由 Ville 不等式控制 I 类错误
    - $H_1$ 下 $\log \mathcal{K}_t^*$ 以速率 $\Omega(\Delta^2)$ 增长（$\Delta = \text{MMD} - \tau$）

3. **实用算法（Algorithm 1）**：

    - 挑战：需要学习未知的最优 $\lambda^*$ 和见证函数 $f^*$
    - 用 **Online Newton Step (ONS)** 自适应学习 $\lambda_t$
    - 用 **Online Gradient Ascent (OGA)** 从样本中学习 $f_t$
    - 理论保证（定理 3.3）：$H_0$ 下 $\mathbb{P}(\mathcal{T} < \infty) \leq \alpha$；$H_1$ 下期望停止时间 $\mathbb{E}[\mathcal{T}] = O(\frac{\log(1/(\alpha\Delta^2))}{\Delta^2})$

### 损失函数 / 训练策略

- ONS 的损失：$\ell_t(\lambda) = -\log(1 + \lambda[\langle f_t, K(X_t, \cdot) - K(Y_t, \cdot)\rangle_\mathcal{H} - \tau])$
- OGA 的损失：$h_t(f) = \langle f, K(X_t, \cdot) - K(Y_t, \cdot)\rangle_\mathcal{H}$，投影到 $\|f\|_\mathcal{H} \leq 1$
- 核带宽用前 20 个样本的成对距离中位数设定，这些样本不参与正式检验

## 实验关键数据

### 主实验：加性噪声均值估计机制

| 机制 | $\varepsilon=0.01$ 拒绝率 | 平均样本数 | $\varepsilon=0.1$ 拒绝率 | 平均样本数 |
|------|-------------------------|-----------|------------------------|-----------|
| DPGaussian (✓DP) | 0.0 ± 0.0 | — | 0.0 ± 0.0 | — |
| NonDPGaussian1 | **1.0 ± 0.0** | 264 ± 9 | **1.0 ± 0.0** | 562 ± 29 |
| NonDPGaussian2 | 0.85 ± 0.08 | 1139 ± 126 | 0.05 ± 0.05 | 4776 |
| DPLaplace (✓DP) | 0.0 ± 0.0 | — | 0.0 ± 0.0 | — |
| NonDPLaplace1 | **1.0 ± 0.0** | 331 ± 14 | **1.0 ± 0.0** | 920 ± 62 |
| NonDPLaplace2 | **1.0 ± 0.0** | 192 ± 18 | 0.95 ± 0.05 | 770 ± 262 |

（先前方法 DPAuditorium 在 50 万样本下仍无法检测 NonDPLaplace2 和 NonDPGaussian2）

### DP-SGD 审计实验

| 设置 | 结果 | 样本数 |
|------|------|--------|
| 正确 DP-SGD ($\varepsilon_{\text{canary}}=0.01$) | 5 次运行均未拒绝 | 500 |
| 非隐私 SGD，$H_0: \varepsilon \leq 0.01$ | 平均 **60** 个样本即拒绝 | 60 |
| 非隐私 SGD，$H_0: \varepsilon \leq 0.1$ | 平均 **75** 个样本即拒绝 | 75 |
| 非隐私 SGD，250 步后隐私下界 | $\varepsilon = 0.43$ | 250 |

### 关键发现

- 序贯检验比批处理方法的样本效率提高了**两到三个数量级**
- 对 DP-SGD 可在**不到一次完整训练**过程中检测违规
- 正确的 DP 机制从未被误报（I 类错误控制良好）
- 局限：当真实 $\varepsilon$ 较大（>0.6）时，MMD 和阈值 $\tau$ 都接近理论上限，检测效率下降

## 亮点与洞察

- **核心贡献是将序贯检验引入 DP 审计**，这是一个自然但非平凡的创新——单侧检验对非零阈值的适配需要大量理论工作
- 更紧的 MMD-Hockey-Stick 散度联系直接提升了检验功效，且界不会在 $\varepsilon$ 增大时失效
- 并行运行多个 $\varepsilon$ 水平的检验可以从单个观测流中估计隐私下界，这对实践非常有用
- 白盒 DP-SGD 审计的设置（梯度投影到 canary 方向）使得每个训练步都产生一个样本，完美契合序贯框架

## 局限与展望

- 大 $\varepsilon$ 区间的审计能力不足（$\Delta^2$ 太小导致样本复杂度过高）
- 依赖固定的相邻数据集对，理想情况应自适应寻找最坏情况对
- 核函数和带宽的选择可能影响检验功效，未充分探讨最优选择策略
- 可扩展到 Rényi DP 或 $f$-DP 等其他隐私定义

## 相关工作与启发

- 基于 e-values/test supermartingales 的序贯检验近年在审计公平性、金融风险、选举等领域成功应用，本文是首次用于通用黑盒 DP 审计
- 与 Richter 等人（语言模型行为漂移检测）的对比特别有意义：本文的参数化过程 $\mathcal{K}_t^*(\lambda)$ 在 $H_1$ 下增长更快
- 对于 LLM 的隐私审计（RLHF 中的用户数据泄露？）也是潜在的应用方向

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将现代序贯检验应用于 DP 审计，有实质理论贡献
- 实验充分度: ⭐⭐⭐⭐ — 覆盖均值估计和 DP-SGD，与先前方法对比充分
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导清晰，动机陈述有力
- 价值: ⭐⭐⭐⭐⭐ — 实用价值极高，大幅降低 DP 审计的计算成本

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Multi-Class Support Vector Machine with Differential Privacy](multi-class_support_vector_machine_with_differential_privacy.md)
- [\[NeurIPS 2025\] Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy](mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)
- [\[ICML 2025\] On Differential Privacy for Adaptively Solving Search Problems via Sketching](../../ICML2025/ai_safety/on_differential_privacy_for_adaptively_solving_search_problems.md)
- [\[NeurIPS 2025\] Differential Privacy for Euclidean Jordan Algebra with Applications to Private Symmetric Cone Programming](differential_privacy_for_euclidean_jordan_algebra_with_applications_to_private_s.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)

</div>

<!-- RELATED:END -->
