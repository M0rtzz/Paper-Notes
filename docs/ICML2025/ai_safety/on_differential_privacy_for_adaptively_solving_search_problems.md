---
title: >-
  [论文解读] On Differential Privacy for Adaptively Solving Search Problems via Sketching
description: >-
  [ICML 2025][AI安全][differential privacy] 首次将差分隐私技术从数值估计问题扩展到搜索问题（需要返回解向量而非单一数值），提出在温和的稀疏近邻假设下用 $\tilde{O}(\sqrt{T} \cdot s)$ 份数据结构副本即可正确回答 $T$ 个自适应近似近邻查询的算法，同时给出依赖条件数的自适应回归数据结构。
tags:
  - "ICML 2025"
  - "AI安全"
  - "differential privacy"
  - "adaptive queries"
  - "approximate nearest neighbor"
  - "regression"
  - "sketching"
---

# On Differential Privacy for Adaptively Solving Search Problems via Sketching

**会议**: ICML 2025  
**arXiv**: [2506.05503](https://arxiv.org/abs/2506.05503)  
**代码**: 无  
**领域**: AI安全 / 差分隐私 / 理论算法  
**关键词**: differential privacy, adaptive queries, approximate nearest neighbor, regression, sketching

## 一句话总结

首次将差分隐私技术从数值估计问题扩展到搜索问题（需要返回解向量而非单一数值），提出在温和的稀疏近邻假设下用 $\tilde{O}(\sqrt{T} \cdot s)$ 份数据结构副本即可正确回答 $T$ 个自适应近似近邻查询的算法，同时给出依赖条件数的自适应回归数据结构。

## 研究背景与动机

**领域现状**：现代算法中，数据结构经常被嵌入到迭代优化或在线学习流程中——当前迭代的输出会影响下一轮输入。这要求数据结构能对抗自适应对手（adversary 可根据之前的输出设计后续查询），但传统随机化数据结构仅保证对 oblivious adversary（输入序列预先确定、与随机性无关）的正确性。

**现有痛点**：已有三种处理自适应查询的方案，各有瓶颈：(1) 准备 $T$ 个独立副本，每次查询用新的——空间和预处理开销 $O(T)$ 倍，不可接受；(2) 利用 $\varepsilon$-net 参数，需 $O(d)$ 个副本——当 $d \approx T$ 或 $d \gg T$ 时无效；(3) 差分隐私方法（Hassidim et al., Ben-Eliezer et al.）可将副本数降到 $O(\sqrt{T})$——但仅适用于输出单一数值的估计问题（如距离估计、回归代价）。

**核心矛盾**：搜索问题的输出是一个完整向量（如近邻点的坐标、回归解向量），相比单一数值会泄露更多关于数据结构内部随机性的信息。直接将差分隐私中位数机制推广到高维空间行不通——因为中位数机制无法保证输出是原始数据集中的合法点。

**本文要解决什么**：在自适应查询场景下，能否为搜索问题（返回解向量而非数值）设计出使用 $o(\min\{d, T\})$ 份数据结构副本的高效算法？

**切入角度**：将搜索问题归约为差分隐私选择问题——不是对数值取中位数，而是对候选答案进行隐私投票/选择。关键洞察：如果每次查询的候选答案集合大小有界（$\leq s$），就可以利用稀疏性设计高效的隐私选择机制。

**核心idea一句话**：利用搜索问题候选解集合的稀疏性，通过差分隐私选择机制（而非中位数机制）将副本数从 $\min\{d, T\}$ 降至 $\tilde{O}(\sqrt{T} \cdot s)$。

## 方法详解

### 整体框架

针对两类经典搜索问题分别设计算法：(1) 近似近邻搜索（ANN）——给定查询点，从数据集中返回一个距离足够近的点；(2) 自适应回归——设计矩阵和响应向量可被自适应更新，返回回归解向量。两者共享的核心思想是"将内部随机性视为需要保护的私有数据库"，但在技术实现上针对各自特点采取不同策略。

### 关键设计

1. **差分隐私选择机制（用于 ANN）**:

    - 功能：在自适应查询场景下，从 $\tilde{O}(\sqrt{T} \cdot s)$ 个 LSH 数据结构副本中正确选出近似近邻
    - 核心思路：对每个查询 $v_t$，采样一组数据结构副本并查询，每个副本返回零个或多个候选近邻。将所有候选近邻的出现次数构成计数向量，加入 Laplace 噪声，选择噪声计数最大的候选点输出。关键在于：由高级组合定理（Advanced Composition Theorem），采样 $\tilde{O}(\sqrt{T})$ 组独立副本就够了；候选解集合大小 $\leq s$ 的假设（Assumption 1.2）保证了隐私预算可控
    - 设计动机：与数值问题的中位数机制不同，ANN 要求输出数据集中的实际点。投票+加噪的选择机制天然满足这一约束，同时 Laplace 噪声提供差分隐私保证

2. **稀疏 Argmax 机制**:

    - 功能：解决 naive 选择机制中 $O(n)$ 噪声生成瓶颈
    - 核心思路：naive 方法需要为数据集中所有 $n$ 个点都生成 Laplace 噪声再取最大值，即使大部分点计数为零。稀疏 Argmax 利用计数向量的 $s$-稀疏结构：只对 $s$ 个非零位置加噪声，同时通过采样第 $n$ 阶顺序统计量（从 Laplace 分布）来模拟"所有零位置中噪声最大的那个"。最终在 $O(s \log n)$ 时间内完成选择
    - 设计动机：直接实现需要 $O(n)$ 时间，当数据集很大（$n \gg s$）时成为瓶颈。稀疏结构是 ANN 假设的自然推论，不利用它就浪费了问题结构

3. **坐标级私有中位数 + $\ell_\infty$ 保证（用于回归）**:

    - 功能：在自适应更新场景下返回回归解向量
    - 核心思路：准备 $\tilde{O}(\sqrt{Td})$ 个独立 sketch 矩阵 $S_i$，存储 $S_i U$ 和 $S_i b$。每次查询采样 $\tilde{O}(1)$ 个 sketch，求解 sketched 回归问题得到多个候选解 $x_{(1)}, ..., x_{(m)}$。对解向量的每个坐标独立取私有中位数（加 Laplace 噪声后取中位数）。关键创新：利用 SRHT（Subsampled Randomized Hadamard Transform）的特殊性质获得 $\ell_\infty$ 保证——每个坐标的误差为 $O(\frac{\alpha}{\sqrt{d}} \cdot \frac{\|Ux^* - b\|_2}{\sigma_{\min}(U)})$，使得坐标级中位数汇聚后整体误差可控
    - 设计动机：搜索问题不能像数值问题那样直接用标量中位数。但回归解是连续向量，可以按坐标分解——每个坐标变成一维数值估计问题，就可以复用标量隐私中位数。SRHT 的 $\ell_\infty$ 保证确保了按坐标分解不会放大误差

### 损失函数 / 训练策略

纯理论工作，不涉及训练。核心度量是复杂度——空间、预处理时间、查询时间、摊销代价。

## 实验关键数据

### 复杂度对比表格（ANN 问题）

| 方法 | 空间 | 均摊预处理/查询 | 查询时间 |
|------|------|-------------|---------|
| $T$ 份副本 | $Tn^{1+\rho}+nd$ | $n^{1+\rho}d$ | $n^\rho d$ |
| $d$ 份副本 (ε-net) | $n^{1+\rho}d$ | $\frac{d}{T}n^{1+\rho}d$ | $n^\rho d$ |
| **本文** | $\sqrt{T} \cdot s \cdot n^{1+\rho}+nd$ | $\frac{s}{\sqrt{T}} n^{1+\rho}d$ | $s \cdot n^\rho d$ |

当 $s = o(\sqrt{T})$ 时，本文在空间和均摊预处理上均优于 $T$ 份副本方案。

### 复杂度对比表格（回归问题）

| 方法 | 空间 | 查询时间 | 关键依赖 |
|------|------|---------|---------|
| $T$ 份 sketches | $Td^2/\alpha^2$ | $d^{\omega+1}/\alpha^2$ | 线性依赖 $T$ |
| $nd$ 份 sketches (ε-net) | $nd^3/\alpha^2$ | $d^{\omega+1}/\alpha^2$ | 线性依赖 $nd$ |
| **本文 (Thm 1.8)** | $\sqrt{T} \cdot d^{2.5}\kappa^2/\alpha^2$ | $d^{\omega+1}\kappa^2/\alpha^2$ | $\sqrt{T}$，含条件数 $\kappa$ |
| **本文 (Thm 1.9, 仅更新 b)** | — | $d^2$ | 预条件消除 $\kappa$ |

### 关键发现

- **从 $T$ 到 $\sqrt{T}$ 的改进是非平凡的**：对于搜索问题，这是首次突破线性副本数的下界
- **假设条件温和**：稀疏近邻假设（$s \leq n^\rho$）可通过预处理聚类满足，有界条件数是标准回归假设
- **ANN 和回归的技术路线根本不同**：ANN 用离散选择（投票），回归用坐标分解（中位数），反映了两类搜索问题的结构差异
- **Theorem 1.9 消除条件数依赖**：当只有响应向量 $b$ 被更新时，可以预计算 $U$ 的 preconditioner，将查询时间降到 $O(d^2)$

## 亮点与洞察

- **解决了重要的 open question**：将 DP 框架从数值估计扩展到搜索问题，是该方向的概念性突破。之前的工作留下的核心疑问"搜索问题能否做到 $\sqrt{T}$"本文给出了肯定回答
- **稀疏 Argmax 机制**避免了为 $n$ 个候选点全部生成噪声的瓶颈，巧妙利用了候选集的稀疏结构，运行时间从 $O(n)$ 降到 $O(s \log n)$
- **SRHT 的 $\ell_\infty$ 保证**在回归 sketch 中的创新应用：坐标级分解+私有中位数的组合思路可推广到其他需要返回向量的隐私问题

## 局限性

- 纯理论贡献，缺乏实验验证——不清楚常数因子和 polylog 项在实践中的影响
- ANN 假设要求最多 $s$ 个近似近邻，高密度数据集（如图像特征空间）可能违反
- 回归结果依赖条件数 $\kappa$（Theorem 1.8），对病态问题可能表现不佳
- 一般度量空间（非范数空间）下的 ANN 结果仍然开放

## 相关工作与启发

- **vs Hassidim et al./Ben-Eliezer et al.**：前者将 DP 用于数值估计达到 $\sqrt{T}$ 副本，本文将其扩展到搜索问题——技术难度更高因为输出维度从 1 变成 $d$
- **vs ε-net 方法**：ε-net 需要 $O(d)$ 副本，当 $d \approx T$ 时无改善。本文在 $s = o(\sqrt{T})$ 时始终优于两种 baseline
- **应用启发**：可直接用于在线加权匹配（匹配时需要自适应查询近邻）和 terminal embedding（降维后保持距离）两个下游问题，论文 Section 4 给出了具体的复杂度改进

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 解决了自适应搜索问题的核心 open question，概念性突破
- 实验充分度: ⭐⭐⭐ 纯理论工作，复杂度表格代替实验，但缺乏实际数据集验证
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，假设合理，证明结构完整
- 价值: ⭐⭐⭐⭐ 理论贡献卓越，为 DP+自适应数据结构领域开辟新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Solving Probabilistic Verification Problems of Neural Networks Using Branch and Bound](solving_probabilistic_verification_problems_of_neural_networks_using_branch_and_.md)
- [\[NeurIPS 2025\] Sequentially Auditing Differential Privacy](../../NeurIPS2025/ai_safety/sequentially_auditing_differential_privacy.md)
- [\[NeurIPS 2025\] Differential Privacy for Euclidean Jordan Algebra with Applications to Private Symmetric Cone Programming](../../NeurIPS2025/ai_safety/differential_privacy_for_euclidean_jordan_algebra_with_applications_to_private_s.md)
- [\[NeurIPS 2025\] Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy](../../NeurIPS2025/ai_safety/mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)
- [\[NeurIPS 2025\] Multi-Class Support Vector Machine with Differential Privacy](../../NeurIPS2025/ai_safety/multi-class_support_vector_machine_with_differential_privacy.md)

</div>

<!-- RELATED:END -->
