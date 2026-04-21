---
title: >-
  [论文解读] OrdShap: Feature Position Importance for Sequential Black-Box Models
description: >-
  [NeurIPS 2025][Shapley Value] 提出 OrdShap，一种针对序列模型的特征归因方法，首次将特征的**值重要性（Value Importance, VI）**与**位置重要性（Position Importance, PI）**解耦，基于 Sanchez-Bergantiños 博弈论值提供理论保证。
tags:
  - NeurIPS 2025
  - Shapley Value
  - 特征位置重要性
  - 序列模型
  - 可解释性
  - GAN
---

# OrdShap: Feature Position Importance for Sequential Black-Box Models

**会议**: NeurIPS 2025  
**arXiv**: [2507.11855](https://arxiv.org/abs/2507.11855)  
**代码**: 暂无  
**领域**: 可解释AI / 特征归因  
**关键词**: Shapley Value, 特征位置重要性, 序列模型, 可解释性, Sanchez-Bergantiños值

## 一句话总结

提出 OrdShap，一种针对序列模型的特征归因方法，首次将特征的**值重要性（Value Importance, VI）**与**位置重要性（Position Importance, PI）**解耦，基于 Sanchez-Bergantiños 博弈论值提供理论保证。

## 研究背景与动机

深度学习序列模型（Transformer、RNN）在时序数据、自然语言等领域表现出色，但其黑盒特性需要事后归因方法来解释预测。现有方法（如 KernelSHAP、LIME、Integrated Gradients）存在一个根本性的假设缺陷：**它们假设特征顺序固定，将特征值和特征位置的影响混为一谈**。

举一个医疗场景的例子：预测住院时长时，某项血糖检测出现在序列早期还是晚期会显著影响预测结果，即使检测值本身不变。现有方法无法区分"这个特征重要是因为它的值"还是"因为它出现的位置"。

**核心矛盾**：序列模型的预测同时依赖特征值和特征位置，但现有归因方法无法分离这两种效应。

**切入角度**：定义一个新的博弈论框架，在 Shapley 值的基础上引入置换维度，对每个特征在每个位置上的边际贡献分别归因，然后通过汇总得到独立的值重要性和位置重要性。

## 方法详解

### 整体框架

OrdShap 的核心是一个 $d \times d$ 的归因矩阵 $\gamma_{i,\ell}$，其中行对应特征 $i$，列对应位置 $\ell$。每个元素表示"特征 $i$ 被置换到位置 $\ell$ 时的重要性"。在此基础上，通过边际化和线性回归分别提取 OrdShap-VI（值重要性）和 OrdShap-PI（位置重要性）。

### 关键设计

1. **广义特征函数 $\tilde{\omega}$（Section 4.1）**：传统 Shapley 值的特征函数 $\nu(S)$ 只考虑特征子集 $S$，OrdShap 引入了同时依赖子集和置换的新特征函数：

    $\tilde{\omega}_{f,x}(S,\sigma) = \mathbb{E}_{x' \sim \mathcal{X}}\left[f(x') \mid x'_{\sigma^{-1}(i)} = \begin{cases} x_i & \forall i \in S \\ x'_i & \forall i \in N \setminus S \end{cases}\right]$

   该函数同时对特征子集做消融（与 SHAP 类似），并对保留特征做置换（新增维度）。当置换为恒等映射时，退化为标准 SHAP 的特征函数。

2. **OrdShap 定义（Definition 1）**：对每个特征 $i$ 和位置 $\ell$，OrdShap 值定义为：

    $\gamma_{i,\ell}(N,\tilde{\omega}) = \sum_{\substack{S \subseteq N \\ i \in S}} \sum_{\substack{\sigma \in \mathfrak{S}_N \\ \sigma^{-1}(i)=\ell}} \frac{(|S|-1)!(|N|-|S|)!}{(|N|-1)!|N|!}\left[\tilde{\omega}(S,\sigma) - \tilde{\omega}(S\setminus\{i\},\sigma)\right]$

    - **OrdShap-VI**（Eq. 9）：对所有位置取平均 $\bar{\gamma}_i = \frac{1}{|N|}\sum_\ell \gamma_{i,\ell}$。Theorem 1 证明这等价于 Sanchez-Bergantiños 值，满足效率性、对称性、空玩家和可加性公理。
    - **OrdShap-PI**（Eq. 10）：对 $\gamma_{i,\ell}$ 关于位置 $\ell$ 做线性回归，斜率 $\beta_i$ 表示位置变化对重要性的影响方向和强度。

3. **高效近似算法**：精确计算需 $\mathcal{O}(d! \cdot 2^d)$，提出两种近似：

    - **采样算法**（Section 5.1）：随机采样子集和置换，逐个估计 $\gamma_{i,\ell}$，复杂度 $\mathcal{O}(dKL\delta_f + d^2KL)$
    - **最小二乘算法**（Section 5.2, Definition 2）：利用 Corollary 2.1（OrdShap-VI 等价于在平均特征函数 $\bar{\nu}$ 上的 Shapley 值），用 KernelSHAP 求解 $\alpha$，再回归得 $\beta$。复杂度 $\mathcal{O}(KL\delta_f + d^2KL + d^3)$，通常更快。

### 损失函数 / 训练策略

OrdShap 是事后解释方法，不涉及模型训练。最小二乘近似的优化目标（Eq. 11）是加权最小二乘问题，权重 $\mu(|S|)$ 按子集大小调整，取 $\mu = \frac{|N|-1}{\binom{|N|}{s}|S|(|N|-s)}$ 时最优解恢复 SB 值（Theorem 2）。

## 实验关键数据

### 主实验：Inclusion/Exclusion AUC（特征值重要性评估）

| 方法 | EICU-LOS (Inc↑) | EICU-Mort (Inc↑) | MIMICIII-LOS (Inc↑) | MIMICIII-Mort (Inc↑) | IMDB (Inc↑) |
|------|---------|---------|---------|---------|---------|
| KernelSHAP | 0.904 | 0.801 | 0.898 | 0.855 | 0.863 |
| DeepLIFT | 0.906 | 0.804 | 0.893 | 0.854 | 0.784 |
| LIME | 0.866 | 0.776 | 0.885 | 0.804 | 0.859 |
| Random | 0.812 | 0.769 | 0.851 | 0.705 | 0.797 |
| **OrdShap-VI** | **0.913** | **0.809** | **0.899** | **0.862** | **0.866** |

| 方法 | EICU-LOS (Exc↓) | EICU-Mort (Exc↓) | MIMICIII-Mort (Exc↓) | IMDB (Exc↓) |
|------|---------|---------|---------|---------|
| KernelSHAP | 0.626 | 0.730 | 0.485 | **0.766** |
| **OrdShap-VI** | **0.573** | **0.724** | **0.472** | 0.779 |

### 消融实验：位置重要性评估（特征置换后模型输出变化）

| 数据集 | OrdShap-PI 效果 | 传统方法效果 | 说明 |
|------|---------|---------|------|
| EICU-LOS | 输出不变/略升 | 无影响 | OrdShap-PI 正确识别位置依赖 |
| EICU-Mort | 输出显著上升 | 输出下降 | 传统方法无法捕捉位置重要性 |
| MIMICIII-LOS | 输出上升 | 输出下降 | 按 OrdShap-PI 置换确实增强预测 |
| IMDB | 输出基本不变 | 输出不变 | DistilBERT 对句子顺序不敏感 |

### 关键发现

- OrdShap-VI 在所有 EHR 数据集上的 Inclusion AUC 均优于所有对比方法，证明了引入位置信息后值归因更准确
- OrdShap-PI 是唯一能正确量化特征位置影响的方法；按 OrdShap-PI 排序置换特征能增强模型预测，而按传统方法排序则降低或不影响
- 在合成数据上，OrdShap 能完全分离 7 种 token 的值/位置效应（Figure 5），而传统方法无法区分
- IMDB 数据上位置效应较弱，说明 DistilBERT 的情感分析对句子顺序不太敏感（符合直觉）

## 亮点与洞察

- **首次解耦值与位置**：填补了序列模型可解释性中的重要空白，尤其对医疗等时序敏感场景有实际意义
- **博弈论基础扎实**：与 Sanchez-Bergantiños 值的联系（Theorem 1）为方法提供了公理化保证
- **LS 近似的巧妙设计**：利用 Corollary 2.1 将 $\alpha$ 和 $\beta$ 的求解解耦，先用 KernelSHAP 得到值重要性，再回归得到位置重要性
- 医疗案例分析（Figure 6）生动展示了临床意义：床旁血糖检测的高 KernelSHAP 值实际上主要来自位置效应

## 局限与展望

- 计算复杂度仍较高，尽管 LS 算法减少了模型调用次数
- 线性 OrdShap-PI 假设位置效应是线性的，可能无法捕捉非线性位置依赖
- 当前假设每个特征可被置换到任意位置，对于有严格时序约束的场景（如因果链）需要额外处理（附录 B.1 有讨论）
- 未探索与注意力权重等模型内部信息的结合
- Exclusion AUC 在 IMDB 上不如 KernelSHAP，可能因为 NLP 中位置效应确实较弱

## 相关工作与启发

- 与 KernelSHAP 的关系：OrdShap 是其到有序联盟博弈的自然推广
- PoSHAP 是最相关的工作，但它是全局方法（跨样本平均），OrdShap 是局部方法（解耦单个样本的值与位置）
- TimeSHAP 虽然针对序列模型，但不显式建模位置重要性
- SB 值在网络理论中已有应用，本文首次将其引入特征归因上下文

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 值与位置解耦是一个优雅且此前未被解决的问题
- 实验充分度: ⭐⭐⭐⭐ 多数据集、定量+定性评估、合成验证，但缺少大型 LLM 上的评估
- 写作质量: ⭐⭐⭐⭐⭐ 博弈论定义清晰，玩具示例非常直观
- 价值: ⭐⭐⭐⭐ 对医疗时序数据分析有直接价值，方法论可推广到所有序列模型

<!-- RELATED:START -->

## 相关论文

- [Position: We Need An Algorithmic Understanding of Generative AI](../../ICML2025/interpretability/position_we_need_an_algorithmic_understanding_of_generative_ai.md)
- [Position-aware Automatic Circuit Discovery](../../ACL2025/interpretability/position-aware_automatic_circuit_discovery.md)
- [TangledFeatures: Robust Feature Selection in Highly Correlated Spaces](tangledfeatures_robust_feature_selection_in_highly_correlated_spaces.md)
- [A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)
- [Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](../../ICLR2026/interpretability/temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)

<!-- RELATED:END -->
