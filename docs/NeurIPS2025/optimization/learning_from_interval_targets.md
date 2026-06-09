---
title: >-
  [论文解读] Learning from Interval Targets
description: >-
  [NeurIPS 2025][优化/理论][interval regression] 研究仅有区间标签（上下界）的回归问题，建立了基于假设类平滑性的非渐进泛化界（不依赖小 ambiguity degree 假设），并提出 minmax 学习框架利用平滑约束限制最坏情况标签…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "interval regression"
  - "weak supervision"
  - "generalization bound"
  - "Lipschitz constraint"
  - "minmax learning"
---

# Learning from Interval Targets

**会议**: NeurIPS 2025  
**arXiv**: [2510.20925](https://arxiv.org/abs/2510.20925)  
**代码**: [bloomberg/interval_targets](https://github.com/bloomberg/interval_targets)  
**领域**: 优化  
**关键词**: interval regression, weak supervision, generalization bound, Lipschitz constraint, minmax learning

## 一句话总结
研究仅有区间标签（上下界）的回归问题，建立了基于假设类平滑性的非渐进泛化界（不依赖小 ambiguity degree 假设），并提出 minmax 学习框架利用平滑约束限制最坏情况标签，在 18 个真实数据集上显著优于无约束方法。

## 研究背景与动机
**问题场景**：很多实际任务中精确标签昂贵或不可得——医疗测量费用高、传感器仅记录离散时刻、债券定价仅有买卖价差——但区间上下界往往容易获取。

**现有方法局限**：Cheng et al. (2023a) 分析了投影损失方法，但依赖两个强假设：(a) 可实现性（$f^* \in \mathcal{F}$）；(b) ambiguity degree $< 1$（无穷多区间的交集能恢复真实标签）。然而对回归问题，即使 $[y-\epsilon, y+\epsilon]$ 这样简单的区间 ambiguity degree 就等于 1。

**核心矛盾**：(a) 之前结论仅为渐进性的，缺乏有限样本保证；(b) 平滑假设类在区间学习中的作用未被充分利用。

**切入角度**：平滑（Lipschitz）假设类使得函数值在相近输入处不能差异太大，从而能"去噪"原始区间——将宽区间缩窄为更紧的有效区间（Fig. 2-3）。

## 方法详解

### 问题定义

训练数据 $\{(x_i, l_i, u_i)\}_{i=1}^n$，其中 $l_i \le f^*(x_i) \le u_i$。目标：学 $f \in \mathcal{F}$ 最小化 $\text{err}(f) = \mathbb{E}[\ell(f(X), Y)]$。

### 方法一：投影损失（Projection Loss）

定义投影损失：

$$\pi_\ell(f(x), l, u) = \min_{\tilde{y} \in [l,u]} \ell(f(x), \tilde{y})$$

由 Proposition 2.1，可简化为边界处的评估：

$$\pi_\ell(f(x), l, u) = \mathbf{1}[f(x)<l]\cdot\ell(f(x),l) + \mathbf{1}[f(x)>u]\cdot\ell(f(x),u)$$

即：函数值在区间内时损失为 0，超出区间时惩罚到最近边界的距离。

### 区间缩窄的关键机制（Proposition 3.4 + Theorem 3.6）

**核心 insight**：对 $m$-Lipschitz 假设类 $\mathcal{F}$（$|f(x) - f(x')| \le m\|x - x'\|$），任何满足投影损失为 0 的函数 $f$ 在点 $x$ 处的值被限制在缩窄区间内：

$$f(x) \in [l_{\mathcal{D}\to x}^{(m)},\ u_{\mathcal{D}\to x}^{(m)}]$$

其中 $l_{\mathcal{D}\to x}^{(m)} = \sup_{x'} (l_{x'} - m\|x-x'\|)$，$u_{\mathcal{D}\to x}^{(m)} = \inf_{x'} (u_{x'} + m\|x-x'\|)$。

**直觉**：如果 $f(x)$ 要很大，那么附近所有 $x'$ 处 $f(x')$ 也必须大（Lipschitz 约束），但这些 $f(x')$ 又必须在各自区间内——矛盾。因此可用邻近点的区间信息收紧当前点的有效区间。

对一般的 $f \in \widetilde{\mathcal{F}}_\eta$（投影损失 $\le \eta$），Theorem 3.6 给出带缓冲的扩展界：

$$f(x) \in [l_{\mathcal{D}\to x}^{(m)} - r_\eta(x),\ u_{\mathcal{D}\to x}^{(m)} + s_\eta(x)]$$

缓冲 $r_\eta, s_\eta$ 由等式 $\mathbb{E}_X[(r - lg_{X\to x}^{(m)})_+^p] = \eta$ 隐式定义，且 $\eta \to 0$ 时 $r,s \to 0$。

### 主要泛化界

**Theorem 4.1（可实现情形）**：对 Rademacher 复杂度 $O(1/\sqrt{n})$ 的 $m$-Lipschitz 假设类，以高概率：

$$\text{err}(f) \le \underbrace{\mathbb{E}_X[|u_{\mathcal{D}\to X}^{(m)} - l_{\mathcal{D}\to X}^{(m)}|]}_{(a)\ \text{不可约误差}} + \underbrace{\tau + O\left(\frac{1}{\sqrt{n}}\right)\Gamma(\tau)}_{(b)\ \text{随}\ n\ \text{衰减}}$$

- (a) 依赖于假设类平滑度和区间质量——$m$ 越小区间越窄，但 $m$ 太小假设类表达力不够
- (b) 随样本量 $n$ 衰减到 $\tau$（$\tau$ 可任取足够小）

**Theorem 4.2（不可知情形）**：额外引入最优假设误差 OPT 项，上界收敛到 $\text{OPT} + \mathbb{E}[|u^{(m)} - l^{(m)}|] + 2\tau + 2\text{OPT}\cdot\Gamma(\tau)$。

### 方法二：Minmax 学习

**基础 Minmax**：对最坏情况标签优化

$$\min_f \sum_i \max_{\tilde{y}_i \in [l_i, u_i]} \ell(f(x_i), \tilde{y}_i)$$

由 Proposition 5.1，对 $\ell_1$ 损失等价于用区间中点做回归：$f' = \arg\min_f \sum_i |f(x_i) - (l_i+u_i)/2|$

**约束 Minmax**（利用平滑性，Proposition 5.3）：限制最坏标签来自假设类

$$\min_{f \in \mathcal{F}} \max_{f' \in \widetilde{\mathcal{F}}_0} \mathbb{E}[\ell(f(X), f'(X))]$$

**Proposition 5.4**：存在场景使约束 Minmax 误差为 0 而无约束 Minmax 误差任意大——平滑约束至关重要。

**两种实用近似**：
1. **Minmax (reg)**：对 $f'$ 加投影损失正则项，用 GDA 交替优化
2. **PL (Mean/Max)**：先训练 $k$ 个 $f_j \in \widetilde{\mathcal{F}}_\eta$ 作伪标签，再对 $f$ 做 $\min_f \max/\text{mean}_{j} \sum_i \ell(f(x_i), f_j(x_i))$

## 实验关键数据

### Lipschitz MLP vs 标准 MLP（18 个表格回归数据集，投影损失）

| 数据集 | LipMLP MAE | MLP MAE |
|--------|:---:|:---:|
| Ailerons | **3.278±0.034** | 4.323±0.098 |
| CPU Activity | **10.271±0.026** | 10.560±0.087 |
| Mercedes | **8.791±0.187** | 11.207±0.218 |
| Miami House | **1.013±0.028** | 1.671±0.055 |
| Sulfur | **10.681±0.082** | 14.421±0.279 |
| Superconduct | **0.540±0.021** | 1.459±0.099 |
| Topo 21 | **1.305±0.013** | 2.192±0.177 |
| YProp 4 | **2.360±0.050** | 3.828±0.435 |
| Allstate Claims | 86.547 | **86.542** |
| GPU | 29.817 | **25.123** |

**18 个数据集中 LipMLP 在 14 个上显著优于标准 MLP**，验证了平滑性在区间学习中的关键作用。

### Lipschitz 常数的影响（Fig. 7）

- Lipschitz 常数过小：假设类过于受限，OPT 项增大，误差上升
- Lipschitz 常数过大：退化为标准 MLP，失去区间缩窄优势，误差上升
- 最优 Lipschitz 常数在两者之间，与从训练集近似的 $m$ 值接近
- PL (Mean) 方法的误差作为水平线对比，LipMLP 在最优 $m$ 处通常更优

### 不同学习方法对比

| 方法 | 均匀区间最佳 | 中心对称区间最佳 | 通用推荐 |
|------|:---:|:---:|:---:|
| Projection | ✓ | — | 当区间质量好时 |
| Midpoint | — | ✓ | 当标签在区间中心附近 |
| Minmax (naive) | — | ✓ | = Midpoint |
| PL (Mean) | ✓ | ✓ | 综合最佳 |
| PL (Max) | ✓ | — | 保守估计 |

## 亮点
1. **去除 ambiguity degree 假设**——之前是区间学习理论的核心限制，本文用 Lipschitz 平滑性替代
2. 区间缩窄机制（Theorem 3.6）直觉清晰且实用，将平滑性转化为更小有效区间
3. 约束 Minmax（Proposition 5.4）可任意优于无约束版本——理论保证强
4. 非渐进泛化界可直接指导有限样本实践
5. Lipschitz MLP 的谱归一化实现简单，作为超参调节的"免费午餐"

## 局限与展望
1. 假设区间一定包含真实标签——"嘈杂区间"（标签可能在区间外）未处理
2. Theorem 4.1 中 $\Gamma(\tau)$ 依赖分布，某些分布下可能退化很大
3. Minmax (reg) 需 GDA 交替优化，训练不如投影损失稳定
4. 仅考虑 i.i.d. 设定，时序场景（如传感器数据）需新理论
5. 实验主要是中等规模表格数据，缺少图像/NLP 等高维验证

## 与相关工作的对比
- **vs. Cheng et al. (2023a)**：后者需 ambiguity degree $<1$ + 可实现性 + 仅渐进结论；本文无需这些假设，$O(1/\sqrt{n})$ 非渐进界
- **vs. 部分标签学习（Lv et al. 2020）**：后者针对分类（有限标签集），投影损失是其在回归上的自然推广
- **vs. 半监督学习**：区间学习是弱监督的一种，理论框架不同
- **vs. 区间回归（传统统计）**：传统方法多用似然/EM，本文从学习论角度给出泛化保证

## 启发与关联
- 平滑性 ↔ 区间缩窄 的关系启发在其他弱监督场景（如标签噪声、锚框回归）中利用先验结构性
- Lipschitz 常数作为超参的调节策略 can be applied to other constrained learning problems
- 约束 Minmax 框架可推广到其他不确定性集合（如置信区间、分布鲁棒优化）

## 评分
- ⭐ 新颖性: 4/5 — 平滑性驱动区间缩窄的理论洞察新颖，约束 Minmax 框架有实际意义
- ⭐ 实验充分度: 4/5 — 18 数据集 + 多方法对比 + Lipschitz 常数消融，较全面
- ⭐ 写作质量: 4/5 — 理论推导层次清晰，直觉图示有效
- ⭐ 综合价值: 4/5 — 填补了区间回归理论空白，方法实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[NeurIPS 2025\] Wasserstein Transfer Learning](wasserstein_transfer_learning.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](learning_theory_for_kernel_bilevel_optimization.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)

</div>

<!-- RELATED:END -->
