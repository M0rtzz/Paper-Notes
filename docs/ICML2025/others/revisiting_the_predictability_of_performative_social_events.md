---
title: >-
  [论文解读] Revisiting the Predictability of Performative, Social Events
description: >-
  [ICML2025][performative prediction] 本文用现代学习理论工具（performative prediction + outcome indistinguishability）重新回答了20世纪社会科学中的经典问题：在预测会主动影响结果的情况下，社会事件是否仍可被准确预测？答案是肯定的——但这种"准确"的预测可能毫无用处。
tags:
  - ICML2025
  - performative prediction
  - multicalibration
  - outcome indistinguishability
  - 社会预测
  - online learning
---

# Revisiting the Predictability of Performative, Social Events

**会议**: ICML2025  
**arXiv**: [2503.11713](https://arxiv.org/abs/2503.11713)  
**代码**: 无（理论工作）  
**领域**: 社会预测 / 性能预测 / 学习理论  
**关键词**: performative prediction, multicalibration, outcome indistinguishability, 社会预测, online learning

## 一句话总结

本文用现代学习理论工具（performative prediction + outcome indistinguishability）重新回答了20世纪社会科学中的经典问题：在预测会主动影响结果的情况下，社会事件是否仍可被准确预测？答案是肯定的——但这种"准确"的预测可能毫无用处。

## 研究背景与动机

社会预测本质上不是被动描述未来，而是主动塑造未来：经济预测影响市场价格、选举预测影响投票率、气候预测影响政策。这种"预测影响数据"的动态称为 **performativity（表演性）**。

早在20世纪，多位学者就提出了核心问题：

- **Morgenstern (1928)**：经济预测通常不可能准确，因为公开的预测会自我否定
- **Simon (1954)、Grunberg & Modigliani (1954)**：用拓扑不动点定理证明了"自洽预测"的存在性，但未解决算法问题
- **Lucas critique (1976)**：宏观经济理论的转折点

本文用 **performative prediction** 框架（Perdomo et al., 2020）和 **multicalibration / outcome indistinguishability**（Hébert-Johnson et al., 2018; Dwork et al., 2021）这些现代工具，重新审视并给出算法层面的完整解答。

## 方法详解

### 核心形式化：Outcome Performative 分布映射

预测器 $f$ 发布后，数据生成过程为：

$$x \sim \mathcal{D}_x, \quad p \sim f(x), \quad y \sim \mathcal{D}_y(x, p)$$

其中特征 $x$ 的分布是固定的，但结果 $y$ 的条件分布 $\mathcal{D}_y(x,p)$ 依赖于预测值 $p$。这精确刻画了健康、教育等领域的预测动态。

### 核心定义：Performative Multicalibration

预测器 $f$ 是 $\varepsilon$-performatively multicalibrated（等价于 outcome indistinguishable），如果对所有 $c \in \mathcal{C}$：

$$\left| \mathbb{E}_{\substack{x \sim \mathcal{D}_x, p \sim f(x) \\ y \sim \mathcal{D}_y(x,p)}} [c(x,p)(y-p)] \right| \leq \varepsilon$$

当 $\mathcal{C}$ 仅含常数函数时，退化为 Simon (1954) 的条件；当 $\mathcal{C}$ 为所有有界可测函数时，要求 $f(x) = \mathbb{E}_{\mathcal{D}(f)}[y|x]$。

### 核心算法：Online-to-Batch 归约

**关键思想**：将 performative multicalibration 问题归约到一个更困难的在线问题，然后做 online-to-batch 转换。

**步骤**：

1. 使用在线算法 $\mathcal{A}$ 在每轮 $t$ 产生确定性预测函数 $f_t$
2. Nature 从分布映射中采样 $(x_t, y_t) \sim \mathcal{D}(f_t)$
3. 经过 $n$ 轮后，构造批处理预测器 $f_{\mathcal{A}}$：对给定 $x$，从 $\{f_1, \ldots, f_n\}$ 中均匀随机选取 $f_i$，预测 $p = f_i(x)$

**主定理 (Theorem 3.4)**：如果 $\mathcal{A}$ 保证在线 multicalibration regret 为 $\mathsf{Regret}_{\mathcal{A}}(T)$，则批处理版本以概率 $1-\delta$ 满足：

$$\left| \mathbb{E}_{\mathcal{D}(f_{\mathcal{A}})} [c(x,p)(p-y)] \right| \leq \frac{\mathsf{Regret}_{\mathcal{A}}(n)}{n} + 4\sqrt{\frac{\log|\mathcal{C}| + \log(1/\delta)}{n}}$$

证明核心：将 transcript 视为随机过程，通过 Martingale 论证和 Azuma-Hoeffding 不等式建立高概率上界。

### 具体实例化（Corollary 3.5）

使用 K29 核方法算法（Vovk et al., 2005）实例化归约：

| 函数类 $\mathcal{C}$ | Regret 率 | 批处理误差 | 运行时间 |
|---|---|---|---|
| 任意有限集合（连续于 $p$） | $\sqrt{T \cdot |\mathcal{C}|}$ | $\sqrt{|\mathcal{C}|/n}$ | $O(n^2 |\mathcal{C}|)$ |
| 线性函数 $\theta^\top x + p$ | $\sqrt{2T}$ | $\sqrt{2/n}$ | $O(n^2 d)$ |
| 低阶布尔函数（degree $s$） | $10\sqrt{d^s \cdot T}$ | $10\sqrt{d^s/n}$ | $O(ds \cdot n^2)$ |

### 结构性结果：Multicalibration → Stability

**Theorem 4.2**：对于二值结果 $y$ 和平方损失，如果 $f$ 是 $\varepsilon$-performatively multicalibrated w.r.t. $\mathcal{C} = \{p - 1/2\} \cup \{h(x) - 1/2 : h \in \mathcal{H}\}$，则 $f$ 是 $2\varepsilon$-performatively stable：

$$\mathbb{E}_{\mathcal{D}(f)} (y-p)^2 \leq \min_{h \in \mathcal{H}} \mathbb{E}_{\mathcal{D}(f)} (y - h(x))^2$$

### 负面结果：Stability ≠ Optimality

**Theorem 5.1（核心反例）**：存在分布映射 $\mathcal{D}(\cdot)$，使得预测器 $f$ 可以对所有有界连续函数 $c(x,p)$ 达到 performatively multicalibrated，但同时**最大化**上下文风险 (performative risk)：

$$\mathbb{E}_{\mathcal{D}(f)} (p-y)^2 \geq \max_{h \in \mathcal{H}} \mathbb{E}_{\mathcal{D}(h)} (y - h(x))^2$$

**构造**：无特征设定，$g(p) = p + 0.01$ 若 $p \leq 0.5$，$g(p) = p - 0.01$ 若 $p > 0.5$。不存在满足 $g(p) = p$ 的确定性不动点。唯一校准的随机预测器混合使用 $1/2$ 和 $1/2 + \varepsilon$，但这导致 $y$ 变成公平硬币，方差最大化（$1/4$），而最优解预测 0 或 1，风险仅为 0.01。

## 实验关键数据

本文为**纯理论工作**，无实验数据。核心贡献通过数学定理和构造性反例呈现：

| 贡献 | 内容 | 条件 |
|---|---|---|
| 可达性 | 总能高效找到 performatively multicalibrated 预测器 | 无需 $\mathcal{D}(\cdot)$ 的光滑性假设 |
| 收敛率 | $O(n^{-1/2})$，与监督学习相同 | 仅需结果有界 |
| Stability | Multicalibration 蕴含 performative stability | 平方损失 + 二值结果 |
| 不可避免的失败 | 完美校准可能最大化 performative risk | 不连续分布映射 |
| 与监督学习的反转 | 条件期望最优性在 performative 设定下完全反转 | — |

## 亮点与洞察

1. **弥合70年的理论鸿沟**：从 Morgenstern (1928) 和 Simon (1954) 提出的存在性问题到本文给出的完整算法解答，跨越近一个世纪
2. **无需光滑性假设**：不同于 performative prediction 文献中几乎所有先前工作要求 $\mathcal{D}(\cdot)$ 关于预测值 Lipschitz 连续，本文仅需结果有界，覆盖了教育领域中常见的阈值决策场景
3. **深刻的概念洞察**："准确≠有用"——在 performative 设定下，完美校准的预测器可能完全没解释任何结果方差，与监督学习中的直觉完全相反
4. **优雅的技术路径**：通过 online-to-batch 归约将复杂的 performative 问题简化为已有大量算法的在线学习问题
5. **随机化的必要性**：在不连续分布映射下，确定性预测器可能无法实现自洽，随机化预测器是必须的

## 局限与展望

1. **仅限 outcome performativity**：假设预测只影响结果 $y$ 的分布，不影响特征 $x$；真实场景中人们可能因预测改变行为特征
2. **stateless 设定**：未考虑历史预测对当前结果的累积影响（stateful performativity）
3. **二值结果限制**：结构性结果（Theorem 4.2）限于 $y \in \{0,1\}$，虽然作者指出可推广但未展开
4. **缺乏实证验证**：纯理论框架，未在真实社会预测场景（如选举预测、经济预测）中验证
5. **未解决 performative optimality**：仅保证 stability 而非更强的 optimality，且证明了两者之间存在不可弥合的鸿沟
6. **分布映射 $\mathcal{D}(\cdot)$ 未知**：实际中学习者只能通过部署预测器并观察样本来间接探索，但论文未深入讨论探索-利用的权衡

## 相关工作与启发

- **Performative Prediction** (Perdomo et al., 2020)：提供了形式化框架
- **Multicalibration** (Hébert-Johnson et al., 2018)：校准的多群体推广
- **Outcome Indistinguishability** (Dwork et al., 2021)：计算不可区分性概念
- **K29 算法** (Vovk et al., 2005)：核方法在线校准算法
- **Kim & Perdomo (2023)**：performative optimality 与 OI 的关系（受限设定）
- **Lucas Critique (1976)**：宏观经济学中预测影响政策的经典批判

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （跨越社会科学与机器学习理论的原创性桥接，解决近百年开放问题）
- 实验充分度: ⭐⭐⭐ （纯理论工作，构造性反例清晰但无实证）
- 写作质量: ⭐⭐⭐⭐⭐ （历史脉络清晰，技术呈现优雅，动机阐述出色）
- 价值: ⭐⭐⭐⭐⭐ （对社会预测的基础理论做出实质贡献，"准确≠有用"的洞察深远）
