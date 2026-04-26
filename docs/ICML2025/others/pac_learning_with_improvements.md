---
title: >-
  [论文解读] PAC Learning with Improvements
description: >-
  [ICML 2025][PAC learning] 提出"带改进的 PAC 学习"框架：当 agent 能真正提升自身特征至多 $r$ 时，保守分类器可实现零误差（将标准 PAC 中不可能的目标变为可能），有限 VC 维既非充分也非必要条件，改进学习与标准 PAC 和策略性分类存在本质分离。
tags:
  - ICML 2025
  - PAC learning
  - strategic improvement
  - sample complexity
  - VC dimension
  - intersection-closed
  - zero-error
---

# PAC Learning with Improvements

**会议**: ICML 2025  
**arXiv**: [2503.03184](https://arxiv.org/abs/2503.03184)  
**代码**: https://github.com/idanattias/PAC-Learning-with-Improvements  
**领域**: 学习理论  
**关键词**: PAC learning, strategic improvement, sample complexity, VC dimension, intersection-closed, zero-error

## 一句话总结

提出"带改进的 PAC 学习"框架：当 agent 能真正提升自身特征至多 $r$ 时，保守分类器可实现零误差（将标准 PAC 中不可能的目标变为可能），有限 VC 维既非充分也非必要条件，改进学习与标准 PAC 和策略性分类存在本质分离。

## 研究背景与动机

**领域现状**：标准 PAC 学习最基本的下界是：在几乎任何非平凡设置中，学到误差 $\epsilon$ 至少需要 $1/\epsilon$ 个样本，且零误差在连续分布上不可能。策略性分类领域关注 agent 对分类器的响应行为，已有大量研究分析操纵（gaming）和改进（improvement）的区别。

**现有痛点**：已有策略性改进研究主要关注如何激励 agent 改进或最大化社会福利，但没有人从可学性和样本复杂度的基础理论角度分析 agent 的改进能力如何影响学习。也没有人注意到改进可以使零误差成为可能——这在标准 PAC 和策略性分类中都不可能。

**核心矛盾**：标准 PAC 模型假设数据被动不变，策略性分类假设 agent 伪装（不真正改进）。但现实中 agent 的改进是真实的（如还清高息债务提升信用分），这种真实改进改变了可学性的根本特征。

**本文目标**：系统理解 agent 的改进能力如何影响可学性、样本复杂度和算法设计。

**切入角度**：以阈值学习为例——真实阈值为 $\theta$，若学到 $\hat{\theta}$ 满足 $\theta \le \hat{\theta} \le \theta + r$，则所有被判正的 agent 确实合格（无假阳性），所有真正合格的 agent 可通过努力被正确分类（假阴性可通过改进消除），实现零误差。

**核心 idea**：agent 的改进能力允许学习器"接近即可"，从而达成标准模型中不可能的零误差，同时改变了可学性的结构特征。

## 方法详解

### 整体框架

设实例空间 $\mathcal{X}$，改进函数 $\Delta: \mathcal{X} \to 2^{\mathcal{X}}$ 将每个 agent $x$ 映射到其可改进到的点集合 $\Delta(x)$。学习器发布分类器 $h$ 后，agent 若被判负且能通过改进到达正区域则会改进。损失函数采用对抗性 tie-breaking：$\text{Loss}(x; h, f^*) = \max_{x' \in \Delta_h(x)} \mathbb{I}[h(x') \ne f^*(x')]$，偏好保守分类器——对不确定的点标记为负（假阴性可被 agent 自行修复，假阳性不可挽回）。目标是给定 $m$ 个样本学到 $h$ 使得 $\text{Loss}_\mathcal{D}(h, f^*) \le \epsilon$。

### 关键设计

1. **与标准 PAC 和策略性分类的本质分离（Section 3）**:

    - 功能：建立改进学习与已有模型的基础关系
    - 核心思路：**Theorem 3.1** 证明有限 VC 维既非 PAC 改进可学的必要条件也非充分条件。非必要：若 $\Delta(x) = \mathcal{X}$（全局改进），对任意 VC 维无限的类，只需 $O(\frac{1}{\epsilon}\ln\frac{1}{\delta})$ 样本找到一个正例并输出单点分类器即可零误差。非充分：两区间并集类（VC 维 4）在特定改进函数下不可学，期望误差 $\ge 25\%$——因为非交闭类中保守策略无法保证安全。**Theorem 3.4** 进一步证明有限 SVC 维（策略性VC维，策略性分类的可学条件）也不蕴含改进可学性
    - 设计动机：揭示改进学习不是标准 PAC 或策略性分类的简单推广，而是一个本质不同的学习模型

2. **交闭类的闭包算法（Section 4.2）**:

    - 功能：为交闭（intersection-closed）概念类提供带改进的 PAC 学习算法
    - 核心思路：闭包算法返回包含所有正样本的最小假设 $h_S^c = \text{CLOS}_\mathcal{H}(\{x_i: y_i=1\})$。关键性质是 $h_S^c \subseteq f^*$（正区域是真概念正区域的子集），因此不产生假阳性。正区域外的 agent 若能改进到正区域中就不算误差。**Theorem 4.7** 证明任何 VC 维 $d$ 的交闭类可在 $O(\frac{1}{\epsilon}(d + \log\frac{1}{\delta}))$ 样本下带改进 PAC 学习，对任意改进函数和分布成立。改进增益为 $\mathbb{P}[x \in \text{IR}(h; f^*, \Delta)]$，即"改进区域"的概率质量
    - 设计动机：交闭性保证闭包算法的正区域始终是真概念的子集，从而自然实现"保守且安全"的分类。许多自然概念类（阈值、超矩形、半空间交、k-CNF）都是交闭的。**Theorem 4.8** 反向证明非交闭类在适当条件下不可 proper 学习

3. **几何概念的零误差学习（Section 4.1, 4.3）**:

    - 功能：具体量化改进带来的误差减少
    - 核心思路：**阈值**（Theorem 4.1）：改进半径 $r$ 的闭球，均匀分布 $[0,1]$，误差上界为 $(\epsilon - r)_+$，当 $\epsilon \le r$ 时达零误差，样本复杂度 $O(\frac{1}{\epsilon}\log\frac{1}{\delta})$ 与标准 PAC 相同。**超矩形**（Theorem 4.6）：$\ell_\infty$ 半径 $r$ 改进下，误差上界为 $(\epsilon - \mathbb{P}[\text{IR}])_+$，均匀分布 $[0,1]^2$ 时改进增益 $= 2r(l_1+l_2) + 4r^2$（与矩形周长成正比）。**半空间**（Theorem 4.9）：$d$ 维齐次半空间，角度改进 $r$ 下仅需 $\tilde{O}((d+\log(1/\delta))/r)$ 样本即可零误差
    - 设计动机：从一维阈值到高维半空间，逐步展示改进如何在不同几何结构中降低甚至消除分类误差

### 损失函数 / 训练策略

本文核心是理论框架。学习器的设计原则是"保守"——偏向将不确定的点标为负。具体策略包括：(1) 出最右一致阈值（阈值学习）；(2) 闭包算法（交闭类）；(3) 一致假设的正区域交集（半空间）。图模型（Section 5）中进一步证明：只需看到正子图支配集的标签，保守学习器即可零误差，样本复杂度 $O(n(\log n + \log(1/\delta))/(d_{\min}^+ + 1))$。

## 实验关键数据

### 理论结果对比

| 概念类 | 标准 PAC 误差 | 带改进 PAC 误差 | 零误差条件 | 样本复杂度 |
|--------|--------------|----------------|-----------|-----------|
| 阈值 (均匀分布) | $\epsilon$ | $(\epsilon - r)_+$ | $\epsilon \le r$ | $O(\frac{1}{\epsilon}\log\frac{1}{\delta})$ |
| $d$-维超矩形 | $\epsilon$ | $(\epsilon - \text{IR})_+$ | $\text{IR} \ge \epsilon$ | $O(\frac{1}{\epsilon}(d + \log\frac{1}{\delta}))$ |
| $d$-维半空间 | $\epsilon$ | 0 | — | $\tilde{O}(\frac{d + \log(1/\delta)}{r})$ |
| 图模型 | $\epsilon$ | 0 | — | $O(\frac{n \log n}{d_{\min}^+ + 1})$ |

### 三种学习模型的可学性分离

| 性质 | 标准 PAC | 策略性分类 | 带改进 PAC |
|------|---------|-----------|------------|
| 可学性等价条件 | 有限 VC 维 | 有限 SVC 维 | 无已知等价条件 |
| 零误差可达 | 否（非平凡类） | 否（一般） | 是（如阈值、交闭类） |
| 无限 VC 维 | 不可学 | 取决于 SVC | 可能可学 |
| 两区间并集类 | 可学 (VC=4) | 可学 (SVC≤4) | **不可学** (误差≥25%) |

### 关键发现

- 改进带来的误差减少量 $(\epsilon - r)_+$ 直接减去改进预算 $r$——比传统 PAC 的 $\epsilon$ 上界严格更紧
- 保守策略（偏向负分类）在改进场景中本质优于激进策略——假阴性可被 agent 自行修复，假阳性不可挽回
- 非交闭类（如两区间并集）在改进设置下反而更难学，因为假阳性会"引诱"agent 改进到错误区域
- 实验验证：3 个真实数据集 + 1 个合成数据集上，保守模型（SVM 高 FP 惩罚）初始误差较高但随 $r$ 增大急剧下降至零

## 亮点与洞察

- **零误差的可能性**：首次在学习理论中证明非平凡概念类可以零误差学习，前提是 agent 有真实改进能力。这一结果在标准 PAC 和策略性分类中都不可能
- **保守优于激进的反直觉结论**：传统学习中偏保守意味着更高的假阴性率，但在改进场景中假阴性可被 agent 自行修复——这改变了算法设计的基本原则
- **交闭性作为可学性的关键结构**：自然概念类（阈值、矩形、半空间交）恰好是交闭的，而非交闭类（区间并集）在改进下变得不可学——揭示了概念类代数结构与可学性的深层联系

## 局限性

- 仅研究可实现（realizable）设置，不可知（agnostic）设置下的推广是重要的开放问题
- 假设改进函数 $\Delta$ 已知，但实际中可能需要从数据中估计改进能力
- 对抗性 tie-breaking 可能过于悲观，放松该假设可能扩大可学类的范围
- 主要关注信息论（样本复杂度），未分析算法的计算效率

## 相关工作与启发

- **vs 策略性分类 (HMPW 2016; SVXY 2023)**：策略性分类中 agent "伪装"（不真正改进），可学性由 SVC 维刻画；本文中 agent 真正改进，可学性特征完全不同。Theorem 3.4 证明有限 SVC 不蕴含改进可学性
- **vs Kleinberg & Raghavan (2020)**：开创了改进 vs 操纵的区分，但主要关注激励设计；本文关注可学性和样本复杂度的基建问题
- **vs Haghtalab et al. (2020)**：从社会福利角度分析改进 agent 的学习，目标是最大化真阳性；本文以分类误差为目标，对假阳性更敏感，并证明两个目标可同时优化

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首建带改进PAC学习框架，发现零误差现象和三模型本质分离
- 实验充分度: ⭐⭐⭐ 理论为主，4个数据集实验验证基本结论
- 写作质量: ⭐⭐⭐⭐⭐ 定义精确，例子直观（贷款/阈值/矩形），证明结构清晰
- 价值: ⭐⭐⭐⭐ 为策略性ML和学习理论开辟新的基础研究方向

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] On Agnostic PAC Learning in the Small Error Regime](../../NeurIPS2025/others/on_agnostic_pac_learning_in_the_small_error_regime.md)
- [\[ICML 2025\] Learning-Augmented Hierarchical Clustering](learning-augmented_hierarchical_clustering.md)
- [\[ICML 2025\] Learning Distances from Data with Normalizing Flows and Score Matching](learning_distances_from_data_with_normalizing_flows_and_score_matching.md)
- [\[ICML 2025\] AutoAL: Automated Active Learning with Differentiable Query Strategy Search](autoal_automated_active_learning_with_differentiable_query_strategy_search.md)
- [\[ICML 2025\] SUICA: Learning Super-high Dimensional Sparse Implicit Neural Representations for Spatial Transcriptomics](suica_learning_super-high_dimensional_sparse_implicit_neural_representations_for.md)

<!-- RELATED:END -->
