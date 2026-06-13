---
title: >-
  [论文解读] Transformer-Based Spatial-Temporal Counterfactual Outcomes Estimation
description: >-
  [ICML2025][因果推理][因果推断] 提出基于 Transformer 的时空反事实结果估计框架，利用 CNN 计算高维倾向性得分、Transformer 估计强度函数，在合成与真实数据上均优于传统因果推理方法。 现实世界的数据天然具有时间和空间两个维度（如冲突对森林损失的因果影响），因此需要在时空属性下估计反事实结…
tags:
  - "ICML2025"
  - "因果推理"
  - "因果推断"
  - "反事实结果估计"
  - "时空数据"
  - "Transformer"
  - "逆概率加权"
  - "空间点过程"
---

# Transformer-Based Spatial-Temporal Counterfactual Outcomes Estimation

**会议**: ICML2025  
**arXiv**: [2506.21154](https://arxiv.org/abs/2506.21154)  
**代码**: 有（论文附链接）  
**领域**: 因果推理  
**关键词**: 因果推断, 反事实结果估计, 时空数据, Transformer, 逆概率加权, 空间点过程

## 一句话总结

提出基于 Transformer 的时空反事实结果估计框架，利用 CNN 计算高维倾向性得分、Transformer 估计强度函数，在合成与真实数据上均优于传统因果推理方法。

## 研究背景与动机

现实世界的数据天然具有时间和空间两个维度（如冲突对森林损失的因果影响），因此需要在时空属性下估计反事实结果。然而现有因果推理框架存在明显不足：

- **Pearl/Rubin 框架**：不能直接处理时空属性的反事实结果估计
- **Christiansen et al. (2022)**：将结构因果模型扩展到时空数据，但当前结果主要受当前治疗影响，未充分解决**时间滞后效应** (temporal carryover effects)
- **Papadogeorgou et al. (2022)**：将潜在结果框架扩展到时空，但没有显式提出时空属性的倾向性得分计算方法，且采用核方法，面对复杂数据模式存在泛化困难

核心动机：需要一个能同时捕获空间相关性和时间滞后因果效应的深度学习框架。

## 方法详解

### 问题设定

将时空数据建模为**空间点模式的时间序列**。在每个时间步 $t$，治疗 $Z_t(s)$ 和结果 $Y_t(s)$ 都是二值变量，描述位置 $s$ 是否发生治疗/结果事件。整体结构可视为高维张量。

### 时空潜在结果框架

- 历史治疗集合：$Z_{\leq t} = \{Z_1, Z_2, \dots, Z_t\}$
- 潜在结果：$Y_t(Z_{\leq t})(s)$ 表示在历史治疗策略 $Z_{\leq t}$ 下，位置 $s$ 在时间 $t$ 的结果
- 估计目标：在反事实治疗分配策略 $F_H$ 下，区域 $\omega$ 内结果事件的期望数量

$$N_t^\omega(F_H) = \int_{Z^M} |S_{Y_t^{ob}(z_{\leq t}(F_H))} \cap \omega| \, dF_H(z_{[t-M+1,t]})$$

### 核心估计器（IPW）

基于逆概率加权 (Inverse Probability Weighting) 推导估计器：

$$\hat{Y}_t(F_H, s) = \prod_{j=t-M+1}^{t} \frac{p_{h_j}(z_j)}{e_j(z_j)} \cdot \lambda_{Y_t^{ob}(z_{\leq t})}(s)$$

其中：

- $e_j(z_j)$：倾向性得分，即给定历史信息下治疗的条件概率
- $p_{h_j}(z_j)$：反事实干预分布下的治疗概率
- $\lambda_{Y_t^{ob}}(s)$：观测结果的强度函数

### 倾向性得分计算（CNN）

高维治疗（$2^{100}$ 种可能取值）无法直接分类，因此引入**维度约减映射**：

$$R(Z_t) = |\{Z_t(s); Z_t(s) = 1, s \in \Omega\}|$$

将治疗映射为已治疗位置的计数（标量）。根据空间 Poisson 点过程假设，$R(Z_t)|h_{\leq t-1} \sim \text{Poisson}(\lambda_1)$，用 CNN 回归 $\lambda_1$（MSE loss），再由 Poisson PMF 计算倾向性得分：

$$e_t(R(z_t)) = \frac{\lambda_1^{R(z_t)}}{R(z_t)!} e^{-\lambda_1}$$

### 强度函数估计（Transformer）

用 Transformer 建模结果点过程的强度函数 $\lambda_{Y_t^{ob}}(s)$，训练目标为最大化似然：

$$\mathcal{L} = -\sum_{i=1}^{|S|} \ln(\text{net}(s_i)) + \int_\Omega \text{net}(s)\,ds - \text{KL}(q \| p)$$

- 第一项：事件位置处的对数似然
- 第二项：区域内强度积分（Poisson 似然的一部分）
- 第三项：KL 散度正则化（Transformer 编码器输出分布 $q$ 与标准高斯先验 $p$）

选择 Transformer 的理由：能捕获长程高阶依赖，计算效率优于 RNN。

### 理论保证

- **Proposition 1**：倾向性得分是平衡分数
- **Proposition 2**：倾向性得分具有维度约减性质
- **Proposition 3**：估计器具有**一致性**和**渐近正态性**：$\sqrt{T}(\hat{N}_\omega(F_H) - N_\omega(F_H)) \xrightarrow{d} \mathcal{N}(0, v)$

## 实验关键数据

### 合成实验

生成三组时间长度（T=32, 48, 64）的合成数据，每组实验独立运行 20 次。

| 方法 | 估计能力 |
|------|----------|
| **Ours (Transformer)** | 在所有 T 和 c 设定下 RER 最低 |
| MSMs | 中等偏高 RER |
| RMSNs | 中等偏高 RER |
| Causal Forest | 较高 RER |
| LR | 最高 RER |

结论：本文方法在不同时间长度和干预强度下均一致优于所有基线。

### 真实数据实验（哥伦比亚冲突→森林损失，2002-2022）

| M\c | c=3 | c=4 | c=5 | c=6 | c=7 |
|-----|-----|-----|-----|-----|-----|
| M=1 | 20.6±2.3 | 20.5±2.2 | 20.7±2.8 | 20.5±1.9 | 20.8±2.0 |
| M=3 | 21.5±1.4 | 21.6±2.4 | 22.3±1.9 | 23.0±1.3 | 23.3±1.9 |
| M=5 | 22.4±2.3 | 22.9±1.8 | 23.6±1.7 | 24.2±1.2 | 24.7±2.2 |
| M=7 | 24.7±1.3 | 23.6±1.7 | 26.7±1.2 | 27.2±1.5 | 28.0±2.1 |

结论：冲突持续时间（M）和强度（c）增加均导致森林损失增加，与已有文献一致。

### 消融实验

- **Transformer vs RNN**：RNN 估计能力明显弱于 Transformer
- **放松 Poisson 假设**：加入 Gaussian kernel 打破标准 Poisson 设定后，性能没有显著下降，验证了方法的鲁棒性

## 亮点与洞察

1. **巧妙的维度约减**：将高维点模式治疗映射为标量计数，使原本不可行的倾向性得分计算变得可行
2. **完整的理论支撑**：估计器具有一致性和渐近正态性，不仅仅是"用深度学习堆效果"
3. **CNN + Transformer 分工合理**：CNN 处理时空高维数据提取局部特征回归倾向性得分，Transformer 捕获长程依赖估计强度函数
4. **鲁棒性好**：放松 Poisson 假设后性能不降，实用性强
5. **真实实验有意义**：哥伦比亚冲突与森林损失的因果分析具有环境科学价值

## 局限与展望

1. **Poisson 点过程假设**：虽然消融实验表明放松后性能尚可，但更复杂的真实数据可能违反该假设（如聚集性点过程）
2. **仅处理二值治疗**：每个位置治疗为 0/1，未推广到连续或多值治疗
3. **无混杂假设**：要求无未观测混杂，在观察性时空数据中很难完全满足
4. **真实实验无 ground truth**：只能通过与已有文献的一致性来间接验证，缺乏定量评估
5. **可扩展性**：论文提到最大维度为 (100,100,192)，对于更大规模的遥感/城市数据可能面临计算瓶颈
6. **单一应用场景**：仅在冲突-森林数据上验证，缺乏跨领域泛化实验

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将 Transformer 嵌入时空因果推理框架，维度约减策略巧妙
- 实验充分度: ⭐⭐⭐⭐ — 合成+真实+消融覆盖全面，但真实实验缺乏定量 ground truth
- 写作质量: ⭐⭐⭐⭐ — 符号定义清晰，理论推导完整，但公式密度较高
- 价值: ⭐⭐⭐⭐ — 填补了深度学习时空因果推理的空白，框架通用性好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models](../../ACL2025/causal_inference/counterfactual-consistency_prompting_for_relative_temporal_understanding_in_larg.md)
- [\[ICLR 2026\] An Orthogonal Learner for Individualized Outcomes in Markov Decision Processes](../../ICLR2026/causal_inference/an_orthogonal_learner_for_individualized_outcomes_in_markov_decision_processes.md)
- [\[NeurIPS 2025\] LLM Interpretability with Identifiable Temporal-Instantaneous Representation](../../NeurIPS2025/causal_inference/llm_interpretability_with_identifiable_temporal-instantaneous_representation.md)
- [\[NeurIPS 2025\] Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features](../../NeurIPS2025/causal_inference/causality-induced_positional_encoding_for_transformer-based_representation_learn.md)
- [\[NeurIPS 2025\] Do-PFN: In-Context Learning for Causal Effect Estimation](../../NeurIPS2025/causal_inference/do-pfn_in-context_learning_for_causal_effect_estimation.md)

</div>

<!-- RELATED:END -->
