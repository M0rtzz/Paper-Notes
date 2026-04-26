---
title: >-
  [论文解读] Unifying Formal Explanations: A Complexity-Theoretic Perspective
description: >-
  [ICLR 2026][优化][可解释AI] 提出统一框架将充分理由和对比理由（局部/全局、概率/非概率）归结为对统一概率值函数的最小化问题，揭示全局值函数具有单调性、子模性/超模性等组合优化关键性质，从而证明全局解释在多项式时间内可计算——即使对应的局部解释是 NP-hard 的。
tags:
  - ICLR 2026
  - 优化
  - 可解释AI
  - 计算复杂性
  - 充分理由
  - 对比理由
  - 子模/超模函数
---

# Unifying Formal Explanations: A Complexity-Theoretic Perspective

**会议**: ICLR 2026  
**arXiv**: [2602.18160](https://arxiv.org/abs/2602.18160)  
**领域**: 优化  
**关键词**: 可解释AI, 计算复杂性, 充分理由, 对比理由, 子模/超模函数

## 一句话总结

提出统一框架将充分理由和对比理由（局部/全局、概率/非概率）归结为对统一概率值函数的最小化问题，揭示全局值函数具有单调性、子模性/超模性等组合优化关键性质，从而证明全局解释在多项式时间内可计算——即使对应的局部解释是 NP-hard 的。

## 研究背景与动机

### 可解释性的两种基本形式

**充分理由（Sufficient Reasons）**：固定特征子集 $S$ 的值后，模型预测不变——回答"什么足以支持这个预测？"

**对比理由（Contrastive Reasons）**：修改特征子集 $S$ 的值后，模型预测改变——回答"什么改变可以翻转预测？"

越小的理由越有解释价值，因此**最小性**是核心追求。

### 从局部到全局，从确定到概率

literature 中的解释问题沿两个维度扩展：

- **局部 vs 全局**：针对单个预测 vs 针对模型在整个输入空间的行为
- **非概率 vs 概率**：绝对保证 vs 以概率 $\delta$ 保证

**已知困难结果**：

| 设置 | 充分理由 | 对比理由 |
|------|----------|----------|
| 决策树（局部非概率） | NP-hard | P |
| 神经网络（局部非概率） | $\Sigma_2^P$-hard | NP-hard |
| 决策树（局部概率） | NP-hard | NP-hard |

这些令人沮丧的复杂性结果引出核心问题：**是否存在某种解释形式在保持有意义保证的同时可以高效计算？**

## 方法详解

### 统一框架

所有解释问题统一为：

$$\text{找最小子集 } S \subseteq [n] \text{ 使得 } v(S) \geq \delta$$

其中 $v$ 是不同设置下的值函数：

- **局部充分值函数**：$v_{\text{suff}}^\ell(S) = \Pr_{\boldsymbol{z} \sim \mathcal{D}}(f(\boldsymbol{z}) = f(\boldsymbol{x}) \mid \boldsymbol{z}_S = \boldsymbol{x}_S)$
- **全局充分值函数**：$v_{\text{suff}}^g(S) = \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}[\Pr_{\boldsymbol{z}}(f(\boldsymbol{z}) = f(\boldsymbol{x}) \mid \boldsymbol{z}_S = \boldsymbol{x}_S)]$
- **局部对比值函数**：$v_{\text{con}}^\ell(S) = \Pr_{\boldsymbol{z} \sim \mathcal{D}}(f(\boldsymbol{z}) \neq f(\boldsymbol{x}) \mid \boldsymbol{z}_{\bar{S}} = \boldsymbol{x}_{\bar{S}})$
- **全局对比值函数**：$v_{\text{con}}^g(S) = \mathbb{E}_{\boldsymbol{x}}[\Pr_{\boldsymbol{z}}(f(\boldsymbol{z}) \neq f(\boldsymbol{x}) \mid \boldsymbol{z}_{\bar{S}} = \boldsymbol{x}_{\bar{S}})]$

### 三大关键性质

**1. 单调性**：$v(S \cup \{i\}) \geq v(S)$（添加特征不会降低值函数）

**2. 超模性**：$v(S \cup \{i\}) - v(S) \leq v(S' \cup \{i\}) - v(S')$（边际贡献递增）

**3. 子模性**：$v(S \cup \{i\}) - v(S) \geq v(S' \cup \{i\}) - v(S')$（边际贡献递减）

### 核心发现：局部与全局的惊人反差

| 性质 | 局部充分 $v_{\text{suff}}^\ell$ | 全局充分 $v_{\text{suff}}^g$ | 局部对比 $v_{\text{con}}^\ell$ | 全局对比 $v_{\text{con}}^g$ |
|------|:---:|:---:|:---:|:---:|
| 单调性 | ✗ | ✓ | ✗ | ✓ |
| 超模性 | ✗ | ✓（独立分布） | ✗ | ✗ |
| 子模性 | ✗ | ✗ | ✗ | ✓（独立分布） |

这些性质的发现是**反直觉的**：

- 局部值函数在所有设置下都不具备三大性质中的任何一个
- 全局值函数在所有设置下都满足单调性
- 充分理由的全局值函数是超模的，对比理由的是子模的——两者恰好"对偶"

### 贪心算法设计

**Algorithm 1（子集最小解释）**：自顶向下，从全特征集开始，每次移除使值函数降最少的特征：

$$j = \arg\max_{i \in S} v(S \setminus \{i\}), \quad S \leftarrow S \setminus \{j\}$$

单调性保证此算法在全局设置下收敛到子集最小解释，但在局部设置下不保证。

**Algorithm 2（基数最小解释近似）**：自底向上，从空集开始，每次加入使值函数增最多的特征：

$$j = \arg\max_{i \notin S} v(S \cup \{i\}), \quad S \leftarrow S \cup \{j\}$$

## 实验关键数据

本文为纯理论工作，核心贡献是复杂性分析结果。

### 复杂性结果总结

| 解释类型 | 局部（任意模型） | 全局（经验分布） |
|----------|:---:|:---:|
| 子集最小充分理由 | NP-hard（即使决策树） | **P** |
| 子集最小对比理由 | NP-hard（神经网络/树集成） | **P** |
| 基数最小充分理由近似 | 无有界近似 | $O\left(\frac{1}{1-k^f} + \ln\frac{v([n])}{\min_i v(\{i\})}\right)$ |
| 基数最小对比理由近似 | 无有界近似 | $O\left(\ln\frac{v([n])}{\min_i v(\{i\})}\right)$ |

### 近似保证对比

| 设置 | 全局 | 局部 |
|------|------|------|
| 子集最小 | 多项式时间精确解 | NP-hard（即使决策树+均匀分布） |
| 基数最小（对比） | $O(\ln |D|)$ 近似 | 无有界近似（即使 $|D|=1$） |
| 基数最小（充分） | 常数因子近似 | 无有界近似（即使 $|D|=1$） |

### 关键理论发现

1. 虽然非概率全局充分理由是唯一的（Bassan et al. 2024），但概率设置下可有指数多个子集最小全局充分理由：$\Theta(2^n/\sqrt{n})$
2. 全局对比理由的子模性可直接使用 Wolsey (1982) 经典贪心保证
3. 全局充分理由的超模性需要利用 Shi et al. (2021) 的有界曲率技术
4. 经验分布下的近似保证仅关于数据集大小 $|D|$ 对数增长

## 亮点与洞察

1. **统一框架极具优雅性**：将四大维度（充分/对比 × 局部/全局 × 概率/非概率 × 子集/基数最小）统一到一个值函数最小化任务中
2. **局部与全局的"相变"**：局部值函数完全缺乏结构（非单调、非子模、非超模），而全局值函数恰好具备所有有用性质——这不是巧合，而是期望运算的平滑效应
3. **充分与对比的对偶性**：一个超模一个子模，反映了"固定特征保持预测"与"变动特征改变预测"的对称关系
4. **实用意义深远**：全局解释可以高效计算意味着"这个模型在整体上依赖哪些特征"这个问题比"这个预测依赖哪些特征"更容易回答
5. **理论贡献涵盖可解释性谱系**：结果适用于决策树、神经网络和树集成，覆盖从"可解释"到"黑箱"的全谱模型

## 局限性

1. 纯理论工作，缺乏实验验证——贪心算法的实际运行效率和解释质量未知
2. 子模/超模性质仅在特征独立假设下成立，现实中特征往往高度相关
3. 经验分布的近似保证中，数据集大小 $|D|$ 作为隐含常数可能实际上很大
4. 全局解释的语义可能与用户真正关心的局部解释需求不完全匹配
5. 未讨论值函数计算本身的复杂度——对于神经网络，即使是单次推理也可能很慢

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 统一框架和局部/全局性质反差的发现极具原创性
- **实验**: ⭐⭐ — 纯理论工作，无实验支撑
- **写作**: ⭐⭐⭐⭐⭐ — 定义严谨、定理层次分明、证明路线清晰
- **价值**: ⭐⭐⭐⭐⭐ — 为可解释 AI 的复杂性理论奠定了重要基础

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit](neural_networks_learn_generic_multi-index_models_near_information-theoretic_limi.md)
- [\[ICML 2025\] Learning Mixtures of Experts with EM: A Mirror Descent Perspective](../../ICML2025/optimization/learning_mixtures_of_experts_with_em_a_mirror_descent_perspective.md)
- [\[ICML 2025\] Nearly Optimal Sample Complexity for Learning with Label Proportions](../../ICML2025/optimization/nearly_optimal_sample_complexity_for_learning_with_label_proportions.md)
- [\[ICML 2025\] Improved Sample Complexity for Private Nonsmooth Nonconvex Optimization](../../ICML2025/optimization/improved_sample_complexity_for_private_nonsmooth_nonconvex_optimization.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)

<!-- RELATED:END -->
