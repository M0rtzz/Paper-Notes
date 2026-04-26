---
title: >-
  [论文解读] Exploration via Feature Perturbation in Contextual Bandits
description: >-
  [NeurIPS 2025][contextual bandits] 提出特征扰动（Feature Perturbation）作为上下文 bandit 的新型随机探索策略：直接在特征输入上注入噪声，而非扰动参数或奖励，从而在广义线性 bandit 中实现 $\tilde{O}(d\sqrt{T})$ 最优遗憾界，首次消除了随机化算法相较确定性方法的 $\sqrt{d}$ 因子劣势。
tags:
  - NeurIPS 2025
  - contextual bandits
  - feature perturbation
  - generalized linear bandits
  - regret bound
  - exploration
---

# Exploration via Feature Perturbation in Contextual Bandits

**会议**: NeurIPS 2025  
**arXiv**: [2510.17390](https://arxiv.org/abs/2510.17390)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: contextual bandits, feature perturbation, generalized linear bandits, regret bound, exploration

## 一句话总结

提出特征扰动（Feature Perturbation）作为上下文 bandit 的新型随机探索策略：直接在特征输入上注入噪声，而非扰动参数或奖励，从而在广义线性 bandit 中实现 $\tilde{O}(d\sqrt{T})$ 最优遗憾界，首次消除了随机化算法相较确定性方法的 $\sqrt{d}$ 因子劣势。

## 研究背景与动机

上下文 bandit 中的探索策略分为两大类：
- **确定性方法**（UCB/OFU）：理论最优 $\tilde{O}(d\sqrt{T})$ 遗憾，但实际过于保守
- **随机化方法**（Thompson Sampling, PHE）：实际表现优秀，但遗憾界为 $\tilde{O}(d^{3/2}\sqrt{T})$，多出 $\sqrt{d}$ 因子

Hamidi & Bayati (2021) 证明这一差距**并非分析松弛**，而是参数级随机化的固有代价：在 Thompson Sampling 框架下消除 $\sqrt{d}$ 因子会导致 $T$ 的线性依赖。

这引出核心问题：**是否存在随机化算法能达到 $\tilde{O}(d\sqrt{T})$ 最优遗憾？**

关键洞察：之前所有随机化方法要么扰动参数（TS），要么扰动奖励（PHE）。本文提出第三条路——**扰动特征**。这从根本上改变了随机化的"作用空间"，打破了固有限制。

## 方法详解

### 整体框架

**GLM-FP 算法**（Algorithm 1）流程：
1. 每轮 $t$：计算 MLE $\hat{\theta}_t$
2. 采样**单个**噪声向量 $\zeta_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$
3. 对每个臂 $i$ 的特征构造扰动：$\tilde{x}_{ti} = x_{ti} + c_t \cdot \frac{\|x_{ti}\|_{\hat{H}_t^{-1}}}{\|\hat{\theta}_t\|} \cdot \zeta_t$
4. 选择使 $\mu(\tilde{x}_{ti}^\top\hat{\theta}_t)$ 最大的臂

核心：所有臂共享同一噪声 $\zeta_t$，扰动幅度按每个臂的不确定性 $\|x_{ti}\|_{\hat{H}_t^{-1}}$ 自适应缩放。

### 关键设计

**为何特征扰动避免了 $\sqrt{d}$ 因子？** 直接对比 TS 和 FP 的扰动项：

- **TS扰动**：$|x_t^\top(\tilde{\theta}_t - \hat{\theta}_t)| = c_t \|x_{ti}\|_{V_t^{-1}} \cdot \|\zeta_t\|$ → 需要控制 $d$ 维高斯向量的范数
- **FP扰动**：$|(\tilde{x}_t - x_t)^\top\hat{\theta}_t| = c_t \|x_{ti}\|_{V_t^{-1}} \cdot |u^\top\zeta_t|$ → 只需控制 1 维投影

TS 需要对 $d$ 个坐标做 union bound → 引入 $\sqrt{d}$；FP 只涉及标量随机变量 $u^\top\zeta_t$ → 无需 union bound。

**加权 Gram 矩阵** $\hat{H}_t = \lambda\boldsymbol{I} + \nabla^2 L_t(\hat{\theta}_t)$：不同于以往使用普通 Gram 矩阵 $V_t$，本文使用包含链接函数曲率信息的加权版本，实现更精确的扰动控制。

**共享噪声设计**：所有臂使用同一 $\zeta_t$，消除了遗憾界对臂数 $K$ 的依赖。

### 损失函数 / 训练策略

**参数估计**：使用负对数似然的 MLE，线性情况用加权最小二乘，logistic 情况用 IRLS。

**调参策略**：$c_t = \beta_t(\delta')$，其中 $\beta_t$ 是置信宽度，$\delta' = \delta/(4T)$；正则化 $\lambda = O(d)$。

**证明的关键技术步骤**：
1. **置信集构造**（Lemma 1）：$\theta^* \in \{\theta: \|\theta - \hat{\theta}_t\|_{\hat{H}_t} \leq \beta_t(\delta)\}$
2. **集中性**（Lemma 2）：高概率下扰动特征保持在原特征附近
3. **随机乐观性**（Lemma 3）：$\Pr_t(\mu(\tilde{x}_t^\top\hat{\theta}_t) \geq \mu(x_{t*}^\top\theta^*)) \geq \frac{1}{4\sqrt{e\pi}}$ — 常数级下界
4. **遗憾分解**：$R(T) = \text{Reg}_{\text{FP}} + \text{Reg}_{\text{EST}}$，两部分各自有界

## 实验关键数据

### 主实验：线性 & Logistic Bandit

| 算法 | 线性 Bandit (d=10) | 线性 Bandit (d=50) | Logistic Bandit (d=10) | Logistic Bandit (d=50) |
|------|-------------------|-------------------|----------------------|----------------------|
| ε-greedy | 最高遗憾 | 最高遗憾 | 最高遗憾 | 最高遗憾 |
| UCB | 中等 | 中等 | 中等 | 中等 |
| TS | 中等偏低 | 中等 | 中等 | 中等偏高 |
| PHE | 中等偏低 | 中等 | 中等 | 中等偏高 |
| RandUCB | 接近最优 | 接近最优 | 中等偏低 | 中等偏低 |
| **GLM-FP** | **最低** | **最低** | **最低** | **最低** |

在所有测试维度和设置中，GLM-FP 一致达到最低遗憾。

### 维度依赖实验

| 维度 d | TS 终端遗憾 | FP 终端遗憾 | 比值 |
|--------|-----------|-----------|------|
| 10 | ~400 | ~200 | 2.0x |
| 20 | ~900 | ~400 | 2.25x |
| 30 | ~1500 | ~600 | 2.5x |
| 50 | ~3000 | ~1000 | 3.0x |

验证了 FP 遗憾随 $d$ 线性增长，而 TS 随 $d^{3/2}$ 增长。

### Neural Bandit 实验

| 数据集 | ε-greedy | NeuralUCB | NeuralTS | FTPL | **DeepFP** |
|--------|----------|-----------|----------|------|-----------|
| shuttle | 中等 | 中等偏低 | 中等偏低 | 中等 | **最低** |
| isolet | 高 | 中等 | 中等 | 中等 | **最低** |
| mushroom | 中等 | 低 | 低 | 中等 | **最低** |

DeepFP 在三个 UCI 数据集上均优于所有基线，展示了特征扰动在非参数模型上的可扩展性。

### 关键发现

1. **理论实证一致**：FP vs. TS 的遗憾比值随 $d$ 增大而增大，符合 $d$ vs $d^{3/2}$ 的理论预测
2. **GLM-FP 在 logistic 设置中优势更明显**：受益于使用加权 Gram 矩阵精确建模链接函数曲率
3. **Neural 扩展简单有效**：DeepFP 仅用 $\zeta_{ti} \sim \mathcal{N}(0, I/t)$ 的衰减噪声即可驱动探索，无需访问模型参数或梯度

## 亮点与洞察

- **概念简洁而深刻**：将探索的随机性从参数/奖励空间转移到特征空间，这一视角转换看似微小但理论影响深远
- **几何解释清晰**：TS 在 $V_t^{-1/2}$-变换空间中投影所有臂到共享随机方向，可能给已充分探索的臂分配大 bonus；FP 按不确定性独立缩放，系统性偏向未探索臂
- **实用性突出**：避免参数采样的计算开销，天然适用于神经网络等高维参数模型

## 局限与展望

- 理论分析限于广义线性模型，Neural bandit 的 DeepFP 缺乏理论保证
- 需要 $\|\hat{\theta}_t\|$ 参与缩放，当估计值接近零时可能不稳定
- 实验均为合成或 UCI 数据，缺乏大规模真实推荐/广告系统的验证
- 未探讨与 batched/delayed feedback 设置的兼容性

## 相关工作与启发

- **LinTS / GLM-TS**：经典参数随机化方法，$\tilde{O}(d^{3/2}\sqrt{T})$ 遗憾不可改进
- **RandUCB**：另一个 $\tilde{O}(d\sqrt{T})$ 随机方法，但缺乏实例依赖常数 $\kappa_*$
- **PHE**：奖励扰动方法，同样受限于 $\sqrt{d}$ 因子
- **Neural bandits**：NeuralUCB/NeuralTS 需梯度信息构造置信集，FP 仅需输入扰动，更加轻量

## 评分

- **新颖性**: 9/10 — 特征扰动视角新颖，理论突破显著
- **理论深度**: 9/10 — 严格证明最优遗憾界，几何解释精彩
- **实验充分度**: 8/10 — 合成+UCI 覆盖全面，缺乏大规模实验
- **实用性**: 8/10 — 实现简单且天然适配深度模型

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Tractable Multinomial Logit Contextual Bandits with Non-Linear Utilities](tractable_multinomial_logit_contextual_bandits_with_non-linear_utilities.md)
- [\[NeurIPS 2025\] Variance-Aware Feel-Good Thompson Sampling for Contextual Bandits](variance-aware_feel-good_thompson_sampling_for_contextual_bandits.md)
- [\[NeurIPS 2025\] Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown](feel-good_thompson_sampling_for_contextual_bandits_a_markov_chain_monte_carlo_sh.md)
- [\[NeurIPS 2025\] Improved Regret and Contextual Linear Extension for Pandora's Box and Prophet Inequality](improved_regret_and_contextual_linear_extension_for_pandoras_box_and_prophet_ine.md)
- [\[NeurIPS 2025\] Exploration with Foundation Models: Capabilities, Limitations, and Hybrid Approaches](exploration_with_foundation_models_capabilities_limitations_and_hybrid_approache.md)

<!-- RELATED:END -->
