---
title: >-
  [论文解读] Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data
description: >-
  [NeurIPS 2025 Spotlight][优化/理论][implicit bias] 本文首次完整刻画了归一化最速下降（NSD）和归一化动量最速下降（NMD）在多分类线性可分数据上的隐式偏差：这些算法以 $\mathcal{O}(1/\sqrt{t})$ 的速率收敛到相应 $p$-范数的最大 margin 解，涵盖 Spectral Descent（谱范数）和 Muon 作为特例，并扩展至 Adam（max-范数 margin）。
tags:
  - "NeurIPS 2025 Spotlight"
  - "优化/理论"
  - "implicit bias"
  - "spectral descent"
  - "Muon"
  - "margin maximization"
  - "multiclass classification"
  - "Schatten norm"
---

# Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2502.04664](https://arxiv.org/abs/2502.04664)  
**代码**: 无  
**领域**: 优化  
**关键词**: implicit bias, spectral descent, Muon, margin maximization, multiclass classification, Schatten norm

## 一句话总结
本文首次完整刻画了归一化最速下降（NSD）和归一化动量最速下降（NMD）在多分类线性可分数据上的隐式偏差：这些算法以 $\mathcal{O}(1/\sqrt{t})$ 的速率收敛到相应 $p$-范数的最大 margin 解，涵盖 Spectral Descent（谱范数）和 Muon 作为特例，并扩展至 Adam（max-范数 margin）。

## 研究背景与动机
**领域现状**：Adam/AdamW 是大语言模型训练的事实标准优化器。Muon 通过 Newton-Schulz 迭代对更新进行正交化（即谱下降），在 NanoGPT 上展示了优异性能，近期已扩展到大规模 LLM 训练。

**理论缺口**：Spectral Descent / Muon 的理论研究主要关注收敛速率（如非凸设定下梯度范数的下降速率），但对其隐式偏差——即在过参数化模型中偏好哪种解——缺乏分析。

**核心问题**：在多分类线性可分数据 + 交叉熵损失下，Spectral Descent 及其动量变体的隐式偏差是什么？

**多分类设定的丰富性**：多分类中参数是矩阵而非向量，自然地容纳了 Schatten 范数族（谱范数、核范数、Frobenius 范数），使得分析比二分类更丰富且更适合研究谱下降类算法。

## 方法详解

### 问题设定
- 多分类线性模型：$W \in \mathbb{R}^{k \times d}$，输入 $h_i \in \mathbb{R}^d$，标签 $y_i \in [k]$
- 交叉熵损失：$\mathcal{L}(W) = -\frac{1}{n} \sum_{i} \log \mathbb{S}_{y_i}(W h_i)$
- 最大 margin：$\gamma := \max_{\|W\| \leq 1} \min_{i, c \neq y_i} (e_{y_i} - e_c)^\top W h_i$

### 归一化最速下降（NSD）
更新方向为：

$$\Delta_t := \arg\max_{\|\Delta\| \leq 1} \langle \nabla_t, \Delta \rangle$$

- **max-范数** → SignGD：$\Delta_t = \text{sign}(\nabla_t)$
- **Frobenius 范数** → NGD：$\Delta_t = \nabla_t / \|\nabla_t\|_2$
- **谱范数** → Spectral-GD：$\Delta_t = U_t V_t^\top$（$\nabla_t = U_t \Sigma_t V_t^\top$）

### 归一化动量最速下降（NMD）
同 NSD 但对动量 $M_t$ 而非梯度取最速方向：

$$M_t = \beta_1 M_{t-1} + (1-\beta_1) \nabla_t, \quad \Delta_t := \arg\max_{\|\Delta\| \leq 1} \langle M_t, \Delta \rangle$$

- **谱范数** → **Muon**：$\Delta_t = \tilde{U}_t \tilde{V}_t^\top$（对 $M_t$ 做 SVD）

### 统一分析框架：代理函数 $\mathcal{G}(W)$

**核心创新**：构造代理函数统一处理所有 NSD/NMD 变体：

$$\mathcal{G}(W) := \frac{1}{n} \sum_{i \in [n]} (1 - \mathbb{S}_{y_i}(W h_i))$$

**关键性质**：
1. **梯度对偶范数下界**（Lemma 1）：$\|\nabla \mathcal{L}(W)\|_* \geq \gamma \cdot \mathcal{G}(W)$
2. **二阶项上界**（Lemma 2）：Hessian 项可由 $\mathcal{G}(W) \cdot \|\Delta\|^2$ 控制
3. **与损失的近似等价**（Lemma 3）：当损失足够小时 $\mathcal{L}(W) \leq 2\mathcal{G}(W)$

统一性来自范数排序关系：$\|A\|_{\max} \leq \|||A|||_p \leq \|A\|_{\text{sum}}$（对所有 entry-wise 和 Schatten $p$-范数）。

### 主要定理

**Theorem 1（NSD margin 收敛）**：学习率 $\eta_t = \Theta(1/\sqrt{t})$，NSD 的 margin gap 满足：

$$\gamma - \frac{\min_{i,c \neq y_i} (e_{y_i} - e_c)^\top W_t h_i}{\|W_t\|} \leq \mathcal{O}\left(\frac{\log t + n}{\sqrt{t}}\right)$$

**Theorem 2（NMD margin 收敛）**：NMD（含 Muon）的 margin gap 为 $\mathcal{O}\left(\frac{d\log t + dn}{\sqrt{t}}\right)$。

**Adam 扩展**：Adam（无 $\epsilon$ 常数）收敛到 max-范数最大 margin 解，速率 $\mathcal{O}\left(\frac{d\log t + nd}{t^{1/3}}\right)$。

### NMD 分析关键：逐类代理函数分解
定义逐类代理函数 $\mathcal{G}_c(W)$ 和 $\mathcal{Q}_c(W)$，利用 softmax 的良好性质控制动量与梯度差 $\Omega_t = M_t - \nabla_t$ 的 sum-范数：

$$\|\Omega_t\|_{\text{sum}} \leq 2B\beta_1^{t/2} \mathcal{G}(W_t) + 2\alpha_M d \eta_t \mathcal{G}(W_t)$$

避免了朴素分析中额外的 $k$ 因子。

## 实验关键数据

### 合成数据实验（$k=10$, $d=25$, 每类 50 样本）

| 算法 | 偏好的 margin 范数 | 与 $V_\infty$ 相关性 | 与 $V_2$ 相关性 | 与 $V_{\text{spec}}$ 相关性 |
|------|-------------------|---------------------|----------------|--------------------------|
| SignGD | max-范数 ✅ | 高 | 低 | 低 |
| NGD | 2-范数 ✅ | 低 | 高 | 低 |
| Spectral-GD | 谱范数 ✅ | 低 | 低 | 高 |
| Muon | 谱范数 ✅ | 低 | 低 | 高 |
| Signum | max-范数 ✅ | 高 | 低 | 低 |
| NMD-GD | 2-范数 ✅ | 低 | 高 | 低 |

### 两层神经网络实验（MNIST, 100 层隐维度）

| 设定 | 观察结果 |
|------|---------|
| 仅训练第一层 | Spectral-GD 和 Muon 的谱范数 margin $\gamma_a^V$ 增长最快 |
| 联合训练两层 | Spectral-GD 和 Muon 的 $\gamma_b^{V,W}$ 仍增长最快 |
| 对比 SGD 系列 | SignGD/NGD 的谱范数 margin 增长明显慢于 Spectral-GD/Muon |

**关键结论**：线性设定下的范数偏好趋势在非线性设定中同样存在。

## 亮点与洞察
- **首个非渐近 Spectral-GD/Muon 隐式偏差结果**：在多分类可分数据上建立了具体的 $\mathcal{O}(1/\sqrt{t})$ margin 收敛速率
- **统一框架的优雅性**：通过范数排序 + 代理函数 $\mathcal{G}(W)$，一个分析覆盖所有 entry-wise 和 Schatten 范数族
- **逐类分解技巧**：避免了 NMD 分析中 $k$ 因子的出现，不仅简化了证明也改进了结果
- **实践意义**：为理解 Muon/Shampoo 等新兴优化器为何在 LLM 训练中表现优异提供了理论基础——它们隐式偏好谱范数 margin 最大化

## 局限与展望
- **NMD 速率中有额外 $d$ 因子**：Theorem 2 比 Theorem 1 多一个 $d$，能否去除是开放问题
- **Adam 的速率仅为 $\mathcal{O}(t^{-1/3})$**：慢于 NSD/NMD 的 $\mathcal{O}(t^{-1/2})$，有空间改进
- **仅限线性多分类**：扩展到同质/非同质神经网络是重要方向
- **未考虑非可分数据**：实际 LLM 训练数据不满足线性可分假设
- **Muon 实现差异**：实际 Muon 使用 Newton-Schulz 近似 SVD，理论分析假设精确 SVD

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 Muon/Spectral Descent 纳入统一隐式偏差框架，填补了重要空白
- 理论深度: ⭐⭐⭐⭐⭐ 代理函数构造和范数排序技巧都非常精巧
- 实验充分度: ⭐⭐⭐⭐ 合成数据 + 非线性扩展，充分验证了理论预测
- 写作质量: ⭐⭐⭐⭐ 结构清晰，表格总结有效，但符号较多需要耐心阅读
- 价值: ⭐⭐⭐⭐ 对理解现代优化器（Muon, Shampoo, Adam）的隐式偏差有重要贡献

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [\[NeurIPS 2025\] The Implicit Bias of Structured State Space Models Can Be Poisoned With Clean Labels](the_implicit_bias_of_structured_state_space_models_can_be_poisoned_with_clean_la.md)
- [\[NeurIPS 2025\] A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias](a_unified_stability_analysis_of_sam_vs_sgd_role_of_data_cohe.md)
- [\[NeurIPS 2025\] Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)
- [\[ICML 2025\] How Transformers Learn Regular Language Recognition: A Theoretical Study on Training Dynamics and Implicit Bias](../../ICML2025/optimization/how_transformers_learn_regular_language_recognition_a_theoretical_study_on_train.md)

</div>

<!-- RELATED:END -->
