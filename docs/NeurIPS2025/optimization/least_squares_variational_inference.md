---
title: >-
  [论文解读] Least Squares Variational Inference
description: >-
  [NeurIPS 2025][优化][variational inference] 提出 LSVI（Least Squares Variational Inference），一种无梯度、基于普通最小二乘回归的变分推断方法，在指数族内通过对温控 log-target 做 OLS 回归来迭代求解最优变分近似，对高斯族有高效的 $O(d^3)$（全协方差）或 $O(d)$（平均场）实现。
tags:
  - NeurIPS 2025
  - 优化
  - variational inference
  - natural gradient descent
  - exponential family
  - least squares
  - gradient-free
---

# Least Squares Variational Inference

**会议**: NeurIPS 2025  
**arXiv**: [2502.18475](https://arxiv.org/abs/2502.18475)  
**代码**: https://github.com/ylefay/LSVI (有)  
**领域**: Optimization / Bayesian Inference  
**关键词**: variational inference, natural gradient descent, exponential family, least squares, gradient-free

## 一句话总结

提出 LSVI（Least Squares Variational Inference），一种无梯度、基于普通最小二乘回归的变分推断方法，在指数族内通过对温控 log-target 做 OLS 回归来迭代求解最优变分近似，对高斯族有高效的 $O(d^3)$（全协方差）或 $O(d)$（平均场）实现。

## 研究背景与动机

变分推断（VI）是概率机器学习的核心工具，其目标是在参数族 $\mathcal{Q}$ 中找到与目标分布 $\pi$ KL 散度最小的近似分布。当前主流方法依赖梯度优化（SGD 或自然梯度下降 NGD），实现在 STAN、NumPyro、PyMC3 等软件中。

**现有痛点**：

**梯度依赖**：标准方法需要 $\log \pi$ 可自动微分，或依赖重参数化技巧。但在多种重要场景下不可用：离散分布、不可微的 $\pi$、似然无法解析计算（如 likelihood-free inference）

**方差问题**：不使用重参数化技巧时，log-derivative trick 的梯度估计方差极高

**调参困难**：SGD 收敛慢且需精心调节步长，NGD 的朴素实现需要昂贵的 Fisher 信息矩阵求逆

**Fisher 矩阵规模**：高斯族中 Fisher 矩阵大小与维度 $d$ 平方成正比，直接求逆代价为 $O(m^3)$，其中 $m = O(d^2)$

**核心 idea**：利用指数族的数学结构，将 uKL（非归一化 KL）最小化的一阶优化条件转化为一个不动点方程 $\eta = \phi(\eta)$，而 $\phi(\eta)$ 恰好是 $f(X)$ 对 $s(X)$ 的 OLS 回归系数（$X \sim q_\eta$）。这样每步迭代只需做一次普通最小二乘回归，完全不需要目标函数的梯度。

## 方法详解

### 整体框架

LSVI 的迭代流程：
1. 从当前近似 $q_{\hat\eta_t}$ 中采样 $X_1, ..., X_N$
2. 计算 Monte Carlo 估计的 $\hat{F}$ 和 $\hat{z}$
3. 求解 OLS 得到 $\hat\eta'_{t+1} = \hat{F}^{-1} \hat{z}$
4. 动量松弛更新 $\hat\eta_{t+1} = \varepsilon_t \hat\eta'_{t+1} + (1 - \varepsilon_t) \hat\eta_t$

### 关键设计

1. **精确 LSVI 映射与不动点迭代**：

    - **功能**：将 VI 转化为不动点迭代问题
    - **核心思路**：uKL 最小化的一阶条件等价于 $\{\mathbb{E}_\eta[ss^\top]\}\eta = \mathbb{E}_\eta[fs]$，即 $\eta = F_\eta^{-1} z_\eta$。这恰好是以 $s(X)$ 为回归量、$f(X)$ 为响应变量的 OLS 解
    - **动量松弛**：采用 $\eta_{t+1} = \varepsilon_t \phi(\eta_t) + (1-\varepsilon_t) \eta_t$ 防止迭代超出自然参数空间 $\mathcal{V}$。松弛系数 $\varepsilon_t$ 对应对温控密度 $q_{\eta_t}^{1-\varepsilon_t} \pi^{\varepsilon_t}$ 做回归
    - **设计动机**：当目标在变分族内时 $\phi$ 能一步恢复精确解，这是其他方法不具备的优美性质

2. **LSVI 与自然梯度 / 镜像下降的等价关系**：

    - **功能**：建立理论收敛保证
    - **核心思路**：证明 LSVI 迭代 (5) 等价于自然参数空间的自然梯度下降 $\eta_{t+1} = \eta_t - \varepsilon_t F_{\eta_t}^{-1} \nabla_\eta l(\eta_t) / Z_{\eta_t}$，也等价于矩参数空间的镜像下降
    - **收敛速率**：在 $L$-光滑、$\mu$-强凸假设下，收敛率为 $O(k^{-\mu/\alpha}) + O(N^{-1})$，其中 $k$ 为迭代次数，$N$ 为采样数。当 $\alpha = \mu$ 时达到最优 $O(k^{-1}) + O(N^{-1})$ 速率

3. **高斯族的高效重参数化**：

    - **功能**：消除 Fisher 矩阵求逆，大幅降低计算复杂度
    - **核心思路**：对全协方差高斯，将 $f(X)$ 对 $s(X)$ 的回归重参数化为 $f(\mu + CZ)$ 对 $t(Z)$ 的回归（$Z \sim N(0,I)$），其中 $C = \text{Chol}(\Sigma)$。精心构造 $t(z)$ 使得 $\mathbb{E}[t(Z)t(Z)^\top] = I$，从而 OLS 估计变为简单的 $\hat\gamma = N^{-1} \sum_i t(Z_i) f(\mu + CZ_i)$，无需矩阵求逆
    - **复杂度**：全协方差 $O(d^3)$（由 Cholesky 分解主导），平均场 $O(d)$
    - **Theorem 4.1** 给出了从 $\gamma$ 到 $\eta$ 的显式递推公式

4. **自适应步长选择**：

    - **功能**：自动确定合适的松弛步长 $\varepsilon_t$
    - **核心思路**：观察到步长 $\varepsilon$ 会将回归残差方差缩小 $\varepsilon^2$ 倍。设定残差方差上界 $u^2$，取 $\varepsilon \leq u/v$（$v$ 为当前残差标准差），结合回溯搜索确保迭代不超出参数空间
    - **设计动机**：光滑性和强凸性参数通常未知，固定步长要么不稳定要么过慢

### 损失函数 / 训练策略

- 使用非归一化 KL 散度（uKL）作为优化目标，与标准 KL 的最优解一致（Proposition 2.2）
- 每步迭代成本：通用 LSVI 为 $O(m^3 + m^2 N)$，高斯 LSVI-FC 为 $O(d^3 + dN)$，高斯 LSVI-MF 为 $O(d + dN)$

## 实验关键数据

### 主实验

**逻辑回归 (Pima 数据集, 全协方差)**：

| 方法 | 收敛速度 | 特点 |
|------|---------|------|
| LSVI (Algorithm 1) | ~1 步 | 本质上一步收敛，但需要 Fisher 矩阵求逆 |
| LSVI-FC (Algorithm 3) | <100 步 | 高效，$O(d^3)$ |
| NGD | ~100 步 | 需要自动微分 |
| ADVI (pyMC3/Blackjax) | >100 步 | 需要步长调节 |
| GMMVI | ~100 步 | 无梯度但仅适合低维 |

**MNIST 逻辑回归 (平均场)**：LSVI-MF 在时间效率上优于 ADVI 和 NGD。

**变量选择 (离散分布, Bernoulli 族)**：

| 方法 | 适用性 | 结果 |
|------|--------|------|
| LSVI (Algorithm 1) | 适用于离散族 | 后验边际概率与 SMC 精确推断吻合 |
| ADVI | 不适用（需重参数化） | - |
| SGD | 不适用（需梯度） | - |

首次在 Bernoulli 乘积族上实现变分推断。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| LSVI 线性递减步长 | KL ~$O(1/k)$ | 标准收敛 |
| LSVI 自适应步长 | 更快收敛 | 残差控制策略有效 |
| 两组独立样本 vs OLS | OLS 方差更低 | OLS 联合估计更高效 |
| LSVI-FC vs LSVI 通用 | LSVI-FC 在高维更优 | 避免 Fisher 矩阵求逆 |

**贝叶斯合成似然 (BSL, 蟾蜍位移模型)**：
- LSVI-FC 的变分近似与 MCMC 后验高度吻合
- CPU 成本远低于 MCMC（不需要多次运行数据模拟器）

### 关键发现

- LSVI 在目标分布属于变分族时能一步恢复精确解
- 与 NGD/ADVI 相比，LSVI 的迭代噪声更低（因 OLS 是最优估计量）
- 在梯度不可用的场景（离散分布、BSL）中，LSVI 是唯一可行的 VI 方法

## 亮点与洞察

1. **将 VI 转化为回归问题**：这一视角极具优雅性，将复杂的优化问题映射到经典的最小二乘框架
2. **理论完备性**：建立了 LSVI ↔ NGD ↔ 镜像下降的完整等价关系，并给出了有条件的收敛速率
3. **高斯族的高效实现**：通过重参数化消除 Fisher 矩阵求逆，是关键的实践贡献
4. **无梯度 + 梯度友好**：既能在无梯度场景下工作，也能与子采样等技术结合处理大数据

## 局限与展望

- 目前限于指数族，混合指数族的扩展有待探索
- 如果后验在某些方向上强烈非高斯，高斯近似效果有限
- 离散指数族中独立性假设的限制可通过树结构依赖来放松
- 强凸/光滑假设在实践中不一定成立，但局部凸性通常足够

## 相关工作与启发

- **Salimans & Knowles (2013) 的 OLS 视角**：本文的理论基础源于此，但进一步建立了与 NGD 的等价性并给出了高效实现
- **Khan & Rue (2023) 的贝叶斯学习规则**：本文属于自然梯度 VI 的大框架，但强调无梯度场景
- **启发**：对于需要进行似然无法求解的贝叶斯推断（如模拟器模型），LSVI 提供了一个实用且有理论保证的方案

## 评分

- 新颖性: ⭐⭐⭐⭐ OLS-VI 视角虽非全新，但高效实现和完整理论分析是重要贡献
- 实验充分度: ⭐⭐⭐⭐ 覆盖了可微/离散/不可微三种场景，但大规模实验有限
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨，结构清晰，理论与实践结合好
- 价值: ⭐⭐⭐⭐ 为无梯度变分推断提供了优雅且实用的解决方案

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Brain-like Variational Inference](brain-like_variational_inference.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference](neuro-symbolic_entity_alignment_via_variational_inference.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)
- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)

<!-- RELATED:END -->
