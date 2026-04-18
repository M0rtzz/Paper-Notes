---
title: >-
  [论文解读] PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression
description: >-
  [ICLR 2026][Shapley值] 本文提出 PolySHAP，通过将 KernelSHAP 的线性近似扩展为高阶多项式回归来捕获特征间的非线性交互，从而提升 Shapley 值的估计精度；并从理论上证明了配对采样（paired sampling）等价于二阶 PolySHAP，首次解释了配对采样启发式方法优越性能的根本原因。
tags:
  - ICLR 2026
  - Shapley值
  - 可解释AI
  - 多项式回归
  - 特征交互
  - KernelSHAP
---

# PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression

**会议**: ICLR 2026  
**arXiv**: [2601.18608](https://arxiv.org/abs/2601.18608)  
**代码**: [GitHub](https://github.com/FFmgll/PolySHAP)  
**领域**: 其他  
**关键词**: Shapley值, 可解释AI, 多项式回归, 特征交互, KernelSHAP

## 一句话总结

本文提出 PolySHAP，通过将 KernelSHAP 的线性近似扩展为高阶多项式回归来捕获特征间的非线性交互，从而提升 Shapley 值的估计精度；并从理论上证明了配对采样（paired sampling）等价于二阶 PolySHAP，首次解释了配对采样启发式方法优越性能的根本原因。

## 研究背景与动机

Shapley 值是可解释 AI 中最核心的博弈论工具之一，用于量化各特征对模型预测的贡献。然而，对于 $d$ 个特征的模型，精确计算 Shapley 值需要 $2^d$ 次博弈评估，计算代价极高。KernelSHAP 通过将博弈函数 $\nu$ 近似为线性函数来规避指数级开销，但线性近似本质上无法捕获特征间的非线性交互效应，限制了估计精度。

此外，配对采样（paired sampling）作为一种广泛使用的启发式策略，能显著提升 KernelSHAP 的估计质量，但其优越性能背后的理论机制一直未被充分理解。本文从多项式回归的角度，为以上两个问题提供了统一的理论框架和实践方案。

## 方法详解

### 整体框架

PolySHAP 的核心思路是将 KernelSHAP 的线性函数近似扩展为高阶多项式，用交互项来捕获特征间的非线性关系。具体步骤为：
1. 定义交互前沿 $\mathcal{I}$，选择需要建模的交互项集合
2. 通过带权最小二乘拟合多项式
3. 利用理论公式将 PolySHAP 表示转换回 Shapley 值

### 关键设计

1. **PolySHAP 交互表示**: 将博弈函数近似为包含交互项的多项式，定义 PolySHAP 表示 $\phi^{\mathcal{I}} \in \mathbb{R}^{d'}$（$d' = d + |\mathcal{I}|$），通过加权最小二乘求解。核心公式为：
$$\phi^{\mathcal{I}}[\nu] := \arg\min_{\phi \in \mathbb{R}^{d'}: \langle\phi,\mathbf{1}\rangle = \nu(D)} \sum_{S \subseteq D} \mu(S)\left(\nu(S) - \sum_{T \in D \cup \mathcal{I}} \phi_T \prod_{j \in T} \mathbb{1}[j \in S]\right)^2$$
然后通过定理 4.3 将 PolySHAP 表示转换为 Shapley 值：$\phi_i^{SV}[\nu] = \phi_i^{\mathcal{I}} + \sum_{S \in \mathcal{I}: i \in S} \frac{\phi_S^{\mathcal{I}}}{|S|}$。设计动机：更表达力强的多项式能更准确地近似博弈函数，从而产生更精确的 Shapley 值估计。

2. **配对采样等价性定理 (Theorem 5.1)**: 证明了在配对采样条件下（同时采样 $S$ 和 $D \setminus S$），KernelSHAP 的输出**精确等于**二阶 PolySHAP（2-PolySHAP）的输出，即配对 KernelSHAP 隐式地捕获了所有二阶交互。这首次从理论上解释了配对采样为何能大幅提升估计精度。设计动机：为实践中广泛使用的配对采样启发式方法提供理论支撑。

3. **$k$-可加交互前沿**: 定义 $\mathcal{I}_{\leq k} = \{S \subseteq D : 2 \leq |S| \leq k\}$，逐阶扩展交互项（$k$-PolySHAP）。$k=1$ 退化为 KernelSHAP，$k=2$ 含所有二阶交互。对于高维场景，引入**部分交互前沿** $\mathcal{I}_\ell$，在计算预算不足以支撑完整 $k$ 阶交互时，选择性地加入部分高阶项。

4. **杠杆分数采样**: 采用 leverage score sampling 策略，按杠杆分数而非 Shapley 权重进行子集采样，这在 $m = O(d' \log(d'/\delta) + d'/({\epsilon\delta}))$ 的预算下，能以 $1-\delta$ 的概率保证近似质量。

### 损失函数 / 训练策略

PolySHAP 求解的是带约束的加权最小二乘问题，约束条件为 Shapley 值加和等于 $\nu(D)$（efficiency 性质）。通过投影矩阵 $\mathbf{P}_{d'}$ 将约束问题转化为无约束问题求解。采用 border trick 对小尺寸子集进行穷举而非采样。

## 实验关键数据

### 主实验

在 15 个不同的解释博弈上（涵盖表格、图像、语言等领域，$d$ 从 8 到 101），对比 PolySHAP 与多种基线方法。

| 数据集/博弈 | 指标 | PolySHAP (3阶) | KernelSHAP | 改进 |
|-------------|------|---------------|------------|------|
| Housing ($d=8$) | MSE | 最优 | 基线 | 显著降低 MSE |
| Adult ($d=14$) | MSE | 最优 | 基线 | 显著降低 MSE |
| Estate ($d=15$) | MSE | 最优 | 基线 | 显著降低 MSE |
| Cancer ($d=30$) | MSE | 最优 | 基线 | 显著降低 MSE |
| CG60 ($d=60$) | MSE | 小幅改进 | 基线 | 仅小幅提升（高维限制） |

### 消融实验

| 配置 | 关键指标 (MSE) | 说明 |
|------|---------------|------|
| 1-PolySHAP (= KernelSHAP) | 基线 | 无交互项 |
| 2-PolySHAP | 显著改进 | 加入所有二阶交互 |
| 2-PolySHAP (50%) | 中等改进 | 仅加入 50% 二阶交互 |
| 3-PolySHAP | 最优 | 加入三阶交互，低维场景增益最大 |
| 配对 KernelSHAP vs 配对 2-PolySHAP | 完全一致 | 实验验证定理 5.1 |
| 配对 3-PolySHAP vs 配对 4-PolySHAP | 几乎一致 | 暗示更高阶存在类似等价关系 |

### 关键发现

- 加入任意数量的交互项都能改善 Shapley 值近似质量
- 在配对采样下，KernelSHAP 自动获得了 2-PolySHAP 的性能，因此实践中 PolySHAP 的增益从三阶交互开始体现
- 高维场景（$d \geq 60$）中，可加入的三阶交互数量有限，增益较小
- RegressionMSR 是唯一能与 PolySHAP 媲美的基线，但它依赖 XGBoost 树模型，在某些博弈上表现不稳定

## 亮点与洞察

- **理论贡献重大**: 配对采样等价于 2-PolySHAP 的发现是一个优美的理论结果，解答了长期以来的实践困惑
- **方法自然优雅**: 从线性到多项式的扩展思路简洁，且保持了一致性保证
- **统一视角**: 将 KernelSHAP、Faith-SHAP、$k_{ADD}$-SHAP 等方法纳入统一框架
- **投影引理**: 提出的技术性投影引理（Lemma A.1）在证明多个定理中发挥关键作用

## 局限与展望

- 高维场景中三阶交互项组合数爆炸（$\binom{d}{3}$），实际可加入的交互项有限
- 猜测配对 $k$-PolySHAP 等价于 $(k+1)$-PolySHAP（$k$ 为奇数）但未能证明
- 交互前沿的选择较为通用（按阶全加），未利用特定问题的交互结构信息
- 运行时间分析较为理论化，大规模实际应用中的效率有待验证

## 相关工作与启发

- 与 RegressionMSR（Witter et al., 2025）对比：PolySHAP 无需额外回归调整步骤即可保持一致性
- 与 $k_{ADD}$-SHAP（Pelegrina et al., 2023）关系：PolySHAP 简化并推广了其收敛性证明
- 启发：未来可结合交互检测方法（如 Tsang et al., 2020）或图结构信息来构建更智能的交互前沿

## 评分

- 新颖性: ⭐⭐⭐⭐ 多项式扩展思路自然但不算突破性，配对采样等价性定理是亮点
- 实验充分度: ⭐⭐⭐⭐⭐ 15 个博弈覆盖表格/图像/语言，多种基线对比全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，图表直观，叙述流畅
- 价值: ⭐⭐⭐⭐ 对 XAI 领域的 Shapley 值估计方法有实质推进，配对采样理论解释意义深远
