---
title: >-
  [论文解读] Deep Legendre Transform
description: >-
  [NeurIPS 2025][凸共轭] DLT 利用凸共轭的隐式 Fenchel 表示 $f^*(\nabla f(x)) = \langle x, \nabla f(x) \rangle - f(x)$ 将凸共轭计算转化为标准回归问题，避免求解 max/min-max 优化，且能提供后验误差估计，结合 KAN 还可获得精确解析解。
tags:
  - NeurIPS 2025
  - 凸共轭
  - Legendre变换
  - ICNN
  - KAN
  - 后验误差估计
  - Hamilton-Jacobi方程
---

# Deep Legendre Transform

**会议**: NeurIPS 2025  
**arXiv**: [2512.19649](https://arxiv.org/abs/2512.19649)  
**代码**: [GitHub](https://github.com/lexmar07/Deep-Legendre-Transform)  
**领域**: others (数值方法 / 凸优化 / 深度学习)  
**关键词**: 凸共轭, Legendre变换, ICNN, KAN, 后验误差估计, Hamilton-Jacobi方程

## 一句话总结
DLT 利用凸共轭的隐式 Fenchel 表示 $f^*(\nabla f(x)) = \langle x, \nabla f(x) \rangle - f(x)$ 将凸共轭计算转化为标准回归问题，避免求解 max/min-max 优化，且能提供后验误差估计，结合 KAN 还可获得精确解析解。

## 研究背景与动机

**领域现状**：凸共轭（Legendre-Fenchel 变换）是凸分析、优化、物理学、经济学的基础运算。定义为 $f^*(y) = \sup_{x \in C}\{\langle x,y\rangle - f(x)\}$。

**现有痛点**：
   - 经典网格方法复杂度 $O(N^d)$，受维度灾难限制，仅适用 $d \leq 4$
   - 基于神经网络的方法（用于最优传输）需要解复杂的 max-min 问题
   - 学习 $\nabla f$ 的逆映射（Korotin 等）需设计额外优化

**核心矛盾**：高维凸共轭缺乏高效、可扩展、带误差估计的数值方法

**切入角度**：可微凸函数满足隐式等式 $f^*(\nabla f(x)) = \langle x, \nabla f(x)\rangle - f(x)$——这直接给出了无需求解优化问题的训练目标

## 方法详解

### 隐式 Fenchel 公式

对可微凸函数 $f: C \to \mathbb{R}$（$C \subseteq \mathbb{R}^d$ 为开凸集），有：

$$f^*(\nabla f(x)) = \langle x, \nabla f(x) \rangle - f(x), \quad x \in C$$

这意味着无需知道 $f^*$ 的解析形式，就能在 $\nabla f(x)$ 点处得到精确目标值。

### DLT 训练

用参数化模型 $g_\theta: D \to \mathbb{R}$ 逼近 $f^*$，最小化：

$$\min_{\theta \in \Theta} \sum_{x \in \mathcal{X}_{\text{train}}} \left(g_\theta(\nabla f(x)) + f(x) - \langle x, \nabla f(x)\rangle\right)^2$$

关键优势：
- 训练目标值**精确**等于 $f^*(\nabla f(x))$，无近似误差
- 标准 MSE 回归——SGD 直接可解
- 如果用 ICNN 作为 $g_\theta$，输出保证是凸函数

### 后验误差估计

利用相同的隐式公式，可用独立测试集 $\mathcal{X}_{\text{test}}$ 估计 $L^2$ 误差：

$$\frac{1}{|\mathcal{X}_{\text{test}}|}\sum_{x \in \mathcal{X}_{\text{test}}} (g(\nabla f(x)) - f^*(\nabla f(x)))^2 \to \|g - f^*\|^2_{L^2(D, \nu)}$$

其中 $\nu = \mu \circ (\nabla f)^{-1}$ 是采样分布的推前度量。**即使 $f^*$ 未知也能精确评估误差**——这是该方法独有的优势。

### 梯度空间采样

当 $\nabla f$ 严重扭曲 $C$ 时，直接在 $C$ 上均匀采样会导致 $D$ 中分布不均。解决方案：
1. 训练网络 $h_\vartheta: D \to \mathbb{R}^d$ 逼近 $\nabla f$ 的广义逆
2. 预训练：$\min_\vartheta \sum_{x \in \mathcal{W}_{\text{train}}} \|\nabla f(h_\vartheta(\nabla f(x))) - \nabla f(x)\|_2^2$
3. 微调：加入 $\min_\vartheta \sum_{y \in \mathcal{Y}_{\text{train}}} \|\nabla f(h_\vartheta(y)) - y\|_2^2$

### 模型选择

- **MLP/ResNet**：通用逼近，ResNet 效果最好但不保证凸性
- **ICNN**：保证输出凸性，适用于下游优化任务
- **KAN（Kolmogorov-Arnold 网络）**：可做符号回归，获得**精确解析解**

## 实验关键数据

### DLT vs 直接学习（$d=10$ 和 $d=50$）

| 函数 | 模型 | RMSE (DLT / 直接) $d$=10 | RMSE (DLT / 直接) $d$=50 |
|------|------|:----------------------:|:----------------------:|
| Quadratic | MLP | 2.96e-5 / 2.86e-5 | 3.55e-3 / 3.59e-3 |
| Quadratic | ResNet | 2.43e-5 / 3.00e-5 | 2.92e-3 / 3.25e-3 |
| Neg-Log | ResNet | 7.34e-4 / 9.22e-4 | 1.05e-1 / 1.23e-1 |
| Neg-Entropy | ResNet | 6.62e-4 / 7.41e-4 | 6.93e-2 / 7.27e-2 |

DLT 精度与直接学习持平甚至更优（尤其 ResNet），训练时间也相当。ICNN 在高维下误差较大，MLP-ICNN（无 skip connection）效果最差。

### 模型架构差异

| 架构 | 精度（$d=10$） | 凸性保证 | 符号回归能力 |
|------|:------------:|:------:|:---------:|
| MLP | 高 | ✗ | ✗ |
| ResNet | **最高** | ✗ | ✗ |
| ICNN | 中 | ✓ | ✗ |
| KAN | 高 | ✗ | ✓ |

### KAN 精确解析解
对特定凸函数（如 $f(x) = \frac{1}{2}\sum x_i^2$），KAN + 符号回归可恢复精确的 $f^*(y) = \frac{1}{2}\sum y_i^2$——这是数值方法罕见实现的。

### Hamilton-Jacobi PDE 应用
DLT 可用于通过 Hopf 公式求解 Hamilton-Jacobi 方程的初始条件，验证了在 PDE 领域的实际用途。

## 亮点与洞察
- **隐式公式的视角转换极为巧妙**：将传统的 $\sup$ 问题变成 MSE 回归，原理简洁但此前被忽视
- **免费后验误差估计**是独特优势——不需要知道真实 $f^*$ 就能评估任何逼近方法的精度
- KAN 获得精确解析解的能力展示了深度学习在符号计算中的潜力
- 方法与 ICNN 结合可保证凸性，对下游最优传输等任务有直接价值
- 计算复杂度摆脱维度灾难，$d=50$ 仍可工作

## 局限与展望
- 仅适用于**可微凸函数**——非光滑凸函数需要其他方法
- 需要能评估 $f(x)$ 和 $\nabla f(x)$——黑箱函数不适用
- 当 $\nabla f$ 的像 $D$ 无界时，采样策略更复杂
- ICNN 在高维（$d=50$）下精度明显下降，实际保证凸性的代价不小
- 未与最优传输求解器做端到端对比

## 评分
- 新颖性: ⭐⭐⭐⭐ 隐式 Fenchel 公式的 ML 应用是新的视角，KAN 精确解是亮点
- 实验充分度: ⭐⭐⭐⭐ 多架构 × 多维度 × 多函数类型 × HJ-PDE 应用
- 写作质量: ⭐⭐⭐⭐⭐ 数学严谨且表述清晰，理论与实验结合好
- 价值: ⭐⭐⭐⭐ 对凸分析和最优传输社区有工具性价值
