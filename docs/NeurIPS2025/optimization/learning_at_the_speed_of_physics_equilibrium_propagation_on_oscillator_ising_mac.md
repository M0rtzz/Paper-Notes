---
title: >-
  [论文解读] Learning at the Speed of Physics: Equilibrium Propagation on Oscillator Ising Machines
description: >-
  [NeurIPS 2025][优化][equilibrium propagation] 首次将 Equilibrium Propagation（EP）完整映射到振荡器 Ising Machine（OIM）硬件上，利用 GHz 物理动力学实现无反向传播的局部学习，在 MNIST/Fashion-MNIST 上达到 97.2%/88.0% 精度，并展示在参数量化和噪声下的鲁棒性。
tags:
  - NeurIPS 2025
  - 优化
  - equilibrium propagation
  - oscillator Ising machine
  - neuromorphic computing
  - energy-based model
  - local learning rule
---

# Learning at the Speed of Physics: Equilibrium Propagation on Oscillator Ising Machines

**会议**: NeurIPS 2025  
**arXiv**: [2510.12934](https://arxiv.org/abs/2510.12934)  
**代码**: [alexgower/OIM-Equilibrium-Propagation](https://github.com/alexgower/OIM-Equilibrium-Propagation)  
**领域**: optimization  
**关键词**: equilibrium propagation, oscillator Ising machine, neuromorphic computing, energy-based model, local learning rule

## 一句话总结
首次将 Equilibrium Propagation（EP）完整映射到振荡器 Ising Machine（OIM）硬件上，利用 GHz 物理动力学实现无反向传播的局部学习，在 MNIST/Fashion-MNIST 上达到 97.2%/88.0% 精度，并展示在参数量化和噪声下的鲁棒性。

## 研究背景与动机
**领域现状**：物理系统天然执行能量下降过程，可直接加速能量基模型（EBM）的优化。OIM 由耦合非线性振荡器网络构成，其 GHz 频率动力学天然对应梯度下降。

**现有痛点**：(a) EP 在传统处理器上受限于长弛豫和采样时间；(b) 先前在振荡器上实现 EP 的工作存在初始化或同步问题；(c) 离散 Ising 求解器不支持 EP 所需的连续相位动力学。

**核心动机**：OIM 原为组合优化设计，但其连续相位动力学 + 能量下降特性恰好满足 EP 要求——能否零硬件改动将其转为神经形态学习处理器？

**切入角度**：证明 OIM 的能量函数可精确编码 MLP 的总能量（含 MSE 损失），EP 更新规则在 OIM 上自然为局部相位测量。

## 方法详解

### OIM 动力学基础

$n$ 个耦合振荡器，每个以相位 $\phi_i \in [0, 2\pi]$ 参数化，动力学为能量函数 $V$ 的梯度下降：

$$\frac{d\phi_i}{dt'} = -\frac{\partial V}{\partial\phi_i}$$

$$V = -\frac{1}{2}\sum_{i,j\ne i} J_{ij}\cos(\phi_i - \phi_j) - \sum_i h_i\cos(\phi_i) - \sum_i \frac{S_i}{2}\cos(2\phi_i)$$

其中 $J_{ij}$ 是耦合强度，$h_i$ 是偏置，$S_i$ 是同步场。真实时间 $t = t'/\bar{\omega}$ 与振荡器频率成反比。

### EP 在 OIM 上的映射

**网络结构**：MLP 有 $n_x$ 输入、$n_h$ 隐层、$n_y$ 输出神经元。非输入神经元 = 振荡器。激活为 $s_i = \cos(\phi_i) \in [-1, 1]$。

**总能量分解**：$F = E + \beta\ell$，其中 $E$ 为自由能，$\ell$ 为 MSE 损失，$\beta$ 为 nudge 因子。

**OIM 参数对应**：

| MLP 组件 | OIM 参数 |
|---------|----------|
| 隐层偏置 $b_i^{(h)}$ + 输入权重 | $h_i^{(h)} = b_i^{(h)} + \sum_j w_{ji}^{(x,h)} x_j$ |
| 隐层-输出权重 $w_{ij}^{(h,y)}$ | $J_{ij}^{(h,y)} = w_{ij}^{(h,y)}$ |
| 输出偏置 + 目标 | $h_i^{(y)} = b_i^{(y)} + \beta\hat{y}_i$ |
| MSE 损失项 | $S_i^{(y)} = -\beta/2$ |

关键：MSE 损失 $\frac{1}{2}(\cos\phi_i - \hat{y}_i)^2$ 展开后恰好包含 $\cos(2\phi_i)$ 和 $\cos(\phi_i)$ 项，与 OIM 能量函数中已有的 $S_i$ 和 $h_i$ 项天然对应。

### EP 三阶段训练

对每个训练样本 $x$：
1. **自由阶段**（$\beta=0$）：从参考态 $\phi_0 = \{\pi/2\}$（对应激活 $\cos(\pi/2)=0$）演化至稳态 $\phi_*$
2. **正 nudge 阶段**（$\beta>0$）：从 $\phi_*$ 出发演化至 $\phi_*^{+\beta}$
3. **负 nudge 阶段**（$\beta<0$）：从 $\phi_*$ 出发演化至 $\phi_*^{-\beta}$

**参数更新（全局部）**：

$$\Delta w_{ij}^{(h,y)} \propto -\frac{1}{2\beta}[\cos(\phi_i^{(h),-\beta} - \phi_j^{(y),-\beta}) - \cos(\phi_i^{(h),+\beta} - \phi_j^{(y),+\beta})]$$

仅依赖相连振荡器的相位差——无需全局反向传播电路。更新在 mini-batch 上平均，兼容标准优化器。

### EP-BPTT 等价性

在 $\beta \to 0$ 极限下，EP 更新严格等于反向传播穿越时间（BPTT）：

$$\lim_{\beta\to 0} \hat{\nabla}^{\rm EP}(\beta) = -\frac{\partial\ell}{\partial\theta}(y_*, \hat{y})$$

实验验证了 OIM 的非线性正弦耦合下此对应关系仍成立（Fig. 1 inset）。

## 实验关键数据

### 分类精度

| 架构 | 数据集 | EP 精度 | BPTT 精度 | 对比方法 |
|------|--------|:---:|:---:|------|
| 784-500-10 | MNIST | **97.2±0.1%** | 96.8±0.1% | — |
| 784-500-10 | Fashion-MNIST | **88.0±0.1%** | — | p-bit Ising: 87.0% |
| 784-120-10 | MNIST/100 | 90.6±1.7% | — | D-Wave: ~85% |
| 784-120-10 (有噪声) | MNIST/100 | **92.0±0.3%** | — | 噪声 $\xi=0.2$ |

EP 精度匹配甚至略高于同架构 BPTT（97.2 vs 96.8），表明性能受架构限制而非训练方法。

### 硬件鲁棒性

| 测试条件 | 精度 | 备注 |
|---------|:---:|------|
| 相位量化 4-bit | 89.8±1.5% | 可行 |
| 相位量化 2-bit | 大幅下降 | 不可行 |
| 参数量化 10-bit | 89.4±1.5% | 可行 |
| 参数量化 8-bit | 下降 | 边界 |
| 高斯相位噪声 $\xi=0.2$ | **92.0±0.3%** | 适度噪声有正则化效果 |
| 高斯相位噪声 $\xi=0.3$ | 保持（$\beta \gtrsim \xi/2$） | OIM 噪声 ≈ Langevin 动力学 |

**关键发现**：当 $\beta \ge \xi/2$ 时噪声鲁棒，适度噪声（$\xi=0.2$）反而提升精度（正则化效应）。

### 潜在加速估计

| 条件 | EP 模拟时间 | BPTT 模拟时间 | 物理 OIM 预计 |
|------|:---------:|:---------:|:-----------:|
| MNIST 50 epochs | ~40 小时 | ~60 小时 | **秒到分钟** |

GHz 振荡器频率使相位更新在微秒量级完成，理论加速数个数量级。

## 亮点
1. **零硬件改动**将组合优化设备转为神经形态学习处理器
2. 局部更新规则无需全局反向传播电路——每个突触/神经元仅测量本地相位
3. 噪声鲁棒性降低了硬件设计精度要求，$\xi=0.2$ 噪声甚至有益
4. EP-BPTT 等价性实验验证了非线性正弦耦合下的理论预测
5. MSE 损失与 OIM 同步场 $S_i$ 的自然对应非常优雅

## 局限与展望
1. 仅验证单隐层 MLP，深层网络（多隐层）需更多振荡器和更复杂耦合拓扑
2. 当前为 PyTorch 模拟，真实 OIM 硬件上的验证尚未完成
3. 仅用 MSE 损失（因 $\cos(2\phi)$ 对应关系），交叉熵等损失不直接适配
4. MNIST/Fashion-MNIST 精度落后于 CNN/Transformer，更多是概念验证
5. 训练需 $T=4000$ 步自由阶段 + $K=400$ 步 nudge 阶段，步数较多

## 与相关工作的对比
- **vs. D-Wave 退火器**（Laydevant et al.）：D-Wave 是离散 Ising 求解器不支持连续 EP，OIM 连续相位天然适配
- **vs. p-bit Ising 机**（Niazi et al.）：p-bit 训练 Deep Boltzmann Machine，OIM-EP 在 FMNIST 上精度更高
- **vs. 传统 EP 实现**（Scellier & Bengio）：首次在物理系统（而非数字模拟）框架下完整实现
- **vs. Wang et al. / Rageau et al.**（振荡器 EP）：先前工作存在初始化/同步问题，本文无需额外硬件改动

## 启发与关联
- "物理动力学 = ML 优化"的范式可推广到光学神经网络、自旋电子学等其他物理平台
- OIM 噪声对应 Langevin 动力学，暗示可原生支持基于采样的生成模型
- EP 的局部性使其特别适合大规模分布式/类脑芯片

## 评分
- ⭐ 新颖性: 4/5 — OIM 做 EP 的映射巧妙，MSE → 同步场的对应关系是亮点
- ⭐ 实验充分度: 3/5 — MNIST 级验证充分，但缺少真实硬件和更复杂任务
- ⭐ 写作质量: 4/5 — 概念清晰，物理-ML 映射解释到位
- ⭐ 综合价值: 3/5 — 概念验证阶段，实际应用待硬件成熟
