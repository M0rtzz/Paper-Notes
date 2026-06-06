---
title: >-
  [论文解读] FastLSQ: Solving PDEs in One Shot via Fourier Features with Exact Analytical Derivatives
description: >-
  [ICLR2026][偏微分方程] 利用正弦基函数的循环导数闭式结构，实现了无需自动微分、无需迭代训练的 PDE 一次性求解框架，在线性 PDE 上 0.07s 达到 $10^{-7}$ 精度，非线性 PDE 上 <9s 达到 $10^{-8}$–$10^{-9}$ 精度…
tags:
  - "ICLR2026"
  - "偏微分方程"
  - "random Fourier features"
  - "physics-informed computing"
  - "one-shot solver"
  - "Newton-Raphson"
  - "inverse problems"
---

# FastLSQ: Solving PDEs in One Shot via Fourier Features with Exact Analytical Derivatives

**会议**: ICLR2026  
**arXiv**: [2602.10541](https://arxiv.org/abs/2602.10541)  
**代码**: [sulcantonin/FastLSQ](https://github.com/sulcantonin/FastLSQ) (`pip install fastlsq`)  
**领域**: 其他  
**关键词**: PDE solving, random Fourier features, physics-informed computing, one-shot solver, Newton-Raphson, inverse problems

## 一句话总结
利用正弦基函数的循环导数闭式结构，实现了无需自动微分、无需迭代训练的 PDE 一次性求解框架，在线性 PDE 上 0.07s 达到 $10^{-7}$ 精度，非线性 PDE 上 <9s 达到 $10^{-8}$–$10^{-9}$ 精度，比 PINNs 快数千倍且精确数个数量级。

## 背景与动机
- **经典数值方法**（有限元、有限差分、谱方法）是科学计算主力，但高维问题（$d \geq 5$）计算量随 $h^{-d}$ 爆炸，且需要大量问题特定的实现工作
- **PINNs** 提供了无网格替代方案，但存在严重缺陷：训练需要数分钟到数小时、存在 spectral bias、因果性违反、损失权重敏感等问题
- **随机特征方法**（如 PIELM、RF-PDE）是折中路线，冻结随机参数只训练线性输出层。但 PIELM 使用 $\tanh$ 激活函数，缺乏闭式循环导数结构，需要为每个 PDE 算子手动推导符号微积分；RF-PDE 仍需 600–2000 轮迭代优化
- **核心观察**：正弦特征 $\phi_j(\mathbf{x}) = \sin(\mathbf{W}_j \cdot \mathbf{x} + b_j)$ 的任意阶导数具有循环闭式结构（$\sin \to \cos \to -\sin \to -\cos$），可在 $\mathcal{O}(1)$ 内完成任意线性微分算子矩阵的组装，无需自动微分或计算图

## 核心问题
如何构建一个**算子无关**（operator-agnostic）的 PDE 求解框架，既能避免 PINNs 的迭代训练开销，又能消除 PIELM 针对每个新 PDE 算子手动推导导数公式的负担？

## 方法详解

### 1. Random Fourier Feature 逼近
用正弦随机特征逼近 PDE 解 $u(\mathbf{x})$：

$$u_N(\mathbf{x}) = \frac{1}{\sqrt{N}} \sum_{j=1}^{N} \beta_j \sin(\mathbf{W}_j^\top \mathbf{x} + b_j)$$

其中 $\mathbf{W}_j \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_d)$、$b_j \sim \mathcal{U}(0, 2\pi)$ 冻结不变，仅训练线性系数 $\boldsymbol{\beta}$。$1/\sqrt{N}$ 归一化保证经验核收敛到高斯 RBF 核，防止系数膨胀到 $\mathcal{O}(10^6)$–$10^8$ 导致病态。采用多块（multi-block）架构，$B$ 个块使用不同带宽 $\sigma_b$ 以捕捉多尺度特征。

### 2. 正弦基的精确解析导数（关键创新）
对任意多重指标 $\alpha = (\alpha_1, \dots, \alpha_d)$：

$$D^\alpha \phi_j(\mathbf{x}) = \left(\prod_{k=1}^d W_{jk}^{\alpha_k}\right) \cdot \Phi_{|\alpha| \bmod 4}(\mathbf{W}_j^\top \mathbf{x} + b_j)$$

其中 $\Phi_0 = \sin$，$\Phi_1 = \cos$，$\Phi_2 = -\sin$，$\Phi_3 = -\cos$。这意味着：
- **Laplacian**：$\Delta \phi_j = -\|\mathbf{W}_j\|^2 \sin(\mathbf{W}_j^\top \mathbf{x} + b_j)$
- **Biharmonic**：$\Delta^2 \phi_j = \|\mathbf{W}_j\|^4 \sin(\mathbf{W}_j^\top \mathbf{x} + b_j)$
- **Advection**：$\mathbf{v} \cdot \nabla \phi_j = (\mathbf{v} \cdot \mathbf{W}_j) \cos(\mathbf{W}_j^\top \mathbf{x} + b_j)$

每项仅需一次三角函数求值乘以权重的单项式，无需自动微分或计算图。$\tanh$ 不存在类似的闭式模式（其 $n$ 阶导数是 $n+1$ 次多项式）。

### 3. 线性 PDE：一次最小二乘求解
将特征代入线性 PDE $\mathcal{L}[u] = f$ 及边界条件 $\mathcal{B}[u] = g$，得到增广线性系统：

$$\begin{pmatrix} \mathbf{A}^{\text{pde}} \\ \lambda \mathbf{A}^{\text{bc}} \end{pmatrix} \boldsymbol{\beta} = \begin{pmatrix} \mathbf{f} \\ \lambda \mathbf{g} \end{pmatrix}$$

通过 QR 或 SVD 分解一次性求解 $\boldsymbol{\beta}^* = \mathbf{A}^\dagger \mathbf{b}$，无需任何迭代。

### 4. 非线性 PDE：Newton-Raphson 扩展
对非线性 PDE $\mathcal{L}[u] + \mathcal{N}[u] = f$，採用 Newton-Raphson 迭代：

$$\mathbf{J}^{(k)} \delta\boldsymbol{\beta} = -\mathbf{R}^{(k)}, \quad \boldsymbol{\beta}^{(k+1)} = \boldsymbol{\beta}^{(k)} + \alpha \delta\boldsymbol{\beta}$$

Jacobian 继承解析闭式结构。四个关键算法改进保证鲁棒收敛：
- **Warm-start**：先求解线性部分作为初始猜测
- **Backtracking line search**：Armijo 型充分下降防止步长过大
- **解级别收敛判据**：用 $\|\Delta u\| / \|u\|$ 代替系数级别变化
- **Continuation（同伦）**：对对流主导问题（如 Burgers）逐步降低粘度 $\nu = 1.0 \to 0.5 \to 0.2 \to 0.1$

### 5. 下游应用
- **PDE 发现**：解析导数字典比有限差分干净 ~6000 倍（RMSE 0.4 vs 2500），大幅扩展 SINDy 的适用噪声范围
- **逆问题**：梯度通过预分解线性求解解析传播，可从 4 个传感器恢复 4 个各向异性高斯热源（24 参数），或从 8 个稀疏磁场测量恢复隐藏线圈位置（误差 <0.02）

## 实验关键数据

### 线性 PDE（Solver Mode）

| 问题 | FastLSQ 时间 | FastLSQ $L^2$ | PINNacle 时间 | PINNacle $L^2$ | 加速比 |
|------|------------|-------------|-------------|--------------|--------|
| Poisson 5D | 0.07s | 4.8e-7 | ~1780s | 4.7e-4 | 25000× |
| Wave 1D | 0.06s | 1.3e-6 | ~272s | 9.8e-2 | 4500× |
| Helmholtz 2D | 0.08s | 1.9e-6 | N/A | N/A | — |
| Maxwell 2D | 0.05s | 6.7e-7 | N/A | N/A | — |

### 非线性 PDE（Newton Solver Mode）
- NL-Poisson：$L^2 = 6.1 \times 10^{-8}$（8.2s），甚至优于拟合精确解的回归基线（$1.9 \times 10^{-7}$）
- Burgers ($\nu=0.1$)：$L^2 = 3.9 \times 10^{-9}$（7.4s，48 次迭代含同伦）
- 与 scikit-fem P2 FEM 对比：FastLSQ 在 1500 特征下达到 $10^{-7}$–$10^{-9}$，FEM 在 ~4000 DoF 下为 $10^{-6}$

### 消融实验
- 去掉 $1/\sqrt{N}$ 归一化：精度下降 4 个数量级或发散
- 去掉 Tikhonov 正则化：精度下降 3 个数量级
- 去掉 warm-start：精度下降 1 个数量级或发散
- 去掉 continuation：Burgers 问题发散

### sin vs tanh 基函数对比（相同求解协议）
精度差距 10×–1000× 完全归因于基函数选择。梯度精度方面，FastLSQ 的梯度误差通常在值误差的一个数量级内，而 PIELM 的梯度误差差 10×–100×。

## 亮点
- **极其简洁的核心洞察**：正弦函数的循环导数性质（$\sin \to \cos \to -\sin \to -\cos$）看似初等，但其对 PDE 求解的实际意义被长期忽视，将其系统化为通用求解框架非常巧妙
- **算子无关性**：一个公式适用于任意线性微分算子，而 PIELM 每换一个 PDE 就要手动推一遍
- **速度与精度兼得**：线性 PDE 在 0.07s 内达到 $10^{-7}$，比最快 PINN 变体快 25000 倍且精确 1000 倍
- **实用下游应用**：PDE 发现（解析导数字典）和逆问题（热源/线圈定位）展示了框架的实际工程价值
- **完整的可复现包**：`pip install fastlsq`，代码公开

## 局限与展望
- 带宽 $\sigma$ 需要网格搜索调参，尚无自动选择策略（虽然可微分优化在附录有演示）
- 高阶 PDE 或大 $\sigma$ 时单项式前因子放大条件数，限制可达精度
- 当前仅支持简单 box 域，不规则几何需要额外的边界采样策略
- $1/\sqrt{N}$ 归一化意味着增加 $N$ 不能简单提升精度，大 $N$ 时核逼近饱和且条件恶化
- Newton 扩展虽有效但比线性模式慢 40–100 倍（4–9s vs <0.1s）
- 对含间断的解（如激波）正弦基会产生 Gibbs 振荡，此时 $\tanh$ 基反而更稳健

## 与相关工作的对比

| 方法 | 类型 | 迭代 | 算子推导 | 典型精度 | 典型时间 |
|------|------|------|---------|---------|---------|
| **FastLSQ** | 正弦随机特征 | 线性一次/NL Newton | 闭式通用 | $10^{-7}$–$10^{-9}$ | 0.07–9s |
| PIELM | tanh 随机特征 | 一次 | 手动逐算子 | $10^{-3}$–$10^{-6}$ | ~0.07s |
| PINNs | 神经网络 | SGD 数千步 | 自动微分 | $10^{-2}$–$10^{-4}$ | 270–7500s |
| RF-PDE | 随机特征 | 600–2000 轮 | 自动微分 | $10^{-3}$–$10^{-5}$ | 38–51s |
| RBF Kansa | 径向基函数 | 一次 | 解析 | ~$10^{-5}$ | 依问题而定 |
| FEM | 有限元 | 直接 | 弱形式 | ~$10^{-6}$ | $d \geq 5$ 不可行 |

## 启发与关联
- 这篇工作表明 **"回到初等数学找闭式结构"** 在深度学习主导的时代仍有巨大价值——正弦的循环导数是高中知识，但系统利用它可以碾压复杂的神经网络方法
- 随机 Fourier 特征（Rahimi & Recht 2007）在核方法领域已经很经典，本文将其与 PDE 求解的需求精准对接，是一个漂亮的跨领域迁移
- 解析导数字典对 SINDy 类方法的提升（6000×）暗示在科学发现任务中，**表示的可微分性质量**可能比模型容量更重要
- 对逆问题的应用（热源定位、线圈恢复）展示了从 "解 PDE" 到 "可微分数字孪生" 的路径

## 评分
- 新颖性: ⭐⭐⭐⭐ （核心洞察虽初等但被系统化为通用框架，与 PIELM 区分明确）
- 实验充分度: ⭐⭐⭐⭐⭐ （17 个 PDE、多基线、消融、逆问题、梯度精度分析，极为全面）
- 写作质量: ⭐⭐⭐⭐⭐ （论述清晰、表格信息量大、与相关工作的对比非常公允）
- 价值: ⭐⭐⭐⭐ （对 PDE 求解社区有很高的实用价值，pip 包降低使用门槛）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Latent Fourier Transform](latent_fourier_transform.md)
- [\[ACL 2025\] The Knowledge Microscope: Features as Better Analytical Lenses than Neurons](../../ACL2025/others/the_knowledge_microscope_features_as_better_analytical_lenses_than_neurons.md)
- [\[ICLR 2026\] Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees](scalable_random_wavelet_features_efficient_non-stationary_kernel_approximation_w.md)
- [\[AAAI 2026\] Online Linear Regression with Paid Stochastic Features](../../AAAI2026/others/online_linear_regression_with_paid_stochastic_features.md)
- [\[ICLR 2026\] Neural Force Field: Few-shot Learning of Generalized Physical Reasoning](neural_force_field_few-shot_learning_of_generalized_physical_reasoning.md)

</div>

<!-- RELATED:END -->
