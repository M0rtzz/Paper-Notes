---
title: >-
  [论文解读] Universal Neural Optimal Transport
description: >-
  [ICML2025][科学计算][最优传输] 提出 UNOT（Universal Neural Optimal Transport），利用 Fourier Neural Operator 学习跨数据集、跨分辨率的熵正则化最优传输对偶势函数，实现对 Sinkhorn 算法最高 7.4× 的加速初始化。
tags:
  - ICML2025
  - 科学计算
  - 最优传输
  - 神经算子
  - Sinkhorn Algorithm
  - 对偶势函数
  - 对抗训练
  - 自举损失
---

# Universal Neural Optimal Transport

**会议**: ICML2025  
**arXiv**: [2212.00133](https://arxiv.org/abs/2212.00133)  
**代码**: [GregorKornhardt/UNOT](https://github.com/GregorKornhardt/UNOT)  
**领域**: 最优传输 (Optimal Transport)  
**关键词**: 最优传输, Fourier Neural Operator, Sinkhorn Algorithm, 对偶势函数, 对抗训练, 自举损失

## 一句话总结

提出 UNOT（Universal Neural Optimal Transport），利用 Fourier Neural Operator 学习跨数据集、跨分辨率的熵正则化最优传输对偶势函数，实现对 Sinkhorn 算法最高 7.4× 的加速初始化。

## 研究背景与动机

最优传输（OT）是机器学习中的基础工具，广泛应用于域适应、单细胞基因组学、图像处理、流匹配等场景。给定概率测度 $\mu, \nu$ 和代价函数 $c$，熵正则化 OT 问题定义为：

$$\text{OT}_\epsilon(\mu,\nu) = \inf_{\pi \in \Pi(\mu,\nu)} \int c \, d\pi - \epsilon \text{KL}(\pi \| \mu \otimes \nu)$$

Sinkhorn 算法可以迭代求解该问题，但计算开销大，尤其在需要反复求解 OT 的场景下瓶颈明显。现有加速方案包括：

- **高斯初始化**（Thornton & Cuturi, 2022）：用高斯 OT 问题的解初始化 Sinkhorn
- **Meta OT**（Amos et al., 2023）：训练神经网络预测传输计划，但仅限于固定维度

两者均无法处理变分辨率输入，也不能跨数据集泛化。UNOT 的目标是构建一个**通用**的神经 OT 求解器，给定任意离散测度对，直接预测对偶势函数。

## 方法详解

### 核心思路：预测对偶势函数

UNOT 不直接预测 $m \times n$ 维传输计划矩阵 $\Pi$，而是预测 $n$ 维对偶势函数 $\boldsymbol{g}$。由对偶 OT 理论，最优传输计划可从对偶势恢复：

$$\Pi = \text{diag}(\boldsymbol{u}) K \text{diag}(\boldsymbol{v}), \quad K = \exp(-C/\epsilon)$$

其中 $(\boldsymbol{u}, \boldsymbol{v}) = (\exp(\boldsymbol{f}/\epsilon), \exp(\boldsymbol{g}/\epsilon))$。给定 $\boldsymbol{g}$，另一势函数可通过一步 Sinkhorn 迭代恢复：$\boldsymbol{u} = \boldsymbol{\mu} ./ K\boldsymbol{v}$，将问题降维为 $n$ 维。

### 网络架构：Fourier Neural Operator

选择 FNO 的理论依据是**离散对偶势的收敛性**（Proposition 2）：当离散测度 $(\mu_n, \nu_n) \to (\mu, \nu)$ 时，对应的对偶势函数一致收敛。这意味着离散测度及其势函数可视为连续函数的离散化，天然适配 FNO 的离散不变性。

网络 $S_\phi$ 接收测度对 $(\boldsymbol{\mu}, \boldsymbol{\nu})$ 作为输入，输出对偶势 $\boldsymbol{g}$：

$$S_\phi(\boldsymbol{\mu}, \boldsymbol{\nu}) = \boldsymbol{g}$$

FNO 通过 $L$ 层傅里叶核层实现：每层对输入做离散傅里叶变换，保留固定数量低频特征，经复数线性层变换后逆变换回时域。SϕS_\phi 共 26M 参数。对于球面代价函数，使用 Spherical FNO (SFNO) 替代。

### 对抗训练生成器

生成器 $G_\theta$ 从高斯噪声 $\boldsymbol{z} \sim \mathcal{N}(0, I)$ 生成训练分布对：

$$G_\theta(\boldsymbol{z}) = R[\text{ReLU}(\text{NN}_\theta(\boldsymbol{z}) + \lambda I_{d,d'}(\boldsymbol{z})) + \delta]$$

其中 $R$ 为归一化+随机降采样算子，$\lambda$ 为残差连接系数，$\delta > 0$ 保证正密度。

**理论保证**（Theorem 3）：当 $\text{Lip}(\text{NN}_\theta) < \lambda$ 时，$G_\theta$ 在所有非负向量上有正密度，即能生成任意离散概率测度对。该结论推广到一般残差网络组合（Corollary 4）。

### 自举损失函数

直接用 Sinkhorn 收敛解做监督代价过高。UNOT 使用**自举损失**：以网络预测 $\boldsymbol{g}_\phi$ 为初始化，仅跑 $k=5$ 步 Sinkhorn 得到 $\boldsymbol{g}_{\tau_k}$，然后最小化两者距离：

$$\mathcal{L} = \|\boldsymbol{g}_\phi - \boldsymbol{g}_{\tau_k}\|_2^2$$

**理论保证**（Proposition 5）：最小化自举损失等价于最小化与真实势函数的距离，上界为 $c(K,k,n) \cdot \|\boldsymbol{g}_\phi - \boldsymbol{g}_{\tau_k}\|_2^2$。

### 完整训练目标

对抗博弈形式，求解器 $S_\phi$ 做 min、生成器 $G_\theta$ 做 max：

$$\max_\theta \min_\phi \mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{g}_{\tau_k} - S_\phi(G_\theta(\boldsymbol{z}))\|_2^2]$$

## 实验关键数据

### 实验设置

- 代价函数：$\|x-y\|^2$（欧氏平方）、$\|x-y\|$（欧氏）、$\arccos(\langle x,y\rangle)$（球面）
- 训练样本：200M，分辨率 $10\times10$ 到 $64\times64$，$\epsilon=0.01$
- 训练时间：约 35h（H100 GPU）
- 测试集：MNIST(28²)、CIFAR10(28²)、LFW(64²)、Bears(64²) 及跨数据集

### Sinkhorn 迭代次数（达到 1% 相对误差，$c=\|x-y\|^2$）

| 数据集 | UNOT | Ones 初始化 | Gaussian 初始化 |
|--------|------|-----------|----------------|
| MNIST | **3±5** | 16±9 | 10±7 |
| CIFAR | **3±6** | 80±22 | 52±19 |
| LFW | **7±8** | 78±20 | 35±14 |
| Bear | **4±6** | 41±16 | 25±13 |
| LFW-Bear | **4±6** | 53±18 | 29±13 |

### 实际加速比（$c=\|x-y\|^2$，达到 1% 误差的挂钟时间）

| 数据集 | UNOT (s) | Ones (s) | 加速比 |
|--------|----------|----------|--------|
| CIFAR | 9.5e-4 | 7.1e-3 | **7.4×** |
| LFW | 3.0e-3 | 1.5e-2 | **5.0×** |
| Bear | 2.6e-3 | 1.0e-2 | **3.8×** |
| LFW-Bear | 2.7e-3 | 1.2e-2 | **4.4×** |

UNOT 在所有数据集上均大幅减少 Sinkhorn 收敛迭代数，在 CIFAR 上获得最高 7.4× 实际加速。

## 亮点与洞察

- **首个跨数据集跨分辨率的通用神经 OT 求解器**：FNO 的离散不变性与对偶势收敛理论完美结合
- **理论扎实**：生成器的万有逼近性（Theorem 3）、自举损失的正确性（Proposition 5）、离散-连续势收敛（Proposition 2）均有严格证明
- **训练巧妙**：对抗式自举训练避免了昂贵的 Sinkhorn 标签计算，$k=5$ 步即可提供有效梯度信号
- **保持 Sinkhorn 优良性质**：UNOT 输出可直接作为 Sinkhorn 初始化，保持可并行、可微分
- **非欧几何支持**：通过 SFNO 无缝扩展到球面 OT 问题

## 局限与展望

- 需要对每种代价函数 $c$ 单独训练一个模型（固定 $\epsilon=0.01$），泛化到不同 $\epsilon$ 或不同 $c$ 需额外训练
- 训练代价较高（200M 样本、35h H100），部署前期投入大
- 当前仅验证了 grid-based 离散化场景（图像类），对非结构化点云的效果未充分探索
- FNO 对高频势函数的捕捉能力可能受限于保留的傅里叶模式数
- 对于 MNIST 等简单数据集加速比相对较小（1.25×），主要收益集中在复杂分布

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — FNO + 对抗自举的 OT 求解框架非常新颖
- 实验充分度: ⭐⭐⭐⭐ — 多代价函数、多数据集、重叠与消融实验较完整
- 写作质量: ⭐⭐⭐⭐⭐ — 理论-方法-实验结构清晰，证明严谨
- 价值: ⭐⭐⭐⭐ — 对大规模重复 OT 求解场景有实际加速价值

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Towards Universal Neural Operators through Multiphysics Pretraining](../../NeurIPS2025/scientific_computing/towards_universal_neural_operators_through_multiphysics_pretraining.md)
- [\[NeurIPS 2025\] Multi-Trajectory Physics-Informed Neural Networks for HJB Equations with Hard-Zero Terminal Inventory: Optimal Execution on Synthetic & SPY Data](../../NeurIPS2025/scientific_computing/multi-trajectory_physics-informed_neural_networks_for_hjb_equations_with_hard-ze.md)
- [\[ICML 2025\] Maximal Update Parametrization and Zero-Shot Hyperparameter Transfer for Fourier Neural Operators](maximal_update_parametrization_and_zero-shot_hyperparameter_transfer_for_fourier.md)
- [\[ICML 2025\] Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)
- [\[ICCV 2025\] JPEG Processing Neural Operator for Backward-Compatible Coding](../../ICCV2025/scientific_computing/jpeg_processing_neural_operator_for_backward-compatible_coding.md)

<!-- RELATED:END -->
