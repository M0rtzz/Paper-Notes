---
title: >-
  [论文解读] Global Minimizers of ℓp-Regularized Objectives Yield the Sparsest ReLU Neural Networks
description: >-
  [NeurIPS 2025][模型压缩][ℓp 正则化] 证明了对于单隐层 ReLU 网络，最小化 $\ell^p$（$0 < p < 1$）路径范数的全局最优解恰好对应于**最稀疏**的数据插值网络，从而将组合优化的稀疏插值问题重新表述为连续可微的优化任务。
tags:
  - NeurIPS 2025
  - 模型压缩
  - ℓp 正则化
  - ReLU 网络
  - 稀疏性
  - 全局最小值
  - 网络剪枝
---

# Global Minimizers of ℓp-Regularized Objectives Yield the Sparsest ReLU Neural Networks

**会议**: NeurIPS 2025  
**arXiv**: [2505.21791](https://arxiv.org/abs/2505.21791)  
**代码**: 无  
**领域**: 神经网络稀疏化 / 优化理论  
**关键词**: ℓp 正则化, ReLU 网络, 稀疏性, 全局最小值, 网络剪枝

## 一句话总结

证明了对于单隐层 ReLU 网络，最小化 $\ell^p$（$0 < p < 1$）路径范数的全局最优解恰好对应于**最稀疏**的数据插值网络，从而将组合优化的稀疏插值问题重新表述为连续可微的优化任务。

## 研究背景与动机

### 稀疏网络的重要性

过参数化的神经网络可以用多种方式插值给定数据集。一个根本性问题是：在所有可能的解中，我们应该偏好哪一个？**最稀疏**的网络（参数最少或神经元最少）在以下方面特别有价值：
- **计算效率**：存储和推理开销更低
- **泛化能力**：经验上稀疏网络通常泛化更好
- **可解释性**：更少的神经元意味着更透明的模型结构
- **模型压缩**：为知识蒸馏和部署提供基础

### 现有方法的不足

现有的稀疏化策略大多是**启发式**的，缺乏理论保证：

| 方法类别 | 典型方法 | 理论保证 |
|:---|:---|:---|
| 剪枝方法 | Lottery Ticket, Magnitude Pruning | 无最稀疏保证 |
| $\ell_1$ 正则化 | Lasso, 路径范数最小化 | 解不唯一，可能非最稀疏 |
| 结构化稀疏 | Group Lasso, $\ell_{2,1}$ | 组级稀疏但非最细粒度 |
| 随机门控 | $\ell_0$ 近似 | 近似解，无精确保证 |

特别值得注意的是，$\ell_1$ 路径范数最小化虽然看似促进稀疏，但在神经网络情境下：
- 解通常是**非唯一**的
- 某些 $\ell_1$ 最优解可以有**任意多**的神经元
- 因此"$\ell_1 =$ 稀疏性"的直觉在神经网络中并不成立

### 本文贡献

提出一种基于 $\ell^p$（$0 < p < 1$）拟范数的连续可微正则化目标，其全局最小值**被证明精确对应**最稀疏的 ReLU 网络。

## 方法详解

### 整体框架

#### 单变量情形（输入维度 $d = 1$）

考虑如下单隐层 ReLU 网络：

$$f_{\boldsymbol{\theta}}(x) = \sum_{k=1}^{K} v_k (w_k x + b_k)_+ + ax + c$$

$\ell^p$ 路径范数最小化问题定义为：

$$\min_{\boldsymbol{\theta}} \sum_{k=1}^{K} |w_k v_k|^p \quad \text{s.t.} \quad f_{\boldsymbol{\theta}}(x_i) = y_i, \; i=1,\ldots,N$$

#### 多变量情形（输入维度 $d > 1$）

对于 $\mathbb{R}^d \to \mathbb{R}$ 的 ReLU 网络，增加 $\ell_\infty$ 约束：

$$\min_{\boldsymbol{\theta}} \sum_{k=1}^{K} \|v_k \mathbf{w}_k\|_p^p \quad \text{s.t.} \quad f_{\boldsymbol{\theta}}(\mathbf{x}_i) = y_i, \; \|v_k \mathbf{w}_k\|_\infty \leq R$$

### 关键设计

#### 变分重表述（Proposition 3.1）

将网络优化问题等价转化为连续分段线性（CPWL）函数的变分问题：

$$\min_f V_p(f), \quad \text{s.t.} \quad f(x_i) = y_i$$

其中 $V_p(f)$ 为函数 $f$ 的导数的 $p$-变差。对于有 $K$ 个折点的 CPWL 函数：

$$V_p(f) = \sum_{k=1}^{K} |c_k|^p$$

$c_k$ 为折点处的斜率变化。这一重表述将参数空间的优化转化为函数空间的几何问题。

#### 几何刻画（Theorem 3.1）

核心定理精确描述了 $0 < p < 1$ 时最优函数的几何行为：

1. **线性区域**：在 $x_2$ 之前和 $x_N$ 之后，函数必须为线性；在离散曲率相反的连续数据点之间也必须为线性
2. **常曲率区域**：在 $m$ 个连续同曲率点 $x_i, \ldots, x_{i+m}$ 之间，函数是凸（或凹）的，至多有 $m-1$ 个折点
3. **斜率约束**：每个折点的斜率 $u_j$ 满足 $s_{i+j-1} \leq u_j \leq s_{i+j}$（假设曲率为正）

#### 唯一性与稀疏性（Theorem 3.2）

对于 Lebesgue 几乎所有 $0 < p < 1$，问题的解是**唯一**的。且存在数据相关的阈值 $p^*$ 使得当 $0 < p < p^*$ 时，$\ell^p$ 最优解恰好是最稀疏（$\ell^0$ 最优）的插值网络。

特别地：
- 任何 $\ell^p$ 最优解至多有 $N-2$ 个活跃神经元
- 对比之下，$\ell^1$ 最优解可能有任意多神经元

#### 多变量推广（Theorem 4.1）

利用 Bauer 最大值原理和凹函数在多面体上的最小化性质：
- 将优化问题转化为有限维凹函数在多面体上的最小化
- 最优解必在多面体的极端点处取得
- 存在 $p^*$ 使得 $\ell^p$-$\ell^0$ 等价

### 损失函数 / 训练策略

本文不涉及具体的训练算法。核心贡献在于证明了正确的目标函数形式。$\ell^p$ 路径范数目标是连续且几乎处处可微的，原则上可以使用梯度下降方法优化（尽管是非凸的）。

## 实验关键数据

### 理论界的对比

本文为纯数学理论工作。核心结果以定理形式呈现：

| 定理 | 维度 | 结论 | 条件 |
|:---|:---|:---|:---|
| Theorem 3.1 | $d=1$ | 几何刻画：最优函数必为线性 + 局部凸/凹 | $0 < p < 1$ |
| Corollary 3.1.1 | $d=1$ | $\ell^p$ 最优 $\Rightarrow$ $\ell^1$ 最优，$\leq N-2$ 神经元 | $0 < p < 1$ |
| Theorem 3.2 | $d=1$ | 几乎唯一，$\exists p^*$ 使得 $\ell^p = \ell^0$ | 无需数据假设 |
| Proposition 4.1 | 任意 $d$ | $\ell^0$ 最优解 $\leq N$ 活跃神经元 | 一般位置假设 |
| Theorem 4.1 | 任意 $d$ | $\exists p^*$ 使得 $\ell^p$ 最优 $= \ell^0$ 最优 | 需 $\ell_\infty$ 约束 |

### 不同正则化策略的稀疏性保证对比

| 正则化策略 | 最稀疏保证 | 神经元数上界 | 解唯一性 |
|:---|:---|:---|:---|
| $\ell_1$ 路径范数 | **否**（可有任意多神经元） | 无（最差情况无限） | 非唯一 |
| $\ell_0$ 路径范数 | 是（定义即此） | $N$ | 非唯一 |
| **$\ell^p$ ($0 < p < 1$)** | **是（充分小 $p$）** | $N-2$（$d=1$）/ $N$（任意 $d$） | **几乎唯一** |
| 权重衰减 ($\ell_2$) | 否 | 无保证 | - |

### 关键发现

1. **$\ell_1$ 在神经网络中不保证稀疏**：与有限维压缩感知中 $\ell_1$ 能恢复稀疏解不同，神经网络中 $\ell_1$ 路径范数最小化的解可以有任意多神经元。这是因为 CPWL 函数的参数化不是唯一的。

2. **$\ell^p (0 < p < 1)$ 同时最小化 $\ell^1$ 和 $\ell^0$**：Theorem 3.1 表明，$\ell^p$ 最优解也是 $\ell^1$ 最优的，因此同时获得了稀疏性（少神经元）和范数控制（权重不大）。

3. **从组合问题到连续优化**：$\ell^0$ 最小化是 NP 难问题，但本文证明通过最小化连续可微的 $\ell^p$ 目标可以精确恢复 $\ell^0$ 解。

4. **多变量推广需要约束**：与 Peng et al. (2015) 的声明不同，无约束的 $\ell^p$ 最小化的解不一定有界（本文修正了其证明中的错误），因此多变量情形需要显式的 $\ell_\infty$ 约束。

## 亮点与洞察

- **概念性突破**：首次严格证明了连续正则化目标能精确恢复最稀疏神经网络
- **变分观点**：将神经网络优化转化为 CPWL 函数空间上的变分问题，几何直觉强
- **Bauer 最大值原理的巧妙应用**：利用凹函数在凸集上的极值性质，将无限维问题化归为有限个候选解的比较
- **修正了文献中的错误**：发现并修正了 Peng et al. (2015) 关于 $\ell^p$ 最小化解有界性的错误声明

## 局限与展望

- 仅针对**单隐层** ReLU 网络，深层网络的推广是重要的未来方向
- 阈值 $p^*$ 的计算方法未给出，实际应用中如何选择 $p$ 仍是开放问题
- 只处理了**精确插值**，引入训练误差的情形（如近似拟合）未涉及
- 多变量情形需要 $\ell_\infty$ 约束，这在实际训练中如何实现需要进一步研究
- 非凸优化的收敛性问题：即使目标函数正确，梯度下降能否找到全局最优仍需分析

## 相关工作与启发

- **Debarre et al. (2022)**：刻画了 $\ell^1$ 路径范数最优解，但未能保证稀疏性
- **Boursier & Flammarion (2023)**：在较强的数据假设下证明的 $\ell^1$ + bias 正则化稀疏性
- **Ergen & Pilanci (2021)**：需要 $d > N$ 和白化数据的假设
- **Pilanci & Ergen (2020)**：凸重表述框架，为本文 Lemma 4.1 的推导提供了灵感
- **Parhi & Nowak (2021, 2022)**：变分框架下的表示定理，本文是其自然延伸
- 本文的结果为设计新的稀疏训练算法（如 reweighted $\ell^1$ 方法）提供了理论基础

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次严格建立 $\ell^p$-$\ell^0$ 等价性
- **技术深度**: ⭐⭐⭐⭐⭐ — 精妙的变分分析和凸几何论证
- **实用性**: ⭐⭐⭐ — 理论突破，但实际算法设计仍需后续工作
- **清晰度**: ⭐⭐⭐⭐ — 几何直觉呈现得很好，但数学细节密度高
- **综合评分**: 8.5/10

<!-- RELATED:START -->

## 相关论文

- [QuadEnhancer: Leveraging Quadratic Transformations to Enhance Deep Neural Networks](quadenhancer_leveraging_quadratic_transformations_to_enhance_deep_neural_network.md)
- [Topology and Geometry of the Learning Space of ReLU Networks: Connectivity and Size](../../ICLR2026/model_compression/topology_and_geometry_of_the_learning_space_of_relu_networks_connectivity_and_si.md)
- [Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)
- [Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)
- [Adaptive Width Neural Networks](../../ICLR2026/model_compression/adaptive_width_neural_networks.md)

<!-- RELATED:END -->
