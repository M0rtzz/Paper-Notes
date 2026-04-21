---
title: >-
  [论文解读] When Diffusion Models Memorize: Inductive Biases in Probability Flow of Minimum-Norm Shallow Neural Nets
description: >-
  [ICML2025][图像生成][扩散模型] 从理论上分析了最小 $\ell^2$ 范数浅层 ReLU 去噪器驱动的扩散模型概率流的收敛行为，证明概率流可以收敛到训练样本（记忆化）、训练样本之和（"虚拟点"）或超盒边界上的流形点（泛化），且扩散时间调度器的"早停"效应决定了收敛目标。
tags:
  - ICML2025
  - 图像生成
  - 扩散模型
  - 记忆化
  - 概率流ODE
  - 最小范数
  - 浅层ReLU网络
  - 归纳偏置
  - 得分流
---

# When Diffusion Models Memorize: Inductive Biases in Probability Flow of Minimum-Norm Shallow Neural Nets

**会议**: ICML2025  
**arXiv**: [2506.19031](https://arxiv.org/abs/2506.19031)  
**代码**: 待确认  
**领域**: image_generation  
**关键词**: 扩散模型, 记忆化, 概率流ODE, 最小范数, 浅层ReLU网络, 归纳偏置, 得分流

## 一句话总结
从理论上分析了最小 $\ell^2$ 范数浅层 ReLU 去噪器驱动的扩散模型概率流的收敛行为，证明概率流可以收敛到训练样本（记忆化）、训练样本之和（"虚拟点"）或超盒边界上的流形点（泛化），且扩散时间调度器的"早停"效应决定了收敛目标。

## 研究背景与动机
扩散模型通过概率流 ODE 生成高质量图像，但其理论理解仍不完整。核心问题是：

**概率流何时收敛到训练样本**（记忆化）vs **何时收敛到更一般的流形点**（泛化）？
2. 实际中使用的神经网络去噪器的 Jacobian 矩阵通常是非对称的，使得分函数估计并非真正的梯度场，采样过程的收敛性存在理论空白
3. 深层网络过于复杂无法理论分析，但浅层 ReLU 网络足够简单又能提供有价值的洞见

**核心创新**：引入更简单的"得分流（score flow）"ODE 辅助分析，揭示概率流与得分流之间的相似性和关键差异——扩散时间调度器诱导的"早停"效应。

## 方法详解

### 问题设置
- 观测模型：$\mathbf{y} = \mathbf{x} + \boldsymbol{\epsilon}$，$\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$
- 去噪器参数化：带跳跃连接的浅层 ReLU 网络 $\mathbf{h}_\theta(\mathbf{y}) = \sum_k \mathbf{a}_k[\mathbf{w}_k^\top \mathbf{y} + b_k]_+ + \mathbf{V}\mathbf{y} + \mathbf{c}$
- 正则化：参数的 $\ell^2$ 范数，即 $C(\theta) = \frac{1}{2}\sum_k (\|\mathbf{a}_k\|^2 + \|\mathbf{w}_k\|^2)$
- 目标：找到最小表示代价（min-cost）的完美插值去噪器

### 概率流 vs 得分流

**得分流（Score Flow）**：固定噪声水平的梯度上升
$$\frac{d\mathbf{y}_r}{dr} = \mathbf{h}^*_\rho(\mathbf{y}_r) - \mathbf{y}_r$$

**概率流（Probability Flow）**：噪声水平随时间变化
$$\frac{d\mathbf{y}_r}{dr} = \mathbf{h}^*_{\rho_{g^{-1}_r}}(\mathbf{y}_r) - \mathbf{y}_r$$

关键差异：概率流中 $\rho_t = \alpha \sigma_t$ 随时间递减，引入了"早停"效应。

### 理论结果（正交数据集）

**定理 4.2（驻点集）**：对于正交训练点 $\{\mathbf{x}_0, ..., \mathbf{x}_{N-1}\}$，得分流的稳定驻点集为训练点的所有子集之和的集合：
$$\mathcal{A} = \left\{\sum_{n \in \mathcal{I}} \mathbf{x}_n \mid \mathcal{I} \subseteq [N-1]\right\}$$
这些驻点恰好构成一个**超盒（hyperbox）**的顶点。

**定理 4.3（得分流收敛）**：
- 得分流收敛到初始化点最近的超盒顶点
- 可能先收敛到超盒边界，再沿边界滑向最近顶点

**定理 4.4（概率流收敛）**：
- 若初始化点最近点是超盒顶点，则收敛到该顶点
- 否则，根据初始时间 $T$ 与临界值 $\tau$ 的关系：$T > \tau$ 时收敛到顶点，$T < \tau$ 时收敛到超盒边界上的非顶点（流形点）

**核心洞见**：扩散时间调度器的"早停"使得概率流可以收敛到超盒边界的任意点（泛化），而得分流只能收敛到顶点（记忆化或虚拟点）。

### 扩展结果
- **钝角单纯形数据**（Appendix B）：稳定驻点是训练点子集之和的子集
- **等边三角形数据**（Appendix C）：得分流先收敛到三角形面再到顶点

## 实验关键数据

### 虚拟训练点存在性验证（$d=30$, 正交数据）

| 组合类型 | 理论虚拟点数 | 稳定率 | 实际稳定虚拟点数 |
|----------|-------------|--------|-----------------|
| 成对 | 全部 | 98.6% | 429 |
| 三元 | 全部 | 较低 | 3390 |
| 四元 | 全部 | 更低 | 6965 |

### 得分流 vs 概率流收敛目的地（500 随机初始化）

| 流类型 | 训练点 | 虚拟点 | 超盒边界 |
|--------|--------|--------|----------|
| 得分流 | 极少 | ~大多数 | 少量 |
| 概率流 | 较多 | 中等 | 较多 |

### 关键发现
- 得分流几乎全部收敛到虚拟点，因虚拟点数量远大于训练点
- 概率流中大噪声阶段将样本偏向训练点均值，进入低噪声后沿超盒边界滑动
- 训练样本数 $N$ 增加时，记忆化减少，更多样本收敛到训练点附近以外的流形点
- 无 weight decay 训练时概率流仅收敛到训练点或边界点；有 weight decay 时也收敛到虚拟点

## 亮点与洞察
1. **"虚拟训练点"概念**新颖重要——扩散模型可以生成「多个训练样本的组合」，这与 Stable Diffusion 拼接前景/背景的经验发现高度一致
2. **超盒结构**精确刻画了最小范数浅层去噪器的泛化空间——数据流形以超盒边界的形式隐式出现
3. **早停效应的理论/实践双重意义**：不仅是计算trick，而是从"只能记忆"到"可以泛化"的关键机制
4. 理论分析虽限于浅层网络+正交数据，但洞见（组合记忆、时间调度器影响泛化）具有普遍性
5. 用 Augmented Lagrangian 训练确保精确插值是巧妙的实验设计

## 局限与展望
1. **仅分析浅层 ReLU 网络**，与实际深层 U-Net 架构差异巨大，理论结论能否直接推广存疑
2. **正交数据假设**虽在高维空间近似成立，但实际数据分布更复杂
3. **低噪声区域的近似误差**未定量分析
4. 虚拟点的"语义组合"在浅层网络中仅是线性叠加，深层网络中的非线性组合需要进一步研究
5. 缺乏实际图像数据集上的验证

## 相关工作与启发
- **Carlini et al., 2023**：实验发现扩散模型会记忆训练数据
- **Somepalli et al., 2023**：Stable Diffusion 拼接前景/背景记忆对象——本文的"虚拟点"给出了理论解释
- **Zeno et al., 2023**：最小范数浅层去噪器的先前分析工作，本文扩展到流动力学
- 启示：理解扩散模型的泛化-记忆边界需要研究去噪器的归纳偏置，时间调度器是关键调控手段

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ （虚拟训练点+超盒结构+早停效应，理论贡献独特）
- 实验充分度: ⭐⭐⭐⭐ （合成数据验证充分，但缺少真实图像实验）
- 写作质量: ⭐⭐⭐⭐ （定理-证明结构清晰，直觉解释到位）
- 价值: ⭐⭐⭐⭐ （对理解扩散模型记忆化/泛化机制有重要理论价值）

<!-- RELATED:START -->

## 相关论文

- [Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes](understanding_and_mitigating_memorization_in_generative_models_via_sharpness_of_.md)
- [DDIS: When Model Knowledge Meets Diffusion Model](when_model_knowledge_meets_diffusion_model_diffusion-assisted_data-free_image_synthesis.md)
- [When Are Concepts Erased From Diffusion Models?](../../NeurIPS2025/image_generation/when_are_concepts_erased_from_diffusion_models.md)
- [Why Diffusion Models Don't Memorize: The Role of Implicit Regularization](../../NeurIPS2025/image_generation/why_diffusion_models_dont_memorize_the_role_of_implicit_regularization.md)
- [Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training](../../NeurIPS2025/image_generation/why_diffusion_models_dont_memorize_the_role_of_implicit_dynamical_regularization.md)

<!-- RELATED:END -->
