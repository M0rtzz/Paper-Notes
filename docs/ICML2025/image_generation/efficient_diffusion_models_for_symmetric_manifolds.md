---
title: >-
  [论文解读] Efficient Diffusion Models for Symmetric Manifolds
description: >-
  [ICML 2025][图像生成][对称流形] 提出一种高效的对称流形（环面、球面、SO(n)、U(n)）扩散模型框架，通过欧几里得布朗运动的投影和Itô引理绕过热核计算，将训练复杂度从指数级降至近线性，并提供多项式级采样精度保证。
tags:
  - ICML 2025
  - 图像生成
  - 对称流形
  - 黎曼扩散模型
  - 热核绕过
  - Itô引理
  - 采样保证
---

# Efficient Diffusion Models for Symmetric Manifolds

**会议**: ICML 2025  
**arXiv**: [2505.21640](https://arxiv.org/abs/2505.21640)  
**代码**: 无  
**领域**: 扩散模型 / 流形生成模型  
**关键词**: 对称流形, 黎曼扩散模型, 热核绕过, Itô引理, 采样保证

## 一句话总结

提出一种高效的对称流形（环面、球面、SO(n)、U(n)）扩散模型框架，通过欧几里得布朗运动的投影和Itô引理绕过热核计算，将训练复杂度从指数级降至近线性，并提供多项式级采样精度保证。

## 研究背景与动机

许多应用（机器人学、药物发现、量子物理）的数据天然约束在非欧流形上，如环面 $\mathbb{T}_d$、球面 $\mathbb{S}_d$、特殊正交群 SO(n)、酉群 U(n)（$d \approx n^2$）。

现有流形扩散模型面临的核心瓶颈：

| 问题 | 欧几里得空间 | 流形上 |
|------|------------|--------|
| 热核计算 | 封闭形式（高斯） | 无封闭形式，需 $2^d$ 级别求和 |
| 梯度计算 | $O(1)$ 次 | $O(d)$ 次（ISM目标需Riemannian散度） |
| 前向采样 | 直接采样高斯 | 需SDE/ODE求解器 |
| 每迭代算力 | $O(d)$ | 指数级或 $O(d)$ 次梯度 |

**本文目标**：缩小流形与欧几里得扩散模型之间在训练效率和采样保证方面的差距。

## 方法详解

### 整体框架

**核心创新**：引入空间变化协方差（spatially-varying covariance）的扩散模型，使前向扩散可以表示为欧几里得布朗运动通过投影映射 $\varphi$ 投射到流形上。

**前向过程**：
- 在 $\mathbb{R}^d$ 中运行Ornstein-Uhlenbeck过程 $Z_t$
- 投影到流形：$X_t := \varphi(Z_t)$
- 由于OU过程有封闭形式的高斯转移核，前向采样**无需SDE求解器**

**反向过程**：通过Itô引理将 $\mathbb{R}^d$ 中的反向SDE投影到流形上：
$$dY_t = f^\star(Y_t, t)dt + g^\star(Y_t, t)dB_t$$

### 关键设计

#### 1. 投影映射的选择

| 流形 | 投影映射 $\varphi$ | 计算复杂度 |
|------|-------------------|----------|
| 环面 $\mathbb{T}_d$ | $\varphi(x)[i] = x[i] \mod 2\pi$ | $O(d)$ |
| 球面 $\mathbb{S}_d$ | $\varphi(x) = x/\|x\|$ | $O(d)$ |
| SO(n) / U(n) | 谱分解 $U^*\Lambda U$，取 $U$ | $O(n^\omega)=O(d^{\omega/2})$ |

其中 $\omega \approx 2.37$ 是矩阵乘法指数。

#### 2. 高效训练目标（绕过热核）

利用Itô引理推导训练目标（Algorithm 1）：
- **漂移项**训练：学习 $f(x,t)$ 逼近 $f^\star(x,t)$，目标函数仅需 $\nabla\varphi$ 和欧几里得热核（封闭形式）
- **协方差项**训练：学习 $g(x,t)$，利用流形对称性将 $d \times d$ 协方差矩阵的自由参数降到 $n^2$ 个标量

#### 3. 平均情况Lipschitz条件（Assumption 2.1）

投影映射 $\varphi$ 不是全局Lipschitz的（如SO(n)上特征值重合时有奇异性）。本文利用随机矩阵理论证明**平均情况**下的Lipschitz性：
- U(n): $L_1 = O(d^{1.5}\sqrt{T}\alpha^{-1/3})$, $L_2 = O(d^2 T \alpha^{-2/3})$
- 球面: $L_1 = L_2 = O(\alpha^{-1/d})$
- 环面: $L_1 = L_2 = 1$（恒成立）

### 损失函数 / 训练策略

训练由两个独立优化问题组成：
1. 最小化漂移模型 $f_\theta$ 的MSE损失（涉及 $\nabla\varphi$, $\nabla^2\varphi$）
2. 最小化协方差模型 $g_\phi$ 使其逼近 $J_\varphi^T J_\varphi$

使用SGD优化，每次迭代仅需**1次**模型梯度评估 + $O(d^{\omega/2})$ 算术运算。

## 实验关键数据

### 主实验

**训练速度**（Table 3）：
- 在SO(n)和U(n)上（$d>1000$），每迭代训练时间仅为欧几里得扩散模型的3倍以内
- 相比RSGM（热核版本）有指数级加速

**样本质量**（Table 2, Figure 1）：
- 在环面、SO(n)、U(n)上的wrapped Gaussian混合模型和量子演化算子数据集上
- C2ST和似然分数均优于先前方法
- 维度越高，改善幅度越大

### 消融实验

- 不同维度下训练时间对比
- 协方差模型 vs 固定协方差的效果对比

### 关键发现

**每迭代复杂度对比**（Table 1）：

| 方法 | 梯度次数 | SO(n)/U(n)算力 |
|------|---------|---------------|
| RSGM (热核) | 1 | $2^d + \text{poly}(d,1/\delta)$ |
| RSGM (ISM) | $d$ | $\text{poly}(d,1/\delta)$ |
| TDM (ISM) | $d$ | $\text{poly}(d,1/\delta)$ |
| **本文** | **1** | $d^{\omega/2}\log(1/\delta)$ |

**采样保证**（Theorem 2.2 / Corollary 2.3）：
- SO(n)/U(n): $\|\nu - \pi\|_{TV} < O(\varepsilon \cdot d^9 \log(d/\varepsilon))$
- 环面/球面: $\|\nu - \pi\|_{TV} < O(\varepsilon \cdot d^6 \log(d/\varepsilon))$
- 迭代次数: poly(d) · log(d/ε)

## 亮点与洞察

1. **从根本上绕过热核**：不是近似热核，而是通过新的扩散定义完全避免热核计算
2. **空间变化协方差的妙用**：引入额外自由度来补偿流形曲率，使得投影框架成为可能
3. **流形对称性的利用**：SO(n)/U(n)的协方差由 $n^2$ 个标量参数完全确定，实现亚线性计算
4. **首个poly(d)采样保证**：先前方法的采样精度/运行时保证有未显式指定的流形依赖常数，本文给出显式多项式界
5. **Optimal Transport耦合**：开发了基于最优传输的概率耦合方法替代Girsanov变换，适用于空间变化协方差SDE

## 局限与展望

1. **仅限对称流形**：理论保证需要流形对称性和平均情况Lipschitz条件，非对称流形仍是开放问题
2. **poly(d)中d的幂次较大**：如 $d^9$ 的精度界和 $d^{5.5}$ 的迭代次数，改善维度依赖是未来方向
3. **实验仅在合成数据上**：缺少在真实应用（如蛋白质、分子构象生成）上的验证
4. **训练目标非凸**：与所有扩散模型一样，无法保证训练过程整体收敛
5. **协方差模型增加了模型复杂度**：需要额外训练 $g_\phi$，增加了实现复杂性

## 相关工作与启发

- **RSGM** [De Bortoli et al., 2022]：基于热核的黎曼score-based模型
- **RDM** [Huang et al., 2022]：黎曼扩散模型，使用ISM目标
- **TDM** [Yim et al., 2023]：三角化动量扩散模型
- **SCRD** [Urain et al., 2023]：缩放黎曼扩散
- **Flow matching on manifolds** [Chen & Lipman, 2023]：流形上的流匹配
- Appendix D讨论了向凸多面体等非光滑空间扩展的可能性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 全新的投影+Itô引理框架，从根本上改变了流形扩散的计算范式
- 实验充分度: ⭐⭐⭐ — 理论贡献突出但实验局限于合成数据
- 写作质量: ⭐⭐⭐⭐⭐ — 论文结构严谨，Algorithm伪代码清晰
- 价值: ⭐⭐⭐⭐⭐ — 在流形扩散模型的效率和理论保证方面迈出重要一步

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)
- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](../../NeurIPS2025/image_generation/encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[CVPR 2025\] Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](../../CVPR2025/image_generation/efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)
- [\[CVPR 2025\] DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention](../../CVPR2025/image_generation/dig_scalable_and_efficient_diffusion_models_with_gated_linear_attention.md)
- [\[CVPR 2026\] Reviving ConvNeXt for Efficient Convolutional Diffusion Models](../../CVPR2026/image_generation/reviving_convnext_for_efficient_convolutional_diffusion_models.md)

<!-- RELATED:END -->
