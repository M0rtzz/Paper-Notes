---
title: >-
  [论文解读] Revisiting Unbiased Implicit Variational Inference
description: >-
  [ICML2025][优化][半隐式变分推断] 重新审视被认为"不实用"的无偏隐式变分推断（UIVI），用重要性采样替代其内部 MCMC 循环，并通过最小化期望前向 KL 散度无偏地学习最优提议分布，在标准 SIVI 基准上达到或超越 SOTA。
tags:
  - ICML2025
  - 优化
  - 半隐式变分推断
  - 重要性采样
  - 条件归一化流
  - 路径梯度估计器
  - score gradient
---

# Revisiting Unbiased Implicit Variational Inference

**会议**: ICML2025  
**arXiv**: [2506.03839](https://arxiv.org/abs/2506.03839)  
**代码**: 待确认  
**领域**: 变分推断 / 优化  
**关键词**: 半隐式变分推断, 重要性采样, 条件归一化流, 路径梯度估计器, score gradient

## 一句话总结

重新审视被认为"不实用"的无偏隐式变分推断（UIVI），用重要性采样替代其内部 MCMC 循环，并通过最小化期望前向 KL 散度无偏地学习最优提议分布，在标准 SIVI 基准上达到或超越 SOTA。

## 研究背景与动机

### 变分推断与半隐式分布

变分推断（VI）通过在分布族 $\mathcal{Q}_z$ 中寻找最接近目标分布 $p_z$ 的近似分布 $q_z^*$ 来实现推断。**半隐式变分推断**（SIVI）提供了一种折中方案：通过从隐式分布 $q_y$ 中采样参数 $y$，再从显式条件分布 $q_{z|y}$ 中采样，构造出表达能力接近隐式分布但密度可估计的半隐式分布：

$$q_z(z) = \mathbb{E}_{\epsilon \sim p_\epsilon}[q_{z|\epsilon}(z|\epsilon)]$$

其中 $\epsilon$ 是潜变量，通过神经网络 $f_\phi$ 映射生成分布参数。

### UIVI 的困境

Titsias & Ruiz (2019) 提出的 UIVI 证明了一个关键等式：

$$\mathbb{E}_{\epsilon \sim q_{\epsilon|z}}[\nabla_z \log q_{z|\epsilon}(z|\epsilon)] = \nabla_z \log q_z(z)$$

即如果能从难以处理的条件分布 $q_{\epsilon|z}$ 中采样，就能无偏估计 score gradient $\nabla_z \log q_z(z)$。然而 UIVI 使用 MCMC 采样 $q_{\epsilon|z}$，由于该分布可能是多模态的，需要极长的马尔可夫链才能打破初始化依赖，使得该方法在计算上不可行，因此被学界基本放弃。

### 路径梯度估计器

本文的关键观察是，利用半隐式分布的可重参数化性质，可以使用 **路径梯度估计器**（path gradient estimator）来降低梯度估计的方差并大幅减少计算开销：

$$\nabla_\phi D_{\mathrm{KL}}(q_z \| p_z) = \mathbb{E}_{\epsilon,\eta}[\nabla_z(\log q_z(z) - \log p_z(z))\big|_{z=h_\phi(\epsilon,\eta)} \cdot \nabla_\phi h_\phi(\epsilon,\eta)]$$

这一结果虽然在之前的文献中出现过，但其深远意义未被充分讨论。

## 方法详解

### 核心思路：用重要性采样替代 MCMC

作者提出用重要性采样（IS）替代 UIVI 中的 MCMC 循环来估计 score gradient：

$$\nabla_z \log q_z(z) = \nabla_z \log\left(\mathbb{E}_{\epsilon \sim \tau_{\epsilon|z}}\left[\frac{p_\epsilon(\epsilon) q_{z|\epsilon}(z|\epsilon)}{\tau_{\epsilon|z}(\epsilon|z)}\right]\right)$$

其中 $\tau_{\epsilon|z}$ 是由**条件归一化流**（CNF）建模的提议分布。

### 关键理论结果

**命题 3.1**：当 $\tau_{\epsilon|z} = q_{\epsilon|z}$ 时，IS 估计器 $s_{\mathrm{IS},k}$ 变为无偏：

$$\mathbb{E}_{\epsilon_i \sim q_{\epsilon|z}} \nabla_z \log\left(\frac{1}{k}\sum_{i=1}^k \frac{p_\epsilon(\epsilon_i) q_{z|\epsilon}(z|\epsilon_i)}{q_{\epsilon|z}(\epsilon_i|z)}\right) = \nabla_z \log q_z(z)$$

**命题 3.2**：最小化期望前向 KL 散度 $\mathbb{E}_{z \sim q_z}[D_{\mathrm{KL}}(q_{\epsilon|z} \| \tau_{\epsilon|z})]$ 等价于最小化 $D_{\mathrm{KL}}(q_{z,\epsilon} \| \tau_{\epsilon|z} \cdot q_z)$，全局最优解恰好为 $\tau_{\epsilon|z}^* = q_{\epsilon|z}$。

### 两个算法

1. **BSIVI**（Base SIVI）：基准方法，使用朴素 Monte Carlo 估计器 $s_{\mathrm{MC},k}$ 来近似 score gradient，不使用重要性采样。尽管在高维中无信息的 $\epsilon_i$ 贡献几乎可忽略，但该方法表现出乎意料地好。

2. **AISIVI**（Adaptively Informed SIVI）：主方法，交替优化：
    - 最小化期望前向 KL 以训练 CNF 提议分布 $\tau_{\epsilon|z}$
    - 最小化反向 KL $D_{\mathrm{KL}}(q_z \| p_z)$ 以训练 SIVI 模型

   CNF 的训练损失可简化为条件对数似然的负均值：$\text{loss}_{\text{flow}} = -\frac{1}{m}\sum_{i=1}^m \log \tau_{\epsilon|z}(\epsilon_i | z_i)$

### 内存高效的批聚合

得益于路径梯度，增加采样数 $k$ 不会增加反向传播的计算成本。作者提出了一种基于 logaddexp 的批聚合方案，使得可以在**恒定内存**下处理任意数量的 $\epsilon_i$ 样本：

$$\ell_3(z,\tilde{z}) = \mathrm{logaddexp}(\ell_1(z,\tilde{z}) + \log j, \ell_2(z,\tilde{z})) - \log(j+1)$$

聚合后的 score 估计为两批次估计的加权组合，权重通过 log 空间计算保证数值稳定。

## 实验关键数据

### 实验 1：二维玩具分布

| 分布 | AISIVI ($D_{\mathrm{KL}}$↓) | BSIVI ($D_{\mathrm{KL}}$↓) |
|------|------|------|
| Banana | **0.0853** | 0.3022 |
| Multimodal | 0.0044 | **0.0017** |
| X-shape | 0.0072 | **0.0034** |

AISIVI 在 Banana 分布上显著优于 BSIVI，其余两例相当。

### 实验 2：贝叶斯逻辑回归（22 维）

在 WAVEFORM 数据集上，AISIVI、BSIVI、KSIVI、PVI 四种方法的边际和成对密度估计均与 SGLD ground truth 吻合良好，无系统性方差过/欠估计。成对相关系数散点图显示所有方法表现可比，PVI 和 KSIVI 略紧凑。

### 实验 3：条件扩散过程（100 维）

| 方法 | Log ML↑ | 训练时间 (s) | 迭代次数 |
|------|---------|-------------|---------|
| KSIVI | 74521 | 0.6k | 100k |
| **AISIVI** | **74062** | 1.4k | 10k |
| IWHVI | 67676 | 1.5k | 10k |
| BSIVI | 60556 | 1.5k | 10k |
| PVI | 53121 | 1.4k | 10k |
| UIVI | 40207 | 1.5k | 10k |

AISIVI 在相同计算预算下（10k 迭代）远超 IWHVI、BSIVI、PVI、UIVI，接近 KSIVI 的金标准水平（KSIVI 需 100k 迭代）。

## 亮点与洞察

- **化腐朽为神奇**：被认为"计算不可行"的 UIVI 通过简单的重要性采样替换 MCMC 而复活，这是一个优雅的理论修正
- **无偏性保证**：当提议分布 $\tau = q_{\epsilon|z}$ 时，IS 估计器严格无偏；即使不精确，只要 support 条件满足，估计器仍然一致
- **前向 KL 训练 CNF**：利用前向 KL 散度训练 CNF 是 mass-covering 的，天然保证了 support 条件，使交替优化在理论上成立
- **恒定内存训练**：基于 logaddexp 的批聚合方案允许任意增大隐变量采样数而不增加内存，非常实用
- **高维有效**：在 100 维条件扩散过程中，AISIVI 仅 10k 迭代即接近需要 100k 迭代的 KSIVI

## 局限与展望

- CNF 使用 affine coupling layers，虽可扩展但可能限制表达力，论文也提到替换为更灵活的 NF 架构可能有额外增益
- 在低维玩具示例中 AISIVI 并非全面优于 BSIVI（Multimodal 和 X-shape 上 BSIVI 更优），说明额外的 CNF 在低维可能带来不必要的开销
- 实验仅覆盖最高 100 维，更高维度（如大规模深度学习模型后验推断）的表现待验证
- 交替优化中 CNF 和 SIVI 模型的训练频率比需要调参，论文未给出系统性指导
- 未与完全隐式 VI 方法（如神经采样器）在同一框架下比较

## 相关工作与启发

- **SIVI 系列**：SIVI (Yin & Zhou 2018) → UIVI (Titsias & Ruiz 2019) → KSIVI (Cheng et al. 2024) → PVI (Lim & Johansen 2024, Wasserstein 梯度流)
- **路径梯度估计器**：Roeder et al. (2017) 的低方差梯度估计思想在 SIVI 设定下绽放新价值
- **重要性采样改进 VI**：IWAE (Burda et al. 2016), NVI (Zimmermann et al. 2021)
- **启发**：该工作说明看似过时的方法 + 新工具组合可以产生 SOTA 结果；条件归一化流作为辅助推断网络是一个值得推广的 pattern

## 评分
- 新颖性: ⭐⭐⭐⭐ — 理论洞察优雅，核心 idea（IS替代MCMC + 前向KL训练CNF）简洁有力
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 2D 玩具、22D 逻辑回归、100D 扩散过程，与多个 baseline 系统比较
- 写作质量: ⭐⭐⭐⭐⭐ — 数学推导严谨清晰，行文流畅，证明自包含
- 价值: ⭐⭐⭐⭐ — 为 SIVI 领域提供了新的强基线（BSIVI）和 SOTA 方法（AISIVI），理论与实践兼顾

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Synonymous Variational Inference for Perceptual Image Compression](synonymous_variational_inference_for_perceptual_image_compression.md)
- [\[NeurIPS 2025\] Least Squares Variational Inference](../../NeurIPS2025/optimization/least_squares_variational_inference.md)
- [\[NeurIPS 2025\] Brain-like Variational Inference](../../NeurIPS2025/optimization/brain-like_variational_inference.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](../../NeurIPS2025/optimization/viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference](../../NeurIPS2025/optimization/neuro-symbolic_entity_alignment_via_variational_inference.md)

<!-- RELATED:END -->
