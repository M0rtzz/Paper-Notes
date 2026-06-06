---
title: >-
  [论文解读] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models
description: >-
  [NeurIPS 2025][图像生成][扩散模型] 在低秩高斯混合（MoLRG）数据模型下，理论证明了扩散模型表示质量随噪声水平呈单峰动态的现象源于去噪强度与类别区分度的权衡，并实证发现单峰动态的出现可作为模型泛化能力的可靠指标。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "扩散模型"
  - "表示学习"
  - "单峰动态"
  - "低秩高斯混合"
  - "泛化-记忆"
---

# Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models

**会议**: NeurIPS 2025  
**arXiv**: [2502.05743](https://arxiv.org/abs/2502.05743)  
**代码**: 无  
**领域**: 图像生成 / 扩散模型理论  
**关键词**: 扩散模型, 表示学习, 单峰动态, 低秩高斯混合, 泛化-记忆

## 一句话总结

在低秩高斯混合（MoLRG）数据模型下，理论证明了扩散模型表示质量随噪声水平呈单峰动态的现象源于去噪强度与类别区分度的权衡，并实证发现单峰动态的出现可作为模型泛化能力的可靠指标。

## 研究背景与动机

扩散模型不仅在生成任务上取得巨大成功，近期研究还揭示了其卓越的**表示学习能力**。训练好的扩散模型的内部特征提取器可作为强大的自监督学习器，在分类、语义分割、图像对齐等下游任务上匹配甚至超越专用自监督方法。

一个广泛观察到的现象是**单峰表示动态（unimodal representation dynamics）**：学习到的表示质量（以下游任务性能衡量）随噪声水平呈单峰趋势——最佳特征出现在中间噪声水平，而完全噪声或完全干净的输入性能都较差（如Figure 1所示）。

**核心矛盾**：这一现象虽被广泛观察，但其底层原因一直缺乏理论理解。具体来说：(1) 为什么中间噪声水平产生最好的表示？(2) 什么驱动了这一单峰模式？(3) 这与模型的泛化能力有什么关系？

本文通过**低秩高斯混合（MoLRG）**数据假设，在简化但可解析的网络架构下回答这些问题。

## 方法详解

### 整体框架

三步走的理论分析框架：
1. 假设数据分布为MoLRG（满足自然图像的低维流形特性）
2. 设计可解析的去噪自编码器网络架构（模仿U-Net结构特性）
3. 定义信噪比（SNR）度量表示质量，推导其随噪声水平的解析表达

### 关键设计

1. **MoLRG数据模型**（Assumption 1）：数据分布为 $K$ 类噪声低秩高斯混合：
    $x_0 = U_k^\star a + \delta \tilde{U}_k^\star e, \quad \text{以概率 } \pi_k$
    - $U_k^\star \in \mathcal{O}^{n \times d}$：第 $k$ 类子空间正交基（类别相关属性）
    - $\tilde{U}_k^\star$：其他类子空间的基（类别无关细粒度属性，如背景）
    - $\delta$：数据噪声水平
   
   动因：(1) MoLRG捕捉真实图像的内禀低维性；(2) 潜扩散模型的KL惩罚推动隐空间趋向高斯分布；(3) 噪声项建模类别无关的复杂度

2. **网络参数化**：DAE和特征表示参数化为：
    $x_\theta(x_t, t) = U h_\theta(x_t, t), \quad h_\theta(x_t, t) = D(x_t, t) U^\top x_t$
    $D(x_t, t) = \text{diag}(\beta_1^t I_d, \ldots, \beta_K^t I_d)$
   
   其中 $\beta_l^t$ 通过softmax权重 $w_l(x_t, t)$ 实现数据和时间依赖的专家选择。这可解释为**浅层U-Net + 块状混合专家（MoE）**机制，包含低维投影、专家加权、对称重建三个组件。

3. **SNR表示质量度量**（Definition 1）：
    $\text{SNR}(\hat{x}_\theta, t) = \mathbb{E}_k\left[\frac{\mathbb{E}_{x_t}[\|U_k^\star \hat{h}_\theta(x_t, t)\|^2 | k]}{\mathbb{E}_{x_t}[\|\hat{x}_\theta(x_t, t) - U_k^\star \hat{h}_\theta(x_t, t)\|^2 | k]}\right]$
   分子衡量特征在正确类子空间上的投影能量（信号），分母衡量去除正确类投影后的残余能量（噪声）。

### 主要理论结果

**Proposition 1**：最优DAE的解析形式为加权投影：
$$\hat{x}_\theta^\star(x_t, t) = \sum_{l=1}^K w_l^\star(x_t, t)(\zeta_t U_l^\star U_l^{\star\top} + \xi_t \tilde{U}_l^\star \tilde{U}_l^{\star\top}) x_t$$
其中 $\zeta_t = 1/(1+\sigma_t^2)$ 和 $\xi_t = \delta^2/(\delta^2+\sigma_t^2)$。当 $\sigma_t$ 增大时 $\xi_t$ 比 $\zeta_t$ 衰减快得多，展现**从精细到粗糙的生成转变**。

**Theorem 1（核心）**：最优DAE的SNR近似为：
$$\text{SNR}(\hat{x}_\theta^\star, t) \approx \frac{C_t}{(K-1)} \cdot \left(\frac{1 + \frac{\sigma_t^2}{\delta^2}h(\hat{w}_t^+, \delta)}{1 + \frac{\sigma_t^2}{\delta^2}h(\hat{w}_t^-, \delta)}\right)^2$$

单峰性的物理图像：
- **去噪率** $\sigma_t^2/\delta^2$：随 $\sigma_t$ 单调递增
- **正类置信率** $h(\hat{w}_t^+, \delta)$：随 $\sigma_t$ 单调递减
- 初期 $\sigma_t$ 小时，类置信率稳定，去噪率增强提升SNR
- 后期 $\sigma_t$ 大时，类置信率急剧下降，$h(\hat{w}_t^+)$ 趋近 $h(\hat{w}_t^-)$，SNR下降
- 中间存在平衡点：类无关成分被最大抑制且类相关特征被最好保留 → SNR峰值

## 实验关键数据

### 理论验证

| 数据集 | SNR单峰 | Feature Probing单峰 | 两者对齐 | 说明 |
|--------|---------|-------------------|---------|------|
| MoLRG合成 | ✓ | ✓ | ✓ | 理论与实验完美匹配 |
| CIFAR-10 | ✓ | ✓ | ✓ | SNR峰值与probing准确率峰值位置一致 |
| TinyImageNet | ✓ | ✓ | ✓ | 同上 |

### 泛化-记忆实验

| 训练数据量 | UNet-32 泛化分数 | 表示动态 | 说明 |
|-----------|-----------------|---------|------|
| $2^{15}$ (大) | 高 | 单峰 | 泛化好 → 单峰 |
| $2^{12}$ (中) | 中 | 弱单峰 | 过渡期 |
| $2^8$ (小) | 低 | 单调递减 | 记忆化 → 单峰消失 |

### 训练动态实验（$N=2^{12}$）

| 训练阶段 | FID | Peak Probing Acc | 表示动态 | 说明 |
|---------|-----|-----------------|---------|------|
| 早期 (Iter≤7.5M) | 下降 | 上升 | 单峰 | 泛化阶段 |
| 晚期 (Iter=15M+) | 上升 | 下降 | 单调递减 | 记忆化阶段 |

### 关键发现

- 单峰表示动态的存在是扩散模型泛化良好的**可靠指标**
- 从单峰到单调递减的转变与泛化→记忆的相变精确对应
- FID与peak probing accuracy呈一致的负相关
- 表示动态的转变可作为有限数据下防止过拟合的**早停准则**

## 亮点与洞察

- 首次给出单峰表示动态的理论解释，揭示其源于"去噪强度与类别区分度的权衡"
- SNR度量简洁有效，在合成和真实数据上都与probing准确率对齐
- MoLRG数据假设虽简化但物理动机充分（低维流形+KL正则化+子空间结构）
- 泛化-记忆的相变与表示动态的联系是一个重要的实践洞察

## 局限与展望

- MoLRG数据假设限制了理论的直接适用性，真实图像分布远比高斯混合复杂
- 网络参数化高度简化（相当于浅层U-Net），无法直接推广到实际的深层U-Net
- 假设了正交子空间、等维度、均匀混合权重，放松这些假设的理论分析是开放问题
- 表示质量定义（SNR）依赖已知的真实子空间基，实际应用中需要PCA近似

## 相关工作与启发

- **vs Chen et al. (2024)**：后者分析二层CNN的扩散学习优化动态，聚焦去噪vs分类目标的对比，不涉及跨时间步的表示质量变化
- **vs Wang et al. (2024)**：后者也研究时间步对扩散表示学习的影响，但关注属性分类和反事实生成，无理论解释
- **vs REPA (2024)**：REPA通过对齐扩散特征与预训练自监督模型特征提升训练效率，本文的理论框架可为其提供更深理解

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次从理论上解释扩散模型表示学习中最重要的经验现象之一
- 实验充分度: ⭐⭐⭐⭐ 合成和真实数据的验证系统，泛化-记忆实验设计巧妙
- 写作质量: ⭐⭐⭐⭐⭐ 理论直觉解释清晰（去噪率vs类置信率的权衡），图示精美
- 价值: ⭐⭐⭐⭐⭐ 为扩散模型表示学习提供理论基础，可能指导更原则性的特征提取和早停策略

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models](shallow_diffuse_robust_and_invisible_watermarking_through_low-dimensional_subspa.md)
- [\[NeurIPS 2025\] Elucidated Rolling Diffusion Models for Probabilistic Forecasting of Complex Dynamics](elucidated_rolling_diffusion_models_for_probabilistic_forecasting_of_complex_dyn.md)
- [\[NeurIPS 2025\] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)
- [\[NeurIPS 2025\] Preconditioned Langevin Dynamics with Score-Based Generative Models for Infinite-Dimensional Linear Bayesian Inverse Problems](preconditioned_langevin_dynamics_with_score-based_generative_models_for_infinite.md)
- [\[NeurIPS 2025\] Cross-fluctuation Phase Transitions Reveal Sampling Dynamics in Diffusion Models](cross-fluctuation_phase_transitions_reveal_sampling_dynamics_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
