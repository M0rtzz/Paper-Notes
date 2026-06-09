---
title: >-
  [论文解读] Information Theoretic Learning for Diffusion Models with Warm Start
description: >-
  [NeurIPS 2025][图像生成][扩散模型] 提出将经典 KL 散度-Fisher 信息关系推广到任意各向同性噪声扰动的似然估计框架，结合 warm-start 噪声注入和重要性采样，消除训练-测试差距并实现更紧的似然上界，在 ImageNet 多分辨率上达到 SOTA NLL。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "扩散模型"
  - "Likelihood Estimation"
  - "information theory"
  - "Fisher Divergence"
  - "Warm Start"
---

# Information Theoretic Learning for Diffusion Models with Warm Start

**会议**: NeurIPS 2025  
**arXiv**: [2510.20903](https://arxiv.org/abs/2510.20903)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: Diffusion Models, Likelihood Estimation, information theory, Fisher Divergence, Warm Start

## 一句话总结

提出将经典 KL 散度-Fisher 信息关系推广到任意各向同性噪声扰动的似然估计框架，结合 warm-start 噪声注入和重要性采样，消除训练-测试差距并实现更紧的似然上界，在 ImageNet 多分辨率上达到 SOTA NLL。

## 研究背景与动机

**领域现状**：似然（NLL）是评估密度估计和生成模型的基础指标，用于数据压缩、异常检测、对抗净化等。扩散模型是当前 SOTA 似然模型之一，VP（方差保持）通过 ELBO 训练，VE（方差爆炸）通过信息论工具直接估计。

**现有痛点**：
   - VP 模型的 ELBO 虽然性能强但收敛慢（数百万迭代）。
   - VE 模型的 IT 界虽收敛快，但似然性能不如 VP。
   - 扩散模型在 $\sigma_t^2 \to 0$ 时 SNR 发散导致数值不稳定，实践中从 $t = \epsilon > 0$ 开始训练引入训练-测试不匹配。
   - 离散数据（如图像）需 dequantization（均匀或变分），引入额外训练阶段且产生 noise mismatch。

**核心矛盾**：Score matching（Fisher 散度）仅在 $\sigma_t^2 \to 0$ 时才精确等价于 MLE，有限噪声下存在偏差；同时 $t=0$ 附近数值不稳定。

**本文目标**：在不增加训练成本的前提下，得到更紧的似然界，消除训练-测试差距，同时支持非高斯噪声。

**切入角度**：将扩散过程视为高斯信道（Gaussian Channel），利用 mismatched entropy 表达模型-数据差距，引入低方差非高斯 warm-start 噪声区间。

**核心 idea**：将 score matching 与 KL 散度的关系从高斯推广到任意各向同性噪声（Theorem 1），配合 warm-start 噪声注入统一训练和评估。

## 方法详解

### 整体框架

扩散模型前向过程 $\mathbf{Y}_t = \alpha_t \mathbf{X} + \sigma_t \mathbf{N}$，包含两个方差区间：
1. **低方差区间** $[0, \sigma_0^2)$：注入任意各向同性 warm-up 噪声 $\Psi$ 生成 $\tilde{\mathbf{x}} = \alpha_0 \mathbf{x} + \sigma(0)\mathbf{u}$。
2. **高方差高斯区间** $[\sigma_0^2, \sigma_1^2]$：标准高斯扩散过程。

训练和评估均对数据施加相同 warm-up 噪声，消除 mismatch。

### 关键设计

#### 1. Score Matching 与 KL 散度的推广关系（Theorem 1）
- **功能**：证明 Fisher 散度是 KL 散度对噪声方差的一阶导数极限，对任意各向同性噪声成立。
- **核心公式**：
  $$\frac{d}{d\sigma_t^2} D_{\text{KL}}(p_{\sigma_t^2} \| q_{\sigma_t^2})\bigg|_{\sigma_t^2 \to 0^+} = -\frac{1}{2} I(p(\mathbf{x}) \| q(\hat{\mathbf{x}}; \theta))$$
  其中 $I(\cdot \| \cdot)$ 为 Fisher 散度（score matching 目标）。
- **意义**：将之前仅限高斯的结果推广到 Poisson、Laplacian、均匀分布等任意噪声。

#### 2. Mismatched Entropy 分解（Proposition 1）
- **功能**：将模型似然精确分解为 output distribution 损失 + score 逼近误差 + 不可约信息量。
- **核心公式**：
  $$\mathcal{H}(p(\mathbf{x}), q(\hat{\mathbf{x}};\theta)) = \mathcal{H}(p(\mathbf{y}_1), \pi) + \mathcal{J}_{\text{DSM}}(\theta) - \frac{1}{2}\int_{\sigma_0^2}^{\sigma_1^2} \mathbb{E}\|\nabla_{\mathbf{y}_t} \log p(\mathbf{y}_t|\mathbf{x})\|^2 d\sigma_t^2 + o(\sigma_0^2)$$
- **设计动机**：提供信息论视角下的似然分解，明确每项来源。

#### 3. 逐点似然上界（Theorem 2）
- **核心公式**：
  $$-\log q(\hat{\mathbf{x}};\theta) \leq \mathcal{H}(p(\mathbf{y}_1|\mathbf{x}), \pi(\mathbf{x})) + \mathcal{L}_{\text{DSM}}(\sigma_t^2;\theta)$$
  其中 $\mathcal{L}_{\text{DSM}}$ 为 denoising score matching 损失，$\mathcal{H}(p(\mathbf{y}_1|\mathbf{x}), \pi)$ 可解析计算。
- **意义**：训练目标直接是逐点 NLL 上界，比 ELBO 更紧。

#### 4. Warm-Start Dequantization
- **功能**：用任意各向同性噪声 $\Psi$ 替代传统均匀 dequantization。
- **De Bruijn 恒等式推广**（Proposition 2）：微分熵对噪声方差的导数与噪声类型无关，仅取决于 Fisher 信息。
- **实践**：训练和测试均添加相同 warm-up 噪声，消除 train-test gap，无需额外训练阶段。

#### 5. 重要性采样
- 在 log-SNR 空间 $\eta = -\log \text{SNR}(t)$ 上设计采样分布 $\rho(\eta) \propto w(\eta)$。
- 使用噪声预测模型 $\hat{\mathbf{n}}(\mathbf{y}_t, \eta_t; \theta)$，时间嵌入改为 log-SNR 嵌入。

### 损失函数/训练策略

- 训练损失：$\nabla_\theta \|\mathbf{n} - \hat{\mathbf{n}}(\mathbf{y}_t, t; \theta)\|^2$（标准 score matching/denoising）。
- 评估损失：$\mathcal{L}_{\text{DSM}} = \frac{1}{2}\int_{\sigma_0^2}^{\sigma_1^2} \sigma_t^{-2} \mathbb{E}\|\mathbf{n} - \hat{\mathbf{n}}\|_2^2 d\sigma_t^2$，通过重要性采样估计。
- 架构：VDM 的 U-Net（无下采样的 ResNet blocks），嵌入 $\eta(t)$ 而非 $t$。
- 训练 0.3M 迭代，无数据增强，4 × V100 GPU。

## 实验关键数据

### 主实验

| 模型 | CIFAR-10 NLL↓ | ImageNet-32 NLL↓ | ImageNet-64 NLL↓ | ImageNet-128 NLL↓ | 训练迭代 |
|------|--------------|-----------------|-----------------|------------------|---------|
| VDM | 2.65 | 3.72 | 3.40 | — | 10M |
| i-DODE | 2.56 | 3.44 | — | — | 6.2M |
| W-PCDM | 2.35 | 3.32 | 2.95 | 2.64 | 2-10M |
| Flow Matching | 2.99 | 3.53 | 3.31 | 2.90 |  |
| **Ours (VP+IS)** | **2.50** | **3.01** | **2.91** | **2.59** | **0.3M** |

- ImageNet-32 NLL 从 3.32 降至 **3.01**（9.3% 提升）。
- ImageNet-128 NLL 从 2.64 降至 **2.59**。
- 训练仅需 **0.3M 迭代**，比 VDM 的 10M 少 33×。

### Warm-up 噪声消融

| 噪声类型 | CIFAR-10 (Ours) | ImageNet-32 (Ours) |
|---------|-----------------|-------------------|
| Gaussian | **2.50** | **3.00** |
| Laplace | 2.51 | 3.01 |
| Logistic | 2.51 | 3.03 |
| Uniform | 2.53 | 3.09 |

重尾指数族噪声（Gaussian、Laplace）优于轻尾噪声（Uniform），符合理论预期。

### 无损压缩

| 模型 | CIFAR-10 压缩率 (bits/dim) |
|------|--------------------------|
| VDM | 2.72 |
| W-PCDM | 2.37 |
| **Ours** | **2.57** |

### 关键发现

- 增大 warm-up 噪声可改善 NLL 但略微损害 FID——存在似然精度与感知质量的权衡。
- ELBO 与 IT 界的差距随噪声增大而扩大——因 ELBO 的像素独立性假设在高噪声下更不准确。
- VP schedule 略优于 SP（Straight-Path）schedule；VE schedule 性能显著落后（3.27 bits/dim）。

## 亮点与洞察

- **理论贡献**：将 Shannon-Fisher 关系从高斯推广到任意各向同性噪声（Theorem 1 + Proposition 2），统一了多种噪声下的 score matching-MLE 等价性。
- **实用价值**：0.3M 迭代即达 SOTA NLL，无数据增强——训练成本降低一个数量级。
- **无extra训练的 dequantization**：warm-start 噪声替代传统变分 dequantization，消除 train-test gap 且零额外训练开销。
- **log-SNR 参数化**：改进时间嵌入提升重要性采样效率。

## 局限与展望

- 仅关注似然估计，**不构建非高斯噪声下的生成过程**——无法用于采样/生成。
- Dequantization 策略和方差配置的最优选择未充分探索。
- FID 性能非最优（因优化目标是似然而非感知质量）。
- 未在更大分辨率（ImageNet-256+）上验证。
- 超参数和网络架构调优受限于计算资源。

## 相关工作与启发

- **VDM (Kingma et al.)**：变分扩散模型，ELBO 训练，CIFAR-10 NLL 2.65 但需 10M 迭代。
- **ScoreFlow (Song et al.)**：首次将 score matching 与 MLE 联系，但仅限高斯噪声。
- **i-DODE**：IT 框架下的 SOTA，但训练慢且需要数据增强。
- **W-PCDM**：级联扩散模型，NLL 强但训练更复杂。
- **启发**：信息论视角为扩散模型训练提供了更深刻的理解，warm-start 技术可推广到其他生成模型。

## 评分

⭐⭐⭐⭐ (4/5)

理论贡献扎实（推广 KL-Fisher 关系到任意噪声），实用价值高（0.3M 迭代 SOTA NLL），方法简洁且与标准扩散训练兼容。主要局限是不涉及采样/生成过程，仅是似然估计方法。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model](itdpdm_information-theoretic_discrete_poisson_diffusion_model.md)
- [\[NeurIPS 2025\] Information-Theoretic Discrete Diffusion](information-theoretic_discrete_diffusion.md)
- [\[NeurIPS 2025\] MMG: Mutual Information Estimation via the MMSE Gap in Diffusion](mmg_mutual_information_estimation_via_the_mmse_gap_in_diffusion.md)
- [\[NeurIPS 2025\] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [\[NeurIPS 2025\] Diffusion Models Meet Contextual Bandits](diffusion_models_meet_contextual_bandits.md)

</div>

<!-- RELATED:END -->
