---
title: >-
  [论文解读] Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis
description: >-
  [ECCV 2024][图像生成] 揭示 IMLE 方法中训练/测试时潜在码分布不对齐问题，提出 RS-IMLE 通过拒绝采样改变训练先验分布，在九个少样本图像数据集上平均降低 45.9% FID。
tags:
  - ECCV 2024
  - 图像生成
---

# Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis

**会议**: ECCV 2024  
**arXiv**: [2409.17439](https://arxiv.org/abs/2409.17439)  
**领域**: 图像生成

## 一句话总结

揭示 IMLE 方法中训练/测试时潜在码分布不对齐问题，提出 RS-IMLE 通过拒绝采样改变训练先验分布，在九个少样本图像数据集上平均降低 45.9% FID。

## 研究背景与动机

- 少样本图像合成旨在用极少训练样本（如 10-100 张）学习深度生成模型
- GAN 在少样本场景下严重受模式崩塌困扰；扩散模型由于学习各向同性高斯混合，在少样本时表现不佳
- IMLE（隐式最大似然估计）通过确保每个训练样本都有生成样本靠近它来避免模式崩塌
- **关键发现**：现有 IMLE 方法中，训练时通过 min 操作选取的潜在码分布与测试时从标准高斯采样的分布存在显著不对齐（misalignment issue）
- 这导致潜在空间中存在大量未被训练覆盖的"空白区域"，测试时采样到这些区域产生低质量样本

## 方法详解

### 整体框架

RS-IMLE 的核心思想：改变训练时的先验分布（从标准高斯变为通过拒绝采样得到的新分布 $\mathcal{P}$），使训练和测试时的潜在码分布更对齐。

### 关键设计

**1. 对齐问题的理论分析**

IMLE 中选取最近样本后的距离 CDF 为：

$$F_{D_i^*}(t) = 1 - (1 - F_{D_{i1}}(t))^m$$

其中 $m$ 是采样数。这说明选取码的距离分布相比随机码偏向更小值，且 $m$ 越大偏移越严重。

对应的 PDF：

$$f_{D_i^*}(t) = m(1-F_{D_{i1}}(t))^{m-1} f_{D_{i1}}(t)$$

**2. 理想先验设计**

目标是找到一个先验 $\mathcal{P}$，使得从 $\mathcal{P}$ 中采样 $m$ 个样本后经过 min 选取的分布与从标准高斯采样 $n$ 个（$n$ 为数据点数）样本后的选取分布一致。通过推导得到目标先验的 PDF：

$$f_{\tilde{D}_{i1}}(t) = \phi(t) \cdot f_{D_{i1}}(t)$$

其中 $\phi(t) = \frac{n}{m} \frac{(1-F_{D_{i1}}(t))^{n-1}}{(1-F_{\tilde{D}_{i1}}(t))^{m-1}}$

**3. 拒绝采样实现**

将上述理论简化为直观的拒绝采样步骤：
- 从标准高斯采样 $\mathbf{z} \sim \mathcal{N}(0, I)$
- 如果 $d(\mathbf{x}_i, T_\theta(\mathbf{z})) < \epsilon$（对所有数据点），则拒绝该样本
- 否则接受该样本

实质上是排除已经靠近某个数据点的样本，迫使模型训练在更具挑战性的非平凡样本上。

**4. 梯度视角的直观解释**

- 传统 IMLE：随着训练收敛，最近样本已非常接近数据点，损失和梯度变小导致学习停滞
- RS-IMLE：通过 $\epsilon$ 半径排除容易样本，每个数据点的损失始终 ≥ $\epsilon$，保持有意义的参数更新

### 损失函数

$$\theta_{\text{RS-IMLE}} = \arg\min_\theta \mathbb{E}_{z_1,...,z_m \sim \mathcal{P}} \left[\sum_{i=1}^n \min_{j \in [m]} d(\mathbf{x}_i, T_\theta(\mathbf{z}_j))\right]$$

约束：$d(\mathbf{x}_i, T_\theta(\tilde{\mathbf{z}}_j)) \geq \epsilon, \forall \tilde{\mathbf{z}}_j \in \tilde{Z}, i \in [n]$

使用 DCI 快速近邻搜索加速距离计算，并通过随机投影降维减少搜索复杂度。

## 实验关键数据

### 主实验

九个少样本数据集上的 FID 比较（256×256 分辨率）：

| 数据集 | FastGAN | FakeCLR | FreGAN | ReGAN | AdaIMLE | **RS-IMLE** | 提升% |
|--------|---------|---------|--------|-------|---------|-------------|-------|
| Obama | 41.1 | 29.9 | 33.4 | 45.7 | 25.0 | **14.0** | 44.0% |
| Grumpy Cat | 26.6 | 20.6 | 24.9 | 27.3 | 19.1 | **11.5** | 39.8% |
| Panda | 10.0 | 8.8 | 9.0 | 12.6 | 7.6 | **3.5** | 54.0% |
| FFHQ-100 | 54.2 | 62.1 | 50.5 | 87.4 | 33.2 | **12.9** | 61.1% |
| Cat | 35.1 | 27.4 | 31.0 | 42.1 | 24.9 | **15.9** | 36.1% |
| Dog | 50.7 | 44.4 | 47.9 | 57.2 | 43.0 | **23.1** | 46.3% |
| Anime | 69.8 | 77.7 | 59.8 | 110.8 | 65.8 | **35.8** | 45.6% |
| Skulls | 109.6 | 106.5 | 163.3 | 130.7 | 81.9 | **51.1** | 37.6% |
| Shells | 120.9 | 148.4 | 169.3 | 236.1 | 108.5 | **55.4** | 48.9% |

### 消融实验

Precision 和 Recall 对比（部分数据集）：

| 数据集 | 指标 | FastGAN | FakeCLR | FreGAN | ReGAN | AdaIMLE | **RS-IMLE** |
|--------|------|---------|---------|--------|-------|---------|-------------|
| Obama | Prec | 0.92 | 0.96 | 0.82 | 0.62 | 0.99 | 0.98 |
| Obama | Rec | 0.09 | 0.30 | 0.33 | 0.01 | 0.68 | **高** |
| Grumpy Cat | Prec | — | — | — | — | — | 接近1.0 |
| FFHQ-100 | Prec | — | — | — | — | — | 接近1.0 |

**核心结论**：RS-IMLE 在保持近乎完美精确度的同时，召回率显著高于所有基线方法。

2D 玩具问题的收敛对比证实：
- IMLE 训练中选取的潜在码形成紧密的窄带，中间有大量未覆盖空白
- RS-IMLE 训练中选取的潜在码更均匀地覆盖先验分布，与测试时分布对齐

### 关键发现

1. **平均 FID 下降 45.9%**，全部九个数据集上均为 SOTA
2. FFHQ-100 上改善最显著（61.1%），从 33.2 降到 12.9
3. Precision 接近 1.0（生成质量高），Recall 远超 GAN 方法（模式覆盖好）
4. 通过 2D 玩具实验直观验证了对齐问题：扩散模型在少样本时无法学习数据流形（混合高斯失效），而 RS-IMLE 有效解决
5. 超参数 $\epsilon$ 通过交叉验证选取，方法对其不过度敏感

## 亮点与洞察

- **理论驱动的方法改进**：从 order statistics 角度严格推导训练/测试分布不对齐的根源，非启发式修改
- **拒绝采样的优雅简化**：复杂的理论推导最终化为简单直观的 $\epsilon$-球排除规则
- **正交于现有 IMLE 改进**：RS-IMLE 与 Adaptive IMLE 互补，可叠加使用
- **扩散模型在少样本下的失败案例**：清楚展示了扩散模型的各向同性高斯假设在极少样本时的根本缺陷

## 局限性

- 拒绝采样会降低采样效率，需要额外采样以弥补拒绝样本
- $\epsilon$ 的选取需要交叉验证，不同数据集可能需要不同值
- 生成器架构基于 VDVAE 的解码器，未探索更现代的架构
- 仅在 256×256 分辨率上验证，未扩展到高分辨率生成
- 未与基于预训练模型适配的少样本方法（transfer learning）对比

## 评分

- 创新性：⭐⭐⭐⭐⭐ — 理论与方法均有突破
- 实用性：⭐⭐⭐⭐ — 少样本场景价值高
- 表现力：⭐⭐⭐⭐⭐ — 全部数据集SOTA，平均45.9%提升
- 综合评分：9/10

<!-- RELATED:START -->

## 相关论文

- [XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](xpsr_cross-modal_priors_for_diffusion-based_image_super-resolution.md)
- [OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization](mixdq_memory-efficient_few-step_text-to-image_diffusion_models_with_metric-decou.md)
- [Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [LCM-Lookahead for Encoder-based Text-to-Image Personalization](lcm-lookahead_for_encoder-based_text-to-image_personalization.md)

<!-- RELATED:END -->
