---
title: >-
  [论文解读] Nested Diffusion Models Using Hierarchical Latent Priors
description: >-
  [CVPR 2025][图像生成][嵌套扩散] 本文提出嵌套扩散模型，用一系列从粗到细的扩散模型逐级生成不同语义层级的潜变量，每级以上级输出为条件，在 ImageNet 256×256 上仅增加 25% 计算量便将无条件 FID 从 45.19 降至 11.05，有条件 FID 降至 3.97。
tags:
  - CVPR 2025
  - 图像生成
  - 嵌套扩散
  - 分层潜变量
  - 语义先验
  - 信息压缩
  - 非马尔可夫生成
---

# Nested Diffusion Models Using Hierarchical Latent Priors

**会议**: CVPR 2025  
**arXiv**: [2412.05984](https://arxiv.org/abs/2412.05984)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 嵌套扩散、分层潜变量、语义先验、信息压缩、非马尔可夫生成

## 一句话总结

本文提出嵌套扩散模型，用一系列从粗到细的扩散模型逐级生成不同语义层级的潜变量，每级以上级输出为条件，在 ImageNet 256×256 上仅增加 25% 计算量便将无条件 FID 从 45.19 降至 11.05，有条件 FID 降至 3.97。

## 研究背景与动机

1. **领域现状**：扩散模型（DiT 等）在图像生成上取得 SOTA，但无条件生成的质量远不如类条件生成（FID 45.19 vs 13.75）。
2. **现有痛点**：无条件生成缺乏语义引导，扩散过程需要从纯噪声中"发明"所有语义信息——极其困难。
3. **核心矛盾**：增大模型（DiT-XL 118 GFlops）收益递减但成本大增；需要更高效的方案引入语义先验。
4. **本文目标**：通过分层潜变量为扩散模型提供从粗到细的语义引导，在低额外开销下大幅提升生成质量。
5. **切入角度**：预训练视觉编码器（MoCo-v3/CLIP）的不同层级和尺度的特征天然包含不同粒度的语义信息。
6. **核心 idea**：$L$ 级嵌套——最粗级从噪声生成全局语义，每个细级以所有更粗级为条件生成更详细的特征，最终级生成像素。

## 方法详解

### 整体框架

预训练编码器提取多尺度特征 → SVD 降维 → 高斯噪声注入控制信息量 → $L$ 级扩散模型：$z_L$（最粗）→ $z_{L-1}$ → ... → $z_1 = x$（图像）。每级去噪器以所有更粗层级的输出为条件（非马尔可夫）。

### 关键设计

1. **分层潜变量构建**

    - 功能：从图像中提取不同语义粒度的特征作为训练目标
    - 核心思路：预训练视觉编码器在不同 patch 尺度提取特征 → SVD 降维防止信息过完备 → 高斯噪声注入 $\tilde{z}_l \sim \mathcal{N}(z_l, \sigma_l^2 I)$ 控制 KL 散度（信息容量）
    - 设计动机：噪声注入至关重要——$\sigma=0$ 时退化为自编码器（方法失效），$\sigma=1$ 时信息容量最大但增加学习难度。消融证实 $\sigma^2=1.0$ 最优

2. **非马尔可夫条件化**

    - 功能：每级扩散模型利用所有更粗层级的信息
    - 核心思路：第 $l$ 级去噪器条件化于 $z_{>l} = \{z_{l+1}, ..., z_L\}$ 的完整集合，而非仅前一级
    - 设计动机：马尔可夫链会丢失粗层级信息（经过多级传递后衰减）；非马尔可夫保证每级都能直接使用全局语义

3. **分层 CFG 权重衰减**

    - 功能：在推理时平衡不同层级的引导强度
    - 核心思路：CFG 权重从粗到细递减 $\{w_i\} = [0.5, 0.4, 0.3, 0.2, 0.1]$——粗层级提供更强引导，细层级更自由
    - 设计动机：粗层级决定全局语义（需要强引导），细层级决定细节多样性（过强引导会丢失多样性）

### 损失函数 / 训练策略

$\mathcal{L} = \sum_{l=1}^{L-1} \mathbb{E}[||\epsilon_l - D_{\theta_l}(\alpha^{(t)} z_l + \beta^{(t)} \epsilon_l, \tilde{z}_{>l}, t)||^2] + \mathbb{E}[||\epsilon_L - D_{\theta_L}(...)||^2]$。U-ViT-Base 架构，ImageNet 200 epochs。

## 实验关键数据

### 主实验

| 方法 | GFlops | 无条件 FID↓ | 条件 FID↓ |
|------|--------|-----------|----------|
| DiT-L/2 | 80.0 | - | 23.3 |
| DiT-XL/2+REPA | 118.6 | - | 12.3 |
| Baseline (L=1) | 27.0 | 45.19 | 13.75 |
| **Nested L=5** | **34.0** | **11.05** | **3.97** |

### 消融实验

| 层数 | 无条件 FID | 条件 FID | 说明 |
|------|-----------|---------|------|
| L=1 | 45.19 | 13.75 | 基线 |
| L=2 | 20.66 | 5.31 | 粗层即大幅提升 |
| L=3 | 19.00 | 4.69 | 递减收益 |
| L=5 | **11.05** | **3.97** | 最优 |

### 关键发现

- 无条件 L=5 (FID 11.05) 超越了条件基线 (FID 13.75)——分层先验比类标签更有效
- 仅 25% 额外计算量（27→34 GFlops），但 FID 降低 75%
- 噪声注入 $\sigma^2=0$ 时 FID 暴涨至 19.04——信息压缩是方法成功的关键

## 亮点与洞察

- **无条件超越有条件**：分层语义先验提供了比类标签更丰富的引导信息
- **25% 开销换 75% FID 下降**：极高的效率-质量比
- **信息压缩理论基础**：通过 KL 散度控制每级信息量有严格的信息论依据

## 局限与展望

- 超参数 $\{\sigma_l\}$ 需要逐级调优，虽然贪心搜索缓解但仍增加调参复杂度
- 仅探索到 L=5，更深层级的收益/代价未知
- 依赖预训练视觉编码器的特征质量

## 相关工作与启发

- **vs DiT-XL+REPA**: 118.6 GFlops 才达 FID 12.3，嵌套模型 34 GFlops 就达 3.97——3.5 倍更高效
- **vs 级联扩散（Imagen）**: 级联在像素空间做分辨率递增，嵌套在语义空间做粒度递增——不同维度的分层

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 分层语义先验的嵌套扩散概念很新颖
- 实验充分度: ⭐⭐⭐⭐ ImageNet全面评测+消融，但缺少更多数据集
- 写作质量: ⭐⭐⭐⭐ 理论分析充分
- 价值: ⭐⭐⭐⭐⭐ 大幅提升无条件生成质量，有广泛影响

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Hierarchical Flow Diffusion for Efficient Frame Interpolation](hierarchical_flow_diffusion_for_efficient_frame_interpolation.md)
- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [\[ICML 2025\] Learning Single Index Models with Diffusion Priors](../../ICML2025/image_generation/learning_single_index_models_with_diffusion_priors.md)
- [\[CVPR 2025\] FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution](faithdiff_unleashing_diffusion_priors_for_faithful_image_super-resolution.md)
- [\[CVPR 2025\] LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting](luminet_latent_intrinsics_meets_diffusion_models_for_indoor_scene_relighting.md)

<!-- RELATED:END -->
