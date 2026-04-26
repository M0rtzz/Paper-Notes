---
title: >-
  [论文解读] Enhancing Diffusion Models with Text-Encoder Reinforcement Learning
description: >-
  [ECCV 2024][图像生成][扩散模型] 提出 TexForce，通过强化学习（DDPO）结合 LoRA 微调扩散模型的文本编码器以提升图文对齐和视觉质量，并可无缝与已有 U-Net 微调方法组合获得更优效果。
tags:
  - ECCV 2024
  - 图像生成
  - 扩散模型
  - 文本编码器
  - 强化学习
  - LoRA
  - 图文对齐
---

# Enhancing Diffusion Models with Text-Encoder Reinforcement Learning

**会议**: ECCV 2024  
**arXiv**: [2311.15657](https://arxiv.org/abs/2311.15657)  
**代码**: [GitHub](https://github.com/chaofengc/TexForce) (有)  
**领域**: 图像生成  
**关键词**: 扩散模型, 文本编码器, 强化学习, LoRA, 图文对齐

## 一句话总结

提出 TexForce，通过强化学习（DDPO）结合 LoRA 微调扩散模型的文本编码器以提升图文对齐和视觉质量，并可无缝与已有 U-Net 微调方法组合获得更优效果。

## 研究背景与动机

当前文本到图像扩散模型（如 Stable Diffusion）的训练目标是优化 log-likelihood，这与下游任务的特定需求（如图像美学、图文对齐）之间存在差距。现有的改进方法主要通过强化学习（DDPO、DPOK）或直接反向传播（ReFL、AlignProp）来微调 U-Net，但存在几个关键问题：

**忽视文本编码器**：几乎所有方法都将预训练的文本编码器固定不变，只调 U-Net。但文本编码器本身是次优的——用户经常需要精心调整 prompt 才能获得满意结果

**U-Net 微调的副作用**：微调 U-Net 容易破坏图像的视觉外观，导致风格退化或模式坍塌

**文本编码器作为瓶颈**：即使微调了 U-Net，次优的文本编码器仍然制约着整体性能

作者观察到一个关键现象：**微调 U-Net 倾向于通过改变视觉外观来提升奖励分数，而微调文本编码器则通过引入新的视觉概念来实现同样目标，后者在语义保持方面更优。**

## 方法详解

### 整体框架

TexForce 的核心很简洁：使用 DDPO（Denoising Diffusion Policy Optimization）算法，对文本编码器 $\tau_\phi$ 应用 LoRA 微调，基于任务特定的奖励函数进行优化。关键特性是：

1. 文本编码器通过 RL 微调，U-Net 冻结
2. 使用 LoRA 实现参数高效微调
3. 可与已有 U-Net 微调模型直接组合，无需额外训练

### 关键设计

#### 扩散模型中的强化学习

将去噪过程视为马尔可夫决策过程。文本编码器 $\tau_\phi$ 充当策略网络，将文本描述映射为动作（文本嵌入），影响扩散模型的生成过程。优化目标为最大化期望奖励：

$$J(\phi) = \mathbb{E}[R(\mathbf{x}_0, s)]$$

策略梯度可计算为：

$$\nabla_\phi J = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\phi \log p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t, \tau_\phi(s)) R(\mathbf{x}_0, s)\right]$$

实际使用 PPO（近端策略优化）来稳定训练：

$$J = \mathbb{E}[\min(r_t(\phi)A, \text{clip}(r_t(\phi), 1-\lambda, 1+\lambda)A)]$$

其中 $A$ 是归一化奖励的优势值，$r_t$ 是新旧策略的概率比。

#### LoRA 低秩适应

在文本编码器的前馈层中插入可训练的低秩矩阵：$W' = W + \alpha \Delta W$，$\Delta W$ 初始化为零。LoRA 的优势：

1. **防止过拟合**：低秩约束限制了参数空间
2. **灵活切换**：不同任务的 LoRA 权重可以灵活替换
3. **权重融合**：不同任务的 LoRA 权重可以直接加权融合以组合能力

#### 为什么微调文本编码器

从理论分析角度，在微调阶段使用少量 prompts $s$ 优化 ELBO 时：

$$\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}|s)}[\log(p_\theta(\mathbf{x}_{0:T}|\mathbf{z}))] - D_{KL}(q_\phi(\mathbf{z}|s) \| p(\mathbf{z}))$$

当 $\phi$ 固定时，$q_\phi$ 可能成为 $p(\mathbf{z})$ 的次优估计，导致 KL 散度项增大。因此**微调文本编码器 $\tau_\phi$ 可以减小 KL 项**，在有限数据下尤为重要。

#### U-Net 与文本编码器 LoRA 权重的直接融合

最重要的设计是：TexForce 的文本编码器 LoRA 权重可以**直接与已有的 U-Net LoRA 权重组合**，无需任何额外训练。实验表明这种简单的融合策略始终优于联合训练。

作者假设原因是：**固定的 U-Net 在文本编码器微调过程中充当了像素生成的先验**，联合微调会使文本编码器的优化变得更复杂。

### 损失函数 / 训练策略

- **奖励函数**：支持多种非可微奖励——ImageReward、HPSv2、JPEG 压缩大小、人脸质量评分、手部检测置信度等
- **优化算法**：PPO with importance sampling
- **LoRA 融合**：$\sum_i \alpha_i \theta_i$，不同任务的 LoRA 权重加权求和

## 实验关键数据

### 主实验

**ImageReward 数据集定量结果**

| 方法 | ImageReward 分数↑ |
|------|-----------------|
| SDv1.4 | 0.2154 |
| ReFL (U-Net微调) | 0.4485 |
| TexForce (文本编码器微调) | 0.4556 |
| ReFL + TexForce | **0.6553** |

**HPSv2 数据集定量结果**

| 方法 | HPSv2 分数↑ |
|------|-----------|
| SDv1.4 | 0.2752 |
| AlignProp (U-Net微调) | 0.2821 |
| TexForce | 0.2767 |
| AlignProp + TexForce | **0.2914** |

**多backbone 结果 (ImageReward)**

| Backbone | 原始 | ReFL | TexForce | ReFL+TexForce |
|----------|------|------|----------|--------------|
| SDv1.5 | 0.2140 | 0.5484 | 0.4086 | **0.6703** |
| SDv2.1 | 0.3891 | 0.5223 | 0.5084 | **0.6158** |

### 消融实验

**简单融合 vs 联合训练**

| 方法 | ImageReward 分数 |
|------|----------------|
| SDv1.4 | 0.2154 |
| ReFL 单独 | 0.4485 |
| TexForce 单独 | 0.4556 |
| 联合训练 | 0.5009 |
| 简单融合 (ReFL+TexForce) | **0.6553** |

简单融合大幅优于联合训练，验证了固定 U-Net 作为先验的重要性。

### 关键发现

1. **行为差异**：U-Net 微调改变视觉外观以追求奖励，文本编码器微调引入新概念实现相同目标——后者语义保持更好
2. **互补性**：两者学到的能力是互补的，简单组合即可获得超越联合训练的效果
3. **GPT-4V 评估**：TexForce 在文本图像一致性方面评分最高，ReFL 在视觉外观方面更好，组合方案最优
4. **跨backbone 鲁棒性**：方法在 SDv1.4、SDv1.5、SDv2.1 上均有效，甚至对已很强的 SDv2.1 仍有改进
5. **LoRA 权重可混合**：不同任务（如 ImageReward + 人脸质量）的 LoRA 权重可直接融合提升特定对象质量

## 亮点与洞察

1. **切入角度新颖**：几乎所有同期工作都聚焦 U-Net 微调，本文是最早系统研究文本编码器微调的工作
2. **简单但强大的组合策略**：无需额外训练即可融合不同微调方案的优势，极大降低了实践门槛
3. **理论与实证一致**：从 ELBO 分析出发给出微调文本编码器的理论动机，实验完美验证
4. **灵活性极高**：不需要可微奖励函数，任何可评价图像质量的指标都可作为奖励

## 局限与展望

1. RL 训练比直接反向传播更慢更昂贵（如图4所示，文本编码器的优化更具挑战性）
2. 仅在 Stable Diffusion 系列上验证，未测试 SDXL 或更新的架构
3. 不同奖励函数的 LoRA 权重融合系数 $\alpha_i$ 需要手动调节
4. 未研究文本编码器微调对 out-of-distribution prompts 的泛化能力
5. 未与同期工作 TextCraftor 进行直接对比

## 相关工作与启发

- **DDPO**：本文使用的基础 RL 算法，将扩散去噪过程建模为 MDP
- **ReFL / AlignProp / DRaFT**：直接反向传播方法，但更容易过拟合和模式坍塌
- **TextCraftor**：同期探索文本编码器微调的工作，但依赖可微奖励，灵活性更低
- 启发：文本编码器微调的思路可推广到视频生成、3D 生成等更多模态的扩散模型优化

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 理论深度 | 3.5 |
| 实验充分性 | 4.5 |
| 实用价值 | 4.5 |
| 写作质量 | 4 |
| 总体评分 | 4.1 |

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning](powerful_and_flexible_personalized_text-to-image_generation_via_reinforcement_le.md)
- [\[ECCV 2024\] LCM-Lookahead for Encoder-based Text-to-Image Personalization](lcm-lookahead_for_encoder-based_text-to-image_personalization.md)
- [\[ECCV 2024\] Lego: Learning to Disentangle and Invert Personalized Concepts Beyond Object Appearance in Text-to-Image Diffusion Models](lego_learning_to_disentangle_and_invert_personalized_concepts_beyond_object_appe.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)
- [\[CVPR 2026\] Refining Few-Step Text-to-Multiview Diffusion via Reinforcement Learning](../../CVPR2026/image_generation/refining_few-step_text-to-multiview_diffusion_via_reinforcement_learning.md)

<!-- RELATED:END -->
