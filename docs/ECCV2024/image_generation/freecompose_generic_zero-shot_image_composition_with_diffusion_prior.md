---
title: >-
  [论文解读] FreeCompose: Generic Zero-Shot Image Composition with Diffusion Prior
description: >-
  [ECCV 2024][图像生成][图像合成] 提出 FreeCompose，利用预训练扩散模型的生成先验实现通用零样本图像合成，统一覆盖图像和谐化（外观编辑）和语义图像合成（语义编辑），无需额外训练。
tags:
  - ECCV 2024
  - 图像生成
  - 图像合成
  - 扩散先验
  - 零样本
  - 图像和谐化
  - 语义图像合成
---

# FreeCompose: Generic Zero-Shot Image Composition with Diffusion Prior

**会议**: ECCV 2024  
**arXiv**: [2407.04947](https://arxiv.org/abs/2407.04947)  
**代码**: [GitHub](https://github.com/aim-uofa/FreeCompose) (有)  
**领域**: 图像生成  
**关键词**: 图像合成, 扩散先验, 零样本, 图像和谐化, 语义图像合成

## 一句话总结

提出 FreeCompose，利用预训练扩散模型的生成先验实现通用零样本图像合成，统一覆盖图像和谐化（外观编辑）和语义图像合成（语义编辑），无需额外训练。

## 研究背景与动机

图像合成（Image Composition）是计算机视觉中的基础任务，目标是将一张图像的前景对象与另一张图像的背景融合生成自然连贯的图像，在图像修复、艺术设计、游戏开发、虚拟现实等领域有广泛应用。

现有方法面临的关键问题：
- **数据稀缺**：传统学习方法依赖前景-背景-合成图像的三元组训练数据，难以获取，导致泛化能力受限
- **任务割裂**：图像和谐化（只调整低级统计信息如颜色光照）和语义图像合成（涉及结构变化）通常由不同模型分别处理，缺乏统一框架
- **领域特定**：现有模型通常在特定领域数据集上训练，难以泛化到开放世界场景

作者发现了一个关键洞察：**预训练的扩散模型能够自动识别简单复制粘贴产生的不自然边界区域**。在去噪过程中，这些区域被识别为低密度区域（梯度更新更大），与实际的不和谐区域高度一致。基于此发现，可以引导合成图像向高密度区域优化，实现自然的图像合成。

## 方法详解

### 整体框架

FreeCompose 的核心思想是利用预训练扩散模型的图像先验来实现通用图像合成，包含三个阶段：

1. **对象移除阶段（Object Removal）**：从背景图像 $I_s$ 中移除指定区域 $M_s$ 内的对象，生成干净背景 $I_b$
2. **图像和谐化阶段（Image Harmonization）**：将目标对象合成到背景中，调整光照、颜色等使之自然融合
3. **语义图像合成阶段（Semantic Image Composition）**：根据额外条件（文本或sketch）进行语义级别的结构编辑

每个阶段共享相同的通用管线：输入图像 $I_i$、原始提示 $P_o$、目标提示 $P_t$，通过特定损失函数优化图像像素。

### 关键设计

#### 1. 核心观察：扩散先验自动定位不自然区域

作者进行了关键实验验证：将简单复制粘贴的合成图像加入不同程度的噪声后，冻结的扩散模型会预测一个梯度用于更新图像。实验发现：
- 低密度区域（梯度更新大的区域）与复制粘贴造成的不和谐区域高度一致
- 这意味着扩散模型的先验能够自动"感知"哪些区域是不自然的

#### 2. 基于 DDS 损失的优化框架

方法基于 Delta Denoising Score (DDS) 损失进行优化。DDS 损失的梯度形式为：

$$\nabla_\theta \mathcal{L}_{DDS} = (\epsilon_\phi^w(\mathbf{z_t}, y, t) - \epsilon_\phi^w(\hat{\mathbf{z_t}}, \hat{y}, t)) \frac{\partial \mathbf{z_t}}{\partial \theta}$$

其中使用两对图像-文本对，通过匹配的时间步和噪声来计算差分去噪分数，指导图像优化方向。

#### 3. 对象移除阶段的 Mask-Guided Loss

在对象移除阶段，仅使用 DDS 损失不足以完全消除对象。作者提出了创新的 mask-guided DDS loss：
- 在 UNet 的自注意力层中，通过掩码 $M$ 选择性丢弃与目标区域对应的 K、V 值
- 具体操作：将掩码调整到序列长度 $l$，选择 $v_i > threshold$ 的索引，以此替换被掩盖区域的语义信息

掩码引导损失的梯度为：
$$\nabla_\theta \mathcal{L}_{DDS}^{rmv} = (\epsilon_\phi^w(\mathbf{z_t}, y, t) - \epsilon_\phi^w(\hat{\mathbf{z_t}}, \hat{y}, t, M)) \frac{\partial \mathbf{z_t}}{\partial \theta}$$

对象移除的总损失：
$$\mathcal{L}_{rmv} = \mathcal{L}_{DDS}^{rmv}(I_s, I_t, P_o, P_t, M_s) + \lambda_{per} \mathcal{L}_{per}(I_s \otimes M_s', I_t \otimes M_s')$$

perceptual loss 用于保持掩码外背景区域的一致性。

#### 4. 图像和谐化阶段

将目标对象复制粘贴到干净背景后形成 $I_p$，使用三项损失函数优化：

$$\mathcal{L}_{har} = \mathcal{L}_{DDS}(I_p, I_t, P_o, P_t) + \lambda_{bak}\mathcal{L}_{per}(I_p \otimes M_p', I_t \otimes M_p') + \lambda_{for}\mathcal{L}_{per}(I_p \otimes M_p, I_t \otimes M_p)$$

关键设计：**前景和背景分别使用不同权重的 perceptual loss**，前景权重 $\lambda_{for}=0.1$ 较小（允许更多变化以融入背景），背景权重 $\lambda_{bak}=0.3$ 较大（保持背景稳定）。默认提示为空字符串和 "A harmonious scene."。

#### 5. 语义图像合成阶段

支持文本或其他形式（sketch、canny edge，通过 T2I-Adapter 转换）的条件输入。关键创新是 **K、V 值替换策略**用于保持对象身份一致性：

$$\text{Attention}(Q, K_i, V_i), \text{if } t > T \text{ and } l > L$$

其中 $T=400$、$L=10$ 是控制替换开始时间和层深度的超参数。只在优化前期和深层进行替换，以同时保证身份一致和编辑灵活性。

### 损失函数 / 训练策略

**整体特点：Training-free**，无需训练扩散模型本身，通过优化合成图像的像素（在latent space中）来实现合成效果。

优化细节：
- 使用 Stable Diffusion V2.1 作为真实图像的预训练模型，AnyLoRA 用于动漫/卡通风格
- 输入分辨率对齐到 512×512
- Adam 优化器，固定学习率 $5 \times 10^{-2}$
- 对象移除：150步，DDS 损失在掩码外乘以0.2抑制背景变化，$\lambda_{per}=0.3$
- 图像和谐化：200步，$\lambda_{bak}=0.3$，$\lambda_{for}=0.1$
- 语义合成（文本条件）：500步；sketch/canny条件：200步

## 实验关键数据

### 主实验

论文通过用户研究定量评估，每项由20+位志愿者参与，5个案例，评分1-5分制。

| 方法 | 图像和谐度 ↑ | 对象移除完整性 ↑ |
|------|------------|---------------|
| Repaint | 3.24±1.23 | 3.82±1.35 |
| SD Inpainting | 2.99±1.37 | 3.55±1.34 |
| Lama | 3.47±1.16 | 4.14±0.94 |
| **FreeCompose** | **3.85±1.01** | **4.47±0.73** |

| 方法 | 图像和谐度 ↑ | 对象身份保持 ↑ |
|------|------------|-------------|
| Diff Harmonization | 3.11±1.04 | 3.83±1.10 |
| DucoNet | 3.14±1.17 | **4.16±1.04** |
| **FreeCompose** | **3.69±1.07** | 4.11±0.92 |

### 消融实验

对象移除阶段各组件的分解验证：

| 组件 | 效果 |
|------|------|
| 仅 perceptual loss | 图像几乎不变 |
| 仅 vanilla DDS | 对象部分变化但无法完全消除 |
| DDS + mask | 成功移除对象，但影响背景 |
| DDS + mask + perceptual | 移除对象且保持背景（完整方法） |

图像和谐化阶段的消融：

| 组件 | 效果 |
|------|------|
| 仅 perceptual | 保持与原始复制粘贴一致，无融合 |
| 仅 DDS | 实现融合但可能丢失前景/背景特征 |
| 完整方法 | 融合度、对象身份和背景特征的最优平衡 |

### 关键发现

1. **Mask-guided K,V 操作**是对象移除成功的关键——仅靠 DDS 损失和文本提示无法完全消除对象
2. **前后景分离的 perceptual loss** 有效平衡了融合度与保真度的矛盾
3. 方法可即插即用到 SDXL 等更强大的模型上，获得更好的效果（特别是和谐化阶段的反射/阴影效果）
4. 运行时间：RTX 3090 FP16 下，前50步约30秒，后续每50步约25秒

## 亮点与洞察

1. **统一框架**：首次将图像和谐化和语义图像合成统一在同一个零样本框架中，不需要针对不同任务训练不同模型
2. **关键洞察的普适性**：扩散先验能自动识别不自然区域的发现具有广泛的理论和应用价值
3. **Mask-guided KV 操作**：在自注意力层面通过掩码控制语义信息流的思路新颖，且可扩展到其他需要空间控制的任务
4. **多样化应用**：除基础合成外，还可用于对象风格化和多角色定制等下游任务

## 局限与展望

1. **对象身份保持不如训练型方法**：图像和谐化中为了更好的融合效果，降低了前景 perceptual loss 权重，导致身份保持略逊于 DucoNet
2. **运行速度偏慢**：每个阶段需要数百步优化迭代，实时应用受限
3. **依赖掩码质量**：需要输入准确的对象掩码，对复杂场景的自动分割能力未涉及
4. **未来方向**：探索视频合成扩展、更多合成任务的应用

## 相关工作与启发

- **DDS (Delta Denoising Score)**：本工作的核心损失函数来源，从文本驱动编辑扩展到图像合成场景
- **SDS/VSD**：3D生成中的类似思路（DreamFusion/ProlificDreamer），说明扩散先验在多领域的通用价值
- **Prompt-to-Prompt / MasaCtrl**：通过注意力操控实现编辑的系列工作，与本文的 KV 操作思路互补

## 评分

- **创新性**：★★★★☆ — 扩散先验定位不自然区域的洞察新颖，mask-guided KV 设计巧妙
- **实用性**：★★★☆☆ — 零样本+即插即用，但速度限制了实际部署
- **实验充分度**：★★★☆☆ — 用户研究设计合理，但缺少大规模定量评估
- **写作质量**：★★★★☆ — 方法阐述清晰，消融实验逻辑严谨

<!-- RELATED:START -->

## 相关论文

- [OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [MultiGen: Zero-Shot Image Generation from Multi-modal Prompts](multigen_zero-shot_image_generation_from_multi-modal_prompts.md)
- [DreamMover: Leveraging the Prior of Diffusion Models for Image Interpolation with Large Motion](dreammover_leveraging_the_prior_of_diffusion_models_for_image_interpolation_with.md)
- [Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)
- [OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching](../../ACL2025/image_generation/ozspeech_one-step_zero-shot_speech_synthesis_with_learned-prior-conditioned_flow.md)

<!-- RELATED:END -->
