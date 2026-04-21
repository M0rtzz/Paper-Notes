---
title: >-
  [论文解读] OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models
description: >-
  [ECCV 2024][图像生成] 提出 OMG，一种遮挡友好的个性化多概念图像生成框架，通过两阶段采样（布局生成 + 概念噪声融合）实现强身份保持和自然光照协调，且可即插即用地搭配各种单概念模型（LoRA、InstantID）无需额外训练。
tags:
  - ECCV 2024
  - 图像生成
---

# OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models

**会议**: ECCV 2024  
**arXiv**: [2403.10983](https://arxiv.org/abs/2403.10983)  
**领域**: 图像生成

## 一句话总结

提出 OMG，一种遮挡友好的个性化多概念图像生成框架，通过两阶段采样（布局生成 + 概念噪声融合）实现强身份保持和自然光照协调，且可即插即用地搭配各种单概念模型（LoRA、InstantID）无需额外训练。

## 研究背景与动机

个性化文本到图像生成中的多概念定制是一个重要但困难的问题。现有方法面临三大挑战：

**身份退化**：在单张图像中同时生成多个概念时，各概念的身份保持显著下降（实验证明同时生成两个概念的 Identity Alignment 明显低于分别生成）

**遮挡处理失败**：Mix-of-show 等方法在概念区域重叠时，通过简单线性加法融合 cross-attention 结果，导致布局冲突和身份退化

**前景-背景不协调**：多概念图像中前景和背景的光照常常不自然

**额外训练开销**：现有方法需要额外的网络优化来合并多个概念模型，计算开销大

## 方法详解

### 整体框架

OMG 采用**两阶段采样方案**：

- **阶段一（布局生成 + 视觉理解信息收集）**：使用不含个性化标识符的文本生成具有合理布局的非定制图像，同时收集 cross-attention map 和概念 mask
- **阶段二（多概念个性化去噪）**：利用收集的视觉理解信息和提出的概念噪声融合策略，在考虑遮挡的情况下整合多个概念

### 关键设计

**视觉理解信息准备（阶段一）**：

输入仅包含类别名（如"man"、"woman"）而非个性化标识符的文本 $p$，通过 SDXL 生成非定制图像 $x_{ncus} = T2I(p)$。在去噪过程中存储每步的 cross-attention map $A_t$。利用视觉理解模型从生成图像和类别名中提取概念 mask $\{M_1, M_2, \cdots, M_k\}$。

**概念噪声融合（Concept Noise Blending）**：

核心公式：

$$z_{t-1} = (1 - \bigcup_{i=0}^{k} M_i) * z'_{t-1} + \sum_{i=0}^{k} M_i * C_{t-1}^i$$

其中 $z'_{t-1}$ 是全局非定制噪声，$C_{t-1}^i = T2I_c^i(z_t, p^i, t)$ 是第 $i$ 个单概念模型在其专属文本 prompt $p^i$ 下生成的概念噪声。每个单概念模型仅负责生成特定区域，有效缓解身份退化。

**遮挡布局保持**：

在第二阶段的每个时间步，用阶段一存储的 attention map 覆盖当前生成的 attention map，保持与非定制图像一致的布局：

$$z'_{t-1} = T2I(z_t, p, t)\{A_t^g \leftarrow A_t\}$$

**去噪起始时间步的影响**：

实验发现概念噪声融合的起始步数对身份保持和布局有关键影响。早期步控制图像布局，后期步展现概念身份。最优起始步约为第 35 步（总 50 步 DDIM）。光照不协调与图像布局信息相关，随步数后移逐渐协调。

### 损失函数

OMG 在推理阶段不需要训练，仅修改采样过程。单概念 LoRA 模型的训练使用标准扩散损失：

$$\mathcal{L} = E_{z_0, \epsilon, t} \|\epsilon - \varepsilon_\theta(z_t, t, c)\|_2^2$$

LoRA rank 设为 256，文本编码器学习率 $3e^{-5}$，UNet 学习率 $3e^{-3}$。

## 实验关键数据

### 主实验

**单概念 & 多概念个性化定量比较**：

| 方法 | Character ID↑ (Single/Multi) | Object ImgAlign↑ (Single/Multi) |
|------|-----|-----|
| DreamBooth | 0.456 / 0.480 | 0.805 / 0.800 |
| Textual Inversion | 0.292 / 0.294 | 0.784 / 0.781 |
| Custom Diffusion | 0.370 / 0.322 | 0.840 / 0.778 |
| Mix-of-show | 0.422 / 0.436 | 0.791 / 0.780 |
| **OMG** | **0.514 / 0.510** | **0.842 / 0.810** |

**多概念组合生成效果**：

| 方法 | man+woman | man+man | woman+woman | object+object | Average |
|------|-----------|---------|-------------|---------------|---------|
| DreamBooth | 0.302 | 0.258 | 0.192 | 0.784 | 0.384 |
| Textual Inversion | 0.122 | 0.131 | 0.064 | 0.675 | 0.248 |
| Custom Diffusion | 0.265 | 0.210 | 0.212 | 0.757 | 0.361 |
| Mix-of-show | 0.361 | 0.219 | 0.143 | 0.736 | 0.365 |
| **OMG** | **0.487** | **0.377** | **0.293** | **0.763** | **0.480** |

### 消融实验

消融验证了每个组件的贡献：

- **布局保持**：加入后图像结构更合理，避免生成变形
- **概念噪声融合 vs 区域可控采样**：Mix-of-show 的区域可控采样在遮挡区域产生概念缺失和布局混乱，而概念噪声融合能有效处理遮挡，同时实现前景背景光照协调
- **不同概念数量**（1→5）：即使概念数增加到 5 个，仍能保持每个概念的身份一致性

### 关键发现

- 单概念到多概念的身份退化差距 $\Delta$ 仅为 -0.004（Identity Alignment），远低于 Custom Diffusion 的 -0.048
- OMG 在多概念平均得分（0.480）显著超越第二名 DreamBooth（0.384），提升 25%
- 与 InstantID 结合时无需额外训练，生成的颜色更自然真实

## 亮点与洞察

1. **即插即用架构**：可直接使用 civitai.com 社区的 LoRA 模型，无需额外训练或模型合并，极大降低了多概念定制的使用门槛
2. **优雅的遮挡处理**：通过两阶段解耦布局和身份注入，巧妙解决了多概念遮挡的老大难问题
3. **关键超参数发现**：噪声融合起始步数同时控制身份保持和光照协调，提供了直观的调节手段
4. **灵活的单概念模型兼容性**：同时支持训练式（LoRA）和无训练式（InstantID）的单概念方法

## 局限性

- 依赖视觉理解模型（如分割模型）提取概念 mask，mask 质量影响最终效果
- 每个概念需要单独的模型推理，概念数增加时计算开销线性增长
- 对同类概念（如两个 woman）的区分能力仍有提升空间（woman+woman 的 IDA 仅 0.293）

## 评分

⭐⭐⭐⭐ (4/5)

- 新颖性：★★★★ — 两阶段采样 + 概念噪声融合的设计新颖直观
- 技术：★★★★ — 无需训练的推理时方法，工程性强
- 实验：★★★★★ — 对比全面，包含单概念/多概念/消融，用户研究和定量指标齐全
- 实用性：★★★★★ — 直接兼容社区模型，落地价值高

<!-- RELATED:START -->

## 相关论文

- [M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning](powerful_and_flexible_personalized_text-to-image_generation_via_reinforcement_le.md)
- [Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)
- [Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)

<!-- RELATED:END -->
