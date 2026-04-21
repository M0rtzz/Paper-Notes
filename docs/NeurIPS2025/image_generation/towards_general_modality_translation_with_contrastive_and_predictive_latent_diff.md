---
title: >-
  [论文解读] Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge
description: >-
  [NeurIPS 2025][图像生成][扩散桥模型] 提出 LDDBM（Latent Denoising Diffusion Bridge Model），将去噪扩散桥模型扩展到共享潜空间中，结合对比对齐损失和预测损失，实现任意模态之间的通用翻译框架。
tags:
  - NeurIPS 2025
  - 图像生成
  - 扩散桥模型
  - 跨模态翻译
  - 对比学习
  - 潜空间
  - Transformer
---

# Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge

**会议**: NeurIPS 2025  
**arXiv**: [2510.20819](https://arxiv.org/abs/2510.20819)  
**代码**: [项目页面](https://sites.google.com/view/lddbm/home)  
**领域**: 扩散模型 / 跨模态翻译  
**关键词**: 扩散桥模型, 跨模态翻译, 对比学习, 潜空间, Transformer

## 一句话总结

提出 LDDBM（Latent Denoising Diffusion Bridge Model），将去噪扩散桥模型扩展到共享潜空间中，结合对比对齐损失和预测损失，实现任意模态之间的通用翻译框架。

## 研究背景与动机

扩散模型在单模态生成（如图像、音频）中表现卓越，但将其用于**模态翻译（Modality Translation, MT）**——即在不同感官模态之间转换信息——仍然是开放挑战。现有方法存在以下局限性：

**维度共享假设**: 去噪扩散桥模型（DDBM）要求源和目标分布共享同一维度空间 $\mathbb{R}^d$，无法处理 2D→3D 等异构模态翻译

**高斯源先验**: 标准扩散模型从简单先验分布开始，限制了在任意分布间翻译的灵活性

**模态特定架构**: U-Net 等架构天然适合网格数据，但在抽象或非结构化模态上表现不佳

**潜在桥方法的局限**: 现有潜空间扩散桥主要出于计算效率考虑，未针对通用 MT 任务设计

作者观察到，独立训练的自编码器会导致潜空间不对齐（通过 t-SNE 可视化验证），而简单的桥损失无法保留高频细节。这些发现驱动了对比损失和预测损失的设计。

## 方法详解

### 整体框架

LDDBM 的框架分为三步：（i）用模态特定编码器将源和目标样本编码到共享潜空间；（ii）在潜空间中用 Transformer 去噪器实现扩散桥；（iii）用模态解码器将预测的潜变量解码回目标模态。推理时只需源编码器 $E_y$、桥模型 $B$、目标解码器 $D_x$。

### 关键设计

1. **潜变量桥建模**: 给定目标模态分布 $p(x)$（$x \in \mathbb{R}^k$）和源模态分布 $p(y)$（$y \in \mathbb{R}^s$，$k \neq s$），引入中间潜变量 $z_0, z_T \in \mathbb{R}^d$ 将条件分布分解为 $p(x|y) = p(z_T|y) \cdot p(z_0|z_T) \cdot q(x|z_0)$。桥 $p(z_0|z_T)$ 用 DDBM 建模，消除了维度匹配的要求。

2. **预测损失（Predictive Loss）**: 与传统两阶段重建损失不同，预测损失直接约束完整的编码-桥-解码流水线：
   $$\mathcal{L}_{\text{pred}} = d(D_x \circ B \circ E_y(y), x)$$
   其中 $B$ 为桥模型。该损失鼓励模型在整个流水线中保持语义一致性，替代了源和目标独立的自编码器损失，减少了计算开销并提供单向监督。

3. **对比对齐损失（Contrastive Loss）**: 利用配对数据的结构，将 $(z_0, z_T)$ 视为正样本对，批内其他样本为负样本对：
   $$\mathcal{L}_{\text{infoNCE}} = \log \frac{\phi(z_0, z_T)}{\phi(z_0, z_T) + \sum_{j=1}^{M} \phi(z_0, z_T^j)}$$
   其中 $\phi(u,v) = \exp(u^T v / \tau |u| |v|)$，温度参数 $\tau = 0.5$。该损失拉近语义相关样本、推远不相关样本，有效对齐跨模态潜空间。

4. **编码器-解码器 Transformer 架构**: 采用 Transformer 编码器处理源模态 token $z_T$ 生成 memory，再通过 Transformer 解码器的交叉注意力层条件化去噪。引入可学习的 $[\text{MASK}]$ token 作为输出 token，增强表达能力。时间步嵌入用于调制自注意力和前馈网络输出。

### 损失函数 / 训练策略

总目标函数为：
$$\mathcal{L} = \mathcal{L}_{\text{bridge}} + \mathcal{L}_{\text{pred}} + \mathcal{L}_{\text{infoNCE}}$$

训练采用**交替迭代策略**——在重建和桥对齐之间交替优化，灵感来自对抗训练。这是因为桥假设边缘分布固定，但可训练编码器会不断改变潜空间结构，两者目标存在冲突。交替训练在稳定性和最终性能之间取得了最佳平衡。

## 实验关键数据

### 主实验：多视图→3D 形状生成（ShapeNet）

| 方法 | 1-NNA ↓ | IoU ↑ |
|------|---------|-------|
| Pix2Vox-A（专用模型） | - | 0.697 |
| EDM | 0.532±0.013 | 0.631±0.006 |
| 3D-EDM | 0.575±0.009 | 0.602±0.003 |
| DiT | 0.548±0.004 | 0.613±0.011 |
| SiT | 0.563±0.007 | 0.604±0.003 |
| **LDDBM** | **0.508±0.005** | **0.664±0.002** |

### 零样本超分辨率（FFHQ→CelebA-HQ, 16×16→128×128）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| EDM | 23.1±0.7 | 0.58±0.05 | 0.41±0.02 |
| DiWa（专用模型） | 23.3 | 0.65 | 0.39 |
| DiT | 22.2±1.1 | 0.52±0.07 | 0.49±0.01 |
| SiT | 21.5±0.4 | 0.57±0.02 | 0.51±0.03 |
| **LDDBM** | **25.6±0.4** | **0.68±0.03** | **0.32±0.01** |

### 消融实验

**架构消融：**

| 组件累加 | ShapeNet IoU ↑ | ShapeNet 1-NNA ↓ | CelebA PSNR ↑ |
|----------|----------------|-------------------|----------------|
| U-Net | 0.635 | 0.518 | 23.2 |
| DiT | 0.613 | 0.548 | 22.2 |
| + Encoder-Decoder | 0.651 | 0.518 | 23.4 |
| + Spatial Embedding | 0.658 | 0.522 | 22.9 |
| + [MASK]（完整模型） | **0.664** | **0.508** | **25.6** |

**损失消融：**

| 配置 | ShapeNet 1-NNA ↓ | CelebA LPIPS ↓ | 说明 |
|------|-------------------|-----------------|------|
| $\mathcal{L}_{\text{rec}}$ | 0.625 | 0.62 | 基础重建 |
| $\mathcal{L}_{\text{pred}}$ | 0.522 | 0.41 | 预测损失显著改善 |
| $\mathcal{L}_{\text{pred}} + \mathcal{L}_{\text{infoNCE}}$ | **0.508** | **0.32** | 完整配置最优 |

### 关键发现

- LDDBM 在所有通用基线上一致取得最优，且在 edges→bags 翻译中推理速度比 DDBM 快 2 倍以上
- 跨模态语音↔人脸翻译中，LDDBM 在通用基线中也表现最强（Face→Voice 71.2% vs SiT 65.7%）
- 预测损失比独立的自编码器损失更有效，同时减少了计算开销

## 亮点与洞察

- **从计算效率到任务通用性的视角转变**: 以往潜空间桥主要为降低计算成本，本文首次系统化地探索其在通用 MT 中的潜力
- **对比+预测的双重约束**: 对比损失负责全局对齐，预测损失保证端到端保真度，两者互补形成强有力的训练信号
- **"翻译器"视角下的架构选择**: 受 NLP 翻译任务启发，选择编码器-解码器 Transformer 而非仅解码器 DiT，这一决策得到消融实验有力验证

## 局限与展望

- 目前仅支持配对翻译，未探索非配对模态翻译
- 编码器/解码器仍为模态特定组件，需要为新模态设计
- 未扩展到序列或高维数据（视频、体积表示）
- 完整框架训练成本较高（交替训练策略）

## 相关工作与启发

- DDBM 是基础框架但限于同维度翻译，本文的潜空间扩展自然消除了这一限制
- CrossFlow/FlowTok 使用 Flow Matching 做文本-图像翻译，但绑定特定任务设计
- CLIP 风格的对比学习被巧妙迁移到桥模型的潜空间对齐中

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将 DDBM 扩展到潜空间并加入对比/预测损失的思路清晰且有效
- **实验充分度**: ⭐⭐⭐⭐⭐ 涵盖 3D、超分、场景生成、音频-图像翻译等多个任务，消融详尽
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，动机与设计逻辑链完整
- **价值**: ⭐⭐⭐⭐ 提供了通用 MT 的强基线，但实际应用仍受限于配对数据需求

<!-- RELATED:START -->

## 相关论文

- [System-Embedded Diffusion Bridge Models](system-embedded_diffusion_bridge_models.md)
- [CORAL: Disentangling Latent Representations in Long-Tailed Diffusion](coral_disentangling_latent_representations_in_longtailed_dif.md)
- [EBDM: Exemplar-guided Image Translation with Brownian-bridge Diffusion Models](../../ECCV2024/image_generation/ebdm_exemplar-guided_image_translation_with_brownian-bridge_diffusion_models.md)
- [Contrastive Flow Matching (ΔFM)](../../ICCV2025/image_generation/contrastive_flow_matching.md)
- [LoRACLR: Contrastive Adaptation for Customization of Diffusion Models](../../CVPR2025/image_generation/loraclr_contrastive_adaptation_for_customization_of_diffusion_models.md)

<!-- RELATED:END -->
