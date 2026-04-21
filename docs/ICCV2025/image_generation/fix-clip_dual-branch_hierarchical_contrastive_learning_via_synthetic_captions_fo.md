---
title: >-
  [论文解读] Fix-CLIP: Dual-Branch Hierarchical Contrastive Learning via Synthetic Captions for Better Understanding of Long Text
description: >-
  [ICCV 2025][图像生成][CLIP] Fix-CLIP 通过三大创新模块提升 CLIP 的长文本理解能力：（1）双分支训练管线用短文本配合 masked 图像、长文本配合原始图像分别对齐；（2）带单向掩码的可学习区域提示（Regional Prompts）提取局部视觉特征；（3）层级特征对齐模块对齐中间层多尺度特征。在 30M 合成长文本数据上增量训练后，长文本检索和短文本检索均大幅超越 SOTA，文本编码器可即插即用提升扩散模型长文本生成质量。
tags:
  - ICCV 2025
  - 图像生成
  - CLIP
  - 长文本理解
  - 双分支训练
  - 区域提示
  - 层级特征对齐
  - 合成数据
---

# Fix-CLIP: Dual-Branch Hierarchical Contrastive Learning via Synthetic Captions for Better Understanding of Long Text

**会议**: ICCV 2025  
**arXiv**: [2507.10095](https://arxiv.org/abs/2507.10095)  
**代码**: [GitHub](https://github.com/bcwang-sjtu/Fix-CLIP)  
**领域**: 视觉语言模型/CLIP改进  
**关键词**: CLIP, 长文本理解, 双分支训练, 区域提示, 层级特征对齐, 合成数据

## 一句话总结
Fix-CLIP 通过三大创新模块提升 CLIP 的长文本理解能力：（1）双分支训练管线用短文本配合 masked 图像、长文本配合原始图像分别对齐；（2）带单向掩码的可学习区域提示（Regional Prompts）提取局部视觉特征；（3）层级特征对齐模块对齐中间层多尺度特征。在 30M 合成长文本数据上增量训练后，长文本检索和短文本检索均大幅超越 SOTA，文本编码器可即插即用提升扩散模型长文本生成质量。

## 研究背景与动机

### 核心问题

CLIP 在短文本任务（图像分类、检索）上表现优异，但受制于文本编码器 77 token 的输入长度限制，面对长文本输入时表现急剧下降。实际上，一张图像往往需要几十句话才能充分描述其内容，这严重限制了 CLIP 在 MLLM 和文生图模型中的应用。

### 现有方法的不足

**长文本能力提升以牺牲短文本为代价**：Long-CLIP 通过 ShareGPT4V 微调 CLIP 提升长文本理解，但短文本任务性能下降

**从头训练成本高**：DreamLIP、LoTLIP、FLAIR 需要在大规模合成数据集上从头训练，资源消耗巨大

**全局对齐不够精细**：标准对比学习仅对齐 [CLS] token 的全局特征，缺乏局部对齐能力，导致细粒度描述任务表现不佳

**显式区域匹配方法代价高**：需要为大量图像区域生成对应描述，数据规模和资源占用大

**隐式局部一致性方法会损害泛化**：会影响预训练模型的泛化能力，导致短文本性能退化

### 本文策略

Fix-CLIP 采用**增量训练**策略，在预训练 CLIP 基础上微调而非从头训练，同时通过精心设计的双分支管线在提升长文本能力的同时**保持甚至增强短文本能力**。

## 方法详解

### 整体框架

Fix-CLIP 包含三个核心模块：
1. **双分支训练管线**：短文本对齐 masked 图像，长文本对齐原始图像
2. **区域提示 + 单向掩码**：在图像编码器中注入可学习的区域提示以提取局部特征
3. **层级特征对齐**：在编码器中间层对齐多尺度视觉-语言特征

### 关键设计 1：双分支训练管线

**位置编码扩展**：继承 CLIP 原始 77 个位置编码 $PE$，冻结前 20 个位置编码（涵盖 CLIP 的有效文本长度），对 21-77 号位置编码进行 4 倍线性插值扩展到 248 token：

$$PE_l = \text{Concat}(PE[:20], \text{Intpol}(PE[20:], 4))$$

其中 $\text{Intpol}(PE, q)[i] = (1-\lambda) \cdot PE[\lfloor \frac{i}{q} \rfloor] + \lambda \cdot PE[\lfloor \frac{i}{q} \rfloor + 1]$，仅这些扩展后的位置编码参数可学习。

**差异化编码策略**：
- **短文本分支**：随机 mask 75% 的图像 patch embedding（替换为可学习零初始化参数），与短文本配对做对比学习。直觉来源于 MAE —— 75% mask 后仍保留足够语义信息
- **长文本分支**：使用完整原始图像 patch，与长文本配对做对比学习。长文本包含的细节信息需要完整图像来匹配

### 关键设计 2：区域提示与单向掩码

**动机**：[CLS] token 通过注意力机制聚合全局特征，但局部信息识别不足。

**设计**：在图像编码器第 $l$ 层 Transformer 中插入 $M$ 个可学习区域提示 $R_1^l, \ldots, R_M^l$，每层替换为新的可学习参数（消除跨深度层的信息干扰）。

**单向注意力掩码**：

$$\text{Attn} = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} \odot \mathbf{Mask}\right) V$$

掩码设计规则（关键创新）：
- **[CLS]** 可以看到所有区域提示和 patch embedding
- **Patch embedding** 只能看到非区域提示部分（不受提示干扰）
- **区域提示 $R_j$** 只能看到自身和对应区域的 patch（实现局部信息提取）

$$\mathbf{Mask}[R_j] = \mathbbm{1}(j, b_j, \ldots, b_j + \lfloor N/M \rfloor - 1)$$

这确保了区域提示专注于局部特征提取，同时不破坏原始 patch embedding 的完整性。

### 关键设计 3：层级特征对齐

**动机**：长文本特征空间更复杂，仅对最后一层做对齐不够。中间层特征也应保持视觉-语言一致性。

**Group Token Aggregation (GTA)**：将 $L$ 层 Transformer 的 [CLS] token 分为 $G$ 组，每组用高斯分布加权聚合：

$$\text{GTA}(\mathbf{T}_g) = \sum_{j=1}^S \mathbf{Gaussian}(j; S, 1) \cdot \mathbf{T}_g[j]$$

然后通过线性投影 + LayerNorm 得到 Group Middle Feature (GMF)。

**损失函数**：对每组 GMF 计算视觉-语言 InfoNCE 损失 $L_{m_i}$，仅对齐第 $K$ 组到第 $G$ 组（发现浅层特征差异过大），最终损失为：

$$L = \sum_{i=K}^G \omega_i L_{m_i} + L_{\text{short}} + L_{\text{long}}$$

### 数据合成

- 使用 Llama3-LLaVA-NeXT-8b 合成长文本描述（20 种多样化提示），平均长度 ~120 tokens
- 构建三个规模的数据集：5M、15M、30M
- 过滤低质量描述（重复词、无意义句子、过短结果）

## 实验

### 长文本检索（R@1）

| 方法 | 数据量 | DCI I2T | DCI T2I | IIW I2T | IIW T2I | ShareGPT4V I2T | Urban I2T | 平均 |
|------|-------|---------|---------|---------|---------|----------------|-----------|------|
| CLIP (B/16) | 400M | 37.3 | 34.5 | 75.2 | 76.4 | 78.2 | 68.1 | 62.8 |
| Long-CLIP | 1M | 51.1 | 57.0 | 89.2 | 86.9 | 94.6 | 78.9 | 76.8 |
| LoTLIP | 100M | 62.1 | 61.0 | 93.9 | 92.5 | 96.5 | 77.8 | 81.9 |
| **Fix-CLIP** | **1M** | **59.7** | **63.0** | **93.8** | **95.6** | **95.5** | **80.9** | **82.6** |
| **Fix-CLIP** | **30M** | **70.7** | **70.7** | **97.4** | **97.4** | **98.6** | **90.8** | **89.8** |

关键发现：
- 仅 1M 数据即超越使用 100M 数据的 LoTLIP（B/16 平均 82.6 vs 81.9）
- 30M 数据版本在所有数据集上大幅超越所有 SOTA，平均提升约 8%
- L/14 模型在 30M 数据上达到 91.2% 的平均 R@1

### 短文本检索（R@1）

| 方法 | COCO I2T | COCO T2I | Flickr I2T | Flickr T2I |
|------|----------|----------|------------|------------|
| Long-CLIP (B/16) | 57.6 | 40.4 | 87.9 | 72.3 |
| **Fix-CLIP (1M, B/16)** | **60.9** | **44.8** | **88.8** | **77.4** |

关键发现：
- Fix-CLIP 在提升长文本能力的同时**增强了短文本性能**（COCO T2I +4.4%，Flickr T2I +5.1%）
- 这归功于双分支训练管线中 masked 图像-短文本分支对原始特征空间的保持

### 扩散模型即插即用

Fix-CLIP 的文本编码器可直接替换 CLIP 文本编码器用于扩散模型（如 SD），在长文本输入场景下显著提升生成质量。支持最长 248 token 输入（原 CLIP 仅 77），更好地表达复杂描述。

## 亮点与洞察

1. **增量训练范式**：相比从头训练（LoTLIP 100M），增量训练 1M 即可超越，数据效率极高
2. **双分支设计的精妙之处**：masked 图像与短文本配对，利用了 MAE 的洞察（75% mask 保留语义），同时避免了长短文本特征空间冲突
3. **区域提示 + 单向掩码**：无需额外标注区域描述，通过注意力机制隐式实现区域对齐
4. **层级对齐**：高斯加权聚合中间层特征，从浅到深逐级对齐，应对长文本特征空间的复杂性
5. **数据扩展性极强**：从 5M 到 30M 持续稳定提升，无饱和迹象

## 局限性

1. 增量训练依赖 CLIP 预训练权重，无法从根本上改变 CLIP 的视觉编码能力
2. 区域提示数量 $M$ 和层级分组数 $G$ 等超参需手动调节
3. 248 token 上限虽已覆盖大多数场景，但对于超长文档级描述仍不够
4. 合成长文本可能存在 MLLM 幻觉，已做过滤但无法完全消除

## 相关工作

- **CLIP 长文本改进**：Long-CLIP（位置编码插值）→ LoTLIP/FLAIR（从头训练）→ 本文（增量训练+双分支）
- **区域对齐**：FILIP（patch-text 对应）→ MaskCLIP（掩码增强）→ 本文（区域提示+单向掩码）
- **数据增强**：LaCLIP（LLM 重写）→ VeCLIP/CAPSFUSION（MLLM 丰富描述）→ 本文（多样化提示合成 30M 数据）

## 评分

- 创新性: ⭐⭐⭐⭐ — 多个模块组合创新，但每个模块思路有迹可循
- 技术深度: ⭐⭐⭐⭐ — 双分支+区域提示+层级对齐设计完整，理论分析充分
- 实验充分度: ⭐⭐⭐⭐⭐ — 长/短文本检索+分类+扩散应用+多规模数据+完善消融
- 实用价值: ⭐⭐⭐⭐⭐ — 即插即用文本编码器对扩散模型有直接价值，代码开源

<!-- RELATED:START -->

## 相关论文

- [Dual Diffusion for Unified Image Generation and Understanding](../../CVPR2025/image_generation/dual_diffusion_for_unified_image_generation_and_understanding.md)
- [Contrastive Flow Matching (ΔFM)](contrastive_flow_matching.md)
- [CURE: Cultural Gaps in the Long Tail of Text-to-Image Systems](cure_cultural_gaps_in_the_long_tail_of_text-to-image_systems.md)
- [Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data (SPARCL)](../../CVPR2025/image_generation/enhancing_vision-language_compositional_understanding_with_multimodal_synthetic_.md)
- [Generating Multi-Image Synthetic Data for Text-to-Image Customization](generating_multi-image_synthetic_data_for_text-to-image_customization.md)

<!-- RELATED:END -->
