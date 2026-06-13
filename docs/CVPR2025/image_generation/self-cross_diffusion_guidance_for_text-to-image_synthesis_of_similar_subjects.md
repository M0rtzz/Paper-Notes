---
title: >-
  [论文解读] Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects
description: >-
  [CVPR 2025][图像生成][扩散模型引导] 提出 Self-Cross Diffusion Guidance，通过惩罚一个主体的聚合自注意力图与另一个主体的交叉注意力图之间的重叠，有效解决扩散模型生成相似主体时的主体混合问题，是首个同时利用自注意力和交叉注意力交互关系的免训练方法。 - 扩散模型在文本到图像生成中已取…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "扩散模型引导"
  - "主体混合"
  - "自注意力-交叉注意力"
  - "免训练推理"
  - "相似主体生成"
---

# Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects

**会议**: CVPR 2025  
**arXiv**: [2411.18936](https://arxiv.org/abs/2411.18936)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 扩散模型引导, 主体混合, 自注意力-交叉注意力, 免训练推理, 相似主体生成

## 一句话总结

提出 Self-Cross Diffusion Guidance，通过惩罚一个主体的聚合自注意力图与另一个主体的交叉注意力图之间的重叠，有效解决扩散模型生成相似主体时的主体混合问题，是首个同时利用自注意力和交叉注意力交互关系的免训练方法。

## 研究背景与动机

- 扩散模型在文本到图像生成中已取得显著进展，但主体混合（subject mixing）仍是未解决的关键问题
- 生成多个外观相似的主体（如豹和老虎）时尤其严重：不同主体的特征会相互渗透
- 现有方法（Attend&Excite、INITNO、CONFORM）分别基于交叉注意力或自注意力单独进行引导，但忽略了两者之间的交互关系
- 仅关注最具区分性的 patch（如鸟的喙）是不够的——其他前景 patch 也可能导致主体混合
- 现有评测基准缺乏相似主体场景的挑战性提示，CLIP score 与人类判断相关性差

## 方法详解

### 整体框架

Self-Cross Guidance 是一种免训练的推理时优化方法。在扩散反向过程的前半段时间步中，从每个主体的交叉注意力图中通过 Otsu 阈值化选择对应 patch，聚合这些 patch 的自注意力图，然后惩罚聚合自注意力与其他主体交叉注意力之间的重叠。结合初始噪声优化和迭代潜变量精修实现。

### 关键设计

**设计一：自注意力图聚合**

- **功能**：获取覆盖整个主体区域的自注意力表示
- **核心思路**：对主体 $i$ 的交叉注意力图 $A_i^c$ 应用 Otsu 阈值化选择高响应 patch。将选中 patch 的自注意力图按交叉注意力值加权求和：$A_i^s = \frac{\sum_{x_m,y_n}(A_i^c[x_m,y_n] \times A_{x_m,y_n}^s)}{\sum_{x_m,y_n} A_i^c[x_m,y_n]}$
- **设计动机**：不同 patch 的自注意力图差异很大，仅用最具区分性的单个 patch 无法覆盖主体完整区域。通过聚合多个 patch 的自注意力，可以获得更全面的主体关注区域表示

**设计二：Self-Cross 引导损失**

- **功能**：惩罚一个主体的自注意力区域与另一个主体的交叉注意力区域的重叠，消除主体混合
- **核心思路**：对于主体对 $(i,j)$，计算重叠 $g(i,j) = \sum_{x,y} \min(A_i^s[x,y], A_j^c[x,y]) + \sum_{x,y} \min(A_i^c[x,y], A_j^s[x,y])$。$N$ 个相似主体时取所有 $C_N^2$ 对的平均。总损失 $\mathcal{L}_{total} = S_{self-cross} + \lambda \cdot S_{cross-attn}$
- **设计动机**：主体混合的本质是一个主体的自注意力侵入了另一个主体的区域。聚合自注意力图与交叉注意力图的重叠比单独使用任一种注意力更能精确捕捉这种侵入

**设计三：SSD 基准与 GPT-4o 评估**

- **功能**：提供挑战性的相似主体生成评测基准
- **核心思路**：发布 Similar-Subject Dataset（SSD），包含两个或三个相似主体的文本提示。利用 GPT-4o 通过视觉问答自动评估生成图像中主体的存在性、可识别性和属性绑定
- **设计动机**：CLIP score 无法有效区分主体混合问题，GPT-4o 评估与人类判断具有更高一致性

### 损失函数

$$\mathcal{L}_{total} = S_{self-cross} + \lambda \cdot S_{cross-attn}$$

其中 $S_{cross-attn}$ 沿用 Attend&Excite 的交叉注意力响应得分，$\lambda$ 为平衡系数。仅在前半段去噪步骤和中间层应用。

## 实验关键数据

### SSD 基准定量结果

| 方法 | 存在性 ↑ | 可识别性 ↑ | 属性绑定 ↑ | FID ↓ |
|------|---------|-----------|-----------|------|
| Stable Diffusion | 基线 | 基线 | 基线 | 基线 |
| Attend&Excite | 改善 | 有限改善 | 有限改善 | — |
| INITNO | 改善 | 部分改善 | 部分改善 | — |
| CONFORM | 改善 | 部分改善 | 部分改善 | — |
| **Self-Cross (Ours)** | **最佳** | **最佳** | **最佳** | **保持** |

### 消融实验

| 配置 | 效果 |
|------|------|
| 仅 cross-attn 引导 | 无法消除主体混合 |
| 仅 self-attn 引导 | 部分改善 |
| 单 patch self-attn + cross-attn | 改善有限 |
| **聚合 self-attn + cross-attn** | **显著消除主体混合** |

### 关键发现

- Self-Cross 引导在消除主体混合方面大幅超越 INITNO 等仅用单个注意力图的方法
- 聚合多 patch 自注意力比单 patch 效果显著更好
- 方法兼容 UNet-based（SD 1.x/2.x）和 Transformer-based（SD3）扩散模型
- 作为副效应，主体遗漏问题也得到改善
- 图像整体质量（FID）不受明显影响

## 亮点与洞察

1. **首次探索自注意力与交叉注意力的交互关系**：提供了关于主体混合成因的新理解——自注意力侵入他区导致特征复制
2. **多 patch 聚合的必要性**：证明了仅关注最具区分性的 patch 不足以消除主体混合
3. **GPT-4o 评估方案**：为扩散模型评估提供了更可靠的自动化评测手段

## 局限与展望

- 初始噪声优化和迭代精修增加了推理时间
- 仅关注相似主体间的混合，对非相似主体的属性绑定改善有限
- 需要用户指定哪些主体是"相似的"，缺乏自动检测机制
- 可探索将 Self-Cross 引导集成到训练过程中扩展到更多场景

## 相关工作与启发

- **Attend&Excite** [Chefer et al.] 通过最大交叉注意力防止主体遗漏
- **INITNO** [Guo et al.] 结合自注意力冲突得分优化初始噪声
- **CONFORM** [Meral et al.] 使用对比损失进行主体分离
- 本文首次揭示了自注意力与交叉注意力交互在主体混合中的关键角色

## 评分

⭐⭐⭐⭐ — 对主体混合成因的分析有深度（自注意力侵入），Self-Cross 引导损失设计直观有效。多 patch 聚合策略比单 patch 方案的显著优势令人信服。SSD 基准和 GPT-4o 评估为社区提供了有价值的工具。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Towards Transformer-Based Aligned Generation with Self-Coherence Guidance](towards_transformer-based_aligned_generation_with_self-coherence_guidance.md)
- [\[CVPR 2025\] Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis](noise_diffusion_for_enhancing_semantic_faithfulness_in_text-to-image_synthesis.md)
- [\[CVPR 2025\] Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis](exploring_sparse_moe_in_gans_for_text-conditioned_image_synthesis.md)
- [\[CVPR 2026\] CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration](../../CVPR2026/image_generation/ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)
- [\[CVPR 2025\] ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts](shapewords_guiding_text-to-image_synthesis_with_3d_shape-aware_prompts.md)

</div>

<!-- RELATED:END -->
