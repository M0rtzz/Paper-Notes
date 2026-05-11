---
title: >-
  [论文解读] Intra-finger Variability of Diffusion-based Latent Fingerprint Generation
description: >-
  [CVPR 2026][图像生成][指纹合成] 本文系统评估了扩散模型合成指纹的同指变异性，通过构建包含40种表面和15种处理技术的潜指纹风格库提升生成多样性，并量化了生成过程中引入的局部/全局身份不一致性。
tags:
  - CVPR 2026
  - 图像生成
  - 指纹合成
  - 扩散模型
  - 潜指纹
  - 身份一致性
  - 风格多样性
---

# Intra-finger Variability of Diffusion-based Latent Fingerprint Generation

**会议**: CVPR 2026  
**arXiv**: [2604.10040](https://arxiv.org/abs/2604.10040)  
**代码**: 无  
**领域**: 图像生成/生物特征  
**关键词**: 指纹合成, 扩散模型, 潜指纹, 身份一致性, 风格多样性

## 一句话总结

本文系统评估了扩散模型合成指纹的同指变异性，通过构建包含40种表面和15种处理技术的潜指纹风格库提升生成多样性，并量化了生成过程中引入的局部/全局身份不一致性。

## 研究背景与动机

**领域现状**：GenAI（GAN和DDPM）已能生成高质量合成指纹数据集。指纹合成通常分两阶段：生成唯一身份（指间变异）和生成同一身份的多种变体（指内变异）。对于潜指纹领域，第二阶段更为关键。

**现有痛点**：(1) 现有模型依赖整体或随机风格迁移，无法精确指定法医场景（如"从玻璃瓶上提取的用荧光粉显现的潜指纹"）；(2) 生成过程的随机性可能改变指纹脊线和细节点（minutiae），破坏身份真实性。

**核心矛盾**：多样性和身份保持之间存在张力——增加风格多样性的同时可能引入更多身份不一致。

**本文目标**：(1) 增强潜指纹生成的风格多样性；(2) 严格量化身份保持能力。

**切入角度**：构建包含28,000张真实潜指纹的风格库实现精确风格控制，并设计半自动化框架评估身份一致性。

**核心idea**：通过风格库实现40+种可控潜指纹风格，同时揭示扩散模型在低质量区域的局部不一致和风格-参考不匹配时的全局不一致。

## 方法详解

### 整体框架

基于GenPrint框架的第二阶段扩展：(1) 从7个数据集策划28,000张真实潜指纹构建风格库，覆盖40种表面和15种处理技术；(2) 使用风格嵌入精确控制生成的潜指纹风格；(3) 通过半自动化框架对比参考图像和生成图像的细节点和脊线模式。

### 关键设计

1. **潜指纹风格库**:

    - 功能：实现精确的潜指纹风格控制
    - 核心思路：从7个多样化数据集中策划28,000张真实潜指纹，分类为40种表面（玻璃、金属、纸张等）和15种处理技术（粉末、化学、光学等），共40+种离散风格。每种风格提取嵌入作为GenPrint的条件输入
    - 设计动机：GenPrint原始能力局限于通用或随机潜指纹风格，无法生成特定法医场景的潜指纹

2. **身份一致性半自动化评估框架**:

    - 功能：量化生成过程中引入的细节点和脊线变化
    - 核心思路：使用带手动标注的参考指纹生成合成版本，对齐后比较：(1) 局部不一致——细节点的增加和移除（用自动检测器+人工验证）；(2) 全局不一致——不存在于参考中的整块脊线幻觉模式
    - 设计动机：传统的自动匹配分数不够精细，需要细节点级别的分析来理解身份破坏的具体机制

3. **不一致性根因分析**:

    - 功能：理解身份不一致的产生原因
    - 核心思路：局部不一致主要发生在参考图像质量差的区域——模型在不确定时倾向于"编造"细节。全局不一致发生在参考图像与选择的风格嵌入不匹配时——模型幻觉出不存在的脊线模式
    - 设计动机：知道"为什么"不一致才能指导模型改进

### 损失函数 / 训练策略

基于GenPrint的ID-Net（微调ControlNet），使用风格嵌入和文本提示作为条件。训练策略沿用GenPrint框架。

## 实验关键数据

### 主实验

| 评估维度 | 结果 |
|---------|------|
| 风格覆盖 | 40种表面 × 15种处理技术 |
| 数据规模 | 28,000张真实潜指纹 |
| 身份保持 | 大部分保持，少量局部不一致 |
| 全局幻觉 | 风格不匹配时出现 |

### 消融实验

| 条件 | 局部不一致 | 全局不一致 | 说明 |
|------|-----------|-----------|------|
| 高质量参考 | 少 | 少 | 最佳情况 |
| 低质量参考 | 多 | 少 | 差质量区域催生细节点变化 |
| 风格不匹配 | 少 | 多 | 参考和风格嵌入不一致 |

### 关键发现

- 生成过程在大多数情况下保持了身份，但低质量区域更容易引入局部不一致
- 风格嵌入与参考图像的不匹配是全局幻觉的主要原因
- 这些发现为改进合成指纹生成器提供了明确的改进方向

## 亮点与洞察

- **系统性的身份一致性分析**：首次在细节点级别量化了扩散模型指纹生成的身份保持能力
- **法医场景可控性**：40种表面×15种处理技术的风格库使得潜指纹生成具有实际的法医训练价值
- **根因分析的实用性**：低质量区域和风格不匹配两个根因可直接指导模型改进

## 局限与展望

- 半自动化评估仍需人工参与，无法完全自动化
- 风格库的覆盖范围虽广但可能仍不完整
- 未提出解决不一致性的方法，仅做了分析

## 相关工作与启发

- **vs Wyzykowski et al.**: 仅支持3种粗粒度风格（good/bad/ugly），本文实现40+种精细风格
- **vs Joshi et al.**: 使用神经风格迁移但缺乏风格控制，本文通过风格库实现精确控制

## 评分

- 新颖性: ⭐⭐⭐ 风格库构建有价值但方法创新有限
- 实验充分度: ⭐⭐⭐⭐ 身份一致性分析深入细致
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰
- 价值: ⭐⭐⭐ 对法医和指纹识别社区有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Self-Corrected Image Generation with Explainable Latent Rewards](self-corrected_image_generation_with_explainable_latent_rewards.md)
- [\[CVPR 2025\] LEDiff: Latent Exposure Diffusion for HDR Generation](../../CVPR2025/image_generation/lediff_latent_exposure_diffusion_for_hdr_generation.md)
- [\[CVPR 2026\] SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](slice_semantic_latent_injection_via_compartmentalized_embedding_for_image_waterm.md)
- [\[CVPR 2026\] DA-VAE: Plug-in Latent Compression for Diffusion via Detail Alignment](da-vae_plug-in_latent_compression_for_diffusion_via_detail_alignment.md)
- [\[CVPR 2026\] Learning Latent Transmission and Glare Maps for Lens Veiling Glare Removal](learning_latent_transmission_and_glare_maps_for_lens_veiling_glare_removal.md)

</div>

<!-- RELATED:END -->
