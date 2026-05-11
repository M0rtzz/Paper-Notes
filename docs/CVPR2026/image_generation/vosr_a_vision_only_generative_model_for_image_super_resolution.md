---
title: >-
  [论文解读] VOSR: A Vision-Only Generative Model for Image Super-Resolution
description: >-
  [CVPR 2026][图像生成][超分辨率] 提出 VOSR，首个证明纯视觉训练的生成式超分模型可以媲美甚至超越基于 T2I 预训练方法的工作，通过视觉语义条件和面向恢复的引导策略实现高质量 SR，训练成本仅为 T2I 方法的 1/10。
tags:
  - CVPR 2026
  - 图像生成
  - 超分辨率
  - vision-only
  - 扩散模型
  - classifier-free guidance
  - one-step distillation
---

# VOSR: A Vision-Only Generative Model for Image Super-Resolution

**会议**: CVPR 2026  
**arXiv**: [2604.03225](https://arxiv.org/abs/2604.03225)  
**代码**: [https://github.com/cswry/VOSR](https://github.com/cswry/VOSR)  
**领域**: 图像生成  
**关键词**: super-resolution, vision-only, diffusion model, classifier-free guidance, one-step distillation

## 一句话总结

提出 VOSR，首个证明纯视觉训练的生成式超分模型可以媲美甚至超越基于 T2I 预训练方法的工作，通过视觉语义条件和面向恢复的引导策略实现高质量 SR，训练成本仅为 T2I 方法的 1/10。

## 研究背景与动机

当前生成式图像超分领域被基于 Text-to-Image (T2I) 扩散模型（如 Stable Diffusion）的方法主导，它们通过适配预训练 T2I 生成器来进行超分恢复。然而作者指出这种范式存在根本性矛盾：SR 是一个以低分辨率输入为条件的图像恢复任务，而 T2I 方法从通用生成器出发，通过文本或文本对齐表示引入语义，增加了细节幻觉（hallucination）的风险。

核心问题：一个纯视觉训练的生成式模型，不依赖多模态预训练，能否匹敌 T2I-based SR 方法？

作者通过 VOSR 给出了肯定回答。VOSR 需要的训练成本仅约为代表性 T2I-based SR 方法的 1/10，却在感知质量和保真度上达到有竞争力甚至更优的结果。

## 方法详解

### 整体框架

VOSR 基于 LightningDiT 骨干，在潜空间中进行 flow matching 训练。给定 LR 图像，构建两个互补条件：结构条件（VAE 编码的 LR 潜在表示）和视觉语义条件（DINO 编码器提取的高级特征），注入 DiT 进行 HR 预测。

### 关键设计

1. **视觉语义条件**：不同于先前仅靠 LR 结构条件的视觉-only SR 方法，VOSR 引入 DINO 预训练视觉编码器提取语义特征。结构条件通过空间对齐的潜在条件注入保持保真度，语义条件通过交叉注意力提供高级上下文，两者互补——结构保真度，语义解歧义。关键在于语义完全在视觉域内，避免文本对齐条件的空间粗粒度问题。

2. **面向恢复的引导策略 (Restoration-Oriented Guidance)**：重新审视 CFG 在视觉-only 恢复中的应用，发现标准无条件分支不适合从头训练的 SR 模型。提出用部分条件分支替代无条件分支——保留弱化的 LR 结构线索但去除语义条件。这让两个分支都锚定在输入上，引导方向从弱锚定向强锚定移动。有趣的行为反转：增大引导尺度→更高保真度（靠近全条件分支），减小引导尺度→更强生成能力（靠近部分条件分支）。

3. **单步蒸馏**：将多步 VOSR 教师蒸馏为单步学生模型，保持相同的条件接口和恢复导向引导，仅改变采样效率。采用递归一致性蒸馏变体，实现感知质量与结构保真度的最佳平衡。

### 损失函数 / 训练策略

多步模型使用标准 velocity 参数化的扩散训练目标。训练时随机切换全条件模式和部分条件模式。约 1 亿网页图像训练，使用 Real-ESRGAN 退化管线合成 LR-HR 对。提供 0.5B 和 1.4B 两个尺寸变体。

## 实验关键数据

### 主实验

| 数据集 | 设置 | 本文 (VOSR-1.4B-ms) | T2I SOTA (SeeSR) | 说明 |
|--------|------|-------------------|-----------------|------|
| RealSR | 多步 | 感知指标竞争力强 | 对比方法之一 | VOSR 在保真度指标上更优 |
| ScreenSR | 多步 | 多项指标最优 | — | 新构建的真实世界测试集 |
| LSDIR | 单步 | 超越 OSEDiff 等 | — | 单步推理效率与 T2I 单步方法相当 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无视觉语义条件 | 感知质量下降 | 语义条件对解决歧义至关重要 |
| 标准 CFG（全无条件） | 效果差 | 无条件分支太难学，引导方向不适合恢复 |
| 面向恢复的引导 | 最优 | 部分条件分支保持输入锚定 |

### 关键发现

- 视觉-only 框架首次在感知质量上可与 T2I-based SR 竞争，同时保真度更优、幻觉更少
- 多步模型效率远高于现有 T2I-based SR 系统，单步模型与最新单步 T2I 系统效率相当
- 训练成本仅约 T2I 代表方法的 1/10

## 亮点与洞察

- 从根本上质疑 T2I 预训练对于 SR 的必要性，给出了有力的反面论证
- 面向恢复的引导策略设计巧妙，引导尺度语义反转现象（大尺度→保真，小尺度→生成）非常有趣
- 首次构建 ScreenSR 真实世界配对测试集，为 SR 评估提供更高质量参考
- 证明强语义可以完全在视觉域内获取，无需文本中介

## 局限与展望

- 仍需大规模数据和算力训练（虽然比 T2I 方法少得多）
- 在某些极端退化下可能仍不如 T2I 方法的强先验
- 视觉编码器（DINO）本身的预训练也需要大量数据

## 相关工作与启发

- 与 ResShift、SinSR 等先前视觉-only SR 方法相比，VOSR 显著提升了感知质量
- 面向恢复的引导策略可推广到其他图像恢复任务

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 首次证明 vision-only 可媲美 T2I-based SR
- 技术深度：⭐⭐⭐⭐⭐ — 引导策略设计精巧，理论分析深入
- 实验充分度：⭐⭐⭐⭐⭐ — 多尺度、多步/单步、新测试集，非常全面
- 实用价值：⭐⭐⭐⭐⭐ — 低训练成本高效率，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution](oars_process-aware_online_alignment_for_generative_real-world_image_super-resolu.md)
- [\[NeurIPS 2025\] Image Super-Resolution with Guarantees via Conformalized Generative Models](../../NeurIPS2025/image_generation/image_super-resolution_with_guarantees_via_conformalized_generative_models.md)
- [\[CVPR 2026\] AlignVAR: Towards Globally Consistent Visual Autoregression for Image Super-Resolution](alignvar_towards_globally_consistent_visual_autoregression_for_image_super-resol.md)
- [\[AAAI 2026\] GEWDiff: Geometric Enhanced Wavelet-based Diffusion Model for Hyperspectral Image Super-resolution](../../AAAI2026/image_generation/gewdiff_geometric_enhanced_wavelet-based_diffusion_model_for_hyperspectral_image.md)
- [\[CVPR 2025\] Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model](../../CVPR2025/image_generation/uncertainty-guided_perturbation_for_image_super-resolution_diffusion_model.md)

</div>

<!-- RELATED:END -->
