---
title: >-
  [论文解读] Everything to the Synthetic: Diffusion-driven Test-time Adaptation via Synthetic-Domain Alignment
description: >-
  [CVPR 2025][图像生成][测试时自适应] 本文揭示了扩散驱动TTA方法中源域与合成域之间存在隐性不对齐问题，提出Synthetic-Domain Alignment (SDA)框架，通过Mix of Diffusion (MoD)技术将源模型和目标数据同时对齐到同一个合成域，在分类、分割和多模态大语言模型上均取得了一致的性能提升。
tags:
  - CVPR 2025
  - 图像生成
  - 测试时自适应
  - 扩散模型
  - 域对齐
  - 合成域
  - 数据适配
---

# Everything to the Synthetic: Diffusion-driven Test-time Adaptation via Synthetic-Domain Alignment

**会议**: CVPR 2025  
**arXiv**: [2406.04295](https://arxiv.org/abs/2406.04295)  
**代码**: https://github.com/SHI-Labs/Diffusion-Driven-Test-Time-Adaptation-via-Synthetic-Domain-Alignment  
**领域**: 扩散模型  
**关键词**: 测试时自适应, 扩散模型, 域对齐, 合成域, 数据适配

## 一句话总结

本文揭示了扩散驱动TTA方法中源域与合成域之间存在隐性不对齐问题，提出Synthetic-Domain Alignment (SDA)框架，通过Mix of Diffusion (MoD)技术将源模型和目标数据同时对齐到同一个合成域，在分类、分割和多模态大语言模型上均取得了一致的性能提升。

## 研究背景与动机

**领域现状**：测试时自适应（TTA）是一个新兴研究方向，旨在改善源域预训练模型在未见过的目标域上的表现。传统TTA方法通过不断更新模型权重来适配目标数据流，但这种方式对目标数据的数量和顺序敏感。近年来，扩散模型驱动的TTA方法（如DiffPure、DDA、GDA）转而适配输入数据而非模型权重，通过无条件扩散模型将目标域数据映射到合成域。

**现有痛点**：尽管扩散驱动TTA方法在视觉上生成的合成数据与源数据难以区分，但作者发现对于深度网络而言，合成数据实际上与源数据存在显著的域不对齐。实验表明，即使在ImageNet源数据上应用DDA进行扩散适配（不涉及域迁移），Swin-B和ConvNeXt-B的准确率分别下降了超过21.8%和18.8%。

**核心矛盾**：现有扩散驱动TTA方法的理论假设是将目标数据映射回源域，但实际上映射的终点是扩散模型的合成域，而非真正的源域。源域与合成域之间存在隐性gap——这种gap肉眼不可见，但深度网络对此非常敏感。

**本文目标** (1) 如何量化和理解源域-合成域的不对齐问题？(2) 如何在不访问源数据的前提下，将模型也适配到合成域？(3) 如何处理条件扩散模型和无条件扩散模型之间的合成域差异？

**切入角度**：既然扩散适配后的目标数据不可避免地落在合成域中，那么不妨反过来将源模型也对齐到相同的合成域，把跨域TTA问题转化为域内预测问题。

**核心 idea**：不再试图将数据拉回源域，而是将模型和数据同时对齐到扩散模型的合成域，消除域间gap。

## 方法详解

### 整体框架

SDA框架包含三个阶段：(1) 源域模型预训练阶段——在源数据上正常训练源模型；(2) 源→合成的模型适配阶段——通过MoD技术生成合成数据集，将源模型微调到合成域模型；(3) 目标→合成的数据适配阶段——使用无条件扩散模型将目标数据映射到合成域。最终，合成域的数据输入合成域模型进行推理，并与源模型对原始目标数据的预测进行集成。

### 关键设计

1. **源-合成域不对齐的发现与量化**:

    - 功能：揭示扩散驱动TTA中被忽视的域不对齐问题
    - 核心思路：在ImageNet验证集上，先对源域数据进行扩散加噪-去噪操作（即只涉及source→synthetic，不涉及target→source），然后用源模型在去噪后的数据上测试。结果显示随着时间步$t$的增大，准确率单调下降。在$t \geq 500$的TTA合理范围内，ConvNeXt-B准确率从83.4%降至41.5%-65.1%，证实了合成域与源域的巨大差距。
    - 设计动机：虽然合成数据和源数据在视觉上几乎无差别，但深度网络对这种隐性域差异极其敏感，这正是之前方法性能受限的根本原因。

2. **Mix of Diffusion (MoD) 合成数据生成**:

    - 功能：生成与无条件扩散模型合成域对齐的带标签合成数据集
    - 核心思路：包含两步操作。第一步，使用条件扩散模型（如DiT）按类别标签生成合成数据 $x_{0,c}^{syn}$，得到带标签的合成数据集。第二步，对生成的条件合成数据进行加噪（到时间步$t^*$）再用无条件扩散模型去噪，将数据从条件合成域 $p_{0,c}^{syn}$ 对齐到无条件合成域 $p_{0,u}^{syn}$。这一步利用了KL散度随加噪步数单调递减的性质，在高噪声水平下两个合成域的分布趋同。
    - 设计动机：条件扩散模型可以生成带标签数据，但其合成域与无条件扩散模型的合成域存在差异（架构和训练方式不同）。通过第二步的加噪-去噪操作，消除两个合成域之间的gap，确保微调数据与测试时数据适配的域完全一致。

3. **合成域模型微调与集成推理**:

    - 功能：将源模型对齐到合成域并进行鲁棒推理
    - 核心思路：用MoD生成的50K合成数据对源模型进行15个epoch的微调，得到合成域模型$f'$。推理时，将源模型$f$对原始目标数据的预测$q(y|x_0^{trg})$与合成域模型$f'$对适配后数据的预测$q'(y|x_{0,u}^{syn})$进行概率集成：$\hat{y} = \arg\max_y (q(y|x_0^{trg}) + q'(y|x_{0,u}^{syn}))$。
    - 设计动机：扩散适配偶尔会产生识别度低于原始数据的样本，集成策略可以取两者之长。合成数据只需生成一次即可适配不同的源模型，边际成本很低。

### 损失函数 / 训练策略

模型适配阶段使用标准分类交叉熵损失进行微调。合成数据生成一次后可复用于不同源模型。分割任务和MLLM上也采用类似策略：用条件扩散模型生成带分割标注或VQA标注的合成数据，经MoD处理后微调。

## 实验关键数据

### 主实验

| 模型 | Source | MEMO | DiffPure | GDA | DDA | SDA (Ours) |
|------|--------|------|----------|-----|-----|------------|
| ResNet-50 | 18.7 | 24.7 | 16.8 | 31.8 | 29.7 | **32.5** (+2.8) |
| Swin-T | 33.5 | 29.5 | 24.8 | 42.2 | 40.0 | **42.5** (+2.5) |
| ConvNeXt-T | 39.3 | 37.8 | 28.8 | 44.8 | 44.2 | **47.0** (+2.8) |
| Swin-B | 40.5 | 37.0 | 28.9 | - | 44.5 | **47.4** (+2.9) |
| ConvNeXt-B | 45.6 | 45.8 | 32.7 | - | 49.4 | **51.9** (+2.5) |

ImageNet-C severity=5，15个corruption平均准确率。SDA在所有模型上一致超越DDA 2.5-2.9%。

### 消融实验

| 配置 | Swin-B (t=700) | ConvNeXt-B (t=700) | 说明 |
|------|---------------|-------------------|------|
| Source-Synthetic (Misaligned) | 55.7 | 60.3 | 源模型直接测合成数据 |
| Synthetic-Synthetic (Aligned, SDA) | 65.0 | 67.4 | 对齐后的模型测合成数据 |
| Δ | **+9.3** | **+7.1** | 域对齐带来的提升 |
| 仅条件合成域微调 $f_c'$ | 低于 $f_u'$ | 低于 $f_u'$ | 证实条件/无条件合成域存在差异 |

### 关键发现

- SDA在ImageNet-C的15个corruption中的14个上超越DDA，唯一例外是contrast corruption。
- 域对齐的提升随时间步$t$增大而增大（$t=500$时+6.0%，$t=1000$时+9.9%），说明合成域偏移越大，对齐收益越高。
- SDA框架也适用于分割任务（DeepLabv3在PASCAL VOC-C上提升1.2% mIOU）和MLLM（LLaVA在corrupted VQA上有改善），展现了通用性。
- 合成数据只需生成一次，不同源模型可以复用。

## 亮点与洞察

- **隐性域不对齐的发现非常精准**：虽然合成数据视觉上与源数据无异，但深度网络的特征空间对此极度敏感。这个insight揭示了扩散驱动TTA方法的根本瓶颈，对整个领域有启发意义。
- **MoD是一个优雅的对齐机制**：通过"加噪到高噪声水平→用目标扩散模型去噪"这一简单操作，将任意分布对齐到目标合成域，避免了复杂的域适配训练。
- **框架的正交性很有价值**：SDA关注"模型→合成域"的对齐，与DDA/GDA关注"数据→合成域"的适配互不冲突，未来可以直接与更好的数据适配方法组合获得更大提升。

## 局限与展望

- SDA需要额外的50K合成数据生成和微调步骤，增加了前置开销（尽管只需一次）。
- 扩散适配本身的计算开销依然很高（每张图需要完整的加噪-去噪过程），限制了实时应用场景。
- 对于与源域差距极大的目标域（如跨模态），扩散模型本身的合成域可能也无法有效覆盖目标分布。
- 集成策略较为简单（概率加和），可以探索自适应加权或基于置信度的融合方案。

## 相关工作与启发

- **vs DDA**: DDA只做Target→Synthetic的数据适配，忽略了Source与Synthetic之间的gap。SDA通过额外的模型适配补齐了这块缺失，是DDA的自然延伸。
- **vs GDA**: GDA使用更好的结构引导进行数据适配，SDA则从模型适配的角度切入。两者正交互补，GDA的数据适配+SDA的模型对齐可以叠加。
- **vs 传统TTA (MEMO等)**: 传统方法需要不断更新模型权重，受目标数据流的顺序和批量影响大。SDA的模型适配是离线的一次性操作，推理时不需要任何在线更新。

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心insight（合成域≠源域）简单但深刻，MoD设计巧妙但技术手段相对直接
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖分类/分割/MLLM三类任务，多种模型架构，消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，问题→发现→方案的叙事链非常流畅
- 价值: ⭐⭐⭐⭐ 对扩散驱动TTA领域有重要启发，框架通用性强

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation](../../NeurIPS2025/image_generation/diffusion-driven_progressive_target_manipulation_for_source-free_domain_adaptati.md)
- [\[CVPR 2025\] Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?](training_data_provenance_verification_did_your_model_use_synthetic_data_from_my_.md)
- [\[CVPR 2025\] DoraCycle: Domain-Oriented Adaptation of Unified Generative Model in Multimodal Cycles](doracycle_domain-oriented_adaptation_of_unified_generative_model_in_multimodal_c.md)
- [\[CVPR 2025\] Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data (SPARCL)](enhancing_vision-language_compositional_understanding_with_multimodal_synthetic_.md)
- [\[CVPR 2025\] Color Alignment in Diffusion](color_alignment_in_diffusion.md)

<!-- RELATED:END -->
