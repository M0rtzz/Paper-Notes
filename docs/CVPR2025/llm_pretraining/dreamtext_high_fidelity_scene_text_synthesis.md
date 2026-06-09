---
title: >-
  [论文解读] DreamText: High Fidelity Scene Text Synthesis
description: >-
  [CVPR 2025][预训练][场景文字合成] DreamText重构扩散模型训练流程，引入字符级别的均衡监督(balanced supervision)和启发式交替优化策略来校正字符注意力，结合文本编码器与生成器的联合训练学习多样化字体风格…
tags:
  - "CVPR 2025"
  - "预训练"
  - "场景文字合成"
  - "扩散模型"
  - "字符注意力"
  - "均衡监督"
  - "交替优化"
---

# DreamText: High Fidelity Scene Text Synthesis

**会议**: CVPR 2025  
**arXiv**: [2405.14701](https://arxiv.org/abs/2405.14701)  
**代码**: [https://github.com/OpenBMB/DreamText](https://github.com/OpenBMB/DreamText)  
**领域**: 扩散模型 / 文字生成  
**关键词**: 场景文字合成, 扩散模型, 字符注意力, 均衡监督, 交替优化

## 一句话总结

DreamText重构扩散模型训练流程，引入字符级别的均衡监督(balanced supervision)和启发式交替优化策略来校正字符注意力，结合文本编码器与生成器的联合训练学习多样化字体风格，在场景文字合成任务上大幅超越SOTA方法（SeqAcc从UDiffText的0.763提升至0.940）。

## 研究背景与动机

**领域现状**：场景文字合成(scene text synthesis)旨在将指定文本渲染到任意图像中。现有方法分为GAN-based（如MOSTEL）和diffusion-based（如TextDiffuser、AnyText、UDiffText）两类。Diffusion-based方法凭借更强的生成能力逐渐成为主流。

**现有痛点**：当前扩散方法存在三个核心问题——(1) **字符扭曲(distortion)**：生成的字符形状不正确；(2) **字符重复(repetition)**：同一字符被多次生成；(3) **字符缺失(absence)**：部分字符未被渲染。这些问题在多字体风格(polystylistic)场景中尤为严重。根本原因是现有方法在训练过程中缺乏有效的**字符级指导(character-level guidance)**——模型不知道每个字符应该出现在哪里，注意力分散到错误区域。

**核心矛盾**：文字合成需要精确控制每个字符的位置和形态，但diffusion模型的端到端训练只有图像级损失，缺乏字符级的空间约束。此外，大多数方法的文本编码器是在单一字体上预训练的，无法适应实际应用中多样化的字体风格。

**本文目标**：(1) 在diffusion训练过程中引入精细的字符级引导，暴露并纠正模型在字符层面的注意力；(2) 联合训练文本编码器和图像生成器，使其能学习训练集中的多样化字体。

**切入角度**：从cross-attention map中可以提取每个字符的"潜在字符mask"(latent character mask)，这个mask反映了模型认为该字符应该出现的位置。通过将这个mask与字符的实际位置进行对齐，可以校正模型的注意力。

**核心 idea**：设计一个均衡监督(balanced supervision)策略，在每个训练步中先从cross-attention map编码出潜在字符mask，然后用这些mask更新字符嵌入表示，使生成器在下一步能纠正字符注意力。文本编码器和生成器在这个交替优化过程中协同学习。

## 方法详解

### 整体框架

DreamText基于stable diffusion的inpainting管线。输入包括原始图像、mask区域和要渲染的目标文本。文本编码器将目标文本编码为字符嵌入序列，输入扩散模型的UNet进行条件去噪。核心创新在于训练过程：每个训练step先从UNet的cross-attention map中提取潜在字符mask，利用均衡监督信号更新字符嵌入，然后让UNet在更新后的嵌入指导下纠正注意力分布。这构成了一个离散变量（字符mask）和连续变量（嵌入权重）的混合优化问题。

### 关键设计

1. **潜在字符Mask提取与均衡监督 (Balanced Supervision)**:

    - 功能：为每个字符提取空间位置先验，并以适度的监督强度引导字符注意力校准
    - 核心思路：在训练过程中，从UNet各层的cross-attention map中提取每个字符token对应的注意力图，聚合后得到"潜在字符mask"——表示模型认为该字符应出现的位置。均衡监督将这些mask与真实的字符分割mask进行比较，但不使用过强的监督（如纯交叉熵），而是设计了一种介于无监督和强监督之间的"均衡"策略。具体来说，通过给予模型一定的自由度来估计最优字符位置，同时又通过真实mask提供位置修正信号
    - 设计动机：无监督(FID 62.36, SeqAcc 0.212)完全失败，因为模型没有字符位置的先验知识；强监督(FID 14.92, SeqAcc 0.862)过度约束了字符位置，限制了模型适应复杂场景（如弯曲文本、非标准排版）的灵活性；均衡监督(FID 12.13, SeqAcc 0.940)取得最优效果，在引导和灵活性之间找到了平衡

2. **启发式交替优化 (Heuristic Alternate Optimization)**:

    - 功能：求解涉及离散变量（字符mask）和连续变量（字符嵌入、生成器参数）的混合优化问题
    - 核心思路：每个训练step分为两个交替执行的子步骤。**E-step**：固定生成器参数，从当前的cross-attention map中"估计"(Estimate)每个字符的潜在mask，即从注意力中编码出position信息。**M-step**：固定estimated mask，利用均衡监督信号更新字符嵌入表示，使嵌入中融入了字符位置信息，然后生成器在更新后的嵌入指导下进行去噪，从而在下一步修正注意力分布。这种交替优化使得"字符位置估计"和"注意力修正"形成正反馈循环
    - 设计动机：文字合成本质上是一个混合优化问题——字符mask是离散的空间分配，嵌入权重是连续的。EM式的交替优化是解决此类问题的经典策略

3. **文本编码器与生成器联合训练 (Joint Training)**:

    - 功能：使文本编码器能学习训练集中的多样化字体风格
    - 核心思路：与现有方法冻结预训练文本编码器不同，DreamText将文本编码器纳入训练循环。在交替优化的M-step中，字符嵌入的更新不仅修正了位置信息，还让编码器学习不同字体的视觉特征。这种联合训练自然地整合在交替优化框架中——编码器学习字体嵌入和位置估计,生成器学习基于这些嵌入来正确渲染字符
    - 设计动机：预训练的文本编码器通常在单一字体（如Arial）上训练，面对手写体、艺术字等多样化风格时表征能力不足。联合训练让编码器"见过"训练集中的各种字体

### 损失函数 / 训练策略

使用标准的diffusion去噪损失（MSE between predicted noise and actual noise）加上均衡监督损失。训练数据使用LAION-OCR子集，提供了多样化的场景文字图片和对应的字符级分割mask（用于均衡监督）。

## 实验关键数据

### 主实验 (LAION-OCR测试集)

| 方法 | SeqAcc (Recon)↑ | SeqAcc (Editing)↑ | FID↓ | 类型 |
|------|-----------------|-------------------|------|------|
| MOSTEL | 低 | 低 | 高 | GAN |
| SD-inpainting v2.0 | 低 | 低 | 高 | Diffusion |
| DiffSTE | 中 | 中 | 中 | Diffusion |
| TextDiffuser | 中 | 中 | 中 | Diffusion |
| AnyText | 中 | 中 | 中 | Diffusion |
| UDiffText | 0.763 | — | ~15 | Diffusion |
| **DreamText (Ours)** | **0.940** | **0.887** | **12.13** | Diffusion |

### 均衡监督消融

| 监督策略 | SeqAcc (Recon) | SeqAcc (Editing) | FID | mIoU |
|---------|---------------|------------------|-----|------|
| 无监督 | 0.212 | 0.157 | 62.36 | 0.203 |
| 强监督 (Cross-Entropy) | 0.862 | 0.813 | 14.92 | 0.617 |
| **均衡监督 (Ours)** | **0.940** | **0.887** | **12.13** | **0.722** |

### 关键发现

- **SeqAcc提升巨大**：DreamText的序列准确率(0.940)比最好的baseline UDiffText(0.763)高出23.2%，说明字符级引导极为有效
- **均衡监督的必要性**：无监督完全失败(SeqAcc 0.212)，说明diffusion模型在没有字符级引导时根本无法可靠地生成正确文字。强监督(0.862)已经很好但仍有gap——过度约束限制了灵活性
- **mIoU指标验证了注意力校准效果**：均衡监督下潜在字符mask与真实字符位置的mIoU达0.722，远高于无监督(0.203)和强监督(0.617)。字符位置预测越准确，生成质量越好
- **联合训练对多字体场景至关重要**：冻结编码器的模型在polystylistic场景下字符错误率更高
- **Human Study**：与UDiffText的50组对比中，DreamText在多样性和质量方面均获得更多人类偏好

## 亮点与洞察

- **均衡监督是一个精妙的设计**：它既不放任模型自由（导致完全失败），也不过度约束字符位置（损害灵活性），而是找到"引导但不强制"的平衡点。这个思路可以迁移到任何需要细粒度空间控制但又不想丧失生成灵活性的条件生成任务中
- **混合优化的EM框架巧妙**：将文字合成建模为离散（字符mask）+连续（嵌入参数）的混合优化问题，EM式交替优化是优雅的解决方案。这种问题抽象方式可以启发其他需要同时优化结构和参数的生成任务
- **从cross-attention中提取字符mask的做法**：利用diffusion模型已有的attention结构来获取字符位置信息，不需要额外的定位网络，设计非常轻量

## 局限与展望

- **无法同时生成多个区域的文字**：当前版本一次只能在一个mask区域内渲染文字，对多区域文字编辑需要多次推理
- **依赖字符级分割标注**：训练时的均衡监督需要字符级的分割mask，数据获取成本较高
- **长文本生成能力有限**：随着字符数增加，注意力校准的难度上升，生成错误率可能增大
- **隐私与安全风险**：高保真文字合成技术可能被用于伪造签名等恶意用途
- 改进方向：探索多区域同时编辑、减少对字符级标注的依赖（如用弱监督或自监督替代部分均衡监督）

## 相关工作与启发

- **vs UDiffText**: UDiffText也是diffusion-based的场景文字合成，但缺乏字符级引导。DreamText通过均衡监督在SeqAcc上大幅超越(0.940 vs 0.763)
- **vs AnyText**: AnyText支持中英文混合生成且有字形引导，但整体SeqAcc不如DreamText。DreamText的优势在于更精确的字符注意力控制
- **vs TextDiffuser**: TextDiffuser用标准字体渲染的字符mask作为条件输入，是一种"硬"引导。DreamText的注意力校准是"软"引导，更灵活
- **vs MOSTEL (GAN-based)**: GAN方法在文字合成任务上已被diffusion方法全面超越，但MOSTEL的局部编辑思路仍有参考价值

## 评分

- 新颖性: ⭐⭐⭐⭐ 均衡监督和交替优化框架是精巧的设计，对文字合成问题有深入洞察
- 实验充分度: ⭐⭐⭐⭐ 包含定量对比、消融实验和human study，分析全面
- 写作质量: ⭐⭐⭐⭐ 问题建模清晰，混合优化的表述严谨
- 价值: ⭐⭐⭐⭐ 为diffusion-based文字合成树立了新的SOTA和方法论基准

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] The Scene Language: Representing Scenes with Programs, Words, and Embeddings](the_scene_language_representing_scenes_with_programs_words_and_embeddings.md)
- [\[CVPR 2025\] AMO Sampler: Enhancing Text Rendering with Overshooting](amo_sampler_enhancing_text_rendering_with_overshooting.md)
- [\[ACL 2025\] Data-Constrained Synthesis of Training Data for De-Identification](../../ACL2025/llm_pretraining/data-constrained_synthesis_of_training_data_for_de-identification.md)
- [\[CVPR 2025\] ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval](context-cir_learning_from_concepts_in_text_for_composed_image_retrieval.md)
- [\[CVPR 2025\] MR-PLIP: Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](mr_plip_multi_resolution_pathology.md)

</div>

<!-- RELATED:END -->
