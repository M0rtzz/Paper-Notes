---
title: >-
  [论文解读] Less-to-More Generalization: Unlocking More Controllability by In-Context Generation
description: >-
  [ICCV 2025][图像生成][Subject-driven Generation] 本文提出 UNO，一种基于 DiT 的通用定制化生成模型，通过"模型-数据共进化"范式——利用较弱模型生成的合成数据逐步训练更强模型——结合渐进式跨模态对齐和 Universal RoPE，实现了单主体和多主体驱动图像生成的 SOTA 性能（DreamBench DINO 0.760, CLIP-I 0.835）。
tags:
  - ICCV 2025
  - 图像生成
  - Subject-driven Generation
  - 多主体生成
  - DiT
  - 模型-数据共进化
  - 位置编码
---

# Less-to-More Generalization: Unlocking More Controllability by In-Context Generation

**会议**: ICCV 2025  
**arXiv**: [2504.02160](https://arxiv.org/abs/2504.02160)  
**代码**: https://github.com/bytedance/UNO  
**领域**: 扩散模型 / 图像生成  
**关键词**: Subject-driven Generation, 多主体生成, DiT, 模型-数据共进化, 位置编码

## 一句话总结

本文提出 UNO，一种基于 DiT 的通用定制化生成模型，通过"模型-数据共进化"范式——利用较弱模型生成的合成数据逐步训练更强模型——结合渐进式跨模态对齐和 Universal RoPE，实现了单主体和多主体驱动图像生成的 SOTA 性能（DreamBench DINO 0.760, CLIP-I 0.835）。

## 研究背景与动机

**领域现状**：定制化图像生成（Subject-driven Generation）旨在根据参考图像中的主体和文本描述生成新图像，广泛应用于内容创作和工业设计。现有方法分为两大类：(a) 少数据微调方法（DreamBooth、Textual Inversion、LoRA），每个主体需单独微调，推理开销大；(b) 大规模数据训练方法（IP-Adapter、BLIP-Diffusion），使用额外图像编码器实现零样本生成，但依赖大量配对数据。

**现有痛点**：
   - **数据瓶颈**：获取多视角、多姿态的主体配对图像极其困难。从单主体扩展到多主体数据集更是难上加难
   - **合成数据质量有限**：现有合成数据通常分辨率低（≤512×512）且领域覆盖窄
   - **多主体困境**：大多数方法仅支持单主体生成，面对多主体场景时容易出现属性混淆（attribute confusion）或 copy-paste 现象
   - **主体相似度与文本可控性的 trade-off**：提高主体一致性往往牺牲文本编辑能力

**核心矛盾**：现有方法都是"先有数据再设计模型"，数据瓶颈直接限制了模型能力的天花板。如何突破数据瓶颈实现可扩展的定制化生成？

**本文目标** (a) 如何建立一个可持续演化的合成数据管道，从单主体扩展到多主体？(b) 如何设计一个能从单主体无缝扩展到多主体的通用定制化模型？

**切入角度**：受 LLM 中合成数据驱动自我提升的启发（weak-to-strong generalization），提出"模型-数据共进化"范式——先用弱模型（T2I）生成单主体配对数据训练出 S2I 模型，再用 S2I 模型生成多主体配对数据训练更强变体，实现持续共进化。

**核心 idea**：利用 DiT 内在的 in-context 生成能力合成高一致性配对数据，结合渐进式跨模态对齐和 Universal RoPE，以最小修改将 T2I 模型迭代升级为支持多主体的 S2I 模型。

## 方法详解

### 整体框架

UNO 基于 FLUX.1 dev（MM-DiT 架构），包含两大核心系统：(1) 合成数据管道——从单主体到多主体的渐进式数据生成+多阶段过滤；(2) 模型训练框架——通过渐进式跨模态对齐（Stage I 单主体 → Stage II 多主体）和 UnoPE 位置编码将 T2I 模型迭代训练为 S2I 模型。

### 关键设计

1. **合成数据管线（Synthetic Data Curation Framework）**:

    - 功能：利用 DiT 模型的 in-context 生成能力，系统化地生成高分辨率（1024×1024）、高一致性的主体配对数据
    - 核心思路：
        - **单主体阶段**：构建包含 365 个大类和细粒度子类的分类体系，利用 LLM 生成多样化的主体和场景文本，通过精心设计的文本模板让 T2I 模型生成主体一致的图像对。使用 DINOv2 做初步过滤（去掉一致性低的），再用 VLM 从 appearance、details、attributes 多维度评分：$score = \text{Average}(\text{VLM}(I_{ref}, I_{tgt}, c_y))$
        - **多主体阶段**：用 Stage I 训练好的 S2I 模型 + OVD（开放词汇检测器）从目标图像中检测并裁剪额外主体，输入 S2I 模型生成一致的第二参考图像 $I_{ref}^2$。关键：直接用裁剪图像作为 $I_{ref}^2$ 会导致 copy-paste 问题，必须通过 S2I 模型重新生成
    - 设计动机：解决多主体配对数据的获取瓶颈。实验证明高质量评分的数据可显著提升模型性能（DINO 和 CLIP-I 分数随数据质量评分提高而提升）

2. **渐进式跨模态对齐（Progressive Cross-Modal Alignment）**:

    - 功能：分两阶段逐步让 DiT 学会基于参考图像生成
    - 核心思路：
        - Stage I：单参考图像输入。将参考图像通过 VAE 编码后直接拼接到文本 token 和 noisy latent 上：$z_1 = \text{Concatenate}(c, z_t, \mathcal{E}(I_{ref}^1))$。在 FLUX 的 MM-DiT attention 中与其他 token 共同参与计算
        - Stage II：扩展到多参考图像：$z_2 = \text{Concatenate}(c, z_t, z_{ref}^1, z_{ref}^2, \ldots, z_{ref}^N)$，N=2
    - 设计动机：直接训练多参考图像输入效果差（DINO 从 0.542 降至 0.511），因为无噪声的参考图像 token 会破坏原始收敛分布。先从简单到复杂逐步适应更有效

3. **Universal Rotary Position Embedding (UnoPE)**:

    - 功能：为多参考图像设计合理的位置编码，避免属性混淆
    - 核心思路：FLUX 使用 RoPE，文本 token 位于 (0,0)，noisy image token 位于 (i,j) 其中 $i \in [0,w-1], j \in [0,h-1]$。参考图像从对角线位置开始：$(i', j') = (i + w^{(N-1)}, j + h^{(N-1)})$，确保不同参考图像之间有足够的位置间隔
    - 设计动机：直接复制目标图像的位置索引（无偏移）会导致模型无法区分参考图像和目标图像，DINO 大幅下降（从 0.730 降至 0.470 单主体，从 0.542 降至 0.386 多主体）。UnoPE 通过对角线偏移打破了参考图像间的空间关联，迫使模型从文本而非位置获取布局信息

### 损失函数 / 训练策略

使用标准扩散模型流匹配损失。训练配置：学习率 $10^{-5}$，batch size 16，Stage I 5000 步（230K 单主体数据），Stage II 5000 步（15K 多主体数据），LoRA rank 512，8×A100 GPU。

## 实验关键数据

### 主实验

**单主体生成（DreamBench）**：

| 方法 | DINO ↑ | CLIP-I ↑ | CLIP-T ↑ |
|------|--------|----------|----------|
| DreamBooth (微调) | 0.668 | 0.803 | 0.305 |
| RealCustom++ | 0.702 | 0.794 | 0.318 |
| OmniGen | 0.693 | 0.801 | 0.315 |
| OminiControl | 0.684 | 0.799 | 0.312 |
| FLUX IP-Adapter | 0.582 | 0.820 | 0.288 |
| **UNO (Ours)** | **0.760** | **0.835** | 0.304 |
| Oracle (参考图像间) | 0.774 | 0.885 | - |

**多主体生成**：

| 方法 | DINO ↑ | CLIP-I ↑ | CLIP-T ↑ |
|------|--------|----------|----------|
| MS-Diffusion | 0.525 | 0.726 | 0.319 |
| OmniGen | 0.511 | 0.722 | 0.331 |
| MIP-Adapter | 0.482 | 0.726 | 0.311 |
| **UNO (Ours)** | **0.542** | **0.733** | 0.322 |

### 消融实验

| 配置 | DINO ↑ | CLIP-I ↑ | CLIP-T ↑ |
|------|--------|----------|----------|
| **UNO (Full)** | **0.542** | **0.733** | 0.322 |
| w/o generated $I_{ref}^2$ (用裁剪图替代) | 0.529 | 0.730 | 0.308 |
| w/o cross-modal alignment (直接多主体训练) | 0.511 | 0.721 | 0.322 |
| w/o UnoPE (克隆位置索引) | 0.386 | 0.674 | 0.323 |
| w/o offset (无位置偏移) | 0.470 / 0.386 | 0.722 / 0.674 | 0.308 / 0.323 |
| w/ width-offset only | 0.717 / 0.508 | 0.813 / 0.724 | 0.304 / 0.321 |
| w/ height-offset only | 0.678 / 0.501 | 0.797 / 0.719 | 0.308 / 0.306 |
| **w/ diagonal-offset (UnoPE)** | **0.730 / 0.542** | **0.821 / 0.733** | **0.309 / 0.322** |

### 关键发现

- **UnoPE 是最关键组件**：去掉后 DINO 从 0.542 暴跌至 0.386（-28.8%），说明位置编码设计对多主体生成至关重要。对角线偏移优于仅宽度偏移或仅高度偏移
- **渐进式训练显著有效**：直接训练多主体（不做 progressive alignment）DINO 降 5.7%。Stage II 训练反过来还能提升单主体性能（DINO 0.730→0.760），说明多主体能力和单主体能力正向迁移
- **合成数据质量至关重要**：生成的 $I_{ref}^2$ 比直接裁剪的效果更好（DINO +2.5%），且经 VLM 评分过滤的高质量数据能持续提升模型性能
- **UNO 的 DINO 分数 0.760 已接近 Oracle 的 0.774**，说明主体一致性已接近上界

## 亮点与洞察

- **"模型-数据共进化"范式新颖**：不再是"被动等数据"，而是主动利用模型能力生成训练数据来训练更强的模型。这种 self-improvement 思路借鉴了 LLM 领域的 weak-to-strong generalization，在视觉生成领域是首次系统化实现
- **利用 DiT 的内在 in-context 能力**：发现 DiT 模型天然具有生成主体一致图像的能力（通过精心构造的文本模板），省去了复杂的数据采集流程
- **UnoPE 的对角线偏移设计简洁有效**：通过位置编码的空间隔离来解决多主体属性混淆问题，思路可迁移到其他需要处理多条件输入的 DiT 应用

## 局限与展望

- 当前仅支持 N=2 个参考主体，扩展到更多主体的效果和效率需进一步验证
- 合成数据管道依赖于基础 T2I 模型的 in-context 能力，对于不具备此能力的模型不适用
- CLIP-T 分数并非最高（0.304 vs OmniGen 的 0.315），说明在文本可控性上仍有提升空间
- LoRA rank 512 的参数量较大，微调效率有待优化

## 相关工作与启发

- **vs IP-Adapter/BLIP-Diffusion**: 使用额外图像编码器注入参考信息，UNO 则直接利用 DiT 的 VAE + attention 机制，无需额外编码器
- **vs OminiControl/IC-LoRA**: 也利用了 DiT 的内在参考图像能力，但仅限于单主体且分辨率低（512×512），UNO 通过完整的数据管道和位置编码设计扩展到高分辨率多主体
- **vs DreamBooth**: 需要 per-subject 微调，UNO 实现了零样本推理且 DINO 更高

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 模型-数据共进化范式和 UnoPE 都是新贡献
- 实验充分度: ⭐⭐⭐⭐ 单/多主体评测全面，消融详细，但缺少更多主体(>2)的评估
- 写作质量: ⭐⭐⭐⭐ 方法阐述清晰，但数据管道部分稍显冗长
- 价值: ⭐⭐⭐⭐⭐ 开源代码，实用性强，ByteDance 的商业级方案

<!-- RELATED:START -->

## 相关论文

- [Less is More: Improving Motion Diffusion Models with Sparse Keyframes](less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)
- [Erasing More Than Intended? How Concept Erasure Degrades the Generation of Non-Target Concepts](erasing_more_than_intended_how_concept_erasure_degrades_the_generation_of_non-ta.md)
- [VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning](visualcloze_a_universal_image_generation_framework_via_visua.md)
- [Unlocking the Potential of Diffusion Priors in Blind Face Restoration](unlocking_the_potential_of_diffusion_priors_in_blind_face_restoration.md)
- [BVINet: Unlocking Blind Video Inpainting with Zero Annotations](bvinet_unlocking_blind_video_inpainting_with_zero_annotations.md)

<!-- RELATED:END -->
