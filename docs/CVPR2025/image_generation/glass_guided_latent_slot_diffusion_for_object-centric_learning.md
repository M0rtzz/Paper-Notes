---
title: >-
  [论文解读] GLASS: Guided Latent Slot Diffusion for Object-Centric Learning
description: >-
  [CVPR 2025][图像生成][物体中心学习] 本文提出 GLASS，一种基于 Slot Attention 的物体中心学习方法，通过在扩散模型生成的图像空间中学习，利用语义引导模块（扩散模型的交叉注意力生成伪语义掩码）和实例引导模块（MLP 重建编码器特征）协同解决过分割和欠分割问题，在真实场景的物体发现和条件/组合生成任务上大幅超越前方法。
tags:
  - CVPR 2025
  - 图像生成
  - 物体中心学习
  - 注意力机制
  - 扩散模型
  - 语义引导
  - 实例分割
---

# GLASS: Guided Latent Slot Diffusion for Object-Centric Learning

**会议**: CVPR 2025  
**arXiv**: [2407.17929](https://arxiv.org/abs/2407.17929)  
**代码**: https://visinf.github.io/glass/ (有)  
**领域**: 扩散模型 / 多模态VLM  
**关键词**: 物体中心学习, Slot Attention, 扩散模型, 语义引导, 实例分割

## 一句话总结
本文提出 GLASS，一种基于 Slot Attention 的物体中心学习方法，通过在扩散模型生成的图像空间中学习，利用语义引导模块（扩散模型的交叉注意力生成伪语义掩码）和实例引导模块（MLP 重建编码器特征）协同解决过分割和欠分割问题，在真实场景的物体发现和条件/组合生成任务上大幅超越前方法。

## 研究背景与动机

**领域现状**：物体中心学习（OCL）旨在将图像分解为一组潜在物体表示（slots），每个 slot 竞争性地绑定到一个物体。Slot Attention 是最流行的 OCL 方法，但长期受限于合成数据集（如 CLEVR），在真实图像上表现不佳。近期方法（如 DINOSAUR、SPOT、StableLSD）使用预训练 DINOv2 编码器和扩散模型解码器来处理真实图像，但仍存在显著的物体分割质量问题。

**现有痛点**：真实场景中 slot 表示面临严重的 part-whole 层次歧义——过分割（一个物体被拆分到多个 slot）和欠分割（多个物体被合并到一个 slot）同时存在。DINOSAUR 用 MLP 重建 DINOv2 特征作为训练信号，虽然特征本身具有实例感知能力，但缺乏语义约束导致边界不精确。StableLSD 用扩散模型解码但 slot 质量仍有限。没有方法能同时支持物体发现、条件生成和组合生成。

**核心矛盾**：语义级别的引导信号可以解决过分割（确保同一物体的不同部分被同一 slot 覆盖），但会导致 slot 漂向语义类别而非实例（欠分割）。实例级别的特征重建信号有利于实例区分，但缺乏语义边界约束。

**本文目标** 在 Slot Attention 框架下同时解决过分割和欠分割问题，使 slot 既尊重语义边界又区分不同实例，并同时支持物体发现、条件图像生成和组合场景生成。

**切入角度**：在扩散模型生成的图像上训练（而非真实图像），这样可以利用扩散模型的交叉注意力层提取伪语义掩码作为引导信号。将语义引导（BCE 损失对齐 slot 掩码与伪语义掩码）和实例引导（MSE 损失重建 DINOv2 特征）结合，两者互补解决 part-whole 歧义。

**核心 idea**：在生成图像上训练 Slot Attention，用扩散模型自身的交叉注意力提取语义引导 + DINOv2 特征重建提供实例引导，协同解决真实场景的过分割和欠分割问题。

## 方法详解

### 整体框架
输入真实图像 → BLIP-2 生成文本描述 → 提取名词类标签构建 prompt → 用预训练 Stable Diffusion 从随机噪声生成图像 $\mathbf{I}_{gen}$ → 从扩散模型提取语义掩码 $\mathbf{M}_{gen}$ → 将 $\mathbf{I}_{gen}$ 输入 DINOv2 编码器 → Slot Attention 生成 slots → 扩散解码器重建图像 + MLP 解码器重建特征 → 用 Hungarian 匹配将 slot 掩码与语义掩码对齐 → 三项损失联合训练。推理时直接对真实图像编码运行 Slot Attention。

### 关键设计

1. **语义引导模块（Semantic Guidance）**:

    - 功能：利用扩散模型的交叉注意力层生成伪语义掩码，引导 slot 掩码对齐物体边界，解决过分割
    - 核心思路：为每个类标签 $c_k \in \mathcal{C}$ 单独创建单 token prompt，提取扩散模型在多个时间步和分辨率上的交叉注意力图，取平均得到粗糙注意力图。用自注意力图对其进行细化（指数化自注意力 × 交叉注意力），再 argmax + 范围阈值化得到最终语义掩码 $\mathbf{M}_{gen}$。通过 Hungarian 匹配将 slot 掩码与语义掩码组件配对，对匹配的 slot 施加 BCE 损失
    - 设计动机：扩散模型训练时已学会将文本 token 与图像区域对齐，其交叉注意力天然提供了粗粒度语义分割信息。将这个信号用于引导 slot 可以确保 slot 不会把同一语义物体分成多块

2. **实例引导模块（Instance Guidance）**:

    - 功能：通过重建 DINOv2 编码器特征来保持 slot 的实例感知能力，解决欠分割（防止语义引导导致的 slot 漂移到语义类别）
    - 核心思路：每个 slot 通过一个轻量 MLP 解码器重建对应区域的 DINOv2 特征 $\mathbf{F}_{inp}$，用 MSE 损失 $\mathcal{L}_{instance} = \text{MSE}(\mathbf{F}_{inp}, \mathbf{F}_{recon})$ 训练。DINOv2 的 patch 特征已经展示出实例感知属性（同一物体的不同 patch 特征相似），因此重建这些特征会自然推动 slot 向实例级别绑定
    - 设计动机：单独使用语义引导会使 slot 偏向语义类别（例如两只猫被合并到一个"猫"类 slot），加入特征重建后 slot 必须区分不同实例才能准确重建每个区域的特征

3. **在生成图像空间中学习**:

    - 功能：使 slot 在分布匹配真实数据的生成图像上训练，实现对真实图像的零样本泛化
    - 核心思路：用 BLIP-2 从真实图像生成文本描述，用 SD 生成对应图像，所有训练都在生成图像上进行。推理时直接对真实图像运行 Slot Attention（无需 SD 生成步骤）
    - 设计动机：在生成图像上训练有两大优势——(1) 可以利用 SD 的交叉注意力获取语义掩码（真实图像无此信号）；(2) SD 生成图像的分布已接近真实数据分布，模型可以零样本迁移到真实图像

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \mathcal{L}_{MSE}(\mathbf{I}_{gen}, \mathbf{I}_{recon}) + \lambda_s \mathcal{L}_{BCE}(\mathbf{P}(\mathbf{M}_{gen}, \mathbf{A})) + \lambda_i \mathcal{L}_{MSE}(\mathbf{F}_{inp}, \mathbf{F}_{recon})$，$\lambda_s = 0.7$，$\lambda_i = 0.9$。训练分两阶段：Phase 1 只训练 Slot Attention + MLP 解码器（学习实例级 slot 表示）；Phase 2 联合训练扩散解码器（用低学习率微调 slot attention），使扩散解码器对齐到 slot 嵌入。

## 实验关键数据

### 主实验（实例级物体发现）

| 方法 | VOC mIoU_i↑ | VOC mBO_i↑ | VOC mBO_c↑ | COCO mIoU_i↑ | COCO mBO_i↑ |
|------|------------|-----------|-----------|-------------|------------|
| DINOSAUR-Trans. | 42.0 | 43.2 | 47.8 | 31.6 | 33.3 |
| SPOT | 48.8 | 48.3 | 55.6 | 34.0 | 35.0 |
| StableLSD | 31.5 | 32.1 | 35.4 | 24.7 | 25.9 |
| **GLASS** | **58.1 (+9.3)** | **58.9 (+8.5)** | **62.2 (+6.6)** | **38.9 (+4.9)** | **40.6 (+5.6)** |

在 VOC 上 mIoU_i 相比前 SOTA（SPOT）提升 9.3%，COCO 上提升 4.9%。

### 消融实验

| 配置 | VOC mIoU_i↑ | COCO mIoU_i↑ |
|------|------------|-------------|
| Only Instance Guidance | ~42 | ~31 |
| Only Semantic Guidance | ~50 | ~35 |
| Semantic + Instance (GLASS) | 58.1 | 38.9 |

### 关键发现
- **语义和实例引导缺一不可**：单用语义引导过分割减少但出现欠分割（同类物体合并），单用实例引导有实例区分但边界不精确。两者组合在 VOC 上比单用任一高出约 8-16%
- GLASS 甚至超越了使用语言条件的语义分割方法（如 DiffuMask、SegCLIP）在语义级物体发现上的表现
- 首次在真实复杂场景上实现了 slot attention 的组合生成——可以交换不同图像的 slot 来组合新场景
- GLASS 在零样本跨数据集评估（Object365、CLEVRTex）上也保持优势
- 与使用额外弱监督（如边界框定位、已知物体数量）的 StableLSD 变体相比，GLASS 无需这些信息仍然更优

## 亮点与洞察
- **在生成图像上训练是核心创新**：这个设计一箭双雕——既获得了扩散模型的语义掩码（交叉注意力），又利用了生成分布接近真实分布的特性实现零样本迁移。这个"在合成数据上训练，在真实数据上部署"的范式可迁移到其他视觉任务
- **语义 + 实例引导的互补机制设计精妙**：过分割和欠分割是 OCL 的核心困境，本文通过两个正交方向的引导信号优雅地解决了两者。BCE 语义损失确保 slot 覆盖完整物体，MSE 特征重建确保 slot 区分不同实例
- **slot attention 首次实现真实场景组合生成**：可以从不同图像中取出 slot 组合成新场景，这在视觉推理和场景编辑中有重要价值

## 局限与展望
- 依赖 BLIP-2 生成描述和 CLIP 嵌入，描述不准确会影响语义掩码质量
- 伪语义掩码的质量受扩散模型交叉注意力精度限制，对小物体和部分遮挡场景可能不够精细
- 训练需要运行完整的 SD 生成流程获取伪掩码，计算开销较大
- 组合生成的结果虽然首次在真实场景可行，但精细度和一致性仍有提升空间
- Phase 2 的扩散解码器微调需要仔细调参，学习率过大会破坏 Phase 1 学到的 slot 表示

## 相关工作与启发
- **vs DINOSAUR**: DINOSAUR 只用 MLP 重建 DINOv2 特征，缺乏语义引导，边界不精确。GLASS 增加了语义引导模块显著改善了物体发现质量
- **vs SPOT**: SPOT 使用 Transformer 编解码器但无语义引导，GLASS 在 VOC mIoU_i 上超越 9.3%，且不会将同类不同实例混淆
- **vs StableLSD**: StableLSD 也用扩散解码器但缺乏引导模块，GLASS 大幅超越。即使 StableLSD 使用额外弱监督（bbox），GLASS 仍然更优
- **vs 语义分割方法（DiffuMask 等）**: 这些方法只做语义分割，无法区分实例、不能生成图像。GLASS 在语义发现上也超越这些专门方法

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 在生成空间训练 + 语义/实例双重引导的设计非常新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 多任务评估（发现/生成/组合/属性预测）、多数据集、跨数据集零样本
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐⭐ 对物体中心学习领域有重要推动，首次实现真实场景组合生成

<!-- RELATED:START -->

## 相关论文

- [Improved Object-Centric Diffusion Learning with Registers and Contrastive Alignment (CODA)](../../ICLR2026/image_generation/improved_object-centric_diffusion_learning_with_registers_and_contrastive_alignm.md)
- [MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)
- [GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning](../../ICCV2025/image_generation/genflowrl_shaping_rewards_with_generative_object-centric_flow_in_visual_reinforc.md)
- [LEDiff: Latent Exposure Diffusion for HDR Generation](lediff_latent_exposure_diffusion_for_hdr_generation.md)
- [Multitwine: Multi-Object Compositing with Text and Layout Control](multitwine_multi-object_compositing_with_text_and_layout_control.md)

<!-- RELATED:END -->
