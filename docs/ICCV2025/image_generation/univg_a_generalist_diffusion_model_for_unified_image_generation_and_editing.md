---
title: >-
  [论文解读] UniVG: A Generalist Diffusion Model for Unified Image Generation and Editing
description: >-
  [ICCV 2025][图像生成][统一生成模型] 提出UniVG,基于MM-DiT的统一图像生成模型,通过通道维拼接输入、渐进式多任务训练和外部条件注入,用单套权重支持T2I生成、编辑、ID保持、布局引导、深度估计等多种任务。 扩散模型在T2I、编辑、个性化等各类任务上取得了巨大进展,但导致了模型碎片化：每个任务需要独立的…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "统一生成模型"
  - "MM-DiT"
  - "多任务训练"
  - "指令编辑"
  - "Flow Matching"
---

# UniVG: A Generalist Diffusion Model for Unified Image Generation and Editing

**会议**: ICCV 2025  
**arXiv**: [2503.12652](https://arxiv.org/abs/2503.12652)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 统一生成模型, MM-DiT, 多任务训练, 指令编辑, Flow Matching

## 一句话总结

提出UniVG,基于MM-DiT的统一图像生成模型,通过通道维拼接输入、渐进式多任务训练和外部条件注入,用单套权重支持T2I生成、编辑、ID保持、布局引导、深度估计等多种任务。

## 研究背景与动机

扩散模型在T2I、编辑、个性化等各类任务上取得了巨大进展,但导致了**模型碎片化**:每个任务需要独立的架构、训练和参数集。

现有统一模型(OmniGen, OneDiffusion)已证明可行性,但存在不足:

**OmniGen**: 将输入图像沿序列维度拼接,训练/推理效率低
2. 两者均**缺乏关键消融研究** — 最优数据配比和训练策略不清晰
3. 任务间的协同/冲突关系未被深入探索

核心问题:如何在不牺牲T2I核心能力的前提下,统一支持多种图像生成任务?

## 方法详解

### 最小化架构修改

将输入图像latent、噪声latent和掩码沿**通道维度拼接**(而非序列维度):

$$d = [z_t \oplus \text{VAE}_{Enc}(\mathcal{V}) \oplus \text{Resize}(\mathcal{M})]$$

优势:固定长度序列,避免OmniGen的可变长度导致的效率问题和上下文感知中断。

### 条件注入

外部条件(如人脸特征)通过**嵌入替换**注入:提取条件特征 $f = \mathcal{H}(\mathcal{C})$,替换预设占位符token的嵌入。

### 多任务训练

Flow Matching损失:
$$\mathcal{L} = \mathbb{E}[\|\mathcal{F}([\{p\}, t, d]) - u_t\|^2]$$

**三阶段训练配方**:
1. **基础训练**: 从头训练T2I (400K步, bs=512)
2. **多任务训练**: 引入编辑(47%)、补全(20%)、辅助(3%)、布局(2%)等 (400K步)
3. **进一步微调**: 添加ID保持生成(1:1配比, 40K步)

### 各任务输入格式

- **T2I**: 空白图像+全白掩码+`<t2i>`
- **指令编辑**: 输入图像+空白掩码+`<ie>`指令
- **深度估计**: 输入图像+全白掩码+`<depth>`
- **布局引导**: 布局可视化图+全白掩码+`<lg>`+描述
- **ID保持**: CLIP人脸嵌入替换`<p>`占位符

### 双CFG推理

$$\mathcal{F} \implies \mathcal{F}(\varnothing,t,\{z_t,\varnothing,\varnothing\}) + \alpha_\mathcal{V}(\cdot) + \alpha_\mathcal{X}(\cdot)$$

分别控制图像条件和文本条件的引导强度。

## 实验

### 文本到图像生成

| 方法 | 参数量 | GenEval↑ | CompBench↑ | DSG↑ | HPSv2↑ |
|------|--------|----------|------------|------|--------|
| SDXL | 2.6B | 0.55 | 0.42 | 0.72 | 27.7 |
| FLUX.1 | 12.0B | 0.66 | 0.47 | 0.73 | 29.2 |
| SD3 | 8.0B | 0.71 | 0.49 | 0.76 | 28.9 |
| OmniGen | 3.8B | 0.70 | 0.46 | 0.66 | 27.7 |
| **UniVG** | **3.7B** | **0.70** | **0.48** | **0.75** | **28.2** |

### 指令编辑 (CLIP-T / CLIP-I)

| 方法 | MagicBrush CLIP-T↑ | MagicBrush CLIP-I↑ | EmuEdit CLIP-T↑ |
|------|-------------------|-------------------|----------------|
| InsP2P | 24.5 | 83.7 | 21.9 |
| EmuEdit | 26.1 | 89.7 | 23.1 |
| OmniGen | 25.8 | 86.3 | 23.1 |
| **UniVG** | **29.5** | 86.3 | **25.9** |

### 关键发现

1. **T2I与编辑可共存** — 多任务训练不损害核心T2I能力,GenEval达到0.70
2. **辅助任务增强编辑** — 深度估计和分割提升了图像编辑的空间理解
3. UniVG在CLIP-T上显著超越所有方法(包括任务专用模型),编辑指令遵循能力极强
4. 参数量(3.7B)小于OmniGen(3.8B),但性能更优,体现架构效率

## 亮点与洞察

1. **通道拼接vs序列拼接** — 通道维度拼接是效率的关键,保持固定长度序列
2. **渐进式训练** — 先T2I基础,再多任务,最后ID保持,避免灾难性遗忘
3. **任务间协同关系的发现** — 辅助感知任务(深度/分割)增强生成/编辑任务
4. **占位符token的条件注入** — 维持序列长度不变,灵活支持各种条件类型

## 局限性

- ID保持生成需要额外阶段三训练(会与其他任务冲突)
- 法线估计质量不足限制了其作为辅助任务的贡献
- 未开源,验证困难

## 相关工作

- **T2I模型**: Stable Diffusion, FLUX.1, SD3
- **统一模型**: OmniGen, OneDiffusion, TransFusion
- **编辑模型**: InstructPix2Pix, EmuEdit, MGIE

## 评分

- 新颖性: ⭐⭐⭐⭐ (统一模型非全新,但训练策略研究深入)
- 技术深度: ⭐⭐⭐⭐ (全面的消融研究和训练配方)
- 实验充分度: ⭐⭐⭐⭐⭐ (多任务全面评测)
- 实用价值: ⭐⭐⭐⭐ (单模型多任务,部署友好)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] UniEgoMotion: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation](uniegomotion_a_unified_model_for_egocentric_motion_reconstruction_forecasting_an.md)
- [\[CVPR 2025\] A Unified Image-Dense Annotation Generation Model for Underwater Scenes](../../CVPR2025/3d_vision/a_unified_image-dense_annotation_generation_model_for_underwater_scenes.md)
- [\[ICCV 2025\] Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)
- [\[ICCV 2025\] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)

</div>

<!-- RELATED:END -->
