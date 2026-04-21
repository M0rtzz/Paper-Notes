---
title: >-
  [论文解读] RAD: Region-Aware Diffusion Models for Image Inpainting
description: >-
  [CVPR 2025][图像生成][扩散模型] RAD通过为每个像素分配不同的噪声调度实现区域异步生成，仅需对vanilla扩散模型进行最小的结构改动（将FC层改为1×1卷积），就能在推理速度提升100倍的同时达到SOTA修复质量。
tags:
  - CVPR 2025
  - 图像生成
  - 扩散模型
  - 图像修复
  - 区域感知
  - 像素级噪声调度
  - LoRA微调
---

# RAD: Region-Aware Diffusion Models for Image Inpainting

**会议**: CVPR 2025  
**arXiv**: [2412.09191](https://arxiv.org/abs/2412.09191)  
**代码**: 无  
**领域**: 图像生成 / 图像修复  
**关键词**: 扩散模型, 图像修复, 区域感知, 像素级噪声调度, LoRA微调

## 一句话总结

RAD通过为每个像素分配不同的噪声调度实现区域异步生成，仅需对vanilla扩散模型进行最小的结构改动（将FC层改为1×1卷积），就能在推理速度提升100倍的同时达到SOTA修复质量。

## 研究背景与动机

扩散模型在图像修复中的应用主要有两类方法，都存在显著缺陷：

1. **劫持预训练模型的反向过程**：RePaint、MCG等方法无需额外训练，但需要复杂的嵌套循环（如反复重采样步骤），导致推理时间极长——比RAD慢100倍。

2. **条件生成框架**：SmartBrush等方法需要额外模块处理条件信息（文本+mask），增加了复杂性和计算负担。

两类方法的根本问题在于：标准扩散模型设计用于全局生成，对所有像素施加统一的噪声调度，天然不适合只需生成局部区域的修复任务。

RAD的核心洞察：如果让不同像素有不同的噪声强度，某些区域可以完全去噪（已知区域）而其他区域保持噪声（待生成区域），就天然模拟了修复场景。这个想法概念简单，但需要解决噪声调度设计和网络适配的细节问题。

## 方法详解

### 整体框架

RAD是对DDPM的逐元素重新表述：前向过程为每个像素$i$定义独立的噪声方差$b_{t,i}$，使得$q(x_{t,i}|x_{t-1,i}) = \mathcal{N}(\sqrt{1-b_{t,i}} x_{t-1,i}, b_{t,i})$。反向过程使用同一U-Net预测噪声，但通过空间噪声嵌入感知各像素的噪声强度。训练基于Perlin噪声生成的伪真实修复mask，并使用LoRA在预训练ADM上微调。

### 关键设计1: 空间变化噪声调度

- **功能**: 为每个像素分配不同的噪声调度，实现区域异步生成
- **核心思路**: 将扩散过程分为两个阶段：Phase 1仅对mask内像素加噪（$T_1$步），Phase 2对mask外像素加噪（$T_2$步），$T_1 + T_2 = T$。实际生成时只需反向Phase 1即可完成修复。使用**Perlin噪声**生成训练mask：其光滑且自然的模式能模拟多样的真实修复模式。通过随机采样空间尺度和黑白转换阈值实现mask的多样性
- **设计动机**: 独立像素的随机噪声调度缺乏空间模式，与实际修复场景不一致，训练效果差。Perlin噪声能提供既多样又自然的mask形状，成功避免了这一问题

### 关键设计2: 空间噪声嵌入

- **功能**: 让去噪网络感知每个像素的噪声强度，适应空间变化的噪声问题
- **核心思路**: 在标准DDPM中，时间步$t$通过cos-sin编码+FC层嵌入并加到U-Net特征图的所有像素上。RAD将FC层替换为$1\times 1$卷积，用$\bar{b}_{t}$（像素级累积噪声强度）替代标量$t$作为输入。这样每个像素独立获得其噪声强度条件，无需改变其他任何组件
- **设计动机**: 原始$t$嵌入的作用是告知网络噪声的整体强度。在RAD中不同像素噪声强度不同，需要像素级的条件信息。将FC层改为$1\times 1$卷积是最小的结构改动

### 关键设计3: LoRA微调与时间步反映射

- **功能**: 利用预训练扩散模型减少训练开销
- **核心思路**: 直接LoRA微调时，空间噪声嵌入对预训练模型的改动过于剧烈。解决方案是将$\bar{b}_{t,i}$通过线性插值反映射回DDPM的等效时间步值（可为非整数），使输入分布与预训练模型更兼容。这使得LoRA可以有效微调，大幅降低训练资源需求
- **设计动机**: RAD框架需要重新训练，这是相比劫持预训练模型方法的缺点。LoRA+时间步反映射解决了这一问题

### 损失函数

使用iDDPM/ADM的混合损失——变分损失（公式6）与简化损失$L = \sum_{t \geq 1} \mathbb{E}_q[\|\epsilon_t - \epsilon_\theta(x_t, t)\|^2]$（公式7）的组合，其中所有项均为逐元素版本。

## 实验关键数据

### 主实验: FFHQ和LSUN Bedroom上的修复性能

| 方法 | FFHQ-Box FID↓ | FFHQ-Wide FID↓ | LSUN-Box FID↓ | 推理速度 |
|------|--------------|---------------|--------------|---------|
| LaMa | 27.7 | 23.2 | - | 快 |
| Score-SDE | 30.3 | 29.8 | 23.7 | 慢 |
| RePaint | ~高 | ~高 | ~高 | RAD的100x |
| DDRM | ~中 | ~中 | ~中 | 慢 |
| **RAD** | **最优** | **最优/次优** | **最优** | **最快** |

### 消融实验: 各组件贡献

| 配置 | FID | LPIPS |
|------|-----|-------|
| 无空间噪声嵌入 | 性能下降 | 性能下降 |
| 独立像素随机mask | 性能差 | 性能差 |
| Perlin噪声mask | 最优 | 最优 |
| 无LoRA(从头训练) | 可行但慢 | 类似 |

### 关键发现

- 推理速度比SOTA扩散修复方法快100倍（因为只需要普通反向过程，无嵌套循环）
- 在FFHQ和LSUN上多种mask类型下FID和LPIPS最优或次优
- 即使mask边界尖锐，修复结果无明显边界效应
- 空间噪声嵌入和Perlin噪声mask对性能至关重要
- LoRA微调可成功利用预训练ADM，大幅减少训练成本

## 亮点与洞察

1. **极简的重新表述**：仅将"所有像素统一噪声"改为"每像素独立噪声"，加上FC→1×1卷积的最小改动，就实现了SOTA修复
2. **100倍加速**：通过将修复内化到扩散框架中（而非外部操纵），无需嵌套循环
3. **Perlin噪声作为mask代理分布**：巧妙利用计算机图形学中的程序化噪声作为训练mask的来源

## 局限与展望

- 需要重新训练（虽然LoRA缓解了这一问题）
- 当前实验仅在256×256分辨率上进行
- 未与文本引导的修复方法直接对比（问题设置不同）
- 未来可将RAD扩展到stable diffusion等模型

## 相关工作与启发

- **RePaint**: 通过重采样步骤协调mask/非mask区域，但速度极慢
- **SmartBrush**: 也只在修复区域加噪，但需要额外模块
- **DiffEdit**: 使用DDIM反演和文本生成的mask，但依赖stable diffusion
- 启发：扩散模型的"简单"变体（逐元素噪声调度）蕴含着巨大的实用价值

## 评分

⭐⭐⭐⭐ — 概念极其简洁，实现优雅，100倍的速度提升具有重要的实用意义。从FC到1×1卷积的最小改动就解锁了全新能力，展示了对扩散模型基础理论的深刻理解。

<!-- RELATED:START -->

## 相关论文

- [Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation](focus-n-fix_region-aware_fine-tuning_for_text-to-image_generation.md)
- [MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)
- [From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](../../CVPR2026/image_generation/from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting](turbofill_adapting_few-step_text-to-image_model_for_fast_image_inpainting.md)
- [One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image Inpainting](../../NeurIPS2025/image_generation/one_stone_with_two_birds_a_null-text-null_frequency-aware_diffusion_models_for_t.md)

<!-- RELATED:END -->
