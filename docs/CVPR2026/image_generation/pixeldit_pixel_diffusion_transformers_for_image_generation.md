---
title: >-
  [论文解读] PixelDiT: Pixel Diffusion Transformers for Image Generation
description: >-
  [CVPR 2026][图像生成][像素扩散] PixelDiT 提出完全基于Transformer的双层像素空间扩散模型：patch级DiT捕捉全局语义 + pixel级DiT细化纹理细节，无需VAE即可在ImageNet上达到1.61 FID，并直接在1024分辨率像素空间训练文本到图像模型。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "像素扩散"
  - "Transformer"
  - "端到端生成"
  - "像素建模"
  - "文本到图像"
---

# PixelDiT: Pixel Diffusion Transformers for Image Generation

**会议**: CVPR 2026  
**arXiv**: [2511.20645](https://arxiv.org/abs/2511.20645)  
**代码**: [https://github.com/](https://github.com/)  
**领域**: 图像生成  
**关键词**: 像素扩散, 双层Transformer, 端到端生成, 像素建模, 文本到图像

## 一句话总结
PixelDiT 提出完全基于Transformer的双层像素空间扩散模型：patch级DiT捕捉全局语义 + pixel级DiT细化纹理细节，无需VAE即可在ImageNet上达到1.61 FID，并直接在1024分辨率像素空间训练文本到图像模型。

## 研究背景与动机
1. **领域现状**：潜空间扩散是DiT的标准范式，但依赖预训练autoencoder引入有损重建，限制了采样保真度并阻碍联合优化。
2. **现有痛点**：像素空间扩散面临像素建模的核心挑战——需要同时处理全局语义和高频细节。激进patchification损失细节，小patch/长序列则计算爆炸。
3. **核心矛盾**：缺乏一种高效的像素建模机制能同时捕捉全局语义和逐像素更新。
4. **本文目标**：设计纯Transformer的像素空间扩散模型，显式结构化像素建模。
5. **切入角度**：将语义学习与像素级更新解耦为两个层次，各用不同粒度的Transformer处理。
6. **核心idea**：patch级pathway做长距离语义注意力（粗粒度），pixel级pathway做密集逐像素建模（细粒度），通过pixel-wise AdaLN和token compaction连接。

## 方法详解

### 整体框架
双层架构：patch级DiT以aggressive patch size处理短token序列捕获全局布局；pixel级DiT（PiT blocks）在像素粒度上细化纹理。pixel-wise AdaLN用语义token调制每个像素token，pixel token compaction在全局注意力前压缩像素token后解压回。

### 关键设计

1. **Pixel-wise AdaLN调制**:
    - 功能：将patch级的语义信息注入到每个像素token的处理中
    - 核心思路：不同于标准AdaLN使用全局条件（如timestep），PixelDiT用patch级语义token为每个像素token生成独立的调制参数。每个像素token根据其空间位置对应的语义token获得特定的scale和shift。
    - 设计动机：全局条件对所有像素一视同仁，但不同空间位置需要不同的语义引导。像素级调制实现了空间自适应的条件注入。

2. **Pixel Token Compaction**:
    - 功能：在保持逐像素建模的同时使全局注意力计算可行
    - 核心思路：在全注意力之前，将每个像素token通过线性投影压缩到更低维度；注意力计算后再解压回原始维度。这使得像素级pathway可以在全分辨率token序列上执行全局注意力而不导致计算爆炸。
    - 设计动机：像素级token数量庞大（256×256分辨率=65536 tokens），直接做全注意力不可行。Compaction在维度上压缩而非数量上减少，保留了空间分辨率。

3. **双层路径融合**:
    - 功能：将语义学习和纹理细化在架构层面分离
    - 核心思路：patch级pathway由N个增强DiT blocks组成，使用RMSNorm和2D RoPE。pixel级pathway的PiT blocks接收patch级输出作为语义条件，通过pixel-wise AdaLN和compaction attention生成最终像素级速度预测。
    - 设计动机：将大部分语义推理集中在低分辨率网格上，减轻了像素级pathway的负担并加速了学习。

### 损失函数 / 训练策略
标准conditional flow matching损失，在像素空间直接训练。文本到图像使用multi-modal DiT blocks。

## 实验关键数据

### 主实验

| 方法 | 类型 | FID↓ (256) | FID↓ (512) | 说明 |
|------|------|-----------|-----------|------|
| PixelDiT | 像素 | 1.61 | 1.81 | 像素空间SOTA |
| DeCo | 像素 | 1.62 | 2.22 | 频率解耦方法 |
| DiT-XL/2 | 潜空间 | 2.27 | - | 需要VAE |
| PixelFlow | 像素 | - | - | 层级方法 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Pixel-wise AdaLN | 优于全局AdaLN | 空间自适应调制有效 |
| Token Compaction | 优于无压缩 | 使全局注意力可行 |
| 双层 vs 单层 | 双层显著更优 | 解耦设计是关键 |

### 关键发现
- PixelDiT在像素空间模型中达到最低FID，证明纯Transformer架构在像素空间也能高效工作。
- 像素空间模型在图像编辑任务中天然避免了VAE重建伪影，背景保持更好。
- 可以直接在1024分辨率像素空间训练T2I模型，GenEval达0.74，DPG-Bench达83.5。

## 亮点与洞察
- **完全端到端**：无VAE的纯Transformer架构是最简洁的生成pipeline。
- **Token Compaction**是实用的工程创新：维度压缩而非空间下采样，保留了全空间分辨率。
- 证明了像素空间扩散可以在所有指标上接近甚至超越潜空间扩散。

## 局限与展望
- 相比LDM，训练成本仍然更高。
- 文本到图像在benchmark分数上略逊于最好的LDM（如FLUX）。
- 未来可结合更先进的训练技巧进一步缩小差距。

## 相关工作与启发
- **vs DeCo**: DeCo用无注意力的线性解码器，PixelDiT用带注意力的PiT blocks。两者思路类似但实现不同。
- **vs PixNerd**: 用神经场层预测像素速度，PixelDiT用纯Transformer更标准。

## 评分
- 新颖性: ⭐⭐⭐⭐ 双层像素Transformer设计新颖但与DeCo并行
- 实验充分度: ⭐⭐⭐⭐⭐ ImageNet+T2I+编辑多任务验证
- 写作质量: ⭐⭐⭐⭐ 架构描述详细清晰
- 价值: ⭐⭐⭐⭐ 推动像素扩散重新成为可行范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation](deco_frequency-decoupled_pixel_diffusion_for_end-to-end_image_generation.md)
- [\[CVPR 2026\] EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation](edgedit_hardware-aware_diffusion_transformers_for_efficient_on-device_image_gene.md)
- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](dip_taming_diffusion_models_in_pixel_space.md)
- [\[CVPR 2026\] Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers](circuit_mechanisms_for_spatial_relation_generation_in_diffusion_models.md)
- [\[CVPR 2026\] Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](pluggable_pruning_with_contiguous_layer_distillation_for_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
