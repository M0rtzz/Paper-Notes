---
title: >-
  [论文解读] Multi-focal Conditioned Latent Diffusion for Person Image Synthesis
description: >-
  [CVPR 2025][图像生成][姿态引导人物图像生成] MCLD通过将源人物图像解耦为面部区域、外观纹理和整体图像三个焦点条件，设计多焦点条件聚合模块(MFCA)在UNet不同阶段选择性注入不同条件，有效缓解了LDM压缩导致的面部和纹理细节退化问题，在DeepFashion上取得SOTA。
tags:
  - CVPR 2025
  - 图像生成
  - 姿态引导人物图像生成
  - 潜在扩散模型
  - 多焦点条件
  - 人脸身份保持
  - 外观纹理保持
---

# Multi-focal Conditioned Latent Diffusion for Person Image Synthesis

**会议**: CVPR 2025  
**arXiv**: [2503.15686](https://arxiv.org/abs/2503.15686)  
**代码**: https://github.com/jqliu09/mcld  
**领域**: 扩散模型 / 人物图像生成  
**关键词**: 姿态引导人物图像生成, 潜在扩散模型, 多焦点条件, 人脸身份保持, 外观纹理保持

## 一句话总结
MCLD通过将源人物图像解耦为面部区域、外观纹理和整体图像三个焦点条件，设计多焦点条件聚合模块(MFCA)在UNet不同阶段选择性注入不同条件，有效缓解了LDM压缩导致的面部和纹理细节退化问题，在DeepFashion上取得SOTA。

## 研究背景与动机
1. **领域现状**：姿态引导人物图像生成(PGPIS)旨在将源图像转换为目标姿态同时保持外观和身份。GAN方法受训练不稳定和模式坍塌限制，基于LDM的方法（如PIDM、CFLD、PoCoLD）因在潜空间操作而具有更好的生成质量和高分辨率支持。
2. **现有痛点**：LDM的自编码器压缩过程是有损的，特别会劣化高频区域的细节——面部特征和衣物纹理在编码过程中丢失严重。这种退化在推理时更加恶化，因为生成的latent与真实压缩latent存在偏差（如图1所示，添加微小偏移ε=0.2即显著恶化细节）。
3. **核心矛盾**：LDM方法使用整张源图像作为条件，无法聚焦到面部和纹理这些关键的敏感区域。姿态和外观信息的纠缠也使得跨多样姿态的细节重建更加困难。
4. **本文目标** (1) 如何缓解LDM压缩对面部和纹理的退化？(2) 如何设计条件控制使模型关注敏感区域？(3) 如何在不同姿态下保持身份一致性？
5. **切入角度**：不再用整图作为唯一条件，而是将图像解耦为面部身份、外观纹理和全局语义三个独立焦点，分别用不同预训练模型编码为姿态无关的嵌入，然后在UNet不同阶段选择性注入。
6. **核心 idea**：通过多焦点条件解耦+阶段选择性注入，让LDM在不同处理阶段关注不同层次的信息，从而在压缩限制下最大化保持面部身份和外观细节。

## 方法详解

### 整体框架
基于Stable Diffusion 1.5构建双分支条件扩散模型。输入为源人物图像和目标DensePose。第一分支通过ReferenceNet处理外观纹理图的VAE编码，提供低层语义和细节特征 $c_{ref}$，与UNet特征逐阶段拼接。第二分支利用CLIP图像编码器、CLIP编码器和面部识别模型分别提取源图像嵌入 $\mathcal{I}_{emb}$、纹理图嵌入 $\mathcal{A}_{emb}$ 和面部嵌入 $\mathcal{F}_{emb}$，通过多焦点条件聚合模块(MFCA)融合注入UNet。目标姿态通过DensePose经轻量级Pose Guider模块处理后融入UNet。

### 关键设计

1. **多焦点区域提取与嵌入**:

    - 功能：从源图像解耦出三种姿态无关的条件表示
    - 核心思路：(1) 面部区域 $\mathcal{F}$：用人脸检测器裁剪，经预训练面部识别模型(antelopev2)提取身份特征，投影到统一维度得到 $\mathcal{F}_{emb}$。面部识别模型天然具有跨视角/跨表情的鲁棒性。(2) 外观区域 $\mathcal{A}$：通过DensePose估计将源图变形为SMPL纹理图（200×200到512×512），完全解耦外观与姿态。CLIP编码得到 $\mathcal{A}_{emb}$，VAE编码经ReferenceNet得到 $c_{ref}$。(3) 源图像通过CLIP得到 $\mathcal{I}_{emb}$
    - 设计动机：面部和纹理区域虽然只占图像小部分，却包含最重要的感知变化信息。通用CLIP编码器无法准确捕获面部身份特征，因此用专门的面部识别模型

2. **多焦点条件聚合模块(MFCA)**:

    - 功能：在UNet不同阶段选择性注入不同焦点条件
    - 核心思路：在UNet的编码器 $\mathcal{U}_\mathcal{E}$、中间层 $\mathcal{U}_\mathcal{M}$、解码器 $\mathcal{U}_\mathcal{D}$ 分别注入不同条件——编码阶段注入全局语义 $\mathcal{I}_{emb}$（高层语义如衣服类别），中间层同时注入 $\mathcal{I}_{emb}$ 和 $\mathcal{A}_{emb}$，解码阶段注入细粒度 $\mathcal{A}_{emb}$（纹理细节）。面部嵌入 $\mathcal{F}_{emb}$ 保持全程注入。通过多个cross-attention层实现：$\phi = \sum_{i\in\{s, F_{emb}\}} \lambda_i Attn(Q, K_i, V_i)$
    - 设计动机：不同于简单拼接所有条件（会导致每个条件的关注区域模糊不清），根据UNet不同层的功能特点精确投放信息：编码阶段处理高层语义、解码阶段重建细节。这种策略减少了参数量并引导模型在每个阶段优先处理最相关的信息

3. **面部专注损失与DensePose姿态引导**:

    - 功能：额外的面部区域监督 + 3D姿态引导
    - 核心思路：在标准MSE损失基础上增加面部区域的masked loss：$\mathcal{L}_{face} = \mathbb{E}[||(\epsilon - \epsilon_\theta) \odot m||]$，其中 $m$ 是从DensePose解析的面部分割mask。DensePose同时建立UV纹理图与目标图像像素之间的双射映射，隐式桥接两个焦点的外观对齐
    - 设计动机：面部只占图像很小比例，标准MSE损失中面部贡献被淹没，需要额外监督力度

### 损失函数 / 训练策略
总损失 $\mathcal{L}_{overall} = \mathcal{L}_{mse} + \mathcal{L}_{face}$。使用Adam优化器，学习率1e-5，两块A100 GPU训练60k迭代，batch size为12/GPU。推理时使用CFG，scale=3.5，MFCA中 $\lambda_i$ 分别为1和0.5。

## 实验关键数据

### 主实验

| 分辨率 | 指标 | MCLD | CFLD(CVPR24) | PIDM(CVPR23) | 提升 |
|--------|------|------|-------------|-------------|------|
| 256×176 | FID↓ | **6.693** | 6.804 | 6.36 | 较CFLD -0.111 |
| 256×176 | LPIPS↓ | **0.1482** | 0.1519 | 0.1678 | 较CFLD -0.0037 |
| 256×176 | SSIM↑ | **0.7511** | 0.7378 | 0.7312 | 较CFLD +0.0133 |
| 256×176 | PSNR↑ | **18.84** | 18.235 | - | 较CFLD +0.605 |
| 512×352 | FID↓ | **7.079** | 7.149 | 5.837 | 较CFLD -0.070 |
| 512×352 | SSIM↑ | **0.7557** | 0.7478 | 0.7419 | 较CFLD +0.0079 |

面部身份保持指标（256×176）：

| 指标 | MCLD | CFLD | PIDM | 说明 |
|------|------|------|------|------|
| FSref↑ | **0.301** | 0.243 | 0.270 | 与源图的面部相似度 |
| FStgt↑ | **0.413** | 0.363 | 0.394 | 与GT目标的面部相似度 |

### 消融实验

| 配置 | 条件 | 聚合方式 | FID↓ | LPIPS↓ | SSIM↑ | PSNR↑ |
|------|------|---------|------|--------|-------|-------|
| B1 | I | - | 6.427 | 0.1629 | 0.7371 | 18.18 |
| B3 | I+A+F | concat | 6.858 | 0.1536 | 0.7340 | 18.03 |
| B5 | I+A | MFCA | 6.723 | 0.1483 | 0.7499 | 18.72 |
| **Ours** | **I+A+F** | **MFCA** | **6.693** | **0.1482** | **0.7511** | **18.84** |

### 关键发现
- **MFCA vs 简单拼接**：拼接所有条件（B3）反而比仅用图像条件（B1）的FID更差（6.858 vs 6.427），说明简单拼接无法有效利用多条件。MFCA通过选择性注入解决了这个问题
- **面部条件的贡献**：B5（无面部）和Ours在整体指标上接近，但面部保持指标差异显著——面部只占图像小部分对整体指标影响有限，但对人类感知至关重要
- VAE重建的FID上限为7.967，因此LDM方法中更低的FID不一定代表更好的整体表现，LPIPS/SSIM/PSNR更具参考价值
- 模型支持灵活的外观编辑：只需修改部分条件区域即可实现身份/姿态/衣服的独立编辑，无需额外训练或mask

## 亮点与洞察
- **阶段选择性注入策略**非常巧妙：利用UNet编码器/解码器的天然功能分工来分配不同粒度的条件信息，比全程注入所有条件效率更高且效果更好
- **SMPL纹理图作为姿态无关外观表示**是关键trick：通过DensePose→SMPL映射完全解耦外观和姿态，使同一人的不同姿态图像共享完全相同的纹理条件
- 面部识别模型提取身份特征比CLIP更适合跨姿态面部表示，这个选择直接决定了身份保持的效果

## 局限与展望
- 对DensePose估计质量高度依赖，DensePose估计不准时纹理图会不完整（消融B4展示了这个问题）
- 复杂衣物纹理（如细条纹、图案）所有方法都难以准确重建
- FID在LDM方法中的指导意义有限（受VAE重建FID上限约束）
- 训练参数量较大（1717M），可考虑参数高效的条件注入方式
- 可扩展到视频人物动画任务，利用时序一致性进一步提升

## 相关工作与启发
- **vs CFLD**: CFLD强调语义理解的混合粒度注意力，但仍用整图作条件；MCLD通过焦点解耦+选择性注入更好地保持细节
- **vs PoCoLD**: PoCoLD建立3D姿态对应关系，但未解决LDM压缩退化问题；MCLD直接在条件层面缓解退化
- **vs InstantID**: MFCA的交叉注意力设计受InstantID启发，但扩展为多焦点+阶段选择性策略

## 评分
- 新颖性: ⭐⭐⭐⭐ 多焦点条件解耦+阶段选择性注入是有洞察力的设计
- 实验充分度: ⭐⭐⭐⭐ 全面的消融验证了每个组件的贡献，新增面部保持指标有说服力
- 写作质量: ⭐⭐⭐⭐ 动机分析透彻，VAE压缩退化的问题定义很清晰
- 价值: ⭐⭐⭐⭐ 多焦点条件聚合思路可推广到其他LDM退化场景

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis](exploring_sparse_moe_in_gans_for_text-conditioned_image_synthesis.md)
- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [\[CVPR 2025\] Learning Flow Fields in Attention for Controllable Person Image Generation](learning_flow_fields_in_attention_for_controllable_person_image_generation.md)
- [\[CVPR 2025\] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](enhancing_image_aesthetics_with_dual-conditioned_diffusion_models_guided_by_mult.md)
- [\[CVPR 2025\] Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis](noise_diffusion_for_enhancing_semantic_faithfulness_in_text-to-image_synthesis.md)

<!-- RELATED:END -->
