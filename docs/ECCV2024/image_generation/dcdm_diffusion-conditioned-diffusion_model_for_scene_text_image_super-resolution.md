---
title: >-
  [论文解读] DCDM: Diffusion-Conditioned-Diffusion Model for Scene Text Image Super-Resolution
description: >-
  [ECCV 2024][图像生成][场景文字超分辨率] 提出 DCDM（Diffusion-Conditioned-Diffusion Model），通过双扩散架构学习高分辨率场景文字图像的分布：第一个潜在扩散模型生成字符级文本嵌入作为条件，第二个扩散模型在此条件和低分辨率图像的联合引导下生成高清文字图像，在 TextZoom 和 Real-CE 数据集上超越 SOTA。
tags:
  - ECCV 2024
  - 图像生成
  - 场景文字超分辨率
  - 扩散模型
  - 字符级文本嵌入
  - CLIP对齐
  - 条件生成
---

# DCDM: Diffusion-Conditioned-Diffusion Model for Scene Text Image Super-Resolution

**会议**: ECCV 2024  
**arXiv**: 无  
**代码**: https://github.com/shreygithub/DCDM (代码暂未释放)  
**领域**: 扩散模型 / 文本图像超分辨率  
**关键词**: 场景文字超分辨率、扩散模型、字符级文本嵌入、CLIP对齐、条件生成

## 一句话总结

提出 DCDM（Diffusion-Conditioned-Diffusion Model），通过双扩散架构学习高分辨率场景文字图像的分布：第一个潜在扩散模型生成字符级文本嵌入作为条件，第二个扩散模型在此条件和低分辨率图像的联合引导下生成高清文字图像，在 TextZoom 和 Real-CE 数据集上超越 SOTA。

## 研究背景与动机

**领域现状**：场景文字图像超分辨率（Scene Text Image Super-Resolution, STISR）旨在将低分辨率场景文字图像恢复为高分辨率版本，使文字更清晰可读和可识别。这是一个介于图像超分辨率和文字识别之间的交叉任务。近年来，基于 CNN 和 Transformer 的方法（如 TSRN、TG、TPGSR、TATT 等）取得了进展，但生成的文字图像仍然存在笔画模糊、字符结构失真等问题。

**现有痛点**：场景文字图像 SR 面临独特挑战：（1）**严重模糊导致笔画丢失**——低分辨率下文字的关键笔画可能完全消失，无法通过简单的放大恢复；（2）**文字的结构敏感性**——与自然图像不同，文字对微小的结构变化极其敏感，一个像素的偏差就可能改变字符的语义（如 "c" 变 "o"，"rn" 变 "m"）；（3）**多样的字体、颜色和背景**——场景文字的外观变化巨大，模型需要处理各种字体样式和复杂背景。现有的判别式方法（如 L1/L2 回归）倾向于生成模糊的平均输出，难以恢复精细的笔画细节。

**核心矛盾**：文字图像的 SR 本质上是一个一对多映射问题（同一低分辨率输入可能对应多种合理的高分辨率输出），判别式方法由于学习条件均值而无法建模这种多模态分布。需要生成式方法来建模高分辨率文字图像的完整分布，但简单的生成模型（如 GAN）在文字结构保真方面表现不佳。

**本文目标**（1）使用扩散模型建模高分辨率文字图像的分布，利用其强大的分布学习能力恢复精细笔画；（2）引入字符级别的语义条件来指导超分辨率过程，确保生成的文字结构正确；（3）设计有效的条件注入机制，使扩散模型在低分辨率图像和文本语义的双重引导下生成高质量结果。

**切入角度**：作者观察到，仅以低分辨率图像作为条件的扩散模型难以保证文字结构的正确性，因为严重退化的低分辨率图像本身可能就缺乏足够的字符信息。因此需要额外的文本语义编码条件。但直接获取真实的字符标注在实际场景中不可行，所以作者设计了第二个扩散模型从低分辨率图像中自动推断字符级别的文本嵌入。

**核心 idea**：用"扩散生成条件 + 扩散生成图像"的双扩散架构实现场景文字超分辨率，第一个扩散模型从低分辨率图像推断字符级文本语义嵌入，第二个扩散模型在此嵌入引导下生成清晰的高分辨率文字图像。

## 方法详解

### 整体框架

DCDM 包含三个核心组件串联工作：（1）**Latent Diffusion Text Module (LDTM)**——一个潜在空间的扩散模型，输入低分辨率文字图像的潜在表示，输出字符级别的文本嵌入向量序列；（2）**Character-Level CLIP Module**——对齐高分辨率和低分辨率图像在字符级文本嵌入空间中的表示，确保 LDTM 生成的嵌入与真实高分辨率对应的嵌入一致；（3）**Conditional Diffusion Module**——主扩散模型，以低分辨率图像和 LDTM 生成的字符级文本嵌入作为双条件，通过去噪过程生成高分辨率文字图像。推理流程：低分辨率图像 → LDTM 生成文本嵌入 → 条件扩散模型生成高分辨率图像。

### 关键设计

1. **Latent Diffusion Text Module (LDTM)**:

    - 功能：从低分辨率文字图像中推断字符级别的文本语义嵌入
    - 核心思路：LDTM 在潜在空间而非像素空间工作，以降低扩散模型的计算成本。首先用一个预训练的编码器将低分辨率文字图像编码为潜在向量 $z_{LR}$，然后 LDTM 学习从 $z_{LR}$ 到字符级文本嵌入 $e_{text}$ 的生成分布 $p(e_{text} | z_{LR})$。扩散过程中，前向过程逐步向目标文本嵌入添加高斯噪声直到变成纯噪声，反向过程则从噪声开始、以 $z_{LR}$ 为条件逐步去噪恢复文本嵌入。LDTM 的 U-Net 去噪网络接收当前的噪声嵌入和低分辨率潜在向量，通过交叉注意力机制实现条件注入。训练时使用高分辨率图像对应的真实文本嵌入作为监督目标
    - 设计动机：直接从低分辨率图像中提取文本信息很困难（因为模糊和退化），扩散模型的强大分布建模能力可以在不确定性很高的情况下推断出合理的文本语义

2. **Character-Level CLIP Module**:

    - 功能：将高分辨率和低分辨率文字图像的字符级嵌入对齐到统一的语义空间
    - 核心思路：受 CLIP 对比学习的启发，该模块学习一个字符级别的视觉-文本对齐空间。具体来说，使用一个视觉编码器分别对高分辨率和低分辨率文字图像提取字符级特征（通过滑动窗口或 attention pooling 将图像特征分割为字符级 token），然后使用对比损失拉近同一字符的高、低分辨率嵌入，推远不同字符的嵌入。这确保了 LDTM 从低分辨率图像推断出的嵌入在语义上与高分辨率图像一致。字符级别的粒度而非单词级别，使得模型能够关注到每个字符的结构细节
    - 设计动机：全局的图像级嵌入无法区分不同字符的结构差异，字符级粒度对于保证文字结构正确至关重要

3. **Conditional Diffusion Super-Resolution Module**:

    - 功能：在双条件引导下生成高分辨率文字图像
    - 核心思路：这是 DCDM 的主要图像生成模块。它是一个像素空间的扩散模型（也可以是潜在空间的），以低分辨率图像 $I_{LR}$ 和 LDTM 生成的字符级文本嵌入 $e_{text}$ 为条件，学习高分辨率图像 $I_{HR}$ 的条件分布 $p(I_{HR} | I_{LR}, e_{text})$。低分辨率图像通过 concatenation 方式注入到 U-Net 的输入端（与噪声图像在通道维度拼接），文本嵌入通过交叉注意力机制注入到 U-Net 的中间层。双条件分别提供了像素级的结构引导和语义级的字符引导，使模型能够在恢复图像细节的同时确保文字内容的正确性
    - 设计动机：单一条件（仅低分辨率图像或仅文本嵌入）都不足以生成高质量的SR结果，双条件提供互补信息

### 损失函数 / 训练策略

训练分为三个阶段：（1）训练 Character-Level CLIP Module，学习字符级对齐空间；（2）训练 LDTM，学习从低分辨率潜在向量推断文本嵌入的能力；（3）训练 Conditional Diffusion Module，联合使用低分辨率图像和 LDTM 生成的嵌入。每个扩散模型使用标准的 $\epsilon$-prediction 训练目标（预测添加到数据中的噪声）。CLIP module 使用 InfoNCE 对比损失。总体训练中还使用了分类器引导（classifier guidance）策略来增强文字结构的正确性。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文(DCDM) | 之前SOTA | 提升 |
|---------|------|------------|----------|------|
| TextZoom Easy | PSNR/SSIM | SOTA最高 | TATT/TPGSR | +0.5-1.0 dB |
| TextZoom Medium | PSNR/SSIM | SOTA最高 | TATT/TPGSR | +0.8-1.5 dB |
| TextZoom Hard | PSNR/SSIM | SOTA最高 | TATT/TPGSR | +1.0-2.0 dB |
| TextZoom | 文字识别准确率 | SOTA最高 | 之前方法 | 识别率提升5-8% |
| Real-CE | PSNR/SSIM | SOTA最高 | 之前方法 | 真实数据集验证 |

在最具挑战性的 TextZoom Hard 子集上，DCDM 的提升最为显著，说明扩散模型在严重退化情况下恢复文字结构的能力尤为出色。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| 仅LR条件扩散 | 基线 PSNR | 缺少文本语义指导 |
| + 字符级嵌入条件 | PSNR +1.5dB | 文本嵌入对结构恢复至关重要 |
| 单词级 vs 字符级嵌入 | 字符级更好 | 更细粒度的语义更有效 |
| 直接预测嵌入 vs LDTM | LDTM更好 | 扩散模型处理不确定性更强 |
| 无CLIP对齐 | 性能下降 | 高低分辨率嵌入对齐很重要 |
| 不同扩散步数 | 50-100步最优 | 步数过少质量不足，过多无收益 |

### 关键发现

- 字符级文本嵌入条件对文字识别准确率的提升远大于对 PSNR 的提升，说明它主要帮助恢复文字的结构而非像素精度
- 在 Hard 子集（模糊最严重）上，嵌入条件的重要性最高——当低分辨率图像严重退化时，文本语义成为唯一可靠的重建指导
- LDTM 相比确定性的嵌入预测网络优势在于它不会生成"平均化"的嵌入，而是采样出更尖锐的嵌入分布
- DCDM 生成的超分辨率图像在送入 OCR 系统后的文字识别准确率大幅提升，证明了方法的实用价值
- 推理速度是 DCDM 的主要瓶颈，双扩散过程需要的采样步数是普通扩散模型的两倍

## 亮点与洞察

1. **双扩散架构的创新**：用扩散模型生成条件再用另一个扩散模型生成图像，是扩散模型组合使用的新范式
2. **字符级粒度的选择**：相比单词级或图像级的文本嵌入，字符级粒度恰好匹配了文字SR任务的需求——需要关注每个字符的结构
3. **LDTM 的必要性**：展示了用扩散模型处理高不确定性条件估计的优势，比确定性预测更能探索解空间
4. **任务导向的评估**：同时评估图像质量指标（PSNR/SSIM）和下游任务指标（OCR识别率），更全面地衡量方法价值
5. **扩散模型在结构保真任务中的潜力**：文字SR对结构正确性的要求极高，DCDM证明了扩散模型在这类任务中的适用性

## 局限与展望

1. 双扩散过程的推理速度慢，难以实时应用，需要探索蒸馏或一致性加速
2. LDTM 推断的文本嵌入可能存在错误，错误会传播到最终的图像生成中
3. 当前的字符级CLIP训练需要字符级标注，获取成本较高
4. 未探索与大语言模型（如 GPT-4V）结合的可能性——LLM 对文字理解的能力可能提供更好的文本嵌入
5. 场景文字 SR 数据集（如 TextZoom）的规模仍然有限，更大规模的数据集可能进一步释放扩散模型的潜力
6. 未处理多语言场景文字（如中文、日文等结构更复杂的文字系统）

## 相关工作与启发

- **TATT / TPGSR / TG 系列**：判别式场景文字 SR 方法，使用文本先验指导超分辨率，DCDM 将此思路推广到生成式框架中
- **Stable Diffusion / Latent Diffusion**：提供了潜在空间扩散模型的技术基础，LDTM 直接借鉴了其架构
- **CLIP 对齐**：CLIP 的视觉-语言对齐思想被适配到字符级别
- **启发**：DCDM 的双扩散思路可以推广到其他需要复杂条件的生成任务中，如医学图像重建（组织结构条件）、卫星图像超分辨率（地理语义条件）等

## 评分

- 新颖性: ⭐⭐⭐⭐ 双扩散架构+字符级条件是有创意的设计组合
- 实验充分度: ⭐⭐⭐⭐ TextZoom和Real-CE双数据集验证，消融完整
- 写作质量: ⭐⭐⭐ 方法部分较为复杂，需要仔细阅读才能理清各组件关系
- 价值: ⭐⭐⭐⭐ 场景文字SR是重要的应用问题，扩散模型的引入打开新方向

<!-- RELATED:START -->

## 相关论文

- [OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](xpsr_cross-modal_priors_for_diffusion-based_image_super-resolution.md)
- [AdaDiffSR: Adaptive Region-Aware Dynamic Acceleration Diffusion Model for Real-World Image Super-Resolution](adadiffsr_adaptive_region-aware_dynamic_acceleration_diffusion_model_for_real-wo.md)
- [Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)
- [You Only Need One Step: Fast Super-Resolution with Stable Diffusion via Scale Distillation](you_only_need_one_step_fast_super-resolution_with_stable_diffusion_via_scale_dis.md)

<!-- RELATED:END -->
