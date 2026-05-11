---
title: >-
  [论文解读] MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed-Precision Quantization
description: >-
  [ECCV 2024][图像生成][扩散模型量化] 针对少步扩散模型（如SDXL-turbo 1-step）比多步模型更难量化的问题，提出MixDQ混合精度量化方法，包含BOS-aware文本嵌入量化、指标解耦敏感度分析和整数规划比特分配，在W4A8下仅增加0.5 FID，实现3倍模型压缩和1.5倍加速。
tags:
  - ECCV 2024
  - 图像生成
  - 扩散模型量化
  - 混合精度
  - 文本到图像生成
  - BOS-aware量化
  - 指标解耦
---

# MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed-Precision Quantization

**会议**: ECCV 2024  
**arXiv**: [2405.17873](https://arxiv.org/abs/2405.17873)  
**代码**: 有（将发布）  
**领域**: 图像生成  
**关键词**: 扩散模型量化, 混合精度, 文本到图像生成, BOS-aware量化, 指标解耦

## 一句话总结
针对少步扩散模型（如SDXL-turbo 1-step）比多步模型更难量化的问题，提出MixDQ混合精度量化方法，包含BOS-aware文本嵌入量化、指标解耦敏感度分析和整数规划比特分配，在W4A8下仅增加0.5 FID，实现3倍模型压缩和1.5倍加速。

## 研究背景与动机
1. **领域现状**：少步扩散模型（如SDXL-turbo）将推理步数降到1-4步，大幅减少计算量，但模型内存消耗仍达5-10GB，限制了移动端部署。PTQ是有效的压缩方式。
2. **现有痛点**：(1) 少步模型比多步模型对量化更敏感——多步模型后续去噪步骤可补偿量化误差，1步模型无此容错机制；(2) 现有方法只关注图像质量保持，忽略了量化对image-text alignment的破坏；(3) 层敏感度存在"长尾"分布，均匀量化被高敏感层"拖累"。
3. **核心矛盾**：少步模型中极少数超敏感层（主要是cross-attention的to_k/to_v）决定了整体量化质量，而这些层的敏感来源是文本嵌入中BOS token的异常大值（828 vs 10-15）。同时，现有敏感度度量（SQNR）会过度强调内容变化，导致质量相关层被牺牲。
4. **本文要解决什么？** (1) 处理文本嵌入中BOS token的outlier问题；(2) 解耦量化对图像质量和内容的不同影响来做敏感度分析；(3) 在给定资源预算下找到最优混合精度配置。
5. **切入角度**：深入分析层敏感度分布和数据分布特性，发现问题根源并逐一解决。
6. **核心idea一句话**：通过BOS-aware量化处理outlier、指标解耦分离质量/内容敏感度、整数规划求解最优比特配置，实现少步扩散模型的几乎无损量化。

## 方法详解

### 整体框架
MixDQ由三步组成：(1) BOS-aware量化处理高敏感的文本嵌入层；(2) 指标解耦敏感度分析分别评估层对图像质量和内容的影响；(3) 整数规划根据敏感度分配最优比特宽度。整个流程应用于UNet中的所有linear和conv层。

### 关键设计

1. **BOS-aware文本嵌入量化**：
    - 做什么：消除CLIP文本嵌入中BOS token造成的量化瓶颈
    - 核心思路：发现第一个token（BOS）的最大值为823.5，其余token仅10-15。由于BOS特征对所有prompt固定不变，预计算BOS token的浮点输出并跳过量化，与其余量化token拼接
    - 设计动机：量化含BOS的tensor会导致大部分值被压缩到0附近，丢失关键文本信息，这是量化后image-text alignment崩溃的核心原因
    - 与之前方法的区别：FP8等非均匀量化仍有残余误差，BOS-aware直接跳过量化，彻底消除问题

2. **指标解耦敏感度分析**：
    - 做什么：将模型层分为内容相关（cross-attention + FFN）和质量相关（self-attention + conv），分别用不同指标评估敏感度
    - 核心思路：内容相关层用SSIM衡量结构变化，质量相关层用SQNR衡量误差。$\text{SSIM}(x,y) = l(x,y)^\alpha \cdot c(x,y)^\beta \cdot s(x,y)^\gamma$
    - 设计动机：单一SQNR指标会过度惩罚内容变化（-3.51dB）而忽视质量退化（-0.26dB），导致content层占用过多高比特预算，quality层被迫降到极低比特

3. **整数规划比特分配**：
    - 做什么：在给定资源预算下，为每层选择2/4/8-bit的最优配置
    - 核心思路：$\text{argmax}_{c_{i,b}} \sum_i \sum_b c_{i,b} \cdot S_{i,b}$ s.t. $\sum c_{i,b} \cdot M_{i,b} \leq B$
    - 设计动机：naive排序方法不能获得全局最优，整数规划可在秒级找到Pareto前沿上的最优点

### 损失函数 / 训练策略
采用训练后量化（PTQ），无需额外训练。使用1024个COCO prompt做校准。对于激活量化，保留1%最敏感层为FP16。分别对weight和activation、content层和quality层做4组独立的整数规划。

## 实验关键数据

### 主实验

| 模型 | 方法 | W/A | FID↓ | CLIP Score↑ | ImageReward↑ | 存储压缩 |
|------|------|-----|------|-------------|-------------|---------|
| SDXL-turbo 1step | FP16 | 16/16 | 17.15 | 0.2722 | 0.8631 | - |
| SDXL-turbo 1step | Q-Diffusion | 8/8 | 76.18 | 0.1772 | -1.3112 | 2× |
| SDXL-turbo 1step | Q-Diffusion | 4/8 | 118.93 | 0.1662 | -1.6353 | 4× |
| SDXL-turbo 1step | **MixDQ** | 8/8 | **17.03** | **0.2703** | **0.8415** | 2× |
| SDXL-turbo 1step | **MixDQ** | 4/8 | **17.68** | **0.2698** | **0.7822** | 4× |

### 消融实验

| 配置 | FID↓ | CLIP↑ | IR↑ | 说明 |
|------|------|-------|-----|------|
| Naive W8A8 | 103.96 | 0.1478 | -1.72 | 基线严重退化 |
| + BOS-aware | 31.65 | 0.2652 | - | 内容恢复，FID大幅下降 |
| + Mixed-Precision (SQNR only) | 37.35 | 0.2624 | - | SQNR偏向content层，质量变差 |
| + Metric-Decouple + MP | **17.03** | **0.2703** | 0.84 | 接近FP16 |

### 关键发现
- BOS-aware量化贡献最大（FID从103.96降到31.65），根本解决了文本alignment丢失问题
- 不做指标解耦的混合精度反而让质量更差（FID从31.65升到37.35），验证了SQNR偏向content的问题
- W4A8下MixDQ仅比FP16增加0.53 FID，而Q-Diffusion的FID超过100
- 实际GPU上W8A8实现1.52×加速，W4A16实现3.03×内存压缩

## 亮点与洞察
- **BOS-aware量化的洞察极为精妙**：发现BOS token对所有prompt固定不变，因此可以预计算跳过量化。这不仅解决了扩散模型问题，也对所有Transformer-based文本编码器的量化有借鉴意义（"attention sink"现象）。
- **指标解耦的思想具有通用性**：任何需要同时保持多个指标的生成任务压缩都可以采用类似的分组+独立评估策略。
- **实用性极强**：整个方法无需任何再训练，仅需几分钟获取Pareto前沿，且完全兼容现有硬件INT8内核。

## 局限性 / 可改进方向
- 仅使用INT8 GPU内核，INT4潜力未充分发掘
- 未结合Adaround或QAT等更先进量化技术
- conv_in和conv_out层仍然高敏感，可以为它们设计专门的量化方案
- 当前只在SDXL-turbo和LCM-LoRA上验证，可扩展到更多架构

## 相关工作与启发
- **vs Q-Diffusion**：Q-Diffusion对多步模型有效但在1步模型上完全失败（FID 76.18），因为多步模型有迭代去噪容错。MixDQ通过BOS-aware+指标解耦彻底解决此挑战。
- **vs PTQD**：PTQD的噪声校正在1步确定性采样中不适用（线性相关仅0.59），反而更差。
- **vs Yang et al. (SQNR-based MP)**：纯SQNR混合精度仍有30+的FID增加，证实了指标解耦的必要性。

## 补充说明
- 与NF4/FP4等非均匀量化的对比：FP8在BOS问题上仍有残余误差，且需要特殊硬件支持
- 时间嵌入在少步模型中不敏感（与多步模型不同），因为蒸馏使网络能从任意时间步去噪
- MixDQ可与QAT结合，框架具有良好的扩展性
- 在W4A16下也能获得几乎无损的结果（FID 17.23 vs FP16的17.15）
- Pareto前沿上有400种配置可选，整数规划在秒级完成
- T5编码器中虽然首token不是outlier但channel imbalance问题依然存在

## 评分
- 新颖性: ⭐⭐⭐⭐ BOS-aware量化和指标解耦是原创性贡献，问题分析深入
- 实验充分度: ⭐⭐⭐⭐⭐ 完整的消融、Pareto分析、硬件实测、多指标多模型评估
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，分析逐层递进
- 价值: ⭐⭐⭐⭐ 实用性极强，对扩散模型部署有直接帮助

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization](mixdq_memoryefficient_fewstep_texttoimage_diffusion_models_w.md)
- [\[ECCV 2024\] Memory-Efficient Fine-Tuning for Quantized Diffusion Model](memory-efficient_fine-tuning_for_quantized_diffusion_model.md)
- [\[ECCV 2024\] Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] LCM-Lookahead for Encoder-based Text-to-Image Personalization](lcmlookahead_for_encoderbased_texttoimage_personalization.md)

</div>

<!-- RELATED:END -->
