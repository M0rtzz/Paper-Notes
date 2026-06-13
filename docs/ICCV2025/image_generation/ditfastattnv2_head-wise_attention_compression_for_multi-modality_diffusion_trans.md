---
title: >-
  [论文解读] DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers
description: >-
  [ICCV 2025][图像生成][Transformer] 针对多模态扩散Transformer（MMDiT）提出DiTFastAttnV2，通过Head-wise Arrow Attention和Head-wise Caching机制实现细粒度的注意力压缩，在2K图像生成中减少68%注意力FLOPs并实现1.5倍端到端加速，且不损失视觉质量。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "Transformer"
  - "注意力压缩"
  - "MMDiT"
  - "稀疏注意力"
  - "推理加速"
---

# DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers

**会议**: ICCV 2025  
**arXiv**: [2503.22796](https://arxiv.org/abs/2503.22796)  
**代码**: 无  
**领域**: 图像生成 / 扩散模型加速  
**关键词**: 扩散Transformer, 注意力压缩, MMDiT, 稀疏注意力, 推理加速

## 一句话总结

针对多模态扩散Transformer（MMDiT）提出DiTFastAttnV2，通过Head-wise Arrow Attention和Head-wise Caching机制实现细粒度的注意力压缩，在2K图像生成中减少68%注意力FLOPs并实现1.5倍端到端加速，且不损失视觉质量。

## 研究背景与动机

MMDiT（如SD3、FLUX）是当前主流的文生图架构，其将视觉和文本token拼接后做联合自注意力，但注意力计算是推理的主要瓶颈。现有加速方法（如DiTFastAttn）存在三个关键局限：

**跨模态注意力模式复杂性**：MMDiT中视觉token呈对角局部性，文本token交互高度语义依赖，统一的滑动窗口注意力无法捕捉这种差异，强行应用会截断文本信息

**Head间冗余异质性**：同一层的不同注意力头行为差异巨大（有的接近全局注意力，有的高度局部化），层级统一的缓存或稀疏策略会丢失关键信息

**搜索成本过高**：DiTFastAttn搜索压缩方案需要超过10小时（50步2K图像生成FLUX），扩展到head级别则超过200小时

## 方法详解

### 整体框架

DiTFastAttnV2是一个训练后压缩框架，包含三个组件：Head-wise Arrow Attention（处理空间冗余异质性）、Head-wise Caching（处理时间步冗余异质性）、高效融合核（实现实际加速），以及一个高效的压缩方案搜索算法。

### 关键设计

1. **Head-wise Arrow Attention**：

    - 将联合注意力图分为四个区域：视觉-视觉、视觉-文本、文本-视觉、文本-文本
    - 对视觉-视觉区域应用局部注意力（仅保留对角线附近的注意力分数），丢弃远距离token
    - 对涉及文本token的三个区域保留完整注意力，不做任何压缩
    - 这种模式形似箭头，故命名arrow attention
    - 每个注意力头可独立选择全注意力或arrow attention（混合注意力设计）
    - 设计动机：视觉token的对角局部性跨提示一致，文本交互高度语义依赖不可压缩

2. **Head-wise Caching**：

    - 分析发现同一层不同head在相邻时间步的相似度差异显著
    - 对相似度高的head跳过当前时间步的注意力计算，直接复用上一步的缓存输出
    - 每个head可独立决定是否使用缓存
    - 设计动机：利用时间步冗余但需按head细粒度处理，避免丢失快速变化head的关键信息

3. **高效融合核（Fused Kernel）**：

    - 集成arrow attention和caching，每个head可独立选择三种模式：全注意力、计算跳过（缓存复用）、arrow attention（指定窗口大小）
    - 基于FlashAttention2实现，采用block-sparse模式确保每个计算块是密集的，最小化不规则内存访问开销
    - 将混合block转换为密集block以减少内存访问开销

### 损失函数 / 训练策略

**高效压缩方案搜索**：

- **单层RSE指标**：用单层相对平方误差（而非最终输出MSE）衡量每种压缩方法的影响，将校准成本从$T \times L \times M \times H$次完整推理降低到$T \times M$次
  $$\mathcal{I}(m) = \frac{\sum(y_m - \bar{y}_o)^2}{\sum(y_o - \bar{y}_o)^2}$$

- **Head-wise压缩方案优化**：将每层每时间步的压缩配置建模为整数优化问题，目标是在RSE预算$\delta$约束下最小化延迟
- **Head约束系数c**：引入约束$\mathcal{I}(h,m) \leq \frac{c}{n}\delta$防止单个head承担过多压缩预算，默认c=1.5
- 采用逐时间步、逐层的渐进式更新搜索策略

## 实验关键数据

### 主实验 - SD3和FLUX生成质量（表格）

| 模型 | 分辨率 | 阈值δ | 注意力稀疏度 | LPIPS↓ | SSIM↑ | HPSv2↑ | CLIP↑ |
|------|--------|-------|------------|--------|-------|--------|-------|
| SD3 | 1024 | Original | 0 | - | - | 0.2926 | 0.3254 |
| SD3 | 1024 | δ=0.2 | 0.41 | 0.182 | 0.716 | 0.2933 | 0.3251 |
| SD3 | 1024 | δ=0.6 | 0.63 | 0.266 | 0.616 | 0.2933 | 0.3246 |
| FLUX | 2048 | Original | 0 | - | - | 0.2862 | 0.3169 |
| FLUX | 2048 | δ=0.2 | 0.43 | 0.242 | 0.646 | 0.2883 | 0.3164 |
| FLUX | 2048 | δ=1.0 | **0.68** | 0.393 | 0.497 | 0.2852 | 0.3163 |

### 消融实验 - 方法组合与约束系数（表格）

| 方法集 | 注意力稀疏度 | LPIPS↓ | SSIM↑ | HPSv2↑ |
|--------|------------|--------|-------|--------|
| Original | 0 | - | - | 0.2926 |
| AA only | 0.30 | 0.275 | 0.608 | 0.2943 |
| **AA + OC** | **0.55** | **0.238** | **0.644** | **0.2935** |
| + CFG Sharing | 0.54 | 0.249 | 0.649 | 0.2913 |
| + Residual Sharing | 0.56 | 0.196 | 0.704 | 0.2906 |

| 约束系数c | 稀疏度 | LPIPS↓ | SSIM↑ |
|-----------|--------|--------|-------|
| 无约束 | 0.50 | 0.249 | 0.640 |
| c=1 | 0.55 | 0.240 | 0.641 |
| **c=1.5** | **0.55** | **0.238** | **0.644** |
| c=2 | 0.55 | 0.253 | 0.627 |

### 关键发现

- **显著加速**：FLUX 2K图像生成实现1.5倍端到端加速，最高减少68%注意力FLOPs
- **质量无损**：δ=0.2/0.6时HPSv2和CLIP分数与原始模型相当甚至更高
- **大幅超越前作**：在SD3上同等注意力稀疏度下，DiTFastAttnV2在所有指标上优于DiTFastAttn及其变体
- **搜索效率提升**：2K图像生成的方案搜索从10小时降至15分钟
- **CFG Sharing对MMDiT无效**：MMDiT设计消除了CFG的需要，CFG共享带来的冗余压缩空间微乎其微
- **融合核性能优异**：在75%稀疏度时达到3.55倍加速，接近甚至超过理论上限

## 亮点与洞察

1. **从head粒度重新审视注意力压缩**：揭示了MMDiT中注意力头的高度异质性，并据此设计细粒度策略
2. **Arrow Attention设计巧妙**：精准捕捉了视觉token的局部性和文本token的全局性差异
3. **搜索效率提升两个数量级**：单层RSE指标+head级优化使分钟级搜索成为可能
4. **实际部署价值高**：通过融合核实现了真实的1.5倍加速，而非仅停留在FLOPs减少

## 局限与展望

- 仅在SD3和FLUX两个模型上验证，对其他MMDiT架构（如CogVideoX、HunyuanVideo等视频模型）的适用性未验证
- Arrow Attention的窗口大小是通过搜索确定的，缺乏自适应机制
- 当前实现基于A100 GPU，在其他硬件上的加速比可能不同
- 未探索与量化、蒸馏等其他压缩方法的组合效果
- 高压缩率（δ=1.0）下细节和背景会发生变化，可能不适合要求严格一致性的场景

## 相关工作与启发

- DiTFastAttn的直接改进版，从层级扩展到head级别，解决了MMDiT适配问题
- Arrow Attention启发自LLM中的Attention Sink和稀疏注意力研究
- 单层指标校准思路可推广到其他需要搜索压缩配置的场景
- 头级别异质性的发现可能对ViT剪枝/蒸馏研究也有参考价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ Head-wise粒度的arrow attention + caching组合新颖，搜索效率提升显著
- **实验充分度**: ⭐⭐⭐⭐ 多指标、多阈值、消融详尽，但仅两个模型略显不足
- **写作质量**: ⭐⭐⭐⭐ 分析深入透彻，图示清晰，算法伪代码完整
- **价值**: ⭐⭐⭐⭐⭐ 解决了MMDiT推理加速的实际痛点，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Head Pursuit: Probing Attention Specialization in Multimodal Transformers](../../NeurIPS2025/image_generation/head_pursuit_probing_attention_specialization_in_multimodal.md)
- [\[ICCV 2025\] M2SFormer: Multi-Spectral and Multi-Scale Attention with Edge-Aware Difficulty Guidance for Image Forgery Localization](m2sformer_multi-spectral_and_multi-scale_attention_with_edge-aware_difficulty_gu.md)
- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [\[ICCV 2025\] FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers](freecus_free_lunch_subject-driven_customization_in_diffusion_transformers.md)
- [\[ICCV 2025\] UniCombine: Unified Multi-Conditional Combination with Diffusion Transformer](unicombine_unified_multi-conditional_combination_with_diffusion_transformer.md)

</div>

<!-- RELATED:END -->
