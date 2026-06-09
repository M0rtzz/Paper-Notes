---
title: >-
  [论文解读] DiC: Rethinking Conv3x3 Designs in Diffusion Models
description: >-
  [CVPR 2025][图像生成][纯卷积扩散模型] 本文重新审视3x3卷积在扩散模型中的潜力，通过一系列架构改进（沙漏U-Net+稀疏跳连）和条件注入改进（阶段特定嵌入+中间块注入+条件门控），构建了纯3x3卷积的扩散模型DiC，在ImageNet生成上超越同规模DiT且推理速度显著更快。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "纯卷积扩散模型"
  - "3x3卷积"
  - "U-Net架构"
  - "稀疏跳连"
  - "条件注入"
---

# DiC: Rethinking Conv3x3 Designs in Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2501.00603](https://arxiv.org/abs/2501.00603)  
**代码**: [GitHub](https://github.com/YuchuanTian/DiC)  
**领域**: 扩散模型 / 模型架构  
**关键词**: 纯卷积扩散模型, 3x3卷积, U-Net架构, 稀疏跳连, 条件注入

## 一句话总结
本文重新审视3x3卷积在扩散模型中的潜力，通过一系列架构改进（沙漏U-Net+稀疏跳连）和条件注入改进（阶段特定嵌入+中间块注入+条件门控），构建了纯3x3卷积的扩散模型DiC，在ImageNet生成上超越同规模DiT且推理速度显著更快。

## 研究背景与动机

**领域现状**：扩散模型架构从CNN-Attention混合（如ADM）演进为全Transformer（如DiT、PixArt），后者展现出优秀的可扩展性和性能，但self-attention的计算开销大。

**现有痛点**：Transformer扩散模型推理慢，不适合实时或资源受限场景。现有加速方案（如下采样token、线性注意力、SSM替代）仍在注意力范式内，或latency不理想。

**核心矛盾**：3x3卷积极快（受益于Winograd加速，硬件友好），但感受野天然受限。直接将纯3x3卷积放入现有可扩展架构（isotropic DiT），效果远不如Transformer。

**本文目标**：探索纯3x3卷积能否通过精心设计达到与Transformer扩散模型竞争的生成质量，同时保持速度优势。

**切入角度**：3x3卷积的感受野问题可以通过编解码器下采样自然扩大——在下采样后的特征图上，3×3覆盖原始图的6×6甚至12×12。

**核心 idea**：U-Net沙漏架构（扩大感受野）+ 稀疏跳连（减少冗余）+ 阶段特定条件嵌入（适配不同分辨率特征空间）。

## 方法详解

### 整体框架
基本块为两层3x3卷积+残差连接（移除了传统UNet块中的self-attention）。采用编解码器沙漏结构，通过下采样/上采样扩大感受野。引入稀疏跳连和多项条件改进。

### 关键设计

1. **沙漏架构 + 稀疏跳连**:

    - 功能：为纯3x3卷积提供足够的感受野，同时保持可扩展性
    - 核心思路：实验对比isotropic（FID 29.31）、isotropic+skip（15.07）、U-Net hourglass（14.65）、U-Net+sparse skip（11.49）。沙漏架构通过下采样天然扩大感受野。传统block-wise dense skip在大量堆叠卷积时开销过大，改为每隔几个block才设一个skip
    - 设计动机：纯isotropic架构中3x3卷积每层只扩展1像素感受野，需要极深网络才能获得全局感受野。沙漏的层级结构高效解决此问题

2. **阶段特定条件嵌入**:

    - 功能：适配编解码器各阶段不同的特征空间
    - 核心思路：传统扩散模型各阶段共享同一组条件嵌入表。在沙漏架构中，各阶段通道维度不同，代表不同层次的特征。DiC为每个阶段训练独立的嵌入表（维度匹配该阶段通道数），仅增加2%参数
    - 设计动机：encoder底层处理高级语义、顶层处理局部细节，用同一组嵌入无法最优适配所有层级

3. **中间块条件注入 + 条件门控**:

    - 功能：优化条件信号的注入位置和方式
    - 核心思路：将条件注入到每个基本块的第二层卷积（中间位置），而非块开头的LayerNorm。从DiT借鉴AdaLN的门控向量，在通道维度上缩放特征。全部激活函数从SiLU换为GELU
    - 设计动机：实验验证中间注入优于开头注入；门控提供更细粒度的条件控制

### 损失函数 / 训练策略
标准扩散去噪损失（与DiT相同超参数设置以确保公平比较）。

## 实验关键数据

### 主实验（ImageNet 256×256，条件生成）

| 模型 | 参数量 | FLOPs | FID↓ | IS↑ | 吞吐量 |
|------|--------|-------|------|-----|--------|
| DiT-XL | 675M | 119G | 2.27 | 278 | 基准 |
| **DiC-XL** | **708M** | **119G** | **2.10** | **286** | **~2×更快** |

### 消融实验（200K iterations）

| 配置 | FID↓ | 说明 |
|------|------|------|
| Isotropic Conv3x3 | 29.31 | 极差，感受野不足 |
| U-Net Hourglass | 14.65 | 大幅改善 |
| + Sparse Skip | 11.49 | 继续改善 |
| + Stage-Spec. Emb. | 10.07 | 条件改进 |
| + Mid-Block Injection | 8.80 | 注入位置优化 |
| + Gating | 6.54 | 门控显著提升 |
| + GELU (DiC) | **6.26** | 最终模型 |
| DiT-XL同条件 | 12.96 | DiC大幅领先 |

### 关键发现
- 纯3x3卷积扩散模型通过架构和条件改进可超越同规模DiT
- 沙漏架构对3x3卷积至关重要——isotropic架构完全不可行
- 稀疏跳连优于密集跳连，减少了冗余级联开销
- 阶段特定嵌入仅增加2%参数但效果显著
- DiC在推理吞吐量上有约2倍速度优势（得益于Winograd加速和高并行度）

## 亮点与洞察
- 将被"全Transformer趋势"遗忘的3x3卷积重新带回扩散模型前沿，且性能超越Transformer，具有很强的反直觉启发性
- 系统性的roadmap展示了每一步改进的贡献，分析非常清晰
- "感受野是3x3卷积的核心瓶颈"→"用沙漏结构通过下采样自然扩大"的推理链简洁有力

## 局限与展望
- 目前在ImageNet 256×256上验证，更高分辨率和文本条件生成有待证明
- 纯卷积模型缺乏全局attention对长距离依赖的建模，某些需要全局一致性的任务可能受限
- 稀疏跳连的间隔是超参，不同规模可能需调整
- 可考虑与少量attention层混合，在保持速度优势的同时增强全局建模能力

## 相关工作与启发
- **vs DiT**: DiT用full attention实现可扩展性；DiC证明3x3卷积+适当架构设计同样可扩展且更快
- **vs U-ViT**: U-ViT在isotropic架构上加skip，仍用attention；DiC用沙漏+稀疏skip完全避免attention
- **vs ConvNeXt**: ConvNeXt在分类上证明CNN可媲美ViT；DiC将类似思路推广到生成模型

## 评分
- 新颖性: ⭐⭐⭐⭐ 逆潮流之作，证明纯卷积扩散模型的可行性
- 实验充分度: ⭐⭐⭐⭐ 完整的ablation roadmap，多规模对比
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，每一步改进动机明确
- 价值: ⭐⭐⭐⭐ 为扩散模型架构设计提供新选择，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](../../NeurIPS2025/image_generation/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[ICCV 2025\] Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers](../../ICCV2025/image_generation/rethinking_cross-modal_interaction_in_multimodal_diffusion_transformers.md)
- [\[CVPR 2025\] Decentralized Diffusion Models](decentralized_diffusion_models.md)
- [\[ICCV 2025\] EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing](../../ICCV2025/image_generation/eedit_rethinking_the_spatial_and_temporal_redundancy_for_efficient_image_editing.md)
- [\[ICCV 2025\] Rethinking Layered Graphic Design Generation with a Top-Down Approach](../../ICCV2025/image_generation/rethinking_layered_graphic_design_generation_with_a_top-down_approach.md)

</div>

<!-- RELATED:END -->
