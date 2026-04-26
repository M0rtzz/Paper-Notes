---
title: >-
  [论文解读] Degradation-Aware Feature Perturbation for All-in-One Image Restoration
description: >-
  [CVPR 2025][图像恢复][全能图像恢复] 本文提出DFPIR框架，通过退化类型引导的通道打乱扰动和注意力选择性掩码扰动两种机制，在编解码器之间调整特征空间以适配统一参数空间，在去噪/去雾/去雨/去模糊/低光增强五个任务上取得SOTA。
tags:
  - CVPR 2025
  - 图像恢复
  - 全能图像恢复
  - 特征扰动
  - 通道打乱
  - 注意力选择
  - 退化感知
---

# Degradation-Aware Feature Perturbation for All-in-One Image Restoration

**会议**: CVPR 2025  
**arXiv**: [2505.12630](https://arxiv.org/abs/2505.12630)  
**代码**: [GitHub](https://github.com/TxpHome/DFPIR)  
**领域**: 图像恢复  
**关键词**: 全能图像恢复, 特征扰动, 通道打乱, 注意力选择, 退化感知

## 一句话总结
本文提出DFPIR框架，通过退化类型引导的通道打乱扰动和注意力选择性掩码扰动两种机制，在编解码器之间调整特征空间以适配统一参数空间，在去噪/去雾/去雨/去模糊/低光增强五个任务上取得SOTA。

## 研究背景与动机

**领域现状**：单任务图像恢复已取得显著进展，但为每种退化单独训练模型的方式在实际应用中代价过高。All-in-One方法试图用一个统一模型处理多种退化。

**现有痛点**：共享参数时不同退化任务之间存在梯度冲突——去噪和去雾的优化方向可能相反。现有方法分为两类：调整参数空间（引入额外退化参数，计算开销大）和调整特征空间（如PromptIR的隐式提示、InstructIR的文本指令），但前者增加复杂度，后者难以有效隔离退化间干扰。

**核心矛盾**：需要保留图像固有特征（对所有任务通用有益），同时减少不同退化特征之间的互相干扰。现有prompt方法要么忽略退化类型信息（PromptIR），要么只用通道注意力调制（InstructIR），难以充分隔离不同退化的影响。

**本文目标**：设计一种既保留图像固有特征又能按退化类型调制特征的轻量机制。

**切入角度**：通过"扰动"而非"替换"来调制特征——通道打乱改变特征排列但保留信息，注意力掩码选择性过滤但不完全丢弃。

**核心 idea**：用退化类型prompt引导通道维度的打乱（channel shuffle）实现特征重排列，再用选择性注意力掩码进行注意力空间的扰动，二者配合使编码特征适配统一解码器。

## 方法详解

### 整体框架
基于Restormer的4级编解码器架构。在编码器和解码器之间的skip connection位置插入DGPB（退化引导扰动块）。退化类型描述通过预训练CLIP文本编码器获得prompt向量，用于引导扰动。

### 关键设计

1. **退化引导通道扰动模块（DGCPM）**:

    - 功能：按退化类型重新排列特征通道
    - 核心思路：先将通道数扩展2倍到高维空间，用退化prompt经MLP映射到与通道数等长的向量，取top-K索引对通道重新排序（channel shuffle），再halve回原始通道数。不同退化类型因prompt不同而产生不同的通道排列
    - 设计动机：直接在原通道上shuffle扰动过大导致难以收敛，先扩展到高维再打乱再降维是关键。通道shuffle保留了所有信息（只是重排），比通道attention的加权/过滤更温和

2. **通道适应注意力扰动模块（CAAPM）**:

    - 功能：让打乱后的特征与原始特征交互，并在注意力空间施加选择性扰动
    - 核心思路：以打乱特征为query、原始特征为key/value做通道维cross-attention，在attention map上用top-K mask（扰动因子γ=0.9，即保留90%）选择性屏蔽部分注意力权重，再经FFN输出
    - 设计动机：打乱特征包含退化类型信息但缺少与原始特征的交互，cross-attention融合两者；attention masking进一步过滤各退化不需要的信息，实现"干扰隔离与固有特征保留"的平衡

3. **退化类型Prompt系统**:

    - 功能：为DGPB提供退化类型条件
    - 核心思路：利用预训练CLIP文本编码器编码退化类型描述（如"denoising"、"dehazing"），得到的嵌入向量经DGM模块映射后用于引导通道打乱的索引序列
    - 设计动机：CLIP文本嵌入自然包含退化类型的语义信息，且不同退化类型的嵌入向量差异足够大

### 损失函数 / 训练策略
标准L1重建损失。训练时需要退化类型标签来获取对应prompt。

## 实验关键数据

### 主实验（3任务：去雾+去雨+去噪）

| 方法 | 去雾SOTS | 去雨Rain100L | 去噪CBSD68(σ=25) | 平均PSNR |
|------|---------|-------------|-----------------|---------|
| PromptIR | 30.58 | 36.37 | 31.31 | 32.06 |
| InstructIR | 30.22 | **37.98** | **31.52** | 32.43 |
| **DFPIR(Ours)** | **31.87** | 38.65 | 31.47 | **32.88** |

### 消融实验（5任务设定）

| 方法 | 去雾 | 去雨 | 去噪 | 去模糊 | 低光 | 平均 |
|------|------|------|------|--------|------|------|
| Restormer* | 24.09 | 34.81 | 31.49 | 27.22 | 20.41 | 27.60 |
| PromptIR | 30.58 | 36.37 | 31.31 | 29.40 | 23.15 | 30.16 |
| **DFPIR** | **31.87** | **38.65** | 31.47 | **29.86** | **23.38** | **31.05** |

### 关键发现
- 相比InstructIR平均PSNR提升0.45dB，特别是去雾任务提升显著（+1.65dB）
- t-SNE可视化显示DFPIR比PromptIR有更紧密的任务内特征聚类，验证了扰动策略有效隔离了不同退化
- 扰动因子γ=0.9最优，过小（丢弃太多信息）或过大（扰动不足）都会降低性能
- 通道打乱在高维空间进行是必要条件，直接在原通道打乱会导致训练不稳定

## 亮点与洞察
- "通道打乱"作为特征调制手段是新颖的——保留全部信息的重排列vs常规attention的加权过滤，思路清晰且高效
- DGPB只增加少量参数，但效果显著，说明特征空间的"扰动"比"替换"更适合多任务共享参数场景
- t-SNE可视化很有说服力，直观展示了扰动如何使不同退化的特征聚类更清晰

## 局限与展望
- 推理时需要已知退化类型以获取prompt，不支持未知退化的盲恢复
- 退化类型为离散标签，不支持退化程度的连续调控
- 五种退化类型仍有限，未验证对更多退化类型的可扩展性
- 可考虑用退化估计网络自动获取prompt，实现全盲恢复

## 相关工作与启发
- **vs PromptIR**: PromptIR用隐式prompt在feature中学习，忽略退化类型显式信息；DFPIR用CLIP文本嵌入作为显式退化条件
- **vs InstructIR**: InstructIR用文本指令+通道attention调制；DFPIR的通道打乱+attention masking更有效隔离干扰
- **vs MedIR**: MedIR的硬路由策略分离任务参数可能忽略固有特征；DFPIR的"软扰动"保留全部信息

## 评分
- 新颖性: ⭐⭐⭐⭐ 通道打乱作为特征调制的idea新颖
- 实验充分度: ⭐⭐⭐⭐ 3任务和5任务设定，多baseline对比，消融完整
- 写作质量: ⭐⭐⭐⭐ 图示清晰，动机分析到位
- 价值: ⭐⭐⭐⭐ All-in-One恢复的通用轻量模块

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Visual-Instructed Degradation Diffusion for All-in-One Image Restoration](visual-instructed_degradation_diffusion_for_all-in-one_image_restoration.md)
- [\[CVPR 2025\] Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks](vision-language_gradient_descent-driven_all-in-one_deep_unfolding_networks.md)
- [\[AAAI 2026\] ClearAIR: A Human-Visual-Perception-Inspired All-in-One Image Restoration](../../AAAI2026/image_restoration/clearair_a_human-visual-perception-inspired_all-in-one_image_restoration.md)
- [\[CVPR 2025\] One-Step Event-Driven High-Speed Autofocus](one-step_event-driven_high-speed_autofocus.md)
- [\[CVPR 2025\] DnLUT: Ultra-Efficient Color Image Denoising via Channel-Aware Lookup Tables](dnlut_ultra-efficient_color_image_denoising_via_channel-aware_lookup_tables.md)

<!-- RELATED:END -->
