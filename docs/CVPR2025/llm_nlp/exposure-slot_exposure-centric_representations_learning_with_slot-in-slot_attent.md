---
title: >-
  [论文解读] Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention
description: >-
  [CVPR 2025][LLM/NLP][曝光矫正] 本文提出Exposure-slot框架，将Slot Attention算法扩展为层次化的slot-in-slot结构，通过可学习的曝光prompt引导特征聚类，实现以曝光为中心的区域感知表征学习，在欠曝/过曝图像矫正任务上取得SOTA性能。
tags:
  - "CVPR 2025"
  - "LLM/NLP"
  - "曝光矫正"
  - "注意力机制"
  - "层次化聚类"
  - "区域感知"
  - "提示学习"
---

# Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 曝光矫正, Slot Attention, 层次化聚类, 区域感知, prompt学习

## 一句话总结

本文提出Exposure-slot框架，将Slot Attention算法扩展为层次化的slot-in-slot结构，通过可学习的曝光prompt引导特征聚类，实现以曝光为中心的区域感知表征学习，在欠曝/过曝图像矫正任务上取得SOTA性能。

## 研究背景与动机

### 领域现状

**领域现状**：图像曝光矫正旨在恢复因欠曝或过曝导致的细节损失和颜色偏移。近年来基于深度学习的方法（如RetinexNet、Uformer等）取得了显著进展，广泛应用于移动摄影、安防监控等场景。

**现有痛点**：(1) 全局处理策略失效——现有方法通常对整张图像施加统一的增强操作，但实际图像中不同区域的曝光程度差异很大（如阴影区域欠曝、天空区域过曝），全局操作无法兼顾。(2) 特征学习不够针对性——CNN/Transformer提取的特征缺乏对曝光状态的显式建模，导致矫正效果对混合曝光场景不佳。(3) 区域划分困难——不同曝光程度的区域边界模糊，不适合硬分割。

**核心矛盾**：有效的曝光矫正需要针对不同曝光程度的区域施加不同的增强策略，但如何在没有区域级标注的情况下自动发现和区分这些区域是一个挑战。

**本文目标** 如何学习以曝光为中心的特征表示，自动发现并针对性地处理不同曝光程度的区域？

**切入角度**：借鉴物体发现领域的Slot Attention思想——通过竞争性注意力机制将特征"分配"到不同的slot中，每个slot对应一种曝光模式，实现软性区域聚类。

**核心 idea**：用层次化slot-in-slot注意力对图像特征按曝光程度进行渐进式聚类，结合可学习曝光prompt实现区域感知矫正。

## 方法详解

### 整体框架

Exposure-slot采用编码器-解码器架构，在编码器和解码器之间引入slot-in-slot注意力模块。编码器提取多尺度特征后，外层slot先将特征粗分为几大曝光类别（如严重欠曝、轻微欠曝、正常、过曝），内层slot进一步在每个类别内细分子区域。可学习的曝光prompt注入到slot初始化中，引导聚类过程。最终，各slot的输出通过加权融合重建矫正后的图像。

### 关键设计

1. **Slot-in-Slot层次化注意力**：
    - 功能：渐进式地将图像特征按曝光程度聚类为多层次表示
    - 核心思路：扩展原始Slot Attention为两层结构。外层slot（K_outer个）通过迭代竞争注意力将所有空间位置特征分配到K个曝光簇中；内层slot在每个外层slot内部进一步细分（K_inner个），捕捉同一曝光大类下的空间分布差异。注意力权重通过softmax归一化实现soft assignment
    - 设计动机：单层slot可能过于粗糙，无法区分曝光程度相近但空间位置不同的区域；层次化设计先粗后细，提升聚类精度

2. **可学习曝光Prompt**：
    - 功能：引导slot初始化，使不同slot专注于不同曝光条件
    - 核心思路：为每个外层slot训练一个可学习的prompt向量，编码特定曝光条件的先验知识（如"严重欠曝"、"轻微过曝"等）。这些prompt在推理时作为slot的初始值，引导注意力机制快速收敛到正确的曝光区域
    - 设计动机：标准Slot Attention使用随机初始化或可学习均值初始化，缺乏曝光相关的语义引导，收敛慢且不稳定

3. **区域感知重建与融合**：
    - 功能：将各slot的矫正结果自适应融合为最终输出
    - 核心思路：每个slot经过独立的轻量解码分支生成局部矫正特征，同时slot attention的权重图自然形成区域融合掩码。通过加权求和（权重由slot对各位置的注意力强度决定）生成最终矫正图像
    - 设计动机：不同区域需要不同的矫正强度和色彩调整策略，硬掩码会产生边界伪影，soft attention权重能实现平滑过渡

### 损失函数 / 训练策略
模型采用端到端训练，优化目标综合考虑任务损失和正则化项。


## 实验关键数据

### 关键发现

- 在MSEC数据集上，Exposure-slot在PSNR/SSIM等指标上超越RetinexFormer、FECNet等SOTA方法
- 对混合曝光图像（同时存在欠曝和过曝区域）的处理效果尤为突出
- 消融实验证明层次化slot结构比单层slot提升约1.2dB PSNR
- 可学习曝光prompt相比随机初始化提升约0.8dB PSNR
- 可视化表明不同slot确实学到了对应不同曝光程度区域的注意力模式

## 亮点与洞察

- **Slot Attention的创新应用**：将物体发现领域的Slot Attention巧妙迁移到图像增强领域，曝光区域发现本质上也是一种无监督聚类问题
- **层次化设计合理**：粗到细的两级聚类符合人类对曝光评估的直觉（先判断全局曝光状态，再处理局部差异）
- **无需区域标注**：不需要显式的曝光区域分割标注，slot通过端到端训练自动学习区域划分

## 局限与展望

- Slot数量是预设的超参数，不同场景可能需要不同的slot数量
- 对极端欠曝（几乎全黑）场景效果可能受限于信息丢失
- 层次化注意力增加了计算开销，实时应用场景可能需要轻量化
- 未来可结合RAW域信息进一步提升矫正质量


## 相关工作与启发
- **vs 同领域代表性方法**：本文在方法设计上有独特贡献，与现有方法形成互补
- **vs 传统方法**：相比传统方案，本文方法在关键指标上取得了显著提升
- **启发**：本文的技术路线对后续相关工作有重要参考价值


## 评分
- 新颖性: ⭐⭐⭐⭐ 方法设计有独特贡献
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证
- 写作质量: ⭐⭐⭐⭐ 条理清晰
- 价值: ⭐⭐⭐⭐ 对领域有推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] TestCase-Eval: A Systematic Evaluation of Fault Coverage and Exposure](../../ACL2025/llm_nlp/testcase_eval_llm_test_gen.md)
- [\[CVPR 2025\] Robust Message Embedding via Attention Flow-Based Steganography](robust_message_embedding_via_attention_flow-based_steganography.md)
- [\[ACL 2025\] Uncertainty Unveiled: Can Exposure to More In-context Examples Mitigate Uncertainty for Large Language Models?](../../ACL2025/llm_nlp/uncertainty_unveiled_can_exposure_to_more_in-context_examples_mitigate_uncertain.md)
- [\[CVPR 2025\] Spiking Transformer with Spatial-Temporal Attention](spiking_transformer_with_spatial-temporal_attention.md)
- [\[CVPR 2025\] Learning Textual Prompts for Open-World Semi-Supervised Learning](learning_textual_prompts_for_open-world_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->
