---
title: >-
  [论文解读] B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens
description: >-
  [ICCV 2025][模型压缩][video LLM] 提出B-VLLM框架，通过文本条件自适应帧选择、时间帧token合并和空间token采样三个模块，在VLLM上下文窗口限制内动态平衡视频的时空token，解决均匀采样忽略时间动态和每帧token减少丢失空间细节的困境，在MVBench上提升10%。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "video LLM"
  - "spatio-temporal balance"
  - "frame selection"
  - "token merging"
  - "adaptive token sampling"
---

# B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens

**会议**: ICCV 2025  
**arXiv**: 无  
**代码**: [GitHub](https://github.com/zhuqiangLu/B-VLLM)  
**领域**: 模型压缩  
**关键词**: video LLM, spatio-temporal balance, frame selection, token merging, adaptive token sampling

## 一句话总结

提出B-VLLM框架，通过文本条件自适应帧选择、时间帧token合并和空间token采样三个模块，在VLLM上下文窗口限制内动态平衡视频的时空token，解决均匀采样忽略时间动态和每帧token减少丢失空间细节的困境，在MVBench上提升10%。

## 研究背景与动机

VLLM将视觉内容编码为token序列与文本联合处理，但视频（尤其长视频）的视觉token快速增长会超出上下文窗口且计算成本剧增。现有方法要么均匀采样固定帧数（忽略时间动态，关键帧可能被漏掉），要么减少每帧token数（丢失空间细节）。核心问题是**时空token不平衡**：减少每帧token使时间线索占主导，均匀帧采样使空间线索被遮蔽。

## 方法详解

### 整体框架

B-VLLM先用ViT将所有帧编码为视觉token（包含[CLS]和patch token）。[CLS]提供粗粒度语义用于帧选择，patch token提供细粒度空间信息。流程：(1) 文本条件帧选择——利用[CLS] token和文本提示选择最相关的L*帧；(2) 时间帧token合并——去除冗余帧；(3) 空间token采样+可选空间token合并——控制最终token数不超过预算θ。

### 关键设计

1. **文本条件自适应帧选择**: 使用Q-Former联合编码所有帧的[CLS] token和文本上下文T，生成L*个查询Q，通过Gumbel-Softmax从L帧中选择L*个最相关帧。利用[CLS]token（而非全部patch token）实现高效帧选择。选择是文本条件的——不同任务选择不同关键帧。

2. **时间帧token合并**: 在帧选择后进一步去除时间冗余。基于帧间[CLS] token的相似度，合并高度相似的相邻帧的视觉token（取平均），减少时间维度冗余。

3. **空间token采样与合并**: 对选中帧进行帧内空间token采样——通过文本条件选择每帧最相关的Z个视觉token。可选的迭代空间token合并策略——类似ToMe但采用迭代方式，每次合并最相似的token对直到达到目标数量，实现对最终token数的精细控制。

### 损失函数 / 训练策略

标准VLLM训练（视觉指令微调）。帧选择模块使用Gumbel-Softmax实现可微的离散选择。Q-Former轻量级设计减少开销。

## 实验关键数据

### 主实验

| 方法 | MVBench | 长视频理解 | 空间细节保持 |
|------|---------|----------|-------------|
| 均匀帧采样 | 基线 | 中等 | 好 |
| 减少每帧token | 差 | 好 | 差 |
| **B-VLLM** | **+10%** | **好** | **好** |

在MVBench上实现10%的性能提升，在时间和空间任务上均表现优异。

### 消融实验

- 帧选择 vs 均匀采样：帧选择在时间相关任务上显著更优
- 空间token采样：保持空间细节能力
- 迭代空间token合并：精细控制token数量
- 不同VLLM骨干：框架可泛化到不同架构

### 关键发现

- [CLS] token足以做高效帧选择，无需使用全部patch token
- 文本条件选择使不同任务关注不同帧和空间区域
- 时空token的平衡对视频理解至关重要
- 迭代合并比一次性合并提供更精细的控制

## 亮点与洞察

- "时空token平衡"问题定义清晰，直击现有方法痛点
- 利用[CLS] token的粗粒度信息做帧选择，计算高效
- 文本条件的自适应选择让模型"看需要看的"
- 可泛化到不同VLLM架构

## 局限与展望

- Q-Former增加了模型参数和训练复杂度
- Gumbel-Softmax在极长视频（成百上千帧）上可能不稳定
- 帧选择的可解释性有限——难以验证选择是否最优
- 仅在标准视频QA基准上验证，对实时应用的延迟评估不足

## 相关工作与启发

- LLaMA-VID极端压缩每帧到2个token，牺牲空间但支持长视频
- ToMe的token合并思路被扩展到迭代版本
- Q-Former从BLIP-2借鉴，用于轻量级帧选择

## 评分

- 新颖性: ⭐⭐⭐⭐ — 时空token平衡的问题定义和文本条件选择新颖
- 技术深度: ⭐⭐⭐ — 各组件设计较标准，创新在组合和控制
- 实验充分性: ⭐⭐⭐⭐ — 多基准、多架构、消融完整
- 写作质量: ⭐⭐⭐⭐ — 问题展示（图1）直观有力
- 实用价值: ⭐⭐⭐⭐ — 10%性能提升，框架通用性好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)
- [\[ICCV 2025\] OuroMamba: A Data-Free Quantization Framework for Vision Mamba](ouromamba_a_data-free_quantization_framework_for_vision_mamba.md)
- [\[ICCV 2025\] MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](mixa-q_revisiting_activation_sparsity_for_vision_transformers_from_a_mixed-preci.md)
- [\[ICCV 2025\] EA-ViT: Efficient Adaptation for Elastic Vision Transformer](ea-vit_efficient_adaptation_for_elastic_vision_transformer.md)
- [\[ICCV 2025\] Beyond Low-Rank Tuning: Model Prior-Guided Rank Allocation for Effective Transfer in Low-Data and Large-Gap Regimes](beyond_low-rank_tuning_model_prior-guided_rank_allocation_for_effective_transfer.md)

</div>

<!-- RELATED:END -->
