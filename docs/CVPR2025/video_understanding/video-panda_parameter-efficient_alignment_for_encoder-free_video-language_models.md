---
title: >-
  [论文解读] Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models
description: >-
  [CVPR 2025][视频理解][无编码器视频语言模型] Video-Panda 提出了首个无编码器（encoder-free）的视频语言模型，通过仅 45M 参数的时空对齐模块（STAB）直接处理视频输入，在开放式视频问答任务上达到与使用 300M-1.4B 参数编码器的方法相当的性能，且推理速度提升 3-4 倍。
tags:
  - CVPR 2025
  - 视频理解
  - 无编码器视频语言模型
  - 时空对齐
  - 参数高效
  - 视频问答
  - 轻量化
---

# Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models

**会议**: CVPR 2025  
**arXiv**: [2412.18609](https://arxiv.org/abs/2412.18609)  
**代码**: https://jh-yi.github.io/Video-Panda  
**领域**: 视频理解  
**关键词**: 无编码器视频语言模型, 时空对齐, 参数高效, 视频问答, 轻量化

## 一句话总结
Video-Panda 提出了首个无编码器（encoder-free）的视频语言模型，通过仅 45M 参数的时空对齐模块（STAB）直接处理视频输入，在开放式视频问答任务上达到与使用 300M-1.4B 参数编码器的方法相当的性能，且推理速度提升 3-4 倍。

## 研究背景与动机

**领域现状**：当前视频语言模型（如 Video-ChatGPT、Video-LLaVA）通常依赖预训练的大型视觉编码器（300M-1.4B 参数）来提取帧级特征，然后通过对齐模块将视觉特征映射到 LLM 的嵌入空间。部分方法甚至同时使用图像编码器和视频编码器。

**现有痛点**：重量级编码器带来巨大的计算开销，处理多帧视频时需要反复通过编码器，延迟显著。此外简单地将图像语言模型架构适配到视频理解会导致性能下降，因为它们无法捕捉视频特有的时空关系。

**核心矛盾**：视频理解既需要丰富的时空建模能力，又面临多帧处理带来的计算压力。现有方法通过堆叠更大的编码器来解决前者，却加剧了后者。

**本文目标**：设计一个极轻量的 encoder-free 架构，直接从原始视频像素出发，用不到 50M 参数完成视觉处理，同时保持与 encoder-based 方法竞争力的视频问答性能。

**切入角度**：作者观察到图像领域的 encoder-free 方法（如 Fuyu-8B、EVE）已经展示了不依赖预训练编码器的可行性，但这些方法简单扩展到视频会失败，因为缺乏时空建模。因此需要专门为视频设计的对齐模块。

**核心 idea**：用一个专用的时空对齐模块（STAB）替代传统的视觉编码器，通过显式的局部/全局时空建模和帧级空间建模，以极低参数量实现视频-语言对齐。

## 方法详解

### 整体框架
输入视频均匀采样 8 帧，每帧分割为 patch 后直接进入 STAB 模块处理，不经过任何预训练编码器。STAB 内部包含局部时空编码（LSTE）、局部空间下采样（LSD）、帧级空间关系聚合（FSRA）和全局时空关系聚合（GSTRA）四个子模块。处理后的视觉 token 通过 MLP 映射到 LLM（Vicuna-7B）的嵌入空间，配合文本 token 完成视频问答。训练分三阶段：初始对齐、视觉-语言联合训练和指令微调。

### 关键设计

1. **局部时空编码（LSTE）+ 动态位置编码**:

    - 功能：在局部时空窗口内提取细粒度特征并编码位置信息
    - 核心思路：使用三层级联的 3D 卷积处理 patch embedding，其中 Conv3D$_1$ 和 Conv3D$_3$ 用 $1\times1\times1$ 核做通道压缩/恢复，Conv3D$_2$ 用 $3\times1\times1$ 核建模时序上下文，加上残差连接。然后通过 $3\times3\times3$ 的 depthwise 3D 卷积实现动态位置编码（DPE），$L_{st} = L'_{st} + \text{DPE}(L'_{st})$
    - 设计动机：视频在局部邻域展示丰富的时序动态，3D 卷积可以高效捕获这些模式，同时 DPE 提供结构感知的位置信息，比固定位置编码更灵活

2. **空间下采样 + 帧级/全局时空双路聚合**:

    - 功能：压缩空间分辨率并分别建模帧级内容和视频级上下文
    - 核心思路：LSD 用 $2\times2$ 窗口内的可学习 query 做注意力下采样，将空间维度减半。然后 FSRA 用帧特定的可学习 query 对每帧做全局空间聚合得到帧摘要 $F_{s,t}$，GSTRA 用全局 query 对所有帧 token 做跨时空聚合得到视频级上下文 $G_{st}$。最后通过可学习权重 $\alpha$ 融合：$F_{r,t} = f_{\text{proj}}(\alpha F_{s,t} + (1-\alpha)G_{st})$
    - 设计动机：不同帧的空间内容差异大，FSRA 保留帧特异性；而视频理解需要全局上下文（如整体事件理解），GSTRA 提供这种跨帧信息。消融实验证明两者互补，去掉 GSTRA 在 ActivityNet-QA 上掉 1.8 个点

3. **三阶段渐进训练策略**:

    - 功能：稳定优化并实现知识迁移
    - 核心思路：Stage 1（351K 数据）冻结 LLM 只训练 STAB，用 LanguageBind 作为 teacher 的蒸馏损失 $\mathcal{L}_{\text{distill}} = -\frac{1}{M}\sum_i \frac{v_{\text{pred},i}^\top v_{\text{teacher},i}}{\|v_{\text{pred},i}\|\|v_{\text{teacher},i}\|}$ + 文本交叉熵。Stage 2（702K 数据）端到端微调。Stage 3（100K 指令数据）强化指令跟随
    - 设计动机：直接端到端训练会导致 collapse，先对齐视觉表示再逐步解锁 LLM 参数更稳定

### 损失函数 / 训练策略
训练目标包含视觉蒸馏损失（余弦相似度）和文本生成的标准交叉熵损失。Teacher 模型为 LanguageBind（基于 ViT-L/14 预训练的对比学习模型），输入缩放到 224×224。

## 实验关键数据

### 主实验（同数据集对比）

| 方法 | 视觉参数量 | MSVD-QA Acc/Score | MSRVTT-QA Acc/Score | TGIF-QA Acc/Score | ActivityNet-QA Acc/Score |
|------|-----------|-------------------|---------------------|-------------------|------------------------|
| Video-ChatGPT | 307M | 64.9/3.3 | 49.3/2.8 | 40.7/3.1 | 35.2/2.8 |
| Video-LLaVA | 425M | 64.8/- | 58.3/- | 41.7/- | 40.7/- |
| **Video-Panda** | **45M** | **64.7/3.8** | **54.8/3.4** | **42.9/3.2** | **40.0/3.3** |

### 消融实验

| 配置 | MSVD-QA Acc/Score | ActivityNet-QA Acc/Score |
|------|-------------------|------------------------|
| w/o LSTE | 63.6/3.7 | 39.4/3.3 |
| w/o GSTRA | 63.0/3.7 | 38.2/3.2 |
| w/o GSTRA & LSTE | 62.2/3.7 | 38.1/3.2 |
| w/o LSD (avg pool) | 58.0/3.6 | 38.1/3.2 |
| **Video-Panda** | **64.7/3.8** | **40.0/3.3** |

### 关键发现
- LSD（注意力下采样）是最关键组件，替换为平均池化后 MSVD-QA 从 64.7 暴跌到 58.0，说明可学习的空间聚合远优于简单池化
- 去掉 GSTRA 对 ActivityNet-QA（长视频）影响最大（-1.8），因为长视频更依赖全局上下文
- 计算效率：视觉部分 FLOPs 仅 105.5G（Video-ChatGPT 的 1/77），推理延迟 41ms vs 171ms（4.2× 加速）
- 在细粒度评估中，Video-Panda 在正确性（2.74 vs 2.40）和时序理解（2.26 vs 1.98）上均超过 Video-ChatGPT

## 亮点与洞察
- **极致的参数效率**：45M 参数做到 307M-425M 的效果，证明了视频理解不一定需要大型预训练编码器，关键在于时空建模的设计
- **双路聚合的互补性**：帧级（空间）和视频级（时空）表示的分离再融合是优雅的设计，可学习的融合权重自动平衡局部细节和全局语义
- **蒸馏驱动的 encoder-free 训练**：用预训练编码器作为 teacher 做蒸馏，是 encoder-free 方法获得良好视觉表示的有效路径，可迁移到其他模态

## 局限与展望
- 仍然依赖 LanguageBind 作为蒸馏的 teacher，并非完全摆脱预训练视觉模型的依赖
- 仅采样 8 帧，对于长视频或需要精细时序推理的任务可能不足
- 分辨率限制在 448×448，在需要高分辨率视觉细节的任务上可能受限
- 未来可探索自蒸馏或无 teacher 的训练范式，进一步降低对预训练模型的依赖

## 相关工作与启发
- **vs Video-ChatGPT**: 核心区别在于去掉了 ViT 编码器, 用 STAB 替代。参数量减少 6.8 倍但性能更优
- **vs EVE**: EVE 是图像领域的 encoder-free 先驱，但简单扩展到视频效果很差（MSVD-QA 60.5 vs 64.7），说明视频需要专门的时空建模
- 这种 encoder-free + 蒸馏的范式可以推广到其他多模态场景，如音频-语言或点云-语言模型

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个 encoder-free 视频语言模型，STAB 设计合理
- 实验充分度: ⭐⭐⭐⭐ 四个数据集、细粒度评估、详细消融和效率分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观
- 价值: ⭐⭐⭐⭐ 开辟了视频语言模型轻量化的新方向

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Efficient Transfer Learning for Video-language Foundation Models](efficient_transfer_learning_for_video-language_foundation_models.md)
- [\[CVPR 2025\] Temporal Alignment-Free Video Matching for Few-Shot Action Recognition](temporal_alignment-free_video_matching_for_few-shot_action_recognition.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)
- [\[CVPR 2025\] M-LLM Based Video Frame Selection for Efficient Video Understanding](m-llm_based_video_frame_selection_for_efficient_video_understanding.md)
- [\[CVPR 2025\] On the Consistency of Video Large Language Models in Temporal Comprehension](on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)

<!-- RELATED:END -->
