---
title: >-
  [论文解读] Understanding Co-speech Gestures in-the-wild
description: >-
  [ICCV 2025][人体理解][共语手势] 本文提出 JEGAL——一个联合手势-语音-文本的三模态嵌入空间，通过全局短语对比损失和局部手势-词耦合损失在弱监督条件下学习共语手势表征，定义了三个新的手势理解任务和基准，超越了包括大型视觉语言模型在内的多种方法。
tags:
  - ICCV 2025
  - 人体理解
  - 共语手势
  - 三模态表征学习
  - 手势检索
  - 手势词定位
  - 活跃说话人检测
---

# Understanding Co-speech Gestures in-the-wild

**会议**: ICCV 2025  
**arXiv**: [2503.22668](https://arxiv.org/abs/2503.22668)  
**代码**: [项目主页](https://www.robots.ox.ac.uk/~vgg/research/jegal)  
**领域**: 人体理解  
**关键词**: 共语手势, 三模态表征学习, 手势检索, 手势词定位, 活跃说话人检测

## 一句话总结

本文提出 JEGAL——一个联合手势-语音-文本的三模态嵌入空间，通过全局短语对比损失和局部手势-词耦合损失在弱监督条件下学习共语手势表征，定义了三个新的手势理解任务和基准，超越了包括大型视觉语言模型在内的多种方法。

## 研究背景与动机

人类说话时会做手势——手势是人类交流的重要组成部分。共语手势种类丰富：从节拍手势（beat gestures，强调特定词汇的节奏性手部运动）到图示性手势（iconic gestures，表示语义内容，如用手臂展开表示"巨大"）。非语言交流占总体交流的 55%，但机器对手势语义的理解仍然非常有限。

学习手势与语音/文本之间的关联异常困难，原因包括：

**稀疏且模糊的跨模态相关性**：通常只有少数几个词会被清晰地用手势表达，同一句话在不同上下文/不同人的表达中手势差异巨大。

**高度的个体和文化差异**：手势取决于说话人的情绪、文化、社交场景等因素。

**部分手势无语义信息**：节拍手势仅与语音韵律对齐，不携带语义内容，无法直接映射到特定词汇。

现有工作的不足：
- GestSync 通过音视频同步来学习手势表征，但只捕捉低层次的时序关联而非高层语义。
- GestureDiffuCLIP 学习了手势-文本联合嵌入，但缺少词级对应关系。
- 大型视觉语言模型（如 CLIP、LanguageBind）不是为手势理解设计的，处理长视频和手势特征的能力有限。

## 方法详解

### 整体框架

JEGAL（Joint Embedding space for Gestures, Audio, and Language）学习三模态表征，通过两个互补的对比学习目标：

1. **全局短语对比损失**：鼓励模型学习手势片段与语音/文本片段之间的整体语义对应。
2. **局部手势-词耦合损失**：鼓励模型发现手势片段中与特定词汇对应的帧级关联。

### 关键设计

1. **三个模态编码器 + 融合模块**

    - **手势编码器 $\mathbb{G}$**：3D 卷积层（首层时序感受野 5 帧）+ Transformer 编码器，输出帧级特征 $\mathbf{g}^T \in \mathbb{R}^{T \times d}$。骨干网络从 GestSync 初始化并冻结，仅训练 Transformer 头。面部区域被遮罩以防止唇部运动信息泄漏。
    - **文本编码器 $\mathbb{L}$**：多语言 RoBERTa XLM-Base 提取子词特征，经 Transformer 头编码和投影得到 $\mathbf{l}^w \in \mathbb{R}^{W \times d/2}$。
    - **语音编码器 $\mathbb{S}$**：2D-CNN 编码梅尔频谱图，输出 $\mathbf{s} \in \mathbb{R}^{T' \times d/2}$。
    - **融合模块**：将子词文本嵌入聚合为词级嵌入 $\mathbf{l}^w$，将语音特征按词边界聚合为词级特征 $\mathbf{s}^w$，在特征维度上拼接得到联合词级表示 $\mathbf{c}^w \in \mathbb{R}^{W \times d}$。

   训练中 50% 概率随机将语音或文本输入置零（模态丢弃），促使模型均衡学习两种模态，也允许推理时仅用单一模态。

2. **手势-词对齐机制**

   词的边界与语音对齐，但不一定与手势对齐——手势可能比词更长/更短或有偏移。为此设计了基于注意力的池化：将词的时间窗口向两侧扩展 $p=10$ 帧，然后用词嵌入 $c^{w_i}$ 对扩展窗口内的手势帧进行注意力加权聚合：

   $$g^{w_i} = \sum_{j=S}^{E} \frac{\exp(\gamma \cdot g^{T_j} \cdot c^{w_i})}{\sum_{j=S}^{E} \exp(\gamma \cdot g^{T_j} \cdot c^{w_i})} \cdot g^{T_j}$$

   这使得模型能够灵活地在语音边界之外找到手势的真正时间范围。

3. **双重训练目标**

   **全局短语对比损失**：对帧级手势和词级语音文本分别平均池化得到全局嵌入 $\mathbf{g}$ 和 $\mathbf{c}$，使用 InfoNCE 损失：

   $$\mathcal{L}_{seq} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\gamma \cdot \cos(g_i, c_i))}{\sum_{j=1}^{N} \exp(\gamma \cdot \cos(g_i, c_j))}$$

   **局部手势-词耦合损失**：对于每个语音文本词 $c^{w_i}$，找到其最相似的手势帧 $g^{w_j}$，计算耦合评分 $\lambda(g^w, c^w) = \frac{1}{W}\sum_{i=1}^{W}\max_{j}\cos(g^{w_i}, c^{w_j})$。核心假设：匹配的手势-语音文本对有更多强词级耦合。同样使用 InfoNCE 形式的对比损失。

   最终损失：$\mathbb{L} = \beta \cdot \mathcal{L}_{seq} + (1-\beta) \cdot \mathcal{L}_{couple}$

### 损失函数 / 训练策略

- 数据来源：PATS（25 个说话人、162 小时）+ MultiVSR 子集（6,934 个说话人、556 小时），共约 720 小时、7,000+ 说话人。
- 预处理：重采样 25FPS / 16kHz，WhisperX 生成词对齐转录，基于身体关键点 L2 距离过滤低手势活动样本。
- 优化器：AdamW, lr=5e-5, weight decay=1e-4, betas=(0.9, 0.98)。
- 手势头 6 层 Transformer，文本头 3 层，hidden dim=512, FFN=2048, 8 heads。

## 实验关键数据

### 主实验

**跨模态检索（AVS-Ret 基准，500 个多样性手势片段）**：

| 方法 | 模态 | S→G R@5↑ | S→G R@10↑ | S→G MR↓ | G→S R@5↑ | G→S R@10↑ | G→S MR↓ |
|------|------|----------|-----------|---------|----------|-----------|---------|
| GestSync (FT) | Audio | 10.0 | 18.2 | 70.5 | 11.6 | 16.6 | 82.5 |
| Clip4Clip (FT) | Text | 8.0 | 12.6 | 132.0 | 3.6 | 7.0 | 125.0 |
| **JEGAL** | **T+A** | **18.8** | **30.8** | **31.0** | **18.2** | **20.2** | **24.5** |

JEGAL 大幅领先所有基线，多模态融合（T+A）显著优于单模态。

**手势词定位（AVS-Spot 基准，500 个标注片段）**：

| 方法 | 模态 | 准确率↑ |
|------|------|--------|
| GestSync (FT) | Audio | 21.04 |
| GestureDiffuCLIP (FT) | Text | 19.50 |
| JEGAL (Text) | Text | 61.00 |
| **JEGAL (T+A)** | **T+A** | **63.60** |

JEGAL 在词定位上碾压所有基线（63.6% vs 21.04%），核心优势来自局部手势-词耦合损失。

**活跃说话人检测（AVS-Asd 基准）**：

| 方法 | 2人↑ | 4人↑ | 6人↑ |
|------|------|------|------|
| GestSync (FT) | 81.2 | 64.8 | 54.4 |
| **JEGAL (T+A)** | **76.8** | **57.8** | **48.0** |

GestSync 在此任务上最优（因其有强帧级同步监督），JEGAL 紧随其后。

### 消融实验

| 损失配置 | 检索 R@5↑ | 检索 MR↓ | 定位 Acc↑ | ASD Acc↑ |
|---------|-----------|---------|----------|----------|
| 仅全局对比 | 12.20 | 45 | 20.83 | 44.2 |
| 仅词耦合 | 8.50 | 76 | 52.46 | 14.8 |
| **全局 + 词耦合** | **18.80** | **31** | **63.60** | **48.0** |

| 融合策略 | 检索 R@5↑ | 定位 Acc↑ | ASD Acc↑ |
|---------|-----------|----------|----------|
| 独立成对对比 (text) | 9.39 | 34.31 | 29.6 |
| 独立成对对比 (audio) | 9.80 | 23.67 | 31.4 |
| 晚期融合 (平均) | 17.00 | 56.04 | 41.2 |
| **晚期融合 (拼接)** | **18.80** | **63.60** | **48.0** |

### 关键发现

- **语音和文本捕捉互补的手势信号**：文本在词级语义对应上更有效（定位 61.0% vs 41.8%），而语音对重音/强调词更敏感（重音词定位差异 39.4% vs 非重音词差异 14.8%）。
- **两个损失缺一不可**：全局对比损失对检索和 ASD 至关重要，词耦合损失对定位至关重要，组合使用在所有任务上最优。
- 拼接融合优于平均融合和独立成对对比——模型需要在同一嵌入空间内整合多个信息流。

## 亮点与洞察

- **弱监督下的词级学习**：仅有短语级监督（不知道哪些词被手势表达），但通过耦合损失的 max-coupling 策略成功学到了词级对应关系。
- **三个新任务和基准的定义**：为共语手势理解建立了系统性的评估框架（检索/定位/ASD），有助于推动该领域的发展。
- **面部遮罩设计**：遮罩面部区域避免唇部运动信息泄漏，确保学到的是纯手势信号。

## 局限与展望

- 训练数据以英语为主，跨语言/跨文化的手势理解能力未验证。
- 手势编码器使用 RGB 视频，计算开销大；未来可探索基于 2D/3D 关键点的轻量化输入。
- 当前仅考虑手部手势，未纳入头部动作、面部表情等其他非语言信号。
- 活跃说话人检测任务中，JEGAL 仍落后于 GestSync，说明帧级同步对齐能力仍有提升空间。

## 相关工作与启发

- 与手语理解的本质区别：手语中手势是主要交流方式（文本是翻译），而共语手势是语音的补充（手势与词汇共现但非翻译关系），需要完全不同的建模方法。
- 耦合损失的 max-coupling 策略与 MIL（多示例学习）中的思想类似，在仅有包级标签时学习实例级信息。
- 该表征可用于数字人手势生成、语言学习辅助、隐私保护的说话人检测等实际应用。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 三模态共语手势理解的系统性框架，词耦合损失设计巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个新base任务和消融研究覆盖全面，语音 vs 文本的深入分析有洞察
- **写作质量**: ⭐⭐⭐⭐⭐ 论文动机清晰，图示出色，分析深入
- **价值**: ⭐⭐⭐⭐ 为共语手势理解开辟了系统性研究方向，三个基准对社区有持续价值

<!-- RELATED:START -->

## 相关论文

- [I See What You Mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue](../../ACL2025/human_understanding/i_see_what_you_mean_co-speech_gestures_for_reference_resolution_in_multimodal_di.md)
- [Streaming Generation of Co-Speech Gestures via Accelerated Rolling Diffusion](../../AAAI2026/human_understanding/streaming_generation_of_co-speech_gestures_via_accelerated_rolling_diffusion.md)
- [SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis](semtalk_holistic_co-speech_motion_generation_with_frame-level_semantic_emphasis.md)
- [SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning](semges_semantics-aware_co-speech_gesture_generation_using_semantic_coherence_and.md)
- [GestureHYDRA: Semantic Co-speech Gesture Synthesis via Hybrid Modality Diffusion Transformer and Cascaded-Synchronized Retrieval-Augmented Generation](gesturehydra_semantic_co-speech_gesture_synthesis_via_hybrid_modality_diffusion_.md)

<!-- RELATED:END -->
