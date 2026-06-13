---
title: >-
  [论文解读] Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing
description: >-
  [ECCV2024][音频/语音][Audio-Visual Video Parsing] 提出 LEAP（Label semantic-based Projection）解码范式，利用事件类别的标签文本嵌入作为语义锚点，通过跨模态注意力机制将音频/视觉隐特征中潜在重叠的事件语义解耦到独立的标签嵌入中，配合基于 EIoU 的音视觉语义相似度损失，在 AVVP 任务上取得 SOTA。
tags:
  - "ECCV2024"
  - "音频/语音"
  - "Audio-Visual Video Parsing"
  - "Event Disentanglement"
  - "Label Semantic Projection"
  - "Weakly Supervised"
---

# Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing

**会议**: ECCV2024  
**arXiv**: [2407.08126](https://arxiv.org/abs/2407.08126)  
**作者**: Jinxing Zhou, Dan Guo, Yuxin Mao, Yiran Zhong, Xiaojun Chang, Meng Wang (合肥工业大学, 西北工业大学, 上海AI Lab, 中科大, MBZUAI)
**代码**: 待确认  
**领域**: 音频语音  
**关键词**: Audio-Visual Video Parsing, Event Disentanglement, Label Semantic Projection, Weakly Supervised

## 一句话总结

提出 LEAP（Label semantic-based Projection）解码范式，利用事件类别的标签文本嵌入作为语义锚点，通过跨模态注意力机制将音频/视觉隐特征中潜在重叠的事件语义解耦到独立的标签嵌入中，配合基于 EIoU 的音视觉语义相似度损失，在 AVVP 任务上取得 SOTA。

## 背景与动机

Audio-Visual Video Parsing (AVVP) 任务要求从可听视频中识别并时间定位所有音频事件、视觉事件和音视觉事件。该任务在弱监督设置下进行——训练时仅有视频级事件标签。

现有方法主要着力于改进音视觉编码器以获得更好的特征表示，但对解码阶段关注不足。主流的 MMIL（Multi-modal Multi-Instance Learning）解码策略仅使用简单的线性层将隐特征直接映射到事件类别空间，存在两个问题：

1. **语义解耦不充分**：当一个时间片段包含多个重叠事件时，线性层难以清晰地展示重叠语义如何从混合特征中被分离
2. **可解释性差**：解码过程缺乏直观的语义指导，难以追踪事件是如何被识别的

## 核心问题

如何设计一个更具可解释性的事件解码范式，使得音视觉隐特征中潜在重叠的多个事件语义能够被显式地解耦和识别？

## 方法详解

### 1. LEAP 解码范式（Label Semantic-based Projection）

核心思想：利用事件类别的自然语言文本（如 "dog"、"guitar"）获取语义独立的标签嵌入，将其作为解码的语义锚点。

**标签嵌入获取**：使用预训练的 GloVe 模型将 C 个事件类别的文本编码为标签语义矩阵 $F^l \in \mathbb{R}^{C \times d}$。

**跨模态投影**：采用 Transformer 的交叉注意力机制实现投影：

- **Query**：标签嵌入 $F^l$（代表各事件语义）
- **Key/Value**：音频或视觉特征 $F^m$（$m \in \{a, v\}$）
- 计算交叉注意力矩阵 $A^{lm} \in \mathbb{R}^{C \times T}$，反映每个事件类别与每个时间片段的相似度
- 根据注意力权重聚合相关语义信息来增强标签嵌入

**迭代精炼**：LEAP 模块可迭代堆叠 N 次（实验中 N=2），每次重复使用编码特征来逐步增强与实际事件对应的标签嵌入，使其更具判别性。

**事件预测**：

- **片段级预测**：直接对最后一轮的交叉注意力矩阵 $A_N^{lm}$ 做 sigmoid 得到片段级事件概率 $P^m \in \mathbb{R}^{T \times C}$
- **视频级预测**：对增强后的标签嵌入 $F_N^{lm}$ 通过线性层 + sigmoid 得到视频级事件概率 $p^m \in \mathbb{R}^{1 \times C}$

### 2. 语义感知优化策略

**基础损失 $\mathcal{L}_{basic}$**：结合视频级弱标签和片段级伪标签（来自 VALOR 方法），对音频和视觉事件预测施加 BCE 约束。

**音视觉语义相似度损失 $\mathcal{L}_{avss}$**：

- 提出 EIoU（Event Intersection over Union）度量：计算每对音频-视觉片段中事件类别集合的 IoU，作为跨模态语义相似度的标定值
- 例如音频片段包含事件 {c1, c2, c3}，视觉片段包含 {c1, c2}，则 EIoU = 2/3
- 构建 EIoU 矩阵 $r \in \mathbb{R}^{T \times T}$ 作为监督目标
- 计算编码特征的余弦相似度矩阵 $s \in \mathbb{R}^{T \times T}$，用 MSE 损失使 $s$ 逼近 $r$

**总损失**：$\mathcal{L} = \mathcal{L}_{basic} + \lambda \mathcal{L}_{avss}$，$\lambda = 1$。

### 3. 与现有编码器的兼容性

LEAP 作为解码器可与任意音视觉编码器（如 HAN、MM-Pyr）即插即用地结合，替换原有的 MMIL 解码策略。

## 实验关键数据

**数据集**：LLP（Look, Listen, and Parse），11,849 个 YouTube 视频，25 个事件类别。

**LEAP vs MMIL 对比（MM-Pyr 编码器）**：

| 指标 | MMIL | LEAP | 提升 |
|------|------|------|------|
| Segment Type@AV | 62.2 | 64.8 | +2.6 |
| Segment Event@AV | 60.6 | 63.6 | +3.0 |
| Event Type@AV | 57.1 | 60.2 | +3.1 |
| Event Event@AV | 53.0 | 57.4 | +4.4 |

**与 SOTA 对比**：在所有事件类型解析指标上取得最优，超越 CMPAE（CVPR'23）和 VALOR（NeurIPS'23）等方法。

**重叠事件处理**：在重叠事件子集上，LEAP 相比 MMIL 平均提升 1.7%（MM-Pyr 编码器）。

**消融实验**：

- LEAP 模块数 N=2 时性能/计算的平衡最佳（Avg. 61.3%）
- 标签嵌入策略中 GloVe 最优，Bert 和 CLIP 也有效（方法对嵌入选择鲁棒）
- $\mathcal{L}_{avss}$ 在 MM-Pyr 编码器上带来 ~1.0% 的额外提升

## 亮点

1. **新颖的解码范式**：将标签文本语义引入解码过程，用语义独立的标签嵌入作为"投影目标"解耦重叠事件，思路直觉且有效
2. **强可解释性**：交叉注意力矩阵直接反映事件-片段对应关系，解码过程可追踪
3. **即插即用**：LEAP 可替换任意 AVVP 方法中的 MMIL 解码器，通用性好
4. **EIoU 度量**：利用事件集合的 IoU 作为跨模态语义相似度指标，巧妙处理了不同模态事件密度不同的问题

## 局限与展望

1. 标签嵌入采用简单的 GloVe 词嵌入，未利用更丰富的语义描述（如事件的声音/视觉特征描述），语义表达能力有限
2. 片段级伪标签依赖 VALOR 等外部方法生成，伪标签质量对 LEAP 性能有较大影响
3. EIoU 矩阵基于伪标签计算，伪标签的噪声会传播到相似度监督信号中
4. 实验仅在 LLP 单一数据集上验证，缺乏在更大规模或更多类别数据集上的验证
5. LEAP 引入了额外的 Transformer 解码模块，计算开销相比简单线性层有所增加

## 与相关工作的对比

| 方法 | 核心改进点 | Event@AV (Event) |
|------|-----------|-----------------|
| HAN (ECCV'20) | 基线编码器 + MMIL | 48.0 |
| VALOR (NeurIPS'23) | 片段级伪标签 + MMIL | 54.2 |
| CMPAE (CVPR'23) | 更强编码器 + 类别自适应阈值 | 55.7 |
| **LEAP (本文)** | **标签语义投影解码** | **57.4** |

关键区别：先前工作主要改进编码器或标签生成，本文首次系统改进解码阶段，与编码器改进正交互补。

## 启发与关联

1. **标签语义引导的解码**思路可推广到其他多标签分类场景（如多标签图像分类、动作识别），尤其是存在标签重叠的场景
2. EIoU 作为跨模态语义对齐的度量指标，可借鉴到其他需要异构模态对齐的任务中
3. "将隐特征投影到语义独立锚点"的范式与 DETR 中 object query 的思想相通，可进一步探索与 query-based 检测框架的结合

## 评分

- 新颖性: ⭐⭐⭐⭐ — 从解码范式角度切入，标签语义投影是有新意的设计
- 实验充分度: ⭐⭐⭐⭐ — 消融全面，与 MMIL 的对比详实，但仅一个数据集
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，图示直观，公式推导完整
- 价值: ⭐⭐⭐⭐ — 即插即用的解码改进，实用性强，思路有推广价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing](coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)
- [\[ICCV 2025\] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](../../ICCV2025/audio_speech/mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [\[CVPR 2025\] UWAV: Uncertainty-Weighted Weakly-Supervised Audio-Visual Video Parsing](../../CVPR2025/audio_speech/uwav_uncertainty-weighted_weakly-supervised_audio-visual_video_parsing.md)
- [\[CVPR 2025\] Towards Open-Vocabulary Audio-Visual Event Localization](../../CVPR2025/audio_speech/towards_open-vocabulary_audio-visual_event_localization.md)
- [\[ECCV 2024\] Siamese Vision Transformers are Scalable Audio-Visual Learners](siamese_vision_transformers_are_scalable_audio-visual_learners.md)

</div>

<!-- RELATED:END -->
