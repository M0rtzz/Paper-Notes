---
title: >-
  [论文解读] Causality Matters: How Temporal Information Emerges in Video Language Models
description: >-
  [AAAI 2026][视频理解][视频语言模型] 通过系统性消融实验揭示VideoLM的时序理解能力并非来源于位置编码(PE)，而是由因果注意力掩码的序列敏感性产生——时序信息沿"帧间交互→末帧聚合→query融合"的因果路径逐层构建，并据此提出两种无损推理加速策略。
tags:
  - AAAI 2026
  - 视频理解
  - 视频语言模型
  - 时序理解
  - 因果注意力
  - 位置编码
  - 模型可解释性
  - 推理加速
---

# Causality Matters: How Temporal Information Emerges in Video Language Models

**会议**: AAAI 2026  
**arXiv**: [2508.11576](https://arxiv.org/abs/2508.11576)  
**作者**: Yumeng Shi, Quanyu Long, Yin Wu, Wenya Wang (NTU)
**代码**: [github.com/ANDgate99/Causality-Matters](https://github.com/ANDgate99/Causality-Matters)  
**领域**: video_understanding  
**关键词**: 视频语言模型, 时序理解, 因果注意力, 位置编码, 模型可解释性, 推理加速

## 一句话总结

通过系统性消融实验揭示VideoLM的时序理解能力并非来源于位置编码(PE)，而是由因果注意力掩码的序列敏感性产生——时序信息沿"帧间交互→末帧聚合→query融合"的因果路径逐层构建，并据此提出两种无损推理加速策略。

## 研究背景与动机

### 问题背景

视频语言模型（VideoLM）在视频问答、描述生成、时序推理等任务上取得显著进展。其中**时序理解**——即识别事件的先后顺序、持续时间和跨时间关系——是视频理解的核心挑战。从因果推理到长视频叙事连贯性，时序理解对智能视频-语言交互至关重要。

### 已有工作的主流假设及其问题

主流研究普遍认为**位置编码(PE)是VideoLM时序建模的核心机制**，由此衍生出大量PE改进工作：Qwen2.5-VL采用三维PE、V2PE设计变速率编码、VideoRoPE扩展至高维空间等。然而这一假设从未被严格验证——位置编码到底在多大程度上支撑了时序理解？如果PE并非关键，那真正的时序信息来源又是什么？

### 核心动机

本文的出发点是一个反直觉的初始观察：**移除或修改视频输入的PE对时序理解性能影响极小，而反转帧序列（保持PE不变）则导致性能大幅下降**。这意味着模型对帧的物理顺序高度敏感，却不依赖PE来感知这种顺序。由此提出核心假设：时序理解是因果注意力机制的**涌现现象**，源于token间的序列敏感交互，而非显式的位置信号。

### 研究目标

1. 量化PE对时序理解的实际贡献
2. 追踪时序信息在模型内部的产生、流动和聚合路径
3. 验证该路径的因果性（而非单纯相关性）
4. 基于机制洞察提出实用的推理加速策略

## 方法详解

### 实验框架设计

研究以Qwen2.5-VL-7B和LLaVA-OneVision-7B为主要分析对象，在TempCompass基准（覆盖动作、速度、方向、属性变化、事件顺序等时序维度）上展开。评估指标为扰动后ground-truth token概率的变化$P_C = \tilde{P}_\text{next}(t_\text{gt}) - P_\text{next}(t_\text{gt})$，直接衡量扰动对时序推理的影响程度。

### 第一阶段：位置编码的角色解构

**逐层PE消融**：在每一层单独移除PE项，考察各层对PE的依赖程度。结果发现：仅第一层移除PE会导致正确答案概率大幅下降（最高达60%），中间层和深层移除PE的影响几乎可以忽略（通常在±2%以内）。这表明PE的作用被严格限制在模型的第一层。

**模态特异性分析**：在第一层分别打乱视频token和query token的position ID。打乱query位置ID导致性能显著下降（Multiple Choice任务降53%），而打乱视频位置ID的影响不到0.1%。这意味着即便在PE起作用的第一层，其效果也集中在文本模态而非视频模态。

**PE vs. 帧序列对比**：反转所有层的position ID（保持帧序不变）仅引起1-2%的性能损失；而反转帧序列（保持position ID不变）导致Multiple Choice任务下降15.45%。这构成了本文最关键的实证：**帧的物理顺序远比位置编码重要**。

### 第二阶段：因果信息路径的追踪

本文采用**注意力敲除(attention knockout)**方法系统追踪时序信息的流动路径。具体做法是在注意力计算中引入额外掩码$M$，阻止特定token集合$\mathcal{T}$关注源token集合$\mathcal{S}$，以滑动窗口$k=5$层为单位逐步扫描。

**发现1：Query驱动的最终预测**。阻止最终输出token关注视频token对性能几乎无影响，而阻止其关注query token导致超过40%的性能下降。说明在最终解码阶段，模型主要从query而非直接从视频中获取时序信息——query已经在前面的层中完成了时序信息的吸收。

**发现2：两阶段时序处理**。分别阻断帧间注意力和帧→query注意力，发现两者在不同层产生影响：帧间注意力阻断在第10层附近引起峰值退化（4-6%），帧→query阻断在第15层附近引起更大退化（超过10%）。这揭示了清晰的两阶段流程——**早期层构建帧间时序关系，中间层将时序信息整合到query表示中**。

**发现3：末帧聚合效应**。将query限制为只关注单个帧时，仅关注Frame 1导致Multiple Choice任务下降超8%，而仅关注Frame 4（最后一帧）的性能损失不到2%。这说明时序信息在帧间逐步前向累积，最终帧作为聚合节点汇总了全部时序线索。

**发现4：因果性验证**。在Yes/No任务中反转帧序列后，归因模式完全翻转：原本贡献最大的末帧变成损害最大的帧，原本损害最大的首帧反而提升性能。即使同时反转position ID（消除PE混淆因素），该翻转模式依然成立。这证明模型执行的是**序列敏感的因果推理**，而非简单的视觉信息累积。

**发现5：时空组装机制**。在帧间注意力的空间维度上，稀疏的长程对应区域注意力（每帧只关注前序帧的对应空间位置）比密集的短程注意力（每帧关注紧邻前帧的全部token）效果更好。早期层偏好全局空间上下文，深层转向局部细节处理。

### 第三阶段：推理加速策略

基于上述机制洞察，提出两类加速方案：

1. **分阶段稀疏注意力**：在中间层(10-20层)限制query只关注末帧视频token，在深层(20-28层)禁用帧间注意力，降低FLOPs
2. **时序退出机制(Temporal Exit)**：在第20层后从KV cache中移除所有帧token，减少GPU显存占用

## 实验关键数据

### 表1：PE反转 vs. 帧序反转对时序理解的影响（Qwen2.5-VL-7B, TempCompass）

| 扰动方式 | Yes/No | Multiple Choice | Captioning | Caption Matching |
|---------|--------|----------------|------------|-----------------|
| 反转Position ID | -1.13% | -1.42% | -1.76% | -1.55% |
| **反转帧序列** | **-10.28%** | **-15.45%** | **-8.73%** | **-11.82%** |

帧序反转的性能下降是PE反转的7-10倍，有力证明时序理解依赖帧物理顺序而非位置编码。

### 表2：推理加速策略在TempCompass和NExT-QA上的准确率

| 策略 | Yes/No | MCQ | Caption | Match | NExT-QA |
|------|--------|-----|---------|-------|---------|
| Baseline | 70.8 | 66.5 | 59.0 | 57.4 | 75.1 |
| (1) Query仅关注末帧(L10-20) | 67.9 | 64.1 | 57.6 | 55.0 | 71.9 |
| (2) 禁用帧间注意力(L20-28) | 70.5 | 66.5 | 59.3 | 57.4 | 75.2 |
| (3) 移除帧KV cache(L20+) | 70.7 | 66.6 | 59.0 | 57.2 | 75.1 |

策略(2)和(3)在所有任务上与baseline几乎无差（差异<0.3%），策略(3)甚至在MCQ上微超baseline。这验证了深层帧间交互和帧token存储确实可被安全剪除。

## 亮点与洞察

- **颠覆性发现**：推翻了"PE是时序理解核心"的社区共识，证明因果注意力掩码才是时序建模的真正来源，为PE研究方向提供了重要修正信号
- **完整的机制图景**：首次系统刻画了VideoLM内部时序信息的完整因果路径——帧间构建→末帧汇聚→query吸收→独立解码，层次清晰、因果关系经过验证
- **跨模型一致性**：在架构差异显著的三个模型（Qwen2.5-VL-7B/3B、LLaVA-OneVision）上结论一致，增强了普适性
- **机制驱动的应用**：推理加速策略并非启发式设计，而是直接由机制分析推导得出，逻辑闭环完整
- **方法论价值**：attention knockout + backward tracing的分析框架可推广到其他模态理解问题的机制研究中

## 局限性

- **模型覆盖范围有限**：仅分析了两个模型系列，未涉及非因果架构（如双向注意力或encoder-decoder结构），结论对非自回归模型的适用性未知
- **帧数受限**：实验仅使用4-8帧输入，长视频（数十到数百帧）场景下末帧聚合机制是否依然成立有待验证
- **加速策略初步**：提出的加速方案层区间为手动设定（如L10-20、L20-28），缺乏自适应机制，且策略(1)有3%左右的性能损失
- **未涉及训练侧**：分析聚焦推理阶段的注意力机制，未探讨训练过程中PE对学习时序表征的影响，论文也承认PE在训练阶段可能仍然重要
- **任务类型限制**：主要在判断/选择类任务上验证，开放生成任务（ActivityNet-QA）上策略(3)性能下降较多（42.7% vs 46.9%），说明结论对生成任务的适用性需进一步研究

## 相关工作对比

- **PE设计方向（V2PE, VideoRoPE, Qwen2.5-VL 3D-PE）**：本文从根本上质疑了这条技术路线的有效性——若PE对视频时序理解贡献微弱，则精心设计PE可能并非正确方向
- **T3 (Li et al.)**：通过合成文本数据迁移时序技能到VideoLM，属于数据驱动方法；本文则从机制角度揭示时序能力的内在来源
- **Basu et al. (2024)**：用因果追踪分析多模态模型的视觉信息流动，但仅限于单图像场景；本文首次将此类分析扩展到视频时序维度
- **Zhang et al. (2025)**：分析跨模态信息流的工作，本文借鉴了其backward tracing思路并适配为注意力敲除方案

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次系统揭示VideoLM时序理解的内在机制，颠覆PE中心论
- 实验充分度: ⭐⭐⭐⭐ — 多模型多任务验证，消融设计精巧，但帧数和视频时长覆盖有限
- 写作质量: ⭐⭐⭐⭐⭐ — 逻辑链清晰递进，从现象到机制到应用环环相扣
- 价值: ⭐⭐⭐⭐ — 为VideoLM时序建模提供全新认知框架，加速策略实用但尚需完善

<!-- RELATED:START -->

## 相关论文

- [Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models](listening_between_the_frames_bridging_temporal_gaps_in_large_audio-language_mode.md)
- [ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding](reason_reinforced_causal_search_with_information_bottleneck_for_video_understand.md)
- [Understanding Temporal Logic Consistency in Video-Language Models through Cross-Modal Attention Discriminability](../../CVPR2026/video_understanding/understanding_temporal_logic_consistency_in_video-language_models_through_cross-.md)
- [TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)
- [On the Consistency of Video Large Language Models in Temporal Comprehension](../../CVPR2025/video_understanding/on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)

<!-- RELATED:END -->
