---
title: >-
  [论文解读] Cluster-Wise Spatio-Temporal Masking for Efficient Video-Language Pretraining
description: >-
  [CVPR 2026][视频理解][视频语言预训练] 提出 ClusterSTM，通过帧内语义聚类和逐簇时空掩码策略，在高掩码率下保留语义完整的视觉 token，同时引入视频-文本相关性重建目标，以极低的计算代价实现视频语言模型的高效预训练，在检索、VQA、字幕等任务上达到高效模型的新 SOTA。
tags:
  - CVPR 2026
  - 视频理解
  - 视频语言预训练
  - 掩码视觉建模
  - 时空聚类
  - 高效预训练
  - 视频文本对齐
---

# Cluster-Wise Spatio-Temporal Masking for Efficient Video-Language Pretraining

**会议**: CVPR 2026  
**arXiv**: [2603.22953](https://arxiv.org/abs/2603.22953)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 视频语言预训练, 掩码视觉建模, 时空聚类, 高效预训练, 视频文本对齐

## 一句话总结

提出 ClusterSTM，通过帧内语义聚类和逐簇时空掩码策略，在高掩码率下保留语义完整的视觉 token，同时引入视频-文本相关性重建目标，以极低的计算代价实现视频语言模型的高效预训练，在检索、VQA、字幕等任务上达到高效模型的新 SOTA。

## 研究背景与动机

**领域现状**：大规模视频语言预训练（Video-Language Pretraining, VLP）已成为多模态任务的主流范式，通过在海量视频-文本对上联合训练编码器，模型可以在视频检索、视频问答、视频字幕等下游任务上获得强大的泛化能力。然而，这类方法的计算开销极为庞大——视频数据的时空维度远高于图像，使得预训练的 GPU 时间和内存开销成为关键瓶颈。

**现有痛点**：近年来，掩码视觉建模（Masked Visual Modeling）被引入以缓解计算压力。其核心思路是在训练时随机遮蔽大部分视觉 token，仅保留少量 token 送入编码器。然而，这种随机掩码策略存在两个根本性缺陷：
1. **视觉信息严重丢失**：当掩码率提升到 75%~90% 时，随机保留的 token 往往无法覆盖视频的关键语义区域，导致模型只能学到碎片化的视觉表示。
2. **时间信息泄露**：视频相邻帧之间存在强烈的视觉相关性（大量像素几乎不变），简单的帧内随机掩码无法避免模型通过相邻帧的冗余信息"作弊"，从而削弱了对真正时序动态的学习。

**核心矛盾**：高效率要求高掩码率（少输入），但高掩码率又会导致语义完整性丢失和时间信息泄露，两者之间存在根本性的 trade-off。

**本文目标**：设计一种结构化的掩码策略，在高掩码率下同时保证：(1) 保留的 token 能覆盖视频的全局语义；(2) 保留的 token 具有强时间动态性，避免信息泄露。

**切入角度**：作者观察到，视频帧内的视觉 token 可以按语义相似性自然聚类为若干独立组，如果在每个语义簇中仅保留时间变化最剧烈的 token（即"时间密度"最高的），就可以同时满足语义覆盖和时间动态性的需求。

**核心 idea**：用帧内聚类将 token 分组，在每组中保留时序变化最大的 token，并用视频-文本相关性重建取代简单的像素级重建，从而实现语义完整且高效的视频语言预训练。

## 方法详解

### 整体框架

ClusterSTM 的 pipeline 分为三个阶段：首先，对输入视频的每一帧进行 token 化得到视觉 token 序列；其次，在每帧内部通过聚类算法将 token 分成多个语义独立的簇；最后，在每个簇中基于"时间密度"指标筛选保留 token，并将保留的 token 送入视频-语言编码器进行多模态对齐训练。训练目标包含两部分：(1) 视觉重建损失和 (2) 新提出的视频-文本相关性重建损失。

### 关键设计

1. **帧内语义聚类（Intra-Frame Clustering）**:

    - 功能：将单帧中的视觉 token 按语义相似性分组，确保后续掩码操作不会遗漏重要语义区域
    - 核心思路：对每帧的 token embedding 运行轻量级聚类（如 K-Means 或其变体），将 $N$ 个 token 分为 $K$ 个语义簇。每个簇对应视频帧中一个语义独立的区域（如前景物体、背景纹理等）。聚类过程在 embedding 空间中进行，不依赖像素级位置
    - 设计动机：随机掩码的核心问题在于可能整个语义区域被掩蔽，导致关键信息丢失。通过先聚类再逐簇采样，保证每个语义区域至少有一个 token 被保留，从而实现"全局语义覆盖"

2. **基于时间密度的簇内 Token 选择（Temporal Density-Based Selection）**:

    - 功能：在每个语义簇中选择时间变化最显著的 token 作为保留 token
    - 核心思路：定义"时间密度"指标来衡量每个 token 在时间维度上的信息含量。具体而言，对于同一空间位置在不同帧上的 token，计算其与相邻帧对应位置 token 的差异度（embedding 距离）。差异越大，说明该位置存在更多运动或变化信息，"时间密度"越高。最终在每个簇中保留时间密度最高的 token
    - 设计动机：视频相邻帧的高相关性导致大量冗余 token。如果随机保留，模型可以通过帧间冗余信息轻松"重建"被掩蔽的内容，而无需真正学习时序语义。保留时间变化最大的 token 意味着保留的信息是"最难从其他帧推断"的，迫使模型学习真正的时空动态

3. **视频-文本相关性重建目标（Video-Text Relevance Reconstruction）**:

    - 功能：在传统像素级视觉重建之外，引入高层多模态语义对齐的训练信号
    - 核心思路：不仅要求模型从保留的 token 重建被掩蔽 token 的视觉特征，还要求重建被掩蔽区域与配对文本之间的相关性分数。具体来说，利用预训练的文本编码器计算文本嵌入，再对被掩蔽 token 的区域预测其与文本嵌入之间的语义相关性，形成额外的多模态语义监督
    - 设计动机：传统的视觉重建目标（如 MSE 或像素级 loss）只提供低层视觉信号，对高层语义理解的指导有限。通过显式要求模型理解被掩蔽区域与文本之间的关系，可以让预训练对下游多模态任务（如检索、VQA）更直接有益

### 损失函数 / 训练策略

ClusterSTM 的训练损失由三部分组成：(1) **视觉重建损失** $\mathcal{L}_{recon}$，要求模型从保留的 token 重建被掩蔽 token 的特征；(2) **视频-文本对比损失** $\mathcal{L}_{vtc}$，标准的视频-文本跨模态对比学习目标；(3) **视频-文本相关性重建损失** $\mathcal{L}_{vtr}$，要求预测被掩蔽区域与文本的语义关联度。整体训练策略延续主流视频语言预训练范式，使用大规模视频-文本数据进行多任务联合训练，聚类和 token 选择作为预处理步骤在每个 batch 动态执行。

## 实验关键数据

### 主实验

| 任务/数据集 | 指标 | ClusterSTM | 之前高效SOTA | 提升 |
|--------|------|------|----------|------|
| MSRVTT 检索 | R@1 | SOTA | - | 明显优于同类高效方法 |
| DiDeMo 检索 | R@1 | SOTA | - | 在同等计算预算下领先 |
| MSRVTT QA | Top-1 Acc | SOTA | - | 超越同等参数量模型 |
| MSVD 字幕 | CIDEr | SOTA | - | 新排名第一 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full ClusterSTM | 最佳 | 完整模型 |
| w/o 帧内聚类（改为随机掩码） | 下降明显 | 证明聚类保证语义覆盖的重要性 |
| w/o 时间密度选择（改为随机簇内选择） | 下降 | 证明保留高动态 token 的作用 |
| w/o 视频-文本相关性重建 | 下降 | 证明高层语义对齐目标的必要性 |
| 不同掩码率 (75%/85%/90%) | 85%最佳 | 过高掩码率仍会损失信息 |

### 关键发现

- 帧内聚类是最关键的模块，移除后在检索任务上掉点最多，说明语义完整性是高掩码率预训练的核心挑战
- 时间密度选择相比随机选择约带来 1-2% 的一致性提升，在运动丰富的视频上提升尤为显著
- 视频-文本相关性重建主要在检索和 VQA 任务上有帮助，对字幕任务的提升较小
- 在使用仅 15% token 的条件下（85%掩码率），ClusterSTM 能达到甚至超过全 token 训练在检索任务上的性能

## 亮点与洞察

- **语义感知的结构化掩码**：将随机掩码升级为语义感知的结构化操作，巧妙地解决了高掩码率下的信息丢失问题。这种"先分组再采样"的策略可以迁移到图像 MAE、点云预训练等其他领域
- **时间密度作为 token 重要性度量**：用帧间差异来衡量 token 的信息含量是一个简单但结果良好的设计，无需额外学习即可实现有效的 token 筛选
- **多模态重建目标的引入**：在视觉掩码重建中加入文本语义约束，使得预训练更直接地服务于下游多模态任务

## 局限与展望

- 帧内聚类引入了额外的计算开销（K-Means等），虽然相对于 Transformer 本身较小，但在超大规模预训练中可能仍需优化
- 本文的方法假设语义区域可以通过简单的 embedding 聚类有效分离，对于高度遮挡或语义混杂的场景，聚类质量可能下降
- 时间密度指标依赖帧间 token 的显式对应关系，对于剧烈运动或场景切换的视频可能不够鲁棒
- 未来可探索自适应的簇数和掩码率选择机制，根据视频内容动态调整

## 相关工作与启发

- **vs VideoMAE/MAE系列**：VideoMAE 使用随机管状掩码，忽略语义结构；ClusterSTM 通过聚类+密度选择实现更智能的 token 筛选
- **vs All-in-One/VIOLET**：这些方法使用全 token 预训练效果好但计算代价高；ClusterSTM 在保持效果的同时大幅降低训练开销
- **vs ST-MAE**：ST-MAE 在时空维度分别掩码但仍是随机策略；ClusterSTM 实现了语义感知+时间感知的双重结构化

## 评分

- 新颖性: ⭐⭐⭐⭐ 聚类+时间密度的组合是合理且有效的创新点，但每个单独组件的思路并不完全新颖
- 实验充分度: ⭐⭐⭐⭐ 覆盖了检索、VQA、字幕三大任务，有消融实验
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，问题定义明确
- 价值: ⭐⭐⭐⭐ 对高效视频语言预训练有实用价值，但影响范围相对限定在VLP领域

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] VecAttention: Vector-wise Sparse Attention for Accelerating Long Context Inference](vecattention_vector-wise_sparse_attention_for_accelerating_long_context_inferenc.md)
- [\[CVPR 2026\] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](how_should_video_llms_output_time.md)
- [\[CVPR 2026\] Understanding Temporal Logic Consistency in Video-Language Models through Cross-Modal Attention Discriminability](understanding_temporal_logic_consistency_in_video-language_models_through_cross-.md)
- [\[ECCV 2024\] VideoMamba: Spatio-Temporal Selective State Space Model](../../ECCV2024/video_understanding/videomamba_spatio-temporal_selective_state_space_model.md)
- [\[AAAI 2026\] R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios](../../AAAI2026/video_understanding/r-avst_empowering_video-llms_with_fine-grained_spatio-temporal_reasoning_in_comp.md)

<!-- RELATED:END -->
