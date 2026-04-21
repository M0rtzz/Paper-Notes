---
title: >-
  [论文解读] Video Summarization with Large Language Models
description: >-
  [CVPR 2025][视频理解][视频摘要] LLMVS 提出一个基于 LLM 的视频摘要框架，先用多模态 LLM 将视频帧转换为文本描述，再用 LLM 通过滑动窗口上下文学习评估每帧的局部重要性分数，最后通过全局自注意力机制聚合全局上下文生成最终预测，在 SumMe 和 TVSum 上取得了 SOTA 性能。
tags:
  - CVPR 2025
  - 视频理解
  - 视频摘要
  - 大语言模型
  - 关键帧选择
  - 局部到全局
  - 上下文学习
---

# Video Summarization with Large Language Models

**会议**: CVPR 2025  
**arXiv**: [2504.11199](https://arxiv.org/abs/2504.11199)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 视频摘要, 大语言模型, 关键帧选择, 局部到全局, 上下文学习

## 一句话总结

LLMVS 提出一个基于 LLM 的视频摘要框架，先用多模态 LLM 将视频帧转换为文本描述，再用 LLM 通过滑动窗口上下文学习评估每帧的局部重要性分数，最后通过全局自注意力机制聚合全局上下文生成最终预测，在 SumMe 和 TVSum 上取得了 SOTA 性能。

## 研究背景与动机

**领域现状**：视频内容呈指数级增长，高效的视频导航、搜索和检索需要先进的视频摘要技术。现有方法主要分为两类：纯视觉方法（VASNet、DSNet、CSTA 等）基于视觉特征和时间动态选择关键帧；多模态方法（CLIP-It、A2Summ 等）整合视觉和文本特征，但文本仅作为辅助增强视觉表示。

**现有痛点**：纯视觉方法主要依赖视觉显著性，无法捕捉视频内容的语义信息，导致摘要不完整或不连贯。多模态方法虽然引入了文本，但仍以视觉为核心——文本特征作为 cross-attention 的 key/value 来增强视觉查询，本质上还是"看哪些帧显眼"而非"理解哪些帧重要"。

**核心矛盾**：视频摘要的"什么是关键帧"本质上是主观的、语义驱动的——需要理解内容的叙事结构和信息价值，而非仅靠视觉显著性。传统方法缺乏深层语义理解能力。

**本文目标**：利用 LLM 从海量数据中学到的知识来评估视频帧的重要性，期望 LLM 的判断能更好地与人类对关键帧的多样化认知对齐。

**切入角度**：LLM 擅长上下文理解和跨域推理，是天然的"重要性评估器"。但 LLM 不能直接处理视频，需要将视觉信号转换为文本空间。

**核心 idea**：将视频帧翻译为文本描述后，利用 LLM 的上下文学习在局部窗口内评估帧重要性，再用可训练的全局自注意力模块在视频整体上下文中精化预测——关键是用 LLM 的输出嵌入（而非最终答案）作为中间表示。

## 方法详解

### 整体框架

输入为视频帧序列 $\mathbf{F} = [F_1, ..., F_T]$，输出为每帧的重要性分数 $\mathbf{s} \in \mathbb{R}^{T \times 1}$。流程分三步：（1）用冻结的 M-LLM（LLaVA-1.5-7B）为每帧生成一句话文字描述；（2）用冻结的 LLM（Llama-2-13B-chat）在局部窗口内通过上下文学习评估中心帧的重要性，提取输出嵌入；（3）将所有帧的嵌入送入可训练的全局自注意力模块输出最终分数。M-LLM 和 LLM 均冻结不训练，只训练自注意力块。

### 关键设计

1. **文本描述生成（M-LLM→文本空间转换）**:

    - 功能：将视频帧从视觉空间转换到文本空间，使 LLM 能够处理
    - 核心思路：用冻结的 LLaVA-1.5-7B 对每帧进行描述，prompt 为"Provide a detailed one-sentence description"，限制每帧最多 77 tokens。输出为文本序列 $\mathbf{C} = \phi(\mathbf{F})$
    - 设计动机：直接让 M-LLM 做重要性评分不如显式生成描述再交给专用 LLM 处理效果好（消融实验证实）。通用简单的描述比分区域描述（中心+背景）效果更好，因为整体描述更容易捕捉场景动态

2. **局部重要性评分（LLM + 滑动窗口上下文学习）**:

    - 功能：在局部时间上下文中评估每帧的相对重要性
    - 核心思路：对时间步 $t$ 的帧，取窗口大小 $w=7$ 内的描述 $C_{t-3:t+3}$ 送入 Llama-2-13B-chat，通过上下文学习（提供 3 个示例的 instruction-example-query prompt）输出中心帧重要性。**关键创新**：不使用 LLM 的最终文本答案，而是从 RMS Norm 层后提取查询嵌入 $\mathbf{q}_t \in \mathbb{R}^{L^q \times D}$ 和答案嵌入 $\mathbf{a}_t \in \mathbb{R}^{L^a \times D}$，因为中间嵌入保留了更丰富的上下文和语义信息
    - 设计动机：视频帧高度冗余，需要在局部上下文中比较才能筛选出关键帧。使用嵌入而非 LLM 最终答案的直觉是——最终答案是高度压缩的信息（一个数字），而嵌入保留了 LLM 内部推理的全部细节。消融实验证实嵌入方法显著优于直接使用 LLM 答案

3. **全局上下文聚合（Self-Attention Blocks）**:

    - 功能：从视频整体上下文出发，精化局部重要性分数以产生连贯的摘要
    - 核心思路：将每帧的查询嵌入和答案嵌入拼接 $\mathbf{x}_t = \text{concat}(\mathbf{q}_t, \mathbf{a}_t)$，经 max pooling 和 MLP 降维到 $\mathbb{R}^{1 \times M}$（$M=2048$）。所有 $T$ 帧的嵌入组成序列 $\mathbf{x}' \in \mathbb{R}^{T \times M}$，通过 3 层 2 头的自注意力模块 $\psi$ 建模全局依赖关系，再经 MLP 输出最终分数：$\mathbf{s} = \text{MLP}(\psi(\mathbf{x}'))$
    - 设计动机：LLM 的局部窗口无法感知全局叙事结构。例如，一个在局部看起来不重要的帧可能在全局中是转折点。自注意力机制让模型能捕捉跨窗口的依赖关系。只训练这个轻量级模块（3 层 SA），保留 LLM 的通用知识

### 损失函数 / 训练策略

使用 MSE 损失：$\mathcal{L} = \frac{1}{T}\sum_{t=1}^{T}(s_t - \hat{s}_t)^2$。AdamW 优化器，200 epochs，5 张 A100，batch size 1，学习率 SumMe 1.19e-4 / TVSum 7e-5。M-LLM 和 LLM 完全冻结，仅训练全局注意力模块和 MLP。总训练约 10 小时。

## 实验关键数据

### 主实验

| 方法 | 类型 | SumMe τ↑ | SumMe ρ↑ | TVSum τ↑ | TVSum ρ↑ |
|------|------|----------|----------|----------|----------|
| Human | - | 0.205 | 0.213 | 0.177 | 0.204 |
| CSTA | 视觉 | 0.246 | 0.274 | 0.194 | 0.255 |
| MSVA | 视觉 | 0.200 | 0.230 | 0.190 | 0.210 |
| SSPVS | 视觉+文本 | 0.192 | 0.257 | 0.181 | 0.238 |
| LLM (zero-shot) | LLM | 0.170 | 0.189 | 0.051 | 0.056 |
| **LLMVS** | **LLM** | **0.253** | **0.282** | **0.211** | **0.275** |

### 消融实验

| 配置 | SumMe τ↑ | SumMe ρ↑ |
|------|----------|----------|
| LLaVA 直接评分 (无LLM) | 0.119 | 0.132 |
| LLaVA* 微调直接评分 | 0.140 | 0.156 |
| LLaVA→Llama (无全局聚合) | 0.170 | 0.189 |
| LLaVA→Llama* 微调 (无全局聚合) | 0.181 | 0.201 |
| **LLaVA→Llama + SA* (LLMVS)** | **0.253** | **0.282** |

嵌入选择消融：

| 配置 | τ↑ | ρ↑ |
|------|-----|-----|
| 仅答案嵌入 $\mathbf{a}$ + SA | 0.233 | 0.260 |
| 仅查询嵌入 $\mathbf{q}$ + SA | 0.238 | 0.265 |
| **$\mathbf{q} + \mathbf{a}$ + SA** | **0.253** | **0.282** |
| $\mathbf{q} + \mathbf{a}$ + MLP (无SA) | 0.182 | 0.203 |

### 关键发现

- 全局自注意力是最大贡献模块：从零样本 LLM（0.170）到 LLMVS（0.253）的提升主要来自全局聚合器，证明局部窗口不足以产生连贯摘要
- LLM 的输出嵌入远优于直接答案：使用嵌入的 LLMVS 显著优于仅用 LLM 最终数字答案的版本，中间表示确实保留了更多有用信息
- LLM 零样本在 SumMe 上有竞争力（0.170 vs 人类 0.205），但在 TVSum 上很差（0.051），因为 TVSum 针对每个标注者分别评估——LLM 适合通用摘要但不擅长捕捉个人偏好
- 查询嵌入比答案嵌入略好（0.238 vs 0.233），说明窗口内所有帧的上下文比最终评估更有信息量
- 数值评分 prompt 优于文本总结 prompt（0.253 vs 0.239），直接的重要性评分任务更明确
- 从 RMS Norm 层提取嵌入优于从 Linear 层（0.253 vs 0.241）

## 亮点与洞察

- **"用嵌入而非答案"的核心洞察**：LLM 的能力不仅体现在最终输出上，中间层的表示包含了更丰富的推理过程信息。这个发现可迁移到任何需要利用 LLM 做判断的下游任务——不要只看答案，用嵌入
- **极简架构设计**：M-LLM 和 LLM 完全冻结，只训练 3 层自注意力块，10 小时完成训练。证明了"检索 LLM 知识+轻量适配"的范式在视频理解中也有效
- **语言为中心的视频理解**：将视频摘要重新框架化为语言理解问题，让 LLM 而非视觉模型成为"主角"。这与传统多模态方法（文本辅助视觉）形成鲜明反差

## 局限与展望

- 文字描述必然丢失部分视觉信息，对纯视觉驱动的重要性（如美学、构图）可能不敏感
- 滑动窗口的 token 成本较高——每帧需要一次完整的 LLM 前向推理，长视频的推理开销大
- 仅在 SumMe 和 TVSum 两个较小的 benchmark 上评估，这两个数据集规模有限且广泛使用，可能存在过拟合风险
- 上下文学习的 3 个示例来自训练集随机采样，示例质量可能影响 LLM 的评估
- 冻结 LLM 限制了模型适应数据集特性的潜力——微调 LLM 可能进一步提升

## 相关工作与启发

- **vs CSTA**: 基于 CNN 的视觉方法，将帧序列作为图像处理并用 2D CNN 提取时空注意力；LLMVS 用语言理解代替视觉分析，两者互补
- **vs CLIP-It**: 用 CLIP 的 cross-attention 融合视觉和文本特征，文本作为辅助；LLMVS 反转这一范式，以语言为中心
- **vs AntGPT/MovieChat**: 也利用 LLM 做视频理解，但用于问答或动作预测；LLMVS 首次将 LLM 用于帧级重要性评估

## 评分

- 新颖性: ⭐⭐⭐⭐ 用 LLM 嵌入做视频摘要的思路新颖，局部-全局两阶段设计合理
- 实验充分度: ⭐⭐⭐⭐ 消融实验非常详尽（嵌入选择、prompt策略、提取位置等），但仅两个小数据集
- 写作质量: ⭐⭐⭐⭐ 文章结构清晰，图表直观，方法描述详实
- 价值: ⭐⭐⭐⭐ 展示了 LLM 在视频摘要中的有效性，"嵌入优于答案"的发现有广泛启发意义

<!-- RELATED:START -->

## 相关论文

- [VoCo-LLaMA: Towards Vision Compression with Large Language Models](voco-llama_towards_vision_compression_with_large_language_models.md)
- [PAVE: Patching and Adapting Video Large Language Models](pave_patching_and_adapting_video_large_language_models.md)
- [On the Consistency of Video Large Language Models in Temporal Comprehension](on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)
- [FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [DisTime: Distribution-based Time Representation for Video Large Language Models](../../ICCV2025/video_understanding/distime_distribution-based_time_representation_for_video_large_language_models.md)

<!-- RELATED:END -->
