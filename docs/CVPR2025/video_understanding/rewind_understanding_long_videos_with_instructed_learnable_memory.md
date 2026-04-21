---
title: >-
  [论文解读] ReWind: Understanding Long Videos with Instructed Learnable Memory
description: >-
  [CVPR 2025][视频理解][长视频理解] 本文提出 ReWind，一种基于可学习记忆模块的视觉语言模型架构，通过新颖的 read-perceive-write 循环机制和指令引导的动态帧选择，在使用更少 token 和帧的情况下，大幅超越先前方法在长视频 VQA 和时序定位任务上的表现。
tags:
  - CVPR 2025
  - 视频理解
  - 长视频理解
  - 可学习记忆
  - 视觉语言模型
  - 动态帧选择
  - 视频问答
---

# ReWind: Understanding Long Videos with Instructed Learnable Memory

**会议**: CVPR 2025  
**arXiv**: [2411.15556](https://arxiv.org/abs/2411.15556)  
**代码**: 无  
**领域**: 视频理解 / 多模态VLM  
**关键词**: 长视频理解, 可学习记忆, 视觉语言模型, 动态帧选择, 视频问答

## 一句话总结

本文提出 ReWind，一种基于可学习记忆模块的视觉语言模型架构，通过新颖的 read-perceive-write 循环机制和指令引导的动态帧选择，在使用更少 token 和帧的情况下，大幅超越先前方法在长视频 VQA 和时序定位任务上的表现。

## 研究背景与动机

**领域现状**：视觉语言模型（VLMs）在多模态理解领域取得了显著进展，能够将文本和视觉信息整合进行问答、描述生成等任务。然而，现有 VLMs 在处理长视频（10 分钟以上）时面临严峻挑战，主要源于自注意力机制的二次方计算复杂度、内存限制，以及在长序列上维持连贯时序理解的困难。

**现有痛点**：(1) **信息过度压缩**——MovieChat、MA-LMM 等方法使用 FIFO 队列或层次记忆模块，但严重压缩时序信息，牺牲了对事件动态的准确理解；(2) **孤立帧处理**——LLaMA-VID、VTimeLLM 等方法独立处理每一帧，无法捕获跨帧的连贯时序表示；(3) **固定密度表示**——现有方法对所有帧保持相同的空间表示密度，对非关键帧存储了不必要的细节，浪费内存资源。

**核心矛盾**：长视频理解需要同时做到两件事——(a) 高效压缩存储整个视频的时序信息，(b) 对关键时刻保留足够的空间细节。但现有方法要么过度压缩丢失细节，要么均匀保留导致内存爆炸。

**本文目标**：设计一种记忆驱动的长视频理解框架，能够根据用户指令选择性地存储和检索视频信息，在低内存开销下实现高质量的视频问答和时序定位。

**切入角度**：作者提出"先粗后细"的两阶段处理策略：第一阶段通过记忆模块渐进式压缩视频信息（少量 token 表示每帧），第二阶段"回放"视频，选择指令相关的关键帧保留高空间分辨率细节。

**核心 idea**：用 read-perceive-write 循环的记忆模块渐进式构建指令感知的压缩视频表示，再通过指令引导的动态帧选择"倒带"到关键时刻补充细节。

## 方法详解

ReWind 将长视频理解分为两个阶段。第一阶段（Stage-1），视频被分割为子片段逐段处理，每个子片段经过视觉编码器提取特征后，通过一个带有记忆模块的 perceiver 层进行指令引导的编码和存储。第二阶段（Stage-2），基于记忆内容和用户指令，动态选择关键帧保留高分辨率空间信息，最后将记忆内容和选中帧一起送入 LLM 生成回答。

### 整体框架

输入是一整段长视频 V（如 10 分钟，以 1fps 采样得到 ~600 帧）和用户的文本指令。首先将视频分为 N 个子片段，每个包含 F 帧。Stage-1：对每个子片段逐帧提取 ViT 特征，经 read-perceive-write 循环更新记忆库 M（每帧只存 2 个 token）。Stage-2：基于 M 的内容和指令编码，用两阶段选择机制（指令相关性排序 + KNN 密度峰聚类）选出 8 个关键帧，从特征缓冲区取出这些帧的高分辨率表示（每帧 32 个 token）。最后，将 M 的内容、关键帧特征和指令拼接后送入 LLaMA-2 7B（LoRA 微调）生成回答。

### 关键设计

1. **Read-Perceive-Write 记忆循环**:

    - 功能：渐进式构建指令感知的压缩视频表示
    - 核心思路：三步循环——(a) **Read**：用 $N_R=32$ 个可学习读查询 $Q_R$ 通过 cross-attention 从记忆库 M 中检索历史上下文，获取当前记忆的摘要；(b) **Perceive**：将读查询（携带历史信息）作为 perceiver 的初始查询，与当前帧的 ViT 特征 + 指令文本编码进行 cross-attention，产生指令感知的帧级表示 $\hat{Q}_{ij}$，随后在时间维度上进行 self-attention 捕获片段内的时序关系；(c) **Write**：用 2 个可学习写查询 $Q_W$ 通过 cross-attention 将 perceiver 输出蒸馏为极紧凑的每帧 2 token 表示，按时序存入 M。
    - 设计动机：与之前的 Q-Former 方法（如 Video-LLaMA）不同，ReWind 的 perceiver 在帧级别独立处理后再做时序注意力，保留了时序保真度而非产生片段级的压缩表示。读操作确保每次编码都"知道"前面已经存了什么，写操作用极少 token 实现高效存储。

2. **动态帧选择（DFS）**:

    - 功能：从整个视频中选出指令最相关的关键帧，补充高分辨率空间细节
    - 核心思路：两阶段选择——(a) **指令相关性选择**：计算指令编码 $\bar{I}$ 与记忆库 M 中每帧表示的 attention 分数，选出 top $L=64$ 帧；(b) **密度峰聚类**：对这 L 帧用 KNN 密度峰聚类（DPC-KNN）选出 $K_c=8$ 个最具代表性的帧。对每帧计算局部密度 $\sigma_l$ 和距离指标 $\rho_l$，取 $\sigma_l \times \rho_l$ 最大的 $K_c$ 帧作为最终选择。选中帧从特征缓冲区取出原始 ViT 特征，池化为 32 token/帧。
    - 设计动机：均匀采样会浪费大量 token 在不相关帧上。两阶段选择先缩小范围（从数百帧到 64 帧），再用聚类去除冗余，确保选中帧既与指令相关又彼此多样，覆盖视频的关键时刻。

3. **LLM 输入构建**:

    - 功能：将压缩的记忆内容和详细的关键帧表示组合为 LLM 的输入
    - 核心思路：拼接 M 的内容（渐进式时序信息）和 DFS 选出的帧表示 $\hat{Z}$（关键时刻的空间细节），中间用特殊 token $\tau$ 分隔：$\langle m_0, m_1, \dots, \tau, \hat{Z} \rangle$。这两部分提供互补信息：M 提供全局时序脉络，$\hat{Z}$ 提供局部空间细节。
    - 设计动机：仅用压缩记忆会丢失空间细节，仅用选中帧会缺少时序上下文。两者结合让 LLM 既能理解"发生了什么"，也能看清"具体怎样"。

### 损失函数 / 训练策略

两阶段训练：(1) **多模态预训练**：冻结除 perceiver 外的所有组件，用 SigLIP 对比损失对齐视觉和文本特征（100K 视频-字幕对）；(2) **指令微调**：启用记忆模块、DFS 和 LLM（LoRA rank=64, alpha=32），在视频指令数据上训练 100K 步。时序定位任务额外在 DiDemo 和 ActivityNet 上微调 15K 步。仅需 8×V100 GPU。

## 实验关键数据

### 主实验（长视频 VQA - MovieChat-1K）

| 模型 | #Frames | #Tokens | Global Acc | Global Score | Breakpoint Acc |
|------|---------|---------|------------|-------------|----------------|
| Video-LLaMA | 32 | 32 | 51.4 | 3.10 | 38.2 |
| MovieChat | 2048 | 8192 | 67.8 | 3.81 | 50.4 |
| **ReWind** | **548*** | **1184*** | **80.6** | **4.46** | **57.2** |

### 时序定位（Charades-STA）

| 模型 | R@0.3 | R@0.5 | R@0.7 | mIoU |
|------|-------|-------|-------|------|
| VTimeLLM | 51.0 | 27.5 | 11.4 | 31.2 |
| **ReWind** | **59.0** | **41.6** | **20.5** | **39.3** |

### 消融实验

| 配置 | Global Acc | Global Score |
|------|-----------|-------------|
| Baseline (64帧均匀采样, 无记忆) | 61.5 | 3.21 |
| + Memory | 提升显著 | 提升显著 |
| + Memory + DFS | **80.6** | **4.46** |

### 关键发现

- ReWind 在使用 MovieChat 约 1/8 token 数和 1/4 帧数的情况下，VQA 准确率提升 +13%（67.8% → 80.6%）
- 时序定位 mIoU 比 VTimeLLM 提升 +8%（31.2% → 39.3%）
- 在短视频 benchmark（VideoChatGPT）上同样达到最优平均分，说明方法不仅适用于长视频
- 记忆模块和 DFS 各自独立都能带来提升，两者结合效果最佳

## 亮点与洞察

- **read-perceive-write 循环设计精妙**：读操作让新帧编码感知历史；perceive 在帧级别保留时序保真度；写操作用极少 token 高效存储——三者协同实现了渐进式信息积累
- **"先粗后细"的两阶段策略**：用压缩记忆构建全局时序理解，再"倒带"到关键帧获取细节，这个思路符合人类观看长视频的认知模式
- **极高的 token 效率**：每帧仅用 2 个 token 存入记忆，但通过 DFS 对关键帧补充 32 token 的细节，实现了效率和精度的理想平衡
- **指令引导贯穿全程**：从 perceiver 编码到帧选择，用户指令始终参与信息筛选，确保存储和检索都与任务相关

## 局限与展望

- 目前以 1fps 采样，可能遗漏快速发生的事件
- DFS 的帧数（8 帧）是固定的，未来可以根据视频复杂度自适应调节
- 仅使用 LLaMA-2 7B，未探索更大规模 LLM 的效果
- 记忆模块的读写查询数量需要预设，缺乏自适应机制
- 可以考虑引入音频模态进一步增强长视频理解

## 相关工作与启发

- **MovieChat**：使用 FIFO 短期记忆 + 合并长期记忆，但过度压缩丢失时序动态
- **MA-LMM**：层次记忆模块，但同样存在压缩过度的问题
- **LLaMA-VID**：每帧仅 2 token 表示，但独立处理帧缺少时序建模
- **Q-Former (BLIP-2)**：在片段级别压缩信息，不如 ReWind 的帧级别处理精细
- **启发**：记忆模块 + 动态选择的范式可以推广到其他需要处理长序列的多模态任务（如长文档理解、多轮视频对话）

## 评分

- 新颖性：⭐⭐⭐⭐ — read-perceive-write 循环和两阶段框架设计新颖
- 实验充分度：⭐⭐⭐⭐⭐ — 长/短视频 VQA、时序定位、充分消融
- 写作质量：⭐⭐⭐⭐ — 结构清晰，方法描述详细
- 价值：⭐⭐⭐⭐⭐ — 长视频理解的显著突破，效率和精度兼顾

<!-- RELATED:START -->

## 相关论文

- [Goldfish: Vision-Language Understanding of Arbitrarily Long Videos](../../ECCV2024/video_understanding/goldfish_vision-language_understanding_of_arbitrarily_long_videos.md)
- [VideoLucy: Deep Memory Backtracking for Long Video Understanding](../../NeurIPS2025/video_understanding/videolucy_deep_memory_backtracking_for_long_video_understanding.md)
- [MLVU: Benchmarking Multi-task Long Video Understanding](mlvu_benchmarking_multi-task_long_video_understanding.md)
- [VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos](vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)
- [DrVideo: Document Retrieval Based Long Video Understanding](drvideo_document_retrieval_based_long_video_understanding.md)

<!-- RELATED:END -->
