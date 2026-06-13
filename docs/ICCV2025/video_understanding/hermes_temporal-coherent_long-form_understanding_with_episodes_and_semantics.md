---
title: >-
  [论文解读] HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics
description: >-
  [ICCV 2025][视频理解][长视频理解] 提出 HERMES 框架，通过情景压缩器 (ECO) 和语义检索器 (SeTR) 两个通用模块分别捕获视频的情景记忆和语义信息，既可作为独立系统达到 SOTA，也可即插即用地增强现有视频语言模型，同时降低推理延迟达 43% 和内存消耗达 46%。 长视频理解（分钟级至小时级）…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "长视频理解"
  - "情景压缩"
  - "语义检索"
  - "视频问答"
  - "即插即用模块"
---

# HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics

**会议**: ICCV 2025  
**arXiv**: [2408.17443](https://arxiv.org/abs/2408.17443)  
**代码**: [GitHub](https://github.com/joslefaure/HERMES)  
**领域**: 视频理解  
**关键词**: 长视频理解, 情景压缩, 语义检索, 视频问答, 即插即用模块

## 一句话总结

提出 HERMES 框架，通过情景压缩器 (ECO) 和语义检索器 (SeTR) 两个通用模块分别捕获视频的情景记忆和语义信息，既可作为独立系统达到 SOTA，也可即插即用地增强现有视频语言模型，同时降低推理延迟达 43% 和内存消耗达 46%。

## 研究背景与动机

长视频理解（分钟级至小时级）面临三大核心挑战：

**时间复杂度**：需处理成千上万帧，现有方法计算代价高昂。

**语义理解**：除了逐帧事件外，还需理解高层叙事结构和主题概念。

**内存约束**：在有限计算资源下同时解决上述两个问题极困难。

现有长视频方法多从短视频方法（池化、3D 卷积等）简单扩展而来，没有充分考虑长视频的独特性质。作者受**人类认知**启发，区分了视频中的两类信息：

- **情景信息 (Episodic)**：具体的、时间有序的事件序列（如"青少年在棒球场互动"）
- **语义信息 (Semantic)**：跨越具体时刻的高层主题和概念（如"青年体育文化"）

现有方法通常只关注其中一类，HERMES 的动机是**同时捕获两类信息**，实现更全面的长视频理解。

## 方法详解

### 整体框架

HERMES 采用流式处理架构：视频按窗口逐步输入冻结的 ViT-G/14 编码器提取特征，然后分别经过 ECO（情景压缩）和 SeTR（语义检索）处理，最后将两类表示拼接后送入冻结的 LLM（Vicuna-7B）生成文本输出。

形式化表述：$U = G(ECO(V, I), SeTR(V))$，其中 $U$ 为综合理解，$G$ 为整合函数。

### 关键设计

1. **ECO: 情景压缩器 (Episodic COmpressor)**

   维护一个最大容量为 $E$ 的内存缓冲区。每当新窗口特征 $\mathcal{W}_k$ 到来时：

    $\mathcal{M} = \begin{cases} \mathcal{M} \oplus \mathcal{W}_k & \text{if } \|\mathcal{M}\| + \|\mathcal{W}_k\| \leq E \\ \text{ECO}(\mathcal{M}, \mathcal{W}_k) & \text{otherwise} \end{cases}$

   压缩算法：(1) 拼接缓冲区和新特征 → (2) 找到余弦相似度最高的帧对 $(i^*, j^*)$ → (3) 取均值合并 → (4) 移除被合并帧 → 重复至帧数 $\leq E$。

   核心思想：最相似的帧包含冗余信息，合并它们在减少数据量的同时保留情景结构。最优 episode 数为 20。

2. **SeTR: 语义检索器 (Semantics reTRiever)**

   捕获分散在整个视频中的高层语义信息。给定特征张量 $F \in \mathbb{R}^{B \times N \times T \times C}$：

    - 归一化特征
    - 按步长 $k$ 分为两组：$X$（每第 $k$ 帧）和 $Y$（其余帧）
    - 计算 $X$ 和 $Y$ 之间的点积相似度
    - 将 $Y$ 中每帧合并到 $X$ 中最相似的帧

   效果：帧数从 $N$ 降至 $N/k$，最优 keep ratio 为 20%（即 $k=5$，减少 80% 的帧）。

3. **层级 Q-Former**

    - **情景 Q-Former**：对 ECO 输出的情景记忆执行自注意力 + 交叉注意力，再通过类似 ECO 的合并过程压缩 query
    - **层级 Frame-to-Sequence Q-Former**：先用 frame Q-Former 独立增强每帧语义，再用 video Q-Former 跨帧聚合

   最终两类 query 拼接，经线性投影后送入 LLM：$U = W[Q; Q_{sem}] + b$

### 损失函数 / 训练策略

标准的语言建模损失（交叉熵），支持零样本和全监督两种评估设置。ECO 和 SeTR 不需要额外学习即可作为即插即用模块，集成到现有模型时仅需极少适配。

## 实验关键数据

### 主实验

**MovieChat-1k 零样本 VQA**

| 模型 | Global Acc. | Global Score | Breakpoint Acc. |
|------|------------|-------------|----------------|
| MovieChat | 63.7 | 3.15 | 48.1 |
| Video-ChatGPT | 58.7 | 2.89 | 47.8 |
| Video-LLaMA | 56.3 | 2.72 | 45.8 |
| **HERMES** | **78.6** | **4.23** | **57.3** |
| **HERMES (全监督)** | **84.9** | **4.40** | **65.8** |

相比前 SOTA 提升 **+14.9%** (Global Acc.)。

**LVU + Breakfast + COIN 分类**

| 模型 | LVU 平均 | Breakfast | COIN |
|------|---------|-----------|------|
| MA-LMM | 63.0 | 93.0 | 93.2 |
| S5 | 59.2 | 90.7 | 90.8 |
| **HERMES** | **70.3** | **95.2** | **93.5** |

LVU 上相比前 SOTA 提升 **+7.3%**。

### 消融实验

**ECO 内存更新策略对比**

| 策略 | Acc. | Score |
|------|------|-------|
| 无 ECO | 55.1 | 3.55 |
| 随机保留 | 76.9 | 4.13 |
| FIFO | 77.1 | 4.15 |
| **ECO** | **78.6** | **4.23** |

**语义压缩方法对比**

| 方法 | Acc. | Score |
|------|------|-------|
| 无 SeTR | 73.3 | 4.09 |
| MaxPool | 70.4 | 3.99 |
| AvgPool | 73.3 | 4.04 |
| K-Means | 75.7 | 4.11 |
| **SeTR** | **78.6** | **4.23** |

**即插即用效果（集成到现有 SOTA 模型）**

| 基础模型 | +ECO 精度变化 | +ECO 延迟变化 | +SeTR 精度变化 |
|---------|-------------|-------------|--------------|
| MA-LMM | +3.4% | -43% | +3.8% |
| LongVA | +0.08% | -30% | +0.45% |
| LLaVA-OV | +0.67% | -35% | +1.04% |

### 关键发现

- ECO 和 SeTR 各自独立地提升性能，组合使用效果叠加——说明情景信息和语义信息确实互补。
- ECO 的余弦相似度合并策略优于随机保留和 FIFO，验证了"合并最冗余帧"的直觉。
- SeTR 大幅优于 MaxPool/AvgPool 等朴素方法，因为它是基于相似度的选择性合并，保留最具代表性的帧。
- 层级 Q-Former（95.2%）显著优于单独的 frame/video Q-Former（93.2%/94.1%）。
- episode 数 20、keep ratio 20% 在不同数据集上均为最优，说明超参数鲁棒性良好。
- HERMES 仅处理 14k 帧视频中的 100 帧（0.7%），而 MovieChat 处理 2048 帧，但 HERMES 精度更高。

## 亮点与洞察

- **情景 vs 语义的认知双通路设计**是全文最大贡献——这一认知科学启发的框架优雅地分离了两类互补信息。
- ECO 作为**即插即用模块**的通用性出色：集成到 MA-LMM 后精度提升 3.4% 的同时推理加速 43%，甚至不牺牲任何内存。
- SeTR 的 keep ratio 可低至 20%（丢弃 80% 帧），说明长视频确实存在大量冗余。
- 定性分析显示 HERMES 能诚实地表达不确定性（"I'm not sure"），而非像其他模型一样产生幻觉。

## 局限与展望

- ECO 和 SeTR 依赖启发式规则（余弦相似度合并/步长采样），可能在某些微妙的时间细节上失效。
- 两个模块独立运行，缺乏联合优化，可能产生冗余。
- 由于计算限制，未在大规模视频数据上预训练，限制了与 LLaVA-OneVision 等模型在 VideoMME 上的直接对比。
- LLM 后端仅使用 Vicuna-7B，未探索更强的语言模型。

## 相关工作与启发

- MA-LMM 的内存机制可被 ECO 直接替换且效果更好——说明任务无关的通用压缩可能优于特定任务设计。
- ToMe (Token Merging) 做的是 ViT 层内的 token 合并，而 SeTR 做的是跨帧级别的语义汇聚，两者目标和粒度完全不同。
- HERMES 的模块化设计为未来的"乐高式"视频理解系统提供了思路：不同模块各司其职，可自由组合。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 情景+语义双通路的认知启发设计新颖，即插即用验证有说服力
- **实验充分度**: ⭐⭐⭐⭐⭐ 5 个数据集 + 3 个 SOTA 模型集成 + 详尽消融
- **写作质量**: ⭐⭐⭐⭐ 叙述流畅，认知科学动机阐述清楚
- **价值**: ⭐⭐⭐⭐ ECO 和 SeTR 作为即插即用模块具有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] T*: Re-thinking Temporal Search for Long-Form Video Understanding](../../CVPR2025/video_understanding/re-thinking_temporal_search_for_long-form_video_understanding.md)
- [\[ICCV 2025\] Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow](flow4agent_long-form_video_understanding_via_motion_prior_from_optical_flow.md)
- [\[AAAI 2026\] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](../../AAAI2026/video_understanding/tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)
- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)
- [\[ICCV 2025\] VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges](videollamb_long_streaming_video_understanding_with_recurrent_memory_bridges.md)

</div>

<!-- RELATED:END -->
