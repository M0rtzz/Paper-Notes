---
title: >-
  [论文解读] Video-KTR: 通过关键 Token 归因增强视频推理
description: >-
  [ICLR 2026][视频理解][视频推理] 提出 Video-KTR，一种模态感知的策略塑造框架，通过反事实分析识别视觉感知型、时序敏感型和高熵 Token 三类关键 Token，仅对这些 Token 执行选择性强化学习更新，在多个视频推理基准上达到 SOTA（Video-Holmes 42.7%，超越 GPT-4o）。
tags:
  - ICLR 2026
  - 视频理解
  - 视频推理
  - 强化学习
  - Token归因
  - 多模态LLM
  - GRPO
---

# Video-KTR: 通过关键 Token 归因增强视频推理

**会议**: ICLR 2026  
**arXiv**: [2601.19686](https://arxiv.org/abs/2601.19686)  
**领域**: 视频理解  
**关键词**: 视频推理, 强化学习, Token归因, 多模态LLM, GRPO

## 一句话总结

提出 Video-KTR，一种模态感知的策略塑造框架，通过反事实分析识别视觉感知型、时序敏感型和高熵 Token 三类关键 Token，仅对这些 Token 执行选择性强化学习更新，在多个视频推理基准上达到 SOTA（Video-Holmes 42.7%，超越 GPT-4o）。

## 研究背景与动机

强化学习（RL）在提升多模态 LLM 推理能力方面展现出强大潜力，但现有视频推理方法存在三个关键缺陷：

**粗粒度奖励**：依赖序列级奖励，无法精确指导哪些 Token 需要重点学习

**单一因素选择**：仅基于信息熵选择 Token，忽略模态特异性依赖

**语言先验过度依赖**：缺乏视觉输入与输出 Token 的细粒度语义对齐，导致幻觉风险增加

现有方法如 T-GRPO 虽引入时序约束（惩罚帧打乱后的预测），但属于全局粗糙假设，忽略了某些任务可仅靠静态线索解决的事实。

## 方法详解

### 整体框架

Video-KTR 在 GRPO 框架基础上引入模态感知的 Token 级策略塑造机制，核心包含三步：(1) 多视角 Token 重要性分析；(2) Token 选择；(3) 选择性策略更新。

### 关键设计：三类归因信号

**1. 视觉感知型 Token（Visual-Aware）**

通过**反事实遮蔽**量化每个 Token 对视觉输入的依赖程度。将视频特征置零后计算 logit 变化：

$$\Delta^{\text{vis}}_i = |\log \text{softmax}(\mathbf{z}^{\text{full}}_i)_{y_i} - \log \text{softmax}(\mathbf{z}^{\text{masked}}_i)_{y_i}|$$

高 $\Delta^{\text{vis}}_i$ 的 Token（如"person"、"door"、"blue"）表明其预测强烈依赖视觉输入。

**2. 时序敏感型 Token（Temporal-Aware）**

通过**帧顺序打乱**检测对时序结构的敏感度：

$$\Delta^{\text{temp}}_i = |\log \text{softmax}(\mathbf{z}^{\text{ordered}}_i)_{y_i} - \log \text{softmax}(\mathbf{z}^{\text{shuffled}}_i)_{y_i}|$$

高 $\Delta^{\text{temp}}_i$ 的 Token（如"first"、"then"、"appear"）反映对事件顺序和因果关系的依赖。

**3. 高熵 Token（Entropy-Aware）**

捕获预测不确定性，识别推理关键点：

$$\mathcal{H}(i) = -\sum_w p(z_i = w) \log p(z_i = w)$$

高熵 Token（如"however"、"wait"）通常标记语篇转折或决策点。

### Token 选择与策略更新

选取每种归因策略中 top $r\%$ 的 Token，取并集 $\mathcal{S} = \mathcal{S}_{\text{vis}} \cup \mathcal{S}_{\text{temp}} \cup \mathcal{S}_{\text{ent}}$，构建二值掩码 $m_{i,t}$。修改后的 GRPO 目标函数：

$$\mathcal{J}_{\text{Video-KTR}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} m_{i,t} \cdot \min(r_{i,t}\hat{A}_{i,t}, \text{clip}(r_{i,t})\hat{A}_{i,t})\right]$$

仅 $m_{i,t}=1$ 的关键 Token 参与损失计算。

## 实验关键数据

### 主实验：跨基准性能对比

| 模型 | 规模 | Video-Holmes | VideoMMMU | MMVU(mc) | TempCompass | VideoMME |
|------|------|-------------|-----------|----------|-------------|---------|
| GPT-4o | — | 42.0 | 61.2 | 75.4 | 73.8 | 71.9 |
| GPT-5 | — | 46.7 | 84.6 | 82.6 | 83.3 | 86.7 |
| Video-R1 | 7B | 36.5 | 52.3 | 63.8 | 73.2 | 59.3 |
| TW-GRPO | 7B | 32.9 | 51.3 | 65.8 | 73.3 | 55.1 |
| **Video-KTR** | **7B** | **42.7** | **53.1** | **66.6** | **73.5** | **62.5** |

### 消融实验：归因信号组合

| 策略 | E | V | T | Video-Holmes | VideoMMMU | MMVU | 平均 |
|------|---|---|---|-------------|-----------|------|------|
| Vanilla GRPO | ✗ | ✗ | ✗ | 38.8 | 49.8 | 64.8 | 51.1 |
| 仅 T | ✗ | ✗ | ✓ | 42.1 | 50.1 | 65.5 | 52.6 |
| 仅 V | ✗ | ✓ | ✗ | 40.5 | 51.9 | 65.1 | 52.5 |
| V+E+T | ✓ | ✓ | ✓ | **41.6** | **52.6** | **65.9** | **53.4** |

### 关键发现

1. **三种信号互补**：单独使用任一信号均优于 vanilla GRPO，但完整组合效果最佳
2. **硬选择优于软加权**：top-20% 二值掩码一致优于 Softmax/Sigmoid/线性/指数加权
3. **语言学分布差异化**：视觉 Token 以名词为主（24.8%），时序 Token 以动词为主（21.2%），熵 Token 副词比例更高（8.8%）
4. **最优更新比例为 20%**：更高比例引入噪声，过低则信号不足

## 亮点与洞察

1. **反事实分析的巧妙应用**：通过视觉遮蔽和帧打乱两种扰动，自然地解耦了视觉和时序依赖
2. **即插即用设计**：Video-KTR 可无缝集成到任何基于 GRPO 的 RL 训练中
3. **7B 模型超越 GPT-4o**：在 Video-Holmes 上 42.7% vs 42.0%，证明精细的 Token 级优化可弥补模型规模差距
4. **未选中 Token 的分析**：被过滤的低信息 Token 主要是功能词（助动词、代词、介词等），验证了归因机制能有效过滤冗余

## 局限性

1. 反事实分析需要额外的前向传播（遮蔽视觉 + 打乱帧序），增加训练开销
2. 仅在 7B 规模模型上验证，更大规模模型是否仍有同等收益未知
3. Token 选择比例 $r$ 作为固定超参数，未能根据样本难度自适应调整
4. 帧数限制为 16-64 帧，对超长视频的处理能力未验证

## 评分 ⭐⭐⭐⭐⭐

精巧的方法设计、扎实的实验分析、显著的性能提升。将 RL 从粗粒度序列级奖励转向细粒度模态感知 Token 级更新，是视频推理 RL 训练的重要进步。

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)
- [\[ICLR 2026\] FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging](flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](../../CVPR2026/video_understanding/streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[NeurIPS 2025\] Token Bottleneck: One Token to Remember Dynamics](../../NeurIPS2025/video_understanding/token_bottleneck_one_token_to_remember_dynamics.md)
- [\[CVPR 2026\] Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention](../../CVPR2026/video_understanding/unified_spatiotemporal_token_compression_for_video-llms_at_ultra-low_retention.md)

<!-- RELATED:END -->
