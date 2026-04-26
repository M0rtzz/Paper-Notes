---
title: >-
  [论文解读] Giraffe: Design Choices for Extending the Context Length of Visual Language Models
description: >-
  [ACL2025][LLM效率][VLM] 系统性地探索了将现有视觉语言模型（VLM）的上下文窗口扩展到128K的设计空间，从数据配方、位置编码扩展到上下文利用三个维度提出最佳实践，并提出 M-RoPE++ 和混合分辨率训练两项技术，构建的 Giraffe 模型在长上下文 VLM 中达 SOTA。
tags:
  - ACL2025
  - LLM效率
  - VLM
  - Long Context
  - M-RoPE++
  - Hybrid-Resolution
  - Context Extension
  - 视频理解
---

# Giraffe: Design Choices for Extending the Context Length of Visual Language Models

**会议**: ACL2025  
**arXiv**: [2412.12735](https://arxiv.org/abs/2412.12735)  
**代码**: [GitHub](https://github.com/kiaia/GIRAFFE)  
**领域**: LLM Efficiency / 视觉语言模型长上下文  
**关键词**: VLM, Long Context, M-RoPE++, Hybrid-Resolution, Context Extension, Video Understanding  

## 一句话总结

系统性地探索了将现有视觉语言模型（VLM）的上下文窗口扩展到128K的设计空间，从数据配方、位置编码扩展到上下文利用三个维度提出最佳实践，并提出 M-RoPE++ 和混合分辨率训练两项技术，构建的 Giraffe 模型在长上下文 VLM 中达 SOTA。

## 研究背景与动机

### 问题定义
视觉语言模型（VLM）在处理多模态输入方面展现了出色能力，但多图像和高分辨率长视频等高级场景对模型的长程建模能力提出了更高要求。例如，2K 的上下文长度只能处理几帧视频图像，严重限制了长视频理解任务的上限。

### 核心挑战
现有开源 VLM 缺乏对上下文窗口扩展的系统性探索，商业模型又不公开技术细节。已有的工作（如 LongVA, LongVILA, LongLLaVA）各自采用不同策略，但没有系统性比较各种设计选择的效果。

### 三个核心研究问题
1. 如何有效组织和整理训练数据？
2. 如何高效训练更长上下文的 VLM？
3. 如何更好地利用扩展后的上下文窗口？

## 方法详解

### 整体框架

论文围绕三个研究问题，从数据整理、上下文扩展训练、上下文利用三个维度展开系统性实验研究，最终基于 Qwen-VL 系列模型构建 Giraffe。

### 关键设计1：ETVLM 数据配方

数据来源包含四类：

| 类别 | 数据源 | 占比 |
|------|--------|------|
| 长上下文纯文本指令 | LongAlign, LongAlpaca | 20% |
| 短视觉指令数据 | LLaVA-Instruct, M3IT | 25% |
| 图像交错数据 | MMDU, Mantis, ArXivQA | 25% |
| 视频QA+摘要 | ShareGPT4O, MLVU, LLaVA-Video | 30% |

**数据配比实验发现**：
- 短多模态指令数据对扩展长上下文能力和保持短上下文性能都至关重要
- 均衡的数据比例有助于下游任务的均衡表现
- 增加任何单一数据类型的比例只会主要提升其对应任务的性能

**数据长度实验发现**：
- 长数据（>8K tokens）比例达到60%时，长上下文任务性能趋于饱和
- 长数据比例超过60%后，短上下文任务性能明显下降
- 因此选择60%的长数据比例作为最优平衡点

### 关键设计2：M-RoPE++ 位置编码扩展

**问题分析**：M-RoPE 将旋转位置编码分解为时间(temporal)、高度(height)、宽度(width)三个维度，分配比例为 2:3:3。

**现有方法的局限**：
- PI（位置插值）和 NTK 方法会无差别地压缩高频信号，可能混淆模型对相邻帧时序的感知
- 实验发现 VLM 和 LLM 类似，存在"有效长度短缺"（falls short）现象——扩展后的有效长度远小于训练长度

**M-RoPE++ 的设计思路**：
- **时间维度（低维 → 高频）**：保持外推（extrapolation），因为时间信息处于高频部分，预训练阶段已经充分覆盖
- **空间维度（高维 → 低频）**：应用插值（interpolation），因为高度和宽度占据高维空间，预训练时可能未充分覆盖旋转域

分段函数定义：
$$\theta_d' = \begin{cases} \theta_d & \text{if } 0 < d \leq 2x \text{ (temporal)} \\ (\frac{1}{s} + (1-\frac{1}{s})\cdot\frac{d-r_{5x}}{r_{2x}-r_{5x}})\cdot\theta_d & \text{if } 2x < d \leq 5x \text{ (height)} \\ \frac{\theta_d}{s} & \text{if } 5x < d \leq 8x \text{ (width)} \end{cases}$$

其中 $s = L'/L_V$ 为扩展比例。

### 关键设计3：训练策略选择

论文对比了三种训练策略：

| 策略 | MMBench | BLINK | VideoMME |
|------|---------|-------|----------|
| 一阶段多模态指令tuning | **82.8** | **54.6** | **58.5** |
| 两阶段：文本扩展+MM指令 | 79.8 | 52.9 | 58.1 |
| 两阶段：MM对齐+MM指令 | 80.5 | 51.2 | 57.8 |

**结论**：直接对VLM进行长上下文多模态指令微调即可，不需要多阶段训练。原因是：
- 长上下文多模态数据本身已经涵盖了多样的长度分布
- Qwen2-VL 已经经过指令微调，再次进行对齐训练会破坏已学习的分布

### 关键设计4：混合分辨率训练（Hybrid-Resolution）

受 SlowFast 网络启发：
- 将视频帧分为 $N$ 组，每组 $L$ 帧
- 每组第一帧使用高分辨率（$m$ tokens）
- 后续 $L-1$ 帧使用低分辨率（$m/s$ tokens）

Token 使用量从 $L \times N \times m$ 降低到 $(1+\frac{L-1}{s}) \times N \times m$

实验表明在等价token预算下，混合分辨率推理能提升关键帧的分辨率并提高下游任务性能。

## 实验

### 评估体系
- **短上下文**：MME, MMBench（单图理解）
- **多图像**：Mantis-Eval, QBench, BLINK
- **长视频**：VideoMME, LongVideoBench
- **有效长度**：Visual Haystack（视觉针测试）

### 位置编码方法对比

| 方法 | VideoMME Long (512帧) | Visual Haystack (100图) |
|------|----------------------|------------------------|
| 直接外推 | 55.4 | 51.3 |
| PI | 56.0 | 57.8 |
| NTK | 56.2 | 56.7 |
| **M-RoPE++** | **58.5** | **61.3** |

M-RoPE++ 在所有设置下持续优于其他方法。

### 视频理解任务（主结果）

| 模型 | 帧数 | VideoMME Overall | LongVideoBench Avg |
|------|------|-----------------|-------------------|
| GPT-4V | 10 | 59.9 | 59.1 |
| GPT-4o | 384 | 71.9 | 66.7 |
| Qwen2-VL-7B | 256 | 63.2 | 61.5 |
| **Giraffe** | 768 | 64.8 | 63.3 |
| **Giraffe + Hybrid-res** | 1024 | **65.9** | **64.3** |

Giraffe 在开源长 VLM 中达到 SOTA，在部分类别上超越 GPT-4V。

### 多图像和单图像评估

- **多图像**：Giraffe-QwenVL 相较 Qwen-VL 有显著提升（Mantis-Eval: 39.2→48.3, QBench: 45.9→57.4）
- **单图像**：Giraffe 维持了与 Qwen2-VL 相当的短上下文性能（MMBench: 82.1 vs 82.8）

### 帧数-分辨率权衡

| 帧数 | 每帧Token | VideoMME Medium | VideoMME Long |
|------|----------|----------------|---------------|
| 128 | 960 | 62.5 | 55.6 |
| 512 | 240 | 64.6 | 58.2 |
| 768 | 160 | 64.8 | 58.5 |
| 1024 | 120 | 64.7 | 58.5 |

- 中等长度任务在512帧后趋于饱和
- 长视频任务受益于更多帧数，但当单帧分辨率过低时性能下降

## 亮点与洞察

1. **系统性的设计空间探索**：论文不是简单提出一个方法，而是通过大量受控实验系统性地回答了三个关键设计问题，结论具有很强的实用指导意义
2. **M-RoPE++ 对不同维度差异化处理**：基于对时间（高频）和空间（低频）维度在 RoPE 中本质差异的深刻理解，提出针对性的外推/插值策略
3. **一阶段训练即最优**：打破了多阶段训练的直觉，发现直接指令微调就能达到最佳效果，大幅简化训练流程
4. **混合分辨率的实用性**：SlowFast 思想在 VLM 中的巧妙应用，在固定 token 预算下有效提升性能
5. **有效长度分析**：揭示了 VLM 与 LLM 类似的"有效长度短缺"现象，为后续研究提供重要基准

## 局限性

1. 仅基于 Qwen-VL 系列模型验证，结论的可迁移性需要更多验证
2. 128K 的上下文扩展需要大量 GPU 资源（8×80G H100），训练成本较高
3. 有效长度仍远小于训练长度（约40K vs 128K），说明位置编码扩展仍有潜力
4. 混合分辨率中高分辨率帧的选择策略较为简单（固定每组第一帧），未探索自适应选择
5. 评估主要集中在英文数据集，多语言场景未涉及

## 相关工作

- **长上下文LLM**：NTK, PI, YaRN, StreamingLLM, InfLLM 等位置编码扩展和高效推理方法
- **VLM 系列**：LLaVA, MiniGPT-4, Qwen-VL, VideoLLaVA 等
- **长上下文VLM**：LongVA（先扩展LLM再转VLM）, LongVILA（多阶段+序列并行）, LongLLaVA（Mamba+Transformer）

## 评分 ⭐⭐⭐⭐

非常扎实的系统性研究工作，实验充分且结论有很强的实践指导价值。M-RoPE++ 设计有理论洞察，混合分辨率训练实用性强。主要不足在于仅限 Qwen-VL 系列验证，以及有效长度仍有较大提升空间。

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)
- [\[ACL 2025\] How to Train Long-Context Language Models (Effectively)](train_long_context_effectively.md)
- [\[ACL 2025\] Literary Evidence Retrieval via Long-Context Language Models](literary_evidence_retrieval_via_long-context_language_models.md)
- [\[ICML 2025\] Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling](../../ICML2025/llm_efficiency/efficient_length-generalizable_attention_via_causal_retrieval_for_long-context_l.md)
- [\[ACL 2025\] LongSafety: Evaluating Long-Context Safety of Large Language Models](longsafety_evaluating_long-context_safety_of_large_language_models.md)

<!-- RELATED:END -->
