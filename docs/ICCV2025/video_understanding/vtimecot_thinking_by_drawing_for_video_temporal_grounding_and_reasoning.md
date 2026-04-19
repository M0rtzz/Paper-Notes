---
title: >-
  [论文解读] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning
description: >-
  [ICCV 2025][视频理解][video temporal grounding] > 提出 VTimeCoT，一种无需训练的视觉-时间链式思维框架，通过在视频帧底部叠加可视化进度条和高亮关键片段，使多模态大模型能准确感知时间戳，在时间定位和推理问答任务上大幅超越 GPT-4o 和 Qwen2VL-7B 基线。
tags:
  - ICCV 2025
  - 视频理解
  - video temporal grounding
  - chain-of-thought
  - progress bar
  - 提示学习
  - training-free
---

# VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning

**会议**: ICCV 2025  
**arXiv**: [2510.14672](https://arxiv.org/abs/2510.14672)  
**代码**: [项目页面](https://vtimecot.github.io)  
**领域**: video_understanding  
**关键词**: video temporal grounding, chain-of-thought, progress bar, visual prompt, training-free  

## 一句话总结

> 提出 VTimeCoT，一种无需训练的视觉-时间链式思维框架，通过在视频帧底部叠加可视化进度条和高亮关键片段，使多模态大模型能准确感知时间戳，在时间定位和推理问答任务上大幅超越 GPT-4o 和 Qwen2VL-7B 基线。

## 研究背景与动机

视频理解中的时间定位（temporal grounding）是关键挑战。现有方法面临以下问题：

**端到端视频 MLLM**（如 VideoChat、VideoLLaMA、Qwen2VL）虽然能生成看似合理的回答，但在时间定位上严重不足——无法准确感知每帧的精确时间戳，也无法定位事件的起止边界。

**基于工具的 agent 方法**（VisProg、VideoAgent 等）通过调用外部工具（如目标检测、字幕模型）收集文本线索，但**完全依赖语言媒介**，无法直接捕获视频中的视觉-时间动态。

**VTimeLLM** 等方法通过微调引入时间定位能力，但需要特定数据和训练成本。

核心洞察：人类在观看视频时通过**进度条**直觉地理解时间进展和定位事件片段。进度条作为视觉提示，配合并行的视频内容，直接传达了时间进展的概念。本文利用这一洞察，让 MLLM 通过"画"进度条来思考时间。

## 方法详解

### 整体框架

VTimeCoT 是一个迭代式推理框架：
1. **初始化**：给 MLLM 格式化 prompt（定义 THOUGHT/ACTION/TERMINATE 结构 + 工具声明）
2. **循环推理**：每步生成 Thought（分析推理）→ Action（调用工具操作视频）→ 更新视频记忆
3. **终止**：模型判断可直接回答时生成 TERMINATE

### 关键设计1：帧同步进度条集成工具

通过 OpenCV 在视频帧底部绘制进度条，标注每帧对应的精确时间戳（秒数格式）：

$$V'_t = V_t \oplus T_t$$

其中 $\oplus$ 为垂直拼接，$T_t$ 为 OpenCV 生成的进度条图像。进度条包含：
- 矩形轨道 + 圆形当前位置指示器 + 秒数标注
- **帧同步步骤**：将帧索引转换为真实秒数，适配任意 FPS 采样率

利用 MLLM 的 OCR 能力直接从视觉进度条读取时间信息，无需额外训练。

### 关键设计2：基于视频-文本相似度的高亮工具

利用 VideoCLIP-XL 基础模型进行零样本时刻检索：

1. 以 $r=1$ FPS 采样视频帧，每 8 帧为一个 clip，共 $N$ 个 clip
2. 计算每个 clip 与查询文本的余弦相似度：

$$\text{sim}(i) = \frac{e_i^v \cdot e^q}{|e_i^v| |e^q|}$$

3. Top-$k$（$k=8$）选取最相似的 clip，提取连续片段的起止时间
4. 用 OpenCV 在进度条对应区间绘制彩色高亮掩码

由于时刻检索作为**外部模块**运行，不受 MLLM token 限制，天然适配长视频。

### 视觉-时间链式思维过程

```
Algorithm：Visuotemporal CoT
输入：视频 V, 问题 Q, 初始化 prompt I
1. s←0; V_0←V; H_0←{I, Q}
2. while true:
3.   y_s ← MLLM(V_s, H_s)
4.   解析 THOUGHT_s, ACTION_s, TERMINATE_s
5.   if not TERMINATE_s:
6.     C_s ← 构建代码(ACTION_s, Toolset)
7.     V_{s+1} ← C_s(V_s)      // 更新视频（叠加进度条/高亮）
8.     H_{s+1} ← {H_s, y_s}    // 更新历史
9.   else: return y_s
```

最大推理步数 $T=3$，默认输入 32 帧，长边缩放至 480 像素。

## 实验关键数据

### 主实验：Charades-STA 时间定位

| 方法 | R1@0.3 | R1@0.5 | R1@0.7 | mIoU |
|------|--------|--------|--------|------|
| VTimeLLM-13B | 55.30 | 34.30 | 14.70 | 34.60 |
| Qwen2VL-7B | 37.31 | 12.85 | 4.11 | 24.34 |
| **VTimeCoT_Qwen2VL** | **66.96** | **38.79** | **20.83** | **43.41** |
| GPT4o | 63.76 | 37.12 | 14.65 | 40.20 |
| **VTimeCoT_GPT4o** | **74.06** | **51.02** | **22.45** | **46.78** |

### 主实验：QVHighlights 不连续片段定位

| 方法 | R1@0.3 | R1@0.5 | R1@0.7 | mIoU |
|------|--------|--------|--------|------|
| VTimeLLM-7B | 44.58 | 25.03 | 9.29 | 28.99 |
| GPT4o | 55.61 | 35.68 | 19.29 | 37.66 |
| **VTimeCoT_GPT4o** | **79.35** | **59.74** | **33.81** | **54.49** |

VTimeCoT 在 QVHighlights 上 mIoU 提升 16.83%（GPT4o 基线）。

### 消融实验：关键模块贡献

| MLLM | CoT | 进度条 | 高亮 | QVHighlights mIoU | Vript-RR M-Acc |
|------|-----|--------|------|--------------------|----------------|
| GPT4o | ✗ | ✗ | ✗ | 37.66 | 70.39 |
| GPT4o | ✓ | ✗ | ✗ | 41.85 | 73.68 |
| GPT4o | ✓ | ✓ | ✗ | 49.40 | 76.32 |
| GPT4o | ✓ | ✓ | ✓ | **54.49** | **83.55** |

### 长视频问答：VideoMME

| 方法 | 帧数 | 无字幕 Acc | 有字幕 Acc |
|------|------|-----------|-----------|
| GPT4o (512帧) | 384 | 71.9 | 77.2 |
| GPT4o (32帧) | 32 | 61.6 | 65.1 |
| **VTimeCoT_GPT4o** | 32 | **64.2** | **73.6** |
| VideoLLaMA3-7B | 180 | 66.2 | 70.3 |

### 关键发现

- VTimeCoT 在**无需任何训练**的情况下超越了经过时间定位数据微调的 VTimeLLM-13B。
- 进度条贡献最大（mIoU +7.55），高亮进一步提升（+5.09），总计提升 16.83 个点。
- 在 Vript-RR 长视频推理中，VTimeCoT_GPT4o 多选题准确率达 83.55%（vs GPT4o 70.39%）。
- 仅用 32 帧的 VTimeCoT 在 VideoMME 有字幕设置下（73.6%）超越了使用 384 帧的原版 GPT4o（77.2% 的接近水平）。

## 亮点与洞察

1. **视觉进度条作为时间推理媒介**：突破了现有方法只用文本做 CoT 推理的范式，首次将空间维度的 visual prompt 思想扩展到时间维度。
2. **极致简洁的实现**：仅用 OpenCV 画进度条 + VideoCLIP-XL 计算相似度，零训练、零微调。
3. **长视频友好**：时刻检索模块作为外部组件不受 MLLM token 限制，且进度条自动适配任意 FPS。
4. **可解释性强**：每步推理可视化（进度条+高亮+thought），让模型的时间推理过程透明。
5. **MLLM 的 OCR 能力**被巧妙复用于时间感知——从进度条上"读"时间。

## 局限性

- 依赖 MLLM 强大的代码生成和 OCR 能力，对较弱的模型可能效果不佳。
- 最大推理步数 $T=3$ 限制了复杂推理的深度，但更多步数会增加延迟和成本。
- VideoCLIP-XL 的检索质量决定了高亮的准确性，对 out-of-distribution 视频可能不鲁棒。
- 进度条占用了帧底部空间，可能遮挡部分视觉信息。
- 使用 GPT-4o API 推理成本显著。

## 相关工作与启发

- 与 Visual Sketchpad 等空间视觉提示方法的对比：VTimeCoT 是首个**时间维度的视觉提示**，补充了 SoM 等空间提示的不足。
- 与 VTimeLLM 的关系：VTimeLLM 通过微调获得时间定位能力，VTimeCoT 通过工具使用零样本达到更好效果，展示了 tool-use 范式的强大潜力。
- 进度条+高亮的思路可推广到音频理解（频谱进度条）、时间序列分析等其他时间敏感任务。

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 实验 | ⭐⭐⭐⭐ |
| 写作 | ⭐⭐⭐⭐ |
| 价值 | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](../../NeurIPS2025/video_understanding/when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)
- [Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding](towards_video_thinking_test_a_holistic_benchmark_for_advanced_video_reasoning_an.md)
- [TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding](timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)
- [Moment Quantization for Video Temporal Grounding](moment_quantization_for_video_temporal_grounding.md)
- [Hierarchical Event Memory for Accurate and Low-latency Online Video Temporal Grounding](hierarchical_event_memory_for_accurate_and_low-latency_online_video_temporal_gro.md)

<!-- RELATED:END -->
