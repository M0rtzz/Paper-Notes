---
title: "Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs"
authors: "Lucas Music, Stanislas Music, Antoine Yang, Cordelia Schmid, Ivan Laptev"
venue: "CVPR 2025"
date: 2025-03-31
tags: [video-chaptering, long-video, llm, speech-guided, efficient-inference]
arxiv: "2504.00072"
---

# Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs

**作者**: Lucas Music, Stanislas Music, Antoine Yang, Cordelia Schmid, Ivan Laptev  
**机构**: École des Ponts ParisTech / Inria / Google DeepMind  
**会议**: CVPR 2025  
**arXiv**: 2504.00072  

## 研究背景与动机

视频章节化（Video Chaptering）是将长视频自动分割为语义连贯的章节并生成章节标题的任务，对视频浏览、搜索和理解至关重要。随着 YouTube、Bilibili 等平台上长视频（1小时以上）内容的爆发式增长，自动章节化需求日益迫切：

**长视频处理的计算瓶颈**：小时级视频包含数十万帧，即使以低帧率采样（如 1fps），仍有数千帧需要处理。现有 Video LLM 的上下文窗口和计算资源难以支撑如此大规模的视觉输入

**均匀采样的低效性**：传统方法等间隔采样固定数量的帧（如100帧），但章节边界处的信息密度远高于章节内部，均匀采样大量浪费计算在冗余帧上

**视觉与语音信息的互补性未被充分利用**：长视频通常包含丰富的语音信息（如讲解、对话），这些语音信号天然地标记了内容的语义转换点，但现有方法主要依赖视觉特征

**现有方法的局限**：
   - Vid2Seq 等 seq2seq 模型在处理超长视频时性能急剧下降
   - 通用 Video LLM（如 Gemini）虽具备长上下文能力，但缺乏针对章节化任务的优化
   - 基于滑动窗口的方法难以捕捉跨窗口的全局语义关联

Chapter-Llama 提出了语音引导的帧选择策略，以极少的帧数（10.3帧 vs 100帧）实现超越现有方法的章节化性能。

## 方法详解

### 整体框架

Chapter-Llama 包含三个核心组件：语音引导帧选择（Speech-Guided Frame Selection）、视觉-语言编码（Visual-Language Encoding）和基于 LLM 的章节生成。

### 语音引导帧选择

**核心思想**：利用语音转录文本的语义变化点来指导视觉帧的选择。

**具体流程**：

1. **语音转录**：使用 ASR（自动语音识别）获取视频的时间戳对齐的转录文本
2. **文本语义分割**：通过 sentence embedding 计算相邻语音片段之间的语义相似度
3. **变化点检测**：在语义相似度序列上检测突变点（sharp drops），这些点对应内容转换
4. **帧选择**：在每个语义变化点附近选择代表性帧

| 帧选择策略 | 平均帧数 | F1 ↑ |
|-----------|---------|------|
| 均匀采样 100帧 | 100 | 38.2 |
| 均匀采样 50帧 | 50 | 35.6 |
| 随机采样 10帧 | 10 | 28.7 |
| **语音引导 (ours)** | **10.3** | **45.3** |

### 视觉-语言编码

选定的关键帧通过视觉编码器（CLIP ViT-L/14）提取视觉特征，与对应时间段的语音转录文本一起组织为多模态输入序列：

$$\text{Input} = [\text{SYS}] \oplus \bigoplus_{i=1}^{K} [\text{IMG}_i, \text{TIME}_i, \text{SPEECH}_i]$$

其中 $K \approx 10.3$ 为选定的关键帧数量。

### LLM 章节生成

**模型选择**：Llama-3.1-8B + LoRA 微调

| 配置项 | 设置 |
|--------|------|
| 基础模型 | Llama-3.1-8B |
| 微调方法 | LoRA (rank=16, alpha=32) |
| 训练时间 | 40 分钟 |
| 训练硬件 | 4× H100 GPU |
| 输出格式 | JSON (时间戳 + 章节标题) |

**Prompt 设计**：

模型接收多模态输入后，生成结构化的章节输出：
```
{"chapters": [
  {"start": "00:00:00", "title": "Introduction to..."},
  {"start": "00:05:32", "title": "Method overview..."},
  ...
]}
```

## 实验结果

### 主要结果 (VidChapters-7M 验证集)

| 方法 | F1 ↑ | 模型规模 | 帧数 |
|------|------|---------|------|
| Vid2Seq | 26.7 | 0.3B | 100 |
| VideoLLaMA2 | 31.2 | 7B | 32 |
| LLaVA-Video | 35.8 | 7B | 64 |
| Gemini-1.5-Pro (zero-shot) | 42.2 | >1T | 全部 |
| **Chapter-Llama (ours)** | **45.3** | 8B | **10.3** |
| vs Vid2Seq 提升 | **+69.8%** | - | - |

### 与 Gemini 系列对比

| 模型 | F1 | 设置 | 成本 |
|------|-----|------|------|
| Gemini-1.5-Flash (zero-shot) | 38.7 | API | ~$0.5/视频 |
| Gemini-1.5-Pro (zero-shot) | 42.2 | API | ~$2.0/视频 |
| **Chapter-Llama** | **45.3** | 本地 | ~$0.01/视频 |

Chapter-Llama 不仅超越 Gemini-1.5-Pro 零样本结果 (+3.1 F1)，且运行成本低两个数量级。

### 按视频时长分析

| 视频时长 | Vid2Seq F1 | Chapter-Llama F1 | 提升 |
|----------|-----------|-----------------|------|
| < 10 min | 32.1 | 47.8 | +48.9% |
| 10-30 min | 27.4 | 45.6 | +66.4% |
| 30-60 min | 23.8 | 44.1 | +85.3% |
| > 60 min | 19.2 | 42.7 | +122.4% |

随着视频时长增加，Chapter-Llama 的优势越发明显，证明了语音引导帧选择在长视频上的有效性。

### 消融实验

| 组件 | F1 |
|------|-----|
| 仅视觉（均匀100帧） | 38.2 |
| 仅语音转录 | 41.5 |
| 视觉+语音（均匀采样） | 42.1 |
| **视觉+语音引导帧选择** | **45.3** |

## 核心创新点

1. **语音引导帧选择**：利用语音语义变化点指导视觉帧采样，以平均 10.3 帧处理小时级视频，效率提升 10 倍
2. **极高效训练**：仅需 4 块 H100 训练 40 分钟，即可超越万亿参数的 Gemini-1.5-Pro
3. **多模态融合**：有效结合视觉和语音信息的互补优势
4. **强泛化能力**：在不同时长的视频上均表现稳健，特别是超长视频（1小时+）

## 效率分析

| 指标 | Vid2Seq | Gemini-1.5-Pro | Chapter-Llama |
|------|---------|----------------|---------------|
| 输入帧数 | 100 | ~3600 (1fps) | 10.3 |
| 推理时间/视频 | ~15s | ~60s | ~5s |
| 可训练参数 | 300M | 不可训练 | 4.2M (LoRA) |
| 训练成本 | 多天/多卡 | N/A | **40min/4×H100** |

## 局限性

- 依赖语音转录质量，对无语音视频（如音乐 MV、无声纪录片）效果受限
- ASR 在嘈杂环境或多语言场景下可能出错，影响帧选择质量
- 章节标题生成质量受 LLM 能力限制，对专业领域内容可能不够准确
- 未探索与视频中字幕、评论等其他文本信息的结合

## 相关工作

- Vid2Seq: 基于 seq2seq 的视频密集事件描述模型
- Gemini-1.5: Google 的长上下文多模态大模型
- LLaVA-Video: 基于 LLaVA 的视频理解模型
- Llama-3.1: Meta 的开源大语言模型

<!-- RELATED:START -->

## 相关论文

- [Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method](towards_long-horizon_vision-language_navigation_platform_benchmark_and_method.md)
- [Closed-loop Long-horizon Robotic Planning via Equilibrium Sequence Modeling](../../ICML2025/robotics/closed-loop_long-horizon_robotic_planning_via_equilibrium_sequence_modeling.md)
- [RDD: Retrieval-Based Demonstration Decomposer for Planner Alignment in Long-Horizon Tasks](../../NeurIPS2025/robotics/rdd_retrieval-based_demonstration_decomposer_for_planner_alignment_in_long-horiz.md)
- [RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](../../NeurIPS2025/robotics/robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)
- [Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](../../ICCV2025/robotics/moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)

<!-- RELATED:END -->
