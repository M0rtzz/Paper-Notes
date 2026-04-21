---
title: >-
  [论文解读] DrVideo: Document Retrieval Based Long Video Understanding
description: >-
  [CVPR 2025][视频理解][长视频理解] 提出DrVideo，将**长视频理解转化为长文档理解**任务：先将视频帧转为文本文档，通过**文档检索**定位关键帧并**增强信息**，再通过**Planning-Interaction双Agent循环**迭代补充缺失信息，最终以CoT方式回答问题。在EgoSchema（3分钟）、MovieChat-1K（10分钟）和Video-MME长视频分割（平均44分钟）上大幅超越现有LLM-based SOTA。
tags:
  - CVPR 2025
  - 视频理解
  - 长视频理解
  - 文档检索
  - LLM Agent
  - 信息增强
  - Chain-of-Thought
---

# DrVideo: Document Retrieval Based Long Video Understanding

**会议**: CVPR 2025  
**arXiv**: [2406.12846](https://arxiv.org/abs/2406.12846)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 长视频理解, 文档检索, LLM Agent, 信息增强, Chain-of-Thought

## 一句话总结

提出DrVideo，将**长视频理解转化为长文档理解**任务：先将视频帧转为文本文档，通过**文档检索**定位关键帧并**增强信息**，再通过**Planning-Interaction双Agent循环**迭代补充缺失信息，最终以CoT方式回答问题。在EgoSchema（3分钟）、MovieChat-1K（10分钟）和Video-MME长视频分割（平均44分钟）上大幅超越现有LLM-based SOTA。

## 研究背景与动机

长视频理解是计算机视觉的核心挑战，需要在长达数十分钟甚至几小时的视频中进行时空信息处理和长程推理。

**现有痛点**：
- **Video-LM方法**（如LLaVA-NeXT-Video）：将视频编码为视觉token序列后与文本token拼接，但(i)无法将整个长视频作为输入（token长度限制），(ii)通常采用大步长均匀采样（如每16帧取1帧）导致关键帧丢失，(iii)简单拼接使LLM难以在长token序列中定位关键信息
- **LLM-based方法**（如LLoVi、VideoAgent）：将视频转为文本描述后利用LLM推理，但(i)**关键帧定位不准确**——VideoAgent依赖LLM先验推断缺失信息再用CLIP相似度选帧，缺乏对视频内容的整体把握；(ii)**信息损失严重**——即使找到关键帧，生成的caption无法覆盖所有关键细节（如"女人在照镜子"的描述无法回答"她穿的什么"）

**核心矛盾**：长视频中关键信息稀疏且分散，既需要全局视频理解来定位关键帧，又需要针对性的信息增强来避免caption遗漏。

**核心idea**：模拟人类理解长视频的方式——先概览全片了解大致内容，再定位问题相关的关键段落，仔细查看后回答。将这一过程形式化为**文档检索+增强+多阶段Agent交互**。

## 方法详解

### 整体框架

DrVideo由五个组件构成（Figure 1）：(1) **视频-文档转换模块**：将每帧转为短文本描述构成初始文档；(2) **检索模块**：计算问题与文档的语义相似度，检索top-K关键帧；(3) **文档增强模块**：用VLM为关键帧生成针对性的详细描述；(4) **多阶段Agent交互循环**：Planning Agent和Interaction Agent迭代查找缺失信息并增强；(5) **回答模块**：基于最终文档以CoT方式生成答案。

### 关键设计

1. **视频-文档转换 + 文档检索**
    - **功能**：将长视频转化为可检索的文本文档，并定位问题相关的关键帧
    - **核心思路**：
     - 先用VLM（如LLaVA-NeXT）为每帧生成50词以内的简短描述，构成初始文档 $Doc_{init} = \{\{1, S_{V_1}\}, \{2, S_{V_2}\}, \ldots, \{T, S_{V_T}\}\}$
     - 用OpenAI embedding模型编码文档和问题，通过余弦相似度检索top-K（默认K=5）关键帧
    - **设计动机**：文本空间的语义检索比CLIP的图文相似度更**精确**（充分利用LLM的长文本检索能力），且能在文档增强后进一步提升检索效果

2. **文档增强模块**
    - **功能**：为检索到的关键帧补充详细的、问题相关的信息，弥补初始简短描述的信息损失
    - **核心思路**：对每个关键帧 $t'$，使用LLaVA-NeXT配合不同prompt生成详细描述 $L_{V_{t'}}$。初始augmented prompt为通用QA prompt，后续Agent循环中会生成更针对性的prompt（如请求特定类型信息）
    - **设计动机**：初始的50词描述必然遗漏大量细节，但为每帧都生成详细描述成本过高（T可能数百帧），仅对top-K关键帧增强是成本效率的最优解

3. **多阶段Agent交互循环**
    - **功能**：迭代式地发现和补充仍然缺失的关键信息
    - **核心思路**：
     - **Planning Agent**：给定问题Q和当前增强文档 $\mathcal{AD}_i$，判断当前信息是否足以confident回答。不足则分析原因，更新分析历史 $\mathcal{H}_i$
     - **Interaction Agent**：根据分析历史和当前文档，找出N个缺失信息的关键帧（$n \notin \text{topk\_doc}$，$N < K$），并确定每帧需要的信息类型（A: 图像描述 or B: 视觉问答），然后与文档增强模块交互获取信息
     - 循环直到Planning Agent认为信息充分或达到最大迭代次数
    - **设计动机**：单次检索+增强可能遗漏间接相关但对推理至关重要的帧信息。Agent循环通过上下文推理发现"missing pieces"，比静态检索更加智能

## 实验关键数据

### 主实验

| 基准（视频长度） | 方法 | LLM | 核心指标 |
|----------------|------|-----|---------|
| **EgoSchema子集**（3min） | VideoAgent | GPT-4 | 60.2% |
| | LLoVi | GPT-4 | 61.2% |
| | **DrVideo** | **GPT-4** | **66.4%** (+5.2%) |
| **MovieChat-1K Global**（10min） | LLoVi | GPT-4 | 58.3% Acc |
| | **DrVideo** | **GPT-4** | **93.1%** (+34.8%) |
| **MovieChat-1K Breakpoint**（10min） | VideoAgent* | GPT-4 | 31.6% Acc |
| | **DrVideo** | **GPT-4** | **56.4%** (+24.8%) |
| **Video-MME长视频w/o subs**（44min） | LLoVi* | GPT | 45.4% |
| | **DrVideo** | **DeepSeek** | **51.7%** (+6.3%) |
| **Video-MME长视频w/ subs**（44min） | GPT-4o mini | - | 63.4% |
| | Gemini 1.5 Flash | - | 68.8% |
| | **DrVideo** | **DeepSeek** | **71.7%** |

- 在Video-MME+字幕设置下，DrVideo甚至**超越**GPT-4o mini（71.7% vs 63.4%）和Gemini 1.5 Flash（71.7% vs 68.8%）

### 消融实验（EgoSchema子集，GPT-3.5）

| 配置 | 准确率(%) |
|------|----------|
| 无检索模块+无Agent循环+有CoT | 57.4 |
| +检索模块 | 60.6 (+3.2) |
| +Agent循环 | 62.6 (+2.0) |
| 完整DrVideo（无CoT） | 62.2 |
| 完整DrVideo | **62.6** |

- 检索模块贡献最大（+3.2%），Agent循环进一步提升（+2.0%）
- VQA+Caption两种增强类型都有必要（去掉VQA：-2.2%，去掉Caption：-0.8%）
- Top-K=5效果最佳，K增大反而引入噪声

### 关键发现

- 将长视频理解转为文档理解是完全可行的，且能充分**利用LLM的长文本检索和推理能力**
- Agent交互循环通过上下文推理能找到纯相似度检索遗漏的关键帧
- 字幕信息对长视频理解极为重要——仅用字幕（无视觉）就能达到68.5%，视觉信息的增强贡献额外的3.2%
- DrVideo是training-free的，可在单张RTX 4090上以合理的API调用次数复现

## 亮点与洞察

- **范式转换**：从"视频→视觉token→LLM"转为"视频→文本文档→LLM"，巧妙地将长视频问题转化为LLM擅长的长文档理解问题
- **渐进式信息收集**：检索→增强→Agent补充的流程模拟了人类理解长视频的认知过程，信息充分性由LLM自主判断
- **异常强劲的实验结果**：在MovieChat-1K上从58.3%→93.1%（全局模式）的提升幅度惊人
- Training-free+单卡可复现，对研究者极为友好

## 局限性

- 高度依赖LLM API（GPT-4/DeepSeek），**推理成本**与视频长度和Agent循环次数成正比
- 视频-文档转换阶段的caption质量是性能的上限——如果captioner遗漏了关键信息且后续Agent也未能发现，则无法挽回
- 仅依赖语言空间的推理，对于需要精确空间定位或视觉细节判断的任务（如"物体在画面哪个位置"）可能力不从心
- 帧采样率固定（0.5FPS/0.2FPS），对包含快速动作的视频可能遗漏关键帧

## 相关工作与启发

- **LLoVi**开创了"短clip描述→LLM摘要→回答"的范式，DrVideo在此基础上增加了文档检索和信息增强
- **VideoAgent**的agent-based设计启发了DrVideo的多阶段循环，但DrVideo通过文档检索替代了CLIP相似度定位，更加精准
- **RAG（检索增强生成）**在NLP中的成功经验被成功迁移到视频理解场景
- 启发：其他长序列理解任务（如长音频理解、多轮对话历史理解）也可尝试转化为文档检索问题

## 评分

⭐⭐⭐⭐ — 将长视频理解转化为文档检索+增强是一个清晰优雅的框架设计。实验结果在多个基准上大幅超越SOTA，尤其是MovieChat-1K上34.8%的提升和在Video-MME上超越GPT-4o mini/Gemini 1.5 Flash的结果令人信服。Training-free+单卡可复现的特性极具实用价值。但对LLM API的重度依赖是一个实际限制。

<!-- RELATED:START -->

## 相关论文

- [MLVU: Benchmarking Multi-task Long Video Understanding](mlvu_benchmarking_multi-task_long_video_understanding.md)
- [VISTA: Enhancing Long-Duration and High-Resolution Video Understanding by Video SpatioTemporal Augmentation](vista_enhancing_long-duration_and_high-resolution_video_understanding_by_video_s.md)
- [VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](../../NeurIPS2025/video_understanding/vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)
- [ReWind: Understanding Long Videos with Instructed Learnable Memory](rewind_understanding_long_videos_with_instructed_learnable_memory.md)
- [AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](../../NeurIPS2025/video_understanding/adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)

<!-- RELATED:END -->
