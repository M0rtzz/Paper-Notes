---
title: >-
  [论文解读] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering
description: >-
  [CVPR 2025][视频理解][第一人称视频问答] 提出 EgoTextVQA 基准，包含 1.5K 第一人称视频和 7K 场景文字相关问答对，揭示了当前 MLLM 在以自我中心视角进行实时场景文字问答辅助时的严重不足（最佳模型 Gemini 1.5 Pro 仅约 33% 准确率）。
tags:
  - CVPR 2025
  - 视频理解
  - 第一人称视频问答
  - 场景文字识别
  - 多模态大语言模型
  - 基准数据集
  - 自中心视觉
---

# EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering

**会议**: CVPR 2025  
**arXiv**: [2502.07411](https://arxiv.org/abs/2502.07411)  
**代码**: https://github.com/zhousheng97/EgoTextVQA (有)  
**领域**: 视频理解  
**关键词**: 第一人称视频问答, 场景文字识别, 多模态大语言模型, 基准数据集, 自中心视觉

## 一句话总结

提出 EgoTextVQA 基准，包含 1.5K 第一人称视频和 7K 场景文字相关问答对，揭示了当前 MLLM 在以自我中心视角进行实时场景文字问答辅助时的严重不足（最佳模型 Gemini 1.5 Pro 仅约 33% 准确率）。

## 研究背景与动机

### 领域现状

**领域现状**：现有场景文字 VQA 数据集（如 TextVQA、ST-VQA）假设用户能拍到清晰聚焦的图片且问题直接指向文字区域，这在实际应用中不太现实。考虑视障人士等场景，用户很难拍出清晰图片或指向文字区域。而已有的视频文字问答数据集（如 RoadTextVQA）仍然假设用户知道场景文字的位置，问题设计过于简单，只需要 OCR 抽取即可回答。

本文的动机是构建一个更贴近真实需求的第一人称场景文字视频问答基准：

### 现有痛点

**现有痛点**：问题反映**真实用户需求**，而非直接指向场景文字

### 核心矛盾

**核心矛盾**：支持**实时流式问答**（每个问题有时间戳，模型只能访问时间戳前的内容）

### 解决思路

**解决思路**：覆盖**室内外多种场景**（户外驾驶 + 室内家务）

## 方法详解

### 整体框架

EgoTextVQA 是一个**评测基准**而非方法论文。其核心贡献在于数据集的精心构建和全面的模型评测与启发式探索。

### 关键设计

1. **数据集构建流水线**:
    - 功能：从已有第一人称视频数据集中筛选、生成并精修高质量 QA 对
    - 核心思路：先用场景文字检测系统自动筛选含文字的视频（RoadTextVQA 阈值 15%，Ego4D 阈值 5%），再用 GPT-4o 按精心设计的 prompt 生成 QA 对，最后经 5 轮共 9 名标注员逐步筛选、修正和精修
    - 设计动机：自动生成保证多样性，多轮人工审核保证质量。最终只保留约 30% 的自动生成 QA

2. **实时流式问答设定**:
    - 功能：模拟真实场景中的即时问答
    - 核心思路：为每个问题设置时间戳，模型只能访问该时间戳之前的视频内容
    - 设计动机：现有基准允许访问完整视频，与实际辅助应用不符。实时 QA 子集的最高准确率仅 20.2%，远低于全集 33.4%

3. **多维度问题分类体系**:
    - 功能：支持细粒度的模型行为分析
    - 核心思路：户外分为 Location/Description/Direction/Intention Reasoning 等类别；室内分为 Hands-on/Shopping/Kitchen/Book-related/Gameplay 等场景
    - 设计动机：不同类型的问题考察模型的不同能力，便于定位模型弱点

### 损失函数 / 训练策略

本文为基准论文，不涉及模型训练。评估使用 GPT-4o mini 作为语义相似度评判器，输出两个指标：Accuracy（0-100%）和 Score（0-5 分）。

## 实验关键数据

### 主实验

| 模型 | EgoTextVQA-Outdoor Acc. | EgoTextVQA-Indoor Acc. | 类型 |
|------|------------------------|----------------------|------|
| Gemini 1.5 Pro | **33.4%** | **34.4%** | 闭源 |
| Gemini 1.5 Flash | 30.1% | 32.0% | 闭源 |
| GPT-4o | 30.3% | 28.3% | 闭源 |
| Qwen2-VL | 28.2% | 23.3% | 开源 |
| LLaVA-NeXT-Video | 19.5% | 25.4% | 开源 |
| Human | 43.1% | 27.7% | - |

### 启发式探索实验

| 策略 | Outdoor Acc. 变化 | Indoor Acc. 变化 | 说明 |
|------|------------------|-----------------|------|
| 视频 + 场景文字辅助 (GPT-4o) | 30.3→52.9 (+22.6) | 28.3→37.9 (+9.6) | OCR 辅助极大提升 |
| 高分辨率 QA 帧 (Qwen2-VL) | 28.2→46.8 (+18.6) | - | 高分辨率关键 |
| 单帧 vs 视频 (Gemini 1.5 Pro) | 33.4→30.4 (-3.0) | 34.4→15.8 (-18.6) | 室内需多帧推理 |
| 场景文字超分 1.5× (Qwen2-VL) | 28.2→34.1 (+5.9) | 23.3→22.3 (-1.0) | 户外有效 |

### 关键发现

1. **所有模型在 EgoTextVQA 上都表现很差**，最佳闭源模型仅 ~33%，而人类在室内场景甚至更低（27.7%），说明场景文字识别对人和模型都是挑战
2. **辅助 OCR 输入是提升最大的手段**：GPT-4o 在加入场景文字后户外准确率从 30.3% 跃升至 52.9%
3. **室内场景尤其依赖多帧时序推理**：单帧输入在室内导致高达 18.6% 的性能暴跌
4. **高分辨率图像对场景文字识别至关重要**，但需要平衡计算效率

## 亮点与洞察

- 数据集设计非常扎实：5 轮人工审核、问题不直接指向文字、模拟实时问答，都体现了对真实应用的深度思考
- 人类表现低于闭源模型（室内场景），暴露了问题的真正难度——不仅是文字识别，还需要外部知识
- 启发式探索部分非常系统，从时序定位、分辨率、OCR 辅助、多模态输入四个角度全面分析

## 局限与展望

- 答案多样性导致人类评分偏低，GT 答案可以进一步丰富
- 目前只有开放式问答，未来可扩展到多选题等形式
- GPT-4o mini 作为评估器可能引入偏差
- 室内视频分辨率较低（480×360），限制了场景文字识别能力

## 相关工作与启发

- 与 TextVQA/ST-VQA 等静态图像 VQA 不同，本文强调**动态视频中的第一人称视角**
- 与 QAEgo4D/AssistQ 等第一人称 VQA 不同，本文**聚焦场景文字**
- 启示：未来的视频理解模型需要同时具备高分辨率文字识别、时序推理和用户意图理解能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个聚焦第一人称场景文字实时问答的基准，定位精准
- 实验充分度: ⭐⭐⭐⭐⭐ 10个模型 + 4个方向的启发式探索，分析极为详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据集构建过程透明
- 价值: ⭐⭐⭐⭐ 揭示了 MLLM 在实际辅助应用中的关键瓶颈，对后续研究有指导意义

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] QA-TIGER: Question-Aware Gaussian Experts for Audio-Visual Question Answering](question-aware_gaussian_experts_for_audio-visual_question_answering.md)
- [\[NeurIPS 2025\] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark](../../NeurIPS2025/video_understanding/egogazevqa_egocentric_gaze_guided_video_question_answering.md)
- [\[CVPR 2025\] BIMBA: Selective-Scan Compression for Long-Range Video Question Answering](bimba_selective-scan_compression_for_long-range_video_question_answering.md)
- [\[CVPR 2026\] EgoPointVQA: Gesture-Based Egocentric Video Question Answering](../../CVPR2026/video_understanding/egopointvqa_gesture_based_egocentric_video_qa.md)
- [\[CVPR 2025\] HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation](hyperglm_hypergraph_for_video_scene_graph_generation_and_anticipation.md)

<!-- RELATED:END -->
