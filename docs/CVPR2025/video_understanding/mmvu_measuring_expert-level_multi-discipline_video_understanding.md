---
title: >-
  [论文解读] MMVU: Measuring Expert-Level Multi-Discipline Video Understanding
description: >-
  [CVPR 2025][视频理解][视频理解基准] 提出 MMVU 基准，包含 3,000 个专家标注的跨 27 个学科的视频理解题目，评估多模态基础模型在专业领域视频中的专家级知识推理能力，揭示即使最强模型仍显著落后于人类专家。
tags:
  - CVPR 2025
  - 视频理解
  - 视频理解基准
  - 专家级推理
  - 多学科评测
  - 领域知识
  - 多模态基础模型
---

# MMVU: Measuring Expert-Level Multi-Discipline Video Understanding

**会议**: CVPR 2025  
**arXiv**: [2501.12380](https://arxiv.org/abs/2501.12380)  
**代码**: https://github.com/yale-nlp/MMVU (有)  
**领域**: 视频理解  
**关键词**: 视频理解基准, 专家级推理, 多学科评测, 领域知识, 多模态基础模型

## 一句话总结

提出 MMVU 基准，包含 3,000 个专家标注的跨 27 个学科的视频理解题目，评估多模态基础模型在专业领域视频中的专家级知识推理能力，揭示即使最强模型仍显著落后于人类专家。

## 研究背景与动机

现有视频理解基准主要关注通用场景（动作识别、字幕生成等），缺乏对 **专业领域专家级推理** 的评估。然而，视频是许多专业领域（医疗、工程、科学研究）传递复杂动态信息的关键模态。例如分析化学反应视频，模型需要识别颜色变化等视觉线索并结合化学知识推理。

已有多学科基准（MMLU、MMMU 等）主要针对文本或图像，视频维度的专家级推理评测严重不足。唯一的相关工作 MMWorld 中仅 39.5% 的样本需要领域专业知识，且 76.4% 由 GPT-4V 自动生成。MMVU 通过完全人工从零标注、教科书引导的标注流程填补这一空白。

## 方法详解

### 整体框架

MMVU 构建分三阶段：(1) 前期准备——通过 133 名学生的用户研究确定 27 个学科，招募 67 名专家标注者；(2) 教科书引导的 QA 标注——标注者从教科书概念出发寻找 CC 许可视频并创建专家级问答；(3) 数据质量控制——包括基于时间的标注补偿和人工专家验证。

### 关键设计

1. **教科书引导的标注流程**: 标注者先从教科书中识别关键概念（如实验流程、机械操作等），再搜索 YouTube 上符合 CC 许可的相关视频，最后设计需要领域知识和专家推理才能回答的问题。这确保了知识的广度（覆盖教科书各章节）和推理深度（需要专业推理而非简单视觉识别）。每个样本附带专家标注的 **推理过程** 和 **相关领域知识**（链接到 Wikipedia 页面），支撑细粒度分析。

2. **严格的视频质量约束**: 视频必须是视觉密集型，排除音频（防止语音捷径）、排除屏幕文字过多的内容（如讲座录像），确保模型必须通过视觉理解才能回答。每个样本经专家验证确认必须看视频才能回答——纯文本或仅靠单帧不可作答。

3. **多层次人类基线评估**: 设计三阶段人类评测——闭卷（3.5小时，平均49.7%）、开卷（可查资料，86.8%）、Oracle（给定正确领域知识后修改，95.3%），提供了对任务难度的精确标定。

### 损失函数 / 训练策略

本文是评测基准论文，不涉及模型训练。评估使用两种 prompt 策略：Direct Answer 和 Chain-of-Thought (CoT)。准确率评估由 GPT-4o 进行答案提取和判断。涵盖 32 个前沿多模态模型，包括16系列开源和8系列闭源模型。

## 实验关键数据

### 主实验

| 模型 | 科学 | 医疗 | 人文社科 | 工程 | 测试集均值 |
|------|------|------|---------|------|-----------|
| 人类 (开卷) | 84.7 | 92.7 | 83.3 | 86.8 | 86.8 |
| o1 | 78.0 | 76.0 | 74.0 | 79.0 | 77.0 |
| Gemini 2.0 Flash Thinking | 71.2 | 73.4 | 67.3 | 69.1 | 69.5 |
| GPT-4o | 71.8 | 72.0 | 61.6 | 67.4 | 66.7 |
| Claude 3.5 Sonnet | 64.0 | 70.9 | 64.5 | 65.2 | 64.1 |
| Qwen2-VL-72B (开源最强) | 53.6 | 61.7 | 53.9 | 53.0 | 53.2 |
| 人类 (闭卷) | 54.7 | 42.7 | 44.7 | 56.7 | 49.7 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| GPT-4o Direct Answer → CoT | 65.4 → 66.7 (+1.3) | CoT 对 GPT-4o 提升有限 |
| Claude 3.5 Sonnet Direct → CoT | 53.1 → 64.1 (+11.0) | CoT 对 Claude 提升显著 |
| o1 (System-2 思维) | 77.0 | 长链推理带来最强表现 |
| Gemini 2.0 Flash Thinking | 69.5 | System-2 思维效果显著 |
| Qwen2-VL-72B (开源) vs GPT-4o | 53.2 vs 66.7 | 开源与闭源差距 13.5% |

### 关键发现

1. 即使最强模型 o1 (77.0%) 仍明显落后于人类专家开卷水平 (86.8%)，差距达 9.8%
2. GPT-4o 在 MMLU（文本）和 MMMU（图像）上已接近人类，但在视频专家推理上差距仍大
3. CoT 推理总体上提升性能，但不同模型收益差异巨大（Claude +11.0% vs GPT-4o +1.3%）
4. System-2 思维（o1、Gemini Thinking）在专家级视频推理中表现优势显著
5. 六大错误类型：领域知识推理错误 (27%) > 视觉感知中缺乏领域知识 (20%) = 过度依赖文本 (20%) > 视觉感知错误 (18%)

## 亮点与洞察

- 每个样本都附带专家级推理过程和领域知识标注，这使得错误分析可以精确定位模型在"看到什么"、"知道什么"、"如何推理"各环节的失败
- 教科书引导的标注确保了知识覆盖的系统性，而非随意选题
- 限定 CC 许可视频虽然增加了标注难度，但解决了版权问题，使基准可持续使用
- 三阶段人类评测（闭卷→开卷→Oracle）精确标定了任务难度，为模型能力分析提供有力参照

## 局限与展望

- 限定 CC 许可导致某些领域（如体育）视频稀缺，覆盖面受限
- 视频平均长度仅 51.4 秒，未涉及长视频场景
- 开放题评估依赖 GPT-4o，可能存在评判偏差
- 音频信息被排除，但某些专业场景（如音乐、语言学习）中音频很重要
- 当前仅 27 个学科，可进一步扩展覆盖面

## 相关工作与启发

- 与 MMLU（文本）、MMMU（图像）形成模态递进的评测体系，MMVU 填补了视频模态的空白
- 与 MMWorld 相比，MMVU 的标注质量更高（100% 人工 vs 76% GPT 生成），且专注于真正需要领域知识的问题
- 教科书引导的标注流程可推广到其他需要系统知识覆盖的基准构建中
- 错误分类法（视觉感知/领域知识/推理/文本依赖）为模型改进提供了清晰方向

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个高质量多学科视频专家级推理基准，填补重要空白
- 实验充分度: ⭐⭐⭐⭐⭐ 32个模型全面评测，详细的错误分析和人类基线
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，标注流程严谨，分析深入
- 价值: ⭐⭐⭐⭐⭐ 对推动多模态模型在专业领域的应用有重要价值

<!-- RELATED:START -->

## 相关论文

- [Omni-RGPT: Unifying Image and Video Region-level Understanding via Token Marks](omni-rgpt_unifying_image_and_video_region-level_understanding_via_token_marks.md)
- [MLVU: Benchmarking Multi-task Long Video Understanding](mlvu_benchmarking_multi-task_long_video_understanding.md)
- [MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](../../NeurIPS2025/video_understanding/muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)
- [Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding](beyond_single-sample_reliable_multi-sample_distillation_for_video_understanding.md)
- [TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding](../../ICCV2025/video_understanding/timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)

<!-- RELATED:END -->
