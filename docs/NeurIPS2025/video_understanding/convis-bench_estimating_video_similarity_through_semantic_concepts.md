---
title: >-
  [论文解读] ConViS-Bench: Estimating Video Similarity Through Semantic Concepts
description: >-
  [NeurIPS 2025][视频理解][video similarity] 提出基于语义概念的视频相似度估计任务 ConViS 及配套 benchmark ConViS-Bench（610对视频、16领域、5概念），系统评测了10+主流模型在概念条件下的视频比较能力…
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "video similarity"
  - "benchmark"
  - "semantic concepts"
  - "多模态"
  - "video retrieval"
---

# ConViS-Bench: Estimating Video Similarity Through Semantic Concepts

**会议**: NeurIPS 2025  
**arXiv**: [2509.19245](https://arxiv.org/abs/2509.19245)  
**代码**: [GitHub](https://github.com/benedettaliberatori/convisbench)  
**领域**: 视频理解  
**关键词**: video similarity, benchmark, semantic concepts, Large Multimodal Models, video retrieval

## 一句话总结

提出基于语义概念的视频相似度估计任务 ConViS 及配套 benchmark ConViS-Bench（610对视频、16领域、5概念），系统评测了10+主流模型在概念条件下的视频比较能力，揭示当前模型在时序结构和空间语境理解上的显著短板。

## 研究背景与动机

- **全局相似度粒度不足**：传统视频相似度方法仅计算一个全局分数（如嵌入空间余弦相似度），无法解释"哪些方面相似、哪些方面不同"，对下游应用（如检索、异常检测）缺乏可解释性。
- **人类比较视频是多概念的**：认知科学研究表明，人类自然地沿语义维度（动作、主体、地点等）选择性地关注和比较事件，视频相似性取决于关注哪个概念，而非一个固定量。
- **现有视频差异描述局限于窄领域**：VidDiffBench 仅关注单一概念（动作差异）且仅覆盖5个领域，StepDiff 局限于烹饪类视频，均为纯文本描述而缺少量化分数。
- **LMM 视频理解能力需要新评测维度**：现有 benchmark（Video-MME、MVBench 等）主要通过问答评估，缺少对模型"概念级比较推理"能力的系统测试。
- **概念条件检索的需求**：实际应用中常需按特定概念检索视频（如"相同动作不同主体"），但现有方法不支持这种细粒度条件化。
- **量化+可解释的空白**：video differencing 方法提供文本描述但不量化，全局方法量化但不可解释——ConViS 旨在同时提供结构化量化和可解释性。

## 方法详解

### 整体框架：Concept-based Video Similarity (ConViS)

给定视频对 $(V_1, V_2)$ 和预定义概念集 $\mathcal{C} = \{C_1, \ldots, C_K\}$（自然语言表述），ConViS 计算每个概念上的相似度分数：

$$s(V_1, V_2 \mid C_i) \in \mathbb{R}$$

各概念分数可通过加权聚合为整体分数：$s(V_1, V_2) = \sum_{i=1}^{K} \lambda_i \cdot s(V_1, V_2 \mid C_i)$，其中 $\sum \lambda_i = 1$。这种设计兼具**灵活性**（用户可引入任意概念）和**可组合性**（可按需加权）。

### 关键设计1：ConViS-Bench 数据集构建

- **概念选择**：基于认知科学（人类沿语义和时间特征组织事件记忆），定义5个通用概念——**main action**（主要动作）、**main subjects**（主要主体）、**main objects**（主要物体）、**location**（地点）、**order of actions**（动作顺序）。
- **视频来源**：从 FineVideo 数据集中选取16个视觉多样性强的领域（排除静态说话类），按事件时间戳裁剪为独立片段。
- **配对策略**：使用 DINOv2（视觉嵌入）和 Sentence-BERT（文本嵌入）分别计算余弦相似度，选择仅在单一模态上高相似的配对（确保既有共同点又有差异），最终人工筛选得 610 对。
- **标注流程**：通过 Prolific 招募150名标注者，每对视频按5个概念在 1-5 Likert 量表上打分，同时提供相似/差异的自由文本标签。平均每对6.2次标注，剔除7.75%低质量标注后保留。

### 关键设计2：LMM 概念条件评测

将两个视频的帧拼接输入 LMM，通过 prompt 要求模型"仅基于 \<concept\> 输出1-5的相似度分数"。评测使用 Spearman's $\rho$ 和 Kendall's $\tau$ 衡量与人类判断的一致性。覆盖9个开源模型（mPLUG-Owl3、LLaVA系列、Qwen-VL系列、InternVL系列）和 Gemini 2.0-Flash。

### 关键设计3：全局表征偏好探测

设计三类方法探测全局相似度隐式偏向哪些概念：① Video-to-video（VideoMAE/DINOv2 嵌入余弦相似度）；② Text-to-text（先用 LMM 生成描述再用 Sentence-BERT 比较文本）；③ Cross-modal（CLIPScore/VQAScore 跨模态对齐分数）。

### 关键设计4：概念条件检索任务

给定锚视频和目标概念，从4个候选视频中检索最相似的。构建532个概念级偏序排名，用 R@1/P@1/F1@1 评测。

## 损失函数与训练

本文是一个 **benchmark 论文**，不涉及模型训练。所有评测均使用现有模型的 zero-shot 推理能力，通过 prompt engineering 在概念条件下获取预测分数。

## 实验

### 表1：LMM 概念条件相似度估计（Spearman's ρ × 100）

| 模型 | Main Action | Main Subjects | Main Objects | Location | Order of Actions |
|:---|:---:|:---:|:---:|:---:|:---:|
| mPLUG-Owl3-7B | 30.64 | 20.59 | 28.53 | 21.00 | 23.11 |
| LLaVA-OV-0.5B | 1.95 | -5.05 | -4.00 | 5.66 | 1.30 |
| **LLaVA-OV-7B** | **51.76** | **48.43** | **58.64** | **58.94** | **41.02** |
| LLaVA-Video-7B | 44.17 | 39.81 | 45.85 | 55.96 | 41.25 |
| Qwen2.5-VL-7B | 37.88 | 17.53 | 26.97 | 23.63 | 23.85 |
| InternVL2.5-8B | 28.70 | 28.60 | 25.06 | 19.64 | 18.15 |
| InternVL3-8B | 40.69 | 36.54 | 42.50 | 45.47 | 32.74 |

**LLaVA-OV-7B 在所有概念上一致最优**，但即便最好的模型在 order of actions 上也仅 41.02，显著低于其他概念。InternVL 系列虽然预训练数据包含 FineVideo，但表现并不突出，暗示预训练数据包含≠真正理解。

### 表2：全局表征在各概念上的隐式偏好（Spearman's ρ × 100）

| 模型 | 方法 | Main Action | Main Subjects | Main Objects | Location | Order of Actions |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| VideoMAE | Cosine | 13.0 | 23.1 | 13.2 | 37.8 | 15.1 |
| DINOv2 | Cosine | 33.3 | 40.9 | 37.4 | **57.4** | 34.6 |
| mPLUG-Owl3 | SBERT | **52.1** | 45.5 | **55.1** | 28.4 | **49.9** |
| LLaVA-OV | VQAScore | 51.1 | **55.8** | 58.3 | 46.5 | 48.1 |

**关键发现**：视觉编码器（DINOv2）偏向 location，文本方法偏向 action/objects，所有方法在 order of actions 上持续低分。VQAScore 整体平衡性最好。

### 概念条件检索结果亮点

LLaVA-OV-7B 在 main subjects 上 P@1 达 66.4%，在 location 上 P@1 达 68.7%，显著超越随机基线（~35-50%），但 main action 的 P@1 仅 54.8%，说明当前模型在按动作检索时仍较弱。

## 亮点

- **任务定义新颖且认知科学扎实**：ConViS 恰好填补了全局相似度（量化但不可解释）和视频差异描述（可解释但不量化）之间的空白，概念选择有认知科学理论支撑。
- **数据集高质量**：150名标注者、610对视频、16领域、5概念、每对平均6.2次标注，配有标注质量控制和 IRB 审批。
- **评测全面深入**：覆盖 LMM 概念评分、全局表征偏好探测、概念条件检索三个维度，帧数消融实验揭示了时序依赖性和记忆效应。
- **发现有洞察力**：order of actions 是所有模型的软肋、InternVL 预训练包含测试数据但不帮助概念理解、视觉/文本表征各有概念偏向。

## 局限性

- **概念集有限**：仅5个通用概念，可能遗漏领域特定的重要维度（如"技能水平"、"拍摄视角"、"光照条件"——标注者自定义概念中频繁出现）。
- **数据集规模较小**：610对视频对于训练概念感知模型来说偏少，主要作为测试集使用。
- **无训练方法**：纯 benchmark 论文，未提出学习概念级相似度的方法，仅评测现有模型的 zero-shot 能力。
- **标注者一致性一般**：Krippendorff's α 最高仅 0.361（location），main subjects 仅 0.244，反映概念级相似度本身带有主观性。
- **Gemini 预训练数据不透明**：私有模型的预训练数据可能已含 FineVideo，影响评测公平性。

## 相关工作

- **视频全局相似度**（DNS、SSVL等）：计算单一全局分数但不可解释，ConViS 是对其的结构化升级。
- **视频差异描述**（VidDiffBench [Burgess+ ICLR'25]、StepDiff [Nagarajan+ CVPR'24]）：提供文本差异描述但不量化、领域受限（动作/烹饪），ConViS 覆盖16领域且提供量化分数。
- **图像概念相似度**（Achille+ CVPR'24）：定义图像间的概念相似度，ConViS 将其扩展到更复杂的视频域。
- **LMM 视频 benchmark**（Video-MME、MVBench、TempCompass等）：主要测 QA 能力，ConViS 独特地测试概念级比较推理。
- **组合视频检索**（CoVR等）：查询=参考视频+文本修改，ConViS 支持多概念维度探索且提供显式量化分数。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 概念级视频比较是一个定义清晰且有认知科学根基的新任务
- 实验充分度: ⭐⭐⭐⭐ — 三维度评测+帧数消融+记忆效应分析，覆盖面广
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，动机论述有说服力
- 价值: ⭐⭐⭐⭐ — 为视频理解社区提供了重要的评测新维度，发现具有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution](cloud4d_estimating_cloud_properties_at_a_high_spatial_and_temporal_resolution.md)
- [\[NeurIPS 2025\] Disentangled Concepts Speak Louder Than Words: Explainable Video Action Recognition](disentangled_concepts_speak_louder_than_words_explainable_video_action_recogniti.md)
- [\[CVPR 2025\] Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs](../../CVPR2025/video_understanding/q-bench-video_benchmark_the_video_quality_understanding_of_lmms.md)
- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [\[CVPR 2025\] Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking](../../CVPR2025/video_understanding/similarity-guided_layer-adaptive_vision_transformer_for_uav_tracking.md)

</div>

<!-- RELATED:END -->
