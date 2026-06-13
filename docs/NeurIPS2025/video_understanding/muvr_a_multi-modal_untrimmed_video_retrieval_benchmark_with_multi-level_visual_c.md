---
title: >-
  [论文解读] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence
description: >-
  [NeurIPS 2025][视频理解][视频检索] 提出 MUVR 基准，面向长视频平台的多模态未剪辑视频检索任务，设计了以视频为中心的多模态查询格式（视频+文本+标签+掩码）和六级视觉对应匹配准则，包含 53K 视频和 1050 个查询，系统评估了检索模型和 MLLM 的局限性。 领域现状：视频检索是推荐系统和内容搜索的…
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "视频检索"
  - "多模态查询"
  - "未剪辑视频"
  - "多级视觉对应"
  - "基准数据集"
---

# MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence

**会议**: NeurIPS 2025  
**arXiv**: [2510.21406](https://arxiv.org/abs/2510.21406)  
**代码**: [GitHub](https://github.com/debby-0527/MUVR)  
**领域**: 视频理解  
**关键词**: 视频检索, 多模态查询, 未剪辑视频, 多级视觉对应, 基准数据集

## 一句话总结

提出 MUVR 基准，面向长视频平台的多模态未剪辑视频检索任务，设计了以视频为中心的多模态查询格式（视频+文本+标签+掩码）和六级视觉对应匹配准则，包含 53K 视频和 1050 个查询，系统评估了检索模型和 MLLM 的局限性。

## 研究背景与动机

**领域现状**：视频检索是推荐系统和内容搜索的核心技术。现有任务包括文本到视频检索（TVR）、组合视频检索（CVR）、细粒度视频检索（FVR）等，但各有局限。

**现有痛点**：(1) 纯文本查询难以描述精细视觉信息；(2) 纯视频查询会引入无关视觉信息；(3) 现有基准仅支持单一匹配准则（如近重复、事件级），无法覆盖多样化视频类别；(4) 大多数基准基于剪辑后视频和一对一检索，不符合实际平台的使用场景。

**核心矛盾**：真实视频平台需要的是一对多检索 + 未剪辑视频 + 多模态查询 + 多类别覆盖，但没有基准同时满足这些要求。

**本文目标**：构建一个与真实视频平台应用对齐的综合性视频检索基准。

**切入角度**：以视频查询为主、文本描述为辅，结合标签和掩码 prompt，在未剪辑视频库上进行一对多检索。

**核心 idea**：提出多级视觉对应（copy、event、scene、instance、action、others）作为通用检索匹配准则，设计五个分区覆盖多样化视频类别。

## 方法详解

### 整体框架

MUVR 基准包含三个版本：
- **MUVR-Base**：53K 视频，1050 个多模态查询，84K 标注匹配
- **MUVR-Filter**：在 Base 基础上标注 74K 多标签标签，支持标签过滤检索（9979 个查询）
- **MUVR-QA**：200 个判别问题，评估 MLLM 的重排序能力

### 关键设计

1. **多模态查询设计**：每个查询包含四个组件：

    - **视频查询**：携带主要视觉信息，适合描述难以用文字表达的细节
    - **文本描述**：平均 20 个词，指定关键视觉内容和检索需求
    - **标签 Prompt**：用户指定期望/不期望的视频特征（如"动画风格"、"第一视角"），支持精细过滤
    - **掩码 Prompt**：使用 SAM2 标注关键区域，引导检索模型关注视频中的特定部分

2. **多级视觉对应**：根据用户感兴趣的视频内容层次定义六级对应：

    - **Copy**：目标是查询的复制/编辑版本
    - **Event**：共享相同事件（时空交叉）
    - **Scene**：共享相同场景/背景/地区
    - **Instance**：共享相同实例/物体
    - **Action**：共享相同人体动作
    - **Others**：主观感觉相关

3. **五分区设计（Partitions）**：News（新闻事件）、Region（旅行场景）、Instance（商品宠物）、Dance（舞蹈动作）、Others（梗/电影），每个分区侧重不同的视觉对应层级和视频类别。

4. **重排序评分（Reranking Score）**：针对 MLLM 重排序能力设计的评估指标。每个查询包含一个真阳性和一个假阳性，MLLM 可能产生四种结果：

    - 10（正确保留真+移除假）：+1 分
    - 11（全保留，无操作）：0 分
    - 00（全移除）：-1 分
    - 01（错误翻转）：-2 分

### 损失函数 / 训练策略

MUVR 是一个评估基准而非训练方法。检索评分计算：

$$S_v = \text{Score}(V_{\text{query}}, V_{\text{target}}), \quad S_t = \text{Score}(T_{\text{description}}, V_{\text{target}})$$
$$S_{tv} = (S_t + S_v)/2, \quad S_{tag} = S_{tv} + p \times \text{Score}(T_{\text{tag}}, V_{\text{target}})$$

其中 $p = \pm 0.3$ 根据标签 prompt 的正负符号确定。

## 实验关键数据

### 主实验

**MUVR-Base 多模态查询检索性能（mAP %）：**

| 方法 | 平均 mAP | News | Others | Instance | Region | Dance |
|------|----------|------|--------|----------|--------|-------|
| CLIP (RN50x4) | 42.9 | 49.4 | 53.6 | 46.5 | 43.8 | 21.2 |
| EVA-CLIP | **58.0** | **63.1** | 66.1 | **68.2** | **63.8** | 28.7 |
| InternVideo2 | 52.1 | 57.3 | **66.3** | 55.3 | 52.5 | 28.9 |
| S2VS | 47.2 | 51.3 | 63.7 | 49.5 | 49.1 | 22.5 |
| CoVR | 43.3 | 50.5 | 54.3 | 46.9 | 44.0 | 20.8 |

**MUVR-QA MLLM 重排序评估：**

| 方法 | 参数 | 帧数 | 准确率 (All) | 重排序分 (All) |
|------|------|------|-------------|---------------|
| InternVL2 (多图) | 8B | 6 | 58.5 | -0.23 |
| GPT-4o (多图) | N/A | 6 | **65.0** | **0.19** |
| Gemini-2.0-Flash | N/A | 12 | 63.5 | 0.07 |
| InternVL2.5 (文本) | 8B | 12 | 58.5 | -0.37 |

### 消融实验

**查询格式对检索性能的影响（EVA-CLIP, mAP %）：**

| 查询格式 | 平均 mAP | 最佳分区 |
|----------|----------|----------|
| 纯文本 | 43.0 | Instance (59.7) |
| 纯视频 | 50.7 | Others (59.2) |
| 多模态 | **58.0** | Instance (68.2) |
| 多模态+标签 | 34.0 (Filter) | News (38.3) |

### 关键发现

- **视频查询比文本重要**：纯视频检索普遍优于纯文本（50.7 vs 43.0 mAP），因为视频能精确表达文字难以描述的细节
- **多模态查询有显著增益**：视频+文本组合带来 7.3% mAP 提升（58.0 vs 50.7），说明两种模态互补
- **不同分区需要不同能力**：VLM 在 Instance/Region/News 分区表现最好（静态空间理解），视频模型在 Dance/Others 分区更强（动态时序理解）
- **所有方法在 Dance 分区表现最差**：最高仅 28.9% mAP，说明动作级视觉对应仍是巨大挑战
- **标签 Prompt 对 MLLM 挑战极大**：某些模型在无标签问题上达 70%+ 准确率，但加入标签后显著下降
- **MLLM 重排序能力不足**：大多数模型重排序分为负值，说明现有 MLLM 尚不可靠用于视频重排序

## 亮点与洞察

- **基准设计精心且实用**：五分区设计覆盖了视频平台的核心类别，避免了单一类别基准的偏见
- **多级视觉对应是一个有价值的形式化**：将用户检索意图抽象为 6 个层次，比简单的语义匹配更精确
- **标签 Prompt 是创新点**：允许用户用简单的标签过滤结果，贴近实际产品体验
- **重排序评分设计合理**：区分了"无操作"和"错误翻转"并赋予不同惩罚，反映了实际重排序场景的需求
- 规模大（53K 视频）、标注质量高（专业标注员+双轮核验）

## 局限与展望

- 数据来源限于 Bilibili 平台，可能存在中文视频的偏差，需要验证在英文平台的适用性
- 视频裁切为最长 2 分钟，对更长视频的检索场景覆盖不足
- 标签 Prompt 的设计（$p = \pm 0.3$ 固定权重）较为简单，可探索自适应权重
- MUVR-QA 仅包含 200 个问题，规模有限
- 未探索端到端的多模态查询理解模型，仅评估现有模型

## 相关工作与启发

- 与 FIVR（事件级检索）相比，MUVR 扩展到了六级视觉对应和多视频类别
- 与 CoVR（组合检索）相比，MUVR 支持一对多检索和未剪辑视频
- 暴露了 Dance 分区的极低性能，启发未来工作关注动作级视频理解
- MLLM 重排序的不可靠性启发了 retrieval + MLLM 两阶段框架的进一步研究

## 评分

- 新颖性: ⭐⭐⭐⭐ 多级视觉对应和多模态查询范式新颖，但核心是数据集贡献而非方法创新
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个检索模型 + 6 个 VLM + 10 个 MLLM 的系统评估，分析全面
- 写作质量: ⭐⭐⭐⭐⭐ 基准设计描述详尽，发现总结清晰，可读性极好
- 价值: ⭐⭐⭐⭐⭐ 填补了多模态未剪辑视频检索领域的基准空白，对社区有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] PixFoundation 2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?](pixfoundation_20_do_video_multi-modal_llms_use_motion_in_visual_grounding.md)
- [\[ICCV 2025\] Multi-modal Multi-platform Person Re-Identification: Benchmark and Method](../../ICCV2025/video_understanding/multi-modal_multi-platform_person_re-identification_benchmark_and_method.md)
- [\[NeurIPS 2025\] When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions](when_one_moment_isnt_enough_multi-moment_retrieval_with_cross-moment_interaction.md)
- [\[ICCV 2025\] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding](../../ICCV2025/video_understanding/dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)
- [\[CVPR 2025\] MMVU: Measuring Expert-Level Multi-Discipline Video Understanding](../../CVPR2025/video_understanding/mmvu_measuring_expert-level_multi-discipline_video_understanding.md)

</div>

<!-- RELATED:END -->
