---
title: >-
  [论文解读] QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning
description: >-
  [ACL2025][语音质量评估] 本文提出 QualiSpeech，首个包含 11 个维度标注和详细自然语言推理描述的语音质量评估数据集，以及配套的评测基准，证明了微调后的听觉 LLM 能生成关于噪声和失真的详细描述，并展示了推理增强质量评估的潜力。
tags:
  - ACL2025
  - 语音质量评估
  - 自然语言描述
  - 听觉大语言模型
  - 低级语音感知
  - 推理
---

# QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning

**会议**: ACL2025  
**arXiv**: [2503.20290](https://arxiv.org/abs/2503.20290)  
**代码**: [HuggingFace: tsinghua-ee/QualiSpeech](https://huggingface.co/datasets/tsinghua-ee/QualiSpeech)  
**领域**: others (语音质量评估)  
**关键词**: 语音质量评估, 自然语言描述, 听觉大语言模型, 低级语音感知, 推理  

## 一句话总结

本文提出 QualiSpeech，首个包含 11 个维度标注和详细自然语言推理描述的语音质量评估数据集，以及配套的评测基准，证明了微调后的听觉 LLM 能生成关于噪声和失真的详细描述，并展示了推理增强质量评估的潜力。

## 研究背景与动机

语音质量评估是评价语音合成系统和通信网络失真的核心任务。当前主流方法主要聚焦于 **MOS 分数预测**——生成一个代表感知质量的数值分数。虽然数值分数便于比较，但无法揭示评分背后的原因，不提供具体的质量维度分析。

自然语言描述能提供更丰富的反馈。例如，"在 1.5~2.5 秒存在类似电流质量的失真声音"比单独的失真分数具有更丰富的信息。但现有数据集缺乏支持自然语言评估所需的全面标注。

同时，随着听觉 LLM（如 SALMONN、Qwen-Audio）的发展，自然语言语音质量评估在技术上变得可行，但低级语音感知任务在这些模型的训练和评测中仍被忽视。

## 方法详解

### 数据集构建

#### 数据来源

QualiSpeech 训练集包含 **10,558** 个样本，来源均衡分布：

- **合成语音**：
    - BVCC 数据集（历年 Blizzard/VCC 挑战赛样本）
    - 10 个最新开源 TTS 模型（ChatTTS、XTTS v2、CosyVoice、F5-TTS、E2 TTS、OpenVoice V1/V2、Parler-TTS Mini/Large、VoiceCraft-830M）
    - 每个 TTS 模型生成 72 个样本，句子来源于 SOMOS 语料（涵盖对话、新闻、维基百科等 10 个领域）
    - 20% 合成数据与噪声混合（DNS Challenge 噪声源，SNR 0-15dB）

- **真实语音**：
    - NISQA 数据集（模拟和实况通信网络失真）
    - GigaSpeech（有声书、播客、YouTube 录音）

#### 标注流程（三步法）

**步骤 1：听力测试标注**

11 个低级语音特征的详细标注：
- **7 个数值评分**（5 分制）：噪声、失真、语速、连续性、聆听努力度、自然度、整体质量
- **4 个具体描述**：噪声（类型+时间）、失真（类型+时间）、不自然停顿（时间）、声音特征（年龄/性别/语调）

**步骤 2：GPT 生成自然语言描述**

将所有标注维度输入 GPT-4o-mini，以 Chain-of-Thought 格式生成描述：先逐维度分析低级特征，再综合得出整体质量评估。

**步骤 3：人工修正**

标注者审核并修正：
- 描述与标注的不一致
- GPT 的幻觉或不支持的断言
- 补充遗漏的维度
- 改善推理逻辑的连贯性

### QualiSpeech Benchmark

建立多选题基准测试，覆盖 7 个低级语音理解维度。听觉 LLM 需为语音样本选择最合适的分数，评估其低级语音感知能力。

### 评估指标

- **数值评分**：PCC（Pearson 相关系数）
- **具体描述**：
    - Precision/Recall：检测噪声/失真存在的能力
    - GPT 生成的 Correlation Score：描述整体相关性
    - IoU：预测时间段与真实值的交集比
- **自然语言描述**：先用 GPT 提取各维度信息，再应用对应指标

## 实验

### 开源听觉 LLM 在 Benchmark 上的表现

| 模型 | Noise PCC | Distortion PCC | Overall PCC |
|------|-----------|----------------|-------------|
| SALMONN-7B | 0.003 | 0.013 | 0.084 |
| SALMONN-13B | 0.001 | 0.002 | 0.100 |
| Qwen-Audio-Chat | 0.014 | -0.003 | 0.250 |
| Qwen2-Audio-7B | -0.048 | 0.056 | 0.112 |
| WavLLM | -0.021 | -0.069 | 0.071 |

**结论**：现有开源听觉 LLM 在低级语音质量评估上几乎完全失效。SALMONN-7B 对 80%+ 样本预测相同分数，存在严重的数值偏好问题。

### 微调模型学习低级特征

基于 SALMONN-7B 微调（Whisper + BEATs 编码器，Vicuna 骨干，仅训练 Q-former + LoRA）：

| 训练策略 | Noise PCC | Distortion PCC | Overall PCC |
|---------|-----------|----------------|-------------|
| basic | 0.721 | 0.553 | 0.597 |
| balance | 0.696 | 0.547 | 0.600 |
| joint | 0.693 | 0.595 | 0.636 |
| **joint + balance** | **0.696** | **0.614** | **0.660** |

- 噪声 PCC 达 ~0.7，说明模型能区分语义和非语义成分
- 联合训练提升了连续性等维度，无显著退化
- **具体描述方面更有亮点**：噪声/失真的 IoU 达 ~0.8，模型能精准定位时间段；语音性别分类准确率达 98%

### 自然语言描述学习

| 描述格式 | Noise PCC | Distortion PCC | Overall PCC |
|---------|-----------|----------------|-------------|
| revised concise | 0.656 | 0.579 | 0.630 |
| concise with num | **0.703** | 0.571 | 0.622 |
| concise | 0.642 | 0.559 | 0.582 |
| detailed | 0.686 | 0.518 | 0.572 |

- 听觉 LLM 确实能生成段落式自然语言语音质量评估
- 在描述中加入数值分数能提升输出质量
- 修正 GPT 幻觉对高质量评估至关重要
- 描述长度对性能影响较小

### 推理能力探索

| 模型 | Overall 预测准确率 |
|------|-------------------|
| Vicuna-v1.5-7B（基于真实特征推理） | 0.28 |
| GPT-4o-mini（基于真实特征推理） | **0.46** |

- 微调模型在自然语言推理方面未能通过推理提升整体分数预测
- 但 GPT-4o-mini 使用真实低级特征推理时超越所有微调模型
- 说明推理能力受限于 LLM backbone 的推理能力，而非方法本身的限制

### 关键发现

1. 微调后模型生成的噪声/失真描述具有很高的时间精度（IoU~0.8）
2. 联合训练多个维度不会造成冲突
3. 仅在一种数据类型上训练会导致对其他类型的泛化很差
4. 多源数据融合能带来全面性能提升

## 亮点与洞察

1. **首个自然语言语音质量评估数据集**：填补了语音质量评估从数值评分迈向自然语言描述的关键空白
2. **11 维度全面标注**：比 NISQA（4 维度）更全面，且融合数值和描述两种标注形式
3. **合成+真实语音统一评估**：以往研究通常将两者分开处理，QualiSpeech 首次统一
4. **三步标注流程务实高效**：GPT 生成+人工修正的方式兼顾了质量和效率
5. **揭示了推理潜力**：虽然当前模型推理能力有限，但用更强的 LLM backbone 有望实现基于推理的质量评估

## 局限性

1. 每个样本仅由一名标注者评估（MOS 通常需要多人评估）
2. 部分语音维度和数据来源未覆盖
3. 微调模型受限于 LLM backbone（Vicuna-7B）的推理能力，未能充分发挥自然语言推理的潜力
4. 英语数据集，未覆盖其他语言

## 相关工作

- **语音质量评估数据集**：BVCC（合成语音 MOS）、NISQA（通信网络失真+多维度评分）、SOMOS（句子级评估）
- **听觉 LLM**：SALMONN、Qwen-Audio、WavLLM 等在高级语言理解上表现出色，但低级感知被忽视
- **LLM 用于语音质量评估**：现有工作仍聚焦 MOS 预测，未充分利用 LLM 的自然语言生成能力

## 评分

⭐⭐⭐⭐ — 数据集设计全面且创新，填补了重要空白。实验系统地验证了听觉 LLM 在低级语音特征理解上的能力和不足。不足在于微调模型的推理能力受限于较弱的 backbone，且单标注者的数据质量仍有提升空间。

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] A Multi-Persona Framework for Argument Quality Assessment](a_multi-persona_framework_for_argument_quality_assessment.md)
- [\[ACL 2025\] Cooperating and Competing Through Natural Language](cooperating_and_competing_through_natural_language.md)
- [\[ACL 2025\] Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation](leveraging_unit_language_guidance_to_advance_speech_modeling_in_textless_speech-.md)
- [\[ACL 2025\] SPOT: Bridging Natural Language and Geospatial Search for Investigative Journalists](spot_bridging_natural_language_and_geospatial_search_for_investigative_journalis.md)
- [\[ACL 2025\] ChildMandarin: A Comprehensive Mandarin Speech Dataset for Young Children Aged 3-5](childmandarin_a_comprehensive_mandarin_speech_dataset_for_young_children_aged_3-.md)

<!-- RELATED:END -->
