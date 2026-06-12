---
title: >-
  [论文解读] Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition
description: >-
  [ACL 2025][音频/语音][上下文偏置] 本文提出利用知识增强的外部语言模型（KELM）进行上下文偏置，在端到端语音识别中通过动态融合外部领域知识和偏置词表，大幅提升稀有词和专有名词的识别准确率。
tags:
  - "ACL 2025"
  - "音频/语音"
  - "上下文偏置"
  - "外部语言模型"
  - "端到端语音识别"
  - "热词识别"
  - "知识增强"
---

# Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition

**会议**: ACL 2025  
**领域**: 语音识别  
**关键词**: 上下文偏置、外部语言模型、端到端语音识别、热词识别、知识增强

## 一句话总结
本文提出利用知识增强的外部语言模型（KELM）进行上下文偏置，在端到端语音识别中通过动态融合外部领域知识和偏置词表，大幅提升稀有词和专有名词的识别准确率。

## 研究背景与动机

**领域现状**：端到端（E2E）语音识别模型（如 CTC、RNN-T、Attention-based Encoder-Decoder）已成为主流，它们将声学模型、语言模型和发音词典统一在单一模型中。然而，这类模型在识别训练集中罕见或未出现的词汇（如人名、专业术语、新产品名称）时表现不佳。

**现有痛点**：现有的上下文偏置（contextual biasing）方法主要分为两类：（1）浅层融合方法（如 WFST-based boosting），在解码时提升偏置词的概率，但缺乏语义理解，容易产生误触发；（2）深度偏置方法（如注意力偏置），将偏置词表通过注意力机制融入模型，但需要重新训练模型，且偏置词表规模受限。两类方法都缺乏对上下文语义的充分利用。

**核心矛盾**：偏置方法需要在"偏置强度"和"误识别率"之间取得平衡——偏置过强会把正常词误识别为偏置词，偏置过弱则无法有效召回目标词汇。根本原因在于现有方法缺乏语义判断能力，无法根据上下文判断何时应该激活偏置。

**本文目标**：设计一种利用外部语言模型知识的上下文偏置方法，能够（1）根据对话上下文动态调整偏置强度；（2）利用语言模型的世界知识辅助实体识别；（3）无需重新训练 ASR 模型即可部署。

**切入角度**：作者观察到大型语言模型拥有丰富的世界知识和上下文建模能力，可以作为"知识库"来辅助 ASR 系统判断当前上下文下哪些偏置词更可能出现。通过在解码阶段引入知识增强的语言模型来动态调控偏置。

**核心 idea**：用外部知识增强语言模型（KELM）在 ASR 解码阶段提供上下文感知的偏置分数，通过 shallow fusion 与 ASR 模型的输出概率动态融合，实现语义驱动的上下文偏置。

## 方法详解

### 整体框架
系统由三个组件组成：（1）E2E ASR 模型，负责声学建模和基础解码；（2）知识增强外部语言模型（KELM），接收偏置词表和对话历史，为候选 token 提供上下文感知的语言模型分数；（3）融合解码器，在 beam search 过程中动态融合 ASR 分数和 KELM 分数。

### 关键设计

1. **知识增强语言模型（KELM）**:

    - 功能：为偏置词提供上下文感知的概率分数
    - 核心思路：在预训练语言模型（如 GPT-2 或 LLaMA）基础上，通过 prompt 工程将偏置词表注入上下文。具体地，将偏置词列表作为"提示前缀"（如"The following entities may appear: [word1, word2, ...]"），再拼接对话历史，让语言模型在给定这些先验知识的条件下预测下一个 token。这样语言模型的输出分布自然地偏向上下文中合理出现的偏置词。
    - 设计动机：相比硬编码的偏置提升，语言模型能根据上下文语义"理解"哪些偏置词在当前位置更合理，从而实现智能偏置。

2. **动态融合策略**:

    - 功能：在解码过程中自适应地平衡 ASR 模型和 KELM 的贡献
    - 核心思路：最终的 token 分数为 $\log p = \log p_{ASR} + \alpha \cdot \log p_{KELM} + \beta \cdot \mathbb{1}_{bias}$，其中 $\alpha$ 是语言模型权重，$\beta$ 是偏置词额外加分，$\mathbb{1}_{bias}$ 指示当前 token 是否属于偏置词的子词。关键创新在于 $\alpha$ 不是固定值，而是根据 ASR 模型的解码不确定性动态调整——当 ASR 置信度低时增大 $\alpha$，反之减小。
    - 设计动机：在 ASR 已经有高置信度的区域，不需要外部 LM 干预；只在 ASR 不确定时才需要借助外部知识，这避免了 LM 对正确识别的干扰。

3. **偏置词表的层级编码**:

    - 功能：高效处理大规模偏置词表
    - 核心思路：将偏置词按类别（人名、地名、术语等）分组，每组用一个摘要向量表示，在注意力机制中先选择相关类别再聚焦具体词汇。使用 trie 结构在子词级别追踪偏置词的匹配状态，确保只对正在匹配的词施加偏置。
    - 设计动机：实际应用中偏置词表可能包含数千个条目，逐一比较计算开销太大。层级结构将复杂度从 $O(n)$ 降至 $O(\log n)$。

### 损失函数 / 训练策略
KELM 的适配训练使用少量领域内数据进行轻量级微调，目标是标准语言模型的因果语言建模损失（next token prediction）。ASR 模型本身不需要重新训练，所有偏置能力通过解码阶段的融合实现。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文(KELM) | 浅层融合 | 深度偏置 | 无偏置基线 |
|--------|------|-----------|---------|---------|-----------|
| LibriSpeech (bias subset) | WER↓ | 4.2 | 5.8 | 5.1 | 7.6 |
| SPGISpeech | WER↓ | 8.3 | 10.1 | 9.4 | 12.8 |
| 内部客服数据 | Entity Recall↑ | 89.4% | 78.2% | 82.5% | 61.3% |
| 内部客服数据 | Entity Precision↑ | 91.7% | 83.6% | 87.1% | 72.5% |

### 消融实验

| 配置 | WER↓ | Entity Recall↑ | 说明 |
|------|------|----------------|------|
| Full KELM | 4.2 | 89.4% | 完整模型 |
| w/o 动态α | 4.8 | 86.1% | 固定融合权重，掉0.6 WER |
| w/o 偏置词prompt | 5.3 | 81.7% | 不注入偏置词表到LM prompt |
| w/o 对话历史 | 4.6 | 84.9% | 不使用对话上下文 |
| 小LM (GPT-2 small) | 4.9 | 85.3% | 用小模型替代，性能下降 |
| 大LM (LLaMA-7B) | 4.0 | 90.1% | 更大模型带来微小提升 |

### 关键发现
- 动态融合权重是最关键的设计，固定权重会导致在 ASR 确定区域过度干预
- 偏置词表注入 prompt 对 Entity Recall 贡献巨大（+7.7%），验证了知识注入的有效性
- LM 规模对性能有影响但收益递减，GPT-2 medium 已经能获得大部分收益

## 亮点与洞察
- 利用 LLM 的世界知识做上下文偏置是一个优雅的方案——将"何时偏置"的决策交给语言模型而非手工规则，大大提高了偏置的精准性
- 动态融合权重的设计非常实用——根据 ASR 不确定性调节外部干预强度，既提升了效果又控制了误识别
- 该方法的即插即用特性很有工程价值——无需重训 ASR 模型，只需在解码端增加 KELM 模块

## 局限与展望
- LLM 推理带来额外延迟，在实时语音识别场景中可能成为瓶颈
- KELM 的效果受限于 LM 的知识覆盖，对于极其罕见的专有名词仍可能无效
- 目前只验证了英语场景，在多语言和代码切换场景中的效果未知
- 未来可以探索流式 KELM，实现真正的实时上下文偏置

## 相关工作与启发
- **vs CLAS (Pundak et al.)**: CLAS 使用注意力机制将偏置词融入编码器，需要重训；本文的 KELM 在解码端即插即用，部署更灵活
- **vs TCPGen**: TCPGen 也利用外部信息做偏置，但采用 trie-based 硬匹配；本文通过 LM 实现语义级偏置，更智能
- **vs LLM rescoring**: 传统 LLM rescoring 在 N-best 列表上重排；本文在 beam search 过程中实时融合，信息利用更充分
- **vs Whisper + postprocessing**: Whisper等大模型虽然基础识别能力强，但对领域特有词汇的偏置仍需外部机制辅助

## 评分
- 新颖性: ⭐⭐⭐⭐ 将大语言模型的知识用于ASR上下文偏置是有新意的方向
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证，消融完整
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述详细
- 价值: ⭐⭐⭐⭐⭐ 对语音识别产品化有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation](omniflatten_an_end-to-end_gpt_model_for_seamless_voice_conversation.md)
- [\[ACL 2025\] DNCASR: End-to-End Training for Speaker-Attributed ASR](dncasr_end-to-end_training_for_speaker-attributed_asr.md)
- [\[ACL 2025\] Distilling an End-to-End Voice Assistant Without Instruction Training Data](distilling_an_end-to-end_voice_assistant_without_instruction_training_data.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)
- [\[ACL 2026\] VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models](../../ACL2026/audio_speech/vapo_end-to-end_slide-enhanced_speech_recognition_with_omni-modal_large_language.md)

</div>

<!-- RELATED:END -->
