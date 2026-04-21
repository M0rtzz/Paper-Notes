---
title: >-
  [论文解读] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations
description: >-
  [ICCV 2025][语音][零样本语音识别] 提出 Zero-AVSR 框架，通过将语音转写为语言无关的罗马化文本（Roman text），再利用 LLM 将罗马文本转换为目标语言文字，实现无需目标语言语音数据的零样本视听语音识别，并构建了覆盖 82 种语言、2916 小时的 MARC 数据集。
tags:
  - ICCV 2025
  - 语音
  - 零样本语音识别
  - 视听语音识别
  - 罗马化文本
  - 大语言模型
  - 多语言
---

# Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations

**会议**: ICCV 2025  
**arXiv**: [2503.06273](https://arxiv.org/abs/2503.06273)  
**代码**: 有（论文中提到 code and models are available online）  
**领域**: Audio & Speech  
**关键词**: 零样本语音识别, 视听语音识别, 罗马化文本, 大语言模型, 多语言

## 一句话总结

提出 Zero-AVSR 框架，通过将语音转写为语言无关的罗马化文本（Roman text），再利用 LLM 将罗马文本转换为目标语言文字，实现无需目标语言语音数据的零样本视听语音识别，并构建了覆盖 82 种语言、2916 小时的 MARC 数据集。

## 研究背景与动机

视听语音识别（AVSR）结合音频和唇部运动信息来增强语音理解，特别是在噪声环境下效果显著。然而，现有 AVSR 研究主要集中在英语上，多语言 AVSR 数据集仅覆盖 9 种语言。为每种语言获取足够的带标注视听数据非常困难，这严重限制了 AVSR 向更多语言的扩展。

核心问题：**如何在不使用目标语言语音数据的情况下进行语音识别？**

关键洞察：不同语言在音素层面共享发音特征，可以通过罗马化（romanization）将所有语言统一到一个语言无关的表示空间中。同时，预训练 LLM 已经具备将罗马文本转换为各语言文字的能力。

## 方法详解

### 整体框架

Zero-AVSR 由两个核心组件组成：(1) AV-Romanizer：将多语言视听语音输入预测为语言无关的罗马化文本；(2) LLM 解码器：将罗马化文本转换为目标语言的特定文字（graphemes）。框架分为 Cascaded 和 Unified 两种形式。

### 关键设计

1. **MARC 数据集**: 整合 LRS3、MuAViC、VoxCeleb2、AVSpeech 四个数据集，构建覆盖 82 种语言、2916 小时视听数据的多语言罗马化语料库。通过 GPT-4o-mini 进行罗马化标注（实验证明其在 romanization 和 de-romanization 重建测试中效果最优）。对无标签数据集使用预训练语言识别和 ASR 模型获取语言 ID 和转写。

2. **AV-Romanizer（视听语音罗马化器）**: 基于 AV-HuBERT 架构，包含音频编码器 $\mathcal{F}_a$（线性层）、视觉编码器 $\mathcal{F}_v$（ResNet-18 + 3D 卷积）、Transformer 编码器 $\mathcal{B}$（24 层）和线性分类头。将音频特征 $f_a$ 和视觉特征 $f_v$ 沿通道维度拼接后通过线性层降维，输入 Transformer 编码器得到融合特征 $f_{av} = \mathcal{B}((f_a \oplus f_v)W)$。使用 CTC 损失训练，预测罗马化文本。

3. **Cascaded Zero-AVSR**: 级联 AV-Romanizer 和预训练 LLM（如 GPT-4o-mini），不需要微调 LLM。AV-Romanizer 将视听语音转为罗马文本，然后通过指令引导 LLM 将罗马文本转为目标语言文字。优势是可以使用任意 LLM（包括 API 形式）。

4. **Zero-AVSR（统一模型）**: 将 AV-Romanizer 编码的视听特征直接嵌入 LLM（Llama3.2-3B），通过多任务训练实现端到端的零样本识别。

    - **Task 1（对齐）**: 使用长度压缩器（1D 卷积，kernel=2, stride=2）和 adapter 将视听特征映射到 LLM 嵌入空间，在已见语言上用语言建模目标训练。冻结 AV-Romanizer 和 LLM 原始权重，仅训练 LoRA 权重、压缩器和 adapter。
    - **Task 2（学习去罗马化）**: 纯文本任务，训练 LLM 将罗马文本转为目标语言文字，覆盖已见和未见语言，防止 LLM 遗忘多语言能力。仅训练 LoRA 权重。

### 损失函数 / 训练策略

- AV-Romanizer 使用 CTC 损失训练
- 三阶段学习率调度：10K warmup、40K hold、50K decay，峰值学习率 1e-4
- 训练中对音频随机添加 MUSAN 噪声（0 dB SNR）
- Zero-AVSR LLM 阶段使用 cosine 调度器（0.5K warmup + 29.5K decay），使用 QLoRA 微调
- 推理时使用 beam search（width=2, temperature=0.3）

## 实验关键数据

### 主实验

| 方法 | 模态 | 训练时长 | 支持语言数 | Ara | Deu | Ell | Spa | Fra | Ita | Por | Rus | Eng | Avg(含Eng) |
|------|------|---------|-----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----------|
| AV-HuBERT | AVSR | 1759h | 9 | 89.4 | 52.0 | 46.2 | 17.4 | 20.3 | 20.8 | 22.1 | 44.7 | 1.7 | 35.0 |
| XLAVS-R 2B | AVSR | 437Kh | 9 | 79.3 | 44.4 | 19.0 | 9.1 | 12.3 | 10.6 | 11.2 | 25.0 | 1.7 | 23.6 |
| MMS Zero-shot | ASR | 476Kh | 1078+ | 84.9 | 31.5 | 47.9 | 17.7 | 33.6 | 19.0 | 35.5 | 42.8 | 35.7 | 38.9 |
| Cascaded Zero-AVSR | AVSR | 2916h | 82+ | 82.1 | 29.3 | 47.2 | 16.3 | 28.9 | 21.6 | 20.2 | 42.9 | 2.9 | 30.2 |
| **Zero-AVSR** | AVSR | 2916h | 82+ | **81.4** | **27.8** | **38.4** | **13.1** | **14.3** | **15.9** | **15.4** | **32.6** | **1.5** | **25.2** |

### 消融实验

**MARC 数据集有效性（Cascaded Zero-AVSR，目标未见语言：Rus）**:

| 训练数据 | 训练语言数 | 训练时长 | 零样本CER↓ | 平均CER↓ |
|---------|-----------|---------|----------|---------|
| MuAViC | 8 | 745h | 62.3 | 48.3 |
| +MARC (8语言) | 8 | 1944h | 61.0 | 28.3 |
| +MARC (40语言) | 40 | 2418h | 49.5 | 25.1 |
| +MARC (81语言) | 81 | 2793h | **40.0** | **21.9** |

**不同 LLM 对 Cascaded Zero-AVSR 的影响**:

| LLM | 平均CER↓ |
|-----|---------|
| Llama3.2-3B | 35.7 |
| Mistral-7B | 29.6 |
| Llama3.1-8B | 27.3 |
| Llama3.1-70B | 21.3 |
| GPT-4o-mini | **19.5** |

### 关键发现

- Zero-AVSR 在仅使用 2916h 视听数据的情况下，平均 WER 达到 25.2%，与使用 436K 小时音频数据的 XLAVS-R 2B 模型（23.6%）相当
- 增加语言多样性（8→81 种语言）显著提升零样本性能，Rus 的零样本 CER 从 62.3% 降至 40.0%
- 同语族语言的数据对零样本性能帮助更大（验证了语言学先验）
- LLM 规模越大，Cascaded 方案的解码效果越好

## 亮点与洞察

- **罗马化作为语言无关表示的巧妙设计**：相比音素（phoneme），罗马文本更简单且 LLM 天然具备转换能力
- **级联方案的实用性**：不需要微调 LLM，可直接使用 API 形式的闭源模型
- **数据量和语言多样性的 scaling 效应**：更多语言和数据能显著提升零样本能力
- **LLM 作为通用去罗马化器**：消除了为每种语言训练独立语言模型的需求

## 局限与展望

- 零样本性能与监督方法仍有差距（特别是阿拉伯语等书写系统差异大的语言）
- Unified Zero-AVSR 使用较小的 Llama3.2-3B，更大模型可能进一步提升
- 罗马化本身不是完全可逆的转换，存在信息损失
- 82 种语言仍未覆盖全球所有语言，特别是低资源语言

## 相关工作与启发

- AV-HuBERT：自监督视听预训练的基础架构
- MMS Zero-shot：音频领域的零样本语音识别先驱
- XLAVS-R：多语言视听自监督学习
- 启发：罗马化+LLM 的范式可推广到其他多语言任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个零样本 AVSR 框架，罗马化+LLM 的设计思路新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ 12个消融实验、8种语言的全面评估、多个LLM对比
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，动机阐述充分
- **价值**: ⭐⭐⭐⭐ 82种语言的多语言视听识别，数据集和方法均有贡献

<!-- RELATED:START -->

## 相关论文

- [Zero-Shot Text-to-Speech for Vietnamese](../../ACL2025/audio_speech/zero-shot_text-to-speech_for_vietnamese.md)
- [ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control](../../ACL2025/audio_speech/controlspeech_zero_shot.md)
- [TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis](../../ACL2025/audio_speech/tcsinger_2_customizable_multilingual_zero-shot_singing_voice_synthesis.md)
- [Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment](../../ACL2025/audio_speech/advancing_zero-shot_text-to-speech_intelligibility_across_diverse_domains_via_pr.md)
- [MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition](../../NeurIPS2025/audio_speech/mome_mixture_of_matryoshka_experts_for_audio-visual_speech_recognition.md)

<!-- RELATED:END -->
