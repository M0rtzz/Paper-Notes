---
title: >-
  [论文解读] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models
description: >-
  [ACL 2025][语音][文本转语音] SpeechWeave 提出了一套端到端的合成语音数据生成流水线，通过关键词采样提升文本多样性、在生成时即完成文本归一化（准确率达97%）、并利用跨语言语音克隆实现说话人标准化，生成的数据在多样性上比直接提示LLM高出10-48%，并能有效提升下游TTS模型性能。
tags:
  - ACL 2025
  - 语音
  - 文本转语音
  - 合成数据
  - 多语言
  - 文本归一化
  - 语音克隆
---

# SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models

**会议**: ACL 2025  
**arXiv**: [2509.14270](https://arxiv.org/abs/2509.14270)  
**代码**: 无  
**领域**: Audio & Speech  
**关键词**: 文本转语音, 合成数据, 多语言, 文本归一化, 语音克隆

## 一句话总结

SpeechWeave 提出了一套端到端的合成语音数据生成流水线，通过关键词采样提升文本多样性、在生成时即完成文本归一化（准确率达97%）、并利用跨语言语音克隆实现说话人标准化，生成的数据在多样性上比直接提示LLM高出10-48%，并能有效提升下游TTS模型性能。

## 研究背景与动机

在文本转语音（TTS）领域，高质量的训练数据是模型成功的关键。然而，现有数据获取面临三大核心困境：

**领域特异性**：公开TTS数据集（如LibriSpeech）主要是书籍朗读或通用段落，缺乏特定商业领域（汽车、医疗、零售等）的数据。从网络爬取或购买数据则涉及成本和版权问题。

**文本多样性不足**：虽然LLM可以生成文本，但实验表明，即使调高temperature和top_p参数，当输入提示保持不变时，LLM生成的短句仍高度相似。极高的参数设置会导致输出不稳定、质量下降。这使得直接使用LLM生成大规模训练数据不可行。

**文本归一化挑战**：书面文本和口语形式往往不同（如日期、地址、缩写等"半语义类"），现有归一化工具（如NeMo）在处理多种变体时容易出错或遗漏。

**语音录制不可扩展**：商业TTS系统需要特定说话人的标准化语音，依赖人工语音艺术家录音成本高且不可扩展。

基于以上痛点，作者提出 SpeechWeave——一套集文本生成、归一化和语音合成于一体的自动化合成数据流水线，核心思路是：**在数据生成阶段就解决多样性和归一化问题，而非事后修补**。

## 方法详解

### 整体框架

SpeechWeave 包含四个核心模块：关键词采样器（Keyphrase Sampler）、实体采样器与源头归一化器（Entity Sampler with At-Source Normalizer）、后处理器（Postprocessor）和语音生成模块（Audio Generation Module）。整体流程为：先生成多样化的关键词 → 生成带有语义类实体的文本 → 同步完成归一化 → 后处理 → 生成标准化语音。

### 关键设计

1. **关键词采样与多步提示（Keyphrase Sampling）**:

    - **核心问题**：LLM在固定提示下生成重复文本。
    - **解决思路**：利用关键词注入（keyphrase infusion）增加提示多样性。例如，将"生成金融领域的句子"改为"生成包含'抵押贷款、资产金融'关键词的金融领域句子"。
    - **多步生成**：先让LLM生成子领域列表 → 随机选一个 → 生成创意段落 → 提取关键词。这种迭代式多步提示大幅提升了idea多样性。
    - **去重机制**：使用内存关键词存储，基于模糊搜索（token sort ratio + Levenshtein距离）进行去重，实验发现比PhraseBERT嵌入相似度方法产生更多样的关键词。

2. **实体采样与源头归一化（Entity Sampler & At-Source Normalization）**:

    - **核心创新**：不是先生成文本再归一化，而是在生成实体时就同时产生其归一化形式。
    - **设计动机**：传统归一化器面对同一实体的多种格式（如日期03/01/2005、01-Mar-2005、March 01, 2005）可能遗漏某些变体。而生成时归一化是确定性的，因为实体组件在拼接过程中即完成归一化。
    - **覆盖范围**：支持9种实体类型（地址、电话、邮箱、URL、日期、时间、百分比、人名+称谓），可生成数千种唯一组合，且支持多语言和区域设置敏感。

3. **语音生成与跨语言语音克隆（Speech Audio Generation & Cross-Lingual Voice Cloning）**:

    - 将归一化文本输入预训练TTS模型（Zhao et al. 2023）生成基础语音。
    - 使用 OpenVoiceV2 的音色转换器，以参考语音艺术家的音频进行说话人标准化。
    - **关键优势**：音色转换器与语言无关，可以用英语参考音频标准化其他语言的语音，实现跨语言的说话人一致性。

### 损失函数 / 训练策略

本文主要是数据生成流水线而非训练模型，没有特定的损失函数设计。文本生成使用 Mistral-7b-Instruct，通过 lm-format-enforcer 强制JSON格式输出。后处理阶段有一个基础算法来扩展缩写、转换数字等。

## 实验关键数据

### 主实验

实验生成了3000个数据点，覆盖16个商业领域、5种句型、9种语义类、2个参考说话人（男/女），分别在英语和西班牙语下进行。

| 指标 | 基线（直接提示） | SpeechWeave | LibriSpeech | 说明 |
|------|-----------------|-------------|-------------|------|
| 均值相似度（分组，英语）| 0.48 | 0.26 | - | 降低45.8% |
| 均值相似度（分组，西班牙语）| 0.54 | 0.30 | - | 降低44.4% |
| TTR（英语）| 0.118 | 0.167 | 0.123 | 词汇丰富度最高 |
| MATTR（英语）| 0.761 | 0.803 | 0.758 | 移动平均TTR最高 |
| 双音素覆盖（英语）| 1442 | 1694 | 1792 | 比基线高17.4% |
| 归一化准确率（英语）| 0.67（NeMo）| 0.97 | - | 提升显著 |
| 归一化准确率（西班牙语）| 0.54（NeMo）| 0.94 | - | 提升显著 |

### 消融实验 / 下游模型训练

| 配置 | 英语WER(%) | 西班牙语WER(%) | 说明 |
|------|-----------|---------------|------|
| LibriTTS Checkpoint（基线）| 15.37 | 85.05 | 微调前 |
| 基线+SpeechWeave数据微调 | 9.36 | 48.44 | WER分别降低40%和27% |

语音质量：英语 SNR=59.82dB, MOS=4.95, WER=9.32%；西班牙语 SNR=53.01dB, MOS=4.87, WER=15.21%。

### 关键发现

- 直接提示LLM即使使用高temperature，文本多样性仍然很低，关键词注入是有效的多样性增强手段
- 源头归一化（at-source normalization）准确率远超后处理归一化器NeMo（英语97% vs 67%，西班牙语94% vs 54%）
- 跨语言语音克隆使得单一参考说话人可在多语言间复用
- 合成数据有效提升下游TTS模型在真实测试集上的表现

## 亮点与洞察

- **"生成即归一化"的设计理念**：将归一化从后处理前移到数据生成阶段，从根本上避免了归一化器对多种实体格式的处理盲区，是一个简单但巧妙的工程创新
- **关键词采样的去重策略**：使用模糊搜索而非语义嵌入进行去重效果更好，说明多样性控制需要考虑字面层面的差异
- **跨语言语音克隆的实用价值**：用英语参考音频标准化其他语言语音，大幅降低了多语言TTS数据的制作门槛

## 局限与展望

- 实体采样器支持的语义类数量有限（仅9种），扩展新类型需要手动编写规则
- 仅在英语和西班牙语上评估，形态复杂的语言（如阿拉伯语、日语）效果未知
- 后处理器在处理不支持的实体类型时仍可能引入归一化错误
- 缺乏主观人工评测（如大规模MOS测试），语音自然度还需进一步验证
- 不支持带风格/情感的语音生成

## 相关工作与启发

- 与 Cornell et al. (2024) 的LLM+TTS用于ASR数据的流水线类似，但本文还集成了文本归一化和说话人标准化
- Gunduz et al. (2024) 的开源TTS数据生成工具缺少文本脚本生成和归一化功能
- 关键词注入技术来自 Eldan and Li (2023)，但本文将其系统化为多步采样+去重的完整方案
- 对其他需要大规模领域特定数据的场景（如ASR、NLU数据增强）有借鉴意义

## 评分

- 新颖性: ⭐⭐⭐ 各模块技术不新，但组合方式和源头归一化思路有价值
- 实验充分度: ⭐⭐⭐⭐ 多维度评估（多样性、归一化、语音质量、下游任务），但缺乏更多语言验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，表格和示例丰富，易于理解和复现
- 价值: ⭐⭐⭐⭐ 解决了TTS数据生成的实际痛点，对工业界有较高实用价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] ATRI: Mitigating Multilingual Audio Text Retrieval Inconsistencies by Reducing Data Distribution Errors](atri_mitigating_multilingual_audio_text_retrieval_inconsistencies_by_reducing_da.md)
- [\[ACL 2025\] Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment](advancing_zero-shot_text-to-speech_intelligibility_across_diverse_domains_via_pr.md)
- [\[ACL 2025\] Zero-Shot Text-to-Speech for Vietnamese](zero-shot_text-to-speech_for_vietnamese.md)
- [\[ACL 2025\] T2A-Feedback: Improving Basic Capabilities of Text-to-Audio Generation via Fine-grained AI Feedback](t2a_feedback_audio_gen.md)
- [\[ACL 2025\] In-the-wild Audio Spatialization with Flexible Text-guided Localization](tas_audio_spatialization.md)

<!-- RELATED:END -->
