---
title: >-
  [论文解读] Soundwave: Less is More for Speech-Text Alignment in LLMs
description: >-
  [ACL 2025][音频/语音][语音大模型] 提出 Soundwave 模型，通过高效训练策略和新颖架构解决语音和文本之间的表示空间差距与序列长度不一致问题，仅用五十分之一的训练数据即超越 Qwen2-Audio 在语音翻译和 AIR-Bench 语音任务上的表现。
tags:
  - "ACL 2025"
  - "音频/语音"
  - "语音大模型"
  - "语音文本对齐"
  - "数据高效训练"
  - "序列长度不一致"
  - "表示空间鸿沟"
---

# Soundwave: Less is More for Speech-Text Alignment in LLMs

**会议**: ACL 2025  
**arXiv**: [2502.12900](https://arxiv.org/abs/2502.12900)  
**代码**: [GitHub](https://github.com/FreedomIntelligence/Soundwave)  
**领域**: NLP/语音  
**关键词**: 语音大模型, 语音文本对齐, 数据高效训练, 序列长度不一致, 表示空间鸿沟  

## 一句话总结

提出 Soundwave 模型，通过高效训练策略和新颖架构解决语音和文本之间的表示空间差距与序列长度不一致问题，仅用五十分之一的训练数据即超越 Qwen2-Audio 在语音翻译和 AIR-Bench 语音任务上的表现。

## 研究背景与动机

**领域现状**：端到端语音大语言模型（Speech LLM）近年快速发展，主流方法通常需要大规模标注语音-文本配对数据进行训练，来实现语音理解、翻译和对话等功能。代表性模型如 Qwen2-Audio 通常依赖数十万到百万小时级别的语音训练数据。

**现有痛点**：大规模数据驱动的训练策略带来了高昂的计算成本和数据收集成本。更关键的是，语音和文本之间存在两个根本性问题：(1) **表示空间差距**——语音编码器输出的特征和 LLM 的文本嵌入处于不同的语义空间中；(2) **序列长度不一致**——同样的语义内容，语音特征序列比对应的文本 token 序列长得多，这为对齐带来了巨大困难。

**核心矛盾**：现有方法试图通过堆积更多数据来弥合这两个鸿沟，但这本质上是以量取胜，没有从架构层面解决问题。数据高效训练在语音 LLM 领域尚未被深入探讨。

**本文目标**：设计一种数据高效的语音-文本对齐方案，用极少量数据（仅 10k 小时）达到甚至超越大规模训练模型的效果。

**切入角度**：作者观察到，如果能从架构层面直接解决表示空间差距和序列长度不一致问题，就可以大幅减少对训练数据的依赖。Less is More——精心设计的架构比堆数据更有效。

**核心 idea**：通过一个新颖的语音-文本桥接架构和高效训练策略，显式解决表示空间对齐和序列压缩问题，用 1/50 的数据实现 SOTA 性能。

## 方法详解

### 整体框架

Soundwave 的整体 pipeline 为：语音输入 → 语音编码器（如 Whisper encoder）提取声学特征 → 桥接模块进行空间对齐和长度压缩 → LLM backbone（基于 Qwen2）进行理解和生成 → 文本输出。核心创新在于中间的桥接模块设计和高效的训练策略。

### 关键设计

1. **语音-文本表示空间对齐模块**:

    - 功能：将语音编码器的输出特征映射到 LLM 的文本嵌入空间
    - 核心思路：使用一个专门设计的投影网络，将语音特征从声学空间转换到 LLM 能理解的语义空间。不同于简单的线性映射，该模块通过多层变换确保语音特征与文本 token 嵌入在同一流形上
    - 设计动机：此前的方法虽然也用投影层，但往往需要大量数据来"暴力"学习对齐关系。本文的设计使得对齐更加精确，从而减少数据需求

2. **序列长度压缩机制**:

    - 功能：将过长的语音特征序列压缩到与文本 token 序列相当的长度
    - 核心思路：语音信号的帧率很高（如 Whisper 编码器输出每秒 50 帧），而对应文本可能只有几个 token。该模块通过下采样和池化操作将语音序列长度缩短数倍，使其与文本序列长度接近，从而让 LLM 能更自然地处理语音输入
    - 设计动机：序列长度不匹配是导致注意力机制效率低下和对齐困难的核心原因。通过显式压缩，避免了 LLM 处理过长序列的计算负担

3. **数据高效训练策略**:

    - 功能：仅用约 10k 小时语音数据实现高质量的语音-文本对齐
    - 核心思路：采用分阶段训练策略——先冻结 LLM 和语音编码器，仅训练桥接模块完成初步对齐；再解冻部分 LLM 参数进行端到端微调。同时精心筛选高质量训练数据，确保数据多样性和覆盖度
    - 设计动机：对比 Qwen2-Audio 使用约 500k 小时数据，本文证明精心设计的架构可以将数据需求降低 50 倍

### 损失函数 / 训练策略

采用标准的自回归语言建模损失（next-token prediction），在语音理解和翻译任务上进行联合训练。分阶段训练确保桥接模块先收敛，避免大模型参数在早期被噪声梯度破坏。

## 实验关键数据

### 主实验

| 任务/数据集 | 指标 | Soundwave | Qwen2-Audio | 对比 |
|-------------|------|-----------|-------------|------|
| 语音翻译 (CoVoST2 en→de) | BLEU | 领先 | 基线 | Soundwave 超越 |
| 语音翻译 (CoVoST2 en→zh) | BLEU | 领先 | 基线 | Soundwave 超越 |
| AIR-Bench Speech | 综合分 | 更高 | 基线 | 使用 1/50 数据即超越 |
| 语音识别 (ASR) | WER | 有竞争力 | 基线 | 接近或持平 |
| 对话智能保持 | 主观评分 | 保持 | 基线 | 未显著下降 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full Soundwave | 最优 | 完整模型 |
| w/o 序列压缩 | 下降明显 | 长序列导致注意力效率和对齐质量下降 |
| w/o 表示空间对齐 | 下降显著 | 简单线性投影无法充分弥合空间差距 |
| w/o 分阶段训练 | 下降中等 | 直接端到端训练收敛困难 |
| 使用更多数据（50k h） | 略有提升 | 边际收益递减，说明架构设计已足够 |

### 关键发现

- 表示空间对齐模块和序列压缩机制是两个最关键的组件，缺少任一都会导致显著性能下降
- 训练数据量从 10k 增加到 50k 小时带来的提升很小，印证了 "Less is More" 的核心论点
- Soundwave 在对话场景中保持了 LLM 的通用智能，说明桥接模块没有破坏 LLM 的原有能力

## 亮点与洞察

- **数据效率惊人**：50 倍的数据差距下仍能超越强基线，证明架构设计在语音-文本对齐中比数据规模更重要。这个发现对资源受限场景下部署语音 LLM 有重大意义
- **解耦式设计**：将表示空间对齐和序列长度压缩作为两个独立问题分别解决，思路清晰且有效。这种"分而治之"的策略可以迁移到其他模态对齐任务（如视频-文本）
- **对话能力保持**：不像一些多模态微调方法会导致语言能力退化（catastrophic forgetting），Soundwave 通过分阶段训练有效保留了 LLM 的通用能力

## 局限与展望

- 论文主要在语音翻译和语音理解任务上评测，对语音情感分析、声纹识别等更细粒度任务的效果尚不清楚
- 目前仅基于 Qwen2 backbone，是否能推广到其他 LLM（如 LLaMA）需要验证
- 10k 小时数据虽然比 500k 少很多，但对于低资源语言仍可能是较大的门槛
- 桥接模块的具体架构细节（层数、维度等）的选择依据可进一步讨论
- 未来可探索将类似的高效对齐策略应用到视频理解等其他多模态场景

## 相关工作与启发

- **vs Qwen2-Audio**: Qwen2-Audio 依赖大规模数据训练，效果虽好但成本高。Soundwave 证明了通过架构优化可以极大降低数据需求
- **vs SALMONN**: SALMONN 也是语音 LLM，但其对齐方式较为简单（线性投影），Soundwave 的多层桥接设计更精细
- **vs Whisper**: Whisper 是纯语音模型，不具备对话能力。Soundwave 将 Whisper 级别的识别能力和 LLM 的语言理解能力统一在一个框架中

## 评分

- 新颖性: ⭐⭐⭐⭐ 数据高效的语音-文本对齐方向有价值，但具体技术手段（投影+压缩）不算全新
- 实验充分度: ⭐⭐⭐⭐ 多个任务上验证，消融实验完整，但缺少一些细粒度任务评测
- 写作质量: ⭐⭐⭐⭐ 论述清晰，"Less is More" 定位明确
- 价值: ⭐⭐⭐⭐ 对低资源语音大模型部署有直接参考意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment](advancing_zero-shot_text-to-speech_intelligibility_across_diverse_domains_via_pr.md)
- [\[AAAI 2026\] Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding](../../AAAI2026/audio_speech/say_more_with_less_variable-frame-rate_speech_tokenization_via_adaptive_clusteri.md)
- [\[AAAI 2026\] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR](../../AAAI2026/audio_speech/hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)
- [\[ACL 2025\] Zero-Shot Text-to-Speech for Vietnamese](zero-shot_text-to-speech_for_vietnamese.md)
- [\[ACL 2025\] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)

</div>

<!-- RELATED:END -->
