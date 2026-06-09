---
title: >-
  [论文解读] VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction
description: >-
  [NeurIPS 2025][音频/语音][多模态] VITA-1.5 提出了一套精心设计的三阶段渐进式训练策略，将视觉和语音能力逐步整合进 LLM 中，实现了无需独立 ASR/TTS 模块的端到端视觉-语音实时交互，在图像、视频和语音基准上均达到开源模型领先水平。
tags:
  - "NeurIPS 2025"
  - "音频/语音"
  - "多模态"
  - "vision-speech interaction"
  - "end-to-end speech"
  - "three-stage training"
  - "omni model"
---

# VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction

**会议**: NeurIPS 2025  
**arXiv**: [2501.01957](https://arxiv.org/abs/2501.01957)  
**代码**: [https://github.com/VITA-MLLM/VITA](https://github.com/VITA-MLLM/VITA)  
**领域**: 多模态LLM / 语音交互  
**关键词**: multimodal LLM, vision-speech interaction, end-to-end speech, three-stage training, omni model

## 一句话总结

VITA-1.5 提出了一套精心设计的三阶段渐进式训练策略，将视觉和语音能力逐步整合进 LLM 中，实现了无需独立 ASR/TTS 模块的端到端视觉-语音实时交互，在图像、视频和语音基准上均达到开源模型领先水平。

## 研究背景与动机

多模态大语言模型（MLLM）在视觉-文本整合方面取得了显著进展（如 LLaVA、InternVL、Qwen-VL 等），但语音模态的整合却相对滞后。在实际人机交互系统中，语音既是信息传输的关键媒介，也极大提升了交互的自然性和便捷性。

**核心矛盾**：视觉和语音的模态差异根本性地不同——视觉数据承载空间信息，语音数据承载时序信息。同时优化两种模态经常导致训练冲突：加入语音数据可能降低视觉任务性能，反之亦然。

传统的语音-对-语音系统依赖独立的 ASR + LLM + TTS 三模块级联，存在延迟高、丢失韵律/情感等副语言信息、实时性差等问题。GPT-4o 展示了端到端多模态交互的可能性，但开源社区在同时具备视觉和语音能力的模型方面仍有很大差距。VITA-1.0 做了初步尝试，但引入语音数据对视觉性能造成了干扰，且语音生成仍依赖外部 TTS。

**本文切入角度**：通过精心设计的三阶段训练策略，**渐进式**引入不同模态，让模型在增强新模态能力的同时保持已有模态的性能。最终实现端到端的语音输出，消除对外部 TTS 的依赖。

## 方法详解

### 整体框架

VITA-1.5 的模型架构包含：

**输入端**：
- 视觉编码器：InternViT-300M（448×448 输入，256 视觉 token/图像）+ 动态分块（高分辨率图像）
- 音频编码器：350M 参数（4× 下采样卷积 + 24 层 Transformer，隐藏维度 1024），输出帧率 12.5Hz
- 视觉适配器：两层 MLP
- 语音适配器：多层卷积（2× 下采样）

**输出端**：
- 文本输出：LLM（Qwen2-7B）直接输出
- 语音输出：端到端语音生成模块
    - TiCodec 编解码器：单码本（大小 1024），40Hz 将语音编码为离散 token，解码回 24kHz 波形
    - NAR（非自回归）语音解码器：4 层 LLaMA 解码器，处理文本 token 全局语义
    - AR（自回归）语音解码器：4 层 LLaMA 解码器，逐步生成高质量语音 token
    - 两个解码器各约 120M 参数（隐藏维度 896）

### 关键设计

1. **视频处理策略**：

    - <4秒视频：均匀采样 4 帧
    - 4-16秒视频：每秒 1 帧
    - >16秒视频：均匀采样 16 帧
    - 视频帧不使用动态分块以避免 token 过多

2. **端到端语音生成**：

    - 传统方法：LLM → 文本 → TTS → 语音（级联，延迟高）
    - VITA-1.5：LLM 文本 token → NAR 解码器（全局语义特征）→ AR 解码器（逐步生成语音 token）→ Codec 解码器（语音波形）
    - LLM 冻结，不影响多模态理解性能

3. **输入分类头**：在 Stage 2.2 中为 LLM 输出添加分类头，区分输入来自语音还是文本，使模型能灵活处理不同模态

### 三阶段训练策略

核心思路：**渐进式引入不同模态，避免模态冲突**。

**Stage 1：视觉-语言训练**

- Stage 1.1 视觉对齐：仅训练视觉适配器，20% caption 数据，冻结其余模块
- Stage 1.2 视觉理解：训练视觉编码器+适配器+LLM，100% caption 数据，学习图像描述
- Stage 1.3 视觉 SFT：训练视觉模块+LLM，100% QA 数据 + 20% caption 数据，获得指令跟随和视觉 QA 能力

**Stage 2：音频输入调优**

- Stage 2.1 音频对齐：
    - (a) CTC 损失训练语音编码器（ASR 任务）
    - (b) 训练语音适配器+LLM，11000 小时语音-转写对，使 LLM 理解音频输入
- Stage 2.2 音频 SFT：4% caption + 20% QA 数据，约半数文本问题替换为 TTS 生成的语音版本，全模块可训练

**Stage 3：音频输出调优**

- Stage 3.1 Codec 训练：用 3000 小时语音数据训练单码本 codec 模型
- Stage 3.2 NAR+AR 解码器训练：文本-语音配对数据，LLM 冻结，仅训练语音解码器

### 训练数据

- 多模态指令调优：共 22133.16K 条 QA（中英双语），涵盖：
    - 通用图像描述/QA：ShareGPT4V、LLaVA系列、LVIS-Instruct 等
    - OCR & 图表：Anyword-3M、UReader、SynDOG 等
    - 视频：ShareGemini + 合成数据
    - 纯文本 QA：合成数据
- ASR 数据：110,000 小时内部语音-转写配对（中英）
- 语音生成数据：3000 小时 TTS 生成的文本-语音对

## 实验关键数据

### 图像理解基准

| 方法 | LLM | MMB | MMS | MMMU | MathV | OCR | Avg |
|------|-----|-----|-----|------|-------|-----|-----|
| GPT-4o | - | 82.8 | 61.6 | 62.8 | 56.5 | 663 | 69.3 |
| InternVL2 | InternLM2.5-7B | 79.4 | 61.5 | 51.2 | 58.3 | 794 | 67.3 |
| MiniCPM-V 2.6 | Qwen2-7B | 78.0 | 57.5 | 49.8 | 60.6 | 852 | 68.5 |
| VITA-1.0 | Mixtral-8x7B | 71.8 | 46.4 | 47.3 | 44.9 | 678 | 57.8 |
| VITA-1.5 (Stage 1) | Qwen2-7B | 77.1 | 59.1 | 53.1 | 66.2 | 752 | 67.1 |
| VITA-1.5 (Stage 3) | Qwen2-7B | **76.7** | **59.9** | **52.1** | **66.2** | **732** | **66.8** |

### 语音识别（ASR）

| 模型 | aishell-1 (CER↓) | test_net (CER↓) | dev_clean (WER↓) | test_clean (WER↓) |
|------|-------------------|-----------------|-------------------|-------------------|
| Wav2vec2-base | - | - | 6.0 | - |
| Mini-Omni2 | - | - | 4.8 | 4.7 |
| Freeze-Omni | 2.8 | 12.6 | 4.2 | 4.1 |
| VITA-1.0 | - | 12.2 | 7.6 | 8.1 |
| VITA-1.5 | **2.2** | **8.4** | **3.3** | **3.4** |

### 消融：三阶段训练的效果

| 阶段 | MMBench | MMStar | MMMU | MathVista | Video-MME | 说明 |
|------|---------|--------|------|-----------|-----------|------|
| Stage 1（仅视觉） | 77.1 | 59.1 | 53.1 | 66.2 | 56.8 | 视觉基线 |
| Stage 3（完整） | 76.7 | 59.9 | 52.1 | 66.2 | 56.1 | 加入语音后视觉几乎无损 |

### 关键发现

- **模态冲突基本被消除**：Stage 3 相比 Stage 1，图像理解平均分仅下降 0.3%（67.1→66.8），视频理解仅下降 0.7%，证明三阶段训练策略有效缓解了模态冲突
- VITA-1.5 的 ASR 性能全面超越专用语音模型（Freeze-Omni、Mini-Omni2），中文 CER 低至 2.2%
- 相比 VITA-1.0，图像理解提升 9%（57.8→66.8），ASR 错误率大幅降低（英文 WER 从 8.1 降至 3.4）
- 在图像理解上已接近 GPT-4o（66.8 vs 69.3），超越 GPT-4V（58.5）和 GPT-4o-mini（66.3）
- 端到端语音输出避免了 TTS 模块的额外延迟，显著提升实时交互体验

## 亮点与洞察

- **三阶段渐进式训练**：是本文最核心的贡献，通过精心安排数据和可训练模块的顺序，有效解决了视觉-语音模态冲突问题。这一策略具有很强的方法论价值。
- **端到端语音生成**：NAR + AR 双解码器设计在保持 LLM 冻结的前提下实现语音输出，不影响已有的多模态理解能力。
- **数据工程精细**：各阶段的数据配比（如 Stage 2.2 中语音问题替换率约 50%、Stage 1.3 中保留 20% caption 数据增加多样性）经过仔细调优。
- **产品化潜力**：VITA-1.5 展示了开源 omni 模型可以在视觉、语音两个维度同时达到 SOTA 级别的可行性。

## 局限与展望

- 视频理解与 GPT-4o（71.9）仍有约 16 分的差距（56.1），说明视频模态还有很大提升空间
- 语音输出的质量和自然度未做详细评估（如 MOS 评分），实际体验有待验证
- 语音端的训练数据依赖内部 ASR 数据（110K 小时）和 TTS 生成数据，可能不易复现
- 缺乏双工对话（duplex dialogue）能力的详细评估
- 单码本 codec 的语音质量可能不如多码本方案（如 EnCodec），但简化了解码

## 相关工作与启发

- **vs VITA-1.0**：1.5 版本的关键改进是端到端语音输出（替代外部 TTS）和更精细的三阶段训练策略
- **vs Mini-Omni2/LLaMA-Omni/Moshi**：这些模型实现了双工语音交互，但缺乏视觉理解能力
- **vs GPT-4o**：VITA-1.5 是目前最接近 GPT-4o 的开源 omni 模型，但在视频理解和语音自然度上仍有差距
- **冻结 LLM 训练语音解码器**的策略来自 Freeze-Omni，VITA-1.5 证明这一策略在保护多模态性能方面非常有效
- 启发：多模态模型的训练策略设计可能比架构设计更重要，渐进式引入模态是处理模态冲突的有效范式

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] RTCFake: Speech Deepfake Detection in Real-Time Communication](../../ACL2026/audio_speech/rtcfake_speech_deepfake_detection_in_real-time_communication.md)
- [\[NeurIPS 2025\] E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models](e-bats_efficient_backpropagation-free_test-time_adaptation_for_speech_foundation.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](../../AAAI2026/audio_speech/hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)
- [\[NeurIPS 2025\] Instance-Specific Test-Time Training for Speech Editing in the Wild](instance-specific_test-time_training_for_speech_editing_in_the_wild.md)
- [\[NeurIPS 2025\] AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time](textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)

</div>

<!-- RELATED:END -->
