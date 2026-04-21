---
title: >-
  [论文解读] Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens
description: >-
  [ACL 2025][语音][text-to-speech] 提出 Spark-TTS，基于新型单流语音编解码器 BiCodec 和 Qwen2.5 LLM 的高效 TTS 系统，通过将语音解耦为低码率语义 token 和固定长度全局 token，实现零样本语音克隆和从粗到细的属性控制，在 Seed-TTS-eval 上达到 SOTA 可懂度。
tags:
  - ACL 2025
  - 语音
  - text-to-speech
  - speech codec
  - LLM-based TTS
  - voice cloning
  - controllable speech synthesis
---

# Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens

**会议**: ACL 2025  
**arXiv**: [2503.01710](https://arxiv.org/abs/2503.01710)  
**代码**: [GitHub](https://github.com/SparkAudio/Spark-TTS)  
**领域**: 语音  
**关键词**: text-to-speech, speech codec, LLM-based TTS, voice cloning, controllable speech synthesis

## 一句话总结

提出 Spark-TTS，基于新型单流语音编解码器 BiCodec 和 Qwen2.5 LLM 的高效 TTS 系统，通过将语音解耦为低码率语义 token 和固定长度全局 token，实现零样本语音克隆和从粗到细的属性控制，在 Seed-TTS-eval 上达到 SOTA 可懂度。

## 研究背景与动机

**领域现状**: 基于 LLM 的 codec TTS 已成为零样本 TTS 的主流范式，通过大规模训练数据和大模型架构，合成语音自然度接近真人。代表方法包括 VALL-E、CosyVoice、Seed-TTS 等。

**现有痛点**: 
   - 现有 codec TTS 架构复杂，需要双生成模型（如语义→声学两阶段）或并行多流 codebook 预测机制（如 group VQ），偏离标准文本 LLM 框架
   - 语义 token 虽然紧凑但缺乏音色控制能力，需要额外的声学特征预测模块
   - 声学 token 依赖复杂的 codebook 架构
   - 现有系统主要限于参考音频驱动的生成，无法精确指定声音特征（如精确的音高值）来创建新声音
   - 大量研究使用私有数据，难以公平对比

**核心矛盾**: 如何在保持架构与文本 LLM 统一（单流自回归）的同时，既实现高质量语音重建，又支持灵活的声音属性控制？

**本文目标**: 设计统一架构实现零样本 TTS + 属性可控语音生成，且完全对齐标准文本 LLM 范式。

**切入角度**: 设计 BiCodec 将语音解耦为语义 token（时变语言内容）和全局 token（时不变说话人属性），配合 Chain-of-Thought 生成策略实现从粗到细的属性控制。

**核心 idea**: 用 BiCodec 将语音分解为语义 token + 全局 token 的单流结构，让标准文本 LLM 直接建模语音生成。

## 方法详解

### 整体框架

系统由三部分组成：
1. **BiCodec**: 单流语音编解码器，将语音编码为语义 token（50 TPS）+ 全局 token（固定 32 个）
2. **Speech LLM**: 基于 Qwen2.5-0.5B 的解码器-only Transformer，统一文本和语音 token 预测
3. **VoxBox 数据集**: 精心清洗标注的 100K 小时语音数据

### 关键设计

1. **BiCodec — 双重 token 化架构**:

    - **Semantic Tokenizer**: 输入 wav2vec 2.0 (XLSR-53) 的第 11/14/16 层特征平均值，经卷积编码器 $E_s$ + VQ 量化 $Q_s$，生成 50 TPS 的语义 token $\bm{z}_q$
    - **Global Tokenizer**: 输入 Mel 频谱，经 ECAPA-TDNN 编码器 $E_g$ + 可学习 query 的 Cross Attention + FSQ 量化 $Q_g$，生成固定长度（32 个）全局 token $\bm{g}_q$
    - **Decoder**: 卷积解码器 $G$，从量化后的语义 token 和聚合后的全局 embedding 重建波形

$$\bm{z} = E_s(F(\bm{x})), \quad \bm{g} = E_g(\text{Mel}(\bm{x}))$$
$$\bm{g}_f = \text{CrossAttention}(\bm{g}, \bm{h}), \quad \hat{\bm{x}} = G(\bm{z}_q, A_g(\bm{g}_q))$$

2. **语音语言模型**: 基于 Qwen2.5-0.5B，支持两种生成模式：

    - **零样本 TTS**: 输入文本 prompt $\mathcal{T}$ + 参考音频的全局 token $\mathcal{G}$，预测语义 token $\bm{o}$
    - **属性控制生成（CoT 方式）**: 输入文本 + 粗粒度属性标签（性别/音高等级/语速等级）$\mathcal{A}$，模型以 Chain-of-Thought 方式依次预测：细粒度属性值 $\mathcal{F}$ → 全局 token $\mathcal{G}$ → 语义 token $\mathcal{S}$

3. **VoxBox 数据集**:

    - 102.5K 小时语音，来自 29 个开源数据集，470 万音频文件
    - 标注：性别（WavLM fine-tune 分类，99.4% 准确率）、音高（PyWorld 提取 + Mel 尺度百分位分级）、语速（音节/秒 SPS + 百分位分级）
    - 数据清洗：Whisper 原始转录用 FunASR 重新识别，WER > 0.05 的样本剔除

### 损失函数/训练策略

- **BiCodec 训练**: GAN 训练（多周期判别器 + 多频段 STFT 判别器），L1 Mel 重建损失 + L1 特征匹配损失 + VQ codebook loss + commitment loss + wav2vec 2.0 重建损失。训练约 800K steps，batch 614.4s 语音
- **LLM 训练**: 负对数似然损失，零样本和属性控制两种目标混合训练。每条音频构造两个训练样本。AdamW 优化器，3 epochs，batch 768

$$\mathcal{L}_{zst} = -\sum_{t=1}^{T_o} \log P(o_t | \mathcal{T}, \mathcal{G}, \bm{o}_{<t}; \theta_{LM})$$
$$\mathcal{L}_{control} = -\sum_{t=1}^{T_c} \log P(c_t | \mathcal{T}, \mathcal{A}, \bm{c}_{<t}; \theta_{LM})$$

## 实验关键数据

### BiCodec 重建性能（LibriSpeech test-clean）

| 模型 | 码率(bps) | STOI↑ | PESQ-WB↑ | UTMOS↑ | SIM↑ |
|------|-----------|-------|----------|--------|------|
| Encodec (8 codebook) | 6000 | 0.94 | 2.75 | 3.07 | 0.89 |
| DAC (12 codebook) | 6000 | 0.95 | 4.01 | 4.00 | 0.98 |
| X-codec2 | 800 | 0.92 | 2.43 | 4.13 | 0.82 |
| StableCodec | 697 | 0.91 | 2.24 | 4.23 | 0.62 |
| **BiCodec** | **650** | **0.92** | **2.51** | **4.18** | **0.80** |

BiCodec 在 <1kbps 低码率范围内几乎所有指标 SOTA。

### 全局 token 长度消融

| 全局 Token | STOI | PESQ-WB | UTMOS | SIM |
|-----------|------|---------|-------|-----|
| w/o FSQ | 0.915 | 2.52 | 4.15 | 0.83 |
| GVQ-32 | 0.912 | 2.30 | 4.06 | 0.74 |
| FSQ-8 | 0.916 | 2.41 | 4.16 | 0.74 |
| FSQ-16 | 0.919 | 2.45 | 4.15 | 0.77 |
| **FSQ-32** | **0.922** | **2.51** | **4.18** | **0.80** |

### 零样本 TTS（Seed-TTS-eval）

| 模型 | test-zh CER↓ | test-zh SIM↑ | test-en WER↓ | test-en SIM↑ |
|------|-------------|-------------|-------------|-------------|
| Seed-TTS (闭源) | **1.12** | **0.796** | 2.25 | **0.762** |
| CosyVoice2 | 1.45 | 0.748 | 2.57 | 0.652 |
| F5-TTS | 1.56 | 0.741 | **1.83** | 0.647 |
| Llasa-8B-250k | 1.59 | 0.684 | 2.97 | 0.574 |
| **Spark-TTS** | 1.20 | 0.672 | 1.98 | 0.584 |

### 性别控制准确率

| 方法 | 准确率 |
|------|--------|
| VoxInstruct | 82.99% |
| Parler-TTS | 98.12% |
| **Spark-TTS** | **99.77%** |

### 语音质量（LibriSpeech test-clean UTMOS）

| 方法 | GT | CosyVoice | CosyVoice2 | Spark-TTS |
|------|-----|-----------|-----------|-----------|
| UTMOS↑ | 4.08 | 4.09 | 4.23 | **4.35** |

### 关键发现

- BiCodec 在 650 bps（仅 50 TPS 语义 + 32 全局 token）下实现低码率 SOTA 重建质量
- Spark-TTS 仅 0.5B 参数 + 100K 小时数据，在可懂度上超过 Llasa-8B（250K 小时数据）
- 中文 CER 1.20% 仅次于闭源 Seed-TTS（1.12%），英文 WER 1.98% 仅次于 F5-TTS（1.83%）
- 说话人相似度（SIM）是弱项：0.672/0.584，低于多阶段或 NAR 方法，这是单阶段 AR 方法的固有局限
- 音高和语速控制的混淆矩阵显示五级分类控制准确率高（对角线集中）
- 生成音质 UTMOS 4.35 超过真实音频的 4.08，说明模型有"美化"效果
- FSQ + 可学习 query 方案显著优于 GVQ 方案（SIM: 0.80 vs 0.74）

## 亮点与洞察

- **架构极简优雅**: BiCodec 将语音分解为语义（时变）和全局（时不变）两类 token，整个 TTS pipeline 完全对齐标准文本 LLM，不需要 flow matching 或扩散模型
- **CoT 生成策略巧妙**: 粗粒度标签→细粒度数值→全局 token→语义 token 的链式推理，让属性控制有了层次感
- **VoxBox 数据集贡献**: 100K 小时全开源 + 性别/音高/语速标注，填补了可控 TTS 训练数据的空白
- **效率优势**: 0.5B 参数打败 8B 同类模型，50 TPS 的低帧率减少了序列长度

## 局限与展望

1. 说话人相似度（SIM）低于多阶段方法，AR 语言模型推理时引入的说话人变异性是根本原因
2. 全局 token 和语义 token 之间未施加显式解耦约束，可考虑通过 formant/pitch 扰动增强解耦
3. BiCodec 训练数据仅约 3000 小时，扩大训练规模可能进一步提升重建质量
4. 缺乏情感控制能力，VoxBox 虽包含情感数据集但未显式利用
5. 仅支持英语和中文，多语言扩展有待探索

## 相关工作与启发

- TiCodec (Ren et al., 2024) 最相似但使用 GVQ 做全局信息，BiCodec 用 FSQ + 可学习 query 的 Cross Attention 更优
- Llasa (Ye et al., 2025) 用 FSQ 单 codebook + LLaMA 的方案，但没有全局 token 分离，需要更大模型（8B）才能达到类似效果
- CosyVoice2 需要额外 flow matching 预测声学特征，Spark-TTS 直接由 BiCodec decoder 重建，架构更简

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — BiCodec 的语义+全局双 token 解耦 + CoT 控制生成，设计优雅且原创
- **实验充分度**: ⭐⭐⭐⭐ — codec 重建/零样本 TTS/控制实验全面，但说话人相似度弱项需更多分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，系统描述完整
- **价值**: ⭐⭐⭐⭐⭐ — 代码+模型+数据集全开源，VoxBox 100K 小时是重大贡献

<!-- RELATED:START -->

## 相关论文

- [MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens](mms-llama_efficient_llm-based_audio-visual_speech_recognition_with_minimal_multi.md)
- [Zero-Shot Text-to-Speech for Vietnamese](zero-shot_text-to-speech_for_vietnamese.md)
- [SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)
- [Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment](advancing_zero-shot_text-to-speech_intelligibility_across_diverse_domains_via_pr.md)
- [Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation](eta-wavlm_efficient_speaker_identity_removal_in_self-supervised_speech_represent.md)

<!-- RELATED:END -->
