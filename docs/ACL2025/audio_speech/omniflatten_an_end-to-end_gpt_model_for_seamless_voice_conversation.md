---
title: >-
  [论文解读] OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation
description: >-
  [ACL 2025][语音][全双工对话] 提出 OmniFlatten——基于 Qwen2-0.5B 的端到端全双工语音对话模型，通过三阶段渐进式后训练（模态对齐→半双工→全双工对话学习）和统一的 flatten 操作，在不修改 GPT 架构的前提下实现了低延迟的自然全双工语音交互，turn-taking 响应时间仅 193ms，显著优于 Moshi 的 553ms。
tags:
  - ACL 2025
  - 语音
  - 全双工对话
  - 端到端语音模型
  - GPT架构
  - chunk分块
  - 多阶段训练
  - 模态对齐
  - turn-taking
---

# OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation

**会议**: ACL 2025  
**arXiv**: [2410.17799](https://arxiv.org/abs/2410.17799)  
**代码**: 未公开  
**Demo**: [https://omniflatten.github.io/](https://omniflatten.github.io/)  
**作者**: Qinglin Zhang, Luyao Cheng, Chong Deng, Qian Chen, Wen Wang 等  
**机构**: 阿里巴巴通义实验室 (Tongyi Lab)  
**领域**: 语音对话 / 多模态  
**关键词**: 全双工对话, 端到端语音模型, GPT架构, chunk分块, 多阶段训练, 模态对齐, turn-taking

## 一句话总结

提出 OmniFlatten——基于 Qwen2-0.5B 的端到端全双工语音对话模型，通过三阶段渐进式后训练（模态对齐→半双工→全双工对话学习）和统一的 flatten 操作，在不修改 GPT 架构的前提下实现了低延迟的自然全双工语音交互，turn-taking 响应时间仅 193ms，显著优于 Moshi 的 553ms。

## 研究背景与动机

**领域现状**：
   - **协作式系统**（如 Qwen-audio）：LLM 对话模块外接 ASR/TTS，半双工交互，延迟高
   - **端到端系统**（如 SpeechGPT、LLaMA-Omni、GLM-4-Voice）：直接建模语音到语音，但大多为 turn-based，不支持全双工
   - **全双工系统**：Moshi 通过多流并行建模实现全双工，但需要复杂的声学延迟和内心独白设计，不被 GPT 模型原生支持

**核心挑战**：
   - 全双工对话需要同时处理**说话、倾听和思考**，涉及打断、回声应答、语音重叠等复杂人机交互行为
   - 如何在不修改 GPT 架构的前提下实现全双工能力
   - 缺乏大规模全双工对话训练数据

**关键创新点**：提出 "flatten" 操作——将多流（用户语音/文本 + 助手语音/文本）数据统一展平为单一序列，使标准 GPT 模型无需结构修改即可处理全双工对话

## 方法详解

### 整体架构

OmniFlatten 由以下核心组件构成：
- **音频 Tokenizer**：采用 CosyVoice 的语音 tokenizer（单码本，4096 codes），将语音转为离散 token 序列
- **基础模型**：Qwen2-0.5B（文本 LLM，参数量小但性价比高）
- **音频 Detokenizer**：OT-CFM + HifiGAN vocoder，将语音 token 转回音频

### 三阶段训练方案

#### Stage 1: 模态对齐（Modality Alignment）

- **目标**：让文本 LLM 学会语音-文本对应关系，获得 ASR 和 TTS 能力
- **训练数据**：约 10 万小时音频（30% 开源 + 70% 私有），包括 Aishell-3、LibriTTS、Wenetspeech 等
- **训练格式**：
    - ASR: `[ASR][SOS]语音tokens[EOS][SOT]文本tokens[EOT]`
    - TTS: `[TTS][SOT]文本tokens[EOT][SOS]语音tokens[EOS]`
- **序列长度**：1024 tokens

#### Stage 2: 半双工对话学习（Half-duplex Dialogue Training）

- **目标**：学习基本的多轮对话能力（用户和助手交替说话，无重叠）
- **四流数据**：用户语音、用户文本、助手文本、助手语音
- **Flatten 操作**：将四流数据按实际说话顺序展平为单一序列
- **课程学习**：先掌握简单的 turn-based 对话，再进入复杂的全双工学习

#### Stage 3: 全双工对话学习（Full-duplex Dialogue Training）

分两步渐进式训练：

**Step 1——三流训练**：
- 移除用户文本流，保留用户语音 + 助手文本 + 助手语音
- 引入 **chunk 分块**：将语音和文本序列切分为固定大小的短 chunk（语音 chunk = 10 tokens，文本 chunk = 2 tokens）
- chunk 间交替排列：输入语音 → 输出文本 → 输出语音
- 用 `silent_text_token` 和 `silent_speech_token` 填充静默区域

**Step 2——两流训练**：
- 进一步移除助手文本流，仅保留用户语音 → 助手语音
- 消除对文本中间表示的依赖，实现纯语音到语音的全双工生成

### 数据合成流水线

由于缺乏真实全双工对话数据，设计了完整的合成管道：

1. **文本对话收集**：从 Alpaca、Moss、BelleCN、UltraChat 收集约 39 万多轮对话文本
2. **语音合成**：用 CosyVoice 将文本转语音（用户音色从 LibriSpeech 和 3D-Speaker 采样，助手固定音色）
3. **交互模拟**：模拟三种关键场景——正常对话衔接、用户打断、助手等待
4. **噪声添加**：从 MUSAN 数据集采样背景噪声，以 15-30dB SNR 混入用户通道
5. **总量**：2000 小时多通道语音对话数据

### 训练细节

- 对话学习阶段最大序列长度：8192 tokens
- 用户通道应用 loss masking（屏蔽用户输入的损失计算）
- AdamW 优化器，weight decay=0.1，最大学习率 2e-05
- 5 个 epoch，每 batch 包含 100M tokens

## 实验

### ASR 性能评估（Table 1）

模态对齐阶段后的 ASR 性能：

| 模型 | Librispeech clean | Librispeech other | Wenetspeech meeting | Wenetspeech net |
|------|------------------|------------------|--------------------|-----------------| 
| Whisper-Small | 3.13 (WER) | 7.37 | 25.62 (CER) | 16.66 |
| Whisper-Large | 1.82 | 3.50 | 18.87 | 10.48 |
| VITA | 8.14 | 18.4 | 12.15 | 16.53 |
| OmniFlatten | 7.91 | 19.21 | 26.1 | 19.0 |

OmniFlatten ASR 性能与 VITA 相当，虽不及专用 Whisper 模型，但证明模态对齐阶段成功建立了语音-文本对应关系。

### TTS 性能评估（Table 2）

| 模型 | LibriTTS (WER↓) | AIShell-3 (CER↓) |
|------|-----------------|-------------------|
| Original | 2.66 | 2.52 |
| ChatTTS | 8.32 | 3.87 |
| CosyVoice | 2.89 | 3.82 |
| OmniFlatten | 4.51 | 4.46 |

TTS 质量合理，位于 CosyVoice 和 ChatTTS 之间。

### 全双工对话质量评估（Table 3, LLM 评分 1-10）

| 模型 | 参数 | En Text | En Speech | Zh Text | Zh Speech |
|------|------|---------|-----------|---------|-----------|
| Qwen2-0.5B-Instruct | 0.5B | 6.75 | - | 6.98 | - |
| Qwen2-7B-Instruct | 7B | 8.37 | - | 8.09 | - |
| LLaMA-Omni | 8B | 6.01 | 5.50 | 4.17 | 3.89 |
| Moshi | 7B | 3.92 | 3.46 | - | - |
| GLM-Voice | 9B | 6.97 | 6.40 | 7.02 | 6.69 |
| OmniFlatten直接3流 | 0.5B | 2.99 | 2.59 | 4.94 | 3.95 |
| OmniFlatten 3流无半双工 | 0.5B | 3.89 | 3.54 | 5.25 | 4.76 |
| OmniFlatten 3流完整 | 0.5B | **4.88** | **3.92** | **5.60** | **5.15** |
| OmniFlatten 2流完整 | 0.5B | - | 2.19 | - | 3.06 |
| Ground Truth | - | 7.65 | - | 6.83 | - |

**关键发现**：
- 每个训练阶段都贡献了性能提升：直接三流(2.99) → 加模态对齐(3.89) → 再加半双工(4.88)
- OmniFlatten 在英文上优于 Moshi（4.88 vs 3.92），在中文上优于 LLaMA-Omni
- 2 流模型（仅语音输出）性能显著下降，说明文本中间表示仍然重要

### Turn-taking 性能（Table 4）

| 模型 | Asst Turn-taking Acc@1/5/10/25 | Asst 响应时间 | User Turn-taking Acc@1/5/10/25 | User 响应时间 |
|------|-------------------------------|-------------|-------------------------------|-------------|
| Moshi | 2.9/18.8/38.5/55.1% | 553ms | 0.0/6.2/14.8/45.7% | 753ms |
| OmniFlatten | **20.6/53.6/66.3/71.7%** | **193ms** | **10.9/30.9/41.8/51.8%** | **287ms** |

OmniFlatten 在 turn-taking 上全面超越 Moshi：
- 助手响应时间：193ms vs 553ms（快 **2.9 倍**）
- 用户打断响应时间：287ms vs 753ms（快 **2.6 倍**）
- Acc@1：20.6% vs 2.9%（高 **7 倍**）

## 亮点与洞察

1. **Flatten 操作的统一性**：通过将多流数据展平为单一序列，统一了不同模态和任务的训练方式，避免了 Moshi 式的复杂并行架构设计
2. **渐进式训练的有效性**：消融实验清晰展示了模态对齐(+0.9) → 半双工(+1.0) 每阶段的边际贡献，验证了课程学习在多模态对话中的价值
3. **极小模型的全双工能力**：仅 0.5B 参数即实现全双工对话，证明全双工能力可以通过训练策略而非模型规模获得
4. **极低 turn-taking 延迟**：193ms 的助手响应时间已接近人类对话的自然响应延迟（约 200ms），远优于 Moshi
5. **不修改 GPT 架构**：方法完全兼容标准 Transformer 解码器，可直接应用于任意预训练 LLM

## 局限性

1. **基础模型过小**：Qwen2-0.5B 的容量限制了对话质量（文本评分仅 4.88 vs Ground Truth 的 7.65），paper 也承认扩大模型规模可大幅提升性能
2. **合成数据质量有限**：全双工数据通过 TTS 合成 + 规则模拟生成，无法覆盖真实对话的丰富交互模式（如回应、语气变化等）
3. **User Turn-taking 成功率偏低**：即使在 25 tokens 内，OmniFlatten 的用户打断识别准确率也仅 51.8%，远未达到实用水平
4. **2 流模型性能退化严重**：去掉文本中间表示后性能大幅下降（5.60→3.06），说明纯语音到语音的全双工仍是未解难题
5. **缺乏 Backchannel 建模**：当前模型不支持更复杂的交互行为（如用户 "嗯嗯" 的回应声、助手的确认反馈等）
6. **评估依赖 LLM 打分**：对话质量评估依赖 QWen-max 的 LLM 评分，与人类评估的相关性未验证

## 相关工作

- **半双工语音对话**：SpeechGPT (Zhang et al., 2023)、Mini-Omni (Xie & Wu, 2024a)、LLaMA-Omni (Fang et al., 2024)、GLM-4-Voice (Zeng et al., 2024)
- **全双工语音对话**：Moshi (Défossez et al., 2024) 多流并行、dGSLM (Nguyen et al., 2023)、VITA (Fu et al., 2024)、SyncLM (Veluri et al., 2024) chunk 交替
- **语音 Tokenizer**：CosyVoice (Du et al., 2024a)、HifiGAN (Kong et al., 2020)
- **多模态 LLM**：Qwen-Audio (Chu et al., 2024)、SALMONN (Tang et al., 2024)
- **全双工探索**：LSLM (Ma et al., 2024) 实时监听+说话

## 评分

⭐⭐⭐⭐ — 方法简洁优雅（不改架构、渐进训练、flatten 统一），turn-taking 延迟指标出色；但基础模型太小导致对话质量不够好，合成数据也限制了真实场景的覆盖度，是一项有价值的探索性工作。

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Distilling an End-to-End Voice Assistant Without Instruction Training Data](distilling_an_end-to-end_voice_assistant_without_instruction_training_data.md)
- [\[ACL 2025\] Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition](contextual_biasing_with_the_knowledgeable_external_language_model_for_end-to-end.md)
- [\[ACL 2025\] DNCASR: End-to-End Training for Speaker-Attributed ASR](dncasr_end-to-end_training_for_speaker-attributed_asr.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)
- [\[NeurIPS 2025\] E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis](../../NeurIPS2025/audio_speech/e2e-vguard_adversarial_prevention_for_production_llm-based_end-to-end_speech_syn.md)

<!-- RELATED:END -->
