---
title: >-
  [论文解读] EMoVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions
description: >-
  [CVPR 2025][语音][全模态LLM] 提出 EMoVA，首个端到端的全模态 LLM，通过语义-声学解耦的语音 tokenizer 同时实现视觉理解、语音识别和情感可控的语音合成，在视觉语言基准上超越 GPT-4o，语音识别 WER 达 2.9%。
tags:
  - CVPR 2025
  - 语音
  - 全模态LLM
  - 语义-声学解耦
  - 情感语音生成
  - 语音理解
  - 端到端对话
---

# EMoVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions

**会议**: CVPR 2025  
**arXiv**: [2409.18042](https://arxiv.org/abs/2409.18042)  
**代码**: https://emova-ollm.github.io/ (有)  
**领域**: 音频语音 / 多模态LLM  
**关键词**: 全模态LLM, 语义-声学解耦, 情感语音生成, 语音理解, 端到端对话

## 一句话总结

提出 EMoVA，首个端到端的全模态 LLM，通过语义-声学解耦的语音 tokenizer 同时实现视觉理解、语音识别和情感可控的语音合成，在视觉语言基准上超越 GPT-4o，语音识别 WER 达 2.9%。

## 研究背景与动机

**领域现状**：多模态 LLM 领域快速发展，视觉-语言模型（如 LLaVA、Qwen-VL）和语音-语言模型（如 SpeechGPT、VITA）分别取得了显著进展，但同时支持"看-听-说"三种模态的端到端模型仍很少。

**现有痛点**：现有全模态方法面临两个关键问题：（1）语音 token 化时语义和声学信息纠缠——如果直接量化语音为离散 token（如 HuBERT K-means），说话风格（情感/语调）会干扰语义内容的学习，反之亦然；（2）多模态联合训练时不同模态互相干扰——先训练视觉再训练语音（或反之）会导致灾难性遗忘。

**核心矛盾**：语音本质上是"内容+风格"的复合信号。如果把两者混在一起用统一码本量化，LLM 在预测下一个 token 时既要推理内容又要预测语调，任务过于复杂。

**本文目标** 设计一个能同时高质量处理视觉、语音输入和情感可控的语音输出的端到端全模态 LLM。

**切入角度**：将语音信号解耦为语义表示（量化为离散 token 给 LLM）和声学风格表示（保持连续，用于控制合成语音的情感和音调），让 LLM 只需专注于语义推理。

**核心 idea**：语义-声学解耦的语音 tokenizer + 多模态联合训练 = 端到端"看-听-说"全模态LLM。

## 方法详解

### 整体框架

四个核心组件：（1）视觉编码器 QwenViT + MLP 投影器处理图像；（2）S2U（Speech-to-Unit）分词器将语音解耦并量化为 4096 码本的语义 token（25 token/秒）；（3）核心 LLM 为 Qwen-2.5（3B/7B/72B）；（4）U2S（Unit-to-Speech）去分词器基于 VITS 将语义 token + 风格嵌入合成语音波形。三阶段训练：视觉对齐 → 全模态联合对齐 → 全模态指令微调。

### 关键设计

1. **语义-声学解耦的语音 Tokenizer (S2U)**:

    - 功能：将语音分离为语义内容和声学风格两个独立表示
    - 核心思路：SPIRAL 编码器先提取语音嵌入，然后分叉为两路：语义路通过 Finite Scalar Quantization (FSQ) 量化为 4096 码本的离散 token $E_{\text{semantic}}$，声学路保持连续表示 $E_{\text{style}}$ 用于控制合成时的情感/音调。语义分支附加一个音素解码器确保保留足够的语言学信息
    - 设计动机：消融显示联合训练中，解耦比纠缠版本（HuBERT K-means）在 ASR 和视觉任务上都有显著优势。解耦让 LLM 只处理"说了什么"而非"怎么说的"

2. **全模态联合对齐训练**:

    - 功能：同时训练视觉-文本和语音-文本对齐，避免模态间灾难性遗忘
    - 核心思路：Stage 2 使用 7.4M 样本（图像-文本 + 语音-文本）同时训练。所有模态统一为文本为中心的序列——视觉特征通过投影器映射到文本嵌入空间，语音 token 直接进入 LLM 词表
    - 设计动机：消融显示联合训练 > 先视觉后语音或先语音后视觉的序列训练——后者都因灾难性遗忘导致先训练模态性能下降

3. **情感可控的语音合成 (U2S)**:

    - 功能：生成带有指定情感和语调的语音
    - 核心思路：基于 VITS 的条件 VAE，输入语义 token + 风格嵌入（4 种情感 × 3 种音调 × 2 种性别 = 24 种组合）。LLM 先生成文本回复，再预测情感标签和语音 token，U2S 将其合成为音频
    - 设计动机：Chain-of-Modality 生成——文本→情感→语音的串行生成让每步的决策空间更小更可控

### 损失函数 / 训练策略

Stage 1: 视觉投影器对齐（LCS-558K）。Stage 2: 自回归语言模型损失 $\mathcal{L} = -\sum_i \log P(x_i|x_{<i})$，7.4M 联合对齐数据。Stage 3: 4.4M 多任务指令微调+情感标签。S2U 用对比损失 + 音素重建损失预训练（20K 小时无标注语音）。U2S 用 VITS 的 VAE+GAN 损失。训练在 128 × Ascend 910B NPU 上完成。

## 实验关键数据

### 主实验

视觉-语言基准（EMOVA-72B vs 其他全模态LLM）：

| 基准 | EMOVA-72B | GPT-4o | Gemini-1.5-Pro |
|------|-----------|--------|---------------|
| MME | **2402** | 2329 | 2228 |
| MMBench | **86.4** | 83.4 | 73.9 |
| TextVQA | 81.4 | - | 73.5 |
| DocVQA | **95.9** | 92.8 | 93.1 |
| OCRBench | **843** | 736 | 754 |

语音识别（LibriSpeech test-clean WER↓）：

| 模型 | WER |
|------|-----|
| Mini-Omni | 4.8 |
| VITA | 8.1 |
| **EMOVA-72B** | **2.9** |
| Whisper-Large | 3.0 |

### 消融实验

| 训练策略 | MMBench | ASR WER |
|---------|---------|---------|
| 仅视觉 | 83.2 | - |
| 仅语音 | - | 4.1 |
| 序列（VL→Speech） | 81.5 | 4.3 |
| 序列（Speech→VL） | 83.0 | 5.9 |
| **联合（解耦）** | **83.8** | **4.1** |
| 联合（纠缠/HuBERT） | 82.1 | 6.3 |

### 关键发现
- **联合训练优于序列训练**：避免了灾难性遗忘，所有模态同时受益
- **解耦 >> 纠缠**：语义-声学解耦让 ASR WER 从 6.3 降到 4.1，视觉指标也从 82.1 升到 83.8
- **72B 模型支撑更好的 ASR**：WER 2.9 接近甚至超越 Whisper-Large（3.0），证明 LLM 规模对跨模态任务有直接帮助
- **情感分类准确率 >75%**：合成语音的情感可控性有效

## 亮点与洞察

- **语义-声学解耦的根本洞察**：语音的"内容"和"风格"是两种本质不同的信息——LLM 擅长推理内容但不擅长建模声学细节。解耦让各模块各司其职
- **11/15 基准超越 GPT-4o**：72B 版本在大部分视觉-语言基准上超越 GPT-4o，说明开源全模态 LLM 正在追上闭源模型
- **Chain-of-Modality 生成策略**：先生成文本→再生成情感标签→再生成语音 token 的串行方式，在每个步骤中减少了决策空间

## 局限与展望

- **半双工限制**：模型只能交替处理输入/输出，无法像人类对话那样同时听和说
- **文本中介依赖**：语音生成必须先生成文本再转语音，增加了延迟
- **单视觉编码器**：只用 QwenViT，未利用自监督（DINOv2）或专家混合视觉模型
- **仅支持视觉理解**：不支持图像生成（vs Emu3）
- **训练资源巨大**：128 个 Ascend 910B NPU，复现门槛极高

## 相关工作与启发

- **vs VITA**: VITA 也支持语音但 WER 8.1 远不如 EMOVA 的 2.9，且不支持情感可控合成
- **vs Mini-Omni**: Mini-Omni 的实时语音交互更快但理解能力弱（WER 4.8）
- **vs AnyGPT**: AnyGPT 用纠缠的 SpeechTokens，未做解耦，多模态联合训练效果差

## 评分
- 新颖性: ⭐⭐⭐⭐ 语义-声学解耦 + 联合训练策略的组合有效且新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 视觉+语音+情感全面评估，消融充分，3种模型规模对比
- 写作质量: ⭐⭐⭐⭐ 架构清晰，但训练细节略显分散
- 价值: ⭐⭐⭐⭐⭐ 首个全面超越 GPT-4o 的开源全模态 LLM，方向意义重大

<!-- RELATED:START -->

## 相关论文

- [A Controllable Examination for Long-Context Language Models](../../NeurIPS2025/audio_speech/a_controllable_examination_for_longcontext_language_models.md)
- [AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](../../NeurIPS2025/audio_speech/audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)
- [Investigating and Enhancing Vision-Audio Capability in Omnimodal Large Language Models](../../ACL2025/audio_speech/investigating_and_enhancing_vision-audio_capability_in_omnimodal_large_language_.md)
- [Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models](../../ACL2025/audio_speech/audio_dialogue_benchmark.md)
- [ControlLLM: Augment Language Models with Tools by Searching on Graphs](../../ECCV2024/audio_speech/controlllm_augment_language_models_with_tools.md)

<!-- RELATED:END -->
