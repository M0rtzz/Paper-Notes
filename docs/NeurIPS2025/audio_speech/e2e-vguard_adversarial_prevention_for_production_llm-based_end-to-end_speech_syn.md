---
title: >-
  [论文解读] E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis
description: >-
  [NEURIPS2025][语音][voice cloning defense] 针对基于 LLM 的端到端语音合成中的声音克隆威胁，提出 E2E-VGuard 主动防御框架，通过编码器集成扰动音色、对抗样本干扰 ASR 发音识别、以及心理声学模型保证不可感知性，在 19 个 TTS 模型和 7 个 ASR 系统上验证了有效性。
tags:
  - NEURIPS2025
  - 语音
  - voice cloning defense
  - adversarial examples
  - speech synthesis
  - psychoacoustic model
  - LLM-based TTS
---

# E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis

**会议**: NEURIPS2025  
**arXiv**: [2511.07099](https://arxiv.org/abs/2511.07099)  
**代码**: [wxzyd123/e2e-vguard](https://wxzyd123.github.io/e2e-vguard/)  
**领域**: audio_speech  
**关键词**: voice cloning defense, adversarial examples, speech synthesis, psychoacoustic model, LLM-based TTS

## 一句话总结

针对基于 LLM 的端到端语音合成中的声音克隆威胁，提出 E2E-VGuard 主动防御框架，通过编码器集成扰动音色、对抗样本干扰 ASR 发音识别、以及心理声学模型保证不可感知性，在 19 个 TTS 模型和 7 个 ASR 系统上验证了有效性。

## 背景与动机

- 基于 LLM 的语音合成（如 CosyVoice、GPT-SoVITS）已实现人级别合成质量，但也带来了严重的声音克隆欺诈风险（如电信诈骗）
- 现有防御方法（AntiFake、AttackVC、POP、SafeSpeech）主要针对传统 DNN-based TTS 或假设文本已手动标注，无法应对两个新兴场景：
    1. **产业级 LLM-based TTS**：音频被编码为离散 token 输入 LLM，与传统连续特征提取方式不同
    2. **端到端（E2E）场景**：商业 API（如字节跳动、阿里）仅接收音频输入，后端自动使用 ASR 系统转录文本，攻击者无需手动标注
- 现实中攻击者从 YouTube/Bilibili 等平台收集音频，不附带文本，必须依赖 ASR 自动识别——这意味着 ASR 环节本身成为新的可利用防御点

## 核心问题

如何在端到端语音合成流程中，同时从**音色**和**发音**两个维度实施主动防御，使被保护音频在被 LLM-based TTS 模型克隆后产生不可辨识的合成语音，同时保证扰动对人耳不可感知？

## 方法详解

E2E-VGuard 的总体优化目标为：

$$\mathcal{L}(x') = \mathcal{L}_{asr}(x') + \alpha \cdot \mathcal{L}_{fea}(x') + \beta \cdot \mathcal{L}_{psy}(x')$$

其中 $x'$ 为受保护音频，$\alpha=500$，$\beta=5 \times 10^{-3}$，扰动约束 $\epsilon = 8/255$，优化迭代 500 步。

### 1. 音色防御（Timbre Prevention）

采用**编码器集成**策略，使用 6 个异构编码器（VITS/GSV posterior encoder、MFCC、WavLM、CAM++、StyleTTS2 style encoder）提取音频特征，提升跨模型泛化性。

- **无目标保护（Untargeted）**：最大化原始音频 $x$ 与受保护音频 $x'$ 之间的特征距离

$$\mathcal{L}_{fea}(x') = \sum_{i=1}^{k} \text{CS}(E_i(x), E_i(x')) + \text{CS}(M(x), M(x'))$$

- **有目标保护（Targeted）**：将音频特征引导至预选最不相似说话人 $x_t$

$$\mathcal{L}_{fea}(x') = -\sum_{i=1}^{k} [\text{CS}(E_i(x_t), E_i(x')) + \text{CS}(M(x_t), M(x'))]$$

引入 MFCC 特征提取器专门应对 LLM-based TTS 中音频离散化编码的特点，干扰 LLM 组件获取的韵律和语调信息。

### 2. 发音防御（Pronunciation Prevention）

利用对抗样本攻击 ASR 系统，使其将受保护音频误识别为指定目标文本：

$$\mathcal{L}_{asr}(x') = \mathcal{F}(\text{ASR}(x'), y_t)$$

- 采用**有目标攻击**（非乱码），生成可读但错误的识别文本，降低攻击者的警觉性
- 目标文本选择策略：有目标音色保护时使用目标说话人文本；无目标音色保护时选择等长度不同文本
- 错误的文本-音频对会破坏 TTS 模型中文本与发音的对齐学习（如 VITS 的单调对齐搜索）

### 3. 心理声学模型（Psychoacoustic Model）

利用频率掩蔽效应确保扰动不可感知：

$$\mathcal{L}_{psy}(x') = \frac{1}{F} \sum_{f=1}^{F} \max(0, p_{x'-x}(f) - \theta_x(f))$$

结合 $\ell_2$ 范数约束进一步降低人耳对嵌入扰动的感知度。最终音频特征映射回 $[-1, 1]$ 范围保证波形合法性。

## 实验关键数据

**实验规模**：16 个开源 TTS + 3 个商业 API（字节跳动/阿里/MiniMax），7 个 ASR 系统，中英文数据集（LibriTTS/CMU ARCTIC/THCHS30），在单张 NVIDIA 4090 上完成。

### 端到端微调场景（核心结果，Table 1）

| 方法 | GSV WER↑ | GSV SIM↓ | CosyVoice WER↑ | CosyVoice SIM↓ | VITS WER↑ | VITS SIM↓ | SNR↑ |
|------|----------|----------|-----------------|-----------------|-----------|-----------|------|
| Clean | 3.4 | 0.685 | 4.3 | 0.700 | 7.8 | 0.710 | - |
| AntiFake | 28.8 | 0.149 | 7.8 | 0.232 | 41.5 | 0.257 | 12.8 |
| SafeSpeech | 44.8 | 0.339 | 8.6 | 0.459 | 105.5 | 0.180 | 7.6 |
| **E2E-VGuard (UT)** | **66.5** | **0.123** | **21.6** | **0.091** | **95.7** | **0.106** | **18.5** |
| **E2E-VGuard (T)** | **94.8** | 0.284 | **72.1** | 0.375 | **125.3** | 0.245 | **20.5** |

- WER 平均提升 19.8%（T 模式）对比最佳基线；SIM 平均降低 0.043（UT 模式）
- 零样本场景提升更显著：WER 平均提升 32.8%（UT）/ 50.1%（T），SIM 降低 0.119（UT）

### 零样本场景（Table 2，工业级 LLM-based 模型）

在 Index-TTS、FireRedTTS-1S、Step-Audio-TTS、Spark-TTS 等 7 个最新模型上，E2E-VGuard (UT) 的 SIM 在所有模型上均达到 SOTA，WER 均值 21.6%（UT）/ 23.6%（T），大幅超越 AntiFake 的 4.9% 和 SafeSpeech 的 19.3%。

### 感知质量

E2E-VGuard 的 SNR 高于所有基线（18.5-20.5 dB），PESQ 达到 1.9-2.3，表明扰动噪声比最低、音频质量退化最小。

## 亮点

1. **首次系统化定义 E2E 语音合成防御场景**：明确 ASR→TTS 流水线中 ASR 环节作为新防御点的重要性，贴合商业 API 实际部署
2. **音色+发音双维度联合防御**：不仅干扰音色特征还破坏文本-发音对齐，形成双重保护
3. **编码器集成 + MFCC**：用异构编码器集成提升跨模型泛化性，用 MFCC 专门应对 LLM-based TTS 的离散 token 编码
4. **心理声学模型保证隐蔽性**：频率掩蔽 + $\ell_2$ 约束使扰动几乎不可感知
5. **评估极其全面**：19 个 TTS 模型（含 3 个商业 API）× 7 个 ASR 系统 × 中英文数据集，包含真实部署验证

## 局限与展望

1. **对抗鲁棒性**：论文虽然测试了数据增强和扰动移除的鲁棒性，但高级自适应攻击（知道防御方法后的针对性绕过）尚未充分讨论
2. **实时性**：500 步迭代优化的计算开销较大，难以满足实时音频保护需求
3. **ASR 系统依赖**：对抗样本针对特定 ASR 系统生成，若攻击者使用完全未知的 ASR 系统可能降低效果
4. **感知质量权衡**：PESQ 得分（1.9-2.3）仍有提升空间，尤其在高质量音频分享场景中
5. **目标文本选择策略**：当前目标文本选择较为启发式，可以探索更自动化的最优目标文本搜索

## 与相关工作的对比

| 方法 | 防御类型 | 音色保护 | 发音保护 | LLM-based TTS | E2E 场景 | 心理声学 |
|------|---------|---------|---------|---------------|---------|---------|
| AttackVC | 对抗样本 | ✓ | ✗ | ✗ | ✗ | ✗ |
| AntiFake | 对抗样本+编码器集成 | ✓ | ✗ | ✗ | ✗ | ✗ |
| POP / SafeSpeech | 不可学习样本 | ✓ | ✗ | ✗ | ✗ | ✗ |
| **E2E-VGuard** | **对抗样本+编码器集成+ASR攻击** | **✓** | **✓** | **✓** | **✓** | **✓** |

E2E-VGuard 是首个同时覆盖 LLM-based TTS 防御和 E2E 场景的方法，通过引入 ASR 攻击实现发音维度防御——这在之前工作中完全缺失。

## 启发与关联

- **攻击面扩展思路**：将 ASR 系统作为间接攻击目标是一个巧妙的思路——在多阶段流水线中，任何中间环节都可能成为防御点
- **与多模态安全的联系**：类似思想可推广到图像→文本→生成流水线（如 OCR→LLM 场景）的主动防御
- **编码器集成策略**：异构编码器集成提升对抗迁移性的方法可借鉴到其他对抗攻防领域（图像、视频）
- **工业落地参考**：论文验证了商业 API（字节跳动/阿里/MiniMax），提供了从学术到产业部署的可行路径

## 评分
- 新颖性: 8/10 — E2E 场景定义和 ASR 攻击防御发音的思路新颖
- 实验充分度: 9/10 — 19 个 TTS + 7 个 ASR + 商业 API + 真实部署，极为充分
- 写作质量: 7/10 — 内容丰富但部分公式符号说明可更清晰
- 价值: 8/10 — 直击当前语音克隆安全痛点，具有较高实用价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition](../../ACL2025/audio_speech/contextual_biasing_with_the_knowledgeable_external_language_model_for_end-to-end.md)
- [\[ACL 2025\] DNCASR: End-to-End Training for Speaker-Attributed ASR](../../ACL2025/audio_speech/dncasr_end-to-end_training_for_speaker-attributed_asr.md)
- [\[ACL 2025\] OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation](../../ACL2025/audio_speech/omniflatten_an_end-to-end_gpt_model_for_seamless_voice_conversation.md)
- [\[ACL 2025\] Distilling an End-to-End Voice Assistant Without Instruction Training Data](../../ACL2025/audio_speech/distilling_an_end-to-end_voice_assistant_without_instruction_training_data.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)

<!-- RELATED:END -->
