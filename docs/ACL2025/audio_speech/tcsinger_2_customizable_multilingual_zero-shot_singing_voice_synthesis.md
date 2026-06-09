---
title: >-
  [论文解读] TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis
description: >-
  [ACL2025][音频/语音][歌声合成] 提出 TCSinger 2，一个多任务多语言零样本歌声合成模型，通过模糊边界编码器、对比学习音频编码器和基于 Flow 的自定义 Transformer（含 Cus-MOE），实现基于歌声/语音/文本提示的风格迁移与多层级风格控制。
tags:
  - "ACL2025"
  - "音频/语音"
  - "歌声合成"
  - "零样本"
  - "多语言"
  - "风格迁移"
  - "风格控制"
  - "Flow Matching"
  - "Mixture of Experts"
---

# TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis

**会议**: ACL2025  
**arXiv**: [2505.14910](https://arxiv.org/abs/2505.14910)  
**代码**: [AaronZ345/TCSinger2](https://github.com/AaronZ345/TCSinger2)  
**领域**: 音频语音  
**关键词**: 歌声合成, 零样本, 多语言, 风格迁移, 风格控制, Flow Matching, Mixture of Experts

## 一句话总结

提出 TCSinger 2，一个多任务多语言零样本歌声合成模型，通过模糊边界编码器、对比学习音频编码器和基于 Flow 的自定义 Transformer（含 Cus-MOE），实现基于歌声/语音/文本提示的风格迁移与多层级风格控制。

## 研究背景与动机

### 问题背景

可定制的多语言零样本歌声合成（SVS）在音乐创作和短视频配音中有广泛应用前景。现有 SVS 模型面临两大核心挑战：

**过度依赖精确的音素和音符边界标注**：数据集（如 OpenCpop）依赖 MFA 和人耳对齐，边界标注存在大量误差；在零样本场景下，音素和音符之间的过渡尤其不自然

**缺乏有效的多层级风格控制**：现有模型无法通过自然语言文本、语音或歌声等多种提示灵活控制歌声风格（包括音色、唱法、情感、技巧等多个层级）

### 前作局限

- TCSinger（前作）仅支持标签或音频提示的风格控制，无法处理自然语言文本提示
- Choi and Nam (2022) 提出旋律无监督模型减少对边界标注的依赖，但合成质量下降且无法保证平滑过渡
- StyleTTS 2、CosyVoice 等语音合成模型无法建模多层级歌唱风格

## 方法详解

### 整体架构

TCSinger 2 包含三个核心模块，输入为歌词 l、乐谱 n 和提示 P（歌声/语音/文本三选一），输出合成歌声。

### 1. 模糊边界内容编码器（BBC Encoder）

- **动机**：现有模型依赖精确的音素/音符边界，但标注误差普遍存在，尤其在多语言数据集中
- **方法**：将歌词和乐谱分别编码后预测时长，扩展为帧级序列；在每个音素和音符边界随机遮蔽 m=8 个 token，产生模糊边界
- **效果**：迫使模型学习隐式对齐路径，提升过渡自然度和零样本生成鲁棒性；同时可扩充训练数据

### 2. 自定义音频编码器（Custom Audio Encoder）

- **基于 VAE 的歌声/语音编码器**：分别从歌声提示和语音提示提取风格表征
- **文本编码器**：通过交叉注意力融合乐谱和文本提示，获得包含内容和多层级风格的表征
- **对比学习对齐**：设计三种对比类型——(1) 相同内容不同风格，(2) 相似风格不同内容，(3) 不同风格和内容；使用 InfoNCE 目标函数对齐三模态的 triplet pair
- **重建训练**：使用 L2 损失和 LSGAN 对抗损失训练音频解码器，确保歌声表征不丢失完整性

### 3. 基于 Flow 的自定义 Transformer

- **Flow Matching 生成**：将高斯噪声与内容嵌入和提示嵌入拼接，通过 Transformer 自注意力学习内容和风格；训练 1000 步，推理仅需 25 步（Euler ODE 求解器）
- **Cus-MOE（自定义混合专家）**：
    - **Lingual-MOE**：根据歌词语言选择专家，每个专家专注于某一语系（如拉丁语系），提升多语言生成质量
    - **Stylistic-MOE**：根据音频或文本提示选择专家，匹配细粒度风格（如"女中音+欢快流行假声"）
    - 路由策略采用 dense-to-sparse Gumbel-Softmax，带负载平衡损失
- **F0 监督**：利用第一个 Transformer block 的输出预测 F0，为后续 block 提供音高监督信息
- **CFG 策略**：训练时以 0.2 概率随机丢弃提示，推理时使用 gamma=3 的 classifier-free guidance 提升生成质量和风格可控性

### 训练损失

- 音频编码器阶段：对比损失 + L2 重建损失 + LSGAN 对抗损失
- TCSinger 2 主模型：时长损失 + 音高损失 + 负载平衡损失 + Flow Matching 损失

### 支持的推理任务

- 零样本风格迁移（同语言/跨语言）
- 多层级文本风格控制（音色、唱法、情感、技巧）
- 语音到歌声（STS）风格迁移

## 实验关键数据

### 数据集

- 自行收集 50 小时干净歌声 + 多个开源数据集（Opencpop、M4Singer、OpenSinger、PopBuTFy、GTSinger），覆盖 9 种语言（中英法西德意日韩俄）
- 部分数据人工标注多层级风格标签；30 位未见歌手作为测试集
- 模型配置：4 个 Transformer blocks，hidden size 768，8 attention heads，每组 4 个专家
- 训练硬件：8x NVIDIA RTX-4090

### 表1：零样本风格迁移（Parallel / Cross-Lingual）

| 方法 | MOS-Q ↑ | MOS-S ↑ | FFE ↓ | Cos ↑ | MOS-Q (跨语言) ↑ | MOS-S (跨语言) ↑ |
|------|---------|---------|-------|-------|------------------|------------------|
| GT | 4.58 | - | - | - | - | - |
| GT (vocoder) | 4.36 | 4.41 | 0.04 | 0.95 | - | - |
| StyleTTS 2 | 3.71 | 3.79 | 0.42 | 0.71 | 3.58 | 3.63 |
| CosyVoice | 3.74 | 3.93 | 0.33 | 0.87 | 3.63 | 3.77 |
| VISinger 2 | 3.79 | 3.88 | 0.31 | 0.83 | 3.69 | 3.72 |
| TCSinger | 3.94 | 4.01 | 0.26 | 0.91 | 3.77 | 3.87 |
| **TCSinger 2** | **4.13** | **4.27** | **0.21** | **0.93** | **3.96** | **4.09** |

### 表2：文本提示多层级风格控制

| 方法 | MOS-Q ↑ | MOS-C ↑ | FFE ↓ | MOS-Q (非平行) ↑ | MOS-C (非平行) ↑ |
|------|---------|---------|-------|------------------|------------------|
| GT | 4.56 | - | - | - | - |
| GT (vocoder) | 4.26 | 4.32 | 0.06 | - | - |
| StyleTTS 2 | 3.61 | 3.67 | 0.43 | 3.51 | 3.59 |
| CosyVoice | 3.72 | 3.73 | 0.37 | 3.60 | 3.67 |
| VISinger 2 | 3.81 | 3.81 | 0.30 | 3.69 | 3.75 |
| TCSinger | 3.99 | 3.97 | 0.27 | 3.90 | 3.93 |
| **TCSinger 2** | **4.07** | **4.19** | **0.22** | **3.98** | **4.11** |

### 表3：语音到歌声（STS）风格迁移

| 方法 | FFE ↓ | Cos ↑ | MOS-Q ↑ | MOS-S ↑ |
|------|-------|-------|---------|---------|
| GT (vocoder) | 0.06 | 0.93 | 4.21 | 4.20 |
| StyleTTS 2 | 0.41 | 0.71 | 3.60 | 3.52 |
| CosyVoice | 0.39 | 0.79 | 3.66 | 3.65 |
| VISinger 2 | 0.32 | 0.75 | 3.72 | 3.59 |
| TCSinger | 0.28 | 0.82 | 3.89 | 3.84 |
| **TCSinger 2** | **0.24** | **0.89** | **3.97** | **3.96** |

### 表4：消融实验（CMOS 变化）

| 设置 | CMOS-Q (迁移) | CMOS-S (迁移) | CMOS-Q (控制) | CMOS-C (控制) |
|------|-------------|-------------|-------------|-------------|
| TCSinger 2 (完整) | 0.00 | 0.00 | 0.00 | 0.00 |
| w/o BBC Encoder | -0.36 | -0.23 | -0.39 | -0.26 |
| w/o Custom Audio Encoder | -0.21 | -0.37 | -0.19 | -0.41 |
| w/o F0 监督 | -0.33 | -0.24 | -0.31 | -0.27 |
| w/o CFG | -0.26 | -0.22 | -0.25 | -0.31 |
| w/o Cus-MOE | -0.31 | -0.32 | -0.38 | -0.35 |
| w/o Lingual-MOE | -0.29 | -0.17 | -0.32 | -0.21 |
| w/o Stylistic-MOE | -0.21 | -0.26 | -0.23 | -0.33 |

**消融关键发现**：BBC Encoder 对合成质量影响最大（CMOS-Q -0.36/-0.39）；Custom Audio Encoder 对风格控制影响最大（CMOS-C -0.41）；Cus-MOE 整体影响全面且显著。

## 亮点

1. **模糊边界策略新颖实用**：通过随机遮蔽边界 token 而非追求精确对齐，同时解决了标注误差敏感性和过渡不自然两个问题，还附带扩充了训练数据
2. **三模态对比学习统一风格空间**：将歌声、语音、文本提示对齐到同一表征空间，使得模型支持灵活的多模态输入与多任务推理
3. **Cus-MOE 设计精巧**：将语言条件和风格条件分别路由到不同专家组，实现了细粒度的质量与风格的解耦控制
4. **多任务多语言全面评估**：覆盖 9 种语言、4 种任务，在所有任务上全面超越基线

## 局限性

1. **依赖人工标注风格标签**：多层级风格（情感、唱法、技巧等）仍需人工标注，标注成本高且可能引入误差；作者计划未来使用自动标注工具
2. **推理速度不够快**：虽然 Flow Matching 比扩散模型更快（25 步推理），但仍未达到工业级实时流式生成的需求
3. **数据集规模有限**：总训练数据约 268 小时，对于覆盖 9 种语言的零样本场景仍显不足，可能限制泛化能力

## 相关工作

- **歌声合成**：VISinger 2（高保真 SVS）、SiFiSinger（音高控制）、TCSinger（前作，风格迁移+标签控制）
- **风格建模**：StyleTTS 2（韵律预测）、CosyVoice（x-vector + LLM 解耦风格）、PromptSinger（文本描述控制身份）
- **零样本语音合成**：Attentron（注意力机制提取风格）、ZSM-SS（wav2vec 2.0 外部编码器）、MegaTTS 3（稀疏对齐扩散）
- **MoE 相关**：Switch Transformer（负载平衡损失）、Gumbel-Softmax（可微路由）

## 评分

- 新颖性: ⭐⭐⭐⭐ — BBC Encoder 和 Cus-MOE 的设计在 SVS 领域具有原创性，三模态对比学习统一风格空间是有意义的探索
- 实验充分度: ⭐⭐⭐⭐ — 4 种任务、9 种语言、主观+客观指标、完整消融实验，评估全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，方法描述详细，图表丰富
- 价值: ⭐⭐⭐⭐ — 首个支持歌声/语音/文本三种提示的多语言零样本 SVS 系统，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Adapting Speech Language Model to Singing Voice Synthesis](../../NeurIPS2025/audio_speech/adapting_speech_language_model_to_singing_voice_synthesis.md)
- [\[AAAI 2026\] HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios](../../AAAI2026/audio_speech/hq-svc_towards_high-quality_zero-shot_singing_voice_conversion_in_low-resource_s.md)
- [\[ACL 2025\] ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control](controlspeech_zero_shot.md)
- [\[ACL 2025\] Zero-Shot Text-to-Speech for Vietnamese](zero-shot_text-to-speech_for_vietnamese.md)
- [\[ACL 2026\] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis](../../ACL2026/audio_speech/restyle-tts_relative_and_continuous_style_control_for_zero-shot_speech_synthesis.md)

</div>

<!-- RELATED:END -->
