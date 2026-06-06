---
title: >-
  [论文解读] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text
description: >-
  [CVPR 2026][音频/语音][视频到音频生成] 提出 Universal Holistic Audio Generation (UniHAGen) 任务和 OmniSonic 框架，通过 TriAttn-DiT 架构的三路交叉注意力和 MoE 门控机制…
tags:
  - "CVPR 2026"
  - "音频/语音"
  - "视频到音频生成"
  - "全景音频"
  - "扩散模型"
  - "语音合成"
  - "混合专家"
---

# OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text

**会议**: CVPR 2026  
**arXiv**: [2604.04348](https://arxiv.org/abs/2604.04348)  
**代码**: [https://weiguopian.github.io/OmniSonic_webpage/](https://weiguopian.github.io/OmniSonic_webpage/)  
**领域**: 音频生成 / 多模态  
**关键词**: 视频到音频生成, 全景音频, 扩散模型, 语音合成, 混合专家

## 一句话总结

提出 Universal Holistic Audio Generation (UniHAGen) 任务和 OmniSonic 框架，通过 TriAttn-DiT 架构的三路交叉注意力和 MoE 门控机制，首次实现同时生成屏幕内/屏外环境声和人声的统一音频合成，在新构建的 UniHAGen-Bench 上全面超越 SOTA。

## 研究背景与动机

1. **领域现状**：扩散模型在音频生成领域取得显著进展，V2A（视频到音频）方法如 Diff-Foley、MMAudio 生成的音频在质量和语义对齐上不断提升。联合文本-视频到音频（VT2A）方法如 VinTAGe 开始同时考虑屏幕内外的声音。

2. **现有痛点**：（1）V2A 模型仅能生成画面中可见事件对应的声音，忽略屏幕外的听觉事件；（2）VT2A 模型虽然考虑了屏幕内外声音，但仅限于环境声，无法生成人类语音；（3）环境语音生成模型（如 VoiceLDM）仅依赖文本输入，缺乏视觉 grounding。

3. **核心矛盾**：真实世界的听觉场景是复杂的——一个说话的人前面有鸟叫声，或者背景中有机器声。现有模型无法在一个统一框架中处理"环境声 + 人声 + 屏幕内/外"的全排列组合。

4. **本文目标** 定义一个新任务 UniHAGen，要求模型同时生成屏幕内环境声、屏幕外环境声和人类语音三种声源的混合音频。

5. **切入角度**：将问题拆解为三路条件（屏幕内环境描述、屏幕外环境描述、语音转录），设计专门的三路交叉注意力机制分别处理，再用 MoE 门控动态融合。

6. **核心 idea**：用 TriAttn-DiT 三路交叉注意力分别处理屏幕内环境声、屏幕外环境声和语音条件，通过 MoE 门控自适应平衡三路贡献，实现全景音频生成。

## 方法详解

### 整体框架

OmniSonic 基于 Flow Matching 扩散框架，在音频 VAE 的潜空间中进行去噪。输入条件包括四部分：视频帧（CLIP 视觉编码器）、屏幕内环境声描述（FLAN-T5）、屏幕外环境声描述（FLAN-T5）、语音转录（SpeechT5 + Durator）。核心是 TriAttn-DiT 模块，堆叠多个 block 预测音频潜空间的速度场。推理时通过 ODE 求解从噪声生成音频潜表示，再经 VAE 解码器和 HiFi-GAN 声码器恢复波形。

### 关键设计

1. **TriAttn-DiT 三路交叉注意力**:

    - 功能：分别处理三种条件信号（屏幕内环境、屏幕外环境、语音）与音频潜表示的交互
    - 核心思路：视觉特征 $\mathbf{c}_v$ 按屏幕内环境声描述是否为空，选择性地与对应条件拼接——若描述非空则视觉与屏幕内环境拼接，否则与语音拼接。三路独立进行交叉注意力：$\mathbf{x}_t^{on} = \text{CA}_{env}(\text{RoPE}(\mathbf{x}_t), \text{RoPE}(\mathbf{c}^{on}_{txt,v}[L_{on}:,:]), \mathbf{c}^{on}_{txt,v})$，类似地处理 off-screen 和 speech。RoPE 仅应用于视觉 token 部分以编码时间位置信息。
    - 设计动机：环境声和语音的声学特性差异巨大，共享注意力层会导致相互干扰；分离处理后各路可以专注于各自的语义对齐

2. **MoE 门控融合机制**:

    - 功能：自适应平衡三路交叉注意力输出的贡献权重
    - 核心思路：对三路条件 embedding 取序列维度均值得到代表 token，拼接后经 MLP + Softmax 得到三个归一化权重 $[\omega^{sp}, \omega^{on}, \omega^{off}]$，加权求和得到最终速度预测：$\mathbf{v}_t = \omega^{sp}\mathbf{x}_t^{sp} + \omega^{on}\mathbf{x}_t^{on} + \omega^{off}\mathbf{x}_t^{off}$
    - 设计动机：不同场景中三种声源的重要性不同（如纯环境声场景 vs 语音主导场景），静态权重无法适应这种变化

3. **Frame-Aligned Adaptive Layer Normalization**:

    - 功能：增强生成音频与视频帧的时间对齐
    - 核心思路：将视觉条件 $\mathbf{c}_v$ 投影到与时间步 embedding 相同的空间并相加得到 $\mathbf{c}_{vt}$，通过最近邻插值上采样到音频时间分辨率，生成逐帧的 adaLN 参数 $[\alpha_1, \beta_1, \gamma_1, \alpha_2, \beta_2, \gamma_2]$
    - 设计动机：逐帧调制确保音频特征与对应视频帧精确对齐，增强时间同步性

### 损失函数 / 训练策略

使用 Flow Matching 目标函数：$\mathcal{L}_{FM} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}[\|\mathcal{V}_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0)\|_2^2]$

训练数据从 VGGSound（~195K 环境声）、LRS3（~33K 语音视频）和 CommonVoice（~1.67M 语音）合成，按随机 SNR 混合。FLAN-T5 和 CLIP 视觉编码器冻结，SpeechT5 和 Durator 可训练。

## 实验关键数据

### 主实验

在 UniHAGen-Bench（1003 样本，3种场景）上的客观评估：

| 方法 | FAD↓ | MKL↓ | Mean(AT+AV)↑ | WER↓ | DeSync↓ |
|------|------|------|-------------|------|---------|
| VoiceLDM | 3.58 | 5.74 | 14.03 | 0.15 | 1.25 |
| MMAudio | 5.82 | 5.60 | 17.25 | 1.50 | **0.51** |
| HunyuanVideo-Foley | 6.00 | 5.88 | 16.95 | 1.36 | 0.38 |
| **OmniSonic** | **3.07** | **2.79** | **18.54** | **0.14** | 0.72 |

主观评估 MOS 评分：

| 方法 | MOS-Q↑ | MOS-EF↑ | MOS-SF↑ | MOS-T↑ |
|------|--------|---------|---------|--------|
| VoiceLDM | 3.13 | 3.40 | 4.05 | 2.54 |
| MMAudio | 3.74 | 3.24 | 1.15 | 3.71 |
| **OmniSonic** | **4.35** | **4.42** | **4.74** | **4.29** |

### 消融实验

| 配置 | FAD↓ | Mean↑ | WER↓ | DeSync↓ |
|------|------|-------|------|---------|
| OmniSonic (完整) | 3.07 | 18.54 | 0.14 | 0.72 |
| w/o MoE Gating | 6.12 | 15.94 | 0.56 | 1.23 |

### 关键发现

- 移除 MoE 门控后 FAD 从 3.07 翻倍到 6.12，WER 从 0.14 增到 0.56（4倍），证明门控机制对多源平衡至关重要
- OmniSonic 在 DeSync 上不如 MMAudio 和 HunyuanVideo-Foley，因为后两者使用了 Synchformer 提取的时间细粒度视觉特征，而 OmniSonic 仅用 CLIP 特征
- MMAudio、HunyuanVideo-Foley 的 MOS-SF（语音保真度）仅 1.15/1.17，几乎无法生成语音；VoiceLDM 语音好但环境声差（MOS-EF 3.40）
- 手动抑制 MoE 各分支的可视化分析表明，抑制语音分支则无法生成语音，抑制环境分支则失去背景声，验证了各分支的功能专一性

## 亮点与洞察

- **任务定义有前瞻性**：UniHAGen 定义了三种"屏幕内/外×环境声/语音"场景，首次将语音纳入全景音频生成的考量范围，填补了重要空白
- **TriAttn-DiT 设计优雅**：三路独立注意力 + 共享MoE门控的架构，既保证各声源条件的独立处理，又实现动态融合，避免了多条件混合带来的干扰
- **视觉-条件动态绑定**：根据屏幕内描述是否为空，决定视觉特征与环境描述还是语音转录绑定——这个简洁的设计巧妙地区分了"画面中是声音事件还是说话的人"
- 这种多路注意力 + MoE 门控的模式可以迁移到其他需要处理多种异构条件的生成任务

## 局限与展望

- 时间同步性（DeSync）不如使用 Synchformer 的方法，引入更细粒度的时间视觉特征可能改善
- 训练数据是合成 mixture，真实场景中声源的空间分布和混响效果未被建模
- 仅支持 10 秒音频生成，更长音频的连贯性未验证
- 语音质量虽好但未与专门的 TTS 系统做详细对比
- UniHAGen-Bench 仅 1003 样本，评估规模有限

## 相关工作与启发

- **vs MMAudio**: MMAudio 使用多模态 DiT 联合建模视频和文本，但仅针对环境声；OmniSonic 通过三路注意力扩展到语音领域，且环境声质量也更优
- **vs VoiceLDM**: VoiceLDM 是纯文本条件的语音生成，缺乏视觉 grounding；OmniSonic 加入视频条件后可以区分屏幕内外声源
- **vs VinTAGe**: VinTAGe 提出了"全景"音频生成的概念，但局限于环境声；OmniSonic 真正实现了"全景"——覆盖环境声和语音

## 评分

- 新颖性: ⭐⭐⭐⭐ UniHAGen 任务定义和 TriAttn-DiT 架构都很新颖，MoE 门控是锦上添花
- 实验充分度: ⭐⭐⭐⭐ 客观+主观评估全面，消融验证了核心组件，但 benchmark 规模较小
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，方法描述详尽，定性分析丰富
- 价值: ⭐⭐⭐⭐ 填补了音频生成领域"环境声+语音"统一的空白，对影视后期制作有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation](../../CVPR2025/audio_speech/vintage_joint_video_and_text_conditioning_for_holistic_audio_generation.md)
- [\[CVPR 2026\] Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models](echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](../../NeurIPS2025/audio_speech/node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)
- [\[CVPR 2026\] ViDscribe: Multimodal AI for Customizing Audio Description and Question Answering in Online Videos](vidscribe_multimodal_ai_for_customizing_audio_description_and_question_answering.md)

</div>

<!-- RELATED:END -->
