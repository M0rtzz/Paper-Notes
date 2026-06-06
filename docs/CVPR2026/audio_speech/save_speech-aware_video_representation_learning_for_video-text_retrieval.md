---
title: >-
  [论文解读] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval
description: >-
  [音频/语音] 提出 SAVE 方法，通过添加专用语音分支（Whisper ASR + CLIP 文本编码器）和 soft-ALBEF 视觉-音频早期对齐策略，实现语音感知的视频表示学习，在五个视频-文本检索基准上全面超越 SOTA。
tags:
  - "音频/语音"
---

# SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval

| 信息 | 内容 |
|------|------|
| **会议** | CVPR 2026 |
| **arXiv** | [2603.08224](https://arxiv.org/abs/2603.08224) |
| **领域** | 人体理解 |
| **关键词** | 视频-文本检索, 语音感知, 音视频融合, soft-ALBEF, 多模态学习 |

## 一句话总结

提出 SAVE 方法，通过添加专用语音分支（Whisper ASR + CLIP 文本编码器）和 soft-ALBEF 视觉-音频早期对齐策略，实现语音感知的视频表示学习，在五个视频-文本检索基准上全面超越 SOTA。

## 研究背景与动机

视频-文本检索（VTR）领域普遍采用 CLIP 作为基础，但由于 CLIP 仅提供图像和文本编码器，现有方法自然忽略了视频的声音轨道。近期音视觉方法（EclipSE、TEFAL、AVIGATE）引入音频编码器，但存在两个关键问题：

**音频编码器无法有效表征语音内容**：现有音频编码器（ResNet-18、AST）是在环境声音数据集上训练的，对语音语义的编码效果很差。作者通过一个实验证明：在 AST 的特征空间中，不同类别的语音样本完全混杂在一起，无法区分

**视觉-音频融合前缺乏对齐**：视觉特征（CLIP 图像编码器）和音频特征（AST）从未经过预对齐，直接融合效果受限。虽然 ALBEF（先对齐再融合）已在视觉-语言预训练中成功，但视频-音频对往往缺乏语义对应关系（如背景音乐与视频内容无关），直接套用 hard ALBEF 会引入虚假关联

## 方法详解

### 整体框架：三分支网络

SAVE 在 AVIGATE 的双分支（视觉 + 音频）基础上扩展为三分支：

1. **视觉分支**：CLIP ViT-B/32 提取帧特征 $\{v_i\}$
2. **音频分支**：AST（冻结）提取音频 token，经 Resampler 后通过 Gated-Fusion 与视觉 token 融合得到 $\{\hat{a}_i\}$
3. **语音分支（新增）**：Whisper large-v3 → ASR 文本 → CLIP 文本编码器 → 语音 token $\{s_i\}$，再通过 Gated-Fusion 得到 $\{\hat{s}_i\}$

最终语音感知的视频表示：$\{\tilde{v}_i\} = \{v_i\} + (\{\hat{a}_i\} + \{\hat{s}_i\})/2$

设计动机：视觉为主（权重较大），语音和音频等权（缺乏先验），简单融合鼓励 Gate-Fusion 学习真正重要的信号。

### Soft-ALBEF 早期对齐

关键创新：用 ImageBind 计算视频-音频亲和力矩阵 $M_0$ 作为软标签，替代 ALBEF 的 hard label。

$$\ell_{\text{pearson}} = \frac{1}{b}\sum_{i=1}^{b} d_p(\sigma(M_0[i,\cdot]), \sigma(M_1[i,\cdot])) + \frac{1}{b}\sum_{j=1}^{b} d_p(\sigma(M_0[\cdot,j]), \sigma(M_1[\cdot,j]))$$

其中 $M_1$ 是当前网络生成的亲和力矩阵，$d_p$ 为 Pearson 距离。选择 Pearson 距离而非 MSE/Huber 是因为其对尺度和位移变化的不变性，使网络专注学习相对排序结构。

### 缺失数据处理

- 无声音轨道：Mel 滤波器组设为零
- ASR 识别失败：使用空字符串，tokenizer 填充为零向量

### 训练细节

- Pearson 距离损失作为辅助目标与 AVIGATE 的自适应边距对比损失等权结合
- CLIP 主干微调学习率 1e-7，其他模块 1e-4（防止灾难性遗忘）
- 8× RTX 3090 GPU

## 实验关键数据

### 主实验：文本到视频检索 SumR

| 方法 | MSRVTT-9k | MSRVTT-7k | VATEX | Charades | LSMDC | mR1 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| CLIP4Clip | 197.5 | 150.1 | 248.5 | 107.6 | 112.7 | 35.1 |
| PIG | 203.0 | 157.1 | 252.1 | - | - | - |
| AVIGATE | 207.7 | 162.7 | 249.3 | 110.6 | 125.7 | 37.9 |
| **SAVE** | **216.2** | **165.8** | **255.5** | **121.4** | **128.3** | **39.6** |

SAVE 相比 AVIGATE 的 SumR 提升：MSRVTT-9k +8.5, VATEX +6.2, Charades +10.8。

### 分组分析（MSRVTT-9k）

| 组别 | SAVE vs AVIGATE SumR差 |
|------|:---:|
| 视觉相关 (499例) | 正提升 |
| 声音相关 (226例) | +11.5 |
| 语音相关 (171例) | +12.9 |
| 声音+语音相关 (104例) | **+16.4** |

### 效率分析

| 方法 | 计算复杂度 | 推理时间 | SumR |
|------|:---:|:---:|:---:|
| TEFAL | $O(n_{\mathcal{A}} n_{\mathcal{T}} + n_{\mathcal{V}} n_{\mathcal{T}})$ | 140.57ms | 209.2 |
| AVIGATE | $O(n_{\mathcal{A}} + n_{\mathcal{V}} + n_{\mathcal{T}})$ | 9.90ms | 207.7 |
| **SAVE** | $O(n_{\mathcal{S}} + n_{\mathcal{A}} + n_{\mathcal{V}} + n_{\mathcal{T}})$ | **9.90ms** | **216.2** |

SAVE 保持与 AVIGATE 相同的推理延迟（9.90ms），因为视频特征可离线提取。

### 消融：语音分支 vs 音频分支

- 去掉语音分支：SumR -4.3
- 去掉音频分支：SumR -8.7
- 两者均有贡献，音频分支影响更大因数据集中声音相关查询更多

## 亮点与洞察

1. **问题洞察精准**：通过 toy 实验直接展示 AST 在语音特征空间中的聚类失败，动机非常有说服力
2. **语音分支设计优雅**：Whisper ASR → CLIP 文本编码器的流水线巧妙利用了 CLIP 的文本-视觉对齐能力来编码语音
3. **soft-ALBEF 通用性强**：用 ImageBind 提供噪声容忍的软监督信号，解决了视觉-音频对缺乏对应关系的根本问题
4. **零额外推理成本**：所有新增计算可离线完成
5. **Charades 上的惊人提升**：即使仅 13.5% 视频有 ASR 文本，SumR 仍提升 10.8，说明 soft-ALBEF 有效利用了声音模态

## 局限性

- 仅在短视频片段上验证，长视频（如电商直播）中 ASR 文本通常更长更噪
- 依赖 Whisper 的 ASR 质量，非英语语言场景可能效果不同
- 使用 ViT-B/32，未探索更大骨干（受 GPU 预算限制）
- ImageBind 用于 soft-ALBEF 引入额外离线计算成本
- 对于完全无声视频的提升空间有限

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 实验 | ⭐⭐⭐⭐⭐ |
| 写作 | ⭐⭐⭐⭐⭐ |
| 综合价值 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] LeVo: High-Quality Song Generation with Multi-Preference Alignment](../../NeurIPS2025/audio_speech/levo_high-quality_song_generation_with_multi-processing_refined_supervision.md)
- [\[CVPR 2025\] DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations](../../CVPR2025/audio_speech/dualtalk_dual-speaker_interaction_for_3d_talking_head_conversations.md)
- [\[CVPR 2026\] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)
- [\[ACL 2026\] MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation](../../ACL2026/audio_speech/marquis_a_three-stage_pipeline_for_video_retrieval-augmented_generation.md)
- [\[ACL 2026\] ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment](../../ACL2026/audio_speech/immersivetts_environment-aware_text-to-speech_with_multimodal_diffusion_transfor.md)

</div>

<!-- RELATED:END -->
