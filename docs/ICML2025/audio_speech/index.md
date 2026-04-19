---
title: >-
  ICML2025 音频/语音方向 6篇论文解读
description: >-
  6篇ICML2025 音频/语音方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**🧪 ICML2025** · **6** 篇论文解读

**[Bridging the Language Gap: Synthetic Voice Diversity via Latent Mixup for Equitable Speech Recognition](bridging_the_language_gap_synthetic_voice_diversity_via_latent_mixup_for_equitab.md)**

:   本文提出 LatentVoiceMix，在语音转换模型 Diff-HierVC 的说话人风格编码器潜在空间中进行 mixup 插值，生成具有新颖声音特征的合成语音数据用于增强 ASR 训练，在低资源语言 Wolof 上取得了优于波形增强、频谱增强和标准语音转换的 WER 改善效果。

**[FLAM: Frame-Wise Language-Audio Modeling](flam_frame-wise_language-audio_modeling.md)**

:   提出 FLAM，一个帧级音频-语言对比模型，通过文本依赖的 logit 偏置校正和百万级合成 SED 数据集，实现开放词汇声音事件的精确时间定位，同时保持全局检索和零样本分类性能。

**[OmniAudio: Generating Spatial Audio from 360-Degree Video](omniaudio_generating_spatial_audio_from_360-degree_video.md)**

:   提出 OmniAudio 框架，首次实现从 360 度全景视频生成 First-order Ambisonics (FOA) 空间音频，通过 coarse-to-fine 自监督预训练和双分支视频编码架构，在自建的 Sphere360 数据集上取得 SOTA 性能。

**[One Wave To Explain Them All: A Unifying Perspective On Feature Attribution](one_wave_to_explain_them_all_a_unifying_perspective_on_feature_attribution.md)**

:   提出 Wavelet Attribution Method (WAM)，将特征归因从像素域迁移到小波域，利用小波系数的空间-尺度局部性为音频、图像、体数据提供统一且更具结构信息的模型解释。

**[Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems](sortformer_a_novel_approach_for_permutation-resolved_speaker_supervision_in_spee.md)**

:   提出 Sortformer——一个基于编码器的说话人日志模型，通过 Sort Loss 按说话人到达时间排序来解决排列问题，替代或辅助传统的排列不变损失（PIL），并设计正弦核函数将说话人标签注入 ASR 编码器，使多说话人 ASR 训练可直接使用标准交叉熵损失，在 LibriSpeechMix 上实现 2-mix/3-mix 相对误差降低 30%/25%。

**[Teaching Physical Awareness to LLMs through Sounds](teaching_physical_awareness_to_llms_through_sounds.md)**

:   提出 ACORN 框架，通过基于物理的声学通道仿真器生成大规模训练数据，配合同时捕获幅度和相位信息的音频编码器，教会 LLM 从声音中理解物理世界现象。
