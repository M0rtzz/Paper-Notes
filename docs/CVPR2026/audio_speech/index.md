---
title: >-
  CVPR2026 音频/语音方向15篇论文解读
description: >-
  15篇CVPR2026的音频/语音方向论文解读，涵盖语音、多模态、导航、情感分析、个性化生成、问答等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "音频/语音"
  - "论文解读"
  - "论文笔记"
  - "语音"
  - "多模态"
  - "导航"
  - "情感分析"
  - "个性化生成"
  - "问答"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**📷 CVPR2026** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (7)](../../ICML2026/audio_speech/index.md) · [💬 ACL2026 (47)](../../ACL2026/audio_speech/index.md) · [🔬 ICLR2026 (32)](../../ICLR2026/audio_speech/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/audio_speech/index.md) · [🧠 NeurIPS2025 (50)](../../NeurIPS2025/audio_speech/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/audio_speech/index.md)

🔥 **高频主题：** 语音 ×6 · 多模态 ×5

**[BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models](babyvlm-v2_toward_developmentally_grounded_pretraining_and_benchmarking_of_visio.md)**

:   提出BabyVLM-V2框架，从婴儿第一视角的SAYCam纵向语料构建三种格式预训练数据（768K图像对+181K视频对+63K交错序列），设计基于NIH Baby Toolbox®的DevCV Toolbox（10个发育认知任务），从零训练的紧凑模型在部分数学任务上超越GPT-4o，首次系统探索人工发育智能(ADI)。

**[Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning](cleaning_the_pool_progressive_filtering_of_unlabeled_pools_in_deep_active_learni.md)**

:   提出 Refine 集成主动学习方法，通过两阶段策略——渐进过滤（多策略迭代精炼无标签池）+ 覆盖选择（从精炼池中选择多样性高价值样本）——在不预知最佳策略的情况下一致超越单一 AL 策略和现有集成方法。

**[Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models](echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo.md)**

:   提出 MMHNet，一种基于层级结构和非因果 Mamba-2 的多模态层级网络，实现了在短片段（8秒）上训练、在长视频（5分钟以上）上生成高质量对齐音频的长度泛化能力，在 UnAV100 和 LongVale 基准上大幅超越现有方法。

**[GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization](gem-tfl_bridging_weak_and_full_supervision_for_forgery_localization_through_em-g.md)**

:   提出 GEM-TFL，通过两阶段分类-回归框架弥合弱监督与全监督之间的差距，用 EM 分解二元标签为多维潜在属性、训练无关的时序一致性精化、图扩散提案精化三大模块，在弱监督时序伪造定位上平均 mAP 提升 4-8%。

**[Omni-MMSI: Toward Identity-Attributed Social Interaction Understanding](omni-mmsi_toward_identity-attributed_social_interaction_understanding.md)**

:   提出 Omni-MMSI 任务——从原始音视频输入（而非预处理的 oracle 社交线索）理解多人社交交互，并设计 Omni-MMSI-R 参考引导流水线，通过工具生成身份归因社交线索 + 链式思维推理实现准确的社交交互理解。

**[OmniRet: Efficient and High-Fidelity Omni Modality Retrieval](omniret_efficient_and_high-fidelity_omni_modality_retrieval.md)**

:   提出首个支持文本-视觉-音频三模态组合查询的统一检索模型 OmniRet，通过共享媒体重采样器（Shared Media Resampler）提升计算效率，并引入注意力切片 Wasserstein 池化（ASWP）保留细粒度信息，在 13 个检索任务上取得 12 项领先。

**[OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)**

:   提出 Universal Holistic Audio Generation (UniHAGen) 任务和 OmniSonic 框架，通过 TriAttn-DiT 架构的三路交叉注意力和 MoE 门控机制，首次实现同时生成屏幕内/屏外环境声和人声的统一音频合成，在新构建的 UniHAGen-Bench 上全面超越 SOTA。

**[SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](save_speech-aware_video_representation_learning_for_video-text_retrieval.md)**

:   提出 SAVE 方法，通过添加专用语音分支（Whisper ASR + CLIP 文本编码器）和 soft-ALBEF 视觉-音频早期对齐策略，实现语音感知的视频表示学习，在五个视频-文本检索基准上全面超越 SOTA。

**[Semantic Audio-Visual Navigation in Continuous Environments](semantic_audio-visual_navigation_in_continuous_environments.md)**

:   本文提出 SAVN-CE 任务，将语义音视觉导航扩展到连续3D环境中，并设计 MAGNet（记忆增强目标描述网络），通过融合历史上下文和自运动线索实现在目标声音消失后的稳健目标推理，成功率绝对提升最高达 12.1%。

**[Solution for 10th Competition on Ambivalence/Hesitancy (AH) Video Recognition Challenge using Divergence-Based Multimodal Fusion](solution_for_10th_competition_on_ambivalencehesitancy_ah_video_recognition_chall.md)**

:   针对第10届 ABAW 竞赛的矛盾/犹豫 (A/H) 视频识别任务，提出基于散度的多模态融合策略，通过计算视觉（AU）、音频（Wav2Vec 2.0）和文本（BERT）三个模态嵌入的逐对绝对差来显式建模跨模态冲突，在 BAH 数据集上以 Macro F1 0.6808 大幅超越基线 0.2827。

**[Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach](team_ras_in_10th_abaw_competition_multimodal_valen.md)**

:   首次将 VLM（Qwen3-VL-4B-Instruct）提取的情感行为描述嵌入作为独立第三模态，与 GRADA 人脸编码器和 WavLM 音频特征通过 DCMMOE 和 RAAV 两种融合策略组合，在 Aff-Wild2 上达到连续 VA 估计 CCC 0.658（dev）/ 0.62（test），验证了 VLM 行为语义对连续情感识别的价值。

**[Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)**

:   提出 TSD 框架，将多模态特征显式分解为全局共享/成对共享/模态专属三个互补子空间，并通过子空间感知跨注意力融合模块自适应整合三层信息，在 CMU-MOSI/MOSEI 上全面 SOTA。

**[UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark](unim_a_unified_any-to-any_interleaved_multimodal_benchmark.md)**

:   提出首个统一的任意到任意交错多模态基准 UniM（31K 样本、7 种模态、30 个领域），配套三维评估体系和基于可追溯推理的智能体基线 UniMA，揭示现有 MLLM 在交错多模态范式下的严重不足。

**[Unlocking Strong Supervision: A Data-Centric Study of General-Purpose Audio Pre-Training Methods](unlocking_strong_supervision_a_data-centric_study_of_general-purpose_audio_pre-t.md)**

:   本文通过系统的数据中心实验证明音频预训练性能主要由标签/监督质量驱动而非模型设计，提出 Unified Tag System (UTS) 将语音、音乐、环境音统一到 800-3k 标签的高粒度词表中，UTS 训练的模型用 5 倍更少的数据在语音（VoxCeleb2）和音乐（MusicCaps）等域外任务上超越 AudioSet 基线。

**[ViDscribe: Multimodal AI for Customizing Audio Description and Question Answering in Online Videos](vidscribe_multimodal_ai_for_customizing_audio_description_and_question_answering.md)**

:   ViDscribe 是一个基于 Web 的平台，利用多模态大语言模型(Gemini 3 Pro)为盲人和低视力(BLV)用户提供可定制的 AI 生成音频描述(AD)和交互式视觉问答(VQA)功能，支持任意 YouTube 视频，通过为期一周的纵向用户研究验证了定制化 AD 在有效性、享受度和沉浸感方面均优于默认 AD。
