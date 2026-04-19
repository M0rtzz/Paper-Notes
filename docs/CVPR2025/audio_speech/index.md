---
title: >-
  CVPR2025 音频/语音方向 15篇论文解读
description: >-
  15篇CVPR2025 音频/语音方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**📷 CVPR2025** · **15** 篇论文解读

**[Contextual AD Narration with Interleaved Multimodal Sequence](contextual_ad_narration_with_interleaved_multimodal_sequence.md)**

:   提出 Uni-AD 统一框架，以交错多模态序列（视频特征+文本+角色库+上下文）作为输入，通过视觉映射网络对齐特征 + 角色精化模块识别主要角色 + 对比损失增强上下文一致性，在 MAD-eval-Named 上达到 SOTA。

**[Crab: A Unified Audio-Visual Scene Understanding Model with Explicit Cooperation](crab_a_unified_audio-visual_scene_understanding_model_with_explicit_cooperation.md)**

:   提出统一音视频场景理解模型 Crab，通过构建带显式推理过程的 AV-UIE 数据集（200K 样本）阐明跨任务协作关系，结合交互感知 LoRA（多头 LoRA）学习不同音视频交互模式，在多个任务上超越专用模型。

**[DistinctAD: Distinctive Audio Description Generation in Contexts](distinctad_distinctive_audio_description_generation_in_contexts.md)**

:   生成上下文中有区分度的音频描述（AD），避免生成泛化无特色的描述，通过对比学习鼓励与前后AD的差异性

**[EMoVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions](emova_empowering_language_models_to_see_hear_and_speak_with_vivid_emotions.md)**

:   提出 EMoVA，首个端到端的全模态 LLM，通过语义-声学解耦的语音 tokenizer 同时实现视觉理解、语音识别和情感可控的语音合成，在视觉语言基准上超越 GPT-4o，语音识别 WER 达 2.9%。

**[Exploring Timeline Control for Facial Motion Generation](exploring_timeline_control_for_facial_motion_generation.md)**

:   本文首次提出面部动作生成的时间线控制方式——用户指定多轨道时间轴上各面部动作的精确帧区间，通过TICC时序聚类实现省力的帧级面部动作标注，并设计base-branch扩散模型在解耦各面部区域的同时保留自然耦合，生成精确对齐时间线且自然流畅的面部动作。

**[Improving Sound Source Localization with Joint Slot Attention on Image and Audio](improving_sound_source_localization_with_joint_slot_attention_on_image_and_audio.md)**

:   提出联合槽注意力机制将图像和音频同时分解为目标/非目标表示，通过跨模态注意力匹配和对比学习实现精确声源定位，在 Flickr-SoundNet 上达到 65.16% AUC、86.00% cIoU SOTA。

**[ImViD: Immersive Volumetric Videos for Enhanced VR Engagement](imvid_immersive_volumetric_videos_for_enhanced_vr_engagement.md)**

:   构建首个沉浸式体积视频数据集——用 46 台同步 GoPro 的移动多视角系统拍摄 7 个场景（含室内/室外），提出 STG++ 增加可学习仿射颜色变换解决跨相机颜色不一致，实现 110.47 FPS 渲染/387MB 存储，并集成 HRTF 空间音频。

**[Learning to Highlight Audio by Watching Movies](learning_to_highlight_audio_by_watching_movies.md)**

:   提出视觉引导的声学高亮任务(visually-guided acoustic highlighting)，利用电影中精心制作的音视频数据作为免费监督，通过基于Transformer的多模态框架VisAH，将"混音不佳"的音频转换为视觉语义对齐的高亮音频，在所有指标上显著超越基线方法。

**[LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale](livecc_learning_video_llm_with_streaming_speech_transcription_at_scale.md)**

:   提出 LiveCC，通过将 ASR 转录词与视频帧沿时间轴密集交织训练视频 LLM，构建了 Live-CC-5M 预训练数据集，使 7B 模型在实时视频解说任务上超越 72B 模型（包括 Qwen2.5-VL-72B）。

**[Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](team_leya_in_10th_abaw_competition_multimodal_ambivalencehesitancy_recognition_a.md)**

:   本文提出面向视频级矛盾/犹豫（A/H）识别的多模态方法，整合场景（VideoMAE）、面部（EmotionEfficientNetB0）、音频（EmotionWav2Vec2.0+Mamba）和文本（EmotionDistilRoBERTa）四种模态，通过原型增强的 Transformer 融合模型实现 83.25% 平均 MF1，最终以五模型集成在测试集达到 71.43%。

**[Towards Lossless Implicit Neural Representation via Bit Plane Decomposition](towards_lossless_implicit_neural_representation_via_bit_plane_decomposition.md)**

:   发现隐式神经表示（INR）的模型容量上界随比特精度指数增长（$\mathcal{P}(f_\theta) \propto 2^n$），提出比特平面分解——将 n-bit 信号分解为 n 个独立的 1-bit 平面分别训练 INR，首次实现 16-bit 图像的无损（BER=0）隐式神经表示。

**[Towards Open-Vocabulary Audio-Visual Event Localization](towards_open-vocabulary_audio-visual_event_localization.md)**

:   首次定义开放词汇音视频事件定位（OV-AVEL）任务，构建了包含 24800 个视频、67 类事件的 OV-AVEBench 基准，并提出基于 ImageBind 的训练免和微调两种基线方法，其中仅用 1 层时序 Transformer 微调即达 57.8% 平均性能。

**[UWAV: Uncertainty-Weighted Weakly-Supervised Audio-Visual Video Parsing](uwav_uncertainty-weighted_weakly-supervised_audio-visual_video_parsing.md)**

:   提出 UWAV，一个弱监督音视频视频解析框架，通过在大规模标注数据上预训练时序感知模块生成高质量伪标签，再用不确定性加权软标签+类别平衡重加权+特征混合三种技术提升弱监督训练效果，在 LLP 数据集上刷新 SOTA。

**[MultiFoley: Video-Guided Foley Sound Generation with Multimodal Controls](video-guided_foley_sound_generation_with_multimodal_controls.md)**

:   提出 MultiFoley，基于 Diffusion Transformer 的视频引导 Foley 音效生成系统，支持文本语义控制和参考音频风格控制，通过联合训练视频-音频和文本-音频数据集实现 48kHz 高质量音频生成，在人类评估中以 90% 胜率碾压现有方法。

**[VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation](vintage_joint_video_and_text_conditioning_for_holistic_audio_generation.md)**

:   提出 VinTAGe，首个联合视频+文本条件的音频生成模型，通过可学习层权重平衡视觉/文本引导，用教师-学生框架缓解模态偏置，在画内音和画外音生成上实现全面最优（FAD 3.05，MOS 3.36）。
