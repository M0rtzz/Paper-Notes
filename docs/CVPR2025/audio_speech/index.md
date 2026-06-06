---
title: >-
  CVPR2025 音频/语音方向19篇论文解读
description: >-
  19篇CVPR2025的音频/语音方向论文解读，涵盖语音、多模态、对话系统、情感分析、压缩/编码、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "音频/语音"
  - "论文解读"
  - "论文笔记"
  - "语音"
  - "多模态"
  - "对话系统"
  - "情感分析"
  - "压缩/编码"
  - "LLM"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**📷 CVPR2025** · **19** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (30)](../../ICML2026/audio_speech/index.md) · [💬 ACL2026 (69)](../../ACL2026/audio_speech/index.md) · [📷 CVPR2026 (14)](../../CVPR2026/audio_speech/index.md) · [🔬 ICLR2026 (32)](../../ICLR2026/audio_speech/index.md) · [🤖 AAAI2026 (30)](../../AAAI2026/audio_speech/index.md) · [🧠 NeurIPS2025 (50)](../../NeurIPS2025/audio_speech/index.md)

🔥 **高频主题：** 语音 ×10 · 多模态 ×4

**[Contextual AD Narration with Interleaved Multimodal Sequence](contextual_ad_narration_with_interleaved_multimodal_sequence.md)**

:   提出 Uni-AD 统一框架，以交错多模态序列（视频特征+文本+角色库+上下文）作为输入，通过视觉映射网络对齐特征 + 角色精化模块识别主要角色 + 对比损失增强上下文一致性，在 MAD-eval-Named 上达到 SOTA。

**[Crab: A Unified Audio-Visual Scene Understanding Model with Explicit Cooperation](crab_a_unified_audio-visual_scene_understanding_model_with_explicit_cooperation.md)**

:   提出统一音视频场景理解模型 Crab，通过构建带显式推理过程的 AV-UIE 数据集（200K 样本）阐明跨任务协作关系，结合交互感知 LoRA（多头 LoRA）学习不同音视频交互模式，在多个任务上超越专用模型。

**[DistinctAD: Distinctive Audio Description Generation in Contexts](distinctad_distinctive_audio_description_generation_in_contexts.md)**

:   生成上下文中有区分度的音频描述（AD），避免生成泛化无特色的描述，通过对比学习鼓励与前后AD的差异性

**[DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations](dualtalk_dual-speaker_interaction_for_3d_talking_head_conversations.md)**

:   提出 DualTalk——首个统一建模说话者和倾听者行为的多轮双人交互 3D 说话人头生成框架，配套构建了包含 50 小时、1000+ 身份的双人对话数据集。

**[EMoVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions](emova_empowering_language_models_to_see_hear_and_speak_with_vivid_emotions.md)**

:   提出 EMoVA，首个端到端的全模态 LLM，通过语义-声学解耦的语音 tokenizer 同时实现视觉理解、语音识别和情感可控的语音合成，在视觉语言基准上超越 GPT-4o，语音识别 WER 达 2.9%。

**[Hearing Anywhere in Any Environment](hearing_anywhere_in_any_environment.md)**

:   提出 xRIR，一个可跨房间泛化的声脉冲响应（RIR）预测统一模型，结合全景深度图的几何特征提取器和少量参考 RIR 的声学编码器，配合新构建的 AcousticRooms 数据集（260 个房间、30 万+ RIR），在已见/未见模拟环境和真实环境中均大幅超越基线方法。

**[HOP: Heterogeneous Topology-based Multimodal Entanglement for Co-Speech Gesture Generation](hop_heterogeneous_topology-based_multimodal_entanglement_for_co-speech_gesture_g.md)**

:   本文提出 HOP，一种基于异构拓扑的多模态纠缠方法，通过将音频作为桥梁，利用重编程模块对齐音频-文本语义、利用时空图网络对齐音频-动作节奏，实现更自然连贯的语音伴随手势生成，在 FGD、BC 和多样性指标上达到 SOTA。

**[Improving Sound Source Localization with Joint Slot Attention on Image and Audio](improving_sound_source_localization_with_joint_slot_attention_on_image_and_audio.md)**

:   提出联合槽注意力机制将图像和音频同时分解为目标/非目标表示，通过跨模态注意力匹配和对比学习实现精确声源定位，在 Flickr-SoundNet 上达到 65.16% AUC、86.00% cIoU SOTA。

**[ImViD: Immersive Volumetric Videos for Enhanced VR Engagement](imvid_immersive_volumetric_videos_for_enhanced_vr_engagement.md)**

:   构建首个沉浸式体积视频数据集——用 46 台同步 GoPro 的移动多视角系统拍摄 7 个场景（含室内/室外），提出 STG++ 增加可学习仿射颜色变换解决跨相机颜色不一致，实现 110.47 FPS 渲染/387MB 存储，并集成 HRTF 空间音频。

**[Learning-enabled Polynomial Lyapunov Function Synthesis via High-Accuracy Counterexample-Guided Framework](learning-enabled_polynomial_lyapunov_function_synthesis_via_high-accuracy_counte.md)**

:   提出一种学习与验证结合的多项式 Lyapunov 函数合成方法，通过数据驱动的机器学习引导多项式形式选择，并利用高精度反例引导框架迭代优化，在灵活性和数学严格性之间取得平衡。

**[Learning to Highlight Audio by Watching Movies](learning_to_highlight_audio_by_watching_movies.md)**

:   提出视觉引导的声学高亮任务(visually-guided acoustic highlighting)，利用电影中精心制作的音视频数据作为免费监督，通过基于Transformer的多模态框架VisAH，将"混音不佳"的音频转换为视觉语义对齐的高亮音频，在所有指标上显著超越基线方法。

**[LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale](livecc_learning_video_llm_with_streaming_speech_transcription_at_scale.md)**

:   提出 LiveCC，通过将 ASR 转录词与视频帧沿时间轴密集交织训练视频 LLM，构建了 Live-CC-5M 预训练数据集，使 7B 模型在实时视频解说任务上超越 72B 模型（包括 Qwen2.5-VL-72B）。

**[Object-aware Sound Source Localization via Audio-Visual Scene Understanding](object-aware_sound_source_localization_via_audio-visual_scene_understanding.md)**

:   本文提出 OA-SSL：在训练阶段用 MLLM 为每张图生成"K 个发声物 + 1 个静音物"的细粒度描述作为额外监督锚点，再用 OCA (object-aware contrastive alignment) 和 ORI (object region isolation) 两个损失，让模型即使在画面里有多把吉他、只有一把在弹的复杂场景下也能只定位真正在发声的物体。

**[Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach](team_ras_in_10th_abaw_competition_multimodal_valence_and_arousal_estimation_appr.md)**

:   提出结合面部（GRADA+Transformer）、行为描述（Qwen3-VL+Mamba）和音频（WavLM）三模态的连续情感估计方法，通过 Directed Cross-Modal MoE 和 Reliability-Aware Audio-Visual 两种融合策略在 Aff-Wild2 上达到 CCC 0.6576（dev）/ 0.62（test）。

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
