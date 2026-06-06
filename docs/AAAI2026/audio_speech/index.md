---
title: >-
  AAAI2026 音频/语音方向30篇论文解读
description: >-
  30篇AAAI2026的音频/语音方向论文解读，涵盖语音、情感分析、多模态、对话系统、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "音频/语音"
  - "论文解读"
  - "论文笔记"
  - "语音"
  - "情感分析"
  - "多模态"
  - "对话系统"
  - "LLM"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**🤖 AAAI2026** · **30** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (30)](../../ICML2026/audio_speech/index.md) · [💬 ACL2026 (69)](../../ACL2026/audio_speech/index.md) · [📷 CVPR2026 (14)](../../CVPR2026/audio_speech/index.md) · [🔬 ICLR2026 (32)](../../ICLR2026/audio_speech/index.md) · [🧠 NeurIPS2025 (50)](../../NeurIPS2025/audio_speech/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/audio_speech/index.md)

🔥 **高频主题：** 语音 ×9 · 情感分析 ×7 · 多模态 ×4 · 对话系统 ×2 · LLM ×2

**[A Mind Cannot Be Smeared Across Time](a_mind_cannot_be_smeared_across_time.md)**

:   本文从形式化角度证明，机器是否具有意识不仅取决于计算什么，还取决于何时计算——严格顺序执行的系统不满足意识统一性所需的时间共现（co-instantiation）条件，因此纯软件意识在严格顺序硬件上是不可能的。

**[DeepDebater: A Superpersuasive Autonomous Policy Debating System](a_superpersuasive_autonomous_policy_debating_system.md)**

:   提出DeepDebater，首个能参与并赢得完整美式策略辩论赛（八轮发言+交叉质询）的自主多Agent系统，基于层级式Agent工作流分工完成正方（Advantage）/反方（DA+CP+Kritik）论证构建，以OpenDebateEvidence的300万+张证据卡做检索增强，辅以GPT-4o TTS语音合成和EchoMimic数字人动画，在专家评估中各项指标显著超越人类编写案例（Quality 4.32 vs 3.65），模拟对局胜率达85%。

**[AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions](ahamask_reliable_task_specification_for_large_audio_language.md)**

:   通过对大音频语言模型（LALM）Transformer 骨干中的注意力头进行二值掩码（AHAMask），无需文本指令即可可靠触发特定声学任务功能，同时揭示了 LALM 内部存在"声学功能通路"。

**[Aligning Generative Music AI with Human Preferences: Methods and Challenges](aligning_generative_music_ai_with_human_preferences_methods_and_challenges.md)**

:   综述/立场论文，系统梳理偏好对齐技术在音乐生成中的三条路线——MusicRL（大规模 RLHF，~30 万偏好对）、DiffRhythm+（扩散模型多偏好 DPO）、Text2midi-InferAlign（推理时树搜索，CLAP +29.4%），深入分析音乐领域独有的对齐挑战（多尺度时间连贯性、和声一致性、文化主观性、评估悖论），并给出未来路线图。

**[CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation](ccfqa_a_benchmark_for_cross-lingual_and_cross-modal_speech_and_text_factuality_e.md)**

:   提出 CCFQA——首个覆盖 8 种语言、14,400 条完全平行语音-文本事实问答样本的跨语言跨模态基准，支持 QA/XQA/SQA/XSQA 四种任务设定，系统揭示了现有 MLLM 在语言和模态切换下的事实不一致性；同时提出 LLM-SQA，以英语为桥接语言、仅 5-shot 即实现跨语言语音问答迁移，在 XSQA 上 F1 达 51.4 超越 GPT-4o-mini-Audio（45.7）。

**[Characterizing AI Manipulation Risks in Brazilian YouTube Climate Discourse](characterizing_ai_manipulation_risks_in_brazilian_youtube_climate_discourse.md)**

:   通过心理语言学框架分析巴西 YouTube 上 22.6 万条气候变化视频和 275 万条评论，揭示情感/道德修辞显著驱动用户互动，并展示微调 LLM 可自动生成高互动性的气候否认评论，警示生成式 AI 在舆论操控中的潜在风险。

**[Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)**

:   提出 Cross-Space Synergy（CSS）框架，通过表示空间的协同多项式融合（SPF）和梯度空间的 Pareto 梯度调节器（PGM）双管齐下，同时解决多模态对话情感识别中融合表达力不足和多目标梯度冲突两大难题。

**[DeformTrace: A Deformable State Space Model with Relay Tokens for Temporal Forgery Localization](deformtrace_a_deformable_state_space_model_with_relay_tokens_for_temporal_forger.md)**

:   提出 DeformTrace，将可变形动态感受野和中继令牌机制引入状态空间模型，结合 Transformer 的全局建模与 SSM 的高效推理，实现时序伪造定位的 SOTA 精度与显著效率提升。

**[Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning](do_llms_feel_teaching_emotion_recognition_with_prompts_retrieval_and_curriculum_.md)**

:   提出 PRC-Emo 框架，通过显式/隐式情感提示、专用检索库和课程学习策略三位一体地提升 LLM 在对话情感识别（ERC）任务上的表现，在 IEMOCAP 和 MELD 两个基准上取得 SOTA。

**[DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling](dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)**

:   提出 DualSpeechLM 框架，通过理解驱动语音分词器（USTokenizer）提取高层语义 token 作为 LLM 输入、声学 token 作为输出，在一个端到端框架中同时优化语音理解和生成能力。

**[End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)**

:   提出 CLSR，一种端到端对比式语言-语音检索器，通过将声学表示先转换为 text-like representation 再与文本对齐，高效地从长音频中提取与问题相关的片段，为下游 LALM 的长语音问答提供 RAG 支持。

**[Factor(U,T): Controlling Untrusted AI by Monitoring their Plans](factorut_controlling_untrusted_ai_by_monitoring_their_plans.md)**

:   研究不可信 AI 做任务分解、可信 AI 做执行的 Factor(U,T) 协议安全性，发现监控分解计划的 AUROC 仅 0.52（接近随机），而监控具体代码实现可达 0.96——恶意意图在抽象计划中难以检测但在具体实现中暴露，结论是"结构性预防（可信分解器）优于事后监控"。

**[Generalizing Analogical Inference from Boolean to Continuous Domains](generalizing_analogical_inference_from_boolean_to_continuous_domains.md)**

:   从基础理论层面重新审视类比推理：首先构造反例证明布尔域上经典泛化界失效，然后提出基于参数化广义均值的统一类比推理框架，将离散分类扩展到连续回归域。

**[GOMPSNR: Reflourish the Signal-to-Noise Ratio Metric for Audio Generation Tasks](gompsnr_reflourish_the_signal-to-noise_ratio_metric_for_audio_generation_tasks.md)**

:   通过引入全方位相位导数（omnidirectional phase derivatives）替换瞬时相位来重构 SNR 指标，提出 GOMPSNR 作为更可靠的音频质量评估指标，并衍生出一系列新的损失函数显著提升神经声码器性能。

**[Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR](hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)**

:   MARS 提出多模态检索-选择方法为对话式 LLM-ASR 挑选最相关的历史上下文（而非固定前几句或全部历史），在仅用 1.5K 小时训练数据的情况下超越了用 179K 小时数据训练的 SOTA 系统 TEA-ASLP。

**[HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)**

:   提出 HPSU 基准，包含 20,000+ 中英文专家标注样本和 16 项任务，系统评估 Speech LLM 在真实口语场景下的深层感知与推理能力，发现最强模型（Gemini 2.5 Pro，62.6%）与人类表现（87.3%）仍有巨大差距。

**[Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)**

:   提出 MODS 框架，通过图卷积动态序列压缩（GDC）消除非语言模态冗余，并设计样本级动态主模态选择器（MSelector）和主模态中心交叉注意力（PCCA），实现 MSA 中按样本自适应选择主导模态。

**[Incremental Maintenance of DatalogMTL Materialisations](incremental_maintenance_of_datalogmtl_materialisations.md)**

:   提出 DRed$_{\text{MTL}}$ 算法，将经典 Delete/Rederive 增量维护技术扩展到 DatalogMTL（带度量时序逻辑的 Datalog），通过在周期化物化表示上设计新的 seminaïve 评估算子和周期识别算法，实现高效增量更新，性能可达重新物化的数量级提升。

**[Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition](let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)**

:   通过 MoGE 诊断策略系统发现 MIDIBERT 未有效编码调式-情感关联，提出 MoFi 注入框架通过 FiLM 机制将大调/小调先验注入 MIDIBERT 第 1 层（诊断确定的最弱情感信息层），在 EMOPIA 上准确率 75.2%（+11.8%），VGMIDI 上 59.1%（+11.8%），F1 提升 12.3%/15.5%。

**[Listen Like a Teacher: Mitigating Whisper Hallucinations using Adaptive Layer Attention and Knowledge Distillation](listen_like_a_teacher_mitigating_whisper_hallucinations_using_adaptive_layer_att.md)**

:   提出两阶段框架——自适应层注意力（ALA）融合Whisper编码器多层表示以增强噪声鲁棒性，多目标知识蒸馏（MOKD）将clean teacher的语义和注意力分布对齐到noisy student——在多语言噪声ASR基准上显著降低幻觉率和WER。

**[Modelling the Effects of Hearing Loss on Neural Coding in the Auditory Midbrain with Variational Conditioning](modelling_the_effects_of_hearing_loss_on_neural_coding_in_the_auditory_midbrain_.md)**

:   提出 ψ-ICNet，一种变分条件深度神经网络模型，通过仅 6 个可学习的条件参数 ψ 来编码听力损失的效应，从真实神经活动记录中直接学习听力损失的低维表示空间，在预测正常和听力受损动物的听觉中脑神经响应方面达到与动物特定模型相当的精度，并可通过贝叶斯优化快速拟合未见过的新动物。

**[Multi-granularity Interactive Attention Framework for Residual Hierarchical Pronunciation Assessment](multi-granularity_interactive_attention_framework_for_residual_hierarchical_pron.md)**

:   提出HIA框架，通过交互注意力模块（Interactive Attention Module）实现音素、词、句三粒度间的双向信息交互，结合残差层级结构缓解特征遗忘问题，在speechocean762数据集上所有粒度和方面指标均达到SOTA。

**[PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis](pase_prototype-aligned_calibration_and_shapley-based_equilibrium_for_multimodal_.md)**

:   提出 PaSE 框架，通过原型引导校准对齐（Entropic Optimal Transport）与 Shapley 值梯度调制的双阶段优化策略，显式解决多模态情感分析中的模态竞争问题。

**[PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis](psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)**

:   首次在多模态情感分析（MSA）中引入预训练人格模型提取个性化情感特征，通过人格-情感对比学习对齐和多层（预融合→交叉模态交互→增强融合）渐进融合架构，在CMU-MOSI和CMU-MOSEI上达到SOTA。

**[REINA: Regularized Entropy Information-Based Loss for Efficient Simultaneous Speech Translation](reina_regularized_entropy_information-based_loss_for_efficient_simultaneous_spee.md)**

:   提出 REINA（Regularized Entropy INformation Adaptation）损失函数，基于互信息理论高效地将非流式语音翻译模型转换为流式同声传译模型，在多语言方向上达到 SOTA 流式翻译性能，并提出新的流式效率评估指标 NoSE。

**[Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding](say_more_with_less_variable-frame-rate_speech_tokenization_via_adaptive_clusteri.md)**

:   提出 VARSTok，首个全动态可变帧率语音 tokenizer，通过时序感知密度峰聚类和隐式时长编码，实现自适应 token 分配，在使用更少 token 的同时超越固定帧率基线。

**[A Text-Routed Sparse Mixture-of-Experts Model with Explanation and Temporal Alignment for Multi-Modal Sentiment Analysis](text-routed_sparse_mixture-of-experts_model_with_explanation_and_temporal_alignm.md)**

:   提出 TEXT 模型，利用 MLLM 为音视频生成自然语言解释来增强模态表示，设计融合 Mamba 与时序交叉注意力优点的轻量时序对齐模块，并以文本路由的稀疏专家混合进行跨模态融合，在四个 MSA 数据集上全面超越 SOTA 及 GPT-4o 等大模型。

**[Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases](thucy_an_llm-based_multi-agent_system_for_claim_verification_across_relational_d.md)**

:   提出首个跨数据库、跨表的多 Agent 声明验证系统 Thucy，由 Verifier 领导三个专家 Agent（Data/Schema/SQL Expert），对数据源完全无先验知识，能自主发现、推理并生成 SQL 证据，在 TabFact 上超越 SOTA 5.6 个百分点（94.3%）。

**[Towards Authentic Movie Dubbing with Retrieve-Augmented Director-Actor Interaction Learning](towards_authentic_movie_dubbing_with_retrieve-augmented_director-actor_interacti.md)**

:   Authentic-Dubber 模拟真实配音工作流程中导演与演员的交互过程，通过构建多模态参考素材库、基于情感相似度的检索增强策略和渐进式图语音生成方法，显著提升了自动电影配音的情感表现力，在V2C-Animation数据集上的情感准确率和MOS评分均达到SOTA。

**[USE: A Unified Model for Universal Sound Separation and Extraction](use_a_unified_model_for_universal_sound_separation_and_extraction.md)**

:   提出 USE 统一框架，通过 EDA 网络推断声源数量和声学线索实现声音分离 (SS)，多模态融合网络解释用户提供的文本/视频/标签线索实现目标声音提取 (TSE)，联合训练+跨任务对齐使两项任务互相增强，SS +1.4dB SDR，TSE 匹配准确率 86%。
