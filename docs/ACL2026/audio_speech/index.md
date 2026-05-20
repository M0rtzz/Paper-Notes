---
title: >-
  ACL2026 音频/语音方向47篇论文解读
description: >-
  47篇ACL2026的音频/语音方向论文解读，涵盖语音、对话系统、对抗鲁棒、LLM、问答、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "音频/语音"
  - "论文解读"
  - "论文笔记"
  - "语音"
  - "对话系统"
  - "对抗鲁棒"
  - "LLM"
  - "问答"
  - "多模态"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**💬 ACL2026** · **47** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (7)](../../ICML2026/audio_speech/index.md) · [📷 CVPR2026 (15)](../../CVPR2026/audio_speech/index.md) · [🔬 ICLR2026 (32)](../../ICLR2026/audio_speech/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/audio_speech/index.md) · [🧠 NeurIPS2025 (50)](../../NeurIPS2025/audio_speech/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/audio_speech/index.md)

🔥 **高频主题：** 语音 ×39 · 对话系统 ×5 · 对抗鲁棒 ×3 · LLM ×3 · 问答 ×3

**[Affectron: Emotional Speech Synthesis with Affective and Contextually Aligned Nonverbal Vocalizations](affectron_emotional_speech_synthesis_with_affective_and_contextually_aligned_non.md)**

:   本文提出 Affectron 框架，通过情感驱动的 Top-K NV 匹配和情感感知的 Top-K 路由两个训练时增强策略，在小规模开源解耦语料上实现了多样且情感对齐的非语言发声（如笑声、叹息）合成，显著超越了基于纯语言预训练的 VoiceCraft 基线。

**[An Exploration of Mamba for Speech Self-Supervised Models](an_exploration_of_mamba_for_speech_self-supervised_models.md)**

:   首次全面探索Mamba架构作为语音自监督学习（SSL）基础模型的潜力，发现Mamba-based HuBERT在长上下文ASR、流式ASR和因果设置的probing任务中优于Transformer，同时保持线性时间复杂度。

**[Analyzing Reasoning Shifts in Audio Deepfake Detection under Adversarial Attacks: The Reasoning Tax versus Shield Bifurcation](analyzing_reasoning_shifts_in_audio_deepfake_detection_under_adversarial_attacks.md)**

:   本文为带推理链的音频语言模型（ALM）做深度伪造检测设计了"三维取证审计"框架（声学感知 / 认知一致性 / 认知失调），发现 CoT 推理并非普适增强——对声学感知强的模型（Qwen2-Audio）是"防护盾"，对感知弱的模型（Gemma-3n、Phi-4）反而是"推理税"；且当模型被攻破时，高认知失调可作为"无声警报"提醒人类审计员。

**[Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)**

:   本文提出锚定循环生成（ACG）范式，通过在自回归过程中用已确认的音乐内容作为锚点来校准生成方向，有效缓解长序列符号音乐生成中的误差累积问题，并构建了层次化框架Hi-ACG实现从全局到局部的音乐生成。

**[[b] = [d] − [t] + [p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](bd-tp_self-supervised_speech_models_discover_phonological_vector_arithmetic.md)**

:   系统性地证明自监督语音模型（S3M）的表示空间中存在线性的音韵特征向量，这些向量满足类似 word2vec 的向量算术关系，且其缩放比例与声学测量呈连续相关性。

**[Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs](beyond_transcription_unified_audio_schema_for_perception-aware_audiollms.md)**

:   揭示当前 AudioLLM 的感知弱点源于 ASR 中心的训练范式（系统性抑制副语言和非语言信息），提出 Unified Audio Schema（UAS）将音频信息结构化为转录、副语言和非语言事件三个维度的 JSON 格式，在 MMSU 基准上感知精度提升 10.9% 同时保持推理能力。

**[Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)**

:   本文提出 TARS（Trajectory Alignment for Reasoning in Speech），一个基于强化学习的框架，通过表示对齐和行为对齐两种密集奖励信号，将语音条件下的推理轨迹与文本条件下的推理轨迹对齐，在 7B 规模模型中达到 SOTA，MRR（模态恢复率）接近甚至超过 100%。

**[Computational Narrative Understanding for Expressive Text-to-Speech](computational_narrative_understanding_for_expressive_text-to-speech.md)**

:   本文从有声书虚构作品中提取角色直接引语，构建了大规模表达性语音数据集 LibriQuote（5.3K 小时引语 + 12.7K 小时叙述），并用语音动词和副词伪标签标注说话风格，实验表明在 flow-matching 模型上微调可同时提升表达性和可懂度，且 LibriQuote-test 构成了一个具有挑战性的表达性 TTS 基准。

**[ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling](controlaudio_tackling_text-guided_timing-indicated_and_intelligible_audio_genera.md)**

:   本文提出 ControlAudio，一个统一的渐进式扩散建模框架，通过三阶段渐进训练（TTA 预训练→时序控制微调→时序+可懂语音联合训练）和渐进引导采样，在单个扩散模型中实现文本引导、时序精确控制和可懂语音生成三种能力，在时序精度和语音清晰度上显著超越现有方法。

**[Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech](data-efficient_targeted_token-level_preference_optimization_for_llm-based_text-t.md)**

:   针对 LLM-based TTS 中模糊发音（如日语「辛い」既可读 karai 也可读 tsurai）的对齐难题，作者提出 TKTO：先用两个标签对调训练的对比 KTO 模型估计每个 token 的重要度权重 $w_t$，再把 KTO 的 utterance 级 value function 拆到 token 级并加权聚合，实现「无需配对数据 + 自动定位目标 token」的双重升级，把日语发音准确率从 0.668 抬到 0.958（+39%），CER 降 54%。

**[Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps](detecting_hallucinations_in_speechllms_at_inference_time_using_attention_maps.md)**

:   提出四种基于音频注意力的指标（AudioRatio、AudioConsistency、AudioEntropy、TextEntropy），训练轻量级逻辑回归分类器在推理时检测语音大模型（SpeechLLM）的幻觉，在域内数据上 PR-AUC 提升最高达 +0.23。

**[Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)**

:   本文通过逐层oracle干预实验揭示了大语音语言模型（LSLM）中语音token表示的结构化冗余层次——浅层编码必要声学细节而深层极度冗余——并提出Affinity Pooling这一免训练的基于相似度的token合并机制，在减少27.48% FLOPs的同时保持竞争力的准确率。

**[Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)**

:   作者提出 Full-Duplex-Bench-v2，让一个 GPT-Realtime 扮演的 Examiner 通过 WebRTC 与被测全双工模型实时对话，按 Daily/Correction/Entity/Safety 四类任务、Fast/Slow 两种节奏对其打 turn-taking、instruction-following、task-specific 三类分，发现 GPT-Realtime、Moshi、Freeze-Omni 都会随对话推进性能持续下滑，且开源模型在 correction 和 entity tracking 上尤其拉胯。

**[SEPT: Semantically Expanded Prompt Tuning for Audio-Language Models](generalizable_prompt_tuning_for_audio-language_models_via_semantic_expansion.md)**

:   SEPT 通过利用 LLM 生成语义邻居并设计带边距约束的语义扩展损失来正则化提示嵌入空间，显著缓解了音频语言模型（ALM）提示调优中的 Base-New Tradeoff 问题，建立了 ALM 提示泛化的首个系统性评估基准。

**[HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models](halluaudio_a_comprehensive_benchmark_for_hallucination_detection_in_large_audio-.md)**

:   本文提出 HalluAudio，首个大规模跨领域（语音/环境声/音乐）的音频幻觉检测基准，包含 5000+ 人工验证的 QA 对和系统化的对抗性提示设计，通过多维指标（准确率/幻觉率/Yes-No偏差/拒绝率/错误类型）评估主流 LALM，揭示了当前模型在声学锚定、时间推理和音乐属性理解方面的显著缺陷。

**[Hard to Be Heard: Phoneme-Level ASR Analysis of Phonologically Complex, Low-Resource Endangered Languages](hard_to_be_heard_phoneme-level_asr_analysis_of_phonologically_complex_low-resour.md)**

:   本文对两种音系极端复杂的低资源濒危东高加索语言（Archi和Rutul）进行音素级ASR分析，发现音素识别准确率与训练频率呈S型学习曲线关系，许多归因于音系复杂性的错误实际上更多源于数据稀缺。

**[HCFD: A Benchmark for Audio Deepfake Detection in Healthcare](hcfd_a_benchmark_for_audio_deepfake_detection_in_healthcare.md)**

:   本文提出医疗场景下的编解码器伪造语音检测任务 HCFD，构建了首个包含多种临床病理条件（抑郁、阿尔茨海默、构音障碍）的编解码器伪造语音数据集 HCFK，并提出 PHOENIX-Mamba 框架——通过在双曲空间中建模多模式伪造证据原型，在英文抑郁检测上达到 97.04% 准确率。

**[How Tokenization Limits Phonological Knowledge Representation in Language Models and How to Improve Them](how_tokenization_limits_phonological_knowledge_representation_in_language_models.md)**

:   本文用三个音韵 probing 任务（rhyme / G2P / 音节数）证明 BPE 类 subword tokenization 既"粒度太粗"难以捕捉局部音韵，又"边界错位"难以捕捉韵律结构，并提出 STAD 度量 + IPA-augmented 轻量微调，让 Llama3.1-8B 在三个音韵任务全面提升而 GSM8K / MMLU 只掉 1.1% / 0.9%。

**[Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages](indic-codecfake_meets_satyam_towards_detecting_neural_audio_codec_synthesized_sp.md)**

:   本文构建了首个多印度语言的 CodecFake 检测基准 ICF，并提出 SATYAM——一个双曲音频大语言模型，通过在双曲空间中用 Bhattacharyya 距离对齐语义和副语言表示再与提示对齐，仅训练 3.75M 参数即达到 98.32% 的检测准确率。

**[Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)**

:   构建 Jamendo-MT-QA，一个包含 36,519 个比较问答对（覆盖 12,173 个音轨对）的多音轨比较音乐问答基准，首次系统评估音频-语言模型在跨音轨比较推理上的能力，揭示现有模型在句子级比较生成上的显著不足。

**[MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus](mcga_a_multi-task_classical_chinese_literary_genre_audio_corpus.md)**

:   本文构建了首个面向中国古典文学的大规模（119小时、22000条样本）全版权音频语料库 MCGA，涵盖赋、诗、文、词、曲五大文体和六项语音任务（ASR/S2TT/SEC/SQA/SU/SR），并通过评测 10 个多模态大模型揭示了当前模型在古典文学语音理解上的显著不足。

**[Mind the Pause: Disfluency-Aware Objective Tuning for Multilingual Speech Correction with LLMs](mind_the_pause_disfluency-aware_objective_tuning_for_multilingual_speech_correct.md)**

:   作者提出一个多语言 disfluency 修正流水线：先用 MuRIL 在 token 级标注 fluent/disfluent 标签，再把"原始转录 + token 标签"一起喂给 Llama-3.2-3B / Qwen2.5-3B 做 instruction fine-tuning，关键创新是引入一个**对比损失项**，对生成 disfluent token 的概率显式惩罚（penalize $-\log(1-\sum_v w_v P_\theta(v))$），在 Hindi/Bengali/Marathi 三语种实ASR数据上比无对比 baseline +1.97 BLEU、比 mBART +8.54 BLEU，且 3B 模型在多数 setting 上能匹配甚至超越 GPT-4o。

**[MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)**

:   提出 MTR-DuplexBench，一个针对全双工语音语言模型（FD-SLM）的多轮综合评估基准，通过创新的轮次分割方法解决了全双工对话中轮次边界模糊和上下文不一致的挑战，涵盖对话特性、对话质量、指令遵循和安全性四个维度，实验揭示了现有 FD-SLM 在多轮交互中性能持续衰退的问题。

**[Multimodal In-Context Learning for ASR of Low-Resource Languages](multimodal_in-context_learning_for_asr_of_low-resource_languages.md)**

:   系统研究多模态上下文学习（MICL）能否使语音 LLM 学习未见过的濒危语言，并提出基于 MICL 的假设选择系统，结合声学模型与语音 LLM 的互补优势，在三种濒危语言上显著提升 ASR 性能。

**[Music Audio-Visual Question Answering Requires Specialized Multimodal Designs](music_audio-visual_question_answering_requires_specialized_multimodal_designs.md)**

:   本文作为音乐视听问答（Music AVQA）领域首篇综合综述，系统分析了数据集演进和方法设计，论证了专门的输入处理、时空架构设计和音乐领域知识对该任务至关重要，通用多模态模型不足以应对音乐表演的独特挑战。

**[MSU-Bench: Musical Score Understanding Benchmark](musical_score_understanding_benchmark_evaluating_large_language_models39_compreh.md)**

:   MSU-Bench 是首个针对完整乐谱理解的人工标注基准，包含 150 首作品的 1800 个生成式 QA 对，覆盖四级难度，评估揭示了 LLM/VLM 在乐谱定位和幻觉方面的严重不足，而 ABC 记谱法的文本输入显著缓解了这些问题。

**[Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval](omni-embed-audio_leveraging_multimodal_llms_for_robust_audio-text_retrieval.md)**

:   本文提出 OEA（Omni-Embed-Audio），利用多模态 LLM 作为统一编码器构建检索导向的音频-文本嵌入空间，并引入 User-Intent Queries（UIQ）基准和硬负例区分指标（HNSR/TFR），发现 LLM 主干在 T2T 检索（+22%）和硬负例区分（+4.3%p HNSR@10）上显著优于 CLAP 系列方法。

**[Protecting Bystander Privacy via Selective Hearing in Audio LLMs](protecting_bystander_privacy_via_selective_hearing_in_audio_llms.md)**

:   提出首个旁观者隐私基准 SH-Bench 和旁观者隐私微调（BPFT）方法，评估和提升音频 LLM 在多说话人环境中仅关注主说话人、拒绝泄漏旁观者信息的能力，BPFT 后 SE 指标比 Gemini 2.5 Pro 高 16%。

**[Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition](pseudo2real_task_arithmetic_for_pseudo-label_correction_in_automatic_speech_reco.md)**

:   本文提出 Pseudo2Real，一种参数空间校正方法，通过在源域中计算真实标签模型与伪标签模型的权重差得到"校正向量"，将其应用于目标域伪标签微调模型以纠正系统性伪标签偏差，在 AfriSpeech-200 的十种非洲口音上最高实现 35% 相对 WER 降低。

**[ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis](restyle-tts_relative_and_continuous_style_control_for_zero-shot_speech_synthesis.md)**

:   ReStyle-TTS 通过解耦文本/参考音频 guidance、可连续缩放的风格 LoRA、正交 LoRA 融合和音色一致性优化，让零样本 TTS 不再被参考音频风格锁死，可以相对地调高/调低音高、能量和情绪，同时保持文本可懂度与说话人音色。

**[Retrieving to Recover: Towards Incomplete Audio-Visual Question Answering via Semantic-consistent Purification](retrieving_to_recover_towards_incomplete_audio-visual_question_answering_via_sem.md)**

:   本文提出R2ScP框架，将AVQA中缺失模态处理范式从传统的生成式补全转变为基于检索的恢复，通过跨模态检索和上下文感知自适应净化机制消除检索噪声，在模态不完整场景下显著提升了问答性能。

**[RTCFake: Speech Deepfake Detection in Real-Time Communication](rtcfake_speech_deepfake_detection_in_real-time_communication.md)**

:   RTCFake 构建了约 600 小时面向真实实时通信平台的语音伪造检测数据集，并提出音素引导一致性学习 PCL，使 XLSR+AASIST 在离线、在线、跨平台和未见噪声场景下的平均 EER 从混合训练的 7.33% 降到 5.81%。

**[S2S-Arena: Evaluating Paralinguistic Instruction Following in Speech-to-Speech Models](s2s-arena_evaluating_paralinguistic_instruction_following_in_speech-to-speech_mo.md)**

:   S2S-Arena 提出一个直接在语音模态评测 S2S 模型的 benchmark，用四级副语言交互协议、1,243 条语音样本和 1,001 次 pairwise comparison 揭示当前系统在复杂语气、情绪、说话风格和表达控制上的明显差距。

**[SDiaReward: Modeling and Benchmarking Spoken Dialogue Rewards with Modality and Colloquialness](sdiareward_modeling_and_benchmarking_spoken_dialogue_rewards_with_modality_and_c.md)**

:   SDiaReward 构建了面向多轮语音对话的成对偏好数据集与 ESDR-Bench，并训练端到端语音 reward model，让评测不再只看文本语义，而能同时判断韵律/情感等 modality gap 与自然口语风格的 colloquialness gap。

**[Speculative End-Turn Detector for Efficient Speech Chatbot Assistant](speculative_end-turn_detector_for_efficient_speech_chatbot_assistant.md)**

:   论文构建首个公开 end-turn detection 数据集 OpenETD，并提出 SpeculativeETD，让端侧 GRU 持续检测 speaking/non-speaking，只有遇到 200 ms 静音时才调用服务端 Wav2Vec2 区分 Gap 与 Pause，从而在真实语音上以 38 倍更低 FLOPs 和亚毫秒端侧延迟换取接近大模型的实时 turn-taking 效果。

**[SpeechLLM-as-Judges: Towards General and Interpretable Speech Quality Evaluation](speechllm-as-judges_towards_general_and_interpretable_speech_quality_evaluation.md)**

:   这篇论文把语音质量评估从“给一个分数”扩展为“可解释的语音评审”，构建了含 32,207 条多语音频和 128,754 条标注的 SpeechEval 数据集，并用 CoT 指令微调与 GRPO 训练出 SQ-LLM，在质量评分、成对比较、改进建议和深伪检测四类任务上整体优于现有语音大模型与专家模型。

**[Standard-to-Dialect Transfer Trends Differ across Text and Speech: A Case Study on Intent and Topic Classification in German Dialects](standard-to-dialect_transfer_trends_differ_across_text_and_speech_a_case_study_o.md)**

:   这篇论文用德语-巴伐利亚语意图分类和德语-瑞士德语主题分类系统比较文本、语音、ASR级联三种迁移路径，发现标准语上的最佳方案不一定适合方言：文本模型最适合标准德语，而语音模型在方言输入上通常更稳。

**[Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions](still_between_us_evaluating_and_improving_voice_assistant_robustness_to_third-pa.md)**

:   针对语音助手无法区分第三方打断（TPI）与主用户发言的问题，提出包含88K训练实例的TPI-Train数据集和TPI-Bench评测框架，通过说话人感知的困难负样本挖掘策略消除语义捷径学习，使模型真正依赖声学线索进行打断检测。

**[StressTest: Can YOUR Speech LM Handle the Stress?](stresstest_can_your_speech_lm_handle_the_stress.md)**

:   提出 StressTest 基准评估语音语言模型（SLMs）对句子重音含义的理解能力，发现现有模型几乎无法基于重音模式推理说话者意图，并通过合成数据管线 Stress-17k 训练的 StresSLM 在重音检测和推理任务上大幅超越前沿模型。

**[TellWhisper: Tell Whisper Who Speaks When](tellwhisper_tell_whisper_who_speaks_when.md)**

:   本文提出TellWhisper，通过设计时间-说话人感知的旋转位置编码（TS-RoPE）将说话人身份和时间信息统一编码到语音编码器的自注意力中，配合双曲空间说话人日志模型（Hyper-SD），实现了对"谁在何时说了什么"的联合建模，在多说话人ASR任务上取得最优性能。

**[Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)**

:   提出 TCD，一种无训练的推理时解码方法：通过对比原始音频和时间模糊慢速路径的 logits 差异，配合稳定性引导的模糊窗口和不确定性门控，使统一音频语言模型更好地利用瞬态声学线索，在 MMAU 和 AIR-Bench 上一致提升。

**[Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training](towards_fine-grained_and_multi-granular_contrastive_language-speech_pre-training.md)**

:   本文提出FCaps大规模数据集（47k小时语音、19M细粒度标注）和CLSP对比学习模型，通过端到端标注管线和细粒度多粒度对比监督，实现了首个能统一表征全局和细粒度语音风格的语音-文本对齐模型。

**[UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions](unisonate_a_unified_model_for_speech_music_and_sound_effect_generation_with_text.md)**

:   UniSonate 用统一的 Instruction-Content 表示、动态 SFX token 注入和多阶段课程学习，把文本转语音、文本转音乐和文本转音效放进同一个 flow-matching MM-DiT 中，在 TTS 与 TTM 上达到或超过专用模型，同时在 TTA 上保持可用的音效生成能力。

**[VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models](vapo_end-to-end_slide-enhanced_speech_recognition_with_omni-modal_large_language.md)**

:   本文发现端到端全模态大模型做 SlideASR 时会把幻灯片文字误抄成语音内容，并提出 VAPO 用“先看后听”的结构化推理链和多目标强化学习，把幻灯片文字变成语音识别的语义锚点而不是干扰源。

**[VoxMind: An End-to-End Agentic Spoken Dialogue System](voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)**

:   提出 VoxMind，一个赋予端到端语音对话模型智能体能力的统一框架：通过"Think-before-Speak"机制实现显式推理，结合多智能体动态工具管理架构解耦推理延迟与工具规模，任务完成率从基线 34.88% 提升至 74.57%，超越 Gemini-2.5-Pro。

**[When Misinformation Speaks and Converses: Rethinking Fact-Checking in Audio Platforms](when_misinformation_speaks_and_converses_rethinking_fact-checking_in_audio_platf.md)**

:   本文为Position Paper，论证音频平台上的虚假信息在本质上不同于文本虚假信息——它同时具有口语性（prosody、pacing、emotion）和对话性（多轮、多说话人、跨集节），现有以文本为中心的事实核查流水线无法有效处理，需要围绕音频特有属性重新设计验证框架。

**[ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching](zipvoice-dialog_non-autoregressive_spoken_dialogue_generation_with_flow_matching.md)**

:   提出 ZipVoice-Dialog，首个基于流匹配的非自回归零样本对话语音生成模型，通过课程学习策略和说话人轮次嵌入两个简单设计，解决了流匹配直接用于对话场景时的语音不可懂和轮次混乱问题，同时发布了首个大规模开源对话语音数据集 OpenDialog（6.8k 小时）。
