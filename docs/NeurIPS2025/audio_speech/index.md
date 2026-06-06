---
title: >-
  NeurIPS2025 音频/语音方向50篇论文解读
description: >-
  50篇NeurIPS2025的音频/语音方向论文解读，涵盖语音、对抗鲁棒、多模态、对齐/RLHF、推荐系统、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "NeurIPS2025"
  - "音频/语音"
  - "论文解读"
  - "论文笔记"
  - "语音"
  - "对抗鲁棒"
  - "多模态"
  - "对齐/RLHF"
  - "推荐系统"
  - "Agent"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**🧠 NeurIPS2025** · **50** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (30)](../../ICML2026/audio_speech/index.md) · [💬 ACL2026 (69)](../../ACL2026/audio_speech/index.md) · [📷 CVPR2026 (14)](../../CVPR2026/audio_speech/index.md) · [🔬 ICLR2026 (32)](../../ICLR2026/audio_speech/index.md) · [🤖 AAAI2026 (30)](../../AAAI2026/audio_speech/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/audio_speech/index.md)

🔥 **高频主题：** 语音 ×12 · 对抗鲁棒 ×4 · 多模态 ×3 · 对齐/RLHF ×3 · 推荐系统 ×2

**[A Controllable Examination for Long-Context Language Models](a_controllable_examination_for_longcontext_language_models.md)**

:   提出LongBioBench，通过生成虚构传记作为可控的needle和haystack，构建满足"无缝上下文、可控设置、可靠评估"三大原则的长上下文LLM评估框架，测试18个模型后揭示当前LCLM在检索能力尚可的情况下推理和可信性仍有显著短板。

**[A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings](a_multitask_benchmark_for_abusive_language_detection_in_lowr.md)**

:   提出 TiALD（Tigrinya Abusive Language Detection），首个面向 Tigrinya 低资源语言的大规模多任务基准数据集，包含 13,717 条 YouTube 评论的辱骂/情感/主题三任务联合标注，同时发现小型微调模型（TiRoBERTa, 125M）在所有任务上全面超越 GPT-4o 和 Claude Sonnet 3.7 等前沿 LLM。

**[A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity](a_triangle_enables_multimodal_alignment_beyond_cosine_simila.md)**

:   TRIANGLE提出用高维空间中三模态嵌入向量构成的三角形面积作为相似度度量，替代传统的成对余弦相似度，实现了视频-音频-文本三模态的联合对齐，在视频文本检索等任务上超越SOTA最多9个Recall@1点。

**[Accelerate Creation of Product Claims Using Generative AI](accelerate_creation_of_product_claims_using_generative_ai.md)**

:   开发 Claim Advisor 平台，利用 LLM 的 in-context learning 和 LoRA 微调加速消费品产品宣称的搜索、生成、优化和排序，通过模仿 MaxDiff 研究方法论让微调的 Phi-3 14B 模型在宣称排序上超越 GPT-4o（仅用 1 个示例 vs GPT 的 100 个示例），三轮迭代后 100% 的生成宣称达到"高吸引力"级别。

**[AdaptDel: Adaptable Deletion Rate Randomized Smoothing for Certified Robustness](adaptdel_adaptable_deletion_rate_randomized_smoothing_for_ce.md)**

:   提出 AdaptDel 方法，将随机平滑中用于离散序列的固定删除率扩展为根据输入长度等属性自适应调整的可变删除率，在理论上证明了可变率下认证的 soundness，实验在 NLP 序列分类任务上实现认证区域基数最高 30 个数量级的提升。

**[Associative Syntax and Maximal Repetitions Reveal Context-Dependent Complexity in Fruit Bat Communication](associative_syntax_and_maximal_repetitions_reveal_context-dependent_complexity_i.md)**

:   本文提出一种无监督方法来推断果蝠发声的离散单元、语法类型和时序结构，并首次将最大重复子序列（Maximal Repetitions）引入动物通信领域，发现冲突行为中的通信复杂度显著高于合作行为。

**[AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)**

:   AudSemThinker 为音频语言模型引入结构化语义推理框架——定义 9 类声音语义描述符（谁/什么/如何/何时/何地等），在 Qwen2.5-Omni-7B 上通过 SFT + GRPO（含可验证奖励和长度约束）训练产生 \<think\>\<semantic_elements\>\<answer\> 三阶段输出，MMAU 基准达 66.70%（超越 Audio-Reasoner 61.71% 和 Qwen2.5-Omni 65.60%）。

**[Benchmarking Egocentric Multimodal Goal Inference for Assistive Wearable Agents](benchmarking_egocentric_multimodal_goal_inference_for_assist.md)**

:   Meta 提出 WAGIBench，一个针对可穿戴辅助智能体的多模态目标推断基准，包含 348 名参与者的 3,477 条第一视角录制（29小时），涵盖视觉/音频/数字/纵向四种模态，人类准确率 93% vs 最佳 VLM 84%（MCQ），生成式评估中模型仅 55% 时间产生相关目标，揭示了当前 VLM 在实际可穿戴场景中的显著差距。

**[BNMusic: Blending Environmental Noises into Personalized Music](bnmusic_blending_environmental_noises_into_personalized_music.md)**

:   提出 BNMusic，一个两阶段框架将环境噪声融合到个性化生成音乐中：第一阶段通过 mel-spectrogram 的 outpainting + inpainting 生成与噪声节奏对齐的音乐，第二阶段利用听觉掩蔽理论自适应放大音乐信号以降低噪声感知，无需额外训练，在 EPIC-SOUNDS 和 ESC-50 上显著优于 baseline。

**[Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation](can_llms_outshine_conventional_recommenders_a_comparative_evaluation.md)**

:   提出 RecBench 综合评估框架，在5个领域数据集上系统对比17个LLM与10个传统DLRM，发现LLM推荐器在CTR任务上AUC提升最高5%、在序列推荐上NDCG@10提升最高170%，但推理速度慢10-1000倍，而传统DLRM结合LLM语义嵌入（LLM-for-RS）可以20倍更快的速度达到LLM约95%的性能，是当前最具工业可行性的方案。

**[Characterization and Learning of Causal Graphs from Hard Interventions](characterization_and_learning_of_causal_graphs_from_hard_interventions.md)**

:   首次系统分析硬干预（hard interventions）在含隐变量因果发现中的理论优势，提出广义do-演算（4条规则）和孪生增强MAG图表示，给出 $\mathcal{I}$-Markov 等价类的充要图条件，并设计可证明正确的FCI变体学习算法；实验表明硬干预比软干预将等价类缩小37-57%。

**[Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models](data-juicer_20_cloud-scale_adaptive_data_processing_for_and_with_foundation_mode.md)**

:   Data-Juicer 2.0 是面向基础模型的云规模多模态数据处理系统，150+ 跨文本/图像/视频/音频算子，支持自适应分布式执行（Ray/MaxCompute），在 10000+ CPU 核心上高效处理 TB 级数据，已广泛应用于阿里云 PAI 等产品。

**[DeepASA: An Object-Oriented Multi-Purpose Network for Auditory Scene Analysis](deepasa_an_object-oriented_multi-purpose_network_for_auditory_scene_analysis.md)**

:   提出 DeepASA，一个面向对象的多任务统一架构，通过 object-oriented processing 和 chain-of-inference 机制在单一模型中同时完成多通道声源分离（MIMO）、去混响、声事件检测（SED）、音频分类和到达方向估计（DoAE），在多个空间音频基准上达到 SOTA。

**[E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models](e-bats_efficient_backpropagation-free_test-time_adaptation_for_speech_foundation.md)**

:   提出首个面向语音基础模型的无反向传播测试时自适应框架 E-BATS，通过轻量级 prompt 自适应、多尺度损失函数和测试时 EMA 机制，在保持高精度的同时实现 2.0×–6.4× 的 GPU 显存节省。

**[E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis](e2e-vguard_adversarial_prevention_for_production_llm-based_end-to-end_speech_syn.md)**

:   针对基于 LLM 的端到端语音合成中的声音克隆威胁，提出 E2E-VGuard 主动防御框架，通过编码器集成扰动音色、对抗样本干扰 ASR 发音识别、以及心理声学模型保证不可感知性，在 19 个 TTS 模型和 7 个 ASR 系统上验证了有效性。

**[Echoes of Humanity: Exploring the Perceived Humanness of AI Music](echoes_of_humanity_exploring_the_perceived_humanness_of_ai_music.md)**

:   通过随机对照交叉试验(RCCT)和混合方法内容分析，系统研究听众区分AI生成音乐(AIM)与人类创作音乐的能力，发现随机配对时听众无法区分（准确率≈随机猜测），但相似配对时显著提升至66%，且声音/技术/人声线索是成功区分的关键因素。

**[Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space](efficient_speech_language_modeling_via_energy_distance_in_continuous_latent_spac.md)**

:   提出 SLED，将语音波形编码为连续潜在表示序列，在连续空间中通过 energy distance 目标进行自回归建模，避免了离散化信息损失和 RVQ 所需的复杂层级架构，同时实现高效的零样本与流式语音合成。

**[Ethics Statements in AI Music Papers: The Effective and the Ineffective](ethics_statements_in_ai_music_papers_the_effective_and_the_ineffective.md)**

:   对 AI 音乐领域论文中伦理声明（ethics statements）的使用现状进行系统审查，发现绝大多数伦理声明未被有效利用，并提出面向会议与研究者的改进建议。

**[EuroSpeech: A Multilingual Speech Corpus](eurospeech_a_multilingual_speech_corpus.md)**

:   提出可扩展的开源 pipeline，从 22 个欧洲议会录音中自动构建 EuroSpeech 数据集——61K 小时、覆盖 22 种语言的高质量语音-文本对齐数据，其中 19 种语言超 1K 小时，微调 Whisper 后平均 WER 降低 41.8%。

**[From Generation to Attribution: Music AI Agent Architectures for the Post-Streaming Era](from_generation_to_attribution_music_ai_agent_architectures_for_the_post-streami.md)**

:   提出了一种基于内容的 Music AI Agent 架构，通过将音乐分解为细粒度的 Block 组件并构建 Attribution Layer，将版权归因直接嵌入 AI 音乐创作流程中，为后流媒体时代建立公平的 AI 媒体平台。

**[Generating Physically Sound Designs from Text and a Set of Physical Constraints](generating_physically_sound_designs_from_text_and_a_set_of_physical_constraints.md)**

:   提出 TIDES 框架，将预训练文本-图像模型（CLIP）的视觉引导与可微有限元物理仿真器结合，通过联合优化视觉相似度损失和结构合规性损失，从文本描述和物理约束出发生成既满足工程性能要求又具备文本指定特征的承载结构设计，并通过 3D 打印三点弯曲实验验证了方法的有效性。

**[Inductive Transfer Learning for Graph-Based Recommenders](inductive_transfer_learning_for_graph-based_recommenders.md)**

:   提出 NBF-Rec，一个基于神经 Bellman-Ford 网络的图推荐模型，支持在用户和物品完全不相交的数据集之间进行归纳式迁移学习，实现零样本跨域推荐和轻量微调适配。

**[Instance-Specific Test-Time Training for Speech Editing in the Wild](instance-specific_test-time_training_for_speech_editing_in_the_wild.md)**

:   提出面向野外语音编辑的实例特定测试时训练方法：在推理前利用未编辑区域的真实声学特征做直接监督、编辑区域通过时长约束和音素预测辅助损失做间接监督，对模型进行实例级自适应微调，有效缓解编辑边界的带宽不连续问题，并支持通过 mask 长度调整精确控制语速，在野外 benchmark 上主客观评估均超越现有系统。

**[Latent Space Factorization in LoRA](latent_space_factorization_in_lora.md)**

:   提出 FVAE-LoRA，在 LoRA 框架中引入具有双潜空间的 VAE，通过新型 ELBO 目标将任务相关特征 ($\mathbf{z}_1$) 与残差信息 ($\mathbf{z}_2$) 显式分解，在文本、图像、音频任务上一致优于标准 LoRA。

**[LeVo: High-Quality Song Generation with Multi-Preference Alignment](levo_high-quality_song_generation_with_multi-preference_alignment.md)**

:   提出 LeVo 歌曲生成框架，通过语言模型并行建模混合 token 和双轨 token 以兼顾人声-伴奏和谐性和音质，并创新性地引入基于 DPO 的多偏好对齐方法提升音乐性和指令跟随能力。

**[LeVo: High-Quality Song Generation with Multi-Preference Alignment](levo_high-quality_song_generation_with_multi-processing_refined_supervision.md)**

:   LeVo 提出一种基于语言模型的歌曲生成框架，通过并行预测混合 token 和双轨 token 来同时优化人声-伴奏和谐度与音质，并引入基于 DPO 的多偏好对齐方法提升音乐性和指令跟随能力，在学术方法中全面领先且接近工业系统水平。

**[LUMIA: A Handheld Vision-to-Music System for Real-Time, Embodied Composition](lumia_a_handheld_vision-to-music_system_for_real-time_embodied_composition.md)**

:   提出Lumia——一个手持相机式设备，通过GPT-4 Vision分析拍摄画面生成结构化提示，再由Stable Audio合成音乐循环段，实现从视觉到音乐的实时、具身化即兴创作工作流。

**[MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation](megadance_mixture-of-experts_architecture_for_genre-aware_3d_dance_generation.md)**

:   提出 MEGADance，首个基于混合专家 (MoE) 架构的音乐驱动 3D 舞蹈生成方法，通过将编舞一致性解耦为"舞蹈通用性"（Universal Expert）和"风格特异性"（Specialized Expert），配合 FSQ 量化和 Mamba-Transformer 混合骨干网络，实现了 SOTA 的舞蹈质量和强风格可控性。

**[Merlin L48 Spectrogram Dataset](merlin_l48_spectrogram_dataset.md)**

:   本文提出了 L48 数据集——一个基于真实鸟类录音的细粒度频谱图多标签分类基准，天然具备单正标签多标签 (SPML) 设置，揭示了现有 SPML 方法在真实场景下的严重不足，并提出了基于录音内一致性的正则化方案来提升性能。

**[Mixed Monotonicity Reachability Analysis of Neural ODE: A Trade-Off Between Tightness and Efficiency](mixed_monotonicity_reachability_analysis_of_neural_ode_a_trade-off_between_tight.md)**

:   将连续时间混合单调性技术应用于 Neural ODE 的可达性分析，通过将 Neural ODE 动力学嵌入混合单调系统，利用区间盒的几何简洁性实现高效过逼近，在紧致性（tightness）和计算效率之间提供可控的权衡。

**[MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition](mome_mixture_of_matryoshka_experts_for_audio-visual_speech_recognition.md)**

:   MoME将稀疏MoE集成到Matryoshka表示学习框架中，用于LLM-based音视频语音识别，通过共享路由器实现跨粒度知识迁移，在单一模型权重下支持多种压缩率的弹性推理，同时达到AVSR/ASR/VSR的SOTA性能。

**[Multi-head Temporal Latent Attention](multi-head_temporal_latent_attention.md)**

:   MTLA 在 MLA 低秩潜在维度压缩基础上，用超网络动态融合时序相邻的 KV 向量，实现 KV 缓存在特征维度和时序维度的双重压缩，配合 stride-aware 因果 mask 保证训练-推理一致性，在语音翻译等任务上达到 4.29× 加速和 6.58× 内存降低，质量持平甚至略优于标准 MHA。

**[Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)**

:   提出一个节点图式故事编辑系统，允许创作者通过自然语言和节点级操作迭代地生成、编辑和比较多模态内容（文本、音频、图像、视频），支持线性和分支叙事结构。

**[Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders](perceptually_aligning_representations_of_music_via_noise-augmented_autoencoders.md)**

:   证明在自编码器训练中对潜变量加噪（noise-augmented latent training）配合感知损失，能使编码空间形成"感知层次结构"——感知最显著的音乐特征（如音高）编码在最粗粒度的潜在结构中，而次要特征（如音色细节）编码在细粒度结构中。这种对齐改善了潜在扩散解码下的音乐惊奇感估计和 EEG 脑响应预测。

**[Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers](physics_of_language_models_part_41_architecture_design_and_the_magic_of_canon_la.md)**

:   通过受控合成预训练任务系统性比较语言模型架构，发现 Canon 层——一种轻量级的邻近token加权求和组件——能显著提升推理深度（2-4倍）、推理广度、知识容量等核心能力，让 NoPE 匹配 RoPE，让 GLA 匹敌 Mamba2/GDN。

**[Resounding Acoustic Fields with Reciprocity](resounding_acoustic_fields_with_reciprocity.md)**

:   利用声波传播的互易性原理，提出Versa方法（ELE数据增强+SSL自监督学习），通过交换发射器和接收器角色来生成物理有效的虚拟训练样本，在稀疏发射器配置下大幅提升声场估计性能。

**[SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers](sand-math_using_llms_to_generate_novel_difficult_and_useful_mathematics_question.md)**

:   提出 SAND-Math，一个无需种子数据集的全自动合成数学问题生成管线，通过 Difficulty Hiking 系统性提升题目难度，仅 500 道增强 LIMO 基线即可在 AIME25 上提升 4.39pp。

**[Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization](seeing_sound_hearing_sight_uncovering_modality_bias_and_conflict_of_ai_models_in.md)**

:   通过6种受控视听条件和人类心理物理实验，系统揭示现有AI声源定位模型存在严重视觉偏见（视听冲突时降至随机水平），并提出神经科学启发的EchoPin模型——HRTF滤波+ERB耳蜗图+立体声，在自建AudioCOCO数据集上大幅超越现有方法，且无需人类行为监督即涌现出类人的水平>垂直定位精度不对称性。

**[Segment-Factorized Full-Song Generation on Symbolic Piano Music](segment-factorized_full-song_generation_on_symbolic_piano_music.md)**

:   提出Segmented Full-Song模型（SFS），将歌曲分解为片段，通过选择性注意结构相关上下文自回归生成各片段，实现比现有方法更快速、更结构化的钢琴全曲生成，并支持交互式人机共创。

**[Sensorium Arc: AI Agent System for Oceanic Data Exploration and Interactive Eco-Art](sensorium_arc_ai_agent_system_for_oceanic_data_exploration_and_interactive_eco-a.md)**

:   本文构建了一个名为 Sensorium Arc 的多模态交互式 AI 智能体系统，通过将海洋拟人化为一个诗意的"讲述者"角色，利用多智能体 RAG 架构将 NASA 海洋科学数据与生态美学文本相结合，使用户能够以自然对话的方式探索复杂的海洋环境数据，同时在视听层面生成动态的科学可视化和艺术化反馈，实现从"被动数据观察"到"主动生态对话"的范式转变。

**[SHAP Meets Tensor Networks: Provably Tractable Explanations with Parallelism](shap_meets_tensor_networks_provably_tractable_explanations_with_parallelism.md)**

:   本文首次为张量网络（Tensor Networks）提供可证明精确的 SHAP 解释计算框架，证明张量列车（Tensor Train）结构下 SHAP 可在多对数时间内并行计算（NC² 复杂度），并通过归约揭示二值化神经网络中**宽度而非深度**才是 SHAP 计算的核心瓶颈。

**[SimulMEGA: MoE Routers are Advanced Policy Makers for Simultaneous Speech Translation](simulmega_moe_routers_are_advanced_policy_makers_for_simultaneous_speech_transla.md)**

:   提出SimulMEGA框架，结合前缀训练与混合专家(MoE)精炼模块，实现无监督的读/写策略学习，使500M参数模型在6种语言的同时语音翻译中以1.5秒延迟仅损失<7% BLEU，并扩展到流式TTS。

**[Slimmable NAM: Neural Amp Models with Adjustable Runtime Computational Cost](slimmable_nam_neural_amp_models_with_adjustable_runtime_computational_cost.md)**

:   将 Slimmable Networks 思想应用到 Neural Amp Modeler (NAM) 中，通过训练期间随机裁剪 WaveNet 层宽度，实现模型在推理时可以无额外训练代价地动态调整网络大小，使音乐家能实时平衡音质精度与计算成本。

**[Sound Logical Explanations for Mean Aggregation Graph Neural Networks](sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)**

:   针对使用均值聚合函数的 GNN（MAGNN，即非负权重的 mean-GNN），证明了能够作为其 sound 解释的单调逻辑规则的精确类别，并构造了一个一阶逻辑的受限片段来解释任意 MAGNN 预测，实验表明限制非负权重不显著影响性能且能有效提取 sound 规则。

**[Target Speaker Extraction Through Comparing Noisy Positive and Negative Audio Enrollments](target_speaker_extraction_through_comparing_noisy_positive_and_negative_audio_en.md)**

:   提出一种利用噪声正样本（目标说话人在说话的段落）和负样本（目标说话人沉默的段落）对比来编码目标说话人特征的新型注册策略，在单声道噪声注册目标说话人提取任务上取得 SOTA 性能，SI-SNRi 比此前最优方法高出 2.1 dB 以上。

**[AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time](textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)**

:   提出 AVRobustBench，首个系统评估音视频模型在 **双模态共现关联腐蚀** 下测试时鲁棒性的基准，包含 4 个数据集 × 75 种腐蚀，并提出基于低熵样本筛选的 TTA 方法 AV2C。

**[The Impact of Scaling Training Data on Adversarial Robustness](the_impact_of_scaling_training_data_on_adversarial_robustness.md)**

:   系统评估 36 个 SOTA 视觉模型在 6 类黑盒攻击下的鲁棒性，发现攻击成功率(ASR)随数据量和模型规模按对数律下降，但 **数据质量和模型规模比数据量本身更关键**。

**[Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization](unifying_symbolic_music_arrangement_track-aware_reconstruction_and_structured_to.md)**

:   提出一个统一的符号音乐编排框架，通过段级自监督重建目标（解耦内容和乐器风格）和新的多轨token化方案REMI-z，使单个预训练模型能够处理乐队编排、钢琴缩编和鼓编排等多种编排任务，并在三个典型任务上超越了任务特定的SOTA。

**[VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction](vita-15_towards_gpt-4o_level_real-time_vision_and_speech_interaction.md)**

:   VITA-1.5 提出了一套精心设计的三阶段渐进式训练策略，将视觉和语音能力逐步整合进 LLM 中，实现了无需独立 ASR/TTS 模块的端到端视觉-语音实时交互，在图像、视频和语音基准上均达到开源模型领先水平。

**[WhAM: Towards A Translative Model of Sperm Whale Vocalization](wham_towards_a_translative_model_of_sperm_whale_vocalization.md)**

:   提出 WhAM（Whale Acoustics Model），首个基于 Transformer 的抹香鲸 coda 生成模型，通过微调 VampNet 实现声学翻译、合成生成与下游分类的三合一能力。
