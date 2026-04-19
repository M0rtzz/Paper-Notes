---
title: >-
  AAAI2026 人体理解方向 65篇论文解读
description: >-
  65篇AAAI2026 人体理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**🤖 AAAI2026** · **65** 篇论文解读

**[10 Open Challenges Steering the Future of Vision-Language-Action Models](10_open_challenges_steering_the_future_of_vision-language-ac.md)**

:   系统梳理 VLA 模型面临的 10 大开放挑战——多模态感知、鲁棒推理、高质量训练数据、评估、跨机器人动作泛化、资源效率、全身协调、安全保障、Agent 框架、人机协作——并讨论空间理解、世界动力学建模、后训练和数据合成四大新兴趋势。

**[AHAN: Asymmetric Hierarchical Attention Network for Identical Twin Face Verification](ahan_asymmetric_hierarchical_attention_network_for_identical.md)**

:   针对同卵双胞胎人脸验证这一极端细粒度识别挑战，提出 AHAN 多流架构，通过层次交叉注意力 (HCA) 对语义面部区域做多尺度分析、面部不对称注意力模块 (FAAM) 捕获左右脸差异签名、以及双胞胎感知配对交叉注意力 (TA-PWCA) 训练正则化，在 ND_TWIN 数据集上将双胞胎验证精度从 88.9% 提升至 92.3%（+3.4%）。

**[Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](anti-adversarial_learning_desensitizing_prompts_for_large_la.md)**

:   提出 PromptObfus，通过"反对抗学习"思路将用户 prompt 中的敏感词替换为语义不同但不影响任务输出的词，从而在不降低远端 LLM 任务表现的前提下彻底消除显式隐私泄露，并将隐式隐私推理攻击成功率降低 62.70%。

**[Authority Backdoor: A Certifiable Backdoor Mechanism for Authoring DNNs](authority_backdoor_a_certifiable_backdoor_mechanism_for_authoring_dnns.md)**

:   提出 Authority Backdoor，将硬件指纹作为后门触发器嵌入 DNN，使模型仅在授权设备上正常工作，并通过随机平滑实现可认证鲁棒性，抵御自适应触发器逆向攻击。

**[Auto-PRE: An Automatic and Cost-Efficient Peer-Review Framework for Language Generation Evaluation](auto-pre_an_automatic_and_cost-efficient_peer-review_framework_for_language_gene.md)**

:   提出 Auto-PRE 框架，通过自动资格考试从一致性、相关性、自信度三个维度筛选合格的 LLM 评估者，在无需人工标注的前提下实现了 SOTA 评估性能并大幅降低成本。

**[Behavior Tokens Speak Louder: Disentangled Explainable Recommendation with Behavior Vocabulary](behavior_tokens_speak_louder_disentangled_explainable_recommendation_with_behavi.md)**

:   提出 BEAT 框架，通过向量量化自编码将用户/物品的行为表征离散化为可解释的 behavior tokens，结合多层级语义监督将协同过滤信号对齐到冻结 LLM 的语义空间，实现零样本可解释推荐。

**[Bias Association Discovery Framework for Open-Ended LLM Generations](bias_association_discovery_framework_for_open-ended_llm_generations.md)**

:   提出偏见关联发现框架 BADF，通过分析 LLM 开放式故事生成中的叙事内容，系统性地提取人口统计身份与描述性概念之间的已知和未知偏见关联，突破了以往依赖预定义偏见概念的局限。

**[Can LLMs Truly Embody Human Personality? Analyzing AI and Human Behavior Alignment in Dispute Resolution](can_llms_truly_embody_human_personality_analyzing_ai_and_human_behavior_alignmen.md)**

:   提出首个系统对比框架，在配对的冲突调解场景中直接比较人类与人格提示LLM的策略行为差异，发现LLM在人格-行为映射上与人类存在显著偏差，挑战了"人格提示即可代理人类行为"的假设。

**[CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation](ccfqa_a_benchmark_for_cross-lingual_and_cross-modal_speech_and_text_factuality_e.md)**

:   提出 CCFQA——首个覆盖 8 种语言、14,400 条完全平行语音-文本事实问答样本的跨语言跨模态基准，支持 QA/XQA/SQA/XSQA 四种任务设定，系统揭示了现有 MLLM 在语言和模态切换下的事实不一致性；同时提出 LLM-SQA，以英语为桥接语言、仅 5-shot 即实现跨语言语音问答迁移，在 XSQA 上 F1 达 51.4 超越 GPT-4o-mini-Audio（45.7）。

**[CLIP-FTI: Fine-Grained Face Template Inversion via CLIP-Driven Attribute Conditioning](clip-fti_fine-grained_face_template_inversion_via_clip-driven_attribute_conditio.md)**

:   首次利用 CLIP 提取面部细粒度语义属性嵌入来辅助人脸模板反演（FTI），通过跨模态特征交互网络将泄露模板与属性嵌入融合并投影到 StyleGAN 潜空间，生成身份一致且属性细节更丰富的人脸图像，在识别准确率、属性相似度和跨模型攻击迁移性上均超越 SOTA。

**[CLIPPan: Adapting CLIP as A Supervisor for Unsupervised Pansharpening](clippan_adapting_clip_as_a_supervisor_for_unsupervised_pansharpening.md)**

:   提出 CLIPPan，通过轻量微调 CLIP 使其理解多光谱/全色/高分辨率多光谱图像类型及全色锐化过程，然后利用 Wald 协议等文本提示作为语义监督信号，实现无需地面真值的全分辨率无监督全色锐化，可作为即插即用模块兼容任意全色锐化骨干网络。

**[CoordAR: One-Reference 6D Pose Estimation of Novel Objects via Autoregressive Coordinate Map Generation](coordar_one-reference_6d_pose_estimation_of_novel_objects_via_autoregressive_coo.md)**

:   提出 CoordAR，将单参考视图 6D 位姿估计中的 3D-3D 对应关系建模为离散 token 的自回归生成问题，通过坐标图 token 化、模态解耦编码和自回归 Transformer 解码器，在多个基准上显著超越现有单视图方法，并对对称、遮挡等挑战场景展现强鲁棒性。

**[DEIG: Detail-Enhanced Instance Generation with Fine-Grained Semantic Control](deig_detail-enhanced_instance_generation_with_fine-grained_semantic_control.md)**

:   提出 DEIG，一个面向细粒度多实例图像生成的框架，通过实例细节提取器（IDE）将 LLM 编码器的高维嵌入蒸馏为紧凑的实例感知表示，并用细节融合模块（DFM）的实例掩码注意力防止属性泄漏，在多属性（颜色+材质+纹理）复合描述的生成任务上大幅超越现有方法。

**[Dexterous Manipulation Transfer via Progressive Kinematic-Dynamic Alignment](dexterous_manipulation_transfer_via_progressive_kinematic-dynamic_alignment.md)**

:   提出 PKDA 框架，通过渐进式运动学-动力学对齐，将人手操作视频自动转化为多指灵巧手的高质量操作轨迹，平均迁移成功率达 73%。

**[Distributionally Robust Online Markov Game with Linear Function Approximation](distributionally_robust_online_markov_game_with_linear_function_approximation.md)**

:   本文研究具有线性函数近似的在线分布鲁棒马尔可夫博弈，首次识别了该设定下的学习困难性，并提出 DR-CCE-LSI 算法，在特定特征映射条件下实现了关于特征维度 $d$ 的极小极大最优样本复杂度。

**[Efficient and Reliable Hitting-Set Computations for the Implicit Hitting Set Approach](efficient_and_reliable_hitting-set_computations_for_the_implicit_hitting_set_app.md)**

:   针对隐式击中集框架中击中集组件依赖商用IP求解器带来的数值不稳定问题，提出基于伪布尔推理和随机局部搜索的替代方案及混合策略，实现了首个可认证的IHS计算并在1786个基准实例上展示了效率与可靠性的有效权衡。

**[Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer](enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)**

:   提出预测驱动的 Top-K Jaccard 相似度系数提升邻居纯度，配合稀疏差分 Transformer（SDT）消除噪声注意力，在 MS-Celeb-1M 等大规模人脸聚类数据集上达到 SOTA 性能。

**[Enhancing Robustness of Offline RL Under Data Corruption via SAM](enhancing_robustness_of_offline_reinforcement_learning_under_data_corruption_via.md)**

:   首次将 Sharpness-Aware Minimization (SAM) 作为即插即用优化器应用于离线 RL，假设数据损坏导致损失景观中出现尖锐极小值从而泛化差，SAM 通过寻找平坦极小值提升鲁棒性，在 D4RL 基准上 IQL+SAM 平均得分从 34.47 提升到 44.40。

**[Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis](facial-r1_aligning_reasoning_and_recognition_for_facial_emotion_analysis.md)**

:   提出 Facial-R1，一个三阶段对齐训练框架（SFT → RL → 数据合成），通过将 AU 和情绪标签作为可验证奖励信号来对齐 VLM 的推理过程与情绪识别结果，在 8 个基准上达到 SOTA，并构建了 FEA-20K 数据集。

**[Failures to Surface Harmful Contents in Video Large Language Models](failures_to_surface_harmful_contents_in_video_large_language_models.md)**

:   本文首次系统分析了 VideoLLM 的安全性，揭示了三种结构性设计缺陷（稀疏时间采样、空间 token 下采样、模态融合不平衡），使得视频中清晰可见的有害内容在模型生成的文本摘要中被遗漏（omission rate 超 90%），并设计了三种零查询黑盒攻击来验证漏洞严重性。

**[Few-Shot Precise Event Spotting via Unified Multi-Entity Graph and Distillation](few-shot_precise_event_spotting_via_unified_multi-entity_graph_and_distillation.md)**

:   提出 UMEG-Net，面向少样本精确事件定位（PES）任务，通过构建统一多实体图（融合人体骨架、运动物体关键点和环境标志点），结合高效的时空图卷积和无参数多尺度时序平移模块，并通过多模态知识蒸馏将图特征迁移至 RGB 学生网络，在五个运动数据集上以极少标注数据显著超越现有方法。

**[Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)**

:   提出 DFF-Adapter（DeepFake Fine-Grained Adapter），针对 DINOv2 设计的轻量级深度伪造检测微调方案。通过在每个 Transformer 块中注入三分支适配器（真实性检测头、伪造类型分类头、共享头），结合 Forgery-Aware Multi-Head Router 实现子空间级 LoRA 专家动态路由，利用辅助的伪造类型分类任务增强主任务的伪影敏感性，仅 3.5M 可训练参数即在多个跨数据集评估中达到 SOTA。

**[FineXtrol: Controllable Motion Generation via Fine-Grained Text](finextrol_controllable_motion_generation_via_fine-grained_text.md)**

:   提出 FineXtrol 框架，利用带时间标注的细粒度身体部位文本描述作为控制信号，通过双分支 ControlNet 架构和层级对比学习增强文本编码器的区分能力，实现高效、用户友好且精确的可控人体动作生成，在 HumanML3D 上多身体部位控制性能显著优于现有方法。

**[From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization](from_ids_to_semantics_a_generative_framework_for_cross-domain_recommendation_wit.md)**

:   提出 GenCDR 框架，通过领域自适应语义分词和跨域自回归推荐两大模块，首次将生成式语义 ID 范式引入 LLM 驱动的跨域推荐，有效解决传统方法中 item ID 不可迁移和领域个性化建模不足的问题。

**[From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks](from_imitation_to_discrimination_toward_a_generalized_curriculum_advantage_mecha.md)**

:   提出 CAPO（Curriculum Advantage Policy Optimization），一种基于优势信号的自适应课程机制，通过先模仿（仅正向优势样本）再判别（引入负向信号）的两阶段策略，稳定且显著提升 LLM 在数学推理和多模态 GUI 推理任务上的表现。

**[GazeInterpreter: Parsing Eye Gaze to Generate Eye-Body-Coordinated Narrations](gazeinterpreter_parsing_eye_gaze_to_generate_eye-body-coordinated_narrations.md)**

:   提出 GazeInterpreter，一种基于 LLM 的层次化框架，通过符号化眼动解析器将原始注视信号转化为文本叙述，再与身体运动叙述整合生成眼-体协调描述，并通过自我纠正循环迭代优化，显著提升文本驱动的运动生成、动作预测和行为摘要等下游任务的性能。

**[Generating Attribute-Aware Human Motions from Textual Prompt](generating_attribute-aware_human_motions_from_textual_prompt.md)**

:   提出 AttrMoGen 框架，通过基于结构因果模型（SCM）的因果信息瓶颈将动作语义与人体属性（年龄、性别等）解耦，生成属性感知的人体运动，并构建了首个包含广泛属性标注的大规模文本-运动数据集 HumanAttr。

**[HiLoMix: Robust High- and Low-Frequency Graph Learning Framework for Mixing Address Association](hilomix_robust_high-_and_low-frequency_graph_learning_framework_for_mixing_addre.md)**

:   提出 HiLoMix，一种针对混币地址关联任务的鲁棒图学习框架，通过异质属性混合交互图（HAMIG）、频率感知图对比学习和基于置信度的标签加权监督学习，分别解决图稀疏、标签稀缺和标签噪声三大挑战，在 F1、AUC、MRR 上分别超越次优基线 5.69%、7.34% 和 15.61%。

**[Improving Sparse IMU-based Motion Capture with Motion Label Smoothing](improving_sparse_imu-based_motion_capture_with_motion_label_smoothing.md)**

:   提出 Motion Label Smoothing，将经典 label smoothing 从分类任务适配到稀疏IMU运动捕捉中，通过融合骨骼结构感知的Perlin噪声作为平滑标签，在不修改模型架构的前提下以即插即用方式提升三种SOTA方法在四个数据集上的精度，GlobalPose在TotalCapture上SIP误差降低20.41%。

**[IROTE: Human-like Traits Elicitation of Large Language Model via In-Context Self-Reflective Optimization](irote_human-like_traits_elicitation_of_large_language_model_via_in-context_self-.md)**

:   提出 IROTE，一种基于信息瓶颈理论的上下文自我反思优化方法，通过迭代生成并优化紧凑且富有唤起力的文本"自我反思"（self-reflection），无需微调即可稳定地激发 LLM 在多种下游任务中表现出目标人类特质（价值观、道德、人格），一致性超越现有基线。

**[MIDB: Multilingual Instruction Data Booster for Enhancing Cultural Equality in Multilingual Instruction Synthesis](midb_multilingual_instruction_data_booster_for_enhancing_cultural_equality_in_mu.md)**

:   提出 MIDB（多语言指令数据增强器），通过 36.8k 人类语言专家标注的修订样本训练一个统一模型，自动修复多语言合成指令数据中的内容错误、机器翻译缺陷和本地化不足问题，显著提升 16 种语言的指令数据质量和下游 LLM 的多语言/文化理解能力。

**[mmPred: Radar-based Human Motion Prediction in the Dark](mmpred_radar-based_human_motion_prediction_in_the_dark.md)**

:   首次将毫米波雷达引入人体运动预测(HMP)任务，提出mmPred——基于扩散模型的框架，通过双域历史运动表示（时域姿态细化TPR + 频域主导运动FDM）和全局骨骼关系Transformer(GST)，有效抑制雷达特有的噪声和时序不一致性，在mmBody和mm-Fi数据集上分别超越SOTA方法8.6%和22%。

**[Modality-Aware Bias Mitigation and Invariance Learning for Unsupervised Visible-Infrared Person Re-Identification](modality-aware_bias_mitigation_and_invariance_learning_for_unsupervised_visible-.md)**

:   针对无监督可见光-红外行人重识别（USVI-ReID）中跨模态关联不可靠的核心问题，提出模态感知的 Jaccard 距离修正和"分裂-对比"不变性学习策略，通过消除模态偏差实现可靠的全局跨模态聚类和特征对齐，在 SYSU-MM01 和 RegDB 上达到 SOTA。

**[Modelling the Effects of Hearing Loss on Neural Coding in the Auditory Midbrain with Variational Conditioning](modelling_the_effects_of_hearing_loss_on_neural_coding_in_the_auditory_midbrain_.md)**

:   提出 ψ-ICNet，一种变分条件深度神经网络模型，通过仅 6 个可学习的条件参数 ψ 来编码听力损失的效应，从真实神经活动记录中直接学习听力损失的低维表示空间，在预测正常和听力受损动物的听觉中脑神经响应方面达到与动物特定模型相当的精度，并可通过贝叶斯优化快速拟合未见过的新动物。

**[MVGD-Net: A Novel Motion-aware Video Glass Surface Detection Network](mvgd-net_a_novel_motion-aware_video_glass_surface_detection_network.md)**

:   基于"玻璃表面上反射/透射层物体的运动速度与非玻璃区域不一致"的物理观察，提出 MVGD-Net，通过光流运动线索引导视频中玻璃表面检测，包含跨尺度多模态融合（CMFM）、历史引导注意力（HGAM）、时序交叉注意力（TCAM）和时空解码器（TSD）四个核心模块，并构建了包含 312 视频 19,268 帧的大规模数据集 MVGD-D。

**[New Synthetic Goldmine: Hand Joint Angle-Driven EMG Data Generation Framework for Micro-Gesture Recognition](new_synthetic_goldmine_hand_joint_angle-driven_emg_data_generation_framework_for.md)**

:   提出 SeqEMG-GAN，一种基于手部关节角度序列驱动的条件对抗生成框架，通过角度编码器、双层上下文编码器（含新颖 Ang2Gist 单元）、深度卷积生成器和多视角判别器的联合设计，从关节运动学轨迹合成高保真 EMG 信号，实现对未见手势的零样本生成，合成数据与真实数据混合训练将分类精度从 57.77% 提升至 60.53%。

**[NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling](nurbgen_high-fidelity_text-to-cad_generation_through_llm-driven_nurbs_modeling.md)**

:   首次提出基于NURBS表面表示的文本到CAD生成框架NURBGen，通过微调LLM将自然语言描述转换为结构化的NURBS参数JSON，并引入混合表示（untrimmed NURBS + 解析原语）和大规模partABC数据集，在几何保真度和尺寸精度上显著超越现有方法。

**[OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval](opera_a_reinforcement_learning--enhanced_orchestrated_planner-executor_architect.md)**

:   提出 OPERA 框架，通过 Goal Planning Module 和 Reason-Execute Module 的分层架构，结合专为多 agent 设计的 MAPGRPO 训练算法，大幅提升 reasoning-oriented multi-hop retrieval 性能。

**[PA-FAS: Towards Interpretable and Generalizable Multimodal Face Anti-Spoofing via Path-Augmented Reinforcement Learning](pa-fas_towards_interpretable_and_generalizable_multimodal_face_anti-spoofing_via.md)**

:   提出PA-FAS框架，通过推理路径增强（Reasoning Path Augmentation）策略和答案打乱机制，解决了多模态FAS中SFT+RL范式的两大瓶颈（推理路径多样性不足和推理捷径问题），首次在统一框架中同时实现多模态融合、域泛化和可解释性。

**[ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)**

:   提出ParaRevSNN，一种并行可逆脉冲神经网络架构，通过重新设计可逆块间的数据依赖关系解耦顺序计算约束，在保持可逆性（内存高效）的同时实现块间并行，训练时间减少最多35.2%，推理时间降至18.15%。

**[Partially Shared Concept Bottleneck Models](partially_shared_concept_bottleneck_models.md)**

:   提出PS-CBM框架，通过多模态概念生成（结合LLM语义与示例图像视觉线索）、部分共享概念策略（基于激活模式合并概念）和Concept-Efficient Accuracy（CEA）评估指标，在11个数据集上以更少的概念实现了更高的分类精度和可解释性。

**[PathMind: A Retrieve-Prioritize-Reason Framework for Knowledge Graph Reasoning with Large Language Models](pathmind_a_retrieve-prioritize-reason_framework_for_knowledge_graph_reasoning_wi.md)**

:   提出PathMind框架，遵循"检索-优先级排序-推理"范式，通过语义感知的路径优先级函数（结合累积代价和估计未来代价）识别重要推理路径，再通过任务特定指令调优和路径级偏好对齐两阶段训练增强LLM的忠实可解释推理，在复杂推理任务上以更少token实现SOTA。

**[Personality-guided Public-Private Domain Disentangled Hypergraph-Former Network for Multimodal Depression Detection](personality-guided_public-private_domain_disentangled_hypergraph-former_network_.md)**

:   提出 P3HF 框架，通过人格引导的特征门控、时序感知的超图-Transformer（Hypergraph-Former）架构和事件级公私域解耦三大创新，在多事件多模态抑郁检测任务上实现约 10% 的准确率和 F1 提升。

**[Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)**

:   提出 Plug-and-Play Clarifier，一个零样本、模块化的多模态框架，将第一人称视角中的意图歧义问题分解为文本澄清、视觉质量评估和跨模态手势定位三个子任务，使 4-8B 小模型在意图消歧任务上提升约 30%，接近甚至超越大模型水平。

**[Radar-APLANC: Unsupervised Radar-based Heartbeat Sensing via Augmented Pseudo-Label and Noise Contrast](radar-aplanc_unsupervised_radar-based_heartbeat_sensing_via_augmented_pseudo-lab.md)**

:   提出首个雷达心跳感知的无监督学习框架 Radar-APLANC，通过噪声对比三元组损失（NCT loss）和增强伪标签生成器实现两阶段无监督训练，无需昂贵的生理信号标注即可达到接近监督方法的性能。

**[RAGFort: Dual-Path Defense Against Proprietary Knowledge Base Extraction in Retrieval-Augmented Generation](ragfort_dual-path_defense_against_proprietary_knowledge_base_extraction_in_retri.md)**

:   提出 RAGFort，首个系统性防御 RAG 知识库抽取攻击的双路径框架，通过对比重索引（inter-class）隔离主题间边界和约束级联生成（intra-class）抑制敏感内容输出，在安全性上将知识恢复率降低至无保护的 0.51×，同时保持回答质量。

**[REINA: Regularized Entropy Information-Based Loss for Efficient Simultaneous Speech Translation](reina_regularized_entropy_information-based_loss_for_efficient_simultaneous_spee.md)**

:   提出 REINA（Regularized Entropy INformation Adaptation）损失函数，基于互信息理论高效地将非流式语音翻译模型转换为流式同声传译模型，在多语言方向上达到 SOTA 流式翻译性能，并提出新的流式效率评估指标 NoSE。

**[Remember Me: Bridging the Long-Range Gap in LVLMs with Three-Step Inference-Only Decay Resilience Strategies](remember_me_bridging_the_long-range_gap_in_lvlms_with_three-step_inference-only_.md)**

:   提出 T-DRS（Three-step Decay Resilience Strategies），一个无需训练的推理时框架，通过语义驱动增强、距离感知控制和远距离重强化三个阶段协同缓解 RoPE 引起的长程注意力衰减，在 VQA 任务上持续提升多个 LVLM 的性能。

**[RENEW: Risk- and Energy-Aware Navigation in Dynamic Waterways](renew_risk-_and_energy-aware_navigation_in_dynamic_waterways.md)**

:   提出 RENEW 全局路径规划器，为水面自主航行器 (ASV) 在动态水流 (洋流) 环境中引入统一的风险感知和能量感知策略，通过自适应不可导航区域识别、最佳努力应急策略和基于约束 Delaunay 三角化的分层架构实现安全高效导航，应急碰撞测试中实现零碰撞。

**[Renormalization Group Guided Tensor Network Structure Search](renormalization_group_guided_tensor_network_structure_search.md)**

:   提出 RGTN 框架，将统计物理中的重正化群（Renormalization Group）理论引入张量网络结构搜索，通过多尺度粗粒化-扩展-压缩流程和可学习边门控实现连续拓扑演化，在光场压缩、高阶张量分解和视频补全任务上达到 SOTA 压缩率，同时比已有方法快 4–600 倍。

**[Self-Correction Distillation for Structured Data Question Answering](self-correction_distillation_for_structured_data_question_answering.md)**

:   提出自纠正蒸馏（SCD）方法，通过错误提示机制（EPM）和两阶段蒸馏策略，将大规模LLM（GPT4）的结构化数据问答能力高效迁移到小规模LLM（8B），在5个基准上取得最优蒸馏性能。

**[Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning](small_language_models_for_efficient_agentic_tool_calling_outperforming_large_mod.md)**

:   通过对OPT-350M模型进行单epoch的SFT微调，在ToolBench评估中取得77.55%的通过率，大幅超越ChatGPT-CoT（26%）、ToolLLaMA-DFS（30.18%）等大模型基线，证明了针对性微调的小模型在特定任务上可显著超越通用大模型。

**[SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control](soscontrol_enhancing_human_motion_generation_through_saliency-aware_symbolic_ori.md)**

:   提出Salient Orientation Symbolic (SOS) script——基于Labanotation启发的可编程符号化运动表示框架，通过时序约束的凝聚聚类提取关键帧显著性，结合SMS数据增强和梯度优化的SOSControl框架实现对身体部位朝向和运动时序的精确控制，在HumanML3D上SOS-Acc达0.988且FID仅3.892。

**[Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction](spatiotemporal-untrammelled_mixture_of_experts_for_multi-person_motion_predictio.md)**

:   提出ST-MoE框架，首次将混合专家模型（MoE）与双向时空Mamba相结合用于多人运动预测，通过四种异构时空专家灵活捕获复杂时空依赖，实现SOTA精度的同时减少41.38%参数量，训练加速3.6倍。

**[Streaming Generation of Co-Speech Gestures via Accelerated Rolling Diffusion](streaming_generation_of_co-speech_gestures_via_accelerated_rolling_diffusion.md)**

:   提出基于 Rolling Diffusion 的流式共语手势生成框架，通过结构化渐进噪声调度将任意扩散模型转化为流式手势生成器，并引入 Rolling Diffusion Ladder Acceleration (RDLA) 实现最高 4× 加速（200 FPS），在 ZEGGS 和 BEAT 基准上全面超越基线。

**[TalkSketch: Multimodal Generative AI for Real-time Sketch Ideation with Speech](talksketch_multimodal_generative_ai_for_real-time_sketch_ideation_with_speech.md)**

:   提出TalkSketch系统，将手绘草图与实时语音输入相结合，嵌入多模态AI聊天机器人，使设计师在早期构思阶段能够边画边说、流畅地与AI协作，解决了现有GenAI工具中文字提示打断创作流程的问题。

**[Theory of Mind for Explainable Human-Robot Interaction](theory_of_mind_for_explainable_human-robot_interaction.md)**

:   提出将心智理论（ToM）视为可解释AI（XAI）的一种形式，使用VXAI框架的七个评价标准系统评估现有HRI中的ToM研究，发现关键缺陷（特别是忠实度缺失），并主张将ToM整合到XAI框架中以实现用户导向的解释。

**[Towards a Common Framework for Autoformalization](towards_a_common_framework_for_autoformalization.md)**

:   本文系统回顾了"自动形式化"（autoformalization）在数学证明、逻辑推理、规划和知识表示等领域的现有工作，提出了一个统一的跨学科定义框架，将自动形式化定义为从非形式语言到形式推理语言的语义等价转换，旨在促进不同研究社区间的方法共享并加速下一代 AI 推理系统的发展。

**[Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation](transferable_backdoor_attacks_for_code_models_via_sharpness-aware_adversarial_pe.md)**

:   提出 STAB（Sharpness-aware Transferable Adversarial Backdoor），通过 SAM 训练代理模型使其收敛到损失平面的平坦区域，并使用 Gumbel-Softmax 优化生成上下文感知的对抗触发器，首次实现了同时兼顾跨数据集迁移性和隐蔽性的代码模型后门攻击。

**[UniFit: Towards Universal Virtual Try-on with MLLM-Guided Semantic Alignment](unifit_towards_universal_virtual_try-on_with_mllm-guided_semantic_alignment.md)**

:   提出 UniFit，一个由多模态大语言模型（MLLM）驱动的通用虚拟试穿框架，通过 MLLM 引导的语义对齐模块（MGSA）桥接文本指令与参考图像之间的语义鸿沟，并通过两阶段渐进训练+自合成流水线克服复杂场景的数据稀缺问题，首次在单一框架内支持 6 种 VTON 任务。

**[VPHO: Joint Visual-Physical Cue Learning and Aggregation for Hand-Object Pose Estimation](vpho_joint_visual-physical_cue_learning_and_aggregation_for_hand-object_pose_est.md)**

:   提出 VPHO，一个联合视觉和物理线索的手-物体姿态估计框架，通过力预测模块学习 3D 物理线索，并设计两阶段候选姿态聚合策略（视觉引导 + 物理引导），在保持视觉一致性的同时实现物理合理性，在 DexYCB 和 HO3D 两个基准上同时达到姿态精度和物理合理性的 SOTA。

**[W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search](w2s-aligntree_weak-to-strong_inference-time_alignment_for_large_language_models_.md)**

:   提出 W2S-AlignTree，首个将蒙特卡洛树搜索（MCTS）与弱到强泛化（W2SG）范式结合的推理时对齐框架，利用弱模型的步级代理值函数实时引导强模型生成，在情感控制、摘要、指令遵循任务上均显著超越基线，其中 Llama3-8B 摘要任务提升 15.9%。

**[Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study](why_do_open-source_llms_struggle_with_data_analysis_a_systematic_empirical_study.md)**

:   系统研究了开源 LLM 在数据分析任务中的能力瓶颈，将数据分析分解为数据理解、代码生成和战略规划三个维度，发现**战略规划是决定性因素**而非编码或数据理解；并提出了一种策略引导的数据合成方法，使微调后的 7B/14B 模型达到与 GPT-4o 竞争的性能。

**[XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs](xlinear_a_lightweight_and_accurate_mlp-based_model_for_long-term_time_series_for.md)**

:   提出 XLinear，一个基于 MLP + sigmoid gating 的轻量时间序列预测模型，通过 global token 机制高效融合 endogenous 与 exogenous 变量信息，在 12 个数据集上实现精度与效率的最优平衡。

**[Yes FLoReNce, I Will Do Better Next Time! Agentic Feedback Reasoning for Humorous Meme Detection](yes_florence_i_will_do_better_next_time_agentic_feedback_reasoning_for_humorous_.md)**

:   提出 FLoReNce 框架，将幽默 meme 理解建模为闭环控制系统，通过 Judge 反馈+PID 控制器+非参数知识库的闭环学习，在推理时通过检索相似经验调制 prompt，使冻结的 VLM 实现自适应推理，无需微调即可显著提升预测和解释质量。
