---
title: >-
  CVPR2026 人体理解方向 69篇论文解读
description: >-
  69篇CVPR2026 人体理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**📷 CVPR2026** · **69** 篇论文解读

**[All in One: Unifying Deepfake Detection, Tampering Localization, and Source Tracing with a Robust Landmark-Identity Watermark](all_in_one_unifying_deepfake_detection_tampering_localization_and_source_tracing.md)**

:   提出 LIDMark，首个将 deepfake 检测、篡改区域定位和源追踪统一到单一主动取证框架中的方法——通过嵌入 152 维 Landmark-Identity 水印（136D 面部关键点 + 16D 源 ID），利用内在/外在一致性实现三合一取证，PSNR/SSIM 和检测精度均超越现有方法。

**[AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video](avatar_reinforcement_learning_to_see_hear_and_reason_over_video.md)**

:   提出AVATAR框架，通过离线策略训练架构（分层回放缓冲区）和时间优势塑造(TAS)策略解决GRPO在多模态视频推理中的数据低效、优势消失和均匀信用分配三大问题，在音视频理解基准上显著超越标准GRPO（OmniBench +3.7，样本效率提升5倍）。

**[Bilevel Layer-Positioning LoRA for Real Image Dehazing](bilevel_layer-positioning_lora_for_real_image_dehazing.md)**

:   提出 BiLaLoRA，通过双层优化自动定位 LoRA 应插入的最优网络层，配合 H2C Loss（基于 CLIP 语义方向的无监督去雾损失），实现合成数据预训练的去雾模型向真实场景的高效适配——训练时间降低 77.7%，性能持平全量微调，跨模型跨域均有效。

**[Bilevel Layer-Positioning LoRA for Real Image Dehazing](bilevel_lora_real_image_dehazing.md)**

:   利用CLIP跨模态能力将去雾重构为语义对齐问题（H2C损失），并通过双层优化自动搜索最佳LoRA注入层（BiLaLoRA），实现即插即用的高效合成到真实域去雾适配。

**[BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration](bipremanip_learning_affordance-based_bimanual_preparatory_manipulation_through_a.md)**

:   提出 BiPreManip 框架，基于视觉可供性表示实现双臂预备操作：先预想主手的目标交互区域，再引导辅助手进行预备动作（如翻转瓶子使瓶盖朝向主手），在仿真和真实环境中大幅优于基线。

**[Breaking the Tuning Barrier: Zero-Hyperparameters Yield Multi-Corner Analysis Via Learned Priors](breaking_the_tuning_barrier_zero-hyperparameters_yield_multi-corner_analysis_via.md)**

:   提出基于 Learned Priors（TabPFN 基础模型）的零超参良率多角分析框架，通过 in-context Bayesian 推断替代传统 GP/normalizing flow 的超参调优，结合自动特征选择、Cross-Corner 知识迁移和不确定性驱动主动学习，MRE 低至 0.11% 且完全免调参，验证成本降低 10× 以上。

**[Breaking the Tuning Barrier: Zero-Hyperparameters Yield Multi-Corner Analysis Via Learned Priors](breaking_the_tuning_barrier_zerohyperparameters_yi.md)**

:   提出用基础模型 TabPFN 的 learned prior 替代传统人工先验（GP 核、IS 高斯假设），实现零超参数调优的多 PVT Corner 良率分析，在工业级 SRAM 基准上达到 SOTA 精度（MRE 低至 0.11%）的同时提速超 10×。

**[CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation](cigpose_causal_intervention_graph_neural_network_for_whole-body_pose_estimation.md)**

:   提出因果干预图姿态估计框架 CIGPose，通过结构因果模型识别视觉上下文混杂因素，利用预测不确定性定位受混杂影响的关键点并用学习得到的上下文无关规范嵌入替换，再经层次图神经网络建模骨骼解剖约束，在 COCO-WholeBody 上达到 67.0% AP 的新 SOTA。

**[COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)**

:   提出 COG 框架，将跨视图对应关系建模为置信度感知的最优传输(OT)问题，通过预测逐点置信度作为传输边际约束来抑制非重叠区域和离群点，实现无监督条件下媲美有监督方法的单参考图像新物体6DoF位姿估计。

**[CraterBench-R: Instance-Level Crater Retrieval for Planetary Scale](craterbench-r_instance-level_crater_retrieval_for_planetary_scale.md)**

:   首次将陨石坑分析形式化为实例级图像检索问题——提出CraterBench-R基准(~25K火星陨石坑ID, 50K gallery, 5K查询)，诊断发现单向量池化有精度上限+有监督度量学习反而退化，提出无训练的实例token聚合(选K个种子+余弦最近邻残差分配)将196个ViT patch token压缩为K个代表token做late interaction匹配，K=64时匹配全token精度且存储大幅降低，实用两阶段管线(单向量粗筛+实例token精排)恢复89-94%完整精度。

**[AdvMark: Decoupling Defense Strategies for Robust Image Watermarking](decoupling_defense_strategies_for_robust_image_watermarking.md)**

:   提出 AdvMark 两阶段解耦防御框架：Stage 1 Encoder Adversarial Training（EAT）将水印图像移入 non-attackable 区域抵御对抗攻击，Stage 2 直接图像优化抵御失真+再生攻击并保留对抗鲁棒性，在 9 种水印方法 ×10 种攻击上分别提升失真/再生/对抗准确率 29%/33%/46%，且图像质量最优。

**[DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)**

:   提出 DecoVLN 框架，将 VLN 任务中的观察、推理和纠错三个过程解耦，通过自适应记忆优化机制和基于状态-动作对的纠错微调策略，在仅使用自中心 RGB 输入的条件下实现了 R2R-CE 和 RxR-CE 上的 SOTA 性能。

**[E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)**

:   提出 E-3DPSM，一种基于事件相机的自我中心 3D 人体姿态状态机，将姿态估计建模为连续时间状态演化过程，通过双向 SSM 时序建模和可学习的卡尔曼式融合模块融合直接预测与增量预测，实现 80Hz 实时推理，MPJPE 降低 19%、时序稳定性提升 2.7 倍。

**[EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR](egoposeformer_v2_accurate_egocentric_human_motion_estimation_for_arvr.md)**

:   提出 EgoPoseFormer v2 (EPFv2)，通过端到端 Transformer 架构（单一全局查询 + 因果时序注意力 + 条件多视图交叉注意力）和基于不确定性蒸馏的自动标注系统，在 EgoBody3M 基准上以 0.8ms GPU 延迟实现了自我中心 3D 人体运动估计的 SOTA 精度（MPJPE 4.02cm，比前作提升 15-22%）。

**[Face Time Traveller: Travel Through Ages Without Losing Identity](face_time_traveller_travel_through_ages_without_losing_identity.md)**

:   提出 FaceTT 框架，通过面部属性感知提示词精炼、角度反演和自适应注意力控制三大模块，实现高保真、身份一致的人脸年龄变换，在多个基准上超越现有方法。

**[FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)**

:   提出域感知原型联邦学习框架 FedDAP，通过构建域特定全局原型和双重原型对齐策略（域内对齐 + 跨域对比），解决联邦学习中客户端数据域偏移导致的全局模型性能退化问题。

**[FlexAvatar: Learning Complete 3D Head Avatars with Partial Supervision](flexavatar_learning_complete_3d_head_avatars_with_partial_supervision.md)**

:   提出 FlexAvatar，通过引入可学习的"偏置吸收器"（bias sinks）token 统一单目和多视角数据训练，解决了驱动信号与目标视角的纠缠问题，从单张图像生成完整、高质量、可动画的 3D 头部化身。

**[FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation](fozo_forward-only_zeroth-order_prompt_optimization_for_test-time_adaptation.md)**

:   提出 FOZO，一种仅需前向传播的零阶 prompt 优化范式，通过 SPSA 梯度估计 + 动态扰动策略 + 深浅层特征统计对齐，在不修改模型权重的情况下实现高效 TTA，在 ImageNet-C 上以 59.52% 准确率超越所有前向方法（含 FOA 58.13%），并支持 INT8 量化模型。

**[A2P: From 2D Alignment to 3D Plausibility for Occlusion-Robust Two-Hand Reconstruction](from_2d_alignment_to_3d_plausibility_unifying_hete.md)**

:   解耦双手重建为 2D 结构对齐 + 3D 空间交互对齐：Stage 1 用 Fusion Alignment Encoder 隐式蒸馏 Sapiens 的关键点/分割/深度三种 2D 先验（推理时免基础模型，56fps），Stage 2 用穿透感知扩散模型 + 碰撞梯度引导将穿透姿态映射到物理合理配置——InterHand2.6M 上 MPJPE 降至 5.36mm（超 SOTA 4DHands 2.13mm），穿透体积降 7 倍。

**[GardenDesigner: Encoding Aesthetic Principles into Jiangnan Garden Construction via a Chain of Agents](gardendesigner_encoding_aesthetic_principles_into_jiangnan_garden_construction_v.md)**

:   提出 GardenDesigner 框架，通过链式智能体（地形分布→道路生成→资产选择→布局优化）将江南园林的美学原则编码为可计算的约束，结合专家标注的 GardenVerse 数据集，实现非专业用户通过文本输入在一分钟内自动构建符合美学规范的江南园林。

**[GeoWorld: Geometric World Models](geoworld_geometric_world_models.md)**

:   GeoWorld 将预测式世界模型的潜在表征从欧氏空间映射到双曲流形上，通过 Hyperbolic JEPA 保持几何结构和层级关系，并提出 Geometric Reinforcement Learning 来优化多步规划，在 CrossTask 和 COIN 上实现了约 3% SR（3步）和 2% SR（4步）的提升。

**[Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](graph2eval_automatic_multimodal_task_generation_for_agents_via_knowledge_graphs.md)**

:   提出 Graph2Eval，一个知识图谱驱动的 agent 评估任务自动生成框架——通过从文档/网页构建结构化知识图谱、子图采样、LLM 条件生成和多阶段过滤，自动产出语义一致（+20%）且可解（+17%）的多模态 agent 任务，构建了包含 1319 个任务的 Graph2Eval-Bench。

**[Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](graph2eval_multimodal_task_generation_agents.md)**

:   提出 Graph2Eval，利用从异构数据源构建的知识图谱作为结构化任务空间，通过子图采样、任务模板和 meta-path 策略自动生成语义一致且可解的多模态 agent 评估任务，生成的任务在语义一致性和可解性上分别提升 20% 和 17%。

**[GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks](guide_a_benchmark_for_understanding_and_assisting_users_in_open-ended_gui_tasks.md)**

:   本文提出 GUIDE 基准，包含 120 个新手用户在 10 款软件上的 67.5 小时屏幕录像和出声思维标注，定义了行为状态检测、意图预测、协助预测三个分层任务，评估发现当前最强多模态模型在理解用户行为和判断协助需求上表现有限（行为检测仅 44.6% 准确率），但提供结构化用户上下文可显著提升性能（协助预测最高提升 50.2pp）。

**[HandX: Scaling Bimanual Motion and Interaction Generation](handx_scaling_bimanual_motion_and_interaction_generation.md)**

:   构建了 HandX——一个统一的双手运动生成基础设施（包含 54.2 小时运动数据 + 48.5 万条细粒度文本标注），提出解耦式自动标注策略（运动学特征提取 + LLM 推理生成描述），并基准测试了扩散和自回归两种生成范式，展示了明确的数据和模型 scaling 趋势。

**[IDperturb: Enhancing Variation in Synthetic Face Generation via Angular Perturbations](idperturb_enhancing_variation_in_synthetic_face_generation_via_angular_perturbat.md)**

:   提出 IDperturb，一种在单位超球面上对身份嵌入进行角度扰动的几何采样策略，无需修改生成模型即可显著增强合成人脸数据集的类内多样性，提升下游人脸识别性能。

**[L2GTX: From Local to Global Time Series Explanations](l2gtx_from_local_to_global_time_series_explanation.md)**

:   提出 L2GTX——完全模型无关的局部到全局时间序列解释方法，以参数化事件原语(递增/递减趋势、局部极值)为解释单元，经层次聚类合并、贪心预算选择和属性统计聚合，在 6 个 UCR 数据集上生成紧凑忠实的类级全局解释(FCN上ECG200 GF=0.792)。

**[L2GTX: From Local to Global Time Series Explanations](l2gtx_from_local_to_global_time_series_explanations.md)**

:   L2GTX 提出一种完全模型无关的局部到全局解释方法，通过从 LOMATCE 局部解释中提取参数化时间事件原语（趋势/极值），跨实例合并冗余聚类并以子模优化选取代表性实例，最终聚合为简洁的类级别全局解释，在6个时序分类数据集上保持稳定的全局忠实度。

**[LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference](lamogen_language_to_motion_generation_through_llm-guided_symbolic_inference.md)**

:   提出 LabanLite 符号动作表示和 LaMoGen 框架，首次让 LLM 通过可解释的 Laban 符号推理自主组合动作序列，在时序精度和可控性上超越传统文本-动作联合嵌入方法。

**[LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction](laser_layer-wise_scale_alignment_for_training-free_streaming_4d_reconstruction.md)**

:   提出 LASER，一个无需重训练的框架，通过层级深度尺度对齐（Layer-wise Scale Alignment）将离线前馈重建模型（如 VGGT、π³）转换为流式系统，在 RTX A6000 上以 14 FPS、6GB 峰值显存实现千米级视频的实时流式 4D 重建。

**[MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision](matched_crisp_edge_detection_using_end-to-end_matching-based_supervision.md)**

:   MatchED 提出一种轻量（约21K参数）plug-and-play 模块，通过在训练时对预测边缘和 GT 边缘进行基于空间距离+置信度的 one-to-one 二部匹配来生成 crisp（单像素宽）边缘图，可附加到任何边缘检测器端到端训练，首次在不依赖 NMS+thinning 后处理的情况下匹配或超越标准后处理方法。

**[Miburi: Towards Expressive Interactive Gesture Synthesis](miburi_towards_expressive_interactive_gesture_synthesis.md)**

:   提出 Miburi，首个在线因果框架，通过直接利用语音-文本大模型 Moshi 的内部 token 流和二维因果 Transformer，实现实时同步的全身手势与面部表情生成。

**[Mobile-VTON: High-Fidelity On-Device Virtual Try-On](mobile-vton_high-fidelity_on-device_virtual_try-on.md)**

:   提出 Mobile-VTON，首个可完全在移动设备上离线运行的扩散模型虚拟试穿系统，通过 TeacherNet-GarmentNet-TryonNet（TGT）架构和特征引导对抗蒸馏策略，以 415M 参数和 2.84GB 显存实现媲美服务器端基线的高质量试穿效果。

**[Mobile-VTON: High-Fidelity On-Device Virtual Try-On](mobile_vton_ondevice_virtual_tryon.md)**

:   首个全离线移动端扩散式虚拟试穿框架，基于TeacherNet-GarmentNet-TryonNet (TGT)架构，通过特征引导对抗蒸馏(FGA)将SD3.5 Large的能力迁移到415M参数的轻量学生网络，在VITON-HD和DressCode上以1024×768分辨率匹配甚至超越服务器端基线，端到端推理时间约80秒（小米17 Pro Max）。

**[MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation](molingo_motion-language_alignment_for_text-to-motion_generation.md)**

:   MoLingo 通过语义对齐的运动自编码器（SAE）和多 token 交叉注意力文本条件注入，在连续潜空间上执行 masked 自回归 rectified flow，在文本到人体动作生成任务上取得了 FID、R-Precision 和用户研究的全面 SOTA。

**[OMG-Bench: A New Challenging Benchmark for Skeleton-based Online Micro Hand Gesture Recognition](omg-bench_a_new_challenging_benchmark_for_skeleton-based_online_micro_hand_gestu.md)**

:   本文构建了首个大规模公开的基于骨骼数据的在线微手势识别基准OMG-Bench（40类、13948个实例），并提出HMATr框架，通过层次化记忆库和位置感知查询实现检测-分类的端到端统一，在检测率上超越SOTA方法7.6%。

**[OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis](openfs_multi-hand-capable_fingerspelling_recognition_with_implicit_signing-hand_.md)**

:   提出 OpenFS 框架，通过双层位置编码 + 签名手聚焦损失 + 单调对齐损失实现隐式签名手检测的多手指拼识别，并设计帧级字母条件扩散生成器合成 OOV 数据，在 ChicagoFSWild/ChicagoFSWildPlus/FSNeo 三个基准上取得 SOTA，推理速度比 PoseNet 快 100 倍以上。

**[PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery](pad-hand_physics-aware_diffusion_for_hand_motion_recovery.md)**

:   提出 PAD-Hand，一个物理感知的条件扩散框架，将欧拉-拉格朗日动力学残差建模为虚拟观测量融入扩散过程，同时通过最后一层拉普拉斯近似估计逐关节、逐帧的动态方差，实现了兼具物理可信度和不确定性感知的手部运动恢复，在 DexYCB 上加速度误差降低 50.1%。

**[ParTY: Part-Guidance for Expressive Text-to-Motion Synthesis](party_part-guidance_for_expressive_text-to-motion_synthesis.md)**

:   提出 ParTY 框架，通过部位引导网络（Part-Guided Network）和部位感知文本对齐（Part-aware Text Grounding），在保持全身动作连贯性的同时大幅提升身体各部位的文本-动作语义对齐精度，解决了现有整体式方法与部位拆分方法之间"部位表达力 vs 全身连贯性"的根本矛盾。

**[PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors for Embodied Multi-Sensory Learning](pulse_privileged_knowledge_transfer_from_rich_to_deployable_sensors_for_embodied.md)**

:   本文提出 PULSE 框架，通过冻结的特权传感器（如 EDA）教师模型向廉价可部署传感器（如 ECG、BVP、加速度计）学生模型进行知识蒸馏，引入共享-私有嵌入分解和重建防崩塌机制，在不使用 EDA 推理的情况下达到 0.994 AUROC 的压力检测性能，甚至超越使用全部传感器的模型。

**[QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)**

:   提出 QuantVLA，首个面向 Vision-Language-Action (VLA) 模型的免训练后量化框架，通过选择性量化布局和两个轻量级标定机制（注意力温度匹配 ATM 和输出头平衡 OHB），在 W4A8 精度下实现约 70% 的内存节省，同时任务成功率超过全精度基线。

**[RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces](recovermark_robust_watermarking_for_localization_and_recovery_of_manipulated_fac.md)**

:   提出 RecoverMark，一个将人脸内容本身作为水印嵌入背景的鲁棒水印框架，同时实现篡改区域定位、原始内容恢复和版权验证，在水印移除攻击下仍保持有效。

**[Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)**

:   提出 VTON-IQA，一个无需参考图的虚拟试穿图像质量评估框架，通过构建 62,688 张试穿图像 × 431,800 条人工标注的大规模基准 VTON-QBench，以及交错式交叉注意力（ICA）模块建模服装-人物-试穿图之间的交互关系，实现与人类感知高度对齐的图像级质量预测。

**[Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](referencefree_image_quality_assessment_for_virtual.md)**

:   构建 VTON-QBench（62,688 张试穿图像、13,838 名合格标注者、431,800 条标注）并提出 VTON-IQA 无参考质量评估框架，通过非对称交错交叉注意力（ICA）模块联合建模服装保真度和人物保持度，实现与人类感知高度对齐的图像级质量预测。

**[RefTon: Reference Person Shot Assist Virtual Try-on](refton_reference_person_shot_assist_virtual_try-on.md)**

:   本文提出 RefTon，一个基于 Flux-Kontext 的人对人虚拟试穿框架，通过引入额外参考图像（其他人穿着目标服装的照片）来提供更准确的服装细节信息，同时通过两阶段训练策略和缩放位置索引机制实现了无需辅助条件（如 DensePose、分割掩码）的端到端试穿，在 VITON-HD 和 DressCode 上达到 SOTA。

**[ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data](remogen_real-time_human_interaction-to-reaction_generation_via_modular_learning_.md)**

:   提出 ReMoGen，一个模块化框架用于实时人体交互-到-反应的动作生成：利用大规模单人运动数据学习通用运动先验（冻结），通过独立训练的 Meta-Interaction 模块适配不同交互域（人-人/人-场景），并引入 Frame-wise Segment Refinement 实现逐帧低延迟在线更新（0.047s/帧），在 Inter-X 和 LINGO 数据集上全面超越 SOTA。

**[Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)**

:   提出 CBM-Suite 框架，系统性解决概念瓶颈模型的四大缺陷——缺乏概念相关性预评估指标、线性问题导致概念瓶颈被绕过、与黑盒模型的精度差距、以及不同视觉骨干/VLM 影响的研究空白——通过熵度量、非线性层和蒸馏损失显著提升 CBM 的精度与可解释性。

**[SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](save_speech-aware_video_representation_learning_for_video-text_retrieval.md)**

:   提出 SAVE 方法，通过添加专用语音分支（Whisper ASR + CLIP 文本编码器）和 soft-ALBEF 视觉-音频早期对齐策略，实现语音感知的视频表示学习，在五个视频-文本检索基准上全面超越 SOTA。

**[See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles](see_think_act_teaching_multimodal_agents_to_effectively_interact_with_gui_by_ide.md)**

:   提出 State-aware Reasoning (StaR)，通过教会多模态 Agent "感知当前状态→分析目标状态→决定是否操作"的三步推理链，将 GUI 开关控制准确率提升超 30%，同时不损害通用 Agent 任务性能。

**[Seeing without Pixels: Perception from Camera Trajectories](seeing_without_pixels_perception_from_camera_trajectories.md)**

:   本文首次系统性地将相机位姿轨迹（6DoF pose sequence）提升为一种独立的视频感知模态，通过对比学习框架训练轻量级 Transformer 编码器 CamFormer，将相机轨迹映射到与文本对齐的联合嵌入空间，在 5 个数据集的 10 个下游任务上证明相机轨迹是既轻量又鲁棒的视频内容信号——在物理活动上甚至可以超越计算量大数千倍的视频模型。

**[Sketch2Colab: Sketch-Conditioned Multi-Human Animation via Controllable Flow Distillation](sketch2colab_sketch-conditioned_multi-human_animation_via_controllable_flow_dist.md)**

:   提出 Sketch2Colab，通过将草图驱动的扩散先验蒸馏为整流流学生网络，结合能量引导和连续时间马尔可夫链（CTMC）离散事件规划，从故事板草图生成协调的多人-物体交互 3D 动作，在 CORE4D 和 InterHuman 上实现 SOTA 约束遵从度和感知质量。

**[Stable Spike: Dual Consistency Optimization via Bitwise AND Operations for Spiking Neural Networks](stable_spike_dual_consistency_optimization_via_bitwise_and_operations_for_spikin.md)**

:   提出 Stable Spike 双一致性优化框架，利用硬件友好的 AND 位运算从多时间步脉冲图中解耦稳定脉冲骨架，并注入振幅感知脉冲噪声增强泛化，在超低延迟(T=2)下将神经形态物体识别精度提升最高 8.33%。

**[SteelDefectX: A Coarse-to-Fine Vision-Language Dataset and Benchmark for Generalizable Steel Surface Defect Detection](steeldefectx_a_coarse-to-fine_vision-language_dataset_and_benchmark_for_generali.md)**

:   提出 SteelDefectX，首个面向钢材表面缺陷检测的视觉-语言数据集（7778 张图像、25 类缺陷），包含从类级到样本级的粗到细文本标注，并建立了涵盖纯视觉分类、视觉-语言分类、零/少样本识别和零样本迁移的四任务基准，实验证明高质量文本标注显著提升模型的可解释性、泛化性和跨域迁移能力。

**[SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport](subflot_submodel_extraction_for_efficient_and_personalized_federated_learning_vi.md)**

:   提出 SubFLOT 框架，在服务器端利用最优传输（Optimal Transport）将全局模型的参数分布与客户端历史模型对齐，实现无需访问原始数据的个性化剪枝，并通过自适应正则化抑制剪枝导致的参数偏移，在多个数据集上大幅超越现有联邦剪枝方法。

**[Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach](team_ras_in_10th_abaw_competition_multimodal_valen.md)**

:   首次将 VLM（Qwen3-VL-4B-Instruct）提取的情感行为描述嵌入作为独立第三模态，与 GRADA 人脸编码器和 WavLM 音频特征通过 DCMMOE 和 RAAV 两种融合策略组合，在 Aff-Wild2 上达到连续 VA 估计 CCC 0.658（dev）/ 0.62（test），验证了 VLM 行为语义对连续情感识别的价值。

**[Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach](team_ras_in_10th_abaw_competition_multimodal_valence_and_arousal_estimation_appr.md)**

:   提出一种结合人脸视觉特征、VLM行为描述嵌入和音频特征的多模态方法用于连续效价-唤醒（VA）估计，通过两种融合策略（DCMMOE 和 RAAV）在 Aff-Wild2 数据集上取得了竞争力的结果。

**[TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size](teamhoi_learning_a_unified_policy_for_cooperative_human-object_interactions_with.md)**

:   提出 TeamHOI 框架，通过基于 Transformer 的去中心化策略网络和掩码对抗运动先验（Masked AMP），使单一策略能够泛化到任意数量智能体的协作搬运任务，2-8 个仿人智能体协作搬桌子成功率达 97%+。

**[4DSurf: High-Fidelity Dynamic Scene Surface Reconstruction](textit4dsurf_high-fidelity_dynamic_scene_surface_reconstruction.md)**

:   本文提出 4DSurf，一个基于2D高斯泼溅的通用动态场景表面重建框架，通过引入高斯运动诱导的SDF流正则化来约束表面时序一致演化，并采用重叠分段策略处理大变形，在 Hi4D 和 CMU Panoptic 数据集上分别以 49% 和 19% 的 Chamfer 距离改进超越现有 SOTA。

**[Towards Source-Aware Object Swapping with Initial Noise Perturbation](towards_source-aware_object_swapping_with_initial_noise_perturbation.md)**

:   提出 SourceSwap，通过频率分离的初始噪声扰动从单张图像生成高质量伪配对数据，并采用源感知双 U-Net 架构学习跨物体对齐，实现零样本、无逐物体微调的高保真物体替换。

**[Training High-Level Schedulers with Execution-Feedback Reinforcement Learning for Long-Horizon GUI Automation](training_high-level_schedulers_with_execution-feedback_reinforcement_learning_fo.md)**

:   提出 CES（Coordinator-Executor-State Tracker）多智能体框架和分阶段执行反馈强化学习算法，将高层任务规划与低层执行解耦，通过专门训练的 Coordinator 和 State Tracker 显著提升 GUI Agent 在长时序任务上的规划和状态管理能力。

**[TriLite: Efficient WSOL with Universal Visual Features and Tri-Region Disentanglement](trilite_efficient_weakly_supervised_object_localization_with_universal_visual_fe.md)**

:   仅使用冻结 DINOv2 ViT + 不到 800K 可训练参数的 TriHead 模块，通过将 patch 特征解耦为前景/背景/模糊三区域并引入对抗性背景损失，在 WSOL 上以极少参数刷新 SOTA。

**[UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos](unidex_a_robot_foundation_suite_for_universal_dexterous_hand_control_from_egocen.md)**

:   提出UniDex机器人基础套件——包含跨8种灵巧手的大规模数据集（50K+轨迹/9M帧）、功能-执行器对齐的统一动作空间（FAAS）和3D VLA策略（UniDex-VLA），在真实世界工具使用任务上达到81%平均任务进度（vs π₀的38%），并展示了空间、物体和零样本跨手泛化能力。

**[Unified Primitive Proxies for Structured Shape Completion](unified_primitive_proxies_for_structured_shape_completion.md)**

:   提出 UniCo，通过基元代理（primitive proxies）在共享形状特征上学习统一的基元表示，在单次前向传递中联合预测完整点云和装配就绪的二次曲面基元（含几何、语义和成员关系），在合成/真实点云 benchmark 上 Chamfer 距离降低最高 50%，法线一致性提升最高 7%。

**[UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking](unils_end-to-end_audio-driven_avatars_for_unified_listening_and_speaking.md)**

:   提出首个端到端统一说话-倾听面部表情生成框架UniLS，通过两阶段训练范式（先学内在运动先验、再用双轨音频微调），仅需双方音频输入即可同时生成自然的说话和倾听面部动作，倾听指标提升高达44.1%。

**[Unleashing Vision-Language Semantics for Deepfake Video Detection](unleashing_vision-language_semantics_for_deepfake_video_detection.md)**

:   提出VLAForge，通过ForgePerceiver独立学习多样的伪造线索和伪造定位图，并结合身份感知的视觉-语言对齐（VLA）评分机制，释放VLM跨模态语义的潜力来增强深度伪造视频检测的判别能力，在9个数据集上全面超越现有SOTA。

**[Unsafe2Safe: Controllable Image Anonymization for Downstream Utility](unsafe2safe_controllable_image_anonymization_for_downstream_utility.md)**

:   本文提出 Unsafe2Safe 全自动隐私保护流水线，通过 VLM 隐私检查→双字幕生成（私有/公开）→LLM 编辑指令→文本引导扩散编辑的四阶段方案，实现可控图像匿名化，在 VLMScore 隐私指标大幅提升的同时，在 Caltech-101 分类和 OK-VQA 上匿名后准确率甚至超过原始图像。

**[Vision-Language Attribute Disentanglement and Reinforcement for Lifelong Person Re-Identification](vision-language_attribute_disentanglement_and_reinforcement_for_lifelong_person_.md)**

:   VLADR 提出利用视觉-语言模型（VLM）中的细粒度属性知识来增强终身行人重识别，通过多粒度文本属性解耦（MTAD）和跨域跨模态属性强化（ICAR）两阶段训练，显式建模跨域共享的人体属性以实现高效知识转移和遗忘缓解，在抗遗忘和泛化能力上分别超越 SOTA 1.9%-2.2% 和 2.1%-2.5%。

**[VT-Intrinsic: Physics-Based Decomposition of Reflectance and Shading using a Single Visible-Thermal Image Pair](vt-intrinsic_physics-based_decomposition_of_reflectance_and_shading_using_a_sing.md)**

:   VT-Intrinsic 利用可见光和热红外图像之间的物理互补关系（未反射的光被吸收变为热量），推导出可见光-热成像强度的序数关系（ordinality）直接对应反射率和光照的序数关系，以此为自监督信号驱动神经网络优化，实现了无需预训练数据的高质量内在图像分解。

**[When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models](when_robots_obey_the_patch_universal_transferable_patch_attacks_on_vision-langua.md)**

:   提出 UPA-RFAS 框架，学习一个单一物理对抗补丁，通过特征空间偏移、注意力劫持和语义错位三管齐下，实现对 VLA 机器人策略的通用、可迁移黑盒攻击。
