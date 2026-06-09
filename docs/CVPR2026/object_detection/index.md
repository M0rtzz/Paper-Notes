---
title: >-
  CVPR2026 目标检测方向55篇论文解读
description: >-
  55篇CVPR2026的目标检测方向论文解读，涵盖目标检测、异常检测、少样本学习、多模态、3D 目标检测、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2026"
  - "目标检测"
  - "论文解读"
  - "论文笔记"
  - "异常检测"
  - "少样本学习"
  - "多模态"
  - "3D 目标检测"
  - "对齐/RLHF"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**📷 CVPR2026** · **55** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (6)](../../ICML2026/object_detection/index.md) · [🔬 ICLR2026 (14)](../../ICLR2026/object_detection/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/object_detection/index.md) · [🧠 NeurIPS2025 (26)](../../NeurIPS2025/object_detection/index.md) · [📹 ICCV2025 (27)](../../ICCV2025/object_detection/index.md) · [🧪 ICML2025 (12)](../../ICML2025/object_detection/index.md)

🔥 **高频主题：** 目标检测 ×16 · 异常检测 ×16 · 少样本学习 ×13 · 多模态 ×5 · 3D 目标检测 ×3

**[A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning Matters and Parallel Decoder Helps](a_closer_look_at_cross-domain_few-shot_object_detection_fine-tuning_matters_and_.md)**

:   提出混合集成解码器(HED)和渐进微调策略用于跨域少样本目标检测，通过并行化部分解码层并随机初始化去噪查询引入预测多样性，在CD-FSOD/ODinW-13/RF100-VL三个基准上达到SOTA，不引入额外参数。

**[A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection](a_semantically_disentangled_unified_model_for_multi-category_3d_anomaly_detectio.md)**

:   提出 SeDiR 框架，通过粗到细全局标记化（CFGT）、类别条件对比学习（C3L）和几何引导解码器（GGD）三个模块实现语义解纠缠的统一3D异常检测，解决跨类别特征纠缠（ICE）问题，在 Real3D-AD 和 Anomaly-ShapeNet 上分别超出SOTA 2.8% 和 9.1% AUROC。

**[ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](abra_teleporting_fine-tuned_knowledge_across_domains_for_open-vocabulary_object_.md)**

:   提出 ABRA 方法，将域知识与类别知识解耦，通过 Objectification 构建类无关域专家、SVFT 提取轻量类别残差、Orthogonal Procrustes 旋转对齐实现权重空间"传送"，在目标域完全无某些类别数据时仍可迁移这些类别的检测能力。

**[AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)**

:   AnomalyVFM 提出了一个通用框架，通过三阶段合成数据生成方案和参数高效的 LoRA 适配机制，将任意视觉基础模型（VFM）转化为强零样本异常检测器，以 RADIO 为骨干在 9 个工业数据集上达到 94.1% 图像级 AUROC，超越 SOTA 3.3 个百分点。

**[AR²-4FV: Anchored Referring and Re-identification for Long-Term Grounding in Fixed-View Videos](ar2-4fv_anchored_referring_and_re-identification_for_long-term_grounding_in_fixe.md)**

:   利用固定视角视频中背景结构的时不变性，构建离线 Anchor Bank + 在线 Anchor Map 作为语言-场景持久记忆，配合锚点引导的重入先验和 ReID-Gating 身份验证机制，实现目标遮挡/离场后的鲁棒重捕获，RCR 提升 10.3%、RCL 降低 24.2%。

**[Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection](back_to_point_exploring_point-language_models_for_zero-shot_3d_anomaly_detection.md)**

:   BTP 首次将预训练的点-语言模型（PLM，如 ULIP）应用于零样本 3D 异常检测，提出多粒度特征嵌入模块（MGFEM）融合 patch 级语义、几何描述子和全局 CLS token，配合联合表示学习策略，在 Real3D-AD 点级 AUROC 达到 84.5%，大幅超越观 VLM 渲染方案的 PointAD（73.5%）。

**[Beyond Caption-Based Queries for Video Moment Retrieval](beyond_caption-based_queries_for_video_moment_retrieval.md)**

:   揭示了VMR中caption-based查询与真实用户搜索查询之间的巨大鸿沟，提出了三个搜索查询基准，并通过移除自注意力+查询Dropout两项架构修改来缓解DETR中的解码器查询坍塌问题，在多时刻搜索查询上提升高达21.83% mAPm。

**[Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection](beyond_prompt_degradation_prototype-guided_dual-pool_prompting_for_incremental_o.md)**

:   提出 PDP 框架，通过双池提示解耦（共享池 + 私有池）和原型引导伪标签生成（PPG），解决增量目标检测中提示耦合与提示漂移导致的提示退化问题，在 COCO 和 VOC 上取得 SOTA。

**[Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval](beyond_semantic_search_towards_referential_anchoring_in_composed_image_retrieval.md)**

:   提出Object-Anchored Composed Image Retrieval（OACIR）新任务和OACIRR大规模基准（160K+四元组），以及AdaFocal框架通过上下文感知注意力调制器自适应地增强对锚定实例区域的关注，在实例级检索保真度上大幅超越现有方法。

**[Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)**

:   提出AnoPLe——一个轻量级多模态双向提示学习框架，无需手工异常描述或外部辅助模块，通过文本-视觉提示双向交互和尺度感知前缀实现少样本多类别异常检测，在MVTec-AD/VisA/Real-IAD上取得强竞争力的同时保持高效推理（~28 FPS）。

**[Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting](boosting_quantitive_and_spatial_awareness_for_zero-shot_object_counting.md)**

:   提出QICA框架解决零样本目标计数中的数量感知缺失和空间不敏感问题，通过数量条件化的协同提示策略（SPS）联合适配视觉-语言编码器，结合在相似度图上直接操作的代价聚合解码器（CAD）保持零样本迁移能力，在FSC-147上达到零样本SOTA（MAE 12.41）并展现强跨域泛化。

**[BUSSARD: Normalizing Flows for Bijective Universal Scene-Specific Anomalous Relationship Detection](bussard_normalizing_flows_for_bijective_universal_scene-specific_anomalous_relat.md)**

:   提出 BUSSARD，首个基于学习的场景特定异常关系检测方法，利用预训练语言模型嵌入场景图三元组 + 自编码器降维 + 标准化流进行似然估计，在 SARD 数据集上 AUROC 提升约 10%，且对同义词变化鲁棒。

**[CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection](cd-buffer_complementary_dual-buffer_framework_for_test-time_adaptation_in_advers.md)**

:   提出 CD-Buffer 框架，通过统一的域差异度量驱动减性缓冲（通道抑制）和加性缓冲（轻量适配器补偿）的互补协作，实现跨不同严重程度恶劣天气条件下的鲁棒测试时目标检测适应。

**[CompAgent: An Agentic Framework for Visual Compliance Verification](compagent_an_agentic_framework_for_visual_compliance_verification.md)**

:   提出 CompAgent，首个用于视觉合规验证的智能体框架——Planning Agent 根据合规策略动态选择视觉工具（目标检测、人脸分析、NSFW 检测等），Compliance Verification Agent 整合图像、工具输出和策略上下文进行多模态推理，无需训练即在 UnsafeBench 上超越 SOTA 10% 达 76% F1。

**[CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection](cops_conditional_prompt_synthesis_for_zero-shot_anomaly_detection.md)**

:   本文提出 CoPS 框架，通过显式状态token合成（ESTS）和隐式类别token采样（ICTS）两种视觉条件化机制动态生成提示，配合空间感知对齐（SAGA），在13个工业和医学数据集上实现零样本异常检测SOTA。

**[DA-Mamba: Learning Domain-Aware State Space Model for Global-Local Alignment in Domain Adaptive Object Detection](da-mamba_learning_domain-aware_state_space_model_for_global-local_alignment_in_d.md)**

:   提出 DA-Mamba，一种 CNN-SSM 混合架构，通过 Image-Aware SSM（IA-SSM）和 Object-Aware SSM（OA-SSM）两个模块，以线性复杂度实现图像级和实例级的全局-局部域不变特征对齐，在四个域自适应检测基准上达到 SOTA。

**[Detecting Unknown Objects via Energy-Based Separation for Open World Object Detection](detecting_unknown_objects_via_energy-based_separation.md)**

:   提出 DEUS 框架，通过 ETF 子空间未知目标分离（EUS）在几何正交的已知/未知子空间中利用能量分数有效分离已知、未知和背景提案，并设计能量基已知区分损失（EKD）减少增量学习中新旧类的交叉干扰，在 OWOD 基准上大幅提升未知目标召回率。

**[Does YOLO Really Need to See Every Training Image in Every Epoch?](does_yolo_really_need_to_see_every_training_image_in_every_epoch.md)**

:   提出 Anti-Forgetting Sampling Strategy (AFSS)，根据每张训练图像的学习充分度（min(Precision, Recall)）动态决定哪些图像参与训练、哪些可以跳过，实现 YOLO 系列检测器 1.43× 以上的训练加速同时保持甚至提升检测精度。

**[Evaluating Few-Shot Pill Recognition Under Visual Domain Shift](evaluating_few-shot_pill_recognition_under_visual_domain_shift.md)**

:   本文从部署视角系统评估药丸识别在跨域few-shot条件下的泛化能力，揭示语义分类1-shot即饱和但定位/recall在重叠遮挡下急剧下降的解耦现象，并证明训练数据的视觉真实性远比数据量或shot数更关键。

**[EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer](ew-detr_evolving_world_object_detection_via_incremental_low-rank_detection_trans.md)**

:   提出 Evolving World Object Detection (EWOD) 范式及 EW-DETR 框架，通过增量 LoRA 适配器、查询范数物体性适配器和熵感知未知混合三个协同模块，在无样本回放条件下同时解决类别增量学习、域迁移适应和未知目标检测问题，FOGS 指标提升 57.24%。

**[Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection](foundation_model_priors_enhance_object_focus_in_feature_space_for_source-free_ob.md)**

:   提出 FALCON-SFOD 框架，通过基础模型（OV-SAM）生成的类别无关二值掩码正则化检测器特征空间（SPAR），结合不平衡感知的噪声鲁棒伪标签损失（IRPL），在无源域目标检测中增强目标聚焦表征，多个基准上达到 SOTA。

**[Fourier Angle Alignment for Oriented Object Detection in Remote Sensing](fourier_angle_alignment_for_oriented_object_detection_in_remote_sensing.md)**

:   利用傅里叶旋转等变性在频域估计目标主方向并对齐特征，提出 FAAFusion 和 FAA Head 两个即插即用模块分别解决 FPN 跨尺度方向不一致和检测头分类-回归任务冲突，在 DOTA-v1.0/v1.5 和 HRSC2016 上取得新 SOTA。

**[GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)**

:   提出GS-CLIP两阶段框架，通过几何缺陷蒸馏模块将3D点云的全局形状和局部缺陷信息注入文本提示，并用LoRA双流架构协同融合渲染图和深度图，在四个大规模数据集上实现零样本3D异常检测SOTA。

**[HeROD: Heuristic-inspired Reasoning Priors Facilitate Data-Efficient Referring Object Detection](herod_heuristic_inspired_reasoning_data_efficient_rod.md)**

:   HeROD 提出了一种轻量级、模型无关的框架，通过将启发式空间和语义推理先验注入 DETR 风格检测管道的三个阶段（候选排序、预测融合、匈牙利匹配），在标注稀缺条件下显著提升指代目标检测(ROD)的数据效率和收敛性能。

**[Integration of Deep Generative Anomaly Detection Algorithm in High-Speed Industrial Line](integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)**

:   本文提出一个基于 GAN + 残差自编码器（DRAE）的半监督异常检测框架，专门设计用于制药行业 Blow-Fill-Seal（BFS）产线的高速在线质量检测，仅用合格品训练即可实现 96.4% 的准确率，单 patch 推理仅 0.17ms，满足 500ms 检测周期的严格工业约束。

**[InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)**

:   提出 InvAD，将扩散模型异常检测从"RGB 空间去噪重建"范式转变为"潜空间加噪反演"范式，通过 DDIM 反演直接推断最终潜变量并在先验分布下度量偏差来检测异常，仅需 3 步反演即达 SOTA 性能且推理速度提升约 2 倍。

**[Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](learning_multi-modal_prototypes_for_cross-domain_few-shot_object_detection.md)**

:   提出双分支框架 LMP，在 GroundingDINO 基础上引入视觉原型分支（正类原型+硬负原型），与文本分支联合训练并集成推理，在跨域少样本目标检测中取得 SOTA。

**[Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)**

:   提出 InCoM-Net，通过从 VLM 特征中为每个实例分别提取实例内、实例间和全局三层上下文特征，并通过渐进式上下文聚合与检测器特征融合，在 HICO-DET 和 V-COCO 上取得 HOI 检测 SOTA（HICO-DET Full mAP 43.96，V-COCO AP_role^S1 73.6）。

**[MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](mmrad_multimodal_anomaly_detection.md)**

:   MMR-AD 构建了当前最大规模的多模态推理型工业异常检测数据集（127K 图像、188 类产品、395 种异常），并提出基于 GRPO 强化学习的 Anomaly-R1 基线模型，显著优于通用 MLLM。

**[MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)**

:   提出 MoECLIP，将 Mixture-of-Experts 引入零样本异常检测（ZSAD），通过冻结正交特征分离（FOFS）和等角紧框架（ETF）损失实现 patch 级别的动态专家路由与特化，在14个工业/医学基准上达到 SOTA。

**[MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image Understanding](mrd_multi-resolution_retrieval-detection_fusion_for_high-resolution_image_unders.md)**

:   提出 MRD，一个 training-free 的多分辨率检索-检测融合框架，通过多分辨率语义融合缓解目标碎片化，结合开放词汇检测器抑制背景干扰，显著提升 MLLM 对高分辨率图像的理解能力。

**[NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection](noovd_novel_category_discovery_and_embedding_for_open-vocabulary_object_detectio.md)**

:   提出NoOVD框架，在基于冻结VLM的OVD训练中通过无参数K-FPN保留CLIP知识来发现潜在新类别目标、通过自蒸馏将新类别知识嵌入检测器、通过R-RPN在推理时提升新类别召回率，在OV-LVIS/OV-COCO/Objects365上取得SOTA。

**[Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples](novel_anomaly_detection_scenarios_and_evaluation_metrics_to_address_the_ambiguit.md)**

:   针对工业异常检测中"正常"定义随规格变更而变化的实际问题，提出了两种新场景（A2N/N2A）、一个新评价指标（S-AUROC）和一种训练增强方法 RePaste，通过将高异常分数区域重新粘贴到训练图片中来增加其出现频率，使模型灵活适应正常样本定义的变化。

**[PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection](paq-detr_learning_pattern_and_quality-aware_dynamic_queries_for_object_detection.md)**

:   PaQ-DETR 提出基于共享模式的动态查询生成（内容感知权重组合共享基模式）+ 质量感知一对多分配（基于定位-分类一致性自适应选择正样本），统一解决DETR中的查询表示和监督不均衡问题，在多个backbone上稳定提升1.5%-4.2% mAP。

**[Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection](parameter-efficient_semantic_augmentation_for_enhancing_open-vocabulary_object_d.md)**

:   HSA-DINO 提出多尺度 prompt bank 从图像特征金字塔中学习层次化语义 prompt 增强文本表示，并通过语义感知路由器在推理时动态决定是否使用领域特定增强，实现了领域适配与开放词汇泛化的优越平衡（H 值在三个垂直领域数据集上均为最优）。

**[PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training](pet-dino_unifying_visual_cues_into_grounding_dino_with_prompt-enriched_training.md)**

:   PET-DINO 在 Grounding DINO 基础上构建了一个同时支持文本和视觉提示的通用目标检测器，设计了对齐友好的视觉提示生成模块（AFVPG）以及两种提示丰富化训练策略（IBP 和 DMD），在零样本检测任务上以更少的训练数据取得了有竞争力的性能。

**[PHAC: Promptable Human Amodal Completion](phac_promptable_human_amodal_completion.md)**

:   提出可提示人体非模态补全（PHAC）新任务，通过基于点的用户提示（姿态/边界框）配合 ControlNet 注入条件信号，并设计基于修复的精炼模块保留可见区域外观，实现高质量、可控的遮挡人体图像补全。

**[Prompt-Free Universal Region Proposal Network](prompt-free_universal_region_proposal_network.md)**

:   PF-RPN 用可学习视觉嵌入替代文本/图像提示，通过稀疏图像感知适配器、级联自提示和中心性引导查询选择三个模块，仅用 5% COCO 数据训练即可在 19 个跨域数据集上实现 SOTA 零样本区域提案。

**[Random Wins All: Rethinking Grouping Strategies for Vision Tokens](random_wins_all_rethinking_grouping_strategies_for_vision_tokens.md)**

:   提出极简的随机分组策略替代 Vision Transformer 中各种精心设计的 token 分组方法，在图像分类、目标检测、语义分割、点云分割和 VLM 上几乎全面超越所有 baseline，并从位置信息、头特征多样性、全局感受野和固定分组模式四个维度解释了随机分组成功的原因。

**[RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation](rc-nf_robot-conditioned_normalizing_flow_for_real-time_anomaly_detection_in_robo.md)**

:   提出 Robot-Conditioned Normalizing Flow (RC-NF)，通过条件归一化流对机器人状态与物体运动轨迹的联合分布建模，实现 <100ms 实时异常检测，可作为 VLA 模型（如 π₀）的即插即用监控模块，支持任务级重规划和状态级轨迹回滚。

**[Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision](reasoning-driven_anomaly_detection_and_localization_with_image-level_supervision.md)**

:   提出 ReAL 和 CGRO 两个模块，通过提取 MLLM 自回归推理过程中的异常相关 token 并聚合其视觉注意力来生成像素级异常图，再通过一致性引导的强化学习对齐推理与视觉证据，实现仅凭图像级监督的端到端异常检测、定位与可解释推理。

**[Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection](remedying_target-domain_astigmatism_for_cross-domain_few-shot_object_detection.md)**

:   首次发现跨域少样本目标检测（CD-FSOD）中模型注意力在目标域持续分散的"散光"现象，受人类中央凹视觉系统启发，设计正向模式精化（PPR）、负向上下文调制（NCM）和文本语义对齐（TSA）三个互补模块来重塑注意力，在6个跨域基准上以显著优势达到SOTA。

**[Saliency-R1: Enforcing Interpretable and Faithful Vision-language Reasoning via Saliency-map Alignment Reward](saliency-r1_enforcing_interpretable_and_faithful_vision-language_reasoning_via_s.md)**

:   提出 Saliency-R1，通过基于 logit 分解的高效显著性图技术和思维链瓶颈注意力回溯，将显著性图与人工标注 bounding box 的对齐度作为 GRPO 奖励，训练 VLM 在推理时聚焦任务相关的图像区域，提升推理的可解释性和忠实性。

**[SDF-Net: Structure-Aware Disentangled Feature Learning for Optical–SAR Ship Re-Identification](sdf-net_structure-aware_disentangled_feature_learning_for_opticall-sar_ship_re-i.md)**

:   提出 SDF-Net，利用船舶刚体几何结构作为跨模态不变锚点，在中间层提取梯度能量强制结构一致性，在终端层解耦模态共享/特定特征并通过加法残差融合，在 HOSS-ReID 上取得 SOTA（All mAP 60.9%，超 TransOSS 3.5%）。

**[Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](show_dont_tell_detecting_novel_objects_by_watching.md)**

:   提出 "Show, Don't Tell" 范式——通过观看人类演示视频自动创建训练数据集并训练定制化物体检测器，完全绕过语言描述和提示工程，在真实机器人场景中显著超越 SOTA 开集/闭集检测器的新物体识别能力。

**[Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)**

:   提出 ESM-YOLO+，一个轻量级可见光-红外融合小目标检测网络，通过 Mask-Enhanced Attention Fusion (MEAF) 模块实现像素级跨模态自适应融合，并引入训练时结构表示增强提升空间判别力，在 VEDAI 上达 84.71% mAP 同时参数量减少 93.6%。

**[SpiralDiff: Spiral Diffusion with LoRA for RGB-to-RAW Conversion Across Cameras](spiraldiff_spiral_diffusion_with_lora_for_rgb-to-raw_conversion_across_cameras.md)**

:   提出 SpiralDiff，一种面向 RGB-to-RAW 转换的扩散框架，通过信号依赖的噪声加权策略适应不同像素强度区域的重建难度，并引入 CamLoRA 模块实现单一模型跨多相机的轻量适配。

**[SteelDefectX: A Coarse-to-Fine Vision-Language Dataset and Benchmark for Generalizable Steel Surface Defect Detection](steeldefectx_a_coarse-to-fine_vision-language_dataset_and_benchmark_for_generali.md)**

:   提出 SteelDefectX，首个面向钢材表面缺陷检测的视觉-语言数据集（7778 张图像、25 类缺陷），包含从类级到样本级的粗到细文本标注，并建立了涵盖纯视觉分类、视觉-语言分类、零/少样本识别和零样本迁移的四任务基准，实验证明高质量文本标注显著提升模型的可解释性、泛化性和跨域迁移能力。

**[SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling](subspacead_training-free_few-shot_anomaly_detection_via_subspace_modeling.md)**

:   SubspaceAD 证明了在强视觉基础模型（DINOv2-G）特征上做一次 PCA 拟合就足以超越所有需要训练/记忆库/提示调优的少样本异常检测方法，1-shot 下在 MVTec-AD 上达 98.0% 图像级 AUROC 和 97.6% 像素级 AUROC。

**[The COTe Score: A Decomposable Framework for Evaluating Document Layout Analysis Models](the_cote_score_a_decomposable_framework_for_evaluating_document_layout_analysis_.md)**

:   提出面向文档布局分析（DLA）的可分解评估框架 COTe（Coverage, Overlap, Trespass, Excess），以及结构语义单元 SSU，相比传统 IoU/mAP/F1 能更准确地反映页面解析质量，并揭示不同模型的特异性失败模式。

**[Toward Generalizable Whole Brain Representations with High-Resolution Light-Sheet Data](toward_generalizable_whole_brain_representations_with_high-resolution_light-shee.md)**

:   提出 CANVAS——首个大规模亚细胞分辨率光片荧光显微镜（LSFM）全脑基准数据集，涵盖 6 种细胞标记物、约 93,000 个细胞标注和公开排行榜，揭示了现有检测模型在跨标记物和跨脑区泛化上的严重不足，并探索了 3D 掩码自编码器（MAE）的自监督表示学习潜力。

**[UAVGen: Visual Prototype Conditioned Focal Region Generation for UAV-Based Object Detection](uavgen_visual_prototype_conditioned_focal_region_generation_for_uav_based_object_detection.md)**

:   提出 UAVGen，一个面向无人机目标检测的 layout-to-image 数据增强框架，通过视觉原型条件扩散模型和焦点区域增强管线解决小目标生成质量低、模型容量浪费和标签不一致问题。

**[UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)**

:   提出 UniMMAD，首个用单一参数集同时处理多模态、多类别异常检测的统一框架，核心是基于 MoE 的特征解压缩机制，将通用多模态编码特征自适应分解为领域特定的单模态重建，在 9 个数据集（3 个领域、12 种模态、66 个类别）上达到 SOTA。

**[UniSpector: Towards Universal Open-set Defect Recognition via Spectral-Contrastive Visual Prompting](unispector_towards_universal_open-set_defect_recognition_via_spectral-contrastiv.md)**

:   本文提出 UniSpector 开放集工业缺陷检测框架，通过频域-空域双域特征融合（SSPE）和角度间隔对比提示编码（CPE）解决了视觉提示嵌入崩塌问题，在新构建的包含 360 种缺陷类别的 Inspect Anything benchmark 上，AP50 检测和分割分别比最佳基线高 19.7% 和 15.8%。

**[VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)**

:   重新审视零样本异常检测（ZSAD）中文本分支的必要性，提出 VisualAD——一个纯视觉框架：在冻结 ViT 中插入两个可学习 token（anomaly/normal），配合 Spatial-Aware Cross-Attention 和 Self-Alignment Function，去掉文本编码器仍在 13 个工业+医学基准上取得 SOTA。
