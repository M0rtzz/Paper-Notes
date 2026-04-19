---
title: >-
  ICCV2025 目标检测方向 41篇论文解读
description: >-
  41篇ICCV2025 目标检测方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**📹 ICCV2025** · **41** 篇论文解读

**[3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection](3dmood_lifting_2d_to_3d_for_monocular_openset_object_detecti.md)**

:   提出首个端到端的单目开放集3D目标检测器3D-MOOD，通过将开放集2D检测"提升"到3D空间，结合几何感知3D query生成与canonical image space设计，在Omni3D闭集和Argoverse 2/ScanNet开集基准上均达到SOTA。

**[Advancing Textual Prompt Learning with Anchored Attributes](advancing_textual_prompt_learning_with_anchored_attributes.md)**

:   本文提出 ATPrompt，通过在文本 prompt 中嵌入通用属性 token（如颜色、形状），将软 prompt 的学习空间从一维类别级别拓展到多维属性级别，作为即插即用的模块可无缝集成到现有文本 prompt 学习方法中，在 11 个数据集上一致性提升基线性能。

**[Adversarial Attention Perturbations for Large Object Detection Transformers](adversarial_attention_perturbations_for_large_object_detection_transformers.md)**

:   本文提出 AFOG（Attention-Focused Offensive Gradient），一种架构无关的对抗攻击方法，通过可学习注意力机制聚焦扰动到图像脆弱区域，仅需 10 次迭代即可在视觉不可察觉的扰动下将 12 种检测 Transformer 的 mAP 最高降低 37.8 倍，同时在 CNN 检测器上也优于现有方法。

**[Anchor Token Matching: Implicit Structure Locking for Training-free AR Image Editing](anchor_token_matching_implicit_structure_locking_for_training-free_ar_image_edit.md)**

:   提出 ISLock，首个面向自回归(AR)视觉生成模型的无训练图像编辑方法，通过锚点 Token 匹配(ATM)在隐空间中隐式对齐自注意力模式，实现结构一致的文本引导图像编辑。

**[Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!](attention_to_neural_plagiarism_diffusion_models_can_plagiarize_your_copyrighted_.md)**

:   揭示"神经抄袭"威胁——扩散模型可轻松复制受版权保护的图像（包括受水印保护的图像），提出基于"锚点与垫片"的通用攻击框架，通过在交叉注意力机制中搜索扰动实现从粗到细的语义修改，绕过从可见商标到隐形水印的各类版权保护。

**[Augmenting Moment Retrieval: Zero-Dependency Two-Stage Learning](augmenting_moment_retrieval_zero-dependency_two-stage_learning.md)**

:   提出 AMR 框架，通过 Splice-and-Boost 数据增强策略和冷启动-蒸馏两阶段训练，在不依赖任何外部数据/预训练模型的前提下，大幅提升视频时刻检索的边界感知能力和语义辨别力，在 QVHighlights 上超越 SOTA +5%。

**[Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability](automated_model_evaluation_for_object_detection_via_prediction_consistency_and_r.md)**

:   本文提出PCR（Prediction Consistency and Reliability），一种无需人工标注即可估计目标检测模型性能的自动化评估方法，通过分析NMS前后边界框的空间一致性和置信度可靠性来估计mAP，并构建了基于图像腐蚀的元数据集以实现更现实和可扩展的评估。

**[ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning](chartpoint_guiding_mllms_with_grounding_reflection_for_chart_reasoning.md)**

:   提出PointCoT方法，将反思性视觉定位（bounding box）集成到图表推理的思维链中，使MLLM在每个推理步骤都能与图表视觉内容交互验证，并构建了包含19.2K高质量样本的ChartPoint-SFT-62k数据集，在ChartBench上实现+5.04%的提升。

**[DiffDoctor: Diagnosing Image Diffusion Models Before Treating](diffdoctor_diagnosing_image_diffusion_models_before_treating.md)**

:   提出 DiffDoctor，首个利用像素级反馈微调扩散模型的方法：先训练鲁棒的 artifact 检测器（1M+ 样本，类别平衡策略），再通过最小化合成图中每个像素的 artifact 置信度反向传播梯度到扩散模型，使其在未见 prompt 上也能显著减少 artifact 生成。

**[Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion](diffusion_curriculum_synthetic-to-real_data_curriculum_via_image-guided_diffusio.md)**

:   利用扩散模型的图像引导强度控制生成从合成到真实的连续谱系数据，设计"扩散课程学习（DisCL）"策略在训练不同阶段自适应选择最优引导级别的合成数据，有效解决长尾分类和低质量数据学习问题。

**[DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion](distil_data-free_inversion_of_suspicious_trojan_inputs_via_latent_diffusion.md)**

:   DISTIL 提出一种无需干净数据的木马触发器反演方法，通过在预训练引导扩散模型的潜空间中搜索触发器模式（而非像素空间），并注入均匀噪声正则化，有效区分真实后门触发器和对抗扰动，在 BackdoorBench 上精度最高提升 7.1%。

**[Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)**

:   首次将Mixture of Experts引入实时开放词汇目标检测器，通过MoE-Tuning将Grounding DINO 1.5 Edge从dense模型扩展为动态推理框架，提出细粒度专家分解和预训练权重分配策略，仅用1.56M开源数据超越使用20M私有数据训练的原版模型。

**[EA-KD: Entropy-based Adaptive Knowledge Distillation](ea-kd_entropy-based_adaptive_knowledge_distillation.md)**

:   提出 EA-KD，一种基于信息熵的即插即用知识蒸馏方法：通过结合 teacher 和 student 输出的熵值动态重加权蒸馏损失，优先学习高熵（高信息量）样本，在图像分类、目标检测和 LLM 蒸馏任务上均一致提升多种 KD 框架的性能，且计算开销可忽略。

**[EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision](evrt-detr_latent_space_adaptation_of_image_detectors_for_event-based_vision.md)**

:   提出I2EvDet框架，通过在冻结的RT-DETR检测器的潜空间中插入轻量级RNN时序模块，以最小的架构修改将主流图像检测器适配为事件相机视频检测模型，在Gen1和1Mpx基准上分别取得+2.3和+1.4 mAP的SOTA。

**[FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos](fakeradar_probing_forgery_outliers_to_detect_unknown_deepfake_videos.md)**

:   提出FakeRadar深度伪造视频检测框架，通过Forgery Outlier Probing在特征空间中主动生成模拟未知伪造的异常值样本，并设计Outlier-Guided Tri-Training三分类优化策略，在跨数据集/跨操纵类型评估中显著超越现有方法。

**[Few-Shot Pattern Detection via Template Matching and Regression](few-shot_pattern_detection_via_template_matching_and_regression.md)**

:   本文提出TMR方法，通过经典模板匹配结合支持条件化边界框回归，实现了对任意模式（包括非物体级模式）的小样本检测，同时引入RPINE数据集覆盖更广泛的重复模式，在多个基准上超越现有FSCD方法并展现出强大的跨数据集泛化能力。

**[ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection](forgelens_data-efficient_forgery_focus_for_generalizable_forgery_image_detection.md)**

:   提出 ForgeLens，一个基于冻结 CLIP-ViT 的特征引导框架，通过轻量级的权重共享引导模块（WSGM）和伪造感知特征集成器（FAFormer），引导冻结预训练网络聚焦伪造特征，仅用 1% 训练数据即达到 SOTA 泛化性能。

**[Intervening in Black Box: Concept Bottleneck Model for Enhancing Human-Neural Network Mutual Understanding](intervening_in_black_box_concept_bottleneck_model_for_enhancing_human_neural_net.md)**

:   提出 CBM-HNMU 框架，通过概念瓶颈模型（CBM）逼近黑盒模型的推理过程，自动识别并修正有害概念，再将修正后的知识蒸馏回黑盒模型，实现超越样本级别的系统性模型干预与准确率提升。

**[Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation](is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)**

:   提出 Token Condensation as Adaptation（TCA），一种免训练的测试时自适应方法，通过领域感知的 token 库（DTR）引导跨头 token 裁剪/合并和 logits 自校正，在不修改模型参数的情况下，将 CLIP/SigLIP 系列的跨数据集性能提升最高 21.4%，同时减少 12.2%-48.9% 的 GFLOPs。

**[Large-scale Pre-training for Grounded Video Caption Generation](large-scale_pre-training_for_grounded_video_caption_generation.md)**

:   提出 GROVE 模型和大规模自动标注方法，构建包含 1M 视频的 HowToGround1M 预训练数据集和 3513 个视频的手动标注 iGround 数据集，实现联合视频字幕生成与多目标时空边界框定位，在 iGround、VidSTG、ActivityNet-Entities 等数据集上取得 SOTA。

**[LMM-Det: Make Large Multimodal Models Excel in Object Detection](lmm-det_make_large_multimodal_models_excel_in_object_detection.md)**

:   提出 LMM-Det，通过系统分析发现大型多模态模型在目标检测中核心瓶颈是低召回率，并通过数据分布调整（伪标签增强）和推理优化（按类别逐一检测）将 LMM 的 COCO AP 从 0.2 提升至 47.5，无需任何额外专用检测模块。

**[Measuring the Impact of Rotation Equivariance on Aerial Object Detection](measuring_the_impact_of_rotation_equivariance_on_aerial_object_detection.md)**

:   提出 MessDet，一个基于旋转等变网络的航空目标检测器，通过新型下采样过程实现严格旋转等变性，并引入旋转等变通道注意力和多分支检测头，在 DOTA 等数据集上以极低参数量达到 SOTA 性能。

**[OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images](openrsd_towards_open-prompts_for_object_detection_in_remote_sensing_images.md)**

:   提出OpenRSD通用遥感开放提示目标检测框架，支持文本和图像多模态提示，集成对齐头和融合头平衡速度与精度，配合三阶段训练流水线和47万张图像的ORSD+数据集，在7个公开数据集上取得最优平均性能，同时保持20.8 FPS实时推理。

**[PASG: A Closed-Loop Framework for Automated Geometric Primitive Extraction and Semantic Anchoring in Robotic Manipulation](pasg_a_closed-loop_framework_for_automated_geometric_primitive_extraction_and_se.md)**

:   提出 PASG（Primitive-Aware Semantic Grounding），一个闭环框架，通过自动化几何基元提取（关键点、功能轴、主轴）和 VLM 驱动的语义锚定，将低层几何特征与高层任务语义动态耦合，在机器人操作任务中实现了接近人工标注的性能，并构建了 Robocasa-PA 基准和微调模型 Qwen2.5VL-PA。

**[Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)**

:   系统性重新审视 11 种对抗补丁防御方法，建立首个补丁防御基准（含 13 种攻击、11 个检测器、4 种度量），构建 94,000 张图像的大规模 APDE 数据集，并揭示三个关键新发现：自然补丁防御难点在于数据分布而非高频、补丁检测精度与防御性能不一致、自适应攻击可绕过大多数现有防御。

**[Semantic Discrepancy-aware Detector for Image Forgery Identification](semantic_discrepancy-aware_detector_for_image_forgery_identification.md)**

:   提出语义差异感知检测器（SDD），通过语义 token 采样、概念级伪造差异学习和低层伪造特征增强三个模块，利用重建学习将 CLIP 的视觉语义概念空间与伪造空间进行细粒度对齐，在 UnivFD 和 SynRIS 基准上取得 SOTA 性能（$ap_m$ 98.51%，AUROC 95.1%）。

**[SFUOD: Source-Free Unknown Object Detection](sfuod_source-free_unknown_object_detection.md)**

:   提出 Source-Free Unknown Object Detection (SFUOD) 新场景，并设计 CollaPAUL 框架，通过协作调优融合源域和目标域知识 + 基于主轴的未知物体伪标签分配，在无源数据条件下同时检测已知和未知物体。

**[Sim-DETR: Unlock DETR for Temporal Sentence Grounding](sim-detr_unlock_detr_for_temporal_sentence_grounding.md)**

:   系统分析了 DETR 在时序语句定位 (TSG) 任务中的异常行为根因——查询间冲突和查询内全局-局部矛盾，并提出两个简单修改（Query Grouping & Ranking + Global-Local Bridging）构成 Sim-DETR，解锁 DETR 在 TSG 任务的全部潜力。

**[SketchSplat: 3D Edge Reconstruction via Differentiable Multi-view Sketch Splatting](sketchsplat_3d_edge_reconstruction_via_differentiable_multi-view_sketch_splattin.md)**

:   提出 SketchSplat，将 3D 边缘表示为参数化 sketch（直线+Bézier曲线），通过从 sketch 采样高斯点进行可微渲染来直接优化边缘参数，同时提出自适应拓扑控制和改进的 2D 边缘检测器，在 CAD 数据集上实现 SOTA 的准确性、完整性和紧凑性。

**[Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding](sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)**

:   提出 SDST（Sparse-Dense Side-Tuner），首个无锚框（anchor-free）的 Side-Tuning 架构，通过稀疏-稠密双流设计同时处理时刻检索（MR）和高光检测（HD），并提出 Reference-based Deformable Self-Attention（RDSA）解决可变形注意力的上下文缺失问题，在 QVHighlights、TACoS、Charades-STA 上取得 SOTA 或高度竞争性结果，同时将可训练参数量减少至现有 SOTA 的 27%。

**[Synchronization of Multiple Videos](synchronization_of_multiple_videos.md)**

:   提出 Temporal Prototype Learning (TPL)，一个基于原型的视频同步框架，从预训练模型提取的高维嵌入中构建共享的紧凑1D表征，通过学习统一的原型序列锚定关键动作阶段来对齐多个视频，首次解决了生成式AI视频的同步问题。

**[The Devil is in the Spurious Correlations: Boosting Moment Retrieval with Dynamic Learning](the_devil_is_in_the_spurious_correlations_boosting_moment_retrieval_with_dynamic.md)**

:   首次揭示文本查询与视频背景帧之间的虚假相关性是时刻检索性能瓶颈的根本原因，提出 TD-DETR 框架通过动态上下文视频合成和文本-动态交互增强两个策略来缓解该问题，在 QVHighlights 和 Charades-STA 上达到 SOTA。

**[Uncertainty-Aware Gradient Stabilization for Small Object Detection](uncertainty-aware_gradient_stabilization_for_small_object_detection.md)**

:   揭示了传统目标定位方法在小目标上存在因损失曲率陡峭导致的梯度不稳定问题，提出 UGS（不确定性感知梯度稳定化）框架，通过分类式定位 + 不确定性最小化 + 不确定性引导精炼三个组件来稳定梯度，显著提升小目标检测性能。

**[UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement](upre_zero-shot_domain_adaptation_for_object_detection_via_unified_prompt_and_rep.md)**

:   提出 UPRE 框架，通过联合优化多视角域提示（MDP）和统一表示增强（URE）来同时缓解零样本域自适应目标检测中的检测偏差和域偏差，在恶劣天气、跨城市、虚拟到现实三类场景的九个数据集上取得 SOTA 性能。

**[ViewSRD: 3D Visual Grounding via Structured Multi-View Decomposition](viewsrd_3d_visual_grounding_via_structured_multi-view_decomposition.md)**

:   提出 ViewSRD 框架，将 3D 视觉定位建模为结构化多视角分解过程：通过 SRD 模块将复杂多锚点查询解耦为简单单锚点查询，并引入跨模态一致视角 token (CCVT) 解决视角变化导致的空间描述不一致问题。

**[VisRL: Intention-Driven Visual Perception via Reinforced Reasoning](visrl_intention-driven_visual_perception_via_reinforced_reasoning.md)**

:   VisRL是首个将强化学习应用于意图驱动视觉感知的框架，通过迭代DPO训练让大多模态模型学会根据查询意图自主选择关注区域（预测bounding box），无需昂贵的中间bounding box标注即可实现比SFT更强的视觉推理能力。

**[Visual-RFT: Visual Reinforcement Fine-Tuning](visual-rft_visual_reinforcement_fine-tuning.md)**

:   Visual-RFT将DeepSeek R1的强化学习+可验证奖励(RLVR)范式从数学/代码领域扩展到视觉感知任务，设计了IoU奖励（目标检测）和CLS奖励（分类）等任务特异的可验证奖励函数，在细粒度分类、少样本检测、推理定位等任务上以极少数据大幅超越SFT。

**[Visual Modality Prompt for Adapting Vision-Language Object Detectors](visual_modality_prompt_for_adapting_vision-language_object_detectors.md)**

:   提出 ModPrompt，一种基于编码器-解码器的视觉提示策略，将视觉-语言目标检测器（如 YOLO-World、Grounding DINO）适应到红外和深度等新模态，同时保留零样本检测能力。

**[VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions](voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)**

:   提出 VOccl3D，一个基于3DGS渲染的大规模合成视频数据集（25万帧，400视频序列），专注于真实遮挡场景的3D人体姿态与形状估计，在该数据集上微调的模型显著提升了遮挡场景下的HPS性能。

**[YOLO-Count: Differentiable Object Counting for Text-to-Image Generation](yolo-count_differentiable_object_counting_for_text-to-image_generation.md)**

:   提出 YOLO-Count，一个基于 YOLO 架构的全可微分开放词汇目标计数模型，通过创新的"基数图"（cardinality map）回归目标和混合强弱监督训练策略，在通用计数和文本到图像生成的数量控制两个任务上均达到 SOTA。

**[YOLOE: Real-Time Seeing Anything](yoloe_realtime_seeing_anything.md)**

:   提出YOLOE，在YOLO架构中统一支持文本提示、视觉提示和无提示三种开放场景的检测和分割，通过RepRTA（可重参数化区域-文本对齐）、SAVPE（语义激活视觉提示编码器）和LRPC（懒惰区域-提示对比）三个设计实现高效率高性能，以3x更少的训练成本在LVIS上超越YOLO-World v2。
