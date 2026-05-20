---
title: >-
  ICCV2025 目标检测方向26篇论文解读
description: >-
  26篇ICCV2025的目标检测方向论文解读，涵盖目标检测、对抗鲁棒、扩散模型、多模态、遥感等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "目标检测"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "扩散模型"
  - "多模态"
  - "遥感"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**📹 ICCV2025** · **26** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (1)](../../ICML2026/object_detection/index.md) · [📷 CVPR2026 (36)](../../CVPR2026/object_detection/index.md) · [🔬 ICLR2026 (9)](../../ICLR2026/object_detection/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/object_detection/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/object_detection/index.md) · [🧪 ICML2025 (8)](../../ICML2025/object_detection/index.md)

🔥 **高频主题：** 目标检测 ×10 · 对抗鲁棒 ×2 · 扩散模型 ×2 · 多模态 ×2 · 遥感 ×2

**[3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection](3dmood_lifting_2d_to_3d_for_monocular_openset_object_detecti.md)**

:   提出首个端到端的单目开放集3D目标检测器3D-MOOD，通过将开放集2D检测"提升"到3D空间，结合几何感知3D query生成与canonical image space设计，在Omni3D闭集和Argoverse 2/ScanNet开集基准上均达到SOTA。

**[Adversarial Attention Perturbations for Large Object Detection Transformers](adversarial_attention_perturbations_for_large_object_detection_transformers.md)**

:   本文提出 AFOG（Attention-Focused Offensive Gradient），一种架构无关的对抗攻击方法，通过可学习注意力机制聚焦扰动到图像脆弱区域，仅需 10 次迭代即可在视觉不可察觉的扰动下将 12 种检测 Transformer 的 mAP 最高降低 37.8 倍，同时在 CNN 检测器上也优于现有方法。

**[Augmenting Moment Retrieval: Zero-Dependency Two-Stage Learning](augmenting_moment_retrieval_zero-dependency_two-stage_learning.md)**

:   提出 AMR 框架，通过 Splice-and-Boost 数据增强策略和冷启动-蒸馏两阶段训练，在不依赖任何外部数据/预训练模型的前提下，大幅提升视频时刻检索的边界感知能力和语义辨别力，在 QVHighlights 上超越 SOTA +5%。

**[Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability](automated_model_evaluation_for_object_detection_via_prediction_consistency_and_r.md)**

:   本文提出PCR（Prediction Consistency and Reliability），一种无需人工标注即可估计目标检测模型性能的自动化评估方法，通过分析NMS前后边界框的空间一致性和置信度可靠性来估计mAP，并构建了基于图像腐蚀的元数据集以实现更现实和可扩展的评估。

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

**[Intervening in Black Box: Concept Bottleneck Model for Enhancing Human-Neural Network Mutual Understanding](intervening_in_black_box_concept_bottleneck_model_for_enhancing_human_neural_net.md)**

:   提出 CBM-HNMU 框架，通过概念瓶颈模型（CBM）逼近黑盒模型的推理过程，自动识别并修正有害概念，再将修正后的知识蒸馏回黑盒模型，实现超越样本级别的系统性模型干预与准确率提升。

**[Large-scale Pre-training for Grounded Video Caption Generation](large-scale_pre-training_for_grounded_video_caption_generation.md)**

:   提出 GROVE 模型和大规模自动标注方法，构建包含 1M 视频的 HowToGround1M 预训练数据集和 3513 个视频的手动标注 iGround 数据集，实现联合视频字幕生成与多目标时空边界框定位，在 iGround、VidSTG、ActivityNet-Entities 等数据集上取得 SOTA。

**[LMM-Det: Make Large Multimodal Models Excel in Object Detection](lmm-det_make_large_multimodal_models_excel_in_object_detection.md)**

:   提出 LMM-Det，通过系统分析发现大型多模态模型在目标检测中核心瓶颈是低召回率，并通过数据分布调整（伪标签增强）和推理优化（按类别逐一检测）将 LMM 的 COCO AP 从 0.2 提升至 47.5，无需任何额外专用检测模块。

**[Measuring the Impact of Rotation Equivariance on Aerial Object Detection](measuring_the_impact_of_rotation_equivariance_on_aerial_object_detection.md)**

:   提出 MessDet，一个基于旋转等变网络的航空目标检测器，通过新型下采样过程实现严格旋转等变性，并引入旋转等变通道注意力和多分支检测头，在 DOTA 等数据集上以极低参数量达到 SOTA 性能。

**[OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images](openrsd_towards_open-prompts_for_object_detection_in_remote_sensing_images.md)**

:   提出OpenRSD通用遥感开放提示目标检测框架，支持文本和图像多模态提示，集成对齐头和融合头平衡速度与精度，配合三阶段训练流水线和47万张图像的ORSD+数据集，在7个公开数据集上取得最优平均性能，同时保持20.8 FPS实时推理。

**[Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)**

:   系统性重新审视 11 种对抗补丁防御方法，建立首个补丁防御基准（含 13 种攻击、11 个检测器、4 种度量），构建 94,000 张图像的大规模 APDE 数据集，并揭示三个关键新发现：自然补丁防御难点在于数据分布而非高频、补丁检测精度与防御性能不一致、自适应攻击可绕过大多数现有防御。

**[SFUOD: Source-Free Unknown Object Detection](sfuod_source-free_unknown_object_detection.md)**

:   提出 Source-Free Unknown Object Detection (SFUOD) 新场景，并设计 CollaPAUL 框架，通过协作调优融合源域和目标域知识 + 基于主轴的未知物体伪标签分配，在无源数据条件下同时检测已知和未知物体。

**[Sim-DETR: Unlock DETR for Temporal Sentence Grounding](sim-detr_unlock_detr_for_temporal_sentence_grounding.md)**

:   系统分析了 DETR 在时序语句定位 (TSG) 任务中的异常行为根因——查询间冲突和查询内全局-局部矛盾，并提出两个简单修改（Query Grouping & Ranking + Global-Local Bridging）构成 Sim-DETR，解锁 DETR 在 TSG 任务的全部潜力。

**[The Devil is in the Spurious Correlations: Boosting Moment Retrieval with Dynamic Learning](the_devil_is_in_the_spurious_correlations_boosting_moment_retrieval_with_dynamic.md)**

:   首次揭示文本查询与视频背景帧之间的虚假相关性是时刻检索性能瓶颈的根本原因，提出 TD-DETR 框架通过动态上下文视频合成和文本-动态交互增强两个策略来缓解该问题，在 QVHighlights 和 Charades-STA 上达到 SOTA。

**[Uncertainty-Aware Gradient Stabilization for Small Object Detection](uncertainty-aware_gradient_stabilization_for_small_object_detection.md)**

:   揭示了传统目标定位方法在小目标上存在因损失曲率陡峭导致的梯度不稳定问题，提出 UGS（不确定性感知梯度稳定化）框架，通过分类式定位 + 不确定性最小化 + 不确定性引导精炼三个组件来稳定梯度，显著提升小目标检测性能。

**[UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement](upre_zero-shot_domain_adaptation_for_object_detection_via_unified_prompt_and_rep.md)**

:   提出 UPRE 框架，通过联合优化多视角域提示（MDP）和统一表示增强（URE）来同时缓解零样本域自适应目标检测中的检测偏差和域偏差，在恶劣天气、跨城市、虚拟到现实三类场景的九个数据集上取得 SOTA 性能。

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
