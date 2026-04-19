---
title: >-
  CVPR2025 语义分割方向 54篇论文解读
description: >-
  54篇CVPR2025 语义分割方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✂️ 语义分割

**📷 CVPR2025** · **54** 篇论文解读

**[2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification](2dmamba_efficient_state_space_model_for_image_representation_with_applications_o.md)**

:   提出2DMamba，首个具有高效并行算法的**原生2D选择性状态空间模型**，通过保持2D空间连续性（而非展平为1D序列）来建模WSI中的patch间关系，在10个公共病理数据集上全面超越1D Mamba方法，并在ImageNet分类和ADE20K分割上也有提升。

**[A Distractor-Aware Memory for Visual Object Tracking with SAM2](a_distractor-aware_memory_for_visual_object_tracking_with_sam2.md)**

:   提出SAM2.1++的干扰物感知记忆模型（DAM），将SAM2的记忆拆分为近期外观记忆（RAM，确保分割精度）和干扰物解析记忆（DRM，确保跟踪鲁棒性），通过内省式更新策略检测干扰物并自动存储锚帧，在7个基准上设立新SOTA。

**[SAIL: Assessing and Learning Alignment of Unimodal Vision and Language Models](assessing_and_learning_alignment_of_unimodal_vision_and_language_models.md)**

**[Audio-Visual Instance Segmentation](audio-visual_instance_segmentation.md)**

**[G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)**

:   提出 G2HFNet，通过多尺度细节增强 (MDE)、双分支几何-粒度互补 (DGC)、深层语义感知 (DSP) 和局部-全局引导融合 (LGF) 四个模块，针对不同层级特征设计差异化优化策略，在三个遥感显著性检测数据集上全面超越 SOTA。

**[Condensing Action Segmentation Datasets via Generative Network Inversion](condensing_action_segmentation_datasets_via_generative_network_inversion.md)**

**[Continuous Locomotive Crowd Behavior Generation](continuous_locomotive_crowd_behavior_generation.md)**

:   生成连续的人群运动行为，实现轨迹和动作的联合合成，产生自然且多样的群体运动模式

**[COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training](cosmos_cross-modality_self-distillation_for_vision_language_pre-training.md)**

:   COSMOS 提出了一种跨模态自蒸馏框架，通过文本裁剪策略和交叉注意力模块在学生-教师结构中学习细粒度的跨模态表征，在仅使用 30M 数据预训练的情况下，在零样本检索、分类和语义分割任务上全面超越 CLIP 类基线，甚至超越在数十亿数据上训练的 OpenCLIP。

**[CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearth-sar_a_sar-centric_and_billion-scale_geospatial_foundation_model_for_d.md)**

:   提出首个十亿参数级 SAR 视觉基础模型 CrossEarth-SAR，基于物理引导的稀疏混合专家 (MoE) 架构，构建了包含 200K 图像的训练集和 22 个子基准的评估体系，在 20/22 个跨域语义分割基准上达到 SOTA。

**[DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception](declip_decoupled_learning_for_open-vocabulary_dense_perception.md)**

:   DeCLIP 发现 CLIP 的自注意力中存在"代理 token"现象导致图像 token 无法聚合空间相关信息，提出将自注意力模块解耦为"内容"和"上下文"特征并分别用 CLIP 自蒸馏和视觉基础模型蒸馏进行优化的框架，在开放词汇目标检测和语义分割上全面超越现有方法。

**[DefMamba: Deformable Visual State Space Model](defmamba_deformable_visual_state_space_model.md)**

:   DefMamba 提出了一种基于可变形机制的视觉状态空间模型，通过可变形扫描策略动态调整扫描路径（参考点偏移 + 扫描顺序偏移），克服了现有 Visual Mamba 方法使用固定扫描顺序导致的空间结构信息丢失问题，在 ImageNet 分类、COCO 检测和 ADE20K 分割上达到 SOTA。

**[DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation](dformerv2_geometry_self-attention_for_rgbd_semantic_segmentation.md)**

:   提出将深度图作为几何先验而非通过神经网络编码，设计几何自注意力（GSA）将深度距离和空间距离融合为衰减因子调制注意力权重，以约一半 FLOPs 匹配或超越双编码器 RGBD 分割方法。

**[DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)**

:   提出 dino.txt，通过冻结 DINOv2 视觉编码器 + 从头训练文本编码器的 LiT 策略，创新性地用 [CLS]+平均池化拼接作为图像表征，结合文本+图像双模态数据平衡，仅用 50K 迭代（CLIP 训练成本的几分之一）即在零样本分类和开放词汇分割上达到 SOTA。

**[DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation](dpseg_dual-prompt_cost_volume_learning_for_open-vocabulary_semantic_segmentation.md)**

:   DPSeg 提出在开放词汇语义分割中同时利用文本提示和 Stable Diffusion 生成的视觉提示来构建双提示代价体积，通过多尺度视觉代价体积引导解码器和两轮推理的语义精炼策略，在 5 个公开数据集上全面超越现有方法。

**[Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation](dual-agent_optimization_framework_for_cross-domain_few-shot_segmentation.md)**

:   提出 Dual-Agent Optimization (DATO) 框架，包含一致性互聚合（CMA）模块学习跨域不变特征以增强表示，以及相关性修正策略（CRS）将 support-query 匹配转移到域不敏感的特征空间，有效提升跨域小样本分割的泛化能力。

**[Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics](dynamic_derivation_and_elimination_audio_visual_segmentation_with_enhanced_audio.md)**

:   DDESeg 从音频的本质特性出发，针对混合音频的特征混淆和同物体不同声音的类内变异两大问题，提出动态推导模块从混合信号中衍生独立声源表征并增强判别性，再通过动态消除模块过滤掉画外音等无关音频语义，在 AVS 所有基准上取得 SOTA。

**[EdgeTAM: On-Device Track Anything Model](edgetam_on-device_track_anything_model.md)**

:   EdgeTAM 通过详细的延迟分析发现 SAM 2 的瓶颈在记忆注意力而非图像编码器，提出 2D Spatial Perceiver 将帧级记忆从 64×64 维压缩到 ~500 个 token（保留空间结构），配合两阶段知识蒸馏，在 iPhone 15 Pro Max 上实现 16 FPS 的实时 Track Anything。

**[EditAR: Unified Conditional Generation with Autoregressive Models](editar_unified_conditional_generation_with_autoregressive_models.md)**

:   提出 EditAR——首个将图像编辑（纹理修改、物体替换/移除、局部编辑）和图像翻译（深度/边缘/分割图到图像）统一在单一自回归框架中的方法，通过在 LlamaGen 基础上引入条件图像 token 前缀和 DINOv2 蒸馏损失，在标准 next-token prediction 范式下即可对多种条件生成任务取得与专用模型竞争的性能。

**[Effective SAM Combination for Open-Vocabulary Semantic Segmentation](effective_sam_combination_for_open-vocabulary_semantic_segmentation.md)**

:   提出 ESC-Net，一种单阶段开放词汇语义分割模型，通过从 CLIP 图像-文本相关性图中生成伪提示（pseudo prompts）并将其嵌入预训练 SAM 解码器 block 中，高效利用 SAM 的类无关分割能力来增强空间聚合，配合 Vision-Language Fusion (VLF) 模块实现精确的掩码预测，在 ADE20K、PASCAL-VOC、PASCAL-Context 上均取得 SOTA 性能。

**[Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgb-d_scene_understanding_via_multi-task_adaptive_learning_and_cross-d.md)**

:   提出一个高效 RGB-D 多任务场景理解网络，通过改进融合编码器利用冗余特征加速推理，引入归一化聚焦通道层 (NFCL) 和上下文特征交互层 (CFIL) 进行跨维度特征引导，并设计多任务自适应损失函数动态调整任务权重，在 NYUv2/SUN RGB-D/Cityscapes 上达到 SOTA。

**[Exploiting Temporal State Space Sharing for Video Semantic Segmentation](exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)**

:   提出 TV3S（Temporal Video State Space Sharing）架构，利用 Mamba 状态空间模型实现跨视频帧的高效时序信息共享，通过独立处理空间 patch 并结合 shifted window 机制实现高度并行化计算，在 VSPW 和 Cityscapes 数据集上以良好的精度-效率平衡超越了现有的 Transformer 和 RNN 方法。

**[Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation](exploring_clips_dense_knowledge_for_weakly_supervised_semantic_segmentation.md)**

:   ExCEL 提出利用 patch-text 对齐范式（而非传统 image-text 对齐）挖掘 CLIP 的密集知识用于弱监督语义分割，通过文本语义扩充（TSE）和视觉校准（VC）两个模块增强密集对齐能力，在仅需 3.2GB 显存和 6% 训练时间的条件下，在 PASCAL VOC 和 MS COCO 上大幅超越 SOTA。

**[F-LMM: Grounding Frozen Large Multimodal Models](f-lmm_grounding_frozen_large_multimodal_models.md)**

:   F-LMM 冻结现成 LMM 的所有参数，仅训练轻量 CNN mask decoder 将 LMM 注意力图中固有的词-像素对应关系翻译为分割 mask，在完全保持对话能力的同时获得 competitive 的视觉定位性能。

**[Fine-Grained Image-Text Correspondence with Cost Aggregation for Open-Vocabulary Part Segmentation](fine-grained_image-text_correspondence_with_cost_aggregation_for_open-vocabulary.md)**

:   PartCATSeg 通过将物体级和部件级的图文代价体积解耦聚合、引入组合损失约束部件构成关系、并利用 DINO 特征提供结构引导，在多个开放词汇部件分割基准上将 h-IoU 提升超过 10%。

**[FineCaption: Compositional Image Captioning Focusing on Wherever You Want at Any Granularity](finecaption_compositional_image_captioning_focusing_on_wherever_you_want_at_any_.md)**

:   FineCaption 提出一种支持任意 mask 引用和高分辨率图像输入的视觉语言模型，结合 mask 感知 CLIP 编码器、ConvNeXT 和 SAM 高分辨率编码器，以及新构建的 CompositionCap 数据集，实现了多粒度组合式区域图像描述任务。

**[Foveated Instance Segmentation](foveated_instance_segmentation.md)**

:   FSNet 提出一种模拟人眼中央凹视觉机制的实例分割框架，通过可学习的显著性图引导非均匀下采样，在注视目标区域保持高分辨率细节、在外围降低分辨率，实现了在不同预训练分割网络上的即插即用式效率提升。

**[Fractal Calibration for Long-Tailed Object Detection](fractal_calibration_for_long-tailed_object_detection.md)**

:   提出 FRACAL（FRActal CALibration），一种无需训练的后处理方法，首次将分形维数引入长尾目标检测的后校准中，通过对称校准频率轴（类别频率）和空间轴（类别位置均匀度），在 LVIS 数据集上将稀有类 mask AP 提升高达 8.6%，并在 COCO、V3Det、OpenImages 上展示泛化性。

**[Frequency Dynamic Convolution for Dense Image Prediction](frequency_dynamic_convolution_for_dense_image_prediction.md)**

:   FDConv 从频率域角度重新设计动态卷积，通过傅里叶不相交权重（FDW）在不增加参数的前提下构建频率多样的卷积核，结合核空间调制（KSM）和频带调制（FBM）实现精细的频率自适应，仅增加 3.6M 参数即超越需要 65-90M 额外参数的现有动态卷积方法。

**[Generative Video Propagation](generative_video_propagation.md)**

:   提出 GenProp 框架，通过选择性内容编码器（SCE）与 I2V 生成模型的配合，将首帧编辑统一传播到整个视频，在一个模型中同时支持视频编辑、目标去除、目标插入、目标跟踪等多种视频任务。

**[GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation](glus_global-local_reasoning_unified_into_a_single_large_language_model_for_video.md)**

:   提出GLUS框架，通过"上下文帧（全局推理）+ 查询帧（局部追踪）"的帧划分策略，将全局理解和局部时序一致性统一到单个MLLM中，结合端到端训练的VOS记忆库模块，在MeViS上大幅超越所有MLLM-based方法（J&F 51.3%）。

**[Golden Cudgel Network for Real-Time Semantic Segmentation](golden_cudgel_network_for_real-time_semantic_segmentation.md)**

:   提出 GCNet，核心是 Golden Cudgel Block (GCBlock)，训练时自膨胀（多卷积多路径）提升学习能力，推理时自收缩（重参数化为单个 3×3 卷积）加速推理，无需外部教师模型即成为"自蒸馏"方案，在 Cityscapes 上以 77.3% mIoU / 193.3 FPS 超越现有实时分割模型。

**[GroupMamba: Efficient Group-Based Visual State Space Model](groupmamba_efficient_group-based_visual_state_space_model.md)**

:   提出 Modulated Group Mamba 层，将输入通道分为四组分别按四个方向执行单向 SSM 扫描，通过 Channel Affinity Modulation（CAM）增强跨组通道交互，配合蒸馏训练目标解决大模型不稳定问题，在 ImageNet-1K 上以 23M 参数达到 83.3% Top-1 精度。

**[HFP-SAM: Hierarchical Frequency Prompted SAM for Efficient Marine Animal Segmentation](hfp-sam_hierarchical_frequency_prompted_sam_for_efficient_marine_animal_segmenta.md)**

:   HFP-SAM 提出分层频率提示的 SAM 框架，通过频率引导适配器（FGA）注入海洋场景信息、频率感知点选择（FPS）自动生成高质量点提示、全视图 Mamba（FVM）高效解码，在四个海洋动物分割数据集上取得 SOTA。

**[Hierarchical Compact Clustering Attention (COCA) for Unsupervised Object-Centric Learning](hierarchical_compact_clustering_attention_coca_for_unsupervised_object-centric_l.md)**

:   COCA-Net 提出基于物理紧凑性（compactness）的层级聚类注意力层，通过自底向上的层级合并策略发现物体中心，解决了 Slot Attention 在初始化敏感性、slot 数量预设和背景分割等方面的固有缺陷，在六个无监督物体发现数据集上达到 SOTA。

**[Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity](holmes-vau_towards_long-term_video_anomaly_understanding_at_any_granularity.md)**

:   本文提出 Holmes-VAU，构建了包含 70k+ 多粒度标注的视频异常理解基准 HIVAU-70k，并设计异常聚焦时序采样器（ATS）让多模态 VLM 集中关注异常密集区域，在长视频异常检测和推理任务上大幅超越现有方法。

**[Image Quality Assessment: From Human to Machine Preference](image_quality_assessment_from_human_to_machine_preference.md)**

:   本文首次提出面向机器视觉的图像质量评估（IQA for MVS），构建了包含 225 万细粒度标注和 3 万参考/失真图像对的 Machine Preference Database (MPD)，实验证明现有 HVS-centric IQA 指标无法准确表征机器偏好，揭示了人类与机器视觉系统间的根本性差异。

**[Leveraging 3D Geometric Priors in 2D Rotation Symmetry Detection](leveraging_3d_geometric_priors_in_2d_rotation_symmetry_detection.md)**

:   本文提出了一个利用3D几何先验的旋转对称性检测模型，通过在3D空间中直接预测旋转中心和顶点并投影回2D，结合基于种子点和旋转轴的顶点重建模块，在DENDI数据集上以F1-score 33.2超越了之前基于分割的SOTA方法EquiSym (22.5)。

**[LiVOS: Light Video Object Segmentation with Gated Linear Matching](livos_light_video_object_segmentation_with_gated_linear_matching.md)**

:   提出 LiVOS——首个使用门控线性注意力替代 softmax 注意力进行内存匹配的轻量 VOS 网络，将时空注意力矩阵压缩为恒定大小的 2D 状态矩阵，实现任意长视频的恒定内存占用，并在 32G 消费级 GPU 上支持 4096p 推理。

**[MambaOut: Do We Really Need Mamba for Vision?](mambaout_do_we_really_need_mamba_for_vision.md)**

:   本文通过概念分析指出 Mamba 的 SSM 机制适用于长序列+自回归任务，而 ImageNet 图像分类两者都不满足，因此构建了去掉 SSM 的 MambaOut（纯 Gated CNN）系列模型，在图像分类上全面超越所有视觉 Mamba 模型，有力证明了 SSM 对视觉分类是不必要的。

**[MambaVision: A Hybrid Mamba-Transformer Vision Backbone](mambavision_a_hybrid_mamba-transformer_vision_backbone.md)**

:   NVIDIA 提出 MambaVision——首个系统研究 Mamba 与 Transformer 混合方式的视觉骨干网络，通过重设计的 MambaVision Mixer + 在最后几层加入 self-attention 来弥补 SSM 的全局上下文不足，在 ImageNet-1K 上达到精度-吞吐量的新 Pareto 前沿，同时在检测和分割下游任务中也优于同等规模的竞争模型。

**[MammAlps: A Multi-view Video Behavior Monitoring Dataset of Wild Mammals in the Swiss Alps](mammalps_a_multi-view_video_behavior_monitoring_dataset_of_wild_mammals_in_the_s.md)**

:   本文提出 MammAlps——一个来自瑞士国家公园的多模态多视角野生哺乳动物行为监测数据集（8.5 小时稠密标注，5 个物种，11 种活动 + 19 种动作），以及两个基准任务：多模态物种+层级行为识别（B1）和首个多视角长期事件理解（B2），填补了野生动物视频行为分析在层级行为标注、多模态和多视角方面的空白。

**[Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation](mask-adapter_the_devil_is_in_the_masks_for_open-vocabulary_segmentation.md)**

:   揭示了开放词汇分割中 mask pooling 方法的性能上界瓶颈——精确 mask 往往无法获得准确分类，提出 Mask-Adapter 从 proposal mask 和 CLIP 特征中提取语义激活图来替代直接 mask pooling，以即插即用方式显著提升多种 OVS 方法的分类准确率。

**[MaSS13K: A Matting-level Semantic Segmentation Benchmark](mass13k_a_matting-level_semantic_segmentation_benchmark.md)**

:   构建了包含 13,348 张 4K 分辨率图像的 matting 级语义分割数据集 MaSS13K（掩码复杂度比现有数据集高 20-50 倍），并提出 MaSSFormer 模型通过双分支像素解码器（全局语义 + 局部结构）在保持计算效率的同时实现了高分辨率场景下精细边界的高质量分割。

**[MatAnyone: Stable Video Matting with Consistent Memory Propagation](matanyone_stable_video_matting_with_consistent_memory_propagation.md)**

:   提出 MatAnyone 框架，通过区域自适应记忆融合机制在记忆空间中实现一致性传播（核心区域保持语义稳定，边界区域捕获精细 alpha 细节），配合新数据集 VM800 和利用分割数据直接监督 matting head 的训练策略，实现了鲁棒且高质量的目标指定视频抠图。

**[A Two-Stage Progressive Pre-training using Multi-Modal Contrastive Masked Autoencoders](multi-modal_contrastive_masked_autoencoders_a_two-stage_progressive_pre-training.md)**

:   本文提出渐进式两阶段预训练策略——第一阶段用patch级对比学习对齐RGB和深度模态的跨模态表示，第二阶段用掩码自编码+受扩散模型启发的去噪+特征蒸馏联合训练，在ScanNet语义分割上比Mask3D提升+1.3% mIoU，在多个RGB-D下游任务上达到SOTA。

**[PicoSAM3: Real-Time In-Sensor Region-of-Interest Segmentation](picosam3_real-time_in-sensor_region-of-interest_segmentation.md)**

:   PicoSAM3 是一个 1.3M 参数的超轻量可提示分割模型，通过 ROI 隐式提示编码、密集 CNN 架构（无 Transformer）、SAM3 知识蒸馏和 INT8 量化，在 COCO 上达 65.45% mIoU，并实现在 Sony IMX500 视觉传感器上 11.82ms 实时推理。

**[Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](prompt-driven_lightweight_foundation_model_for_instance_segmentation-based_fault.md)**

:   SAM FTI-FDet 提出基于轻量 SAM 的自动提示实例分割框架，通过 Transformer 解码器式的提示生成器自动产生任务相关提示、自适应特征分发器融合多尺度特征、TinyViT backbone 降低计算开销，在货运列车故障检测数据集上达 74.6 $AP^{box}$ / 74.2 $AP^{mask}$。

**[RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)**

:   RDNet 针对遥感图像中目标尺度剧烈变化的问题，提出区域比例感知的动态自适应显著性检测网络，通过动态自适应细节感知模块（DAD，根据目标区域比例选择不同大小卷积核组合）、频率匹配上下文增强模块（FCE，小波域特征交互）和区域比例感知定位模块（RPL，交叉注意力+比例引导），在 EORSSD/ORSSD/ORSI-4199 三个数据集上取得 SOTA。

**[RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)**

:   提出区域引导选择性优化网络 RSONet，通过两阶段（区域引导+显著性生成）解决 RGB 与热红外图像中显著区域不一致问题，利用相似度分数自动选择信息更准确的模态主导后续融合。

**[SAP: Segment Any 4K Panorama](sap_segment_any_4k_panorama.md)**

:   将 360° 全景图分割重新定义为透视视频分割问题，通过沿 zigzag 轨迹分解全景图为重叠 patch 序列并微调 SAM2 的 memory 模块，配合 183K 合成 4K 全景图的大规模训练，实现零样本全景分割 +17.2 mIoU 的提升。

**[SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories](segagent_exploring_pixel_understanding_capabilities_in_mllms_by_imitating_human_.md)**

:   SegAgent 将 referring expression segmentation 建模为人类标注员的迭代操作过程——MLLM 观察当前 mask 状态后预测下一个点击位置，交互式分割模型据此更新 mask，经过多轮迭代得到最终分割结果；通过 StaR+ 策略改进和 PRM+树搜索，在复杂场景下大幅提升分割精度。

**[SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)**

:   提出SGMA框架，通过语义引导融合(SGF)模块构建全局语义原型估计模态鲁棒性并自适应加权融合，以及模态感知采样(MAS)模块动态优先训练脆弱模态，解决遥感不完整多模态分割中的模态不平衡、类内变化和跨模态异质性三大挑战。

**[SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)**

:   提出SPARROW框架，通过目标特定跟踪特征(TSF)和双提示(BOX+SEG)机制，解决视频MLLM中时序引用一致性差和首帧初始化不稳定的问题，在6个基准上对3个主流视频MLLM均取得一致提升。

**[Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation](spatio-semantic_expert_routing_architecture_with_mixture-of-experts_for_referrin.md)**

:   提出 SERA 框架，在预训练视觉语言模型中引入轻量级表达感知的混合专家（MoE）精细化，分别在 backbone 层（SERA-Adapter）和融合层（SERA-Fusion）进行专家路由，仅更新 <1% 参数即在参考图像分割基准上达到 SOTA。
