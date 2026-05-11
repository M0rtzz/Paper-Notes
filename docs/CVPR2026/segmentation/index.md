---
title: >-
  CVPR2026 语义分割方向85篇论文解读
description: >-
  85篇CVPR2026的语义分割方向论文解读，涵盖语义分割、扩散模型、推理、域适应、目标检测、遥感等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✂️ 语义分割

**📷 CVPR2026** · **85** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (4)](../../ACL2026/segmentation/index.md) · [🔬 ICLR2026 (11)](../../ICLR2026/segmentation/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/segmentation/index.md) · [🧠 NeurIPS2025 (47)](../../NeurIPS2025/segmentation/index.md) · [📹 ICCV2025 (74)](../../ICCV2025/segmentation/index.md) · [🧪 ICML2025 (22)](../../ICML2025/segmentation/index.md)

🔥 **高频主题：** 语义分割 ×43 · 扩散模型 ×5 · 推理 ×5 · 域适应 ×4 · 目标检测 ×4

**[3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion](3m-ti_high-quality_mobile_thermal_imaging_via_calibration-free_multi-camera_cros.md)**

:   提出 3M-TI，一个**无需标定**的多相机跨模态扩散框架，通过在 VAE 潜空间中用跨模态自注意力（CSM）自动对齐并融合未标定的 RGB-热红外图像对，结合错位增强策略，在移动端热成像超分辨率任务上达到 SOTA，并显著提升下游目标检测与语义分割性能。

**[MEDISEG: 药物图像实例分割数据集——预防不良药物事件](a_dataset_of_medication_images_with_instance_segme.md)**

:   构建了MEDISEG药物图像实例分割数据集（8262张图像，32类药片，含遮挡/重叠的真实场景），YOLOv8/v9验证3类达99.5% mAP@0.5、32类达80.1%，FsDet few-shot证明MEDISEG预训练在遮挡场景比CURE显著提升（1-shot 0.406 vs 0.131）。

**[A Mixed Diet Makes DINO An Omnivorous Vision Encoder](a_mixed_diet_makes_dino_an_omnivorous_vision_encoder.md)**

:   提出 Omnivorous Vision Encoder，通过轻量级 adapter 在冻结的 DINOv2 之上进行跨模态对齐蒸馏训练（RGB/Depth/Segmentation），使单一编码器对不同视觉模态产生一致嵌入，同时保留原始判别语义。

**[AFRO: Bootstrap Dynamic-Aware 3D Visual Representation for Scalable Robot Learning](bootstrap_dynamic-aware_3d_visual_representation_for_scalable_robot_learning.md)**

:   提出AFRO自监督3D视觉预训练框架，通过逆动力学模型（IDM）推断潜在动作、扩散Transformer前向动力学模型（FDM）预测未来特征、逆一致性约束保证时序对称性，在RH20T大规模数据上预训练后，MetaWorld 14任务平均成功率76.0%（vs DynaMo-3D 64.9%、PointMAE 63.9%），4个real-world任务也取得最优。

**[Combining Boundary Supervision and Segment-Level Regularization for Fine-Grained Action Segmentation](boundary_segment_action_segmentation.md)**

:   提出一种轻量级双损失训练框架用于时序动作分割（TAS），仅增加一个边界输出通道和两个辅助损失（边界回归损失 + CDF 段形状正则化损失），在 MS-TCN、C2F-TCN、FACT 三种架构上一致提升 F1 和 Edit 分数，证明精确分割可以通过简单的损失设计而非更重的架构实现。

**[Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation](brewing_stronger_features_dual-teacher_distillation_for_multispectral_earth_obse.md)**

:   提出**DEO(Distillation for Earth Observation)**，一种双教师对比蒸馏框架——用多光谱自蒸馏教师学习光谱表示、用光学VFM教师（DINOv3）注入高级语义先验，使单一学生网络同时擅长光学和多光谱遥感任务，在语义分割、变化检测和分类上全面达到SOTA。

**[CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)**

:   提出Concept-Aware LoRA (CA-LoRA)，通过自动识别T2I模型中与特定概念（如视角、风格）相关的权重层，仅对这些层施加LoRA微调，实现对目标域的选择性对齐，同时保留预训练模型的多样化生成能力，用于生成高质量的城市场景分割数据集。

**[DeBias-CLIP: CLIP Is Shortsighted — Paying Attention Beyond the First Sentence](clip_shortsighted_beyond_first_sentence.md)**

:   发现CLIP和Long-CLIP模型存在严重的early-token偏向和首句摘要shortcut问题，提出DeBias-CLIP通过去除摘要句、句子子采样和前缀token填充三种简单增强策略，不增加任何额外参数即实现了多个长文本检索基准的SOTA。

**[Comparative Evaluation of Traditional Methods and Deep Learning for Brain Glioma Imaging](comparative_evaluation_of_traditional_methods_and_deep_learning_for_brain_glioma.md)**

:   一篇系统性综述论文，全面对比传统方法（阈值分割、区域生长、模糊聚类等）和深度学习方法（CNN、U-Net、SegNet 等）在脑胶质瘤 MRI 分割与分类任务上的表现，结论指出 CNN 架构在准确性和自动化程度上全面优于传统技术。

**[Concept-Guided Fine-Tuning: Steering ViTs away from Spurious Correlations to Improve Robustness](concept-guided_fine-tuning_steering_vits_away_from_spurious_correlations_to_impr.md)**

:   提出 CFT（Concept-Guided Fine-Tuning），利用 LLM 生成类别级语义概念并通过 GroundedSAM 零样本分割获取概念掩码，再以 AttnLRP 的 relevance map 与概念区域对齐为目标微调 ViT，仅用 1500 张图即可显著提升 5 个 OOD 基准上的鲁棒性。

**[ConceptPrism: Concept Disentanglement in Personalized Diffusion Models via Residual Token Optimization](conceptprism_concept_disentanglement_in_personalized_diffusion_models_via_residu.md)**

:   提出 ConceptPrism，通过引入图像级残余 token 和跨图像排斥损失，在个性化 T2I 扩散模型中自动将共享目标概念与图像特有的残余信息解耦，在 DreamBench 上 CLIP-T/DINO/CLIP-I 全面最优。

**[CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearthsar_a_sarcentric_and_billionscale_geospa.md)**

:   提出首个十亿参数级 SAR 视觉基础模型 CrossEarth-SAR，在 DINOv2 ViT backbone 上将 FFN 替换为物理引导的稀疏 MoE（用方向熵、等效视数、局部粗糙度三个 SAR 物理描述符引导路由选择），配套 200K 级跨域预训练数据集及覆盖 8 种域差异的 22 个基准，在 20/22 个跨域语义分割评测上达到 SOTA。

**[CTFS: Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels](ctfs_collaborative_teacher_framework_for_forward-looking_sonar_image_semantic_se.md)**

:   提出CTFS，首个专为前视声呐图像设计的半监督语义分割框架，引入多教师协作机制（1个通用教师+2个声呐特异教师，分别模拟声学阴影和能量衰减物理特性），配合多视角伪标签可靠性评估（单教师内稳定性+跨教师间一致性），在仅2%标注下达62.32% mIoU，超越SOTA 5.08个百分点。

**[Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training](data_warmup_complexity-aware_curricula_for_efficient_diffusion_training.md)**

:   提出Data Warmup，一种不修改模型或损失函数的课程学习策略，通过语义感知图像复杂度度量（前景显著度×前景典型性）按从简到繁顺序调度训练图像，在ImageNet 256×256上为SiT系列带来IS最高+6.11、FID最低-3.41的改进，且反转课程（先难后简）反而低于均匀基线——证明排序本身是关键机制。

**[DeDelayed: Deleting Remote Inference Delay via On-Device Correction](dedelayed_deleting_remote_inference_delay_via_on-device_correction.md)**

:   提出 DeDelayed 端云协同推理框架，将轻量本地图像模型与延迟感知的云端时序预测视频模型结合，通过时序预测训练补偿网络延迟，在 100ms 延迟下比纯本地推理提升 6.4 mIoU、比纯远程推理提升 9.8 mIoU。

**[Detecting AI-Generated Forgeries via Iterative Manifold Deviation Amplification](detecting_ai-generated_forgeries_via_iterative_manifold_deviation_amplification.md)**

:   提出 IFA-Net，从"建模什么是真"而非"学什么是假"的角度检测 AI 伪造：利用冻结 MAE 重建输入产生残差暴露偏离自然图像流形的区域，再通过两阶段闭环——粗检测→任务自适应先验注入→放大残差→精细化——迭代放大流形偏差，在 diffusion inpainting 和传统篡改检测上均取得 SOTA。

**[Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)**

:   提出一种跳过logits优化过程的开放词汇语义分割方法，基于"同类区域的logits到退化分布的分布差异一致"这一假设，直接通过最优传输路径或最大传输速度的解析解来构造分割图，在8个基准上达到SOTA且无需训练或模型特定调制。

**[DSS: Discover, Segment, and Select for Zero-shot Camouflaged Object Segmentation](discover_segment_and_select_a_progressive_mechanism_for_zero-shot_camouflaged_ob.md)**

:   提出DSS三阶段渐进式pipeline(Discover→Segment→Select)，通过自监督视觉编码器+Leiden聚类发现前景(FOD)、SAM生成候选mask、启发式评分+MLLM成对比较选择最优mask，实现零样本无训练的伪装目标分割，尤其在多实例场景上显著优于现有方法。

**[DPAD: Discriminative Perception via Anchored Description for Reasoning Segmentation](discriminative_perception_via_anchored_description_for_reasoning_segmentation.md)**

:   针对推理分割(RS)中RL+GRPO训练的geometric reward无法约束reasoning chain是否聚焦目标unique attributes的问题，提出DPAD方法：MLLM生成reasoning chain+geometric localization+anchored description，引入基于CLIP的Discriminative Perception Reward比较description与ROI/AOI的相似度差异，迫使caption更具判别性从而间接约束推理链聚焦目标，ReasonSeg上cIoU提升3.09%且推理链长度减少42%。

**[DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime](dsflash_panoptic_scene_graph_realtime.md)**

:   DSFlash 通过合并分割与关系预测 backbone、门控双向关系预测头和 mask-based 动态 patch 剪枝，在 PSG 数据集上以 18ms 延迟（56 FPS）实现 mR@50=30.9 的 SOTA 全景场景图生成。

**[Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgb-d_scene_understanding_via_multi-task_adaptive_learning_and_cross-d.md)**

:   提出一种高效 RGB-D 多任务场景理解网络，通过改进的融合编码器利用通道冗余加速特征提取，设计归一化聚焦通道层（NFCL）和上下文特征交互层（CFIL）进行跨维度特征引导，并引入批级别多任务自适应损失函数动态调整各任务学习权重，在 NYUv2/SUN RGB-D/Cityscapes 上同时完成语义分割、实例分割、朝向估计、全景分割和场景分类五项任务，取得精度与速度的双重优势。

**[ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark](elvis_enhance_low-light_for_video_instance_segmentation_in_the_dark.md)**

:   ELVIS 提出了首个低光视频实例分割（VIS）框架，通过物理驱动的合成低光视频管线（含运动模糊建模）、无标定退化参数估计网络 VDP-Net、以及将增强解码器集成到 VIS 架构中实现退化与内容解耦，在合成和真实低光视频上分别实现 +3.7AP 和 +2.8AP 的提升。

**[EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection](erecu_pseudolabel_evolution_unsupervised_camouflage.md)**

:   提出EReCu统一框架，在DINO师生架构上通过多线索原生感知（MNP）提取纹理+语义先验引导伪标签进化融合（PEF），结合局部伪标签精修（LPR）恢复边界细节，首次统一伪标签引导和特征学习两大UCOD范式，在4个COD数据集上全面SOTA。

**[FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)**

:   提出 FCL-COD 框架，通过频率感知低秩适配（FoRA）将伪装场景知识注入 SAM、梯度感知对比学习（GCL）增强前背景特征分离、多尺度频率注意力（MSFA）提炼边界敏感特征，在仅使用边界框标注的弱监督设定下超越了全监督 SOTA 方法。

**[Follow the Saliency: Supervised Saliency for Retrieval-augmented Dense Video Captioning](follow_the_saliency_supervised_saliency_for_retrieval-augmented_dense_video_capt.md)**

:   提出 STaRC 框架，通过有监督的帧级显著性学习统一驱动检索（显著性引导分割+检索）和描述生成（显著性提示注入解码器），显著提升密集视频描述(DVC)任务中的时序对齐和字幕质量。

**[FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting](fov-net_rotation-invariant_cad_b-rep_learning_via_field-of-view_ray_casting.md)**

:   提出 FoV-Net，首个在 CAD B-rep 学习中同时捕获局部表面几何和全局结构上下文的旋转不变框架，通过局部参考系 UV 网格(LRF UV)和视场光线投射(FoV)描述子实现了在任意 $\mathbf{SO}(3)$ 旋转下的鲁棒分类和分割。

**[From 2D Alignment to 3D Plausibility: Unifying Heterogeneous 2D Priors and Penetration-Free Diffusion for Occlusion-Robust Two-Hand Reconstruction](from_2d_alignment_to_3d_plausibility_unifying_heterogeneous_2d_priors_and_penetr.md)**

:   将双手重建解耦为 2D 结构对齐（融合关键点/分割/深度先验）和 3D 空间交互对齐（穿透消除扩散模型），在 InterHand2.6M 上 MPJPE 达到 5.36mm，大幅超越 SOTA。

**[GenMask: Adapting DiT for Segmentation via Direct Mask Generation](genmask_adapting_dit_for_segmentation_via_direct_mask_generation.md)**

:   本文提出 GenMask，将 DiT 直接训练为生成黑白分割掩码（与生成彩色图像共用同一模型），通过发现二值掩码的 VAE 潜在表示是线性可分的特殊性质，设计了针对分割的极端长尾时间步采样策略，实现了单步推理即可产出分割结果，在 referring 和 reasoning 分割基准上达到 SOTA。

**[GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)**

:   本文提出 GeoGuide，一个层次化几何引导的开放词表 3D 语义分割框架，通过基于不确定性的超点蒸馏、实例级掩码重建和跨实例关系一致性三个互补模块，利用预训练3D模型的几何先验来纠正 2D 到 3D 知识蒸馏中的几何偏差，在 ScanNet v2 上达到 64.8 mIoU 的 SOTA 性能。

**[GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth](geomprompt_rgbd_segmentation.md)**

:   GeomPrompt 为冻结的 RGB-D 分割模型学习轻量的几何提示模块，从 RGB 合成任务驱动的深度代理信号（无深度监督），在深度缺失时提升 6.1 mIoU，在深度退化时提升最高 3.6 mIoU。

**[GeoSURGE: Geo-localization using Semantic Fusion with Hierarchy of Geographic Embeddings](geosurge_geo-localization_using_semantic_fusion_with_hierarchy_of_geographic_emb.md)**

:   GeoSURGE 提出层级地理嵌入和语义融合模块，将全球图像地理定位问题建模为视觉表征与学习得到的地理表征之间的匹配，在 5 个基准的 25 项指标中取得 22 项 SOTA。

**[GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](gkd_generalizable_knowledge_distillation_vfm.md)**

:   提出 GKD 框架，通过将表示学习与任务学习解耦的多阶段蒸馏（先学通用特征 → 冻结编码器 → 再训任务头）+ 查询式软蒸馏机制（QSD），从 VFM 中蒸馏出具有跨域泛化能力的轻量学生模型，在 F2L 设置下平均 mIoU 提升 +10.6%，F2F +1.9%。

**[Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)**

:   本文将无监督域适应中的类别课程学习重新定义为强化学习的序贯决策问题，提出 HeuSCM 框架，通过高维语义状态感知和类别公平策略梯度实现自主学习课程规划，在 ACDC、Dark Zurich 和 Nighttime Driving 上达到 SOTA（72.9 mIoU）。

**[HippoMM: Hippocampal-inspired Multimodal Memory for Long Audiovisual Event Understanding](hippomm_hippocampal-inspired_multimodal_memory_for_long_audiovisual_event_unders.md)**

:   HippoMM 将海马体的三大认知机制——模式分离（情景分割）、记忆固化（语义压缩）和模式补全（层级检索）——映射为计算架构，用于长音视频的情景记忆和跨模态关联回忆，在自建基准 HippoVlog 上达到 78.2% 准确率并比检索增强基线快 5 倍。

**[INSID3: Training-Free In-Context Segmentation with DINOv3](insid3_training-free_in-context_segmentation_with_dinov3.md)**

:   提出INSID3，一种仅依赖冻结DINOv3特征的无训练上下文分割方法，通过位置偏差消除、细粒度聚类和种子聚类聚合三阶段pipeline，在语义/部件/个性化分割任务上以单一自监督骨干网络超越了依赖SAM或微调的方法，平均mIoU提升+7.5%。

**[Kαlos finds Consensus: A Meta-Algorithm for Evaluating Inter-Annotator Agreement in Complex Vision Tasks](kαlos_finds_consensus_a_meta-algorithm_for_evaluating_inter-annotator_agreement_.md)**

:   提出KαLOS元算法，通过"先定位后分类"原则和数据驱动的参数校准，将复杂的空间-类别标注一致性问题转化为标准名义可靠性矩阵，统一评估目标检测、实例分割、姿态估计等多种视觉任务的标注者间一致性（IAA）。

**[Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction](learning_cross-view_object_correspondence_via_cycle-consistent_mask_prediction.md)**

:   提出基于条件二值分割的跨视角物体对应框架 CCMP，通过循环一致性约束提供自监督信号并支持测试时训练 (TTT)，在 Ego-Exo4D 上达到 44.57% mIoU 的 SOTA 性能。

**[LEMMA: Laplacian Pyramids for Efficient Marine Semantic Segmentation](lemma_laplacian_pyramids_for_efficient_marine_semantic_segmentation.md)**

:   提出LEMMA，一种基于拉普拉斯金字塔的轻量级海洋语义分割模型，通过金字塔分解提取边缘信息来替代深层特征计算，在参数量减少71倍的条件下实现了SOTA级别的分割精度（MaSTr1325上98.97% mIoU）。

**[Live Interactive Training for Video Segmentation](live_interactive_training_for_video_segmentation.md)**

:   LIT (Live Interactive Training) 提出了一种让交互式视觉系统（如SAM2）在推理时从用户纠正中在线学习的框架，其轻量实现LIT-LoRA通过实时更新LoRA模块将用户反馈泛化到后续帧，在挑战性VOS基准上减少18-34%用户纠正次数，训练开销仅约0.5秒。

**[LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment](lod-loc_v3_generalized_aerial_localization_in_dense_cities_using_instance_silhou.md)**

:   本文提出LoD-Loc v3，通过构建10万图像的大规模合成实例分割数据集InsLoD-Loc和将定位范式从语义轮廓对齐升级为实例轮廓对齐，解决了基于LoD城市模型的无人机定位中跨场景泛化差和密集城市歧义两大痛点，在Tokyo-LoDv3密集场景上比SOTA的(2m,2°)精度提升2000%。

**[Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation](looking_beyond_the_window_global-local_aligned_clip_for_training-free_open-vocab.md)**

:   针对无训练开放词汇语义分割中滑动窗口带来的跨窗口语义不一致问题，提出 GLA-CLIP 框架，通过全局键值扩展、代理锚点注意力和动态归一化三个机制实现跨窗口全局上下文整合，在8个基准上取得平均 44.0% mIoU 的 SOTA 表现。

**[Love Me, Love My Label: Rethinking the Role of Labels in Prompt Retrieval for Visual In-Context Learning](love_me_love_my_label_rethinking_the_role_of_labels_in_prompt_retrieval_for_visu.md)**

:   揭示了视觉上下文学习（VICL）中 prompt 检索忽略标签信息导致标签不一致的问题，提出 LaPR 框架通过图像-标签联合表示和混合专家机制实现标签感知的 prompt 检索，在前景分割、目标检测和图像着色任务上一致超越 SOTA。

**[Low-Data Supervised Adaptation Outperforms Prompting for Cloud Segmentation Under Domain Shift](low_data_supervised_adaptation_outperforms_prompting_for_cloud_segmentation.md)**

:   本文系统证明了在卫星遥感云分割任务中，提示工程完全无法弥补视觉-语言模型的域差距，而仅需0.1%（约8张图像）的有标签数据进行微调就能超越所有零样本提示策略。

**[Making Training-Free Diffusion Segmentors Scale with the Generative Power](making_training-free_diffusion_segmentors_scale_with_the_generative_power.md)**

:   揭示现有无训练扩散分割方法无法随生成模型能力增强而提升的根本原因——交叉注意力图到语义相关性之间存在两个gap（聚合gap和分数不平衡gap），提出自动聚合（auto aggregation）和逐像素重缩放（per-pixel rescaling）两项技术组成GoCA框架，首次使更强的扩散模型（SDXL、PixArt-Sigma、Flux）在无训练语义分割中显著超越旧模型。

**[MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator](matanyone_2_scaling_video_matting_via_a_learned_quality_evaluator.md)**

:   提出学习型 Matting Quality Evaluator (MQE)，在无 ground-truth 条件下逐像素评估 alpha 质量，既作为在线训练引导又作为离线数据筛选器，构建了 28K 片段 / 240 万帧的真实世界视频抠图数据集 VMReal，配合参考帧训练策略，显著超越所有现有方法。

**[MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention](mixercseg_an_efficient_mixer_architecture_for_crack_segmentation_via_decoupled_m.md)**

:   提出 MixerCSeg，通过解析 Mamba 的隐式注意力机制将通道解耦为全局/局部分支，分别用 Self-Attention 和 CNN 增强，配合方向引导边缘门控卷积，以 2.05 GFLOPs / 2.54M 参数实现裂缝分割 SOTA。

**[MPM: Mutual Pair Merging for Efficient Vision Transformers](mpm_mutual_pair_merging_for_efficient_vision_transformers.md)**

:   提出 Mutual Pair Merging (MPM)，一个无参数、无训练的 ViT token 合并模块，通过互近邻配对+均值融合来减少序列长度，在 ADE20K 上 ViT-Tiny 的 Raspberry Pi 5 延迟降低 60%，H100 上 FlashAttention-2 下吞吐量提升 20%，mIoU 下降控制在 3% 以内。

**[Masked Representation Modeling for Domain-Adaptive Segmentation](mrm_masked_representation_modeling_domain_adaptive.md)**

:   提出 Masked Representation Modeling (MRM)，在编码器输出的潜在特征空间做随机掩码与重建，以像素分类损失监督重建结果，作为即插即用辅助任务在四种 UDA 基线上平均提升 +2.3/+2.8 mIoU (GTA→CS / Synthia→CS)，推理时零额外开销。

**[Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models](occsam_bench_occlusion_robustness_segmentation.md)**

:   提出 OccSAM-Bench 基准，通过合成手术器械遮挡系统评估 SAM 系列模型在内窥镜场景下的鲁棒性，并设计三区域评估协议揭示模型在遮挡下的两种行为模式：遮挡感知型和遮挡无关型。

**[PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation](pca-seg_revisiting_cost_aggregation_for_openvocabulary_semantic_and_part_segmentat.md)**

:   PCA-Seg 提出并行代价聚合(Parallel Cost Aggregation)范式替代传统的串行空间-类别聚合架构，通过专家驱动感知学习(EPL)模块高效整合语义和空间上下文流，并用特征正交解耦(FOD)策略消除两种知识流的冗余，每个并行块仅增加 0.35M 参数即在 8 个开放词汇语义和部件分割基准上达到 SOTA。

**[PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)**

:   PEARL 提出了一种基于 Procrustes 对齐和文本感知拉普拉斯传播的两步推理方法，在不引入额外训练或辅助骨干网络的前提下，通过修正 CLIP 最后一层自注意力中 key-query 的几何失配并利用文本语义引导标签传播，在训练免开放词汇语义分割上达到了新的 SOTA。

**[Phrase-Instance Alignment for Generalized Referring Segmentation](phrase-instance_alignment_for_generalized_referring_segmentation.md)**

:   本文提出 InstAlign，将广义指代分割 (GRES) 重构为实例级推理问题，通过短语-目标对齐 (POA) 损失建立语言短语与视觉实例的细粒度对应关系，并用相关性加权聚合机制统一处理多目标和无目标场景，在 gRefCOCO 上 cIoU 提升 3.22%、N-acc 提升 12.25%。

**[PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation](pixdlm_uav_reasoning_segmentation.md)**

:   本文定义了 UAV Reasoning Segmentation 任务，构建了包含 10K 高分辨率无人机图像和链式推理标注的 DRSeg 基准，并提出了双路径像素级多模态大模型 PixDLM 作为基线。

**[Pointer-CAD: Unifying B-Rep and Command Sequences via Pointer-based Edges & Faces Selection](pointer-cad_unifying_b-rep_and_command_sequences_via_pointer-based_edges_faces_s.md)**

:   提出基于指针 (Pointer) 机制的命令序列表示，将 B-Rep 几何实体（边/面）显式引入自回归 CAD 生成，首次在命令序列方法中支持 chamfer/fillet 操作，同时大幅降低量化误差导致的拓扑错误。

**[Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](promptdriven_lightweight_foundation_model_for_inst.md)**

:   提出SAM FTI-FDet，通过设计一个基于Transformer decoder的自提示生成器（Prompt Generator），让轻量化的TinyViT-SAM自动生成任务相关的query prompt，无需人工交互即可完成货运列车部件的实例级故障检测，在自建数据集上达到74.6 AP_box / 74.2 AP_mask。

**[PRUE: A Practical Recipe for Field Boundary Segmentation at Scale](prue_a_practical_recipe_for_field_boundary_segmentation_at_scale.md)**

:   本文对18个分割和地理空间基础模型（GFM）进行了系统性评估，提出PRUE——一种结合U-Net骨干、复合损失函数和针对性数据增强的农田边界分割方案，在FTW基准上达到76% IoU和47% object-F1，分别比baseline提升6%和9%，同时提出了一套评估部署鲁棒性的新指标。

**[RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)**

:   针对遥感图像中目标尺度变化大的难题，提出区域比例感知的动态自适应显著性目标检测网络 RDNet，通过 Proportion Guidance 动态选择不同大小卷积核组合，结合小波频域交互与交叉注意力定位模块，在三个 ORSI-SOD 数据集上全面超越 SOTA。

**[RealVLG-R1: A Large-Scale Real-World Visual-Language Grounding Benchmark for Robotic Perception and Manipulation](realvlg-r1_a_large-scale_real-world_visual-language_grounding_benchmark_for_robo.md)**

:   提出 RealVLG 框架，包含 11B 级真实世界多粒度标注数据集 RealVLG-11B 和基于强化学习微调的统一模型 RealVLG-R1，首次将视觉语言定位（VLG）与机器人抓取统一到同一范式中，实现从自然语言指令到 bounding box、分割掩码、抓取姿态和接触点的端到端预测，并展现出零样本泛化能力。

**[Reasoning with Pixel-level Precision: QVLM Architecture and SQuID Dataset for Quantitative Geospatial Analytics](reasoning_with_pixel-level_precision_qvlm_architecture_and_squid_dataset_for_qua.md)**

:   提出 QVLM 架构和 SQuID 数据集，通过代码生成+分割模型的解耦设计，在卫星图像上实现像素级精度的定量空间推理，克服了传统 VLM 因 patch embedding 压缩而丢失空间索引的根本限制。

**[RecycleLoRA: Rank-Revealing QR-Based Dual-LoRA Subspace Adaptation for Domain Generalized Semantic Segmentation](recyclelora_rank-revealing_qr-based_dual-lora_subspace_adaptation_for_domain_gen.md)**

:   提出 RecycleLoRA，利用 Rank-Revealing QR 分解(RRQR)系统性地"回收"Vision Foundation Model预训练权重中的子空间结构，通过对次要方向和主要方向分别初始化主/子双适配器，显著提升 LoRA 的表示多样性和参数利用效率，在合成到真实和真实到真实的域泛化语义分割任务上均达到 SOTA（平均 mIoU 68.95 / 72.10）。

**[REL-SF4PASS: Panoramic Semantic Segmentation with REL Depth Representation and Spherical Fusion](rel-sf4pass_panoramic_semantic_segmentation_with_rel_depth_representation_and_sp.md)**

:   提出 REL 深度表示（基于柱面坐标系的 Rectified Depth + EGVIA + LOA 三通道）和球面动态多模态融合（SMMF），用于全景语义分割，在 Stanford2D3D 上实现 63.06% 平均 mIoU（比 HHA 基线提升 2.35%），并将面对 3D 扰动时的性能方差降低约 70%。

**[RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)**

:   本文提出 RobotSeg，第一个同时支持图像和视频的机器人分割基础模型，基于 SAM 2 引入结构增强记忆关联器（SEMA）、机器人提示生成器（RPG）和标签高效训练策略，仅需首帧标注即可训练，在自动模式下 Whole Robot 分割达到 85.1 J&F，比 SAM 2.1 微调版高 4.9 分，同时参数仅 41.3M（远小于现有 638M+ 方案）。

**[RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation](rs-ssm_refining_forgotten_specifics_in_state_space_model_for_video_semantic_segm.md)**

:   提出 RS-SSM，通过频域分析提取各通道的特定信息分布特征（CwAP），并自适应反转遗忘门矩阵来补充性精炼 SSM 状态空间压缩时丢失的时空细节（FGIR），在 4 个视频语义分割基准上达到 SOTA 且保持高效率。

**[RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)**

:   提出两阶段 RGB-T 显著性检测网络 RSONet：先通过区域引导阶段计算 RGB/热红外引导图与联合引导图的相似度，选出更可靠的模态；再在显著性生成阶段利用选择性优化融合双模态特征，配合密集细节增强和互信息语义模块生成高质量显著图，在三个 RGB-T 基准上取得 SOTA 性能。

**[SAP: Segment Any 4K Panorama](sap_segment_any_4k_panorama.md)**

:   提出 SAP（Segment Any 4K Panorama），通过将全景图转化为沿球面固定轨迹采样的透视伪视频序列，解决 SAM2 流式记忆机制在 360° 图像上的结构性失配问题，并合成 183K 实例标注的 4K 全景图进行微调，在真实世界全景基准上实现零样本 mIoU +17.2 的提升。

**[SARMAE: Masked Autoencoder for SAR Representation Learning](sarmae_masked_autoencoder_for_sar_representation_learning.md)**

:   提出 SARMAE 框架，通过百万级 SAR 数据集 SAR-1M、散斑感知表征增强 (SARE) 和光学语义锚约束 (SARC)，实现噪声鲁棒的 SAR 自监督预训练，在分类、检测和分割多个下游任务上取得 SOTA。

**[SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](scope_scene-contextualized_incremental_few-shot_3d_segmentation.md)**

:   SCOPE 提出一种即插即用的背景引导原型增强框架，利用基础训练场景中背景区域的伪实例构建原型库，在增量阶段通过检索+注意力融合增强少样本原型，无需重训骨干或增加参数即可在 ScanNet/S3DIS 上显著提升新类 IoU（最高 +6.98%）并保持低遗忘。

**[SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)**

:   SDDF 提出开放词汇伪装目标检测（OVCOD）新任务，构建了 OVCOD-D 基准，通过子描述主成分对比融合策略去除冗余文本噪声，以及特异性引导的区域弱对齐和动态聚焦机制增强伪装目标与背景的区分能力，在开集设置下达到 56.4 AP。

**[Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)**

:   提出 EDA-PSeg 框架，通过图匹配适配器（GMA）和欧拉-边际注意力（EMA）两个核心模块，首次实现从针孔视图到 360° 全景图像的开放集无监督域自适应语义分割，同时处理几何视场角畸变和未知类别发现。

**[SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semisupervised_framework.md)**

:   本文提出SemiTooth框架，通过多教师-多学生架构和更严格的加权置信度约束（SWC），解决多源CBCT数据在半监督牙齿分割中的分布差异问题，在构建的MS3Toothset数据集上取得SOTA。

**[SemLayer: Semantic-aware Generative Segmentation and Layer Construction for Abstract Icons](semlayer_semantic-aware_generative_segmentation_and_layer_construction_for_abstr.md)**

:   提出 SemLayer，一个基于生成模型的流水线，将扁平化的矢量图标恢复为语义化分层结构——先通过扩散模型将分割重新定义为上色任务，再进行遮挡区域的语义补全，最后用整数线性规划确定层级顺序，实现 mIoU +5.0、PQ +16.7 的分割提升。

**[SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)**

:   提出 SGMA 框架，通过语义引导融合（SGF）模块构建全局语义原型实现自适应跨模态融合，并通过模态感知采样（MAS）模块动态提升脆弱模态的训练频率，解决遥感场景下不完整多模态语义分割中的模态不平衡、类内方差大和跨模态异质性三大挑战。

**[SouPLe: Enhancing Audio-Visual Localization and Segmentation with Learnable Prompt Contexts](souple_enhancing_audio-visual_localization_and_segmentation_with_learnable_promp.md)**

:   提出 SouPLe (Sound-aware Prompt Learning)，通过将CLIP中固定的文本提示替换为基于图像特征生成的可学习上下文tokens，增强音频嵌入token与视觉特征之间的语义对应，在VGG-SS上cIoU提升3.75、开放集设定下cIoU提升6.32，全面超越先前方法。

**[SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation](spar_single-pass_any-resolution_vit_for_open-vocabulary_segmentation.md)**

:   提出 SPAR，一种通过将细步幅滑窗教师的空间推理能力蒸馏到单次前向传递学生的方法，将 ViT 变为分辨率无关的密集特征提取器，在开放词汇分割中比单次前向基线提升 10.5 mIoU，同时比教师快 52 倍。

**[Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation](spatio-semantic_expert_routing_architecture_with_mixture-of-experts_for_referrin.md)**

:   提出 SERA 框架，在冻结的视觉-语言骨干网络中引入两阶段轻量级 MoE 专家精炼（骨干级 SERA-Adapter + 融合级 SERA-Fusion），通过表达式引导的自适应路由实现参考图像分割中的空间一致性和边界精度提升，仅更新不到 1% 的骨干参数。

**[Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation](task-oriented_data_synthesis_and_control-rectify_sampling_for_remote_sensing_sem.md)**

:   本文提出TODSynth框架，通过MM-DiT的统一三模态注意力实现文本-图像-掩码联合控制的遥感图像合成，并创新性地提出控制-校正流匹配（CRFM）方法，在采样阶段利用下游分割模型的语义损失动态调整生成轨迹，使合成数据在FUSU-4k和LoveDA上分别提升4.14%和2.08%的mIoU。

**[The Golden Subspace: Where Efficiency Meets Generalization in Continual Test-Time Adaptation](the_golden_subspace_where_efficiency_meets_generalization_in_continual_test-time.md)**

:   提出 GOLD 框架用于持续测试时适应（CTTA），核心发现是最小特征更新子空间（"黄金子空间"）与分类器权重行空间一致且天然低秩；通过 Average Gradient Outer Product (AGOP) 在线估计该子空间，结合轻量缩放向量进行特征适应，在分类和分割基准上以极低计算开销达到 SOTA 性能。

**[Towards Context-Aware Image Anonymization with Multi-Agent Reasoning](towards_context-aware_image_anonymization_with_multi-agent_reasoning.md)**

:   提出 CAIAMAR 多智能体框架，将预定义的高置信度直接 PII（人体、车牌）处理与基于大视觉语言模型的上下文感知推理相结合，通过 PDCA 迭代优化循环检测间接隐私标识符，使用扩散模型进行外观去相关修复，在 CUHK03-NP 上将行人重识别风险降低 73%，同时在 CityScapes 上保持 FID 9.1 的高图像质量。

**[Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels](towards_high-quality_image_segmentation_improving_topology_accuracy_by_penalizin.md)**

:   提出 Same Class Neighbor Penalization (SCNP)，通过在训练时将每个像素的 logit 替换为其同类邻域中最差预测，迫使模型优先修复邻域中的弱分类像素，从而以极低代价（仅 3 行代码、几毫秒/迭代）显著提升分割的拓扑精度。

**[Unified Spherical Frontend: Learning Rotation-Equivariant Representations of Spherical Images from Any Camera](unified_spherical_frontend_learning_rotation-equivariant_representations_of_sphe.md)**

:   USF 提出了一个模块化、镜头无关的球面视觉前端，通过将任意标定相机图像投影到单位球面上执行空间域球面重采样、卷积和池化操作，仅用距离加权核就能天然保证旋转等变性，在分类、检测和分割任务上展现了对随机旋转和跨镜头的零样本泛化鲁棒性。

**[Universal 3D Shape Matching via Coarse-to-Fine Language Guidance](universal_3d_shape_matching_via_coarse-to-fine_language_guidance.md)**

:   提出 UniMatch，一个语义感知的粗到细 3D 形状匹配框架：粗阶段通过类别无关 3D 分割 + MLLM 命名 + FG-CLIP 语言嵌入建立部件级对应；细阶段通过组级排序对比损失(Group-wise RnC Loss)在扩展的函数映射框架中学习稠密对应，实现跨类别、非等距形状的通用匹配。

**[UnrealPose: Leveraging Game Engine Kinematics for Large-Scale Synthetic Human Pose Data](unrealpose_leveraging_game_engine_kinematics_for_large-scale_synthetic_human_pos.md)**

:   提出 UnrealPose-Gen，一个基于 Unreal Engine 5 的合成人体姿态数据生成管线，利用游戏引擎原生骨骼运动学（而非 SMPL）生成百万级标注数据集 UnrealPose-1M，提供 3D 关节、2D 关键点、遮挡标志、实例分割掩码和相机参数等完整标注。

**[VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](videomt_your_vit_is_secretly_also_a_video_segmentation_model.md)**

:   提出 VidEoMT，一种纯编码器（encoder-only）视频分割架构，通过 query propagation 和 query fusion 将分割与时序关联统一在单个 ViT 编码器中，在保持与 SOTA 可比精度的同时实现 5×–10× 加速（ViT-L 达 160 FPS）。

**[VIRST: Video-Instructed Reasoning Assistant for SpatioTemporal Segmentation](virst_video-instructed_reasoning_assistant_for_spatiotemporal_segmentation.md)**

:   VIRST 提出端到端框架将全局视频推理和像素级 mask 预测统一在单个视觉语言模型中，通过时空融合（STF）和时序动态锚点更新器（TDAU）实现时空一致的视频分割，在 ReVOS 上 J&F 达 70.8（+7.5 over SOTA），MeViS 62.9（+9.2），同时推理速度 5.1 FPS（比 VRS-HQ 快 1.3 倍）。

**[Weakly-Supervised Referring Video Object Segmentation through Text Supervision](wsrvos_weakly_supervised_rvos.md)**

:   提出 WSRVOS，首个仅使用文本表达式作为监督信号的弱监督指称视频目标分割框架，通过 MLLM 驱动的对比表达式增强、双向视觉-语言特征选择、实例感知表达式分类和时序分段排序约束，显著减少了对像素级标注的依赖。
