---
title: >-
  ECCV2024 目标检测方向 42篇论文解读
description: >-
  42篇ECCV2024 目标检测方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**🎞️ ECCV2024** · **42** 篇论文解读

**[A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis](a_multimodal_benchmark_dataset_and_model_for_crop_disease_di.md)**

:   构建了包含13.7万张作物病害图像和100万问答对的CDDM数据集，并提出同时对视觉编码器、adapter和语言模型施加LoRA微调的策略，使Qwen-VL-Chat和LLaVA在作物病害诊断准确率上从个位数跃升至90%以上。

**[A New Dataset and Framework for Real-World Blurred Images Super-Resolution](a_new_dataset_and_framework_for_real-world_blurred_images_super-resolution.md)**

:   针对现有盲超分方法在处理含模糊（散焦/运动模糊）图像时过度纹理化、破坏模糊区域感知质量的问题，构建了包含近3000张模糊图像的ReBlurSR数据集，并提出PBaSR框架，通过双分支解耦训练（CDM）和基于权重插值的跨分支融合（CFM），在不增加任何推理开销的前提下，同时提升模糊图像和普通图像的超分效果，LPIPS提升0.02~0.10。

**[Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction](adaptive_bounding_box_uncertainties_via_twostep_conformal_pr.md)**

:   本文提出一种两步共形预测框架用于多目标检测的不确定性量化：第一步生成类别标签的共形预测集合以处理分类错误，第二步基于集成和分位数回归生成自适应的边界框不确定性区间，在保证覆盖率的同时提供实际可用的紧致预测区间。

**[Adaptive Multi-task Learning for Few-Shot Object Detection](adaptive_multi-task_learning_for_few-shot_object_detection.md)**

:   本文提出了一种自适应多任务学习方法(MTL-FSOD)，通过精度驱动的梯度平衡器动态调整分类和定位任务的梯度比例来缓解两者的冲突，并引入基于 CLIP 的知识蒸馏和分类精化方案来增强各任务的能力，在多个小样本检测基准上取得了一致的性能提升。

**[AFreeCA: Annotation-Free Counting for All](afreeca_annotation-free_counting_for_all.md)**

:   利用 Stable Diffusion 生成合成排序/计数数据，通过先学排序再学计数的两阶段策略 + 密度引导的图像分块，实现了首个适用于任意类别物体的无标注计数方法，在人群计数上超越已有无监督方法。

**[APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension](apl_anchor-based_prompt_learning_for_one-stage_weakly_supervised_referring_expre.md)**

:   本文提出锚框提示学习方法 APL，通过设计锚框提示编码器（APE）生成位置、颜色、类别三类判别性提示，动态融入锚框特征以丰富视觉语义，再配合文本重构损失和视觉对齐损失实现精确的视觉-语言对齐，在四个 REC 基准上超越现有弱监督方法（如 RefCOCO 上比 RefCLIP 高 6.44%）。

**[AugDETR: Improving Multi-scale Learning for Detection Transformer](augdetr_improving_multi-scale_learning_for_detection_transformer.md)**

:   本文提出 AugDETR（Augmented DETR），通过混合注意力编码器（Hybrid Attention Encoder）扩大可变形编码器的感受野并引入全局上下文特征增强特征表示，再通过编码器混合交叉注意力（Encoder-Mixing Cross-Attention）自适应利用多层编码器信息加速收敛，在 COCO 上为 DINO、AlignDETR、DDQ 分别带来 1.2/1.1/1.0 AP 的提升。

**[BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos](bam-detr_boundary-aligned_moment_detection_transformer_for_temporal_sentence_gro.md)**

:   提出边界对齐的时刻检测 Transformer（BAM-DETR），用 anchor-boundary 三元组 $(p, d_s, d_e)$ 替代传统的 center-length 二元组 $(c, l)$ 来建模时刻，配合双路径解码器和基于质量的排序机制，有效解决了中心模糊导致的定位不精确问题。

**[Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation](be_yourself_bounded_attention_for_multi-subject_text-to-image_generation.md)**

:   提出 Bounded Attention，一种无需训练的注意力约束方法，通过在去噪过程中限制 cross-attention 和 self-attention 的信息流动来解决多主体文本到图像生成中的语义泄漏问题。

**[Bridge Past and Future: Overcoming Information Asymmetry in Incremental Object Detection](bridge_past_and_future_overcoming_information_asymmetry_in_incremental_object_de.md)**

:   提出 Bridge Past and Future (BPF) 方法，通过伪标签桥接过去阶段、注意力机制排除未来潜在物体，并结合双教师蒸馏（Distillation with Future），解决增量目标检测中跨阶段信息不对称导致的优化目标不一致问题。

**[Can OOD Object Detectors Learn from Foundation Models?](can_ood_object_detectors_learn_from_foundation_models.md)**

:   SyncOOD 提出一种自动化数据策展方法，利用 LLM 想象语义新颖的 OOD 概念，通过 Stable Diffusion Inpainting 在 ID 图像上进行区域级编辑合成场景级 OOD 样本，再经 SAM 精炼框和特征相似度过滤后训练轻量 MLP 分类器，在多个 OOD 检测基准上以极少量合成数据大幅超越 SOTA。

**[DAMSDet: Dynamic Adaptive Multispectral Detection Transformer](damsdet_dynamic_adaptive_multispectral_detection_transformer_with_competitive_qu.md)**

:   DAMSDet 提出一种基于 DETR 架构的动态自适应红外-可见光目标检测方法，通过模态竞争 Query 选择（为每个目标动态选择主导模态特征作为初始 query）和多光谱可变形交叉注意力（在多语义层级上自适应采样和聚合双模态特征），同时解决互补信息融合和模态未对齐两大挑战，在 4 个公开数据集上显著超越 SOTA。

**[Efficient Inference of Vision Instruction-Following Models with Elastic Cache](efficient_inference_of_vision_instruction-following_models_with_elastic_cache.md)**

:   Elastic Cache 提出一种针对多模态指令遵循模型的 KV Cache 管理方法，在指令编码阶段采用基于重要性的 cache 合并策略（而非丢弃），在输出生成阶段采用固定点淘汰策略，以"一个序列、两种策略"实现任意加速比的高效推理，在 KV Cache 预算仅 0.2 时实现 78% 的实际速度提升且保持生成质量。

**[GRA: Detecting Oriented Objects Through Group-Wise Rotating and Attention](gra_detecting_oriented_objects_through_group-wise_rotating_and_attention.md)**

:   提出轻量级的 Group-wise Rotating and Attention (GRA) 模块，通过将卷积核分组旋转并施加分组空间注意力，在参数量减少近 50% 的同时超越了此前 SOTA 方法 ARC，在 DOTA-v2.0 上取得新的最优性能。

**[HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization](hat_history-augmented_anchor_transformer_for_online_temporal_action_localization.md)**

:   提出HAT——首个在Online Temporal Action Localization（OnTAL）中引入长期历史上下文的anchor-based Transformer框架，通过动作预期引导的历史压缩和未来驱动的历史精炼，在程序性自我中心数据集（EGTEA/EK100）上显著超越OAT，在标准数据集（THUMOS/MUSES）上达到可比或更优性能。

**[Implicit Concept Removal of Diffusion Models](implicit_concept_removal_of_diffusion_models.md)**

:   提出 Geom-Erasing 方法，通过引入外部分类器/检测器提供隐式概念的存在性和几何位置信息，将其编码为文本条件中的位置 token 并作为负提示使用，有效消除扩散模型中水印、不安全内容等"隐式概念"的生成，在 I2P 和自建 ICD 基准上达到 SOTA。

**[LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction](lami-detr_open-vocabulary_detection_with_language_model_instruction.md)**

:   提出 LaMI-DETR，通过利用 GPT 生成视觉概念描述和 T5 挖掘类间视觉相似性关系，解决开放词汇目标检测中概念表示不足和基类过拟合两大问题，在 OV-LVIS 上以 43.4 的 rare AP 超越前最佳方法 7.8 个点。

**[LayoutDETR: Detection Transformer Is a Good Multimodal Layout Designer](layoutdetr_detection_transformer_is_a_good_multimodal_layout_designer.md)**

:   将目标检测框架 DETR 与生成模型（GAN/VAE）统一，提出 LayoutDETR 用于多模态条件下的图形布局自动设计，以背景图像为约束、前景图文元素为驱动，在广告横幅和 UI 布局生成上达到 SOTA。

**[Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence](learn_from_the_learnt_source-free_active_domain_adaptation_via_contrastive_sampl.md)**

:   提出 LFTL（Learn from the Learnt）框架，通过对比主动采样（CAS）和视觉持久性引导适应（VPA）两个核心模块，在无源数据、极少量目标标注（≤5%）的条件下实现高效域适应，在 VisDA-C 上仅用 1% 标注即达到 87.4% 准确率。

**[MutDet: Mutually Optimizing Pre-training for Remote Sensing Object Detection](mutdet_mutually_optimizing_pre-training_for_remote_sensing_object_detection.md)**

:   提出 MutDet，一种面向遥感旋转目标检测的互优化预训练框架，通过双向交叉注意力融合 object embeddings 与 encoder 特征、对比对齐损失、以及辅助孪生头，系统性地缓解了检测预训练中 object embeddings 与 detector features 之间的特征差异问题。

**[Nonverbal Interaction Detection](nonverbal_interaction_detection.md)**

:   首次系统性研究人类非语言交互（手势、表情、注视、姿态、触碰），提出大规模数据集 NVI、新任务 NVI-DET 和基于双重多尺度超图的检测模型 NVI-DEHR，在非语言交互检测和 HOI 检测任务上均取得最优性能。

**[On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines](on_calibration_of_object_detectors_pitfalls_evaluation_and_baselines.md)**

:   本文系统性地揭示了当前目标检测器校准研究中评估框架、评估指标和温度缩放（Temperature Scaling）使用方面的重大缺陷，提出了原则性的联合评估框架以及专为目标检测定制的后处理校准方法（Platt Scaling和Isotonic Regression），证明了正确设计和评估的后处理校准器远优于近期训练时校准方法。

**[Online Temporal Action Localization with Memory-Augmented Transformer](online_temporal_action_localization_with_memory-augmented_transformer.md)**

:   提出 MATR（Memory-Augmented Transformer），通过记忆队列存储过去片段的特征来利用长期上下文，并采用分离的 Start/End Transformer 解码器进行动作实例定位，在在线时序动作定位（On-TAL）任务上取得 SOTA，甚至可比肩部分离线方法。

**[OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection](openkd_opening_prompt_diversity_for_zero-_and_few-shot_keypoint_detection.md)**

:   提出 OpenKD 模型，从模态（视觉+文本）、语义（seen vs. unseen）、语言（多样化文本）三个维度开放 prompt 多样性，通过多模态 prototype set、辅助关键点-文本插值和 LLM 文本解析，实现通用的 zero- and few-shot keypoint detection，在 Animal Pose、AwA、CUB、NABird 上取得 SOTA。

**[Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer](portrait4d-v2_pseudo_multi-view_data_creates_better_4d_head_synthesizer.md)**

:   提出一种利用**伪多视角视频**来训练前馈式单图4D头部合成器的新学习范式：先用合成数据学一个3D头部合成器将单目视频转为多视角，再利用伪多视角视频通过**跨视角自重演**学习4D合成器，避免了对3DMM的过度依赖，在重建保真度、几何一致性和运动控制精度上大幅超越先前方法。

**[Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation](projecting_points_to_axes_oriented_object_detection_via_point-axis_representatio.md)**

:   提出点-轴（Point-Axis）表示方法，将旋转目标的位置（点集）和方向（轴编码）解耦，配合 Max-Projection Loss 和 Cross-Axis Loss 实现无需额外标注的优化，并基于此设计 Oriented DETR 模型，解决传统旋转框表示的损失不连续问题。

**[Rectify the Regression Bias in Long-Tailed Object Detection](rectify_the_regression_bias_in_long-tailed_object_detection.md)**

:   首次揭示并系统解决长尾目标检测中被忽视的**回归偏差**问题：稀有类别的类别专属(class-specific)回归头参数因样本不足导致泛化能力差，通过添加额外的类别不可知(class-agnostic)回归分支进行权衡，在LVIS等数据集上取得了SOTA性能。

**[ReGround: Improving Textual and Spatial Grounding at No Cost](reground_improving_textual_and_spatial_grounding_at_no_cost.md)**

:   通过将 GLIGEN 中 Gated Self-Attention (GSA) 与 Cross-Attention (CA) 的串行连接改为并行连接（网络重连），在不引入任何新参数、不需要微调、不增加计算开销的前提下，显著缓解了文本定位与空间定位之间的权衡问题。

**[Responsible Visual Editing](responsible_visual_editing.md)**

:   定义"负责任视觉编辑"新任务，提出CoEditor认知编辑器，通过感知-行为双阶段认知过程将有害图像转换为负责任的版本，同时最小化修改。

**[SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding](shine_saliency-aware_hierarchical_negative_ranking_for_compositional_temporal_gr.md)**

:   针对组合时序定位任务中现有方法负样本构造不合理、DETR 模型对负查询无法产生合理显著性响应的问题，提出利用 LLM（GPT-3.5 Turbo）生成语义可行的分层硬负样本，并设计粗到细的显著性排序策略建立视频片段与层次负查询之间的多粒度语义关系，显著提升组合泛化能力。

**[Spherical Linear Interpolation and Text-Anchoring for Zero-shot Composed Image Retrieval](spherical_linear_interpolation_and_text-anchoring_for_zero-shot_composed_image_r.md)**

:   提出 Slerp-based ZS-CIR 方法，通过球面线性插值（Slerp）直接融合 VLP 模型的图像和文本嵌入构造组合查询表示，配合 Text-Anchored-Tuning (TAT) 用 LoRA 微调图像编码器缩小模态间隙，在 CIRR/CIRCO/FashionIQ 上达到 SOTA。

**[Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization](stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo.md)**

:   针对点监督时序动作定位中稀疏标注导致的动作边界语义模糊问题，提出逐步多粒度边界检测器（SMBD），通过背景锚点生成器（BAG）和双边界检测器（DBD）为训练提供细粒度的边界监督信号，在THUMOS'14等数据集上达到SOTA。

**[TAPTR: Tracking Any Point with Transformers as Detection](taptr_tracking_any_point_with_transformers_as_detection.md)**

:   TAPTR 将 Tracking Any Point (TAP) 任务重新建模为类 DETR 的检测问题，将每个跟踪点表示为包含位置和内容的 point query，通过多层 Transformer 解码器逐层优化，结合 cost volume 和滑动窗口特征更新策略，在 TAP-Vid 基准上达到 SOTA 且推理速度更快。

**[Tensorial Template Matching for Fast Cross-Correlation with Rotations and Its Application for Tomography](tensorial_template_matching_for_fast_cross-correlation_with_rotations_and_its_ap.md)**

:   提出张量模板匹配（TTM）算法，通过对称张量场将模板在所有旋转下的信息整合为固定数量的相关计算，使得计算复杂度与旋转精度无关，在3D断层扫描图像中实现快速且准确的目标检测与旋转估计。

**[Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching](towards_natural_language-guided_drones_geotext-1652_benchmark_with_spatial_relat.md)**

:   构建了首个自然语言引导的无人机地理定位基准 GeoText-1652（276K bbox-text 对，316K 描述），并提出 blending spatial matching 方法通过 grounding loss + spatial relation loss 实现区域级空间关系匹配，文本检索 Recall@10 达到 31.2%。

**[Tracking Meets LoRA: Faster Training, Larger Model, Stronger Performance](tracking_meets_lora_faster_training_larger_model_stronger_performance.md)**

:   LoRAT 首次将 LoRA 引入视觉目标跟踪，通过解耦位置编码（共享空间 + 独立类型嵌入）和纯 MLP 检测头两个 LoRA-友好设计，使得在实验室级资源上训练 ViT-g 骨干的跟踪器成为可能，在 LaSOT 上达到 0.762 SUC（新 SOTA），最轻变体 LoRAT-B-224 以 209 FPS 运行。

**[Visible and Clear: Finding Tiny Objects in Difference Map](visible_and_clear_finding_tiny_objects_in_difference_map.md)**

:   SR-TOD 首次将图像自重建机制引入目标检测，发现重建差异图与微小目标之间的强相关性，并设计差异图引导的特征增强（DGFE）模块，在自建反无人机数据集 DroneSwarms 和 VisDrone2019、AI-TOD 上均取得显著提升。

**[WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs](walker_self-supervised_multiple_object_tracking_by_walking_on_temporal_appearanc.md)**

:   本文提出Walker——首个自监督多目标跟踪器，通过构建准稠密的时序物体外观图（temporal appearance graph），设计多正样本对比损失优化图上的随机游走来学习实例相似度，并引入互斥连接约束和运动约束双向游走推理策略，在MOT17、DanceTrack和BDD100K上达到自监督跟踪的竞争性能，且在标注需求减少400倍的情况下仍超越之前的自监督方法。

**[Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection](weak-to-strong_compositional_learning_from_generative_models_for_language-based_.md)**

:   提出 WSCL 框架：利用 LLM 生成多样文本描述 + 扩散模型生成对应图像 + 弱检测器分解短语生成伪标框，构建密集合成三元组（image, description, bbox），配合组合对比学习显著提升语言引导目标检测性能，OmniLabel 上 GLIP-T 提升 +5.0AP。

**[WeCromCL: Weakly Supervised Cross-Modality Contrastive Learning for Transcription-only Supervised Text Spotting](wecromcl_weakly_supervised_cross-modality_contrastive_learning_for_transcription.md)**

:   提出 WeCromCL 框架，通过弱监督的原子级跨模态对比学习，仅利用文本转录标注（无位置标注）实现场景文字定位，将检测到的锚点作为伪标签训练单点监督文字检测器，在无边界标注的条件下达到接近全监督的性能。

**[YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](yolov9_learning_what_you_want_to_learn_using_programmable_gradient_information.md)**

:   YOLOv9 提出可编程梯度信息 (PGI) 和广义高效层聚合网络 (GELAN) 来解决深度网络中的信息瓶颈问题，在 MS COCO 上以更少参数和计算量全面超越现有实时目标检测器，从零训练即可超过使用大数据集预训练的方法。

**[Zero-Shot Detection of AI-Generated Images](zero-shot_detection_of_ai-generated_images.md)**

:   本文提出了零样本熵检测器ZED（Zero-shot Entropy-based Detector），通过无损图像编码器估计每个像素在给定上下文下的概率分布，用"图像对真实图像模型的意外程度"作为判别特征，无需任何AI生成训练数据即可检测多种生成器生成的图像，在广泛的生成模型上比SOTA平均准确率提升超过3%。
