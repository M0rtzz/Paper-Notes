---
title: >-
  ECCV2024 目标检测方向37篇论文解读
description: >-
  37篇ECCV2024的目标检测方向论文解读，涵盖目标检测、少样本学习、3D 目标检测、自监督学习、布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**🎞️ ECCV2024** · **37** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (5)](../../ACL2026/object_detection/index.md) · [📷 CVPR2026 (39)](../../CVPR2026/object_detection/index.md) · [🔬 ICLR2026 (9)](../../ICLR2026/object_detection/index.md) · [🤖 AAAI2026 (17)](../../AAAI2026/object_detection/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/object_detection/index.md) · [📹 ICCV2025 (28)](../../ICCV2025/object_detection/index.md)

🔥 **高频主题：** 目标检测 ×7 · 少样本学习 ×4 · 3D 目标检测 ×3 · 自监督学习 ×2 · 布局/合成 ×2

**[A Multimodal Benchmark Dataset and Model for Crop Disease Diagnosis](a_multimodal_benchmark_dataset_and_model_for_crop_disease_di.md)**

:   构建了包含13.7万张作物病害图像和100万问答对的CDDM数据集，并提出同时对视觉编码器、adapter和语言模型施加LoRA微调的策略，使Qwen-VL-Chat和LLaVA在作物病害诊断准确率上从个位数跃升至90%以上。

**[Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction](adaptive_bounding_box_uncertainties_via_twostep_conformal_pr.md)**

:   本文提出一种两步共形预测框架用于多目标检测的不确定性量化：第一步生成类别标签的共形预测集合以处理分类错误，第二步基于集成和分位数回归生成自适应的边界框不确定性区间，在保证覆盖率的同时提供实际可用的紧致预测区间。

**[Adaptive Multi-task Learning for Few-Shot Object Detection](adaptive_multi-task_learning_for_few-shot_object_detection.md)**

:   本文提出了一种自适应多任务学习方法(MTL-FSOD)，通过精度驱动的梯度平衡器动态调整分类和定位任务的梯度比例来缓解两者的冲突，并引入基于 CLIP 的知识蒸馏和分类精化方案来增强各任务的能力，在多个小样本检测基准上取得了一致的性能提升。

**[Adaptive Multi-head Contrastive Learning](adaptive_multihead_contrastive_learning.md)**

:   本文提出AMCL（Adaptive Multi-head Contrastive Learning），通过多个投影头产生不同特征视角，配合基于MLE推导的自适应温度机制为每对样本独立加权，有效解决了多种数据增强下正负样本相似度分布重叠的问题，一致提升SimCLR、MoCo和Barlow Twins的性能。

**[AFreeCA: Annotation-Free Counting for All](afreeca_annotation-free_counting_for_all.md)**

:   利用 Stable Diffusion 生成合成排序/计数数据，通过先学排序再学计数的两阶段策略 + 密度引导的图像分块，实现了首个适用于任意类别物体的无标注计数方法，在人群计数上超越已有无监督方法。

**[Approaching Outside: Scaling Unsupervised 3D Object Detection from 2D Scene](approaching_outside_scaling_unsupervised_3d_object_detection_from_2d_scene.md)**

:   提出 LiSe 方法，将 2D 图像信息引入无监督 3D 目标检测，通过自步学习（self-paced learning）中的自适应采样和弱模型聚合策略，大幅提升远距离和小目标的检测能力。

**[AugDETR: Improving Multi-scale Learning for Detection Transformer](augdetr_improving_multi-scale_learning_for_detection_transformer.md)**

:   本文提出 AugDETR（Augmented DETR），通过混合注意力编码器（Hybrid Attention Encoder）扩大可变形编码器的感受野并引入全局上下文特征增强特征表示，再通过编码器混合交叉注意力（Encoder-Mixing Cross-Attention）自适应利用多层编码器信息加速收敛，在 COCO 上为 DINO、AlignDETR、DDQ 分别带来 1.2/1.1/1.0 AP 的提升。

**[BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos](bam-detr_boundary-aligned_moment_detection_transformer_for_temporal_sentence_gro.md)**

:   提出边界对齐的时刻检测 Transformer（BAM-DETR），用 anchor-boundary 三元组 $(p, d_s, d_e)$ 替代传统的 center-length 二元组 $(c, l)$ 来建模时刻，配合双路径解码器和基于质量的排序机制，有效解决了中心模糊导致的定位不精确问题。

**[Bridge Past and Future: Overcoming Information Asymmetry in Incremental Object Detection](bridge_past_and_future_overcoming_information_asymmetry_in_incremental_object_de.md)**

:   提出 Bridge Past and Future (BPF) 方法，通过伪标签桥接过去阶段、注意力机制排除未来潜在物体，并结合双教师蒸馏（Distillation with Future），解决增量目标检测中跨阶段信息不对称导致的优化目标不一致问题。

**[Can OOD Object Detectors Learn from Foundation Models?](can_ood_object_detectors_learn_from_foundation_models.md)**

:   SyncOOD 提出一种自动化数据策展方法，利用 LLM 想象语义新颖的 OOD 概念，通过 Stable Diffusion Inpainting 在 ID 图像上进行区域级编辑合成场景级 OOD 样本，再经 SAM 精炼框和特征相似度过滤后训练轻量 MLP 分类器，在多个 OOD 检测基准上以极少量合成数据大幅超越 SOTA。

**[DAMSDet: Dynamic Adaptive Multispectral Detection Transformer](damsdet_dynamic_adaptive_multispectral_detection_transformer_with_competitive_qu.md)**

:   DAMSDet 提出一种基于 DETR 架构的动态自适应红外-可见光目标检测方法，通过模态竞争 Query 选择（为每个目标动态选择主导模态特征作为初始 query）和多光谱可变形交叉注意力（在多语义层级上自适应采样和聚合双模态特征），同时解决互补信息融合和模态未对齐两大挑战，在 4 个公开数据集上显著超越 SOTA。

**[DSPDet3D: 3D Small Object Detection with Dynamic Spatial Pruning](dspdet3d_3d_small_object_detection_with_dynamic_spatial_pruning.md)**

:   提出动态空间剪枝（DSP）策略，在多级 3D 检测器的解码器中逐级移除已检测到大物体区域的体素特征，使检测器能以高空间分辨率处理场景、大幅提升小目标检测精度（ScanNet 小目标 mAP@0.25 从 27.5% 提升到 44.8%），同时通过剪枝将显存降低为同分辨率方法的 1/5。

**[GRA: Detecting Oriented Objects Through Group-Wise Rotating and Attention](gra_detecting_oriented_objects_through_group-wise_rotating_and_attention.md)**

:   提出轻量级的 Group-wise Rotating and Attention (GRA) 模块，通过将卷积核分组旋转并施加分组空间注意力，在参数量减少近 50% 的同时超越了此前 SOTA 方法 ARC，在 DOTA-v2.0 上取得新的最优性能。

**[Interactive 3D Object Detection with Prompts](interactive_3d_object_detection_with_prompts.md)**

:   提出"2D提示，3D检测"+"3D检测，3D精化"的多模态交互式 3D 目标检测框架，通过简单的 2D 交互提示（点击或框选）桥接 2D-3D 复杂性差距，并支持迭代精化，大幅降低 3D 标注成本，在 nuScenes 上验证了有效性且展示了出色的开放集能力。

**[LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction](lami-detr_open-vocabulary_detection_with_language_model_instruction.md)**

:   提出 LaMI-DETR，通过利用 GPT 生成视觉概念描述和 T5 挖掘类间视觉相似性关系，解决开放词汇目标检测中概念表示不足和基类过拟合两大问题，在 OV-LVIS 上以 43.4 的 rare AP 超越前最佳方法 7.8 个点。

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

**[Plain-Det: A Plain Multi-Dataset Object Detector](plain-det_a_plain_multi-dataset_object_detector.md)**

:   Plain-Det 提出了一个简洁灵活的多数据集目标检测框架，通过语义空间校准、类感知查询组合器和基于难度的动态采样策略，在 COCO 上达到 51.9 mAP（匹配当时 SOTA），并可灵活扩展到新数据集且保持鲁棒性能。

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

**[Tensorial Template Matching for Fast Cross-Correlation with Rotations and Its Application for Tomography](tensorial_template_matching_for_fast_cross-correlation_with_rotations_and_its_ap.md)**

:   提出张量模板匹配（TTM）算法，通过对称张量场将模板在所有旋转下的信息整合为固定数量的相关计算，使得计算复杂度与旋转精度无关，在3D断层扫描图像中实现快速且准确的目标检测与旋转估计。

**[Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching](towards_natural_language-guided_drones_geotext-1652_benchmark_with_spatial_relat.md)**

:   构建了首个自然语言引导的无人机地理定位基准 GeoText-1652（276K bbox-text 对，316K 描述），并提出 blending spatial matching 方法通过 grounding loss + spatial relation loss 实现区域级空间关系匹配，文本检索 Recall@10 达到 31.2%。

**[Visible and Clear: Finding Tiny Objects in Difference Map](visible_and_clear_finding_tiny_objects_in_difference_map.md)**

:   SR-TOD 首次将图像自重建机制引入目标检测，发现重建差异图与微小目标之间的强相关性，并设计差异图引导的特征增强（DGFE）模块，在自建反无人机数据集 DroneSwarms 和 VisDrone2019、AI-TOD 上均取得显著提升。

**[WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs](walker_self-supervised_multiple_object_tracking_by_walking_on_temporal_appearanc.md)**

:   本文提出Walker——首个自监督多目标跟踪器，通过构建准稠密的时序物体外观图（temporal appearance graph），设计多正样本对比损失优化图上的随机游走来学习实例相似度，并引入互斥连接约束和运动约束双向游走推理策略，在MOT17、DanceTrack和BDD100K上达到自监督跟踪的竞争性能，且在标注需求减少400倍的情况下仍超越之前的自监督方法。

**[Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection](weak-to-strong_compositional_learning_from_generative_models_for_language-based_.md)**

:   提出 WSCL 框架：利用 LLM 生成多样文本描述 + 扩散模型生成对应图像 + 弱检测器分解短语生成伪标框，构建密集合成三元组（image, description, bbox），配合组合对比学习显著提升语言引导目标检测性能，OmniLabel 上 GLIP-T 提升 +5.0AP。

**[Weakly Supervised 3D Object Detection via Multi-Level Visual Guidance](weakly_supervised_3d_object_detection_via_multi-level_visual_guidance.md)**

:   提出 VG-W3D 框架，仅使用 2D 标注（无需任何 3D 标签），通过特征级、输出级和训练级三层视觉引导来训练 3D 目标检测器，在 KITTI 上取得了与使用 500 帧 3D 标注方法相当的性能。

**[YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](yolov9_learning_what_you_want_to_learn_using_programmable_gradient_information.md)**

:   YOLOv9 提出可编程梯度信息 (PGI) 和广义高效层聚合网络 (GELAN) 来解决深度网络中的信息瓶颈问题，在 MS COCO 上以更少参数和计算量全面超越现有实时目标检测器，从零训练即可超过使用大数据集预训练的方法。

**[Zero-Shot Detection of AI-Generated Images](zero-shot_detection_of_ai-generated_images.md)**

:   本文提出了零样本熵检测器ZED（Zero-shot Entropy-based Detector），通过无损图像编码器估计每个像素在给定上下文下的概率分布，用"图像对真实图像模型的意外程度"作为判别特征，无需任何AI生成训练数据即可检测多种生成器生成的图像，在广泛的生成模型上比SOTA平均准确率提升超过3%。
