---
title: >-
  CVPR2025 目标检测方向 27篇论文解读
description: >-
  27篇CVPR2025 目标检测方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**📷 CVPR2025** · **27** 篇论文解读

**[A Bias-Free Training Paradigm for More General AI-generated Image Detection](a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)**

:   提出B-Free训练范式——通过stable diffusion的自条件重构从真实图像生成语义对齐的假图，结合inpainting内容增强，消除格式/内容/分辨率等偏差，使检测器聚焦于生成器特有的伪影痕迹，在27种生成模型（含FLUX、SD 3.5等最新模型）上泛化AUC>99%，balanced accuracy达95.2%。

**[ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](abra_teleporting_fine-tuned_knowledge_across_domains_for_open-vocabulary_object_.md)**

:   提出 ABRA（Aligned Basis Relocation for Adaptation），通过在权重空间中进行 SVD 分解与正交旋转对齐，将源域的类别特定检测知识"传送"到无标注数据的目标域，实现零样本跨域目标检测。

**[Any6D: Model-free 6D Pose Estimation of Novel Objects](any6d_model-free_6d_pose_estimation_of_novel_objects.md)**

:   提出 Any6D 框架，仅从单张 RGB-D 锚点图像即可估计未知物体的 6D 位姿和尺寸，通过 InstantMesh 3D 重建 + 朝向包围盒粗对齐 + 联合尺寸-位姿精细化，在 HO3D 上 ADD-S 达 98.7% 远超 GEDI 的 71.9%。

**[BACON: Improving Clarity of Image Captions via Bag-of-Concept Graphs](bacon_improving_clarity_of_image_captions_via_bag-of-concept_graphs.md)**

:   提出BACON提示方法，将VLM生成的冗长图像描述解构为物体、关系、风格、主题等解耦结构化元素（JSON字典格式），使下游模型无需强文本编码能力即可高效利用描述信息，在开放词汇目标检测中帮助GroundingDINO实现1.51倍的召回率提升。

**[Boosting Domain Incremental Learning: Selecting the Optimal Parameters Is All You Need](boosting_domain_incremental_learning_selecting_the_optimal_parameters_is_all_you.md)**

:   发现在域增量学习中选择最优参数子集比微调全部参数更有效，提出参数选择策略解决域增量目标检测的灾难性遗忘

**[Co-Spy: Combining Semantic and Pixel Features to Detect Synthetic Images by AI](co-spy_combining_semantic_and_pixel_features_to_detect_synthetic_images_by_ai.md)**

:   提出 Co-Spy 融合 VAE 重建伪影特征和 CLIP 语义特征两条互补检测路径——VAE 伪影跨模型泛化但怕 JPEG 压缩，CLIP 语义抗 JPEG 但泛化差——自适应调节器根据输入动态分配两路权重，在 22 个生成模型上建立新 SOTA。

**[DEIM: DETR with Improved Matching for Fast Convergence](deim_detr_with_improved_matching_for_fast_convergence.md)**

:   通过两个简单改进加速 DETR 训练收敛——Dense O2O（用数据增强增加每图目标数实现稠密一对一匹配）和 MAL（替代 VFL 更好地优化低质量匹配），训练 epoch 减半同时性能提升（COCO AP 56.5 with D-FINE-X）。

**[Detecting Adversarial Data Using Perturbation Forgery](detecting_adversarial_data_using_perturbation_forgery.md)**

:   通过建模对抗噪声的高斯分布并证明其近邻性，提出 Perturbation Forgery 方法在训练时持续扰动噪声分布形成开覆盖，配合稀疏掩码生成伪对抗数据训练二分类器，仅需 FGSM 一种攻击的噪声分布就能泛化检测梯度、GAN、扩散和物理等各类未见攻击，AUROC 达 0.99+ 且推理开销极低。

**[Detecting Out-of-Distribution through the Lens of Neural Collapse](detecting_out-of-distribution_through_the_lens_of_neural_collapse.md)**

:   从 Neural Collapse 理论出发，发现中心化后的 ID 特征聚集在预测类别的权重向量附近且远离原点（形成 simplex ETF），据此设计 NCI 检测器——结合特征与权重向量的角度近邻度（pScore）和特征范数过滤，在 CIFAR-10/100 和 ImageNet 多架构上实现最佳综合 OOD 检测性能且推理延迟与 softmax 基线持平。

**[DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation](diffvsgg_diffusion-driven_online_video_scene_graph_generation.md)**

:   提出 DiffVsgg 将视频场景图生成（VSGG）建模为沿时间轴的迭代去噪问题——用共享特征嵌入统一目标分类、框回归和关系预测三个任务，通过潜在扩散模型做空间推理+用前帧预测作条件做时序推理，首次实现在线VSGG且在 Action Genome 三个评估协议上全面 SOTA，R@10 超越 DSG-DETR 3.3 个点。

**[DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](dreamvideo-omni_omni-motion_controlled_multi-subject_video_customization_with_la.md)**

:   提出 DreamVideo-Omni，通过渐进式两阶段训练范式（Omni-Motion SFT + Latent Identity Reward Feedback Learning），在统一的 DiT 框架中实现多主体定制与全运动控制（全局 bbox + 局部轨迹 + 相机运动）的协同生成。

**[Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention](efficient_event-based_object_detection_a_hybrid_neural_network_with_spatial_and_.md)**

:   提出首个面向大规模基准的混合 SNN-ANN 目标检测模型，设计注意力桥接模块（ASAB）将 SNN 的稀疏脉冲表示通过时空注意力转换为 ANN 可处理的密集特征，在 Gen1/Gen4 数据集上以仅 6.6M 参数大幅超越 SNN 方法并接近 ANN/RNN 方法的精度，同时 SNN 部分可部署在 Intel Loihi 2 神经形态芯片上实现低功耗推理。

**[Efficient Test-Time Adaptive Object Detection via Sensitivity-Guided Pruning](efficient_test-time_adaptive_object_detection_via_sensitivity-guided_pruning.md)**

:   提出一种高效的持续测试时自适应目标检测（CTTA-OD）方法，发现源模型中某些特征通道对域偏移敏感且会损害跨域性能，通过在图像级和实例级度量通道敏感性来引导加权稀疏正则化实现选择性剪枝，辅以随机通道重激活机制防止误剪，在减少 12% 计算量的同时超越 SOTA 方法的自适应精度。

**[Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models](enhancing_privacy-utility_trade-offs_to_mitigate_memorization_in_diffusion_model.md)**

:   本文提出 PRSS 方法，通过 Prompt Re-anchoring（将记忆化 prompt 重新用作 CFG 的锚点引导生成偏离记忆内容）和 Semantic Prompt Search（用 LLM 搜索语义相似但不触发记忆的替代 prompt）两个策略，在不修改模型和不需要训练数据的推理阶段改进 CFG 方程，实现了扩散模型记忆化缓解中的最优隐私-效用平衡。

**[ESCAPE: Equivariant Shape Completion via Anchor Point Encoding](escape_equivariant_shape_completion_via_anchor_point_encoding.md)**

:   ESCAPE 提出了一种基于锚点距离编码的旋转等变点云补全方法，通过将点云表示为到高曲率锚点的距离矩阵，使 Transformer 在旋转不变的距离空间中预测完整形状，再通过优化恢复 3D 坐标，在任意旋转输入下大幅超越现有方法（PCN 数据集 CD-L1 从 26.65 降至 10.58）。

**[Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection](generalized_diffusion_detector_mining_robust_features_from_diffusion_models_for_.md)**

:   本文首次将扩散模型引入域泛化目标检测，通过提取扩散过程的多时间步中间特征构建域不变的检测器，并设计特征级+目标级对齐的知识迁移框架将泛化能力蒸馏到轻量检测器中，在6个DG基准上平均提升14.0% mAP，甚至超越大多数域适应方法。

**[HumanMM: Global Human Motion Recovery from Multi-shot Videos](humanmm_global_human_motion_recovery_from_multi-shot_videos.md)**

:   HumanMM首次提出从多镜头视频中恢复世界坐标系下3D人体运动的框架，通过镜头转换检测器、增强SLAM、基于立体标定的朝向对齐和运动积分器，实现了跨镜头的连续运动重建。

**[Image Reconstruction from Readout-Multiplexed Single-Photon Detector Arrays](image_reconstruction_from_readout-multiplexed_single-photon_detector_arrays.md)**

:   本文将行列读出复用的单光子探测器阵列中的多光子碰巧分辨问题形式化为逆成像问题，提出了一种概率性的多光子估计器（Multiphoton Estimator），能够解析最多4个同时入射的光子的空间位置，在32×32阵列上相比传统方法提升3-4 dB PSNR，并将所需帧数减少约4倍。

**[Interpreting Object-level Foundation Models via Visual Precision Search](interpreting_object-level_foundation_models_via_visual_precision_search.md)**

:   针对 Grounding DINO 和 Florence-2 等目标级基础模型的可解释性问题，本文提出 Visual Precision Search (VPS) 方法，通过超像素稀疏化+子模函数引导的贪心搜索精确定位关键决策子区域，在 MS COCO/RefCOCO/LVIS 上的忠实度指标(Insertion)分别超过 SOTA 方法 D-RISE 达 23.7%/20.1%/31.6%。

**[Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection](large_self-supervised_models_bridge_the_gap_in_domain_adaptive_object_detection.md)**

:   DINO Teacher 提出用冻结的 DINOv2 大模型替代传统 Mean Teacher 框架中的 EMA 教师，一方面作为更准确的伪标签生成器，另一方面作为特征对齐的代理目标，在多个域自适应目标检测基准上取得了 SOTA 性能（BDD100k 上 +7.6%）。

**[MCCD: Multi-Agent Collaboration-based Compositional Diffusion for Complex Text-to-Image Generation](mccd_multi-agent_collaboration-based_compositional_diffusion_for_complex_text-to.md)**

:   提出MCCD，通过MLLM驱动的多智能体协作场景解析模块和层次化组合扩散机制（高斯掩码+区域增强滤波），以免训练方式显著提升扩散模型在多物体、多属性、多关系复杂场景下的文生图生成质量。

**[MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)**

:   MI-DETR 提出了并行多次查询（MI）机制替代传统 DETR 级联解码器架构，让 object queries 通过多个参数独立的 inquiry heads 并行地从图像特征中学习多模式信息，配合 U-like Feature Interaction（UFI），在 COCO 上以 ResNet-50 backbone 达到 52.7 AP，超越所有已有 DETR 变体。

**[Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](mitigating_memorization_in_text-to-image_diffusion_via_region-aware_prompt_augme.md)**

:   提出 RAPTA（训练时基于目标检测的区域感知 prompt 变体增强）和 ADMCD（推理时三流注意力融合的多模态复制检测），从缓解和检测两个角度端到端地应对文生图扩散模型的训练数据记忆化问题。

**[MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](mokus_leveraging_cross-modal_knowledge_transfer_for_knowledge-aware_concept_cust.md)**

:   提出 MoKus 框架，发现并利用"跨模态知识迁移"现象——在 LLM 文本编码器中更新知识会自动传递到视觉生成端——实现知识感知的概念定制，两阶段设计：先学视觉锚点表示，再秒级更新文本知识绑定。

**[Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE](mr_detr_instructive_multi-route_training_for_detection_transformers.md)**

:   系统研究 DETR 解码器各组件在 one-to-one/one-to-many 多任务框架下的角色，发现任何单独组件都能有效协调两个目标；基于此提出多路由训练（Instructive Self-Attention + Independent FFN + Route-Aware MoE），推理时丢弃辅助路由不增加任何开销。

**[Multiple Object Tracking as ID Prediction](multiple_object_tracking_as_id_prediction.md)**

:   本文提出MOTIP，将多目标跟踪中的目标关联问题重新定义为in-context ID预测任务：给定携带ID嵌入的历史轨迹，直接用标准Transformer解码器预测当前检测的ID标签，无需启发式匹配算法即在DanceTrack上以69.6 HOTA大幅超越前SOTA CO-MOT (65.3)。

**[Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)**

:   提出 ESM-YOLO+，一种轻量级可见光-红外融合网络，通过 MEAF 模块（可学习空间掩码+空间注意力的像素级融合）和训练时结构表示增强（SR，推理时无开销的超分辅助监督），在 VEDAI 上达到 84.71% mAP 同时参数量仅 5.1M（减少 93.6%）。
