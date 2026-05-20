---
title: >-
  CVPR2025 目标检测方向27篇论文解读
description: >-
  27篇CVPR2025的目标检测方向论文解读，涵盖目标检测、扩散模型、文生图等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "目标检测"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "文生图"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**📷 CVPR2025** · **27** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (1)](../../ICML2026/object_detection/index.md) · [📷 CVPR2026 (36)](../../CVPR2026/object_detection/index.md) · [🔬 ICLR2026 (9)](../../ICLR2026/object_detection/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/object_detection/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/object_detection/index.md) · [📹 ICCV2025 (26)](../../ICCV2025/object_detection/index.md)

🔥 **高频主题：** 目标检测 ×10 · 扩散模型 ×4 · 文生图 ×2

**[ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](abra_teleporting_fine-tuned_knowledge_across_domains_for_open-vocabulary_object_.md)**

:   提出 ABRA（Aligned Basis Relocation for Adaptation），通过在权重空间中进行 SVD 分解与正交旋转对齐，将源域的类别特定检测知识"传送"到无标注数据的目标域，实现零样本跨域目标检测。

**[BACON: Improving Clarity of Image Captions via Bag-of-Concept Graphs](bacon_improving_clarity_of_image_captions_via_bag-of-concept_graphs.md)**

:   提出BACON提示方法，将VLM生成的冗长图像描述解构为物体、关系、风格、主题等解耦结构化元素（JSON字典格式），使下游模型无需强文本编码能力即可高效利用描述信息，在开放词汇目标检测中帮助GroundingDINO实现1.51倍的召回率提升。

**[Boosting Domain Incremental Learning: Selecting the Optimal Parameters Is All You Need](boosting_domain_incremental_learning_selecting_the_optimal_parameters_is_all_you.md)**

:   发现在域增量学习中选择最优参数子集比微调全部参数更有效，提出参数选择策略解决域增量目标检测的灾难性遗忘

**[DEIM: DETR with Improved Matching for Fast Convergence](deim_detr_with_improved_matching_for_fast_convergence.md)**

:   通过两个简单改进加速 DETR 训练收敛——Dense O2O（用数据增强增加每图目标数实现稠密一对一匹配）和 MAL（替代 VFL 更好地优化低质量匹配），训练 epoch 减半同时性能提升（COCO AP 56.5 with D-FINE-X）。

**[DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation](diffvsgg_diffusion-driven_online_video_scene_graph_generation.md)**

:   提出 DiffVsgg 将视频场景图生成（VSGG）建模为沿时间轴的迭代去噪问题——用共享特征嵌入统一目标分类、框回归和关系预测三个任务，通过潜在扩散模型做空间推理+用前帧预测作条件做时序推理，首次实现在线VSGG且在 Action Genome 三个评估协议上全面 SOTA，R@10 超越 DSG-DETR 3.3 个点。

**[Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention](efficient_event-based_object_detection_a_hybrid_neural_network_with_spatial_and_.md)**

:   提出首个面向大规模基准的混合 SNN-ANN 目标检测模型，设计注意力桥接模块（ASAB）将 SNN 的稀疏脉冲表示通过时空注意力转换为 ANN 可处理的密集特征，在 Gen1/Gen4 数据集上以仅 6.6M 参数大幅超越 SNN 方法并接近 ANN/RNN 方法的精度，同时 SNN 部分可部署在 Intel Loihi 2 神经形态芯片上实现低功耗推理。

**[Efficient Test-Time Adaptive Object Detection via Sensitivity-Guided Pruning](efficient_test-time_adaptive_object_detection_via_sensitivity-guided_pruning.md)**

:   提出一种高效的持续测试时自适应目标检测（CTTA-OD）方法，发现源模型中某些特征通道对域偏移敏感且会损害跨域性能，通过在图像级和实例级度量通道敏感性来引导加权稀疏正则化实现选择性剪枝，辅以随机通道重激活机制防止误剪，在减少 12% 计算量的同时超越 SOTA 方法的自适应精度。

**[Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection](generalized_diffusion_detector_mining_robust_features_from_diffusion_models_for_.md)**

:   本文首次将扩散模型引入域泛化目标检测，通过提取扩散过程的多时间步中间特征构建域不变的检测器，并设计特征级+目标级对齐的知识迁移框架将泛化能力蒸馏到轻量检测器中，在6个DG基准上平均提升14.0% mAP，甚至超越大多数域适应方法。

**[Interpreting Object-level Foundation Models via Visual Precision Search](interpreting_object-level_foundation_models_via_visual_precision_search.md)**

:   针对 Grounding DINO 和 Florence-2 等目标级基础模型的可解释性问题，本文提出 Visual Precision Search (VPS) 方法，通过超像素稀疏化+子模函数引导的贪心搜索精确定位关键决策子区域，在 MS COCO/RefCOCO/LVIS 上的忠实度指标(Insertion)分别超过 SOTA 方法 D-RISE 达 23.7%/20.1%/31.6%。

**[Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection](large_self-supervised_models_bridge_the_gap_in_domain_adaptive_object_detection.md)**

:   DINO Teacher 提出用冻结的 DINOv2 大模型替代传统 Mean Teacher 框架中的 EMA 教师，一方面作为更准确的伪标签生成器，另一方面作为特征对齐的代理目标，在多个域自适应目标检测基准上取得了 SOTA 性能（BDD100k 上 +7.6%）。

**[MCCD: Multi-Agent Collaboration-based Compositional Diffusion for Complex Text-to-Image Generation](mccd_multi-agent_collaboration-based_compositional_diffusion_for_complex_text-to.md)**

:   MCCD提出基于多智能体协作的组合式扩散方法，利用MLLM驱动的多智能体系统进行复杂场景解析，并通过层次化组合扩散（高斯mask和区域增强）实现多目标复杂场景的准确高保真生成，且无需训练。

**[MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)**

:   MI-DETR 提出了并行多次查询（MI）机制替代传统 DETR 级联解码器架构，让 object queries 通过多个参数独立的 inquiry heads 并行地从图像特征中学习多模式信息，配合 U-like Feature Interaction（UFI），在 COCO 上以 ResNet-50 backbone 达到 52.7 AP，超越所有已有 DETR 变体。

**[Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](mitigating_memorization_in_text-to-image_diffusion_via_region-aware_prompt_augme.md)**

:   提出 RAPTA（训练时基于目标检测的区域感知 prompt 变体增强）和 ADMCD（推理时三流注意力融合的多模态复制检测），从缓解和检测两个角度端到端地应对文生图扩散模型的训练数据记忆化问题。

**[Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE](mr_detr_instructive_multi-route_training_for_detection_transformers.md)**

:   系统研究 DETR 解码器各组件在 one-to-one/one-to-many 多任务框架下的角色，发现任何单独组件都能有效协调两个目标；基于此提出多路由训练（Instructive Self-Attention + Independent FFN + Route-Aware MoE），推理时丢弃辅助路由不增加任何开销。

**[MulSen-AD: Multi-Sensor Object Anomaly Detection](mulsen_ad_multi_sensor_anomaly_detection.md)**

:   提出首个多传感器异常检测数据集 MulSen-AD，整合 RGB 相机、红外热成像和激光扫描三种模态，以及基线方法 MulSen-TripleAD，通过决策级融合实现 96.1% AUROC 的物体级异常检测。

**[Multiple Object Tracking as ID Prediction](multiple_object_tracking_as_id_prediction.md)**

:   本文提出MOTIP，将多目标跟踪中的目标关联问题重新定义为in-context ID预测任务：给定携带ID嵌入的历史轨迹，直接用标准Transformer解码器预测当前检测的ID标签，无需启发式匹配算法即在DanceTrack上以69.6 HOTA大幅超越前SOTA CO-MOT (65.3)。

**[Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset](object_detection_using_event_camera_a_moe_heat_conduction_based_detector_and_a_n.md)**

:   本文提出 MvHeat-DET，把视觉特征建模为二维热扩散过程，用 MoE 在 DFT/DCT/Haar 三种频域变换之间动态路由，再加上 IoU-aware query selection，做事件流目标检测；同时发布了高清事件相机检测数据集 EvDET200K (10,054 段视频 / 200K bbox / 10 类)。

**[ProbPose: A Probabilistic Approach to 2D Human Pose Estimation](probpose_a_probabilistic_approach_to_2d_human_pose_estimation.md)**

:   ProbPose 提出用标定的概率图（probability map）替代传统热力图进行2D人体关键点定位，引入存在概率（presence probability）显式建模关键点是否在激活窗口内，并通过裁剪数据增强和 OKS 损失的期望风险最小化，显著改善了图像外关键点的定位能力和模型的概率标定质量。

**[ROICtrl: Boosting Instance Control for Visual Generation](roictrl_boosting_instance_control_for_visual_generation.md)**

:   ROICtrl 受目标检测中 ROI-Align 启发，提出互补操作 ROI-Unpool 实现高效精确的 ROI 特征还原，构建了一个与社区微调模型和现有空间/嵌入式插件兼容的扩散模型适配器，在多实例区域控制生成中取得 SOTA 性能并大幅降低计算成本。

**[RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark](rsar_restricted_state_angle_resolver_and_rotated_sar_benchmark.md)**

:   本文从维度映射的统一视角重新审视旋转目标检测中的角度解码器，揭示现有方法忽略单位圆约束导致的预测偏差，提出 Unit Cycle Resolver（UCR），并借助 UCR 构建了目前最大的多类别旋转 SAR 目标检测数据集 RSAR。

**[Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval](search_and_detect_training-free_long_tail_object_detection_via_web-image_retriev.md)**

:   SearchDet提出了一种完全免训练的长尾目标检测框架，通过从Web检索正负样本图像、注意力加权查询生成、SAM区域提议和热力图联合定位，在ODinW上比GroundingDINO提升48.7% mAP、在LVIS上提升59.1% mAP，展示了利用Web作为外部动态记忆进行推理阶段增强的巨大潜力。

**[Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](show_dont_tell_detecting_novel_objects_by_watching_human_videos.md)**

:   本文提出"Show, Don't Tell"范式，通过观看人类操作演示视频自动创建训练数据集，训练专属的物体检测器来识别新颖物体，完全绕过了传统方法中依赖语言描述或 prompt 工程的环节，在真实机器人系统上显著提升了操作物体的检测和识别性能。

**[SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection](simltd_simple_supervised_and_semi-supervised_long-tailed_object_detection.md)**

:   SimLTD 提出一个简洁直观的三阶段框架——先在头部类预训练、再迁移到尾部类、最后在混合采样数据上微调——可选配合无标注图像的半监督学习，在 LVIS v1 基准上全面超越依赖 ImageNet 标签的现有方法。

**[Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)**

:   提出 ESM-YOLO+，一种轻量级可见光-红外融合网络，通过 MEAF 模块（可学习空间掩码+空间注意力的像素级融合）和训练时结构表示增强（SR，推理时无开销的超分辅助监督），在 VEDAI 上达到 84.71% mAP 同时参数量仅 5.1M（减少 93.6%）。

**[Test-Time Backdoor Detection for Object Detection Models](test-time_backdoor_detection_for_object_detection_models.md)**

:   TRACE（TRAnsformation Consistency Evaluation）提出了首个面向目标检测模型的测试时后门样本检测方法，基于两个关键观察——中毒样本在不同背景下检测结果更一致、干净样本在不同聚焦信息下更一致——通过对前景和背景施加变换后计算目标置信度方差来检测中毒样本，实现黑盒通用检测，AUROC 比 SOTA 提升 30%。

**[TornadoNet: Real-Time Building Damage Detection with Ordinal Supervision](tornadonet_real-time_building_damage_detection_with_ordinal_supervision.md)**

:   TornadoNet 构建了首个针对龙卷风灾后街景建筑损坏评估的系统性 benchmark，通过对比 YOLO 系列（CNN）和 RT-DETR（Transformer）在五级损坏检测任务上的表现，并提出序数感知（ordinal-aware）监督策略，使 RT-DETR 的 mAP@0.5 提升 4.8 个百分点，证明了将损坏严重度的有序性质纳入损失函数设计的有效性。

**[Towards RAW Object Detection in Diverse Conditions](towards_raw_object_detection_in_diverse_conditions.md)**

:   提出 AODRaw 数据集（7,785张高分辨率真实RAW图像，62类，9种光照/天气条件），并通过RAW域预训练+跨域蒸馏方案，无需ISP模块即可在多种恶劣条件下实现优异的RAW目标检测性能。
