---
title: >-
  ICLR2026 3D视觉方向 67篇论文解读
description: >-
  67篇ICLR2026 3D视觉方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧊 3D视觉

**🔬 ICLR2026** · 共 **67** 篇

**[3Dgeer 3D Gaussian Rendering Made Exact And Efficient For Generic Cameras](3dgeer_3d_gaussian_rendering_made_exact_and_efficient_for_generic_cameras.md)**

:   提出 3DGEER 框架，通过推导沿光线积分高斯密度的闭式解、设计粒子包围截锥体 (PBF) 进行精确高效的光线-粒子关联、以及引入双极等角投影 (BEAP) 统一宽视场相机表示，在任意相机模型下实现了几何精确且实时高效的 3D 高斯渲染，在鱼眼和针孔数据集上全面超越现有方法。

**[A Genetic Algorithm For Navigating Synthesizable Molecular Spaces](a_genetic_algorithm_for_navigating_synthesizable_molecular_spaces.md)**

:   提出 SynGA，一种直接在合成路线（合成树）上操作的遗传算法，通过自定义的交叉和变异算子将搜索严格约束在可合成分子空间内，结合 ML 驱动的构建块过滤实现 SOTA 的可合成类似物搜索和属性优化性能。

**[A Step To Decouple Optimization In 3Dgs](a_step_to_decouple_optimization_in_3dgs.md)**

:   深入分析 3DGS 优化中被忽视的更新步耦合（不可见视点下的隐式更新和动量重缩放）和梯度耦合（正则化与光度损失在 Adam 动量中的耦合），通过解耦和重组提出 AdamW-GS 优化器，在不引入额外剪枝操作的情况下同时提升重建质量和减少冗余原语。

**[Augmented Radiance Field A General Framework For Enhanced Gaussian Splatting](augmented_radiance_field_a_general_framework_for_enhanced_gaussian_splatting.md)**

:   提出增强辐射场 (Augmented Radiance Field) 框架，通过设计具有视角相关不透明度的增强高斯核来显式建模高光分量，并引入误差驱动的补偿策略（2D 高斯初始化 → 逆投影至 3D → 联合优化），作为后处理即插即用地增强现有 3DGS 场景，在多个数据集上超越 SOTA NeRF 方法，同时仅需二阶球谐即可捕获复杂光照。

**[Cadrille Multi-Modal Cad Reconstruction With Reinforcement Learning](cadrille_multi-modal_cad_reconstruction_with_reinforcement_learning.md)**

:   cadrille 是首个同时处理点云、多视角图像和文本输入的多模态 CAD 重建模型，通过 VLM 基础架构 + SFT + RL 微调的三阶段训练范式，在 10 个 CAD 重建基准上达到 SOTA，尤其是 RL 微调将无效率降至接近 0%。

**[Clods Visual-Only Unsupervised Cloth Dynamics Learning In Unknown Conditions](clods_visual-only_unsupervised_cloth_dynamics_learning_in_unknown_conditions.md)**

:   CloDS 提出首个从多视角视频中无监督学习布料动力学的框架，通过 Spatial Mapping Gaussian Splatting 建立 2D 图像到 3D 网格的可微映射，结合双位置不透明度调制解决自遮挡问题，使 GNN 在无物理参数监督下就能学到接近全监督水平的布料动力学。

**[Color3D Controllable And Consistent 3D Colorization With Personalized Colorizer](color3d_controllable_and_consistent_3d_colorization_with_personalized_colorizer.md)**

:   Color3D 提出"只上色一张关键视角→微调个性化 colorizer→传播颜色到所有视角和时间步"的范式，将复杂的 3D 上色问题转化为单图上色+颜色传播问题，在静态和动态 3D 场景上都实现了丰富色彩、跨视角一致性和用户可控性的统一。

**[Coopertrim Adaptive Data Selection For Uncertainty-Aware Cooperative Perception](coopertrim_adaptive_data_selection_for_uncertainty-aware_cooperative_perception.md)**

:   提出 CooperTrim 自适应特征选择框架，通过共形时序不确定性度量评估特征相关性，并用数据驱动机制动态决定共享数量，在协同语义分割中实现 80.28% 带宽降低且性能可比，首次将选择性共享应用于协同分割任务。

**[Core-3D Context-Aware Open-Vocabulary Retrieval By Embeddings In 3D](core-3d_context-aware_open-vocabulary_retrieval_by_embeddings_in_3d.md)**

:   提出CORE-3D，一个无需训练的开放词汇3D语义分割与自然语言目标检索流水线，通过渐进式粒度掩码生成、上下文感知CLIP编码和多视角3D融合，在Replica和ScanNet上超越现有方法。

**[Crisp Contact-Guided Real2Sim From Monocular Video With Planar Scene Primitives](crisp_contact-guided_real2sim_from_monocular_video_with_planar_scene_primitives.md)**

:   提出 CRISP，一种从单目视频中恢复可仿真人体运动和场景几何的方法，通过拟合平面原语获取干净的仿真就绪几何体，结合人体-场景接触建模重建被遮挡区域，将人形控制器的运动追踪失败率从 55.2% 降至 6.9%。

**[Ctrlshift High-Quality Geometry-Aware Object Manipulation In Visual Generation](ctrlshift_high-quality_geometry-aware_object_manipulation_in_visual_generation.md)**

:   提出Ctrl&Shift，一个端到端扩散框架，通过将物体操纵分解为物体移除+参考引导修复，并注入相对相机位姿控制，首次在不依赖显式3D重建的情况下实现几何一致的细粒度物体操纵。

**[D-Rex Differentiable Real-To-Sim-To-Real Engine For Learning Dexterous Grasping](d-rex_differentiable_real-to-sim-to-real_engine_for_learning_dexterous_grasping.md)**

:   提出D-REX，一个基于高斯表示的可微real-to-sim-to-real引擎，通过视觉观测和机器人控制信号进行端到端物体质量辨识，并利用辨识的质量进行力感知的灵巧抓取策略学习，有效缩小了sim-to-real差距。

**[Diffwind Physics-Informed Differentiable Modeling Of Wind-Driven Object Dynamics](diffwind_physics-informed_differentiable_modeling_of_wind-driven_object_dynamics.md)**

:   提出 DiffWind，一个物理约束的可微分框架，通过将风建模为网格物理场、物体表示为 3D Gaussian Splatting 粒子系统、用 Material Point Method（MPM）建模风-物交互，并引入 Lattice Boltzmann Method（LBM）作为物理约束，实现了从视频中联合重建风力场和物体运动，并支持新风条件下的前向仿真和风力迁移等应用，在自建的 WD-Objects 数据集上显著超越已有动态场景建模方法。

**[Dissecting Chronos Sparse Autoencoders Reveal Causal Feature Hierarchies In Time](dissecting_chronos_sparse_autoencoders_reveal_causal_feature_hierarchies_in_time.md)**

:   首次将稀疏自编码器 (SAE) 应用于时间序列基础模型 Chronos-T5-Large，通过 392 次因果消融实验揭示了深度依赖的特征层级：中层编码器集中了因果关键的突变检测特征，而语义最丰富的末层编码器反而因果重要性最低。

**[Dynamic Novel View Synthesis In High Dynamic Range](dynamic_novel_view_synthesis_in_high_dynamic_range.md)**

:   首次提出 HDR 动态新视角合成 (HDR DNVS) 问题，并设计 HDR-4DGS 框架，通过动态色调映射模块在时变场景中实现时序一致的 HDR 辐射场重建，在合成和真实数据集上均超越现有方法。

**[Efficient-Lvsm Faster Cheaper And Better Large View Synthesis Model Via Decouple](efficient-lvsm_faster_cheaper_and_better_large_view_synthesis_model_via_decouple.md)**

:   提出 Efficient-LVSM，通过解耦输入视图编码与目标视图生成的双流架构，将新视图合成的复杂度从 $O(N_{in}^2)$ 降至 $O(N_{in})$，在 RealEstate10K 上以 50% 训练时间达到 SOTA（29.86 dB PSNR），推理速度提升 4.4 倍。

**[Egonight Towards Egocentric Vision Understanding At Night With A Challenging Ben](egonight_towards_egocentric_vision_understanding_at_night_with_a_challenging_ben.md)**

:   构建首个系统性夜间第一人称视觉基准 EgoNight，包含日夜对齐的合成/真实视频和 3658 个 QA 对（12种类型，300+小时人工标注），揭示所有 SOTA MLLM 在日夜转换中的显著性能下降。

**[Egoworld Translating Exocentric View To Egocentric View Using Rich Exocentric Ob](egoworld_translating_exocentric_view_to_egocentric_view_using_rich_exocentric_ob.md)**

:   提出 EgoWorld 框架，通过从外部视角提取点云、3D 手部姿态和文本描述等多模态观测，利用两阶段管线将单张第三人称图像转换为高质量的第一人称视图。

**[Fast Estimation Of Wasserstein Distances Via Regression On Sliced Wasserstein Di](fast_estimation_of_wasserstein_distances_via_regression_on_sliced_wasserstein_di.md)**

:   提出通过将 Wasserstein 距离回归到 Sliced Wasserstein (SW) 距离的线性模型（RG 框架），实现对 Wasserstein 距离的快速高效估计，在低数据场景下显著优于深度学习方法 Wasserstein Wormhole。

**[Fastgha Generalized Few-Shot 3D Gaussian Head Avatars With Real-Time Animation](fastgha_generalized_few-shot_3d_gaussian_head_avatars_with_real-time_animation.md)**

:   提出 FastGHA，一个前馈式少样本 3D 高斯头部化身生成框架，从 4 张任意表情/视角的输入图像在 ~1 秒内重建可动画的 3D 高斯头部，支持 62 FPS 实时动画，在 Ava-256 上 PSNR 达到 22.5 dB（超越 Avat3r 的 20.7，且快 7.75 倍）。

**[Fedal Federated Dataset Learning For General Time Series Foundation Models](fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)**

:   提出 FeDaL 联邦框架从头训练通用时序基础模型（TSFM），通过客户端域偏差消除（DBE）和服务器全局偏差消除（GBE）处理数据集级异质性，在8个任务上超越54个baseline。

**[Fused-Planes Why Train A Thousand Tri-Planes When You Can Share](fused-planes_why_train_a_thousand_tri-planes_when_you_can_share.md)**

:   提出 Fused-Planes，通过宏观-微观分解将 Tri-Plane 表示分为共享的类级基平面（macro）和对象特有的细节平面（micro），结合潜空间渲染，实现 7× 训练加速、3× 内存压缩，同时保持甚至超越独立 Tri-Plane 的重建质量。

**[Generalizable Coarse-To-Fine Robot Manipulation Via Language-Aligned 3D Keypoint](generalizable_coarse-to-fine_robot_manipulation_via_language-aligned_3d_keypoint.md)**

:   CLAP（Coarse-to-fine Language-Aligned manipulation Policy）通过任务分解、VLM微调的3D关键点预测和3D感知表征三个核心组件，实现了对新指令和新环境的强泛化能力，在 GemBench 上以 1/5 的训练数据比 SOTA 高出 12%。

**[Geometry-Aware 4D Video Generation For Robot Manipulation](geometry-aware_4d_video_generation_for_robot_manipulation.md)**

:   提出几何感知的4D视频生成框架，通过跨视角 pointmap 对齐监督在视频扩散模型中强制多视角3D一致性，无需相机位姿输入即可从新视角生成时空对齐的未来 RGB-D 序列，并可直接用 FoundationPose 从生成视频中恢复机器人末端执行器轨迹。

**[Geopurify A Data-Efficient Geometric Distillation Framework For Open-Vocabulary ](geopurify_a_data-efficient_geometric_distillation_framework_for_open-vocabulary_.md)**

:   提出 GeoPurify 框架，通过从 3D 自监督教师模型蒸馏几何先验来净化 2D VLM 投影到 3D 的噪声特征，仅用约 1.5% 的训练数据即可达到或超越全量训练的 SOTA 开放词汇 3D 分割性能。

**[Giq Benchmarking 3D Geometric Reasoning Of Vision Foundation Models With Simulat](giq_benchmarking_3d_geometric_reasoning_of_vision_foundation_models_with_simulat.md)**

:   提出 GIQ 基准数据集，包含 224 种合成和真实多面体，通过单目 3D 重建、对称性检测、心理旋转测试和零样本分类四项任务系统评估视觉基础模型的几何推理能力，揭示了当前模型在基本几何理解上的显著不足。

**[Hdr-Nsff High Dynamic Range Neural Scene Flow Fields](hdr-nsff_high_dynamic_range_neural_scene_flow_fields.md)**

:   提出 HDR-NSFF，将 HDR 视频重建从传统的 2D 像素级融合范式转变为 4D 时空建模，从交替曝光单目视频中联合重建 HDR 辐射场、3D 场景流、几何和色调映射，实现了时空一致的动态 HDR 新视角合成。

**[Into The Rabbit Hull From Task-Relevant Concepts](into_the_rabbit_hull_from_task-relevant_concepts.md)**

:   本文通过稀疏自编码器（SAE）从 DINOv2 中提取 32,000 个视觉概念字典，系统研究了不同下游任务（分类/分割/深度估计）如何选择性地使用这些概念，揭示了表示空间的几何结构超越了线性稀疏编码假说（LRH），并提出了基于 Minkowski 和的新表示假说（MRH），认为 token 是多个凸混合的叠加。

**[Into The Rabbit Hull From Task-Relevant Concepts In Dino To Minkowski Geometry](into_the_rabbit_hull_from_task-relevant_concepts_in_dino_to_minkowski_geometry.md)**

:   通过在 DINOv2 上训练 32,000 单元的 Sparse Autoencoder 字典，系统分析了下游任务如何招募不同概念，发现表征几何偏离线性稀疏假说（LRH），进而提出 Minkowski Representation Hypothesis（MRH），认为 token 表征是多个凸多面体的 Minkowski 和，概念由原型点的邻近性而非线性方向定义。

**[Joint Shadow Generation And Relighting Via Light-Geometry Interaction Maps](joint_shadow_generation_and_relighting_via_light-geometry_interaction_maps.md)**

:   提出 Light-Geometry Interaction (LGI) maps，一种从单目深度估计中编码光照-遮挡关系的 2.5D 表示，嵌入 bridge matching 生成框架中实现阴影生成与物体重光照的联合建模，在合成和真实图像上均取得 SOTA 效果。

**[Lavca Llm-Assisted Visual Cortex Captioning](lavca_llm-assisted_visual_cortex_captioning.md)**

:   提出 LaVCa 方法，利用 LLM 为人类视觉皮层的每个体素生成自然语言描述（caption），通过"编码模型→最优图像选取→MLLM生成描述→LLM关键词提炼+句子组合"四步流程，比已有方法 BrainSCUBA 更准确、更多样地揭示了体素级视觉选择性。

**[Learning Part-Aware Dense 3D Feature Field For Generalizable Articulated Object ](learning_part-aware_dense_3d_feature_field_for_generalizable_articulated_object_.md)**

:   提出 PA3FF（Part-Aware 3D Feature Field），一种原生 3D 的稠密部件感知特征表示，通过 Sonata 预训练骨干 + 几何/语义对比学习获得零部件级特征，结合 Part-Aware Diffusion Policy (PADP) 实现少样本、高泛化性的关节物体操作，在仿真和真实环境中均大幅超越 CLIP/DINOv2/GenDP 等基线。

**[Learning Physics-Grounded 4D Dynamics With Neural Gaussian Force Fields](learning_physics-grounded_4d_dynamics_with_neural_gaussian_force_fields.md)**

:   提出NGFF框架，从多视角RGB图像构建3D高斯表示并学习显式神经力场驱动物理动力学，通过ODE求解实现交互式物理真实4D视频生成，比传统高斯模拟器快两个数量级，超越Veo3和NVIDIA Cosmos。

**[Learning Unified Representation Of 3D Gaussian Splatting](learning_unified_representation_of_3d_gaussian_splatting.md)**

:   发现3DGS的原生参数表示（位置+四元数+缩放+SH系数+透明度）因非唯一性和异质性不适合神经网络学习，提出基于等概率面子流形场的统一表示，建立唯一映射并消除数值异质性，配合VAE+流形距离实现更好的3D学习。

**[Lito Surface Light Field Tokenization](lito_surface_light_field_tokenization.md)**

:   提出LiTo——通过将表面光场(surface light field)编码为紧凑latent向量集合来同时建模3D几何和视角依赖外观：输入RGB-D多视角图像的光场随机子采样 -> Perceiver IO编码器(支持100万token输入的3D局部attention) + flow-matching几何解码器 + 高阶球谐Gaussian解码器 -> 实现重建和单图到3D生成都超越TRELLIS，首次在latent 3D表示中建模高光/菲涅尔反射等视角依赖效果。

**[Megs2 Memory-Efficient Gaussian Splatting Via Spherical Gaussians And Unified Pr](megs2_memory-efficient_gaussian_splatting_via_spherical_gaussians_and_unified_pr.md)**

:   提出MEGS2——从渲染VRAM角度出发压缩3DGS：用可裁剪的任意方向球面高斯(SG)完全替代球谐函数(SH)降低每个primitive的参数量 + 统一软剪枝框架将primitive数量和lobe数量的裁剪建模为单一内存约束优化问题 -> 实现8x静态VRAM压缩和6x渲染VRAM压缩，同时保持渲染质量，首次让3DGS在移动端实时运行。

**[Moe-Gs Mixture Of Experts For Dynamic Gaussian Splatting](moe-gs_mixture_of_experts_for_dynamic_gaussian_splatting.md)**

:   提出 MoE-GS，首个将混合专家架构引入动态高斯泼溅的框架，通过 Volume-aware Pixel Router 自适应融合多种异构变形先验（HexPlane/逐高斯/多项式/插值），在 N3V 和 Technicolor 数据集上一致超越 SOTA，并通过单次渲染、门控剪枝和知识蒸馏保持效率。

**[Mono4Dgs-Hdr High Dynamic Range 4D Gaussian Splatting From Alternating-Exposure ](mono4dgs-hdr_high_dynamic_range_4d_gaussian_splatting_from_alternating-exposure_.md)**

:   首次解决从无位姿交替曝光单目视频重建可渲染 4D HDR 场景的问题，通过两阶段优化（正交视频空间 → 世界空间）、Video-to-World 高斯变换策略和时间亮度正则化，在合成数据上达到 37.64 dB HDR PSNR、161 FPS，全面超越现有方法。

**[Multimat Multimodal Program Synthesis For Procedural Materials Using Large Multi](multimat_multimodal_program_synthesis_for_procedural_materials_using_large_multi.md)**

:   提出MultiMat——首个利用大型多模态模型(LMM)进行程序化材质合成的框架,在生成过程中同时处理文本程序表示和中间节点的视觉渲染结果,配合约束树搜索推理算法确保生成图的静态正确性,在产级程序化材质上的无条件和条件合成均显著优于纯文本基线。

**[Nova3R Non-Pixel-Aligned Visual Transformer For Amodal 3D Reconstruction](nova3r_non-pixel-aligned_visual_transformer_for_amodal_3d_reconstruction.md)**

:   提出NOVA3R——从无位姿图像进行非像素对齐的完整3D重建：用可学习场景token跨视角聚合全局信息 + 基于flow-matching的扩散3D解码器生成完整(含遮挡区域)的点云，解决像素对齐方法只能重建可见面且重叠区域有冗余几何的两大根本限制，在SCRREAM/GSO等数据集上场景级和物体级重建均超越SOTA。

**[Omni-View Unlocking How Generation Facilitates Understanding In Unified 3D Model](omni-view_unlocking_how_generation_facilitates_understanding_in_unified_3d_model.md)**

:   构建统一的3D场景理解与生成模型 Omni-View，通过纹理模块（新视角合成）和几何模块（深度/位姿估计）的生成能力增强理解性能，在 VSI-Bench 上达到 55.4 分超越所有现有专用3D理解模型。

**[One2Scene Geometric Consistent Explorable 3D Scene Generation From A Single Ima](one2scene_geometric_consistent_explorable_3d_scene_generation_from_a_single_ima.md)**

:   提出 One2Scene 三阶段框架，将单图生成可探索 3D 场景分解为全景生成→前馈 3D 高斯溅射构建几何支架→支架引导的新视角合成，通过将全景深度估计重新表述为多视图立体匹配问题，实现几何一致且可自由探索的 3D 场景生成。

**[One2Scene Geometric Consistent Explorable 3D Scene Generation From A Single Imag](one2scene_geometric_consistent_explorable_3d_scene_generation_from_a_single_imag.md)**

:   提出One2Scene——将单图到可探索3D场景的病态问题分解为三个子任务：(1)全景图生成扩展视觉覆盖 (2)前馈3DGS网络从稀疏锚点视角构建显式3D几何scaffold (3)scaffold引导的新视角合成，通过Dual-LoRA融合高质量锚点视角和几何先验，在大视角变化下实现几何一致且逼真的场景生成，显著超越SOTA。

**[Openfly A Comprehensive Platform For Aerial Vision-Language Navigation](openfly_a_comprehensive_platform_for_aerial_vision-language_navigation.md)**

:   构建OpenFly——航空视觉-语言导航(VLN)综合平台：集成4种渲染引擎(UE/GTA V/Google Earth/3DGS)+开发全自动数据生成工具链(点云获取→语义分割→轨迹生成→GPT-4o指令)+构建10万轨迹大规模数据集(18场景)+提出关键帧感知VLN模型OpenFly-Agent(关键帧选择+视觉token融合)，在已见/未见场景分别以14.0%/7.9%的成功率优势超越现有方法。

**[Partsam A Scalable Promptable Part Segmentation Model Trained On Native 3D Data](partsam_a_scalable_promptable_part_segmentation_model_trained_on_native_3d_data.md)**

:   提出首个在大规模原生 3D 数据上训练的可提示部件分割模型 PartSAM，采用 triplane 双分支编码器（冻结 SAM 先验 + 可学习 3D 分支）和 SAM 风格解码器，通过模型在环标注流程构建 500 万+形状-部件对，在开放世界设置下单次点击即超越 Point-SAM 90%+。

**[Pd2Gs Part-Level Decoupling And Continuous Deformation Of Articulated Objects Vi](pd2gs_part-level_decoupling_and_continuous_deformation_of_articulated_objects_vi.md)**

:   提出 PD²GS 框架，通过学习共享的 canonical 高斯场并将每个交互状态建模为其连续形变，实现铰接物体的部件级解耦、重建和连续控制，采用粗到细的运动轨迹聚类 + SAM 引导的边界细化，无需手动监督。

**[Peering Into The Unknown Active View Selection With Neural Uncertainty Maps For ](peering_into_the_unknown_active_view_selection_with_neural_uncertainty_maps_for_.md)**

:   提出 PUN 方法，通过轻量级 UPNet（ViT）从单张图像预测神经不确定性图（球面坐标系下所有候选视点的不确定性分布），避免迭代 NeRF 重训练，仅用一半视点达到全视点上界的重建质量，实现 400 倍加速和 50%+ 计算节省。

**[Pyspatial Generating 3D Visual Programs For Zero-Shot Spatial Reasoning](pyspatial_generating_3d_visual_programs_for_zero-shot_spatial_reasoning.md)**

:   提出pySpatial——视觉编程框架让MLLM通过生成Python代码调用3D空间工具(重建/相机恢复/新视角渲染)实现零样本3D空间推理：将2D输入转化为可探索的3D场景→MLLM在结构化3D表示上显式推理而非隐式想象,在MindCube上超越GPT-4.1-mini 12.94%,并成功用于真实室内四足机器人导航。

**[Quadgpt Native Quadrilateral Mesh Generation With Autoregressive Models](quadgpt_native_quadrilateral_mesh_generation_with_autoregressive_models.md)**

:   提出QuadGPT——首个端到端自回归生成原生四边形网格的框架：设计统一tokenization处理三角形/四边形混合拓扑(三角形面用padding统一为四顶点)，采用Hourglass Transformer压缩面序列+截断序列训练支持高面数网格，引入tDPO(截断DPO)强化学习微调奖励结构化边环形成，在几何精度和拓扑质量上显著超越三角形→四边形转换流水线。

**[Quantized Visual Geometry Grounded Transformer](quantized_visual_geometry_grounded_transformer.md)**

:   针对十亿级 3D 重建模型 VGGT 的部署需求，提出首个专用 PTQ 框架 QuantVGGT，通过双重平滑细粒度量化（Hadamard 旋转 + 通道平滑）解决特殊 token 导致的重尾分布，以及噪声过滤多样化采样解决校准不稳定问题，4-bit 量化实现 3.7× 内存压缩和 2.5× 加速，保持 98%+ 精度。

**[Scaling Sequence-To-Sequence Generative Neural Rendering](scaling_sequence-to-sequence_generative_neural_rendering.md)**

:   提出 Kaleido，一系列将 3D 视为视频特殊子域的生成模型，通过序列到序列的图像合成范式和 Masked Autoregressive 框架实现无需显式 3D 表示的新视角合成，首次在多视角设置下匹配逐场景优化方法的质量。

**[Scenetransporter Optimal Transport-Guided Compositional Latent Diffusion For Sin](scenetransporter_optimal_transport-guided_compositional_latent_diffusion_for_sin.md)**

:   提出SceneTransporter——用最优传输(OT)引导组合latent扩散实现单图结构化3D场景生成：通过去偏聚类探查揭示部件级生成器在open-world场景中失败的原因(缺乏分配约束)→将结构化生成重新建模为全局关联分配问题→在去噪循环中求解熵OT目标→(1)OT计划门控交叉注意力实现排他性一对一路由(防止特征纠缠) (2)竞争性传输鼓励相似patch分组+边缘正则化确保清晰边界→显著提升实例级一致性和几何保真度。

**[Sharp Monocular View Synthesis In Less Than A Second](sharp_monocular_view_synthesis_in_less_than_a_second.md)**

:   提出SHARP——从单张照片在不到1秒内通过单次前馈回归度量3D高斯表示→支持100+FPS实时高分辨率渲染→在多个数据集上零样本泛化LPIPS降低25-34%/DISTS降低21-43%/合成速度比扩散方法快1000倍，设定了单目视图合成的新SOTA。

**[Splat And Distill Augmenting Teachers With Feed-Forward 3D Reconstruction For 3D](splat_and_distill_augmenting_teachers_with_feed-forward_3d_reconstruction_for_3d.md)**

:   提出Splat and Distill(SnD)——通过前馈3D重建增强teacher对student进行3D感知蒸馏：将teacher 2D特征提升到3D高斯表示→从新视角渲染特征→监督student→与逐场景优化不同→前馈提升避免特征平均化→teacher一致性随student迭代改善(EMA)→在深度/法线/分割/对应4个任务上全面超越FiT3D/MEF等先前方法。

**[Splat Feature Solver](splat_feature_solver.md)**

:   将3D特征提升(从2D语义→3D高斯)形式化为稀疏线性逆问题AX=B→闭式求解→证明凸损失下全局最优误差上界→Tikhonov引导+后处理聚合两种正则化稳定解→核/特征无关(通用于3DGS/2DGS/Beta Splatting+CLIP/DINO/ViT/CNN)→开放词汇3D分割SOTA且仅需分钟级计算。

**[Station2Radar Query Conditioned Gaussian Splatting For Precipitation Field](station2radar_query_conditioned_gaussian_splatting_for_precipitation_field.md)**

:   提出QCGS(Query-Conditioned Gaussian Splatting)——首个将气象站观测+卫星图像融合生成降水场的框架(无需雷达)：关键洞察→传统高斯加权插值=高斯溅射的特例→QCGS学习自适应高斯参数+选择性只渲染降水区域→比传统网格化降水产品RMSE降低50%+→分辨率灵活/实时生成。

**[Streamsplat Towards Online Dynamic 3D Reconstruction From Uncalibrated Video Str](streamsplat_towards_online_dynamic_3d_reconstruction_from_uncalibrated_video_str.md)**

:   StreamSplat 提出了一个完全前馈的在线动态3D重建框架，通过概率位置采样、双向形变场和自适应高斯融合三大创新，能从未标定视频流中即时生成动态3DGS表示，速度比优化方法快1200倍。

**[Stroke3D Lifting 2D Strokes Into Rigged 3D Model Via Latent Diffusion Models](stroke3d_lifting_2d_strokes_into_rigged_3d_model_via_latent_diffusion_models.md)**

:   Stroke3D 首次实现从用户绘制的2D笔画和文本提示直接生成绑骨3D网格模型，采用骨骼优先的两阶段流水线：先用图VAE+图DiT生成可控3D骨骼，再通过TextuRig数据集增强和SKA-DPO优化生成高质量网格。

**[Stylos Multi-View 3D Stylization With Single-Forward Gaussian Splatting](stylos_multi-view_3d_stylization_with_single-forward_gaussian_splatting.md)**

:   Stylos 提出了一个单次前馈的3D风格迁移框架，通过共享Transformer骨干的双路径设计（几何自注意力+风格交叉注意力）和体素级3D风格损失，实现从未标定输入的零样本3D风格化，支持单视角到数百视角的扩展。

**[Surfsplat Conquering Feedforward 2D Gaussian Splatting With Surface Continuity P](surfsplat_conquering_feedforward_2d_gaussian_splatting_with_surface_continuity_p.md)**

:   SurfSplat 提出基于2DGS的前馈3D重建框架，通过表面连续性先验将高斯的旋转和尺度与邻域位置绑定、以及强制透明度混合策略解决颜色偏差，并引入HRRC指标揭示高分辨率下的重建质量差异。

**[Text-To-3D By Stitching A Multi-View Reconstruction Network To A Video Generator](text-to-3d_by_stitching_a_multi-view_reconstruction_network_to_a_video_generator.md)**

:   提出VIST3A框架——通过模型拼接(model stitching)将预训练视频生成器的latent空间与前馈3D重建模型(如AnySplat/MVDUSt3R/VGGT)无缝对接，再用直接奖励微调(direct reward finetuning)对齐生成模型与拼接后的3D解码器，实现高质量端到端text-to-3DGS和text-to-pointmap生成，在T3Bench/SceneBench/DPG-Bench上全面超越现有方法。

**[Topology-Preserved Auto-Regressive Mesh Generation In The Manner Of Weaving Silk](topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)**

:   提出一种类似"织丝"的网格 tokenization 算法，通过顶点分层和排序提供规范的拓扑框架，保证生成网格的流形性、水密性、法线一致性和部件感知性，同时达到 SOTA 压缩效率。

**[Ufo-4D Unposed Feedforward 4D Reconstruction From Two Images](ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)**

:   提出 UFO-4D，一个统一的前馈框架，仅从两张无位姿图像直接预测动态 3D 高斯表示，实现 3D 几何、3D 运动和相机位姿的联合一致估计，在几何和运动基准上比现有方法提升达 3 倍。

**[Uncertainty Matters In Dynamic Gaussian Splatting For Monocular 4D Reconstructio](uncertainty_matters_in_dynamic_gaussian_splatting_for_monocular_4d_reconstructio.md)**

:   提出 USplat4D，一种不确定性感知的动态高斯泼溅框架，通过估计每个高斯的时变不确定性并构建不确定性引导的时空图来传播可靠运动线索，显著提升了遮挡区域和极端新视角下的单目 4D 重建质量。

**[Universal Beta Splatting](universal_beta_splatting.md)**

:   提出 Universal Beta Splatting (UBS)，将 3D 高斯 Splatting 推广为 N 维各向异性 Beta 核，通过逐维度形状控制在单一表示中统一建模空间几何、视角依赖外观和场景动态，实现了可解释的场景分解和 SOTA 渲染质量。

**[Urbangs A Scalable And Efficient Architecture For Geometrically Accurate Large-S](urbangs_a_scalable_and_efficient_architecture_for_geometrically_accurate_large-s.md)**

:   提出 UrbanGS，一个面向城市级场景的可扩展 3DGS 重建框架，通过深度一致的 D-Normal 正则化、空间自适应高斯剪枝和统一分区策略，同时提升几何精度、渲染质量和内存效率。

**[Weight Space Representation Learning On Diverse Nerf Architectures](weight_space_representation_learning_on_diverse_nerf_architectures.md)**

:   提出首个能处理多种 NeRF 架构（MLP/tri-plane/hash table）权重的表示学习框架，通过 Graph Meta-Network 编码器 + SigLIP 对比损失构建架构无关的潜在空间，在 13 种 NeRF 架构上实现分类、检索和语言任务，并能泛化到训练时未见的架构。
