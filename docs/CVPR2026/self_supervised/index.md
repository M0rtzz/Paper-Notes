---
title: >-
  CVPR2026 自监督/表示学习方向34篇论文解读
description: >-
  34篇CVPR2026的自监督/表示学习方向论文解读，涵盖自监督学习、持续学习、布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "自监督/表示学习"
  - "论文解读"
  - "论文笔记"
  - "自监督学习"
  - "持续学习"
  - "布局/合成"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**📷 CVPR2026** · **34** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (24)](../../ICML2026/self_supervised/index.md) · [💬 ACL2026 (1)](../../ACL2026/self_supervised/index.md) · [🔬 ICLR2026 (16)](../../ICLR2026/self_supervised/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/self_supervised/index.md) · [🧠 NeurIPS2025 (35)](../../NeurIPS2025/self_supervised/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/self_supervised/index.md)

🔥 **高频主题：** 自监督学习 ×6 · 持续学习 ×4 · 布局/合成 ×2

**[A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)**

:   提出 PL-Stitch 自监督框架，利用 Plackett-Luce 概率排序模型将视频帧的时序排序作为预训练信号，学习具有"程序感知"能力的视频表示，在手术阶段识别和烹饪动作分割上全面超越现有自监督方法。

**[An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning](an_optimal_transport_driven_approach_for_cultivating_latent_space_in_online_incr.md)**

:   提出基于最优传输理论的在线混合模型学习框架 (MMOT)，通过为每个类别维护多个自适应质心来更精确地表征在线数据流的多模态特性，结合动态保持策略增强类别区分能力，在在线类增量学习 (OCIL) 中有效缓解灾难性遗忘。

**[BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)**

:   提出 BoSS——一种可扩展的 oracle 策略选择框架：在每轮主动学习中，并行运行多种查询策略在随机子池上生成候选 batch，通过冻结 backbone 仅重训最后一层快速评估每个候选 batch 的性能增益，选出最优 batch，从而量化现有 AL 策略与理论最优之间的差距。

**[Breaking the Tuning Barrier: Zero-Hyperparameters Yield Multi-Corner Analysis Via Learned Priors](breaking_the_tuning_barrier_zerohyperparameters_yi.md)**

:   用 TabPFN 基础模型的学习先验替代人工设计先验，实现零超参数调优的多角（PVT）良率分析，结合自动特征选择（1152D→48D）和不确定性引导主动学习，在工业级 SRAM 基准上达到 SOTA 精度（MRE 低至 0.11%）同时减少 10× 以上验证成本。

**[CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach](cheem_continual_learning_by_reuse_new_adapt_and_skip_--_a_hierarchical_explorati.md)**

:   提出 CHEEM 框架，通过分层探索-利用采样的 NAS 自动学习任务感知的动态 ViT 骨干——在每一层选择 Reuse/New/Adapt/Skip 四种操作——在 MTIL 和 VDD 两个挑战性持续学习基准上显著超越提示类方法，接近全量微调上界。

**[Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](com_pt_chain_of_models_pretraining.md)**

:   提出 Chain-of-Models Pre-Training (CoM-PT)，将视觉基础模型按大小排列形成"模型链"，通过从小到大的逆向知识转移（权重初始化+特征蒸馏）逐步加速训练，实现性能无损的训练加速且效率随模型家族规模增长而提升。

**[CraterBench-R: Instance-Level Crater Retrieval for Planetary Scale](craterbench-r_instance-level_crater_retrieval_for_planetary_scale.md)**

:   首次将陨石坑分析形式化为实例级图像检索问题——提出CraterBench-R基准(~25K火星陨石坑ID, 50K gallery, 5K查询)，诊断发现单向量池化有精度上限+有监督度量学习反而退化，提出无训练的实例token聚合(选K个种子+余弦最近邻残差分配)将196个ViT patch token压缩为K个代表token做late interaction匹配，K=64时匹配全token精度且存储大幅降低，实用两阶段管线(单向量粗筛+实例token精排)恢复89-94%完整精度。

**[D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping](d2dewarp_dual_dimensions_geometric_representation_learning_based_document_image_.md)**

:   提出 D2Dewarp——首个从水平和垂直双维度学习文档几何表示的去畸变方法：UNet 双解码器分别预测水平线（文档/表格/文本行的上下边界）和垂直线（左右边界），HV Fusion Module 通过混合注意力交叉融合两个方向的特征，并构建了包含 114K 张图的 DocDewarpHV 数据集提供双维度标注。

**[DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)**

:   通过系统分析发现 DiT 各 block 间的表示多样性是有效学习的关键因素，提出 DiverseDiT：用长残差连接多样化输入 + 表示多样性损失显式促进 block 间特征差异化，无需外部引导模型即可加速收敛并提升生成质量。

**[Free-Grained Hierarchical Visual Recognition](free-grained_hierarchical_visual_recognition.md)**

:   提出"自由粒度"层级视觉识别（free-grained hierarchical recognition），允许训练标签出现在分类法的任意层级，并提出文本引导伪属性和分类法引导半监督学习两种方法来弥补缺失监督，推理时模型自适应选择预测深度。

**[GeoBridge: A Semantic-Anchored Multi-View Foundation Model for Geo-Localization](geobridge_semantic-anchored_multi-view_foundation_model_for_geo-localization.md)**

:   GeoBridge 提出语义锚定的多视角地理定位基础模型，将无人机/街景/卫星影像通过文本描述构建跨模态语义桥梁，实现双向跨视角匹配和语言到图像定位，并构建了GeoLoc数据集（50K+对，36国）。

**[Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning](group_dinomics_incorporating_people_dynamics_into_dino_for_self_supervised_group_activity_feature_learning.md)**

:   提出利用 DINOv3 结合两个自监督预训练任务（人物光流估计和群体相关物体定位）来学习群体活动特征（GAF），在无群体活动标注的情况下大幅超越现有方法。

**[Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces](hier-cos_making_deep_features_hierarchy-aware_via_composition_of_orthogonal_subs.md)**

:   提出 Hier-COS 框架，通过为层次树中每个节点分配正交基向量，构造理论上保证层次一致性的层次感知向量空间(HAVS)，首次统一了"层次感知细粒度分类"和"层次多级分类"，同时提出新评估指标HOPS，在4个数据集上全面超越SOTA。

**[HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)**

:   本文识别了异质域持续学习中的"域引力"（Domain Gravity）偏差——数据丰富或低熵域在共享嵌入空间中产生不成比例的影响，并提出 HyCal，一种无训练方法，通过融合余弦相似度和马氏距离进行原型校准，在跨学科不平衡少样本增量学习中实现稳健分类。

**[Learning Like Humans: Analogical Concept Learning for Generalized Category Discovery](learning_like_humans_analogical_concept_learning_for_generalized_category_discov.md)**

:   提出 AL-GCD 框架，通过模拟人类类比推理机制设计"类比文本概念生成器"（ATCG）——从已知类别的视觉-文本知识库中类比生成未知样本的文本概念，将类别发现转化为视觉-文本联合推理任务，在六个基准上平均提升 5.0%，细粒度数据集提升 7.1%。

**[MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)**

:   提出 Mine-JEPA，首个面向侧扫声纳（SSS）水雷分类的域内自监督学习流水线——基于 SIGReg 正则化损失、声纳适配增强策略和 ImageNet 初始化，仅用 1,170 张未标注声纳图像预训练即超越了在 17 亿图像上预训练的 DINOv3 基础模型。

**[MOMO: Mars Orbital Model — Foundation Model for Mars Orbital Applications](momo_mars_orbital_model_foundation_model_for_mars_orbital_applications.md)**

:   MOMO 是首个火星遥感基础模型，通过在三种火星传感器（HiRISE/CTX/THEMIS）上分别预训练 MAE 并提出 Equal Validation Loss（EVL）检查点选择策略进行模型融合，在 Mars-Bench 的 9 个下游任务上超越 ImageNet 预训练和地球观测基础模型。

**[OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism](omnigcd_abstracting_generalized_category_discovery_for_modality_agnosticism.md)**

:   提出 OmniGCD，首个模态无关的广义类别发现方法，利用合成数据训练的 GCDformer 在测试时将任意模态的 GCD 潜空间变换为更适合聚类的表示，在 16 个跨四种模态的数据集上实现零样本 GCD。

**[Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting](redepth_anything_test-time_depth_refinement_via_self-supervised_re-lighting.md)**

:   提出 Re-Depth Anything，通过在推理时对预测深度图进行重光照增强并利用 2D 扩散模型的 SDS 损失进行自监督优化，在无标签的情况下精细化 Depth Anything V2/3 的深度预测。

**[Reframing Long-Tailed Learning via Loss Landscape Geometry](reframing_long-tailed_learning_via_loss_landscape_geometry.md)**

:   从损失景观几何的角度重新审视长尾学习中的head-tail seesaw困境，发现尾类退化的根源是优化收敛到尖锐且远离尾类最优点的区域，提出基于持续学习思想的GKP（分组知识保存）和GSA（分组锐度感知）双模块框架，无需额外数据即在CIFAR-LT/ImageNet-LT/iNat2018四个基准上取得SOTA。

**[Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physica.md)**

:   在三个 PDE 物理系统（活性物质、剪切流、Rayleigh-Bénard 对流）上系统比较四种自监督/物理建模方法，发现隐空间预测（JEPA）在物理参数估计任务上全面优于像素级预测（VideoMAE）——MSE 相对改善 28%~51%，且 10% 微调数据即可超越 VideoMAE 的 100% 数据表现。同时，专为物理建模设计的方法并非总是最优选择。

**[Robustness of Vision Foundation Models to Common Perturbations](robustness_of_vision_foundation_models_to_common_perturbations.md)**

:   首次系统研究视觉基础模型对常见扰动（JPEG 压缩、亮度调节等）的鲁棒性，提出三种鲁棒性度量并形式化五个数学性质，发现基础模型普遍不鲁棒，并提出微调方法改善鲁棒性而不牺牲效用。

**[Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild](shape-of-you_fused_gromov-wasserstein_optimal_transport_for_semantic_corresponde.md)**

:   将语义对应问题重新建模为 Fused Gromov-Wasserstein (FGW) 最优传输问题，利用 3D 基础模型提供的几何结构约束来生成全局一致的伪标签，解决了传统最近邻匹配因局部性和 2D 外观歧义导致的几何不一致问题。

**[SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks](sphor_a_representation_learning_perspective_on_open-set_recognition_for_identify.md)**

:   提出 SpHOR，一种两阶段解耦训练的开放集识别方法，通过球面表示学习（vMF 分布）、正交标签嵌入和 Mixup/Label Smoothing 集成，显式塑造特征空间以更好地分离已知/未知类别，在 Semantic Shift Benchmark 上取得最高 5.1% 的 OSCR 提升。

**[Suppressing Non-Semantic Noise in Masked Image Modeling Representations](suppressing_non-semantic_noise_in_masked_image_modeling_representations.md)**

:   本文揭示了掩码图像建模（MIM）学到的表征中保留了大量非语义信息（如纹理、颜色等底层特征），并提出了一种无需训练的后处理方法 SOAP（Semantically Orthogonal Artifact Projection），通过 PCA 识别并投影去除非语义成分，在多种 MIM 模型上一致提升零样本性能。

**[TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)**

:   提出 TALO，一种基于 Thin Plate Spline 的高自由度对齐框架，通过全局传播控制点和点无关的子图配准设计，纠正 3D 视觉基础模型在在线重建中的空间变化不一致性，兼容多种基础模型和相机配置，在 Waymo/nuScenes 数据集上显著降低轨迹误差。

**[TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation](teflow_enabling_multi-frame_supervision_for_self-supervised_feed-forward_scene_f.md)**

:   提出TeFlow——首个将多帧监督引入自监督前馈场景流估计的方法：通过时序集成策略构建运动候选池并基于共识投票聚合时序一致的监督信号，在Argoverse 2上Three-way EPE达3.57cm（媲美优化方法Floxels）同时保持实时推理（8s vs 24min），较SeFlow++提升22.3%。

**[Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning](temporal_imbalance_of_positive_and_negative_supervision_in_class-incremental_lea.md)**

:   提出时序不平衡（Temporal Imbalance）这一被忽视的类增量学习偏差来源，并设计 Temporal-Adjusted Loss（TAL）通过时间衰减记忆核动态降低旧类的负监督权重，以即插即用的方式显著缓解灾难性遗忘。

**[Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)**

:   提出TPSNet，将CLIP学习的域提示（domain prompt）作为文本先验提供精细语义监督，同时引入相位谱特征作为相位先验来桥接域分布差异并保持语义完整性，通过文本-相位双先验的协同实现无监督跨域图像检索的显著提升。

**[TrackMAE: Video Representation Learning via Track, Mask, and Predict](trackmae_video_representation_learning_via_track_mask_and_predict.md)**

:   在masked video modeling（MVM）框架中引入显式的运动信号：使用CoTracker3提取点轨迹作为额外的重建目标，并设计运动感知遮掩策略，联合学习空间重建和运动预测，在运动敏感基准（SSv2、FineGym）上显著超越现有视频自监督方法。

**[UniGeoCLIP: Unified Geospatial Contrastive Learning](unigeoclip_geospatial_contrastive.md)**

:   UniGeoCLIP 首次通过纯对比学习将五种互补的地理空间模态（航拍图、街景图、数字表面模型、文本、GPS 坐标）对齐到统一嵌入空间，并提出多尺度坐标编码器提升空间表示能力。

**[Vision Transformers Need More Than Registers](vision_transformers_need_more_than_registers.md)**

:   这篇论文认为 ViT 在标签监督、文本监督和自监督下普遍存在的 dense feature 伪影，本质上不是单纯的 high-norm token 问题，而是模型在粗粒度监督和全局注意力共同作用下学会了用背景 patch 充当全局语义捷径；作者据此提出 LaSt-ViT，用频域稳定性引导的选择性聚合替代原始 CLS 聚合，在 12 个基准上稳定改善定位、分割和开放词汇任务。

**[VT-Intrinsic: Physics-Based Decomposition of Reflectance and Shading using a Single Visible-Thermal Image Pair](vt-intrinsic_physics-based_decomposition_of_reflectance_and_shading_using_a_sing.md)**

:   VT-Intrinsic 利用可见光和热红外图像之间的物理互补关系（未反射的光被吸收变为热量），推导出可见光-热成像强度的序数关系（ordinality）直接对应反射率和光照的序数关系，以此为自监督信号驱动神经网络优化，实现了无需预训练数据的高质量内在图像分解。

**[Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers](zero_ablation_overstates_register_content_dependence_in_dino_vision_transformers.md)**

:   通过三种替换控制实验（均值替换、噪声替换、跨图像洗牌）证明 DINO 系列 ViT 中零消融方法夸大了对 register token 精确内容的依赖性——模型实际只需"合理的 register-like 激活"而非图像特定值。
