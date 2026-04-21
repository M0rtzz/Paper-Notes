---
title: >-
  CVPR2026 自监督方向 19篇论文解读
description: >-
  19篇CVPR2026 自监督方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督

**📷 CVPR2026** · **19** 篇论文解读

**[BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning](bd-merging_bias-aware_dynamic_model_merging_with_evidence-guided_contrastive_lea.md)**

:   提出 BD-Merging 框架，通过 Dirichlet 证据建模 + 邻域差异分数（ADS）+ 差异感知对比学习，训练去偏路由器来自适应分配模型合并权重，显著提升合并模型在测试时分布偏移和未见任务上的鲁棒性与泛化能力。

**[BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)**

:   提出 BoSS——一种可扩展的 oracle 策略选择框架：在每轮主动学习中，并行运行多种查询策略在随机子池上生成候选 batch，通过冻结 backbone 仅重训最后一层快速评估每个候选 batch 的性能增益，选出最优 batch，从而量化现有 AL 策略与理论最优之间的差距。

**[BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](boss_a_bestofstrategies_selector_as_an_oracle_for.md)**

:   提出 BoSS（Best-of-Strategies Selector），通过集成10种互补的AL选择策略生成100个候选批次，冻结预训练backbone仅重训最后线性层来高效评估每个批次的性能增益，选取最优批次作为Oracle上界参考——首个可扩展到ImageNet的深度主动学习Oracle策略，揭示当前SOTA策略在大规模多类数据集上仍有约2倍的准确率提升空间。

**[D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping](d2dewarp_dual_dimensions_geometric_representation_learning_based_document_image_.md)**

:   提出 D2Dewarp——首个从水平和垂直双维度学习文档几何表示的去畸变方法：UNet 双解码器分别预测水平线（文档/表格/文本行的上下边界）和垂直线（左右边界），HV Fusion Module 通过混合注意力交叉融合两个方向的特征，并构建了包含 114K 张图的 DocDewarpHV 数据集提供双维度标注。

**[DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)**

:   通过系统分析发现 DiT 各 block 间的表示多样性是有效学习的关键因素，提出 DiverseDiT：用长残差连接多样化输入 + 表示多样性损失显式促进 block 间特征差异化，无需外部引导模型即可加速收敛并提升生成质量。

**[LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency](las-comp_zero-shot_3d_completion_with_latent-spatial_consistency.md)**

:   提出 LaS-Comp，一种零样本、类别无关的 3D 形状补全框架，通过 Explicit Replacement Stage 在空间域注入已知几何 + Implicit Alignment Stage 在隐空间梯度优化边界一致性，桥接了预训练 3D 基础模型的隐空间与空间域之间的 gap，在多种部分观测模式下达到 SOTA。

**[MOMO: Mars Orbital Model — Foundation Model for Mars Orbital Applications](momo_mars_orbital_model_foundation_model_for_mars_orbital_applications.md)**

:   MOMO 是首个火星遥感基础模型，通过在三种火星传感器（HiRISE/CTX/THEMIS）上分别预训练 MAE 并提出 Equal Validation Loss（EVL）检查点选择策略进行模型融合，在 Mars-Bench 的 9 个下游任务上超越 ImageNet 预训练和地球观测基础模型。

**[Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting](redepth_anything_test-time_depth_refinement_via_self-supervised_re-lighting.md)**

:   提出 Re-Depth Anything，通过在推理时对预测深度图进行重光照增强并利用 2D 扩散模型的 SDS 损失进行自监督优化，在无标签的情况下精细化 Depth Anything V2/3 的深度预测。

**[Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physica.md)**

:   在三个 PDE 物理系统（活性物质、剪切流、Rayleigh-Bénard 对流）上系统比较四种自监督/物理建模方法，发现隐空间预测（JEPA）在物理参数估计任务上全面优于像素级预测（VideoMAE）——MSE 相对改善 28%~51%，且 10% 微调数据即可超越 VideoMAE 的 100% 数据表现。同时，专为物理建模设计的方法并非总是最优选择。

**[Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physical_systems.md)**

:   在三个 PDE 物理系统上系统对比 JEPA、VideoMAE、自回归基础模型(MPP)和算子学习(DISCO) 四种范式，发现隐空间预测目标(JEPA)在物理参数估计下游任务上全面优于像素级预测方法，MSE 相对改善 28-51%，且数据效率更高。

**[SpHOR: A Representation Learning Perspective on Open-set Recognition](sphor_a_representation_learning_perspective_on_ope.md)**

:   提出SpHOR两阶段解耦训练框架：Stage 1通过正交标签嵌入+球面约束（vMF分布）+Mixup/Label Smoothing做专为OSR设计的表征学习，Stage 2冻结特征训练分类器——在Semantic Shift Benchmark上OSCR/AUROC最高提升5.1%/5.2%，同时引入Angular Separability和Norm Separability两个新度量。

**[SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks](sphor_a_representation_learning_perspective_on_open-set_recognition_for_identify.md)**

:   提出 SpHOR，一种两阶段解耦训练的开放集识别方法，通过球面表示学习（vMF 分布）、正交标签嵌入和 Mixup/Label Smoothing 集成，显式塑造特征空间以更好地分离已知/未知类别，在 Semantic Shift Benchmark 上取得最高 5.1% 的 OSCR 提升。

**[Suppressing Non-Semantic Noise in Masked Image Modeling Representations](suppressing_non-semantic_noise_in_masked_image_modeling_representations.md)**

:   本文揭示了掩码图像建模（MIM）学到的表征中保留了大量非语义信息（如纹理、颜色等底层特征），并提出了一种无需训练的后处理方法 SOAP（Semantically Orthogonal Artifact Projection），通过 PCA 识别并投影去除非语义成分，在多种 MIM 模型上一致提升零样本性能。

**[TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)**

:   提出 TALO，一种基于 Thin Plate Spline 的高自由度对齐框架，通过全局传播控制点和点无关的子图配准设计，纠正 3D 视觉基础模型在在线重建中的空间变化不一致性，兼容多种基础模型和相机配置，在 Waymo/nuScenes 数据集上显著降低轨迹误差。

**[TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation](teflow_enabling_multi-frame_supervision_for_self-supervised_feed-forward_scene_f.md)**

:   提出TeFlow——首个将多帧监督引入自监督前馈场景流估计的方法：通过时序集成策略构建运动候选池并基于共识投票聚合时序一致的监督信号，在Argoverse 2上Three-way EPE达3.57cm（媲美优化方法Floxels）同时保持实时推理（8s vs 24min），较SeFlow++提升22.3%。

**[Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)**

:   提出TPSNet，将CLIP学习的域提示（domain prompt）作为文本先验提供精细语义监督，同时引入相位谱特征作为相位先验来桥接域分布差异并保持语义完整性，通过文本-相位双先验的协同实现无监督跨域图像检索的显著提升。

**[TrackMAE: Video Representation Learning via Track, Mask, and Predict](trackmae_video_representation_learning_via_track_mask_and_predict.md)**

:   在masked video modeling（MVM）框架中引入显式的运动信号：使用CoTracker3提取点轨迹作为额外的重建目标，并设计运动感知遮掩策略，联合学习空间重建和运动预测，在运动敏感基准（SSv2、FineGym）上显著超越现有视频自监督方法。

**[Vision Transformers Need More Than Registers](vision_transformers_need_more_than_registers.md)**

:   这篇论文认为 ViT 在标签监督、文本监督和自监督下普遍存在的 dense feature 伪影，本质上不是单纯的 high-norm token 问题，而是模型在粗粒度监督和全局注意力共同作用下学会了用背景 patch 充当全局语义捷径；作者据此提出 LaSt-ViT，用频域稳定性引导的选择性聚合替代原始 CLS 聚合，在 12 个基准上稳定改善定位、分割和开放词汇任务。

**[Vision Transformers Need More Than Registers](vit_need_more_than_registers.md)**

:   系统分析了 ViT 中广泛存在的 artifact 现象（跨全监督、文本监督、自监督），揭示其根本原因是"lazy aggregation"——ViT 利用语义无关的背景 patch 作为捷径来表示全局语义，提出 LaSt-ViT（LazyStrike ViT）通过频率感知的选择性通道聚合将 CLS token 锚定到前景区域，在 12 个 benchmark 上一致消除 artifact 并提升性能。
