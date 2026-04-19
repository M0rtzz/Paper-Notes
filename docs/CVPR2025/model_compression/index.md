---
title: >-
  CVPR2025 模型压缩方向 46篇论文解读
description: >-
  46篇CVPR2025 模型压缩方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📦 模型压缩

**📷 CVPR2025** · **46** 篇论文解读

**[Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning](adapter_merging_with_centroid_prototype_mapping_for_scalable_class-incremental_l.md)**

:   提出ACMap框架，通过将每个任务独立训练的adapter增量平均合并为单一adapter（保持O(1)推理复杂度），结合centroid prototype mapping对齐旧任务原型在新子空间中的表示，在5个基准上实现与SOTA EASE相当的精度同时推理速度快39倍。

**[Alternating Gradient Flow Utility: A Unified Metric for Structural Pruning and Dynamic Routing in Deep Networks](alternating_gradient_flow_utility_a_unified_metric_for_structural_pruning_and_dy.md)**

:   提出基于交替梯度流(AGF)的统一效用度量，将特征空间总变差作为结构化剪枝指标，并结合置信度级联路由实现离线拓扑构建与在线动态推理的解耦，在ImageNet-1K极端压缩下避免传统指标导致的结构崩溃，在ImageNet-100动态推理中以0.92x计算代价匹配全模型精度。

**[An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS](an_fpga_implementation_of_displacement_vector_search_for_intra_pattern_copy_in_j.md)**

:   首次提出JPEG XS帧内模式复制(IPC)中位移向量(DV)搜索模块的FPGA架构实现，采用四级流水线设计和优化的存储组织方式，在Xilinx Artix-7上实现38.3 Mpixels/s吞吐量和277 mW功耗，为IPC实际硬件部署和ASIC转化奠定基础。

**[ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation](arche_autoregressive_residual_compression_with_hyperprior_and_excitation.md)**

:   提出ARCHE端到端学习型图像压缩框架，在统一概率架构中整合分层Hyperprior、掩码空间自回归上下文、通道条件化和SE激励通道重校准，无需Transformer或循环组件，在Kodak上相对Ballé基线BD-Rate降低约48%，相对VVC Intra降低约5.6%，仅95M参数和222ms解码时间。

**[BHViT: Binarized Hybrid Vision Transformer](bhvit_binarized_hybrid_vision_transformer.md)**

:   针对 ViT 二值化性能严重下降的问题，提出专为二值化设计的混合 ViT 架构 BHViT，包含多尺度分组空洞卷积 token mixer、量化分解注意力矩阵二值化、shift 增强的 MLP 和正则化损失，在 ImageNet-1K 上达到 1-bit 二值化模型的 SOTA 性能。

**[Charm: The Missing Piece in ViT Fine-Tuning for Image Aesthetic Assessment](charm_the_missing_piece_in_vit_fine-tuning_for_image_aesthetic_assessment.md)**

**[CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)**

:   提出 CL-LoRA，设计双适配器架构（任务共享 + 任务特定 LoRA），结合知识蒸馏与梯度重分配以及可学习块级权重，在仅 0.3% 可训练参数下实现 SOTA 持续学习性能。

**[CoA: Towards Real Image Dehazing via Compression-and-Adaptation](coa_towards_real_image_dehazing_via_compression-and-adaptation.md)**

:   提出压缩-适应（CoA）框架实现实际图像去雾：先在合成数据上训练大模型，然后压缩+适应到真实域，平衡性能和部署效率

**[Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation](curriculum_coarse-to-fine_selection_for_high-ipc_dataset_distillation.md)**

:   提出CCFS方法，通过课程学习框架渐进式地从原始数据集中选择合适的真实样本补充蒸馏数据，解决高IPC场景下蒸馏数据与真实数据的不兼容问题，在CIFAR-10/100和Tiny-ImageNet上大幅超越SOTA（最高+6.6%）。

**[Dataset Distillation with Neural Characteristic Function: A Minmax Perspective](dataset_distillation_with_neural_characteristic_function_a_minmax_perspective.md)**

:   提出NCFM方法，通过在复平面上用神经网络参数化的特征函数差异（NCFD）作为分布距离度量，将数据集蒸馏重构为minmax对抗优化问题，同时对齐相位（真实性）和幅值（多样性）信息，在ImageNet子集上最高提升20.5%，且GPU内存降低300倍以上。

**[DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation](delt_a_simple_diversity-driven_earlylate_training_for_dataset_distillation.md)**

:   提出EarlyLate训练策略，通过让不同IPC子批次从不同优化起点开始、经历不同迭代次数来生成难度各异的合成图像，在batch-to-global匹配框架下显著提升类内多样性，同时减少39.3%计算时间，在ImageNet-1K上以IPC=50达到66.1%（ResNet-101，超越RDED 4.9%）。

**[DeRS: Towards Extremely Efficient Upcycled Mixture-of-Experts Models](ders_towards_extremely_efficient_upcycled_mixture-of-experts_models.md)**

:   提出DeRS（Decompose-Replace-Synthesis）范式，利用upcycled MoE专家间的极高相似性（余弦相似度>0.999），将N个专家分解为1个共享基础权重+N个轻量delta权重，通过稀疏化/量化/低秩表示压缩delta权重，在MoE层参数减少65%的同时性能不降，或训练时额外参数减少2270倍。

**[Distilling Long-tailed Datasets](distilling_long-tailed_datasets.md)**

:   首次系统研究长尾数据集蒸馏问题，发现现有方法在长尾场景下严重退化（甚至不如随机选择），提出Distribution-agnostic Matching（DAM）和Expert Decoupling（ED）两个策略，在CIFAR-10/100-LT和Tiny-ImageNet-LT上大幅超越现有方法（如在imbalance factor=100时超越DATM 19.7%）。

**[DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models](dycoke_dynamic_compression_of_tokens_for_fast_video_large_language_models.md)**

:   提出DyCoke，一种免训练的动态视觉Token压缩方法，通过两阶段策略——时序Token合并（消除跨帧冗余50-60%）和KV Cache动态剪枝（在每个解码步动态保留最相关的token，进一步减少70-90%），将视频LLM的每帧平均token数降至15个，实现1.5倍加速且性能不降反微升。

**[ECVC: Exploiting Non-Local Correlations in Multiple Frames for Contextual Video Compression](ecvc_exploiting_non-local_correlations_in_multiple_frames_for_contextual_video_c.md)**

:   提出ECVC视频压缩模型，通过多帧非局部上下文挖掘（MNLC）和多头线性交叉注意力（MHLCA）捕获多参考帧间的非局部相关性，结合部分级联微调策略（PCFS）解决训练-测试序列长度不匹配问题，在IP=32和IP=-1设置下分别比DCVC-FM节省10.5%和11.5%码率。

**[EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality](efficientvim_efficient_vision_mamba_with_hidden_state_mixer_based_state_space_du.md)**

:   提出EfficientViM，通过将SSD层中的通道混合操作从token空间（$O(LD^2)$）迁移到压缩的隐藏状态空间（$O(ND^2)$，$N \ll L$），实现了比现有Vision Mamba模型快2-4倍的推理速度，同时保持竞争性精度（ImageNet-1K上M3模型77.9%/11952 img/s）。

**[Embracing Collaboration Over Competition: Condensing Multiple Prompts for Visual In-Context Learning](embracing_collaboration_over_competition_condensing_multiple_prompts_for_visual_.md)**

:   提出 Condenser 将多个 Visual ICL 的 prompt 候选通过 Patch-wise 跨注意力凝聚为单一 prompt，实现多 prompt 协作而非竞争选择，在分割/检测/上色等任务上以 16 个 prompt 输入达到 46.63 mIoU（vs 单 prompt 44.14），推理速度比逐一评估快 15×。

**[Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios](emphasizing_discriminative_features_for_dataset_distillation_in_complex_scenario.md)**

:   提出EDF方法，通过Common Pattern Dropout（丢弃轨迹匹配中低损失的通用模式参数梯度）和Discriminative Area Enhancement（用Grad-CAM加权放大判别性区域的梯度），解决数据集蒸馏在复杂场景（ImageNet子集）上的性能退化问题，在ImageMeow/ImageYellow等数据集上仅用23%数据实现无损压缩。

**[Enhancing Dataset Distillation via Non-Critical Region Refinement](enhancing_dataset_distillation_via_non-critical_region_refinement.md)**

:   提出NRR-DD三阶段框架：用CAM选低置信度patch初始化合成图像、固定关键区域仅优化非关键区域提升信息密度、用2个距离值替代1000维软标签实现500倍存储压缩。在ImageNet-1K上IPC=10时达到46.1%（超RDED 25.7%），软标签存储从120GB降至0.2GB。

**[Exploration-Driven Generative Interactive Environments](exploration-driven_generative_interactive_environments.md)**

:   开源实现 Genie 世界模型（GenieRedux），增加真实动作条件、Token 距离交叉熵（TDCE）损失和 token 跳连得到 GenieRedux-G，并提出 AutoExplore 探索智能体用世界模型的 token 预测不确定性作为内在奖励驱动多样数据收集，将仿真质量提升高达 7.4 PSNR。

**[Exploring Contextual Attribute Density in Referring Expression Counting (CAD-GD)](exploring_contextual_attribute_density_in_referring_expression_counting.md)**

:   提出上下文属性密度（Contextual Attribute Density, CAD）概念来增强指代表达计数（Referring Expression Counting），通过 U 形密度估计器、CAD 注意力和动态查询初始化三个模块，在 REC-8K 数据集上相比 GroundingREC 降低了约 30% 的计数误差（MAE 从 6.80 降至 5.43）。

**[Faster Parameter-Efficient Tuning with Token Redundancy Reduction (FPET)](faster_parameter-efficient_tuning_with_token_redundancy_reduction.md)**

:   提出 FPET（Faster Parameter-Efficient Tuning），在参数高效微调（PET）中引入即插即用的 token 冗余压缩模块——在 ViT 中间层用可微的二分匹配策略合并约一半的 token，实现比原始 backbone 更快 20% 的推理速度、减少约 40% GPU显存、且精度与 SOTA PET 方法持平。

**[FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](fima-q_post-training_quantization_for_vision_transformers_by_fisher_information_.md)**

:   提出 FIMA-Q，通过对角+低秩（DPLR）的 Fisher 信息矩阵近似替代传统对角近似，更准确地捕捉量化误差对输出分布的影响，在 3-bit 极低比特 ViT 量化中大幅超越现有方法（ViT-B 77.63% vs QDrop 74.75%）。

**[Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders](gaze-lle_gaze_target_estimation_via_large-scale_learned_encoders.md)**

:   提出 Gaze-LLE，一个基于冻结 DINOv2 编码器的极简视线目标估计框架——仅用 ~2.8M 可训练参数（比先前方法少 1-2 个数量级）、无需辅助深度/姿态模型、无需独立头部编码器，通过人物位置提示 + 轻量 transformer 解码器即在 GazeFollow/VideoAttentionTarget 等基准上达到 SOTA（AUC 0.958）。

**[GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](geochemad_benchmarking_unsupervised_geochemical_anomaly_detection_for_mineral_ex.md)**

:   提出 GeoChemAD 开源基准数据集（8 个子集，覆盖多区域/多采样源/多目标元素）和 GeoChemFormer 框架，通过空间上下文自监督预训练和元素依赖建模实现无监督地球化学异常检测，在所有子集上取得最优 AUC。

**[HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)**

:   HiAP 提出了一种多粒度自动剪枝框架，通过在宏观（attention heads、FFN blocks）和微观（intra-head dimensions、FFN neurons）两级部署可学习 Gumbel-Sigmoid 门控，在单阶段端到端训练中自动发现最优子网络，无需手工重要性排序或后处理阈值。

**[HOT: Hadamard-based Optimized Training](hot_hadamard-based_optimized_training.md)**

:   提出HOT方法，通过对反向传播中不同梯度路径（激活梯度$g_x$和权重梯度$g_m$）的差异化灵敏度分析，选择性地应用Hadamard变换+量化——$g_x$用HT+INT4加速计算、$g_m$用HLA+INT8节省激活内存，实现75%激活内存节省和2.6倍GPU加速，ViT-B在ImageNet上精度仅降0.17%。

**[HyperLoRA: Parameter-Efficient Adaptive Generation for Portrait Synthesis](hyperlora_parameter-efficient_adaptive_generation_for_portrait_synthesis.md)**

:   提出 HyperLoRA，一种通过自适应网络直接生成 LoRA 权重的零样本个性化肖像生成方法——将 LoRA 参数投影到低维线性空间（原参数的 1.2%），用 perceiver resampler 从输入人脸预测组合系数，并将 LoRA 显式分解为 ID-LoRA 和 Base-LoRA 以解耦身份与无关信息，实现高保真度+高可编辑性+快速推理的平衡。

**[Incremental Object Keypoint Learning (KAMP)](incremental_object_keypoint_learning.md)**

:   首次定义增量关键点学习（IKL）范式——新任务只标注新关键点、不保留旧数据的增量训练，提出 KAMP 框架通过知识关联网络（KA-Net）建模新旧关键点间的解剖学空间关系，配合关键点导向的空间蒸馏损失，在 4 个数据集上不仅有效防遗忘，甚至实现了对旧关键点的正向迁移提升（MPII AAA 79.93% vs LWF 75.75%）。

**[InsTaG: Learning Personalized 3D Talking Head from Few-Second Video](instag_learning_personalized_3d_talking_head_from_few-second_video.md)**

:   提出 InsTaG，通过 Identity-Free Pre-training 从多人长视频中提取通用运动先验，再通过 Motion-Aligned Adaptation 仅用 5 秒视频即可快速学习高保真个性化 3D 说话人头像，实现 82.5 FPS 实时推理。

**[JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba](jamma_ultra-lightweight_local_feature_matching_with_joint_mamba.md)**

:   JamMa提出了基于Joint Mamba的超轻量级半密集特征匹配器，通过JEGO扫描-合并策略实现跨视角联合扫描、高效四方向扫描、全局感受野和全方向特征表示，以不到50%的参数和FLOPs实现了优于Transformer-based匹配器的性能-效率平衡。

**[Layered Image Vectorization via Semantic Simplification](layered_image_vectorization_via_semantic_simplification.md)**

:   本文提出一种渐进式图像矢量化方法，利用 Score Distillation Sampling（SDS）的特征平均效应生成逐级简化的图像序列，以此引导从宏观语义结构到精细细节的分层矢量重建，在视觉保真度、语义对齐和紧凑分层表示上显著优于现有方法。

**[Learned Image Compression with Dictionary-based Entropy Model](learned_image_compression_with_dictionary-based_entropy_model.md)**

:   提出基于字典的交叉注意力熵模型 (DCAE)，引入可学习字典从训练数据集中提取自然图像的典型纹理结构先验，通过多尺度特征聚合 + 交叉注意力实现精确的概率分布估计，在编解码速度仅 193ms 的条件下实现 -17.0%/-21.1%/-19.7% 的 BD-rate（Kodak/Tecnick/CLIC），全面超越 SOTA。

**[PrunNet: Learning Compatible Multi-Prize Subnetworks for Asymmetric Retrieval](learning_compatible_multi-prize_subnetworks_for_asymmetric_retrieval.md)**

:   提出 PrunNet（可剪枝网络），通过为每个权重学习重要性分数并结合冲突感知梯度集成，训练一个可以在任意容量（20%-100%）下产生兼容子网络的统一模型，在 GLDv2 上 46.29 mAP 超越密集网络基线，且所有容量子网络间特征兼容。

**[LALIC: Linear Attention Modeling for Learned Image Compression](linear_attention_modeling_for_learned_image_compression.md)**

:   首次将 RWKV 线性注意力机制引入学习图像压缩，设计 Bi-RWKV 变换块实现线性复杂度的全局感受野特征提取，配合 RWKV 时空通道上下文熵模型，以较低复杂度超越 VTM-9.1 达 15.26% BD-rate。

**[Logits DeConfusion with CLIP for Few-Shot Learning](logits_deconfusion_with_clip_for_few-shot_learning.md)**

:   发现 CLIP 在下游任务中 logits 存在严重的类间混淆问题，提出 Logits DeConfusion（LDC）方法，通过多层级 Adapter 融合（MAF）增强特征表示，结合类间去混淆模块（ICD）以残差结构学习并消除混淆模式，在 11 个基准上取得 SOTA。

**[LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning](lora_subtraction_for_drift-resistant_space_in_exemplar-free_continual_learning.md)**

:   LoRA-DRS 提出"LoRA 减法"操作——在学习新任务前将旧任务的 LoRA 权重从预训练权重中减去以构建漂移抵抗空间（DRS），然后在该空间中通过梯度投影训练新任务的 LoRA，结合增强三元组损失提升可塑性，在无样本持续学习中实现了 SOTA 性能，尤其在长任务序列上优势显著。

**[LSNet: See Large, Focus Small](lsnet_see_large_focus_small.md)**

:   受人类视觉外周（广域感知）-中央（精细聚合）的双尺度机制启发，提出 LS 卷积（大核深度卷积感知 + 小核动态卷积聚合），构建 LSNet 轻量网络家族，在 0.3~1.3G FLOPs 下全面超越现有 SOTA 轻量模型。

**[Mamba-Adaptor: State Space Model Adaptor for Visual Recognition](mamba-adaptor_state_space_model_adaptor_for_visual_recognition.md)**

:   提出 Mamba-Adaptor，通过两个模块增强 Vision Mamba/SSM：Adaptor-T（时序）用可学习记忆选择机制保留关键历史状态，Adaptor-S（空间）用多尺度空心深度卷积增强空间局部性，在 ImageNet 上 83.0% Top-1（Mamba-Adaptor-b2），检测/分割+迁移学习全面提升。

**[MambaIC: State Space Models for High-Performance Learned Image Compression](mambaic_state_space_models_for_high-performance_learned_image_compression.md)**

:   首次将 SSM 同时整合到学习型图像压缩的非线性变换和上下文模型中，通过 VSS block 增强通道-空间上下文建模 + 窗口局部注意力消除空间冗余，在 Kodak 上比 VVC 节省 12.52% BD-rate，且高分辨率图像压缩优势更加显著。

**[Masking Meets Supervision: A Strong Learning Alliance](masking_meets_supervision_a_strong_learning_alliance.md)**

:   提出 Masked Sub-branch (MaskSub)——在监督学习中引入高比例 (50%) mask 增强的通用框架，通过主分支(无mask)和子分支(有mask)的自蒸馏结构解决强 mask 增强导致训练不稳定的问题，在 DeiT-III、MAE 微调、CLIP 微调、BERT 训练以及 ResNet/Swin 等多种场景中均取得一致性能提升。

**[MobileMamba: Lightweight Multi-Receptive Visual Mamba Network](mobilemamba_lightweight_multi-receptive_visual_mamba_network.md)**

:   提出 MobileMamba 轻量级视觉网络，通过三阶段粗粒度架构设计和 MRFFI 细粒度模块（融合 Mamba 全局建模、多核卷积多尺度感知和 Identity 冗余消除），在分类和下游高分辨率任务上实现速度与精度的最优平衡。

**[MuTri: Multi-view Tri-alignment for OCT to OCTA 3D Image Translation](mutri_multi-view_tri-alignment_for_oct_to_octa_3d_image_translation.md)**

:   本文提出MuTri，首次将向量量化（VQ）引入OCT到OCTA的3D体积翻译任务，通过两阶段训练——先预训练OCT和OCTA重建VQVAE提供多视图先验，再用对比语义对齐（3D OCT/OCTA视图）和血管结构对齐（2D OCTA投影图视图）三视图指导翻译VQVAE的码本学习，在三个数据集上全面超越SOTA。

**[NADER: Neural Architecture Design via Multi-Agent Collaboration](nader_neural_architecture_design_via_multi-agent_collaboration.md)**

:   NADER 将神经架构设计建模为多 LLM Agent 协作任务——Reader 读论文提炼知识、Proposer 生成改进方案、Modifier 用 DAG 图实现修改、Reflector 从失败中学习经验，仅 10 次试验即突破 NAS-Bench-201 搜索空间的准确率上限，在 CIFAR-100 上达 74.51%（搜索空间最优 73.51%）。

**[Targeted Forgetting of Image Subgroups in CLIP Models](targeted_forgetting_of_image_subgroups_in_clip_models.md)**

:   提出三阶段 CLIP 模型子群遗忘方法：遗忘阶段用相对 Fisher 信息定位关键层+LoRA 微调遗忘目标类；提醒阶段用分布对齐在保留集上恢复；恢复阶段用模型汤（model souping）恢复零样本能力。在 ImageNet-1K 上遗忘分数达 91.0（基线 68.9）。

**[Understanding Multi-Layered Transmission Matrices](understanding_multi-layered_transmission_matrices.md)**

:   分析体散射介质的多层传输矩阵近似，发现"缺失锥"（光学孔径导致的频率域约束）使所需 SLM 层数远少于 Nyquist 采样预测——200μm 组织理论需要 100 层但实际只需 3-4 层即可实现良好聚焦，视场从 1×1μm 扩展到 13×13μm。
