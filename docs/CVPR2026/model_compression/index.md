---
title: >-
  CVPR2026 模型压缩方向 43篇论文解读
description: >-
  43篇CVPR2026 模型压缩方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📦 模型压缩

**📷 CVPR2026** · **43** 篇论文解读

**[A Paradigm Shift: Fully End-to-End Training for Temporal Sentence Grounding in Videos](a_paradigm_shift_fully_end-to-end_training_for_temporal_sentence_grounding_in_vi.md)**

:   提出首个完全端到端的时序语句定位(TSGV)框架，通过语句条件适配器(SCADA)将语句嵌入注入视频backbone的中间层来动态调制视觉特征，配合视频中心学习策略加速训练，在Charades-STA和ActivityNet上超越SOTA。

**[An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS](an_fpga_implementation_of_displacement_vector_sear.md)**

:   本文首次为 JPEG XS 标准中的 Intra Pattern Copy (IPC) 工具设计了 FPGA 硬件加速架构，通过四级流水线 DV 比较引擎和按 IPC Group 对齐的存储组织，在 Artix-7 上实现 38.3 Mpixels/s 吞吐量和 277mW 功耗。

**[An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS](an_fpga_implementation_of_displacement_vector_search_for_intra_pattern_copy_in_j.md)**

:   针对 JPEG XS 屏幕内容编码中 Intra Pattern Copy（IPC）模块的位移向量（DV）搜索计算瓶颈，首次提出四级流水线 FPGA 架构并设计基于 IPC Group 对齐的内存组织方式，在 Xilinx Artix-7 上实现 38.3 Mpixels/s 吞吐量和 277 mW 功耗，为 IPC 的实际硬件部署提供了可行方案。

**[ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation](arche_autoregressive_residual_compression_with_hyp.md)**

:   在全卷积架构内统一层级超先验、Masked PixelCNN 空间自回归、通道条件建模和 SE 通道激励，不依赖 Transformer 或循环组件，以 95M 参数和 222ms 解码时间实现相对 Ballé 基线 48% BD-Rate 降低并超越 VVC Intra 5.6%。

**[ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation](arche_autoregressive_residual_compression_with_hyperprior_and_excitation.md)**

:   提出 ARCHE 端到端图像压缩框架，在无 Transformer 和循环模块的纯卷积架构下，通过统一层级超先验、Masked PixelCNN 空间自回归上下文、通道条件化、SE 通道重标定和潜在残差预测五个互补组件，在 Kodak 上相对 Balle 基线降低 48% BD-Rate、相对 VVC Intra 降低 5.6%，同时仅需 95M 参数和 222ms 解码时间。

**[Beyond Loss Values: Robust Dynamic Pruning via Loss Trajectory Alignment](beyond_loss_values_robust_dynamic_pruning_via_loss_trajectory_alignment.md)**

:   提出AlignPrune——一个基于损失轨迹对齐的即插即用模块，通过Dynamic Alignment Score（DAS）替代传统损失值排序，使动态数据剪枝在噪声标签场景下准确率提升最高6.3%。

**[BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)**

:   提出 BinaryAttention，将 Transformer 注意力中的 Query 和 Key 量化为 1-bit 二值表示，通过 XNOR + popcount 位运算替代浮点点积，在 A100 上实现比 FlashAttention2 快 2 倍以上的加速，同时在视觉分类/检测/分割/扩散生成等任务上性能持平甚至超越全精度注意力。

**[CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach](cheem_continual_learning_by_reuse_new_adapt_and_skip_--_a_hierarchical_explorati.md)**

:   提出 CHEEM 框架，通过分层探索-利用采样的 NAS 自动学习任务感知的动态 ViT 骨干——在每一层选择 Reuse/New/Adapt/Skip 四种操作——在 MTIL 和 VDD 两个挑战性持续学习基准上显著超越提示类方法，接近全量微调上界。

**[Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](critical_patch-aware_sparse_prompting_with_decoupled_training_for_continual_lear.md)**

:   提出 CPS-Prompt 框架，通过任务感知的关键 patch 采样（CPS）和解耦 prompt-分类器训练（DPCT）两个模块，在边缘设备上实现 Prompt-based 持续学习的训练时内存和计算效率提升约 1.6 倍，同时准确率仅下降约 2%。

**[DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry Estimation](dage_dual-stream_architecture_for_efficient_and_fine-grained_geometry_estimation.md)**

:   提出 DAGE 双流 Transformer 架构，将全局一致性建模（低分辨率流）与细粒度细节保持（高分辨率流）解耦，通过轻量 Cross-Attention Adapter 融合，实现 2K 分辨率和 1000 帧长序列上的高质量深度/点图估计和位姿预测，速度比 Pi3 快 2-28 倍，视频几何估计取得新 SOTA。

**[Distilling Balanced Knowledge from a Biased Teacher](distilling_balanced_knowledge_from_a_biased_teacher.md)**

:   针对长尾分布下知识蒸馏中教师模型向头部类偏斜的问题，将传统 KL 散度损失分解为跨组损失和组内损失两个组件，通过重平衡跨组损失校准教师的组级预测、重加权组内损失保证各组等贡献，在 CIFAR-100-LT/TinyImageNet-LT/ImageNet-LT 上全面超越现有方法，甚至超过教师模型自身表现。

**[FAIR-Pruner: Leveraging Tolerance of Difference for Flexible Automatic Layer-Wise Neural Network Pruning](fair-pruner_leveraging_tolerance_of_difference_for_flexible_automatic_layer-wise.md)**

:   提出 FAIR-Pruner 结构化剪枝框架，通过 Tolerance of Differences（ToD）指标协调两个互补视角：基于类条件可分性的 Wasserstein Utilization Score（识别冗余单元）和基于 Taylor 展开的 Reconstruction Score（保护关键单元），自动确定逐层非均匀剪枝率且支持免搜索灵活调整压缩比，在 CIFAR-10/SVHN/ImageNet 上取得 SOTA。

**[Frequency Switching Mechanism for Parameter-Efficient Multi-Task Learning](frequency_switching_mechanism_for_parameter-ecient_multi-task_learning.md)**

:   Free Sinewich 提出基于频率切换的参数高效多任务学习框架，通过对共享低秩基矩阵施加不同任务特定频率的正弦变换 $M_t = \sin(\omega_t \cdot M_{AWB})$，以接近零成本实现真正的参数复用和任务特化，在密集预测基准上以最少可训练参数达到SOTA。

**[From Fewer Samples to Fewer Bits: Reframing Dataset Distillation as Joint Optimization of Precision and Compactness](from_fewer_samples_to_fewer_bits_reframing_dataset_distillation_as_joint_optimiz.md)**

:   提出 QuADD 框架，将可微量化模块嵌入数据集蒸馏循环中，联合优化合成数据与量化参数，实现在固定比特预算下"更少样本 + 更低精度"的帕累托最优压缩。

**[Generative Video Compression with One-Dimensional Latent Representation](generative_video_compression_with_one-dimensional_latent_representation.md)**

:   提出 GVC1D，首次将视频压缩的潜在表示从2D网格替换为紧凑的1D token序列，结合1D记忆模块建模长期时序上下文，在感知质量指标上实现 60%+ 的码率节省。

**[GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](geochemad_benchmarking_unsupervised_geochemical_an.md)**

:   发布首个开源多区域多元素地球化学异常检测基准 GeoChemAD（8 子集，覆盖沉积物/岩屑/土壤三类采样源和 Au/Cu/Ni/W 四种目标元素），并提出 GeoChemFormer——两阶段 Transformer 框架，先学空间上下文再做元素依赖建模，平均 AUC 达 0.7712 超越所有基线。

**[GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](geochemad_benchmarking_unsupervised_geochemical_anomaly_detection_for_mineral_ex.md)**

:   提出 GeoChemAD 开源基准数据集和 GeoChemFormer 框架，通过空间上下文学习与元素依赖建模实现无监督地球化学异常检测，在8个子集上平均 AUC 达到 0.7712。

**[GeoFusion-CAD: Structure-Aware Diffusion with Geometric State Space for Parametric 3D Design](geofusion-cad_structure-aware_diffusion_with_geometric_state_space_for_parametri.md)**

:   本文提出 GeoFusion-CAD，一个端到端的扩散框架，通过将 CAD 程序编码为层次化树结构并引入几何感知的 G-Mamba 块（线性时间复杂度）替代二次复杂度的 Transformer，实现了对长序列参数化 CAD 程序的可扩展和结构感知生成，在新构建的 DeepCAD-240（最长240步命令）基准上大幅超越 Transformer 方法。

**[HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)**

:   HiAP 把 ViT 剪枝写成一个端到端的预算感知学习问题，同时对整头/整块和头内维度/FFN 神经元两种粒度做随机可微门控，在一次训练里自动长出满足算力预算的稠密子网络，省掉了常见的排序、阈值搜索和额外微调流程。

**[HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](hiap_a_multigranular_stochastic_autopruning_framew.md)**

:   提出HiAP——统一宏观（整头/FFN块）和微观（头内维度/FFN神经元）的层级Gumbel-Sigmoid门控框架，在单次端到端训练中自动发现满足算力预算的高效ViT子网络，无需手动重要性排序或多阶段流程。

**[HierAmp: Coarse-to-Fine Autoregressive Amplification for Generative Dataset Distillation](hieramp_coarse-to-fine_autoregressive_amplification_for_generative_dataset_disti.md)**

:   提出 HierAmp，在视觉自回归（VAR）模型的粗到细生成过程中，向每个尺度注入可学习的类别 token 识别语义显著区域，并通过正 logit 偏置放大这些区域的注意力，使蒸馏数据在粗尺度获得更丰富多样的布局、在细尺度聚焦于类别相关细节，在多个数据集蒸馏基准上达到 SOTA。

**[Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](iapl_aigenerated_image_detection_adaptive_prompt.md)**

:   提出IAPL（Image-Adaptive Prompt Learning），在CLIP编码器输入端引入动态prompt——由条件信息学习器（从纹理丰富区域提取伪造特异和通用线索）和测试时token调优（通过多视角一致性最小化熵）两条路径生成，使模型能在推理时根据每张测试图像自适应调整，在未见过的生成器上显著提升检测泛化性。

**[Learning through Creation: A Hash-Free Framework for On-the-Fly Category Discovery](learning_through_creation_a_hash-free_framework_for_on-the-fly_category_discover.md)**

:   提出 LTC 框架，通过在训练阶段利用 MKEE（最小化核能量+最大化熵）在线生成伪未知类样本，配合双最大间隔损失和自适应阈值，在7个数据集上实现1.5%–13.1%的全类精度提升，彻底摆脱了哈希编码对细粒度语义的损害。

**[Markovian Scale Prediction: A New Era of Visual Autoregressive Generation](markovian_scale_prediction_a_new_era_of_visual_autoregressive_generation.md)**

:   将视觉自回归模型 (VAR) 从全上下文依赖的 next-scale prediction 重构为基于马尔可夫过程的 Markovian scale prediction，通过滑动窗口历史补偿机制实现非全上下文建模，在 ImageNet 上 FID 降低 10.5%、峰值内存减少 83.8%。

**[MARVO: Marine-Adaptive Radiance-aware Visual Odometry](marvo_marine-adaptive_radiance-aware_visual_odometry.md)**

:   提出 MARVO 水下视觉里程计框架，将物理感知辐射适配器 (PARA) 嵌入 LoFTR 特征匹配器补偿水下波长衰减、结合 GTSAM 多传感器因子图融合和强化学习位姿图优化 (RL-PGO)，在水下场景实现鲁棒定位。

**[MEMO: Human-like Crisp Edge Detection Using Masked Edge Prediction](memo_human-like_crisp_edge_detection_using_masked_edge_prediction.md)**

:   提出 MEMO 框架，通过掩码边缘训练和基于置信度排序的渐进式推理策略，仅使用交叉熵损失就能生成清晰的单像素边缘图，在 crispness-aware 评估上大幅超越现有方法（BSDS 上 CEval ODS 从 0.749 提升到 0.836）。

**[MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)**

:   提出 Mine-JEPA，首个面向侧扫声纳（SSS）水雷分类的域内自监督学习流水线——基于 SIGReg 正则化损失、声纳适配增强策略和 ImageNet 初始化，仅用 1,170 张未标注声纳图像预训练即超越了在 17 亿图像上预训练的 DINOv3 基础模型。

**[On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)**

:   首次系统研究了扩散模型图像压缩在比特翻转错误下的鲁棒性，发现基于反向信道编码（RCC）的扩散压缩方法天然比传统和学习型编解码器更耐错，并提出 Robust Turbo-DDCM 变体通过独立编码原子索引进一步提升鲁棒性，在 BER 达 $10^{-3}$ 时仍保持良好重建质量。

**[Parallax to Align Them All: An OmniParallax Attention Mechanism for Distributed Multi-View Image Compression](parallax_to_align_them_all_an_omniparallax_attention_mechanism_for_distributed_m.md)**

:   提出 OmniParallax Attention Mechanism (OPAM) 用于分布式多视角图像压缩（DMIC），通过两阶段视差注意力显式建模任意视角对之间的相关性和对齐特征，构建的 ParaHydra 框架首次让 DMIC 方法显著超越 SOTA MIC 编码器，同时大幅降低计算开销。

**[PlanaReLoc: Camera Relocalization in 3D Planar Primitives via Region-Based Structure Matching](planareloc_camera_relocalization_in_3d_planar_primitives_via_region-based_struct.md)**

:   首次提出基于平面基元（planar primitives）和 3D 平面地图的相机重定位范式 PlanaReLoc，通过深度匹配器在统一嵌入空间中关联查询图像的平面区域与地图平面基元，实现了无需真实纹理地图、位姿先验或逐场景训练的轻量化 6-DoF 相机重定位。

**[Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model](planning_in_8_tokens_a_compact_discrete_tokenizer_for_latent_world_model.md)**

:   提出 CompACT，将每张图像压缩至仅 8 个离散 token（约 128 bits），通过冻结预训练视觉编码器保留规划关键语义信息、生成式解码补充感知细节，使基于世界模型的规划速度提升约 40 倍且精度不降。

**[PPCL: Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](ppcl_pluggable_pruning_dit_distillation.md)**

:   提出 PPCL 框架，针对超大规模 Multi-Modal Diffusion Transformer (MMDiT, 8–20B 参数) 设计结构化剪枝方案：通过线性探针 (Linear Probe) 学习每层的可替代性，结合 CKA 一阶差分自动定位连续冗余层区间，再以非顺序交替蒸馏实现深度+宽度双轴剪枝，最终在 Qwen-Image 20B 上实现 50% 参数缩减、1.8× 推理加速，平均性能仅下降 2.61%。

**[Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy](preference-aligned_lora_merging_preserving_subspace_coverage_and_addressing_dire.md)**

:   本文从子空间覆盖（subspace coverage）和方向各向异性（anisotropy）两个视角重新审视LoRA合并问题，提出TARA-Merging框架，通过保留LoRA方向并结合偏好加权的交叉熵伪损失进行方向级重新加权，在8个视觉和6个NLI基准上持续超越现有合并方法。

**[PriVi: Towards a General-Purpose Video Model for Primate Behavior in the Wild](privi_towards_a_general-purpose_video_model_for_primate_behavior_in_the_wild.md)**

:   PriVi 构建了 424 小时的大规模灵长类视频预训练数据集，并通过在 V-JEPA 上进行**领域级预训练**（非目标数据集级别），首次证明了视频模型的领域级预训练可以跨数据集泛化，在四个灵长类行为识别基准上用仅 220K 参数的冻结分类器超越了全量微调的专用模型。

**[QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)**

:   提出 QuantVLA，首个面向 Vision-Language-Action (VLA) 模型的免训练后量化框架，通过选择性量化布局和两个轻量级标定机制（注意力温度匹配 ATM 和输出头平衡 OHB），在 W4A8 精度下实现约 70% 的内存节省，同时任务成功率超过全精度基线。

**[RL-ScanIQA: Reinforcement-Learned Scanpaths for Blind 360° Image Quality Assessment](rl-scaniqa_reinforcement-learned_scanpaths_for_blind_360image_quality_assessment.md)**

**[SODA: Sensitivity-Oriented Dynamic Acceleration for Diffusion Transformer](soda_sensitivity-oriented_dynamic_acceleration_for_diffusion_transformer.md)**

:   提出 SODA，通过离线细粒度敏感度建模 + 动态规划优化缓存间隔 + 统一自适应剪枝策略，在无需训练的条件下对 Diffusion Transformer 实现可控加速比下的高保真生成。

**[Stronger Normalization-Free Transformers](stronger_normalization-free_transformers.md)**

:   通过系统分析逐点函数替代归一化层所需的四个关键属性（零中心性、有界性、中心敏感性、单调性），在大规模搜索中发现 $\text{Derf}(x) = \text{erf}(\alpha x + s)$ 是最优的归一化层替代函数，在视觉识别、图像生成、语音表示和DNA序列建模等多个领域持续超越LayerNorm和DyT，且性能增益主要来自更强的泛化而非拟合能力。

**[TALON: Test-time Adaptive Learning for On-the-Fly Category Discovery](talon_test-time_adaptive_learning_for_on-the-fly_category_discovery.md)**

:   提出首个面向 on-the-fly 类别发现（OCD）的测试时自适应框架 TALON，通过语义感知原型更新 + 稳定编码器适应 + 边距感知 logit 校准，摒弃哈希编码在连续特征空间直接建模，大幅缓解类别爆炸并显著提升新类发现精度。

**[F²HDR: Two-Stage HDR Video Reconstruction via Flow Adapter and Physical Motion Modeling](textf2texthdr_two-stage_hdr_video_reconstruction_via_flow_adapter_and_physical_m.md)**

:   提出 F²HDR，一个两阶段 HDR 视频重建框架，通过 Flow Adapter 将通用预训练光流适配到交替曝光场景以实现鲁棒对齐，并利用物理运动建模从光流中提取连续运动掩码来引导第二阶段的伪影消除，在真实 HDR 视频基准上达到 SOTA。

**[Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](towards_generalizable_ai-generated_image_detection_via_image-adaptive_prompt_lea.md)**

:   提出 Image-Adaptive Prompt Learning (IAPL)，在推理时根据每张测试图像动态调整 CLIP 编码器的 prompt，通过测试时 token 调优和条件信息学习器实现对未见生成器的强泛化，在 UniversalFakeDetect 和 GenImage 上分别达到 95.61% 和 96.7% 平均准确率的 SOTA 性能。

**[UniComp: Rethinking Video Compression Through Informational Uniqueness](unicomp_rethinking_video_compression_through_informational_uniqueness.md)**

:   提出基于信息唯一性（而非注意力）的视频 token 压缩框架 UniComp，通过帧组融合、token 分配和空间动态压缩三个模块在时序-空间-全局维度上最大化保留唯一信息，在仅保留 10% token 时仍能超越未压缩基线性能。

**[Unlocking ImageNet's Multi-Object Nature: Automated Large-Scale Multilabel Annotation](unlocking_imagenets_multi-object_nature_automated_large-scale_multilabel_annotat.md)**

:   提出全自动流水线，利用自监督 ViT 特征进行无监督目标发现，为 ImageNet-1K 全部 128 万训练图像生成带空间定位的多标签标注，无需人工标注，模型在域内和下游多标签任务上均获一致提升（ReaL +2.0 top-1, COCO +4.2 mAP）。
