---
title: >-
  CVPR2026 图像恢复方向41篇论文解读
description: >-
  41篇CVPR2026的图像恢复方向论文解读，涵盖图像恢复、超分辨率、扩散模型、对抗鲁棒、多模态、自监督学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "图像恢复"
  - "论文解读"
  - "论文笔记"
  - "超分辨率"
  - "扩散模型"
  - "对抗鲁棒"
  - "多模态"
  - "自监督学习"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**📷 CVPR2026** · **41** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (16)](../../ICML2026/image_restoration/index.md) · [🔬 ICLR2026 (15)](../../ICLR2026/image_restoration/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/image_restoration/index.md) · [🧠 NeurIPS2025 (26)](../../NeurIPS2025/image_restoration/index.md) · [📹 ICCV2025 (30)](../../ICCV2025/image_restoration/index.md) · [🧪 ICML2025 (5)](../../ICML2025/image_restoration/index.md)

🔥 **高频主题：** 图像恢复 ×11 · 超分辨率 ×9 · 扩散模型 ×5 · 对抗鲁棒 ×4 · 多模态 ×2

**[AceTone: Bridging Words and Colors for Conditional Image Grading](acetone_bridging_words_and_colors_for_conditional_image_grading.md)**

:   提出AceTone，首个支持文本和参考图像多模态条件色彩调色的统一框架，通过VQ-VAE将3D-LUT压缩为64个离散token，训练VLM预测LUT token序列，再用GRPO强化学习对齐色彩相似度和美学偏好，在风格迁移和指令调色上LPIPS改善50%。

**[Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)**

:   提出IQPIR框架，引入预训练NR-IQA模型的图像质量先验(IQP)作为条件信号，通过质量条件化Transformer、双Codebook结构和离散表示空间质量优化三个机制，引导图像修复过程趋向最高感知质量，在盲人脸修复等任务上全面超越SOTA。

**[Beyond the Ground Truth: Enhanced Supervision for Image Restoration](beyond_the_ground_truth_enhanced_supervision_for_image_restoration.md)**

:   提出通过超分辨率+频域自适应混合来增强现有数据集中次优GT图像的感知质量，并训练轻量级ORNet精修模块，无需修改预训练修复模型即可提升输出的感知质量。

**[BHCast: Unlocking Black Hole Plasma Dynamics from a Single Blurry Image with Long-Term Forecasting](bhcast_unlocking_black_hole_plasma_dynamics_from_a_single_blurry_image_with_long.md)**

:   BHCast从单张模糊的EHT黑洞图像出发，通过U-Net动力学代理模型进行超分辨率+长期自回归预测（100步稳定），从预测的等离子体动力学中提取物理特征（旋转速度、螺旋角等），再通过XGBoost推断黑洞自旋和倾角，在真实M87*观测图像上也展示了有效性。

**[Blink: Dynamic Visual Token Resolution for Enhanced Multimodal Understanding](blink_dynamic_visual_token_resolution_for_enhanced_multimodal_understanding.md)**

:   提出 Blink 框架，通过在 MLLM 不同 Transformer 层动态扩展和丢弃视觉 token（模拟人类"快速眨眼"式扫描），在单次前向传播中自适应增强视觉感知能力，在多个多模态基准上提升 LLaVA-1.5 性能。

**[BluRef: Unsupervised Image Deblurring with Dense-Matching References](bluref_unsupervised_image_deblurring_with_dense-matching_references.md)**

:   提出 BluRef，首个利用非配对参考清晰图像通过稠密匹配生成伪 ground truth 来训练去模糊网络的无监督框架，性能逼近甚至超越有监督方法。

**[Bridging the Perception Gap in Image Super-Resolution Evaluation](bridging_the_perception_gap_in_image_super-resolution_evaluation.md)**

:   通过大规模用户研究揭示现有 SR 评估指标（PSNR、SSIM、LPIPS 等）与人类感知严重不一致，分析其内在缺陷后提出极简但有效的 RQI（Relative Quality Index）框架，通过学习图像对之间的相对质量差异实现更可靠的 SR 评估，且可作为损失函数指导 SR 训练。

**[PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning](diffusion-based_srgb_real_noise_generation_via_prompt-driven_noise_representatio.md)**

:   PNG提出用可学习的Global/Local Prompt组件从真实噪声中自动提取噪声特征（替代ISO/相机型号等metadata），通过Prompt AutoEncoder编码噪声到latent空间+Prompt DiT（基于一致性模型）一步生成latent code，实现无需任何metadata的真实sRGB噪声合成，下游DnCNN去噪在SIDD上仅落后真实数据0.08dB。

**[Disentangled Textual Priors for Diffusion-based Image Super-Resolution](disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)**

:   提出 DTPSR，通过将文本先验沿空间层级（全局/局部）和频率语义（低频/高频）两个维度解耦，构建解耦的跨注意力注入管线和多分支 CFG 策略，实现感知质量优越的扩散超分辨率。

**[DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)**

:   提出退化感知扩散框架 DRFusion，通过直接回归融合图像（而非显式预测噪声）和联合观测模型校正机制，在少量扩散步骤内实现任意退化场景下的多模态图像融合。

**[EVLF: Early Vision-Language Fusion for Generative Dataset Distillation](evlf_early_vision-language_fusion_for_generative_dataset_distillation.md)**

:   提出 EVLF，一种在编码器-骨干网络接口处进行视觉-语言早期融合的即插即用方法，解决了扩散模型数据集蒸馏中晚期语义注入导致的文本过度主导和视觉保真度下降问题。

**[FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution](fidesr_high-fidelity_and_detail-preserving_one-step_diffusion_super-resolution.md)**

:   提出 FiDeSR，一种高保真和细节保持的单步扩散超分框架，通过细节感知加权（DAW）、隐空间残差精炼块（LRRB）和潜在频率注入模块（LFIM）三个互补组件，同时解决单步扩散超分中的结构保真度退化和高频细节恢复不足问题。

**[FinPercep-RM: A Fine-grained Reward Model and Co-evolutionary Curriculum for RL-based Real-world Super-Resolution](finpercep_rm_a_fine_grained_reward_model_and_co_evolutionary_curriculum_for_rl_ba.md)**

:   提出细粒度感知奖励模型 FinPercep-RM 和协同进化课程学习（CCL）策略，解决 RLHF 应用于真实世界超分辨率时的奖励黑客和训练不稳定问题，通过同时输出全局质量分数和空间退化热力图实现局部缺陷感知。

**[GSNR: Graph Smooth Null-Space Representation for Inverse Problems](gsnr_graph_smooth_null_space_representation_for_inverse_problems.md)**

:   提出图平滑零空间表示（GSNR），通过谱图理论构建零空间受限拉普拉斯矩阵并选择最平滑的 p 个谱模式作为零空间投影基，为 PnP、DIP 和扩散模型等逆问题求解器提供结构化的零空间约束，在去模糊、压缩感知、去马赛克和超分辨率上提升高达 4.3dB PSNR。

**[IA-CLAHE: Image-Adaptive Clip Limit Estimation for CLAHE](ia_clahe_image_adaptive_clip_limit.md)**

:   IA-CLAHE 通过证明 CLAHE 的直方图重分配过程几乎处处可微，首次实现了逐图块自适应 clip limit 的端到端学习，无需预搜索 ground truth clip limit 即可在恶劣天气条件下零样本提升识别性能和视觉质量。

**[Flickerformer: A Duet of Periodicity and Directionality for Burst Flicker Removal](it_takes_two_a_duet_of_periodicity_and_directionality_for_burst_flicker_removal.md)**

:   揭示闪烁伪影具有周期性和方向性两个内在物理特性，设计Flickerformer三模块（PFM/AFFN/WDAM）分别针对帧间/帧内周期性和方向性建模，以3.92M参数量在BurstDeflicker基准上达到31.226dB PSNR，超越第二名AST +0.580dB且仅用其19.70%参数。

**[Learning to Translate Noise for Robust Image Denoising](learning_to_translate_noise_for_robust_image_denoising.md)**

:   提出噪声翻译框架，通过轻量级噪声翻译网络将未知真实噪声转换为高斯噪声，再由预训练的高斯去噪网络处理，在 OOD 真实噪声基准上平均 PSNR 提升 1.5dB 以上，且翻译网络仅 0.29M 参数、可跨去噪器迁移。

**[MAD-Avatar: Motion-Aware Animatable Gaussian Avatars Deblurring](motionaware_animatable_gaussian_avatars_deblurring.md)**

:   首次实现从模糊视频直接重建清晰可驱动3D高斯人体avatar：提出3D感知的物理模糊形成模型(将模糊分解为子帧SMPL运动+canonical 3DGS)，用B-spline插值+位姿变形网络建模子帧运动，帧间正则化解决运动方向歧义，在合成和真实数据集上大幅超越"2D去模糊+3DGS"两阶段方案(PSNR提升约2.5dB)。

**[NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness](nec-diff_noise-robust_event-raw_complementary_diffusion_for_seeing_motion_in_ext.md)**

:   提出 NEC-Diff，一个基于扩散模型的事件-RAW 混合成像框架，利用 RAW 图像的光照先验引导事件去噪、事件的高动态范围边缘辅助图像去噪，结合双模态 SNR 引导的可靠信息提取和交叉模态注意力扩散，在极暗环境下（0.001-0.8 lux）实现高质量动态场景重建，PSNR 达 24.51 dB（REAL 数据集）。

**[NTIRE 2026 The 3rd RAIM Challenge: AI Flash Portrait (Track 3)](ntire_2026_ai_flash_portrait_challenge.md)**

:   NTIRE 2026第三届RAIM挑战赛AI Flash Portrait赛道：将弱闪光灯低光照人像映射为强闪光灯专业级人像，提供800组真实配对数据（含专业设计师修图GT），采用区域感知客观指标+专家盲评的双重评估体系，118支队伍注册、3187次有效提交。

**[NTIRE 2026 The Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images](ntire_2026_raindrop_removal_challenge.md)**

:   NTIRE 2026第二届日夜双焦点雨滴去除挑战赛总结报告：基于Raindrop Clarity真实数据集（14,139训练/407验证/593测试），168支队伍参赛，17支提交有效方案，冠军AIIA-Lab以MSDT骨干+伪GT精修流水线取得35.24分最佳成绩。

**[PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors](phasr_generalized_image_shadow_removal_with_physically_aligned_priors.md)**

:   提出PhaSR框架，通过双层物理先验对齐——全局级的PAN执行无参数Retinex分解抑制色彩偏差、局部级的GSRA利用差分注意力对齐DepthAnything深度先验和DINO-v2语义嵌入——实现从单光源直射阴影到多光源环境光场景的泛化阴影去除，在WSRD+和Ambient6K上达到SOTA且FLOPs最低。

**[POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)**

:   在 POLISH 框架基础上提出 POLISH+ 和 POLISH++，通过分块训练-拼接策略和基于 arcsinh 的非线性变换，实现宽视场（12,960×12,960 像素）和高动态范围（$\sim 10^6$）条件下的射电干涉图像重建与超分辨率，并首次展示深度学习方法可超分辨强引力透镜系统。

**[RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)**

:   RAR 将图像质量评估（IQA）与图像修复（IR）深度集成为统一端到端模型，在潜在空间中迭代执行"评估-修复-验证"循环，在复合退化场景下 PSNR 提升 +2.71 dB 且速度比 AgenticIR 快 11.27×。

**[RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](rawdomain_degradation_models_smartphone_sr.md)**

:   证明了精心设计的设备特定退化建模（通过标定获取真实的 blur 和 noise 参数）可以显著提升手机超分辨率的真实场景性能——通过将公开渲染图像 unprocess 到不同手机的 RAW 域生成高低分辨率训练对，训练的 SR 模型在保留设备的真实数据上明显优于使用大量任意退化组合训练的基线。

**[Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](real_iisr_infrared_image_super_resolution_autoregressive.md)**

:   提出 Real-IISR 统一自回归框架，通过热-结构引导模块、条件自适应码本和热序一致性损失解决真实红外图像超分辨率的特有挑战，并构建了 FLIR-IISR 数据集（1457 对真实 LR-HR 红外图像）。

**[ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation](reflexsplit_single_image_reflection_separation_via_layer_fusion-separation.md)**

:   ReflexSplit提出一种显式层融合-分离框架，通过跨尺度门控融合(CrGF)自适应聚合多尺度特征，层融合-分离模块(LFSB)中的差分双维度注意力 $\mathbf{A}^t - \lambda_\ell \mathbf{A}^r$ 进行跨流干扰抑制，配合深度依赖初始化+epoch-wise warmup的课程训练，在合成和真实世界反射分离基准上取得SOTA。

**[SAT: Selective Aggregation Transformer for Image Super-Resolution](sat_selective_aggregation_transformer_for_image_super_resolution.md)**

:   提出选择性聚合 Transformer (SAT)，通过密度驱动 token 聚合将 Key-Value 矩阵 token 数减少 97%、保持 Query 全分辨率，实现高效全局注意力建模，超越 SOTA PFT 达 0.22dB 且 FLOPs 降低 27%。

**[SelfHVD: Self-Supervised Handheld Video Deblurring](selfhvd_self-supervised_handheld_video_deblurring.md)**

:   SelfHVD 利用手持视频中自然存在的清晰帧作为监督信号，通过自增强视频去模糊（SEVD）构建高质量训练对和自约束空间一致性维护（SCSCM）防止位移偏移，实现了无需配对数据的手持视频去模糊。

**[Winner of CVPR2026 NTIRE Challenge on Image Shadow Removal: Semantic and Geometric Guidance for Shadow Removal via Cascaded Refinement](shadow_removal_cascaded_refinement.md)**

:   基于 OmniSR 构建三级级联精炼 pipeline，结合冻结 DINOv2 语义特征与单目深度/法线几何引导，通过收缩约束损失稳定多阶段训练，在 NTIRE 2026 阴影去除挑战赛中获得第一名。

**[ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration](shiftlut_spatial_shift_enhanced_look-up_tables_for_efficient_image_restoration.md)**

:   提出 ShiftLUT，通过可学习空间偏移模块（LSS）实现 LUT 方法中最大感受野（65×65），配合非对称双分支架构和误差有界自适应采样（EAS），在存储 104KB + 推理 84ms 的条件下超越所有现有 LUT 方法。

**[Spectral Super-Resolution via Adversarial Unfolding and Data-Driven Spectrum Regularization](spectral_super-resolution_via_adversarial_unfolding_and_data-driven_spectrum_reg.md)**

:   提出 UALNet，通过将数据驱动的光谱先验（PriorNet）和对抗学习项同时嵌入深度展开框架，实现从 Sentinel-2 多光谱数据（12 波段）到 NASA AVIRIS 高光谱图像（186 波段）的光谱超分辨率，性能超越 Transformer 的同时仅需 15% 计算量和 1/20 参数。

**[Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging](statistical_characteristic-guided_denoising_for_rapid_high-resolution_transmissi.md)**

:   提出统计特征引导去噪网络 SCGN，利用空间域的窗口标准差加权和频域的频带引导加权，分别在空间和频率两个域自适应地增强信号、抑制噪声，结合 HRTEM 专用噪声标定方法生成含无序结构的真实噪声数据集，实现毫秒级高分辨率透射电子显微镜图像的高质量去噪。

**[The Surprising Effectiveness of Noise Pretraining for Implicit Neural Representations](the_surprising_effectiveness_of_noise_pretraining_for_implicit_neural_representa.md)**

:   本文通过系统的实验分析发现：用非结构化噪声（均匀/高斯分布）预训练 INR 可在图像拟合中达到惊人的 ~80dB PSNR，远超所有数据驱动初始化方法；而具有自然图像 $1/|f^\alpha|$ 频谱结构的噪声则在信号拟合和去噪之间实现最佳平衡，无需任何真实数据即可匹配 SOTA 数据驱动初始化性能。

**[TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising](tm-bsn_triangular-masked_blind-spot_network_for_real-world_self-supervised_image.md)**

:   提出三角掩码盲点网络 TM-BSN，通过将盲点区域设计为与真实 sRGB 噪声的菱形空间相关模式精确对齐的形状，在原始分辨率上实现无需下采样的自监督图像去噪，并通过知识蒸馏进一步提升性能，在 SIDD 和 DND 基准上达到自监督去噪 SOTA。

**[UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](ucan_unified_convolutional_attention_lightweight_sr.md)**

:   提出 UCAN 轻量级超分辨率网络，统一卷积和注意力机制来高效扩展有效感受野，通过 Hedgehog 注意力解决线性注意力的秩坍缩问题，引入大核蒸馏模块和半共享参数策略，在 Manga109 (4×) 上以仅 48.4G MACs 达到 31.63 dB PSNR。

**[UDAPose: Unsupervised Domain Adaptation for Low-Light Human Pose Estimation](udapose_unsupervised_domain_adaptation_for_low_light_human_pose_estimation.md)**

:   UDAPose通过基于稳定扩散的低光照图像合成（保持高频低光特征）和动态注意力控制模块（自适应平衡视觉线索与姿态先验），在低光照硬集上实现56.4%的AP提升。

**[UniBlendNet: Unified Global, Multi-Scale, and Region-Adaptive Modeling for Ambient Lighting Normalization](uniblendnet_unified_global_multi_scale_and_region_adaptive_modeling_for_ambient_lighting_normalization.md)**

:   提出 UniBlendNet，在 IFBlend 基础上统一融合全局上下文建模、多尺度特征聚合和区域自适应残差精修三个模块，用于复杂空间变化光照条件下的环境光归一化任务。

**[Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis](unicac_universal_computational_aberration_correction_benchmark.md)**

:   本文构建了首个大规模通用计算像差校正(CAC)基准 UniCAC，提出光学退化评估器(ODE)量化像差难度，并对24种图像恢复/CAC算法进行了全面评估，揭示了先验利用、网络架构和训练策略三大关键因素对CAC性能的影响。

**[UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization](unirain_unified_image_deraining_rag_dataset_distillation.md)**

:   提出 UniRain 统一图像去雨框架，通过 RAG 驱动的数据蒸馏从百万级公开数据集筛选高质量样本，结合非对称 MoE 架构和多目标重加权优化策略，在雨条纹和雨滴（白天/夜间）四种退化类型上实现一致优异性能。

**[Variational Garrote for Sparse Inverse Problems](variational_garrote_for_sparse_inverse_problems.md)**

:   在统一的稀疏逆问题框架下，系统比较 $\ell_1$ 正则化（LASSO）与 Variational Garrote（VG，一种通过变分二值门控近似 $\ell_0$ 的方法），在信号重采样、去噪和稀疏视角 CT 重建三个任务上证明 VG 在强欠定场景下能显著降低最小泛化误差，尤其在采样率 <20% 或投影角度极少时优势最大。
