---
title: >-
  CVPR2025 图像恢复方向 24篇论文解读
description: >-
  24篇CVPR2025 图像恢复方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**📷 CVPR2025** · **24** 篇论文解读

**[A Flag Decomposition for Hierarchical Datasets](a_flag_decomposition_for_hierarchical_datasets.md)**

:   提出Flag Decomposition（FD）——一种保持层次结构的矩阵分解方法，将具有嵌套列层次的数据矩阵分解为Stiefel坐标表示的flag（嵌套子空间序列）、块上三角矩阵和置换矩阵，在去噪、聚类和小样本学习任务上优于SVD等标准方法。

**[A Physics-Informed Blur Learning Framework for Imaging Systems](a_physics-informed_blur_learning_framework_for_imaging_systems.md)**

:   提出基于物理的 PSF 学习框架，设计新型波前基（每个基仅影响单一 SFR 方向）消除梯度冲突，结合课程学习（中心→边缘），无需镜头参数即可精确估计成像系统的空间变化 PSF。

**[EQ-Reg: A Regularization-Guided Equivariant Approach for Image Restoration](a_regularization-guided_equivariant_approach_for_image_restoration.md)**

**[AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution](adversarial_diffusion_compression_for_real-world_image_super-resolution.md)**

:   提出对抗扩散压缩（ADC）框架，将一步扩散模型 OSEDiff 蒸馏为精简的扩散-GAN 混合模型，实现 73% 推理时间压缩、78% 计算量削减、74% 参数缩减，同时保持生成质量，达到 34.79 FPS 实时超分。

**[Augmenting Perceptual Super-Resolution via Image Quality Predictors](augmenting_perceptual_super-resolution_via_image_quality_predictors.md)**

:   利用无参考图像质量评估（NR-IQA）模型代替人工标注，通过加权采样和直接优化两种方式提升感知超分辨率的图像质量，在无需人工数据的条件下超越依赖人工反馈的 SOTA 方法。

**[Classic Video Denoising in a Machine Learning World: Robust, Fast, and Controllable](classic_video_denoising_in_a_machine_learning_world_robust_fast_and_controllable.md)**

:   重新审视经典视频去噪方法并与现代ML工具结合，实现鲁棒、快速且噪声级别可控的视频去噪

**[Complexity Experts are Task-Discriminative Learners for Any Image Restoration](complexity_experts_are_task-discriminative_learners_for_any_image_restoration.md)**

:   提出 MoCE-IR，用具有不同计算复杂度和感受野大小的"复杂度专家"替代传统均匀 MoE 的统一架构，配合偏向低复杂度的弹簧式路由机制，意外地实现了任务判别性分配——不同退化类型自动路由到适当复杂度的专家，可在推理时跳过无关专家。

**[DarkIR: Robust Low-Light Image Restoration](darkir_robust_low-light_image_restoration.md)**

:   提出 DarkIR，一种面向低光照多任务修复（去噪+去模糊+增强）的 CNN 端到端框架，编码器用频域注意力 FreMLP 纠正光照、解码器用膨胀空间注意力 Di-SpAM 去模糊，以 Restormer 1/5 参数和 1/5 计算量超越其 PSNR。

**[Degradation-Aware Feature Perturbation for All-in-One Image Restoration](degradation-aware_feature_perturbation_for_all-in-one_image_restoration.md)**

:   提出 DFPIR，通过退化引导的特征扰动——通道级扰动（退化引导的通道混洗在高维空间中重组特征）和注意力级扰动（top-K 掩码交叉注意力选择性传递信息）——将特征空间对齐到统一的参数空间，在 3/5 任务全合一修复中平均 PSNR 超越 InstructIR +0.45/+1.09 dB。

**[Detail-Preserving Latent Diffusion for Stable Shadow Removal](detail-preserving_latent_diffusion_for_stable_shadow_removal.md)**

:   提出两阶段阴影去除框架，Stage 1 微调 Stable Diffusion 去噪器以 z₀-prediction 方式生成粗阴影去除结果（低方差高稳定性），Stage 2 用 Detail Injection 模块通过 RRDB 块选择性地将阴影感知高频细节注入冻结 VAE 解码器，在四个基准上达到无掩码方法最优。

**[DiffFNO: Diffusion Fourier Neural Operator](difffno_diffusion_fourier_neural_operator.md)**

:   提出 DiffFNO，将加权傅里叶神经算子（WFNO，通过模式重平衡保留高频傅里叶模式）与扩散框架结合，配合门控融合机制和自适应时间步 ODE 求解器，实现任意尺度图像超分辨率，在 DIV2K 多尺度上比 SRNO/LMI 提升 0.3-1.0 dB PSNR。

**[EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation](echomimicv2_towards_striking_simplified_and_semi-body_human_animation.md)**

:   提出 Audio-Pose Dynamic Harmonization（APDH）策略渐进式将控制权从全身姿态转移到音频——逐步移除关键点（保留手部）同时扩大音频控制范围（从唇部到全身），实现仅需音频+参考图+手部姿态的高质量半身动画。

**[Efficient Diffusion as Low Light Enhancer (ReDDiT)](efficient_diffusion_as_low_light_enhancer.md)**

:   提出 ReDDiT 将扩散式低光增强从 10+ 步蒸馏到 2-4 步——通过线性外推修正拟合误差、用 Retinex 分解的反射率做轨迹精炼弥合推理间隙，4 步即在 10 个基准上全面达到 SOTA。

**[Efficient Visual State Space Model for Image Deblurring](efficient_visual_state_space_model_for_image_deblurring.md)**

:   本文提出一种高效视觉状态空间模型 EVSSM，通过几何变换替代多方向扫描策略捕获非局部信息，并设计高效频域前馈网络增强局部细节，在图像去模糊任务上以仅四分之一的计算代价超越现有 SSM 方法，达到 SOTA 效果。

**[FiRe: Fixed-points of Restoration Priors for Solving Inverse Problems](fire_fixed-points_of_restoration_priors_for_solving_inverse_problems.md)**

:   本文提出 FiRe 框架，通过将通用图像恢复模型（去模糊、超分、修复等）与其训练时的退化算子复合，利用不动点理论推导出显式先验公式，扩展了传统 PnP 中仅限去噪先验的范围，并支持多恢复模型的集成，在多种逆问题上显著超越现有 PnP 和扩散方法。

**[Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise](generalized_recorrupted-to-recorrupted_self-supervised_learning_beyond_gaussian_.md)**

:   本文提出Generalized R2R (GR2R)，将原始R2R自监督去噪框架从高斯噪声推广到自然指数族（NEF）分布——包括Poisson/Gamma/Binomial噪声，证明GR2R损失是有监督损失的无偏估计，并且SURE可视为其特例，在低光成像和SAR等应用中达到接近监督学习的性能。

**[Gyro-based Neural Single Image Deblurring](gyro-based_neural_single_image_deblurring.md)**

:   提出 GyroDeblurNet，通过新颖的相机运动场嵌入表示复杂手抖、陀螺仪细化模块利用图像模糊信息校正陀螺仪误差、陀螺仪去模糊模块用校正后的运动信息去除模糊，配合课程学习策略，在合成和真实数据集上大幅超越现有方法。

**[INFP: Audio-Driven Interactive Head Generation in Dyadic Conversations](infp_audio-driven_interactive_head_generation_in_dyadic_conversations.md)**

:   INFP 提出了一个统一的音频驱动交互式头部生成框架，通过双轨音频（agent + 对话伙伴）驱动 agent 在说话和倾听状态间自然切换，无需手动角色分配或显式角色切换，同时引入大规模 DyConv 数据集支持研究。

**[Iterative Predictor-Critic Code Decoding for Real-World Image Dehazing](iterative_predictor-critic_code_decoding_for_real-world_image_dehazing.md)**

:   IPC-Dehaze 提出了一种基于 VQGAN 码本先验的迭代式 Predictor-Critic 解码框架，通过 Code-Critic 评估码本序列间的相互关联来决定哪些码应保留或重采样，实现了从清晰区域到密集雾区的由易到难渐进去雾，在真实场景中显著超越 SOTA。

**[MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration](mair_a_locality-_and_continuity-preserving_mamba_for_image_restoration.md)**

:   提出 MaIR，核心创新是嵌套 S 形扫描策略（NSS）通过条带划分保持局部性 + S 形路径保持连续性，以及序列洗牌注意力（SSA）通过通道级注意力智能聚合不同扫描方向的序列，在超分、去噪、去模糊、去雾 4 大任务 14 个数据集上达到 SOTA。

**[MambaIRv2: Attentive State Space Restoration](mambairv2_attentive_state_space_restoration.md)**

:   提出 MambaIRv2，通过 Attentive State-space Equation（ASE）在 Mamba 的输出矩阵 $\mathbf{C}$ 中注入可学习 prompt 实现类似注意力的非因果全局查询，并用 Semantic Guided Neighboring（SGN）按语义标签重排序列缓解长距离衰减，仅需单方向扫描即超越多方向方法，轻量 SR 上以 9.3% 更少参数超 SRFormer 0.35dB。

**[POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction with Application to Strong Lens Discovery](polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)**

:   在 POLISH 框架基础上提出 POLISH+/++，通过**分块训练+拼接推理**和**arcsinh 非线性变换**两项改进，使深度学习方法首次能处理宽视场（12,960×12,960 像素）、高动态范围（~10⁶）的射电干涉成像，并展示了超分辨率对强引力透镜发现的 10× 提升潜力。

**[OptiFusion: Towards Universal Computational Aberration Correction in Photographic Cameras](towards_universal_computational_aberration_correction_in_photographic_cameras_a_.md)**

:   扩展 OptiFusion 自动设计 120 种多样化镜头，提出 ODE 综合评估指标和大规模 benchmark，系统对比 24 种算法，发现 CNN 模型在像差校正中提供最佳速度-精度权衡，反直觉地超越 Transformer。

**[Variational Garrote for Sparse Inverse Problems](variational_garrote_for_sparse_inverse_problems.md)**

:   系统比较 $\ell_1$ 正则化 (LASSO) 与 Variational Garrote (VG, 概率 $\ell_0$ 近似) 在信号重采样、去噪和稀疏视角 CT 重建三种逆问题上的表现，发现 VG 在强欠定情况下（采样率低/角度稀疏）通常获得更低的泛化误差，因为 spike-and-slab 先验与真实稀疏分布更匹配。
