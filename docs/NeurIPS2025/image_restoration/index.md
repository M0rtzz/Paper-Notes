---
title: >-
  NeurIPS2025 图像恢复方向 27篇论文解读
description: >-
  27篇NeurIPS2025 图像恢复方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**🧠 NeurIPS2025** · 共 **27** 篇

**[Adaptive Discretization For Consistency Models](adaptive_discretization_for_consistency_models.md)**

:   提出ADCM框架，将一致性模型(CM)的离散化步长选择形式化为约束优化问题，通过Gauss-Newton方法得到解析解，在局部一致性（可训练性）和全局一致性（稳定性）之间自适应平衡，以仅4%的额外计算开销实现显著的训练效率提升和FID改善。

**[Audio Super-Resolution With Latent Bridge Models](audio_super-resolution_with_latent_bridge_models.md)**

:   提出 AudioLBM，在波形隐空间中用桥模型实现 LR-to-HR latent-to-latent 音频超分，配合频率感知训练和级联设计，LSD 平均改善 21.5%，首次实现 any-to-192kHz 音频超分。

**[Denoiserotator Enhance Pruning Robustness For Llms Via Importance Concentration](denoiserotator_enhance_pruning_robustness_for_llms_via_importance_concentration.md)**

:   提出 DenoiseRotator，在剪枝前通过可学习正交变换最小化参数重要性分数的信息熵，将重要性集中到少数参数上，使 LLaMA3-70B 在 2:4 半结构化稀疏下困惑度退化缩小 58%（8.1→3.4），可即插即用组合 Magnitude/Wanda/SparseGPT。

**[Dynaguide Steering Diffusion Polices With Active Dynamic Guidance](dynaguide_steering_diffusion_polices_with_active_dynamic_guidance.md)**

:   提出 DynaGuide，在推理时通过外部潜在动力学模型对预训练扩散策略施加 classifier guidance，无需修改策略权重即可引导机器人朝向任意正/负目标，在 CALVIN 仿真上平均成功率 70%，真实机器人达 80%。

**[Enhancing Infrared Vision Progressive Prompt Fusion Network And Benchmark](enhancing_infrared_vision_progressive_prompt_fusion_network_and_benchmark.md)**

:   针对热红外(TIR)图像中低对比度、模糊、噪声等多种退化耦合的问题，提出基于双提示融合的渐进式网络PPFN和选择性渐进训练策略SPT，并构建首个大规模多场景TIR基准数据集HM-TIR，在复合退化场景下PSNR提升8.76%。

**[Fiper Factorized Features For Robust Image Super-Resolution And Compression](fiper_factorized_features_for_robust_image_super-resolution_and_compression.md)**

:   提出 Factorized Features 统一表示——将图像分解为可学习的非均匀基与空间变化系数，配合锯齿坐标变换和多频调制，在 4× 超分辨率上 PSNR 相对提升 204.4%（HAT-L-F vs SwinIR），在图像压缩上 BD-rate 相比 VTM 降低 21.09%。

**[Gc4Nc A Benchmark Framework For Graph Condensation On Node Classification With N](gc4nc_a_benchmark_framework_for_graph_condensation_on_node_classification_with_n.md)**

:   提出 GC4NC——首个系统化的图凝缩（Graph Condensation）评估基准框架，跨 8 个维度（性能/效率/隐私保护/去噪/NAS有效性/可迁移性等）统一评估多种图凝缩方法，发现轨迹匹配方法最优、无结构方法效率最高，并在 1000x 压缩下图凝缩显著优于图像凝缩。

**[Implicit Augmentation From Distributional Symmetry In Turbulence Super-Resolutio](implicit_augmentation_from_distributional_symmetry_in_turbulence_super-resolutio.md)**

:   本文揭示湍流的统计各向同性本身就是一种隐式数据增强，使得标准CNN在超分辨率任务中无需显式旋转增强或等变架构即可部分习得旋转等变性，并展示了等变误差的尺度依赖性与Kolmogorov局部各向同性假说一致。

**[Improving Diffusion-Based Inverse Algorithms Under Few-Step Constraint Via Learn](improving_diffusion-based_inverse_algorithms_under_few-step_constraint_via_learn.md)**

:   提出 Learnable Linear Extrapolation (LLE)——用可学习的线性组合系数将当前和历史 clean data estimate 组合，以增强任何符合 Sampler-Corrector-Noiser 范式的扩散逆问题算法在少步（3-5步）下的表现，仅需 50 个样本、几分钟训练，跨 9+ 算法 × 5 个任务一致提升。

**[Latent Harmony Synergistic Unified Uhd Image Restoration Via Latent Space Regula](latent_harmony_synergistic_unified_uhd_image_restoration_via_latent_space_regula.md)**

:   提出 Latent Harmony 两阶段框架，通过潜在空间正则化构建泛化性 VAE（LH-VAE），并引入高频引导的可控 LoRA 微调机制，在保持结构完整性的同时实现 UHD 图像多退化类型统一修复的保真度-感知质量灵活权衡。

**[Latent Harmony Synergistic Unified Uhd Image Restoration With Pre-Trained Diffus](latent_harmony_synergistic_unified_uhd_image_restoration_with_pre-trained_diffus.md)**

:   提出 Latent Harmony 两阶段框架，通过潜在空间正则化构建退化鲁棒的 LH-VAE，再用高频引导的 LoRA 微调分别优化编码器（保真度）和解码器（感知质量），实现 UHD 全能图像复原中泛化-重建-感知三重权衡的统一解决方案。

**[Learning Cocoercive Conservative Denoisers Via Helmholtz Decomposition For Poiss](learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss.md)**

:   提出共循环保守(CoCo)去噪器概念，通过广义Helmholtz分解设计新的训练策略——Hamiltonian正则化促进保守性 + 谱正则化促进共循环性——使去噪器成为隐式弱凸先验的近端算子，从而在Poisson逆问题（光子受限去卷积、低剂量CT等）中实现有收敛保证且性能优越的PnP方法。

**[Luminance-Aware Statistical Quantization Unsupervised Hierarchical Learning For ](luminance-aware_statistical_quantization_unsupervised_hierarchical_learning_for_.md)**

:   提出 LASQ 框架，将低光图像增强重新定义为基于分层亮度分布的统计采样过程，利用自然亮度转换中固有的幂律分布特性，通过 MCMC 采样生成层次化亮度适配算子，嵌入扩散模型前向过程实现无监督增强，无需正常光照参考即可工作。

**[Map Estimation With Denoisers Convergence Rates And Guarantees](map_estimation_with_denoisers_convergence_rates_and_guarantees.md)**

:   证明了一个简单的 MMSE 去噪器迭代平均算法（与 Cold Diffusion 等实践方法密切相关）在对数凹先验假设下可证明收敛到负对数先验的近端算子，收敛速率为 Õ(1/k)，为一类经验上成功但缺乏理论保证的去噪方法提供了严格的理论基础，并将其嵌入近端梯度下降框架解决 MAP 估计问题。

**[Modem A Morton-Order Degradation Estimation Mechanism For Adverse Weather Image ](modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_.md)**

:   提出 MODEM 框架，通过 Morton 编码空间扫描与选择性状态空间模型（SSM）结合，建模空间异质性天气退化特征，配合双重退化估计模块提供全局和局部先验，实现多种天气退化图像的统一自适应复原 SOTA。

**[Moe-Gyro Self-Supervised Over-Range Reconstruction And Denoising For Mems Gyrosc](moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc.md)**

:   提出MoE-Gyro自监督专家混合框架，通过超量程重建专家(ORE，含高斯衰减注意力和物理信息约束)和降噪专家(DE，含双分支互补掩码和FFT引导增强)同时解决MEMS陀螺仪量程-噪声的根本权衡，将可测量范围从±450°/s扩展到±1500°/s，偏置不稳定性降低98.4%。

**[Mro Enhancing Reasoning In Diffusion Language Models Via Multi-Reward Optimizati](mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)**

:   MRO通过多奖励优化捕获扩散语言模型内/间序列token相关性，加速DLM推理同时保持性能。

**[Ms-Bart Unified Modeling Of Mass Spectra And Molecules For Structure Elucidation](ms-bart_unified_modeling_of_mass_spectra_and_molecules_for_structure_elucidation.md)**

:   提出 MS-Bart，通过统一词表将分子指纹和分子结构（SELFIES）映射到共享的 token 空间，在 400 万指纹-分子对上进行多任务预训练，再通过实验谱微调和化学反馈对齐，实现从质谱到分子结构的高效生成。

**[Real-World Adverse Weather Image Restoration Via Dual-Level Reinforcement Learni](real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni.md)**

:   提出双层强化学习框架（DRL），结合物理驱动的百万级合成天气数据集HFLS-Weather进行高质量冷启动训练，通过局部扰动驱动图像质量优化（PIQO）和全局元控制器多智能体协作，实现真实恶劣天气图像的自适应复原。

**[Rethinking Circuit Completeness In Language Models And Or And Adder Gates](rethinking_circuit_completeness_in_language_models_and_or_and_adder_gates.md)**

:   系统引入AND、OR、ADDER三种逻辑门来分解语言模型电路，揭示电路不完整性主要源于OR门的遗漏，提出结合noising和denoising干预的框架来完整恢复三种逻辑门，同时保证忠实度和完整性。

**[Rethinking Nighttime Image Deraining Via Learnable Color Space Transformation](rethinking_nighttime_image_deraining_via_learnable_color_space_transformation.md)**

:   提出CST-Net用于夜间图像去雨：基于夜间雨在Y通道（亮度）上比RGB更显著的观察，设计可学习颜色空间转换器(CSC)在YCbCr空间去雨，配合隐式光照引导模块(IIG)和新构建的光照感知合成数据集HQ-NightRain，在多个基准上达到SOTA。

**[Scan Self-Denoising Monte Carlo Annotation For Robust Process Reward Learning](scan_self-denoising_monte_carlo_annotation_for_robust_process_reward_learning.md)**

:   提出 SCAN 框架，通过分析 Monte Carlo 注释中的噪声分布，设计自去噪采样策略和鲁棒学习损失，仅用 1.5B 模型生成的 101K 样本训练的 PRM 即超越人工标注数据集 PRM800K 的效果。

**[Scsplit Bringing Severity Cognizance To Image Decomposition In Fluorescence Micr](scsplit_bringing_severity_cognizance_to_image_decomposition_in_fluorescence_micr.md)**

:   提出 scSplit，通过引入混合比例感知的归一化模块（SCIN）和回归网络（Reg），使基于 InDI 的迭代图像分解方法能够感知荧光显微镜图像中两种结构叠加的严重程度，在5个公开数据集上统一解决图像分离和渗透去除两个任务。

**[Spend Wisely Maximizing Post-Training Gains In Iterative Synthetic Data Bootstra](spend_wisely_maximizing_post-training_gains_in_iterative_synthetic_data_bootstra.md)**

:   首次从理论上分析了迭代合成数据自举训练中的预算分配问题，证明恒定策略无法高概率收敛，而指数增长策略在最坏情况下优于多项式策略，并在图像去噪（DPM）和数学推理（LLM）实验中验证了该结论。

**[Spiking Meets Attention Efficient Remote Sensing Image Super-Resolution With Att](spiking_meets_attention_efficient_remote_sensing_image_super-resolution_with_att.md)**

:   提出 SpikeSR，首个基于注意力脉冲神经网络(SNN)的遥感图像超分辨率框架，通过脉冲注意力块(SAB)结合混合维度注意力(HDA)和可变形相似度注意力(DSA)，在 AID/DOTA/DIOR 上达到 SOTA 性能同时保持高计算效率。

**[The Effect Of Optimal Self-Distillation In Noisy Gaussian Mixture Model](the_effect_of_optimal_self-distillation_in_noisy_gaussian_mixture_model.md)**

:   用统计物理replica方法分析噪声高斯混合模型上的自蒸馏，证明硬伪标签的去噪是性能提升主因，CIFAR-10实验验证。

**[Video Killed The Energy Budget Characterizing The Latency And Power Regimes Of O](video_killed_the_energy_budget_characterizing_the_latency_and_power_regimes_of_o.md)**

:   对开源T2V模型进行系统性延迟与能耗分析：建立了基于FLOP的compute-bound理论模型，验证了WAN2.1-T2V的二次空间/时间缩放和线性去噪步数缩放规律，并横向对比7个T2V模型发现能耗差异达3000倍（AnimateDiff 0.14Wh vs WAN2.1-14B 415Wh）。
