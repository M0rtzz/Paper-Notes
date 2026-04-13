---
title: >-
  CVPR2025 图像恢复方向 11篇论文解读
description: >-
  11篇CVPR2025 图像恢复方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🖼️ 图像恢复

**📷 CVPR2025** · 共 **11** 篇

**[A Flag Decomposition For Hierarchical Datasets](a_flag_decomposition_for_hierarchical_datasets.md)**

:   提出Flag Decomposition（FD）——一种保持层次结构的矩阵分解方法，将具有嵌套列层次的数据矩阵分解为Stiefel坐标表示的flag（嵌套子空间序列）、块上三角矩阵和置换矩阵，在去噪、聚类和小样本学习任务上优于SVD等标准方法。

**[A Physics-Informed Blur Learning Framework For Imaging Systems](a_physics-informed_blur_learning_framework_for_imaging_systems.md)**

:   提出基于物理的 PSF 学习框架，设计新型波前基（每个基仅影响单一 SFR 方向）消除梯度冲突，结合课程学习（中心→边缘），无需镜头参数即可精确估计成像系统的空间变化 PSF。

**[A Regularization-Guided Equivariant Approach For Image Restoration](a_regularization-guided_equivariant_approach_for_image_restoration.md)**

**[Adversarial Diffusion Compression For Real-World Image Super-Resolution](adversarial_diffusion_compression_for_real-world_image_super-resolution.md)**

:   提出对抗扩散压缩（ADC）框架，将一步扩散模型 OSEDiff 蒸馏为精简的扩散-GAN 混合模型，实现 73% 推理时间压缩、78% 计算量削减、74% 参数缩减，同时保持生成质量，达到 34.79 FPS 实时超分。

**[Augmenting Perceptual Super-Resolution Via Image Quality Predictors](augmenting_perceptual_super-resolution_via_image_quality_predictors.md)**

:   利用无参考图像质量评估（NR-IQA）模型代替人工标注，通过加权采样和直接优化两种方式提升感知超分辨率的图像质量，在无需人工数据的条件下超越依赖人工反馈的 SOTA 方法。

**[Bf-Stvsr B-Splines And Fourier---Best Friends For High Fidelity Spatia](bf-stvsr_b-splines_and_fourier---best_friends_for_high_fidelity_spatia.md)**

:   提出 BF-STVSR 框架，用 B-spline Mapper 建模时间运动插值、Fourier Mapper 捕获空间高频细节，无需外部光流网络即可实现连续时空视频超分辨率的 SOTA 性能。

**[Bf-Stvsr B-Splines And Fourier---Best Friends For High Fidelity Spatial-Temporal](bf-stvsr_b-splines_and_fourier---best_friends_for_high_fidelity_spatial-temporal.md)**

:   提出 BF-STVSR，结合 B 样条映射器（时间平滑插值）和傅里叶映射器（空间高频捕获）实现连续时空视频超分辨率，完全无需预训练光流网络（RAFT），在 GoPro 数据集上 PSNR 达 30.22dB，FLOPs 在所有方法中最低。

**[Classic Video Denoising In A Machine Learning World Robust Fast And Controllable](classic_video_denoising_in_a_machine_learning_world_robust_fast_and_controllable.md)**

:   重新审视经典视频去噪方法并与现代ML工具结合，实现鲁棒、快速且噪声级别可控的视频去噪

**[Polishing The Sky Wide-Field And High-Dynamic Range Interferometric Image Recons](polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)**

:   在 POLISH 框架基础上提出 POLISH+/++，通过**分块训练+拼接推理**和**arcsinh 非线性变换**两项改进，使深度学习方法首次能处理宽视场（12,960×12,960 像素）、高动态范围（~10⁶）的射电干涉成像，并展示了超分辨率对强引力透镜发现的 10× 提升潜力。

**[Towards Universal Computational Aberration Correction In Photographic Cameras A ](towards_universal_computational_aberration_correction_in_photographic_cameras_a_.md)**

:   扩展 OptiFusion 自动设计 120 种多样化镜头，提出 ODE 综合评估指标和大规模 benchmark，系统对比 24 种算法，发现 CNN 模型在像差校正中提供最佳速度-精度权衡，反直觉地超越 Transformer。

**[Variational Garrote For Sparse Inverse Problems](variational_garrote_for_sparse_inverse_problems.md)**

:   系统比较 $\ell_1$ 正则化 (LASSO) 与 Variational Garrote (VG, 概率 $\ell_0$ 近似) 在信号重采样、去噪和稀疏视角 CT 重建三种逆问题上的表现，发现 VG 在强欠定情况下（采样率低/角度稀疏）通常获得更低的泛化误差，因为 spike-and-slab 先验与真实稀疏分布更匹配。
