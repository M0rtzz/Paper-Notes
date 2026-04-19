---
title: >-
  ICCV2025 预训练方向 7篇论文解读
description: >-
  7篇ICCV2025 预训练方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练

**📹 ICCV2025** · **7** 篇论文解读

**[ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training](aceg_improving_generalization_of_scene_coordinate_regression.md)**

:   将场景坐标回归器拆分为「场景无关的Transformer」和「场景特定的map code」，通过在数万场景上进行交替的mapping/query预训练，显著提升SCR方法在光照、视角变化下的泛化能力，同时保持轻量化的计算开销。

**[ConstStyle: Robust Domain Generalization with Unified Style Transformation](conststyle_robust_domain_generalization_with_unified_style_transformation.md)**

:   提出ConstStyle框架，通过构建一个理论驱动的"统一域"（Unified Domain），在训练时将所有样本风格对齐到该统一域，测试时将未见域样本部分投影到统一域，有效缩小域间差距并提升泛化性能。

**[Dataset Ownership Verification for Pre-trained Masked Models](dataset_ownership_verification_for_pre-trained_masked_models.md)**

:   DOV4MM 提出了首个针对掩码预训练模型的数据集所有权验证方法，通过比较"见过"与"未见过"样本在嵌入空间中遮掩信息重构难度的差异，利用配对 t 检验判断黑盒模型是否使用了特定数据集进行预训练，在 10 种掩码图像模型和 4 种掩码语言模型上均实现 p 值远低于 0.05 的准确验证。

**[ETA: Energy-based Test-time Adaptation for Depth Completion](eta_energy-based_test-time_adaptation_for_depth_completion.md)**

:   提出ETA方法，利用能量模型量化深度预测属于源域分布的可能性，并在测试时通过最小化目标域预测的能量值来引导预训练深度补全模型适配到新环境，在室外和室内场景平均比先前SOTA分别提升6.94%和10.23%。

**[FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)**

:   提出 FlowMo，一种基于 Transformer 的扩散自编码器 (diffusion autoencoder)，通过两阶段训练（mode-matching 预训练 + mode-seeking 后训练），首次实现扩散自编码器在 ImageNet-1K 离散图像 tokenization 上的 SOTA 性能，无需使用卷积、对抗损失、2D 空间对齐 latent 或从其他 tokenizer 蒸馏。

**[Image Intrinsic Scale Assessment: Bridging the Gap Between Quality and Resolution](image_intrinsic_scale_assessment_bridging_the_gap_between_quality_and_resolution.md)**

:   本文定义了图像内在尺度（IIS）这一新概念——即图像展现最高感知质量的最大缩放比例，并提出 IISA 任务、构建了 785 张图像的数据集，以及基于弱标签的 WIISA 训练策略，在多个 NR-IQA 方法上一致提升了 IIS 预测性能。

**[Make Your Training Flexible: Towards Deployment-Efficient Video Models](make_your_training_flexible_towards_deployment-efficient_video_models.md)**

:   本文提出Flux——一种使视频模型训练灵活化的数据增强工具，通过灵活采样网格+组动态token选择，使单一模型在不同计算预算下都能高效工作；并提出Token Optimization新测试范式，在1/4 token下即可匹配前SOTA性能，节省约90%计算。
