---
title: >-
  ECCV2024 自监督/表示学习方向15篇论文解读
description: >-
  15篇ECCV2024的自监督/表示学习方向论文解读，涵盖自监督学习、布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**🎞️ ECCV2024** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (2)](../../ACL2026/self_supervised/index.md) · [📷 CVPR2026 (31)](../../CVPR2026/self_supervised/index.md) · [🔬 ICLR2026 (15)](../../ICLR2026/self_supervised/index.md) · [🤖 AAAI2026 (14)](../../AAAI2026/self_supervised/index.md) · [🧠 NeurIPS2025 (35)](../../NeurIPS2025/self_supervised/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/self_supervised/index.md)

🔥 **高频主题：** 自监督学习 ×5

**[COHO: Context-Sensitive City-Scale Hierarchical Urban Layout Generation](coho_context-sensitive_city-scale_hierarchical_urban_layout_generation.md)**

:   提出基于图掩码自编码器 (GMAE) 的城市级 2.5D 布局生成方法，通过规范图表示捕获建筑-街区-社区的多层语义上下文，结合优先级调度的迭代采样，在 330 个美国城市上实现了兼具真实感、语义一致性和正确性的大规模城市布局生成。

**[Efficient Image Pre-Training with Siamese Cropped Masked Autoencoders](efficient_image_pre-training_with_siamese_cropped_masked_autoencoders.md)**

:   提出CropMAE——用同一图像的两个随机裁剪视图替代视频帧对来训练孪生掩码自编码器，在98.5%的极高掩码率下仅用2个可见patch即可学习物体边界感知表征，训练速度比SiamMAE提升最高23.8倍，同时在视频传播任务上达到竞争性能。

**[Exemplar-Free Continual Representation Learning via Learnable Drift Compensation](exemplar-free_continual_representation_learning_via_learnable_drift_compensation.md)**

:   提出可学习漂移补偿(LDC)，通过训练一个前向投影器将旧特征空间映射到新特征空间，在无需存储旧样本的情况下有效补偿类原型的语义漂移，首次实现了无样本半监督持续学习。

**[FlowCon: Out-of-Distribution Detection using Flow-Based Contrastive Learning](flowcon_out-of-distribution_detection_using_flow-based_contrastive_learning.md)**

:   提出FlowCon，一种基于密度估计的OOD检测方法，创新性地将正规化流（normalizing flow）与监督对比学习结合——在流模型的潜在空间中使用基于Bhattacharyya系数的对比损失学习类别条件高斯分布，无需外部OOD数据或重训分类器即可实现高效的OOD检测。

**[InfMAE: A Foundation Model in the Infrared Modality](infmae_a_foundation_model_in_the_infrared_modality.md)**

:   提出 InfMAE——首个红外模态基础模型，构建了 30 万张红外图像数据集 Inf30，设计信息感知掩码策略和多尺度编码器，在红外语义分割、目标检测和小目标检测三个下游任务上超越现有方法。

**[MarineInst: A Foundation Model for Marine Image Analysis with Instance Visual Description](marineinst_a_foundation_model_for_marine_image_analysis_with_instance_visual_des.md)**

:   本文提出MarineInst，一个面向海洋图像分析的基础模型，能够同时输出实例掩码和语义描述；并构建了MarineInst20M——迄今最大的海洋图像数据集（2000万张），支持从图像级场景理解到区域级实例理解的多层次海洋视觉分析任务。

**[OmniSat: Self-Supervised Modality Fusion for Earth Observation](omnisat_self-supervised_modality_fusion_for_earth_observation.md)**

:   提出OmniSat统一框架，通过模态特异编码器+跨模态对比自监督预训练，将多光谱时序（S2）、SAR时序（S1）、高分辨率单时相（SPOT/Aerial）等异构遥感数据融合为统一表示，在语义分割和作物分类上超越所有单模态和多模态基线。

**[PosFormer: Recognizing Complex Handwritten Mathematical Expression with Position Forest Transformer](posformer_recognizing_complex_handwritten_mathematical_expression_with_position_.md)**

:   提出位置森林 Transformer（PosFormer），通过将数学表达式的 LaTeX 序列编码为位置森林结构，显式建模符号间的层级与位置关系，并设计隐式注意力校正模块，在不增加推理开销的前提下，在单行/多行/复杂表达式数据集上全面超越 SOTA。

**[PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery](promptccd_learning_gaussian_mixture_prompt_pool_for_continual_category_discovery.md)**

:   提出PromptCCD框架，利用高斯混合模型（GMM）作为提示池，实现在无标签数据流中的持续新类别发现，同时缓解灾难性遗忘。

**[Rethinking Unsupervised Outlier Detection via Multiple Thresholding](rethinking_unsupervised_outlier_detection_via_multiple_thresholding.md)**

:   提出 Multi-T（多阈值）模块，通过生成两个阈值分别隔离目标数据集中的 inlier 和 outlier，利用识别出的 inlier 训练干净的正常流形、利用 outlier 进行特征去噪，从而大幅提升已有离群值评分方法的性能。

**[Revisiting Supervision for Continual Representation Learning](revisiting_supervision_for_continual_representation_learning.md)**

:   挑战了"自监督学习在持续表征学习中优于监督学习"的普遍观点，发现**监督学习加上 MLP 投影头**即可在持续学习场景下构建出比 SSL 更强的表征——关键不在于有无标签，而在于 MLP projector 对特征可迁移性的提升作用。

**[Self-supervised Video Copy Localization with Regional Token Representation](self-supervised_video_copy_localization_with_regional_token_representation.md)**

:   提出了一种自监督视频拷贝定位框架，通过在 Vision Transformer 中引入 Regional Token 捕获局部区域信息，并利用传递性（Transitivity Property）自动生成训练数据，在无需人工标注的情况下超越了有监督方法的性能。

**[Semantically Guided Representation Learning For Action Anticipation](semantically_guided_representation_learning_for_action_anticipation.md)**

:   提出 S-GEAR 框架，通过学习视觉动作原型并利用语言模型的语义关联来引导原型之间的几何关系，使模型理解动作间的语义互联性，从而提升动作预测性能，在 Epic-Kitchens 55/100、EGTEA Gaze+、50 Salads 四个基准上取得 SOTA 或极具竞争力的结果。

**[ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders](vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)**

:   ViC-MAE 将对比学习和掩码自编码器统一到一个框架中，通过把短视频片段当作增强视角（而非把图像重复为视频），在图像和视频下游任务上同时取得优秀表现——ImageNet-1K top-1 达 87.1%（超越 OmniMAE +2.4%），SSv2 达 75.9%。

**[WeCromCL: Weakly Supervised Cross-Modality Contrastive Learning for Transcription-only Supervised Text Spotting](wecromcl_weakly_supervised_cross-modality_contrastive_learning_for_transcription.md)**

:   提出 WeCromCL 框架，通过弱监督的原子级跨模态对比学习，仅利用文本转录标注（无位置标注）实现场景文字定位，将检测到的锚点作为伪标签训练单点监督文字检测器，在无边界标注的条件下达到接近全监督的性能。
