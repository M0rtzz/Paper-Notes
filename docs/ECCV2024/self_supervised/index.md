---
title: >-
  ECCV2024 自监督/表示学习方向 9篇论文解读
description: >-
  9篇ECCV2024 自监督/表示学习方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**🎞️ ECCV2024** · 共 **9** 篇

**[Adaptive Multihead Contrastive Learning](adaptive_multihead_contrastive_learning.md)**

:   AMCL提出使用多个投影头（各自产生不同特征）+ 对每个样本对和每个头自适应学习温度参数，从最大似然估计推导出损失函数，作为通用插件在SimCLR/MoCo/Barlow Twins/CAN/LGP上一致提升1-5%性能。

**[Coho Context-Sensitive City-Scale Hierarchical Urban Layout Generation](coho_context-sensitive_city-scale_hierarchical_urban_layout_generation.md)**

:   提出基于图掩码自编码器 (GMAE) 的城市级 2.5D 布局生成方法，通过规范图表示捕获建筑-街区-社区的多层语义上下文，结合优先级调度的迭代采样，在 330 个美国城市上实现了兼具真实感、语义一致性和正确性的大规模城市布局生成。

**[Efficient Image Pre-Training With Siamese Cropped Masked Autoencoders](efficient_image_pre-training_with_siamese_cropped_masked_autoencoders.md)**

:   提出CropMAE——用同一图像的两个随机裁剪视图替代视频帧对来训练孪生掩码自编码器，在98.5%的极高掩码率下仅用2个可见patch即可学习物体边界感知表征，训练速度比SiamMAE提升最高23.8倍，同时在视频传播任务上达到竞争性能。

**[Flowcon Out-Of-Distribution Detection Using Flow-Based Contrastive Learning](flowcon_out-of-distribution_detection_using_flow-based_contrastive_learning.md)**

:   提出FlowCon，一种基于密度估计的OOD检测方法，创新性地将正规化流（normalizing flow）与监督对比学习结合——在流模型的潜在空间中使用基于Bhattacharyya系数的对比损失学习类别条件高斯分布，无需外部OOD数据或重训分类器即可实现高效的OOD检测。

**[Infmae A Foundation Model In The Infrared Modality](infmae_a_foundation_model_in_the_infrared_modality.md)**

:   提出 InfMAE——首个红外模态基础模型，构建了 30 万张红外图像数据集 Inf30，设计信息感知掩码策略和多尺度编码器，在红外语义分割、目标检测和小目标检测三个下游任务上超越现有方法。

**[Posformer Recognizing Complex Handwritten Mathematical Expression With Position ](posformer_recognizing_complex_handwritten_mathematical_expression_with_position_.md)**

:   提出位置森林 Transformer（PosFormer），通过将数学表达式的 LaTeX 序列编码为位置森林结构，显式建模符号间的层级与位置关系，并设计隐式注意力校正模块，在不增加推理开销的前提下，在单行/多行/复杂表达式数据集上全面超越 SOTA。

**[Rethinking Unsupervised Outlier Detection Via Multiple Thresholding](rethinking_unsupervised_outlier_detection_via_multiple_thresholding.md)**

:   提出 Multi-T（多阈值）模块，通过生成两个阈值分别隔离目标数据集中的 inlier 和 outlier，利用识别出的 inlier 训练干净的正常流形、利用 outlier 进行特征去噪，从而大幅提升已有离群值评分方法的性能。

**[Revisiting Supervision For Continual Representation Learning](revisiting_supervision_for_continual_representation_learning.md)**

:   挑战了"自监督学习在持续表征学习中优于监督学习"的普遍观点，发现**监督学习加上 MLP 投影头**即可在持续学习场景下构建出比 SSL 更强的表征——关键不在于有无标签，而在于 MLP projector 对特征可迁移性的提升作用。

**[Vic-Mae Self-Supervised Representation Learning From Images And Video With Contr](vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)**

:   ViC-MAE 将对比学习和掩码自编码器统一到一个框架中，通过把短视频片段当作增强视角（而非把图像重复为视频），在图像和视频下游任务上同时取得优秀表现——ImageNet-1K top-1 达 87.1%（超越 OmniMAE +2.4%），SSv2 达 75.9%。
