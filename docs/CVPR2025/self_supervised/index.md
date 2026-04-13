---
title: >-
  CVPR2025 自监督/表示学习方向 8篇论文解读
description: >-
  8篇CVPR2025 自监督/表示学习方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**📷 CVPR2025** · 共 **8** 篇

**[Autossvh Exploring Automated Frame Sampling For Efficient Self-Supervised Video ](autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_.md)**

**[Boss A Best-Of-Strategies Selector As An Oracle For Deep Active Learning](boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)**

:   提出BoSS——一种可扩展的主动学习oracle策略，通过集成多种选择策略生成候选批次、冻结backbone仅重训最后一层来评估性能增益，选择最优批次；在ImageNet等大规模数据集上首次展示了oracle性能，揭示SOTA主动学习策略仍有显著提升空间。

**[Chexworld Exploring Image World Modeling For Radiograph Representation Learning](chexworld_exploring_image_world_modeling_for_radiograph_representation_learning.md)**

**[Do Your Best And Get Enough Rest For Continual Learning](do_your_best_and_get_enough_rest_for_continual_learning.md)**

:   受Ebbinghaus遗忘曲线理论启发，提出View-Batch Model(VBM)——通过将batch中多个不同样本替换为同一样本的多个增强视图（replay），延长回忆间隔V倍至最优范围，同时用one-to-many KL散度自监督损失从单样本中学习更多知识（do your best），作为drop-in替代方案在多种持续学习方法上一致提升性能。

**[Escaping Platos Cave Towards The Alignment Of 3D And Text Latent Spaces](escaping_platos_cave_towards_the_alignment_of_3d_and_text_latent_spaces.md)**

**[Few-Shot Implicit Function Generation Via Equivariance](few-shot_implicit_function_generation_via_equivariance.md)**

:   通过等变性约束从少量样本生成隐式函数（NeRF/SDF），利用对称性先验减少对数据的需求

**[Representation Learning For Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physical_systems.md)**

:   系统评估通用自监督方法在时空物理系统上学习物理相关表征的能力，发现在潜空间做预测的 JEPA 显著优于像素级重建的 MAE 和自回归模型，接近专用物理建模方法 DISCO。

**[Text-Phase Synergy Network With Dual Priors For Unsupervised Cross-Domain Image ](text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)**

:   提出 TPSNet，利用文本-相位双先验解决无监督跨域图像检索：域提示（text prior）提供比伪标签更精确的语义监督，相位特征（phase prior）实现保持语义的域不变对齐，两者通过交叉注意力协同融合。
