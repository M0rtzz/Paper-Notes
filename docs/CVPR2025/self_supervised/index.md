---
title: >-
  CVPR2025 自监督方向 16篇论文解读
description: >-
  16篇CVPR2025 自监督方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督

**📷 CVPR2025** · **16** 篇论文解读

**[AutoSSVH: Automated Frame Sampling for Self-Supervised Video Hashing](autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_.md)**

**[BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)**

:   提出BoSS——一种可扩展的主动学习oracle策略，通过集成多种选择策略生成候选批次、冻结backbone仅重训最后一层来评估性能增益，选择最优批次；在ImageNet等大规模数据集上首次展示了oracle性能，揭示SOTA主动学习策略仍有显著提升空间。

**[CheXWorld: Image World Modeling for Radiograph Representation Learning](chexworld_exploring_image_world_modeling_for_radiograph_representation_learning.md)**

**[Do Your Best and Get Enough Rest for Continual Learning](do_your_best_and_get_enough_rest_for_continual_learning.md)**

:   受Ebbinghaus遗忘曲线理论启发，提出View-Batch Model(VBM)——通过将batch中多个不同样本替换为同一样本的多个增强视图（replay），延长回忆间隔V倍至最优范围，同时用one-to-many KL散度自监督损失从单样本中学习更多知识（do your best），作为drop-in替代方案在多种持续学习方法上一致提升性能。

**[Escaping Plato's Cave: Towards the Alignment of 3D and Text Latent Spaces](escaping_platos_cave_towards_the_alignment_of_3d_and_text_latent_spaces.md)**

**[Few-Shot Implicit Function Generation via Equivariance](few-shot_implicit_function_generation_via_equivariance.md)**

:   通过等变性约束从少量样本生成隐式函数（NeRF/SDF），利用对称性先验减少对数据的需求

**[From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling](from_prototypes_to_general_distributions_an_efficient_curriculum_for_masked_imag.md)**

:   提出原型驱动的 MAE 课程学习——用 K-means 聚类识别数据集中的"原型"样本（靠近聚类中心的代表性图像），通过温度控制的采样策略从原型逐步过渡到全分布训练，实现 8× 训练加速（200 epoch 原型课程 ≈ 800 epoch 标准 MAE）。

**[Hyperbolic Category Discovery](hyperbolic_category_discovery.md)**

:   提出HypCD框架，将广义类别发现（GCD）中的表示学习从欧氏/球面空间迁移到双曲空间（Poincaré球模型），利用双曲空间指数级体积增长天然适合编码层次结构的特性，通过距离-角度混合相似度学习和双曲分类器，在CUB上将SelEx从69.1%提升到71.8%，在ImageNet-100上从87.1%提升到88.3%。

**[Learning to Normalize on the SPD Manifold under Bures-Wasserstein Geometry](learning_to_normalize_on_the_spd_manifold_under_bures-wasserstein_geometry.md)**

:   提出 GBWBN，在 SPD（对称正定）流形上用广义 Bures-Wasserstein 距离实现黎曼批归一化，通过可学习度量参数 M 和矩阵幂变形 θ 处理病态 SPD 矩阵问题，在骨架动作识别/脑电分类/无人机识别等 SPD 学习任务上一致提升。

**[MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining](map_unleashing_hybrid_mamba-transformer_vision_backbones_potential_with_masked_a.md)**

:   提出 Masked Autoregressive Pretraining（MAP），通过局部 MAE 建模 + 行级自回归解码的层次化预训练目标，首次有效预训练混合 Mamba-Transformer 视觉骨干，显著超越 MAE 和 AR 单一策略。

**[MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning](metawriter_personalized_handwritten_text_recognition_using_meta-learned_prompt_t.md)**

:   MetaWriter 将手写文字识别的个性化适配形式化为 prompt tuning 问题，结合 MAE 自监督辅助任务实现无标签测试时适应，并用元学习优化 prompt 初始化使自监督损失与识别损失对齐，仅更新不到1%参数即在IAM和RIMES上达到SOTA。

**[Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physical_systems.md)**

:   系统评估通用自监督方法在时空物理系统上学习物理相关表征的能力，发现在潜空间做预测的 JEPA 显著优于像素级重建的 MAE 和自回归模型，接近专用物理建模方法 DISCO。

**[SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning](smile_infusing_spatial_and_motion_semantics_in_masked_video_learning.md)**

:   提出 SMILE，通过合成运动增强（在视频上叠加沿随机轨迹运动的分割物体）和 CLIP 特征重建目标来增强掩码视频建模，结合轨迹引导的掩码策略，在 K400 线性探测上大幅提升至 56.2%（前 SOTA 47.5%）。

**[Spectral State Space Model for Rotation-Invariant Visual Representation Learning](spectral_state_space_model_for_rotation-invariant_visual_representation_learning.md)**

:   提出 Spectral VMamba，用谱图拉普拉斯的特征向量排序 patch 遍历顺序（替代预定义扫描线），结合旋转特征归一化器（RFN，聚合 4 个正则旋转的特征），在 miniImageNet 上达到 87.86% 准确率且对正则旋转完全不变。

**[Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)**

:   提出 TPSNet，利用文本-相位双先验解决无监督跨域图像检索：域提示（text prior）提供比伪标签更精确的语义监督，相位特征（phase prior）实现保持语义的域不变对齐，两者通过交叉注意力协同融合。

**[Transformers without Normalization](transformers_without_normalization.md)**

:   发现 LayerNorm 的输入-输出映射呈 tanh 形状，提出 Dynamic Tanh (DyT) 作为归一化层的即插即用替代：$\text{DyT}(x) = \gamma \odot \tanh(\alpha x) + \beta$，在视觉/语言/扩散/语音等多任务中与 LN 性能持平甚至更优。
