---
title: >-
  CVPR2025 自监督/表示学习方向22篇论文解读
description: >-
  22篇CVPR2025的自监督/表示学习方向论文解读，涵盖少样本学习、自监督学习、持续学习、对齐/RLHF、异常检测、个性化生成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**📷 CVPR2025** · **22** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (2)](../../ACL2026/self_supervised/index.md) · [📷 CVPR2026 (31)](../../CVPR2026/self_supervised/index.md) · [🔬 ICLR2026 (15)](../../ICLR2026/self_supervised/index.md) · [🤖 AAAI2026 (14)](../../AAAI2026/self_supervised/index.md) · [🧠 NeurIPS2025 (35)](../../NeurIPS2025/self_supervised/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/self_supervised/index.md)

🔥 **高频主题：** 少样本学习 ×2

**[AutoSSVH: Automated Frame Sampling for Self-Supervised Video Hashing](autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_.md)**

**[BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)**

:   提出BoSS——一种可扩展的主动学习oracle策略，通过集成多种选择策略生成候选批次、冻结backbone仅重训最后一层来评估性能增益，选择最优批次；在ImageNet等大规模数据集上首次展示了oracle性能，揭示SOTA主动学习策略仍有显著提升空间。

**[Breaking the Tuning Barrier: Zero-Hyperparameters Yield Multi-Corner Analysis Via Learned Priors](breaking_the_tuning_barrier_zero-hyperparameters_yield_multi-corner_analysis_via.md)**

:   用预训练的Foundation Model（TabPFN）替代传统手工先验，实现零超参数调优的电路Yield Multi-Corner Analysis：冻结backbone做in-context learning，自动跨corner迁移知识，结合自动特征选择（1152D→48D），在SRAM benchmarks上达到SOTA精度（MRE低至0.11%）且验证成本降低10倍以上。

**[CheXWorld: Image World Modeling for Radiograph Representation Learning](chexworld_exploring_image_world_modeling_for_radiograph_representation_learning.md)**

**[Do Your Best and Get Enough Rest for Continual Learning](do_your_best_and_get_enough_rest_for_continual_learning.md)**

:   受Ebbinghaus遗忘曲线理论启发，提出View-Batch Model(VBM)——通过将batch中多个不同样本替换为同一样本的多个增强视图（replay），延长回忆间隔V倍至最优范围，同时用one-to-many KL散度自监督损失从单样本中学习更多知识（do your best），作为drop-in替代方案在多种持续学习方法上一致提升性能。

**[Escaping Plato's Cave: Towards the Alignment of 3D and Text Latent Spaces](escaping_platos_cave_towards_the_alignment_of_3d_and_text_latent_spaces.md)**

**[Few-Shot Implicit Function Generation via Equivariance](few-shot_implicit_function_generation_via_equivariance.md)**

:   通过等变性约束从少量样本生成隐式函数（NeRF/SDF），利用对称性先验减少对数据的需求

**[From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling](from_prototypes_to_general_distributions_an_efficient_curriculum_for_masked_imag.md)**

:   提出原型驱动的 MAE 课程学习——用 K-means 聚类识别数据集中的"原型"样本（靠近聚类中心的代表性图像），通过温度控制的采样策略从原型逐步过渡到全分布训练，实现 8× 训练加速（200 epoch 原型课程 ≈ 800 epoch 标准 MAE）。

**[GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](geochemad_benchmarking_unsupervised_geochemical_anomaly_detection_for_mineral_ex.md)**

:   提出 GeoChemAD 开源基准数据集（8 个子集，覆盖多区域/多采样源/多目标元素）和 GeoChemFormer 框架，通过空间上下文自监督预训练和元素依赖建模实现无监督地球化学异常检测，在所有子集上取得最优 AUC。

**[Hyperbolic Category Discovery](hyperbolic_category_discovery.md)**

:   提出HypCD框架，将广义类别发现（GCD）中的表示学习从欧氏/球面空间迁移到双曲空间（Poincaré球模型），利用双曲空间指数级体积增长天然适合编码层次结构的特性，通过距离-角度混合相似度学习和双曲分类器，在CUB上将SelEx从69.1%提升到71.8%，在ImageNet-100上从87.1%提升到88.3%。

**[Learning to Normalize on the SPD Manifold under Bures-Wasserstein Geometry](learning_to_normalize_on_the_spd_manifold_under_bures-wasserstein_geometry.md)**

:   本文提出 GBWBN，首个基于广义 Bures-Wasserstein 几何的 SPD 流形批归一化方法，引入可学习的度量参数和矩阵幂非线性变形来有效处理病态协方差矩阵，在骨骼动作识别和脑电分类上取得 SOTA。

**[MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining](map_unleashing_hybrid_mamba-transformer_vision_backbones_potential_with_masked_a.md)**

:   提出 Masked Autoregressive Pretraining（MAP），通过局部 MAE 建模 + 行级自回归解码的层次化预训练目标，首次有效预训练混合 Mamba-Transformer 视觉骨干，显著超越 MAE 和 AR 单一策略。

**[MaRI: Material Retrieval Integration across Domains](mari_material_retrieval_integration_across_domains.md)**

:   提出 MaRI 框架，用双 DINOv2 编码器（图像 + 材质）通过对比学习构建共享嵌入空间，结合 Blender 合成数据和 ZeST 生成的真实世界材质数据，实现跨域准确的 PBR 材质检索。

**[MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning](metawriter_personalized_handwritten_text_recognition_using_meta-learned_prompt_t.md)**

:   MetaWriter 将手写文字识别的个性化适配形式化为 prompt tuning 问题，结合 MAE 自监督辅助任务实现无标签测试时适应，并用元学习优化 prompt 初始化使自监督损失与识别损失对齐，仅更新不到1%参数即在IAM和RIMES上达到SOTA。

**[OCRT: Boosting Foundation Models in the Open World with Object-Concept-Relation Triad](ocrt_boosting_foundation_models_in_the_open_world_with_object-concept-relation_t.md)**

:   OCRT 提出一个即插即用的三阶段管道——Object (Slot Attention 解耦)、Concept (重要性筛选)、Relation (概念图推理)——在不改 FM 主干的前提下显著提升 SAM 在弱监督医学/伪装分割上的精度，以及 CLIP 在对抗攻击下的鲁棒性。

**[Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physical_systems.md)**

:   系统评估通用自监督方法在时空物理系统上学习物理相关表征的能力，发现在潜空间做预测的 JEPA 显著优于像素级重建的 MAE 和自回归模型，接近专用物理建模方法 DISCO。

**[ScaleLSD: Scalable Deep Line Segment Detection Streamlined](scalelsd_scalable_deep_line_segment_detection_streamlined.md)**

:   ScaleLSD 通过精简线段检测架构（引入 HAT 诱导的提案验证）和设计高效伪标签生成管线（LSD-Rectifier），首次实现了在1000万无标注图像上的大规模自监督线段检测训练，在零样本评测中全面超越经典非深度 LSD 方法。

**[SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning](smile_infusing_spatial_and_motion_semantics_in_masked_video_learning.md)**

:   提出 SMILE，通过合成运动增强（在视频上叠加沿随机轨迹运动的分割物体）和 CLIP 特征重建目标来增强掩码视频建模，结合轨迹引导的掩码策略，在 K400 线性探测上大幅提升至 56.2%（前 SOTA 47.5%）。

**[Spectral State Space Model for Rotation-Invariant Visual Representation Learning](spectral_state_space_model_for_rotation-invariant_visual_representation_learning.md)**

:   提出 Spectral VMamba，用谱图拉普拉斯的特征向量排序 patch 遍历顺序（替代预定义扫描线），结合旋转特征归一化器（RFN，聚合 4 个正则旋转的特征），在 miniImageNet 上达到 87.86% 准确率且对正则旋转完全不变。

**[Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)**

:   提出 TPSNet，利用文本-相位双先验解决无监督跨域图像检索：域提示（text prior）提供比伪标签更精确的语义监督，相位特征（phase prior）实现保持语义的域不变对齐，两者通过交叉注意力协同融合。

**[Transformers without Normalization](transformers_without_normalization.md)**

:   发现 LayerNorm 的输入-输出映射呈 tanh 形状，提出 Dynamic Tanh (DyT) 作为归一化层的即插即用替代：$\text{DyT}(x) = \gamma \odot \tanh(\alpha x) + \beta$，在视觉/语言/扩散/语音等多任务中与 LN 性能持平甚至更优。

**[UniSTD: Towards Unified Spatio-Temporal Learning Across Diverse Disciplines](unistd_towards_unified_spatio-temporal_learning_across_diverse_disciplines.md)**

:   提出 UniSTD 框架，利用标准 Transformer + 自适应秩混合专家（RA-MoE）+ 轻量时序模块，实现了一个模型同时处理 4 个学科 10 个时空预测任务且无性能损失，在多任务联合训练中比现有方法高出 18.8 PSNR。
