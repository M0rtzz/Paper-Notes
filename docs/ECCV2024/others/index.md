---
title: >-
  ECCV2024 其他方向 25篇论文解读
description: >-
  25篇ECCV2024 其他方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**🎞️ ECCV2024** · 共 **25** 篇

**[A Closer Look At Gan Priors Exploiting Intermediate Features](a_closer_look_at_gan_priors_exploiting_intermediate_features.md)**

:   提出 IF-GMI，将预训练 StyleGAN2 的生成器拆解为多个 block，在中间特征层逐层优化（配合 $\ell_1$ 球约束防止图像崩塌），把模型反演攻击的搜索空间从潜码扩展到中间特征，在 OOD 场景下攻击准确率提升高达 38.8%。

**[A Framework For Efficient Model Evaluation Through Stratific](a_framework_for_efficient_model_evaluation_through_stratific.md)**

:   提出一个统计框架，通过分层（stratification）、采样设计（sampling）和估计器（estimation）三个组件的协同设计，在仅标注少量测试样本的情况下精确估计CV模型准确率，最高可实现10倍的效率增益（即用1/10的标注量达到同等精度）。

**[A Highquality Robust Diffusion Framework For Corrupted Datas](a_highquality_robust_diffusion_framework_for_corrupted_datas.md)**

:   提出 RDUOT 框架，首次将非平衡最优传输(UOT)融入扩散模型(DDGAN)中，通过学习 $q(x_0|x_t)$ 而非 $q(x_{t-1}|x_t)$ 来有效过滤训练数据中的离群值，在污染数据集上实现鲁棒生成的同时，在干净数据集上也超越了 DDGAN 基线。

**[Abc Easy As 123 A Blind Counter For Exemplar-Free Multi-Class Class-Agnostic Cou](abc_easy_as_123_a_blind_counter_for_exemplar-free_multi-class_class-agnostic_cou.md)**

:   提出首个无需样例图像即可同时计数图像中多类未知物体的方法ABC123，通过ViT回归多通道密度图+匈牙利匹配训练+SAM示例发现机制，在自建合成数据集MCAC上大幅超越需要样例的方法，且能泛化到FSC-147真实数据集。

**[Action2Sound Ambientaware Generation Of Action Sounds From E](action2sound_ambientaware_generation_of_action_sounds_from_e.md)**

:   提出 AV-LDM，通过在训练时引入同一视频不同时间段的音频作为环境音条件，隐式解耦前景动作声和背景环境音，结合检索增强生成(RAG)在推理时选择合适的环境音条件，在 Ego4D 和 EPIC-KITCHENS 上大幅超越已有方法。

**[Active Generation For Image Classification](active_generation_for_image_classification.md)**

:   ActGen将主动学习思想引入扩散模型数据增强，通过识别分类器的错分样本并以注意力掩码引导+梯度对抗引导生成"难样本"，仅用10%的合成数据量即超越了此前需要近等量合成数据的方法，在ImageNet上ResNet-50获得+2.26%的精度提升。

**[Adaptive Highfrequency Transformer For Diverse Wildlife Reid](adaptive_highfrequency_transformer_for_diverse_wildlife_reid.md)**

:   提出自适应高频Transformer（AdaFreq），通过频域混合增强、目标感知的高频token动态选择、特征均衡损失三大策略，将高频信息（毛皮纹理、轮廓边缘等）统一用于多种野生动物的重识别，在8个跨物种数据集上超越现有ReID方法。

**[Bidirectional Uncertainty-Based Active Learning For Open-Set Annotation](bidirectional_uncertainty-based_active_learning_for_open-set_annotation.md)**

:   提出 BUAL 框架，通过 Random Label Negative Learning 将未知类样本推向高置信区域、已知类样本推向低置信区域，结合双向不确定性采样策略，在开放集场景下有效选出高信息量的已知类样本。

**[Dc-Solver Improving Predictor-Corrector Diffusion Sampler Via Dynamic Compensati](dc-solver_improving_predictor-corrector_diffusion_sampler_via_dynamic_compensati.md)**

:   提出 DC-Solver，通过动态补偿（Dynamic Compensation）缓解 predictor-corrector 扩散采样器中的 misalignment 问题，仅需 10 个数据点即可优化补偿比率，并通过级联多项式回归（CPR）实现对未见 NFE/CFG 配置的即时泛化。

**[Et The Exceptional Trajectories Text-To-Camera-Trajectory Generation With Charac](et_the_exceptional_trajectories_text-to-camera-trajectory_generation_with_charac.md)**

:   提出首个从真实电影中提取的**相机-角色轨迹数据集 E.T.**（115K 样本，11M 帧），以及基于扩散模型的 **Director** 方法，能根据文本描述和角色轨迹生成复杂的相机运动轨迹，同时设计了 **CLaTr** 对比嵌入用于轨迹生成质量评估。

**[Event-Based Mosaicing Bundle Adjustment](event-based_mosaicing_bundle_adjustment.md)**

:   提出 EMBA，首个针对纯旋转事件相机的光度 Bundle Adjustment 方法，利用线性化事件生成模型将问题形式化为正则化非线性最小二乘优化，并利用法方程矩阵的块对角稀疏结构设计高效求解器，同时优化相机旋转轨迹和全景梯度图。

**[Foster Adaptivity And Balance In Learning With Noisy Labels](foster_adaptivity_and_balance_in_learning_with_noisy_labels.md)**

:   提出SED方法，通过自适应且类别平衡的样本选择与重加权机制来应对标签噪声问题，在无需预定义阈值等先验知识的前提下，在合成和真实噪声数据集上取得SOTA性能。

**[Freeaugment Data Augmentation Search Across All Degrees Of Freedom](freeaugment_data_augmentation_search_across_all_degrees_of_freedom.md)**

:   提出 FreeAugment，首个能够同时全局优化数据增强策略的四个自由度（变换数量/类型/顺序/强度）的全可微搜索方法，通过 Gumbel-Softmax 学习深度分布、Gumbel-Sinkhorn 学习排列分布来避免重复采样，在多个基准上取得 SOTA。

**[Hpff Hierarchical Locally Supervised Learning With Patch Feature Fusion](hpff_hierarchical_locally_supervised_learning_with_patch_feature_fusion.md)**

:   提出 HPFF，通过层次化局部监督学习（HiLo，将网络划分为独立+级联两级局部模块）和 Patch 特征融合（PFF，将辅助网络的输入切块计算再平均）解决局部学习中的模块间信息缺失和 GPU 内存占用过高问题，在多个数据集上显著超越已有局部学习方法并接近甚至超越 BP。

**[Intrinsic Single-Image Hdr Reconstruction](intrinsic_single-image_hdr_reconstruction.md)**

:   > 提出基于内在图像分解（intrinsic decomposition）的 HDR 重建方法，将问题分解为明暗域（shading）的动态范围扩展和反照率域（albedo）的颜色恢复两个子任务，分别训练网络以提升重建质量。

**[Mahalanobis Distance-Based Multi-View Optimal Transport For Multi-View Crowd Loc](mahalanobis_distance-based_multi-view_optimal_transport_for_multi-view_crowd_loc.md)**

:   提出基于马氏距离的多视角最优传输损失（M-MVOT），通过视线方向和目标到相机的距离自适应调整传输代价，首次将点监督最优传输引入多视角人群定位任务，显著超越基于密度图MSE损失的方法。

**[Online Temporal Action Localization With Memory-Augmented Transformer](online_temporal_action_localization_with_memory-augmented_transformer.md)**

:   提出 MATR（Memory-Augmented Transformer），通过记忆队列存储过去片段的特征来利用长期上下文，并采用分离的 Start/End Transformer 解码器进行动作实例定位，在在线时序动作定位（On-TAL）任务上取得 SOTA，甚至可比肩部分离线方法。

**[Partcraft Crafting Creative Objects By Parts](partcraft_crafting_creative_objects_by_parts.md)**

:   提出 PartCraft，首次实现了基于部件选择的文本到图像生成控制——用户可以从不同物体中"挑选"各部件（如鸟的头、翅膀、身体），模型将它们自然地组合为一个全新且结构合理的创意物体。

**[Power Variable Projection For Initialization-Free Large-Scale Bundle Adjustment](power_variable_projection_for_initialization-free_large-scale_bundle_adjustment.md)**

:   提出 Power Variable Projection (PoVar) 算法，将幂级数展开方法扩展到变量投影（VarPro）框架，并进一步推广到黎曼流形优化，首次实现了无初始化大规模光束法平差（BA）的高效求解。

**[Raindrop Clarity A Dual-Focused Dataset For Day And Night Raindrop Removal](raindrop_clarity_a_dual-focused_dataset_for_day_and_night_raindrop_removal.md)**

:   提出了一个大规模真实世界雨滴去除数据集 Raindrop Clarity，包含15,186组高质量图像对/三元组，首次涵盖雨滴聚焦（清晰雨滴+模糊背景）和夜间雨滴两种现有数据集缺失的场景。

**[Shifted Autoencoders For Point Annotation Restoration In Object Counting](shifted_autoencoders_for_point_annotation_restoration_in_object_counting.md)**

:   提出**Shifted AutoEncoders (SAE)**，一种受MAE启发的点标注修复方法：通过随机位移点标注后训练UNet恢复，使模型学到"通用位置知识"而忽略个体标注噪声；用训练好的SAE修复原始标注使其更一致，可为任意计数模型（密度图/定位型）稳定提升性能，在9个数据集上创下新记录。

**[Spatio-Temporal Proximity-Aware Dual-Path Model For Panoramic Activity Recogniti](spatio-temporal_proximity-aware_dual-path_model_for_panoramic_activity_recogniti.md)**

:   提出 SPDP-Net，通过时空邻近性建模个体间社会关系，并利用双路径 Transformer (DPATr) 架构在个体-全局和个体-社交两条路径上协同识别多粒度活动，在 JRDB-PAR 数据集上以 46.5% overall F1 大幅刷新 SOTA。

**[Superpixel-Informed Implicit Neural Representation For Multi-Dimensional Data](superpixel-informed_implicit_neural_representation_for_multi-dimensional_data.md)**

:   提出超像素引导的隐式神经表示（S-INR），用广义超像素替代像素作为INR的基本单元，通过专属注意力MLP和共享字典矩阵两个模块，充分挖掘广义超像素内部和之间的语义信息，在图像重建/补全/去噪以及点数据恢复等任务上超越现有INR方法。

**[Teaching Tailored To Talent Adverse Weather Restoration](teaching_tailored_to_talent_adverse_weather_restoration.md)**

:   提出 T3-DiffWeather，采用 prompt pool 自主组合子 prompt 构建天气退化信息，结合 Depth-Anything 约束的通用 prompt 提供场景信息，以对比 prompt 损失约束两类 prompt，在恶劣天气图像恢复任务上仅用 WeatherDiffusion 十分之一的采样步数达到 SOTA。

**[Teaching Tailored To Talent Adverse Weather Restoration Via Prompt Pool And Dept](teaching_tailored_to_talent_adverse_weather_restoration_via_prompt_pool_and_dept.md)**

:   提出 T3-DiffWeather，一种基于 diffusion 的 all-in-one 恶劣天气恢复框架，通过 prompt pool 让网络自主组合 sub-prompts 构建实例级 weather-prompts 来建模多样化天气退化，同时利用 Depth-Anything 特征约束 general prompts 来建模场景信息，仅需 2 步采样即达到 SOTA，计算量仅为 WeatherDiffusion 的 1/52。
