---
title: >-
  ECCV2024 其他方向 65篇论文解读
description: >-
  65篇ECCV2024 其他方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📂 其他

**🎞️ ECCV2024** · **65** 篇论文解读

**[3DFG-PIFu: 3D Feature Grids for Human Digitization from Sparse Views](3dfg-pifu_3d_feature_grids_for_human_digitization_from_sparse_views.md)**

:   本文提出 3DFG-PIFu，通过引入3D特征网格（3D Feature Grids）在整个 pipeline 中全局融合多视图特征，替代传统逐点局部融合方式，并结合迭代网格精炼机制和基于 SDF 的 SMPL-X 特征，显著超越现有稀疏视图人体数字化 SOTA 方法。

**[A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks](a_closer_look_at_gan_priors_exploiting_intermediate_features.md)**

:   提出 IF-GMI，将预训练 StyleGAN2 的生成器拆解为多个 block，在中间特征层逐层优化（配合 $\ell_1$ 球约束防止图像崩塌），把模型反演攻击的搜索空间从潜码扩展到中间特征，在 OOD 场景下攻击准确率提升高达 38.8%。

**[A Direct Approach to Viewing Graph Solvability](a_direct_approach_to_viewing_graph_solvability.md)**

:   本文对视图图（Viewing Graph）可解性问题提出了一种比以往更直接的新形式化方法，引入了新概念用于理解实际 SfM 图的可解性，并给出了更高效的不可解情况检测与分解算法。

**[A Framework for Efficient Model Evaluation through Stratification, Sampling, and Estimation](a_framework_for_efficient_model_evaluation_through_stratific.md)**

:   提出一个统计框架，通过分层（stratification）、采样设计（sampling）和估计器（estimation）三个组件的协同设计，在仅标注少量测试样本的情况下精确估计CV模型准确率，最高可实现10倍的效率增益（即用1/10的标注量达到同等精度）。

**[A High-Quality Robust Diffusion Framework for Corrupted Dataset](a_highquality_robust_diffusion_framework_for_corrupted_datas.md)**

:   提出 RDUOT 框架，首次将非平衡最优传输(UOT)融入扩散模型(DDGAN)中，通过学习 $q(x_0|x_t)$ 而非 $q(x_{t-1}|x_t)$ 来有效过滤训练数据中的离群值，在污染数据集上实现鲁棒生成的同时，在干净数据集上也超越了 DDGAN 基线。

**[ABC Easy as 123: A Blind Counter for Exemplar-Free Multi-Class Class-Agnostic Counting](abc_easy_as_123_a_blind_counter_for_exemplar-free_multi-class_class-agnostic_cou.md)**

:   提出首个无需样例图像即可同时计数图像中多类未知物体的方法ABC123，通过ViT回归多通道密度图+匈牙利匹配训练+SAM示例发现机制，在自建合成数据集MCAC上大幅超越需要样例的方法，且能泛化到FSC-147真实数据集。

**[Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos](action2sound_ambientaware_generation_of_action_sounds_from_e.md)**

:   提出 AV-LDM，通过在训练时引入同一视频不同时间段的音频作为环境音条件，隐式解耦前景动作声和背景环境音，结合检索增强生成(RAG)在推理时选择合适的环境音条件，在 Ego4D 和 EPIC-KITCHENS 上大幅超越已有方法。

**[Active Generation for Image Classification](active_generation_for_image_classification.md)**

:   本文提出ActGen，将主动学习思想融入扩散模型的图像生成过程，通过识别模型误分类的验证样本作为引导图像、结合注意力引导和基于梯度的生成控制，仅用10%的合成图像即可在ImageNet上实现+2.26%的准确率提升，超过了使用94%合成数据的先前方法。

**[Adaptive High-Frequency Transformer for Diverse Wildlife Re-Identification](adaptive_highfrequency_transformer_for_diverse_wildlife_reid.md)**

:   提出自适应高频Transformer（AdaFreq），通过频域混合增强、目标感知的高频token动态选择、特征均衡损失三大策略，将高频信息（毛皮纹理、轮廓边缘等）统一用于多种野生动物的重识别，在8个跨物种数据集上超越现有ReID方法。

**[AddMe: Zero-Shot Group-Photo Synthesis by Inserting People Into Scenes](addme_zero-shot_group-photo_synthesis_by_inserting_people_into_scenes.md)**

:   本文提出 AddMe，一个基于扩散模型的零样本人像生成器，通过身份解耦适配器和增强型人像注意力模块，能够将给定的人像自然地插入到现有场景图像的指定位置，同时保持身份一致性和群体交互的合理性。

**[ADMap: Anti-disturbance Framework for Vectorized HD Map Construction](admap_anti-disturbance_framework_for_vectorized_hd_map_construction.md)**

:   本文提出 ADMap 框架，通过多尺度感知颈部(MPN)、实例交互注意力(IIA)和矢量方向差异损失(VDDL)三个模块，从实例间和实例内两个层面级联式监控点序列预测过程，有效缓解了矢量化高精地图构建中的点序列抖动/锯齿问题，在 nuScenes 和 Argoverse2 上取得了 SOTA 性能。

**[Align before Collaborate: Mitigating Feature Misalignment for Robust Multi-Agent Perception](align_before_collaborate_mitigating_feature_misalignment_for_robust_multi-agent_.md)**

:   提出NEAT——一种模型无关的轻量级插件，通过重要性引导的查询提议、可变形特征对齐和区域交叉注意力增强三个模块，显式解决协同感知中因位姿误差和通信延迟导致的特征级空间错位问题，在四个协同3D检测数据集的噪声设置下为多种基线方法带来一致性增益。

**[An Incremental Unified Framework for Small Defect Inspection](an_incremental_unified_framework_for_small_defect_inspection.md)**

:   提出增量统一框架IUF，首次将增量学习集成到统一重建式缺陷检测方法中，通过目标感知自注意力（OASA）建立语义边界、语义压缩损失（SCL）压缩非主要语义空间、以及基于SVD的权重更新策略保护旧对象特征，在MVTec-AD和VisA上实现图像级和像素级的SOTA增量缺陷检测性能。

**[AttnZero: Efficient Attention Discovery for Vision Transformers](attnzero_efficient_attention_discovery_for_vision_transformers.md)**

:   本文提出 AttnZero，首个自动发现高效注意力模块的框架，通过构建包含六类计算图和丰富算子的搜索空间、利用进化算法进行多目标搜索，自动发现了适用于多种 ViT 的线性注意力公式，在 DeiT/PVT/Swin/CSwin 上分别达到 74.9%/78.1%/82.1%/82.9% 的 ImageNet top-1 准确率，并构建了包含 2000 种注意力变体的 Attn-Bench-101 基准。

**[Auto-GAS: Automated Proxy Discovery for Training-Free Generative Architecture Search](auto-gas_automated_proxy_discovery_for_training-free_generative_architecture_sea.md)**

:   本文提出 Auto-GAS，首个面向生成模型（GAN）的免训练架构搜索框架，通过自动发现并优化零成本代理指标来替代传统训练式搜索，实现 110 倍搜索加速，同时保持与训练式方法相当的生成质量。

**[Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation](bidirectional_uncertainty-based_active_learning_for_open-set_annotation.md)**

:   提出 BUAL 框架，通过 Random Label Negative Learning 将未知类样本推向高置信区域、已知类样本推向低置信区域，结合双向不确定性采样策略，在开放集场景下有效选出高信息量的已知类样本。

**[Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals](brain_netflix_scaling_data_to_reconstruct_videos_from_brain_signals.md)**

:   本文提出了一种从功能磁共振成像（fMRI）信号重建视频的新方法，通过多数据集多被试训练和三阶段pipeline，利用预训练的文本到视频和视频到视频模型，实现了跨数据集和跨被试的SOTA视频重建能力。

**[CLR-GAN: Improving GANs Stability and Quality via Consistent Latent Representation and Reconstruction](clr-gan_improving_gans_stability_and_quality_via_consistent_latent_representatio.md)**

:   本文提出了CLR-GAN训练范式，通过让判别器恢复生成器的预定义隐码、让生成器重建真实输入，建立了G和D隐空间之间的一致性约束，使GAN训练更公平稳定，在CIFAR10上FID提升31.22%，在AFHQ-Cat上提升39.5%。

**[COIN-Matting: Confounder Intervention for Image Matting](coin-matting_confounder_intervention_for_image_matting.md)**

:   本文从因果推断角度分析图像抠图任务中的数据集偏差问题，识别出对比度偏差和透明度偏差两种典型偏差及其根源——混淆因子，并通过后门调整提出模型无关的 COIN 抠图框架，显著缓解偏差影响、提升现有抠图模型性能。

**[DC-Solver: Improving Predictor-Corrector Diffusion Sampler via Dynamic Compensation](dc-solver_improving_predictor-corrector_diffusion_sampler_via_dynamic_compensati.md)**

:   提出 DC-Solver，通过动态补偿（Dynamic Compensation）缓解 predictor-corrector 扩散采样器中的 misalignment 问题，仅需 10 个数据点即可优化补偿比率，并通过级联多项式回归（CPR）实现对未见 NFE/CFG 配置的即时泛化。

**[De-confounded Gaze Estimation](de-confounded_gaze_estimation.md)**

:   本文提出基于因果干预的视线估计框架 FSCI，通过特征分离将视线相关特征与身份/光照等无关特征解耦，并利用动态混杂因子库对无关特征进行因果干预，在跨域设置下较基线提升36.2%、较SOTA提升11.5%。

**[Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning](dropout_mixture_low-rank_adaptation_for_visual_parameters-efficient_fine-tuning.md)**

:   本文提出 DMLoRA（Dropout-Mixture Low-Rank Adaptation），通过引入多分支上下投影结构并在训练过程中逐步dropout分支来平衡精度与正则化，配合两阶段学习缩放因子策略优化每层的缩放系数，在VTAB-1k和FGVC视觉微调基准上取得SOTA性能且推理无额外开销。

**[Elegantly Written: Disentangling Writer and Character Styles for Enhancing Online Chinese Handwriting](elegantly_written_disentangling_writer_and_character_styles_for_enhancing_online.md)**

:   本文提出了一种基于序列模型的在线中文手写轨迹美化方法，通过交叉注意力机制解耦书写者风格和字符结构风格，将用户潦草的手写轨迹转化为保持个人风格的美观书写，同时通过笛卡尔积分解有效去除冗余风格特征。

**[Enhancing Optimization Robustness in 1-bit Neural Networks through Stochastic Sign Descent](enhancing_optimization_robustness_in_1-bit_neural_networks_through_stochastic_si.md)**

:   提出Diode优化器，专为二值神经网络（BNN）设计，通过利用梯度符号的低阶矩估计实现无潜在权重（latent-weight-free）的参数更新，在ImageNet上将BNext-18的Top-1准确率提升0.96%且训练迭代次数减少8倍，并在NLP任务上达到新SOTA。

**[ET: The Exceptional Trajectories - Text-to-Camera-Trajectory Generation with Character Awareness](et_the_exceptional_trajectories_text-to-camera-trajectory_generation_with_charac.md)**

:   提出首个从真实电影中提取的**相机-角色轨迹数据集 E.T.**（115K 样本，11M 帧），以及基于扩散模型的 **Director** 方法，能根据文本描述和角色轨迹生成复杂的相机运动轨迹，同时设计了 **CLaTr** 对比嵌入用于轨迹生成质量评估。

**[Event-based Mosaicing Bundle Adjustment](event-based_mosaicing_bundle_adjustment.md)**

:   提出 EMBA，首个针对纯旋转事件相机的光度 Bundle Adjustment 方法，利用线性化事件生成模型将问题形式化为正则化非线性最小二乘优化，并利用法方程矩阵的块对角稀疏结构设计高效求解器，同时优化相机旋转轨迹和全景梯度图。

**[Exploring Guided Sampling of Conditional GANs](exploring_guided_sampling_of_conditional_gans.md)**

:   本文提出在条件GAN中引入类似扩散模型的引导采样（guided sampling）策略，通过隐空间向量运算估计数据-条件联合分布，无需预训练分类器或学习无条件模型，即可显著提升GAN生成质量，将ImageNet 64×64上的FID从8.87降至4.37。

**[FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information](fisherrf_active_view_selection_and_mapping_with_radiance_fields_using_fisher_inf.md)**

:   本文提出FisherRF，利用Fisher信息直接量化辐射场（Radiance Fields）模型参数的观测信息量，通过最大化期望信息增益（Expected Information Gain）选择最优视角，在视角选择、主动建图和不确定性量化三个任务上均达到SOTA，且通过稀疏性利用和自定义CUDA核实现了70 fps的视角评估速度。

**[Foster Adaptivity and Balance in Learning with Noisy Labels](foster_adaptivity_and_balance_in_learning_with_noisy_labels.md)**

:   提出SED方法，通过自适应且类别平衡的样本选择与重加权机制来应对标签噪声问题，在无需预定义阈值等先验知识的前提下，在合成和真实噪声数据集上取得SOTA性能。

**[Free-Viewpoint Video of Outdoor Sports Using a Flying Camera](free-viewpoint_video_of_outdoor_sports_using_a_flying_camera.md)**

:   提出了一种基于无人机RGB相机的系统，能够重建户外运动场景中的4D动态人体和3D无界背景，实现任意时刻的自由视点视频渲染。

**[FreeAugment: Data Augmentation Search Across All Degrees of Freedom](freeaugment_data_augmentation_search_across_all_degrees_of_freedom.md)**

:   提出 FreeAugment，首个能够同时全局优化数据增强策略的四个自由度（变换数量/类型/顺序/强度）的全可微搜索方法，通过 Gumbel-Softmax 学习深度分布、Gumbel-Sinkhorn 学习排列分布来避免重复采样，在多个基准上取得 SOTA。

**[Functional Transform-Based Low-Rank Tensor Factorization for Multi-Dimensional Data Recovery](functional_transform-based_low-rank_tensor_factorization_for_multi-dimensional_d.md)**

:   提出了基于函数变换的低秩张量分解方法（FLRTF），利用隐式神经表示替代传统离散变换来捕获数据在第三维度上的连续平滑性，有效解决时间/光谱退化问题。

**[Gaze Target Detection Based on Head-Local-Global Coordination](gaze_target_detection_based_on_head-local-global_coordination.md)**

:   提出了一种基于头部-局部-全局三视图协调的注视目标检测方法，通过引入基于FOV（视野范围）的局部视图，并设计全局-局部位置与表示一致性机制，显著提升了注视目标预测的准确性。

**[GazeXplain: Learning to Predict Natural Language Explanations of Visual Scanpaths](gazexplain_learning_to_predict_natural_language_explanations_of_visual_scanpaths.md)**

:   提出GazeXplain，首次将视觉扫描路径预测与自然语言解释结合，通过注意力-语言解码器、语义对齐机制和跨数据集联合训练，实现对人类注视行为的可解释预测。

**[HiEI: A Universal Framework for Generating High-quality Emerging Images from Natural Images](hiei_a_universal_framework_for_generating_high-quality_emerging_images_from_natu.md)**

:   本文提出了一个通用框架 HiEI，通过人类中心的颜色量化模块（TTNet）、感知难度控制模块（PDC）和模板矢量化模块（TV），将自然图像转化为高质量的新兴图像（Emerging Images），在内容和风格质量上超越现有方法，同时可有效对抗深度视觉模型的攻击，适用于 CAPTCHA 机制。

**[High-Fidelity 3D Textured Shapes Generation by Sparse Encoding and Adversarial Decoding](high-fidelity_3d_textured_shapes_generation_by_sparse_encoding_and_adversarial_d.md)**

:   本文提出了一种基于稀疏编码模块和对抗解码模块的 3D 纹理形状生成框架，通过对 StableDiffusion 的最小适配扩展到 3D 领域，在 ShapeNet 和 G-Objaverse（200K 样本）上实现了开放词汇的高保真 3D 生成，超越了现有 SOTA 方法。

**[HPFF: Hierarchical Locally Supervised Learning with Patch Feature Fusion](hpff_hierarchical_locally_supervised_learning_with_patch_feature_fusion.md)**

:   提出 HPFF，通过层次化局部监督学习（HiLo，将网络划分为独立+级联两级局部模块）和 Patch 特征融合（PFF，将辅助网络的输入切块计算再平均）解决局部学习中的模块间信息缺失和 GPU 内存占用过高问题，在多个数据集上显著超越已有局部学习方法并接近甚至超越 BP。

**[I Can't Believe It's Not Scene Flow!](i_canapost_believe_itaposs_not_scene_flow.md)**

:   揭示现有场景流方法在行人等小目标上的灾难性失败被现有评估指标所掩盖，提出类别感知且速度归一化的Bucket Normalized EPE评估协议，以及一个简单但SOTA的TrackFlow基线（检测器+跟踪器生成场景流），在行人运动描述上实现1.5倍提升。

**[Image Demoiréing in RAW and sRGB Domains](image_demoiréing_in_raw_and_srgb_domains.md)**

:   提出RRID框架联合利用RAW和sRGB双域数据进行图像去摩尔纹，设计了带GFM（门控反馈）和FSM（频域选择）的SCDM去摩尔纹模块，以及RGISP实现设备相关ISP学习辅助颜色恢复，在PSNR上超越SOTA 0.62dB。

**[Intrinsic Single-Image HDR Reconstruction](intrinsic_single-image_hdr_reconstruction.md)**

:   > 提出基于内在图像分解（intrinsic decomposition）的 HDR 重建方法，将问题分解为明暗域（shading）的动态范围扩展和反照率域（albedo）的颜色恢复两个子任务，分别训练网络以提升重建质量。

**[Learning Anomalies with Normality Prior for Unsupervised Video Anomaly Detection](learning_anomalies_with_normality_prior_for_unsupervised_video_anomaly_detection.md)**

:   本文提出了一种基于"正常性先验"的无监督视频异常检测方法（LANP），通过利用"视频首尾段大概率为正常事件"这一数据无关先验知识生成初始正常标签，再通过正常性传播将正常知识扩散到全部片段，最后配合损失重加权策略训练异常检测器，在 ShanghaiTech 和 UCF-Crime 上取得了优异性能。

**[Mahalanobis Distance-Based Multi-View Optimal Transport for Multi-View Crowd Localization](mahalanobis_distance-based_multi-view_optimal_transport_for_multi-view_crowd_loc.md)**

:   提出基于马氏距离的多视角最优传输损失（M-MVOT），通过视线方向和目标到相机的距离自适应调整传输代价，首次将点监督最优传输引入多视角人群定位任务，显著超越基于密度图MSE损失的方法。

**[MemBN: Robust Test-Time Adaptation via Batch Norm with Statistics Memory](membn_robust_test-time_adaptation_via_batch_norm_with_statistics_memory.md)**

:   本文提出 MemBN（Memory-based Batch Normalization），通过在每个 BN 层中维护统计量记忆队列并设计专用的记忆管理与聚合算法，使得 TTA 方法在各种批量大小下都能稳健估计测试域的统计量，大幅提升小批量场景下的准确率和鲁棒性。

**[Momentum Auxiliary Network for Supervised Local Learning](momentum_auxiliary_network_for_supervised_local_learning.md)**

:   本文提出动量辅助网络（MAN），通过指数移动平均（EMA）将相邻局部块的参数信息传递到当前块的辅助网络，并引入可学习偏置弥补跨块特征差异，解决了监督局部学习中块间信息交换缺失导致的"短视"问题，在 ImageNet 上以不到 E2E 训练一半的 GPU 显存实现更高性能。

**[Non-parametric Sensor Noise Modeling and Synthesis](non-parametric_sensor_noise_modeling_and_synthesis.md)**

:   本文提出一种非参数传感器噪声模型，通过直接从实拍图像中为每个亮度级别构建概率质量函数(PMF)来建模真实噪声分布，无需假设特定噪声分布形式，并提出了ISO插值和在含噪图像上合成噪声的方法，在下游去噪任务上显著优于现有参数化噪声模型。

**[Object-Aware NIR-to-Visible Translation](object-aware_nir-to-visible_translation.md)**

:   本文提出一种对象感知的近红外(NIR)到可见光图像翻译框架，通过将可见光图像分解为与对象无关的光照分量和对象特定的反射分量分别处理，结合分割先验知识，在缺乏大规模配对数据的条件下实现了高质量的NIR彩色化，并构建了首个完全对齐的NIR-可见光大规模配对数据集。

**[Online Temporal Action Localization with Memory-Augmented Transformer](online_temporal_action_localization_with_memory-augmented_transformer.md)**

:   本文提出 MATR（Memory-Augmented Transformer），通过记忆队列选择性地保存历史片段特征来建模长期上下文，并采用双 Transformer 解码器分别定位动作的结束和起始时间，在 THUMOS14 和 MUSES 两个在线时序动作定位基准上刷新了 SOTA，甚至可与部分离线方法媲美。

**[Operational Open-Set Recognition and PostMax Refinement](operational_open-set_recognition_and_postmax_refinement.md)**

:   本文提出了一种面向实际部署场景的开放集识别评估指标 OOSA（Operational Open-Set Accuracy）以及后处理算法 PostMax，通过对最大类别 logit 进行深度特征幅度归一化和广义 Pareto 分布映射，将 logit 转化为合理的概率估计，在大规模评估中取得了统计显著的 SOTA 性能。

**[PartCraft: Crafting Creative Objects by Parts](partcraft_crafting_creative_objects_by_parts.md)**

:   提出 PartCraft，首次实现了基于部件选择的文本到图像生成控制——用户可以从不同物体中"挑选"各部件（如鸟的头、翅膀、身体），模型将它们自然地组合为一个全新且结构合理的创意物体。

**[Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment](power_variable_projection_for_initialization-free_large-scale_bundle_adjustment.md)**

:   提出 Power Variable Projection (PoVar) 算法，将幂级数展开方法扩展到变量投影（VarPro）框架，并进一步推广到黎曼流形优化，首次实现了无初始化大规模光束法平差（BA）的高效求解。

**[Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal](raindrop_clarity_a_dual-focused_dataset_for_day_and_night_raindrop_removal.md)**

:   提出了一个大规模真实世界雨滴去除数据集 Raindrop Clarity，包含15,186组高质量图像对/三元组，首次涵盖雨滴聚焦（清晰雨滴+模糊背景）和夜间雨滴两种现有数据集缺失的场景。

**[Real-Data-Driven 2000 FPS Color Video from Mosaicked Chromatic Spikes](real-data-driven_2000_fps_color_video_from_mosaicked_chromatic_spikes.md)**

:   针对马赛克彩色脉冲相机（mosaicked chromatic spikes），提出一种完全基于真实数据驱动的 2000FPS 彩色高动态范围视频重建方法，通过自监督去噪模块和渐进式配准模块解决短时帧噪声和运动模糊问题，无需合成数据即可重建高质量高速彩色视频。

**[Rebalancing Using Estimated Class Distribution for Imbalanced Semi-Supervised Learning under Class Distribution Mismatch](rebalancing_using_estimated_class_distribution_for_imbalanced_semi-supervised_le.md)**

:   本文提出 RECD 算法，通过蒙特卡洛近似估计未标注数据的未知类别分布，基于估计分布重新平衡分类器，并引入特征聚类压缩缓解特征图不平衡，在标注-未标注数据类别分布失配的半监督学习场景中取得 SOTA 性能。

**[Rethinking Data Bias: Dataset Copyright Protection via Embedding Class-Wise Hidden Bias](rethinking_data_bias_dataset_copyright_protection_via_embedding_class-wise_hidde.md)**

:   本文提出"Undercover Bias"数据集水印方法，通过在训练数据中嵌入与目标任务无关但与标签对应的隐蔽水印图案，使未授权使用者训练的模型不自觉地学会分类这些水印，水印分类能力作为未授权使用的不可抵赖证据，实现了隐蔽、模型无关、对目标任务无损的数据集版权保护。

**[Shifted Autoencoders for Point Annotation Restoration in Object Counting](shifted_autoencoders_for_point_annotation_restoration_in_object_counting.md)**

:   提出**Shifted AutoEncoders (SAE)**，一种受MAE启发的点标注修复方法：通过随机位移点标注后训练UNet恢复，使模型学到"通用位置知识"而忽略个体标注噪声；用训练好的SAE修复原始标注使其更一致，可为任意计数模型（密度图/定位型）稳定提升性能，在9个数据集上创下新记录。

**[SpatialFormer: Towards Generalizable Vision Transformers with Explicit Spatial Understanding](spatialformer_towards_generalizable_vision_transformers_with_explicit_spatial_un.md)**

:   提出SpatialFormer架构，通过引入自适应空间token显式建模场景的全局空间关系，采用decoder-only架构与双边交叉注意力块实现上下文与空间信息的高效交互，在分类、分割和检测任务上展示了优异的泛化性和可迁移性。

**[Spatio-Temporal Proximity-Aware Dual-Path Model for Panoramic Activity Recognition](spatio-temporal_proximity-aware_dual-path_model_for_panoramic_activity_recogniti.md)**

:   提出 SPDP-Net，通过时空邻近性建模个体间社会关系，并利用双路径 Transformer (DPATr) 架构在个体-全局和个体-社交两条路径上协同识别多粒度活动，在 JRDB-PAR 数据集上以 46.5% overall F1 大幅刷新 SOTA。

**[SpectraM-PS: Spectrally Multiplexed Photometric Stereo Under Unknown Spectral Composition](spectram-ps_spectrally_multiplexed_photometric_stereo_under_unknown_spectral_com.md)**

:   提出一种无需物理模型约束的光谱复用光度立体方法（SpectraM-PS），在光源光谱组成完全未知的条件下，通过数据驱动的方式从单张RGB图像中恢复表面法线，实现了传统多次拍摄光度立体到单次拍摄的突破。

**[STSP: Spatial-Temporal Subspace Projection for Video Class-Incremental Learning](stsp_spatial-temporal_subspace_projection_for_video_class-incremental_learning.md)**

:   提出空间-时间子空间投影（STSP）方法解决视频类增量学习中的灾难性遗忘问题，通过时间子空间分类器（TSC）用正交子空间基表示每个类别，并通过空间梯度投影（SGP）将梯度约束在旧任务特征的零空间中，在HMDB51、UCF101和SSv2上达到SOTA。

**[Superpixel-Informed Implicit Neural Representation for Multi-Dimensional Data](superpixel-informed_implicit_neural_representation_for_multi-dimensional_data.md)**

:   提出超像素引导的隐式神经表示（S-INR），用广义超像素替代像素作为INR的基本单元，通过专属注意力MLP和共享字典矩阵两个模块，充分挖掘广义超像素内部和之间的语义信息，在图像重建/补全/去噪以及点数据恢复等任务上超越现有INR方法。

**[Synchronous Diffusion for Unsupervised Smooth Non-Rigid 3D Shape Matching](synchronous_diffusion_for_unsupervised_smooth_non-rigid_3d_shape_matching.md)**

:   提出同步扩散正则化方法用于无监督非刚性3D形状匹配，核心思想是"在两个形状上同步地扩散同一函数应产生一致输出"，通过这一简单而高效的正则化可以显著提升现有深度功能映射方法的匹配平滑性，在FAUST、SCAPE、TOPKIDS等多个数据集上达到SOTA。

**[Synergy of Sight and Semantics: Visual Intention Understanding with CLIP](synergy_of_sight_and_semantics_visual_intention_understanding_with_clip.md)**

:   提出了 IntCLIP 框架，通过双分支编码策略将 CLIP 中的"视觉感知"（Sight）知识迁移到"语义中心"（Semantic）的多标签意图理解任务中，结合层次化类别整合和视觉辅助聚合，在标准 MIU benchmark 和图像情感识别任务上显著超越 SOTA。

**[Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint](teaching_tailored_to_talent_adverse_weather_restoration.md)**

:   提出 T3-DiffWeather，采用 prompt pool 自主组合子 prompt 构建天气退化信息，结合 Depth-Anything 约束的通用 prompt 提供场景信息，以对比 prompt 损失约束两类 prompt，在恶劣天气图像恢复任务上仅用 WeatherDiffusion 十分之一的采样步数达到 SOTA。

**[Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint](teaching_tailored_to_talent_adverse_weather_restoration_via_prompt_pool_and_dept.md)**

:   提出 T3-DiffWeather，一种基于 diffusion 的 all-in-one 恶劣天气恢复框架，通过 prompt pool 让网络自主组合 sub-prompts 构建实例级 weather-prompts 来建模多样化天气退化，同时利用 Depth-Anything 特征约束 general prompts 来建模场景信息，仅需 2 步采样即达到 SOTA，计算量仅为 WeatherDiffusion 的 1/52。

**[Wavelength-Embedding-guided Filter-Array Transformer for Spectral Demosaicing](wavelength-embedding-guided_filter-array_transformer_for_spectral_demosaicing.md)**

:   本文提出 WeFAT，通过波长嵌入引导的多头自注意力（We-MSA）赋予模型"波长记忆"能力，配合滤波器阵列注意力机制（MaM）聚焦高质量光谱区域，仅在 ARAD 数据集上训练就能在不同相机和不同光谱分布下保持稳定性能，超越现有 SOTA。
