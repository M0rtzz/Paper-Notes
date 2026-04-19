---
title: >-
  CVPR2026 LLM评测方向 20篇论文解读
description: >-
  20篇CVPR2026 LLM评测方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM评测

**📷 CVPR2026** · **20** 篇论文解读

**[AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks](adabet_gradient-free_layer_selection_for_efficient_training_of_deep_neural_netwo.md)**

:   提出 AdaBet，一种基于代数拓扑（第一 Betti 数 $b_1$）的无梯度层选择方法，仅通过前向传播计算每层激活空间的拓扑复杂度来决定哪些层需要微调，无需标签、梯度或反向传播，在 ResNet50/VGG16/MobileNetV2/ViT-B16 上以仅 10% 层微调达到优于全量训练的准确率，同时峰值内存降低约 40%。

**[Cross-Scale Pansharpening via ScaleFormer and the PanScale Benchmark](cross-scale_pansharpening_via_scaleformer_and_the_panscale_benchmark.md)**

:   提出首个跨尺度全色锐化数据集PanScale和评测基准PanScale-Bench，以及ScaleFormer框架——将分辨率变化重新解释为序列长度变化，通过Scale-Aware Patchify分桶采样+解耦空间-序列建模+RoPE实现跨尺度泛化。

**[CryoHype: Reconstructing a Thousand Cryo-EM Structures with Transformer-Based Hypernetworks](cryohype_reconstructing_a_thousand_cryo-em_structures_with_transformer-based_hyp.md)**

:   提出 CryoHype，一种基于 Transformer 超网络的冷冻电镜重建方法，通过动态调整隐式神经表示（INR）的权重来减少参数共享，首次实现了从无标签冷冻电镜图像中同时重建 1000 种不同蛋白质结构。

**[Enhancing Out-of-Distribution Detection with Extended Logit Normalization](enhancing_out-of-distribution_detection_with_extended_logit_normalization.md)**

:   本文发现 LogitNorm 在训练中会导致两种特征坍塌（维度坍塌和原点坍塌），提出了一种无超参数的 Extended Logit Normalization（ELogitNorm），用特征到决策边界的距离替代到原点的距离作为缩放因子，在不损失分类精度的前提下显著提升各种 post-hoc OOD 检测方法的性能和置信度校准。

**[Flow3r: Factored Flow Prediction for Scalable Visual Geometry Learning](flow3r_factored_flow_prediction_for_scalable_visual_geometry_learning.md)**

:   提出"分解式光流预测"（Factored Flow）模块，用源视图的几何 latent + 目标视图的位姿 latent 预测光流，使无标注视频可作为三维几何学习的监督信号，在静态/动态场景的 8 个基准上达到 SOTA。

**[Free-Grained Hierarchical Visual Recognition](free-grained_hierarchical_visual_recognition.md)**

:   提出"自由粒度"层级视觉识别（free-grained hierarchical recognition），允许训练标签出现在分类法的任意层级，并提出文本引导伪属性和分类法引导半监督学习两种方法来弥补缺失监督，推理时模型自适应选择预测深度。

**[HeSS: Head Sensitivity Score for Sparsity Redistribution in VGGT](hess_head_sensitivity_score_for_sparsity_redistribution_in_vggt.md)**

:   HeSS 提出 Head Sensitivity Score 来量化 VGGT 全局注意力层中每个注意力头对稀疏化的敏感程度，并基于此将注意力预算从不敏感的头重新分配到敏感头，在高稀疏度下显著优于均匀稀疏化方法 SparseVGGT，几乎不增加运行时开销。

**[Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces](hier-cos_making_deep_features_hierarchy-aware_via_composition_of_orthogonal_subs.md)**

:   提出 Hier-COS 框架，通过为层次树中每个节点分配正交基向量，构造理论上保证层次一致性的层次感知向量空间(HAVS)，首次统一了"层次感知细粒度分类"和"层次多级分类"，同时提出新评估指标HOPS，在4个数据集上全面超越SOTA。

**[Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces](hiercos_making_deep_features_hierarchyaware_via_co.md)**

:   提出Hier-COS框架，为层次标签树中的每个节点分配正交基向量，通过子空间组合（祖先基+自身基+后代基）构建层次感知向量空间（HAVS），理论保证特征空间的距离结构与层次树一致，同时提出HOPS评估指标解决现有层次化评估指标的排列不变性缺陷。

**[Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models](out_of_sight_out_of_mind_evaluating_state_evolutio.md)**

:   本文提出StEvo-Bench，一个包含225个任务的benchmark，通过在视频生成过程中插入遮挡或相机转向来测试视频世界模型能否在不可观测期间继续正确演化场景状态，发现当前最先进模型（包括Veo 3、Sora 2 Pro等）的成功率不到10%，揭示了视频模型将状态演化与观察高度耦合的根本问题。

**[Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models](out_of_sight_out_of_mind_evaluating_state_evolution_in_video_world_models.md)**

:   提出 StEvo-Bench 基准（225个任务×6类演化），通过遮挡或相机移开等观测控制手段系统评测9个视频世界模型能否将状态演化与观测解耦，发现所有模型在观测中断时成功率不足10%，并通过5个专项验证器精准定位失败模式。

**[Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline](pioneering_perceptual_video_fluency_assessment_a_novel_task_with_benchmark_datas.md)**

:   本文首次将视频流畅度评估（VFA）从传统视频质量评估（VQA）中独立出来，构建了首个流畅度评估数据集 FluVid（4,606 视频），并提出 FluNet 基线模型，通过时序排列自注意力（T-PSA）实现高效帧间交互，SRCC/PLCC 分别达到 0.816/0.821。

**[ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation](reflexsplit_single_image_reflection_separation_via_layer_fusion-separation.md)**

:   针对单图反射分离中的透射-反射混淆问题（尤其是在深层解码器中），提出ReflexSplit双流框架，通过跨尺度门控融合(CrGF)稳定多尺度特征流、层级融合-分离块(LFSB)的差分双维注意力实现跨流减法解纠缠、课程训练渐进增强分离强度，在合成和真实世界数据集上达到SOTA性能。

**[Reframing Long-Tailed Learning via Loss Landscape Geometry](reframing_long-tailed_learning_via_loss_landscape_geometry.md)**

:   从损失景观几何的角度重新审视长尾学习中的head-tail seesaw困境，发现尾类退化的根源是优化收敛到尖锐且远离尾类最优点的区域，提出基于持续学习思想的GKP（分组知识保存）和GSA（分组锐度感知）双模块框架，无需额外数据即在CIFAR-LT/ImageNet-LT/iNat2018四个基准上取得SOTA。

**[SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval](sattc_structure-aware_label-free_test-time_calibration_for_cross-subject_eeg-to-.md)**

:   提出SATTC，一个无标签的测试时校准头，通过几何专家（被试自适应白化+自适应CSLS）和结构专家（互最近邻+双向top-k排名+类别流行度）的乘积专家融合，在冻结的EEG和图像编码器上直接操作相似度矩阵，显著改善跨被试EEG-to-image检索的Top-1精度并降低hubness效应。

**[Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)**

:   提出 SemiCP 框架，通过最近邻匹配（NNM）分数将无标签数据引入 conformal prediction 的校准流程，在标注数据极少时将平均覆盖率偏差降低最多 77%，同时缩小预测集。

**[SparseCam4D: Spatio-Temporally Consistent 4D Reconstruction from Sparse Cameras](sparsecam4d_spatio-temporally_consistent_4d_reconstruction_from_sparse_cameras.md)**

:   提出 SparseCam4D，首个在标准多相机动态场景基准上实现稀疏相机（2-3个）4D重建的方法，核心创新是时空扭曲场（STDF），通过将生成式观测中的时空不一致性显式建模并与真实4D高斯表示解耦，实现高保真、时空一致的动态场景渲染。

**[TacSIm: A Dataset and Benchmark for Football Tactical Style Imitation](tacsim_a_dataset_and_benchmark_for_football_tactical_style_imitation.md)**

:   本文提出 TacSIm，首个从真实英超比赛转播画面中重建全队轨迹并在虚拟足球环境中进行战术风格模仿的大规模数据集与基准，通过空间占据相似度和运动向量相似度两个指标量化战术模仿保真度。

**[Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning](temporal_imbalance_of_positive_and_negative_supervision_in_class-incremental_lea.md)**

:   提出时序不平衡（Temporal Imbalance）这一被忽视的类增量学习偏差来源，并设计 Temporal-Adjusted Loss（TAL）通过时间衰减记忆核动态降低旧类的负监督权重，以即插即用的方式显著缓解灾难性遗忘。

**[Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)**

:   提出 LAS-VAD 框架，通过异常连通分量机制（ACC）将视频帧划分为语义一致的组来生成伪标签弥补帧级标注缺失，并通过意图感知机制（IAM）利用位置-速度-加速度特征区分外观相似但意图不同的正常/异常行为，在 XD-Violence 上达 89.96% AP (I3D)。
