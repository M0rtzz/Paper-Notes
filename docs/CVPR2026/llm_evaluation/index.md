---
title: >-
  CVPR2026 LLM 评测方向25篇论文解读
description: >-
  25篇CVPR2026的 LLM 评测方向论文解读，涵盖持续学习、布局/合成、少样本学习、动态场景、形状补全、异常检测等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2026"
  - "LLM 评测"
  - "论文解读"
  - "论文笔记"
  - "持续学习"
  - "布局/合成"
  - "少样本学习"
  - "动态场景"
  - "形状补全"
  - "异常检测"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM 评测

**📷 CVPR2026** · **25** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (8)](../../ICML2026/llm_evaluation/index.md) · [💬 ACL2026 (79)](../../ACL2026/llm_evaluation/index.md) · [🔬 ICLR2026 (53)](../../ICLR2026/llm_evaluation/index.md) · [🤖 AAAI2026 (39)](../../AAAI2026/llm_evaluation/index.md) · [🧠 NeurIPS2025 (79)](../../NeurIPS2025/llm_evaluation/index.md) · [📹 ICCV2025 (27)](../../ICCV2025/llm_evaluation/index.md)

🔥 **高频主题：** 持续学习 ×2

**[ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation](ace-merging_data-free_model_merging_with_adaptive_covariance_estimation.md)**

:   本文从理论上证明了微调参数差蕴含输入协方差信息，据此提出 ACE-Merging，通过自适应协方差估计、集体结构先验和谱精炼三步实现无数据闭式模型合并，在 GPT-2 上比之前方法平均提升 4%，在 RoBERTa-Base 上提升 5%。

**[AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks](adabet_gradient-free_layer_selection_for_efficient_training_of_deep_neural_netwo.md)**

:   提出 AdaBet，一种基于代数拓扑（第一 Betti 数 $b_1$）的无梯度层选择方法，仅通过前向传播计算每层激活空间的拓扑复杂度来决定哪些层需要微调，无需标签、梯度或反向传播，在 ResNet50/VGG16/MobileNetV2/ViT-B16 上以仅 10% 层微调达到优于全量训练的准确率，同时峰值内存降低约 40%。

**[Anchoring and Rescaling Attention for Semantically Coherent Inbetweening](anchoring_and_rescaling_attention_for_semantically_coherent_inbetweening.md)**

:   提出 KAB（Keyframe-Anchored Attention Bias）和 ReTRo（Rescaled Temporal RoPE）两个无需训练的推理时方法，基于 Wan2.1 视频扩散模型解决稀疏关键帧下大运动生成式帧插值（GI）中的语义不忠、帧不一致和节奏不稳问题，并构建首个文本条件 GI 评估基准 TGI-Bench。

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

**[HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)**

:   本文识别了异质域持续学习中的"域引力"（Domain Gravity）偏差——数据丰富或低熵域在共享嵌入空间中产生不成比例的影响，并提出 HyCal，一种无训练方法，通过融合余弦相似度和马氏距离进行原型校准，在跨学科不平衡少样本增量学习中实现稳健分类。

**[Learning Like Humans: Analogical Concept Learning for Generalized Category Discovery](learning_like_humans_analogical_concept_learning_for_generalized_category_discov.md)**

:   提出 AL-GCD 框架，通过模拟人类类比推理机制设计"类比文本概念生成器"（ATCG）——从已知类别的视觉-文本知识库中类比生成未知样本的文本概念，将类别发现转化为视觉-文本联合推理任务，在六个基准上平均提升 5.0%，细粒度数据集提升 7.1%。

**[Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models](out_of_sight_out_of_mind_evaluating_state_evolutio.md)**

:   本文提出StEvo-Bench，一个包含225个任务的benchmark，通过在视频生成过程中插入遮挡或相机转向来测试视频世界模型能否在不可观测期间继续正确演化场景状态，发现当前最先进模型（包括Veo 3、Sora 2 Pro等）的成功率不到10%，揭示了视频模型将状态演化与观察高度耦合的根本问题。

**[Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline](pioneering_perceptual_video_fluency_assessment_a_novel_task_with_benchmark_datas.md)**

:   本文首次将视频流畅度评估（VFA）从传统视频质量评估（VQA）中独立出来，构建了首个流畅度评估数据集 FluVid（4,606 视频），并提出 FluNet 基线模型，通过时序排列自注意力（T-PSA）实现高效帧间交互，SRCC/PLCC 分别达到 0.816/0.821。

**[PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion](prism_video_dataset_condensation_with_progressive_refinement_and_insertion_for_s.md)**

:   本文提出 PRISM，一种整体式视频数据集压缩方法：从仅两个时间锚点（首尾帧）出发,通过检测梯度方向冲突来自适应插入关键帧，在保持内容与运动的耦合完整性的同时实现 SOTA 的存储效率——在 miniUCF 1VPC 上用 20MB 达到 17.9% 准确率，比先前方法的 94MB 少 5 倍。

**[R2G: A Multi-View Circuit Graph Benchmark Suite from RTL to GDSII](r2g_multi_view_circuit_graph_benchmark_suite_from_rtl_to_gdsii.md)**

:   提出 R2G，首个标准化的多视图电路图基准套件，在 30 个 IP 核上提供 5 种阶段感知的图表示（具有信息对等性），系统研究发现图表示选择比 GNN 模型选择对性能影响更大。

**[ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation](reflexsplit_single_image_reflection_separation_via_layer_fusion-separation.md)**

:   针对单图反射分离中的透射-反射混淆问题（尤其是在深层解码器中），提出ReflexSplit双流框架，通过跨尺度门控融合(CrGF)稳定多尺度特征流、层级融合-分离块(LFSB)的差分双维注意力实现跨流减法解纠缠、课程训练渐进增强分离强度，在合成和真实世界数据集上达到SOTA性能。

**[Reframing Long-Tailed Learning via Loss Landscape Geometry](reframing_long-tailed_learning_via_loss_landscape_geometry.md)**

:   从损失景观几何的角度重新审视长尾学习中的head-tail seesaw困境，发现尾类退化的根源是优化收敛到尖锐且远离尾类最优点的区域，提出基于持续学习思想的GKP（分组知识保存）和GSA（分组锐度感知）双模块框架，无需额外数据即在CIFAR-LT/ImageNet-LT/iNat2018四个基准上取得SOTA。

**[Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)**

:   提出 SemiCP 框架，通过最近邻匹配（NNM）分数将无标签数据引入 conformal prediction 的校准流程，在标注数据极少时将平均覆盖率偏差降低最多 77%，同时缩小预测集。

**[SparseCam4D: Spatio-Temporally Consistent 4D Reconstruction from Sparse Cameras](sparsecam4d_spatio-temporally_consistent_4d_reconstruction_from_sparse_cameras.md)**

:   提出 SparseCam4D，首个在标准多相机动态场景基准上实现稀疏相机（2-3个）4D重建的方法，核心创新是时空扭曲场（STDF），通过将生成式观测中的时空不一致性显式建模并与真实4D高斯表示解耦，实现高保真、时空一致的动态场景渲染。

**[TacSIm: A Dataset and Benchmark for Football Tactical Style Imitation](tacsim_a_dataset_and_benchmark_for_football_tactical_style_imitation.md)**

:   本文提出 TacSIm，首个从真实英超比赛转播画面中重建全队轨迹并在虚拟足球环境中进行战术风格模仿的大规模数据集与基准，通过空间占据相似度和运动向量相似度两个指标量化战术模仿保真度。

**[Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning](temporal_imbalance_of_positive_and_negative_supervision_in_class-incremental_lea.md)**

:   提出时序不平衡（Temporal Imbalance）这一被忽视的类增量学习偏差来源，并设计 Temporal-Adjusted Loss（TAL）通过时间衰减记忆核动态降低旧类的负监督权重，以即插即用的方式显著缓解灾难性遗忘。

**[Unified Primitive Proxies for Structured Shape Completion](unified_primitive_proxies_for_structured_shape_completion.md)**

:   提出 UniCo，通过基元代理（primitive proxies）在共享形状特征上学习统一的基元表示，在单次前向传递中联合预测完整点云和装配就绪的二次曲面基元（含几何、语义和成员关系），在合成/真实点云 benchmark 上 Chamfer 距离降低最高 50%，法线一致性提升最高 7%。

**[VGA-Bench: A Unified Benchmark for Video Aesthetics and Generation Quality Evaluation](vga_bench_unified_benchmark_for_video_aesthetics_and_generation_quality.md)**

:   VGA-Bench提出了一个统一的AIGC视频评估基准，包含三层分类体系（美学质量、美学标签、生成质量）、1016个提示词、60000个视频和三个专用评估模型，实现了与人类判断对齐的自动化评估。

**[Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)**

:   提出 LAS-VAD 框架，通过异常连通分量机制（ACC）将视频帧划分为语义一致的组来生成伪标签弥补帧级标注缺失，并通过意图感知机制（IAM）利用位置-速度-加速度特征区分外观相似但意图不同的正常/异常行为，在 XD-Violence 上达 89.96% AP (I3D)。
