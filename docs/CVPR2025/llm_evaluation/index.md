---
title: >-
  CVPR2025 LLM 评测方向31篇论文解读
description: >-
  31篇CVPR2025的 LLM 评测方向论文解读，涵盖持续学习、Agent、LLM、人脸/视线、扩散模型、域适应等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2025"
  - "LLM 评测"
  - "论文解读"
  - "论文笔记"
  - "持续学习"
  - "Agent"
  - "LLM"
  - "人脸/视线"
  - "扩散模型"
  - "域适应"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM 评测

**📷 CVPR2025** · **31** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (8)](../../ICML2026/llm_evaluation/index.md) · [💬 ACL2026 (79)](../../ACL2026/llm_evaluation/index.md) · [📷 CVPR2026 (25)](../../CVPR2026/llm_evaluation/index.md) · [🔬 ICLR2026 (53)](../../ICLR2026/llm_evaluation/index.md) · [🤖 AAAI2026 (39)](../../AAAI2026/llm_evaluation/index.md) · [🧠 NeurIPS2025 (79)](../../NeurIPS2025/llm_evaluation/index.md)

🔥 **高频主题：** 持续学习 ×2 · Agent ×2

**[ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems](comfybench_benchmarking_llm-based_agents_in_comfyui_for_autonomously_designing_c.md)**

:   ComfyBench 提出了首个评估LLM Agent在ComfyUI中自主设计协作AI系统能力的综合性Benchmark（200个任务、3205个节点文档、20个课程工作流），并提出ComfyAgent框架通过代码化工作流表示和多Agent协作，达到了与o1-preview相当的解决率，但在创意任务上仅解决15%，揭示了LLM Agent在自主系统设计上的巨大差距。

**[ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval](context-cir_learning_from_concepts_in_text_for_composed_image_retrieval.md)**

:   提出 ConText-CIR 框架，通过 Text Concept-Consistency 损失让文本修改中的名词短语更好地关注查询图像的相关部分，配合合成数据生成管线，在多个 CIR 基准上取得 SOTA。

**[Do ImageNet-trained Models Learn Shortcuts? The Impact of Frequency Shortcuts on Generalization](do_imagenet-trained_models_learn_shortcuts_the_impact_of_frequency_shortcuts_on_.md)**

:   提出层次化频率捷径搜索方法（HFSS），首次在ImageNet-1K规模上高效发现CNN和Transformer学到的频率捷径（仅5%频率即可正确分类），揭示频率捷径在保留纹理的OOD测试中反而有益但在风格化测试（IN-R/IN-S）上有害，指出现有OOD评估框架忽视了频率捷径的影响。

**[Dora: Sampling and Benchmarking for 3D Shape Variational Auto-Encoders](dora_sampling_and_benchmarking_for_3d_shape_variational_auto-encoders.md)**

:   提出 Dora-VAE，通过 Sharp Edge Sampling (SES) 关注几何锐边区域、Dual Cross-Attention 分别处理均匀和显著采样点，以仅 1,280 个 latent codes（8× 小于 XCube-VAE 的 10,000+）实现更优的 3D 形状重建质量，同时建立了新的 Dora-Bench 评测基准。

**[Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning](dual_consolidation_for_pre-trained_model-based_domain-incremental_learning.md)**

:   提出Duct方法，通过表征合并（累加任务向量构建统一嵌入空间）和分类器合并（利用类别语义信息通过最优传输估计旧域分类器权重），在预训练模型基础上实现无样本存储的域增量学习，在四个基准上以1~7%的优势超越SOTA。

**[Enhancing 3D Gaze Estimation in the Wild Using Weak Supervision with Gaze Following Labels](enhancing_3d_gaze_estimation_in_the_wild_using_weak_supervision_with_gaze_follow.md)**

:   提出一种两阶段自训练弱监督框架 ST-WSGE，利用 2D 注视跟随数据集（如 GazeFollow）生成 3D 伪标签来增强野外 3D 注视估计的泛化能力，同时设计了模态无关的 Gaze Transformer（GaT）统一处理图像和视频输入，在 Gaze360、GFIE、MPIIFaceGaze 等数据集上取得 SOTA。

**[Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways (EraDiff)](erase_diffusion_empowering_object_removal_through_calibrating_diffusion_pathways.md)**

:   本文提出EraDiff，通过链式校正优化范式（CRO）建立从"含物体"到"纯背景"的渐进扩散路径，并用自校正注意力机制（SRA）在采样时抑制伪影，使扩散模型真正理解"擦除意图"，在OpenImages V5上取得SOTA的Local FID（3.799），在复杂真实场景中显著优于SD2-Inpaint和LaMa。

**[Event Ellipsometer: Event-based Mueller-Matrix Video Imaging](event_ellipsometer_event-based_mueller-matrix_video_imaging.md)**

:   首个实现 30fps 视频级穆勒矩阵成像的系统——用事件相机捕捉快速旋转 QWP 产生的光强调制，将事件时间差映射到穆勒矩阵比值，通过 SVD 估计+时空传播重建物理有效的穆勒矩阵视频。

**[Gradient-Guided Annealing for Domain Generalization](gradient-guided_annealing_for_domain_generalization.md)**

:   提出GGA方法，在训练早期通过模拟退火搜索参数空间中梯度跨域对齐的点（最小化域间梯度余弦相似度的最小值），引导模型在优化初期找到域不变特征的起始点，从而在无需数据增强的情况下提升域泛化性，可与现有DG方法组合获得显著提升。

**[Improving Accuracy and Calibration via Differentiated Deep Mutual Learning](improving_accuracy_and_calibration_via_differentiated_deep_mutual_learning.md)**

:   提出 Diff-DML（Differentiated Deep Mutual Learning），通过差异化训练策略（DTS）和多样性保持学习目标（DPLO）两个核心设计，在保持集成模型预测多样性的同时，同时提升准确率和不确定性校准质量。

**[KAC: Kolmogorov-Arnold Classifier for Continual Learning](kac_kolmogorov-arnold_classifier_for_continual_learning.md)**

:   首次将 Kolmogorov-Arnold Network (KAN) 应用于持续学习，通过将 B-spline 替换为径向基函数 (RBF) 构建分类器 KAC，仅增加 0.23M 参数即可在多种持续学习方法上获得一致且显著的性能提升（CUB200 40-step 最高 +20.70%）。

**[LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty](lotus_large-scale_machine_unlearning_with_a_taste_of_uncertainty.md)**

:   提出 LoTUS，用 logits 温度调节+Gumbel-Softmax 平滑遗忘样本的预测，通过动态温度调度收敛到"遗忘集准确率=未见集准确率"的目标——在 ImageNet-1K 大规模设置中高效遗忘（ViT 上 Avg Gap 0.0150），且提出 RF-JSD 免重训评估指标（与 JSD Pearson 相关 0.92）。

**[MagicArticulate: Make Your 3D Models Articulation-Ready](magicarticulate_make_your_3d_models_articulation-ready.md)**

:   提出 MagicArticulate 两阶段框架，第一阶段用自回归 Transformer 将骨架生成建模为序列预测任务，第二阶段用函数扩散过程结合体积测地距离先验预测蒙皮权重，搭配 33K+ 大规模 Articulation-XL 数据集，实现静态 3D 模型到可动画化资产的自动转换。

**[Making Old Film Great Again: Degradation-aware State Space Model for Old Film Restoration](making_old_film_great_again_degradation-aware_state_space_model_for_old_film_res.md)**

:   本文提出MambaOFR框架，针对老电影特有的复合退化问题，设计退化感知prompt引导Mamba模型动态调整修复模式，配合光流引导的掩码变形对齐模块防止结构缺陷传播，并引入首个包含合成与真实数据的老电影修复benchmark数据集。

**[NADER: Neural Architecture Design via Multi-Agent Collaboration](nader_neural_architecture_design_via_multi-agent_collaboration.md)**

:   NADER 将神经架构设计建模为多 LLM Agent 协作任务——Reader 读论文提炼知识、Proposer 生成改进方案、Modifier 用 DAG 图实现修改、Reflector 从失败中学习经验，仅 10 次试验即突破 NAS-Bench-201 搜索空间的准确率上限，在 CIFAR-100 上达 74.51%（搜索空间最优 73.51%）。

**[On the Generalization of Handwritten Text Recognition Models](on_the_generalization_of_handwritten_text_recognition_models.md)**

:   本文首次系统性地分析了 HTR 模型在域外（OOD）数据上的泛化能力，通过对 8 个 SOTA 模型在 7 个数据集（5 种语言）上的 336 种 OOD 评估发现：文本差异是影响泛化的最关键因素，OOD 误差在 70% 的情况下可以被可靠预估（偏差 <10 个百分点）。

**[OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)**

:   提出 OODD，通过优先队列维护动态 OOD 字典在测试时实时收集潜在 OOD 样本特征来校准 OOD 分数，在 CIFAR-100 Far OOD 上相比 SOTA 方法 FPR95 降低 26.0%，且无需微调。

**[Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models](out_of_sight_out_of_mind_evaluating_state_evolution_in_video_world_models.md)**

:   StEvo-Bench 提出了一个评估视频世界模型"不可观测状态演化"能力的 benchmark——测试当物理过程不被观察时（相机移开/遮挡/关灯），世界模型能否继续正确推理状态变化，结果发现当前所有前沿模型（Veo 3、Sora 2 Pro 等）的任务成功率均低于 10%，揭示了"眼不见，心不在"的严重缺陷。

**[PosterO: Structuring Layout Trees to Enable Language Models in Generalized Content-Aware Layout Generation](postero_structuring_layout_trees_to_enable_language_models_in_generalized_conten.md)**

:   提出 PosterO，将海报版面结构化为 SVG 布局树，通过设计意图向量化和层次节点表示实现与 LLM 的对接，利用意图对齐的上下文学习生成高质量内容感知版面，在多个基准上达到 SOTA 并引入首个支持多用途和多形状元素的 PStylish7 数据集。

**[Potential Field Based Deep Metric Learning](potential_field_based_deep_metric_learning.md)**

:   提出 PFML，用物理势能场概念替代传统的 tuple mining 进行度量学习——每个样本在嵌入空间中创建连续的引力场（同类）和斥力场（异类），具有距离衰减特性（远处交互力弱），在 Cars-196 上 R@1 达 92.7%（前 SOTA 89.6%）。

**[Practical Solutions to the Relative Pose of Three Calibrated Cameras](practical_solutions_to_the_relative_pose_of_three_calibrated_cameras.md)**

:   本文针对三个标定相机的四点三视图（4p3v）相对位姿估计这一经典难题，提出了基于近似几何的实用求解方案——利用仿射相机近似或均值点近似对应来估计前两个相机的相对位姿，再通过P3P注册第三个相机，配合局部优化在真实数据上取得了SOTA精度。

**[RoadSocial: A Diverse VideoQA Dataset and Benchmark for Road Event Understanding from Social Video Narratives](roadsocial_a_diverse_videoqa_dataset_and_benchmark_for_road_event_understanding_.md)**

:   本文提出RoadSocial，一个来源于社交媒体的大规模多样化VideoQA数据集（13.2K视频、260K问答对），覆盖全球多地域多视角的道路事件场景，通过半自动标注框架和12类QA任务系统性评测了18种Video LLM的道路事件理解能力。

**[SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers](sata_spatial_autocorrelation_token_analysis_for_enhancing_the_robustness_of_visi.md)**

:   本文提出SATA（Spatial Autocorrelation Token Analysis），一种免训练的ViT鲁棒性增强方法，通过空间自相关分析将token按空间关联模式分组，利用分组信息重新加权token表示，提升ViT在分布偏移和对抗攻击下的鲁棒性，且不影响干净样本性能。

**[Scene-Agnostic Pose Regression for Visual Localization](scene-agnostic_pose_regression_for_visual_localization.md)**

:   提出"场景无关位姿回归"（SPR）新任务范式，以序列首帧为坐标原点回归后续帧的相对位姿，避免了APR需重训练、RPR需检索数据库、VO存在累积漂移的困境，并建立了200K全景图的360SPR大规模数据集和双分支SPR-Mamba模型。

**[Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection](seeing_what_matters_empowering_clip_with_patch_generation-to-selection.md)**

:   提出 CLIP-PGS（Patch Generation-to-Selection），一种简洁有效的掩码策略，通过渐进式的"生成-选择"过程——先预选候选掩码patch、再用 Sobel 边缘检测保护关键语义区域、最后用最优传输归一化精细化选择——在提升 CLIP 训练效率（降至 0.5-0.6× 训练时间）的同时在零样本分类、检索等任务上取得 SOTA。

**[Sufficient Invariant Learning for Distribution Shift](sufficient_invariant_learning_for_distribution_shift.md)**

:   本文提出充分不变学习（SIL）框架，通过学习多样化的不变特征子集而非单一不变特征来提升分布偏移下的鲁棒性，并设计ASGDRO算法通过寻找跨环境的公共平坦极小值来实现SIL，在多个分布偏移基准上取得SOTA性能。

**[TensoFlow: Tensorial Flow-based Sampler for Inverse Rendering](tensoflow_tensorial_flow-based_sampler_for_inverse_rendering.md)**

:   提出 TensoFlow，通过张量化归一化流（Tensorial Normalizing Flow）学习空间-方向感知的重要性采样器，替代逆渲染中固定的预定义采样器（如 cosine-weighted、GGX），大幅降低渲染方程蒙特卡洛估计的方差，提升材质和光照分解质量。

**[Towards In-the-Wild 3D Plane Reconstruction from a Single Image](towards_in-the-wild_3d_plane_reconstruction_from_a_single_image.md)**

:   ZeroPlane 提出了首个跨域零样本3D平面重建框架，通过构建包含14个数据集/56万标注的大规模平面基准数据集，并设计法向量-偏移解耦的分类-回归范式和像素几何增强嵌入模块，实现了在室内外多样场景中显著优于现有方法的泛化性能。

**[TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception](traf-align_trajectory-aware_feature_alignment_for_asynchronous_multi-agent_perce.md)**

:   提出 TraF-Align 框架，通过在特征级别预测目标运动轨迹来学习特征的时空流动路径，沿轨迹生成时序有序的采样点将当前时刻 query 引导至相关历史特征，实现异步多智能体感知中的精确特征对齐，在 V2V4Real 和 DAIR-V2X-Seq 两个真实数据集上刷新SOTA。

**[Uncertainty Weighted Gradients for Model Calibration](uncertainty_weighted_gradients_for_model_calibration.md)**

:   通过分析 Focal Loss 等方法的统一框架，揭示了直接将不确定性权重应用于损失函数会导致梯度与不确定性不对齐的问题，提出将不确定性权重直接应用于梯度的 Uncertainty-GRA 框架，并用广义 Brier Score 作为更精确的不确定性度量，取得了 SOTA 校准性能。

**[VinaBench: Benchmark for Faithful and Consistent Visual Narratives](vinabench_benchmark_for_faithful_and_consistent_visual_narratives.md)**

:   构建了 VinaBench 基准，为视觉叙事样本标注常识链接和话语约束，提出忠实度和一致性评估指标，并验证利用这些约束可显著提升视觉叙事生成的质量。
