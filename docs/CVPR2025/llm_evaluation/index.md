---
title: >-
  CVPR2025 LLM评测方向 19篇论文解读
description: >-
  19篇CVPR2025 LLM评测方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📊 LLM评测

**📷 CVPR2025** · **19** 篇论文解读

**[ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems](comfybench_benchmarking_llm-based_agents_in_comfyui_for_autonomously_designing_c.md)**

:   ComfyBench 提出了首个评估LLM Agent在ComfyUI中自主设计协作AI系统能力的综合性Benchmark（200个任务、3205个节点文档、20个课程工作流），并提出ComfyAgent框架通过代码化工作流表示和多Agent协作，达到了与o1-preview相当的解决率，但在创意任务上仅解决15%，揭示了LLM Agent在自主系统设计上的巨大差距。

**[ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval](context-cir_learning_from_concepts_in_text_for_composed_image_retrieval.md)**

:   提出 ConText-CIR 框架，通过 Text Concept-Consistency 损失让文本修改中的名词短语更好地关注查询图像的相关部分，配合合成数据生成管线，在多个 CIR 基准上取得 SOTA。

**[Do ImageNet-trained Models Learn Shortcuts? The Impact of Frequency Shortcuts on Generalization](do_imagenet-trained_models_learn_shortcuts_the_impact_of_frequency_shortcuts_on_.md)**

:   提出层次化频率捷径搜索方法（HFSS），首次在ImageNet-1K规模上高效发现CNN和Transformer学到的频率捷径（仅5%频率即可正确分类），揭示频率捷径在保留纹理的OOD测试中反而有益但在风格化测试（IN-R/IN-S）上有害，指出现有OOD评估框架忽视了频率捷径的影响。

**[Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning](dual_consolidation_for_pre-trained_model-based_domain-incremental_learning.md)**

:   提出Duct方法，通过表征合并（累加任务向量构建统一嵌入空间）和分类器合并（利用类别语义信息通过最优传输估计旧域分类器权重），在预训练模型基础上实现无样本存储的域增量学习，在四个基准上以1~7%的优势超越SOTA。

**[Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways (EraDiff)](erase_diffusion_empowering_object_removal_through_calibrating_diffusion_pathways.md)**

:   本文提出EraDiff，通过链式校正优化范式（CRO）建立从"含物体"到"纯背景"的渐进扩散路径，并用自校正注意力机制（SRA）在采样时抑制伪影，使扩散模型真正理解"擦除意图"，在OpenImages V5上取得SOTA的Local FID（3.799），在复杂真实场景中显著优于SD2-Inpaint和LaMa。

**[Event Ellipsometer: Event-based Mueller-Matrix Video Imaging](event_ellipsometer_event-based_mueller-matrix_video_imaging.md)**

:   首个实现 30fps 视频级穆勒矩阵成像的系统——用事件相机捕捉快速旋转 QWP 产生的光强调制，将事件时间差映射到穆勒矩阵比值，通过 SVD 估计+时空传播重建物理有效的穆勒矩阵视频。

**[Gradient-Guided Annealing for Domain Generalization](gradient-guided_annealing_for_domain_generalization.md)**

:   提出GGA方法，在训练早期通过模拟退火搜索参数空间中梯度跨域对齐的点（最小化域间梯度余弦相似度的最小值），引导模型在优化初期找到域不变特征的起始点，从而在无需数据增强的情况下提升域泛化性，可与现有DG方法组合获得显著提升。

**[LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty](lotus_large-scale_machine_unlearning_with_a_taste_of_uncertainty.md)**

:   提出 LoTUS，用 logits 温度调节+Gumbel-Softmax 平滑遗忘样本的预测，通过动态温度调度收敛到"遗忘集准确率=未见集准确率"的目标——在 ImageNet-1K 大规模设置中高效遗忘（ViT 上 Avg Gap 0.0150），且提出 RF-JSD 免重训评估指标（与 JSD Pearson 相关 0.92）。

**[On the Generalization of Handwritten Text Recognition Models](on_the_generalization_of_handwritten_text_recognition_models.md)**

:   本文首次系统性地分析了 HTR 模型在域外（OOD）数据上的泛化能力，通过对 8 个 SOTA 模型在 7 个数据集（5 种语言）上的 336 种 OOD 评估发现：文本差异是影响泛化的最关键因素，OOD 误差在 70% 的情况下可以被可靠预估（偏差 <10 个百分点）。

**[OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)**

:   提出 OODD，通过优先队列维护动态 OOD 字典在测试时实时收集潜在 OOD 样本特征来校准 OOD 分数，在 CIFAR-100 Far OOD 上相比 SOTA 方法 FPR95 降低 26.0%，且无需微调。

**[Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models](out_of_sight_out_of_mind_evaluating_state_evolution_in_video_world_models.md)**

:   StEvo-Bench 提出了一个评估视频世界模型"不可观测状态演化"能力的 benchmark——测试当物理过程不被观察时（相机移开/遮挡/关灯），世界模型能否继续正确推理状态变化，结果发现当前所有前沿模型（Veo 3、Sora 2 Pro 等）的任务成功率均低于 10%，揭示了"眼不见，心不在"的严重缺陷。

**[PolarFree: Polarization-based Reflection-Free Imaging](polarfree_polarization-based_reflection-free_imaging.md)**

:   构建 6500 对的大规模 RGB-偏振图像数据集 PolaRGB，并提出 PolarFree 两阶段网络——先用条件扩散模型生成无反射先验，再用去反射骨干网络分离透射层，在偏振引导的反射去除任务上超越先前方法约 2dB PSNR。

**[Potential Field Based Deep Metric Learning](potential_field_based_deep_metric_learning.md)**

:   提出 PFML，用物理势能场概念替代传统的 tuple mining 进行度量学习——每个样本在嵌入空间中创建连续的引力场（同类）和斥力场（异类），具有距离衰减特性（远处交互力弱），在 Cars-196 上 R@1 达 92.7%（前 SOTA 89.6%）。

**[Practical Solutions to the Relative Pose of Three Calibrated Cameras](practical_solutions_to_the_relative_pose_of_three_calibrated_cameras.md)**

:   本文针对三个标定相机的四点三视图（4p3v）相对位姿估计这一经典难题，提出了基于近似几何的实用求解方案——利用仿射相机近似或均值点近似对应来估计前两个相机的相对位姿，再通过P3P注册第三个相机，配合局部优化在真实数据上取得了SOTA精度。

**[Scene-Agnostic Pose Regression for Visual Localization](scene-agnostic_pose_regression_for_visual_localization.md)**

:   提出"场景无关位姿回归"（SPR）新任务范式，以序列首帧为坐标原点回归后续帧的相对位姿，避免了APR需重训练、RPR需检索数据库、VO存在累积漂移的困境，并建立了200K全景图的360SPR大规模数据集和双分支SPR-Mamba模型。

**[Sufficient Invariant Learning for Distribution Shift](sufficient_invariant_learning_for_distribution_shift.md)**

:   本文提出充分不变学习（SIL）框架，通过学习多样化的不变特征子集而非单一不变特征来提升分布偏移下的鲁棒性，并设计ASGDRO算法通过寻找跨环境的公共平坦极小值来实现SIL，在多个分布偏移基准上取得SOTA性能。

**[TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception](traf-align_trajectory-aware_feature_alignment_for_asynchronous_multi-agent_perce.md)**

:   提出TraF-Align框架，通过预测特征级目标轨迹来学习特征的流动路径，生成沿时间排列的采样点将当前时刻query引导至相关历史特征，解决异步多智能体感知中的时空失配问题。

**[Uncertainty Weighted Gradients for Model Calibration](uncertainty_weighted_gradients_for_model_calibration.md)**

:   通过分析 Focal Loss 等方法的统一框架，揭示了直接将不确定性权重应用于损失函数会导致梯度与不确定性不对齐的问题，提出将不确定性权重直接应用于梯度的 Uncertainty-GRA 框架，并用广义 Brier Score 作为更精确的不确定性度量，取得了 SOTA 校准性能。

**[VinaBench: Benchmark for Faithful and Consistent Visual Narratives](vinabench_benchmark_for_faithful_and_consistent_visual_narratives.md)**

:   构建了 VinaBench 基准，为视觉叙事样本标注常识链接和话语约束，提出忠实度和一致性评估指标，并验证利用这些约束可显著提升视觉叙事生成的质量。
