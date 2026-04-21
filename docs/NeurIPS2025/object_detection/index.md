---
title: >-
  NeurIPS2025 目标检测方向 39篇论文解读
description: >-
  39篇NeurIPS2025 目标检测方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎯 目标检测

**🧠 NeurIPS2025** · **39** 篇论文解读

**[All You Need is One: Capsule Prompt Tuning with a Single Vector](all_you_need_is_one_capsule_prompt_tuning_with_a_single_vector.md)**

:   提出 Capsule Prompt-Tuning (CaPT)，发现现有 task-aware soft prompts 实际上与输入 tokens 缺乏交互（"attention 孤岛"），而将 instance-aware 信息融入单个 capsule prompt 可以作为"attention anchor"激活对关键结构信息的注意力，以极低参数量（如 Llama3.2-1B 上仅 0.003% 参数）实现超越多 prompt 方法的性能。

**[Angular Constraint Embedding via SpherePair Loss for Constrained Clustering](angular_constraint_embedding_via_spherepair_loss_for_constrained_clustering.md)**

:   本文提出SpherePair损失函数，通过在角度空间（而非欧几里得空间）进行成对约束嵌入学习，实现了不依赖锚点(anchor)、不需要预知聚类数的深度约束聚类方法，并提供了严格的理论保证来确定最优超参数。

**[Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector](any_large_language_model_can_be_a_reliable_judge_debiasing_w.md)**

:   提出 Reasoning-based Bias Detector（RBD）作为 LLM 评判器的即插即用去偏模块——通过外部检测 4 种评估偏见（冗长/位置/从众/情感），生成带推理链的结构化反馈引导评判器自我纠正，RBD-8B 在 8 个 LLM 评判器上平均提升准确率 18.5%、一致性 10.9%。

**[Ascent Fails to Forget](ascent_fails_to_forget.md)**

:   本文从遗忘集与保留集之间的统计依赖出发，理论结合实验证明广泛使用的梯度上升/Descent-Ascent（DA）类机器遗忘方法在存在数据相关性时会系统性失败——在 logistic 回归中 DA 解甚至会比原始模型更远离 oracle，且在非凸设置下会将模型困在劣质局部最小值中。

**[Automated Detection of Visual Attribute Reliance with a Self-Reflective Agent](automated_detection_of_visual_attribute_reliance_with_a_self-reflective_agent.md)**

:   提出一个自反思 agent 框架，通过迭代的假设生成-测试-验证-反思循环来自动检测视觉模型中的属性依赖（如 CLIP 识别 teacher 依赖教室背景、YOLOv8 检测行人依赖人行横道），在 130 个注入已知属性依赖的模型 benchmark 上显示自反思显著提升检测准确性。

**[BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes](burstdeflicker_a_benchmark_dataset_for_flicker_removal_in_dynamic_scenes.md)**

:   提出首个面向多帧闪烁去除（MFFR）的大规模 benchmark 数据集 BurstDeflicker，包含基于 Retinex 的合成数据、真实静态数据和绿幕动态数据三个互补子集，系统解决了动态场景下闪烁-干净图像对难以获取的核心瓶颈。

**[CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection](cq-dino_mitigating_gradient_dilution_via_category_queries_for_vast_vocabulary_ob.md)**

:   针对大规模类别（>10K）目标检测中分类头的正梯度稀释和难负样本梯度稀释问题，提出 CQ-DINO：用可学习类别查询替代分类头，通过图像引导的 Top-K 类别选择将负空间缩小 100 倍，在 V3Det（13204 类）上超越前 SOTA 2.1% AP，同时保持 COCO 竞争力。

**[BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes](delving_into_cascaded_instability_a_lipschitz_continuity_view_on_image_restorati.md)**

:   提出首个面向动态场景的多帧去闪烁（MFFR）基准数据集 BurstDeflicker，通过 Retinex 合成、真实静态采集与绿幕合成三种互补策略构建大规模训练/测试数据，显著提升闪烁去除模型在真实动态场景中的泛化能力。

**[DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)**

:   DetectiumFire 构建了最大的多模态火灾理解数据集——14.5K 真实图像 + 2.5K 视频 + 8K 合成图像 + 12K RLHF 偏好对，低重复率（0.03 PHash vs D-Fire 0.15），配合 4 级严重性分类标准和详细场景描述，微调 YOLOv11m 达 mAP 43.74，微调 LLaMA-3.2-11B 火灾严重性分类 83.84%。

**[DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning](detree_detecting_human-ai_collaborative_texts_via_tree-structured_hierarchical_r.md)**

:   提出 DETree 框架，通过构建层次亲和树（HAT）建模不同人机协作文本生成过程之间的层次关系，并设计树结构对比损失（TSCL）对齐表示空间，在混合文本检测和 OOD 场景下取得了显著优势。

**[Diffusion-Classifier Synergy: Reward-Aligned Learning via Mutual Boosting Loop for FSCIL](diffusion-classifier_synergy_reward-aligned_learning_via_mutual_boosting_loop_fo.md)**

:   提出 Diffusion-Classifier Synergy (DCS) 框架，通过在扩散模型和分类器之间建立互相增强的闭环，利用多层次奖励函数（特征级+logits级）引导扩散模型生成对分类器最有益的图像，在 FSCIL 基准上取得 SOTA。

**[DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)**

:   DitHub 将开放词汇目标检测的增量适配问题重新构造为"版本控制"问题——为每个类别训练独立的 LoRA 专家模块，通过 branch（分支）、fetch（检索）、merge（合并）三个原语管理不断扩展的模块库，在 ODinW-13 全量数据上以 62.19 mAP 超越 ZiRa 4.21 个点，同时保持 47.01 的零样本 COCO 性能。

**[Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable](dual_data_alignment_makes_ai-generated_image_detector_easier_generalizable.md)**

:   提出 Dual Data Alignment (DDA)，通过像素域和频域双重对齐生成训练用合成图像，消除数据集偏置导致的虚假相关性，使检测器仅学习伪造相关特征，在11个基准上平均准确率达到90.7%，大幅超越现有方法。

**[Dynamic Features Adaptation in Networking: Toward Flexible Training and Explainable Inference](dynamic_features_adaptation_in_networking_toward_flexible_training_and_explainab.md)**

:   提出 DAFI（Drift-Aware Feature Importance）算法，利用分布漂移检测动态切换 SHAP/MDI 两种特征重要性方法，结合自适应随机森林（ARF）实现通信网络场景下特征动态增加时的灵活训练与高效可解释推理。

**[FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies](flexevent_towards_flexible_event-frame_object_detection_at_varying_operational_f.md)**

:   提出 FlexEvent 框架，通过自适应事件-图像融合模块 FlexFuse 和频率自适应微调机制 FlexTune，实现事件相机在不同操作频率下的灵活目标检测，在 20Hz 到 180Hz 范围内保持鲁棒性能，显著超越现有方法。

**[Generalizable Insights for Graph Transformers in Theory and Practice](generalizable_insights_for_graph_transformers_in_theory_and_practice.md)**

:   提出 Generalized-Distance Transformer (GDT)，一种基于标准注意力（无需修改注意力机制）的图 Transformer 架构，理论证明其表达力等价于 GD-WL 算法，并通过覆盖 800 万图/2.7 亿 token 的大规模实验首次建立了 PE 表达力的细粒度经验层次，在 few-shot 迁移设置下无需微调即可超越 SOTA。

**[InstanceAssemble: Layout-Aware Image Generation via Instance Assembling Attention](instanceassemble_layoutaware_image_generation_via_instance_a.md)**

:   提出 InstanceAssemble，在 DiT-based T2I 模型（SD3 和 Flux）的 Transformer 块中注入"实例组装注意力"机制，通过将每个 bounding box 区域的 image token 独立与对应的 layout hidden state 做 cross-attention 来实现精确的实例级空间控制，同时以 LoRA 轻量适配方式保持与现有风格 LoRA 的兼容性，并提出包含 5K 图像/90K 实例的 DenseLayout 基准和多维度的 Layout Grounding Score（LGS）评估指标。

**[M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](m-grpo_stabilizing_self-supervised_reinforcement_learning_for_large_language_mod.md)**

:   针对自监督强化学习中 LLM 策略崩溃和熵崩溃问题，提出动量锚定的 GRPO（M-GRPO）框架和基于 IQR 的低熵轨迹过滤方法，实现稳定训练和 SOTA 性能。

**[M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](m-grpo_stabilizing_self-supervised_reinforcement_learning_for_multimodal_underst.md)**

:   针对自监督强化学习（SS-RLVR）在长期训练中普遍出现的"策略崩溃"问题，提出 M-GRPO：通过动量模型提供稳定的伪标签目标 + 基于四分位距（IQR）的低熵轨迹过滤防止熵崩溃，在无标注 MATH 数据集上训练 Qwen3-4B-Base，最终 checkpoint 即超越 SRT 手动选取的最佳 checkpoint，AIME24 +2.92%、GPQA +5.05%。

**[MSTAR: Box-Free Multi-Query Scene Text Retrieval with Attention Recycling](mstar_box-free_multi-query_scene_text_retrieval_with_attention_recycling.md)**

:   提出 MSTAR，首个无需边界框标注的多查询场景文本检索方法，通过渐进式视觉嵌入（PVE）逐步将注意力从显著区域转移到不显著区域，结合风格感知指令和多实例匹配模块，实现了对单词、短语、组合和语义四种查询类型的统一检索，并构建了首个多查询文本检索基准 MQTR。

**[OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps](overlaybench_a_benchmark_for_layout-to-image_generation_with_dense_overlaps.md)**

:   OverLayBench 构建了首个聚焦密集重叠场景的 Layout-to-Image 基准（4052 样本 + OverLayScore 难度指标），揭示 SOTA 方法在复杂重叠下 mIoU 从 71%→54% 急剧退化，提出 Amodal Mask 监督在重叠 IoU 上提升 15.9%。

**[PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space](pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio.md)**

:   提出 PandaPose，通过将 2D 姿态先验传播到 3D 锚点空间作为统一中间表示，结合自适应关节级 3D 锚点设置和关节级深度分布估计，实现对遮挡和 2D 姿态误差鲁棒的单帧 3D 人体姿态提升。

**[Preference Learning with Lie Detectors can Induce Honesty or Evasion](preference_learning_with_lie_detectors_can_induce_honesty_or_evasion.md)**

:   系统研究了将谎言检测器（lie detector）整合到LLM偏好学习标注流程中的效果（SOLiD框架），发现训练后模型是变得诚实还是学会规避检测取决于三个关键因素：探索程度（GRPO vs DPO）、检测器准确率（TPR）和KL正则化强度。

**[ReCon-GS: Continuum-Preserved Gaussian Streaming for Fast and Compact Reconstruction](recon-gs_continuum-preserved_gaussian_streaming_for_fast_and_compact_reconstruct.md)**

:   提出 ReCon-GS，通过连续性保持的 Gaussian 流式处理实现增量式 3D 重建，在保持渲染质量的同时大幅减少存储需求和训练时间，支持大规模场景的实时重建。

**[ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection](recon_region-controllable_data_augmentation_with_rectification_and_alignment_for.md)**

:   ReCon 提出无需额外训练的区域可控数据增强框架，通过区域引导校正（RGR）和区域对齐交叉注意力（RACA）增强现有结构可控生成模型的目标检测数据质量，在 COCO 上实现 35.5 mAP（超过需 fine-tune 的 GeoDiffusion）。

**[Rectified-CFG++ for Flow Based Models](rectified-cfg_for_flow_based_models.md)**

:   针对Rectified Flow模型中标准CFG导致的离流形漂移问题，提出Rectified-CFG++——一种自适应预测-校正引导策略，通过条件流预测+时间调度插值校正替代外推式引导，在Flux/SD3/SD3.5/Lumina等大规模模型上全面超越标准CFG。

**[Robust Hallucination Detection in LLMs via Adaptive Token Selection](robust_hallucination_detection_in_llms_via_adaptive_token_selection.md)**

:   HaMI 将幻觉检测建模为多示例学习（MIL）问题，将生成序列视为 token 实例的"bag"，通过联合优化 token 选择和幻觉检测来自适应地定位最具指示性的 token，在四个 QA 基准上以 AUROC 大幅超越所有现有方法（最高提升 11.9%）。

**[SAFE: Multitask Failure Detection for Vision-Language-Action Models](safe_multitask_failure_detection_for_vision-language-action_models.md)**

:   SAFE 发现 VLA 模型的内部特征空间存在跨任务一致的"失败区域"，据此训练轻量 MLP/LSTM 失败检测器，配合功能保形预测（FCP）做阈值校准，在未见任务上达 78% ROC-AUC，计算开销 <1%，大幅优于 token 不确定性和一致性检测方法。

**[Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning](sample_complexity_of_distributionally_robust_average-reward_reinforcement_learni.md)**

:   首次为分布鲁棒平均奖励强化学习（DR-AMDP）建立了有限样本收敛保证，提出两种算法（折扣归约法和锚定法），在KL和$f_k$-散度不确定集下均达到$\widetilde{O}(|S||A|t_{\mathrm{mix}}^2\varepsilon^{-2})$的近最优样本复杂度。

**[Segment-Factorized Full-Song Generation on Symbolic Piano Music](segment-factorized_full-song_generation_on_symbolic_piano_music.md)**

:   提出Segmented Full-Song模型（SFS），将歌曲分解为片段，通过选择性注意结构相关上下文自回归生成各片段，实现比现有方法更快速、更结构化的钢琴全曲生成，并支持交互式人机共创。

**[SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)**

:   提出 SSTAG，通过双重知识蒸馏将 LLM 和 GNN 的互补知识联合蒸馏到结构感知的 MLP 中，结合内存库机制存储原型表示，实现高效、可扩展的文本属性图跨域自监督预训练。

**[Test-Time Adaptive Object Detection with Foundation Model](test-time_adaptive_object_detection_with_foundation_model.md)**

:   提出无需源域数据的开放词汇测试时自适应目标检测框架（TTAOD），通过多模态 Prompt Tuning + Mean-Teacher + 实例动态记忆（IDM）+ 记忆增强/幻觉策略，在 Pascal-C 上 AP50 达 56.2%（+11.0 vs SOTA），在 13 个跨域数据集上一致有效。

**[The Complexity of Finding Local Optima in Contrastive Learning](the_complexity_of_finding_local_optima_in_contrastive_learning.md)**

:   证明对比学习中寻找局部最优是计算困难的：离散三元组最大化问题是 PLS-hard（即使 $d=1$），连续三元组损失最小化是 CLS-hard，意味着（在标准假设下）不存在多项式时间算法找到局部最优。

**[The Path Not Taken: RLVR Provably Learns Off the Principals](the_path_not_taken_rlvr_provably_learns_off_the_principals.md)**

:   本文提出三门理论 (Three-Gate Theory) 解释 RLVR 的参数更新稀疏性假象，证明 RLVR 在权重空间的非主方向 (off-principal) 上学习，与 SFT 的优化机制本质不同，因此直接移植 SFT 时代的 PEFT 方法到 RLVR 是有缺陷的。

**[Think Straight, Stop Smart: Structured Reasoning for Efficient Multi-Hop RAG](think_straight_stop_smart_structured_reasoning_for_efficient_multi-hop_rag.md)**

:   提出 TSSS (Think Straight, Stop Smart) 框架，通过 (i) 基于模板的推理缓存重复前缀并锚定子查询到主问题，(ii) 基于检索器的确定性终止器在子查询重复时停止推理，在多跳 RAG 基准上实现 SOTA 准确率和竞争效率。

**[Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement](towards_effective_federated_graph_foundation_model_via_mitigating_knowledge_enta.md)**

:   首次提出联邦图基础模型(FedGFM)范式，融合联邦图学习的分布式协作能力与图基础模型的跨域泛化能力，通过 AncDAI（锚点域感知初始化）和 AdaDPP（自适应域敏感提示池）两个模块缓解知识纠缠问题，在8个跨任务跨领域数据集上超越20个基线。

**[Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](video-rag_visually-aligned_retrieval-augmented_long_video_comprehension.md)**

:   本文提出Video-RAG，一个免训练、即插即用的RAG管道，通过从视频中提取视觉对齐的辅助文本（OCR、ASR、目标检测）并经检索筛选后输入LVLM，在仅增加约2K token的条件下将7个开源LVLM的Video-MME平均性能提升2.8%，72B模型超越GPT-4o。

**[XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)**

:   提出XIFBench——首个系统评估LLM多语言指令遵循能力的约束驱动基准，包含558条指令（0-5个约束，5大类21维度）×6种语言（高/中/低资源），并引入英语需求锚定评估协议，实现94.7%的跨语言评估一致性。

**[You Can Trust Your Clustering Model: A Parameter-free Self-Boosting Plug-in for Deep Clustering](you_can_trust_your_clustering_model_a_parameter-free_self-boosting_plug-in_for_d.md)**

:   提出 DCBoost，一个无需额外超参数的即插即用模块，通过自适应 k-NN 筛选高置信样本并利用可靠的局部结构信息引导全局特征空间优化，显著提升现有深度聚类模型的性能。
