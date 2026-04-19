---
title: >-
  CVPR2025 多模态VLM方向 120篇论文解读
description: >-
  120篇CVPR2025 多模态VLM方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧩 多模态VLM

**📷 CVPR2025** · **120** 篇论文解读

**[4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models](4d_langsplat_4d_language_gaussian_splatting_via_multimodal_large_language_models.md)**

:   提出4D LangSplat，通过多模态大语言模型生成逐物体视频caption来构建4D语言场，结合状态可变形网络建模语义的时间连续演变，首次实现动态场景中时间敏感和时间无关的开放词汇查询。

**[A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](a_closed-form_solution_for_debiasing_vision-language_models_with_utility_guarant.md)**

:   提出一个 training-free、data-free 的 VLM 去偏方法，通过在 cross-modal 空间中推导闭式解，实现 Pareto-optimal 的公平性与效用保持，在零样本分类、text-to-image 检索和生成三个下游任务中全面超越已有方法。

**[Active Data Curation Effectively Distills Large-Scale Multimodal Models](active_data_curation_effectively_distills_large-scale_multimodal_models.md)**

:   提出 ACID（主动数据筛选即隐式蒸馏）和 ACED（结合显式蒸馏），证明用大模型作为参考来主动筛选训练数据是一种比传统知识蒸馏更有效的多模态模型压缩方式，两者互补结合后在 27 个零样本任务上以更少推理 FLOPs 达到 SOTA。

**[Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation](beyond_final_answers_crystal_benchmark_for_transparent_multimodal_reasoning_eval.md)**

:   提出 CRYSTAL benchmark（6372 实例），通过 Match F1 和 Ordered Match F1 两个指标在中间推理步骤层面评估 MLLM，揭示了普遍的 cherry-picking 行为和推理顺序混乱问题，并提出 CPR-Curriculum 训练策略改善推理质量。

**[Beyond Words: Augmenting Discriminative Richness via Diffusions in Unsupervised Prompt Learning](beyond_words_augmenting_discriminative_richness_via_diffusions_in_unsupervised_p.md)**

:   提出AiR（Augmenting discriminative Richness）方法，利用LoRA微调的Stable Diffusion生成合成图像构建辅助分类器，与文本分类器互补融合，将无监督prompt learning中的文本-图像匹配扩展为图像-图像匹配，显著提升细粒度/遥感等困难数据集上的分类准确率。

**[Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?](can_large_vision-language_models_correct_semantic_grounding_errors_by_themselves.md)**

:   系统研究了VLM在语义定位任务中的自我纠错能力，发现内在自我纠错（无外部反馈）反而损害性能（-7至-17点），但通过同一VLM作为二值验证器提供反馈的迭代纠错最多可提升8.4个百分点，揭示了反馈质量是自我纠错的关键瓶颈。

**[COAP: Memory-Efficient Training with Correlation-Aware Gradient Projection](coap_memory-efficient_training_with_correlation-aware_gradient_projection.md)**

**[CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation](comm_a_coherent_interleaved_image-text_dataset_for_multimodal_understanding_and_.md)**

:   针对现有交错图文数据集（MMC4/OBELICS）叙事连贯性差、实体风格不一致的核心问题，构建 CoMM 数据集（227K 文档、2.28M 图片），通过定向采集指令型内容 + 三维质量过滤策略确保文本连贯、图像一致、图文对齐，并提出 4 个交错生成评测任务。

**[Completion as Enhancement: A Degradation-Aware Selective Image Guided Network](completion_as_enhancement_a_degradation-aware_selective_image_guided_network_for.md)**

:   将图像增强重构为'补全'范式，通过退化感知选择机制引导网络聚焦于需要增强的区域，避免对已清晰区域的过度处理

**[Context-Aware Multimodal Pretraining](context-aware_multimodal_pretraining.md)**

:   本文提出LIxP（Language-Image Contextual Pretraining），通过在对比式图文预训练中引入交叉注意力上下文化机制，使视觉-语言模型在不损失零样本性能的前提下，显著提升了基于度量的few-shot适应能力（21个下游任务平均提升5%以上，样本效率提升可达4倍）。

**[Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)**

:   提出 SeGP-CL 框架，通过对抗性锚点（DPGD）精准探测新旧任务语义边界的脆弱区域，结合跨模态几何蒸馏（ACGD）和文本语义正则化（TSGR）保护 VLM 的跨模态几何结构，在五个持续学习 benchmark 上达到 SOTA。

**[COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts](counts_benchmarking_object_detectors_and_multimodal_large_language_models_under_.md)**

:   本文构建了COUNTS——一个包含14种自然分布偏移、222K+样本和119万+标注框的大规模OOD数据集，并提出O(OD)²和OODG两个基准，系统评估了目标检测器和多模态大模型在分布偏移下的泛化能力，发现即使是GPT-4o也仅能达到56.7%的定位准确率。

**[Critic-V: VLM Critics Help Catch VLM Errors in Multimodal Reasoning](critic-v_vlm_critics_help_catch_vlm_errors_in_multimodal_reasoning.md)**

:   本文提出Critic-V框架，将VLM推理过程解耦为Reasoner（推理器）和Critic（评价器），通过DPO训练的Critic模型提供自然语言反馈迭代优化推理路径，在8个基准上的5个超越GPT-4V，数学推理任务提升尤为显著（MathVista +11.8%）。

**[Cropper: Vision-Language Model for Image Cropping through In-Context Learning](cropper_vision-language_model_for_image_cropping_through_in-context_learning.md)**

:   本文提出Cropper框架，首次利用大型视觉-语言模型（VLM）的上下文学习（ICL）能力来解决图像裁剪任务，通过高效的prompt检索和基于反馈的迭代裁剪优化策略，无需任何训练即可在自由裁剪、主体感知裁剪和宽高比裁剪三种任务上大幅超越有监督SOTA方法。

**[Cross-modal Information Flow in Multimodal Large Language Models](cross-modal_information_flow_in_multimodal_large_language_models.md)**

:   通过"attention knockout"方法系统性地追踪 MLLM 中视觉和语言信息的流动路径，发现视觉信息分两阶段（先全局后局部）融入语言表征，最终在中间层由问题位置传播到最后位置生成答案。

**[Data Distributional Properties as Inductive Bias for Systematic Generalization](data_distributional_properties_as_inductive_bias_for_systematic_generalization.md)**

:   发现仅通过操纵训练数据的分布性质（多样性、突发性、潜在干预）就能诱导多模态遮蔽语言模型实现系统性泛化，其中增加属性多样性可将 OOD 形状预测准确率从 0.6% 提升到 90%，无需任何模型架构或训练策略修改。

**[Distraction is All You Need for Multimodal Large Language Model Jailbreaking](distraction_is_all_you_need_for_multimodal_large_language_model_jailbreaking.md)**

:   提出"分散假说"——通过构造高对比度多子图复合输入增加视觉复杂度来制造 OOD 效果，配合查询分解和精心设计的无害指令，实现对 GPT-4o 等闭源 MLLM 高达 42-64% 攻击成功率的黑盒越狱。

**[Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents](document_haystacks_vision-language_reasoning_over_piles_of_1000_documents.md)**

:   提出 DocHaystack 和 InfoHaystack 两个大规模文档检索基准（每个问题对应 1000+ 文档），以及 V-RAG——一个视觉中心的检索增强生成框架，在 Recall@1 上比最佳基线提升 9%-11%。

**[DocVLM: Make Your VLM an Efficient Reader](docvlm_make_your_vlm_an_efficient_reader.md)**

:   提出一种模型无关的 OCR 编码模块，将 OCR 提取的文本和布局信息压缩为 64 个 learned query token 并注入冻结的 VLM，在极低视觉 token 数量下大幅提升文档理解能力（DocVQA 最高 +30.6 分），并零样本泛化到多页文档。

**[DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models](dpc_dual-prompt_collaboration_for_tuning_vision-language_models.md)**

:   提出双提示协作（DPC）框架，通过冻结原始调优提示保持新类泛化、训练并行提示强化基类性能，配合加权解耦推理机制，作为即插即用模块在 4 种 prompt tuning 基线上一致提升 base-new 调和均值。

**[DynRefer: Delving into Region-level Multimodal Tasks via Dynamic Resolution](dynrefer_delving_into_region-level_multimodal_tasks_via_dynamic_resolution.md)**

:   模拟人眼"注视+扫视"的动态分辨率机制，围绕目标区域构建多层嵌套视图并在训练时随机采样、推理时根据任务或图像先验选择性组合，以 4.2B 参数在区域描述、属性检测、密集描述等任务上全面超越 7B+ 模型。

**[EgoLM: Multi-Modal Language Model of Egocentric Motions](egolm_multi-modal_language_model_of_egocentric_motions.md)**

:   提出统一自我中心动作追踪（稀疏传感器→全身动作）和动作理解（动作→语言描述）的多模态语言模型框架，通过 VQ-VAE 动作 tokenizer + GPT-2 骨干实现四种模态（文本、动作 token、传感器、视频）的联合建模，加入自我中心视频后追踪误差降低 10-20mm。

**[Embodied Scene Understanding for Vision Language Models via MetaVQA](embodied_scene_understanding_for_vision_language_models_via_metavqa.md)**

:   构建了一个基于 Set-of-Mark 标注和场景图的大规模 VQA 基准（430 万问题），系统评估 VLM 的空间推理和具身理解能力，发现在 MetaVQA 上微调可显著提升空间推理（+28 点），且训练于仿真数据的能力可零样本迁移到真实场景和未见过的闭环驾驶任务。

**[ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models](espire_a_diagnostic_benchmark_for_embodied_spatial_reasoning_of_vision-language_.md)**

:   提出 Espire，一个基于仿真环境的具身空间推理诊断基准，将 VLM 评估分解为定位和执行两阶段，通过全生成式范式系统评估 VLM 在多种空间推理维度和粒度上的能力。

**[Evaluating Model Perception of Color Illusions in Photorealistic Scenes](evaluating_model_perception_of_color_illusions_in_photorealistic_scenes.md)**

:   本文提出了一套自动化框架生成包含 19,000 张真实感颜色错觉图像的 RCID 数据集，首次系统性揭示了 VLM 确实存在类人的颜色感知偏差，并通过混合训练方法使模型能同时理解人类感知和真实像素值。

**[EventGPT: Event Stream Understanding with Multimodal Large Language Models](eventgpt_event_stream_understanding_with_multimodal_large_language_models.md)**

:   首个专为事件相机流设计的 MLLM，通过三阶段渐进训练范式（视觉-语言对齐→事件-语言对齐→指令微调）跨越异步事件数据与语言之间的巨大领域差距，在事件场景描述和 VQA 上大幅超越通用 MLLM。

**[Every SAM Drop Counts: Embracing Semantic Priors for Multi-Modality Image Fusion and Beyond](every_sam_drop_counts_embracing_semantic_priors_for_multi-modality_image_fusion_.md)**

:   利用 SAM 的语义先验通过持久注意力模块增强红外-可见光图像融合，再通过双层优化知识蒸馏将语义知识转移到仅 0.136M 参数的轻量子网络，实现无需 SAM 的 10.47ms 推理，同时在分割任务上超越所有专用融合方法 3+ mIoU。

**[FastVLM: Efficient Vision Encoding for Vision Language Models](fastvlm_efficient_vision_encoding_for_vision_language_models.md)**

:   提出混合卷积-Transformer视觉编码器 FastViTHD，通过 5 阶段架构实现 32× 空间下采样，在同等精度下比 ViT-L/14 生成 16× 更少的视觉 token 且编码速度提升 3.7×，TTFT 降低高达 85×。

**[Finer-CAM: Spotting the Difference Reveals Finer Details for Visual Explanation](finer-cam_spotting_the_difference_reveals_finer_details_for_visual_explanation.md)**

:   将 CAM 的解释目标从单类 logit $y^c$ 改为类间差值 $y^c - \gamma \cdot y^d$（目标类与相似类的 logit 差），零额外参数地将任何 CAM 方法升级为细粒度版本，使激活图从"整体轮廓"细化到"区分性局部细节"。

**[FLAIR: VLM with Fine-grained Language-informed Image Representations](flair_vlm_with_fine-grained_language-informed_image_representations.md)**

:   提出文本条件注意力池化（text-conditioned attention pooling），用文本 embedding 作为 query 从局部图像 token 中自适应聚合相关视觉信息，仅用 30M 合成描述数据训练就在细粒度检索和零样本分割上大幅超越用数十亿数据训练的 SigLIP/OpenCLIP。

**[Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion](florence-vl_enhancing_vision-language_models_with_generative_vision_encoder_and_.md)**

:   用生成式视觉基础模型 Florence-2 替换 CLIP 作为 VLM 视觉编码器，通过"深度-广度融合"（DBFusion）整合底层 DaViT 特征和三种任务提示（描述/OCR/定位）的高层特征，以单编码器 576 token 实现超越多编码器方案的性能。

**[ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models](forensiczip_more_tokens_are_better_but_not_necessary_in_forensic_vision-language.md)**

:   发现语义驱动的视觉 token 剪枝会丢弃 forensic 证据（篡改痕迹在低显著性区域），提出 ForensicZip 用 Birth-Death 最优传输量化帧间物理不连续性 + 高频先验保留取证信号，在 10% token 保留率下实现 2.97x 加速、90%+ FLOPs 降低且性能不降。

**[Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM](free_on_the_fly_enhancing_flexibility_in_test-time_adaptation_with_online_em.md)**

:   FreeTTA 提出一种无需训练、无需存储历史数据的测试时适应方法，通过在线 EM 算法显式建模目标域分布，利用 CLIP 零样本预测作为先验迭代估计每个类别的高斯分布参数，在 15 个数据集上稳定超越现有 TTA 方法。

**[From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons](from_multimodal_llms_to_generalist_embodied_agents_methods_and_lessons.md)**

:   GEA 将预训练的多模态 LLM（LLaVA-OneVision）通过学习式多具身动作分词器适配到操控/导航/游戏/UI控制/规划五大领域，先用 220 万条跨域专家轨迹 SFT，再用在线 PPO 强化学习微调，单模型在多个基准上超越或接近领域专用模型。

**[Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)**

:   GFS-VL 提出一种广义小样本 3D 点云分割框架，通过将 3D 视觉语言模型（3D VLM）生成的稠密但有噪声的伪标签与精确但稀疏的小样本标注协同融合——经由原型引导的伪标签筛选、自适应填充和 novel-base 混合增强——在现有和新设的高难度 benchmark 上取得了 SOTA 性能。

**[GeoMM: On Geodesic Perspective for Multi-Modal Learning](geomm_on_geodesic_perspective_for_multi-modal_learning.md)**

:   首次将测地距离（Geodesic Distance）引入多模态对比学习，通过构建层次化图结构高效计算样本间的流形距离，替代传统余弦距离，从而更准确地挖掘正负样本关系，提升图文检索、VQA等下游任务性能。

**[Ground-V: Teaching VLMs to Ground Complex Instructions in Pixels](ground-v_teaching_vlms_to_ground_complex_instructions_in_pixels.md)**

:   构建了Ground-V，一个包含50万指令-分割对的数据集，系统性解决真实世界指代分割中的五大挑战（幻觉引用、多对象、推理、多粒度、部件引用），训练后的VLM在gRefCOCO上N-Acc超越前SOTA 20%以上。

**[HalLoc: Token-Level Localization of Hallucinations for Vision Language Models](halloc_token-level_localization_of_hallucinations_for_vision_language_models.md)**

:   提出HalLoc，一个15.5万样本、覆盖VQA/指令跟随/图像描述三类任务的token级幻觉标注数据集，并基于此训练了轻量级幻觉检测模型HalLocalizer，可在不影响效率的前提下即插即用地集成到现有VLM中实现实时概率化幻觉检测。

**[HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_high-fidelity_in-context_learning_for_multimodal_tasks.md)**

:   通过对 attention 公式的精确分解，揭示 ICL 的效果本质上是 query-dependent 的标准自注意力输出与上下文 value 的动态混合，据此提出直接参数化"虚拟 KV 对"（低秩分解）来高保真模拟 ICL，仅 2.2M 参数即超越 MimIC/LoRA，且训练快 7.5 倍。

**[HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios](homesafe-bench_evaluating_vision-language_models_on_unsafe_action_detection_for_.md)**

:   提出HomeSafe-Bench——一个包含438个案例覆盖六大家庭功能区域的不安全动作检测基准，并设计HD-Guard双脑分层流式架构，以轻量FastBrain做高频筛查、大模型SlowBrain做深度推理，平衡家庭机器人安全监控的实时性与准确性。

**[Hyperbolic Safety-Aware Vision-Language Models](hyperbolic_safety-aware_vision-language_models.md)**

:   HySAC 提出在双曲空间中构建安全感知的视觉语言模型，通过蕴含锥（entailment cone）将安全/不安全内容映射到双曲空间的不同区域（安全内容靠近原点、不安全内容远离原点），使模型具备安全内容分类和动态重定向能力，在检索安全性和NSFW检测上显著超越现有遗忘方法。

**[Identifying and Mitigating Position Bias of Multi-image Vision-Language Models](identifying_and_mitigating_position_bias_of_multi-image_vision-language_models.md)**

:   本文发现多图大视觉语言模型（LVLM）存在严重的位置偏差——开源模型偏重后置图片、闭源模型忽视中间图片——并提出了一种无需训练的SoFt Attention（SoFA）方法，通过在图像间因果注意力与双向注意力之间做线性插值来缓解该偏差，在多个基准上提升了2~3%的平均准确率。

**[Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models](img-diff_contrastive_data_synthesis_for_multimodal_large_language_models.md)**

:   提出一种受对比学习启发的数据合成方法，自动生成包含细微物体差异的相似图像对及其差异描述，用于微调MLLM后在MMVP上超越GPT-4V和Gemini达12个点，并在8个通用MLLM基准上平均提升3.06%。

**[Improving Personalized Search with Regularized Low-Rank Parameter Updates](improving_personalized_search_with_regularized_low-rank_parameter_updates.md)**

:   本文提出POLAR方法，通过对CLIP文本编码器**最后一层**的value矩阵施加**rank-1的LoRA更新**加正则化，仅用少量样本即可学习个性化概念并保留通用知识，在DeepFashion2和ConCon-Chi基准上超越基于文本反转的先前方法4%~22%。

**[Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models](insight-v_exploring_long-chain_visual_reasoning_with_multimodal_large_language_m.md)**

:   Insight-V 提出一个包含数据生成 pipeline 和多智能体推理系统的视觉推理增强方案：通过渐进式生成+多粒度评估构建高质量长链推理数据，设计推理Agent和总结Agent协作解题，配合迭代DPO进一步提升推理质量，在7个视觉推理基准上实现平均7%的提升。

**[It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data](its_a_blind_match_towards_vision-language_correspondence_without_parallel_data.md)**

:   本文首次系统研究了在**完全无配对数据**的情况下，仅利用视觉和语言嵌入空间各自内部的成对距离进行"盲匹配"的可行性，提出了一种分解式Hahn-Grant QAP求解器（内存从 $O(N^4)$ 降到 $O(N^3)$），并在33个视觉模型×27个语言模型的大规模实验中证明了该匹配的可行性，甚至实现了无监督图像分类。

**[LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models](layoutvlm_differentiable_optimization_of_3d_layout_via_vision-language_models.md)**

:   提出LayoutVLM，利用VLM的语义知识生成包含数值位姿估计和空间关系约束的双重场景布局表示，通过可微分优化联合优化语义目标和物理合理性约束，在11种房间类型上显著超越现有方法。

**[LLaVA-Critic: Learning to Evaluate Multimodal Models](llava-critic_learning_to_evaluate_multimodal_models.md)**

:   LLaVA-Critic 是首个开源的通用多模态评估模型，通过在精心构建的113k评估指令数据上训练，使开源LMM具备了接近GPT-4o水平的Pointwise评分和Pairwise排序能力，并可作为奖励模型为迭代DPO提供有效的偏好信号，超越基于人类反馈训练的LLaVA-RLHF奖励模型。

**[MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding](marten_visual_question_answering_with_mask_generation_for_multi-modal_document_u.md)**

:   提出VQAMask预训练范式，在VQA文本解析基础上引入辅助的Mask生成任务（推理时丢弃），通过显式的空间对齐监督增强视觉编码器对文档图像中文字区域的感知能力，建立Marten模型在多项文档理解任务上达到8B级MLLM的SOTA。

**[Mastering Negation: Boosting Grounding Models via Grouped Opposition-Based Learning](mastering_negation_boosting_grounding_models_via_grouped_opposition-based_learni.md)**

:   构建首个包含正负语义描述的视觉定位数据集 D-Negation，并提出 Grouped Opposition-Based Learning (GOBL) 微调机制，通过对立语义约束显著增强 grounding 模型对否定语义的理解能力。

**[MBQ: Modality-Balanced Quantization for Large Vision-Language Models](mbq_modality-balanced_quantization_for_large_vision-language_models.md)**

:   发现大型VLM中视觉token和语言token对量化误差的敏感度差异超过10倍，提出MBQ方法在量化校准过程中引入基于梯度的模态平衡因子，在W3A16和W4A8设置下分别提升精度最高4.4%和11.6%，并实现1.4倍端到端加速。

**[Mimic In-Context Learning for Multimodal Tasks](mimic_in-context_learning_for_multimodal_tasks.md)**

:   本文从数学角度分析了ICL中in-context demonstrations (ICDs)对自注意力的"移位效应"，并提出MimIC方法通过在每个注意力头插入可学习移位向量+query依赖的缩放因子来模拟ICL行为，在VQA和Captioning任务上以仅0.26M参数超越32-shot ICL和所有现有移位向量方法。

**[MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output](mimo_a_medical_vision_language_model_with_visual_referring_multimodal_input_and_.md)**

:   本文提出MIMO——首个同时支持"视觉引用多模态输入"（用户通过点/框指定感兴趣区域）和"像素级定位多模态输出"（模型在文本回答中嵌入分割mask）的医学视觉语言模型，并构建了895K样本的MIMOSeg数据集，在多种医学VQA和分割任务上展示了独特的referring+grounding能力。

**[MMRL: Multi-Modal Representation Learning for Vision-Language Models](mmrl_multi-modal_representation_learning_for_vision-language_models.md)**

:   MMRL 提出了一个共享的、模态无关的可学习表征空间，将表征 token 投影到图像和文本编码器的高层（保留低层泛化知识），并通过解耦推理策略（基类用表征+类别特征，新类只用类别特征）在 15 个数据集上实现了 few-shot 适配与泛化的最优平衡，刷新了 base-to-novel 泛化的 SOTA。

**[Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](molmo_and_pixmo_open_weights_and_open_data_for_state-of-the-art_vision-language_.md)**

:   本文提出Molmo系列VLM和PixMo数据集，完全不依赖闭源VLM的合成数据，通过创新的数据收集方式（语音描述图像、交互式问答标注、2D指向标注）从零构建高质量训练数据，其72B模型在学术基准和人类评估中超越Claude 3.5 Sonnet和Gemini 1.5 Pro，仅次于GPT-4o。

**[Mosaic of Modalities: A Comprehensive Benchmark for Multimodal Graph Learning](mosaic_of_modalities_a_comprehensive_benchmark_for_multimodal_graph_learning.md)**

:   本文提出MM-Graph——首个同时包含文本和视觉节点属性的综合性图学习基准，涵盖7个不同规模的真实数据集和3类图任务（链接预测/节点分类/知识图谱补全），系统评估了视觉信息对图学习的影响，揭示了"多模态GNN不如传统GNN"和"特征对齐至关重要"等关键发现。

**[MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders](move-kd_knowledge_distillation_for_vlms_with_mixture_of_visual_encoders.md)**

:   本文提出MoVE-KD——首个从知识蒸馏角度将多个视觉编码器（CLIP/EVA/ConvNeXt/SAM）的特长融合到单个编码器的框架，通过Mixture-of-LoRA-Experts (MoLE)缓解多教师知识冲突、利用CLIP的[CLS]注意力自适应加权蒸馏token和教师，在LLaVA/LLaVA-NeXT上实现一致提升。

**[Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices](multi-layer_visual_feature_fusion_in_multimodal_llms_methods_analysis_and_best_p.md)**

:   本文系统研究了多模态 LLM 中多层视觉特征融合的两个核心问题：**(1) 如何选择最有效的视觉层**和 **(2) 如何最好地融合到语言模型中**，发现从不同表示相似性阶段各选一层 + 外部直接融合是最优实践。

**[Multimodal OCR: Parse Anything from Documents](multimodal_ocr_parse_anything_from_documents.md)**

:   提出 Multimodal OCR (MOCR) 范式，将文档中的文本和图形（图表、图标、UI 等）统一解析为结构化文本表示（包括 SVG 代码），3B 模型在 olmOCR-Bench 上达到 83.9 SOTA，图形解析超越 Gemini 3 Pro。

**[MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts](mv-math_evaluating_multimodal_math_reasoning_in_multi-visual_contexts.md)**

:   本文提出 MV-MATH 基准，包含 2,009 道高质量多图数学题（来自真实 K-12 场景），系统评估了 25 个多模态大模型在多图数学推理场景下的能力，发现所有模型远低于人类水平（最佳 Claude 仅 33.9%），揭示了多图数学推理仍是 MLLM 的重大挑战。

**[NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](nlprompt_noise-label_prompt_learning_for_vision-language_models.md)**

:   本文发现在 CLIP 提示学习中简单替换 MAE 损失就能显著提升对噪声标签的鲁棒性，并通过特征学习理论证明了这一现象，进而提出 NLPrompt 方法——结合基于最优传输的数据净化（PromptOT）将数据分为干净/噪声子集后分别用 CE 和 MAE 损失训练，在多种噪声设置下大幅超越现有方法。

**[NVILA: Efficient Frontier Visual Language Models](nvila_efficient_frontier_visual_language_models.md)**

:   NVILA 提出"先放大再压缩"(Scale-then-Compress)的范式，通过提升空间和时间分辨率后再压缩视觉Token，在保持甚至超越SOTA精度的同时，将训练成本降低1.9-5.1倍、推理预填充延迟降低1.6-2.2倍、解码延迟降低1.2-2.8倍。

**[Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding](octopus_alleviating_hallucination_via_dynamic_contrastive_decoding.md)**

:   本文揭示了多模态大模型幻觉的混合特性——不同样本甚至同一回答中的不同 token 面临不同类型的幻觉挑战（语言先验、视觉信息丢失、注意力偏差），据此提出 Octopus 框架，通过可学习的"眼睛"模块自适应识别幻觉类型，动态选择最适合的对比解码策略（"触手"），在四个基准上实现了 SOTA。

**[ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models](ode_open-set_evaluation_of_hallucinations_in_multimodal_large_language_models.md)**

:   本文提出 ODE（Open-set Dynamic Evaluation）协议，通过图结构建模现实世界物体概念及其分布关联，从中动态提取概念组合并生成合成测试图像，实现了开放集、持续更新的多模态幻觉评估，有效避免了现有静态基准可能存在的数据污染问题。

**[OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation](opening_a_comprehensive_benchmark_for_judging_open-ended_interleaved_image-text_.md)**

:   本文提出 OpenING 基准（5,400 条人工标注实例、56 个真实场景任务）和 IntJudge 评判模型（与人类判断一致率 82.42%），填补了开放式图文交错生成评估的真空，发现当前集成管线（如 Gemini+Flux）大幅领先端到端模型，但所有方法仍远不及人类标注质量。

**[Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy](optimus-2_multimodal_minecraft_agent_with_goal-observation-action_conditioned_po.md)**

:   提出Optimus-2，通过MLLM进行高层规划，结合Goal-Observation-Action Conditioned Policy (GOAP)进行底层控制，其中GOAP使用Action-guided Behavior Encoder建模观察-动作因果关系，并用MLLM对齐行为token与语言指令，在Minecraft原子任务上平均提升27%、长程任务提升10%、开放指令任务提升18%。

**[PARC: A Quantitative Framework Uncovering the Symmetries within Vision Language Models](parc_a_quantitative_framework_uncovering_the_symmetries_within_vision_language_m.md)**

:   提出PARC框架，通过**11种语言/视觉提示变异**、**可靠性评分**和**指标校准**三大支柱，首次系统量化分析了22个VLM在7个数据集上的提示敏感性，发现VLM继承了LLM的语言敏感性并在视觉域呈现对称表现，InternVL2家族对提示变化最鲁棒。

**[Period-LLM: Extending the Periodic Capability of Multimodal Large Language Model](period-llm_extending_the_periodic_capability_of_multimodal_large_language_model.md)**

:   提出Period-LLM——首个具备周期性感知能力的MLLM，采用"从易到难"渐进式训练范式（文本重复→宏观周期视频→微观周期信号），配合"抵抗逻辑遗忘"（RLO）梯度优化策略，在重复动作计数、rPPG心率估计等跨模态周期任务上显著超越现有MLLM。

**[Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy](playing_the_fool_jailbreaking_llms_and_multimodal_llms_with_out-of-distribution_.md)**

:   提出 JOOD 框架，通过将恶意输入进行分布外（OOD）化变换（如图像/文本混合），大幅提升模型不确定性，从而绕过 LLM 和 MLLM 的安全对齐防护，实现高成功率的黑盒越狱攻击。

**[Post-pre-training for Modality Alignment in Vision-Language Foundation Models](post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)**

:   提出 CLIP-Refine，一种介于预训练和微调之间的"后预训练"方法，通过随机特征对齐（RaFA）和混合对比蒸馏（HyCD）两个技术，仅用 1 个 epoch 在小数据集上训练即可缩小 CLIP 的模态间隙并提升零样本性能。

**[Quantization without Tears](quantization_without_tears.md)**

:   提出 QwT（Quantization without Tears）方法，通过在量化网络的每个 block 后添加一个轻量级线性补偿层来弥补量化信息损失，该补偿层参数可通过闭式解在2分钟内求得，在视觉、语言、多模态等多种任务上均显著提升了 PTQ 精度。

**[RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models](rap_retrieval-augmented_personalization_for_multimodal_large_language_models.md)**

:   提出 RAP（Retrieval-Augmented Personalization）框架，通过"记忆-检索-生成"三步实现 MLLM 的个性化：用外部数据库存储用户概念，用多模态检索器动态检索相关概念信息，再注入 MLLM 生成个性化响应，每个概念仅需1张图+描述即可，且支持实时更新。

**[Realistic Test-Time Adaptation of Vision-Language Models](realistic_test-time_adaptation_of_vision-language_models.md)**

:   本文揭示现有VLM测试时适应（TTA）/转导方法在realistic场景下（有效类数可变、非i.i.d.数据流）会严重损害CLIP的零样本鲁棒性，并提出StatA方法，通过在高斯聚类模型参数上引入基于文本编码器知识的KL散度正则化（统计锚），在所有部署场景中保持稳定提升。

**[Reasoning to Attend: Try to Understand How \<SEG\> Token Works](reasoning_to_attend_try_to_understand_how_seg_token_works.md)**

:   深入分析了 \<SEG\> token 在推理分割任务中的工作机制——发现其学到了与文本直接提及相似的语义特征并用于图像-文本语义对齐，在此基础上提出 READ 方法，将 \<SEG\> token 与图像 token 的相似度图转换为点提示，以即插即用方式指导 SAM 解码器生成更精确的分割掩码。

**[Recognition-Synergistic Scene Text Editing](recognition-synergistic_scene_text_editing.md)**

:   提出 RS-STE（Recognition-Synergistic Scene Text Editing）方法，将文字识别与文字编辑统一到一个多模态并行解码器中，利用识别模型隐式解耦风格与内容的天然能力来辅助编辑，并设计循环自监督微调策略使模型能在无配对标注的真实数据上有效训练。

**[Relation-Rich Visual Document Generator for Visual Information Extraction](relation-rich_visual_document_generator_for_visual_information_extraction.md)**

:   提出 RIDGE，一个关系丰富的视觉文档生成器，通过 LLM 生成层次化结构文本内容 + 自监督学习生成内容驱动的布局，合成带有实体类别和链接标注的文档图像，显著提升 VIE 模型在多个基准上的性能。

**[Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)**

:   通过分析 PEFT 在少样本适配中的学习动态，发现训练过程天然分为"任务级特征提取"和"可用类别特化"两个阶段，据此提出 2SFS：先调 LayerNorm 学通用特征，再训练线性分类器提升已知类判别，在 base-to-novel 和 all-to-all 两种设定下均达到或超越 SOTA。

**[Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector](rethinking_vision-language_model_in_face_forensics_multi-modal_interpretable_for.md)**

:   提出 M2F2-Det，首个同时输出深度伪造检测得分和文本解释的多模态人脸伪造检测器，通过 Forgery Prompt Learning 适配 CLIP 学习伪造特征、Bridge Adapter 融合 CLIP 与 deepfake 编码器特征、频域 token 引导 LLM 生成可信解释。

**[ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos](revisionllm_recursive_vision-language_model_for_temporal_grounding_in_hour-long_.md)**

:   提出 ReVisionLLM，首个能在小时级长视频中进行时序定位的视觉语言模型，模仿人类搜索策略递归处理视频——先粗粒度锁定相关片段，再逐级细化至精确时间边界，在 MAD 数据集上超越 SOTA +2.6% R1@0.1。

**[Revisiting Model Stitching in the Foundation Model Era](revisiting_model_stitching_in_the_foundation_model_era.md)**

:   系统研究异构 Vision Foundation Model（如 CLIP、DINOv2、SigLIP 2）之间的 stitchability，发现用 Final Feature Matching 预训练 stitch layer 可实现可靠拼接，并提出 VFM Stitch Tree 架构实现多 VFM 的高效共享。

**[RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness](rlaif-v_open-source_ai_feedback_leads_to_super_gpt-4v_trustworthiness.md)**

:   RLAIF-V 提出一套完全基于开源MLLM的反馈对齐框架，通过去混淆的候选回复生成策略和分治式反馈标注方法来产生高质量偏好数据，并结合DPO迭代训练与自反馈推理时扩展，使7B模型幻觉率降低80.7%，12B模型仅用自身反馈即超越GPT-4V的可信度。

**[RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics](robospatial_teaching_spatial_understanding_to_2d_and_3d_vision-language_models_f.md)**

:   RoboSpatial 构建了一个包含 1M 图像、5k 3D 扫描和 3M 空间关系标注的大规模机器人空间理解数据集，通过自动化 pipeline 从已有 3D 场景数据中生成三类空间问答对（空间上下文/兼容性/配置），并引入三种参考坐标系（自我/世界/物体），在多个 2D 和 3D VLM 上训练后显著提升空间推理性能，并在真实机器人操作实验中验证了有效性。

**[Seeing the Abstract: Translating the Abstract Language for Vision Language Models](seeing_the_abstract_translating_the_abstract_language_for_vision_language_models.md)**

:   提出 ACT（Abstract-to-Concrete Translator），通过 PCA 分析抽象-具象文本在 VLM 隐空间的表征差异，在推理时无训练地将抽象描述的表征向具象方向偏移，解决 VLM 对抽象语言理解不足的问题，在时尚领域文本-图像检索任务上显著超越微调模型。

**[Self-Evolving Visual Concept Library using Vision-Language Critics](self-evolving_visual_concept_library_using_vision-language_critics.md)**

:   提出 Escher 框架，通过 VLM 作为评判者 + LLM 作为概念生成器的迭代循环，自动进化视觉概念库以提升概念瓶颈模型在图像分类中的表现，在 CUB 数据集上将 LM4CV 从 63.26% 提升至 83.17%（+19.91%）。

**[Self-Supervised Spatial Correspondence Across Modalities](self-supervised_spatial_correspondence_across_modalities.md)**

:   将对比随机游走（CRW）框架扩展到跨模态像素级对应问题，通过同时学习模态内和模态间的循环一致性特征表示，在无需配对标注的情况下实现 RGB-Depth、RGB-Thermal、Photo-Sketch 等跨模态密集匹配，显著超越现有方法。

**[SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model](seqafford_sequential_3d_affordance_reasoning_via_multimodal_large_language_model.md)**

:   提出 Sequential 3D Affordance Reasoning 任务，构建180K指令-点云对基准，通过在3D MLLM中引入 `<SEG>` token 和多粒度语言-点云融合模块，从复杂人类指令中推理并分割出序列化的affordance区域。

**[Single Domain Generalization for Few-Shot Counting via Universal Representation Matching](single_domain_generalization_for_few-shot_counting_via_universal_representation_.md)**

:   提出首个面向少样本计数的单域泛化模型URM，通过将CLIP的通用视觉-语言表征蒸馏到可学习原型中参与相关性构建，在不损失域内性能的前提下大幅提升跨域泛化能力（MAE降低27.5%）。

**[SketchAgent: Language-Driven Sequential Sketch Generation](sketchagent_language-driven_sequential_sketch_generation.md)**

:   SketchAgent 无需任何训练或微调，通过为预训练多模态 LLM 设计网格画布坐标系统 + 上下文示例 + 贝塞尔曲线拟合的后处理流水线，使模型以逐笔画方式生成语义丰富、接近人类风格的草图，Top-1 识别率达人类水平的 85%，并支持交互式协作绘图和对话编辑。

**[Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves](skip_tuning_pre-trained_vision-language_models_are_effective_and_efficient_adapt.md)**

:   揭示 prompt tuning 冻结 VLM 参数既不促进知识迁移也未显著提升效率（仅减 6% 内存/16% 时间），提出 Skip Tuning 通过层级跳过（LSkip）和类别跳过（CSkip）缩短全微调的梯度传播流，实现 15× 时间效率和 6.4× 内存效率提升的同时精度更优。

**[SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Models](spa-vl_a_comprehensive_safety_preference_alignment_dataset_for_vision_language_m.md)**

:   SPA-VL 构建了一个包含 100,788 个四元组（问题、图像、优选回答、劣选回答）的大规模VLM安全偏好对齐数据集，覆盖6大领域/13类/53子类有害内容，基于12个VLM的多样化回答和全自动化标注流程，使用DPO/PPO训练后模型在安全性上大幅提升同时保持帮助性。

**[Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA](spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava.md)**

:   通过在 LLaVA 框架中系统替换图像编码器（CLIP/SigLIP/SigLIP2/AIMv2）和引入 2D-RoPE 位置编码，发现 VLM 的空间推理能力主要由编码器的训练目标决定，指望仅靠 2D 位置结构改善空间理解是不够的。

**[StarVector: Generating Scalable Vector Graphics Code from Images and Text](starvector_generating_scalable_vector_graphics_code_from_images_and_text.md)**

:   提出 StarVector，一个基于多模态大语言模型的 SVG 生成框架，将图像矢量化重新定义为逆渲染+代码生成任务，通过视觉语义理解直接生成包含丰富SVG基元（圆形、多边形、文本等）的紧凑SVG代码，在10个数据集3个任务上建立了新的SOTA。

**[BadVision: Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models](stealthy_backdoor_attack_in_self-supervised_learning_vision_encoders_for_large_v.md)**

:   首次揭示SSL视觉编码器对LVLM的后门安全威胁，提出BadVision——通过双层触发器优化和触发器聚焦后门学习机制，仅篡改视觉编码器即可使下游LVLM产生自由文本形式的视觉幻觉（ASR>99%），同时绕过SOTA检测方法。

**[Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks](steering_away_from_harm_an_adaptive_approach_to_defending_vision_language_model_.md)**

:   提出ASTRA，通过**图像归因**定位对抗图像中与越狱最相关的视觉token，构建**转向向量**表征有害响应方向，并在推理时进行**自适应激活转向**将模型远离有害方向，实现了比JailGuard低12%毒性分数、低18% ASR且快9倍的SOTA防御效果。

**[STING-BEE: Towards Vision-Language Model for Real-World X-ray Baggage Security Inspection](sting-bee_towards_vision-language_model_for_real-world_x-ray_baggage_security_in.md)**

:   构建了首个多模态X射线行李安全数据集**STCray**（46,642张图像-描述对，21类威胁含IED和3D打印枪），设计**STING协议**系统生成领域感知的高质量描述，并训练领域特化VLM **STING-BEE**，在场景理解、威胁定位、视觉定地和VQA四项任务上建立新基线，并展现SOTA跨域泛化能力。

**[Stop Learning It All to Mitigate Visual Hallucination, Focus on the Hallucination Target](stop_learning_it_all_to_mitigate_visual_hallucination_focus_on_the_hallucination.md)**

:   提出**TL-DPO**（Target-Learning DPO），将传统DPO的全句级偏好学习限制到**幻觉发生的目标chunk**和**对应的图像区域**，通过目标生成损失和目标条件损失排除无关信号，在LLaVA-1.5上将CHAIR_s从66.8降至20.1，同时LLaVA-Bench从63.4提升至71.2。

**[SVLTA: Benchmarking Vision-Language Temporal Alignment via Synthetic Video Situation](svlta_benchmarking_vision-language_temporal_alignment_via_synthetic_video_situat.md)**

:   提出**SVLTA**，一个通过合成模拟环境生成的视觉-语言时序对齐基准，包含25.3K动态场景、96种组合动作和77.1K高质量时序标注，具备**可控、组合、无偏**的时序分布，从时序问答、分布偏移敏感性和时序适应三个维度揭示当前VidLLM严重缺乏时序对齐能力（最强GPT-4o在IoU=0.5时R@1仅11.69%）。

**[Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](synthetic_data_is_an_elegant_gift_for_continual_vision-language_models.md)**

:   用 Stable Diffusion 从类名生成合成图像，通过对比蒸馏 + 图文对齐约束 + 自适应权重固化进行知识蒸馏，仅用每任务 1K 合成图像就超越使用 100K 真实 ImageNet 图像的持续学习方法 ZSCL。

**[Synthetic Visual Genome](synthetic_visual_genome.md)**

:   提出**SVG**（Synthetic Visual Genome）数据引擎，通过GPT-4在已有人工标注基础上**补全缺失关系**（Stage 1）和**Robin自蒸馏+GPT-4编辑**（Stage 2/SG-Edit）两阶段管道，生成146K图像、2.6M物体、5.6M关系的密集场景图数据集，训练的**Robin-3B**模型仅用<3M实例即超越300M实例训练的同尺寸模型，在指代表达理解上达到88.9的SOTA。

**[TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models](tapt_test-time_adversarial_prompt_tuning_for_robust_inference_in_vision-language.md)**

:   首个 VLM 测试时对抗防御方法，通过最小化多视图增强的熵一致性 + 对抗-干净 embedding 统计对齐来学习每个测试样本的防御性 prompt，仅需一步优化即可将 CLIP 对 AutoAttack 的鲁棒性从 0.1% 提升到 48.9%。

**[Taxonomy-Aware Evaluation of Vision-Language Models](taxonomy-aware_evaluation_of_vision-language_models.md)**

:   提出taxonomy-aware VLM评估框架，通过将VLM的自由文本输出映射到分类学树上，利用**层次精度(hP)**和**层次召回(hR)**来量化预测的正确性和具体性，解决了传统精确匹配/文本相似度无法给"部分正确"答案打分的问题。

**[Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution](teaching_large_language_models_to_regress_accurate_image_quality_scores_using_sc.md)**

:   提出DeQA-Score，通过将质量分数的**高斯分布离散化为soft label**（替代Q-Align的one-hot label），大幅减少离散化信息损失（10-35倍），并引入基于Thurstone模型的**fidelity loss**实现多IQA数据集联合训练，在分数回归任务上全面超越基线。

**[CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)**

:   CleanSight 发现 LVLM 后门攻击的机制不在像素层面而在注意力层面——触发器通过"注意力窃取"（trigger token 抢夺 text token 的注意力）来激活后门，据此提出了一种免训练、即插即用的 test-time 防御方法：通过检测跨模态注意力比例异常来识别中毒输入，再通过剪枝高注意力视觉 token 来中和后门，ASR 降至接近 0% 且几乎不影响模型性能。

**[Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces](thinking_in_space_how_multimodal_large_language_models_see_remember_and_recall_s.md)**

:   本文提出 VSI-Bench，一个基于视频的视觉空间智能基准（5000+ QA对），系统评估了 MLLM 的空间推理能力，发现空间推理是主要瓶颈，传统语言推理技术（CoT等）无法提升性能，但显式生成认知地图可改善空间距离推理。

**[Topo-R1: Detecting Topological Anomalies via Vision-Language Models](topo-r1_detecting_topological_anomalies_via_vision-language_models.md)**

:   发现现有 VLM（包括 GPT-5.2、Gemini-2.5）在拓扑异常检测上几乎为零（F1@0.5 < 1.5%），提出 Topo-R1 框架通过 SFT + GRPO（含拓扑感知复合 reward，集成 type-aware Hungarian matching + clDice）赋予 VLM 拓扑感知能力，最佳 F1@0.5 达 45.2%。

**[Towards Understanding How Knowledge Evolves in Large Vision-Language Models](towards_understanding_how_knowledge_evolves_in_large_vision-language_models.md)**

:   首次系统分析LVLM中多模态知识的演化过程，从**单token概率**、**token概率分布**和**特征编码**三个层次揭示知识演化的"关键层-突变层"双节点模式，将演化过程划分为**快速演化→稳定→突变**三个阶段，并发现深层突变与幻觉现象密切相关。

**[Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models](towards_zero-shot_anomaly_detection_and_reasoning_with_multimodal_large_language.md)**

:   首个专用于零样本异常检测和推理的 MLLM（Anomaly-OV），通过 Look-Twice Feature Matching 机制生成异常显著性图，配合视觉 Token 选择器聚焦可疑区域，在 9 个基准上实现 88.6% 平均 AUROC 的零样本异常检测 SOTA。

**[UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning](unem_unrolled_generalized_em_for_transductive_few-shot_learning.md)**

:   提出UNEM，将广义EM（GEM）算法的每次迭代展开为神经网络的一层，通过端到端学习自动优化**类平衡超参数λ**和**温度缩放T**，在11个细粒度数据集上实现vision-language设置下平均77.8%的准确率（vs. EM-Dirichlet的73.6%），vision-only设置下提升最高达10%。

**[Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly](unveiling_the_ignorance_of_mllms_seeing_clearly_answering_incorrectly.md)**

:   揭示MLLM"**理解了视觉内容但仍给出错误回答**"的普遍现象，构建包含12类正负样本对的**MMVU基准**，发现根因在于训练数据正样本偏倚和视觉token注意力不足，提出**MMVU-Train数据集**（112K正负样本对）+ **内容引导精炼（CGR）**+ **视觉注意力精炼（VAR）**三管齐下的解决方案。

**[UPME: An Unsupervised Peer Review Framework for Multimodal Large Language Model Evaluation](upme_an_unsupervised_peer_review_framework_for_multimodal_large_language_model_e.md)**

:   提出UPME框架，通过**无监督同行评审机制**、**视觉-语言评分系统**和**动态权重优化**，仅使用图像数据就能让多个MLLM互相出题评审，在MMStar上与人工评估的Pearson相关性达0.944，有效缓解了MLLM评估对人工标注的依赖和评审偏差问题。

**[VidComposition: Can MLLMs Analyze Compositions in Compiled Videos?](vidcomposition_can_mllms_analyze_compositions_in_compiled_videos.md)**

:   提出VidComposition基准，专门评估MLLM对**编辑合成视频**（影视、动画等）的构图理解能力，涵盖**5大类15个子任务**（镜头运动、叙事结构、角色理解等），对33个MLLM的评测揭示了模型与人类在电影级视频理解上的巨大差距（最佳模型63.3% vs 人类86.3%）。

**[Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding](video-xl_extra-long_vision_language_model_for_hour-scale_video_understanding.md)**

:   利用 LLM 内部的 KV 稀疏化能力实现长视频 token 压缩——引入视觉摘要 token（VST）将每段视频的视觉信息压缩到其 KV 中并卸载原始视觉 KV，配合动态压缩和课程学习，在单 A100 上处理 2048 帧，MLVU Dev 上超越 GPT-4o。

**[VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge](vila-m3_enhancing_vision-language_models_with_medical_expert_knowledge.md)**

:   提出VILA-M3框架，通过四阶段训练方案将医学领域专家模型（分割/分类）的知识按需集成到通用VLM中，在VQA、报告生成、分类等多个医学基准上以远小于Med-Gemini的模型规模（3B-40B vs 1.5T）实现了平均约9%的SOTA提升。

**[Vision-Language Model IP Protection via Prompt-based Learning](vision-language_model_ip_protection_via_prompt-based_learning.md)**

:   提出IP-CLIP框架，通过轻量级IP-Prompt学习（域token+图像token）和风格增强分支，在冻结CLIP骨干上实现VLM的知识产权保护——让模型在授权域保持高准确率的同时故意降低在非授权域的性能，授权域准确率下降为0%。

**[VisionZip: Longer is Better but Not Necessary in Vision Language Models](visionzip_longer_is_better_but_not_necessary_in_vision_language_models.md)**

:   VisionZip 发现视觉编码器（CLIP/SigLIP）生成的视觉Token存在严重冗余——仅少数Token聚集了绝大部分注意力和信息，基于此提出一种文本无关的Token选择与合并方法，在仅保留10%Token的情况下保持95%的模型性能，并实现8倍预填充加速。

**[Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning](visual_and_semantic_prompt_collaboration_for_generalized_zero-shot_learning.md)**

:   提出视觉语义提示协作网络（VSPCN），通过在预训练ViT中同时学习视觉提示和语义提示，并设计浅层弱融合+深层强融合机制，高效适配ViT提取语义相关的判别性视觉特征，在CUB/SUN/AWA2三个GZSL基准上均达到SOTA。

**[VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models](vlsi_verbalized_layers-to-interactions_from_large_to_small_vision_language_model.md)**

:   VLsI 提出了一种基于自然语言的层间蒸馏方法，通过在大小 VLM 的中间层引入 "verbalizer" 将特征映射到语言空间，并采用自适应层匹配策略对齐推理过程，使 2B/7B 小模型在 10 个 VL 基准上平均超过 GPT-4V 达 11.0%/17.4%，无需改变架构或增加参数。

**[What's in the Image? A Deep-Dive into the Vision of Vision Language Models](whats_in_the_image_a_deep-dive_into_the_vision_of_vision_language_models.md)**

:   本文通过 Attention Knockout 实验系统分析了 VLM（InternVL2-76B 和 LLaVA-1.5-7B）的视觉信息处理机制，揭示了三个关键发现：(1) query text token 充当全局图像描述器压缩高层视觉信息，(2) 中间层（约 25%）主导跨模态信息传递而早晚层贡献极少，(3) 细粒度物体细节通过空间局部化的方式从 image token 中提取。基于这些发现提出了 Image Re-prompting 应用，用仅 5% 的 image token 即可保持 96% 的 VQA 性能。

**[Words or Vision: Do Vision-Language Models Have Blind Faith in Text?](words_or_vision_do_vision-language_models_have_blind_faith_in_text.md)**

:   本文发现VLM存在"盲目信任文本"现象——当视觉与文本输入不一致时，模型系统性地偏向文本（即使文本是错误的），通过构建包含Match/Corruption/Irrelevance三类文本变体的benchmark评估了10个VLM，分析了5个影响因素，并证明SFT+文本增强可有效缓解，同时从理论上解释了该现象源于纯文本与多模态训练数据的不平衡。

**[Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](your_large_vision-language_model_only_needs_a_few_attention_heads_for_visual_gro.md)**

:   发现冻结 LVLM 中天然存在少量"定位头"（localization heads）持续捕捉文本语义对应的物体位置，仅用 3 个注意力头的注意力图即可实现超越微调 LISA-7B 的无训练视觉定位，RefCOCO val 达 86.5%。
