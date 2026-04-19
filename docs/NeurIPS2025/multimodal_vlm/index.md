---
title: >-
  NeurIPS2025 多模态VLM方向 143篇论文解读
description: >-
  143篇NeurIPS2025 多模态VLM方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧩 多模态VLM

**🧠 NeurIPS2025** · **143** 篇论文解读

**[A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1](a_frustratingly_simple_yet_highly_effective_attack_baseline.md)**

:   提出 M-Attack，通过对源图像做随机裁剪后与目标图像在嵌入空间做局部-全局/局部-局部匹配，配合多 CLIP 模型集成，使对抗扰动自然聚集在语义关键区域形成清晰的语义细节，在 GPT-4.5/4o/o1 等商业黑盒 LVLM 上实现 >90% 的定向攻击成功率。

**[A Multimodal Benchmark for Framing of Oil & Gas Advertising and Potential Greenwashing Detection](a_multimodal_benchmark_for_framing_of_oil_gas_advertising_an.md)**

:   构建了首个面向石油天然气行业视频广告的多模态框架分析基准数据集（706 个视频、13 种框架类型、50+ 实体、20 个国家），系统评估了 6 款 VLM 在检测 greenwashing 相关 framing 中的能力，发现 GPT-4.1 零样本在环境类标签上达 79% F1 但绿色创新仅 46%，揭示了隐式框架分析和文化背景理解仍是 VLM 的核心挑战。

**[ACT as Human: Multimodal Large Language Model Data Annotation with Critical Thinking](act_as_human_multimodal_large_language_model_data_annotation.md)**

:   提出ACT（Annotation with Critical Thinking）数据流水线，MLLM批量标注全部数据后由另一个MLLM作为批评者估计每条标注的错误概率，仅将高可疑样本交给人类审核，配合理论推导的ACT损失函数，在6个跨模态数据集上节省70-90%人工成本且下游性能差距<2%。

**[AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining](adalrs_lossguided_adaptive_learning_rate_search_for_efficien.md)**

:   提出AdaLRS，一种即插即用的在线学习率搜索算法，通过监控损失下降速度（loss velocity）来自适应调整学习率，将学习率超参搜索的成本从多次独立训练降低到单次训练，实现~50%的训练成本节省。

**[Adapting Vision-Language Models for Evaluating World Models](adapting_visionlanguage_models_for_evaluating_world_models.md)**

:   提出 UNIVERSE（UNIfied Vision-language Evaluator for Rollouts in Simulated Environments），通过对 PaliGemma 2 进行轻量级投影头微调（仅 0.07% 参数），构建统一的世界模型 rollout 语义评估器，在动作识别和角色识别任务上达到与任务专属模型相当的性能并与人类判断高度对齐。

**[ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources](admn_a_layerwise_adaptive_multimodal_network_for_dynamic_inp.md)**

:   提出 ADMN（Adaptive Depth Multimodal Network），通过两阶段训练——(1) Multimodal LayerDrop 微调使 backbone 适应任意层配置，(2) QoI感知控制器动态分配层预算给各模态——在严格计算约束下根据每个模态的信息质量(QoI)自适应分配层数，匹配全量模型精度同时减少 75% FLOPs 和 60% 延迟。

**[Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning](advancing_compositional_awareness_in_clip_with_efficient_fin.md)**

:   提出 CLIC，通过拼接两张图像并基于跨图词汇交换生成 hard negatives，同时创建多个正样本描述，仅微调 CLIP 文本编码器就能同时提升组合推理能力（SugarCrepe++ SOTA）和下游检索性能，打破了之前方法中组合性与检索性不可兼得的困局。

**[AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)**

:   提出细粒度 3D 具身推理任务（预测可操作元素的空间位置+运动类型+运动轴），通过将 3D 点云渲染为环视图并投影 affordance 候选，结合定制的 CoT 推理范式指导 MLLM 实现 SOTA，AP25 达 23.3%。

**[Aligning by Misaligning: Boundary-aware Curriculum Learning for Multimodal Alignment](aligning_by_misaligning_boundaryaware_curriculum_learning_fo.md)**

:   提出 BACL（Boundary-Aware Curriculum with Local Attention），通过可学习的边界感知负样本采样器（由易到难课程学习）+ 对比局部注意力损失（定位 token 级 mismatch），在 LAION-400M 上为 CLIP 带来 +32% R@1 提升，并在四个大规模基准上取得 SOTA。

**[AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making](antigrounding_lifting_robotic_actions_into_vlm_representatio.md)**

:   反转传统指令接地范式——不将 VLM 知识压缩到中间表征（符号技能或约束），而是将候选机器人轨迹渲染到多视角场景图像中，直接在 VLM 的原生高维表征空间中评估动作方案，实现零样本闭环机器人操作控制。

**[Approximate Domain Unlearning for Vision-Language Models](approximate_domain_unlearning_for_visionlanguage_models.md)**

:   提出 Approximate Domain Unlearning (ADU) 新任务，通过 Domain Disentangling Loss (DDL) 和 Instance-wise Prompt Generator (InstaPG) 两个模块，让预训练 VLM 选择性遗忘指定域（如插画、素描）的识别能力，同时保持其他域（如真实照片）的分类精度，在四个多域数据集上大幅超越所有基线。

**[AQuaMaM: An Autoregressive, Quaternion Manifold Model for Rapidly Estimating Complex SO(3) Distributions](aquamam_an_autoregressive_quaternion_manifold_model_for_rapidly_estimating_compl.md)**

:   提出AQuaMaM——一种基于Transformer的自回归四元数流形模型，将单位四元数的三个投影分量建模为受几何约束的均匀分布混合，在SO(3)旋转流形上实现精确似然计算和快速采样，比IPDF推理速度快52倍、对数似然高14%，且采样分布与真实分布匹配极为精确。

**[Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering](are_vision_language_models_ready_for_clinical_diagnosis_a_3d_medical_benchmark_f.md)**

:   本文提出 DeepTumorVQA，一个针对腹部CT肿瘤的3D诊断级视觉问答基准，包含9,262个CT体积（370万切片）和395K专家级问题，系统评估了4个先进VLM的临床诊断能力，发现当前模型在测量任务上尚可但在病灶识别和推理上远未达到临床要求。

**[Attention! Your Vision Language Model Could Be Maliciously Manipulated](attention_your_vision_language_model_could_be_maliciously_manipulated.md)**

:   本文提出 Vision-language Model Manipulation Attack (VMA)，一种结合一阶和二阶动量优化及可微变换机制的图像对抗攻击方法，能够精确操控VLM的每个输出token，可用于实施多种攻击（越狱、劫持、隐私泄露、DoS、海绵样本）同时也可用于版权保护水印注入。

**[Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](balanced_token_pruning_accelerating_vision_language_models_b.md)**

:   提出 Balanced Token Pruning (BTP)，通过联合考虑剪枝对当前层（局部）和后续层（全局）的影响，在浅层侧重多样性保留以维护下游表示质量、在深层侧重注意力选择以保持局部输出一致性，在 LLaVA/Qwen2.5-VL 等多个 LVLM 上仅保留 22% 视觉 token 即保持原模型 98% 性能。

**[Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)**

:   提出 BTB3D，一种基于因果卷积编解码器 + 3D Haar 小波压缩 + 三阶段渐进训练的 3D CT tokenizer，在放射报告生成和文本条件 CT 合成两大下游任务上大幅刷新 SOTA，证明"更好的 token 比更大的语言模型更重要"。

**[Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability](beyond_greedy_exits_improved_early_exit_decisions_for_risk_control_and_reliabili.md)**

:   UAT（Unsupervised Adaptive Thresholding）为早退 DNN 设计了可靠性函数来评估中间层输出质量，并用多臂赌博机（MAB）算法在推理时动态学习最优退出阈值，实现 1.7-2.1× 加速且性能损失 <2%，同时对分布偏移鲁棒。

**[BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](bioclip_2_emergent_properties_from_scaling_hierarchical_contrastive_learning.md)**

:   BioCLIP 2 在 TreeOfLife-200M（2.14 亿图像/95.2 万物种）上用层级对比学习训练 ViT-L，零样本物种识别比 BioCLIP 提升 18%，并发现规模化带来的涌现性质——嵌入自动编码生态关系（如达尔文雀喙大小排列）且种内变异与种间差异正交。

**[Breaking the Compression Ceiling: Data-Free Pipeline for Ultra-Efficient Delta Compression](breaking_the_compression_ceiling_data-free_pipeline_for_ultra-efficient_delta_co.md)**

:   提出 UltraDelta——首个无数据 delta 权重压缩流水线，通过方差引导的混合稀疏分配、分布感知压缩和迹范数引导缩放三个组件，在 LLM/NLP/视觉/多模态模型上实现最高 224× 的超高压缩比且性能不降甚至超越微调模型。

**[BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models](bridgevla_input-output_alignment_for_efficient_3d_manipulation_learning_with_vis.md)**

:   提出 BridgeVLA，通过将 3D 点云投影为多视角 2D 图像并以 2D 热力图作为中间表示来对齐输入输出空间，实现了高效且有效的 3D 机器人操作学习。

**[Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning](can_llms_reason_over_non-text_modalities_in_a_training-free_manner_a_case_study_.md)**

:   提出 In-Context Representation Learning（ICRL），首个训练无关框架，将非文本模态基础模型（FM）的表征注入纯文本 LLM 进行少样本推理，通过 PCA 文本注入和最优传输嵌入对齐两种策略实现跨模态知识利用。

**[Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?](can_multi-modal_llms_provide_live_step-by-step_task_guidance.md)**

:   提出 Qualcomm Interactive Cooking 基准和 LiveMamba 模型，首次系统评估多模态 LLM 在实时流式视频中提供分步任务指导（包括指令下发、完成检测和错误反馈）的能力。

**[CAPability: A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Thoroughness](capability_a_comprehensive_visual_caption_benchmark_for_eval.md)**

:   提出CAPability，一个涵盖6大视角12个维度的综合视觉描述评测基准，通过人工标注近11K图像/视频的视觉元素（而非句子），同时评估描述的正确性（precision）和全面性（hit），并引入"知道但说不出"（$K\bar{T}$）指标揭示MLLM在QA与caption任务之间的显著能力差距。

**[Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](causalllava_causal_disentanglement_for_mitigating_hallucinat.md)**

:   揭示 MLLM 中物体幻觉的表示层根因——数据集共现偏差导致的语义纠缠，提出双路因果解纠缠框架（Causal-Driven Projector + Causal Intervention Module），通过后门调整在 projector 和最终 Transformer 层分离共现物体表示，使 MME-Perception 提升 22.6%。

**[ChartMuseum: 测试大型视觉语言模型的图表视觉推理能力](chartmuseum_testing_visual_reasoning_capabilities_of_large_v.md)**

:   提出ChartMuseum图表问答基准，包含1162个专家标注问题和184个来源的真实图表，首次系统区分视觉推理与文本推理能力，揭示当前最强模型Gemini-2.5-Pro仅63.0%而人类达93%，视觉推理性能比文本推理低35%-55%。

**[CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models](choice_benchmarking_the_remote_sensing_capabilities_of_large_vision-language_mod.md)**

:   提出 CHOICE，一个面向遥感领域的大规模多层级 VLM 基准，包含 10,507 道全新采集题目，覆盖感知与推理 2 大维度、6 个子维度、23 个叶任务，首次实现对 VLM 遥感能力的系统化与客观化评估。

**[CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization](coido_efficient_data_selection_for_visual_instruction_tuning_via_coupled_importa.md)**

:   提出 CoIDO，一个双目标优化数据选择框架，通过联合优化数据重要性和多样性，仅用 20% 随机数据训练轻量评分器，即可从 LLaVA-665K 中选出 20% 子集达到全量微调 98.2% 的性能，同时计算开销为所有方法最低。

**[Context Informs Pragmatic Interpretation in Vision-Language Models](context_informs_pragmatic_interpretation_in_vision-language_models.md)**

:   通过迭代参考游戏（iterated reference games）系统评估 VLM 的语用推理能力，发现模型在无上下文时表现远逊于人类，但在获得相关对话历史后能快速学习达到约 80% 准确率，揭示了 VLM 对上下文信息的强烈依赖性。

**[Continual Multimodal Contrastive Learning](continual_multimodal_contrastive_learning.md)**

:   本文首次形式化定义了持续多模态对比学习（CMCL）问题，提出双侧零空间梯度投影（DNS）方法，将新数据的梯度投影到不影响旧知识的子空间上，在 7 个数据集上实现了稳定性和可塑性的最佳平衡。

**[CovMatch: Cross-Covariance Guided Multimodal Dataset Distillation with Trainable Text Encoder](covmatch_crosscovariance_guided_multimodal_dataset_distillat.md)**

:   提出 CovMatch，通过将多模态对比学习的双层优化简化为跨协方差矩阵对齐的闭式解，首次实现图文双编码器的联合优化进行多模态数据集蒸馏，仅用 500 个合成图文对在 Flickr30K 上获得 38.4 平均检索精度（+6.8% 超越 SOTA LoRS），在极端数据高效场景下大幅超越冻结文本编码器的方法。

**[CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning](cyin_cyclic_informative_latent_space_for_bridging_complete_and_incomplete_multim.md)**

:   提出 CyIN 框架，通过 token 级和 label 级信息瓶颈（IB）构建信息化潜空间，结合循环跨模态翻译重建缺失信息，在单一统一模型中同时优化完整和不完整多模态学习。

**[DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding](danmakutppbench_a_multimodal_benchmark_for_temporal_point_pr.md)**

:   构建首个多模态时间点过程基准DanmakuTPPBench：DanmakuTPP-Events提供7250个序列共1080万弹幕事件（时间-文本-视频三模态天然对齐），DanmakuTPP-QA通过多智能体pipeline自动生成10类推理问答，系统暴露了经典TPP模型和MLLM在多模态事件动态理解上的显著短板。

**[Don't Just Chase "Highlighted Tokens" in MLLMs: Revisiting Visual Holistic Context Retention](dont_just_chase_highlighted_tokens_in_mllms_revisiting_visual_holistic_context_r.md)**

:   提出 HoloV，一个即插即用的视觉 token 剪枝框架，通过在不同空间裁剪区域自适应分配剪枝预算，保留全局视觉上下文而非仅保留注意力高亮 token，在 LLaVA-1.5 上剪枝 88.9% token 仍保留 95.8% 原始性能。

**[DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)**

:   DOTA提出将测试时自适应从"缓存样本实例"范式转变为"持续估计测试数据分布"范式，通过在线高斯判别分析结合零样本预测概率估计类别分布，实现无梯度、抗遗忘的高效测试时自适应，在10个跨域基准上平均准确率超越所有基线。

**[DynamicVL: Benchmarking MLLMs for Dynamic City Understanding](dynamicvl_benchmarking_multimodal_large_language_models_for_dynamic_city_underst.md)**

:   提出 DVL-Suite 框架，包含 DVL-Bench 基准和 DVL-Instruct 指令微调数据集，覆盖 42 座美国城市、14,871 张高分辨率多时相遥感影像，系统评估 18 个 MLLM 在长期城市动态理解上的能力，并开发了 DVLChat 基线模型。

**[Efficient Multi-modal Large Language Models via Progressive Consistency Distillation](efficient_multi-modal_large_language_models_via_progressive_consistency_distilla.md)**

:   提出EPIC框架，通过渐进式一致性蒸馏（Token和Layer两个维度）解决视觉token压缩训练中特征空间扰动导致的学习困难，在不修改模型架构的前提下实现高效多模态LLM。

**[ElasticMM: Efficient MLLM Serving with Elastic Multimodal Parallelism](elasticmm_efficient_multimodal_llms_serving_with_elastic_multimodal_parallelism.md)**

:   提出弹性多模态并行（EMP）范式和 ElasticMM 系统，通过模态感知负载均衡和弹性分区调度将多模态推理的不同阶段解耦到独立实例，相比 vLLM TTFT 降低最高 4.2 倍、吞吐量提升 3.2-4.5 倍。

**[READ: Enhancing Compositional Reasoning in CLIP via Reconstruction and Alignment of Text Descriptions](enhancing_compositional_reasoning_in_clip_via_reconstruction.md)**

:   提出 READ 微调方法，通过两个辅助目标——(1) token-level 重建（冻结解码器从文本嵌入重建替代描述）和 (2) sentence-level 对齐（强制改述的嵌入一致）——增强 CLIP 文本编码器的组合推理能力，在 5 个组合推理基准上达到 SOTA（超 NegCLIP 4.5%，超 FSC-CLIP 4.1%）。

**[Enhancing Outcome Reward-Based RL Training of MLLMs with Self-Consistency Sampling](enhancing_the_outcome_reward-based_rl_training_of_mllms_with_self-consistency_sa.md)**

:   针对多模态多选题中"结果奖励 RL 训练导致不忠实推理轨迹"的问题，提出 Self-Consistency Sampling (SCS)，通过截断-重采样和视觉扰动获得一致性奖励来惩罚虚假推理，搭载 RLOO 后在六个基准上平均提升 7.7 个百分点。

**[Enhancing Vision-Language Model Reliability with Uncertainty-Guided Dropout Decoding](enhancing_visionlanguage_model_reliability_with_uncertaintyg.md)**

:   提出Dropout Decoding——将视觉token投影到文本空间后量化其认知不确定性，选择性遮掩高不确定性视觉token并通过多组遮掩结果的集成投票增强输出可靠性，无需额外训练即可显著减少LVLM的对象幻觉。

**[Evaluating Multimodal Large Language Models on Core Music Perception Tasks](evaluating_multimodal_large_language_models_on_core_music_perception_tasks.md)**

:   本文通过三项核心音乐感知任务（切分节奏评分、移调检测、和弦辨识）系统性评估了多模态LLM在音频与MIDI两种输入下的表现，揭示了模型在符号推理上接近理想但在音频感知上存在显著缺陷的关键差距。

**[ExGra-Med: Extended Context Graph Alignment for Medical Vision-Language Models](exgra-med_extended_context_graph_alignment_for_medical_vision-language_models.md)**

:   ExGra-Med 提出了一种多图对齐（multi-graph alignment）框架，通过联合对齐图像、指令响应和扩展上下文描述在潜空间中的图结构关系，仅用10%预训练数据即可匹配 LLaVA-Med 的100%数据性能，并在多个医学VQA任务上超越现有SOTA。

**[FineGRAIN: Evaluating Failure Modes of Text-to-Image Models with Vision Language Model Judges](finegrain_evaluating_failure_modes_of_text-to-image_models_with_vision_language_.md)**

:   FineGRAIN 提出了一个结构化的联合评测框架，通过定义27种细粒度失败模式和利用 VLM+LLM agentic pipeline 来同时评估文本到图像模型的 prompt 遵循能力和视觉语言模型的图像理解能力，揭示了两类模型在特定任务上的系统性缺陷。

**[First SFT, Second RL, Third UPT: Continual Improving Multi-Modal LLM Reasoning via Unsupervised Post-Training](first_sft_second_rl_third_upt_continual_improving_multi-modal_llm_reasoning_via_.md)**

:   提出 MM-UPT 框架，在 SFT 和 RL 之后引入第三阶段"无监督后训练"，通过多数投票作为伪奖励信号结合 GRPO 实现 MLLM 的自我改进，在 MathVista 上将 Qwen2.5-VL-7B 从 66.3% 提升至 72.9%。

**[FlexAC: Towards Flexible Control of Associative Reasoning in Multimodal Large Language Models](flexac_towards_flexible_control_of_associative_reasoning_in_multimodal_large_lan.md)**

:   FlexAC 发现 MLLM 的联想推理行为主要编码在中间层，通过从幻觉响应中提取引导向量并在推理时注入中间层表示，实现忠实性与创造力的灵活调控——幻觉率降低 29%(CHAIR)，创造力提升 5.8×(Creation-MMBench)，且无需训练。

**[FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models](flowcut_rethinking_redundancy_via_information_flow_for_effic.md)**

:   从信息流（Information Flow）视角重新理解VLM中视觉token冗余性的涌现机制，提出FlowCut框架通过层自适应剪枝比例、多标准融合评分和累积重要性跟踪实现与模型内在信息传播行为对齐的token剪枝，在LLaVA-1.5-7B上以88.9% token减少率超越SOTA 1.6%，LLaVA-NeXT-7B上以94.4%减少率超越4.3%。

**[FlySearch: Exploring how vision-language models explore](flysearch_exploring_how_vision-language_models_explore.md)**

:   FlySearch 提出了一个基于 Unreal Engine 5 的 3D 户外真实感环境，评估 VLM 的探索能力，发现最先进的 VLM 在简单搜索任务上也无法可靠完成，且与人类的差距随任务难度增加而急剧扩大。

**[FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)**

:   提出 FOCUS，一种无需训练的视觉裁剪方法，利用 MLLM 内部 KV-cache 中 value 特征的余弦相似度构建目标相关性图，高效定位问题相关的图像区域，在细粒度 VQA 上实现与 SOTA 可比的精度，同时计算效率提升 3-6.5 倍。

**[ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation](forcevla_enhancing_vla_models_with_a_force-aware_moe_for_contact-rich_manipulati.md)**

:   提出 ForceVLA，在 VLA 框架中将 6 轴力/力矩传感引入为一等模态，通过 FVLMoE（力感知混合专家）模块在动作解码阶段动态融合视觉-语言嵌入与实时力反馈，在 5 项接触密集操作任务上平均成功率提升 23.2%，个别任务达 80%。

**[GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images](gem_empowering_mllm_for_grounded_ecg_understanding_with_time_series_and_images.md)**

:   提出 GEM，首个统一 ECG 时间序列、12 导联 ECG 图像和文本的多模态大语言模型，通过双编码器框架、跨模态对齐和知识引导的指令数据生成，实现了基于可量化生理特征的接地心电图诊断，诊断准确率提升 7.4%，可解释性提升 22.7%，接地能力提升 25.3%。

**[Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling](generate_but_verify_reducing_hallucination_in_visionlanguage.md)**

:   提出REVERSE框架，首次将生成调整和事后验证统一到单个VLM中：通过1.3M半合成样本的幻觉感知训练+推理时回溯重采样，使VLM能在生成过程中自动检测并修正幻觉，在CHAIR-MSCOCO上降低12%、HaloQuest上提升34%。

**[GeoRanker: Distance-Aware Ranking for Worldwide Image Geolocalization](georanker_distance-aware_ranking_for_worldwide_image_geolocalization.md)**

:   提出 GeoRanker，一种距离感知排序框架，利用大视觉语言模型建模查询-候选之间的空间关系，通过多阶距离损失实现全球图像地理定位的 SOTA。

**[GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)**

:   提出GLSim，一种无训练的LVLM物体幻觉检测方法，通过融合全局场景相似度（物体token与最后instruction token的余弦相似度）和局部视觉定位相似度（物体token与Visual Logit Lens定位的Top-K图像patch的余弦相似度），在MSCOCO上以83.7% AUROC超越SVAR 9%、Internal Confidence 10.8%。

**[GoalLadder: Incremental Goal Discovery with Vision-Language Models](goalladder_incremental_goal_discovery_with_vision-language_models.md)**

:   提出 GoalLadder，利用 VLM 渐进式发现并排序候选目标状态，结合 ELO 评分系统抵抗噪声反馈，在学习的嵌入空间中定义距离奖励，仅凭单条语言指令就能训练 RL 智能体达到约 95% 的成功率。

**[Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)**

:   提出 MAPLE 框架，利用现成 MLLM 的内在模态对齐能力自动构建偏好数据，通过 Relative Preference Alignment（RPA）损失引导跨模态表示学习，在细粒度检索任务上取得显著提升。

**[HAWAII: Hierarchical Visual Knowledge Transfer for Efficient VLM](hawaii_hierarchical_visual_knowledge_transfer_for_efficient_vision-language_mode.md)**

:   提出 Hawaii 框架，通过混合 LoRA 适配器（MoLA）和分层知识蒸馏（HKD），将多个视觉专家的知识蒸馏到单个视觉编码器中，在不增加推理成本的前提下显著提升 VLM 的视觉理解能力。

**[HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation](hermesflow_seamlessly_closing_the_gap_in_multimodal_understanding_and_generation.md)**

:   首次揭示统一多模态大模型中理解能力普遍强于生成能力的现象，提出 HermesFlow 框架，通过同源偏好数据构建配对理解-生成偏好对，利用 Pair-DPO 和自博弈迭代优化，在不引入外部高质量数据的情况下同步提升理解与生成能力并缩小两者差距。

**[Hierarchical Self-Attention: Generalizing Neural Attention Mechanics to Multi-Scale Problems](hierarchical_self-attention_generalizing_neural_attention_mechanics_to_multi-sca.md)**

:   从熵最小化第一性原理推导出层次化自注意力（HSA）机制，为嵌套信号（多模态、多尺度数据）提供理论最优的注意力计算方法，并证明 HSA 是在保持层次约束下最接近标准 Softmax 注意力的 KL 散度最优解。

**[HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](hope_hybrid_of_position_embedding_for_long_context_visionlan.md)**

:   首次从理论上分析多模态RoPE的频率分配策略对长上下文VLM的影响，提出HoPE，将最低频率设为零用于时间建模以保证语义偏好性质，配合动态时间缩放机制，在长视频理解任务上提升8.35%、检索任务上提升22.23%。

**[iFinder: Structured Zero-Shot VLM Grounding for Dash-Cam Video Reasoning](ifinder_structured_zero-shot_vision-based_llm_grounding_for_dash-cam_video_reaso.md)**

:   提出 iFinder，一个模块化免训练框架，将行车记录仪视频解耦为感知（结构化场景表示）与推理（LLM），通过层级数据结构和三块式提示策略使 LLM 获得可解释的时空推理能力，在四个驾驶视频基准上零样本超越端到端 V-VLM，事故推理准确率提升高达 39%。

**[Intervene-All-Paths: Unified Mitigation of LVLM Hallucinations across Alignment Formats](intervene-all-paths_unified_mitigation_of_lvlm_hallucinations_across_alignment_f.md)**

:   提出 AllPath，一个基于 Transformer 因果架构的多路径幻觉干预框架，首次发现 LVLM 的幻觉不来自单一因果路径而是 image-to-input-text、image-to-output-text、text-to-text 三条路径的交互，并且模型会根据问答对齐格式自适应选择不同路径；通过为每条路径设计轻量级关键 head 识别方法并自适应干预，在 POPE、MCQ-POPE、CHAIR、MME 四个不同格式 benchmark 上一致降低幻觉。

**[JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models](jailbound_jailbreaking_internal_safety_boundaries_of_vision-language_models.md)**

:   受 Eliciting Latent Knowledge (ELK) 框架启发，首次揭示 VLM 在 fusion layer 潜空间中存在可近似的安全决策边界，提出 JailBound 两阶段攻击框架（Safety Boundary Probing + Safety Boundary Crossing），通过联合优化图像和文本对抗扰动跨越该边界，在白盒和黑盒场景分别达到 94.32% 和 67.28% 平均攻击成功率，显著超越 SOTA。

**[Learning Shared Representations from Unpaired Data](learning_shared_representations_from_unpaired_data.md)**

:   提出 SUE (Spectral Universal Embedding)，首次证明几乎完全依赖非配对数据即可学习跨模态共享表示：通过独立的频谱嵌入从各模态随机游走中提取模态不变的"通用"结构，再用极少量配对样本（~100对）做 CCA 线性对齐 + MMD 非线性微调，在检索上超越使用同等配对数的对比学习 250%+。

**[Learning to Instruct for Visual Instruction Tuning](learning_to_instruct_for_visual_instruction_tuning.md)**

:   提出 L2T（Learning to Instruct），仅通过将训练损失扩展到指令序列（不再只在回答上计算 loss）来改善视觉指令调优——无额外数据和几乎零计算开销，在 16 个多模态基准上获得高达 9% 的相对提升，captioning 提升 18%，同时缓解幻觉。

**[Learning to Steer: Input-dependent Steering for Multimodal LLMs](learning_to_steer_input-dependent_steering_for_multimodal_llms.md)**

:   针对现有模型引导(steering)方法使用固定方向向量无法适配不同输入的局限，提出 L2S (Learn-to-Steer)：先通过输入特定的对比提示生成理想的引导向量（P2S），再训练一个轻量 2 层 MLP 从输入上下文预测该向量，以极低开销实现了输入依赖的行为引导，在安全执行和幻觉缓解两个应用上显著优于静态 steering 基线。

**[Metacognitive Sensitivity for Test-Time Dynamic Model Selection](metacognitive_sensitivity_for_test-time_dynamic_model_selection.md)**

:   借鉴人类认知科学中的元认知灵敏度（meta-d'）概念，提出一种测试时动态模型选择框架：用 meta-d' 量化模型"知道自己知不知道"的能力，结合即时置信度构成上下文向量，通过 contextual bandit 在线选择最优模型，在多数据集上超越单模型性能。

**[MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning](midas_misalignment-based_data_augmentation_strategy_for_imbalanced_multimodal_le.md)**

:   首次提出将跨模态不对齐样本作为有监督训练信号（而非噪声/干扰）来缓解多模态学习中的模态不平衡问题，设计 MIDAS 数据增强框架：通过置信度标注不对齐样本 + 弱模态加权 + 难样本加权三重机制，在四个多模态分类基准上显著超越现有方法。

**[Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)**

:   发现 CLIP 在图像损坏下的性能退化根源在于**嵌入方差坍缩**——类内与类间方差同步缩小导致嵌入空间判别性丧失；提出 Mint，通过最大化伪标签类间方差（PL-inter）在线修复嵌入几何，仅凭均值累加器和梯度累加器两个极简组件即可在 BS=1 的在线场景下稳定提升 CLIP 在多种损坏基准上的分类精度，同时比最强 baseline 快 45 倍。

**[MIRAGE: A Benchmark for Multimodal Information-Seeking and Reasoning in Agriculture](mirage_a_benchmark_for_multimodal_information-seeking_and_reasoning_in_agricultu.md)**

:   MIRAGE 是首个基于真实农业专家咨询对话（35,000+条）构建的多模态基准，评估视觉语言模型在领域级实体识别、因果推理和"澄清还是回答"决策方面的能力，揭示了即使 GPT-4.1 识别准确率也仅 43.9% 的严峻挑战。

**[MM-OPERA: Benchmarking Open-ended Association Reasoning for Large Vision-Language Models](mm-opera_benchmarking_open-ended_association_reasoning_for_large_vision-language.md)**

:   提出 MM-OPERA，一个包含 11,497 实例的开放式联想推理基准，通过远程物品关联（RIA）和上下文关联（ICA）两大任务评估 LVLM 的关联推理能力，配套设计了 LLM-as-a-Judge 评分策略和过程奖励评估方法，揭示当前最强 LVLM 仍显著落后于人类。

**[MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)**

:   提出 MME-VideoOCR，一个包含 25 个任务、44 个场景、1464 个视频和 2000 个人工标注 QA 对的视频 OCR 综合评估基准，涵盖文本识别、理解和推理三个层次。评估 18 个 SOTA MLLM 揭示最强模型（Gemini-2.5 Pro）仅达 73.7%，跨帧理解任务低至 25% 以下。

**[MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly](mmlongbench_benchmarking_longcontext_visionlanguage_models_e.md)**

:   构建首个全面的长上下文视觉语言模型（LCVLM）评估基准 MMLongBench——13,331 个样本覆盖 5 类下游任务、混合图像类型、5 级标准化输入长度（8K-128K tokens），评估 46 个模型后发现单任务性能是整体能力的弱代理，且强推理能力与长上下文性能正相关。

**[MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness](mmperspective_do_mllms_understand_perspective_a_comprehensive_benchmark_for_pers.md)**

:   首个系统评估多模态大语言模型 (MLLMs) 透视理解能力的基准，包含10个任务、3个维度、2711张图像和5083个问答对，揭示了43个SOTA模型在透视推理和鲁棒性方面的显著不足。

**[MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection](monitor_exploiting_large_language_models_with_instruction_for_online_video_anoma.md)**

:   提出 MoniTor，一个基于记忆的在线评分队列方案，利用 LLM 进行免训练的在线视频异常检测（VAD），通过双层记忆机制、行为预测和标准评分队列引导 LLM 实现实时异常识别。

**[Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology](multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)**

:   构建了包含 134,533 个星系的图像-光谱-红移多模态数据集（GalaxiesML-Spectra），适配多模态掩码自编码器（MMAE）同时进行图像和光谱的联合重建与红移回归，证明在测试时即使光谱完全缺失，仅用 25% 掩码图像即可实现优于 AstroCLIP 的红移预测散度 $\sigma_{NMAD} = 0.016$。

**[Multimodal Bandits: Regret Lower Bounds and Optimal Algorithms](multimodal_bandits_regret_lower_bounds_and_optimal_algorithms.md)**

:   针对奖励函数至多有 $m$ 个极值的多模态多臂赌博机问题，提出首个计算可行的算法求解 Graves-Lai 优化问题，实现渐近最优的遗憾界，并证明局部搜索策略是次优的。

**[Multimodal Negative Learning](multimodal_negative_learning.md)**

:   提出多模态负学习（MNL）范式，通过让强势模态引导弱势模态抑制非目标类别（而非强制对齐目标类别），稳定决策空间并保留模态特有信息，理论上收紧了多模态融合的鲁棒性下界。

**[Nautilus: A Large Multimodal Model for Underwater Scene Understanding](nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)**

:   构建了首个支持八种水下场景理解任务的大型多模态模型 Nautilus，通过物理先验驱动的视觉特征增强（VFE）模块显式修复水下图像退化，提升 LMM 在水下环境中的鲁棒性。

**[NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints](navil_rethinking_scaling_properties_of_native_multimodal_large_language_models_u.md)**

:   本文系统研究了在数据约束条件下原生多模态大语言模型(Native MLLM)的设计空间与缩放特性，发现视觉编码器与LLM之间存在正相关的最优缩放关系，并基于此提出了NaViL模型，仅用约6亿预训练图文对即可达到顶级MLLM的竞争性性能。

**[NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables](needleinatable_exploring_long-context_capability_of_large_language_models_toward.md)**

:   提出 NeedleInATable (NIAT) 基准，将表格中每个单元格视为"针"，评估 LLM 对长结构化表格的细粒度感知能力，揭示现有模型在复杂下游任务上的高分可能依赖数据捷径而非真正的表格理解。

**[NegoCollab: A Common Representation Negotiation Approach for Heterogeneous Collaborative Perception](negocollab_a_common_representation_negotiation_approach_for_heterogeneous_collab.md)**

:   提出 NegoCollab 框架，通过引入协商者（Negotiator）在训练期间从多模态 agent 的局部表示中协商生成公共表示，有效消除异质协作 agent 之间的域差异，实现低训练成本的协同网联感知。

**[Omni-Mol: Multitask Molecular Model for Any-to-Any Modalities](omni-mol_multitask_molecular_model_for_any-to-any_modalities.md)**

:   提出 Omni-Mol，一个基于多模态 LLM 的统一分子理解与生成框架，通过构建 142 万样本的指令微调数据集、Gradient Adaptive LoRA (GAL) 和 Mixture-of-GAL-Experts (MoGE) 架构，首次在单一模型中统一学习 16 个分子任务（Mol2Mol/Mol2Text/Mol2Num/Text2Mol），以仅 2.2B 参数在 13 个任务上达到 SOTA。

**[OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild](omnigaze_reward-inspired_generalizable_gaze_estimation_in_the_wild.md)**

:   提出OmniGaze，一个半监督3D注视估计框架，利用融合视觉嵌入、MLLM生成的语义注视描述和几何方向向量的奖励模型来评估伪标签质量，在140万无标签人脸数据上训练，在5个数据集的域内/跨域设置下达到SOTA，并在4个未见数据集上展示零样本泛化能力。

**[On the Value of Cross-Modal Misalignment in Multimodal Representation Learning](on_the_value_of_cross-modal_misalignment_in_multimodal_representation_learning.md)**

:   提出潜变量模型将跨模态失配形式化为选择偏差和扰动偏差两种机制，理论证明MMCL学到的表征恰好捕获与两种偏差无关的不变语义子集，统一了"失配有害/有益"两种对立观点。

**[OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Models](openhoi_open-world_hand-object_interaction_synthesis_with_multimodal_large_langu.md)**

:   提出 OpenHOI 框架，利用多模态大语言模型（MLLM）的常识推理能力来推断陌生物体的接触区域和抓取类型，实现开放世界的手物交互合成，无需针对每个物体收集训练数据。

**[PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models](permllm_learnable_channel_permutation_for_nm_sparse_large_language_models.md)**

:   提出 PermLLM，首个可学习通道排列（LCP）框架，通过Sinkhorn归一化将离散排列矩阵松弛为可微分的软排列矩阵实现端到端优化，结合块级排列策略大幅降低计算开销，有效提升N:M稀疏LLM的性能。

**[PhysVLM-AVR: Active Visual Reasoning for Multimodal Large Language Models in Physical Environments](physvlm-avr_active_visual_reasoning_for_multimodal_large_language_models_in_phys.md)**

:   本文提出主动视觉推理（AVR）任务范式，构建了CLEVR-AVR仿真基准和AVR-152k数据集（含丰富CoT标注），训练PhysVLM-AVR模型在部分可观测交互环境中通过感知-推理-动作闭环迭代获取信息并正确回答问题，显著优于现有MLLM。

**[Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning](praxisvlm_visiongrounded_decision_making_via_textdriven_rein.md)**

:   发现VLM的决策推理能力可与视觉感知解耦——用文本描述替代图像时决策准确率不降反升；据此提出Praxis-VLM，在纯文本场景上通过多阶段GRPO与自适应reward训练决策推理能力，推理时零样本迁移到视觉输入，在三大决策benchmark上全面超越SFT基线，尤其在OOD场景泛化优势显著。

**[PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation](prefixkv_adaptive_prefix_kv_cache_is_what_vision_instruction.md)**

:   PrefixKV 发现不同层 KV 缓存的重要性分布差异显著，将逐层缓存大小确定问题形式化为全局前缀配置搜索，通过二分搜索找到最优信息保留阈值使每层保持最大上下文信息，在 20% 压缩率下仅有 0.49 PPL 下降且提供 1.8× 推理加速。

**[Reading Recognition in the Wild](reading_recognition_in_the_wild.md)**

:   提出了阅读识别新任务及首个大规模多模态"野外阅读"数据集（100小时），利用RGB、眼动和IMU三种互补模态的轻量级Transformer模型，在智能眼镜上实现实时阅读检测。

**[Recognition through Reasoning: Reinforcing Image Geo-localization with Large Vision-Language Models](recognition_through_reasoning_reinforcing_image_geo-localization_with_large_visi.md)**

:   本文提出GLOBE——一个基于GRPO强化学习训练的LVLM图像地理定位系统，通过构建推理导向数据集MP16-Reason（含定位可行性评估、视觉线索推理链和地理准确性标注），仅用33K样本就在多个基准上超越基于数百万样本训练的SOTA方法和大规模开源VLM。

**[Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](rethinking_multimodal_learning_from_the_perspective_of_mitig.md)**

:   提出"**分类能力不均衡**"视角理解多模态学习中的模态不平衡，设计 Sustained Boosting 算法（共享编码器 + 多可配置分类器，同时优化分类和残差误差）配合自适应分类器分配（ACA），理论证明跨模态 gap loss 以 $\mathcal{O}(1/T)$ 收敛，在 CREMAD 等 6 个数据集上大幅超越 SOTA。

**[Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieval](retrv-r1_a_reasoning-driven_mllm_framework_for_universal_and_efficient_multimoda.md)**

:   提出首个R1风格的推理型多模态检索框架Retrv-R1，通过信息压缩模块降低token消耗、细节检查机制保留困难候选的完整信息、课程式RL奖励兼顾效果与效率，在通用多模态检索benchmark上实现SOTA。

**[Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)**

:   提出 LogitGap，一种新的 post-hoc OOD 检测评分函数，通过显式利用最大 logit 与其余 logit 之间的"间隔"来区分 ID 和 OOD 样本，并引入 top-N 选择策略过滤噪声 logit，理论和实验证明其在多种场景下超越 MCM 和 MaxLogit。

**[RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics](roborefer_towards_spatial_referring_with_reasoning_in_vision-language_models_for.md)**

:   提出 **RoboRefer**，一个 3D 感知的推理型 VLM，通过 **SFT + RFT** 两阶段训练策略（含度量敏感的过程奖励函数），在空间指代任务中实现精确的单步空间理解和多步空间推理，在 RefSpatial-Bench 上超越 Gemini-2.5-Pro 达 17.4%。

**[RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness](robustmerge_parameter-efficient_model_merging_for_mllms_with_direction_robustnes.md)**

:   本文从低秩分解的角度揭示了参数高效模块合并中"方向鲁棒性"是关键因素（而非全参数合并中的符号冲突），提出RobustMerge通过互补参数自适应缩放和跨任务归一化维持奇异值方向稳定性，在多模态生成任务上平均提升3.4%（已见任务）和4.5%（未见任务）。

**[RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video](rtv-bench_benchmarking_mllm_continuous_perception_understanding_and_reasoning_th.md)**

:   提出 RTV-Bench 基准，包含 552 个视频和 4608 个 QA 对，通过**多时间戳问答**（同一问题在不同时间点答案不同）、**层级问题结构**和**多维评估**三大设计，系统评测 MLLM 在实时视频流中的持续分析能力，揭示了在线模型优于离线模型、单纯增大模型或增加帧数收益有限等关键发现。

**[Scene-Aware Urban Design: A Human-AI Recommendation Framework Using Co-Occurrence Embeddings and Vision-Language Models](scene-aware_urban_design_a_human-ai_recommendation_framework_using_co-occurrence.md)**

:   提出一个人机协同的计算机视觉框架，使用Grounding DINO进行城市物体检测，基于ADE20K数据集构建共现嵌入捕捉真实空间配置，再通过VLM进行场景感知的第三物体推荐，并生成3D模型用于AR预览，旨在让居民参与微观城市设计。

**[SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)**

:   提出 SCOPE，一种联合建模显著性和覆盖率的视觉 Token 剪枝策略，通过迭代选择 SCOPE 得分最高的 Token 来保持语义完整性，在 9 倍 Token 缩减下保留 LLaVA-1.5 96% 的性能。

**[SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models](sd-vlm_spatial_measuring_and_understanding_with_depth-encoded_vision-language_mo.md)**

:   提出MSMU大规模定量空间推理数据集（700K QA对、250万数值标注）和深度位置编码（DPE）方法，使VLM在不引入3D点云的前提下获得强大的定量空间测量和理解能力，在MSMU-Bench上超越GPT-4o达26.91%。

**[Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)**

:   针对多模态大模型在退化文档场景下的OCR幻觉问题，提出首个退化文档幻觉评测基准KIE-HVQA，并设计基于GRPO的多目标奖励强化学习框架，在7B参数模型上实现比GPT-4o高约28%的幻觉抑制准确率提升。

**[See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model](seetrek_training-free_spatial_prompting_for_multimodal_large_language_model.md)**

:   提出 See&Trek，一个无需训练和GPU的空间提示框架，通过最大语义丰富度采样和运动重建来增强 MLLM 的空间理解能力，在 VSI-Bench 上最高提升 3.5%。

**[Sherlock: Self-Correcting Reasoning in Vision-Language Models](sherlock_selfcorrecting_reasoning_in_visionlanguage_models.md)**

:   首个系统研究VLM推理自纠正能力的框架：发现现有推理VLM几乎不能自纠正（<10%出现aha moment），提出Sherlock三阶段训练框架（SFT冷启动→离线轨迹级偏好学习→在线自我迭代）仅用20K标注数据超越使用100K-260K数据的LLaVA-CoT/Mulberry/LlamaV-o1。

**[Situat3DChange: Situated 3D Change Understanding Dataset for Multimodal Large Language Models](situat3dchange_situated_3d_change_understanding_dataset_for_multimodal_large_lan.md)**

:   构建 Situat3DChange 数据集（174K 数据实例），统一了动态场景变化与情境感知理解的感知-行动范式，并提出 SCReasoner——一种高效的 3D MLLM 用于点云对比推理。

**[Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models](sparse_autoencoders_learn_monosemantic_features_in_visionlan.md)**

:   本文将稀疏自编码器（SAE）扩展到视觉-语言模型（如CLIP）上，提出了 MonoSemanticity score（MS）来定量评估神经元的单义性，并展示了通过操控 SAE 神经元可以直接引导多模态大模型（如 LLaVA）的输出，实现概念的插入与抑制。

**[SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)**

:   提出 SpatialThinker，通过在线 RL 结合多目标密集空间奖励（格式→计数→准确性→空间定位的字典序门控）训练 MLLM 构建场景图并进行结构化空间推理，仅用 7K 样本超越 GPT-4o 在 3DSRBench 上 12.1%。

**[SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation](spatialtracegen_high-fidelity_traces_for_efficient_vlm_spatial_reasoning_distill.md)**

:   提出 SpatialTraceGen 框架，通过自动化验证器从大型教师模型蒸馏高质量多步工具使用推理轨迹，用于高效微调小型 VLM 的空间推理能力。

**[SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)**

:   提出 SRPO（Self-Reflection enhanced reasoning with Group Relative Policy Optimization），一个两阶段反思感知 RL 框架：第一阶段用大模型生成反思数据做 SFT cold-start，第二阶段设计反思感知奖励函数在 GRPO 中强化简洁有效的自我反思能力，在 MathVista/MathVision/MMMU-Pro 等多模态推理基准上以 7B/32B 模型显著超越同规模 SOTA。

**[SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning](ssr_enhancing_depth_perception_in_vision-language_models_via_rationale-guided_sp.md)**

:   提出 SSR 框架，将原始深度信息转化为结构化文本推理 rationale，并通过知识蒸馏压缩为紧凑潜在嵌入，以即插即用方式增强现有 VLM 的空间推理能力。

**[Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs](struct2d_a_perception-guided_framework_for_spatial_reasoning_in_mllms.md)**

:   提出 Struct2D，一种感知引导的提示框架，通过将3D感知输出转化为结构化2D表示（BEV图像+对象标记+元数据），使MLLM无需显式3D输入即可完成复杂空间推理任务，并构建了200K QA对的大规模指令微调数据集 Struct2D-Set。

**[Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning](structure-aware_fusion_with_progressive_injection_for_multimodal_molecular_repre.md)**

:   提出 MuMo 框架，通过结构化融合管线（SFP）将 2D 拓扑与 3D 几何融合为稳定的结构先验，再通过渐进注入（PI）机制非对称地整合到序列流中，在 29 个分子属性预测基准任务中平均提升 2.7%，在 22 个任务上排名第一。

**[Systematic Reward Gap Optimization for Mitigating VLM Hallucinations](systematic_reward_gap_optimization_for_mitigating_vlm_hallucinations.md)**

:   提出 Topic-level Preference Rewriting（TPR），通过 topic 级别的细粒度语义控制系统性优化偏好数据中的 reward gap 配置，结合课程学习策略逐步提高负样本难度，在多个幻觉基准上实现约 93% 的幻觉减少。

**[Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models](test-time_spectrum-aware_latent_steering_for_zero-shot_generalization_in_vision-.md)**

:   提出STS（Spectrum-Aware Test-Time Steering），一种轻量级测试时适应方法：通过文本嵌入的SVD分解提取低维语义子空间，学习少量系数在该子空间内对文本原型进行"转向"以适应域偏移，无需反向传播通过大编码器，推理速度比TPT快8倍、内存占用减少12倍，同时在OOD数据集上大幅超越现有TTA方法。

**[Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models](text_to_robotic_assembly_of_multi_component_objects_using_3d_generative_ai_and_v.md)**

:   提出了一个端到端流水线，将自然语言输入通过3D生成式AI转化为网格模型，再利用VLM的零样本多模态推理自动分解为多组件3D模型（结构件+面板件），最终由机器人臂自动装配成物理对象，并支持用户通过对话反馈调整组件分配。

**[The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)**

:   提出TTA-VLM benchmark，在统一实验条件下评估8种episodic和7种online测试时适应(TTA)方法在15个数据集上的表现，发现三个令人意外的结论：(1) 现有TTA方法相比早期TPT基线提升有限；(2) TTA与训练时微调方法协作效果差；(3) 准确率提升以牺牲校准、OOD检测和鲁棒性为代价。

**[To Think or Not To Think: A Study of Explicit Thinking in Rule-Based Visual Reinforcement Fine-Tuning](think_or_not_think_a_study_of_explicit_thinking_in_rule-based_visual_reinforceme.md)**

:   系统研究了基于规则的强化微调（RFT）中显式思维过程的必要性，发现视觉感知任务中"不思考"的RFT（No-Thinking-RFT）往往优于传统的"先思考再回答"策略，并提出了自适应思维方法让模型根据自身能力和任务复杂度决定是否思考。

**[To See or To Read: User Behavior Reasoning in Multimodal LLMs](to_see_or_to_read_user_behavior_reasoning_in_multimodal_llms.md)**

:   提出BehaviorLens基准框架，系统比较文本、散点图和流程图三种用户行为历史的表示方式对MLLM次购预测的影响，发现图像表示相比等效文本表示最高可提升87.5%的预测准确率，且无需额外计算开销。

**[TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)**

:   提出 TOMCAT，通过在测试时从无标签数据中累积文本和视觉双模态知识来动态更新组合原型，克服标签分布偏移问题，在四个 CZSL 基准上实现 SOTA。

**[Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs](towards_comprehensive_scene_understanding_integrating_first_and_third-person_vie.md)**

:   提出 E3VQA 基准（首个多视角 VQA 基准）和 M3CoT 提示技术（融合三种互补视角的场景图），增强大型视觉语言模型 (LVLM) 的多视角场景理解能力，GPT-4o 提升 4.84%、Gemini 2.0 Flash 提升 5.94%。

**[Towards Evaluating Proactive Risk Awareness of Multimodal Language Models](towards_evaluating_proactive_risk_awareness_of_multimodal_language_models.md)**

:   提出PaSBench基准评估多模态语言模型的主动风险感知能力——要求模型在无用户提问的情况下主动观察环境并发出安全预警。评测36个模型发现最强模型（Gemini-2.5-pro）仅达71%准确率且45%的风险无法稳定检测，核心瓶颈是不稳定的主动推理能力而非知识缺失。

**[Training-free Online Video Step Grounding](training-free_online_video_step_grounding.md)**

:   提出BaGLM，一种无需训练的在线视频步骤定位方法，利用贝叶斯滤波将LLM估计的步骤依赖关系和LMM估计的步骤进度融入零样本LMM预测中，在三个数据集上超越现有需训练的离线方法。

**[TRoVe: Discovering Error-Inducing Static Feature Biases in Temporal Vision-Language Models](trove_discovering_errorinducing_static_feature_biases_in_tem.md)**

:   TRoVe提出自动化方法发现时序VLM中导致系统性预测错误的静态特征偏差，通过结合"错误贡献分数"和"静态偏差分数"的双评分机制，在101个合成模型上以28.6%优势超越基线，并成功应用于7个真实VLM揭示新偏差。

**[Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition](uni-mumer_unified_multi-task_fine-tuning_of_vision-language_model_for_handwritte.md)**

:   提出 Uni-MuMER，通过三种数据驱动任务（Tree-CoT、Error-Driven Learning、Symbol Counting）对开源 VLM 进行统一多任务微调，在 CROHME 和 HME100K 数据集上大幅超越专用轻量模型和零样本商用 VLM。

**[Unified Reinforcement and Imitation Learning for Vision-Language Models](unified_reinforcement_and_imitation_learning_for_vision-language_models.md)**

:   提出 RIL（Unified Reinforcement and Imitation Learning）训练框架，结合 GRPO 强化学习和 GAIL 对抗模仿学习，让小型 VLM（7B）通过学习大型 VLM（72B）的文本生成风格来大幅提升性能，无需增加推理延迟或"思考"过程。

**[Unifying Vision-Language Latents for Zero-Label Image Caption Enhancement](unifying_vision-language_latents_for_zero-label_image_caption_enhancement.md)**

:   本文提出ViZer框架，通过统一视觉-语言潜空间对齐的训练范式，在无任何文本标注的情况下提升VLM的图像描述能力——仅使用原始图像数据就能让模型生成更接地、更描述性的caption。

**[UniTok: A Unified Tokenizer for Visual Generation and Understanding](unitok_a_unified_tokenizer_for_visual_generation_and_understanding.md)**

:   提出 UniTok，一种统一视觉生成和理解的tokenizer，通过多码本量化（MCQ）突破离散token表示容量瓶颈，在ImageNet上实现0.38 rFID和78.6%零样本精度的双项记录，并可无缝集成到MLLM中同时启用生成和理解能力。

**[Unveiling Chain of Step Reasoning for Vision-Language Models with Fine-grained Rewards](unveiling_chain_of_step_reasoning_for_visionlanguage_models.md)**

:   提出Chain-of-Step (CoS)推理框架，将VLM的推理链拆解为由Name+Thought+Reflection组成的结构化步骤，训练步骤级Process Reward Model (PRM)提供精细奖励信号，配合迭代DPO和step-level beam search系统性提升VLM推理能力——在InternVL-2.5-MPO-8B上6个benchmark平均73.4%（+4.0%），在LLaVA-NeXT-8B上平均64.2%（+12.1%），并揭示了"VLM推理中质量远比长度重要"这一与LLM领域相反的发现。

**[VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents](vagen_reinforcing_world_model_reasoning_for_multi-turn_vlm_agents.md)**

:   提出VAGEN框架，通过将VLM智能体的推理过程结构化为StateEstimation和TransitionModeling来构建内部世界模型，结合WorldModeling Reward和Bi-Level GAE实现高效的多轮RL训练，使3B模型（0.82）超越GPT-5（0.75）和Gemini 2.5 Pro（0.67）。

**[VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)**

:   提出变分多模态提示学习框架VaMP，将文本侧提示建模为隐变量并通过变分推断进行实例级不确定性建模，结合类感知先验正则化隐空间，在少样本和域泛化设置下显著提升CLIP的下游适配能力。

**[Video-R1: Reinforcing Video Reasoning in MLLMs](video-r1_reinforcing_video_reasoning_in_mllms.md)**

:   受DeepSeek-R1启发，首次系统探索将R1范式（规则RL）应用于视频推理，提出T-GRPO算法显式鼓励模型利用时序信息，并构建图文混合训练数据集，在VSI-Bench上以37.1%准确率超越GPT-4o。

**[Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs](video-safetybench_a_benchmark_for_safety_evaluation_of_video_lvlms.md)**

:   构建首个面向视频 LVLM 安全评估的综合基准 Video-SafetyBench，包含 2264 个视频-文本对覆盖 48 个细粒度不安全类别，通过可控视频生成管线和基于 LLM 置信度的 RJScore 指标，对 24 个 LVLM 进行大规模安全评测，揭示良性查询下视频攻击平均成功率达 67.2%。

**[VIPAMIN: Visual Prompt Initialization via Embedding Selection and Subspace Expansion](vipamin_visual_prompt_initialization_via_embedding_selection_and_subspace_expans.md)**

:   提出VIPAMIN——一种零额外参数的视觉prompt初始化策略，通过注意力引导的语义匹配（Matching）和正交子空间注入（Orthogonalizing）两个模块，解决自监督VPT中prompt注意力均匀化和子空间坍塌两大失效模式，仅需单次前向传播即在24个视觉任务上刷新SOTA。

**[Vision Function Layer in Multimodal LLMs](vision_function_layer_in_multimodal_llms.md)**

:   发现MLLM中视觉相关的功能解码分布在特定的窄层块中（Vision Function Layer），且跨模型家族呈现一致的层级顺序（识别→计数→定位→OCR），据此提出VFL-LoRA（仅用1/3参数匹配full-LoRA性能）和VFL-select（20%数据达98%全量性能）。

**[ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding](vispec_accelerating_vision-language_models_with_vision-aware_speculative_decodin.md)**

:   针对VLM推测解码（speculative decoding）中草稿模型难以处理冗余视觉token的问题，提出ViSpec框架，通过视觉适配器压缩图像token+全局视觉特征注入+合成训练数据，首次在VLM推测解码中实现了显著加速（最高3.22×）。

**[Visual Instruction Bottleneck Tuning](visual_instruction_bottleneck_tuning.md)**

:   首次将信息瓶颈（IB）原理应用于多模态大语言模型的端到端指令微调，提出Visual Instruction Bottleneck Tuning（Vittle），在LLM内部插入轻量瓶颈层学习最小充分表征，在30种分布偏移场景下一致提升鲁棒性，同时不牺牲标准基准性能。

**[Visual Structures Help Visual Reasoning: Addressing the Binding Problem in LVLMs](visual_structures_helps_visual_reasoning_addressing_the_binding_problem_in_vlms.md)**

:   提出 VISER（Visual Input Structure for Enhanced Reasoning），通过在图像上叠加等距水平线+数字标注构建空间分区，配合"逐行扫描"文本指令，将 LVLM 的并行视觉处理转化为串行逐区域解析，在不修改模型、不训练、单次查询的条件下，大幅缓解绑定问题并提升计数、视觉搜索、场景描述、空间关系等视觉推理性能。

**[VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning](vt-fsl_bridging_vision_and_text_with_llms_for_few-shot_learning.md)**

:   提出VT-FSL框架，通过跨模态迭代提示（CIP）联合利用类名和支持图像驱动LLM生成精确文本描述并零样本合成语义一致图像，再通过核化体积对比学习（CGA）实现全局非线性跨模态对齐，在10个少样本学习基准上平均提升4.2%分类准确率。

**[Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM](watch_and_listen_understanding_audio-visual-speech_moments_with_multimodal_llm.md)**

:   提出 TriSense——一个三模态（视觉+音频+语音）大语言模型，通过 Query-Based Connector 自适应调节各模态权重实现鲁棒的视频时序理解，并构建了包含200万标注的 TriSense-2M 数据集支撑训练。

**[WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)**

:   提出 WearVQA，首个专为可穿戴设备（智能眼镜）场景设计的 VQA 基准，包含 2520 个第一人称视角图像-问答三元组，系统覆盖 7 个视觉领域、10 种认知任务类型和 6 类可穿戴特有的图像质量问题，配套 96% 准确率的 LLM-as-a-judge 评估框架，揭示当前 SOTA 多模态模型在此场景下仅达 24-52% 准确率。

**[What Can RL Bring to VLA Generalization? An Empirical Study](what_can_rl_bring_to_vla_generalization_an_empirical_study.md)**

:   本文系统研究了RL微调对VLA（视觉-语言-动作）模型泛化能力的影响，发现PPO是最有效的RL算法且显著优于DPO和GRPO，RL在语义理解和执行鲁棒性方面的OOD泛化远超SFT，同时在视觉鲁棒性上与SFT持平。

**[When One Modality Sabotages the Others: A Diagnostic Lens on Multimodal Reasoning](when_one_modality_sabotages_the_others_a_diagnostic_lens_on_multimodal_reasoning.md)**

:   提出"模态破坏"（modality sabotage）这一诊断性失败模式概念，设计轻量级、模型无关的评估层，将每个模态视为独立代理并通过简单融合暴露"贡献者"与"破坏者"，在多模态情感识别任务上揭示了系统性的模态可靠性差异。

**[When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations](when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_.md)**

:   发现大多模态模型（LMMs）在场景文字识别中存在"语义幻觉"问题（将无语义文本误识为语义合理的词），分析发现注意力集中于文本区域的Transformer层更不易幻觉，据此提出训练无关的ZoomText+Grounded Layer Correction框架，在TextHalu-Bench上提升约4-5%，在ST-VQA上提升约4%。

**[Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting](zero-shot_robustness_of_vision_language_models_via_confidence-aware_weighting.md)**

:   提出 CAW（Confidence-Aware Weighting），一种针对CLIP模型的对抗微调损失函数，通过置信度感知加权重点关注困难对抗样本，结合特征对齐正则化保留预训练语义知识，在AutoAttack下实现零样本鲁棒性SOTA，且内存占用更低。
