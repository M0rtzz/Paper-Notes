---
title: >-
  AAAI2026 模型压缩方向53篇论文解读
description: >-
  53篇AAAI2026的模型压缩方向论文解读，涵盖模型压缩、压缩/编码、LLM、Agent、推理、布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "模型压缩"
  - "论文解读"
  - "论文笔记"
  - "压缩/编码"
  - "LLM"
  - "Agent"
  - "推理"
  - "布局/合成"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📦 模型压缩

**🤖 AAAI2026** · **53** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (21)](../../ICML2026/model_compression/index.md) · [💬 ACL2026 (49)](../../ACL2026/model_compression/index.md) · [📷 CVPR2026 (50)](../../CVPR2026/model_compression/index.md) · [🔬 ICLR2026 (90)](../../ICLR2026/model_compression/index.md) · [🧠 NeurIPS2025 (134)](../../NeurIPS2025/model_compression/index.md) · [📹 ICCV2025 (48)](../../ICCV2025/model_compression/index.md)

🔥 **高频主题：** 模型压缩 ×9 · 压缩/编码 ×7 · LLM ×6 · Agent ×4 · 推理 ×3

**[AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating and Fused Kernel Optimization](adafuse_accelerating_dynamic_adapter_inference_via_token-lev.md)**

:   针对动态MoE-LoRA适配器推理延迟暴增（250%-950%）的问题，提出了一种token级预门控架构，只在第一层做一次全局路由决策，配合自研的SGMM融合CUDA内核将所有激活的LoRA适配器一次性合并进骨干网络，在保持精度的同时将解码延迟降低2.4倍。

**[AgentODRL: A Large Language Model-based Multi-agent System for ODRL Generation](agentodrl_a_large_language_model-based_multi-agent_system_fo.md)**

:   提出AgentODRL，一个基于Orchestrator-Workers架构的LLM多智能体系统，通过任务分解、语法验证循环和LoRA驱动的语义反思机制，将自然语言数据权限规则高质量地转换为ODRL格式。

**[BD-Net: Has Depth-Wise Convolution Ever Been Applied in Binary Neural Networks?](bd-net_has_depth-wise_convolution_ever_been_applied_in_binary_neural_networks.md)**

:   本文提出 BD-Net，通过引入 1.58-bit 卷积和 pre-BN 残差连接，首次成功将深度可分离卷积（depth-wise convolution）应用于二值神经网络（BNN），在 ImageNet 上以 33M OPs 的极低计算量实现了 BNN 领域的新 SOTA，多个数据集上精度提升最高达 9.3 个百分点。

**[Beyond Sharpness: A Flatness Decomposition Framework for Efficient Continual Learning](beyond_sharpness_a_flatness_decomposition_framework_for_efficient_continual_lear.md)**

:   提出 FLAD 框架，将 sharpness-aware 扰动方向分解为梯度对齐分量与随机噪声分量，仅保留噪声分量进行正则化，结合零阶与一阶 sharpness 以极低额外开销提升持续学习的泛化能力。

**[CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis](camera_multi-matrix_joint_compression_for_moe_models_via_mic.md)**

:   提出"micro-expert"概念将MoE层的输出分解为跨矩阵（up/gate/down_proj）的微专家线性组合，基于能量排序进行结构化剪枝(Camera-P)和混合精度量化(Camera-Q)，在Deepseek-MoE-16B/Qwen2-57B/Qwen3-30B上20%-60%剪枝率全面超越NAEE和D²-MoE，且分析Qwen2-57B仅需单卡A100不到5分钟。

**[Can You Tell the Difference? Contrastive Explanations for ABox Entailments](can_you_tell_the_difference_contrastive_explanations_for_abox_entailments.md)**

:   提出对比式ABox解释（Contrastive ABox Explanations）的形式化框架，用于回答"为什么a是C的实例而b不是"的问题，在描述逻辑知识库中同时考虑正向蕴涵和缺失蕴涵，并分析不同描述逻辑和优化准则下的计算复杂度。

**[CLIPPan: Adapting CLIP as A Supervisor for Unsupervised Pansharpening](clippan_adapting_clip_as_a_supervisor_for_unsupervised_pansharpening.md)**

:   提出 CLIPPan，通过轻量微调 CLIP 使其理解多光谱/全色/高分辨率多光谱图像类型及全色锐化过程，然后利用 Wald 协议等文本提示作为语义监督信号，实现无需地面真值的全分辨率无监督全色锐化，可作为即插即用模块兼容任意全色锐化骨干网络。

**[Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers](compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)**

:   提出 Sequential Learning with Drift Compensation (SLDC)，通过学习潜在空间转换算子（线性/弱非线性）来补偿预训练 ViT 在类增量学习中因序列微调导致的分布漂移，结合知识蒸馏后性能接近联合训练上界。

**[Condensed Data Expansion Using Model Inversion for Knowledge Distillation](condensed_data_expansion_using_model_inversion_for_knowledge_distillation.md)**

:   提出用浓缩数据集作为原型指导模型反演（MI）过程，通过特征对齐判别器使生成的合成数据与浓缩样本分布一致，从而扩展浓缩数据集用于知识蒸馏，在 CIFAR/ImageNet 上比标准 MI 蒸馏提升高达 11.4%。

**[Correcting False Alarms from Unseen: Adapting Graph Anomaly Detectors at Test Time](correcting_false_alarms_from_unseen_adapting_graph_anomaly_detectors_at_test_tim.md)**

:   提出 TUNE，一个即插即用的测试时适应框架，通过图对齐器变换节点特征来解决图异常检测中因新正常类别出现导致的"正常性偏移"问题，利用聚合污染程度作为无监督适应信号，在 10 个真实数据集上显著增强多种预训练 GAD 模型的泛化能力。

**[Credal Ensemble Distillation for Uncertainty Quantification](credal_ensemble_distillation_for_uncertainty_quantification.md)**

:   提出Credal Ensemble Distillation（CED）框架，将深度集成教师蒸馏为单模型CREDIT，该模型预测类别概率区间（定义credal集）而非单一softmax分布，在OOD检测任务上实现了优于或可比的不确定性估计，同时大幅降低推理开销（推理时间从5×降为1×）。

**[CTPD: Cross Tokenizer Preference Distillation](ctpd_cross_tokenizer_preference_distillation.md)**

:   提出 Cross-Tokenizer Preference Distillation (CTPD)，首个支持不同分词器间偏好蒸馏的统一框架，通过 Aligned Span Projection、跨分词器重要性加权和 Teacher-Anchored Reference 三项创新，在多个 benchmark 上显著超越现有方法。

**[Distilling Cross-Modal Knowledge via Feature Disentanglement](distilling_cross-modal_knowledge_via_feature_disentanglement.md)**

:   提出频域解耦跨模态知识蒸馏（FD-CMKD），通过傅里叶变换将特征分解为低频（模态共享语义）和高频（模态特有细节）分量，分别施加强一致性 MSE 和弱一致性 logMSE 损失，并引入尺度标准化与共享分类器对齐特征空间，在音频-视觉、图像-文本、语义分割等多个跨模态场景全面超越现有蒸馏方法。

**[Don't Start Over: A Cost-Effective Framework for Migrating Personalized Prompts Between LLMs](dont_start_over_a_cost-effective_framework_for_migrating_personalized_prompts_be.md)**

:   提出PUMA框架，通过轻量级适配器和分组用户选择策略，高效地将个性化软提示从源LLM迁移到不同架构的目标LLM，在三个大规模数据集上匹配甚至超越从头训练的性能，同时减少计算成本高达98%。

**[DOS: Distilling Observable Softmaps of Zipfian Prototypes for Self-Supervised Point Representation](dos_distilling_observable_softmaps_of_zipfian_prototypes_for_self-supervised_poi.md)**

:   提出DOS框架，通过仅在可观测（未掩码）点上蒸馏语义软图（Softmap），结合Zipfian先验的Zipf-Sinkhorn正则化来处理3D语义的长尾分布，在六个3D基准上实现了自监督学习的SOTA，线性探测可达监督性能的95%。

**[DP-GenG: Differentially Private Dataset Distillation Guided by DP-Generated Data](dp-geng_differentially_private_dataset_distillation_guided_by_dp-generated_data.md)**

:   提出 DP-GenG 框架，利用差分隐私生成数据（DP-generated data）引导数据集蒸馏的初始化、特征匹配和专家校准三个阶段，在有限隐私预算下显著提升蒸馏数据集的实用性和隐私保护能力。

**[DynaQuant: Dynamic Mixed-Precision Quantization for Learned Image Compression](dynaquant_dynamic_mixed-precision_quantization_for_learned_i.md)**

:   针对学习图像压缩（LIC）模型部署效率低的痛点，提出DynaQuant框架，在参数层面通过可学习scale/zero-point + Distance-Aware Gradient Modulator实现内容自适应量化，在架构层面通过轻量Bit-Width Selector动态为每层分配最优比特宽度，在Cheng2020/ELIC/Ballé三个基线上实现接近FP32的R-D性能，同时获得最高5.17×加速和模型大小降至原来的~1/4。

**[Earth-Adapter: Bridge Geospatial Domain Gaps with Mixture of Frequency Adaptation](earth-adapter_bridge_the_geospatial_domain_gaps_with_mixture_of_frequency_adapta.md)**

:   提出 Earth-Adapter，首个针对遥感图像**伪影问题**设计的参数高效微调 (PEFT) 方法，通过频率引导的混合适配器 (MoA) 将特征分解为高低频子空间、独立优化后动态聚合，在遥感语义分割 (SS)、域自适应 (DA) 和域泛化 (DG) 三个设定中均超越基线 Rein。

**[EEG-DLite: Dataset Distillation for Efficient Large EEG Model Training](eeg-dlite_dataset_distillation_for_efficient_large_eeg_model_training.md)**

:   提出 EEG-DLite 数据蒸馏框架，通过自监督编码+异常值过滤+多样性采样，将2500小时 EEG 数据集压缩至仅5%即可达到甚至超越全数据集预训练的基础模型性能，GPU预训练时间从30小时降至2小时。

**[Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)**

:   提出 CGRS（Certainty-Guided Reflection Suppression），一种无需训练的高效推理方法，通过在模型高置信度时动态抑制反思触发词（如"Wait""But"），将大型推理语言模型的 token 消耗降低18.5%~41.9%，同时保持推理精度不变。

**[EfficientFSL: Enhancing Few-Shot Classification via Query-Only Tuning in Vision Transformers](efficientfsl_enhancing_few-shot_classification_via_query-only_tuning_in_vision_t.md)**

:   提出 EfficientFSL，一种针对 ViT 少样本分类的 query-only 参数高效微调框架，通过 Forward Block（解耦的主动/冻结子块）、Combine Block（自适应多层特征融合）和 SQ Attention Block（支持-查询分布对齐）三个模块，仅用1.25M~2.48M可训练参数即可在4个域内+6个跨域基准上达到 SOTA。

**[Explore and Establish Synergistic Effects between Weight Pruning and Coreset Selection](explore_and_establish_synergistic_effects_between_weight_pruning_and_coreset_sel.md)**

:   首次系统探索权重剪枝与核心集选择之间的交互关系，提出SWaST机制交替执行两者以建立协同效应，并设计状态保持机制解决"双重损失"问题，在10%–90% FLOPs削减下实现最高17.83%的精度提升。

**[Failures to Surface Harmful Contents in Video Large Language Models](failures_to_surface_harmful_contents_in_video_large_language_models.md)**

:   本文首次系统分析了 VideoLLM 的安全性，揭示了三种结构性设计缺陷（稀疏时间采样、空间 token 下采样、模态融合不平衡），使得视频中清晰可见的有害内容在模型生成的文本摘要中被遗漏（omission rate 超 90%），并设计了三种零查询黑盒攻击来验证漏洞严重性。

**[First-Order Error Matters: Accurate Compensation for Quantized Large Language Models](first-order_error_matters_accurate_compensation_for_quantized_large_language_mod.md)**

:   本文揭示了LLM后训练量化中逐列补偿过程会导致一阶梯度项不可忽略的问题，提出FOEM方法通过将一阶项纳入误差补偿公式来提升量化精度，在3-bit量化下将Llama3-8B的困惑度降低17.3%，且几乎不增加计算开销。

**[HCF: Hierarchical Cascade Framework for Distributed Multi-Stage Image Compression](hcf_hierarchical_cascade_framework_for_distributed_multi-stage_image_compression.md)**

:   本文提出HCF框架，通过直接在潜在空间进行跨节点变换（避免像素域重压缩）并引入策略驱动的量化控制，在分布式多级图像压缩中实现了最高12.64% BD-Rate的PSNR提升，同时节省高达97.8%的FLOPs和96.5%的GPU内存。

**[Hierarchical Pedagogical Oversight: A Multi-Agent Adversarial Framework for Reliable AI Tutoring](hierarchical_pedagogical_oversight_a_multi-agent_adversarial_framework_for_relia.md)**

:   本文提出HPO框架，通过三阶段流水线（情报蒸馏→对抗辩论→综合判定）实现可靠的AI辅导评估，仅用8B参数的模型在MRBench中学数学对话数据集上以Macro F1 0.845超越GPT-4o（0.812）3.3%，证明了交互结构而非模型规模是可靠AI辅导的关键。

**[InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression](infocom_kilobyte-scale_communication-efficient_collaborative_perception_with_inf.md)**

:   提出InfoCom框架，基于扩展的信息瓶颈原理将协同感知的通信量从MB级压缩至KB级（相比Where2comm降低440倍），同时保持近无损的感知性能，核心包含信息感知编码、稀疏掩码生成和多尺度解码三个模块。

**[KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache](kvmix_gradient-based_layer_importance-aware_mixed-precision_.md)**

:   提出 KVmix，通过计算 Key/Value 投影权重梯度的 $L_2$ 范数来评估各层 KV Cache 的重要性，实现层级混合精度量化（Key 平均 2.19bit、Value 平均 2.38bit），并结合动态关键上下文选择（RPC）策略，在 Llama/Mistral 等模型上实现近无损推理、4.9× 内存压缩和 5.3× 吞吐加速。

**[LexChronos: An Agentic Framework for Structured Event Timeline Extraction in Indian Jurisprudence](lexchronos_an_agentic_framework_for_structured_event_timeline_extraction_in_indi.md)**

:   本文提出LexChronos，一个双智能体迭代框架，用于从印度最高法院判决书中提取结构化事件时间线：通过LoRA微调的抽取智能体识别候选事件，预训练的反馈智能体通过置信度驱动的循环进行评分和精炼，在合成数据集上取得BERT F1 0.8751，且结构化时间线在下游的法律文本摘要中被GPT-4在75%的案例中评为优于非结构化基线。

**[Lightweight Optimal-Transport Harmonization on Edge Devices](lightweight_optimal-transport_harmonization_on_edge_devices.md)**

:   提出 MKL-Harmonizer，利用经典最优传输理论中的 Monge-Kantorovich 线性映射（MKL），训练一个轻量级编码器预测 12 维颜色变换参数，实现边缘设备上的实时图像颜色协调，在 AR 场景的感知质量-速度综合指标上达到最优。

**[Parametric Pareto Set Learning for Expensive Multi-Objective Optimization](parametric_pareto_set_learning_for_expensive_multi-objective_optimization.md)**

:   本文提出 PPSL-MOBO 框架，通过超网络 + LoRA 架构学习从偏好和外在参数到 Pareto 最优解的统一映射，结合高斯过程代理模型和超体积改进采集策略，高效解决昂贵的参数化多目标优化问题。

**[PocketLLM: Ultimate Compression of Large Language Models via Meta Networks](pocketllm_ultimate_compression_of_large_language_models_via_meta_networks.md)**

:   PocketLLM提出通过元网络（编码器-码本-解码器）在潜空间中压缩LLM权重向量，用小型解码器+紧凑码本+索引替代原始权重矩阵，在Llama 2-7B上实现10×压缩且精度损失可忽略，突破了传统量化/剪枝在极端压缩比下的精度瓶颈。

**[Post Training Quantization for Efficient Dataset Condensation](post_training_quantization_for_efficient_dataset_condensation.md)**

:   首次将训练后量化（PTQ）应用于数据集蒸馏，提出基于补丁的量化框架（PAQ+分组+精炼），在 2-bit 极低比特下将蒸馏数据集的测试精度几乎翻倍（如 DM IPC=1 从 26.0% 提升至 54.1%），作为即插即用框架可应用于各种蒸馏方法。

**[Prototype-Based Semantic Consistency Alignment for Domain Adaptive Retrieval](prototype-based_semantic_consistency_alignment_for_domain_adaptive_retrieval.md)**

:   提出 PSCA 两阶段框架，通过正交 prototype 建立类级语义连接，结合几何-语义一致性对齐动态修正伪标签可靠性，并在重建特征上进行 hash 编码，在多个跨域检索数据集上大幅超越现有方法。

**[Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge](put_the_space_of_lora_initialization_to_the_extreme_to_preserve_pre-trained_know.md)**

:   提出 LoRA-Null，将 LoRA 初始化在预训练知识 input activation 的 null space 中（而非权重的 null space），从信息论角度论证 activation 的 effective rank 远小于权重，因此其 null space 包含更少预训练知识信息，显著减轻微调时的灾难性遗忘。

**[QuEPT: Quantized Elastic Precision Transformers with One-Shot Calibration for Multi-Bit Switching](quept_quantized_elastic_precision_transformers_with_one-shot_calibration_for_mul.md)**

:   提出QuEPT弹性精度量化框架，通过Multi-Bit Token Merging和Multi-Bit Cascaded LoRA两大核心模块，实现一次校准即可在ViT/LLM/MLLM上实时切换任意预定义位宽，性能媲美甚至超越单位宽SOTA PTQ方法。

**[Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness](reinforced_rate_control_for_neural_video_compression_via_inter-frame_rate-distor.md)**

:   提出首个基于约束马尔可夫决策过程（CMDP）的强化学习速率控制框架，通过时空状态建模联合捕获帧内容特征与帧间率-失真耦合依赖，直接映射到逐帧编码参数，在多种神经视频编解码器上将平均比特率误差降至1.20%，BD-Rate节省最高达13.98%。

**[Renormalization Group Guided Tensor Network Structure Search](renormalization_group_guided_tensor_network_structure_search.md)**

:   提出 RGTN 框架，将统计物理中的重正化群（Renormalization Group）理论引入张量网络结构搜索，通过多尺度粗粒化-扩展-压缩流程和可学习边门控实现连续拓扑演化，在光场压缩、高阶张量分解和视频补全任务上达到 SOTA 压缩率，同时比已有方法快 4–600 倍。

**[Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling](rethinking_long-tailed_dataset_distillation_a_uni-level_framework_with_unbiased_.md)**

:   提出首个面向长尾分布的单层(uni-level)数据集蒸馏框架，通过专家模型去偏、BN统计量公平校准和置信度引导初始化三大策略，在CIFAR-100-LT上提升15.6%、Tiny-ImageNet-LT上提升11.8%，全面超越DAMED。

**[SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication](safesieve_from_heuristics_to_experience_in_progressive_pruning_for_llm-based_mul.md)**

:   提出SafeSieve，一种渐进式自适应多智能体通信剪枝框架，通过语义启发初始化→历史反馈驱动的双阶段边评分和0-extension聚类机制，在6个基准上实现94.01%平均准确率同时减少12.4%-27.8% token消耗，并展现出对prompt注入攻击的天然鲁棒性。

**[Satisficing and Optimal Generalised Planning via Goal Regression (Extended Version)](satisficing_and_optimal_generalised_planning_via_goal_regression_extended_versio.md)**

:   提出 Moose 规划器，利用目标回归（goal regression）从训练问题中合成泛化规划程序：将训练问题的目标拆解为单目标子问题逐个最优求解，通过回归和提升（lifting）得到一阶条件-动作规则集，用于满足性规划（直接执行规则）或最优规划（编码为公理剪枝搜索空间）。

**[Share Your Attention: Transformer Weight Sharing via Matrix-Based Dictionary Learning](share_your_attention_transformer_weight_sharing_via_matrix-based_dictionary_lear.md)**

:   受字典学习启发，提出 MASA 框架，将 Transformer 各层注意力投影矩阵（Q/K/V/O）分解为共享矩阵原子的线性组合，以 66.7% 的注意力参数压缩率实现与原始 Transformer 持平甚至更优的性能。

**[Sharp Eyes and Memory for VideoLLMs: Information-Aware Visual Token Pruning for Efficient and Reliable VideoLLM Reasoning](sharp_eyes_and_memory_for_videollms_information-aware_visual_token_pruning_for_e.md)**

:   SharpV 提出一个两阶段无训练视觉Token剪枝框架，在Pre-LLM阶段基于时空信息自适应调整每帧剪枝比例，在Intra-LLM阶段基于视觉信息退化假说进行KV Cache剪枝，首次实现与Flash Attention完全兼容，在多个视频理解基准上以约12%的Token保留率达到与稠密模型相当甚至更优的性能。

**[SIGN: Schema-Induced Games for Naming](sign_schema-induced_games_for_naming.md)**

:   SIGN 提出在LLM多智能体命名博弈中引入轻量级消息Schema（如 `@say {name: Ck}`），发现结构化先验可将群体约定一致性提升5.8×，收敛所需Token减少一个数量级，为高效多智能体协调提供了简单可控的"调节旋钮"。

**[SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping](skipcat_rank-maximized_low-rank_compression_of_large_language_models_via_shared_.md)**

:   SkipCat 提出了一种秩最大化的低秩压缩框架，通过层内共享投影（Cat）和块跳跃（Skip）两项技术，在相同压缩率下保留更多有效秩，无需微调即可在零样本任务上比现有低秩方法提升7%准确率。

**[SparseRM: A Lightweight Preference Modeling with Sparse Autoencoder](sparserm_a_lightweight_preference_modeling_with_sparse_autoencoder.md)**

:   SparseRM 利用稀疏自编码器（SAE）从LLM中间表示中提取偏好相关方向，通过投影向量构建轻量级奖励模型，仅需不到1%的可训练参数即可超越大多数主流奖励模型，并在在线迭代对齐框架中表现出更强的泛化能力。

**[SpecQuant: Spectral Decomposition and Adaptive Truncation for Ultra-Low-Bit LLMs Quantization](specquant_spectral_decomposition_and_adaptive_truncation_for_ultra-low-bit_llms_.md)**

:   SpecQuant 提出一种基于自适应傅里叶域分解的两阶段量化框架：先将激活离群值平滑迁移到权重，再通过通道级低频傅里叶截断吸收权重中的高频噪声，在LLaMA-3 8B上实现W4A4量化仅1.5%精度损失，同时获得2×加速和3×内存节省。

**[Steering Pretrained Drafters during Speculative Decoding](steering_pretrained_drafters_during_speculative_decoding.md)**

:   提出 SD²，通过从验证器隐藏状态中提取转向向量（steering vector）并注入预训练 drafter 的 MLP 层，实现推测解码中 drafter-verifier 的动态对齐，在标准采样下将被接受 token 数提升高达 35%，同时计算开销可忽略不计。

**[StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion](stepfun-formalizer_unlocking_the_autoformalization_potential_of_llms_through_kno.md)**

:   提出 ThinkingF 流水线，通过大规模知识蒸馏与模板引导的推理轨迹合成分别增强 LLM 的形式语言领域知识和非形式到形式的推理能力，再经两阶段 SFT + RLVR 融合两种能力，7B/32B 模型在 FormalMATH-Lite 和 ProverBench 上达到 SOTA。

**[Stratified Knowledge-Density Super-Network for Scalable Vision Transformers](stratified_knowledge-density_super-network_for_scalable_vision_transformers.md)**

:   提出将预训练 ViT 转化为"分层知识密度超网络"（SKD Super-Network），通过 WPAC（加权 PCA 注意力收缩）和 PIAD（渐进式重要性感知 Dropout）两步实现知识的分层组织，使得任意大小的子网络均可以 O(1) 代价提取，且无需额外微调即可达到或超越 SOTA 压缩方法的性能。

**[TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)**

:   提出 TGDD，将静态分布匹配重新定义为沿训练轨迹的动态对齐过程，通过阶段式分布匹配（Stage-wise Distribution Matching）捕获演化语义 + 分布约束正则化（Stage-wise Distribution Constraint）减少类间重叠，在 10 个数据集上达到 SOTA，高分辨率基准上准确率提升 5.0%。

**[Towards Test-time Efficient Visual Place Recognition via Asymmetric Query Processing](towards_test-time_efficient_visual_place_recognition_via_asymmetric_query_proces.md)**

:   提出面向视觉位置识别（VPR）的高效非对称框架 AsymVPR，通过**地理记忆库**替代昂贵的 k-NN 预计算，以及**隐式嵌入增强**弥合轻量查询网络与高容量图库网络的能力差距，实现仅用 ~8% FLOPs 的轻量网络达到接近全尺寸模型的检索性能。

**[Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated](your_ai-generated_image_detector_can_secretly_achieve_sota_accuracy_if_calibrate.md)**

:   提出一种基于贝叶斯决策理论的轻量级后验校准方法，通过在模型输出logit上添加可学习标量偏移α，无需重训练即可显著提升现有AI生成图像检测器在分布偏移下的准确率。
