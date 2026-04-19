---
title: >-
  CVPR2026 多模态VLM方向 203篇论文解读
description: >-
  203篇CVPR2026 多模态VLM方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧩 多模态VLM

**📷 CVPR2026** · **203** 篇论文解读

**[A3: Towards Advertising Aesthetic Assessment](a3_towards_advertising_aesthetic_assessment.md)**

:   提出A3框架，包含理论驱动的三阶段广告美学评估范式A3-Law（感知注意力→形式兴趣→欲望影响）、12万条标注数据集A3-Dataset、经SFT+GRPO对齐的模型A3-Align以及评测基准A3-Bench，在广告美学自动评估上超越现有MLLM。

**[A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](a_closed-form_solution_for_debiasing_vision-language_models_with_utility_guarant.md)**

:   提出一种在VLM跨模态空间中具有闭式解的去偏方法，在无需训练、无需标注数据的条件下，通过正交分解实现Pareto最优的公平性与效用权衡，同时为效用损失提供理论上界。

**[A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](a_closedform_solution_for_debiasing_visionlanguage.md)**

:   提出VLM去偏的闭式解方法，通过在跨模态嵌入空间中对属性子空间做正交分解并利用Chebyshev标量化求解，实现Pareto最优公平性与有界效用损失，免训练、免标注，统一覆盖零样本分类、文本-图像检索和文本-图像生成三大下游任务。

**[Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models](activation_matters_test-time_activated_negative_labels_for_ood_detection_with_vi.md)**

:   提出 TANL（Test-time Activated Negative Labels），通过在测试时动态评估负标签在OOD样本上的"激活程度"来挖掘最有效的负标签，配合激活感知评分函数，在 ImageNet 基准上将 FPR95 从 17.5% 大幅降至 9.8%，且完全免训练、测试高效。

**[AVR: Adaptive VLM Routing for Computer Use Agents](adaptive_vision-language_model_routing_for_computer_use_agents.md)**

:   提出 AVR 自适应路由框架，通过轻量多模态嵌入模型评估动作难度 + 小模型 logprob 置信度探测 + warm agent 记忆注入，实现三层路由（简单→小模型，困难→大模型，高风险→大模型+guardrail），在推理成本降低 78% 的同时仅损失 2pp 准确率。

**[Adaptive Vision-Language Model Routing for Computer Use Agents](adaptive_visionlanguage_model_routing_for_computer.md)**

:   提出 Adaptive VLM Routing (AVR) 框架，在 CUA 编排器和 VLM 模型池之间插入轻量语义路由层，通过多模态难度分类、logprob 置信度探测和历史记忆注入三种机制动态选择最经济的模型，推理成本降低最高 78% 且精度仅下降 2 个百分点以内。

**[AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition](adaptvision_efficient_vision-language_models_via_adaptive_visual_acquisition.md)**

:   提出 AdaptVision，通过由粗到精的主动视觉机制和强化学习训练，让 VLM 自主决定每个样本所需的最少视觉 token 数量，配合解耦式多轮策略优化 (DTPO) 实现效率与精度的最优平衡。

**[AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)**

:   AGFT 提出了一种对齐引导的微调框架，通过文本引导的对抗训练和分布一致性校准，在增强 VLM 零样本对抗鲁棒性的同时保持预训练的跨模态语义结构，在 15 个零样本基准上平均鲁棒准确率达到 46.57%，超越 SOTA 3.1 个百分点。

**[AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)**

:   AnomalyVFM 提出了一个通用框架，通过三阶段合成数据生成方案和参数高效的 LoRA 适配机制，将任意视觉基础模型（VFM）转化为强零样本异常检测器，以 RADIO 为骨干在 9 个工业数据集上达到 94.1% 图像级 AUROC，超越 SOTA 3.3 个百分点。

**[ApET: Approximation-Error Guided Token Compression for Efficient VLMs](apet_approximation-error_guided_token_compression_for_efficient_vlms.md)**

:   从信息论角度提出基于线性近似重建误差的视觉 token 重要性评估方法，不依赖 attention 权重，天然兼容 FlashAttention，在 LLaVA-1.5 上压缩 88.9% 视觉 token 仍保持 95.2% 性能。

**[ApET: Approximation-Error Guided Token Compression for Efficient VLMs](apet_approximation_error_token_compression.md)**

:   从信息论角度出发，通过线性近似重建每个visual token并用重建误差衡量其信息量（误差大=信息多=应保留），提出完全不依赖注意力权重的ApET框架，在LLaVA-1.5-7B上88.9%压缩保留95.2%精度，视频任务甚至达100.4%超基线，且完全兼容FlashAttention。

**[BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates](balm_a_model-agnostic_framework_for_balanced_multimodal_learning_under_imbalance.md)**

:   BALM 提出一个模型无关的即插即用框架来解决**不均衡缺失率（IMR）**下的多模态学习问题，通过特征校准模块（FCM）对齐不同缺失模式下的表征、以及梯度再平衡模块（GRM）从分布和空间两个维度平衡各模态的优化动态，在多个多模态情感识别基准上持续提升各类骨干网络的鲁棒性。

**[Benchmarking Vision-Language Models under Contradictory Virtual Content Attacks in Augmented Reality](benchmarking_vision-language_models_under_contradictory_virtual_content_attacks_.md)**

:   构建首个 AR 环境下矛盾虚拟内容攻击基准 ContrAR（312 个真实 Meta Quest 3 录制视频，10 名标注者验证，平均 Likert 4.66/5），系统评估 11 个 VLM（含 GPT-5/Gemini-2.5/Grok-4）的语义矛盾检测能力，发现 GPT-5 准确率最高（88.14%）但延迟 19s，GPT-4o 在准确率-延迟平衡最佳（84.62%/7.26s），OCR 纯文本基线仅 56%，证明视觉推理不可或缺。

**[Beyond Heuristic Prompting: A Concept-Guided Bayesian Framework for Zero-Shot Image Recognition](beyond_heuristic_prompting_a_concept-guided_bayesian_framework_for_zero-shot_ima.md)**

:   将 VLM 零样本图像识别重构为贝叶斯框架，通过 LLM 驱动的多阶段概念合成流水线构建概念提案分布，并用自适应 soft-trim 似然函数抑制离群概念影响，在 11 个分类基准上优于 SOTA 方法。

**[Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models](beyond_recognition_evaluating_visual_perspective_taking_in_vision_language_model.md)**

:   通过心理学启发的受控LEGO场景构建Isle-Brick-V2基准，系统揭示当前VLM在视觉透视能力(VPT)上的显著不足——即使场景理解近乎完美，空间推理和透视能力仍大幅退化，且存在顽固的方向偏置。

**[Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models](beyond_static_artifacts_a_forensic_benchmark_for_video_deepfake_reasoning_in_vis.md)**

:   提出 FAQ（Forensic Answer-Questioning），首个关注深度伪造视频时序不一致性的大规模多选问答基准（33K QA 对、~4500 视频），通过三层级任务体系（面部感知→时序定位→取证推理）渐进式增强 VLM 取证能力，微调后在域内基准和跨数据集检测中均取得显著提升（Qwen2.5-VL 平均准确率从 21.6% 提升至 52.4%）。

**[Beyond the Mean: Modelling Annotation Distributions in Continuous Affect Prediction](beyond_the_mean_modelling_annotation_distributions_in_continuous_affect_predicti.md)**

:   提出基于Beta分布的情感标注共识建模框架，模型仅预测标注分布的均值和标准差，即可通过矩匹配闭式推导出偏度、峰度、分位数等高阶描述子，在SEWA和RECOLA上证明Beta分布能有效捕获标注者分歧的完整分布特性。

**[BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment](brima_bridged_modality_adaptation_for_multi-modal_continual_action_quality_asses.md)**

:   提出 BriMA，通过记忆引导的桥接补全和模态感知回放机制，解决多模态持续动作质量评估中非平稳模态不平衡问题，在三个基准上平均提升 6-8% 相关系数、降低 12-15% 误差。

**[BUSSARD: Normalizing Flows for Bijective Universal Scene-Specific Anomalous Relationship Detection](bussard_normalizing_flows_for_bijective_universal_scene-specific_anomalous_relat.md)**

:   提出 BUSSARD，首个基于学习的场景特定异常关系检测方法，利用预训练语言模型嵌入场景图三元组 + 自编码器降维 + 标准化流进行似然估计，在 SARD 数据集上 AUROC 提升约 10%，且对同义词变化鲁棒。

**[Can Vision-Language Models Count? A Synthetic Benchmark and Analysis of Attention-Based Interventions](can_vision-language_models_count_a_synthetic_benchmark_and_analysis_of_attention.md)**

:   构建了一个合成计数基准数据集，系统评估了开源 VLM 在不同图像/提示条件下的计数能力，并通过解码器层面的视觉注意力重加权实验探索改善计数行为的机制。

**[CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment](capt_confusion-aware_prompt_tuning_for_reducing_vision-language_misalignment.md)**

:   提出 CAPT 混淆感知 prompt tuning 框架，通过语义混淆挖掘器（SEM）和样本混淆挖掘器（SAM）显式建模 VLM 的系统性误对齐模式，配合多粒度差异专家（MGDE）融合不同层次的混淆信息，在 11 个基准上取得 HM 83.90% 的最优表现。

**[ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding](chartnet_a_million-scale_high-quality_multimodal_dataset_for_robust_chart_unders.md)**

:   发布 ChartNet——150 万规模的高质量多模态图表数据集，通过代码引导合成管线生成包含图像-代码-数据表-文本-推理QA 的对齐五元组，在图表理解和推理任务上显著提升 VLM 性能，小模型微调后超越 GPT-4o。

**[Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking](circuit_tracing_in_vision-language_models_understanding_the_internal_mechanisms_.md)**

:   提出首个面向 VLM 的电路追踪框架，在 Gemma-3-4B 中训练 per-layer transcoder 并构建归因图，揭示了多模态推理的层次化整合机制、视觉数学电路和六指幻觉的内部成因，并通过 steering 和 circuit patching 验证电路的因果可控性。

**[CLIP-Free, Label-Free, Unsupervised Concept Bottleneck Models](clip-free_label_free_unsupervised_concept_bottleneck_models.md)**

:   提出 TextUnlock 方法将任意冻结视觉分类器的输出分布对齐到视觉-语言对应空间，进而构建无需CLIP、无需标签、无需训练线性探针的全无监督概念瓶颈模型 (U-F²-CBM)，在40+模型上超越有监督CLIP-based CBM。

**[CodeDance: A Dynamic Tool-integrated MLLM for Executable Visual Reasoning](codedance_a_dynamic_tool-integrated_mllm_for_executable_visual_reasoning.md)**

:   提出 CodeDance，将可执行代码作为视觉推理的统一媒介，通过 SFT 教授原子能力 + RL 中的难度自适应工具调用奖励（BAT），实现动态工具编排与自检推理，7B 模型在计数/视觉搜索/图表 QA 等任务上超越 GPT-4o。

**[Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling](conditional_factuality_controlled_llms_with_generalization_certificates_via_conf.md)**

:   提出 CFC（Conditional Factuality Control），一种后处理保形框架，通过增强分位数回归学习特征条件的接受阈值函数，为 LLM 采样输出提供条件覆盖保证（而非仅边际保证），并推导 PAC 风格的有限样本证书 CFC-PAC，在合成数据、推理/QA 基准和 VLM 设置上验证有效性。

**[Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)**

:   提出 SeGP-CL，通过对抗锚点探测旧-新语义边界的脆弱区域，结合锚点引导的跨模态几何蒸馏（ACGD）和文本语义几何正则化（TSGR），在无样本回放条件下有效保持 VLM 的跨模态语义几何结构，显著缓解灾难性遗忘。

**[Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_visionlanguage_models_via.md)**

:   提出 SeGP-CL，通过对抗性 PGD 在旧新语义边界构造锚点样本，配合锚点引导的跨模态几何蒸馏（ACGD）和文本语义几何正则化（TSGR），在无需旧数据回放条件下保护 VLM 持续学习中的跨模态语义几何结构，五个基准上达到 SOTA。

**[CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)**

:   发现 MLLM 中视觉编码器微调的"视觉偏好冲突"问题，提出 CoVFT 框架，通过上下文向量提取（CVE）和上下文混合专家（CoMoE）实现上下文感知的视觉微调，在 12 个多模态基准上达到 SOTA 且稳定性显著优于现有方法。

**[CoVR-R: Reason-Aware Composed Video Retrieval](covr-rreason-aware_composed_video_retrieval.md)**

:   CoVR-R 提出了推理优先的零样本组合视频检索框架，利用大型多模态模型（Qwen3-VL）显式推理编辑操作隐含的"后效应"（状态转换、时间阶段、镜头变化等），并构建了包含结构化推理轨迹和困难干扰项的 CoVR-R 基准来评估推理能力，在检索准确率上大幅超越现有方法。

**[CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning](crit_graph-based_automatic_data_synthesis_to_enhance_cross-modal_multi-hop_reaso.md)**

:   提出基于图结构的自动数据生成 pipeline，构建了 CRIT 数据集与 benchmark，用于训练和评测 VLM 在交错图文内容上的跨模态多跳推理能力，训练后的模型在 SPIQA 等多个基准上取得显著提升。

**[CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods](crosshoi-bench_a_unified_benchmark_for_hoi_evaluation_across_vision-language_mod.md)**

:   提出 CrossHOI-Bench，首个统一评估 VLM 和 HOI 专用模型的多选题 HOI 基准，通过精心策划的正负例避免不完整标注的错误惩罚，揭示大型 VLM 零样本在 Instance-F1 上超越 SOTA HOI 方法 +5.18%，但在多动作识别和跨人归因上仍存在系统性弱点。

**[Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens](cubic_discrete_diffusion_discrete_visual_generation_on_high-dimensional_represen.md)**

:   提出 CubiD，首个在高维表征 token（768维）上做离散扩散生成的模型，通过在 $h \times w \times d$ 三维张量上进行细粒度 mask 预测实现高质量图像生成，同时保留理解能力。

**[Customized Visual Storytelling with Unified Multimodal LLMs](customized_visual_storytelling_with_unified_multimodal_llms.md)**

:   提出 VstoryGen 框架和核心组件 CustFilmer，基于统一多模态大语言模型（UMLLM）实现多模态故事定制生成，支持文本描述、角色/场景参考图像和镜头类型的联合条件控制，并构建了 MSB 和 M2SB 两个新 benchmark。

**[DC-Merge: Improving Model Merging with Directional Consistency](dc-merge_improving_model_merging_with_directional_consistency.md)**

:   DC-Merge 发现模型合并的关键在于保持合并后多任务向量与原始单任务向量之间**奇异空间方向的一致性**，通过奇异值平滑 + 共享正交子空间投影两步操作，在 Vision 和 Vision-Language 任务上均取得 SOTA 合并效果。

**[DeAR: Fine-Grained VLM Adaptation by Decomposing Attention Head Roles](dear_fine-grained_vlm_adaptation_by_decomposing_attention_head_roles.md)**

:   提出 DeAR，通过 Concept Entropy 指标将 ViT 深层注意力头分解为属性头/泛化头/混合头三类功能角色，并设计基于角色的注意力掩码机制精确控制信息流，在15个数据集上实现任务适配与零样本泛化的最佳平衡。

**[Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)**

:   提出 DASP，通过冗余度评分诊断偏置模态，再用非对称适应策略解耦稳定性与可塑性，解决多模态测试时适应中的负迁移和灾难性遗忘问题。

**[Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification](demographic_fairness_in_multimodal_llms_a_benchmark_of_gender_and_ethnicity_bias.md)**

:   首次系统性地评估了 9 个开源 MLLM 在人脸验证任务上的人口统计公平性，在 IJB-C 和 RFW 两个 benchmark 上使用 4 种 FMR-based 公平性指标衡量性别和族裔偏差，发现 MLLM 的偏见模式与传统人脸识别系统不同。

**[Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models](devil_is_in_narrow_policy_unleashing_exploration_in_driving_vla_models.md)**

:   揭示驾驶 VLA 模型中被忽视的"窄策略"（Narrow Policy）瓶颈——IL 阶段过度利用导致探索坍缩，进而限制 RL 阶段。提出 Curious-VLA 框架，通过可行轨迹扩展 + 多样性感知 RL 在 Navsim 上达到 SOTA（PDMS 90.3，Best-of-N 94.8）。

**[Diagnosing and Repairing Unsafe Channels in Vision-Language Models via Causal Discovery and Dual-Modal Safety Subspace Projection](diagnosing_and_repairing_unsafe_channels_in_vision-language_models_via_causal_di.md)**

:   提出 CARE 框架，先用因果中介分析精确定位 VLM 中与不安全行为因果相关的神经元和层（诊断），再通过广义特征分解构建双模态安全子空间并在推理时投影激活值（修复），将攻击成功率降至 10% 以下且几乎不损失通用能力。

**[Disentangle-then-Align: Non-Iterative Hybrid Multimodal Image Registration via Cross-Scale Feature Disentanglement](disentangle-then-align_non-iterative_hybrid_multimodal_image_registration_via_cr.md)**

:   提出 HRNet，通过跨尺度特征解纠缠和自适应投影（CDAP）学习干净的共享表示，并在统一的粗到细管线中非迭代地联合预测刚性和非刚性变换，在四个多模态数据集上达到SOTA。

**[Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks](do_vision-language_models_leak_what_they_learn_adaptive_token-weighted_model_inv.md)**

:   首次系统研究 VLM 的模型逆向（Model Inversion）攻击，提出基于自适应 token 注意力权重的序列级逆向方法 SMI-AW，通过动态加权视觉关联度不同的 token 梯度，从 VLM 中重建隐私训练图像，人类评估攻击准确率达 61.21%。

**[Downscaling Intelligence: Exploring Perception and Reasoning Bottlenecks in Small VLMs](downscaling_intelligence_exploring_perception_and_reasoning_bottlenecks_in_small.md)**

:   系统研究LLM缩放对多模态能力的影响，发现视觉任务而非LLM依赖任务受影响最大，且感知退化与推理退化同等严重；提出Extract+Think方法（视觉提取调优+逐步推理），以0.6B感知+1.7B推理的极小模型超越了12倍大的PrismCaptioner和LLaVA-OneVision-0.5B。

**[DSCA: Dynamic Subspace Concept Alignment for Lifelong VLM Editing](dsca_dynamic_subspace_concept_alignment_for_lifelong_vlm_editing.md)**

:   DSCA通过将VLM的表征空间分解为一组正交语义子空间，在每个子空间内进行门控残差干预来实现知识编辑，从而在1000次连续编辑后仍保持>95%的编辑成功率且近乎零遗忘。

**[DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference](duet-vlm_dual_stage_unified_efficient_token_reduction_for_vlm_training_and_infer.md)**

:   提出 DUET-VLM 双阶段视觉 token 压缩框架：第一阶段在视觉编码器内通过 V2V self-attention 选取 dominant tokens 并将剩余 tokens 通过注意力引导局部聚类合并为 contextual tokens；第二阶段在 LLM 内通过 T2V cross-attention 层级裁剪视觉 tokens。在 LLaVA-1.5-7B 上实现 67% token 压缩保持 99%+ 精度、89% 压缩保持 97%+ 精度，训练时间减少 31%。

**[DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference](duet_vlm_dual_stage_token_reduction.md)**

:   提出DUET-VLM双阶段视觉token压缩框架：先在视觉编码器侧通过局部聚类聚合将冗余token合并为信息保持的紧凑表示（V2V），再在语言骨干侧通过文本引导的层级自适应剪枝逐步删减低信息量token（T2V），在LLaVA-1.5-7B上67%压缩保留99%精度，89%压缩保留97%精度。

**[Dynamic Token Reweighting for Robust Vision-Language Models](dynamic_token_reweighting_for_robust_vision-language_models.md)**

:   提出Dtr（Dynamic Token Reweighting），首个通过优化VLM的KV缓存来防御多模态越狱攻击的推理时防御方法，通过定义"反向安全偏移"（RSS）来识别导致安全退化的视觉token，动态调整其权重以恢复模型的安全对齐能力，同时保持良性任务性能。

**[DTR: Dynamic Token Reweighting for Robust Vision-Language Models](dynamic_token_reweighting_for_robust_visionlanguag.md)**

:   提出DTR——首个通过KV cache优化防御多模态越狱攻击的方法：利用反转安全偏移（Reversal Safety-Relevant Shift）识别对抗性视觉token，通过动态重加权衰减其影响，仅4步优化即可在不依赖图生文转换的前提下，大幅降低攻击成功率（HADES S+T+A: 56.9%→15.9%）同时保持VLM性能和推理效率。

**[DynamicGTR: Leveraging Graph Topology Representation Preferences to Boost VLM Capabilities on Graph QAs](dynamicgtr_leveraging_graph_topology_representation_preferences_to_boost_vlm_cap.md)**

:   提出 DynamicGTR 框架，通过动态路由在推理时为每个查询选择最优的图拓扑表示（GTR，视觉/文本共8种），显著提升 VLM 在零样本图算法问答中的性能，并可迁移到链接预测和节点分类等真实场景。

**[EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval](eaglenet_energy-aware_fine-grained_relationship_learning_network_for_text-video_.md)**

:   EagleNet 通过构建文本-帧关系图并使用关系图注意力网络学习文本-帧和帧-帧之间的细粒度关系，生成融合视频上下文信息的增强文本嵌入，并引入基于能量模型的匹配机制捕获真实文本-视频对分布，在四个基准数据集上取得 SOTA。

**[Efficient Document Parsing via Parallel Token Prediction](efficient_document_parsing_via_parallel_token_prediction.md)**

:   提出 PTP（Parallel Token Prediction），一种模型无关的即插即用加速方法，通过在训练序列中插入可学习 register token 实现并行多 token 预测，在 OmniDocBench 上实现 1.6×-2.2× 吞吐提升且不损失精度。

**[EgoMind: Activating Spatial Cognition through Linguistic Reasoning in MLLMs](egomind_activating_spatial_cognition_through_linguistic_reasoning_in_mllms.md)**

:   提出 EgoMind，一种无需几何先验的 CoT 框架，通过角色扮演字幕 (RPC) 和渐进式空间分析 (PSA) 两个核心组件，仅用 5K SFT + 20K RL 样本即可实现多帧空间推理的竞争性能力。

**[EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning in Multimodal Large Language Models](emo-r3_reflective_reinforcement_learning_for_emotional_reasoning_in_multimodal_l.md)**

:   提出 EMO-R3，通过结构化情感思维（SET）引导 MLLM 逐步进行情感推理，并设计反思情感奖励（RER）让模型重新评估推理的视觉-文本一致性和情感连贯性，显著提升多模态情感理解的可解释性和准确性。

**[EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards](evolmm_self-evolving_large_multimodal_models_with_continuous_rewards.md)**

:   提出 EvoLMM，一个完全无监督的自演化框架：从单一骨干 LMM 中分出 Proposer（生成视觉问题）和 Solver（多次回答），通过连续自一致性奖励取代离散多数投票，让模型仅用原始图片即可自我提升多模态数学推理能力（ChartQA +2.7%, MathVista +2.1%）。

**[EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards](evolmm_self_evolving_lmm_continuous_rewards.md)**

:   提出 EvoLMM，一个纯无监督的自进化框架：从单一LMM分出Proposer（生成图像相关问题）和Solver（回答问题），通过连续自一致性奖励（替代离散多数投票）形成闭环训练信号，仅使用原始图像（无标注、无外部奖励模型），在8个多模态数学推理基准上获得约2-3%的一致性提升。

**[Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory](evolving_contextual_safety_in_multi-modal_large_language_models_via_inference-ti.md)**

:   提出 MM-SafetyBench++ 基准和 EchoSafe 框架，通过推理时维护自反思记忆库来累积安全洞察，使 MLLM 能够根据上下文区分看起来相似但安全意图不同的场景，无需训练即可提升上下文安全性。

**[EvoPrompt: Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_vision-language_models.md)**

:   EvoPrompt 通过轨迹感知的 prompt 进化策略（统一 embedding 投影 + 方向-幅度解耦训练 + 特征几何正则化）解决 VLM prompt learning 中的灾难性遗忘和模态偏差问题，在 few-shot/跨数据集/域泛化任务上全面 SOTA 且保持 zero-shot 能力。

**[Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_visionlanguage_mode.md)**

:   提出 EvoPrompt 框架，将提示训练视为从通用语义锚点到任务特征的渐进进化过程，通过模态共享提示投影器（MPP）统一跨层跨模态提示生成、进化轨迹感知策略（方向-幅度解耦冻结历史方向）防止遗忘、特征几何正则化（FGR）防止表示坍缩，在 11 个数据集 base-to-novel 泛化上平均 HM 达 80.73%，超越所有现有提示学习方法。

**[Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration](explore_with_long-term_memory_a_benchmark_and_multimodal_llm-based_reinforcement.md)**

:   本文提出 LMEE 基准和 MemoryExplorer 框架，通过将多目标导航与记忆问答统一评估具身探索的过程与结果，并用强化学习微调 MLLM 使其主动调用记忆检索工具，在 LMEE-Bench 上 SR 达 23.53%（超越 3D-Mem 的 16.91%）、GOAT-Bench 上 SR 达 46.40%。

**[FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)**

:   提出 FairLLaVA，一种参数高效的公平性微调方法，通过最小化隐藏状态与人口学属性之间的互信息来消除多模态大语言模型中的人口学捷径，在胸片报告生成和皮肤病变问答中显著缩小了群体间性能差距。

**[Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)**

:   提出量化感知积分梯度（QIG），将 LVLM 量化的灵敏度分析从模态级推进到 token 级，利用公理化归因原理精确量化每个 token 对量化误差的贡献，在 W4A8 和 W3A16 设置下显著提升量化模型精度，且几乎无额外计算开销。

**[FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](flashcache_frequency_kv_cache_compression.md)**

:   提出 FlashCache，首次从频域角度分析多模态 KV Cache 的重要性分布，发现偏离低频主成分的"离群 KV"编码了推理关键特征，通过 DCT 低通滤波识别并优先保留离群 KV + 动态逐层预算分配，在 80% KV 内存压缩下实现 1.69× 解码加速且基本不损失任务性能，天然兼容 FlashAttention。

**[FluoCLIP: Stain-Aware Focus Quality Assessment in Fluorescence Microscopy](fluoclip_stain-aware_focus_quality_assessment_in_fluorescence_microscopy.md)**

:   提出 FluoCLIP，一个两阶段视觉-语言框架：先通过染色锚定（stain-grounding）让 CLIP 学习荧光染色的语义，再通过染色引导排序（stain-guided ranking）实现染色感知的对焦质量评估，并引入首个多染色组织级荧光显微镜数据集 FluoMix。

**[PinPoint: Focus, Don't Prune — Identifying Instruction-Relevant Regions for Information-Rich Image Understanding](focus_dont_prune_identifying_instruction-relevant_regions_for_information-rich_i.md)**

:   提出 PinPoint，一个两阶段框架：先通过 Instruction-Region Alignment 定位与指令相关的图像区域，再对选中区域精细化编码，以更少的 visual token 实现更高的 VQA 精度。

**[From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing](from_intuition_to_investigation_a_tool-augmented_reasoning_mllm_framework_for_ge.md)**

:   提出 TAR-FAS 框架，首次将人脸反欺骗（FAS）任务重构为 Chain-of-Thought with Visual Tools（CoT-VT）范式，让 MLLM 在推理过程中自适应调用外部视觉工具（LBP/FFT/HOG等），从"直觉判断"升级为"精细调查"，在 1-to-11 跨域协议上取得 SOTA。

**[From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings](from_observation_to_action_latent_action-based_primitive_segmentation_for_vla_pr.md)**

:   提出 LAPS（Latent Action-based Primitive Segmentation）流水线，通过在潜在动作空间中定义"Latent Action Energy"指标，从未标注的工业视频流中无监督发现和分割语义动作原语，为 VLA 模型预训练提供结构化数据。

**[GACD: Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](gacd_gradient_self_reflection_hallucination.md)**

:   通过一阶Taylor梯度估计每个token（视觉/文本/输出）对当前预测的贡献，设计GACD框架同时缓解文本-视觉偏差（增强视觉token影响力）和共现偏差（抑制与已有物体锚定的视觉token），在AMBER上提升8%总分、POPE F1提升8%，无需训练或辅助模型。

**[Generate, Analyze, and Refine: Training-Free Sound Source Localization via MLLM Meta-Reasoning](generate_analyze_and_refine_training-free_sound_source_localization_via_mllm_met.md)**

:   本文提出了一个无需训练的声源定位框架 GAR-SSL，通过将声源定位重新建模为"生成-分析-精炼"的三阶段元认知推理过程，直接利用多模态大语言模型 (MLLM) 的内在推理能力进行音视频定位，在单源和多源定位基准上取得了与训练方法可比甚至更优的性能。

**[GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning](graphvlm_benchmark_vlm_graph_learning.md)**

:   提出 GraphVLM benchmark，系统评估VLM在多模态图学习中的三种角色——VLM-as-Encoder（增强GNN特征）、VLM-as-Aligner（桥接模态用于LLM推理）、VLM-as-Predictor（直接作为图学习backbone）。在6个数据集上的实验表明，VLM-as-Predictor持续取得最佳性能，揭示了VLM作为多模态图学习新基础的巨大潜力。

**[GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning](graphvlm_benchmarking_vision_language_models_for_multimodal_graph_learning.md)**

:   提出 GraphVLM benchmark，系统评估 VLM 在多模态图学习中的三种角色（Encoder/Aligner/Predictor），发现 VLM-as-Predictor 范式一致性最优，揭示 VLM 作为多模态图推理骨干的巨大潜力。

**[GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding](groundvts_visual_token_sampling_in_multimodal_large_language_models_for_video_te.md)**

:   提出 GroundVTS，一种在视频大语言模型中进行查询引导的细粒度视觉token采样架构，通过在 token 级别自适应保留与查询相关的时空信息，在 Charades-STA 上 mIoU 提升 18.4 点，QVHighlights 上 mAP 提升 20.6 点。

**[GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)**

:   提出GTR-Turbo框架，通过合并RL训练过程中产生的历史checkpoint作为免费教师模型，在无需依赖昂贵外部API模型的条件下，实现了与GTR相当甚至更优的多轮视觉代理训练效果，同时将训练时间减少50%、计算成本降低60%。

**[GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr_turbo_merged_checkpoint_free_teacher.md)**

:   本文提出 GTR-Turbo，通过将 RL 训练过程中的历史 checkpoint 经 TIES 合并产生"免费教师模型"来指导后续训练（可选 SFT 或 KL 蒸馏方式），在多个视觉智能体任务上匹配甚至超过依赖 GPT-4o 等外部教师的 GTR 方法，同时减少 50% 训练时间和 60% 计算成本。

**[HAMMER: Harnessing MLLM via Cross-Modal Integration for Intention-Driven 3D Affordance Grounding](hammer_harnessing_mllm_via_cross-modal_integration_for_intention-driven_3d_affor.md)**

:   提出 HAMMER 框架，通过从 MLLM 中提取接触感知的意图嵌入、层次化跨模态融合增强点云特征、以及多粒度几何提升模块为意图嵌入注入3D空间信息，实现基于交互图像的3D可供性定位，在 PIAD 基准上全面超越现有方法。

**[HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models](handvqa_diagnosing_and_improving_fine-grained_spatial_reasoning_about_hands_in_v.md)**

:   构建了 HandVQA——一个包含 160 万+选择题的大规模诊断性基准，基于 3D 手部关节标注自动生成关于关节角度、距离和相对位置的 VQA 问题，系统暴露了当前 VLM 在细粒度手部空间推理上的严重缺陷，并证明在 HandVQA 上微调后的模型可零样本迁移到手势识别（+10.33%）和手-物交互识别（+2.63%）等下游任务。

**[HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models](hawk_head_importance-aware_visual_token_pruning_in_multimodal_models.md)**

:   提出 HAWK，一种基于注意力头重要性感知的视觉 token 剪枝方法，通过离线计算各注意力头对视觉理解的贡献权重，并结合文本引导的注意力分数动态评估每个视觉 token 的重要性，在 Qwen2.5-VL 上剪枝 80.2% 视觉 token 后仍保留 96.0% 原始性能，同时减少 26% 推理延迟。

**[HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_high-fidelity_in-context_learning_for_multimodal_tasks.md)**

:   通过精确分解注意力公式揭示 ICL 效应的数学本质（动态混合标准注意力输出与示例值矩阵），提出 HiFICL——用可学习低秩虚拟 key-value 对直接参数化 ICL 源头而非近似其效果，以 2.2M 参数在多模态基准上全面超越现有 ICL 近似方法。

**[HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_highfidelity_incontext_learning_for_multimo.md)**

:   HiFICL 通过严格的注意力公式推导，将 ICL 近似问题从"拟合 shift vector"重构为"直接参数化 ICL 的源头"——在注意力头中注入可学习的低秩虚拟键值对，以端到端训练实现一种动态的、上下文感知的参数高效微调方法，在多个多模态基准上以极少参数超越现有 ICL 近似方法和 LoRA。

**[HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)**

:   HiSpatial 提出将 3D 空间智能分解为四层认知层级（几何感知 → 物体属性 → 物体关系 → 抽象推理），构建了处理约 500 万张图像、4500 万个物体、20 亿 QA 对的自动化数据管线，并设计了以度量尺度点云图为辅助输入的 RGB-D VLM，以仅 3B 参数在多个空间推理基准上超越 GPT-5 和 Gemini-2.5-Pro。

**[HIVE: Query, Hypothesize, Verify — An LLM Framework for Multimodal Reasoning-Intensive Retrieval](hive_query_hypothesize_verify_an_llm_framework_for_multimodal_reasoning-intensiv.md)**

:   HIVE 是一个即插即用的多模态检索框架，通过四个阶段——初始检索 → LLM 驱动的补偿性查询合成（显式表达视觉推理缺口）→ 二次检索 → LLM 验证重排序——将推理密集型多模态检索的 nDCG@10 从最佳多模态模型的 27.6 提升至 41.7（+14.1 绝对点），无需任何额外训练。

**[HoneyBee: Data Recipes for Vision-Language Reasoners](honeybee_data_recipes_for_vision-language_reasoners.md)**

:   系统研究视觉语言推理数据集的构建原则——上下文来源策略、数据干预（图像描述辅助信号+纯文本推理）、多维度数据扩展——并据此构建 250 万样本的 HoneyBee CoT 推理数据集，训练的 3B VLM 在 MathVerse 上超越 SOTA 7.8%，同时提出降低 73% 解码成本的测试时扩展策略。

**[HoneyBee: Data Recipes for Vision-Language Reasoners](honeybee_data_recipes_vl_reasoners.md)**

:   系统性地研究了 VL 推理训练数据的设计空间——数据来源选择、干预策略筛选、图像/问题/CoT 三维度缩放——基于洞察构建了 250 万样本的 HoneyBee 数据集，3B VLM 在 MathVerse 上超越 SOTA 7.8pp，并提出共享 Caption 解码的测试时缩放策略节省 73% token。

**[HouseMind: Tokenization Allows MLLMs to Understand, Generate and Edit Architectural Floor Plans](housemind_tokenization_mllm_floor_plan.md)**

:   提出HouseMind框架，通过层次化VQ-VAE将建筑平面图离散化为轮廓token和房间实例token的结构化序列，结合三阶段多模态对齐和指令微调，以Qwen3-0.6B为backbone实现了平面图理解、生成、编辑三项任务的统一建模，几何有效性和可控性大幅超越现有方法。

**[HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)**

:   提出HulluEdit，一种单次前向、无参考模型的子空间编辑框架，通过将隐藏状态分解为正交的视觉证据子空间、冲突先验子空间和残差不确定性子空间，选择性抑制幻觉模式而不干扰视觉定位，在POPE和CHAIR基准上达到SOTA幻觉缓解效果。

**[HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs](hulluedit_subspace_editing_hallucination.md)**

:   提出HulluEdit，一个单次推理、无参考模型的幻觉缓解框架，通过将隐藏状态正交分解为视觉证据子空间、冲突先验子空间和残差不确定性子空间，选择性抑制幻觉模式而不干扰视觉接地，在POPE和CHAIR上达到SOTA。

**[Interpretable Debiasing of Vision-Language Models for Social Fairness](interpretable_debiasing_of_vision-language_models_for_social_fairness.md)**

:   提出 DeBiasLens，通过在 VLM 编码器上训练稀疏自编码器（SAE）来定位编码社会属性的"社会神经元"，然后在推理时选择性去激活这些神经元以缓解偏见，在 CLIP 上降低 Max Skew 9-16%，在 InternVL2 上降低性别偏差比例 40-50%，同时保持通用性能。

**[It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models](its_time_to_get_it_right_improving_analog_clock_reading_and_clock-hand_spatial_r.md)**

:   揭示 SOTA VLM 仍无法可靠读取真实场景中的模拟时钟（零样本准确率不到10%），提出 TickTockVQA 真实场景数据集（12K图像）和 Swap-DPO 微调框架，将 Llama-3.2-11B 的时间读取准确率从1.43%提升至46.22%。

**[Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild](joint-aligned_latent_action_towards_scalable_vla_pretraining_in_the_wild.md)**

:   提出 JALA 框架，通过联合对齐预测嵌入与逆动力学生成的潜在动作，构建统一的潜在动作空间，使 VLA 能同时从标注数据和未标注的野外人类视频中学习，配合 7.5M 样本的 UniHand-Mix 数据集显著提升机器人操作泛化性。

**[KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing](kvsmooth_mitigating_hallucination_in_multi-modal_large_language_models_through_k.md)**

:   提出KVSmooth，一种免训练的即插即用方法，通过注意力行熵引导的自适应指数移动平均（EMA）对KV-Cache进行平滑，有效抑制多模态大语言模型（MLLM）在解码过程中因sink token引发的语义漂移与幻觉生成，在LLaVA-1.5上将CHAIR_S从41.8降至18.2（降幅56%），同时F1从77.5提升至79.2。

**[KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing](kvsmooth_mitigating_hallucination_in_multimodal_la.md)**

:   KVSmooth 提出免训练即插即用方法，通过对 KV-Cache 施加注意力行熵引导的自适应 EMA 平滑，将 LLaVA-1.5 的 CHAIR_S 从 41.8 降至 18.2（降 56%），同时 F1 从 77.5 提升到 79.2，精度召回同时提升。

**[Label-Free Cross-Task LoRA Merging with Null-Space Compression](label-free_cross-task_lora_merging_with_null-space_compression.md)**

:   观察到LoRA微调过程中下投影矩阵A的零空间比率随训练下降且与性能强相关，据此提出NSC Merging，一种无标签、任务无关的LoRA合并方法，在20个异构视觉任务、6个NLI任务和VLM评估上达到SOTA。

**[Learning What Matters: Prioritized Concept Learning via Relative Error-driven Sample Selection](learning_what_matters_prioritized_concept_learning_via_relative_error-driven_sam.md)**

:   提出 PROGRESS 框架，通过追踪 VLM 在自动发现的多模态概念集群上的学习进度来动态选择最有信息量的训练样本，仅用 16-20% 的标注数据就达到全数据 99-100% 的性能，且总训练时间更短。

**[LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models](llavashield_multimodal_multiturn_safety.md)**

:   针对VLM多模态多轮对话中的恶意意图隐蔽性、上下文风险累积和跨模态联合风险三大挑战，构建4,484个标注对话的MMDS数据集和基于MCTS的MMRT红队框架，提出LLaVAShield审计模型，在用户/助手两侧分别达到F1 95.71%/92.24%，大幅超越GPT-5-mini等基线。

**[LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models](llavashield_safeguarding_multimodal_multi-turn_dialogues_in_vision-language_mode.md)**

:   提出 LLaVAShield——首个面向多模态多轮对话的内容审核模型，配套构建了 MMDS 数据集（4,484条对话、8大类60子类风险体系）和基于 MCTS 的自动化红队攻击框架 MMRT，在用户/助手双端安全审计上大幅超越 GPT-5-mini 等基线。

**[LLMind: Bio-inspired Training-free Adaptive Visual Representations for Vision-Language Models](llmind_bio-inspired_training-free_adaptive_visual_representations_for_vision-lan.md)**

:   受人眼中央凹编码和皮层放大机制启发，提出无需训练的自适应采样框架 LLMind，通过 Möbius 变换实现非均匀像素分配，并利用闭环语义反馈在测试时优化采样参数，在仅使用 1%-5% 像素的紧张预算下大幅超越均匀采样。

**[Locate-then-Sparsify: Attribution Guided Sparse Strategy for Visual Hallucination Mitigation](locate-then-sparsify_attribution_guided_sparse_strategy_for_visual_hallucination.md)**

:   提出 LTS-FS（Locate-Then-Sparsify for Feature Steering）框架，通过因果干预归因方法定位幻觉相关层，并根据归因分数逐层稀疏地控制特征引导强度，在有效缓解 LVLM 幻觉的同时保持模型泛化能力。

**[MA-Bench: Towards Fine-grained Micro-Action Understanding](ma-bench_towards_fine-grained_micro-action_understanding.md)**

:   提出 MA-Bench 微动作理解基准，包含 1000 个视频和 12000 个结构化 QA 对，通过"感知-理解-推理"三层评估架构系统测试 23 个 MLLM 的细粒度微动作理解能力，并构建 20.5K 训练语料 MA-Bench-Train 用于模型微调提升。

**[MarkushGrapher-2: End-to-end Multimodal Recognition of Chemical Structures](markushgrapher-2_end-to-end_multimodal_recognition_of_chemical_structures.md)**

:   MarkushGrapher-2 提出了一个端到端多模态化学结构识别模型，通过专用化学 OCR 模块联合编码图像、文本和布局信息，结合两阶段训练策略（先适配 OCSR 特征再融合多模态编码），在 Markush 结构识别上大幅超越现有方法（M2S 准确率 56% vs 38%），同时保持分子结构识别的竞争力。

**[MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](masquant_modality-aware_smoothing_quantization_for_multimodal_large_language_mod.md)**

:   揭示了通道平滑量化（如 SmoothQuant）直接应用于 MLLM 时的"平滑失配"问题——不同模态激活幅度差异巨大导致非主导模态被过度平滑，提出 MASQuant 通过模态感知平滑因子和基于 SVD 白化的跨模态低秩补偿解决该问题。

**[Mastering Negation: Boosting Grounding Models via Grouped Opposition-Based Learning](mastering_negation_boosting_grounding_models_via_g.md)**

:   构建首个包含正负语义成对描述的视觉定位数据集 D-Negation (14K 图片, 140K 标注), 并提出 Grouped Opposition-Based Learning (GOBL) 微调机制, 通过 PNC 和 TSO 两个对立损失函数, 仅调不到 10% 参数即让 Grounding DINO 和 APE 在否定语义评估上提升最高 5.7 mAP, 且正面语义也同步提升.

**[Mastering Negation: Boosting Grounding Models via Grouped Opposition-Based Learning](mastering_negation_boosting_grounding_models_via_grouped_opposition-based_learni.md)**

:   提出 D-Negation 数据集和 Grouped Opposition-Based Learning (GOBL) 微调机制，通过对立语义配对和两个专用损失函数，仅微调不到 10% 参数即大幅提升视觉定位模型对否定语义的理解能力（最高 +5.7 mAP）。

**[Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](mitigating_multimodal_hallucinations_via_gradient-based_self-reflection.md)**

:   提出 GACD（Gradient-based Influence-Aware Constrained Decoding），利用一阶 Taylor 梯度估计每个 token 对输出的影响力，在推理阶段同时缓解文本-视觉偏差和共现偏差导致的多模态幻觉，无需辅助模型或微调。

**[MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](modes_accelerating_mixture-of-experts_multimodal_large_language_models_via_dynam.md)**

:   提出 MoDES，首个面向 MoE 多模态大模型的训练免调专家跳过框架，通过全局调制的局部门控（GMLG）和双模态阈值（DMT）机制自适应跳过冗余专家，在跳过 88% 专家时仍保留 97%+ 原始性能，并实现 2.16× prefill 加速。

**[MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](modes_moe_dynamic_expert_skipping.md)**

:   首个针对MoE多模态大模型的专家跳过框架MoDES,通过全局调制局部门控(GMLG)将层级重要性融入路由概率、双模态阈值(DMT)对文本/视觉token分别设定跳过策略、前沿搜索高效优化阈值,在Qwen3-VL-MoE-30B上88%专家跳过仍保留97.33%精度,prefill加速2.16×。

**[MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)**

:   将 MoE 中的专家选择建模为序列决策问题，通过 GRPO 强化学习优化路由策略，引入模态感知路由引导，在 VLM 的图像和视频理解任务上一致超越确定性 top-K 路由及其变体。

**[More than the Sum: Panorama-Language Models for Adverse Omni-Scenes](more_than_the_sum_panorama-language_models_for_adverse_omni-scenes.md)**

:   提出 Panorama-Language Modeling（PLM）范式和 PanoVQA 大规模全景 VQA 数据集（653K QA 对），设计即插即用的全景稀疏注意力模块让现有 VLM 无需重训练即可处理等距柱状投影全景图，在遮挡和事故等恶劣场景下实现优于多视角拼接方案的全局推理。

**[Mixture of States (MoS): Routing Token-Level Dynamics for Multimodal Generation](mos_mixture_of_states_multimodal_generation.md)**

:   提出Mixture of States (MoS)——一种新的多模态扩散模型融合范式，用可学习的token级路由器将理解塔(冻结LLM/VLM)的任意层hidden state动态路由到生成塔(DiT)的任意层，以3-5B参数在图像生成和编辑上匹配或超越20B的Qwen-Image。

**[Mostly Text, Smart Visuals: Asymmetric Text-Visual Pruning for Large Vision-Language Models](mostly_text_smart_visuals_asymmetric_text-visual_pruning_for_large_vision-langua.md)**

:   通过 MoT 探针实验揭示 LVLM 中文本通路和视觉通路对剪枝的不对称敏感性——文本通路高度敏感必须用文本 token 校准、视觉通路高度冗余可承受 60% 稀疏度，据此提出 ATV-Pruning 使用全部文本 token + 逐层自适应选择的少量视觉 token 构建校准池。

**[MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding](msjoe_jointly_evolving_mllm_and_sampler_for_efficient_long-form_video_understand.md)**

:   提出 MSJoE 框架，将 MLLM 和轻量关键帧采样器通过强化学习联合进化——MLLM 生成视觉查询引导帧检索，1D U-Net 采样器从 CLIP 相似度矩阵中学习选帧，两者端到端联合优化实现长视频问答中 +8% 的准确率提升。

**[Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following](multi-crit_benchmarking_multimodal_judges_on_pluralistic_criteria-following.md)**

:   构建首个评估多模态 Judge 模型多准则遵循能力的基准 Multi-Crit，包含准则级人类标注和偏好冲突样本，配合 PAcc/TOS/CMR 三个新指标，全面评估 25 个 LMM 并揭示闭源最强模型在开放生成任务上仅 32.78% 的多准则一致性。

**[Multi-Modal Image Fusion via Intervention-Stable Feature Learning](multi-modal_image_fusion_via_intervention-stable_feature_learning.md)**

:   提出一个受因果推理启发的多模态图像融合框架，通过三种结构化干预策略（互补掩码、随机掩码、模态丢弃）探测模态间的真实依赖关系，并设计因果特征整合器 (CFI) 学习干预稳定特征，在 MSRS 上 PSNR 达到 66.02、AG 达到 4.129，目标检测 mAP 达到 0.821。

**[Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](multi-modal_representation_learning_via_semi-supervised_rate_reduction_for_gener.md)**

:   提出 SSR²-GCD 框架，通过半监督编码率减少（Semi-Supervised Rate Reduction）损失学习模态内均匀压缩的结构化表征，并结合检索式文本聚合策略增强跨模态知识迁移，在8个数据集上超越现有多模态GCD方法。

**[Multimodal OCR: Parse Anything from Documents](multimodal_ocr_parse_anything_from_documents.md)**

:   提出Multimodal OCR (MOCR)范式，将文档中的文本和图形（图表、图示、UI组件等）统一解析为结构化文本表示（文本+SVG代码），训练3B参数的dots.mocr模型在OCR Arena排名仅次于Gemini 3 Pro，在olmOCR Bench达到83.9 SOTA，在image-to-SVG基准上超越Gemini 3 Pro。

**[Narrative Weaver: Towards Controllable Long-Range Visual Consistency with Multi-Modal Conditioning](narrative_weaver_towards_controllable_long-range_visual_consistency_with_multi-m.md)**

:   提出 Narrative Weaver 框架，结合 MLLM 的叙事规划与扩散模型的精细生成，通过可学习查询和动态 Memory Bank 实现多模态条件下的长程视觉一致性生成，并构建首个电商广告视频分镜数据集 EAVSD（330K+ 图像）。

**[No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models](no_hard_negatives_required_concept_centric_learning_leads_to_compositionality_wi.md)**

:   C2LIP 提出不依赖 hard negatives 的对比学习微调方案：通过将文本拆解为名词短语概念并引入跨模态注意力池化，在 SugarCrepe/SugarCrepe++ 组合性基准上达到 SOTA，同时保持甚至提升零样本和检索性能。

**[No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection](no_need_for_real_anomaly_mllm_empowered_zero-shot_video_anomaly_detection.md)**

:   提出端到端零样本视频异常检测框架 LAVIDA，通过异常暴露采样器将语义分割数据集转化为伪异常进行训练，结合 MLLM 提取深层异常语义特征和反注意力 token 压缩处理时空稀疏性，无需任何真实 VAD 数据即实现帧级/像素级 SOTA。

**[Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](noise-aware_few-shot_learning_through_bi-directional_multi-view_prompt_alignment.md)**

:   提出NA-MVP框架，通过双向（clean + noise-aware）多视图prompt设计配合非平衡最优传输（UOT）实现细粒度patch-to-prompt对齐，并用经典OT对识别出的噪声样本做选择性标签修正，在噪声小样本学习场景下持续超越SOTA。

**[Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](noiseaware_fewshot_learning_through_bidirectional.md)**

:   提出NA-MVP框架，通过双向（clean+noise-aware）多视图prompt设计配合非平衡最优传输（UOT）实现细粒度patch-to-prompt对齐，并用经典OT对识别出的噪声样本做选择性标签修正，在噪声小样本学习场景下持续超越SOTA。

**[OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models](oddgridbench_exposing_the_lack_of_fine-grained_visual_discrepancy_sensitivity_in.md)**

:   提出 OddGridBench 评估 MLLM 的细粒度视觉差异感知能力（找出网格中与其他元素在颜色/大小/旋转/位置上不同的那个），发现所有 MLLM 远低于人类水平，进而提出 OddGrid-GRPO（课程学习 + 距离感知奖励）显著提升模型的视觉辨别力。

**[OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens](omnilottie_generating_vector_animations_via_parameterized_lottie_tokens.md)**

:   OmniLottie 提出一种将 Lottie JSON 文件转化为结构化命令-参数序列的 Lottie Tokenizer，使预训练 VLM 可以基于多模态交叉指令生成高质量矢量动画，并构建了 MMLottie-2M 大规模数据集支撑训练。

**[On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)**

:   揭示了动态 MoE 持续学习中"token 困境"——新任务数据中的模糊和旧 token 对新知识贡献微弱却会导致路由漂移和灾难性遗忘，提出 LLaVA-DyMoE 通过 Token Assignment Guidance 和 Routing Score Regularization 缓解路由漂移，在 CoIN 基准上 MFN 提升超 7%，遗忘降低 12%。

**[Overthinking Causes Hallucination: Tracing Confounder Propagation in Vision Language Models](overthinking_causes_hallucination_tracing_confounder_propagation_in_vision_langu.md)**

:   揭示VLM幻觉的新机制——"过度思考"(overthinking)：模型在中间解码层产生过多竞争性物体假设，混杂因子沿层传播至最终预测引发幻觉；提出Overthinking Score量化层间假设多样性×不确定性，在MSCOCO上F1达78.9%，OOD AMBER上71.58%。

**[Overthinking Causes Hallucination: Tracing Confounder Propagation in Vision Language Models](overthinking_hallucination_confounder_propagation.md)**

:   发现 VLM 幻觉的新机制——"过度思考"（overthinking）：模型在解码器中间层产生过多竞争性物体假设，导致语义关联但不存在的"混杂因子"传播到最终层引发幻觉，提出 Overthinking Score 量化层间假设多样性与不确定性的乘积，在 MSCOCO 上达 87.33% AUC / 78.9% F1，AMBER OOD 上 71.58% F1。

**[Parallel In-context Learning for Large Vision Language Models](parallel_in-context_learning_for_large_vision_language_models.md)**

:   提出 Parallel-ICL，将多模态 in-context learning 的长 demonstration 上下文分块并行处理，通过加权 Product-of-Experts 在 logit 层集成，实现与全上下文 MM-ICL 相当甚至更优的性能，同时显著降低推理延迟。

**[Phantasia: Context-Adaptive Backdoors in Vision Language Models](phantasia_context-adaptive_backdoors_in_vision_language_models.md)**

:   Phantasia 首次提出上下文自适应的 VLM 后门攻击——攻击者预设一个目标问题，中毒模型在接收到触发图片后不再回答用户原始问题，而是回答攻击者的目标问题，且生成的答案与输入图像语义一致、在语言上自然流畅，从而绕过 STRIP-P 和 ONION-R 等防御；同时本文首次证明了现有 VLM 后门攻击的隐蔽性被严重高估。

**[PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models](pointalign_feature-level_alignment_regularization_for_3d_vision-language_models.md)**

:   提出 PointAlign，在 3D VLM 的 LLM 中间层对点云 token 施加特征级对齐正则化（与 Q-Former 输出对齐），仅训练轻量对齐投影器和 LoRA 适配器，即可有效防止几何信息在语言建模过程中退化，在开放词汇分类上提升 7.50pp。

**[Proof-of-Perception: Certified Tool-Using Multimodal Reasoning with Compositional Conformal Guarantees](pop_proof_of_perception_conformal_reasoning.md)**

:   提出 Proof-of-Perception (PoP)，将多模态推理建模为可执行的有向无环图(DAG)，每个感知/逻辑节点输出带有保形预测证书的集合值（提供逐步可靠性保证），并用轻量控制器基于这些证书在计算预算内自适应分配算力，在文档、图表和多图QA基准上超越CoT、ReAct和PoT基线。

**[Predictive Regularization Against Visual Representation Degradation in Multimodal Large Language Models](predictive_regularization_against_visual_representation_degradation_in_multimoda.md)**

:   本文系统诊断了MLLM中LLM中间层视觉表征在全局功能和patch语义结构两个层面的退化现象，揭示其本质是纯文本生成目标下的"视觉牺牲"，并提出Predictive Regularization (PRe) 通过让退化的中间层特征预测初始视觉特征来缓解退化，在多个VL基准上取得一致提升。

**[Prime Once, then Reprogram Locally: An Efficient Alternative to Black-Box Service Model Adaptation](prime_once_then_reprogram_locally_an_efficient_alternative_to_black-box_service_.md)**

:   本文提出AReS方法，用单次API查询预热本地编码器代替传统零阶优化（ZOO）的持续API调用，在GPT-4o上获得+27.8%提升（ZOO方法几乎无效），同时将API调用量减少99.99%以上，实现了无成本推理。

**[Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving](prune2drive_a_plug-and-play_framework_for_accelerating_vision-language_models_in.md)**

:   首个面向多视角自动驾驶 VLM 的即插即用 token 剪枝框架，通过 T-FPS（token 级最远点采样）保持语义与空间多样性，配合视图自适应剪枝率优化自动分配各摄像头 token 预算，在 DriveLM 上仅保留 10% token 即实现 6.40× prefill 加速且性能仅降 3%。

**[Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving](prune2drive_vlm_accel_autonomous_driving.md)**

:   首个面向多视角自动驾驶VLM的即插即用token剪枝框架Prune2Drive，通过T-FPS（token级最远点采样）保持语义/空间多样性 + 视图自适应剪枝率优化自动分配不同视角的token预算,在DriveLM上仅保留10% token即实现6.40×prefill加速且性能仅降3%。

**[Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher](purify-then-align_towards_robust_human_sensing_under_modality_missing_with_knowl.md)**

:   本文提出PTA（Purify-then-Align）框架，通过元学习驱动的模态加权机制先"净化"噪声多模态教师，再用扩散模型驱动的知识蒸馏"对齐"每个单模态学生，使单模态编码器在模态缺失场景下保持强鲁棒性，在MM-Fi和XRF55上实现SOTA。

**[Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token-aware_adaptive_error_reconstruction_with_mixture_of_experts_.md)**

:   本文提出 Quant Experts (QE)，一种基于混合专家（MoE）的 token 感知自适应量化误差补偿框架，通过将重要通道分为 token 无关和 token 依赖两组，分别用共享专家和路由专家进行全局和局部量化误差重建，在 2B-72B 规模的 VLM 上实现了显著的量化精度恢复。

**[Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token_aware_vlm_quantization.md)**

:   提出 Quant Experts (QE)，一种基于 Mixture-of-Experts 的 token 感知自适应量化误差重建框架——将重要通道分为 token-independent（高频出现、全局性）和 token-dependent（低频出现、局部性）两组，分别用共享专家和路由专家的低秩适配器来补偿全局和局部量化误差，在 W4A6 到 W3A16 的多种量化设置下一致提升 VLM 性能。

**[Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)**

:   提出 Reason-SVG 框架，通过"Drawing-with-Thought"(DwT)范式让 LLM 在生成 SVG 之前先进行显式的多阶段设计推理，并结合 SFT + GRPO 强化学习与混合奖励函数进行训练，在语义对齐、结构有效性和视觉质量上全面超越现有方法。

**[ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps](reasonmap_towards_fine-grained_visual_reasoning_from_transit_maps.md)**

:   提出 ReasonMap 基准，利用 30 个城市的高分辨率公交地图构建 1,008 个 QA 对，通过两级评估框架（正确性+质量）系统评估 16 个 MLLM 的细粒度视觉推理能力，发现开源模型中 base 优于 reasoning 而闭源模型相反。

**[ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps](reasonmap_towards_finegrained_visual_reasoning_fro.md)**

:   提出 ReasonMap 基准，利用 30 个城市的高分辨率地铁线路图构建 1,008 个问答对，系统评估 16 个 MLLM 的细粒度视觉理解和空间推理能力，揭示了开源模型中 base 变体反超推理变体的反直觉现象，并建立了 GRPO 强化微调的训练基线。

**[Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress](recurrent_reasoning_with_vision-language_models_for_estimating_long-horizon_embo.md)**

:   提出 R²VLM，通过循环推理框架逐步处理本地视频片段，维护动态更新的 CoT 记录任务分解和完成状态，结合多维 RL 奖励实现长时域具身任务进度估计的 SOTA，并支持策略学习、奖励建模、主动辅助等下游应用。

**[World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training](rehearsevla_simulated_post-training_for_vlas_with_physically-consistent_world_mo.md)**

:   提出 World-Env 框架，利用物理一致的世界模型作为虚拟环境替代真实交互，对 VLA 模型进行 RL post-training，仅需每任务 5 条示教即可显著提升操控成功率。

**[World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training](rehearsevla_simulated_posttraining_world_model.md)**

:   提出 World-Env 框架，用物理一致的世界模型作为虚拟仿真器替代真实世界交互，结合 VLM 引导的即时反射器提供连续奖励和动态终止信号，实现 VLA 模型在仅 5 条示范轨迹下的安全高效 RL 后训练，平均成功率从 74.85% 提升至 79.6%。

**[ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)**

:   提出 ReMoRa，直接操作视频压缩表示（I帧 + 运动向量），通过 Refined Motion Representation (RMR) 模块将粗糙的块级运动向量精化为接近光流的细粒度运动表征，再用 Hierarchical Motion State Space (HMSS) 模块进行线性时间的长程时间建模，在 LongVideoBench、NExT-QA、MLVU 等基准上超越基线。

**[Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)**

:   提出 Residual Decoding (ResDec)——一种训练免的即插即用解码策略，通过分析历史 token 的 logit 分布中的 U 型 JSD 模式发现语义锚定阶段，聚合该阶段的历史 logits 作为残差引导融入当前解码，以近乎零的额外推理开销有效抑制 LVLM 中的语言先验幻觉。

**[Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token](rethinking_mllm_itself_as_a_segmenter_with_a_single_segmentation_token.md)**

:   提出 SELF1E，首次实现不依赖专用 mask 解码器且仅用单个 [SEG] token 的 MLLM 分割方法，通过 Residual Features Refilling (RFR) 和 Residual Features Amplifier (RFA) 恢复 pixel-shuffle 压缩造成的分辨率损失，在多个分割任务上达到与解码器方法竞争力相当的性能。

**[Revisiting Model Stitching In the Foundation Model Era](revisiting_model_stitching_in_the_foundation_model_era.md)**

:   提出针对异构视觉基础模型(VFM)的两阶段拼接训练方法(Final Feature Matching + Task Loss Training)，证明异构VFM可以可靠拼接且融合互补知识，并设计VFM Stitch Tree (VST)架构实现多VFM系统的可控精度-效率权衡。

**[Revisiting Multimodal KV Cache Compression: A Frequency-Domain-Guided Outlier-KV-Aware Approach](revisiting_multimodal_kv_cache_compression_a_frequency-domain-guided_outlier-kv-.md)**

:   提出FlashCache——首个不依赖注意力分数、无需训练的多模态KV Cache压缩框架，通过频域低通滤波识别Outlier KV并动态分配各层预算，在保持性能的前提下实现80%内存节省和1.69×解码加速。

**[SALMUBench: A Benchmark for Sensitive Association-Level Multimodal Unlearning](salmubench_a_benchmark_for_sensitive_association-level_multimodal_unlearning.md)**

:   提出 SALMUBench——首个针对 CLIP 类模型的关联级别机器遗忘基准，包含 60K 合成人物-敏感属性配对数据集、从头训练的 Compromised/Clean 模型对，以及结构化 holdout 集评估协议，首次系统揭示了现有遗忘方法的三种失败模式（灾难性破坏、过度泛化遗忘、无效遗忘）。

**[Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework](scaling_test-time_robustness_of_vision-language_models_via_self-critical_inferen.md)**

:   提出 Self-Critical Inference (SCI) 框架，通过多轮文本+视觉反事实推理的 logit 聚合来同时解决 LVLM 的语言偏差和语言敏感性问题，并提出 DRBench 动态鲁棒性基准来模型特异地评估鲁棒性。增加反事实推理轮次可持续提升鲁棒性，开辟了测试时缩放的新方向。

**[Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)**

:   提出 FlexMem——一种训练免的视觉记忆机制，通过迭代式双路径 KV 缓存压缩构建视觉记忆库，结合编码式和快速索引式记忆召回策略，让 MLLM 在单张 3090 GPU 上处理 1000+ 帧长视频，大幅超越现有高效视频理解方法。

**[Scene-VLM: Multimodal Video Scene Segmentation via Vision-Language Models](scene-vlm_multimodal_video_scene_segmentation_via_vision-language_models.md)**

:   提出 Scene-VLM——首个基于微调 VLM 的视频场景分割框架，通过结构化多模态镜头表征（视觉帧+对白+元数据）、因果序列预测、上下文-焦点窗口机制和 token logits 置信度提取，在 MovieNet 上取得 +6 AP 和 +13.7 F1 的大幅提升，并展示了自然语言解释能力。

**[Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness](seeing_clearly_reasoning_confidently_plug-and-play_remedies_for_vision_language_.md)**

:   提出一种高效的即插即用模块，通过学习多模态类嵌入来增强 VLM 对稀有物体的识别和推理能力：在视觉端用 cross-attention 适配器精化视觉 token，在文本端注入物体检测提示，无需微调 VLM 即可在 CODA-LM 上获得 72.8→75.4 的显著提升。

**[Self-Consistency for LLM-Based Motion Trajectory Generation and Verification](self-consistency_for_llm-based_motion_trajectory_generation_and_verification.md)**

:   将 LLM 的自一致性范式从自然语言推理扩展到视觉域——用 Lie 变换群层次结构定义运动轨迹的形状族，通过在变换不变距离度量下聚类 LLM 采样的多条轨迹，实现无监督的轨迹生成改进（+4-6%）和验证（精度+11.8%），无需训练。

**[SoPE: Spherical Coordinate-Based Positional Embedding for 3D LVLMs](sope_spherical_positional_encoding_3d_lvlm.md)**

:   揭示 RoPE 在 3D LVLM 中的空间感知偏差问题（1D 索引破坏 3D 局部性且忽视方向），提出球面坐标位置编码 SoPE（$(t,r,\theta,\phi)$ 四维索引 + 多维频率分配 + 多尺度混合），在 SpatialLM 上实现 3D 布局估计和物体检测 SOTA。

**[SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)**

:   提出SpatiaLQA基准（9605个QA对、241个真实室内场景），系统评估41个VLM在空间逻辑推理上的表现，并设计递归场景图辅助推理方法来提升VLM的空间逻辑推理能力。

**[SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning](spatialstack_layered_geometry-language_fusion_for_3d_vlm_spatial_reasoning.md)**

:   提出SpatialStack框架，将多视图几何编码器（VGGT）的多层级几何特征逐层注入LLM解码器的不同层（而非仅融合最后一层），通过浅层→细粒度空间感知、深层→高层语义推理的层级对齐，在多个3D空间推理基准上达到开源SOTA。

**[SSR2-GCD: Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](ssr2gcd_rate_reduction_category_discovery.md)**

:   提出SSR2-GCD框架，通过半监督率缩减(SSR2)损失替代传统对比损失来学习均匀压缩的结构化表示，并发现模态间对齐在多模态GCD中不仅不必要甚至有害，在Stanford Cars和Flowers102上分别领先SOTA 3.1%和6.3%。

**[See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles (StaR)](star_see_think_act_gui_agent_toggles.md)**

:   揭示现有多模态GUI Agent在开关控制(toggle)任务上的严重失败（GPT-5仅37% O-AMR），提出State-aware Reasoning (StaR)方法通过三步推理链（感知当前状态→分析目标状态→决定是否操作）将执行准确率提升30%+，同时不损害通用Agent能力。

**[StructXLIP: Enhancing Vision-Language Models with Multimodal Structural Cues](structxlip_enhancing_vision-language_models_with_multimodal_structural_cues.md)**

:   StructXLIP 将边缘图（edge map）作为视觉结构的代理表示，在 CLIP 微调中引入三种结构中心损失（边缘-结构文本对齐 + 局部区域-文本块匹配 + 边缘-彩色图连接），通过最大化多模态结构表示的互信息引导模型走向更鲁棒的语义稳定最优解，在跨模态检索任务上超越现有竞争者。

**[Taxonomy-Aware Representation Alignment for Hierarchical Visual Recognition with Large Multimodal Models](taxonomy-aware_representation_alignment_for_hierarchical_visual_recognition_with.md)**

:   提出TARA框架，通过将LMM的中间表示与生物基础模型(BFM)的分类学感知特征对齐，为大型多模态模型注入分类层次知识，显著提升已知和新颖类别的层次化视觉识别性能。

**[Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)**

:   提出Vision-Guided Attention (VGA)，一种免训练的方法，通过利用视觉token的语义特征构建精确的视觉定位，引导模型注意力聚焦于相关视觉区域，有效缓解MLLM幻觉，且兼容FlashAttention。

**[Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)**

:   发现LVLM后门行为的本质是跨模态注意力窃取（trigger视觉token抢夺文本token的注意力），提出CleanSight——首个无需训练的测试时后门防御框架，通过检测和剪枝高注意力trigger token来消除后门效应。

**[Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction](text-only_training_for_image_captioning_with_retrieval_augmentation_and_modality.md)**

:   提出TOMCap——一种纯文本训练的图像描述方法，通过检索增强+模态差距修正+LoRA微调，在训练时只用文本而推理时处理图像，超越了已有的无训练和纯文本方法。

**[The LLM Bottleneck: Why Open-Source Vision LLMs Struggle with Hierarchical Visual Recognition](the_llm_bottleneck_why_open-source_vision_llms_struggle_with_hierarchical_visual.md)**

:   揭示开源LLM缺乏关于视觉世界的层次分类知识（甚至不知道基本的生物分类体系），这使得LLM成为Vision LLM层次视觉识别的瓶颈。

**[The More, the Merrier: Contrastive Fusion for Higher-Order Multimodal Alignment](the_more_the_merrier_contrastive_fusion_for_higher-order_multimodal_alignment.md)**

:   提出Contrastive Fusion (ConFu)框架，将CLIP式的双模态对比学习推广到三模态高阶对齐，在统一目标中同时学习配对和融合表示，支持1→1和2→1检索。

**[Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth](think_360_evaluating_the_width-centric_reasoning_capability_of_mllms_beyond_dept.md)**

:   本文提出 Think360，一个聚焦于"推理宽度"（即模型在多路径搜索、多约束剪枝、回溯试错等方面的能力）的多模态基准，包含 1200+ 高质量样本，并设计细粒度 Tree-of-Thought 评估协议，揭示当前 MLLM 在宽度方向推理上的显著短板。

**[Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)**

:   提出 Dyn-Bench——一个面向 4D 物理世界动态理解的大规模基准（1k 视频、7k VQA 对、3k 动态 grounding 对），系统评估了通用/空间/区域级 MLLM 的时空推理能力，发现现有模型无法同时维持推理和 grounding 的一致性，并提出 Mask-Guided Fusion 和 ST-TCM 两种结构化集成方法显著提升动态感知。

**[TIGeR: A Unified Framework for Time, Images and Geo-location Retrieval](tiger_a_unified_framework_for_time_images_and_geo-location_retrieval.md)**

:   提出TIGeR框架，通过多模态Transformer联合学习图像-位置-时间的统一地理时间嵌入空间，实现地理定位、拍摄时间预测和地理时间感知图像检索三个任务的统一，并构建了4.5M规模的高质量基准数据集。

**[TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)**

:   系统调查构建MLLM视频时间定位（VTG）能力的关键因素，从数据质量和算法设计两个维度出发，发布高质量基准TimeLens-Bench和训练集TimeLens-100K，并通过交错文本时间编码+thinking-free RLVR训练范式构建TimeLens系列模型，在开源模型中达到SOTA并超越GPT-5和Gemini-2.5-Flash。

**[Token Warping Helps MLLMs Look from Nearby Viewpoints](token_warping_helps_mllms_look_from_nearby_viewpoints.md)**

:   提出对 MLLM 的 ViT image token 做空间 warping（而非传统的像素级 warping）来模拟视角变换，发现 backward token warping 在保持语义一致性同时对深度估计噪声鲁棒，在自建的 ViewBench 上大幅超越像素级 warping、专用空间推理 MLLM 和生成式 warping 方法。

**[Tokenization Allows Multimodal Large Language Models to Understand, Generate and Edit Architectural Floor Plans (HouseMind)](tokenization_allows_multimodal_large_language_models_to_understand_generate_and_.md)**

:   提出 HouseMind，通过层次化 VQ-VAE 将建筑平面图离散化为房间级空间 token，在统一的 MLLM 框架中实现平面图理解、生成和编辑三大任务，在几何有效性和可控性上全面超越扩散模型和通用 VLM 基线。

**[Topo-R1: Detecting Topological Anomalies via Vision-Language Models](topo-r1_detecting_topological_anomalies_via_vision-language_models.md)**

:   提出Topo-R1——首个赋予VLM拓扑感知能力的框架，通过自动化数据构建管线+SFT+GRPO强化学习（含拓扑感知复合奖励），实现无标注的管状结构拓扑异常检测与分类。

**[Towards Calibrating Prompt Tuning of Vision-Language Models](towards_calibrating_prompt_tuning_of_vision-language_models.md)**

:   针对prompt tuning后CLIP面临的"双重误校准"问题（基类欠自信+新类过自信），提出均值-方差margin正则化和文本矩匹配损失两个互补正则项，作为即插即用模块在7种prompt tuning方法和11个数据集上显著降低ECE。

**[Towards Multimodal Domain Generalization with Few Labels](towards_multimodal_domain_generalization_with_few_labels.md)**

:   定义并研究半监督多模态域泛化(SSMDG)新问题，提出融合一致性驱动伪标签、分歧感知正则化和跨模态原型对齐的统一框架，在少量标注下实现多模态模型的跨域泛化。

**[Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training](towards_real-world_document_parsing_via_realistic_scene_synthesis_and_document-a.md)**

:   提出数据-训练协同设计框架 DocHumming：通过 Realistic Scene Synthesis 构建 DocMix-3M 大规模合成数据集，结合渐进学习和结构 token 加权的 Document-Aware Training Recipe，在仅 1B 参数的 MLLM 上实现 OmniDocBench Overall 93.75（超越 Qwen3-VL-235B 的 89.15），且在真实拍摄场景下仅退化 6.72 分（模块化方法退化 18-20 分）。

**[TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition](trivia_self-supervised_fine-tuning_of_vision-language_models_for_table_recogniti.md)**

:   提出 TRivia 自监督微调框架，通过表格问答（QA）驱动的 GRPO 强化学习，让 VLM 直接从无标注表格图像中学习表格识别能力，3B 参数的 TRivia-3B 在多个基准上超越 Gemini 2.5 Pro 和 GPT-5 等私有模型。

**[Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models](uncertainty-aware_knowledge_distillation_for_multimodal_large_language_models.md)**

:   提出Beta-KD，一种基于贝叶斯视角的不确定性感知知识蒸馏框架，通过将教师监督建模为Gibbs先验并用Laplace近似推导闭形解，自动调节数据与教师信号的平衡，在多模态VQA基准上持续提升蒸馏效果。

**[Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models](uncertainty-guided_compositional_alignment_with_part-to-whole_semantic_represent.md)**

:   提出UNCHA框架，在双曲VLM中用双曲不确定性建模部分图像对整体场景的语义代表性，通过不确定性引导的对比损失和蒸含损失增强组合性场景理解，在多个下游任务上超趇现有双曲VLM。

**[Understanding Task Transfer in Vision-Language Models](understanding_task_transfer_in_vision-language_models.md)**

:   本文首次系统研究了 VLM 在一个视觉感知任务上微调后对其他感知任务零样本性能的影响，提出 Perfection Gap Factor (PGF) 归一化指标量化跨任务迁移，在 Qwen-2.5-VL 三个尺度模型上揭示了任务迁移的结构性规律（正/负迁移团、任务角色分类、尺度依赖等），并证明 PGF 可指导数据选择提升微调效率。

**[UniGame: Turning a Unified Multimodal Model Into Its Own Adversary](unigame_turning_a_unified_multimodal_model_into_its_own_adversary.md)**

:   UniGame 提出首个针对统一多模态模型（UMM）的自对抗后训练框架，通过在共享视觉 token 接口安装轻量扰动器，让生成分支主动创造语义一致的对抗样本来挑战理解分支，形成极小极大自博弈，显著提升一致性 (+4.6%)、理解 (+3.6%)、生成和鲁棒性。

**[UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](unimmad_multimodal_moe_anomaly_detection.md)**

:   提出 UniMMAD, 首个统一多模态 (RGB/Depth/IR 等) 多类别异常检测框架, 通过 General-to-Specific 范式: 通用多模态编码器压缩特征, Cross Mixture-of-Experts (C-MoE) 解压为域特定特征, 在 5 个数据集 (含工业/医学/合成场景) 上取得 SOTA, 59 FPS 推理速度.

**[UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)**

:   提出 UniMMAD，首个用单一参数集同时处理多模态、多类别异常检测的统一框架，核心是基于 MoE 的特征解压缩机制，将通用多模态编码特征自适应分解为领域特定的单模态重建，在 9 个数据集（3 个领域、12 种模态、66 个类别）上达到 SOTA。

**[V2Drop: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models](v2drop_variation_aware_token_dropping.md)**

:   首次从token变化量视角出发，发现LLM层间变化小的"懒惰"视觉token对输出影响可忽略，提出V2Drop渐进式剪除低变化token，在图像理解上保留94.0%性能同时减少31.5%生成延迟，视频理解上保留98.6%性能减少74.2%延迟，且完全兼容FlashAttention。

**[Variation-Aware Vision Token Dropping for Faster Large Vision-Language Models](variation-aware_vision_token_dropping_for_faster_large_vision-language_models.md)**

:   提出 V2Drop，首次从 token 变化量（variation）视角出发，通过渐进式丢弃 LLM 内部变化量最小的"懒惰"视觉 token，实现无训练、无位置偏差、兼容高效算子的 LVLM 推理加速，在图像和视频理解任务中分别保留 94.0% 和 98.6% 原始性能，同时降低 LLM 生成延迟 31.5% 和 74.2%。

**[VecGlypher: Unified Vector Glyph Generation with Language Models](vecglypher_unified_vector_glyph_generation_with_language_models.md)**

:   提出VecGlypher——首个统一文本和图像引导的矢量字形生成语言模型，通过两阶段训练(大规模SVG语法学习+专家标注对齐)直接自回归生成可编辑SVG路径，无需光栅中间步骤或向量化后处理。

**[Venus: Benchmarking and Empowering Multimodal Large Language Models for Aesthetic Guidance and Cropping](venus_benchmarking_and_empowering_multimodal_large_language_models_for_aesthetic.md)**

:   定义审美指导(AG)新任务并构建AesGuide基准(10748张照片含审美评分、分析和指导标注)，提出Venus两阶段框架——先通过渐进式审美问答赋能MLLM审美指导能力，再通过CoT推理激活审美裁剪能力，在两个任务上均达到SOTA。

**[VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)**

:   提出VGGDrive框架，通过冻结的3D视觉基础模型VGGT为VLM注入跨视图几何感知能力，设计插拔式CVGE模块分层自适应地将3D特征注入VLM各层的2D视觉嵌入中，在五个自动驾驶基准上实现显著性能提升。

**[Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)**

:   提出VisionToM，一个基于视觉的轻量级干预框架，通过探测和干预MLLM中对视觉输入和ToM推理敏感的注意力头，在不微调模型的情况下显著增强多模态大语言模型的心智理论推理能力，在EgoToM基准上大幅提升表现。

**[VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatio-temporal_collaborative_network_for_multi-modal_video_fusion.md)**

:   提出首个大规模红外-可见光视频融合框架 VideoFusion，通过跨模态差分增强、完整模态引导融合和双向时序协同注意力机制，联合建模跨模态互补性与时序动态，生成时空一致的高质量融合视频，并构建了包含220个视频/15.4万帧的 M3SVD 数据集。

**[VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatiotemporal_collaborative_network.md)**

:   构建M3SVD大规模红外-可见光视频数据集（220视频/15万帧），并提出VideoFusion框架，通过跨模态差分强化模块(CmDRM)+完整模态引导融合(CMGF)+双向时序共注意力(BiCAM)+变分一致性损失，实现时空协同的多模态视频融合，在融合质量和时序一致性上超越现有图像融合和视频融合方法。

**[ViRC: Enhancing Visual Interleaved Mathematical CoT with Reason Chunking](virc_enhancing_visual_interleaved_mathematical_cot_with_reason_chunking.md)**

:   ViRC 提出 Reason Chunking 机制，将多模态数学 CoT 结构化为连续的"关键推理单元（CRU）"，模拟人类专家反复审视图像并逐步证明中间命题的过程，通过 CRUX 数据集和渐进式训练策略（Instructional SFT → Practice SFT → Strategic RL），实现ViRC-7B 在数学基准上平均提升 18.8%。

**[Vision-Language Models Encode Clinical Guidelines for Concept-Based Medical Reasoning](vision-language_models_encode_clinical_guidelines_for_concept-based_medical_reas.md)**

:   提出MedCBR框架，通过将临床诊断指南（如BI-RADS）融入概念瓶颈模型的训练和推理过程，利用LVLM生成指南一致性报告增强概念监督，结合多任务CLIP训练和大推理模型生成结构化临床解释，在超声和乳腺X光癌症检测上达到94.2%和84.0%的AUROC。

**[VISion On Request: Enhanced VLLM Efficiency with Sparse, Dynamically Selected, Vision-Language Interactions](vision_on_request_enhanced_vllm_efficiency_with_sparse_dynamically_selected_visi.md)**

:   VISOR 提出了一种区别于视觉 token 压缩的新效率范式——通过稀疏化 LLM 内部视觉-语言交互层（少量交叉注意力 + 动态选择的自注意力层），在保留完整高分辨率视觉 token 的同时实现 8.6-18 倍 FLOPs 节省，尤其在需要细粒度理解的困难任务上大幅超越 token 压缩方法。

**[VL-RouterBench: A Benchmark for Vision-Language Model Routing](vl-routerbench_a_benchmark_for_vision-language_model_routing.md)**

:   提出VL-RouterBench，首个面向视觉-语言模型的系统性路由基准，涵盖14个数据集、17个候选模型和519,180个样本-模型对，评估10种路由方法，并发现当前最优路由器与理想Oracle之间仍存在显著差距。

**[VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models](vlm-loc_localization_in_point_cloud_maps_via_vision-language_models.md)**

:   提出VLM-Loc框架，将3D点云地图转换为BEV图像和场景图供VLM进行结构化空间推理，结合部分节点分配（PNA）机制实现文本-点云精细定位，在自建的CityLoc基准上以Recall@5m提升14.20%大幅超越先前SOTA。

**[VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)**

:   提出VLM-Pruner，一种免训练的离心式token剪枝方法，通过空间稀疏缓冲（BSS）准则平衡冗余消除与局部细节完整性，在88.9%剪枝率下跨5个VLM一致超越现有方法，同时实现端到端推理加速。

**[Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks](vlm_model_inversion_adaptive_token_weight.md)**

:   首次系统研究 VLM 的模型反转（Model Inversion）攻击，提出一套面向 token 生成特性的反转策略（TMI/TMI-C/SMI），以及基于视觉注意力强度动态加权 token 梯度贡献的 SMI-AW 方法，在 4 种 VLM 和 3 个数据集上实现最高 61.21% 的人类评估攻击准确率，揭示了 VLM 严重的训练数据隐私泄露风险。

**[WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs](weavetime_stream_from_earlier_frames_into_emergent_memory_in_videollms.md)**

:   诊断了当前 Video-LLM 存在的"时间不可知"（Time-Agnosticism）问题，提出 WeaveTime 框架，通过训练时的时序重建辅助任务（SOPE）赋予模型时序感知能力，推理时用不确定性门控的粗到细记忆缓存（PCDF-Cache）实现高效自适应记忆检索，在流式视频 QA 上取得显著提升。

**[WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs](weavetime_streaming_video_llm_memory.md)**

:   诊断了当前 Video-LLM 存在的"时间不可知"（Time-Agnosticism）问题，提出 WeaveTime 框架，通过训练时的时序重建辅助任务（SOPE）赋予模型时序感知能力，推理时用不确定性门控的粗到细记忆缓存（PCDF-Cache）实现高效自适应记忆检索，在流式视频 QA 上取得显著提升。

**[What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)**

:   提出EmbedLens探针工具系统分析MLLM中视觉token的内部结构，发现视觉token分为sink/dead/alive三类（约40%为无用token），alive token已在进入LLM前编码丰富语义（"预语言"特性），且LLM内部视觉计算对大多数任务冗余，直接中层注入即可。

**[When to Think and When to Look: Uncertainty-Guided Lookback](when_to_think_and_when_to_look_uncertainty-guided_lookback.md)**

:   本文首次系统分析了 LVLM 中 test-time thinking 对视觉推理的影响，发现"多想不如多看"——长推理链常忽略图像导致"long-wrong"轨迹，并据此提出不确定性引导的 lookback 解码策略，通过在推理链漂移时注入视觉回看提示，在不修改模型的前提下将 MMMU 等 6 个基准提升 2-6 个点。

**[When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs](when_token_pruning_is_worse_than_random_understanding_visual_token_information_i.md)**

:   发现VLLM深层中现有token剪枝方法不如随机剪枝的现象，提出基于输出概率变化量化视觉token信息的方法，揭示了"信息地平线"——视觉token信息在某层均匀消散至零的临界层，其位置受任务视觉复杂度和模型能力动态影响，并证明简单集成随机剪枝能有效提升现有方法。

**[Which Concepts to Forget and How to Refuse? Decomposing Concepts for Continual Unlearning in Large Vision-Language Models](which_concepts_to_forget_and_how_to_refuse_decomposing_concepts_for_continual_un.md)**

:   本文提出CORE(COncept-aware REfuser)，一个面向大视觉语言模型(LVLM)持续遗忘的框架：通过将待删除的视觉-语言对分解为细粒度的视觉属性和文本意图概念，使用概念调制器识别需要拒绝的概念组合，再通过混合拒绝专家(refusers)生成概念对齐的拒绝回复，在16个连续遗忘任务上实现了90.67% CRR和88.02% AR的最佳遗忘-保留权衡。

**[Widget2Code: From Visual Widgets to UI Code via Multimodal LLMs](widget2code_from_visual_widgets_to_ui_code_via_multimodal_llms.md)**

:   首次形式化 Widget-to-Code 任务，构建了首个纯图像 widget 数据集和多维评估体系，提出基于感知代理和 WidgetFactory 基础设施的模块化基线，通过组件分解、图标检索、可复用可视化模板和自适应渲染实现高保真 widget 重建。

**[Zina: Multimodal Fine-grained Hallucination Detection and Editing](zina_multimodal_fine-grained_hallucination_detection_and_editing.md)**

:   Zina 提出了多模态细粒度幻觉检测与编辑任务，设计了两阶段系统（detector MLLM + reviewer MLLM）将 token 复制委托给确定性函数以简化模型负担，同时构建了 VisionHall 数据集（6.9K 人工标注 + 20K 图结构合成数据），在检测 F1 上超过 GPT-4o 达 15.8 个点。
