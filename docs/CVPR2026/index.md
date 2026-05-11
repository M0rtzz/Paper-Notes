---
title: >-
  CVPR2026 1648篇论文解读
description: >-
  1648篇CVPR2026论文解读，涵盖多模态 VLM(241篇)、3D 视觉(230篇)、图像生成(208篇)、医学图像(114篇)、自动驾驶(88篇)、语义分割(85篇)、视频理解(77篇)、人体理解(56篇)等 42个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📷 CVPR2026 论文笔记

1648篇CVPR2026论文解读，涵盖多模态 VLM(241篇)、3D 视觉(230篇)、图像生成(208篇)、医学图像(114篇)、自动驾驶(88篇)、语义分割(85篇)、视频理解(77篇)、人体理解(56篇)等 42个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。

<div class="conf-index" markdown>

---

## 🧩 多模态 VLM { #multimodal_vlm }

**[A3: Towards Advertising Aesthetic Assessment](multimodal_vlm/a3_towards_advertising_aesthetic_assessment.md)**

:   提出A3框架，包含理论驱动的三阶段广告美学评估范式A3-Law（感知注意力→形式兴趣→欲望影响）、12万条标注数据集A3-Dataset、经SFT+GRPO对齐的模型A3-Align以及评测基准A3-Bench，在广告美学自动评估上超越现有MLLM。

**[A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](multimodal_vlm/a_closedform_solution_for_debiasing_visionlanguage.md)**

:   提出VLM去偏的闭式解方法，通过在跨模态嵌入空间中对属性子空间做正交分解并利用Chebyshev标量化求解，实现Pareto最优公平性与有界效用损失，免训练、免标注，统一覆盖零样本分类、文本-图像检索和文本-图像生成三大下游任务。

**[Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models](multimodal_vlm/activation_matters_test-time_activated_negative_labels_for_ood_detection_with_vi.md)**

:   提出 TANL（Test-time Activated Negative Labels），通过在测试时动态评估负标签在OOD样本上的"激活程度"来挖掘最有效的负标签，配合激活感知评分函数，在 ImageNet 基准上将 FPR95 从 17.5% 大幅降至 9.8%，且完全免训练、测试高效。

**[AVR: Adaptive VLM Routing for Computer Use Agents](multimodal_vlm/adaptive_vision-language_model_routing_for_computer_use_agents.md)**

:   提出 AVR 自适应路由框架，通过轻量多模态嵌入模型评估动作难度 + 小模型 logprob 置信度探测 + warm agent 记忆注入，实现三层路由（简单→小模型，困难→大模型，高风险→大模型+guardrail），在推理成本降低 78% 的同时仅损失 2pp 准确率。

**[AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition](multimodal_vlm/adaptvision_efficient_vision-language_models_via_adaptive_visual_acquisition.md)**

:   提出 AdaptVision，通过由粗到精的主动视觉机制和强化学习训练，让 VLM 自主决定每个样本所需的最少视觉 token 数量，配合解耦式多轮策略优化 (DTPO) 实现效率与精度的最优平衡。

**[AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](multimodal_vlm/agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)**

:   AGFT 提出了一种对齐引导的微调框架，通过文本引导的对抗训练和分布一致性校准，在增强 VLM 零样本对抗鲁棒性的同时保持预训练的跨模态语义结构，在 15 个零样本基准上平均鲁棒准确率达到 46.57%，超越 SOTA 3.1 个百分点。

**[Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](multimodal_vlm/aif_adaptive_information_flow_vlm.md)**

:   本文发现 VLM 中文本 token 对无关视觉 token 的过度注意力是"看到但感知错误"的根本原因，提出基于 token 动态熵的自适应信息流调控方法（AIF），通过推理时修改因果掩码来阻断无关视觉-文本连接，免训练提升多种 VLM 的感知能力。

**[AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](multimodal_vlm/anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)**

:   AnomalyVFM 提出了一个通用框架，通过三阶段合成数据生成方案和参数高效的 LoRA 适配机制，将任意视觉基础模型（VFM）转化为强零样本异常检测器，以 RADIO 为骨干在 9 个工业数据集上达到 94.1% 图像级 AUROC，超越 SOTA 3.3 个百分点。

**[ApET: Approximation-Error Guided Token Compression for Efficient VLMs](multimodal_vlm/apet_approximation-error_guided_token_compression_for_efficient_vlms.md)**

:   从信息论角度提出基于线性近似重建误差的视觉 token 重要性评估方法，不依赖 attention 权重，天然兼容 FlashAttention，在 LLaVA-1.5 上压缩 88.9% 视觉 token 仍保持 95.2% 性能。

**[Asking like Socrates: Socrates helps VLMs understand remote sensing images](multimodal_vlm/asking_like_socrates_socrates_helps_vlms_understand_remote_sensing_images.md)**

:   揭示遥感VLM中的"伪推理"现象（显式推理链反而导致性能下降），归因于"一瞥效应"（单次粗浅感知不足），提出RS-EoT(Evidence-of-Thought)迭代证据搜索范式，通过SocraticAgent自博弈合成推理轨迹做SFT冷启动，再用两阶段渐进RL（grounding→VQA）增强和泛化，RS-EoT-7B在多个遥感VQA和grounding基准上达SOTA。

**[AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention](multimodal_vlm/ava_vla_improving_vision_language_action_models_with_active_visual_attention.md)**

:   从POMDP视角重新审视VLA模型的视觉处理，提出AVA-VLA框架通过循环状态和主动视觉注意力模块，根据历史上下文动态调制当前帧的视觉token重要性，在LIBERO和CALVIN等基准上达到SOTA。

**[BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates](multimodal_vlm/balm_a_model-agnostic_framework_for_balanced_multimodal_learning_under_imbalance.md)**

:   BALM 提出一个模型无关的即插即用框架来解决**不均衡缺失率（IMR）**下的多模态学习问题，通过特征校准模块（FCM）对齐不同缺失模式下的表征、以及梯度再平衡模块（GRM）从分布和空间两个维度平衡各模态的优化动态，在多个多模态情感识别基准上持续提升各类骨干网络的鲁棒性。

**[Benchmarking Vision-Language Models under Contradictory Virtual Content Attacks in Augmented Reality](multimodal_vlm/benchmarking_vision-language_models_under_contradictory_virtual_content_attacks_.md)**

:   构建首个 AR 环境下矛盾虚拟内容攻击基准 ContrAR（312 个真实 Meta Quest 3 录制视频，10 名标注者验证，平均 Likert 4.66/5），系统评估 11 个 VLM（含 GPT-5/Gemini-2.5/Grok-4）的语义矛盾检测能力，发现 GPT-5 准确率最高（88.14%）但延迟 19s，GPT-4o 在准确率-延迟平衡最佳（84.62%/7.26s），OCR 纯文本基线仅 56%，证明视觉推理不可或缺。

**[Beyond Heuristic Prompting: A Concept-Guided Bayesian Framework for Zero-Shot Image Recognition](multimodal_vlm/beyond_heuristic_prompting_a_concept-guided_bayesian_framework_for_zero-shot_ima.md)**

:   将 VLM 零样本图像识别重构为贝叶斯框架，通过 LLM 驱动的多阶段概念合成流水线构建概念提案分布，并用自适应 soft-trim 似然函数抑制离群概念影响，在 11 个分类基准上优于 SOTA 方法。

**[Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models](multimodal_vlm/beyond_recognition_evaluating_visual_perspective_taking_in_vision_language_model.md)**

:   通过心理学启发的受控LEGO场景构建Isle-Brick-V2基准，系统揭示当前VLM在视觉透视能力(VPT)上的显著不足——即使场景理解近乎完美，空间推理和透视能力仍大幅退化，且存在顽固的方向偏置。

**[Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models](multimodal_vlm/beyond_static_artifacts_a_forensic_benchmark_for_video_deepfake_reasoning_in_vis.md)**

:   提出 FAQ（Forensic Answer-Questioning），首个关注深度伪造视频时序不一致性的大规模多选问答基准（33K QA 对、~4500 视频），通过三层级任务体系（面部感知→时序定位→取证推理）渐进式增强 VLM 取证能力，微调后在域内基准和跨数据集检测中均取得显著提升（Qwen2.5-VL 平均准确率从 21.6% 提升至 52.4%）。

**[Beyond the Mean: Modelling Annotation Distributions in Continuous Affect Prediction](multimodal_vlm/beyond_the_mean_modelling_annotation_distributions_in_continuous_affect_predicti.md)**

:   提出基于Beta分布的情感标注共识建模框架，模型仅预测标注分布的均值和标准差，即可通过矩匹配闭式推导出偏度、峰度、分位数等高阶描述子，在SEWA和RECOLA上证明Beta分布能有效捕获标注者分歧的完整分布特性。

**[BiCLIP: Domain Canonicalization via Structured Geometric Transformation](multimodal_vlm/biclip_domain_canonicalization_via_structured_geometric_transformation.md)**

:   提出 BiCLIP，一个极简的 CLIP 少样本适配方法，通过一个上三角结构约束的双线性变换矩阵对图像特征进行几何对齐，在 11 个标准基准上以极低参数量达到 SOTA。

**[BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment](multimodal_vlm/brima_bridged_modality_adaptation_for_multi-modal_continual_action_quality_asses.md)**

:   提出 BriMA，通过记忆引导的桥接补全和模态感知回放机制，解决多模态持续动作质量评估中非平稳模态不平衡问题，在三个基准上平均提升 6-8% 相关系数、降低 12-15% 误差。

**[BUSSARD: Normalizing Flows for Bijective Universal Scene-Specific Anomalous Relationship Detection](multimodal_vlm/bussard_normalizing_flows_for_bijective_universal_scene-specific_anomalous_relat.md)**

:   提出 BUSSARD，首个基于学习的场景特定异常关系检测方法，利用预训练语言模型嵌入场景图三元组 + 自编码器降维 + 标准化流进行似然估计，在 SARD 数据集上 AUROC 提升约 10%，且对同义词变化鲁棒。

**[Can Vision-Language Models Count? A Synthetic Benchmark and Analysis of Attention-Based Interventions](multimodal_vlm/can_vision-language_models_count_a_synthetic_benchmark_and_analysis_of_attention.md)**

:   构建了一个合成计数基准数据集，系统评估了开源 VLM 在不同图像/提示条件下的计数能力，并通过解码器层面的视觉注意力重加权实验探索改善计数行为的机制。

**[CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment](multimodal_vlm/capt_confusion-aware_prompt_tuning_for_reducing_vision-language_misalignment.md)**

:   提出 CAPT 混淆感知 prompt tuning 框架，通过语义混淆挖掘器（SEM）和样本混淆挖掘器（SAM）显式建模 VLM 的系统性误对齐模式，配合多粒度差异专家（MGDE）融合不同层次的混淆信息，在 11 个基准上取得 HM 83.90% 的最优表现。

**[ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding](multimodal_vlm/chartnet_a_million-scale_high-quality_multimodal_dataset_for_robust_chart_unders.md)**

:   发布 ChartNet——150 万规模的高质量多模态图表数据集，通过代码引导合成管线生成包含图像-代码-数据表-文本-推理QA 的对齐五元组，在图表理解和推理任务上显著提升 VLM 性能，小模型微调后超越 GPT-4o。

**[Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking](multimodal_vlm/circuit_tracing_in_vision-language_models_understanding_the_internal_mechanisms_.md)**

:   提出首个面向 VLM 的电路追踪框架，在 Gemma-3-4B 中训练 per-layer transcoder 并构建归因图，揭示了多模态推理的层次化整合机制、视觉数学电路和六指幻觉的内部成因，并通过 steering 和 circuit patching 验证电路的因果可控性。

**[CLIP-Free, Label-Free, Unsupervised Concept Bottleneck Models](multimodal_vlm/clip-free_label_free_unsupervised_concept_bottleneck_models.md)**

:   提出 TextUnlock 方法将任意冻结视觉分类器的输出分布对齐到视觉-语言对应空间，进而构建无需CLIP、无需标签、无需训练线性探针的全无监督概念瓶颈模型 (U-F²-CBM)，在40+模型上超越有监督CLIP-based CBM。

**[Concept-wise Attention for Fine-grained Concept Bottleneck Models](multimodal_vlm/coat_cbm_concept_wise_attention.md)**

:   CoAt-CBM 通过可学习的概念级视觉 query 和概念对比优化（CCO）实现了自适应细粒度图像-概念对齐，在保持高可解释性的同时超越现有概念瓶颈模型和黑盒模型。

**[CodeDance: A Dynamic Tool-integrated MLLM for Executable Visual Reasoning](multimodal_vlm/codedance_a_dynamic_tool-integrated_mllm_for_executable_visual_reasoning.md)**

:   提出 CodeDance，将可执行代码作为视觉推理的统一媒介，通过 SFT 教授原子能力 + RL 中的难度自适应工具调用奖励（BAT），实现动态工具编排与自检推理，7B 模型在计数/视觉搜索/图表 QA 等任务上超越 GPT-4o。

**[CodePercept: Code-Grounded Visual STEM Perception for MLLMs](multimodal_vlm/codepercept_code-grounded_visual_stem_perception_for_mllms.md)**

:   通过系统性缩放分析发现感知（perception）而非推理（reasoning）是 MLLM 在 STEM 领域的真正瓶颈，提出以可执行 Python 代码为锚定媒介的 CodePercept 范式——构建 100 万级 ICC-1M 数据集和 STEM2Code-Eval 基准，在 SFT+RL 两阶段训练后显著提升 MLLM 的 STEM 视觉感知和下游推理能力。

**[CoMP: Collaborative Multi-Mode Pruning for Vision-Language Models](multimodal_vlm/comp_collaborative_multi-mode_pruning_for_vision-language_models.md)**

:   CoMP 提出协同多模式剪枝框架，通过协同重要性度量（CIM）消除参数和 token 剪枝指标间的不一致性，通过多模式剪枝策略（MPS）自适应选择每阶段的最优剪枝模式，在高剪枝比例下显著优于单模式和简单联合剪枝方案。

**[Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling](multimodal_vlm/conditional_factuality_controlled_llms_with_generalization_certificates_via_conf.md)**

:   提出 CFC（Conditional Factuality Control），一种后处理保形框架，通过增强分位数回归学习特征条件的接受阈值函数，为 LLM 采样输出提供条件覆盖保证（而非仅边际保证），并推导 PAC 风格的有限样本证书 CFC-PAC，在合成数据、推理/QA 基准和 VLM 设置上验证有效性。

**[Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](multimodal_vlm/continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)**

:   提出 SeGP-CL，通过对抗锚点探测旧-新语义边界的脆弱区域，结合锚点引导的跨模态几何蒸馏（ACGD）和文本语义几何正则化（TSGR），在无样本回放条件下有效保持 VLM 的跨模态语义几何结构，显著缓解灾难性遗忘。

**[CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](multimodal_vlm/covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)**

:   发现 MLLM 中视觉编码器微调的"视觉偏好冲突"问题，提出 CoVFT 框架，通过上下文向量提取（CVE）和上下文混合专家（CoMoE）实现上下文感知的视觉微调，在 12 个多模态基准上达到 SOTA 且稳定性显著优于现有方法。

**[CoVR-R: Reason-Aware Composed Video Retrieval](multimodal_vlm/covr-rreason-aware_composed_video_retrieval.md)**

:   CoVR-R 提出了推理优先的零样本组合视频检索框架，利用大型多模态模型（Qwen3-VL）显式推理编辑操作隐含的"后效应"（状态转换、时间阶段、镜头变化等），并构建了包含结构化推理轨迹和困难干扰项的 CoVR-R 基准来评估推理能力，在检索准确率上大幅超越现有方法。

**[CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning](multimodal_vlm/crit_graph-based_automatic_data_synthesis_to_enhance_cross-modal_multi-hop_reaso.md)**

:   提出基于图结构的自动数据生成 pipeline，构建了 CRIT 数据集与 benchmark，用于训练和评测 VLM 在交错图文内容上的跨模态多跳推理能力，训练后的模型在 SPIQA 等多个基准上取得显著提升。

**[CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](multimodal_vlm/cropvlm_learning_to_zoom_for_fine_grained_vision_language_perception.md)**

:   提出CropVLM——一个256M参数的轻量裁剪网络，通过GRPO强化学习训练（无需人工标注边界框），动态选择图像最有信息量的区域供VLM聚焦，可与开源和商用VLM即插即用地提升细粒度视觉理解性能。

**[CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods](multimodal_vlm/crosshoi-bench_a_unified_benchmark_for_hoi_evaluation_across_vision-language_mod.md)**

:   提出 CrossHOI-Bench，首个统一评估 VLM 和 HOI 专用模型的多选题 HOI 基准，通过精心策划的正负例避免不完整标注的错误惩罚，揭示大型 VLM 零样本在 Instance-F1 上超越 SOTA HOI 方法 +5.18%，但在多动作识别和跨人归因上仍存在系统性弱点。

**[Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens](multimodal_vlm/cubic_discrete_diffusion_discrete_visual_generation_on_high-dimensional_represen.md)**

:   提出 CubiD，首个在高维表征 token（768维）上做离散扩散生成的模型，通过在 $h \times w \times d$ 三维张量上进行细粒度 mask 预测实现高质量图像生成，同时保留理解能力。

**[Customized Visual Storytelling with Unified Multimodal LLMs](multimodal_vlm/customized_visual_storytelling_with_unified_multimodal_llms.md)**

:   提出 VstoryGen 框架和核心组件 CustFilmer，基于统一多模态大语言模型（UMLLM）实现多模态故事定制生成，支持文本描述、角色/场景参考图像和镜头类型的联合条件控制，并构建了 MSB 和 M2SB 两个新 benchmark。

**[DC-Merge: Improving Model Merging with Directional Consistency](multimodal_vlm/dc-merge_improving_model_merging_with_directional_consistency.md)**

:   DC-Merge 发现模型合并的关键在于保持合并后多任务向量与原始单任务向量之间**奇异空间方向的一致性**，通过奇异值平滑 + 共享正交子空间投影两步操作，在 Vision 和 Vision-Language 任务上均取得 SOTA 合并效果。

**[DeAR: Fine-Grained VLM Adaptation by Decomposing Attention Head Roles](multimodal_vlm/dear_fine-grained_vlm_adaptation_by_decomposing_attention_head_roles.md)**

:   提出 DeAR，通过 Concept Entropy 指标将 ViT 深层注意力头分解为属性头/泛化头/混合头三类功能角色，并设计基于角色的注意力掩码机制精确控制信息流，在15个数据集上实现任务适配与零样本泛化的最佳平衡。

**[Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](multimodal_vlm/decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)**

:   提出 DASP，通过冗余度评分诊断偏置模态，再用非对称适应策略解耦稳定性与可塑性，解决多模态测试时适应中的负迁移和灾难性遗忘问题。

**[Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification](multimodal_vlm/demographic_fairness_in_multimodal_llms_a_benchmark_of_gender_and_ethnicity_bias.md)**

:   首次系统性地评估了 9 个开源 MLLM 在人脸验证任务上的人口统计公平性，在 IJB-C 和 RFW 两个 benchmark 上使用 4 种 FMR-based 公平性指标衡量性别和族裔偏差，发现 MLLM 的偏见模式与传统人脸识别系统不同。

**[Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models](multimodal_vlm/devil_is_in_narrow_policy_unleashing_exploration_in_driving_vla_models.md)**

:   揭示驾驶 VLA 模型中被忽视的"窄策略"（Narrow Policy）瓶颈——IL 阶段过度利用导致探索坍缩，进而限制 RL 阶段。提出 Curious-VLA 框架，通过可行轨迹扩展 + 多样性感知 RL 在 Navsim 上达到 SOTA（PDMS 90.3，Best-of-N 94.8）。

**[Diagnosing and Repairing Unsafe Channels in Vision-Language Models via Causal Discovery and Dual-Modal Safety Subspace Projection](multimodal_vlm/diagnosing_and_repairing_unsafe_channels_in_vision-language_models_via_causal_di.md)**

:   提出 CARE 框架，先用因果中介分析精确定位 VLM 中与不安全行为因果相关的神经元和层（诊断），再通过广义特征分解构建双模态安全子空间并在推理时投影激活值（修复），将攻击成功率降至 10% 以下且几乎不损失通用能力。

**[Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs](multimodal_vlm/dictionary_aligned_concept_control_for_safeguarding_multimodal_llms.md)**

:   本文提出 DACO 框架，通过从 WordNet 和 CC-3M 构建包含 15,000 个多模态概念的字典，结合稀疏自编码器（SAE）实现对冻结 MLLM 激活空间的细粒度概念控制，在多个安全基准上显著提升安全性的同时保持通用能力。

**[Disentangle-then-Align: Non-Iterative Hybrid Multimodal Image Registration via Cross-Scale Feature Disentanglement](multimodal_vlm/disentangle-then-align_non-iterative_hybrid_multimodal_image_registration_via_cr.md)**

:   提出 HRNet，通过跨尺度特征解纠缠和自适应投影（CDAP）学习干净的共享表示，并在统一的粗到细管线中非迭代地联合预测刚性和非刚性变换，在四个多模态数据集上达到SOTA。

**[Do Vision Language Models Need to Process Image Tokens?](multimodal_vlm/do_vision_language_models_need_to_process_image_tokens.md)**

:   本文系统揭示了VLM中图像token表征在浅层即趋于稳定且跨层可互换，而文本token持续动态重构——图像处理深度的必要性高度依赖输出任务类型。

**[DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](multimodal_vlm/docseeker_long_document_understanding.md)**

:   提出 DocSeeker，通过 ALR（分析-定位-推理）视觉推理范式和两阶段训练（SFT+EviGRPO）实现长文档理解中的结构化推理和证据定位，仅在短文档上训练即可鲁棒泛化到超长文档。

**[Downscaling Intelligence: Exploring Perception and Reasoning Bottlenecks in Small VLMs](multimodal_vlm/downscaling_intelligence_exploring_perception_and_reasoning_bottlenecks_in_small.md)**

:   系统研究LLM缩放对多模态能力的影响，发现视觉任务而非LLM依赖任务受影响最大，且感知退化与推理退化同等严重；提出Extract+Think方法（视觉提取调优+逐步推理），以0.6B感知+1.7B推理的极小模型超越了12倍大的PrismCaptioner和LLaVA-OneVision-0.5B。

**[DSCA: Dynamic Subspace Concept Alignment for Lifelong VLM Editing](multimodal_vlm/dsca_dynamic_subspace_concept_alignment_for_lifelong_vlm_editing.md)**

:   DSCA通过将VLM的表征空间分解为一组正交语义子空间，在每个子空间内进行门控残差干预来实现知识编辑，从而在1000次连续编辑后仍保持>95%的编辑成功率且近乎零遗忘。

**[DSERT-RoLL: Robust Multi-Modal Perception for Diverse Driving Conditions](multimodal_vlm/dsert_roll_robust_multi_modal_perception_for_diverse_driving_conditions.md)**

:   提出 DSERT-RoLL 驾驶数据集，首次集成立体事件相机、RGB、热成像、4D 雷达和双 LiDAR 六种传感器，覆盖多种天气和光照条件，并提出统一多模态 3D 检测融合框架。

**[DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference](multimodal_vlm/duet-vlm_dual_stage_unified_efficient_token_reduction_for_vlm_training_and_infer.md)**

:   提出 DUET-VLM 双阶段视觉 token 压缩框架：第一阶段在视觉编码器内通过 V2V self-attention 选取 dominant tokens 并将剩余 tokens 通过注意力引导局部聚类合并为 contextual tokens；第二阶段在 LLM 内通过 T2V cross-attention 层级裁剪视觉 tokens。在 LLaVA-1.5-7B 上实现 67% token 压缩保持 99%+ 精度、89% 压缩保持 97%+ 精度，训练时间减少 31%。

**[Dynamic Token Reweighting for Robust Vision-Language Models](multimodal_vlm/dynamic_token_reweighting_for_robust_vision-language_models.md)**

:   提出Dtr（Dynamic Token Reweighting），首个通过优化VLM的KV缓存来防御多模态越狱攻击的推理时防御方法，通过定义"反向安全偏移"（RSS）来识别导致安全退化的视觉token，动态调整其权重以恢复模型的安全对齐能力，同时保持良性任务性能。

**[DynamicGTR: Leveraging Graph Topology Representation Preferences to Boost VLM Capabilities on Graph QAs](multimodal_vlm/dynamicgtr_leveraging_graph_topology_representation_preferences_to_boost_vlm_cap.md)**

:   提出 DynamicGTR 框架，通过动态路由在推理时为每个查询选择最优的图拓扑表示（GTR，视觉/文本共8种），显著提升 VLM 在零样本图算法问答中的性能，并可迁移到链接预测和节点分类等真实场景。

**[EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval](multimodal_vlm/eaglenet_energy-aware_fine-grained_relationship_learning_network_for_text-video_.md)**

:   EagleNet 通过构建文本-帧关系图并使用关系图注意力网络学习文本-帧和帧-帧之间的细粒度关系，生成融合视频上下文信息的增强文本嵌入，并引入基于能量模型的匹配机制捕获真实文本-视频对分布，在四个基准数据集上取得 SOTA。

**[EBMC: Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis](multimodal_vlm/ebmc_multimodal_sentiment_analysis.md)**

:   提出 EBMC 两阶段框架，先通过语义解缠和跨模态增强提升弱模态表示质量，再通过能量引导的模态协调和实例感知信任蒸馏实现平衡的多模态情感分析，在缺失模态场景下保持强鲁棒性。

**[Echoes of Ownership: Adversarial-Guided Dual Injection for Copyright Protection in MLLMs](multimodal_vlm/echoes_of_ownership_adversarial-guided_dual_injection_for_copyright_protection_i.md)**

:   提出 AGDI 框架，通过对抗优化生成 trigger image 进行 MLLM 黑盒版权追踪：双注入机制同时在 response 级（CE loss 驱动辅助模型输出 target answer）和 semantic 级（最小化 trigger image 与 target text 的 CLIP 余弦距离）注入版权信息，并引入模型对抗训练模拟 fine-tune 抵抗，在 Qwen2-VL/LLaVA-1.5 上全面超越 PLA 和 RNA 基线。

**[Efficient Document Parsing via Parallel Token Prediction](multimodal_vlm/efficient_document_parsing_via_parallel_token_prediction.md)**

:   提出 PTP（Parallel Token Prediction），一种模型无关的即插即用加速方法，通过在训练序列中插入可学习 register token 实现并行多 token 预测，在 OmniDocBench 上实现 1.6×-2.2× 吞吐提升且不损失精度。

**[EgoMind: Activating Spatial Cognition through Linguistic Reasoning in MLLMs](multimodal_vlm/egomind_activating_spatial_cognition_through_linguistic_reasoning_in_mllms.md)**

:   提出 EgoMind，一种无需几何先验的 CoT 框架，通过角色扮演字幕 (RPC) 和渐进式空间分析 (PSA) 两个核心组件，仅用 5K SFT + 20K RL 样本即可实现多帧空间推理的竞争性能力。

**[EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning in Multimodal Large Language Models](multimodal_vlm/emo-r3_reflective_reinforcement_learning_for_emotional_reasoning_in_multimodal_l.md)**

:   提出 EMO-R3，通过结构化情感思维（SET）引导 MLLM 逐步进行情感推理，并设计反思情感奖励（RER）让模型重新评估推理的视觉-文本一致性和情感连贯性，显著提升多模态情感理解的可解释性和准确性。

**[EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis](multimodal_vlm/emoverse_a_mllms-driven_emotion_representation_dataset_for_interpretable_visual_.md)**

:   构建 EmoVerse——首个同时覆盖 CES（Mikels 8 类离散情感）和 DES（1024 维连续情感空间）的大规模可解释视觉情感数据集（219K+ 图像），提出 B-A-S（Background-Attribute-Subject）三元组知识图谱标注体系和 Annotation & Verification Pipeline（Gemini/GPT-4o + EmoViT + CoT Critic Agent），并基于 Qwen2.5-VL-3B 微调实现 1024 维 DES 投射与情感归因解释。

**[Empowering Semantic-Sensitive Underwater Image Enhancement with VLM](multimodal_vlm/empowering_semanticsensitive_underwater_image_enha.md)**

:   提出一种 VLM 驱动的语义敏感学习策略，通过 LLaVA 生成物体描述、BLIP 构建空间语义引导图、cross-attention 与语义对齐损失双重引导 UIE decoder 重建，使增强图像在感知质量和下游检测/分割任务上同时获得显著提升。

**[ENC-Bench: A Benchmark for Evaluating MLLMs in Electronic Navigational Chart Understanding](multimodal_vlm/enc-bench_a_benchmark_for_evaluating_multimodal_large_language_models_in_electro.md)**

:   提出首个面向电子航海图(ENC)理解的专业级基准 ENC-Bench，包含 20,490 样本和三级层次评估体系（感知→空间推理→海事决策），系统评估 10 个 MLLM 后发现最佳模型仅 47.88% 准确率，揭示了通用模型在安全关键专业领域的严重能力缺口。

**[EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards](multimodal_vlm/evolmm_self_evolving_lmm_continuous_rewards.md)**

:   提出 EvoLMM，一个纯无监督的自进化框架：从单一LMM分出Proposer（生成图像相关问题）和Solver（回答问题），通过连续自一致性奖励（替代离散多数投票）形成闭环训练信号，仅使用原始图像（无标注、无外部奖励模型），在8个多模态数学推理基准上获得约2-3%的一致性提升。

**[Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory](multimodal_vlm/evolving_contextual_safety_in_multi-modal_large_language_models_via_inference-ti.md)**

:   提出 MM-SafetyBench++ 基准和 EchoSafe 框架，通过推理时维护自反思记忆库来累积安全洞察，使 MLLM 能够根据上下文区分看起来相似但安全意图不同的场景，无需训练即可提升上下文安全性。

**[EvoPrompt: Evolving Prompt Adaptation for Vision-Language Models](multimodal_vlm/evolving_prompt_adaptation_for_vision-language_models.md)**

:   EvoPrompt 通过轨迹感知的 prompt 进化策略（统一 embedding 投影 + 方向-幅度解耦训练 + 特征几何正则化）解决 VLM prompt learning 中的灾难性遗忘和模态偏差问题，在 few-shot/跨数据集/域泛化任务上全面 SOTA 且保持 zero-shot 能力。

**[Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration](multimodal_vlm/explore_with_long-term_memory_a_benchmark_and_multimodal_llm-based_reinforcement.md)**

:   本文提出 LMEE 基准和 MemoryExplorer 框架，通过将多目标导航与记忆问答统一评估具身探索的过程与结果，并用强化学习微调 MLLM 使其主动调用记忆检索工具，在 LMEE-Bench 上 SR 达 23.53%（超越 3D-Mem 的 16.91%）、GOAT-Bench 上 SR 达 46.40%。

**[FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](multimodal_vlm/fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)**

:   提出 FairLLaVA，一种参数高效的公平性微调方法，通过最小化隐藏状态与人口学属性之间的互信息来消除多模态大语言模型中的人口学捷径，在胸片报告生成和皮肤病变问答中显著缩小了群体间性能差距。

**[FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Alignment](multimodal_vlm/falcon_false-negative_aware_learning_of_contrastive_negatives_in_vision-language.md)**

:   提出 FALCON，一种**基于学习的 mini-batch 构造策略**，通过负样本挖掘调度器自适应平衡硬负样本与假负样本之间的权衡，显著提升视觉语言预训练的跨模态对齐质量。

**[Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](multimodal_vlm/fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)**

:   提出量化感知积分梯度（QIG），将 LVLM 量化的灵敏度分析从模态级推进到 token 级，利用公理化归因原理精确量化每个 token 对量化误差的贡献，在 W4A8 和 W3A16 设置下显著提升量化模型精度，且几乎无额外计算开销。

**[FINER: MLLMs Hallucinate under Fine-grained Negative Queries](multimodal_vlm/finer_mllms_hallucinate_under_fine-grained_negative_queries.md)**

:   发现 MLLM 在细粒度负查询（涉及多个对象/属性/关系的查询中仅有一个细微错误）下幻觉率急剧上升，提出 FINER 基准和 FINER-Tuning 方法（基于 DPO），在 InternVL3.5-14B 上最高提升 24.2%。

**[FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](multimodal_vlm/flashcache_frequency_kv_cache_compression.md)**

:   提出 FlashCache，首次从频域角度分析多模态 KV Cache 的重要性分布，发现偏离低频主成分的"离群 KV"编码了推理关键特征，通过 DCT 低通滤波识别并优先保留离群 KV + 动态逐层预算分配，在 80% KV 内存压缩下实现 1.69× 解码加速且基本不损失任务性能，天然兼容 FlashAttention。

**[FlowComposer: Composable Flows for Compositional Zero-Shot Learning](multimodal_vlm/flowcomposer_composable_flows_for_compositional_zeroshot_learning.md)**

:   FlowComposer 首次将 Flow Matching 引入组合零样本学习(CZSL)，学习两个原始流(属性流和物体流)将视觉特征传输到对应文本嵌入空间，并通过可学习的 Composer 显式组合速度场得到组合流，同时利用泄露引导增强策略将不完美的特征解耦转化为辅助监督信号，作为即插即用模块在三个基准上持续提升 CZSL 性能。

**[FlowHijack: A Dynamics-Aware Backdoor Attack on Flow-Matching VLA Models](multimodal_vlm/flowhijack_dynamics_aware_backdoor_attack_on_flow_matching_vla_models.md)**

:   FlowHijack是首个系统性针对流匹配VLA模型向量场动态的后门攻击框架，通过τ条件注入策略和动态模仿正则化实现高攻击成功率和行为隐蔽性。

**[FluoCLIP: Stain-Aware Focus Quality Assessment in Fluorescence Microscopy](multimodal_vlm/fluoclip_stain-aware_focus_quality_assessment_in_fluorescence_microscopy.md)**

:   提出 FluoCLIP，一个两阶段视觉-语言框架：先通过染色锚定（stain-grounding）让 CLIP 学习荧光染色的语义，再通过染色引导排序（stain-guided ranking）实现染色感知的对焦质量评估，并引入首个多染色组织级荧光显微镜数据集 FluoMix。

**[PinPoint: Focus, Don't Prune — Identifying Instruction-Relevant Regions for Information-Rich Image Understanding](multimodal_vlm/focus_dont_prune_identifying_instruction-relevant_regions_for_information-rich_i.md)**

:   提出 PinPoint，一个两阶段框架：先通过 Instruction-Region Alignment 定位与指令相关的图像区域，再对选中区域精细化编码，以更少的 visual token 实现更高的 VQA 精度。

**[From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing](multimodal_vlm/from_intuition_to_investigation_a_tool-augmented_reasoning_mllm_framework_for_ge.md)**

:   提出 TAR-FAS 框架，首次将人脸反欺骗（FAS）任务重构为 Chain-of-Thought with Visual Tools（CoT-VT）范式，让 MLLM 在推理过程中自适应调用外部视觉工具（LBP/FFT/HOG等），从"直觉判断"升级为"精细调查"，在 1-to-11 跨域协议上取得 SOTA。

**[From Masks to Pixels and Meaning: A New Taxonomy, Benchmark, and Metrics for VLM Image Tampering](multimodal_vlm/from_masks_to_pixels_and_meaning_a_new_taxonomy_benchmark_and_metrics_for_vlm_im.md)**

:   本文指出现有图像篡改检测基准依赖粗糙的mask标注与真实编辑信号严重不对齐,提出 PIXAR——一个包含 420K+ 图像对的像素级、语义感知篡改检测基准,配合新的训练框架和评估指标,在精确定位和语义理解方面大幅超越现有方法。

**[From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings](multimodal_vlm/from_observation_to_action_latent_action-based_primitive_segmentation_for_vla_pr.md)**

:   提出 LAPS（Latent Action-based Primitive Segmentation）流水线，通过在潜在动作空间中定义"Latent Action Energy"指标，从未标注的工业视频流中无监督发现和分割语义动作原语，为 VLA 模型预训练提供结构化数据。

**[G-MIXER: Geodesic Mixup-based Implicit Semantic Expansion and Explicit Semantic Re-ranking for Zero-Shot Composed Image Retrieval](multimodal_vlm/g_mixer_geodesic_mixup_based_implicit_semantic_expansion_for_zero_shot_cir.md)**

:   提出 G-MIXER，通过测地线混合隐式语义扩展（在球面上沿不同混合比例扩展检索范围）和显式语义重排序（利用 MLLM 生成的属性过滤噪声候选），实现免训练零样本组合图像检索的 SOTA 性能。

**[Generate, Analyze, and Refine: Training-Free Sound Source Localization via MLLM Meta-Reasoning](multimodal_vlm/generate_analyze_and_refine_training-free_sound_source_localization_via_mllm_met.md)**

:   本文提出了一个无需训练的声源定位框架 GAR-SSL，通过将声源定位重新建模为"生成-分析-精炼"的三阶段元认知推理过程，直接利用多模态大语言模型 (MLLM) 的内在推理能力进行音视频定位，在单源和多源定位基准上取得了与训练方法可比甚至更优的性能。

**[GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning](multimodal_vlm/graphvlm_benchmark_vlm_graph_learning.md)**

:   提出 GraphVLM benchmark，系统评估VLM在多模态图学习中的三种角色——VLM-as-Encoder（增强GNN特征）、VLM-as-Aligner（桥接模态用于LLM推理）、VLM-as-Predictor（直接作为图学习backbone）。在6个数据集上的实验表明，VLM-as-Predictor持续取得最佳性能，揭示了VLM作为多模态图学习新基础的巨大潜力。

**[GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding](multimodal_vlm/groundvts_visual_token_sampling_in_multimodal_large_language_models_for_video_te.md)**

:   提出 GroundVTS，一种在视频大语言模型中进行查询引导的细粒度视觉token采样架构，通过在 token 级别自适应保留与查询相关的时空信息，在 Charades-STA 上 mIoU 提升 18.4 点，QVHighlights 上 mAP 提升 20.6 点。

**[GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](multimodal_vlm/gtr_turbo_merged_checkpoint_free_teacher.md)**

:   本文提出 GTR-Turbo，通过将 RL 训练过程中的历史 checkpoint 经 TIES 合并产生"免费教师模型"来指导后续训练（可选 SFT 或 KL 蒸馏方式），在多个视觉智能体任务上匹配甚至超过依赖 GPT-4o 等外部教师的 GTR 方法，同时减少 50% 训练时间和 60% 计算成本。

**[GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks](multimodal_vlm/guide_a_benchmark_for_understanding_and_assisting_users_in_open-ended_gui_tasks.md)**

:   本文提出 GUIDE 基准，包含 120 个新手用户在 10 款软件上的 67.5 小时屏幕录像和出声思维标注，定义了行为状态检测、意图预测、协助预测三个分层任务，评估发现当前最强多模态模型在理解用户行为和判断协助需求上表现有限（行为检测仅 44.6% 准确率），但提供结构化用户上下文可显著提升性能（协助预测最高提升 50.2pp）。

**[HAMMER: Harnessing MLLM via Cross-Modal Integration for Intention-Driven 3D Affordance Grounding](multimodal_vlm/hammer_harnessing_mllm_via_cross-modal_integration_for_intention-driven_3d_affor.md)**

:   提出 HAMMER 框架，通过从 MLLM 中提取接触感知的意图嵌入、层次化跨模态融合增强点云特征、以及多粒度几何提升模块为意图嵌入注入3D空间信息，实现基于交互图像的3D可供性定位，在 PIAD 基准上全面超越现有方法。

**[HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models](multimodal_vlm/handvqa_diagnosing_and_improving_fine-grained_spatial_reasoning_about_hands_in_v.md)**

:   构建了 HandVQA——一个包含 160 万+选择题的大规模诊断性基准，基于 3D 手部关节标注自动生成关于关节角度、距离和相对位置的 VQA 问题，系统暴露了当前 VLM 在细粒度手部空间推理上的严重缺陷，并证明在 HandVQA 上微调后的模型可零样本迁移到手势识别（+10.33%）和手-物交互识别（+2.63%）等下游任务。

**[HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models](multimodal_vlm/hawk_head_importance-aware_visual_token_pruning_in_multimodal_models.md)**

:   提出 HAWK，一种基于注意力头重要性感知的视觉 token 剪枝方法，通过离线计算各注意力头对视觉理解的贡献权重，并结合文本引导的注意力分数动态评估每个视觉 token 的重要性，在 Qwen2.5-VL 上剪枝 80.2% 视觉 token 后仍保留 96.0% 原始性能，同时减少 26% 推理延迟。

**[HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models](multimodal_vlm/hif-vla_hindsight_insight_and_foresight_through_motion_representation_for_vision.md)**

:   提出 HiF-VLA 框架，通过运动向量（Motion Vector）作为紧凑时间原语，统一回顾（Hindsight）、洞察（Insight）和前瞻（Foresight）三种时间推理能力，实现 VLA 模型的双向时间扩展，在长时操作任务中以极低计算开销大幅超越基线。

**[HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](multimodal_vlm/hificl_highfidelity_incontext_learning_for_multimo.md)**

:   HiFICL 通过严格的注意力公式推导，将 ICL 近似问题从"拟合 shift vector"重构为"直接参数化 ICL 的源头"——在注意力头中注入可学习的低秩虚拟键值对，以端到端训练实现一种动态的、上下文感知的参数高效微调方法，在多个多模态基准上以极少参数超越现有 ICL 近似方法和 LoRA。

**[HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](multimodal_vlm/hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)**

:   HiSpatial 提出将 3D 空间智能分解为四层认知层级（几何感知 → 物体属性 → 物体关系 → 抽象推理），构建了处理约 500 万张图像、4500 万个物体、20 亿 QA 对的自动化数据管线，并设计了以度量尺度点云图为辅助输入的 RGB-D VLM，以仅 3B 参数在多个空间推理基准上超越 GPT-5 和 Gemini-2.5-Pro。

**[HIVE: Query, Hypothesize, Verify — An LLM Framework for Multimodal Reasoning-Intensive Retrieval](multimodal_vlm/hive_query_hypothesize_verify_an_llm_framework_for_multimodal_reasoning-intensiv.md)**

:   HIVE 是一个即插即用的多模态检索框架，通过四个阶段——初始检索 → LLM 驱动的补偿性查询合成（显式表达视觉推理缺口）→ 二次检索 → LLM 验证重排序——将推理密集型多模态检索的 nDCG@10 从最佳多模态模型的 27.6 提升至 41.7（+14.1 绝对点），无需任何额外训练。

**[HOG-Layout: Hierarchical 3D Scene Generation, Optimization and Editing via Vision-Language Models](multimodal_vlm/hog_layout_hierarchical_3d_scene_generation_optimization_and_editing.md)**

:   本文提出 HOG-Layout，一个基于 VLM 和 LLM 的层次化 3D 室内场景生成、优化和编辑框架，通过 RAG 增强语义一致性、力导向层次优化确保物理合理性，在 SceneEval 上以 4.5 倍更快的速度超越 LayoutVLM。

**[HoneyBee: Data Recipes for Vision-Language Reasoners](multimodal_vlm/honeybee_data_recipes_for_vision-language_reasoners.md)**

:   系统研究视觉语言推理数据集的构建原则——上下文来源策略、数据干预（图像描述辅助信号+纯文本推理）、多维度数据扩展——并据此构建 250 万样本的 HoneyBee CoT 推理数据集，训练的 3B VLM 在 MathVerse 上超越 SOTA 7.8%，同时提出降低 73% 解码成本的测试时扩展策略。

**[HouseMind: Tokenization Allows MLLMs to Understand, Generate and Edit Architectural Floor Plans](multimodal_vlm/housemind_tokenization_mllm_floor_plan.md)**

:   提出HouseMind框架，通过层次化VQ-VAE将建筑平面图离散化为轮廓token和房间实例token的结构化序列，结合三阶段多模态对齐和指令微调，以Qwen3-0.6B为backbone实现了平面图理解、生成、编辑三项任务的统一建模，几何有效性和可控性大幅超越现有方法。

**[HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](multimodal_vlm/hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)**

:   提出HulluEdit，一种单次前向、无参考模型的子空间编辑框架，通过将隐藏状态分解为正交的视觉证据子空间、冲突先验子空间和残差不确定性子空间，选择性抑制幻觉模式而不干扰视觉定位，在POPE和CHAIR基准上达到SOTA幻觉缓解效果。

**[HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks](multimodal_vlm/humanvbench_probing_human_centric_video_understanding_in_mllms_with_automatica.md)**

:   提出 HumanVBench，一个包含 16 个细粒度任务的人体中心视频理解基准，配套两个自动化流水线（视频标注 + 干扰项感知 QA 合成），对 30 个主流视频 MLLM 的评测揭示了当前模型在细微情感感知和语音-视觉对齐方面的关键不足。

**[IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding](multimodal_vlm/iag_input-aware_backdoor_attack_on_vlm-based_visual_grounding.md)**

:   提出IAG，首个针对VLM视觉定位的多目标后门攻击方法，通过文本条件U-Net动态生成输入感知触发器，将任意指定目标物体的语义信息嵌入视觉输入中，在12种设置下的11种达到最高攻击成功率。

**[Interpretable Debiasing of Vision-Language Models for Social Fairness](multimodal_vlm/interpretable_debiasing_of_vision-language_models_for_social_fairness.md)**

:   提出 DeBiasLens，通过在 VLM 编码器上训练稀疏自编码器（SAE）来定位编码社会属性的"社会神经元"，然后在推理时选择性去激活这些神经元以缓解偏见，在 CLIP 上降低 Max Skew 9-16%，在 InternVL2 上降低性别偏差比例 40-50%，同时保持通用性能。

**[IsoCLIP: Decomposing CLIP Projectors for Efficient Intra-modal Alignment](multimodal_vlm/isoclip_decomposing_clip_projectors_for_efficient_intramodal_alignment.md)**

:   IsoCLIP 从理论上分析 CLIP 投影头的结构，发现余弦相似度计算中隐含一个模态间算子 $\Psi = W_i^\top W_t$ 负责跨模态对齐，和一个模态内算子 $\Psi_i = W_i^\top W_i$ 仅负责归一化但不促进模态内对齐；通过对 $\Psi$ 的奇异值分解识别出近似各向同性(isotropic)的对齐子空间，去除各向异性方向后无需训练即可显著改善模态内检索和分类性能。

**[It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models](multimodal_vlm/its_time_to_get_it_right_improving_analog_clock_reading_and_clock-hand_spatial_r.md)**

:   揭示 SOTA VLM 仍无法可靠读取真实场景中的模拟时钟（零样本准确率不到10%），提出 TickTockVQA 真实场景数据集（12K图像）和 Swap-DPO 微调框架，将 Llama-3.2-11B 的时间读取准确率从1.43%提升至46.22%。

**[Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild](multimodal_vlm/joint-aligned_latent_action_towards_scalable_vla_pretraining_in_the_wild.md)**

:   提出 JALA 框架，通过联合对齐预测嵌入与逆动力学生成的潜在动作，构建统一的潜在动作空间，使 VLA 能同时从标注数据和未标注的野外人类视频中学习，配合 7.5M 样本的 UniHand-Mix 数据集显著提升机器人操作泛化性。

**[KEC: Hierarchical Textual Knowledge for Enhanced Image Clustering](multimodal_vlm/kec_hierarchical_textual_knowledge_clustering.md)**

:   KEC 利用 LLM 构建层级化的概念-属性结构化文本知识来引导图像聚类，在 20 个数据集上无需训练即超越零样本 CLIP 14 个数据集，证明了判别性属性比简单类名更有效。

**[KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing](multimodal_vlm/kvsmooth_mitigating_hallucination_in_multi-modal_large_language_models_through_k.md)**

:   提出KVSmooth，一种免训练的即插即用方法，通过注意力行熵引导的自适应指数移动平均（EMA）对KV-Cache进行平滑，有效抑制多模态大语言模型（MLLM）在解码过程中因sink token引发的语义漂移与幻觉生成，在LLaVA-1.5上将CHAIR_S从41.8降至18.2（降幅56%），同时F1从77.5提升至79.2。

**[Label-Free Cross-Task LoRA Merging with Null-Space Compression](multimodal_vlm/label-free_cross-task_lora_merging_with_null-space_compression.md)**

:   观察到LoRA微调过程中下投影矩阵A的零空间比率随训练下降且与性能强相关，据此提出NSC Merging，一种无标签、任务无关的LoRA合并方法，在20个异构视觉任务、6个NLI任务和VLM评估上达到SOTA。

**[Learning What Matters: Prioritized Concept Learning via Relative Error-driven Sample Selection](multimodal_vlm/learning_what_matters_prioritized_concept_learning_via_relative_error-driven_sam.md)**

:   提出 PROGRESS 框架，通过追踪 VLM 在自动发现的多模态概念集群上的学习进度来动态选择最有信息量的训练样本，仅用 16-20% 的标注数据就达到全数据 99-100% 的性能，且总训练时间更短。

**[LFPC: Learning to Focus and Precise Cropping for MLLMs](multimodal_vlm/lfpc_learning_to_focus_and_precise_cropping_for_mllms.md)**

:   LFPC 提出两阶段纯强化学习框架，通过"信息差"机制（降低全局图像分辨率迫使模型依赖高分辨率裁剪区域）和接地损失（提升裁剪精度），解决了现有 agent-based MLLM 中"先答后裁"的虚假工具调用问题，在高分辨率 VQA 上达到 SOTA。

**[Linking Perception, Confidence and Accuracy in MLLMs](multimodal_vlm/linking_perception_confidence_and_accuracy_in_mllms.md)**

:   揭示 MLLM 的严重置信度失校准问题（视觉输入退化时准确率暴跌但置信度不变），提出 CDRL（基于原始-噪声图像对的置信度驱动 RL）进行感知敏感性训练，并利用校准后的置信度实现自适应测试时缩放（CA-TTS），在四个基准上平均提升 8.8%。

**[LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models](multimodal_vlm/llavashield_multimodal_multiturn_safety.md)**

:   针对VLM多模态多轮对话中的恶意意图隐蔽性、上下文风险累积和跨模态联合风险三大挑战，构建4,484个标注对话的MMDS数据集和基于MCTS的MMRT红队框架，提出LLaVAShield审计模型，在用户/助手两侧分别达到F1 95.71%/92.24%，大幅超越GPT-5-mini等基线。

**[LLMind: Bio-inspired Training-free Adaptive Visual Representations for Vision-Language Models](multimodal_vlm/llmind_bio-inspired_training-free_adaptive_visual_representations_for_vision-lan.md)**

:   受人眼中央凹编码和皮层放大机制启发，提出无需训练的自适应采样框架 LLMind，通过 Möbius 变换实现非均匀像素分配，并利用闭环语义反馈在测试时优化采样参数，在仅使用 1%-5% 像素的紧张预算下大幅超越均匀采样。

**[Locate-then-Sparsify: Attribution Guided Sparse Strategy for Visual Hallucination Mitigation](multimodal_vlm/locate-then-sparsify_attribution_guided_sparse_strategy_for_visual_hallucination.md)**

:   提出 LTS-FS（Locate-Then-Sparsify for Feature Steering）框架，通过因果干预归因方法定位幻觉相关层，并根据归因分数逐层稀疏地控制特征引导强度，在有效缓解 LVLM 幻觉的同时保持模型泛化能力。

**[MA-Bench: Towards Fine-grained Micro-Action Understanding](multimodal_vlm/ma-bench_towards_fine-grained_micro-action_understanding.md)**

:   提出 MA-Bench 微动作理解基准，包含 1000 个视频和 12000 个结构化 QA 对，通过"感知-理解-推理"三层评估架构系统测试 23 个 MLLM 的细粒度微动作理解能力，并构建 20.5K 训练语料 MA-Bench-Train 用于模型微调提升。

**[MarkushGrapher-2: End-to-end Multimodal Recognition of Chemical Structures](multimodal_vlm/markushgrapher-2_end-to-end_multimodal_recognition_of_chemical_structures.md)**

:   MarkushGrapher-2 提出了一个端到端多模态化学结构识别模型，通过专用化学 OCR 模块联合编码图像、文本和布局信息，结合两阶段训练策略（先适配 OCSR 特征再融合多模态编码），在 Markush 结构识别上大幅超越现有方法（M2S 准确率 56% vs 38%），同时保持分子结构识别的竞争力。

**[MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](multimodal_vlm/masquant_modality-aware_smoothing_quantization_for_multimodal_large_language_mod.md)**

:   揭示了通道平滑量化（如 SmoothQuant）直接应用于 MLLM 时的"平滑失配"问题——不同模态激活幅度差异巨大导致非主导模态被过度平滑，提出 MASQuant 通过模态感知平滑因子和基于 SVD 白化的跨模态低秩补偿解决该问题。

**[Mastering Negation: Boosting Grounding Models via Grouped Opposition-Based Learning](multimodal_vlm/mastering_negation_boosting_grounding_models_via_grouped_opposition-based_learni.md)**

:   提出 D-Negation 数据集和 Grouped Opposition-Based Learning (GOBL) 微调机制，通过对立语义配对和两个专用损失函数，仅微调不到 10% 参数即大幅提升视觉定位模型对否定语义的理解能力（最高 +5.7 mAP）。

**[Medic-AD: Towards Medical Vision-Language Model's Clinical Intelligence](multimodal_vlm/medic-ad_towards_medical_vision-language_models_clinical_intelligence.md)**

:   Medic-AD 通过三阶段渐进式训练框架——异常检测（<Ano> token）、时序差异推理（<Diff> token）、可视化解释（热力图），将通用医学 VLM 升级为具备病灶检测、症状追踪和视觉可解释性的临床智能模型，在多项医学任务上达到 SOTA。

**[Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs](multimodal_vlm/mind_the_way_you_select_negative_texts_pursuing_the_distance_consistency_in_ood_.md)**

:   指出现有基于 VLM 的 OOD 检测方法使用模态内距离（文本-文本或图像-图像）选择负文本，与 CLIP 优化的跨模态距离不一致，提出 InterNeg 从文本和视觉两个视角系统地利用跨模态距离，在 ImageNet 上实现 FPR95 降低 3.47%。

**[MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](multimodal_vlm/mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)**

:   MindPower提出以机器人为中心的心智理论（ToM）推理框架，将感知→信念→欲望→意图→决策→行动组织为六层推理层级，并用Mind-Reward（基于GRPO）优化推理一致性，在决策和动作生成上分别超过GPT-4o 12.77%和12.49%。

**[Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](multimodal_vlm/mitigating_multimodal_hallucinations_via_gradient-based_self-reflection.md)**

:   提出 GACD（Gradient-based Influence-Aware Constrained Decoding），利用一阶 Taylor 梯度估计每个 token 对输出的影响力，在推理阶段同时缓解文本-视觉偏差和共现偏差导致的多模态幻觉，无需辅助模型或微调。

**[MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](multimodal_vlm/mmrad_multimodal_anomaly_detection.md)**

:   MMR-AD 构建了当前最大规模的多模态推理型工业异常检测数据集（127K 图像、188 类产品、395 种异常），并提出基于 GRPO 强化学习的 Anomaly-R1 基线模型，显著优于通用 MLLM。

**[MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](multimodal_vlm/modes_accelerating_mixture-of-experts_multimodal_large_language_models_via_dynam.md)**

:   提出 MoDES，首个面向 MoE 多模态大模型的训练免调专家跳过框架，通过全局调制的局部门控（GMLG）和双模态阈值（DMT）机制自适应跳过冗余专家，在跳过 88% 专家时仍保留 97%+ 原始性能，并实现 2.16× prefill 加速。

**[MODIX: Training-Free Multimodal Information-Driven Positional Index Scaling for VLMs](multimodal_vlm/modix_positional_index_scaling.md)**

:   提出 MODIX，一个免训练框架，通过信息论分析（协方差熵+跨模态对齐）动态调整 VLM 中视觉和文本 token 的位置编码步长，将位置粒度分配给信息密集的模态以提升多模态推理。

**[MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](multimodal_vlm/moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)**

:   将 MoE 中的专家选择建模为序列决策问题，通过 GRPO 强化学习优化路由策略，引入模态感知路由引导，在 VLM 的图像和视频理解任务上一致超越确定性 top-K 路由及其变体。

**[More than the Sum: Panorama-Language Models for Adverse Omni-Scenes](multimodal_vlm/more_than_the_sum_panorama-language_models_for_adverse_omni-scenes.md)**

:   提出 Panorama-Language Modeling（PLM）范式和 PanoVQA 大规模全景 VQA 数据集（653K QA 对），设计即插即用的全景稀疏注意力模块让现有 VLM 无需重训练即可处理等距柱状投影全景图，在遮挡和事故等恶劣场景下实现优于多视角拼接方案的全局推理。

**[Mixture of States (MoS): Routing Token-Level Dynamics for Multimodal Generation](multimodal_vlm/mos_mixture_of_states_multimodal_generation.md)**

:   提出Mixture of States (MoS)——一种新的多模态扩散模型融合范式，用可学习的token级路由器将理解塔(冻结LLM/VLM)的任意层hidden state动态路由到生成塔(DiT)的任意层，以3-5B参数在图像生成和编辑上匹配或超越20B的Qwen-Image。

**[Mostly Text, Smart Visuals: Asymmetric Text-Visual Pruning for Large Vision-Language Models](multimodal_vlm/mostly_text_smart_visuals_asymmetric_text-visual_pruning_for_large_vision-langua.md)**

:   通过 MoT 探针实验揭示 LVLM 中文本通路和视觉通路对剪枝的不对称敏感性——文本通路高度敏感必须用文本 token 校准、视觉通路高度冗余可承受 60% 稀疏度，据此提出 ATV-Pruning 使用全部文本 token + 逐层自适应选择的少量视觉 token 构建校准池。

**[MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding](multimodal_vlm/msjoe_jointly_evolving_mllm_and_sampler_for_efficient_long-form_video_understand.md)**

:   提出 MSJoE 框架，将 MLLM 和轻量关键帧采样器通过强化学习联合进化——MLLM 生成视觉查询引导帧检索，1D U-Net 采样器从 CLIP 相似度矩阵中学习选帧，两者端到端联合优化实现长视频问答中 +8% 的准确率提升。

**[Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following](multimodal_vlm/multi-crit_benchmarking_multimodal_judges_on_pluralistic_criteria-following.md)**

:   构建首个评估多模态 Judge 模型多准则遵循能力的基准 Multi-Crit，包含准则级人类标注和偏好冲突样本，配合 PAcc/TOS/CMR 三个新指标，全面评估 25 个 LMM 并揭示闭源最强模型在开放生成任务上仅 32.78% 的多准则一致性。

**[Multi-Modal Image Fusion via Intervention-Stable Feature Learning](multimodal_vlm/multi-modal_image_fusion_via_intervention-stable_feature_learning.md)**

:   提出一个受因果推理启发的多模态图像融合框架，通过三种结构化干预策略（互补掩码、随机掩码、模态丢弃）探测模态间的真实依赖关系，并设计因果特征整合器 (CFI) 学习干预稳定特征，在 MSRS 上 PSNR 达到 66.02、AG 达到 4.129，目标检测 mAP 达到 0.821。

**[Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](multimodal_vlm/multi-modal_representation_learning_via_semi-supervised_rate_reduction_for_gener.md)**

:   提出 SSR²-GCD 框架，通过半监督编码率减少（Semi-Supervised Rate Reduction）损失学习模态内均匀压缩的结构化表征，并结合检索式文本聚合策略增强跨模态知识迁移，在8个数据集上超越现有多模态GCD方法。

**[Multimodal OCR: Parse Anything from Documents](multimodal_vlm/multimodal_ocr_parse_anything_from_documents.md)**

:   提出Multimodal OCR (MOCR)范式，将文档中的文本和图形（图表、图示、UI组件等）统一解析为结构化文本表示（文本+SVG代码），训练3B参数的dots.mocr模型在OCR Arena排名仅次于Gemini 3 Pro，在olmOCR Bench达到83.9 SOTA，在image-to-SVG基准上超越Gemini 3 Pro。

**[MUPO: All Roads Lead to Rome - Incentivizing Divergent Thinking in Vision-Language Models](multimodal_vlm/mupo_all_roads_lead_to_rome_incentivizing_divergent_thinking_in_vlms.md)**

:   MUPO 揭示了 GRPO 训练导致推理多样性坍缩的问题——模型过早收敛到少数推理策略而丢弃大多数替代方案。通过将响应分组进行局部化优势估计并引入多样性奖励，MUPO 激励 VLM 保持发散思维，在多个推理基准上提升 2-7%。

**[Nano-EmoX: Unifying Multimodal Emotional Intelligence from Perception to Empathy](multimodal_vlm/nano-emox_unifying_multimodal_emotional_intelligence_from_perception_to_empathy.md)**

:   Nano-EmoX 提出认知启发的三级情感任务层次（感知→理解→交互），是首个以2.2B紧凑参数统一六项核心情感任务的多模态语言模型，通过P2E渐进式训练框架从基础感知逐步培养到高层共情能力。

**[Narrative Weaver: Towards Controllable Long-Range Visual Consistency with Multi-Modal Conditioning](multimodal_vlm/narrative_weaver_towards_controllable_long-range_visual_consistency_with_multi-m.md)**

:   提出 Narrative Weaver 框架，结合 MLLM 的叙事规划与扩散模型的精细生成，通过可学习查询和动态 Memory Bank 实现多模态条件下的长程视觉一致性生成，并构建首个电商广告视频分镜数据集 EAVSD（330K+ 图像）。

**[No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models](multimodal_vlm/no_hard_negatives_required_concept_centric_learning_leads_to_compositionality_wi.md)**

:   C2LIP 提出不依赖 hard negatives 的对比学习微调方案：通过将文本拆解为名词短语概念并引入跨模态注意力池化，在 SugarCrepe/SugarCrepe++ 组合性基准上达到 SOTA，同时保持甚至提升零样本和检索性能。

**[No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection](multimodal_vlm/no_need_for_real_anomaly_mllm_empowered_zero-shot_video_anomaly_detection.md)**

:   提出端到端零样本视频异常检测框架 LAVIDA，通过异常暴露采样器将语义分割数据集转化为伪异常进行训练，结合 MLLM 提取深层异常语义特征和反注意力 token 压缩处理时空稀疏性，无需任何真实 VAD 数据即实现帧级/像素级 SOTA。

**[Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](multimodal_vlm/noise-aware_few-shot_learning_through_bi-directional_multi-view_prompt_alignment.md)**

:   提出NA-MVP框架，通过双向（clean + noise-aware）多视图prompt设计配合非平衡最优传输（UOT）实现细粒度patch-to-prompt对齐，并用经典OT对识别出的噪声样本做选择性标签修正，在噪声小样本学习场景下持续超越SOTA。

**[OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models](multimodal_vlm/oddgridbench_exposing_the_lack_of_fine-grained_visual_discrepancy_sensitivity_in.md)**

:   提出 OddGridBench 评估 MLLM 的细粒度视觉差异感知能力（找出网格中与其他元素在颜色/大小/旋转/位置上不同的那个），发现所有 MLLM 远低于人类水平，进而提出 OddGrid-GRPO（课程学习 + 距离感知奖励）显著提升模型的视觉辨别力。

**[OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens](multimodal_vlm/omnilottie_generating_vector_animations_via_parameterized_lottie_tokens.md)**

:   OmniLottie 提出一种将 Lottie JSON 文件转化为结构化命令-参数序列的 Lottie Tokenizer，使预训练 VLM 可以基于多模态交叉指令生成高质量矢量动画，并构建了 MMLottie-2M 大规模数据集支撑训练。

**[On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](multimodal_vlm/on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)**

:   揭示了动态 MoE 持续学习中"token 困境"——新任务数据中的模糊和旧 token 对新知识贡献微弱却会导致路由漂移和灾难性遗忘，提出 LLaVA-DyMoE 通过 Token Assignment Guidance 和 Routing Score Regularization 缓解路由漂移，在 CoIN 基准上 MFN 提升超 7%，遗忘降低 12%。

**[Overthinking Causes Hallucination: Tracing Confounder Propagation in Vision Language Models](multimodal_vlm/overthinking_causes_hallucination_tracing_confounder_propagation_in_vision_langu.md)**

:   揭示VLM幻觉的新机制——"过度思考"(overthinking)：模型在中间解码层产生过多竞争性物体假设，混杂因子沿层传播至最终预测引发幻觉；提出Overthinking Score量化层间假设多样性×不确定性，在MSCOCO上F1达78.9%，OOD AMBER上71.58%。

**[PaddleOCR-VL: Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing](multimodal_vlm/paddleocr_vl_coarse_to_fine_document_parsing.md)**

:   PaddleOCR-VL 提出粗到细的文档解析架构：粗阶段用轻量级有效区域聚焦模块(VRFM)定位文档中的有效视觉区域并预测阅读顺序，细阶段用紧凑的0.9B视觉语言模型对裁剪区域进行精细识别，在最少视觉token和参数下实现文档解析SOTA。

**[Parallel In-context Learning for Large Vision Language Models](multimodal_vlm/parallel_in-context_learning_for_large_vision_language_models.md)**

:   提出 Parallel-ICL，将多模态 in-context learning 的长 demonstration 上下文分块并行处理，通过加权 Product-of-Experts 在 logit 层集成，实现与全上下文 MM-ICL 相当甚至更优的性能，同时显著降低推理延迟。

**[PersonaVLM: Long-Term Personalized Multimodal LLMs](multimodal_vlm/personavlm_long_term_personalized_multimodal_llms.md)**

:   本文提出 PersonaVLM，一个面向长期个性化的多模态智能体框架，通过主动记忆管理（四类记忆数据库）、多步推理检索和动量式人格演化机制，将通用 MLLM 转化为能适应用户偏好变化的个性化助手，在 128K 上下文下超越 GPT-4o 5.2%。

**[Phantasia: Context-Adaptive Backdoors in Vision Language Models](multimodal_vlm/phantasia_context-adaptive_backdoors_in_vision_language_models.md)**

:   Phantasia 首次提出上下文自适应的 VLM 后门攻击——攻击者预设一个目标问题，中毒模型在接收到触发图片后不再回答用户原始问题，而是回答攻击者的目标问题，且生成的答案与输入图像语义一致、在语言上自然流畅，从而绕过 STRIP-P 和 ONION-R 等防御；同时本文首次证明了现有 VLM 后门攻击的隐蔽性被严重高估。

**[PhysInOne: Visual Physics Learning and Reasoning in One Suite](multimodal_vlm/physisinone_visual_physics_learning_and_reasoning_in_one_suite.md)**

:   PhysInOne是一个包含153,810个动态3D场景和200万个标注视频的大规模合成数据集，覆盖力学、光学、流体动力学和磁学的71种基本物理现象，为物理感知的世界模型建立了新基准。

**[Pixels Don't Lie (But Your Detector Might): Bootstrapping MLLM-as-a-Judge for Trustworthy Deepfake Detection and Reasoning Supervision](multimodal_vlm/pixels_dont_lie_but_your_detector_might_bootstrapping_mllm-as-a-judge_for_trustw.md)**

:   提出 DeepfakeJudge 框架，通过 bootstrapped generator-evaluator 流程将人类标注的推理监督扩展为大规模结构化评分数据，训练出 3B/7B 视觉语言模型作为 deepfake 检测推理质量的自动评判者，在 pointwise 和 pairwise 评估上均达到与人类高度一致的水平。

**[PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models](multimodal_vlm/pointalign_feature-level_alignment_regularization_for_3d_vision-language_models.md)**

:   提出 PointAlign，在 3D VLM 的 LLM 中间层对点云 token 施加特征级对齐正则化（与 Q-Former 输出对齐），仅训练轻量对齐投影器和 LoRA 适配器，即可有效防止几何信息在语言建模过程中退化，在开放词汇分类上提升 7.50pp。

**[Proof-of-Perception: Certified Tool-Using Multimodal Reasoning with Compositional Conformal Guarantees](multimodal_vlm/pop_proof_of_perception_conformal_reasoning.md)**

:   提出 Proof-of-Perception (PoP)，将多模态推理建模为可执行的有向无环图(DAG)，每个感知/逻辑节点输出带有保形预测证书的集合值（提供逐步可靠性保证），并用轻量控制器基于这些证书在计算预算内自适应分配算力，在文档、图表和多图QA基准上超越CoT、ReAct和PoT基线。

**[Predictive Regularization Against Visual Representation Degradation in Multimodal Large Language Models](multimodal_vlm/predictive_regularization_against_visual_representation_degradation_in_multimoda.md)**

:   本文系统诊断了MLLM中LLM中间层视觉表征在全局功能和patch语义结构两个层面的退化现象，揭示其本质是纯文本生成目标下的"视觉牺牲"，并提出Predictive Regularization (PRe) 通过让退化的中间层特征预测初始视觉特征来缓解退化，在多个VL基准上取得一致提升。

**[Prime Once, then Reprogram Locally: An Efficient Alternative to Black-Box Service Model Adaptation](multimodal_vlm/prime_once_then_reprogram_locally_an_efficient_alternative_to_black-box_service_.md)**

:   本文提出AReS方法，用单次API查询预热本地编码器代替传统零阶优化（ZOO）的持续API调用，在GPT-4o上获得+27.8%提升（ZOO方法几乎无效），同时将API调用量减少99.99%以上，实现了无成本推理。

**[Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving](multimodal_vlm/prune2drive_a_plug-and-play_framework_for_accelerating_vision-language_models_in.md)**

:   首个面向多视角自动驾驶 VLM 的即插即用 token 剪枝框架，通过 T-FPS（token 级最远点采样）保持语义与空间多样性，配合视图自适应剪枝率优化自动分配各摄像头 token 预算，在 DriveLM 上仅保留 10% token 即实现 6.40× prefill 加速且性能仅降 3%。

**[Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher](multimodal_vlm/purify-then-align_towards_robust_human_sensing_under_modality_missing_with_knowl.md)**

:   本文提出PTA（Purify-then-Align）框架，通过元学习驱动的模态加权机制先"净化"噪声多模态教师，再用扩散模型驱动的知识蒸馏"对齐"每个单模态学生，使单模态编码器在模态缺失场景下保持强鲁棒性，在MM-Fi和XRF55上实现SOTA。

**[Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](multimodal_vlm/quant_experts_token_aware_vlm_quantization.md)**

:   提出 Quant Experts (QE)，一种基于 Mixture-of-Experts 的 token 感知自适应量化误差重建框架——将重要通道分为 token-independent（高频出现、全局性）和 token-dependent（低频出现、局部性）两组，分别用共享专家和路由专家的低秩适配器来补偿全局和局部量化误差，在 W4A6 到 W3A16 的多种量化设置下一致提升 VLM 性能。

**[Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](multimodal_vlm/reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)**

:   提出 Reason-SVG 框架，通过"Drawing-with-Thought"(DwT)范式让 LLM 在生成 SVG 之前先进行显式的多阶段设计推理，并结合 SFT + GRPO 强化学习与混合奖励函数进行训练，在语义对齐、结构有效性和视觉质量上全面超越现有方法。

**[ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps](multimodal_vlm/reasonmap_towards_fine-grained_visual_reasoning_from_transit_maps.md)**

:   提出 ReasonMap 基准，利用 30 个城市的高分辨率公交地图构建 1,008 个 QA 对，通过两级评估框架（正确性+质量）系统评估 16 个 MLLM 的细粒度视觉推理能力，发现开源模型中 base 优于 reasoning 而闭源模型相反。

**[ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval](multimodal_vlm/recall_recalibrating_capability_degradation_for_mllm-based_composed_image_retrie.md)**

:   揭示了将生成式MLLM适配为判别式检索器时的"能力退化"现象（Capability Degradation），提出ReCALL框架通过诊断检索器盲点→利用基座MLLM的CoT推理生成纠正性三元组→分组对比精炼三阶段管线，有效恢复退化的细粒度组合推理能力，在CIRR上R@1达55.52%、FashionIQ上R@10达57.04%。

**[Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress](multimodal_vlm/recurrent_reasoning_with_vision-language_models_for_estimating_long-horizon_embo.md)**

:   提出 R²VLM，通过循环推理框架逐步处理本地视频片段，维护动态更新的 CoT 记录任务分解和完成状态，结合多维 RL 奖励实现长时域具身任务进度估计的 SOTA，并支持策略学习、奖励建模、主动辅助等下游应用。

**[Recursive Think-Answer Process for LLMs and VLMs](multimodal_vlm/recursive_think-answer_process_for_llms_and_vlms.md)**

:   R-TAP 提出一种递归思考-回答过程，通过置信度生成器评估模型回答确定性并引导迭代推理修正，配合递归置信度增长奖励和最终答案置信度奖励的双重强化信号，在 LLM 和 VLM 上均一致超越单次推理方法，同时显著减少推理中的"Oops!"式自我反思表达。

**[ReHARK: Refined Hybrid Adaptive RBF Kernels for Robust One-Shot Vision-Language Adaptation](multimodal_vlm/rehark_refined_hybrid_adaptive_rbf_kernels_for_robust_one-shot_vision-language_a.md)**

:   提出 ReHARK 框架，通过混合语义-视觉先验构建、支撑集增强、自适应分布校正和多尺度 RBF 核集成四阶段精炼管道，在 11 个基准上实现 65.83% 的单样本适应 SOTA 准确率，显著超越 Tip-Adapter 和 ProKeR。

**[World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training](multimodal_vlm/rehearsevla_simulated_post-training_for_vlas_with_physically-consistent_world_mo.md)**

:   提出 World-Env 框架，利用物理一致的世界模型作为虚拟环境替代真实交互，对 VLA 模型进行 RL post-training，仅需每任务 5 条示教即可显著提升操控成功率。

**[Relational Visual Similarity](multimodal_vlm/relational_visual_similarity.md)**

:   本文首次形式化定义关系视觉相似度问题（两图像间的内在关系/功能对应，而非表面属性相似），构建114K匿名描述数据集并训练relsim模型，揭示了现有相似度指标（CLIP/DINO等）在捕捉关系相似度方面的根本性缺陷。

**[ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](multimodal_vlm/remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)**

:   提出 ReMoRa，直接操作视频压缩表示（I帧 + 运动向量），通过 Refined Motion Representation (RMR) 模块将粗糙的块级运动向量精化为接近光流的细粒度运动表征，再用 Hierarchical Motion State Space (HMSS) 模块进行线性时间的长程时间建模，在 LongVideoBench、NExT-QA、MLVU 等基准上超越基线。

**[Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](multimodal_vlm/residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)**

:   提出 Residual Decoding (ResDec)——一种训练免的即插即用解码策略，通过分析历史 token 的 logit 分布中的 U 型 JSD 模式发现语义锚定阶段，聚合该阶段的历史 logits 作为残差引导融入当前解码，以近乎零的额外推理开销有效抑制 LVLM 中的语言先验幻觉。

**[Responses Fall Short of Understanding: Revealing the Gap between Internal Representations and Responses in VDU](multimodal_vlm/responses_fall_short_of_understanding_gap_between_internal_representations_and_responses_in_vdu.md)**

:   通过逐层线性探测分析发现 LVLM 在视觉文档理解中存在内部表示与生成响应之间的显著差距，且中间层比最终层编码了更线性可访问的任务信息，微调中间层可同时提升准确率和缩小差距。

**[Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token](multimodal_vlm/rethinking_mllm_itself_as_a_segmenter_with_a_single_segmentation_token.md)**

:   提出 SELF1E，首次实现不依赖专用 mask 解码器且仅用单个 [SEG] token 的 MLLM 分割方法，通过 Residual Features Refilling (RFR) 和 Residual Features Amplifier (RFA) 恢复 pixel-shuffle 压缩造成的分辨率损失，在多个分割任务上达到与解码器方法竞争力相当的性能。

**[Rethinking VLMs for Image Forgery Detection and Localization](multimodal_vlm/rethinking_vlms_for_image_forgery_detection_and_localization.md)**

:   提出 IFDL-VLM 框架，发现 VLM 固有的语义合理性偏向（而非真实性）会阻碍伪造检测性能，因此将检测/定位与语言解释解耦为两阶段优化，并利用定位掩码作为 VLM 的辅助输入增强可解释性，在 9 个基准上全面达到 SOTA。

**[Revisiting Model Stitching in the Foundation Model Era](multimodal_vlm/revisiting_model_stitching_in_the_foundation_model.md)**

:   本文系统研究视觉基础模型（VFM）之间的拼接可行性，发现传统方法在VFM上失效，提出"Final Feature Matching + Task Loss"两阶段训练策略使异构VFM可靠拼接，拼接模型甚至能超越两个单独VFM，进而提出VFM Stitch Tree（VST）架构为多VFM系统提供可控的精度-效率权衡方案。

**[SALMUBench: A Benchmark for Sensitive Association-Level Multimodal Unlearning](multimodal_vlm/salmubench_a_benchmark_for_sensitive_association-level_multimodal_unlearning.md)**

:   提出 SALMUBench——首个针对 CLIP 类模型的关联级别机器遗忘基准，包含 60K 合成人物-敏感属性配对数据集、从头训练的 Compromised/Clean 模型对，以及结构化 holdout 集评估协议，首次系统揭示了现有遗忘方法的三种失败模式（灾难性破坏、过度泛化遗忘、无效遗忘）。

**[Scaling Spatial Intelligence with Multimodal Foundation Models](multimodal_vlm/scaling_spatial_intelligence_with_multimodal_foundation_models.md)**

:   SenseNova-SI 通过系统化构建800万级多样化空间数据（SenseNova-SI-8M），在 Qwen3-VL、InternVL3 和 Bagel 等多模态基础模型上培养空间智能能力，在 VSI-Bench、MMSI 等多个空间基准上取得前所未有的性能，同时保持通用多模态理解能力。

**[Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework](multimodal_vlm/scaling_test-time_robustness_of_vision-language_models_via_self-critical_inferen.md)**

:   提出 Self-Critical Inference (SCI) 框架，通过多轮文本+视觉反事实推理的 logit 聚合来同时解决 LVLM 的语言偏差和语言敏感性问题，并提出 DRBench 动态鲁棒性基准来模型特异地评估鲁棒性。增加反事实推理轮次可持续提升鲁棒性，开辟了测试时缩放的新方向。

**[Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](multimodal_vlm/scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)**

:   提出 FlexMem——一种训练免的视觉记忆机制，通过迭代式双路径 KV 缓存压缩构建视觉记忆库，结合编码式和快速索引式记忆召回策略，让 MLLM 在单张 3090 GPU 上处理 1000+ 帧长视频，大幅超越现有高效视频理解方法。

**[Scene-VLM: Multimodal Video Scene Segmentation via Vision-Language Models](multimodal_vlm/scene-vlm_multimodal_video_scene_segmentation_via_vision-language_models.md)**

:   提出 Scene-VLM——首个基于微调 VLM 的视频场景分割框架，通过结构化多模态镜头表征（视觉帧+对白+元数据）、因果序列预测、上下文-焦点窗口机制和 token logits 置信度提取，在 MovieNet 上取得 +6 AP 和 +13.7 F1 的大幅提升，并展示了自然语言解释能力。

**[SciPostGen: Bridging the Gap between Scientific Papers and Poster Layouts](multimodal_vlm/scipostgen_bridging_the_gap_between_scientific_papers_and_poster_layouts.md)**

:   构建了包含 18,097 个论文-海报对的大规模数据集 SciPostGen，分析发现论文结构与海报布局元素数量存在中等相关性，并提出检索增强海报布局生成框架，通过对比学习检索与论文匹配的布局模板来指导 LLM 生成海报布局。

**[SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker](multimodal_vlm/seatrack_multimodal_tracker.md)**

:   提出 SEATrack 多模态跟踪器，通过 AMG-LoRA 实现跨模态注意力图的动态对齐，以及 HMoE 实现高效全局关系建模的跨模态融合，在 RGB-T/D/E 跟踪中以极少参数实现 SOTA 的性能-效率平衡。

**[See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models](multimodal_vlm/see_hear_and_understand_benchmarking_audiovisual_human_speech_understanding_in_mul.md)**

:   提出 AV-SpeakerBench 基准，包含 3,212 道以说话人为中心的音视频推理多选题，系统评估多模态大语言模型在"谁在说话、说了什么、何时说的"上的细粒度音视频融合能力，揭示当前最强模型与人类表现仍有超 20% 的差距。

**[See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles](multimodal_vlm/see_think_act_teaching_multimodal_agents_to_effectively_interact_with_gui_by_ide.md)**

:   提出 State-aware Reasoning (StaR)，通过教会多模态 Agent "感知当前状态→分析目标状态→决定是否操作"的三步推理链，将 GUI 开关控制准确率提升超 30%，同时不损害通用 Agent 任务性能。

**[Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness](multimodal_vlm/seeing_clearly_reasoning_confidently_plug-and-play_remedies_for_vision_language_.md)**

:   提出一种高效的即插即用模块，通过学习多模态类嵌入来增强 VLM 对稀有物体的识别和推理能力：在视觉端用 cross-attention 适配器精化视觉 token，在文本端注入物体检测提示，无需微调 VLM 即可在 CODA-LM 上获得 72.8→75.4 的显著提升。

**[Seeing Through Touch: Tactile-Driven Visual Localization of Material Regions](multimodal_vlm/seeing_through_touch_tactile_localization.md)**

:   提出触觉定位任务——给定触觉输入识别图像中具有相同材质属性的区域，通过局部视觉-触觉对齐和材质多样性配对策略学习密集跨模态特征，构建两个新的触觉-材质分割数据集。

**[Self-Consistency for LLM-Based Motion Trajectory Generation and Verification](multimodal_vlm/self-consistency_for_llm-based_motion_trajectory_generation_and_verification.md)**

:   将 LLM 的自一致性范式从自然语言推理扩展到视觉域——用 Lie 变换群层次结构定义运动轨迹的形状族，通过在变换不变距离度量下聚类 LLM 采样的多条轨迹，实现无监督的轨迹生成改进（+4-6%）和验证（精度+11.8%），无需训练。

**[Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning](multimodal_vlm/similarity-as-evidence_calibrating_overconfident_vlms_for_interpretable_and_labe.md)**

:   提出 Similarity-as-Evidence (SaE) 框架，将 VLM 的文本-图像相似度重新解释为 Dirichlet 证据，通过 Similarity Evidence Head (SEH) 校准过度自信的 softmax 输出，并基于 vacuity（知识空缺）和 dissonance（证据冲突）的双因子采集策略实现可解释、高效的医学主动学习，在 10 个数据集上以 20% 标注预算达到 82.57% 的 SOTA 宏平均准确率。

**[SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models](multimodal_vlm/simpact_simulation-enabled_action_planning_using_vision-language_models.md)**

:   SIMPACT 提出一种测试时的仿真增强动作规划框架，从单张 RGB-D 图像自动构建物理仿真环境，使 VLM 能够提出动作、观察仿真结果并迭代优化推理，无需额外训练即可在刚体和可变形物体操作任务上达到 SOTA 性能。

**[SoPE: Spherical Coordinate-Based Positional Embedding for 3D LVLMs](multimodal_vlm/sope_spherical_positional_encoding_3d_lvlm.md)**

:   揭示 RoPE 在 3D LVLM 中的空间感知偏差问题（1D 索引破坏 3D 局部性且忽视方向），提出球面坐标位置编码 SoPE（$(t,r,\theta,\phi)$ 四维索引 + 多维频率分配 + 多尺度混合），在 SpatialLM 上实现 3D 布局估计和物体检测 SOTA。

**[SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](multimodal_vlm/sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)**

:   提出 SPARROW 框架，通过 **目标特定追踪特征（TSF）** 注入时间一致性监督、**双提示（[BOX]+[SEG]）粗到细解码** 稳定首帧初始化，以即插即用方式集成到现有视频 MLLM 上，在 6 个基准 3 个任务上取得一致提升。

**[SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](multimodal_vlm/spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)**

:   提出SpatiaLQA基准（9605个QA对、241个真实室内场景），系统评估41个VLM在空间逻辑推理上的表现，并设计递归场景图辅助推理方法来提升VLM的空间逻辑推理能力。

**[SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence](multimodal_vlm/spatialscore_towards_comprehensive_evaluation_for_spatial_intelligence.md)**

:   本文提出了目前最全面的多模态空间智能基准 SpatialScore（5K样本/30任务），并通过数据驱动的 SpatialCorpus（331K QA）微调方案和免训练的 SpatialAgent（12个工具）两条互补路径来提升 MLLM 的空间理解能力。

**[SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning](multimodal_vlm/spatialstack_layered_geometry-language_fusion_for_3d_vlm_spatial_reasoning.md)**

:   提出SpatialStack框架，将多视图几何编码器（VGGT）的多层级几何特征逐层注入LLM解码器的不同层（而非仅融合最后一层），通过浅层→细粒度空间感知、深层→高层语义推理的层级对齐，在多个3D空间推理基准上达到开源SOTA。

**[StructXLIP: Enhancing Vision-Language Models with Multimodal Structural Cues](multimodal_vlm/structxlip_enhancing_vision-language_models_with_multimodal_structural_cues.md)**

:   StructXLIP 将边缘图（edge map）作为视觉结构的代理表示，在 CLIP 微调中引入三种结构中心损失（边缘-结构文本对齐 + 局部区域-文本块匹配 + 边缘-彩色图连接），通过最大化多模态结构表示的互信息引导模型走向更鲁棒的语义稳定最优解，在跨模态检索任务上超越现有竞争者。

**[Taxonomy-Aware Representation Alignment for Hierarchical Visual Recognition with Large Multimodal Models](multimodal_vlm/taxonomy-aware_representation_alignment_for_hierarchical_visual_recognition_with.md)**

:   提出TARA框架，通过将LMM的中间表示与生物基础模型(BFM)的分类学感知特征对齐，为大型多模态模型注入分类层次知识，显著提升已知和新颖类别的层次化视觉识别性能。

**[Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](multimodal_vlm/tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)**

:   提出Vision-Guided Attention (VGA)，一种免训练的方法，通过利用视觉token的语义特征构建精确的视觉定位，引导模型注意力聚焦于相关视觉区域，有效缓解MLLM幻觉，且兼容FlashAttention。

**[Test-Time Attention Purification for Backdoored Large Vision Language Models](multimodal_vlm/test-time_attention_purification_for_backdoored_large_vision_language_models.md)**

:   发现LVLM后门行为的本质是跨模态注意力窃取（trigger视觉token抢夺文本token的注意力），提出CleanSight——首个无需训练的测试时后门防御框架，通过检测和剪枝高注意力trigger token来消除后门效应。

**[Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction](multimodal_vlm/text-only_training_for_image_captioning_with_retrieval_augmentation_and_modality.md)**

:   提出TOMCap——一种纯文本训练的图像描述方法，通过检索增强+模态差距修正+LoRA微调，在训练时只用文本而推理时处理图像，超越了已有的无训练和纯文本方法。

**[The Coherence Trap: When MLLM-Crafted Narratives Exploit Manipulated Visual Contexts](multimodal_vlm/the_coherence_trap_when_mllm-crafted_narratives_exploit_manipulated_visual_conte.md)**

:   揭示现有多模态篡改检测忽视了MLLM能生成语义一致的欺骗性叙事这一核心威胁，构建441k样本的MDSM语义对齐篡改数据集，并提出基于Artifact Token和操纵导向推理的AMD框架，在跨域检测中以仅0.27B参数达到88.18 ACC / 60.25 mAP / 61.02 mIoU的最优泛化性能。

**[The LLM Bottleneck: Why Open-Source Vision LLMs Struggle with Hierarchical Visual Recognition](multimodal_vlm/the_llm_bottleneck_why_open-source_vision_llms_struggle_with_hierarchical_visual.md)**

:   揭示开源LLM缺乏关于视觉世界的层次分类知识（甚至不知道基本的生物分类体系），这使得LLM成为Vision LLM层次视觉识别的瓶颈。

**[The More, the Merrier: Contrastive Fusion for Higher-Order Multimodal Alignment](multimodal_vlm/the_more_the_merrier_contrastive_fusion_for_higher-order_multimodal_alignment.md)**

:   提出Contrastive Fusion (ConFu)框架，将CLIP式的双模态对比学习推广到三模态高阶对齐，在统一目标中同时学习配对和融合表示，支持1→1和2→1检索。

**[Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth](multimodal_vlm/think_360_evaluating_the_width-centric_reasoning_capability_of_mllms_beyond_dept.md)**

:   本文提出 Think360，一个聚焦于"推理宽度"（即模型在多路径搜索、多约束剪枝、回溯试错等方面的能力）的多模态基准，包含 1200+ 高质量样本，并设计细粒度 Tree-of-Thought 评估协议，揭示当前 MLLM 在宽度方向推理上的显著短板。

**[Thinking Diffusion: Penalize and Guide Visual-Grounded Reasoning in Diffusion Multimodal Language Models](multimodal_vlm/thinking_diffusion_penalize_and_guide_visual-grounded_reasoning_in_diffusion_mul.md)**

:   首次定量分析扩散多模态LLM (dMLLM)的CoT推理过程，发现"早期回答生成"和"弱视觉依赖"两个关键问题，提出PSP（位置-步骤惩罚）和VRG（视觉推理引导）两种免训练方法，在3倍加速下获得最高7.5%的精度提升。

**[Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](multimodal_vlm/thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)**

:   提出 Dyn-Bench——一个面向 4D 物理世界动态理解的大规模基准（1k 视频、7k VQA 对、3k 动态 grounding 对），系统评估了通用/空间/区域级 MLLM 的时空推理能力，发现现有模型无法同时维持推理和 grounding 的一致性，并提出 Mask-Guided Fusion 和 ST-TCM 两种结构化集成方法显著提升动态感知。

**[TIGeR: A Unified Framework for Time, Images and Geo-location Retrieval](multimodal_vlm/tiger_a_unified_framework_for_time_images_and_geo-location_retrieval.md)**

:   提出TIGeR框架，通过多模态Transformer联合学习图像-位置-时间的统一地理时间嵌入空间，实现地理定位、拍摄时间预测和地理时间感知图像检索三个任务的统一，并构建了4.5M规模的高质量基准数据集。

**[TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](multimodal_vlm/timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)**

:   系统调查构建MLLM视频时间定位（VTG）能力的关键因素，从数据质量和算法设计两个维度出发，发布高质量基准TimeLens-Bench和训练集TimeLens-100K，并通过交错文本时间编码+thinking-free RLVR训练范式构建TimeLens系列模型，在开源模型中达到SOTA并超越GPT-5和Gemini-2.5-Flash。

**[TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment](multimodal_vlm/tipsv2_patch_text_alignment.md)**

:   提出 TIPSv2，通过发现蒸馏能显著提升 patch-text 对齐能力，并将该洞察转化为新的预训练目标 iBOT++（可见 token 也参与损失计算），结合头部EMA和多粒度文本增强，在 9 个任务 20 个数据集上达到 SOTA。

**[Token Warping Helps MLLMs Look from Nearby Viewpoints](multimodal_vlm/token_warping_helps_mllms_look_from_nearby_viewpoints.md)**

:   提出对 MLLM 的 ViT image token 做空间 warping（而非传统的像素级 warping）来模拟视角变换，发现 backward token warping 在保持语义一致性同时对深度估计噪声鲁棒，在自建的 ViewBench 上大幅超越像素级 warping、专用空间推理 MLLM 和生成式 warping 方法。

**[Topo-R1: Detecting Topological Anomalies via Vision-Language Models](multimodal_vlm/topo-r1_detecting_topological_anomalies_via_vision-language_models.md)**

:   提出Topo-R1——首个赋予VLM拓扑感知能力的框架，通过自动化数据构建管线+SFT+GRPO强化学习（含拓扑感知复合奖励），实现无标注的管状结构拓扑异常检测与分类。

**[Towards Calibrating Prompt Tuning of Vision-Language Models](multimodal_vlm/towards_calibrating_prompt_tuning_of_vision-language_models.md)**

:   针对prompt tuning后CLIP面临的"双重误校准"问题（基类欠自信+新类过自信），提出均值-方差margin正则化和文本矩匹配损失两个互补正则项，作为即插即用模块在7种prompt tuning方法和11个数据集上显著降低ECE。

**[Towards Multimodal Domain Generalization with Few Labels](multimodal_vlm/towards_multimodal_domain_generalization_with_few_labels.md)**

:   定义并研究半监督多模态域泛化(SSMDG)新问题，提出融合一致性驱动伪标签、分歧感知正则化和跨模态原型对齐的统一框架，在少量标注下实现多模态模型的跨域泛化。

**[Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training](multimodal_vlm/towards_real-world_document_parsing_via_realistic_scene_synthesis_and_document-a.md)**

:   提出数据-训练协同设计框架 DocHumming：通过 Realistic Scene Synthesis 构建 DocMix-3M 大规模合成数据集，结合渐进学习和结构 token 加权的 Document-Aware Training Recipe，在仅 1B 参数的 MLLM 上实现 OmniDocBench Overall 93.75（超越 Qwen3-VL-235B 的 89.15），且在真实拍摄场景下仅退化 6.72 分（模块化方法退化 18-20 分）。

**[Training High-Level Schedulers with Execution-Feedback Reinforcement Learning for Long-Horizon GUI Automation](multimodal_vlm/training_high-level_schedulers_with_execution-feedback_reinforcement_learning_fo.md)**

:   提出 CES（Coordinator-Executor-State Tracker）多智能体框架和分阶段执行反馈强化学习算法，将高层任务规划与低层执行解耦，通过专门训练的 Coordinator 和 State Tracker 显著提升 GUI Agent 在长时序任务上的规划和状态管理能力。

**[TreeTeaming: Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration](multimodal_vlm/treeteaming_autonomous_red-teaming_of_vision-language_models_via_hierarchical_s.md)**

:   TreeTeaming 提出了一个基于层次策略树的自动化红队测试框架，通过 LLM 驱动的 Orchestrator 动态地探索和进化攻击策略，在12个主流 VLM 上实现了 SOTA 的攻击成功率（GPT-4o 达 87.60%），并发现了超越已知策略集的多样化新攻击手段。

**[TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition](multimodal_vlm/trivia_self-supervised_fine-tuning_of_vision-language_models_for_table_recogniti.md)**

:   提出 TRivia 自监督微调框架，通过表格问答（QA）驱动的 GRPO 强化学习，让 VLM 直接从无标注表格图像中学习表格识别能力，3B 参数的 TRivia-3B 在多个基准上超越 Gemini 2.5 Pro 和 GPT-5 等私有模型。

**[Unbiased Dynamic Multimodal Fusion](multimodal_vlm/unbiased_dynamic_multimodal_fusion.md)**

:   UDML 提出无偏动态多模态学习框架，包含噪声感知不确定性估计器（通过注入可控噪声并预测其强度来实现在低噪和高噪条件下均准确的模态质量评估）和模态依赖计算器（通过 Dropout 量化模型对各模态的固有依赖偏差并融入加权机制），解决了现有方法的双重抑制问题，在多个多模态基准上一致提升性能。

**[Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models](multimodal_vlm/uncertainty-aware_knowledge_distillation_for_multimodal_large_language_models.md)**

:   提出Beta-KD，一种基于贝叶斯视角的不确定性感知知识蒸馏框架，通过将教师监督建模为Gibbs先验并用Laplace近似推导闭形解，自动调节数据与教师信号的平衡，在多模态VQA基准上持续提升蒸馏效果。

**[Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models](multimodal_vlm/uncertainty-guided_compositional_alignment_with_part-to-whole_semantic_represent.md)**

:   提出UNCHA框架，在双曲VLM中用双曲不确定性建模部分图像对整体场景的语义代表性，通过不确定性引导的对比损失和蒸含损失增强组合性场景理解，在多个下游任务上超趇现有双曲VLM。

**[Understanding Task Transfer in Vision-Language Models](multimodal_vlm/understanding_task_transfer_in_vision-language_models.md)**

:   本文首次系统研究了 VLM 在一个视觉感知任务上微调后对其他感知任务零样本性能的影响，提出 Perfection Gap Factor (PGF) 归一化指标量化跨任务迁移，在 Qwen-2.5-VL 三个尺度模型上揭示了任务迁移的结构性规律（正/负迁移团、任务角色分类、尺度依赖等），并证明 PGF 可指导数据选择提升微调效率。

**[UNICBench: UNIfied Counting Benchmark for MLLM](multimodal_vlm/unicbench_unified_counting_benchmark_for_mllm.md)**

:   推出UNICBench，首个统一的跨模态（图像/文本/音频）多层级计数基准，包含5,508+5,888+2,905共14,301个QA对及三级能力(Pattern/Semantic/Reasoning)×三级难度(Easy/Medium/Hard)分类，系统评估45个SOTA MLLM，揭示基本计数任务趋近但推理级和困难任务存在显著差距。

**[UniGame: Turning a Unified Multimodal Model Into Its Own Adversary](multimodal_vlm/unigame_turning_a_unified_multimodal_model_into_its_own_adversary.md)**

:   UniGame 提出首个针对统一多模态模型（UMM）的自对抗后训练框架，通过在共享视觉 token 接口安装轻量扰动器，让生成分支主动创造语义一致的对抗样本来挑战理解分支，形成极小极大自博弈，显著提升一致性 (+4.6%)、理解 (+3.6%)、生成和鲁棒性。

**[UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](multimodal_vlm/unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)**

:   提出 UniMMAD，首个用单一参数集同时处理多模态、多类别异常检测的统一框架，核心是基于 MoE 的特征解压缩机制，将通用多模态编码特征自适应分解为领域特定的单模态重建，在 9 个数据集（3 个领域、12 种模态、66 个类别）上达到 SOTA。

**[Variation-Aware Vision Token Dropping for Faster Large Vision-Language Models](multimodal_vlm/variation-aware_vision_token_dropping_for_faster_large_vision-language_models.md)**

:   提出 V2Drop，首次从 token 变化量（variation）视角出发，通过渐进式丢弃 LLM 内部变化量最小的"懒惰"视觉 token，实现无训练、无位置偏差、兼容高效算子的 LVLM 推理加速，在图像和视频理解任务中分别保留 94.0% 和 98.6% 原始性能，同时降低 LLM 生成延迟 31.5% 和 74.2%。

**[VecGlypher: Unified Vector Glyph Generation with Language Models](multimodal_vlm/vecglypher_unified_vector_glyph_generation_with_language_models.md)**

:   提出VecGlypher——首个统一文本和图像引导的矢量字形生成语言模型，通过两阶段训练(大规模SVG语法学习+专家标注对齐)直接自回归生成可编辑SVG路径，无需光栅中间步骤或向量化后处理。

**[Venus: Benchmarking and Empowering Multimodal Large Language Models for Aesthetic Guidance and Cropping](multimodal_vlm/venus_benchmarking_and_empowering_multimodal_large_language_models_for_aesthetic.md)**

:   定义审美指导(AG)新任务并构建AesGuide基准(10748张照片含审美评分、分析和指导标注)，提出Venus两阶段框架——先通过渐进式审美问答赋能MLLM审美指导能力，再通过CoT推理激活审美裁剪能力，在两个任务上均达到SOTA。

**[VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](multimodal_vlm/vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)**

:   提出VGGDrive框架，通过冻结的3D视觉基础模型VGGT为VLM注入跨视图几何感知能力，设计插拔式CVGE模块分层自适应地将3D特征注入VLM各层的2D视觉嵌入中，在五个自动驾驶基准上实现显著性能提升。

**[Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](multimodal_vlm/video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)**

:   提出VisionToM，一个基于视觉的轻量级干预框架，通过探测和干预MLLM中对视觉输入和ToM推理敏感的注意力头，在不微调模型的情况下显著增强多模态大语言模型的心智理论推理能力，在EgoToM基准上大幅提升表现。

**[VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](multimodal_vlm/videofusion_a_spatio-temporal_collaborative_network_for_multi-modal_video_fusion.md)**

:   提出首个大规模红外-可见光视频融合框架 VideoFusion，通过跨模态差分增强、完整模态引导融合和双向时序协同注意力机制，联合建模跨模态互补性与时序动态，生成时空一致的高质量融合视频，并构建了包含220个视频/15.4万帧的 M3SVD 数据集。

**[ViKey: Enhancing Temporal Understanding in Videos via Visual Prompting](multimodal_vlm/vikey_enhancing_temporal_understanding_in_videos_via_visual_prompting.md)**

:   ViKey 通过在视频帧上叠加帧序号的视觉提示（Visual Prompting），配合轻量的关键词-帧映射（KFM）模块，在免训练条件下显著提升 VideoLLM 的时序推理能力，即使只用 20% 的帧也能接近密集帧的性能。

**[ViRC: Enhancing Visual Interleaved Mathematical CoT with Reason Chunking](multimodal_vlm/virc_enhancing_visual_interleaved_mathematical_cot_with_reason_chunking.md)**

:   ViRC 提出 Reason Chunking 机制，将多模态数学 CoT 结构化为连续的"关键推理单元（CRU）"，模拟人类专家反复审视图像并逐步证明中间命题的过程，通过 CRUX 数据集和渐进式训练策略（Instructional SFT → Practice SFT → Strategic RL），实现ViRC-7B 在数学基准上平均提升 18.8%。

**[Vision-Language Models Encode Clinical Guidelines for Concept-Based Medical Reasoning](multimodal_vlm/vision-language_models_encode_clinical_guidelines_for_concept-based_medical_reas.md)**

:   提出MedCBR框架，通过将临床诊断指南（如BI-RADS）融入概念瓶颈模型的训练和推理过程，利用LVLM生成指南一致性报告增强概念监督，结合多任务CLIP训练和大推理模型生成结构化临床解释，在超声和乳腺X光癌症检测上达到94.2%和84.0%的AUROC。

**[VISion On Request: Enhanced VLLM Efficiency with Sparse, Dynamically Selected, Vision-Language Interactions](multimodal_vlm/vision_on_request_enhanced_vllm_efficiency_with_sparse_dynamically_selected_visi.md)**

:   VISOR 提出了一种区别于视觉 token 压缩的新效率范式——通过稀疏化 LLM 内部视觉-语言交互层（少量交叉注意力 + 动态选择的自注意力层），在保留完整高分辨率视觉 token 的同时实现 8.6-18 倍 FLOPs 节省，尤其在需要细粒度理解的困难任务上大幅超越 token 压缩方法。

**[VL-RouterBench: A Benchmark for Vision-Language Model Routing](multimodal_vlm/vl-routerbench_a_benchmark_for_vision-language_model_routing.md)**

:   提出VL-RouterBench，首个面向视觉-语言模型的系统性路由基准，涵盖14个数据集、17个候选模型和519,180个样本-模型对，评估10种路由方法，并发现当前最优路由器与理想Oracle之间仍存在显著差距。

**[VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery](multimodal_vlm/vlm-guided_group_preference_alignment_for_diffusion-based_human_mesh_recovery.md)**

:   提出基于VLM的双记忆自反思评判代理（Critique Agent）为扩散式人体网格恢复生成组级偏好信号，再通过组偏好对齐（Group Preference Alignment）微调扩散模型，无需3D标注即可大幅提升野外场景下的HMR精度。

**[VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models](multimodal_vlm/vlm-loc_localization_in_point_cloud_maps_via_vision-language_models.md)**

:   提出VLM-Loc框架，将3D点云地图转换为BEV图像和场景图供VLM进行结构化空间推理，结合部分节点分配（PNA）机制实现文本-点云精细定位，在自建的CityLoc基准上以Recall@5m提升14.20%大幅超越先前SOTA。

**[VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](multimodal_vlm/vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)**

:   提出VLM-Pruner，一种免训练的离心式token剪枝方法，通过空间稀疏缓冲（BSS）准则平衡冗余消除与局部细节完整性，在88.9%剪枝率下跨5个VLM一致超越现有方法，同时实现端到端推理加速。

**[Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks](multimodal_vlm/vlm_model_inversion_adaptive_token_weight.md)**

:   首次系统研究 VLM 的模型反转（Model Inversion）攻击，提出一套面向 token 生成特性的反转策略（TMI/TMI-C/SMI），以及基于视觉注意力强度动态加权 token 梯度贡献的 SMI-AW 方法，在 4 种 VLM 和 3 个数据集上实现最高 61.21% 的人类评估攻击准确率，揭示了 VLM 严重的训练数据隐私泄露风险。

**[VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments](multimodal_vlm/vs_bench_evaluating_vlms_for_strategic_abilities_in_multi_agent_environments.md)**

:   本文提出 VS-Bench，一个包含十个视觉化博弈环境的多模态基准，从感知、策略推理和决策三个维度系统评估 VLM 在多智能体环境中的策略能力，发现当前最强模型在推理和决策上仍与最优表现有显著差距。

**[Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training](multimodal_vlm/wan-weaver_interleaved_multi-modal_generation_via_decoupled_training.md)**

:   Wan-Weaver 提出规划器（VLM）+ 可视化器（DiT）的解耦架构，通过大规模文本代理数据训练规划器而非真实交错数据，在 OpenING 上 Overall 8.67 分超越 Nano Banana 的 8.85，在保持理解能力（MMMU 74.9）的同时实现 SOTA 交错文图生成。

**[WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs](multimodal_vlm/weavetime_streaming_video_llm_memory.md)**

:   诊断了当前 Video-LLM 存在的"时间不可知"（Time-Agnosticism）问题，提出 WeaveTime 框架，通过训练时的时序重建辅助任务（SOPE）赋予模型时序感知能力，推理时用不确定性门控的粗到细记忆缓存（PCDF-Cache）实现高效自适应记忆检索，在流式视频 QA 上取得显著提升。

**[What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](multimodal_vlm/what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)**

:   提出EmbedLens探针工具系统分析MLLM中视觉token的内部结构，发现视觉token分为sink/dead/alive三类（约40%为无用token），alive token已在进入LLM前编码丰富语义（"预语言"特性），且LLM内部视觉计算对大多数任务冗余，直接中层注入即可。

**[When to Think and When to Look: Uncertainty-Guided Lookback](multimodal_vlm/when_to_think_and_when_to_look_uncertainty-guided_lookback.md)**

:   本文首次系统分析了 LVLM 中 test-time thinking 对视觉推理的影响，发现"多想不如多看"——长推理链常忽略图像导致"long-wrong"轨迹，并据此提出不确定性引导的 lookback 解码策略，通过在推理链漂移时注入视觉回看提示，在不修改模型的前提下将 MMMU 等 6 个基准提升 2-6 个点。

**[When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs](multimodal_vlm/when_token_pruning_is_worse_than_random_understanding_visual_token_information_i.md)**

:   发现VLLM深层中现有token剪枝方法不如随机剪枝的现象，提出基于输出概率变化量化视觉token信息的方法，揭示了"信息地平线"——视觉token信息在某层均匀消散至零的临界层，其位置受任务视觉复杂度和模型能力动态影响，并证明简单集成随机剪枝能有效提升现有方法。

**[Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation](multimodal_vlm/where_mllms_attend_and_what_they_rely_on_explaining_autoregressive_token_generat.md)**

:   提出Eagle，一个轻量级黑盒归因框架，通过insight score（充分性）和necessity score（不可或缺性）的统一目标函数对MLLM的自回归token生成进行空间归因，并量化每个token依赖语言先验还是感知证据，在忠实度/定位/幻觉诊断上全面超越现有方法且GPU显存需求大幅降低。

**[Which Concepts to Forget and How to Refuse? Decomposing Concepts for Continual Unlearning in Large Vision-Language Models](multimodal_vlm/which_concepts_to_forget_and_how_to_refuse_decomposing_concepts_for_continual_un.md)**

:   本文提出CORE(COncept-aware REfuser)，一个面向大视觉语言模型(LVLM)持续遗忘的框架：通过将待删除的视觉-语言对分解为细粒度的视觉属性和文本意图概念，使用概念调制器识别需要拒绝的概念组合，再通过混合拒绝专家(refusers)生成概念对齐的拒绝回复，在16个连续遗忘任务上实现了90.67% CRR和88.02% AR的最佳遗忘-保留权衡。

**[Widget2Code: From Visual Widgets to UI Code via Multimodal LLMs](multimodal_vlm/widget2code_from_visual_widgets_to_ui_code_via_multimodal_llms.md)**

:   首次形式化 Widget-to-Code 任务，构建了首个纯图像 widget 数据集和多维评估体系，提出基于感知代理和 WidgetFactory 基础设施的模块化基线，通过组件分解、图标检索、可复用可视化模板和自适应渲染实现高保真 widget 重建。

**[Zina: Multimodal Fine-grained Hallucination Detection and Editing](multimodal_vlm/zina_multimodal_fine-grained_hallucination_detection_and_editing.md)**

:   Zina 提出了多模态细粒度幻觉检测与编辑任务，设计了两阶段系统（detector MLLM + reviewer MLLM）将 token 复制委托给确定性函数以简化模型负担，同时构建了 VisionHall 数据集（6.9K 人工标注 + 20K 图结构合成数据），在检测 F1 上超过 GPT-4o 达 15.8 个点。

---

## 🧊 3D 视觉 { #3d_vision }

**[3D-Fixer: Coarse-to-Fine In-place Completion for 3D Scenes from a Single Image](3d_vision/3d-fixer_coarse-to-fine_in-place_completion_for_3d_scenes_from_a_single_image.md)**

:   提出"就地补全"（in-place completion）新范式，将预训练物体级生成先验扩展到场景级，直接在原始位置对碎片化几何进行补全，无需显式位姿对齐，同时构建110K规模场景级数据集 ARSG-110K，大幅超越 MIDI 和 Gen3DSR 等基线。

**[3D-IDE: 3D Implicit Depth Emergent](3d_vision/3d-ide_3d_implicit_depth_emergent.md)**

:   提出"隐式几何涌现原则"（IGEP），通过训练时的轻量级几何验证器和全局3D教师进行特权监督，使视觉编码器在仅输入RGB视频时即具备3D感知能力，推理时零延迟开销，在多个3D场景理解基准上超越同类方法。

**[3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_vision/3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)**

:   提出自约束先验（Self-Constrained Prior），通过融合当前3D高斯渲染的深度图构建TSDF距离场，以此为先验对高斯施加几何感知约束（异常值移除、不透明度约束、向表面移动），实现高保真表面重建，在NeRF-Synthetic和DTU上达到SOTA。

**[3D sans 3D Scans: Scalable Pre-training from Video-Generated Point Clouds](3d_vision/3d_sans_3d_scans_scalable_pre-training_from_video-generated_point_clouds.md)**

:   提出LAM3C框架，首次证明从无标注网络视频（房产导览等）重建的视频生成点云(VGPC)可替代真实3D扫描进行3D自监督预训练，通过拉普拉斯平滑损失和噪声一致性损失稳定噪声点云上的表示学习，配合自建RoomTours数据集(49K场景)在室内语义和实例分割上匹配甚至超越使用真实扫描的方法。

**[3DrawAgent: Teaching LLM to Draw in 3D with Early Contrastive Experience](3d_vision/3drawagent_teaching_llm_to_draw_in_3d_with_early_contrastive_experience.md)**

:   提出免训练的 3DrawAgent 框架，让冻结的 LLM 通过"对比经验优化"（contrastive knowledge extraction）自我学习3D空间推理，以自回归方式生成语言驱动的3D Bezier草图，无需参数更新即可达到接近有训练方法的水平。

**[4C4D: 4 Camera 4D Gaussian Splatting](3d_vision/4c4d_4_camera_4d_gaussian_splatting.md)**

:   提出 4C4D 框架，通过神经衰减函数（Neural Decaying Function）自适应控制高斯不透明度衰减，解决稀疏（仅4个相机）4D高斯溅射中几何与外观学习的不平衡问题，在多个数据集上达到SOTA。

**[4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](3d_vision/4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)**

:   提出 4DEquine 框架，将单目视频的马科动物 4D 重建**解耦**为动态运动估计（AniMoFormer）和静态外观重建（EquineGS）两个子问题，仅用合成数据训练即在真实数据上达到 SOTA。

**[A2Z-10M+: Geometric Deep Learning with A-to-Z BRep Annotations for AI-Assisted CAD Modeling and Reverse Engineering](3d_vision/a2z-10m_geometric_deep_learning_with_a-to-z_brep_annotations_for_ai-assisted_cad.md)**

:   构建了包含 1000 万+ 多模态标注（高分辨率3D扫描、手绘3D草图、文本描述、BRep拓扑标签）的 100 万+ CAD 模型数据集 A2Z，为 Scan-to-BRep 逆向工程和多模态 BRep 学习提供了前所未有的数据基础，并训练基础模型在边界/角点检测上大幅超越现有方法。

**[A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection](3d_vision/a_semantically_disentangled_unified_model_for_multi-category_3d_anomaly_detectio.md)**

:   提出 SeDiR 框架，通过粗到细全局标记化（CFGT）、类别条件对比学习（C3L）和几何引导解码器（GGD）三个模块实现语义解纠缠的统一3D异常检测，解决跨类别特征纠缠（ICE）问题，在 Real3D-AD 和 Anomaly-ShapeNet 上分别超出SOTA 2.8% 和 9.1% AUROC。

**[GAP: Action-Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation](3d_vision/action-geometry_prediction_with_3d_geometric_prior_for_bimanual_manipulation.md)**

:   GAP利用预训练3D几何基础模型（π³）提取3D特征，融合2D语义和本体感知，通过条件扩散联合预测未来动作序列和未来3D pointmap，在RoboTwin 2.0和真实双臂实验中达到SOTA。

**[Action-guided Generation of 3D Functionality Segmentation Data](3d_vision/action-guided_generation_of_3d_functionality_segmentation_data.md)**

:   提出 SynthFun3D，首个从动作描述自动生成3D功能性分割训练数据的方法，通过元数据驱动的3D物体检索和场景布局，无需人工标注即可生成精确的部件级交互掩码，合成+真实数据训练在 SceneFun3D 基准上提升 +2.2 mAP / +6.3 mAR / +5.7 mIoU。

**[ActionMesh: Animated 3D Mesh Generation with Temporal 3D Diffusion](3d_vision/actionmesh_animated_3d_mesh_generation_with_temporal_3d_diffusion.md)**

:   提出 ActionMesh，通过最小化扩展预训练3D扩散模型增加时间轴（时序3D扩散），再用时序3D自编码器将独立形状序列转为拓扑一致的动画网格，仅2分钟即可从视频/文本/3D网格等多种输入生成产品级动画3D网格，在几何精度和时间一致性上均达SOTA。

**[Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](3d_vision/ada3drift_adaptive_trainingtime_drifting_for_onest.md)**

:   Ada3Drift 提出将扩散策略中的迭代精炼从推理时转移到训练时，通过训练时漂移场（吸引预测动作至专家模式+排斥其他生成样本）实现高保真单步（1 NFE）3D 视觉运动策略，在 Adroit、Meta-World、RoboTwin 和真实机器人任务上达到 SOTA，同时推理速度提升 10 倍。

**[Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning](3d_vision/adapting_point_cloud_analysis_via_multimodal_bayesian_distribution_learning.md)**

:   BayesMM 提出了一个无需训练的动态贝叶斯分布学习框架，将文本和几何模态建模为高斯分布，并通过贝叶斯模型平均自动调节模态权重，在多个点云基准上实现了鲁棒的测试时适配，平均提升超过 4%。

**[AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction](3d_vision/aerodgs_physically_consistent_dynamic_gaussian_splatting_for_single-sequence_aer.md)**

:   提出 AeroDGS，一个面向单目无人机视频的物理引导 4D 高斯泼溅框架，通过单目几何提升模块重建可靠的静态与动态几何，并引入可微的地面支撑、直立稳定性和轨迹平滑性物理先验，将模糊的图像线索转化为物理一致的运动估计，在合成与真实 UAV 场景上均优于现有方法。

**[AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis](3d_vision/affordgrasp_cross-modal_diffusion_for_affordance-aware_grasp_synthesis.md)**

:   AffordGrasp 提出了一个基于扩散的跨模态框架，通过可供性引导的潜空间扩散和分布调节模块（DAM），从文本指令和物体点云生成物理可行且语义一致的人手抓取姿态，在四个基准上显著超越现有方法。

**[AffordMatcher: Affordance Learning in 3D Scenes from Visual Signifiers](3d_vision/affordmatcher_affordance_learning_in_3d_scenes_from_visual_signifiers.md)**

:   AffordMatcher 提出了一种从视觉信号（RGB 图像中的人物交互）定位 3D 场景中可供性区域的方法，通过大规模 AffordBridge 数据集和基于不相似度矩阵的 Match-to-Match 注意力机制，在零样本可供性分割上达到 53.4 mAP，超越次优方法 7.8 个点。

**[Affostruction: 3D Affordance Grounding with Generative Reconstruction](3d_vision/affostruction_3d_affordance_grounding_with_generative_reconstruction.md)**

:   提出Affostruction，通过稀疏体素融合的生成式重建完成物体几何（包括未观测区域），并用Flow Matching建模功能可供性的多模态分布，在完整3D形状上实现功能区域定位，重建IoU提升54.8%、affordance aIoU提升40.4%。

**[AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](3d_vision/anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)**

:   AnchorSplat 提出了一种锚点对齐的前馈 3DGS 框架，以 3D 几何先验（稀疏点云）为锚点直接在 3D 空间预测高斯，用约 20 倍更少的高斯数量和一半的重建时间在 ScanNet++ v2 上达到 SOTA 性能（PSNR 21.48），同时具备更好的深度估计精度。

**[AnthroTAP: Learning Point Tracking with Real-World Motion](3d_vision/anthrotap_learning_point_tracking_with_real-world_motion.md)**

:   AnthroTAP 提出了一种自动化管线，从真实人体运动视频中通过 SMPL 拟合和光流过滤生成大规模伪标签点跟踪数据，仅用 1.4K 视频 + 4 GPU 一天训练即达到TAP-Vid 基准的 SOTA 性能，超越使用 15M 视频的 BootsTAPIR。

**[AnyPcc: Compressing Any Point Cloud with a Single Universal Model](3d_vision/anypcc_compressing_any_point_cloud_with_a_single_universal_model.md)**

:   提出 AnyPcc，通过 Universal Context Model（融合空间+通道双粒度先验）和 Instance-Adaptive Fine-Tuning（实例自适应微调）策略，用单一模型在 15 个多样化数据集上实现 SOTA 点云几何压缩，相比 G-PCC v23 获得 ~12% 的码率增益。

**[APC: Transferable and Efficient Adversarial Point Counterattack for Robust 3D Point Cloud Recognition](3d_vision/apc_adversarial_point_counterattack.md)**

:   APC 提出一种轻量级输入级净化模块，通过生成逐点反扰动来中和对抗攻击，同时在几何一致性和语义一致性双重约束下训练，实现了跨攻击和跨模型的强鲁棒性。

**[ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](3d_vision/arthoi_taming_foundation_models_for_monocular_4d_reconstruction_of_hand-articula.md)**

:   ArtHOI 首次实现了从单目 RGB 视频重建手与铰接物体（如剪刀、眼镜、笔记本电脑）4D 交互的完整流水线，通过自适应采样精化（ASR）优化物体度量尺度和位姿、以及 MLLM 引导的手物对齐方法，在多个数据集上超越了需要预扫描物体几何的基线 RSRD。

**[ArtLLM: Generating Articulated Assets via 3D LLM](3d_vision/artllm_generating_articulated_assets_via_3d_llm.md)**

:   ArtLLM 将铰接物体生成建模为语言生成问题，使用 3D 多模态 LLM 从点云自回归预测部件布局和运动关节参数（离散化为 token），再结合 XPart 生成高保真部件几何，在 PartNet-Mobility 数据集上显著超越现有方法（mIoU 0.69, 推理仅需 19 秒）。

**[AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](3d_vision/ava-bench_atomic_visual_ability_benchmark_for_vision_foundation_models.md)**

:   提出 AVA-Bench，首个将视觉基础模型（VFM）的能力解耦为 14 种原子视觉能力（AVA）的系统性评测基准，通过训练-测试分布对齐和单一能力隔离测试，精准定位 VFM 的强项与短板，并发现 0.5B 小模型即可保持与 7B 模型相当的 VFM 排名一致性。

**[AvatarPointillist: AutoRegressive 4D Gaussian Avatarization](3d_vision/avatarpointillist_autoregressive_4d_gaussian_avatarization.md)**

:   AvatarPointillist 提出了一种自回归（AR）生成框架来构建 4D 高斯头像：用 decoder-only Transformer 逐点生成 3DGS 点云（含绑定信息），再用 Gaussian Decoder 预测渲染属性，打破了固定模板拓扑的限制，实现了自适应点密度调整，在 NeRSemble 上全面超越 LAM、GAGAvatar 等基线。

**[Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection](3d_vision/back_to_point_exploring_point-language_models_for_zero-shot_3d_anomaly_detection.md)**

:   BTP 首次将预训练的点-语言模型（PLM，如 ULIP）应用于零样本 3D 异常检测，提出多粒度特征嵌入模块（MGFEM）融合 patch 级语义、几何描述子和全局 CLS token，配合联合表示学习策略，在 Real3D-AD 点级 AUROC 达到 84.5%，大幅超越观 VLM 渲染方案的 PointAD（73.5%）。

**[BRepGaussian: CAD Reconstruction from Multi-View Images with Gaussian Splatting](3d_vision/brepgaussian_cad_reconstruction_from_multi-view_images_with_gaussian_splatting.md)**

:   BRepGaussian 首次实现了从多视图图像直接重建完整 B-rep CAD 模型，通过两阶段的 2D 高斯泼溅学习边缘和面片特征，再经参数化拟合生成水密的边界表示，无需点云监督。

**[BulletGen: Improving 4D Reconstruction with Bullet-Time Generation](3d_vision/bulletgen_improving_4d_reconstruction_with_bullet-time_generation.md)**

:   提出 BulletGen，在选定的"子弹时间"冻结帧用静态视频扩散模型生成新视角，精确定位后用于监督 4D 高斯场景优化，在仅有单目视频输入的情况下实现极端新视角合成和 2D/3D 追踪的 SOTA。

**[Can Natural Image Autoencoders Compactly Tokenize fMRI Volumes for Long-Range Dynamics Modeling?](3d_vision/can_natural_image_autoencoders_compactly_tokenize_fmri_volumes_for_long-range_dy.md)**

:   提出 TABLeT，利用预训练的 2D 自然图像自编码器（DCAE）将 3D fMRI 体积压缩为仅 27 个连续 token，配合简单 Transformer 编码器实现前所未有的长时序建模（256 帧），在 UKB、HCP、ADHD-200 上多任务超越 SOTA 体素方法，且计算效率大幅提升。

**[CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interaction](3d_vision/cari4d_category_agnostic_4d_reconstruction_of_human_object_interaction.md)**

:   提出CARI4D，首个类别无关的方法，从单目RGB视频中重建度量尺度的4D人物交互——包括物体形状重建、位姿跟踪、手部接触推理和物理约束优化，零样本泛化到未见类别。

**[Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation](3d_vision/catalyst4d_highfidelity_3dto4d_scene_editing_via_d.md)**

:   提出Catalyst4D框架，通过锚点运动引导（AMG，基于最优传输建立区域级对应）和颜色不确定性引导外观精炼（CUAR，自动识别并修复遮挡伪影），将成熟的3D静态编辑结果传播到4D动态高斯场景中，在CLIP语义相似度上一致性超越现有方法。

**[CGHair: Compact Gaussian Hair Reconstruction with Card Clustering](3d_vision/cghair_compact_gaussian_hair_reconstruction_with_card_clustering.md)**

:   提出 CGHair，通过发片（hair card）引导的分层聚类和共享高斯外观码本，在保持可比视觉质量的同时实现 200 倍以上的外观参数压缩和 4 倍发丝重建加速。

**[Changes in Real Time: Online Scene Change Detection with Multi-View Fusion](3d_vision/changes_in_real_time_online_scene_change_detection_with_multi-view_fusion.md)**

:   提出首个同时具备在线、姿态无关、无标注、多视角一致性的场景变化检测（SCD）方法，通过自监督融合损失将像素级和特征级变化线索集成到 3DGS 变化表示中，在超过 10 FPS 的实时速率下超越了所有已有离线方法的检测精度。

**[CLIPoint3D: Language-Grounded Few-Shot Unsupervised 3D Point Cloud Domain Adaptation](3d_vision/clipoint3d_language-grounded_few-shot_unsupervised_3d_point_cloud_domain_adaptat.md)**

:   首个基于 CLIP 的少样本无监督 3D 点云域自适应框架，通过知识驱动的 prompt tuning、参数高效微调、熵引导视图选取和不确定性感知对齐损失，在 PointDA-10 和 GraspNetPC-10 上以仅 ~11M 可训练参数取得 3-16% 的一致性精度提升。

**[CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration](3d_vision/cmhanet_a_cross-modal_hybrid_attention_network_for_point_cloud_registration.md)**

:   提出 CMHANet，通过跨模态混合注意力机制将 2D 图像纹理语义特征与 3D 点云几何特征深度融合，结合对比学习优化函数，在 3DMatch/3DLoMatch 上实现 SOTA 点云配准性能。

**[Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass](3d_vision/coherent_humanscene_reconstruction_from_multiperso.md)**

:   提出CHROMM统一框架，整合Pi3X几何先验和Multi-HMR人体先验到单一前馈网络，从多人多视图视频中一次性联合重建相机、场景点云和SMPL-X人体网格，无需外部模块、预处理或迭代优化，RICH上多视图WA-MPJPE达53.1mm且比HAMSt3R快8倍以上。

**[Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation](3d_vision/context-nav_context-driven_exploration_and_viewpoint-aware_3d_spatial_reasoning_.md)**

:   Context-Nav 将长文本描述的上下文信息从后验验证信号提升为前驱探索先验——通过上下文驱动的 value map 引导前沿选择，并在候选目标处执行视点感知的 3D 空间关系验证，在 InstanceNav 和 CoIN-Bench 上无需任何训练即取得 SOTA。

**[Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment](3d_vision/cross-instance_gaussian_splatting_registration_via_geometry-aware_feature-guided.md)**

:   提出 GSA（Gaussian Splatting Alignment），首个实现跨实例类别级 3DGS 模型配准的方法，通过几何感知特征引导的粗配准（扩展 ICP 求解相似变换）和多视角特征一致性的精配准，在同物体和跨物体场景下均大幅超越现有方法。

**[CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image](3d_vision/crowdgaussian_reconstructing_high-fidelity_3d_gaussians_for_human_crowd_from_a_s.md)**

:   CrowdGaussian 提出了从单张图像重建多人 3D 高斯泼溅表示的统一框架，通过自监督适配的大型遮挡人体重建模型（LORM）恢复被遮挡区域的完整几何，再通过自校准学习（SCL）训练的单步扩散精炼器（CrowdRefiner）提升纹理细节质量。

**[CUBE: Representing 3D Faces with Learnable B-Spline Volumes](3d_vision/cube_bspline_3d_faces.md)**

:   提出 CUBE（Control-based Unified B-spline Encoding），一种结合 B 样条体和可学习高维控制特征的混合几何表示，通过两阶段解码（B 样条基插值 + 轻量 MLP 残差）实现可编辑、高精度的 3D 人脸重建和扫描配准。

**[CustomTex: High-fidelity Indoor Scene Texturing via Multi-Reference Customization](3d_vision/customtex_high-fidelity_indoor_scene_texturing_via_multi-reference_customization.md)**

:   提出CustomTex框架，通过实例级的多参考图像驱动和双蒸馏训练策略（语义级VSD蒸馏+像素级超分蒸馏），实现3D室内场景的高保真、实例可控纹理生成，在语义一致性、纹理清晰度和减少"烘焙阴影"方面全面超越现有方法。

**[Dark3R: Learning Structure from Motion in the Dark](3d_vision/dark3r_learning_structure_from_motion_in_the_dark.md)**

:   提出 Dark3R 框架，通过教师-学生蒸馏将 MASt3R 的3D先验迁移到极端低光照（SNR < −4 dB）原始图像上，实现了传统方法完全失败的暗光环境下的运动恢复结构（SfM）和新视角合成。

**[DeepShapeMatchingKit: Accelerated Functional Map Solver and Shape Matching Pipelines Revisited](3d_vision/deepshapematchingkit_accelerated_functional_map_solver.md)**

:   本文提出了功能图求解器的向量化重构实现33倍加速，识别并记录了DiffusionNet的两个未记录实现变体，引入平衡准确率作为部分匹配评估的补充指标，并发布了统一的开源代码库。

**[Deformation-based In-Context Learning for Point Cloud Understanding](3d_vision/deformation-based_in-context_learning_for_point_cloud_understanding.md)**

:   提出 DeformPIC，将点云 In-Context Learning 从"掩码重建"范式重新定义为"形变迁移"范式，通过 Deformation Extraction Network 提取任务语义 + Deformation Transfer Network 迁移形变到查询点云，在重建/去噪/配准上分别降低 CD 1.6/1.8/4.7。

**[DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization](3d_vision/directfisheye-gs_enabling_native_fisheye_input_in_gaussian_splatting_with_cross-.md)**

:   将 Kannala-Brandt 鱼眼投影模型原生集成到 3DGS 流程中，并提出基于特征重叠的跨视图联合优化策略，避免了预去畸变带来的信息损失，在多个公开数据集上达到或超越 SOTA。

**[DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis](3d_vision/dmaligner_enhancing_image_alignment_via_diffusion_model_based_view_synthesis.md)**

:   提出 DMAligner，将图像对齐问题从传统的光流 warp 范式转化为"对齐导向的视图合成"任务，利用条件扩散模型直接生成对齐后的完整图像，配合专门构建的 DSIA 合成数据集和动态感知掩码模块（DMP），有效避免了 warp 方法固有的 ghosting 和遮挡伪影，在多个基准上全面超越现有方法。

**[DROID-W: DROID-SLAM in the Wild](3d_vision/droid-slam_in_the_wild.md)**

:   提出 DROID-W，通过将不确定性估计引入可微分 Bundle Adjustment（Uncertainty-aware BA），结合 DINOv2 特征驱动的动态不确定性更新机制和单目深度正则化，使 DROID-SLAM 在高度动态的野外（in-the-wild）场景中实现鲁棒的相机位姿估计和场景重建，约 10 FPS 实时运行。

**[DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](3d_vision/dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)**

:   针对 3DGS 在稀疏视角下的过拟合问题，提出 DropAnSH-GS：用 Anchor-based Dropout（丢弃锚点及其邻域的 Gaussian 簇）替代独立随机 Dropout 来破坏局部冗余补偿效应，同时引入球谐函数（SH）Dropout 抑制高阶 SH 过拟合并支持训练后无损压缩。

**[DuoMo: Dual Motion Diffusion for World-Space Human Reconstruction](3d_vision/duomo_dual_motion_diffusion_for_world-space_human_reconstruction.md)**

:   提出 DuoMo，将世界空间人体运动重建分解为两个独立的扩散模型：camera-space 模型从视频提取泛化性强的相机坐标运动估计，world-space 模型将 lifting 后的噪声提案精炼为全局一致的世界坐标运动。直接生成 mesh 顶点运动而非 SMPL 参数，在 EMDB 上 W-MPJPE 降低 16%，RICH 上降低 30%。

**[Dynamic Black-hole Emission Tomography with Physics-informed Neural Fields](3d_vision/dynamic_black-hole_emission_tomography_with_physics-informed_neural_fields.md)**

:   提出 PI-DEF，利用物理信息约束的坐标神经网络同时重建黑洞附近气体的 4D（时间+3D）发射率场和 3D 速度场，在稀疏 EHT 测量下显著优于硬约束 Keplerian 动力学的 BH-NeRF。

**[E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training](3d_vision/e-rayzer_self-supervised_3d_reconstruction_as_spatial_visual_pre-training.md)**

:   E-RayZer是首个真正自监督的前馈式3D高斯重建模型，用显式3D高斯替代RayZer的隐式潜空间场景表示，配合基于视觉重叠度的课程学习策略，在零3D标注条件下学到几何接地的3D感知表征，位姿估计上碾压RayZer（RPA@5°从≈0提升至90.8），下游3D任务frozen-backbone probing大幅领先DINOv3/CroCo v2等主流预训练模型，甚至比肩有监督VGGT。

**[E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](3d_vision/e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)**

:   提出 E2EGS，一个完全基于事件流的无位姿 3D 重建框架：通过 patch-based 时间一致性分析从事件流中提取抗噪边缘图，利用边缘信息指导高斯初始化和加权损失优化，在无需深度模型或 RGB 输入的情况下实现了高质量的轨迹估计和 3D 重建。

**[Easy3E: Feed-Forward 3D Asset Editing via Rectified Voxel Flow](3d_vision/easy3e_feed-forward_3d_asset_editing_via_rectified_voxel_flow.md)**

:   提出基于 TRELLIS 3D 生成骨干的前馈式 3D 资产编辑框架，通过 Voxel FlowEdit 在稀疏体素潜空间中实现全局一致的几何形变，并结合法线引导的多视角纹理精修恢复高频细节。

**[Efficient Hybrid SE(3)-Equivariant Visuomotor Flow Policy via Spherical Harmonics](3d_vision/efficient_hybrid_se3-equivariant_visuomotor_flow_policy_via_spherical_harmonics_.md)**

:   提出E3Flow，首个基于球谐表示的等变flow matching策略框架，通过特征增强模块（FEM）动态融合点云和图像两种模态的视觉信息，结合rectified flow实现高效等变动作生成，在MimicGen 8个任务上平均成功率超过最强基线SDP 3.12%的同时推理速度提升7倍。

**[Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision](3d_vision/ego-1k_--_a_large-scale_multiview_video_dataset_for_egocentric_vision.md)**

:   提出 Ego-1K，一个包含 956 段短视频的大规模时间同步第一人称多视角视频数据集（12+4 相机、60Hz），填补了第一人称动态 3D 重建领域的数据空白，并展示立体深度引导可大幅提升 4D 新视角合成质量。

**[EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding](3d_vision/embodiedsplat_online_feed-forward_semantic_3dgs_for_open-vocabulary_3d_scene_und.md)**

:   提出 EmbodiedSplat，首个在线前馈式语义 3DGS 框架，通过稀疏系数场+CLIP全局码本实现内存高效的逐高斯语义表示，结合3D几何感知特征，在300+帧流式输入下以5-6 FPS实现全场景开放词汇3D理解。

**[EMGauss: Continuous Slice-to-3D Reconstruction via Dynamic Gaussian Modeling in Volume Electron Microscopy](3d_vision/emgauss_continuous_slice-to-3d_reconstruction_via_dynamic_gaussian_modeling_in_v.md)**

:   将体电子显微镜(vEM)的各向异性切片重建问题重新建模为基于可变形2D高斯溅射的动态3D场景渲染任务，通过Teacher-Student伪标签机制在数据稀疏条件下实现高保真连续切片合成。

**[EmoTaG: Emotion-Aware Talking Head Synthesis on Gaussian Splatting with Few-Shot Personalization](3d_vision/emotag_emotion-aware_talking_head_synthesis_on_gaussian_splatting_with_few-shot_.md)**

:   提出 EmoTaG，一个基于 FLAME-Gaussian 结构先验和门控残差运动网络（GRMN）的情感感知 3D 说话人头合成框架，仅需 5 秒视频即可实现 few-shot 个性化适配，同时兼顾情感表达、唇音同步和几何稳定性。

**[Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator](3d_vision/enhancing_hands_in_3d_whole-body_pose_estimation_with_conditional_hands_modulato.md)**

:   提出Hand4Whole++模块化框架，通过轻量级CHAM模块将预训练手部估计器的特征注入冻结的全身姿态估计器中，实现手腕方向的精准预测，并通过可微刚性对齐从手部模型迁移精细手指关节和手部形状。

**[EventHub: Data Factory for Generalizable Event-Based Stereo Networks without Active Sensors](3d_vision/eventhub_data_factory_for_generalizable_event-based_stereo_networks_without_acti.md)**

:   本文提出 EventHub，一个无需 LiDAR 等主动传感器标注的事件相机立体匹配训练数据工厂，通过新视角合成生成代理事件+深度标签和跨模态蒸馏从 RGB 立体模型迁移知识，训练出的事件立体模型在跨域泛化上超越 LiDAR 监督模型（M3ED 和 MVSEC 上误差降低最高 50%）。

**[Extend3D: Town-Scale 3D Generation](3d_vision/extend3d_town-scale_3d_generation.md)**

:   本文提出 Extend3D，一个无需训练的 3D 场景生成流水线，通过扩展预训练物体级 3D 生成模型（Trellis）的体素隐空间并引入重叠 patch 联合去噪、under-noising SDEdit 初始化和 3D 感知优化，从单张图像生成城镇级大规模 3D 场景，在人类偏好和定量评估中均超越现有方法。

**[ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting](3d_vision/extrinsplat_decoupling_geometry_and_semantics_for_open-vocabulary_understanding_.md)**

:   提出外在范式（extrinsic paradigm），将语义从3DGS几何中完全解耦，通过多粒度物体分组+VLM文本假设构建轻量语义索引层，实现无训练、低存储、支持多义性的开放词汇3D场景理解。

**[FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning](3d_vision/facecam_portrait_video_camera_control_via_scale-aware_conditioning.md)**

:   提出FaceCam系统，通过面部地标(facial landmarks)作为尺度感知的相机表示来解决单目人像视频的相机控制问题，避免了传统相机外参表示的尺度歧义，并设计了合成相机运动和多镜头拼接两种数据增强策略支持连续相机轨迹推理。

**[FACT-GS: Frequency-Aligned Complexity-Aware Texture Reparameterization for 2D Gaussian Splatting](3d_vision/fact-gs_frequency-aligned_complexity-aware_texture_reparameterization_for_2d_gau.md)**

:   提出FACT-GS，将纹理参数化重新定义为采样密度分配问题，通过可学习变形场实现频率自适应的非均匀纹理采样，在固定参数预算下显著提升高频细节恢复能力。

**[Fall Risk and Gait Analysis using World-Spaced 3D Human Mesh Recovery](3d_vision/fall_risk_gait_analysis_hmr.md)**

:   提出基于 GVHMR（世界坐标系 3D 人体网格恢复）的步态分析管线，从单目视频中提取老年人定时起立行走测试的时空步态参数，验证了视频衍生指标与可穿戴传感器的相关性及与跌倒风险的关联。

**[Fast3Dcache: Training-free 3D Geometry Synthesis Acceleration](3d_vision/fast3dcache_training-free_3d_geometry_synthesis_acceleration.md)**

:   本文提出 Fast3Dcache，一个面向 3D 扩散模型的无需训练的几何感知缓存框架，通过预测性缓存调度约束（PCSC）根据体素稳定化模式动态分配缓存预算，以及时空稳定性准则（SSC）基于速度和加速度选择稳定 token 进行复用，实现最高 27.12% 的吞吐提升和 54.83% 的 FLOPs 降低，几何质量仅损失约 2%。

**[Fast SceneScript: Fast and Accurate Language-Based 3D Scene Understanding via Multi-Token Prediction](3d_vision/fast_scenescript_fast_and_accurate_language-based_3d_scene_understanding_via_mul.md)**

:   本文提出 Fast SceneScript，通过将多 token 预测（MTP）引入结构化语言模型实现 3D 场景理解的推理加速，配合自投机解码（SSD）和置信度引导解码（CGD）过滤不可靠 token，以及参数高效的头共享机制，在布局估计和目标检测上分别实现 5.09× 和 5.14× 加速且不损失精度。

**[FastGS: Training 3D Gaussian Splatting in 100 Seconds](3d_vision/fastgs_training_3d_gaussian_splatting_in_100_seconds.md)**

:   提出 FastGS，一个基于多视角一致性的 3DGS 加速框架，通过多视角一致性密集化（VCD）和多视角一致性剪枝（VCP）策略精准控制 Gaussian 数量，在 Mip-NeRF 360 等数据集上实现约 100 秒完成场景训练，相比 vanilla 3DGS 加速 15× 以上，且渲染质量可比。

**[FF3R: Feedforward Feature 3D Reconstruction from Unconstrained Views](3d_vision/ff3r_feedforward_feature_3d_reconstruction_from_unconstrained_views.md)**

:   FF3R是首个完全无标注的前馈框架，能从无约束多视角图像序列中同时进行几何重建和开放词汇语义理解，处理64+张图像的速度比优化方法快180倍。

**[FluidGaussian: Propagating Simulation-Based Uncertainty Toward Functionally-Intelligent 3D Reconstruction](3d_vision/fluidgaussian_propagating_simulation-based_uncertainty_toward_functionally-intel.md)**

:   提出 FluidGaussian，通过流体模拟传播的不确定性指标来指导 3D 重建中的主动视角选择，使重建结果不仅视觉逼真，还具备物理交互的合理性。

**[ForgeDreamer: Industrial Text-to-3D Generation with Multi-Expert LoRA and Cross-View Hypergraph](3d_vision/forgedreamer_industrial_text-to-3d_generation_with_multi-expert_lora_and_cross-v.md)**

:   提出 ForgeDreamer 框架，通过多专家 LoRA 师生蒸馏解决工业领域语义适配问题，结合跨视角超图几何增强实现高阶几何一致性约束，在工业文本到3D生成任务上超越现有方法。

**[Foundry: Distilling 3D Foundation Models for the Edge](3d_vision/foundry_distilling_3d_foundation_models_for_the_edge.md)**

:   提出 Foundation Model Distillation（FMD）范式和 Foundry 框架，通过 compress-and-reconstruct 目标让学生模型学习一组可学习的 SuperToken 来压缩教师的潜空间基向量，生成的单一蒸馏模型在分类、分割、少样本等多任务上保持通用性，同时将 FLOPs 从 478G 降至最低 137G。

**[FreeArtGS: Articulated Gaussian Splatting Under Free-Moving Scenario](3d_vision/freeartgs_articulated_gaussian_splatting_under_free-moving_scenario.md)**

:   FreeArtGS 提出在"自由移动场景"(物体姿态和关节状态同时任意变化)下从单目RGB-D视频重建铰接物体的方法，通过运动驱动的部件分割、鲁棒关节估计和端到端3DGS优化的三阶段流程，在自建FreeArt-21基准和现有数据集上远超所有基线。

**[FreeScale: Scaling 3D Scenes via Certainty-Aware Free-View Generation](3d_vision/freescale_scaling_3d_scenes.md)**

:   FreeScale 通过从已有场景重建中以确定性引导的方式采样高质量自由视角图像，将有限的真实世界数据扩展为大规模训练数据，在前馈新视角合成模型上获得 2.7 dB PSNR 提升。

**[FE2E: From Editor to Dense Geometry Estimator](3d_vision/from_editor_to_dense_geometry_estimator.md)**

:   本文系统分析了图像编辑模型与生成模型在稠密几何估计任务中的微调行为差异，发现编辑模型具有天然的结构先验优势，并基于此提出 FE2E 框架，首次将 DiT 架构的图像编辑模型适配为深度和法线联合估计器，在零样本场景下大幅超越现有 SOTA（ETH3D 上 AbsRel 降低 35%）。

**[From Orbit to Ground: Generative City Photogrammetry from Extreme Off-Nadir Satellite Images](3d_vision/from_orbit_to_ground_generative_city_photogrammetry_from_extreme_off-nadir_satel.md)**

:   提出从稀疏卫星图像重建城市级 3D 模型的两阶段方法：用 Z-Monotonic SDF 建模几何保证建筑结构完整性，再用微调 FLUX 扩散模型做"确定性修复"从退化贴图合成写实纹理，实现从轨道到地面近 90° 视点外推。

**[From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection](3d_vision/from_pairs_to_sequences_track-aware_policy_gradients_for_keypoint_detection.md)**

:   将关键点检测从「图像对匹配」范式转变为「序列级可追踪性优化」，通过强化学习框架 TraqPoint 在图像序列上直接优化关键点的长期追踪质量，在位姿估计、视觉定位、视觉里程计和三维重建任务上均超越 SOTA。

**[FunREC: Reconstructing Functional 3D Scenes from Egocentric Interaction Videos](3d_vision/funrec_reconstructing_functional_3d_scenes_from_egocentric_interaction_videos.md)**

:   本文提出 FunREC，一个无需训练的优化式方法，直接从自我中心 RGB-D 交互视频中重建功能性的铰接式 3D 数字孪生场景——自动发现铰接部件、估计运动学参数、追踪 3D 运动并重建静态和运动几何，在所有基准上大幅超越先前方法（部件分割 mIoU 提升 50+，关节角度误差降低 5-10 倍），并支持仿真导出和机器人交互。

**[GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator](3d_vision/gaussfusion_improving_3d_reconstruction_in_the_wild_with_a_geometry-informed_vid.md)**

:   提出 GaussFusion，一个几何信息引导的视频到视频生成模型，通过渲染包含深度、法线、不透明度和协方差的 Gaussian Primitives Buffer（GP-Buffer）来条件化视频生成器，有效去除 3DGS 重建中的浮动伪影、闪烁和模糊，且能同时适用于优化式和前馈式两种重建范式，蒸馏版本达到 16 FPS 实时推理。

**[GaussianGrow: Geometry-aware Gaussian Growing from 3D Point Clouds with Text Guidance](3d_vision/gaussiangrow_geometry-aware_gaussian_growing_from_3d_point_clouds_with_text_guid.md)**

:   提出 GaussianGrow，通过从易获取的 3D 点云"生长"3D 高斯来替代从零预测几何+外观的传统方案，利用多视图扩散模型生成一致的外观监督，并通过重叠区域检测+迭代补全机制解决视图融合伪影和不可见区域问题，在合成和真实扫描点云上大幅超越 SOTA。

**[GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis](3d_vision/geodesicnvs_probability_density_geodesic_flow_matching_for_novel_view_synthesis.md)**

:   提出概率密度测地线 Flow Matching (PDG-FM) 框架，通过数据到数据的确定性流匹配替代噪声到数据的扩散过程，并利用基于概率密度的测地线优化使插值路径沿数据流形高密度区域行进，实现更几何一致的新视角合成。

**[GGPT: Geometry-Grounded Point Transformer](3d_vision/ggpt_geometry_grounded_point_transformer.md)**

:   提出GGPT框架：通过改进的轻量SfM管线(密集匹配+稀疏BA+DLT三角化)获取几何一致稀疏点云，再用3D Point Transformer V3在三维空间直接融合稀疏几何引导与前馈稠密预测进行residual refinement，仅在ScanNet++上训练即可跨架构、跨数据集显著提升多种前馈3D重建模型。

**[GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport](3d_vision/glint_modeling_scene-scale_transparency_via_gaussian_radiance_transport.md)**

:   GLINT 通过将高斯表征分解为界面、透射、反射三个组件，结合光栅化+光线追踪的混合渲染管线，在场景级透明表面（如玻璃墙、展示柜）的几何和外观重建上取得了 SOTA 效果。

**[Global-Aware Edge Prioritization for Pose Graph Initialization](3d_vision/global-aware_edge_prioritization_for_pose_graph_initialization.md)**

:   提出基于GNN的全局边优先级排序方法，将位姿图初始化从独立的逐对图像检索升级为全局结构感知的边排序+多最小生成树构建，在极稀疏设置下显著提升SfM重建精度。

**[Glove2Hand: Synthesizing Natural Hand-Object Interaction from Multi-Modal Sensing Gloves](3d_vision/glove2hand_synthesizing_natural_hand-object_interaction_from_multi-modal_sensing.md)**

:   提出 Glove2Hand 框架，将佩戴传感手套的第一人称视频翻译为逼真的裸手视频，同时保留触觉和 IMU 信号，并构建了首个多模态手物交互数据集 HandSense，显著提升下游裸手接触估计和遮挡手部追踪性能。

**[GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes](3d_vision/gp-4dgs_probabilistic_4d_gaussian_splatting_from_monocular_video_via_variational.md)**

:   提出 GP-4DGS，将变分高斯过程（GP）整合到 4D 高斯溅射中，通过时空组合核和变分推断实现概率化运动建模，同时赋予 4DGS 不确定性量化、运动外推和自适应运动先验三大新能力。

**[GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](3d_vision/gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)**

:   提出GS-CLIP两阶段框架，通过几何缺陷蒸馏模块将3D点云的全局形状和局部缺陷信息注入文本提示，并用LoRA双流架构协同融合渲染图和深度图，在四个大规模数据集上实现零样本3D异常检测SOTA。

**[Hg-I2P: Bridging Modalities for Generalizable Image-to-Point-Cloud Registration via Heterogeneous Graphs](3d_vision/hg-i2p_bridging_modalities_for_generalizable_image-to-point-cloud_registration_v.md)**

:   Hg-I2P 引入异构图（Heterogeneous Graph）来统一建模 2D 图像区域和 3D 点云区域之间的关系，通过多路径邻接关系挖掘学习跨模态边、基于异构边的特征适配和基于图的投影一致性剪枝，在六个室内外跨域基准上实现了最优的泛化能力和精度。

**[Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting](3d_vision/hierarchical_visual_relocalization_with_nearest_view_synthesis_from_feature_gaus.md)**

:   SplatHLoc 提出了一种基于 Feature Gaussian Splatting (FGS) 的层级视觉重定位框架，通过自适应视点检索合成更接近查询视角的虚拟视图，以及混合特征匹配策略（渲染特征做粗匹配 + 半稠密匹配器做细匹配），在室内外数据集上达到了新的 SOTA 精度。

**[Human Interaction-Aware 3D Reconstruction from a Single Image](3d_vision/human_interaction-aware_3d_reconstruction_from_a_single_image.md)**

:   提出HUG3D框架，通过透视-正交视图变换、群体-个体多视图扩散模型和物理感知几何重建，从单张图片实现交互多人的高保真纹理3D重建，在CD/P2S/NC等指标上全面超越现有方法。

**[Hybrid eTFCE–GRF: Exact Cluster-Size Retrieval with Analytical p-Values for Voxel-Based Morphometry](3d_vision/hybrid_etfcegrf_exact_clustersize_retrieval_with_a.md)**

:   将 eTFCE 的并查集数据结构（精确聚类大小查询）与 pTFCE 的 GRF 解析推断相结合，首次在单一框架中同时实现精确聚类大小提取和无置换检验的解析 $p$ 值，全脑 VBM 分析比 R pTFCE 快 4.6–75 倍、比置换 TFCE 快三个数量级。

**[HyperMVP: Hyperbolic Multiview Pretraining for Robotic Manipulation](3d_vision/hyperbolic_multiview_pretraining_for_robotic_manipulation.md)**

:   提出 HyperMVP，首个在双曲空间中进行3D多视角自监督预训练的框架，通过 GeoLink 编码器学习双曲多视角表征并迁移到机器人操作任务，在 COLOSSEUM 最困难的 All Perturbations 设置下实现 2.1× 性能提升。

**[HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](3d_vision/hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)**

:   提出HyperGaussians，将3DGS扩展到高维多元高斯，通过条件分布建模表情相关的属性变化+逆协方差技巧实现高效条件化，作为即插即用模块集成到FlashAvatar和GaussianHeadAvatar中可显著提升高频细节质量。

**[ICTPolarReal: A Polarized Reflection and Material Dataset of Real World Objects](3d_vision/ictpolarreal_a_polarized_reflection_and_material_dataset_of_real_world_objects.md)**

:   本文构建了首个大规模真实世界偏振反射与材质数据集 ICTPolarReal，利用 8 相机 346 光源的 Light Stage 系统对 218 个日常物体进行交叉/平行偏振捕获，获得超 120 万张高分辨率图像及漫反射-镜面反射分离的地面真值，显著提升了逆渲染、前向重光照和稀疏视角三维重建的效果。

**[Indoor Asset Detection in Large Scale 360° Drone-Captured Imagery via 3D Gaussian Splatting](3d_vision/indoor_asset_detection_in_large_scale_360_drone-captured_imagery_via_3d_gaussian.md)**

:   提出一种基于3D目标编码簿(Object Codebook)的pipeline，将2D分割mask通过语义+空间约束关联为3DGS中的一致3D物体实例，在大规模室内360°无人机图像上实现目标级检测，F1 score比SOTA GAGA提升65%，mAP提升11%。

**[InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](3d_vision/instanthdr_singleforward_gaussian_splatting_for_hi.md)**

:   提出首个前馈HDR新视角合成方法InstantHDR，设计几何引导的外观建模模块解决多曝光融合中的外观不一致问题，并通过MetaNet预测场景特定色调映射参数实现泛化，从未标定多曝光LDR图像中秒级重建HDR 3D高斯场景，稀疏4视角下PSNR超GaussianHDR +2.90 dB，速度快约700倍。

**[Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation](3d_vision/iris_bringing_realworld_priors_into_diffusion_model_for_monocular_depth_estimation.md)**

:   Iris 提出一种确定性扩散框架，通过两阶段"先验到几何"(PGD)调度将真实世界先验注入扩散模型：第一阶段在高时间步用频谱门控蒸馏(SGD)从教师模型提取低频布局先验，第二阶段在低时间步用合成数据精细化高频几何细节，同时引入频谱门控一致性(SGC)实现跨阶段高频信息对齐，在有限数据和计算预算下达到 SOTA 零样本深度估计性能。

**[JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](3d_vision/jopp3d_joint_open_vocabulary_semantic_segmentation.md)**

:   提出 JOPP-3D——首个联合处理 3D 点云和全景图的开放词汇语义分割框架：通过正二十面体切向分解将全景图转为 20 张透视图以适配 SAM/CLIP，提取掩码隔离的实例级 CLIP 嵌入实现 3D 语义分割，再经深度对应回投到全景域，免训练即在 S3DIS 上以 80.9% mIoU 超越所有监督方法。

**[ECKConv: Learning Coordinate-based Convolutional Kernels for Continuous SE(3) Equivariant Point Cloud Analysis](3d_vision/learning_coordinate-based_convolutional_kernels_for_continuous_se3_equivariant_a.md)**

:   提出ECKConv，在intertwiner框架下将卷积核定义在双陪集空间 $\text{SO(2)}\backslash\text{SE(3)}/\text{SO(2)}$ 上，通过坐标网络显式参数化核函数，首次实现连续SE(3)等变性与大规模可扩展性的兼得，在分类、配准、分割四类任务上全面验证。

**[Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](3d_vision/learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)**

:   本文提出通过自适应 SE(3) B 样条运动基显式建模动态高斯的连续位置和朝向变形轨迹，配合软分段重建策略和多视角扩散模型先验，实现单目视频的高质量动态场景新视角合成，在 iPhone 和 NVIDIA 数据集上超越现有方法。

**[Learning Multi-View Spatial Reasoning from Cross-View Relations](3d_vision/learning_multi-view_spatial_reasoning_from_cross-view_relations.md)**

:   XVR（Cross-View Relations）构建了一个 10 万样本的大规模多视角视觉问答数据集，通过对应关系、几何验证和视点定位三类任务显式训练 VLM 的跨视图空间推理能力，在多视角基准和机器人操作任务上均取得显著提升。

**[Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation](3d_vision/let_it_snow_animating_3d_gaussian_scenes_with_dynamic_weather_effects_via_physic.md)**

:   提出 Physics-Guided Score Distillation 框架，利用物理仿真（MPM）作为运动先验引导 Video-SDS 优化，在静态 3DGS 场景中生成具有物理合理运动和真实感外观的动态天气效果（降雪、降雨、雾、沙尘暴）。

**[Lifting Unlabeled Internet-level Data for 3D Scene Understanding](3d_vision/lifting_unlabeled_internet-level_data_for_3d_scene_understanding.md)**

:   构建SceneVerse++，通过自动化数据引擎从6,687个无标注互联网视频中生成3D场景理解训练数据，在3D目标检测（F1@.25提升20.6）、空间VQA（+14.9%）和视觉语言导航（+14% SR）三个任务上展示了利用互联网级数据推进3D场景理解的可行性。

**[LightSplat: Fast and Memory-Efficient Open-Vocabulary 3D Scene Understanding in Five Seconds](3d_vision/lightsplat_fast_and_memory-efficient_open-vocabulary_3d_scene_understanding_in_f.md)**

:   LightSplat 提出了一种快速且内存高效的无训练框架，通过为3D高斯分配紧凑的2字节语义索引（而非高维CLIP特征），配合轻量级索引-特征映射和单步3D聚类，实现了比现有SOTA快50-400倍、内存降低64倍的开放词汇3D场景理解。

**[Lite Any Stereo: Efficient Zero-Shot Stereo Matching](3d_vision/lite_any_stereo_efficient_zero-shot_stereo_matching.md)**

:   提出Lite Any Stereo，通过混合2D-3D代价聚合模块和三阶段百万级数据训练策略（监督→自蒸馏→真实数据知识蒸馏），以不到SOTA精确方法1%的计算量（33G MACs），在四个real-world benchmark上ranking 1st，首次证明超轻量模型可具备强零样本泛化能力。

**[LitePT: Lighter Yet Stronger Point Transformer](3d_vision/litept_lighter_yet_stronger_point_transformer.md)**

:   LitePT 通过深入分析卷积和注意力在U-Net各层级的角色，提出在浅层使用稀疏卷积、深层使用注意力的分层混合架构，并引入无参数的PointROPE位置编码，实现了比Point Transformer V3少3.6倍参数、快2倍、省2倍内存，同时在多个点云基准上性能持平或超越。

**[Long-SCOPE: Fully Sparse Long-Range Cooperative 3D Perception](3d_vision/long_scope_fully_sparse_long_range_cooperative_3d_perception.md)**

:   Long-SCOPE提出了全稀疏的长距离协同3D感知框架，通过几何引导查询生成和上下文感知关联模块，在100-150m远距离场景下实现了SOTA性能，同时保持高效的计算和通信成本。

**[LongStream: Long-Sequence Streaming Autoregressive Visual Geometry](3d_vision/longstream_long-sequence_streaming_autoregressive_visual_geometry.md)**

:   提出LongStream，一种gauge-decoupled的流式视觉几何模型，通过关键帧相对位姿预测、正交尺度学习和缓存一致性训练，实现千帧级别稳定的度量尺度实时（18 FPS）场景重建。

**[LoST: Level of Semantics Tokenization for 3D Shapes](3d_vision/lost_level_of_semantics_tokenization_for_3d_shapes.md)**

:   提出Level-of-Semantics Tokenization (LoST)，按语义显著性排序3D形状token，使短前缀即可解码出完整且语义合理的形状，配合RIDA语义对齐损失和GPT式自回归生成，仅用128个token即显著超越现有需数万token的3D AR方法。

**[LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates](3d_vision/ltgs_long-term_gaussian_scene_chronology_from_sparse_view_updates.md)**

:   提出 LTGS 框架，通过构建可复用的物体级高斯模板，从时空稀疏的观测图像中高效更新 3DGS 场景重建，实现长期环境演化的时序建模。

**[LumiMotion: Improving Gaussian Relighting with Scene Dynamics](3d_vision/lumimotion_gaussian_relighting_dynamics.md)**

:   LumiMotion 是首个利用场景动态（运动区域）作为监督信号来改善逆渲染的 Gaussian-based 方法，通过动静分离和运动揭示的材质变化来更好地分离光照与材质，albedo 估计 LPIPS 提升 23%，重光照提升 15%。

**[M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation](3d_vision/m3dlayout_a_multi-source_dataset_of_3d_indoor_layouts_and_structured_description.md)**

:   构建了多源大规模 3D 室内布局数据集 M3DLayout（21,367 布局、433k+ 物体实例），融合真实扫描、专业设计和程序化生成三种来源，配以结构化文本描述，为文本驱动的 3D 场景生成提供高质量训练基础。

**[MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping](3d_vision/magician_efficient_long-term_planning_with_imagined_gaussians_for_active_mapping.md)**

:   提出MAGICIAN框架，利用预训练占据度网络生成"想象高斯"（Imagined Gaussians）来高效估计表面覆盖增益，结合束搜索实现主动建图中的长期轨迹规划，在室内外场景均达到SOTA，覆盖率提升超10%。

**[Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding](3d_vision/mamba_learns_in_context_structure-aware_domain_generalization_for_multi-task_poi.md)**

:   提出SADG框架，首次将Mamba引入多任务点云域泛化的上下文学习，通过结构感知序列化（质心距离谱+测地曲率谱）、分层域感知建模和谱图对齐三个模块，在重建、去噪、配准三个任务上全面超越SOTA。

**[MARCO: Navigating the Unseen Space of Semantic Correspondence](3d_vision/marco_semantic_correspondence.md)**

:   提出 MARCO，基于单一 DINOv2 backbone 的语义对应模型，通过粗到细高斯 RBF 损失逐步提升空间精度，并用自蒸馏框架将稀疏关键点监督扩展为稠密伪对应标签，在标准基准和未见关键点/类别上均达到 SOTA，同时比双编码器方法小 3 倍、快 10 倍。

**[Masking Matters: Unlocking the Spatial Reasoning Capabilities of LLMs for 3D Scene-Language Understanding](3d_vision/masking_matters_unlocking_the_spatial_reasoning_capabilities_of_llms_for_3d_scen.md)**

:   发现 LLM 解码器中的 causal mask 与 3D 场景理解存在两个根本冲突（顺序偏置和指令隔离），提出 3D-SLIM 掩码策略（Geometry-adaptive Mask + Instruction-aware Mask）替换 causal mask，无需架构修改和额外参数即可在多个 3D 场景语言任务上获得显著提升。

**[Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding](3d_vision/meta-learning_in-context_enables_training-free_cross_subject_brain_decoding.md)**

:   提出 BrainCoDec 框架，通过两阶段层级式上下文学习（先为每个体素估计编码器参数，再跨体素聚合做功能反演），实现了无需微调即可泛化到新被试的 fMRI 视觉解码，Top-1 检索准确率从 MindEye2 的 3.9% 提升到 22.7%。

**[MimiCAT: Mimic with Correspondence-Aware Cascade-Transformer for Category-Free 3D Pose Transfer](3d_vision/mimicat_mimic_with_correspondence-aware_cascade-transformer_for_category-free_3d.md)**

:   本文提出 MimiCAT，一个级联 Transformer 框架，通过语义关键点标签学习柔性多对多软对应关系，结合百万级多类别动作数据集 PokeAnimDB，首次实现了跨类别（如人形到四足动物/鸟类）的高质量 3D 姿态迁移。

**[Modeling Spatiotemporal Neural Frames for High Resolution Brain Dynamics](3d_vision/modeling_spatiotemporal_neural_frames_for_high_resolution_brain_dynamic.md)**

:   提出基于扩散 Transformer 的 EEG 条件 fMRI 重建框架，将脑活动建模为时空神经帧序列而非独立快照，在皮层顶点级分辨率下实现时空一致的 fMRI 重建，并通过零空间采样支持中间帧插值，下游视觉解码任务验证了功能信息的保留。

**[MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](3d_vision/more_motion-aware_feed-forward_4d_reconstruction_transformer.md)**

:   提出 MoRe，一种前馈式运动感知 4D 重建 Transformer，通过注意力强制策略在训练时解耦动态运动与静态结构，结合分组因果注意力实现高效流式推理，在动态场景的相机位姿估计和深度预测上达到 SOTA。

**[MoRel: Long-Range Flicker-Free 4D Motion Modeling via Anchor Relay-based Bidirectional Blending with Hierarchical Densification](3d_vision/morel_long-range_flicker-free_4d_motion_modeling_via_anchor_relay-based_bidirect.md)**

:   针对4D高斯泼溅在长视频动态场景建模中面临的内存爆炸、时序闪烁和遮挡处理等挑战，提出了基于锚点接力双向混合 (ARBB) 的MoRel框架，通过关键帧锚点的渐进式构建和可学习时序不透明度控制实现了无闪烁、内存有界的长程4D运动重建。

**[Motion-Aware Animatable Gaussian Avatars Deblurring](3d_vision/motion-aware_animatable_gaussian_avatars_deblurring.md)**

:   提出首个从模糊视频直接重建清晰可动画3D人体高斯Avatar的方法，通过3D感知的物理模糊形成模型和基于SMPL的人体运动模型，联合优化Avatar表示和运动参数。

**[MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](3d_vision/motionanymesh_physics-grounded_articulation_for_simulation-ready_digital_twins.md)**

:   提出MotionAnymesh，一个零样本自动框架，通过运动感知分割（SP4D先验+VLM推理）和几何-物理联合优化关节估计，将静态3D网格转化为无碰撞的仿真就绪铰接数字孪生，在PartNet-Mobility和Objaverse上物理可执行性达87%。

**[MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting](3d_vision/motionscale_reconstructing_appearance_geometry_and_motion_of_dynamic_scenes_with.md)**

:   提出 MotionScale，一个可扩展的 4D 高斯泼溅框架，通过基于聚类的自适应运动场和渐进式优化策略，从单目视频中高保真重建大规模动态场景的外观、几何和运动，在 DyCheck 上 PSNR 达到 17.98，3D 跟踪 EPE 降至 0.070，显著超越现有方法。

**[MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second](3d_vision/movies_motion-aware_4d_dynamic_view_synthesis_in_one_second.md)**

:   提出 MoVieS，一个前馈式 4D 动态场景重建框架，通过 **动态溅射像素 (Dynamic Splatter Pixel)** 表示将外观、几何和运动统一建模，从单目视频在约 1 秒内完成 4D 重建，并支持新视角合成、3D 点跟踪、场景流估计和运动物体分割等多种任务。

**[MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation](3d_vision/msgnav_unleashing_the_power_of_multi-modal_3d_scene_graph_for_zero-shot_embodied.md)**

:   提出多模态3D场景图（M3DSG），用动态分配的图像边替代传统文本关系边来保留视觉信息，构建零样本导航系统MSGNav，并提出可见性视点决策模块解决导航"最后一公里"问题，在GOAT-Bench和HM3D-ObjNav上取得SOTA。

**[MV-RoMa: From Pairwise Matching into Multi-View Track Reconstruction](3d_vision/mv-roma_from_pairwise_matching_into_multi-view_track_reconstruction.md)**

:   提出 MV-RoMa，首个多视图稠密匹配模型，通过 Track-Guided 多视图编码器和像素对齐多视图精炼器从一张源图同时估计到多个目标图的稠密对应关系，产生几何一致的轨迹用于 SfM，在 HPatches/ETH3D/IMC 等基准上全面超越现有方法。

**[MVGGT: Multimodal Visual Geometry Grounded Transformer for Multiview 3D Referring Expression Segmentation](3d_vision/mvggt_multimodal_visual_geometry_grounded_transformer_for_multiview_3d_referring.md)**

:   提出 MV-3DRES 新任务（从稀疏多视图 RGB 直接做语言引导的 3D 分割）和 MVGGT 框架（双分支设计融合冻结几何分支 + 可训练多模态分支），通过 PVSO 优化策略解决前景梯度稀释问题，在自建 MVRefer 基准上以 39.9 mIoU 大幅超越基线。

**[NanoSD: Edge Efficient Foundation Model for Real Time Image Restoration](3d_vision/nanosd_edge_efficient_foundation_model_for_real_time_image_restoration.md)**

:   提出 NanoSD，通过对 SD 1.5 进行硬件感知的 U-Net 分解、逐块特征蒸馏和多目标贝叶斯优化，构建了一族 Pareto 最优的轻量扩散基础模型（130M–315M 参数，最快 12ms 推理），可作为 drop-in backbone 在超分、人脸修复、去模糊、单目深度估计等多任务上达到 SOTA 级表现。

**[NeAR: Coupled Neural Asset–Renderer Stack](3d_vision/near_coupled_neural_asset-renderer_stack.md)**

:   NeAR 提出将神经资产创作和神经渲染联合设计为一个耦合栈，通过光照均匀化的结构化 3D 潜变量（LH-SLAT）消除输入图像中的烘焙光照，再用光照感知的神经解码器实时合成可重光照的 3D 高斯场，在前向渲染、重建、重光照和新视角重光照四类任务上超越现有方法。

**[Neu-PiG: Neural Preconditioned Grids for Fast Dynamic Surface Reconstruction on Long Sequences](3d_vision/neu-pig_neural_preconditioned_grids_for_fast_dynamic_surface_reconstruction_on_l.md)**

:   Neu-PiG 提出一种基于预条件多分辨率潜在网格的快速优化方法，将关键帧参考网格的位置和法线方向编码为统一潜在空间，通过轻量级 MLP 解码为每帧 6-DoF 形变，在无需类别先验或显式对应关系的前提下，实现了比现有无训练方法快 60 倍以上的高保真动态曲面重建。

**[Neural Field-Based 3D Surface Reconstruction of Microstructures from Multi-Detector Signals in Scanning Electron Microscopy](3d_vision/neural_field-based_3d_surface_reconstruction_of_microstructures_from_multi-detec.md)**

:   本文提出 NFH-SEM，一个基于神经场的混合框架，通过将 SEM 电子散射物理模型嵌入神经场优化过程，从多视角多检测器 SEM 图像重建高保真的微观结构 3D 表面，实现了自标定、抗阴影的纳米级精度重建（478nm 层叠特征、782nm 花粉纹理、1.559μm 断裂台阶）。

**[Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction](3d_vision/neural_gabor_splatting.md)**

:   Neural Gabor Splatting 为每个高斯原语嵌入一个轻量级 MLP（SIREN 架构），使单个原语能表示复杂的空间变化颜色模式，配合频率感知致密化策略，在相同数据预算下显著提升高频表面重建质量。

**[NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation](3d_vision/ng_gs_nerf_guided_3d_gaussian_splatting_segmentation.md)**

:   提出 NG-GS 框架，利用 NeRF 的连续建模能力解决 3DGS 分割中的边界离散化问题，通过 RBF 插值构建连续特征场结合多分辨率哈希编码和 NeRF-GS 联合优化实现高质量对象分割。

**[NI-Tex: Non-isometric Image-based Garment Texture Generation](3d_vision/ni-tex_non-isometric_image-based_garment_texture_generation.md)**

:   提出NI-Tex框架，通过构建3D Garment Videos数据集、基于图像编辑的跨拓扑增强以及不确定性引导的迭代烘焙算法，首次以前馈架构实现了非等距条件下从单图到3D服装PBR纹理的高质量生成。

**[NimbusGS: Unified 3D Scene Reconstruction under Hybrid Weather](3d_vision/nimbusgs_unified_3d_scene_reconstruction_under_hybrid_weather.md)**

:   NimbusGS 提出统一的3D场景重建框架，通过将天气退化分解为连续散射场（雾/霾）和逐视图粒子残差层（雨/雪），配合几何引导梯度缩放机制，在单一框架内实现跨天气和混合天气条件下的SOTA重建。

**[No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D Consistency](3d_vision/no_calibration_no_depth_no_problem_cross-sensor_view_synthesis_with_3d_consisten.md)**

:   提出首个无需标定和深度的跨传感器视图合成框架，通过匹配-稠密化-3D整合 (match-densify-consolidate) 流程，将稀疏跨模态关键点扩展为稠密的、与 RGB 视角对齐的 X 模态图像（热成像/NIR/SAR），并通过置信度感知融合与自匹配过滤提升合成质量。

**[Node-RF: Learning Generalized Continuous Space-Time Scene Dynamics with Neural ODE-based NeRFs](3d_vision/node-rf_learning_generalized_continuous_space-time_scene_dynamics_with_neural_od.md)**

:   Node-RF 将 Neural ODE 与 NeRF 紧密耦合，用连续时间微分方程驱动隐式场景表征的时序演化，实现了远超训练时域的长程外推与跨轨迹泛化，在 Bouncing Balls、Pendulum、Oscillating Ball 等数据集上显著优于 D-NeRF、4D-GS 等基线。

**[NTK-Guided Implicit Neural Teaching](3d_vision/ntk-guided_implicit_neural_teaching.md)**

:   提出 NINT，利用 Neural Tangent Kernel (NTK) 的行向量来度量每个坐标对全局函数更新的影响力，从而动态选择既有高拟合误差又有高全局影响力的坐标进行训练，将 INR 训练时间减少近一半且不损失重建质量。

**[Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](3d_vision/off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)**

:   本文提出一种基于关键点检测思路的前馈式3DGS解码器，将高斯原语从像素网格中解放出来，在亚像素级别自适应放置原语，结合自适应密度机制和置信度剪枝，仅使用输入像素数1/7的原语就在新视角合成上超越了SOTA前馈方法。

**[OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting](3d_vision/onlinepg_online_open-vocabulary_panoptic_mapping_with_3d_gaussian_splatting.md)**

:   提出 OnlinePG，首个基于 3DGS 的在线开放词汇全景建图系统，通过 local-to-global 范式——在滑窗内用多线索聚类图（几何重叠+语义相似+视图共识）构建局部一致 3D 实例，再通过双向二部匹配增量融合到全局地图——实现了在线方法中最优的语义和全景分割性能，ScanNet 上 mIoU 48.48 超越 OnlineAnySeg +17.2，且达到 10-18 FPS 实时效率。

**[OpenVO: Open-World Visual Odometry with Temporal Dynamics Awareness](3d_vision/openvo_open-world_visual_odometry_with_temporal_dynamics_awareness.md)**

:   提出 OpenVO，一个面向开放世界的单目视觉里程计框架，通过时间感知流编码器和几何感知上下文编码器，在无相机标定、帧率变化的条件下实现鲁棒的真实尺度自车运动估计，跨数据集 ATE 提升超 20%，变帧率场景误差降低 46%-92%。

**[PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery](3d_vision/pad-hand_physics-aware_diffusion_for_hand_motion_recovery.md)**

:   提出 PAD-Hand，一个物理感知的条件扩散框架，将欧拉-拉格朗日动力学残差建模为虚拟观测量融入扩散过程，同时通过最后一层拉普拉斯近似估计逐关节、逐帧的动态方差，实现了兼具物理可信度和不确定性感知的手部运动恢复，在 DexYCB 上加速度误差降低 50.1%。

**[Pano360: Perspective to Panoramic Vision with Geometric Consistency](3d_vision/pano360_perspective_to_panoramic_vision_with_geometric_consistency.md)**

:   提出 Pano360，将全景拼接从传统的 2D 逐对匹配扩展到 3D 摄影测量空间，利用 Transformer 架构实现多视图全局几何一致性对齐，在弱纹理、大视差、重复纹理等挑战场景下达到 97.8% 成功率。

**[Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](3d_vision/pano3dcomposer_feed-forward_compositional_3d_scene_generation_from_single_panora.md)**

:   提出 Pano3DComposer，一个从单张全景图出发的模块化前馈式组合3D场景生成框架，通过即插即用的 Object-World Transformation Predictor（基于 Alignment-VGGT）将生成的3D物体从局部坐标转换到世界坐标，约20秒即可在 RTX 4090 上生成高保真3D场景。

**[PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery](3d_vision/panovggt_feed-forward_3d_reconstruction_from_panoramic_imagery.md)**

:   提出 PanoVGGT，一个置换等变的 Transformer 框架，能从一张或多张无序全景图像中在单次前馈中联合预测相机位姿、深度图和全局一致3D点云；同时贡献了 PanoCity——一个包含超过12万张室外全景图像的大规模数据集。

**[Parallelised Differentiable Straightest Geodesics for 3D Meshes](3d_vision/parallelised_differentiable_straightest_geodesics_for_3d_meshes.md)**

:   提出 straightest geodesics 的并行 GPU 实现及两种可微分方案（外在代理函数法和测地线有限差分法），使三角网格上的指数映射可高效并行且可微分，并以此构建测地线卷积层、网格上的流匹配方法和二阶优化器三个下游应用。

**[Particulate: Feed-Forward 3D Object Articulation](3d_vision/particulate_feed-forward_3d_object_articulation.md)**

:   Particulate 提出了一个前馈式模型，给定静态 3D 网格即可在数秒内推断出完整的铰接结构（部件分割、运动学树、运动约束），基于 Part Articulation Transformer 在公开数据集上端到端训练，显著优于需要逐物体优化的现有方法，并能与 3D 生成模型结合实现从单张图像到铰接 3D 物体的生成。

**[PCSTracker: Long-Term Scene Flow Estimation for Point Cloud Sequences](3d_vision/pcstracker_long-term_scene_flow_estimation_for_point_cloud_sequences.md)**

:   PCSTracker 是首个端到端的点云序列长程场景流估计框架，通过迭代几何-运动联合优化、时空轨迹更新和重叠滑动窗口策略，在合成数据集 PointOdyssey3D 上将 EPE_3D 降低 57.9%，并以 32.5 FPS 实时运行。

**[PE3R: Perception-Efficient 3D Reconstruction](3d_vision/pe3r_perception-efficient_3d_reconstruction.md)**

:   PE3R 提出一个免调优的前馈式3D语义重建框架，通过像素嵌入消歧、语义点云重建和全局视图感知三个模块，从无位姿的2D图像直接生成语义3D点云，实现了9倍加速且在开放词汇分割和深度估计上达到新SOTA。

**[PhyGaP: Physically-Grounded Gaussians with Polarization Cues](3d_vision/phygap_physically-grounded_gaussians_with_polarization_cues.md)**

:   提出 PhyGaP，通过偏振延迟渲染（PolarDR）将偏振线索融入 2DGS 优化，并设计自遮挡感知的 GridMap 环境图技术，实现光泽物体的精确反射分解与真实重光照。

**[PhysGaia: A Physics-Aware Benchmark with Multi-Body Interactions for Dynamic Novel View Synthesis](3d_vision/physgaia_a_physics-aware_benchmark_with_multi-body_interactions_for_dynamic_nove.md)**

:   PhysGaia 构建了一个包含 17 个场景的物理感知基准数据集，涵盖液体/气体/织物/流变物质等多种材料的多体交互，提供 3D 粒子轨迹和物理参数（如粘度）的 ground truth，并提出 Trajectory Distance (TD) 和 AUOP 两个新指标来量化 4DGS 方法的物理真实性，揭示了现有 DyNVS 方法在物理推理上的严重不足。

**[PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis](3d_vision/physgm_large_physical_gaussian_4d_synthesis.md)**

:   首个从单张图像前馈预测3DGS+物理属性（材质类别/杨氏模量/泊松比）的框架，两阶段训练（监督预训练+DPO偏好微调）完全绕过SDS和可微物理引擎，配合50K+ PhysAssets数据集，1分钟内生成高保真4D物理仿真，CLIP_sim和人类偏好率均超越逐场景优化方法。

**[PhysGS: Bayesian-Inferred Gaussian Splatting for Physical Property Estimation](3d_vision/physgs_bayesian-inferred_gaussian_splatting_for_physical_property_estimation.md)**

:   提出 PhysGS，将贝叶斯推断嵌入3D高斯溅射管线，利用视觉-语言模型先验和多视角置信度加权更新，实现逐点物理属性（摩擦力、硬度、密度、刚度）的概率估计与不确定性量化，在质量估计上比 NeRF2Physics 提升 22.8%（APE），岸氏硬度误差降低 61.2%。

**[PhysHead: Simulation-Ready Gaussian Head Avatars](3d_vision/physhead_simulation-ready_gaussian_head_avatars.md)**

:   提出PhysHead——首个将物理驱动头发动力学与可动画3DGS头部Avatar结合的方法：用FLAME网格+3DGS建模可表达面部、用发丝(strand)+3DGS建模头发外观、用物理引擎驱动头发动画，并通过VLM生成秃头图像实现头发与面部的分层优化。

**[Physically Inspired Gaussian Splatting for HDR Novel View Synthesis](3d_vision/physically_inspired_gaussian_splatting_for_hdr_novel_view_synthesis.md)**

:   提出PhysHDR-GS——一个物理渲染启发的HDR新视角合成框架：将高斯颜色分解为固有反射率和可调环境光照，通过图像-曝光(IE)分支和高斯-光照(GI)分支互补捕获HDR细节，跨分支HDR一致性损失提供无GT的显式HDR监督，光照引导梯度缩放解决曝光偏差的梯度饥饿问题，在多个基准上优于HDR-GS 2.04dB且保持76FPS实时渲染。

**[PIP-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching](3d_vision/pip-stereo_progressive_iterations_pruner_for_iterative_optimization_based_stereo.md)**

:   揭示迭代立体匹配中视差更新的空间稀疏性和时间冗余性，提出渐进迭代裁剪（PIP）将32次迭代压缩到1次、协同学习范式实现无需独立单目编码器的深度先验迁移、以及硬件感知的 FlashGRU 算子（7.28× 加速），使高精度迭代立体匹配首次在 Jetson Orin NX 上实现实时推理（75ms/帧，320×640）。

**[PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction](3d_vision/pixarmesh_autoregressive_mesh-native_single-view_scene_reconstruction.md)**

:   提出 PixARMesh，首个在原生 mesh 空间（而非 SDF）中进行单视图场景重建的自回归框架，通过像素对齐图像特征和全局场景上下文增强点云编码器，在统一的 token 序列中同时预测物体位姿和mesh，在 3D-FRONT 上达到场景级 SOTA 且输出紧凑、可编辑的 artist-ready mesh。

**[PointINS: Instance-Aware Self-Supervised Learning for Point Clouds](3d_vision/pointins_instance-aware_self-supervised_learning_for_point_clouds.md)**

:   PointINS 提出首个显式学习语义一致性和几何推理的点云自监督框架，通过无标签的偏移分支配合偏移分布正则化（ODR）和空间聚类正则化（SCR），在室内实例分割上平均提升 +3.5% mAP，室外全景分割提升 +4.1% PQ。

**[PointTPA: Dynamic Network Parameter Adaptation for 3D Scene Understanding](3d_vision/pointtpa_dynamic_network_parameter_adaptation_for_3d_scene_understanding.md)**

:   提出 PointTPA 框架，通过序列化邻域分组（SNG）和动态参数投射器（DPP）两个轻量模块在推理时为每个输入场景生成定制化的网络参数，仅增加 <2% 参数量即在 ScanNet 上达到 78.4% mIoU，超越现有参数高效微调（PEFT）方法。

**[PoseMaster: A Unified 3D Native Framework for Stylized Pose Generation](3d_vision/posemaster_a_unified_3d_native_framework_for_stylized_pose_generation.md)**

:   PoseMaster 提出了一个将姿态风格化与 3D 生成统一在端到端框架中的 3D 原生方法，直接使用 3D 骨骼作为姿态控制信号（而非 2D 骨骼图），设计了骨骼稠密化策略和 Point Transformer 编码器提取精细的空间拓扑特征，并通过大规模"Image-Skeleton-Mesh"三元组数据引擎训练，在姿态规范化和任意姿态风格化上达到 SOTA。

**[PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](3d_vision/pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)**

:   本文提出 PR-IQA，一种跨参考图像质量评估方法，先在多视图重叠区域计算几何一致的局部质量图，再通过参考条件化的交叉注意力网络将质量信息"补全"到非重叠区域，生成逼近全参考精度的密集质量图，集成到 3DGS 流水线中通过双重过滤策略显著提升稀疏视角 3D 重建质量。

**[ProgressiveAvatars: Progressive Animatable 3D Gaussian Avatars](3d_vision/progressiveavatars_progressive_animatable_3d_gaussian_avatars.md)**

:   提出 ProgressiveAvatars，一种基于模板网格自适应隐式细分构建层级3DGS的渐进式头像表示，支持在不同带宽和算力约束下渐进传输和渲染——仅传输5%数据（2.6MB）即可获得可用头像，后续增量加载平滑提升质量至与 SOTA 方法可比。

**[PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts](3d_vision/promptstereo_zero-shot_stereo_matching_via_structure_and_motion_prompts.md)**

:   提出 Prompt Recurrent Unit (PRU)，将单目深度基础模型的 DPT 解码器作为迭代精炼模块（替代 GRU），通过 Structure Prompt 和 Motion Prompt 将单目结构和立体运动线索以残差方式注入，在不破坏单目先验的情况下实现零样本 SOTA 立体匹配（Middlebury 2021 上误差降低近50%）。

**[Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives](3d_vision/prune_wisely_reconstruct_sharply_compact_3d_gaussian_splatting_via_adaptive_prun.md)**

:   提出自适应重建感知剪枝策略（RPS）和 3D DoG 原语，在保持渲染质量的同时实现 90% 的高斯点裁减。

**[QD-PCQA: Quality-Aware Domain Adaptation for Point Cloud Quality Assessment](3d_vision/qd-pcqa_quality-aware_domain_adaptation_for_point_cloud_quality_assessment.md)**

:   提出质量感知域适应框架 QD-PCQA，通过 Rank-weighted Conditional Alignment 和 Quality-guided Feature Augmentation 两大策略，将图像域的质量评估先验迁移到点云域。

**[QuadSync: Quadrifocal Tensor Synchronization via Tucker Decomposition](3d_vision/quadsync_quadrifocal_tensor_synchronization_via_tucker_decomposition.md)**

:   首次提出四焦张量(quadrifocal tensor)的全局同步算法 QuadSync，通过构造块四焦张量并证明其承认多线性秩为 (4,4,4,4) 的 Tucker 分解，利用 ADMM-IRLS 优化框架从四视图测量中恢复相机位姿，在密集视图场景下取得优于两视图/三视图方法的同步精度。

**[R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection# R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](3d_vision/r4det_4d_radar_camera_fusion_3d_detection.md)**

:   提出R4Det，通过全景深度融合（PDF）、可变形门控时序融合（DGTF）和实例引导动态精炼（IGDR）三个即插即用模块，解决4D雷达-相机融合中深度估计不准、时序融合依赖ego pose、小目标检测困难的问题，在TJ4DRadSet和VoD上取得SOTA。

**[Random Wins All: Rethinking Grouping Strategies for Vision Tokens](3d_vision/random_wins_all_rethinking_grouping_strategies_for_vision_tokens.md)**

:   提出极简的随机分组策略替代 Vision Transformer 中各种精心设计的 token 分组方法，在图像分类、目标检测、语义分割、点云分割和 VLM 上几乎全面超越所有 baseline，并从位置信息、头特征多样性、全局感受野和固定分组模式四个维度解释了随机分组成功的原因。

**[RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing](3d_vision/rap_fast_feedforward_rendering-free_attribute-guided_primitive_importance_score_.md)**

:   提出 RAP，一种无需渲染的前馈式高斯原语重要性评分方法，通过从内在属性和局部邻域统计量提取 15 维特征，用轻量 MLP 预测重要性评分，训练一次即可泛化到未见场景。

**[RayNova: Scale-Temporal Autoregressive World Modeling in Ray Space](3d_vision/raynova_scale-temporal_autoregressive_world_modeling_in_ray_space.md)**

:   提出 RayNova，一种基于双因果（尺度+时间）自回归的几何无关多视角世界模型，利用相对 Plücker 光线位置编码实现统一的 4D 时空推理，在 nuScenes 上取得 SOTA 多视角视频生成效果。

**[Real2Edit2Real: Generating Robotic Demonstrations via a 3D Control Interface](3d_vision/real2edit2real_generating_robotic_demonstrations_via_a_3d_control_interface.md)**

:   提出 Real2Edit2Real 框架，通过"3D 重建 → 点云编辑生成新轨迹 → 深度引导视频生成合成演示"的三阶段管线，从仅 1-5 个真实演示生成大量多样化的操作演示数据，使策略性能达到甚至超过 50 个真实演示训练的水平，数据效率提升 10-50 倍。

**[Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron Computed Tomography Data](3d_vision/regularizing_inr_with_diffusion_prior_self-supervised_3d_reconstruction_of_neutr.md)**

:   提出 DINR (Diffusive INR)，在 DD3IP 扩散框架内用 INR 替代传统反演求解器，通过近端损失将扩散去噪估计注入 INR 优化过程，在极端稀疏视角（低至 4-5 视图）的中子 CT 重建中超越现有 SOTA 方法。

**[ReLaGS: Relational Language Gaussian Splatting](3d_vision/relags_relational_language_gaussian_splatting.md)**

:   提出首个统一多层级语言高斯场与开放词汇3D场景图的无训练框架 ReLaGS，通过最大权重剪枝和鲁棒异常值感知特征聚合改进场景表示，结合GNN关系预测实现高效的结构化3D场景理解。

**[Reliev3R: Relieving Feed-forward 3D Reconstruction from Multi-View Geometric Annotations](3d_vision/reliev3r_relieving_feed-forward_3d_reconstruction_from_multi-view_geometric_annot.md)**

:   Reliev3R 首次提出无需多视图几何标注（无需 SfM/MVS 生成的点云和位姿）即可从头训练前馈3D重建模型（FFRM）的弱监督范式，利用单目相对深度和稀疏图像对应作为替代监督，性能追平甚至超过部分全监督 FFRM。

**[Reparameterized Tensor Ring Functional Decomposition for Multi-Dimensional Data Recovery](3d_vision/reparameterized_tensor_ring_functional_decomposition_for_multi-dimensional_data_.md)**

:   提出 RepTRFD：通过将 Tensor Ring 因子重参数化为"可学习隐张量 × 固定基"的形式，解决 INR 参数化 TR 因子的频谱偏置问题，在图像修复/去噪/超分/点云恢复等任务上全面超越 SOTA。

**[Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty](3d_vision/rethinking_pose_refinement_in_3d_gaussian_splatting_under_pose_prior_and_geometr.md)**

:   提出 UGS-Loc 框架，通过蒙特卡洛位姿采样和 Fisher 信息引导的 PnP 优化，联合建模位姿先验不确定性和几何不确定性，在无需重训练的条件下显著提升 3DGS 场景中的相机位姿精化鲁棒性。

**[RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](3d_vision/retimegs_continuous-time_reconstruction_of_4d_gaussian_splatting.md)**

:   提出 RetimeGS，通过正则化时间不透明度 + Catmull-Rom 样条轨迹 + 双向光流监督 + 三重渲染等策略，解决 4DGS 在离散帧间插值时的鬼影/时间别名问题，实现任意时间戳的无鬼影连续时间 4D 重建。

**[ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction](3d_vision/reweaver_towards_simulation-ready_and_topology-accurate_garment_reconstruction.md)**

:   提出 ReWeaver 框架，从最少4张多视图RGB图像中联合重建3D服装几何与2D缝纫图案（sewing pattern），通过双路径Transformer预测3D曲面片/曲线及其拓扑连接，再经组内注意力将3D结构展平为2D面板边缘，首次实现拓扑准确且可直接用于物理仿真的服装资产恢复。

**[Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](3d_vision/rewis3d_reconstruction_improves_weaklysupervised_s.md)**

:   提出 Rewis3d 框架，首次将前馈式 3D 场景重建作为辅助监督信号整合到弱监督语义分割中，通过双学生-教师架构和双置信度加权的跨模态一致性损失，在仅有稀疏标注的情况下将 mIoU 提升 2-7%，且推理时仅使用 2D 图像。

**[RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations](3d_vision/rng_a_unified_transformer_for_complete_3d_modeling_from_partial_observations.md)**

:   RnG 提出重构引导因果注意力（Reconstruction-Guided Causal Attention），将 Transformer 的 KV-Cache 重新解释为隐式 3D 表示，用单个前馈 Transformer 统一完成从无位姿稀疏图像到完整 3D 几何与外观的重建与生成，速度比扩散方法快 100 倍以上。

**[S2AM3D: Scale-controllable Part Segmentation of 3D Point Clouds](3d_vision/s2am3d_scale-controllable_part_segmentation_of_3d_point_cloud.md)**

:   提出融合2D预训练先验与3D对比监督的点云部件分割框架S2AM3D，通过点一致性编码器获得全局一致的点特征，并设计尺度感知提示解码器实现连续可控的分割粒度调节，在多个基准上大幅超越现有方法。

**[Sampling-Aware 3D Spatial Analysis in Multiplexed Imaging](3d_vision/sampling-aware_3d_spatial_analysis_in_multiplexed_imaging.md)**

:   本文系统研究了多重成像中采样几何（2D切片 vs 3D序列切片）对空间统计量恢复精度的影响，并提出了一种几何感知的稀疏3D重建模块，在有限的成像预算下实现可靠的深度感知空间分析。

**[SASNet: Spatially-Adaptive Sinusoidal Networks for INRs](3d_vision/sasnet_spatially_adaptive_sinusoidal_networks_for_inrs.md)**

:   提出 SASNet，通过将冻结的频率嵌入层与轻量级哈希网格 MLP 学习的空间自适应掩码相结合，解决 SIREN 中频率初始化敏感和高频泄漏问题，在图像拟合、体数据拟合和 SDF 重建任务上实现更快收敛和更高重建质量。

**[Scalable Object Relation Encoding for Better 3D Spatial Reasoning in Large Language Models](3d_vision/scalable_object_relation_encoding_for_better_3d_spatial_reasoning_in_large_langu.md)**

:   提出 QuatRoPE，一种基于四元数旋转的3D位置编码方法，仅需 $O(n)$ 输入token即可保留所有 $O(n^2)$ 物体间空间关系，并配合 IGRE 机制减少与语言 RoPE 的干扰，在多个3D视觉语言基准上取得大幅提升。

**[Scaling View Synthesis Transformers (SVSM)](3d_vision/scaling_view_synthesis_transformers.md)**

:   首次为无几何先验的 NVS Transformer 建立缩放定律：提出有效批量大小假设（B_eff = B·V_T）揭示 encoder-decoder 被低估的根因，设计单向 encoder-decoder 架构 SVSM，在 RealEstate10K 上以不到一半训练 FLOPs 达到新 SOTA（30.01 PSNR），Pareto 前沿比 LVSM decoder-only 左移 3×。

**[Scene Grounding In the Wild](3d_vision/scene_grounding_in_the_wild.md)**

:   提出一种基于语义特征的逆优化框架，将野外拍摄的局部3D重建（SfM）对齐到完整的伪合成参考模型（如Google Earth Studio），通过DINOv2特征和鲁棒优化解决巨大的域差异问题，实现非重叠局部重建的全局一致性融合。

**[SceneScribe-1M: A Large-Scale Video Dataset with Comprehensive Geometric and Semantic Annotations](3d_vision/scenescribe-1m_a_large-scale_video_dataset_with_comprehensive_geometric_and_sema.md)**

:   提出SceneScribe-1M——一个包含100万个野外视频、超4000小时的大规模多模态视频数据集，提供详细文本描述、精确相机参数、连续深度图和一致性3D点轨迹等全面标注，为3D几何感知和视频生成任务提供统一资源。

**[SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](3d_vision/scope_scenecontextualized_incremental_fewshot_3d_s.md)**

:   提出即插即用的SCOPE框架，利用类无关分割模型从基础训练场景的背景区域挖掘伪实例原型，通过检索+注意力融合增强few-shot新类原型，无需重训backbone即可在ScanNet上将新类IoU提升6.98%。

**[SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation](3d_vision/seethrough3d_occlusion_aware_3d_control_in_text-to-image_generation.md)**

:   提出 SeeThrough3D，通过半透明 3D 包围盒渲染的遮挡感知场景表示（OSCR）来条件化 FLUX 模型，实现了精确的 3D 布局控制与遮挡一致的文本到图像生成。

**[SEPatch3D: Revisiting Token Compression for Accelerating ViT-based Sparse Multi-View 3D Object Detectors](3d_vision/sepatch3d_revisiting_token_compression_for_accelerating_vit_based_sparse_3d_detectors.md)**

:   提出 SEPatch3D，通过时空感知的动态 patch 大小选择和基于熵的信息性 patch 筛选增强机制，在 ViT 基稀疏多视角 3D 检测中实现 57% 推理加速且保持可比检测精度。

**[SGAD-SLAM: Splatting Gaussians at Adjusted Depth for Better Radiance Fields in RGBD SLAM](3d_vision/sgad-slam_splatting_gaussians_at_adjusted_depth_for_better_radiance_fields_in_rg.md)**

:   提出SGAD-SLAM，采用像素对齐的简化高斯表示并允许高斯沿射线调整深度偏移以提升渲染质量和可扩展性，同时引入基于几何相似度的GICP跟踪策略加速相机位姿估计，在Replica、TUM、ScanNet和ScanNet++上全面超越最新方法。

**[SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation](3d_vision/sgi_structured_2d_gaussians_for_efficient_and_compact_large_image_representation.md)**

:   SGI 提出基于种子点(seed)的结构化 2D 高斯表示框架，通过将无结构高斯原语组织为种子驱动的神经高斯、结合上下文引导的熵编码和多尺度拟合策略，在高分辨率图像表示中实现最高 7.5× 压缩比和 6.5× 优化加速，同时保持甚至提升重建保真度。

**[SGS-Intrinsic: Semantic-Invariant Gaussian Splatting for Sparse-View Indoor Inverse Rendering](3d_vision/sgs-intrinsic_semantic-invariant_gaussian_splatting_for_sparse-view_indoor_invers.md)**

:   SGS-Intrinsic 提出两阶段室内逆渲染框架，第一阶段利用语义和几何先验构建稠密几何一致的高斯场，第二阶段结合混合光照模型和材质先验进行材质-光照分解，并通过去阴影模块防止阴影烘焙到反照率中。

**[Sky2Ground: A Benchmark for Site Modeling under Varying Altitude](3d_vision/sky2ground_a_benchmark_for_site_modeling_under_varying_altitude.md)**

:   本文提出Sky2Ground数据集（51个场景，80k图像，统一覆盖卫星/航拍/地面三种视角的合成+真实图像）和SkyNet模型（双流编码器+掩码卫星注意力+渐进式视角采样），首次系统研究了跨地面/航拍/卫星三视角联合相机定位问题，在RRA@5上提升9.6%，在RTA@5上提升18.1%。

**[SonoWorld: From One Image to a 3D Audio-Visual Scene](3d_vision/sonoworld_from_one_image_to_a_3d_audio-visual_scene.md)**

:   提出 SonoWorld，一个 training-free 的框架，可以从单张图片出发，生成可探索的3D音频-视觉场景：先将图片扩展为360°全景并重建为3D高斯场景，再通过VLM驱动的语义定位放置声源锚点，最后用 Ambisonics 编码渲染空间音频，实现视觉与听觉的几何和语义对齐。

**[SoPE: Spherical Coordinate-Based Positional Embedding for Enhancing Spatial Perception of 3D LVLMs](3d_vision/sope_spherical_coordinate-based_positional_embedding_for_enhancing_spatial_perce.md)**

:   提出球坐标位置编码 SoPE，将点云 token 从一维序列索引重映射到球坐标 $(t,r,\theta,\phi)$ 空间，并配合多维频率分配与多尺度频率混合策略，显著增强 3D 大视觉-语言模型的空间感知能力。

**[SPAN: Spatial-Projection Alignment for Monocular 3D Object Detection](3d_vision/span_spatial_projection_alignment_mono3d.md)**

:   提出 SPAN 即插即用几何协同约束框架，通过 Spatial Point Alignment（3D角点MGIoU对齐）和 3D-2D Projection Alignment（投影包围矩形GIoU对齐）两个可微损失，强制解耦预测的各属性满足全局几何一致性，配合 Hierarchical Task Learning 策略确保训练稳定，在 KITTI 上将 MonoDGP 的 Car Moderate AP3D 提升 0.92% 达到新 SOTA，推理零额外开销。

**[Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting](3d_vision/spectral_defense_against_resource-targeting_attack_in_3d_gaussian_splatting.md)**

:   提出首个针对 3DGS 资源耗尽攻击的频域防御框架，通过 3D 频率滤波器选择性剪枝异常高频高斯 + 2D 频谱正则化约束渲染图像的各向异性噪声，在攻击下将高斯过生长抑制最高 5.92×、显存降低最高 3.66×、渲染加速最高 4.34×，同时保持重建质量。

**[Speed3R: Sparse Feed-forward 3D Reconstruction Models](3d_vision/speed3r_sparse_feed-forward_3d_reconstruction_models.md)**

:   Speed3R 为 feed-forward 3D重建模型设计了可训练的双分支全局稀疏注意力机制（GSA），通过压缩分支提供粗粒度场景摘要、选择分支聚焦关键 token 精细注意力，在1000视图序列上实现 **12.4倍推理加速**，同时仅引入微小精度下降。

**[Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists](3d_vision/speeding_up_the_learning_of_3d_gaussians_with_much_shorter_gaussian_lists.md)**

:   通过定期重置高斯尺度（Scale Reset）和对 alpha blending 权重施加熵约束（Entropy Constraint），缩短每个像素的高斯列表长度，实现 3DGS 训练 **5-12 倍加速**，同时保持可比的渲染质量。

**[SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)**

:   将3D超分辨率(3DSR)重新定义为从稀疏低分辨率视图到高分辨率3DGS的**前馈映射**问题，通过高斯偏移学习和特征精炼实现高保真HR 3DGS重建，无需逐场景优化即可实现强零样本泛化。

**[STAC: Plug-and-Play Spatio-Temporal Aware Cache Compression for Streaming 3D Reconstruction](3d_vision/stac_plug-and-play_spatio-temporal_aware_cache_compression_for_streaming_3d_reco.md)**

:   提出STAC框架，利用因果Transformer中KV缓存的时空稀疏性，通过工作时序token缓存、长期空间token缓存和分块多帧优化三个模块，在不需要额外训练的情况下将流式3D重建的内存消耗降低约10倍、推理速度提升4倍，同时几乎不损失重建质量。

**[STAvatar: Soft Binding and Temporal Density Control for Monocular 3D Head Avatars Reconstruction](3d_vision/stavatar_soft_binding_and_temporal_density_control_for_monocular_3d_head_avatars.md)**

:   提出 STAvatar，通过 UV 自适应软绑定框架和时序自适应密度控制策略，从单目视频重建高保真可驱动的 3D 头部化身，在遮挡区域（口腔内部、眼睑）和精细细节方面显著优于现有方法。

**[Stepper: Stepwise Immersive Scene Generation with Multiview Panoramas](3d_vision/stepper_stepwise_immersive_scene_generation_with_multiview_panoramas.md)**

:   提出 Stepper 框架，通过逐步生成多视角全景图并结合前馈式3D重建管线，实现文本驱动的高保真沉浸式3D场景生成，在PSNR上比现有方法平均提升3.3 dB。

**[STS-Mixer: Spatio-Temporal-Spectral Mixer for 4D Point Cloud Video Understanding](3d_vision/sts_mixer_4d_point_cloud.md)**

:   STS-Mixer 首次将图傅里叶变换（GFT）引入 4D 点云视频理解，通过频域分解捕获不同尺度的几何结构（低频=全局形状、高频=局部细节），与时空信息混合后在动作识别和语义分割上达到 SOTA。

**[SwiftTailor: Efficient 3D Garment Generation with Geometry Image Representation](3d_vision/swifttailor_efficient_3d_garment_generation_with_geometry_image_representation.md)**

:   提出两阶段轻量框架SwiftTailor，通过PatternMaker预测缝纫样板 + GarmentSewer将其转换为统一UV空间的Garment Geometry Image，结合逆映射与动态拼接直接生成3D服装网格，推理速度比现有方法快数十倍且达到SOTA质量。

**[TagSplat: Topology-Aware Gaussian Splatting for Dynamic Mesh Modeling and Tracking](3d_vision/tagsplat_topology-aware_gaussian_splatting_for_dynamic_mesh_modeling_and_trackin.md)**

:   提出拓扑感知的高斯泼溅框架 TagSplat，通过显式编码高斯基元间的空间连接关系，在动态场景重建中生成拓扑一致的网格序列，并支持精确的3D关键点跟踪。

**[Learning 3D Reconstruction with Priors in Test Time](3d_vision/tco_learning_3d_reconstruction_with_priors_in_test_time.md)**

:   提出测试时约束优化（TCO）框架，无需重训练或修改预训练多视图 Transformer 架构，通过在推理时将先验（相机位姿、内参、深度）作为预测约束进行优化，大幅提升 3D 重建精度。

**[Text–Image Conditioned 3D Generation](3d_vision/text-image_conditioned_3d_generation.md)**

:   本文发现图像条件和文本条件在3D生成中提供互补信息——图像给出精确外观但受视角限制，文本提供全局语义但缺乏视觉细节——并提出TIGON，一个最小化双分支DiT基线，通过零初始化跨模态桥(early fusion)和步级预测平均(late fusion)实现联合文本-图像条件的原生3D生成。

**[TopoMesh: High-Fidelity Mesh Autoencoding via Topological Unification](3d_vision/topomesh_high-fidelity_mesh_autoencoding_via_topological_unification.md)**

:   提出 TopoMesh，通过将GT网格和预测网格统一到 Dual Marching Cubes (DMC) 拓扑框架下，首次实现了顶点和面片级别的显式对应，从而支持直接网格级别监督（拓扑、顶点位置、面法向量），F1-Sharp 指标比现有SOTA提升 5.9-7.1%，尤其在锐利特征保持上优势显著。

**[Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](3d_vision/towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)**

:   提出 World Scene Graph Generation (WSGG) 任务，从单目视频构建包含所有物体（含被遮挡/出画面物体）的时空持久、世界坐标系锚定的场景图，并引入 ActionGenome4D 数据集和三种互补方法（PWG/MWAE/4DST）。

**[TR2M: Transferring Monocular Relative Depth to Metric Depth with Language Descriptions and Dual-Level Scale-Oriented Contrast](3d_vision/tr2m_transferring_monocular_relative_depth_to_metric_depth_with_language_descrip.md)**

:   提出 TR2M 框架，利用图像和文本描述预测像素级的 scale/shift 映射图，将泛化性强但无尺度的相对深度转换为度量深度，仅用 19M 可训练参数和 102K 训练图像即可实现跨域零样本度量深度估计。

**[tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction](3d_vision/tttlrm_test-time_training_for_long_context_and_autoregressive_3d_reconstruction.md)**

:   tttLRM 首次将 Test-Time Training (TTT) 引入大规模3D重建模型，利用 LaCT 层以线性复杂度实现长上下文和自回归3D高斯重建，通过将多视图观测压缩到 TTT 快速权重中形成隐式3D表示，再解码为显式3DGS等格式，在物体和场景级数据集上达到了 SOTA 性能。

**[Unblur-SLAM: Dense Neural SLAM for Blurry Inputs](3d_vision/unblur-slam_dense_neural_slam_for_blurry_inputs.md)**

:   Unblur-SLAM 不是简单把去模糊网络塞进 SLAM 前端，而是围绕“哪些模糊帧可以先去模糊再跟踪、哪些模糊帧必须直接在 3D 空间里建模”这一关键决策，设计了模糊检测、物理约束去模糊、3D Gaussian blur refinement 和严重模糊 fallback 的完整流水线，因此能同时处理运动模糊和散焦模糊，并显著提升跟踪与重建质量。

**[UniSplat: Learning 3D Representations for Spatial Intelligence from Unposed Multi-View Images](3d_vision/unisplat_3d_representations_unposed.md)**

:   UniSplat 通过双掩码策略、粗到细高斯溅射和位姿条件重校准三个组件，从无位姿多视角图像中学习统一的几何-外观-语义 3D 表示，为空间智能奠定感知基础。

**[Using Gaussian Splats to Create High-Fidelity Facial Geometry and Texture](3d_vision/using_gaussian_splats_to_create_high-fidelity_facial_geometry_and_texture.md)**

:   提出一套基于改进 Gaussian Splatting 的人脸重建管线：通过软约束和语义分割监督将高斯与三角网格紧耦合，从仅 11 张未标定图像重建高精度三角面片几何，并利用 PCA 先验 + 可重光照高斯模型分离光照获取去光照 albedo 纹理，最终兼容标准图形管线（MetaHuman）。

**[UTrice: Unifying Primitives in Differentiable Ray Tracing and Rasterization via Triangles for Particle-Based 3D Scenes](3d_vision/utrice_unifying_primitives_in_differentiable_ray_tracing_and_rasterization_via_t.md)**

:   UTrice 提出以三角形替代高斯椭球作为可微光线追踪的统一图元，无需代理几何体即可直接在 OptiX BVH 中追踪三角形，在保持实时渲染性能的同时显著超越 3DGRT 的渲染质量，并天然兼容光栅化方法 Triangle Splatting 优化的三角形，实现了光栅化与光线追踪的图元统一。

**[VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](3d_vision/varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)**

:   提出 VarSplat，首个在3DGS-SLAM中学习**逐splat外观方差** $\sigma^2$ 并通过全方差定律渲染**逐像素不确定性图** $V$ 的系统，将不确定性统一应用于跟踪、子图配准和回环检测，在4个数据集上取得鲁棒且领先的性能。

**[VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control](3d_vision/versecrafter_dynamic_realistic_video_world_model_with_4d_geometric_control.md)**

:   提出 VerseCrafter，一个基于4D几何控制表示（静态背景点云 + 逐物体3D高斯轨迹）的视频世界模型，通过轻量 GeoAdapter 将4D控制信号注入冻结的 Wan2.1-14B 视频扩散模型，实现了对相机和多物体运动的精确、解耦控制，同时构建了包含 35K 样本的真实世界数据集 VerseControl4D。

**[VGG-T3: Offline Feed-Forward 3D Reconstruction at Scale](3d_vision/vgg-t3_offline_feed-forward_3d_reconstruction_at_scale.md)**

:   提出VGG-T3，通过**测试时训练(TTT)**将VGGT中全局注意力层的变长KV表示压缩为固定大小MLP，将离线前馈三维重建的计算复杂度从 $O(n^2)$ 降至 $O(n)$，实现了千张图片级别的大规模场景重建（1k张图仅需58秒）。

**[VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object Detection](3d_vision/vggt-det_mining_vggt_internal_priors_for_sensor-geometry-free_multi-view_indoor_.md)**

:   提出 VGGT-Det，首个面向无传感器几何输入 (SG-Free) 的多视图室内3D目标检测框架，通过挖掘 VGGT 编码器内部的语义先验（注意力引导查询生成 AG）和几何先验（查询驱动特征聚合 QD），在 ScanNet 和 ARKitScenes 上分别超越最优方法 4.4 和 8.6 mAP@0.25。

**[VGGT-SLAM++: Visual SLAM with DEM-Based Covisibility and Local Bundle Adjustment](3d_vision/vggt-slam.md)**

:   VGGT-SLAM++ 在 VGGT 前馈 Transformer 里程计基础上，引入数字高程图（DEM）作为紧凑的几何保持表示，利用 DINOv2 嵌入实现高效环路检测和共视图构建，配合高频 Sim(3) 局部光束法平差修正短期漂移，在 TUM RGB-D 上 ATE 降低 45%（0.079m→0.036m）。

**[VirPro: Visual-referred Probabilistic Prompt Learning for Weakly-Supervised Monocular 3D Detection](3d_vision/virpro_visual-referred_probabilistic_prompt_learning_for_weakly-supervised_monoc.md)**

:   提出 VirPro——一种自适应多模态预训练范式，通过视觉引导的概率提示（Adaptive Prompt Bank + Multi-Gaussian Prompt Modeling）为弱监督单目3D检测提供场景感知的语义监督信号，可无缝集成到现有 WS-M3D 框架中，在 KITTI 上最高带来 4.8% AP 提升。

**[Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI](3d_vision/wanderland_geometrically_grounded_simulation_for_open-world_embodied_ai.md)**

:   提出 Wanderland real-to-sim 框架：利用手持多传感器扫描仪（LiDAR+IMU+RGB）采集开放世界室内外场景，通过 LIV-SLAM 获取度量级精确几何与相机位姿，结合 3DGS 实现光学真实感渲染 + 几何接地碰撞仿真，构建 530 场景/42 万帧/380 万 m² 的大规模数据集，系统证明纯视觉重建在度量精度、Mesh 质量和导航策略训练/评估可靠性上远不及 LiDAR 增强方案。

**[What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?](3d_vision/what_makes_good_synthetic_training_data_for_zero-shot_stereo_matching.md)**

:   系统消融合成立体匹配训练数据的设计空间（浮动物体、背景、材质、基线等），发现"真实室内场景 + 密集浮动物体 + 宽基线"是最优组合，据此构建的 WMGStereo-150k 仅用单一数据集即超越四大经典数据集的混合训练。

**[Where, What, Why: Toward Explainable 3D-GS Watermarking](3d_vision/where_what_why_toward_explainable_3d-gs_watermarking.md)**

:   提出一种表示原生的 3D-GS 水印框架，通过 Trio-Experts 选载体（where）、Channel-wise Group Mask 控梯度（what）、解耦微调实现可审计归因（why），在渲染质量（PSNR +0.83 dB）和比特精度（+1.24%）上均超越 SOTA。

**[Yo'City: Personalized and Boundless 3D Realistic City Scene Generation via Self-Critic Expansion](3d_vision/yocity_personalized_and_boundless_3d_realistic_city_scene_generation_via_self-cr.md)**

:   提出 Yo'City 多智能体框架，通过"City–District–Grid"层次化规划 + produce–refine–evaluate 等距图像合成环 + 场景图引导扩展机制，实现用户个性化文本驱动的无界 3D 城市生成，在语义一致性和视觉质量上全面超过 SynCity 等现有方法。

**[Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image](3d_vision/zero-shot_reconstruction_of_animatable_3d_avatars_with_cloth_dynamics_from_a_sin.md)**

:   DynaAvatar 提出首个零样本框架，从单张图像重建具有运动依赖布料动态效果的可动画化3D人体Avatar，核心通过静态-动态知识迁移策略和光流引导的 DynaFlow 损失函数，在有限动态数据下实现了逼真的衣物动态建模，全面超越现有方法。

---

## 🎨 图像生成 { #image_generation }

**[2ndMatch: Finetuning Pruned Diffusion Models via Second-Order Jacobian Matching](image_generation/2ndmatch_finetuning_pruned_diffusion_models_via_second-order_jacobian_matching.md)**

:   提出2ndMatch微调框架，通过对齐剪枝模型与原始模型的二阶Jacobian矩阵 $J^\top J$（灵感来自有限时间Lyapunov指数），匹配两者对输入扰动的时间敏感性，从而显著缩小剪枝扩散模型与原始模型的生成质量差距。

**[ADAPT: Attention Driven Adaptive Prompt Scheduling and InTerpolating Orthogonal Complements for Rare Concepts Generation](image_generation/adapt_attention_driven_adaptive_prompt_scheduling_and_interpolating_orthogonal_c.md)**

:   提出 ADAPT 框架，通过注意力驱动的自适应 Prompt 调度（APS）、池化嵌入操控（PEM）和潜空间操控（LSM）三个零样本模块，确定性且语义对齐地控制从通用到罕见概念的生成过渡，在 RareBench 上显著超越 R2F 基线。

**[HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](image_generation/adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)**

:   提出HINGE框架，首次将预训练的表达空间单细胞基础模型(sc-FM, CellFM)改装为组织学图像条件的空间基因表达生成器，通过恒等初始化的SoftAdaLN调制轻量注入视觉上下文、表达空间掩码扩散过程对齐预训练目标、warm-start课程稳定训练，在三个ST数据集上达SOTA并保持优越的基因共表达一致性。

**[Adaptive Auxiliary Prompt Blending for Target-Faithful Diffusion Generation](image_generation/adaptive_auxiliary_prompt_blending_for_target-faithful_diffusion_generation.md)**

:   提出 Adaptive Auxiliary Prompt Blending (AAPB)，通过 Tweedie 公式推导闭式自适应混合系数，在每个去噪步动态平衡辅助锚定提示与目标提示的贡献，无需训练即可显著改善稀有概念生成和零样本图像编辑的语义准确性与结构保真度。

**[Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](image_generation/adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)**

:   提出 Spectrum，一种基于切比雪夫多项式的全局谱域特征预测方法，将扩散模型去噪器的中间特征视为时间函数并用岭回归拟合系数，实现误差不随步长增长的长程特征预测，在 FLUX.1 上达到 4.79× 加速、在 Wan2.1-14B 上达到 4.67× 加速而质量几乎无损。

**[Agentic Retoucher for Text-To-Image Generation](image_generation/agentic_retoucher_for_texttoimage_generation.md)**

:   Agentic Retoucher 将 T2I 生成图像的局部缺陷修复重构为感知→推理→行动的多 agent 闭环决策流程，通过上下文感知的显著性检测、人类偏好对齐的诊断推理和自适应工具选择实现自主修复，在 GenBlemish-27K 上 plausibility 提升 2.89 分，83.2% 修复结果被人类评为优于原图。

**[AHS: Adaptive Head Synthesis via Synthetic Data Augmentations](image_generation/ahs_adaptive_head_synthesis.md)**

:   AHS 通过使用头部重演模型（GAGAvatar）生成合成增强数据来克服自监督训练的局限性，结合双编码器注意力机制和自适应掩码策略，在全身图像的头部替换任务中实现了 SOTA 效果。

**[AlignVAR: Towards Globally Consistent Visual Autoregression for Image Super-Resolution](image_generation/alignvar_towards_globally_consistent_visual_autoregression_for_image_super-resol.md)**

:   针对视觉自回归（VAR）模型在图像超分辨率中的两个一致性问题——注意力局部偏差导致的空间不连贯和残差监督导致的跨尺度误差累积，提出 AlignVAR 框架，通过空间一致性自回归（SCA）和层级一致性约束（HCC）协同解决，实现比扩散方法快 10× 以上的推理速度且重建质量更优。

**[All-in-One Slider for Attribute Manipulation in Diffusion Models](image_generation/all_in_one_slider_attribute_manipulation.md)**

:   提出 All-in-One Slider 框架，通过在文本嵌入空间上训练一个属性稀疏自编码器（Attribute Sparse Autoencoder），将多种人脸属性解耦为稀疏的语义方向，实现单一轻量模块对 52+ 种属性的细粒度连续控制，并支持多属性组合和未见属性的零样本操控。

**[Ani3DHuman: Photorealistic 3D Human Animation with Self-guided Stochastic Sampling](image_generation/ani3dhuman_photorealistic_3d_human_animation_with_self-guided_stochastic_samplin.md)**

:   提出 Ani3DHuman 框架，将运动学驱动的网格动画与视频扩散先验相结合，通过自引导随机采样（Self-guided Stochastic Sampling）将低质量的刚体渲染恢复为高保真视频，从而实现逼真的非刚体服装动态建模。

**[APPLE: Attribute-Preserving Pseudo-Labeling for Diffusion-Based Face Swapping](image_generation/apple_attribute-preserving_pseudo-labeling_for_diffusion-based_face_swapping.md)**

:   APPLE 提出了一种基于扩散模型的教师-学生框架，通过条件去模糊（代替传统条件修复）训练教师模型生成属性对齐的伪标签，再利用这些高质量伪标签训练学生模型，在保持身份迁移能力的同时实现了 SOTA 的属性保留性能（FID 2.18, Pose Error 1.85）。

**[Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation](image_generation/ar2can_an_architect_and_an_artist_leveraging_a_canvas_for_multi-human_generation.md)**

:   Ar2Can 提出将多人图像生成分解为空间规划（Architect）和身份保留渲染（Artist）两阶段，通过 GRPO 强化学习配合基于匈牙利匹配的空间锚定人脸奖励函数训练 Artist 模型，在 MultiHuman-Testbench 上实现了 68.2 的身份保留分数和 90.2 的计数准确率，大幅超越所有基线。

**[AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](image_generation/as-bridge_a_bidirectional_generative_framework_bridging_next-generation_astronom.md)**

:   提出 AS-Bridge，一个基于 Brownian Bridge 扩散过程的双向生成框架，在地基 LSST 与空基 Euclid 天文巡天之间建模概率条件分布，实现跨巡天图像翻译和罕见事件检测（引力透镜），并通过 $\epsilon$-prediction 训练目标改进了标准 Brownian Bridge 的似然估计。

**[Attention, May I Have Your Decision? Localizing Generative Choices in Diffusion Models](image_generation/attention_may_i_have_your_decision_localizing_generative_choices_in_diffusion_mo.md)**

:   本文通过线性探针（linear probing）发现扩散模型中**隐式决策**（如未指定性别时默认生成男性）主要由自注意力层而非交叉注意力层控制，并基于此提出 ICM 方法，仅在少量关键自注意力层上进行干预即可实现 SOTA 的去偏见效果，同时最小化图像质量退化。

**[Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution](image_generation/attribution_as_retrieval_modelagnostic_aigenerated.md)**

:   提出 LIDA，将 AI 生成图像溯源从分类问题转化为检索问题，利用低位平面指纹捕获生成器特异性伪影，配合无监督预训练和少样本自适应，在零/少样本设置下实现 SOTA 的 Deepfake 检测和图像溯源。

**[AutoDebias: An Automated Framework for Detecting and Mitigating Backdoor Biases in Text-to-Image Models](image_generation/autodebias_automated_framework_for_debiasing_text-to-image_models.md)**

:   提出 AutoDebias——首个同时检测和缓解 T2I 模型中恶意后门偏见的统一框架，利用 VLM 开放集检测发现触发词-偏见关联并构建查找表，再通过 CLIP 引导的分布对齐训练消除后门关联，在 17 种后门场景中将攻击成功率从 90% 降至接近 0 且保持图像质量。

**[Banana100: Breaking NR-IQA Metrics by 100 Iterative Image Replications with Nano Banana Pro](image_generation/banana100_breaking_nr-iqa_metrics_by_100_iterative_image_replications_with_nano_.md)**

:   Banana100 通过让 Nano Banana Pro 迭代复制图像 100 次来系统性研究多轮编辑中的质量退化问题，构建了包含 28,000 张退化图像的数据集，并揭示了一个惊人发现：21 种主流无参考图像质量评估（NR-IQA）指标均无法可靠检测迭代退化——大多数指标甚至给噪声图像打出比干净图像更高的分数。

**[BeautyGRPO: Aesthetic Alignment for Face Retouching via Dynamic Path Guidance and Fine-Grained Preference Modeling](image_generation/beautygrpo_aesthetic_alignment_for_face_retouching_via_dynamic_path_guidance_and.md)**

:   提出 BeautyGRPO，一个基于强化学习的人脸修图框架，通过构建细粒度偏好数据集 FRPref-10K 训练专用奖励模型，并设计动态路径引导（DPG）机制在随机探索与高保真之间取得平衡，实现与人类美学偏好对齐的自然修图效果。

**[Beyond the Golden Data: Resolving the Motion-Vision Quality Dilemma via Timestep Selective Training](image_generation/beyond_the_golden_data_resolving_the_motion-vision_quality_dilemma_via_timestep_.md)**

:   发现视频数据中运动质量（MQ）和视觉质量（VQ）呈负相关的"Motion-Vision Quality Dilemma"，通过梯度分析揭示不平衡数据在适当时间步可产生等效学习信号，提出TQD框架使仅用不平衡数据训练即可超越黄金数据训练。

**[BiGain: Unified Token Compression for Joint Generation and Classification](image_generation/bigain_unified_token_compression_for_joint_generation_and_classification.md)**

:   BiGain 提出频率感知的 token 压缩框架，通过拉普拉斯门控 token 合并和插值-外推 KV 下采样两个无训练算子，首次在扩散模型加速中同时保持生成质量并显著提升判别分类性能。

**[BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation](image_generation/bimotion_b-spline_motion_for_text-guided_dynamic_3d_character_generation.md)**

:   提出 BiMotion，用连续可微的 B 样条曲线将变长运动序列压缩为固定数量控制点，配合专用 VAE 和 flow-matching 扩散模型，实现快速、高表达力、语义完整的文本引导动态 3D 角色生成，在质量和效率上均超越现有方法。

**[BioVITA: Biological Dataset, Model, and Benchmark for Visual-Textual-Acoustic Alignment](image_generation/biovita_biological_dataset_model_and_benchmark_for_visual-textual-acoustic_align.md)**

:   提出 BioVITA 框架，包含百万级三模态（图像-文本-音频）生物数据集、两阶段对齐模型和六方向跨模态物种级检索基准，首次实现生物领域视觉-文本-声音统一表示学习。

**[BlackMirror: Black-Box Backdoor Detection for Text-to-Image Models via Instruction-Response Deviation](image_generation/blackmirror_black-box_backdoor_detection_for_text-to-image_models_via_instructio.md)**

:   提出 BlackMirror 框架，通过细粒度的指令-响应语义偏差检测（MirrorMatch）和跨 prompt 稳定性验证（MirrorVerify）两阶段流程，在黑盒条件下实现对 T2I 模型多种后门攻击的通用检测，F1 平均达 89.46%，大幅超越已有黑盒方法 UFID。

**[CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing](image_generation/care-edit_condition-aware_routing_of_experts_for_contextual_image_editing.md)**

:   提出 CARE-Edit，一种条件感知的专家路由框架，通过异构专家（Text/Mask/Reference/Base）配合轻量级 latent-attention 路由器，在 DiT 骨干上实现动态计算分配，有效解决统一图像编辑器中多条件信号（文本、掩码、参考图）冲突导致的颜色溢出、身份漂移等问题。

**[CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion](image_generation/careflow_cyclic_adaptive_rectified_flow_for_multimodal_fusion.md)**

:   提出 CaReFlow，首次将 rectified flow 用于多模态分布映射以缩小模态间隙：通过 one-to-many mapping 让源模态数据点观测目标模态全局分布，adaptive relaxed alignment 对不同关联度的模态对施加不同对齐强度，cyclic rectified flow 保证映射后信息不丢失，即使用简单拼接融合也能在多个多模态情感计算 benchmark 上达到 SOTA。

**[Causal Motion Diffusion Models for Autoregressive Motion Generation](image_generation/causal_motion_diffusion_models_for_autoregressive_motion_generation.md)**

:   提出 CMDM 框架，在运动-语言对齐的因果隐空间中统一扩散去噪与自回归生成，通过帧级独立噪声和因果不确定性采样调度，实现高质量、低延迟的文本到动作生成和长序列流式合成。

**[CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance](image_generation/cfg-ctrl_control-based_classifier-free_diffusion_guidance.md)**

:   将 Classifier-Free Guidance (CFG) 重新解释为流匹配扩散模型中的反馈控制过程，提出统一框架 CFG-Ctrl，并基于滑模控制 (SMC) 设计非线性反馈引导机制 SMC-CFG，在大引导尺度下显著提升语义一致性和生成鲁棒性。

**[ChangeBridge: Spatiotemporal Image Generation with Multimodal Controls for Remote Sensing](image_generation/changebridge_spatiotemporal_image_generation_with_multimodal_controls_for_remote.md)**

:   提出 ChangeBridge，通过漂移异步扩散桥（drift-asynchronous diffusion bridge）实现遥感场景中从前事件到后事件的条件时空图像生成，支持坐标文本、语义掩码、实例布局等多模态控制，并可作为变化检测任务的数据生成引擎。

**[ChordEdit: One-Step Low-Energy Transport for Image Editing](image_generation/chordedit_one-step_low-energy_transport_for_image_editing.md)**

:   基于动态最优传输理论，推导出低能量的 Chord 控制场，将不稳定的朴素编辑场平滑化，首次实现了对蒸馏单步 T2I 模型的无训练、无反演、高保真实时图像编辑。

**[Cinematic Audio Source Separation Using Visual Cues](image_generation/cinematic_audio_source_separation_using_visual_cues.md)**

:   提出首个音视频影视音频源分离（AV-CASS）框架，利用面部和场景双视频流的视觉线索，通过条件流匹配进行生成式三路音频分离（语音/音效/音乐），仅在合成数据上训练即可泛化到真实电影。

**[Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers](image_generation/circuit_mechanisms_for_spatial_relation_generation_in_diffusion_models.md)**

:   通过机械可解释性方法揭示了扩散Transformer（DiT）生成空间关系的内部电路机制：随机嵌入模型使用两阶段模块化电路（关系头+物体生成头），T5编码器模型则将关系信息融合到物体token中通过单token解码，两种机制的鲁棒性差异显著。

**[CoD: A Diffusion Foundation Model for Image Compression](image_generation/cod_a_diffusion_foundation_model_for_image_compression.md)**

:   提出首个面向压缩的扩散基础模型 CoD，从零训练学习端到端的压缩-生成联合优化，替换 Stable Diffusion 后在下游扩散编解码器中实现超低码率（0.0039 bpp）下的 SOTA 性能，训练成本仅为 SD 的 0.3%。

**[coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](image_generation/codrawagents_a_multiagent_dialogue_framework_for_c.md)**

:   提出coDrawAgents交互式多智能体对话框架，Interpreter、Planner、Checker、Painter四个专业智能体闭环协作，以分治策略按语义优先级逐组增量规划布局，基于画布视觉上下文接地推理并显式纠错，在GenEval上以0.94 Overall Score大幅领先GPT Image 1（0.84），在DPG-Bench上达85.17 SOTA。

**[CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment](image_generation/cognitioncapturerpro_towards_highfidelity_visual_d.md)**

:   CognitionCapturerPro通过不确定性加权掩蔽解决保真度损失、多模态融合编码器整合图像/文本/深度/边缘信息解决表征偏移，配合轻量共享主干对齐替代扩散先验，在THINGS-EEG数据集上Top-1/Top-5检索准确率分别提升25.9%和10.6%。

**[CoLoGen: Progressive Learning of Concept-Localization Duality for Unified Image Generation](image_generation/cologen_progressive_learning_of_concept-localization_duality_for_unified_image_g.md)**

:   提出 CoLoGen，一个基于"概念-定位对偶性"（Concept-Localization Duality）的统一图像生成框架，通过渐进式分阶段训练和 Progressive Representation Weaving（PRW）动态专家路由架构，在指令编辑、可控生成和个性化生成三大任务上同时达到或超越专用模型水平。

**[ConsistCompose: Unified Multimodal Layout Control for Image Composition](image_generation/consistcompose_multimodal_layout_control.md)**

:   提出 ConsistCompose，通过将布局坐标直接嵌入语言prompt（LELG范式），在统一多模态框架中实现布局可控的多实例图像生成；构建340万样本的ConsistCompose3M数据集提供布局+身份监督；配合坐标感知CFG机制，在COCO-Position上实现布局IoU 7.2%提升和AP 13.7%提升，同时保持通用理解能力。

**[COT-FM: Cluster-wise Optimal Transport Flow Matching](image_generation/cot-fm_cluster-wise_optimal_transport_flow_matching.md)**

:   提出 COT-FM，一个即插即用的 Flow Matching 增强框架：通过聚类目标样本、反转预训练模型获取簇级源分布、在簇内近似最优传输，显著拉直传输路径，在不改变模型架构的前提下同时加速采样和提升生成质量。

**[CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think](image_generation/craft_aligning_diffusion_models_with_finetuning_is_easier_than_you_think.md)**

:   CRAFT 提出一种超轻量的扩散模型对齐方法：通过组合奖励过滤(CRF)策略自动构建高质量训练集，然后执行增强版 SFT，理论证明 CRAFT 实际优化的是分组强化学习的下界，仅用 100 个样本就超越了需要数千偏好对的 SOTA 方法，且训练速度快 11-220 倍。

**[Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video](image_generation/cross-modal_emotion_transfer_for_emotion_editing_in_talking_face_video.md)**

:   提出 C-MET（Cross-Modal Emotion Transfer），通过建模语音和面部表情空间之间的情感语义向量映射，首次实现了基于语音驱动的扩展情感（如讽刺、魅力）说话人脸视频生成，情感准确率超越 SOTA 14%。

**[CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration](image_generation/ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)**

:   提出 CTCal（Cross-Timestep Self-Calibration），利用扩散模型在小时间步（低噪声）下形成的可靠文本-图像对齐（cross-attention maps）来校准大时间步（高噪声）下的表征学习，为文本到图像生成提供显式的跨时间步自监督，在 T2I-CompBench++ 和 GenEval 上全面超越现有方法。

**[Cycle-Consistent Tuning for Layered Image Decomposition](image_generation/cycle-consistent_tuning_for_layered_image_decomposition.md)**

:   提出基于扩散模型的循环一致性微调框架，通过联合训练分解模型和合成模型实现图像层分离（如logo-物体分解），并引入渐进式自改进数据扩增策略，在非线性层交互场景下实现鲁棒分解。

**[D2C: Accelerating Diffusion Model Training under Minimal Budgets via Condensation](image_generation/d2c_diffusion_dataset_condensation.md)**

:   首次将数据集压缩引入扩散模型训练，提出D2C两阶段框架（Select+Attach），仅用0.8% ImageNet数据在40K步达到FID 4.3，比REPA快100倍、比vanilla SiT快233倍。

**[DA-VAE: Plug-in Latent Compression for Diffusion via Detail Alignment](image_generation/da-vae_plug-in_latent_compression_for_diffusion_via_detail_alignment.md)**

:   提出 Detail-Aligned VAE (DA-VAE)，通过结构化潜在空间（base + detail channels）和对齐损失，在不从头训练扩散模型的前提下将预训练 VAE 的压缩率提升至原来的 4 倍，仅需 5 H100-days 即可适配 SD3.5 生成 1024×1024 图像。

**[Elucidating the SNR-t Bias of Diffusion Probabilistic Models](image_generation/dcw_snr_t_bias_diffusion.md)**

:   本文揭示了扩散模型中普遍存在的 SNR-t 偏差（逆过程样本的信噪比与时间步不匹配），并提出小波域动态差分校正方法（DCW），在不训练的情况下即插即用地提升多种扩散模型的生成质量。

**[DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation](image_generation/deco_frequency-decoupled_pixel_diffusion_for_end-to-end_image_generation.md)**

:   DeCo 提出频率解耦的像素扩散框架，用轻量像素解码器处理高频细节并让DiT专注低频语义建模，配合频率感知flow matching损失，在ImageNet上达到FID 1.62（256）和2.22（512），缩小了像素扩散与潜空间扩散的差距。

**[Depth Adaptive Efficient Visual Autoregressive Modeling](image_generation/depthvar_depth_adaptive_var.md)**

:   揭示了 VAR 模型中频率驱动的硬剪枝范式存在根本性局限，提出 DepthVAR，一种免训练的推理加速框架，通过自适应分配每个 token 的 Transformer 层计算深度（而非二值化的保留/剪除），实现 2.3×-3.1× 加速且质量损失极小。

**[Diffusion Mental Averages](image_generation/diffusion_mental_averages.md)**

:   提出 Diffusion Mental Averages (DMA)，通过在扩散模型的语义空间中对齐多个去噪轨迹，从预训练扩散模型中提取概念的"心理平均"原型图像——首次实现一致、逼真的概念平均可视化。

**[Diffusion Probe: Generated Image Result Prediction Using CNN Probes](image_generation/diffusion_probe_generated_image_result_prediction_using_cnn_probes.md)**

:   发现扩散模型早期去噪步骤的交叉注意力分布与最终图像质量高度相关，提出 Diffusion Probe——用轻量CNN从早期注意力图预测生成结果质量，实现在完成10%去噪即可预筛选低质量生成路径，加速 Prompt 优化、Seed 选择和 GRPO 训练。

**[DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization](image_generation/diflowdubber_discrete_flow_matching_for_automated_video_dubbing_via_cross-modal_.md)**

:   提出DiFlowDubber，基于**离散流匹配(DFM)**的自动视频配音框架，通过两阶段训练（零样本TTS预训练→视频配音适配）将大规模TTS知识迁移到视频驱动配音，设计FaPro模块捕获面部表情-韵律映射、Synchronizer模块实现精准唇音同步。

**[DiP: Taming Diffusion Models in Pixel Space](image_generation/dip_taming_diffusion_models_in_pixel_space.md)**

:   提出 DiP，一个高效的像素空间扩散框架，通过将 DiT backbone 在大patch上建模全局结构 + 轻量 Patch Detailer Head 恢复局部细节，实现了与LDM可比的计算效率但无需VAE，在ImageNet 256×256上达到1.79 FID。

**[Disentangling to Re-couple: Resolving the Similarity-Controllability Paradox in Subject-Driven Text-to-Image Generation](image_generation/disentangling_to_re-couple_resolving_the_similarity-controllability_paradox_in_s.md)**

:   提出 DisCo 框架，通过先解耦文本与视觉信息（用代词替换实体词消除文本对 subject 的干扰）、再用 GRPO + 专用 reward model 重新耦合二者，有效解决了 subject-driven 图像生成中"相似度-可控性"不可兼得的悖论。

**[DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](image_generation/ditic_aligned_diffusion_transformer_for_efficient.md)**

:   将预训练文生图DiT（SANA）适配为高效单步图像压缩解码器，通过方差引导重建流（像素级自适应去噪强度）、自蒸馏对齐（编码器潜变量做蒸馏目标）、潜空间条件引导（替代文本编码器）三种对齐机制，在32×下采样的深层潜空间中实现SOTA感知质量（BD-rate DISTS -87.88%），解码快30倍且16GB笔电显存可重建2K图像。

**[Diversity over Uniformity: Rethinking Representation in Generated Image Detection](image_generation/diversity_over_uniformity_rethinking_representation_in_generated_image_detection.md)**

:   提出反特征坍塌学习框架 AFCL，通过信息瓶颈过滤无关特征并抑制不同伪造线索之间的过度重叠，保持判别表征的多样性和互补性，在跨模型生成图像检测上取得显著提升。

**[DMin: Scalable Training Data Influence Estimation for Diffusion Models](image_generation/dmin_scalable_training_data_influence_estimation_for_diffusion_models.md)**

:   提出 DMin，一个可扩展的扩散模型训练数据影响力估计框架，通过高效梯度压缩将存储需求从数百 TB 降至 MB/KB 级别，首次实现对数十亿参数扩散模型的影响力估计，支持亚秒级 top-k 检索。

**[Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](image_generation/dpcache_denoising_path_planning_diffusion_accel.md)**

:   将扩散模型采样加速形式化为全局路径规划问题，通过构建路径感知代价张量 (PACT) 并使用动态规划选择最优关键时间步序列，实现 training-free 的 4.87× 加速且生成质量超越全步基线。

**[DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](image_generation/dreamvideo-omni_omni-motion_controlled_multi-subject_video_customization_with_la.md)**

:   提出 DreamVideo-Omni，通过两阶段渐进训练范式（全运动身份监督微调 + 潜空间身份奖励反馈学习），在单一 DiT 架构中首次统一实现多主体定制与全粒度运动控制（全局包围盒 + 局部轨迹 + 相机运动）。

**[DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](image_generation/duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md)**

:   提出 DUO-VSR 三阶段蒸馏框架，通过渐进引导蒸馏初始化 + 双流蒸馏（DMD + RFS-GAN 联合优化）+ 偏好引导精调，将多步视频超分模型压缩为单步生成器，实现约 50× 加速且超越先前单步 VSR 方法的视觉质量。

**[DynaVid: Learning to Generate Highly Dynamic Videos using Synthetic Motion Data](image_generation/dynavid_learning_to_generate_highly_dynamic_videos_using_synthetic_motion_data.md)**

:   DynaVid 提出利用计算机图形学渲染的合成光流（而非合成视频）来训练视频扩散模型，通过运动生成器+运动引导视频生成器的两阶段框架，实现了高度动态运动的逼真视频合成和精细相机控制。

**[EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation](image_generation/edgedit_hardware-aware_diffusion_transformers_for_efficient_on-device_image_gene.md)**

:   EdgeDiT 提出一种硬件感知的扩散 Transformer 优化框架，通过层级知识蒸馏训练轻量级代理块、多目标贝叶斯优化搜索 Pareto 最优架构，实现了 20-30% 参数缩减、36-46% FLOPs 降低、1.65x 端侧加速，同时保持甚至超越原始 DiT-XL/2 的生成质量。

**[Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](image_generation/editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)**

:   本文从理论和实验两方面统一分析了非对抗性扩散编辑如何无意中破坏鲁棒隐形水印，推导了水印 SNR 衰减和互信息衰减的界，并在指令编辑、拖拽编辑、无训练合成等场景下验证了水印恢复的系统性失效。

**[EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing](image_generation/effecterase_joint_video_object_removal_and_insertion_for_high-quality_effect_era.md)**

:   提出 EffectErase 框架，将视频物体插入作为移除的逆辅助任务进行联合学习，并构建包含 60K 视频对的大规模 VOR 数据集，实现对物体及其遮挡、阴影、反射、光照、变形等视觉副效应的高质量擦除。

**[EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation](image_generation/egoflow_gradient-guided_flow_matching_for_egocentric_6dof_object_motion_generati.md)**

:   EgoFlow 提出一种基于 Flow Matching 的生成框架，通过 Mamba-Transformer-Perceiver 混合架构融合多模态场景条件，并在推理时用梯度引导采样施加可微的物理约束（碰撞避免、运动平滑性），从第一人称视频生成物理合理的 6DoF 物体运动轨迹，碰撞率降低高达 79%。

**[Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](image_generation/emf_meanflow_text_to_image.md)**

:   首次将 MeanFlow 框架从类别标签条件扩展到文本条件图像生成，发现限制步数下文本表示的语义区分性和解耦性是关键瓶颈，基于 BLIP3o-NEXT 文本编码器实现了高质量的少步/单步 T2I 生成。

**[EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories](image_generation/emma_concept_erasure_benchmark_with_comprehensive_semantic_metrics_and_diverse_c.md)**

:   提出 EMMA 基准，从五个维度（擦除能力、保留能力、效率、质量、偏见）共 12 个指标系统评估 T2I 模型的概念擦除方法，覆盖 206 个概念类别和 5 个领域，首次揭示现有方法在隐式提示下的浅层擦除本质和偏见放大问题。

**[Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](image_generation/enhancing_image_aesthetics_with_dualconditioned_di.md)**

:   DIAE提出多模态美学感知（MAP）模块将模糊的美学指令转为HSV+轮廓图+文本的显式控制信号，并构建"不完美配对"数据集IIAEData配合双分支监督框架进行弱监督训练，实现内容一致的美学增强，LAION美学评分提升17.4%。

**[Enhancing Spatial Understanding in Image Generation via Reward Modeling](image_generation/enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)**

:   构建 80K 对抗性偏好数据集 SpatialReward-Dataset，训练专门评估空间关系准确性的奖励模型 SpatialScore（准确率超越 GPT-5），并用 top-k 过滤策略结合 GRPO 在线 RL 显著提升 FLUX.1-dev 的空间生成能力。

**[Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](image_generation/erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)**

:   本文系统评估了16种文本到图像扩散模型概念擦除（unlearning）方法在安全性（擦除成功率）与组合性生成能力之间的权衡，揭示了激进擦除策略在去除不良内容的同时严重破坏了模型的属性绑定、空间推理和计数能力，强调安全干预不应以牺牲模型语义逻辑为代价。

**[EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation](image_generation/evatok_adaptive_length_video_tokenization_for_eff.md)**

:   提出四阶段框架EVATok：先用proxy tokenizer估计每个视频的最优token分配方案，再训练轻量路由器一次前向预测这些分配，最终训练自适应tokenizer按内容复杂度灵活分配token数，在UCF-101上以24.4%的token节省达到SOTA生成质量。

**[Exploring Conditions for Diffusion Models in Robotic Control](image_generation/exploring_conditions_for_diffusion_models_in_robotic_control.md)**

:   本文探索了如何用预训练文本到图像扩散模型的条件机制为机器人控制生成任务自适应的视觉表示，发现文本条件在控制环境中因域差距而无效，提出 ORCA 框架通过可学习的任务提示词(task prompts)和逐帧视觉提示词(visual prompts)作为条件机制，在 DMC/MetaWorld/Adroit 三个基准的 12 个任务上达到 SOTA。

**[ExpPortrait: Expressive Portrait Generation via Personalized Representation](image_generation/expportrait_expressive_portrait_generation_via_personalized_representation.md)**

:   提出高保真度的个性化头部表征（静态身份偏移 + 动态表情偏移），解决 SMPL-X 等参数化模型表达力不足的问题，结合身份自适应表情迁移模块和 DiT 生成器，在人像视频自驱动和跨身份重演任务上取得 SOTA 表现。

**[ExpressEdit: Fast Editing of Stylized Facial Expressions with Diffusion Models in Photoshop](image_generation/expressedit_fast_editing_of_stylized_facial_expressions_with_diffusion_models_in.md)**

:   本文提出 ExpressEdit，一个完全开源的 Photoshop 插件，通过基于 SPICE 的扩散模型后端结合 Danbooru 表情标签数据库和 RAG 系统，在单个消费级 GPU 上 3 秒内完成风格化面部表情的无噪声编辑，显著优于 GPT/Grok/Nano Banana 2 等商业模型。

**[Face2Scene: Using Facial Degradation as an Oracle for Diffusion-Based Scene Restoration](image_generation/face2scene_using_facial_degradation_as_an_oracle_for_diffusion-based_scene_resto.md)**

:   提出 Face2Scene 两阶段框架：先用参考人脸复原模型(Ref-FR)获得 HQ-LQ 人脸对，从中提取退化编码作为"oracle"，再以此条件化单步扩散模型完成包含身体与背景的全场景图像复原。

**[FDeID-Toolbox: Face De-Identification Toolbox](image_generation/fdeidtoolbox_face_deidentification_toolbox.md)**

:   提出 FDeID-Toolbox，一个模块化的人脸去标识化研究工具箱，通过标准化数据加载、统一方法实现、灵活推理流程和系统评估协议四大组件，首次实现了对多种去标识化方法在隐私保护、效用保持和视觉质量三个维度上的公平可复现对比。

**[Few-shot Acoustic Synthesis with Multimodal Flow Matching](image_generation/few-shot_acoustic_synthesis_with_multimodal_flow_matching.md)**

:   提出 FLAC，首个基于 flow matching 的少样本房间脉冲响应（RIR）生成框架，仅凭单次录音即可在未见场景中合成空间一致的声学响应，并引入 AGREE 联合嵌入用于几何-声学一致性评估。

**[FG-Portrait: 3D Flow Guided Editable Portrait Animation](image_generation/fg-portrait_3d_flow_guided_editable_portrait_animation.md)**

:   提出 FG-Portrait，通过引入基于 FLAME 参数化 3D 头部模型直接计算的「3D 光流」作为无需学习的几何驱动运动对应关系，结合深度引导采样的 3D 光流编码作为扩散模型 ControlNet 的运动条件，显著提升驱动运动迁移精度（APD 降低 22%+），还支持推理时的表情和头部姿态编辑。

**[Flash-Unified: Training-Free and Task-Aware Acceleration for Native Unified Models](image_generation/flash-unified_a_training-free_and_task-aware_acceleration_framework_for_native_u.md)**

:   FlashU 首次对原生统一多模态模型进行系统性冗余分析，发现参数特化和计算异质性现象，据此提出免训练任务感知加速框架，通过 FFN 剪枝、动态层跳过、自适应引导缩放和扩散头缓存，在 Show-o2 上实现 1.78x-2.01x 加速同时保持 SOTA 性能。

**[FontCrafter: High-Fidelity Element-Driven Artistic Font Creation with Visual In-Context Generation](image_generation/fontcrafter_high-fidelity_element-driven_artistic_font_creation_with_visual_in-c.md)**

:   FontCrafter 将艺术字体生成重新定义为视觉上下文生成任务，通过将参考元素图像与空白画布拼接并输入预训练修复模型(FLUX.1-Fill)，实现高保真的元素驱动字体创建，在纹理和结构保真度上显著超越现有方法。

**[Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems](image_generation/fractals_made_practical_denoising_diffusion_as_partitioned_iterated_function_sys.md)**

:   证明了DDIM确定性反向链本质上是一个分区迭代函数系统(PIFS)，并从该框架推导出三个无需模型评估的可计算几何量，从第一性原理统一解释了扩散模型的双阶段去噪动力学、自注意力的有效性，以及四种经验设计选择（cosine schedule offset、分辨率相关logSNR偏移、Min-SNR损失加权、Align Your Steps采样）。

**[FRAMER: Frequency-Aligned Self-Distillation with Adaptive Modulation Leveraging Diffusion Priors for Real-World Image Super-Resolution](image_generation/framer_frequency-aligned_self-distillation_with_adaptive_modulation_leveraging_d.md)**

:   FRAMER 提出频率对齐的自蒸馏训练框架，通过将最终层特征图作为教师监督中间层，并按低频/高频分别施加 IntraCL 和 InterCL 对比损失，配合自适应权重调节(FAW)和对齐门控(FAM)，在不改变网络结构和推理流程的情况下，显著提升扩散模型在真实图像超分辨率任务的高频细节恢复能力。

**[Frequency-Aware Flow Matching for High-Quality Image Generation](image_generation/freqflow_frequency_aware_flow_matching.md)**

:   FreqFlow 通过在流匹配框架中显式引入频域感知条件，采用双分支架构分别处理低频全局结构和高频细节信息，在 ImageNet-256 上以 1.38 FID 达到 SOTA。

**[From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](image_generation/from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)**

:   本文观察到图像图层分解（layer decomposition）与图像修复/外绘（inpainting/outpainting）任务之间的内在联系，提出 Outpaint-and-Remove 方法，通过轻量级 LoRA 微调将预训练的 inpainting DiT 模型（FLUX.1-Fill-dev）高效适配为图层分解模型，同时引入多模态上下文融合模块保留细节，仅用 10 万合成训练数据即达到 SOTA 性能。

**[Garments2Look: A Multi-Reference Dataset for High-Fidelity Outfit-Level Virtual Try-On with Clothing and Accessories](image_generation/garments2look_a_multi-reference_dataset_for_high-fidelity_outfit-level_virtual_t.md)**

:   提出 Garments2Look，首个大规模多模态整套搭配级虚拟试穿数据集（80K 对，40 类，300+ 子类），每组包含 3-12 件参考服饰图、模特穿搭图和详细文本标注，揭示现有方法在多层搭配和配饰一致性上的重大不足。

**[Gaussian Shannon: High-Precision Diffusion Model Watermarking Based on Communication](image_generation/gaussian_shannon_high-precision_diffusion_model_watermarking_based_on_communicat.md)**

:   将扩散模型的水印嵌入和提取过程建模为噪声信道通信，提出 Gaussian Shannon 框架，通过级联的多数投票和 LDPC 纠错码实现水印的比特精确恢复（而非仅阈值检测），在三种 Stable Diffusion 版本和七种扰动下达到 SOTA 的比特精度和检测率。

**[GIST: Towards Design Compositing](image_generation/gist_towards_design_compositing.md)**

:   提出 GIST，一种免训练的身份保持图像合成方法，通过交叉注意力引导的 token 注入和 Flow Matched 潜空间初始化，在布局预测和排版生成之间作为即插即用的合成阶段，实现多来源视觉元素的风格协调。

**[gQIR: Generative Quanta Image Reconstruction](image_generation/gqir_generative_quanta_image_reconstruc_tion.md)**

:   将大规模 text-to-image latent diffusion model 适配到单光子雪崩二极管（SPAD）的极端光子受限成像场景，通过三阶段框架（Quanta-aligned VAE → 对抗微调 LoRA U-Net → FusionViT 时空融合）实现从稀疏二值光子检测到高质量 RGB 图像的重建，在 10K-100K fps 极端条件下显著超越所有现有方法。

**[GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](image_generation/groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)**

:   GrOCE 提出基于动态语义图的免训练概念擦除框架，通过构建语义图→自适应聚类识别→选择性切除三个协同组件，实现对文本到图像扩散模型中目标概念的精确、上下文感知的在线移除。

**[Group Editing: Edit Multiple Images in One Go](image_generation/group_editing_edit_multiple_images_in_one_go.md)**

:   本文提出 GroupEditing，将一组相关图像重构为伪视频帧，结合 VGGT 提供的显式几何对应和视频模型的隐式时序先验，通过 Ge-RoPE 和 Identity-RoPE 两种增强位置编码实现跨视角一致的群组图像编辑，在视觉质量、编辑一致性和语义对齐上显著优于现有方法。

**[Guiding a Diffusion Model by Swapping Its Tokens](image_generation/guiding_a_diffusion_model_by_swapping_its_tokens.md)**

:   本文提出 Self-Swap Guidance (SSG)，一种无需条件信息的扩散模型采样引导方法，通过在模型中间表示空间中选择性地交换语义最不相似的 token 对来构造扰动版本，相比 SAG/PAG/SEG 等方法在更宽的引导强度范围内稳定生成高保真图像，在条件和无条件生成上均取得最优 FID。

**[Guiding a Diffusion Transformer with the Internal Dynamics of Itself](image_generation/guiding_a_diffusion_transformer_with_the_internal_dynamics_of_itself.md)**

:   本文提出 Internal Guidance (IG)，通过在 Diffusion Transformer 的中间层添加辅助监督损失使其产生较弱的生成输出，然后在采样时外推中间层和深层输出的差异来实现类似 Autoguidance 的引导效果，无需额外采样步骤或外部模型训练，在 ImageNet 256×256 上将 LightningDiT-XL/1 的 FID 推至 1.34（无 CFG）和 1.19（+CFG），达到同期 SOTA。

**[Guiding Diffusion Models with Semantically Degraded Conditions](image_generation/guiding_diffusion_models_with_semantically_degraded_conditions.md)**

:   提出 Condition-Degradation Guidance (CDG)，用语义退化的条件 $\boldsymbol{c}_{\text{deg}}$ 替代 CFG 中的空提示 $\emptyset$，将引导从粗粒度"好 vs. 空"转变为细粒度"好 vs. 差一点"的对比，通过分层退化策略（先退化内容 token 再退化上下文聚合 token）构建自适应负样本，在 SD3/FLUX/Qwen-Image 等模型上即插即用地提升组合生成精度，几乎零额外开销。

**[HaltNav: Reactive Visual Halting over Lightweight Topological Priors for Robust Vision-Language Navigation](image_generation/haltnav_reactive_visual_halting_over_lightweight_topological_priors_for_robust_v.md)**

:   提出 HaltNav，一个层级化导航框架，结合轻量级文本拓扑先验（osmAG）做全局规划，用 VLN 模型做局部执行，并通过 Reactive Visual Halting 机制检测意外障碍、动态更新拓扑并重规划，在仿真和真机上均显著提升长程导航鲁棒性。

**[HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](image_generation/ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)**

:   提出 HAM，一种无需训练的风格迁移方法，通过对扩散模型中 self-attention 和 cross-attention 实施异构调制（GAR+LAT），并配合风格注入式噪声初始化，在不牺牲内容身份信息的前提下实现高质量风格迁移，在多项指标上达到 SOTA。

**[HazeMatching: Dehazing Light Microscopy Images with Guided Conditional Flow Matching](image_generation/hazematching_dehazing_light_microscopy_images_with_guided_conditional_flow_match.md)**

:   提出 HazeMatching，一种基于引导式条件流匹配（Guided CFM）的显微图像去雾方法，通过在速度场中引入退化观测条件，在不需要显式退化算子的前提下，同时实现高数据保真度和高感知质量，并能生成校准良好的不确定性估计。

**[Heterogeneous Decentralized Diffusion Models](image_generation/heterogeneous_decentralized_diffusion_models.md)**

:   提出异构去中心化扩散框架，允许不同专家使用不同扩散目标（DDPM ε-prediction 与 Flow Matching velocity-prediction）完全独立训练，在推理时通过确定性 schedule-aware 转换统一到速度空间进行融合，相比同构基线同时提升 FID 和生成多样性，并将计算量压缩 16 倍。

**[HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images](image_generation/hifi-inpaint_towards_high-fidelity_reference-based_inpainting_for_generating_det.md)**

:   提出 HiFi-Inpaint 框架，通过共享增强注意力（SEA）利用高频信息增强产品细节特征，结合细节感知损失（DAL）实现像素级高频监督，在人-产品图像生成中达到 SOTA 的细节保真度。

**[High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](image_generation/high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)**

:   提出身份约束的属性调优框架用于扩散模型人脸替换：先约束身份解空间，再注入属性条件，最后端到端精炼身份损失和对抗损失，结合解耦条件注入设计，在 FFHQ 上实现 SOTA 的 FID（3.61）和身份检索准确率（97.9% Top-1）。

**[Image Diffusion Preview with Consistency Solver](image_generation/image_diffusion_preview_with_consistency_solver.md)**

:   本文提出 Diffusion Preview 范式和 ConsistencySolver——一个基于强化学习训练的轻量级高阶 ODE 求解器，在低步数采样时生成高质量预览图像并确保与全步数输出的一致性，用 47% 更少的步数达到与 Multistep DPM-Solver 相当的 FID，用户交互时间减少近 50%。

**[Image Generation as a Visual Planner for Robotic Manipulation](image_generation/image_generation_as_a_visual_planner_for_robotic_manipulation.md)**

:   将预训练图像生成模型（DiT）通过 LoRA 微调适配为机器人操作的视觉规划器，以 3×3 网格图像形式生成时序连贯的操作序列，支持文本条件和轨迹条件两种控制模式。

**[Imagine Before Concentration: Diffusion-Guided Registers Enhance Partially Relevant Video Retrieval](image_generation/imagine_before_concentration_diffusion-guided_registers_enhance_partially_releva.md)**

:   本文提出 DreamPRVR，采用"先想象后集中"的粗到细策略：通过截断扩散模型在文本监督下生成全局语义注册令牌（registers），然后将其融合到细粒度视频表征中，有效抑制局部噪音响应，在三个 PRVR 基准上取得了 SOTA。

**[InnoAds-Composer: Efficient Condition Composition for E-Commerce Poster Generation](image_generation/innoads-composer_efficient_condition_composition_for_e-commerce_poster_generatio.md)**

:   提出 InnoAds-Composer，一个基于 MM-DiT 的单阶段电商海报生成框架，通过统一 token 化将商品主体、字形文本和背景风格三类条件映射到同一空间，结合文本特征增强模块（TFEM）和重要性感知条件注入策略，在保持高质量生成的同时显著降低推理开销。

**[InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](image_generation/interedit_navigating_textguided_multihuman_3d_moti.md)**

:   首次定义文本引导的多人3D运动编辑(TMME)任务，构建含5161个源-目标-指令三元组的InterEdit3D数据集，提出InterEdit条件扩散模型——通过语义感知规划Token对齐捕捉高层编辑意图、交互感知频域Token对齐建模周期性交互动态，在指令跟随(g2t R@1 30.82%)和源保持(g2s R@1 17.08%)上全面超越4个基线。

**[Interpretable and Steerable Concept Bottleneck Sparse Autoencoders](image_generation/interpretable_and_steerable_concept_bottleneck_sparse_autoencoders.md)**

:   揭示了SAE中大多数神经元（~81%）的可解释性或可控性不足的问题，提出CB-SAE框架——通过裁剪低效用SAE神经元并增加概念瓶颈模块，在LVLM和图像生成任务上分别提升可解释性+32.1%和可控性+14.5%。

**[Intra-finger Variability of Diffusion-based Latent Fingerprint Generation](image_generation/intra_finger_variability_of_diffusion_based_latent_fingerprint_generation.md)**

:   本文系统评估了扩散模型合成指纹的同指变异性，通过构建包含40种表面和15种处理技术的潜指纹风格库提升生成多样性，并量化了生成过程中引入的局部/全局身份不一致性。

**[Intrinsic Concept Extraction Based on Compositional Interpretability](image_generation/intrinsic_concept_extraction_based_on_compositional_interpretability.md)**

:   HyperExpress 提出组合可解释本征概念提取（CI-ICE）新任务，利用双曲空间的层次建模能力和等球面投影模块，从单张图像中提取可组合的物体级和属性级概念，实现可逆的复杂视觉概念分解。

**[Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](image_generation/just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)**

:   提出 Just-in-Time (JiT) 框架，通过在空间域动态选择稀疏 anchor token 驱动生成 ODE 演化，并设计确定性 micro-flow 保证新 token 无缝激活，在 FLUX.1-dev 上实现最高 7× 加速且几乎无损。

**[Language-Free Generative Editing from One Visual Example](image_generation/language-free_generative_editing_from_one_visual_example.md)**

:   揭示文本引导扩散模型在雨、雾、模糊等简单视觉变换上存在严重的文本-视觉对齐失败，提出VDC框架——仅需一对视觉示例（变换前后）学习纯视觉条件信号来引导扩散编辑，无需文本、无需训练，在去雨/去雾/去噪等任务上超越文本和微调方法。

**[Layer Consistency Matters: Elegant Latent Transition Discrepancy for Generalizable Synthetic Image Detection](image_generation/layer_consistency_matters_elegant_latent_transition_discrepancy_for_generalizabl.md)**

:   发现真实图像在冻结CLIP ViT中间层的特征表示呈现稳定的层间过渡，而合成图像在中间层出现显著的注意力突变，提出Layer Transition Discrepancy (LTD) 方法建模该差异，在UFD上mean Acc达96.90%，DRCT-2M上达99.54%，GenImage上达91.62%，全面超越SOTA。

**[LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](image_generation/leapalign_post_training_flow_matching_models_at_any_generation_step.md)**

:   提出 LeapAlign，通过构建两步跳跃轨迹将长生成路径缩短为两步，使奖励梯度可直接反向传播到早期生成步骤，结合轨迹相似性加权和梯度折扣策略实现 flow matching 模型的高效后训练对齐。

**[Learnability-Guided Diffusion for Dataset Distillation](image_generation/learnability-guided_diffusion_for_dataset_distillation.md)**

:   提出可学习性驱动的增量式数据集蒸馏框架LGD，将蒸馏数据集分阶段构建，每阶段条件化于当前模型状态生成互补而非冗余的训练样本，通过在扩散采样中注入可学习性梯度引导，将现有方法80-90%的样本间信息冗余降低39.1%，在ImageNet-1K上达60.1%（50 IPC）、ImageNette上达87.2%（100 IPC）。

**[Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: Towards Robust Zero-Shot Skeleton Action Recognition](image_generation/learning_by_neighbor-aware_semantics_deciding_by_open-form_flows_towards_robust_.md)**

:   Flora 通过邻居感知语义校准实现稳健的骨架-语义跨模态对齐，并利用无噪声流匹配构建分布感知的开放式分类器，在零样本骨架动作识别上取得 SOTA，尤其在低样本训练场景中表现突出。

**[Learning Latent Proxies for Controllable Single-Image Relighting](image_generation/learning_latent_proxies_for_controllable_single-image_relighting.md)**

:   提出 LightCtrl，一个基于扩散模型的单图重光照框架，通过小样本潜在代理编码器（few-shot latent proxy）提供轻量材质-几何先验、光照感知掩码引导空间选择性去噪、DPO 后训练增强物理一致性，实现对光照方向/强度/色温的精确连续控制，在合成和真实场景上均优于现有方法。

**[Learning Latent Transmission and Glare Maps for Lens Veiling Glare Removal](image_generation/learning_latent_transmission_and_glare_maps_for_lens_veiling_glare_removal.md)**

:   提出 VeilGen + DeVeiler 框架，通过物理引导的 Stable Diffusion 生成模型学习潜在透射率和眩光图以合成逼真的复合退化训练数据，并用可逆约束训练修复网络，实现简化光学系统中像差与雾化眩光的联合去除。

**[Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](image_generation/learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)**

:   提出 GvU，利用统一多模态模型（UMM）自身的视觉理解分支作为内在奖励信号，通过 token 级文图对齐概率构建自监督 RL 框架（基于 GRPO），在无外部监督下迭代提升 T2I 生成质量，GenEval++ 上实现 43.3% 提升，且生成增强反过来促进细粒度理解。

**[LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration](image_generation/lesa_learnable_stage-aware_predictors_for_diffusion_model_acceleration.md)**

:   提出 LESA 框架，用 KAN（Kolmogorov-Arnold Network）作为可学习时序预测器，结合多阶段多专家架构和两阶段训练策略，在 FLUX 上实现 5× 加速仅 1.0% 质量下降，在 Qwen-Image 上 6.25× 加速比 TaylorSeer 质量提升 20.2%，在 HunyuanVideo 上 5× 加速 PSNR 提升 24.7%。

**[Leveraging Multispectral Sensors for Color Correction in Mobile Cameras](image_generation/leveraging_multispectral_sensors_for_color_correction_in_mobile_cameras.md)**

:   提出一个统一的端到端色彩校正框架，联合融合高分辨率RGB传感器和辅助低分辨率多光谱(MS)传感器的数据，将光源估计、光源补偿和色彩空间转换整合在单一模型中，色彩误差($\Delta E_{00}$)相比纯RGB和MS基线降低高达50%。

**[Low-Resolution Editing is All You Need for High-Resolution Editing](image_generation/low-resolution_editing_is_all_you_need_for_high-resolution_editing.md)**

:   ScaleEdit 首次提出高分辨率图像编辑任务，通过在预训练生成模型的中间特征空间学习 1×1 卷积迁移函数来注入源图像的精细纹理细节，配合基于 Blended-Tweedie 的分块同步策略保证全局一致性，以测试时优化方式实现 2K 甚至 8K 分辨率的高质量编辑。

**[LumiCtrl: Learning Illuminant Prompts for Lighting Control in Personalized Text-to-Image Models](image_generation/lumictrl_learning_illuminant_prompts_for_lighting_control_in_personalized_text-t.md)**

:   发现T2I模型文本编码器无法理解标准光照术语（如 tungsten、6500K）的语义鸿沟，提出 LumiCtrl 通过物理光照增强、边缘引导 prompt 解耦和掩码重建损失三个组件学习光照 prompt，在保持目标概念身份的同时实现精确的文本引导光照控制。

**[MAGIC: Few-Shot Mask-Guided Anomaly Inpainting with Prompt Perturbation, Spatially Adaptive Guidance, and Context Awareness](image_generation/magic_few-shot_mask-guided_anomaly_inpainting_with_prompt_perturbation_spatially.md)**

:   提出 MAGIC 框架，通过微调 inpainting 扩散模型，结合高斯 prompt 扰动、掩码引导空间噪声注入和上下文感知掩码对齐三个互补模块，在少样本条件下生成高保真、多样化、空间合理的工业异常图像，在 MVTec-AD 下游任务上达到 SOTA。

**[Match-and-Fuse: Consistent Generation from Unstructured Image Sets](image_generation/match-and-fuse_consistent_generation_from_unstructured_image_sets.md)**

:   提出 Match-and-Fuse，首个面向非结构化图像集合的训练无关一致性生成方法。以图为节点、图对为边建立成对一致性图，通过多视角特征融合（MFF）和特征引导在扩散推理中操控内部特征，实现集合级跨图一致性，DINO-MatchSim 达 0.80 远超所有基线。

**[Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping](image_generation/memory-efficient_fine-tuning_diffusion_transformers_via_dynamic_patch_sampling_a.md)**

:   提出 DiT-BlockSkip 框架，通过时间步感知的动态补丁采样（低分辨率训练但动态调整裁剪范围）和基于交叉注意力分析的关键块选择+残差特征预计算的块跳过策略，在 FLUX 上将 LoRA 微调显存减少约 50%，同时维持与标准 LoRA 可比的个性化生成质量。

**[MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](image_generation/micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)**

:   提出 MICON-Bench，覆盖 6 项任务（1043 案例）的多图上下文生成基准，配合 MLLM 驱动的 Evaluation-by-Checkpoint 自动评估框架；同时提出 DAR（Dynamic Attention Rebalancing）训练无关机制，通过动态调整推理时注意力权重提升 UMM 的多图生成一致性和质量。

**[Mixture of States: Routing Token-Level Dynamics for Multimodal Generation](image_generation/mixture_of_states_routing_token-level_dynamics_for_multimodal_generation.md)**

:   提出 Mixture of States (MoS)——一种基于可学习 token 级稀疏路由的多模态融合范式，使视觉 token 能在每个去噪步骤自适应地从文本编码器任意层选取隐藏状态，仅用 3-5B 参数即可匹敌或超越 20B 级模型。

**[MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing](image_generation/morphany3d_unleashing_the_power_of_structured_latent_in_3d_morphing.md)**

:   提出 MorphAny3D，首个基于 Structured Latent（SLAT）表示的无训练 3D 变形框架，通过 Morphing Cross-Attention（MCA）融合源/目标信息保证结构合理、Temporal-Fused Self-Attention（TFSA）增强时序一致性、方向校正策略消除突变，在跨类别 3D 变形中实现了 SOTA 质量。

**[MOS: Mitigating Optical-SAR Modality Gap for Cross-Modal Ship Re-Identification](image_generation/mos_mitigating_optical-sar_modality_gap_for_cross-modal_ship_re-identification.md)**

:   提出 MOS 框架解决光学-SAR 跨模态船舶重识别问题，包含两个核心模块：(1) MCRL 通过 SAR 图像去噪和类别级模态对齐损失在训练阶段缩小模态差距；(2) CDGF 利用布朗桥扩散模型在推理阶段从光学图像生成伪 SAR 样本并融合特征，在 HOSS ReID 数据集上 SAR→Optical 的 R1 提升 +16.4%。

**[MPDiT: Multi-Patch Global-to-Local Transformer Architecture for Efficient Flow Matching](image_generation/mpdit_multi-patch_global-to-local_transformer_architecture_for_efficient_flow_ma.md)**

:   提出 MPDiT，一个多尺度 patch 的全局到局部扩散 Transformer 架构，前期用大 patch（4×4）处理全局上下文仅需 64 个 token，后期上采样到小 patch（2×2）的 256 个 token 精修局部细节，将 GFLOPs 降低高达 50%，且 XL 模型在 240 epoch 即达到 FID 2.05（cfg）。

**[MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation](image_generation/multibanana_a_challenging_benchmark_for_multi_reference_text_to_image_generation.md)**

:   提出MultiBanana——首个系统评估多参考图像生成能力的大规模基准，包含3769个评测样本、最多8张参考图、5个难度维度（跨域/尺度/稀有概念/多语言），揭示了闭源模型"过拟合参考细节"和开源模型"忽略参考主体"的互补失败模式。

**[Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](image_generation/neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)**

:   提出 NLCE，一个 training-free 的三阶段概念擦除框架，通过谱加权表征调制、注意力引导空间门控和门控特征清理三步实现目标概念的精确局部擦除，同时显式保留语义邻近概念，在 Oxford Flowers、Stanford Dogs、名人身份和敏感内容擦除任务上均优于现有方法。

**[Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models](image_generation/neighbor_grpo_contrastive_ode_policy_optimization_aligns_flow_models.md)**

:   重新解释 SDE-based GRPO 为距离优化/对比学习，提出 Neighbor GRPO——完全绕过 SDE 转换，通过扰动 ODE 初始噪声构建邻域候选轨迹 + softmax 距离代理策略实现策略梯度优化，保留确定性 ODE 采样的所有优势。

**[OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution](image_generation/oars_process-aware_online_alignment_for_generative_real-world_image_super-resolu.md)**

:   提出 OARS 框架，通过基于 MLLM 的过程感知奖励模型 COMPASS 和渐进式在线强化学习（冷启动→有参考 RL→无参考 RL），首次系统解决生成式真实世界图像超分辨率中的人类偏好对齐问题，在保持保真度的同时显著提升感知质量。

**[Object-WIPER: Training-Free Object and Associated Effect Removal in Videos](image_generation/object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)**

:   提出 Object-WIPER，首个无训练的视频物体及其关联效应（阴影、反射、镜像等）移除框架，利用 DiT 中的文本-视觉交叉注意力和视觉自注意力定位关联效应区域，通过前景重初始化和注意力缩放实现干净移除，并提出 TokSim 指标和 WIPER-Bench 真实世界基准。

**[One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers](image_generation/one_model_many_budgets_elastic_latent_interfaces_for_diffusion_transformers.md)**

:   提出 ELIT（Elastic Latent Interface Transformer），在 DiT 中插入可变长度的潜变量接口（latent interface）和轻量 Read/Write 跨注意力层，使单一模型能在推理时动态调节计算预算，同时将计算非均匀地分配到图像中更难的区域，在 ImageNet 512px 上 FID 最高降低 53%。

**[OpenDPR: Open-Vocabulary Change Detection via Vision-Centric Diffusion-Guided Prototype Retrieval for Remote Sensing Imagery](image_generation/opendpr_open-vocabulary_change_detection_via_vision-centric_diffusion-guided_pro.md)**

:   OpenDPR 提出了一种免训练的视觉中心框架，利用扩散模型离线生成目标类别的多样化视觉原型，在推理时通过视觉空间的相似度检索来识别遥感图像中的开放词汇变化，在四个基准数据集上取得 SOTA 性能。

**[OPRO: Orthogonal Panel-Relative Operators for Panel-Aware In-Context Image Generation](image_generation/opro_orthogonal_panel-relative_operators_for_panel-aware_in-context_image_genera.md)**

:   提出 OPRO，一种基于正交矩阵的参数高效适配方法，通过在 frozen backbone 的位置感知 query/key 上施加可学习的面板特异性正交算子，在保持预训练同面板合成行为的同时显式调制跨面板注意力交互，仅增加 0.93M 参数即在 MagicBrush 上显著提升多种 SOTA 方法的编辑质量。

**[Organizing Unstructured Image Collections using Natural Language](image_generation/organizing_unstructured_image_collections_using_natural_language.md)**

:   本文定义了开放式语义多聚类（OpenSMC）新任务，并提出 X-Cluster 框架，利用 MLLM 将图像转为文本后通过 LLM 自动发现聚类标准和语义子结构，无需任何人类先验输入即可将大规模无标签图像集组织为多维度、多粒度、可解释的语义聚类。

**[PhysGen: Physically Grounded 3D Shape Generation for Industrial Design](image_generation/physgen_physically_grounded_3d_shape_generation_for_industrial_design.md)**

:   本文提出 PhysGen，一个将物理约束（空气动力学效率）融入 3D 形状生成的统一框架：通过 Shape-and-Physics VAE 将几何和物理信息联合编码到统一潜空间，然后用交替更新的 Flow Matching 模型在速度更新和物理精炼之间迭代，生成既视觉逼真又物理高效的 3D 形状（如低阻力系数的汽车）。

**[Physics-Consistent Diffusion for Efficient Fluid Super-Resolution via Multiscale Residual Correction](image_generation/physics-consistent_diffusion_for_efficient_fluid_super-resolution_via_multiscale.md)**

:   提出 ReMD（Residual-Multigrid Diffusion），在扩散模型的每一步反向采样中嵌入多重网格残差修正，利用多小波基构建跨尺度层次结构，无需显式 PDE 即可实现物理一致的高效流体超分辨率。

**[Pixel Motion Diffusion Is What We Need for Robot Control](image_generation/pixel_motion_diffusion_is_what_we_need_for_robot_control.md)**

:   DAWN 提出两阶段全扩散框架——Motion Director 生成稠密像素运动场作为可解释中间表征，Action Expert 将其转化为可执行机器人动作序列，在 CALVIN（Avg Len 4.00）、MetaWorld（Overall 65.4%）和真实世界均达到 SOTA，且模型容量和训练数据远小于竞争方法。

**[PixelDiT: Pixel Diffusion Transformers for Image Generation](image_generation/pixeldit_pixel_diffusion_transformers_for_image_generation.md)**

:   PixelDiT 提出完全基于Transformer的双层像素空间扩散模型：patch级DiT捕捉全局语义 + pixel级DiT细化纹理细节，无需VAE即可在ImageNet上达到1.61 FID，并直接在1024分辨率像素空间训练文本到图像模型。

**[PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](image_generation/pixelrush_ultrafast_trainingfree_highresolution_im.md)**

:   PixelRush是首个将免训练高分辨率图像生成推入实用化的方法——通过部分DDIM反转跳过冗余的低频重建步骤，使少步扩散模型在patch精炼中可行，配合高斯滤波融合和噪声注入消除伪影，4秒生成2K图像、20秒生成4K图像，比SOTA快10-35倍且FID更优。

**[Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](image_generation/pluggable_pruning_with_contiguous_layer_distillation_for_diffusion_transformers.md)**

:   提出 PPCL 框架，通过线性探针检测 MMDiT 中连续冗余层区间，结合非顺序蒸馏实现深度剪枝（即插即用）和宽度剪枝（用线性投影替换文本流/FFN），将 Qwen-Image 从 20B 压缩到 10B 时性能仅下降 3.29%。

**[Pose-dIVE: Pose-Diversified Augmentation with Diffusion Model for Person Re-Identification](image_generation/pose-dive_pose-diversified_augmentation_with_diffusion_model_for_person_re-ident.md)**

:   Pose-dIVE通过SMPL模型联合控制人体姿态和相机视角，利用扩散模型生成具有多样化姿态和视角的行人图像，系统性地弥补Re-ID训练数据中的分布偏差，在多个基准上持续提升任意Re-ID模型的泛化能力。

**[PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation](image_generation/posteriq_a_design_perspective_benchmark_for_poster_understanding_and_generation.md)**

:   本文提出 PosterIQ，一个面向海报设计的综合基准，包含 7,765 条理解标注和 822 条生成提示，覆盖 OCR、字体感知、布局推理、设计意图理解和组合感知生成等 24 类任务，系统评估了 MLLM 和扩散模型在设计认知方面的差距。

**[Precise Object and Effect Removal with Adaptive Target-Aware Attention](image_generation/precise_object_and_effect_removal_with_adaptive_target-aware_attention.md)**

:   提出 ObjectClear 框架，通过自适应目标感知注意力（ATA）将前景移除与背景重建解耦，配合注意力引导融合（AGF）和空间变化去噪强度（SVDS）策略，实现对目标物体及其阴影、反射等附带效果的精准移除，同时构建了首个大规模 Object-Effect Removal 数据集 OBER。

**[Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](image_generation/preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)**

:   提出 LivingSwap，首个视频参考引导的人脸替换模型，通过关键帧身份注入 + 源视频参考补全 + 时序拼接的可控流水线，实现长视频中的高保真人脸替换，在保持源视频表情、光照、运动等细节的同时稳定注入目标身份，将人工编辑量减少 40 倍。

**[Probing and Bridging Geometry–Interaction Cues for Affordance Reasoning in Vision Foundation Models](image_generation/probing_and_bridging_geometry-interaction_cues_for_affordance_reasoning_in_visio.md)**

:   系统性地探测视觉基础模型（VFM）中的可供性（affordance）能力，发现 DINO 编码了部件级几何结构、Flux 编码了动词条件化的交互先验，并通过 training-free 融合两者实现了可与弱监督方法竞争的零样本可供性估计。

**[PROMO: Promptable Outfitting for Efficient High-Fidelity Virtual Try-On](image_generation/promo_promptable_virtual_tryon_efficient.md)**

:   PROMO基于FLUX Flow Matching DiT骨干，通过潜空间多模态条件拼接、时序自参考KV缓存、3D-RoPE分组条件、以及fine-tuned VLM风格提示系统，在去除传统参考网络的前提下实现了高保真且高效的多件服装虚拟试穿，推理速度比无加速版快2.4倍，在VITON-HD和DressCode上超越现有VTON和通用图像编辑方法。

**[Prototype-Guided Concept Erasure in Diffusion Models](image_generation/prototype-guided_concept_erasure_in_diffusion_models.md)**

:   针对扩散模型中宽泛概念（如暴力、色情）难以彻底擦除的问题，提出基于概念原型的 training-free 擦除方法：通过聚类 CLIP 嵌入空间中的概念差分方向获取图像原型，再优化迁移到文本原型空间，推理时选择最匹配的原型作为负引导信号进行 classifier-free guidance 式的概念抑制。

**[PSDesigner: Automated Graphic Design with a Human-Like Creative Workflow](image_generation/psdesigner_automated_graphic_design_with_a_human-like_creative_workflow.md)**

:   本文提出PSDesigner，一个模拟人类设计师创意工作流的自动图形设计系统，通过AssetCollector（资源收集）、GraphicPlanner（规划工具调用）和ToolExecutor（执行PSD操作）三个模块协作，利用首个PSD格式设计数据集CreativePSD训练模型学习专业设计流程，能直接生成可编辑的PSD设计文件。

**[PSR: Scaling Multi-Subject Personalized Image Generation with Pairwise Subject-Consistency Rewards](image_generation/psr_scaling_multi-subject_personalized_image_generation_with_pairwise_subject-co.md)**

:   针对多主体个性化图像生成中主体一致性差和文本遵循不足的问题，提出可扩展的多主体数据构建管线和成对主体一致性奖励（PSR），通过两阶段训练（SFT + RL）在自建的 PSRBench 上全面超越现有 SOTA。

**[PureCC: Pure Learning for Text-to-Image Concept Customization](image_generation/purecc_pure_learning_for_text-to-image_concept_customization.md)**

:   提出 PureCC 方法，通过分离"目标概念隐式引导"和"原始条件预测"的解耦学习目标，配合冻结表示提取器+可训练流模型的双分支训练管线和自适应引导缩放 $\lambda^{\star}$，实现高保真概念定制的同时最小化对原始模型行为和能力的影响。

**[Quantization with Unified Adaptive Distillation to enable multi-LoRA based one-for-all Generative Vision Models on edge](image_generation/quantization_with_unified_adaptive_distillation_to_enable_multi-lora_based_one-f.md)**

:   本文提出QUAD框架，将LoRA权重作为运行时输入而非编译到模型图中，结合跨LoRA共享量化参数的蒸馏微调策略，实现单个编译模型在移动端NPU上动态切换多个GenAI任务，达到6倍内存压缩和4倍延迟改善。

**[RAISE: Requirement-Adaptive Evolutionary Refinement for Training-Free Text-to-Image Alignment](image_generation/raise_requirement-adaptive_evolutionary_refinement_for_training-free_text-to-ima.md)**

:   提出 RAISE 框架，将 T2I 生成建模为需求驱动的自适应进化过程：通过需求分析器将提示词分解为结构化检查清单，用多动作变异（提示重写+噪声重采样+指令编辑）并发进化候选群体，再通过工具增强的视觉验证逐轮淘汰不满足需求的候选，实现自适应推理时缩放——在 GenEval 上达到 0.94 SOTA，同时比反射微调基线减少 30-40% 生成样本和 80% VLM 调用。

**[RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models](image_generation/razor_ratio-aware_layer_editing_for_targeted_unlearning_in_vision_transformers_a.md)**

:   提出 RAZOR，一种基于比率感知的多层/多头选择性编辑框架，可在 CLIP、Stable Diffusion 和 VLM 等 Transformer 视觉模型中高效精准地完成目标遗忘，同时保持模型整体性能与量化鲁棒性。

**[RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark](image_generation/realunify_do_unified_models_truly_benefit_from_unification_a_comprehensive_bench.md)**

:   本文提出 RealUnify，首个专门评估统一模型中理解与生成能力双向协同效果的基准，通过1000个人工标注实例和直接/分步双重评估协议，揭示了当前统一模型虽然具备理解和生成能力，但在端到端场景中仍无法实现真正的能力协同。

**[Refining Few-Step Text-to-Multiview Diffusion via Reinforcement Learning](image_generation/refining_few-step_text-to-multiview_diffusion_via_reinforcement_learning.md)**

:   提出 MVC-ZigAL 框架，通过多视图感知 MDP 建模、zigzag 自反思优势学习和 Lagrangian 对偶约束优化，有效提升少步文本到多视图扩散模型的单视图保真度和跨视图一致性。

**[Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing](image_generation/rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)**

:   本文发现图像patch对之间的关系距离在AI编辑后保持不变，并利用该不变性构建了一种零水印框架Rel-Zero，无需修改原图即可实现对多种生成式编辑的鲁棒内容认证。

**[RenderFlow: Single-Step Neural Rendering via Flow Matching](image_generation/renderflow_single-step_neural_rendering_via_flow_matching.md)**

:   提出 RenderFlow，将神经渲染重新建模为从 albedo 到全光照图像的单步条件流匹配问题，以 G-buffer 为条件、预训练视频 DiT 为骨干，实现了比扩散方法快 10 倍以上（~0.19s/帧）的确定性渲染，可选的稀疏关键帧引导进一步提升物理精度，还支持通过冻结骨干 + 轻量 adapter 实现逆渲染。

**[Resolving the Identity Crisis in Text-to-Image Generation](image_generation/resolving_the_identity_crisis_in_text-to-image_generation.md)**

:   本文揭示了文本到图像模型在多人场景生成中的"身份危机"问题（重复面孔、身份合并），提出 DisCo 框架，通过组合式奖励函数和 GRPO 强化学习微调 flow-matching 模型，实现了 98.6% 的唯一面孔准确率，超越包括 GPT-Image-1 在内的闭源模型。

**[Reviving ConvNeXt for Efficient Convolutional Diffusion Models](image_generation/reviving_convnext_for_efficient_convolutional_diffusion_models.md)**

:   本文提出FCDM（Fully Convolutional Diffusion Model），将ConvNeXt架构适配为条件扩散模型backbone，仅用DiT-XL 50%的FLOPs即可在ImageNet上达到竞争性FID（2.03），且能在4块RTX 4090上训练XL模型，展示了全卷积架构在生成建模中被严重低估的效率优势。

**[RewardFlow: Generate Images by Optimizing What You Reward](image_generation/rewardflow_generate_images_by_optimizing_what_you_reward.md)**

:   RewardFlow 提出一种无需反转的推理时框架，通过多奖励 Langevin 动力学融合语义对齐、感知保真度、局部定位、物体一致性和人类偏好等多种可微分奖励信号，在图像编辑和组合式生成任务上实现 SOTA 的编辑保真度和组合对齐效果。

**[Score2Instruct: Scaling Up Video Quality-Centric Instructions via Automated Dimension Scoring](image_generation/score2instruct_scaling_up_video_quality-centric_instructions_via_automated_dimen.md)**

:   Score2Instruct 提出了一个无需人工标注和闭源 API 的自动化视频质量指令生成管线 SIG，通过自动评估 14 个质量维度并用层级 CoT 聚合为完整质量推理文本，构建了 320K+ 条指令数据集 S2I，配合两阶段渐进式微调策略，使多个视频 LMM 同时获得质量评分和质量推理能力，在 5 个 VQA 数据集上 SRCC 平均提升 26-31%。

**[SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models](image_generation/seacache_spectral-evolution-aware_cache_for_accelerating_diffusion_models.md)**

:   提出 SeaCache，一种基于频谱演化感知（SEA）滤波器的无训练动态缓存策略，通过在频域中分离信号与噪声分量来测量时间步间的冗余度，显著提升扩散模型推理的延迟-质量权衡。

**[SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models](image_generation/segquant_a_semantics-aware_and_generalizable_quantization_framework_for_diffusio.md)**

:   提出 SegQuant 框架，通过基于静态计算图的语义分割量化（SegLinear）和硬件原生的双尺度极性保持量化（DualScale），在不依赖手工规则或运行时动态信息的前提下，实现了跨架构通用、部署管线兼容的扩散模型高保真后训练量化。

**[Self-Corrected Image Generation with Explainable Latent Rewards](image_generation/self-corrected_image_generation_with_explainable_latent_rewards.md)**

:   提出 xLARD 框架，在文生图生成过程中通过一个轻量残差修正器在潜空间进行语义自修正，利用可解释的潜空间奖励信号（计数/颜色/位置）引导生成，在 GenEval 上提升 +4.1%，DPGBench 上提升 +2.97%，且以即插即用方式适配多种 backbone。

**[SHOE: Semantic HOI Open-Vocabulary Evaluation Metric](image_generation/shoe_semantic_hoi_open-vocabulary_evaluation_metric.md)**

:   提出SHOE评估框架，通过将HOI预测分解为动词和物体分别计算LLM驱动的语义相似度，替代传统mAP的精确匹配方式，在开放词汇HOI检测评估中达到85.73%的人类判断一致性，超过人类标注者之间78.61%的平均一致性。

**[ShowTable: Unlocking Creative Table Visualization with Collaborative Reflection and Refinement](image_generation/showtable_unlocking_creative_table_visualization_with_collaborative_reflection_a.md)**

:   ShowTable 提出了"创意表格可视化"这一新任务（将数据表格生成为信息图），并设计了一个 MLLM（推理+反思）与扩散模型（生成+精修）协同的渐进式自纠错 pipeline，通过针对性训练的重写模块和用 RL 优化的精修模块，在自建的 TableVisBench 基准上显著提升所有基线模型的可视化质量。

**[SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images](image_generation/simlbr_learning_to_detect_fake_images_by_learning_to_detect_real_images.md)**

:   本文提出SimLBR，通过在DINOv3潜空间中将少量假图信息混入真图嵌入作为正则化手段，迫使检测器学习真实图像分布的紧致决策边界，从而实现对未知生成器的强泛化能力，在GenImage上平均准确率达94.54%，在硬测试集Chameleon上比AIDE提升25%准确率和70%召回率。

**[SJD-PAC: Accelerating Speculative Jacobi Decoding via Proactive Drafting and Adaptive Continuation](image_generation/sjd-pac_accelerating_speculative_jacobi_decoding_via_proactive_drafting_and_adap.md)**

:   本文分析了 Speculative Jacobi Decoding (SJD) 在文本到图像生成中接受长度分布严重偏斜的瓶颈，提出 SJD-PAC 框架，通过 Proactive Drafting (PD) 和 Adaptive Continuation (AC) 两项技术，在严格无损的前提下实现 3.8× 推理加速，显著超越原始 SJD 的约 2× 加速。

**[SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](image_generation/slice_semantic_latent_injection_via_compartmentalized_embedding_for_image_waterm.md)**

:   提出 SLICE 语义水印框架，将图像语义分解为主体/环境/动作/细节四个因子并绑定到初始高斯噪声的不同空间分区，实现不仅可检测水印存在还可定位语义篡改的三状态验证机制，对最强 CSI 攻击的攻击成功率仅 19%（SEAL 为 81%）。

**[Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework](image_generation/smoothing_the_score_function_for_generalization_in_diffusion_models.md)**

:   本文从理论上证明扩散模型的记忆化问题源于经验得分函数中 softmax 权重的尖锐性（单个训练样本主导），并提出 Noise Unconditioning 和 Temperature Smoothing 两种平滑方法来缓解记忆化、增强泛化，同时保持生成质量。

**[SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](image_generation/solace_self_confidence_rewards_t2i.md)**

:   用T2I模型自身的去噪自信心（对注入噪声的恢复精度）作为内在奖励替代外部奖励模型做后训练，在组合生成、文字渲染、文图对齐上获一致提升，且与外部奖励互补可缓解reward hacking。

**[Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning](image_generation/spatial-ssrl_enhancing_spatial_understanding_via_self-supervised_reinforcement_l.md)**

:   本文提出Spatial-SSRL，一种自监督强化学习范式，通过从普通RGB/RGB-D图像自动构造五种pretext任务（patch重排、翻转识别、裁剪修补、深度排序、相对3D位置预测），利用GRPO优化LVLM的空间理解能力，在七个空间benchmark上平均提升3.89%-4.63%，且无需人工标注或外部工具。

**[SPDMark: Selective Parameter Displacement for Robust Video Watermarking](image_generation/spdmark_selective_parameter_displacement_for_robust_video_watermarking.md)**

:   SPDMark 提出了一种基于选择性参数位移（SPD）的视频扩散模型内嵌水印框架，通过在解码器中学习低秩基 shift 字典并根据水印密钥选择组合，实现了逐帧水印嵌入、不可感知、高鲁棒性和低计算开销，同时支持时序篡改检测与定位。

**[StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars](image_generation/streamavatar_streaming_diffusion_models_for_real-time_interactive_human_avatars.md)**

:   提出两阶段自回归适配加速框架（自回归蒸馏 + 对抗精炼），将双向人体视频扩散模型转化为实时流式生成器，通过 Reference Sink、RAPR 位置重编码和一致性感知判别器保证长视频稳定性，实现首个支持说话和倾听交互的全身实时数字人。

**[TAG-MoE: Task-Aware Gating for Unified Generative Mixture-of-Experts](image_generation/tag-moe_task-aware_gating_for_unified_generative_mixture-of-experts.md)**

:   针对统一图像生成与编辑模型中严重的任务干扰问题，提出 TAG-MoE 框架，通过层次化任务语义标注方案和预测性对齐正则化将高层任务意图注入 MoE 局部路由决策，使门控网络从任务无关的执行器进化为语义感知的调度中心，在 ICE-Bench、EmuEdit、GEdit、DreamBench++ 等五个基准上取得开源模型最优综合性能。

**[Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning](image_generation/taming_preference_mode_collapse_via_directional_decoupling_alignment_in_diffusio.md)**

:   提出 D2-Align 框架，通过在奖励模型嵌入空间中学习方向性修正向量来纠偏奖励信号，解决扩散模型 RLHF 对齐中的偏好模式坍塌（PMC）问题——即模型过度优化奖励导致生成多样性严重下降；同时提出 DivGenBench 基准用于量化评估生成多样性。

**[Taming Sampling Perturbations with Variance Expansion Loss for Latent Diffusion Models](image_generation/taming_sampling_perturbations_with_variance_expansion_loss_for_latent_diffusion_.md)**

:   揭示了潜在扩散模型中β-VAE tokenizer因方差坍缩导致潜空间过于紧凑、对扩散采样扰动极敏感的问题，提出Variance Expansion (VE) Loss通过重构与方差扩展的对抗式平衡来自适应学习鲁棒的潜空间方差，在多种扩散架构上一致提升生成质量（FID 1.18）。

**[Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](image_generation/taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)**

:   提出 AC-DC 三阶段去噪器（自动校正 + 方向校正 + Score 去噪），解决 ADMM 迭代与 score 训练流形不匹配的问题，并首次为 ADMM-PnP + score denoiser 建立了收敛性保证，在多种逆问题上取得 SOTA。

**[Taming Video Models for 3D and 4D Generation via Zero-Shot Camera Control](image_generation/taming_video_models_for_3d_and_4d_generation_via_zero-shot_camera_control.md)**

:   WorldForge 提出一个完全无训练的推理时引导框架，通过三个协同组件——步内递归精化（IRR）、光流门控潜变量融合（FLF）和双路径自校正引导（DSG）——将预训练视频扩散模型改造为精确相机轨迹可控的 3D/4D 生成工具，在轨迹精度和感知质量上同时超越训练式和推理式基线。

**[TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](image_generation/tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)**

:   提出 TAP 框架，通过第一层探针（probe）为每个 token 在每一步自适应选择最优预测器（Taylor 展开族），实现无需训练的扩散模型加速，在 FLUX.1-dev 上以 6.24× 加速且无感知质量损失。

**[TAUE: Training-free Noise Transplant and Cultivation Diffusion Model](image_generation/taue_training-free_noise_transplant_and_cultivation_diffusion_model.md)**

:   TAUE 提出一种**免训练**的分层图像生成框架，通过将去噪中间潜变量"移植"到新生成过程的初始噪声中，并结合跨层注意力共享，实现前景、背景和合成图像的三层一致生成，性能匹配甚至超越微调方法。

**[TC-Padé: Trajectory-Consistent Padé Approximation for Diffusion Acceleration](image_generation/tc-padé_trajectory-consistent_padé_approximation_for_diffusion_acceleration.md)**

:   提出基于 Padé 有理函数近似的特征残差预测框架 TC-Padé，通过自适应系数调节和分阶段感知策略，在低步数（20-30步）扩散采样场景下实现轨迹一致的加速（FLUX.1-dev 2.88×、Wan2.1 1.72×），显著优于基于 Taylor 展开的现有方法。

**[Test-Time Instance-Specific Parameter Composition: A New Paradigm for Adaptive Generative Modeling](image_generation/test-time_instance-specific_parameter_composition_a_new_paradigm_for_adaptive_ge.md)**

:   本文提出 Composer，一个即插即用的元生成器框架，在推理时根据每个输入条件动态生成低秩参数更新并注入预训练模型权重，以极低的计算开销（时间+0.2%、内存+3.6%）实现逐实例自适应的高质量图像生成，在类条件生成、文本到图像、后训练量化和测试时缩放等场景中均显著提升性能。

**[TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering](image_generation/textpecker_rewarding_structural_anomaly_quantification_for_enhancing_visual_text.md)**

:   提出 TextPecker——一种即插即用的结构异常感知 RL 策略，通过构建字符级结构异常标注数据集训练结构感知识别器，替代传统 OCR 的噪声奖励信号，联合优化语义对齐和结构保真度，在多个文本到图像模型（FLUX、SD3.5、Qwen-Image）上显著提升视觉文本渲染质量。

**[The Universal Normal Embedding](image_generation/the_universal_normal_embedding.md)**

:   提出 Universal Normal Embedding (UNE) 假说：生成模型（扩散模型）和视觉编码器（CLIP、DINO）的隐空间共享一个近似高斯的底层几何结构，二者可视为该共享空间的含噪线性投影；通过 NoiseZoo 数据集和大量实验验证了该假说，并展示了在 DDIM 反演噪声空间中直接进行线性语义编辑的能力。

**[TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models](image_generation/tina_text-free_inversion_attack_for_unlearned_text-to-image_diffusion_models.md)**

:   提出 TINA（Text-free INversion Attack），通过在 null-text 条件下优化 DDIM 反演找到精确的初始噪声，绕过所有基于文本的概念擦除防御，证明当前擦除方法仅切断了文本-图像映射而未真正删除模型内部的视觉知识。

**[Tiny Inference-Time Scaling with Latent Verifiers](image_generation/tiny_inference-time_scaling_with_latent_verifiers.md)**

:   提出VHS（Verifier on Hidden States）——一种直接在DiT生成器中间层隐状态上工作的验证器，跳过解码-重编码开销，在单步图像生成的推理时扩展（inference-time scaling）场景下将联合生成-验证时间减少63.3%、FLOPs降低51%，同时在GenEval上相同时间预算下提升2.7%的性能。

**[TokenLight: Precise Lighting Control in Images using Attribute Tokens](image_generation/tokenlight_precise_lighting_control_in_images_using_attribute_tokens.md)**

:   提出 TokenLight，将图像重光照表述为以属性 token（强度、颜色、环境光、漫反射级别、3D 光源位置）为条件的端到端图像生成任务，在扩散 Transformer 框架中实现精确、连续、可解释的光照控制。

**[Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity](image_generation/too_vivid_to_be_real_benchmarking_and_calibrating_generative_color_fidelity.md)**

:   针对 T2I 模型生成图像"太鲜艳不像真实照片"的问题，提出 Color Fidelity Dataset (CFD, 130 万图像)、Color Fidelity Metric (CFM, 基于 Qwen2-VL + softrank loss) 和 Color Fidelity Refinement (CFR, 无训练的时空自适应 guidance 调制)，形成评估-改善一体化框架。

**[Towards Robust Content Watermarking Against Removal and Forgery Attacks](image_generation/towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)**

:   提出实例特定双侧检测水印方法 ISTS，通过根据图像语义动态选择水印注入时间和位置来抵抗去除攻击和伪造攻击，并设计双侧检测机制抵御反向潜在表示攻击，在三种去除攻击和三种伪造攻击的平均和最坏情况下均达到 SOTA 鲁棒性。

**[TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking](image_generation/trace_structure-aware_character_encoding_for_robust_and_generalizable_document_w.md)**

:   提出 TRACE——基于字符结构编码的文档水印框架，利用扩散模型（DragDiffusion）精确移动字符骨架关键点来嵌入信息，通过自适应扩散初始化（ADI）、引导扩散编码（GDE）和掩码区域替换（MRR）三大组件，同时实现跨介质传输鲁棒性、多语言/多字体泛化性和高隐蔽性。

**[Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods](image_generation/training-free_detection_of_generated_videos_via_spatial-temporal_likelihoods.md)**

:   提出 STALL，一种无需训练的零样本生成视频检测器，通过在白化嵌入空间中联合建模逐帧空间似然和帧间时序似然，仅依赖真实视频校准即可实现对多种生成模型的鲁棒检测。

**[TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection](image_generation/tridf_evaluating_perception_detection_and_hallucination_for_interpretable_deepfa.md)**

:   提出TriDF——首个从感知 (Perception)、检测 (Detection) 和幻觉 (Hallucination) 三个维度综合评估可解释深度伪造检测的基准，包含55K高质量样本覆盖16种DeepFake类型和3种模态，揭示了准确感知是可靠检测的基础但幻觉会严重破坏决策的三方耦合关系。

**[Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](image_generation/uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)**

:   提出 Uni-DAD，首个将扩散模型蒸馏（distillation）与域适应（adaptation）统一为单阶段流程的方法，通过双域 DMD 损失和多头 GAN 损失，在仅 1–4 步采样下实现少样本域的高质量多样生成。

**[Unified Vector Floorplan Generation via Markup Representation](image_generation/unified_vector_floorplan_generation_via_markup_representation.md)**

:   本文提出 Floorplan Markup Language (FML) 标记语言，将房间、门等户型元素编码为结构化 token 序列，用一个 LLaMA 风格的 Transformer 模型（FMLM）统一解决无条件/边界条件/图条件/补全等多种户型图生成任务，FID 指标比 HouseDiffusion 低 80%+。

**[V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](image_generation/v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)**

:   将图像修复重新定义为渐进式视频生成过程，利用预训练视频模型（Wan2.2-TI2V-5B）的丰富视觉先验，仅用 1,000 个多任务训练样本（不到现有方法的 2%）就实现了多种退化类型的全能修复，超越了在百万级数据上训练的专用架构。

**[VeCoR — Velocity Contrastive Regularization for Flow Matching](image_generation/vecor_--_velocity_contrastive_regularization_for_flow_matching.md)**

:   提出 VeCoR（速度对比正则化），在标准 Flow Matching 训练中引入"负速度"对比信号，通过同时指导模型"该往哪走"和"不该往哪走"，实现更稳定的轨迹演化和更高的感知保真度——在 ImageNet-1K 上 SiT-XL/2 和 REPA-SiT-XL/2 分别获得 22% 和 35% 的 FID 相对降低。

**[Verify Claimed Text-to-Image Models via Boundary-Aware Prompt Optimization](image_generation/verify_claimed_text-to-image_models_via_boundary-aware_prompt_optimization.md)**

:   BPO 提出一种无需参考模型的白盒 T2I 模型验证方法，通过三阶段流程（对抗锚点识别→二分搜索边界探索→目标优化）找到模型特有的语义边界区域，生成的验证 prompt 在 5 个 T2I 模型上达到平均 96% 准确率和 0.93 F1，比 TVN 方法快 2 倍。

**[ViHOI: Human-Object Interaction Synthesis with Visual Priors](image_generation/vihoi_human-object_interaction_synthesis_with_visual_priors.md)**

:   提出ViHOI，一个即插即用框架，利用VLM从2D参考图像中提取解耦的视觉和文本先验，通过Q-Former压缩为紧凑条件token来增强扩散模型的HOI运动生成质量，推理时借助文生图模型合成参考图像实现对未见物体的强泛化。

**[Vinedresser3D: Agentic Text-guided 3D Editing](image_generation/vinedresser3d_agentic_text-guided_3d_editing.md)**

:   提出 Vinedresser3D，一个以多模态大语言模型（MLLM）为核心的 3D 编辑智能体，无需用户提供 3D 掩码，通过自动解析编辑意图、定位编辑区域、生成多模态引导，并在原生 3D 生成模型（Trellis）的潜空间中执行基于反演的修补编辑，实现高质量文本引导的 3D 资产编辑。

**[ViStoryBench: Comprehensive Benchmark Suite for Story Visualization](image_generation/vistorybench_comprehensive_benchmark_suite_for_story_visualization.md)**

:   ViStoryBench 构建了一个包含 80 个多风格故事、344 个角色、1317 个镜头的综合基准，提出 12 项自动化评估指标（涵盖角色一致性、风格相似度、提示对齐、copy-paste 检测等），系统评估了超过 25 种开源/商业故事可视化方法，填补了该领域缺乏统一评估标准的空白。

**[VOSR: A Vision-Only Generative Model for Image Super-Resolution](image_generation/vosr_a_vision_only_generative_model_for_image_super_resolution.md)**

:   提出 VOSR，首个证明纯视觉训练的生成式超分模型可以媲美甚至超越基于 T2I 预训练方法的工作，通过视觉语义条件和面向恢复的引导策略实现高质量 SR，训练成本仅为 T2I 方法的 1/10。

**[WaDi: Weight Direction-aware Distillation for One-step Image Synthesis](image_generation/wadi_weight_direction-aware_distillation_for_one-step_image_synthesis.md)**

:   通过分析蒸馏过程中权重变化的范数-方向分解，发现方向变化是蒸馏的关键驱动因素（变化幅度比范数大 22×），提出 LoRaD（低秩权重方向旋转）适配器，集成到 VSD 框架中构成 WaDi，仅用 ~10% 可训练参数即在 COCO 上取得一步生成 SOTA FID。

**[When Identities Collapse: A Stress-Test Benchmark for Multi-Subject Personalization](image_generation/when_identities_collapse_a_stress-test_benchmark_for_multi-subject_personalizati.md)**

:   本文揭示多主体个性化生成中的"身份坍塌"瓶颈——MOSAIC、XVerse、PSR 三个 SOTA 模型在 2 主体时 SCR 已达 ~50%，10 主体时飙升至 ~97%；提出基于 DINOv2 的 Subject Collapse Rate (SCR) 指标替代失效的 CLIP-I，并构建了覆盖 2-10 主体×3 种场景类型的系统化 benchmark。

**[When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance](image_generation/when_safety_collides_resolving_multi-category_harmful_conflicts_in_text-to-image.md)**

:   提出 Conflict-aware Adaptive Safety Guidance (CASG)，一种无训练的即插即用框架，通过动态识别与当前生成状态最对齐的有害类别并仅沿该方向施加安全引导，解决了现有安全引导方法在多类别聚合时因方向冲突导致的安全性退化问题。

**[When Understanding Becomes a Risk: Authenticity and Safety Risks in the Emerging Image Generation Paradigm](image_generation/when_understanding_becomes_a_risk_authenticity_and_safety_risks_in_the_emerging_.md)**

:   系统性对比分析了 MLLM（多模态大语言模型）与扩散模型在安全风险上的差异，发现 MLLM 因更强的语义理解能力而更容易生成不安全图像（抽象/非英语提示也能理解），且其生成的图像更难被现有假图检测器识别，即便针对性微调检测器也可通过丰富提示细节来规避。

**[WISER: Wider Search, Deeper Thinking, and Adaptive Fusion for Training-Free Zero-Shot Composed Image Retrieval](image_generation/wiser_wider_search_deeper_thinking_and_adaptive_fusion_for_training-free_zero-sh.md)**

:   提出 WISER，一个无训练的零样本组合图像检索（ZS-CIR）框架，通过"检索–验证–精化"迭代循环统一 T2I 和 I2I 双路径检索，利用 VLM 验证器显式建模意图感知和不确定性感知，实现自适应融合与结构化自反思精化。在 CIRCO mAP@5 上相对提升 45%，CIRR Recall@1 上相对提升 57%，甚至超越许多训练式方法。

**[YOEO: You Only Erase Once - Erasing Anything without Bringing Unexpected Content](image_generation/yoeo_you_only_erase_once_erasing_anything_without_bringing_unexpected_content.md)**

:   YOEO 提出一个单次擦除框架，通过将多步扩散模型蒸馏为少步模型实现高效推理，并设计杂物抑制损失（基于实体分割检测新生成的不应出现的物体）和实体特征一致性损失（确保擦除区域与周围语义一致），解决扩散模型在目标擦除中的幻觉问题。

---

## 🏥 医学图像 { #medical_imaging }

**[A protocol for evaluating robustness to H&E staining variation in computational pathology models](medical_imaging/a_protocol_for_evaluating_robustness_to_he_stainin.md)**

:   提出三步评估协议（选参考染色条件→表征测试集染色属性→模拟染色条件推理），系统量化306个MSI分类模型对H&E染色差异的鲁棒性，发现鲁棒性与分类性能呈弱负相关(r=-0.28)，高性能不代表高鲁棒性。

**[A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](medical_imaging/a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)**

:   提出面向乳腺超声（BUS）图像分割的半监督框架，利用 GPT-5 生成外观描述 + Grounding DINO + SAM 免训练生成伪标签（APPG），结合双教师框架（静态+动态）通过不确定性-熵加权融合（UEWF）和自适应不确定性引导反向对比学习（AURCL）精炼标签，仅用 2.5% 标注即接近全监督性能。

**[Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](medical_imaging/accelerating_stroke_mri_with_diffusion_probabilist.md)**

:   提出一种受基础模型范式启发的训练策略，先在大规模多对比度脑部 MRI 数据上预训练扩散概率模型（DPM），再用仅 20 例目标域数据微调，实现数据受限场景下与大数据集训练可比的 MRI 加速重建质量，临床盲评显示从 2× 加速数据重建的图像与标准诊疗不相上下。

**[Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning](medical_imaging/act_like_a_pathologist_tissue-aware_whole_slide_image_reasoning.md)**

:   提出 HistoSelect 框架，模拟病理学家从粗到细的推理过程，通过组织分割→Group Sampler→Patch Selector 的三级筛选机制，基于信息瓶颈(IB)理论压缩无关视觉token，在减少约70%计算量的同时实现三个数据集上的SOTA。

**[Active Inference for Micro-Gesture Recognition: EFE-Guided Temporal Sampling and Adaptive Learning](medical_imaging/active_inference_for_micro-gesture_recognition_efe-guided_temporal_sampling_and_.md)**

:   提出 UAAI 框架，首次将主动推理(Active Inference)引入微手势识别，通过 EFE 引导的时间帧选择 + 空间注意力 + UMIX不确定性感知增强，在SMG数据集RGB模态上达到63.47%，大幅超越传统RGB方法。

**[Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](medical_imaging/adaptation_of_weakly_supervised_localization_in_histopathology_by_debiasing_pred.md)**

:   提出 SFDA-DeP，受机器遗忘启发，将 SFDA 重新定义为"识别并纠正预测偏差"的迭代过程：对 dominant class 中高熵不确定样本执行"遗忘"操作迫使模型放弃偏向性预测，对可靠样本保持自训练，同时用像素级分类器锚定定位能力，在跨器官/跨中心病理基准上持续优于现有 SFDA 方法。

**[Adaptive Confidence Regularization for Multimodal Failure Detection](medical_imaging/adaptive_confidence_regularization_for_multimodal_failure_detection.md)**

:   提出 ACR 框架，通过自适应置信度损失（惩罚多模态融合置信度低于单模态的"置信度退化"现象）和多模态特征交换（在特征空间合成失败样本）两个互补模块，首次系统解决多模态场景下的误分类检测问题，在四个数据集上全面超越已有方法。

**[Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](medical_imaging/addressing_data_scarcity_in_3d_trauma_detection_through_self-supervised_and_semi.md)**

:   提出两阶段标签高效框架：先用 patch-based MIM 在1,206个无标注CT上自监督预训练3D U-Net编码器，再用VDETR+3D顶点相对位置编码做3D损伤检测，配合Mean Teacher半监督一致性正则化利用2,000个无标注体数据，仅用144个有标注样本即实现56.57% val mAP@0.50（比纯监督提升115%）。

**[From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](medical_imaging/apex_adaptive_visual_prompting.md)**

:   提出 APEX（Adaptive Prompt EXtraction），通过从可学习 prompt 记忆中自适应检索输入特定的 visual prompt（而非为每个域固定一个 prompt），结合低频特征对比学习增强域间区分能力，显著提升医学图像分割在已见域和未见域上的泛化性能。

**[Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study](medical_imaging/are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)**

:   通过统一训练协议在三个异质医学数据集上对比 11 种模型，发现通用视觉模型（GP-VMs）在标准化条件下系统性超越大多数专用医学分割架构（SMAs），挑战了"医学分割必须使用专用架构"的传统认知。

**[Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](medical_imaging/association_of_radiologic_ppfe_change_with_mortali.md)**

:   在两个独立大规模肺癌筛查队列中，利用深度学习自动分割量化PPFE纵向变化，首次验证其在筛查人群中的独立预后价值。

**[Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](medical_imaging/automated_detection_of_malignant_lesions_in_the_ov.md)**

:   系统对比 15 种 CNN 变体（LeNet/ResNet/VGG/Inception）在卵巢癌组织病理图像五分类上的表现，最终选出 InceptionV3-A（ReLU）达 94% 综合指标，并用 LIME/SHAP/Integrated Gradients 三种 XAI 方法做对比解释分析。

**[Benchmarking Endoscopic Surgical Image Restoration and Beyond](medical_imaging/benchmarking_endoscopic_surgical_image_restoration_and_beyond.md)**

:   构建了首个多源真实世界内窥镜手术图像复原数据集 SurgClean（3,113张图像，覆盖去烟/去雾/去飞溅三种退化类型），在其上系统评测了22种代表性图像复原方法（12种通用+10种任务特定），揭示现有方法与临床需求间仍存在显著差距，并进一步分析了手术场景退化与自然场景退化的本质差异。

**[Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance](medical_imaging/better_than_average_spatially-aware_aggregation_of_segmentation_uncertainty_impr.md)**

:   首次系统研究分割任务中像素级不确定性到图像级分数的聚合策略，提出融合空间结构信息的聚合方法（基于Moran's I、Edge Density、Shannon Entropy的空间质量比SMR），以及GMM元聚合器，在10个数据集的OoD和故障检测任务上验证了空间感知聚合显著优于全局平均。

**[Beyond Pixel Simulation: Pathology Image Generation via Diagnostic Semantic Tokens and Prototype Control](medical_imaging/beyond_pixel_simulation_pathology_image_generation_via_diagnostic_semantic_token.md)**

:   UniPath提出语义驱动的病理图像生成框架，通过多流控制（原始文本 + 从冻结病理MLLM蒸馏的诊断语义Token + 原型库形态控制）实现诊断级可控生成，Patho-FID达80.9，比第二名优51%。

**[BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](medical_imaging/biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)**

:   提出 BiCLIP 框架，通过双向多模态融合（BMF）实现视觉信息反向精炼文本表示，并通过图像增强一致性（IAC）约束中间特征的扰动不变性，在 COVID-19 CT 分割上超越 SOTA，仅 1% 标注数据仍保持鲁棒。

**[Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](medical_imaging/bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)**

:   提出AnoPLe——一个轻量级多模态双向提示学习框架，无需手工异常描述或外部辅助模块，通过文本-视觉提示双向交互和尺度感知前缀实现少样本多类别异常检测，在MVTec-AD/VisA/Real-IAD上取得强竞争力的同时保持高效推理（~28 FPS）。

**[Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD](medical_imaging/bridging_the_skill_gap_in_clinical_cbct_interpreta.md)**

:   构建覆盖55种口腔疾病的7,408例大规模CBCT-报告配对数据集，开发双语口腔颌面CBCT报告生成系统CBCTRepD，通过AI生成草稿+放射科医生编辑的协作模式，在多层级临床评估中证明其可帮助初级医生达到中级水平、中级医生接近高级水平、高级医生减少遗漏。

**[CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis](medical_imaging/care_a_molecular-guided_foundation_model_with_adaptive_region_modeling_for_whole.md)**

:   提出 CARE，一种病理学 slide-level 基础模型，通过自适应区域生成器（ARG）将 WSI 划分为形态学相关的不规则区域（类似 NLP 中的词级 token），并结合 RNA/蛋白质表达谱的跨模态对齐进行两阶段预训练，仅用主流模型约 1/10 的数据即在 33 个下游任务上取得最优平均性能。

**[Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](medical_imaging/cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)**

:   提出 CPNN，利用公开单细胞 RNA-seq 数据构建细胞类型原型（cell-type prototype），将 slide/patch 级基因表达建模为原型的加权组合，在基因表达估计任务上取得 SOTA 并提供可解释性。

**[CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](medical_imaging/chips_efficient_clip_adaptation_via_curvature-aware_hybrid_influence-based_data_.md)**

:   提出 CHIPS，一种基于曲率感知混合影响力的数据选择方法，在 CLIP 端点子空间中计算 Newton 风格对齐分数并结合可学习性与领域相关性权重，仅用 30% 数据即可匹配全量数据集持续预训练效果，在 17 个医学基准上达到 SOTA。

**[CLoE: Expert Consistency Learning for Missing Modality Segmentation](medical_imaging/cloe_expert_consistency_learning_for_missing_modality_segmentation.md)**

:   提出 CLoE（Consistency Learning of Experts），将缺失模态鲁棒性问题建模为决策层面的专家一致性控制，通过模态专家一致性（MEC）和区域专家一致性（REC）双分支约束减少专家漂移，并用一致性分数驱动的门控网络实现可靠性加权融合。

**[CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration](medical_imaging/crft_consistent-recurrent_feature_flow_transformer_for_cross-modal_image_registr.md)**

:   提出CRFT，统一的粗到精跨模态图像配准框架——在Transformer架构中学习模态无关的特征流表示，粗阶段1/8分辨率全局对应+精阶段1/2-1/4多尺度局部细化，配合迭代差异引导注意力和空间几何变换(SGT)递归精化流场捕捉微妙空间不一致，在光学/红外/SAR/多光谱等多种跨模态数据集上超越RAFT/GMFlow/LoFTR等SOTA。

**[Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](medical_imaging/cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)**

:   提出 SpaHGC，一种基于多模态异构图的框架，通过构建目标切片内、跨切片和参考切片内三种子图，结合 masked graph 对比学习和跨节点双注意力机制，实现从 H&E 病理图像预测空间基因表达，在七个数据集上 PCC 指标提升 7.3%-27.1%。

**[cryoSENSE: Compressive Sensing Enables High-throughput Microscopy with Sparse and Generative Priors on the Protein Cryo-EM Image Manifold](medical_imaging/cryosense_compressive_sensing_enables_high-throughput_microscopy_with_sparse_and.md)**

:   提出 cryoSENSE，首个冷冻电镜压缩成像的计算框架，证明蛋白质 cryo-EM 图像在稀疏先验（DCT/小波/TV）和生成先验（扩散模型）下均可从欠采样测量中高保真重建，在保持 3D 分辨率的同时实现最高 2.5× 通量提升。

**[CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](medical_imaging/cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)**

:   提出 CURE——一种基于误差感知课程学习的多任务训练框架，在不引入额外数据的前提下，通过动态调节采样分布重点训练困难样本，将医学 VLM 的视觉定位精度提升 +0.37 IoU，幻觉率降低 18.6%。

**[Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](medical_imaging/decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)**

:   提出 Deco-Mamba，一种以解码器为中心的 Transformer-CNN-Mamba 混合架构，通过 Co-Attention Gate、视觉状态空间模块（VSSM）和可变形卷积增强解码过程，同时引入基于窗口化 KL 散度的分布感知深度监督策略，在 7 个医学图像分割基准上取得 SOTA。

**[Decoupling Vision and Language: Codebook Anchored Visual Adaptation](medical_imaging/decoupling_vision_and_language_codebook_anchored_visual_adaptation.md)**

:   提出 CRAFT，通过离散 codebook 将视觉编码器与语言模型解耦，仅微调视觉编码器即可实现领域适配，且适配后的编码器可跨 LLM 架构无缝复用，在 10 个领域基准上平均提升 13.51%。

**[Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](medical_imaging/deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)**

:   提出ScleraGluNet，通过5个注视方向的巩膜血管照片，用并行CNN提取方向特异性血管特征，再经MRFO特征筛选和Transformer跨视角融合，同时完成三类代谢状态分类（93.8%准确率）和空腹血糖连续估计（MAE=6.42 mg/dL, r=0.983）。

**[Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](medical_imaging/deep_learningbased_assessment_of_the_relation_betw.md)**

:   在按8个独立标注者划分的全景口腔X光裁剪片上，系统对比本地学习（LL）、联邦学习（FL）和集中学习（CL）三种训练范式在第三磨牙-下颌管重叠二分类任务上的表现，验证了CL > FL > LL的性能排序（AUC分别为0.831、0.757和0.672），证明FL在保护数据隐私的前提下显著优于各站点独立训练。

**[Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](medical_imaging/developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)**

:   构建迄今最大的全身 PET 分割数据集 PETWB-Seg11K（11,041 例 3D PET + 59,831 masks），并提出 SegAnyPET——首个面向功能性 PET 影像的 3D 可提示分割基础模型，在多中心、多示踪剂、多疾病场景下实现了强零样本泛化能力。

**[Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](medical_imaging/diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)**

:   本文提出 NNMF+CNN+扩散防御框架用于脑肿瘤 MRI 分类：先用 NNMF 将图像分解为紧凑可解释的低秩特征，通过 AUC/Cohen's d/p-value 统计指标筛选最强判别组件，再用轻量 CNN 分类；推理时引入前向扩散加噪 + 学习去噪器的特征空间净化模块，在 AutoAttack ($L_\infty$, $\epsilon=0.10$) 下将鲁棒准确率从 0.47% 提升至 59.53%。

**[EchoAgent: Towards Reliable Echocardiography Interpretation with "Eyes", "Hands" and "Minds"](medical_imaging/echoagent_towards_reliable_echocardiography_interpretation_with_eyeshands_and_mi.md)**

:   提出 EchoAgent，一个模拟心脏超声医师"眼-手-脑"协同工作流程的 Agent 系统，通过专业知识引擎（mind）、分层工具箱（eyes+hands）和编排推理中枢（reasoning hub）三阶段实现端到端超声心动图可靠解读，在多个基准上达到 SOTA。

**[Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models](medical_imaging/eda_arbitrary_noise_diffusion_design_space.md)**

:   提出 EDA 框架，将 EDM 的设计空间从纯高斯噪声扩展至任意噪声模式，通过多元高斯分布和多独立维纳过程驱动的 SDE 实现灵活噪声扩散，且证明噪声复杂度的提升不引入额外采样开销；仅用 5 步采样即可在 MRI 偏置场矫正、CT 金属伪影去除和自然图像阴影去除三项任务上取得媲美或优于百步 Refusion 和专用方法的效果。

**[EI: Early Intervention for Multimodal Imaging based Disease Recognition](medical_imaging/ei_early_intervention_for_multimodal_imaging_based_disease_recognition.md)**

:   EI 提出在单模态嵌入（UIE）**之前**就注入跨模态语义引导（[INT] token），模拟临床医生"先看一个模态形成初步判断再指导另一个模态检查"的工作流程，同时设计 MoR（多种秩 LoRA + 带旁路的松弛路由器）实现参数高效的 VFM 医学域适配，在视网膜/皮肤/膝关节三个数据集上以 <9M 可训练参数超越所有全参微调和 prompt learning 基线。

**[EMAD: Evidence-Centric Grounded Multimodal Diagnosis for Alzheimer's Disease](medical_imaging/emad_evidence-centric_grounded_multimodal_diagnosis_for_alzheimers_disease.md)**

:   提出 EMAD，一个端到端多模态视觉-语言框架，为 AD 诊断生成结构化报告，通过分层 Sentence–Evidence–Anatomy (SEA) Grounding 将每个诊断声明显式关联到临床证据和 3D 脑部解剖，并用可执行规则驱动的 GRPO 强化微调确保临床一致性。

**[EquivAnIA: A Spectral Method for Rotation-Equivariant Anisotropic Image Analysis](medical_imaging/equivania_a_spectral_method_for_rotation-equivariant_anisotropic_image_analysis.md)**

:   提出EquivAnIA，用定向滤波器族（cake wavelets和ridge filters）在频域中做带权平均来估计图像的角度分布，替代传统angular binning方法，实现对数值旋转真正鲁棒的各向异性分析，合成图像主方向估计误差仅0.03°，CT配准误差仅0.02°。

**[Event-Level Detection of Surgical Instrument Handovers in Videos](medical_imaging/event_level_detection_of_surgical_instrument_handovers_in_videos.md)**

:   提出面向真实手术视频中器械交接检测的时空视觉框架，结合 ViT 空间特征提取和单向 LSTM 时序建模，通过多任务学习联合预测交接事件和方向，在肾移植手术视频上达到 F1=0.84 的检测性能。

**[Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning](medical_imaging/every_error_has_its_magnitude_asymmetric_mistake_severity_training_for_multiclas.md)**

:   提出 PAMS（Priority-Aware Mistake Severity）方法，通过非对称严重性感知的交叉熵损失（MSCE）、语义特征混合（SFR）和非对称 Mikel's Wheel 指标，在多分类 MIL WSI 诊断中显著降低严重误诊风险。

**[Extending ZACH-ViT to Robust Medical Imaging: Corruption and Adversarial Stress Testing in Low-Data Regimes](medical_imaging/extending_zach-vit_to_robust_medical_imaging_corruption_and_adversarial_stress_t.md)**

:   在低数据医学影像场景下，对置换不变的紧凑型 ViT 架构 ZACH-ViT 进行首次鲁棒性扩展评估。在 7 个 MedMNIST 数据集上，ZACH-ViT 在干净数据和常见损坏下均排名第一（Mean Rank 1.57），在 FGSM 下排名最佳（2.00），PGD 下排名第二（2.29）。

**[Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](medical_imaging/fair_lung_disease_diagnosis_from_chest_ct_via_gend.md)**

:   在 ConvNeXt-Base 骨干上构建注意力 MIL 模型，用 GRL 对抗性消除扫描表示中的性别信息，配合 focal loss（$\gamma=2$）+ 标签平滑（$\varepsilon=0.1$）、子群过采样和 5-fold 集成，在 889 例胸部 CT 四类诊断中实现均值竞赛分数 0.685±0.030，女性 macro-F1（0.691）略高于男性（0.679），验证了 GRL 能有效闭合公平性差距。

**[Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](medical_imaging/federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)**

:   提出 FedMEPD 框架，用模态专属编码器处理模态间异质性、滤波器级动态部分个性化解码器平衡知识共享与个性化、多锚点跨注意力校准补偿缺失模态信息，在 BraTS 2018/2020 上全面超越现有多模态联邦学习方法。

**[FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning](medical_imaging/fedvg_gradient-guided_aggregation_for_enhanced_federated_learning.md)**

:   FedVG 提出利用全局验证集上的逐层梯度范数为各客户端打分，梯度越平坦（范数越小）的客户端获得越高聚合权重，从而在高度数据异质性场景下显著提升联邦学习的泛化性能。

**[Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis](medical_imaging/focus-to-perceive_representation_learning_a_cognition-inspired_hierarchical_fram.md)**

:   提出 FPRL，一个受临床认知启发的层次化自监督框架，通过先"聚焦"帧内病灶关键静态语义、再"感知"帧间上下文演化来缓解运动偏差，在 11 个内窥镜数据集上取得 SOTA。

**[Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning](medical_imaging/forecasting_epileptic_seizures_from_contactless_ca.md)**

:   首次系统定义基于纯视频的癫痫发作预测（forecasting）任务（用 3-10 秒发作前片段预测未来 5 秒内是否发作），提出两阶段跨物种迁移学习框架——在啮齿类+人类混合视频上自监督预训练 VideoMAE，再在极少人类癫痫视频上做少样本微调——在 2/3/4-shot 设定下平均 bacc 达 72.30%、roc_auc 达 75.58%，超越所有视频理解 baseline。

**[Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](medical_imaging/forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)**

:   提出 FORGE，首个专为跨站点 fMRI 脑疾病诊断设计的持续学习框架，通过结构感知 VAE 生成逼真的功能连接矩阵进行隐私保护式生成回放，结合双层知识蒸馏和层次化上下文赌博机采样策略，有效缓解灾难性遗忘。

**[GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction](medical_imaging/gaussianpile_a_unified_sparse_gaussian_splatting_framework_for_slice-based_volum.md)**

:   提出 GaussianPile，通过引入焦点感知的物理成像模型（Focus Gaussian），将 3D 高斯溅射从表面外观建模扩展到切片体数据重建，在超声和光片显微镜数据上实现了比 NeRF 方法快 11 倍、比体素网格储存缩小 16 倍的高质量体数据压缩与重建。

**[GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](medical_imaging/giim_graphbased_learning_of_inter_and_intraview_de.md)**

:   提出 GIIM 框架，基于多异构图（MHG）通过四类边关系同时建模同一病灶跨期相动态变化和不同病灶间空间关联，并设计四种缺失视图填充策略，在肝脏 CT、乳腺 X 光和乳腺 MRI 三种模态上均显著优于现有方法。

**[GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](medical_imaging/gleam_a_multimodal_imaging_dataset_and_hamm_for_gl.md)**

:   提出首个公开三模态青光眼数据集 GLEAM（SLO 眼底图 + 环乳头 OCT + 视野偏差图，1200例，四阶段标注），以及基于 CNN 的层级注意力掩码建模框架 HAMM，通过临床启发式的多头模态门控和关系图注意力实现跨模态融合，四分类准确率达 81.08%。

**[Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](medical_imaging/human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)**

:   提出 GenEval，通过域共形界（DCB）量化因果覆盖差距，并将人类专家知识量化精炼后与医学 VLM（MedGemma-4B）融合，以 LoRA 微调实现单源域泛化，在 DR 分级和癫痫灶检测上显著超越基线。

**[Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](medical_imaging/instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)**

:   提出指令引导的胸部X光病变分割任务（ILS），构建了首个大规模自动生成的指令-回答数据集MIMIC-ILS（1.1M样本、192K图像、91K mask），并训练ROSALIA模型实现gIoU 71.2%和空目标准确率91.8%，远超现有通用和医学分割模型。

**[Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](medical_imaging/interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)**

:   发现并解决了 CLIP 在跨域少样本学习（CDFSL）中的局部特征对齐退化问题，提出基于循环一致性的 CC-CDFSL 框架，通过 T-I-T 和 I-T-I 双向循环路径和语义锚点机制改善 patch 级视觉-语言对齐，同时增强模型的可解释性。

**[InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](medical_imaging/invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)**

:   提出 InvAD，将扩散模型异常检测从"RGB 空间去噪重建"范式转变为"潜空间加噪反演"范式，通过 DDIM 反演直接推断最终潜变量并在先验分布下度量偏差来检测异常，仅需 3 步反演即达 SOTA 性能且推理速度提升约 2 倍。

**[Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](medical_imaging/learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)**

:   提出 MASS（MAsk-guided Self-Supervised learning），利用 SAM2 自动生成的类别无关 mask 作为伪标注，以 in-context 分割为 pretext task 进行自监督预训练，无需任何人工标注即可学到语义丰富、泛化性强的 3D 医学图像表征，在 few-shot 分割和冻结编码器分类上均取得优异表现。

**[LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](medical_imaging/lemon_a_large_endoscopic_monocular_dataset_and_foundation_model_for_perception_in.md)**

:   构建了包含 4194 个手术视频（938 小时）的大规模内窥镜数据集 LEMON，并提出基于增强知识蒸馏的自监督基础模型 LemonFM，在手术阶段识别、工具检测、动作识别和语义分割四大下游任务上全面超越现有手术基础模型。

**[LUMINA: A Multi-Vendor Mammography Benchmark with Energy Harmonization Protocol](medical_imaging/lumina_a_multi-vendor_mammography_benchmark_with_energy_harmonization_protocol.md)**

:   提出 LUMINA 多厂商乳腺 FFDM 数据集（468 例患者、1824 张图像），附带前景像素直方图匹配的能量协调预处理方法，在诊断/BI-RADS/密度三任务上系统评估了 CNN 与 Transformer 模型。

**[Marker-Based 3D Reconstruction of Aggregates with a Comparative Analysis of 2D and 3D Morphologies](medical_imaging/markerbased_3d_reconstruction_of_aggregates_with_a.md)**

:   提出一种基于标记物（marker）的低成本摄影测量方法，实现骨料（aggregate）颗粒的高质量三维重建，并通过 2D 与 3D 形态学指标的系统对比分析，揭示了仅依赖 2D 图像进行骨料形态评估的显著局限性。

**[MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](medical_imaging/medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)**

:   在冻结CLIP编码器的基础上，通过概率交叉模态注意力（PVL）实现图文双向交互与预测不确定性建模，配合软patch级对比损失，在16个医学分割数据集上兼顾数据效率、域泛化能力和可解释性。

**[MedGEN-Bench: Contextually Entangled Benchmark for Open-Ended Multimodal Medical Generation](medical_imaging/medgen-bench_contextually_entangled_benchmark_for_open-ended_multimodal_medical_.md)**

:   提出 MedGEN-Bench，首个面向开放式多模态医学生成的综合基准，包含 6,422 个专家验证的图文对、6 种成像模态、16 个临床任务，配套三层评估框架，揭示了组合框架优于统一模型的跨模态一致性问题。

**[MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](medical_imaging/medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)**

:   MedGRPO 提出了两项关键创新解决医学视频多数据集强化学习中的训练崩溃问题：跨数据集奖励归一化（用 logistic 函数将不同难度数据集的中位表现映射到相同奖励值）和医学 LLM 评审（通过五个临床维度的比较性评分），基于 Qwen2.5-VL-7B 在 MedVidBench（532K 视频指令对）上超越 GPT-4.1 和 Gemini-2.5-Flash。

**[MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](medical_imaging/medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)**

:   提出 MedKCO，一种知识驱动的认知编排策略用于医学视觉-语言预训练：通过分层课程（label-level 按诊断敏感度排序 + description-level 按样本代表性排序）和自步非对称对比损失，让模型从简单到复杂渐进学习，在三种医学模态的零样本和下游任务上显著超越基线。

**[MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](medical_imaging/milpf_multiple_instance_learning_on_precomputed_fe.md)**

:   将冻结的通用基础编码器（DINOv2 ViT-Giant / MedSigLIP）与仅 ~40k 参数的轻量 MIL 聚合头结合，通过预计算特征 + 双流聚合（全局均值 + 局部 Perceiver 交叉注意力），在 EMBED 等大规模乳腺 X 线分类基准上以 5-7 分钟训练达到 SOTA（AUC 0.916, Spec@Sens=0.9 达 0.762），可训练参数比基线少 35-458 倍。

**[Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](medical_imaging/mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)**

:   揭示了在 VLM 的跨域小样本微调中，增强视觉判别性反而损害跨模态对齐（"判别性陷阱"），提出 SVL + RA 两个即插即用模块来抑制视觉学习捷径并引导跨模态对齐，在 4 个 CDFSL 数据集和 11 个 FSL 数据集上取得 SOTA。

**[Mitigating Object Hallucination in LVLMs via Attention Imbalance Rectification](medical_imaging/mitigating_object_hallucinations_in_lvlms_via_attention_imbalance_rectification.md)**

:   提出注意力失衡（Attention Imbalance）概念来解释 LVLM 中的对象幻觉现象，并设计轻量级解码时干预方法 AIR，通过跨模态注意力重新分配和方差约束投影正则化矫正注意力失衡，在四个 LVLM 上将幻觉率最高降低 35.1%，同时提升通用能力最高达 15.9%。

**[MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](medical_imaging/moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)**

:   提出 MoECLIP，将 Mixture-of-Experts 引入零样本异常检测（ZSAD），通过冻结正交特征分离（FOFS）和等角紧框架（ETF）损失实现 patch 级别的动态专家路由与特化，在14个工业/医学基准上达到 SOTA。

**[Momentum Memory for Knowledge Distillation in Computational Pathology](medical_imaging/momentum_memory_for_knowledge_distillation_in_computational_pathology.md)**

:   提出 MoMKD，用动量更新的类条件记忆库替代传统 batch-local 特征对齐，实现基因组→病理切片的跨模态知识蒸馏，仅用 H&E 切片推理即可获得基因组级预测能力。

**[MozzaVID: Mozzarella Volumetric Image Dataset](medical_imaging/mozzavid_mozzarella_volumetric_image_dataset.md)**

:   本文发布 MozzaVID——一个基于同步辐射 X 射线 CT 的马苏里拉奶酪微结构体积图像分类数据集，包含 591-37,824 个 192³ 体积样本、25 种奶酪/149 个样本的分类目标，弥补了 3D 体积数据集在数量级和任务设计上与 2D 数据集的巨大差距，实验表明 3D 模型显著优于 2D 模型。

**[MRI Contrast Enhancement Kinetics World Model](medical_imaging/mri_contrast_enhancement_kinetics_world_model.md)**

:   首次提出 MRI 造影增强动力学世界模型（MRI CEKWorld），通过时空一致性学习（STCL）在稀疏采样数据上实现从无造影 MRI 到连续高保真造影增强序列的生成，解决了内容失真和时序不连续两大难题。

**[Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](medical_imaging/multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)**

:   提出 RICE-NET，一个多模态 3D ResNet-18 模型，整合纵向 MRI 数据与放疗剂量分布图，用于自动区分胶质母细胞瘤术后放射诱导对比增强（RICE）与肿瘤复发，在独立测试集上达到 F1=0.92。

**[Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation](medical_imaging/multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_.md)**

:   提出**ERBA(Enzyme-Reaction Bridging Adapter)**，将酶动力学参数预测重新建模为**分阶段多模态条件生成问题**——先通过MRCA注入底物信息捕获底物识别特异性，再通过G-MoE整合活性位点3D结构捕获构象适应，配合ESDA分布对齐保持PLM语义先验。

**[MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning](medical_imaging/multimodalpfn_extending_prior-data_fitted_networks_for_multimodal_tabular_learni.md)**

:   提出 MMPFN，首次将预训练表格基础模型 TabPFN 扩展到多模态（表格+图像/文本）场景，通过多头门控 MLP（MGM）和交叉注意力池化器（CAP）解决非表格嵌入过压缩和 token 数量不平衡问题，在医学和通用数据集上超越 SOTA。

**[Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](medical_imaging/multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)**

:   提出 MSG-LDM，在潜在扩散模型中引入多尺度结构-风格解耦机制，通过高频注入、多模态结构特征融合和结构感知损失，实现缺失模态场景下保留解剖结构和精细细节的多模态 MRI 合成。

**[MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification](medical_imaging/muse_harnessing_precise_and_diverse_semantics_for_few-shot_whole_slide_image_cla.md)**

:   提出 MUSE 框架，通过 MoE 驱动的样本级细粒度语义增强（SFSE）和基于 LLM 知识库的随机多视角语义优化（SMMO），在少样本全切片图像分类任务上显著提升泛化能力。

**[MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](medical_imaging/must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)**

:   提出 MUST 框架，通过代数约束将多模态表征显式分解为模态特有和跨模态共享两部分，并用条件潜在扩散模型在模态缺失时生成特有信息，在五个 TCGA 癌症数据集上以 0.742 C-index 达到 SOTA，且在模态缺失场景下仅降约 0.4%-3.5%。

**[MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy](medical_imaging/muvit_multi-resolution_vision_transformers_for_learning_across_scales_in_microsc.md)**

:   提出 MuViT，一种基于世界坐标 RoPE 位置编码的多分辨率 Vision Transformer，能在单一编码器中联合处理同一场景不同物理分辨率的裁剪图，在显微镜图像分割任务上显著优于单分辨率基线。

**[NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization](medical_imaging/neuroseg_meets_dinov3_transferring_2d_self-supervised_visual_priors_to_3d_neuron.md)**

:   NeurINO 提出通过将 DINOv3 预训练的 2D 卷积核膨胀（inflate）为 3D 算子来初始化 3D 神经元分割模型，同时引入拓扑感知骨架损失（TASL）显式监督骨架级结构保真性，在四个神经影像数据集上 ESA 平均提升 2.9%、DSA 提升 2.8%、PDS 提升 3.8%。

**[Novel Architecture of RPA In Oral Cancer Lesion Detection](medical_imaging/novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)**

:   将软件设计模式（Singleton + Batch Processing）集成到基于 EfficientNetV2B1 的口腔癌病变检测 Python 流水线中，相比传统 RPA 平台（UiPath/Automation Anywhere）实现 60-100x 推理加速（每张图 0.06s vs 2.58s），同时保持诊断准确性。

**[OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](medical_imaging/omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)**

:   提出 OmniFM，一个模态鲁棒且任务无关的联邦学习框架，通过频域频谱知识检索、嵌入式交叉注意力融合和前缀-后缀频谱提示三个互补组件，在一个统一的 FL pipeline 下支持分类、分割、超分辨率、VQA 和多模态融合五种医学影像任务，并在跨模态异构场景下显著超越现有基线。

**[OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](medical_imaging/orapo_oracle-educated_reinforcement_learning_for_data-efficient_and_factual_radi.md)**

:   提出 OraPO（Oracle-educated GRPO），在 GRPO 探索失败时注入轻量 DPO 监督将失败 rollout 转化为偏好样本，配合 FactScore 奖励实现仅用 1K 样本、3B 小模型在 CheXpert Plus 和 MIMIC-CXR 上达到放射报告生成 SOTA（F1=0.341/0.357），训练数据量比前最优减少 2-3 个数量级。

**[Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](medical_imaging/parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)**

:   HIPSS 提出了两个关键创新用于少样本 WSI 分类：(1) 基于缩放和偏移特征（SSF）的参数高效 prompt 调优替代 CoOp，大幅减少可训练参数；(2) 软层次化文本引导策略无需硬过滤即可利用 VLM 的预训练知识和 WSI 的固有层次结构。在三个癌症数据集上最高提升 13.8%。

**[PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](medical_imaging/pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)**

:   PGR-Net 提出了一种显式 ROI 感知的脑肿瘤 MRI 分割网络，通过从训练集构建数据驱动的空间先验模板、层级 Top-K ROI 选择机制和窗口高斯-空间衰减引导模块（WinGS-ROI），将计算资源集中于病灶区域，仅用 8.64M 参数就在 BraTS-2019/2023 和 MSD Task01 上达到了 SOTA。

**[Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](medical_imaging/prototypebased_knowledge_guidance_for_finegrained.md)**

:   提出 ProtoSR，通过 LLM 驱动的管道从 22.7 万篇 MIMIC-CXR 自由文本报告中挖掘模板对齐的视觉原型知识库，并设计原型条件化迟融合模块将检索到的原型证据作为 logit 残差注入层级式结构化报告模型，在 Rad-ReStruct 基准上达到 SOTA，L3 细粒度属性 F1 从 4.3 提升到 7.4（+72.1% 相对提升）。

**[RDFace: A Benchmark Dataset for Rare Disease Facial Image Analysis under Extreme Data Scarcity and Phenotype-Aware Synthetic Generation](medical_imaging/rdface_a_benchmark_dataset_for_rare_disease_facial_image_analysis_under_extreme_.md)**

:   构建了包含 456 张儿童面部图像、覆盖 103 种罕见遗传疾病的标准化基准数据集 RDFace，并系统研究了表型感知的合成数据增强（DreamBooth/FastGAN）在超低样本罕见病诊断中的效果，DreamBooth 增强在极端低数据场景下最高可提升 13.7% 的诊断准确率。

**[Reclaiming Lost Text Layers for Source-Free Cross-Domain Few-Shot Learning](medical_imaging/reclaiming_lost_text_layers_for_source-free_cross-domain_few-shot_learning.md)**

:   发现 CLIP 文本编码器中存在"Lost Layers"——在 Source-Free Cross-Domain Few-Shot Learning (SF-CDFSL) 中移除某些中间层反而提升性能；论文证明这些层并非冗余而是因视觉域偏移未被充分利用，提出 VtT 模型在层级和编码器级别重新利用这些信息，取得 SOTA。

**[Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](medical_imaging/reinforcing_the_weakest_links_modernizing_siena_wi.md)**

:   将 SIENA 纵向脑萎缩管线中的经典颅骨剥离(BET2)和组织分割(FAST)模块定向替换为深度学习方案(SynthStrip/SynthSeg)，在 ADNI (N=1006) 和 PPMI (N=310) 两个大规模纵向队列上显著增强了 PBVC 与临床疾病进展的关联性(相关系数提升超 100%)，扫描顺序误差降低高达 99.1%。

**[RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference](medical_imaging/relativeflow_taming_medical_image_denoising_learning_with_noisy_reference.md)**

:   提出 RelativeFlow，基于 flow matching 的框架，通过将绝对噪声到干净映射分解为相对更噪到噪声映射，结合一致传输约束和基于模拟的速度场，从异质噪声参考中学习统一的去噪流，突破参考偏差限制。

**[Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](medical_imaging/residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)**

:   提出 Residual SODAP 框架，通过 α-entmax 稀疏提示选择+残差聚合、无数据统计蒸馏+伪特征回放、提示使用模式漂移检测，以及不确定性加权多损失平衡，联合解决提示端表征适应和分类器端知识保持问题，在医学域增量学习上达到 SOTA。

**[Robust Fair Disease Diagnosis in CT Images](medical_imaging/robust_fair_disease_diagnosis_in_ct_images.md)**

:   本文提出结合Logit调整交叉熵（处理类别不平衡）和CVaR聚合（处理人口统计公平性）的双层目标函数，在CT诊断中实现了性别平均macro F1达0.8403且公平性差距仅0.0239。

**[Robust Multi-Source Covid-19 Detection in CT Images](medical_imaging/robust_multi-source_covid-19_detection_in_ct_images.md)**

:   提出一种多任务学习框架，在共享 EfficientNet-B7 骨干上同时训练 COVID-19 诊断头和来源医院识别头（使用 logit-adjusted 损失），推动特征提取器学习跨机构不变的表示，在多源 CT 数据集上 F1 达到 0.9098。

**[SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](medical_imaging/sd_fsmis_adapting_stable_diffusion_for_few_shot_medical_image_segmentation.md)**

:   提出 SD-FSMIS，一个将预训练 Stable Diffusion 适配到少样本医学图像分割的框架，通过支持-查询交互模块和视觉到文本条件转换器实现高效适配，在跨域场景中表现尤为突出。

**[Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](medical_imaging/semantic_class_distribution_learning_for_debiasing.md)**

:   本文提出 SCDL（Semantic Class Distribution Learning），一个即插即用模块，通过类别分布双向对齐（CDBA）学习结构化的类条件特征分布并与可学习类代理双向对齐，结合语义锚点约束（SAC）利用标注数据引导代理学习正确语义，缓解了半监督医学图像分割中的监督偏差和特征表示偏差，在尾类器官上取得了显著提升。

**[SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](medical_imaging/semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)**

:   提出 SemiTooth 框架，通过多教师多学生架构和严格加权置信度约束（SWC），解决多源 CBCT 牙齿分割中的标注稀缺和跨源域间差异问题，同时构建了首个多源半监督牙齿数据集 MS3Toothset。

**[Solving a Nonlinear Blind Inverse Problem for Tagged MRI with Physics and Deep Generative Priors](medical_imaging/solving_a_nonlinear_blind_inverse_problem_for_tagged_mri_with_physics_and_deep_g.md)**

:   提出 InvTag 框架，首次将 MR 物理前向模型与预训练扩散生成先验结合，统一解决 3D Tagged MRI 的解剖恢复、Cine 合成和运动估计三大子任务，且无需任何额外训练数据。

**[STEPH: Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in WSI Prognosis](medical_imaging/sparse_task_vector_mixup_wsi_prognosis.md)**

:   STEPH 提出基于任务向量混合（TVM）+ 超网络驱动稀疏聚合的模型合并方案，将多个癌种特定预后模型的知识高效融入目标癌种模型，在 13 个 TCGA 数据集上 C-Index 平均 0.6949（+5.14% vs 癌种特定学习、+2.01% vs ROUPKT），且推理仅需单模型前向传播，远低于多模型表示迁移方案。

**[SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering for Medical Image Segmentation](medical_imaging/spegc_continual_test-time_adaptation_via_semantic-prompt-enhanced_graph_clusteri.md)**

:   提出 SPEGC 框架，通过语义提示增强特征 + 可微分图聚类求解器，将原始相似度矩阵精炼为高阶结构表示，用于指导医学图像分割模型在持续变化的目标域上自适应，有效缓解误差累积与灾难性遗忘。

**[SVC 2026: The Second Multimodal Deception Detection Challenge and the First Domain Generalized Remote Physiological Measurement Challenge](medical_imaging/svc_2026_the_second_multimodal_deception_detection_challenge_and_the_first_domai.md)**

:   组织SVC 2026挑战赛，包含跨域多模态欺骗检测和域泛化远程生理信号测量两个赛道，提供统一评估框架和基线模型，共22支队伍提交最终结果。

**[Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos](medical_imaging/synergistic_bleeding_region_and_point_detection_in_laparoscopic_surgical_videos.md)**

:   构建首个腹腔镜手术出血区域+出血点标注数据集 SurgBlood，并提出基于 SAM2 的双分支双向引导在线检测器 BlooDet，通过 Mask/Point 分支协同优化实现出血区域分割与出血点定位的联合检测。

**[T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation](medical_imaging/t-gated_adapter_a_lightweight_temporal_adapter_for_vision-language_medical_segme.md)**

:   提出轻量级时序门控适配器（T-Gated Adapter），为2D视觉语言模型CLIPSeg注入相邻切片上下文，在仅30个标注CT体积上训练即可实现平均Dice 0.704（+0.206），跨域零样本评估和CT到MRI跨模态评估中均一致提升。

**[Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](medical_imaging/tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)**

:   提出 Tell2Adapt 统一框架，利用视觉基础模型（BiomedParse）的泛化知识，通过上下文感知提示正则化（CAPR）生成高质量伪标签，再经视觉合理性精炼（VPR）去除解剖学不合理预测，实现跨 10 个域迁移方向、22 个解剖目标的统一无源域自适应医学图像分割。

**[The Invisible Gorilla Effect in Out-of-distribution Detection](medical_imaging/the_invisible_gorilla_effect_in_out-of-distribution_detection.md)**

:   揭示了OOD检测中一个此前未被报告的偏差——"隐形大猩猩效应"：当OOD伪影与模型关注区域（ROI）视觉外观相似时检测性能显著更好，不相似时则大幅下降，尤其影响基于特征的OOD方法。

**[Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](medical_imaging/towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)**

:   提出 Difficulty-Influence Quadrant (DIQ) 数据选择策略，联合考量样本难度和梯度影响力，使 VLM 语言骨干仅用 1% 精选数据即可匹配全量 SFT 性能，10% 数据则可超越全量训练。

**[Transformer-Based Multi-Region Segmentation and Radiomic Analysis of HR-pQCT Imaging for Osteoporosis Classification](medical_imaging/transformer-based_multi-region_segmentation_and_radiomic_analysis_of_hr-pqct_ima.md)**

:   提出基于 SegFormer 的全自动多区域 HR-pQCT 分割框架，结合影像组学特征与机器学习实现骨质疏松二分类，发现软组织（肌腱/脂肪）特征的诊断价值优于传统骨骼特征。

**[Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding](medical_imaging/ultrasound-clip_semantic-aware_contrastive_pre-training_for_ultrasound_image-tex.md)**

:   这篇论文的核心贡献不是只做了一个“超声版 CLIP”，而是围绕超声特有的解剖层级和诊断属性重新定义了图文对齐目标：先构建超声知识体系 UDT 和大规模 US-365K 数据集，再用语义软标签与属性异构图把文本里的临床关系显式注入对比学习，从而得到更像“懂超声”的视觉语言表示。

**[Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](medical_imaging/uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)**

:   提出 SMART 框架，基于 SAM3 的概念提示分割构建 Teacher-Student 半监督模型，结合渐进置信度正则化和双流时序一致性策略，仅用极少标注在 X 射线冠脉造影视频中实现 SOTA 血管分割。

**[UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC](medical_imaging/unistainnet_foundation-model-guided_virtual_staining_of_he_to_ihc.md)**

:   提出 UNIStainNet，首次将冻结的病理基础模型 UNI 的密集空间 token 作为 SPADE 调制信号直接注入生成器，配合错位感知损失和可学习染色嵌入，用单一模型同时生成 HER2/Ki67/ER/PR 四种 IHC 染色，在 MIST 和 BCI 基准上取得 SOTA 分布式指标。

**[Unleashing Video Language Models for Fine-grained HRCT Report Generation](medical_imaging/unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)**

:   提出 AbSteering 两阶段框架，利用异常中心的 CoT 推理和 DPO 硬负样本对比学习，将通用 VideoLM 高效适配到 HRCT 报告生成，在临床效能指标上大幅超越专用 CT 基础模型。

**[Unlocking Multi-Site Clinical Data: A Federated Approach to Privacy-First Child Autism Behavior Analysis](medical_imaging/unlocking_multi-site_clinical_data_a_federated_approach_to_privacy-first_child_a.md)**

:   本文提出首个面向儿童自闭症行为识别的联邦学习框架，通过 3D 骨骼抽象化（消除身份信息）+ 联邦优化（数据不出站点）的双层隐私策略，在 MMASD 数据集上用 APFL 个性化联邦方法达到 87.80% 准确率，比本地训练高 5.2%，同时满足 HIPAA/GDPR 隐私合规要求。

**[Unlocking Positive Transfer in Incrementally Learning Surgical Instruments: A Self-reflection Hierarchical Prompt Framework](medical_imaging/unlocking_positive_transfer_in_incrementally_learning_surgical_instruments_a_sel.md)**

:   这篇论文把每个器械类的提示参数从“彼此隔离的独立 prompt”改造成“共享知识逐层拆解的树结构”，让新器械可以继承旧知识快速学会，同时让新知识反过来温和修正旧知识，从而在手术器械类增量分割中同时提升新类、常见类和旧类表现。

**[Unsupervised Domain Adaptation with Target-Only Margin Disparity Discrepancy](medical_imaging/unsupervised_domain_adaptation_with_target-only_margin_disparity_discrepancy.md)**

:   针对 CT→CBCT 肝脏分割的无监督域自适应问题，发现经典 MDD 优化目标中存在矛盾项（源域上特征提取器被优化为最大化 $f$ 和 $f'$ 的差异），提出 Target-Only MDD 改进，去除矛盾项并在两域上统一最小化预测差异，在 2D 和 3D 实验中均取得 UDA SOTA。

**[Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code](medical_imaging/virtual_full-stack_scanning_of_brain_mri_via_imputing_any_quantised_code.md)**

:   提出 CodeBrain，将脑 MRI 任意到任意模态补全问题重新表述为区域级全栈量化码预测任务，通过两阶段流程（标量量化重建 + 分级损失码预测）实现统一的缺失模态合成，超越五种 SOTA 方法。

**[VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](medical_imaging/visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)**

:   重新审视零样本异常检测（ZSAD）中文本分支的必要性，提出 VisualAD——一个纯视觉框架：在冻结 ViT 中插入两个可学习 token（anomaly/normal），配合 Spatial-Aware Cross-Attention 和 Self-Alignment Function，去掉文本编码器仍在 13 个工业+医学基准上取得 SOTA。

**[Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation](medical_imaging/weakly_supervised_teacher-student_framework_with_progressive_pseudo-mask_refinem.md)**

:   提出弱监督教师-学生框架，利用稀疏病理标注和 EMA 稳定的教师网络生成渐进式精炼伪掩码，结合置信度过滤、自适应融合和课程引导精炼策略，实现结直肠癌病理图像中腺体结构的高效分割。

**[X-WIN: Building Chest Radiograph World Model via Predictive Sensing](medical_imaging/x-win_building_chest_radiograph_world_model_via_predictive_sensing.md)**

:   提出 X-WIN 胸片世界模型，首次将 3D CT 空间知识融入 CXR 表征学习：通过学习预测 CT 在不同旋转角度下的 2D 投影来内化 3D 解剖结构，配合亲和力引导的对比对齐和结构保持域自适应，在 6 个 CXR 基准上通过线性探测取得 SOTA。

**[XSeg: A Large-scale X-ray Contraband Segmentation Benchmark for Real-World Security Screening](medical_imaging/xseg_a_large-scale_x-ray_contraband_segmentation_benchmark_for_real-world_securi.md)**

:   本文构建了目前最大的 X 光违禁品分割数据集 XSeg（98,644 张图像、295,932 个实例 mask、30 个细粒度类别），并提出域特化模型 APSAM，通过 Energy-Aware Encoder 利用 X 光双能量物理特性 + Adaptive Point Generator 智能扩展用户点击提示，mIoU 达 72.83%，比 SAM 微调高 4.96%。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[A Prediction-as-Perception Framework for 3D Object Detection](autonomous_driving/a_prediction-as-perception_framework_for_3d_object_detection.md)**

:   受人脑"预测性感知"机制启发，提出 PAP 框架——将历史帧的轨迹预测结果作为 query 注入当前帧的感知模块，在 UniAD 上实现跟踪精度提升 10%、推理速度提升 15%。

**[AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception](autonomous_driving/adaradar_rate_adaptive_spectral_compression_for_radar-based_perception.md)**

:   提出 AdaRadar——基于 DCT 频谱剪枝与零阶代理梯度的在线自适应雷达数据压缩框架，在 100× 以上压缩率下仅损失 ~1%p 检测/分割性能，有效缓解雷达传感器到计算端的带宽瓶颈。

**[An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](autonomous_driving/an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)**

:   提出ADMesh（15K+高质量3D模型库）和CarlaOcc（10万帧、0.05m精度的全景占据数据集），首次为自动驾驶3D全景占据预测提供实例级标注和物理一致的地面真值，并引入占据质量评估指标和系统基准测试。

**[BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird's-Eye View Images](autonomous_driving/bev-sld_self-supervised_scene_landmark_detection_for_global_localization_with_li.md)**

:   提出BEV-SLD，一种基于自监督场景地标检测(Scene Landmark Detection)的LiDAR全局定位方法，将检测与对应关系预测解耦，仅需20MB即可在多种场景下实现高精度(x, y, azimuth)位姿估计。

**[BuildAnyPoint: 3D Building Structured Abstraction from Diverse Point Clouds](autonomous_driving/buildanypoint_3d_building_structured_abstraction_from_diverse_point_clouds.md)**

:   提出BuildAnyPoint，通过**松耦合级联扩散Transformer(Loca-DiT)**实现从多样分布的点云（机载LiDAR、SfM、稀疏噪声点云）到结构化3D建筑Mesh的统一重建——先用分层潜在扩散恢复底层点云分布，再用自回归Transformer生成紧凑多边形Mesh。

**[C2T: LLM-Aligned Common-Sense Reward Learning for Traffic-Vehicle Coordination](autonomous_driving/c2t_llm_traffic_coordination.md)**

:   提出 C2T 框架，通过将交通状态转换为结构化描述（caption），利用 LLM 进行离线偏好判断并蒸馏为内在奖励函数，替代手工设计的交通信号控制奖励，在 CityFlow 基准的多个真实城市网络上提升效率、安全性和能耗指标。

**[CausalVAD: De-confounding End-to-End Autonomous Driving via Causal Intervention](autonomous_driving/causalvad_de-confounding_end-to-end_autonomous_driving_via_causal_intervention.md)**

:   提出 CausalVAD，通过将 Pearl 后门调整理论参数化为即插即用模块（SCIS），在 VAD 架构的感知-预测-规划三个阶段进行多级因果干预，消除虚假关联，实现更安全、更鲁棒的端到端自动驾驶。

**[CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection](autonomous_driving/ccf_complementary_collaborative_fusion_for_domain_generalized_multi-modal_3d_obj.md)**

:   针对双分支多模态3D检测器在域迁移场景下的模态不平衡问题，提出 CCF 框架，通过解耦损失、LiDAR引导深度先验和互补跨模态掩码三个组件系统提升相机查询的利用率和跨域鲁棒性。

**[ClimaOoD: Improving Anomaly Segmentation via Physically Realistic Synthetic Data](autonomous_driving/climaood_improving_anomaly_segmentation_via_physically_realistic_synthetic_data.md)**

:   提出ClimaDrive数据生成框架和ClimaOoD基准数据集，通过语义引导的多天气场景生成+透视感知的异常物体放置，构建10K+训练集覆盖6种天气×93类异常，训练后四个SOTA方法平均AP提升3.25%。

**[CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection](autonomous_driving/coin3d_revisiting_configuration-invariant_multi-camera_3d_object_detection.md)**

:   提出 CoIn3D 框架，通过空间感知特征调制（SFM）和相机感知数据增强（CDA）两个模块，显式建模相机内参/外参/阵列布局的空间先验差异，实现多相机3D检测模型从源配置到未见目标配置的强泛化迁移，适用于 BEVDepth / BEVFormer / PETR 三大主流范式。

**[ColaVLA: Leveraging Cognitive Latent Reasoning for Hierarchical Parallel Trajectory Planning in Autonomous Driving](autonomous_driving/colavla_leveraging_cognitive_latent_reasoning_for_hierarchical_parallel_trajecto.md)**

:   ColaVLA 提出统一的视觉-语言-动作(VLA)框架，将 VLM 的推理从文本链式思考迁移到潜空间，通过认知潜空间推理器(Cognitive Latent Reasoner)和层次化并行规划器(Hierarchical Parallel Planner)，仅需两次 VLM 前向传播即可高效完成场景理解与轨迹解码，在 nuScenes 开环和闭环评测上均达到 SOTA。

**[CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](autonomous_driving/colc_communication-efficient_collaborative_perception_with_lidar_completion.md)**

:   CoLC 提出一种通信高效的早期协同感知框架，通过前景感知点采样(FAPS)减少传输量，结合 VQ-based LiDAR 补全(CEEF)在 ego 端恢复稠密 pillar 表示，并用稠密引导双对齐(DGDA)保证语义和几何一致性，在大幅降低通信带宽的同时保持甚至超越早期融合的检测性能。

**[Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation](autonomous_driving/composing_driving_worlds_through_disentangled_cont.md)**

:   提出 CompoSIA，一个组合式驾驶视频模拟器，将场景结构、物体身份和自车动作三个控制因素通过独立路径解耦注入 Flow Matching DiT，支持独立与组合编辑，实现系统性对抗场景合成，身份编辑 FVD 提升 17%，动作控制旋转/平移误差降低 30%/47%，下游规划器碰撞率平均提升 173%。

**[CycleBEV: Regularizing View Transformation Networks via View Cycle Consistency for Bird's-Eye-View Semantic Segmentation](autonomous_driving/cyclebev_regularizing_view_transformation_networks_via_view_cycle_consistency_fo.md)**

:   提出 CycleBEV 正则化框架：训练时引入逆视角变换（IVT）网络将 BEV 分割图映射回透视图（PV）分割图，通过循环一致性损失及高度感知几何正则化、跨视角隐空间对齐两项新目标来增强现有 BEV 语义分割模型，推理时不增加任何开销。

**[Den-TP: A Density-Balanced Data Curation and Evaluation Framework for Trajectory Prediction](autonomous_driving/den_tp_a_density_balanced_data_curation_and_evaluation_framework_for_trajectory.md)**

:   从数据中心视角出发，提出 Den-TP 框架通过密度感知的数据集筛选和评估协议来解决轨迹预测数据集中场景密度的长尾不平衡问题，仅用 50% 数据就能保持整体性能并显著改善高密度场景的鲁棒性。

**[DLWM: Dual Latent World Models enable Holistic Gaussian-centric Pre-training in Autonomous Driving](autonomous_driving/dlwm_dual_latent_world_models_enable_holistic_gaussian-centric_pre-training_in_a.md)**

:   提出 DLWM，一个两阶段的高斯中心自监督预训练范式：第一阶段通过重建深度和语义图学习3D高斯表示，第二阶段训练双隐世界模型——高斯流引导的时序预测（用于占据感知/预测）和自车规划引导的时序预测（用于运动规划），显著提升三大核心任务性能。

**[Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving](autonomous_driving/drive_my_way_preference_alignment_of_vision-language-action_model_for_personaliz.md)**

:   提出 DMW（Drive My Way），一个个性化 VLA 驾驶框架，通过用户嵌入学习长期驾驶习惯并结合自然语言指令进行短期偏好适配，使用 GRPO 强化微调和风格感知奖励实现个性化驾驶行为生成。

**[DriverGaze360: OmniDirectional Driver Attention with Object-Level Guidance](autonomous_driving/drivergaze360_omnidirectional_driver_attention_with_object-level_guidance.md)**

:   提出首个360°全视角驾驶员注意力数据集（~100万帧/19名驾驶员），并设计DriverGaze360-Net通过辅助语义分割头联合学习注意力图与被关注物体，在全景驾驶图像上达到SOTA注意力预测性能。

**[Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving](autonomous_driving/drocc_depth_region_guided_3d_occupancy.md)**

:   提出 Dr.Occ，一个统一的纯视觉 3D 占用预测框架，通过深度引导的双投影视图变换器（D2-VFormer）利用 MoGe-2 高质量深度先验实现精确几何对齐，以及区域引导的 MoE/MoR 专家 Transformer（R-EFormer / R2-EFormer）自适应分配区域专家解决空间语义不平衡，在 Occ3D-nuScenes 上将 BEVDet4D 基线提升 7.43% mIoU。

**[Efficient Equivariant Transformer for Self-Driving Agent Modeling](autonomous_driving/efficient_equivariant_transformer_for_self-driving_agent_modeling.md)**

:   提出 DriveGATr，一种基于 2D 射影几何代数（Projective Geometric Algebra）的等变 Transformer 架构，无需显式成对相对位置编码即可实现 SE(2)-等变性，在交通模拟任务中达到 SOTA 性能的同时显著降低计算成本。

**[EMDUL: Expanding mmWave Datasets for Human Pose Estimation with Unlabeled Data and LiDAR Datasets](autonomous_driving/expanding_mmwave_datasets_for_human_pose_estimation_with_unlabeled_data_and_lida.md)**

:   提出 EMDUL 管线，通过伪标签标注无标注毫米波数据（含新设计的无监督时序一致性损失 UTCL）和闭式 LiDAR→mmWave 点云转换器（含基于流的点过滤 FPF），大幅扩展毫米波 HPE 数据集的规模与多样性，域内误差降低 15.1%、跨域误差降低 18.9%。

**[F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling](autonomous_driving/f3dgs_federated_3d_gaussian_splatting_for_decentralized_multi-agent_world_modeli.md)**

:   提出F3DGS，首个将联邦学习框架应用于3DGS的方法，通过冻结几何+可见性感知聚合实现多智能体分布式3D重建，无需原始数据共享。

**[Failure Modes for Deep Learning-Based Online Mapping: How to Measure and Address Them](autonomous_driving/failure_modes_for_deep_learning-based_online_mapping_how_to_measure_and_address_.md)**

:   本文系统性地定义和量化了深度学习在线建图模型的两种失败模式——定位过拟合和地图几何过拟合，提出基于 Fréchet 距离的性能度量和基于最小生成树（MST）的训练集稀疏化策略，在 nuScenes 和 Argoverse 2 上验证了几何多样且均衡的训练集能改善模型泛化能力。

**[FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts](autonomous_driving/fedbprompt_federated_domain_generalization_person_re-identification_via_body_dis.md)**

:   提出FedBPrompt框架，通过身体分布感知视觉提示机制(BAPM)将prompt分为Body Part Alignment Prompts和Holistic Full Body Prompts两组，配合Prompt-based Fine-Tuning Strategy(PFTS)冻结ViT backbone仅训练轻量prompt（通信量降至~1%），在FedDG-ReID任务上平均mAP提升3.3%、Rank-1提升4.9%。

**[FlashCap: Millisecond-Accurate Human Motion Capture via Flashing LEDs and Event-Based Vision](autonomous_driving/flashcap_millisecond-accurate_human_motion_capture_via_flashing_leds_and_event-b.md)**

:   提出 FlashCap，首个基于闪烁 LED + 事件相机的运动捕捉系统，通过为每个 LED 配置不同的闪烁频率实现身份识别，构建了首个 1000Hz 标注精度的人体运动数据集 FlashMotion（715 万帧），并提出 ResPose 基线方法，将运动定时误差从 ~50ms 降至 ~5ms，姿态估计 MPJPE 降低约 40%。

**[FoSS: Modeling Long-Range Dependencies and Multimodal Uncertainty in Trajectory Prediction via Fourier–State Space Integration](autonomous_driving/foss_modeling_long_range_dependencies_and_multimodal_uncertainty_in_trajectory_p.md)**

:   FoSS 提出一种频域-时域双分支框架，通过渐进螺旋重排序（HelixSort）将傅里叶频谱有序化后输入选择性状态空间模型（SSM），结合时域动态 SSM 和交叉注意力融合，在 Argoverse 1/2 上取得 SOTA 轨迹预测精度，同时参数量减少 40%+、推理延迟降低 22%。

**[Generalizing Visual Geometry Priors to Sparse Gaussian Occupancy Prediction](autonomous_driving/generalizing_visual_geometry_priors_to_sparse_gaussian_occupancy_prediction.md)**

:   GPOcc 提出利用可泛化的视觉几何先验（如 VGGT、DepthAnything）进行单目 3D 占据预测，通过沿相机射线向内延伸表面点生成体积采样，以稀疏高斯基元进行概率占据推断，并设计免训练增量更新策略处理流式输入，在 Occ-ScanNet 上单目 mIoU 提升 +9.99、流式提升 +11.79 超越前 SOTA，同时在相同深度先验下速度快 2.65 倍。

**[Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal](autonomous_driving/ghost-fwl_a_large-scale_full-waveform_lidar_dataset_for_ghost_detection_and_remo.md)**

:   Ghost-FWL 提出首个大规模移动端全波形 LiDAR 数据集（24K帧、75亿峰值级标注），并设计 FWL-MAE 自监督预训练框架实现鬼影检测与去除，将 SLAM 轨迹误差降低 66% 以上、3D 检测假阳性率减少 50 倍。

**[HG-Lane: High-Fidelity Generation of Lane Scenes under Adverse Weather and Lighting Conditions without Re-annotation](autonomous_driving/hg-lane_high-fidelity_generation_of_lane_scenes_under_adverse_weather_and_lighti.md)**

:   针对车道检测数据集（CULane/TuSimple）极端天气样本严重不足的问题，提出HG-Lane——一个无需重标注的两阶段扩散生成框架：Stage-I通过Control Information Fusion+Structure-aware Reverse Diffusion保留车道几何结构，Stage-II通过Appearance-aware Refinement调整光照风格，生成snow/rain/fog/night/dusk共30K图。CLRNet整体mF1提升+20.87%，snow场景+38.8%。

**[HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles](autonomous_driving/horizonforge_driving_scene_editing_with_any_trajectories_and_any_vehicles.md)**

:   HorizonForge 提出一个统一框架，将驾驶场景重建为可编辑的 Gaussian Splats + Mesh 表示，通过轨迹控制实现精细 3D 操控和语言驱动的车辆插入，再经视频扩散模型渲染生成时空一致的高质量驾驶视频，在用户偏好率上以 91.02% 碾压所有对比方法。

**[IGASA: Integrated Geometry-Aware and Skip-Attention Modules for Enhanced Point Cloud Registration](autonomous_driving/igasa_integrated_geometry-aware_and_skip-attention_modules_for_enhanced_point_cl.md)**

:   提出 IGASA 框架，通过分层金字塔架构 (HPA) + 分层跨层注意力 (HCLA) + 迭代几何感知精修 (IGAR) 三级流水线，弥合多尺度特征的语义鸿沟并动态抑制离群点，在 3D(Lo)Match、KITTI、nuScenes 四大基准上全面超越 SOTA。

**[InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset](autonomous_driving/incarpose_in-cabin_relative_camera_pose_estimation_model_and_dataset.md)**

:   本文提出 InCaRPose，一个基于冻结 ViT 骨干和 Transformer 解码器的车内相对相机位姿估计模型，仅在合成数据上训练即可泛化到真实车内环境，实现绝对度量级翻译预测和实时推理（>45 FPS），同时发布了配套的真实世界高畸变车内测试数据集 In-Cabin-Pose。

**[KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System](autonomous_driving/knowval_a_knowledge-augmented_and_value-guided_autonomous_driving_system.md)**

:   提出KnowVal端到端自驾系统，通过三大核心解决知识推理和价值对齐缺失：(1)Retrieval-guided Open-world Perception融合标准3D检测+VL-SAMv2长尾物体+VLM场景理解；(2)Perception-guided Knowledge Retrieval从驾驶知识图谱（交通法/防御驾驶/道德规范）检索相关知识；(3)World Model预测未来状态+Value Model（human-preference训练）评估轨迹价值，实现可解释决策。nuScenes最低碰撞率，Bench2Drive/NVISIM SOTA。

**[Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens](autonomous_driving/le_mumo_jepa_multi-modal_self-supervised_representation_learning_with_learnable_.md)**

:   将LeJEPA自监督框架扩展到多模态设置，引入可学习融合token作为Perceiver式潜在瓶颈在共享Transformer内高效融合RGB与伴随模态（LiDAR深度/热红外），采用剪枝策略将注意力开销降低约9倍，在Waymo上CenterNet 3D检测mAP XY达23.6（比RGB-only LeJEPA提升4.3），Depth MAE从4.704降至2.860。

**[LEADER: Learning Reliable Local-to-Global Correspondences for LiDAR Relocalization](autonomous_driving/leader_lidar_relocalization.md)**

:   LEADER 通过鲁棒的投影式几何编码器（偏航不变）和截断相对可靠性损失（抑制不可靠点），在 LiDAR 重定位任务上分别实现 24.1% 和 73.9% 的位置误差相对降低。

**[Learnability-Driven Submodular Optimization for Active Roadside 3D Detection](autonomous_driving/learnability-driven_submodular_optimization_for_active_roadside_3d_detection.md)**

:   提出 LH3D 框架，通过「深度置信度→语义平衡→几何多样性」三阶段子模优化的主动学习策略，抑制路侧单目 3D 检测中固有歧义样本的选取，仅用 20% 标注预算即显著优于传统不确定性/多样性 AL 方法。

**[Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization](autonomous_driving/learning_geometric_and_photometric_features_from_p.md)**

:   本文利用3D LiDAR获取的全景深度图和反射率图作为CNN的输入，构建了一个大规模户外场景分类数据集MPO，并提出了水平循环卷积(HCC)和行最大池化(RWMP)两种改进策略，实现了对六类户外场景的高精度分类（最高97.87%），显著优于传统手工特征方法。

**[Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception](autonomous_driving/learning_mutual_view_information_graph_for_adaptive_adversarial_collaborative_pe.md)**

:   提出 MVIG 攻击框架，通过将不同防御型协作感知系统的脆弱性统一建模为互视图信息图(Mutual View Information Graph)，结合时序图学习与熵感知漏洞搜索，实现自适应的伪造攻击，使防御成功率最高下降 62%。

**[Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos](autonomous_driving/learning_to_drive_is_a_free_gift_large-scale_label-free_autonomy_pretraining_fro.md)**

:   提出LFG（Learning to drive is a Free Gift），一个完全无标签、教师引导的自动驾驶预训练框架，从大规模无姿态YouTube驾驶视频中学习几何、语义和运动感知的统一伪4D表示，在NAVSIM基准上仅用单目前视相机即超越多相机+LiDAR的BEV方法（PDMS 85.2），并展示了出色的数据效率（10%标签即达81.4 PDMS）。

**[LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration](autonomous_driving/lirec-net_a_target-free_and_learning-based_network_for_lidar_rgb_and_event_calib.md)**

:   提出LiREC-Net，首个统一框架同时完成LiDAR-RGB和LiDAR-Event相机的无靶标外参标定，通过共享LiDAR表示（融合3D点特征和投影深度特征）和成对代价体积实现跨模态对齐，在KITTI上达到1.80cm/0.11°、DSEC上达到2.51cm/0.14°（LiDAR-RGB）和1.18cm/0.07°（LiDAR-Event）的标定精度。

**[Look Before You Fuse: 2D-Guided Cross-Modal Alignment for Robust 3D Detection](autonomous_driving/look_before_you_fuse_2d-guided_cross-modal_alignment_for_robust_3d_detection.md)**

:   揭示了LiDAR-Camera融合中特征不对齐主要集中在**前景-背景深度突变边界**，提出PGDC（2D先验引导深度校准）+DAGF（不连续感知几何融合）+SGDM（结构引导深度调制器）三个协同模块，在融合前主动修正不对齐问题，在nuScenes验证集达到mAP 71.5%、NDS 73.6%的SOTA。

**[LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](autonomous_driving/lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)**

:   LR-SGS 提出利用 LiDAR 反射率引导的结构感知 Salient Gaussian 表示，通过将 LiDAR 强度校准为光照不变的反射率通道附加到每个 Gaussian、从几何与反射率特征点初始化结构化 Salient Gaussian、以及 RGB-反射率跨模态梯度一致性约束，在 Waymo 数据集的复杂光照场景中以更少 Gaussian 数量和更短训练时间超越 OmniRe 达 1.18 dB PSNR。

**[M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](autonomous_driving/m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)**

:   M²-Occ 针对相机故障导致视图缺失的真实场景，提出 MMR（利用相邻相机 FoV 重叠在特征空间重建缺失视图表示）+ FMM（可学习语义原型 memory bank 精炼模糊 voxel 特征），在 SurroundOcc 基线上缺失后视摄像头 IoU +4.93%，缺失 5 个摄像头时仍维持 18.36% IoU（基线崩到 13.35%），且完整视图下性能不妥协。

**[MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](autonomous_driving/mapgclr_geospatial_contrastive_learning_of_represe.md)**

:   提出 MapGCLR，一种基于地理空间一致性的对比学习策略，通过强制不同遍历中重叠区域的 BEV 特征具有一致表示，以半监督方式显著提升在线矢量化高精地图构建性能，在仅 5%-20% 标注数据下获得 13%-42% 的相对增益。

**[MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving](autonomous_driving/meanfuser_fast_one-step_multi-modal_trajectory_generation_and_adaptive_reconstru.md)**

:   提出MeanFuser端到端自动驾驶框架，用高斯混合噪声替代离散轨迹词汇表实现连续多模态轨迹建模，通过MeanFlow Identity实现一步采样消除ODE数值误差，并设计ARM模块隐式判断是选择现有proposal还是重构新轨迹，在NAVSIM上以仅RGB输入+ResNet-34骨干达到89.0 PDMS且59 FPS。

**[MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating](autonomous_driving/metadat_generalizable_trajectory_prediction_via_meta_pre-training_and_data-adapt.md)**

:   提出 MetaDAT 框架，通过元预训练获得适合在线自适应的模型初始化，并在测试时利用动态学习率优化和难样本驱动更新实现数据自适应的模型调整，在 nuScenes/Lyft/Waymo 跨数据集分布偏移场景下超越所有 TTT 方法。

**[Mind the Hitch: Dynamic Calibration and Articulated Perception for Autonomous Trucks](autonomous_driving/mind_the_hitch_dynamic_calibration_and_articulated_perception_for_autonomous_tru.md)**

:   提出 dCAP 框架，通过基于 Transformer 的跨视角和时序注意力机制，实现拖挂式自动驾驶卡车中拖头与挂车之间的实时 6-DoF 相对位姿估计，并集成到 BEVFormer 中提升铰接运动下的 3D 目标检测性能（平移误差 0.452m，旋转误差 0.042 rad）。

**[MindDriver: Introducing Progressive Multimodal Reasoning for Autonomous Driving](autonomous_driving/minddriver_introducing_progressive_multimodal_reasoning_for_autonomous_driving.md)**

:   提出渐进式多模态推理框架 MindDriver，模仿人类"感知→想象→行动"机制——先文本语义理解，再想象未来场景图像（桥接语义和物理空间），最后预测轨迹，配合反馈引导数据标注和渐进式强化微调，在 nuScenes 开环和 Bench2Drive 闭环评估上均取得最优表现。

**[Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)](autonomous_driving/monocular_open_vocabulary_occupancy_prediction_for_indoor_scenes.md)**

:   提出 LegoOcc，利用语言嵌入高斯（LE-Gaussians）作为统一的几何-语义中间表示，结合基于 Poisson 过程的高斯到占用（G2O）算子和渐进温度衰减策略，在仅使用二值占用标签（无语义标注）的情况下实现室内场景的单目开放词汇占用预测，在 Occ-ScanNet 上达到 59.50 IoU / 21.05 mIoU。

**[Neural Distribution Prior for LiDAR Out-of-Distribution Detection](autonomous_driving/neural_distribution_prior_for_lidar_ood_detection.md)**

:   NDP提出了可学习的神经分布先验模块来建模网络预测的分布结构，结合Perlin噪声生成的伪OOD样本和软异常暴露策略，在STU基准上实现61.31% AP，超越之前最佳结果10倍以上。

**[NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning](autonomous_driving/nord_a_data-efficient_vision-language-action_model_that_drives_without_reasoning.md)**

:   NoRD 证明自动驾驶 VLA 不需要大规模推理标注和海量数据：通过识别 GRPO 在弱 SFT 策略上失败的根因是 **difficulty bias**（高方差 rollout 组的学习信号被压制），采用 Dr. GRPO 替代标准 GRPO 做 RL 后训练，仅用 <60% 数据、无推理标注、3× 更少 token，在 NAVSIM（85.6 PDMS）和 WaymoE2E（7.709 RFS）上达到与推理型 VLA 竞争的性能。

**[O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](autonomous_driving/o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)**

:   O3N 首次提出全向开放词汇占用预测任务，设计纯视觉端到端框架：Polar-spiral Mamba (PsM) 在极坐标空间以螺旋扫描建模全景几何连续性；Occupancy Cost Aggregation (OCA) 构建 voxel-text 匹配代价体积避免直接特征对齐的过拟合；Natural Modality Alignment (NMA) 通过无梯度随机游走对齐 pixel-voxel-text 三模态嵌入。在 QuadOcc 上达 16.54 mIoU / 21.16 Novel mIoU（SOTA），大幅超越 OVO 基线。

**[OccAny: Generalized Unconstrained Urban 3D Occupancy](autonomous_driving/occany_generalized_unconstrained_urban_3d_occupancy.md)**

:   OccAny 提出了首个泛化无约束城市 3D 占用预测框架，能在无标定、域外场景中从单目/序列/环视图像预测度量级占用体素，通过 Segmentation Forcing 和 Novel View Rendering 两项关键设计，在 KITTI 和 nuScenes 上超越所有视觉几何基线。

**[OccuFly: A 3D Vision Benchmark for Semantic Scene Completion from the Aerial Perspective](autonomous_driving/occufly_a_3d_vision_benchmark_for_semantic_scene_completion_from_the_aerial_pers.md)**

:   OccuFly 提出了首个真实世界航拍视角的相机基语义场景补全（SSC）基准数据集，包含 2 万+ 样本、21 个语义类别，覆盖多季节多海拔的城市/工业/农村场景，并揭示了当前视觉基础模型在航拍场景下的根本局限。

**[On the Feasibility and Opportunity of Autoregressive 3D Object Detection](autonomous_driving/on_the_feasibility_and_opportunity_of_autoregressive_3d_object_detection.md)**

:   提出 AutoReg3D，首个将 LiDAR 3D 目标检测建模为自回归序列生成的框架，利用近到远排序和参数特定词表将 bounding box 离散为 token 序列，无需 anchor/NMS 即可达到与主流方法竞争的性能，并解锁 RL 微调和级联精炼等新能力。

**[OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera](autonomous_driving/oneocc_semantic_occupancy_prediction_for_legged_robots_with_a_single_panoramic_c.md)**

:   提出 OneOcc，一个面向足式/人形机器人的纯视觉全景语义占用预测框架，通过双投影融合、双网格体素化、步态位移补偿和层级混合专家解码器，仅用单个全景相机即可实现 360° 语义场景补全，在真实四足和仿真人形数据集上超越 LiDAR 基线。

**[Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](autonomous_driving/open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)**

:   提出 OVDG-SS 新设定，统一处理语义分割中的未见域和未见类别问题，并设计基于状态空间模型的 S2-Corr 模块来修复域偏移导致的文本-图像相关性退化，在自动驾驶场景中实现高效且鲁棒的跨域开放词汇分割。

**[Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](autonomous_driving/panoramic_multimodal_semantic_occupancy_prediction.md)**

:   面向四足机器人构建首个全景多模态（RGB+热成像+偏振+LiDAR）语义占据数据集PanoMMOcc，并提出VoxelHound框架，通过垂直抖动补偿（VJC）和多模态信息提示融合（MIPF）模块实现鲁棒的3D占据预测，达到23.34% mIoU（+4.16%）。

**[Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule](autonomous_driving/perception_characteristics_distance_measuring_stability_and_robustness_of_percep.md)**

:   提出 Perception Characteristics Distance (PCD)，一种量化感知系统在不同距离下可靠检测能力的新指标，通过统计建模检测置信度随距离的均值和方差变化，定义感知系统的最大可靠检测距离，弥补传统 AP/IoU 等静态指标无法反映距离依赖性和随机性的不足。

**[Plant Taxonomy Meets Plant Counting: A Fine-Grained, Taxonomic Dataset for Counting Hundreds of Plant Species](autonomous_driving/plant_taxonomy_meets_plant_counting_a_fine-grained_taxonomic_dataset_for_countin.md)**

:   本文构建了首个融合植物分类学的大规模计数数据集 TPC-268，包含 10,000 张图、678,050 个点标注和 268 个可计数类别（覆盖 242 个物种），按林奈分类体系标注完整层级信息，并在类无关计数（CAC）范式下进行了全面基准测试。

**[Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors](autonomous_driving/points-to-3d_structure-aware_3d_generation_with_point_cloud_priors.md)**

:   提出 Points-to-3D，将可见区域点云编码为 TRELLIS 的稀疏结构潜变量（SS latent）并用 mask-aware inpainting 网络补全不可见区域，结合结构补全+边界精炼两阶段采样策略，实现几何可控的高保真 3D 资产/场景生成，在 Toys4K 上 F-Score 达 0.964（可见区域 0.998）。

**[ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction](autonomous_driving/proood_prototype-guided_out-of-distribution_3d_occupancy_prediction.md)**

:   本文提出ProOOD框架，首次从体素原型引导的视角统一处理3D占用预测中的长尾识别与分布外（OOD）检测，通过原型引导的语义补全（PGSI）、尾部类增强（PGTM）和无训练的EchoOOD评分机制，在SemanticKITTI上提升+3.57% mIoU（尾部类+24.80%），在VAA-KITTI上OOD检测AuPRCr提升+19.34。

**[PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency](autonomous_driving/ptc-depth_pose-refined_monocular_depth_estimation_with_temporal_consistency.md)**

:   本文提出PTC-Depth，一个结合光流三角化和轮式里程计的单目深度估计框架，通过递归贝叶斯更新追踪深度基础模型的度量尺度，实现时间一致的度量深度预测，在KITTI、TartanAir和热红外等多个数据集上展现强泛化能力。

**[R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](autonomous_driving/r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)**

:   提出 R4Det，通过三个即插即用 BEV 模块——全景深度融合（PDF）、可变形门控时序融合（DGTF）、实例引导动态精炼（IGDR）——系统性解决 4D 雷达-相机融合中的深度估计不准、无位姿时序融合以及小目标检测三大难题，在 TJ4DRadSet 上 3D mAP 达 47.29%（+5.47%），VoD 上 mAP 66.69%。

**[Rascene: High-Fidelity 3D Scene Imaging with mmWave Communication Signals](autonomous_driving/rascene_high-fidelity_3d_scene_imaging_with_mmwave_communication_signals.md)**

:   提出 Rascene，一种利用毫米波 OFDM 通信信号（5G/Wi-Fi）进行高保真 3D 场景成像的集成感知与通信（ISAC）框架，通过置信度加权的多帧融合实现对稀疏、多径干扰的射频观测的几何一致性恢复。

**[Recover to Predict: Progressive Retrospective Learning for Variable-Length Trajectory Prediction](autonomous_driving/recover_to_predict_progressive_retrospective_learning_for_variable-length_trajec.md)**

:   提出渐进式回溯框架 PRF，通过级联回溯单元逐步将不完整观测的特征对齐到完整观测，大幅提升变长轨迹预测性能，且即插即用兼容现有方法。

**[ReMoT: Reinforcement Learning with Motion Contrast Triplets](autonomous_driving/remot_reinforcement_learning_with_motion_contrast_triplets.md)**

:   提出 ReMoT 统一训练范式，通过规则驱动的多专家协作管线自动构建 16.5K 运动对比三元组数据集 (ReMoT-16K)，并结合 GRPO 强化学习与复合奖励（逻辑一致性+长度正则化），系统性解决 VLM 在时空一致性推理上的根本缺陷，实现 25.1% 的性能提升。

**[RESBev: Making BEV Perception More Robust](autonomous_driving/resbev_making_bev_perception_more_robust.md)**

:   提出 RESBev，一个即插即用的 BEV 感知鲁棒性增强框架，通过隐空间世界模型从历史干净帧预测当前 BEV 语义先验，再由异常重建器将先验与被损坏的当前观测通过交叉注意力融合，在 nuScenes 上为四种 LSS 模型在 10 种干扰（含自然损坏 + 对抗攻击）下平均提升 15~20 个 IoU 点，且能泛化到训练未见过的干扰类型。

**[ReScene4D: Temporally Consistent Semantic Instance Segmentation of Evolving Indoor 3D Scenes](autonomous_driving/rescene4d_temporally_consistent_semantic_instance_segmentation_of_evolving_indoo.md)**

:   定义并形式化了时间稀疏的 4D 室内语义实例分割（4DSIS）任务，提出 ReScene4D 方法通过时空对比损失、时空掩码池化和时空序列化三种时序信息共享策略，将 3D 实例分割架构扩展到 4D 维度，在 3RScan 数据集上实现 SOTA，同时提出新的 t-mAP 指标联合评估分割质量和时序身份一致性。

**[SABER: Spatially Consistent 3D Universal Adversarial Objects for BEV Detectors](autonomous_driving/saber_spatially_consistent_3d_universal_adversarial_objects_for_bev_detectors.md)**

:   提出首个面向BEV 3D检测器的非侵入式、3D一致的通用对抗物体生成框架SABER，通过在场景中放置优化后的3D mesh来干扰多视角多帧检测，揭示BEV模型对环境上下文先验的过度依赖。

**[Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems](autonomous_driving/scaling-aware_data_selection_for_end-to-end_autonomous_driving_systems.md)**

:   提出MOSAIC框架——通过聚类数据、拟合各域对评估指标的缩放律、贪心迭代选择边际收益最大的数据簇样本，实现端到端自动驾驶模型的高效数据选择，用80%更少的数据达到甚至超越基线性能。

**[SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving](autonomous_driving/searchad_large-scale_rare_image_retrieval_dataset_for_autonomous_driving.md)**

:   SearchAD 构建了首个面向自动驾驶的大规模稀有图像检索数据集，包含42万+帧图像、51万+标注框、90个稀有类别，支持文本到图像和图像到图像检索，并通过全面评估揭示当前多模态检索模型在稀有物体检索上的不足。

**[SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](autonomous_driving/sgnlf_spectralgeometric_neural_fields_for_posefre.md)**

:   SG-NLF提出一种无需精确位姿的LiDAR NeRF框架，通过谱-几何混合表示解决LiDAR稀疏数据导致的几何空洞问题，利用置信感知图实现全局位姿优化，并引入对抗学习强化跨帧一致性，在nuScenes上重建质量和位姿精度分别比SOTA提升35.8%和68.8%。

**[SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting](autonomous_driving/sharp_short-window_streaming_for_accurate_and_robust_prediction_in_motion_foreca.md)**

:   提出 SHARP，一种基于短窗口流式推理的运动预测框架，通过实例感知上下文流模块显式维护和更新跨时间步的智能体潜在表示，结合双目标训练策略，在 Argoverse 2 多智能体基准上达到流式推理 SOTA，同时保持极低延迟。

**[SimScale: Learning to Drive via Real-World Simulation at Scale](autonomous_driving/simscale_learning_to_drive_via_real-world_simulation_at_scale.md)**

:   提出 SimScale 框架，通过对现有驾驶日志进行轨迹扰动 + 反应式环境仿真 + 神经渲染生成大规模高保真模拟数据，配合伪专家轨迹监督和 sim-real co-training 策略，使端到端规划器在 NAVSIM v2 上取得显著提升（navhard +8.6 EPDMS），且性能随仿真数据量平滑扩展。

**[Single Pixel Image Classification using an Ultrafast Digital Light Projector](autonomous_driving/single_pixel_image_classification_using_an_ultrafa.md)**

:   利用microLED-on-CMOS超快光投影器（330kfps全局快门）进行单像素成像，将12×12 Hadamard pattern投射到MNIST数字上，用单像素光电检测器采集叠加光强的时间序列，完全跳过图像重建，直接用ELM和DNN对时间序列分类，实验实现1.2kfps下>90%多分类精度和>99% AUC的二分类（异常检测）能力。

**[SparseWorld-TC: Trajectory-Conditioned Sparse Occupancy World Model](autonomous_driving/sparseworld_tc_trajectory_conditioned_sparse_occupancy_world_model.md)**

:   提出一种基于纯注意力的稀疏占用世界模型SparseWorld-TC，绕过VAE离散化和BEV中间表示，直接从原始图像特征端到端预测轨迹条件的多帧未来占用，在nuScenes上大幅超越现有方法。

**[Sparsity-Aware Voxel Attention and Foreground Modulation for 3D Semantic Scene Completion](autonomous_driving/sparsity-aware_voxel_attention_and_foreground_modulation_for_3d_semantic_scene_c.md)**

:   提出 VoxSAMNet，一个显式建模体素稀疏性和语义不均衡的单目语义场景补全框架，通过 Dummy Shortcut 跳过空体素、Foreground Dropout + Text-Guided Image Filter 缓解长尾过拟合，在 SemanticKITTI 上达到 18.19% mIoU 的 SOTA（超越现有单目和立体方法）。

**[TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR](autonomous_driving/terraseg_self-supervised_ground_segmentation_for_any_lidar.md)**

:   本文提出 TerraSeg，首个自监督的域无关 LiDAR 地面分割模型，通过构建统一的 OmniLiDAR 大规模数据集（12个公开基准、15种传感器、近2200万次扫描）和创新的 PseudoLabeler 自监督伪标签生成模块，在不使用任何人工标注的情况下在 nuScenes、SemanticKITTI 和 Waymo 上达到 SOTA。

**[TT-Occ: Test-Time 3D Occupancy Prediction](autonomous_driving/test-time_3d_occupancy_prediction.md)**

:   提出 TT-Occ，一种无需预训练的测试时3D占用预测框架，通过在推理时集成视觉基础模型（VFMs）来增量构建、优化和体素化时间感知的3D高斯，在 Occ3D-nuScenes 和 nuCraft 上超越了所有需要大量训练的自监督方法。

**[TopoMaskV3: 3D Mask Head with Dense Offset and Height Predictions for Road Topology Understanding](autonomous_driving/topomaskv3_3d_mask_head_with_dense_offset_and_height_predictions_for_road_topolo.md)**

:   本文提出 TopoMaskV3，通过引入稠密偏移场和稠密高度图两个预测头，将基于掩码的道路拓扑理解范式从 2D 弱模块升级为独立的 3D 中心线预测器，并首次在道路拓扑评估中引入地理不重叠划分和远距离基准，揭示了现有基准因地理重叠导致的性能虚高现象，在地理不重叠基准上达到 SOTA 28.5 OLS。

**[Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](autonomous_driving/towards_balanced_multi-modal_learning_in_3d_human_pose_estimation.md)**

:   提出基于 Shapley 值的模态贡献评估和 Fisher 信息矩阵加权的自适应权重约束（AWC）正则化，解决多模态（RGB/LiDAR/mmWave/WiFi）3D 人体姿态估计中的模态不平衡问题，无需引入额外可学习参数即可实现平衡优化。

**[Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model](autonomous_driving/traffic_scene_generation_from_natural_language_description_for_autonomous_vehicl.md)**

:   提出 TTSG，一个无需训练的模块化框架，能够直接从自由格式自然语言描述生成逼真的交通场景，通过 LLM 驱动的提示分析、道路检索、智能体规划和计划感知道路排序算法，无需预定义路线或生成点，在 SafeBench 上实现最低 3.5% 平均碰撞率。

**[U4D: Uncertainty-Aware 4D World Modeling from LiDAR Sequences](autonomous_driving/u4d_uncertainty-aware_4d_world_modeling_from_lidar_sequences.md)**

:   提出 U4D，首个不确定性感知的 4D LiDAR 世界建模框架，通过"先难后易"的两阶段扩散生成策略，先重建高不确定性区域再条件补全整个场景，并设计 MoST 模块自适应融合时空特征以保证时序一致性。

**[VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation](autonomous_driving/vird_view-invariant_representation_through_dual-axis_transformation_for_cross-vi.md)**

:   提出 VIRD，通过双轴变换（极坐标变换 + 上下文增强位置注意力）构建视图不变表示，在无方向先验条件下实现 SOTA 的跨视角位姿估计，在 KITTI 上位置和方向误差分别降低 50.7% 和 76.5%。

**[Learning Vision-Language-Action World Models for Autonomous Driving](autonomous_driving/vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)**

:   VLA-World将世界模型的预测想象与VLA模型的反思推理统一到一个框架中，通过生成未来帧并对其进行推理来改进轨迹规划，实现了最低的碰撞率和FID分数。

**[WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation](autonomous_driving/walkgpt_grounded_vision-language_conversation_with_depth-aware_segmentation_for_.md)**

:   提出 WalkGPT——首个面向行人无障碍导航的像素定位大视觉语言模型，统一对话推理、分割掩码与深度估计于单一架构中，并构建了 41k 规模的 PAVE 数据集。

**[x2-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](autonomous_driving/x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)**

:   提出 x2-Fusion，以事件相机的时空边缘信号为锚构建统一的 Event Edge Space，将图像/LiDAR/事件特征对齐到同质边缘空间后进行可靠性感知自适应融合和跨维度对比学习，同时估计 2D 光流和 3D 场景流，在合成和真实数据上达到 SOTA。

---

## ✂️ 语义分割 { #segmentation }

**[3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion](segmentation/3m-ti_high-quality_mobile_thermal_imaging_via_calibration-free_multi-camera_cros.md)**

:   提出 3M-TI，一个**无需标定**的多相机跨模态扩散框架，通过在 VAE 潜空间中用跨模态自注意力（CSM）自动对齐并融合未标定的 RGB-热红外图像对，结合错位增强策略，在移动端热成像超分辨率任务上达到 SOTA，并显著提升下游目标检测与语义分割性能。

**[MEDISEG: 药物图像实例分割数据集——预防不良药物事件](segmentation/a_dataset_of_medication_images_with_instance_segme.md)**

:   构建了MEDISEG药物图像实例分割数据集（8262张图像，32类药片，含遮挡/重叠的真实场景），YOLOv8/v9验证3类达99.5% mAP@0.5、32类达80.1%，FsDet few-shot证明MEDISEG预训练在遮挡场景比CURE显著提升（1-shot 0.406 vs 0.131）。

**[A Mixed Diet Makes DINO An Omnivorous Vision Encoder](segmentation/a_mixed_diet_makes_dino_an_omnivorous_vision_encoder.md)**

:   提出 Omnivorous Vision Encoder，通过轻量级 adapter 在冻结的 DINOv2 之上进行跨模态对齐蒸馏训练（RGB/Depth/Segmentation），使单一编码器对不同视觉模态产生一致嵌入，同时保留原始判别语义。

**[AFRO: Bootstrap Dynamic-Aware 3D Visual Representation for Scalable Robot Learning](segmentation/bootstrap_dynamic-aware_3d_visual_representation_for_scalable_robot_learning.md)**

:   提出AFRO自监督3D视觉预训练框架，通过逆动力学模型（IDM）推断潜在动作、扩散Transformer前向动力学模型（FDM）预测未来特征、逆一致性约束保证时序对称性，在RH20T大规模数据上预训练后，MetaWorld 14任务平均成功率76.0%（vs DynaMo-3D 64.9%、PointMAE 63.9%），4个real-world任务也取得最优。

**[Combining Boundary Supervision and Segment-Level Regularization for Fine-Grained Action Segmentation](segmentation/boundary_segment_action_segmentation.md)**

:   提出一种轻量级双损失训练框架用于时序动作分割（TAS），仅增加一个边界输出通道和两个辅助损失（边界回归损失 + CDF 段形状正则化损失），在 MS-TCN、C2F-TCN、FACT 三种架构上一致提升 F1 和 Edit 分数，证明精确分割可以通过简单的损失设计而非更重的架构实现。

**[Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation](segmentation/brewing_stronger_features_dual-teacher_distillation_for_multispectral_earth_obse.md)**

:   提出**DEO(Distillation for Earth Observation)**，一种双教师对比蒸馏框架——用多光谱自蒸馏教师学习光谱表示、用光学VFM教师（DINOv3）注入高级语义先验，使单一学生网络同时擅长光学和多光谱遥感任务，在语义分割、变化检测和分类上全面达到SOTA。

**[CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](segmentation/ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)**

:   提出Concept-Aware LoRA (CA-LoRA)，通过自动识别T2I模型中与特定概念（如视角、风格）相关的权重层，仅对这些层施加LoRA微调，实现对目标域的选择性对齐，同时保留预训练模型的多样化生成能力，用于生成高质量的城市场景分割数据集。

**[DeBias-CLIP: CLIP Is Shortsighted — Paying Attention Beyond the First Sentence](segmentation/clip_shortsighted_beyond_first_sentence.md)**

:   发现CLIP和Long-CLIP模型存在严重的early-token偏向和首句摘要shortcut问题，提出DeBias-CLIP通过去除摘要句、句子子采样和前缀token填充三种简单增强策略，不增加任何额外参数即实现了多个长文本检索基准的SOTA。

**[Comparative Evaluation of Traditional Methods and Deep Learning for Brain Glioma Imaging](segmentation/comparative_evaluation_of_traditional_methods_and_deep_learning_for_brain_glioma.md)**

:   一篇系统性综述论文，全面对比传统方法（阈值分割、区域生长、模糊聚类等）和深度学习方法（CNN、U-Net、SegNet 等）在脑胶质瘤 MRI 分割与分类任务上的表现，结论指出 CNN 架构在准确性和自动化程度上全面优于传统技术。

**[Concept-Guided Fine-Tuning: Steering ViTs away from Spurious Correlations to Improve Robustness](segmentation/concept-guided_fine-tuning_steering_vits_away_from_spurious_correlations_to_impr.md)**

:   提出 CFT（Concept-Guided Fine-Tuning），利用 LLM 生成类别级语义概念并通过 GroundedSAM 零样本分割获取概念掩码，再以 AttnLRP 的 relevance map 与概念区域对齐为目标微调 ViT，仅用 1500 张图即可显著提升 5 个 OOD 基准上的鲁棒性。

**[ConceptPrism: Concept Disentanglement in Personalized Diffusion Models via Residual Token Optimization](segmentation/conceptprism_concept_disentanglement_in_personalized_diffusion_models_via_residu.md)**

:   提出 ConceptPrism，通过引入图像级残余 token 和跨图像排斥损失，在个性化 T2I 扩散模型中自动将共享目标概念与图像特有的残余信息解耦，在 DreamBench 上 CLIP-T/DINO/CLIP-I 全面最优。

**[CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](segmentation/crossearthsar_a_sarcentric_and_billionscale_geospa.md)**

:   提出首个十亿参数级 SAR 视觉基础模型 CrossEarth-SAR，在 DINOv2 ViT backbone 上将 FFN 替换为物理引导的稀疏 MoE（用方向熵、等效视数、局部粗糙度三个 SAR 物理描述符引导路由选择），配套 200K 级跨域预训练数据集及覆盖 8 种域差异的 22 个基准，在 20/22 个跨域语义分割评测上达到 SOTA。

**[CTFS: Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels](segmentation/ctfs_collaborative_teacher_framework_for_forward-looking_sonar_image_semantic_se.md)**

:   提出CTFS，首个专为前视声呐图像设计的半监督语义分割框架，引入多教师协作机制（1个通用教师+2个声呐特异教师，分别模拟声学阴影和能量衰减物理特性），配合多视角伪标签可靠性评估（单教师内稳定性+跨教师间一致性），在仅2%标注下达62.32% mIoU，超越SOTA 5.08个百分点。

**[Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training](segmentation/data_warmup_complexity-aware_curricula_for_efficient_diffusion_training.md)**

:   提出Data Warmup，一种不修改模型或损失函数的课程学习策略，通过语义感知图像复杂度度量（前景显著度×前景典型性）按从简到繁顺序调度训练图像，在ImageNet 256×256上为SiT系列带来IS最高+6.11、FID最低-3.41的改进，且反转课程（先难后简）反而低于均匀基线——证明排序本身是关键机制。

**[DeDelayed: Deleting Remote Inference Delay via On-Device Correction](segmentation/dedelayed_deleting_remote_inference_delay_via_on-device_correction.md)**

:   提出 DeDelayed 端云协同推理框架，将轻量本地图像模型与延迟感知的云端时序预测视频模型结合，通过时序预测训练补偿网络延迟，在 100ms 延迟下比纯本地推理提升 6.4 mIoU、比纯远程推理提升 9.8 mIoU。

**[Detecting AI-Generated Forgeries via Iterative Manifold Deviation Amplification](segmentation/detecting_ai-generated_forgeries_via_iterative_manifold_deviation_amplification.md)**

:   提出 IFA-Net，从"建模什么是真"而非"学什么是假"的角度检测 AI 伪造：利用冻结 MAE 重建输入产生残差暴露偏离自然图像流形的区域，再通过两阶段闭环——粗检测→任务自适应先验注入→放大残差→精细化——迭代放大流形偏差，在 diffusion inpainting 和传统篡改检测上均取得 SOTA。

**[Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](segmentation/direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)**

:   提出一种跳过logits优化过程的开放词汇语义分割方法，基于"同类区域的logits到退化分布的分布差异一致"这一假设，直接通过最优传输路径或最大传输速度的解析解来构造分割图，在8个基准上达到SOTA且无需训练或模型特定调制。

**[DSS: Discover, Segment, and Select for Zero-shot Camouflaged Object Segmentation](segmentation/discover_segment_and_select_a_progressive_mechanism_for_zero-shot_camouflaged_ob.md)**

:   提出DSS三阶段渐进式pipeline(Discover→Segment→Select)，通过自监督视觉编码器+Leiden聚类发现前景(FOD)、SAM生成候选mask、启发式评分+MLLM成对比较选择最优mask，实现零样本无训练的伪装目标分割，尤其在多实例场景上显著优于现有方法。

**[DPAD: Discriminative Perception via Anchored Description for Reasoning Segmentation](segmentation/discriminative_perception_via_anchored_description_for_reasoning_segmentation.md)**

:   针对推理分割(RS)中RL+GRPO训练的geometric reward无法约束reasoning chain是否聚焦目标unique attributes的问题，提出DPAD方法：MLLM生成reasoning chain+geometric localization+anchored description，引入基于CLIP的Discriminative Perception Reward比较description与ROI/AOI的相似度差异，迫使caption更具判别性从而间接约束推理链聚焦目标，ReasonSeg上cIoU提升3.09%且推理链长度减少42%。

**[DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime](segmentation/dsflash_panoptic_scene_graph_realtime.md)**

:   DSFlash 通过合并分割与关系预测 backbone、门控双向关系预测头和 mask-based 动态 patch 剪枝，在 PSG 数据集上以 18ms 延迟（56 FPS）实现 mR@50=30.9 的 SOTA 全景场景图生成。

**[Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](segmentation/efficient_rgb-d_scene_understanding_via_multi-task_adaptive_learning_and_cross-d.md)**

:   提出一种高效 RGB-D 多任务场景理解网络，通过改进的融合编码器利用通道冗余加速特征提取，设计归一化聚焦通道层（NFCL）和上下文特征交互层（CFIL）进行跨维度特征引导，并引入批级别多任务自适应损失函数动态调整各任务学习权重，在 NYUv2/SUN RGB-D/Cityscapes 上同时完成语义分割、实例分割、朝向估计、全景分割和场景分类五项任务，取得精度与速度的双重优势。

**[ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark](segmentation/elvis_enhance_low-light_for_video_instance_segmentation_in_the_dark.md)**

:   ELVIS 提出了首个低光视频实例分割（VIS）框架，通过物理驱动的合成低光视频管线（含运动模糊建模）、无标定退化参数估计网络 VDP-Net、以及将增强解码器集成到 VIS 架构中实现退化与内容解耦，在合成和真实低光视频上分别实现 +3.7AP 和 +2.8AP 的提升。

**[EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection](segmentation/erecu_pseudolabel_evolution_unsupervised_camouflage.md)**

:   提出EReCu统一框架，在DINO师生架构上通过多线索原生感知（MNP）提取纹理+语义先验引导伪标签进化融合（PEF），结合局部伪标签精修（LPR）恢复边界细节，首次统一伪标签引导和特征学习两大UCOD范式，在4个COD数据集上全面SOTA。

**[FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](segmentation/fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)**

:   提出 FCL-COD 框架，通过频率感知低秩适配（FoRA）将伪装场景知识注入 SAM、梯度感知对比学习（GCL）增强前背景特征分离、多尺度频率注意力（MSFA）提炼边界敏感特征，在仅使用边界框标注的弱监督设定下超越了全监督 SOTA 方法。

**[Follow the Saliency: Supervised Saliency for Retrieval-augmented Dense Video Captioning](segmentation/follow_the_saliency_supervised_saliency_for_retrieval-augmented_dense_video_capt.md)**

:   提出 STaRC 框架，通过有监督的帧级显著性学习统一驱动检索（显著性引导分割+检索）和描述生成（显著性提示注入解码器），显著提升密集视频描述(DVC)任务中的时序对齐和字幕质量。

**[FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting](segmentation/fov-net_rotation-invariant_cad_b-rep_learning_via_field-of-view_ray_casting.md)**

:   提出 FoV-Net，首个在 CAD B-rep 学习中同时捕获局部表面几何和全局结构上下文的旋转不变框架，通过局部参考系 UV 网格(LRF UV)和视场光线投射(FoV)描述子实现了在任意 $\mathbf{SO}(3)$ 旋转下的鲁棒分类和分割。

**[From 2D Alignment to 3D Plausibility: Unifying Heterogeneous 2D Priors and Penetration-Free Diffusion for Occlusion-Robust Two-Hand Reconstruction](segmentation/from_2d_alignment_to_3d_plausibility_unifying_heterogeneous_2d_priors_and_penetr.md)**

:   将双手重建解耦为 2D 结构对齐（融合关键点/分割/深度先验）和 3D 空间交互对齐（穿透消除扩散模型），在 InterHand2.6M 上 MPJPE 达到 5.36mm，大幅超越 SOTA。

**[GenMask: Adapting DiT for Segmentation via Direct Mask Generation](segmentation/genmask_adapting_dit_for_segmentation_via_direct_mask_generation.md)**

:   本文提出 GenMask，将 DiT 直接训练为生成黑白分割掩码（与生成彩色图像共用同一模型），通过发现二值掩码的 VAE 潜在表示是线性可分的特殊性质，设计了针对分割的极端长尾时间步采样策略，实现了单步推理即可产出分割结果，在 referring 和 reasoning 分割基准上达到 SOTA。

**[GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](segmentation/geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)**

:   本文提出 GeoGuide，一个层次化几何引导的开放词表 3D 语义分割框架，通过基于不确定性的超点蒸馏、实例级掩码重建和跨实例关系一致性三个互补模块，利用预训练3D模型的几何先验来纠正 2D 到 3D 知识蒸馏中的几何偏差，在 ScanNet v2 上达到 64.8 mIoU 的 SOTA 性能。

**[GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth](segmentation/geomprompt_rgbd_segmentation.md)**

:   GeomPrompt 为冻结的 RGB-D 分割模型学习轻量的几何提示模块，从 RGB 合成任务驱动的深度代理信号（无深度监督），在深度缺失时提升 6.1 mIoU，在深度退化时提升最高 3.6 mIoU。

**[GeoSURGE: Geo-localization using Semantic Fusion with Hierarchy of Geographic Embeddings](segmentation/geosurge_geo-localization_using_semantic_fusion_with_hierarchy_of_geographic_emb.md)**

:   GeoSURGE 提出层级地理嵌入和语义融合模块，将全球图像地理定位问题建模为视觉表征与学习得到的地理表征之间的匹配，在 5 个基准的 25 项指标中取得 22 项 SOTA。

**[GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](segmentation/gkd_generalizable_knowledge_distillation_vfm.md)**

:   提出 GKD 框架，通过将表示学习与任务学习解耦的多阶段蒸馏（先学通用特征 → 冻结编码器 → 再训任务头）+ 查询式软蒸馏机制（QSD），从 VFM 中蒸馏出具有跨域泛化能力的轻量学生模型，在 F2L 设置下平均 mIoU 提升 +10.6%，F2F +1.9%。

**[Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](segmentation/heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)**

:   本文将无监督域适应中的类别课程学习重新定义为强化学习的序贯决策问题，提出 HeuSCM 框架，通过高维语义状态感知和类别公平策略梯度实现自主学习课程规划，在 ACDC、Dark Zurich 和 Nighttime Driving 上达到 SOTA（72.9 mIoU）。

**[HippoMM: Hippocampal-inspired Multimodal Memory for Long Audiovisual Event Understanding](segmentation/hippomm_hippocampal-inspired_multimodal_memory_for_long_audiovisual_event_unders.md)**

:   HippoMM 将海马体的三大认知机制——模式分离（情景分割）、记忆固化（语义压缩）和模式补全（层级检索）——映射为计算架构，用于长音视频的情景记忆和跨模态关联回忆，在自建基准 HippoVlog 上达到 78.2% 准确率并比检索增强基线快 5 倍。

**[INSID3: Training-Free In-Context Segmentation with DINOv3](segmentation/insid3_training-free_in-context_segmentation_with_dinov3.md)**

:   提出INSID3，一种仅依赖冻结DINOv3特征的无训练上下文分割方法，通过位置偏差消除、细粒度聚类和种子聚类聚合三阶段pipeline，在语义/部件/个性化分割任务上以单一自监督骨干网络超越了依赖SAM或微调的方法，平均mIoU提升+7.5%。

**[Kαlos finds Consensus: A Meta-Algorithm for Evaluating Inter-Annotator Agreement in Complex Vision Tasks](segmentation/kαlos_finds_consensus_a_meta-algorithm_for_evaluating_inter-annotator_agreement_.md)**

:   提出KαLOS元算法，通过"先定位后分类"原则和数据驱动的参数校准，将复杂的空间-类别标注一致性问题转化为标准名义可靠性矩阵，统一评估目标检测、实例分割、姿态估计等多种视觉任务的标注者间一致性（IAA）。

**[Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction](segmentation/learning_cross-view_object_correspondence_via_cycle-consistent_mask_prediction.md)**

:   提出基于条件二值分割的跨视角物体对应框架 CCMP，通过循环一致性约束提供自监督信号并支持测试时训练 (TTT)，在 Ego-Exo4D 上达到 44.57% mIoU 的 SOTA 性能。

**[LEMMA: Laplacian Pyramids for Efficient Marine Semantic Segmentation](segmentation/lemma_laplacian_pyramids_for_efficient_marine_semantic_segmentation.md)**

:   提出LEMMA，一种基于拉普拉斯金字塔的轻量级海洋语义分割模型，通过金字塔分解提取边缘信息来替代深层特征计算，在参数量减少71倍的条件下实现了SOTA级别的分割精度（MaSTr1325上98.97% mIoU）。

**[Live Interactive Training for Video Segmentation](segmentation/live_interactive_training_for_video_segmentation.md)**

:   LIT (Live Interactive Training) 提出了一种让交互式视觉系统（如SAM2）在推理时从用户纠正中在线学习的框架，其轻量实现LIT-LoRA通过实时更新LoRA模块将用户反馈泛化到后续帧，在挑战性VOS基准上减少18-34%用户纠正次数，训练开销仅约0.5秒。

**[LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment](segmentation/lod-loc_v3_generalized_aerial_localization_in_dense_cities_using_instance_silhou.md)**

:   本文提出LoD-Loc v3，通过构建10万图像的大规模合成实例分割数据集InsLoD-Loc和将定位范式从语义轮廓对齐升级为实例轮廓对齐，解决了基于LoD城市模型的无人机定位中跨场景泛化差和密集城市歧义两大痛点，在Tokyo-LoDv3密集场景上比SOTA的(2m,2°)精度提升2000%。

**[Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation](segmentation/looking_beyond_the_window_global-local_aligned_clip_for_training-free_open-vocab.md)**

:   针对无训练开放词汇语义分割中滑动窗口带来的跨窗口语义不一致问题，提出 GLA-CLIP 框架，通过全局键值扩展、代理锚点注意力和动态归一化三个机制实现跨窗口全局上下文整合，在8个基准上取得平均 44.0% mIoU 的 SOTA 表现。

**[Love Me, Love My Label: Rethinking the Role of Labels in Prompt Retrieval for Visual In-Context Learning](segmentation/love_me_love_my_label_rethinking_the_role_of_labels_in_prompt_retrieval_for_visu.md)**

:   揭示了视觉上下文学习（VICL）中 prompt 检索忽略标签信息导致标签不一致的问题，提出 LaPR 框架通过图像-标签联合表示和混合专家机制实现标签感知的 prompt 检索，在前景分割、目标检测和图像着色任务上一致超越 SOTA。

**[Low-Data Supervised Adaptation Outperforms Prompting for Cloud Segmentation Under Domain Shift](segmentation/low_data_supervised_adaptation_outperforms_prompting_for_cloud_segmentation.md)**

:   本文系统证明了在卫星遥感云分割任务中，提示工程完全无法弥补视觉-语言模型的域差距，而仅需0.1%（约8张图像）的有标签数据进行微调就能超越所有零样本提示策略。

**[Making Training-Free Diffusion Segmentors Scale with the Generative Power](segmentation/making_training-free_diffusion_segmentors_scale_with_the_generative_power.md)**

:   揭示现有无训练扩散分割方法无法随生成模型能力增强而提升的根本原因——交叉注意力图到语义相关性之间存在两个gap（聚合gap和分数不平衡gap），提出自动聚合（auto aggregation）和逐像素重缩放（per-pixel rescaling）两项技术组成GoCA框架，首次使更强的扩散模型（SDXL、PixArt-Sigma、Flux）在无训练语义分割中显著超越旧模型。

**[MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator](segmentation/matanyone_2_scaling_video_matting_via_a_learned_quality_evaluator.md)**

:   提出学习型 Matting Quality Evaluator (MQE)，在无 ground-truth 条件下逐像素评估 alpha 质量，既作为在线训练引导又作为离线数据筛选器，构建了 28K 片段 / 240 万帧的真实世界视频抠图数据集 VMReal，配合参考帧训练策略，显著超越所有现有方法。

**[MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention](segmentation/mixercseg_an_efficient_mixer_architecture_for_crack_segmentation_via_decoupled_m.md)**

:   提出 MixerCSeg，通过解析 Mamba 的隐式注意力机制将通道解耦为全局/局部分支，分别用 Self-Attention 和 CNN 增强，配合方向引导边缘门控卷积，以 2.05 GFLOPs / 2.54M 参数实现裂缝分割 SOTA。

**[MPM: Mutual Pair Merging for Efficient Vision Transformers](segmentation/mpm_mutual_pair_merging_for_efficient_vision_transformers.md)**

:   提出 Mutual Pair Merging (MPM)，一个无参数、无训练的 ViT token 合并模块，通过互近邻配对+均值融合来减少序列长度，在 ADE20K 上 ViT-Tiny 的 Raspberry Pi 5 延迟降低 60%，H100 上 FlashAttention-2 下吞吐量提升 20%，mIoU 下降控制在 3% 以内。

**[Masked Representation Modeling for Domain-Adaptive Segmentation](segmentation/mrm_masked_representation_modeling_domain_adaptive.md)**

:   提出 Masked Representation Modeling (MRM)，在编码器输出的潜在特征空间做随机掩码与重建，以像素分类损失监督重建结果，作为即插即用辅助任务在四种 UDA 基线上平均提升 +2.3/+2.8 mIoU (GTA→CS / Synthia→CS)，推理时零额外开销。

**[Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models](segmentation/occsam_bench_occlusion_robustness_segmentation.md)**

:   提出 OccSAM-Bench 基准，通过合成手术器械遮挡系统评估 SAM 系列模型在内窥镜场景下的鲁棒性，并设计三区域评估协议揭示模型在遮挡下的两种行为模式：遮挡感知型和遮挡无关型。

**[PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation](segmentation/pca-seg_revisiting_cost_aggregation_for_openvocabulary_semantic_and_part_segmentat.md)**

:   PCA-Seg 提出并行代价聚合(Parallel Cost Aggregation)范式替代传统的串行空间-类别聚合架构，通过专家驱动感知学习(EPL)模块高效整合语义和空间上下文流，并用特征正交解耦(FOD)策略消除两种知识流的冗余，每个并行块仅增加 0.35M 参数即在 8 个开放词汇语义和部件分割基准上达到 SOTA。

**[PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](segmentation/pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)**

:   PEARL 提出了一种基于 Procrustes 对齐和文本感知拉普拉斯传播的两步推理方法，在不引入额外训练或辅助骨干网络的前提下，通过修正 CLIP 最后一层自注意力中 key-query 的几何失配并利用文本语义引导标签传播，在训练免开放词汇语义分割上达到了新的 SOTA。

**[Phrase-Instance Alignment for Generalized Referring Segmentation](segmentation/phrase-instance_alignment_for_generalized_referring_segmentation.md)**

:   本文提出 InstAlign，将广义指代分割 (GRES) 重构为实例级推理问题，通过短语-目标对齐 (POA) 损失建立语言短语与视觉实例的细粒度对应关系，并用相关性加权聚合机制统一处理多目标和无目标场景，在 gRefCOCO 上 cIoU 提升 3.22%、N-acc 提升 12.25%。

**[PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation](segmentation/pixdlm_uav_reasoning_segmentation.md)**

:   本文定义了 UAV Reasoning Segmentation 任务，构建了包含 10K 高分辨率无人机图像和链式推理标注的 DRSeg 基准，并提出了双路径像素级多模态大模型 PixDLM 作为基线。

**[Pointer-CAD: Unifying B-Rep and Command Sequences via Pointer-based Edges & Faces Selection](segmentation/pointer-cad_unifying_b-rep_and_command_sequences_via_pointer-based_edges_faces_s.md)**

:   提出基于指针 (Pointer) 机制的命令序列表示，将 B-Rep 几何实体（边/面）显式引入自回归 CAD 生成，首次在命令序列方法中支持 chamfer/fillet 操作，同时大幅降低量化误差导致的拓扑错误。

**[Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](segmentation/promptdriven_lightweight_foundation_model_for_inst.md)**

:   提出SAM FTI-FDet，通过设计一个基于Transformer decoder的自提示生成器（Prompt Generator），让轻量化的TinyViT-SAM自动生成任务相关的query prompt，无需人工交互即可完成货运列车部件的实例级故障检测，在自建数据集上达到74.6 AP_box / 74.2 AP_mask。

**[PRUE: A Practical Recipe for Field Boundary Segmentation at Scale](segmentation/prue_a_practical_recipe_for_field_boundary_segmentation_at_scale.md)**

:   本文对18个分割和地理空间基础模型（GFM）进行了系统性评估，提出PRUE——一种结合U-Net骨干、复合损失函数和针对性数据增强的农田边界分割方案，在FTW基准上达到76% IoU和47% object-F1，分别比baseline提升6%和9%，同时提出了一套评估部署鲁棒性的新指标。

**[RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](segmentation/rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)**

:   针对遥感图像中目标尺度变化大的难题，提出区域比例感知的动态自适应显著性目标检测网络 RDNet，通过 Proportion Guidance 动态选择不同大小卷积核组合，结合小波频域交互与交叉注意力定位模块，在三个 ORSI-SOD 数据集上全面超越 SOTA。

**[RealVLG-R1: A Large-Scale Real-World Visual-Language Grounding Benchmark for Robotic Perception and Manipulation](segmentation/realvlg-r1_a_large-scale_real-world_visual-language_grounding_benchmark_for_robo.md)**

:   提出 RealVLG 框架，包含 11B 级真实世界多粒度标注数据集 RealVLG-11B 和基于强化学习微调的统一模型 RealVLG-R1，首次将视觉语言定位（VLG）与机器人抓取统一到同一范式中，实现从自然语言指令到 bounding box、分割掩码、抓取姿态和接触点的端到端预测，并展现出零样本泛化能力。

**[Reasoning with Pixel-level Precision: QVLM Architecture and SQuID Dataset for Quantitative Geospatial Analytics](segmentation/reasoning_with_pixel-level_precision_qvlm_architecture_and_squid_dataset_for_qua.md)**

:   提出 QVLM 架构和 SQuID 数据集，通过代码生成+分割模型的解耦设计，在卫星图像上实现像素级精度的定量空间推理，克服了传统 VLM 因 patch embedding 压缩而丢失空间索引的根本限制。

**[RecycleLoRA: Rank-Revealing QR-Based Dual-LoRA Subspace Adaptation for Domain Generalized Semantic Segmentation](segmentation/recyclelora_rank-revealing_qr-based_dual-lora_subspace_adaptation_for_domain_gen.md)**

:   提出 RecycleLoRA，利用 Rank-Revealing QR 分解(RRQR)系统性地"回收"Vision Foundation Model预训练权重中的子空间结构，通过对次要方向和主要方向分别初始化主/子双适配器，显著提升 LoRA 的表示多样性和参数利用效率，在合成到真实和真实到真实的域泛化语义分割任务上均达到 SOTA（平均 mIoU 68.95 / 72.10）。

**[REL-SF4PASS: Panoramic Semantic Segmentation with REL Depth Representation and Spherical Fusion](segmentation/rel-sf4pass_panoramic_semantic_segmentation_with_rel_depth_representation_and_sp.md)**

:   提出 REL 深度表示（基于柱面坐标系的 Rectified Depth + EGVIA + LOA 三通道）和球面动态多模态融合（SMMF），用于全景语义分割，在 Stanford2D3D 上实现 63.06% 平均 mIoU（比 HHA 基线提升 2.35%），并将面对 3D 扰动时的性能方差降低约 70%。

**[RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](segmentation/robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)**

:   本文提出 RobotSeg，第一个同时支持图像和视频的机器人分割基础模型，基于 SAM 2 引入结构增强记忆关联器（SEMA）、机器人提示生成器（RPG）和标签高效训练策略，仅需首帧标注即可训练，在自动模式下 Whole Robot 分割达到 85.1 J&F，比 SAM 2.1 微调版高 4.9 分，同时参数仅 41.3M（远小于现有 638M+ 方案）。

**[RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation](segmentation/rs-ssm_refining_forgotten_specifics_in_state_space_model_for_video_semantic_segm.md)**

:   提出 RS-SSM，通过频域分析提取各通道的特定信息分布特征（CwAP），并自适应反转遗忘门矩阵来补充性精炼 SSM 状态空间压缩时丢失的时空细节（FGIR），在 4 个视频语义分割基准上达到 SOTA 且保持高效率。

**[RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](segmentation/rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)**

:   提出两阶段 RGB-T 显著性检测网络 RSONet：先通过区域引导阶段计算 RGB/热红外引导图与联合引导图的相似度，选出更可靠的模态；再在显著性生成阶段利用选择性优化融合双模态特征，配合密集细节增强和互信息语义模块生成高质量显著图，在三个 RGB-T 基准上取得 SOTA 性能。

**[SAP: Segment Any 4K Panorama](segmentation/sap_segment_any_4k_panorama.md)**

:   提出 SAP（Segment Any 4K Panorama），通过将全景图转化为沿球面固定轨迹采样的透视伪视频序列，解决 SAM2 流式记忆机制在 360° 图像上的结构性失配问题，并合成 183K 实例标注的 4K 全景图进行微调，在真实世界全景基准上实现零样本 mIoU +17.2 的提升。

**[SARMAE: Masked Autoencoder for SAR Representation Learning](segmentation/sarmae_masked_autoencoder_for_sar_representation_learning.md)**

:   提出 SARMAE 框架，通过百万级 SAR 数据集 SAR-1M、散斑感知表征增强 (SARE) 和光学语义锚约束 (SARC)，实现噪声鲁棒的 SAR 自监督预训练，在分类、检测和分割多个下游任务上取得 SOTA。

**[SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](segmentation/scope_scene-contextualized_incremental_few-shot_3d_segmentation.md)**

:   SCOPE 提出一种即插即用的背景引导原型增强框架，利用基础训练场景中背景区域的伪实例构建原型库，在增量阶段通过检索+注意力融合增强少样本原型，无需重训骨干或增加参数即可在 ScanNet/S3DIS 上显著提升新类 IoU（最高 +6.98%）并保持低遗忘。

**[SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](segmentation/sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)**

:   SDDF 提出开放词汇伪装目标检测（OVCOD）新任务，构建了 OVCOD-D 基准，通过子描述主成分对比融合策略去除冗余文本噪声，以及特异性引导的区域弱对齐和动态聚焦机制增强伪装目标与背景的区分能力，在开集设置下达到 56.4 AP。

**[Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](segmentation/seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)**

:   提出 EDA-PSeg 框架，通过图匹配适配器（GMA）和欧拉-边际注意力（EMA）两个核心模块，首次实现从针孔视图到 360° 全景图像的开放集无监督域自适应语义分割，同时处理几何视场角畸变和未知类别发现。

**[SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](segmentation/semitooth_a_generalizable_semisupervised_framework.md)**

:   本文提出SemiTooth框架，通过多教师-多学生架构和更严格的加权置信度约束（SWC），解决多源CBCT数据在半监督牙齿分割中的分布差异问题，在构建的MS3Toothset数据集上取得SOTA。

**[SemLayer: Semantic-aware Generative Segmentation and Layer Construction for Abstract Icons](segmentation/semlayer_semantic-aware_generative_segmentation_and_layer_construction_for_abstr.md)**

:   提出 SemLayer，一个基于生成模型的流水线，将扁平化的矢量图标恢复为语义化分层结构——先通过扩散模型将分割重新定义为上色任务，再进行遮挡区域的语义补全，最后用整数线性规划确定层级顺序，实现 mIoU +5.0、PQ +16.7 的分割提升。

**[SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](segmentation/sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)**

:   提出 SGMA 框架，通过语义引导融合（SGF）模块构建全局语义原型实现自适应跨模态融合，并通过模态感知采样（MAS）模块动态提升脆弱模态的训练频率，解决遥感场景下不完整多模态语义分割中的模态不平衡、类内方差大和跨模态异质性三大挑战。

**[SouPLe: Enhancing Audio-Visual Localization and Segmentation with Learnable Prompt Contexts](segmentation/souple_enhancing_audio-visual_localization_and_segmentation_with_learnable_promp.md)**

:   提出 SouPLe (Sound-aware Prompt Learning)，通过将CLIP中固定的文本提示替换为基于图像特征生成的可学习上下文tokens，增强音频嵌入token与视觉特征之间的语义对应，在VGG-SS上cIoU提升3.75、开放集设定下cIoU提升6.32，全面超越先前方法。

**[SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation](segmentation/spar_single-pass_any-resolution_vit_for_open-vocabulary_segmentation.md)**

:   提出 SPAR，一种通过将细步幅滑窗教师的空间推理能力蒸馏到单次前向传递学生的方法，将 ViT 变为分辨率无关的密集特征提取器，在开放词汇分割中比单次前向基线提升 10.5 mIoU，同时比教师快 52 倍。

**[Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation](segmentation/spatio-semantic_expert_routing_architecture_with_mixture-of-experts_for_referrin.md)**

:   提出 SERA 框架，在冻结的视觉-语言骨干网络中引入两阶段轻量级 MoE 专家精炼（骨干级 SERA-Adapter + 融合级 SERA-Fusion），通过表达式引导的自适应路由实现参考图像分割中的空间一致性和边界精度提升，仅更新不到 1% 的骨干参数。

**[Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation](segmentation/task-oriented_data_synthesis_and_control-rectify_sampling_for_remote_sensing_sem.md)**

:   本文提出TODSynth框架，通过MM-DiT的统一三模态注意力实现文本-图像-掩码联合控制的遥感图像合成，并创新性地提出控制-校正流匹配（CRFM）方法，在采样阶段利用下游分割模型的语义损失动态调整生成轨迹，使合成数据在FUSU-4k和LoveDA上分别提升4.14%和2.08%的mIoU。

**[The Golden Subspace: Where Efficiency Meets Generalization in Continual Test-Time Adaptation](segmentation/the_golden_subspace_where_efficiency_meets_generalization_in_continual_test-time.md)**

:   提出 GOLD 框架用于持续测试时适应（CTTA），核心发现是最小特征更新子空间（"黄金子空间"）与分类器权重行空间一致且天然低秩；通过 Average Gradient Outer Product (AGOP) 在线估计该子空间，结合轻量缩放向量进行特征适应，在分类和分割基准上以极低计算开销达到 SOTA 性能。

**[Towards Context-Aware Image Anonymization with Multi-Agent Reasoning](segmentation/towards_context-aware_image_anonymization_with_multi-agent_reasoning.md)**

:   提出 CAIAMAR 多智能体框架，将预定义的高置信度直接 PII（人体、车牌）处理与基于大视觉语言模型的上下文感知推理相结合，通过 PDCA 迭代优化循环检测间接隐私标识符，使用扩散模型进行外观去相关修复，在 CUHK03-NP 上将行人重识别风险降低 73%，同时在 CityScapes 上保持 FID 9.1 的高图像质量。

**[Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels](segmentation/towards_high-quality_image_segmentation_improving_topology_accuracy_by_penalizin.md)**

:   提出 Same Class Neighbor Penalization (SCNP)，通过在训练时将每个像素的 logit 替换为其同类邻域中最差预测，迫使模型优先修复邻域中的弱分类像素，从而以极低代价（仅 3 行代码、几毫秒/迭代）显著提升分割的拓扑精度。

**[Unified Spherical Frontend: Learning Rotation-Equivariant Representations of Spherical Images from Any Camera](segmentation/unified_spherical_frontend_learning_rotation-equivariant_representations_of_sphe.md)**

:   USF 提出了一个模块化、镜头无关的球面视觉前端，通过将任意标定相机图像投影到单位球面上执行空间域球面重采样、卷积和池化操作，仅用距离加权核就能天然保证旋转等变性，在分类、检测和分割任务上展现了对随机旋转和跨镜头的零样本泛化鲁棒性。

**[Universal 3D Shape Matching via Coarse-to-Fine Language Guidance](segmentation/universal_3d_shape_matching_via_coarse-to-fine_language_guidance.md)**

:   提出 UniMatch，一个语义感知的粗到细 3D 形状匹配框架：粗阶段通过类别无关 3D 分割 + MLLM 命名 + FG-CLIP 语言嵌入建立部件级对应；细阶段通过组级排序对比损失(Group-wise RnC Loss)在扩展的函数映射框架中学习稠密对应，实现跨类别、非等距形状的通用匹配。

**[UnrealPose: Leveraging Game Engine Kinematics for Large-Scale Synthetic Human Pose Data](segmentation/unrealpose_leveraging_game_engine_kinematics_for_large-scale_synthetic_human_pos.md)**

:   提出 UnrealPose-Gen，一个基于 Unreal Engine 5 的合成人体姿态数据生成管线，利用游戏引擎原生骨骼运动学（而非 SMPL）生成百万级标注数据集 UnrealPose-1M，提供 3D 关节、2D 关键点、遮挡标志、实例分割掩码和相机参数等完整标注。

**[VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](segmentation/videomt_your_vit_is_secretly_also_a_video_segmentation_model.md)**

:   提出 VidEoMT，一种纯编码器（encoder-only）视频分割架构，通过 query propagation 和 query fusion 将分割与时序关联统一在单个 ViT 编码器中，在保持与 SOTA 可比精度的同时实现 5×–10× 加速（ViT-L 达 160 FPS）。

**[VIRST: Video-Instructed Reasoning Assistant for SpatioTemporal Segmentation](segmentation/virst_video-instructed_reasoning_assistant_for_spatiotemporal_segmentation.md)**

:   VIRST 提出端到端框架将全局视频推理和像素级 mask 预测统一在单个视觉语言模型中，通过时空融合（STF）和时序动态锚点更新器（TDAU）实现时空一致的视频分割，在 ReVOS 上 J&F 达 70.8（+7.5 over SOTA），MeViS 62.9（+9.2），同时推理速度 5.1 FPS（比 VRS-HQ 快 1.3 倍）。

**[Weakly-Supervised Referring Video Object Segmentation through Text Supervision](segmentation/wsrvos_weakly_supervised_rvos.md)**

:   提出 WSRVOS，首个仅使用文本表达式作为监督信号的弱监督指称视频目标分割框架，通过 MLLM 驱动的对比表达式增强、双向视觉-语言特征选择、实例感知表达式分类和时序分段排序约束，显著减少了对像素级标注的依赖。

---

## 📹 视频理解 { #video_understanding }

**[A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](video_understanding/a_multi-agent_perception-action_alliance_for_efficient_long_video_reasoning.md)**

:   提出 A4VL，一个无训练的多智能体感知-行动联盟框架，通过事件驱动视频分块、线索引导的关键帧选择和多轮智能体协商剪枝机制，在五个视频问答基准上以显著更低的推理延迟全面超越 28 个基线方法。

**[AdaSpark: Adaptive Sparsity for Efficient Long-Video Understanding](video_understanding/adaspark_adaptive_sparsity_for_efficient_long_video_understanding.md)**

:   提出 AdaSpark，通过 3D 时空 cube 分区和两个协同的自适应稀疏机制（cube 级注意力选择 + token 级 FFN 选择），将长视频处理 FLOPs 降低最多 57% 同时保持性能。

**[AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing](video_understanding/autogaze_attend_before_attention_efficient_video.md)**

:   提出 AutoGaze，一个仅 3M 参数的轻量模块，通过自回归地选择最小化重建损失的多尺度 patch 集合，在 ViT 之前移除视频中的冗余信息，实现 4×~100× 的 token 压缩和最高 19× 的 ViT 加速，使 MLLM 能够扩展至 1K 帧 4K 分辨率视频并在 VideoMME 上达到 67.0%。

**[Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding](video_understanding/beyond_single-sample_reliable_multi-sample_distillation_for_video_understanding.md)**

:   揭示视频 LVLM 黑盒蒸馏中单样本 teacher 响应存在严重不可靠性（跨问题方差 σ=0.22、采样内方差 σ=0.07~0.15、格式违规 1%~10%），提出 R-MSD 框架通过多样本 teacher pool + 任务自适应匹配 + 两阶段 SFT→RL 对抗蒸馏解决该问题，4B student 在 VideoMME/Video-MMMU/WorldSense 上全面超越同规模 Qwen3-VL-4B。

**[Temporally Consistent Long-Term Memory for 3D Single Object Tracking](video_understanding/chronotrack_temporally_consistent_long_term_memory_for_3d_single_object_tracking.md)**

:   提出 ChronoTrack，通过紧凑的可学习记忆 token 和两个互补目标（时间一致性损失 + 记忆循环一致性损失）构建鲁棒的长程 3D 单目标跟踪框架，在多个基准上达到 SOTA 并以 42 FPS 实时运行。

**[CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization](video_understanding/cinesrd_leveraging_visual_acoustic_and_linguistic_cues_for_open-world_visual_med.md)**

:   提出 CineSRD，一个免训练的多模态说话人分离框架，通过视觉锚点聚类进行说话人注册，结合音频语言模型进行说话人转换检测，解决影视作品中长视频、大量角色、音视频不同步等开放世界挑战。

**[CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning](video_understanding/clcr_cross-level_semantic_collaborative_representation_for_multimodal_learning.md)**

:   提出 CLCR 框架，将每个模态特征组织为三层语义层级（浅/中/深），通过层内受控交换域（IntraCED）限制跨模态交互仅在共享子空间进行，通过层间协同聚合域（InterCAD）实现跨层自适应融合，解决多模态学习中的跨层语义不同步问题。

**[Cluster-Wise Spatio-Temporal Masking for Efficient Video-Language Pretraining](video_understanding/cluster-wise_spatio-temporal_masking_for_efficient_video-language_pretraining.md)**

:   提出 ClusterSTM，通过帧内语义聚类和逐簇时空掩码策略，在高掩码率下保留语义完整的视觉 token，同时引入视频-文本相关性重建目标，以极低的计算代价实现视频语言模型的高效预训练，在检索、VQA、字幕等任务上达到高效模型的新 SOTA。

**[Color When It Counts: Grayscale-Guided Online Triggering for Always-On Streaming Video Sensing](video_understanding/color_when_it_counts_grayscale-guided_online_triggering_for_always-on_streaming_.md)**

:   提出"灰度常开、彩色按需"新范式，通过 ColorTrigger 在灰度流上用轻量二次规划在线检测色彩冗余，仅使用 8.1% 的 RGB 帧即保持全彩基线 91.6% 的性能，实现资源受限设备的 always-on 视频感知。

**[CVA: Context-aware Video-text Alignment for Video Temporal Grounding](video_understanding/cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)**

:   提出 CVA（Context-aware Video-text Alignment）框架，通过 Query-aware Context Diversification（QCD）、Context-invariant Boundary Discrimination（CBD）损失和 Context-enhanced Transformer Encoder（CTE）三个协同组件，解决视频时序定位中的假阴性和背景关联问题，在 QVHighlights 上 R1@0.7 提升约 5 个点。

**[Decompose and Transfer: CoT-Prompting Enhanced Alignment for Open-Vocabulary Temporal Action Detection](video_understanding/decompose_and_transfer_cot-prompting_enhanced_alignment_for_open-vocabulary_temp.md)**

:   提出 Phase-wise Decomposition and Alignment (PDA) 框架，利用 LLM 的 CoT 推理能力将动作标签分解为"开始-中间-结束"三个阶段描述，通过文本引导的前景过滤和自适应阶段对齐实现细粒度动作模式迁移，在 THUMOS14 OV-TAD 上 Avg mAP 达 46.9（超越 SOTA Ti-FAD 的 41.2）。

**[DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding](video_understanding/divide_then_ground_adapting_frame_selection_to_query_types_for_long-form_video_u.md)**

:   提出 DIG，一个免训练的帧选择框架，通过将查询分为全局查询和定位查询两类，对全局查询使用均匀采样、对定位查询使用一套专门的内容自适应帧选择+LMM奖励评分+视频精炼流水线，在三个长视频理解基准上持续超越现有方法。

**[Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](video_understanding/do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)**

:   提出 EgoPointVQA 数据集和 HINT（Hand Intent Tokens）方法，通过将 3D 手部关键点编码为手意图 token 并与视觉 token 交错输入 MLLM，解决第一人称视频中基于手势指向的指示性问答任务，HINT-14B 达 68.1% 准确率超越 InternVL3-14B 5.4pp。

**[Drift-Resilient Temporal Priors for Visual Tracking](video_understanding/drift-resilient_temporal_priors_for_visual_tracking.md)**

:   提出 DTPTrack——一个轻量即插即用的时序建模模块，通过时序可靠性校准器（TRC）为历史帧分配可靠性分数过滤噪声，并通过时序引导合成器（TGS）将校准后的历史信息合成为动态先验 token 抑制跟踪漂移，在多个基准上达到 SOTA。

**[Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry](video_understanding/dual-agent_reinforcement_learning_for_adaptive_and_cost-aware_visual-inertial_od.md)**

:   提出双智能体强化学习框架，通过 Select Agent（基于IMU信号决定是否启动视觉前端）和 Fusion Agent（自适应融合视觉-惯性状态）两个轻量RL策略，在不完全移除VIBA的前提下大幅降低其调用频率和计算开销，实现精度-效率-显存的更优折中。

**[EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions](video_understanding/egoxtreme_a_dataset_for_robust_object_pose_estimation_in_egocentric_views_under_.md)**

:   提出 EgoXtreme，首个面向极端条件下第一人称视角的大规模 6D 物体位姿估计基准数据集，涵盖严重运动模糊、动态光照和烟雾遮挡三种真实挑战，揭示了当前 SOTA 位姿估计器在这些条件下的严重失效。

**[Enhancing Accuracy of Uncertainty Estimation in Appearance-based Gaze Tracking with Probabilistic Evaluation and Calibration](video_understanding/enhancing_accuracy_of_uncertainty_estimation_in_appearance-based_gaze_tracking_w.md)**

:   提出一种数据高效的后验校准方法，通过等保序回归将不确定性感知视线追踪模型的预测分布与真实观测分布对齐，并引入 Coverage Probability Error (CPE) 指标替代不可靠的误差-不确定性相关性(EUC)来评估不确定性质量。

**[Envisioning the Future, One Step at a Time](video_understanding/envisioning_the_future_one_step_at_a_time.md)**

:   本文将开放集未来场景动态预测建模为稀疏点轨迹的逐步推理，通过自回归扩散模型实现从单张图像快速生成数千种多样化未来假设，速度比稠密模型快数个数量级。

**[Event6D: Event-based Novel Object 6D Pose Tracking](video_understanding/event6d_event-based_novel_object_6d_pose_tracking.md)**

:   EventTrack6D 提出事件-深度融合的 6D 位姿追踪框架，通过在任意时间戳重建强度和深度图像来弥补事件相机与深度帧率的差异，在仅合成数据训练的条件下以 120+ FPS 实现了对未见目标的鲁棒追踪。

**[FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](video_understanding/fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)**

:   提出 FC-Track，一种轻量级的后关联校正框架，通过基于 IoA（Intersection over Area）的重叠感知外观特征过滤和局部不匹配重分配策略，在在线 MOT 中显式纠正由目标重叠引起的身份切换错误，将长期身份切换比例降至 29.55%。

**[FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](video_understanding/fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)**

:   提出 FluxMem，一个无需训练的流式视频理解框架，通过层级化记忆设计（短期/中期/长期）和两个自适应 token 压缩模块（TAS 去时间冗余 + SDC 去空间冗余），在丢弃 60-70% 视觉 token 的同时在 StreamingBench 和 OVO-Bench 上取得新 SOTA。

**[Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding](video_understanding/frame2freq_spectral_adapters_for_fine-grained_video_understanding.md)**

:   提出 Frame2Freq——首个在频域进行时序建模的 PEFT 适配器族，通过 FFT 将冻结 VFM 的帧嵌入变换到频谱空间并学习频带级滤波，在五个细粒度动作识别基准上以 <10% 的可训练参数超越全量微调模型。

**[GoalForce: Teaching Video Models to Accomplish Physics-Conditioned Goals](video_understanding/goal_force_teaching_video_models_to_accomplish_physics-conditioned_goals.md)**

:   提出 Goal Force 框架，通过多通道物理控制信号（目标力、直接力、质量）在简单合成数据上训练视频生成模型，使其学会从目标效果逆向规划因果链，实现零样本泛化到工具使用、人-物交互等复杂现实场景。

**[Hear What Matters! Text-conditioned Selective Video-to-Audio Generation](video_understanding/hear_what_matters_text-conditioned_selective_video-to-audio_generation.md)**

:   SelVA 提出了文本条件的选择性视频到音频（V2A）生成任务，通过可学习的补充 token [SUP] 和自监督视频混合策略，使模型能够根据文本提示从多声源视频中仅生成用户指定的目标声音，在音频质量、语义对齐和时间同步上均超越现有方法。

**[HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering](video_understanding/herbench_a_benchmark_for_multi-evidence_integration_in_video_question_answering.md)**

:   HERBench 是一个专为多证据整合设计的视频问答基准，包含 26,806 个五选一问题，每题结构性地要求融合 ≥3 个时间分散的非重叠视觉线索；通过提出最小必需帧集（MRFS）指标揭示了当前 Video-LLM 的两个关键瓶颈：帧检索不足和证据融合失败。

**[HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling](video_understanding/hieramamba_video_temporal_grounding_via_hierarchical_anchor-mamba_pooling.md)**

:   HieraMamba 提出了基于 Mamba 的层次化视频时间定位架构，核心是 Anchor-MambaPooling（AMP）模块，用 Mamba 的选择性扫描将视频特征逐层压缩为多尺度锚点 token，配合 anchor-conditioned 和 segment-pooled 对比损失增强层次表示的紧凑性和判别性，在 Ego4D-NLQ、MAD 和 TACoS 上达到 SOTA。

**[How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](video_understanding/how_should_video_llms_output_time.md)**

:   本文在统一框架下对比了视频时序定位的三种主流时间输出范式（文本数字生成、时间token生成、连续时间解码），发现连续分布范式在效率-精度帕累托前沿上始终表现最优。

**[LAOF: Robust Latent Action Learning with Optical Flow Constraints](video_understanding/laof_robust_latent_action_learning_with_optical_flow_constraints.md)**

:   提出LAOF框架，利用智能体的光流作为伪监督信号约束潜动作学习，使潜动作表示对干扰更鲁棒，在LIBERO和PROCGEN上显著超越无监督基线，且在无标签条件下匹配或超越使用1%动作标签的监督方法。

**[Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning](video_understanding/learning_to_assist_physics-grounded_human-human_control_via_multi-agent_reinforc.md)**

:   提出 AssistMimic，将人-人辅助交互动作的物理模仿建模为多智能体强化学习（MARL）问题，通过运动先验初始化、动态参考重定向和接触促进奖励，首次实现了力交换型辅助动作的物理仿真跟踪。

**[LensWalk: Agentic Video Understanding by Planning How You See in Videos](video_understanding/lenswalk_agentic_video_understanding_by_planning_how_you_see_in_videos.md)**

:   提出LensWalk，一个让LLM推理器主动控制视频观测范围和采样密度的智能体框架，通过reason-plan-observe循环实现自适应视频理解，无需微调即可在长视频基准上带来5%以上的即插即用性能提升。

**[LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding](video_understanding/longvideo-r1_smart_navigation_for_low-cost_long_video_understanding.md)**

:   提出 LongVideo-R1，一个配备推理能力的多模态 Agent，通过层次化视频树结构和智能导航策略，以平均仅 10.5 轮工具调用实现高效长视频问答，在精度-效率权衡上显著优于穷举式方法。

**[Mamba-VMR: Multimodal Query Augmentation via Generated Videos for Precise Temporal Grounding](video_understanding/mamba-vmr_multimodal_query_augmentation_via_generated_videos_for_precise_tempora.md)**

:   提出一个两阶段视频时刻检索框架：第一阶段用LLM引导字幕匹配并生成辅助短视频作为时序先验，第二阶段用多模态控制Mamba网络高效融合生成先验与长序列，在TVR数据集上超越SOTA（R@1/IoU=0.5达45.20%），同时降低计算开销。

**[MaskAdapt: Learning Flexible Motion Adaptation via Mask-Invariant Prior for Physics-Based Characters](video_understanding/maskadapt_learning_flexible_motion_adaptation_via_mask-invariant_prior_for_physi.md)**

:   本文提出 MaskAdapt 框架，通过两阶段残差学习范式——先训练遮蔽不变的鲁棒基础策略，再训练冻结基础控制器上的残差策略来修改目标身体部位——实现灵活精准的物理仿真人形角色动作适应。

**[MINERVA-Cultural: A Benchmark for Cultural and Multilingual Long Video Reasoning](video_understanding/minerva-cultural_a_benchmark_for_cultural_and_multilingual_long_video_reasoning.md)**

:   提出 MINERVA-Cultural 基准，包含 18 个语种/地区的 2400 个人工标注视频推理问题，通过证据图（evidence graph）和迭代错误隔离策略揭示当前 SOTA Video-LLM 在文化视觉感知上的严重不足（最强模型 Gemini-2.5-Pro 仅 45.07% vs 人类 95.22%）。

**[Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](video_understanding/mistake_attribution_fine-grained_mistake_understanding_in_egocentric_videos.md)**

:   本文提出 Mistake Attribution (MATT) 任务，将第一人称视频中的操作错误归因到语义（违反了指令的哪个成分）、时间（不可逆转点 PNR 在哪一帧）和空间（PNR 帧中错误区域在哪里）三个维度，通过 MisEngine 数据引擎自动从已有动作数据集构建大规模错误样本，并设计统一的 Transformer 模型 MisFormer 同时完成三个归因子任务，在多个基准上超越各子任务的专用 SOTA 方法。

**[MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark](video_understanding/movierecapsqa_a_multimodal_open-ended_video_question-answering_benchmark.md)**

:   提出 MovieRecapsQA，一个基于电影解说视频构建的多模态开放式视频问答基准，包含 60 部电影的约 8.2K 个问题，并设计了基于原子事实 (atomic facts) 的无参考评估指标，揭示了当前 MLLM 在视觉感知而非推理上的关键瓶颈。

**[Ninja Codes: Neurally Generated Fiducial Markers for Stealthy 6-DoF Tracking](video_understanding/ninja_codes_neurally_generated_fiducial_markers_for_stealthy_6-dof_tracking.md)**

:   Ninja Codes 利用深度隐写术技术，通过端到端训练的编码器将任意图像转化为视觉上不显眼的基准标记，可用标准打印机打印并用RGB相机检测，实现隐蔽的6-DoF位置追踪。

**[Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](video_understanding/occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)**

:   提出遮挡感知跟踪框架 OA-SORT，通过显式建模目标遮挡状态来缓解位置代价混淆和 Kalman Filter 估计不稳定问题，在 DanceTrack/SportsMOT/MOT17 上均取得 SOTA 级提升，且组件可即插即用地集成到多种跟踪器中。

**[OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments](video_understanding/openmarcie_dataset_for_multimodal_action_recognition_in_industrial_environments.md)**

:   提出目前最大规模的工业场景多模态动作识别数据集 OpenMarcie，融合可穿戴传感器与视觉数据共 8 种模态、200+ 通道、37+ 小时录制，并在 HAR 分类、开放词表描述、跨模态对齐三个基准上验证了惯性+视觉融合的优越性。

**[Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation](video_understanding/out_of_sight_out_of_track_adversarial_attacks_on_propagation-based_multi-object_.md)**

:   首次系统分析 Tracking-by-Query-Propagation（TBP）跟踪器的对抗脆弱性，提出 FADE 攻击框架，通过时序查询洪泛（TQF）耗尽固定查询预算和时序记忆腐蚀（TMC）破坏隐状态传播两种策略，在 MOT17/MOT20 上对 MOTR/MOTRv2/MeMOTR/Samba/CO-MOT 造成最高约 30 点 HOTA 下降和 10 倍以上身份切换。

**[Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](video_understanding/question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)**

:   提出 QViC-MF 框架，通过问题引导的多帧视觉压缩（QMSA）和上下文记忆反馈机制，在长视频理解任务上以极少的视觉 token（每帧仅 16 个）实现了 MLVU/LVBench/VNBench 等多个基准上的 SOTA。

**[RAGTrack: Language-aware RGBT Tracking with Retrieval-Augmented Generation](video_understanding/ragtrack_language-aware_rgbt_tracking_with_retrieval-augmented_generation.md)**

:   首次将文本描述引入 RGBT 跟踪，提出基于检索增强生成（RAG）的框架 RAGTrack，通过多模态 Transformer 编码器、自适应 Token 融合和上下文感知推理模块，在四个 RGBT 基准上取得 SOTA。

**[Real-World Point Tracking with Verifier-Guided Pseudo-Labeling](video_understanding/realworld_point_tracking_with_verifierguided_pseud.md)**

:   提出一个可学习的Verifier元模型，在合成数据上训练"判断tracker预测可靠性"的能力并迁移到真实世界，通过逐帧评估6个预训练tracker的预测来选取最可靠的作为伪标签，仅用~5K真实视频即微调出在4个真实世界基准上全面SOTA的Track-On-R模型。

**[Reconstruction-Guided Slot Curriculum: Addressing Object Over-Fragmentation in Video Object-Centric Learning](video_understanding/reconstruction-guided_slot_curriculum_addressing_object_over-fragmentation_in_vi.md)**

:   提出 SlotCurri，一种重建引导的 slot 数量课程学习策略，从极少 slot 开始训练并仅在重建误差高的区域逐步扩展 slot 容量，配合结构感知损失和循环推理，有效解决视频物体中心学习中单一物体被多个 slot 错误拆分的过度碎片化问题，在 YouTube-VIS 上实现 +6.8 FG-ARI 提升。

**[FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT](video_understanding/rethinking_two-stage_referring-by-tracking_in_referring_multi-object_tracking_ma.md)**

:   提出 FlexHook，一种新颖的两阶段 Referring-by-Tracking 框架，通过基于采样的 Conditioning Hook（C-Hook）重新定义特征构建，并用 Pairwise Correspondence Decoder（PCD）替换 CLIP 余弦相似度匹配，首次使两阶段方法全面超越当前 SOTA 的一阶段方法。

**[SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning](video_understanding/sail_similarity-aware_guidance_and_inter-caption_augmentation-based_learning_for.md)**

:   提出 SAIL，通过跨模态相似度引导的语义感知掩码生成和 LLM 合成字幕的辅助监督，在仅有字幕标注（无时间边界）的弱监督设置下，在 ActivityNet 和 YouCook2 上实现密集视频描述和事件定位的双 SOTA。

**[SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion](video_understanding/savax_egotoexo_imitation_error_detection_via_scene.md)**

:   形式化 Ego→Exo 模仿错误检测任务，并提出 SAVA-X (Align–Fuse–Detect) 框架，通过自适应采样、场景自适应视角嵌入和双向交叉注意力融合三个模块联合解决时序不对齐、视频冗余和跨视角域差距三大挑战。

**[Seen-to-Scene: Keep the Seen, Generate the Unseen for Video Outpainting](video_understanding/seen_to_scene_keep_the_seen_generate_the_unseen_for_video_outpainting.md)**

:   提出 Seen-to-Scene，统一传播式和生成式范式的视频外推框架，通过参考帧引导的潜空间传播与视频扩散模型结合，在零样本推理中实现了超越需要输入特定适配的先进方法的时空一致性和视觉保真度。

**[SHOW3D: Capturing Scenes of 3D Hands and Objects in the Wild](video_understanding/show3d_capturing_scenes_of_3d_hands_and_objects_in_the_wild.md)**

:   提出首个真正野外环境下具有精确3D标注的手-物体交互数据集SHOW3D，通过设计轻便可穿戴多相机背包系统和ego-exo融合标注pipeline，采集430万帧多视角数据，手部和物体均达到亚厘米级标注精度，跨数据集实验验证其训练模型的泛化优势。

**[SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)**

:   提出SkeletonContext框架，通过跨模态上下文提示模块从预训练语言模型重建骨骼数据缺失的环境和物体上下文语义，并用关键部位解耦模块增强运动关键关节的判别力，在NTU-60/120和PKU-MMD上的零样本和广义零样本设置中达到SOTA。

**[SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding](video_understanding/slotvtg_object-centric_adapter_for_generalizable_video_temporal_grounding.md)**

:   提出SlotVTG框架，通过在MLLM解码器早期层插入轻量级Slot Adapter将视觉token分解为对象级slot表示，辅以DINOv2先验的Slot Alignment Loss引导语义一致的slot形成，显著提升视频时序定位的域外泛化性能（OOD R1@0.5最大提升+4.3），同时仅增加约0.25%的可训练参数。

**[SpikeTrack: A Spike-driven Framework for Efficient Visual Tracking](video_understanding/spiketrack_a_spike-driven_framework_for_efficient_visual_tracking.md)**

:   提出 SpikeTrack，首个完全符合脉冲驱动范式的 RGB 视觉跟踪框架，通过非对称时间步扩展、单向信息流和脑启发记忆检索模块（MRM），在 SNN 跟踪器中达到 SOTA 并与 ANN 跟踪器持平，同时能耗仅为 TransT 的 1/26。

**[Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning](video_understanding/stay_in_your_lane_role_specific_queries_with_overlap_suppression_loss_for_dense_.md)**

:   提出 ROS-DVC，通过将 DETR-based DVC 框架中的共享 query 分离为独立的 localization query 和 caption query，并设计 Overlap Suppression Loss 惩罚 query 间的时序重叠、Cross-Task Contrastive Alignment 保证跨任务语义一致性，在 YouCook2 和 ActivityNet Captions 上实现了 SOTA 的 captioning 和 localization 性能。

**[STORM: End-to-End Referring Multi-Object Tracking in Videos](video_understanding/storm_referring_multi_object_tracking.md)**

:   STORM 是首个端到端的多模态大语言模型框架用于指代多目标跟踪（RMOT），通过任务组合学习策略大幅减少对 RMOT 标注数据的依赖，并构建了高质量 STORM-Bench 数据集。

**[StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](video_understanding/streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)**

:   提出首个注视引导的流式视频理解基准 StreamGaze，包含 8521 个 QA 对覆盖过去/现在/主动预测三类任务，通过注视轨迹-视频对齐的数据构建管线生成时空grounded的QA，揭示了当前 MLLM 在利用注视信号进行时间推理方面的巨大差距。

**[StreamingTOM: Streaming Token Compression for Efficient Video Understanding](video_understanding/streamingtom_streaming_token_compression_for_efficient_video_understanding.md)**

:   提出 StreamingTOM，一个无需训练的两阶段流式视频理解框架：Causal Temporal Reduction (CTR) 在 LLM 前通过因果时序选择将每帧 token 从 196 压缩到 50，Online Quantized Memory (OQM) 在 LLM 后通过 4-bit 量化和按需检索限制 kv-cache 增长，实现 15.7× 压缩比、1.2× 更低峰值显存和 2× 更快 TTFT。

**[StreamReady: Learning What to Answer and When in Long Streaming Videos](video_understanding/streamready_learning_what_to_answer_and_when_in_long_streaming_videos.md)**

:   提出就绪性感知的流式视频理解范式，通过可学习的 `<RDY>` token 和 Answer Readiness Score (ARS) 指标，让模型不仅回答正确，还能在证据出现的恰当时刻作答，在 9 个流式/离线视频基准上取得 SOTA。

**[SVAgent: Storyline-Guided Long Video Understanding via Cross-Modal Multi-Agent Collaboration](video_understanding/svagent_storyline_guided_long_video_understanding_via_cross_modal_multi_agent_collaboration.md)**

:   提出 SVAgent，一个故事线引导的跨模态多智能体框架用于长视频问答，通过渐进式构建叙事表示、DPP 证据选择、跨模态一致性验证和迭代精炼实现超越基线 5.5%-11.5% 的性能提升。

**[Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition](video_understanding/tcei_test_time_calibration_experience_intuition_mot.md)**

:   TCEI 框架受 Kahneman 双系统理论启发，提出直觉系统（利用近期观测对象的瞬时记忆快速推断）和经验系统（利用历史视频积累的经验校准直觉预测）相结合的测试时自适应方法，无需反向传播即可在分布偏移下显著提升多目标跟踪性能。

**[Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models](video_understanding/token_reduction_via_local_and_global_contexts_optimization_for_efficient_video_l.md)**

:   提出 AOT 框架，通过建立局部-全局 token anchors 并利用最优传输（Optimal Transport）在帧内和帧间两级聚合被裁剪/合并 token 的语义信息，实现 training-free 的视频 token 压缩，在裁剪 90% token 的情况下仍保留 97.6% 的原始性能。

**[TrajTok: Learning Trajectory Tokens Enhances Video Understanding](video_understanding/trajtok_learning_trajectory_tokens_enables_better_video_understanding.md)**

:   提出 TrajTok——一种端到端可微的轨迹 tokenizer，将视频像素隐式聚类为目标轨迹 token，取代外部分割+跟踪流水线；在从头训练 (TrajViT2)、特征适配 (TrajAdapter) 和视觉语言模型连接器 (TrajVLM) 三种场景下均取得显著提升，尤其在长视频 QA 上大幅超越 patch pooling。

**[U2Flow: Uncertainty-Aware Unsupervised Optical Flow Estimation](video_understanding/u2flow_uncertainty_aware_unsupervised_optical_flow_estimation.md)**

:   U2Flow是首个联合估计光流和逐像素不确定性的循环无监督框架，通过基于增强一致性的解耦不确定性学习和不确定性引导的双向光流融合，在KITTI和Sintel上实现无监督SOTA。

**[UETrack: A Unified and Efficient Framework for Single Object Tracking](video_understanding/uetrack_a_unified_and_efficient_framework_for_single_object_tracking.md)**

**[UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models](video_understanding/ufvideo_towards_unified_fine-grained_video_cooperative_understanding_with_large_.md)**

:   UFVideo 是首个统一全局、像素级和时序级三种粒度视频理解能力的 Video LLM，通过视觉-语言引导对齐策略和 SAM2 mask decoder，在单一模型内同时支持视频问答、目标引用、视频分割和时序定位，并构建了多粒度协同理解基准 UFVideo-Bench。

**[Understanding Temporal Logic Consistency in Video-Language Models through Cross-Modal Attention Discriminability](video_understanding/understanding_temporal_logic_consistency_in_video-language_models_through_cross-.md)**

:   本文从可解释性角度分析了视频语言模型（Video-LLMs）时间理解逻辑不一致的根本原因——跨模态注意力头无法有效区分不同时间戳的视频token——并提出 TCAS（Temporally Conditioned Attention Sharpening）方法通过优化注意力分布显著提升了时间逻辑一致性和通用时序定位性能。

**[Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention](video_understanding/unified_spatiotemporal_token_compression_for_video-llms_at_ultra-low_retention.md)**

:   提出统一时空token压缩方法，通过全局保留池联合评估token的贡献度和语义冗余度，并在LLM内部引入文本感知合并机制，在仅保留约2%视觉token的极端压缩下仍保留90.1%的基线性能，同时将FLOPs降至约2.6%。

**[UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking](video_understanding/utptrack_towards_simple_and_unified_token_pruning_for_visual_tracking.md)**

:   提出 UTPTrack，首个在 one-stream Transformer 跟踪器中**同时对搜索区域 (SR)、动态模板 (DT) 和静态模板 (ST) 三个组件进行联合 token 剪枝**的统一框架，在 RGB 和多模态/语言引导跟踪中实现 65–67% 的视觉 token 裁减，且保持 99.7%–100.5% 的基线性能。

**[VecAttention: Vector-wise Sparse Attention for Accelerating Long Context Inference](video_understanding/vecattention_vector-wise_sparse_attention_for_accelerating_long_context_inferenc.md)**

:   本文发现视频模型注意力图中存在强烈的"垂直向量"稀疏模式，据此提出 VecAttention 细粒度向量级稀疏注意力框架，通过 TilingSelect + minS 过滤实现高效重要向量选择，在 78%+ 稀疏度下视频理解准确率与全注意力持平，注意力计算加速 2.65 倍。

**[VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](video_understanding/videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)**

:   VideoARM 提出了一种基于分层多模态记忆（HM3）的 Agent 推理范式，通过"观察-思考-行动-记忆"的自适应循环和粗到细的工具调用策略，在长视频理解基准上超越 SOTA 的同时将 token 消耗降低到 DVD 的 1/34。

**[VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice](video_understanding/videoauto-r1_video_auto_reasoning_via_thinking_once_answering_twice.md)**

:   提出 VideoAuto-R1，一个"按需推理"的视频理解框架：训练时采用"思考一次、回答两次"（answer→think→answer）范式，推理时通过首次回答的置信度决定是否启动 CoT 推理，在保持 SOTA 精度的同时将平均响应长度从 149 降至 44 token（约 3.3 倍压缩）。

**[VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](video_understanding/videochatm1_collaborative_policy_planning_for_vide.md)**

:   VideoChat-M1 提出协作策略规划（CPP）范式和多智能体强化学习（MARL）训练方法，让 4 个异构 VLM agent 动态生成和更新工具调用策略来理解视频，在 LongVideoBench 上超过 Gemini 2.5 Pro 3.6%、GPT-4o 15.6%。

**[VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking](video_understanding/videoseek_long-horizon_video_agent_with_tool-guided_seeking.md)**

:   VideoSeek 提出一种长程视频 Agent，利用视频逻辑流主动"寻找"关键证据而非穷举解析所有帧，通过 think-act-observe 循环和多粒度工具包（overview/skim/focus），在 LVBench 上比基座模型 GPT-5 提升 10.2 个点的同时减少 93% 的帧使用量。

**[VidTAG: Temporally Aligned Video to GPS Geolocalization](video_understanding/vidtag_video_gps_geolocalization.md)**

:   提出 VidTAG，一个双编码器（CLIP+DINOv2）帧到GPS检索框架，通过 TempGeo 模块实现帧间时间对齐，GeoRefiner 编码器-解码器模块精炼GPS预测，在全球尺度下实现时间一致的逐帧视频地理定位。

**[VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding](video_understanding/virtuebench_evaluating_trustworthiness_under_uncertainty_in_long_video_understan.md)**

:   提出 VirtueBench，首个评估 VLM 在不确定性下可信度的长视频理解基准，通过为每个视频构建多级帧采样并标注可回答/不可回答的 ground truth，揭示了现有模型普遍倾向于猜测而非诚实拒绝的问题。

**[VRR-QA: Visual Relational Reasoning in Videos Beyond Explicit Cues](video_understanding/vrr-qa_visual_relational_reasoning_in_videos_beyond_explicit_cues.md)**

:   本文提出 VRR-QA 基准，包含 1K 精心标注的视频问答对，专门测试模型对视频中隐式视觉关系的推理能力（如屏幕外事件、跨帧因果、空间关系推断），揭示当前最强 VideoQA 模型（包括 GPT-O3）在隐式推理上的显著不足——最优模型仅达 64% 准确率，远低于人类的 83%。

**[VSI: Visual-Subtitle Integration for Keyframe Selection to Enhance Long Video Understanding](video_understanding/vsi_visual-subtitle_integration_for_keyframe_selection_to_enhance_long_video_un.md)**

:   VSI 提出双分支协作检索框架（视频搜索 + 字幕匹配），通过融合视觉和文本信息实现精确的关键帧定位，在文本相关任务中将搜索准确率从29.48提升至45.00，是首个跨模态关键帧检索方法。

**[Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding](video_understanding/wavelet-based_frame_selection_by_detecting_semantic_boundary_for_long_video_unde.md)**

:   提出 WFS-SB，一种免训练的帧选择框架，利用小波变换从查询-帧相似度信号中检测语义边界，将视频分割为语义连贯的片段后自适应分配帧预算并做多样性采样，在 VideoMME/MLVU/LongVideoBench 上大幅超越 SOTA。

---

## 🧑 人体理解 { #human_understanding }

**[A Two-Stage Dual-Modality Model for Facial Expression Recognition](human_understanding/a_two_stage_dual_modality_model_for_facial_expression_recognition.md)**

:   提出两阶段双模态面部表情识别框架：Stage I 通过填充感知增强和训练期 MoE 头在外部数据集上适配 DINOv2 编码器；Stage II 通过多尺度面部裁剪、Wav2Vec 2.0 音频特征提取和门控融合实现帧级音视觉表情分类，在 ABAW 2026 竞赛中取得 0.5368 Macro-F1。

**[All in One: Unifying Deepfake Detection, Tampering Localization, and Source Tracing with a Robust Landmark-Identity Watermark](human_understanding/all_in_one_unifying_deepfake_detection_tampering_localization_and_source_tracing.md)**

:   提出 LIDMark，首个将 deepfake 检测、篡改区域定位和源追踪统一到单一主动取证框架中的方法——通过嵌入 152 维 Landmark-Identity 水印（136D 面部关键点 + 16D 源 ID），利用内在/外在一致性实现三合一取证，PSNR/SSIM 和检测精度均超越现有方法。

**[AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video](human_understanding/avatar_reinforcement_learning_to_see_hear_and_reason_over_video.md)**

:   提出AVATAR框架，通过离线策略训练架构（分层回放缓冲区）和时间优势塑造(TAS)策略解决GRPO在多模态视频推理中的数据低效、优势消失和均匀信用分配三大问题，在音视频理解基准上显著超越标准GRPO（OmniBench +3.7，样本效率提升5倍）。

**[Beyond the Fold: Quantifying Split-Level Noise and the Case for Leave-One-Dataset-Out AU Evaluation](human_understanding/beyond_the_fold_quantifying_split-level_noise_and_the_case_for_leave-one-dataset.md)**

:   揭示面部AU检测中被试独立交叉验证本身引入±0.065 F1的随机噪声（noise floor），许多声称的SOTA提升落入此噪声带内不可区分，并提出Leave-One-Dataset-Out（LODO）协议作为更稳定可靠的替代评估方案。

**[BROTHER: Behavioral Recognition Optimized Through Heterogeneous Ensemble Regularization for Ambivalence and Hesitancy](human_understanding/brother_behavioral_recognition_optimized_through_heterogeneous_ensemble_regulari.md)**

:   提出一个高度正则化的多模态融合管线，通过视觉(SigLip2)、音频(HuBERT)、文本(F2LLM)及统计特征四模态的异质分类器委员会，结合带训练-验证差距惩罚的 PSO 硬投票集成，实现自然场景下矛盾与犹豫（A/H）行为的鲁棒视频级识别，在 ABAW10 测试集上取得 Macro F1 = 0.7465。

**[CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation](human_understanding/cigpose_causal_intervention_graph_neural_network_for_whole-body_pose_estimation.md)**

:   提出因果干预图姿态估计框架 CIGPose，通过结构因果模型识别视觉上下文混杂因素，利用预测不确定性定位受混杂影响的关键点并用学习得到的上下文无关规范嵌入替换，再经层次图神经网络建模骨骼解剖约束，在 COCO-WholeBody 上达到 67.0% AP 的新 SOTA。

**[COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](human_understanding/cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)**

:   提出 COG 框架，将跨视图对应关系建模为置信度感知的最优传输(OT)问题，通过预测逐点置信度作为传输边际约束来抑制非重叠区域和离群点，实现无监督条件下媲美有监督方法的单参考图像新物体6DoF位姿估计。

**[E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](human_understanding/e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)**

:   提出 E-3DPSM，一种基于事件相机的自我中心 3D 人体姿态状态机，将姿态估计建模为连续时间状态演化过程，通过双向 SSM 时序建模和可学习的卡尔曼式融合模块融合直接预测与增量预测，实现 80Hz 实时推理，MPJPE 降低 19%、时序稳定性提升 2.7 倍。

**[Editing Physiological Signals in Videos Using Latent Representations](human_understanding/editing_physiological_signals_in_videos_using_latent_representations.md)**

:   提出PhysioLatent框架，将输入面部视频编码到3D VAE潜空间，与目标心率CLIP文本嵌入融合，通过AdaLN增强的时空融合层捕捉rPPG时间相干性，结合FiLM调制解码器和微调输出层实现精确心率修改，在保持PSNR 38.96dB/SSIM 0.98的视觉质量下达到10 bpm MAE的心率调制精度。

**[Efficient Onboard Spacecraft Pose Estimation with Event Cameras and Neuromorphic Hardware](human_understanding/efficient_onboard_spacecraft_pose_estimation_with_event_cameras_and_neuromorphic_hardware.md)**

:   首次在 BrainChip Akida 神经形态硬件上实现端到端航天器 6-DoF 位姿估计，探索事件相机表示和量化感知训练在低功耗星载部署中的精度-效率权衡。

**[EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR](human_understanding/egoposeformer_v2_accurate_egocentric_human_motion_estimation_for_arvr.md)**

:   提出 EgoPoseFormer v2 (EPFv2)，通过端到端 Transformer 架构（单一全局查询 + 因果时序注意力 + 条件多视图交叉注意力）和基于不确定性蒸馏的自动标注系统，在 EgoBody3M 基准上以 0.8ms GPU 延迟实现了自我中心 3D 人体运动估计的 SOTA 精度（MPJPE 4.02cm，比前作提升 15-22%）。

**[Face Time Traveller: Travel Through Ages Without Losing Identity](human_understanding/face_time_traveller_travel_through_ages_without_losing_identity.md)**

:   提出 FaceTT 框架，通过面部属性感知提示词精炼、角度反演和自适应注意力控制三大模块，实现高保真、身份一致的人脸年龄变换，在多个基准上超越现有方法。

**[FlexAvatar: Learning Complete 3D Head Avatars with Partial Supervision](human_understanding/flexavatar_learning_complete_3d_head_avatars_with_partial_supervision.md)**

:   提出 FlexAvatar，通过引入可学习的"偏置吸收器"（bias sinks）token 统一单目和多视角数据训练，解决了驱动信号与目标视角的纠缠问题，从单张图像生成完整、高质量、可动画的 3D 头部化身。

**[A2P: From 2D Alignment to 3D Plausibility for Occlusion-Robust Two-Hand Reconstruction](human_understanding/from_2d_alignment_to_3d_plausibility_unifying_hete.md)**

:   解耦双手重建为 2D 结构对齐 + 3D 空间交互对齐：Stage 1 用 Fusion Alignment Encoder 隐式蒸馏 Sapiens 的关键点/分割/深度三种 2D 先验（推理时免基础模型，56fps），Stage 2 用穿透感知扩散模型 + 碰撞梯度引导将穿透姿态映射到物理合理配置——InterHand2.6M 上 MPJPE 降至 5.36mm（超 SOTA 4DHands 2.13mm），穿透体积降 7 倍。

**[FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation](human_understanding/fsmc-pose_frequency_and_spatial_fusion_with_multiscale_selfcalibration_for_cattle.md)**

:   FSMC-Pose 提出面向牛只爬跨(mounting)姿态估计的轻量级 top-down 框架，包含频率-空间融合骨干网络 CattleMountNet（通过 SFEBlock 的小波变换+高斯滤波分离前景-背景，RABlock 的多尺度扩张卷积聚合上下文）和多尺度自校准头 SC2Head（空间-通道共校准 + 自校准分支纠正结构偏移），同时构建了首个牛只爬跨数据集 MOUNT-Cattle，在复杂群养环境中以极低计算成本(4.41 GFLOPS, 2.698M 参数)达到 89% AP。

**[FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition](human_understanding/fusionagent_a_multimodal_agent_with_dynamic_model_selection_for_human_recognitio.md)**

:   本文提出 FusionAgent，一个基于多模态大语言模型（MLLM）的智能体框架，用于全身生物特征识别中的动态样本级模型选择——将每个专家模型（人脸识别/步态识别/行人重识别）封装为工具，通过强化微调（RFT）让 agent 学会根据每个测试样本的特征自适应选择最优模型组合，配合新提出的 ACT 分数融合策略，显著超越现有 SOTA 融合方法。

**[HandDreamer: Zero-Shot Text to 3D Hand Model Generation](human_understanding/handdreamer_zero_shot_text_to_3d_hand_model_generation.md)**

:   提出 HandDreamer，首个从文本提示零样本生成 3D 手部模型的方法，通过 MANO 初始化、骨架引导扩散和校正手形损失解决 SDS 中的视图不一致和几何畸变问题。

**[HandX: Scaling Bimanual Motion and Interaction Generation](human_understanding/handx_scaling_bimanual_motion_and_interaction_generation.md)**

:   构建了 HandX——一个统一的双手运动生成基础设施（包含 54.2 小时运动数据 + 48.5 万条细粒度文本标注），提出解耦式自动标注策略（运动学特征提取 + LLM 推理生成描述），并基准测试了扩散和自回归两种生成范式，展示了明确的数据和模型 scaling 趋势。

**[How to Take a Memorable Picture? Empowering Users with Actionable Feedback](human_understanding/how_to_take_a_memorable_picture_empowering_users_with_actionable_feedback.md)**

:   定义了记忆性反馈（MemFeed）新任务，提出 MemCoach——一种 training-free 的 MLLM 激活导向方法，通过教师-学生策略将记忆性感知知识注入模型激活空间，使 MLLM 能生成提升照片记忆性的自然语言可操作建议。

**[HUM4D: A Dataset and Evaluation for Complex 4D Markerless Human Motion Capture](human_understanding/hum4d_markerless_motion_capture.md)**

:   提出 HUM4D 数据集，包含复杂单人和多人运动场景（快速运动、遮挡、身份交换），提供同步多视角 RGB/RGB-D 序列、精确 Vicon 标记运动捕捉真值和 SMPL/SMPL-X 参数，基准测试揭示 SOTA 无标记方法在真实条件下的显著性能退化。

**[HumanOrbit: 3D Human Reconstruction as 360° Orbit Generation](human_understanding/humanorbit_3d_human_reconstruction_as_360_orbit_generation.md)**

:   将单图3D人体重建转化为360°轨道视频生成问题，用仅500个3D扫描数据LoRA微调视频扩散模型（Wan 2.1）生成81帧环绕视频，再通过VGGT+Mesh Carving重建高质量纹理网格，无需位姿标注且在多视图一致性和身份保持上超越现有方法。

**[IDperturb: Enhancing Variation in Synthetic Face Generation via Angular Perturbations](human_understanding/idperturb_enhancing_variation_in_synthetic_face_generation_via_angular_perturbat.md)**

:   提出 IDperturb，一种在单位超球面上对身份嵌入进行角度扰动的几何采样策略，无需修改生成模型即可显著增强合成人脸数据集的类内多样性，提升下游人脸识别性能。

**[LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference](human_understanding/lamogen_language_to_motion_generation_through_llm-guided_symbolic_inference.md)**

:   提出 LabanLite 符号动作表示和 LaMoGen 框架，首次让 LLM 通过可解释的 Laban 符号推理自主组合动作序列，在时序精度和可控性上超越传统文本-动作联合嵌入方法。

**[LaScA: Language-Conditioned Scalable Modelling of Affective Dynamics](human_understanding/lasca_language-conditioned_scalable_modelling_of_affective_dynamics.md)**

:   提出 LaScA 框架，利用大语言模型生成确定性语义词典为手工制作的面部和声学特征提供语义先验，通过冻结的句子编码器生成语义嵌入并与原始特征融合，在 Aff-Wild2 和 SEWA 数据集上的情感变化预测中一致性地超越纯特征基线，并在一致性、效率和可解释性上与端到端深度模型持平或更优。

**[LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction](human_understanding/laser_layer-wise_scale_alignment_for_training-free_streaming_4d_reconstruction.md)**

:   提出 LASER，一个无需重训练的框架，通过层级深度尺度对齐（Layer-wise Scale Alignment）将离线前馈重建模型（如 VGGT、π³）转换为流式系统，在 RTX A6000 上以 14 FPS、6GB 峰值显存实现千米级视频的实时流式 4D 重建。

**[LCA: Large-scale Codec Avatars - The Unreasonable Effectiveness of Large-scale Avatar Pretraining](human_understanding/lca_large-scale_codec_avatars_the_unreasonable_effectiveness_of_large-scale_avata.md)**

:   LCA 首次将大规模预训练/后训练范式应用于 3D 头像建模：在 100 万野外视频上预训练学习广泛的外观和几何先验，再在高质量多视图工作室数据上后训练增强精细表情和保真度，打破了泛化性与保真度的固有矛盾。

**[MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision](human_understanding/matched_crisp_edge_detection_using_end-to-end_matching-based_supervision.md)**

:   MatchED 提出一种轻量（约21K参数）plug-and-play 模块，通过在训练时对预测边缘和 GT 边缘进行基于空间距离+置信度的 one-to-one 二部匹配来生成 crisp（单像素宽）边缘图，可附加到任何边缘检测器端到端训练，首次在不依赖 NMS+thinning 后处理的情况下匹配或超越标准后处理方法。

**[Miburi: Towards Expressive Interactive Gesture Synthesis](human_understanding/miburi_towards_expressive_interactive_gesture_synthesis.md)**

:   提出 Miburi，首个在线因果框架，通过直接利用语音-文本大模型 Moshi 的内部 token 流和二维因果 Transformer，实现实时同步的全身手势与面部表情生成。

**[MMGait: Towards Multi-Modal Gait Recognition](human_understanding/mmgait_multi_modal_gait_recognition.md)**

:   MMGait 构建了目前最全面的多模态步态识别基准数据集（5 种传感器、12 种模态、725 人、334K 序列），并提出全模态步态识别新任务和统一基线模型 OmniGait。

**[Mobile-VTON: High-Fidelity On-Device Virtual Try-On](human_understanding/mobile_vton_ondevice_virtual_tryon.md)**

:   首个全离线移动端扩散式虚拟试穿框架，基于TeacherNet-GarmentNet-TryonNet (TGT)架构，通过特征引导对抗蒸馏(FGA)将SD3.5 Large的能力迁移到415M参数的轻量学生网络，在VITON-HD和DressCode上以1024×768分辨率匹配甚至超越服务器端基线，端到端推理时间约80秒（小米17 Pro Max）。

**[MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation](human_understanding/molingo_motion-language_alignment_for_text-to-motion_generation.md)**

:   MoLingo 通过语义对齐的运动自编码器（SAE）和多 token 交叉注意力文本条件注入，在连续潜空间上执行 masked 自回归 rectified flow，在文本到人体动作生成任务上取得了 FID、R-Precision 和用户研究的全面 SOTA。

**[OMG-Bench: A New Challenging Benchmark for Skeleton-based Online Micro Hand Gesture Recognition](human_understanding/omg-bench_a_new_challenging_benchmark_for_skeleton-based_online_micro_hand_gestu.md)**

:   本文构建了首个大规模公开的基于骨骼数据的在线微手势识别基准OMG-Bench（40类、13948个实例），并提出HMATr框架，通过层次化记忆库和位置感知查询实现检测-分类的端到端统一，在检测率上超越SOTA方法7.6%。

**[OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery](human_understanding/onlinehmr_video-based_online_world-grounded_human_mesh_recovery.md)**

:   提出 OnlineHMR，首个同时满足系统因果性、忠实性、时序一致性和高效性四项准则的在线世界坐标人体网格恢复框架，通过滑动窗口因果学习 + KV 缓存推理实现流式相机坐标 HMR，结合以人为中心的增量 SLAM 和 EMA 轨迹校正实现在线全局定位。

**[OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis](human_understanding/openfs_multi-hand-capable_fingerspelling_recognition_with_implicit_signing-hand_.md)**

:   提出 OpenFS 框架，通过双层位置编码 + 签名手聚焦损失 + 单调对齐损失实现隐式签名手检测的多手指拼识别，并设计帧级字母条件扩散生成器合成 OOV 数据，在 ChicagoFSWild/ChicagoFSWildPlus/FSNeo 三个基准上取得 SOTA，推理速度比 PoseNet 快 100 倍以上。

**[ParTY: Part-Guidance for Expressive Text-to-Motion Synthesis](human_understanding/party_part-guidance_for_expressive_text-to-motion_synthesis.md)**

:   提出 ParTY 框架，通过部位引导网络（Part-Guided Network）和部位感知文本对齐（Part-aware Text Grounding），在保持全身动作连贯性的同时大幅提升身体各部位的文本-动作语义对齐精度，解决了现有整体式方法与部位拆分方法之间"部位表达力 vs 全身连贯性"的根本矛盾。

**[PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement](human_understanding/phase-net_physics-grounded_harmonic_attention_system_for_efficient_remote_photop.md)**

:   从Navier-Stokes方程出发，通过严格数学推导揭示rPPG脉搏信号遵循二阶阻尼谐振子模型，其离散解形式等价于因果卷积算子，从而为TCN架构的选择提供了第一性原理依据，设计出仅0.29M参数的PHASE-Net在多个数据集上达到SOTA。

**[RAM: Recover Any 3D Human Motion in-the-Wild](human_understanding/ram_recover_any_3d_human_motion_in-the-wild.md)**

:   RAM 提出统一的多人 3D 运动恢复框架，集成运动感知语义跟踪器 SegFollow（基于 SAM2 + 自适应卡尔曼滤波）、记忆增强的时序人体网格恢复模块 T-HMR、轻量运动预测器和门控组合器，在 PoseTrack 和 3DPW 等基准上实现零样本跟踪稳定性和 3D 精度的 SOTA，且推理速度比之前方法快 2-3 倍。

**[Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](human_understanding/reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)**

:   提出 VTON-IQA，一个无需参考图的虚拟试穿图像质量评估框架，通过构建 62,688 张试穿图像 × 431,800 条人工标注的大规模基准 VTON-QBench，以及交错式交叉注意力（ICA）模块建模服装-人物-试穿图之间的交互关系，实现与人类感知高度对齐的图像级质量预测。

**[RefTon: Reference Person Shot Assist Virtual Try-on](human_understanding/refton_reference_person_shot_assist_virtual_try-on.md)**

:   本文提出 RefTon，一个基于 Flux-Kontext 的人对人虚拟试穿框架，通过引入额外参考图像（其他人穿着目标服装的照片）来提供更准确的服装细节信息，同时通过两阶段训练策略和缩放位置索引机制实现了无需辅助条件（如 DensePose、分割掩码）的端到端试穿，在 VITON-HD 和 DressCode 上达到 SOTA。

**[RegFormer: Transferable Relational Grounding for Efficient Weakly-Supervised HOI Detection](human_understanding/regformer_transferable_relational_grounding_for_weakly-supervised_hoi_detection.md)**

:   RegFormer 提出一个轻量级关系接地 Transformer 模块，在仅图像级标注的弱监督下，通过空间接地查询和交互性感知学习，直接从图像级推理迁移到实例级 HOI 检测，无需额外训练，性能接近全监督方法。

**[ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data](human_understanding/remogen_real-time_human_interaction-to-reaction_generation_via_modular_learning_.md)**

:   提出 ReMoGen，一个模块化框架用于实时人体交互-到-反应的动作生成：利用大规模单人运动数据学习通用运动先验（冻结），通过独立训练的 Meta-Interaction 模块适配不同交互域（人-人/人-场景），并引入 Frame-wise Segment Refinement 实现逐帧低延迟在线更新（0.047s/帧），在 Inter-X 和 LINGO 数据集上全面超越 SOTA。

**[rPPG-VQA: A Video Quality Assessment Framework for Unsupervised rPPG Training](human_understanding/rppg_vqa_video_quality_assessment.md)**

:   rPPG-VQA 提出首个面向远程心率检测（rPPG）的视频质量评估框架，结合信号级多方法共识 SNR 和场景级 MLLM 干扰识别，配合两阶段自适应采样策略筛选野外视频构建训练集。

**[Seeing without Pixels: Perception from Camera Trajectories](human_understanding/seeing_without_pixels_perception_from_camera_trajectories.md)**

:   本文首次系统性地将相机位姿轨迹（6DoF pose sequence）提升为一种独立的视频感知模态，通过对比学习框架训练轻量级 Transformer 编码器 CamFormer，将相机轨迹映射到与文本对齐的联合嵌入空间，在 5 个数据集的 10 个下游任务上证明相机轨迹是既轻量又鲁棒的视频内容信号——在物理活动上甚至可以超越计算量大数千倍的视频模型。

**[Sketch2Colab: Sketch-Conditioned Multi-Human Animation via Controllable Flow Distillation](human_understanding/sketch2colab_sketch-conditioned_multi-human_animation_via_controllable_flow_dist.md)**

:   提出 Sketch2Colab，通过将草图驱动的扩散先验蒸馏为整流流学生网络，结合能量引导和连续时间马尔可夫链（CTMC）离散事件规划，从故事板草图生成协调的多人-物体交互 3D 动作，在 CORE4D 和 InterHuman 上实现 SOTA 约束遵从度和感知质量。

**[Stake the Points: Structure-Faithful Instance Unlearning](human_understanding/stake_the_points_structure-faithful_instance_unlearning.md)**

:   提出 Structguard，通过语义锚点（semantic anchors）保持遗忘过程中保留实例间的语义关系结构，避免结构性崩塌，在图像分类/人脸识别/检索三任务上平均提升 32.9%/19.3%/22.5%。

**[Talking Together: Synthesizing Co-Located 3D Conversations from Audio](human_understanding/talking_together_synthesizing_co-located_3d_conversations_from_audio.md)**

:   首次提出从单一混合音频流生成两个**共处同一3D空间**的对话参与者完整面部动画的方法，通过双流扩散架构（共享 U-Net + 跨注意力）、两阶段混合数据训练策略、LLM 驱动的文本-空间布局控制以及辅助眼神损失，实现自然的互视、转头和空间感知的双人对话3D动画合成。

**[Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](human_understanding/team_leya_in_10th_abaw_competition_multimodal_ambi.md)**

:   提出四模态（场景 VideoMAE + 人脸 EfficientNetB0 + 音频 Wav2Vec2.0/Mamba + 文本 EmotionDistilRoBERTa）融合管线，通过原型增强 Transformer 融合模块将各模态嵌入投影到共享 128 维空间并以原型分类辅助损失正则化，在 BAH 语料的最终测试集上以 5 模型集成达到 **71.43% Macro F1**，显著超越所有单模态基线。

**[TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](human_understanding/tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)**

:   TeHOR 利用文本描述作为语义引导，通过预训练扩散模型的 Score Distillation Sampling 联合优化 3D 人体和物体的几何与纹理，突破了传统方法对接触信息的依赖，实现了包括非接触交互在内的准确且语义一致的 3D 重建。

**[4DSurf: High-Fidelity Dynamic Scene Surface Reconstruction](human_understanding/textit4dsurf_high-fidelity_dynamic_scene_surface_reconstruction.md)**

:   本文提出 4DSurf，一个基于2D高斯泼溅的通用动态场景表面重建框架，通过引入高斯运动诱导的SDF流正则化来约束表面时序一致演化，并采用重叠分段策略处理大变形，在 Hi4D 和 CMU Panoptic 数据集上分别以 49% 和 19% 的 Chamfer 距离改进超越现有 SOTA。

**[TriLite: Efficient WSOL with Universal Visual Features and Tri-Region Disentanglement](human_understanding/trilite_efficient_weakly_supervised_object_localization_with_universal_visual_fe.md)**

:   仅使用冻结 DINOv2 ViT + 不到 800K 可训练参数的 TriHead 模块，通过将 patch 特征解耦为前景/背景/模糊三区域并引入对抗性背景损失，在 WSOL 上以极少参数刷新 SOTA。

**[UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos](human_understanding/unidex_a_robot_foundation_suite_for_universal_dexterous_hand_control_from_egocen.md)**

:   提出UniDex机器人基础套件——包含跨8种灵巧手的大规模数据集（50K+轨迹/9M帧）、功能-执行器对齐的统一动作空间（FAAS）和3D VLA策略（UniDex-VLA），在真实世界工具使用任务上达到81%平均任务进度（vs π₀的38%），并展示了空间、物体和零样本跨手泛化能力。

**[UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking](human_understanding/unils_end-to-end_audio-driven_avatars_for_unified_listening_and_speaking.md)**

:   提出首个端到端统一说话-倾听面部表情生成框架UniLS，通过两阶段训练范式（先学内在运动先验、再用双轨音频微调），仅需双方音频输入即可同时生成自然的说话和倾听面部动作，倾听指标提升高达44.1%。

**[Unleashing Vision-Language Semantics for Deepfake Video Detection](human_understanding/unleashing_vision-language_semantics_for_deepfake_video_detection.md)**

:   提出VLAForge，通过ForgePerceiver独立学习多样的伪造线索和伪造定位图，并结合身份感知的视觉-语言对齐（VLA）评分机制，释放VLM跨模态语义的潜力来增强深度伪造视频检测的判别能力，在9个数据集上全面超越现有SOTA。

**[ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body](human_understanding/vibes_a_conversational_agent_with_behaviorally_intelligent_3d_virtual_body.md)**

:   提出 ViBES，一个统一语言、语音和身体动作的 3D 对话代理，通过模态专家混合（MoME）架构和跨模态注意力机制，在保留预训练语音 LLM 对话能力的同时生成时间对齐的面部表情和全身动作，超越了将行为视为简单"模态翻译"的范式。

**[Vision-Language Attribute Disentanglement and Reinforcement for Lifelong Person Re-Identification](human_understanding/vision-language_attribute_disentanglement_and_reinforcement_for_lifelong_person_.md)**

:   VLADR 提出利用视觉-语言模型（VLM）中的细粒度属性知识来增强终身行人重识别，通过多粒度文本属性解耦（MTAD）和跨域跨模态属性强化（ICAR）两阶段训练，显式建模跨域共享的人体属性以实现高效知识转移和遗忘缓解，在抗遗忘和泛化能力上分别超越 SOTA 1.9%-2.2% 和 2.1%-2.5%。

**[WildCap: Facial Albedo Capture in the Wild via Hybrid Inverse Rendering](human_understanding/wildcap_facial_albedo_capture_in_the_wild_via_hybrid_inverse_rendering.md)**

:   提出 WildCap，通过混合逆渲染框架（数据驱动 SwitchLight 去光照 + 基于模型的 texel grid lighting 优化 + 扩散先验采样），从手机野外视频中重建高质量 4K 面部漫反射 albedo 贴图，大幅缩小野外捕捉与受控光照方法之间的质量差距。

---

## 🎬 视频生成 { #video_generation }

**[ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos](video_generation/activityforensics_a_comprehensive_benchmark_for_localizing_manipulated_activity_.md)**

:   首次提出活动级视频伪造定位任务和ActivityForensics大规模基准数据集（6K+伪造片段），通过grounding辅助的自动化数据构造管线制造高度逼真的活动篡改，并提出Temporal Artifact Diffuser (TADiff)基线方法，通过扩散式特征正则化放大伪造线索。

**[Anti-I2V: Safeguarding your photos from malicious image-to-video generation](video_generation/anti-i2v_safeguarding_your_photos_from_malicious_image-to-video_generation.md)**

:   Anti-I2V 提出了一种针对恶意图像到视频生成的防御方法，通过在 L\*a\*b\* 和频域双空间优化扰动，并设计内部表示崩塌（IRC）和锚定（IRA）损失破坏去噪网络的语义特征传播，在 CogVideoX、DynamiCrafter 和 Open-Sora 三种不同架构上实现 SOTA 防护效果。

**[AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation](video_generation/autocut_end-to-end_advertisement_video_editing_based_on_multimodal_discretizatio.md)**

:   AutoCut 提出了一个端到端的广告视频编辑框架，通过残差向量量化（RQVAE）将视频、音频和文本统一到共享的离散 token 空间中，在 Qwen3-8B 上进行多模态对齐和监督微调，实现了视频选择、排序、脚本生成和背景音乐选择四项任务的统一处理，在多项指标上超越 GPT-4o 基线。

**[Chain of Event-Centric Causal Thought for Physically Plausible Video Generation](video_generation/chain_of_event-centric_causal_thought_for_physically_plausible_video_generation.md)**

:   将物理合理性视频生成(PPVG)建模为因果连接的事件序列，通过物理公式驱动的事件链推理将复杂物理现象分解为有序事件，再通过过渡感知的跨模态提示生成语义-视觉双重条件，引导视频扩散模型生成遵循因果演进的物理现象视频。

**[Compressed-Domain-Aware Online Video Super-Resolution](video_generation/compressed-domain-aware_online_video_super-resolution.md)**

:   CDA-VSR 提出利用视频压缩域信息（运动矢量、残差图、帧类型）指导在线视频超分辨率的三个关键环节：运动矢量引导的可变形对齐实现高效精准配准、残差图门控融合抑制错配区域、帧类型感知重建自适应分配计算资源，在 REDS4 上以 93 FPS（>2倍于SOTA速度）达到最优 PSNR。

**[CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video](video_generation/cubecomposer_spatio-temporal_autoregressive_4k_360_video_generation_from_perspec.md)**

:   提出 CubeComposer，将360°视频分解为 cubemap 六面表示并按时空自回归方式逐面生成，首次实现从透视视频原生生成4K（3840×1920）分辨率的360°全景视频，无需后处理超分辨率。

**[Diff4Splat: Repurposing Video Diffusion Models for Dynamic Scene Generation](video_generation/diff4splat_controllable_4d_scene_generation_with_latent_dynamic_reconstruction_m.md)**

:   提出 Diff4Splat，一个前馈式框架，将视频扩散模型与可变形3D高斯场统一到端到端可训练的模型中，从单张图像在约30秒内直接生成动态4D场景表示，比优化方法快60倍。

**[DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching](video_generation/disca_accelerating_video_diffusion_transformers_wi.md)**

:   DisCa 首次将可学习特征缓存与步蒸馏统一为兼容框架，用轻量神经预测器（<4% 模型参数）替代手工缓存策略，配合 Restricted MeanFlow 稳定大规模视频 DiT 蒸馏，在 HunyuanVideo 上实现 11.8× 近无损加速。

**[DreamShot: Personalized Storyboard Synthesis with Video Diffusion Prior](video_generation/dreamshot_storyboard_synthesis.md)**

:   提出 DreamShot，利用视频扩散模型的时空先验来生成人物一致、场景连贯的多镜头故事板，通过 Role-Attention Consistency Loss 解决多角色混淆问题，统一支持文本到镜头、参考到镜头和镜头到镜头三种模式。

**[DriveLaW: Unifying Planning and Video Generation in a Latent Driving World](video_generation/drivelaw_unifying_planning_and_video_generation_in_a_latent_driving_world.md)**

:   提出 DriveLaW，一个通过共享潜在空间将视频生成与运动规划统一的驾驶世界模型，将视频生成器的中间潜在特征直接注入扩散规划器，在 nuScenes 视频预测和 NAVSIM 规划基准上同时达到 SOTA。

**[FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters](video_generation/fastlightgen_fast_and_light_video_generation_with_fewer_steps_and_parameters.md)**

:   FastLightGen 提出三阶段蒸馏算法，首次实现采样步数与模型大小的联合蒸馏，通过识别冗余层、动态概率剪枝和 well-guided teacher guidance 分布匹配，将 HunyuanVideo/WanX 压缩为 4 步 30% 参数剪枝的轻量生成器，实现约 35 倍加速且性能超越教师模型。

**[First Frame Is the Place to Go for Video Content Customization](video_generation/first_frame_is_the_place_to_go_for_video_content_customization.md)**

:   发现视频生成模型将第一帧隐式地当作「概念记忆缓冲区」来存储和复用多个视觉实体的内在能力，提出 FFGo——仅用 20-50 个训练样本的轻量级 LoRA 适配方法，无需修改架构即可激活这一能力，实现多参考物体的视频内容定制，在用户研究中 81.2% 的情况下被评为最佳。

**[FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](video_generation/flashmotion_fewstep_controllable_video_generation.md)**

:   提出 FlashMotion，首个实现少步（4步）轨迹可控视频生成的三阶段训练框架，通过训练轨迹适配器→蒸馏快速生成器→混合对抗+扩散微调适配器的策略，在 4 步推理下同时超越现有多步方法的视觉质量和轨迹精度，实现 47 倍加速。

**[Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction](video_generation/free-lunch_long_video_generation_via_layer-adaptive_ood_correction.md)**

:   FreeLOC 提出一种免训练的层自适应框架，通过识别视频DiT中各层对"帧级相对位置OOD"和"上下文长度OOD"两种分布外问题的敏感度差异，选择性地在敏感层应用多粒度位置重编码(VRPR)和分层稀疏注意力(TSA)，在不增加训练成本的情况下实现SOTA的长视频生成质量。

**[From Static to Dynamic: Exploring Self-supervised Image-to-Video Representation Transfer Learning](video_generation/from_static_to_dynamic_exploring_self-supervised_image-to-video_representation_t.md)**

:   本文提出 Co-Settle 框架，通过在冻结的图像预训练编码器上训练一个轻量线性投影层，利用时间循环一致性损失和语义可分性约束，仅需5个epoch的自监督训练即可在8个图像基础模型上一致性提升多粒度视频下游任务性能。

**[Generative Neural Video Compression via Video Diffusion Prior](video_generation/generative_neural_video_compression_via_video_diffusion_prior.md)**

:   本文提出 GNVC-VD，首个基于 DiT 的生成式神经视频压缩框架，通过将视频扩散变换器作为视频原生生成先验，在统一编解码器中实现时空潜在压缩和序列级生成精炼，在极低码率（<0.03 bpp）下大幅超越传统和学习型编解码器的感知质量，并显著减少先前生成方法中的闪烁伪影。

**[Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context](video_generation/geometry-as-context_modulating_explicit_3d_in_scene-consistent_video_generation_.md)**

:   提出 Geometry-as-Context (GaC) 框架，将基于重建的场景视频生成中的不可微算子（3D重建+渲染）替换为统一的自回归视频生成模型，通过将几何信息（深度图）作为交错上下文嵌入生成序列，实现端到端训练并缓解累积误差。

**[Gloria: Consistent Character Video Generation via Content Anchors](video_generation/gloria_consistent_character_video_generation_via_content_anchors.md)**

:   Gloria 提出用一组紧凑的"内容锚帧"（Content Anchors）表征角色的多视角外观和表情身份，通过超集内容锚定（防止复制粘贴）和 RoPE 弱条件（区分多锚帧）两个机制，实现超过 10 分钟的长时一致角色视频生成。

**[Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning](video_generation/goal-driven_reward_by_video_diffusion_models_for_reinforcement_learning.md)**

:   提出 GenReward 框架，利用预训练视频扩散模型生成目标条件视频，通过视频级和帧级两层目标驱动奖励信号引导强化学习智能体，无需手工设计奖励函数即可在 Meta-World 机器人操控任务上显著超越基线。

**[Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization](video_generation/identity-preserving_image-to-video_generation_via_reward-guided_optimization.md)**

:   本文提出 IPRO，通过强化学习和可微分人脸身份评分器直接优化视频扩散模型，在不修改模型架构的情况下显著提升图像到视频生成中的人脸身份一致性，在 Wan 2.2 上实现了 20%-45% 的 FaceSim 提升。

**[Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout](video_generation/infinity-rope_action-controllable_infinite_video_generation_emerges_from_autoreg.md)**

:   提出 ∞-RoPE，一个训练免调的推理时框架，通过 Block-Relativistic RoPE、KV Flush 和 RoPE Cut 三个组件，将仅在5秒视频上训练的自回归视频扩散模型扩展为支持无限时长生成、精细动作控制和电影级场景切换的系统。

**[I'm a Map! Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers](video_generation/interpretable_motion-attentive_maps_spatio-temporally_localizing_concepts_in_vid.md)**

:   提出IMAP(可解释运动注意力图)，通过GramCol空间定位和运动头选择时序定位两个无训练模块，从Video DiT中提取运动概念的时空显著性图，在运动定位和零样本视频语义分割上超越现有方法。

**[LAMP: Language-Assisted Motion Planning for Controllable Video Generation](video_generation/lamp_language-assisted_motion_planning_for_controllable_video_generation.md)**

:   提出LAMP框架，将运动控制建模为语言到程序合成问题：设计电影摄影启发的运动DSL，训练LLM将自然语言描述转化为结构化运动程序，再确定性映射为3D对象和相机轨迹来条件化视频生成，首次实现从自然语言同时生成对象和相机运动。

**[Let Your Image Move with Your Motion! – Implicit Multi-Object Multi-Motion Transfer](video_generation/let_your_image_move_with_your_motion_--_implicit_multi-object_multi-motion_trans.md)**

:   本文提出 FlexiMMT，首个支持隐式多目标多运动迁移的 I2V 框架，通过运动解耦掩码注意力机制（MDMA）约束 motion/text token 仅影响对应目标区域、差异化掩码提取机制（DMEM）从扩散注意力中推导目标掩码并渐进传播，实现了精确的组合式多目标运动迁移。

**[Lighting-grounded Video Generation with Renderer-based Agent Reasoning](video_generation/lighting-grounded_video_generation_with_renderer-based_agent_reasoning.md)**

:   LiVER 提出了一种光照驱动的视频生成框架，通过渲染器Agent将文本描述转化为显式3D场景代理（包含布局、光照、相机轨迹），再利用物理渲染生成diffuse/glossy/rough GGX的场景proxy，注入视频扩散模型实现物理准确的光照效果与精确场景控制。

**[LightMover: Generative Light Movement with Color and Intensity Controls](video_generation/lightmover_generative_light_movement_with_color_and_intensity_controls.md)**

:   LightMover 利用视频扩散先验，将光源编辑建模为序列到序列预测问题，通过统一的控制token表示实现光源位置、颜色和亮度的精确操控，并提出自适应token剪枝机制将控制序列长度减少41%，在光源移动和物体移动任务上均超越现有方法。

**[LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation](video_generation/linvideo_a_post-training_framework_towards_on_attention_in_efficient_video_gener.md)**

:   提出 LinVideo，一种无需训练数据的后训练框架，通过选择性地将视频扩散模型中的二次注意力替换为线性注意力，实现 1.43–1.71× 加速，结合蒸馏可达 15.9–20.9× 加速，同时保持生成质量。

**[MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer](video_generation/moviedrive_urban_scene_synthesis_with_multi-modal_multi-view_video_diffusion_tra.md)**

:   MoVieDrive 提出统一的多模态多视图视频扩散 Transformer，通过 modal-shared + modal-specific 的双层架构设计，在单一模型中同时生成 RGB 视频、深度图和语义图，配合多样的条件输入（文本、布局、上下文参考），在 nuScenes 上取得 FVD 46.8（SOTA），同时实现跨模态一致的高质量驾驶场景合成。

**[NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos](video_generation/neoverse_enhancing_4d_world_model_with_in-the-wild_monocular_videos.md)**

:   NeoVerse 提出了一个可扩展的 4D 世界模型，通过前馈式无位姿 4DGS 重建和在线单目退化模拟，使整个训练流程可以利用海量野外单目视频（百万级），在 4D 重建和新轨迹视频生成上均达到 SOTA。

**[NOVA: Sparse Control, Dense Synthesis for Pair-Free Video Editing](video_generation/nova_sparse_control_dense_synthesis_for_pair-free_video_editing.md)**

:   提出 NOVA，首次形式化"稀疏控制、密集合成"范式用于视频编辑：稀疏分支从用户编辑的多关键帧提供语义引导，密集分支从原始视频注入运动和纹理信息；配合退化模拟训练策略实现无需配对数据的学习，在编辑保真度、运动保持和时序一致性上全面超越现有方法。

**[Towards Realistic and Consistent Orbital Video Generation via 3D Foundation Priors](video_generation/orbital_video_3d_foundation_priors.md)**

:   提出利用 3D 基础生成模型（Hunyuan3D）的潜在特征作为形状先验，通过多尺度 3D 适配器注入基础视频扩散模型，实现从单张图像生成几何真实且视图一致的轨道视频。

**[PAM: A Pose-Appearance-Motion Engine for Sim-to-Real HOI Video Generation](video_generation/pam_a_pose-appearance-motion_engine_for_sim-to-real_hoi_video_generation.md)**

:   提出PAM——首个仅需初始/目标手部姿态和物体几何即可生成逼真手物交互视频的引擎，通过解耦姿态生成、外观生成和运动生成三阶段，在DexYCB上FVD 29.13（vs InterDyn 38.83）、MPJPE 19.37mm（vs CosHand 30.05mm），生成的合成数据还能有效增强下游手部姿态估计任务。

**[PerformRecast: Expression and Head Pose Disentanglement for Portrait Video Editing](video_generation/performrecast_expression_and_head_pose_disentanglement_for_portrait_video_editin.md)**

:   PerformRecast 提出了一种基于改进 3DMM 关键点变换公式的 GAN 人像视频编辑方法，通过将表情形变加在头部旋转之前（与 FLAME 模型一致）实现表情与头部姿态的精确解耦，并引入边界对齐模块解决面部/非面部区域的拼接错位问题，在表情替换和表情增强两种模式下均显著优于现有方法。

**[Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics](video_generation/phantom_physics-infused_video_generation_via_joint_modeling_of_visual_and_latent.md)**

:   提出Phantom框架，在预训练视频扩散模型（Wan2.2-TI2V）之上增加一个物理动力学分支，利用V-JEPA2提取的物理感知嵌入作为潜在物理状态，通过双向交叉注意力联合建模视觉内容和物理动力学演化，在物理一致性基准上大幅超越基线（VideoPhy PC提升50.4%），同时保持视觉质量。

**[Physical Simulator In-the-Loop Video Generation](video_generation/physical_simulator_in-the-loop_video_generation.md)**

:   提出PSIVG——首个将物理模拟器嵌入视频扩散生成循环的训练-free推理时框架：从模板视频中重建4D场景和物体网格，在MPM模拟器中生成物理一致轨迹，用光流引导视频生成，并通过TTCO测试时优化保证运动物体纹理一致性，用户偏好率达82.3%。

**[PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](video_generation/posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)**

:   PoseGen 通过 in-context LoRA 微调实现双重条件注入（token级外观 + 通道级姿态），并提出分段交错生成策略（KV共享+姿态感知插帧），仅用33小时视频数据即可生成高保真长时人体视频。

**[Rethinking Position Embedding as a Context Controller for Multi-Reference and Multi-Shot Video Generation](video_generation/rethinking_position_embedding_as_a_context_controller_for_multi-reference_and_mu.md)**

:   提出 PoCo（Position Embedding as Context Controller），通过在 RoPE 中引入额外的 SideInfo 轴编码参考实体信息，解决多参考多镜头视频生成中的"参考混淆"问题——当参考图像外观高度相似时模型无法正确关联镜头与参考。在 VACE-Wan2.1-14B 框架上实现 SOTA 的跨镜头一致性（CrossShot-FaceSim 89.35，CrossShot-DINO 92.66）。

**[SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](video_generation/seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)**

:   提出 SeeU，一个 2D→4D→2D 的学习框架：从稀疏单目 2D 帧重建 4D 世界表示，在低秩表示上学习连续且物理一致的 4D 动力学（B 样条参数化 + 物理约束），最后将 4D 世界重投影回 2D 并用时空上下文感知的视频生成器补全未知区域，实现跨时间和空间的未见视觉内容生成。

**[Semantic Satellite Communications for Synchronized Audiovisual Reconstruction](video_generation/semantic_satellite_communications_for_synchronized_audiovisual_reconstruction.md)**

:   提出面向卫星通信的自适应多模态语义传输系统，通过双流生成架构（视频驱动音频 / 音频驱动视频）实现动态模态优先级切换，结合知识库动态更新机制和 LLM 智能决策模块，在严苛带宽约束下实现高保真音视频同步重建。

**[SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation](video_generation/slvmeval_synthetic_meta_evaluation_benchmark_for_text-to-long_video_generation.md)**

:   提出SLVMEval元评估基准，通过从密集视频描述数据集合成受控退化的"高质量vs低质量"视频对（最长约3小时），测试现有T2V评估系统识别长视频质量差异的能力，发现人类在10个维度上达84.7%-96.8%准确率，而现有自动评估系统在9/10维度上落后于人类。

**[StreamDiT: Real-Time Streaming Text-to-Video Generation](video_generation/streamdit_real-time_streaming_text-to-video_generation.md)**

:   StreamDiT 提出了一套完整的流式视频生成方案（包括训练、建模和蒸馏），通过在 Flow Matching 中引入带渐进去噪的移动缓冲区和混合分区训练策略，结合时变 DiT 架构和窗口注意力，以及定制化的多步蒸馏方法，使 4B 参数模型在单 GPU 上达到 512p@16FPS 的实时流式视频生成。

**[SWIFT: Sliding Window Reconstruction for Few-Shot Training-Free Generated Video Attribution](video_generation/swift_sliding_window_reconstruction_for_few-shot_training-free_generated_video_a.md)**

:   SWIFT 首次定义了"少样本免训练生成视频溯源"任务，利用 3D VAE 中"多帧像素↔单帧潜变量"的时间映射特性，通过固定长度滑动窗口执行正常和损坏两次重建，用重叠帧的损失比值作为溯源信号，仅需 20 个样本即可达到 90%+ 平均溯源准确率，5 模型平均 94%。

**[SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](video_generation/switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)**

:   提出 SwitchCraft，一个无需训练的多事件视频生成框架，通过 Event-Aligned Query Steering (EAQS) 将帧级注意力对齐到对应事件提示、Auto-Balance Strength Solver (ABSS) 自适应平衡引导强度，在不修改模型权重的情况下实现多事件视频的清晰时序切换和场景一致性。

**[SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation](video_generation/symphomotion_joint_control_of_camera_motion_and_object_dynamics_for_coherent_vid.md)**

:   提出 SymphoMotion 统一运动控制框架，通过相机轨迹控制（CTC）和物体动态控制（ODC）两个机制同时精确控制视频中的相机运动和物体3D轨迹，并构建了25K规模的真实世界联合标注数据集 RealCOD-25K。

**[TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models](video_generation/tear_temporal-aware_automated_red-teaming_for_text-to-video_models.md)**

:   提出 TEAR，首个针对 T2V 模型时序维度漏洞的自动化红队测试框架，通过两阶段优化的时序感知测试生成器和迭代精炼模型，生成文本上无害但能利用时序动态触发有害视频的提示，在开源和商业 T2V 模型上达到 80%+ 的攻击成功率。

**[The Devil is in the Details: Enhancing Video Virtual Try-On via Keyframe-Driven Details Injection](video_generation/the_devil_is_in_the_details_enhancing_video_virtual_try-on_via_keyframe-driven_d.md)**

:   提出 KeyTailor 框架，通过关键帧驱动的细节注入策略（服装动态增强 + 协同背景优化）在不修改 DiT 架构的前提下，大幅提升视频虚拟试穿的服装保真度与背景一致性，同时发布 15K 高清数据集 ViT-HD。

**[Training-free Motion Factorization for Compositional Video Generation](video_generation/training-free_motion_factorization_for_compositional_video_generation.md)**

:   提出一个运动分解框架，将场景中多实例的运动分解为静止、刚体运动和非刚体运动三类，通过结构化运动图推理（SMR）解决 prompt 的语义歧义，通过解耦运动引导（DMG）在扩散过程中针对性地调控三类运动的生成，无需额外训练即可在 VideoCrafter-v2.0 和 CogVideoX-2B 上显著提升运动多样性和保真度。

**[U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](video_generation/u-mind_a_unified_framework_for_real-time_multimodal_interaction_with_audiovisual.md)**

:   提出 U-Mind，首个统一实时全栈多模态交互系统，支持高层推理对话和指令跟随，在单一交互循环中联合生成文本、语音、动作，并渲染为逼真视频，通过排练驱动学习和文本优先解码策略兼顾推理保持与跨模态对齐。

**[UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions](video_generation/uniavgen_unified_audio_and_video_generation_with_asymmetric_cross-modal_interact.md)**

:   UniAVGen 提出了一个基于对称双分支 DiT 的音视频联合生成框架，通过**非对称跨模态交互机制**和**人脸感知调制模块**实现精确的时空同步，仅用 1.3M 训练样本就在唇音同步、音色一致性和情感一致性上全面超越使用 30M 数据的竞品。

**[UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation](video_generation/unitalking_a_unified_audio-video_framework_for_talking_portrait_generation.md)**

:   提出 UniTalking，一个基于 MM-DiT 的端到端说话人肖像生成框架，通过双流对称架构中的联合注意力机制显式建模音视频 token 的细粒度时序对应关系，实现 SOTA 的唇音同步精度，同时支持个性化语音克隆。

**[Vanast: Virtual Try-On with Human Image Animation via Synthetic Triplet Supervision](video_generation/vanast_virtual_try-on_with_human_image_animation_via_synthetic_triplet_supervisi.md)**

:   Vanast 提出一种统一框架，通过 Dual Module 架构（HAM + GTM）和三阶段合成数据构建流水线，在单阶段内同时完成服装迁移和人体动画生成，在 Internet 数据集上 PSNR 达到 17.95dB（+5.5dB vs 最佳两阶段方案），LPIPS 仅 0.237。

**[VideoCoF: Unified Video Editing with Temporal Reasoner](video_generation/videocof_unified_video_editing_with_temporal_reasoner.md)**

:   提出 VideoCoF，一种受 Chain-of-Thought 启发的"看→推理→编辑"视频编辑框架，通过让视频扩散模型先预测编辑区域的推理 token（灰度高亮 latent），再生成目标视频 token，在无需用户提供 mask 的前提下实现精确的指令-区域对齐，仅用 50K 视频对训练即达到 SOTA 性能，且支持 16 倍训练长度的视频外推。

**[When Numbers Speak: Aligning Textual Numerals and Visual Instances in Text-to-Video Diffusion Models](video_generation/when_numbers_speak_aligning_textual_numerals_and_visual_instances_in_text-to-vid.md)**

:   NUMINA 的核心思想是，不去重训视频扩散模型，而是在推理时先从 DiT 的注意力中提取一个“可计数的实例布局”，判断数量词和当前布局是否不一致，再对布局做保守的增删修改，并用该布局回头引导重生成，从而显著提升文本到视频模型对“两个苹果、八只鸭子”这类数量约束的遵从能力。

**[When to Lock Attention: Training-Free KV Control in Video Diffusion](video_generation/when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)**

:   提出 KV-Lock，基于扩散模型幻觉检测动态调度背景 KV 缓存融合比例和 CFG 引导强度，在无需训练的前提下同时保证视频编辑的背景一致性和前景生成质量。

---

## 📦 模型压缩 { #model_compression }

**[4D-RGPT: Toward Region-level 4D Understanding via Perceptual Distillation](model_compression/4d_rgpt_toward_region_level_4d_understanding_via_perceptual_distillation.md)**

:   提出4D-RGPT和感知4D蒸馏（P4D）框架，通过从冻结的4D感知专家模型中蒸馏深度和光流等知识到MLLM中增强4D感知，同时构建R4D-Bench——首个区域级4D视频问答基准。

**[A Paradigm Shift: Fully End-to-End Training for Temporal Sentence Grounding in Videos](model_compression/a_paradigm_shift_fully_end-to-end_training_for_temporal_sentence_grounding_in_vi.md)**

:   提出首个完全端到端的时序语句定位(TSGV)框架，通过语句条件适配器(SCADA)将语句嵌入注入视频backbone的中间层来动态调制视觉特征，配合视频中心学习策略加速训练，在Charades-STA和ActivityNet上超越SOTA。

**[Adversarial Concept Distillation for One-Step Diffusion Personalization](model_compression/adversarial_concept_distillation_for_one-step_diffusion_personalization.md)**

:   OPAD 首次解决单步扩散模型的个性化问题（1-SDP），通过教师-学生联合训练 + 对齐损失 + 对抗监督实现单步高质量概念生成，并引入协作学习阶段利用学生生成样本反馈增强双方。

**[An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS](model_compression/an_fpga_implementation_of_displacement_vector_search_for_intra_pattern_copy_in_j.md)**

:   针对 JPEG XS 屏幕内容编码中 Intra Pattern Copy（IPC）模块的位移向量（DV）搜索计算瓶颈，首次提出四级流水线 FPGA 架构并设计基于 IPC Group 对齐的内存组织方式，在 Xilinx Artix-7 上实现 38.3 Mpixels/s 吞吐量和 277 mW 功耗，为 IPC 的实际硬件部署提供了可行方案。

**[ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation](model_compression/arche_autoregressive_residual_compression_with_hyp.md)**

:   在全卷积架构内统一层级超先验、Masked PixelCNN 空间自回归、通道条件建模和 SE 通道激励，不依赖 Transformer 或循环组件，以 95M 参数和 222ms 解码时间实现相对 Ballé 基线 48% BD-Rate 降低并超越 VVC Intra 5.6%。

**[Batch Loss Score for Dynamic Data Pruning](model_compression/batch_loss_score_for_dynamic_data_pruning.md)**

:   提出 Batch Loss Score (BLS)，一种仅用均值 batch loss（而非难以获取的逐样本 loss）来估计样本重要性的方法，通过 EMA 低通滤波的信号处理视角提供理论保证，仅需 3 行代码即可集成到现有动态剪枝框架中。

**[Beyond Loss Values: Robust Dynamic Pruning via Loss Trajectory Alignment](model_compression/beyond_loss_values_robust_dynamic_pruning_via_loss_trajectory_alignment.md)**

:   提出AlignPrune——一个基于损失轨迹对齐的即插即用模块，通过Dynamic Alignment Score（DAS）替代传统损失值排序，使动态数据剪枝在噪声标签场景下准确率提升最高6.3%。

**[Bilevel Layer-Positioning LoRA for Real Image Dehazing](model_compression/bilevel_layer-positioning_lora_for_real_image_dehazing.md)**

:   提出 BiLaLoRA，通过双层优化自动定位 LoRA 应插入的最优网络层，配合 H2C Loss（基于 CLIP 语义方向的无监督去雾损失），实现合成数据预训练的去雾模型向真实场景的高效适配——训练时间降低 77.7%，性能持平全量微调，跨模型跨域均有效。

**[BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](model_compression/binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)**

:   提出 BinaryAttention，将 Transformer 注意力中的 Query 和 Key 量化为 1-bit 二值表示，通过 XNOR + popcount 位运算替代浮点点积，在 A100 上实现比 FlashAttention2 快 2 倍以上的加速，同时在视觉分类/检测/分割/扩散生成等任务上性能持平甚至超越全精度注意力。

**[Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](model_compression/critical_patch-aware_sparse_prompting_with_decoupled_training_for_continual_lear.md)**

:   提出 CPS-Prompt 框架，通过任务感知的关键 patch 采样（CPS）和解耦 prompt-分类器训练（DPCT）两个模块，在边缘设备上实现 Prompt-based 持续学习的训练时内存和计算效率提升约 1.6 倍，同时准确率仅下降约 2%。

**[DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry Estimation](model_compression/dage_dual-stream_architecture_for_efficient_and_fine-grained_geometry_estimation.md)**

:   提出 DAGE 双流 Transformer 架构，将全局一致性建模（低分辨率流）与细粒度细节保持（高分辨率流）解耦，通过轻量 Cross-Attention Adapter 融合，实现 2K 分辨率和 1000 帧长序列上的高质量深度/点图估计和位姿预测，速度比 Pi3 快 2-28 倍，视频几何估计取得新 SOTA。

**[Distilling Balanced Knowledge from a Biased Teacher](model_compression/distilling_balanced_knowledge_from_a_biased_teacher.md)**

:   针对长尾分布下知识蒸馏中教师模型向头部类偏斜的问题，将传统 KL 散度损失分解为跨组损失和组内损失两个组件，通过重平衡跨组损失校准教师的组级预测、重加权组内损失保证各组等贡献，在 CIFAR-100-LT/TinyImageNet-LT/ImageNet-LT 上全面超越现有方法，甚至超过教师模型自身表现。

**[DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration](model_compression/dualreg_dual-space_filtering_and_reinforcement_for_rigid_registration.md)**

:   DualReg提出双空间配准范式，先用轻量级1-point RANSAC + 3-point RANSAC渐进过滤特征空间对应点，再基于过滤后的锚点构建几何代理点集进行双空间联合优化，在3DMatch上实现SOTA精度的同时比MAC快32倍。

**[Enhancing Mixture-of-Experts Specialization via Cluster-Aware Upcycling](model_compression/enhancing_mixture_of_experts_specialization_via_cluster_aware_upcycling.md)**

:   提出 Cluster-aware Upcycling，通过球面 k-means 聚类提取密集模型的语义结构来初始化 MoE 的专家和路由器参数，打破专家对称性并促进早期专业化，配合专家集成自蒸馏损失在 CLIP ViT 上一致超越现有 upcycling 方法。

**[FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection](model_compression/faar_efficient_frequency-aware_multi-task_fine-tuning_via_automatic_rank_selecti.md)**

:   提出 FAAR，一种频率感知的多任务参数高效微调方法，通过 Performance-Driven Rank Shrinking (PDRS) 为每个任务和层动态选择最优秩，并设计 Task-Spectral Pyramidal Decoder (TS-PD) 利用 FFT 频率信息增强空间感知和跨任务一致性，以传统微调 1/9 的参数量实现更优性能。

**[FAIR-Pruner: Leveraging Tolerance of Difference for Flexible Automatic Layer-Wise Neural Network Pruning](model_compression/fair-pruner_leveraging_tolerance_of_difference_for_flexible_automatic_layer-wise.md)**

:   提出 FAIR-Pruner 结构化剪枝框架，通过 Tolerance of Differences（ToD）指标协调两个互补视角：基于类条件可分性的 Wasserstein Utilization Score（识别冗余单元）和基于 Taylor 展开的 Reconstruction Score（保护关键单元），自动确定逐层非均匀剪枝率且支持免搜索灵活调整压缩比，在 CIFAR-10/SVHN/ImageNet 上取得 SOTA。

**[Fixed Anchors Are Not Enough: Dynamic Retrieval and Persistent Homology for Dataset Distillation](model_compression/fixed_anchors_are_not_enough_dynamic_retrieval_and_persistent_homology_for_datas.md)**

:   RETA解耦数据蒸馏中残差匹配的两个失败模式（fit-complexity gap和pull-to-anchor effect），通过动态检索连接（DRC）自适应选择real patch anchor并用持久同调拓扑对齐（PTA）保持类内多样性，在ImageNet-1K ResNet-18 IPC=50上达到64.3%（+3.1% vs FADRM）。

**[FlashVGGT: Efficient and Scalable Visual Geometry Transformers with Compressed Descriptor Attention](model_compression/flashvggt_efficient_and_scalable_visual_geometry_transformers_with_compressed_descr.md)**

:   通过将VGGT中的全局自注意力替换为基于描述符的交叉注意力，实现了1000张图像推理时间降至VGGT的9.3%，同时保持竞争性重建精度，并可扩展至3000+张图像序列。

**[FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation](model_compression/fozo_forward-only_zeroth-order_prompt_optimization_for_test-time_adaptation.md)**

:   提出 FOZO，一种仅需前向传播的零阶 prompt 优化范式，通过 SPSA 梯度估计 + 动态扰动策略 + 深浅层特征统计对齐，在不修改模型权重的情况下实现高效 TTA，在 ImageNet-C 上以 59.52% 准确率超越所有前向方法（含 FOA 58.13%），并支持 INT8 量化模型。

**[Frequency Switching Mechanism for Parameter-Efficient Multi-Task Learning](model_compression/frequency_switching_mechanism_for_parameter-ecient_multi-task_learning.md)**

:   Free Sinewich 提出基于频率切换的参数高效多任务学习框架，通过对共享低秩基矩阵施加不同任务特定频率的正弦变换 $M_t = \sin(\omega_t \cdot M_{AWB})$，以接近零成本实现真正的参数复用和任务特化，在密集预测基准上以最少可训练参数达到SOTA。

**[From Fewer Samples to Fewer Bits: Reframing Dataset Distillation as Joint Optimization of Precision and Compactness](model_compression/from_fewer_samples_to_fewer_bits_reframing_dataset_distillation_as_joint_optimiz.md)**

:   提出 QuADD 框架，将可微量化模块嵌入数据集蒸馏循环中，联合优化合成数据与量化参数，实现在固定比特预算下"更少样本 + 更低精度"的帕累托最优压缩。

**[Generative Video Compression with One-Dimensional Latent Representation](model_compression/generative_video_compression_with_one-dimensional_latent_representation.md)**

:   提出 GVC1D，首次将视频压缩的潜在表示从2D网格替换为紧凑的1D token序列，结合1D记忆模块建模长期时序上下文，在感知质量指标上实现 60%+ 的码率节省。

**[GeoFusion-CAD: Structure-Aware Diffusion with Geometric State Space for Parametric 3D Design](model_compression/geofusion-cad_structure-aware_diffusion_with_geometric_state_space_for_parametri.md)**

:   本文提出 GeoFusion-CAD，一个端到端的扩散框架，通过将 CAD 程序编码为层次化树结构并引入几何感知的 G-Mamba 块（线性时间复杂度）替代二次复杂度的 Transformer，实现了对长序列参数化 CAD 程序的可扩展和结构感知生成，在新构建的 DeepCAD-240（最长240步命令）基准上大幅超越 Transformer 方法。

**[HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](model_compression/hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)**

:   HiAP 把 ViT 剪枝写成一个端到端的预算感知学习问题，同时对整头/整块和头内维度/FFN 神经元两种粒度做随机可微门控，在一次训练里自动长出满足算力预算的稠密子网络，省掉了常见的排序、阈值搜索和额外微调流程。

**[HierAmp: Coarse-to-Fine Autoregressive Amplification for Generative Dataset Distillation](model_compression/hieramp_coarse-to-fine_autoregressive_amplification_for_generative_dataset_disti.md)**

:   提出 HierAmp，在视觉自回归（VAR）模型的粗到细生成过程中，向每个尺度注入可学习的类别 token 识别语义显著区域，并通过正 logit 偏置放大这些区域的注意力，使蒸馏数据在粗尺度获得更丰富多样的布局、在细尺度聚焦于类别相关细节，在多个数据集蒸馏基准上达到 SOTA。

**[Learning through Creation: A Hash-Free Framework for On-the-Fly Category Discovery](model_compression/learning_through_creation_a_hash-free_framework_for_on-the-fly_category_discover.md)**

:   提出 LTC 框架，通过在训练阶段利用 MKEE（最小化核能量+最大化熵）在线生成伪未知类样本，配合双最大间隔损失和自适应阈值，在7个数据集上实现1.5%–13.1%的全类精度提升，彻底摆脱了哈希编码对细粒度语义的损害。

**[LLaVA-LE: Large Language-and-Vision Assistant for Lunar Exploration](model_compression/llava-le_large_language-and-vision_assistant_for_lunar_exploration.md)**

:   LLaVA-LE 是首个面向月球探测的视觉语言模型，通过构建大规模真实月球图像-文本数据集 LUCID（96K 图像+81K QA对）和两阶段课程学习微调 LLaVA，在月球地质理解和多模态推理上实现 3.3× 基线提升。

**[MaMe & MaRe: Matrix-Based Token Merging and Restoration for Efficient Visual Perception and Synthesis](model_compression/mame_and_mare_matrix_based_token_merging_and_restoration_for_efficient_visual_perception_and_synthesis.md)**

:   提出 MaMe，一种基于全矩阵运算的免训练可微分 token 合并方法，以及其逆操作 MaRe 用于 token 恢复，在图像分类、视频识别和图像生成等任务中实现高效加速且性能损失极小。

**[Markovian Scale Prediction: A New Era of Visual Autoregressive Generation](model_compression/markovian_scale_prediction_a_new_era_of_visual_autoregressive_generation.md)**

:   将视觉自回归模型 (VAR) 从全上下文依赖的 next-scale prediction 重构为基于马尔可夫过程的 Markovian scale prediction，通过滑动窗口历史补偿机制实现非全上下文建模，在 ImageNet 上 FID 降低 10.5%、峰值内存减少 83.8%。

**[MARVO: Marine-Adaptive Radiance-aware Visual Odometry](model_compression/marvo_marine-adaptive_radiance-aware_visual_odometry.md)**

:   提出 MARVO 水下视觉里程计框架，将物理感知辐射适配器 (PARA) 嵌入 LoFTR 特征匹配器补偿水下波长衰减、结合 GTSAM 多传感器因子图融合和强化学习位姿图优化 (RL-PGO)，在水下场景实现鲁棒定位。

**[MEMO: Human-like Crisp Edge Detection Using Masked Edge Prediction](model_compression/memo_human-like_crisp_edge_detection_using_masked_edge_prediction.md)**

:   提出 MEMO 框架，通过掩码边缘训练和基于置信度排序的渐进式推理策略，仅使用交叉熵损失就能生成清晰的单像素边缘图，在 crispness-aware 评估上大幅超越现有方法（BSDS 上 CEval ODS 从 0.749 提升到 0.836）。

**[Memory-Efficient Transfer Learning with Fading Side Networks via Masked Dual Path Distillation](model_compression/memory_efficient_transfer_learning_with_fading_side_networks.md)**

:   MDPD提出通过冻结骨干网络与轻量侧网络之间的双向知识蒸馏实现高效微调，训练完成后丢弃侧网络，从而同时实现训练时的参数/内存高效和推理时的速度高效。

**[On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](model_compression/on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)**

:   首次系统研究了扩散模型图像压缩在比特翻转错误下的鲁棒性，发现基于反向信道编码（RCC）的扩散压缩方法天然比传统和学习型编解码器更耐错，并提出 Robust Turbo-DDCM 变体通过独立编码原子索引进一步提升鲁棒性，在 BER 达 $10^{-3}$ 时仍保持良好重建质量。

**[Parallax to Align Them All: An OmniParallax Attention Mechanism for Distributed Multi-View Image Compression](model_compression/parallax_to_align_them_all_an_omniparallax_attention_mechanism_for_distributed_m.md)**

:   提出 OmniParallax Attention Mechanism (OPAM) 用于分布式多视角图像压缩（DMIC），通过两阶段视差注意力显式建模任意视角对之间的相关性和对齐特征，构建的 ParaHydra 框架首次让 DMIC 方法显著超越 SOTA MIC 编码器，同时大幅降低计算开销。

**[PlanaReLoc: Camera Relocalization in 3D Planar Primitives via Region-Based Structure Matching](model_compression/planareloc_camera_relocalization_in_3d_planar_primitives_via_region-based_struct.md)**

:   首次提出基于平面基元（planar primitives）和 3D 平面地图的相机重定位范式 PlanaReLoc，通过深度匹配器在统一嵌入空间中关联查询图像的平面区域与地图平面基元，实现了无需真实纹理地图、位姿先验或逐场景训练的轻量化 6-DoF 相机重定位。

**[Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model](model_compression/planning_in_8_tokens_a_compact_discrete_tokenizer_for_latent_world_model.md)**

:   提出 CompACT，将每张图像压缩至仅 8 个离散 token（约 128 bits），通过冻结预训练视觉编码器保留规划关键语义信息、生成式解码补充感知细节，使基于世界模型的规划速度提升约 40 倍且精度不降。

**[PPCL: Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](model_compression/ppcl_pluggable_pruning_dit_distillation.md)**

:   提出 PPCL 框架，针对超大规模 Multi-Modal Diffusion Transformer (MMDiT, 8–20B 参数) 设计结构化剪枝方案：通过线性探针 (Linear Probe) 学习每层的可替代性，结合 CKA 一阶差分自动定位连续冗余层区间，再以非顺序交替蒸馏实现深度+宽度双轴剪枝，最终在 Qwen-Image 20B 上实现 50% 参数缩减、1.8× 推理加速，平均性能仅下降 2.61%。

**[Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy](model_compression/preference-aligned_lora_merging_preserving_subspace_coverage_and_addressing_dire.md)**

:   本文从子空间覆盖（subspace coverage）和方向各向异性（anisotropy）两个视角重新审视LoRA合并问题，提出TARA-Merging框架，通过保留LoRA方向并结合偏好加权的交叉熵伪损失进行方向级重新加权，在8个视觉和6个NLI基准上持续超越现有合并方法。

**[PriVi: Towards a General-Purpose Video Model for Primate Behavior in the Wild](model_compression/privi_towards_a_general-purpose_video_model_for_primate_behavior_in_the_wild.md)**

:   PriVi 构建了 424 小时的大规模灵长类视频预训练数据集，并通过在 V-JEPA 上进行**领域级预训练**（非目标数据集级别），首次证明了视频模型的领域级预训练可以跨数据集泛化，在四个灵长类行为识别基准上用仅 220K 参数的冻结分类器超越了全量微调的专用模型。

**[QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](model_compression/quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)**

:   提出 QuantVLA，首个面向 Vision-Language-Action (VLA) 模型的免训练后量化框架，通过选择性量化布局和两个轻量级标定机制（注意力温度匹配 ATM 和输出头平衡 OHB），在 W4A8 精度下实现约 70% 的内存节省，同时任务成功率超过全精度基线。

**[RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](model_compression/rdvq_differentiable_vq_image_compression.md)**

:   RDVQ 通过对码本分布的可微松弛，首次实现了 VQ-based 图像压缩的端到端率失真联合优化，在极低码率下以不到 20% 的参数量取得了优于或竞争性的感知质量。

**[RL-ScanIQA: Reinforcement-Learned Scanpaths for Blind 360° Image Quality Assessment](model_compression/rl-scaniqa_reinforcement-learned_scanpaths_for_blind_360image_quality_assessment.md)**

**[SODA: Sensitivity-Oriented Dynamic Acceleration for Diffusion Transformer](model_compression/soda_sensitivity-oriented_dynamic_acceleration_for_diffusion_transformer.md)**

:   提出 SODA，通过离线细粒度敏感度建模 + 动态规划优化缓存间隔 + 统一自适应剪枝策略，在无需训练的条件下对 Diffusion Transformer 实现可控加速比下的高保真生成。

**[TALON: Test-time Adaptive Learning for On-the-Fly Category Discovery](model_compression/talon_test-time_adaptive_learning_for_on-the-fly_category_discovery.md)**

:   提出首个面向 on-the-fly 类别发现（OCD）的测试时自适应框架 TALON，通过语义感知原型更新 + 稳定编码器适应 + 边距感知 logit 校准，摒弃哈希编码在连续特征空间直接建模，大幅缓解类别爆炸并显著提升新类发现精度。

**[F²HDR: Two-Stage HDR Video Reconstruction via Flow Adapter and Physical Motion Modeling](model_compression/textf2texthdr_two-stage_hdr_video_reconstruction_via_flow_adapter_and_physical_m.md)**

:   提出 F²HDR，一个两阶段 HDR 视频重建框架，通过 Flow Adapter 将通用预训练光流适配到交替曝光场景以实现鲁棒对齐，并利用物理运动建模从光流中提取连续运动掩码来引导第二阶段的伪影消除，在真实 HDR 视频基准上达到 SOTA。

**[Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](model_compression/towards_generalizable_ai-generated_image_detection_via_image-adaptive_prompt_lea.md)**

:   提出 Image-Adaptive Prompt Learning (IAPL)，在推理时根据每张测试图像动态调整 CLIP 编码器的 prompt，通过测试时 token 调优和条件信息学习器实现对未见生成器的强泛化，在 UniversalFakeDetect 和 GenImage 上分别达到 95.61% 和 96.7% 平均准确率的 SOTA 性能。

**[Towards Source-Aware Object Swapping with Initial Noise Perturbation](model_compression/towards_source-aware_object_swapping_with_initial_noise_perturbation.md)**

:   提出 SourceSwap，通过频率分离的初始噪声扰动从单张图像生成高质量伪配对数据，并采用源感知双 U-Net 架构学习跨物体对齐，实现零样本、无逐物体微调的高保真物体替换。

**[Understanding and Enforcing Weight Disentanglement in Task Arithmetic](model_compression/understanding_and_enforcing_weight_disentanglement_in_task_arithmetic.md)**

:   本文提出任务特征专业化（TFS）作为权重解耦的充分条件，揭示其几何结果是权重向量正交性，并基于此提出 OrthoReg 正则化方法，通过在微调时强制权重更新矩阵的列向量正交来促进任务向量解耦，显著提升各种任务算术方法的性能。

**[UniComp: Rethinking Video Compression Through Informational Uniqueness](model_compression/unicomp_rethinking_video_compression_through_informational_uniqueness.md)**

:   提出基于信息唯一性（而非注意力）的视频 token 压缩框架 UniComp，通过帧组融合、token 分配和空间动态压缩三个模块在时序-空间-全局维度上最大化保留唯一信息，在仅保留 10% token 时仍能超越未压缩基线性能。

**[Unlocking ImageNet's Multi-Object Nature: Automated Large-Scale Multilabel Annotation](model_compression/unlocking_imagenets_multi-object_nature_automated_large-scale_multilabel_annotat.md)**

:   提出全自动流水线，利用自监督 ViT 特征进行无监督目标发现，为 ImageNet-1K 全部 128 万训练图像生成带空间定位的多标签标注，无需人工标注，模型在域内和下游多标签任务上均获一致提升（ReaL +2.0 top-1, COCO +4.2 mAP）。

**[WPT: World-to-Policy Transfer via Online World Model Distillation](model_compression/wpt_world-to-policy_transfer_via_online_world_model_distillation.md)**

:   WPT 提出世界-策略转移训练范式，通过可训练的奖励模型将世界模型的未来预测知识注入教师策略，再通过策略蒸馏和世界奖励蒸馏转移到轻量学生策略，实现79.23驾驶得分（闭环）且推理速度提升4.9倍。

---

## 🖼️ 图像恢复 { #image_restoration }

**[Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](image_restoration/beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)**

:   提出IQPIR框架，引入预训练NR-IQA模型的图像质量先验(IQP)作为条件信号，通过质量条件化Transformer、双Codebook结构和离散表示空间质量优化三个机制，引导图像修复过程趋向最高感知质量，在盲人脸修复等任务上全面超越SOTA。

**[Beyond the Ground Truth: Enhanced Supervision for Image Restoration](image_restoration/beyond_the_ground_truth_enhanced_supervision_for_image_restoration.md)**

:   提出通过超分辨率+频域自适应混合来增强现有数据集中次优GT图像的感知质量，并训练轻量级ORNet精修模块，无需修改预训练修复模型即可提升输出的感知质量。

**[BHCast: Unlocking Black Hole Plasma Dynamics from a Single Blurry Image with Long-Term Forecasting](image_restoration/bhcast_unlocking_black_hole_plasma_dynamics_from_a_single_blurry_image_with_long.md)**

:   BHCast从单张模糊的EHT黑洞图像出发，通过U-Net动力学代理模型进行超分辨率+长期自回归预测（100步稳定），从预测的等离子体动力学中提取物理特征（旋转速度、螺旋角等），再通过XGBoost推断黑洞自旋和倾角，在真实M87*观测图像上也展示了有效性。

**[Blink: Dynamic Visual Token Resolution for Enhanced Multimodal Understanding](image_restoration/blink_dynamic_visual_token_resolution_for_enhanced_multimodal_understanding.md)**

:   提出 Blink 框架，通过在 MLLM 不同 Transformer 层动态扩展和丢弃视觉 token（模拟人类"快速眨眼"式扫描），在单次前向传播中自适应增强视觉感知能力，在多个多模态基准上提升 LLaVA-1.5 性能。

**[BluRef: Unsupervised Image Deblurring with Dense-Matching References](image_restoration/bluref_unsupervised_image_deblurring_with_dense-matching_references.md)**

:   提出 BluRef，首个利用非配对参考清晰图像通过稠密匹配生成伪 ground truth 来训练去模糊网络的无监督框架，性能逼近甚至超越有监督方法。

**[Bridging the Perception Gap in Image Super-Resolution Evaluation](image_restoration/bridging_the_perception_gap_in_image_super-resolution_evaluation.md)**

:   通过大规模用户研究揭示现有 SR 评估指标（PSNR、SSIM、LPIPS 等）与人类感知严重不一致，分析其内在缺陷后提出极简但有效的 RQI（Relative Quality Index）框架，通过学习图像对之间的相对质量差异实现更可靠的 SR 评估，且可作为损失函数指导 SR 训练。

**[PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning](image_restoration/diffusion-based_srgb_real_noise_generation_via_prompt-driven_noise_representatio.md)**

:   PNG提出用可学习的Global/Local Prompt组件从真实噪声中自动提取噪声特征（替代ISO/相机型号等metadata），通过Prompt AutoEncoder编码噪声到latent空间+Prompt DiT（基于一致性模型）一步生成latent code，实现无需任何metadata的真实sRGB噪声合成，下游DnCNN去噪在SIDD上仅落后真实数据0.08dB。

**[Disentangled Textual Priors for Diffusion-based Image Super-Resolution](image_restoration/disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)**

:   提出 DTPSR，通过将文本先验沿空间层级（全局/局部）和频率语义（低频/高频）两个维度解耦，构建解耦的跨注意力注入管线和多分支 CFG 策略，实现感知质量优越的扩散超分辨率。

**[DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](image_restoration/drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)**

:   提出退化感知扩散框架 DRFusion，通过直接回归融合图像（而非显式预测噪声）和联合观测模型校正机制，在少量扩散步骤内实现任意退化场景下的多模态图像融合。

**[EVLF: Early Vision-Language Fusion for Generative Dataset Distillation](image_restoration/evlf_early_vision-language_fusion_for_generative_dataset_distillation.md)**

:   提出 EVLF，一种在编码器-骨干网络接口处进行视觉-语言早期融合的即插即用方法，解决了扩散模型数据集蒸馏中晚期语义注入导致的文本过度主导和视觉保真度下降问题。

**[FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution](image_restoration/fidesr_high-fidelity_and_detail-preserving_one-step_diffusion_super-resolution.md)**

:   提出 FiDeSR，一种高保真和细节保持的单步扩散超分框架，通过细节感知加权（DAW）、隐空间残差精炼块（LRRB）和潜在频率注入模块（LFIM）三个互补组件，同时解决单步扩散超分中的结构保真度退化和高频细节恢复不足问题。

**[FinPercep-RM: A Fine-grained Reward Model and Co-evolutionary Curriculum for RL-based Real-world Super-Resolution](image_restoration/finpercep_rm_a_fine_grained_reward_model_and_co_evolutionary_curriculum_for_rl_ba.md)**

:   提出细粒度感知奖励模型 FinPercep-RM 和协同进化课程学习（CCL）策略，解决 RLHF 应用于真实世界超分辨率时的奖励黑客和训练不稳定问题，通过同时输出全局质量分数和空间退化热力图实现局部缺陷感知。

**[GSNR: Graph Smooth Null-Space Representation for Inverse Problems](image_restoration/gsnr_graph_smooth_null_space_representation_for_inverse_problems.md)**

:   提出图平滑零空间表示（GSNR），通过谱图理论构建零空间受限拉普拉斯矩阵并选择最平滑的 p 个谱模式作为零空间投影基，为 PnP、DIP 和扩散模型等逆问题求解器提供结构化的零空间约束，在去模糊、压缩感知、去马赛克和超分辨率上提升高达 4.3dB PSNR。

**[IA-CLAHE: Image-Adaptive Clip Limit Estimation for CLAHE](image_restoration/ia_clahe_image_adaptive_clip_limit.md)**

:   IA-CLAHE 通过证明 CLAHE 的直方图重分配过程几乎处处可微，首次实现了逐图块自适应 clip limit 的端到端学习，无需预搜索 ground truth clip limit 即可在恶劣天气条件下零样本提升识别性能和视觉质量。

**[Flickerformer: A Duet of Periodicity and Directionality for Burst Flicker Removal](image_restoration/it_takes_two_a_duet_of_periodicity_and_directionality_for_burst_flicker_removal.md)**

:   揭示闪烁伪影具有周期性和方向性两个内在物理特性，设计Flickerformer三模块（PFM/AFFN/WDAM）分别针对帧间/帧内周期性和方向性建模，以3.92M参数量在BurstDeflicker基准上达到31.226dB PSNR，超越第二名AST +0.580dB且仅用其19.70%参数。

**[Learning to Translate Noise for Robust Image Denoising](image_restoration/learning_to_translate_noise_for_robust_image_denoising.md)**

:   提出噪声翻译框架，通过轻量级噪声翻译网络将未知真实噪声转换为高斯噪声，再由预训练的高斯去噪网络处理，在 OOD 真实噪声基准上平均 PSNR 提升 1.5dB 以上，且翻译网络仅 0.29M 参数、可跨去噪器迁移。

**[MAD-Avatar: Motion-Aware Animatable Gaussian Avatars Deblurring](image_restoration/motionaware_animatable_gaussian_avatars_deblurring.md)**

:   首次实现从模糊视频直接重建清晰可驱动3D高斯人体avatar：提出3D感知的物理模糊形成模型(将模糊分解为子帧SMPL运动+canonical 3DGS)，用B-spline插值+位姿变形网络建模子帧运动，帧间正则化解决运动方向歧义，在合成和真实数据集上大幅超越"2D去模糊+3DGS"两阶段方案(PSNR提升约2.5dB)。

**[NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness](image_restoration/nec-diff_noise-robust_event-raw_complementary_diffusion_for_seeing_motion_in_ext.md)**

:   提出 NEC-Diff，一个基于扩散模型的事件-RAW 混合成像框架，利用 RAW 图像的光照先验引导事件去噪、事件的高动态范围边缘辅助图像去噪，结合双模态 SNR 引导的可靠信息提取和交叉模态注意力扩散，在极暗环境下（0.001-0.8 lux）实现高质量动态场景重建，PSNR 达 24.51 dB（REAL 数据集）。

**[NTIRE 2026 The 3rd RAIM Challenge: AI Flash Portrait (Track 3)](image_restoration/ntire_2026_ai_flash_portrait_challenge.md)**

:   NTIRE 2026第三届RAIM挑战赛AI Flash Portrait赛道：将弱闪光灯低光照人像映射为强闪光灯专业级人像，提供800组真实配对数据（含专业设计师修图GT），采用区域感知客观指标+专家盲评的双重评估体系，118支队伍注册、3187次有效提交。

**[NTIRE 2026 The Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images](image_restoration/ntire_2026_raindrop_removal_challenge.md)**

:   NTIRE 2026第二届日夜双焦点雨滴去除挑战赛总结报告：基于Raindrop Clarity真实数据集（14,139训练/407验证/593测试），168支队伍参赛，17支提交有效方案，冠军AIIA-Lab以MSDT骨干+伪GT精修流水线取得35.24分最佳成绩。

**[PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors](image_restoration/phasr_generalized_image_shadow_removal_with_physically_aligned_priors.md)**

:   提出PhaSR框架，通过双层物理先验对齐——全局级的PAN执行无参数Retinex分解抑制色彩偏差、局部级的GSRA利用差分注意力对齐DepthAnything深度先验和DINO-v2语义嵌入——实现从单光源直射阴影到多光源环境光场景的泛化阴影去除，在WSRD+和Ambient6K上达到SOTA且FLOPs最低。

**[POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](image_restoration/polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)**

:   在 POLISH 框架基础上提出 POLISH+ 和 POLISH++，通过分块训练-拼接策略和基于 arcsinh 的非线性变换，实现宽视场（12,960×12,960 像素）和高动态范围（$\sim 10^6$）条件下的射电干涉图像重建与超分辨率，并首次展示深度学习方法可超分辨强引力透镜系统。

**[RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](image_restoration/rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)**

:   RAR 将图像质量评估（IQA）与图像修复（IR）深度集成为统一端到端模型，在潜在空间中迭代执行"评估-修复-验证"循环，在复合退化场景下 PSNR 提升 +2.71 dB 且速度比 AgenticIR 快 11.27×。

**[RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](image_restoration/rawdomain_degradation_models_smartphone_sr.md)**

:   证明了精心设计的设备特定退化建模（通过标定获取真实的 blur 和 noise 参数）可以显著提升手机超分辨率的真实场景性能——通过将公开渲染图像 unprocess 到不同手机的 RAW 域生成高低分辨率训练对，训练的 SR 模型在保留设备的真实数据上明显优于使用大量任意退化组合训练的基线。

**[Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](image_restoration/real_iisr_infrared_image_super_resolution_autoregressive.md)**

:   提出 Real-IISR 统一自回归框架，通过热-结构引导模块、条件自适应码本和热序一致性损失解决真实红外图像超分辨率的特有挑战，并构建了 FLIR-IISR 数据集（1457 对真实 LR-HR 红外图像）。

**[SAT: Selective Aggregation Transformer for Image Super-Resolution](image_restoration/sat_selective_aggregation_transformer_for_image_super_resolution.md)**

:   提出选择性聚合 Transformer (SAT)，通过密度驱动 token 聚合将 Key-Value 矩阵 token 数减少 97%、保持 Query 全分辨率，实现高效全局注意力建模，超越 SOTA PFT 达 0.22dB 且 FLOPs 降低 27%。

**[SelfHVD: Self-Supervised Handheld Video Deblurring](image_restoration/selfhvd_self-supervised_handheld_video_deblurring.md)**

:   SelfHVD 利用手持视频中自然存在的清晰帧作为监督信号，通过自增强视频去模糊（SEVD）构建高质量训练对和自约束空间一致性维护（SCSCM）防止位移偏移，实现了无需配对数据的手持视频去模糊。

**[Winner of CVPR2026 NTIRE Challenge on Image Shadow Removal: Semantic and Geometric Guidance for Shadow Removal via Cascaded Refinement](image_restoration/shadow_removal_cascaded_refinement.md)**

:   基于 OmniSR 构建三级级联精炼 pipeline，结合冻结 DINOv2 语义特征与单目深度/法线几何引导，通过收缩约束损失稳定多阶段训练，在 NTIRE 2026 阴影去除挑战赛中获得第一名。

**[ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration](image_restoration/shiftlut_spatial_shift_enhanced_look-up_tables_for_efficient_image_restoration.md)**

:   提出 ShiftLUT，通过可学习空间偏移模块（LSS）实现 LUT 方法中最大感受野（65×65），配合非对称双分支架构和误差有界自适应采样（EAS），在存储 104KB + 推理 84ms 的条件下超越所有现有 LUT 方法。

**[Spectral Super-Resolution via Adversarial Unfolding and Data-Driven Spectrum Regularization](image_restoration/spectral_super-resolution_via_adversarial_unfolding_and_data-driven_spectrum_reg.md)**

:   提出 UALNet，通过将数据驱动的光谱先验（PriorNet）和对抗学习项同时嵌入深度展开框架，实现从 Sentinel-2 多光谱数据（12 波段）到 NASA AVIRIS 高光谱图像（186 波段）的光谱超分辨率，性能超越 Transformer 的同时仅需 15% 计算量和 1/20 参数。

**[Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging](image_restoration/statistical_characteristic-guided_denoising_for_rapid_high-resolution_transmissi.md)**

:   提出统计特征引导去噪网络 SCGN，利用空间域的窗口标准差加权和频域的频带引导加权，分别在空间和频率两个域自适应地增强信号、抑制噪声，结合 HRTEM 专用噪声标定方法生成含无序结构的真实噪声数据集，实现毫秒级高分辨率透射电子显微镜图像的高质量去噪。

**[The Surprising Effectiveness of Noise Pretraining for Implicit Neural Representations](image_restoration/the_surprising_effectiveness_of_noise_pretraining_for_implicit_neural_representa.md)**

:   本文通过系统的实验分析发现：用非结构化噪声（均匀/高斯分布）预训练 INR 可在图像拟合中达到惊人的 ~80dB PSNR，远超所有数据驱动初始化方法；而具有自然图像 $1/|f^\alpha|$ 频谱结构的噪声则在信号拟合和去噪之间实现最佳平衡，无需任何真实数据即可匹配 SOTA 数据驱动初始化性能。

**[TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising](image_restoration/tm-bsn_triangular-masked_blind-spot_network_for_real-world_self-supervised_image.md)**

:   提出三角掩码盲点网络 TM-BSN，通过将盲点区域设计为与真实 sRGB 噪声的菱形空间相关模式精确对齐的形状，在原始分辨率上实现无需下采样的自监督图像去噪，并通过知识蒸馏进一步提升性能，在 SIDD 和 DND 基准上达到自监督去噪 SOTA。

**[UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](image_restoration/ucan_unified_convolutional_attention_lightweight_sr.md)**

:   提出 UCAN 轻量级超分辨率网络，统一卷积和注意力机制来高效扩展有效感受野，通过 Hedgehog 注意力解决线性注意力的秩坍缩问题，引入大核蒸馏模块和半共享参数策略，在 Manga109 (4×) 上以仅 48.4G MACs 达到 31.63 dB PSNR。

**[UDAPose: Unsupervised Domain Adaptation for Low-Light Human Pose Estimation](image_restoration/udapose_unsupervised_domain_adaptation_for_low_light_human_pose_estimation.md)**

:   UDAPose通过基于稳定扩散的低光照图像合成（保持高频低光特征）和动态注意力控制模块（自适应平衡视觉线索与姿态先验），在低光照硬集上实现56.4%的AP提升。

**[UniBlendNet: Unified Global, Multi-Scale, and Region-Adaptive Modeling for Ambient Lighting Normalization](image_restoration/uniblendnet_unified_global_multi_scale_and_region_adaptive_modeling_for_ambient_lighting_normalization.md)**

:   提出 UniBlendNet，在 IFBlend 基础上统一融合全局上下文建模、多尺度特征聚合和区域自适应残差精修三个模块，用于复杂空间变化光照条件下的环境光归一化任务。

**[Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis](image_restoration/unicac_universal_computational_aberration_correction_benchmark.md)**

:   本文构建了首个大规模通用计算像差校正(CAC)基准 UniCAC，提出光学退化评估器(ODE)量化像差难度，并对24种图像恢复/CAC算法进行了全面评估，揭示了先验利用、网络架构和训练策略三大关键因素对CAC性能的影响。

**[UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization](image_restoration/unirain_unified_image_deraining_rag_dataset_distillation.md)**

:   提出 UniRain 统一图像去雨框架，通过 RAG 驱动的数据蒸馏从百万级公开数据集筛选高质量样本，结合非对称 MoE 架构和多目标重加权优化策略，在雨条纹和雨滴（白天/夜间）四种退化类型上实现一致优异性能。

**[Variational Garrote for Sparse Inverse Problems](image_restoration/variational_garrote_for_sparse_inverse_problems.md)**

:   在统一的稀疏逆问题框架下，系统比较 $\ell_1$ 正则化（LASSO）与 Variational Garrote（VG，一种通过变分二值门控近似 $\ell_0$ 的方法），在信号重采样、去噪和稀疏视角 CT 重建三个任务上证明 VG 在强欠定场景下能显著降低最小泛化误差，尤其在采样率 <20% 或投影角度极少时优势最大。

---

## 🎯 目标检测 { #object_detection }

**[A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning Matters and Parallel Decoder Helps](object_detection/a_closer_look_at_cross-domain_few-shot_object_detection_fine-tuning_matters_and_.md)**

:   提出混合集成解码器(HED)和渐进微调策略用于跨域少样本目标检测，通过并行化部分解码层并随机初始化去噪查询引入预测多样性，在CD-FSOD/ODinW-13/RF100-VL三个基准上达到SOTA，不引入额外参数。

**[ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](object_detection/abra_teleporting_fine-tuned_knowledge_across_domains_for_open-vocabulary_object_.md)**

:   提出 ABRA 方法，将域知识与类别知识解耦，通过 Objectification 构建类无关域专家、SVFT 提取轻量类别残差、Orthogonal Procrustes 旋转对齐实现权重空间"传送"，在目标域完全无某些类别数据时仍可迁移这些类别的检测能力。

**[AR²-4FV: Anchored Referring and Re-identification for Long-Term Grounding in Fixed-View Videos](object_detection/ar2-4fv_anchored_referring_and_re-identification_for_long-term_grounding_in_fixe.md)**

:   利用固定视角视频中背景结构的时不变性，构建离线 Anchor Bank + 在线 Anchor Map 作为语言-场景持久记忆，配合锚点引导的重入先验和 ReID-Gating 身份验证机制，实现目标遮挡/离场后的鲁棒重捕获，RCR 提升 10.3%、RCL 降低 24.2%。

**[Beyond Caption-Based Queries for Video Moment Retrieval](object_detection/beyond_caption-based_queries_for_video_moment_retrieval.md)**

:   揭示了VMR中caption-based查询与真实用户搜索查询之间的巨大鸿沟，提出了三个搜索查询基准，并通过移除自注意力+查询Dropout两项架构修改来缓解DETR中的解码器查询坍塌问题，在多时刻搜索查询上提升高达21.83% mAPm。

**[Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection](object_detection/beyond_prompt_degradation_prototype-guided_dual-pool_prompting_for_incremental_o.md)**

:   提出 PDP 框架，通过双池提示解耦（共享池 + 私有池）和原型引导伪标签生成（PPG），解决增量目标检测中提示耦合与提示漂移导致的提示退化问题，在 COCO 和 VOC 上取得 SOTA。

**[Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval](object_detection/beyond_semantic_search_towards_referential_anchoring_in_composed_image_retrieval.md)**

:   提出Object-Anchored Composed Image Retrieval（OACIR）新任务和OACIRR大规模基准（160K+四元组），以及AdaFocal框架通过上下文感知注意力调制器自适应地增强对锚定实例区域的关注，在实例级检索保真度上大幅超越现有方法。

**[CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection](object_detection/cd-buffer_complementary_dual-buffer_framework_for_test-time_adaptation_in_advers.md)**

:   提出 CD-Buffer 框架，通过统一的域差异度量驱动减性缓冲（通道抑制）和加性缓冲（轻量适配器补偿）的互补协作，实现跨不同严重程度恶劣天气条件下的鲁棒测试时目标检测适应。

**[CompAgent: An Agentic Framework for Visual Compliance Verification](object_detection/compagent_an_agentic_framework_for_visual_compliance_verification.md)**

:   提出 CompAgent，首个用于视觉合规验证的智能体框架——Planning Agent 根据合规策略动态选择视觉工具（目标检测、人脸分析、NSFW 检测等），Compliance Verification Agent 整合图像、工具输出和策略上下文进行多模态推理，无需训练即在 UnsafeBench 上超越 SOTA 10% 达 76% F1。

**[DA-Mamba: Learning Domain-Aware State Space Model for Global-Local Alignment in Domain Adaptive Object Detection](object_detection/da-mamba_learning_domain-aware_state_space_model_for_global-local_alignment_in_d.md)**

:   提出 DA-Mamba，一种 CNN-SSM 混合架构，通过 Image-Aware SSM（IA-SSM）和 Object-Aware SSM（OA-SSM）两个模块，以线性复杂度实现图像级和实例级的全局-局部域不变特征对齐，在四个域自适应检测基准上达到 SOTA。

**[Detecting Unknown Objects via Energy-Based Separation for Open World Object Detection](object_detection/detecting_unknown_objects_via_energy-based_separation.md)**

:   提出 DEUS 框架，通过 ETF 子空间未知目标分离（EUS）在几何正交的已知/未知子空间中利用能量分数有效分离已知、未知和背景提案，并设计能量基已知区分损失（EKD）减少增量学习中新旧类的交叉干扰，在 OWOD 基准上大幅提升未知目标召回率。

**[Does YOLO Really Need to See Every Training Image in Every Epoch?](object_detection/does_yolo_really_need_to_see_every_training_image_in_every_epoch.md)**

:   提出 Anti-Forgetting Sampling Strategy (AFSS)，根据每张训练图像的学习充分度（min(Precision, Recall)）动态决定哪些图像参与训练、哪些可以跳过，实现 YOLO 系列检测器 1.43× 以上的训练加速同时保持甚至提升检测精度。

**[Evaluating Few-Shot Pill Recognition Under Visual Domain Shift](object_detection/evaluating_few-shot_pill_recognition_under_visual_domain_shift.md)**

:   本文从部署视角系统评估药丸识别在跨域few-shot条件下的泛化能力，揭示语义分类1-shot即饱和但定位/recall在重叠遮挡下急剧下降的解耦现象，并证明训练数据的视觉真实性远比数据量或shot数更关键。

**[EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer](object_detection/ew-detr_evolving_world_object_detection_via_incremental_low-rank_detection_trans.md)**

:   提出 Evolving World Object Detection (EWOD) 范式及 EW-DETR 框架，通过增量 LoRA 适配器、查询范数物体性适配器和熵感知未知混合三个协同模块，在无样本回放条件下同时解决类别增量学习、域迁移适应和未知目标检测问题，FOGS 指标提升 57.24%。

**[Few-Shot Incremental 3D Object Detection in Dynamic Indoor Environments](object_detection/few-shot_incremental_3d_object_detection_in_dynamic_indoor_environments.md)**

:   提出 FI3Det，首个少样本增量 3D 目标检测框架：在基础训练阶段通过 VLM 引导的未知对象学习模块提前感知潜在新类别，在增量阶段通过门控多模态原型铸造模块融合 2D 语义和 3D 几何特征进行新类检测，在 ScanNet V2 和 SUN RGB-D 上的新类 mAP 平均提升 17.37%。

**[Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection](object_detection/foundation_model_priors_enhance_object_focus_in_feature_space_for_source-free_ob.md)**

:   提出 FALCON-SFOD 框架，通过基础模型（OV-SAM）生成的类别无关二值掩码正则化检测器特征空间（SPAR），结合不平衡感知的噪声鲁棒伪标签损失（IRPL），在无源域目标检测中增强目标聚焦表征，多个基准上达到 SOTA。

**[Fourier Angle Alignment for Oriented Object Detection in Remote Sensing](object_detection/fourier_angle_alignment_for_oriented_object_detection_in_remote_sensing.md)**

:   利用傅里叶旋转等变性在频域估计目标主方向并对齐特征，提出 FAAFusion 和 FAA Head 两个即插即用模块分别解决 FPN 跨尺度方向不一致和检测头分类-回归任务冲突，在 DOTA-v1.0/v1.5 和 HRSC2016 上取得新 SOTA。

**[HeROD: Heuristic-inspired Reasoning Priors Facilitate Data-Efficient Referring Object Detection](object_detection/herod_heuristic_inspired_reasoning_data_efficient_rod.md)**

:   HeROD 提出了一种轻量级、模型无关的框架，通过将启发式空间和语义推理先验注入 DETR 风格检测管道的三个阶段（候选排序、预测融合、匈牙利匹配），在标注稀缺条件下显著提升指代目标检测(ROD)的数据效率和收敛性能。

**[Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](object_detection/learning_multi-modal_prototypes_for_cross-domain_few-shot_object_detection.md)**

:   提出双分支框架 LMP，在 GroundingDINO 基础上引入视觉原型分支（正类原型+硬负原型），与文本分支联合训练并集成推理，在跨域少样本目标检测中取得 SOTA。

**[Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](object_detection/mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)**

:   提出 InCoM-Net，通过从 VLM 特征中为每个实例分别提取实例内、实例间和全局三层上下文特征，并通过渐进式上下文聚合与检测器特征融合，在 HICO-DET 和 V-COCO 上取得 HOI 检测 SOTA（HICO-DET Full mAP 43.96，V-COCO AP_role^S1 73.6）。

**[Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](object_detection/mitigating_memorization_in_texttoimage_diffusion_v.md)**

:   提出训练时区域感知提示增强(RAPTA)和注意力驱动多模态复制检测(ADMCD)两个互补模块，前者通过目标检测器proposal生成语义接地的提示变体来缓解扩散模型训练数据记忆化，后者融合patch级/CLIP/纹理三流特征实现零训练复制检测与分类，在LAION-10k上将复制率从7.4降至2.6。

**[MonoSAOD: Monocular 3D Object Detection with Sparsely Annotated Label](object_detection/monosaod_monocular_3d_object_detection_with_sparsely_annotated_label.md)**

:   首次定义并解决稀疏标注单目 3D 目标检测问题，提出道路感知补丁增强（RAPA）和原型过滤（PBF）两个模块，在 KITTI 30% 标注设置下大幅超越现有 2D SAOD 方法（AP3D Easy: 21.28 vs 17.14）。

**[MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image Understanding](object_detection/mrd_multi-resolution_retrieval-detection_fusion_for_high-resolution_image_unders.md)**

:   提出 MRD，一个 training-free 的多分辨率检索-检测融合框架，通过多分辨率语义融合缓解目标碎片化，结合开放词汇检测器抑制背景干扰，显著提升 MLLM 对高分辨率图像的理解能力。

**[NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection](object_detection/noovd_novel_category_discovery_and_embedding_for_open-vocabulary_object_detectio.md)**

:   提出NoOVD框架，在基于冻结VLM的OVD训练中通过无参数K-FPN保留CLIP知识来发现潜在新类别目标、通过自蒸馏将新类别知识嵌入检测器、通过R-RPN在推理时提升新类别召回率，在OV-LVIS/OV-COCO/Objects365上取得SOTA。

**[PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection](object_detection/paq-detr_learning_pattern_and_quality-aware_dynamic_queries_for_object_detection.md)**

:   PaQ-DETR 提出基于共享模式的动态查询生成（内容感知权重组合共享基模式）+ 质量感知一对多分配（基于定位-分类一致性自适应选择正样本），统一解决DETR中的查询表示和监督不均衡问题，在多个backbone上稳定提升1.5%-4.2% mAP。

**[Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection](object_detection/parameter-efficient_semantic_augmentation_for_enhancing_open-vocabulary_object_d.md)**

:   HSA-DINO 提出多尺度 prompt bank 从图像特征金字塔中学习层次化语义 prompt 增强文本表示，并通过语义感知路由器在推理时动态决定是否使用领域特定增强，实现了领域适配与开放词汇泛化的优越平衡（H 值在三个垂直领域数据集上均为最优）。

**[PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training](object_detection/pet-dino_unifying_visual_cues_into_grounding_dino_with_prompt-enriched_training.md)**

:   PET-DINO 在 Grounding DINO 基础上构建了一个同时支持文本和视觉提示的通用目标检测器，设计了对齐友好的视觉提示生成模块（AFVPG）以及两种提示丰富化训练策略（IBP 和 DMD），在零样本检测任务上以更少的训练数据取得了有竞争力的性能。

**[PHAC: Promptable Human Amodal Completion](object_detection/phac_promptable_human_amodal_completion.md)**

:   提出可提示人体非模态补全（PHAC）新任务，通过基于点的用户提示（姿态/边界框）配合 ControlNet 注入条件信号，并设计基于修复的精炼模块保留可见区域外观，实现高质量、可控的遮挡人体图像补全。

**[Prompt-Free Universal Region Proposal Network](object_detection/prompt-free_universal_region_proposal_network.md)**

:   PF-RPN 用可学习视觉嵌入替代文本/图像提示，通过稀疏图像感知适配器、级联自提示和中心性引导查询选择三个模块，仅用 5% COCO 数据训练即可在 19 个跨域数据集上实现 SOTA 零样本区域提案。

**[Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection](object_detection/remedying_target-domain_astigmatism_for_cross-domain_few-shot_object_detection.md)**

:   首次发现跨域少样本目标检测（CD-FSOD）中模型注意力在目标域持续分散的"散光"现象，受人类中央凹视觉系统启发，设计正向模式精化（PPR）、负向上下文调制（NCM）和文本语义对齐（TSA）三个互补模块来重塑注意力，在6个跨域基准上以显著优势达到SOTA。

**[Saliency-R1: Enforcing Interpretable and Faithful Vision-language Reasoning via Saliency-map Alignment Reward](object_detection/saliency-r1_enforcing_interpretable_and_faithful_vision-language_reasoning_via_s.md)**

:   提出 Saliency-R1，通过基于 logit 分解的高效显著性图技术和思维链瓶颈注意力回溯，将显著性图与人工标注 bounding box 的对齐度作为 GRPO 奖励，训练 VLM 在推理时聚焦任务相关的图像区域，提升推理的可解释性和忠实性。

**[SDF-Net: Structure-Aware Disentangled Feature Learning for Optical–SAR Ship Re-Identification](object_detection/sdf-net_structure-aware_disentangled_feature_learning_for_opticall-sar_ship_re-i.md)**

:   提出 SDF-Net，利用船舶刚体几何结构作为跨模态不变锚点，在中间层提取梯度能量强制结构一致性，在终端层解耦模态共享/特定特征并通过加法残差融合，在 HOSS-ReID 上取得 SOTA（All mAP 60.9%，超 TransOSS 3.5%）。

**[Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](object_detection/show_dont_tell_detecting_novel_objects_by_watching.md)**

:   提出 "Show, Don't Tell" 范式——通过观看人类演示视频自动创建训练数据集并训练定制化物体检测器，完全绕过语言描述和提示工程，在真实机器人场景中显著超越 SOTA 开集/闭集检测器的新物体识别能力。

**[Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](object_detection/small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)**

:   提出 ESM-YOLO+，一个轻量级可见光-红外融合小目标检测网络，通过 Mask-Enhanced Attention Fusion (MEAF) 模块实现像素级跨模态自适应融合，并引入训练时结构表示增强提升空间判别力，在 VEDAI 上达 84.71% mAP 同时参数量减少 93.6%。

**[SPAN: Spatial-Projection Alignment for Monocular 3D Object Detection](object_detection/span_spatial-projection_alignment_for_monocular_3d_object_detection.md)**

:   提出 Spatial-Projection Alignment (SPAN)，通过3D角点空间对齐和3D-2D投影对齐两个几何协同约束，配合分层任务学习策略，作为即插即用模块提升任意单目3D检测器的定位精度。

**[SpiralDiff: Spiral Diffusion with LoRA for RGB-to-RAW Conversion Across Cameras](object_detection/spiraldiff_spiral_diffusion_with_lora_for_rgb-to-raw_conversion_across_cameras.md)**

:   提出 SpiralDiff，一种面向 RGB-to-RAW 转换的扩散框架，通过信号依赖的噪声加权策略适应不同像素强度区域的重建难度，并引入 CamLoRA 模块实现单一模型跨多相机的轻量适配。

**[The COTe Score: A Decomposable Framework for Evaluating Document Layout Analysis Models](object_detection/the_cote_score_a_decomposable_framework_for_evaluating_document_layout_analysis_.md)**

:   提出面向文档布局分析（DLA）的可分解评估框架 COTe（Coverage, Overlap, Trespass, Excess），以及结构语义单元 SSU，相比传统 IoU/mAP/F1 能更准确地反映页面解析质量，并揭示不同模型的特异性失败模式。

**[Toward Generalizable Whole Brain Representations with High-Resolution Light-Sheet Data](object_detection/toward_generalizable_whole_brain_representations_with_high-resolution_light-shee.md)**

:   提出 CANVAS——首个大规模亚细胞分辨率光片荧光显微镜（LSFM）全脑基准数据集，涵盖 6 种细胞标记物、约 93,000 个细胞标注和公开排行榜，揭示了现有检测模型在跨标记物和跨脑区泛化上的严重不足，并探索了 3D 掩码自编码器（MAE）的自监督表示学习潜力。

**[Towards Intrinsic-Aware Monocular 3D Object Detection](object_detection/towards_intrinsic-aware_monocular_3d_object_detection.md)**

:   MonoIA 提出将数值型相机内参转化为语言引导的语义表征（通过 LLM 生成内参描述 + CLIP 编码），并通过分层自适应模块将其融入检测网络，实现对未见焦距的零样本泛化和跨数据集统一训练，在 KITTI/Waymo/nuScenes 上达到新 SOTA。

**[UAVGen: Visual Prototype Conditioned Focal Region Generation for UAV-Based Object Detection](object_detection/uavgen_visual_prototype_conditioned_focal_region_generation_for_uav_based_object_detection.md)**

:   提出 UAVGen，一个面向无人机目标检测的 layout-to-image 数据增强框架，通过视觉原型条件扩散模型和焦点区域增强管线解决小目标生成质量低、模型容量浪费和标签不一致问题。

---

## 🤖 机器人/具身智能 { #robotics }

**[Action–Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation](robotics/actiongeometry_prediction_with_3d_geometric_prior.md)**

:   利用预训练3D几何基础模型π3作为感知骨干，融合3D几何、2D语义和本体感知特征，通过扩散模型联合预测未来动作chunk和未来3D Pointmap，仅使用RGB输入就在RoboTwin双臂基准上全面超越点云方法。

**[Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](robotics/adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)**

:   提出自适应动作分块(AAC)策略，利用动作熵作为线索在推理时动态确定最优分块大小，无需额外训练或架构修改，在RoboCasa和LIBERO等基准上持续提升GR00T N1.5和π0.5的任务成功率。

**[AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots](robotics/atomicvla_unlocking_the_potential_of_atomic_skill_learning_in_robots.md)**

:   提出AtomicVLA，在π₀基础上构建统一规划-执行框架，通过自适应Think-Act切换生成原子技能抽象，并用技能引导的MoE（SG-MoE）将动作路由到专精expert执行，LIBERO-LONG成功率从85.2%提升至95.2%（+10%），真实Franka长任务+18.3%，持续学习+21%。

**[BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration](robotics/bipremanip_learning_affordance-based_bimanual_preparatory_manipulation_through_a.md)**

:   提出 BiPreManip 框架，基于视觉可供性表示实现双臂预备操作：先预想主手的目标交互区域，再引导辅助手进行预备动作（如翻转瓶子使瓶盖朝向主手），在仿真和真实环境中大幅优于基线。

**[Boosting Vision-Language-Action Finetuning with Feasible Action Neighborhood Prior](robotics/boosting_vision-language-action_finetuning_with_feasible_action_neighborhood_pri.md)**

:   提出可行动作邻域（FAN）正则化器，将 VLA 模型的输出分布塑造为与物理动作容差匹配的高斯形状，在 SFT 和 RFT 两种微调范式下均显著提升成功率、泛化性和样本效率（RFT 仅需 1/3 训练步数达到 90% 成功率）。

**[Chain of World: World Model Thinking in Latent Motion (CoWVLA)](robotics/chain_of_world_world_model_thinking_in_latent_motion.md)**

:   提出CoWVLA，统一世界模型VLA和隐动作VLA的优势：通过Latent Motion Extractor将视频分解为结构隐变量和运动隐变量，VLA在隐运动空间做世界模型预测而非重建冗余像素，配合Co-Fine-tuning交替生成关键帧和动作token，LIBERO-LONG达95.2%超越π₀(85.2%)，SimplerEnv-WidowX avg 0.560超π₀(0.425)。

**[CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning](robotics/como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)**

:   提出 CoMo，通过早期时序差分（Td）和时序对比学习（Tcl）两个机制协同解决连续隐运动学习中的捷径学习问题，从互联网视频中提取精细的连续伪动作标签，使视频数据与机器人动作在统一连续分布下联合训练，显著提升策略性能。

**[Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning](robotics/cross-domain_demo-to-code_via_neurosymbolic_counterfactual_reasoning.md)**

:   提出 NeSyCR 神经符号反事实推理框架，将视频示教抽象为符号世界模型，通过反事实状态推演检测跨域不兼容并自动修正程序步骤，在跨域 demo-to-code 任务上比最强基线 Statler 提升 31.14% 成功率。

**[CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception and Understanding](robotics/cyclemanip_enabling_cyclic_task_manipulation_via_effective_historical_percepti.md)**

:   CycleManip 首次系统性地解决机器人循环操作任务（如摇瓶子N次），通过成本感知的历史采样策略增强历史感知，配合多任务学习辅助目标提升历史理解，以端到端模仿学习方式实现循环次数可控的操作。

**[DAWN: Pixel Motion Diffusion is What We Need for Robot Control](robotics/dawn_pixel_motion_diffusion_robot_control.md)**

:   提出 DAWN，一个两阶段全扩散的视觉语言动作框架——Motion Director（潜扩散模型）生成稠密像素运动场作为可解释的中间表示，Action Expert（扩散 Transformer 策略）将像素运动转换为可执行机器人动作；在 CALVIN 基准上取得 SOTA（平均长度 4.00），并在真实世界单臂/双臂操控中展现强泛化能力。

**[DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](robotics/decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)**

:   提出 DecoVLN 框架，将 VLN 任务中的观察、推理和纠错三个过程解耦，通过自适应记忆优化机制和基于状态-动作对的纠错微调策略，在仅使用自中心 RGB 输入的条件下实现了 R2R-CE 和 RxR-CE 上的 SOTA 性能。

**[DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning](robotics/deepsketcher_internalizing_visual_manipulation_for_multimodal_reasoning.md)**

:   提出DeepSketcher套件——包含31k高质量代码渲染的图文交错CoT数据集和一个自包含的Embedding Editor模型，使VLM无需外部工具即可在视觉嵌入空间直接生成"视觉思考"进行多模态推理。

**[Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols](robotics/diagnose_correct_and_learn_from_manipulation_failures_via_visual_symbols.md)**

:   提出 ViFailback 框架，利用可视化符号（箭头、准星、标签等）高效标注真实世界机器人操作失败，构建 58,128 个 VQA 对的数据集，并训练 ViFailback-8B VLM 实现失败诊断和视觉+文本纠正指导，集成到 VLA 后实现 22.2% 的任务成功率提升。

**[Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](robotics/expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)**

:   针对MoE-LoRA方法中所有expert结构相同（统一rank）导致无法适配不同复杂度任务的问题，提出EPT：通过共享meta-knowledge子空间 + 不同kernel size的反卷积expert构建参数金字塔，配合Adaptive LoRA Pruner和对比学习Task Embedding，在GLUE上以仅0.41M参数/任务达到87.0%平均分，超越所有MoE-LoRA变体。

**[Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning](robotics/fast-thinkact_efficient_vision-language-action_reasoning_via_verbalizable_latent.md)**

:   提出 Fast-ThinkAct，通过将冗长的文本 CoT 推理（~250 token）压缩为 6 个可语言化的连续 latent token，结合 reward-guided preference distillation 和 visual trajectory alignment，实现 89.3% 推理延迟降低（9.3× faster than ThinkAct-7B）同时保持甚至超越 SOTA reasoning VLA 的性能。

**[FineCog-Nav: Integrating Fine-grained Cognitive Modules for Zero-shot Multimodal UAV Navigation](robotics/finecog_nav_fine_grained_cognitive_modules_for_zero_shot_uav_navigation.md)**

:   本文提出 FineCog-Nav，一个受人类认知启发的零样本 UAV 视觉语言导航框架，将导航分解为语言处理、感知、注意力、记忆、想象、推理和决策七个细粒度认知模块，每个模块使用中等规模基础模型驱动，无需训练即可在复杂 3D 环境中完成长程导航。

**[FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](robotics/force_transferable_visual_jailbreaking_attacks_via_feature_over_reliance_correct.md)**

:   分析发现视觉 jailbreak attack 迁移性差的根因是 attack 处于 high-sharpness loss region——源于浅层特征过度依赖 model-specific 表示和高频信息过度影响；提出 FORCE 方法通过 layer-aware regularization 扩展浅层 feasible region + spectral rescaling 抑制高频非语义成分，引导 attack 进入 flatter loss landscape，显著提升跨模型迁移性。

**[ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation](robotics/forcevla2_unleashing_hybrid_force-position_control_with_force_awareness_for_cont.md)**

:   提出ForceVLA2，首个在VLA框架中统一力感知(force awareness)与混合力-位置控制(hybrid force-position control)的端到端模型：通过Force-based Prompts在VLM中构建跨阶段力感知任务概念，Cross-Scale MoE自适应融合任务语义与实时交互力实现闭环力-位置调节，在5个contact-rich任务上平均成功率66%，超π₀和π₀.5分别48.0%和35.0%。

**[GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](robotics/gecosrt_geometryaware_continual_adaptation_for_rob.md)**

:   GeCo-SRT提出首个持续跨任务Sim-to-Real迁移范式，利用局部几何特征的域不变性和任务不变性，通过Geo-MoE模块提取可复用的几何知识并用Geo-PER防止专家级遗忘，在4个真实机器人任务上平均成功率63.3%（比基线提升52%），且仅需1/6数据即可匹配基线性能。

**[IGen: Scalable Data Generation for Robot Learning from Open-World Images](robotics/igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)**

:   IGen 从单张开放世界图像出发，通过3D场景重建→VLM任务规划→SE(3)动作生成→点云合成→帧渲染，自动生成大规模视觉-动作训练数据，仅用生成数据训练的策略即可完成真实世界操作。

**[Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](robotics/influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)**

:   本文揭示线性化注意力机制在 NTK 框架下不收敛至无穷宽极限，并提出"影响力可塑性"(influence malleability) 度量，证明注意力的强大能力与对抗脆弱性共享同一来源——偏离核regime的数据依赖核结构。

**[Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](robotics/lada_robotic_manipulation.md)**

:   提出 LaDA 框架，用自然语言作为语义桥梁将连续 7-DoF 动作解耦为平移/旋转/夹爪三个可解释原语，通过软标签对比学习在共享嵌入空间中对齐跨任务动作表示，仅 0.6B 参数在 LIBERO 上达 93.6% 成功率，超越 1.3B~8.5B 参数的所有基线。

**[Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](robotics/learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)**

:   提出 TVVE 框架，通过强化学习驱动的多视角探索策略（MVEP）选择最优虚拟相机视角并在线重渲染观测，同时设计任务感知 MoE 视觉编码器（TaskMoE）解决多任务特征干扰问题，在 RLBench 18 个任务上平均成功率达 86.6%。

**[ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation](robotics/maniparena_comprehensive_real-world_evaluation_of_reasoning-oriented_generalist_.md)**

:   ManipArena 提出了一个标准化的真实世界机器人操作评估框架，包含 20 个推理导向任务和 10,812 条专家轨迹，通过绿幕受控环境、系统化多样性设计和分层 OOD 评估，为 VLA 模型和世界模型提供公平、可复现的评测基准。

**[MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent](robotics/mergevla_cross-skill_model_merging_toward_a_generalist_vision-language-action_ag.md)**

:   首次系统诊断 VLA 模型不可合并的两大根因（LoRA 自私参数冲突 + 动作专家自注意力导致的任务耦合），提出 MergeVLA——通过任务掩码稀疏激活 LoRA、去自注意力动作专家、无训练测试时路由，将多个单技能 VLA 专家合并为一个通用 agent，在 LIBERO 上达 90.2% 成功率，真机 SO101 达 90%。

**[PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](robotics/palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)**

:   提出 PALM，一个统一的 VLA 框架，通过结构化的细粒度可供性预测（全局/局部/空间/动态四类）作为隐式推理锚点，结合连续子任务进度估计实现无缝任务切换，在 CALVIN ABCD 上平均完成长度达 4.48（超越前 SOTA 12.5%），LIBERO-LONG 成功率 91.8%，真实世界长时域泛化测试中达到基线 2 倍以上。

**[PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](robotics/panoaffordancenet_towards_holistic_affordance_grounding_in_360_indoor_environmen.md)**

:   提出首个面向360°全景室内环境的整体affordance定位框架PanoAffordanceNet，通过畸变感知频谱调制器(DASM)和全球面稠密化头(OSDH)系统性解决ERP几何畸变、稀疏功能区域和语义漂移问题，并构建了首个全景affordance数据集360-AGD。

**[Pixel-level Scene Understanding in One Token: Visual States Need What-is-Where Composition](robotics/pixel-level_scene_understanding_in_one_token_visual_states_need_what-is-where_co.md)**

:   本文提出CroBo，一个通过全局-局部重建目标学习视觉状态表示的自监督框架：将全局参考图压缩为单个瓶颈token，用它来重建高度遮蔽（90%）的局部裁剪视图，迫使瓶颈token编码像素级的"what-is-where"场景组成信息，在Franka Kitchen和DMC机器人策略学习benchmark上达到SOTA。

**[Probabilistic Concept Graph Reasoning for Multimodal Misinformation Detection](robotics/probabilistic_concept_graph_reasoning_for_multimodal_misinformation_detection.md)**

:   本文将多模态虚假信息检测（MMD）重构为基于概念图的结构化概率推理问题，提出PCGR框架，通过MLLM自动发现并验证人类可理解的概念节点，构建层次化概率概念图，实现可解释的虚假信息检测，在三个基准上全面超越13个baseline。

**[ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation](robotics/profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)**

:   提出 ProFocus，一个 training-free 框架，通过推理引导的主动感知（构建语义地图并迭代生成定向视觉查询）和分支多样化蒙特卡洛树搜索（BD-MCTS，筛选 top-k 高价值路点实现聚焦推理），在 R2R 和 REVERIE 上达到零样本 VLN 的 SOTA。

**[PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors for Embodied Multi-Sensory Learning](robotics/pulse_privileged_knowledge_transfer_from_rich_to_deployable_sensors_for_embodied.md)**

:   本文提出 PULSE 框架，通过冻结的特权传感器（如 EDA）教师模型向廉价可部署传感器（如 ECG、BVP、加速度计）学生模型进行知识蒸馏，引入共享-私有嵌入分解和重建防崩塌机制，在不使用 EDA 推理的情况下达到 0.994 AUROC 的压力检测性能，甚至超越使用全部传感器的模型。

**[QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](robotics/quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)**

:   提出 QuantVLA，首个面向 Vision-Language-Action (VLA) 模型的免训练后量化框架，通过选择性量化布局和两个轻量级标定机制（注意力温度匹配 ATM 和输出头平衡 OHB），在 W4A8 精度下实现约 70% 的内存节省，同时任务成功率超过全精度基线。

**[RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation](robotics/rc-nf_robot-conditioned_normalizing_flow_for_real-time_anomaly_detection_in_robo.md)**

:   提出 Robot-Conditioned Normalizing Flow (RC-NF)，通过条件归一化流对机器人状态与物体运动轨迹的联合分布建模，实现 <100ms 实时异常检测，可作为 VLA 模型（如 π₀）的即插即用监控模块，支持任务级重规划和状态级轨迹回滚。

**[SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](robotics/sapave_active_perception_manipulation_vla_roboti.md)**

:   SaPaVe提出端到端主动操作框架，通过解耦相机动作和操作动作的底层到顶层训练策略，先用20万对语义相机控制数据学习主动感知先验，再联合优化实现主动操作，在真实世界中超越π₀和GR00T N1达31.25%成功率提升。

**[STRNet: Visual Navigation with Spatio-Temporal Representation through Dynamic Graph Aggregation](robotics/strnet_visual_navigation_with_spatio-temporal_representation_through_dynamic_gra.md)**

:   STRNet 提出统一的时空表征框架用于视觉导航，通过图推理模块建模帧内空间拓扑结构，结合混合时间偏移和多分辨率差分卷积建模时序动态，显著提升了目标条件导航的成功率（比 NoMaD 提升 70%）。

**[Test-time Ego-Exo-centric Adaptation for Action Anticipation via Multi-Label Prototype Growing and Dual-Clue Consistency](robotics/test-time_ego-exo-centric_adaptation_for_action_anticipation_via_multi-label_pro.md)**

:   首次提出 Test-time Ego-Exo Adaptation for Action Anticipation（TE2A3）任务，设计 DCPGN 网络通过多标签原型增长和双线索（视觉+文本）一致性，在测试时将源视角训练模型在线适配到目标视角进行动作预测，大幅超越现有 TTA 方法。

**[Towards Open Environments and Instructions: General Vision-Language Navigation via Fast-Slow Interactive Reasoning](robotics/towards_open_environments_and_instructions_general_vision-language_navigation_vi.md)**

:   针对开放环境下视觉语言导航（GSA-VLN）任务，受人类快慢认知双系统启发，提出 slow4fast-VLN 框架：快推理模块基于端到端策略网络实时导航并积累历史记忆，慢推理模块借助 LLM 反思生成结构化泛化经验，经验通过注意力融合反馈增强快推理网络，实现在未见环境和多样指令下的持续适应，在 GSA-R2R 数据集上全面超越前 SOTA（GR-DUET）。

**[Towards Training-Free Scene Text Editing](robotics/towards_training-free_scene_text_editing.md)**

:   提出TextFlow，一个免训练的场景文字编辑框架，通过在去噪早期阶段使用Flow Manifold Steering（FMS）保持风格一致性、后期阶段使用Attention Boost（AttnBoost）增强文字渲染准确性，在不需要任务特定训练的情况下达到与训练方法可比甚至更优的编辑质量。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](self_supervised/a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)**

:   提出 PL-Stitch 自监督框架，利用 Plackett-Luce 概率排序模型将视频帧的时序排序作为预训练信号，学习具有"程序感知"能力的视频表示，在手术阶段识别和烹饪动作分割上全面超越现有自监督方法。

**[AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](self_supervised/actta_rethinking_test-time_adaptation_via_dynamic_activation.md)**

:   提出AcTTA框架，首次将激活函数作为测试时适应(TTA)的可学习组件，通过参数化的激活中心偏移 $c$ 和非对称梯度缩放 $\lambda_{pos}, \lambda_{neg}$ 替代或增强传统归一化层适应，在CIFAR-10/100-C和ImageNet-C上一致超越所有归一化基TTA方法，并支持10倍大的学习率。

**[An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning](self_supervised/an_optimal_transport_driven_approach_for_cultivating_latent_space_in_online_incr.md)**

:   提出基于最优传输理论的在线混合模型学习框架 (MMOT)，通过为每个类别维护多个自适应质心来更精确地表征在线数据流的多模态特性，结合动态保持策略增强类别区分能力，在在线类增量学习 (OCIL) 中有效缓解灾难性遗忘。

**[BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning](self_supervised/bd-merging_bias-aware_dynamic_model_merging_with_evidence-guided_contrastive_lea.md)**

:   提出 BD-Merging 框架，通过 Dirichlet 证据建模 + 邻域差异分数（ADS）+ 差异感知对比学习，训练去偏路由器来自适应分配模型合并权重，显著提升合并模型在测试时分布偏移和未见任务上的鲁棒性与泛化能力。

**[BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](self_supervised/boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)**

:   提出 BoSS——一种可扩展的 oracle 策略选择框架：在每轮主动学习中，并行运行多种查询策略在随机子池上生成候选 batch，通过冻结 backbone 仅重训最后一层快速评估每个候选 batch 的性能增益，选出最优 batch，从而量化现有 AL 策略与理论最优之间的差距。

**[Breaking the Tuning Barrier: Zero-Hyperparameters Yield Multi-Corner Analysis Via Learned Priors](self_supervised/breaking_the_tuning_barrier_zerohyperparameters_yi.md)**

:   提出用基础模型 TabPFN 的 learned prior 替代传统人工先验（GP 核、IS 高斯假设），实现零超参数调优的多 PVT Corner 良率分析，在工业级 SRAM 基准上达到 SOTA 精度（MRE 低至 0.11%）的同时提速超 10×。

**[Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](self_supervised/com_pt_chain_of_models_pretraining.md)**

:   提出 Chain-of-Models Pre-Training (CoM-PT)，将视觉基础模型按大小排列形成"模型链"，通过从小到大的逆向知识转移（权重初始化+特征蒸馏）逐步加速训练，实现性能无损的训练加速且效率随模型家族规模增长而提升。

**[CraterBench-R: Instance-Level Crater Retrieval for Planetary Scale](self_supervised/craterbench-r_instance-level_crater_retrieval_for_planetary_scale.md)**

:   首次将陨石坑分析形式化为实例级图像检索问题——提出CraterBench-R基准(~25K火星陨石坑ID, 50K gallery, 5K查询)，诊断发现单向量池化有精度上限+有监督度量学习反而退化，提出无训练的实例token聚合(选K个种子+余弦最近邻残差分配)将196个ViT patch token压缩为K个代表token做late interaction匹配，K=64时匹配全token精度且存储大幅降低，实用两阶段管线(单向量粗筛+实例token精排)恢复89-94%完整精度。

**[D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping](self_supervised/d2dewarp_dual_dimensions_geometric_representation_learning_based_document_image_.md)**

:   提出 D2Dewarp——首个从水平和垂直双维度学习文档几何表示的去畸变方法：UNet 双解码器分别预测水平线（文档/表格/文本行的上下边界）和垂直线（左右边界），HV Fusion Module 通过混合注意力交叉融合两个方向的特征，并构建了包含 114K 张图的 DocDewarpHV 数据集提供双维度标注。

**[DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](self_supervised/diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)**

:   通过系统分析发现 DiT 各 block 间的表示多样性是有效学习的关键因素，提出 DiverseDiT：用长残差连接多样化输入 + 表示多样性损失显式促进 block 间特征差异化，无需外部引导模型即可加速收敛并提升生成质量。

**[GeoBridge: A Semantic-Anchored Multi-View Foundation Model for Geo-Localization](self_supervised/geobridge_semantic-anchored_multi-view_foundation_model_for_geo-localization.md)**

:   GeoBridge 提出语义锚定的多视角地理定位基础模型，将无人机/街景/卫星影像通过文本描述构建跨模态语义桥梁，实现双向跨视角匹配和语言到图像定位，并构建了GeoLoc数据集（50K+对，36国）。

**[GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](self_supervised/geochemad_benchmarking_unsupervised_geochemical_anomaly_detection_for_mineral_ex.md)**

:   提出 GeoChemAD 开源基准数据集和 GeoChemFormer 框架，通过空间上下文学习与元素依赖建模实现无监督地球化学异常检测，在8个子集上平均 AUC 达到 0.7712。

**[Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning](self_supervised/group_dinomics_incorporating_people_dynamics_into_dino_for_self_supervised_group_activity_feature_learning.md)**

:   提出利用 DINOv3 结合两个自监督预训练任务（人物光流估计和群体相关物体定位）来学习群体活动特征（GAF），在无群体活动标注的情况下大幅超越现有方法。

**[LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency](self_supervised/las-comp_zero-shot_3d_completion_with_latent-spatial_consistency.md)**

:   提出 LaS-Comp，一种零样本、类别无关的 3D 形状补全框架，通过 Explicit Replacement Stage 在空间域注入已知几何 + Implicit Alignment Stage 在隐空间梯度优化边界一致性，桥接了预训练 3D 基础模型的隐空间与空间域之间的 gap，在多种部分观测模式下达到 SOTA。

**[MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](self_supervised/mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)**

:   提出 Mine-JEPA，首个面向侧扫声纳（SSS）水雷分类的域内自监督学习流水线——基于 SIGReg 正则化损失、声纳适配增强策略和 ImageNet 初始化，仅用 1,170 张未标注声纳图像预训练即超越了在 17 亿图像上预训练的 DINOv3 基础模型。

**[MOMO: Mars Orbital Model — Foundation Model for Mars Orbital Applications](self_supervised/momo_mars_orbital_model_foundation_model_for_mars_orbital_applications.md)**

:   MOMO 是首个火星遥感基础模型，通过在三种火星传感器（HiRISE/CTX/THEMIS）上分别预训练 MAE 并提出 Equal Validation Loss（EVL）检查点选择策略进行模型融合，在 Mars-Bench 的 9 个下游任务上超越 ImageNet 预训练和地球观测基础模型。

**[OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism](self_supervised/omnigcd_abstracting_generalized_category_discovery_for_modality_agnosticism.md)**

:   提出 OmniGCD，首个模态无关的广义类别发现方法，利用合成数据训练的 GCDformer 在测试时将任意模态的 GCD 潜空间变换为更适合聚类的表示，在 16 个跨四种模态的数据集上实现零样本 GCD。

**[Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting](self_supervised/redepth_anything_test-time_depth_refinement_via_self-supervised_re-lighting.md)**

:   提出 Re-Depth Anything，通过在推理时对预测深度图进行重光照增强并利用 2D 扩散模型的 SDS 损失进行自监督优化，在无标签的情况下精细化 Depth Anything V2/3 的深度预测。

**[Representation Learning for Spatiotemporal Physical Systems](self_supervised/representation_learning_for_spatiotemporal_physica.md)**

:   在三个 PDE 物理系统（活性物质、剪切流、Rayleigh-Bénard 对流）上系统比较四种自监督/物理建模方法，发现隐空间预测（JEPA）在物理参数估计任务上全面优于像素级预测（VideoMAE）——MSE 相对改善 28%~51%，且 10% 微调数据即可超越 VideoMAE 的 100% 数据表现。同时，专为物理建模设计的方法并非总是最优选择。

**[Robustness of Vision Foundation Models to Common Perturbations](self_supervised/robustness_of_vision_foundation_models_to_common_perturbations.md)**

:   首次系统研究视觉基础模型对常见扰动（JPEG 压缩、亮度调节等）的鲁棒性，提出三种鲁棒性度量并形式化五个数学性质，发现基础模型普遍不鲁棒，并提出微调方法改善鲁棒性而不牺牲效用。

**[Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild](self_supervised/shape-of-you_fused_gromov-wasserstein_optimal_transport_for_semantic_corresponde.md)**

:   将语义对应问题重新建模为 Fused Gromov-Wasserstein (FGW) 最优传输问题，利用 3D 基础模型提供的几何结构约束来生成全局一致的伪标签，解决了传统最近邻匹配因局部性和 2D 外观歧义导致的几何不一致问题。

**[SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks](self_supervised/sphor_a_representation_learning_perspective_on_open-set_recognition_for_identify.md)**

:   提出 SpHOR，一种两阶段解耦训练的开放集识别方法，通过球面表示学习（vMF 分布）、正交标签嵌入和 Mixup/Label Smoothing 集成，显式塑造特征空间以更好地分离已知/未知类别，在 Semantic Shift Benchmark 上取得最高 5.1% 的 OSCR 提升。

**[Suppressing Non-Semantic Noise in Masked Image Modeling Representations](self_supervised/suppressing_non-semantic_noise_in_masked_image_modeling_representations.md)**

:   本文揭示了掩码图像建模（MIM）学到的表征中保留了大量非语义信息（如纹理、颜色等底层特征），并提出了一种无需训练的后处理方法 SOAP（Semantically Orthogonal Artifact Projection），通过 PCA 识别并投影去除非语义成分，在多种 MIM 模型上一致提升零样本性能。

**[TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](self_supervised/talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)**

:   提出 TALO，一种基于 Thin Plate Spline 的高自由度对齐框架，通过全局传播控制点和点无关的子图配准设计，纠正 3D 视觉基础模型在在线重建中的空间变化不一致性，兼容多种基础模型和相机配置，在 Waymo/nuScenes 数据集上显著降低轨迹误差。

**[TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation](self_supervised/teflow_enabling_multi-frame_supervision_for_self-supervised_feed-forward_scene_f.md)**

:   提出TeFlow——首个将多帧监督引入自监督前馈场景流估计的方法：通过时序集成策略构建运动候选池并基于共识投票聚合时序一致的监督信号，在Argoverse 2上Three-way EPE达3.57cm（媲美优化方法Floxels）同时保持实时推理（8s vs 24min），较SeFlow++提升22.3%。

**[Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](self_supervised/text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)**

:   提出TPSNet，将CLIP学习的域提示（domain prompt）作为文本先验提供精细语义监督，同时引入相位谱特征作为相位先验来桥接域分布差异并保持语义完整性，通过文本-相位双先验的协同实现无监督跨域图像检索的显著提升。

**[TrackMAE: Video Representation Learning via Track, Mask, and Predict](self_supervised/trackmae_video_representation_learning_via_track_mask_and_predict.md)**

:   在masked video modeling（MVM）框架中引入显式的运动信号：使用CoTracker3提取点轨迹作为额外的重建目标，并设计运动感知遮掩策略，联合学习空间重建和运动预测，在运动敏感基准（SSv2、FineGym）上显著超越现有视频自监督方法。

**[UniGeoCLIP: Unified Geospatial Contrastive Learning](self_supervised/unigeoclip_geospatial_contrastive.md)**

:   UniGeoCLIP 首次通过纯对比学习将五种互补的地理空间模态（航拍图、街景图、数字表面模型、文本、GPS 坐标）对齐到统一嵌入空间，并提出多尺度坐标编码器提升空间表示能力。

**[Vision Transformers Need More Than Registers](self_supervised/vision_transformers_need_more_than_registers.md)**

:   这篇论文认为 ViT 在标签监督、文本监督和自监督下普遍存在的 dense feature 伪影，本质上不是单纯的 high-norm token 问题，而是模型在粗粒度监督和全局注意力共同作用下学会了用背景 patch 充当全局语义捷径；作者据此提出 LaSt-ViT，用频域稳定性引导的选择性聚合替代原始 CLS 聚合，在 12 个基准上稳定改善定位、分割和开放词汇任务。

**[VT-Intrinsic: Physics-Based Decomposition of Reflectance and Shading using a Single Visible-Thermal Image Pair](self_supervised/vt-intrinsic_physics-based_decomposition_of_reflectance_and_shading_using_a_sing.md)**

:   VT-Intrinsic 利用可见光和热红外图像之间的物理互补关系（未反射的光被吸收变为热量），推导出可见光-热成像强度的序数关系（ordinality）直接对应反射率和光照的序数关系，以此为自监督信号驱动神经网络优化，实现了无需预训练数据的高质量内在图像分解。

**[Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers](self_supervised/zero_ablation_overstates_register_content_dependence_in_dino_vision_transformers.md)**

:   通过三种替换控制实验（均值替换、噪声替换、跨图像洗牌）证明 DINO 系列 ViT 中零消融方法夸大了对 register token 精确内容的依赖性——模型实际只需"合理的 register-like 激活"而非图像特定值。

---

## 🔬 可解释性 { #interpretability }

**[Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP](interpretability/beyond_semantics_disentangling_information_scope_in_sparse_autoencoders_for_clip.md)**

:   提出"信息范围"（information scope）作为SAE特征可解释性的新维度，通过Contextual Dependency Score（CDS）将CLIP的SAE特征分为局部特征（低CDS）和全局特征（高CDS），揭示两类特征在分类、分割、深度估计中的差异化功能角色。

**[CI-ICE: Intrinsic Concept Extraction Based on Compositional Interpretability](interpretability/ciice_intrinsic_concept_extraction_compositional.md)**

:   提出CI-ICE新任务和HyperExpress方法：在双曲空间(Poincaré球)中利用层次建模能力提取可组合的物体级/属性级内在概念，通过Horosphere投影保证概念嵌入空间的可组合性，在UCEBench上概念解耦ACC₁达0.504(较ICE的0.325提升55%)。

**[Cut to the Chase: Training-free Multimodal Summarization via Chain-of-Events](interpretability/cut_to_the_chase_training-free_multimodal_summarization_via_chain-of-events.md)**

:   提出 CoE，一个免训练的多模态摘要框架，通过构建层次事件图（HEG）引导链式事件推理，在8个数据集上超越SOTA视频CoT基线，平均提升 +3.04 ROUGE、+9.51 CIDEr、+1.88 BERTScore。

**[DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](interpretability/dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)**

:   提出 DINO-QPM，一种轻量级可解释性适配器，将冻结的 DINOv2 骨干网络的复杂高维特征转换为对比性的、类无关的可解释表示，通过二次规划进行稀疏特征选择和类级特征分配，在 CUB-2011 和 Stanford Cars 上同时超越了 DINOv2 线性探测的准确率和所有可比方法的可解释性。

**[Draft and Refine with Visual Experts](interpretability/draft_and_refine_with_visual_experts.md)**

:   提出 DnR（Draft and Refine），一个基于问题条件视觉利用度（Visual Utilization）指标的 Agent 框架，量化 LVLM 对视觉证据的实际依赖程度，并通过外部视觉专家（检测/分割/OCR等）的渲染反馈迭代改善视觉定位，减少幻觉。

**[Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing](interpretability/edit-as-act_goal-regressive_planning_for_open-vocabulary_3d_indoor_scene_editing.md)**

:   将开放词汇的3D室内场景编辑重新定义为目标回归规划问题，设计PDDL风格的EditLang符号语言，通过LLM驱动的Planner-Validator循环从目标状态逆向推导最小编辑序列，在63个编辑任务上同时实现指令忠实度（69.1%）、语义一致性（86.6%）和物理合理性（91.7%）三个指标的最佳平衡。

**[ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization](interpretability/ermoe_eigen-reparameterized_mixture-of-experts_for_stable_routing.md)**

:   ERMoE 提出在正交特征基（eigenbasis）中重参数化MoE专家权重，并用特征基分数（cosine similarity）替代传统路由logits，无需辅助负载均衡损失即可实现稳定路由和可解释的专家特化。

**[Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](interpretability/feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)**

:   提出 FASS 基准，通过强制预测不变性过滤、三轴稳定性分解（空间/排序/显著区域）和多类型扰动（几何/光度/压缩），系统评估后验特征归因方法的稳定性，揭示了现有评估体系的根本性缺陷。

**[From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition](interpretability/from_weights_to_concepts_data-free_interpretability_of_clip_via_singular_vector_.md)**

:   本文提出 SITH（Semantic Inspection of Transformer Heads），一个完全无需数据和训练的 CLIP 可解释性框架：直接对注意力头的 Value-Output 权重矩阵做 SVD 分解，然后用自研的 COMP 算法将每个奇异向量解释为语义一致的概念稀疏组合，实现了比现有方法更细粒度的 intra-head 级别可解释性，并支持精准的权重编辑来改善下游性能。

**[Geometry-Guided Camera Motion Understanding in VideoLLMs](interpretability/geometry-guided_camera_motion_understanding_in_videollms.md)**

:   本文揭示了 VideoLLM 在细粒度相机运动原语（pan/tilt/dolly等）识别上几乎等于随机猜测，构建了 CameraMotionDataset（12K 段 × 15 种原子运动）和 CameraMotionVQA benchmark，并提出通过冻结 3DFM（VGGT）提取几何相机线索 + 轻量时序分类器 + structured prompting 注入的 model-agnostic 方案来弥补这一能力缺口。

**[Inside-Out: Measuring Generalization in Vision Transformers Through Inner Workings](interpretability/inside-out_measuring_generalization_in_vision_transformers_through_inner_working.md)**

:   提出基于模型内部电路（circuits）的泛化性能预测指标，包括部署前模型选择的Dependency Depth Bias（DDB）和部署后性能监控的Circuit Shift Score（CSS），分别比现有代理指标的相关性平均提升13.4%和34.1%。

**[Language Models Can Explain Visual Features via Steering](interpretability/language_models_can_explain_visual_features_via_steering.md)**

:   提出通过对VLM视觉编码器进行SAE特征因果干预（steering），在输入空白图像后让语言模型描述其"看到"的视觉概念，从而实现无需评估图像集的可扩展视觉特征自动解释，并提出混合方法Steering-informed Top-k达到SOTA。

**[Measuring the (Un)Faithfulness of Concept-Based Explanations](interpretability/measuring_the_unfaithfulness_of_concept-based_explanations.md)**

:   本文揭示了现有无监督概念解释方法 (U-CBEMs) 的忠实度被高估——原因是使用了过于复杂的代理模型和有缺陷的删除式评估。作者提出 SURF（Surrogate Faithfulness），一个简单的线性代理 + 双空间度量框架，通过"随机概念应该更不忠实"的 sanity check 验证了其正确性，并首次系统地揭示了多个 SOTA U-CBEMs 实际上并不忠实。

**[Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared](interpretability/missing_no_more_dictionary-guided_cross-modal_image_fusion_under_missing_infrare.md)**

:   提出首个在系数域（而非像素域）进行红外缺失条件下跨模态融合的框架：通过共享卷积字典建立 IR-VIS 统一原子空间，在系数域完成 VIS→IR 推理和自适应融合，配合冻结 LLM 提供弱语义先验进行热信息补全，在仅输入可见光图像的条件下达到接近双模态融合方法的性能。

**[Neurodynamics-Driven Coupled Neural P Systems for Multi-Focus Image Fusion](interpretability/neurodynamics-driven_coupled_neural_p_systems_for_multi-focus_image_fusion.md)**

:   提出 ND-CNPFuse，通过对耦合神经 P (CNP) 系统进行神经动力学分析，建立网络参数与输入信号的约束关系以避免神经元异常持续放电，从而在多焦点图像融合 (MFIF) 任务上无需训练即可生成高质量、可解释的决策图。

**[On the Possible Detectability of Image-in-Image Steganography](interpretability/on_the_possible_detectability_of_image-in-image_steganography.md)**

:   揭示主流 image-in-image 深度隐写方案的根本安全缺陷：嵌入过程本质上是一个混合过程，可被独立成分分析 (ICA) 轻易分离，并提出基于小波域独立成分统计矩的可解释隐写分析方法（仅 8 维特征即达 84.6% 准确率），同时证明经典 SRM+SVM 方法可达 99% 以上检测率。

**[Pixel2Phys: Distilling Governing Laws from Visual Dynamics](interpretability/pixel2phys_distilling_governing_laws_from_visual_dynamics.md)**

:   提出 Pixel2Phys，一个基于 MLLM 的多智能体协作框架，通过 Plan-Variable-Equation-Experiment 四个 Agent 的迭代假设-验证-精化循环，从原始视频中自动发现可解释的物理控制方程，外推精度比基线提升 45.35%。

**[Reallocating Attention Across Layers to Reduce Multimodal Hallucination](interpretability/reallocating_attention_across_layers_to_reduce_multimodal_hallucination.md)**

:   提出一种轻量级、无需训练的插件方法，通过识别感知型和推理型注意力头并进行类别条件缩放（Class-Conditioned Rescaling），重新平衡跨层注意力分配，从而缓解多模态大推理模型（MLRM）中的幻觉问题，在5个基准上平均提升4.2%，几乎无额外推理开销。

**[Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](interpretability/rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)**

:   提出 CBM-Suite 框架，系统性解决概念瓶颈模型的四大缺陷——缺乏概念相关性预评估指标、线性问题导致概念瓶颈被绕过、与黑盒模型的精度差距、以及不同视觉骨干/VLM 影响的研究空白——通过熵度量、非线性层和蒸馏损失显著提升 CBM 的精度与可解释性。

**[RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation](interpretability/riskprop_collision-anchored_self-supervised_risk_propagation_for_early_accident_.md)**

:   提出 RiskProp，一种以碰撞帧为锚点的自监督风险传播范式，通过未来帧正则化损失和自适应单调约束损失，仅依赖碰撞帧标注即可学习时序连贯的风险演化曲线，在 CAP 和 Nexar 数据集上达到 SOTA。

**[SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World](interpretability/safedrive_fine-grained_safety_reasoning_for_end-to-end_driving_in_a_sparse_world.md)**

:   提出 SafeDrive 端到端规划框架，通过轨迹条件化的稀疏世界模型（SWNet）模拟关键实体的未来行为，再由细粒度推理网络（FRNet）进行逐实例碰撞评估和逐时刻可行驶区域合规评估，在 NAVSIM 上 PDMS 达 91.6、仅 0.5% 碰撞率，Bench2Drive 驾驶分 66.8%。

**[SteelDefectX: A Coarse-to-Fine Vision-Language Dataset and Benchmark for Generalizable Steel Surface Defect Detection](interpretability/steeldefectx_a_coarse-to-fine_vision-language_dataset_and_benchmark_for_generali.md)**

:   提出 SteelDefectX，首个面向钢材表面缺陷检测的视觉-语言数据集（7778 张图像、25 类缺陷），包含从类级到样本级的粗到细文本标注，并建立了涵盖纯视觉分类、视觉-语言分类、零/少样本识别和零样本迁移的四任务基准，实验证明高质量文本标注显著提升模型的可解释性、泛化性和跨域迁移能力。

**[SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling](interpretability/subspacead_training-free_few-shot_anomaly_detection_via_subspace_modeling.md)**

:   SubspaceAD 证明了在强视觉基础模型（DINOv2-G）特征上做一次 PCA 拟合就足以超越所有需要训练/记忆库/提示调优的少样本异常检测方法，1-shot 下在 MVTec-AD 上达 98.0% 图像级 AUROC 和 97.6% 像素级 AUROC。

**[TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment](interpretability/tdatr_improving_end-to-end_table_recognition_via_table_detail-aware_learning_and.md)**

:   提出TDATR框架，通过"先感知后融合"策略和结构引导的单元格定位模块，在有限标注数据下实现端到端表格识别，在7个基准上无需数据集特定微调即达到SOTA。

**[Text-guided Fine-Grained Video Anomaly Understanding](interpretability/text-guided_fine-grained_video_anomaly_understanding.md)**

:   提出T-VAU框架，通过异常热力图解码器(AHD)实现像素级时空异常定位，并设计区域感知异常编码器(RAE)将热力图证据注入LVLM进行异常判断、定位和语义解释的统一推理。

**[Towards Faithful Multimodal Concept Bottleneck Models](interpretability/towards_faithful_multimodal_concept_bottleneck_models.md)**

:   提出f-CBM——首个忠实的多模态概念瓶颈模型框架，通过可微分泄漏损失减少概念表示中的非预期信息泄漏，同时用Kolmogorov-Arnold Network (KAN) 预测头提升概念检测精度，在任务准确率、概念检测和泄漏减少间取得最优Pareto前沿。

**[VIRO: Robust and Efficient Neuro-Symbolic Reasoning with Verification for Referring Expression Comprehension](interpretability/viro_robust_and_efficient_neuro-symbolic_reasoning_with_verification_for_referri.md)**

:   VIRO在神经符号REC管道中嵌入轻量算子级验证机制（CLIP不确定性验证+空间逻辑验证），使每个推理步骤能自我验证并在无目标时提前终止，在零样本设置下以61.1%平衡准确率大幅超越组合推理baselines，同时保持0.3%以下的程序失败率和高效推理速度。

**[Why Does It Look There? Structured Explanations for Image Classification](interpretability/why_does_it_look_there_structured_explanations_for_image_classification.md)**

:   提出 I2X 框架，通过在训练检查点追踪从 GradCAM 提取的原型强度与模型置信度的协同演化，将非结构化的可解释性（显著性图）转化为结构化的可解释性，揭示模型"为什么关注那里"的推理结构，并利用这种理解指导微调提升性能。

---

## 📊 LLM 评测 { #llm_evaluation }

**[ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation](llm_evaluation/ace-merging_data-free_model_merging_with_adaptive_covariance_estimation.md)**

:   本文从理论上证明了微调参数差蕴含输入协方差信息，据此提出 ACE-Merging，通过自适应协方差估计、集体结构先验和谱精炼三步实现无数据闭式模型合并，在 GPT-2 上比之前方法平均提升 4%，在 RoBERTa-Base 上提升 5%。

**[AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks](llm_evaluation/adabet_gradient-free_layer_selection_for_efficient_training_of_deep_neural_netwo.md)**

:   提出 AdaBet，一种基于代数拓扑（第一 Betti 数 $b_1$）的无梯度层选择方法，仅通过前向传播计算每层激活空间的拓扑复杂度来决定哪些层需要微调，无需标签、梯度或反向传播，在 ResNet50/VGG16/MobileNetV2/ViT-B16 上以仅 10% 层微调达到优于全量训练的准确率，同时峰值内存降低约 40%。

**[Anchoring and Rescaling Attention for Semantically Coherent Inbetweening](llm_evaluation/anchoring_and_rescaling_attention_for_semantically_coherent_inbetweening.md)**

:   提出 KAB（Keyframe-Anchored Attention Bias）和 ReTRo（Rescaled Temporal RoPE）两个无需训练的推理时方法，基于 Wan2.1 视频扩散模型解决稀疏关键帧下大运动生成式帧插值（GI）中的语义不忠、帧不一致和节奏不稳问题，并构建首个文本条件 GI 评估基准 TGI-Bench。

**[Cross-Scale Pansharpening via ScaleFormer and the PanScale Benchmark](llm_evaluation/cross-scale_pansharpening_via_scaleformer_and_the_panscale_benchmark.md)**

:   提出首个跨尺度全色锐化数据集PanScale和评测基准PanScale-Bench，以及ScaleFormer框架——将分辨率变化重新解释为序列长度变化，通过Scale-Aware Patchify分桶采样+解耦空间-序列建模+RoPE实现跨尺度泛化。

**[CryoHype: Reconstructing a Thousand Cryo-EM Structures with Transformer-Based Hypernetworks](llm_evaluation/cryohype_reconstructing_a_thousand_cryo-em_structures_with_transformer-based_hyp.md)**

:   提出 CryoHype，一种基于 Transformer 超网络的冷冻电镜重建方法，通过动态调整隐式神经表示（INR）的权重来减少参数共享，首次实现了从无标签冷冻电镜图像中同时重建 1000 种不同蛋白质结构。

**[Enhancing Out-of-Distribution Detection with Extended Logit Normalization](llm_evaluation/enhancing_out-of-distribution_detection_with_extended_logit_normalization.md)**

:   本文发现 LogitNorm 在训练中会导致两种特征坍塌（维度坍塌和原点坍塌），提出了一种无超参数的 Extended Logit Normalization（ELogitNorm），用特征到决策边界的距离替代到原点的距离作为缩放因子，在不损失分类精度的前提下显著提升各种 post-hoc OOD 检测方法的性能和置信度校准。

**[Flow3r: Factored Flow Prediction for Scalable Visual Geometry Learning](llm_evaluation/flow3r_factored_flow_prediction_for_scalable_visual_geometry_learning.md)**

:   提出"分解式光流预测"（Factored Flow）模块，用源视图的几何 latent + 目标视图的位姿 latent 预测光流，使无标注视频可作为三维几何学习的监督信号，在静态/动态场景的 8 个基准上达到 SOTA。

**[Free-Grained Hierarchical Visual Recognition](llm_evaluation/free-grained_hierarchical_visual_recognition.md)**

:   提出"自由粒度"层级视觉识别（free-grained hierarchical recognition），允许训练标签出现在分类法的任意层级，并提出文本引导伪属性和分类法引导半监督学习两种方法来弥补缺失监督，推理时模型自适应选择预测深度。

**[HeSS: Head Sensitivity Score for Sparsity Redistribution in VGGT](llm_evaluation/hess_head_sensitivity_score_for_sparsity_redistribution_in_vggt.md)**

:   HeSS 提出 Head Sensitivity Score 来量化 VGGT 全局注意力层中每个注意力头对稀疏化的敏感程度，并基于此将注意力预算从不敏感的头重新分配到敏感头，在高稀疏度下显著优于均匀稀疏化方法 SparseVGGT，几乎不增加运行时开销。

**[Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces](llm_evaluation/hier-cos_making_deep_features_hierarchy-aware_via_composition_of_orthogonal_subs.md)**

:   提出 Hier-COS 框架，通过为层次树中每个节点分配正交基向量，构造理论上保证层次一致性的层次感知向量空间(HAVS)，首次统一了"层次感知细粒度分类"和"层次多级分类"，同时提出新评估指标HOPS，在4个数据集上全面超越SOTA。

**[HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](llm_evaluation/hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)**

:   本文识别了异质域持续学习中的"域引力"（Domain Gravity）偏差——数据丰富或低熵域在共享嵌入空间中产生不成比例的影响，并提出 HyCal，一种无训练方法，通过融合余弦相似度和马氏距离进行原型校准，在跨学科不平衡少样本增量学习中实现稳健分类。

**[Learning Like Humans: Analogical Concept Learning for Generalized Category Discovery](llm_evaluation/learning_like_humans_analogical_concept_learning_for_generalized_category_discov.md)**

:   提出 AL-GCD 框架，通过模拟人类类比推理机制设计"类比文本概念生成器"（ATCG）——从已知类别的视觉-文本知识库中类比生成未知样本的文本概念，将类别发现转化为视觉-文本联合推理任务，在六个基准上平均提升 5.0%，细粒度数据集提升 7.1%。

**[Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models](llm_evaluation/out_of_sight_out_of_mind_evaluating_state_evolutio.md)**

:   本文提出StEvo-Bench，一个包含225个任务的benchmark，通过在视频生成过程中插入遮挡或相机转向来测试视频世界模型能否在不可观测期间继续正确演化场景状态，发现当前最先进模型（包括Veo 3、Sora 2 Pro等）的成功率不到10%，揭示了视频模型将状态演化与观察高度耦合的根本问题。

**[Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline](llm_evaluation/pioneering_perceptual_video_fluency_assessment_a_novel_task_with_benchmark_datas.md)**

:   本文首次将视频流畅度评估（VFA）从传统视频质量评估（VQA）中独立出来，构建了首个流畅度评估数据集 FluVid（4,606 视频），并提出 FluNet 基线模型，通过时序排列自注意力（T-PSA）实现高效帧间交互，SRCC/PLCC 分别达到 0.816/0.821。

**[PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion](llm_evaluation/prism_video_dataset_condensation_with_progressive_refinement_and_insertion_for_s.md)**

:   本文提出 PRISM，一种整体式视频数据集压缩方法：从仅两个时间锚点（首尾帧）出发,通过检测梯度方向冲突来自适应插入关键帧，在保持内容与运动的耦合完整性的同时实现 SOTA 的存储效率——在 miniUCF 1VPC 上用 20MB 达到 17.9% 准确率，比先前方法的 94MB 少 5 倍。

**[R2G: A Multi-View Circuit Graph Benchmark Suite from RTL to GDSII](llm_evaluation/r2g_multi_view_circuit_graph_benchmark_suite_from_rtl_to_gdsii.md)**

:   提出 R2G，首个标准化的多视图电路图基准套件，在 30 个 IP 核上提供 5 种阶段感知的图表示（具有信息对等性），系统研究发现图表示选择比 GNN 模型选择对性能影响更大。

**[ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation](llm_evaluation/reflexsplit_single_image_reflection_separation_via_layer_fusion-separation.md)**

:   针对单图反射分离中的透射-反射混淆问题（尤其是在深层解码器中），提出ReflexSplit双流框架，通过跨尺度门控融合(CrGF)稳定多尺度特征流、层级融合-分离块(LFSB)的差分双维注意力实现跨流减法解纠缠、课程训练渐进增强分离强度，在合成和真实世界数据集上达到SOTA性能。

**[Reframing Long-Tailed Learning via Loss Landscape Geometry](llm_evaluation/reframing_long-tailed_learning_via_loss_landscape_geometry.md)**

:   从损失景观几何的角度重新审视长尾学习中的head-tail seesaw困境，发现尾类退化的根源是优化收敛到尖锐且远离尾类最优点的区域，提出基于持续学习思想的GKP（分组知识保存）和GSA（分组锐度感知）双模块框架，无需额外数据即在CIFAR-LT/ImageNet-LT/iNat2018四个基准上取得SOTA。

**[SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval](llm_evaluation/sattc_structure-aware_label-free_test-time_calibration_for_cross-subject_eeg-to-.md)**

:   提出SATTC，一个无标签的测试时校准头，通过几何专家（被试自适应白化+自适应CSLS）和结构专家（互最近邻+双向top-k排名+类别流行度）的乘积专家融合，在冻结的EEG和图像编码器上直接操作相似度矩阵，显著改善跨被试EEG-to-image检索的Top-1精度并降低hubness效应。

**[Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](llm_evaluation/semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)**

:   提出 SemiCP 框架，通过最近邻匹配（NNM）分数将无标签数据引入 conformal prediction 的校准流程，在标注数据极少时将平均覆盖率偏差降低最多 77%，同时缩小预测集。

**[SparseCam4D: Spatio-Temporally Consistent 4D Reconstruction from Sparse Cameras](llm_evaluation/sparsecam4d_spatio-temporally_consistent_4d_reconstruction_from_sparse_cameras.md)**

:   提出 SparseCam4D，首个在标准多相机动态场景基准上实现稀疏相机（2-3个）4D重建的方法，核心创新是时空扭曲场（STDF），通过将生成式观测中的时空不一致性显式建模并与真实4D高斯表示解耦，实现高保真、时空一致的动态场景渲染。

**[TacSIm: A Dataset and Benchmark for Football Tactical Style Imitation](llm_evaluation/tacsim_a_dataset_and_benchmark_for_football_tactical_style_imitation.md)**

:   本文提出 TacSIm，首个从真实英超比赛转播画面中重建全队轨迹并在虚拟足球环境中进行战术风格模仿的大规模数据集与基准，通过空间占据相似度和运动向量相似度两个指标量化战术模仿保真度。

**[Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning](llm_evaluation/temporal_imbalance_of_positive_and_negative_supervision_in_class-incremental_lea.md)**

:   提出时序不平衡（Temporal Imbalance）这一被忽视的类增量学习偏差来源，并设计 Temporal-Adjusted Loss（TAL）通过时间衰减记忆核动态降低旧类的负监督权重，以即插即用的方式显著缓解灾难性遗忘。

**[Unified Primitive Proxies for Structured Shape Completion](llm_evaluation/unified_primitive_proxies_for_structured_shape_completion.md)**

:   提出 UniCo，通过基元代理（primitive proxies）在共享形状特征上学习统一的基元表示，在单次前向传递中联合预测完整点云和装配就绪的二次曲面基元（含几何、语义和成员关系），在合成/真实点云 benchmark 上 Chamfer 距离降低最高 50%，法线一致性提升最高 7%。

**[VGA-Bench: A Unified Benchmark for Video Aesthetics and Generation Quality Evaluation](llm_evaluation/vga_bench_unified_benchmark_for_video_aesthetics_and_generation_quality.md)**

:   VGA-Bench提出了一个统一的AIGC视频评估基准，包含三层分类体系（美学质量、美学标签、生成质量）、1016个提示词、60000个视频和三个专用评估模型，实现了与人类判断对齐的自动化评估。

**[Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](llm_evaluation/weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)**

:   提出 LAS-VAD 框架，通过异常连通分量机制（ACC）将视频帧划分为语义一致的组来生成伪标签弥补帧级标注缺失，并通过意图感知机制（IAM）利用位置-速度-加速度特征区分外观相似但意图不同的正常/异常行为，在 XD-Violence 上达 89.96% AP (I3D)。

---

## 🛡️ AI 安全 { #ai_safety }

**[A Unified Perspective on Adversarial Membership Manipulation in Vision Models](ai_safety/a_unified_perspective_on_adversarial_membership_manipulation_in_vision_models.md)**

:   首次揭示视觉模型成员推断攻击(MIA)面临的对抗性成员操纵漏洞——不可感知扰动可将非成员伪造为成员欺骗审计，发现伪造成员的梯度范数塌缩特征签名，并提出基于梯度几何的检测策略和对抗鲁棒推断框架。

**[All Vehicles Can Lie: Efficient Adversarial Defense in Fully Untrusted-Vehicle Collaborative Perception via Pseudo-Random Bayesian Inference](ai_safety/all_vehicles_can_lie_efficient_adversarial_defense_in_fully_untrusted-vehicle_co.md)**

:   提出 Pseudo-Random Bayesian Inference (PRBI) 框架，在**所有车辆均不可信**的协同感知场景中，利用帧间时序一致性作为自参考信号，通过伪随机分组 + 贝叶斯推断，仅需平均 2.5 次验证/帧即可高效识别并排除恶意车辆，检测精度恢复至攻击前的 79.4%–86.9%。

**[ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering](ai_safety/clustermark_towards_robust_watermarking_for_autoregressive_image_generators_with.md)**

:   提出基于视觉 token 聚类的水印方案 ClusterMark，将 KGW 风格的 LLM 水印适配到自回归图像生成器，通过将相似 token 分到同一绿/红集合来显著提升水印在图像扰动下的鲁棒性，同时保持图像质量。

**[Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression](ai_safety/computation_and_communication_efficient_federated_unlearning_via_on-server_gradi.md)**

:   提出 FOUL 框架，通过"学习阶段解耦因果/非因果特征 + 遗忘阶段服务器端梯度冲突匹配"两阶段策略，在不访问客户端数据的前提下实现高效且低通信开销的联邦遗忘。

**[AdvMark: Decoupling Defense Strategies for Robust Image Watermarking](ai_safety/decoupling_defense_strategies_for_robust_image_watermarking.md)**

:   提出 AdvMark 两阶段解耦防御框架：Stage 1 Encoder Adversarial Training（EAT）将水印图像移入 non-attackable 区域抵御对抗攻击，Stage 2 直接图像优化抵御失真+再生攻击并保留对抗鲁棒性，在 9 种水印方法 ×10 种攻击上分别提升失真/再生/对抗准确率 29%/33%/46%，且图像质量最优。

**[Domain-Skewed Federated Learning with Feature Decoupling and Calibration](ai_safety/domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)**

:   提出 F²DC 框架，通过域特征解耦器（DFD）和域特征校正器（DFC）将联邦学习中客户端的局部特征分离为域鲁棒特征和域相关特征，并对域相关特征进行校准以挽救被丢弃的类别信息，配合域感知聚合策略，在三个多域数据集上一致超越 SOTA。

**[FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning](ai_safety/fecalfed_privacy-preserving_poultry_disease_detection_via_federated_learning.md)**

:   提出 FecalFed 隐私保护联邦学习框架，首先通过双哈希去重清理公开禽类粪便数据集中 46.89% 的重复污染并发布 8,770 张清洁基准数据集 poultry-fecal-fl，随后在 Dirichlet α=0.5 的高度非 IID 条件下验证：FedAdam + Swin-Small 可将单农场训练崩溃的 64.86% 准确率恢复至 90.31%，仅比中心化上界 95.10% 低 4.79%；边缘优化的 Swin-Tiny（28M 参数）仍达 89.74%，为农场端部署提供高效可行方案。

**[FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](ai_safety/fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)**

:   提出 FedAFD 框架，通过双层对抗对齐、粒度感知特征融合和相似度引导的集成蒸馏三阶段设计，在多模态联邦学习中同时提升异构客户端和服务器的模型性能。

**[FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](ai_safety/feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)**

:   提出域感知原型联邦学习框架 FedDAP，通过构建域特定全局原型和双重原型对齐策略（域内对齐 + 跨域对比），解决联邦学习中客户端数据域偏移导致的全局模型性能退化问题。

**[Federated Active Learning Under Extreme Non-IID and Global Class Imbalance](ai_safety/federated_active_learning_extreme_noniid.md)**

:   系统分析全局类不平衡与客户端异构性对联邦主动学习中 query model 选择的影响，归纳出3个核心 Observation，据此提出 FairFAL——自适应选择 query model + 原型引导伪标签 + 两阶段不确定性-多样性平衡采样的类公平 FAL 框架，在5个基准数据集上一致超越所有基线。

**[FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](ai_safety/fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)**

:   提出 FedRE 框架，通过"纠缠表示"（entangled representation）——将每个客户端的所有局部表示用归一化随机权重聚合为单一跨类别表示，实现模型异构联邦学习中性能、隐私保护和通信开销的三方平衡。

**[Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting](ai_safety/generative_adversarial_perturbations_with_cross-paradigm_transferability_on_loca.md)**

:   提出首个跨范式（密度图 + 点回归）对抗攻击框架 CrowdGen，利用轻量级 UNet 生成器和多任务损失（logit 抑制 + 密度抑制 + GradCAM 引导 + 频域约束），在保持视觉隐蔽性（~19dB PSNR）的同时实现对七个 SOTA 人群计数模型的高迁移率（TR 最高 1.69），攻击 MAE 平均提升 7 倍。

**[LogitDynamics: Reliable ViT Error Detection from Layerwise Logit Trajectories](ai_safety/logitdynamics_vit_error_detection.md)**

:   LogitDynamics 通过在 ViT 各层附加轻量分类头，提取层间 logit 轨迹和 top-K 竞争动态特征，训练线性探针来预测模型错误，在跨数据集泛化上优于现有方法。

**[Monte Carlo Stochastic Depth for Uncertainty Estimation in Deep Learning](ai_safety/mcsd_uncertainty_estimation.md)**

:   将随机深度（Stochastic Depth）正式连接到贝叶斯变分推理框架，提出 Monte Carlo Stochastic Depth (MCSD) 作为不确定性估计方法，并在 YOLO、RT-DETR 等现代检测器上进行首次系统基准测试，证明其在校准和不确定性排名上与 MC Dropout 竞争力强。

**[One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control](ai_safety/one-to-more_high-fidelity_training-free_anomaly_generation_with_attention_control.md)**

:   O2MAG 提出一种无需训练的少样本异常生成方法，通过三分支扩散过程中的自注意力嫁接(TriAG)从单张参考异常图像合成更多逼真异常，配合异常引导优化(AGO)对齐文本语义和异常引导增强(DAE)确保掩码区域完整填充，在 MVTec-AD 下游异常检测任务中显著优于现有方法。

**[ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](ai_safety/proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)**

:   提出 ProxyFL 框架，利用分类器权重作为统一代理 (proxy) 同时缓解联邦半监督学习中的外部异质性（跨客户端分布差异）和内部异质性（标注/未标注数据分布不匹配），在多个数据集上显著超越现有 FSSL 方法。

**[RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces](ai_safety/recovermark_robust_watermarking_for_localization_and_recovery_of_manipulated_fac.md)**

:   提出 RecoverMark，一个将人脸内容本身作为水印嵌入背景的鲁棒水印框架，同时实现篡改区域定位、原始内容恢复和版权验证，在水印移除攻击下仍保持有效。

**[SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport](ai_safety/subflot_submodel_extraction_for_efficient_and_personalized_federated_learning_vi.md)**

:   提出 SubFLOT 框架，在服务器端利用最优传输（Optimal Transport）将全局模型的参数分布与客户端历史模型对齐，实现无需访问原始数据的个性化剪枝，并通过自适应正则化抑制剪枝导致的参数偏移，在多个数据集上大幅超越现有联邦剪枝方法。

**[TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking](ai_safety/tiacam_text-anchored_invariant_feature_learning_with_auto-augmentation_for_camer.md)**

:   提出 TIACam 框架，通过可学习自动增强器模拟相机失真、文本锚定跨模态对抗训练学习不变特征、零水印头在特征空间绑定消息，实现无需修改图像像素的相机鲁棒零水印方案，在屏幕翻拍/打印翻拍/截图三种真实场景下均达到 SOTA 提取精度。

**[Towards Highly Transferable Vision-Language Attack via Semantic-Augmented Dynamic Contrastive Interaction](ai_safety/towards_highly_transferable_vision-language_attack_via_semantic-augmented_dynami.md)**

:   提出 SADCA（语义增强动态对比攻击），通过动态对比交互机制和语义增强模块，迭代地破坏对抗图像与文本之间的跨模态语义一致性，显著提升对视觉语言预训练模型（VLP）的对抗可迁移性，在跨模型和跨任务攻击中均超越现有 SOTA 方法。

**[Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection](ai_safety/tutor-student_reinforcement_learning_a_dynamic_curriculum_for_robust_deepfake_de.md)**

:   提出 Tutor-Student 强化学习（TSRL）框架，将深度伪造检测器的训练过程建模为马尔可夫决策过程，由"导师"（PPO agent）根据每个样本的视觉特征和历史学习动态（EMA 损失、遗忘次数）动态分配损失权重，通过"状态变化"奖励信号引导"学生"（检测器）优先学习高价值样本，在跨数据集和跨方法评估中显著提升泛化能力。

**[When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models](ai_safety/when_robots_obey_the_patch_universal_transferable_patch_attacks_on_vision-langua.md)**

:   提出 UPA-RFAS 框架，学习一个单一物理对抗补丁，通过特征空间偏移、注意力劫持和语义错位三管齐下，实现对 VLA 机器人策略的通用、可迁移黑盒攻击。

---

## 🎮 强化学习 { #reinforcement_learning }

**[AceTone: Bridging Words and Colors for Conditional Image Grading](reinforcement_learning/acetone_bridging_words_and_colors_for_conditional_image_grading.md)**

:   提出AceTone，首个支持文本和参考图像多模态条件色彩调色的统一框架，通过VQ-VAE将3D-LUT压缩为64个离散token，训练VLM预测LUT token序列，再用GRPO强化学习对齐色彩相似度和美学偏好，在风格迁移和指令调色上LPIPS改善50%。

**[Anticipatory Planning for Multimodal AI Agents](reinforcement_learning/anticipatory_planning_for_multimodal_ai_agents.md)**

:   提出 TraceR1，一个两阶段 RL 框架：第一阶段通过轨迹级奖励优化让智能体学会"向前看几步"的前瞻性规划，第二阶段通过工具执行反馈做 grounded fine-tuning 来提升单步精度，在 7 个 GUI 和工具使用 benchmark 上取得了开源 SOTA。

**[AnyDoc: Enhancing Document Generation via Large-Scale HTML/CSS Data Synthesis and Height-Aware Reinforcement Optimization](reinforcement_learning/anydoc_enhancing_document_generation_via_large-scale_htmlcss_data_synthesis_and_.md)**

:   AnyDoc 提出了一个基于统一 HTML/CSS 表示的通用文档生成框架，通过自动化数据合成管线构建 265K 文档数据集 DocHTML，结合 SFT 和高度感知强化学习（HARL）微调多模态大模型，在意图到文档、文档反渲染和元素到文档三个任务上超越 GPT-4o 等基线。

**[BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](reinforcement_learning/bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)**

:   提出 BRIDGE 系统，通过 FORGE（RL 训练的查询对齐模型）将噪声多模态查询蒸馏为检索优化的纯文本查询，配合 LENS 推理增强检索器，在 MM-BRIGHT 上达到 29.7 nDCG@10，作为插件进一步将 Nomic-Vision 提升到 33.3，超越最佳纯文本检索器。

**[CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning](reinforcement_learning/cccaption_dual-reward_reinforcement_learning_for_complete_and_correct_image_capt.md)**

:   提出 CCCaption 双奖励强化学习框架，通过 completeness reward（基于多 MLLM 生成的视觉 query 集）和 correctness reward（基于 caption 分解后的子 query 幻觉检测）联合优化图像描述的完整性和正确性，2B 模型超越 32B 基线。

**[Cross-modal Identity Mapping: Minimizing Information Loss in Modality Conversion via Reinforcement Learning](reinforcement_learning/cross-modal_identity_mapping_minimizing_information_loss_in_modality_conversion_.md)**

:   提出 Cross-modal Identity Mapping (CIM)，通过分析用 caption 检索到的图像的表示一致性（GRC）和与源图像的相关性（QIR）来量化图像描述中的信息损失，将其作为 RL 奖励信号训练 LVLM 生成细粒度且精确的描述，无需额外标注。

**[GeoWorld: Geometric World Models](reinforcement_learning/geoworld_geometric_world_models.md)**

:   GeoWorld 将预测式世界模型的潜在表征从欧氏空间映射到双曲流形上，通过 Hyperbolic JEPA 保持几何结构和层级关系，并提出 Geometric Reinforcement Learning 来优化多步规划，在 CrossTask 和 COIN 上实现了约 3% SR（3步）和 2% SR（4步）的提升。

**[GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion](reinforcement_learning/graspldp_towards_generalizable_grasping_policy_via_latent_diffusion.md)**

:   提出 GraspLDP，将预训练抓取检测器的 grasp pose 先验和 graspness map 视觉线索注入潜在扩散策略框架，通过 VAE 编码的动作潜空间引导和自监督重建目标，显著提升抓取精度和泛化能力。

**[Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](reinforcement_learning/lifelong_imitation_learning_multimodal_latent_rep.md)**

:   提出终身模仿学习框架，通过多模态潜在回放（MLR）在冻结编码器的特征空间中存储和回放紧凑表示，并引入增量特征调整（IFA）机制用角距离约束维持任务间可分性，在LIBERO基准上AUC提升10-17点、遗忘降低最多65%。

**[MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning](reinforcement_learning/msrl_scaling_generative_multimodal_reward_modeling.md)**

:   提出多阶段强化学习（MSRL）方法，通过先在大规模文本偏好数据上学习奖励推理能力，再逐步迁移到多模态任务，解决多模态奖励模型训练中标注数据稀缺的瓶颈问题，在 VL-RewardBench 上将准确率从 66.6% 提升至 75.9%。

**[RADAR: Closed-Loop Robotic Data Generation via Semantic Planning and Autonomous Causal Environment Reset](reinforcement_learning/radar_closedloop_robotic_data_generation_via_seman.md)**

:   提出RADAR——一个完全自主的闭环机器人操作数据生成引擎，通过VLM语义规划+GNN策略执行+VQA成功评估+FSM驱动的LIFO因果逆序环境重置四个模块，仅需2-5个人工演示即可持续生成高保真操作数据，在仿真中复杂长horizon任务达到90%成功率。

**[ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering](reinforcement_learning/reag_reasoning-augmented_generation_for_knowledge-based_visual_question_answerin.md)**

:   提出 ReAG，一个推理增强的多模态 RAG 方法，结合粗细粒度检索与 Critic 过滤模型减少噪声，并通过 GRPO 强化学习训练生成器进行显式推理，在知识密集型 VQA 上达到新 SOTA。

**[Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision](reinforcement_learning/reasoning-driven_anomaly_detection_and_localization_with_image-level_supervision.md)**

:   提出 ReAL 和 CGRO 两个模块，通过提取 MLLM 自回归推理过程中的异常相关 token 并聚合其视觉注意力来生成像素级异常图，再通过一致性引导的强化学习对齐推理与视觉证据，实现仅凭图像级监督的端到端异常检测、定位与可解释推理。

**[Reinforce to Learn, Elect to Reason: A Dual Paradigm for Video Reasoning](reinforcement_learning/reinforce_to_learn_elect_to_reason_a_dual_paradigm_for_video_reasoning.md)**

:   提出 RLER 双范式框架，训练阶段用 GRPO 配合三种新颖奖励（Frame-sensitive、Think-transparency、Anti-repetition）教模型生成结构化证据，推理阶段用无训练编排器在多候选之间基于证据一致性进行加权选举和自检，在 8 个视频基准上全面超越开源和 RL-based LMM，平均提升 6.3%，仅需约 3.1 个候选。

**[Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation](reinforcement_learning/rethinking_camera_choice_an_empirical_study_on_fisheye_camera_properties_in_robo.md)**

:   首次系统性地对腕部鱼眼相机在机器人操作模仿学习中的特性进行实证研究，围绕空间定位、场景泛化和硬件泛化三个核心问题揭示了宽视场角的优势与局限，并提出 Random Scale Augmentation (RSA) 策略解决跨相机迁移中的尺度过拟合问题。

**[RoboAgent: Chaining Basic Capabilities for Embodied Task Planning](reinforcement_learning/roboagent_chaining_basic_capabilities_for_embodied_task_planning.md)**

:   提出 RoboAgent，一种能力驱动的具身任务规划框架，用单个 VLM 同时实现调度器和 5 种基本能力（探索引导、物体定位、场景描述、动作解码、经验总结），通过三阶段训练（SFT + DAgger + 专家引导 RL）在 EB-ALFRED 和 ALFWorld 上达到 SOTA。

**[See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs](reinforcement_learning/see_it_say_it_sorted_an_iterative_training-free_framework_for_visually-grounded_.md)**

:   提出Evidence-Constrained Reweighting Decoding（ECRD）框架：在LVLM解码时维护动态文本证据池，通过分布协商重加权候选token，不确定时自动调用轻量视觉决策器提取微证据，无需训练即可在多个LVLM上显著减少视觉幻觉、提升推理准确率。

**[Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement](reinforcement_learning/seeing_is_improving_visual_feedback_for_iterative_text_layout_refinement.md)**

:   VFLM 提出一个利用视觉反馈进行迭代优化的布局生成框架，通过结合 OCR 准确率的视觉奖励模型和强化学习训练，使多模态大语言模型能够"看到"渲染结果并反复修正，在文本排版质量上显著超越仅生成代码的方法。

**[Specificity-aware Reinforcement Learning for Fine-grained Open-world Classification](reinforcement_learning/specificity-aware_reinforcement_learning_for_fine-grained_open-world_classificat.md)**

:   提出 SpeciaRL——一种特异性感知的强化学习框架，通过基于在线 rollout 最佳预测的动态奖励信号，引导推理型大型多模态模型在开放世界细粒度图像分类中同时提升预测的特异性和正确性。

---

## � LLM 安全 { #llm_safety }

**[Association and Consolidation: Evolutionary Memory-Enhanced Incremental Multi-View Clustering](llm_safety/association_and_consolidation_evolutionary_memory-enhanced_incremental_multi-vie.md)**

:   提出 EMIMC 框架，受大脑海马-前额叶协作记忆机制启发，通过 Rapid Associative Module (正交映射保证可塑性)、Cognitive Forgetting Module (幂律衰减模拟遗忘曲线) 和 Knowledge Consolidation Module (时序张量低秩分解提炼长期记忆) 三模块协同，解决增量多视图聚类中的稳定性-可塑性困境。

**[Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations](llm_safety/beyond_global_scores_fine_grained_token_grounding_as_robust_detector_of_lvlm_hallucinations.md)**

:   提出基于 patch 级别的 LVLM 幻觉检测框架，发现幻觉 token 表现出弥散注意力模式和低语义对齐两个特征标志，据此设计注意力弥散分数（ADS）和跨模态接地一致性（CGC）两个轻量指标，检测准确率达 90%。

**[The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models](llm_safety/blind_spot_of_adaptation_quantifying_and_mitigating_forgetting_in_fine_tuned_driving_models.md)**

:   系统研究 VLM 微调到自动驾驶场景时的灾难性遗忘问题，构建 180K 场景大规模基准 FidelityDrivingBench，并提出 Drive Expert Adapter (DEA) 通过提示空间路由在不腐蚀基础参数的前提下增强驾驶任务性能。

**[DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions](llm_safety/damp_class_unlearning_via_depth_aware_removal_of_forget_specific_directions.md)**

:   提出 DAMP（深度感知投影调制），一种一次性闭式权重手术方法用于类遗忘，通过在每个网络阶段的编辑空间中移除遗忘类特有方向来实现选择性遗忘，深度感知缩放规则确保浅层保守编辑、深层强力编辑。

**[Designing to Forget: Deep Semi-parametric Models for Unlearning](llm_safety/designing_to_forget_deep_semi-parametric_models_for_unlearning.md)**

:   提出"Designing to Forget"理念，设计了一族深度半参数模型 (SPM)，在推理时通过简单删除训练样本即可实现遗忘（无需修改模型参数），在 ImageNet 分类上将与重训基线的预测差距减少 11%，遗忘速度提升 10 倍以上。

**[Elastic Weight Consolidation Done Right for Continual Learning](llm_safety/elastic_weight_consolidation_done_right_for_continual_learning.md)**

:   本文从梯度视角系统分析了 EWC 及其变体在权重重要性估计上的根本缺陷（EWC 的梯度消失和 MAS 的冗余保护），并提出了一个极其简单的 Logits Reversal 操作来修正 Fisher 信息矩阵的计算，在无样例类增量学习和多模态持续指令微调任务上大幅超越原始 EWC 及其所有变体。

**[HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs](llm_safety/hulluedit_subspace_editing_hallucination.md)**

:   提出HulluEdit，一个单次推理、无参考模型的幻觉缓解框架，通过将隐藏状态正交分解为视觉证据子空间、冲突先验子空间和残差不确定性子空间，选择性抑制幻觉模式而不干扰视觉接地，在POPE和CHAIR上达到SOTA。

**[Learning from Oblivion: Predicting Knowledge-Overflowed Weights via Retrodiction of Forgetting](llm_safety/learning_from_oblivion_predicting_knowledge_overflowed_weights_via_retrodiction_.md)**

:   提出KNOW prediction：通过在逐步缩小的数据子集上sequential fine-tuning诱导结构化遗忘过程，收集权重转变轨迹，然后用meta-learned hyper-model（KNOWN）反转forgetting方向，预测"仿佛在更大数据集上训练"的虚拟知识增强权重。跨多数据集(CIFAR/ImageNet/PACS等)和多架构(ResNet/PVTv2/DeepLabV3+)持续超越naive fine-tuning及多种weight prediction基线，在图像分类、语义分割、图像描述、域泛化等下游任务上均有显著提升。

**[Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](llm_safety/multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)**

:   提出 MPCAttack 框架，联合跨模态对齐、多模态理解和视觉自监督三种学习范式的特征表示，通过多范式协同优化策略生成高迁移性对抗样本，在开源和闭源 MLLM 上均取得 SOTA 攻击效果。

**[⊘ Source Models Leak What They Shouldn't ↛: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization](llm_safety/oslash_source_models_leak_what_they_shouldnt_nrightarrow_unlearning_zero-shot_tr.md)**

:   发现无源域自适应（SFDA）方法会不经意地将源域独有类别的知识泄漏到目标域（零样本迁移现象），提出 SCADA-UL 框架通过对抗生成遗忘样本和重缩放标签策略，在域自适应过程中同时完成类别遗忘，达到接近从头训练的遗忘效果。

**[Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP](llm_safety/perturb_and_recover_fine-tuning_for_effective_backdoor_removal_from_clip.md)**

:   本文提出 PAR（Perturb and Recover），一种简单而有效的 CLIP 模型后门清洗方法：通过显式地将模型embedding推离中毒状态（Perturb），同时用标准 CLIP 损失恢复干净性能（Recover），在不依赖强数据增强的情况下实现对任意触发器的鲁棒后门移除，甚至仅用合成数据即可有效清洗。

**[PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing](llm_safety/pinpoint_evaluation_of_composed_image_retrieval_with_explicit_negatives_multi-im.md)**

:   提出 PinPoint 基准，包含 7,635 个查询和 329K 人工验证的相关性判断，通过显式负样本、多图像查询、释义变体和人口统计元数据四个维度，揭示了现有 CIR 方法在假阳性抑制、语言鲁棒性和多图像推理上的严重缺陷，并提出基于 MLLM 的无训练重排方法作为改进基线。

**[Select, Hypothesize and Verify: Towards Verified Neuron Concept Interpretation](llm_safety/select_hypothesize_and_verify_towards_verified_neuron_concept_interpretation.md)**

:   提出 SIEVE（Select–Hypothesize–Verify）框架，通过筛选高激活样本、生成概念假设、再用文生图验证的闭环流程来解释神经元功能，生成的概念激活对应神经元的概率约为现有 SOTA 的 1.5 倍。

**[SineProject: Machine Unlearning for Stable Vision–Language Alignment](llm_safety/sineproject_machine_unlearning_for_stable_vision_language_alignment.md)**

:   针对多模态大模型（MLLM）在机器遗忘过程中投影层 Jacobian 严重病态导致视觉-语言对齐漂移的问题，提出 SineProject——通过对投影层权重施加正弦调制（sin(ΔW)）来约束参数范围至 [-1,1]，从而将 Jacobian 条件数降低 3-4 个数量级，在完全遗忘目标知识的同时将良性查询误拒率（SARR）降低 15%。

**[Unsafe2Safe: Controllable Image Anonymization for Downstream Utility](llm_safety/unsafe2safe_controllable_image_anonymization_for_downstream_utility.md)**

:   本文提出 Unsafe2Safe 全自动隐私保护流水线，通过 VLM 隐私检查→双字幕生成（私有/公开）→LLM 编辑指令→文本引导扩散编辑的四阶段方案，实现可控图像匿名化，在 VLMScore 隐私指标大幅提升的同时，在 Caltech-101 分类和 OK-VQA 上匿名后准确率甚至超过原始图像。

**[V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs](llm_safety/v-attack_targeting_disentangled_value_features_for_controllable_adversarial_atta.md)**

:   发现 ViT 中 Value 特征相比 Patch 特征具有更解耦的局部语义表示，提出 V-Attack 通过自增强 Value 特征 + 文本引导语义操控实现精确可控的 LVLM 局部语义攻击，ASR 平均提升 36%。

---

## 🎵 音频/语音 { #audio_speech }

**[BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models](audio_speech/babyvlm-v2_toward_developmentally_grounded_pretraining_and_benchmarking_of_visio.md)**

:   提出BabyVLM-V2框架，从婴儿第一视角的SAYCam纵向语料构建三种格式预训练数据（768K图像对+181K视频对+63K交错序列），设计基于NIH Baby Toolbox®的DevCV Toolbox（10个发育认知任务），从零训练的紧凑模型在部分数学任务上超越GPT-4o，首次系统探索人工发育智能(ADI)。

**[Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning](audio_speech/cleaning_the_pool_progressive_filtering_of_unlabeled_pools_in_deep_active_learni.md)**

:   提出 Refine 集成主动学习方法，通过两阶段策略——渐进过滤（多策略迭代精炼无标签池）+ 覆盖选择（从精炼池中选择多样性高价值样本）——在不预知最佳策略的情况下一致超越单一 AL 策略和现有集成方法。

**[Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models](audio_speech/echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo.md)**

:   提出 MMHNet，一种基于层级结构和非因果 Mamba-2 的多模态层级网络，实现了在短片段（8秒）上训练、在长视频（5分钟以上）上生成高质量对齐音频的长度泛化能力，在 UnAV100 和 LongVale 基准上大幅超越现有方法。

**[GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization](audio_speech/gem-tfl_bridging_weak_and_full_supervision_for_forgery_localization_through_em-g.md)**

:   提出 GEM-TFL，通过两阶段分类-回归框架弥合弱监督与全监督之间的差距，用 EM 分解二元标签为多维潜在属性、训练无关的时序一致性精化、图扩散提案精化三大模块，在弱监督时序伪造定位上平均 mAP 提升 4-8%。

**[Omni-MMSI: Toward Identity-Attributed Social Interaction Understanding](audio_speech/omni-mmsi_toward_identity-attributed_social_interaction_understanding.md)**

:   提出 Omni-MMSI 任务——从原始音视频输入（而非预处理的 oracle 社交线索）理解多人社交交互，并设计 Omni-MMSI-R 参考引导流水线，通过工具生成身份归因社交线索 + 链式思维推理实现准确的社交交互理解。

**[OmniRet: Efficient and High-Fidelity Omni Modality Retrieval](audio_speech/omniret_efficient_and_high-fidelity_omni_modality_retrieval.md)**

:   提出首个支持文本-视觉-音频三模态组合查询的统一检索模型 OmniRet，通过共享媒体重采样器（Shared Media Resampler）提升计算效率，并引入注意力切片 Wasserstein 池化（ASWP）保留细粒度信息，在 13 个检索任务上取得 12 项领先。

**[OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](audio_speech/omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)**

:   提出 Universal Holistic Audio Generation (UniHAGen) 任务和 OmniSonic 框架，通过 TriAttn-DiT 架构的三路交叉注意力和 MoE 门控机制，首次实现同时生成屏幕内/屏外环境声和人声的统一音频合成，在新构建的 UniHAGen-Bench 上全面超越 SOTA。

**[SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)**

:   提出 SAVE 方法，通过添加专用语音分支（Whisper ASR + CLIP 文本编码器）和 soft-ALBEF 视觉-音频早期对齐策略，实现语音感知的视频表示学习，在五个视频-文本检索基准上全面超越 SOTA。

**[Semantic Audio-Visual Navigation in Continuous Environments](audio_speech/semantic_audio-visual_navigation_in_continuous_environments.md)**

:   本文提出 SAVN-CE 任务，将语义音视觉导航扩展到连续3D环境中，并设计 MAGNet（记忆增强目标描述网络），通过融合历史上下文和自运动线索实现在目标声音消失后的稳健目标推理，成功率绝对提升最高达 12.1%。

**[Solution for 10th Competition on Ambivalence/Hesitancy (AH) Video Recognition Challenge using Divergence-Based Multimodal Fusion](audio_speech/solution_for_10th_competition_on_ambivalencehesitancy_ah_video_recognition_chall.md)**

:   针对第10届 ABAW 竞赛的矛盾/犹豫 (A/H) 视频识别任务，提出基于散度的多模态融合策略，通过计算视觉（AU）、音频（Wav2Vec 2.0）和文本（BERT）三个模态嵌入的逐对绝对差来显式建模跨模态冲突，在 BAH 数据集上以 Macro F1 0.6808 大幅超越基线 0.2827。

**[Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach](audio_speech/team_ras_in_10th_abaw_competition_multimodal_valen.md)**

:   首次将 VLM（Qwen3-VL-4B-Instruct）提取的情感行为描述嵌入作为独立第三模态，与 GRADA 人脸编码器和 WavLM 音频特征通过 DCMMOE 和 RAAV 两种融合策略组合，在 Aff-Wild2 上达到连续 VA 估计 CCC 0.658（dev）/ 0.62（test），验证了 VLM 行为语义对连续情感识别的价值。

**[Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](audio_speech/tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)**

:   提出 TSD 框架，将多模态特征显式分解为全局共享/成对共享/模态专属三个互补子空间，并通过子空间感知跨注意力融合模块自适应整合三层信息，在 CMU-MOSI/MOSEI 上全面 SOTA。

**[UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark](audio_speech/unim_a_unified_any-to-any_interleaved_multimodal_benchmark.md)**

:   提出首个统一的任意到任意交错多模态基准 UniM（31K 样本、7 种模态、30 个领域），配套三维评估体系和基于可追溯推理的智能体基线 UniMA，揭示现有 MLLM 在交错多模态范式下的严重不足。

**[Unlocking Strong Supervision: A Data-Centric Study of General-Purpose Audio Pre-Training Methods](audio_speech/unlocking_strong_supervision_a_data-centric_study_of_general-purpose_audio_pre-t.md)**

:   本文通过系统的数据中心实验证明音频预训练性能主要由标签/监督质量驱动而非模型设计，提出 Unified Tag System (UTS) 将语音、音乐、环境音统一到 800-3k 标签的高粒度词表中，UTS 训练的模型用 5 倍更少的数据在语音（VoxCeleb2）和音乐（MusicCaps）等域外任务上超越 AudioSet 基线。

**[ViDscribe: Multimodal AI for Customizing Audio Description and Question Answering in Online Videos](audio_speech/vidscribe_multimodal_ai_for_customizing_audio_description_and_question_answering.md)**

:   ViDscribe 是一个基于 Web 的平台，利用多模态大语言模型(Gemini 3 Pro)为盲人和低视力(BLV)用户提供可定制的 AI 生成音频描述(AD)和交互式视觉问答(VQA)功能，支持任意 YouTube 视频，通过为期一周的纵向用户研究验证了定制化 AD 在有效性、享受度和沉浸感方面均优于默认 AD。

---

## 🦾 LLM Agent { #llm_agent }

**[ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search](llm_agent/argos_agentic_multi_camera_person_search.md)**

:   本文提出 ARGOS，首个将多摄像头行人搜索重新定义为交互式推理问题的基准和框架，智能体通过与目击者进行多轮对话、调用时空工具并在信息不对称下推理排除候选人，包含 2,691 个任务、3 个渐进式赛道。

**[CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](llm_agent/carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)**

:   提出CareFlow基准（1050个医疗软件长视界工作流任务，8-24步，覆盖DICOM/3D Slicer/EMR/LIS四大系统）和CarePilot框架（基于Actor-Critic范式，集成工具grounding和双记忆机制），在CareFlow上超越GPT-5约15%的任务准确率。

**[EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration](llm_agent/echotrail-gui_building_actionable_memory_for_gui_agents_via_critic-guided_self-e.md)**

:   提出EchoTrail-GUI三阶段闭环框架：探索Agent自主与GUI环境交互生成轨迹 → Critic奖励模型过滤仅保留高质量轨迹构建记忆库(EchoTrail-4K) → 新任务到来时通过密集+稀疏混合检索注入最相关记忆引导推理，将无状态GUI Agent转变为记忆增强系统，在AndroidWorld上GPT-4o达51.7% SR(+17.2pp)，在AndroidLab上Qwen2.5-VL-72B SR从23.9%提升至37.5%。

**[Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos](llm_agent/ego2web_a_web_agent_benchmark_grounded_in_egocentric_videos.md)**

:   提出 Ego2Web，首个将第一人称视频感知与 Web 代理执行相结合的基准测试，配套半自动数据构建流程和 Ego2WebJudge 自动评测框架，实验揭示当前最强 Agent 在真实视觉感知到在线行动的跨模态迁移上仍有巨大差距，最高仅 48.2% 成功率。

**[EpiAgent: An Agent-Centric System for Ancient Inscription Restoration](llm_agent/epiagent_agent_centric_system_for_ancient_inscription_restoration.md)**

:   EpiAgent是首个面向古代铭文修复的Agent系统，通过LLM中央规划器协调多模态分析、专用修复工具和迭代自我优化，在文字真实性和视觉保真度上超越现有方法。

**[Gen-n-Val: Agentic Image Data Generation and Validation](llm_agent/gen_n_val_agentic_image_data_generation_and_validation.md)**

:   本文提出 Gen-n-Val，一个基于智能体的合成数据生成与验证框架，通过 LLM 优化 Layer Diffusion 的 prompt 生成高质量单物体透明图像，再用 VLLM 过滤低质量样本，将无效合成数据从 50% 降至 7%，在 LVIS 稀有类实例分割上提升 7.6% mAP。

**[GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](llm_agent/gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)**

:   提出 GUI-CEval，首个面向中文移动端 GUI Agent 的综合评测基准，覆盖 201 个主流中文 App、4 种设备类型，采用"基础能力+应用能力"两层结构从感知、规划、反思、执行、评估五个维度进行细粒度诊断，在 20 个代表性模型上的实验揭示当前模型在反思和自我评估方面仍有明显短板。

**[HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](llm_agent/hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)**

:   提出难度感知的轨迹合成框架 HATS，通过 hardness-driven exploration 和 alignment-guided refinement 的闭环机制，专注采集和修正语义歧义动作的训练轨迹，大幅提升 GUI Agent 在复杂真实场景中的泛化能力。

**[HAVEN: Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](llm_agent/haven_hierarchical_long_video_understanding_with_audiovisual_entity_cohesion.md)**

:   HAVEN 提出音视频实体凝聚 + 层次索引 + Agent搜索的统一框架，通过说话人身份作为跨模态一致性信号，构建全局-场景-片段-实体四级层次数据库，在LVBench上达到84.1%整体准确率的SOTA。

**[Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code](llm_agent/nerfify_multiagent_nerf_paper_to_code.md)**

:   提出 Nerfify，一个领域感知的多智能体框架，通过上下文无关文法（CFG）约束、图思维（GoT）代码合成和组合式引用依赖恢复，将 NeRF 论文自动转化为可训练的 Nerfstudio 插件，实现 100% 可执行率，视觉质量与专家实现仅差 ±0.5 dB PSNR。

**[REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting](llm_agent/realm_mllm_agent_3d_reasoning_gaussian.md)**

:   提出 REALM 框架，利用 MLLM 的推理能力通过全局到局部空间定位策略在 3DGS 上进行开放世界 3D 推理分割，无需 3D 后训练即可处理隐式指令，在 LERF 上 mIoU 达 92.88%，远超基线方法 40+ 个百分点，并支持物体移除、替换和风格迁移等编辑任务。

**[SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](llm_agent/sceneassistant_a_visual_feedback_agent_for_openvoc.md)**

:   提出SceneAssistant——基于纯视觉反馈的VLM agentic框架，设计14个功能完备的Action API让Gemini-3.0-Flash在ReAct闭环中迭代生成和优化开放词汇3D场景，无需预定义空间关系模板或外部布局求解器，在30个场景的人类评估中Layout得分7.600（vs SceneWeaver 5.800），Human Preference 65%。

**[Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](llm_agent/think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)**

:   提出 VideoHV-Agent，将长视频问答重新建模为"假设-验证"过程：Thinker 将答案选项改写为可测试假设，Judge 提取区分性线索，Verifier 在视频中定位证据进行验证，Answer 综合证据给出最终答案，在 EgoSchema/NextQA/IntentQA 三个基准上取得 SOTA，同时推理效率优于现有 Agent 方法。

**[Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](llm_agent/towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)**

:   首次系统研究离散扩散视觉语言模型（DVLM）在 GUI Grounding 中的应用，将 LLaDA-V 适配为单步动作预测，并提出混合掩码调度（线性+确定性）以捕获边界框坐标间的几何层次依赖，在 Web/Desktop/Mobile 界面上展示了扩散模型作为 GUI Agent 基础的可行性。

**[WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](llm_agent/worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)**

:   提出 WorldMM，一个基于多模态记忆的视频推理 agent，构建情景记忆（多时间尺度文本知识图）、语义记忆（持续更新的关系知识图）和视觉记忆（帧级检索库）三类互补记忆，通过自适应多轮检索 agent 动态选择最相关的记忆源和时间粒度，在五个长视频 QA 基准上平均超越前 SOTA 8.4%。

---

## 🛰️ 遥感 { #remote_sensing }

**[ACPV-Net: All-Class Polygonal Vectorization for Seamless Vector Map Generation from Aerial Imagery](remote_sensing/acpv-net_all-class_polygonal_vectorization_for_seamless_vector_map_generation_fr.md)**

:   提出 ACPV-Net，首个从航空影像一次性生成拓扑一致的全类别多边形矢量地图的框架，通过语义监督条件化扩散模型生成顶点热图，并借助命题驱动的 PSLG 重建确保零间隙/零重叠。

**[AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network](remote_sensing/avion_aerial_visionlanguage_instruction_from_offli.md)**

:   AVION 提出一种知识蒸馏框架，通过 LLM 生成语义丰富的遥感文本原型作为 Teacher 监督、同时在 Student 的视觉和文本编码器中注入可学习 prompt，实现三维度对齐蒸馏，在少样本分类和跨模态检索上显著优于现有 PEFT 方法。

**[Conflated Inverse Modeling for Urban Vegetation Patterns](remote_sensing/conflated_inverse_urban_vegetation.md)**

:   提出融合正向预测模型和扩散逆向生成模型的框架，在指定温度变化目标下生成多样且物理合理的城市植被空间配置（NDVI 模式），多样性提升 3.4 倍同时温度控制误差降低 37%。

**[Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](remote_sensing/cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)**

:   提出跨模态模糊对齐网络 CFAN，利用模糊逻辑量化 token 级可靠性实现精细对齐，并引入地面视图作为桥接代理缓解航拍图像与文本的语义鸿沟，同时构建了大规模文本-航拍行人检索基准 AERI-PEDES。

**[Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction](remote_sensing/exploring_spatiotemporal_feature_propagation_for_video-level_compressive_spectra.md)**

:   首次将光谱压缩成像（SCI）从图像级推进到视频级重建，构建首个高质量动态高光谱数据集 DynaSpec（30 序列/300 帧），提出 PG-SVRT 通过空间-然后-时间注意力 + 桥接 token 实现 41.52dB PSNR 和最优时间一致性，且 FLOPs（28.18G）低于多个图像级 SOTA。

**[GeoFlow: Real-Time Fine-Grained Cross-View Geolocalization via Iterative Flow Prediction](remote_sensing/geoflow_real-time_fine-grained_cross-view_geolocalization.md)**

:   提出 GeoFlow，一种受流匹配启发的轻量级跨视图精细地理定位框架，通过学习概率位移场结合迭代精化采样（IRS）算法，在连续空间内实现从地面图像到卫星图像的精确 2-DoF 定位，以 29 FPS 的实时速度达到了与 SOTA 可比的精度。

**[GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing](remote_sensing/geommbench_and_geommagent_toward_expert_level_multimodal_intelligence_in_geoscience_and_remote_sensing.md)**

:   提出 GeoMMBench（1053 道专家级地球科学多选题）和 GeoMMAgent（检索-感知-推理多智能体框架），系统评估 36 个 MLLM 在遥感领域的能力，揭示领域知识、感知接地和推理方面的系统性不足。

**[Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users](remote_sensing/joint_and_streamwise_distributed_mimo_satellite_communications_with_multi-antenn.md)**

:   提出面向多天线地面用户的分布式LEO卫星下行链路两种传输方案（联合传输 & 流式传输），通过基于统计CSI的WMMSE预编码设计和基于匈牙利算法的流-卫星关联策略，在无需卫星间相位同步的前提下实现了高频谱效率与低前传开销的灵活折中。

**[Lumosaic: Hyperspectral Video via Active Illumination and Coded-Exposure Pixels](remote_sensing/lumosaic_hyperspectral_video_via_active_illumination_and_coded-exposure_pixels.md)**

:   提出Lumosaic主动高光谱视频系统，将12个窄带LED阵列与编码曝光像素（CEP）相机在微秒级同步，在每帧158个子帧内联合编码空间-时间-光谱信息，实现30fps VGA分辨率31通道（400–700nm）运动鲁棒高光谱视频重建，PSNR比被动快照系统高10+dB。

**[MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging](remote_sensing/metaspectra_a_compact_broadband_metasurface_camera_for_snapshot_hyperspectral_im.md)**

:   MetaSpectra+ 提出超表面-折射透镜混合光学范式，通过双层超表面独立控制4通道色散/曝光/偏振，实现250nm宽带、17mm最短光程的快照式高光谱+HDR/偏振多功能成像，在KAUST基准上PSNR达33.31dB全面超越现有快照高光谱系统。

**[No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors](remote_sensing/no_labels_no_look-ahead_unsupervised_online_video_stabilization_with_classical_p.md)**

:   提出无监督在线视频稳定框架 LightStab，通过经典三阶段管线（运动估计→运动传播→运动补偿）搭配多线程异步缓冲，在 5 个基准数据集上首次让在线方法全面媲美离线 SOTA，并发布首个包含可见光和红外的多模态无人机航拍稳定测试集 UAV-Test。

**[Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments](remote_sensing/olbedo_an_albedo_and_shading_aerial_dataset_for_large-scale_outdoor_environments.md)**

:   Olbedo 提出首个大规模真实航拍反照率-着色分解数据集（5664 张 UAV 图像、4 种地貌、跨年多光照），通过物理逆渲染管线生成多视图一致的伪真值标注，证明合成预训练+Olbedo LoRA 微调可以显著提升室外反照率预测并支持重光照/材质编辑/场景变化分析等下游应用。

**[Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration?](remote_sensing/pretrained_image_matchers_for_sar_optical_satellite_registration.md)**

:   本文在零样本设置下评估了24个预训练图像匹配器族在SAR-光学卫星配准上的表现，发现部署协议选择（几何模型、tile大小等）对精度的影响可达33倍，有时超过更换匹配器本身的效果。

**[RHO: Robust Holistic OSM-Based Metric Cross-View Geo-Localization](remote_sensing/rho_robust_holistic_osm-based_metric_cross-view_geo-localization.md)**

:   提出首个面向恶劣天气和传感器噪声的OSM-based度量级跨视角定位基准CV-RHO（270万+ 图像），并设计双分支Pin-Pan架构RHO模型，结合全景去畸变（SUM）和位置-朝向融合（POF）机制，在多种退化条件下将定位性能提升高达20%。

**[SDF-Net: Structure-Aware Disentangled Feature Learning for Optical-SAR Ship Re-identification](remote_sensing/sdfnet_structureaware_disentangled_feature_learnin.md)**

:   提出SDF-Net——物理引导的结构感知解耦特征学习网络，通过中间层梯度能量提取几何结构一致性(SCL)和终端层共享/模态专用特征解耦(DFL)+无参数加法融合，在HOSS-ReID上mAP达60.9%(+3.5% vs SOTA TransOSS)。

---

## 💡 LLM 推理 { #llm_reasoning }

**[Beyond Geometry: Artistic Disparity Synthesis for Immersive 2D-to-3D](llm_reasoning/beyond_geometry_artistic_disparity_synthesis_for_immersive_2d-to-3d.md)**

:   提出"艺术视差合成"新范式（Art3D），将2D-to-3D转换目标从几何精度转向艺术表达，通过双路径架构解耦全局深度风格与局部艺术效果，从专业3D电影数据中学习导演意图。

**[E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought](llm_reasoning/e-comiq-zh_a_human-aligned_dataset_and_benchmark_for_fine-grained_evaluation_of_.md)**

:   构建首个面向中文电商海报的多维度质量评估框架 E-comIQ-ZH，包含18K专家标注数据集（含CoT推理链）、专用评估模型 E-comIQ-M（SFT+GRPO训练）和标准化基准 E-comIQ-Bench。

**[EagleVision: A Dual-Stage Framework with BEV-grounding-based Chain-of-Thought for Spatial Intelligence](llm_reasoning/eaglevision_a_dual-stage_framework_with_bev-grounding-based_chain-of-thought_for.md)**

:   提出EagleVision双阶段框架，宏观感知阶段用语义-视角融合DPP(SPF-DPP)在SE(3)空间联合优化语义相关性和视角多样性选择关键帧，微观验证阶段让模型在BEV平面上主动查询新视角帧进行迭代空间CoT推理（假设→查看→验证闭环），查询策略纯RL训练无需人工标注，在VSI-Bench和SQA3D上达开源SOTA。

**[FaceCoT: Harnessing Chain-of-Thought Reasoning in MLLMs for Face Anti-Spoofing](llm_reasoning/facecot_cot_reasoning_face_anti_spoofing.md)**

:   构建了首个面向人脸反欺骗（FAS）的大规模 VQA 数据集 FaceCoT（108 万样本，覆盖 14 种攻击类型），包含六层级 CoT 推理标注；提出 CoT-Enhanced Progressive Learning (CEPL) 两阶段训练策略（先视觉增强再联合训练），在 11 个跨域基准上平均 AUC 提升 4.06%、HTER 降低 5.00%。

**[GRAZE: Grounded Refinement and Motion-Aware Zero-Shot Event Localization](llm_reasoning/graze_grounded_refinement_and_motion-aware_zero-shot_event_localization.md)**

:   提出GRAZE，一种完全无训练的时空事件定位管线——用Grounding DINO发现候选player-dummy交互对，通过运动感知的几何评分（位移幅度+方向余弦相似度）排序候选，再用SAM2掩码传播作为独立的像素级接触验证器（而非依赖检测置信度），配合两阶段后向精化恢复事件起始帧，在738个橄榄球练习视频上97.4%有效输出率、77.5%在±10帧内定位。

**[Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving](llm_reasoning/latent_chain-of-thought_world_modeling_for_end-to-end_autonomous_driving.md)**

:   LCDrive 提出潜在链式思考（Latent CoT）框架，用动作提议token和世界模型预测token替代自然语言CoT进行推理，通过冷启动+RL后训练实现更低延迟、更好轨迹质量的端到端自动驾驶。

**[Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](llm_reasoning/rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)**

:   发现现有LVLM在CoT推理时实际上忽略了中间rationale的内容，提出 RED (Rationale-Enhanced Decoding)——将图像条件和rationale条件的next-token分布在logit层面相乘，理论上等价于KL约束奖励最大化的最优解，无需训练即可显著提升多模态推理准确率。

**[Reinforcing Structured Chain-of-Thought for Video Understanding](llm_reasoning/reinforcing_structured_chain-of-thought_for_video_understanding.md)**

:   提出 SDRL（Summary-Driven Reinforcement Learning），一种无需 SFT 的单阶段 RL 框架，通过结构化 CoT（Summarize→Think→Answer）和两个自监督机制（CVK 和 DVR）增强视频时序推理，在 7 个 VideoQA 基准上达到 SOTA。

**[Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering](llm_reasoning/step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)**

:   构建首个对齐临床诊断工作流的结构化多步CoT医学推理数据集Step-CoT（10K+病例/70K QA对），并提出基于图注意力网络的教师-学生框架实现逐步推理监督，提升Med-VQA的准确性和可解释性。

**[Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models](llm_reasoning/understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model.md)**

:   本文系统分析了多模态 CoT 模型中幻觉的成因，发现"发散思维"（associative reasoning）是幻觉的核心触发因素，并提出基于视觉熵的免训练检测+解码干预策略，在 Object HalBench 上将 CHAIRS 降低超过 30%，同时保持甚至提升通用推理能力。

**[Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models](llm_reasoning/understanding_the_role_of_hallucination_in_reinforcement_post-training_of_multim.md)**

:   本文提出 Hallucination-as-Cue 分析框架，通过三种模态特定腐蚀策略（空白图像、随机图像、文本移除）系统研究 RL 后训练对多模态推理模型的真实作用机制，发现即使在 100% 腐蚀视觉输入下 GRPO 训练仍能显著提升推理性能，挑战了"RL 训练能有效利用视觉信息"的主流假设。

**[VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models](llm_reasoning/visref_visual_refocusing_test_time_scaling.md)**

:   提出 VisRef，一个免训练的视觉重聚焦框架——在多模态大推理模型（MLRM）的推理过程中，通过行列式点过程（DPP）在每步自适应选择与当前推理状态语义相关且视觉覆盖多样的 token 子集并重新注入，同时用基于熵的停止准则防止过度推理，在固定计算预算下将视觉推理准确率提升最高 6.4%。

---

## ⚖️ 对齐 / RLHF { #llm_alignment }

**[Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group](llm_alignment/bases_of_steerable_kernels_for_equivariant_cnns_from_2d_rotations_to_the_lorentz.md)**

:   提出一种绕过 Clebsch-Gordan 系数的方法来求解等变CNN中的可转向核（steerable kernel）约束，通过在稳定子群上求解简单的不变性条件再"转向（steer）"到任意点，为 SO(2) 到 Lorentz 群等不同对称群给出了显式的核基底。

**[Bias at the End of the Score: Demographic Biases in Reward Models for T2I](llm_alignment/bias_reward_models_t2i.md)**

:   对文本到图像生成中广泛使用的奖励模型（PickScore、ImageReward、HPS 等）进行大规模人口统计偏差审计，发现奖励引导优化会不成比例地性化女性图像、使人口统计收敛到白人、且奖励分数与现实世界的人口频率先验相关。

**[GlyphPrinter: Region-Grouped Direct Preference Optimization for Glyph-Accurate Visual Text Rendering](llm_alignment/glyphprinter_region-grouped_direct_preference_optimization_for_glyph-accurate_vi.md)**

:   提出 GlyphPrinter，通过构建区域级字形偏好数据集 GlyphCorrector 和区域分组 DPO（R-GDPO）目标函数，在不依赖显式奖励模型的情况下显著提升视觉文本渲染的字形准确度，并引入推理时 Regional Reward Guidance 实现可控生成。

**[MapReduce LoRA: Advancing the Pareto Front in Multi-Preference Optimization for Generative Models](llm_alignment/mapreduce_lora_advancing_the_pareto_front_in_multi-preference_optimization_for_g.md)**

:   提出 MapReduce LoRA 和 RaTE 两种互补方法来推进多偏好优化的 Pareto 前沿：前者通过"Map（并行训偏好专家）+ Reduce（迭代合并）"的策略渐进推进 Pareto 前沿；后者通过学习奖励感知的 token embedding 实现推理时可组合的偏好控制。

**[Mesh-Pro: Asynchronous Advantage-guided Ranking Preference Optimization for Artist-style Quadrilateral Mesh Generation](llm_alignment/mesh-pro_asynchronous_advantage-guided_ranking_preference_optimization_for_artis.md)**

:   提出 Mesh-Pro，首个面向3D四边形网格生成的异步在线强化学习框架，核心算法 ARPO（Advantage-guided Ranking Preference Optimization）通过 Plackett-Luce 排名模型与优势函数加权相结合，在效率（较离线 DPO 快 3.75x）和泛化性上同时取得提升，实现 artist-style 和 dense mesh 的 SOTA 生成质量。

**[LocalDPO: Direct Localized Detail Preference Optimization for Video Diffusion Models](llm_alignment/mind_the_generative_details_direct_localized_detail_preference_optimization_for_.md)**

:   提出LocalDPO，通过对真实高质量视频进行随机时空Bézier掩码的局部腐蚀生成负样本（单次推理、无需外部排序），配合区域感知DPO损失在局部细节级别进行偏好对齐，在Wan2.1和CogVideoX上一致超越传统DPO和SFT的视频质量。

**[MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization](llm_alignment/mod-dpo_towards_mitigating_cross-modal_hallucinations_in_omni_llms_using_modalit.md)**

:   提出 MoD-DPO（Modality-Decoupled DPO），通过不变性正则化、敏感性正则化和语言先验去偏三个机制解耦多模态 LLM 中各模态的贡献，有效缓解跨模态幻觉（如用听觉信息回答视觉问题），并推导出闭式最优策略。

**[PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization](llm_alignment/physmodpo_physicallyplausible_humanoid_motion_with.md)**

:   提出PhysMoDPO，将预训练的全身控制器（WBC/DeepMimic）集成到扩散运动生成器的后训练流程中，通过物理仿真自动构造偏好对并用DPO微调，使生成运动在WBC执行后同时满足物理可行性和文本/空间条件忠实度，实现零样本迁移到Unitree G1真实机器人。

**[Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](llm_alignment/principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)**

:   提出 NullSteer，一种基于零空间投影的激活转向防御框架，通过将转向操作限制在良性激活的零空间中，在不损害模型通用能力的前提下有效抵御视觉越狱攻击。

**[$\varphi$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models](llm_alignment/φ-dpo_fairness_direct_preference_optimization_approach_to_continual_learning_in_.md)**

:   提出 $\varphi$-DPO，将 DPO 作为持续学习范式（以前一步模型为参考策略），并引入受 focal loss 启发的公平性调制因子 $(1-p)^\gamma$ 来平衡不同数据组间的梯度贡献，在理论上证明 $\gamma \to \infty$ 时梯度偏差趋于零，在 CoIN 和 MLLM-CL 基准上达到 SOTA。

---

## 💬 LLM / NLP { #llm_nlp }

**[Bi-CMPStereo: Bidirectional Cross-Modal Prompting for Event-Frame Asymmetric Stereo](llm_nlp/bi_cmpstereo_bidirectional_cross_modal_prompting_for_event_frame_asymmetric_stereo.md)**

:   提出 Bi-CMPStereo，一种双向跨模态提示框架，交替将事件和帧设为目标域进行立体规范化约束和跨域嵌入适配，同时利用两个方向的代价体实现鲁棒的事件-帧非对称立体匹配。

**[Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting](llm_nlp/boosting_quantitive_and_spatial_awareness_for_zero-shot_object_counting.md)**

:   提出QICA框架解决零样本目标计数中的数量感知缺失和空间不敏感问题，通过数量条件化的协同提示策略（SPS）联合适配视觉-语言编码器，结合在相似度图上直接操作的代价聚合解码器（CAD）保持零样本迁移能力，在FSC-147上达到零样本SOTA（MAE 12.41）并展现强跨域泛化。

**[Composing Concepts from Images and Videos via Concept-prompt Binding](llm_nlp/composing_concepts_from_images_and_videos_via_concept-prompt_binding.md)**

:   提出 Bind & Compose (BiCo)，一种one-shot方法，通过层次化binder结构将视觉概念绑定到prompt token，并通过token组合实现图像-视频概念的灵活组合，在概念一致性、prompt保真度和运动质量上全面超越前作。

**[CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection](llm_nlp/cops_conditional_prompt_synthesis_for_zero-shot_anomaly_detection.md)**

:   本文提出 CoPS 框架，通过显式状态token合成（ESTS）和隐式类别token采样（ICTS）两种视觉条件化机制动态生成提示，配合空间感知对齐（SAGA），在13个工业和医学数据集上实现零样本异常检测SOTA。

**[GUIDE: Guided Updates for In-context Decision Evolution in LLM-Driven Spacecraft Operations](llm_nlp/guide_guided_updates_for_in-context_decision_evolution_in_llm-driven_spacecraft_.md)**

:   提出GUIDE框架，利用LLM的in-context学习能力为航天器自主操作提供引导式决策进化，通过结构化的上下文信息和反馈机制让LLM在无需微调的情况下逐步改善航天任务规划和故障诊断决策的质量。

**[Perception Programs: Unlocking Visual Tool Reasoning in Language Models](llm_nlp/perception_programs_visual_tool_reasoning.md)**

:   提出 Perception Programs (P2)，一种训练免费、模型无关的方法，将视觉工具（深度、光流、对应等）的原始输出转换为紧凑的语言原生结构化摘要，使 MLLM 能直接"阅读"视觉模态而非从密集像素推断，在 BLINK 6 个任务上平均提升 19.66%。

**[PhysVid: Physics Aware Local Conditioning for Generative Video](llm_nlp/physvid_physics_aware_local_conditioning_for_generative_video_models.md)**

:   提出 PhysVid，一种物理感知的局部条件化方案——将视频分为时间片段（chunk），由 VLM 为每个 chunk 标注物理现象描述，通过 chunk 级交叉注意力注入生成模型；推理时引入"负物理提示"（反事实引导）引导生成远离物理违规，在 VideoPhy 上将物理常识分数提升约 33%。

**[Sign Language Recognition in the Age of LLMs](llm_nlp/sign_language_recognition_llms.md)**

:   首个系统评估现代 VLM 在零样本孤立手语识别（ISLR）上能力的研究，发现开源 VLM 远落后于专用分类器，但大型商用模型（GPT-5）展现出令人惊讶的潜力。

**[SketchDeco: Training-Free Latent Composition for Precise Sketch Colourisation](llm_nlp/sketchdeco_training-free_latent_composition_for_precise_sketch_colourisation.md)**

:   提出SketchDeco，一种无需训练的线稿上色方法，通过全局-局部两阶段策略将区域蒙版和调色板作为精确控制信号，利用扩散模型反演和自注意力注入在隐空间中实现区域精准着色与全局和谐过渡，在消费级GPU上15-20步即可完成。

---

## 🕸️ 图学习 { #graph_learning }

**[Adaptive Learned Image Compression with Graph Neural Networks](graph_learning/adaptive_learned_image_compression_with_graph_neural_networks.md)**

:   GLIC 把学习图像压缩里的非线性变换从固定卷积或窗口注意力，改造成由图神经网络驱动的内容自适应连接：先用双尺度图决定“连到哪里”，再用复杂度感知机制决定“连多少”，从而更好地建模局部与远程冗余，在三个标准数据集上都显著超过传统编解码器和近期 LIC 强基线。

**[Graph-to-Frame RAG: Visual-Space Knowledge Fusion for Training-Free and Auditable Video Reasoning](graph_learning/graph-to-frame_rag_visual-space_knowledge_fusion_for_training-free_and_auditable.md)**

:   提出 G2F-RAG 范式，将检索到的结构化知识渲染为单帧"推理帧"附加到视频末尾，使大模型在视觉空间内统一推理，避免了文本追加导致的注意力稀释和认知负荷，在 8 个视频基准上实现免训练的一致性提升。

**[Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](graph_learning/graph2eval_automatic_multimodal_task_generation_for_agents_via_knowledge_graphs.md)**

:   提出 Graph2Eval，一个知识图谱驱动的 agent 评估任务自动生成框架——通过从文档/网页构建结构化知识图谱、子图采样、LLM 条件生成和多阶段过滤，自动产出语义一致（+20%）且可解（+17%）的多模态 agent 任务，构建了包含 1319 个任务的 Graph2Eval-Bench。

**[Hyperbolic Busemann Neural Networks](graph_learning/hyperbolic_busemann_neural_networks.md)**

:   利用 Busemann 函数将多类逻辑回归（MLR）和全连接层（FC）内蕴地提升到双曲空间，提出 BMLR 和 BFC 两个统一组件，在 Poincaré 球和 Lorentz 模型上同时适用，且在图像分类、基因组序列、节点分类、链接预测四类任务上均优于已有双曲层。

**[M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](graph_learning/m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)**

:   提出M3KG-RAG，通过轻量多Agent流水线构建多跳多模态知识图谱（M3KG），并设计GRASP机制进行实体定位和选择性剪枝，仅保留查询相关且有助回答的知识，大幅提升MLLM的音视觉推理能力。

**[Mario: Multimodal Graph Reasoning with Large Language Models](graph_learning/mario_multimodal_graph_reasoning_with_large_language_models.md)**

:   提出 Mario，针对多模态图（MMG）上的 LLM 推理，通过图条件视觉语言模型（GVLM）实现拓扑感知的跨模态对齐，再用模态自适应提示路由器（MAPR）为每个节点选择最优模态配置，在节点分类和链接预测上达到 SOTA。

**[ViterbiPlanNet: Injecting Procedural Knowledge via Differentiable Viterbi for Planning](graph_learning/viterbiplannet_injecting_procedural_knowledge_via_differentiable_viterbi_for_pla.md)**

:   将过程知识图（PKG）通过可微Viterbi层端到端嵌入规划模型，使神经网络只需学习发射概率而非记忆完整过程结构，在CrossTask/COIN/NIV上以仅5-7M参数（比扩散/LLM方法少1-3个数量级）达到SOTA成功率，并建立了统一的评估基准。

**[WSGG: Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](graph_learning/wsgg_spatiotemporal_world_scene_graph.md)**

:   本文提出世界场景图生成（WSGG）任务，将传统帧级场景图扩展为在统一世界坐标系下追踪所有物体（包括被遮挡/不可见的），配合 ActionGenome4D 数据集和 PWG/MWAE/4DST 三种互补方法实现持久化场景推理。

---

## 📚 预训练 { #llm_pretraining }

**[Defending Unauthorized Model Merging via Dual-Stage Weight Protection](llm_pretraining/defending_unauthorized_model_merging_via_dual-stage_weight_protection.md)**

:   提出 MergeGuard，一种主动式双阶段权重保护框架：Stage 1通过L2正则化分散任务关键权重，Stage 2注入结构化扰动破坏合并兼容性，在保持保护模型<1.5%性能损失的同时使合并模型精度下降高达90%。

**[Evidential Transformation Network: Turning Pretrained Models into Evidential Models for Post-hoc Uncertainty Estimation](llm_pretraining/evidential_transformation_network_post_hoc_uncertainty_estimation.md)**

:   本文提出 Evidential Transformation Network (ETN)，一个轻量级后置模块，通过在 logit 空间学习样本相关的仿射变换，将预训练分类器或 LLM 转化为证据模型，以最小的计算开销实现可靠的不确定性估计。

**[FlowMotion: Training-Free Flow Guidance for Video Motion Transfer](llm_pretraining/flowmotion_training-free_flow_guidance_for_video_motion_transfer.md)**

:   提出 FlowMotion，一种无需训练的视频运动迁移框架，通过直接利用 flow-based T2V 模型的预测输出（latent prediction）构建运动引导信号，避免对模型内部层做梯度回传，在保持运动保真度的同时大幅降低推理时间和显存开销。

**[Linking Modality Isolation in Heterogeneous Collaborative Perception](llm_pretraining/linking_modality_isolation_in_heterogeneous_collaborative_perception.md)**

:   提出 CodeAlign 框架，通过码本构建离散代码空间和跨模态 Feature-Code-Feature (FCF) 翻译，首次解决异构协同感知中不同模态从未在训练数据中共现的"模态隔离"问题，仅需 HEAL 8% 训练参数、通信量降低 1024 倍，同时达到 SOTA 感知性能。

**[LottieGPT: Tokenizing Vector Animation for Autoregressive Generation](llm_pretraining/lottiegpt_vector_animation_generation.md)**

:   提出首个矢量动画自回归生成框架 LottieGPT，设计了 Lottie 分词器将层级几何体、变换和关键帧运动编码为紧凑 token 序列，构建 660K 动画数据集，基于 Qwen-VL 微调实现从文本/图像直接生成可编辑矢量动画。

**[Model Merging in the Essential Subspace](llm_pretraining/model_merging_in_the_essential_subspace.md)**

:   提出 ESM 框架，通过对参数更新引起的激活偏移做 PCA 构建"本质子空间"（而非直接对参数做 SVD），并用三级极化缩放增强关键参数、抑制噪声，在 ViT-B/32 的 20 任务合并中比 Iso-CTS 提升 3.2%（绝对准确率）。

**[MXNorm: Reusing MXFP Block Scales for Efficient Tensor Normalisation](llm_pretraining/mxnorm_reusing_mxfp_block_scales_for_efficient_tensor_normalisation.md)**

:   GPU矩阵乘法吞吐量提升(80x)远超reduction/elementwise操作(5-9x)，RMSNorm正成为低精度训练的新瓶颈。MXNorm直接复用MXFP8量化时已计算的block scales来估计RMS，实现32倍reduction大小缩减。理论上证明block absmax的广义p-mean可收敛到RMS的常数倍。Llama 3 125M/1B/8B预训练验证MXNorm(p=2)与RMSNorm训练精度差异minimal，torch.compile实测isolated kernel最高2.4x加速、Llama 3 8B transformer layer在MXFP8下+1.3%、NVFP4下+2.6%加速。Drop-in replacement，无额外超参数。

**[Watch and Learn: Learning to Use Computers from Online Videos](llm_pretraining/watch_and_learn_learning_to_use_computers_from_online_videos.md)**

:   提出 Watch & Learn (W&L) 框架，通过逆动力学模型 (IDM) 将互联网上的人类计算机操作视频自动转化为可执行的 UI 轨迹数据，生成 53K+ 高质量轨迹，作为 ICL 示例或 SFT 训练数据显著提升各类 CUA 性能。

---

## 📐 优化/理论 { #optimization }

**[BlazeFL: Fast and Deterministic Federated Learning Simulation](optimization/blazefl_fast_and_deterministic_federated_learning_simulation.md)**

:   提出 BlazeFL，一个基于 Python free-threading 的轻量级单机联邦学习仿真框架，通过共享内存执行和客户端隔离 RNG 流实现最高 3.1× 加速与比特级可复现。

**[Dynamic Momentum Recalibration in Online Gradient Learning](optimization/dynamic_momentum_recalibration_in_online_gradient_learning.md)**

:   从信号处理视角揭示固定动量系数在偏差-方差权衡上的固有缺陷，提出SGDF优化器，通过在线计算最优时变增益（基于最小均方误差原则）动态平衡梯度估计的噪声抑制和信号保持，在多种视觉任务上超越SGD动量和Adam变体。

**[Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning](optimization/enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p.md)**

:   针对联邦原型学习中现有方法破坏类间语义关系的问题，提出FedTSP方法利用预训练语言模型构建保留语义结构的文本原型，在异构联邦学习中显著提升性能并加速收敛。

**[Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift](optimization/fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_.md)**

:   提出 Fed-ADE 框架，通过 uncertainty dynamics estimation 和 representation dynamics estimation 两个轻量级分布漂移信号，为每个客户端在每个时间步自适应调整学习率，实现联邦部署后无监督适应。

**[OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport](optimization/otprune_distribution-aligned_visual_token_pruning_via_optimal_transport.md)**

:   将视觉 token 裁剪建模为最优传输（OT）下的分布对齐问题，通过最小化完整与裁剪后 token 集合间的 2-Wasserstein 距离，以 Gaussian 代理 + log-det 子模目标 + 贪心 Cholesky 选择实现 training-free、$O(mk^2)$ 复杂度的高效裁剪，在 11 个多模态基准上取得 SOTA 精度-效率折中。

**[SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated learning](optimization/scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)**

:   SCOPE 用一个零训练的视觉语言几何打分器，把每个样本压缩成表示性、多样性和负类边界接近度三个标量，再由服务器只聚合这些轻量统计量形成全局共识，指导各客户端先删语义异常样本、再删多数类冗余样本，从而在强非 IID 和长尾联邦场景下兼顾精度、鲁棒性和极低通信开销。

**[The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers](optimization/the_power_of_decaying_steps_enhancing_attack_stability_and_transferability_for_s.md)**

:   将 sign-based 对抗攻击优化器重构为坐标级梯度下降，揭示其非衰减步长是导致不收敛和不稳定的根因，提出单调递减坐标步长策略 MDCS，理论证明 MDCS-MI 达到最优 $O(1/\sqrt{T})$ 收敛率，在图像分类和跨模态检索任务上显著提升攻击迁移性与稳定性。

**[UniFusion: A Unified Image Fusion Framework with Robust Representation and Source-Aware Preservation](optimization/unifusion_a_unified_image_fusion_framework_with_robust_representation_and_source.md)**

:   提出 UniFusion 统一图像融合框架，利用 DINOv3 自监督语义先验构建跨模态共享特征空间，通过重建对齐机制保留源图信息，并以双层优化策略解耦重建与融合目标，在红外-可见光、多曝光、多焦点、医学图像等多任务上均达到 SOTA。

---

## 🔍 信息检索/RAG { #information_retrieval }

**[Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval](information_retrieval/beyond_global_similarity_towards_fine-grained_multi-condition_multimodal_retriev.md)**

:   提出MCMR（Multi-Conditional Multimodal Retrieval）大规模基准，通过双证据设计（部分属性仅可从图像推断、部分仅可从文本获取）确保检索任务不可被单模态解决，系统评估5个检索器和7个MLLM重排器，揭示模态不对称性和细粒度推理差距。

**[CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering](information_retrieval/cc-vqa_conflict-_and_correlation-aware_method_for_mitigating_knowledge_conflict_.md)**

:   提出 CC-VQA，一种 training-free 的知识冲突缓解方法，通过视觉中心的上下文冲突推理和相关度引导的编码/解码两阶段策略，在 E-VQA、InfoSeek、OK-VQA 三个基准上取得 3.3%-6.4% 的绝对精度提升。

**[Explaining CLIP Zero-shot Predictions Through Concepts](information_retrieval/explaining_clip_zero-shot_predictions_through_concepts.md)**

:   本文提出 EZPC，通过学习一个线性投影矩阵将 CLIP 的图像-文本嵌入映射到可解释的概念空间，在几乎不损失零样本分类精度的前提下（CIFAR-100/CUB/ImageNet-100 上 H-mean 仅差约 1%），为 CLIP 的预测提供基于人类可理解概念的忠实解释，且推理开销仅增加约 0.1ms。

**[M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](information_retrieval/m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)**

:   提出首个大规模多语言多文化多模态 RAG 评估框架 M4-RAG，覆盖 42 种语言和 189 个国家的 80K+ 文化 VQA 实例，系统性揭示了 RAG 对小模型有效但无法随模型规模正向扩展、跨语言检索存在严重性能退化的关键发现。

**[MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model](information_retrieval/muco_multi-turn_contrastive_learning_for_multimodal_embedding_model.md)**

:   MuCo 提出了一种基于多轮对话的对比学习框架，利用 MLLM 的对话能力在单次前向传播中同时处理多个关联的 query-target 对，大幅提升训练效率，并在 MMEB 和 M-BEIR 检索基准上取得 SOTA 性能。

**[NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval](information_retrieval/nanovdr_distilling_a_2b_vision-language_retriever_into_a_70m_text-only_encoder_f.md)**

:   NanoVDR 利用查询-文档的模态不对称性，将 2B VLM 教师的查询编码能力通过 pointwise cosine alignment 蒸馏到 69M 纯文本编码器，在 ViDoRe 基准上保留 95.1% 教师性能、查询延迟降低 50 倍，训练仅需 13 GPU 小时。

**[RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations](information_retrieval/robustvisrag_causality-aware_vision-based_retrieval-augmented_generation_under_v.md)**

:   提出 RobustVisRAG，一个因果引导的双路径框架，通过非因果路径捕获退化信号、因果路径学习纯净语义来解耦 VisRAG 中的语义-退化纠缠，在真实世界退化条件下检索、生成和端到端性能分别提升 7.35%、6.35% 和 12.40%，同时保持干净数据上的性能。

---

## 📈 时间序列 { #time_series }

**[A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens](time_series/a_frame_is_worth_one_token_efficient_generative_world_modeling_with_delta_tokens.md)**

:   提出 DeltaTok 将连续帧的 VFM 特征差压缩为单个 delta token，配合 Best-of-Many 训练的 DeltaWorld 在单次前向传播中高效生成多样化未来预测，参数量仅为 Cosmos 的 1/35、FLOPs 仅为 1/2000，但在密集预测任务上表现更优。

**[Competition-Aware CPC Forecasting with Near-Market Coverage](time_series/competition-aware_cpc_forecasting_with_near-market_coverage.md)**

:   这篇论文把搜索广告中的 CPC 预测重新表述为“竞争状态部分不可观测”下的时间序列预测问题，用语义相似性、CPC 轨迹对齐和地理意图三个可观测代理去近似隐含竞争，再分别以协变量和图先验两种形式注入预测器，在中长期预测上显著优于纯自回归基线。

**[L2GTX: From Local to Global Time Series Explanations](time_series/l2gtx_from_local_to_global_time_series_explanations.md)**

:   L2GTX 提出一种完全模型无关的局部到全局解释方法，通过从 LOMATCE 局部解释中提取参数化时间事件原语（趋势/极值），跨实例合并冗余聚类并以子模优化选取代表性实例，最终聚合为简洁的类级别全局解释，在6个时序分类数据集上保持稳定的全局忠实度。

**[PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning](time_series/pfgnet_a_fully_convolutional_frequency-guided_peripheral_gating_network_for_effi.md)**

:   提出 PFGNet，一种纯卷积时空预测框架，通过像素级频率引导门控（PFG）动态调制多尺度大核外周响应并施加可学习中心抑制，模拟生物视觉的 center-surround 带通滤波机制，在 Moving MNIST、TaxiBJ、KTH、Human3.6M 四个基准上以极少参数和计算量达到 SOTA 或近 SOTA 性能。

**[Stable Spike: Dual Consistency Optimization via Bitwise AND Operations for Spiking Neural Networks](time_series/stable_spike_dual_consistency_optimization_via_bitwise_and_operations_for_spikin.md)**

:   提出 Stable Spike 双一致性优化框架，利用硬件友好的 AND 位运算从多时间步脉冲图中解耦稳定脉冲骨架，并注入振幅感知脉冲噪声增强泛化，在超低延迟(T=2)下将神经形态物体识别精度提升最高 8.33%。

**[STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting](time_series/stcast_adaptive_boundary_alignment_for_global_and_regional_weather_forecasting.md)**

:   提出STCast框架，通过Spatial-Aligned Attention（SAA）用可学习的全球-区域分布替代静态边界来自适应融合全球大气信息到区域预报，并用Temporal Mixture-of-Experts（TMoE）按月动态路由专家增强时序建模，在全球预报、高分辨率区域预报、台风路径预测和集合预测四个任务上全面超越现有方法。

---

## 📡 信号/通信 { #signal_comm }

**[AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](signal_comm/actta_rethinking_test-time_adaptation_via_dynamic_activation.md)**

:   本文提出 AcTTA，一种基于动态激活函数调制的测试时自适应框架，通过将传统固定激活函数重参数化为可学习形式（包含激活中心偏移和非对称梯度斜率），在推理时自适应调整激活行为以应对分布偏移，在 CIFAR10-C/CIFAR100-C/ImageNet-C 上一致超越基于归一化层的 TTA 方法。

**[ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding](signal_comm/chartnet_a_million-scale_high-quality_multimodal_dataset_for_robust_chart_unders.md)**

:   提出 ChartNet，一个包含 150 万条高质量多模态对齐样本的百万级图表理解数据集，通过代码引导的合成管线生成涵盖 24 种图表类型、6 种绘图库的五元组数据（代码、图像、数据表、文本描述、带推理的 QA），在 ChartNet 上微调的 2B 模型可超越 GPT-4o 和 72B 开源模型。

**[CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space](signal_comm/clay_conditional_visual_similarity.md)**

:   CLAY 提出免训练的条件视觉相似度计算方法，通过在 VLM 嵌入空间中构建文本条件子空间来调制相似度，无需重新计算数据库特征即可适应不同检索条件，并支持多条件检索。

**[Dual-Imbalance Continual Learning for Real-World Food Recognition](signal_comm/dual-imbalance_continual_learning_for_real-world_food_recognition.md)**

:   提出 DIME 框架，通过类别计数引导的光谱适配器合并和秩自适应阈值调制机制，在双重不平衡（类内长尾分布 + 步间类别数不均匀）的持续学习场景下，在四个长尾食物数据集上持续超越 baseline 3% 以上。

**[FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection](signal_comm/faar_efficient_frequency-aware_multi-task_fine-tuning_via_automatic_rank_selecti.md)**

:   提出 FAAR，一种频率感知的多任务参数高效微调方法，通过 Performance-Driven Rank Shrinking (PDRS) 为每个任务和层动态选择最优秩，并设计 Task-Spectral Pyramidal Decoder (TS-PD) 利用 FFT 频率信息增强空间感知和跨任务一致性，以传统微调 1/9 的参数量实现更优性能。

---

## ⚡ LLM 效率 { #llm_efficiency }

**[GeoCodeBench: Benchmarking PhD-Level Coding in 3D Geometric Computer Vision](llm_efficiency/benchmarking_phd-level_coding_in_3d_geometric_computer_vision.md)**

:   首个面向3D几何计算机视觉的PhD级代码生成基准GeoCodeBench，包含100个从2025年顶会论文+代码库中精选的函数补全任务，配套自动化多样化单元测试，最强模型GPT-5仅36.6%通过率，揭示LLM在科学级3D代码实现上的巨大差距。

**[CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach](llm_efficiency/cheem_continual_learning_by_reuse_new_adapt_and_skip_--_a_hierarchical_explorati.md)**

:   提出 CHEEM 框架，通过分层探索-利用采样的 NAS 自动学习任务感知的动态 ViT 骨干——在每一层选择 Reuse/New/Adapt/Skip 四种操作——在 MTIL 和 VDD 两个挑战性持续学习基准上显著超越提示类方法，接近全量微调上界。

**[SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration](llm_efficiency/sparvar_exploring_sparsity_in_visual_autoregressive_modeling_for_training-free_a.md)**

:   对VAR模型注意力激活模式进行系统分析，揭示三大稀疏特性（注意力汇、跨尺度相似性、空间局部性），并提出SparVAR无训练加速框架，通过跨尺度自相似稀疏注意力（CS⁴A）和跨尺度局部稀疏注意力（CSLA）两个即插即用模块，实现8B模型1024×1024生成降至1秒级（1.57×加速），且几乎不损失高频细节。

**[StoryTailor: A Zero-Shot Pipeline for Action-Rich Multi-Subject Visual Narratives](llm_efficiency/storytailora_zero-shot_pipeline_for_action-rich_multi-subject_visual_narratives.md)**

:   提出StoryTailor零样本视觉叙事生成管线，通过高斯中心注意力（GCA）缓解主体重叠和背景泄漏、动作增强奇异值重加权（AB-SVR）放大动作语义、选择性遗忘缓存（SFC）维护跨帧背景连续性，在单张RTX 4090上实现多主体、动作丰富的图像叙事生成，CLIP-T较基线提升10-15%。

---

## 🧮 科学计算 { #scientific_computing }

**[Continuous Exposure-Time Modeling for Realistic Atmospheric Turbulence Synthesis](scientific_computing/continuous_exposure-time_modeling_for_realistic_atmospheric_turbulence_synthesis.md)**

:   提出曝光时间依赖的调制传递函数（ET-MTF），将曝光时间建模为连续变量，构建了大规模合成湍流数据集 ET-Turb（5083视频、200万帧），显著提升湍流复原模型在真实数据上的泛化能力。

**[EHETM: High-Quality and Efficient Turbulence Mitigation with Events](scientific_computing/high-quality_and_efficient_turbulence_mitigation_with_events.md)**

:   提出EHETM，首次利用事件相机的微秒时间分辨率突破传统多帧湍流缓解(TM)方法的精度-效率瓶颈，发现两个关键物理现象——湍流诱导事件的极性交替与清晰梯度相关、动态物体形成时空相干"事件管"——设计极性加权梯度和事件管约束两个互补模块，数据开销降低77.3%、系统延迟降低89.5%，尤其在动态物体场景显著超越SOTA。

**[NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training](scientific_computing/nestor_a_nested_moe-based_neural_operator_for_large-scale_pde_pre-training.md)**

:   提出嵌套式 MoE 神经算子 NESTOR，通过 image-level MoE 捕获不同 PDE 类型的全局特征 + token-level Sub-MoE 捕获物理场内局部相关性，在 12 个 PDE 数据集上实现大规模预训练并有效迁移到下游任务。

**[PhysSkin: Real-Time and Generalizable Physics-Based Skin Simulation](scientific_computing/physskin_real-time_and_generalizable_physics-based_animation_via_self-supervised.md)**

:   提出 PhysSkin，一个泛化的物理信息框架——通过神经蒙皮场自编码器从静态 3D 几何体直接学习连续蒙皮权重场，配合物理信息自监督学习策略（能量最小化+平滑性+正交性约束），实现跨形状、跨离散化的实时物理动画，无需任何标注数据或仿真轨迹。

---

## 👥 社会计算 { #social_computing }

**[As Language Models Scale, Low-order Linear Depth Dynamics Emerge](social_computing/as_language_models_scale_low-order_linear_depth_dynamics_emerge.md)**

:   这篇论文把 Transformer 的层深看成离散时间系统，证明在给定上下文附近可以用一个 32 维的低阶线性状态空间代理去近似 GPT-2 的层间传播与干预响应，而且模型越大，这个低阶代理越准确，还能据此算出比启发式注入更省能量的多层干预策略。

**[Bridging Pixels and Words: Mask-Aware Local Semantic Fusion for Multimodal Media Verification](social_computing/bridging_pixels_and_words_mask-aware_local_semantic_fusion_for_multimodal_media_.md)**

:   提出 MaLSF 框架，利用掩码-标签对作为语义锚点，通过双向跨模态验证（BCV）和层级语义聚合（HSA）模块实现主动式局部语义冲突检测，在 DGM4 和假新闻检测任务上取得 SOTA。

**[Learning from Synthetic Data via Provenance-Based Input Gradient Guidance](social_computing/learning_from_synthetic_data_via_provenance-based_input_gradient_guidance.md)**

:   本文提出利用合成数据生成过程中自动获得的"出处信息"（provenance）作为辅助监督信号，通过输入梯度引导（抑制非目标区域的输入梯度）直接促进模型学习聚焦于目标区域的判别性表示，在弱监督定位、时空动作检测和图像分类等多任务多模态上验证了有效性。

**[Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning](social_computing/revisiting_unknowns_towards_effective_and_efficient_open-set_active_learning.md)**

:   提出 E2OAL，一个无需额外检测器的开放集主动学习框架，通过标签引导聚类发现未知类潜在结构、Dirichlet 校准辅助头联合建模已知/未知类别，并设计两阶段自适应查询策略，在多个基准上同时实现高准确率、高查询纯度和高训练效率。

---

## 🔗 因果推理 { #causal_inference }

**[Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression](causal_inference/fighting_hallucinations_with_counterfactuals_diffusion-guided_perturbations_for_.md)**

:   提出 CIPHER，一种无需训练的测试时幻觉抑制方法：离线阶段用扩散模型生成反事实图像构建 OHC-25K 数据集，通过 SVD 提取视觉幻觉子空间；推理阶段将隐状态投影到该子空间的正交补空间，在不修改模型参数、不增加推理开销的前提下显著降低 LVLM 的视觉幻觉。

**[MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations](causal_inference/maskdime_adaptive_masked_diffusion_for_precise_and_efficient_visual_counterfactu.md)**

:   提出 MaskDiME，一个免训练的扩散框架，通过自适应双掩码机制将全局分类器引导转化为决策驱动的局部编辑，实现精确高效的视觉反事实解释，推理速度比 DiME 快 30 倍以上，GPU 内存仅为 ACE/RCSB 的十分之一。

**[Retrieving Counterfactuals Improves Visual In-Context Learning](causal_inference/retrieving_counterfactuals_improves_visual_in-context_learning.md)**

:   提出 CIRCLES 框架，通过属性引导的 composed image retrieval 检索反事实示例，构建因果+相关性双通道 in-context demonstration，显著提升 VLM 的细粒度视觉推理能力。

---

## 💻 代码智能 { #code_intelligence }

**[GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning](code_intelligence/geotikzbridge_advancing_multimodal_code_generation_for_geometric_perception_and_.md)**

:   GeoTikzBridge 通过构建最大的 2.5M 图像-TikZ 代码数据集和首个辅助线指令数据集，训练出能精准重建几何图形的代码生成模型，并可作为即插即用模块增强任意 MLLM/LLM 的几何推理能力。

**[MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](code_intelligence/mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)**

:   提出 MM-ReCoder，首个具备自我纠错能力的图表转代码多模态 LLM，通过两阶段多轮 GRPO 强化学习（先共享首轮优化纠错能力，再全轨迹优化编码能力），在 ChartMimic 上以仅 7B 参数达到 86.5% low-level score，媲美 Qwen3-VL-235B。

---

## ✏️ 知识编辑 { #knowledge_editing }

**[Attribution-Guided Model Rectification of Unreliable Neural Network Behaviors](knowledge_editing/attribution-guided_model_rectification_of_unreliable_neural_network_behaviors.md)**

:   提出归因引导的动态模型纠正框架，将rank-one model editing从领域适配重定位为行为纠正，通过Integrated Gradients量化各层可编辑性自动定位嫌疑层，仅需1个清洁样本即可修复后门攻击、虚假相关和特征泄漏三类不可靠行为。

**[MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](knowledge_editing/mokus_leveraging_crossmodal_knowledge_transfer_for.md)**

:   提出"知识感知概念定制"新任务，发现LLM文本编码器中的知识编辑可以自然迁移到视觉生成模态（跨模态知识迁移），基于此提出MoKus框架：先用LoRA微调将稀有token绑定为视觉概念的锚表征，再通过知识编辑技术将多条自然语言知识高效映射到锚表征上，每条知识更新仅需约7秒。

---

## 🌐 多语言/翻译 { #multilingual_mt }

**[MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation](multilingual_mt/mmtit-bench_a_multilingual_and_multi-scenario_benchmark_with_cognition-perceptio.md)**

:   构建了覆盖 14 种非英非中语言的多语言多场景文字图像翻译基准 MMTIT-Bench，并提出 CPR-Trans 数据范式（认知→感知→翻译推理），在 3B 和 7B 模型上显著提升端到端翻译质量，7B 模型达到与 235B 模型竞争的性能。

**[SEA-Vision: A Multilingual Benchmark for Document and Scene Text Understanding in Southeast Asia](multilingual_mt/sea-vision_a_multilingual_benchmark_for_comprehensive_document_and_scene_text_un.md)**

:   推出 SEA-Vision 基准，统一评估 11 种东南亚语言的文档解析（15,234 页）与文本中心 VQA（7,496 QA 对），通过重渲染策略消除多语言 VQA 的视觉-文本错位，揭示 MLLM 在低资源东南亚语言上存在 3–7 倍的严重性能退化。

---

## 🔎 AIGC 检测 { #aigc_detection }

**[Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks](aigc_detection/fine-grained_image_aesthetic_assessment_learning_discriminative_scores_from_rela.md)**

:   定义"细粒度图像美学评估"新任务，构建含32,217张图像/10,028个系列的FGAesthetics基准，提出FGAesQ模型：通过差异保留Tokenization（DiffToken）+ 对比文本辅助对齐（CTAlign）+ 排序感知回归（RankReg）从相对排序中学习判别性审美评分，在细粒度场景准确率0.779的同时保持粗粒度SRCC 0.770。

---

## 🗣️ 对话系统 { #dialogue }

**[Evolutionary Multimodal Reasoning via Hierarchical Semantic Representation for Intent Recognition](dialogue/evolutionary_multimodal_reasoning_via_hierarchical_semantic_representation_for_i.md)**

:   提出 HIER，通过层次语义表示（token→概念→关系三级）结合基于 MLLM 反馈的自进化推理机制，在三个多模态意图识别 benchmark 上一致超越 SOTA 方法和领先 MLLM（1-3% 增益）。

---

## ⚛️ 物理学 { #physics }

**[QKD: Quantum-Gated Task-interaction Knowledge Distillation for Class-Incremental Learning](physics/qkd_quantum_gated_incremental_learning.md)**

:   QKD 将量子门控引入类增量学习，通过参数化量子电路在高维 Hilbert 空间中建模样本-任务相关性，引导跨任务知识蒸馏和推理时适配器融合，在 5 个基准上达到 SOTA。

---

## 📂 其他 { #others }

**[AdaSFormer: Adaptive Serialized Transformers for Monocular Semantic Scene Completion from Indoor Environments](others/adasformer_adaptive_serialized_transformers_for_monocular_semantic_scene_complet.md)**

:   提出AdaSFormer，一种针对室内单目语义场景补全(MSSC)的序列化Transformer框架，通过自适应序列化注意力(可学习偏移量)、中心相对位置编码和卷积调制层归一化三个核心设计，在NYUv2和Occ-ScanNet上达到SOTA。

**[AssistMimic: Physics-Grounded Humanoid Assistance via Multi-Agent RL](others/assistmimic_physics_grounded_humanoid_assistance.md)**

:   首个在物理仿真中实现接触式人-人辅助行为模仿学习的多智能体RL框架，通过运动先验初始化、动态参考重定向和接触促进奖励使MARL在高接触设置中可行。

**[BenDFM: A Taxonomy and Synthetic CAD Dataset for Manufacturability Assessment in Sheet Metal Bending](others/bendfm_a_taxonomy_and_synthetic_cad_dataset_for_ma.md)**

:   提出可制造性指标的二维分类法（配置依赖性 x 可行性/复杂度）和首个钣金弯曲合成 CAD 数据集 BenDFM（20,000 零件，含可制造和不可制造设计），基准测试显示拓扑感知的图表示（UV-Net, AUC 0.896）在四类任务上全面优于点云方法（PointNext, AUC 0.844）。

**[Bounds on Agreement between Subjective and Objective Measurements](others/bounds_on_agreement_between_subjective_and_objecti.md)**

:   从MOS的数学性质出发推导出主观测试结果与任何客观估计器之间PCC上界和MSE下界的理论公式，并提出BinoVotes/BinoMOS投票模型，在18项主观测试数据上验证了界的有效性和模型的准确性。

**[U-F²-CBM: CLIP-Free, Label Free, Unsupervised Concept Bottleneck Models](others/clipfree_label_free_unsupervised_concept_bottlenec.md)**

:   提出TextUnlock方法，通过训练轻量MLP将任意冻结视觉分类器的特征投射到文本嵌入空间（同时保持原分类器分布不变），无需CLIP、无需标注、无需训练线性探针，即可将任何legacy分类器转化为可解释的概念瓶颈模型——在40+架构上测试，超越甚至有监督的CLIP基CBM。

**[Coded-E2LF: Coded Aperture Light Field Imaging from Events](others/coded-e2lf_coded_aperture_light_field_imaging_from_events.md)**

:   首次证明仅用 event camera（无需传统 intensity 图像）即可重建像素级精度的 4D 光场，提出 Coded-E2LF 系统：通过编码光圈序列触发 events 并累积为 event images，利用全黑 pattern 建立 event-based 与 intensity-based coded aperture imaging 的数学等价性，结合端到端 deep optics 训练实现 8×8 视点光场重建。

**[Crowdsourcing of Real-world Image Annotation via Visual Properties](others/crowdsourcing_of_real_world_image_annotation_via_visual_properties.md)**

:   提出一种基于视觉属性约束的图像标注方法论，通过知识表示构建对象类别层次结构并结合交互式众包框架，利用视觉属和视觉差引导标注过程，减少标注者主观性和语义鸿沟问题。

**[Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis](others/deconstructing_the_failure_of_ideal_noise_correction_a_three-pillar_diagnosis.md)**

:   本文通过受控实验证明，即使给定完美的噪声转移矩阵 T，前向校正方法仍会在训练后期发生性能崩溃，并从宏观收敛状态、微观优化动力学、信息论三个层面系统诊断了这一失败的根本原因。

**[DiffBMP: Differentiable Rendering with Bitmap Primitives](others/diffbmp_differentiable_rendering_with_bitmap_primitives.md)**

:   提出 DiffBMP——首个面向**位图图元**的通用可微渲染引擎，通过自定义 CUDA 并行管线实现对数千张位图图元的位置、旋转、缩放、颜色和透明度的高效梯度优化，填补了 2D 可微渲染仅限矢量图形的空白。

**[DirPA: Addressing Prior Shift in Imbalanced Few-shot Crop-type Classification](others/dirpa_addressing_prior_shift_in_imbalanced_fewshot.md)**

:   提出 Dirichlet 先验增强（DirPA），在少样本学习训练阶段从 Dirichlet 分布中采样类别比例向量来构造不平衡 episode，主动模拟真实世界长尾分布以消除先验偏移，在欧盟多个国家的作物分类任务中展示了一致的鲁棒性提升和稀有类别精度改善。

**[Do Vision Models Perceive Illusory Motion in Static Images Like Humans?](others/do_vision_models_perceive_illusory_motion_in_static_images_like_humans.md)**

:   本文系统评估了多种光流模型在旋转蛇等静态图像运动错觉上的表现，发现仅人类启发的Dual-Channel模型在模拟眼跳条件下能再现人类感知的旋转运动。

**[Dual-Band Thermal Videography: Separating Time-Varying Reflection and Emission Near Ambient Conditions](others/dual_band_thermal_videography_separating_time-varying_reflection_and_emission_ne.md)**

:   提出一种双波段长波红外视频分析框架，利用光谱线索（双波段发射率比恒定）和时间线索（物体辐射平滑变化、背景辐射突变）联合约束，首次实现近环境温度条件下动态场景中反射与发射分量的逐像素分离，并恢复物体发射率和温度场。

**[ELogitNorm: Enhancing OOD Detection with Extended Logit Normalization](others/enhancing_outofdistribution_detection_with_extende.md)**

:   诊断LogitNorm的特征坍缩问题(维度坍缩+原点坍缩)，提出ELogitNorm——用到决策边界的平均距离(而非特征范数)做自适应温度缩放，无超参数、兼容所有post-hoc OOD检测方法——CIFAR-10上far-OOD AUROC提升10.48%(SCALE)，ImageNet-1K上FPR95从51.45%降至27.74%，同时改善分类精度和ECE校准。

**[FEAT: Federated Geometry-Aware Correction for Exemplar Replay under Continual Dynamic Heterogeneity](others/feat_federated_geometry_aware_correction_for_exemplar_replay_under_continual_dynamic_heterogeneity.md)**

:   提出 FEAT 方法解决联邦持续学习中回放样本利用不足的问题，通过几何结构对齐（基于 ETF 原型的角度蒸馏）和能量基几何校正（推理时去偏）缓解跨客户端异构和任务级数据不平衡。

**[GardenDesigner: Encoding Aesthetic Principles into Jiangnan Garden Construction via a Chain of Agents](others/gardendesigner_encoding_aesthetic_principles_into_jiangnan_garden_construction_v.md)**

:   提出 GardenDesigner 框架，通过链式智能体（地形分布→道路生成→资产选择→布局优化）将江南园林的美学原则编码为可计算的约束，结合专家标注的 GardenVerse 数据集，实现非专业用户通过文本输入在一分钟内自动构建符合美学规范的江南园林。

**[GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion](others/gazeonce360_fisheye-based_360_multi-person_gaze_estimation_with_global-local_fea.md)**

:   本文提出 GazeOnce360，一个端到端的双分辨率 CNN 模型，用于从单个朝上放置的桌面鱼眼相机进行 360° 多人视线方向估计，同时构建了首个面向该场景的大规模合成数据集 MPSGaze360，在精度和速度两方面均大幅超越现有多阶段方法 GAM360。

**[HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition](others/hypevpr_exploring_hyperbolic_space_for_perspective_to_equirectangular_visual_pla.md)**

:   本文提出 HypeVPR，一个基于双曲空间层次化嵌入的视觉位置识别框架，专门解决透视图像（查询）与全景图像（数据库）之间的跨视场匹配问题，通过在 Poincaré 球中从局部到全局构建多级描述子，实现精度-效率-存储的灵活平衡，检索速度比滑窗基线快数倍且精度相当。

**[Integration of Deep Generative Anomaly Detection Algorithm in High-Speed Industrial Line](others/integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)**

:   本文提出一个基于 GAN + 残差自编码器（DRAE）的半监督异常检测框架，专门设计用于制药行业 Blow-Fill-Seal（BFS）产线的高速在线质量检测，仅用合格品训练即可实现 96.4% 的准确率，单 patch 推理仅 0.17ms，满足 500ms 检测周期的严格工业约束。

**[IrisFP: Adversarial-Example-based Model Fingerprinting with Enhanced Uniqueness and Robustness](others/irisfp_adversarial-example-based_model_fingerprinting_with_enhanced_uniqueness_a.md)**

:   提出IrisFP模型指纹框架，通过将指纹放置在多类决策边界交叉点处、构建复合样本指纹、以及基于统计可分性的指纹筛选三项创新，同时增强指纹的唯一性和鲁棒性，在5个数据集上AUC一致超过SOTA方法。

**[LoViF 2026 Challenge on Human-oriented Semantic Image Quality Assessment](others/lovif_2026_semantic_quality_assessment_challenge.md)**

:   LoViF 2026首届人类导向语义图像质量评估挑战赛：提出SeIQA基准数据集（510/80/160训练/验证/测试对），衡量图像退化是否改变了人类关注的语义信息而非传统感知保真度；冠军RedpanQA Alliance基于Qwen3-VL多模态大模型+LoRA微调+PLCC损失达到0.8724最终得分。

**[Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning](others/mitigating_instance_entanglement_in_instance-dependent_partial_label_learning.md)**

:   针对实例依赖偏标签学习 (ID-PLL) 中相似类别实例因特征和候选标签重叠导致的"实例纠缠"问题，提出 CAD 框架，通过类别特定增强的类内对齐和加权惩罚损失的类间分离，双管齐下缓解类混淆。

**[MyoVision: A Mobile Research Tool and NEATBoost-Attention Ensemble Framework for Real Time Chicken Breast Myopathy Detection](others/myovision_a_mobile_research_tool_and_neatboost_attention_ensemble_framework.md)**

:   提出 MyoVision 智能手机透射成像框架和 NEATBoost-Attention 神经进化优化集成模型，用于低成本实时鸡胸肉肌病（木质胸、意面肉）三分类检测。

**[NaiLIA: Multimodal Nail Design Retrieval Based on Dense Intent Descriptions and Palette Queries](others/nailia_multimodal_nail_design_retrieval_based_on_dense_intent_descriptions_and_p.md)**

:   提出 NaiLIA，一种面向美甲设计图像的多模态检索方法，通过密集意图描述和调色板查询实现细粒度匹配，引入基于置信度分数的松弛对比损失（CRC loss）处理未标注正样本问题，在自建 NAIL-STAR 基准和 Marqo Fashion200K 上大幅超越现有方法。

**[Neural Collapse in Test-Time Adaptation](others/neural_collapse_in_test-time_adaptation.md)**

:   将神经坍缩 (Neural Collapse) 理论从类级别扩展到样本级别，发现了NC3+现象（样本特征嵌入与对应分类器权重对齐），基于此揭示了分布偏移下性能退化的根本原因是样本级特征-分类器错位，并提出NCTTA方法通过几何邻近度与预测置信度的混合目标引导特征重新对齐，在ImageNet-C上比Tent提升14.52%。

**[Next-Scale Autoregressive Models for Text-to-Motion Generation](others/next-scale_autoregressive_models_for_text-to-motion_generation.md)**

:   MoScale 提出了一种 next-scale 自回归动作生成框架，替代传统 next-token 预测，通过从粗到细的层次化因果生成来捕获全局语义结构，并引入跨尺度层次精化和尺度内时间精化，在 HumanML3D 和 KIT-ML 上达到 SOTA（Top-1 0.540，FID 0.046）。

**[Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples](others/novel_anomaly_detection_scenarios_and_evaluation_metrics_to_address_the_ambiguit.md)**

:   针对工业异常检测中"正常"定义随规格变更而变化的实际问题，提出了两种新场景（A2N/N2A）、一个新评价指标（S-AUROC）和一种训练增强方法 RePaste，通过将高异常分数区域重新粘贴到训练图片中来增加其出现频率，使模型灵活适应正常样本定义的变化。

**[OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion](others/omnifood8k_nutrition_estimation.md)**

:   构建了涵盖 8036 个样本的中式食物多模态营养数据集 OmniFood8K 和 115K 合成数据集 NutritionSynth-115K，并提出端到端框架通过 Scale-Shift 深度适配器、频域对齐融合和掩码预测头从单张 RGB 图像预测营养信息。

**[Order Matters: 3D Shape Generation from Sequential VR Sketches](others/order_matters_3d_shape_generation_from_sequential_vr_sketches.md)**

:   提出 VRSketch2Shape 框架，首次建模 VR 草图的笔画时序信息，通过序列感知的 BERT 编码器与基于扩散的 3D 生成器（SDFusion），从有序 VR 草图生成高保真 3D 形状，同时贡献了包含 20k 合成 + 900 真实草图的多类别数据集。

**[POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](others/polishing_the_sky_widefield_and_highdynamic_range.md)**

:   POLISH++在POLISH框架基础上引入分块训练+拼接策略和arcsinh非线性变换，解决了射电干涉成像中宽视场（万级像素）和高动态范围（$10^4$-$10^6$）两大实际部署难题，在T-RECS仿真数据上大幅超越CLEAN方法的源探测精度，且能超分辨恢复PSF尺度附近的强引力透镜系统，有望将DSA巡天的透镜发现数量提升约10倍。

**[Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model](others/rethinking_snn_online_training_and_deployment_grad.md)**

:   提出 HD-LIF（混合驱动 LIF）脉冲神经元模型族，通过在阈值上下区域采用不同脉冲计算机制，理论证明梯度可分离性和对齐性，解决 SNN 在线训练的前后向传播不一致问题，同时实现学习精度、内存复杂度和功耗的全阶段优化——以 10× 参数压缩、11× 功耗降低和 30% NOPs 节省达到 CIFAR-100 上 78.61% 精度。

**[Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](others/rooftop_wind_field_reconstruction_using_sparse_sen.md)**

:   建立基于PIV风洞实验数据的学习-观测框架，系统比较Kriging插值与三种深度学习模型（UNet/ViTAE/CWGAN）在5–30个稀疏传感器下的屋顶风场重建能力，揭示混合风向训练（MDT）下深度学习一致优于Kriging（SSIM提升18–34%），并通过QR分解优化传感器布局提升系统鲁棒性达27.8%。

**[Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation](others/shoe_style-invariant_and_ground-aware_learning_for_dense_foot_contact_estimation.md)**

:   提出 FECO 框架，通过鞋款风格–内容随机化（对抗训练）和地面感知学习（像素高度图 + 地面法线），从单张 RGB 图像实现鲁棒的密集足部接触估计，在多个基准上显著超越现有方法。

**[SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](others/shrec_a_spectral_embedding-based_approach_for_ab-initio_reconstruction_of_helica.md)**

:   提出 SHREC 算法，通过谱嵌入（spectral embedding）从冷冻电镜 2D 投影图像中直接恢复螺旋分子片段的投影角度，无需预先知道螺旋对称参数（rise/twist），实现了真正的 ab-initio 螺旋结构重建。

**[SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](others/simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)**

:   提出 SimRecon 框架，通过"感知→生成→仿真"三阶段流水线，从真实视频自动构建仿真就绪的组合式 3D 场景，核心创新在于主动视角优化（AVO）为单物体生成寻找最优投影视角和场景图合成器（SGS）引导物理可信的层级化组装。

**[SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](others/sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)**

:   构建了包含 242,000+ 工业零件的大规模多模态 CAD 数据集 SldprtNet，每个样本包含 .sldprt/.step 3D 模型、七视图合成图像、参数化建模脚本和自然语言描述四种模态的完整对齐数据，配套开发支持 13 种 CAD 命令的无损编码器/解码器工具，baseline 实验验证了多模态输入相比纯文本输入在 CAD 生成任务上的显著优势。

**[Stronger Normalization-Free Transformers](others/stronger_normalization-free_transformers.md)**

:   通过系统分析逐点函数替代归一化层所需的四个关键属性（零中心性、有界性、中心敏感性、单调性），在大规模搜索中发现 $\text{Derf}(x) = \text{erf}(\alpha x + s)$ 是最优的归一化层替代函数，在视觉识别、图像生成、语音表示和DNA序列建模等多个领域持续超越LayerNorm和DyT，且性能增益主要来自更强的泛化而非拟合能力。

**[TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size](others/teamhoi_learning_a_unified_policy_for_cooperative_human-object_interactions_with.md)**

:   提出 TeamHOI 框架，通过基于 Transformer 的去中心化策略网络和掩码对抗运动先验（Masked AMP），使单一策略能够泛化到任意数量智能体的协作搬运任务，2-8 个仿人智能体协作搬桌子成功率达 97%+。

**[UniSpector: Towards Universal Open-set Defect Recognition via Spectral-Contrastive Visual Prompting](others/unispector_towards_universal_open-set_defect_recognition_via_spectral-contrastiv.md)**

:   本文提出 UniSpector 开放集工业缺陷检测框架，通过频域-空域双域特征融合（SSPE）和角度间隔对比提示编码（CPE）解决了视觉提示嵌入崩塌问题，在新构建的包含 360 种缺陷类别的 Inspect Anything benchmark 上，AP50 检测和分割分别比最佳基线高 19.7% 和 15.8%。

**[V-Nutri: Dish-Level Nutrition Estimation from Egocentric Cooking Videos](others/v_nutri_nutrition_estimation_cooking_videos.md)**

:   提出 V-Nutri 框架，首次利用第一人称烹饪视频中的过程信息来辅助菜品营养估计，通过 VideoMamba 关键帧选择模块提取食材添加时刻，与最终成品图像融合来预测热量和宏量营养素。

**[ViT3: Unlocking Test-Time Training in Vision](others/vit3_unlocking_test_time_training_in_vision.md)**

:   系统性探索Test-Time Training（TTT）在视觉任务中的设计空间，总结六条实用设计洞察，提出ViT3——一个线性复杂度的纯TTT视觉架构，在分类/生成/检测/分割任务中匹配或超越Mamba和线性注意力方法。

**[What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁](others/what_is_the_optimal_ranking_score_between_precision_and_recall_we_can_always_fin.md)**

:   本文从排名理论角度系统研究了 $F_\beta$ 分数族作为 Precision 与 Recall 排名折中的性质，证明 $F_\beta$ 诱导的排名构成 Precision 和 Recall 排名之间的测地线（最短路径），进而提出闭式公式来找到最优的 $\beta$ 值，并证明常用的 $F_1$ 和 skew-insensitive $F_1$ 在大多数情况下都不是最优排名折中。

**[What Is Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution](others/what_is_wrong_with_synthetic_data_for_scene_text_recognition_a_strong_synthetic_.md)**

:   系统分析了现有渲染合成数据在语料、字体、布局多样性上的不足，提出 UnionST 合成引擎和自演化学习框架（SEL），仅用合成数据即大幅超越传统合成集，结合 SEL 仅需 9% 真实标注即可逼近全监督性能。

**[Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation](others/your_classifier_can_do_more_towards_balancing_the.md)**

:   通过能量景观分析揭示 AT 和 JEM 的互补性（AT 对齐 clean-adv 能量分布 → 鲁棒性；JEM 对齐 clean-generated 能量分布 → 精度+生成），提出 EB-JDAT 建模联合分布 $p(\mathbf{x}, \tilde{\mathbf{x}}, y)$ 并用 min-max 能量优化对齐三种数据能量分布，CIFAR-10 AutoAttack 鲁棒性 68.76%（超 SOTA AT +10.78%），同时保持 90.39% 清洁精度和 FID=27.42 的竞争力生成质量。

**[ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)**

:   提出 ZO-SAM，在 SAM 的扰动步骤中用零阶梯度估计替代反向传播，将 SAM 的计算开销从 2 次反传减少为 1 次，首次让 SAM 在稀疏训练中变得实用，在 CIFAR-10/100 和 ImageNet-1K 上一致提升所有主流稀疏训练方法 0.38%-2.54%。

</div>