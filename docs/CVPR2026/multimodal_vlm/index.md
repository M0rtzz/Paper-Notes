---
title: >-
  CVPR2026 多模态 VLM方向217篇论文解读
description: >-
  217篇CVPR2026的多模态 VLM 方向论文解读，涵盖多模态、推理、LLM、对齐/RLHF、对抗鲁棒、模型压缩等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2026"
  - "多模态 VLM"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "推理"
  - "LLM"
  - "对齐/RLHF"
  - "对抗鲁棒"
  - "模型压缩"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧩 多模态 VLM

**📷 CVPR2026** · **217** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (106)](../../ICML2026/multimodal_vlm/index.md) · [💬 ACL2026 (99)](../../ACL2026/multimodal_vlm/index.md) · [🔬 ICLR2026 (88)](../../ICLR2026/multimodal_vlm/index.md) · [🤖 AAAI2026 (88)](../../AAAI2026/multimodal_vlm/index.md) · [🧠 NeurIPS2025 (148)](../../NeurIPS2025/multimodal_vlm/index.md) · [📹 ICCV2025 (149)](../../ICCV2025/multimodal_vlm/index.md)

🔥 **高频主题：** 多模态 ×119 · 推理 ×23 · LLM ×21 · 对齐/RLHF ×13 · 对抗鲁棒 ×10

**[A3: Towards Advertising Aesthetic Assessment](a3_towards_advertising_aesthetic_assessment.md)**

:   提出A3框架，包含理论驱动的三阶段广告美学评估范式A3-Law（感知注意力→形式兴趣→欲望影响）、12万条标注数据集A3-Dataset、经SFT+GRPO对齐的模型A3-Align以及评测基准A3-Bench，在广告美学自动评估上超越现有MLLM。

**[Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models](activation_matters_test-time_activated_negative_labels_for_ood_detection_with_vi.md)**

:   提出 TANL（Test-time Activated Negative Labels），通过在测试时动态评估负标签在OOD样本上的"激活程度"来挖掘最有效的负标签，配合激活感知评分函数，在 ImageNet 基准上将 FPR95 从 17.5% 大幅降至 9.8%，且完全免训练、测试高效。

**[AVR: Adaptive VLM Routing for Computer Use Agents](adaptive_vision-language_model_routing_for_computer_use_agents.md)**

:   提出 AVR 自适应路由框架，通过轻量多模态嵌入模型评估动作难度 + 小模型 logprob 置信度探测 + warm agent 记忆注入，实现三层路由（简单→小模型，困难→大模型，高风险→大模型+guardrail），在推理成本降低 78% 的同时仅损失 2pp 准确率。

**[AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition](adaptvision_efficient_vision-language_models_via_adaptive_visual_acquisition.md)**

:   提出 AdaptVision，通过由粗到精的主动视觉机制和强化学习训练，让 VLM 自主决定每个样本所需的最少视觉 token 数量，配合解耦式多轮策略优化 (DTPO) 实现效率与精度的最优平衡。

**[AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)**

:   AGFT 提出了一种对齐引导的微调框架，通过文本引导的对抗训练和分布一致性校准，在增强 VLM 零样本对抗鲁棒性的同时保持预训练的跨模态语义结构，在 15 个零样本基准上平均鲁棒准确率达到 46.57%，超越 SOTA 3.1 个百分点。

**[Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](aif_adaptive_information_flow_vlm.md)**

:   本文发现 VLM 中文本 token 对无关视觉 token 的过度注意力是"看到但感知错误"的根本原因，提出基于 token 动态熵的自适应信息流调控方法（AIF），通过推理时修改因果掩码来阻断无关视觉-文本连接，免训练提升多种 VLM 的感知能力。

**[ApET: Approximation-Error Guided Token Compression for Efficient VLMs](apet_approximation-error_guided_token_compression_for_efficient_vlms.md)**

:   从信息论角度提出基于线性近似重建误差的视觉 token 重要性评估方法，不依赖 attention 权重，天然兼容 FlashAttention，在 LLaVA-1.5 上压缩 88.9% 视觉 token 仍保持 95.2% 性能。

**[Asking like Socrates: Socrates helps VLMs understand remote sensing images](asking_like_socrates_socrates_helps_vlms_understand_remote_sensing_images.md)**

:   揭示遥感VLM中的"伪推理"现象（显式推理链反而导致性能下降），归因于"一瞥效应"（单次粗浅感知不足），提出RS-EoT(Evidence-of-Thought)迭代证据搜索范式，通过SocraticAgent自博弈合成推理轨迹做SFT冷启动，再用两阶段渐进RL（grounding→VQA）增强和泛化，RS-EoT-7B在多个遥感VQA和grounding基准上达SOTA。

**[AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](ava-bench_atomic_visual_ability_benchmark_for_vision_foundation_models.md)**

:   提出 AVA-Bench，首个将视觉基础模型（VFM）的能力解耦为 14 种原子视觉能力（AVA）的系统性评测基准，通过训练-测试分布对齐和单一能力隔离测试，精准定位 VFM 的强项与短板，并发现 0.5B 小模型即可保持与 7B 模型相当的 VFM 排名一致性。

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

**[BiCLIP: Domain Canonicalization via Structured Geometric Transformation](biclip_domain_canonicalization_via_structured_geometric_transformation.md)**

:   提出 BiCLIP，一个极简的 CLIP 少样本适配方法，通过一个上三角结构约束的双线性变换矩阵对图像特征进行几何对齐，在 11 个标准基准上以极低参数量达到 SOTA。

**[BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment](brima_bridged_modality_adaptation_for_multi-modal_continual_action_quality_asses.md)**

:   提出 BriMA，通过记忆引导的桥接补全和模态感知回放机制，解决多模态持续动作质量评估中非平稳模态不平衡问题，在三个基准上平均提升 6-8% 相关系数、降低 12-15% 误差。

**[Can Vision-Language Models Count? A Synthetic Benchmark and Analysis of Attention-Based Interventions](can_vision-language_models_count_a_synthetic_benchmark_and_analysis_of_attention.md)**

:   构建了一个合成计数基准数据集，系统评估了开源 VLM 在不同图像/提示条件下的计数能力，并通过解码器层面的视觉注意力重加权实验探索改善计数行为的机制。

**[CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment](capt_confusion-aware_prompt_tuning_for_reducing_vision-language_misalignment.md)**

:   提出 CAPT 混淆感知 prompt tuning 框架，通过语义混淆挖掘器（SEM）和样本混淆挖掘器（SAM）显式建模 VLM 的系统性误对齐模式，配合多粒度差异专家（MGDE）融合不同层次的混淆信息，在 11 个基准上取得 HM 83.90% 的最优表现。

**[Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking](circuit_tracing_in_vision-language_models_understanding_the_internal_mechanisms_.md)**

:   提出首个面向 VLM 的电路追踪框架，在 Gemma-3-4B 中训练 per-layer transcoder 并构建归因图，揭示了多模态推理的层次化整合机制、视觉数学电路和六指幻觉的内部成因，并通过 steering 和 circuit patching 验证电路的因果可控性。

**[CLIP-Free, Label-Free, Unsupervised Concept Bottleneck Models](clip-free_label_free_unsupervised_concept_bottleneck_models.md)**

:   提出 TextUnlock 方法将任意冻结视觉分类器的输出分布对齐到视觉-语言对应空间，进而构建无需CLIP、无需标签、无需训练线性探针的全无监督概念瓶颈模型 (U-F²-CBM)，在40+模型上超越有监督CLIP-based CBM。

**[Concept-wise Attention for Fine-grained Concept Bottleneck Models](coat_cbm_concept_wise_attention.md)**

:   CoAt-CBM 通过可学习的概念级视觉 query 和概念对比优化（CCO）实现了自适应细粒度图像-概念对齐，在保持高可解释性的同时超越现有概念瓶颈模型和黑盒模型。

**[CodeDance: A Dynamic Tool-integrated MLLM for Executable Visual Reasoning](codedance_a_dynamic_tool-integrated_mllm_for_executable_visual_reasoning.md)**

:   提出 CodeDance，将可执行代码作为视觉推理的统一媒介，通过 SFT 教授原子能力 + RL 中的难度自适应工具调用奖励（BAT），实现动态工具编排与自检推理，7B 模型在计数/视觉搜索/图表 QA 等任务上超越 GPT-4o。

**[CodePercept: Code-Grounded Visual STEM Perception for MLLMs](codepercept_code-grounded_visual_stem_perception_for_mllms.md)**

:   通过系统性缩放分析发现感知（perception）而非推理（reasoning）是 MLLM 在 STEM 领域的真正瓶颈，提出以可执行 Python 代码为锚定媒介的 CodePercept 范式——构建 100 万级 ICC-1M 数据集和 STEM2Code-Eval 基准，在 SFT+RL 两阶段训练后显著提升 MLLM 的 STEM 视觉感知和下游推理能力。

**[CoMP: Collaborative Multi-Mode Pruning for Vision-Language Models](comp_collaborative_multi-mode_pruning_for_vision-language_models.md)**

:   CoMP 提出协同多模式剪枝框架，通过协同重要性度量（CIM）消除参数和 token 剪枝指标间的不一致性，通过多模式剪枝策略（MPS）自适应选择每阶段的最优剪枝模式，在高剪枝比例下显著优于单模式和简单联合剪枝方案。

**[Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)**

:   提出 SeGP-CL，通过对抗锚点探测旧-新语义边界的脆弱区域，结合锚点引导的跨模态几何蒸馏（ACGD）和文本语义几何正则化（TSGR），在无样本回放条件下有效保持 VLM 的跨模态语义几何结构，显著缓解灾难性遗忘。

**[CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)**

:   发现 MLLM 中视觉编码器微调的"视觉偏好冲突"问题，提出 CoVFT 框架，通过上下文向量提取（CVE）和上下文混合专家（CoMoE）实现上下文感知的视觉微调，在 12 个多模态基准上达到 SOTA 且稳定性显著优于现有方法。

**[CoVR-R: Reason-Aware Composed Video Retrieval](covr-rreason-aware_composed_video_retrieval.md)**

:   CoVR-R 提出了推理优先的零样本组合视频检索框架，利用大型多模态模型（Qwen3-VL）显式推理编辑操作隐含的"后效应"（状态转换、时间阶段、镜头变化等），并构建了包含结构化推理轨迹和困难干扰项的 CoVR-R 基准来评估推理能力，在检索准确率上大幅超越现有方法。

**[CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning](crit_graph-based_automatic_data_synthesis_to_enhance_cross-modal_multi-hop_reaso.md)**

:   提出基于图结构的自动数据生成 pipeline，构建了 CRIT 数据集与 benchmark，用于训练和评测 VLM 在交错图文内容上的跨模态多跳推理能力，训练后的模型在 SPIQA 等多个基准上取得显著提升。

**[CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](cropvlm_learning_to_zoom_for_fine_grained_vision_language_perception.md)**

:   提出CropVLM——一个256M参数的轻量裁剪网络，通过GRPO强化学习训练（无需人工标注边界框），动态选择图像最有信息量的区域供VLM聚焦，可与开源和商用VLM即插即用地提升细粒度视觉理解性能。

**[CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods](crosshoi-bench_a_unified_benchmark_for_hoi_evaluation_across_vision-language_mod.md)**

:   提出 CrossHOI-Bench，首个统一评估 VLM 和 HOI 专用模型的多选题 HOI 基准，通过精心策划的正负例避免不完整标注的错误惩罚，揭示大型 VLM 零样本在 Instance-F1 上超越 SOTA HOI 方法 +5.18%，但在多动作识别和跨人归因上仍存在系统性弱点。

**[Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens](cubic_discrete_diffusion_discrete_visual_generation_on_high-dimensional_represen.md)**

:   提出 CubiD，首个在高维表征 token（768维）上做离散扩散生成的模型，通过在 $h \times w \times d$ 三维张量上进行细粒度 mask 预测实现高质量图像生成，同时保留理解能力。

**[Customized Visual Storytelling with Unified Multimodal LLMs](customized_visual_storytelling_with_unified_multimodal_llms.md)**

:   提出 VstoryGen 框架和核心组件 CustFilmer，基于统一多模态大语言模型（UMLLM）实现多模态故事定制生成，支持文本描述、角色/场景参考图像和镜头类型的联合条件控制，并构建了 MSB 和 M2SB 两个新 benchmark。

**[DeAR: Fine-Grained VLM Adaptation by Decomposing Attention Head Roles](dear_fine-grained_vlm_adaptation_by_decomposing_attention_head_roles.md)**

:   提出 DeAR，通过 Concept Entropy 指标将 ViT 深层注意力头分解为属性头/泛化头/混合头三类功能角色，并设计基于角色的注意力掩码机制精确控制信息流，在15个数据集上实现任务适配与零样本泛化的最佳平衡。

**[Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)**

:   提出 DASP，通过冗余度评分诊断偏置模态，再用非对称适应策略解耦稳定性与可塑性，解决多模态测试时适应中的负迁移和灾难性遗忘问题。

**[DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning](deepsketcher_internalizing_visual_manipulation_for_multimodal_reasoning.md)**

:   提出DeepSketcher套件——包含31k高质量代码渲染的图文交错CoT数据集和一个自包含的Embedding Editor模型，使VLM无需外部工具即可在视觉嵌入空间直接生成"视觉思考"进行多模态推理。

**[Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models](devil_is_in_narrow_policy_unleashing_exploration_in_driving_vla_models.md)**

:   揭示驾驶 VLA 模型中被忽视的"窄策略"（Narrow Policy）瓶颈——IL 阶段过度利用导致探索坍缩，进而限制 RL 阶段。提出 Curious-VLA 框架，通过可行轨迹扩展 + 多样性感知 RL 在 Navsim 上达到 SOTA（PDMS 90.3，Best-of-N 94.8）。

**[Diagnosing and Repairing Unsafe Channels in Vision-Language Models via Causal Discovery and Dual-Modal Safety Subspace Projection](diagnosing_and_repairing_unsafe_channels_in_vision-language_models_via_causal_di.md)**

:   提出 CARE 框架，先用因果中介分析精确定位 VLM 中与不安全行为因果相关的神经元和层（诊断），再通过广义特征分解构建双模态安全子空间并在推理时投影激活值（修复），将攻击成功率降至 10% 以下且几乎不损失通用能力。

**[Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs](dictionary_aligned_concept_control_for_safeguarding_multimodal_llms.md)**

:   本文提出 DACO 框架，通过从 WordNet 和 CC-3M 构建包含 15,000 个多模态概念的字典，结合稀疏自编码器（SAE）实现对冻结 MLLM 激活空间的细粒度概念控制，在多个安全基准上显著提升安全性的同时保持通用能力。

**[Disentangle-then-Align: Non-Iterative Hybrid Multimodal Image Registration via Cross-Scale Feature Disentanglement](disentangle-then-align_non-iterative_hybrid_multimodal_image_registration_via_cr.md)**

:   提出 HRNet，通过跨尺度特征解纠缠和自适应投影（CDAP）学习干净的共享表示，并在统一的粗到细管线中非迭代地联合预测刚性和非刚性变换，在四个多模态数据集上达到SOTA。

**[Do Vision Language Models Need to Process Image Tokens?](do_vision_language_models_need_to_process_image_tokens.md)**

:   本文系统揭示了VLM中图像token表征在浅层即趋于稳定且跨层可互换，而文本token持续动态重构——图像处理深度的必要性高度依赖输出任务类型。

**[DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](docseeker_long_document_understanding.md)**

:   提出 DocSeeker，通过 ALR（分析-定位-推理）视觉推理范式和两阶段训练（SFT+EviGRPO）实现长文档理解中的结构化推理和证据定位，仅在短文档上训练即可鲁棒泛化到超长文档。

**[Downscaling Intelligence: Exploring Perception and Reasoning Bottlenecks in Small VLMs](downscaling_intelligence_exploring_perception_and_reasoning_bottlenecks_in_small.md)**

:   系统研究LLM缩放对多模态能力的影响，发现视觉任务而非LLM依赖任务受影响最大，且感知退化与推理退化同等严重；提出Extract+Think方法（视觉提取调优+逐步推理），以0.6B感知+1.7B推理的极小模型超越了12倍大的PrismCaptioner和LLaVA-OneVision-0.5B。

**[DSCA: Dynamic Subspace Concept Alignment for Lifelong VLM Editing](dsca_dynamic_subspace_concept_alignment_for_lifelong_vlm_editing.md)**

:   DSCA通过将VLM的表征空间分解为一组正交语义子空间，在每个子空间内进行门控残差干预来实现知识编辑，从而在1000次连续编辑后仍保持>95%的编辑成功率且近乎零遗忘。

**[DSERT-RoLL: Robust Multi-Modal Perception for Diverse Driving Conditions](dsert_roll_robust_multi_modal_perception_for_diverse_driving_conditions.md)**

:   提出 DSERT-RoLL 驾驶数据集，首次集成立体事件相机、RGB、热成像、4D 雷达和双 LiDAR 六种传感器，覆盖多种天气和光照条件，并提出统一多模态 3D 检测融合框架。

**[DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference](duet-vlm_dual_stage_unified_efficient_token_reduction_for_vlm_training_and_infer.md)**

:   提出 DUET-VLM 双阶段视觉 token 压缩框架：第一阶段在视觉编码器内通过 V2V self-attention 选取 dominant tokens 并将剩余 tokens 通过注意力引导局部聚类合并为 contextual tokens；第二阶段在 LLM 内通过 T2V cross-attention 层级裁剪视觉 tokens。在 LLaVA-1.5-7B 上实现 67% token 压缩保持 99%+ 精度、89% 压缩保持 97%+ 精度，训练时间减少 31%。

**[Dynamic Token Reweighting for Robust Vision-Language Models](dynamic_token_reweighting_for_robust_vision-language_models.md)**

:   提出Dtr（Dynamic Token Reweighting），首个通过优化VLM的KV缓存来防御多模态越狱攻击的推理时防御方法，通过定义"反向安全偏移"（RSS）来识别导致安全退化的视觉token，动态调整其权重以恢复模型的安全对齐能力，同时保持良性任务性能。

**[DynamicGTR: Leveraging Graph Topology Representation Preferences to Boost VLM Capabilities on Graph QAs](dynamicgtr_leveraging_graph_topology_representation_preferences_to_boost_vlm_cap.md)**

:   提出 DynamicGTR 框架，通过动态路由在推理时为每个查询选择最优的图拓扑表示（GTR，视觉/文本共8种），显著提升 VLM 在零样本图算法问答中的性能，并可迁移到链接预测和节点分类等真实场景。

**[EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval](eaglenet_energy-aware_fine-grained_relationship_learning_network_for_text-video_.md)**

:   EagleNet 通过构建文本-帧关系图并使用关系图注意力网络学习文本-帧和帧-帧之间的细粒度关系，生成融合视频上下文信息的增强文本嵌入，并引入基于能量模型的匹配机制捕获真实文本-视频对分布，在四个基准数据集上取得 SOTA。

**[EBMC: Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis](ebmc_multimodal_sentiment_analysis.md)**

:   提出 EBMC 两阶段框架，先通过语义解缠和跨模态增强提升弱模态表示质量，再通过能量引导的模态协调和实例感知信任蒸馏实现平衡的多模态情感分析，在缺失模态场景下保持强鲁棒性。

**[Echoes of Ownership: Adversarial-Guided Dual Injection for Copyright Protection in MLLMs](echoes_of_ownership_adversarial-guided_dual_injection_for_copyright_protection_i.md)**

:   提出 AGDI 框架，通过对抗优化生成 trigger image 进行 MLLM 黑盒版权追踪：双注入机制同时在 response 级（CE loss 驱动辅助模型输出 target answer）和 semantic 级（最小化 trigger image 与 target text 的 CLIP 余弦距离）注入版权信息，并引入模型对抗训练模拟 fine-tune 抵抗，在 Qwen2-VL/LLaVA-1.5 上全面超越 PLA 和 RNA 基线。

**[Efficient Document Parsing via Parallel Token Prediction](efficient_document_parsing_via_parallel_token_prediction.md)**

:   提出 PTP（Parallel Token Prediction），一种模型无关的即插即用加速方法，通过在训练序列中插入可学习 register token 实现并行多 token 预测，在 OmniDocBench 上实现 1.6×-2.2× 吞吐提升且不损失精度。

**[EgoMind: Activating Spatial Cognition through Linguistic Reasoning in MLLMs](egomind_activating_spatial_cognition_through_linguistic_reasoning_in_mllms.md)**

:   提出 EgoMind，一种无需几何先验的 CoT 框架，通过角色扮演字幕 (RPC) 和渐进式空间分析 (PSA) 两个核心组件，仅用 5K SFT + 20K RL 样本即可实现多帧空间推理的竞争性能力。

**[EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning in Multimodal Large Language Models](emo-r3_reflective_reinforcement_learning_for_emotional_reasoning_in_multimodal_l.md)**

:   提出 EMO-R3，通过结构化情感思维（SET）引导 MLLM 逐步进行情感推理，并设计反思情感奖励（RER）让模型重新评估推理的视觉-文本一致性和情感连贯性，显著提升多模态情感理解的可解释性和准确性。

**[EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis](emoverse_a_mllms-driven_emotion_representation_dataset_for_interpretable_visual_.md)**

:   构建 EmoVerse——首个同时覆盖 CES（Mikels 8 类离散情感）和 DES（1024 维连续情感空间）的大规模可解释视觉情感数据集（219K+ 图像），提出 B-A-S（Background-Attribute-Subject）三元组知识图谱标注体系和 Annotation & Verification Pipeline（Gemini/GPT-4o + EmoViT + CoT Critic Agent），并基于 Qwen2.5-VL-3B 微调实现 1024 维 DES 投射与情感归因解释。

**[Empowering Semantic-Sensitive Underwater Image Enhancement with VLM](empowering_semanticsensitive_underwater_image_enha.md)**

:   提出一种 VLM 驱动的语义敏感学习策略，通过 LLaVA 生成物体描述、BLIP 构建空间语义引导图、cross-attention 与语义对齐损失双重引导 UIE decoder 重建，使增强图像在感知质量和下游检测/分割任务上同时获得显著提升。

**[ENC-Bench: A Benchmark for Evaluating MLLMs in Electronic Navigational Chart Understanding](enc-bench_a_benchmark_for_evaluating_multimodal_large_language_models_in_electro.md)**

:   提出首个面向电子航海图(ENC)理解的专业级基准 ENC-Bench，包含 20,490 样本和三级层次评估体系（感知→空间推理→海事决策），系统评估 10 个 MLLM 后发现最佳模型仅 47.88% 准确率，揭示了通用模型在安全关键专业领域的严重能力缺口。

**[EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards](evolmm_self_evolving_lmm_continuous_rewards.md)**

:   提出 EvoLMM，一个纯无监督的自进化框架：从单一LMM分出Proposer（生成图像相关问题）和Solver（回答问题），通过连续自一致性奖励（替代离散多数投票）形成闭环训练信号，仅使用原始图像（无标注、无外部奖励模型），在8个多模态数学推理基准上获得约2-3%的一致性提升。

**[Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory](evolving_contextual_safety_in_multi-modal_large_language_models_via_inference-ti.md)**

:   提出 MM-SafetyBench++ 基准和 EchoSafe 框架，通过推理时维护自反思记忆库来累积安全洞察，使 MLLM 能够根据上下文区分看起来相似但安全意图不同的场景，无需训练即可提升上下文安全性。

**[EvoPrompt: Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_vision-language_models.md)**

:   EvoPrompt 通过轨迹感知的 prompt 进化策略（统一 embedding 投影 + 方向-幅度解耦训练 + 特征几何正则化）解决 VLM prompt learning 中的灾难性遗忘和模态偏差问题，在 few-shot/跨数据集/域泛化任务上全面 SOTA 且保持 zero-shot 能力。

**[Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration](explore_with_long-term_memory_a_benchmark_and_multimodal_llm-based_reinforcement.md)**

:   本文提出 LMEE 基准和 MemoryExplorer 框架，通过将多目标导航与记忆问答统一评估具身探索的过程与结果，并用强化学习微调 MLLM 使其主动调用记忆检索工具，在 LMEE-Bench 上 SR 达 23.53%（超越 3D-Mem 的 16.91%）、GOAT-Bench 上 SR 达 46.40%。

**[FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Alignment](falcon_false-negative_aware_learning_of_contrastive_negatives_in_vision-language.md)**

:   提出 FALCON，一种**基于学习的 mini-batch 构造策略**，通过负样本挖掘调度器自适应平衡硬负样本与假负样本之间的权衡，显著提升视觉语言预训练的跨模态对齐质量。

**[Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)**

:   提出量化感知积分梯度（QIG），将 LVLM 量化的灵敏度分析从模态级推进到 token 级，利用公理化归因原理精确量化每个 token 对量化误差的贡献，在 W4A8 和 W3A16 设置下显著提升量化模型精度，且几乎无额外计算开销。

**[FINER: MLLMs Hallucinate under Fine-grained Negative Queries](finer_mllms_hallucinate_under_fine-grained_negative_queries.md)**

:   发现 MLLM 在细粒度负查询（涉及多个对象/属性/关系的查询中仅有一个细微错误）下幻觉率急剧上升，提出 FINER 基准和 FINER-Tuning 方法（基于 DPO），在 InternVL3.5-14B 上最高提升 24.2%。

**[FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](flashcache_frequency_kv_cache_compression.md)**

:   提出 FlashCache，首次从频域角度分析多模态 KV Cache 的重要性分布，发现偏离低频主成分的"离群 KV"编码了推理关键特征，通过 DCT 低通滤波识别并优先保留离群 KV + 动态逐层预算分配，在 80% KV 内存压缩下实现 1.69× 解码加速且基本不损失任务性能，天然兼容 FlashAttention。

**[FlowComposer: Composable Flows for Compositional Zero-Shot Learning](flowcomposer_composable_flows_for_compositional_zeroshot_learning.md)**

:   FlowComposer 首次将 Flow Matching 引入组合零样本学习(CZSL)，学习两个原始流(属性流和物体流)将视觉特征传输到对应文本嵌入空间，并通过可学习的 Composer 显式组合速度场得到组合流，同时利用泄露引导增强策略将不完美的特征解耦转化为辅助监督信号，作为即插即用模块在三个基准上持续提升 CZSL 性能。

**[FlowHijack: A Dynamics-Aware Backdoor Attack on Flow-Matching VLA Models](flowhijack_dynamics_aware_backdoor_attack_on_flow_matching_vla_models.md)**

:   FlowHijack是首个系统性针对流匹配VLA模型向量场动态的后门攻击框架，通过τ条件注入策略和动态模仿正则化实现高攻击成功率和行为隐蔽性。

**[FluoCLIP: Stain-Aware Focus Quality Assessment in Fluorescence Microscopy](fluoclip_stain-aware_focus_quality_assessment_in_fluorescence_microscopy.md)**

:   提出 FluoCLIP，一个两阶段视觉-语言框架：先通过染色锚定（stain-grounding）让 CLIP 学习荧光染色的语义，再通过染色引导排序（stain-guided ranking）实现染色感知的对焦质量评估，并引入首个多染色组织级荧光显微镜数据集 FluoMix。

**[PinPoint: Focus, Don't Prune — Identifying Instruction-Relevant Regions for Information-Rich Image Understanding](focus_dont_prune_identifying_instruction-relevant_regions_for_information-rich_i.md)**

:   提出 PinPoint，一个两阶段框架：先通过 Instruction-Region Alignment 定位与指令相关的图像区域，再对选中区域精细化编码，以更少的 visual token 实现更高的 VQA 精度。

**[From Masks to Pixels and Meaning: A New Taxonomy, Benchmark, and Metrics for VLM Image Tampering](from_masks_to_pixels_and_meaning_a_new_taxonomy_benchmark_and_metrics_for_vlm_im.md)**

:   本文指出现有图像篡改检测基准依赖粗糙的mask标注与真实编辑信号严重不对齐,提出 PIXAR——一个包含 420K+ 图像对的像素级、语义感知篡改检测基准,配合新的训练框架和评估指标,在精确定位和语义理解方面大幅超越现有方法。

**[From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings](from_observation_to_action_latent_action-based_primitive_segmentation_for_vla_pr.md)**

:   提出 LAPS（Latent Action-based Primitive Segmentation）流水线，通过在潜在动作空间中定义"Latent Action Energy"指标，从未标注的工业视频流中无监督发现和分割语义动作原语，为 VLA 模型预训练提供结构化数据。

**[G-MIXER: Geodesic Mixup-based Implicit Semantic Expansion and Explicit Semantic Re-ranking for Zero-Shot Composed Image Retrieval](g_mixer_geodesic_mixup_based_implicit_semantic_expansion_for_zero_shot_cir.md)**

:   提出 G-MIXER，通过测地线混合隐式语义扩展（在球面上沿不同混合比例扩展检索范围）和显式语义重排序（利用 MLLM 生成的属性过滤噪声候选），实现免训练零样本组合图像检索的 SOTA 性能。

**[Generate, Analyze, and Refine: Training-Free Sound Source Localization via MLLM Meta-Reasoning](generate_analyze_and_refine_training-free_sound_source_localization_via_mllm_met.md)**

:   本文提出了一个无需训练的声源定位框架 GAR-SSL，通过将声源定位重新建模为"生成-分析-精炼"的三阶段元认知推理过程，直接利用多模态大语言模型 (MLLM) 的内在推理能力进行音视频定位，在单源和多源定位基准上取得了与训练方法可比甚至更优的性能。

**[GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning](graphvlm_benchmark_vlm_graph_learning.md)**

:   提出 GraphVLM benchmark，系统评估VLM在多模态图学习中的三种角色——VLM-as-Encoder（增强GNN特征）、VLM-as-Aligner（桥接模态用于LLM推理）、VLM-as-Predictor（直接作为图学习backbone）。在6个数据集上的实验表明，VLM-as-Predictor持续取得最佳性能，揭示了VLM作为多模态图学习新基础的巨大潜力。

**[GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding](groundvts_visual_token_sampling_in_multimodal_large_language_models_for_video_te.md)**

:   提出 GroundVTS，一种在视频大语言模型中进行查询引导的细粒度视觉token采样架构，通过在 token 级别自适应保留与查询相关的时空信息，在 Charades-STA 上 mIoU 提升 18.4 点，QVHighlights 上 mAP 提升 20.6 点。

**[GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr_turbo_merged_checkpoint_free_teacher.md)**

:   本文提出 GTR-Turbo，通过将 RL 训练过程中的历史 checkpoint 经 TIES 合并产生"免费教师模型"来指导后续训练（可选 SFT 或 KL 蒸馏方式），在多个视觉智能体任务上匹配甚至超过依赖 GPT-4o 等外部教师的 GTR 方法，同时减少 50% 训练时间和 60% 计算成本。

**[GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks](guide_a_benchmark_for_understanding_and_assisting_users_in_open-ended_gui_tasks.md)**

:   本文提出 GUIDE 基准，包含 120 个新手用户在 10 款软件上的 67.5 小时屏幕录像和出声思维标注，定义了行为状态检测、意图预测、协助预测三个分层任务，评估发现当前最强多模态模型在理解用户行为和判断协助需求上表现有限（行为检测仅 44.6% 准确率），但提供结构化用户上下文可显著提升性能（协助预测最高提升 50.2pp）。

**[HAMMER: Harnessing MLLM via Cross-Modal Integration for Intention-Driven 3D Affordance Grounding](hammer_harnessing_mllm_via_cross-modal_integration_for_intention-driven_3d_affor.md)**

:   提出 HAMMER 框架，通过从 MLLM 中提取接触感知的意图嵌入、层次化跨模态融合增强点云特征、以及多粒度几何提升模块为意图嵌入注入3D空间信息，实现基于交互图像的3D可供性定位，在 PIAD 基准上全面超越现有方法。

**[HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models](handvqa_diagnosing_and_improving_fine-grained_spatial_reasoning_about_hands_in_v.md)**

:   构建了 HandVQA——一个包含 160 万+选择题的大规模诊断性基准，基于 3D 手部关节标注自动生成关于关节角度、距离和相对位置的 VQA 问题，系统暴露了当前 VLM 在细粒度手部空间推理上的严重缺陷，并证明在 HandVQA 上微调后的模型可零样本迁移到手势识别（+10.33%）和手-物交互识别（+2.63%）等下游任务。

**[HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models](hawk_head_importance-aware_visual_token_pruning_in_multimodal_models.md)**

:   提出 HAWK，一种基于注意力头重要性感知的视觉 token 剪枝方法，通过离线计算各注意力头对视觉理解的贡献权重，并结合文本引导的注意力分数动态评估每个视觉 token 的重要性，在 Qwen2.5-VL 上剪枝 80.2% 视觉 token 后仍保留 96.0% 原始性能，同时减少 26% 推理延迟。

**[HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_highfidelity_incontext_learning_for_multimo.md)**

:   HiFICL 通过严格的注意力公式推导，将 ICL 近似问题从"拟合 shift vector"重构为"直接参数化 ICL 的源头"——在注意力头中注入可学习的低秩虚拟键值对，以端到端训练实现一种动态的、上下文感知的参数高效微调方法，在多个多模态基准上以极少参数超越现有 ICL 近似方法和 LoRA。

**[HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)**

:   HiSpatial 提出将 3D 空间智能分解为四层认知层级（几何感知 → 物体属性 → 物体关系 → 抽象推理），构建了处理约 500 万张图像、4500 万个物体、20 亿 QA 对的自动化数据管线，并设计了以度量尺度点云图为辅助输入的 RGB-D VLM，以仅 3B 参数在多个空间推理基准上超越 GPT-5 和 Gemini-2.5-Pro。

**[HIVE: Query, Hypothesize, Verify — An LLM Framework for Multimodal Reasoning-Intensive Retrieval](hive_query_hypothesize_verify_an_llm_framework_for_multimodal_reasoning-intensiv.md)**

:   HIVE 是一个即插即用的多模态检索框架，通过四个阶段——初始检索 → LLM 驱动的补偿性查询合成（显式表达视觉推理缺口）→ 二次检索 → LLM 验证重排序——将推理密集型多模态检索的 nDCG@10 从最佳多模态模型的 27.6 提升至 41.7（+14.1 绝对点），无需任何额外训练。

**[HOG-Layout: Hierarchical 3D Scene Generation, Optimization and Editing via Vision-Language Models](hog_layout_hierarchical_3d_scene_generation_optimization_and_editing.md)**

:   本文提出 HOG-Layout，一个基于 VLM 和 LLM 的层次化 3D 室内场景生成、优化和编辑框架，通过 RAG 增强语义一致性、力导向层次优化确保物理合理性，在 SceneEval 上以 4.5 倍更快的速度超越 LayoutVLM。

**[HoneyBee: Data Recipes for Vision-Language Reasoners](honeybee_data_recipes_for_vision-language_reasoners.md)**

:   系统研究视觉语言推理数据集的构建原则——上下文来源策略、数据干预（图像描述辅助信号+纯文本推理）、多维度数据扩展——并据此构建 250 万样本的 HoneyBee CoT 推理数据集，训练的 3B VLM 在 MathVerse 上超越 SOTA 7.8%，同时提出降低 73% 解码成本的测试时扩展策略。

**[HouseMind: Tokenization Allows MLLMs to Understand, Generate and Edit Architectural Floor Plans](housemind_tokenization_mllm_floor_plan.md)**

:   提出HouseMind框架，通过层次化VQ-VAE将建筑平面图离散化为轮廓token和房间实例token的结构化序列，结合三阶段多模态对齐和指令微调，以Qwen3-0.6B为backbone实现了平面图理解、生成、编辑三项任务的统一建模，几何有效性和可控性大幅超越现有方法。

**[HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)**

:   提出HulluEdit，一种单次前向、无参考模型的子空间编辑框架，通过将隐藏状态分解为正交的视觉证据子空间、冲突先验子空间和残差不确定性子空间，选择性抑制幻觉模式而不干扰视觉定位，在POPE和CHAIR基准上达到SOTA幻觉缓解效果。

**[HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks](humanvbench_probing_human_centric_video_understanding_in_mllms_with_automatica.md)**

:   提出 HumanVBench，一个包含 16 个细粒度任务的人体中心视频理解基准，配套两个自动化流水线（视频标注 + 干扰项感知 QA 合成），对 30 个主流视频 MLLM 的评测揭示了当前模型在细微情感感知和语音-视觉对齐方面的关键不足。

**[IsoCLIP: Decomposing CLIP Projectors for Efficient Intra-modal Alignment](isoclip_decomposing_clip_projectors_for_efficient_intramodal_alignment.md)**

:   IsoCLIP 从理论上分析 CLIP 投影头的结构，发现余弦相似度计算中隐含一个模态间算子 $\Psi = W_i^\top W_t$ 负责跨模态对齐，和一个模态内算子 $\Psi_i = W_i^\top W_i$ 仅负责归一化但不促进模态内对齐；通过对 $\Psi$ 的奇异值分解识别出近似各向同性(isotropic)的对齐子空间，去除各向异性方向后无需训练即可显著改善模态内检索和分类性能。

**[It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models](its_time_to_get_it_right_improving_analog_clock_reading_and_clock-hand_spatial_r.md)**

:   揭示 SOTA VLM 仍无法可靠读取真实场景中的模拟时钟（零样本准确率不到10%），提出 TickTockVQA 真实场景数据集（12K图像）和 Swap-DPO 微调框架，将 Llama-3.2-11B 的时间读取准确率从1.43%提升至46.22%。

**[Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild](joint-aligned_latent_action_towards_scalable_vla_pretraining_in_the_wild.md)**

:   提出 JALA 框架，通过联合对齐预测嵌入与逆动力学生成的潜在动作，构建统一的潜在动作空间，使 VLA 能同时从标注数据和未标注的野外人类视频中学习，配合 7.5M 样本的 UniHand-Mix 数据集显著提升机器人操作泛化性。

**[KEC: Hierarchical Textual Knowledge for Enhanced Image Clustering](kec_hierarchical_textual_knowledge_clustering.md)**

:   KEC 利用 LLM 构建层级化的概念-属性结构化文本知识来引导图像聚类，在 20 个数据集上无需训练即超越零样本 CLIP 14 个数据集，证明了判别性属性比简单类名更有效。

**[KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing](kvsmooth_mitigating_hallucination_in_multi-modal_large_language_models_through_k.md)**

:   提出KVSmooth，一种免训练的即插即用方法，通过注意力行熵引导的自适应指数移动平均（EMA）对KV-Cache进行平滑，有效抑制多模态大语言模型（MLLM）在解码过程中因sink token引发的语义漂移与幻觉生成，在LLaVA-1.5上将CHAIR_S从41.8降至18.2（降幅56%），同时F1从77.5提升至79.2。

**[Learning What Matters: Prioritized Concept Learning via Relative Error-driven Sample Selection](learning_what_matters_prioritized_concept_learning_via_relative_error-driven_sam.md)**

:   提出 PROGRESS 框架，通过追踪 VLM 在自动发现的多模态概念集群上的学习进度来动态选择最有信息量的训练样本，仅用 16-20% 的标注数据就达到全数据 99-100% 的性能，且总训练时间更短。

**[LFPC: Learning to Focus and Precise Cropping for MLLMs](lfpc_learning_to_focus_and_precise_cropping_for_mllms.md)**

:   LFPC 提出两阶段纯强化学习框架，通过"信息差"机制（降低全局图像分辨率迫使模型依赖高分辨率裁剪区域）和接地损失（提升裁剪精度），解决了现有 agent-based MLLM 中"先答后裁"的虚假工具调用问题，在高分辨率 VQA 上达到 SOTA。

**[Linking Perception, Confidence and Accuracy in MLLMs](linking_perception_confidence_and_accuracy_in_mllms.md)**

:   揭示 MLLM 的严重置信度失校准问题（视觉输入退化时准确率暴跌但置信度不变），提出 CDRL（基于原始-噪声图像对的置信度驱动 RL）进行感知敏感性训练，并利用校准后的置信度实现自适应测试时缩放（CA-TTS），在四个基准上平均提升 8.8%。

**[LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models](llavashield_multimodal_multiturn_safety.md)**

:   针对VLM多模态多轮对话中的恶意意图隐蔽性、上下文风险累积和跨模态联合风险三大挑战，构建4,484个标注对话的MMDS数据集和基于MCTS的MMRT红队框架，提出LLaVAShield审计模型，在用户/助手两侧分别达到F1 95.71%/92.24%，大幅超越GPT-5-mini等基线。

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

**[Mastering Negation: Boosting Grounding Models via Grouped Opposition-Based Learning](mastering_negation_boosting_grounding_models_via_grouped_opposition-based_learni.md)**

:   提出 D-Negation 数据集和 Grouped Opposition-Based Learning (GOBL) 微调机制，通过对立语义配对和两个专用损失函数，仅微调不到 10% 参数即大幅提升视觉定位模型对否定语义的理解能力（最高 +5.7 mAP）。

**[Medic-AD: Towards Medical Vision-Language Model's Clinical Intelligence](medic-ad_towards_medical_vision-language_models_clinical_intelligence.md)**

:   Medic-AD 通过三阶段渐进式训练框架——异常检测（<Ano> token）、时序差异推理（<Diff> token）、可视化解释（热力图），将通用医学 VLM 升级为具备病灶检测、症状追踪和视觉可解释性的临床智能模型，在多项医学任务上达到 SOTA。

**[Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)**

:   揭示了在 VLM 的跨域小样本微调中，增强视觉判别性反而损害跨模态对齐（"判别性陷阱"），提出 SVL + RA 两个即插即用模块来抑制视觉学习捷径并引导跨模态对齐，在 4 个 CDFSL 数据集和 11 个 FSL 数据集上取得 SOTA。

**[Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs](mind_the_way_you_select_negative_texts_pursuing_the_distance_consistency_in_ood_.md)**

:   指出现有基于 VLM 的 OOD 检测方法使用模态内距离（文本-文本或图像-图像）选择负文本，与 CLIP 优化的跨模态距离不一致，提出 InterNeg 从文本和视觉两个视角系统地利用跨模态距离，在 ImageNet 上实现 FPR95 降低 3.47%。

**[MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)**

:   MindPower提出以机器人为中心的心智理论（ToM）推理框架，将感知→信念→欲望→意图→决策→行动组织为六层推理层级，并用Mind-Reward（基于GRPO）优化推理一致性，在决策和动作生成上分别超过GPT-4o 12.77%和12.49%。

**[Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](mitigating_multimodal_hallucinations_via_gradient-based_self-reflection.md)**

:   提出 GACD（Gradient-based Influence-Aware Constrained Decoding），利用一阶 Taylor 梯度估计每个 token 对输出的影响力，在推理阶段同时缓解文本-视觉偏差和共现偏差导致的多模态幻觉，无需辅助模型或微调。

**[MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization](mod-dpo_towards_mitigating_cross-modal_hallucinations_in_omni_llms_using_modalit.md)**

:   提出 MoD-DPO（Modality-Decoupled DPO），通过不变性正则化、敏感性正则化和语言先验去偏三个机制解耦多模态 LLM 中各模态的贡献，有效缓解跨模态幻觉（如用听觉信息回答视觉问题），并推导出闭式最优策略。

**[MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](modes_accelerating_mixture-of-experts_multimodal_large_language_models_via_dynam.md)**

:   提出 MoDES，首个面向 MoE 多模态大模型的训练免调专家跳过框架，通过全局调制的局部门控（GMLG）和双模态阈值（DMT）机制自适应跳过冗余专家，在跳过 88% 专家时仍保留 97%+ 原始性能，并实现 2.16× prefill 加速。

**[MODIX: Training-Free Multimodal Information-Driven Positional Index Scaling for VLMs](modix_positional_index_scaling.md)**

:   提出 MODIX，一个免训练框架，通过信息论分析（协方差熵+跨模态对齐）动态调整 VLM 中视觉和文本 token 的位置编码步长，将位置粒度分配给信息密集的模态以提升多模态推理。

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

**[MUPO: All Roads Lead to Rome - Incentivizing Divergent Thinking in Vision-Language Models](mupo_all_roads_lead_to_rome_incentivizing_divergent_thinking_in_vlms.md)**

:   MUPO 揭示了 GRPO 训练导致推理多样性坍缩的问题——模型过早收敛到少数推理策略而丢弃大多数替代方案。通过将响应分组进行局部化优势估计并引入多样性奖励，MUPO 激励 VLM 保持发散思维，在多个推理基准上提升 2-7%。

**[Nano-EmoX: Unifying Multimodal Emotional Intelligence from Perception to Empathy](nano-emox_unifying_multimodal_emotional_intelligence_from_perception_to_empathy.md)**

:   Nano-EmoX 提出认知启发的三级情感任务层次（感知→理解→交互），是首个以2.2B紧凑参数统一六项核心情感任务的多模态语言模型，通过P2E渐进式训练框架从基础感知逐步培养到高层共情能力。

**[Narrative Weaver: Towards Controllable Long-Range Visual Consistency with Multi-Modal Conditioning](narrative_weaver_towards_controllable_long-range_visual_consistency_with_multi-m.md)**

:   提出 Narrative Weaver 框架，结合 MLLM 的叙事规划与扩散模型的精细生成，通过可学习查询和动态 Memory Bank 实现多模态条件下的长程视觉一致性生成，并构建首个电商广告视频分镜数据集 EAVSD（330K+ 图像）。

**[No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models](no_hard_negatives_required_concept_centric_learning_leads_to_compositionality_wi.md)**

:   C2LIP 提出不依赖 hard negatives 的对比学习微调方案：通过将文本拆解为名词短语概念并引入跨模态注意力池化，在 SugarCrepe/SugarCrepe++ 组合性基准上达到 SOTA，同时保持甚至提升零样本和检索性能。

**[Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](noise-aware_few-shot_learning_through_bi-directional_multi-view_prompt_alignment.md)**

:   提出NA-MVP框架，通过双向（clean + noise-aware）多视图prompt设计配合非平衡最优传输（UOT）实现细粒度patch-to-prompt对齐，并用经典OT对识别出的噪声样本做选择性标签修正，在噪声小样本学习场景下持续超越SOTA。

**[OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models](oddgridbench_exposing_the_lack_of_fine-grained_visual_discrepancy_sensitivity_in.md)**

:   提出 OddGridBench 评估 MLLM 的细粒度视觉差异感知能力（找出网格中与其他元素在颜色/大小/旋转/位置上不同的那个），发现所有 MLLM 远低于人类水平，进而提出 OddGrid-GRPO（课程学习 + 距离感知奖励）显著提升模型的视觉辨别力。

**[On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)**

:   揭示了动态 MoE 持续学习中"token 困境"——新任务数据中的模糊和旧 token 对新知识贡献微弱却会导致路由漂移和灾难性遗忘，提出 LLaVA-DyMoE 通过 Token Assignment Guidance 和 Routing Score Regularization 缓解路由漂移，在 CoIN 基准上 MFN 提升超 7%，遗忘降低 12%。

**[Overthinking Causes Hallucination: Tracing Confounder Propagation in Vision Language Models](overthinking_causes_hallucination_tracing_confounder_propagation_in_vision_langu.md)**

:   揭示VLM幻觉的新机制——"过度思考"(overthinking)：模型在中间解码层产生过多竞争性物体假设，混杂因子沿层传播至最终预测引发幻觉；提出Overthinking Score量化层间假设多样性×不确定性，在MSCOCO上F1达78.9%，OOD AMBER上71.58%。

**[PaddleOCR-VL: Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing](paddleocr_vl_coarse_to_fine_document_parsing.md)**

:   PaddleOCR-VL 提出粗到细的文档解析架构：粗阶段用轻量级有效区域聚焦模块(VRFM)定位文档中的有效视觉区域并预测阅读顺序，细阶段用紧凑的0.9B视觉语言模型对裁剪区域进行精细识别，在最少视觉token和参数下实现文档解析SOTA。

**[Parallel In-context Learning for Large Vision Language Models](parallel_in-context_learning_for_large_vision_language_models.md)**

:   提出 Parallel-ICL，将多模态 in-context learning 的长 demonstration 上下文分块并行处理，通过加权 Product-of-Experts 在 logit 层集成，实现与全上下文 MM-ICL 相当甚至更优的性能，同时显著降低推理延迟。

**[PersonaVLM: Long-Term Personalized Multimodal LLMs](personavlm_long_term_personalized_multimodal_llms.md)**

:   本文提出 PersonaVLM，一个面向长期个性化的多模态智能体框架，通过主动记忆管理（四类记忆数据库）、多步推理检索和动量式人格演化机制，将通用 MLLM 转化为能适应用户偏好变化的个性化助手，在 128K 上下文下超越 GPT-4o 5.2%。

**[PhysInOne: Visual Physics Learning and Reasoning in One Suite](physisinone_visual_physics_learning_and_reasoning_in_one_suite.md)**

:   PhysInOne是一个包含153,810个动态3D场景和200万个标注视频的大规模合成数据集，覆盖力学、光学、流体动力学和磁学的71种基本物理现象，为物理感知的世界模型建立了新基准。

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

**[Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher](purify-then-align_towards_robust_human_sensing_under_modality_missing_with_knowl.md)**

:   本文提出PTA（Purify-then-Align）框架，通过元学习驱动的模态加权机制先"净化"噪声多模态教师，再用扩散模型驱动的知识蒸馏"对齐"每个单模态学生，使单模态编码器在模态缺失场景下保持强鲁棒性，在MM-Fi和XRF55上实现SOTA。

**[Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)**

:   提出 Reason-SVG 框架，通过"Drawing-with-Thought"(DwT)范式让 LLM 在生成 SVG 之前先进行显式的多阶段设计推理，并结合 SFT + GRPO 强化学习与混合奖励函数进行训练，在语义对齐、结构有效性和视觉质量上全面超越现有方法。

**[ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps](reasonmap_towards_fine-grained_visual_reasoning_from_transit_maps.md)**

:   提出 ReasonMap 基准，利用 30 个城市的高分辨率公交地图构建 1,008 个 QA 对，通过两级评估框架（正确性+质量）系统评估 16 个 MLLM 的细粒度视觉推理能力，发现开源模型中 base 优于 reasoning 而闭源模型相反。

**[ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval](recall_recalibrating_capability_degradation_for_mllm-based_composed_image_retrie.md)**

:   揭示了将生成式MLLM适配为判别式检索器时的"能力退化"现象（Capability Degradation），提出ReCALL框架通过诊断检索器盲点→利用基座MLLM的CoT推理生成纠正性三元组→分组对比精炼三阶段管线，有效恢复退化的细粒度组合推理能力，在CIRR上R@1达55.52%、FashionIQ上R@10达57.04%。

**[Recursive Think-Answer Process for LLMs and VLMs](recursive_think-answer_process_for_llms_and_vlms.md)**

:   R-TAP 提出一种递归思考-回答过程，通过置信度生成器评估模型回答确定性并引导迭代推理修正，配合递归置信度增长奖励和最终答案置信度奖励的双重强化信号，在 LLM 和 VLM 上均一致超越单次推理方法，同时显著减少推理中的"Oops!"式自我反思表达。

**[ReHARK: Refined Hybrid Adaptive RBF Kernels for Robust One-Shot Vision-Language Adaptation](rehark_refined_hybrid_adaptive_rbf_kernels_for_robust_one-shot_vision-language_a.md)**

:   提出 ReHARK 框架，通过混合语义-视觉先验构建、支撑集增强、自适应分布校正和多尺度 RBF 核集成四阶段精炼管道，在 11 个基准上实现 65.83% 的单样本适应 SOTA 准确率，显著超越 Tip-Adapter 和 ProKeR。

**[World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training](rehearsevla_simulated_post-training_for_vlas_with_physically-consistent_world_mo.md)**

:   提出 World-Env 框架，利用物理一致的世界模型作为虚拟环境替代真实交互，对 VLA 模型进行 RL post-training，仅需每任务 5 条示教即可显著提升操控成功率。

**[Relational Visual Similarity](relational_visual_similarity.md)**

:   本文首次形式化定义关系视觉相似度问题（两图像间的内在关系/功能对应，而非表面属性相似），构建114K匿名描述数据集并训练relsim模型，揭示了现有相似度指标（CLIP/DINO等）在捕捉关系相似度方面的根本性缺陷。

**[ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)**

:   提出 ReMoRa，直接操作视频压缩表示（I帧 + 运动向量），通过 Refined Motion Representation (RMR) 模块将粗糙的块级运动向量精化为接近光流的细粒度运动表征，再用 Hierarchical Motion State Space (HMSS) 模块进行线性时间的长程时间建模，在 LongVideoBench、NExT-QA、MLVU 等基准上超越基线。

**[Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)**

:   提出 Residual Decoding (ResDec)——一种训练免的即插即用解码策略，通过分析历史 token 的 logit 分布中的 U 型 JSD 模式发现语义锚定阶段，聚合该阶段的历史 logits 作为残差引导融入当前解码，以近乎零的额外推理开销有效抑制 LVLM 中的语言先验幻觉。

**[Responses Fall Short of Understanding: Revealing the Gap between Internal Representations and Responses in VDU](responses_fall_short_of_understanding_gap_between_internal_representations_and_responses_in_vdu.md)**

:   通过逐层线性探测分析发现 LVLM 在视觉文档理解中存在内部表示与生成响应之间的显著差距，且中间层比最终层编码了更线性可访问的任务信息，微调中间层可同时提升准确率和缩小差距。

**[Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token](rethinking_mllm_itself_as_a_segmenter_with_a_single_segmentation_token.md)**

:   提出 SELF1E，首次实现不依赖专用 mask 解码器且仅用单个 [SEG] token 的 MLLM 分割方法，通过 Residual Features Refilling (RFR) 和 Residual Features Amplifier (RFA) 恢复 pixel-shuffle 压缩造成的分辨率损失，在多个分割任务上达到与解码器方法竞争力相当的性能。

**[Rethinking VLMs for Image Forgery Detection and Localization](rethinking_vlms_for_image_forgery_detection_and_localization.md)**

:   提出 IFDL-VLM 框架，发现 VLM 固有的语义合理性偏向（而非真实性）会阻碍伪造检测性能，因此将检测/定位与语言解释解耦为两阶段优化，并利用定位掩码作为 VLM 的辅助输入增强可解释性，在 9 个基准上全面达到 SOTA。

**[Revisiting Model Stitching in the Foundation Model Era](revisiting_model_stitching_in_the_foundation_model.md)**

:   本文系统研究视觉基础模型（VFM）之间的拼接可行性，发现传统方法在VFM上失效，提出"Final Feature Matching + Task Loss"两阶段训练策略使异构VFM可靠拼接，拼接模型甚至能超越两个单独VFM，进而提出VFM Stitch Tree（VST）架构为多VFM系统提供可控的精度-效率权衡方案。

**[SALMUBench: A Benchmark for Sensitive Association-Level Multimodal Unlearning](salmubench_a_benchmark_for_sensitive_association-level_multimodal_unlearning.md)**

:   提出 SALMUBench——首个针对 CLIP 类模型的关联级别机器遗忘基准，包含 60K 合成人物-敏感属性配对数据集、从头训练的 Compromised/Clean 模型对，以及结构化 holdout 集评估协议，首次系统揭示了现有遗忘方法的三种失败模式（灾难性破坏、过度泛化遗忘、无效遗忘）。

**[Scaling Spatial Intelligence with Multimodal Foundation Models](scaling_spatial_intelligence_with_multimodal_foundation_models.md)**

:   SenseNova-SI 通过系统化构建800万级多样化空间数据（SenseNova-SI-8M），在 Qwen3-VL、InternVL3 和 Bagel 等多模态基础模型上培养空间智能能力，在 VSI-Bench、MMSI 等多个空间基准上取得前所未有的性能，同时保持通用多模态理解能力。

**[Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework](scaling_test-time_robustness_of_vision-language_models_via_self-critical_inferen.md)**

:   提出 Self-Critical Inference (SCI) 框架，通过多轮文本+视觉反事实推理的 logit 聚合来同时解决 LVLM 的语言偏差和语言敏感性问题，并提出 DRBench 动态鲁棒性基准来模型特异地评估鲁棒性。增加反事实推理轮次可持续提升鲁棒性，开辟了测试时缩放的新方向。

**[Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)**

:   提出 FlexMem——一种训练免的视觉记忆机制，通过迭代式双路径 KV 缓存压缩构建视觉记忆库，结合编码式和快速索引式记忆召回策略，让 MLLM 在单张 3090 GPU 上处理 1000+ 帧长视频，大幅超越现有高效视频理解方法。

**[Scene-VLM: Multimodal Video Scene Segmentation via Vision-Language Models](scene-vlm_multimodal_video_scene_segmentation_via_vision-language_models.md)**

:   提出 Scene-VLM——首个基于微调 VLM 的视频场景分割框架，通过结构化多模态镜头表征（视觉帧+对白+元数据）、因果序列预测、上下文-焦点窗口机制和 token logits 置信度提取，在 MovieNet 上取得 +6 AP 和 +13.7 F1 的大幅提升，并展示了自然语言解释能力。

**[SciPostGen: Bridging the Gap between Scientific Papers and Poster Layouts](scipostgen_bridging_the_gap_between_scientific_papers_and_poster_layouts.md)**

:   构建了包含 18,097 个论文-海报对的大规模数据集 SciPostGen，分析发现论文结构与海报布局元素数量存在中等相关性，并提出检索增强海报布局生成框架，通过对比学习检索与论文匹配的布局模板来指导 LLM 生成海报布局。

**[SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker](seatrack_multimodal_tracker.md)**

:   提出 SEATrack 多模态跟踪器，通过 AMG-LoRA 实现跨模态注意力图的动态对齐，以及 HMoE 实现高效全局关系建模的跨模态融合，在 RGB-T/D/E 跟踪中以极少参数实现 SOTA 的性能-效率平衡。

**[See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models](see_hear_and_understand_benchmarking_audiovisual_human_speech_understanding_in_mul.md)**

:   提出 AV-SpeakerBench 基准，包含 3,212 道以说话人为中心的音视频推理多选题，系统评估多模态大语言模型在"谁在说话、说了什么、何时说的"上的细粒度音视频融合能力，揭示当前最强模型与人类表现仍有超 20% 的差距。

**[See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles](see_think_act_teaching_multimodal_agents_to_effectively_interact_with_gui_by_ide.md)**

:   提出 State-aware Reasoning (StaR)，通过教会多模态 Agent "感知当前状态→分析目标状态→决定是否操作"的三步推理链，将 GUI 开关控制准确率提升超 30%，同时不损害通用 Agent 任务性能。

**[Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness](seeing_clearly_reasoning_confidently_plug-and-play_remedies_for_vision_language_.md)**

:   提出一种高效的即插即用模块，通过学习多模态类嵌入来增强 VLM 对稀有物体的识别和推理能力：在视觉端用 cross-attention 适配器精化视觉 token，在文本端注入物体检测提示，无需微调 VLM 即可在 CODA-LM 上获得 72.8→75.4 的显著提升。

**[Seeing Through Touch: Tactile-Driven Visual Localization of Material Regions](seeing_through_touch_tactile_localization.md)**

:   提出触觉定位任务——给定触觉输入识别图像中具有相同材质属性的区域，通过局部视觉-触觉对齐和材质多样性配对策略学习密集跨模态特征，构建两个新的触觉-材质分割数据集。

**[Self-Consistency for LLM-Based Motion Trajectory Generation and Verification](self-consistency_for_llm-based_motion_trajectory_generation_and_verification.md)**

:   将 LLM 的自一致性范式从自然语言推理扩展到视觉域——用 Lie 变换群层次结构定义运动轨迹的形状族，通过在变换不变距离度量下聚类 LLM 采样的多条轨迹，实现无监督的轨迹生成改进（+4-6%）和验证（精度+11.8%），无需训练。

**[Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning](similarity-as-evidence_calibrating_overconfident_vlms_for_interpretable_and_labe.md)**

:   提出 Similarity-as-Evidence (SaE) 框架，将 VLM 的文本-图像相似度重新解释为 Dirichlet 证据，通过 Similarity Evidence Head (SEH) 校准过度自信的 softmax 输出，并基于 vacuity（知识空缺）和 dissonance（证据冲突）的双因子采集策略实现可解释、高效的医学主动学习，在 10 个数据集上以 20% 标注预算达到 82.57% 的 SOTA 宏平均准确率。

**[SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models](simpact_simulation-enabled_action_planning_using_vision-language_models.md)**

:   SIMPACT 提出一种测试时的仿真增强动作规划框架，从单张 RGB-D 图像自动构建物理仿真环境，使 VLM 能够提出动作、观察仿真结果并迭代优化推理，无需额外训练即可在刚体和可变形物体操作任务上达到 SOTA 性能。

**[SoPE: Spherical Coordinate-Based Positional Embedding for 3D LVLMs](sope_spherical_positional_encoding_3d_lvlm.md)**

:   揭示 RoPE 在 3D LVLM 中的空间感知偏差问题（1D 索引破坏 3D 局部性且忽视方向），提出球面坐标位置编码 SoPE（$(t,r,\theta,\phi)$ 四维索引 + 多维频率分配 + 多尺度混合），在 SpatialLM 上实现 3D 布局估计和物体检测 SOTA。

**[SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)**

:   提出 SPARROW 框架，通过 **目标特定追踪特征（TSF）** 注入时间一致性监督、**双提示（[BOX]+[SEG]）粗到细解码** 稳定首帧初始化，以即插即用方式集成到现有视频 MLLM 上，在 6 个基准 3 个任务上取得一致提升。

**[SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)**

:   提出SpatiaLQA基准（9605个QA对、241个真实室内场景），系统评估41个VLM在空间逻辑推理上的表现，并设计递归场景图辅助推理方法来提升VLM的空间逻辑推理能力。

**[SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence](spatialscore_towards_comprehensive_evaluation_for_spatial_intelligence.md)**

:   本文提出了目前最全面的多模态空间智能基准 SpatialScore（5K样本/30任务），并通过数据驱动的 SpatialCorpus（331K QA）微调方案和免训练的 SpatialAgent（12个工具）两条互补路径来提升 MLLM 的空间理解能力。

**[SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning](spatialstack_layered_geometry-language_fusion_for_3d_vlm_spatial_reasoning.md)**

:   提出SpatialStack框架，将多视图几何编码器（VGGT）的多层级几何特征逐层注入LLM解码器的不同层（而非仅融合最后一层），通过浅层→细粒度空间感知、深层→高层语义推理的层级对齐，在多个3D空间推理基准上达到开源SOTA。

**[StructXLIP: Enhancing Vision-Language Models with Multimodal Structural Cues](structxlip_enhancing_vision-language_models_with_multimodal_structural_cues.md)**

:   StructXLIP 将边缘图（edge map）作为视觉结构的代理表示，在 CLIP 微调中引入三种结构中心损失（边缘-结构文本对齐 + 局部区域-文本块匹配 + 边缘-彩色图连接），通过最大化多模态结构表示的互信息引导模型走向更鲁棒的语义稳定最优解，在跨模态检索任务上超越现有竞争者。

**[Taxonomy-Aware Representation Alignment for Hierarchical Visual Recognition with Large Multimodal Models](taxonomy-aware_representation_alignment_for_hierarchical_visual_recognition_with.md)**

:   提出TARA框架，通过将LMM的中间表示与生物基础模型(BFM)的分类学感知特征对齐，为大型多模态模型注入分类层次知识，显著提升已知和新颖类别的层次化视觉识别性能。

**[Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)**

:   提出Vision-Guided Attention (VGA)，一种免训练的方法，通过利用视觉token的语义特征构建精确的视觉定位，引导模型注意力聚焦于相关视觉区域，有效缓解MLLM幻觉，且兼容FlashAttention。

**[Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction](text-only_training_for_image_captioning_with_retrieval_augmentation_and_modality.md)**

:   提出TOMCap——一种纯文本训练的图像描述方法，通过检索增强+模态差距修正+LoRA微调，在训练时只用文本而推理时处理图像，超越了已有的无训练和纯文本方法。

**[The Coherence Trap: When MLLM-Crafted Narratives Exploit Manipulated Visual Contexts](the_coherence_trap_when_mllm-crafted_narratives_exploit_manipulated_visual_conte.md)**

:   揭示现有多模态篡改检测忽视了MLLM能生成语义一致的欺骗性叙事这一核心威胁，构建441k样本的MDSM语义对齐篡改数据集，并提出基于Artifact Token和操纵导向推理的AMD框架，在跨域检测中以仅0.27B参数达到88.18 ACC / 60.25 mAP / 61.02 mIoU的最优泛化性能。

**[The LLM Bottleneck: Why Open-Source Vision LLMs Struggle with Hierarchical Visual Recognition](the_llm_bottleneck_why_open-source_vision_llms_struggle_with_hierarchical_visual.md)**

:   揭示开源LLM缺乏关于视觉世界的层次分类知识（甚至不知道基本的生物分类体系），这使得LLM成为Vision LLM层次视觉识别的瓶颈。

**[The More, the Merrier: Contrastive Fusion for Higher-Order Multimodal Alignment](the_more_the_merrier_contrastive_fusion_for_higher-order_multimodal_alignment.md)**

:   提出Contrastive Fusion (ConFu)框架，将CLIP式的双模态对比学习推广到三模态高阶对齐，在统一目标中同时学习配对和融合表示，支持1→1和2→1检索。

**[Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth](think_360_evaluating_the_width-centric_reasoning_capability_of_mllms_beyond_dept.md)**

:   本文提出 Think360，一个聚焦于"推理宽度"（即模型在多路径搜索、多约束剪枝、回溯试错等方面的能力）的多模态基准，包含 1200+ 高质量样本，并设计细粒度 Tree-of-Thought 评估协议，揭示当前 MLLM 在宽度方向推理上的显著短板。

**[Thinking Diffusion: Penalize and Guide Visual-Grounded Reasoning in Diffusion Multimodal Language Models](thinking_diffusion_penalize_and_guide_visual-grounded_reasoning_in_diffusion_mul.md)**

:   首次定量分析扩散多模态LLM (dMLLM)的CoT推理过程，发现"早期回答生成"和"弱视觉依赖"两个关键问题，提出PSP（位置-步骤惩罚）和VRG（视觉推理引导）两种免训练方法，在3倍加速下获得最高7.5%的精度提升。

**[Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)**

:   提出 Dyn-Bench——一个面向 4D 物理世界动态理解的大规模基准（1k 视频、7k VQA 对、3k 动态 grounding 对），系统评估了通用/空间/区域级 MLLM 的时空推理能力，发现现有模型无法同时维持推理和 grounding 的一致性，并提出 Mask-Guided Fusion 和 ST-TCM 两种结构化集成方法显著提升动态感知。

**[TIGeR: A Unified Framework for Time, Images and Geo-location Retrieval](tiger_a_unified_framework_for_time_images_and_geo-location_retrieval.md)**

:   提出TIGeR框架，通过多模态Transformer联合学习图像-位置-时间的统一地理时间嵌入空间，实现地理定位、拍摄时间预测和地理时间感知图像检索三个任务的统一，并构建了4.5M规模的高质量基准数据集。

**[TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)**

:   系统调查构建MLLM视频时间定位（VTG）能力的关键因素，从数据质量和算法设计两个维度出发，发布高质量基准TimeLens-Bench和训练集TimeLens-100K，并通过交错文本时间编码+thinking-free RLVR训练范式构建TimeLens系列模型，在开源模型中达到SOTA并超越GPT-5和Gemini-2.5-Flash。

**[TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment](tipsv2_patch_text_alignment.md)**

:   提出 TIPSv2，通过发现蒸馏能显著提升 patch-text 对齐能力，并将该洞察转化为新的预训练目标 iBOT++（可见 token 也参与损失计算），结合头部EMA和多粒度文本增强，在 9 个任务 20 个数据集上达到 SOTA。

**[Token Warping Helps MLLMs Look from Nearby Viewpoints](token_warping_helps_mllms_look_from_nearby_viewpoints.md)**

:   提出对 MLLM 的 ViT image token 做空间 warping（而非传统的像素级 warping）来模拟视角变换，发现 backward token warping 在保持语义一致性同时对深度估计噪声鲁棒，在自建的 ViewBench 上大幅超越像素级 warping、专用空间推理 MLLM 和生成式 warping 方法。

**[Topo-R1: Detecting Topological Anomalies via Vision-Language Models](topo-r1_detecting_topological_anomalies_via_vision-language_models.md)**

:   提出Topo-R1——首个赋予VLM拓扑感知能力的框架，通过自动化数据构建管线+SFT+GRPO强化学习（含拓扑感知复合奖励），实现无标注的管状结构拓扑异常检测与分类。

**[Towards Calibrating Prompt Tuning of Vision-Language Models](towards_calibrating_prompt_tuning_of_vision-language_models.md)**

:   针对prompt tuning后CLIP面临的"双重误校准"问题（基类欠自信+新类过自信），提出均值-方差margin正则化和文本矩匹配损失两个互补正则项，作为即插即用模块在7种prompt tuning方法和11个数据集上显著降低ECE。

**[Towards Multimodal Domain Generalization with Few Labels](towards_multimodal_domain_generalization_with_few_labels.md)**

:   定义并研究半监督多模态域泛化(SSMDG)新问题，提出融合一致性驱动伪标签、分歧感知正则化和跨模态原型对齐的统一框架，在少量标注下实现多模态模型的跨域泛化。

**[Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training](towards_real-world_document_parsing_via_realistic_scene_synthesis_and_document-a.md)**

:   提出数据-训练协同设计框架 DocHumming：通过 Realistic Scene Synthesis 构建 DocMix-3M 大规模合成数据集，结合渐进学习和结构 token 加权的 Document-Aware Training Recipe，在仅 1B 参数的 MLLM 上实现 OmniDocBench Overall 93.75（超越 Qwen3-VL-235B 的 89.15），且在真实拍摄场景下仅退化 6.72 分（模块化方法退化 18-20 分）。

**[Training High-Level Schedulers with Execution-Feedback Reinforcement Learning for Long-Horizon GUI Automation](training_high-level_schedulers_with_execution-feedback_reinforcement_learning_fo.md)**

:   提出 CES（Coordinator-Executor-State Tracker）多智能体框架和分阶段执行反馈强化学习算法，将高层任务规划与低层执行解耦，通过专门训练的 Coordinator 和 State Tracker 显著提升 GUI Agent 在长时序任务上的规划和状态管理能力。

**[TreeTeaming: Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration](treeteaming_autonomous_red-teaming_of_vision-language_models_via_hierarchical_s.md)**

:   TreeTeaming 提出了一个基于层次策略树的自动化红队测试框架，通过 LLM 驱动的 Orchestrator 动态地探索和进化攻击策略，在12个主流 VLM 上实现了 SOTA 的攻击成功率（GPT-4o 达 87.60%），并发现了超越已知策略集的多样化新攻击手段。

**[TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition](trivia_self-supervised_fine-tuning_of_vision-language_models_for_table_recogniti.md)**

:   提出 TRivia 自监督微调框架，通过表格问答（QA）驱动的 GRPO 强化学习，让 VLM 直接从无标注表格图像中学习表格识别能力，3B 参数的 TRivia-3B 在多个基准上超越 Gemini 2.5 Pro 和 GPT-5 等私有模型。

**[Unbiased Dynamic Multimodal Fusion](unbiased_dynamic_multimodal_fusion.md)**

:   UDML 提出无偏动态多模态学习框架，包含噪声感知不确定性估计器（通过注入可控噪声并预测其强度来实现在低噪和高噪条件下均准确的模态质量评估）和模态依赖计算器（通过 Dropout 量化模型对各模态的固有依赖偏差并融入加权机制），解决了现有方法的双重抑制问题，在多个多模态基准上一致提升性能。

**[Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models](uncertainty-aware_knowledge_distillation_for_multimodal_large_language_models.md)**

:   提出Beta-KD，一种基于贝叶斯视角的不确定性感知知识蒸馏框架，通过将教师监督建模为Gibbs先验并用Laplace近似推导闭形解，自动调节数据与教师信号的平衡，在多模态VQA基准上持续提升蒸馏效果。

**[Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models](uncertainty-guided_compositional_alignment_with_part-to-whole_semantic_represent.md)**

:   提出UNCHA框架，在双曲VLM中用双曲不确定性建模部分图像对整体场景的语义代表性，通过不确定性引导的对比损失和蒸含损失增强组合性场景理解，在多个下游任务上超趇现有双曲VLM。

**[Understanding Task Transfer in Vision-Language Models](understanding_task_transfer_in_vision-language_models.md)**

:   本文首次系统研究了 VLM 在一个视觉感知任务上微调后对其他感知任务零样本性能的影响，提出 Perfection Gap Factor (PGF) 归一化指标量化跨任务迁移，在 Qwen-2.5-VL 三个尺度模型上揭示了任务迁移的结构性规律（正/负迁移团、任务角色分类、尺度依赖等），并证明 PGF 可指导数据选择提升微调效率。

**[UNICBench: UNIfied Counting Benchmark for MLLM](unicbench_unified_counting_benchmark_for_mllm.md)**

:   推出UNICBench，首个统一的跨模态（图像/文本/音频）多层级计数基准，包含5,508+5,888+2,905共14,301个QA对及三级能力(Pattern/Semantic/Reasoning)×三级难度(Easy/Medium/Hard)分类，系统评估45个SOTA MLLM，揭示基本计数任务趋近但推理级和困难任务存在显著差距。

**[Variation-Aware Vision Token Dropping for Faster Large Vision-Language Models](variation-aware_vision_token_dropping_for_faster_large_vision-language_models.md)**

:   提出 V2Drop，首次从 token 变化量（variation）视角出发，通过渐进式丢弃 LLM 内部变化量最小的"懒惰"视觉 token，实现无训练、无位置偏差、兼容高效算子的 LVLM 推理加速，在图像和视频理解任务中分别保留 94.0% 和 98.6% 原始性能，同时降低 LLM 生成延迟 31.5% 和 74.2%。

**[Venus: Benchmarking and Empowering Multimodal Large Language Models for Aesthetic Guidance and Cropping](venus_benchmarking_and_empowering_multimodal_large_language_models_for_aesthetic.md)**

:   定义审美指导(AG)新任务并构建AesGuide基准(10748张照片含审美评分、分析和指导标注)，提出Venus两阶段框架——先通过渐进式审美问答赋能MLLM审美指导能力，再通过CoT推理激活审美裁剪能力，在两个任务上均达到SOTA。

**[Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)**

:   提出VisionToM，一个基于视觉的轻量级干预框架，通过探测和干预MLLM中对视觉输入和ToM推理敏感的注意力头，在不微调模型的情况下显著增强多模态大语言模型的心智理论推理能力，在EgoToM基准上大幅提升表现。

**[VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatio-temporal_collaborative_network_for_multi-modal_video_fusion.md)**

:   提出首个大规模红外-可见光视频融合框架 VideoFusion，通过跨模态差分增强、完整模态引导融合和双向时序协同注意力机制，联合建模跨模态互补性与时序动态，生成时空一致的高质量融合视频，并构建了包含220个视频/15.4万帧的 M3SVD 数据集。

**[ViKey: Enhancing Temporal Understanding in Videos via Visual Prompting](vikey_enhancing_temporal_understanding_in_videos_via_visual_prompting.md)**

:   ViKey 通过在视频帧上叠加帧序号的视觉提示（Visual Prompting），配合轻量的关键词-帧映射（KFM）模块，在免训练条件下显著提升 VideoLLM 的时序推理能力，即使只用 20% 的帧也能接近密集帧的性能。

**[ViRC: Enhancing Visual Interleaved Mathematical CoT with Reason Chunking](virc_enhancing_visual_interleaved_mathematical_cot_with_reason_chunking.md)**

:   ViRC 提出 Reason Chunking 机制，将多模态数学 CoT 结构化为连续的"关键推理单元（CRU）"，模拟人类专家反复审视图像并逐步证明中间命题的过程，通过 CRUX 数据集和渐进式训练策略（Instructional SFT → Practice SFT → Strategic RL），实现ViRC-7B 在数学基准上平均提升 18.8%。

**[Vision-Language Models Encode Clinical Guidelines for Concept-Based Medical Reasoning](vision-language_models_encode_clinical_guidelines_for_concept-based_medical_reas.md)**

:   提出MedCBR框架，通过将临床诊断指南（如BI-RADS）融入概念瓶颈模型的训练和推理过程，利用LVLM生成指南一致性报告增强概念监督，结合多任务CLIP训练和大推理模型生成结构化临床解释，在超声和乳腺X光癌症检测上达到94.2%和84.0%的AUROC。

**[VISion On Request: Enhanced VLLM Efficiency with Sparse, Dynamically Selected, Vision-Language Interactions](vision_on_request_enhanced_vllm_efficiency_with_sparse_dynamically_selected_visi.md)**

:   VISOR 提出了一种区别于视觉 token 压缩的新效率范式——通过稀疏化 LLM 内部视觉-语言交互层（少量交叉注意力 + 动态选择的自注意力层），在保留完整高分辨率视觉 token 的同时实现 8.6-18 倍 FLOPs 节省，尤其在需要细粒度理解的困难任务上大幅超越 token 压缩方法。

**[VL-RouterBench: A Benchmark for Vision-Language Model Routing](vl-routerbench_a_benchmark_for_vision-language_model_routing.md)**

:   提出VL-RouterBench，首个面向视觉-语言模型的系统性路由基准，涵盖14个数据集、17个候选模型和519,180个样本-模型对，评估10种路由方法，并发现当前最优路由器与理想Oracle之间仍存在显著差距。

**[VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery](vlm-guided_group_preference_alignment_for_diffusion-based_human_mesh_recovery.md)**

:   提出基于VLM的双记忆自反思评判代理（Critique Agent）为扩散式人体网格恢复生成组级偏好信号，再通过组偏好对齐（Group Preference Alignment）微调扩散模型，无需3D标注即可大幅提升野外场景下的HMR精度。

**[VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models](vlm-loc_localization_in_point_cloud_maps_via_vision-language_models.md)**

:   提出VLM-Loc框架，将3D点云地图转换为BEV图像和场景图供VLM进行结构化空间推理，结合部分节点分配（PNA）机制实现文本-点云精细定位，在自建的CityLoc基准上以Recall@5m提升14.20%大幅超越先前SOTA。

**[VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)**

:   提出VLM-Pruner，一种免训练的离心式token剪枝方法，通过空间稀疏缓冲（BSS）准则平衡冗余消除与局部细节完整性，在88.9%剪枝率下跨5个VLM一致超越现有方法，同时实现端到端推理加速。

**[VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments](vs_bench_evaluating_vlms_for_strategic_abilities_in_multi_agent_environments.md)**

:   本文提出 VS-Bench，一个包含十个视觉化博弈环境的多模态基准，从感知、策略推理和决策三个维度系统评估 VLM 在多智能体环境中的策略能力，发现当前最强模型在推理和决策上仍与最优表现有显著差距。

**[Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training](wan-weaver_interleaved_multi-modal_generation_via_decoupled_training.md)**

:   Wan-Weaver 提出规划器（VLM）+ 可视化器（DiT）的解耦架构，通过大规模文本代理数据训练规划器而非真实交错数据，在 OpenING 上 Overall 8.67 分超越 Nano Banana 的 8.85，在保持理解能力（MMMU 74.9）的同时实现 SOTA 交错文图生成。

**[WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs](weavetime_streaming_video_llm_memory.md)**

:   诊断了当前 Video-LLM 存在的"时间不可知"（Time-Agnosticism）问题，提出 WeaveTime 框架，通过训练时的时序重建辅助任务（SOPE）赋予模型时序感知能力，推理时用不确定性门控的粗到细记忆缓存（PCDF-Cache）实现高效自适应记忆检索，在流式视频 QA 上取得显著提升。

**[What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)**

:   提出EmbedLens探针工具系统分析MLLM中视觉token的内部结构，发现视觉token分为sink/dead/alive三类（约40%为无用token），alive token已在进入LLM前编码丰富语义（"预语言"特性），且LLM内部视觉计算对大多数任务冗余，直接中层注入即可。

**[When to Think and When to Look: Uncertainty-Guided Lookback](when_to_think_and_when_to_look_uncertainty-guided_lookback.md)**

:   本文首次系统分析了 LVLM 中 test-time thinking 对视觉推理的影响，发现"多想不如多看"——长推理链常忽略图像导致"long-wrong"轨迹，并据此提出不确定性引导的 lookback 解码策略，通过在推理链漂移时注入视觉回看提示，在不修改模型的前提下将 MMMU 等 6 个基准提升 2-6 个点。

**[When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs](when_token_pruning_is_worse_than_random_understanding_visual_token_information_i.md)**

:   发现VLLM深层中现有token剪枝方法不如随机剪枝的现象，提出基于输出概率变化量化视觉token信息的方法，揭示了"信息地平线"——视觉token信息在某层均匀消散至零的临界层，其位置受任务视觉复杂度和模型能力动态影响，并证明简单集成随机剪枝能有效提升现有方法。

**[Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation](where_mllms_attend_and_what_they_rely_on_explaining_autoregressive_token_generat.md)**

:   提出Eagle，一个轻量级黑盒归因框架，通过insight score（充分性）和necessity score（不可或缺性）的统一目标函数对MLLM的自回归token生成进行空间归因，并量化每个token依赖语言先验还是感知证据，在忠实度/定位/幻觉诊断上全面超越现有方法且GPU显存需求大幅降低。

**[Widget2Code: From Visual Widgets to UI Code via Multimodal LLMs](widget2code_from_visual_widgets_to_ui_code_via_multimodal_llms.md)**

:   首次形式化 Widget-to-Code 任务，构建了首个纯图像 widget 数据集和多维评估体系，提出基于感知代理和 WidgetFactory 基础设施的模块化基线，通过组件分解、图标检索、可复用可视化模板和自适应渲染实现高保真 widget 重建。

**[Zina: Multimodal Fine-grained Hallucination Detection and Editing](zina_multimodal_fine-grained_hallucination_detection_and_editing.md)**

:   Zina 提出了多模态细粒度幻觉检测与编辑任务，设计了两阶段系统（detector MLLM + reviewer MLLM）将 token 复制委托给确定性函数以简化模型负担，同时构建了 VisionHall 数据集（6.9K 人工标注 + 20K 图结构合成数据），在检测 F1 上超过 GPT-4o 达 15.8 个点。
