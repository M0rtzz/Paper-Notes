---
title: >-
  CVPR2026 LLM 安全方向29篇论文解读
description: >-
  29篇CVPR2026的 LLM 安全方向论文解读，涵盖多模态、对抗鲁棒、人脸/视线、持续学习、LLM、域适应等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2026"
  - "LLM 安全"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "对抗鲁棒"
  - "人脸/视线"
  - "持续学习"
  - "LLM"
  - "域适应"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM 安全

**📷 CVPR2026** · **29** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (18)](../../ICML2026/llm_safety/index.md) · [💬 ACL2026 (115)](../../ACL2026/llm_safety/index.md) · [🔬 ICLR2026 (54)](../../ICLR2026/llm_safety/index.md) · [🤖 AAAI2026 (43)](../../AAAI2026/llm_safety/index.md) · [🧠 NeurIPS2025 (84)](../../NeurIPS2025/llm_safety/index.md) · [📹 ICCV2025 (13)](../../ICCV2025/llm_safety/index.md)

🔥 **高频主题：** 多模态 ×11 · 对抗鲁棒 ×6

**[A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](a_closedform_solution_for_debiasing_visionlanguage.md)**

:   提出VLM去偏的闭式解方法，通过在跨模态嵌入空间中对属性子空间做正交分解并利用Chebyshev标量化求解，实现Pareto最优公平性与有界效用损失，免训练、免标注，统一覆盖零样本分类、文本-图像检索和文本-图像生成三大下游任务。

**[Association and Consolidation: Evolutionary Memory-Enhanced Incremental Multi-View Clustering](association_and_consolidation_evolutionary_memory-enhanced_incremental_multi-vie.md)**

:   提出 EMIMC 框架，受大脑海马-前额叶协作记忆机制启发，通过 Rapid Associative Module (正交映射保证可塑性)、Cognitive Forgetting Module (幂律衰减模拟遗忘曲线) 和 Knowledge Consolidation Module (时序张量低秩分解提炼长期记忆) 三模块协同，解决增量多视图聚类中的稳定性-可塑性困境。

**[Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations](beyond_global_scores_fine_grained_token_grounding_as_robust_detector_of_lvlm_hallucinations.md)**

:   提出基于 patch 级别的 LVLM 幻觉检测框架，发现幻觉 token 表现出弥散注意力模式和低语义对齐两个特征标志，据此设计注意力弥散分数（ADS）和跨模态接地一致性（CGC）两个轻量指标，检测准确率达 90%。

**[The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models](blind_spot_of_adaptation_quantifying_and_mitigating_forgetting_in_fine_tuned_driving_models.md)**

:   系统研究 VLM 微调到自动驾驶场景时的灾难性遗忘问题，构建 180K 场景大规模基准 FidelityDrivingBench，并提出 Drive Expert Adapter (DEA) 通过提示空间路由在不腐蚀基础参数的前提下增强驾驶任务性能。

**[DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions](damp_class_unlearning_via_depth_aware_removal_of_forget_specific_directions.md)**

:   提出 DAMP（深度感知投影调制），一种一次性闭式权重手术方法用于类遗忘，通过在每个网络阶段的编辑空间中移除遗忘类特有方向来实现选择性遗忘，深度感知缩放规则确保浅层保守编辑、深层强力编辑。

**[Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification](demographic_fairness_in_multimodal_llms_a_benchmark_of_gender_and_ethnicity_bias.md)**

:   首次系统性地评估了 9 个开源 MLLM 在人脸验证任务上的人口统计公平性，在 IJB-C 和 RFW 两个 benchmark 上使用 4 种 FMR-based 公平性指标衡量性别和族裔偏差，发现 MLLM 的偏见模式与传统人脸识别系统不同。

**[Designing to Forget: Deep Semi-parametric Models for Unlearning](designing_to_forget_deep_semi-parametric_models_for_unlearning.md)**

:   提出"Designing to Forget"理念，设计了一族深度半参数模型 (SPM)，在推理时通过简单删除训练样本即可实现遗忘（无需修改模型参数），在 ImageNet 分类上将与重训基线的预测差距减少 11%，遗忘速度提升 10 倍以上。

**[Elastic Weight Consolidation Done Right for Continual Learning](elastic_weight_consolidation_done_right_for_continual_learning.md)**

:   本文从梯度视角系统分析了 EWC 及其变体在权重重要性估计上的根本缺陷（EWC 的梯度消失和 MAS 的冗余保护），并提出了一个极其简单的 Logits Reversal 操作来修正 Fisher 信息矩阵的计算，在无样例类增量学习和多模态持续指令微调任务上大幅超越原始 EWC 及其所有变体。

**[FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)**

:   提出 FairLLaVA，一种参数高效的公平性微调方法，通过最小化隐藏状态与人口学属性之间的互信息来消除多模态大语言模型中的人口学捷径，在胸片报告生成和皮肤病变问答中显著缩小了群体间性能差距。

**[HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs](hulluedit_subspace_editing_hallucination.md)**

:   提出HulluEdit，一个单次推理、无参考模型的幻觉缓解框架，通过将隐藏状态正交分解为视觉证据子空间、冲突先验子空间和残差不确定性子空间，选择性抑制幻觉模式而不干扰视觉接地，在POPE和CHAIR上达到SOTA。

**[IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding](iag_input-aware_backdoor_attack_on_vlm-based_visual_grounding.md)**

:   提出IAG，首个针对VLM视觉定位的多目标后门攻击方法，通过文本条件U-Net动态生成输入感知触发器，将任意指定目标物体的语义信息嵌入视觉输入中，在12种设置下的11种达到最高攻击成功率。

**[Interpretable Debiasing of Vision-Language Models for Social Fairness](interpretable_debiasing_of_vision-language_models_for_social_fairness.md)**

:   提出 DeBiasLens，通过在 VLM 编码器上训练稀疏自编码器（SAE）来定位编码社会属性的"社会神经元"，然后在推理时选择性去激活这些神经元以缓解偏见，在 CLIP 上降低 Max Skew 9-16%，在 InternVL2 上降低性别偏差比例 40-50%，同时保持通用性能。

**[Learning from Oblivion: Predicting Knowledge-Overflowed Weights via Retrodiction of Forgetting](learning_from_oblivion_predicting_knowledge_overflowed_weights_via_retrodiction_.md)**

:   提出KNOW prediction：通过在逐步缩小的数据子集上sequential fine-tuning诱导结构化遗忘过程，收集权重转变轨迹，然后用meta-learned hyper-model（KNOWN）反转forgetting方向，预测"仿佛在更大数据集上训练"的虚拟知识增强权重。跨多数据集(CIFAR/ImageNet/PACS等)和多架构(ResNet/PVTv2/DeepLabV3+)持续超越naive fine-tuning及多种weight prediction基线，在图像分类、语义分割、图像描述、域泛化等下游任务上均有显著提升。

**[Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)**

:   提出 MPCAttack 框架，联合跨模态对齐、多模态理解和视觉自监督三种学习范式的特征表示，通过多范式协同优化策略生成高迁移性对抗样本，在开源和闭源 MLLM 上均取得 SOTA 攻击效果。

**[⊘ Source Models Leak What They Shouldn't ↛: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization](oslash_source_models_leak_what_they_shouldnt_nrightarrow_unlearning_zero-shot_tr.md)**

:   发现无源域自适应（SFDA）方法会不经意地将源域独有类别的知识泄漏到目标域（零样本迁移现象），提出 SCADA-UL 框架通过对抗生成遗忘样本和重缩放标签策略，在域自适应过程中同时完成类别遗忘，达到接近从头训练的遗忘效果。

**[Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP](perturb_and_recover_fine-tuning_for_effective_backdoor_removal_from_clip.md)**

:   本文提出 PAR（Perturb and Recover），一种简单而有效的 CLIP 模型后门清洗方法：通过显式地将模型embedding推离中毒状态（Perturb），同时用标准 CLIP 损失恢复干净性能（Recover），在不依赖强数据增强的情况下实现对任意触发器的鲁棒后门移除，甚至仅用合成数据即可有效清洗。

**[Phantasia: Context-Adaptive Backdoors in Vision Language Models](phantasia_context-adaptive_backdoors_in_vision_language_models.md)**

:   Phantasia 首次提出上下文自适应的 VLM 后门攻击——攻击者预设一个目标问题，中毒模型在接收到触发图片后不再回答用户原始问题，而是回答攻击者的目标问题，且生成的答案与输入图像语义一致、在语言上自然流畅，从而绕过 STRIP-P 和 ONION-R 等防御；同时本文首次证明了现有 VLM 后门攻击的隐蔽性被严重高估。

**[PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing](pinpoint_evaluation_of_composed_image_retrieval_with_explicit_negatives_multi-im.md)**

:   提出 PinPoint 基准，包含 7,635 个查询和 329K 人工验证的相关性判断，通过显式负样本、多图像查询、释义变体和人口统计元数据四个维度，揭示了现有 CIR 方法在假阳性抑制、语言鲁棒性和多图像推理上的严重缺陷，并提出基于 MLLM 的无训练重排方法作为改进基线。

**[Pixels Don't Lie (But Your Detector Might): Bootstrapping MLLM-as-a-Judge for Trustworthy Deepfake Detection and Reasoning Supervision](pixels_dont_lie_but_your_detector_might_bootstrapping_mllm-as-a-judge_for_trustw.md)**

:   提出 DeepfakeJudge 框架，通过 bootstrapped generator-evaluator 流程将人类标注的推理监督扩展为大规模结构化评分数据，训练出 3B/7B 视觉语言模型作为 deepfake 检测推理质量的自动评判者，在 pointwise 和 pairwise 评估上均达到与人类高度一致的水平。

**[RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models](razor_ratio-aware_layer_editing_for_targeted_unlearning_in_vision_transformers_a.md)**

:   提出 RAZOR，一种基于比率感知的多层/多头选择性编辑框架，可在 CLIP、Stable Diffusion 和 VLM 等 Transformer 视觉模型中高效精准地完成目标遗忘，同时保持模型整体性能与量化鲁棒性。

**[Select, Hypothesize and Verify: Towards Verified Neuron Concept Interpretation](select_hypothesize_and_verify_towards_verified_neuron_concept_interpretation.md)**

:   提出 SIEVE（Select–Hypothesize–Verify）框架，通过筛选高激活样本、生成概念假设、再用文生图验证的闭环流程来解释神经元功能，生成的概念激活对应神经元的概率约为现有 SOTA 的 1.5 倍。

**[SineProject: Machine Unlearning for Stable Vision–Language Alignment](sineproject_machine_unlearning_for_stable_vision_language_alignment.md)**

:   针对多模态大模型（MLLM）在机器遗忘过程中投影层 Jacobian 严重病态导致视觉-语言对齐漂移的问题，提出 SineProject——通过对投影层权重施加正弦调制（sin(ΔW)）来约束参数范围至 [-1,1]，从而将 Jacobian 条件数降低 3-4 个数量级，在完全遗忘目标知识的同时将良性查询误拒率（SARR）降低 15%。

**[Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)**

:   发现LVLM后门行为的本质是跨模态注意力窃取（trigger视觉token抢夺文本token的注意力），提出CleanSight——首个无需训练的测试时后门防御框架，通过检测和剪枝高注意力trigger token来消除后门效应。

**[TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection](tridf_evaluating_perception_detection_and_hallucination_for_interpretable_deepfa.md)**

:   提出TriDF——首个从感知 (Perception)、检测 (Detection) 和幻觉 (Hallucination) 三个维度综合评估可解释深度伪造检测的基准，包含55K高质量样本覆盖16种DeepFake类型和3种模态，揭示了准确感知是可靠检测的基础但幻觉会严重破坏决策的三方耦合关系。

**[TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition](trivia_self-supervised_fine-tuning_of_vision-language_models_for_table_recogniti.md)**

:   提出 TRivia 自监督微调框架，通过表格问答（QA）驱动的 GRPO 强化学习，让 VLM 直接从无标注表格图像中学习表格识别能力，3B 参数的 TRivia-3B 在多个基准上超越 Gemini 2.5 Pro 和 GPT-5 等私有模型。

**[Unsafe2Safe: Controllable Image Anonymization for Downstream Utility](unsafe2safe_controllable_image_anonymization_for_downstream_utility.md)**

:   本文提出 Unsafe2Safe 全自动隐私保护流水线，通过 VLM 隐私检查→双字幕生成（私有/公开）→LLM 编辑指令→文本引导扩散编辑的四阶段方案，实现可控图像匿名化，在 VLMScore 隐私指标大幅提升的同时，在 Caltech-101 分类和 OK-VQA 上匿名后准确率甚至超过原始图像。

**[V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs](v-attack_targeting_disentangled_value_features_for_controllable_adversarial_atta.md)**

:   发现 ViT 中 Value 特征相比 Patch 特征具有更解耦的局部语义表示，提出 V-Attack 通过自增强 Value 特征 + 文本引导语义操控实现精确可控的 LVLM 局部语义攻击，ASR 平均提升 36%。

**[Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks](vlm_model_inversion_adaptive_token_weight.md)**

:   首次系统研究 VLM 的模型反转（Model Inversion）攻击，提出一套面向 token 生成特性的反转策略（TMI/TMI-C/SMI），以及基于视觉注意力强度动态加权 token 梯度贡献的 SMI-AW 方法，在 4 种 VLM 和 3 个数据集上实现最高 61.21% 的人类评估攻击准确率，揭示了 VLM 严重的训练数据隐私泄露风险。

**[Which Concepts to Forget and How to Refuse? Decomposing Concepts for Continual Unlearning in Large Vision-Language Models](which_concepts_to_forget_and_how_to_refuse_decomposing_concepts_for_continual_un.md)**

:   本文提出CORE(COncept-aware REfuser)，一个面向大视觉语言模型(LVLM)持续遗忘的框架：通过将待删除的视觉-语言对分解为细粒度的视觉属性和文本意图概念，使用概念调制器识别需要拒绝的概念组合，再通过混合拒绝专家(refusers)生成概念对齐的拒绝回复，在16个连续遗忘任务上实现了90.67% CRR和88.02% AR的最佳遗忘-保留权衡。
