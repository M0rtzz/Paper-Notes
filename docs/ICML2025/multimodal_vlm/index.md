---
title: >-
  ICML2025 多模态VLM方向 50篇论文解读
description: >-
  50篇ICML2025 多模态VLM方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧩 多模态VLM

**🧪 ICML2025** · 共 **50** 篇

**[Bring Reason To Vision Understanding Perception And Reasoning Through Model Merg](bring_reason_to_vision_understanding_perception_and_reasoning_through_model_merg.md)**

:   通过将数学推理 LLM 的参数与 VLM 的文本部分直接加权平均（模型融合），在无需训练的情况下将推理能力迁移到 VLM，并发现感知能力集中在前层、推理能力集中在中后层的层级分布规律。

**[Cocoa-Mix Confusion-And-Confidence-Aware Mixture Model For Context Optimization](cocoa-mix_confusion-and-confidence-aware_mixture_model_for_context_optimization.md)**

:   提出 CoCoA-Mix 框架，通过混淆感知损失 (CoA-loss) 和置信度感知权重 (CoA-weights) 构建提示混合模型，在不引入额外网络参数的情况下同时提升 VLM prompt tuning 的专精性 (specialization) 和泛化性 (generalization)。

**[Comemo Lvlms Need Image Context With Image Memory](comemo_lvlms_need_image_context_with_image_memory.md)**

:   提出CoMemo双路径架构：Context路径保持自回归处理图像上下文，Memory路径用交叉注意力作为图像记忆避免"中间遗失"问题，配合RoPE-DHR位置编码保持2D空间感知，在7类基准上全面超越单路径LVLM。

**[Core Knowledge Deficits In Multi-Modal Language Models](core_knowledge_deficits_in_multi-modal_language_models.md)**

:   提出 CoreCognition 基准（12种核心认知能力、1503题），大规模评测230个MLLM后发现：模型在基础认知能力上系统性落后于人类，且随规模增大并未改善，而是更依赖捷径学习而非真正理解。

**[Corematching A Co-Adaptive Sparse Inference Framework With Token And Neuron Prun](corematching_a_co-adaptive_sparse_inference_framework_with_token_and_neuron_prun.md)**

:   首次揭示 VLM 中 token 稀疏与神经元稀疏之间的内在关联——核心神经元与核心 token 相互决定、相互强化，并据此提出 CoreMatching 协同稀疏推理框架，在 pre-filling 和 decoding 两阶段同时实现加速，达到 5× FLOPs 降低和 10× 整体加速。

**[Defending Lvlms Against Vision Attacks Through Partial-Perception Supervision](defending_lvlms_against_vision_attacks_through_partial-perception_supervision.md)**

:   提出 DPS（Defense through Partial-Perception Supervision），利用裁剪图像的响应作为"弱监督"来引导全图模型在推理时自我修正，实现无需训练的黑盒 LVLM 视觉攻击防御，平均攻击成功率降低 76.3%。

**[Do Vision-Language Models Really Understand Visual Language](do_vision-language_models_really_understand_visual_language.md)**

:   本文通过构建综合测试套件（含合成与真实图表）系统评估了大型视觉语言模型（LVLMs）的图表理解能力，发现模型虽可识别实体但对关系理解极为有限，其看似出色的图表推理表现实际源于利用背景知识作为捷径。

**[Dynamic Mixture Of Curriculum Lora Experts For Continual Multimodal Instruction ](dynamic_mixture_of_curriculum_lora_experts_for_continual_multimodal_instruction_.md)**

:   本文提出 D-MoLE 方法，通过动态层级 LoRA 专家分配器和基于梯度的跨模态持续课程策略，在参数预算约束下自动演化 MLLM 架构以持续适配新任务，相比最优基线平均提升 15%。

**[Efficient Quantification Of Multimodal Interaction At Sample Level](efficient_quantification_of_multimodal_interaction_at_sample_level.md)**

:   提出 LSMI（Lightweight Sample-wise Multimodal Interaction）估计器，首次实现了对真实世界连续分布数据的**逐样本级别**多模态交互（冗余、唯一性、协同）精确且高效的量化，并展示了其在数据分区、知识蒸馏和模型集成中的实用价值。

**[Elemental Interactive Learning From Demonstrations And Vision-Language Models Fo](elemental_interactive_learning_from_demonstrations_and_vision-language_models_fo.md)**

:   ELEMENTAL 将视觉语言模型 (VLM) 与逆强化学习 (IRL) 融合，通过 VLM 提取特征函数 + IRL 优化权重 + 自我反思迭代改进，在 IsaacGym 9 个任务上比 EUREKA 提升 42.3%。

**[Enhancing Rating-Based Reinforcement Learning To Effectively Leverage Feedback F](enhancing_rating-based_reinforcement_learning_to_effectively_leverage_feedback_f.md)**

:   提出 ERL-VLM，用大型视觉语言模型（VLM）对单条轨迹做绝对评分（rating）而非成对比较（preference），结合分层采样和 MAE 损失解决数据不平衡与噪声标签问题，显著提升 VLM 反馈驱动的奖励函数学习效果。

**[Enhancing Target-Unspecific Tasks Through A Features Matrix](enhancing_target-unspecific_tasks_through_a_features_matrix.md)**

:   提出 Features Matrix (FM) 方法，利用多个手工 prompt 模板从冻结 CLIP 中提取通用知识构成特征矩阵，通过对齐 unexpected features 与微调视觉特征来增强模型在目标无关任务（如 base-to-novel 泛化、跨数据集泛化、域泛化）上的表现。

**[Exlm Rethinking The Impact Of Mask Tokens In Masked Language Models](exlm_rethinking_the_impact_of_mask_tokens_in_masked_language_models.md)**

:   本文首次系统分析了 MLM 中 [MASK] 对性能的影响，发现**语义损坏（corrupted semantics）**比**非真实token（unreal tokens）**的负面作用更大，据此提出 ExLM：通过将每个 [MASK] 扩展为多个隐状态并用转移矩阵建模依赖关系，有效缓解语义多模态性问题，在文本和分子建模任务上均取得显著提升。

**[From Black Boxes To Transparent Minds Evaluating And Enhancing The Theory Of Min](from_black_boxes_to_transparent_minds_evaluating_and_enhancing_the_theory_of_min.md)**

:   本文从可解释性角度评估多模态大模型（MLLM）的心智理论（ToM）能力，构建了基于 2D 网格世界的多模态 ToM 数据集 GridToM，并提出一种无需训练的注意力头激活干预方法来显著提升模型的 ToM 表现。

**[Graph4Mm Weaving Multimodal Learning With Structural Information](graph4mm_weaving_multimodal_learning_with_structural_information.md)**

:   提出 Graph4MM 框架，通过 Hop-Diffused Attention 将多跳图结构信息注入自注意力机制，并设计 MM-QFormer 实现跨模态融合，在生成和判别任务上平均提升 6.93%。

**[Handling Imbalanced Pseudolabels For Vision-Language Models With Concept Alignme](handling_imbalanced_pseudolabels_for_vision-language_models_with_concept_alignme.md)**

:   提出 CAP 框架，通过**概念对齐**（检测并修复 concept mismatch）和**混淆感知校准边距**（缓解 concept confusion），解决 VLM 生成伪标签时的类别不平衡问题，在六个数据集三种范式下相对 SOTA 提升 6.29%。

**[Importance Corrected Neural Jko Sampling](importance_corrected_neural_jko_sampling.md)**

:   提出 Importance Corrected Neural JKO Sampling (Neural JKO IC)，将连续归一化流（CNF）的局部 JKO 步与基于重要性权重的拒绝重采样步交替使用，克服 Wasserstein 梯度流在多模态分布上的局部最优问题，同时保持独立同分布采样和密度可评估性。

**[Kernel-Based Unsupervised Embedding Alignment For Enhanced Visual Representation](kernel-based_unsupervised_embedding_alignment_for_enhanced_visual_representation.md)**

:   提出基于核函数的无监督嵌入对齐方法（KUEA），通过在核空间中对齐 CLIP 与 DINOv2 的视觉表示，仅用图像数据微调即可增强 CLIP 的细粒度感知能力，同时保持与文本编码器的兼容性，提升下游 MLLM 性能。

**[Lada Scalable Label-Specific Clip Adapter For Continual Learning](lada_scalable_label-specific_clip_adapter_for_continual_learning.md)**

:   提出 LADA（Label-specific ADApter），通过在冻结 CLIP 图像编码器后追加轻量级的**类别特定记忆向量**，将所有已学任务的判别信息浓缩到统一特征空间，**彻底消除推理阶段的参数选择步骤**，在 X-TAIL 持续学习设定下取得 SOTA。

**[Laion-C An Out-Of-Distribution Benchmark For Web-Scale Vision Models](laion-c_an_out-of-distribution_benchmark_for_web-scale_vision_models.md)**

:   提出LAION-C基准——包含6种专为web-scale数据集设计的新型图像损坏类型（对LAION等大规模数据集truly OOD），发现当代模型在LAION-C上面临显著挑战，且范式转变已发生：最好的模型现在匹配或超越人类观察者。

**[Learning Invariant Causal Mechanism From Vision-Language Models](learning_invariant_causal_mechanism_from_vision-language_models.md)**

:   通过因果分析证明 CLIP 嵌入是真实不变/可变因子的线性变换，提出 CLIP-ICM 框架利用干预数据估计线性投影矩阵，将预测限定在不变子空间中以实现跨环境一致预测。

**[Learning Optimal Multimodal Information Bottleneck Representations](learning_optimal_multimodal_information_bottleneck_representations.md)**

:   提出 OMIB 框架，通过理论推导正则化参数 β 的上界并动态调整各模态权重 r，保证多模态信息瓶颈表示的最优性（包含全部任务相关信息、排除冗余信息）。

**[Lemon Label Error Detection Using Multimodal Neighbors](lemon_label_error_detection_using_multimodal_neighbors.md)**

:   本文提出 LEMoN 方法，利用对比预训练多模态模型（如 CLIP）的嵌入空间中图像-文本对的多模态邻域结构，在分类和图像描述两个场景下自动检测标签错误，在训练无关的基线中 F1 提升 3-4%，过滤后的数据可改善下游分类和描述性能。

**[Look Twice Before You Answer Memory-Space Visual Retracing For Hallucination Mit](look_twice_before_you_answer_memory-space_visual_retracing_for_hallucination_mit.md)**

:   提出 MemVR 解码范式，将视觉 token 作为补充证据通过 FFN 的 key-value memory 机制重新注入到中间触发层，以"再看一次"的方式缓解 MLLM 幻觉问题，不引入额外推理开销。

**[M3-Jepa Multimodal Alignment Via Multi-Gate Moe Based On The Joint-Embedding Pre](m3-jepa_multimodal_alignment_via_multi-gate_moe_based_on_the_joint-embedding_pre.md)**

:   将JEPA(联合嵌入预测架构)推广到任意模态组合的多模态对齐中，用Multi-gate MoE作为跨模态预测器在潜在空间对齐，通过交替梯度下降解决不同模态任务间的梯度冲突，在视觉-语言/音频-语言等任务上达SOTA。

**[Mminference Accelerating Pre-Filling For Long-Context Vlms Via Modality-Aware Pe](mminference_accelerating_pre-filling_for_long-context_vlms_via_modality-aware_pe.md)**

:   本文提出 MMInference，通过“模态感知的置换稀疏注意力 + 头级离线模式搜索 + 在线动态索引 + 定制 GPU Kernel”，在不改模型不微调的前提下，将长上下文 VLM 的 prefill 阶段在 1M token 场景最高加速到 8.3x，同时尽量保持任务精度。

**[Moda Modular Duplex Attention For Multimodal Perception Cognition And Emotion Un](moda_modular_duplex_attention_for_multimodal_perception_cognition_and_emotion_un.md)**

:   针对多模态大语言模型中跨模态注意力不一致与逐层衰减的"注意力缺失障碍"问题，提出模块化双工注意力机制MODA，通过将注意力解耦为模态内自精炼与模态间交互两路，并借助Duplex Aligner和自适应掩码注意力实现"先对齐再校正"的策略，在21个感知、认知与情感基准上验证了有效性。

**[Omnibal Towards Fast Instruction-Tuning For Vision-Language Models Via Omniverse](omnibal_towards_fast_instruction-tuning_for_vision-language_models_via_omniverse.md)**

:   针对大规模视觉语言模型 instruction-tuning 训练中因数据和模型异构性导致的计算不平衡问题，提出 OmniBal 框架从数据、模型、内存三个层面系统性平衡跨设备计算负载，在 InternVL-Chat 上实现约 1.8× 训练加速。

**[Overcoming Multi-Step Complexity In Multimodal Theory-Of-Mind Reasoning](overcoming_multi-step_complexity_in_multimodal_theory-of-mind_reasoning.md)**

:   提出可扩展的贝叶斯心智理论规划器，通过逐步贝叶斯更新分解多步ToM推理复杂度，并用弱到强控制机制将小模型的ToM专业能力迁移到大模型(405B)，在多模态ToM基准上超越SOTA 4.6%。

**[Overcoming Multi-Step Complexity In Multimodal Theory-Of-Mind Reasoning A Scalab](overcoming_multi-step_complexity_in_multimodal_theory-of-mind_reasoning_a_scalab.md)**

:   提出一种可扩展的贝叶斯心智理论（ToM）规划器，通过将多步推理分解为逐步贝叶斯更新，并利用弱到强控制机制将小模型的 ToM 专项能力迁移至大模型（最高 405B），在多模态 ToM 基准上超越 SOTA 4.6%。

**[Parrot Multilingual Visual Instruction Tuning](parrot_multilingual_visual_instruction_tuning.md)**

:   提出 Parrot，通过文本引导的跨注意力机制和 MoE 模块将英语偏置的视觉特征转换为语言特定表示，以极少量多语言数据（每种语言约 10K 样本）显著提升 MLLM 的多语言能力。

**[Ranked From Within Ranking Large Multimodal Models Without Labels](ranked_from_within_ranking_large_multimodal_models_without_labels.md)**

:   系统研究能否在无标签场景下预测 LMM 的相对性能，评估 47 个 SOTA LMM 在 9 个 VQA 基准上的表现，发现基于 softmax 分布的不确定性指标能提供稳健的无监督模型排名（与真实排名 Spearman 相关 $\rho=0.92$）。

**[Re-Ranking Reasoning Context With Tree Search Makes Large Vision-Language Models](re-ranking_reasoning_context_with_tree_search_makes_large_vision-language_models.md)**

:   提出 RCTS 框架，通过自一致性评估机制构建推理上下文丰富的知识库，并用带启发式奖励的蒙特卡罗树搜索（MCTS-HR）重排检索示例，使 LVLM 在多个 VQA 数据集上显著超越 ICL 和 Vanilla-RAG 方法（平均 +3-4%）。

**[Reasoning Limitations Of Multimodal Large Language Models A Case Study Of Bongar](reasoning_limitations_of_multimodal_large_language_models_a_case_study_of_bongar.md)**

:   系统评估了 8 个 MLLM 在 Bongard Problems 上的抽象视觉推理能力，并引入 Bongard-RWR 数据集（合成概念的真实图像版本），揭示 MLLM 在合成 BP 上表现极差并非因领域差异，而是其固有的抽象推理局限。

**[Robust Multimodal Large Language Models Against Modality Conflict](robust_multimodal_large_language_models_against_modality_conflict.md)**

:   揭示 MLLM 幻觉的一个被忽视来源——模态冲突（视觉输入与文本输入之间的固有矛盾），从对象/属性/关系三个层面形式化定义模态冲突，构建 20K 样例的 MMMC 数据集，并提出 prompt engineering、SFT 和 RL 三种缓解方法，其中 RL 效果最佳。

**[Rollingq Reviving The Cooperation Dynamics In Multimodal Transformer](rollingq_reviving_the_cooperation_dynamics_in_multimodal_transformer.md)**

:   揭示多模态 Transformer 中自注意力机制因"自增强循环"导致动态适应性失效（偏向单一模态），并提出 RollingQ 算法通过旋转 query 向量打破这一循环，恢复跨模态协作动态。

**[Sk-Vqa Synthetic Knowledge Generation At Scale For Training Context-Augmented Mu](sk-vqa_synthetic_knowledge_generation_at_scale_for_training_context-augmented_mu.md)**

:   利用 GPT-4 全自动生成包含 200 万+ QA 对的大规模合成 KB-VQA 数据集 SK-VQA，训练 MLLM 适配上下文增强生成，在跨域泛化性能上显著优于已有数据集。

**[Slimllm Accurate Structured Pruning For Large Language Models](slimllm_accurate_structured_pruning_for_large_language_models.md)**

:   提出SlimLLM——LLM结构化剪枝方法：用特征空间重要性（考虑权重方向和幅度）评估通道，用Pearson相似度整体评估注意力头，配合简单线性回归恢复策略和层级剪枝比例分配，在LLaMA上20%剪枝保留98.7%性能。

**[Sparsevlm Visual Token Sparsification For Efficient Vision-Language Model Infere](sparsevlm_visual_token_sparsification_for_efficient_vision-language_model_infere.md)**

:   提出SparseVLM——无训练的文本引导视觉token优化：用自注意力矩阵中文本token对视觉token的评分来确定重要性，自适应确定每层稀疏化比例，并用token回收压缩被剪token为紧凑表示。

**[Targeted Unlearning With Single Layer Unlearning Gradient](targeted_unlearning_with_single_layer_unlearning_gradient.md)**

:   提出 SLUG (Single Layer Unlearning Gradient) 方法，通过层重要性和梯度对齐指标识别最优单层，仅需一次梯度计算和单层参数更新即可实现高效精准的定向遗忘，可应用于 CLIP、Stable Diffusion 和 VLM。

**[The Devil Is In The Details Tackling Unimodal Spurious Correlations For Generali](the_devil_is_in_the_details_tackling_unimodal_spurious_correlations_for_generali.md)**

:   发现多模态奖励模型 (MM-RM) 在训练时会过度依赖文本单模态捷径 (shortcuts)，导致分布外泛化能力差，提出 Shortcut-aware MM-RM 学习算法通过动态样本重加权来减少对单模态伪相关性的依赖，OOD 准确率从 68.1% 提升至 78.5%。

**[Toward Robust Hyper-Detailed Image Captioning A Multiagent Approach And Dual Eva](toward_robust_hyper-detailed_image_captioning_a_multiagent_approach_and_dual_eva.md)**

:   提出 CapMAS 多智能体系统，通过 LLM-MLLM 协作将详细图文描述分解为原子命题并逐一验证真实性来纠正幻觉，同时引入从事实性和覆盖度两个维度评估详细描述的框架，显著提升了包括 GPT-4V 在内的多种 MLLM 的描述质量。

**[Towards Efficient Online Tuning Of Vlm Agents Via Counterfactual Soft Reinforcem](towards_efficient_online_tuning_of_vlm_agents_via_counterfactual_soft_reinforcem.md)**

:   提出 Counterfactual Soft Reinforcement Learning (CoSo)，利用反事实推理评估每个 token 对最终动作的因果影响，通过因果加权熵正则优化集中探索关键 token，解决 VLM 智能体在线 RL 微调中文本动作空间爆炸问题，在 Android 控制、卡牌游戏、具身 AI 上分别提升 12.3%、9.3%、16.7%。

**[Towards Rationale-Answer Alignment Of Lvlms Via Self-Rationale Calibration](towards_rationale-answer_alignment_of_lvlms_via_self-rationale_calibration.md)**

:   提出 Self-Rationale Calibration (SRC) 框架，通过轻量级 rationale 微调引导 LVLM 输出推理过程，再利用句子级 beam search 生成多样候选响应，结合专门设计的 R-Scorer 配对评分策略筛选优劣 rationale-answer 对，以 DPO 偏好对齐方式迭代校准模型的推理-答案一致性，在感知、推理和泛化多个基准上取得显著提升。

**[Understanding And Mitigating Miscalibration In Prompt Tuning For Vision-Language](understanding_and_mitigating_miscalibration_in_prompt_tuning_for_vision-language.md)**

:   揭示 CLIP prompt tuning 在 base 和 novel 类之间存在校准权衡（CoOp 导致 novel 类过度自信，KgCoOp 导致 base 类自信不足），从文本特征散度视角解释原因，并提出 Dynamic Outlier Regularization (DOR) 通过正则化非训练类文本标签的特征偏差同时保持两端校准。

**[Universal Retrieval For Multimodal Trajectory Modeling](universal_retrieval_for_multimodal_trajectory_modeling.md)**

:   首次提出多模态轨迹检索（Multimodal Trajectory Retrieval）任务，构建统一代理轨迹数据集 UATD 和 GAE-Bench 基准，并提出基于 VLM 的 GAE-Retriever 框架，在轨迹级检索上显著优于基线。

**[Unlocking The Capabilities Of Large Vision-Language Models For Generalizable And](unlocking_the_capabilities_of_large_vision-language_models_for_generalizable_and.md)**

:   提出基于 LVLM 的 deepfake 检测框架，通过知识引导伪造检测器（KFD）计算图像特征与真/假描述文本的相关性实现分类和定位，再通过伪造提示学习器（FPL）将细粒度伪造特征注入 LLM 生成可解释的检测结果，在 FF++/CDF2/DFDC/DF40 等多个基准上超越 SOTA 泛化性能。

**[Vision-Language Model Selection And Reuse For Downstream Adaptation](vision-language_model_selection_and_reuse_for_downstream_adaptation.md)**

:   提出 Model Label Learning (MLL) 范式，通过构建语义图对 49 个预训练 VLM 进行离线"标注"（描述各模型在不同视觉概念上的能力），面对新任务时通过语义匹配选择和集成最合适的模型，实现数据高效、计算高效且可扩展的 VLM 选择与复用。

**[Vision-Language Models Create Cross-Modal Task Representations](vision-language_models_create_cross-modal_task_representations.md)**

:   本文发现自回归视觉语言模型（VLMs）会将概念上等价的输入（不论是文本还是图像示例、指令还是少样本）压缩为共享的"任务向量"，并通过跨模态 patching 实验验证了这种表征对齐的存在和实用性。

**[Vision Graph Prompting Via Semantic Low-Rank Decomposition](vision_graph_prompting_via_semantic_low-rank_decomposition.md)**

:   提出 Vision Graph Prompting (VGP)，首个面向 Vision GNN (ViG) 的视觉提示学习框架，利用图中语义连通分量的低秩特性，设计了图/边/节点三层粒度的语义低秩提示（SeLo-Graph/Edge/Node Prompt），在参数高效的前提下达到接近全量微调的下游任务迁移性能。
