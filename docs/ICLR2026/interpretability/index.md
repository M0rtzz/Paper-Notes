---
title: >-
  ICLR2026 可解释性方向 56篇论文解读
description: >-
  56篇ICLR2026 可解释性方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 可解释性

**🔬 ICLR2026** · 共 **56** 篇

**[A Cortically Inspired Architecture For Modular Perceptual Ai](a_cortically_inspired_architecture_for_modular_perceptual_ai.md)**

:   从神经科学出发提出皮层启发的模块化感知 AI 架构蓝图，包含专用编码器、共享跨模态潜空间、路由控制器和递归预测反馈回路四个组件，并通过稀疏自编码器实验验证模块化分解可提升域内特征稳定性 (+15.4pp Jaccard 重叠)。

**[Activationreasoning Logical Reasoning In Latent Activation Spaces](activationreasoning_logical_reasoning_in_latent_activation_spaces.md)**

:   提出 ActivationReasoning (AR) 框架，在 LLM 的潜在激活空间（通过 SAE 提取的特征）上嵌入显式逻辑推理，通过三阶段流程（发现概念表征→检测激活命题→逻辑规则推理）实现多跳推理、概念组合和安全控制，在 PrOntoQA 上 8B 模型达到 95%+ 准确率超越 GPT-4o。

**[Auditing Cascading Risks In Multi-Agent Systems Via Semanti-Geometric Co-Evolut](auditing_cascading_risks_in_multi-agent_systems_via_semanti-geometric_co-evolut.md)**

:   提出 SCCAL 框架，通过耦合语义流（semantic flow）和交互图的 Ollivier–Ricci 曲率（ORC）来建模多智能体系统中语义-几何的协同演化，利用两者的一致性残差作为级联风险的早期预警信号，在语义违规显现前数轮即可检测异常。

**[Behavior Learning Bl Learning Hierarchical Optimization Structures From Data](behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)**

:   受行为科学中效用最大化范式启发，提出 Behavior Learning (BL) 框架，将数据建模为由可解释的模块化效用最大化问题（UMP）层次组合所诱导的 Gibbs 分布，在预测性能、内在可解释性和参数可辨识性三者之间实现了统一。

**[Beyond Linear Probes Dynamic Safety Monitoring For Language Models](beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)**

:   提出截断多项式分类器（TPC），通过对 LLM 激活空间中的多项式逐阶训练和截断评估，实现动态安全监控——在简单输入上用低阶（≈线性探针）快速决策，在困难输入上增加高阶项提供更强防护，在 WildGuardMix 和 BeaverTails 两个数据集上匹敌或超越 MLP 基线且具备内置可解释性。

**[Closing The Curvature Gap Full Transformer Hessians And Their Implications For S](closing_the_curvature_gap_full_transformer_hessians_and_their_implications_for_s.md)**

:   首次推导完整 Transformer block（含 LayerNorm 和 FFN）的显式 Hessian 表达式及谱范数上界，建立了损失面随数据量增加以 $O(1/k)$ 速率收敛的理论框架，为 scaling laws 和曲率感知训练提供了数学基础。

**[Concepts Information Bottleneck Models](concepts_information_bottleneck_models.md)**

:   在概念瓶颈模型(CBM)的概念层引入信息瓶颈(IB)正则化，通过惩罚 I(X;C) 同时保留 I(C;Y) 来学习最小充分概念表示，在六个CBM变体和三个基准上一致提升预测性能和概念干预可靠性。

**[Cross-Modal Redundancy And The Geometry Of Vision-Language Embeddings](cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)**

:   提出 Iso-Energy 假设（真正跨模态共享的概念在不同模态中应具有相同的平均激活能量），并设计 Aligned SAE 作为分析工具，揭示 VLM 嵌入空间中双模态原子承载跨模态对齐信号、单模态原子完全解释模态间隙的几何结构。

**[Decomposing Representation Space Into Interpretable Subspaces With Unsupervised ](decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)**

:   提出 NDM（Neighbor Distance Minimization），通过最小化子空间内的近邻距离来无监督地发现神经网络表征空间中的可解释非基对齐子空间，在 GPT-2 上平均 Gini=0.71（信息高度集中），在 Qwen2.5-1.5B 上发现了参数化知识与上下文知识路由的分离子空间。

**[Decoupling Dynamical Richness From Representation Learning Towards Practical Mea](decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)**

:   提出一种计算高效、与性能无关的动态丰富度度量 $\mathcal{D}_{LR}$，通过比较最后一层前后的激活来衡量 rich/lazy 训练动态，并证明 neural collapse 是该度量的特殊情况。

**[Dynamic Reflections Probing Video Representations With Text Alignment](dynamic_reflections_probing_video_representations_with_text_alignment.md)**

:   本文首次将柏拉图表示假说（Platonic Representation Hypothesis）扩展到时序领域，系统研究了视频-文本跨模态表示对齐，发现通过在测试时增加视频帧数和文本描述数量可以显著提升对齐分数（最高翻倍），并提出了具有强预测力的参数化缩放律。

**[Dynamic Reflections Probing Video Representations With Text Driven Reasoning](dynamic_reflections_probing_video_representations_with_text_driven_reasoning.md)**

:   首次将柏拉图表示假说（PRH）扩展到时序领域，系统研究视频-文本表示对齐，发现通过增加测试时的帧数和描述数量可以显著提升对齐分数（翻倍），并提出了精确的参数化测试时缩放定律。

**[Evolution Of Concepts In Language Model Pre-Training](evolution_of_concepts_in_language_model_pre-training.md)**

:   首次将 crosscoders（跨快照稀疏字典学习）应用于追踪语言模型预训练过程中特征的涌现和演化，发现预训练存在"统计学习→特征学习"两阶段相变，并通过归因分析将微观特征演化与宏观下游任务指标因果关联。

**[Exploring Interpretability For Visual Prompt Tuning With Cross-Layer Concepts](exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)**

:   提出IVPT（Interpretable Visual Prompt Tuning），通过跨层类别无关概念原型将抽象visual prompt关联到人类可理解的语义区域，在保持参数高效微调优势的同时，首次实现了visual prompt的可解释性，在CUB-200等细粒度分类基准上同时提升解释一致性（+8.4%）和准确率。

**[Expo-Hm Learning To Explain-Then-Detect For Hateful Meme Detection](expo-hm_learning_to_explain-then-detect_for_hateful_meme_detection.md)**

:   提出 ExPO-HM，受人类审核员培训流程启发，结合策略手册 SFT 预热、GRPO 课程学习和条件决策熵（CDE）奖励，首次实现 Explain-then-Detect 仇恨 Meme 检测在二分类、细粒度分类和推理质量上全面超越直接检测基线，F1 提升最高达 15-17%。

**[Formal Mechanistic Interpretability Automated Circuit Discovery With Provable Gu](formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)**

:   将神经网络验证（NN verification）引入机制可解释性，提出首个具有可证明保证的电路发现框架：在连续输入域上保证电路忠实度（input robustness）、在连续 patching 域上保证电路一致性（patching robustness），并形式化了四级最小性层次（quasi → local → subset → cardinal），通过单调性理论将三类保证统一连接。

**[Gavel Towards Rule-Based Safety Through Activation Monitoring](gavel_towards_rule-based_safety_through_activation_monitoring.md)**

:   提出 GAVEL 框架，将 LLM 安全从"粗粒度误用数据集训练分类器"范式转向"可组合认知元素 (CE) + 布尔规则"范式：定义可解释的激活级原语（如"发出威胁"、"处理支付"），组合为精确的策略规则，实现高精度、可定制、可审计的实时安全监控。

**[Gepa Reflective Prompt Evolution Can Outperform Reinforcement Learning](gepa_reflective_prompt_evolution_can_outperform_reinforcement_learning.md)**

:   提出 GEPA（Genetic-Pareto）提示优化器，通过自然语言反思从少量执行轨迹中诊断问题并迭代优化提示，在六个任务上平均超越 GRPO 6%（最高20%），同时仅使用 1/35 的采样量。

**[Grokking In Llm Pretraining Monitor Memorization-To-Generalization Without Test](grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)**

:   首次在实际规模 LLM（7B MoE）的近单遍预训练中验证 grokking 现象——不同数据组异步记忆、延迟泛化；通过分析 MoE routing pathway 的演化（从 instance-specific 到 structured/shared），提出两个零成本指标来监控泛化进度，无需 instruction tuning 和 benchmark 评估。

**[Hallucination Begins Where Saliency Drops](hallucination_begins_where_saliency_drops.md)**

:   提出 LVLMs-Saliency 梯度感知诊断框架来量化每个输出 token 的视觉锚定强度，发现"当先前输出 token 对下一个 token 预测的显著性降低时，幻觉就会产生"的关键规律，并基于此设计了 SGRS（显著性引导的拒绝采样）+ LocoRE（局部一致性增强）双机制推理时框架，在多个 LVLM 上显著降低幻觉率。

**[Hidden Breakthroughs In Language Model Training](hidden_breakthroughs_in_language_model_training.md)**

:   提出 POLCA 方法，将训练损失沿低秩训练子空间的任意基方向进行分解，揭示了在整体损失曲线平滑区域中隐藏的概念性突破（hidden breakthroughs），实现了对模型技能习得过程的无监督可解释性分析。

**[How Do Transformers Learn To Associate Tokens Gradient Leading Terms Bring Mecha](how_do_transformers_learn_to_associate_tokens_gradient_leading_terms_bring_mecha.md)**

:   通过对训练梯度的前导项近似分析，推导出Transformer在训练早期阶段各权重矩阵的闭式表达——均可分解为三种基函数（bigram、token-interchangeability、context mapping）的简单组合——从而揭示Transformer如何从自然语言数据中学习"bird"↔"flew"这类语义关联，且理论预测与真实LLM的学到权重高度吻合。

**[Implicit Statistical Inference In Transformers Approximating Likelihood-Ratio Te](implicit_statistical_inference_in_transformers_approximating_likelihood-ratio_te.md)**

:   从统计决策论视角出发，证明Transformer在上下文学习中能近似Bayes最优的**似然比检验**充分统计量，并通过机制分析揭示模型对线性/非线性任务采用不同深度的自适应电路。

**[Information Shapes Koopman Representation](information_shapes_koopman_representation.md)**

**[Initialization Schemes For Kolmogorov-Arnold Networks An Empirical Study](initialization_schemes_for_kolmogorov-arnold_networks_an_empirical_study.md)**

:   首次系统研究样条KAN的初始化策略，提出LeCun/Glorot启发的方差保持方案和经验幂律初始化族，通过大规模网格搜索+NTK动态分析发现幂律初始化整体最优，Glorot在参数多的模型上显著优于基线。

**[Internal Planning In Language Models Characterizing Horizon And Branch Awareness](internal_planning_in_language_models_characterizing_horizon_and_branch_awareness.md)**

:   提出基于VQ-VAE的信息论框架来分析语言模型内部的规划行为，发现规划视野是任务依赖的、模型隐式保留未选择的正确路径信息、下一token决策主要依赖最近的计算。

**[Layer By Layer Module By Module Choose Both For Optimal Ood Probing Of Vit](layer_by_layer_module_by_module_choose_both_for_optimal_ood_probing_of_vit.md)**

:   通过大规模线性探测实验系统研究预训练ViT的中间层行为，发现分布偏移是深层性能退化的主因，并在模块级别揭示了最优探测点取决于偏移程度：显著偏移时探测FFN激活最优，弱偏移时探测MHSA归一化输出最优。

**[Lore Jointly Learning The Intrinsic Dimensionality And Relative Similarity Struc](lore_jointly_learning_the_intrinsic_dimensionality_and_relative_similarity_struc.md)**

:   提出LORE——首个同时从序数三元组比较中联合学习嵌入表示和内在维度的框架：用非凸Schatten-p拟范数(p<1)正则化替代传统的预设维度策略，通过迭代重加权(IRNN)算法求解并证明收敛到稳定点；在合成数据、LLM模拟感知实验和3个众包数据集上，LORE在维度恢复上远超所有基线方法，同时保持高三元组准确率和语义可解释性。

**[Mata A Trainable Hierarchical Automaton System For Multi-Agent Visual Reasoning](mata_a_trainable_hierarchical_automaton_system_for_multi-agent_visual_reasoning.md)**

:   提出MATA（Multi-Agent hierarchical Trainable Automaton），将多Agent视觉推理建模为层次有限状态自动机，顶层状态转移由可训练的hyper agent（基于LLM的状态控制器）学习，每个Agent内部使用规则化的子自动机，通过共享内存实现协作与竞争，在多个视觉推理基准上达到SOTA。

**[Modal Logical Neural Networks For Financial Ai](modal_logical_neural_networks_for_financial_ai.md)**

:   提出模态逻辑神经网络（MLNN），将 Kripke 语义（必然/可能模态算子）集成到神经网络中，在金融合同安全审查、洗售合规和市场串谋检测中实现可审计的逻辑推理与深度学习性能的结合。

**[Narrow Finetuning Leaves Clearly Readable Traces In Activation Differences](narrow_finetuning_leaves_clearly_readable_traces_in_activation_differences.md)**

:   发现窄域微调（narrow finetuning）在 LLM 激活中留下清晰可读的痕迹：即使在无关文本的前几个 token 上，微调前后模型的激活差异也编码了微调目标的语义信息。通过 Activation Difference Lens（ADL）方法，可解释性 agent 识别微调目标的成功率达 91%，比黑盒基线高 2 倍以上。

**[Nimo A Nonlinear Interpretable Model](nimo_a_nonlinear_interpretable_model.md)**

:   NIMO 提出一种混合模型 $y = \sum_j x_j \beta_j (1 + g_{\mathbf{u}_j}(\mathbf{x}_{-j}))$，在保留线性回归系数全局可解释性（通过均值边际效应 MEM）的同时，利用神经网络提供逐实例的非线性修正，并通过参数消去法高效联合优化线性系数和网络参数。

**[Noise Stability Of Transformer Models](noise_stability_of_transformer_models.md)**

:   提出噪声稳定性（noise stability）替代平均敏感度（average sensitivity）作为衡量 Transformer 简单性偏差的更优指标，并基于此设计正则化方法，在合成任务和语言建模上分别加速训练约 35% 和 75%。

**[Polyshap Extending Kernelshap With Interaction-Informed Polynomial Regression](polyshap_extending_kernelshap_with_interaction-informed_polynomial_regression.md)**

:   本文提出 PolySHAP，通过将 KernelSHAP 的线性近似扩展为高阶多项式回归来捕获特征间的非线性交互，从而提升 Shapley 值的估计精度；并从理论上证明了配对采样（paired sampling）等价于二阶 PolySHAP，首次解释了配对采样启发式方法优越性能的根本原因。

**[Posh Using Scene Graphs To Guide Llms-As-A-Judge For Detailed Image Descriptions](posh_using_scene_graphs_to_guide_llms-as-a-judge_for_detailed_image_descriptions.md)**

:   提出PoSh评估指标，用场景图作为结构化评分标准引导LLM-as-Judge对详细图像描述进行细粒度错误定位（属性/关系误附着），配合DOCENT艺术品详细描述基准（1750专家描述+900细粒度人工判断），在人类判断相关性上超越GPT-4o-as-Judge且完全开源可复现。

**[Provably Explaining Neural Additive Models](provably_explaining_neural_additive_models.md)**

:   针对 Neural Additive Models (NAMs) 设计了专用的高效解释算法，仅需对数级别的验证查询即可生成可证明的基数最小解释（cardinally-minimal explanations），在速度和解释质量上均超越了现有的通用子集最小解释算法。

**[Radar Reasoning-Ability And Difficulty-Aware Routing For Reasoning Llms](radar_reasoning-ability_and_difficulty-aware_routing_for_reasoning_llms.md)**

:   本文提出 Radar 框架，将推理语言模型（RLM）的自适应推理问题建模为多目标优化，利用项目反应理论（IRT）联合估计可解释的查询难度和模型配置能力参数，实现轻量级、可扩展的查询级路由，在 8 个推理基准上优于 SOTA 路由方法，且仅增加约 7ms 延迟。

**[Salve Sparse Autoencoder-Latent Vector Editing For Mechanistic Control Of Neural](salve_sparse_autoencoder-latent_vector_editing_for_mechanistic_control_of_neural.md)**

:   提出 SALVE 框架——"发现-验证-控制"三阶段流程：用 L1 正则化稀疏自编码器发现模型的可解释特征基，用 Grad-FAM 可视化验证特征语义，再利用 SAE 解码器矩阵引导永久性权重空间编辑。在 ResNet-18 和 ViT-B/16 上验证了从类别抑制到跨类特征调控的精确、持久、低副作用控制。

**[Seed-Set Scalable Evolving Experimental Design For System-Level Ethical Testing](seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)**

:   提出 SEED-SET 框架，将自主系统的伦理评估建模为层次化贝叶斯实验设计问题，同时整合客观指标和主观价值判断，在有限预算下高效生成高伦理对齐度的测试用例。

**[Semantic Regexes Auto-Interpreting Llm Features With A Structured Language](semantic_regexes_auto-interpreting_llm_features_with_a_structured_language.md)**

:   提出 semantic regexes——一种用于自动描述 LLM 特征的结构化语言，通过 symbol/lexeme/field 三种原语及 context/composition/quantification 修饰符，在保持与自然语言同等准确度的同时，实现了更简洁、更一致的特征描述，并可量化特征复杂度随层的变化趋势。

**[Semantic Regexes Auto-Interpreting Llm Features With A Structured Language Of Re](semantic_regexes_auto-interpreting_llm_features_with_a_structured_language_of_re.md)**

:   本文提出 **Semantic Regexes（语义正则表达式）**，一种用于自动描述 LLM 特征的结构化语言，通过原语（symbol/lexeme/field）+ 修饰符（context/composition/quantification）组合，实现与自然语言同等准确但更简洁、一致且可分析的特征描述。

**[Stretching Beyond The Obvious A Gradient-Free Framework To Unveil The Hidden Lan](stretching_beyond_the_obvious_a_gradient-free_framework_to_unveil_the_hidden_lan.md)**

:   提出 Stretch-and-Squeeze（SnS）算法，一个无梯度、模型无关的双目标优化框架，通过在不同处理层级"拉伸"表征同时"压缩"目标单元激活来系统性地探测视觉系统的不变性流形，揭示了标准与鲁棒 CNN 之间不变性可解释性的分层差异。

**[Temporal Sparse Autoencoders Leveraging The Sequential Nature Of Language For In](temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)**

:   提出 Temporal SAEs (T-SAEs)，通过引入时间对比损失鼓励高层特征在相邻 token 间保持一致激活，在无显式语义信号的自监督训练下实现语义与句法特征的解耦，恢复更平滑、连贯的语义概念且不牺牲重构质量。

**[The Reasoning Trap -- Logical Reasoning As A Mechanistic Pathway To Advanced Jai](the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_advanced_jai.md)**

:   提出 RAISE 框架，论证逻辑推理能力（演绎、归纳、溯因）的改进是 AI 情境意识（situational awareness）的机制性路径，改善推理不可避免地放大了情境意识的危险前提条件。

**[The Reasoning Trap -- Logical Reasoning As A Mechanistic Pathway To Situational ](the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_situational_.md)**

:   立场论文，提出 RAISE 框架论证逻辑推理能力的提升（演绎/归纳/溯因）会系统性地使 LLM 获得情境感知（situational awareness）能力，从而开启自我推理→战略欺骗的升级路径，并指出当前安全措施不足以阻止这一趋势。

**[There Was Never A Bottleneck In Concept Bottleneck Models](there_was_never_a_bottleneck_in_concept_bottleneck_models.md)**

:   指出概念瓶颈模型（CBM）实际上并不存在真正的"瓶颈"——表征变量 $z_j$ 能预测概念 $c_j$ 不意味着它只编码 $c_j$ 的信息。提出 MCBM（Minimal Concept Bottleneck Model），通过信息瓶颈正则化约束每个 $z_j$ 仅保留对应概念的信息，实现真正的解耦表征和可靠的概念干预。

**[Tokenizing Single-Channel Eeg With Time-Frequency Motif Learning](tokenizing_single-channel_eeg_with_time-frequency_motif_learning.md)**

:   提出 TFM-Tokenizer，首个从单通道 EEG 学习时频 motif 词表并编码为离散 token 的框架，在事件分类、癫痫检测等任务上一致提升性能，且可作为即插即用组件增强现有 EEG 基础模型。

**[Tokenseek Memory Efficient Fine Tuning Via Instance-Aware Token Ditching](tokenseek_memory_efficient_fine_tuning_via_instance-aware_token_ditching.md)**

:   提出 TokenSeek，一个通用的 Transformer 微调内存优化插件，通过结合上下文注意力信息和梯度信息进行实例级 token 重要性评估，仅保留 10% 高价值 token 参与梯度更新，实现最高 65.7% 内存节省且性能持平甚至超越全 token 微调。

**[Towards Understanding Subliminal Learning When And How Hidden Biases Transfer](towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)**

:   本文通过受控实验和机制分析揭示了潜意识学习（subliminal learning）的本质——教师模型的隐藏偏好通过少量"分歧token"（divergence tokens）传递给学生模型，且早期层是关键，同时发现该现象非常脆弱，简单的同义改写即可抑制。

**[Uncovering Grounding Ids How External Cues Shape Multimodal Binding](uncovering_grounding_ids_how_external_cues_shape_multimodal_binding.md)**

:   揭示LVLM中外部视觉线索改善推理的内部机制——发现Grounding IDs(潜在标识符，绑定视觉特征到外部线索对应文本)，因果实验(swap accuracy=0.98)证明分区诱导外部线索→准确跨模态对齐→减少幻觉→增强推理。

**[Uni-Ntfm A Unified Foundation Model For Eeg Signal Representation Learning](uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)**

:   Uni-NTFM 是一个受生物神经机制启发的 EEG 统一基础模型，通过异质特征投影模块解耦时域和频域编码、拓扑嵌入机制将不同传感器配置对齐到统一功能拓扑空间、以及混合专家 Transformer 实现功能模块化和稀疏编码，在 28000 小时 EEG 数据上预训练并达到 19 亿参数规模，在 9 个下游任务上的线性探测和微调设定下均超越现有模型。

**[Universal Properties Of Activation Sparsity In Modern Large Language Models](universal_properties_of_activation_sparsity_in_modern_large_language_models.md)**

:   对现代 LLM（GLU 架构 + SiLU/GELU）的激活稀疏性进行系统性研究，提出通用的 top-p 稀疏化框架和临界稀疏度（critical sparsity）指标，发现激活稀疏度随模型规模单调递增、输入稀疏化是最实用的免训练加速方案，并首次证明扩散型 LLM 也具有显著的激活稀疏性。

**[Vcworld A Biological World Model For Virtual Cell Simulation](vcworld_a_biological_world_model_for_virtual_cell_simulation.md)**

:   提出 VCWorld，一个细胞级白盒模拟器，通过整合结构化生物知识与大语言模型的迭代推理能力来构建生物世界模型，以数据高效的方式生成可解释的扰动预测和机制假说。

**[When Machine Learning Gets Personal Evaluating Prediction And Explanation](when_machine_learning_gets_personal_evaluating_prediction_and_explanation.md)**

:   本文提出统一框架量化模型个性化对预测准确性和解释质量的影响，证明二者可以分离（预测不变但解释变好/变差），推导了基于数据集统计量的假设检验误差概率有限样本下界，揭示了许多实际场景中个性化效果在统计上根本不可检验。

**[When Thinking Backfires Mechanistic Insights Into Reasoning-Induced Misalignment](when_thinking_backfires_mechanistic_insights_into_reasoning-induced_misalignment.md)**

:   发现并机制性地解释"推理诱导失对齐"（RIM）现象：增强推理能力（CoT prompting 或数学微调）会削弱安全守护，原因是推理和安全共享神经元资源，训练推理时安全关键神经元的激活发生不成比例的偏移。

**[Zerotuning Unlocking The Initial Tokens Power To Enhance Large Language Models W](zerotuning_unlocking_the_initial_tokens_power_to_enhance_large_language_models_w.md)**

:   提出 ZeroTuning，仅需对初始 token（如 `<BOS>`）的注意力分数进行头部特异性缩放，即可在无训练情况下提升 LLM 在 15 个数据集上的表现，仅需修改 4 行代码。
