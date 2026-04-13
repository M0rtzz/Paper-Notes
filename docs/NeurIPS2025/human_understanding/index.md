---
title: >-
  NeurIPS2025 人体理解方向 77篇论文解读
description: >-
  77篇NeurIPS2025 人体理解方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**🧠 NeurIPS2025** · 共 **77** 篇

**[A Differential And Pointwise Control Approach To Reinforceme](a_differential_and_pointwise_control_approach_to_reinforceme.md)**

:   将RL问题通过连续时间控制的微分对偶形式重新表述，利用哈密顿结构嵌入物理先验，提出dfPO算法实现逐点策略优化，在科学计算任务（曲面建模、网格控制、分子动力学）上以更少样本超越12个RL基线。

**[A Practical Guide For Incorporating Symmetry In Diffusion Policy](a_practical_guide_for_incorporating_symmetry_in_diffusion_policy.md)**

:   本文提出了一套将对称性融入扩散策略的实用指南——通过不变性表征（相对轨迹动作 + 手眼感知）、等变视觉编码器和 Frame Averaging 三种简单方法，在 MimicGen 12 个任务上达到了接近甚至超越完全等变扩散策略的性能，同时实现复杂度大幅降低。

**[A Regularized Newton Method For Nonconvex Optimization With](a_regularized_newton_method_for_nonconvex_optimization_with.md)**

:   提出一类基于当前与历史梯度构造的新型正则化器，结合带负曲率监测的共轭梯度法求解正则化Newton方程，在不需要Hessian Lipschitz常数先验知识的自适应框架下，首次同时实现了$O(\epsilon^{-3/2})$最优全局迭代复杂度和二次局部收敛速率。

**[A Simple Linear Patch Revives Layerpruned Large Language Mod](a_simple_linear_patch_revives_layerpruned_large_language_mod.md)**

:   提出 LinearPatch，一种即插即用的轻量修补技术，通过在剪枝界面插入一个融合了 Hadamard 变换（压制 token 级outlier）和通道缩放（对齐通道幅度）的对称矩阵，有效弥合层剪枝后的激活幅度失配问题，在 LLaMA-3-8B 上剪掉 5/32 层后仍保留 94.15% 性能（无训练），加上 30 分钟蒸馏可达 95.16%。

**[Ada-Kv Optimizing Kv Cache Eviction By Adaptive Budget Allocation For Efficient ](ada-kv_optimizing_kv_cache_eviction_by_adaptive_budget_allocation_for_efficient_.md)**

:   发现现有 KV cache 驱逐方法对所有注意力头均匀分配预算忽略了头间注意力集中度的巨大差异,提出 Ada-KV——首个 head-wise 自适应预算分配策略,将稀疏头的预算重新分配给分散头,理论证明最小化驱逐损失上界,在 29 个数据集上即插即用地提升现有方法。

**[Agint Agentic Graph Compilation For Software Engineering Age](agint_agentic_graph_compilation_for_software_engineering_age.md)**

:   提出 Agint，一个将自然语言意图编译为类型化、效果感知的DAG（有向无环图）的 agentic 图编译器，通过六层类型地板（TEXT→TYPED→SPEC→STUB→SHIM→PURE）渐进式精化自然语言为可执行代码，支持中间表示可执行、混合JIT运行时和Unix风格的可组合工具链。

**[Bedlam20 Synthetic Humans And Cameras In Motion](bedlam20_synthetic_humans_and_cameras_in_motion.md)**

:   BEDLAM 数据集的重大升级版，新增多样化相机运动（合成+手持+头戴设备捕获）、更广的焦距范围（14-400mm）、更多样化体型/发型/鞋子/服装，总计 27K 序列 8M+ 帧，显著提升世界坐标 3D 人体估计的精度。

**[Breaking The Gradient Barrier Unveiling Large Language Models For Strategic Clas](breaking_the_gradient_barrier_unveiling_large_language_models_for_strategic_clas.md)**

:   提出 GLIM（Gradient-free Learning In-context Method），首次利用 LLM 的 In-Context Learning 机制隐式模拟策略分类中的双层优化（特征操纵 + 决策规则优化），无需微调即可在大规模数据上高效完成策略分类任务。

**[Bubbleformer Forecasting Boiling With Transformers](bubbleformer_forecasting_boiling_with_transformers.md)**

:   提出 BubbleFormer，基于分解时空轴注意力的 Transformer 架构用于预测沸腾动力学——包括难以预测的自主气泡成核事件，配合 BubbleML 2.0 数据集（160+ 高保真仿真），在多种流体、几何和壁面条件下实现准确的沸腾时空过程预测。

**[Consistent Supervised-Unsupervised Alignment For Generalized Category Discovery](consistent_supervised-unsupervised_alignment_for_generalized_category_discovery.md)**

:   提出 NC-GCD 框架，通过预分配固定的 Equiangular Tight Frame (ETF) 原型为已知类和新类建立统一优化目标，结合语义一致性匹配器 (SCM) 稳定跨迭代伪标签分配，在 6 个 GCD 基准上显著提升新类发现精度。

**[Counteractive Rl Rethinking Core Principles For Efficient And Scalable Deep Rein](counteractive_rl_rethinking_core_principles_for_efficient_and_scalable_deep_rein.md)**

:   CoAct TD Learning 颠覆 ε-greedy 的随机探索范式——以概率 ε 选择最小化 $Q(s,a)$ 的动作（而非随机动作）来获取高时间差分信号，理论证明其产生更大 TD 误差，在 Atari 100K 上实现 248% 性能提升，仅需改动 2 行代码且零额外计算。

**[Cpep Contrastive Pose-Emg Pre-Training Enhances Gesture Generalization On Emg Si](cpep_contrastive_pose-emg_pre-training_enhances_gesture_generalization_on_emg_si.md)**

:   提出 CPEP 框架，通过对比学习将低质量 EMG 信号表征与高质量手部姿态表征对齐，使 EMG 编码器获得姿态感知能力，首次实现从 EMG 信号零样本识别未见手势，分布内手势分类提升 21%、未见手势分类提升 72%。

**[Cycle-Sync Robust Global Camera Pose Estimation Through Enhanced Cycle-Consisten](cycle-sync_robust_global_camera_pose_estimation_through_enhanced_cycle-consisten.md)**

:   提出 Cycle-Sync 全局相机位姿估计框架，通过将消息传递最小二乘 (MPLS) 扩展到相机位置估计、引入 Welsch 型鲁棒损失和环一致性加权，在无需 bundle adjustment 的情况下超越了包括完整 SfM pipeline（含 BA）在内的所有基线方法。

**[Data-Juicer 20 Cloud-Scale Adaptive Data Processing For And With Foundation Mode](data-juicer_20_cloud-scale_adaptive_data_processing_for_and_with_foundation_mode.md)**

:   Data-Juicer 2.0 是面向基础模型的云规模多模态数据处理系统，150+ 跨文本/图像/视频/音频算子，支持自适应分布式执行（Ray/MaxCompute），在 10000+ CPU 核心上高效处理 TB 级数据，已广泛应用于阿里云 PAI 等产品。

**[Decomposition Of Small Transformer Models](decomposition_of_small_transformer_models.md)**

:   将 Stochastic Parameter Decomposition (SPD) 扩展到 Transformer，设计适用于序列数据的因果重要性函数和新损失函数，在玩具 induction head 上恢复期望两步电路，在 GPT-2-small 上定位到"高尔夫""篮球"等可解释概念对应的 rank-1 参数子空间。

**[Devfd Developmental Face Forgery Detection By Learning Shared And Orthogonal Lor](devfd_developmental_face_forgery_detection_by_learning_shared_and_orthogonal_lor.md)**

:   提出 DevFD——一种发展式 MoE 架构，用共享 Real-LoRA 建模真实人脸共性、正交 Fake-LoRA 序列逐步建模新伪造类型，并通过将正交梯度集成到正交损失中缓解灾难性遗忘，在持续学习人脸伪造检测中达到最高准确率和最低遗忘率。

**[Discovering Transformer Circuits Via A Hybrid Attribution And Pruning Framework](discovering_transformer_circuits_via_a_hybrid_attribution_and_pruning_framework.md)**

:   提出混合归因与剪枝框架 HAP，先用快速的边归因修补（EAP）筛选高潜力子图，再在缩小后的搜索空间上运行精确的边剪枝（EP），在 GPT-2 Small 的 IOI 任务上比纯 EP 快 46% 且保持相当的电路忠实度，同时成功保留了 EAP 单独使用时会遗漏的 S-inhibition 头。

**[Distillation Robustifies Unlearning](distillation_robustifies_unlearning.md)**

:   揭示了"蒸馏能使遗忘变得鲁棒"的核心发现——将遗忘后的模型蒸馏到随机初始化的学生网络中能有效丢弃潜在能力，并基于此提出UNDO方法（Unlearn-Noise-Distill-on-Outputs），通过对遗忘模型权重加噪再蒸馏，建立了计算量与鲁棒性之间的可调权衡，在合成任务和WMDP基准上接近从头重训的黄金标准。

**[Distribution Learning Meets Graph Structure Sampling](distribution_learning_meets_graph_structure_sampling.md)**

:   本文建立了高维概率图模型 PAC 学习与图结构高效计数/采样之间的新联系，利用在线学习框架（EWA/RWM）将指数级专家集合的维护问题转化为 DAG 结构的加权采样问题，首次给出了弦图骨架贝叶斯网络的高效 agnostic 学习算法，并将树结构分布的样本复杂度从 O(nk³/ε) 改进到最优的 O(nk²/ε)。

**[Emergent World Beliefs Exploring Transformers In Stochastic Games](emergent_world_beliefs_exploring_transformers_in_stochastic_games.md)**

:   将LLM涌现世界模型的研究从完全信息游戏（Othello、国际象棋）扩展到不完全信息领域（德州扑克），通过在PHH格式扑克数据上预训练GPT-2并探测其内部激活，证明模型不仅学习了确定性特征（牌型识别~98%准确率），还自发发展了对随机性特征（胜率/equity，相关系数0.59）的内部表示。

**[Evolutionary Learning In Spatial Agent-Based Models For Physical Climate Risk As](evolutionary_learning_in_spatial_agent-based_models_for_physical_climate_risk_as.md)**

:   提出一种整合地理空间气候灾害数据与进化学习机制的Agent-Based Model（ABM），在包含商品-制造-零售三级供应链的简化经济网络上，通过RCP8.5洪水投影模拟2025-2100年的经济响应，证明了进化自适应机制使企业在气候压力下维持显著更高的生产、资本、流动性和就业水平，同时揭示了传统资产级评估无法捕捉的供应链系统性风险。

**[Exploration Of Incremental Synthetic Non-Morphed Images For Single Morphing Atta](exploration_of_incremental_synthetic_non-morphed_images_for_single_morphing_atta.md)**

:   系统研究了在单图像变形攻击检测（S-MAD）训练中增量引入合成非变形人脸图像的效果，发现适量的合成数据（~75%增量）可提升跨数据集泛化能力（EER从6.17%降至6.10%），但过度使用或仅用合成数据会导致性能严重退化（EER升至~38%）。

**[Face-Human-Bench A Comprehensive Benchmark Of Face And Human Understanding For M](face-human-bench_a_comprehensive_benchmark_of_face_and_human_understanding_for_m.md)**

:   提出 Face-Human-Bench，首个系统评估多模态大模型人脸与人体理解能力的基准，包含三级能力分类体系（2个L1 × 10个L2 × 18个L3），开发集与测试集各 1800 题，支持中英双语，评测 25 个主流 MLLM 并揭示其与专家模型的显著差距。

**[Face A General Framework For Mapping Collaborative Filtering Embeddings Into Llm](face_a_general_framework_for_mapping_collaborative_filtering_embeddings_into_llm.md)**

:   FACE 提出将协同过滤（CF）嵌入通过解纠缠投影 + 残差量化映射为 LLM 预训练 token（描述符），再用对比学习对齐语义，无需微调 LLM 即可实现 CF 嵌入的语义解读和推荐性能增强。

**[Face Faithful Automatic Concept Extraction](face_faithful_automatic_concept_extraction.md)**

:   提出 FACE 框架，在非负矩阵分解 (NMF) 中加入 KL 散度正则项，约束概念重建后的激活值保持与原始模型预测一致，从而提取真正忠实于模型决策过程的概念解释，在 ImageNet/COCO/CelebA 上全面超越 CRAFT 和 ICE。

**[Faster Algorithm For Structured John Ellipsoid Computation](faster_algorithm_for_structured_john_ellipsoid_computation.md)**

:   针对对称凸多面体 $P = \{x \in \mathbb{R}^d : -\mathbf{1}_n \leq Ax \leq \mathbf{1}_n\}$ 的 John 椭球计算问题，提出两个快速算法：基于 sketching 的近输入稀疏度算法 $\widetilde{O}(\text{nnz}(A) + d^\omega)$ 每次迭代，和基于树宽的算法 $O(n\tau^2)$ 每次迭代，均显著优于已有最优 $O(nd^2)$。

**[Firstaidqa A Synthetic Dataset For First Aid And Emergency Response In Low-Conne](firstaidqa_a_synthetic_dataset_for_first_aid_and_emergency_response_in_low-conne.md)**

:   构建 FirstAidQA，一个包含 5500 条合成急救问答对的数据集，基于认证急救教材用 ChatGPT-4o-mini 生成，经人工验证，旨在支撑低连接/离线环境下急救 AI 系统的微调训练。

**[Graphchain Large Language Models For Large-Scale Graph Analysis Via Tool Chainin](graphchain_large_language_models_for_large-scale_graph_analysis_via_tool_chainin.md)**

:   提出 GraphChain 框架，通过渐进式图蒸馏（RL驱动的工具链序列生成）和结构感知测试时自适应（基于图拓扑指纹的轻量适配器），使 LLM 能像人类探索未知环境一样，通过动态工具链序列逐步分析大规模图数据，平均准确率 84.7% 超越最优基线 20.7%，可扩展至 20 万节点。

**[Gui-Rise Structured Reasoning And History Summarization For Gui Navigation](gui-rise_structured_reasoning_and_history_summarization_for_gui_navigation.md)**

:   提出 GUI-Rise 框架，通过结构化推理（进度估计 + 决策推理）、动作预测和历史摘要三个子任务的联合设计，结合 GRPO 强化学习与历史摘要奖励，显著提升 GUI 导航智能体在跨域场景下的泛化能力。

**[Hoi-Dyn Learning Interaction Dynamics For Human-Object Motion Diffusion](hoi-dyn_learning_interaction_dynamics_for_human-object_motion_diffusion.md)**

:   将人体-物体交互（HOI）生成建模为 Driver-Responder 系统，通过轻量级 Transformer 交互动力学模型显式预测物体对人体动作的响应，利用残差动力学损失在训练时增强因果一致性，同时保持推理效率。

**[Human-Machine Ritual Synergic Performance Through Real-Time Motion Recognition](human-machine_ritual_synergic_performance_through_real-time_motion_recognition.md)**

:   提出一种轻量级实时动作识别系统，利用可穿戴 IMU 传感器 + MiniRocket 时序分类器实现 <50ms 延迟的舞者特定动作识别（96.05% 准确率），通过"具身记忆映射"将舞者的个人动作-声音关联编码到系统中，构建了一种尊重人体表达深度的人机协作表演范式。

**[In-Context Compositional Learning Via Sparse Coding Transformer](in-context_compositional_learning_via_sparse_coding_transformer.md)**

:   受稀疏编码启发，将 Transformer 注意力机制重新解释为在编码字典和解码字典上的投影，通过稀疏系数显式表示组合规则，并利用提升方案（lifting scheme）将上下文任务的组合规则迁移到目标任务。

**[Incentivizing Reasoning For Advanced Instruction-Following Of Large Language Mod](incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)**

:   提出 RAIF，通过 RL+规则中心奖励培养 LLM 在复杂指令（含 And/Chain/Selection/Nested 组合约束）下的深度推理能力：发现 vanilla CoT 对指令跟随有负面影响（因 LLM 只会浅层复述指令），设计 superior CoT enforcement（样本级对比过滤无效推理）+ 行为克隆控制分布偏移，1.5B 模型匹配 8B 性能，7 个 benchmark 平均提升 11.74%。

**[K-Decore Facilitating Knowledge Transfer In Continual Structured Knowledge Reaso](k-decore_facilitating_knowledge_transfer_in_continual_structured_knowledge_reaso.md)**

:   提出 K-DeCore 框架，通过知识解耦将结构化知识推理分为任务无关的 schema 过滤和任务特定的 query 构建两阶段，配合双视角记忆构建和结构引导的伪数据合成策略，在固定参数量下实现跨异构 SKR 任务的有效知识迁移。

**[Learning Dense Hand Contact Estimation From Imbalanced Data](learning_dense_hand_contact_estimation_from_imbalanced_data.md)**

:   提出 HACO 框架，通过平衡接触采样（BCS）解决类别不平衡和顶点级类别平衡损失（VCB Loss）解决空间不平衡，首次在 14 个数据集（65.5 万图像）上训练稠密手部接触估计模型，在多种交互场景下达到 SOTA。

**[Learning From Design Procedure To Generate Cad Programs For Data Augmentation](learning_from_design_procedure_to_generate_cad_programs_for_data_augmentation.md)**

:   提出一种受工业设计流程启发的CAD程序数据增强范式，通过向LLM提供参考曲面程序和设计流程描述来引导生成包含B-Spline有机形状的CAD程序，显著缩小了公开CAD数据集与工业级设计在几何复杂度上的差距。

**[Learning Skill-Attributes For Transferable Assessment In Video](learning_skill-attributes_for_transferable_assessment_in_video.md)**

:   提出CrossTrainer方法，通过发现跨运动通用的技能属性（如平衡、控制、手部定位）作为中间表示，训练多模态语言模型从视频中生成可操作反馈和水平评估，在跨运动零样本迁移中相对SOTA提升高达60%。

**[Learning To Watermark A Selective Watermarking Framework For Large Language Mode](learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)**

:   提出LTW（Learning to Watermark）框架，使用一个轻量级选择器网络基于句子嵌入、token熵和当前水印比例来自适应决定何时施加水印，通过多目标优化（MGDA）在可检测性和文本质量之间达到Pareto最优，在不降低检测性能的前提下显著提升水印文本质量。

**[Lumia A Handheld Vision-To-Music System For Real-Time Embodied Composition](lumia_a_handheld_vision-to-music_system_for_real-time_embodied_composition.md)**

:   提出Lumia——一个手持相机式设备，通过GPT-4 Vision分析拍摄画面生成结构化提示，再由Stable Audio合成音乐循环段，实现从视觉到音乐的实时、具身化即兴创作工作流。

**[Mapping Faithful Reasoning In Language Models](mapping_faithful_reasoning_in_language_models.md)**

:   提出Concept Walk框架，通过将推理模型每步的残差流激活投影到从对比数据学到的概念方向上，追踪内部概念表示在推理过程中的演化轨迹，以此区分CoT链是真正参与计算的还是仅为事后合理化的装饰性输出。

**[Mdreid Modality-Decoupled Learning For Any-To-Any Multi-Modal Object Re-Identifi](mdreid_modality-decoupled_learning_for_any-to-any_multi-modal_object_re-identifi.md)**

:   提出MDReID框架，通过将模态特征解耦为模态共享（modality-shared）和模态特有（modality-specific）两部分，实现任意模态组合下的目标重识别（any-to-any ReID），在模态匹配和模态不匹配场景下均大幅超越现有方法。

**[Mingle Mixture Of Null-Space Gated Low-Rank Experts For Test-Time Continual Mode](mingle_mixture_of_null-space_gated_low-rank_experts_for_test-time_continual_mode.md)**

:   提出测试时持续模型合并(TTCMM)新范式及Mingle框架，通过低秩专家混合架构和自适应零空间约束门控机制，在测试时利用少量无标签样本实现模型动态合并，在多个基准上以7-9%的优势超越SOTA，同时将遗忘降至接近零。

**[Mixing Expert Knowledge Bring Human Thoughts Back To The Game Of Go](mixing_expert_knowledge_bring_human_thoughts_back_to_the_game_of_go.md)**

:   提出 LoGos，通过混合领域专家数据（围棋）与通用长 CoT 推理数据进行冷启动微调 + GRPO 强化学习，使通用 LLM 在围棋中达到职业棋手水平的同时保持优秀的通用推理能力。

**[Mospa Human Motion Generation Driven By Spatial Audio](mospa_human_motion_generation_driven_by_spatial_audio.md)**

:   首次提出空间音频驱动的人体运动生成：构建 SAM 数据集（9+ 小时 Ambisonics 空间音频-运动配对数据），设计 MOSPA 扩散模型框架融合空间位置信息 + 语义音频特征，在 VR/游戏/辅助技术等方面有应用前景。

**[Mouse-Guided Gaze Semi-Supervised Learning Of Intention-Aware Representations Fo](mouse-guided_gaze_semi-supervised_learning_of_intention-aware_representations_fo.md)**

:   提出一种半监督框架，利用鼠标轨迹作为弱监督信号预训练眼动表征，然后在标注数据上微调以区分阅读与扫描行为，在推理时仅使用眼动信号，实现免手操作的辅助阅读检测。

**[Neural Collapse In Cumulative Link Models For Ordinal Regression An Analysis Wit](neural_collapse_in_cumulative_link_models_for_ordinal_regression_an_analysis_wit.md)**

:   将Neural Collapse (NC)理论扩展到基于累积链接模型(CLM)的序数回归(OR)任务中，在无约束特征模型(UFM)框架下证明了Ordinal Neural Collapse (ONC)的三个标志性质：类内均值坍缩(ONC1)、特征坍缩到一维子空间(ONC2)、以及潜变量按类别顺序排列(ONC3)，并在零正则极限下揭示了潜变量与阈值之间的简洁几何关系。

**[Nnterp A Standardized Interface For Mechanistic Interpretability Of Transformers](nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)**

:   开发 nnterp 库，通过自动模块重命名和验证测试，为 50+ 个 Transformer 变体提供统一的可解释性分析接口，解决 TransformerLens 正确性与 NNsight 可用性之间的权衡。

**[Node-Based Editing For Multimodal Generation Of Text Audio Image And Video](node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)**

:   提出一个节点图式故事编辑系统，允许创作者通过自然语言和节点级操作迭代地生成、编辑和比较多模态内容（文本、音频、图像、视频），支持线性和分支叙事结构。

**[Offline Policy Evaluation Of Multi-Turn Llm Health Coaching With Real Users](offline_policy_evaluation_of_multi-turn_llm_health_coaching_with_real_users.md)**

:   在实际部署的 LLM 健康教练系统上进行离线策略评估（OPE），发现统一的高工具使用策略虽提升平均奖励但损害特定用户子群，并通过模拟器验证了早期信息增益探索（好奇心奖励）可加速用户特征识别和提升任务成功率。

**[Parco Parallel Autoregressive Models For Multi-Agent Combinatorial Optimization](parco_parallel_autoregressive_models_for_multi-agent_combinatorial_optimization.md)**

:   提出 PARCO 框架，通过 Communication Layers 实现智能体间协调、Multiple Pointer Mechanism 实现并行解码、Priority-based Conflict Handler 解决冲突，高效求解多智能体组合优化问题。

**[Policy Compatible Skill Incremental Learning Via Lazy Learning Interface](policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)**

:   提出SIL-C框架，通过双向惰性学习接口(bilateral lazy learning interface)实现技能增量学习中的技能-策略兼容性，使增量更新的技能能直接提升下游策略性能而无需重训练或结构调整。

**[Power Ensemble Aggregation For Improved Extreme Event Ai Prediction](power_ensemble_aggregation_for_improved_extreme_event_ai_prediction.md)**

:   提出基于幂均值的自适应集成聚合方法，通过对生成式天气预测模型的集成成员得分施加非线性聚合（幂指数$p>1$），显著提升极端高温事件的分类性能，尤其在高分位数阈值下效果更佳。

**[Preference-Based Reinforcement Learning Beyond Pairwise Comparisons Benefits Of ](preference-based_reinforcement_learning_beyond_pairwise_comparisons_benefits_of_.md)**

:   在偏好强化学习中提出 M-AUPO 算法,利用 Plackett-Luce 排序模型处理多选项比较反馈,首次从理论上证明更大的子集规模直接改善样本效率。

**[Raptr Radar-Based 3D Pose Estimation Using Transformer](raptr_radar-based_3d_pose_estimation_using_transformer.md)**

:   提出RAPTR，首个利用弱监督（3D BBox + 2D关键点标签）进行雷达3D人体姿态估计的Transformer框架，通过伪3D可变形注意力和结构化损失函数在两个室内数据集上大幅超过基线。

**[Recurrent Attention-Based Token Selection For Efficient Streaming Video-Llms](recurrent_attention-based_token_selection_for_efficient_streaming_video-llms.md)**

:   提出 rLiVS（Recurrent LLM-informed Visual Selection），一种无需训练的通用流式视频理解方法，通过LLM注意力权重选择关键视觉token（仅保留~6%）、循环复用历史token、基于字幕的检索问答三重设计，在流式视频基准上取得SOTA。

**[Reflective Translation Improving Low-Resource Machine Translation Via Structured](reflective_translation_improving_low-resource_machine_translation_via_structured.md)**

:   提出Reflective Translation提示框架，让LLM通过结构化的多轮自我评估和修正来改善低资源语言翻译质量，在英语-isiZulu和英语-isiXhosa上取得统计显著的改进。

**[Searching Latent Program Spaces](searching_latent_program_spaces.md)**

:   提出 Latent Program Network（LPN），通过编码器将输入-输出示例映射为潜在程序表示，在测试时通过梯度搜索潜在空间来适应新任务，在 ARC-AGI 基准上显著优于 in-context learning 和 test-time training 方法。

**[Semantic Retrieval Augmented Contrastive Learning For Sequential Recommendation](semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)**

:   提出SRA-CL框架，利用LLM的语义理解能力构建高质量对比样本对，通过语义检索+可学习样本合成器增强序列推荐的对比学习，以即插即用的方式在4个数据集上取得SOTA。

**[Sharpness-Aware Minimization With Z-Score Gradient Filtering](sharpness-aware_minimization_with_z-score_gradient_filtering.md)**

:   提出 Z-Score Filtered SAM (ZSAM)，通过对每层梯度进行 Z-Score 统计过滤，仅保留最显著的梯度分量进行扰动上升步骤，从而引导优化器更有效地搜索平坦极小值，在多个数据集和架构上一致提升测试精度。

**[Spatial Understanding From Videos Structured Prompts Meet Simulation Data](spatial_understanding_from_videos_structured_prompts_meet_simulation_data.md)**

:   提出 SpatialMind 结构化提示策略与 ScanForgeQA 合成QA数据集的双管齐下方案，在不修改VLM架构的前提下显著增强其从扫描视频进行3D空间推理的能力。

**[Specattn Speculating Sparse Attention](specattn_speculating_sparse_attention.md)**

:   SpecAttn 提出一种无需训练的方法，利用投机解码中草稿模型已计算的注意力权重来预测验证模型的重要 token，通过 KL 散度层映射 + 免排序 top-p 核选择 + 动态 KV 缓存剪枝，实现 78.4% 的 KV 缓存访问减少，困惑度仅增加 15.29%，显著优于现有稀疏注意力方法。

**[Spurious-Aware Prototype Refinement For Reliable Out-Of-Distribution Detection](spurious-aware_prototype_refinement_for_reliable_out-of-distribution_detection.md)**

:   SPROD 是一种后置（post-hoc）OOD 检测方法，专门应对训练数据中的虚假相关——通过将每个类别的原型细分为"正确分类"和"误分类"子组（后者共享虚假特征），配合 K-means 式精炼和距离式（生成式）评分，在 5 个虚假相关 OOD 基准上平均 AUROC 85.1%（+4.8% vs 次优 KNN），FPR@95 49.0%（-9.3% vs 次优）。

**[Stable Coresets Via Posterior Sampling Aligning Induced And Full Loss Landscapes](stable_coresets_via_posterior_sampling_aligning_induced_and_full_loss_landscapes.md)**

:   提出基于后验采样的 coreset 选择框架，通过在 BatchNorm 层上采样权重扰动来平滑损失曲面，保证 coreset 与全数据集的损失景观对齐（包含 Hessian 和 Newton 步的近似），在高标签噪声下显著优于现有方法。

**[Stealthy Yet Effective Distribution-Preserving Backdoor Attacks On Graph Classif](stealthy_yet_effective_distribution-preserving_backdoor_attacks_on_graph_classif.md)**

:   提出 DPSBA，一种面向图分类的 clean-label 后门攻击框架，通过对抗训练生成分布内（in-distribution）触发子图，同时抑制结构异常和语义异常，在保持高攻击成功率的同时显著提升隐蔽性。

**[Stochastic Momentum Methods For Non-Smooth Non-Convex Finite-Sum Coupled Composi](stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)**

:   针对非光滑非凸有限和耦合复合优化 (FCCO) 问题，提出两种随机动量方法 SONEX（单循环）和 ALEXR2（双循环），通过外层 Moreau 包络平滑和嵌套平滑技术将迭代复杂度从 $O(1/\epsilon^6)$ 改进至 $O(1/\epsilon^5)$，并在非凸不等式约束优化中取得同等最优复杂度。

**[Succeed Or Learn Slowly Sample Efficient Off-Policy Reinforcement Learning For M](succeed_or_learn_slowly_sample_efficient_off-policy_reinforcement_learning_for_m.md)**

:   提出SoLS算法，通过不对称策略更新机制（成功时激进学习、失败时保守正则化）和成功转换回放（STR），实现基础模型在移动应用控制任务上的高效强化学习微调，在AndroidWorld上达到51.3%成功率。

**[Switchable Token-Specific Codebook Quantization For Face Image Compression](switchable_token-specific_codebook_quantization_for_face_image_compression.md)**

:   提出可切换的token专属码本量化机制（STSCQ），通过图像级码本路由和token级码本分割的层次动态结构，在超低比特率下显著提升人脸图像的压缩重建质量和识别精度。

**[Symrtlo Enhancing Rtl Code Optimization With Llms And Neuron-Inspired Symbolic R](symrtlo_enhancing_rtl_code_optimization_with_llms_and_neuron-inspired_symbolic_r.md)**

:   提出 SymRTLO，首个将LLM与符号推理集成的神经符号框架用于RTL代码优化，通过检索增强优化规则、AST模板引导代码生成和FSM符号系统，在功耗、性能和面积(PPA)上分别获得最高43.9%、62.5%和51.1%的提升。

**[Tensorrl-Qas Reinforcement Learning With Tensor Networks For Improved Quantum Ar](tensorrl-qas_reinforcement_learning_with_tensor_networks_for_improved_quantum_ar.md)**

:   提出 TensorRL-QAS 框架，通过用张量网络（MPS/DMRG）对强化学习量子架构搜索进行 warm-start，显著降低电路深度和 CNOT 门数量（最高 10 倍），同时加速训练（最高 98%），有效解决了 RL-QAS 在大规模量子系统上的可扩展性瓶颈。

**[The Last Vote A Multi-Stakeholder Framework For Language Model Governance](the_last_vote_a_multi-stakeholder_framework_for_language_model_governance.md)**

:   提出一个面向语言模型治理的综合框架，包含七类民主风险分类体系、利益相关方自适应事件严重度评分(ISS)、以及分阶段六年实施路线图，旨在将民主价值融入AI监管的制度设计中。

**[The Transparent Earth A Multimodal Foundation Model For The Earths Subsurface](the_transparent_earth_a_multimodal_foundation_model_for_the_earths_subsurface.md)**

:   提出Transparent Earth，一种基于Transformer的多模态基础模型，通过位置编码和文本衍生的模态嵌入融合8种异质地球物理观测数据，实现地球地下属性的零样本推断和上下文学习预测。

**[Uncle Towards Scalable Dynamic Causal Discovery In Non-Linear Temporal Systems](uncle_towards_scalable_dynamic_causal_discovery_in_non-linear_temporal_systems.md)**

:   提出 UnCLe，一种基于 TCN 自编码器解耦和自回归依赖矩阵的可扩展动态因果发现方法，通过时序扰动后逐数据点预测误差增量推断时变因果关系，在静态和动态因果发现基准上均达到 SOTA。

**[Uncovering Strategic Egoism Behaviors In Large Language Models](uncovering_strategic_egoism_behaviors_in_large_language_models.md)**

:   本文首次系统研究了大语言模型中的"策略性自利"（Strategic Egoism）行为，发布了包含160个场景的SEBench基准，发现7个主流LLM中自利行为平均占比69.11%，且与毒性语言正相关。

**[Vasa-3D Lifelike Audio-Driven Gaussian Head Avatars From A Single Image](vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)**

:   提出VASA-3D，通过将VASA-1的2D运动隐空间适配到3D高斯溅射表征，并利用VASA-1合成训练数据进行单图定制优化，实现了从单张肖像照到逼真音频驱动3D头部化身的实时生成（512×512, 75fps）。

**[Vimorag Video-Based Retrieval-Augmented 3D Motion Generation For Motion Language](vimorag_video-based_retrieval-augmented_3d_motion_generation_for_motion_language.md)**

:   提出 VimoRAG 框架，利用大规模野外视频数据库作为2D运动先验来增强3D运动生成，通过 Gemini-MVR 检索器和 McDPO 训练策略解决人体动作视频检索和错误传播两大瓶颈。

**[Vision Transformers For Cosmological Fields Application To Weak Lensing Mass Map](vision_transformers_for_cosmological_fields_application_to_weak_lensing_mass_map.md)**

:   首次将 Vision Transformers（ViT 和 Swin Transformer）应用于弱引力透镜收敛场的宇宙学参数（$\Omega_m$ 和 $S_8$）约束，通过模拟推断框架系统比较了注意力架构与 CNN 的性能。

**[Words That Unite The World A Unified Framework For Deciphering Central Bank Comm](words_that_unite_the_world_a_unified_framework_for_deciphering_central_bank_comm.md)**

:   构建 World Central Banks (WCB) 数据集——迄今最全面的货币政策语料库（25家央行、38万句、28年历史），定义三项任务（立场检测、时间分类、不确定性估计），在 15,075 个基准实验中验证"整体大于部分之和"的跨央行聚合训练优势。
