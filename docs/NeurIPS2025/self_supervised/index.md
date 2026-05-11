---
title: >-
  NeurIPS2025 自监督/表示学习方向35篇论文解读
description: >-
  35篇NeurIPS2025的自监督/表示学习方向论文解读，涵盖自监督学习、持续学习、对抗鲁棒、推理、少样本学习、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**🧠 NeurIPS2025** · **35** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (2)](../../ACL2026/self_supervised/index.md) · [📷 CVPR2026 (31)](../../CVPR2026/self_supervised/index.md) · [🔬 ICLR2026 (15)](../../ICLR2026/self_supervised/index.md) · [🤖 AAAI2026 (14)](../../AAAI2026/self_supervised/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/self_supervised/index.md) · [🧪 ICML2025 (24)](../../ICML2025/self_supervised/index.md)

🔥 **高频主题：** 自监督学习 ×5 · 持续学习 ×2

**[A Joint Learning Approach to Hardware Caching and Prefetching](a_joint_learning_approach_to_hardware_caching_and_prefetching.md)**

:   提出将硬件缓存替换策略和预取策略进行联合训练的学习框架，通过共享编码器和对比学习两种方式构建共享特征表征，打破两个策略独立训练时的性能瓶颈。

**[Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)**

:   提出 Adv-SSL，通过将协方差正则项的 Frobenius 范数重写为 minimax 对偶形式，消除了 Barlow Twins 等方法中样本级风险的有偏估计问题，在不增加额外计算成本的前提下显著提升下游分类性能，并给出端到端的理论收敛保证。

**[Angular Constraint Embedding via SpherePair Loss for Constrained Clustering](angular_constraint_embedding_via_spherepair_loss_for_constrained_clustering.md)**

:   本文提出SpherePair损失函数，通过在角度空间（而非欧几里得空间）进行成对约束嵌入学习，实现了不依赖锚点(anchor)、不需要预知聚类数的深度约束聚类方法，并提供了严格的理论保证来确定最优超参数。

**[Asymptotic and Finite-Time Guarantees for Langevin-Based Temperature Annealing in InfoNCE](asymptotic_and_finite-time_guarantees_for_langevin-based_temperature_annealing_i.md)**

:   本文通过将嵌入演化建模为紧致黎曼流形上的 Langevin 动力学，证明了经典模拟退火的收敛保证可以扩展到对比学习的温度调度设定中：缓慢对数逆温度调度保证概率收敛到全局最优表示集合，而更快的调度则可能陷入次优极小值。

**[BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)**

:   提出 BrainOmni——首个统一 EEG 和 MEG 的脑信号基础模型，通过 BrainTokenizer（含物理传感器编码器）将异构脑电/脑磁信号离散化为统一 token，再用 Criss-Cross Transformer 进行自监督掩码预测预训练，在阿尔茨海默病检测上提升 11.7 个百分点，并实现对完全未见设备的零样本重建泛化。

**[Connecting Jensen-Shannon and Kullback-Leibler Divergences: A New Bound for Representation Learning](connecting_jensenshannon_and_kullbackleibler_divergences_a_n.md)**

:   推导了一般情况下KL散度关于JS散度的最优紧致下界$\Xi(D_{\text{JS}}) \leq D_{\text{KL}}$，证明训练判别器最小化交叉熵损失等价于最大化互信息的一个保证下界，为JSD基于的判别式表示学习方法提供了缺失的理论基础，并在MI估计和Information Bottleneck框架中验证了紧致性与实用性。

**[Continuous Subspace Optimization for Continual Learning (CoSO)](continuous_subspace_optimization_for_continual_learning.md)**

:   提出 CoSO 框架，通过从每步梯度的 SVD 动态导出连续子空间（而非 LoRA 的固定子空间），结合历史任务正交投影防止干扰和 Frequent Directions 高效聚合梯度信息，在 ImageNet-R 20 任务上以 78.19% 最终准确率超越最佳 baseline 2.77 个百分点。

**[Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)**

:   提出 Task-Modulated Contrastive Learning (TMCL)，受大脑新皮层自顶向下调制启发，在持续学习中通过 affine modulation 集成稀疏标签信息（仅需 1% 标签），再利用对比学习将调制信息固化到前馈权重中，在 class-incremental 和迁移学习上超越无监督和有监督基线。

**[Contrastive Representations for Temporal Reasoning](contrastive_representations_for_temporal_reasoning.md)**

:   提出 CRTR（Contrastive Representations for Temporal Reasoning），通过在训练批次中重复同一轨迹来引入轨迹内负样本对，消除标准时间对比学习对静态上下文特征的依赖，学习到反映时间结构的表征，在魔方等组合推理任务上首次实现无搜索求解。

**[Curiosity-driven RL for Symbolic Equation Solving](curiosity-driven_rl_for_symbolic_equation_solving.md)**

:   将好奇心驱动探索（RND、ICM 等）与基于表达式树的图动作空间结合，使 PPO 智能体能够求解包含根号、指数和三角函数的非线性方程，超越了此前仅限于线性方程的 RL 方法。

**[DataRater: Meta-Learned Dataset Curation](datarater_meta-learned_dataset_curation.md)**

:   提出 DataRater，一个基于元梯度（meta-gradient）的数据价值评估框架，通过元学习自动为每个训练数据点打分并过滤低质量数据，在多个预训练数据集上实现最高 46.6% 的净计算量节省，且在 400M 内部模型上训练的 DataRater 可直接泛化到 50M–1B 规模的 LLM 训练中。

**[Disentangling Hyperedges through the Lens of Category Theory](disentangling_hyperedges_through_the_lens_of_category_theory.md)**

:   首次从范畴论视角分析超边解耦，基于自然性条件导出"因子表示一致性"标准（聚合后解耦 vs 解耦后聚合应一致），提出 Natural-HNN 模型在6个癌症分型数据集上全面超越14个baseline（BRCA F1 从75.7%提升至80.4%），并能100%正确捕获基因通路的功能上下文。

**[Foundation Models for Scientific Discovery: From Paradigm Enhancement to Paradigm Transition](foundation_models_for_scientific_discovery_from_paradigm_enhancement_to_paradigm.md)**

:   提出三阶段框架（元科学整合→混合人机共创→自主科学发现）来描绘基础模型正推动科学范式从工具增强向范式转型演变的图景，并系统综述了 FM 在实验/理论/计算/数据四大科学范式中的整合应用。

**[Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings](hybrid_autoencoders_for_tabular_data_leveraging_model-based_augmentation_in_low-.md)**

:   提出 TANDEM（Tree-And-Neural Dual Encoder Model），一种混合自编码器架构，通过联合训练神经网络编码器和遗忘软决策树（OSDT）编码器，并引入样本级随机门控网络作为可学习的数据增强，在低标签表格数据场景下实现了超越强基线（包括树模型和深度学习方法）的性能。

**[Implicit Modeling for Transferability Estimation of Vision Foundation Models](implicit_modeling_for_transferability_estimation_of_vision_foundation_models.md)**

:   提出隐式可迁移性建模（ITM）框架，通过隐变量z隐式编码模型-任务对的迁移能力，结合分治变分近似（DVA）高效模拟嵌入空间演化，在10个下游任务和10个多样化预训练模型上的加权Kendall tau_w从此前最优的0.45提升至0.61。

**[Know Thyself by Knowing Others: Learning Neuron Identity from Population Context](know_thyself_by_knowing_others_learning_neuron_identity_from_population_context.md)**

:   提出NuCLR自监督框架，通过对比学习对群体神经活动中同一神经元的不同时间窗口拉近、不同神经元推远，学习包含群体上下文的神经元级表征，在细胞类型和脑区解码上达到新SOTA，并首次展示了跨动物零样本泛化和数据缩放规律。

**[Long-Tailed Recognition via Information-Preservable Two-Stage Learning](long-tailed_recognition_via_information-preservable_two-stage_learning.md)**

:   提出信息保持的两阶段学习框架：第一阶段用 Balanced Negative Sampling (BNS) 基于互信息最大化学习有效且可分的特征空间，第二阶段用 Information-Preservable DPP (IP-DPP) 采样数学上信息量最大的样本来纠正多数类偏向的决策边界，在多个长尾数据集上取得 SOTA。

**[M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](m-grpo_stabilizing_self-supervised_reinforcement_learning_for_multimodal_underst.md)**

:   针对自监督强化学习（SS-RLVR）在长期训练中普遍出现的"策略崩溃"问题，提出 M-GRPO：通过动量模型提供稳定的伪标签目标 + 基于四分位距（IQR）的低熵轨迹过滤防止熵崩溃，在无标注 MATH 数据集上训练 Qwen3-4B-Base，最终 checkpoint 即超越 SRT 手动选取的最佳 checkpoint，AIME24 +2.92%、GPQA +5.05%。

**[Manifolds and Modules: How Function Develops in a Neural Foundation Model](manifolds_and_modules_how_function_develops_in_a_neural_foundation_model.md)**

:   从计算神经科学视角"打开黑箱"分析 SOTA 神经活动基础模型 (FNN)，通过构建解码流形和编码流形发现其各处理模块（编码器、循环、读出）展现出质性不同的表征结构，且与生物视觉系统存在关键差异。

**[Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks](memory-integrated_reconfigurable_adapters_a_unified_framework_for_settings_with_.md)**

:   MIRA 将 Hopfield 式联想记忆模块嵌入 ViT 各层，以键值对方式存储和检索 LoRA 适配器权重，通过两阶段训练（适应+巩固），在一个统一架构下同时解决领域泛化（DG）、类增量学习（CIL）和域增量学习（DIL）三类任务，在多个基准上显著超过各任务的专用方法。

**[Minimal Semantic Sufficiency Meets Unsupervised Domain Generalization](minimal_semantic_sufficiency_meets_unsupervised_domain_generalization.md)**

:   MS-UDG 在无类别标签和域标签的条件下，通过信息解纠缠模块（IDM）将表征分解为语义和变异成分，配合最小语义充分性优化模块（SROM）最大化语义信息同时最小化变异干扰，在 PACS 上达 72.89% 准确率（+1.5% vs CycleMAE），理论证明最小充分语义表征最小化下游贝叶斯错误率。

**[Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models](mitra_mixed_synthetic_priors_for_enhancing_tabular_foundation_models.md)**

:   首次系统研究合成先验的设计原则，发现多样性、独特性和真实数据对齐是关键属性，据此提出 Mitra——一个基于精心筛选的混合合成先验训练的表格基础模型，在分类和回归基准上一致超越 TabPFNv2 和 TabICL。

**[One Filters All: A Generalist Filter for State Estimation](one_filters_all_a_generalist_filter_for_state_estimation.md)**

:   提出 LLM-Filter，将 LLM 重编程为通用状态估计器，通过 System-as-Prompt（SaP）机制使冻结的 LLM 在未见动力系统上实现零样本泛化，性能超越 SOTA 学习型滤波器。

**[SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)**

:   提出 SEAL 框架，利用自然存在的语义层级结构（而非手工设计的抽象层级）指导广义类别发现，通过层级语义引导的软对比学习和跨粒度一致性模块，在细粒度基准上取得 SOTA 性能。

**[Soft Task-Aware Routing of Experts for Equivariant Representation Learning](soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)**

:   提出 STAR（Soft Task-Aware Routing），通过 MoE 路由机制协调不变性和等变性表示学习任务间的共享与专属信息，减少冗余特征学习，提升下游任务迁移性能。

**[STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)**

:   提出 STaRFormer，通过动态注意力区域掩码（DAReM）识别任务关键区域并施加掩码扰动，配合批内+类内半监督对比学习将任务信息嵌入潜在表示，在 56 个数据集（含非平稳、不规则采样、分类/异常检测/回归）上全面超越 SOTA。

**[T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)**

:   提出 T-REGS——一种基于最小生成树(MST)长度最大化的自监督学习正则化框架，理论证明可同时防止维度坍缩并促进表示分布均匀性，在紧致黎曼流形上成立，实验在标准 JE-SSL 基准上验证了有效性。

**[TabArena: A Living Benchmark for Machine Learning on Tabular Data](tabarena_a_living_benchmark_for_machine_learning_on_tabular_data.md)**

:   提出 TabArena，首个持续维护的"活跃"表格数据基准系统，从 1053 个数据集中精选 51 个、纳入 16 个模型，通过大规模实验（约 2500 万次模型训练）发现：后验集成下深度学习模型已追平甚至超越 GBDT，表格基础模型在小数据上表现突出，跨模型集成可进一步推进 SOTA。

**[TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)**

:   提出 TabSTAR，一个专为含文本字段的表格数据设计的基础模型：通过解冻文本编码器（e5-small-v2）端到端优化文本表征 + 目标感知 token 注入分类目标语义信息 + 无数据集特定参数的架构实现跨数据集迁移学习，在 350 个数据集上预训练后，分类任务上 14 个数据集中 12 个超越 CatBoost-Tuned（4h 调参），8/11 超越 TabPFN-v2。

**[The Complexity of Finding Local Optima in Contrastive Learning](the_complexity_of_finding_local_optima_in_contrastive_learning.md)**

:   证明对比学习中寻找局部最优是计算困难的：离散三元组最大化问题是 PLS-hard（即使 $d=1$），连续三元组损失最小化是 CLS-hard，意味着（在标准假设下）不存在多项式时间算法找到局部最优。

**[Towards Reliable and Holistic Visual In-Context Learning Prompt Selection](towards_reliable_and_holistic_visual_in-context_learning_prompt_selection.md)**

:   提出RH-Partial2Global方法，首次用Spearman秩相关检验证明VICL中"相似性优先假设"虽统计显著但相关强度极弱($\bar{\rho} \approx 0.03\text{-}0.05$)，通过Jackknife共形预测构建可靠候选集+覆盖设计实现全面均匀的成对偏好采样，在分割/检测/着色三个视觉任务上一致超越SOTA。

**[TRIDENT: Tri-Modal Molecular Representation Learning with Taxonomic Annotations and Structural Relationships](trident_tri-modal_molecular_representation_learning_with_taxonomic_annotations_a.md)**

:   提出 TRIDENT 三模态分子表示学习框架，引入层次分类标注（HTA）作为第三模态，结合体积对比损失做全局三模态对齐和功能团-文本局部对齐，通过动量机制动态平衡两者，在 18 个分子属性预测任务上达到 SOTA。

**[Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)**

:   本文提出OligoICP方法，利用TabPFN模型的预测分位数间距（IQR）作为无标签模型选择启发式指标，在siRNA敲低效率预测中实现了优于专用SOTA模型和朴素集成的性能。

**[Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)**

:   本文首次将自监督学习（SSL）应用于冰晶图像的潜在表征学习，通过在大规模云粒子图像上预训练ViT，学习冰晶形态的连续潜在表征，并用vMF浓度参数量化冰晶多样性，实现30倍计算效率提升的同时取得最佳分类准确率84.39%。

**[You Can Trust Your Clustering Model: A Parameter-free Self-Boosting Plug-in for Deep Clustering](you_can_trust_your_clustering_model_a_parameter-free_self-boosting_plug-in_for_d.md)**

:   提出 DCBoost，一个无需额外超参数的即插即用模块，通过自适应 k-NN 筛选高置信样本并利用可靠的局部结构信息引导全局特征空间优化，显著提升现有深度聚类模型的性能。
