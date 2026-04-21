---
title: >-
  NeurIPS2025 自监督方向 28篇论文解读
description: >-
  28篇NeurIPS2025 自监督方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督

**🧠 NeurIPS2025** · **28** 篇论文解读

**[A Joint Learning Approach to Hardware Caching and Prefetching](a_joint_learning_approach_to_hardware_caching_and_prefetching.md)**

:   提出将硬件缓存替换策略和预取策略进行联合训练的学习框架，通过共享编码器和对比学习两种方式构建共享特征表征，打破两个策略独立训练时的性能瓶颈。

**[Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)**

:   提出 Adv-SSL，通过将协方差正则项的 Frobenius 范数重写为 minimax 对偶形式，消除了 Barlow Twins 等方法中样本级风险的有偏估计问题，在不增加额外计算成本的前提下显著提升下游分类性能，并给出端到端的理论收敛保证。

**[Asymptotic and Finite-Time Guarantees for Langevin-Based Temperature Annealing in InfoNCE](asymptotic_and_finite-time_guarantees_for_langevin-based_temperature_annealing_i.md)**

:   本文通过将嵌入演化建模为紧致黎曼流形上的 Langevin 动力学，证明了经典模拟退火的收敛保证可以扩展到对比学习的温度调度设定中：缓慢对数逆温度调度保证概率收敛到全局最优表示集合，而更快的调度则可能陷入次优极小值。

**[BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)**

:   提出 BrainOmni——首个统一 EEG 和 MEG 的脑信号基础模型，通过 BrainTokenizer（含物理传感器编码器）将异构脑电/脑磁信号离散化为统一 token，再用 Criss-Cross Transformer 进行自监督掩码预测预训练，在阿尔茨海默病检测上提升 11.7 个百分点，并实现对完全未见设备的零样本重建泛化。

**[Connecting Jensen-Shannon and Kullback-Leibler Divergences: A New Bound for Representation Learning](connecting_jensenshannon_and_kullbackleibler_divergences_a_n.md)**

:   推导了一般情况下KL散度关于JS散度的最优紧致下界$\Xi(D_{\text{JS}}) \leq D_{\text{KL}}$，证明训练判别器最小化交叉熵损失等价于最大化互信息的一个保证下界，为JSD基于的判别式表示学习方法提供了缺失的理论基础，并在MI估计和Information Bottleneck框架中验证了紧致性与实用性。

**[Continuous Subspace Optimization for Continual Learning (CoSO)](continuous_subspace_optimization_for_continual_learning.md)**

:   提出 CoSO 框架，通过从每步梯度的 SVD 动态导出连续子空间（而非 LoRA 的固定子空间），结合历史任务正交投影防止干扰和 Frequent Directions 高效聚合梯度信息，在 ImageNet-R 20 任务上以 78.19% 最终准确率超越最佳 baseline 2.77 个百分点。

**[Contrastive Representations for Temporal Reasoning](contrastive_representations_for_temporal_reasoning.md)**

:   提出 CRTR（Contrastive Representations for Temporal Reasoning），通过在训练批次中重复同一轨迹来引入轨迹内负样本对，消除标准时间对比学习对静态上下文特征的依赖，学习到反映时间结构的表征，在魔方等组合推理任务上首次实现无搜索求解。

**[DataRater: Meta-Learned Dataset Curation](datarater_meta-learned_dataset_curation.md)**

:   提出 DataRater，一个基于元梯度（meta-gradient）的数据价值评估框架，通过元学习自动为每个训练数据点打分并过滤低质量数据，在多个预训练数据集上实现最高 46.6% 的净计算量节省，且在 400M 内部模型上训练的 DataRater 可直接泛化到 50M–1B 规模的 LLM 训练中。

**[Foundation Cures Personalization: Improving Personalized Models' Prompt Consistency via Hidden Foundation Knowledge](foundation_cures_personalization_improving_personalized_models_prompt_consistenc.md)**

:   FreeCure发现面部个性化模型的身份嵌入会覆盖但不破坏基础模型的prompt控制能力，据此提出无训练框架，通过Foundation-Aware Self-Attention（FASA）将基础模型的属性信息注入个性化生成过程，在保持身份保真度的同时大幅提升prompt一致性，可无缝集成到SD/SDXL/FLUX等主流模型。

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

**[Manifolds and Modules: How Function Develops in a Neural Foundation Model](manifolds_and_modules_how_function_develops_in_a_neural_foundation_model.md)**

:   从计算神经科学视角"打开黑箱"分析 SOTA 神经活动基础模型 (FNN)，通过构建解码流形和编码流形发现其各处理模块（编码器、循环、读出）展现出质性不同的表征结构，且与生物视觉系统存在关键差异。

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

**[Towards Reliable and Holistic Visual In-Context Learning Prompt Selection](towards_reliable_and_holistic_visual_in-context_learning_prompt_selection.md)**

:   提出RH-Partial2Global方法，首次用Spearman秩相关检验证明VICL中"相似性优先假设"虽统计显著但相关强度极弱($\bar{\rho} \approx 0.03\text{-}0.05$)，通过Jackknife共形预测构建可靠候选集+覆盖设计实现全面均匀的成对偏好采样，在分割/检测/着色三个视觉任务上一致超越SOTA。

**[TRIDENT: Tri-Modal Molecular Representation Learning with Taxonomic Annotations and Structural Relationships](trident_tri-modal_molecular_representation_learning_with_taxonomic_annotations_a.md)**

:   提出 TRIDENT 三模态分子表示学习框架，引入层次分类标注（HTA）作为第三模态，结合体积对比损失做全局三模态对齐和功能团-文本局部对齐，通过动量机制动态平衡两者，在 18 个分子属性预测任务上达到 SOTA。

**[Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)**

:   本文提出OligoICP方法，利用TabPFN模型的预测分位数间距（IQR）作为无标签模型选择启发式指标，在siRNA敲低效率预测中实现了优于专用SOTA模型和朴素集成的性能。

**[Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)**

:   本文首次将自监督学习（SSL）应用于冰晶图像的潜在表征学习，通过在大规模云粒子图像上预训练ViT，学习冰晶形态的连续潜在表征，并用vMF浓度参数量化冰晶多样性，实现30倍计算效率提升的同时取得最佳分类准确率84.39%。
