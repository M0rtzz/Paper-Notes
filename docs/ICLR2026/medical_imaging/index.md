---
title: >-
  ICLR2026 医学图像方向79篇论文解读
description: >-
  79篇ICLR2026的医学图像方向论文解读，涵盖医学影像、扩散模型、生物分子、LLM、推理、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "医学图像"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "扩散模型"
  - "生物分子"
  - "LLM"
  - "推理"
  - "少样本学习"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**🔬 ICLR2026** · **79** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (17)](../../ICML2026/medical_imaging/index.md) · [💬 ACL2026 (51)](../../ACL2026/medical_imaging/index.md) · [📷 CVPR2026 (119)](../../CVPR2026/medical_imaging/index.md) · [🤖 AAAI2026 (106)](../../AAAI2026/medical_imaging/index.md) · [🧠 NeurIPS2025 (147)](../../NeurIPS2025/medical_imaging/index.md) · [📹 ICCV2025 (38)](../../ICCV2025/medical_imaging/index.md)

🔥 **高频主题：** 医学影像 ×13 · 扩散模型 ×10 · 生物分子 ×9 · LLM ×5 · 推理 ×4

**[Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)**

:   提出CDTSDE框架，在扩散模型的逆向SDE中嵌入可学习的空间自适应域混合场 $\Lambda_t$，使跨模态翻译路径沿低能量流形前进，在MRI模态转换、SAR→光学、工业缺陷语义映射任务上以更少去噪步数实现更高保真度。

**[Adaptive Test-Time Training for Predicting Need for Invasive Mechanical Ventilation in Multi-Center Cohorts](adaptive_test-time_training_for_predicting_need_for_invasive_mechanical_ventilat.md)**

:   提出AdaTTT框架，通过动态特征感知self-supervised学习（自适应掩码策略）和原型引导的部分最优传输对齐，在ICU多中心EHR数据上实现鲁棒的测试时适应，用于提前24小时预测有创机械通气需求。

**[AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)**

:   构建了首个大规模抗体功能注释指令数据集AFD-Instruction（430K+条目），通过多智能体文献抽取pipeline对齐抗体序列与自然语言功能描述，用于指令微调通用LLM使其掌握抗体理解和功能导向设计能力，在5类分类任务上平均准确率提升20+点。

**[An Orthogonal Learner for Individualized Outcomes in Markov Decision Processes](an_orthogonal_learner_for_individualized_outcomes_in_markov_decision_processes.md)**

:   将因果推断中的半参数效率理论系统引入MDP的Q函数估计，证明经典的Q-regression和FQE本质上是有plug-in偏差的朴素学习器，并提出DRQQ-learner——一个同时具备双重鲁棒性、Neyman正交性和准oracle效率的元学习器，通过推导有效影响函数(EIF)构造去偏二阶段损失，在Taxi和Frozen Lake环境中全面超越基线方法。

**[AntigenLM: Structure-Aware DNA Language Modeling for Influenza](antigenlm_structure-aware_dna_language_modeling_for_influenza.md)**

:   AntigenLM 是一个保留基因组功能单元完整性的 GPT-2 风格 DNA 语言模型，通过在流感病毒全基因组上预训练并微调，能够自回归预测未来流行毒株的抗原序列，在氨基酸错配率上显著优于进化模型 beth-1 和通用基因组模型。

**[ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue.md)**

:   提出 ATPO（自适应树策略优化）算法，将多轮医疗对话建模为层级马尔可夫决策过程（H-MDP），通过不确定性感知的自适应树扩展机制动态分配rollout预算，结合Bellman误差和动作值方差的复合不确定性度量来引导探索，在三个医学对话基准上以Qwen3-8B超越GPT-4o。

**[Augmenting Representations with Scientific Papers](augmenting_representations_with_scientific_papers.md)**

:   提出首个将 X 射线光谱与科学文献通过对比学习对齐的多模态基础模型框架，在共享潜在空间中实现 20% Recall@1% 的跨模态检索，物理参数估计提升 16–18%，同时发现候选脉动超亮 X 射线源等罕见天体。

**[Benchmarking ECG FMs: A Reality Check Across Clinical Tasks](benchmarking_ecg_fms_a_reality_check_across_clinical_tasks.md)**

:   对8个ECG基础模型在12个数据集、26个临床任务上进行"现实检验"式全面基准评测，发现紧凑的结构化状态空间模型（SSM）ECG-CPC在7个任务类别中的5个上超越了大规模Transformer，证明架构设计比模型规模更重要。

**[Biologically Plausible Online Hebbian Meta-Learning: Two-Timescale Local Rules for Spiking Neural Brain Interfaces](biologically_plausible_online_hebbian_meta-learning_two-timescale_local_rules_fo.md)**

:   提出一种无需BPTT的在线SNN解码器，通过三因子Hebbian局部学习规则结合双时间尺度eligibility trace和自适应学习率控制，在O(1)内存下实现可比离线训练方法的BCI神经解码精度（Pearson R≥0.63/0.81），并在闭环仿真中展现了对神经信号非平稳性的持续适应能力。

**[BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases](biomedsql_text-to-sql_for_scientific_reasoning_on_biomedical_knowledge_bases.md)**

:   提出 BiomedSQL，首个专门评估 Text-to-SQL 系统在生物医学知识库上科学推理能力的基准，包含 68,000 个问题/SQL/答案三元组，揭示当前最强模型（GPT-o3-mini 62.6%）与领域专家（90%）之间仍有巨大差距。

**[Boosting Medical Visual Understanding From Multi-Granular Language Learning](boosting_medical_visual_understanding_from_multi-granular_language_learning.md)**

:   提出 Multi-Granular Language Learning (MGLL)，一个即插即用的对比学习框架，通过 soft CLIP loss、point-wise loss 和 smooth KL 散度联合优化，实现医学图像与多标签多粒度文本描述的对齐，在眼底和 X 光数据集上全面超越 SOTA 方法，并可作为视觉编码器嵌入多模态大语言模型提升诊断准确率最高达 34.1%。

**[Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer](brain-it_image_reconstruction_from_fmri_via_brain-interaction_transformer.md)**

:   提出 Brain-IT 框架，通过脑启发式的 Brain Interaction Transformer (BIT) 将功能相似的脑体素聚类为跨被试共享的 Brain Token，并从中预测局部化的语义和结构图像特征，实现从 fMRI 到图像的高保真重建，仅用 1 小时数据即达到先前方法 40 小时的性能。

**[Brain-Semantoks: Learning Semantic Tokens of Brain Dynamics with a Self-Distilled Foundation Model](brain-semantoks_learning_semantic_tokens_of_brain_dynamics_with_a_self-distilled.md)**

:   提出 Brain-Semantoks，一种基于语义分词器和自蒸馏目标的 fMRI 基础模型，将大脑功能网络聚合为鲁棒的语义 token，并通过跨时间视角的一致性学习抽象的脑动态表征，在线性探测设置下即可达到 SOTA 性能。

**[Bridging Explainability and Embeddings: BEE Aware of Spuriousness](bridging_explainability_and_embeddings_bee_aware_of_spuriousness.md)**

:   提出BEE框架，通过分析微调如何扰动预训练表征的权重空间几何结构，直接从分类器学到的权重中识别和命名虚假相关性（spurious correlations），无需反例样本即可发现隐藏的数据偏差，在ImageNet-1k上发现可导致准确率下降高达95%的虚假关联。

**[Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?](can_saes_reveal_and_mitigate_racial_biases_of_llms_in_healthcare.md)**

:   研究稀疏自编码器（SAE）能否揭示和缓解 LLM 在医疗场景中的种族偏见：发现 SAE 能识别出与种族相关的有害联想（如黑人与暴力），但在复杂临床任务中缓解偏见的效果有限（FLDD < 3%），远不如简单的提示策略（FLDD 8-15%）。

**[CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework](care_towards_clinical_accountability_in_multi-modal_medical_reasoning_with_an_ev.md)**

:   提出 CARE 框架——将医学 VQA 拆分为"实体提议→指称分割→证据引导问答"三阶段专家管道，用 RLVR 微调各 VLM，并引入 GPT-5 作为动态协调器进行工具规划与 CoT 审查，在 4 个医学 VQA 基准上以 10B 参数量（77.54% 平均准确率）超越 32B 端到端 SOTA（72.29%）。

**[Causal Interpretation of Neural Network Computations with Contribution Decomposition](causal_interpretation_of_neural_network_computations_with_contribution_decomposi.md)**

:   提出 CODEC（Contribution Decomposition），用 Integrated Gradients 计算隐藏层神经元对输出的贡献（而非仅分析激活），再用 Sparse Autoencoder 将贡献分解为稀疏模式（modes），实现比激活分析更强的因果可解释性和网络控制能力，并成功应用于 ResNet-50 和视网膜生物神经网络模型。

**[Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space](characterizing_human_semantic_navigation_in_concept_production_as_trajectories_i.md)**

:   提出将人类概念产生过程建模为 Transformer 嵌入空间中的累积轨迹，定义 5 个运动学指标（距离、速度、加速度、熵、质心距离），在 4 个数据集（3 种语言、神经退行性疾病/脏话流畅性/属性列举）上成功区分临床组和概念类别，且不同嵌入模型产生高度一致的结果。

**[COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)**

:   COMPASS 通过在分割网络的中间特征空间沿**对目标度量最敏感的低维子空间**进行线性扰动来构建 conformal prediction 区间，在四个医学分割任务上实现了比传统 CP 方法显著更窄的预测区间，同时保持有效覆盖率。

**[ConfHit: Conformal Generative Design with Oracle Free Guarantees](confhit_conformal_generative_design_with_oracle_free_guarantees.md)**

:   提出 ConfHit 框架，利用密度比加权的共形排列 p 值实现"认证"（判断生成批次是否包含 hit）和"设计"（精简候选集同时保持统计保证），在无需实验验证 oracle 和存在分布偏移的条件下，为生成式分子设计提供有限样本 $1-\alpha$ 覆盖保证。

**[Controllable Sequence Editing for Biological and Clinical Trajectories](controllable_sequence_editing_for_biological_and_clinical_trajectories.md)**

:   提出 Clef，一个基于"时间概念"（temporal concepts）的可控序列编辑模型，能够在给定条件（如药物、手术）下对生物/临床多变量轨迹进行即时和延迟编辑，在细胞重编程和患者实验室检测数据上，即时编辑 MAE 提升 16.28%，延迟编辑提升 26.73%，零样本反事实生成提升达 62.84%。

**[Controlling Repetition in Protein Language Models](controlling_repetition_in_protein_language_models.md)**

:   首次系统性研究蛋白质语言模型（PLM）中的病态重复问题，提出统一的重复度量指标 $R(x)$ 和效用指标 $U(x)$，并设计 UCCS（Utility-Controlled Contrastive Steering）方法，通过在隐层注入与重复解耦的引导向量，在不重训模型的前提下有效抑制重复同时保持折叠可信度。

**[CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](counselbench_llm_mental_health_qa.md)**

:   联合100名持证心理健康专家构建CounselBench双组件基准——CounselBench-EVAL（2,000条六维度专家评估）和CounselBench-Adv（120个对抗性问题+1,080条响应标注），系统性揭示LLM在心理健康开放式问答中表面得分高但存在过度泛化、擅自医疗建议等安全隐患，同时证明LLM-as-Judge在安全关键领域严重不可靠。

**[CryoNet.Refine: A One-step Diffusion Model for Rapid Refinement of Structural Models with Cryo-EM Density Map Restraints](cryonetrefine_a_one-step_diffusion_model_for_rapid_refinement_of_structural_mode.md)**

:   提出CryoNet.Refine——首个基于AI的冷冻电镜(cryo-EM)原子模型精修框架：设计单步扩散模型(初始化自Boltz-2权重)→创新可微分密度生成器(物理模拟合成密度图)→首次将密度图相关性作为可微损失函数(余弦相似度)→联合Ramachandran/Rotamer/键角等几何约束损失→测试时优化策略逐案定制→在120个蛋白质/DNA-RNA复合物上全面超越Phenix.real_space_refine(CC_mask 0.59 vs 0.54, Ramachandran favored 98.92%)。

**[Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series](decentralized_attention_fails_centralized_signals_rethinking_transformers_for_me.md)**

:   提出 TeCh 框架，核心是用 CoTAR（Core Token Aggregation-Redistribution）模块替代 Transformer 中的标准注意力来建模医学时间序列的通道依赖——通过引入全局"核心 token"充当代理，先聚合所有通道信息再重分配回每个通道，复杂度从 $O(n^2)$ 降至 $O(n)$，在 APAVA 数据集上精度 86.86%（超 Medformer 12.13%），内存仅 33%、推理时间仅 20%。

**[Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models](deep_hierarchical_learning_with_nested_subspace_networks_for_large_language_mode.md)**

:   提出嵌套子空间网络（NSN），通过低秩分解使线性层形成严格嵌套的子空间层次，配合不确定性感知多秩训练，使单个模型在测试时可即时调节计算量与性能的权衡（50% FLOPs 减少仅损失 5% 精度），且可后验应用于预训练 LLM。

**[DISCO: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring](disco_densely-overlapping_cell_instance_segmentation_via_adjacency-aware_collabo.md)**

:   将密集重叠细胞实例分割建模为图着色问题，提出"显式标记冲突节点 + 隐式邻接约束消歧"的分治框架 Disco，通过 BFS 分解细胞邻接图并引入五种协同损失函数，在高密度病理数据集 GBC-FS 2025 上 PQ 提升 7.08%，同时在四个异质数据集上均取得 SOTA。

**[Discrete Diffusion Trajectory Alignment via Stepwise Decomposition](discrete_diffusion_trajectory_alignment_via_stepwise_decomposition.md)**

:   提出 SDPO（Stepwise Decomposition Preference Optimization），将离散扩散模型的轨迹对齐问题分解为逐步后验对齐子问题，避免了在整条去噪链上反传梯度的困难，在 DNA 序列设计、蛋白质逆折叠和语言建模三个任务上均显著超越现有方法。

**[DistMLIP: A Distributed Inference Platform for Machine Learning Interatomic Potentials](distmlip_a_distributed_inference_platform_for_machine_learning_interatomic_poten.md)**

:   提出 DistMLIP 分布式推理平台，基于零冗余图级并行化策略（graph-level parallelization），解决现有机器学习原子间势（MLIP）缺乏多 GPU 支持的问题，在 8 GPU 上实现接近百万原子的模拟，比空间分区方法快达 8 倍且能模拟 3.4 倍更大的系统。

**[Distributional Consistency Loss: Beyond Pointwise Data Terms in Inverse Problems](distributional_consistency_loss_beyond_pointwise_data_terms_in_inverse_problems.md)**

:   提出分布一致性（DC）损失，用分布级别的校准替代传统逐点数据保真项（如MSE/NLL），避免对噪声的过拟合，在DIP去噪和PET图像重建中显著提升性能且无需早停。

**[DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)**

:   提出DM4CT——首个系统性的CT重建扩散模型基准，涵盖十种扩散方法和七种基线方法，在医疗、工业和同步辐射三类数据集上进行全面评估，揭示了扩散模型在CT重建中的优势与局限。

**[DriftLite: Lightweight Drift Control for Inference-Time Scaling of Diffusion Models](driftlite_lightweight_drift_control_for_inference-time_scaling_of_diffusion_mode.md)**

:   DriftLite 提出在 Fokker-Planck 方程中利用漂移-势函数的自由度，通过轻量级线性系统求解最优控制漂移来主动稳定粒子权重，以最小代价解决 Sequential Monte Carlo 中的权重退化问题，在高斯混合、分子系统和蛋白质-配体共折叠任务上大幅超越 Guidance-SMC 基线。

**[Dual Distillation for Few-Shot Anomaly Detection](dual_distillation_for_few-shot_anomaly_detection.md)**

:   提出双蒸馏框架 D24FAD，结合 query 图像上的教师-学生蒸馏（TSD）和 support 图像上的学生自蒸馏（SSD），辅以学习权重机制（L2W）自适应评估 support 重要性，在 APTOS 眼底数据集上仅用 2-shot 达到 100% AUROC。

**[EMR-AGENT: Automating Cohort and Feature Extraction from EMR Databases](emr-agent_automating_cohort_and_feature_extraction_from_emr_databases.md)**

:   提出EMR-AGENT，首个基于LLM Agent的电子病历（EMR）自动化预处理框架，通过动态SQL交互替代手工规则编写，实现跨数据库的队列选择、特征提取和代码映射，在MIMIC-III/eICU/SICdb上表现优异并具强泛化能力。

**[EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering](evoflows_evolutionary_edit-based_flow-matching_for_protein_engineering.md)**

:   EvoFlows 提出一种基于编辑操作的 Flow Matching 方法，通过学习进化相关蛋白质序列间的突变轨迹，能在模板序列上执行可控数量的突变（插入、删除、替换），同时预测"突变什么"和"在哪里突变"。

**[Exo-Plore: Exploring Exoskeleton Control Space through Human-Aligned Simulation](exo-plore_exploring_exoskeleton_control_space_through_human-aligned_simulation.md)**

:   提出 Exo-plore 框架，通过神经力学仿真与深度强化学习相结合，无需真人实验即可优化髋关节外骨骼控制参数，并能推广到病理步态场景。

**[Exploiting Low-Dimensional Manifold of Features for Few-Shot Whole Slide Image Classification](exploiting_low-dimensional_manifold_of_features_for_few-shot_whole_slide_image_c.md)**

:   发现病理基础模型特征具有低维流形几何结构（有效秩仅29.7/512维），而线性层会破坏这种结构导致少样本过拟合，提出即插即用的MR Block（冻结随机矩阵做几何锚+低秩残差路径做任务适配）在少样本WSI分类上达到SOTA。

**[Fine-Tuning Diffusion Models via Intermediate Distribution Shaping](fine-tuning_diffusion_models_via_intermediate_distribution_shaping.md)**

:   统一拒绝采样微调方法为GRAFT框架并证明其隐式执行KL正则化奖励最大化，进而提出P-GRAFT在中间去噪步骤做分布整形（偏差-方差权衡更优），以及Inverse Noise Correction无需奖励即可改进流模型质量，在T2I上VQAScore提升8.81%。

**[From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents](from_conversation_to_query_execution_benchmarking_user_and_tool_interactions_for.md)**

:   提出EHR-ChatQA基准，首次评估数据库Agent在电子病历场景中的端到端交互工作流（澄清模糊查询→解决术语不匹配→生成SQL→返回答案），发现最强模型(o4-mini)的Pass@5超90%但Pass∧5(全部成功)大幅下降(差距达60%)，暴露了安全关键领域的鲁棒性缺陷。

**[Fusing Pixels and Genes: Spatially-Aware Learning in Computational Pathology](fusing_pixels_and_genes_spatially-aware_learning_in_computational_pathology.md)**

:   本文提出 Stamp 框架，利用空间转录组学基因表达数据作为监督信号，通过空间感知基因编码器预训练和层次化多尺度对比对齐，实现病理图像与空间转录组数据的联合表示学习，在 6 个数据集 4 个下游任务上取得 SOTA。

**[Glance and Focus Reinforcement for Pan-cancer Screening](glance_and_focus_reinforcement_for_pan-cancer_screening.md)**

:   提出 GF-Screen 两阶段框架——轻量 Glance 模型用强化学习快速定位含病灶的 CT 子体积，Focus 模型只对选中区域做精细分割；通过将 GRPO 的"组内相对比较"思想从 NLP 迁移到视觉子体积组，首次在纯视觉任务中实现无价值网络的 RL 优化，在 FLARE25 泛癌挑战中以 +25.6% DSC 大幅领先冠军方案且推理快 5.7 倍。

**[HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction](histoprism_unlocking_functional_pathway_analysis_from_pan-cancer_histology_via_g.md)**

:   本文提出 HistoPrism，一个高效的 Transformer 架构，通过交叉注意力注入癌症类型条件来从 H&E 病理图像预测泛癌基因表达，并提出基于 Hallmark/GO 通路的 Gene Pathway Coherence (GPC) 评估框架，在通路级别预测上大幅超越 STPath，尤其在低方差核心生物通路上优势显著。

**[How to Make the Most of Your Masked Language Model for Protein Engineering](how_to_make_the_most_of_your_masked_language_model_for_protein_engineering.md)**

:   提出基于温度退火随机束搜索（SBS）的MLM采样方法，利用伪似然的野生型边际近似实现高效全序列评估，在真实抗体治疗优化的体外实验中证明采样算法选择至少与模型选择同等重要，SBS+引导达到100%成功率。

**[Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding](human_behavior_atlas_benchmarking_unified_psychological_and_social_behavior_unde.md)**

:   构建 Human Behavior Atlas——首个覆盖情感、认知、病理和社会过程四大维度的大规模多模态行为理解统一基准（101K+ 样本），并训练三种 OmniSapiens-7B 模型变体验证其在多任务训练和迁移学习中的有效性。

**[Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity](improving_2d_diffusion_models_for_3d_medical_imaging_with_inter-slice_consistent.md)**

:   提出 Inter-Slice Consistent Stochasticity (ISCS)，通过球面线性插值(Slerp)在扩散采样的 re-noising 步骤中生成层间相关噪声，从根源消除 2D 扩散先验做 3D 医学重建时的层间不连续伪影——零额外计算/超参数/训练开销，即插即用到任何 2D 扩散逆问题求解器，在稀疏视角 CT、限角 CT 和 MRI 超分辨率上均持续提升。

**[Incentives in Federated Learning with Heterogeneous Agents](incentives_in_federated_learning_with_heterogeneous_agents.md)**

:   从博弈论视角分析异构联邦学习中的激励问题，证明在异构数据分布和 PAC 准确率目标下纯策略纳什均衡的存在性，并提出基于线性规划的近似算法来确定最优贡献量。

**[Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification](inference-time_dynamic_modality_selection_for_incomplete_multimodal_classificati.md)**

:   提出DyMo——推理时动态模态选择框架，通过理论推导将多模态任务相关信息增益转化为可计算的MTIR奖励函数（基于分类损失降低代理 + 类原型距离 + 类内相似性校准），在推理时迭代选择性融合可靠的恢复模态，首次系统性解决"丢弃缺失模态损失信息 vs 补全可能引入噪声"的困境。

**[Intrinsic Lorentz Neural Network](intrinsic_lorentz_neural_network.md)**

:   提出完全内禀（fully intrinsic）的双曲神经网络 ILNN，所有运算均在 Lorentz 模型内完成，消除了现有方法中混合欧几里得操作的几何不一致性，在图像分类、基因组学和图分类上取得 SOTA。

**[Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine](knowledgeable_language_models_as_black-box_optimizers_for_personalized_medicine.md)**

:   提出 LEON（LLM-based Entropy-guided Optimization with kNowledgeable priors），一种数学原理严格的方法，将个性化医疗治疗方案设计建模为条件黑箱优化问题，通过熵约束和对抗性源批评模型引导 LLM 在不微调的情况下作为零样本优化器提出个性化治疗计划。

**[LaVCa: LLM-assisted Visual Cortex Captioning](lavca_llm-assisted_visual_cortex_captioning.md)**

:   提出 LaVCa 方法，利用 LLM 为人类视觉皮层的每个体素生成自然语言描述（caption），通过"编码模型→最优图像选取→MLLM生成描述→LLM关键词提炼+句子组合"四步流程，比已有方法 BrainSCUBA 更准确、更多样地揭示了体素级视觉选择性。

**[Learning Domain-Aware Task Prompt Representations for Multi-Domain All-in-One Image Restoration](learning_domain-aware_task_prompt_representations_for_multi-domain_all-in-one_im.md)**

:   提出首个多域全能图像复原方法DATPRL-IR，通过双提示池（任务提示池+域提示池）学习域感知的任务提示表征，利用MLLM蒸馏域先验并通过自适应门控融合指导复原，在自然/医学/遥感三域9任务上显著超越SOTA。

**[Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation](learning_patient-specific_disease_dynamics_with_latent_flow_matching_for_longitu.md)**

:   提出 Δ-LFM 框架：用 ArcRank 损失在潜在空间构建患者特异性时间对齐轨迹（角度一致 + 幅度单调递增），将流匹配时间范围从 [0,1] 扩展到 [0,T] 实际时间间隔实现任意时间点预测，在三个阿尔茨海默纵向 MRI 基准上全面超越 8 种基线方法，并提出进展专用指标 Δ-RMAE。

**[mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules](mclm_a_modular_chemical_language_model_that_generates_functional_and_makeable_mo.md)**

:   提出 mCLM（模块化化学语言模型），通过将分子表示为可合成构建模块的序列，使 LLM 能生成同时满足药理功能和自动化合成可行性的分子，在 430 种 FDA 批准药物上显著改善了药代动力学和毒性性质。

**[MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science](medagentgym_agentic_training_biomedical.md)**

:   构建了首个统一的生物医学数据科学 Agent 训练环境 MedAgentGym，包含 72,413 个任务实例（覆盖 12 个真实场景、129 个类别），配备可执行沙盒和可验证 ground truth，系统基准评估 29 个 LLM 揭示商业/开源差距，并通过高效多线程轨迹采样 + 离线/在线 RL 训练出 Med-Copilot，分别获得 +43.02%/+45.28% 提升，达到与 GPT-4o 竞争的性能。

**[MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](mmedagent-rl_optimizing_multi-agent_collaboration_for_multimodal_medical_reasoni.md)**

:   提出 MMedAgent-RL，通过 RL 优化模拟临床会诊流程（分诊→专科→主治）的多智能体系统，核心创新是课程学习引导的熵感知 RL（C-MARL），让主治医师智能体在面对正确/冲突/错误的专科意见时分别采取不同的探索-利用策略，在域内外共 5 个医学 VQA 基准上实现 SOTA。

**[Neural Synchrony Between Socially Interacting Language Models](neural_synchrony_between_socially_interacting_language_models.md)**

:   首次研究社会交互中 LLM 间的神经同步现象：通过训练仿射变换预测交互伙伴的未来表征，定义 $SyncR^2$ 指标量化同步强度，发现该同步依赖于社会参与和时间邻近性，且与 LLM 的社会行为表现高度相关（Pearson $r$ = 0.88-0.99），呼应了人类脑间同步（IBS）的神经科学发现。

**[Neuro-Symbolic Decoding of Neural Activity](neuro-symbolic_decoding_of_neural_activity.md)**

:   提出 NEURONA，一个神经符号框架用于 fMRI 解码和概念基础，通过将视觉场景分解为符号程序（概念的逻辑组合），在 fMRI 问答任务上显著优于端到端神经解码和线性模型。

**[NeuroCircuitry-Inspired Hierarchical Graph Causal Attention Networks for Explainable Depression Identification](neurocircuitry-inspired_hierarchical_graph_causal_attention_networks_for_explain.md)**

:   提出 NH-GCAT 框架，将神经科学中的抑郁症神经环路先验知识显式融入 GNN，在区域、环路和网络三个空间尺度上建模，在 REST-meta-MDD 数据集上取得 SOTA 分类效果（AUC 78.5%、ACC 73.8%），并提供与神经科学相符的可解释性分析。

**[Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research](omni-ieeg_a_large-scale_comprehensive_ieeg_dataset_and_benchmark_for_epilepsy_re.md)**

:   本文构建了 Omni-iEEG 数据集（302 名患者、178 小时高分辨率颅内脑电记录），定义了基于临床先验的标准化基准任务和评估指标，并展示端到端建模在癫痫手术规划中可匹配或超越传统生物标志物方法。

**[Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling](overthinking_reduction_with_decoupled_rewards_and_curriculum_data_scheduling.md)**

:   从理论上揭示了现有长度惩罚方法的两个根本缺陷——错误惩罚高熵探索token和错误奖励冗余token，提出 DeCS 框架，通过解耦token级奖励和课程批次调度，在7个基准上将推理token减少50%以上同时保持甚至提升模型性能。

**[Protein as a Second Language for LLMs](protein_as_a_second_language_for_llms.md)**

:   将氨基酸序列视为 LLM 的"第二语言"，通过构建蛋白质-自然语言双语数据集和自适应上下文构造机制，无需任何训练即可让通用 LLM 在蛋白质问答任务上平均提升 7% ROUGE-L，最高 17.2%，甚至超越领域专用微调模型。

**[Protein Counterfactuals via Diffusion-Guided Latent Optimization](protein_counterfactuals_via_diffusion-guided_latent_optimization.md)**

:   提出MCCOP框架，在蛋白质的连续序列-结构联合潜空间中，利用预训练扩散模型作为流形先验进行梯度引导的反事实优化，以最少突变（2-3个）生成生物学可信的蛋白质变体来翻转预测器输出，同时实现模型解释和蛋白质设计假说生成。

**[Protein Structure Tokenization via Geometric Byte Pair Encoding](protein_structure_tokenization_via_geometric_byte_pair_encoding.md)**

:   提出 GeoBPE——首个将 BPE（字节对编码）从离散文本扩展到连续蛋白质骨架几何的 tokenizer，通过交替执行"局部合并（k-medoids聚类+量化）"和"全局校正（可微逆运动学）"构建层次化结构 motif 词汇表，以 >10× 压缩比和 >10× 数据效率超越 VQ-VAE 类 PST，在 12 个下游任务 24 个测试集上排名第一。

**[Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering](q-fsru_quantum-augmented_frequency-spectral_for_medical_visual_question_answerin.md)**

:   提出 Q-FSRU 框架，通过 FFT 将医学图像和文本特征变换到频率域进行融合，并引入量子启发的检索增强机制（Quantum RAG）从外部知识库中获取医学事实，在 VQA-RAD 数据集上取得 90.0% 准确率。

**[Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis](resp-agent_an_agent-based_system_for_multimodal_respiratory_sound_generation_and.md)**

:   提出 Resp-Agent 闭环多智能体框架，通过主动对抗课程规划器（Thinker-A2CA）协调可控呼吸音生成器与多模态诊断器，在 229k 规模基准上实现生成↔诊断协同设计，大幅提升长尾类别诊断性能。

**[Reverse Distillation: Consistently Scaling Protein Language Model Representations](reverse_distillation_consistently_scaling_protein_language_model_representations.md)**

:   针对蛋白质语言模型（PLM）"模型越大性能不一定越好"的反常缩放现象，提出反向蒸馏框架：以小模型表示为基底、用SVD提取大模型正交残差信息，构造Matryoshka嵌套嵌入，使得更大的反向蒸馏模型一致优于更小的，ESM-2 15B经反向蒸馏后首次成为全家族最强。

**[Scalable Spatio-Temporal SE(3) Diffusion for Long-Horizon Protein Dynamics](scalable_spatio-temporal_se3_diffusion_for_long-horizon_protein_dynamics.md)**

:   提出 STAR-MD，一个 SE(3) 等变的因果扩散 Transformer，通过联合时空注意力和上下文噪声扰动实现微秒级蛋白质动力学轨迹生成，在 ATLAS 基准上所有指标达到 SOTA，且能稳定外推到训练中未见的微秒时间尺度。

**[Scaling with Collapse: Efficient and Predictable Training of LLM Families](scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)**

:   证明 LLM 家族的训练损失曲线在优化超参数与数据预算匹配时会“崩塞”到同一条通用曲线上，并利用这一现象实现两个实用应用：(1) 偏离崩塞作为训练病理的早期诊断信号，(2) 崩塞曲线的可预测性实现大规模超参调优的早停。

**[SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)**

:   提出 SEED（Semantic Evaluation for Visual Brain Decoding），一个结合 Object F1、Cap-Sim 和 EffNet 三个互补指标的组合评估度量，在与人类评估的对齐度上显著超越现有所有指标。

**[Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People](shoot_first_ask_questions_later_building_rational_agents_that_explore_and_act_li.md)**

:   提出 Collaborative Battleship 任务评估语言模型的信息搜索能力，设计三种贝叶斯推断策略（Bayes-Q/M/D）增强 LM 的提问、行动和决策能力，使弱模型（Llama-4-Scout）以 GPT-5 约 1% 的成本达到超人表现（82% 胜率）。

**[SONIC: Spectral Oriented Neural Invariant Convolutions](sonic_spectral_oriented_neural_invariant_convolutions.md)**

:   SONIC 将状态空间模型的思想迁移到多维频域，用 6 个连续参数（幅度、方向、阻尼、振荡等）定义一组方向选择性的频谱传递函数，再通过低秩矩阵 $B$、$C$ 跨通道混合，实现天然具备全局感受野和分辨率不变性的卷积替代算子，在 3D 医学分割上匹配 nnU-Net 且参数少近两个数量级，在 ImageNet 上也具有竞争力。

**[SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis](survhte-bench_a_benchmark_for_heterogeneous_treatment_effect_estimation_in_survi.md)**

:   提出 SurvHTE-Bench，首个面向右删失生存数据的异质处理效应（HTE）估计综合基准，涵盖 40 个合成数据集、10 个半合成数据集和 2 个真实数据集，系统评估了 53 种估计方法在不同因果假设违反和删失水平下的表现，发现没有单一方法占主导地位，生存 meta-learner（特别是 S-Learner-Survival 和 Matching-Survival）在高删失和假设违反场景下表现最为稳健。

**[SynCoGen: Synthesizable 3D Molecule Generation via Joint Reaction and Coordinate Modeling](syncogen_synthesizable_3d_molecule_generation_via_joint_reaction_and_coordinate_.md)**

:   SynCoGen 提出了一种结合掩码图扩散和流匹配的多模态生成框架，能够同时采样分子构建块反应图和3D原子坐标，在保证合成可行性的同时实现高质量的3D分子生成。

**[Thompson Sampling via Fine-Tuning of LLMs](thompson_sampling_via_fine-tuning_of_llms.md)**

:   提出 ToSFiT，通过微调大语言模型直接参数化最大概率（Probability of Maximality），将 Thompson Sampling 扩展到大规模非结构化离散空间，避免了获取函数最大化的难题。

**[Towards Interpretable Visual Decoding with Attention to Brain Representations](towards_interpretable_visual_decoding_with_attention_to_brain_representations.md)**

:   提出 NeuroAdapter，将 fMRI 信号按脑区分割为独立 token 并通过交叉注意力直接条件化 Stable Diffusion，跳过传统的 CLIP/DINO 中间嵌入空间，在 NSD 等数据集上高层语义指标超越或持平现有方法，同时引入 IBBI 双向可解释性框架，首次动态揭示不同皮层区域在去噪轨迹中如何驱动图像生成。

**[Tracing Pharmacological Knowledge in Large Language Models](tracing_pharmacological_knowledge_in_large_language_models.md)**

:   首次系统性地对生物医学 LLM 中药物分组语义的编码机制进行因果分析，发现药物组知识存储在早期层、分布在多个 token 上（非最后一个 token），线性可分的语义信息在嵌入层即已存在。

**[Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct](ultra-fast_language_generation_via_discrete_diffusion_divergence_instruct.md)**

:   提出 DiDi-Instruct，一种基于积分 KL 散度 (IKL) 最小化的蒸馏框架，将预训练的扩散大语言模型 (dLLM) 蒸馏为少步学生模型，通过对抗性密度比估计 + 分组奖励归一化 + 分数分解 + 奖励引导祖先采样器 (RGAS) 四大关键设计，在 OpenWebText 上仅用 16 步即超越 1024 步教师模型的 PPL，实现最高 64× 推理加速，同时训练成本仅需 1 GPU 小时。

**[Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)**

:   Uni-NTFM 从神经科学第一性原理出发，设计异质特征投影（HFPM）解耦时频编码、分层拓扑嵌入（TE）统一异构电极配置、MoE Transformer 实现功能模块化与稀疏编码，在 28000 小时 EEG 数据上预训练 1.9B 参数模型，9 个下游任务上的线性探测和微调均达到 SOTA。

**[Unified Biomolecular Trajectory Generation via Pretrained Variational Bridge](unified_biomolecular_trajectory_generation_via_pretrained_variational_bridge.md)**

:   PVB（Pretrained Variational Bridge）通过编码器-解码器架构结合增强桥匹配，统一了单结构预训练和配对轨迹微调的训练目标，实现了跨领域生物分子轨迹生成，并通过RL微调加速蛋白质-配体holo态探索。
