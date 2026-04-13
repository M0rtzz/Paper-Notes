---
title: >-
  NeurIPS2025 AI安全方向 116篇论文解读
description: >-
  116篇NeurIPS2025 AI安全方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI安全

**🧠 NeurIPS2025** · 共 **116** 篇

**[A Set Of Generalized Components To Achieve Effective Poison-Only Clean-Label Bac](a_set_of_generalized_components_to_achieve_effective_poison-only_clean-label_bac.md)**

:   提出一组通用化组件（Component A/B/C），通过充分挖掘样本选择与触发器之间的双向协作关系，同时提升 Poison-only Clean-label 后门攻击的攻击成功率（ASR）和隐蔽性，并在多种攻击类型上展现了良好的泛化能力。

**[Adaptive Lora Experts Allocation And Selection For Federated Fine-Tuning](adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)**

:   提出 FedLEASE——解决联邦 LoRA 微调中两个关键问题：(1) 用 LoRA B 矩阵相似度聚类自动确定最优专家数量和分配，(2) 用扩展路由空间（$2M-1$ 维）实现自适应 top-M 专家选择（每个客户端自动决定用几个专家），在 GLUE 上比最强基线平均提升 5.53%。

**[Adversarial Paraphrasing A Universal Attack For Humanizing Ai-Generated Text](adversarial_paraphrasing_a_universal_attack_for_humanizing_ai-generated_text.md)**

:   提出 Adversarial Paraphrasing——一种无需训练的通用攻击框架，在逐 token 改写时利用 AI 文本检测器的反馈信号选择"最像人写"的 token，使改写后的 AI 文本在 8 种检测器上平均 T@1%F 下降 87.88%，且具有跨检测器的强迁移性。

**[Agentstealth Reinforcing Large Language Model For Anonymizing User-Generated Tex](agentstealth_reinforcing_large_language_model_for_anonymizing_user-generated_tex.md)**

:   提出 AgentStealth 框架，通过对抗式匿名化工作流、监督微调（SFT）和在线强化学习三阶段训练小型语言模型（SLM），实现在保持文本效用的同时有效匿名化用户生成内容，匿名化效果提升12.3%、效用提升6.8%。

**[Ai Should Sense Better Not Just Scale Bigger Adaptive Sensin](ai_should_sense_better_not_just_scale_bigger_adaptive_sensin.md)**

:   提出"自适应感知"作为AI发展的范式级转变——受生物感觉系统启发，主张在传感器层面动态调整输入参数（如曝光、增益、多模态配置），而非仅靠扩大模型规模来应对分布偏移，实证表明5M参数的EfficientNet-B0通过自适应感知可超越632M参数的OpenCLIP-H。

**[Almguard Safety Shortcuts And Where To Find Them As Guardrails For Audio-Languag](almguard_safety_shortcuts_and_where_to_find_them_as_guardrails_for_audio-languag.md)**

:   首个针对音频语言模型（ALM）越狱攻击的防御框架——发现对齐过的 ALM 存在可被激活的潜在安全快捷路径（safety shortcuts），通过 Mel 梯度稀疏掩码（M-GSM）定位关键频率段，施加快捷路径激活扰动（SAP），将平均攻击成功率从 41.6% 降至 4.6%，同时几乎不影响正常任务性能。

**[Beyond Last-Click An Optimal Mechanism For Ad Attribution](beyond_last-click_an_optimal_mechanism_for_ad_attribution.md)**

:   从博弈论角度分析广告归因中 Last-Click 机制的策略操纵漏洞——平台可以通过篡改时间戳获取不公正的归因信用，提出 Peer-Validated Mechanism（PVM）——每个平台的信用仅取决于其他平台的报告（类比同行评审），理论证明 PVM 是占优策略激励兼容（DSIC）且在同质设置下最优，准确率从 34% 提升到 75%（2 平台）。

**[Bits Leaked Per Query Information-Theoretic Bounds On Adversarial Attacks Agains](bits_leaked_per_query_information-theoretic_bounds_on_adversarial_attacks_agains.md)**

:   将 LLM 对抗攻击建模为信息通道问题——定义每次查询的"泄漏比特数" $I(Z;T)$ 为攻击目标属性 $T$ 与可观测信号 $Z$ 的互信息，证明攻击达到误差 $\varepsilon$ 所需最少查询数为 $\log(1/\varepsilon)/I(Z;T)$，在 7 个 LLM 上验证：暴露 answer tokens 需 ~1000 次查询，加 logits 降到 ~100 次，加思维链降到 ~几十次，为透明性-安全性权衡提供首个原则性标尺。

**[Boosting Adversarial Transferability With Spatial Adversarial Alignment](boosting_adversarial_transferability_with_spatial_adversarial_alignment.md)**

:   提出 Spatial Adversarial Alignment (SAA)，通过空间感知对齐和对抗感知对齐两个模块微调代理模型，使其特征与见证模型对齐，从而显著提升对抗样本的跨架构迁移性（CNN→ViT 迁移率提升 25-39%）。

**[Bridging Symmetry And Robustness On The Role Of Equivariance In Enhancing Advers](bridging_symmetry_and_robustness_on_the_role_of_equivariance_in_enhancing_advers.md)**

:   通过在 CNN 中嵌入旋转等变（P4群）和尺度等变卷积层，提出 Parallel 和 Cascaded 两种对称性感知架构，无需对抗训练即可显著提升对抗鲁棒性，并从 CLEVER 框架出发理论证明等变架构能压缩假设空间、正则化梯度、收紧认证鲁棒性界。

**[Causally Reliable Concept Bottleneck Models](causally_reliable_concept_bottleneck_models.md)**

:   提出 C2BM（Causally reliable Concept Bottleneck Models），将概念瓶颈（concept bottleneck）按照因果图结构化组织，通过结合观测数据与背景知识自动学习因果关系，在保持分类精度的同时显著提升因果可靠性、干预响应和公平性。

**[Collective Narrative Grounding Community-Coordinated Data Contributions To Impro](collective_narrative_grounding_community-coordinated_data_contributions_to_impro.md)**

:   提出 Collective Narrative Grounding 协议，通过参与式工坊收集社区叙事并结构化为"叙事单元"，用 RAG 管道将本地知识注入 LLM 问答系统，在 LocalBench 上发现 76.7% 的错误可由本地叙事直接修复，GPT-5 在参与式 QA 集上仅 21% 正确率凸显了本地知识鸿沟。

**[Contextual Integrity In Llms Via Reasoning And Reinforcement Learning](contextual_integrity_in_llms_via_reasoning_and_reinforcement_learning.md)**

:   提出 CI-RL 框架，通过 Chain-of-Thought 推理提示 + GRPO 强化学习，用仅约 700 个合成样本训练 LLM 理解"上下文完整性"（contextual integrity），在 PrivacyLens 基准上将隐私泄露率降低最高 40%，且小模型训练后可超越更大基线模型。

**[Coreguard Safeguarding Foundational Capabilities Of Llms Against Model Stealing ](coreguard_safeguarding_foundational_capabilities_of_llms_against_model_stealing_.md)**

:   提出 CoreGuard，通过行置换（row permutation）锁定 Transformer 线性层权重 + 列置换传播协议（propagation protocol）将 TEE 授权次数降至 1 次，以极低计算和通信开销保护边缘部署 LLM 的基础能力不被模型窃取攻击利用。

**[Cost Efficient Fairness Audit Under Partial Feedback](cost_efficient_fairness_audit_under_partial_feedback.md)**

:   在部分反馈（partial feedback）设定下，提出了一套包含新颖成本模型的公平性审计框架，分别在黑盒与混合模型两种场景给出近最优审计算法，审计成本比自然基线降低约 50%。

**[Cpret A Dataset Benchmark And Model For Retrieval In Competitive Programming](cpret_a_dataset_benchmark_and_model_for_retrieval_in_competitive_programming.md)**

:   针对竞赛编程中重复/相似题目泛滥导致比赛不公平及 LLM 评测分数虚高的问题，构建了包含四种检索任务的大规模基准 CPRet，并提出 Group-InfoNCE 损失训练的专用检索模型 CPRetriever，在所有任务上超越 20+ 现有嵌入模型，同时揭示了题目相似性对 LiveCodeBench 评测的系统性偏差。

**[Cryptomoe Privacy-Preserving And Scalable Mixture Of Experts Inference Via Balan](cryptomoe_privacy-preserving_and_scalable_mixture_of_experts_inference_via_balan.md)**

:   首个支持 MoE 架构 LLM 隐私推理的框架 CryptoMoE，通过平衡专家路由隐藏路由信息、置信度感知调度协议和批量密文矩阵乘法协议，相比 dense baseline 实现 2.8~3.5× 延迟降低和 2.9~4.3× 通信量降低，准确率损失仅 0.8%。

**[Ctrl-Alt-Deceit Sabotage Evaluations For Automated Ai Rd](ctrl-alt-deceit_sabotage_evaluations_for_automated_ai_rd.md)**

:   扩展 MLE-Bench 构建了 20 个代码破坏(code-sabotage)任务和 sandbagging 评测，发现前沿 AI agent 能在完成正常 ML 工程任务的同时成功植入后门等破坏，且在部分情况下逃避 LM monitor 的检测。

**[Deeppersona A Generative Engine For Scaling Deep Synthetic Personas](deeppersona_a_generative_engine_for_scaling_deep_synthetic_personas.md)**

:   提出 DeepPersona——一个两阶段分类引导的合成人格生成引擎：先从真实用户-ChatGPT 对话中挖掘构建 8000+ 节点的人类属性分类树，再通过渐进式属性采样生成平均 200+ 结构化属性的叙事完整人格，在个性化 QA 准确率上提升 11.6%，社会调查模拟偏差缩小 31.7%。

**[Design Encrypted Gnn Inference Via Server-Side Input Graph Pruning](design_encrypted_gnn_inference_via_server-side_input_graph_pruning.md)**

:   提出 DESIGN 框架，在全同态加密(FHE)下通过服务器端输入图剪枝和自适应多项式激活度分配两阶段优化，相比 SEAL 基线加速 FHE GNN 推理约 2× 并维持有竞争力的准确率。

**[Dictpfl Efficient And Private Federated Learning On Encrypted Gradients](dictpfl_efficient_and_private_federated_learning_on_encrypted_gradients.md)**

:   提出 DictPFL 框架，通过将模型权重分解为静态字典+可训练查找表，并结合加密感知剪枝，在联邦学习中实现全梯度同态加密保护的同时，将通信开销降低 402–748 倍、训练速度提升 28–65 倍，运行时间仅为明文 FL 的 2 倍以内。

**[Differential Privacy For Euclidean Jordan Algebra With Applications To Private S](differential_privacy_for_euclidean_jordan_algebra_with_applications_to_private_s.md)**

:   提出了基于 Euclidean Jordan Algebra (EJA) 的通用 Gaussian 隐私机制，并在此基础上设计了首个差分隐私的 Symmetric Cone Programming (SCP) 求解算法，解决了 Hsu et al. (ICALP 2014) 提出的关于差分隐私半定规划的重要开放问题。

**[Differentially Private Bilevel Optimization Efficient Algorithms With Near-Optim](differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)**

:   本文系统研究差分隐私 (DP) 下的双层优化问题，在凸情形下通过指数机制和正则化指数机制给出近紧的上下界（匹配单层 DP-ERM 最优率），在非凸情形下提出二阶 DP 方法实现不依赖内层维度的 SOTA 收敛率。

**[Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)**

:   提出FedASK框架，通过**双阶段sketching流水线**（randomized SVD启发），首次在差分隐私下实现联邦LoRA中**两个低秩矩阵A和B的同步有效更新**，在Llama-2 7B/13B上MMLU提升最高11.5%，GSM8K提升46%。

**[Differentially Private High-Dimensional Variable Selection Via Integer Programmi](differentially_private_high-dimensional_variable_selection_via_integer_programmi.md)**

:   本文提出两种纯差分隐私的稀疏变量选择方法 (top-R 和 mistakes)，利用现代混合整数规划 (MIP) 技术高效探索非凸目标景观，在高维设置（p 达 10000）下实现 SOTA 支撑集恢复率，同时提供理论恢复保证。

**[Distributional Adversarial Attacks And Training In Deep Hedging](distributional_adversarial_attacks_and_training_in_deep_hedging.md)**

:   本文首次将分布对抗攻击引入深度对冲框架，提出基于 Wasserstein 球的可计算对抗训练方法（WPGD 和 WBPGD），显著提升了对冲策略在分布偏移和真实市场数据下的鲁棒性与样本外表现。

**[Distributive Fairness In Large Language Models Evaluating Alignment With Human V](distributive_fairness_in_large_language_models_evaluating_alignment_with_human_v.md)**

:   本文系统评估多个 SOTA LLM（GPT-4o、Claude-3.5S、Llama3-70b、Gemini-1.5P）在非策略性资源分配任务中的分配公平性偏好，发现 LLM 与人类存在显著偏差：LLM 偏好效率和无嫉妒性 (EF) 而忽视人类更看重的公平性/平等性 (EQ)，但在选择题模式下 GPT-4o 和 Claude 能正确识别公平方案。

**[Dna-Detectllm Unveiling Ai-Generated Text Via A Dna-Inspired Mutation-Repair Par](dna-detectllm_unveiling_ai-generated_text_via_a_dna-inspired_mutation-repair_par.md)**

:   本文提出 DNA-DetectLLM，一种受 DNA 突变修复机制启发的零样本 AI 文本检测方法，通过构造理想 AI 序列并量化将输入文本修复到该序列的累积难度作为检测信号，在多个基准数据集上取得 AUROC 相对提升 5.55%、F1 提升 2.08% 的 SOTA 效果。

**[Dual-Flow Transferable Multi-Target Instance-Agnostic Attacks Via In-The-Wild Ca](dual-flow_transferable_multi-target_instance-agnostic_attacks_via_in-the-wild_ca.md)**

:   本文提出 Dual-Flow 框架，利用预训练扩散模型的正向 ODE 流和微调 LoRA 速度函数的逆向流进行多目标实例无关对抗攻击，通过级联分布偏移训练策略显著提升迁移攻击成功率（从 Inc-v3 到 Res-152 成功率提升 34.58%），在防御模型上也表现出强鲁棒性。

**[Efficient Fairness-Performance Pareto Front Computation](efficient_fairness-performance_pareto_front_computation.md)**

:   提出 MIFPO 方法，无需训练复杂的公平表示模型即可高效计算公平性-性能 Pareto 前沿，通过理论分析将问题化简为紧凑的离散凹优化问题。

**[Efficient Verified Machine Unlearning For Distillation](efficient_verified_machine_unlearning_for_distillation.md)**

:   提出 PURGE 框架，通过教师-学生 constituent mapping 和增量式多教师蒸馏策略，将 SISA 的验证式遗忘扩展到知识蒸馏场景，在教师端遗忘时仅需部分重训学生模型，实现至少 $N\times$ 的加速。

**[Enabling Differentially Private Federated Learning For Speech Recognition Benchm](enabling_differentially_private_federated_learning_for_speech_recognition_benchm.md)**

:   首次为端到端ASR建立FL+DP的实用基准，通过**逐层裁剪（per-layer clipping）**结合**LAMB优化器**的层级梯度归一化，在强隐私保证下实现仅1.3%~4.6%的WER绝对退化。

**[Enhancing Clip Robustness Via Crossmodality Alignment](enhancing_clip_robustness_via_crossmodality_alignment.md)**

:   提出COLA——一个training-free的框架，通过将对抗扰动后的图像特征投影到文本特征张成的子空间来消除非语义噪声，再用最优传输(OT)在分布层面细粒度对齐图文特征，在14个零样本分类基准上平均提升6.7%的对抗鲁棒准确率，同时维持干净样本性能。

**[Enhancing Graph Classification Robustness With Singular Pooling](enhancing_graph_classification_robustness_with_singular_pooling.md)**

:   首次系统分析 flat pooling（Sum/Avg/Max）对图分类对抗鲁棒性的影响，推导各自的对抗风险上界，并提出 RS-Pool——利用节点嵌入矩阵的主奇异向量构建图级表示，在不牺牲 clean accuracy 的前提下显著提升对抗鲁棒性。

**[Environment Inference For Learning Generalizable Dynamical System](environment_inference_for_learning_generalizable_dynamical_system.md)**

:   提出 DynaInfer 框架，通过分析固定神经网络的预测误差来推断未标注轨迹的环境标签，实现无环境标签条件下的动态系统泛化学习，在 ODE/PDE 系统上性能匹配甚至超越 Oracle（已知标签）。

**[Evaluating The Promise And Pitfalls Of Llms In Hiring Decisions](evaluating_the_promise_and_pitfalls_of_llms_in_hiring_decisions.md)**

:   在约 10,000 个真实招聘候选人-职位配对上系统评测了 GPT-4o/4.1、Claude 3.5、Gemini 2.5、Llama 3.1/4、DeepSeek R1 等主流 LLM 的招聘匹配表现，发现专用领域模型 Match Score 在准确性（AUC 0.85 vs 0.77）和公平性（种族 IR 0.957 vs ≤0.809）上全面优于通用 LLM。

**[Exploring The Limits Of Strong Membership Inference Attacks On Large Language Mo](exploring_the_limits_of_strong_membership_inference_attacks_on_large_language_mo.md)**

:   首次将强成员推断攻击（LiRA）扩展到10M~1B参数的GPT-2规模LLM，训练超过4000个参考模型，揭示四个关键发现：强MIA可以在LLM上成功但效果有限（AUC<0.7），且大量个体样本决策在训练随机性下**与抛硬币无法区分**。

**[Factor Decorrelation Enhanced Data Removal From Deep Predictive Models](factor_decorrelation_enhanced_data_removal_from_deep_predictive_models.md)**

:   提出 DecoRemoval 框架，通过判别性保持的因子去相关（基于随机傅里叶特征的空间映射+自适应权重）和平滑损失扰动两大模块，在不重训的前提下实现数据移除，尤其在分布外（OOD）场景下显著优于现有方法。

**[Fair Minimum Labeling Efficient Temporal Network Activations For Reachability An](fair_minimum_labeling_efficient_temporal_network_activations_for_reachability_an.md)**

:   本文提出公平最小标注（FML）问题，旨在设计最小代价的时序边激活方案，使网络中各节点组均有足够的时序路径可达性以满足公平覆盖要求；证明该问题是 NP-hard 且难以近似，并基于概率树嵌入给出匹配下界的近似算法。

**[Fair Representation Learning With Controllable High Confidence Guarantees Via Ad](fair_representation_learning_with_controllable_high_confidence_guarantees_via_ad.md)**

:   提出 FRG（Fair Representation learning with high-confidence Guarantees），首个允许用户指定公平性阈值 $\varepsilon$ 和置信水平 $1-\delta$ 的公平表征学习框架：通过 VAE 候选选择 + 对抗推断最大化协方差 + Student's t-检验构造高置信上界，保证对**任意**下游模型和任务，$\Delta_{DP} \leq \varepsilon$ 以至少 $1-\delta$ 概率成立。

**[Faircontrast Enhancing Fairness Through Contrastive Learning And Customized Augm](faircontrast_enhancing_fairness_through_contrastive_learning_and_customized_augm.md)**

:   FairContrast 提出一种面向表格数据的公平对比学习框架，通过策略性的正对样本选择（将优势组有利结果样本与对应弱势组样本配对），结合有监督或自监督对比损失与交叉熵损失的端到端训练，在不引入额外公平约束损失的前提下显著降低了预测偏差，且精度损失极小。

**[Fairness-Regularized Online Optimization With Switching Costs](fairness-regularized_online_optimization_with_switching_costs.md)**

:   提出 FairOBD 算法，首次在平滑在线凸优化中同时处理**长期公平性正则项**和**切换代价**，通过引入辅助变量分解长期公平代价并用镜像下降更新对偶变量，证明了渐近竞争比保证。

**[Fairness Under Competition](fairness_under_competition.md)**

:   本文首次研究竞争环境下多个公平分类器的联合公平性问题，理论证明即使每个分类器都满足 Equal Opportunity (EO)，生态系统可能仍然不公平，且对偏差分类器进行公平性调整反而可能降低生态系统公平性。

**[Fedfact A Provable Framework For Controllable Group-Fairness Calibration In Fede](fedfact_a_provable_framework_for_controllable_group-fairness_calibration_in_fede.md)**

:   提出FedFACT框架，通过刻画联邦学习下的**贝叶斯最优公平分类器**结构，将公平联邦学习分别在训练中（in-processing）化归为**个性化代价敏感学习**、在训练后（post-processing）化归为**双层优化**，首次实现多类别场景下全局公平性与局部公平性的可控协调，并提供收敛及泛化保证。

**[Fedrw Efficient Privacy-Preserving Data Reweighting For Enhancing Federated Lear](fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)**

:   FedRW 提出首个无需可信第三方的联邦学习隐私保护软去重框架，通过安全多方计算获取全局样本频率并进行频率感知的样本加权，在预处理上实现最高 28.78× 加速，在模型性能上实现约 11.42% 的 perplexity 改善。

**[Fedsvd Adaptive Orthogonalization For Private Federated Learning With Lora](fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)**

:   FedSVD 提出通过 SVD 对 LoRA 矩阵进行全局重参数化，在每轮通信后用聚合的 BA 乘积的右奇异向量更新 A 矩阵，避免 DP-SGD 下的二次噪声放大同时保持 A 的自适应能力，在多个 NLU 基准上一致超越固定 A 的基线。

**[Flux Efficient Descriptor-Driven Clustered Federated Learning Under Arbitrary Di](flux_efficient_descriptor-driven_clustered_federated_learning_under_arbitrary_di.md)**

:   提出Flux——基于描述符驱动聚类的联邦学习框架，通过提取隐私保护的客户端数据描述符（分布统计量的矩近似）和无监督密度聚类，自动处理四种分布偏移（特征/标签/P(Y|X)/P(X|Y)），在CheXpert医疗数据集上测试时精度比最佳基线高14.6pp。

**[Forensichub A Unified Benchmark Codebase For All-Domain Fake Image Detection And](forensichub_a_unified_benchmark_codebase_for_all-domain_fake_image_detection_and.md)**

:   ForensicHub 提出首个统一所有域（Deepfake/IMDL/AIGC/文档篡改）的假图检测与定位基准平台，包含 4 个任务、23 个数据集、42 个模型、6 个骨干网络和 11 个 GPU 加速评估指标，通过模块化架构和适配器设计打破领域孤岛，并进行了 16 种跨域评估得出 8 条关键洞察。

**[Geo-Sign Hyperbolic Contrastive Regularisation For Geometrically Aware Sign Lang](geo-sign_hyperbolic_contrastive_regularisation_for_geometrically_aware_sign_lang.md)**

:   Geo-Sign 提出将骨架特征投影到 Poincaré 球模型的双曲空间中，通过双曲对比损失正则化 mT5 语言模型，使其感知手语运动的层次结构，仅用骨架数据就在 CSL-Daily 上超越了基于 RGB 的 SOTA 方法（BLEU-4 +1.81, ROUGE-L +3.03）。

**[Healthslm-Bench Benchmarking Small Language Models For Mobile And Wearable Healt](healthslm-bench_benchmarking_small_language_models_for_mobile_and_wearable_healt.md)**

:   首个系统评估小语言模型 (SLMs, 1-4B参数) 在移动与可穿戴健康监测任务上表现的基准，覆盖zero-shot/few-shot/指令微调三种范式，并在iPhone上验证了端侧部署的可行性。

**[Impact Of Dataset Properties On Membership Inference](impact_of_dataset_properties_on_membership_inference.md)**

:   本文理论推导并实验验证了深度迁移学习中成员推理攻击（MIA）脆弱性与每类样本数之间的幂律关系 $\log(\text{tpr}-\text{fpr}) = -\beta_S \log(S) - \beta_0$，发现增加数据量可降低平均和最坏情况脆弱性，但保护最脆弱样本需要极大量数据。

**[Impact Of Dataset Properties On Membership Inference Vulnerability Of Deep Trans](impact_of_dataset_properties_on_membership_inference_vulnerability_of_deep_trans.md)**

:   从理论和实验两方面揭示深度迁移学习中成员推断攻击 (MIA) 脆弱性与每类样本数之间存在幂律关系：随着每类样本数 $S$ 增加，MIA 优势按 $S^{-1/2}$ 下降，但保护最脆弱样本所需的数据量极大，凸显了差分隐私形式化保障的不可替代性。

**[Improved Balanced Classification With Theoretically Grounded Loss Functions](improved_balanced_classification_with_theoretically_grounded_loss_functions.md)**

:   提出两个理论驱动的代理损失函数族——广义Logit调整(GLA)损失和广义类感知加权(GCA)损失，为类别不平衡下的多类分类提供更强的理论保证和实证性能。

**[Incentivizing Time-Aware Fairness In Data Sharing](incentivizing_time-aware_fairness_in_data_sharing.md)**

:   提出了一个时间感知的数据共享框架，设计了新的激励机制（F6-F8）和两种奖励方案（时间感知奖励累计和时间感知数据估值），保证早加入协作的参与方能获得更高价值的奖励，同时兼顾公平性和个体理性。

**[Influence Functions For Edge Edits In Non-Convex Graph Neural Networks](influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)**

:   提出适用于非凸 GNN 的边编辑影响函数，通过 proximal Bregman 响应函数放松凸性假设，并同时考虑参数偏移和消息传播两方面的影响，支持边的删除和插入。

**[Invisibleink High-Utility And Low-Cost Text Generation With Differential Privacy](invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)**

:   提出 InvisibleInk 框架，通过差分裁剪（DClip）隔离敏感信息和 Top-$k^+$ 截断采样两项创新，将差分隐私长文本生成的计算成本降低 8 倍以上，首次实现不到非隐私生成 4-8 倍开销的高质量隐私文本生成。

**[Its Complicated The Relationship Of Algorithmic Fairness And Non-Discrimination ](its_complicated_the_relationship_of_algorithmic_fairness_and_non-discrimination_.md)**

:   系统分析了欧盟 AI 法案（EU AI Act）中高风险系统的非歧视条款与机器学习算法公平性概念之间的关系，指出两者存在显著脱节并提出未来标准化方向。

**[Keep It Real Challenges In Attacking Compression-Based Adversarial Purification](keep_it_real_challenges_in_attacking_compression-based_adversarial_purification.md)**

:   本文系统评估了基于图像压缩的对抗净化防御，发现重建图像的"真实感"（realism）是提升防御鲁棒性的关键因素——高真实感压缩模型在面对强自适应攻击时仍能保持显著鲁棒性，而这并非源于梯度掩蔽。

**[Learning-Augmented Facility Location Mechanisms For Envy Ratio](learning-augmented_facility_location_mechanisms_for_envy_ratio.md)**

:   针对一维设施选址问题中的**嫉妒比**（envy ratio）目标，设计了确定性和随机化的学习增强机制：确定性的 $\alpha$-BIM 机制在一致性和鲁棒性之间实现最优权衡，随机化的BAM机制进一步改善保证；同时解决了Ding等人提出的公开问题，将无预测的随机机制近似比从2改进至约1.8944。

**[Llm Strategic Reasoning Agentic Study Through Behavioral Gam](llm_strategic_reasoning_agentic_study_through_behavioral_gam.md)**

:   论文不再把大模型战略推理简单等同于“是否接近纳什均衡”，而是基于 behavioral game theory 构建评测框架，区分真实推理能力与上下文因素，系统测评 22 个 LLM 的互动决策行为，发现模型规模并不决定战略水平，CoT 提升也并非普遍有效，同时暴露出显著的人口属性偏置。

**[Locally Optimal Private Sampling Beyond The Global Minimax](locally_optimal_private_sampling_beyond_the_global_minimax.md)**

:   在本地差分隐私（LDP）下的采样问题中，提出**局部minimax**框架，利用公共数据分布 $P_0$ 定义的邻域约束，推导出闭式最优采样器，在理论和实验上均**一致优于全局minimax采样器**。

**[Machine Unlearning Doesnt Do What You Think Lessons For Generative Ai Policy And](machine_unlearning_doesnt_do_what_you_think_lessons_for_generative_ai_policy_and.md)**

:   系统性论证"机器遗忘"（machine unlearning）并非万能方案，识别出遗忘动机与可行实现之间的五大根本错配（mismatches），为 ML 研究者和政策制定者提供严谨的分析框架。

**[Mars A Malignity-Aware Backdoor Defense In Federated Learning](mars_a_malignity-aware_backdoor_defense_in_federated_learning.md)**

:   提出 MARS 防御方法，通过计算神经元的后门能量（Backdoor Energy）来感知模型的恶意程度，并利用 Wasserstein 距离聚类有效识别联邦学习中的后门模型。

**[Masksql Safeguarding Privacy For Llm-Based Text-To-Sql Via Abstraction](masksql_safeguarding_privacy_for_llm-based_text-to-sql_via_abstraction.md)**

:   提出 MaskSQL 框架，通过提示抽象（abstraction）将敏感的表名、列名和数据值替换为抽象符号后发送给远程 LLM，结合本地 SLM 做 schema linking 和 SQL 重建，在保护隐私同时超越 SLM-only 方案的 SQL 生成精度。

**[Matchings Under Biased And Correlated Evaluations](matchings_under_biased_and_correlated_evaluations.md)**

:   在两机构稳定匹配模型中引入评估相关性参数 $\gamma$（机构间评分的对齐程度），分析偏差 $\beta$ 和相关性 $\gamma$ 如何联合影响弱势群体的代表性比率，证明即使轻微的相关性损失也可导致代表性急剧下降，并提出公平性干预策略的 Pareto 前沿。

**[Mitigating Disparate Impact Of Differentially Private Learning Through Bounded A](mitigating_disparate_impact_of_differentially_private_learning_through_bounded_a.md)**

:   通过在自适应梯度剪裁中引入可调整的下界（bounded adaptive clipping），防止 clipping bound 在训练过程中过度萎缩，从而改善少数群体的精度，在 DP 约束下缓解算法不公平。

**[Mitigating Privacy-Utility Trade-Off In Decentralized Federated Learning Via F-D](mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)**

:   提出基于 f-DP 框架的两种去中心化联邦学习隐私记账方法——PN-f-DP 和 Sec-f-LDP，通过更精细的假设检验隐私度量，一致性地获得比 Rényi DP 更紧的隐私界，从而在相同隐私保证下减少噪声注入、提升模型效用。

**[Mixat Combining Continuous And Discrete Adversarial Training For Llms](mixat_combining_continuous_and_discrete_adversarial_training_for_llms.md)**

:   提出MixAT方法，将离散对抗攻击（PAP改写）与连续嵌入空间扰动相结合进行LLM对抗训练，在保持高效用的同时实现对多种攻击的鲁棒性（ALO-ASR从50%+降至20%以下），且训练成本仅与纯连续方法相当。

**[Model Inversion With Layer-Specific Modeling And Alignment For Data-Free Continu](model_inversion_with_layer-specific_modeling_and_alignment_for_data-free_continu.md)**

:   在无数据持续学习场景中，提出逐层模型反演（PMI）来加速图像合成，并通过类别级高斯特征建模和对比学习缓解合成-真实数据间的特征漂移，实现高效且高质量的无数据知识回放。

**[Mpcache Mpc-Friendly Kv Cache Eviction For Efficient Private Llm Inference](mpcache_mpc-friendly_kv_cache_eviction_for_efficient_private_llm_inference.md)**

:   本文提出MPCache，一个面向安全多方计算（MPC）的KV缓存淘汰框架，结合一次性静态淘汰和查询感知的动态选择，配合层次化聚类、线性化相似度近似和跨层索引共享等优化，在不牺牲LLM性能的前提下实现最高2.01倍延迟降低和8.37倍通信量削减。

**[Multi-Class Support Vector Machine With Differential Privacy](multi-class_support_vector_machine_with_differential_privacy.md)**

:   提出PMSVM框架，利用all-in-one多类SVM的单次数据访问特性，结合权重扰动和梯度扰动方法，在保持差分隐私的前提下显著降低多类SVM的隐私预算消耗，实现了更优的隐私-效用权衡。

**[Music Arena Live Evaluation For Text-To-Music](music_arena_live_evaluation_for_text-to-music.md)**

:   提出 Music Arena，一个面向文本到音乐（TTM）生成模型的在线实时人类偏好评估平台，通过众包式对比实验收集偏好数据，编制排行榜并定期开放数据。

**[Nearly-Linear Time Private Hypothesis Selection With The Optimal Approximation F](nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)**

:   首次提出在中心差分隐私模型下同时实现近线性时间复杂度和最优近似因子 $\alpha=3$ 的假设选择算法，解决了Bun等人（NeurIPS 2019）提出的开放问题。

**[Not All Deepfakes Are Created Equal Triaging Audio Forgeries For Robust Deepfake](not_all_deepfakes_are_created_equal_triaging_audio_forgeries_for_robust_deepfake.md)**

:   提出基于"最有害的深伪是质量最高的"这一前提的两阶段流水线：先用判别器过滤低质量伪造以减少噪声，再用仅在真实录音上训练的歌手识别模型进行声纹匹配，在多个数据集上一致超越基线。

**[Omnifc Rethinking Federated Clustering Via Lossless And Secure Distance Reconstr](omnifc_rethinking_federated_clustering_via_lossless_and_secure_distance_reconstr.md)**

:   提出 OmniFC，一个模型无关的联邦聚类框架：通过 Lagrange 编码计算在有限域上精确重建全局成对距离矩阵，任意集中式聚类方法（K-Means/谱聚类/DBSCAN/层次聚类等）可直接在其上运行，仅需一轮通信，天然抵抗 Non-IID，在 7 个数据集上全面超越 k-FED/MUFC/FedSC 等专用方法。

**[On The Empirical Power Of Goodness-Of-Fit Tests In Watermark Detection](on_the_empirical_power_of_goodness-of-fit_tests_in_watermark_detection.md)**

:   系统性地评估了八种经典拟合优度（GoF）检验在 LLM 文本水印检测中的效果，发现 GoF 检验在检测功效和鲁棒性上均显著优于现有基线方法。

**[On The Hardness Of Conditional Independence Testing In Practice](on_the_hardness_of_conditional_independence_testing_in_practice.md)**

:   系统分析了基于核的条件独立性（CI）检验在实践中失败的根本原因：条件均值嵌入的估计误差是导致Type-I错误膨胀的核心因素，同时揭示了选择条件核$k_C$对检验功效至关重要但会加剧假阳性的内在张力。

**[On The Robustness Of Verbal Confidence Of Llms In Adversarial Attacks](on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)**

:   首次系统研究 LLM 语言化置信度（verbal confidence）在对抗攻击下的鲁棒性，提出基于扰动和越狱的攻击框架，揭示攻击可导致置信度下降最高 30%、答案翻转率高达 100%，且现有防御策略基本无效。

**[On The Sample Complexity Of Differentially Private Policy Optimization](on_the_sample_complexity_of_differentially_private_policy_optimization.md)**

:   首次系统性研究差分隐私（DP）约束下策略优化的样本复杂度，提出统一的元算法框架，分析DP-PG、DP-NPG和DP-REBEL三种隐私策略优化算法，证明隐私代价通常仅作为样本复杂度的低阶项出现。

**[Optimal Adjustment Sets For Nonparametric Estimation Of Weighted Controlled Dire](optimal_adjustment_sets_for_nonparametric_estimation_of_weighted_controlled_dire.md)**

:   针对加权控制直接效应（WCDE）建立三项基础理论：唯一可识别性的充要条件、非参数估计的影响函数推导、以及最小化渐近方差的最优协变量调整集刻画。

**[Orbit -- Open Recommendation Benchmark For Reproducible Research With Hidden Tes](orbit_--_open_recommendation_benchmark_for_reproducible_research_with_hidden_tes.md)**

:   提出ORBIT统一推荐系统基准，包含5个标准化公开数据集评估和基于真实浏览历史构建的隐私安全ClueWeb-Reco隐藏测试集，系统评估了12个推荐模型并引入LLM-QueryGen基线，揭示了现有方法在大规模真实推荐场景中的局限性。

**[Poly-Guard Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset](poly-guard_massive_multi-domain_safety_policy-grounded_guardrail_dataset.md)**

:   提出 Poly-Guard，首个大规模多领域安全策略驱动的护栏数据集，覆盖8个安全关键领域，基于真实行业安全指南构建风险分类体系，并对19个先进护栏模型进行系统性评估。

**[Position Bridge The Gaps Between Machine Unlearning And Ai Regulation](position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)**

:   系统分析了机器遗忘（Machine Unlearning）在欧盟人工智能法案（AIA）合规中的六大潜在应用场景，指出每个场景中 SOTA 与实际需求之间的技术差距，呼吁研究社区弥补这些差距以释放机器遗忘在 AI 监管中的潜力。

**[Preserving Task-Relevant Information Under Linear Concept Removal](preserving_task-relevant_information_under_linear_concept_removal.md)**

:   SPLINCE通过构造一种斜投影(oblique projection)，在保证线性守护性（不可被线性分类器预测敏感属性）的同时，精确保留表征与目标标签之间的协方差，解决了现有概念擦除方法在移除敏感概念的同时误删任务相关信息的问题。

**[Private Continual Counting Of Unbounded Streams](private_continual_counting_of_unbounded_streams.md)**

:   提出基于对数扰动的新型矩阵分解方法，首次实现同时满足"无界流"、"平滑误差"和"近最优渐近误差"三大性质的差分隐私持续计数算法，对任意 $\alpha > 0$ 在时间步 $t$ 处的方差为 $O(\log^{2+2\alpha}(t))$。

**[Private Zeroth-Order Optimization With Public Data](private_zeroth-order_optimization_with_public_data.md)**

:   提出 PAZO 框架，利用公共数据引导私有零阶优化算法的梯度近似，在视觉和文本任务上实现了优于 DP-SGD 的隐私-效用权衡，同时获得最高 16 倍的速度提升。

**[Probabilistic Reasoning With Llms For K-Anonymity Estimation](probabilistic_reasoning_with_llms_for_k-anonymity_estimation.md)**

:   本文提出Branch框架，利用大语言模型将用户文本中的个人信息建模为贝叶斯网络的联合概率分布，分别估计各属性的条件概率后组合计算k-匿名值（全球匹配该信息的人数），在隐私风险估计任务上达到73%准确率，比o3-mini链式思维提升13%。

**[Provable Watermarking For Data Poisoning Attacks](provable_watermarking_for_data_poisoning_attacks.md)**

:   本文提出两种可证明的水印方案（后投毒水印和投毒并行水印），为数据投毒攻击提供透明性声明机制，理论证明在特定水印长度条件下可同时保证水印可检测性和投毒有效性。

**[Pubsub-Vfl Towards Efficient Two-Party Split Learning In Heterogeneous Environme](pubsub-vfl_towards_efficient_two-party_split_learning_in_heterogeneous_environme.md)**

:   本文提出PubSub-VFL，一种基于发布/订阅架构的高效两方纵向联邦学习框架，通过分层异步机制和基于系统画像的超参数优化，在保证隐私和模型精度的前提下实现2~7倍的训练加速和高达91%的计算资源利用率。

**[Pulse Practical Evaluation Scenarios For Large Multimodal Model Unlearning](pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)**

:   本文提出 PULSE 评估协议，从预训练知识遗忘和多次顺序遗忘的可持续性两个实际维度出发，揭示了现有遗忘方法在 LMM 上的严重不足——遗忘预训练知识会导致 90% 以上通用能力丧失，连续遗忘 5 次后模型泛化能力几乎完全崩溃。

**[Reconstruction And Secrecy Under Approximate Distance Queries](reconstruction_and_secrecy_under_approximate_distance_queries.md)**

:   在近似距离查询模型下，通过学习理论视角研究重建博弈（reconstruction game），证明了最优重建误差等于Chebyshev半径的几何特征刻画，并对欧氏凸空间的伪有限性给出了完整分类。

**[Reliabilityrag Effective And Provably Robust Defense For Rag-Based Web-Search](reliabilityrag_effective_and_provably_robust_defense_for_rag-based_web-search.md)**

:   ReliabilityRAG 提出了一种利用文档可靠性信号（如搜索排名）进行对抗防御的 RAG 框架，通过在矛盾图上寻找最大独立集（MIS）来识别一致的文档子集并优先选择高可靠性文档，提供可证明的鲁棒性保证，同时在良性场景和长文本生成任务上保持高准确率。

**[Reverse Engineering Human Preferences With Reinforcement Learning](reverse_engineering_human_preferences_with_reinforcement_learning.md)**

:   使用强化学习训练前导文本生成器来提升下游 LLM 的评分成绩,揭示了 LLM-as-a-Judge 评估框架的脆弱性,且该攻击方式几乎不可检测并具有跨模型迁移能力。

**[Rewind-To-Delete Certified Machine Unlearning For Nonconvex Functions](rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)**

:   本文提出R2D（Rewind-to-Delete），首个适用于一般非凸损失函数的一阶、黑盒认证机器遗忘算法，通过"回溯"到训练过程中的较早检查点再对保留数据执行梯度下降来实现数据删除，同时提供(ε,δ)认证遗忘保证和隐私-效用-效率的理论权衡。

**[Robust Graph Condensation Via Classification Complexity Mitigation](robust_graph_condensation_via_classification_complexity_mitigation.md)**

:   本文揭示图凝缩（GC）本质上是降低分类复杂度的过程，而对抗攻击恰好破坏了这一特性，据此提出MRGC框架，通过内在维度正则化、曲率感知流形平滑和类间流形解耦三个流形约束模块来增强GC的鲁棒性，首次在结构、特征和标签均可能被篡改的条件下系统研究GC鲁棒性。

**[Saemark Steering Personalized Multilingual Llm Watermarks With Sparse Autoencode](saemark_steering_personalized_multilingual_llm_watermarks_with_sparse_autoencode.md)**

:   提出SAEMark框架，利用稀疏自编码器（SAE）提取文本的语义特征浓度评分，通过推理阶段的特征引导拒绝采样实现多比特水印嵌入，无需修改模型权重或logits，天然支持黑盒API、多语言和代码等场景，在英文/中文/代码上均达到领先的水印精度与文本质量。

**[Seca Semantically Equivalent And Coherent Attacks For Eliciting Llm Hallucinatio](seca_semantically_equivalent_and_coherent_attacks_for_eliciting_llm_hallucinatio.md)**

:   提出 SECA（Semantically Equivalent and Coherent Attacks），通过保持语义等价和语义连贯性的现实主义提示修改来诱发 LLM 幻觉，在多选 QA 任务上实现更高攻击成功率且几乎无语义错误。

**[Self-Refining Language Model Anonymizers Via Adversarial Distillation](self-refining_language_model_anonymizers_via_adversarial_distillation.md)**

:   提出 SEAL 框架，通过对抗蒸馏将 GPT-4 级 LLM 的文本匿名化能力蒸馏到 8B 小模型中，结合 SFT + DPO 训练和自我精炼机制，使小模型在隐私-效用权衡上达到甚至超越 GPT-4 匿名化器的水平，且可完全本地部署。

**[Sequentially Auditing Differential Privacy](sequentially_auditing_differential_privacy.md)**

:   提出基于序贯假设检验和核 MMD 统计量的差分隐私审计框架，可以在流式处理机制输出时随时有效地检测隐私违规，将所需样本量从现有方法的 50K 降低到数百个，并能在不到一次完整训练的过程中识别 DP-SGD 的隐私违规。

**[Some Optimizers Are More Equal Understanding The Role Of Optimizers In Group Fai](some_optimizers_are_more_equal_understanding_the_role_of_optimizers_in_group_fai.md)**

:   本文首次系统研究了优化算法选择对深度学习群体公平性的影响，通过随机微分方程（SDE）分析和两个新定理证明，自适应优化器（RMSProp/Adam）比SGD更容易收敛到公平的极小值点，特别是在数据严重不平衡时。

**[Spectral Perturbation Bounds For Low-Rank Approximation With Applications To Pri](spectral_perturbation_bounds_for_low-rank_approximation_with_applications_to_pri.md)**

:   建立了对称矩阵低秩近似在谱范数下的新型高概率扰动界，改进了经典 Eckart-Young-Mirsky 定理，并解决了差分隐私 PCA 中的一个公开问题。

**[Stochastic Regret Guarantees For Online Zeroth- And First-Order Bilevel Optimiza](stochastic_regret_guarantees_for_online_zeroth-_and_first-order_bilevel_optimiza.md)**

:   提出了一种新的搜索方向并证明基于该方向的一阶和零阶在线双层优化算法能够在不需要窗口平滑的条件下实现次线性随机双层遗憾保证，同时通过降低 oracle 依赖、并行更新和零阶 Hessian/Jacobian 估计来提升效率。

**[Swe-Sql Illuminating Llm Pathways To Solve User Sql Issues In Real-World Applica](swe-sql_illuminating_llm_pathways_to_solve_user_sql_issues_in_real-world_applica.md)**

:   提出 BIRD-CRITIC 基准（首个 SQL 问题调试基准）和 Six-Gym 训练环境，并开发 Bird-Fixer 智能体，通过 f-Plan Boosting 策略将 14B 开源模型的 SQL 调试能力提升至超越 Claude-3.7-Sonnet 和 GPT-4.1 的水平，在保护数据隐私的同时实现高效的 SQL 问题修复。

**[Systematizing Llm Persona Design A Four-Quadrant Technical Taxonomy For Ai Compa](systematizing_llm_persona_design_a_four-quadrant_technical_taxonomy_for_ai_compa.md)**

:   提出 LLM persona 设计的四象限技术分类框架，沿"虚拟 vs 具身"和"情感陪伴 vs 功能增强"两轴，系统化分析了从虚拟伴侣、游戏 NPC 到护理机器人等不同场景下的技术栈、核心挑战和伦理风险。

**[Taught Well Learned Ill Towards Distillation-Conditional Backdoor Attack](taught_well_learned_ill_towards_distillation-conditional_backdoor_attack.md)**

:   本文提出了蒸馏条件后门攻击（DCBA）范式及其实现方法SCAR，通过双层优化在教师模型中植入"休眠"后门，该后门在教师模型上不可检测但会在知识蒸馏过程中被激活传递到学生模型，即使蒸馏数据集完全干净。

**[The Unseen Threat Residual Knowledge In Machine Unlearning Under Perturbed Sampl](the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)**

:   发现机器遗忘的关键安全漏洞：即使遗忘后的模型在统计意义上与重训练模型不可区分，对遗忘样本施加微小对抗扰动后，遗忘模型仍能正确识别而重训练模型则失败——揭示了"残余知识"这一新型隐私风险。提出 RURK 微调策略，通过惩罚对扰动遗忘样本的正确预测来消除残余知识，在 CIFAR-10 和 ImageNet-100 上有效抑制 11 种遗忘方法的残余知识。

**[Toxictextclip Text-Based Poisoning And Backdoor Attacks On Clip Pre-Training](toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)**

:   提出 ToxicTextCLIP 框架，通过背景感知选择和背景驱动增强两个模块，在 CLIP 预训练阶段生成高质量对抗文本，实现高达 95.83% 投毒成功率和 98.68% 后门 Hit@1，且能突破 RoCLIP、CleanCLIP、SafeCLIP 三种防御。

**[Trans-Env A Framework For Evaluating The Linguistic Robustness Of Llms Against E](trans-env_a_framework_for_evaluating_the_linguistic_robustness_of_llms_against_e.md)**

:   提出Trans-EnV框架，结合语言学专家知识和LLM变换能力，将标准美式英语（SAE）数据集自动转换为38种英语变体（18种方言+20种ESL英语），揭示LLM在非标准英语上最高46.3%的性能下降，凸显了语言公平性问题。

**[Trap Targeted Redirecting Of Agentic Preferences](trap_targeted_redirecting_of_agentic_preferences.md)**

:   TRAP 提出了一种基于扩散模型的语义注入对抗框架，通过在 CLIP 嵌入空间中优化图像语义，在黑盒条件下以视觉自然的方式系统性地误导多个主流 VLM 智能体的决策偏好，在 LLaVA-34B、GPT-4o 等六个模型上实现了高达 100% 的攻击成功率。

**[Understanding And Improving Adversarial Robustness Of Neural Probabilistic Circu](understanding_and_improving_adversarial_robustness_of_neural_probabilistic_circu.md)**

:   理论分析神经概率电路（NPC）的对抗鲁棒性仅取决于属性识别模型而与概率电路无关，并提出 RNPC 通过类级推理集成方式实现可证明的鲁棒性提升，在保持良性准确率的同时显著增强对抗鲁棒性。

**[Understanding Challenges To The Interpretation Of Disaggregated Evaluations Of A](understanding_challenges_to_the_interpretation_of_disaggregated_evaluations_of_a.md)**

:   通过因果图模型分析表明，分组评估（disaggregated evaluation）中跨子群体的性能差异不一定意味着不公平，而可能是数据生成过程中分布差异的自然结果，建议结合因果假设和加权评估补充标准分组评估。

**[Unifying Proportional Fairness In Centroid And Non-Centroid Clustering](unifying_proportional_fairness_in_centroid_and_non-centroid_clustering.md)**

:   将质心聚类(centroid)和非质心聚类(non-centroid)的比例公平性研究统一到"半质心聚类"框架中，证明了两者不可同时实现的不可能性定理，并设计了新算法在双度量损失下实现常数倍近似的核(core)保证。

**[Unifying Re-Identification Attribute Inference And Data Reconstruction Risks In ](unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)**

:   基于假设检验解释的 f-DP 框架，统一了差分隐私中重识别、属性推断和数据重建三类隐私风险的界定，提供更紧致一致的风险上界，使噪声校准可减少 20% 且不降低安全性。

**[Unlearning As Ablation Toward A Falsifiable Benchmark For Generative Scientific ](unlearning_as_ablation_toward_a_falsifiable_benchmark_for_generative_scientific_.md)**

:   本文提出将机器遗忘重新定义为认识论探针工具（"遗忘即消融"），通过系统性移除目标知识及其遗忘闭包后测试模型能否从公理出发重新推导，从而提供可证伪的测试来区分 LLM 是"真正生成新知识"还是"仅仅检索记忆片段"。

**[Virus Infection Attack On Llms Your Poisoning Can Spread Via Synthetic Data](virus_infection_attack_on_llms_your_poisoning_can_spread_via_synthetic_data.md)**

:   本文首次系统研究了合成数据在LLM训练中的安全风险，发现现有投毒/后门攻击难以通过合成数据传播，进而提出Virus Infection Attack (VIA)框架，通过劫持点搜索和外壳构造将投毒内容嵌入正常训练样本中，使恶意内容即使在干净查询下也能被模型生成并传播到下游模型。

**[When Ai Democratizes Exploitation Llm-Assisted Strategic Manipulation Of Fair Di](when_ai_democratizes_exploitation_llm-assisted_strategic_manipulation_of_fair_di.md)**

:   通过实证分析展示 LLM 如何民主化战略操纵能力，使普通用户无需数学专业知识即可通过简单对话获取 Spliddit 公平分租算法的协调操纵策略，揭示了 AI 辅助下算法公平性机制面临的新威胁。
