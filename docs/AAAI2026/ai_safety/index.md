---
title: >-
  AAAI2026 AI安全方向 61篇论文解读
description: >-
  61篇AAAI2026 AI安全方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI安全

**🤖 AAAI2026** · 共 **61** 篇

**[Alternative Fairness And Accuracy Optimization In Criminal J](alternative_fairness_and_accuracy_optimization_in_criminal_j.md)**

:   本文系统综述了算法公平性的三大维度（群体公平、个体公平、过程公平），提出了一种基于容差约束的改进群体公平性优化公式，并构建了面向公共决策系统的"公平三支柱"部署框架。

**[An Improved Privacy And Utility Analysis Of Differentially P](an_improved_privacy_and_utility_analysis_of_differentially_p.md)**

:   在仅假设损失函数L-光滑（不需要凸性）的条件下，为DPSGD推导出了更紧的闭式RDP隐私界，并首次在有界域场景下给出了完整的收敛性/效用分析，揭示了较小的参数域直径可以同时改善隐私和效用。

**[An Information Theoretic Evaluation Metric For Strong Unlear](an_information_theoretic_evaluation_metric_for_strong_unlear.md)**

:   提出 Information Difference Index (IDI)，一种基于信息论的白盒评估指标，通过度量中间层特征与遗忘标签之间的互信息来衡量机器遗忘的彻底程度，揭示了现有黑盒指标（MIA、JSD等）无法捕捉的中间层残留信息问题，并提出 COLA 方法在特征层面消除残余信息。

**[An Information Theoretic Evaluation Metric For Strong Unlearning](an_information_theoretic_evaluation_metric_for_strong_unlearning.md)**

:   揭示现有黑盒遗忘评估指标（MIA/JSD等）的根本缺陷——仅修改最后一层即可满足所有黑盒指标但中间层完整保留遗忘数据信息，提出IDI白盒指标通过InfoNCE估计各层与遗忘标签的互信息差异来量化遗忘效果，并提出COLA方法在CIFAR-10/100和ImageNet-1K上实现接近Retrain的IDI得分。

**[An Llm-Based Simulation Framework For Embodied Conversationa](an_llm-based_simulation_framework_for_embodied_conversationa.md)**

:   提出 ECAs 框架，基于认知行为治疗(CBT)等心理学理论，利用 LLM 将真实咨询案例扩展为具身认知记忆空间，模拟心理咨询中来访者的完整认知过程，生成高保真度的咨询对话数据，在专家评估和自动评估中均显著优于基线。

**[Angular Gradient Sign Method Uncovering Vulnerabilities In H](angular_gradient_sign_method_uncovering_vulnerabilities_in_h.md)**

:   提出Angular Gradient Sign Method (AGSM)，将双曲空间中的梯度分解为径向（层次深度）和角度（语义）分量，仅沿角度方向施加扰动来生成对抗样本，在图像分类和跨模态检索任务上比标准FGSM/PGD多降低5-13%的准确率。

**[Argumentative Debates For Transparent Bias Detection Technic](argumentative_debates_for_transparent_bias_detection_technic.md)**

:   提出ABIDE框架，将偏见检测过程结构化为基于量化二极论辩框架（QBAF）的辩论：通过邻域级局部统计公平性（neighbourhood-based local statistical parity）生成偏见论据，利用批判性问题（critical questions）作为攻击机制挑战不可靠论据，在合成/真实/LLM模型上均优于IRB基线。

**[Auvic Adversarial Unlearning Of Visual Concepts For Multi-Mo](auvic_adversarial_unlearning_of_visual_concepts_for_multi-mo.md)**

:   提出AUVIC框架，通过对抗性扰动生成器 + 动态锚点保留机制，在MLLM中精确遗忘目标视觉概念（如特定人脸），同时避免对语义相似概念的附带遗忘，并构建了首个面向群体场景视觉概念遗忘的评测基准VCUBench。

**[Beyond Superficial Forgetting Thorough Unlearning Through Knowledge Density Esti](beyond_superficial_forgetting_thorough_unlearning_through_knowledge_density_esti.md)**

:   提出 KUnBR 框架，通过梯度引导的知识密度估计定位有害知识富集层，并采用块重插入策略绕过 cover layer 的梯度遮蔽效应，实现对 LLM 有害知识的深度遗忘而非表面抑制。

**[Breaking The Adversarial Robustness-Performance Trade-Off In Text Classification](breaking_the_adversarial_robustness-performance_trade-off_in_text_classification.md)**

:   提出 Manifold-Correcting Causal Flow (MC²F) 框架，通过分层黎曼连续正则化流 (SR-CNF) 学习干净数据嵌入的流形密度进行对抗样本检测，再用测地线净化求解器 (Geodesic Purification Solver) 将被检测为对抗的嵌入沿最短路径投影回干净流形，在 SST-2/AGNews/YELP 三个数据集上对抗鲁棒性全面超越 SOTA，同时完全不损失（甚至略微提升）干净数据精度。

**[Breaking The Dyadic Barrier Rethinking Fairness In Link Prediction Beyond Demogr](breaking_the_dyadic_barrier_rethinking_fairness_in_link_prediction_beyond_demogr.md)**

:   本文揭示了链接预测中二元公平性（dyadic fairness）和 Demographic Parity（ΔDP）的三大根本缺陷——GNN 表达力不足、子群偏差被掩盖、对排序不敏感——并提出基于 NDKL 的排序感知公平度量和后处理算法 MORAL，在六个数据集上实现了 SOTA 的公平性-效用权衡。

**[Can Editing Llms Inject Harm](can_editing_llms_inject_harm.md)**

:   本文将知识编辑技术重新定义为一种新型 LLM 安全威胁（Editing Attack），系统性地研究了通过 ROME、FT、ICE 三种编辑方法向 LLM 注入虚假信息和偏见的可行性，发现其效果显著且极具隐蔽性。

**[Core-Fed Bridging Collaborative And Representation Fairness Via Federated Embedd](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)**

:   提出 CoRe-Fed 框架，通过嵌入级对比对齐与贡献感知聚合两个协同模块，同时解决联邦学习中的表示公平性和协作公平性问题，在异构数据分布下显著提升全局模型的公平性与泛化能力。

**[Deeptracer Tracing Stolen Model Via Deep Coupled Watermarks](deeptracer_tracing_stolen_model_via_deep_coupled_watermarks.md)**

:   提出DeepTracer鲁棒水印框架，通过自适应源类选择（K-Means聚类覆盖特征空间）+ 同类耦合损失（拉近水印样本与目标类在输出空间的距离）+ 两阶段关键样本过滤，使水印任务与主任务深度耦合，在6种模型窃取攻击（含hard-label和data-free）下水印成功率平均达77-100%，远超现有方法。

**[Democratizing Llm Efficiency From Hyperscale Optimizations To Universal Deployab](democratizing_llm_efficiency_from_hyperscale_optimizations_to_universal_deployab.md)**

:   本文是一篇立场论文（position paper），指出当前 LLM 效率研究被超大规模假设所主导，提出面向中小规模部署者的五大开放研究挑战，并倡导以开销感知效率（OAE）重新定义效率指标。

**[Detect All-Type Deepfake Audio Wavelet Prompt Tuning For Enhanced Auditory Perce](detect_all-type_deepfake_audio_wavelet_prompt_tuning_for_enhanced_auditory_perce.md)**

:   首次建立全类型（语音/声音/歌声/音乐）音频深伪检测基准，提出小波提示调优（WPT）方法通过离散小波变换增强 SSL 特征的全频域感知能力，在不增加训练参数的前提下超越全量微调，co-training 后平均 EER 仅 3.58%。

**[Diversifying Counterattacks Orthogonal Exploration For Robust Clip Inference](diversifying_counterattacks_orthogonal_exploration_for_robust_clip_inference.md)**

:   提出方向正交反攻击（DOC）方法，通过在反攻击优化中引入正交梯度分量和动量更新扩展搜索空间，结合基于余弦相似度的方向敏感度评分自适应调控反攻击强度，在 16 个数据集上显著提升 CLIP 的测试时对抗鲁棒性。

**[Easy To Learn Yet Hard To Forget Towards Robust Unlearning Under Bias](easy_to_learn_yet_hard_to_forget_towards_robust_unlearning_under_bias.md)**

:   提出 CUPID 框架，通过损失景观的锐度分析将遗忘集划分为因果/偏差子集，并识别和分离模型中的因果/偏差通路，实现对有偏模型的精准类别遗忘，有效解决"捷径遗忘"问题。

**[Efx And Po Allocation Exists For Two Types Of Goods](efx_and_po_allocation_exists_for_two_types_of_goods.md)**

:   证明了当物品只有两种类型且所有估值为正时，满足 EFX（任意物品无嫉妒）和 Pareto 最优的分配总是存在的，并给出了准线性时间算法。

**[Enhancing Dpsgd Via Per-Sample Momentum And Low-Pass Filtering](enhancing_dpsgd_via_per-sample_momentum_and_low-pass_filtering.md)**

:   提出 DP-PMLF，通过逐样本动量（per-sample momentum）降低裁剪偏差，同时利用低通滤波器（low-pass filter）抑制高频 DP 噪声，首次同时从两个方向缓解 DPSGD 的精度退化问题。

**[Fair Model-Based Clustering](fair_model-based_clustering.md)**

:   提出基于有限混合模型的公平聚类算法 FMC，通过在模型参数（而非样本级赋值）上施加公平性约束，实现参数量与样本量无关的可扩展公平聚类，支持小批量学习和分类数据，在大规模数据集上显著优于现有方法。

**[Fairgse Fairness-Aware Graph Neural Network Without High False Positive Rates](fairgse_fairness-aware_graph_neural_network_without_high_false_positive_rates.md)**

:   首次揭示公平感知 GNN 中的"FPR 捷径"问题——现有方法通过大量误判负样本为正来达到公平指标，提出 FairGSE 框架通过最大化二维结构熵重新加权图边来同时改善公平性并降低假阳性率，FPR 降低 39%。

**[Fedalt Federated Fine-Tuning Through Adaptive Local Training With Rest-Of-World ](fedalt_federated_fine-tuning_through_adaptive_local_training_with_rest-of-world_.md)**

:   提出 FedALT，通过为每个客户端维护独立的 Individual LoRA（本地训练更新）和冻结的 Rest-of-World (RoW) LoRA（其他客户端平均），配合自适应 MoE 混合器动态平衡本地知识与全局知识，彻底避免 FedAvg 聚合导致的跨客户端干扰，在异构任务联邦 LLM 微调上显著优于 SOTA。

**[From Single To Societal Analyzing Persona-Induced Bias In Multi-Agent Interactio](from_single_to_societal_analyzing_persona-induced_bias_in_multi-agent_interactio.md)**

:   本文首次系统研究了 LLM 多智能体交互中的人格诱导偏见，通过在协作问题解决和说服任务中的受控实验，揭示了三个关键发现：(1) 不同人格在可信度和坚持度上存在显著偏差（优势群体如男性和白人被视为更不可信）；(2) 智能体表现出显著的内群体偏好；(3) 这些偏见在多轮、多智能体场景中持续存在且有放大趋势。

**[Gender Bias In Emotion Recognition By Large Language Models](gender_bias_in_emotion_recognition_by_large_language_models.md)**

:   系统性地评估了多个 LLM（GPT-4/5、Mistral、LLaMA 等）在情感识别任务中的性别偏见，发现大多数模型对至少一个情感标签存在显著性别偏见，并通过实验证明推理时 prompt 策略（提示工程、上下文学习、CoT）无法有效去偏，而基于训练的微调方法可以有效缓解偏见。

**[Generalizing Fair Clustering To Multiple Groups Algorithms And Applications](generalizing_fair_clustering_to_multiple_groups_algorithms_and_applications.md)**

:   将最近公平聚类（Closest Fair Clustering）问题从仅两个群体推广到任意多群体，证明三群体以上等比例情形已为NP-hard，提出近线性时间近似算法（等比例 $O(|\chi|^{1.6}\log^{2.81}|\chi|)$、任意比例 $O(|\chi|^{3.81})$），并将结果推广至公平相关聚类和公平共识聚类问题。

**[Ghost In The Transformer Detecting Model Reuse With Invariant Spectral Signature](ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)**

:   提出 GhostSpec，一种无需数据、不修改模型行为的白盒方法，通过对注意力权重矩阵的不变乘积做 SVD 提取光谱指纹，可在微调、剪枝、合并、扩展甚至对抗性变换下稳健地验证 LLM 血统。

**[Graphtextack A Realistic Black-Box Node Injection Attack On Llm-Enhanced Gnns](graphtextack_a_realistic_black-box_node_injection_attack_on_llm-enhanced_gnns.md)**

:   提出 GraphTextack——首个针对 LLM 增强 GNN 的黑盒多模态节点注入投毒攻击，通过进化优化框架联合优化注入节点的图结构连接和语义特征，不依赖模型内部信息或代理模型，在5个数据集和2类LLM-GNN模型上显著优于12种基线方法。

**[Hashed Watermark As A Filter Defeating Forging And Overwriting Attacks In Weight](hashed_watermark_as_a_filter_defeating_forging_and_overwriting_attacks_in_weight.md)**

:   提出 NeuralMark——一种基于哈希水印过滤器的权重水印方法，利用哈希函数从秘钥生成不可逆二值水印作为私有过滤器选择嵌入参数，借助雪崩效应阻断伪造攻击的梯度反推，通过多轮过滤减少参数重叠抵御覆写攻击，在13种CNN/Transformer架构、5个图像分类和1个文本生成任务上验证了有效性和鲁棒性。

**[Healsplit Towards Self-Healing Through Adversarial Distillation In Split Federat](healsplit_towards_self-healing_through_adversarial_distillation_in_split_federat.md)**

:   提出 HealSplit，首个针对分割联邦学习（SFL）的统一防御框架，通过拓扑感知检测（TAS）识别中毒样本、GAN 生成语义一致的替代表示、对抗多教师蒸馏训练一致性验证学生模型，实现端到端检测与恢复，在五类投毒攻击下均大幅超越十种 SOTA 防御方法。

**[Infodecom Decomposing Information For Defending Against Privacy Leakage In Split](infodecom_decomposing_information_for_defending_against_privacy_leakage_in_split.md)**

:   提出 InfoDecom，通过两级信息分解（频域视觉信息去除 + 互信息抑制）减少 smashed data 中的冗余信息，再添加闭式计算的高斯噪声提供理论隐私保证，在浅层客户端模型下实现远优于现有方法的 utility-privacy trade-off。

**[Lamp Learning Universal Adversarial Perturbations For Multi-Image Tasks Via Pre-](lamp_learning_universal_adversarial_perturbations_for_multi-image_tasks_via_pre-.md)**

:   提出 LAMP，一种针对多图 MLLM 的 black-box Universal Adversarial Perturbation 学习方法，通过 attention 约束和"传染式"损失实现仅扰动少量图像即可跨模型/任务迁移攻击。

**[Learning To Collaborate An Orchestrated-Decentralized Framework For Peer-To-Peer](learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)**

:   提出 KNEXA-FL 框架，通过一个不接触模型的中央配对器（CPM）将 P2P 协作建模为上下文 Bandit 问题，使用 LinUCB 学习最优配对策略，在异构 LLM 联邦学习中实现比随机 P2P 高约 50% 的 Pass@1 提升，且避免了中心化蒸馏的灾难性崩溃。

**[Matrix-Free Two-To-Infinity And One-To-Two Norms Estimation](matrix-free_two-to-infinity_and_one-to-two_norms_estimation.md)**

:   提出 TwINEst 和 TwINEst++ 两种 matrix-free 随机算法用于估计 $\|A\|_{2\to\infty}$ 和 $\|A\|_{1\to 2}$ 范数，并将其应用于深度网络 Jacobian 正则化（提升分类泛化和对抗鲁棒性）和推荐系统对抗防御。

**[Mpd-Sgr Robust Spiking Neural Networks With Membrane Potential Distribution-Driv](mpd-sgr_robust_spiking_neural_networks_with_membrane_potential_distribution-driv.md)**

:   从理论上建立了 SNN 鲁棒性误差与代理梯度（SG）幅值之间的联系，揭示减少膜电位分布（MPD）与 SG 梯度可用区间的重叠比例可有效降低对抗扰动敏感度，据此提出 MPD-SGR 正则化方法，在 vanilla training 和 adversarial training 设置下均大幅超越现有 SNN 防御方法。

**[Perturb Your Data Paraphrase-Guided Training Data Watermarking](perturb_your_data_paraphrase-guided_training_data_watermarking.md)**

:   提出SPECTRA——一种基于paraphrase采样的训练数据水印方法，通过LLM生成改写文本并利用Min-K%++评分选择与原文分数接近的paraphrase作为水印，在数据仅占训练语料0.001%的情况下，member与non-member的p-value差距稳定超过9个数量级。

**[Plug-And-Play Parameter-Efficient Tuning Of Embeddings For Federated Recommendat](plug-and-play_parameter-efficient_tuning_of_embeddings_for_federated_recommendat.md)**

:   提出一个即插即用的联邦推荐框架，通过将 PEFT（Parameter-Efficient Fine-Tuning）理念引入物品嵌入，冻结预训练的全量嵌入并仅传输轻量级压缩嵌入（LoRA / Hash / RQ-VAE），大幅降低通信开销的同时提升推荐精度。

**[Principles2Plan Llm-Guided System For Operationalising Ethical Principles Into P](principles2plan_llm-guided_system_for_operationalising_ethical_principles_into_p.md)**

:   提出 Principles2Plan，一个交互式原型系统，通过人类与 LLM 协作将高层伦理原则（如仁善、隐私）转化为上下文相关的伦理规则，并嵌入 PDDL 规划器生成符合伦理的行动计划。

**[Prism Privacy-Aware Routing For Adaptive Cloud-Edge Llm Inference Via Semantic S](prism_privacy-aware_routing_for_adaptive_cloud-edge_llm_inference_via_semantic_s.md)**

:   提出 PRISM 框架，通过上下文感知的软门控路由机制将用户 prompt 动态分配到云端/边缘/协作三种推理模式，并在协作模式中使用自适应两层本地差分隐私（LDP）和语义草图协作，实现隐私-效用-效率的三方平衡。

**[Privacy-Protected Retrieval-Augmented Generation For Knowledge Graph Question An](privacy-protected_retrieval-augmented_generation_for_knowledge_graph_question_an.md)**

:   首次探索知识图谱问答（KGQA）中的隐私保护 RAG 场景，提出 ARoG（Abstraction Reasoning on Graph）框架，通过关系中心抽象和结构导向抽象两种策略，在实体被匿名化（替换为无意义的 MID）的条件下，仍能有效检索和利用知识图谱回答问题。

**[Privacy Auditing Of Multi-Domain Graph Pre-Trained Model Under Membership Infere](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)**

:   提出 MGP-MIA 框架，首次针对多域图预训练模型开展成员推理攻击（MIA），通过机器遗忘放大成员信号、增量学习构建影子模型、基于相似度的推理机制，有效揭示多域图预训练的隐私泄漏风险。

**[Privacy On The Fly A Predictive Adversarial Transformation Network For Mobile Se](privacy_on_the_fly_a_predictive_adversarial_transformation_network_for_mobile_se.md)**

:   提出 PATN（Predictive Adversarial Transformation Network），首个将对抗扰动引入传感器数据隐私保护的框架，利用历史传感器数据生成面向未来的对抗扰动，实现零延迟的实时隐私保护，同时保持传感器数据的语义保真度。

**[Problog4Fairness A Neurosymbolic Approach To Modeling And Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)**

:   提出 ProbLog4Fairness 框架，利用概率逻辑编程语言 ProbLog 将数据中的偏差机制形式化为可解释的逻辑程序，并通过 DeepProbLog 的远程监督将偏差假设集成到神经网络训练中，实现灵活、原则性的偏差缓解。

**[Psm Prompt Sensitivity Minimization Via Llm-Guided Black-Box Optimization](psm_prompt_sensitivity_minimization_via_llm-guided_black-box_optimization.md)**

:   提出 PSM 框架，将系统提示防护形式化为效用约束下的黑盒优化问题，利用 LLM-as-Optimizer 自动搜索最优"盾牌"后缀，在不降低模型功能的前提下将提示泄漏攻击成功率降至接近零。

**[Reference Recommendation Based Membership Inference Attack Against Hybrid-Based ](reference_recommendation_based_membership_inference_attack_against_hybrid-based_.md)**

:   提出基于参考推荐的成员推理攻击（MIA），设计相对成员度量 $\rho(u) = d(v_t, v_h) / d(v_t, v_r)$，利用混合推荐系统的个性化特性获取参考推荐，首次有效攻击混合推荐系统，攻击成功率高达 93.4% 且计算成本仅需 10 秒。

**[Regionmarker A Region-Triggered Semantic Watermarking Framework For Embedding-As](regionmarker_a_region-triggered_semantic_watermarking_framework_for_embedding-as.md)**

:   提出基于语义区域触发的水印框架 RegionMarker，在低维空间中定义触发区域并注入语义水印，是首个能同时抵御 CSE 攻击、改写攻击和维度扰动攻击的 EaaS 版权保护方法。

**[Rethinking Target Label Conditioning In Adversarial Attacks A 2D Tensor-Guided G](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)**

:   提出 TGAF 框架，利用扩散模型将目标标签编码为 2D 语义张量来引导对抗噪声生成，并设计随机遮挡策略保留完整语义信息，显著提升目标对抗攻击的可迁移性。

**[Revisiting Unfairness In Recourse By Minimizing Worst-Case Social Burden](revisiting_unfairness_in_recourse_by_minimizing_worst-case_social_burden.md)**

:   指出现有 recourse 公平性指标忽略了分类器决策偏差和 ground-truth 信息，提出基于 social burden 的整体公平性框架和 MISOB 算法，通过 minimax 重加权减少所有群体的 worst-case 社会负担。

**[Robust Watermarking On Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)**

:   提出首个针对 GBDT 模型的鲁棒水印框架，通过 in-place 微调嵌入水印，设计了四种嵌入策略（Wrong Prediction Flip、Outlier Flip、Cluster Center Flip、Confidence Flip），实现高嵌入成功率、低精度损失和强抗微调鲁棒性。

**[Secmoe Communication-Efficient Secure Moe Inference Via Select-Then-Compute](secmoe_communication-efficient_secure_moe_inference_via_select-then-compute.md)**

:   提出 SecMoE 框架，通过 Select-Then-Compute 范式在两方安全计算中高效实现稀疏 MoE 推理，避免冗余专家计算，通信量降低最高 29.8 倍，端到端加速最高 16.1 倍。

**[Sim-To-Real An Unsupervised Noise Layer For Screen-Camera Watermarking Robustnes](sim-to-real_an_unsupervised_noise_layer_for_screen-camera_watermarking_robustnes.md)**

:   提出 Simulation-to-Real (S2R) 框架，首创"数学建模 → 无监督域迁移"两阶段噪声近似策略：先用数学模型将清晰图像变换到已知噪声域 $\mathcal{C}$，再用无监督 Image-to-Image 网络 $G$ 将 $\mathcal{C}$ 映射到真实屏幕-相机噪声域 $\mathcal{U}$，无需配对数据即可精确逼近真实 SC 噪声，在多设备、多角度、多距离条件下均取得最优水印鲁棒性（BER 降低 30-60%）和图像质量（PSNR 42.27 dB / SSIM 0.962）。

**[Sproutbench A Benchmark For Safe And Ethical Large Language Models For Youth](sproutbench_a_benchmark_for_safe_and_ethical_large_language_models_for_youth.md)**

:   提出 SproutBench，一个包含 1,283 个发展心理学驱动的对抗性提示的评估基准，系统评估 47 个 LLM 在儿童和青少年（0-6、7-12、13-18 岁）场景下的安全性，发现安全性与风险预防强相关（$\rho = 0.86$），交互性与年龄适配性存在显著权衡（$\rho = -0.48$）。

**[Stylebreak Revealing Alignment Vulnerabilities In Large Audio-Language Models Vi](stylebreak_revealing_alignment_vulnerabilities_in_large_audio-language_models_vi.md)**

:   提出 StyleBreak，首个基于语音风格的音频越狱框架，通过两阶段风格感知变换管道和查询自适应策略网络，系统研究语言学、副语言学和超语言学属性对 LAM 对齐鲁棒性的影响，在多种攻击范式下将 ASR 提升 7.1%-22.3%。

**[The Confidence Trap Gender Bias And Predictive Certainty In Llms](the_confidence_trap_gender_bias_and_predictive_certainty_in_llms.md)**

:   提出Gender-ECE指标，系统评估六种开源LLM在性别代词预测任务中的置信度校准与人类偏见对齐程度，发现Gemma-2模型校准最差且存在极端的男女代词校准差异，而训练数据过滤较少的GPT-J-6B反而校准最好。

**[Toporeformer Mitigating Adversarial Attacks Using Topological Purification In Oc](toporeformer_mitigating_adversarial_attacks_using_topological_purification_in_oc.md)**

:   提出 TopoReformer，一种基于拓扑自编码器的模型无关对抗纯化管线，利用持久同调（persistent homology）在潜空间中强制拓扑一致性，无需对抗训练即可过滤对抗扰动，有效保护 OCR 系统免受经典攻击、自适应攻击和 OCR 专用水印攻击。

**[Towards Multiple Missing Values-Resistant Unsupervised Graph Anomaly Detection](towards_multiple_missing_values-resistant_unsupervised_graph_anomaly_detection.md)**

:   提出 M2V-UGAD 框架，首次解决节点属性和图拓扑同时缺失下的无监督图异常检测问题，通过双通路独立填补、超球潜空间融合和伪异常生成三个核心机制，克服跨视图干扰和填补偏差，在7个基准数据集上一致超越现有方法。

**[Transferable Hypergraph Attack Via Injecting Nodes Into Pivotal Hyperedges](transferable_hypergraph_attack_via_injecting_nodes_into_pivotal_hyperedges.md)**

:   提出 TH-Attack，一种面向超图神经网络（HGNNs）的可迁移节点注入攻击框架，通过识别信息聚合路径中的关键超边并注入语义反转的恶意节点，在黑盒场景下实现对多种 HGNN 架构的有效攻击，Accuracy 可从 80%+ 降至 30% 以下。

**[Truth Justice And Secrecy Cake Cutting Under Privacy Constraints](truth_justice_and_secrecy_cake_cutting_under_privacy_constraints.md)**

:   首个隐私保护蛋糕切割协议，在保持无嫉妒性和策略防谋性的同时，通过秘密共享和安全多方计算（MPC）技术确保参与者的估值函数不被泄露。

**[Uncovering Bias Paths With Llm-Guided Causal Discovery An Active Learning And Dy](uncovering_bias_paths_with_llm-guided_causal_discovery_an_active_learning_and_dy.md)**

:   提出一种融合LLM语义先验与统计信号的混合因果发现框架，通过主动学习（Active Learning）和动态评分机制优先查询信息量最大的变量对，在噪声和混淆条件下有效恢复公平性关键因果路径（如 sex→education→income），显著优于传统CD方法和朴素LLM方法。

**[Watermod Modular Token-Rank Partitioning For Probability-Balanced Llm Watermarki](watermod_modular_token-rank_partitioning_for_probability-balanced_llm_watermarki.md)**

:   提出 WaterMod，一种基于模算术 ($\text{rank} \bmod k$) 的 LLM 文本水印方法，通过对概率排序后的词表进行模残差类划分，在零比特（$k=2$）和多比特（$k>2$）水印场景下统一实现高检测率和低质量损耗，无需外部同义词库或哈希技巧。

**[Yours Or Mine Overwriting Attacks Against Neural Audio Watermarking](yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)**

:   首次系统研究神经音频水印的覆写攻击（overwriting attack），提出白盒、灰盒、黑盒三级攻击方案，在 AudioSeal、Timbre、WavMark 三种 SOTA 方法上均实现接近 100% 的攻击成功率，暴露了现有音频水印系统严重的安全缺陷。
