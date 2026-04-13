---
title: >-
  ICML2025 AI安全方向 60篇论文解读
description: >-
  60篇ICML2025 AI安全方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI安全

**🧪 ICML2025** · 共 **60** 篇

**[A Certified Unlearning Approach Without Access To Source Data](a_certified_unlearning_approach_without_access_to_source_data.md)**

:   提出首个无需访问原始训练数据的认证遗忘框架，利用代理数据集（surrogate dataset）近似原始数据统计特性，通过基于源分布与代理分布之间统计距离的噪声缩放机制，实现可证明的数据删除保证。

**[Accelerating Spectral Clustering Under Fairness Constraints](accelerating_spectral_clustering_under_fairness_constraints.md)**

:   将公平谱聚类（Fair SC）问题转化为凸差分（DC）优化框架，通过变量增广策略和 ADMM 类型算法，避免了昂贵的特征分解计算，在大规模问题上实现显著加速。

**[Activation Space Interventions Can Be Transferred Between Large Language Models](activation_space_interventions_can_be_transferred_between_large_language_models.md)**

:   本文证明了 LLM 之间存在共享的激活空间结构，通过训练自编码器（autoencoder）学习模型间的激活映射，可以将安全干预（如后门移除、有害拒绝转向向量）从源模型迁移到目标模型，实现"小模型对齐大模型"的高效安全干预范式。

**[Adversarial Inception Backdoor Attacks Against Reinforcement Learning](adversarial_inception_backdoor_attacks_against_reinforcement_learning.md)**

:   提出"inception"后门攻击框架——通过在 RL 智能体的训练轨迹中插入触发器并将高回报动作替换为目标对抗动作，首次在严格奖励约束下实现 100% 攻击成功率，同时保持智能体在正常任务上的表现。

**[Align-Then-Unlearn Embedding Alignment For Llm Unlearning](align-then-unlearn_embedding_alignment_for_llm_unlearning.md)**

:   提出 Align-then-Unlearn 框架，通过在语义嵌入空间（而非 token 级别）执行遗忘操作，先训练嵌入预测模块对齐未来语义表示，再微调 LLM 使预测嵌入远离目标概念嵌入，实现对 prompt 改写鲁棒的概念级知识遗忘。

**[An Attack To Break Permutation-Based Private Third-Party Inference Schemes For L](an_attack_to_break_permutation-based_private_third-party_inference_schemes_for_l.md)**

:   提出一种攻击方法可以从置换后的LLM隐藏状态近乎完美地恢复原始提示词，打破了三种近期提出的基于置换的隐私推理方案(Zheng/Yuan/Luo)的安全声明，并剖析了其理论证明的缺陷。

**[An Efficient Private Gpt Never Autoregressively Decodes](an_efficient_private_gpt_never_autoregressively_decodes.md)**

:   提出 POST（Public decOding and Secure verificaTion）方法，利用公开 GPT 模型生成草稿 token 并通过私有模型安全验证，借助安全解码对输入长度不敏感的特性，实现 2.1×~6.0× 的隐私推理加速，同时保持与标准安全解码相同的隐私和生成质量。

**[Breaking The N15 Additive Error Barrier For Private And Efficient Graph Sparsifi](breaking_the_n15_additive_error_barrier_for_private_and_efficient_graph_sparsifi.md)**

:   本文突破了差分隐私图割稀疏化的 $n^{1.5}$ 加性误差壁垒，提出了一种多项式时间的 $(\varepsilon,\delta)$-DP 算法，将加性误差降至 $n^{1.25+o(1)}$，核心技术是首个隐私保护的 expander decomposition 算法。

**[Can One Safety Loop Guard Them All Agentic Guard Rails For Federated Computing](can_one_safety_loop_guard_them_all_agentic_guard_rails_for_federated_computing.md)**

:   提出 Guardian-FC——一个双层框架，通过 Agentic-AI 控制平面和后端无关的领域特定语言（DSL），统一管理不同隐私机制（FHE/MPC/DP）的安全护栏，实现联邦计算中的跨后端一致性安全执行和可审计性。

**[Cape Context-Aware Prompt Perturbation Mechanism With Differential Privacy](cape_context-aware_prompt_perturbation_mechanism_with_differential_privacy.md)**

:   提出 Cape——一种上下文感知的 prompt 扰动机制，通过混合效用函数（结合 token 嵌入距离和上下文 logit）以及分桶指数采样机制，在 local DP 保证下实现比现有方法更优的隐私-效用权衡。

**[Cascade Token-Sharded Private Llm Inference](cascade_token-sharded_private_llm_inference.md)**

:   提出 Cascade——一种基于 token 维度分片的多方推理协议，通过将隐藏状态按 token 维度分发给不同计算节点，避免密码学原语的高昂开销，在保持抵抗 vocab-matching 攻击能力的同时实现比 SMPC 方案快 100× 的推理速度。

**[Clients Collaborate Flexible Differentially Private Federated Learning With Guar](clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)**

:   提出 FedCEO 框架，通过在服务器端对堆叠的客户端模型参数进行张量低秩近端优化，利用不同客户端间的语义互补性恢复 DP 噪声破坏的语义信息，将效用-隐私权衡界改进了 $O(\sqrt{d})$ 量级。

**[Collaborative Mean Estimation Among Heterogeneous Strategic Agents Individual Ra](collaborative_mean_estimation_among_heterogeneous_strategic_agents_individual_ra.md)**

:   针对异构成本的多智能体协作均值估计问题，设计了同时满足个体理性(IR)、激励相容(IC)和公平性的无货币机制，在最坏情况下实现 $\mathcal{O}(\sqrt{m})$ 近似比，并证明了三条不可能性结果。

**[Connecting Thompson Sampling And Ucb Towards More Efficient Best-Fixed Action ](connecting_thompson_sampling_and_ucb_towards_more_efficient_best-fixed_action_.md)**

:   本文提出 DP-TS-UCB 算法，通过限制每轮高斯采样次数并在采样预算耗尽后切换为 UCB 式探索，实现了隐私与遗憾的参数化权衡，将 GDP 保证从 $O(\sqrt{T})$ 大幅改善至 $\tilde{O}(T^{0.25(1-\alpha)})$，同时保持近最优的遗憾界。

**[Connecting Thompson Sampling And Ucb Towards More Efficient Trade-Offs Between P](connecting_thompson_sampling_and_ucb_towards_more_efficient_trade-offs_between_p.md)**

:   提出 DP-TS-UCB 算法，通过限制高斯采样次数并复用最大模型值，在 Thompson Sampling 和 UCB 之间建立连接，实现 $\tilde{O}(T^{0.25(1-\alpha)})$-GDP 隐私保证和 $O(K\ln^{\alpha+1}(T)/\Delta)$ 遗憾上界的参数化权衡。

**[Convex Markov Games A New Frontier For Multi-Agent Reinforcement Learning](convex_markov_games_a_new_frontier_for_multi-agent_reinforcement_learning.md)**

:   提出**凸 Markov 博弈 (cMG)** 框架，将单 agent 凸 MDP 推广到多 agent 设定，允许对占用度量 (occupancy measure) 施加一般凸偏好（如熵、KL 散度、公平性惩罚、安全约束），证明纯策略 Nash 均衡存在，并设计可微的投影梯度损失 (PGL) 算法逼近均衡。

**[Crow Eliminating Backdoors From Large Language Models Via Internal Consistency R](crow_eliminating_backdoors_from_large_language_models_via_internal_consistency_r.md)**

:   提出 CROW（Internal Consistency Regularization），通过对抗扰动 + 层间隐藏状态一致性正则化来消除 LLM 中的后门，仅需 100 条干净样本、单卡 4 分钟微调即可将攻击成功率降至 5% 以下，且不需要干净参考模型或触发器先验知识。

**[De-Antifake Rethinking The Protective Perturbations Against Voice Cloning Attack](de-antifake_rethinking_the_protective_perturbations_against_voice_cloning_attack.md)**

:   本文首次系统评估了基于保护性扰动的语音克隆（Voice Cloning）防御方法在面对对抗净化时的脆弱性，并提出了一种两阶段的"净化-精炼"（Purification-Refinement）框架 PhonePuRe，利用音素引导的扩散模型有效消除保护性扰动，使语音克隆模型能够重新准确复制说话人特征，揭示了现有防御方案的根本局限性。

**[De-Mark Watermark Removal In Large Language Models](de-mark_watermark_removal_in_large_language_models.md)**

:   提出De-mark框架，通过随机选择探测(random selection probing)策略估计n-gram水印强度并重建红绿列表，无需知道哈希函数即可去除水印，并提供去除后LM分布与原始分布之间的理论差距保证。

**[Disparate Conditional Prediction In Multiclass Classifiers](disparate_conditional_prediction_in_multiclass_classifiers.md)**

:   提出 Disparate Conditional Prediction (DCP) 度量从二分类到多类分类的扩展，通过局部优化和线性规划方法为多类分类器的公平性偏离程度提供上下界估计，支持在混淆矩阵已知或仅有人口级别统计信息两种场景下进行公平性审计。

**[Distributed And Decentralised Training Technical Governance Challenges In A Shif](distributed_and_decentralised_training_technical_governance_challenges_in_a_shif.md)**

:   本文系统区分了分布式训练（multi-data centre）与去中心化训练（community-driven）两种新兴范式，分析了低通信训练算法（如 DiLoCo）如何使这两种范式成为可能，并深入讨论了它们对AI技术治理（计算结构化、能力扩散、可关停性）带来的挑战与机遇。

**[Do Not Mimic My Voice Speaker Identity Unlearning For Zero-Shot Text-To-Speech](do_not_mimic_my_voice_speaker_identity_unlearning_for_zero-shot_text-to-speech.md)**

:   首次提出零样本TTS中的说话人身份遗忘任务，设计了Teacher-Guided Unlearning (TGU) 框架，通过引入随机性使模型"忘记"目标说话人的声纹特征，同时保持对其他说话人的高质量语音合成能力，并提出 spk-ZRF 指标量化遗忘效果。

**[Egoprivacy What Your First-Person Camera Says About You](egoprivacy_what_your_first-person_camera_says_about_you.md)**

:   提出 EgoPrivacy——首个大规模第一人称视频隐私基准，定义三类隐私（人口统计/个体/情境）七大任务，并设计检索增强攻击 (RAA) 将 ego-to-exo 检索与分类联合，证明基础模型零样本即可以 70–80% 准确率推断佩戴者性别、种族等敏感属性。

**[Emergent Misalignment Narrow Finetuning Can Produce Broadly Misaligned Llms](emergent_misalignment_narrow_finetuning_can_produce_broadly_misaligned_llms.md)**

:   在窄领域任务（如写不安全代码）上微调对齐LLM，会导致模型在完全无关的开放式问答中表现出广泛的失对齐行为——反人类言论、恶意建议和欺骗性回答——作者将此称为"涌现式失对齐"。

**[Empirical Privacy Variance](empirical_privacy_variance.md)**

:   揭示了在相同 $(ε,δ)$-DP 保证下，DP-SGD 不同超参数配置训练出的语言模型在经验隐私（记忆化程度）上存在显著差异，并提出了兼顾经验隐私的超参数选择启发式方法。

**[Faster Rates For Private Adversarial Bandits](faster_rates_for_private_adversarial_bandits.md)**

:   为差分隐私对抗性 bandits 问题提出简洁高效的非私有→私有转换框架，通过批量化损失+Laplace 噪声实现 O(√(KT/ε)) 的后悔界，首次证明中心 DP 和本地 DP 在该问题上存在分离，并给出首个私有 bandits with expert advice 算法。

**[Federated In-Context Learning Iterative Refinement For Improved Answer Quality](federated_in-context_learning_iterative_refinement_for_improved_answer_quality.md)**

:   本文提出 Fed-ICL，一种联邦 In-Context Learning 框架，通过客户端与服务端之间的多轮迭代协作，在不传输模型参数的情况下利用分散在各客户端的高质量示例逐步改善回答质量，并建立了收敛保证。

**[Ferret Federated Full-Parameter Tuning At Scale For Large Language Models](ferret_federated_full-parameter_tuning_at_scale_for_large_language_models.md)**

:   提出 Ferret，首个结合一阶优化与共享随机性的联邦全参数微调方法，通过将本地更新投影到低维空间实现 $10^6\times$ 通信压缩和 $6\times$ 计算加速，同时保持与 FedAvg 相当的模型精度。

**[Ficgcn Unveiling The Homomorphic Encryption Efficiency From Irregular Graph Conv](ficgcn_unveiling_the_homomorphic_encryption_efficiency_from_irregular_graph_conv.md)**

:   提出FicGCN框架利用GCN的稀疏特性在同态加密(HE)下实现高效隐私推理：延迟感知打包方案+稀疏密文内聚合(SpIntra-CA)+区域数据重排，在保护数据隐私的同时比最新方案加速4.10倍。

**[Generalization In Federated Learning A Conditional Mutual Information Framework](generalization_in_federated_learning_a_conditional_mutual_information_framework.md)**

:   首次将条件互信息（CMI）框架引入联邦学习的两层泛化分析，通过构造"超级客户端"和"超级样本"，推导出有界的 CMI 泛化界，证明隐私约束蕴含泛化保证，并给出在低经验风险区间恢复最优收敛率的快速率界。

**[Iclshield Exploring And Mitigating In-Context Learning Backdoor Attacks](iclshield_exploring_and_mitigating_in-context_learning_backdoor_attacks.md)**

:   首次提出"双重学习假说"揭示 ICL 后门攻击的理论机制，并设计 ICLShield 防御方法，通过动态添加高置信度和高相似度的干净示例来调节概念偏好比，平均降低攻击成功率 26.02%。

**[Identifying And Understanding Cross-Class Features In Adversarial Training](identifying_and_understanding_cross-class_features_in_adversarial_training.md)**

:   从类别级特征归因的角度揭示对抗训练(AT)中的"跨类特征"如何先被学习后被遗忘，统一解释了鲁棒过拟合和软标签训练优势两大现象。

**[Improving The Variance Of Differentially Private Randomized Experiments Through ](improving_the_variance_of_differentially_private_randomized_experiments_through_.md)**

:   提出 Cluster-DP 机制，利用非敏感的聚类结构信息改善差分隐私随机实验中因果效应估计的隐私-方差权衡，在不牺牲隐私保证的前提下，通过更同质的聚类结构显著降低 ATE 估计的方差损失。

**[Improving Your Model Ranking On Chatbot Arena By Vote Rigging](improving_your_model_ranking_on_chatbot_arena_by_vote_rigging.md)**

:   论文揭示 Chatbot Arena 的众包投票机制可被恶意操纵：提出 target-only 和 omnipresent 两类投票操纵策略，其中 omnipresent 策略利用 Bradley-Terry 评分系统的全局耦合特性，仅需操纵数百票即可将目标模型排名提升 15 位，凸显当前 LLM 评估平台的安全脆弱性。

**[Invariance Makes Llm Unlearning Resilient Even To Unanticipated Downstream Fine-](invariance_makes_llm_unlearning_resilient_even_to_unanticipated_downstream_fine-.md)**

:   将不变风险最小化（IRM）引入 LLM 遗忘框架，提出 ILU 正则化方法，使被遗忘的知识在后续下游微调中不会被恢复，仅用单个无关微调数据集即可泛化到多个未知下游任务。

**[Is Your Model Fairly Certain Uncertainty-Aware Fairness Evaluation For Llms](is_your_model_fairly_certain_uncertainty-aware_fairness_evaluation_for_llms.md)**

:   提出不确定性感知的公平性指标 UCerF，以及大规模性别-职业偏见评估数据集 SynthBias（31,756样本），通过联合分析预测正确性与模型不确定性来更精细地评估LLM的内在偏见。

**[Learning Safety Constraints For Large Language Models](learning_safety_constraints_for_large_language_models.md)**

:   论文提出 SaP（Safety Polytope）：在 LLM 表征空间中学习一个“安全多面体”，并在推理时把不安全生成轨迹几何地拉回安全区域，以在不改模型权重的前提下实现可解释的安全约束。

**[On Differential Privacy For Adaptively Solving Search Problems](on_differential_privacy_for_adaptively_solving_search_problems.md)**

:   本文首次将差分隐私技术扩展到搜索问题（返回解向量而非数值），提出用于自适应近似近邻搜索和回归问题的高效数据结构，在温和假设下实现 $O(\sqrt{T})$ 份数据结构副本即可正确回答 $T$ 个自适应查询。

**[On Differential Privacy For Adaptively Solving Search Problems Via Sketching](on_differential_privacy_for_adaptively_solving_search_problems_via_sketching.md)**

:   首次将差分隐私技术拓展到**搜索问题**（近似最近邻查询和回归解向量输出），在稀疏邻域假设和良好条件数假设下，实现仅需 $\widetilde{O}(\sqrt{T})$ 份数据结构副本即可应对 $T$ 次自适应查询的搜索型数据结构。

**[Privacy-Shielded Image Compression Defending Against Exploitation From Vision-La](privacy-shielded_image_compression_defending_against_exploitation_from_vision-la.md)**

:   提出了 Privacy-Shielded Image Compression (PSIC)，通过在学习图像压缩解码阶段注入条件触发偏置，实现一条码流的双模式解码——默认模式保留视觉感知质量但屏蔽 VLP 模型的语义理解，授权模式则完整恢复图像语义，从而在压缩阶段为用户提供即插即用的隐私保护能力。

**[Private Model Personalization Revisited](private_model_personalization_revisited.md)**

:   在用户级差分隐私约束下，提出高效联邦模型个性化算法 Private FedRep，通过共享低维嵌入学习实现个性化，在自然参数区间下将隐私误差相比先前工作降低 O(dk) 倍，并给出维度无关的分类风险界。

**[Quadratic Upper Bound For Boosting Robustness](quadratic_upper_bound_for_boosting_robustness.md)**

:   利用交叉熵损失关于 logit 的凸性，推导出对抗训练损失的二次上界 (QUB)，作为即插即用的损失函数替换应用于现有快速对抗训练方法，显著提升鲁棒性。

**[Relative Error Fair Clustering In The Weak-Strong Oracle Model](relative_error_fair_clustering_in_the_weak-strong_oracle_model.md)**

:   提出首个在弱强预言机模型下实现 $(1+\varepsilon)$ 逼近的公平 $k$-median 聚类算法，仅需 $\text{poly}(k \log n / \varepsilon)$ 次昂贵的强预言机查询，相比此前大于 10 的常数因子逼近有根本性提升。

**[Rethinking The Bias Of Foundation Model Under Long-Tailed Distribution](rethinking_the_bias_of_foundation_model_under_long-tailed_distribution.md)**

:   揭示基础模型微调在长尾任务上受"参数不平衡"（预训练数据偏差）和"数据不平衡"（下游数据偏差）的双重影响，发现参数不平衡更关键且无法被现有 logit 调整方法解决，提出基于因果后门调整的方法消除不完整语义因子的混杂效应，在三个长尾基准上平均提升约 1.67%。

**[Retraining With Predicted Hard Labels Provably Increases Model Accurac](retraining_with_predicted_hard_labels_provably_increases_model_accurac.md)**

:   在噪声标签下，用模型自身预测的硬标签（0/1标签）对训练集重新标注并重训练，可以**理论上可证明地**提升模型准确率；进一步提出 consensus-based retraining（仅对预测标签与给定标签一致的样本重训练），在 label DP 场景下无额外隐私代价即可大幅提升性能。

**[Retraining With Predicted Hard Labels Provably Increases Model Accuracy](retraining_with_predicted_hard_labels_provably_increases_model_accuracy.md)**

:   在噪声标签场景下，用模型自身预测的硬标签（0/1）对训练集重新标注并重训练，可以**可证明地**提升分类精度；进一步提出共识筛选策略（仅对预测标签与给定标签一致的样本重训练），在标签差分隐私训练中无额外隐私代价即可大幅提升性能。

**[Revealing Weaknesses In Text Watermarking Through Self-Information Rewrite Attac](revealing_weaknesses_in_text_watermarking_through_self-information_rewrite_attac.md)**

:   提出 SIRA（Self-Information Rewrite Attack），利用自信息识别水印嵌入的高熵 token 并进行定向替换，在 7 种主流水印方法上实现近 100% 攻击成功率，成本仅 $0.88/百万 token，且完全黑盒、可迁移至任意 LLM 甚至移动端模型。

**[Robust Multi-Bit Text Watermark With Llm-Based Paraphrasers](robust_multi-bit_text_watermark_with_llm-based_paraphrasers.md)**

:   提出基于LLM释义器（paraphraser）的多比特文本水印方法，通过共训练一对行为差异化的释义器和一个解码分类器，利用PPO强化学习优化编码-解码对，在1.1B小模型上实现>99.99% AUC的检测精度，同时保持文本语义不变。

**[Secemb Sparsity-Aware Secure Federated Learning Of On-Device Recommender System ](secemb_sparsity-aware_secure_federated_learning_of_on-device_recommender_system_.md)**

:   提出 SecEmb，一种利用嵌入更新稀疏性的无损安全联邦推荐协议，通过函数秘密共享（FSS）在保护用户评分物品索引和梯度隐私的同时，将上传/下载通信开销降低最高 90 倍、用户端计算时间降低最高 70 倍。

**[Solving Probabilistic Verification Problems Of Neural Networks Using Branch And ](solving_probabilistic_verification_problems_of_neural_networks_using_branch_and_.md)**

:   本文提出一种基于分支定界（Branch and Bound）的神经网络概率验证算法，通过迭代精化输出概率的上下界来回答"给定输入分布下，网络输出满足特定条件的概率是多少"，速度比已有方法快一到两个数量级。

**[Sorbet A Neuromorphic Hardware-Compatible Transformer-Based Spiking Language Mod](sorbet_a_neuromorphic_hardware-compatible_transformer-based_spiking_language_mod.md)**

:   提出 Sorbet，首个完全兼容神经形态硬件的 Transformer 脉冲语言模型，通过两项关键创新——基于位移的 PTsoftmax 和 Bit Shifting PowerNorm (BSPN)——替代传统的 softmax 和层归一化，在 GLUE 基准上实现与 BERT 可比的性能，同时节省 27.16 倍能耗。

**[The Canarys Echo Auditing Privacy Risks Of Llm-Generated Synthetic Text](the_canarys_echo_auditing_privacy_risks_of_llm-generated_synthetic_text.md)**

:   本文设计了针对 LLM 生成的合成数据的成员推断攻击（MIA），揭示合成数据会泄露训练数据信息；进一步发现针对模型的金丝雀（canary）在合成数据发布场景下效果不佳，提出利用自回归模型特性设计的新型金丝雀——拥有同分布前缀和高困惑度后缀，能在合成数据中留下可检测的痕迹，显著提升隐私审计能力。

**[The Ripple Effect On Unforeseen Complications Of Backdoor Attacks](the_ripple_effect_on_unforeseen_complications_of_backdoor_attacks.md)**

:   首次系统量化了后门预训练语言模型在无关下游任务上的"并发症"现象——后门触发词会使下游模型的输出分布严重偏斜（甚至99%集中到单一类别），并提出基于多任务学习的无需下游任务知识的缓解方法。

**[Theoretically Unmasking Inference Attacks Against Ldp-Protected Client Data In ](theoretically_unmasking_inference_attacks_against_ldp-protected_client_data_in_.md)**

:   本文为联邦学习中恶意服务器的主动成员推断攻击（AMI）提供了首个理论分析框架，推导出即使在 LDP 保护下攻击成功率的下界和上界，揭示 LDP 保护强度与模型效用之间的根本矛盾。

**[Theoretically Unmasking Inference Attacks Against Ldp-Protected Clients In Feder](theoretically_unmasking_inference_attacks_against_ldp-protected_clients_in_feder.md)**

:   首次为联邦学习中基于全连接层和自注意力层的**主动成员推断攻击（AMI）**在**LDP保护下**推导出理论成功率的上下界，揭示即使在LDP保护下，隐恓风险仍依赖于隐私预算 $\varepsilon$，且要有效缓解攻击所需的噪声会严重损害模型效用。

**[Towards Trustworthy Federated Learning With Untrusted Participants](towards_trustworthy_federated_learning_with_untrusted_participants.md)**

:   提出 CafCor 算法，通过参与者间的共享随机性实现关联噪声注入，结合新型鲁棒聚合方法 CAF，在不信任服务器、存在恶意参与者的联邦学习场景下，实现接近中心化 DP 的隐私-效用权衡。

**[Tuco Measuring The Contribution Of Fine-Tuning To Individual Responses Of Llms](tuco_measuring_the_contribution_of_fine-tuning_to_individual_responses_of_llms.md)**

:   提出 Tuning Contribution (TuCo) 指标，通过将微调后 LLM 的前向传播精确分解为预训练分量 (PTC) 和微调分量 (FTC)，首次实现在推理时逐 prompt 量化微调对模型输出的贡献，并揭示越狱攻击通过削弱 FTC 幅度来绕过安全防护。

**[Understanding Model Ensemble In Transferable Adversarial Attack](understanding_model_ensemble_in_transferable_adversarial_attack.md)**

:   首次为模型集成对抗攻击建立理论框架，定义 transferability error 并将其分解为脆弱性（vulnerability）与多样性（diversity），再利用信息论工具给出上界，从理论上验证了"更多模型+更高多样性+更低复杂度"三条实践指南。

**[Watch Out Your Album On The Inadvertent Privacy Memorization In Multi-Modal Larg](watch_out_your_album_on_the_inadvertent_privacy_memorization_in_multi-modal_larg.md)**

:   揭示多模态大语言模型（MLLM）在微调过程中会不经意地记忆与训练任务完全无关的私密内容（如随机水印），这种记忆源于 mini-batch 内的虚假相关性，并提出基于层级探针的检测框架证明模型内部表示已编码此类信息，即使模型输出不直接显示。

**[X-Transfer Attacks Towards Super Transferable Adversarial Attacks On Clip](x-transfer_attacks_towards_super_transferable_adversarial_attacks_on_clip.md)**

:   提出 X-Transfer 攻击方法，通过高效的代理模型缩放策略（基于多臂老虎机的动态选择），生成具有"超级迁移性"的通用对抗扰动（UAP），单一扰动可同时跨数据、跨领域、跨模型、跨任务攻击各种 CLIP 编码器和下游 VLM。
