---
title: >-
  NeurIPS2025 LLM 安全方向60篇论文解读
description: >-
  60篇NeurIPS2025的 LLM 安全方向论文解读，涵盖对抗鲁棒、LLM、联邦学习、推理、强化学习、水印/隐写等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# � LLM 安全

**🧠 NeurIPS2025** · **60** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (21)](../../ACL2026/llm_safety/index.md) · [📷 CVPR2026 (16)](../../CVPR2026/llm_safety/index.md) · [🔬 ICLR2026 (36)](../../ICLR2026/llm_safety/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/llm_safety/index.md) · [📹 ICCV2025 (8)](../../ICCV2025/llm_safety/index.md) · [🧪 ICML2025 (32)](../../ICML2025/llm_safety/index.md)

🔥 **高频主题：** 对抗鲁棒 ×13 · LLM ×11 · 联邦学习 ×4 · 推理 ×3 · 强化学习 ×3

**[A Cramér–von Mises Approach to Incentivizing Truthful Data Sharing](a_cramrvon_mises_approach_to_incentivizing_truthful_data_sha.md)**

:   提出一种基于 Cramér-von Mises 两样本检验统计量的激励机制，在贝叶斯和无先验两种设定下均能证明"如实提交数据"构成（近似）Nash 均衡，同时鼓励参与者提交更多真实数据，且不依赖对数据分布的强假设（如高斯、伯努利）。

**[A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](a_reliable_cryptographic_framework_for_empirical_machine_unl.md)**

:   将机器遗忘的评估问题建模为密码学博弈（unlearning sample inference game），通过定义adversary的"advantage"来衡量遗忘质量，克服了传统MIA准确率作为评估指标的多种缺陷（不以retrain为零基准、对数据划分敏感、对MIA选择敏感），并提出SWAP test作为高效的实用近似方案。

**[Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)**

:   提出 FedLEASE——解决联邦 LoRA 微调中两个关键问题：(1) 用 LoRA B 矩阵相似度聚类自动确定最优专家数量和分配，(2) 用扩展路由空间（$2M-1$ 维）实现自适应 top-M 专家选择（每个客户端自动决定用几个专家），在 GLUE 上比最强基线平均提升 5.53%。

**[Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text](adversarial_paraphrasing_a_universal_attack_for_humanizing_ai-generated_text.md)**

:   提出 Adversarial Paraphrasing——一种无需训练的通用攻击框架，在逐 token 改写时利用 AI 文本检测器的反馈信号选择"最像人写"的 token，使改写后的 AI 文本在 8 种检测器上平均 T@1%F 下降 87.88%，且具有跨检测器的强迁移性。

**[AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text](agentstealth_reinforcing_large_language_model_for_anonymizing_user-generated_tex.md)**

:   提出 AgentStealth 框架，通过对抗式匿名化工作流、监督微调（SFT）和在线强化学习三阶段训练小型语言模型（SLM），实现在保持文本效用的同时有效匿名化用户生成内容，匿名化效果提升12.3%、效用提升6.8%。

**[ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models](almguard_safety_shortcuts_and_where_to_find_them_as_guardrails_for_audio-languag.md)**

:   首个针对音频语言模型（ALM）越狱攻击的防御框架——发现对齐过的 ALM 存在可被激活的潜在安全快捷路径（safety shortcuts），通过 Mel 梯度稀疏掩码（M-GSM）定位关键频率段，施加快捷路径激活扰动（SAP），将平均攻击成功率从 41.6% 降至 4.6%，同时几乎不影响正常任务性能。

**[Angular Steering: Behavior Control via Rotation in Activation Space](angular_steering_behavior_control_via_rotation_in_activation_space.md)**

:   提出Angular Steering，将LLM激活引导统一建模为固定2D子空间中的旋转操作——通过旋转角度提供0°-360°的连续、细粒度、范数保持的行为控制旋钮，统一了激活加法和方向消融为旋转的特例，在Llama 3/Qwen 2.5/Gemma 2（3B-14B）上实现鲁棒的行为调控。

**[Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks Against LLMs](bits_leaked_per_query_information-theoretic_bounds_on_adversarial_attacks_agains.md)**

:   将 LLM 对抗攻击建模为信息通道问题——定义每次查询的"泄漏比特数" $I(Z;T)$ 为攻击目标属性 $T$ 与可观测信号 $Z$ 的互信息，证明攻击达到误差 $\varepsilon$ 所需最少查询数为 $\log(1/\varepsilon)/I(Z;T)$，在 7 个 LLM 上验证：暴露 answer tokens 需 ~1000 次查询，加 logits 降到 ~100 次，加思维链降到 ~几十次，为透明性-安全性权衡提供首个原则性标尺。

**[Buffer Layers for Test-Time Adaptation](buffer_layers_for_test-time_adaptation.md)**

:   提出 Buffer 层作为测试时自适应 (TTA) 的新范式，替代传统的归一化层更新，从根本上保留预训练骨干网络的完整性，有效缓解灾难性遗忘并在多种架构和 TTA 框架中实现一致的性能提升。

**[Collective Narrative Grounding: Community-Coordinated Data Contributions to Improve Local AI Systems](collective_narrative_grounding_community-coordinated_data_contributions_to_impro.md)**

:   提出 Collective Narrative Grounding 协议，通过参与式工坊收集社区叙事并结构化为"叙事单元"，用 RAG 管道将本地知识注入 LLM 问答系统，在 LocalBench 上发现 76.7% 的错误可由本地叙事直接修复，GPT-5 在参与式 QA 集上仅 21% 正确率凸显了本地知识鸿沟。

**[Contextual Integrity in LLMs via Reasoning and Reinforcement Learning](contextual_integrity_in_llms_via_reasoning_and_reinforcement_learning.md)**

:   提出 CI-RL 框架，通过 Chain-of-Thought 推理提示 + GRPO 强化学习，用仅约 700 个合成样本训练 LLM 理解"上下文完整性"（contextual integrity），在 PrivacyLens 基准上将隐私泄露率降低最高 40%，且小模型训练后可超越更大基线模型。

**[CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment](coreguard_safeguarding_foundational_capabilities_of_llms_against_model_stealing_.md)**

:   提出 CoreGuard，通过行置换（row permutation）锁定 Transformer 线性层权重 + 列置换传播协议（propagation protocol）将 TEE 授权次数降至 1 次，以极低计算和通信开销保护边缘部署 LLM 的基础能力不被模型窃取攻击利用。

**[CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming](cpret_a_dataset_benchmark_and_model_for_retrieval_in_competitive_programming.md)**

:   针对竞赛编程中重复/相似题目泛滥导致比赛不公平及 LLM 评测分数虚高的问题，构建了包含四种检索任务的大规模基准 CPRet，并提出 Group-InfoNCE 损失训练的专用检索模型 CPRetriever，在所有任务上超越 20+ 现有嵌入模型，同时揭示了题目相似性对 LiveCodeBench 评测的系统性偏差。

**[CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing](cryptomoe_privacy-preserving_and_scalable_mixture_of_experts_inference_via_balan.md)**

:   首个支持 MoE 架构 LLM 隐私推理的框架 CryptoMoE，通过平衡专家路由隐藏路由信息、置信度感知调度协议和批量密文矩阵乘法协议，相比 dense baseline 实现 2.8~3.5× 延迟降低和 2.9~4.3× 通信量降低，准确率损失仅 0.8%。

**[DeepPersona: A Generative Engine for Scaling Deep Synthetic Personas](deeppersona_a_generative_engine_for_scaling_deep_synthetic_personas.md)**

:   提出 DeepPersona——一个两阶段分类引导的合成人格生成引擎：先从真实用户-ChatGPT 对话中挖掘构建 8000+ 节点的人类属性分类树，再通过渐进式属性采样生成平均 200+ 结构化属性的叙事完整人格，在个性化 QA 准确率上提升 11.6%，社会调查模拟偏差缩小 31.7%。

**[Demystifying Language Model Forgetting with Low-Rank Example Associations](demystifying_language_model_forgetting_with_low-rank_example_associations.md)**

:   发现 LLM 微调后上游样本遗忘与新学任务之间的关联矩阵具有低秩结构（rank-3 即 $R^2 > 0.69$），利用矩阵补全预测未见任务导致的遗忘，指导选择性回放以减轻遗忘。

**[Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)**

:   提出FedASK框架，通过**双阶段sketching流水线**（randomized SVD启发），首次在差分隐私下实现联邦LoRA中**两个低秩矩阵A和B的同步有效更新**，在Llama-2 7B/13B上MMLU提升最高11.5%，GSM8K提升46%。

**[Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values](distributive_fairness_in_large_language_models_evaluating_alignment_with_human_v.md)**

:   本文系统评估多个 SOTA LLM（GPT-4o、Claude-3.5S、Llama3-70b、Gemini-1.5P）在非策略性资源分配任务中的分配公平性偏好，发现 LLM 与人类存在显著偏差：LLM 偏好效率和无嫉妒性 (EF) 而忽视人类更看重的公平性/平等性 (EQ)，但在选择题模式下 GPT-4o 和 Claude 能正确识别公平方案。

**[DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm](dna-detectllm_unveiling_ai-generated_text_via_a_dna-inspired_mutation-repair_par.md)**

:   本文提出 DNA-DetectLLM，一种受 DNA 突变修复机制启发的零样本 AI 文本检测方法，通过构造理想 AI 序列并量化将输入文本修复到该序列的累积难度作为检测信号，在多个基准数据集上取得 AUROC 相对提升 5.55%、F1 提升 2.08% 的 SOTA 效果。

**[Enhancing CLIP Robustness via Cross-Modality Alignment](enhancing_clip_robustness_via_crossmodality_alignment.md)**

:   提出COLA——一个training-free的框架，通过将对抗扰动后的图像特征投影到文本特征张成的子空间来消除非语义噪声，再用最优传输(OT)在分布层面细粒度对齐图文特征，在14个零样本分类基准上平均提升6.7%的对抗鲁棒准确率，同时维持干净样本性能。

**[Enhancing Sample Selection Against Label Noise by Cutting Mislabeled Easy Examples](enhancing_sample_selection_against_label_noise_by_cutting_mislabeled_easy_exampl.md)**

:   发现并定义了误标注易学样本（Mislabeled Easy Examples, MEEs）——被模型早期训练即正确预测为错误标签的样本对泛化伤害最大，并提出 Early Cutting 方法利用模型后期状态重新校准早期置信子集来过滤MEEs。

**[Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions](evaluating_the_promise_and_pitfalls_of_llms_in_hiring_decisions.md)**

:   在约 10,000 个真实招聘候选人-职位配对上系统评测了 GPT-4o/4.1、Claude 3.5、Gemini 2.5、Llama 3.1/4、DeepSeek R1 等主流 LLM 的招聘匹配表现，发现专用领域模型 Match Score 在准确性（AUC 0.85 vs 0.77）和公平性（种族 IR 0.957 vs ≤0.809）上全面优于通用 LLM。

**[Exploring the Limits of Strong Membership Inference Attacks on Large Language Models](exploring_the_limits_of_strong_membership_inference_attacks_on_large_language_mo.md)**

:   首次将强成员推断攻击（LiRA）扩展到10M~1B参数的GPT-2规模LLM，训练超过4000个参考模型，揭示四个关键发现：强MIA可以在LLM上成功但效果有限（AUC<0.7），且大量个体样本决策在训练随机性下**与抛硬币无法区分**。

**[FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models](fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)**

:   FedRW 提出首个无需可信第三方的联邦学习隐私保护软去重框架，通过安全多方计算获取全局样本频率并进行频率感知的样本加权，在预处理上实现最高 28.78× 加速，在模型性能上实现约 11.42% 的 perplexity 改善。

**[FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)**

:   FedSVD 提出通过 SVD 对 LoRA 矩阵进行全局重参数化，在每轮通信后用聚合的 BA 乘积的右奇异向量更新 A 矩阵，避免 DP-SGD 下的二次噪声放大同时保持 A 的自适应能力，在多个 NLU 基准上一致超越固定 A 的基线。

**[Finding Structure in Continual Learning](finding_structure_in_continual_learning.md)**

:   提出基于Douglas-Rachford Splitting (DRS)的持续学习优化框架，将稳定性与可塑性解耦为两个独立的近端子问题，并结合Rényi散度替代KL散度实现更鲁棒的先验对齐，从而在无需回放缓冲区或额外模块的条件下有效缓解灾难性遗忘。

**[Geo-Sign: Hyperbolic Contrastive Regularisation for Geometrically Aware Sign Language Translation](geo-sign_hyperbolic_contrastive_regularisation_for_geometrically_aware_sign_lang.md)**

:   Geo-Sign 提出将骨架特征投影到 Poincaré 球模型的双曲空间中，通过双曲对比损失正则化 mT5 语言模型，使其感知手语运动的层次结构，仅用骨架数据就在 CSL-Daily 上超越了基于 RGB 的 SOTA 方法（BLEU-4 +1.81, ROUGE-L +3.03）。

**[HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring](healthslm-bench_benchmarking_small_language_models_for_mobile_and_wearable_healt.md)**

:   首个系统评估小语言模型 (SLMs, 1-4B参数) 在移动与可穿戴健康监测任务上表现的基准，覆盖zero-shot/few-shot/指令微调三种范式，并在iPhone上验证了端侧部署的可行性。

**[InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)**

:   提出 InvisibleInk 框架，通过差分裁剪（DClip）隔离敏感信息和 Top-$k^+$ 截断采样两项创新，将差分隐私长文本生成的计算成本降低 8 倍以上，首次实现不到非隐私生成 4-8 倍开销的高质量隐私文本生成。

**[Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)**

:   提出LTW（Learning to Watermark）框架，使用一个轻量级选择器网络基于句子嵌入、token熵和当前水印比例来自适应决定何时施加水印，通过多目标优化（MGDA）在可检测性和文本质量之间达到Pareto最优，在不降低检测性能的前提下显著提升水印文本质量。

**[LLM Strategic Reasoning: Agentic Study through Behavioral Game Theory](llm_strategic_reasoning_agentic_study_through_behavioral_gam.md)**

:   本文提出基于行为博弈论的LLM战略推理评估框架，使用截断量子响应均衡(TQRE)量化推理深度τ，在13个矩阵博弈上评估22个SOTA模型，揭示推理风格差异和人口统计persona引发的偏差问题。

**[MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction](masksql_safeguarding_privacy_for_llm-based_text-to-sql_via_abstraction.md)**

:   提出 MaskSQL 框架，通过提示抽象（abstraction）将敏感的表名、列名和数据值替换为抽象符号后发送给远程 LLM，结合本地 SLM 做 schema linking 和 SQL 重建，在保护隐私同时超越 SLM-only 方案的 SQL 生成精度。

**[MixAT: Combining Continuous and Discrete Adversarial Training for LLMs](mixat_combining_continuous_and_discrete_adversarial_training_for_llms.md)**

:   提出MixAT方法，将离散对抗攻击（PAP改写）与连续嵌入空间扰动相结合进行LLM对抗训练，在保持高效用的同时实现对多种攻击的鲁棒性（ALO-ASR从50%+降至20%以下），且训练成本仅与纯连续方法相当。

**[MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference](mpcache_mpc-friendly_kv_cache_eviction_for_efficient_private_llm_inference.md)**

:   本文提出MPCache，一个面向安全多方计算（MPC）的KV缓存淘汰框架，结合一次性静态淘汰和查询感知的动态选择，配合层次化聚类、线性化相似度近似和跨层索引共享等优化，在不牺牲LLM性能的前提下实现最高2.01倍延迟降低和8.37倍通信量削减。

**[Music Arena: Live Evaluation for Text-to-Music](music_arena_live_evaluation_for_text-to-music.md)**

:   Music Arena是首个面向文本到音乐（TTM）生成的在线实时评估平台，通过LLM驱动的审核与路由系统解决TTM系统异构签名问题，收集包含细粒度聆听行为和自然语言反馈的多层次偏好数据，并通过月度滚动数据发布为社区提供可持续的开放偏好数据源。

**[On the Empirical Power of Goodness-of-Fit Tests in Watermark Detection](on_the_empirical_power_of_goodness-of-fit_tests_in_watermark_detection.md)**

:   系统性地评估了八种经典拟合优度（GoF）检验在 LLM 文本水印检测中的效果，发现 GoF 检验在检测功效和鲁棒性上均显著优于现有基线方法。

**[On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)**

:   首次系统研究 LLM 语言化置信度（verbal confidence）在对抗攻击下的鲁棒性，提出基于扰动和越狱的攻击框架，揭示攻击可导致置信度下降最高 30%、答案翻转率高达 100%，且现有防御策略基本无效。

**[On the Sample Complexity of Differentially Private Policy Optimization](on_the_sample_complexity_of_differentially_private_policy_optimization.md)**

:   首次系统性研究差分隐私（DP）约束下策略优化的样本复杂度，提出统一的元算法框架，分析DP-PG、DP-NPG和DP-REBEL三种隐私策略优化算法，证明隐私代价通常仅作为样本复杂度的低阶项出现。

**[ORBIT -- Open Recommendation Benchmark for Reproducible Research with Hidden Tests](orbit_--_open_recommendation_benchmark_for_reproducible_research_with_hidden_tes.md)**

:   提出ORBIT统一推荐系统基准，包含5个标准化公开数据集评估和基于真实浏览历史构建的隐私安全ClueWeb-Reco隐藏测试集，系统评估了12个推荐模型并引入LLM-QueryGen基线，揭示了现有方法在大规模真实推荐场景中的局限性。

**[Poly-Guard: Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset](poly-guard_massive_multi-domain_safety_policy-grounded_guardrail_dataset.md)**

:   提出首个**大规模、多领域、策略驱动**的安全护栏基准 Poly-Guard，从 150+ 真实行业安全策略中提取 400+ 风险类别和 1000+ 安全规则，生成 100K+ 实例覆盖 8 大安全关键领域，并系统评测 19 个护栏模型，揭示了领域特化、模型演进遗忘、模型缩放停滞、对抗脆弱性等 8 项关键发现。

**[Probabilistic Reasoning with LLMs for K-Anonymity Estimation](probabilistic_reasoning_with_llms_for_k-anonymity_estimation.md)**

:   本文提出Branch框架，利用大语言模型将用户文本中的个人信息建模为贝叶斯网络的联合概率分布，分别估计各属性的条件概率后组合计算k-匿名值（全球匹配该信息的人数），在隐私风险估计任务上达到73%准确率，比o3-mini链式思维提升13%。

**[Procurement Auctions with Predictions: Improved Frugality for Facility Location](procurement_auctions_with_predictions_improved_frugality_for_facility_location.md)**

:   研究策略性无容量限制设施选址问题中的采购拍卖设计，证明了经典VCG拍卖的节俭比恰好为3（改进了此前已知的上界4），并设计了利用预测信息的学习增强拍卖机制，在预测准确时实现接近最优的节俭比，同时在预测任意不准确时仍保持常数级鲁棒性。

**[PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning](pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)**

:   本文提出 PULSE 评估协议，从预训练知识遗忘和多次顺序遗忘的可持续性两个实际维度出发，揭示了现有遗忘方法在 LMM 上的严重不足——遗忘预训练知识会导致 90% 以上通用能力丧失，连续遗忘 5 次后模型泛化能力几乎完全崩溃。

**[Reinforcement Learning with Backtracking Feedback](reinforcement_learning_with_backtracking_feedback.md)**

:   提出带回溯反馈的强化学习框架 RLBF，当 agent 陷入死胡同时允许回溯到之前的状态重新探索，通过回溯信号改善信用分配，在稀疏奖励环境中显著提升探索效率。

**[ReliabilityRAG: Effective and Provably Robust Defense for RAG-based Web-Search](reliabilityrag_effective_and_provably_robust_defense_for_rag-based_web-search.md)**

:   ReliabilityRAG 提出了一种利用文档可靠性信号（如搜索排名）进行对抗防御的 RAG 框架，通过在矛盾图上寻找最大独立集（MIS）来识别一致的文档子集并优先选择高可靠性文档，提供可证明的鲁棒性保证，同时在良性场景和长文本生成任务上保持高准确率。

**[Reverse Engineering Human Preferences with Reinforcement Learning](reverse_engineering_human_preferences_with_reinforcement_learning.md)**

:   使用强化学习训练前导文本生成器来提升下游 LLM 的评分成绩,揭示了 LLM-as-a-Judge 评估框架的脆弱性,且该攻击方式几乎不可检测并具有跨模型迁移能力。

**[SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders](saemark_steering_personalized_multilingual_llm_watermarks_with_sparse_autoencode.md)**

:   提出SAEMark框架，利用稀疏自编码器（SAE）提取文本的语义特征浓度评分，通过推理阶段的特征引导拒绝采样实现多比特水印嵌入，无需修改模型权重或logits，天然支持黑盒API、多语言和代码等场景，在英文/中文/代码上均达到领先的水印精度与文本质量。

**[SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations](seca_semantically_equivalent_and_coherent_attacks_for_eliciting_llm_hallucinatio.md)**

:   提出 SECA（Semantically Equivalent and Coherent Attacks），通过保持语义等价和语义连贯性的现实主义提示修改来诱发 LLM 幻觉，在多选 QA 任务上实现更高攻击成功率且几乎无语义错误。

**[Self-Refining Language Model Anonymizers via Adversarial Distillation](self-refining_language_model_anonymizers_via_adversarial_distillation.md)**

:   提出 SEAL 框架，通过对抗蒸馏将 GPT-4 级 LLM 的文本匿名化能力蒸馏到 8B 小模型中，结合 SFT + DPO 训练和自我精炼机制，使小模型在隐私-效用权衡上达到甚至超越 GPT-4 匿名化器的水平，且可完全本地部署。

**[SIMU: Selective Influence Machine Unlearning](simu_selective_influence_machine_unlearning.md)**

:   提出 SIMU 两阶段框架：先通过梯度聚合识别编码遗忘集信息的关键 MLP 神经元，再仅对这些神经元进行二阶（Sophia）优化遗忘，在保持遗忘效果的同时大幅提升模型原有能力的保留。

**[Stop DDoS Attacking the Research Community with AI-Generated Survey Papers](stop_ddos_attacking_the_research_community_with_ai-generated_survey_papers.md)**

:   这篇立场论文将AI生成综述论文的泛滥类比为对学术社区的"DDoS攻击"，通过对arXiv 2020-2024年10,063篇CS综述论文的系统定量分析，揭示了ChatGPT发布后综述论文数量、AI生成分数和异常作者数的同步激增现象，深入剖析了AI综述的四大质量缺陷（结构混乱、分类缺乏原创、引用不准确、内容高度冗余）及其对研究者-审稿人-编辑三方的文化冲击，提出了涵盖透明度要求、严格审查标准、冗余限制、AI检测辅助和"动态活综述"平台在内的全面应对框架。

**[SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications](swe-sql_illuminating_llm_pathways_to_solve_user_sql_issues_in_real-world_applica.md)**

:   提出 BIRD-CRITIC 基准（首个 SQL 问题调试基准）和 Six-Gym 训练环境，并开发 Bird-Fixer 智能体，通过 f-Plan Boosting 策略将 14B 开源模型的 SQL 调试能力提升至超越 Claude-3.7-Sonnet 和 GPT-4.1 的水平，在保护数据隐私的同时实现高效的 SQL 问题修复。

**[Teaming LLMs to Detect and Mitigate Hallucinations](teaming_llms_to_detect_and_mitigate_hallucinations.md)**

:   将单模型一致性方法（Self-Consistency + Semantic Entropy）推广到多个异构 LLM 的"联盟"设置，通过聚合不同训练背景的模型响应来打破单模型一致性幻觉，在 15 个 LLM 组成的模型池中评估大量联盟组合，发现匹配的强模型联盟在 92% 的情况下超越最强单模型基线，同时推理成本更低。

**[ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training](toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)**

:   提出 ToxicTextCLIP 框架，通过背景感知选择和背景驱动增强两个模块，在 CLIP 预训练阶段生成高质量对抗文本，实现高达 95.83% 投毒成功率和 98.68% 后门 Hit@1，且能突破 RoCLIP、CleanCLIP、SafeCLIP 三种防御。

**[Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties](trans-env_a_framework_for_evaluating_the_linguistic_robustness_of_llms_against_e.md)**

:   提出Trans-EnV框架，结合语言学专家知识和LLM变换能力，将标准美式英语（SAE）数据集自动转换为38种英语变体（18种方言+20种ESL英语），揭示LLM在非标准英语上最高46.3%的性能下降，凸显了语言公平性问题。

**[TRAP: Targeted Redirecting of Agentic Preferences](trap_targeted_redirecting_of_agentic_preferences.md)**

:   TRAP 提出了一种基于扩散模型的语义注入对抗框架，通过在 CLIP 嵌入空间中优化图像语义，在黑盒条件下以视觉自然的方式系统性地误导多个主流 VLM 智能体的决策偏好，在 LLaVA-34B、GPT-4o 等六个模型上实现了高达 100% 的攻击成功率。

**[TRUST -- Transformer-Driven U-Net for Sparse Target Recovery](trust_--_transformer-driven_u-net_for_sparse_target_recovery.md)**

:   提出 TRUST 架构，将 Transformer 的注意力机制与 U-Net 解码器结合，在感知矩阵未知的条件下同时学习感知算子和重建稀疏信号，在 SSIM 和 PSNR 上显著超越传统方法。

**[Unlearning as Ablation: Toward a Falsifiable Benchmark for Generative Scientific Discovery](unlearning_as_ablation_toward_a_falsifiable_benchmark_for_generative_scientific_.md)**

:   本文提出将机器遗忘重新定义为认识论探针工具（"遗忘即消融"），通过系统性移除目标知识及其遗忘闭包后测试模型能否从公理出发重新推导，从而提供可证伪的测试来区分 LLM 是"真正生成新知识"还是"仅仅检索记忆片段"。

**[Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data](virus_infection_attack_on_llms_your_poisoning_can_spread_via_synthetic_data.md)**

:   本文首次系统研究了合成数据在LLM训练中的安全风险，发现现有投毒/后门攻击难以通过合成数据传播，进而提出Virus Infection Attack (VIA)框架，通过劫持点搜索和外壳构造将投毒内容嵌入正常训练样本中，使恶意内容即使在干净查询下也能被模型生成并传播到下游模型。

**[When AI Democratizes Exploitation: LLM-Assisted Strategic Manipulation of Fair Division Algorithms](when_ai_democratizes_exploitation_llm-assisted_strategic_manipulation_of_fair_di.md)**

:   本文通过在 Spliddit 公平分租平台上设计四种不同的协调操纵场景（排斥性合谋、防御性反击、善意合谋、成本最小化联盟），实证地证明 LLM 可以将原本需要深厚机制设计专业知识才能进行的算法操纵行为，降低为任何用户仅需一次自然语言对话即可完成的简单操作，从根本上颠覆了"算法复杂性即安全屏障"的传统假设。
