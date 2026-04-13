---
title: >-
  ACL2025 对齐/RLHF方向 61篇论文解读
description: >-
  61篇ACL2025 对齐/RLHF方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚖️ 对齐/RLHF

**💬 ACL2025** · 共 **61** 篇

**[Agentalign Navigating Safety Alignment In The Shift From Informative To Agentic ](agentalign_navigating_safety_alignment_in_the_shift_from_informative_to_agentic_.md)**

:   本文提出 AgentAlign 框架，利用抽象行为链作为中介，在模拟环境中合成高质量的 agent 安全对齐数据（有害+良性），通过 SFT 使三类开源模型的 agent 安全性提升35.8%-79.5%，同时保持甚至提升了任务能力。

**[Agentrm Enhancing Agent Generalization With Reward Modeling](agentrm_enhancing_agent_generalization_with_reward_modeling.md)**

:   提出 AgentRM，一种可泛化的奖励模型，通过显式/隐式奖励建模和 LLM-as-a-judge 三种方式构建，配合 Best-of-N 采样和步级波束搜索引导策略模型进行测试时搜索，在 9 个 Agent 任务上平均提升 8.8 分，且展现出弱到强的泛化能力。

**[Aligned But Blind Implicit Bias](aligned_but_blind_implicit_bias.md)**

:   发现 LLM 对齐训练的矛盾效应：对齐成功消除了显式偏见（Llama 3 70B 降至 8.13%），但反而放大了隐式偏见（从 64.1% 升至 91.4%），机制是对齐使模型在歧义上下文中不再表征种族概念（"种族盲视"），导致安全护栏无法在隐性场景中激活。通过在早期层注入种族感知激活可将隐式偏见从 97.3% 降至 71.2%。

**[Amopo Adaptive Multi-Objective Preference Optimization Without Reward Models And](amopo_adaptive_multi-objective_preference_optimization_without_reward_models_and.md)**

:   提出AMoPO框架，通过将生成空间建模为高斯分布实现维度感知的自适应权重分配，在不依赖奖励模型和参考模型的情况下完成多目标偏好对齐，在HelpSteer2数据集上超越SOTA 28.5%，并在7B/14B/32B模型上验证了缩放能力。

**[Aspo Adaptive Sentence-Level Preference Optimization For Fine-Grained Multimodal](aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)**

:   提出 ASPO（Adaptive Sentence-level Preference Optimization）——将 DPO 的偏好单元从"整个回复"细化到"每个句子"，为每个句子动态计算自适应奖励值（基于模型自身预测评估正确性和重要性），在多模态推理任务上显著优于传统回复级 DPO，有效减少幻觉并提升细粒度推理能力。

**[Atyaephyra At Semeval-2025 Task 4 Low-Rank Negative Preference Optimization](atyaephyra_at_semeval-2025_task_4_low-rank_negative_preference_optimization.md)**

:   在 SemEval 2025 LLM 遗忘共享任务中，将负偏好优化 (NPO) 与低秩适配 (LoRA) 结合，利用 LoRA 的结构特性零开销获取原始模型分布来计算 KL 散度正则化，显著稳定了遗忘过程并超越了任务基线。

**[Automixalign Adaptive Data Mixing](automixalign_adaptive_data_mixing.md)**

:   AutoMixAlign 提出了一种理论驱动的多任务偏好优化数据混合方法：先训练各任务的 specialist model 确定最优 loss 基线，再通过 minimax 优化自适应调整数据混合比例，优先处理 excess loss（与 specialist 的差距）最大的任务，在 helpfulness/harmlessness/reasoning 多任务 DPO 中平均提升 9.42%。

**[Beyond Surface-Level Patterns An Essence-Driven Defense Framework Against Jailbr](beyond_surface-level_patterns_an_essence-driven_defense_framework_against_jailbr.md)**

:   提出 EDDF，一种基于"攻击本质"而非表面模式的越狱防御框架：离线提取已知攻击的本质策略存入向量数据库，在线时对新查询做本质抽象+检索+细粒度判断，将攻击成功率降低至少 20% 且误报率仅 2.18%。

**[Beyond The Tip Of Efficiency Uncovering The Submerged Threats Of Jailbreak Attac](beyond_the_tip_of_efficiency_uncovering_the_submerged_threats_of_jailbreak_attac.md)**

:   系统评估 13 个 SOTA 小语言模型（<4B参数）在 5 种越狱攻击下的安全性，发现 SLM 虽能抵御直接攻击但在越狱攻击下显著比大模型脆弱，进一步分析了架构压缩、量化和知识蒸馏等 SLM 技术对安全性的影响。

**[Boosting Vulnerability Detection Of Llms Via Curriculum Preference Optimization ](boosting_vulnerability_detection_of_llms_via_curriculum_preference_optimization_.md)**

:   提出 ReVD 框架，通过双向漏洞推理数据合成 + 三元组 SFT（同时学习漏洞代码/修复代码/代码差异的推理）+ 课程化在线偏好优化（COPO），将 LLM 的漏洞检测准确率提升 12-23%，在 PrimeVul 和 SVEN 上达到 SOTA。

**[Breaking The Ceiling Exploring The Potential Of Jailbreak Attacks Through Expand](breaking_the_ceiling_exploring_the_potential_of_jailbreak_attacks_through_expand.md)**

:   基于精细化可能性模型 (ELM) 将越狱策略分解为四类可独立进化的组件（角色/内容支撑/语境/沟通技巧），提出 CL-GSO 遗传算法在组件级进行交叉与变异，将策略空间从既有方法的 40 种扩展到 839 种，在 Claude-3.5 上实现 96% 攻击成功率（此前方法最高仅 4%），同时提出基于意图一致性的评估机制，准确率达 96.5% 超越专用安全模型。

**[Call for Rigor in Reporting Quality of Instruction Tuning Data](call_for_rigor_in_reporting_quality_of_instruction_tuning_data.md)**

**[Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step](chain-of-jailbreak_attack_for_image_generation_models_via_ed.md)**

:   提出 CoJ（Chain-of-Jailbreak）攻击，将生成恶意图像的单步请求分解为多步编辑指令链（从无害种子图像逐步编辑到目标），绕过图像生成模型的安全过滤器，在 GPT-4o 等模型上达到高攻击成功率。

**[Chain-Of-Jailbreak Attack For Image Generation Models Via Editing Step By Step](chain-of-jailbreak_attack_for_image_generation_models_via_editing_step_by_step.md)**

:   提出 Chain-of-Jailbreak（CoJ）攻击，将无法直接绕过安全护栏的恶意 query 分解为多步编辑子 query（删然后插、插然后删、改然后改回），在 GPT-4V/4o/Gemini 上达到 60%+ 越狱成功率；同时提出 Think-Twice Prompting 防御，拦截 95%+ 的 CoJ 攻击。

**[Cheems Chinese Reward Models](cheems_chinese_reward_models.md)**

:   为弥补中文 Reward Model 资源的空白，本文构建了 CheemsBench（首个大规模中文 RM 评测基准）和 CheemsPreference（首个大规模中文偏好数据集），通过人机协作标注 + 远程监督过滤策略训练的 CheemsRM 在中文场景显著超越现有所有开源 RM。

**[Curiosity Driven Rlhf](curiosity_driven_rlhf.md)**

:   CD-RLHF 将好奇心驱动探索（curiosity-driven RL）引入 RLHF，通过前向动力学模型的预测误差作为内在奖励，结合 top-k 门控过滤与 reward whitening，在不损失对齐质量的前提下大幅提升 LLM 输出多样性（Llama-3.2-1B 上 Diversity 提升 40.26%，EAD 提升 8.92%）。

**[Debate Reflect And Distill Multi-Agent Feedback With Tree-Structured Preference ](debate_reflect_and_distill_multi-agent_feedback_with_tree-structured_preference_.md)**

:   提出 D&R 框架，让小模型（student）与多个大模型（teacher）进行多轮辩论并收集自我反思和教师反馈，然后将辩论日志组织为偏好树做 Tree-structured DPO (T-DPO) 蒸馏，在 MMLU Pro 和 MATH 上平均提升 14.18 分，且推理效率优于基线。

**[Diffpo Diffusion Alignment](diffpo_diffusion_alignment.md)**

:   提出 DiffPO，将 LLM 对齐重新建模为句子级扩散去噪过程，通过 parallel decoding 实现高效推理时对齐，作为即插即用模块可增强任意底座模型的对齐质量。

**[Dynamic Scaling Of Unit Tests For Code Reward Modeling](dynamic_scaling_of_unit_tests_for_code_reward_modeling.md)**

:   本文发现扩展LLM生成的单元测试数量可以持续提升代码奖励信号质量（尤其对困难问题效果更好），据此训练了轻量级单元测试生成模型CodeRM-8B并实现动态缩放策略，在多个代码生成基准上取得显著提升。

**[Expectation Confirmation Preference Optimization For Multi-Turn Conversational R](expectation_confirmation_preference_optimization_for_multi-turn_conversational_r.md)**

:   提出 ECPO（Expectation Confirmation Preference Optimization），首个面向 LLM 对话推荐 Agent 的多轮偏好优化方法——基于心理学期望确认理论（ECT）显式建模用户满意度在多轮对话中的演变，通过前向期望确认定位不满意根因 + 后向期望推导重写回复构建 turn-level 偏好对，配合 AILO 用户模拟器，在 3 个数据集上显著优于现有 MTPO 方法。

**[Federated Data-Efficient Instruction Tuning For Large Language Models](federated_data-efficient_instruction_tuning_for_large_language_models.md)**

:   提出 FedHDS（Federated Hierarchical Data Selection），通过 intra-client 和 inter-client 两级层次化数据选择消除联邦学习中客户端内部和跨客户端的数据冗余，结合多层 Transformer 特征融合提升 coreset 质量；仅用不到 1.5% 的数据，在 Rouge-L 上相对 SOTA 全数据联邦基线平均提升 10.72%，训练效率提升最高达 48.8 倍。

**[Finding The Sweet Spot Preference Data Construction For Scaling Preference Optim](finding_the_sweet_spot_preference_data_construction_for_scaling_preference_optim.md)**

:   发现传统的 DPO 偏好数据构建策略（max-min）在增加采样量时性能反而下降，通过基于奖励分布的系统性探索发现 rejected 响应应选在 μ−2σ 而非最小值，据此提出了一种随采样量增加而持续提升的偏好数据构建方法。

**[Fine-Grained Video Dubbing Duration Alignment With Segment Supervised Preference](fine-grained_video_dubbing_duration_alignment_with_segment_supervised_preference.md)**

:   提出 Segment Supervised Preference Optimization (SSPO)，将视频配音中译文与源语音的时长对齐问题建模为段级偏好优化，通过逐句采样+细粒度 DPO 损失实现每行对话的时长一致性，同时维持翻译质量和输出格式。

**[Focused-Dpo Enhancing Code Generation Through Focused Preference Optimization On](focused-dpo_enhancing_code_generation_through_focused_preference_optimization_on.md)**

:   发现代码生成错误集中在特定"错误易发点"（error-prone points）——前缀/后缀通常正确，错误集中在中间代码段，提出 Focused-DPO 通过 PageRank 排序定位关键代码段并在 DPO 损失中加权放大，HumanEval+ 提升 4.41%、MBPP+ 提升 6.71%。

**[Haf-Rm A Hybrid Alignment Framework For Reward Model Training](haf-rm_a_hybrid_alignment_framework_for_reward_model_training.md)**

:   提出混合对齐框架 HaF-RM，在奖励模型训练中保留策略层（policy layer），通过同时优化序列级奖励损失和 token 级策略损失来共同监督共享的内部偏好模型，在 5 个数据集上一致性超越标准 Baseline 和 DPO 方法。

**[Hiddendetect Detecting Jailbreak Attacks Against Multimodal Large Language Model](hiddendetect_detecting_jailbreak_attacks_against_multimodal_large_language_model.md)**

:   提出 HiddenDetect，一种免训练（tuning-free）的基于内部激活状态的安全检测框架：通过监控 LVLM 推理时隐藏状态中的拒绝语义信号来检测越狱攻击，在多个模型和多模态基准上 AUROC 大幅超越现有方法。

**[Influence Functions Rlhf](influence_functions_rlhf.md)**

:   首次将影响函数应用于 RLHF 奖励模型的反馈数据审计，结合 OPORP 向量压缩实现 2.5 倍加速，在偏差检测上超越 GPT-4o（AUC 0.8 vs 0.747），并从 Anthropic-HH 数据集中发现 47% 的错标样本。

**[Internal Value Alignment In Large Language Models Through Controlled Value Vecto](internal_value_alignment_in_large_language_models_through_controlled_value_vecto.md)**

:   提出 ConVA（Controlled Value Vector Activation）框架，通过上下文控制的数据集精准识别 LLM 隐空间中的价值向量，并用门控最小扰动机制在推理时激活目标价值，在 Schwartz 10 种基本价值上实现平均 29.6% 的控制成功率提升，同时保持 97%+ 的文本流畅度和通用能力。

**[Iopo Input Output Preference](iopo_input_output_preference.md)**

:   提出 IOPO（Input-Output Preference Optimization），在传统 DPO 仅优化输出偏好的基础上，引入输入偏好建模——让模型学习"给定回复 y，哪个指令 x 更匹配"，从而增强对复杂多约束指令的细粒度感知能力；同时构建了包含 120K 训练数据、1K 评测数据、覆盖 5 大类 26 个约束维度的 Trace 基准。

**[Jailbreaking One Step Is Enough](jailbreaking_one_step_is_enough.md)**

:   本文提出REDA（Reverse Embedded Defense Attack）方法，将攻击意图伪装为"防御"有害内容的任务，通过反转攻击视角+ICL示例引导+请求意图削弱，实现一步生成、跨模型通用的高成功率越狱攻击。

**[Jailbreakradar Comprehensive Assessment Jailbreak Attacks](jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)**

:   本文提出了一个全面的越狱攻击评估框架 JailbreakRadar，收集了17种代表性越狱攻击方法，建立了六类攻击分类体系，并在9个对齐LLM上进行了大规模系统性评测，揭示了不同类型攻击在实用性和防御鲁棒性上的关键差异。

**[Jsontuning Towards Generalizable Robust And Controllable Instruction Tuning](jsontuning_towards_generalizable_robust_and_controllable_instruction_tuning.md)**

:   提出 JsonTuning——将指令微调的输入输出从自然语言文本替换为 JSON 结构化格式，通过显式表示任务元素、关系和输出约束（JSON Schema），在 7 个预训练模型和 6 类任务上一致超越传统 TextTuning，平均性能从 26.78 提升到 30.88，同时显著增强鲁棒性和可控性。

**[Kpo Protein Safety](kpo_protein_safety.md)**

:   提出知识引导偏好优化（KPO）框架，通过蛋白质安全知识图谱识别安全/危险序列作为偏好信号，用强化学习训练蛋白质语言模型减少有害蛋白质序列的生成概率，同时保持功能性——为蛋白质生成的生物安全提供保障框架。

**[Llms Caught In The Crossfire Malware Requests And Jailbreak Challenges](llms_caught_in_the_crossfire_malware_requests_and_jailbreak_challenges.md)**

:   构建 MalwareBench 基准（320 个手工恶意代码需求 × 11 种黑盒越狱方法 = 3520 个 prompt），系统评测 29 个 LLM 在恶意代码生成场景下的安全性，发现越狱攻击将平均拒绝率从 60.93% 降至 39.92%，且模型参数量与防御能力并非正比关系。

**[Lssf Safety Subspace](lssf_safety_subspace.md)**

:   LSSF 提出 LLM 的安全信息存在于低秩子空间中的假设，通过 SVD 提取安全对齐模型的主成分，利用安全奇异值熵自适应确定每层的保留秩，最终将提取的安全主成分线性融合到微调后的模型中，无需额外训练即可恢复因微调而退化的安全对齐，同时保持下游任务性能。

**[M2S Multiturn To Singleturn Jailbreak In](m2s_multiturn_to_singleturn_jailbreak_in.md)**

:   提出 M2S 框架，通过三种简单的格式转换方法（Hyphenize/Numberize/Pythonize）将多轮人类越狱对话压缩为单轮 prompt，不仅保持甚至超越原始多轮攻击效果（ASR 高达 95.9%，比多轮提升最多 17.5%），同时 token 使用量减半以上。

**[Measuring Data Diversity For Instruction Tuning A Systematic Analysis And A Reli](measuring_data_diversity_for_instruction_tuning_a_systematic_analysis_and_a_reli.md)**

:   系统分析 11 种现有多样性度量方法的局限性，提出 NovelSum——一种同时考虑样本间差异和信息密度的数据多样性指标，与指令微调性能达到 0.97 相关性。

**[Mpo Multilingual Safety Alignment](mpo_multilingual_safety_alignment.md)**

:   MPO 发现 LLM 在主导语言（英文）和目标语言间的隐式 Reward Gap 与安全性能强相关，提出直接最小化两者 Reward Gap 差异来将主导语言的安全对齐能力迁移到多语言，在三个模型上显著降低了低资源语言的攻击成功率且不损害通用能力。

**[Mtsa Multi-Turn Safety Alignment For Llms Through Multi-Round Red-Teaming](mtsa_multi-turn_safety_alignment_for_llms_through_multi-round_red-teaming.md)**

:   提出MTSA框架，通过思维引导的多轮红队攻击学习和基于未来奖励的多轮强化学习算法，在对抗迭代优化中同时提升红队模型的攻击能力和目标模型的安全防御能力，在多个安全基准上达到SOTA，且不损失模型通用性能。

**[Mutual Taught Policy Reward](mutual_taught_policy_reward.md)**

:   Mutual-Taught 提出了一种基于 EM 算法的自训练框架，在偏好优化过程中同时迭代更新 policy model 和 reward model：E-step 用当前 RM 优化 PM，M-step 用 PM 更新前后的输出差异构建伪偏好对来更新 RM，解决了分布偏移导致的 reward hacking 问题，8B 模型在 AlpacaEval-2 达到 54.1% LC win rate。

**[Otpo Token Weighting](otpo_token_weighting.md)**

:   OTPO 利用无平衡最优传输（UOT）在 chosen/rejected 回复的 token 表示之间计算语义对齐权重，使偏好优化聚焦于关键差异 token 而非均等对待所有 token，在 AlpacaEval2 上将 DPO 的 LC WR 从 48.14% 提升至 55.84%，并将 DPO/SimPO/SamPO/LDDPO 统一为 token 加权的特例。

**[Personalized Preference Opt](personalized_preference_opt.md)**

:   提出"溯因推理"视角的偏好个性化方法：先用 LLM 推断偏好 chosen/rejected 回答背后的用户画像（Persona Inference），再用画像增强的偏好数据训练模型（Persona Tailoring），显著提升模型对不同用户需求的个性化适配能力。

**[Pig Privacy Jailbreak](pig_privacy_jailbreak.md)**

:   提出 PIG 框架，通过识别隐私查询中的 PII 实体类型、构建隐私上下文示例、并利用三种基于梯度的迭代优化策略更新上下文，实现对 LLM 的高效隐私越狱攻击，在白盒和黑盒模型上均达到 SOTA。

**[Pku-Saferlhf Towards Multi-Level Safety Alignment For Llms With Human Preference](pku-saferlhf_towards_multi-level_safety_alignment_for_llms_with_human_preference.md)**

:   发布 PKU-SafeRLHF 大规模安全偏好数据集，包含 44.6k 精炼 prompt、265k 带安全元标签的 QA 对和 166.8k 偏好数据，首次引入 19 种危害类别和 3 级严重程度标注，并训练了严重程度敏感的审核模型（93% 准确率）和基于该数据的 SafeRLHF 对齐 pipeline。

**[Probability-Consistent Preference Optimization For Enhanced Llm Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)**

:   提出 PCPO（Probability-Consistent Preference Optimization），在偏好对选择时同时考虑答案正确性和 token 级概率一致性（用 Levenshtein 距离过滤+概率一致性评分），并在 DPO 损失中按一致性加权，在 GSM8K/MATH-500/Olympiadbench 上一致超越标准 DPO 和 ScPO。

**[Queryattack Jailbreaking Aligned Large Language Models Using Structured Non-Natu](queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)**

:   提出 QueryAttack，将恶意自然语言查询分解为三个语义组件（内容、修饰符、类别）并填入编程语言模板（SQL/URL/Python/Java/C++ 等 9 种），结合 ICL 引导目标 LLM 直接用自然语言回复有害内容，无需解密步骤，在 GPT-4o 上 Ensemble 配置达到 96.35% ASR，且提出的跨语言 CoT 防御可将 ASR 降低最多 64%。

**[Red Queen Safeguarding Large Language Models Against Concealed Multi-Turn Jailbr](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)**

:   提出 Red Queen Attack——首个基于 Theory of Mind（ToM）构建多轮对话场景并隐藏恶意意图的越狱攻击方法，生成 56K 多轮隐蔽攻击数据，在 GPT-4o 上达到 87.6% ASR；同时提出 Red Queen Guard 防御策略，通过多轮 DPO 数据训练将 ASR 降至 <1%，同时不影响通用基准性能。

**[Rethinking Table Instruction Tuning](rethinking_table_instruction_tuning.md)**

:   系统消融表格指令微调中被忽视的超参数选择（学习率、数据量、epoch），揭示现有表格 LLM 因学习率过大（2e-5）导致通用能力严重退化（MMLU 降 14 分、AI2ARC 降 21 分），提出仅需 13 个数据集各 200 条（共 2600 条）+ 学习率 1e-6 + 2 epoch 微调 LLaMA 3.1 8B Instruct 即可构建 TAMA，在 13 个表格任务上匹配/超越 GPT-3.5 和 GPT-4，同时完整保持通用能力。

**[Reverse Preference Optimization For Complex Instruction Following](reverse_preference_optimization_for_complex_instruction_following.md)**

:   提出反向偏好优化（RPO），通过动态反转指令中未满足的约束将任意回复转化为"完美"chosen 样本，消除多约束偏好对中的噪声，在多轮复杂指令遵循任务上显著超越 DPO 基线。

**[Reward Fairness Rlhf](reward_fairness_rlhf.md)**

:   将 RLHF 中的各种奖励偏差（长度偏差、类别偏差、社会偏差）统一定义为"奖励不公平"问题，从资源分配视角提出两种偏差无关的缓解方法——公平正则化和公平系数——在不针对特定偏差设计的情况下有效缓解多种偏差，实现更公平的人类偏好对齐。

**[Reward Generalization In Rlhf A Topological Perspective](reward_generalization_in_rlhf_a_topological_perspective.md)**

:   从信息拓扑的角度系统刻画 RLHF 中 reward 信息的流动——宏观层面将 RLHF 建模为自编码过程，微观层面提出 Induced Bayesian Network (IBN) 分析偏好数据拓扑对 reward 泛化的影响，进而提出树结构偏好数据方法，在 HH-RLHF/GSM-8K/DialogSum 三个任务上平均 65% win rate 超越链式 baseline。

**[Rewrite To Jailbreak Discover Learnable And Transferable Implicit Harmfulness In](rewrite_to_jailbreak_discover_learnable_and_transferable_implicit_harmfulness_in.md)**

:   提出 R2J（Rewrite to Jailbreak），一种可学习、可迁移的黑盒越狱方法——通过迭代训练 attacker LLM 学习改写有害指令（仅改措辞不改意图），相比 GCG/AutoDAN 等方法攻击成功率提高 20%+，且无额外前缀/后缀，更隐蔽且跨模型可迁移。

**[Rise Error Inject Preference](rise_error_inject_preference.md)**

:   RISE 发现 LLM 约 75% 的数学错误是微妙的步内错误（数字替换、操作数交换、步骤遗漏），通过让 LLM 自编辑向正确解注入预定义微妙错误来构造高质量难负样本，配合错误感知 DPO 训练，仅用 4.5K 样本在 GSM8K 提升 3.0%、MATH 提升 7.9%，并泛化到逻辑推理和代码生成。

**[Rpo Retrieval Preference Optimization For Robust Retrieval-Augmented Generation](rpo_retrieval_preference_optimization_for_robust_retrieval-augmented_generation.md)**

:   提出 Retrieval Preference Optimization (RPO)，一种专为 RAG 设计的轻量级偏好对齐方法，通过将检索质量评估隐式地集成到生成过程中，使 LLM 能够自适应地在参数知识和检索知识之间做出选择，无需额外组件即可缓解知识冲突导致的幻觉问题。

**[Sea Lowresource Safety Alignment For Multimodal](sea_lowresource_safety_alignment_for_multimodal.md)**

:   提出 SEA 框架，通过梯度优化生成合成模态 embedding（不需要真实图像/视频/音频），仅用文本安全数据就能实现多模态 LLM 的安全对齐，在单张 RTX3090 上 24 秒即可合成高质量 embedding，同时发布了视频和音频安全基准 VA-SafetyBench。

**[Synthesizeme Persona Prompts](synthesizeme_persona_prompts.md)**

:   提出SynthesizeMe，通过bootstrap推理→合成用户画像→筛选信息性示例三步流程，无需微调即可为个性化奖励模型构建有效prompt，在Chatbot Arena上提升LLM-as-a-Judge个性化准确率4.4%。

**[Tabledreamer Progressive And Weakness-Guided Data Synthesis From Scratch For Tab](tabledreamer_progressive_and_weakness-guided_data_synthesis_from_scratch_for_tab.md)**

:   提出 TableDreamer 两阶段数据合成框架：第一阶段从零合成多样化表格及种子指令数据，第二阶段通过弱点引导的迭代输入空间探索（在三个方向上演化数据，并用 LLM-as-Judge 筛选模型表现差的样本作为下一轮种子），仅用 27K GPT-4o 合成数据即将 Llama3.1-8B 的平均准确率提升 11.62%，超越使用 80K-100K 数据的所有基线方法。

**[Teaching An Old Llm Secure Coding](teaching_an_old_llm_secure_coding.md)**

:   提出 DiSCo（从前沿 LLM 蒸馏的安全代码偏好数据集，10K 实例覆盖 431 种 CWE）和 LPO（局部偏好优化算法，仅在安全相关 token 上传播损失），在四个安全编码基准上减少 19-40% 的安全问题，同时提升 3-10% 的代码质量。

**[Think Cite Attributed Text Gen](think_cite_attributed_text_gen.md)**

:   将归因文本生成（带引用的文本生成）建模为多步推理问题，提出自引导蒙特卡洛树搜索（SG-MCTS）结合进度奖励建模（PRM），通过多路径搜索+中间状态反思+生成/归因双维度进度奖励，在 ALCE 基准三个数据集上显著超越所有基线。

**[Tmcht Contagious Jailbreak Multiagent](tmcht_contagious_jailbreak_multiagent.md)**

:   提出 TMCHT 多智能体多拓扑攻击评估框架和 ARCJ（对抗性复制传染越狱）方法，解决了多智能体系统中单智能体攻击方法面临的"毒性消散"问题——通过优化检索后缀确保中毒样本易被检索，优化复制后缀使中毒信息具有传染性自我复制能力，在 line/star 拓扑和 100 智能体系统中分别提升 23.51%/18.95%/52.93% 的攻击成功率。

**[Upcycling Instruction Tuning From Dense To Mixture-Of-Experts Via Parameter Merg](upcycling_instruction_tuning_from_dense_to_mixture-of-experts_via_parameter_merg.md)**

:   本文提出UpIT (Upcycling Instruction Tuning)，利用密集模型指令微调过程中的中间checkpoint作为专业化专家，通过遗传算法扩展专家数量和路由预优化，实现数据高效且灵活的dense-to-MoE转换。
