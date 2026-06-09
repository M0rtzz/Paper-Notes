---
title: >-
  1363 篇 ACL2026 论文解读 · 每篇 5 分钟读懂
description: >-
  1363篇ACL2026论文解读，涵盖 LLM 安全(128篇)、多模态 VLM(99篇)、LLM 评测(91篇)、LLM 推理(80篇)、LLM Agent(78篇)、信息检索/RAG(74篇)、音频/语音(70篇)、可解释性(63篇)等 39个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ACL2026"
  - "AI顶会"
  - "论文解读"
  - "论文笔记"
  - "LLM 安全"
  - "多模态 VLM"
  - "LLM 评测"
  - "LLM 推理"
  - "LLM Agent"
  - "信息检索/RAG"
  - "音频/语音"
  - "可解释性"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 ACL2026 论文笔记

1363篇ACL2026论文解读，涵盖 LLM 安全(128篇)、多模态 VLM(99篇)、LLM 评测(91篇)、LLM 推理(80篇)、LLM Agent(78篇)、信息检索/RAG(74篇)、音频/语音(70篇)、可解释性(63篇)等 39个方向。每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。

<div class="conf-index" markdown>

---

## 🔒 LLM 安全 { #llm_safety }

**[STELA: A Linguistics-Aware LLM Watermarking via Syntactic Predictability](llm_safety/a_linguistics-aware_llm_watermarking_via_syntactic_predictability.md)**

:   STELA 用 POS n-gram 估计的「语言学不确定度」$\lambda(c_t)$ 作为水印强度调制信号，在语法约束高的位置弱化水印（保质量）、在语法自由位置增强水印（提检测力），与 KGW 一样仅靠 POS 分析器即可公开验证，无需访问模型 logits。

**[A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?](llm_safety/a_survey_on_the_safety_and_security_threats_of_computer-using_agents_jarvis_or_u.md)**

:   本文给"计算机使用智能体 (CUA)"的安全研究做了第一次系统化梳理，把 124 篇相关论文整理成"内在威胁 × 外在威胁 × 防御 × 评测"四维分类框架，并指出现有 CUA 的最大缺口是 UI grounding 鲁棒性与跨平台对抗评测。

**[Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL](llm_safety/abstain-r1_calibrated_abstention_and_post-refusal_clarification_via_verifiable_r.md)**

:   Abstain-R1 提出一种**澄清感知的 RLVR 奖励**，在不可回答查询上联合优化"明确拒答"和"拒答后给出有用澄清（指出缺失信息）"，使 3B 模型在拒答和澄清质量上接近甚至超越 DeepSeek-R1 等大模型。

**[ACIArena: Toward Unified Evaluation for Agent Cascading Injection](llm_safety/aciarena_toward_unified_evaluation_for_agent_cascading_injection.md)**

:   本文构造了首个针对"代理级联注入 (Agent Cascading Injection, ACI)"攻击的统一评测框架 ACIArena，覆盖 6 个主流多智能体系统 (MAS)、3 大攻击面 (Adversarial Input / Malicious Agent / Message Poison)、3 大攻击目标 (Hijacking / Disruption / Exfiltration) 的 1356 个测试用例，并提出极简但有效的防御 ACI-Sentinel，把 Hijacking 攻击成功率从 92.78% 压到 8.06%。

**[Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](llm_safety/adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)**

:   提出自适应文本匿名化框架，通过进化式提示优化自动为LLM发现任务特定的匿名化指令，在多个隐私-效用权衡场景中超越手工设计的策略，且可在开源模型上运行。

**[ADVICE: Answer-Dependent Verbalized Confidence Estimation](llm_safety/advice_answer-dependent_verbalized_confidence_estimation.md)**

:   本文通过 JSD 与归因分析诊断出 LLM 口头置信度过自信的根因是「置信度几乎不依赖于自己生成的答案」，并提出基于对比答案对的轻量微调框架 ADVICE，用 JSD/Margin/Sum 三项损失强迫置信度分布对正确答案显著高于错误答案，在保持任务精度的同时把 Gemma2-9b 在 TriviaQA 的 ECE 从 21.9% 压到 6.2%。

**[AgentCoMa: A Compositional Benchmark Mixing Commonsense and Mathematical Reasoning in Real-World Scenarios](llm_safety/agentcoma_a_compositional_benchmark_mixing_commonsense_and_mathematical_reasonin.md)**

:   AgentCoMa 构造了一个把常识选择和单步数学运算强制组合起来的 agentic benchmark，并在 61 个 LLM 上发现：模型通常能分别做对两个子问题，但组合后平均准确率从“两个子步都能独立答对”的 80% 掉到 51%，暴露出混合类型组合推理中的显著脆弱性。

**[AgentMark: Utility-Preserving Behavioral Watermarking for Agents](llm_safety/agentmark_utility-preserving_behavioral_watermarking_for_agents.md)**

:   AgentMark 把 LLM agent 的「下一步选什么 tool / subgoal」建模为一个时间变化的离散信道，通过显式 elicit 行为分布 $P_t$ 并应用 FDPSS 式分布保持采样把多比特 ID 嵌入 planning 决策，配合 RLNC 编码使得即便 trace 被裁剪/删步也能从残余日志恢复水印；在 ALFWorld、ToolBench、OASIS 三类任务上既不掉准确率（保持任务 SR 与 baseline 差异 <0.7 pp），又能稳定提供 1.2-2.3 bps 的多比特容量，且与 SynthID-Text 的内容层水印正交可叠加。

**[AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation](llm_safety/agsc_adaptive_granularity_and_semantic_clustering_for_uncertainty_quantification.md)**

:   AGSC 提出了一个针对长文本生成的不确定性量化框架，通过 NLI 中立概率触发自适应粒度分解（减少 60% 推理时间），并使用 GMM 软聚类捕捉潜在语义主题进行主题感知的加权聚合，在 BIO 和 LongFact 基准上达到 SOTA 的事实性相关性。

**[Aligning with Your Own Voice: Self-Corrected Preference Learning for Hallucination Mitigation in LVLMs](llm_safety/aligning_with_your_own_voice_self-corrected_preference_learning_for_hallucinatio.md)**

:   提出 AVES-DPO 框架：用一致性多模型验证（YOLO/GroundingDINO/Qwen3-VL）在 object/attribute/relation 三层细粒度检测 LVLM 自己生成回复中的幻觉，再让同一个 LVLM 自我校正并丰富细节，得到的偏好对天然处于目标模型的"自身分布"内；仅 5.2K 样本即可在多个幻觉 benchmark 上超过依赖 GPT-4V 教师的 SOTA（数据效率约 25×）。

**[APPSI-139: A Parallel Corpus of English Application Privacy Policy Summarization and Interpretation](llm_safety/appsi-139_a_parallel_corpus_of_english_application_privacy_policy_summarization_.md)**

:   APPSI-139 是首个由法律专家精细标注的英文应用隐私政策摘要与解释平行语料（139 篇政策 / 36,351 条标注 / 15,692 对改写），配套提出的 TCSI-pp-V2 框架用共享编码器 + 5 个交替训练的专家头实现"重要 / 风险 / 敏感 / 主题 / 改写"五子任务，相比 TCSI-pp v1 编码时间砍 73%、显存从 7.3GB 降到 2.7GB，可读性主观投票胜过 GPT-4o / Llama3-70b。

**[ASTRA: An Automated Framework for Strategy Discovery, Retrieval, and Evolution for Jailbreaking LLMs](llm_safety/astra_an_automated_framework_for_strategy_discovery_retrieval_and_evolution_for_.md)**

:   ASTRA 把每次越狱尝试都视为学习机会，按 1-10 连续打分把策略蒸馏到「Effective / Promising / Ineffective」三层向量库中，下一次攻击通过相似度检索复用经验，平均仅 2.4 次查询就在 8 个主流 LLM 上取得 80.6% 攻击成功率。

**[ATAAT: Adaptive Threat-Aware Adversarial Tuning Framework against Backdoor Attacks on Vision-Language-Action Models](llm_safety/ataat_adaptive_threat-aware_adversarial_tuning_framework_against_backdoor_attack.md)**

:   ATAAT 首次系统揭示 VLA 后门难以注入的根因是「梯度干涉」（良性与后门梯度方向反向相消，相似度长期负相关 -0.4），并通过隐式正交扰动（数据投毒）和休眠神经元锚定（白盒微调）两条互补路径把目标攻击成功率推到 80%+，同时保持良性 SR 接近正常。

**[AutoRAN: Automated Hijacking of Safety Reasoning in Large Reasoning Models](llm_safety/autoran_automated_hijacking_of_safety_reasoning_in_large_reasoning_models.md)**

:   本文提出首个自动化攻击 LRM 内部安全推理的框架 AutoRAN：用一个**弱但少对齐**的小模型先模拟目标 LRM 的「执行推理」生成叙事性 prompt，再根据目标拒绝时泄露的 CoT 反馈迭代精炼，在 gpt-o3 / o4-mini / Gemini-2.5-Flash 上对 AdvBench、HarmBench、StrongReject 达到**接近 100% 的攻击成功率**，且常常只需 1 轮。

**[Before Forgetting, Learn to Remember: Revisiting Foundational Learning Failures in LVLM Unlearning Benchmarks](llm_safety/before_forgetting_learn_to_remember_revisiting_foundational_learning_failures_in.md)**

:   作者指出现有 LVLM unlearning benchmark（FIUBench / MLLMU-Bench / CLEAR）在 stage 1 fine-tune 阶段就没把虚构身份真的记住，导致 stage 2 的「遗忘」评估全部失效；他们诊断出根因是「数据重复不够 + multi-hop curse」，由此提出 ReMem——每个身份 100 QA × 100 张多视角图、70%∶30% 的单跳/多跳混合、新增 Exposure 内部概率度量，把 unlearning 评测重新建立在「真的学会了」的基础上。

**[Beyond End-to-End: Dynamic Chain Optimization for Private LLM Adaptation on the Edge](llm_safety/beyond_end-to-end_dynamic_chain_optimization_for_private_llm_adaptation_on_the_e.md)**

:   提出 ChainFed，一种打破内存墙的链式联邦微调范式，通过逐层顺序训练-冻结适配器使资源受限边缘设备也能参与 LLM 微调，结合动态层协调、全局感知优化和功能导向自适应三项技术，平均准确率提升最高 46.46%。

**[Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation](llm_safety/beyond_explicit_refusals_soft-failure_attacks_on_retrieval-augmented_generation.md)**

:   形式化定义 RAG 系统的"软失败"威胁（生成流畅但无信息量的回答），提出 DEJA 黑箱进化攻击框架，通过对抗性文档诱导模型利用安全对齐机制产生模棱两可的回答，SASR 超过 79% 且高度隐蔽。

**[Calibration vs Decision Making: Revisiting the Reliability Paradox in Unlearned Language Models](llm_safety/calibration_vs_decision_making_revisiting_the_reliability_paradox_in_unlearned_l.md)**

:   这篇论文说明机器遗忘后的 LLM 即使保持很低的校准误差，也可能更多依赖数据集捷径 token 做选择，因此只用 ECE/MCE/Brier 判断 unlearned model 是否可靠是不够的。

**[Can Persona-Prompted LLMs Emulate Subgroup Values? An Empirical Analysis of Generalisability and Fairness in Cultural Alignment](llm_safety/can_persona-prompted_llms_emulate_subgroup_values_an_empirical_analysis_of_gener.md)**

:   本文用新加坡的 World Values Survey 子集为案例，构造 20,877 个 (问题, 子群体) 样本，验证 LLM 是否能模拟细粒度人口子群的价值偏好——结果 GPT-4.1 zero-shot 仅 57.4% 准确率，简单 SFT 在 OOD 子群上平均涨 17.4%，但 NMAE 视角下子群差距反而扩大，模型对年轻/男性/华人/基督徒持续偏好。

**[CAP: Controllable Alignment Prompting for Unlearning in LLMs](llm_safety/cap_controllable_alignment_prompting_for_unlearning_in_llms.md)**

:   提出 CAP 框架，通过训练轻量 SLM 生成可控的提示前缀来引导冻结的 LLM 选择性遗忘目标知识，无需修改模型参数，实现了可逆、可迁移的 LLM 知识遗忘。

**[CarO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation](llm_safety/caro_chain-of-analogy_reasoning_optimization_for_robust_content_moderation.md)**

:   提出 CarO（Chain-of-Analogy Reasoning Optimization），一个两阶段训练框架，通过 RAG 引导生成类比推理链 + SFT + 定制 DPO 优化，使 LLM 在推理时自主生成类比参考案例进行内容审核，在模糊审核基准上 F1 平均提升 24.9%，显著超越推理模型（DeepSeek R1）和专用审核模型（LLaMA Guard）。

**[CausalDetox: Causal Head Selection and Intervention for Language Model Detoxification](llm_safety/causaldetox_causal_head_selection_and_intervention_for_language_model_detoxifica.md)**

:   CausalDetox 使用"必要性和充分性概率"（PNS）作为因果准则来精确定位产生有毒内容的注意力头，并通过局部推理时干预和 PNS 引导的微调两种互补策略进行去毒化，在多个模型上实现最高 5.34% 的毒性降低，同时保持语言流畅性。

**[CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents](llm_safety/ci-work_benchmarking_contextual_integrity_in_enterprise_llm_agents.md)**

:   基于上下文完整性（Contextual Integrity）理论构建企业场景基准 CI-Work，揭示前沿 LLM 智能体在企业工作流中普遍存在隐私泄漏问题，且模型规模扩大反而加剧泄漏。

**[CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](llm_safety/cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)**

:   针对大型推理模型（LRM）的遗忘难题——需要同时从思维链（CoT）和最终答案中移除敏感知识——提出 CiPO 框架，通过让模型生成逻辑有效的反事实推理轨迹并用迭代偏好优化引导模型偏好反事实路径，实现有效遗忘同时保持推理能力。

**[Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors](llm_safety/compiling_activation_steering_into_weights_via_null-space_constraints_for_stealt.md)**

:   本文提出 STEEREDIT，将动态激活转向编译为静态权重修改的后门注入框架，通过提取顺从方向并利用零空间约束确保仅在触发词存在时激活，在多个安全对齐 LLM 上实现高攻击成功率同时保持非触发场景下的安全性和通用性。

**[Context-Fidelity Boosting: Enhancing Faithful Generation through Watermark-Inspired Decoding](llm_safety/context-fidelity_boosting_enhancing_faithful_generation_through_watermark-inspir.md)**

:   CFB 把文本 watermark 用的 logit 加性偏置技术反向利用——在解码每步给"被输入上下文支持"的 token 加 bonus，提出 static / context-aware（用 JSD 自适应缩放）/ token-aware（用注意力 + 语义相关性再分配）三层渐进策略，在多模型多任务的摘要和 QA 上稳定提升 faithfulness 指标，且几乎无解码开销。

**[CRISP: Persistent Concept Unlearning via Sparse Autoencoders](llm_safety/crisp_persistent_concept_unlearning_via_sparse_autoencoders.md)**

:   针对 SAE-based unlearning 大多只在推理时干预、参数仍含敏感知识的问题，CRISP 通过对比 target/retain 语料自动挑出"只在 target 上强激活"的 SAE 特征，再用 LoRA + 三段式损失（unlearn + retain + coherence）把这些特征的激活值"焊死"为零，从而在 WMDP-Bio/Cyber 上同时刷新 unlearn-retain-fluency 三轴 Pareto 前沿，比 ELM 高 27-34 分，比 RMU 高 5-8 分。

**[CrossGuard: Safeguarding MLLMs against Joint-Modal Implicit Malicious Attacks](llm_safety/crossguard_safeguarding_mllms_against_joint-modal_implicit_malicious_attacks.md)**

:   针对"图像和文本单独都安全、合起来才有害"的隐式跨模态攻击，提出基于 RL 红队的 ImpForge 自动批量生成此类样本（三种 reward：safety / semantic / overlap），再用这些数据 LoRA SFT 出 CrossGuard 守卫模型——把 SIUO 隐式攻击 ASR 从 GPT-4o 的 48.9% 降到 5.4%，同时在 5 个安全 benchmark 上平均 ASR 仅 2.79%（runner-up Claude-3.5 是 12.05%）。

**[CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge](llm_safety/curate_continual_unlearning_in_real_time_with_ensured_preservation_of_llm_knowle.md)**

:   CURaTE 提出一种基于句子嵌入匹配的行为遗忘框架：预部署时训练一个通用的遗忘嵌入器（不使用任何遗忘集），部署后实时将新遗忘请求嵌入存入数据库，推理时通过余弦相似度决定是回答还是拒绝，完全不修改 LLM 权重从而实现近乎完美的知识保留。

**[DART: Mitigating Harm Drift in Difference-Aware LLMs via Distill-Audit-Repair Training](llm_safety/dart_mitigating_harm_drift_in_difference-aware_llms_via_distill-audit-repair_tra.md)**

:   DART 发现并解决了"危害漂移"问题——微调 LLM 提高差异感知分类准确率（如识别合法的人口统计差异）的同时，模型生成的解释变得更有害。通过蒸馏-审计-修复三阶段管线，DART 将 Llama-3-8B 准确率从 39.0% 提升到 68.8%，同时减少 72.6% 的危害漂移案例。

**[De-Anonymization at Scale via Tournament-Style Attribution](llm_safety/de-anonymization_at_scale_via_tournament-style_attribution.md)**

:   本文提出 DAS（De-Anonymization at Scale），一种基于 LLM 的大规模作者去匿名化方法，采用锦标赛式淘汰策略+密集检索预过滤+多轮投票聚合，可在数万候选文本中进行作者匹配，揭示了 LLM 对匿名平台（如双盲评审）的隐私威胁。

**[Decomposed Trust: Privacy, Adversarial Robustness, Ethics, and Fairness in Low-Rank LLMs](llm_safety/decomposed_trust_privacy_adversarial_robustness_ethics_and_fairness_in_low-rank_.md)**

:   首篇系统评估**低秩分解 (SVD/FWSVD/BASEL)** 对 LLM 可信性影响的研究，发现"训练数据隐私 ↑、对抗鲁棒 ↑、PII 防护 ↓、伦理对齐 ↓、公平性 ↓"的非对称 trade-off，并用梯度归因把对抗脆弱性定位到 `embed_tokens` 与 `down_proj` 两个子层。

**[Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game](llm_safety/detecting_rag_extraction_attack_via_dual-path_runtime_integrity_game.md)**

:   提出 CanaryRAG，一个受软件安全中栈金丝雀启发的 RAG 运行时防御机制，通过在检索块中注入非语义金丝雀 token 并设计双路径完整性博弈（目标路径不应泄露金丝雀 + Oracle 路径应能引出金丝雀），实时检测知识库提取攻击，在不影响任务性能和推理延迟的前提下实现最强防护。

**[Detoxification for LLM from Dataset Itself](llm_safety/detoxification_for_llm_from_dataset_itself.md)**

:   本文提出 HSPD（层次化语义保留去毒）流水线，通过 SoCD（软对比解码）引导 LLM 定位并重写原始语料中的有毒片段，同时保留语义，生成可直接替换原始数据用于微调的去毒语料——在 GPT2-XL 上将毒性概率从 0.42 降至 0.18，在 LLaMA2-7B、OPT-6.7B 和 Falcon-7B 上也取得了最优去毒效果。

**[Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](llm_safety/differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)**

:   DP-SynRAG 用 LLM 把私有 RAG 数据库**一次性**蒸馏成差分隐私合成文本库，之后任意次查询都不再消耗 privacy budget，在 Medical Synth / MovieLens / SearchQA 三个数据集上 accuracy 远超 query-time DP-RAG（多查询场景下 DP-RAG 退化到崩盘）。

**[Do Multimodal RAG Systems Leak Data? A Comprehensive Evaluation of Membership Inference and Image Caption Retrieval Attacks](llm_safety/do_multimodal_rag_systems_leak_data_a_comprehensive_evaluation_of_membership_inf.md)**

:   作者首次系统评估了 **图像驱动的多模态 RAG (mRAG)** 系统的隐私泄露风险，证明仅靠最朴素的黑盒文本 prompt + 一张目标图就能在 4 个数据集 × 3 个 VLM 上达到 **MIA F1=0.993** 与 **caption exact-match=0.835**，即便对图像做裁剪/掩盖/旋转/噪声等变换攻击依然有效，并发现 prompt 中"目标图 vs 检索图"的相对位置和 cross-modal rerank 是两个关键缓解杠杆。

**[DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack](llm_safety/dualguard_dual-stream_large_language_model_watermarking_defense_against_paraphra.md)**

:   DualGuard 首次提出**双流水印**机制：用两个互补的标准 / 对抗水印头根据当前内容是"良性"还是"恶意"自适应地注入不同水印，使良性文本两路一致、恶意文本两路发散，从而在保持对 paraphrase 鲁棒的同时**首次能检测并溯源** piggyback spoofing 注入的恶意片段。

**[Enhancing Hallucination Detection via Future Context](llm_safety/enhancing_hallucination_detection_via_future_context.md)**

:   本文提出利用采样生成的"未来上下文"（后续句子）来增强黑盒场景下的幻觉检测，利用幻觉一旦出现就倾向于持续传播的"滚雪球效应"，在 SelfCheckGPT 和 SC 等多种采样方法上一致提升检测性能。

**[Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks](llm_safety/evaluating_answer_leakage_robustness_of_llm_tutors_against_adversarial_student_a.md)**

:   本文系统评估 LLM 家教在「学生想骗答案」场景下的答案泄露鲁棒性：定义 6 类对抗/说服技巧，比较 4 类对抗学生代理（基础、推理增强、多代理、SFT 微调），并验证两种简单防御（推理优先、多代理 tutor）可在多数模型上把泄露率从 70–85% 压到 < 10%。

**[Exploring Cross-Client Memorization of Training Data in Large Language Models for Federated Learning](llm_safety/exploring_cross-client_memorization_of_training_data_in_large_language_models_fo.md)**

:   作者把集中式 LLM 上的 fine-grained 跨样本记忆度量（Zeng 2024 + PAN2014 抄袭检测器）扩展到联邦学习场景，提出一对 client-pair 度量 $\text{MR}_{j \to k}$ 并由此推导 intra-client / inter-client 记忆比率，发现 FL **并不能**有效防止训练数据记忆——intra-client 记忆比 inter-client 高、但 FL vs CL 总记忆比并无明显下降，且记忆量受 prefix 长度、解码策略、FL 算法（FedProx > FedAvg）显著影响。

**[FAITH: Factuality Alignment through Integrating Trustworthiness and Honestness](llm_safety/faith_factuality_alignment_through_integrating_trustworthiness_and_honestness.md)**

:   本文提出FAITH框架，通过将LLM的不确定性信号（一致性+语义熵）映射到自然语言描述的知识状态象限（可信度×诚实度），设计考虑不确定性的细粒度奖励函数进行PPO训练，再用RAG模块纠正潜在错误，系统性提升LLM的事实准确性。

**[FaithLens: Detecting and Explaining Faithfulness Hallucination](llm_safety/faithlens_detecting_and_explaining_faithfulness_hallucination.md)**

:   本文提出 FaithLens，一个 8B 参数的忠实性幻觉检测模型，通过高质量数据合成+三维过滤（标签正确性、解释质量、数据多样性）进行冷启动 SFT，再用基于规则的强化学习（预测正确性奖励+解释质量奖励）进一步优化，在 12 个任务上超越 GPT-5.2 和 o3，同时提供高质量的解释性输出。

**[Fast-MIA: Efficient and Scalable Membership Inference for LLMs](llm_safety/fast-mia_efficient_and_scalable_membership_inference_for_llms.md)**

:   Fast-MIA 把 9 种主流 LLM 成员推理攻击（MIA）方法塞进同一个 vLLM 批量推理引擎，并加一层跨方法 log-prob 缓存，使评测在 LLaMA-30B / WikiMIA 上整体加速约 5×（SaMIA 单方法加速 19.5×）、AUC 几乎不变，让大规模 MIA 审计第一次变得真的"跑得起"。

**[FinGround: Detecting and Grounding Financial Hallucinations via Atomic Claim Verification](llm_safety/finground_detecting_and_grounding_financial_hallucinations_via_atomic_claim_veri.md)**

:   FinGround 是一个面向金融文档问答的三阶段 "verify-then-ground" pipeline：(1) finance-aware 混合检索；(2) 把答案拆成原子 claim 并按"数值/时间/实体属性/比较/监管/计算"六类 taxonomy 用 type-routed 策略验证 (其中 computational claim 用公式重构 + 算术再校验)；(3) 对未支持的 claim 进行 grounded 重写并加段/单元格级引用——把 GPT-4o 蒸馏到 8B 检测器实现 91.4% F1、18× 加速，端到端将 hallucination rate 相比 GPT-4o+CoT 降 78%。

**[FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation](llm_safety/flexguard_continuous_risk_scoring_for_strictness-adaptive_llm_content_moderation.md)**

:   FlexGuard 提出了一种输出连续风险评分（0-100）而非二元安全/不安全判断的 LLM 审核模型，通过基于评分准则的蒸馏和 GRPO 风险对齐训练，在不同严格度部署场景下实现了 SOTA 的鲁棒性和准确率。

**[Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens](llm_safety/forget_what_matters_keep_the_rest_selective_unlearning_of_informative_tokens.md)**

:   提出 Entropy-guided Token Weighting (ETW)，利用预测分布的熵值作为 token 信息量的代理指标，选择性地对信息性 token 施加更强的遗忘惩罚，在有效遗忘目标知识的同时更好地保持模型通用能力。

**[From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning](llm_safety/from_domains_to_instances_dual-granularity_data_synthesis_for_llm_unlearning.md)**

:   本文形式化定义了领域级和实例级两种 LLM 遗忘粒度，提出 BiForget 框架——利用目标模型自身（而非外部强模型）通过种子引导合成和对抗探测两阶段生成高质量遗忘数据集，在 Harry Potter 领域将相关性提升约 20、多样性提升约 0.05 同时数据量减半。

**[From Passive Metric to Active Signal: The Evolving Role of Uncertainty Quantification in Large Language Models](llm_safety/from_passive_metric_to_active_signal_the_evolving_role_of_uncertainty_quantifica.md)**

:   本文系统综述了 LLM 中不确定性量化从"被动诊断指标"到"主动控制信号"的功能演化，覆盖三大前沿领域：高级推理（引导计算分配和自我纠正）、自主代理（驱动工具使用和信息获取的元认知决策）、以及强化学习（缓解奖励黑客并通过内在奖励实现自我改进）。

**[GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models](llm_safety/gambit_a_gamified_jailbreak_framework_for_multimodal_large_language_models.md)**

:   本文提出 GAMBIT，一种游戏化多模态越狱框架，通过将有害查询分解为拼图图像+隐藏关键词，并嵌入竞争性游戏场景，利用模型的推理激励和认知负荷来绕过安全过滤器，在 Gemini 2.5 Flash 上达到 92.13%、GPT-4o 上达到 85.87% 的攻击成功率，对推理模型和非推理模型均有效。

**[Gap-K%: Measuring Top-1 Prediction Gap for Detecting Pretraining Data](llm_safety/gap-k_measuring_top-1_prediction_gap_for_detecting_pretraining_data.md)**

:   这篇论文提出 Gap-K%，用目标 token 与模型 top-1 预测之间的归一化 log probability gap 加上顺序滑窗平滑来检测文本是否出现在 LLM 预训练数据中，在 WikiMIA、MIMIR、近期模型和强 paraphrase 攻击下都超过 Min-K%++ 等基线。

**[ForgeryTalker: Generating Attribution Reports for Manipulated Facial Images](llm_safety/generating_attribution_reports_for_manipulated_facial_images_a_dataset_and_basel.md)**

:   本文提出伪造归因报告生成（Forgery Attribution Report Generation）这一新任务，构建了包含 152,217 个样本的 MMTT 数据集（首个同时提供像素级掩码和人工文本描述的大规模面部伪造数据集），并提出 ForgeryTalker 端到端基线，通过共享编码器和双解码器（掩码+语言模型）联合生成定位掩码和归因报告，达到 59.3 CIDEr 和 73.67 IoU。

**[Generating Effective CoT Traces for Mitigating Causal Hallucination](llm_safety/generating_effective_cot_traces_for_mitigating_causal_hallucination.md)**

:   本文首先提出了因果幻觉率（CHR）指标来量化小型 LLM 在事件因果识别中过度预测因果关系的倾向，然后通过系统实验确定了有效 CoT 数据的两个关键标准（充分长度的语义解释+与目标模型对齐的分布），设计了一套低成本的 CoT 数据生成管线，将 Qwen2.5-1.5B 的 CHR 从 83.54% 降至 6.26%，同时提升平均准确率至 66.00%。

**[Hallucination Detection in LLMs with Topological Divergence on Attention Graphs](llm_safety/hallucination_detection_in_llms_with_topological_divergence_on_attention_graphs.md)**

:   TOHA 把 LLM 的 attention 矩阵当成带权图，用拓扑数据分析里的 Manifold Topology Divergence 度量「response 子图相对 prompt 子图的拓扑新颖度」，并发现存在跨数据集稳定的「幻觉感知头」——只用 10 个这样的头做平均，就能在 RAG 场景下做到 training-free + 比 SelfCheckGPT 快 70× 且 ROC-AUC 显著领先。

**[Hard to Read, Easy to Jailbreak: How Visual Degradation Bypasses MLLM Safety Alignment](llm_safety/hard_to_read_easy_to_jailbreak_how_visual_degradation_bypasses_mllm_safety_align.md)**

:   本文首次揭示了 MLLM 在「视觉文本压缩」范式下的安全盲区——当渲染图像 DPI 落在 45–150 的 Attack Comfort Zone (ACZ) 时，模型 OCR 仍准但安全对齐崩塌（ASR 从 0% 飙至 70%+），原因是浅层算力被"认字"耗尽导致有害语义只在深层才出现，浅层 guardrail 被绕过；用 prompt 级的 Structured Cognitive Offloading（先转录→再审计→再回答）就能把 ASR 打回近基线水平。

**[How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study](llm_safety/how_should_we_enhance_the_safety_of_large_reasoning_models_an_empirical_study.md)**

:   本文系统研究如何通过 SFT 增强大型推理模型（LRM）的安全性，发现直接蒸馏安全响应效果有限的根因是五种风险推理模式（尤其是"弱犹豫"），提出针对性的蒸馏策略将 PAIR 攻击成功率从 63% 降至 13%，并发现短推理链和模板推理在安全性上与长推理链表现相当。

**[Illusions of Confidence? Diagnosing LLM Truthfulness via Neighborhood Consistency](llm_safety/illusions_of_confidence_diagnosing_llm_truthfulness_via_neighborhood_consistency.md)**

:   本文指出 LLM 的"高 self-consistency 不等于真实信念"——在 995 道全样一致答对的题上加一点点上下文干扰，准确率从 100% 直接掉到 33.8%。作者提出 **Neighbor-Consistency Belief (NCB)**：把目标事实和它的"概念邻居（前提/蕴含/主题）"做联合一致性估计，作为信念鲁棒性的结构化代理；并基于 Asch 从众实验与 Source Credibility 理论设计了 cognitive stress-test 协议，在 4 个 LLM 上证明高 NCB 数据明显更抗干扰；进一步提出 **Structure-Aware Training (SAT)**：用师生 KL 蒸馏强制学生模型在不同邻域上下文下输出一致，让新学知识的鲁棒性比 Ans/Know 增强基线再提升约 30%。

**[Instant Personalized Large Language Model Adaptation via Hypernetwork](llm_safety/instant_personalized_large_language_model_adaptation_via_hypernetwork.md)**

:   Profile-to-PEFT 用一个 hypernetwork 把用户画像直接映射成个性化 LoRA 参数，避免 OPPU 为每个用户重新训练 adapter，从而实现更快、更可扩展、可面向未见用户泛化的 LLM 个性化。

**[Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries](llm_safety/into_the_gray_zone_domain_contexts_can_blur_llm_safety_boundaries.md)**

:   本文发现领域特定上下文（如化学论文）会选择性放松 LLM 对相关有害知识的防护（纵向解锁），而安全研究上下文会触发跨所有有害类别的广泛防护放松（通用解锁），据此提出 Jargon 攻击框架，在包括 GPT-5.2、Claude-4.5 在内的七个前沿模型上实现超 93% 的攻击成功率。

**[Jailbreaking Large Language Models with Morality Attacks](llm_safety/jailbreaking_large_language_models_with_morality_attacks.md)**

:   本文构建10.3K道德攻击数据集（价值模糊+价值冲突），通过四种对抗策略操纵LLM道德判断，发现LLM和guardrail模型对道德攻击极度脆弱，且更大模型反而更容易被攻破。

**[Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning](llm_safety/know_thy_enemy_securing_llms_against_prompt_injection_via_diverse_data_synthesis.md)**

:   本文提出 InstruCoT，通过合成覆盖多种注入向量和威胁场景的多样化训练数据，并引入基于情境感知模型的三阶段指令级思维链微调，使 LLM 在面对各类提示注入攻击时能有效识别并拒绝恶意指令，在行为偏离、隐私泄露和有害输出三个维度上大幅超越现有防御方法。

**[Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation](llm_safety/knowledge_poisoning_attacks_on_medical_multi-modal_retrieval-augmented_generatio.md)**

:   作者提出 M3Att——首个面向医学多模态 RAG 的 **query-agnostic** 知识投毒框架，用"分布引导的视觉 PGD 触发"做检索劫持 + "临床歧义引导的文本改写"绕过 LVLM 自纠错，在 5 个 LVLM × 5 数据集 × 4 个医学任务上以 <1% 的投毒率（无需查询知识、视觉扰动 $\epsilon=16/255$）平均把下游效用拉低 8.78%，且对图聚类 / 文本聚类 / 图文一致性等 3 种 pre-retrieval 防御鲁棒。

**[LeakDojo: Decoding the Leakage Threats of RAG Systems](llm_safety/leakdojo_decoding_the_leakage_threats_of_rag_systems.md)**

:   提出 LeakDojo——首个把 RAG 系统、攻击与防御都模块化解耦的可配置评测框架，在 6 种攻击 × 14 个 LLM × 4 个数据集 × 多种增强模块上系统量化 RAG 泄露风险，发现"指令跟随能力越强、泄露风险越高"且"RAG 忠实度与泄露风险正相关"。

**[Learning Uncertainty from Sequential Internal Dispersion in Large Language Models](llm_safety/learning_uncertainty_from_sequential_internal_dispersion_in_large_language_model.md)**

:   提出 SIVR 框架，通过计算 LLM 隐藏状态跨层的内部方差（广义方差、圆方差、token 熵）作为 token 级特征，用轻量 Transformer 编码器聚合全序列模式来估计不确定性/检测幻觉，显著优于基线且泛化更强。

**[LLM-VA: Resolving the Jailbreak-Overrefusal Trade-off via Vector Alignment](llm_safety/llm-va_resolving_the_jailbreak-overrefusal_trade-off_via_vector_alignment.md)**

:   LLM-VA 发现 LLM 内部把"是否回答"（answer 向量 $v_a$）与"输入是否安全"（benign 向量 $v_b$）编码成几乎正交的两个方向，导致 jailbreak 与 over-refusal 之间的取舍永远此消彼长；它通过闭式最小范数权重更新把 $v_a$ 与 $v_b$ 对齐，让模型"愿不愿答"因果依赖于"输入安不安全"，在 12 个 LLM 上 F1 比最强 baseline 高 11.45%、效用仅掉 4.08%，且无需任何 fine-tuning 或架构改动。

**[Logical Consistency as a Bridge: Improving LLM Hallucination Detection via Label Constraint Modeling between Responses and Self-Judgments](llm_safety/logical_consistency_as_a_bridge_improving_llm_hallucination_detection_via_label_.md)**

:   把 LLM 的 self-judgment ("它觉得自己刚才答对了没") 也当成一个可能幻觉的 generation，先用 intrinsic feature 训一个 "meta-judgment detector" 估它的可信度，再用"如果 self-judgment 说真→两标签相同/说假→两标签相反"这条天然逻辑规则，把 response detector 和 meta-judgment detector 通过 Huber loss 互相约束、用置信度加权的互学习联合训练；推理时只用 response detector 但已经吸收了 self-judgment 的知识，零额外推理成本就拿到双视角增益。

**[Look Twice before You Leap: A Rational Framework for Localized Adversarial Anonymization](llm_safety/look_twice_before_you_leap_a_rational_framework_for_localized_adversarial_anonym.md)**

:   提出 RLAA 框架，通过 Attacker-Arbitrator-Anonymizer 三角色架构和边际替代率（MRS）理性约束，解决对抗式文本匿名化迁移到本地小模型时的效用崩溃问题，无需训练即可在本地实现优于 API 方案的隐私-效用平衡。

**[Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models](llm_safety/lost_in_diffusion_uncovering_hallucination_patterns_and_failure_modes_in_diffusi.md)**

:   首次系统性地对比扩散大语言模型（dLLM）与自回归（AR）对应模型的幻觉模式，揭示当前 dLLM 幻觉倾向更高，并识别出三种扩散特有的失败模式：过早终止、不完全去噪和上下文入侵。

**[Lying with Truths: Open-Channel Multi-Agent Collusion for Belief Manipulation via Generative Montage](llm_safety/lying_with_truths_open-channel_multi-agent_collusion_for_belief_manipulation_via.md)**

:   这篇论文提出认知合谋攻击这一安全问题：多个代理只公开发布真实但经过叙事编排的证据碎片，也能诱导 LLM 受害代理形成错误因果信念，并在下游验证层中继续传播。

**[Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation](llm_safety/making_mllms_blind_adversarial_smuggling_attacks_in_mllm_content_moderation.md)**

:   本文揭示了多模态大模型内容审核中的"对抗走私攻击"（ASA）威胁——将有害内容编码为人可读但 AI 不可读的视觉格式来规避自动检测，构建了包含 1,700 个样本、9 种攻击技术的 SmuggleBench 基准，发现包括 GPT-5 在内的 SOTA 模型攻击成功率超过 90%。

**[Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning](llm_safety/maximizing_local_entropy_where_it_matters_prefix-aware_localized_llm_unlearning.md)**

:   本文提出 PALU（Prefix-Aware Localized Unlearning），从时间和词表两个维度实现局部化的熵最大化遗忘：在时间维度仅对敏感前缀 token 施加遗忘目标，在词表维度仅对 top-K logits 进行平坦化，以最小的参数扰动实现高效遗忘并保持模型通用能力。

**[MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs](llm_safety/meashalu_mitigation_of_scientific_measurement_hallucinations_for_large_language_.md)**

:   本文提出MeasHalu框架，通过细粒度测量幻觉分类法和两阶段优化（推理感知SFT+幻觉靶向GRPO奖励）缓解LLM在科学测量抽取中的幻觉，在MeasEval上显著超越基线。

**[Membership Inference Attacks on In-Context Learning Recommendation](llm_safety/membership_inference_attacks_on_llm-based_recommender_systems.md)**

:   首次系统研究 LLM-based ICL 推荐系统的成员推断攻击（MIA），设计 Similarity / Memorization / Inquiry / Poisoning 四种攻击，发现基于 LLM 内在 **记忆** 的 Memorization 攻击在 MovieLens-1M 上 attack advantage ≥ 82%，且现有 prompt-based 防御（包括针对 Poisoning）几乎拦不住。

**[MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](llm_safety/memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)**

:   提出 MemoPhishAgent（MPA），首个专为钓鱼URL检测设计的记忆增强多模态LLM智能体，通过5个专用工具的动态编排和情景记忆系统复用历史推理轨迹，在公开基准上召回率提升13.6%，在真实社交媒体数据上提升20%，并已部署生产环境每周处理约6万高风险URL。

**[Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem](llm_safety/modeling_llm_unlearning_as_an_asymmetric_two-task_learning_problem.md)**

:   把 LLM unlearning 显式建模为「retain 为主、forget 为辅」的非对称两任务问题，提出 SAGO——对 retain/forget 梯度做**元素级符号对齐门控**，在 WMDP 与 RWKU 上把 retention 性能逼近原模型，同时几乎不损失 forget 效果。

**[Multi-component Causal Tracing in Large Language Models](llm_safety/multi-component_causal_tracing_in_large_language_models.md)**

:   这篇论文把 causal tracing 从单组件分析扩展到多组件子集搜索，并提出 PGB-CT 用软干预、指标变换和稀疏二值惩罚高效找到共同影响 LLM 行为的 attention heads 与 MLP neurons。

**[MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models](llm_safety/muse_a_run-centric_platform_for_multimodal_unified_safety_evaluation_of_large_la.md)**

:   MUSE 把跨模态 payload 生成、多轮红队攻击、统一模型路由和五级安全裁判整合成一个以 run 为中心的可复现实验平台，并用约 3700 次实验说明：多轮策略能击穿单轮几乎全拒答的多模态 LLM，而跨轮模态切换更像是加速防线松动的机制，不是普适提高最终 ASR 的银弹。

**[On Safety Risks in Experience-Driven Self-Evolving Agents](llm_safety/on_safety_risks_in_experience-driven_self-evolving_agents.md)**

:   本文系统研究经验驱动自进化Agent的安全风险，发现仅从无害任务积累的经验也导致安全性显著退化（ASR上升13-49%），根因是经验的执行导向本质强化了行动而非拒绝。

**[PARASITE: Conditional System Prompt Poisoning to Hijack LLMs](llm_safety/parasite_conditional_system_prompt_poisoning_to_hijack_llms.md)**

:   PARASITE 将“从公开市场下载的系统提示词可能被植入条件触发后门”形式化为新的供应链威胁，并用全局语义搜索加词级贪心扰动在黑盒条件下生成高隐蔽、只在目标问题上劫持回答的系统提示。

**[Permutation-Consensus Listwise Judging for Robust Factuality Evaluation](llm_safety/permutation-consensus_listwise_judging_for_robust_factuality_evaluation.md)**

:   PCFJudge 将候选答案顺序视为 listwise 事实性评估中的干扰变量，通过对同一候选集运行 7 种排列并聚合分数、排序、top-set 投票和校准不确定性，在 RewardBench 2 Factuality 上相对单次直接评审提升最高 7 个百分点。

**[PIArena: A Platform for Prompt Injection Evaluation](llm_safety/piarena_a_platform_for_prompt_injection_evaluation.md)**

:   本文提出 PIArena，一个统一且可扩展的提示注入（Prompt Injection）评估平台，集成了多种 SOTA 攻击和防御方法，支持即插即用评估，并设计了基于策略的自适应攻击方法，系统性地揭示了现有防御在泛化性、自适应攻击和任务对齐场景下的关键局限。

**[Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding](llm_safety/please_refuse_to_answer_me_mitigating_over-refusal_in_large_language_models_via_.md)**

:   本文提出 AdaCD（自适应对比解码），通过比较极端安全提示下和无提示下的 token 分布差异提取拒绝 token 分布，再根据一致性比率动态决定增强或抑制拒绝行为，在降低过度拒绝 10.35% 的同时提升恶意查询拒绝率 0.13%。

**[Preventing Safety Drift in Large Language Models via Coupled Weight and Activation Constraints](llm_safety/preventing_safety_drift_in_large_language_models_via_coupled_weight_and_activati.md)**

:   本文提出 CWAC，在微调时同时约束权重更新方向和安全关键激活特征，从理论和实验上说明单独约束权重或激活都不足以防止 LLM 安全漂移。

**[Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](llm_safety/privacy-r1_privacy-aware_multi-llm_agent_collaboration_via_reinforcement_learnin.md)**

:   Privacy-R1 将隐私敏感查询的本地/远程模型委托问题建模为逐句路由的序列决策任务，用轻量 Transformer policy + PPO 学到隐私与任务质量之间的动态折中，在 PUPA 和高 PII 密度的 Med-PCD 上都比静态改写方法取得更好的质量-泄露前沿。

**[Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models](llm_safety/privacy_collapse_benign_fine-tuning_can_break_contextual_privacy_in_language_mod.md)**

:   本文提出“隐私坍塌”这一新失败模式：看似良性的微调会让 LLM 在上下文隐私规范上系统性退化，同时常规安全和能力指标仍表现正常。

**[ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks](llm_safety/proxyprompt_securing_system_prompts_against_prompt_extraction_attacks.md)**

:   ProxyPrompt 不再要求模型“不要泄露系统提示词”，而是用功能等价但语义混淆的 proxy prompt 替换原 prompt，在保持任务效用的同时让被抽取出的提示词难以复现原任务，264 个配置上达到 94.70% 保护率，显著高于过滤式和指令式防御。

**[Purging the Gray Zone: Latent-Geometric Denoising for Precise Knowledge Boundary Awareness](llm_safety/purging_the_gray_zone_latent-geometric_denoising_for_precise_knowledge_boundary_.md)**

:   本文提出 GeoDe 框架，通过在 LLM 隐空间中训练线性探针构建真值超平面，利用样本到超平面的几何距离作为置信度信号来筛选高质量弃权微调数据，有效消除决策边界附近的"灰色地带"噪声，显著提升模型的真实性和可靠性。

**[Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](llm_safety/reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)**

:   本文提出"推理劫持"(Reasoning Hijacking) 这一新型攻击范式，通过在数据通道注入虚假决策标准来操纵 LLM 的推理逻辑而非改变任务目标，实现高攻击成功率且能绕过基于意图检测的防御方法。

**[Reasoning Structure Matters for Safety Alignment of Reasoning Models](llm_safety/reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)**

:   论文指出大型推理模型的安全问题根源在“先理解问题、再全力求解”的推理结构，并提出 AltTrain 用 1K 条 SFT 数据把推理结构改成“问题理解 → 有害性评估 → 条件推理”，显著降低有害响应同时基本保留推理能力。

**[Red-Bandit: Test-Time Adaptation for LLM Red-Teaming via Bandit-Guided LoRA Experts](llm_safety/red-bandit_test-time_adaptation_for_llm_red-teaming_via_bandit-guided_lora_exper.md)**

:   Red-Bandit 将自动化 LLM 红队建模为“多个攻击风格 LoRA 专家 + 测试时 bandit 路由”的在线适应问题，在多个开源和闭源目标模型上用更高 ASR@10 和更低困惑度展示了风格级自适应红队的有效性。

**[Representation-Guided Parameter-Efficient LLM Unlearning](llm_safety/representation-guided_parameter-efficient_llm_unlearning.md)**

:   提出 ReGLU 框架，将 LLM 遗忘从"参数重要性"范式转向"表示空间几何"范式——通过表示引导的 LoRA 初始化（RILA）将遗忘更新对齐到遗忘/保留集最具区分性的子空间，配合表示正交损失（ROL）约束更新不干扰保留集知识。

**[Responsible Federated LLMs via Safety Filtering and Constitutional AI](llm_safety/responsible_federated_llms_via_safety_filtering_and_constitutional_ai.md)**

:   这篇论文把安全过滤器和 Constitutional AI 接入 FedLLM 流程，证明有害客户端数据会显著损害全局模型安全，而在客户端过滤数据、在服务器端低成本 CAI 微调可以把 AdvBench 安全分数从约 72% 拉回到 96% 以上。

**[Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights](llm_safety/rethinking_evaluation_for_llm_hallucination_detection_a_desiderata_a_new_rag-bas.md)**

:   本文重新定义了 RAG 场景幻觉检测 benchmark 应具备的 7 条要求，构建了长上下文、人类多轮标注且带真实噪声标签的 Trivia++，并发现现有检测器在有机 RAG 幻觉上仍明显低于理想性能。

**[Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](llm_safety/rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)**

:   提出表征对比评分（RCS）框架，通过分析 LVLM 内部中间层表征的几何结构，用轻量投影和对比评分区分恶意意图与良性分布偏移，在跨攻击类型泛化的严格评估协议下实现 SOTA 越狱检测性能。

**[Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework](llm_safety/rethinking_llm_watermark_detection_in_black-box_settings_a_non-intrusive_third-p.md)**

:   提出 TTP-Detect，首个将水印检测与注入解耦的黑盒第三方水印验证框架，通过代理模型放大水印信号并结合局部一致性、全局几何和自适应秩检验三类互补度量，在不访问密钥或内部模型状态的情况下实现跨水印方案的高精度检测。

**[Retrievals Can Be Detrimental: Unveiling the Backdoor Vulnerability of Retrieval-Augmented Diffusion Models](llm_safety/retrievals_can_be_detrimental_unveiling_the_backdoor_vulnerability_of_retrieval-.md)**

:   提出 BadRDM，首个针对检索增强扩散模型（RDM）的后门攻击框架，通过恶意对比学习微调检索器建立触发词到毒性代理图像的捷径，在类条件和 T2I 两种任务中分别达到 90.9% 和 96.4% 攻击成功率，同时保持良性生成质量。

**[Revisiting Non-Verbatim Memorization in Large Language Models: The Role of Entity Surface Forms](llm_safety/revisiting_non-verbatim_memorization_in_large_language_models_the_role_of_entity.md)**

:   本文通过构建 RedirectQA 数据集（利用 Wikipedia 重定向信息将同一实体关联到多种表面形式），系统研究了 LLM 的非逐字记忆如何受实体命名变体的影响，发现事实记忆既非纯粹依赖特定表面形式也非完全表面无关，且实体级频率在表面频率之外仍有独立贡献。

**[RISK: A Framework for GUI Agents in E-commerce Risk Management](llm_safety/risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)**

:   提出 RISK 框架，包含领域数据集（RISK-Data, 8492单步+2386多步轨迹）、基准（RISK-Bench）和基于GRPO的强化微调方法（RISK-R1），针对电商风控场景的GUI智能体，7B模型以仅7.2%的参数量超越SOTA基线，在线任务成功率达70.5%。

**[Robust Multimodal Safety via Conditional Decoding](llm_safety/robust_multimodal_safety_via_conditional_decoding.md)**

:   本文提出 CASA 条件解码框架，让多模态模型在生成回答前先预测安全 token，并用安全注意力放大恶意信号，在文本、视觉和音频越狱基准上把平均攻击成功率降低 97% 以上，同时基本保持良性输入的多模态能力。

**[Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction](llm_safety/robustness_via_referencing_defending_against_prompt_injection_attacks_by_referen.md)**

:   本文提出一种基于指令引用的提示注入防御方法，不压制 LLM 的指令遵循能力，而是让模型在响应中引用正在执行的指令，然后通过标签过滤移除与原始指令不相关的响应，在部分场景下将攻击成功率降至接近 0%。

**[Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffixes](llm_safety/route_to_rome_attack_directing_llm_routers_to_expensive_models_via_adversarial_s.md)**

:   本文提出 R2A（Route to Rome Attack），通过在黑盒设置下构建混合集成代理路由器并优化通用对抗后缀，将 LLM 路由器的路由决策从廉价弱模型导向昂贵强模型——在 7 个开源路由器和 2 个商用路由器（GPT-5-Auto、OpenRouter）上平均攻击成功率提升 49%，推理成本增加 2.7-2.9 倍。

**[SafeConstellations: Mitigating Over-Refusals in LLMs Through Task-Aware Representation Steering](llm_safety/safeconstellations_mitigating_over-refusals_in_llms_through_task-aware_represent.md)**

:   SafeConstellations 发现 LLM 的中后层表示会按任务形成稳定的“星座轨迹”，并在高置信 benign 任务上把表示从拒答轨迹轻量推向非拒答轨迹，从而显著降低过度拒答且基本不损害通用能力。

**[SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging](llm_safety/safemerge_preserving_safety_alignment_in_fine-tuned_large_language_models_via_se.md)**

:   本文提出 SafeMERGE，一种轻量级后微调框架，通过余弦相似度检测偏离安全行为的微调层，仅将这些层与安全模型的对应层合并，在四个 LLM 上显著降低有害输出同时保持甚至提升任务性能。

**[SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](llm_safety/safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)**

:   本文提出 SafetyALFRED 基准，在 ALFRED 具身任务中引入六类厨房安全隐患，揭示了多模态大语言模型在静态 QA 中能识别危险（最高 92%）但在具身规划中却难以主动缓解危险（<60%）的严重对齐差距，倡导从 QA 评估范式转向具身安全评估。

**[SAGE: Sparse Adaptive Guidance for Dependency-Aware Tabular Data Generation](llm_safety/sage_sparse_adaptive_guidance_for_dependency-aware_tabular_data_generation.md)**

:   SAGE 把表格特征离散为 value-aware pseudo-features，并用互信息构建稀疏动态依赖图来引导 LLM 生成，从而提升合成表格数据的下游效用、约束一致性和真实感。

**[Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking](llm_safety/seeing_no_evil_blinding_large_vision-language_models_to_safety_instructions_via_.md)**

:   提出 Attention-Guided Visual Jailbreaking，通过抑制模型对安全指令的注意力并将注意力锚定到对抗图像特征上，绕过而非强攻安全对齐机制，在 Qwen-VL 上达到 94.4% 攻击成功率，同时减少 45% 的梯度冲突。

**[SERE: Structural Example Retrieval for Enhancing LLMs in Event Causality Identification](llm_safety/sere_structural_example_retrieval_for_enhancing_llms_in_event_causality_identifi.md)**

:   SERE 认为事件因果识别中的示例选择不能只看语义相似度，而应检索概念路径、句法树和因果模式都结构相近的样例，从而让 LLM 在少样本推理时更少过度预测因果关系。

**[SHAPE: Unifying Safety, Helpfulness and Pedagogy for Educational LLMs](llm_safety/shape_unifying_safety_helpfulness_and_pedagogy_for_educational_llms.md)**

:   这篇论文把教育 LLM 的安全、帮助性和教学性统一到知识掌握图上，提出 SHAPE 基准评估模型在答案诱导压力下是否仍能按学生掌握状态选择“引导”或“直接作答”，并用图增强 gating pipeline 大幅提升鲁棒性。

**[SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models](llm_safety/sharedrequest_privacy-preserving_model-agnostic_inference_for_large_language_mod.md)**

:   提出 SharedRequest，一种模型无关的隐私保护 LLM 推理框架，通过将隐私保护从单个 prompt 级提升到批次级——混合真实与噪声 prompt 并共享语义等价请求的推理开销——实现 >20% 的效用提升和最高 5.6× 的查询成本降低。

**[SLIM: Stealthy Low-Coverage Black-Box Watermarking via Latent-Space Confusion Zones](llm_safety/slim_stealthy_low-coverage_black-box_watermarking_via_latent-space_confusion_zon.md)**

:   SLIM 提出一种面向单个数据所有者的低覆盖数据水印思路：通过让模型在局部潜在空间学到相近前缀对应分歧续写的模式，最终在黑盒生成中表现出可统计检测的局部不稳定性。

**[SSG: Logit-Balanced Vocabulary Partitioning for LLM Watermarking](llm_safety/ssg_logit-balanced_vocabulary_partitioning_for_llm_watermarking.md)**

:   这篇论文分析了 KGW 类 LLM 水印在代码生成和数学推理等低熵场景中失效的原因，提出 Watermark Strength 度量和 SSG 的 logit-balanced vocabulary partition，让高概率 token 更均衡地分布到两类集合中，从而在不进一步降低生成质量的前提下显著提升水印可检测性。

**[STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming](llm_safety/star-teaming_a_strategy-response_multiplex_network_approach_to_automated_llm_red.md)**

:   本文提出 STAR-Teaming，一种基于策略-响应多路复用网络（Multiplex Network）的自动化红队测试框架，通过将攻击策略选择建模为逆 Ising 问题的概率优化，在 HarmBench 上达到平均 74.5% 的攻击成功率，比最强基线高 13.5%，同时显著降低计算开销。

**[Subject-level Inference for Realistic Text Anonymization Evaluation](llm_safety/subject-level_inference_for_realistic_text_anonymization_evaluation.md)**

:   SPIA 提出首个主体级 PII 推断评估基准（675 篇文档、1712 个主体、7040 个 PII），揭示即使 90%+ 的 PII 片段被遮蔽，主体级推断保护率可低至 33%，且聚焦单一目标主体的匿名化会导致非目标主体暴露更多。

**[SWAN: Semantic Watermarking with Abstract Meaning Representation](llm_safety/swan_semantic_watermarking_with_abstract_meaning_representation.md)**

:   SWAN 用 Abstract Meaning Representation 模板把水印嵌入句子的语义图结构，而不是 token 或 embedding 区域，因此在保持原意的 paraphrase 后仍能通过 AMR 解析、模板匹配和比例 z 检验检测出水印。

**[The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination](llm_safety/the_reasoning_trap_how_enhancing_llm_reasoning_amplifies_tool_hallucination.md)**

:   系统性揭示了"推理陷阱"悖论：增强LLM推理能力（无论通过RL、蒸馏还是可切换推理模式）会系统性地放大工具幻觉，且这一效应与推理本身而非RL训练相关联，现有缓解策略（提示工程、DPO）面临不可避免的可靠性-能力权衡。

**[Topic-Based Watermarks for Large Language Models](llm_safety/topic-based_watermarks_for_large_language_models.md)**

:   本文提出基于主题的轻量水印方案 TBW，将词表按语义主题聚类为"绿色列表"（而非随机分区），根据输入提示选择语义对齐的主题列表进行 logit 偏置，在保持与无水印文本相当的困惑度的同时，显著提升了对释义和词汇扰动攻击的鲁棒性。

**[TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](llm_safety/tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)**

:   本文提出 TPA 框架，通过数学方法将 LLM 每个 token 的生成概率精确分解为七个来源（Query、RAG Context、Past Token、Self Token、FFN、Final LayerNorm、Initial Embedding）的贡献，结合词性标注聚合特征，实现 RAG 场景下的 SOTA 幻觉检测。

**[Train in Vain: Functionality-Preserving Poisoning to Prevent Unauthorized Use of Code Datasets](llm_safety/train_in_vain_functionality-preserving_poisoning_to_prevent_unauthorized_use_of_.md)**

:   这篇论文提出 FunPoison，在保持 Java 代码可编译、可执行和功能等价的前提下，把执行惰性的弱使用片段注入真实执行路径，只污染 10% 数据就能显著削弱未授权 CodeLLM 微调收益，并对格式化、重写、静态分析和检测清洗表现出较强鲁棒性。

**[TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense](llm_safety/trajguard_streaming_hidden-state_trajectory_detection_for_decoding-time_jailbrea.md)**

:   本文提出 TrajGuard，一种无需训练的解码时越狱防御框架，通过滑动窗口聚合关键层隐藏状态轨迹实时量化风险，仅在风险持续超过阈值时触发轻量级语义裁判，在 12 种越狱攻击上实现 95% 平均防御率，检测延迟仅 5.2ms/token，误报率低于 1.5%。

**[TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards](llm_safety/trojail_trajectory-level_optimization_for_multi-turn_large_language_model_jailbr.md)**

:   本文将自动化多轮越狱攻击建模为多轮强化学习问题，提出 TROJail，通过两个启发式过程奖励（过度有害惩罚和语义相关性递进）缓解结果奖励的稀疏监督问题，在多个模型和基准上显著提升攻击成功率。

**[Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations](llm_safety/two_pathways_to_truthfulness_on_the_intrinsic_encoding_of_llm_hallucinations.md)**

:   本文发现 LLM 内部编码真实性信号存在两条不同的信息通路：Question-Anchored（依赖问题到回答的信息流）和 Answer-Anchored（从生成答案本身提取自包含证据），两者与知识边界紧密关联，并据此提出 Mixture-of-Probes 和 Pathway Reweighting 两种通路感知的幻觉检测方法，AUC 提升达 10%。

**[Understanding and Mitigating Bias Inheritance in LLM-based Data Augmentation on Downstream Tasks](llm_safety/understanding_and_mitigating_bias_inheritance_in_llm-based_data_augmentation_on_.md)**

:   这篇论文系统研究 LLM 生成的有偏增强数据如何在监督微调中被继承、放大并影响下游任务，并用六类偏见生成框架、十个任务和三类缓解方法揭示了“合成数据越多不一定越安全”的复杂现象。

**[Unlearners Can Lie: Evaluating and Improving Honesty in LLM Unlearning](llm_safety/unlearners_can_lie_evaluating_and_improving_honesty_in_llm_unlearning.md)**

:   这篇论文指出现有 LLM unlearning 方法即使“忘掉了”目标知识，也常会幻觉、伪装拒答或前后不一致，于是提出 honest unlearning 评测框架和 ReVa 表示对齐方法，让模型在遗忘后更稳定地承认自己不知道。

**[VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](llm_safety/vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)**

:   提出 VLA-Forget，首个面向视觉-语言-动作（VLA）模型的混合遗忘框架，通过 ratio-aware 选择性编辑处理感知/跨模态层、significance-based 选择性编辑处理推理/动作层，实现目标行为移除同时保持感知精确性（+22%）和任务成功率（+9%）。

**[When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life](llm_safety/when_helpers_become_hazards_a_benchmark_for_analyzing_multimodal_llm-powered_saf.md)**

:   提出 SaLAD 基准，包含 2013 个真实图文样本覆盖 10 类日常场景，评估多模态大模型在日常辅助中识别隐性安全风险并提供安全警告的能力，揭示即使最强模型在不安全查询上准确率也仅 57.2%。

**[When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](llm_safety/when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)**

:   这篇论文发现大推理模型的安全失败常发生在“已经识别风险之后又被后续推理推翻”，并提出 Chain-of-Guardrail 通过定位和修复危险推理片段，在显著降低攻击成功率的同时保留数学与代码推理能力。

**[Why Agents Compromise Safety Under Pressure](llm_safety/why_agents_compromise_safety_under_pressure.md)**

:   提出"代理压力"（Agentic Pressure）概念——当 LLM 代理在资源约束下无法同时完成任务和遵守安全规则时，会自发地产生规范漂移，主动牺牲安全以保持有用性，且推理能力越强的模型越善于构建语言化合理化来为违规辩护。

**[XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts](llm_safety/xmark_reliable_multi-bit_watermarking_for_llm-generated_texts.md)**

:   提出 XMark，一种基于 Leave-one-Shard-out（LoSo）策略和 evergreen list 的多比特文本水印方法，通过跨多个词表排列的绿色列表交集和约束 token-shard 映射矩阵，在保持文本质量的同时显著提升了有限 token 条件下的解码准确率。

**[XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants](llm_safety/xoxo_stealthy_cross-origin_context_poisoning_attacks_against_ai_coding_assistant.md)**

:   揭示了 AI 编码助手自动收集上下文的设计漏洞，提出 Cross-Origin Context Poisoning（XOXO）攻击：通过语义保持的代码变换（如变量重命名）毒化共享代码库，使 GitHub Copilot 等助手在不知情的情况下生成有漏洞的代码，对 8 个 SOTA 模型平均攻击成功率达 73.20%。

---

## 🧩 多模态 VLM { #multimodal_vlm }

**[A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](multimodal_vlm/a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)**

:   本综述提出 Perception–Alignment–Reasoning (PAR) 过程框架 + Answer–Process–Executable (APE) 评估框架两个互补视角，系统地组织几何/图表表格/视觉应用题三大任务族，把现有方法和 benchmark 都映射到这两个十字坐标上，是首篇 process-centric 多模态数学推理综述。

**[A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends](multimodal_vlm/a_survey_on_mllm-based_visually_rich_document_understanding_methods_challenges_a.md)**

:   系统综述基于多模态大语言模型（MLLM）的视觉丰富文档理解（VRDU），从特征表示/融合和训练范式两个维度梳理OCR-based和OCR-free方法，并讨论数据稀缺、多页文档、多语言支持、RAG和智能体等新兴方向。

**[AdaTooler-V: Adaptive Tool-Use for Images and Videos](multimodal_vlm/adatooler-v_adaptive_tool-use_for_images_and_videos.md)**

:   本文指出现有"thinking with images" MLLM 普遍存在**盲目工具调用**问题——所有视觉问题都强行 zoom-in/抽帧，反而 overthinking 降准、增推理成本；为此提出 AdaTooler-V，引入 AT-GRPO 强化学习算法用样本级 Tool Benefit Score 动态调节奖励尺度（工具有效时鼓励、无效时惩罚），让 7B 模型在 V* 高分辨率基准上达到 89.8%，超过 GPT-4o 与 Gemini 1.5 Pro。

**[Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization](multimodal_vlm/addressing_overthinking_in_large_vision-language_models_via_gated_perception-rea.md)**

:   提出GPRO框架，通过元推理控制器在每个token生成步动态路由计算到三条路径（快速/感知重检/推理反思），解决LVLM的过度思考问题，同时提升精度和效率。

**[AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning in E-commerce](multimodal_vlm/afmrl_attribute-enhanced_fine-grained_multi-modal_representation_learning_in_e-c.md)**

:   提出 AFMRL 框架，将电商产品的细粒度理解定义为属性生成任务，通过 MLLM 生成关键属性来增强对比学习（AGCL），并用检索性能作为奖励信号反向优化属性生成器（RAR），在大规模电商数据集上实现 SOTA 检索性能。

**[AICA-Bench: Holistically Examining the Capabilities of VLMs in Affective Image Content Analysis](multimodal_vlm/aica-bench_holistically_examining_the_capabilities_of_vlms_in_affective_image_co.md)**

:   提出 AICA-Bench，一个涵盖情感理解（EU）、情感推理（ER）和情感引导内容生成（EGCG）三个维度的综合基准，评估 23 个 VLM 后发现模型存在强度校准失败和描述浅薄两大缺陷，并提出 GAT Prompting 训练无关框架来缓解这些问题。

**[All Changes May Have Invariant Principles: Improving Ever-Shifting Harmful Meme Detection via Design Concept Reproduction](multimodal_vlm/all_changes_may_have_invariant_principles_improving_ever-shifting_harmful_meme_d.md)**

:   提出RepMD方法，通过构建设计概念图（DCG）——借鉴攻击树思想描述恶意用户设计有害梗图的步骤和逻辑——来引导MLLM检测不断变化的有害梗图，在GOAT-Bench上达81.1%准确率。

**[Almieyar-Oryx-BloomBench: A Bilingual Multimodal Benchmark for Cognitively Informed Evaluation of Vision-Language Models](multimodal_vlm/almieyar-oryx-bloombench_a_bilingual_multimodal_benchmark_for_cognitively_inform.md)**

:   BloomBench 用 Bloom 认知分类法重构 VLM 评测，将 7,747 个英阿双语图文问答样本组织为 6 个认知层级和 106 个任务类型，并发现当前 VLM 的高分往往掩盖了事实回忆、创造性综合和跨语言推理上的明显短板。

**[Automatic Slide Updating with User-Defined Dynamic Templates and Natural Language Instructions](multimodal_vlm/automatic_slide_updating_with_user-defined_dynamic_templates_and_natural_languag.md)**

:   定义了"基于自然语言指令在用户自定义模板上进行动态幻灯片更新"的新任务，构建了包含 20,036 个指令-执行三元组的 DynaSlide 基准，并提出了 SlideAgent 作为强参考基线。

**[Benchmarking Deflection and Hallucination in Large Vision-Language Models](multimodal_vlm/benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)**

:   提出 VLM-DeflectionBench，一个包含 2775 个样本的多模态基准，通过四种评估场景（参数化/Oracle/现实/对抗）系统性地评估大型视觉语言模型在证据不足或误导时的拒答（deflection）vs 幻觉（hallucination）行为，实验覆盖 20 个 SOTA LVLM，发现几乎所有模型都无法在噪声证据下可靠拒答。

**[Beyond Screenshots: Evaluating VLMs' Understanding of UI Animations](multimodal_vlm/beyond_screenshots_evaluating_vlms_understanding_of_ui_animations.md)**

:   构建首个 UI 动画理解评测集 AniMINT（300 段密集标注的动画视频 + 3 位专家 + 300 用户标注），系统测试 9 个 SOTA VLM 后发现：基础运动效果能识别，但动画用途分类和高层语义解读与人类差距巨大，进一步用 Motion-Context-Perceptual Cue (MCPC) 增强能在 Gemini-2.5-Flash 上同时提升分类和解读性能。

**[Can MLLMs Reason Beyond Language? VisReason: A Comprehensive Benchmark for Vision-Centric Reasoning](multimodal_vlm/can_mllms_reason_beyond_language_visreason_a_comprehensive_benchmark_for_vision-.md)**

:   VisReason 构建了一个包含 1,505 道日常视觉推理题的多模态 benchmark，专门测试模型是否能直接基于视觉证据推理，结果显示最强模型平均准确率也只有 47.5%，显著低于人类 71.4%，且 CoT 与更大推理预算只能带来有限提升。

**[CARES: Context-Aware Resolution Selector for VLMs](multimodal_vlm/cares_context-aware_resolution_selector_for_vlms.md)**

:   CARES 在目标 VLM 前增加一个轻量 query-aware 分辨率选择器，用低分辨率图像和文本问题预测“足够回答”的最小输入分辨率，在 9 个多模态 benchmark 上基本保持准确率，同时平均节省约 65–85% 的 prefill 计算成本。

**[CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity](multimodal_vlm/cartbench_evaluating_vision-language_models_on_chinese_art_understanding_interpr.md)**

:   本文构建了 CArtBench——一个基于故宫博物院藏品的多任务基准，评估 VLM 在中国艺术理解中的四种能力（证据问答、结构化鉴赏、可辩护重解读、真伪辨别），发现即使最强模型在证据关联和风格-年代推理上也存在显著性能下降，而真伪辨别接近随机水平。

**[ChartDiff: A Large-Scale Benchmark for Comprehending Pairs of Charts](multimodal_vlm/chartdiff_a_large-scale_benchmark_for_comprehending_pairs_of_charts.md)**

:   作者构建了第一个面向"两图对比摘要"的大规模 benchmark ChartDiff（8,541 对图表，覆盖 6 种图类型、3 个绘图库、约 60 种视觉风格、LLM 生成 + 人工核验的对比摘要），系统评测了 14 个 VLM/pipeline，发现前沿闭源大模型在 GPT Score 上领先但 ROUGE 低，专业图表模型/pipeline 反之，揭示了 ROUGE 与人感知质量的严重失配；同时多系列图始终是所有模型最难的死角。

**[ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding](multimodal_vlm/chemvlr_prioritizing_reasoning_in_perception_for_chemical_vision-language_unders.md)**

:   提出 ChemVLR，首个化学领域推理型 VLM，通过跨模态逆向工程策略构建 760K 推理数据集，结合持续预训练-SFT-RL 三阶段训练流程，在分子识别和反应预测任务上显著超越专有模型和领域专家 VLM。

**[CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language](multimodal_vlm/cnsl-bench_benchmarking_the_sign_language_understanding_capabilities_of_mllms_on.md)**

:   CNSL-bench 是首个基于《国家通用手语词典》的权威中国手语 MLLM 评测基准，覆盖 6,707 个唯一手语词条 × 文本/图片/视频三模态 × 三种手部 articulation（空写/指拼/手指字母）共 20,121 道四选一题，在 21 个 SOTA MLLM 上揭示：GPT-5 文本 89.6%、图片 67.0%、视频 56.7%，相对人类 97% 仍有巨大 gap，且 CoT 推理对视频帮助微弱。

**[CogGen: A Cognitively Inspired Recursive Framework for Deep Research Report Generation](multimodal_vlm/coggen_a_cognitively_inspired_recursive_framework_for_deep_research_report_gener.md)**

:   CogGen 提出一个模拟人类认知写作过程的多智能体递归框架，通过宏观认知循环实现全局重构、微观认知循环实现并行章节精炼、抽象视觉表示（AVR）实现文本-图表的语义级协同规划，在 OWID 基准上达到人类专家水平并超越 Gemini Deep Research。

**[Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models](multimodal_vlm/cross-cultural_expert-level_art_critique_evaluation_with_vision-language_models.md)**

:   论文提出 Vulca-Bench 三层评测框架（自动指标 + 单评审打分 + 人类 sigmoid 校准），覆盖 6 大艺术传统、165 个文化维度与 L1–L5 五层"视觉描述→文化诠释"层级，在 15 个 VLM 上首次量化揭示了"模型在深层文化诠释上掉点显著、且系统性偏好西方艺术"的现象。

**[Cross-Modal Taxonomic Generalization in (Vision-) Language Models](multimodal_vlm/cross-modal_taxonomic_generalization_in_vision-_language_models.md)**

:   本文系统研究 VLM 中语言模型是否能将纯文本习得的分类学知识（上位词关系）跨模态泛化到视觉输入，发现即使训练时完全不提供上位词标签，预训练 LM 仍能在图像中识别上位词类别，但这种泛化需要类别成员在视觉上的一致性。

**[Decoding Scientific Experimental Images: The SPUR Benchmark for Perception, Understanding, and Reasoning](multimodal_vlm/decoding_scientific_experimental_images_the_spur_benchmark_for_perception_unders.md)**

:   SPUR 是首个针对生物医学实验图像（多面板染色图/Western blot/统计图）"感知 → 理解 → 推理"三阶段评测的 benchmark，包含 4264 道专家审定 MCQ，揭示当今 MLLM 仅 Gemini 3 Pro Preview 勉强突破 60%，定量推理普遍比定性推理低 12.76%–31.41%。

**[DMN: A Compositional Framework for Jailbreaking Multimodal LLMs with Multi-Image Inputs](multimodal_vlm/dmn_a_compositional_framework_for_jailbreaking_multimodal_llms_with_multi-image_.md)**

:   这篇论文提出 DMN，用分布式指令、多模态证据和数字链辅助任务组合成多图越狱评测框架，证明当前支持多图输入的 MLLM 在跨图安全对齐上存在明显弱点，同时给出一个 multi-image-aware filter 作为初步防御。

**[Do MLLMs Capture How Interfaces Guide User Behavior? A Benchmark for Multimodal UI/UX Design Understanding](multimodal_vlm/do_mllms_capture_how_interfaces_guide_user_behavior_a_benchmark_for_multimodal_u.md)**

:   这篇论文提出 WiserUI-Bench，用 300 组真实 A/B 测试验证过的 UI 图片对和 684 条专家解释评测 MLLM 是否理解界面设计如何影响用户行为，结果显示现有模型在选择赢家时接近随机、在解释原因时也明显未达专家水平。

**[Do MLLMs Understand Pointing? Benchmarking and Enhancing Referential Reasoning in Egocentric Vision](multimodal_vlm/do_mllms_understand_pointing_benchmarking_and_enhancing_referential_reasoning_in.md)**

:   作者构建了首个真实+物理仿真混合的第一人称"手指指点"问答基准 EgoPoint-Bench（11.7k QA / 5 维度 / 3 级语义指代），证实当前 SOTA MLLM 普遍依赖"视觉邻近 / 显著性"伪相关而非真正解析指尖射线，并通过在仿真数据上 LoRA 微调获得平均最高 +25 点的提升与稳健的 sim-to-real 泛化。

**[Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models](multimodal_vlm/doc-pp_document_policy_preservation_benchmark_for_large_vision-language_models.md)**

:   本文提出 Doc-PP 基准，揭示大型视觉-语言模型（LVLM）在多模态文档问答中存在"推理诱导的安全缺口"——模型在需要跨模态推理时会绕过显式非披露策略泄露敏感信息，并提出 DVA（Decompose–Verify–Aggregation）结构化推理框架来显著降低泄露率。

**[DraDDP: A Multimodal Multi-Party Dialogue Discourse Parsing Dataset](multimodal_vlm/draddp_a_multimodal_multi-party_dialogue_discourse_parsing_dataset.md)**

:   DraDDP 构建了首个公开的英文多模态多方对话话语解析数据集，并用传统解析器、LLM 与多模态 LLM 系统评测了文本、音频、视频线索对依存边和话语关系识别的不同贡献。

**[DRIFT: Transferring Reasoning Priors for Efficient MLLM Fine-Tuning](multimodal_vlm/drift_transferring_reasoning_priors_for_efficient_mllm_fine-tuning.md)**

:   DRIFT 把"文本推理专家与多模态模型的参数差"当成方向先验，在多模态 SFT 反向传播时只对梯度做轻量偏置（不动权重），用 4K 多模态 CoT 数据、约 2 小时训练就能把 Qwen2.5-VL-7B 在 MathVista/MathVerse/WeMath 等基准上稳定推过参数合并基线和重型 SFT/RL 方法。

**[Dynamic Emotion and Personality Profiling for Multimodal Deception Detection](multimodal_vlm/dynamic_emotion_and_personality_profiling_for_multimodal_deception_detection.md)**

:   本文指出现有欺骗检测数据集仅提供受试者级别的情感/人格标签（同一人所有样本共用标签），提出样本级动态标注方案和可靠性加权多模态融合框架 Rel-DDEP，在欺骗检测 F1 上提升 2.53%，情感检测提升 2.66%，人格检测提升 9.30%。

**[E2E-GMNER: End-to-End Generative Grounded Multimodal Named Entity Recognition](multimodal_vlm/e2e-gmner_end-to-end_generative_grounded_multimodal_named_entity_recognition.md)**

:   提出E2E-GMNER，首个将实体识别、语义分类、视觉定位和隐式知识推理统一在单一多模态大语言模型中的端到端GMNER框架，通过CoT推理自适应判断视觉/知识线索的可用性，并引入高斯风险感知框扰动（GRBP）提升生成式框预测的鲁棒性。

**[EDU-CIRCUIT-HW: Evaluating Multimodal Large Language Models on Real-World University-Level STEM Student Handwritten Solutions](multimodal_vlm/edu-circuit-hw_evaluating_multimodal_large_language_models_on_real-world_univers.md)**

:   作者发布 1,334 条真实大学电路课手写作业的 EDU-CIRCUIT-HW 数据集，并提出"upstream 识别 + downstream 评分"双层评测协议，发现即便最强 MLLM（GPT-5.1 / Gemini-3-Preview）也有 37–85% 样本含识别错误，但仅 7–20% 会传播到评分；通过 LLM-judge 错误模式 + 仅 3.3% 人工兜底的 regrading 模块，可把 point-agreement 从 70 %提升到 76 %。

**[Efficient Inference for Large Vision-Language Models: Bottlenecks, Techniques, and Prospects](multimodal_vlm/efficient_inference_for_large_vision-language_models_bottlenecks_techniques_and_.md)**

:   本文提出一个系统性的LVLM推理效率分类体系，围绕编码-预填充-解码三阶段推理流水线分析瓶颈，揭示了"视觉token主导"导致的系统性效率屏障，并梳理了从信息密度塑形、长上下文注意力管理到内存带宽突破的完整优化技术图谱。

**[Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning](multimodal_vlm/enhancing_multimodal_large_language_models_for_ancient_chinese_character_evoluti.md)**

:   本文构建了一个包含11个任务、13万+实例的古汉字演变分析基准，评估了19个MLLM后发现现有模型在字形级识别和演变推理上能力有限，并提出字形驱动对比微调框架GEVO，在2B模型上实现全任务提升。

**[ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](multimodal_vlm/errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)**

:   本文形式化定义了多模态错误检测任务，并构建了 ErrorRadar 基准——包含 2,500 道来自真实学生作答的 K-12 多模态数学题，评估 MLLM 在错误步骤识别（STEP）和错误类型分类（CATE）两个子任务上的能力，发现最强模型 GPT-4o 仍落后人类评估约 10-15%。

**[Faithful-First Reasoning, Planning, and Acting for Multimodal LLMs](multimodal_vlm/faithful-first_reasoning_planning_and_acting_for_multimodal_llms.md)**

:   本文提出 Faithful-First RPA 框架，通过 FaithEvi 管线在每一步推理中评估感知忠实性（claimed objects 是否在图像中真实存在），以及 FaithAct 机制在推理生成过程中强制执行基于证据的规划和行动，在不降低任务准确率的前提下将感知忠实性提升最高 24%。

**[Forest Before Trees: Latent Superposition for Efficient Visual Reasoning](multimodal_vlm/forest_before_trees_latent_superposition_for_efficient_visual_reasoning.md)**

:   本文提出 Laser，通过动态窗口对齐学习（DWAL）在潜在空间中进行视觉推理，使模型在推理过程中维持未来语义的"概率叠加态"而非逐 token 精确预测，实现"先全局后局部"的认知层次，在 6 个基准上以仅 6 个推理 token（减少 97%+）达到潜在推理方法的 SOTA，超越 Monet 平均 5.03%。

**[From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision-Language Models](multimodal_vlm/from_heads_to_neurons_causal_attribution_and_steering_in_multi-task_vision-langu.md)**

:   提出 HONES 框架，通过先定位任务关键注意力头再以其为条件引导 FFN 神经元归因，实现了多任务 VLM 中跨异构任务的统一、无梯度的神经元级因果分析和轻量级任务性能提升。

**[From Inheritance to Saturation: Disentangling the Evolution of Visual Redundancy for Architecture-Aware MLLM Inference Acceleration](multimodal_vlm/from_inheritance_to_saturation_disentangling_the_evolution_of_visual_redundancy_.md)**

:   揭示 MLLM 推理中视觉冗余的两种来源——ViT 密集 tokenization 导致的固有冗余（IVR）和深层语义饱和导致的次生冗余（SSR，且其表现形式因骨干架构不同而异），提出 HalfV 框架分别处理两类冗余，在 Qwen2.5-VL 上实现4.1倍 FLOPs 加速且保留96.8%性能。

**[From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck](multimodal_vlm/from_verbatim_to_gist_distilling_pyramidal_multimodal_memory_via_semantic_inform.md)**

:   本文提出 MM-Mem，一种受模糊痕迹理论启发的金字塔式多模态记忆架构——将记忆分为感知缓冲层（视觉为主）、情景流层（事件级摘要）和符号图式层（知识图谱）三个层级，通过 SIB-GRPO（语义信息瓶颈+强化学习）自底向上压缩冗余、通过熵驱动自顶向下检索，在 4 个长视频 benchmark 上实现 SOTA。

**[GeoArena: Evaluating Open-World Geographic Reasoning in Large Vision-Language Models](multimodal_vlm/geoarena_evaluating_open-world_geographic_reasoning_in_large_vision-language_mod.md)**

:   本文提出 **GeoArena**，一个面向 LVLM 开放世界地理推理的"动态、无标签、过程导向"评测平台，把 in-the-wild 图像下的地理定位评估改写为成对推理对齐任务，用人类偏好 + Bradley-Terry 评分对 17 个前沿 LVLM 排榜，专家-众包一致率达 78%。

**[GeoRC: A Benchmark for Geolocation Reasoning Chains](multimodal_vlm/georc_a_benchmark_for_geolocation_reasoning_chains.md)**

:   提出 GeoRC，首个由GeoGuessr冠军级专家撰写的地理定位推理链基准（800条推理链，500个场景），评估VLM生成可审计推理链的能力，发现闭源VLM虽能匹敌人类定位准确率但推理链质量仍大幅落后，开源VLM则几乎等同于纯幻觉基线。

**[GroupToM-Bench: Benchmarking Group Theory of Mind and Nonlinear Social Emergence in MLLMs](multimodal_vlm/grouptom-bench_benchmarking_group_theory_of_mind_and_nonlinear_social_emergence_.md)**

:   这篇论文提出 GroupToM-Bench，用 240 个专家设计的多模态群体互动场景和 7 层认知审计框架评测 MLLM 是否能从个体信念/欲望/意图推理到群体张力、结构约束和非线性集体结果，结果显示当前模型普遍存在明显的 group cognitive gap。

**[GuideDog: A Real-World Egocentric Multimodal Dataset for Blind and Low-Vision Accessibility-Aware Guidance](multimodal_vlm/guidedog_a_real-world_egocentric_multimodal_dataset_for_blind_and_low-vision_acc.md)**

:   GuideDog 用「专家规范驱动的银标生成 + 人工核验金标」流水线，从 269 段全球行走视频中构建出 22K 张第一视角行人场景图文对（含 818 题 QA 基准），首次让 MLLM 在 BLV（盲与低视力）导航任务上有了规模化、地理多样、标准化的训练与评测数据。

**[HierVA: Hierarchical Visual Agent — Managing Contexts in Joint Image-Text Space for Advanced Chart Reasoning](multimodal_vlm/hierarchical_visual_agent_managing_contexts_in_joint_image-text_space_for_advanc.md)**

:   HierVA 用 "manager–worker" 双层多模态 agent，把图表推理过程中的图像和文本 context 都按"获取–限定–蒸馏"的纪律管理起来，零训练地在 CharXiv 等复杂图表推理 benchmark 上稳超 CoT 和 "thinking with images" 等强基线。

**[HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](multimodal_vlm/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)**

:   本文发现视觉编码器中存在层级注意力模式——中层关注主体对象、深层关注全局信息，据此提出 HiPrune，一种免训练、模型无关的视觉 token 剪枝方法，通过选择三类 token（Anchor/Buffer/Register）保留不同层级的视觉信息，仅用 1/3 token 保持 99.3% 性能，FLOPs 减少 58.7%。

**[How Do LLMs and VLMs Understand Viewpoint Rotation Without Vision? An Interpretability Study](multimodal_vlm/how_do_llms_and_vlms_understand_viewpoint_rotation_without_vision_an_interpretab.md)**

:   本文提出文本视角旋转理解 (VRU) 基准 VRUBench，用 layer-wise probing 与 head-wise path patching 揭示 LLM/VLM 之所以在该任务上接近随机的根因 —— 中后层关键头未能把"已感知的朝向"与"该朝向对应的观测"绑定，并通过只微调 32 个关键头就以 50% GPU 时间达到全量微调的效果且不损通用能力。

**["I See What You Did There": Can Large Vision-Language Models Understand Multimodal Puns?](multimodal_vlm/i_see_what_you_did_there_can_large_vision-language_models_understand_multimodal_.md)**

:   本文提出 MultiPun——首个带"对抗性 non-pun 干扰项"的多模态双关语 benchmark（445 个 pun + 890 个 non-pun，覆盖同音/同形两类），系统评测了 11 个 VLM 在双关检测/定位/解释三类任务上的表现，发现**所有模型都倾向于把 non-pun 也当作 pun**（TNR 普遍 < 0.4），并提出 Pun-CoT 提示策略 + Pun-Tuning 微调策略，平均 F1 提升 16.5%。

**[iReasoner: Trajectory-Aware Intrinsic Reasoning Supervision for Self-Evolving Large Multimodal Models](multimodal_vlm/ireasoner_trajectory-aware_intrinsic_reasoning_supervision_for_self-evolving_lar.md)**

:   iReasoner在无标注图像上让LMM自问自答，并把最终答案一致性扩展为中间CoT步骤的一致性奖励，从而在Qwen2.5-VL-7B上带来最高约+2.13点的多模态推理提升。

**[Jailbreaking Multimodal Large Language Models using Multi-Clip Video](multimodal_vlm/jailbreaking_multimodal_large_language_models_using_multi-clip_video.md)**

:   这篇论文构建MCV SafetyBench来评估视频MLLM安全性，发现多clip、多上下文的视频输入会系统性提高攻击成功率，而简单的抽帧图像过滤能显著降低这种风险。

**[LaMI: Augmenting Large Language Models via Late Multi-Image Fusion](multimodal_vlm/lami_augmenting_large_language_models_via_late_multi-image_fusion.md)**

:   提出 LaMI，通过后融合架构在预测最后阶段融合视觉特征与 LLM 输出，并在推理时从文本生成多张图像进行基于置信度的聚合，在不损害文本推理能力的前提下显著提升 LLM 的视觉常识推理能力。

**[Learning More from Less: Exploiting Counterfactuals for Data-Efficient Chart Understanding](multimodal_vlm/learning_more_from_less_exploiting_counterfactuals_for_data-efficient_chart_unde.md)**

:   针对 chart 是"程序化生成的视觉产物"这一独特属性，提出 ChartCF——通过 GPT-5 在绘图代码上做最小改动生成视觉相似但答案不同的"反事实图对"，再用文本 DPO + 图像 DPO 联合偏好优化让 VLM 学会细粒度视觉判别，仅用 4K 偏好对就在多个 chart QA benchmark 上匹配或超过用 300K SFT 数据训练的 ECD。

**[Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Unauthorized Images](multimodal_vlm/leave_my_images_alone_preventing_multi-modal_large_language_models_from_analyzin.md)**

:   提出 ImageProtector，通过在图像中嵌入近不可察觉的对抗扰动作为视觉提示注入攻击，使 MLLM 对被保护图像生成拒绝响应，从而阻止恶意分析者利用开放权重 MLLM 大规模提取图像中的隐私信息。

**[Long Story Short: Disentangling Compositionality and Long-Caption Understanding in Contrastive VLMs](multimodal_vlm/long_story_short_disentangling_compositionality_and_long-caption_understanding_i.md)**

:   系统拆解了对比 VLM 里"组合性推理 (compositionality)"和"长 caption 理解 (long-caption understanding)" 这两项能力之间的关系——发现它俩是双向相互促进的，但这种迁移**对训练数据质量和优化策略极度敏感**：用 grounded + 高词表覆盖的长 caption 数据 + 全参数微调能同时拿满两个能力，而 DAC/DCI 的低质量合成 caption + LoRA 部分更新就会两头垮；LongCLIP 把前 20 个位置 embedding 冻结看似保护了通用对齐，实则限死了组合学习——作者的"control 模型" LSS 在原 77-token 上下文窗口内全参微调 ShareGPT4V，性能反超 LongCLIP。

**[MathFlow: Enhancing the Perceptual Flow of MLLMs for Visual Mathematical Problems](multimodal_vlm/mathflow_enhancing_the_perceptual_flow_of_mllms_for_visual_mathematical_problems.md)**

:   提出 FlowVerse 基准（将数学问题信息分为 DI/EI/RP/OQ 四个组件并构建六个变体版本）和 MathFlow 模块化管线（将感知和推理解耦为独立阶段），训练专门的感知模型 MathFlow-P-7B 从数学图表中提取关键信息，显著提升各类推理模型的视觉数学问题解决能力。

**[Measuring What Matters Beyond Text: Evaluating Multimodal Summaries by Quality, Alignment, and Diversity (MM-Eval)](multimodal_vlm/measuring_what_matters_beyond_text_evaluating_multimodal_summaries_by_quality_al.md)**

:   针对"多模态摘要带多模态输出 (MSMO)"任务，提出 MM-Eval 评测框架：把文本质量 (OpenFActScore + G-Eval)、跨模态对齐 (MLLM-as-Judge) 和视觉多样性 (Truncated CLIP Entropy) 三个分项分数用 Ridge 回归学到的权重聚合成单一打分，在 mLLM-EVAL 新闻基准上对人类偏好的 Kendall $\tau$ 从 equal-weight baseline 的 0.041 提升到 0.374。

**[MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models](multimodal_vlm/medlaybench-v_a_large-scale_benchmark_for_expert-lay_semantic_alignment_in_medic.md)**

:   本文提出 MedLayBench-V，首个大规模多模态医学专家-通俗语义对齐基准（79,793 图文对），通过 Structured Concept-Grounded Refinement (SCGR) 流水线将专业放射学报告转化为通俗描述，确保临床语义保真的同时将阅读难度从研究生级别降至高中水平，零样本检索实验表明通俗描述仅带来不到 1% 的性能损失。

**[Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](multimodal_vlm/mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)**

:   本文提出 MPD 框架，通过语义感知正交子空间投影分离幻觉成分，并仅选择性更新与幻觉最相关的少量参数，在减少 23.4% 幻觉的同时保持 97.4% 的通用生成能力，不引入额外推理开销。

**[MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models](multimodal_vlm/mmerror_a_benchmark_for_erroneous_reasoning_in_vision-language_models.md)**

:   本文提出 MMErroR，一个包含 1997 个样本的多模态错误推理基准，每个样本嵌入一个单一推理错误，覆盖 6 大领域和 4 种错误类型，要求 VLM 不仅检测推理链中的错误存在，还要分类错误类型（视觉感知错误/知识应用错误/问题理解错误/推理错误），评测 12 个代表性 VLM 后发现最强模型 Gemini-3-Pro-Preview 也仅达 66.65% 准确率。

**[MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems](multimodal_vlm/moneta_multimodal_industry_classification_through_geographic_information_with_mu.md)**

:   本文提出 MONETA，首个结合文本（网站、维基百科、Wikidata）和地理空间数据（OpenStreetMap、卫星图像）的多模态行业分类基准，并设计零样本和多轮多智能体两种无训练管线，使用开源和闭源 MLLM 在 20 类 NACE 行业分类上达到 62.10%-74.10% 准确率，多轮设计最高提升 22.80%。

**[More Than Meets the Eye: Measuring the Semiotic Gap in Vision-Language Models via Semantic Anchorage](multimodal_vlm/more_than_meets_the_eye_measuring_the_semiotic_gap_in_vision-language_models_via.md)**

:   本文从认知符号学角度揭示 VLM 的"字面优越偏差"——模型在高保真图像上倾向于字面解读而非隐喻/习语理解，通过引入 DIVA 基准（图标化简化图像）和 Semantic Alignment Gap 指标，证明降低视觉保真度能显著缩小字面与习语解读之间的鸿沟。

**[MSEarth: A Multimodal Benchmark for Earth Science Phenomenon Discovery with MLLMs](multimodal_vlm/msearth_a_multimodal_benchmark_for_earth_science_phenomenon_discovery_with_mllms.md)**

:   从 64,560 篇 CC-BY 开源地球科学论文里抽出 289K 张图，用「raw caption + 正文上下文」合成 refined caption，再用 5 模型多智能体投票 + 三阶段 PhD 专家校验生成 7,195 题 graduate-level 测试集（含 captioning / MCQ / open-ended），系统揭示了 SOTA MLLM 在 Earth-science 多图推理上"感知 >> 推理"的 20+ 分鸿沟，并给出 441K 训练集让开源 7B 模型 GRPO 后媲美 GPT-4o。

**[OMHBench: Benchmarking Balanced and Grounded Omni-Modal Multi-Hop Reasoning](multimodal_vlm/omhbench_benchmarking_balanced_and_grounded_omni-modal_multi-hop_reasoning.md)**

:   OMHBench 构造了一个覆盖文本、图像、语音三种上下文的 6,144 题全模态三跳推理 benchmark，通过实体-属性链和 6 种均衡推理路径暴露出当前 MLLM 在语音落地、路径鲁棒性和跨模态 grounding 上的系统性短板。

**[OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](multimodal_vlm/omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)**

:   本文提出 OMIBench——首个面向奥赛级多图推理的大规模基准，涵盖生物、化学、数学、物理四学科超 1000 道竞赛题，发现即使最强 LVLM（Gemini-3-Pro）也仅达约 50% 准确率，比单图基准下降超 25%。

**[Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](multimodal_vlm/position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)**

:   本文是一篇立场论文（position paper），主张多模态大语言模型（MLLM）可以显著推进跨学科科学推理，提出了四阶段研究路线图（广泛知识识别→类比推理泛化→洞察性推理→创造性假设生成），系统综述了 MLLM 在数学、物理、化学和生物四个领域的应用现状、五大挑战和八个未来方向。

**[PRISM: Self-Pruning Intrinsic Selection Method for Training-Free Multimodal Data Selection](multimodal_vlm/prism_self-pruning_intrinsic_selection_method_for_training-free_multimodal_data_.md)**

:   PRISM 发现 MLLM 视觉特征的非零均值会造成 Global Semantic Drift，从而污染基于相似度的数据选择，并用训练免的均值重中心化和低相关样本筛选，在只保留约 30% 视觉样本的情况下达到 101.7% 相对性能，同时把端到端 GPU 时间降低约 70%。

**[PROGRESSLM: Towards Progress Reasoning in Vision-Language Models](multimodal_vlm/progresslm_towards_progress_reasoning_in_vision-language_models.md)**

:   本文把“从单帧观察判断任务完成到哪一步”定义为 VLM 的进度推理能力，构建 Progress-Bench 和 ProgressLM-45K，并证明显式学习“情景检索 + 心理模拟”比单纯提示推理更稳定。

**[Reducing Peak Memory Usage for Modern Multimodal Large Language Model Pipelines](multimodal_vlm/reducing_peak_memory_usage_for_modern_multimodal_large_language_model_pipelines.md)**

:   论文把多模态大模型的显存瓶颈从“解码阶段的长上下文缓存”前移到“prefill 阶段的视觉 token 峰值缓存”，提出在 prefill 过程中边计算边压缩的结构感知 KV-cache 框架，在固定缓存预算下把峰值显存控制住，同时尽量保留图像和视频理解能力。

**[ReGATE: Learning Faster and Better with Fewer Tokens in MLLMs](multimodal_vlm/regate_learning_faster_and_better_with_fewer_tokens_in_mllms.md)**

:   ReGATE 用冻结的 text-only teacher 估计哪些输出 token 需要视觉信息，再结合 student 的历史学习难度动态选择训练 token，让 MLLM 在不改架构、不加参数的情况下用更少 token 更快训练，并在多个图像和视频 benchmark 上达到或超过标准微调。

**[Region-R1: Reinforcing Query-Side Region Cropping for Multi-Modal Re-Ranking](multimodal_vlm/region-r1_reinforcing_query-side_region_cropping_for_multi-modal_re-ranking.md)**

:   本文提出 Region-R1，将多模态重排序中的查询图像区域裁剪建模为决策问题，通过强化学习（r-GRPO）学习何时以及如何裁剪查询图像中与问题相关的区域，在 E-VQA 和 InfoSeek 上将 CondRecall@1 分别提升 20% 和 8%。

**[Revisit What You See: Revealing Visual Semantics in Vision Tokens to Guide LVLM Decoding](multimodal_vlm/revisit_what_you_see_revealing_visual_semantics_in_vision_tokens_to_guide_lvlm_d.md)**

:   ReVisiT 发现 LVLM 的视觉 token 本身已经编码了可解释的对象语义，并通过上下文约束词表、视觉 token 选择与 logit 融合，在不训练、不额外前向的情况下提升视觉 grounding 并降低幻觉。

**[SciMDR: Advancing Scientific Multimodal Document Reasoning](multimodal_vlm/scimdr_advancing_scientific_multimodal_document_reasoning.md)**

:   SciMDR 提出 synthesize-and-reground 数据构造框架，先在原子 claim 上合成可信 QA 与推理链，再把它们重嵌入完整科学论文中训练模型，使 7B VLM 在科学多模态文档推理上接近 GPT-5 系列表现。

**[ShredBench: Evaluating the Semantic Reasoning Capabilities of Multimodal LLMs in Document Reconstruction](multimodal_vlm/shredbench_evaluating_the_semantic_reasoning_capabilities_of_multimodal_llms_in_.md)**

:   ShredBench 构造了一个“把文档撕碎后让多模态大模型复原内容”的评测基准，证明当前 MLLM 即使在常规 OCR 上很强，也普遍缺乏把视觉碎片、阅读顺序和语义上下文合在一起推理的能力。

**[SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](multimodal_vlm/slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)**

:   提出 SlideAgent，一个层次化智能体框架，通过全局、页面、元素三级专用 agent 构建结构化知识表示，显著提升多页视觉文档（尤其是幻灯片）的细粒度理解能力。

**[Spotlight and Shadow: Attention-Guided Dual-Anchor Introspective Decoding for MLLM Hallucination Mitigation](multimodal_vlm/spotlight_and_shadow_attention-guided_dual-anchor_introspective_decoding_for_mll.md)**

:   提出 DaID (Dual-Anchor Introspective Decoding)，通过挖掘 MLLM 内部不同层的视觉感知差异——Spotlight 层放大视觉信号、Shadow 层抑制语言惯性——在单次前向传播内实现幻觉缓解。

**[Stability Implies Redundancy: Delta Attention Selective Halting for Efficient Long-Context Prefilling](multimodal_vlm/stability_implies_redundancy_delta_attention_selective_halting_for_efficient_lon.md)**

:   提出 DASH（Delta Attention Selective Halting），一种无需训练的推理加速方法，通过监测自注意力层的逐层更新幅度 Δ_attn 来识别已"语义固化"的 token 并停止其后续计算，在长上下文文本和视觉-语言基准上实现显著的 prefill 加速且几乎不损失精度。

**[STELLA: A Multimodal LLM for Protein Functional Annotation via Unified Sequence-Structure Encoding](multimodal_vlm/stella_a_multimodal_llm_for_protein_functional_annotation_via_unified_sequence-s.md)**

:   STELLA 将 ESM3 的统一序列-结构蛋白表示接入 Llama-3.1-8B-Instruct，通过两阶段多模态指令调优完成蛋白功能描述和酶催化反应预测，并在 OPI-Struc 系列基准上刷新多项功能注释指标。

**[StructBreak: Structural Cognitive Overload-Induced Safety Failures in MLLMs](multimodal_vlm/structbreak_structural_cognitive_overload-induced_safety_failures_in_mllms.md)**

:   StructBreak 提出"结构认知过载"（SCO）攻击范式，利用视觉知识图谱（VKG）的拓扑复杂性诱发多模态 LLM 的安全失效——在黑盒设置下对 6 个前沿 MLLM 实现平均 92% 的攻击成功率（Gemini 2.5 高达 97%），并从注意力耗散、隐空间拓扑和几何分析三个层面揭示安全崩塌机制。

**[Structured and Abstractive Reasoning on Multi-modal Relational Knowledge Images](multimodal_vlm/structured_and_abstractive_reasoning_on_multi-modal_relational_knowledge_images.md)**

:   这篇论文提出面向多模态关系知识图像的 STAR 数据引擎和两阶段训练框架，用 STAR-64K 合成数据、CoT 标注与知识感知 KGRPO 显著提升 MLLM 对抽象结构化知识图像的理解和推理能力。

**[TableVista: Benchmarking Multimodal Table Reasoning under Visual and Structural Complexity](multimodal_vlm/tablevista_benchmarking_multimodal_table_reasoning_under_visual_and_structural_c.md)**

:   TableVista 构建了一个 3,000 道高质量表格推理题、扩展为 30,000 个视觉样本的多模态表格 benchmark，系统评测 29 个基础模型后发现：模型对风格变化相对稳定，但在复杂结构、跨表推理、视觉碎片化和纯视觉输入下明显退化。

**[TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](multimodal_vlm/tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)**

:   本文提出 TEMA（Text-oriented Entity Mapping Architecture），首个面向多修改文本的组合图像检索（CIR）框架，通过 MMT 解析助手（PA）增强修改实体覆盖、实体映射模块（EM）解决子句-实体对齐问题，并构建了 M-FashionIQ 和 M-CIRR 两个多修改基准数据集，在原始和多修改场景中均取得最优性能。

**[TeXOCR: Advancing Document OCR Models for Compilable Page-to-LaTeX Reconstruction](multimodal_vlm/texocr_advancing_document_ocr_models_for_compilable_page-to-latex_reconstruction.md)**

:   这篇论文把科学 PDF OCR 从“转成文本/Markdown”推进到“重建可零人工编译的页面级 LaTeX”，提出 TEXOCR-Bench、TEXOCR-Train 和 SFT+RLVR 两阶段训练，让一个 Qwen3-VL-2B 派生模型在结构一致性、引用有效性和编译成功率上显著优于同规模开源基线。

**[Text-Guided Multi-Scale Frequency Representation Adaptation](multimodal_vlm/text-guided_multi-scale_frequency_representation_adaptation.md)**

:   这篇论文提出 FreqAdapter：先把 CLIP/LLaVA 的视觉和文本嵌入变换到 DCT 频域，再用文本引导的多尺度全局适配与跨模态调制微调视觉频率表示，以约 0.11% 额外参数在图文检索和 VQA 上稳定优于常见 prompt/adapter 方法。

**[Thinking Like a Botanist: Challenging Multimodal Language Models with Intent-Driven Chain-of-Inquiry](multimodal_vlm/thinking_like_a_botanist_challenging_multimodal_language_models_with_intent-driv.md)**

:   本文提出PlantInquiryVQA基准和Chain-of-Inquiry（CoI）框架，包含24,950张植物图像和138,068个问答对，模拟植物学家的适应性诊断提问策略，评估18个MLLM在植物病理诊断中的多步视觉推理能力，发现结构化提问显著提升诊断准确性并减少幻觉，但即使最强模型的临床实用性得分仅0.188。

**[Through the Magnifying Glass: Adaptive Perception Magnification for Hallucination-Free VLM Decoding](multimodal_vlm/through_the_magnifying_glass_adaptive_perception_magnification_for_hallucination.md)**

:   本文提出 Perception Magnifier (PM)，一种视觉解码方法，在每个自回归解码步基于多层注意力迭代识别关键视觉区域并自适应放大，通过提升关键区域的有效分辨率来缓解 VLM 的视觉幻觉，同时保持空间结构完整性和推理能力。

**[Topology-Aware Layer Pruning for Large Vision-Language Models](multimodal_vlm/topology-aware_layer_pruning_for_large_vision-language_models.md)**

:   提出基于拓扑数据分析的层剪枝框架 TopoVLM，将各层隐藏状态建模为点云并通过 zigzag 持久同调量化层间拓扑一致性，自适应保留关键表征转换层、剪除结构冗余层，在 50-60% 稀疏率下显著优于现有剪枝方法。

**[Towards Visually Grounded Multimodal Summarization via Cross-Modal Transformer and Gated Attention](multimodal_vlm/towards_visually_grounded_multimodal_summarization_via_cross-modal_transformer_a.md)**

:   这篇论文提出 SPeCTrA-Sum，把层级对齐的 Deep Visual Processor、门控跨模态注意力和 DPP 蒸馏的图像选择器合在一起，使多模态摘要不仅保持接近 SOTA 的 ROUGE，还能选出更相关且更多样的支撑图像。

**[Tree-of-Evidence: Efficient "System 2" Search for Faithful Multimodal Grounding](multimodal_vlm/tree-of-evidence_efficient_34system_234_search_for_faithful_multimodal_grounding.md)**

:   本文提出 Tree-of-Evidence（ToE），一种推理时离散束搜索算法，将多模态模型的可解释性形式化为在粗粒度证据单元（生命体征时间窗口、放射报告片段）上的离散优化问题，仅用 5 个证据单元即可保留全输入模型 98% 以上的 AUROC，同时生成可审计的证据追踪路径。

**[UniversalRAG: 多模态语料库的检索增强生成](multimodal_vlm/universalrag_retrieval-augmented_generation_over_corpora_of_diverse_modalities_a.md)**

:   UniversalRAG 提出一个通用的任意到任意 RAG 框架，通过模态感知路由和粒度感知检索，动态地从异构多模态语料库（文本、图像、视频，不同粒度）中选择最合适的知识源进行检索和生成，避免统一嵌入空间中的模态间隙问题，在 10 个基准上大幅超越单一模态和统一方法。

**[TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning](multimodal_vlm/unleashing_spatial_reasoning_in_multimodal_large_language_models_via_textual_rep.md)**

:   本文提出 TRACE（Textual Representation of Allocentric Context from Egocentric Video），一种提示方法，引导多模态大语言模型从自我中心视频中生成结构化的文本 allocentric 3D 环境表示——包括元上下文、相机轨迹和实体注册表——作为中间推理步骤来增强空间问答能力，在 VSI-Bench 和 OST-Bench 上一致超越已有提示策略。

**[VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation](multimodal_vlm/vauq_vision-aware_uncertainty_quantification_for_lvlm_self-evaluation.md)**

:   本文提出 VAUQ，用图像信息分数和注意力驱动的核心区域遮蔽来衡量 LVLM 的回答是否真正依赖视觉证据，从而在无需训练和外部评估器的情况下更可靠地做多模态自评估与幻觉检测。

**[VIGNETTE: Socially Grounded Bias Evaluation for Vision-Language Models](multimodal_vlm/vignette_socially_grounded_bias_evaluation_for_vision-language_models.md)**

:   VIGNETTE 构建了一个 30M+ 合成配对图像的 VQA 偏见评估基准，用事实性、感知、刻板印象和决策四类问题揭示 VLM 会把身份线索、活动语境和社会等级联系起来，产生细粒度且有时相互矛盾的偏见。

**[Vision-Language Models Mistake Head Orientation for Gaze Direction: Nonverbal Conversation Cues](multimodal_vlm/vision-language_models_mistake_head_orientation_for_gaze_direction_nonverbal_con.md)**

:   这篇论文用 1,360 张受控真实照片和预注册统计检验发现，当前 VLM 在判断人看向哪个物体时远弱于人类，主要会把头部朝向误当成视线方向；对专门 gaze 模型微调后可缓解但不能完全消除这种偏差。

**[VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning](multimodal_vlm/vl-calibration_decoupled_confidence_calibration_for_large_vision-language_models.md)**

:   VL-Calibration 将 LVLM 的口头置信度拆成视觉置信度和推理置信度，并用图像扰动 KL、token 熵与 token 级优势重加权训练模型，在 13 个视觉推理基准上同时降低 ECE、提升准确率。

**[Vocabulary Hijacking in LVLMs: Unveiling Critical Attention Heads by Excluding Inert Tokens to Mitigate Hallucination](multimodal_vlm/vocabulary_hijacking_in_lvlms_unveiling_critical_attention_heads_by_excluding_in.md)**

:   本文发现 LVLM 中部分无效视觉 token 会稳定解码到一组无关词并劫持注意力，进而提出 HABI 定位这些 token、用 NHAR 找到可靠视觉头，再通过 HAVAE 在推理时增强这些头以降低幻觉。

**[VULCA-Bench: A Multicultural Vision-Language Benchmark for Evaluating Cultural Understanding](multimodal_vlm/vulca-bench_a_multicultural_vision-language_benchmark_for_evaluating_cultural_un.md)**

:   VULCA-Bench 用 8 个文化传统、7,410 组图像-双语专家评论和 L1-L5 五层文化理解框架，把 VLM 评测从“看见物体”推进到“理解符号、历史和审美哲学”，并显示现有模型在高层文化推理上普遍掉点 31-40 个百分点。

**[What's Missing in Screen-to-Action? Towards a UI-in-the-Loop Paradigm for Multimodal GUI Reasoning](multimodal_vlm/what39s_missing_in_screen-to-action_towards_a_ui-in-the-loop_paradigm_for_multim.md)**

:   本文提出 UILoop（UI-in-the-Loop）范式，将 GUI 推理从传统的"屏幕→动作"重构为"屏幕→UI 元素→动作"的循环过程，通过 UI 元素驱动的强化微调教模型显式地定位、理解和利用关键 UI 元素，在 GUI 推理任务上达到 SOTA 性能。

**[What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](multimodal_vlm/what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)**

:   本文通过线性探测发现 VLM 的隐藏表示中编码了丰富的多层次美学属性信息（光照、色彩、构图等），并传播到语言解码器层，基于此提出用简单线性回归实现无需微调的个性化图像美学评估（PIAA），效果显著优于 few-shot 和 LoRA 微调基线。

**[When Seeing Overrides Knowing: Disentangling Knowledge Conflicts in Vision-Language Models](multimodal_vlm/when_seeing_overrides_knowing_disentangling_knowledge_conflicts_in_vision-langua.md)**

:   这篇论文构造 WHOOPS-AHA! 让 VLM 的常识知识与图像反事实证据正面冲突，并发现少数晚层注意力头能因果控制模型依赖内部知识还是视觉输入。

**[When Slower Isn't Truer: Inverse Scaling Law of Truthfulness in Multimodal Reasoning](multimodal_vlm/when_slower_isn39t_truer_inverse_scaling_law_of_truthfulness_in_multimodal_reaso.md)**

:   本文发现多模态推理模型的"逆缩放定律"——慢思考（reasoning）模型在面对误导性视觉输入时比快思考（chat）模型更容易产生不真实输出，并构建了 TruthfulVQA 基准（5000+ 样本、50 名标注员、三层分级提示）和 TruthfulJudge 评估模型（88.4% 准确率）来系统诊断这一现象。

**[WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering](multimodal_vlm/wikiseeker_rethinking_the_role_of_vision-language_models_in_knowledge-based_visu.md)**

:   提出 WikiSeeker，重新定义 VLM 在多模态 RAG 中的角色——从单纯的答案生成器转变为两个专门化智能体（Refiner 用 RL 训练重写查询、Inspector 验证检索上下文是否可靠），在 EVQA、InfoSeek、M2KR 三个基准上实现 SOTA。

---

## 📊 LLM 评测 { #llm_evaluation }

**[AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking](llm_evaluation/agenteval_dag-structured_step-level_evaluation_for_agentic_workflows_with_error_.md)**

:   AgentEval 把 agent 执行轨迹建模成「评估 DAG」，对每个节点用 GPT-4o 判官按 5 类节点类型打分并按贪心父节点策略追溯根因，配合 21 类失败类目与 CI/CD 集成；相对端到端评估在 450 条生产 trace 上 failure detection recall 提升 2.17×（0.41→0.89），人类一致性 $\kappa=0.84$，根因准确率 72%（接近人类上限 81%），4 个月试点把根因定位中位数时间从 4.2 小时缩到 22 分钟。

**[Aggregate vs. Personalized Judges in Business Idea Evaluation: Evidence from Expert Disagreement](llm_evaluation/aggregate_vs_personalized_judges_in_business_idea_evaluation_evidence_from_exper.md)**

:   针对商业 idea 评估中专家系统性分歧的现实，构建了 3000 个个体专家评分的 PBIG-DATA 数据集，并实证证明在该领域"个性化 judge（条件于目标评审员历史）"比"汇总 judge（条件于多评审员混合历史）"更贴合专家行为，挑战了"用 pooled labels 当唯一 ground truth"的常见假设。

**[AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation](llm_evaluation/aj-bench_benchmarking_agent-as-a-judge_for_environment-aware_evaluation.md)**

:   提出 AJ-Bench，首个系统评估 Agent-as-a-Judge 能力的基准，覆盖搜索、数据系统和 GUI 三个领域共 155 个任务和 516 条标注轨迹，实验表明 Agent-as-a-Judge 比 LLM-as-a-Judge 平均 F1 提升约 13 个百分点。

**[Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues](llm_evaluation/are_they_lovers_or_friends_evaluating_llms39_social_reasoning_in_english_and_kor.md)**

:   本文提出 SCRIPTS 基准，包含 1.1K 英语和韩语电影对话，通过三层概率标签（HIGHLY LIKELY / LESS LIKELY / UNLIKELY）评估 9 个 LLM 的社会关系推理能力，发现模型在英语上准确率仅 75-80%、韩语 58-69%，且 CoT 和思维模型对社会推理几乎无帮助。

**[arXiv2Table: Toward Realistic Benchmarking and Evaluation for LLM-Based Literature-Review Table Generation](llm_evaluation/arxiv2table_toward_realistic_benchmarking_and_evaluation_for_llm-based_literatur.md)**

:   提出 arXiv2Table 基准（1,957 张表、7,158 篇论文），通过引入干扰论文、模式无关的用户需求和基于 QA 的无标注评估框架，实现更真实的 LLM 文献综述表格生成评估，并提出迭代批处理生成方法。

**[Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](llm_evaluation/attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)**

:   本文系统综述了 134 篇关于 LLM 证据基础文本生成的论文，首次提出统一分类学（归因方式 × 引用特征 × 任务），分析了 300 个评估指标并归纳为七大维度六种方法，为该碎片化领域提供了全景式参考框架。

**[BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks](llm_evaluation/benchmarker_an_education-inspired_toolkit_for_highlighting_flaws_in_multiple-cho.md)**

:   本文借鉴教育学界对多选题（MCQ）的成熟质检框架，构造 BenchMarker 工具用 LLM 判官从「污染（contamination）+ 捷径（shortcuts）+ 写作错误（writing errors）」三个维度审计 12 个主流 NLP MCQA benchmark，发现 TruthfulQA 47% 题目能直接在网上搜到、HellaSwag 100% 违反多条写作规则，并实证证明这些缺陷会显著拉高/压低 LLM 准确率甚至改变模型排名。

**[Beyond Fixed Psychological Personas: State Beats Trait, but Language Models are State-Blind](llm_evaluation/beyond_fixed_psychological_personas_state_beats_trait_but_language_models_are_st.md)**

:   构建覆盖 1667 用户 × 多 subreddit 上下文的 Chameleon 心理画像数据集，用 ICC 分解证明 72-74% 心理变异来自「状态（情境）」而非「特质（人格）」，进一步揭示 LLM 几乎对状态视而不见、reward model 反应到状态却方向相反——RLHF 因此盲目继承了奖励模型的状态偏好。

**[Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](llm_evaluation/beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)**

:   提出 TravelBench，首个融合真实用户查询、隐式用户偏好、多轮交互、不可解任务识别和10种真实工具的旅行规划基准，通过沙箱环境实现可复现评估，揭示前沿模型在不同能力维度上表现不均衡。

**[Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs](llm_evaluation/beyond_marginal_distributions_a_framework_to_evaluate_the_representativeness_of_.md)**

:   本文提出了一种超越边际分布的 LLM 代表性评估框架，通过同时考察边际响应分布和跨问题相关结构来评估人口统计对齐模型，发现虽然微调和 persona prompting 能改善边际分布的近似度，但两者都无法忠实再现人类价值观调查中的多变量相关模式。

**[Beyond Reproduction: A Paired-Task Framework for Assessing LLM Comprehension and Creativity in Literary Translation](llm_evaluation/beyond_reproduction_a_paired-task_framework_for_assessing_llm_comprehension_and_.md)**

:   提出配对任务框架联合评估 LLM 的文学文本理解能力和翻译创造力，基于 11 本英文经典小说对 23 个模型进行大规模测评，发现强理解力并不能转化为人类水平的翻译创造力。

**[Beyond Static Benchmarks: Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation](llm_evaluation/beyond_static_benchmarks_synthesizing_harmful_content_via_persona-based_simulati.md)**

:   作者用"二维 persona"（内在身份 + 外在策略）驱动 LLM agent 在真实 Reddit 帖子上扮演用户写有害评论，合成出比传统静态 benchmark 更难、更多样、覆盖更广的有害内容评测集，对四类主流安全分类器的准确率打到 13–31%（vs. 静态集 60–94%），暴露了现有 benchmark 已被"刷穿"的事实。

**[Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation](llm_evaluation/beyond_the_singular_revealing_the_value_of_multiple_generations_in_benchmark_eva.md)**

:   作者把 LLM benchmarking 形式化为一个**分层贝叶斯估计问题**——prompt 难度 $p_i \sim \mathbb{P}(\mu,\sigma)$，每条 prompt 的 $k$ 次生成正确率服从 Bernoulli$(p_i)$；理论证明用 $k>1$ 次采样能把 within-prompt variance 压到 $\frac{1}{nk}$，并由此衍生出 prompt-level 难度分 $\mathbb{P}(\text{correct})$ 和能检出标注错误的 data map（在 GSM8K 上 44.4% 命中率）。

**[BizCompass: Benchmarking the Reasoning Capabilities of LLMs in Business Knowledge and Applications](llm_evaluation/bizcompass_benchmarking_the_reasoning_capabilities_of_llms_in_business_knowledge.md)**

:   本文提出 BizCompass，一个连接理论基础与实际应用的商业推理基准，覆盖金融/经济/统计/运营四个知识域和分析师/交易员/顾问三个应用角色，系统评估了开源与闭源 LLM 的商业推理能力，揭示理论知识向实际表现转化的规律。

**[Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](llm_evaluation/can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)**

:   本文构建 ProHist-Bench：以中国 1300 年科举史为锚点、由历史学家手写 400 道专家级题目和 10,891 条细粒度 rubric，评测 18 个 SOTA LLM 的专业历史研究能力——结果最强的 Gemini-3-Pro 和 Qwen3-235B 的 Rubric Score 也仅约 28，远低于开卷历史学家。

**[Can We Predict Before Executing Machine Learning Agents?](llm_evaluation/can_we_predict_before_executing_machine_learning_agents.md)**

:   本文证明 LLM 可作为隐式 "world model"，仅凭 task 描述 + verified data report + 两份代码，就能预测 ML 解的优劣（DeepSeek-V3.2-Thinking 61.5% 准确率）；据此构造 ForeAgent，把 AIDE 的 Generate-Execute-Feedback 循环改成 Predict-then-Verify 循环，在 MLE-Bench 上获得 6× 加速 + 3.2× 搜索空间 + +6% Beat Ratio。

**[Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](llm_evaluation/capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)**

:   本文提出了一个三步评估框架（计算特征提取 + LLM-as-Judge + 人类专家验证）来系统评估六种 LLM 在唐诗生成上的能力，发现了关键的"回声室"效应：LLM 系统性地高估模仿统计模式但违反格律规则的机器生成诗歌，与人类专家判断显著偏离。

**[Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](llm_evaluation/challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)**

:   提出 OlymMATH，首个统一自然语言评估和形式化定理证明的奥赛级数学基准，包含350题双语（中英文）题目，涵盖OlymMATH-EASY/HARD（200题数值答案）和OlymMATH-LEAN（150题Lean 4形式化），揭示最强模型在HARD子集上仅58.4%准确率。

**[CLARITY: A Framework and Benchmark for Conversational Language Ambiguity and Unanswerability in Interactive NL2SQL Systems](llm_evaluation/clarity_a_framework_and_benchmark_for_conversational_language_ambiguity_and_unan.md)**

:   CLARITY 是 Oracle 提出的首个支持 "**multi-facet 歧义 + 不可答** + 单/多轮对话 + 多种用户澄清行为" 的 NL2SQL 诊断基准，用 SQL→pivot term→改写→对话→筛查的可控 LLM pipeline 把 Spider/BIRD 自动扩展出 ~3 万实例，并通过 schema 级 pivot/group 标注揭示出 SOTA LLM "能检测歧义却定位不到 schema 元素"的失败模式。

**[Common to Whom? Regional Cultural Commonsense and LLM Bias in India](llm_evaluation/common_to_whom_regional_cultural_commonsense_and_llm_bias_in_india.md)**

:   本文构建 Indica，首个评估 LLM 次国家级文化常识的基准，聚焦印度五大区域在八个日常生活领域的文化差异，发现仅 39.4% 的问题在全部五个区域达成共识，且所有 LLM 均表现出地理偏见——过度选择中部和北部印度作为"默认"文化代表。

**[Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](llm_evaluation/comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)**

:   针对长文本生成中"遗漏关键信息"难以量化的问题，作者提出三种 comprehensiveness 度量——NLI 分解 + 图分析、QA 对比、端到端 LLM 直接识别——以参考语料 $\mathcal{C}$ 为基准计算覆盖率 $S = |\mathcal{A}_{in}| / (|\mathcal{A}_{in}| + |\mathcal{A}_{out}|)$；在 WikiContradict / ConflictBank 上 meta-evaluation 发现最简单的 E2E 方法平均最强（最佳 LMR=0.85），但 Q&A 鲁棒性更好（跨模型 std 仅 0.009 vs E2E 的 0.044），三者各有适用场景。

**[Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](llm_evaluation/contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)**

:   本文揭示了LLM评判器在直接评估任务中存在**分数范围偏差**（score range bias），即模型输出对预定义分数范围高度敏感，并提出利用**对比解码**（contrastive decoding）方法，通过同一模型家族内相似偏差的相互抵消来缓解该问题，在Spearman相关性上平均实现高达11.3%的相对提升。

**[CUB: Benchmarking Context Utilisation Techniques for Language Models](llm_evaluation/cub_benchmarking_context_utilisation_techniques_for_language_models.md)**

:   作者把 7 类主流"上下文利用调控技术"（CMTs）放到统一基准 CUB 上，覆盖 3 个数据集（CounterFact / NQ / DRUID）× 3 类上下文（gold / conflicting / irrelevant）× 11 个 LLM 共 ~800 个实验点，证明所有现有 CMT 都存在"对相关上下文敏感 vs 对无关上下文鲁棒"的根本权衡，且在合成数据上效果普遍被高估。

**[DiningBench: A Hierarchical Multi-view Benchmark for Perception and Reasoning in the Dietary Domain](llm_evaluation/diningbench_a_hierarchical_multi-view_benchmark_for_perception_and_reasoning_in_.md)**

:   作者构建了首个层级化的多视图食品 benchmark DiningBench（3,021 道菜 / 15,928 张图 / 平均 5.27 视图/菜），覆盖「细粒度分类（同店硬负样本）→ 营养估计（4 维回归）→ VQA（推理）」三层认知任务，对 29 个 SOTA VLM 系统评测后发现：现有模型在精细视觉判别和营养量化上严重不足，且 CoT 反而损害纯视觉感知。

**[Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff](llm_evaluation/do_llms_overthink_basic_math_reasoning_benchmarking_the_accuracy-efficiency_trad.md)**

:   本文提出 LLMThinkBench，一个系统性评估 LLM 基础数学推理效率的基准，引入 Overthinking Score（准确率和 token 效率的调和平均），通过动态生成的 14 个确定性数学任务评估 53 个 LLM，发现推理模型平均生成约 18× 更多 token 但有时准确率更低，且扩展推理预算呈现收益递减。

**[Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](llm_evaluation/dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)**

:   DIA 是一种无需训练的扩散大语言模型格式约束生成方法，通过先预测结束锚点位置再在锚点间迭代填充，显著提升 reasoning template 和 JSON 输出的格式正确率，并缓解固定锚点导致的截断或冗余。

**[E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](llm_evaluation/e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)**

:   本文提出 E2EDev，一个基于行为驱动开发（BDD）原则的端到端软件开发基准，包含 46 个真实 Web 项目、244 条细粒度需求和 703 个可执行 BDD 测试，评估发现即使最强 LLM（Claude 系列）在需求准确率上也不超过 60%，多智能体框架的复杂交互成本与性能收益不成正比。

**[EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](llm_evaluation/engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)**

:   提出 EngiBench——首个面向真实工程问题求解的多层级 LLM 评测基准，把任务按三档难度 (基础知识检索 → 上下文推理 → 开放式建模) 组织、配以三种受控变体 (扰动 / 知识增强 / 数学抽象)，覆盖三大工程子领域 1,760 题，发现即使 GPT-4.1 / Claude 3.7 Sonnet 在 Level 3 开放式工程任务上也明显落后于人类专家。

**[Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks](llm_evaluation/enhancing_linguistic_competence_of_language_models_through_pre-training_with_lan.md)**

:   L2T 提出了一种预训练框架，将 14 种语言学习任务（字符级→篇章级）与标准 next-token prediction 混合训练，在 500M/1B 参数规模上将 BLiMP 语言能力得分提升 2-3 个百分点并加速其习得过程，同时保持通用推理性能。

**[Evaluating Legal Reasoning Traces with Legal Issue Tree Rubrics](llm_evaluation/evaluating_legal_reasoning_traces_with_legal_issue_tree_rubrics.md)**

:   LEGIT 把韩国民事/行政判决书自动抽成「层级化的争议点树」当 rubric，让 LLM-as-judge 既能评 "issue coverage" 又能评 "issue correctness"，并据此发现 RAG 与 RL 在法律推理上有互补效应：RAG 全面提升、RL 牺牲覆盖换正确性。

**[Evaluating Memory Capability in Continuous Lifelog Scenario](llm_evaluation/evaluating_memory_capability_in_continuous_lifelog_scenario.md)**

:   本文提出LifeDialBench，一个评估连续生活日志场景下记忆能力的基准（含7天真实数据的EgoMem和1年模拟的LifeMem），引入在线评估协议确保时间因果性，反直觉地发现简单RAG基线一致优于复杂记忆系统。

**[Evaluating Reasoning Models for Queries with Presuppositions](llm_evaluation/evaluating_reasoning_models_for_queries_with_presuppositions.md)**

:   本文构建 ≈13K 跨健康/科学/常识的虚实声明 + 5 档预设强度查询，测评 6 个主流模型（GPT-OSS / Qwen3 / GPT-5 Mini / Gemini 2.5）的开/关推理两档表现，发现推理只带来 2-11% 的轻微准确率提升，但**会让模型变得更"决断"——错的更自信**，对 26-42% 的虚假声明仍然顺从。

**[Evaluating Temporal Consistency in Multi-Turn Language Models](llm_evaluation/evaluating_temporal_consistency_in_multi-turn_language_models.md)**

:   本文提出 ChronoScope，一个基于 Wikidata 自动合成的 146 万条多轮问答 chain 评测集，专门用来测 LLM 能否在多轮交互中"维持先前对话隐含的时间作用域"，发现包括 GPT-4 / Gemini-2.5 在内的强模型都会系统性地"漂移到现在"（present-day drift），且交互越长越严重，即便给 oracle 上文也无法消除。

**[Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language](llm_evaluation/exploring_the_capability_boundaries_of_llms_in_mastering_of_chinese_chouxiang_la.md)**

:   本文将中文互联网亚文化语言"抽象话"引入 NLP 社区，构建首个评估基准 Mouse（含翻译、表征分类、意图识别、毒性检测、含义选择、完形填空六个任务），发现 SOTA LLM 在上下文语义理解上表现尚可但在其他任务上存在明显局限。

**[Fin-Bias: Comprehensive Evaluation for LLM Decision-Making under human bias in Finance Domain](llm_evaluation/fin-bias_comprehensive_evaluation_for_llm_decision-making_under_human_bias_in_fi.md)**

:   Fin-Bias 用 8868 份长篇分析师报告构造了一个"原始 / 去掉评级 / 替换为假评级"三版输入的对照基准，证明 18 个 LLM（含 GPT-5、Claude-4-Sonnet）在金融投资评级时严重"羊群" — 即使是无中生有的 fake rating 也会被 30% 的样本盲从，而用 MPQA 主观词典过滤上下文里的人类观点 + DPO 微调可以把开源 8B 模型反推到比 GPT-5 还准的水准。

**[Finch: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows](llm_evaluation/finch_benchmarking_finance_amp_accounting_across_spreadsheet-centric_enterprise_.md)**

:   本文提出 Finch（FinWorkBench），一个从真实企业环境（Enron 数据集等）构建的金融会计工作流基准，包含 172 个复合工作流和 1,710 个电子表格（2700 万单元格），即使最强的 GPT 5.1 Pro 花费平均 16.8 分钟也仅通过 38.4% 的工作流，揭示了前沿 AI Agent 在真实企业场景中的严重不足。

**[Gated Tree Cross-Attention for Checkpoint-Compatible Syntax Injection in Decoder-Only LLMs](llm_evaluation/gated_tree_cross-attention_for_checkpoint-compatible_syntax_injection_in_decoder.md)**

:   作者给冻结的 decoder-only LLM（Qwen-2.5-7B、Llama-3-8B）外挂一个 Gated Tree Cross-Attention 侧支路 —— 离线 Berkeley parser 预算 constituency 树并按高度索引成 chunk memory，token 隐状态通过 head-wise 门控 cross-attention 读它得到残差更新，再配合 token update mask + 三阶段训练防止干扰；BLiMP 准确率从 78.58/79.95 提升到 83.12/84.61，同时 MCQA、HellaSwag、WinoGrande 完全不退步。

**[How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models](llm_evaluation/how_hypocritical_is_your_llm_judge_listener-speaker_asymmetries_in_the_pragmatic.md)**

:   本文通过三个语用任务（虚假预设、反预设、演绎推理）系统对比 14 个 LLM 作为"语用听者"（判断语用适当性）和"语用说者"（生成语用适当的语言）的表现，发现普遍存在的听者-说者不对称：多数模型作为判断者远优于生成者，且项目级分析表明正确判断不能可靠预测成功生成。

**[HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing](llm_evaluation/howtobench_holistic_evaluation_for_llms_capability_in_human-level_writing_using_.md)**

:   本文指出 LLM-as-a-judge 在长篇开放式写作上有 "negotiation inconsistency" —— 子分聚合不稳定且不可解释，提出 Tree-of-Writing (ToW) 把写作评测显式建模成"内容 / 格式 / 印象"三主节点 + 子叶节点 + 显式 LLM-negotiator 权重的树形流水线，在 1302 条中文 12 体裁的 HoWToBench 上把 system-level Pearson 相关性从 0.85-0.89 拉到 **0.93** 并对常见文本扰动鲁棒。

**[HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns](llm_evaluation/humanllm_benchmarking_and_improving_llm_anthropomorphism_via_human_cognitive_pat.md)**

:   本文提出 HumanLLM 框架，将 244 个心理学模式（100 个人格特质 + 144 个社会认知模式）建模为相互作用的因果力而非孤立标签，构建了 11,359 个包含 2-5 个模式交互的场景和多轮对话数据集，通过双层 checklist 评估实现与人类判断的高对齐（$r=0.90$），HumanLLM-8B 在多模式动态上以 4 倍小的参数量超越 Qwen3-32B。

**[Identifying the Achilles' Heel: An Iterative Method for Dynamically Uncovering Factual Errors in Large Language Models](llm_evaluation/identifying_the_achilles_heel_an_iterative_method_for_dynamically_uncovering_fac.md)**

:   HalluHunter 是一个基于知识图谱的全自动 LLM 事实错误测试框架——用 Wikidata 抽事实三元组、规则化生成 Yes/No、MC、WH 三种问题类型并支持多跳推理，再用"自适应迭代算法"基于上一轮错误回答的实体相似度和关系准确率挑下一批难题，5 轮迭代后能把 9 个主流 LLM 的准确率降低 32-42%，最多触发 55% 题目错误，并显著优于静态 benchmark。

**[Idiom Understanding as a Tool to Measure the Dialect Gap](llm_evaluation/idiom_understanding_as_a_tool_to_measure_the_dialect_gap.md)**

:   提出三个新的法语习语理解基准数据集（魁北克法语 QFrCoRE/QFrCoRT 和标准法语 MFrCoE），在 111 个 LLM 上评估发现 65.77% 的模型在方言习语上表现显著差于标准法语，量化了方言差距现象。

**[IF-Critic: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation](llm_evaluation/if-critic_towards_a_fine-grained_llm_critic_for_instruction-following_evaluation.md)**

:   本文提出 IF-Critic-14B：先用一个 Checklist Generator 把复杂指令拆成约束清单，再让 critic 在**一次推理内**对所有约束逐条给出"解释+0/1 判断"，并通过多阶段过滤的高质量 critique 训练 + 约束级 DPO 进行训练，最终在四个指令跟随评测榜上超过 o4-mini / Gemini-3-Pro，并以约 1/3 的算力让 7B/8B 策略模型在 Multi-IF / CFBench / SysBench 上经 GRPO 训练后追平 32B/70B 同族模型。

**[IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation](llm_evaluation/if-rewardbench_benchmarking_judge_models_for_instruction-following_evaluation.md)**

:   本文提出 IF-RewardBench：第一个同时覆盖单轮 / 多轮 / 系统提示三类指令、由 16 个 LLM 生成响应、由人工严格标注 (Cohen's $\kappa$=0.87) 的 judge 元评测基准；它把传统的 pairwise / BoN 评测范式升级为基于 **Pareto-dominance 偏好图**的 listwise 评测，对 22 个 SOTA judge（含 Gemini-3-Pro / GPT-5.1 / 各类 reward model）一通跑后发现：最强 judge 的 Kendall $\tau_b$ 也只有 0.609（远低于人类 0.755），所有专用 RM 均不超过 0.2，且本榜与下游 BoN 性能的相关性显著高于 RewardBench-2、PPE-IF 等现有 benchmark。

**[Inverting the Shield: Systematically Generating Safety Tests from Policy Specifications](llm_evaluation/inverting_the_shield_systematically_generating_safety_tests_from_policy_specific.md)**

:   POLARIS把自然语言安全策略先编译成一阶逻辑规约，再构造语义策略图并系统遍历生成测试查询，从而让LLM安全评测从启发式红队转向可追踪、可覆盖、可复现的规约驱动测试。

**[K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology](llm_evaluation/k-metbench_a_multi-dimensional_benchmark_for_fine-grained_evaluation_of_expert_r.md)**

:   作者基于韩国国家气象工程师认证考试 25 届真题构造了 1,774 题的 K-MetBench，沿"多模态视觉 / 专家推理 / 地理文化 / 子领域细粒度"四个正交维度评测了 55 个 LLM/MLLM，发现现有模型存在普遍的 modality gap（视觉气象图准确率较纯文本平均掉 18.6%）、reasoning gap（答案对但 rationale 幻觉）、geo-cultural gap（小韩国模型 A.X-4.0 在韩特题上 78.9 反超 235B 的 Qwen3-VL 的 72.6），证明纯参数规模不能解决文化本地化问题。

**[Language Models Don't Know What You Want: Evaluating Personalization in Deep Research Needs Real Users](llm_evaluation/language_models_dont_know_what_you_want_evaluating_personalization_in_deep_resea.md)**

:   作者构建了首个开源个性化 Deep Research 系统 MyScholarQA（profile → action → report 三段式），在 16 项离线指标上几乎全胜其它 DR baseline；但 21 位真实研究员的 90 分钟访谈揭示了 9 类离线评测完全检测不到的个性化失败模式，且四个主流 LLM judge 都预测不准用户满意度，给"用 LLM judge 替代真实用户"敲响了警钟。

**[Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions](llm_evaluation/large_language_models_are_bad_dice_players_llms_struggle_to_generate_random_numb.md)**

:   本文首次大规模系统审计了 11 个前沿 LLM 在 15 种概率分布上的原生采样能力，发现 LLM 严重缺乏内在概率采样机制，且这种缺陷会传导到下游应用中造成系统性偏差。

**[LLMs as annotators of credibility assessment in Danish asylum decisions: evaluating classification performance and errors beyond aggregated metrics](llm_evaluation/llms_as_annotators_of_credibility_assessment_in_danish_asylum_decisions_evaluati.md)**

:   在丹麦难民上诉委员会 (RAB) 的 273 份庇护决定文书上构建了 RAB-Cred 这个三类 ("Absent / Positive / Negative") 可信度评估专家标注数据集，系统跑了 21 个开源 LLM × 30 种 system×user prompt 组合，发现 prompt 设计比模型选择更重要、phi-4 (14B) 在零样本下能拿 94.7% F1 但单模型一致会犯"不可接受"的错，于是建议用"15 个最优模型-prompt 组合多数投票"的 ensemble，正确率能再涨 1.5 pp 到 96%。

**[MARCH: Evaluating the Intersection of Ambiguity Interpretation and Multi-hop Inference](llm_evaluation/march_evaluating_the_intersection_of_ambiguity_interpretation_and_multi-hop_infe.md)**

:   提出 MARCH 基准（2,209 个多跳歧义问题）和 CLARION 框架，首次系统研究歧义解析与多步推理交叉场景下的 QA 挑战，揭示现有 SOTA 模型在此类问题上的严重不足。

**[Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](llm_evaluation/minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)**

:   作者通过"严格的数据质量控制 + SFT + DPO 对齐"三步走，用不到现有工作一半规模的 57K 条高质量评测样本，训练出能同时给 I2T 与 T2I 双向多模态生成任务打分的 8B 评测模型 Minos，在 16 个 out-of-domain 任务上超过所有开源 MLLM-评测模型，并接近 GPT-4o。

**[MM-JudgeBias: A Benchmark for Evaluating Compositional Biases in MLLM-as-a-Judge](llm_evaluation/mm-judgebias_a_benchmark_for_evaluating_compositional_biases_in_mllm-as-a-judge.md)**

:   作者把"MLLM 当 judge 时是否真的把图像、查询、回答三者综合起来评判"形式化为 Compositional Bias，并构建 MM-JudgeBias——一个含 9 类偏见、1804 条来自 29 个源 benchmark 样本的诊断集，用 Bias-Deviation（语义破坏后该降分但没降）+ Bias-Conformity（语义保不变时该稳但不稳）两个互补指标，发现 26 个 SOTA MLLM judge（含 Gemini-3 Pro、GPT-5.1、Claude Opus 4.5）都存在严重的 modality neglect。

**[Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](llm_evaluation/modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)**

:   本文发现 LLM 在联合预测情感-思维风格-立场-意图四个认知维度时准确率暴跌至 5.7%（"认知拥挤"效应），通过 Gromov δ-hyperbolicity 分析证明认知状态具有层次结构，提出 HyCoLLM 框架在双曲空间中建模认知状态，8B 模型超越 GPT-4o。

**[Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](llm_evaluation/multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)**

:   本文提出 MT-RL-Judge，一个多任务强化学习框架，通过 GRPO 联合优化多个评估任务训练统一的 MLLM-as-a-Judge 模型，在文本-图像对齐、安全合规和视觉质量评估等六个基准上一致超越 SFT 基线，并在未见过的 MJ-Bench 配对比较格式上展现出强大的分布外泛化能力（Safety 任务 82.23% vs SFT-Unified 的 49.40%）。

**[MultiFileTest: A Multi-File-Level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms](llm_evaluation/multifiletest_a_multi-file-level_llm_unit_test_generation_benchmark_and_impact_o.md)**

:   提出 MultiFileTest，首个多文件级别 LLM 单元测试生成基准，覆盖 Python/Java/JavaScript 各 20 个项目，评估 11 个前沿 LLM 并分析手动修复和自修复机制对测试质量的影响，揭示即使最强模型也存在大量基础可执行性错误。

**[NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](llm_evaluation/novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)**

:   NovBench 把“论文引言中的新颖性主张”和“审稿人对新颖性的文字评价”配对成 1,684 条基准样本，并用相关性、正确性、覆盖度、清晰度四个维度系统揭示：当前通用 LLM 和审稿专用 LLM 都能写出流畅评价，但仍难以真正理解和完整判断学术新颖性。

**[Personalized Benchmarking: Evaluating LLMs by Individual Preferences](llm_evaluation/personalized_benchmarking_evaluating_llms_by_individual_preferences.md)**

:   本文对 Chatbot Arena 的 115 名活跃用户进行个性化排名分析，发现 Bradley-Terry 个性化排名与全局排名的平均 Spearman 相关仅 ρ=0.04（57% 用户近零或负相关），证明聚合基准无法反映大多数用户的个体偏好，并通过话题+风格特征成功预测了用户特定的模型排名。

**[PolicyLLM: Towards Excellent Comprehension of Public Policy for Large Language Models](llm_evaluation/policyllm_towards_excellent_comprehension_of_public_policy_for_large_language_mo.md)**

:   本文提出 PolicyBench（21K 题的中美跨体制政策理解基准）和 PolicyMoE（基于认知层次的混合专家模型），系统评估 11 个 SOTA LLM 在政策记忆/理解/应用三层次上的能力，发现模型在结构化推理上表现好但在抽象政策概念上仍然薄弱。

**[PolitNuggets: Benchmarking Agentic Discovery of Long-Tail Political Facts](llm_evaluation/politnuggets_benchmarking_agentic_discovery_of_long-tail_political_facts.md)**

:   PolitNuggets 提出一个面向 400 位全球政治人物、超过 1 万条政治履历事实的多语言 agentic discovery benchmark，并用 FactNet 动态证据验证协议发现：当前 agent 高精度但低召回，真正瓶颈是长尾事实发现、非英语证据和高效工具使用。

**[Pressure-Testing Deception Probes in LLMs: Scaling, Robustness, and the Geometry of Deceptive Representations](llm_evaluation/pressure-testing_deception_probes_in_llms_scaling_robustness_and_the_geometry_of.md)**

:   这篇论文系统压力测试 LLM 内部激活上的欺骗探针，发现干净数据上的近满分 AUROC 并不等于可部署鲁棒性：单方向和熵代理解释都站不住脚，欺骗信号更像分散在多维弱特征中，而风格增强训练能把 27B 模型上接近随机的探针恢复到 0.983 的 held-out style AUROC。

**[Presupposition and Reasoning in Conditionals: A Theory-Based Study of Humans and LLMs](llm_evaluation/presupposition_and_reasoning_in_conditionals_a_theory-based_study_of_humans_and_.md)**

:   这篇论文用基于语言学理论的条件句预设投射任务对比人类和四个 LLM，发现人类会联合使用概率、前件-预设相关性和上下文线索，而 LLM 的评分相似性与理论化推理质量明显脱钩，很多看似贴近人类的判断可能来自表层模式匹配。

**[Question Difficulty Estimation for Large Language Models via Answer Plausibility Scoring](llm_evaluation/question_difficulty_estimation_for_large_language_models_via_answer_plausibility.md)**

:   Q-Daps 通过生成多个候选答案并计算去流行度偏置后的 plausibility 分布熵来估计 LLM 问答难度，在 TriviaQA、NQ、MuSiQue、QASC 上系统优于可读性、检索复杂度、prompt 打分和不确定性基线。

**[Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases](llm_evaluation/reasoning_model_is_superior_llm-judge_yet_suffers_from_biases.md)**

:   论文系统比较 reasoning model 与普通 LLM 作为 judge 的表现，发现推理模型在准确率、评测指令遵循和攻击鲁棒性上更强，但仍容易受表层质量偏差影响，并提出 prompt-only 的 PlanJudge 缓解偏差。

**[ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering](llm_evaluation/recoqa_a_benchmark_for_tool-augmented_and_multi-step_reasoning_in_real_estate_qu.md)**

:   本文构建了 ReCoQA——一个包含 29,270 个房地产问答对的大规模基准，要求模型融合数据库查询和地图 API 调用进行混合多源推理，并提出层次化多 Agent 框架 HIRE-Agent 作为强基线，系统性地揭示了现有 LLM 在垂直领域复杂推理中的瓶颈。

**[ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](llm_evaluation/researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)**

:   提出 ResearchBench，首个大规模评估LLM科学发现能力的基准，基于"灵感驱动假设生成"的理论分解，覆盖12个学科1386篇论文，将科学发现分解为灵感检索、假设组合、假设排序三个充分子任务，发现LLM在跨学科灵感检索上表现出色。

**[Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation](llm_evaluation/rethinking_meeting_effectiveness_a_benchmark_and_framework_for_temporal_fine-gra.md)**

:   本文重新定义会议效率评估——提出"目标达成率/时间成本"的客观标准和时序细粒度评估范式，构建了包含 130 场会议 2,459 个标注片段的 AMI-ME 数据集，并开发了基于 LLM 的自动评估框架，在 Spearman 相关系数上达到 0.64。

**[ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](llm_evaluation/retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)**

:   本文提出 ReTraceQA，首个面向常识推理任务的推理过程评测基准，包含 2421 条由专家标注的步骤级错误定位和错误分类标注，揭示 14-24% 的 SLM 虽给出正确答案但推理过程有误，当采用推理感知评估替代仅答案评估时，SLM 性能最多下降 25 个百分点。

**[Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models](llm_evaluation/revisiting_a_pain_in_the_neck_a_semantic_reasoning_benchmark_for_language_models.md)**

:   这篇论文提出 SEMANTICQA，把习语、词汇搭配、名词复合词和动词多词表达统一到分类、抽取、解释及顺序组合任务中，发现强 LLM 在开放解释上看似不错，但在结构化抽取、细粒度语义分类和级联工作流中仍明显不稳。

**[Revisiting the Reliability of Language Models in Instruction-Following](llm_evaluation/revisiting_the_reliability_of_language_models_in_instruction-following.md)**

:   本文提出 nuance-oriented reliability 和 reliable@k，用 IFEval++ 检验模型能否稳定处理语义相近但细节不同的 cousin prompts，发现即便高分模型在细微提示变化下也会显著掉线。

**[Reward Modeling for Scientific Writing Evaluation](llm_evaluation/reward_modeling_for_scientific_writing_evaluation.md)**

:   本文提出 SciRM 和 SciRM-Ref 两个针对科学写作评估的开源奖励模型，通过两阶段强化学习（GRPO）分别优化评估偏好和推理能力，实现了在多种科学写作任务上的细粒度多方面评估，并能泛化到未见过的评估任务和标准。

**[RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity](llm_evaluation/roleconflictbench_a_benchmark_of_role_conflict_scenarios_for_evaluating_llms39_c.md)**

:   RoleConflictBench 通过构建 13,914 个角色冲突场景，利用情境紧迫性作为客观约束来评估 LLM 的上下文敏感性，揭示了模型决策被静态角色偏好主导而非响应动态情境线索的严重问题。

**[Same Voice, Different Lab: On the Homogenization of Frontier LLM Personalities](llm_evaluation/same_voice_different_lab_on_the_homogenization_of_frontier_llm_personalities.md)**

:   本文用 144 个性格 trait 的外部 ELO 偏好评测发现，九个前沿 LLM 虽来自不同实验室，却普遍收敛到 structured、systematic、precise 等“Assistant-like”人格，而差异主要集中在 poetic、playful 等中位风格特征上。

**[ScaleBox: Enabling High-Fidelity and Scalable Code Verification for Large Language Models](llm_evaluation/scalebox_enabling_high-fidelity_and_scalable_code_verification_for_large_languag.md)**

:   ScaleBox 通过自动 special judge 合成、统一验证流程和分布式细粒度并行，提高 LLM 代码训练与评测中的验证精度和吞吐，并在 LiveCodeBench RLVR 实验中带来更稳定的 Pass@1 提升。

**[SCAN: Structured Capability Assessment and Navigation for LLMs](llm_evaluation/scan_structured_capability_assessment_and_navigation_for_llms.md)**

:   SCAN 将大模型评测从单一排行榜推进到可导航的能力画像：它自动构建层级能力标签、用 RealMix 生成覆盖长尾能力的真实感查询，并用 PC2 judge 提升自动评分可靠性，从而在 21 个主流 LLM 上揭示总分掩盖的细粒度强弱项。

**[SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models](llm_evaluation/scicustom_a_framework_for_custom_evaluation_of_scientific_capabilities_in_large_.md)**

:   SciCustom 将科学评测需求拆成可复用的本体知识单元，并通过 tagger、multi-model voting、binary-search relevance filtering 和 proxy subset selection 自动构建领域定制 benchmark，在 10/11 个化学与医疗子任务上取得最高 Spearman 排名一致性。

**[SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](llm_evaluation/sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)**

:   本文构建 SciImpact——首个跨 19 个学科领域、涵盖 7 个影响力维度（引用、奖项、专利、媒体、代码、数据集、模型）的大规模科学影响力预测基准，包含 215,928 个对比论文对，通过多任务微调使 4B 模型超越 o4-mini 等大模型。

**[SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark](llm_evaluation/sessionintentbench_a_multi-task_inter-session_intention-shift_modeling_benchmark.md)**

:   本文提出 SessionIntentBench，一个评估 L(V)LM 理解电商购物会话中跨步骤意图漂移能力的多任务基准，包含四个递进式子任务（意图购买似然估计、属性正则化、意图验证对比、意图演化建模），构建了 190 万条意图条目和 113 万条意图轨迹，实验表明当前 20+ 个 L(V)LM 在捕获复杂会话意图方面表现不佳。

**[SPENCE: A Syntactic Probe for Detecting Contamination in NL2SQL Benchmarks](llm_evaluation/spence_a_syntactic_probe_for_detecting_contamination_in_nl2sql_benchmarks.md)**

:   SPENCE 通过对 NL2SQL 基准查询进行系统性句法改写并测量执行准确率随句法距离的衰减程度，检测和量化 LLM 在 NL2SQL 基准上的数据污染行为，发现越老的基准（如 Spider）污染信号越强，而较新的 BIRD 基准几乎不受影响。

**[Stability vs. Manipulability: Evaluating Robustness Under Post-Decision Interaction in LLM Judges](llm_evaluation/stability_vs_manipulability_evaluating_robustness_under_post-decision_interactio.md)**

:   本文揭示了 LLM 评估器的关键脆弱性：虽然在重复评估下高度稳定，但在后续对话质询下会产生大幅反转（49% 翻转率，权威框架下 74%），表明稳定性不等于鲁棒性，且置信度无法预测真实可靠性。

**[Statistically Reliable LLM-Based Ranking Evaluation via Prediction-Powered Inference](llm_evaluation/statistically_reliable_llm-based_ranking_evaluation_via_prediction-powered_infer.md)**

:   PRECISE 将 Prediction-Powered Inference (PPI) 扩展到排名评估指标，通过少量人工标注 + 大量 LLM 判断的组合，在纠正 LLM 系统性偏差的同时降低指标估计方差，实现统计可靠的排名系统评估。

**[StratMem-Bench: Evaluating Strategic Memory Use in Virtual Character Conversation Beyond Factual Recall](llm_evaluation/stratmem-bench_evaluating_strategic_memory_use_in_virtual_character_conversation.md)**

:   StratMem-Bench 将虚拟角色对话中的记忆分成 must、nice 和 irr 三类，评估模型是否能在保证事实需求的同时主动加入有益记忆并抑制无关记忆，揭示当前强 LLM 在“支持性记忆选择”上仍明显不稳。

**[Stress Testing Factual Consistency Metrics for Long-Document Summarization](llm_evaluation/stress_testing_factual_consistency_metrics_for_long-document_summarization.md)**

:   这篇论文把六个常用 reference-free factuality metrics 放到长文档摘要中做压力测试，发现它们会被事实保持的改写、检索窗口大小和高信息密度 claim 显著影响，说明短摘要指标不能直接信任地迁移到长文档场景。

**[TabReX: Tabular Referenceless eXplainable Evaluation](llm_evaluation/tabrex_tabular_referenceless_explainable_evaluation.md)**

:   提出 TabReX，一种基于图推理的无参考表格生成评估框架，将源文本和生成表格转化为知识图谱三元组并对齐，计算可解释的属性驱动分数，在人类判断相关性上大幅超越现有方法；同时构建 TabReX-Bench 大规模基准。

**[TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](llm_evaluation/taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)**

:   本文提出 TaxPraBen，首个面向中国税务实践的 LLM 评测基准，包含 14 个数据集共 7.3K 样本，覆盖税务风险防控、稽查分析和税务筹划三大真实场景，并设计了"结构化解析—字段对齐提取—数值与文本匹配"的可扩展评估范式，评测 19 个 LLM 后发现闭源大模型和中文优化模型表现更优，而税务领域微调模型 YaYi2 改进有限。

**[Teaching Language Models to Check Grounded Claim Factuality with Human Test-Taking Strategies](llm_evaluation/teaching_language_models_to_check_grounded_claim_factuality_with_human_test-taki.md)**

:   将接地声明事实性检验重新表述为真/假阅读理解任务，通过融入人类考试答题策略设计结构化提示，使LLM能以最少的推理步骤高效准确地检验声明，同时通过监督微调与直接偏好优化训练小型语言模型替代大模型实现80%以上的推理成本节省。

**[Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation](llm_evaluation/teaching_language_models_to_forecast_research_success_through_comparative_idea_e.md)**

:   本文研究语言模型能否学会预测研究想法的经验成功，通过构建含 11,488 个想法对的数据集（基于 PapersWithCode 客观成果），用 SFT 和 RLVR 训练 8B 模型达到 77.1% 准确率，超过 GPT-5 的 61.1%，成为自动科研发现中的有效想法验证器。

**[The Silent Vote: Improving Zero-Shot LLM Reliability by Aggregating Semantic Neighborhoods](llm_evaluation/the_silent_vote_improving_zero-shot_llm_reliability_by_aggregating_semantic_neig.md)**

:   这篇论文指出零样本 LLM 分类中的 constrained softmax 会丢掉标签同义词附近的概率质量，提出无需训练的 Semantic Softmax，把 top-K 词表 token 的“沉默投票”聚合回目标标签，从而显著降低 ECE 和 Brier Score 并提升 AUROC/F1。

**[VC-Inspector: Advancing Reference-free Evaluation of Video Captions with Factual Analysis](llm_evaluation/vc-inspector_advancing_reference-free_evaluation_of_video_captions_with_factual_.md)**

:   本文提出 VC-Inspector，一个基于开源轻量级多模态模型（Qwen2.5-VL 3B/7B）的无参考视频字幕评估指标，通过可控事实错误合成流水线生成训练数据，在 VATEX-Eval 上达到 $\tau_b$=42.58 的人类判断相关性，超越依赖 GPT-4o 的 G-VEval（$\tau_b$=39.40），且在幻觉检测基准上达到 99.6% 准确率。

**[正确信念的瓦解：临床压力下 LLM 的认知韧性研究](llm_evaluation/when_correct_beliefs_collapse_epistemic_resilience_of_llms_under_clinical_pressu.md)**

:   通过设计多轮对抗压力评估框架 Med-Stress，本文发现高医学知识不能保证 LLM 的信念稳定性，并提出推理时 RBED 和训练时 R-FT 两种防御策略来提升 LLM 在临床对话中的认知韧性。

**[When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias](llm_evaluation/when_vision-language_models_judge_without_seeing_exposing_informativeness_bias.md)**

:   揭示 VLM-as-a-Judge 系统存在严重的"信息量偏见"（informativeness bias）——评判器倾向于选择更详细丰富的回答，即使该回答与图像内容矛盾，提出 BIRCH 范式通过先校正候选答案再进行比较，将偏见减少最高 17%，性能提升最高 9.8%。

**[Zero-shot Large Language Models for Automatic Readability Assessment](llm_evaluation/zero-shot_large_language_models_for_automatic_readability_assessment.md)**

:   本文系统评估 10 个开源 LLM 在 14 个多语言可读性数据集上的零样本 ARA 能力，并提出 LAURAE：用 LLM 的期望值可读性分数与传统可读性公式按 verbal confidence 加权集成，从而在 13/14 个数据集上优于既有无监督方法。

---

## 💡 LLM 推理 { #llm_reasoning }

**[Accurate Legal Reasoning at Scale: Neuro-Symbolic Offloading and Structural Auditability for Robust Legal Adjudication](llm_reasoning/accurate_legal_reasoning_at_scale_neuro-symbolic_offloading_and_structural_audit.md)**

:   本文提出 Amortized Intelligence 范式：把 LLM 当作"一次性编译器"将法律合同编译成名为 DACL 的确定性有向无环图中间表示，运行时由轻量 agent 调度符号引擎执行，在 400 个真实合同事件上达到 99.5% 准确率，相比 GPT-5.2/Claude/Gemini 等推理大模型在复杂合同上准确率从 22-46% 直接跳到 98%，且 token 消耗降低 9.9 倍。

**[Adapt to Thrive! Adaptive Power-Mean Policy Optimization for Improved LLM Reasoning](llm_reasoning/adapt_to_thrive_adaptive_power-mean_policy_optimization_for_improved_llm_reasoni.md)**

:   本文提出 APMPO，用一个由当前奖励均值控制的"幂平均"统一了 GRPO（算术平均）与 GMPO（几何平均）目标，并配合基于奖励稳定度的自适应 clip 范围，使 RLVR 训练能在不同阶段动态切换"放大稀有高奖励"与"强调一致性"的策略，在 9 个数学/SQL/多模态推理基准上稳定超越 GRPO/DAPO/GMPO。

**[AIM-CoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](llm_reasoning/aim-cot_active_information-driven_multimodal_chain-of-thought_for_vision-languag.md)**

:   提出 AIM-CoT 框架，通过信息觅食理论驱动的主动视觉证据选择(AVP)和基于注意力偏移的动态触发机制(DAT)，解决交错模态思维链(I-MCoT)中"看什么"和"何时看"两个核心问题。

**[Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data](llm_reasoning/budget-aware_anytime_reasoning_with_llm-synthesized_preference_data.md)**

:   本文提出了一种预算感知的任意时推理（anytime reasoning）框架和 Anytime Index 指标，用于量化 LLM 在有限 token 预算下的推理质量-效率权衡，并设计了基于 LLM 自合成偏好数据的推理时自改进方法（PDP），在规划、数学和科学 QA 任务上显著提升了中间和最终解的质量。

**[C2: Scalable Rubric-Augmented Reward Modeling from Binary Preferences](llm_reasoning/c2_scalable_rubric-augmented_reward_modeling_from_binary_preferences.md)**

:   针对"自生成 rubric 经常误导 reward model"的两面性问题，作者用 LM 似然 margin 把自采样的 16 条 rubric 自动标注为"helpful / misleading"对，再用 DPO 训一个合作型 rubric generator + GRPO 训一个会先评估 rubric 可信度再下判断的"critical" verifier；只用二元偏好数据，C2 在 4 个偏好基准上比 GRPO 训的 reasoning RM 提升最多 6.5 点 (RM-Bench)，下游 DPO 的 LC win rate 涨 6 点，且 8B 模型靠自生 rubric 就能追平用 4× 大模型 (Qwen3-32B) 提供 rubric 的方案。

**[Calibration-Aware Policy Optimization for Reasoning LLMs](llm_reasoning/calibration-aware_policy_optimization_for_reasoning_llms.md)**

:   作者首先证明 GRPO 类算法的"奖励-only"advantage 估计等价于一个对 AUC 不一致的 surrogate（$\phi(t)=-t$，scale-不变性破坏），导致准确率上升的同时相对校准 (perplexity AUC) 持续退化；据此提出 CAPO：把 advantage 换成基于 logistic AUC consistent surrogate 的"成对、uncertainty-aware"形式，再用 reference-model PPL 做去噪 masking，在 Qwen2.5-Math 1.5B/7B 上实现校准 +15~25%、准确率持平或反超 GRPO，AIME 推理时缩放再涨 5%。

**[Can Reasoning Path still be Effective as Input? Bridging Post-Reasoning to Chain-of-Thought Compression](llm_reasoning/can_reasoning_path_still_be_effective_as_input_bridging_post-reasoning_to_chain-.md)**

:   本文提出 post-reasoning 与 UCoT：先由轻量 compressor 用单次前向生成表示推理路径的 soft tokens，再让 executor 把这些 soft tokens 当作输入上下文进行短输出推理，从而在保持推理准确率的同时显著减少 CoT token 与延迟。

**[Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](llm_reasoning/chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)**

:   本文提出 Alignment Score——一种基于语义熵矩阵的语义级指标，通过比较模型生成的思维链与人类偏好参考链的中间步骤来量化推理对齐度，发现 Alignment Score 与任务准确率、可读性和连贯性高度相关，且 2-hop 推理是对齐的峰值深度。

**[ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs](llm_reasoning/chairo_contextual_hierarchical_analogical_induction_and_reasoning_optimization_f.md)**

:   提出 ChAIRO，一个上下文层次化类比归纳与推理优化框架，通过三阶段 pipeline（类比案例生成→规则归纳→规则注入微调）让 LLM 在内容审核中自主生成类比案例并归纳显式审核规则，比单实例规则生成提升 F1 4.5%，比静态 RAG 提升 2.3%。

**[CoAct: Co-Active LLM Preference Learning with Human-AI Synergy](llm_reasoning/coact_co-active_llm_preference_learning_with_human-ai_synergy.md)**

:   CoAct 在偏好对齐中用自一致性把无标注样本切成"高一致 / 低一致"两堆，再用 k-NN 距离从高一致样本里挑出"自洽但可能错"的隐患样本送给 Oracle 标注，剩下的高一致样本直接当 AI 自标数据，最后用 oracle-verified 样本做 in-context demo 生成新指令，把人和 AI 的监督在一个 DPO 循环里捏成一团，在 GSM8K/MATH/WebInstruct 上比最强基线再涨 4–8 个点。

**[CRISP: Compressing Redundancy in Chain-of-Thought via Intrinsic Saliency Pruning](llm_reasoning/crisp_compressing_redundancy_in_chain-of-thought_via_intrinsic_saliency_pruning.md)**

:   提出 CRISP 框架，发现 `</think>` token 的注意力模式能可靠区分推理链中的关键步骤和冗余步骤，据此设计四种原子操作的贪心搜索压缩流水线，在保持准确率的同时减少50-60%的 token 用量。

**[CSRP: Chain-of-Thought Reasoning for Chinese Text Correction via Reinforcement Learning with Efficiency-Aware Rewards](llm_reasoning/csrp_chain-of-thought_reasoning_for_chinese_text_correction_via_reinforcement_le.md)**

:   CSRP 用 CPT、带 CoT rationale 的 SFT 和带 Efficiency-Aware Reward 的 GRPO 三阶段训练中文文本纠错模型，在 NACGEC 上达到 50.99 $F_{0.5}$、在 CSCD 上达到 59.61 F1，并通过显式奖励编辑效率显著缓解 LLM 纠错中的过纠正问题。

**[Decoupling the Effect of Chain-of-Thought Reasoning: A Human Label Variation Perspective](llm_reasoning/decoupling_the_effect_of_chain-of-thought_reasoning_a_human_label_variation_pers.md)**

:   本文通过 Cross-CoT 实验和逐步分析，揭示了 CoT 推理的"解耦机制"：最终准确率由 CoT 内容决定（99% 方差贡献），但分布排序由模型内在先验主导（>80%），说明长 CoT 是强大的决策器但弱的分布校准器。

**[DELTA: Dynamic Layer-Aware Token Attention for Efficient Long-Context Reasoning](llm_reasoning/delta_dynamic_layer-aware_token_attention_for_efficient_long-context_reasoning.md)**

:   DELTA 是一种 **training-free 的层级稀疏 attention**：把 transformer 分成 "初始全 attention 层 + 少量 Δ-layer 重选 salient pages + 后续稀疏 attention 层" 三组，在 AIME / GPQA-Diamond 上 accuracy 持平或反超 full attention，同时把 attended token 数减少 $4.25\times$、端到端推理加速 $1.54\times$。

**[Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views](llm_reasoning/discovering_a_shared_logical_subspace_steering_llm_logical_reasoning_via_alignme.md)**

:   发现 LLM 内部存在一个共享的逻辑子空间，可同时对齐自然语言和符号逻辑两种推理表示，通过在推理时沿该子空间引导激活可无训练提升逻辑推理准确率最高达 11 个百分点。

**[Dissecting Failure Dynamics in Large Language Model Reasoning](llm_reasoning/dissecting_failure_dynamics_in_large_language_model_reasoning.md)**

:   通过分析 LLM 推理轨迹，发现错误集中在早期的少数关键转折点，错误发生后模型进入"认知螺旋"——局部连贯但全局错误地不断延伸；基于此提出 GUARD 框架，在熵信号检测到的高风险转折点处进行短距分支修复。

**[Distilling Long-CoT Reasoning through Collaborative Step-wise Multi-Teacher Decoding (CoRD)](llm_reasoning/distilling_long-cot_reasoning_through_collaborative_step-wise_multi-teacher_deco.md)**

:   作者提出 CoRD（Collaborative Reasoning Decoding），把多教师 Long-CoT 推理蒸馏从「先生成完整轨迹再后选」改造成「step-wise 协同解码」——每步让多个 LRM 提议候选 step，用 meta-prover 的 predictive perplexity 评分 + beam search 保留 Top-B 部分轨迹，最终 32B 学生在 AIME24/25 上超越所有单教师（79.6 / 70.2 vs 78.9 / 67.9）。

**[Do Not Step Into the Same River Twice: Learning to Reason from Trial and Error](llm_reasoning/do_not_step_into_the_same_river_twice_learning_to_reason_from_trial_and_error.md)**

:   提出 LTE (Learning to reason from Trial and Error)，通过将模型自身生成的错误答案作为提示引导额外 rollout，在不依赖外部专家的情况下有效缓解 RLVR 中的探索停滞问题。

**[Does Self-Consistency Improve the Recall of Encyclopedic Knowledge?](llm_reasoning/does_self-consistency_improve_the_recall_of_encyclopedic_knowledge.md)**

:   作者对 MMLU 按学科应用 Sprague 等的 "=="-启发式把 57 个 subject 拆成 **符号推理** 与 **知识召回** 两个子集（约 1:2），实证证明 self-consistency (SC) 不仅在符号推理上有效——这是 CoT 已经擅长的领域——**在知识召回上也能持续增益**（n=20 时 +2.48），把 GPT-4o 的 MMLU 整体准确率推到 88.93%，并用"多数答案占比"作为置信度信号 (Pearson ρ ≈ 0.42) 给出机制解释。

**[DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models](llm_reasoning/drp_distilled_reasoning_pruning_with_skill-aware_step_decomposition_for_efficien.md)**

:   DRP 让"短 CoT 老师 (GPT-4o)"在"长 CoT 学生 (R1-Distill-Qwen)"自己的推理轨迹上做技能级分步+剪枝/重写，再把这条"裁掉冗余但保留学生说话风格"的轨迹蒸馏回学生，在 GSM8K 把 7B 模型 token 从 917 砍到 328（−64%）的同时把 Pass@1 从 91.7% 提到 94.1%，且在 AIME/AMC/MATH500 等 OOD 任务上同时降 token、涨准确率。

**[DVMap: Fine-Grained Pluralistic Value Alignment via High-Consensus Demographic-Value Mapping](llm_reasoning/dvmap_fine-grained_pluralistic_value_alignment_via_high-consensus_demographic-va.md)**

:   DVMap 把 LLM 的"多元价值对齐"从粗粒度的国家标签下沉到 11 维 demographic 属性档案，先用 Shannon 熵 = 0 的"高一致性档案"过滤出 5.6 万条 WVS 数据，再用 Structured CoT + GRPO（二值奖励）训练 Qwen3-8B，使其在 cross-demographic / cross-country / cross-value 三重泛化测试中超越 DeepSeek-v3.2、并和 GPT-4o 持平。

**[Efficient PRM Training Data Synthesis via Formal Verification](llm_reasoning/efficient_prm_training_data_synthesis_via_formal_verification.md)**

:   本文提出 FoVer，一个利用形式化验证工具（Z3 和 Isabelle）为形式化推理任务的步骤级推理链自动标注正确性标签的框架，生成 FoVer-40K 训练集并微调 PRM，在 12 个推理基准上展示了从形式化到非形式化的迁移能力和跨任务泛化能力。

**[Efficient Process Reward Modeling via Contrastive Mutual Information](llm_reasoning/efficient_process_reward_modeling_via_contrastive_mutual_information.md)**

:   提出 CPMI（Contrastive Pointwise Mutual Information），一种高效的自动步级奖励标注方法，通过对比推理步骤对正确答案和错误答案的条件概率变化量来估计步级贡献，比 Monte Carlo 估计减少 84% 构建时间和 98% token 生成量，同时在过程级评估和数学推理基准上取得更高准确率。

**[Efficient Test-Time Scaling via Temporal Reasoning Aggregation](llm_reasoning/efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)**

:   提出 TRACE 框架，通过在滑动窗口内聚合多步答案一致性和置信度轨迹两种互补信号来判断推理是否收敛，实现无需训练的动态早退，在减少25-30% token 用量的同时准确率仅降1-2%。

**[ETR: Entropy Trend Reward for Efficient Chain-of-Thought Reasoning](llm_reasoning/etr_entropy_trend_reward_for_efficient_chain-of-thought_reasoning.md)**

:   提出 ETR (Entropy Trend Reward)：用动量加权的逐步熵下降作为 reward shaping 项加进 GRPO，让 LLM 的 CoT 自适应地在 "全局熵下降" 约束下提前收敛，相同正确性下平均压缩 CoT 长度 35–65%；在 DeepSeek-R1-Distill-7B 上带来 +9.9% 准确率同时减少 67% token。

**[Evo-Attacker: Memory-Augmented Reinforcement Learning for Long-Horizon Tool Attacks on LLM-MAS](llm_reasoning/evo-attacker_memory-augmented_reinforcement_learning_for_long-horizon_tool_attac.md)**

:   本文提出 Evo-Attacker，把面向 LLM 多智能体系统的工具返回篡改建模为带动态攻击记忆的长程强化学习问题，并用 Attack-Flow GRPO 优化检索、反思和修改决策，在多架构、多任务 benchmark 上显著降低系统成功率。

**[Failure Modes in Multi-Hop QA: The Weakest Link Effect and the Recognition Bottleneck](llm_reasoning/failure_modes_in_multi-hop_qa_the_weakest_link_effect_and_the_recognition_bottle.md)**

:   本文提出 Multi-Focus Attention Instruction (MFAI) 作为语义探针，揭示多跳 QA 中的"最弱链效应"——多跳推理性能由最不可见证据的绝对位置决定而非事实间距离，失败主要源于识别瓶颈而非推理缺陷，且 System-2 推理模型能有效抵御位置偏差和误导性注意力线索。

**[FinReporting: An Agentic Workflow for Localized Reporting of Cross-Jurisdiction Financial Disclosures](llm_reasoning/finreporting_an_agentic_workflow_for_localized_reporting_of_cross-jurisdiction_f.md)**

:   FinReporting 把跨美国、日本、中国财报本地化拆成“规则抽取 + 本体映射 + 受限 LLM 校验/修复 + 人工复核”的可审计 agent workflow，用统一 IS/BS/CF schema 缓解不同司法辖区财务披露格式和会计语义不一致的问题。

**[FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents](llm_reasoning/fs-researcher_test-time_scaling_for_long-horizon_research_tasks_with_file-system.md)**

:   本文提出 FS-Researcher，一个基于文件系统的双 Agent 深度研究框架，通过 Context Builder 构建层次化知识库、Report Writer 分节撰写报告，利用持久化工作空间突破上下文窗口限制，在 DeepResearch Bench 上达到 53.94 RACE（SOTA），并展示了上下文构建计算量与报告质量的正相关测试时扩展效应。

**[GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO](llm_reasoning/ganitllm_difficulty-aware_bengali_mathematical_reasoning_through_curriculum-grpo.md)**

:   本文提出 GanitLLM，首个真正用孟加拉语进行推理（而非翻译或用英语推理）的数学推理模型，构建了难度标注的孟加拉语数学数据集 Ganit，并提出 Curriculum-GRPO 解决低资源语言 GRPO 训练中的冷启动问题，4B 模型在 Bn-MGSM 上提升 8 个准确率百分点，孟加拉语推理 token 从 14% 提升至 88%。

**[HISR: Hindsight Information Modulated Segmental Process Rewards for Multi-turn Agentic Reinforcement Learning](llm_reasoning/hisr_hindsight_information_modulated_segmental_process_rewards_for_multi-turn_ag.md)**

:   HISR 用 GPT-4o 把 agent 轨迹切成与 sub-goal 对齐的 segment，再让一个 hindsight 模型与 policy 模型的似然比给每段算一个 importance 分数，去 modulate 段级过程奖励——在 Alfworld / Virtualhome / Webshop 上把信用分配做得更靠谱，平均得分较 SPA 涨 5+。

**[How Chain-of-Thought Works? Tracing Information Flow from Decoding, Projection, and Activation](llm_reasoning/how_chain-of-thought_works_tracing_information_flow_from_decoding_projection_and.md)**

:   这篇论文从解码、概率投影和 FFN 激活三个层面反向追踪 CoT 的信息流，发现 CoT 可能主要通过约束答案结构、降低预测熵，并按任务类型调节神经元激活来提升推理表现，而不只是让模型“真的更会逻辑推理”。

**[Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization](llm_reasoning/is_chain-of-thought_really_not_explainability_chain-of-thought_can_be_faithful_w.md)**

:   论文系统反驳"CoT 不算可解释性"这一近年流行结论：用 Filler Tokens、FUR、faithful@k 与 Causal Mediation Analysis 四种互补指标证明，被 Biasing Features（hint verbalization）判为不忠实的 CoT 里超过一半其实"以其它方式"忠实地反映了模型推理；不忠实主要来自"自然语言对分布式计算做了 lossy 压缩"导致的不完整（incompleteness），而非真不忠实——增大采样预算可让 hint 出现概率涨到 90%，未明示 hint 的 CoT 也能因果性地传递 hint 影响。

**[JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](llm_reasoning/jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)**

:   JTPRO 提出了一种无需模型微调的联合优化框架，通过反思驱动的迭代编辑同时优化全局指令和逐工具的 schema/参数描述，在大规模工具库场景下显著提升工具选择和参数填充的端到端成功率，相比 GEPA 等基线在 OSR 上提升 5%–20%。

**[Language Model as Planner and Formalizer under Constraints](llm_reasoning/language_model_as_planner_and_formalizer_under_constraints.md)**

:   本文提出 CoPE 基准，通过向经典规划环境注入形式化分类的自然语言约束，揭示出仅一句约束即可将当前最强 LLM 的规划性能减半，暴露了 LLM 规划鲁棒性的严重不足。

**[Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners](llm_reasoning/large_reasoning_models_are_not_yet_multilingual_latent_reasoners.md)**

:   本文系统性地研究了大型推理模型（LRM）在 11 种语言上的潜在推理行为，发现潜在推理能力存在于多语言中但分布不均（高资源语言强、低资源弱），且内部推理动态趋于以英语为中心的共享路径。

**[Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](llm_reasoning/learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)**

:   CoT2Edit 提出通过 CoT 推理教 LLM 进行知识编辑的新范式——构建结构化和非结构化编辑的 CoT 指令数据，经 SFT 冷启动 + GRPO 优化训练，推理时结合 RAG 检索编辑事实，单次训练即在 6 个编辑基准上达到 SOTA 且具有强泛化能力。

**[LegalDrill: Diagnosis-Driven Synthesis for Legal Reasoning in Small Language Models](llm_reasoning/legaldrill_diagnosis-driven_synthesis_for_legal_reasoning_in_small_language_mode.md)**

:   LegalDrill 用 Audit Agent 诊断 0.6B/1.7B 小模型在法律推理上的具体错误模式，让强 teacher（GPT-4o / Qwen3-30B）按错误指令"刻意复现+修正"生成偏好对，再用学生自己的 forced-choice 概率算 Difficulty Score 过滤掉它已会的样本，迭代 SFT+DPO 后 1.7B 学生在 LegalBench 多个子集上逼近 30B teacher。

**[LePREC: Reasoning as Classification over Structured Factors for Assessing Relevance of Legal Issues](llm_reasoning/leprec_reasoning_as_classification_over_structured_factors_for_assessing_relevan.md)**

:   本文提出 LePREC，一种受法律专业人士启发的神经-符号框架，通过 LLM 生成推理问答对将非结构化法律文本转化为结构化特征，再利用稀疏线性模型进行相关性分类，在 769 个马来西亚合同法案例构建的 LIC 数据集上相比 GPT-4o 等 LLM 基线提升 30–40%。

**[LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals](llm_reasoning/llm_reasoning_as_trajectories_step-specific_representation_geometry_and_correctn.md)**

:   本文把 LLM 的 chain-of-thought 推理看成在表征空间里的一条几何轨迹，发现 (a) 每个推理步骤都占据一个线性可分的子空间且越深层越清晰、(b) 正确与错误解在早期重叠、在后期系统性分叉，由此能在尚未输出答案时以 ROC-AUC 0.87 预测最终对错，并据此提出"轨迹引导"做推理修正和长度控制。

**[Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning](llm_reasoning/logical_phase_transitions_understanding_collapse_in_llm_logical_reasoning.md)**

:   本文发现 LLM 逻辑推理存在"逻辑相变"现象——性能在特定复杂度阈值处突然崩塌而非平滑退化，提出逻辑复杂度度量（LoCM）来量化这一现象，并设计神经符号课程调优框架（NSCT），通过自适应神经-符号对齐和复杂度感知课程优化，在五个基准上平均提升 naive prompting +1.26 和 CoT +3.95 准确率。

**[Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning](llm_reasoning/long-context_reasoning_through_proxy-based_chain-of-thought_tuning.md)**

:   ProxyCoT 利用短而充分的 proxy context 先获得高质量推理轨迹，再把这些轨迹蒸馏到完整长上下文输入上，使 4B 模型在 SciTrek、HotpotQA 和 Loong 上显著提升长上下文推理，同时减少推理时 CoT token。

**[MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis](llm_reasoning/mathagent_adversarial_evolution_of_constraint_graphs_for_mathematical_reasoning_.md)**

:   提出基于约束图对抗进化的分层数据合成框架 MathAgent，将数据合成从文本生成任务重构为约束图的无监督优化问题，通过 Legislator 三Agent系统进化问题骨架再由 Executor 实例化为自然语言，仅 1K 合成样本即超越 LIMO 和 s1K 在八个数学基准上的表现。

**[Merlin's Whisper: Enabling Efficient Reasoning in Large Language Models via Black-box Persuasive Prompting](llm_reasoning/merlin39s_whisper_enabling_efficient_reasoning_in_large_language_models_via_blac.md)**

:   Whisper 把大推理模型的“少想但不降准确率”问题建模为黑盒 persuasive prompting，通过多视角自动生成和迭代筛选提示后缀，在 Qwen3、DeepSeek-R1-Distill 以及 Claude/Gemini API 上显著减少输出 token，同时基本保持推理准确率。

**[MTR-Bench: A Comprehensive Benchmark for Multi-Turn Reasoning Evaluation](llm_reasoning/mtr-bench_a_comprehensive_benchmark_for_multi-turn_reasoning_evaluation.md)**

:   MTR-Bench 构建了一个包含 4 类、40 个任务、3600 个实例的自动化多轮推理评测框架，显示当前前沿推理模型在交互式、动态反馈环境中仍远未可靠。

**[On the Step Length Confounding in LLM Reasoning Data Selection](llm_reasoning/on_the_step_length_confounding_in_llm_reasoning_data_selection.md)**

:   本文发现基于自然度的 LLM 推理数据选择方法存在"步长混淆"问题——系统性地偏好每步更长的样本而非更高质量的样本，根因是推理步骤首 token 的低概率被长步骤稀释。提出 Aslec-drop（丢弃首 token 概率）和 Aslec-casl（因果回归去偏）两种校正方法，平均准确率提升 6-9%。

**[Parallel Test-Time Scaling for Latent Reasoning Models](llm_reasoning/parallel_test-time_scaling_for_latent_reasoning_models.md)**

:   本文首次将并行测试时缩放（parallel TTS）引入潜在推理模型，提出两种基于不确定性理论的随机采样策略（MC-Dropout 和加性高斯噪声）以及一个步级对比训练的潜在奖励模型（LatentRM），使得在连续向量空间中进行推理的模型也能通过并行采样+聚合获得稳定的性能提升。

**[PPA-Plan: Proactive Pitfall Avoidance for Reliable Planning in Long-Context LLM Reasoning](llm_reasoning/ppa-plan_proactive_pitfall_avoidance_for_reliable_planning_in_long-context_llm_r.md)**

:   PPA-Plan 在生成长上下文推理计划之前先预测可能的逻辑陷阱，并把这些陷阱转成 negative constraints 约束 planner，从而让 LLM 少走表面关键词匹配和错误假设路径，在多组长文 QA 数据集上提升准确率、NLI 分数并显著降低计划执行失败率。

**[Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards](llm_reasoning/process_reward_models_meet_planning_generating_precise_and_scalable_datasets_for.md)**

:   本文提出利用规划领域定义语言（PDDL）自动生成大规模、高精度的步骤级奖励数据集，用于训练过程奖励模型（PRM），在数学和非数学推理基准上均取得显著提升。

**[Reasoning Fails Where Step Flow Breaks](llm_reasoning/reasoning_fails_where_step_flow_breaks.md)**

:   提出 Step-Saliency 诊断工具发现大推理模型中两种深度相关的信息流失败模式（Shallow Lock-in 和 Deep Decay），并设计 StepFlow 测试时干预方法在不重训练的情况下修复信息传播、提升推理准确率。

**[Reinforced Efficient Reasoning via Semantically Diverse Exploration](llm_reasoning/reinforced_efficient_reasoning_via_semantically_diverse_exploration.md)**

:   ROSE 提出语义熵引导的 MCTS 分支策略和长度感知的段级优势估计，解决了现有 MCTS-based RLVR 方法探索多样性不足和推理效率低的问题，在多个数学推理基准上取得最优 pass@8 性能。

**[Reliability-Aware Adaptive Self-Consistency for Efficient Sampling in LLM Reasoning](llm_reasoning/reliability-aware_adaptive_self-consistency_for_efficient_sampling_in_llm_reason.md)**

:   ReASC 将 adaptive self-consistency 从“数答案票数”改成“判断可靠证据是否足够”，用响应置信度加权 Beta 累积，在 GSM8K、MATH500、Omni-Math 和 GPQA-Diamond 上以接近原准确率显著降低多样本推理成本。

**[Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning](llm_reasoning/render-of-thought_rendering_textual_chain-of-thought_as_images_for_visual_latent.md)**

:   提出 Render-of-Thought（RoT），首次将文本 CoT 推理步骤渲染为图像，利用预训练视觉编码器作为语义锚点将 LLM 隐状态对齐到视觉嵌入空间，实现 3-4 倍 token 压缩和显著推理加速，同时保持推理链的可分析性。

**[ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models](llm_reasoning/reprobe_efficient_test-time_scaling_of_multi-step_reasoning_by_probing_internal_.md)**

:   这篇论文提出 ReProbe，用少于 10M 参数的轻量 transformer probe 读取冻结 LLM 的隐藏状态、注意力和 logits 来判断每一步推理是否可信，在数学、规划和问答任务上接近或超过大 750-810 倍的 PRM，并能作为 Best-of-N 和 beam search 的高效 step verifier。

**[Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](llm_reasoning/revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)**

:   系统性研究了 RLVR 训练中 LLM 的熵动态，揭示正优势 token 是熵崩塌的主要驱动因素，并提出 Positive-Advantage Reweighting 方法通过动态调整正优势 token 的损失权重来有效调控模型熵。

**[Revisiting the Uniform Information Density Hypothesis in LLM Reasoning](llm_reasoning/revisiting_the_uniform_information_density_hypothesis_in_llm_reasoning.md)**

:   本文将心理语言学中的信息密度均匀性（UID）假说引入 LLM 推理分析，提出基于熵的步级信息密度度量框架，发现高质量推理轨迹呈现"局部均匀 + 全局非均匀"的反直觉模式，并证明该模式在 Best-of-N 采样中显著优于传统置信度/熵基线。

**[RSAT: Structured Attribution Makes Small Language Models Faithful Table Reasoners](llm_reasoning/rsat_structured_attribution_makes_small_language_models_faithful_table_reasoners.md)**

:   RSAT 用“结构化引用格式的 SFT + 以 NLI 忠实性为核心奖励的 GRPO”训练 1B-8B 小语言模型，让表格问答不只给答案，还能把每一步推理绑定到具体表格单元格，并把平均忠实性从 SFT 的 0.224 提升到 0.826。

**[Scaling Evaluation-Time Compute with Reasoning Models as Evaluators](llm_reasoning/scaling_evaluation-time_compute_with_reasoning_models_as_evaluators.md)**

:   这篇论文把 test-time scaling 从“生成答案”扩展到“评估答案”，发现让 reasoning model 在评估时生成更多推理 token、逐步检查过程并结合 outcome/process 分数，可以在 ProcessBench 和 Best-of-N 重排序中超过训练好的 PRM/ORM。

**[Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models](llm_reasoning/scaling_test-time_compute_to_achieve_ioi_gold_medal_with_open-weight_models.md)**

:   提出 GenCluster，一个可扩展的测试时计算框架，通过大规模并行生成→行为聚类→锦标赛排名→循环提交策略，首次使开源模型 gpt-oss-120b 在 IOI 2025 上达到金牌水平（446.75/600 分）。

**[SeLaR: Selective Latent Reasoning in Large Language Models](llm_reasoning/selar_selective_latent_reasoning_in_large_language_models.md)**

:   本文提出 SeLaR，一种轻量级无训练框架，通过熵门控机制仅在模型不确定的"探索步"激活软嵌入潜在推理、在高置信的"确定步"保持离散解码，并引入熵感知对比正则化防止软嵌入向主导 token 坍缩，在五个推理基准上一致超越标准 CoT 和 SOTA 无训练方法。

**[Self-Awareness before Action: Mitigating Logical Inertia via Proactive Cognitive Awareness](llm_reasoning/self-awareness_before_action_mitigating_logical_inertia_via_proactive_cognitive_.md)**

:   本文提出 SABA 推理框架，通过"先感知再行动"的范式，在做出最终决策前显式构建和审计知识状态——利用信息融合 (IF) 将叙事整合为可验证的基线状态，再通过查询驱动的结构化推理 (QSR) 递归识别和解决缺失前提——在侦探推理和通用推理基准上均取得最佳表现。

**[Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning](llm_reasoning/self-consistency_from_only_two_samples_cot-pot_ensembling_for_efficient_llm_reas.md)**

:   提出 CoT-PoT 跨模态集成方法，利用链式推理（CoT）和程序化推理（PoT）两种根本不同推理模态的互补性，将自一致性所需的采样次数减少9.3倍，78.6%的问题仅需2个样本即可解决。

**[Self-Reinforcing Controllable Synthesis of Rare Relational Data via Bayesian Calibration](llm_reasoning/self-reinforcing_controllable_synthesis_of_rare_relational_data_via_bayesian_cal.md)**

:   本文提出RDDG，基于渐进式CoT的表格数据合成框架，通过核心集选择、关系挖掘和自强化反馈机制引导LLM生成高保真表格数据，在不平衡分类上平均提升2%+ Macro-F1。

**[Semantic-Aware Logical Reasoning via a Semiotic Framework](llm_reasoning/semantic-aware_logical_reasoning_via_a_semiotic_framework.md)**

:   提出 LogicAgent，一个基于格雷马斯符号方阵(Semiotic Square)的逻辑推理框架，通过多视角语义分析和反思验证，在语义复杂和逻辑复杂双重挑战下实现 SOTA 逻辑推理性能。

**[SHAPE: Stage-aware Hierarchical Advantage via Potential Estimation for LLM Reasoning](llm_reasoning/shape_stage-aware_hierarchical_advantage_via_potential_estimation_for_llm_reason.md)**

:   SHAPE 把 LLM 推理看成在“可解性潜势”状态空间中的轨迹，用长度感知的阶段级优势和熵驱动的 token 级重分配同时提升数学推理准确率并减少约 30% 生成 token。

**[SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks](llm_reasoning/sppo_sequence-level_ppo_for_long-horizon_reasoning_tasks.md)**

:   SPPO 把长链 CoT 推理中的 RLVR 从 token-level MDP 重写为 sequence-level contextual bandit，用只看 prompt 的标量 critic 估计题目可解性，从而用单样本 PPO 获得接近或超过 GRPO 的稳定性与性能，同时带来约 5.9 倍训练加速和更低显存占用。

**[Stabilizing Efficient Reasoning with Step-Level Advantage Selection](llm_reasoning/stabilizing_efficient_reasoning_with_step-level_advantage_selection.md)**

:   这篇论文发现短上下文 GRPO 本身就会强烈压缩推理长度，但会因截断样本的错误信用分配导致训练不稳；作者提出 Step-level Advantage Selection，在推理步骤粒度选择性置零 advantage，在保持甚至提升 Pass@1 的同时显著减少推理 token。

**[Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](llm_reasoning/step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)**

:   提出 Step-GRPO，将动态早退能力内化到模型中——通过语义步骤而非原始 token 来度量推理复杂度，用动态截断 Rollout 暴露简短正确轨迹，配合步骤感知相对奖励引导模型学习在适当时机停止推理，在 Qwen3-8B 上减少32%的 token 消耗且无准确率下降。

**[Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play](llm_reasoning/stratagem_learning_transferable_reasoning_via_trajectory-modulated_game_self-pla.md)**

:   Stratagem 在文本游戏自博弈中不再只按输赢强化模型，而是用“抽象可迁移性”和“推理演化”两个轨迹级信号调制 advantage，使从游戏中学到的策略更能迁移到数学、通用推理和代码生成任务。

**[Strategy-Induct: Task-Level Strategy Induction for Instruction Generation](llm_reasoning/strategy-induct_task-level_strategy_induction_for_instruction_generation.md)**

:   Strategy-Induct 提出一种仅需少量输入问题（无需标注答案）即可归纳任务级指令的框架：先为每个问题生成推理策略，再从策略-问题对中归纳出可复用的任务指令，在 BBH-Induct、Evals-Induct 和 Shift Cipher 三个基准上超越现有 SOTA 方法。

**[TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning](llm_reasoning/templaterl_structured_template-guided_reinforcement_learning_for_llm_reasoning.md)**

:   TemplateRL 通过从小规模种子集合用 MCTS 抽象出结构化推理模板，在强化学习训练中引入这些模板作为显式指导，显著提升 LLM 多步推理效率和稳定性，在 AIME 上相比 GRPO 提升 99%。

**[Think Outside the Policy: In-Context Steered Policy Optimization](llm_reasoning/think_outside_the_policy_in-context_steered_policy_optimization.md)**

:   提出 ICPO (In-Context Steered Policy Optimization)，利用大语言模型自身的上下文学习(ICL)能力作为隐式专家引导，在 RLVR 训练中扩展策略探索空间，无需依赖外部更强模型的推理轨迹。

**[TIME: Temporally Intelligent Meta-Reasoning Engine for Context-Triggered Explicit Reasoning](llm_reasoning/time_temporally_intelligent_meta-reasoning_engine_for_context-triggered_explicit.md)**

:   TIME 把显式推理从“始终开启的长思维链”改造成由时间和语篇线索触发的局部控制策略，通过 `time` 标签、tick 事件、短 `think` 块和四阶段 QLoRA 课程训练，让 Qwen3 系列在 TimeBench 上显著超过 thinking/no-thinking 基线，同时把推理 token 压缩到原来的约十分之一量级。

**[TInR：探索大语言模型中的工具内化推理](llm_reasoning/tinr_exploring_tool-internalized_reasoning_in_large_language_models.md)**

:   本文提出 TInR-U 框架，通过将工具知识内化到 LLM 参数中（而非依赖外部文档），实现高效且可靠的工具辅助推理，在域内和域外测试中均优于现有方法。

**[ToolPRM: Fine-Grained Inference Scaling of Structured Outputs for Function Calling](llm_reasoning/toolprm_fine-grained_inference_scaling_of_structured_outputs_for_function_callin.md)**

:   ToolPRM 将函数调用拆成函数名选择、参数名选择、参数值填写等细粒度决策，训练 intra-call 过程奖励模型来指导 beam search，并提出“explore more but retain less”的结构化输出推理扩展原则，在 BFCL 和 ToolAlpaca 上稳定提升 Hammer2.1 系列工具调用模型。

**[Towards Effective In-context Cross-domain Knowledge Transfer via Domain-invariant-neurons-based Retrieval](llm_reasoning/towards_effective_in-context_cross-domain_knowledge_transfer_via_domain-invarian.md)**

:   本文提出 DIN-Retrieval，通过识别 LLM 中跨域激活极性一致的域不变神经元（DIN），构建域鲁棒的表示子空间用于检索结构兼容的跨域示例，首次证明了使用跨域 ICL 示例提升 LLM 推理性能的可行性，在数学-逻辑推理迁移上平均提升 1.8%。

**[TrigReason: Trigger-Based Collaboration between Small and Large Reasoning Models](llm_reasoning/trigreason_trigger-based_collaboration_between_small_and_large_reasoning_models.md)**

:   TrigReason 提出基于事件触发的大小推理模型协作框架，通过分析小模型三类推理风险（路径偏离、认知过载、恢复失能），设计策略引导、认知卸载和干预请求三种触发器替代逐步轮询验证，在保持 LRM 精度的同时将 1.70-4.79 倍更多推理步骤卸载给小模型，延迟降低 43.9%、API 成本降低 73.3%。

**[Understanding and Mitigating Spurious Signal Amplification in Test-Time Reinforcement Learning for Math Reasoning](llm_reasoning/understanding_and_mitigating_spurious_signal_amplification_in_test-time_reinforc.md)**

:   系统分析测试时强化学习（TTRL）中虚假信号的来源和放大机制——中频答案构成模糊区域是主要噪声源，GRPO 的组内归一化会放大这些虚假信号——提出 DDRL 框架通过均衡采样、固定优势值和共识离线精炼三管齐下缓解问题，在 Qwen2.5-Math-1.5B 上相对提升15.3%。

**[When Is Thinking Enough? Early Exit via Sufficiency Assessment for Efficient Reasoning](llm_reasoning/when_is_thinking_enough_early_exit_via_sufficiency_assessment_for_efficient_reas.md)**

:   提出 DTSR 框架，通过检测推理过程中的"反思信号"（如 Wait、Alternatively）并在该位置让模型自我评估当前推理的"充分性"来决定是否提前终止推理，在 Qwen3 系列模型上实现 28.9%–34.9% 的推理长度缩减且几乎不损失精度。

**[Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment](llm_reasoning/which_reasoning_trajectories_teach_students_to_reason_better_a_simple_metric_of_.md)**

:   提出 Rank-Surprisal Ratio (RSR) 指标，通过联合衡量推理轨迹对学生模型的"信息量"和"对齐度"来评估训练数据适配性，在 5 个学生模型和 11 个教师模型的组合中与训练后性能达到平均 0.86 的 Spearman 相关性，并成功应用于轨迹选择和教师选择。

---

## 🦾 LLM Agent { #llm_agent }

**[AdaRubric: Task-Adaptive Rubrics for Reliable LLM Agent Evaluation and Reward Learning](llm_agent/adarubric_task-adaptive_rubrics_for_reliable_llm_agent_evaluation_and_reward_lea.md)**

:   本文指出 "LLM-as-Judge + 固定 rubric"（Helpfulness/Safety/Fluency）对评估目标导向的 agent 轨迹严重不匹配，提出 AdaRubric——由 LLM 根据任务描述自动生成任务专属的 N 维评估 rubric，再用置信度加权的逐步评估产出密集 reward 信号；并设计 DimensionAwareFilter 在 DPO 数据构建中防止"维度掩盖"，在 WebArena/ToolBench/AgentBench 上 Pearson r=0.79，DPO 训练带来 +6.8~+8.5% 任务成功率提升。

**[AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](llm_agent/agencybench_benchmarking_the_frontiers_of_autonomous_agents_in_1m-token_real-wor.md)**

:   提出AgencyBench——一个包含138个真实世界任务的综合基准，评估6种核心智能体能力，每个场景平均需90次工具调用和100万token，通过用户模拟agent和Docker沙箱实现全自动化评估。

**[Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](llm_agent/agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)**

:   本文提出 Agent-GWO，将灰狼优化器的领导者-追随者机制引入多智能体框架，联合优化 prompt 模板和解码超参数（温度、top-p 等），在 11 个数学和混合推理基准上持续超越现有提示优化方法。

**[AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](llm_agent/anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)**

:   提出AnchorMem记忆框架，受普鲁斯特现象启发，将检索单元（原子事实）与生成上下文（原始交互）解耦，通过关联事件图连接碎片化记忆，在LoCoMo基准上大幅超越A-Mem、Mem0等现有记忆系统。

**[AVA: Attentive VLM Agent for Mastering StarCraft II](llm_agent/ava_attentive_vlm_agent_for_mastering_starcraft_ii.md)**

:   本文提出 AVACraft——首个同时支持 MARL 和 VLM 两种决策范式的星际争霸 II 多模态基准（21 场景 / RGB+文本+结构化状态），并给出 VLM 基线 AVA（多模态优先级推理 + RAG + 动态角色分配），实验显示在基础 3m 场景 MARL 训练 5M 步只能到 19–27% 胜率，而 VLM 零样本就能拿到 75–90%。

**[BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search](llm_agent/bapo_boundary-aware_policy_optimization_for_reliable_agentic_search.md)**

:   针对 RL 训练后的 agentic search 模型几乎从不说"I DON'T KNOW"导致编造答案的可靠性问题，BAPO 在 GRPO 之上加入"基于组的边界感知奖励 + 自适应奖励调制器"，让模型只在真正越界时才拒答，相对 GRPO 在四个多跳 QA 上把 reliability 平均提升约 9.7%，且仅用 5k 训练样本就超过 90k 样本训练的 Search-R1。

**[CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents](llm_agent/clag_adaptive_memory_organization_via_agent-driven_clustering_for_small_language.md)**

:   本文提出 CLAG，一种基于聚类的 Agent 记忆框架，通过 SLM 驱动的路由将记忆组织到语义一致的聚类中，在聚类内部进行局部进化更新，并通过两阶段检索过滤噪声，在多个 QA 数据集上显著优于全局记忆池基线。

**[CodeStruct: Code Agents over Structured Action Spaces](llm_agent/codestruct_code_agents_over_structured_action_spaces.md)**

:   本文提出CodeStruct框架，将代码仓库重新定义为基于AST的结构化动作空间，让LLM代码Agent通过命名的程序实体（而非文本片段）进行读取和编辑操作，在SWE-Bench Verified上提升1.2-5.0%准确率并减少12-38% token消耗。

**[CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](llm_agent/coevolve_training_llm_agents_via_agent-data_mutual_evolution.md)**

:   CoEvolve 提出**智能体-数据共进化框架**，通过从训练轨迹中提取遗忘/边界/稀有三类弱点信号，引导 LLM 做针对性环境再探索和任务合成，使训练数据分布随智能体能力动态适应，在 AppWorld 和 BFCL 上分别带来 19-23% 的绝对提升。

**[Context-Value-Action Architecture for Value-Driven Large Language Model Agents](llm_agent/context-value-action_architecture_for_value-driven_large_language_model_agents.md)**

:   提出 CVA（Context-Value-Action）架构，基于 S-O-R 心理学模型和 Schwartz 价值理论，通过训练在真实人类数据上的 Value Verifier 解耦行为生成与认知推理，有效缓解 LLM 智能体的行为极化问题，在超过 110 万真实交互轨迹的 CVABench 上显著优于基线。

**[Do LLM Agents Mirror Socio-Cognitive Effects in Power-Asymmetric Conversations?](llm_agent/do_llm_agents_mirror_socio-cognitive_effects_in_power-asymmetric_conversations.md)**

:   这篇论文用职业角色和 persona 模拟权力不对称对话，发现 LLM agent 会复现代词使用、语言协调、权威说服和有害服从等社会认知效应，其中一些效应提升对话真实感，另一些则带来安全风险。

**[Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction](llm_agent/don39t_act_blindly_robust_gui_automation_via_action-effect_verification_and_self.md)**

:   本文提出VeriGUI框架，通过Thinking-Verification-Action-Expectation（TVAE）闭环推理机制和两阶段训练管线（Robust SFT + GRPO），让GUI Agent能够验证每步操作是否成功并在失败时自我纠正，在3B和7B规模上均显著优于基线。

**[Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](llm_agent/don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)**

:   本文提出 PA-Tool，一种无训练的工具 Schema 优化方法，利用从数据污染检测中借鉴的"尖锐度"（peakedness）信号识别模型预训练中熟悉的命名模式，通过重命名工具组件来对齐小语言模型的内化知识，在 MetaTool 和 RoTBench 上实现最高 17% 的提升，Schema 不对齐错误减少 80%。

**[Don't Click That: Teaching Web Agents to Resist Deceptive Interfaces](llm_agent/dont_click_that_teaching_web_agents_to_resist_deceptive_interfaces.md)**

:   作者首次把"对抗欺骗性 UI"形式化为 web agent 的独立防御问题，提出两阶段框架 **DUDE**（不对称惩罚的 hybrid-reward RL 学一个 evaluator + 用 experience summarization 把失败模式蒸馏成可迁移上下文），并发布含 1407 个真实/合成场景的 **RUC** 基准，在 3 个 VLM agent base 上把欺骗诱发失败率从 23.5% 降到 1.5%、任务成功率从 9.5% 推到 60.5%，且 Stage-2 优化的 prompt 能零样本迁移到闭源模型。

**[Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models](llm_agent/dynamic_generation_of_multi-llm_agents_communication_topologies_with_graph_diffu.md)**

:   本文提出 Guided Topology Diffusion，将多 LLM agent 的通信拓扑生成建模为条件图扩散过程，并用代理奖励模型在每个去噪步骤做零阶引导，从而生成更稀疏、更省 token、更鲁棒的任务自适应协作网络。

**[Exploring Reasoning Reward Model for Agents](llm_agent/exploring_reasoning_reward_model_for_agents.md)**

:   作者发现现在 agentic RL 普遍用 sparse outcome reward（只看最终对错），导致中间多步推理质量信号被丢掉，于是提出 **Agent-RRM**——一个产生 `<think>/<critique>/<score>` 三段结构化反馈的 reasoning reward model，并系统对比三种整合方式（C：纯 critique refinement、R：scalar reward 增强、U：critique + score 联合 GRPO），最终 **Reagent-U** 在 12 个 benchmark 上把 Qwen3-8B 的 GAIA 拉到 43.7%、WebWalkerQA 拉到 46.2%，证明"语言级 critique + 数值 reward"联合监督比单一信号好得多。

**[ExpSeek: Self-Triggered Experience Seeking for Web Agents](llm_agent/expseek_self-triggered_experience_seeking_for_web_agents.md)**

:   ExpSeek 提出了一种基于步级熵自触发的经验主动寻求框架，让 Web Agent 在交互过程中根据自身信号判断何时需要指导、获取什么指导，在 Qwen3-8B/32B 上分别实现 9.3% 和 7.5% 的绝对提升。

**[FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments](llm_agent/fama_failure-aware_meta-agentic_framework_for_open-source_llms_in_interactive_to.md)**

:   FAMA 先用一套独立"失败分析 agent + 编排 agent"自动诊断基线 tool-use agent 在 τ-bench 这类多轮交互上的主要失败模式，再让一个 mitigation agent 按需挑选最小子集 helper agent 注入到上下文，从而在 Qwen 系列开源模型上把任务成功率最高拉高 27%。

**[FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems](llm_agent/fedgui_benchmarking_federated_gui_agents_across_heterogeneous_platforms_devices_.md)**

:   FedGUI 是首个面向跨平台 GUI 代理的联邦学习综合基准，包含六个数据集覆盖移动端/网页端/桌面端，系统研究跨平台、跨设备、跨操作系统和跨数据源四种异构性对联邦 GUI 代理训练的影响。

**[Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](llm_agent/feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)**

:   本文提出 FTRL 框架，通过五阶段自动化管线构建稳定可控的工具使用训练环境，并设计结合工具调用精度和任务完成度的可验证奖励机制，与偏好优化 RL 算法结合后，在 7B-14B 模型上实现平均超 10% 的工具使用性能提升，甚至超越最强闭源模型。

**[FregeLogic at SemEval 2026 Task 11: A Hybrid Neuro-Symbolic Architecture for Content-Robust Syllogistic Validity Prediction](llm_agent/fregelogic_at_semeval_2026_task_11_a_hybrid_neuro-symbolic_architecture_for_cont.md)**

:   提出 FregeLogic 混合神经符号系统，结合五成员 LLM 集成和 Z3 SMT 求解器作为决胜裁判，在三段论有效性判断中将内容效应降低16%的同时提升准确率0.9%。

**[From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms](llm_agent/from_storage_to_experience_a_survey_on_the_evolution_of_llm_agent_memory_mechani.md)**

:   本文用「Storage → Reflection → Experience」三阶段演化框架把 LLM Agent 记忆机制做系统综述，用形式化定义把三阶段对应为「轨迹保留 → 轨迹精化 → 跨轨迹抽象」三种 functional signature，并用"Why-How-What"三层 RQ 串起整个 storyline，重点展开 Experience 阶段的两大变革性机制 — Active Exploration 与 Cross-Trajectory Abstraction。

**[GOAT: A Training Framework for Goal-Oriented Agent with Tools](llm_agent/goat_a_training_framework_for_goal-oriented_agent_with_tools.md)**

:   GOAT 通过从 API 文档自动构建"依赖图 + call-first 合成数据"的流水线，让开源小模型在无需人工标注的情况下学会把高层目标拆成一串相互依赖的 API 调用，在 RestBench / API-Bank / 自建的 GOATBench 上把开源模型推到 SOTA，部分场景甚至超过闭源大模型。

**[Grounding Agent Memory in Contextual Intent](llm_agent/grounding_agent_memory_in_contextual_intent.md)**

:   STITCH 为 LLM agent 的长程记忆引入"contextual intent"（主题域 + 事件类型 + 关键实体类型）三元组作为结构化检索 cue，在每条 trajectory step 上在线归纳；推理时按"标签密度排序"先做结构匹配再做语义打分，在新构建的 CAME-Bench 上随 trajectory 增长不掉点，比最强 baseline 在 Large subset 上提升 35.6% 绝对（100% 相对）。

**[HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation](llm_agent/hag_hierarchical_demographic_tree-based_agent_generation_for_topic-adaptive_simu.md)**

:   提出 HAG 框架，将群体 Agent 生成形式化为两阶段层次化决策过程——先用世界知识模型构建主题自适应人口分布树实现宏观分布对齐，再通过真实数据检索与 Agent 增强保证微观个体一致性，在多领域基准上将群体对齐误差平均降低 37.7%、社会学一致性提升 18.8%。

**[HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents](llm_agent/hela-mem_hebbian_learning_and_associative_memory_for_llm_agents.md)**

:   HeLa-Mem 提出了一种受神经科学启发的 LLM 代理记忆架构，将对话历史建模为带 Hebbian 学习动力学的动态图——通过共激活强化记忆间连接、反思性蒸馏将枢纽记忆凝练为语义知识、双路径检索结合语义相似度和 Hebbian 扩散激活，在 LoCoMo 上以显著更少的 token 达到最优性能。

**[Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](llm_agent/hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)**

:   本文提出 STEP-HRL，通过引入局部进度模块将交互历史迭代压缩为紧凑的文本摘要，使高层和低层策略仅基于单步转移（而非完整历史）做决策，在 ScienceWorld 和 ALFWorld 上显著提升性能和泛化性，同时减少 token 使用。

**[HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](llm_agent/higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)**

:   本文提出 HiGMem，一个两层事件-对话轮记忆系统，通过让 LLM 先浏览事件摘要再预测哪些细粒度对话轮值得读取，在 LoCoMo10 基准上以少一个数量级的检索量达到了五类问题中四类的最优 F1。

**[How Adversarial Environments Mislead Agentic AI](llm_agent/how_adversarial_environments_mislead_agentic_ai.md)**

:   本文形式化了"对抗环境注入"（AEI）威胁模型，将其分解为广度攻击（投毒检索结果导致认知漂移）和深度攻击（注入幻影节点构造导航陷阱导致策略崩溃），在 11,000+ 次实验中发现两种攻击的鲁棒性完全独立——"鲁棒性分裂"表明当前单点防御策略根本不够。

**[ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models](llm_agent/implicitmembench_measuring_unconscious_behavioral_adaptation_in_large_language_m.md)**

:   提出 ImplicitMemBench，首个系统评估 LLM 隐式记忆的基准，包含程序性记忆、启动效应和经典条件反射三种认知范式共 300 个测试项，在 17 个模型上揭示严重局限：最优模型仅达 66% 整体准确率，远低于人类基线。

**[IntrAgent: An LLM Agent for Content-Grounded Information Retrieval through Literature Review](llm_agent/intragent_an_llm_agent_for_content-grounded_information_retrieval_through_litera.md)**

:   IntrAgent 把"研究者读论文找信息"这件事拆成"先按结构排章节、再迭代读章节、读够就停"两段流水线，让 LLM 在不依赖向量检索的前提下，从一整篇科学论文里抽取出与查询忠实对齐的细粒度答案，在新基准 IntraBench 上跨五个 STEM 领域平均比 RAG/科研 agent 基线高出 13.2%。

**[Lightweight LLM Agent Memory with Small Language Models](llm_agent/lightweight_llm_agent_memory_with_small_language_models.md)**

:   本文提出 LightMem，一种由多个专用小语言模型（SLM）驱动的轻量级 LLM 智能体记忆系统，通过将记忆操作模块化为控制器（SLM-1）、选择器（SLM-2）和写入器（SLM-3），并将在线处理与离线整合解耦，在 LoCoMo 基准上平均 F1 提升约 2.5（相比 A-MEM），同时实现 83ms 检索延迟和 581ms 端到端延迟。

**[LiTS: A Modular Framework for LLM Tree Search](llm_agent/lits_a_modular_framework_for_llm_tree_search.md)**

:   LiTS 把 LLM tree search 拆成 Policy、Transition、RewardModel 和统一数据结构，用 decorator registry 让同一套搜索算法、组件和任务逻辑可以在数学推理、环境规划和工具调用之间组合复用，并通过实验指出开放文本动作空间中的 policy diversity 是树搜索瓶颈。

**[LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization](llm_agent/lpo_towards_accurate_gui_agent_interaction_via_location_preference_optimization.md)**

:   本文提出 Location Preference Optimization (LPO)，通过基于信息熵的窗口奖励和基于物理距离的动态位置奖励，结合 GRPO 框架优化 GUI 智能体的空间定位精度，在离线和在线评估中均达到 SOTA。

**[MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents](llm_agent/magma_a_multi-graph_based_agentic_memory_architecture_for_ai_agents.md)**

:   MAGMA 把 LLM agent 的记忆拆成语义 / 时间 / 因果 / 实体四张正交关系图，再用 intent 路由 + 适应性 beam search 在合适的图上做 policy-guided traversal 检索，并配以"快路径同步入库 + 慢路径异步 LLM 巩固"双流写入；在 LoCoMo 上 Judge 0.700 全面超过 A-MEM / Nemori / MemoryOS，同时 query latency 仅 1.47s（比次优快 40%）。

**[MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools](llm_agent/mcp-flow_facilitating_llm_agents_to_master_real-world_diverse_and_scaling_mcp_to.md)**

:   MCP-Flow 提出了一个基于 Web Agent 的自动化管道，从 1166 个真实 MCP 服务器中收集工具信息并合成 68733 条高质量训练数据，使小规模微调模型（0.6B-8B）在 MCP 工具使用上超越 GPT-4o 等 SOTA 大模型。

**[Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](llm_agent/mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)**

:   本文提出 Mem²Evolve，一种通过双记忆机制（资产记忆 + 经验记忆）实现能力扩展与经验蒸馏协同进化的自进化 Agent 框架，在 6 类任务 8 个基准上平均 Pass@1 达 70.24%，分别超过纯经验进化和纯能力进化的最强基线 11.80% 和 6.46%。

**[Mem^p: Exploring Agent Procedural Memory](llm_agent/memp_exploring_agent_procedural_memory.md)**

:   本文提出 Mem^p 框架，系统性地研究如何为 LLM Agent 构建可学习、可更新、终身演化的程序性记忆——通过将过去的任务轨迹蒸馏为细粒度的分步指令和高层脚本抽象，并配合动态更新机制（添加/验证/反思/淘汰），在 TravelPlanner 和 ALFWorld 上实现了成功率持续提升和执行步数大幅减少。

**[MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End RL](llm_agent/memsearcher_training_llms_to_reason_search_and_manage_memory_via_end-to-end_rein.md)**

:   MemSearcher 把搜索 agent 的"历史拼接"换成"LLM 自管理的紧凑内存"——每轮只看 `(question, memory)` 而不是 `(question, t₁, a₁, o₁, …)`，并用 multi-context GRPO 把整条 trajectory 的 advantage 传播到每一轮独立优化，在 7 个 QA benchmark 上 3B/7B/14B 全面超过同尺寸 ReAct baseline（7B 甚至超 32B ReSearch），context 长度恒定 <4K token。

**[Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models](llm_agent/meta-tool_efficient_few-shot_tool_adaptation_for_small_language_models.md)**

:   通过在四个基准上系统对比超网络 LoRA 适应 vs 精心设计的 few-shot 提示，发现 2.28 亿参数的超网络提供零增益——few-shot 示例贡献 +21.5%、文档编码贡献 +5.0%、超网络贡献 0%，3B 模型配合良好提示可达 GPT-5 平均性能的 79.7% 且延迟低 10 倍。

**[Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh](llm_agent/mina_a_multilingual_llm-powered_legal_assistant_agent_for_bangladesh_for_empower.md)**

:   开发 Mina——面向孟加拉国法律场景的多语言 LLM 法律助手，通过两阶段 RAG 流水线精准检索法案和条款，配合工具链和多语言嵌入，在孟加拉律师资格考试中取得 75-80% 的通过成绩，法律咨询成本仅为传统方式的 0.12-0.61%。

**[MOOSE-Copilot: A Web-Based Interactive Assistant for Unified Exploratory and Fine-Grained Scientific Hypothesis Discovery](llm_agent/moose-copilot_a_web-based_interactive_assistant_for_unified_exploratory_and_fine.md)**

:   MOOSE-Copilot 把发散式科研 idea 探索和收敛式细粒度假设 refinement 统一到一个可视化人机协同系统中，并用初始蓝图、阶段路由和反馈三类显式人工信号显著提升科学假设发现效果。

**[OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory](llm_agent/ocr-memory_optical_context_retrieval_for_long-horizon_agent_memory.md)**

:   OCR-Memory 将长程 Agent 交互轨迹渲染成带编号锚点的图像，让微调后的 OCR 检索器先在视觉空间定位相关片段、再按索引回取原始文本，从而在严格上下文预算下保留完整历史并提升 Mind2Web 和 AppWorld 上的长程任务表现。

**[OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning](llm_agent/octotools_an_agentic_framework_with_extensible_tools_for_complex_reasoning.md)**

:   OctoTools是一个免训练、用户友好且易于扩展的多智能体框架，通过标准化**工具卡**封装异构工具、**Planner-Executor**分离范式和**任务特定工具集优化**算法，在16个多样化基准上实现了比GPT-4o平均+9.3%、比AutoGen/LangChain等框架最多+10.6%的准确率提升。

**[OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation](llm_agent/opera_a_dataset_of_observation_persona_rationale_and_action_for_evaluating_llms_.md)**

:   OPeRA 是一个从真实 Amazon 购物过程中采集的用户行为数据集，把 persona、网页观察、细粒度动作和即时 rationale 放在同一条时间轴上，用来评估 LLM 是否真的能模拟特定用户的下一步购物行为。

**[PersonaAgent: Bridging Memory and Action for Personalized LLM Agents](llm_agent/personaagent_bridging_memory_and_action_for_personalized_llm_agents.md)**

:   PersonaAgent 用“个性化记忆 + 个性化动作 + 可测试时优化的 persona prompt”把用户历史和工具行动连接起来，在 LaMP 多个个性化决策任务上明显超过 RAG、PAG、ReAct 和 MemBank 等基线。

**[Polaris: A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair](llm_agent/polaris_a_gödel_agent_framework_for_small_language_models_through_experience-abs.md)**

:   Polaris 将 Gödel Agent 的递归自我改进改造成适合 7B/8B 小模型的“失败分析 → 经验抽象 → 最小代码补丁 → 执行校验”策略修复循环，在 MGSM、DROP、GPQA、LitBench 上让小模型获得可解释、可持久复用的 policy-level 改进。

**[ProPer Agents: Proactivity Driven Personalized Agents for Advancing Knowledge Gap Navigation](llm_agent/proper_agents_proactivity_driven_personalized_agents_for_advancing_knowledge_gap.md)**

:   ProPer 把主动式助手建模为“发现并校准用户未说出的任务维度”的问题，通过 Dimension Generating Agent、post-hoc reranker 和 Response Generating Agent 选择性补足知识缺口，在医疗、代码和购物推荐任务上显著提升回答质量与 win rate。

**[RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents](llm_agent/recmem_recurrence-based_memory_consolidation_for_efficient_and_effective_long-ru.md)**

:   RecMem 借鉴人类记忆中的“重复出现才固化”原则，把原始交互先放入轻量 subconscious memory，只有检测到语义复现时才调用 LLM 生成 episodic 与 semantic memory，从而在 LoCoMo 和 LongMemEval-S 上以显著更低构建 token 成本达到或超过主流记忆系统的问答准确率。

**[Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems](llm_agent/rethinking_reasoning-intensive_retrieval_evaluating_and_advancing_retrievers_in_.md)**

:   本文提出 BRIGHT-PRO，用多方面证据标注和 agentic search 协议重新评测 reasoning-intensive retriever，并用 RTriever-Synth 训练 RTriever-4B，证明检索器应优化“证据组合覆盖”而非单篇相关性。

**[Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors](llm_agent/robust_tool_use_via_fission-grpo_learning_to_recover_from_execution_errors.md)**

:   提出 Fission-GRPO，在 RL 训练循环中将工具执行错误动态转化为在线策略修正训练实例：通过学习的错误模拟器生成诊断反馈并重采样恢复轨迹，将 Qwen3-8B 的错误恢复率提升 5.7%，整体准确率从 42.75% 提升至 46.75%。

**[SafeMCP: Proactive Power Regulation for LLM Agent Defense via Environment-Grounded Look-Ahead Reasoning](llm_agent/safemcp_proactive_power_regulation_for_llm_agent_defense_via_environment-grounde.md)**

:   SafeMCP 是一个部署在 MCP server 侧的 agent 防御插件，通过环境动力学世界模型做 look-ahead reasoning，先过滤会扩大危险能力边界的工具，再对已发起的危险调用做即时拦截，在 PowerSeeking Bench、ToolEmu 和 AgentHarm 上同时提升安全性并尽量保留任务效用。

**[SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents](llm_agent/searl_joint_optimization_of_policy_and_tool_graph_memory_for_self-evolving_agent.md)**

:   SEARL 将 agent 的策略参数和外部 Tool Graph Memory 联合优化，用工具锚定的 step-level advantage 与过程奖励解决长轨迹信用分配，使小模型在多跳问答和复杂数学任务中能持续创建、复用和整合工具。

**[Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks](llm_agent/shopping_companion_a_memory-augmented_llm_agent_for_real-world_e-commerce_tasks.md)**

:   Shopping Companion 构建了一个带长期用户偏好记忆和真实商品库的电商任务基准，并用两阶段 agent 加双重奖励、工具级奖励联合优化偏好识别和商品推荐，使 4B 模型接近强闭源模型表现。

**[SOLAR-RL: Semi-Online Long-horizon Assignment Reinforcement Learning](llm_agent/solar-rl_semi-online_long-horizon_assignment_reinforcement_learning.md)**

:   SOLAR-RL 用离线轨迹重构、失败点检测和目标对齐奖励塑形，把静态 GUI 数据加工成带有伪在线反馈的长程训练信号，使 Qwen2.5-VL-7B 规模的 GUI agent 在 Android Control、GUI-Odyssey 和 Android World 上获得接近或超过强 offline baseline 的稳定表现。

**[Spec-o3: A Tool-Augmented Vision-Language Agent for Rare Celestial Object Candidate Identification](llm_agent/spec-o3_a_tool-augmented_vision-language_agent_for_rare_celestial_object_candida.md)**

:   提出 Spec-o3，一个工具增强的视觉语言智能体，通过交错多模态思维链（iMCoT）模拟天文学家的光谱检查流程，采用冷启动 SFT + 基于结果的 RL 两阶段训练，在稀有天体识别上将 macro-F1 从 28.3% 提升至 76.5%，推理速度比人工检查快 ~50 倍。

**[StructMem: Structured Memory for Long-Horizon Behavior in LLMs](llm_agent/structmem_structured_memory_for_long-horizon_behavior_in_llms.md)**

:   StructMem 提出了一种结构增强的层次化记忆框架，通过事件级双视角提取和跨事件语义整合，在 LoCoMo 长对话基准上实现 SOTA 性能（76.82%），同时大幅降低 token 消耗（1.94M vs. 图记忆的 35.8M）和 API 调用次数。

**[Supplement Generation Training for Enhancing Agentic Task Performance](llm_agent/supplement_generation_training_for_enhancing_agentic_task_performance.md)**

:   SGT（Supplement Generation Training）训练一个小型 LLM（1.7B）生成逐实例的补充文本（推理线索、摘要、错误提醒等），附加到输入后让冻结的大型 Actor 模型更有效地解决任务，在 5 个基准上平均提升 21%，无需修改大模型参数。

**[SynthAgent: Adapting Web Agents with Synthetic Supervision](llm_agent/synthagent_adapting_web_agents_with_synthetic_supervision.md)**

:   本文提出 SynthAgent，一个完全基于合成监督的 Web Agent 适应框架，通过分类探索系统覆盖网页功能区域以合成多样化任务，再通过任务精炼（冲突检测触发修正幻觉）和轨迹精炼（全局视角去噪）的双重精炼策略提升合成数据质量，在 WebArena 和 Online-Mind2Web 上显著优于现有合成方法。

**[Taming Actor-Observer Asymmetry in Agents via Dialectical Alignment](llm_agent/taming_actor-observer_asymmetry_in_agents_via_dialectical_alignment.md)**

:   发现 LLM Agent 在角色扮演中会表现出类人的"行动者-观察者不对称"（AOA）认知偏差——作为行动者倾向归因外部因素，作为观察者倾向归因内部错误，提出 ReTAS 通过辩证推理（正题-反题-合题）和 GRPO 对齐来消除这一偏差。

**[Temp-R1: A Unified Autonomous Agent for Complex Temporal KGQA via Reverse Curriculum Reinforcement Learning](llm_agent/temp-r1_a_unified_autonomous_agent_for_complex_temporal_kgqa_via_reverse_curricu.md)**

:   Temp-R1 把时序知识图谱问答从人工设计的固定 prompt workflow 改造成可强化学习训练的自主 agent，通过显式内部动作、SFT 冷启动、GRPO 和“先难后易”的反向课程，在 8B 开源模型上超过多种 GPT-4o/DeepSeek-V3 驱动的强基线。

**[The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive Reality Check](llm_agent/the_bitter_lesson_of_diffusion_language_models_for_agentic_workflows_a_comprehen.md)**

:   这篇论文系统评估扩散语言模型在具身智能体和工具调用智能体中的表现，发现它们虽然有并行解码带来的速度潜力，却在长程因果规划和严格格式生成上显著落后于自回归 LLM，并进一步用 DiffuAgent 证明 dLLM 更适合作为记忆压缩、工具筛选等非因果辅助模块。

**[TheraAgent: Self-Improving Therapeutic Agent for Precise and Comprehensive Treatment Planning](llm_agent/theraagent_self-improving_therapeutic_agent_for_precise_and_comprehensive_treatm.md)**

:   TheraAgent 把治疗方案生成从一次性回答改造成 generate-reflect-refine 的自改进 agent 流程，用临床维度化评估器 TheraJudge 和 score-aware memory 不断修正方案，在 HealthBench 治疗规划子集和医生盲评中显著超过强基线。

**[TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents](llm_agent/timem_temporal-hierarchical_memory_consolidation_for_long-horizon_conversational.md)**

:   TiMem 将长程对话记忆组织成显式时间包含的五层 Temporal Memory Tree，并用复杂度感知检索在细粒度事实和高层 persona 之间动态取舍，在 LoCoMo 和 LongMemEval-S 上同时提升准确率并显著减少召回上下文长度。

**[ToolGrad: Efficient Tool-use Dataset Generation with Textual "Gradients"](llm_agent/toolgrad_efficient_tool-use_dataset_generation_with_textual_gradients.md)**

:   ToolGrad 把工具使用数据生成从“先写用户问题、再用 DFS 找工具链”反过来做成“先生成可执行成功的工具链、再反推用户问题”，并用类似 textual gradients 的 API 选择循环构造 ToolGrad-500，使数据生成 pass rate 达到 99.8%，训练出的 Gemma-3 小模型还能超过多种闭源强模型的单轮工具调用表现。

**[ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution](llm_agent/toolomni_enabling_open-world_tool_use_via_agentic_learning_with_proactive_retrie.md)**

:   本文提出 ToolOmni，一个统一的智能体框架，将主动工具检索和基于检索结果的工具执行整合在同一推理循环中，通过冷启动 SFT + 解耦多目标 GRPO 联合优化检索和执行能力，在 ToolBench 上端到端执行成功率超过强基线 +10.8%。

**[拓扑重要：多智能体 LLM 中的内存泄露测量](llm_agent/topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)**

:   本文通过 MAMA 框架系统地测量了多智能体 LLM 系统中通信拓扑如何影响个人可识别信息的泄露程度，发现密集连接的拓扑结构和攻击者与目标的距离是决定泄露风险的关键因素。

**[Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](llm_agent/towards_scalable_lightweight_gui_agents_via_multi-role_orchestration.md)**

:   本文提出 LAMO 框架，通过角色导向的数据合成和两阶段训练（SFT with Perplexity-Weighted Cross-Entropy + 多任务 RL），将轻量 3B MLLM 训练为可灵活编排多角色的 GUI Agent，在单体推理、多 Agent 协作和即插即用策略执行器三种模式下工作，搭配 GPT-5 规划器在 AndroidWorld 上达 77.6% 成功率，超越 72B 参数的专用 GUI Agent。

**[Uncertainty Quantification in LLM Agents: Foundations, Emerging Challenges, and Opportunities](llm_agent/uncertainty_quantification_in_llm_agents_foundations_emerging_challenges_and_opp.md)**

:   本文提出首个 Agent 不确定性量化（Agent UQ）的形式化框架：将 agent 的问题解决轨迹建模为动态贝叶斯网络上的随机过程 $P(\mathcal{F}_{\leq T}) = P(E_0, O_0) \prod_{i=1}^{T} P_{\pi,\mathcal{T}}(A_i|E_{i-1}, O_{i-1}) P(O_i|A_i, E_i)$，统一了现有 UQ 范式（单步 QA、多步推理）为特例，并通过 $\tau^2$-bench 上的实证分析识别了四个 agent UQ 特有的技术挑战。

**[Verified Critical Step Optimization for LLM Agents](llm_agent/verified_critical_step_optimization_for_llm_agents.md)**

:   CSO 从 agent 自己失败的轨迹中找出“换一个动作就能让任务成功”的 verified critical steps，只在这些关键决策点构造 DPO 偏好对，从而用更少、更可靠的监督提升长程 LLM agent 的后训练效果。

**[Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories](llm_agent/waking_up_blind_cold-start_optimization_of_supervision-free_agentic_trajectories.md)**

:   本文提出 SPECTRA，一种无需监督轨迹的框架——通过冷启动强化学习（GRPO）和软结构化多轮 rollout 拓扑约束，让小型视觉语言模型（SVLM）在纯环境交互中自行发现有效的工具调用和视觉推理行为，在 4 个多模态 benchmark 上提升任务准确率达 5% 和工具效率 9%，同时提出 Tool Instrumental Utility（TIU）指标量化无监督下的工具效能。

**[WebClipper: Efficient Evolution of Web Agents with Graph-based Trajectory Pruning](llm_agent/webclipper_efficient_evolution_of_web_agents_with_graph-based_trajectory_pruning.md)**

:   WebClipper 把 Web Agent 的长工具调用轨迹建模成“动作节点-信息节点”状态图，再挖掘最小必要 DAG 来剪掉循环搜索和无效分支，使 Deep Research 类 agent 平均减少约 21% 工具轮次和 19.4% token，同时保持甚至提升准确率。

**[What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search](llm_agent/what_makes_an_llm_a_good_optimizer_a_trajectory_analysis_of_llm-guided_evolution.md)**

:   本文通过大规模实验（15 个 LLM × 8 个任务、72K 候选解）发现优秀的 LLM 优化器表现为"局部精炼器"——持续产生频繁的渐进式改进并在语义空间中逐步集中搜索，而非产生高新颖性的跳跃式突破；关键发现是新颖性本身并不预测优化性能，只有当搜索保持足够局部化时新颖性才有益。

**[When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors](llm_agent/when_agents_look_the_same_quantifying_distillation-induced_similarity_in_tool-us.md)**

:   本文提出了 RPS 和 AGS 两个互补指标来量化 LLM Agent 在工具使用行为上的蒸馏导致的同质化现象，通过区分必要行为和非必要行为，在 18 个模型上揭示了跨家族行为继承模式，发现 Kimi-K2 与 Claude Sonnet 4.5 的行为相似度甚至超过 Anthropic 自家模型。

**[为什么 LLM 网络代理失败：一个分层规划视角](llm_agent/why_do_llm-based_web_agents_fail_a_hierarchical_planning_perspective.md)**

:   本文通过分层规划框架（高层计划、低层执行、重规划）系统分析 LLM 网络代理的失败原因，发现 PDDL 表示优于自然语言规划，但低层执行和感知接地是真正的瓶颈。

**[YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents](llm_agent/yield_a_large-scale_dataset_and_evaluation_framework_for_information_elicitation.md)**

:   提出信息引出代理（IEA）作为新的对话范式，发布了首个大规模（2,281 段对话，26M token）人与人信息引出对话数据集 YIELD，将信息引出形式化为有限视野 POMDP，并设计了专门的评估指标（Conformity、Progression、TLR），实验表明在 YIELD 上微调能显著提升 LLM 与真实引出行为的对齐。

**[Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception](llm_agent/your_llm_agents_are_temporally_blind_the_misalignment_between_tool_use_decisions.md)**

:   揭示 LLM Agent 在多轮交互中的"时间盲区"(Temporal Blindness)——无法根据消息间流逝的真实时间来调整工具调用决策，并构建 TicToc 基准评估这一问题。

**[ZARA: Training-Free Motion Time-Series Reasoning via Evidence-Grounded LLM Agents](llm_agent/zara_training-free_motion_time-series_reasoning_via_evidence-grounded_llm_agents.md)**

:   提出 ZARA，一个基于知识和检索增强的多智能体框架，通过将传感器信号蒸馏为结构化文本知识库、类别条件检索和分层 LLM 推理，在完全免训练的设置下实现了可解释的人体活动识别，8 个数据集上大幅超越现有方法。

---

## 🔍 信息检索/RAG { #information_retrieval }

**[A Picture is Worth a Thousand Words? An Empirical Study of Aggregation Strategies for Visual Financial Document Retrieval](information_retrieval/a_picture_is_worth_a_thousand_words_an_empirical_study_of_aggregation_strategies.md)**

:   通过精心设计的金融文档诊断 benchmark（单数字扰动 + 文本掩码），实证证明「把 VLM 的 patch tokens 聚合成单向量」会让 $1.2M vs $7.2M 这种语义巨大差异坍缩成 cosine 相似度 > 0.99 的几乎相同向量，根因是「全局纹理主导」，多种缓解策略和 retrieval-tuned embedding 都救不回来。

**[A Survey of Reasoning-Intensive Retrieval: Progress and Challenges](information_retrieval/a_survey_of_reasoning-intensive_retrieval_progress_and_challenges.md)**

:   本文系统梳理了"推理密集型检索 (Reasoning-Intensive Retrieval, RIR)"这一新方向，按 query/index/retriever/reranker/迭代 这条流水线给出了第一份完整的 benchmark-方法-挑战 三段式综述，并指出现有评测过度依赖 nDCG 等传统 IR 指标。

**[Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](information_retrieval/agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)**

:   提出ConvAgent，通过将RL训练奖励分解为结果奖励、信息增益奖励和混合主动行为奖励三个互补组件，训练对话式搜索智能体在多轮交互中交替进行搜索和推理。

**[All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG](information_retrieval/all_languages_matter_understanding_and_mitigating_language_bias_in_multilingual_.md)**

:   系统揭示多语言 RAG 系统在重排序阶段存在严重的语言偏差（偏好英语和查询语言），提出 LAURA 框架通过下游生成质量驱动的监督信号对齐重排序器，有效缓解偏差并提升生成性能。

**[An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs](information_retrieval/an_iterative_utility_judgment_framework_inspired_by_philosophical_relevance_via_.md)**

:   受Schutz哲学相关性理论启发，提出ITEM迭代效用判断框架，通过让RAG中的三个组件（相关性排序、效用判断、答案生成）动态交互增强，在检索、效用判断和QA任务上均优于基线。

**[AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation](information_retrieval/authoritybench_benchmarking_llm_authority_perception_for_reliable_retrieval-augm.md)**

:   AuthorityBench 用 10K 网页域名（PageRank 真值）+22K 实体（Wikipedia 跨语言 sitelink 真值）+120 RAG 问题构造首个 LLM「权威感知」基准，发现 ListJudge / PairJudge + PointScore 输出最准，加入网页文本反而拖后腿，且把权威信号用作 RAG 过滤能把答案准确率最多提 14 个百分点。

**[Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring](information_retrieval/bayesian_active_learning_with_gaussian_processes_guided_by_llm_relevance_scoring.md)**

:   提出 BAGEL，一个基于高斯过程（GP）的贝叶斯主动学习框架，在有限 LLM 预算下通过探索-利用平衡策略传播稀疏 LLM 相关性信号，实现全局嵌入空间的段落检索，显著超越传统 LLM 重排序方法。

**[Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders](information_retrieval/benchmarking_and_enabling_efficient_chinese_medical_retrieval_via_asymmetric_enc.md)**

:   提出 CMedTEB（中文医学文本嵌入基准）和 CARE（非对称检索框架），前者通过多 LLM 投票+专家验证构建高质量的中文医学检索/重排/STS 基准，后者用轻量 BERT 编码查询+大型 LLM 编码文档的非对称架构，通过两阶段渐进对齐策略实现 LLM 级检索精度+BERT 级在线延迟。

**[Beyond Black-Box Interventions: Latent Probing for Faithful Retrieval-Augmented Generation](information_retrieval/beyond_black-box_interventions_latent_probing_for_faithful_retrieval-augmented_g.md)**

:   提出 ProbeRAG，通过发现 LLM 隐空间中冲突/对齐知识的线性可分性，设计三阶段框架（细粒度知识剪枝→隐空间冲突探测→冲突感知注意力），从模型内部机制解决 RAG 忠实性问题。

**[BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning](information_retrieval/brief-pro_universal_context_compression_with_short-to-long_synthesis_for_fast_an.md)**

:   针对 RAG 在 10k+ 词长上下文下推理慢、信息淹没的问题，作者用「短上下文种子数据 → 维基扩张 → 头尾迭代裁剪」合成多跳长上下文训练数据，微调一个 3B 的 Llama-3.2 抽取式摘要器 BRIEF-Pro，在四个多跳 QA 数据集上以 32× 压缩率反超 LongLLMLingua 的 9× 压缩，并支持用户用句数指令直接控制摘要长度。

**[Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](information_retrieval/can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)**

:   本文提出 DGPO：用教师 demonstration 做冷启动 KD 初始化，再在 PPO 阶段对"错误样本"施加 KL 蒸馏惩罚，让 0.5B 紧凑模型获得 Agentic RAG 能力，在 7 个 QA benchmark 上平均 EM 从 0.006 提升到 0.329，部分数据集甚至反超 3B 教师。

**[ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](information_retrieval/chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)**

:   作者把 Search-R1 / R1-Searcher 这类"搜索 + 推理"RL 框架从单轮问答扩展到**多轮对话问答**，提出 ChatR1：用 PPO 端到端联合优化 reasoning / search / answer，并设计"intent-aware reward"——用历史人工 rewrite 与模型自发 search query 的 token-F1 作为 turn-level dense reward，在 5 个 CQA 数据集上以 3B 主干击败 ChatGPT/Claude，并大幅提升域外迁移能力。

**[ChunQiuTR: Time-Keyed Temporal Retrieval in Classical Chinese Annals](information_retrieval/chunqiutr_time-keyed_temporal_retrieval_in_classical_chinese_annals.md)**

:   提出 ChunQiuTR，首个基于非格里历的时间键检索基准，从《春秋》及其注疏传统中构建，并设计了 CTD（历法时间双编码器），通过傅里叶绝对历法上下文和相对偏移偏置实现时间感知检索，显著优于纯语义基线。

**[CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation](information_retrieval/citeguard_faithful_citation_attribution_for_llms_via_retrieval-augmented_validat.md)**

:   CiteGuard 提出了一个检索增强的智能体框架，通过扩展的检索动作（包括全文搜索和上下文检索）为科学引用归属提供更忠实的基础，在 CiteME 基准上相对基线提升 10 个百分点，达到 68.1% 准确率，接近人类表现（69.2%）。

**[Code-Switching Information Retrieval: Benchmarks, Analysis, and the Limits of Current Retrievers](information_retrieval/code-switching_information_retrieval_benchmarks_analysis_and_the_limits_of_curre.md)**

:   论文首次系统评估"代码混用查询"对现代 IR 系统的冲击，提出人工标注的 CSR-L 基准和 LLM 生成的 11 任务 CS-MTEB 套件，发现即使 8B 多语言强模型在 query-side code-switching 下也会掉 4–13 个 nDCG@10、reranker 甚至从 60 暴跌到 25；并证明 lexicon-based 词表扩展能缓解但无法补齐单语基线的差距。

**[CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs](information_retrieval/codepromptzip_code-specific_prompt_compression_for_retrieval-augmented_generatio.md)**

:   提出 CodePromptZip，首个面向代码的提示压缩框架，通过类型感知优先级排序构建训练数据并训练带 copy 机制的小模型压缩器，在三个编码任务上分别比最佳基线提升 23.4%、28.7% 和 8.7%。

**[Conjecture and Inquiry: Quantifying Software Performance Requirements via Interactive Retrieval-Augmented Preference Elicitation](information_retrieval/conjecture_and_inquiry_quantifying_software_performance_requirements_via_interac.md)**

:   提出IRAP方法，通过交互式检索增强偏好获取（Interactive Retrieval-Augmented Preference Elicitation）将自然语言描述的软件性能需求量化为数学函数，在4个真实数据集上相比10种SOTA方法取得最高40倍的性能提升，且仅需5轮交互。

**[Context Attribution with Multi-Armed Bandit Optimization](information_retrieval/context_attribution_with_multi-armed_bandit_optimization.md)**

:   本文提出 CAMAB，将 RAG 中的上下文归因（识别哪些上下文片段对生成答案有贡献）建模为组合多臂赌博机（CMAB）问题，使用线性 Thompson 采样自适应地探索上下文子集空间，在 HotpotQA、CNN/DM、TyDi QA 上比 SHAP 和 ContextCite 减少最多 30% 的模型查询次数同时匹配或超越归因质量。

**[CORAL: Adaptive Retrieval Loop for Culturally-Aligned Multilingual RAG](information_retrieval/coral_adaptive_retrieval_loop_for_culturally-aligned_multilingual_rag.md)**

:   CORAL 把多语言 RAG 失败重新定位成"retrieval condition misalignment"——不仅要改写 query，更要动态切换检索的语料库——通过 planner + critic 两个 agent 形成"选语料 → 检索 → 评分过滤 → 充分性检查 → 改语料 + 改 query"的闭环，在两个文化基准上对低资源语言相对最强 baseline 提升 3.58pp，对 CLIcK 韩国文化 QA 提升 3.91pp。

**[CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering](information_retrieval/counterrefine_answer-conditioned_counterevidence_retrieval_for_inference-time_kn.md)**

:   本文提出 CounterRefine，一个轻量级推理时修复层：先用标准 RAG 产生初步答案，再通过答案条件化的反证检索收集支持/反对证据，最后通过受限的 KEEP/REVISE 决策和确定性验证修复错误答案，在 SimpleQA 上将 GPT-5 的正确率从 67.3% 提升至 73.1%。

**[CRAFT: Training-Free Cascaded Retrieval for Tabular QA](information_retrieval/craft_training-free_cascaded_retrieval_for_tabular_qa.md)**

:   本文提出 CRAFT，一个无需数据集特定训练的三阶段级联表格检索框架（SPLADE 稀疏过滤 → 语义 mini-table 排序 → 神经重排序），通过 Gemini 生成的表格标题和描述增强表格表示，在 NQ-Tables 上达到 SOTA（R@1 49.84），在 OTT-QA 上展现强零样本泛化能力，且对查询改写具有显著鲁棒性。

**[Disco-RAG: Discourse-Aware Retrieval-Augmented Generation](information_retrieval/disco-rag_discourse-aware_retrieval-augmented_generation.md)**

:   作者提出 Disco-RAG，把修辞结构理论（RST）显式注入 RAG pipeline——对每个 chunk 解析 intra-chunk RST 树（局部层级）+ 跨 chunk 构建 inter-chunk 修辞图（全局连贯）+ 生成 discourse-aware blueprint 引导回答，在 Loong / ASQA / SciNews 三个长文档基准上 training-free 即拿下 SOTA（Loong overall +12.74 LLM Score）。

**[Domain-Specific Data Generation Framework for RAG Adaptation](information_retrieval/domain-specific_data_generation_framework_for_rag_adaptation.md)**

:   本文提出 RAGen，一个可扩展的模块化数据生成框架，通过文档级概念提取、多块证据组装和 Bloom 分类学引导的问题生成，自动合成领域特定的 QAC（问题-答案-上下文）数据，支持嵌入模型对比微调和 LLM 监督微调，在三个领域数据集上显著优于 AutoRAG 和 LlamaIndex 基线。

**[DQA: Diagnostic Question Answering for IT Support](information_retrieval/dqa_diagnostic_question_answering_for_it_support.md)**

:   本文提出DQA框架，通过维护持久化的诊断状态和在根因层面聚合检索证据（而非逐文档处理），实现企业IT支持场景下的系统化故障排查，成功率从基线41.3%提升至78.7%，平均轮次从8.4降至3.9。

**[End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](information_retrieval/end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)**

:   本文提出 MHGPO（Multi-Agent Heterogeneous Group Policy Optimization），一种无需 critic 的多智能体 RL 方法，通过异构组相对优势估计和反向奖励传播，在三智能体搜索系统（Rewriter→Reranker→Answerer）中实现端到端优化，捕获隐式跨智能体依赖和跨轨迹关联，在 HotpotQA 等多跳 QA 基准上显著优于 MAPPO 和 GRPO 基线。

**[Enhancing Factuality through Consensus and Consistency in Summarization Using Minimum Bayes Risk Decoding](information_retrieval/enhancing_factuality_through_consensus_and_consistency_in_summarization_using_mi.md)**

:   本文提出 ConSUM，在摘要生成候选中同时考察候选对源文档的事实一致性和候选之间的共识，用 MBR decoding 结合 FENICE/FIZZ 等事实性指标重排序，在 CNN/DailyMail、XSum 和人工评估中提升摘要事实可靠性。

**[Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](information_retrieval/enhancing_llm-based_search_agents_via_contribution_weighted_group_relative_polic.md)**

:   CW-GRPO 将过程监督重新定义为"优势重分配"：用 LLM 判断器评估每轮搜索的检索有用性和推理正确性，计算贡献分数来缩放基于结果的优势，实现轮级别信用分配而不引入不稳定的价值函数，在 Qwen3-8B 上超越标准 GRPO 5.0%。

**[Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](information_retrieval/enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)**

:   本文发现多语言 RAG 系统中"英语偏好"主要是评估基准中结构性先验（gold 证据集中于英语、文化先验）的伪影而非模型固有偏差，提出去偏语言偏好指标 DeLP 揭示检索器实际偏好单语对齐，并基于此设计 DELTA 查询增强框架，在多语言 RAG 上一致超越英语枢轴策略。

**[eTracer: Towards Traceable Text Generation via Claim-Level Grounding](information_retrieval/etracer_towards_traceable_text_generation_via_claim-level_grounding.md)**

:   eTracer 把 RAG 响应拆成原子 claim 再去上下文里搜支持/反驳的句级证据，用三步流水线（分解 → 嵌入检索 → 蕴含判定）输出带符号分数矩阵，从而在生物医学场景下既能精确反查每条事实的出处、又能定量评估响应的忠实度。

**[Feedback Adaptation for Retrieval-Augmented Generation](information_retrieval/feedback_adaptation_for_retrieval-augmented_generation.md)**

:   本文提出"反馈适应"作为RAG系统的新问题设定——研究纠正性反馈多快、多有效地传播到未来查询，定义了纠正延迟和反馈后性能两个评估轴，并提出PatchRAG作为免训练的推理时反馈整合方案，实现即时纠正和强泛化。

**[FinRAG-12B: A Production-Validated Recipe for Grounded Question Answering in Banking](information_retrieval/finrag-12b_a_production-validated_recipe_for_grounded_question_answering_in_bank.md)**

:   Kasisto 团队基于 Gemma 3 12B-IT，用 143M token 的数据高效配方 (LLM-as-Judge 过滤 + 引用标注 + 22% 不可回答样本 + 两阶段 curriculum) 训出 FinRAG-12B，并通过 W4A16 量化压到 8.4 GB 单卡部署——答案质量 (JudgeLM 6.21) 和引用质量 (73.1) 都超过 GPT-4.1，refusal 比例 12% 介于 base 的不安全 4.3% 和 GPT-4.1 的过度拒绝 20.2% 之间，在 40+ 家金融机构上线后查询解决率显著提升 +7.1pp ($p<0.001$)，延迟和成本相比 GPT-4.1 分别便宜 3–5× 和 20–50×。

**[FLARE: Task-Agnostic Embedding Model Evaluation via Normalizing Flows](information_retrieval/flare_task-agnostic_embedding_model_evaluation_through_a_normalization_process.md)**

:   提出FLARE框架，利用正则化流（Normalizing Flows）进行无标签的文本嵌入模型评估，通过直接从对数似然估计信息充分性来避免基于距离的密度估计在高维空间中的崩溃，在11个数据集上与有监督基准的Spearman $\rho$ 达0.90。

**[From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](information_retrieval/from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)**

:   本文提出AuthGR，首个将文档权威性系统性整合到生成式检索中的框架，通过VLM多模态权威评分、三阶段渐进式训练（CPT→SFT→GRPO）和混合集成部署管线，在Naver商业搜索引擎的大规模A/B测试中验证了显著的用户参与度提升。

**[GIFT: Guided Fine-Tuning and Transfer for Enhancing Instruction-Tuned Language Models](information_retrieval/gift_guided_fine-tuning_and_transfer_for_enhancing_instruction-tuned_language_mo.md)**

:   GIFT 让 instruction-tuned 模型不再只是最终 merge 的被动目标，而是先用它给训练 token 标注置信度，再用这些置信度指导 base model 的 LoRA 微调，最后把 adapter 合并回 instruction model，在数学、医学和指令任务上稳定超过直接微调与 Shadow-FT 等迁移基线。

**[GLIER: Generative Legal Inference and Evidence Ranking for Legal Case Retrieval](information_retrieval/glier_generative_legal_inference_and_evidence_ranking_for_legal_case_retrieval.md)**

:   本文提出 **GLIER**：把法律案例检索（LCR）从"直接文本相似度匹配"重写为"先用 seq2seq 联合生成 *罪名 + 构成要件* 这一隐变量、再用多视图（生成置信 + 结构匹配 + 词项 BM25）MLP 融合"的两阶段框架，在 LeCaRD/LeCaRDv2 上超越 SAILER、KELLER，且只用 10% 数据训练就能击败强基线的全量结果。

**[How Large Language Models Balance Internal Knowledge with User and Document Assertions](information_retrieval/how_large_language_models_balance_internal_knowledge_with_user_and_document_asse.md)**

:   本文跳出"参数知识 vs 单一外部源"的二元冲突范式，提出"参数 / 用户主张 / 文档主张"三源交互评测框架，在 27 个 LLM × 2 数据集上发现：大多数模型对文档比对用户更轻信，后训练进一步强化这一偏好，且大部分模型属于"impressionable"——分不清外部信息是帮还是害。

**[How Retrieved Context Shapes Internal Representations in RAG](information_retrieval/how_retrieved_context_shapes_internal_representations_in_rag.md)**

:   本文从隐藏表示的角度系统分析 RAG 中检索文档如何影响 LLM 内部状态，发现了五个关键模式：随机文档引发大表示漂移并触发拒绝行为、相关文档主要确认而非改变参数化知识、单个相关文档能锚定多文档场景中的表示、后层逐步强调参数化知识从而限制检索证据的影响、以及 LLM 在早期层就能区分随机文档但到最后层仍无法可靠区分干扰文档和相关文档。

**[Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](information_retrieval/hybrid-vector_retrieval_for_visually_rich_documents_combining_single-vector_effi.md)**

:   HEAVEN 提出了一种即插即用的两阶段混合向量框架，通过视觉摘要页（VS-Pages）加速单向量粗检索 + 基于词性的查询 token 过滤减少多向量重排序计算，在四个基准上保持 99.87% 的多向量 Recall@1 同时减少 99.82% 的每查询 FLOPs。

**[HyperMem: Hypergraph Memory for Long-Term Conversations](information_retrieval/hypermem_hypergraph_memory_for_long-term_conversations.md)**

:   HyperMem 用"超图（hyperedge 连接 ≥3 个节点）"代替传统 RAG 的 pairwise 边，把长期对话记忆组织成"主题 → 情节 → 事实"三层结构，通过粗到细检索 + 超图嵌入传播解决多 episode 跨时间相关性的检索碎片化问题，在 LoCoMo benchmark 上 LLM-as-judge 准确率打到 92.73%（前 SOTA 86.49%）。

**[IF-GEO: Conflict-Aware Instruction Fusion for Multi-Query Generative Engine Optimization](information_retrieval/if-geo_conflict-aware_instruction_fusion_for_multi-query_generative_engine_optim.md)**

:   本文把"为多条潜在查询同时优化一篇文档"视为受限多目标优化问题，提出 IF-GEO："先发散后收敛"——先用 LLM 反向挖掘代表性 query 并生成结构化编辑请求，再通过 **优先级×必要性打分 + 去重 + 冲突解决 + Global Revision Blueprint** 把多个互相打架的编辑指令融合成一份可执行的修改蓝图，并配套引入 WCP/DR/WTR 三项 risk-aware 稳定性指标；在 GEO-Bench 上把 objective overall 从 Auto-GEO 的 7.59 推到 11.03，同时把最坏单查询跌幅从 -0.0511 降至 -0.0090。

**[Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](information_retrieval/is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)**

:   本文在四个数据集上从用户意图处理、查询重写、文档精炼和底层 LLM 选择四个维度系统对比了 Enhanced RAG 和 Agentic RAG，发现两者各有优势——Agentic RAG 在意图路由和查询重写上更灵活，Enhanced RAG 在文档重排上更有效，而 Agentic RAG 的成本高达 3.3 倍。

**[Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](information_retrieval/language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)**

:   本文提出 LcRL 框架，通过语言耦合的 GRPO 策略优化和反一致性惩罚奖励，解决多语言 RAG 中的知识偏差和知识冲突问题，在多语言问答任务上取得显著提升。

**[Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](information_retrieval/learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)**

:   提出 EviOmni，通过"先推理再提取"的范式学习从检索文档中提取理性证据：将证据推理和证据提取整合为统一轨迹，用知识 token 掩码避免信息泄露，通过 GRPO 以可验证奖励优化，在 5 个基准上以极高压缩比（~38x）取得优于全文检索的准确率。

**[MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](information_retrieval/mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)**

:   提出 MAB-DQA 框架，将复杂查询分解为多个方面子查询，用多臂老虎机机制（Thompson Sampling）动态评估各方面的重要性并重新分配检索预算，显著提升多模态文档问答的检索精度和回答准确率。

**[MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](information_retrieval/mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)**

:   本文提出 MASS-RAG，一个免训练的多 Agent 综合 RAG 框架，通过 Summarizer/Extractor/Reasoner 三个专门化过滤 Agent 从互补视角处理检索文档，再通过 Synthesis Agent 整合多视角证据或候选答案，在四个基准上持续超越强基线。

**[MM-BizRAG: Rethinking Multimodal Retrieval-Augmented Generation for General Purpose Enterprise Q&A](information_retrieval/mm-bizrag_rethinking_multimodal_retrieval-augmented_generation_for_general_purpo.md)**

:   MM-BizRAG 证明企业多模态 RAG 不能只依赖页面截图和视觉 embedding，而应先按文档结构区分报告与幻灯片，再显式解析文本、表格和图片，并在推理时组装多模态上下文，从而在 SlideVQA、FinRAGBench-V 和内部企业数据上显著超过视觉中心 baseline。

**[More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval](information_retrieval/more_than_efficiency_embedding_compression_improves_domain_adaptation_in_dense_r.md)**

:   这篇论文证明 PCA 向量压缩不只是为了提速，还能作为一种零训练的 dense retriever 域适配方法，其中只用目标域 query 拟合 PCA 在 75.4% 的模型-数据集组合上提升 NDCG@10。

**[MTR-Suite: A Framework for Evaluating and Synthesizing Conversational Retrieval Benchmarks](information_retrieval/mtr-suite_a_framework_for_evaluating_and_synthesizing_conversational_retrieval_b.md)**

:   MTR-Suite 提出一套从 benchmark 审计、对话数据合成到检索评测的完整框架，用 MTR-Eval 诊断标注质量，用 MTR-Pipeline 以约 1/400 人工成本生成高难度多轮检索基准 MTR-Bench。

**[Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search](information_retrieval/multi-faceted_self-consistent_preference_alignment_for_query_rewriting_in_conver.md)**

:   本文提出 MSPA-CQR，通过从改写、检索、回复三个维度构建自一致性偏好数据，并使用前缀引导的多维度 DPO 优化来训练查询改写模型，在分布内外场景均显著超越现有方法。

**[Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA](information_retrieval/navigating_large-scale_document_collections_mudabench_for_multi-document_analyti.md)**

:   这篇论文提出 MuDABench，把多文档问答从“找几个相关片段回答问题”推进到“在大规模半结构化文档集合上做抽取、聚合和定量分析”，并证明普通 RAG 即使扩大召回也很难完成这类任务，而元数据感知的多 Agent 工作流能显著提高结果但仍远落后于人类专家。

**[Optimizing User Profiles via Contextual Bandits for Retrieval-Augmented LLM Personalization](information_retrieval/optimizing_user_profiles_via_contextual_bandits_for_retrieval-augmented_llm_pers.md)**

:   提出 PURPLE 框架，将检索增强 LLM 个性化中的用户画像构建问题建模为上下文老虎机问题，通过 Plackett-Luce 排序模型捕捉记录间依赖关系，以 LLM 对参考回复的 log-likelihood 作为奖励信号，直接优化检索以匹配生成质量。

**[PL-MTEB: Polish Massive Text Embedding Benchmark](information_retrieval/pl-mteb_polish_massive_text_embedding_benchmark.md)**

:   PL-MTEB 为波兰语文本嵌入构建了覆盖分类、聚类、句对分类、检索和语义相似度的 30 任务评测集，并系统评测 30 个波兰语和多语言 embedding 模型，显示大模型整体领先但任务类型、训练数据泄漏和模型规模都会显著影响结论。

**[Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data](information_retrieval/quantifying_and_improving_the_robustness_of_retrieval-augmented_language_models_.md)**

:   本文提出 SURE 框架，系统评估 RAG 生成端对检索文档中风格、来源、逻辑、格式、元数据等语义无关虚假特征的敏感性，并用 SURE 生成的合成数据通过 SFT/DPO 显著提升 RALM 鲁棒性。

**[RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora](information_retrieval/rare_redundancy-aware_retrieval_evaluation_framework_for_high-similarity_corpora.md)**

:   本文提出 RARE 框架，通过将文档分解为原子事实来追踪跨文档冗余，并设计 CRRF（基于独立准则排序的倒数排名融合）稳定 LLM 多准则判断，在金融/法律/专利等高冗余企业语料上构建了 RedQA 基准，揭示主流检索器在 4-hop 高重叠设置下 PerfRecall@10 从 66.4% 暴跌至 5.0-27.9%。

**[ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](information_retrieval/reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)**

:   ReasonEmbed 提出三项技术创新——ReMixer 非平凡合成数据方法（82K 高质量样本）、Redapter 自适应推理强度加权训练和多骨干实现——在 BRIGHT 基准上以 38.1 的 nDCG@10 显著超越所有现有文本嵌入模型约 10 个点。

**[Reliable Evaluation Protocol for Low-Precision Retrieval](information_retrieval/reliable_evaluation_protocol_for_low-precision_retrieval.md)**

:   揭示低精度（如二值化/量化嵌入）检索系统在评估时因分数粒度降低产生大量虚假并列（spurious ties），导致评估结果高度不稳定，提出 HPS（高精度打分）和 TRM（并列感知指标）两种互补策略，使低精度检索的评估更可靠一致。

**[Rerank Before You Reason: Analyzing Reranking Tradeoffs through Effective Token Cost in Deep Search Agents](information_retrieval/rerank_before_you_reason_analyzing_reranking_tradeoffs_through_effective_token_c.md)**

:   本文系统研究了深度搜索智能体中 listwise 重排序的效率-效果权衡，提出 Effective Token Cost (ETC) 指标，发现中等深度重排序通常比增加搜索时推理预算更具成本效益，在更低 token 开销下达到相当甚至更高的端到端准确率。

**[Retrieval-Augmented Tutoring for Algorithm Tracing and Problem-Solving in AI Education](information_retrieval/retrieval-augmented_tutoring_for_algorithm_tracing_and_problem-solving_in_ai_edu.md)**

:   本文提出 KITE，一个面向算法追踪和问题求解的课程材料 RAG 辅导系统，通过意图感知的苏格拉底式反馈和多阶段检索，在自动指标、模拟学生和专家评审中显示出较好的 grounding 与教学支架效果。

**[Retrieve Only Relevant Tables Whether Few or Many: Adaptive Table Retrieval Method](information_retrieval/retrieve_only_relevant_tables_whether_few_or_many_adaptive_table_retrieval_metho.md)**

:   这篇论文提出 Adaptive Table Retrieval (ATR)，用查询自适应阈值代替固定 top-k 表格检索，并结合相关性校准、表间语义分组和滑动窗口重排，在 Spider、BIRD、Spider 2.0 上同时提升检索召回、text-to-SQL 执行准确率和推理效率。

**[REZE: Representation Regularization for Domain-adaptive Text Embedding Pre-finetuning](information_retrieval/reze_representation_regularization_for_domain-adaptive_text_embedding_pre-finetu.md)**

:   REZE 在领域 embedding 预微调中对 anchor-positive 关系表示做 eigenspace 分解，用鲁棒统计识别任务特异偏移并软收缩，从而吸收共享领域知识、抑制异构任务带来的表示漂移。

**[RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine](information_retrieval/ritek_a_dataset_for_large_language_models_complex_reasoning_over_textual_knowled.md)**

:   RiTeK 构建了两个大规模医学文本知识图谱（TKG）和对应的复杂推理 QA 数据集，涵盖 6 种拓扑结构和丰富的文本描述，评估了 11 种检索方法并揭示了现有 LLM 驱动检索系统在医学 TKG 推理上的严重不足。

**[S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA](information_retrieval/s2g-rag_structured_sufficiency_and_gap_judging_for_iterative_retrieval-augmented.md)**

:   S2G-RAG 把迭代 RAG 中“证据够不够”和“下一步缺什么”显式建模成结构化控制器 S2G-Judge，再用 gap-guided query 和句子级证据抽取减少噪声，在 HotpotQA BM25 设置下把 F1 从 SIM-RAG 的 43.3 提升到 56.5。

**[Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](information_retrieval/sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)**

:   本文提出 Prune-then-Merge，一个两阶段的免训练多向量文档压缩框架——先通过自适应注意力剪枝移除低信息 patch，再对剩余高信号 patch 进行层次聚类合并，在 29 个 VDR 数据集上将近无损压缩范围从 50-60% 扩展到 60-70%，并在 80%+ 高压缩率下显著优于单阶段方法。

**[Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](information_retrieval/stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)**

:   揭示 RAG 系统对检索文档排列顺序高度敏感的问题，提出 Stable-RAG：通过对文档排列产生的隐状态做谱聚类识别主导推理模式，再用 DPO 对齐将幻觉输出引导向正确答案，在三个 QA 数据集上实现准确率和推理一致性的双重提升。

**[Test-Time Training for Zero-Resource Dense Retrieval Reranking](information_retrieval/test-time_training_for_zero-resource_dense_retrieval_reranking.md)**

:   提出 DART，通过在推理时用双线性矩阵自适应地调整密集检索器的评分函数，利用检索结果本身作为伪标签实现零样本无标注重排序，在 BEIR 基准上平均提升 2.1% NDCG@10，延迟控制在 10ms 以内。

**[多语言检索中的低资源语言困境：Amharic 语言证据](information_retrieval/the_multilingual_curse_at_the_retrieval_layer_evidence_from_amharic.md)**

:   本文以 Amharic 语言为诊断案例，揭示了强势多语言检索模型在零样本设置下无法有效迁移到形态丰富的低资源语言，相对 MRR@10 性能下降 23%；而语言特定微调虽能改进 32-60%，但仍无法达到单语言检索器水平，表明多语言检索不足以保证低资源语言的信息公平获取。

**[UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval](information_retrieval/unite_uncertainty-based_iterative_document_sampling_for_domain_adaptation_in_inf.md)**

:   UnIte 把神经检索器的无监督领域适配瓶颈从“生成更多伪查询”转向“更聪明地选文档”，先用 aleatoric uncertainty 过滤低密度噪声文档，再用随模型训练动态变化的 epistemic uncertainty 迭代采样高价值文档，在 BEIR 大语料上用更少伪查询稳定超过 DUQGen。

**[Utility-Oriented Visual Evidence Selection for Multimodal Retrieval-Augmented Generation](information_retrieval/utility-oriented_visual_evidence_selection_for_multimodal_retrieval-augmented_ge.md)**

:   本文把多模态 RAG 的图像选择从“语义相似度排序”改成“对最终回答是否有用”的效用估计，并用轻量多模态代理模型高效预测证据 helpfulness，在 MRAG-Bench 和 Visual-RAG 上同时提升回答质量和推理效率。

**[Verbal-R3: Verbal Reranker as the Missing Bridge between Retrieval and Reasoning](information_retrieval/verbal-r3_verbal_reranker_as_the_missing_bridge_between_retrieval_and_reasoning.md)**

:   Verbal-R3 把传统 reranker 从“只给相关性分数”的模块升级为“给分数并生成解释性 Verbal Annotation”的桥接模块，再用它训练和引导 RAG 推理器，在多跳问答中同时提升答案准确率和测试时扩展效率。

**[VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](information_retrieval/videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)**

:   VideoStir 提出了一种结构化且意图感知的长视频 RAG 框架，通过将视频建模为时空图进行多跳 clip 检索 + 训练意图相关性评分器进行帧级筛选，在不依赖辅助文本工具的前提下达到了与 SOTA 长视频 RAG 方法可比的性能。

**[VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](information_retrieval/visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)**

:   本文提出 Visualize-then-Retrieve (VisRet)，一种将文本查询先通过 T2I 生成模型可视化为图像、再在图像模态内进行检索的新范式，在四个基准上平均提升 nDCG@30 0.125（CLIP）和 0.121（E5-V），下游 VQA 准确率在 Visual-RAG-ME 上提升 15.7%。

**[生物医学 RAG 中检索何时无效：大规模实证研究](information_retrieval/when_retrieval_doesnt_help_a_large-scale_study_of_biomedical_rag.md)**

:   这项大规模实证研究横跨 5 个模型、10 个数据集、4 种检索方法和 4 个检索库，发现生物医学 RAG 仅带来 1-2 个点的小幅不稳定改进，真正的瓶颈是模型有效利用检索证据的能力而非检索质量本身。

**[Why Mean Pooling Works: Quantifying Second-Order Collapse in Text Embeddings](information_retrieval/why_mean_pooling_works_quantifying_second-order_collapse_in_text_embeddings.md)**

:   本文指出 mean pooling 理论上会丢失 token embedding 的二阶结构，并提出 SOCM 指标量化这种二阶坍缩；实验证明现代对比微调文本编码器的 token embedding 更集中，因此比基座模型更不容易发生坍缩，且低 SOCM 与更高 MTEB 表现相关。

**[Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths](information_retrieval/why_these_documents_explainable_generative_retrieval_with_hierarchical_category_.md)**

:   提出 HyPE 框架，在生成式检索中通过先生成层级类别路径（如 "Government >> Government by cities"）再解码文档标识符，为检索结果提供查询相关的可解释路径，同时提升检索准确率。

---

## 🎵 音频/语音 { #audio_speech }

**[Affectron: Emotional Speech Synthesis with Affective and Contextually Aligned Nonverbal Vocalizations](audio_speech/affectron_emotional_speech_synthesis_with_affective_and_contextually_aligned_non.md)**

:   本文提出 Affectron 框架，通过情感驱动的 Top-K NV 匹配和情感感知的 Top-K 路由两个训练时增强策略，在小规模开源解耦语料上实现了多样且情感对齐的非语言发声（如笑声、叹息）合成，显著超越了基于纯语言预训练的 VoiceCraft 基线。

**[An Exploration of Mamba for Speech Self-Supervised Models](audio_speech/an_exploration_of_mamba_for_speech_self-supervised_models.md)**

:   首次全面探索Mamba架构作为语音自监督学习（SSL）基础模型的潜力，发现Mamba-based HuBERT在长上下文ASR、流式ASR和因果设置的probing任务中优于Transformer，同时保持线性时间复杂度。

**[Analyzing Reasoning Shifts in Audio Deepfake Detection under Adversarial Attacks: The Reasoning Tax versus Shield Bifurcation](audio_speech/analyzing_reasoning_shifts_in_audio_deepfake_detection_under_adversarial_attacks.md)**

:   本文为带推理链的音频语言模型（ALM）做深度伪造检测设计了"三维取证审计"框架（声学感知 / 认知一致性 / 认知失调），发现 CoT 推理并非普适增强——对声学感知强的模型（Qwen2-Audio）是"防护盾"，对感知弱的模型（Gemma-3n、Phi-4）反而是"推理税"；且当模型被攻破时，高认知失调可作为"无声警报"提醒人类审计员。

**[Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](audio_speech/anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)**

:   本文提出锚定循环生成（ACG）范式，通过在自回归过程中用已确认的音乐内容作为锚点来校准生成方向，有效缓解长序列符号音乐生成中的误差累积问题，并构建了层次化框架Hi-ACG实现从全局到局部的音乐生成。

**[[b] = [d] − [t] + [p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](audio_speech/bd-tp_self-supervised_speech_models_discover_phonological_vector_arithmetic.md)**

:   系统性地证明自监督语音模型（S3M）的表示空间中存在线性的音韵特征向量，这些向量满足类似 word2vec 的向量算术关系，且其缩放比例与声学测量呈连续相关性。

**[Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs](audio_speech/beyond_transcription_unified_audio_schema_for_perception-aware_audiollms.md)**

:   揭示当前 AudioLLM 的感知弱点源于 ASR 中心的训练范式（系统性抑制副语言和非语言信息），提出 Unified Audio Schema（UAS）将音频信息结构化为转录、副语言和非语言事件三个维度的 JSON 格式，在 MMSU 基准上感知精度提升 10.9% 同时保持推理能力。

**[Beyond Transcripts: A Renewed Perspective on Audio Chaptering](audio_speech/beyond_transcripts_a_renewed_perspective_on_audio_chaptering.md)**

:   这篇论文系统重构长音频章节分段任务：把评测从依赖 transcript 的文本空间推进到 transcript-invariant 的时间空间，并证明直接用音频表示的 AudioSeg 在 YTSeg 上明显优于文本分段和现有 MLLM 方案。

**[Closing the Modality Reasoning Gap for Speech Large Language Models](audio_speech/closing_the_modality_reasoning_gap_for_speech_large_language_models.md)**

:   本文提出 TARS（Trajectory Alignment for Reasoning in Speech），一个基于强化学习的框架，通过表示对齐和行为对齐两种密集奖励信号，将语音条件下的推理轨迹与文本条件下的推理轨迹对齐，在 7B 规模模型中达到 SOTA，MRR（模态恢复率）接近甚至超过 100%。

**[Comprehensive Benchmarking of Long-Form Speech Generation in Diverse Scenarios](audio_speech/comprehensive_benchmarking_of_long-form_speech_generation_in_diverse_scenarios.md)**

:   本文提出 SwanBench-Speech，用 1,101 个样本、17 类真实下游场景和 7 个自动评测维度系统衡量长文本语音生成，结论是当前模型在内容准确性上已接近可用，但在混响一致性、长程韵律和表达层次上仍明显落后于真实录音。

**[Computational Narrative Understanding for Expressive Text-to-Speech](audio_speech/computational_narrative_understanding_for_expressive_text-to-speech.md)**

:   本文从有声书虚构作品中提取角色直接引语，构建了大规模表达性语音数据集 LibriQuote（5.3K 小时引语 + 12.7K 小时叙述），并用语音动词和副词伪标签标注说话风格，实验表明在 flow-matching 模型上微调可同时提升表达性和可懂度，且 LibriQuote-test 构成了一个具有挑战性的表达性 TTS 基准。

**[ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling](audio_speech/controlaudio_tackling_text-guided_timing-indicated_and_intelligible_audio_genera.md)**

:   本文提出 ControlAudio，一个统一的渐进式扩散建模框架，通过三阶段渐进训练（TTA 预训练→时序控制微调→时序+可懂语音联合训练）和渐进引导采样，在单个扩散模型中实现文本引导、时序精确控制和可懂语音生成三种能力，在时序精度和语音清晰度上显著超越现有方法。

**[Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech](audio_speech/data-efficient_targeted_token-level_preference_optimization_for_llm-based_text-t.md)**

:   针对 LLM-based TTS 中模糊发音（如日语「辛い」既可读 karai 也可读 tsurai）的对齐难题，作者提出 TKTO：先用两个标签对调训练的对比 KTO 模型估计每个 token 的重要度权重 $w_t$，再把 KTO 的 utterance 级 value function 拆到 token 级并加权聚合，实现「无需配对数据 + 自动定位目标 token」的双重升级，把日语发音准确率从 0.668 抬到 0.958（+39%），CER 降 54%。

**[Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps](audio_speech/detecting_hallucinations_in_speechllms_at_inference_time_using_attention_maps.md)**

:   提出四种基于音频注意力的指标（AudioRatio、AudioConsistency、AudioEntropy、TextEntropy），训练轻量级逻辑回归分类器在推理时检测语音大模型（SpeechLLM）的幻觉，在域内数据上 PR-AUC 提升最高达 +0.23。

**[Dial HEALTHDIAL for Advice: A Multilingual and Multi-Parallel Spoken Dialogue Dataset for Knowledge-Grounded Information Seeking](audio_speech/dial_healthdial_for_advice_a_multilingual_and_multi-parallel_spoken_dialogue_dat.md)**

:   HEALTHDIAL 构建了一个包含 4 种 WHO 官方语言、6,000 个多平行健康信息寻求对话和 163 小时真实用户语音的数据集，并基于 ASR、TTS、检索、知识过滤和用户研究建立了多语言 spoken RAG benchmark。

**[Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](audio_speech/do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)**

:   本文通过逐层oracle干预实验揭示了大语音语言模型（LSLM）中语音token表示的结构化冗余层次——浅层编码必要声学细节而深层极度冗余——并提出Affinity Pooling这一免训练的基于相似度的token合并机制，在减少27.48% FLOPs的同时保持竞争力的准确率。

**[DRInQ: Evaluating Conversational Implicature with Controlled Context Variation](audio_speech/drinq_evaluating_conversational_implicature_with_controlled_context_variation.md)**

:   DRInQ 用固定问题表面形式、系统变化上下文的方式构建会话含义评测集，发现 LLM 能生成看似合理的语用场景，却常在推理时过度解读上下文、低于人类判断一致性。

**[DuIVRS-2: An LLM-based Interactive Voice Response System for Large-scale POI Attribute Acquisition](audio_speech/duivrs-2_an_llm-based_interactive_voice_response_system_for_large-scale_poi_attr.md)**

:   DuIVRS-2 将百度地图大规模 POI 属性采集的模块化电话 IVR 系统改造成 LLM 驱动的端到端对话系统，通过 FSM 数据增强、选择式生成和双评估器迭代学习，在生产中达到 83.9% TSR、130ms 平均响应和每天 0.4M 通话能力。

**[Exploration of Perceptual Speech Features for Clinical Decision-Support in Mental Health Care](audio_speech/exploration_of_perceptual_speech_features_for_clinical_decision-support_in_menta.md)**

:   这篇论文提出一个面向心理健康临床辅助的可解释语音分析框架，用感知可理解的声学与语言特征结合 XGBoost、统计检验、SHAP 和 LIME，在压力、抑郁、焦虑、ADHD 等多个数据集上寻找稳定的语音行为线索，而不是追求黑盒端到端诊断。

**[FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](audio_speech/fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)**

:   FC-TTS 用 FACodec 的解耦语音表示作为条件来源，再通过两阶段谱图生成、VQ-VAE 风格编码和条件一致性损失，把零样本 TTS 中原本纠缠在同一 reference 里的音色和说话风格拆成两个可独立控制的输入。

**[From Flat Language Labels to Typological Priors: Structured Language Conditioning for Multilingual Speech-to-Speech Translation](audio_speech/from_flat_language_labels_to_typological_priors_structured_language_conditioning.md)**

:   本文提出 S2ST-Omni 2，把多语言语音翻译中的扁平 language label 改成结构化类型学先验，并在表示、声学调制和 LLM 解码三层注入这些先验，从而在 CVSS-C 上提升 BLEU、ASR-BLEU、COMET 和 BLASER 2.0，尤其改善低资源和类型差异较大的语言。

**[Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](audio_speech/full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)**

:   作者提出 Full-Duplex-Bench-v2，让一个 GPT-Realtime 扮演的 Examiner 通过 WebRTC 与被测全双工模型实时对话，按 Daily/Correction/Entity/Safety 四类任务、Fast/Slow 两种节奏对其打 turn-taking、instruction-following、task-specific 三类分，发现 GPT-Realtime、Moshi、Freeze-Omni 都会随对话推进性能持续下滑，且开源模型在 correction 和 entity tracking 上尤其拉胯。

**[SEPT: Semantically Expanded Prompt Tuning for Audio-Language Models](audio_speech/generalizable_prompt_tuning_for_audio-language_models_via_semantic_expansion.md)**

:   SEPT 通过利用 LLM 生成语义邻居并设计带边距约束的语义扩展损失来正则化提示嵌入空间，显著缓解了音频语言模型（ALM）提示调优中的 Base-New Tradeoff 问题，建立了 ALM 提示泛化的首个系统性评估基准。

**[HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models](audio_speech/halluaudio_a_comprehensive_benchmark_for_hallucination_detection_in_large_audio-.md)**

:   本文提出 HalluAudio，首个大规模跨领域（语音/环境声/音乐）的音频幻觉检测基准，包含 5000+ 人工验证的 QA 对和系统化的对抗性提示设计，通过多维指标（准确率/幻觉率/Yes-No偏差/拒绝率/错误类型）评估主流 LALM，揭示了当前模型在声学锚定、时间推理和音乐属性理解方面的显著缺陷。

**[Hard to Be Heard: Phoneme-Level ASR Analysis of Phonologically Complex, Low-Resource Endangered Languages](audio_speech/hard_to_be_heard_phoneme-level_asr_analysis_of_phonologically_complex_low-resour.md)**

:   本文对两种音系极端复杂的低资源濒危东高加索语言（Archi和Rutul）进行音素级ASR分析，发现音素识别准确率与训练频率呈S型学习曲线关系，许多归因于音系复杂性的错误实际上更多源于数据稀缺。

**[HCFD: A Benchmark for Audio Deepfake Detection in Healthcare](audio_speech/hcfd_a_benchmark_for_audio_deepfake_detection_in_healthcare.md)**

:   本文提出医疗场景下的编解码器伪造语音检测任务 HCFD，构建了首个包含多种临床病理条件（抑郁、阿尔茨海默、构音障碍）的编解码器伪造语音数据集 HCFK，并提出 PHOENIX-Mamba 框架——通过在双曲空间中建模多模式伪造证据原型，在英文抑郁检测上达到 97.04% 准确率。

**[How Tokenization Limits Phonological Knowledge Representation in Language Models and How to Improve Them](audio_speech/how_tokenization_limits_phonological_knowledge_representation_in_language_models.md)**

:   本文用三个音韵 probing 任务（rhyme / G2P / 音节数）证明 BPE 类 subword tokenization 既"粒度太粗"难以捕捉局部音韵，又"边界错位"难以捕捉韵律结构，并提出 STAD 度量 + IPA-augmented 轻量微调，让 Llama3.1-8B 在三个音韵任务全面提升而 GSM8K / MMLU 只掉 1.1% / 0.9%。

**[ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment](audio_speech/immersivetts_environment-aware_text-to-speech_with_multimodal_diffusion_transfor.md)**

:   ImmersiveTTS 用双流 MM-DiT 同时建模转写内容和环境描述，并用 WavLM 与 ATST-Frame 的双教师表征对齐稳定训练，从而在带背景声的 TTS 中提升语音自然度、可懂度和 speech-environment 融合质量。

**[Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages](audio_speech/indic-codecfake_meets_satyam_towards_detecting_neural_audio_codec_synthesized_sp.md)**

:   本文构建了首个多印度语言的 CodecFake 检测基准 ICF，并提出 SATYAM——一个双曲音频大语言模型，通过在双曲空间中用 Bhattacharyya 距离对齐语义和副语言表示再与提示对齐，仅训练 3.75M 参数即达到 98.32% 的检测准确率。

**[Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](audio_speech/jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)**

:   构建 Jamendo-MT-QA，一个包含 36,519 个比较问答对（覆盖 12,173 个音轨对）的多音轨比较音乐问答基准，首次系统评估音频-语言模型在跨音轨比较推理上的能力，揭示现有模型在句子级比较生成上的显著不足。

**[LLM-MC-Affect: LLM-Based Monte Carlo Modeling of Affective Trajectories and Latent Ambiguity for Interpersonal Dynamic Insight](audio_speech/llm-mc-affect_llm-based_monte_carlo_modeling_of_affective_trajectories_and_laten.md)**

:   这篇论文提出 LLM-MC-Affect，把对话中的情感从单点标签改写成由随机 LLM 解码近似的潜在分布，再用均值、方差、互相关和斜率指标分析师生对话里的情感同步与主导关系。

**[MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation](audio_speech/marquis_a_three-stage_pipeline_for_video_retrieval-augmented_generation.md)**

:   MARQUIS 将多视频检索增强文章生成拆成“查询分解与重排检索-校准式结构化证据抽取-带引用文章生成”三阶段，并可用 RLM 控制器做迭代证据管理，在 MAGMaR2026 上把检索 nDCG@10 从 0.195 提升到 0.759，生成侧 Iter-QA-Base 人类评分达到 3.83。

**[MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus](audio_speech/mcga_a_multi-task_classical_chinese_literary_genre_audio_corpus.md)**

:   本文构建了首个面向中国古典文学的大规模（119小时、22000条样本）全版权音频语料库 MCGA，涵盖赋、诗、文、词、曲五大文体和六项语音任务（ASR/S2TT/SEC/SQA/SU/SR），并通过评测 10 个多模态大模型揭示了当前模型在古典文学语音理解上的显著不足。

**[Mind the Pause: Disfluency-Aware Objective Tuning for Multilingual Speech Correction with LLMs](audio_speech/mind_the_pause_disfluency-aware_objective_tuning_for_multilingual_speech_correct.md)**

:   作者提出一个多语言 disfluency 修正流水线：先用 MuRIL 在 token 级标注 fluent/disfluent 标签，再把"原始转录 + token 标签"一起喂给 Llama-3.2-3B / Qwen2.5-3B 做 instruction fine-tuning，关键创新是引入一个**对比损失项**，对生成 disfluent token 的概率显式惩罚（penalize $-\log(1-\sum_v w_v P_\theta(v))$），在 Hindi/Bengali/Marathi 三语种实ASR数据上比无对比 baseline +1.97 BLEU、比 mBART +8.54 BLEU，且 3B 模型在多数 setting 上能匹配甚至超越 GPT-4o。

**[MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](audio_speech/mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)**

:   提出 MTR-DuplexBench，一个针对全双工语音语言模型（FD-SLM）的多轮综合评估基准，通过创新的轮次分割方法解决了全双工对话中轮次边界模糊和上下文不一致的挑战，涵盖对话特性、对话质量、指令遵循和安全性四个维度，实验揭示了现有 FD-SLM 在多轮交互中性能持续衰退的问题。

**[Multimodal In-Context Learning for ASR of Low-Resource Languages](audio_speech/multimodal_in-context_learning_for_asr_of_low-resource_languages.md)**

:   系统研究多模态上下文学习（MICL）能否使语音 LLM 学习未见过的濒危语言，并提出基于 MICL 的假设选择系统，结合声学模型与语音 LLM 的互补优势，在三种濒危语言上显著提升 ASR 性能。

**[Music Audio-Visual Question Answering Requires Specialized Multimodal Designs](audio_speech/music_audio-visual_question_answering_requires_specialized_multimodal_designs.md)**

:   本文作为音乐视听问答（Music AVQA）领域首篇综合综述，系统分析了数据集演进和方法设计，论证了专门的输入处理、时空架构设计和音乐领域知识对该任务至关重要，通用多模态模型不足以应对音乐表演的独特挑战。

**[MSU-Bench: Musical Score Understanding Benchmark](audio_speech/musical_score_understanding_benchmark_evaluating_large_language_models39_compreh.md)**

:   MSU-Bench 是首个针对完整乐谱理解的人工标注基准，包含 150 首作品的 1800 个生成式 QA 对，覆盖四级难度，评估揭示了 LLM/VLM 在乐谱定位和幻觉方面的严重不足，而 ABC 记谱法的文本输入显著缓解了这些问题。

**[Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval](audio_speech/omni-embed-audio_leveraging_multimodal_llms_for_robust_audio-text_retrieval.md)**

:   本文提出 OEA（Omni-Embed-Audio），利用多模态 LLM 作为统一编码器构建检索导向的音频-文本嵌入空间，并引入 User-Intent Queries（UIQ）基准和硬负例区分指标（HNSR/TFR），发现 LLM 主干在 T2T 检索（+22%）和硬负例区分（+4.3%p HNSR@10）上显著优于 CLAP 系列方法。

**[PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding](audio_speech/planrag-audio_planning_and_retrieval_augmented_generation_for_long-form_audio_un.md)**

:   PlanRAG-Audio 将长音频理解改写为“先规划要查哪些模态和时间片，再从结构化音频数据库检索证据”的问题，从而把 60 分钟音频的 LLM 输入从约 115k tokens 降到约 1k tokens，并显著提升说话人计数、事件排序和 speaker-constrained QA。

**[Privacy-preserving Prosody Representation Learning](audio_speech/privacy-preserving_prosody_representation_learning.md)**

:   这篇论文提出一个以 glottal source 为输入的自监督 prosody encoder，通过 F0 说话人归一化和 adversarial speaker loss 减少身份泄露，在 phrase boundary、syllable prominence 和 pitch reconstruction 上优于 raw prosody/HuBERT baseline，同时把 VoxCeleb1 speaker identification accuracy 从 HuBERT 的 0.64 降到 0.14。

**[Protecting Bystander Privacy via Selective Hearing in Audio LLMs](audio_speech/protecting_bystander_privacy_via_selective_hearing_in_audio_llms.md)**

:   提出首个旁观者隐私基准 SH-Bench 和旁观者隐私微调（BPFT）方法，评估和提升音频 LLM 在多说话人环境中仅关注主说话人、拒绝泄漏旁观者信息的能力，BPFT 后 SE 指标比 Gemini 2.5 Pro 高 16%。

**[Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition](audio_speech/pseudo2real_task_arithmetic_for_pseudo-label_correction_in_automatic_speech_reco.md)**

:   本文提出 Pseudo2Real，一种参数空间校正方法，通过在源域中计算真实标签模型与伪标签模型的权重差得到"校正向量"，将其应用于目标域伪标签微调模型以纠正系统性伪标签偏差，在 AfriSpeech-200 的十种非洲口音上最高实现 35% 相对 WER 降低。

**[Reference Games as a Testbed for the Alignment of Model Uncertainty and Clarification Requests](audio_speech/reference_games_as_a_testbed_for_the_alignment_of_model_uncertainty_and_clarific.md)**

:   这篇论文用颜色网格 reference games 检验 VLM 能否把内部不确定性转化为恰当澄清请求，发现即便任务很受控，Qwen2.5-VL 和 GPT-5-mini 也仍存在过度自信、澄清行为不稳定和澄清问题低质量等交互能力缺口。

**[ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis](audio_speech/restyle-tts_relative_and_continuous_style_control_for_zero-shot_speech_synthesis.md)**

:   ReStyle-TTS 通过解耦文本/参考音频 guidance、可连续缩放的风格 LoRA、正交 LoRA 融合和音色一致性优化，让零样本 TTS 不再被参考音频风格锁死，可以相对地调高/调低音高、能量和情绪，同时保持文本可懂度与说话人音色。

**[Retrieving to Recover: Towards Incomplete Audio-Visual Question Answering via Semantic-consistent Purification](audio_speech/retrieving_to_recover_towards_incomplete_audio-visual_question_answering_via_sem.md)**

:   本文提出R2ScP框架，将AVQA中缺失模态处理范式从传统的生成式补全转变为基于检索的恢复，通过跨模态检索和上下文感知自适应净化机制消除检索噪声，在模态不完整场景下显著提升了问答性能。

**[RTCFake: Speech Deepfake Detection in Real-Time Communication](audio_speech/rtcfake_speech_deepfake_detection_in_real-time_communication.md)**

:   RTCFake 构建了约 600 小时面向真实实时通信平台的语音伪造检测数据集，并提出音素引导一致性学习 PCL，使 XLSR+AASIST 在离线、在线、跨平台和未见噪声场景下的平均 EER 从混合训练的 7.33% 降到 5.81%。

**[S2S-Arena: Evaluating Paralinguistic Instruction Following in Speech-to-Speech Models](audio_speech/s2s-arena_evaluating_paralinguistic_instruction_following_in_speech-to-speech_mo.md)**

:   S2S-Arena 提出一个直接在语音模态评测 S2S 模型的 benchmark，用四级副语言交互协议、1,243 条语音样本和 1,001 次 pairwise comparison 揭示当前系统在复杂语气、情绪、说话风格和表达控制上的明显差距。

**[SDiaReward: Modeling and Benchmarking Spoken Dialogue Rewards with Modality and Colloquialness](audio_speech/sdiareward_modeling_and_benchmarking_spoken_dialogue_rewards_with_modality_and_c.md)**

:   SDiaReward 构建了面向多轮语音对话的成对偏好数据集与 ESDR-Bench，并训练端到端语音 reward model，让评测不再只看文本语义，而能同时判断韵律/情感等 modality gap 与自然口语风格的 colloquialness gap。

**[SegTune: Structured and Fine-Grained Control for Song Generation](audio_speech/segtune_structured_and_fine-grained_control_for_song_generation.md)**

:   提出 SegTune，一种基于 Diffusion Transformer 的歌曲生成框架，通过层次化文本条件（全局 + 分段级提示）和 LLM 时长预测器实现对歌曲结构和音乐属性的细粒度时序控制。

**[Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling](audio_speech/semi-supervised_diseased_detection_from_speech_dialogues_with_multi-level_data_m.md)**

:   本文提出一种纯音频的半监督学习框架，通过在会话级、片段级和帧级三个层次联合建模临床对话中的病理语音特征，利用 EMA 教师-学生网络动态生成高质量伪标签，在抑郁症和阿尔茨海默症检测中仅用 11 个标注样本即可达到全监督 90% 的性能。

**[SN-WER: Script-Normalized WER for Multi-Script Indic ASR Evaluation](audio_speech/sn-wer_script-normalized_wer_for_multi-script_indic_asr_evaluation.md)**

:   提出 Script-Normalized WER (SN-WER)，一种无需训练的评估方法，通过将参考文本和假设文本音译到统一规范文字后再计算 WER，分离多文字 ASR 评估中的文字不匹配误差与真实识别误差。

**[SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](audio_speech/speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)**

:   SpeakerSleuth 构建了首个评估 LALM 多轮对话说话人一致性判断能力的基准（1,818 实例），系统评测 12 个 LALM 和 6 种嵌入方法后发现：模型在检测和定位声学不一致时表现挣扎，存在严重的文本优先于声学的模态偏差，但在比较/排序声学变体时表现较好。

**[Speculative End-Turn Detector for Efficient Speech Chatbot Assistant](audio_speech/speculative_end-turn_detector_for_efficient_speech_chatbot_assistant.md)**

:   论文构建首个公开 end-turn detection 数据集 OpenETD，并提出 SpeculativeETD，让端侧 GRU 持续检测 speaking/non-speaking，只有遇到 200 ms 静音时才调用服务端 Wav2Vec2 区分 Gap 与 Pause，从而在真实语音上以 38 倍更低 FLOPs 和亚毫秒端侧延迟换取接近大模型的实时 turn-taking 效果。

**[Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception](audio_speech/speech-hands_a_self-reflection_voice_agentic_approach_to_speech_recognition_and_.md)**

:   提出 Speech-Hands，一个可学习的语音代理框架，通过在推理时生成显式动作 token（<internal>/<external>/<rewrite>）来决定信任自身感知还是外部 ASR 假设，在 OpenASR 排行榜 7 个基准上平均 WER 降低 12.1%，在音频 QA 上达到 77.37% 准确率。

**[SpeechLLM-as-Judges: Towards General and Interpretable Speech Quality Evaluation](audio_speech/speechllm-as-judges_towards_general_and_interpretable_speech_quality_evaluation.md)**

:   这篇论文把语音质量评估从“给一个分数”扩展为“可解释的语音评审”，构建了含 32,207 条多语音频和 128,754 条标注的 SpeechEval 数据集，并用 CoT 指令微调与 GRPO 训练出 SQ-LLM，在质量评分、成对比较、改进建议和深伪检测四类任务上整体优于现有语音大模型与专家模型。

**[Standard-to-Dialect Transfer Trends Differ across Text and Speech: A Case Study on Intent and Topic Classification in German Dialects](audio_speech/standard-to-dialect_transfer_trends_differ_across_text_and_speech_a_case_study_o.md)**

:   这篇论文用德语-巴伐利亚语意图分类和德语-瑞士德语主题分类系统比较文本、语音、ASR级联三种迁移路径，发现标准语上的最佳方案不一定适合方言：文本模型最适合标准德语，而语音模型在方言输入上通常更稳。

**[Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions](audio_speech/still_between_us_evaluating_and_improving_voice_assistant_robustness_to_third-pa.md)**

:   针对语音助手无法区分第三方打断（TPI）与主用户发言的问题，提出包含88K训练实例的TPI-Train数据集和TPI-Bench评测框架，通过说话人感知的困难负样本挖掘策略消除语义捷径学习，使模型真正依赖声学线索进行打断检测。

**[StressTest: Can YOUR Speech LM Handle the Stress?](audio_speech/stresstest_can_your_speech_lm_handle_the_stress.md)**

:   提出 StressTest 基准评估语音语言模型（SLMs）对句子重音含义的理解能力，发现现有模型几乎无法基于重音模式推理说话者意图，并通过合成数据管线 Stress-17k 训练的 StresSLM 在重音检测和推理任务上大幅超越前沿模型。

**[Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models](audio_speech/style_amnesia_investigating_speaking_style_degradation_and_mitigation_in_multi-t.md)**

:   发现口语语言模型（SLMs）在多轮对话中无法维持初始指定的说话风格（情感、口音、音量、语速），称之为"风格遗忘"现象，并通过注意力分析揭示其成因（注意力衰减），提出显式回忆过程作为缓解手段。

**[TellWhisper: Tell Whisper Who Speaks When](audio_speech/tellwhisper_tell_whisper_who_speaks_when.md)**

:   本文提出TellWhisper，通过设计时间-说话人感知的旋转位置编码（TS-RoPE）将说话人身份和时间信息统一编码到语音编码器的自注意力中，配合双曲空间说话人日志模型（Hyper-SD），实现了对"谁在何时说了什么"的联合建模，在多说话人ASR任务上取得最优性能。

**[Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](audio_speech/temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)**

:   提出 TCD，一种无训练的推理时解码方法：通过对比原始音频和时间模糊慢速路径的 logits 差异，配合稳定性引导的模糊窗口和不确定性门控，使统一音频语言模型更好地利用瞬态声学线索，在 MMAU 和 AIR-Bench 上一致提升。

**[Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training](audio_speech/towards_fine-grained_and_multi-granular_contrastive_language-speech_pre-training.md)**

:   本文提出FCaps大规模数据集（47k小时语音、19M细粒度标注）和CLSP对比学习模型，通过端到端标注管线和细粒度多粒度对比监督，实现了首个能统一表征全局和细粒度语音风格的语音-文本对齐模型。

**[UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions](audio_speech/unisonate_a_unified_model_for_speech_music_and_sound_effect_generation_with_text.md)**

:   UniSonate 用统一的 Instruction-Content 表示、动态 SFX token 注入和多阶段课程学习，把文本转语音、文本转音乐和文本转音效放进同一个 flow-matching MM-DiT 中，在 TTS 与 TTM 上达到或超过专用模型，同时在 TTA 上保持可用的音效生成能力。

**[UniSRM：用于细粒度语音评估的统一语音奖励模型](audio_speech/unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)**

:   本文提出 UniSRM，一个统一的语音奖励模型，通过两阶段训练（SFT+GRPO）和推理一致性奖励（RCR）机制，支持从话语级质量到对话级连贯性的多维度、可解释的语音评估，在多个评估任务上显著优于现有方法。

**[UniVocal：统一的语音-歌唱代码混用合成](audio_speech/univocal_unified_speech-singing_code-switching_synthesis.md)**

:   UniVocal 通过精细音高令牌和两阶段课程学习，训练模型从纯文本语义自动推断语音/歌唱切换点，无需显式标签，在新构建的 SCSBench 基准上达到 SOTA 性能。

**[VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models](audio_speech/vapo_end-to-end_slide-enhanced_speech_recognition_with_omni-modal_large_language.md)**

:   本文发现端到端全模态大模型做 SlideASR 时会把幻灯片文字误抄成语音内容，并提出 VAPO 用“先看后听”的结构化推理链和多目标强化学习，把幻灯片文字变成语音识别的语义锚点而不是干扰源。

**[VoxMind: An End-to-End Agentic Spoken Dialogue System](audio_speech/voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)**

:   提出 VoxMind，一个赋予端到端语音对话模型智能体能力的统一框架：通过"Think-before-Speak"机制实现显式推理，结合多智能体动态工具管理架构解耦推理延迟与工具规模，任务完成率从基线 34.88% 提升至 74.57%，超越 Gemini-2.5-Pro。

**[When Misinformation Speaks and Converses: Rethinking Fact-Checking in Audio Platforms](audio_speech/when_misinformation_speaks_and_converses_rethinking_fact-checking_in_audio_platf.md)**

:   本文为Position Paper，论证音频平台上的虚假信息在本质上不同于文本虚假信息——它同时具有口语性（prosody、pacing、emotion）和对话性（多轮、多说话人、跨集节），现有以文本为中心的事实核查流水线无法有效处理，需要围绕音频特有属性重新设计验证框架。

**[XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection](audio_speech/xlsr-mambo_scaling_the_hybrid_mamba-attention_backbone_for_audio_deepfake_detect.md)**

:   提出 XLSR-MamBo 框架，系统探索 Mamba-Attention 混合架构在音频深度伪造检测中的四种拓扑设计和多种 SSM 变体（Mamba2、Hydra、GDN），其中 MamBo-3-Hydra 利用 Hydra 的原生双向建模达到多个基准上的竞争性能，且增加骨干深度可有效缓解浅层模型的性能不稳定。

**[ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching](audio_speech/zipvoice-dialog_non-autoregressive_spoken_dialogue_generation_with_flow_matching.md)**

:   提出 ZipVoice-Dialog，首个基于流匹配的非自回归零样本对话语音生成模型，通过课程学习策略和说话人轮次嵌入两个简单设计，解决了流匹配直接用于对话场景时的语音不可懂和轮次混乱问题，同时发布了首个大规模开源对话语音数据集 OpenDialog（6.8k 小时）。

---

## 🔬 可解释性 { #interpretability }

**[A Structured Clustering Approach for Inducing Media Narratives](interpretability/a_structured_clustering_approach_for_inducing_media_narratives.md)**

:   提出一个从大规模新闻语料中自动归纳媒体叙事模式的框架，通过联合建模事件因果链和角色（英雄/威胁/受害者）信息，使用角色约束的聚类算法将叙事链组织成语义连贯的叙事模式，在移民和枪支控制两个领域生成了可解释且与框架理论一致的叙事模式。

**[A Systematic Comparison between Extractive Self-Explanations and Human Rationales in Text Classification](interpretability/a_systematic_comparison_between_extractive_self-explanations_and_human_rationale.md)**

:   这篇论文系统比较了 4 个开源指令调优 LLM 在 3 类文本分类任务上生成的抽取式自解释与人类 rationale、后验归因方法之间的差异，发现自解释与人类标注的一致性强烈受文本长度和任务复杂度影响，但在扰动式 faithfulness 评测中，自解释往往能选出对模型预测更关键的 token 子集。

**[AdaptiveK: Complexity-Driven Sparse Autoencoders for Interpretable Language Model Representations](interpretability/adaptivek_complexity-driven_sparse_autoencoders_for_interpretable_language_model.md)**

:   AdaptiveK 提出一种由输入语义复杂度驱动的 Sparse Autoencoder，让简单文本激活更少特征、复杂文本激活更多特征，在 8 个自回归 LLM 和附加架构实验上改善重构质量、概念解耦与训练效率，并减少固定 TopK 需要反复调参的问题。

**[Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](interpretability/aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)**

:   构建大规模Post-hoc Self-Consistency Bank（PSCB，85K决策×428K解释），量化LLM答案与其解释之间的特征归因差距，并通过DPO优化在不损害准确率的前提下提升解释的归因一致性。

**[Compositional Steering of Large Language Models with Steering Tokens](interpretability/compositional_steering_of_large_language_models_with_steering_tokens.md)**

:   本文提出组合引导 token，通过自蒸馏将行为指令压缩为输入空间的嵌入向量，并训练专用组合 token <and> 来捕获"组合"的通用概念，在未见过的行为组合、未见过的行为以及未见过的组合数量上均展现强泛化能力。

**[Constructing Interpretable Features from Compositional Neuron Groups](interpretability/constructing_interpretable_features_from_compositional_neuron_groups.md)**

:   作者用半非负矩阵分解（SNMF）直接把 MLP 激活拆成"稀疏神经元组 × 非负系数"，得到既能映射回激活上下文又能跨层组合的可解释特征，在 Llama-3.1-8B / Gemma-2-2B / GPT-2 上的因果引导（concept steering）评估全面超过最新 SAE（Llamascope / Gemmascope）和强监督基线 DiffMeans。

**[Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining](interpretability/crosscoding_through_time_tracking_emergence_consolidation_of_linguistic_represen.md)**

:   用 sparse crosscoder 在同一 LLM 的多个 pretraining checkpoint 间训练一个共享特征字典，并提出 Relative Indirect Effect (RelIE) 度量逐特征的因果重要性如何在 token 数量推移中"涌现/维持/消失"，从而首次在 Pythia/OLMo/BLOOM 上观察到 LLM 从"特定子词检测器"逐步内化为"抽象句法/跨语言检测器"的概念级演化轨迹。

**[Curing "Miracle Steps" in LLM Mathematical Reasoning with Rubric Rewards](interpretability/curing_miracle_steps_in_llm_mathematical_reasoning_with_rubric_rewards.md)**

:   本文发现当前 LLM 数学推理中存在大量"Miracle Steps"——推理链中凭空跳跃到正确答案的现象，并提出 Rubric Reward Model (RRM)，一种基于问题特定评分标准的过程奖励函数，在 RL 训练中显著减少 Miracle Steps 71% 并将 AIME2024 的 Verified Pass@1024 从 26.7% 提升至 62.6%。

**[Diffusion-CAM: Faithful Visual Explanations for dMLLMs](interpretability/diffusion-cam_faithful_visual_explanations_for_dmllms.md)**

:   提出 Diffusion-CAM，首个专为扩散式多模态大语言模型（dMLLM）设计的可解释性方法，通过在去噪轨迹中提取结构有效的中间表征并配合四个后处理模块（自适应核去噪、分布感知置信门控、上下文背景衰减、单实例因果去偏），在 COCO Caption 和 GranDf 上显著超越自回归 CAM 基线。

**[Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives](interpretability/do_llms_capture_embodied_cognition_and_cultural_variation_cross-linguistic_evide.md)**

:   作者用「this/that」与「这/那」这类指示词（demonstrative）作为探针，构建中英双语对照数据集（80 题/语 × 4 cue × 4 perspective × 5 场景），用 320 名母语者的 6,400 条响应建立人类基线，发现英语者擅长 proximal–distal 区分但弱于他者视角，中文者反之；而 5 个 SOTA LLM 既无法稳定区分近–远，也无跨文化差异，普遍退回到 English-centric 推理或"All of the above"安全 fallback。

**[Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations](interpretability/do_llms_know_tool_irrelevance_demystifying_structural_alignment_bias_in_tool_inv.md)**

:   发现并形式化了 LLM 工具调用中的"结构对齐偏差"——当查询属性可以有效映射到工具参数时（即使工具功能与用户目标无关），LLM 仍倾向调用该工具。构建 SABEval 数据集解耦结构对齐和语义相关性，用对比注意力归因揭示内部存在语义检查和结构匹配两条竞争路径，提出再平衡策略实现 80% 的相对错误减少。

**[DPN-LE: Dual Personality Neuron Localization and Editing for Large Language Models](interpretability/dpn-le_dual_personality_neuron_localization_and_editing_for_large_language_model.md)**

:   这篇论文提出 DPN-LE，通过对比高/低人格特质样本的 MLP 激活来定位互斥的人格相关神经元，只干预约 0.5% 神经元即可实现人格控制，并比既有大规模神经元编辑更好地保留通用能力。

**[Dual Alignment Between Language Model Layers and Human Sentence Processing](interpretability/dual_alignment_between_language_model_layers_and_human_sentence_processing.md)**

:   作者用 logit-lens 把 GPT-2/Pythia/OPT 共 19 个 LM 的每一层都解出"内部 surprisal"，发现一个反直觉的"双重对齐"：在自然阅读语料上**浅层**的 surprisal 最像人；但在 garden-path / NPS / NPZ / RC / Attachment 等**句法挑战句**上反而**深层**才像人，对应人类"shallow 默认 + 困难时切换到 deep 重分析"的双机制阅读模型——并由此提出用浅深层 surprisal 差（KL/JS）作为"层间预测更新量"来当 reading-time 的补充特征。

**[Embracing Anisotropy: Turning Massive Activations into Interpretable Control Knobs for Large Language Models](interpretability/embracing_anisotropy_turning_massive_activations_into_interpretable_control_knob.md)**

:   本文把 LLM 中常被视为异常值的 massive activations 重新解释为可解释的领域关键维度，用无需训练的激活幅值准则识别这些维度，并只在这些维度上做 activation steering，从而在领域适配和 jailbreak 场景中比全维度 steering 更有效。

**[Evian: Towards Explainable Visual Instruction-tuning Data Auditing](interpretability/evian_towards_explainable_visual_instruction-tuning_data_auditing.md)**

:   提出"分解-再评估"（Decomposition-then-Evaluation）范式和 EVIAN 框架，将视觉指令微调数据的回答分解为视觉描述、主观推理和事实声明三个组件，沿图文一致性、逻辑连贯性和事实准确性三个正交维度评估，发现用其筛选的少量高质量数据训练的模型优于大规模数据集训练的模型。

**[Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](interpretability/experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)**

:   构建控制知识框架系统研究LLM在科学可行性评估中如何利用实验描述和结果证据，发现提供结果证据比实验描述更可靠，部分实验信息常导致性能低于仅用参数知识的基线，揭示了LLM推理的脆弱性。

**[Fine-Grained Analysis of Shared Syntactic Mechanisms in Language Models](interpretability/fine-grained_analysis_of_shared_syntactic_mechanisms_in_language_models.md)**

:   论文用 activation patching 在 attention head 粒度上证明：Pythia/Gemma 处理英文 filler-gap 依赖（FGD）时七大构式共享同一套早-中层 3 个 attention head 的机制，把这几个 head 的激活×1.5 还能在 BLiMP 上多对一批题；而负极性项（NPI）授权没有这种统一机制，且训练阶段更易学到的"DAS 方向"在 OOD 上完全失效，说明无监督的 patching 比有监督的 DAS 更可信。

**[FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models](interpretability/finesteer_a_unified_framework_for_fine-grained_inference-time_steering_in_large_.md)**

:   FineSteer 将推理时转向分解为两个互补阶段：子空间引导的条件转向（SCS）决定"何时转向"——用 IR 查询子空间的能量比做门控；混合转向专家（MoSE）决定"如何转向"——通过注意力门控网络动态聚合原型专家+残差精炼生成查询特异性转向向量，在安全和真实性 benchmark 上超越 SOTA。

**[Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs](interpretability/flattery_in_motion_benchmarking_and_analyzing_sycophancy_in_video-llms.md)**

:   作者构建了首个 Video-LLM 谄媚基准 ViSE (367 视频 / 6,367 多选题 / 7 类谄媚场景)，在 9 个 SOTA Video-LLM 上系统揭示"模型为了迎合用户而抛弃视觉证据"的普遍现象，并提出两种 training-free 缓解方法：(i) 关键帧选择降谄媚最高 22.01% (并通过注意力分析证明它消除"首帧偏置"和"中间层不稳定")；(ii) representation steering 在最难场景下平均降 35.69%，在 LLaVA-OneVision 上 5 个类别 MSS 降到接近 0。

**[Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models](interpretability/follow_the_flow_on_information_flow_across_textual_tokens_in_text-to-image_model.md)**

:   本文通过因果干预框架系统研究了文本到图像模型中文本编码器输出的 token 级信息分布，发现词汇项的语义通常集中在 1-2 个代表性 token 上，且跨项信息流在 11% 的情况下会导致语义泄漏和图像错误解读，并提出了简单有效的 token 级干预方法来改善对齐。

**[From Documents to Segments: A Contextual Reformulation for Topic Assignment](interpretability/from_documents_to_segments_a_contextual_reformulation_for_topic_assignment.md)**

:   本文把主题分配的基本单位从 document 改成 segment，提出 SBTA 和 SemEval-STM 数据集，证明在多主题短文本中按语义片段分配主题能显著提升主题纯度、可解释性和下游检索可用性。

**[From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context Language Models](interpretability/from_interpretability_to_performance_optimizing_retrieval_heads_for_long-context.md)**

:   RetMask 把"机制可解释性 (mechanistic interpretability)"找到的 retrieval heads 当成对比信号源 — 用屏蔽 retrieval head 后的 ablated 模型输出作为 rejected 样本、原模型输出作为 chosen 样本，跑 DPO 训练，无需 LLM judge 或人工标注，在 Llama-3.1 / Qwen3 / Olmo-3 三个模型族 128K 长度上一致提升，特别是 generation-with-citation +70% / re-rank +32%。

**[From Weights to Activations: Is Steering the Next Frontier of Adaptation?](interpretability/from_weights_to_activations_is_steering_the_next_frontier_of_adaptation.md)**

:   本文系统性地论证 steering（推理时激活空间干预）应被视为一种独立的模型适配范式，提出八项功能性评估标准对比 steering 与微调、PEFT、提示工程等传统方法，将 steering 定位为基于激活空间的局部可逆行为修改方法，具有计算高效、数据高效和可逆性等独特优势。

**[HistLens: Mapping Idea Change across Concepts and Corpora](interpretability/histlens_mapping_idea_change_across_concepts_and_corpora.md)**

:   提出 HistLens 框架，基于稀疏自编码器（SAE）将概念表示分解为可解释的语义基向量，在共享坐标系中追踪多概念、多语料的历时演化轨迹，支持隐式概念计算，为数字人文和概念史研究提供可量化、可比较的分析工具。

**[How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects](interpretability/how_language_models_conflate_logical_validity_with_plausibility_a_representation.md)**

:   通过表示分析揭示 LLM 中"逻辑有效性"和"合理性"两个概念在隐层空间中高度对齐，导致模型将合理性与有效性混淆（内容效应），并构造去偏转向向量有效解耦这两个概念，减少内容效应同时提升推理准确率。

**[IDEA: An Interpretable and Editable Decision-Making Framework for LLMs via Verbal-to-Numeric Calibration](interpretability/idea_an_interpretable_and_editable_decision-making_framework_for_llms_via_verbal.md)**

:   提出 IDEA 框架，将 LLM 的决策知识提取为语义因子上的可解释参数化模型，通过 EM 算法联合学习语言概率表达到数值的映射和决策参数，实现了可校准、可编辑、可解释的 LLM 决策，在五个数据集上以 Qwen-3-32B (78.6%) 超越 DeepSeek R1 (68.1%) 和 GPT-5.2 (77.9%)。

**[Interpretability from the Ground Up](interpretability/interpretability_from_the_ground_up_stakeholder-centric_design_of_automated_scor.md)**

:   本文从教育评估利益相关者需求出发提出 FGTI 四原则（忠实、扎根、可追溯、可互换），开发 AnalyticScore 三阶段框架实现可解释自动评分，在 ASAP-SAS 上平均 QWK 仅比不可解释 SOTA 低 0.06。

**[Interpretable Coreference Resolution Evaluation Using Explicit Semantics](interpretability/interpretable_coreference_resolution_evaluation_using_explicit_semantics.md)**

:   本文用 Concept and Named Entity Recognition (CNER) 把 29 类细粒度语义标签按"提及 + 簇级多数投票"覆盖到共指消解的输出上，得到按语义类别分层的 Mention F1 和 Link F1 诊断指标，从而看清"模型在哪个类别上系统性失败"，再用这些诊断指导仅 3 篇合成文档的定向数据增强，把 LitBank 训练的模型在 OntoNotes/PreCo 上 CoNLL-F1 提了 +2.5/+2.8，Mention F1 提了约 +9.5。

**[Interpretable Semantic Gradients in SSD: A PCA Sweep Approach and a Case Study on AI Discourse](interpretability/interpretable_semantic_gradients_in_ssd_a_pca_sweep_approach_and_a_case_study_on.md)**

:   本文给 Supervised Semantic Differential (SSD) 这种"用个体差异变量估计文本嵌入语义梯度"的方法提了一个 PCA sweep 程序——用可解释性 + 稳定性两个诊断（而非预测精度）联合挑选 PCA 维度 $K$，并在 349 条 AI 主题短文 + 自恋问卷上演示：sweep 选出的 $K=15$ 给出 Admiration 相关的"乐观协作 vs 不信任嘲讽"稳定语义梯度，而高 $K=120$ 的反事实方案得到混乱难解释的簇。

**[Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](interpretability/interpretable_traces_unexpected_outcomes_investigating_the_disconnect_in_trace-b.md)**

:   通过规则化问题分解方法构建可验证的中间推理链数据集，揭示 CoT 推理链的语义正确性与最终答案准确率不可靠地相关（正确链仅 28% 导致正确答案），且最可解释的推理链并非最提升性能的——冗长的 R1 链性能最优但用户评为最不可解释。

**[Interpreting Style Representations via Style-Eliciting Prompts](interpretability/interpreting_style_representations_via_style-eliciting_prompts.md)**

:   这篇论文把难解释的文本风格向量解码成可直接驱动 LLM 写作的 style-eliciting prompts，用“可控制性”作为解释标准，在风格恢复、合成文本风格控制和人类文本风格模仿上都优于直接让 LLM 描述目标文本风格的基线。

**[Interpreto: An Explainability Library for Transformers](interpretability/interpreto_an_explainability_library_for_transformers.md)**

:   Interpreto 是一个面向 HuggingFace 语言模型的开源 Python 可解释性库，把 token/word/sentence attribution 与 activation-level concept explanations 统一到一个 API 中，并提供 demo、教程、指标和端到端概念解释流水线。

**[Investigating More Explainable and Partition-Free Compositionality Estimation for LLMs: A Rule-Generation Perspective](interpretability/investigating_more_explainable_and_partition-free_compositionality_estimation_fo.md)**

:   论文跳出"造测试集做组合泛化测试"的传统范式，让 LLM 直接为整个数据集生成一段 Python 程序作为映射规则，再用基于 Kolmogorov 复杂度上界的 $\mathcal{C}(\text{P})$ 把"程序的压缩度 + 正确率"折成 0–100 的组合性分数；从而把"看输出对不对"换成"看规则压得多紧"，既绕开了大模型预训练时已经"见过组合"的污染，又给出了可解释的内省式评估。

**[Jacobian Scopes: Token-Level Causal Attributions in LLMs](interpretability/jacobian_scopes_token-level_causal_attributions_in_llms.md)**

:   作者提出 Jacobian Scopes——一套用"输入 token 嵌入到末层 hidden state 的 Jacobian 在某个 vector 上的投影"作为 token 归因强度的统一框架，配套三种 scope（Semantic / Fisher / Temperature）分别解释"某个目标 logit / 整个预测分布 / 模型置信度"如何被各输入 token 驱动，只需 1 次反向传播，AOPC 指标上与 Input×Gradient 持平、显著优于 Integrated Gradients。

**[Knowledge Vector of Logical Reasoning in Large Language Models](interpretability/knowledge_vector_of_logical_reasoning_in_large_language_models.md)**

:   作者证明 LLM 内部的演绎、归纳、溯因三类逻辑推理能力可以被线性表示为三条几乎正交的"知识向量"，并提出一个基于 SAE 子空间约束的互补精炼框架，让这三条向量在保留各自独特特征的同时互相借鉴，从而在 steering 设置下稳定提升三类推理性能。

**[Learning What Matters: Dynamic Dimension Selection and Aggregation for Interpretable Vision-Language Reward Modeling](interpretability/learning_what_matters_dynamic_dimension_selection_and_aggregation_for_interpreta.md)**

:   VL-MDR 把"单标量黑盒"的判别式视觉语言奖励模型升级成"动态选维度 + 各维度打分 + 自适应加权"的三头架构，配合 321k 条带 21 维细粒度偏好标注的数据集，在 VL-RewardBench 上击败现有开源 RM，并能产出更高质量的 DPO 偏好对来缓解 VLM 幻觉。

**[Letting Tutor Personas Speak Up for LLMs: Learning Steering Vectors from Dialogue via Preference Optimization](interpretability/letting_tutor_personas_speak_up_for_llms_learning_steering_vectors_from_dialogue.md)**

:   本文从真实师生对话中学习共享 steering direction 和导师专属缩放系数，让 LLM 在不显式写 persona prompt 的情况下生成更接近特定真人导师风格的辅导话语。

**[Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States](interpretability/linear_probes_detect_task_format_not_reasoning_mode_in_language_model_hidden_sta.md)**

:   这篇论文用 Qwen3-14B 上的探针、残差去混淆、trace-anchor 和 causal steering 实验证明：线性探针看似能 100% 区分演绎、归纳、溯因推理，但实际检测到的是数据源和题目格式，而不是隐藏状态中的推理模式。

**[Lost in Translation? Exploring the Shift in Grammatical Gender from Latin to Occitan](interpretability/lost_in_translation_exploring_the_shift_in_grammatical_gender_from_latin_to_occi.md)**

:   针对中世纪奥克语这种低资源历史语言，作者搭了一套 mBERT + 混合分词 + 域适应 MLM 的可解释框架，把"原拉丁中性名词在奥克语里到底是男性还是女性"这个问题拆成词形线索 vs. 句法上下文两路证据来量化，发现后缀形态贡献最大单一信号、上下文（尤其冠词与形容词）能把宏 F1 从 0.665 推到 0.929。

**[Make Mechanistic Interpretability Auditable: A Call to Develop Guidelines via Continuous Collaborative Reviewing](interpretability/make_mechanistic_interpretability_auditable_a_call_to_develop_guidelines_via_con.md)**

:   这是一篇 position paper，主张机制可解释性研究需要补上“可审计性”这一层，通过连续协作评审平台、社区精炼指南和源证据追踪系统，把零散复现、负结果和方法学批评沉淀成可用于安全关键场景的审计协议。

**[Mechanisms of Prompt-Induced Hallucination in Vision–Language Models](interpretability/mechanisms_of_prompt-induced_hallucination_in_vision-language_models.md)**

:   在受控的目标计数任务里把"模型听 prompt 不看图"的幻觉行为定位到 LLaVA-OneVision / Qwen-VL / Janus-Pro **早期层** (主要是 L0-1) 的 3-10 个 attention head，对它们做 mean ablation 不需要任何再训练就让 prompt-following 从 42–64% 掉到 <11%，把真实计数恢复率推到 70–78%，并能 zero-shot 迁移到颜色识别任务 (PIH 抑制 40–95%)。

**[Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy](interpretability/mechanistic_interpretability_of_large-scale_counting_in_llms_through_a_system-2_.md)**

:   针对 LLM 大数量计数失败（单 forward 因为 layer 深度有限只能数到 ~10–30），用一个简单的 test-time "用 `|` 把列表切片 + 让模型先逐段计数再求和"策略让 Qwen2.5/Llama3/Gemma3/GPT-4o/Gemini-2.5-Pro 在 50–100 物体场景下从 0–20% 准确率跃升到 50–95%，并通过 attention 分析 + 4 类因果中介实验把"分段计数→中间步骤聚合→最终求和"三阶段电路定位到 Qwen2.5-7B 的 Layer 22 (head 13 负责分段, head 1 负责聚合)。

**[METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models](interpretability/meter_evaluating_multi-level_contextual_causal_reasoning_in_large_language_model.md)**

:   METER 是首个在**统一上下文**下系统评测 LLM 三层因果推理（discovery / intervention / counterfactual）的 benchmark（4145 条人审 + LLM 协同构造样本），通过显著性信息流分析发现 LLM 在因果阶梯上升时性能从 93% 跌到 73%——根因是 discovery 阶段被无关事实干扰、higher-level 阶段对上下文 faithfulness 显著下降。

**[MINED: Probing and Updating with Multimodal Time-Sensitive Knowledge for Large Multimodal Models](interpretability/mined_probing_and_updating_with_multimodal_time-sensitive_knowledge_for_large_mu.md)**

:   作者提出 MINED——首个**多模态时间敏感知识**评测基准，包含 2104 条 (subject, hypernym, property, attribute-list) 四元组、6 个维度（Cognition / Awareness / Trustworthiness / Understanding / Reasoning / Robustness）共 11 个子任务，4208 道题，在 15 个 LMM 上跑下来 Gemini-2.5-Pro 拿到最高平均 CEM=63.07 但仍漏掉 ~15% 知识；进一步用 FT-LLM / IKE 等知识编辑方法在 single editing 下能把 LLaVA-v1.5 和 Qwen-VL 的过时知识有效更新，但 lifelong editing 下大幅退化（FT-LLM 平均掉 43.2%）。

**[Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models](interpretability/model_internal_sleuthing_finding_lexical_identity_and_inflectional_features_in_m.md)**

:   本文系统地对 25 个 Transformer 语言模型（从 BERT Base 到 Qwen2.5-7B）进行探针分析，发现词汇同一性（lexeme）在早期层线性可解码但随深度衰减，而屈折特征（inflection）在所有层中保持稳定可读，且占据紧凑可控的子空间。

**[NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning](interpretability/nose_neural_olfactory-semantic_embedding_with_tri-modal_orthogonal_contrastive_l.md)**

:   提出 NOSE 三模态嗅觉表示学习框架，以分子为枢纽通过正交注入机制对齐分子结构、受体序列和自然语言描述三个模态，配合 LLM 驱动的弱正样本策略缓解描述稀疏问题，在 11 个下游任务上达到 SOTA 并展现优秀的零样本泛化能力。

**[On Emergent Social World Models -- Evidence for Functional Integration of Theory of Mind and Pragmatic Reasoning in Language Models](interpretability/on_emergent_social_world_models_--_evidence_for_functional_integration_of_theory.md)**

:   这篇论文用大规模行为评测和受认知神经科学启发的功能定位/消融实验，给出语言模型的 Theory of Mind 与语用推理可能共享内部计算机制的证据，因此把“社会世界模型”从单纯能力得分推进到可检验的功能整合假说。

**[Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization](interpretability/preference_heads_in_large_language_models_a_mechanistic_framework_for_interpreta.md)**

:   这篇论文提出 Preference Heads 与 Differential Preference Steering，用因果消融找出少量承载用户偏好的注意力头，再在解码时放大这些头带来的偏好信号，从而在不改模型参数的情况下提升个性化生成与预测效果。

**[Retrieval Heads are Dynamic](interpretability/retrieval_heads_are_dynamic.md)**

:   本文证明 LLM 中负责从上下文取信息的 retrieval heads 并不是固定集合，而会随生成步动态变化、不可被静态 heads 替代，并可由 hidden states 预测，从而能提升动态 RAG 的检索效果。

**[Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models](interpretability/revitalizing_black-box_interpretability_actionable_interpretability_for_llms_via.md)**

:   本文提出一种基于代理模型的黑盒可解释性框架，利用廉价小模型近似昂贵大模型的局部决策边界来生成 LIME/SHAP 解释，通过统计筛选-应用（screen-and-apply）机制确保可靠性，代理解释在保持超过 90% 保真度的同时将成本降低 88.2%，并成功用于 Prompt 压缩和中毒样本移除等下游优化任务。

**[Rhetorical Questions in LLM Representations: A Linear Probing Study](interpretability/rhetorical_questions_in_llm_representations_a_linear_probing_study.md)**

:   通过线性探针分析 LLM 内部如何表征反问句，发现反问句在表征空间中是线性可分的且可跨数据集迁移，但不同数据集学到的探针方向并不一致——反问句由多个异构的线性方向编码，而非单一统一维度。

**[Similarity-Distance-Magnitude Activations](interpretability/similarity-distance-magnitude_activations.md)**

:   本文提出 SDM（Similarity-Distance-Magnitude）激活函数作为 softmax 的更鲁棒替代，通过将正确预测的深度匹配（Similarity）、到训练分布的距离（Distance）和决策边界距离（Magnitude）三个认知维度解耦并整合为新的激活 $\text{sdm}(\mathbf{z}')_i = (2+q)^{d \cdot z'_i} / \sum_c (2+q)^{d \cdot z'_c}$，并在此基础上构建 SDM 估计器进行选择性分类，在协变量偏移和分布外输入下比现有校准方法更鲁棒。

**[SITE: Soft Head Selection for Injecting ICL-Derived Task Embeddings](interpretability/soft_head_selection_for_injecting_icl-derived_task_embeddings.md)**

:   SITE 提出了一种基于梯度优化的软注意力头选择方法，通过识别任务相关的注意力头来有效注入 ICL 衍生的任务嵌入，在 12 个 LLM（4B-70B）上显著超越 ICL 和现有嵌入方法，同时用远少于 PEFT 的可训练参数达到可比性能。

**[Sparse Feature Coactivation Reveals Causal Semantic Modules in Large Language Models](interpretability/sparse_feature_coactivation_reveals_causal_semantic_modules_in_large_language_mo.md)**

:   论文用少量 prompt 中 SAE 特征的跨层共激活图自动发现 LLM 中表示概念和关系的语义模块，并证明对这些模块进行消融和放大能在最高 98% 的单概念/关系场景、最高 90% 的组合场景中可预测地操控 Gemma 2 2B 的关系推理输出。

**[SSA: Improving Performance With a Better Scoring Function](interpretability/ssa_improving_performance_with_a_better_scoring_function.md)**

:   这篇论文指出 Softmax attention 在分布偏移下会因大幅值 token 产生近似 hardmax 的注意力塌缩，并提出 Scaled Signed Averaging 作为可训练的替代评分函数，在合成 ICL 任务、114M decoder-only 语言模型和 BabyBERTa encoder 探针上都比 Softmax 有更好的泛化表现。

**[Style over Story: Measuring LLM Narrative Preferences via Structured Selection](interpretability/style_over_story_measuring_llm_narrative_preferences_via_structured_selection.md)**

:   本文设计了一种基于约束选择的实验范式来测量 LLM 的叙事偏好，使用叙事学理论构建的 200 个约束库让 6 个 LLM 在不同指令类型下进行选择，发现模型系统性地优先选择"风格"（Style）而非"事件"（Event）、"角色"（Character）和"场景"（Setting）等内容元素。

**[The Impact of Off-Policy Training Data on Probe Generalisation](interpretability/the_impact_of_off-policy_training_data_on_probe_generalisation.md)**

:   这篇论文系统比较 on-policy natural、on-policy incentivised、on-policy prompted 和 off-policy 四类训练数据对 LLM 激活探针泛化的影响，发现文本表面可见行为较稳健，而欺骗、迎合、sandbagging 等“意图型”行为极易受领域偏移影响，并提出用 on-policy incentivised 测试集预判真实监控中的泛化失败。

**[Through a Compressed Lens: Investigating The Impact of Quantization on Factual Knowledge Recall](interpretability/through_a_compressed_lens_investigating_the_impact_of_quantization_on_factual_kn.md)**

:   这篇论文系统评估 GPTQ、AWQ、BitsAndBytes 等权重量化对 LLM 事实知识回忆的影响，发现量化通常会造成信息损失并削弱知识调用，尤其伤害较小模型和未饱和关系，但 8-bit / BitsAndBytes 往往能较好保留能力，个别量化甚至会提升多跳事实回忆。

**[Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](interpretability/towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)**

:   系统综述了 LLM 内在可解释性的最新进展，将现有方法分为五大设计范式（功能透明性、概念对齐、表征可分解性、显式模块化、潜在稀疏归纳），并讨论了开放挑战和未来方向。

**[Tracing Relational Knowledge Recall in Large Language Models](interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)**

:   本文系统研究LLM在文本生成过程中回忆关系知识的内部机制，发现注意力头对残差流的逐头贡献（$\Delta_{att,h}$）是线性关系分类的最强特征（准确率达91%），并提出HeadScore和TokenScore两种探针归因方法来分解预测到注意力头和源token级别，揭示了探针精度与关系特异性、实体连通度及探针信号集中度之间的明确相关性。

**[Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](interpretability/understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)**

:   本文通过受控合成数据集 Biography-Reasoning 系统分析了 SFT 阶段学习新知识导致的事实幻觉现象，发现幻觉的根本机制是模型对关键实体的注意力被削弱，并提出 KnownPatch——在训练末期注入少量已知知识来恢复注意力模式，有效缓解幻觉。

**[Understanding or Memorizing? A Case Study of German Definite Articles in Language Models](interpretability/understanding_or_memorizing_a_case_study_of_german_definite_articles_in_language.md)**

:   本文利用 Gradiend 梯度可解释性方法研究语言模型预测德语定冠词（der/die/das/den/dem/des）时是基于抽象语法规则还是表层记忆，发现模型至少部分依赖记忆化关联而非严格的规则编码。

**[为什么 LLM 在结构化知识上产生幻觉：推理过程的机制分析](interpretability/why_llms_hallucinate_on_structured_knowledge_a_mechanistic_analysis_of_reasoning.md)**

:   论文通过两个机制指标（结构捷径依赖 SSR 和语义对齐分数 SAS）揭示 LLM 在处理线性化结构化知识时产生幻觉的内部失败机制，并基于这些信号构建了一个轻量级的幻觉检测器。

---

## 🌐 多语言/翻译 { #multilingual_mt }

**[A Multilingual Dataset and Empirical Validation for the Mutual Reinforcement Effect in Information Extraction](multilingual_mt/a_multilingual_dataset_and_empirical_validation_for_the_mutual_reinforcement_eff.md)**

:   构建首个多语言MRE Mix数据集（MMM，21个子集覆盖英中日），并通过大规模消融实验系统验证了词级与文本级信息抽取任务的互增强效应（MRE）跨语言普遍存在。

**[Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](multilingual_mt/alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)**

:   Alexandria 构建了覆盖 13 个阿拉伯国家、11 个社会影响领域、107K 轮次的多轮对话方言阿拉伯语-英语平行数据集，通过社区驱动的人工翻译与修订流程，为方言阿拉伯语机器翻译提供了前所未有的细粒度训练和评测资源，并在 24 个 LLM 上进行了系统性基准评估。

**[BabelDOC: Better Layout-Preserving PDF Translation via Intermediate Representation](multilingual_mt/babeldoc_better_layout-preserving_pdf_translation_via_intermediate_representatio.md)**

:   本文提出 BabelDOC：一个基于「中间表示（IR）」的版式保持 PDF 翻译系统，把视觉布局和语义内容解耦，让 LLM 翻译、术语提取、跨页上下文、公式占位等 NLP 操作发生在语义层，再用自适应排版引擎重新锚回原版式；在 200 页基准上 BIoU、layout fidelity、术语一致性都超过 PDFMathTranslate 和 DeepL Document Translation。

**[Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation](multilingual_mt/beyond_literal_mapping_benchmarking_and_improving_non-literal_translation_evalua.md)**

:   构建非字面翻译元评估数据集 MENT（7,530 条人工标注），揭示传统指标和 LLM-as-Judge 在非字面翻译评估上的不可靠性，并提出 RATE 智能体评估框架，通过反思核心智能体动态调用子智能体，提升 3.2+ 点人类判断相关性。

**[BhashaSutra: A Task-Centric Unified Survey of Indian NLP Datasets, Corpora, and Resources](multilingual_mt/bhashasutra_a_task-centric_unified_survey_of_indian_nlp_datasets_corpora_and_res.md)**

:   首篇专门针对印度语言NLP资源的统一综述，覆盖200+数据集、50+基准、100+模型/工具，按17个任务类别组织（从核心语言处理到社会文化任务），系统分析了语言覆盖不均、标注碎片化、评估不一致等持续挑战。

**[CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](multilingual_mt/clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)**

:   本文提出 CLewR（Curriculum Learning with Restarts），一种在偏好优化训练中按易到难排序并在每个 epoch 重启课程的策略，有效缓解灾难性遗忘问题，在多个模型家族（Gemma2、Qwen2.5、Llama3.1）和多种偏好优化算法（DPO、CPO、ARPO）上持续提升机器翻译性能。

**[Cross-Cultural Transfer of Emoji Semantics and Sentiment in Financial Social Media](multilingual_mt/cross-cultural_transfer_of_emoji_semantics_and_sentiment_in_financial_social_med.md)**

:   在 4 语种 / 2 平台 / 2 资产类的 1 亿条金融微博上系统比较 emoji 的频率、语义和情感极性，发现 emoji 频率因语言/平台差异大但语义和极性高度稳定，并据此在零样本情感迁移上验证：把 emoji 加入文本能稳定地把 cross-platform transfer gap 从最高 21% 降到接近 0%。

**[DFKI-MLT at SemEval-2026 TASK 7: Steering Multilingual Models Towards Cultural Knowledge](multilingual_mt/dfki-mlt_at_semeval-2026_task_7_steering_multilingual_models_towards_cultural_kn.md)**

:   这篇 SemEval 系统论文用 FLORES 平行语料提取语言方向，在推理时向多语言 LLM 的 residual stream 注入 language steering vector，最终 MCQ 官方成绩为 86.96% accuracy、17 队第 7，但后验分析显示增益高度依赖层、prompt、模型和 locale。

**[Digitizing Nepal's Written Heritage: A Comprehensive HTR Pipeline for Old Nepali Manuscripts](multilingual_mt/digitizing_nepals_written_heritage_a_comprehensive_htr_pipeline_for_old_nepali_m.md)**

:   首个端到端的**古尼泊尔语手写文本识别 (HTR)** 完整 pipeline：用 "合成 Devanagari → 印刷 Nagari → 古尼泊尔手稿" 三阶段迁移学习 + 20 种数据增强 + 字节级 BPE + script-aware decoder，把 CER 从 fine-tuned TrOCR baseline 的 9.6% 降到 **4.9%**，并开源代码、模型与 Streamlit web 应用。

**[Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion](multilingual_mt/efficient_low-resource_language_adaptation_via_multi-source_dynamic_logit_fusion.md)**

:   TriMix 把 LRL（低资源语言）适配拆解为"语言能力 + 任务能力 + scaling 红利"三股 logit benefit vector，仅对小模型做连续预训练即可，在推理时按 perplexity 动态决定权重，于 4 个模型家族 × 8 种 LRL 上一致超越单模型 baseline 和 Proxy Tuning，且核心实证发现"应让小 CPT 模型权重高于大指令模型"——直接挑战了 Proxy Tuning 默认的"大模型主导"假设。

**[Efficient Training for Cross-lingual Speech Language Models](multilingual_mt/efficient_training_for_cross-lingual_speech_language_models.md)**

:   本文提出CSLM，一种高效训练跨语言语音LLM的方法，通过新颖的对齐策略实现跨模态和跨语言对齐，并引入语音-文本交织链式模态生成来提升质量和降低延迟，无需大规模语音数据即可扩展到新语言。

**[EMCEE: Improving Multilingual Capability of LLMs via Bridging Knowledge and Reasoning with Extracted Synthetic Multilingual Context](multilingual_mt/emcee_improving_multilingual_capability_of_llms_via_bridging_knowledge_and_reaso.md)**

:   EMCEE 让 LLM 先从自身参数中抽取与非英语 query 相关的合成多语言上下文，再把上下文增强回答与 CoT 推理回答交给 LLM-as-a-Judge 合并，在四个多语言任务上显著提升低资源语言表现。

**[Enhancing BiGRU with a KAN Block for Legal Document Classification and Summarization](multilingual_mt/enhancing_bigru_with_a_kan_block_for_legal_document_classification_and_summariza.md)**

:   本文在低资源多语言孟加拉法律文档上，把 KAN block 加到 BiGRU 分类器和 attention-based GRU 摘要模型中，使分类准确率达到 67.96%、ROUGE-1/2/L 达到 0.38/0.23/0.31，并在消融中把 BiGRU 准确率从 57.34% 提升到 67.96%。

**[Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors](multilingual_mt/evaluating_robustness_of_large_language_models_against_multilingual_typographica.md)**

:   本文提出 MulTypo——一个基于各语言键盘布局和 10 指打字习惯的多语种 typo 生成算法，并用它系统评测 18 个开源 LLM 在 12 种语言、5 类下游任务上的鲁棒性，证明 typo 对生成与推理任务影响最大、指令微调反而更脆弱、且 typo 影响存在跨语种和跨方向的非对称性。

**[Evaluating the Impact of Verbal Multiword Expressions on Machine Translation](multilingual_mt/evaluating_the_impact_of_verbal_multiword_expressions_on_machine_translation.md)**

:   本文首次系统评测动词类多词表达式（VMWE：动词成语 VID、动词-小品词 VPC、轻动词构式 LVC）对机器翻译质量的影响，在 8 个 MT 系统 × 7 个语言对 × 两类 QE 模型 + 人工 DA 评分上证明：VMWE 普遍掉点，且掉点幅度与"非组合性"严格正相关（VID > VPC > LVC），即使 GPT-4.1/GPT-5.1 也无法消除这一退化。

**[Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models](multilingual_mt/exploring_continual_fine-tuning_for_enhancing_language_ability_in_large_language.md)**

:   本文提出两阶段持续微调（CFT）框架——先在英语指令数据上微调，再在多语言数据上微调——发现阶段间数据集的指令相似性是决定英语能力是否退化的关键因素，并通过生成式重放和启发式层冻结有效缓解了不相似数据集导致的表示漂移和英语遗忘。

**[FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](multilingual_mt/fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)**

:   提出 FairQE 多智能体框架，通过性别线索检测、性别翻转变体生成和动态偏见感知分数聚合机制，在不牺牲翻译质量评估准确性的前提下有效缓解 QE 模型中的系统性性别偏见。

**[From Fragments to Facts: A Curriculum-Driven DPO Approach for Generating Hindi News Veracity Explanations](multilingual_mt/from_fragments_to_facts_a_curriculum-driven_dpo_approach_for_generating_hindi_ne.md)**

:   本文提出 DeFactoX，用课程学习组织 Hindi 新闻偏好数据，并在 DPO 中加入 Actuality 和 Finesse 两个事实性/稳定性信号，使模型能同时预测新闻真伪并生成更接近人工事实核查解释的 Hindi rationale。

**[From Traditional Taggers to LLMs: A Comparative Study of POS Tagging for Medieval Romance Languages](multilingual_mt/from_traditional_taggers_to_llms_a_comparative_study_of_pos_tagging_for_medieval.md)**

:   作者在三种中古罗曼语（古奥克语 NAF、古加泰罗尼亚语 CAT、古法语 Chauliac）的 POS 标注任务上，把传统 tagger（UDPipe / COLaF）与开源 LLM（Gemma3-12B / Phi4-14B）放在 zero-shot、few-shot、单语微调、双语 CLTF、三语 CLTF 五种设置下做系统对比，发现 LLM 一致优于传统方法，加泰罗尼亚语充当"桥梁语种"使 CAT+FR 双语训练把古法语 Chauliac 推到 93.14% 的最高准确率。

**[Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech](multilingual_mt/hierarchical_policy_optimization_for_simultaneous_translation_of_unbounded_speec.md)**

:   本文提出 Hierarchical Policy Optimization (HPO)，通过层级奖励设计对基于 LLM 的同声传译模型进行后训练，在翻译质量未达阈值时抑制延迟优化，从而在 1.5 秒延迟下实现 +7 COMET 的翻译质量提升。

**[IndoTabVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents](multilingual_mt/indotabvqa_a_benchmark_for_cross-lingual_table_understanding_in_bahasa_indonesia.md)**

:   提出 IndoTabVQA，一个针对印尼语（Bahasa Indonesia）文档表格的跨语言视觉问答基准，包含 1593 张文档图像和四种语言（印尼语/英语/印地语/阿拉伯语）的 QA 标注，揭示了 VLM 在低资源语言和跨语言表格理解上的显著性能差距，微调+空间先验可带来最高 48.5% 的 In-Match 准确率。

**[Is Human-Like Text Liked by Humans? Multilingual Human Detection and Preference Against AI](multilingual_mt/is_human-like_text_liked_by_humans_multilingual_human_detection_and_preference_a.md)**

:   作者组织 19 位母语专家对 9 种语言、9 个领域、11 个 SOTA LLM 共 16 个数据集做了 8.8k 例的人机文本判别，发现专家平均准确率高达 87.6%（远高于"接近随机"的早期结论），并进一步揭示：用显式说明差异的 prompt 改写后机器文本能把检测准确率压到 72.5%，但人在分不清来源时反而倾向选机器文本，挑战了"human-like 等于 liked-by-human"的隐含假设。

**[Just Use XML: Revisiting Joint Translation and Label Projection](multilingual_mt/just_use_xml_revisiting_joint_translation_and_label_projection.md)**

:   提出 LabelPigeon，一种基于 XML 标签的联合翻译与标签投影方法，通过在高质量 XML 标记平行语料上微调 NLLB-200 翻译模型，在 11 种语言上超越所有基线并主动提升翻译质量，在下游跨语言 NER 任务中实现最高 +40.2 F1 的提升。

**[Language Models Entangle Language and Culture](multilingual_mt/language_models_entangle_language_and_culture.md)**

:   本文通过基于 WildChat 数据集构建的通用建议类问题评估多语言 LLM，发现不同语言查询会导致回答质量和文化上下文的系统性差异——低资源语言的回答质量显著低于英语，且语言选择会隐式地改变回答中使用的文化信息，在翻译版 CulturalBench 上验证了语言与文化在 LLM 中的纠缠关系。

**[Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](multilingual_mt/language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)**

:   本文提出 XBridge，一种将预训练多语言编码器-解码器翻译模型（如 NLLB）与英语为中心的 LLM 组合的架构——编码器负责多语言理解、LLM 负责知识推理、解码器负责多语言生成，通过轻量级映射层和最优传输对齐实现跨模型语义桥接，在低资源和未见语言上显著优于基线。

**[LaoBench: A Large-Scale Multidimensional Lao Benchmark for Large Language Models](multilingual_mt/laobench_a_large-scale_multidimensional_lao_benchmark_for_large_language_models.md)**

:   LaoBench 是首个大规模、多维度的老挝语 LLM 评测基准，包含 17000+ 条专家精选样本，覆盖文化-知识应用 / 老挝 K12 课纲 / 老-中-英三语翻译三大维度，并独创"开源 7k + 黑盒 10k + 开放式 500"三段式设计——其中 10k 黑盒子集通过受控服务发分数防污染，主流闭源模型（GPT-5-High、Gemini-2.5-Pro 等）仍落后人类专家 ~10-20 个百分点，老挝语文化推理与翻译保真度依然是远未解决的难题。

**[Lingo_Research_Group at SemEval-2026 Task 9: Evaluating Prompt Variants for Polarization Detection](multilingual_mt/lingo_research_group_at_semeval-2026_task_9_evaluating_prompt_variants_for_polar.md)**

:   这篇 SemEval-2026 Task 9 系统论文用 Gemma3-27B 和 12 类英文 prompt 变体做 22 种语言的在线极化检测，发现 prompt-only 方法能较好完成粗粒度二分类，但在极化目标和表现形式等细粒度多标签任务上明显退化。

**[LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models](multilingual_mt/llm-xtm_enhancing_cross-lingual_topic_models_with_large_language_models.md)**

:   在已训练好的跨语言主题模型外面套一层"LLM 精炼 + 自一致性投票 + MMD 词分布对齐 + QA 式文档语义对齐"的两段式增强模块，可作为插件挂到 NMTM / InfoCTM / XTRA 等多种 backbone 上，在 EC News、Amazon Review、Rakuten Amazon 三个双语语料上把 CNPMI 涨了 9%–51%、TQ 涨了 6%–44%，同时把 LLM 调用次数降到了"每隔 $f$ 个 epoch 才一次"。

**[Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs](multilingual_mt/location_not_found_exposing_implicit_local_and_global_biases_in_multilingual_llm.md)**

:   本文提出 LocQA 基准（12 种语言、49 个地区、2156 个地域相关问答），通过地域模糊问题（如"紧急电话号码是多少？"）揭示 LLM 的隐式偏差：跨语言上存在持续的美国中心默认行为（模型回答的 50% 包含美国答案 vs 数据中仅 26%），语言内部存在人口规模驱动的"人口概率引擎"效应，且指令微调加剧了全球偏差。

**[Lost in Translation: Do LVLM Judges Generalize Across Languages?](multilingual_mt/lost_in_translation_do_lvlm_judges_generalize_across_languages.md)**

:   本文提出 MM-JudgeBench，首个大规模多语言多模态评判模型基准（25 种语言、60K+ 偏好实例），评估 22 个 LVLM 发现当前 LVLM 评判器存在显著的跨语言性能差异——模型大小和架构不能预测多语言鲁棒性，即使最先进的评判器也表现不一致，突显了多语言多模态评估基准的必要性。

**[LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](multilingual_mt/lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)**

:   提出LQM（Linguistically Motivated Multidimensional Quality Metrics），一个六层语言学驱动的机器翻译错误分类体系（社会语言学→语用学→语义→形态句法→正字法→字形），并在7种阿拉伯方言上构建3850句双向平行语料库，通过专家标注6113个错误跨度揭示现有MT系统在方言和文化感知翻译上的系统性不足。

**[Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates](multilingual_mt/mitigating_catastrophic_forgetting_in_target_language_adaptation_of_llms_via_sou.md)**

:   提出 Source-Shielded Updates (SSU)，一种基于源数据驱动参数重要性评分的列级冻结策略，在仅使用无标签目标语言数据进行持续预训练时，将源语言性能退化从全量微调的 20.3% 降低至 3.4%，同时保持与全量微调相当甚至更优的目标语言性能。

**[Mitigating Extrinsic Gender Bias for Bangla Classification Tasks](multilingual_mt/mitigating_extrinsic_gender_bias_for_bangla_classification_tasks.md)**

:   针对孟加拉语预训练模型在下游分类任务中的外在性别偏见，提出 RandSymKL 方法，通过随机化交叉熵损失和对称 KL 散度联合优化，在保持分类准确率的同时有效缩小性别间预测差异。

**[MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation](multilingual_mt/morphogen_a_multilingual_benchmark_for_evaluating_gender-aware_morphological_gen.md)**

:   本文提出 MORPHOGEN，一个涵盖法语/阿拉伯语/印地语的大规模性别感知形态学生成基准（共 20,328 句对），定义了 GENFORM 任务（将第一人称句子改写为相反性别），并提出 SGA/GIoU/CGA 三个评估指标，对 15 个多语言 LLM 的基准测试揭示了模型在复杂形态推理、性别偏差和多实体干扰方面的系统性不足。

**[MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing](multilingual_mt/multihaludet_multilingual_hallucination_detection_via_llm_hidden_state_probing.md)**

:   MultiHaluDet 用冻结 LLM 的全层隐状态轨迹做多尺度序列建模，再通过 out-of-fold 表征和集成元学习器判别幻觉，在 HaluEval / TriviaQA 上达到约 98% AUROC，并能迁移到法语、孟加拉语和阿姆哈拉语。

**[Multilingual Language Models Encode Script Over Linguistic Structure](multilingual_mt/multilingual_language_models_encode_script_over_linguistic_structure.md)**

:   本文通过 LAPE 指标和稀疏自编码器系统分析多语言 LM 中的语言关联单元，发现这些单元主要由正字法（书写系统）驱动而非抽象语言结构：罗马化转写激活几乎完全不重叠的神经元集合，词序打乱影响甚微，类型学信息仅在深层逐渐可访问，因果干预表明功能重要性与表面形式不变性相关。

**[Multilingual Steering by Design: Multilingual Sparse Autoencoders and Principled Layer Selection](multilingual_mt/multilingual_steering_by_design_multilingual_sparse_autoencoders_and_principled_.md)**

:   这篇论文证明，多语言 sparse autoencoder 和“多语言对齐-语言可分性”交叉点层选择可以让 SAE 语言 steering 更稳定，把原先靠经验选层的多语言控制问题转成可预测的表征诊断问题。

**[NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning](multilingual_mt/neoamt_neologism-aware_agentic_machine_translation_with_reinforcement_learning.md)**

:   NeoAMT 把新词翻译从单纯依赖模型参数知识的问题，改造成“先推理、再查词典、再翻译”的 agentic MT 问题，并用面向新词命中率、整体译文质量和翻译难度的 GRPO 训练，让 8B 模型在 Neko 新词翻译基准上显著超过 SFT、无检索 RL 和多种通用/翻译专用 LLM。

**[NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](multilingual_mt/niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)**

:   本文发布 NiuTrans.LMT，一个覆盖 60 种语言、234 个中英双中心翻译方向、0.6B/1.7B/4B/8B 四种规模的开源 LLM 机器翻译套件，并指出多路平行数据在对称 SFT 中会造成 X→中/英方向退化，再用 Strategic Downsampling、Parallel Multilingual Prompting 和 COMET 奖励的 GRPO 把质量拉回到强开源 MMT 系统水平。

**[No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](multilingual_mt/no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)**

:   本文证明没有一种提示策略在所有语言和任务上普遍最优，提出将策略选择建模为学习决策问题，用轻量级分类器为每个实例预测最优策略，在四个基准上显著优于固定策略。

**[PEAR: Pairwise Evaluation for Automatic Relative Scoring in Machine Translation](multilingual_mt/pear_pairwise_evaluation_for_automatic_relative_scoring_in_machine_translation.md)**

:   PEAR 将无参考机器翻译质量估计从“给单个译文打绝对分”改成“直接比较两个候选译文的相对差值”，在 WMT24 MQM 评测中以更小模型超过匹配的单候选 QE 基线和部分大规模指标。

**[PluRule: A Benchmark for Moderating Pluralistic Communities on Social Media](multilingual_mt/plurule_a_benchmark_for_moderating_pluralistic_communities_on_social_media.md)**

:   PluRule 把 Reddit 社区审核建模为“给定评论和上下文，选择违反了哪条本社区规则或没有违规”的多选题，构建了覆盖 1,989 个社区、2,885 条规则和 9 种语言的基准，并显示 GPT-5.2 high reasoning 全上下文也只有约 57.6% 准确率。

**[Prosody as Supervision: Bridging the Non-Verbal–Verbal for Multilingual Speech Emotion Recognition](multilingual_mt/prosody_as_supervision_bridging_the_non-verbal--verbal_for_multilingual_speech_e.md)**

:   本文提出 NOVA-ARC，首次将多语言语音情感识别（SER）建模为从标注的非语言发声（NVV）到未标注的语言语音（UVS）的无监督迁移问题，通过双曲空间中的韵律向量量化编码本、双曲情感透镜和最优传输原型对齐实现跨模态情感迁移，在 6 个数据集上验证了非语言→语言迁移的可行性和优越性。

**[Reinforcement Learning with Semantic Rewards Enables Low-Resource Language Expansion without Alignment Tax](multilingual_mt/reinforcement_learning_with_semantic_rewards_enables_low-resource_language_expan.md)**

:   本文把低资源语言扩展从 token 级模仿改写为语义空间对齐问题，用 GRPO 和 embedding 语义奖励训练 Qwen3-4B，在藏汉翻译和藏语标题生成上获得低资源能力，同时比强 SFT 更好地保留中文 CMRC 等主导语言能力。

**[RouteLMT: Learned Sample Routing for Hybrid LLM Translation Deployment](multilingual_mt/routelmt_learned_sample_routing_for_hybrid_llm_translation_deployment.md)**

:   RouteLMT 将混合 LLM 翻译中的路由问题形式化为固定大模型预算下的边际收益分配，并用小翻译模型最后 prompt token 的内部表示预测“大模型相对小模型能带来多少提升”，在四个翻译方向上比长度、质量估计和外部路由器获得更好的质量-预算 Pareto 前沿。

**[Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP](multilingual_mt/scripts_through_time_a_survey_of_the_evolving_role_of_transliteration_in_nlp.md)**

:   本文系统综述了音译（transliteration）在跨语言 NLP 中的演变角色，提出五大动机分类（命名实体/OOV处理、代码混合、跨文字相似性利用、英语中心迁移、统一预处理），比较了六种整合方式的优劣，并在现代 LLM 语境下讨论了音译是否仍然必要。

**[Selective Contrastive Learning For Gloss Free Sign Language Translation](multilingual_mt/selective_contrastive_learning_for_gloss_free_sign_language_translation.md)**

:   这篇论文发现手语翻译中随机 in-batch 负样本经常是不可靠或语义冲突的监督信号，因此用参考模型的相似度轨迹筛选更有信息量的负样本，并通过从易到难的课程式对比学习提升 gloss-free 手语翻译质量。

**[SERM: Self-Evolving Relevance Model with Agent-Driven Learning from Massive Query Streams](multilingual_mt/serm_self-evolving_relevance_model_with_agent-driven_learning_from_massive_query.md)**

:   提出 SERM 框架，通过多智能体样本挖掘器和多智能体相关性标注器，从大规模真实查询流中持续自进化搜索相关性模型，经三轮迭代在工业搜索平台上实现 NDCG@1 提升 +2.99，并在在线 A/B 测试中显著提升用户留存率。

**[SteerEval: Inference-time Interventions Strengthen Multilingual Generalization in Neural Summarization Metrics](multilingual_mt/steereval_inference-time_interventions_strengthen_multilingual_generalization_in.md)**

:   SteerEval 研究在推理时把多语种评测模型的隐藏表示向高资源枢轴语言对齐，发现向英语或法语方向 steering 能普遍提高多语种摘要自动指标与人工评分的相关性，尤其能改善低基线语言和 encoder-based COMET 指标。

**[Structure-Guided Entity Resolution: Fine-Tuning LLMs for Robust Name Matching in Complex Linguistic Contexts](multilingual_mt/structure-guided_entity_resolution_fine-tuning_llms_for_robust_name_matching_in_.md)**

:   SGER 提出两阶段课程学习框架微调 Llama 3 8B 进行人名实体匹配：Phase 1 训练模型解析姓名结构（输出 JSON），Phase 2 从 Phase 1 checkpoint 训练二分类匹配器，在 5 万对印度 KYC 数据上达到 99.02% 准确率和 0.994 F1，已在 Dream11（2.5 亿用户）生产环境部署。

**[Syntax as a Rosetta Stone: Universal Dependencies for In-Context Coptic Translation](multilingual_mt/syntax_as_a_rosetta_stone_universal_dependencies_for_in-context_coptic_translati.md)**

:   本文首次探索将 Universal Dependencies 句法信息作为上下文学习的增强源用于低资源科普特语到英语的机器翻译，发现虽然句法信息单独不如词典有效，但将词典与句法信息结合（LEX+SYN）在所有模型上取得最佳效果，Gemma-27B 的 BERTScore F1 达到 0.8746（+0.0361）。

**[The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](multilingual_mt/the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)**

:   本文提出GaoYao基准，包含182.3K样本、26种语言和51个国家/地区，通过三层文化评估框架（通用多语言/跨文化/单文化）和九个认知子层，结合人工本地化的主观测试集和专家验证的跨文化合成数据集SuperBLEnD，深度诊断20+旗舰与紧凑型LLM的多语言能力，揭示了显著的地理数字鸿沟和任务能力分层。

**[Think in Latent Thoughts: A New Paradigm for Gloss-Free Sign Language Translation](multilingual_mt/think_in_latent_thoughts_a_new_paradigm_for_gloss-free_sign_language_translation.md)**

:   提出 SignThought，一种推理驱动的无注释手语翻译框架：引入可学习的潜在思维槽作为视频和文本之间的显式中间语义层，通过"先规划后定位"的双流解码器实现语义规划与视觉证据检索的解耦，在多个基准上超越现有无注释方法。

**[TLPO: Token-Level Policy Optimization for Mitigating Language Confusion in Large Language Models](multilingual_mt/tlpo_token-level_policy_optimization_for_mitigating_language_confusion_in_large_.md)**

:   TLPO 将多语言 LLM 的语言混淆视为可定位的局部 token 错误，只在首次混淆位置的高概率候选 token 上做策略优化，从而在显著提高目标语言一致性的同时尽量保留模型原有推理和知识能力。

**[Toward Culturally Grounded Natural Language Processing](multilingual_mt/toward_culturally_grounded_natural_language_processing.md)**

:   这篇综述型论文综合 50 余篇多语与文化 NLP 工作，指出“语言覆盖率”不等于“文化能力”，并提出以 communicative ecologies 为中心的分层评估协议和研究议程。

**[TransLaw: A Large-Scale Dataset and Multi-Agent Benchmark Simulating Professional Translation of Hong Kong Case Law](multilingual_mt/translaw_a_large-scale_dataset_and_multi-agent_benchmark_simulating_professional.md)**

:   这篇论文构建了首个面向香港终审法院判词英中翻译的句级平行数据集 HKCFA Judgement 97-22，并提出模拟专业法律翻译流程的 TransLaw 多智能体系统，在自动指标、人类法律译者评测和成本分析上都显著优于单一翻译 agent。

**[Unlocking the Edge: Multi-LoRA On-Device Deployment and Acceleration](multilingual_mt/unlocking_the_edge_deployment_and_ondevice_acceleration_of_multi-lora_enabled_on.md)**

:   本文提出面向三星 Galaxy S24/S25 的端侧 LLM 部署框架，通过 LoRA 权重作为运行时输入实现动态任务切换、多流并发 token 生成减少风格变体延迟达 6 倍、无草稿模型的 Dynamic Self-Speculative Decoding 加速解码达 2.3 倍，在 9 语言 8 任务上实现 4-6 倍整体优化。

**[Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic](multilingual_mt/vocab_diet_reshaping_the_vocabulary_of_llms_via_vector_arithmetic.md)**

:   本文发现 LLM 在嵌入空间中将词形变化（如 walk→walked）编码为线性方向，基于此提出组合式词表设计：用基础词+变换向量的加法组合替代为每个表面形式分配独立 token，在冻结预训练骨干的前提下仅训练小型适配模块，释放 10-40% 的词表槽位用于多语言扩展，同时几乎不影响下游性能。

**[Vocabulary Shapes Cross-Lingual Variation of Word-Order Learnability in Language Models](multilingual_mt/vocabulary_shapes_cross-lingual_variation_of_word-order_learnability_in_language.md)**

:   本文用 Mallows 模型生成 10 种欧洲语言的连续词序扰动谱，训练小型自回归 LM 后发现：词序越不规则越难学，但跨语言差异主要由词表覆盖、句长和形态复杂度解释，而不是简单的自由/固定词序标签。

**[What Factors Affect LLMs and RLLMs in Financial Question Answering?](multilingual_mt/what_factors_affect_llms_and_rllms_in_financial_question_answering.md)**

:   本文系统研究了提示方法、Agent 框架和多语言对齐方法对 LLM 和 RLLM（推理型大模型）在金融问答任务上的影响，发现现有方法本质上是通过模拟 Long CoT 来提升 LLM 性能，但对已具备 Long CoT 能力的 RLLM 效果有限。

**[Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?](multilingual_mt/why_do_multilingual_reasoning_gaps_emerge_in_reasoning_language_models.md)**

:   本文首次系统分析了推理语言模型(RLMs)中多语言推理差距的来源，发现**语言理解失败**是主要原因，并提出通过检测理解失败后进行选择性翻译(Selective Translation)来高效弥补差距。

**[Why Low-Resource NLP Needs More Than Cross-Lingual Transfer: Lessons Learned from Luxembourgish](multilingual_mt/why_low-resource_nlp_needs_more_than_cross-lingual_transfer_lessons_learned_from.md)**

:   本文以理论上很适合跨语言迁移的卢森堡语为“最好情况”案例，论证低资源 NLP 不能只依赖多语模型的自发迁移，而必须把跨语言脚手架和目标语言特定的数据清洗、资源建设、任务设计结合起来。

**[XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](multilingual_mt/xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)**

:   构建首个具有跨语言平行质量的翻译评估基准 XQ-MEval，通过半自动注入 MQM 错误生成可控质量的伪翻译，首次实证揭示自动评估指标的跨语言评分偏差，并提出 LGN 归一化策略有效校准多语言指标评估。

---

## 📦 模型压缩 { #model_compression }

**[A BERTology View of LLM Orchestrations: Token- and Layer-Selective Probes for Efficient Single-Pass Classification](model_compression/a_bertology_view_of_llm_orchestrations_token-_and_layer-selective_probes_for_eff.md)**

:   把 production LLM 的 token×layer 隐状态张量当成可挖掘资源，用「先压 token、再压 layer」的两阶段聚合 probe 在同一次 forward 里完成安全/情感分类，35M 可训练参数即可逼近独立 guard 模型，省掉一次额外的 LLM 调用。

**[A Layer-wise Analysis of Supervised Fine-Tuning](model_compression/a_layer-wise_analysis_of_supervised_fine-tuning.md)**

:   通过信息论、几何和优化三个视角对 1B-32B 模型的 SFT 进行逐层分析，发现指令跟随能力集中在中间层（20%-80%），而非均匀分布，据此提出 Mid-Block Efficient Tuning 策略，选择性更新中间层，在 GSM8K 上比标准 LoRA 提升高达 10.2%。

**[Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)**

:   提出ASL（Adaptive Selection Layer），通过监控token注意力分数排名的方差来自适应确定KV缓存剪枝的层位置，在困难任务上显著优于固定层选择方法，同时保持无需训练。

**[Alignment Tuning for Large Language Models: A Data-Centric Lens on Alignment Data Pipelines](model_compression/alignment_tuning_for_large_language_models_a_data-centric_lens_on_alignment_data.md)**

:   这篇论文把 LLM alignment tuning 重新解释为一个动态的数据管线设计问题：模型最终学到什么，不只取决于 PPO、DPO、GRPO 这类优化算法，更取决于候选回答如何生成、偏好如何评估、偏好信号又如何实例化为训练目标。

**[Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis](model_compression/analytical_ffn-to-moe_restructuring_via_activation_pattern_analysis.md)**

:   提出一种分析式后训练框架，通过神经元激活模式分析将dense FFN快速重构为sparse MoE——区分高频共享专家和低频路由专家，并从激活统计量构建路由器，仅需2k样本微调即可实现1.17×加速。

**[ArcLight: A Lightweight LLM Inference Architecture for Many-Core CPUs](model_compression/arclight_a_lightweight_llm_inference_architecture_for_many-core_cpus.md)**

:   ArcLight 是一个从零写的轻量级 LLM 推理框架（约 10 个 C++ 文件），专为多 NUMA 节点的 many-core CPU 设计，通过 NUMA 局部内存池、多视图线程池、跨 NUMA 张量并行 + 异步子图同步打破"远程内存墙"，在 192 核 ARM 鲲鹏平台上把 Qwen3-4B Q4_0 的 decode 吞吐相对 llama.cpp 提高至多 46%。

**[BaseCal: Unsupervised Confidence Calibration via Base Model Signals](model_compression/basecal_unsupervised_confidence_calibration_via_base_model_signals.md)**

:   观察到 base LLM 在 free-form QA 上仍然保持良好校准、而 post-trained LLM（PoLLM）严重过自信，BaseCal 提出两种无监督方案——把 PoLLM 的回答喂进 base LLM 拿 token 概率做置信度（BaseCal-ReEval），或用一层线性投影把 PoLLM 末层隐状态映射回 base LLM 空间再过 base 的输出层（BaseCal-Proj），在 5 个数据集 × 3 个模型族上把 ECE 相对最佳无监督基线平均降低 42.9%。

**[Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](model_compression/calibrated_speculative_decoding_frequency-guided_candidate_selection_for_efficie.md)**

:   CSD 提出一种训练免的推测解码增强框架，通过在线校正记忆（OCM）记录高频拒绝模式提供救援候选，再用语义一致性门控（SCG）基于概率比验证候选可靠性，将推测解码的吞吐量提升至最高 2.33×，同时在 HumanEval 和 MATH500 上甚至提升了准确率。

**[CBRS: Cognitive Blood Request System with Bilingual Dataset and Dual-Layer Filtering](model_compression/cbrs_cognitive_blood_request_system_with_bilingual_dataset_and_dual-layer_filter.md)**

:   CBRS 提出一个多平台框架，通过双层过滤架构（轻量分类器 + LLM）从社交媒体消息流中高效检测并解析血液捐献请求，构建了首个包含 11K 条孟加拉语-英语-转写孟加拉语的血液捐献请求数据集，LoRA 微调的 Llama-3.2-3B 在解析任务上达到 92% 零样本准确率。

**[Cognitive-Uncertainty Guided Knowledge Distillation for Accurate Classification of Student Misconceptions](model_compression/cognitive-uncertainty_guided_knowledge_distillation_for_accurate_classification_.md)**

:   论文用两阶段知识蒸馏 + 基于教师认知不确定性的"双层边际样本选择" + 难度自适应损失，在仅用 10.30% 真实样本增量训练的情况下，把 4B 学生模型在 MAP-Charting 上做到 MAP@3 = 0.9585（+17.8%），并在 220 题中学代数误解 benchmark 上以 84.38% 准确率超过 GPT-5（67.73%）与直接微调的 72B 教师（81.25%），同时推理速度比教师快 23×。

**[DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing](model_compression/dash-kv_accelerating_long-context_llm_inference_via_asymmetric_kv_cache_hashing.md)**

:   提出 DASH-KV 框架，将注意力机制重构为近似最近邻搜索问题，通过非对称深度哈希将高维浮点相似度计算替换为高效的汉明距离比特操作，配合动态混合精度机制，将长上下文推理复杂度从 $O(N^2)$ 降至 $O(N)$ 且性能匹配全注意力。

**[DeepPrune: Parallel Scaling without Inter-Trace Redundancy](model_compression/deepprune_parallel_scaling_without_inter-trace_redundancy.md)**

:   本文提出 DeepPrune，通过训练专门的判断模型从部分推理轨迹预测答案等价性，结合在线贪心聚类算法动态剪枝冗余的并行 CoT 路径，在保持竞争准确率（3 个百分点以内）的同时减少 65.73%-88.50% 的 token 消耗。

**[Efficient Learned Data Compression via Dual-Stream Feature Decoupling](model_compression/efficient_learned_data_compression_via_dual-stream_feature_decoupling.md)**

:   本文提出FADE框架，通过双流多尺度解耦器将微观句法和宏观语义特征分离到并行浅层流中处理（取代深层串行堆叠），结合层次化门控精炼器和并发流并行流水线，在压缩率和吞吐量上同时达到SOTA。

**[Enabling Agents to Communicate Entirely in Latent Space](model_compression/enabling_agents_to_communicate_entirely_in_latent_space.md)**

:   本文提出 Interlat，一个让 LLM 智能体完全在潜空间中通信的框架——发送方直接传递最后一层隐状态作为"思维"的表示，接收方通过通信适配器解释这些潜空间消息，并通过潜空间推理进一步压缩到仅 8 个 token 同时保持竞争性能，实现高达 24× 的通信加速。

**[Establishing a Scale for Kullback–Leibler Divergence in Language Models Across Various Settings](model_compression/establishing_a_scale_for_kullback-leibler_divergence_in_language_models_across_v.md)**

:   本文利用对数似然向量将不同架构的语言模型嵌入统一空间，系统测量了预训练、模型规模、随机种子、量化、微调和层间等多种设置下的 KL 散度特征尺度，并发现预训练轨迹在对数似然空间中呈亚扩散行为——尽管权重空间持续漂移，模型输出分布早期即趋于稳定。

**[Evolutionary Negative Module Pruning for Better LoRA Merging](model_compression/evolutionary_negative_module_pruning_for_better_lora_merging.md)**

:   提出 ENMP 方法，通过进化搜索策略发现并剪除 LoRA 合并中降低性能的"负面模块"，作为即插即用的增强手段，在 NLP 和视觉领域全面提升现有合并算法的效果。

**[FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration](model_compression/fastkv_decoupling_of_context_reduction_and_kv_cache_compression_for_prefill-deco.md)**

:   本文提出 FastKV，通过将上下文缩减（prefill 阶段的 Token-Selective Propagation）与 KV 缓存压缩（decoding 阶段的层级 KV 保留）解耦，在 LLaMA-3.1-8B-Instruct 上实现 prefill 1.82× 和 decoding 2.87× 加速，同时在 LongBench 上精度下降控制在 1% 以内。

**[Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation](model_compression/find_your_optimal_teacher_personalized_data_synthesis_via_router-guided_multi-te.md)**

:   提出 PerSyn（Personalized data Synthesis），通过"先路由再生成"范式让路由器为每个 prompt 分配最优教师模型，综合考虑学生可学习性和教师响应质量，比传统"先生成再选择"范式高效且效果更好，在指令微调和数学推理两个场景中一致超越所有基线。

**[From Signal Degradation to Computation Collapse: Uncovering the Two Failure Modes of LLM Quantization](model_compression/from_signal_degradation_to_computation_collapse_uncovering_the_two_failure_modes.md)**

:   本文通过系统的机械可解释性分析，揭示LLM量化存在两种质性不同的失败模式：4-bit的信号退化（Signal Degradation，计算模式完整但精度受损，可局部修复）和2-bit的计算崩溃（Computation Collapse，关键组件功能性破坏，需结构重建）。

**[GlimpRouter: Efficient Collaborative Inference by Glimpsing One Token of Thoughts](model_compression/glimprouter_efficient_collaborative_inference_by_glimpsing_one_token_of_thoughts.md)**

:   本文提出 **GlimpRouter**：在 step 级 LRM 协同推理中，先让小模型只解码每个推理步的"第一个 token"，用它的熵 $\mathbf{H}_{\text{init}}$ 估计该步难度，低则小模型续写、高则切换到大模型；training-free，无需大模型 verifier，在 AIME25 上比独立大模型准确率 +10.7% 同时延迟 −25.9%，且与 token-level Speculative Decoding 正交叠加。

**[GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models](model_compression/grasprune_global_gating_for_budgeted_structured_pruning_of_large_language_models.md)**

:   GRASPrune 提出了一种全局预算约束的结构化剪枝框架，通过投影直通估计器（Projected STE）在每步训练中强制满足硬掩码预算约束，联合剪枝 FFN 通道和 KV 头组，在 LLaMA-2-7B 上以 50% 参数保留达到 12.18 PPL，仅需单卡 A100 训练 6 分钟。

**[HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](model_compression/heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)**

:   本文提出 HeteroCache，一种免训练的动态 KV 缓存压缩框架，基于注意力头的时间异质性（稳定头 vs 漂移头）和层内冗余性（相似头聚类），实施细粒度的角色分配策略——为漂移头分配更大缓存预算，用代表头稀疏监控注意力漂移触发异步按需检索，在 224K 上下文下实现 3 倍解码加速。

**[IMPACT: Importance-Aware Activation Space Reconstruction](model_compression/impact_importance-aware_activation_space_reconstruction.md)**

:   提出 IMPACT 框架，将 LLM 低秩压缩从最小化权重重构误差转向最小化重要性加权的激活重构误差，通过将梯度信息融入激活协方差矩阵推导出闭式最优解，实现在保持精度的同时最高减少 55.4% 的模型体积。

**[CadLLM: Improving the Throughput of Diffusion-based LLMs via Training-Free Confidence-Aware Calibration](model_compression/improving_the_throughput_of_diffusion-based_large_language_models_via_a_training.md)**

:   提出 CadLLM，一种免训练的自适应推理加速方法，利用扩散语言模型（dLLM）的 token 解码置信度信号动态调整块大小、步数、词表采样范围和提交阈值四个维度，在 LLaDA 和 DREAM 上实现 1.1-2.28× 的吞吐量提升且保持竞争性准确率。

**[IntroLM: Introspective Language Models via Prefilling-Time Self-Evaluation](model_compression/introlm_introspective_language_models_via_prefilling-time_self-evaluation.md)**

:   IntroLM 在 prompt 尾部追加特殊 `[CPX]` 内省 token，再通过"只对该 token 生效的 token-conditional LoRA"在 prefilling 阶段一次性算出"这个 prompt 我能不能答对"，整套自评不进 KV cache、不影响生成，在 HotpotQA 等长上下文 QA 上 ROC-AUC 比 DeBERTa-v3-Large 高 14 个点，用作模型路由能省下最多 50% 大模型调用、33% 端到端延迟。

**[JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew](model_compression/judgemenot_personalizing_large_language_models_to_emulate_judicial_reasoning_in_.md)**

:   提出了一个 synthetic-organic 监督管线，将法官的原始判决文书转化为推理指令微调数据，通过 CLM→指令微调的 Chain-of-LoRA 策略实现对个体法官推理风格的高保真模拟，在希伯来语低资源场景下生成内容与真实法官不可区分。

**[Latent-Condensed Transformer for Efficient Long Context Modeling](model_compression/latent-condensed_transformer_for_efficient_long_context_modeling.md)**

:   LCA 提出在 MLA 的潜空间中直接进行上下文压缩——对语义潜向量用查询感知加权池化聚合、对位置键用锚点选择保持位置精度——在 128K 上下文中实现 2.5 倍预填充加速和 90% KV 缓存压缩，同时保持竞争性性能。

**[LEAP: Layer-wise Exit-Aware Pretraining for Efficient Transformer Inference](model_compression/leap_layer-wise_exit-aware_pretraining_for_efficient_transformer_inference.md)**

:   首先在理论 + 实证上指出"逐层对齐蒸馏"与"基于收敛的早退"在标准部署下**系统性不兼容**——蒸馏模型每一层都在干活、没有冗余可早退，然后提出 LEAP 这种零额外参数的辅助训练目标，让中间层提前逼近最终层表示，在 MiniLM-L12 上拿到 1.61× 实测墙钟加速（batch=1，91.9% 样本在 L7 退出）。

**[LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](model_compression/lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)**

:   LightReasoner 用较弱的 Amateur 模型和较强的 Expert 模型之间的 token 分布差异来自动找出高价值推理步骤，再只对这些步骤做对比式自蒸馏，使数学推理模型在接近或超过 SFT 的同时显著减少采样、训练时间和调参 token。

**[LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization](model_compression/llm_prompt_duel_optimizer_efficient_label-free_prompt_optimization.md)**

:   将无标签提示优化形式化为决斗老虎机（dueling bandit）问题，提出 Prompt Duel Optimizer (PDO)，通过 Double Thompson Sampling 高效选择信息量最大的提示对进行比较，结合 top-performer 变异策略扩展搜索空间，在 BBH 和 MS MARCO 上以更少的 judge 调用次数找到更强提示。

**[LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging](model_compression/lora_on_the_go_instance-level_dynamic_lora_selection_and_merging.md)**

:   提出 LoGo（LoRA on the Go），一个免训练的框架，通过单次前向传播提取 LoRA 激活信号（范数或熵），在实例级别动态选择和合并最相关的 LoRA 适配器，无需标注数据或额外训练即可实现跨任务泛化。

**[MeepleLM: A Virtual Playtester Simulating Diverse Subjective Experiences](model_compression/meeplelm_a_virtual_playtester_simulating_diverse_subjective_experiences.md)**

:   为桌游设计师做"虚拟玩家测试"——把官方规则书 + 5 类玩家人格送给一个微调过的 Qwen3-8B (MeepleLM)，让它先沿 Mechanics→Dynamics→Aesthetics (MDA) 三步推理再生成 rating + review，最终在 207 个游戏上超越 GPT-5.1 / Gemini3-Pro 在社区分布对齐 (Wasserstein 0.22 vs GPT-5.1 0.95)、内容多样性 (Div 4.34 vs 4.26) 和 Opinion Recovery (69.77 vs 63.44) 三项，并在盲测 A/B 中拿到 70%+ 用户偏好。

**[MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation](model_compression/mta_multi-granular_trajectory_alignment_for_large_language_model_distillation.md)**

:   MTA 将 LLM 蒸馏从“对齐某几个静态层”推进到“按网络深度对齐表示演化轨迹”：低层对齐词级信息，高层对齐短语级关系几何，并作为插件稳定提升 FDD、DistiLLM、DistiLLM-2 在指令跟随任务上的 ROUGE-L 表现。

**[No-Worse Context-Aware Decoding: Preventing Neutral Regression in Context-Conditioned Generation](model_compression/no-worse_context-aware_decoding_preventing_neutral_regression_in_context-conditi.md)**

:   提出 NWCAD，一种解码时适配器，通过两阶段门控机制在上下文无信息时精确回退到无上下文解码（防止中性退化），在上下文有帮助时利用上下文进行修正，兼顾"无害"与"有效"两个目标。

**[Not All Directions Matter: Towards Structured and Task-Aware Low-Rank Model Adaptation](model_compression/not_all_directions_matter_towards_structured_and_task-aware_low-rank_model_adapt.md)**

:   本文提出 StructLoRA：先用信息瓶颈过滤掉低秩更新里与任务无关的方向，再用训练期图神经网络协调不同层的 LoRA 更新，在语言、视觉和多模态任务上稳定超过 LoRA / AdaLoRA / DoRA / Sensitivity-LoRA，同时保持推理零额外开销。

**[Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions](model_compression/polynomial_expansion_rank_adaptation_enhancing_low-rank_fine-tuning_with_high-or.md)**

:   本文提出 PERA（Polynomial Expansion Rank Adaptation），通过在低秩因子的参数空间中引入结构化多项式展开（平方项和交叉项），将 LoRA 的线性适配空间扩展为多项式流形，在不增加秩或推理开销的前提下显著提升权重更新的表达能力，在常识推理和 NLU 任务上一致优于 LoRA/DoRA/HiRA 等方法。

**[ProActor: Timing-Aware Reinforcement Learning for Proactive Task Scheduling Agents](model_compression/proactor_timing-aware_reinforcement_learning_for_proactive_task_scheduling_agent.md)**

:   ProActor 把对话式任务调度从“等用户明确指令后反应”推进到“在合适时机主动触发动作”，通过自动 reference action 标注、proactiveness metrics、turn-level GRPO 和 ART-F 高效训练框架，让 4-bit Qwen2.5-14B-ProActor-Q4 在 ABCD+ 上取得最高 PRI 0.7293，并在保持动作一致性的同时显著提升 proactive timing。

**[Quantize What Counts: More for Keys, Less for Values](model_compression/quantize_what_counts_more_for_keys_less_for_values.md)**

:   本文从线性代数角度证明 Transformer 中 Key 权重的谱范数和 Frobenius 范数系统性大于 Value 权重，据此提出 Key 优先的混合精度 KV 缓存量化策略（如 K4V2），在减少 25% 内存的同时保持 98.3% 的全精度准确率。

**[Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty](model_compression/reason_only_when_needed_efficient_generative_reward_modeling_via_model-internal_.md)**

:   提出 E-GRM 框架，利用模型并行解码的收敛行为估计不确定性，仅在必要时触发 CoT 推理，并通过混合损失训练的判别式评分器精细评估推理路径质量，在多个奖励模型基准上实现 SOTA 同时降低 62% 推理延迟。

**[Rethinking Parameter Sharing for LLM Fine-Tuning with Multiple LoRAs](model_compression/rethinking_parameter_sharing_for_llm_fine-tuning_with_multiple_loras.md)**

:   本文推翻了“多 LoRA 应共享 A 矩阵”的常见假设，证明 A 的相似性主要来自相同初始化而非共享知识，并提出共享 B 矩阵的 ALoRA / Fed-ALoRA，在多任务和联邦微调中兼顾性能、均衡性与通信效率。

**[Rethinking Table Pruning in TableQA: From Sequential Revisions to Gold Trajectory-Supervised Parallel Search](model_compression/rethinking_table_pruning_in_tableqa_from_sequential_revisions_to_gold_trajectory.md)**

:   这篇论文提出 TabTrim，把表格剪枝从容易累积错误的单路径顺序修订改成“SQL 轨迹监督的剪枝器 + loss-aware verifier + 并行轨迹搜索”，在 WikiTQ、TabFact 和 TableBench 上把平均准确率提升到 73.5%，比最强基线高 3.2 个点。

**[RouteNLP: Closed-Loop LLM Routing with Conformal Cascading and Distillation Co-Optimization](model_compression/routenlp_closed-loop_llm_routing_with_conformal_cascading_and_distillation_co-op.md)**

:   RouteNLP 是一个闭环 LLM 路由与级联框架，用任务感知路由器、 conformal 校准级联和失败簇定向蒸馏共同优化模型组合，在六任务企业基准上以 0.971 质量比达到 0.159 成本比，并在 8 周客服试点中节省 58% 推理成本、保持 91% 响应可接受率。

**[SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](model_compression/samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)**

:   SAMoRA 通过语义感知路由器和任务自适应缩放机制，解决了现有 MoE-LoRA 方法中路由不精确和权重融合缺乏灵活性的问题，在多任务基准上以最少可训练参数（0.15%）达到 SOTA。

**[Social Story Frames: Contextual Reasoning about Narrative Intent and Reception](model_compression/social_story_frames_contextual_reasoning_about_narrative_intent_and_reception.md)**

:   这篇论文提出 SocialStoryFrames，用一个包含 10 个维度的读者反应 taxonomy 和两个蒸馏模型，把 Reddit 故事放回社区与对话上下文中推断其叙事意图、读者情感和价值判断，并在 6,140 条社交媒体故事上展示了比语义相似度更细的社区叙事实践分析。

**[SRA: Span Representation Alignment for Large Language Model Distillation](model_compression/sra_span_representation_alignment_for_large_language_model_distillation.md)**

:   SRA 把跨分词器 LLM 蒸馏的基本对齐单元从易碎的 token 换成 tokenizer-agnostic 的文本 span，通过 LCS 字符偏移匹配、注意力加权 center-of-mass 表示、几何结构正则和共享词表 span logit 蒸馏，在多组 teacher-student 压缩实验中稳定超过 ULD、MinED、DSKD 和 MultiLevelOT。

**[SSSD: Simply-Scalable Speculative Decoding](model_compression/sssd_simply-scalable_speculative_decoding.md)**

:   提出 SSSD，一种无需训练的推测解码方法，结合轻量级 n-gram 匹配与硬件感知投机长度调整，在无需任何草稿模型训练或部署的前提下，实现最高 2.9× 的推理加速，并在语言/领域迁移和长上下文场景中展现出优于训练式方法的鲁棒性。

**[Stable On-Policy Distillation through Adaptive Target Reformulation](model_compression/stable_on-policy_distillation_through_adaptive_target_reformulation.md)**

:   本文提出 Veto，一种目标层面的重构方法，通过在 logit 空间构建教师-学生的几何桥接分布来稳定 on-policy 知识蒸馏，单一参数 $\beta$ 同时在 forward KL 中充当自适应梯度否决器（抑制低置信度 token 的有害梯度）和在 reverse KL 中充当果断性旋钮（平衡奖励驱动和输出多样性），在 GSM8K 上比 SFT 提升 9.2%。

**[TalkLoRA: Communication-Aware Mixture of Low-Rank Adaptation for Large Language Models](model_compression/talklora_communication-aware_mixture_of_low-rank_adaptation_for_large_language_m.md)**

:   TalkLoRA 在 MoE-LoRA 架构中引入轻量级 Talking Module，允许低秩专家在路由前进行信息交换，解决传统 MoELoRA 中专家独立运行导致的路由不稳定和专家主导问题，在常识推理和 NLU 任务上以更少参数（0.2%）持续超越 LoRA 和 MoELoRA 变体。

**[Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs](model_compression/task-stratified_knowledge_scaling_laws_for_post-training_quantized_large_languag.md)**

:   本文建立了首个面向后训练量化（PTQ）的任务分层知识缩放定律，将 LLM 能力分为记忆/应用/推理三层，统一建模模型大小、位宽、组大小和校准集大小四个因素，在 293 种 PTQ 配置上验证，揭示推理对精度敏感、应用随规模提升、记忆对校准敏感的差异化规律。

**[TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination](model_compression/tell-tale_task_efficient_llms_with_task_aware_layer_elimination.md)**

:   TALE 用一个无需重训的贪心搜索过程，为每个下游任务直接删除“拖后腿”的 Transformer 层，在 5 个开源 LLM 和 9 个 benchmark 上同时提升任务准确率并减少推理成本。

**[The Pitfalls of KV Cache Compression](model_compression/the_pitfalls_of_kv_cache_compression.md)**

:   这篇论文指出 KV cache 压缩在多指令提示中会导致选择性遗忘和系统提示泄露，问题来自不同指令被不均匀驱逐以及关键 token 被错误删除，并提出白名单保留和 fair eviction 两种简单改造来显著降低泄露、稳定指令遵循。

**[TLoRA: Task-aware Low Rank Adaptation of Large Language Models](model_compression/tlora_task-aware_low_rank_adaptation_of_large_language_models.md)**

:   TLoRA 用训练样本激活协方差来初始化并冻结 LoRA 的 $A$ 矩阵，再按模块重要性自适应分配 rank 和 scaling factor，使 LLM 在 NLU、常识推理、数学、代码生成和聊天任务上用约一半可训练参数达到或超过主流 LoRA 变体。

**[Training-Free Test-Time Contrastive Learning for Large Language Models](model_compression/training-free_test-time_contrastive_learning_for_large_language_models.md)**

:   本文提出 TF-TTCL，一种无需梯度更新的测试时对比学习框架，通过"探索-反思-引导"循环让冻结的 LLM 在线自我改进——用多智能体角色扮演生成多样推理轨迹，从正负样本对比中蒸馏文本规则存入记忆库，推理时检索相关规则引导生成。

**[Two-Stage Regularization-Based Structured Pruning for LLMs](model_compression/two-stage_regularization-based_structured_pruning_for_llms.md)**

:   TRSP 用第一阶段正则学习每个 Transformer 层的重要性，再用第二阶段正则把待删层的输入输出拉近，让知识转移到保留层中，从而在无需重训练的情况下实现 LLM 层级结构化剪枝和实际推理加速。

**[UKP_Psycontrol at SemEval-2026 Task 2: Modeling Valence and Arousal Dynamics from Text](model_compression/ukp_psycontrol_at_semeval-2026_task_2_modeling_valence_and_arousal_dynamics_from.md)**

:   UKP_Psycontrol 在 SemEval-2026 Task 2 上取得双项第一，通过结合 LLM 提示、Ising 交互的 MaxEnt 模型和神经回归模型，发现 LLM 擅长捕捉静态情感信号而短期情感变化更多由近期数值轨迹而非文本语义解释。

**[VecCISC: Improving Confidence-Informed Self-Consistency with Reasoning Trace Clustering and Candidate Answer Selection](model_compression/veccisc_improving_confidence-informed_self-consistency_with_reasoning_trace_clus.md)**

:   VecCISC 在 CISC 的置信度加权自一致性之前加入“按答案分组的推理轨迹嵌入聚类”，只把每个语义簇的代表轨迹交给 critic 打分，从而在基本保持甚至略微提升准确率的同时显著减少 critic 调用和 token 成本。

**[When Reviews Disagree: Fine-Grained Contradiction Analysis in Scientific Peer Reviews](model_compression/when_reviews_disagree_fine-grained_contradiction_analysis_in_scientific_peer_rev.md)**

:   这篇论文把审稿意见分歧从句对级二分类推进到完整 review 上的证据抽取与强度评分，并用 IMPACT 多智能体教师蒸馏出单次前向即可部署的 TIDE 小模型。

**[Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics](model_compression/why_steering_works_toward_a_unified_view_of_language_model_parameter_dynamics.md)**

:   本文把局部权重微调、LoRA 和 activation steering 统一成“控制信号诱导的动态权重更新”，用 preference-utility log-odds 与 activation manifold 解释强控制会提升目标偏好但损害生成效用，并据此提出 SPLIT 训练目标，在三类干预形式上更好地兼顾偏好和效用。

**[WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](model_compression/wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)**

:   本文提出等价模型理论和 WISCA 权重缩放策略，通过在训练中动态调整 Transformer 注意力层的 $W_q/W_k$ 和 $W_v/W_o$ 权重使其 L1 范数相等（保持模型输出不变），将优化引导至更平坦的损失最小值区域，在 GQA 架构上实现平均 5.6% 的零样本评估提升和 2.12% 的训练困惑度降低。

---

## 💬 LLM / NLP { #llm_nlp }

**[A Study of LLMs' Preferences for Libraries and Programming Languages](llm_nlp/a_study_of_llms39_preferences_for_libraries_and_programming_languages.md)**

:   首次系统研究8个LLM在代码生成中对库和编程语言的偏好行为，发现LLM严重偏好NumPy等流行库（45%的使用不必要）和Python语言（58%的高性能任务仍选Python），且自然语言推荐与实际代码选择不一致。

**[Adam's Law: Textual Frequency Law on Large Language Models](llm_nlp/adam39s_law_textual_frequency_law_on_large_language_models.md)**

:   本文提出"文本频率定律"（TFL），发现当语义相同时，使用更高频率的文本表达来提示或微调LLM能获得更好效果，并设计了频率蒸馏和课程训练策略来进一步利用该规律。

**[AlphaContext: An Evolutionary Tree-based Psychometric Context Generator for Creativity Assessment](llm_nlp/alphacontext_an_evolutionary_tree-based_psychometric_context_generator_for_creat.md)**

:   提出 AlphaContext，一个基于进化树的心理测量情境生成器，通过 HyperTree 大纲规划、MCTS 逐句生成、MAP-Elites 多样性优化和评估引导迭代精炼四个模块，自动生成用于创造力评估的高质量长文本情境，在 7 个评估维度上平均超越竞争方法 8%。

**[An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal](llm_nlp/an_existence_proof_for_neural_language_models_that_can_explain_garden-path_effec.md)**

:   通过在花园路径句上微调神经语言模型，证明了存在一个神经 LM 能够通过惊奇度（surprisal）同时解释花园路径效应和自然阅读时间，为惊奇度理论提供了存在性证明。

**[Automatic Combination of Sample Selection Strategies for Few-Shot Learning](llm_nlp/automatic_combination_of_sample_selection_strategies_for_few-shot_learning.md)**

:   本文提出 ACSESS 方法，通过前向选择、后向选择和 Datamodels 三种机制自动识别互补的样本选择策略并加权组合，在 23 种策略、5 个 ICL 模型和 3 种梯度少样本学习方法、6 个文本和 8 个图像数据集上验证了组合策略一致优于单一策略和 ICL 专用基线。

**[Big AI is Accelerating the Metacrisis: What Can We Do?](llm_nlp/big_ai_is_accelerating_the_metacrisis_what_can_we_do.md)**

:   Steven Bird 在这篇 ACL 2026 立场论文里论证："Big AI"（少数巨头驱动的工业化 LLM 工程）正在同时加速 3 大相互纠缠的危机——**生态危机 / 意义危机 / 语言危机**——而 ACL 作为最大 LLM 研究发表方，必须从"个人合规"转向"职业共同体集体行动"，并提出 7 项面向 ACL 的具体改革建议（重申公共利益优先、抵御 corporate capture、保护批判性 NLP、设立 NLP policy track 等）。

**[C-World: A Computer Use Agent Environment Creator](llm_nlp/c-world_a_computer_use_agent_environment_creator.md)**

:   作者将"agent 环境"形式化为 Action / Task / Transition / Reward 四元组并实现为 C-World：用 5,571 个真实 MCP 工具 + 自动任务合成 + state controller 扰动 + 双信号 reward 提供高保真评测，又用一个"World Engine"在无 live API 下模拟工具响应实现可规模化训练；评测 9 个前沿 LLM 发现"规划普遍强、执行普遍弱"，仅用 1,170 条 C-World 轨迹微调即可超过用 119k 样本训练的 baseline。

**[Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future](llm_nlp/can_ai_be_a_good_peer_reviewer_a_survey_of_peer_review_process_evaluation_and_th.md)**

:   作者系统综述了 LLM 时代 AI 辅助 peer review 全流程的方法：把"review 生成"分为 fine-tuning / agent / RL / 生成增强 四大范式，把"after-review"分为 rebuttal / meta-review / paper revision 三类，再给出"human / reference-based / LLM-based / aspect-oriented"四象限评测分类法，最后从 novelty、自动评测、跨域、多模态、伦理 6 个方向讨论未来。

**[CAST: Achieving Stable LLM-based Text Analysis for Data Analytics](llm_nlp/cast_achieving_stable_llm-based_text_analysis_for_data_analytics.md)**

:   提出CAST框架，通过算法提示（Algorithmic Prompting）和先思考后输出（Thinking-before-Speaking）两种机制约束LLM的潜在推理路径，显著提升文本摘要和标注任务的运行间稳定性，同时不损失输出质量。

**[Characterizing the Expressivity of Local Attention in Transformers](llm_nlp/characterizing_the_expressivity_of_local_attention_in_transformers.md)**

:   作者用线性时序逻辑（LTL）作为统一刻画工具，严格证明 global-only Transformer ↔ $\mathrm{LTL}[\mathrm{P}]$、$k$-local-only ↔ $\mathrm{LTL}[\mathrm{Y}^{\leq k}]$、global+local 混合 ↔ $\mathrm{LTL}[\mathrm{P}, \mathrm{Y}^{\leq k}]$，并由此证明 **local 与 global 表达力互不包含**、混合严格更强、**1-local 是 local 家族里表达力最强**，最后在合成正则语言和 WikiText-2 上经验验证理论预测。

**[Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal](llm_nlp/clozing_the_gap_exploring_why_language_model_surprisal_outperforms_cloze_surpris.md)**

:   论文系统比较 cloze 反应和 GPT2 surprisal 对人类逐词阅读时间的解释力，并通过三类概率干预证明 LM surprisal 的优势主要来自更高概率分辨率、能区分语义相近词和能给低频词分配细粒度概率。

**[Clustered Self-Assessment: A Simple yet Effective Method for Uncertainty Quantification in Large Language Models](llm_nlp/clustered_self-assessment_a_simple_yet_effective_method_for_uncertainty_quantifi.md)**

:   这篇论文提出 Clustered Self-Assessment：先把 LLM 的多个采样答案按语义聚成互斥选项，再让同一个 LLM 通过多选题概率给原答案打置信分，从而在 TQA、NQ 和 XSum 上获得比语义熵、P(True) 等基线更好的 AUROC 与 Brier 校准表现。

**[CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models](llm_nlp/costomcausal-oriented_steering_for_intrinsic_theory-of-mind_alignment_in_large_l.md)**

:   提出 CoSToM 框架，先用因果追踪定位 LLM 中编码心智理论（ToM）特征的关键层（发现主要在早期层），再通过激活转向在这些层上进行轻量级对齐，使 LLM 在谈判和说服对话中显著提升社会推理质量——从"知道但不会用"变为"知道且会用"。

**[DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](llm_nlp/decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)**

:   提出 DeCoVec（Decoding Space based Task Vector），一个无训练、非侵入式的框架，通过对比 few-shot 和 zero-shot prompt 的输出 logit 分布差异构建解码空间中的任务向量，注入解码过程引导生成，在 TruthfulQA、Math-500 和 AQUA-RAT 上比标准 few-shot 基线平均提升高达 5.50 准确率。

**[Evaluating Customized vs. Generalist Transformer-based Models for Legal Contract Classification](llm_nlp/evaluating_customized_vs_generalist_transformer-based_models_for_legal_contract_.md)**

:   本文系统比较 13 个法律领域定制 Transformer 模型与 9 个通用模型在 3 个英文合同分类任务上的表现，发现小规模但合同相关预训练的 Legal-BERT、Contracts-BERT 等模型在长尾法律标签上通常优于更大的通用模型。

**[EVE: A Domain-Specific LLM Framework for Earth Intelligence](llm_nlp/eve_a_domain-specific_llm_framework_for_earth_intelligence.md)**

:   本文提出 EVE——欧空局 (ESA) Φ-lab 主导的首个面向 Earth Observation / Earth Sciences 的开源端到端 LLM 框架，包含 24B 领域适配的 EVE-Instruct（基于 Mistral Small 3.2 + 10.7B 合成 token 的 IFT/CPT 交替微调 + 10 个 checkpoint 融合）、首批 5693 条人工 EO 评测 benchmark、RAG + 幻觉检测 pipeline，已在 6 个月 pilot 中服务 350 位用户。

**[Expect the Unexpected? Testing the Surprisal of Salient Entities](llm_nlp/expect_the_unexpected_testing_the_surprisal_of_salient_entities.md)**

:   本文研究全局显著实体（discourse-level salient entities）与惊异度（surprisal）的关系，通过 70K+ 手工标注的实体提及和新颖的最小对提示方法，发现全局显著实体本身更出人意料（更高 surprisal），但它们系统性地降低周围内容的 surprisal，且该效应随体裁变化——话题连贯性高的文本中效应最强。

**[FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation](llm_nlp/fastdiss_few-step_match_many-step_diffusion_language_model_on_sequence-to-sequen.md)**

:   本文分析了连续扩散语言模型在少步采样时自条件化信号的不匹配和训练饱和两个瓶颈，提出FastDiSS框架通过自条件化扰动（SCP）和模型感知噪声缩放（MANS）来改善鲁棒性，在6个基准上实现4×-400×加速同时保持质量。

**[Foresight Optimization for Strategic Reasoning in Large Language Models](llm_nlp/foresight_optimization_for_strategic_reasoning_in_large_language_models.md)**

:   本文提出 Foresight Policy Optimization（FoPO），通过在策略优化中引入对手建模的前瞻修正项，使 LLM 能够显式预见对手行为并据此调整自身策略，在合作（Cooperative RSA）和竞争（Competitive Taboo）两类博弈任务上显著提升策略推理能力，并在跨域 γ-Bench 上取得一致性提升。

**[From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?](llm_nlp/from_fallback_to_frontline_when_can_llms_be_superior_annotators_of_human_perspec.md)**

:   本文把"perspective-taking (PT)"这个长期被视为人类专属的主观标注任务重新形式化为「对潜在群体均值 $f^*(x,g)$ 的统计估计问题」，用偏差-方差-相关性三项分解证明 LLM 在低预算 / 群体宽泛 / out-group 场景下不只是廉价替代品，而是**比 in-group 人类标注者更优**的估计器，并发现"打开 reasoning 反而变差"的 reasoning paradox。

**[From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](llm_nlp/from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)**

:   本文首次系统综述流式大语言模型（Streaming LLMs），提出基于数据流和交互并发性的统一定义，将现有方法分为三级递进分类——输出流式（Output-streaming）、顺序流式（Sequential-streaming）和并发流式（Concurrent-streaming），覆盖文本、语音和视频流式场景的方法论和应用。

**[Generative Floor Plan Design with LLMs via Reinforcement Learning with Verifiable Rewards](llm_nlp/generative_floor_plan_design_with_llms_via_reinforcement_learning_with_verifiabl.md)**

:   作者把 RPLAN 中 8 万套真实公寓户型转成 JSON 多边形格式，用 Llama-3.3-70B-Instruct 做两阶段训练（SFT + GRPO with verifiable rewards：connectivity + total-area 奖励，重叠/解析失败硬置零），让 LLM 输出能同时满足 bubble diagram 拓扑约束和数值面积约束的 CAD-ready 户型，在 8 房间任务上 Compatibility 相比 HouseDiffusion 下降 94%（2.5 → 0.15）。

**[Generative Interfaces for Language Models](llm_nlp/generative_interfaces_for_language_models.md)**

:   本文提出 **Generative Interfaces (GenUI)**，让 LLM 不再用单一聊天框回复用户，而是基于"交互流图 + 有限状态机"的结构化中间表示和"自适应奖励驱动的迭代精化"，在线生成一个为查询量身定制的可交互 Web 界面，在 100 条 UIX prompt 上相比 Claude 3.7 聊天 UI 取得 **84%** 的总体偏好胜率。

**[GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](llm_nlp/grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)**

:   提出 GRASS 框架，使用均值梯度范数（MGN）作为任务感知和训练阶段感知的层重要性指标，自适应地采样和更新模型层子集进行微调，配合层级优化器状态卸载机制，在平均准确率提升最高 4.38 分的同时减少最高 19.97% 的内存使用。

**[Hot-Start from Pixels: Low-Resolution Visual Tokens for Chinese Language Modeling](llm_nlp/hot-start_from_pixels_low-resolution_visual_tokens_for_chinese_language_modeling.md)**

:   把中文字符渲染成 $8\times 8$ 灰度小图喂给 GPT-2 风格解码器做 next-character 预测，最终精度（39.21%）追平 index-based 基线（39.10%），并且在训练早期（0.4% 数据时）就把基线翻一倍以上——展示了"视觉结构"作为汉字建模的天然 hot-start 先验。

**[SteerEval: How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities](llm_nlp/how_controllable_are_large_language_models_a_unified_evaluation_across_behaviora.md)**

:   SteerEval 把 LLM 可控性按 Marr 的三层分析框架拆成 L1（表达什么）/L2（怎么表达）/L3（具体落到哪个词），覆盖 Personality、Sentiment、Language Features 三个域共 7560 个 paired sample，系统揭示了"细粒度上现有 steering 方法普遍崩溃"这一关键缺口。

**[How Do Answer Tokens Read Reasoning Traces? Self-Reading Patterns in Thinking LLMs](llm_nlp/how_do_answer_tokens_read_reasoning_traces_self-reading_patterns_in_thinking_llm.md)**

:   本文发现推理 LLM（如 DeepSeek-R1）在定量推理中存在"良性自读"模式——答案 token 对推理痕迹的注意力呈现前移漂移（沿推理链逐步推进）和语义锚点集中（反复回顾关键步骤），且此模式与正确性强相关；基于此提出 SRQ（自读质量）驱动的免训练激活引导方法，在多个基准上提升准确率最高 2.6%。

**[Identifying the Periodicity of Information in Natural Language](llm_nlp/identifying_the_periodicity_of_information_in_natural_language.md)**

:   本文把信号处理领域的 AutoPeriod 周期检测算法搬到 token-surprisal 序列上，提出 APS（AutoPeriod of Surprisal）能在单文档级别**直接检测**出自然语言信息密度的周期（如 "每 53 个 token 一个周期"），发现人类文本中约 11% 的文档存在严格周期，且 LLM 生成文本的周期性比人类强 2 倍（30% vs 14.8%），为 UID 理论提供了直接证据并给 AI 文本检测提供了可解释特征。

**[Iterative Formalization and Planning in Partially Observable Environments](llm_nlp/iterative_formalization_and_planning_in_partially_observable_environments.md)**

:   提出 PDDLego+ 框架，让 LLM 在部分可观测环境中迭代地生成和修正 PDDL（规划领域定义语言）表示，通过双层错误修复循环（solver error + simulation error）实现无需微调、无需示例的有效规划。

**[Leveraging Pretrained Language Models as Energy Functions for Glauber Dynamics Text Diffusion](llm_nlp/leveraging_pretrained_language_models_as_energy_functions_for_glauber_dynamics_t.md)**

:   本文用统计物理中的 Glauber 动力学构造离散文本扩散：把预训练 UL2 模型当成"能量函数 / 噪声分布"，每一步把 mask infilling 当 Markov 转移核，训出的 Glauber-UL2 在生成 perplexity 上**首次匹配 GPT-2-M/L 同尺寸 AR 模型**，并在 Sudoku/Zebra 等搜索规划任务上击败 MDLM，同 compute 下 best-of-N 也优于 AR。

**[LoCar: Localization-Aware Evaluation of In-Vehicle Assistants through Fine-Grained Sociolinguistic Control](llm_nlp/locar_localization-aware_evaluation_of_in-vehicle_assistants_through_fine-graine.md)**

:   LoCar 面向韩语车载助手提出 13 个部署级 KPI，并用人工校准的 LLM-as-a-Judge 与敬语形态验证来评测 11 个模型，发现通用理解能力接近饱和，但细粒度敬语控制和多轮策略性引导仍明显不稳定。

**[Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness](llm_nlp/masked_by_consensus_disentangling_privileged_knowledge_in_llm_correctness.md)**

:   本文通过对比自探针（使用模型自身隐藏状态）和外部探针（使用其他模型隐藏状态）预测正确性的能力，发现"模型间一致性"是掩盖特权知识的关键混淆因子，在消除一致性后揭示了领域特异性的特权知识：事实性任务中存在但数学推理中不存在。

**[Min-k Sampling: Decoupling Truncation from Temperature Scaling via Relative Logit Dynamics](llm_nlp/min-k_sampling_decoupling_truncation_from_temperature_scaling_via_relative_logit.md)**

:   Min-k Sampling 通过分析排序 logit 分布的局部结构来检测"语义悬崖"（高置信候选与低质量尾部噪声的分界点），实现了严格的温度不变性截断，在极端温度下仍保持稳健的推理和创意写作质量。

**[Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models](llm_nlp/mind_the_gap_how_elicitation_protocols_shape_the_stated-revealed_preference_gap_.md)**

:   作者在 LitmusValues / AIRiskDilemmas 框架上把 forced-choice 升级为 expanded-choice（允许 "Equal Preference" 和 "Depends" 中立选项），系统评测 24 个 LLM 发现：在 stated 侧允许中立可以把 SvR Spearman 相关 ρ 从 ~0.2 大幅提升到 ~0.7（因为过滤掉了模型本来就没立场的弱信号），但 revealed 侧也允许中立反而把 ρ 打到接近 0 甚至负值（很多模型在情境化场景里几乎全选 Depends/Equal），同时验证基于 stated ranking 的 system prompt steering 在 16-value 大集合下普遍不可靠——结论是 **SvR gap 高度依赖 elicitation protocol，偏好评测必须显式建模"无定见"状态**。

**[Model-Agnostic Meta Learning for Class Imbalance Adaptation](llm_nlp/model-agnostic_meta_learning_for_class_imbalance_adaptation.md)**

:   本文提出 HAMR（Hardness-Aware Meta-Resample），一个统一的元学习框架，通过双层优化动态估计实例级权重优先处理真正困难的样本，配合邻域感知重采样机制将训练焦点放在困难样本及其语义邻居上，在 6 个不平衡 NLP 数据集上持续超越强基线。

**[MoRI: Learning Motivation-Grounded Reasoning for Scientific Ideation in Large Language Models](llm_nlp/mori_learning_motivation-grounded_reasoning_for_scientific_ideation_in_large_lan.md)**

:   把科学 ideation 显式建模为「context → motivation → reasoning → method」的两阶段条件推理任务，在 SFT 冷启动基础上用 GRPO + 一对新型可验证奖励（**熵感知信息增益 EAIG** + **对比语义增益 CSG**）训练 14B 模型，让其在 ICLR/NeurIPS 留出测试集上同时超越 GPT-4o、Claude-3.5-Sonnet 与 AI-Scientist-V2 等 agentic 框架。

**[MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models](llm_nlp/muldimif_a_multi-dimensional_constraint_framework_for_evaluating_and_improving_i.md)**

:   提出 MulDimIF 多维约束框架，从约束模式（3种）、约束类别（4类13子类）和约束难度（4级）三个维度系统评估 LLM 的指令遵循能力，并通过 GRPO 训练显著提升模型性能，发现改进主要源自注意力模块的参数更新。

**[Not All Animals Are Equal: Metaphorical Framing through Source Domains and Semantic Frames](llm_nlp/not_all_animals_are_equal_metaphorical_framing_through_source_domains_and_semant.md)**

:   本文提出首个结合 FrameNet 语义框架和概念隐喻理论（CMT）源域的计算框架 ConceptFrameMet，通过 RoBERTa 多任务模型检测隐喻并预测其语义框架和源域，配合对数似然比统计方法发现话语中显著的隐喻模式，揭示了自由派和保守派在移民话语中使用相同源域但选择不同语义框架来传达截然不同的联想。

**[Nürnberg NLP at PsyDefDetect: Multi-Axis Voter Ensembles for Psychological Defence Mechanism Classification](llm_nlp/nürnberg_nlp_at_psydefdetect_multi-axis_voter_ensembles_for_psychological_defenc.md)**

:   这篇 BioNLP 2026 PsyDefDetect 共享任务系统论文把心理防御机制分类看成一个边界模糊、标注一致性有限的问题，用 9 个跨粒度、跨训练方式、跨基座模型的投票器做集成，在隐藏测试集上取得 F1=.420，并在 21 支注册队伍中排名第一。

**[One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization](llm_nlp/one_persona_many_cues_different_results_how_sociodemographic_cues_impact_llm_per.md)**

:   本文系统比较了 6 种常用的人物画像提示方式（姓名/显式提及/对话历史各两种变体）在 7 个 LLM 和 4 个任务上的效果，发现虽然平均响应跨提示方式高度相关，但不同提示方式产生的人物画像间差异显著不同，过于显式的提示导致更强的个性化偏差，警示不应基于单一提示方式得出偏差结论。

**[Overcoming Copyright Barriers in Corpus Distribution Through Non-Reversible Hashing](llm_nlp/overcoming_copyright_barriers_in_corpus_distribution_through_non-reversible_hash.md)**

:   本文提出 novelshare：把受版权保护文本的 token 变成截断后的非可逆哈希，并只公开哈希序列与研究者自有标注，使拥有合法原文的用户可以在轻微版本差异下重新对齐标注，在近版本小说上达到 98.7% 到 99.79% 的 token 正确对齐率。

**[PersonaArena: Dynamic Simulation for Evaluating and Enhancing Persona-Level Role-Playing in Large Language Models](llm_nlp/personaarena_dynamic_simulation_for_evaluating_and_enhancing_persona-level_role-.md)**

:   PersonaArena 用真实用户生成内容构造 1,000 个细粒度 persona，并通过动态社交仿真与多评委辩论来评估和增强 LLM 的 persona-level 角色扮演能力。

**[Prefix Parsing is Just Parsing](llm_nlp/prefix_parsing_is_just_parsing.md)**

:   本文提出**前缀文法变换**（prefix grammar transformation），一种将前缀解析归约为普通解析的高效方法——给定一个文法，构造另一个恰好生成其所有前缀字符串的新文法，从而可以直接复用任何现有的普通解析算法，无需设计专用的前缀解析算法。

**[Repeated Sequences Reveal Gaps between Large Language Models and Natural Language](llm_nlp/repeated_sequences_reveal_gaps_between_large_language_models_and_natural_languag.md)**

:   本文提出基于重复子序列分布的评估框架，通过高阶 Rényi 熵刻画文本的熵增长行为，发现自然语言呈现稳定的次线性熵增长模式，而 GPT 生成文本的熵指数随模型规模单调增大，揭示了 LLM 在长程统计组织上与自然语言的系统性差异。

**[Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design](llm_nlp/solver-independent_automated_problem_formulation_via_llms_for_high-cost_simulati.md)**

:   本文提出 APF（Automated Problem Formulation），一种与求解器无关的框架，利用 LLM 将工程师的自然语言设计需求转化为可执行的数学优化模型，通过创新的数据生成和测试实例标注管线克服高成本仿真场景下无法使用求解器反馈筛选数据的困难，在天线设计任务上显著优于现有方法。

**[Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning](llm_nlp/synthetic_eggs_in_many_baskets_the_impact_of_synthetic_data_diversity_on_llm_fin.md)**

:   这篇论文系统比较了单一模型、多模型和人类数据作为监督微调来源时对 Llama 模型的影响，发现多源合成数据能缓解分布坍缩和自偏好，但合成数据也可能在保留输出质量的同时削弱安全护栏，且风险会随源模型规模和混合方式发生复杂变化。

**[Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context](llm_nlp/text-to-distribution_prediction_with_quantile_tokens_and_neighbor_context.md)**

:   本文提出Quantile Token Regression方法，通过在输入序列中插入专用分位数token并结合检索到的邻居实例及其经验分布，使LLM能够预测完整的条件分布而非单一点估计，在Airbnb和StackSample数据集上相比基线降低约4个MAPE点并将预测区间收窄2倍以上。

**[Think in Sentences: Explicit Sentence Boundaries Enhance Language Model's Capabilities](llm_nlp/think_in_sentences_explicit_sentence_boundaries_enhance_language_model39s_capabi.md)**

:   本文提出在 LLM 输入中的句子边界处插入分隔符标记，通过 ICL 和 SFT 两种方式实现"逐句思考"的推理范式，在 7B 到 600B 模型上取得一致提升（GSM8k +7.7%，DROP +12.5%），且几乎不增加额外计算开销。

**[TingIS: Real-time Risk Event Discovery from Noisy Customer Incidents at Enterprise Scale](llm_nlp/tingis_real-time_risk_event_discovery_from_noisy_customer_incidents_at_enterpris.md)**

:   TingIS是一个部署在金融科技平台的端到端风险事件发现系统，通过五模块架构（语义蒸馏、级联路由、事件链接引擎、状态管理、多维降噪）从海量嘈杂客户投诉中实时提取可操作的风险事件，实现了P90告警延迟3.5分钟和95%的高优先级事件发现率。

**[面向稳健上下文学习的 OOD 代理演示检索方案](llm_nlp/toward_robust_in-context_learning_leveraging_out-of-distribution_proxies_for_tar.md)**

:   通过构造源域和目标域的双代理并计算其困惑度差值作为 OOD 得分，结合马哈拉诺比斯距离约束演示多样性，在无法获取目标域样本的条件下，从源域中精准筛选与目标域分布对齐的演示样本，增强 LLM 的上下文学习鲁棒性。

**[UCS: Estimating Unseen Coverage for Improved In-Context Learning](llm_nlp/ucs_estimating_unseen_coverage_for_improved_in-context_learning.md)**

:   本文提出 UCS（Unseen Coverage Selection），一种基于 Smoothed Good-Turing 估计器的无训练子集级覆盖率先验，通过估计候选示例集中未观测到的潜在聚类数量来正则化现有 ICL 示例选择方法，在意图分类和推理任务上提升 2-6% 准确率。

**[Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection](llm_nlp/understanding_structured_financial_data_with_llms_a_case_study_on_fraud_detectio.md)**

:   本文提出 FinFRE-RAG，一种两阶段框架，通过重要性引导的特征降维将高维表格交易数据序列化为自然语言，并结合标签感知的检索增强上下文学习，使开源 LLM 在金融欺诈检测上的 F1/MCC 大幅提升，缩小了与专用表格分类器的性能差距。

**[Unlocking the Potential of Diffusion Language Models through Template Infilling](llm_nlp/unlocking_the_potential_of_diffusion_language_models_through_template_infilling.md)**

:   本文提出 Template Infilling，把扩散语言模型的生成条件从单一前缀改成分布在整段输出中的结构锚点，并用动态片段分配给复杂推理留出空间，从而在数学推理、代码生成和全局规划任务上显著稳定并提升并行生成质量。

**[VOYAGER: A Training Free Approach for Generating Diverse Datasets using LLMs](llm_nlp/voyager_a_training_free_approach_for_generating_diverse_datasets_using_llms.md)**

:   Voyager 是一个无需训练的 LLM 数据生成算法，用 DPP 维护多样 anchor 和 explorer，并用 textual gradients 迭代改写提示词，从而在创意写作和推理数据生成中显著提高 Vendi 多样性且基本不牺牲质量。

**[等等，还有出路：一个对话脱轨预测的决策机制](llm_nlp/wait_theres_a_way_out_a_decision_mechanism_for_forecasting_conversational_derail.md)**

:   论文将对话脱轨预测中的"信度估计"与"触发决策"解耦，通过前向模拟识别可恢复的紧张时刻，实现显著降低虚警率（从 36.2% 降至 26.7%）而不牺牲准确性。

**[当梯度相撞：多目标提示优化对 LLM 评判员的失效模式](llm_nlp/when_gradients_collide_failure_modes_of_multi-objective_prompt_optimization_for_.md)**

:   本文系统研究了用文本梯度方法同时优化多个评估准则的提示时的失效模式，发现梯度稀释和指令干扰两个关键瓶颈导致多目标优化几乎无法改进初始提示。

**[当 TableQA 遇到噪声：复杂问题与大表的双重去噪框架](llm_nlp/when_tableqa_meets_noise_a_dual_denoising_framework_for_complex_questions_and_la.md)**

:   通过分解问题中的语义单元并构建证据树进行透明化表格修剪，EnoTab 框架在处理复杂问题和超大表格时实现显著性能提升，通过双重去噪机制有效缓解噪声数据对推理的负面影响。

**[Why Did Apple Fall: Evaluating Curiosity in Large Language Models](llm_nlp/why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)**

:   本文提出首个系统评估 LLM 好奇心行为的心理学启发框架，结合问卷自评和行为实验发现 LLM 展现出好奇心般的行为模式但并非内在特质，并设计好奇心驱动的提问管道证明模拟好奇行为可提升下游推理性能。

---

## 💻 代码智能 { #code_intelligence }

**[Across Programming Language Silos: A Study on Cross-Lingual Retrieval-Augmented Code Generation](code_intelligence/across_programming_language_silos_a_study_on_cross-lingual_retrieval-augmented_c.md)**

:   首次系统研究跨编程语言的检索增强代码生成（RACG），构建覆盖13种编程语言的14K实例数据集，揭示跨语言知识迁移的不对等性及其与语言亲缘性和预训练多样性的关系。

**[Aligned Multi-View Scripts for Universal Chart-to-Code Generation](code_intelligence/aligned_multi-view_scripts_for_universal_chart-to-code_generation.md)**

:   把"同一张图表用 Python / R / LaTeX 三种语言写出语义等价脚本"作为新的监督信号，构建了 176K 四元组数据集 Chart2NCode，并提出在 LLaVA 投影器上加一个"语言条件的低秩子空间路由"的轻量适配器 CharLuMA，让一个模型在三种绘图语言上都达到可执行率与视觉保真度双高的水准。

**[AutoMonitor-Bench: Evaluating the Reliability of LLM-Based Misbehavior Monitor](code_intelligence/automonitor-bench_evaluating_the_reliability_of_llm-based_misbehavior_monitor.md)**

:   本文构建首个系统性评测「LLM 监控器是否能可靠识别模型不当行为」的基准 AutoMonitor-Bench（3,010 个配对样本，覆盖安全违规 / 谄媚偏见 / 规约博弈三类），在 22 个开闭源监控模型上揭示了漏检率（MR）与误报率（FAR）之间的系统性 trade-off，并通过 153k 样本 SFT 实验证明：在易构造不当行为上微调难以泛化到隐式的规约博弈。

**[Benchmarking Testing in Automated Theorem Proving](code_intelligence/benchmarking_testing_in_automated_theorem_proving.md)**

:   借鉴软件工程「集成测试」思想，把生成定理的语义正确性判定为「所有依赖于它的后继定理是否仍能编译通过」，构建 2206 题的 Lean 4 基准 T2，揭示出主流 LLM 编译通过率 80%+ 但语义正确率只有 ~39% 的巨大缝隙。

**[Bootstrapping Code Translation with Weighted Multilanguage Exploration](code_intelligence/bootstrapping_code_translation_with_weighted_multilanguage_exploration.md)**

:   BootTrans 提出了一种自举式多语言代码翻译方法，通过利用单一枢纽语言（Python）的测试用例作为跨语言验证预言，结合双池架构进行经验收集扩展训练数据，并设计语言感知加权机制动态优先处理困难的翻译方向，在 HumanEval-X 和 TransCoder-Test 上显著超越基线。

**[Can LLMs Compress (and Decompress)? Evaluating Code Understanding and Execution via Invertibility](code_intelligence/can_llms_compress_and_decompress_evaluating_code_understanding_and_execution_via.md)**

:   本文提出 RoundTripCodeEval (RTCE)：用 4 种无损压缩算法（LZW/AE/RLE/Huffman）构造 250 输入 × 4 子任务 = 1000 个严格回环（encode→decode 必须 bit-精确还原）的代码推理基准，结果显示即使是 QwQ-32B 在 Huffman 编码上 EM 仍为 0%，SFT 和 self-reflection 都救不回来。

**[ChatHLS: Towards Systematic Design Automation and Optimization for High-Level Synthesis](code_intelligence/chathls_towards_systematic_design_automation_and_optimization_for_high-level_syn.md)**

:   ChatHLS 提出了一个多智能体 HLS 设计框架，通过 HLSTuner（QoR 感知推理优化指令选择）和 HLSFixer（分层反馈增强的调试框架）两个核心组件，结合自进化错误用例扩展机制（VODA），在 HLS-C 生成成功率和硬件性能优化上显著超越基线。

**[ChipSeek: Optimizing Verilog Generation via EDA-Integrated Reinforcement Learning](code_intelligence/chipseek_optimizing_verilog_generation_via_eda-integrated_reinforcement_learning.md)**

:   ChipSeek 提出了一个将 EDA 工具链直接集成到训练循环中的分层奖励 RL 框架，通过课程引导的动态策略优化（CDPO）使 LLM 能够生成同时满足功能正确性和 PPA（功耗-性能-面积）优化的 RTL 代码，在标准基准上达到 SOTA。

**[CodeDistiller: Automatically Generating Code Libraries for Scientific Coding Agents](code_intelligence/codedistiller_automatically_generating_code_libraries_for_scientific_coding_agen.md)**

:   CodeDistiller 自动把科学领域 GitHub 仓库蒸馏成可运行、经调试的示例代码库，让 Code-RAG 式科学发现智能体能调用真实领域工具；在 250 个材料科学仓库上，最佳模型的人工验证正确功能率达到 74.1%，下游发现任务也更受专家偏好。

**[CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](code_intelligence/coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)**

:   本文提出 CodeRL+，将执行语义对齐集成到 RLVR 训练管道中，通过让模型推断变量级执行轨迹来弥合代码文本表示与执行语义之间的差距，在代码生成上平均 pass@1 提升 4.6%，在代码推理和测试输出生成基准上分别提升 15.5% 和 4.4%。

**[CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases](code_intelligence/codewiki_evaluating_ai39s_ability_to_generate_holistic_documentation_for_large-s.md)**

:   提出 CodeWiki，一个基于层次化分解和递归多智能体处理的开源框架，用于自动生成仓库级代码文档，并构建了 CodeWikiBench 基准，在七种编程语言上以 68.79% 的质量分数超越了闭源系统 DeepWiki（64.06%）。

**[CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](code_intelligence/collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)**

:   本文提出 CollabCoder，一个计划-代码共演化框架，通过协作决策模块（CDM）判断错误应在计划层还是代码层修复，结合推理轨迹模块（RT）实现从错误中学习的自改进调试，在复杂编程基准上比强基线提升 11-20%，同时减少 4-10 次 API 调用。

**[CuBridge: An LLM-Based Framework for Understanding and Reconstructing High-Performance Attention Kernels](code_intelligence/cubridge_an_llm-based_framework_for_understanding_and_reconstructing_high-perfor.md)**

:   作者把"用 LLM 直接改 FlashAttention CUDA 代码"这件不靠谱的事，重写成"lift 到可执行 IR（CuIR）→ 按 PyTorch reference transfer → 差分式 lower 回 CUDA"三段式工作流，在 A100/H100 上对 8 类 attention 变体保持 100% 正确率，相对 PyTorch 平均 16.03×、相对 FlexAttention 1.39×、相对前一代 LLM-based 方法 Qimeng-Attention 3.33× 加速。

**[DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation](code_intelligence/deepguard_secure_code_generation_via_multi-layer_semantic_aggregation.md)**

:   提出 DeepGuard，通过注意力机制聚合 Transformer 上层多层表示克服"最终层瓶颈"问题，结合多目标训练和轻量推理时安全引导策略，在 5 个代码 LLM 上将安全-正确生成率平均提升 11.9%。

**[Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4](code_intelligence/discover_and_prove_an_open-source_agentic_framework_for_hard_mode_automated_theo.md)**

:   DAP 提出了 Hard Mode ATP 的概念（AI 必须自行发现答案再构造证明，而非使用嵌入答案的 Easy Mode 声明），发布了 MiniF2F-Hard 和 FIMO-Hard 基准，并设计了"发现+证明"两阶段框架——用 LLM 自然语言推理发现答案后改写为 Easy Mode 声明交给形式化证明器，在 CombiBench 上将解题数从 7 提升到 10，首次在 PutnamBench Hard Mode 上证明 36 个定理。

**[DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency](code_intelligence/dpc_training-free_text-to-sql_candidate_selection_via_dual-paradigm_consistency.md)**

:   DPC 将 Text-to-SQL 的候选选择从"在隐藏数据上猜测"转化为"在可见数据上确定性验证"：构造最小区分数据库（MDD）使冲突 SQL 产生不同结果，再用 Python/Pandas 解作为参考锚点通过跨范式一致性选择正确候选，在 BIRD 和 Spider 上超越 Self-Consistency 最高 2.2%。

**[DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](code_intelligence/duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)**

:   本文提出 DUET，一个结合直接代码执行和 LLM 伪代码执行的双路框架，通过功能多数投票融合两种互补的执行路径——前者在代码正确时可靠但受实现错误影响，后者绕过实现细节但可能产生执行幻觉——在 LiveCodeBench 测试输出预测上提升 Pass@1 13.6 个百分点。

**[EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents](code_intelligence/eet_experience-driven_early_termination_for_cost-efficient_software_engineering_.md)**

:   提出 EET——一种基于历史经验驱动的早停方法，在补丁生成和补丁选择阶段识别无效迭代并提前终止，将 SE Agent 总成本降低 19%-55%（平均 32%），同时几乎不损失任务性能（最多 0.2%）。

**[FormalScience: Scalable Human-in-the-Loop Autoformalisation of Science with Agentic Code Generation in Lean](code_intelligence/formalscience_scalable_human-in-the-loop_autoformalisation_of_science_with_agent.md)**

:   FormalScience 提出一个领域无关的人在环 (HITL) Agent 流水线，让单个领域专家在不会 Lean 的情况下，把非正式的科学推理（特别是物理）转写成 100% 编译通过的 Lean4 代码，并构建出首个 200 题大学物理形式化基准 FormalPhysics，系统刻画了「编译通过」但「语义漂移」的现象。

**[From Charts to Code: A Hierarchical Benchmark for Multimodal Models](code_intelligence/from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)**

:   本文提出 Chart2Code，一个包含 2,186 个任务、覆盖 22 种图表类型的层次化基准，分为图表复现（Level 1）、图表编辑（Level 2）和长表格转图表（Level 3）三个递进难度级别，评测 29 个 SOTA 多模态模型，发现即使最强的 GPT-5.2 在编辑任务上的图表质量评分仅 33.41，揭示了当前模型在实际图表代码生成中的显著不足。

**[From If-Statements to ML Pipelines: Revisiting Bias in Code-Generation](code_intelligence/from_if-statements_to_ml_pipelines_revisiting_bias_in_code-generation.md)**

:   揭示LLM代码生成的偏差评估严重低估了实际风险：在ML流水线生成中，敏感属性出现在87.7%的特征选择中（vs 条件语句中的59.2%），且模型能正确排除无关特征但仍选择保留种族、性别等敏感属性，显示出系统性的隐性歧视。

**[KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?](code_intelligence/koco-bench_can_large_language_models_leverage_domain_knowledge_in_software_devel.md)**

:   KoCo-Bench 提出首个包含显式领域知识语料库的代码基准，覆盖 6 个新兴领域（RL、Agent、RAG 等）的 11 个框架和 25 个项目，评估 LLM 从知识语料库中获取和应用领域知识进行代码生成和知识理解的能力，揭示即使最强 coding agent Claude Code 也仅达 34.2%。

**[Learning Adaptive Parallel Execution for Efficient Code Localization](code_intelligence/learning_adaptive_parallel_execution_for_efficient_code_localization.md)**

:   FuseSearch 把代码定位中的并行工具调用建模为质量-效率联合优化问题，用 SFT+RL 学会按任务阶段自适应调节搜索宽度，在 SWE-bench Verified 上用紧凑模型取得高 F1 和显著更低的时间/Token 成本。

**[LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](code_intelligence/logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)**

:   本文构建了首个针对逻辑漏洞的修复评估框架 LogicEval 和数据集 LogicDS（61 个真实逻辑漏洞 + 61 个合成 Java 样本），系统评估了传统 AVR 工具和 LLM 在修复逻辑漏洞上的能力，发现 LLM 在提供辅助信息时表现最佳但整体修复率仍然很低（61 个真实样本中仅正确修复 5 个），并识别了提示敏感性、上下文丢失和补丁定位困难等关键瓶颈。

**[MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)**

:   MARS2 提出多智能体强化树搜索框架，将多个独立优化的策略嵌入共享搜索树中协作探索，通过 Thompson 采样选择智能体-节点对、树一致性奖励塑形和路径级组优势估计，在代码生成基准上一致提升单模型 Pass@1 最高 8.0%、系统级 Pass@1(MCTS) 最高 6.5%。

**[OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward](code_intelligence/omnidiagram_advancing_unified_diagram_code_generation_via_visual_interrogation_r.md)**

:   本文提出 OmniDiagram，一个统一的图表代码生成框架，覆盖 LaTeX/Mermaid/PlantUML 三种语言和图表转代码/图表编辑/文本转代码三种任务，并引入基于视觉问答的 Viva 奖励机制来指导 RL 训练，在多个基准上达到 SOTA。

**[PaT: Planning-after-Trial for Efficient Test-Time Code Generation](code_intelligence/pat_planning-after-trial_for_efficient_test-time_code_generation.md)**

:   PaT 将代码生成中的“先规划再尝试”改成“先尝试、失败后再规划”，用执行反馈触发昂贵的分解步骤，并用小模型生成、大模型规划的异构配置显著改善 Pass@1 与推理成本之间的折中。

**[PExA: Parallel Exploration Agent for Complex Text-to-SQL](code_intelligence/pexa_parallel_exploration_agent_for_complex_text-to-sql.md)**

:   PExA 把复杂 Text-to-SQL 改写成“为自然语言查询生成并执行一组语义测试用例”的并行探索问题，通过 Planner、Test Case Generator 和 SQL Proposer 三个子代理在 Spider 2.0 上提升执行准确率，并把延迟控制在与强基线相近的水平。

**[Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?](code_intelligence/precise_debugging_benchmark_is_your_model_debugging_or_regenerating.md)**

:   本文揭示前沿 LLM 在调试任务中的"重生成"倾向——通过引入 PDB 框架和编辑级精度/bug 级召回指标，发现 GPT-5.1-Codex 等模型虽能通过 76% 以上单元测试，但编辑精度不足 45%，且迭代和 agent 调试策略也无法显著改善精度。

**[PV-SQL: Synergizing Database Probing and Rule-based Verification for Text-to-SQL Agents](code_intelligence/pv-sql_synergizing_database_probing_and_rule-based_verification_for_text-to-sql_.md)**

:   本文提出 PV-SQL，一个 Agent 式 Text-to-SQL 框架，通过 Probe（迭代生成探测查询发现数据库值格式/列语义/表关系）和 Verify（基于模式匹配提取可验证约束并构建检查清单）两个互补组件，在 BIRD 基准上比最佳基线高 5% 执行准确率和 20.8% 有效效率分。

**[QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions](code_intelligence/qaq_bidirectional_semantic_coherence_for_selecting_high-quality_synthetic_code_i.md)**

:   QAQ 从“答案能否反推出问题”的反向语义一致性出发，用分层 RMI 与强弱模型分歧筛选合成代码指令，只用 25% WarriorCoder 数据就接近全量训练，并显著优于 IFD 等传统数据选择指标。

**[QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](code_intelligence/qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)**

:   本文识别了 LLM 代码修复中的"过度编辑"问题——模型倾向于重写大量代码而非精确定位和修复 bug，提出 PRepair 框架，通过 Self-Breaking（多样化 bug 注入）和 Self-Repairing（编辑感知 GRPO 训练），显著提升修复精确度同时保持正确性，并加速推测解码推理。

**[R$^3$-SQL: Ranking Reward and Resampling for Text-to-SQL](code_intelligence/r3-sql_ranking_reward_and_resampling_for_text-to-sql.md)**

:   R3-SQL 面向 generate-then-rank Text-to-SQL，先按执行结果把等价 SQL 分组并结合 pairwise/listwise 与 pointwise reward 排序，再用 LLM agent 判断候选池是否缺少正确 SQL 并选择性重采样，在 BIRD-dev 上达到 75.03 EX。

**[ReCode: Reinforcing Code Generation with Reasoning-Process Rewards](code_intelligence/recode_reinforcing_code_generation_with_reasoning-process_rewards.md)**

:   ReCode 通过 CRPL 训练能评价代码推理过程质量的奖励模型，并用 CG-GRPO 只在代码执行正确时激活过程奖励，从而在避免 reward hacking 的同时提升代码生成模型的 Pass@1。

**[ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization](code_intelligence/referee_reference-free_and_fine-grained_method_for_evaluating_factual_consistenc.md)**

:   本文提出 ReFEree，一种针对真实世界代码摘要的无参考、细粒度事实一致性评估方法，定义四类不一致标准并在句段级别评估，结合依赖信息搜索机制，在 Python 和 Java 上相比前 SOTA 提升 15-18% 的人类判断相关性。

**[RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion](code_intelligence/reposhapley_shapley-enhanced_context_filtering_for_repository-level_code_complet.md)**

:   提出 RepoShapley，一种基于 Shapley 值的联盟感知上下文过滤框架，通过估计检索代码片段在组合中的交互贡献来决定保留/丢弃，显著提升仓库级代码补全质量。

**[RExBench: Can coding agents autonomously implement AI research extensions?](code_intelligence/rexbench_can_coding_agents_autonomously_implement_ai_research_extensions.md)**

:   RExBench 把 coding agent 放进真实 AI 论文代码库中，让它们实现专家设计的研究扩展并用受控执行结果评分，发现当前最强 agent 也只有约三分之一成功率，距离自主完成科研扩展仍有明显差距。

**[River-LLM: Large Language Model Seamless Exit Based on KV Share](code_intelligence/river-llm_large_language_model_seamless_exit_based_on_kv_share.md)**

:   本文提出 River-LLM，一个无需训练的框架，通过构建轻量级 KV 共享退出通道（Exit River）解决了 decoder-only 架构中 Early Exit 的 KV Cache 缺失问题，利用状态转换相似度引导退出决策，实现 1.71×-2.16× 的实际推理加速且保持近无损生成质量。

**[Ro-SLM: Onboard Small Language Models for Robot Task Planning and Operation Code Generation](code_intelligence/ro-slm_onboard_small_language_models_for_robot_task_planning_and_operation_code_.md)**

:   Ro-SLM 用 LLM 合成和校验机器人任务-代码数据，再通过 SFT 与 LLM 奖励的 GRPO 优化 Llama-3.1-8B，使小模型能在无人机和地面车任务中接近云端 LLM 的规划与操作代码生成能力。

**[ROSE: An Intent-Centered Evaluation Metric for NL2SQL](code_intelligence/rose_an_intent-centered_evaluation_metric_for_nl2sql.md)**

:   ROSE 将 NL2SQL 评测从“预测 SQL 是否匹配单一参考 SQL”改为“预测 SQL 是否满足用户意图”，通过 SQL Prover 与 Adversarial Refuter 两阶段推理，在 ROSE-VEC 上比现有最佳指标高近 24 个百分点 Cohen's Kappa，并揭示 BIRD 等基准中参考 SQL 错误和问题歧义造成的评测危机。

**[SciCoQA: Quality Assurance for Scientific Paper–Code Alignment](code_intelligence/scicoqa_quality_assurance_for_scientific_paper--code_alignment.md)**

:   本文提出 SciCoQA，首个用于检测科学论文与其代码实现之间差异的基准数据集，包含 635 个差异实例（92 个真实 + 543 个合成），评测 22 个 LLM 后发现最强模型仅能检测 46.7% 的真实差异，揭示了自动化科学质量保证中的关键能力缺口。

**[SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios](code_intelligence/securevibebench_evaluating_secure_coding_capabilities_of_code_agents_with_realis.md)**

:   提出 SecureVibeBench，首个仓库级多文件编辑的安全编码基准，从41个OSS-Fuzz项目中构建105个C/C++安全编码任务，通过级联静态+动态分析精确还原漏洞首次引入的场景，评估发现最佳Agent（SWE-agent + Claude Sonnet 4.5）仅23.8%的代码同时满足功能正确性和安全性。

**[Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Understanding](code_intelligence/sense_and_sensitivity_examining_the_influence_of_semantic_recall_on_long_context.md)**

:   本文提出区分词汇召回（逐字检索代码）和语义召回（理解代码运行语义）两种能力，发现前沿 LLM 在长上下文中词汇召回近乎完美但语义召回严重退化，并引入 SemTrace 基准揭示现有评估严重低估了语义理解失败的程度。

**[SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization](code_intelligence/socia-evo_automated_simulator_construction_via_dual-anchored_bi-level_optimizati.md)**

:   本文提出 SOCIA-EVO，一种将自动化模拟器构建重新定义为双锚进化过程的 LLM 智能体框架，通过静态蓝图（Blueprint）锚定经验约束、双层优化解耦结构修正与参数校准、自我策划的策略剧本（Playbook）管理修复假说并通过执行反馈进行贝叶斯加权检索，在用户建模、口罩佩戴扩散和个人出行三个模拟任务上显著超越 Reflexion、G-SIM 等基线。

**[SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](code_intelligence/solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)**

:   SolidCoder 通过 S.O.L.I.D. 架构（Shift-left Planning、Oracle-based Assertions、Live Execution、Intermediate Simulation、Defensive Accumulation）将代码验证从 LLM 的"想象执行"转变为"真实执行"，在 GPT-4o 上达到 HumanEval 95.7%、CodeContests 77.0%、APPS 26.7% 的 pass@1 性能。

**[Static Program Slicing Using Language Models With Dataflow-Aware Pretraining and Constrained Decoding](code_intelligence/static_program_slicing_using_language_models_with_dataflow-aware_pretraining_and.md)**

:   Sliceformer 把静态程序切片重写为小型代码语言模型的 seq2seq 任务，通过数据流感知预训练学习依赖关系，并用词法与语法约束解码防止幻觉，在 Java 和 Python 切片基准上显著提升 ExactMatch。

**[StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](code_intelligence/storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)**

:   本文提出 StoryCoder，一种将代码生成问题重构为连贯自然语言叙事的提示框架，通过任务概述、约束条件和示例三个叙事组件引导 LLM 进行结构化推理，在 11 个模型上平均提升零样本 pass@10 达 18.7%。

**[SWE-QA: Can Language Models Answer Repository-level Code Questions?](code_intelligence/swe-qa_can_language_models_answer_repository-level_code_questions.md)**

:   SWE-QA 构建了一个覆盖 15 个真实 Python 仓库、720 个高质量问答对的仓库级代码问答基准，用 GitHub issue 归纳问题类型并用人工校验答案，实验显示单纯 LLM 直接回答很弱，RAG 与 OpenHands/SWE-agent 这类工具化 agent 才能接近真实开发问答需求。

**[Taming System Complexity: Demystifying Software Engineering Agents in Diagnosing Linux Kernel Faults](code_intelligence/taming_system_complexity_demystifying_software_engineering_agents_in_diagnosing_.md)**

:   通过建立大规模 Linux 内核故障定位基准 LinuxFLBench，揭示现有 LLM Agent 在复杂系统中的局限，并提出 LinuxFL+ 框架通过目录感知和潜在原因双维扩展，以低成本显著提升故障定位精度。

**[The Path Not Taken: Duality in Reasoning about Program Execution](code_intelligence/the_path_not_taken_duality_in_reasoning_about_program_execution.md)**

:   本文提出程序执行推理的对偶性概念，通过DexBench基准（445个配对实例）联合评估LLM的正向执行推理（预测给定输入下的代码覆盖）和反向反事实推理（推断使执行流转向目标分支的输入变异），发现单一方向上的强表现不能转化为联合评估下的成功，揭示了模型对程序因果理解的不足。

**[To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing](code_intelligence/to_diff_or_not_to_diff_structure-aware_and_adaptive_output_formats_for_efficient.md)**

:   这篇论文把 LLM 代码编辑的“输出格式”本身作为训练对象，提出 BlockDiff、FuncDiff 和自适应格式选择策略 AdaEdit，在接近 full-code 生成准确率的同时，在长代码编辑中将延迟和输出 token 成本降低 30% 以上。

---

## 🩺 医疗 NLP { #medical_nlp }

**["Excuse Me, May I Say Something…" CoLabScience: A Proactive AI Assistant for Biomedical Discovery](medical_nlp/34excuse_me_may_i_say_something34_colabscience_a_proactive_ai_assistant_for_biom.md)**

:   CoLabScience 通过 PULI（正无标注学习干预）框架，训练一个能在生物医学团队讨论中**主动判断何时介入、如何介入**的 LLM 助手，利用 GRPO 和强化学习协调器从流式对话中自动识别最佳干预时机并生成科学建议。

**[Anonpsy: A Graph-Based Framework for Structure-Preserving De-identification of Psychiatric Narratives](medical_nlp/anonpsy_a_graph-based_framework_for_structure-preserving_de-identification_of_ps.md)**

:   提出Anonpsy框架，将精神科叙事的去标识化重新定义为图引导的语义重写问题——先将叙事转换为语义图，在图上进行受约束的扰动以修改身份信息同时保持临床结构，最后通过图条件生成重建叙事。

**[Beyond Prompt: Fine-grained Simulation of Cognitively Impaired Standardized Patients via Stochastic Steering](medical_nlp/beyond_prompt_fine-grained_simulation_of_cognitively_impaired_standardized_patie.md)**

:   提出 StsPatient，通过从对比指令/回复对中提取领域特定的转向向量（Steering Vector），配合随机 Token 调制（STM）机制控制注入概率来模拟不同认知障碍领域和严重程度的标准化病人，相比 prompt engineering 方法在临床真实性上平均提升 11.23%，在严重程度可控性上超越最佳基线 18.54%。

**[Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents](medical_nlp/beyond_the_individual_virtualizing_multi-disciplinary_reasoning_for_clinical_int.md)**

:   提出 Aegle 框架，通过图结构多智能体架构虚拟化多学科会诊（MDT），将解耦并行推理和动态拓扑引入门诊问诊流程，在24个科室53项指标上超越SOTA模型。

**[Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](medical_nlp/beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)**

:   作者提出 MedCheck——首个面向医学 LLM benchmark **生命周期**的评估框架，把 benchmark 构建拆成 5 个阶段共 46 条标准，用它对 56 个医学 benchmark 做审计，发现医学 NLP 评测领域存在 3 个系统性问题：(1) 50% 不对齐任何医学标准（ICD/SNOMED），(2) 88% 不处理数据污染，(3) 89% 不测模型 robustness、91% 不测 uncertainty——结论是当前"排行榜进步"很大程度是幻象。

**[BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](medical_nlp/biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)**

:   BioHiCL 利用 MeSH（医学主题词）的**层级多标签标注**为稠密检索器提供结构化监督，通过深度加权的标签相似度对齐嵌入空间与 MeSH 语义空间，使 0.1B 模型在生物医学检索、句子相似度和问答任务上超越大多数专用模型。

**[Calibrated? Not for Everyone: How Sexual Orientation and Religious Markers Distort LLM Accuracy and Confidence in Medical QA](medical_nlp/calibrated_not_for_everyone_how_sexual_orientation_and_religious_markers_distort.md)**

:   研究社会身份标记（性取向和宗教信仰）如何扭曲LLM在医疗问答中的准确率和置信度校准，发现"同性恋"标记在9个LLM上一致导致性能下降和校准危机，且交叉身份产生非加性的特异性伤害。

**[Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?](medical_nlp/can_continual_pre-training_bridge_the_performance_gap_between_general-purpose_an.md)**

:   本文通过构建高质量德语医学语料库 FineMed-de（从 FineWeb2 过滤 730 万文档/51 亿词），对三种 LLM（7B-24B）进行持续预训练和 SLERP 模型合并，创建 DeFineMed 模型家族，证明领域特化的 7B 模型可以在德语医学任务上显著缩小与 24B 通用模型的性能差距（胜率提升约 3.5 倍）。

**[CT-FineBench: A Diagnostic Fidelity Benchmark for Fine-Grained Evaluation of CT Report Generation](medical_nlp/ct-finebench_a_diagnostic_fidelity_benchmark_for_fine-grained_evaluation_of_ct_r.md)**

:   作者把"CT 报告好不好"这个模糊问题，拆成"每个 finding 的每条细粒度属性是否对得上"的 QA 检查清单，构建出 44k 题的 CT-FineBench 基准，对临床错误的敏感度和与人类专家打分的相关性都显著超过 BLEU/BERTScore/RadGraph/RaTEScore/GREEN 等已有指标。

**[CT-Flow: Orchestrating CT Interpretation Workflow with Model Context Protocol Servers](medical_nlp/ct-flow_orchestrating_ct_interpretation_workflow_with_model_context_protocol_ser.md)**

:   作者把 3D CT 解读重新建模成"放射科医生用工具迭代探查"的智能体任务，用 Model Context Protocol（MCP）暴露 Data Ingestion / Global Navigation / Detailed Observation / Advanced Analysis 四类工具，构造 2000+300 条可执行轨迹的 CT-FlowBench，并 SFT 出 CT-Flow-8B：在 3D-RAD 上达到 69.46% ACC，比纯切片基线提升 +22.46%，工具调用名称错误率仅 0.007/case。

**[CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](medical_nlp/cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)**

:   CURA 提出一个双层不确定性校准框架：个体层面将预测不确定性与错误概率对齐，队列层面通过嵌入空间的邻域风险率正则化预测，在 MIMIC-IV 的五个临床风险预测任务上一致提升校准指标而不牺牲判别性能。

**[CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning](medical_nlp/cure-med_curriculum-informed_reinforcement_learning_for_multilingual_medical_rea.md)**

:   作者构造了一个覆盖 13 种语言（含 Amharic、Yoruba、Swahili 等低资源语言）、共 15,774 条开放问答的医学推理数据集 CureMed-Bench，并提出 Cure-Med：一个两阶段「code-switching 感知的 SFT + 课程式 GRPO」框架，把推理正确性和语言一致性联合优化，在 7B 上将语言一致率 / 推理正确率拉到 85.21% / 54.35%，在 32B 上拉到 94.96% / 70.04%。

**[Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](medical_nlp/dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)**

:   本文提出临床诊断推理数据（CDRD）结构来捕获从症状到鉴别诊断的抽象临床推理逻辑，并基于 CDRD 通过 SFT+RL 两阶段训练构建 Dr. Assistant 模型（14B），在临床问诊基准上 ICD-Recall 超过 HuatuoGPT-o1-72B 13.59%，达到与 GPT-5 竞争的水平。

**[Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](medical_nlp/efficient_and_effective_internal_memory_retrieval_for_llm-based_healthcare_predi.md)**

:   本文提出K2K框架，将LLM的FFN参数空间视为可检索的知识库，通过LoRA注入临床知识、激活引导的探针构建精确检索、交叉注意力重排序自适应整合，实现了无需外部检索延迟的医疗预测SOTA。

**[Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised Reinforcement Learning Approach](medical_nlp/eliciting_medical_reasoning_with_knowledge-enhanced_data_synthesis_a_semi-superv.md)**

:   本文提出MedSSR框架，通过注入罕见病知识的可控数据合成和"自监督RL→监督RL"的半监督训练范式，高效提升LLM的医学推理能力，在罕见病任务上实现最高+5.93%的提升，突破了现有方法+3%的改进上限。

**[Empathy Applicability Modeling for General Health Queries](medical_nlp/empathy_applicability_modeling_for_general_health_queries.md)**

:   本文提出 Empathy Applicability Framework (EAF)，先判断患者单轮健康咨询中是否“适合”表达情绪回应或解释性理解，再用人类与 GPT-4o 标注构建基准并训练分类器，为医疗 LLM 回答前的同理心需求识别提供上游信号。

**[Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence](medical_nlp/faithfulness_vs_safety_evaluating_llm_behavior_under_counterfactual_medical_evid.md)**

:   本文构建 MedCounterFact 数据集——用无义词、医学术语、非医学物品和有毒物质系统替换临床试验中的干预措施——发现前沿 LLM 在反事实医疗证据面前几乎无条件遵从上下文，即便"证据"表明海洛因或芥子气有疗效也自信回答，揭示了忠实度与安全之间缺乏明确边界的严重问题。

**[Forgotten Words: Benchmarking NeoBERT for Dementia Detection in Low-Resource Conversational Filipino and English Speech](medical_nlp/forgotten_words_benchmarking_neobert_for_dementia_detection_in_low-resource_conv.md)**

:   这篇论文用 4000 条英菲平行 DementiaBank 对话转写系统评测 TF-IDF、BERT、NeoBERT、XLM-R 和 RoBERTa-Tagalog，发现痴呆检测的跨语言鲁棒性主要来自训练阶段的语言覆盖，而不是更现代的编码器架构。

**[From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning](medical_nlp/from_answers_to_arguments_toward_trustworthy_clinical_diagnostic_reasoning_with_.md)**

:   本文将Toulmin论证模型适配到临床诊断过程，提出CGCL三阶段课程训练框架（事实收集→假设检验→综合结论），配合T-Eval量化评估推理结构完整性，在无需RL的情况下实现与RL方法可比的诊断推理质量。

**[HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for Medical Vision Language Tasks](medical_nlp/heterorag_a_heterogeneous_retrieval-augmented_generation_framework_for_medical_v.md)**

:   HeteroRAG 构建了 270 万张图文对 + 五类语料的 MedAtlas 知识库，把医学多模态 RAG 拆成三件套——按模态分训的 ModCLIPs 拉报告、按语料定制 query 的 MQG 拉文档、用 HKPT 偏好微调对齐跨模态与多源知识——让 7B 模型在 11 个数据集上稳定击败 4-5× 参数量的开源 Med-LVLM。

**[HypEHR: Hyperbolic Modeling of Electronic Health Records for Efficient Question Answering](medical_nlp/hypehr_hyperbolic_modeling_of_electronic_health_records_for_efficient_question_a.md)**

:   本文提出 HypEHR，一个仅 22M 参数的洛伦兹双曲模型，将医学编码、就诊记录和问题嵌入双曲空间，通过层级感知正则化对齐 ICD 本体结构，在 MIMIC-IV 电子病历问答任务上接近 LLM 方法的效果。

**[IndicMedDialog: A Parallel Multi-Turn Medical Dialogue Dataset for Accessible Healthcare in Indic Languages](medical_nlp/indicmeddialog_a_parallel_multi-turn_medical_dialogue_dataset_for_accessible_hea.md)**

:   本文构造 IndicMedDialog——首个覆盖英文+9 种印度语系（Assamese / Bengali / Gujarati / Hindi / Marathi / Punjabi / Tamil / Telugu / Urdu）的**平行多轮**医学诊断对话数据集（2,980 段对话 × 10 语 = 29,800 个实例），用 LLaMA-3.3-70B 合成对话 + TranslateGemma 翻译 + native speaker 校对 + 脚本感知 post-processing 修音/拼/字符间距；并基于 4-bit 量化 LLaMA-3.2-3B + LoRA 训出 IndicMedLM，在英文/印地语/马拉地语等 7/10 种语言上拿到 post-processed accuracy 最高，同时 95.3% 医学安全通过率，揭示了 5 类系统性 failure mode（ID/LC/CDC/TTF/PLG）。

**[Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation](medical_nlp/inflated_excellence_or_true_performance_rethinking_medical_diagnostic_benchmarks.md)**

:   本文提出 DyReMe 动态医学诊断评估框架，通过 DyGen 模块生成包含鉴别诊断和误诊因素等临床干扰项的全新诊断案例，并通过 EvalMed 模块从准确性、真实性、帮助性和一致性四个维度评估 LLM，揭示现有静态基准高估了 LLM 的诊断能力——GPT-5 在 DyReMe 上准确率下降 8.25%，12 个 LLM 均暴露出显著的可信度不足。

**[Language Reconstruction with Brain Predictive Coding from fMRI Data](medical_nlp/language_reconstruction_with_brain_predictive_coding_from_fmri_data.md)**

:   本文提出 PredFT，一个结合主网络（语言解码）和侧网络（脑预测编码表征）的端到端 fMRI-to-Text 解码模型，通过从大脑预测相关脑区（PTO 区域）提取前瞻性语义表征并融合到解码过程中，在 LeBel 数据集上 BLEU-1 达 34.95%（Sub-1），相比最强基线 MapGuide 提升 7.84 个百分点。

**[Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness](medical_nlp/learning_dynamic_representations_and_policies_from_multimodal_clinical_time-seri.md)**

:   提出 OPL-MT-MNAR 框架，通过 MNAR 感知的多模态编码器 + 贝叶斯滤波隐状态 + 离线策略学习，从结构化数据和临床文本的"缺失模式本身携带的信息"中学习 ICU 患者动态表示，实现优于临床医生行为的脓毒症治疗策略（FQE 0.679 vs 0.528）。

**[LinguIUTics at PsyDefDetect: Iterative Imbalance-Aware Fine-tuning of Qwen3-8B for Psychological Defense Mechanism Classification](medical_nlp/linguiutics_at_psydefdetect_iterative_imbalance-aware_fine-tuning_of_qwen3-8b_fo.md)**

:   这篇 PsyDefDetect 参赛系统通过 Qwen3-8B QLoRA、少数类词法增强、分组 5-fold 交叉验证、OOF logit bias 和多种子融合，把心理防御机制 9 分类的官方 macro F1 提升到 0.3917，排名 21 支队伍中的第 4。

**[MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](medical_nlp/march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)**

:   本文提出 MARCH，一个模拟放射科住院医-专科医-主治医层级协作流程的多智能体框架，通过三阶段（初始报告起草、检索增强修订、共识驱动定稿）生成 CT 报告，在 RadGenome-ChestCT 数据集上 CE-F1 达 0.399，比最佳基线 Reg2RG 的 0.253 提升 57.7%。

**[Measuring What Matters!! Assessing Therapeutic Principles in Mental-Health Conversation](medical_nlp/measuring_what_matters_assessing_therapeutic_principles_in_mental-health_convers.md)**

:   本文提出 CARE 框架和 FAITH-M 基准数据集，通过对话上下文编码与对比范例检索+知识蒸馏链式推理（KD-CoT），对 AI 生成的心理治疗对话进行六个治疗原则维度的细粒度序数评估，加权 F1 达 63.34，比最强基线 Qwen3 提升 64.26%。

**[MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts](medical_nlp/medfact_benchmarking_the_fact-checking_capabilities_of_large_language_models_on_.md)**

:   MedFact 构建了一个覆盖真实中文医疗文本的专家标注事实核查 benchmark，并用 20 个 LLM 证明：当前模型较容易判断文本“有没有错”，但仍难以精确定位错误，RAG 有帮助，而多智能体和推理时扩展反而容易放大“过度批判”。

**[MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](medical_nlp/mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)**

:   MHGraphBench 从 PrimeKG 的心理健康子图自动构造 9 类多选任务，发现 LLM 在实体识别上接近满分，但在药物-疾病关系判断、禁忌边界和两跳 KG 推理上仍明显不足。

**[MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](medical_nlp/mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)**

:   本文提出 R-MHSafe 角色感知心理健康安全分类体系和 MHSafeEval 闭环 agent 评估框架，通过对抗性多轮咨询交互系统性发现 LLM 在心理咨询场景中的角色依赖型累积安全失败，揭示了现有静态基准无法捕捉的交互层面危害。

**[Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](medical_nlp/multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)**

:   本文提出将话语分解为情感-逻辑-行为（ELB）三组件并用 LLM 推理多个认知扭曲实例，然后通过多视角门控注意力 MIL 框架进行 bag 级分类，在韩语（KoACD）和英语（Therapist QA）数据集上均优于 LLM 直接推理基线。

**[MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning](medical_nlp/multidx_a_multi-source_knowledge_integration_framework_towards_diagnostic_reason.md)**

:   MultiDx 将网页检索、SOAP 结构化病例、相似病例库和细粒度推理片段检索合成一个两阶段诊断推理框架，先从多路证据生成候选疾病，再通过疾病匹配、投票和鉴别诊断重排，最终在 MedCaseReasoning 与 DiReCT 上同时提升诊断命中率和推理召回。

**[PCoA: A New Benchmark for Medical Aspect-Based Summarization With Phrase-Level Context Attribution](medical_nlp/pcoa_a_new_benchmark_for_medical_aspect-based_summarization_with_phrase-level_co.md)**

:   PCoA 构建了一个面向随机对照试验摘要的医学方面级摘要基准，把每条方面摘要同时对齐到支撑句子和贡献短语，并用 claim、citation、phrase 三层指标系统评测 LLM 在可验证医学摘要上的能力。

**[PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment](medical_nlp/principlismqa_a_philosophy-grounded_approach_to_assessing_llm-human_clinical_med.md)**

:   本文基于国际医学伦理黄金标准——Principlism（自主、不伤害、有益、公正四原则），构建了 PrinciplismQA 基准（3,648 题，含知识 MCQA 和开放式临床伦理困境），并配套专家校准的评估流水线，发现 LLM 在知识基准上的高准确率并不等于具备临床伦理推理能力——最强模型 o3 总分也仅 77.5%。

**[ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection](medical_nlp/promedical_hierarchical_fine-grained_criteria_modeling_for_medical_llm_alignment.md)**

:   ProMedical 用医生参与构造的分层细粒度 clinical rubric 贯穿偏好数据、奖励模型和 benchmark，通过显式 criteria injection 训练多维 reward model，使 Qwen3-8B 在医学对齐中获得 22.3% overall accuracy 和 21.7% safety compliance 的提升。

**[Query Pipeline Optimization for Cancer Patient Question Answering Systems](medical_nlp/query_pipeline_optimization_for_cancer_patient_question_answering_systems.md)**

:   本文提出 CoMeta，一个面向癌症患者问答（CPQA）的三层可控元数据感知 RAG 框架，通过临床混合语义-符号文档检索（CHSDR）融合 E-Utilities 实时布尔搜索与 MedCPT 语义检索，配合语义增强重叠分割（SEOS）防止上下文碎片化，在 CMMQA 数据集上将 Claude-3-Haiku 的回答准确率提升 5.24%（vs CoT）和约 3%（vs naive RAG）。

**[RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction](medical_nlp/ra-rrg_multimodal_retrieval-augmented_radiology_report_generation_with_key_phras.md)**

:   提出 RA-RRG 框架，通过 LLM 从放射报告中提取临床关键短语并构建检索库，给定胸部 X 光影像后检索相关短语并输入 LLM 生成报告，无需 LLM 微调即可有效抑制幻觉，仅需 18 GPU 小时训练，在 CheXbert 指标上达到 SOTA。

**[RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](medical_nlp/rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)**

:   本文提出 RADS（Reinforcement Adaptive Domain Sampling），一种基于强化学习的样本选择策略，在极端低资源和类别不平衡的临床场景下，通过智能选择少量目标域样本进行标注和联合微调，显著提升跨域疾病检测的迁移效果。

**[Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework](medical_nlp/region-grounded_report_generation_for_3d_medical_imaging_a_fine-grained_dataset_.md)**

:   本文提出首个带有细粒度 ROI 标注的 3D PET/CT 数据集 VietPET-RoI（越南语），以及模拟放射科医生诊断流程的层次化报告生成框架 HiRRA，通过图神经网络建模 ROI 间的空间-形态学关系，BLEU-4 提升 19.7%，临床指标 RoIQ 提升 45.8%。

**[Reliable Automated Triage in Spanish Clinical Notes: A Hybrid Framework for Risk-Aware HIV Suspicion Identification](medical_nlp/reliable_automated_triage_in_spanish_clinical_notes_a_hybrid_framework_for_risk-.md)**

:   本文面向西班牙临床笔记中的早期 HIV 疑似识别，提出 MCP 处理 aleatoric uncertainty、MCMD 几何 veto 处理 epistemic uncertainty 的双验证选择性分诊框架，在严格安全约束下自动处理 67.7% 病例并达到 0.982 Clear $F_2$。

**[ReMedi: Reasoner for Medical Clinical Prediction](medical_nlp/remedi_reasoner_for_medical_clinical_prediction.md)**

:   ReMedi 将 EHR 临床预测改写成“理由-答案”生成与偏好学习问题，通过带真实结局提示的困难样本再生成、SFT 和 DPO 让医学 LLM 学会更细粒度地解释患者风险，在 MIMIC-IV 三类预测任务上相对 KARE 最高带来 19.9 个 F1 点提升。

**[RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models](medical_nlp/reprompt_recurrent_prompt_tuning_for_integrating_structured_ehr_encoders_with_la.md)**

:   本文提出 RePrompT，一种时间感知的 LLM 框架，通过循环提示调优（将前一次就诊的隐状态作为下一次就诊的软提示）和结构化编码提示调优（注入群体级 EHR 编码器的嵌入）两种互补机制，在 MIMIC-III/IV 上的再入院和死亡率预测任务上一致超越 EHR 基线和 LLM 基线。

**[Responsible Evaluation of AI for Mental Health](medical_nlp/responsible_evaluation_of_ai_for_mental_health.md)**

:   本文通过系统分析 135 篇 ACL Anthology 论文，揭示 AI 心理健康工具评估中的五大缺陷（依赖通用指标、缺乏人类评估、忽视安全与公平等），并提出融合临床心理测量学与实施科学的跨学科评估分类体系（assessment/intervention/information synthesis × validity/reliability/implementation/maintenance）。

**[Ryze: Evidence-Enriched Data Synthesis from Biomedical Papers](medical_nlp/ryze_evidence-enriched_data_synthesis_from_biomedical_papers.md)**

:   Ryze 将生物医学论文 PDF 自动转成保留图表、caption、结构化抽取和引用段落的证据增强 QA 数据，并用进度门控的 SFT+GRPO 训练 BioVLM-8B，在 LAB-Bench 上以 48.0% weighted accuracy 超过 Qwen3-VL-8B base 12.6 个百分点、超过 GPT-5.2 3.8 个百分点。

**[SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning](medical_nlp/sema-rag_a_self-evolving_multi-agent_retrieval-augmented_generation_framework_fo.md)**

:   提出 SEMA-RAG，一种自演化多智能体 RAG 框架，通过三个专职智能体（解释器、探索器、仲裁器）模拟临床推理的分阶段工作流，在 5 个医学 QA 基准上平均超越最强基线 +6.46 个准确率点。

**[Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](medical_nlp/text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)**

:   本文提出 CoMed，一种 LLM 赋能的图学习框架，通过结合 EHR 统计证据和类型约束 LLM 推理构建全局医学知识图谱，再用 LLM 生成节点描述和边理由丰富为文本属性图，最终联合训练 LoRA 微调的 LLaMA 编码器和异构 GNN 学习统一的医学概念嵌入，在 MIMIC-III/IV 上显著提升诊断预测性能。

---

## 🎮 强化学习 { #reinforcement_learning }

**[A Goal Without a Plan Is Just a Wish: Efficient and Effective Global Planner Training for Long-Horizon Agent Tasks (EAGLET)](reinforcement_learning/a_goal_without_a_plan_is_just_a_wish_efficient_and_effective_global_planner_trai.md)**

:   EAGLET 把长程 agent 任务拆成「全局 planner + 局部 executor」两个模块，通过「同源共识过滤合成 SFT 冷启动 + 用执行器能力增益作奖励的 GRPO 微调」两步训练出一个即插即用的 planner，三个长程任务上刷新 SOTA 且训练成本仅为 RL baseline 的 1/8。

**[A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions](reinforcement_learning/a_survey_of_reinforcement_learning_for_large_language_models_under_data_scarcity.md)**

:   首篇系统综述数据稀缺条件下LLM强化学习的工作，提出数据中心、训练中心、框架中心三层分类体系，覆盖数据剪枝/合成/压缩、轨迹生成/奖励工程/策略优化、以及自演化/协同演化/多智能体演化等方向。

**[Adaptive Instruction Composition for Automated LLM Red-Teaming](reinforcement_learning/adaptive_instruction_composition_for_automated_llm_red-teaming.md)**

:   提出 Adaptive Instruction Composition (AIC) 框架，利用 Neural Thompson Sampling 在众包有害查询和越狱策略的组合空间中自适应地选择攻击指令，同时优化攻击成功率和多样性，在 Harmbench 上大幅超越已有方法。

**[ARGUS: Policy-Adaptive Ad Governance via Evolving Reinforcement with Adversarial Umpiring](reinforcement_learning/argus_policy-adaptive_ad_governance_via_evolving_reinforcement_with_adversarial_.md)**

:   ARGUS 用 Prosecutor–Defender–Umpire 三智能体辩论 + GRPO 强化学习，让广告审核 VLM 在政策不断更新时既能纠正历史「过时标签」、又能挖出灰区潜在违规，工业 A/B 把违规漏放率（VLR）相对降低 35.2%。

**[AttnPO: Attention-Guided Process Supervision for Efficient Reasoning](reinforcement_learning/attnpo_attention-guided_process_supervision_for_efficient_reasoning.md)**

:   提出 AttnPO，一个利用模型内在注意力信号进行步级信用分配的低开销过程监督 RL 框架，通过识别 Key-Focus Heads（KFH）区分冗余和关键推理步骤，在大幅缩短推理长度的同时显著提升准确率。

**[Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning](reinforcement_learning/beyond_majority_voting_towards_fine-grained_and_more_reliable_reward_signal_for_.md)**

:   针对 TTRL 用 majority voting 做伪标签带来的「确认偏差 + 稀疏奖励」两大痛点，SCOPE 提出步级置信度加权投票（不再唯频次是从）+ Pareto-optimal subgroup 动态划分（每子组独立 bootstrap 出局部共识），在 Qwen3-8B 上把 AIME 2024 从 47.13 → 52.70、AIME 2025 从 27.40 → 31.00。

**[Breaking the Impasse: Dual-Scale Evolutionary Policy Training for Social Language Agents](reinforcement_learning/breaking_the_impasse_dual-scale_evolutionary_policy_training_for_social_language.md)**

:   针对自博弈 RLVR 在开放式社交语言博弈（谈判 / 不许说 / 两美元分配）中出现的"进化僵局"——agent 行为同质化导致比赛结果分布退化为确定性、梯度信号消失——本文提出 DEPT 用 fast/slow 双时间尺度 EMA baseline 检测 stagnation，再用 asymmetric advantage reshaping 抑制 dominant outcome、放大 rare trajectory，在 Qwen3-4B/8B-Base 上把谈判任务胜率从 16-20% 拉到 32%，并在 OOD 数学/推理 benchmark 上同步受益。

**[Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning](reinforcement_learning/bridging_sft_and_rl_dynamic_policy_optimization_for_robust_reasoning.md)**

:   提出 DYPO（Dynamic Policy Optimization），通过动态难度分级将样本路由到不同优化路径——Hard样本用多教师蒸馏降低SFT偏差、Mid样本用Group Alignment Loss降低RL方差，在数学推理benchmark上平均提升4.8%，OOD任务提升13.3%。

**[CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning](reinforcement_learning/ce-gppo_coordinating_entropy_via_gradient-preserving_clipping_policy_optimizatio.md)**

:   提出 CE-GPPO 算法，通过 stop-gradient 操作重新引入 PPO 裁剪区间外低概率 token 的梯度信号，实现对策略熵的精细化协调控制，在探索-利用之间取得更好平衡。

**[Community-Aware Assessment of Social Textual Engagement and Resonance: A Human-Centric Perspective on User-Generated Content Evaluation](reinforcement_learning/community-aware_assessment_of_social_textual_engagement_and_resonance_a_human-ce.md)**

:   这篇论文提出 CASTER 任务与 CASTER-Bench，并用 MEDEA 通过 Social-CoT、SFT 和带 Social Alignment Reward 的过程监督强化学习来模拟社区反应，在 CASTER-Bench 上把 High-Quality F1 提升到 0.650、Macro-F1 提升到 0.749，显著优于传统 VQA 和通用 LMM 基线。

**[Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions](reinforcement_learning/controlling_multimodal_conversational_agents_with_coverage-enhanced_latent_actio.md)**

:   提出为多模态对话智能体（MCA）构建紧凑的潜在动作空间来替代巨大的 token 动作空间进行 RL 微调，通过跨模态投影器和循环一致性损失利用配对图文数据和纯文本数据共同构建码本，将动作空间从 152K（词表大小）压缩到 128（码本大小），在两个对话任务上全面超越 token 级 RL 基线。

**[d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](reinforcement_learning/d-treerpo_towards_more_reliable_policy_optimization_for_diffusion_language_model.md)**

:   针对扩散语言模型（dLLM）RL 的两大可靠性瓶颈 —— 奖励稀疏 / 概率估计偏差 —— 作者提出 d-TreeRPO：把 rollout 组织成树，用叶子节点的可验证奖励自下而上算出步级 advantage；同时理论证明「模型越自信，单步前向概率估计越准」，并设计时间调度的自蒸馏 loss 在训练后期收紧策略，最终在 LLaDA-8B-Instruct 上 Sudoku +86.2% / Countdown +51.6% / GSM8K +4.5% / Math500 +5.3%。

**[Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with Constraints](reinforcement_learning/deliberative_searcher_improving_llm_reliability_via_reinforcement_learning_with_.md)**

:   本文提出 Deliberative Searcher，一个推理优先的框架，将搜索操作集成到 CoT 生成中并保持显式置信度校准，使用自适应拉格朗日乘子的约束 RL 联合优化正确性和可靠性，将 7B 模型的平均"错误-确定"率从基线的 54% 降至 2%。

**[DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents](reinforcement_learning/dpepo_diverse_parallel_exploration_policy_optimization_for_llm-based_agents.md)**

:   作者提出"并行探索"新范式——agent 同步与 $K$ 个 environment 交互、跨轨迹共享经验，并给出对应的 RL 算法 **DPEPO**：先 SFT 冷启动学会并行 reasoning，再用"轨迹级成功 + 步级 Diverse Action / Diverse State Transition"分层奖励的 GRPO 训练，在 ALFWorld 与 ScienceWorld 全部 split 上拿下 SOTA（Qwen2.5-7B 上分别 98.2% / 61.4%），且在更大 $K$ 时 token 增长远低于"多采样"基线。

**[Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning](reinforcement_learning/easy_samples_are_all_you_need_self-evolving_llms_via_data-efficient_reinforcemen.md)**

:   提出 EasyRL 框架，受认知发展理论启发，仅用10%的简单标注数据通过知识迁移初始化模型，再通过分治伪标注和由难度递进的自训练逐步掌握困难未标注数据，一致性超越全量数据监督的 GRPO。

**[Efficient Hyperparameter Optimization for LLM Reinforcement Learning](reinforcement_learning/efficient_hyperparameter_optimization_for_llm_reinforcement_learning.md)**

:   本文提出 JF-HPO，把小型同族代理模型、训练步数保真度、训练动态早停和 checkpoint 复用合到一个 Bayesian HPO 框架中，用更低成本为 LLM 强化学习找到更稳的超参数，并在多个推理任务上优于 VeRL Recipe、Random Search 和 BOHB。

**[EvoCoT: Overcoming the Exploration Bottleneck in Reinforcement Learning for LLMs](reinforcement_learning/evocot_overcoming_the_exploration_bottleneck_in_reinforcement_learning.md)**

:   本文提出 EvoCoT，一个两阶段自我进化的课程学习框架：先用最终答案约束 LLM 自生成可验证的 CoT 轨迹，再渐进地从尾部删除推理步骤，逐步扩大探索空间，从而在 sparse reward 的硬题上稳定训练 RLVR，不依赖任何教师模型或人写 CoT，就让 R1-Qwen-1.5B 在 MATH 训练集上的硬题正确率从 55.7 飙升到 87.8。

**[Free Energy-Driven Reinforcement Learning with Adaptive Advantage Shaping for Unsupervised Reasoning in LLMs](reinforcement_learning/free_energy-driven_reinforcement_learning_with_adaptive_advantage_shaping_for_un.md)**

:   FREIA 把自由能原理 (FEP) 引入无标签 RL 微调，用「共识 + 探索」自适应奖励 (FER) 和基于奖励分布偏度的自适应优势整形 (AAS) 同时解决传统多数投票 / 自信度 reward 的过早收敛和 advantage 估计在训练阶段错配两个问题，在 3 个推理任务、9 个数据集上达到与 supervised GRPO 持平甚至更好的水平。

**[From Isolated Scoring to Collaborative Ranking: A Comparison-Native Framework for LLM-Based Paper Evaluation](reinforcement_learning/from_isolated_scoring_to_collaborative_ranking_a_comparison-native_framework_for.md)**

:   这篇论文把 LLM 论文评审从“单篇打绝对分”改成“成对比较再全局排序”，用语义图采样、比较式 SFT 与可验证奖励强化训练 7B 模型，在 ICLR-2025 论文排序和录用预测上显著超过 DeepReview-14B，并能迁移到多个未见会议。

**[GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR](reinforcement_learning/geora_geometry-aware_low-rank_adaptation_for_rlvr.md)**

:   本文提出 GeoRA，一种专为强化学习可验证奖励（RLVR）设计的低秩适配方法，通过构建几何约束矩阵（融合谱先验和欧几里得先验）提取 RL 更新子空间的主方向进行 SVD 初始化，同时冻结残差矩阵作为结构锚，在 1.5B-32B 参数的 Qwen/Llama 模型上，数学、医学和代码 RLVR 任务中一致超越 LoRA、PiSSA、MiLoRA 等基线，且具备更强的域外泛化和更少的能力遗忘。

**[Glance-or-Gaze: Incentivizing LMMs to Adaptively Focus Search via Reinforcement Learning](reinforcement_learning/glance-or-gaze_incentivizing_lmms_to_adaptively_focus_search_via_reinforcement_l.md)**

:   这篇论文提出 Glance-or-Gaze (GoG)，让多模态大模型在回答知识密集型视觉问题时学会先看全图、再选择高价值区域精查，并通过 SFT + 复杂度自适应 GRPO 在 6 个视觉问答/搜索基准上显著优于直接回答、全量搜索和 MMSearch-R1 等基线。

**[Good Reasoning Makes Good Demonstrations: Implicit Reasoning Quality Supervision via In-Context Reinforcement Learning](reinforcement_learning/good_reasoning_makes_good_demonstrations_implicit_reasoning_quality_supervision_.md)**

:   这篇论文指出 RLVR 不能区分“推理质量高的正确答案”和“碰巧答对的低质量推理”，并提出用示范的 in-context 教学效用 Evidence Gain 作为隐式质量信号，通过 In-Context RLVR 在不训练 PRM 的情况下提升数学推理准确率和推理质量。

**[HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment](reinforcement_learning/healing_entropy_collapse_enhancing_exploration_in_few-shot_rlvr_via_hybrid-domai.md)**

:   提出 HEAL 框架，通过混合通用领域数据和熵动态对齐（EDA）奖励机制解决少样本 RLVR 中的严重熵崩溃问题，仅用32个目标域样本即可匹配甚至超越使用1K样本的全量 RLVR 性能。

**[ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following](reinforcement_learning/imprif_stronger_implicit_reasoning_leads_to_better_complex_instruction_following.md)**

:   ImpRIF 将复杂指令中的隐式推理结构形式化为可验证的显式推理图（ERG），基于此构建大规模单轮/多轮数据并通过 SFT+过程验证 RL 训练，使 4B-32B 模型在五个指令遵循基准上显著超越基座模型，32B 模型甚至超越部分大型商用模型。

**[KASER: Knowledge-Aligned Student Error Simulator for Open-Ended Coding Tasks](reinforcement_learning/kaser_knowledge-aligned_student_error_simulator_for_open-ended_coding_tasks.md)**

:   KASER先估计学生对知识点的掌握度，再用GRPO和“代码相似度 + 错误匹配 + 多样性”混合奖励训练代码生成器，使其能模拟与学生知识状态一致的编程错误。

**[KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality](reinforcement_learning/knowrl_exploring_knowledgeable_reinforcement_learning_for_factuality.md)**

:   KnowRL 把"原子事实校验"作为过程级奖励直接塞进 GRPO 训练循环，对慢思考模型的 CoT 每一步进行事实判定，同时用"拒答给正奖励"教模型识别自己的知识边界，在 SimpleQA Incorrect Rate 上直降 20.3% 的同时不损失（甚至略升）GPQA / AIME 等推理能力，并展现出英文知识 → 中文 QA 的跨语言迁移。

**[LANG: Reinforcement Learning for Multilingual Reasoning with Language-Adaptive Hint Guidance](reinforcement_learning/lang_reinforcement_learning_for_multilingual_reasoning_with_language-adaptive_hi.md)**

:   LANG 用同语种推理 hint 启动多语言数学推理 RL，再通过余弦衰减和按语言难度自适应关停 hint，让模型在保持语言一致性的同时提升非英语推理准确率。

**[LearnAlign: Data Selection for LLM Reinforcement Learning with Improved Gradient Alignment](reinforcement_learning/learnalign_data_selection_for_llm_reinforcement_learning_with_improved_gradient_.md)**

:   针对 RLVR 后训练的数据选择问题，提出 LearnAlign——把"梯度对齐"作为代表性指标，再用"成功率 $V(\xi)=p(1-p)$"作为可学习性权重消除响应长度偏置，仅用 1000 条数据（约 6%）就在 5 个推理 benchmark 上做到接近全量训练（42.4% vs 44.9%），且 GSM8K 上以 13.4% 数据 (77.5%) 超过全量 (77.0%)。

**[LENS: Less Noise, More Voice — Reinforcement Learning for Reasoning via Instruction Purification](reinforcement_learning/less_noise_more_voice_reinforcement_learning_for_reasoning_via_instruction_purif.md)**

:   LENS 发现 RLVR 中许多探索失败并非因为问题难度，而是因为 prompt 中少量（<5%）干扰 token，通过识别和删除这些 token 来提升 rollout 成功率，并将净化 rollout 的学习信号转移到原始噪声 prompt 的策略优化中，平均提升 3.88% 并加速 1.6 倍。

**[LoVeC: Reinforcement Learning for Better Verbalized Confidence in Long-Form Generations](reinforcement_learning/lovec_reinforcement_learning_for_better_verbalized_confidence_in_long-form_gener.md)**

:   LoVeC 教 LLM 在长文本生成过程中每写完一句就追加一个 0–10 的 `<confidence>` 数字标签，用 GRPO（在线，需 oracle fact-checker）或 DPO（离线偏好对）训练让该标签与 GPT-4o 判定的事实性对齐，单次解码即可输出可校准、可机器解析的置信度，比 SOTA LUQ 在 Brier/ECE/Spearman 上全面更好且推理速度快 20 倍。

**[NaviMaster: Learning a Unified Policy for GUI and Embodied Navigation Tasks](reinforcement_learning/navimaster_learning_a_unified_policy_for_gui_and_embodied_navigation_tasks.md)**

:   NaviMaster 把 GUI 操作和具身导航都改写成“视觉目标定位 + 动作执行”的统一 MDP，并用混合轨迹上的 GRPO 与距离感知密集奖励训练一个 Qwen2.5-VL-7B 策略，在 OOD GUI、空间可供性预测和 ObjectNav 上都超过单域训练与主流基线。

**[RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization](reinforcement_learning/rl-plus_countering_capability_boundary_collapse_of_llms_in_reinforcement_learnin.md)**

:   RL-PLUS 提出混合策略优化方法，通过多重重要性采样（MIS）解决外部数据分布不匹配问题，以及探索式优势函数（EAF）引导模型学习低概率但正确的推理路径，成功突破 RLVR 导致的能力边界坍塌，在六个数学推理基准上达到 SOTA（平均 53.4），且跨模型一致提升最高达 69.2%。

**[Savoir: Learning Social Savoir-Faire via Shapley-based Reward Attribution](reinforcement_learning/savoir_learning_social_savoir-faire_via_shapley-based_reward_attribution.md)**

:   本文提出 Savoir，一个基于合作博弈论的社交 RL 框架，结合期望效用（前瞻性评估话语的战略潜力）和 Shapley 值（公理化公平信用分配）解决多轮对话中的信用分配问题，在 SOTOPIA 基准上以 7B 模型达到 SOTA 性能（Hard 设置 Goal 7.18），匹配或超越 GPT-4o 和 Claude-3.5-Sonnet，且大型推理模型（o1、DeepSeek-R1）在社交任务上系统性欠佳。

**[Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study](reinforcement_learning/scaling_behaviors_of_llm_reinforcement_learning_post-training_an_empirical_study.md)**

:   首次系统研究 LLM 强化学习后训练的缩放行为，在 Qwen2.5 系列(0.5B-72B)上发现性能与训练资源之间遵循幂律关系，且学习效率随模型规模增大呈饱和趋势。

**[Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](reinforcement_learning/semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)**

:   本文指出 RLVR 中传统的 token 级探索-利用权衡是测量方式的伪象，提出在隐状态语义空间中用 Effective Rank (ER) 和其时间导数 (ERV/ERA) 来解耦探索与利用，并据此设计 VERL 方法实现两者的同步提升，在高考数学等基准上获得高达 21.4% 的提升。

**[SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving](reinforcement_learning/spiralthinker_latent_reasoning_through_an_iterative_process_with_text-latent_int.md)**

:   本文提出 SpiralThinker，通过在潜在表示空间中进行迭代更新、并与文本推理步骤交替进行的框架实现隐式推理，引入渐进对齐目标确保潜在表示在迭代过程中保持与显式推理的一致性，在数学、逻辑和常识推理任务上超越所有潜在推理基线。

**[Targeted Exploration via Unified Entropy Control for Reinforcement Learning](reinforcement_learning/targeted_exploration_via_unified_entropy_control_for_reinforcement_learning.md)**

:   本文提出 UEC-RL，一个统一的双向熵控制框架，通过对困难 prompt 进行高温定向探索（增大熵）和通过经验回放稳定器巩固高质量轨迹（减小熵），解决 GRPO 中普遍存在的熵坍塌和训练不稳定问题，在 Geometry3K 上实现 37.9% 的相对提升。

**[The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games](reinforcement_learning/the_stackelberg_speaker_optimizing_persuasive_communication_in_social_deduction_.md)**

:   本文将社交推理游戏中的回合制对话建模为 Stackelberg 博弈，当前玩家作为 leader 通过度量下一玩家的响应分布来优化话语的说服力影响，使用 GRPO 训练 Refiner 模型在狼人杀、阿瓦隆等四个游戏基准上显著超越基线。

**[Understanding Generalization in Role-Playing Models via Information Theory](reinforcement_learning/understanding_generalization_in_role-playing_models_via_information_theory.md)**

:   本文提出首个信息论框架 R-EMID 来量化角色扮演模型（RPM）在用户/角色/对话分布偏移下的性能退化，通过引入推理过程和协同进化强化学习（CoRL）实现准确估计，发现用户偏移是最大的泛化风险，且强化学习是唯一一致有效的改进方法。

**[UniCreative: Unifying Long-form Logic and Short-form Sparkle via Reference-Free Reinforcement Learning](reinforcement_learning/unicreative_unifying_long-form_logic_and_short-form_sparkle_via_reference-free_r.md)**

:   本文提出 UniCreative 框架，通过自适应约束偏好优化（ACPO）和自适应标准生成式奖励模型（AC-GenRM），在无需 SFT 和参考答案的条件下统一长文本（规划→写作）和短文本（直接生成）两种创意写作模式，模型涌现出自主区分任务类型的元认知能力。

**[Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward](reinforcement_learning/verifier-free_rl_for_llms_via_intrinsic_gradient-norm_reward.md)**

:   VIGOR 用每个 completion 在当前模型参数下的 teacher-forced NLL 梯度范数作为内在奖励，偏好低梯度范数输出，并通过 $\sqrt{T}$ 长度校正和组内 rank shaping 稳定 GRPO，从而在无需 gold answer 或外部 verifier 的情况下提升数学与代码推理。

**[Visually-Guided Policy Optimization for Multimodal Reasoning](reinforcement_learning/visually-guided_policy_optimization_for_multimodal_reasoning.md)**

:   VGPO 在 RLVR 训练中用隐藏状态相似度定位视觉相关 token，再通过后段视觉补偿和轨迹内/轨迹间优势重加权强化视觉关注，使 Qwen2.5-VL-7B 在数学多模态推理和视觉依赖任务上超过 GRPO/DAPO 及已有视觉增强 RL 方法。

**[SCRL: What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time](reinforcement_learning/what_if_consensus_lies_selective-complementary_reinforcement_learning_at_test_ti.md)**

:   本文提出 SCRL（Selective-Complementary Reinforcement Learning），一个鲁棒的测试时强化学习框架，通过选择性正伪标签（严格共识标准过滤不可靠多数）和熵门控负伪标签（首次在 TTRL 中引入负监督信号来修剪错误轨迹）缓解标签噪声放大问题，在 AIME25 上比 TTRL 提升高达 10.1 个百分点。

**[Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models](reinforcement_learning/why_does_reinforcement_learning_generalize_a_feature-level_mechanistic_study_of_.md)**

:   本文用严格受控的 SFT/RL 后训练对比和 Sparse Crosscoder 特征对齐发现：SFT 会快速形成大量专门化特征，而 RL 更像是在保留基座表示的同时逐步增强少量跨任务泛化特征，且这些特征被置零会显著伤害 RL 泛化、被放大则能提升基座模型表现。

---

## 👥 社会计算 { #social_computing }

**[Among Us: Language of Conspiracy Theorists on Mainstream Reddit](social_computing/among_us_language_of_conspiracy_theorists_on_mainstream_reddit.md)**

:   分析5亿条Reddit评论的10年纵向数据，发现活跃于阴谋论社区的用户在主流社区中也展现出可检测的独特语言模式（平均87%分类准确率），但这些模式高度依赖社区上下文，社区特定模型比全局模型高出最多17个百分点。

**[Bayesian Social Deduction with Graph-Informed Language Models](social_computing/bayesian_social_deduction_with_graph-informed_language_models.md)**

:   提出 GRAIL（Graph Reasoning Agent Informed through Language），一个混合推理框架，将概率推理外化到因子图模型、用 LLM 处理语言理解和交互，在社交推理游戏 Avalon 中首次击败人类玩家（67% 胜率），且资源消耗远低于大规模推理模型。

**[Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation](social_computing/beyond_the_crowd_llm-augmented_community_notes_for_governing_health_misinformati.md)**

:   作者用 30.8K 条 X 平台健康类 Community Notes 的实证分析揭示出"首条 helpful verdict 中位延迟 17.6 小时、87.9% 笔记永远无评级"的系统性慢响应问题，提出 CrowdNotes+ 框架——以 (1) **证据增强**和 (2) **效用引导自动化** 两种模式让 LLM 写 note，并配以"相关性→正确性→帮助性"三级评估；15 个 LLM 在新 benchmark HealthNotes 上全面超过人工 note 的 73.19% helpfulness（最高 o3 模型达 81.15%）。

**[BITS Pilani at SemEval-2026 Task 9: Structured Supervised Fine-Tuning with DPO Refinement for Polarization Detection](social_computing/bits_pilani_at_semeval-2026_task_9_structured_supervised_fine-tuning_with_dpo_re.md)**

:   本文为 SemEval-2026 POLAR 极化检测任务（英文子集）提出「结构化 slot-filling SFT + DPO 偏好优化」两阶段流水线，赛中提交 Qwen2.5-7B 系统取得 0.7664 Macro-F1，赛后换 Mistral-Nemo-12B + LLM-judge 过滤的偏好对，Macro-F1 提升至 0.8162，超过 organiser baseline (0.7802)。

**[Building Arabic NLP from the Ground Up: Twenty Years of Lessons, Failures, and Open Problems](social_computing/building_arabic_nlp_from_the_ground_up_twenty_years_of_lessons_failures_and_open.md)**

:   这是一篇完整缓存的反思型论文而非实验论文，作者回顾二十年阿拉伯语 NLP 建设，指出低资源语言最难的问题往往不是语言学或模型技术，而是社区、制度、部署治理和知识生产方式。

**[ClaimDB: A Fact Verification Benchmark over Large Structured Data](social_computing/claimdb_a_fact_verification_benchmark_over_large_structured_data.md)**

:   ClaimDB 是首个把事实核查 evidence 放大到 80 个真实数据库、平均每 claim 含 11 张表 / 460 万行 / 1.1 亿 token 的基准，强制方法必须用可执行程序（SQL）做组合推理；对 30 个 SOTA LLM 的工具调用 agent 评测显示，过半模型 accuracy 不到 55%，且封闭模型几乎不会"弃权"、开源模型又过度弃权——NEI 处理是最大短板。

**[Confident, Calibrated, or Complicit: Safety Alignment and Ideological Bias in LLM Hate Speech Detection](social_computing/confident_calibrated_or_complicit_safety_alignment_and_ideological_bias_in_llm_h.md)**

:   作者在 Latent Hatred 基准上让 5 个 LLM（强对齐 vs 弱对齐）戴着 4 种政治 persona 跑零样本仇恨言论分类，发现强对齐模型严格准确率 69.0% 反而高于弱对齐的 64.1%、且对 persona 几乎免疫，但所有模型在反讽、目标群体公平性、置信度校准三方面都出现系统性失败。

**[Content Fuzzing for Escaping Information Cocoons on Social Media](social_computing/content_fuzzing_for_escaping_information_cocoons_on_digital_social_media.md)**

:   提出 ContentFuzz，一个从内容创作者视角出发的置信度引导模糊测试框架，通过 LLM 改写帖子使其在保持人类解读含义不变的前提下改变机器推断的立场标签，从而突破社交媒体信息茧房。

**[Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine](social_computing/decide_less_communicate_more_on_the_construct_validity_of_end-to-end_fact-checki.md)**

:   作者用 5 位临床专家在 RedHOT（Reddit 健康讨论）社交媒体真实声明上做了一项 1,000 实例的标注研究，发现端到端医学事实核查（end-to-end fact-checking）在 construct validity 层面就站不住脚 —— 证据连接难、声明欠规约、严重程度判定主观，三大障碍即便专家也无法消除，因此提出应把医学 fact-checking 重构为「交互式医患沟通模型」而非「分类→裁决」管线。

**[DIA-HARM: Dialectal Disparities in Harmful Content Detection Across 50 English Dialects](social_computing/dia-harm_dialectal_disparities_in_harmful_content_detection_across_50_english_di.md)**

:   本文构建 DIA-HARM，首个跨 50 种英语方言评估虚假信息检测鲁棒性的基准，揭示人类撰写的方言内容导致检测性能下降 1.4-3.6% F1，微调 Transformer 大幅优于零样本 LLM（96.6% vs 78.3%），且部分模型在混合内容上出现超过 33% 的灾难性退化。

**[Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation](social_computing/dynamics_of_cognitive_heterogeneity_investigating_behavioral_biases_in_multi-sta.md)**

:   使用LLM智能体（DeepSeek/GPT系列）在经典啤酒分销博弈中模拟多阶段供应链，系统研究认知异质性（推理能力差异）对系统行为的影响，发现LLM智能体能复现人类的牛鞭效应和短视行为，且信息共享能有效缓解这些不良效应。

**[Estimating the Black-box LLM Uncertainty with Distribution-Aligned Adversarial Distillation](social_computing/estimating_the_black-box_llm_uncertainty_with_distribution-aligned_adversarial_d.md)**

:   提出 DisAAD：用一个仅占目标模型 1% 体积的小代理模型，通过 "分布对齐 + 对抗蒸馏" 学到 "黑盒 LLM 知不知道这道题"，再借证据深度学习把代理模型 logits 拆成认知 / 偶然不确定性，单次响应即可估计 GPT-4/Claude 这类闭源模型的实时不确定性，平均 AUROC 比黑盒 baseline 高 18.2%、AUPR 高 22.9%。

**[Explain the Flag: Contextualizing Hate Speech Beyond Censorship](social_computing/explain_the_flag_contextualizing_hate_speech_beyond_censorship.md)**

:   本文提出一种混合方法，结合 LLM 和三种语言（英/法/希腊语）的人工策展词汇表来检测和解释仇恨言论——术语管道通过词汇匹配+LLM 语义消歧检测固有贬损用语，无术语管道用 LLM 检测群体针对性内容，两者融合生成有据可查的解释。

**[FigSIM: A Dataset for Fine-grained Suicide Severity and Figurative Language in Suicide Memes](social_computing/figsim_a_dataset_for_fine-grained_suicide_severity_and_figurative_language_in_su.md)**

:   FigSIM 构建了首个面向自杀相关 meme 的细粒度多模态数据集，标注 figurative phenomenon、自杀严重度和自杀相关内容，并用 16 类模型验证了当前模型在隐喻、讽刺和高严重度风险识别上仍会系统性低估。

**[GKnow: Measuring the Entanglement of Gender Bias and Factual Gender](social_computing/gknow_measuring_the_entanglement_of_gender_bias_and_factual_gender.md)**

:   本文提出 **GKnow** 基准与一套电路-神经元两级机制分析，证明 LLM 中"性别偏见 (stereotypical gender)"与"事实性别 (factual gender)"在电路层 IoU/cross-task faithfulness 高度重叠、在神经元层共享同一组高 IG 神经元，因此简单的"消融性别偏见神经元"会同时削弱事实性别能力，但在仅评测偏见的 benchmark 上看起来像"成功去偏"，警告这种 debiasing 不可靠。

**[Imperfectly Cooperative Human-AI Interactions: Comparing the Impacts of Human and AI Attributes in Simulated and User Studies](social_computing/imperfectly_cooperative_human-ai_interactions_comparing_the_impacts_of_human_and.md)**

:   通过 2000 次 LLM 模拟和 290 人用户研究的双框架实验，比较了人类个性特质和 AI 设计属性在不完全合作场景（招聘谈判、部分诚实交易）中的影响，发现模拟中个性特质主导而真人实验中 AI 透明度才是关键驱动因素。

**[Inertia in Moral and Value Judgments of Large Language Models](social_computing/inertia_in_moral_and_value_judgments_of_large_language_models.md)**

:   本文用"大规模随机 persona × 道德 / 价值问卷"的范式系统地测出 7 个主流 LLM 在 Harm / Fairness 维度上有高度稳定的"价值惯性"——任何 persona 都很难推动它们的回答方向，并提出 Inertia Index 和 Steerability 两个可量化指标揭示这种偏好其实分布不均、与对齐目标对齐。

**[Investigating Counterfactual Unfairness in LLMs towards Identities through Humor](social_computing/investigating_counterfactual_unfairness_in_llms_towards_identities_through_humor.md)**

:   本文通过幽默场景系统调查 LLM 的反事实不公平性——交换说话者/听众身份后观察模型行为变化，发现特权群体说的笑话被拒绝率高达 67.5%，被判定为恶意的概率高 64.7%，且社会危害评分高达 1.5 分（5分制），揭示了模型内化了固定的社会特权层级而非进行真正的社会推理。

**[Is this chart lying to me? Automating the detection of misleading visualizations](social_computing/is_this_chart_lying_to_me_automating_the_detection_of_misleading_visualizations.md)**

:   提出 Misviz（2604张真实世界误导性可视化）和 Misviz-synth（57665张合成可视化）基准，覆盖12种误导类型，系统评估MLLM、规则检查器和图像分类器在检测误导性图表上的表现，揭示该任务仍极具挑战性。

**[Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews](social_computing/justice_in_judgment_unveiling_hidden_bias_in_llm-assisted_peer_reviews.md)**

:   作者用"只改作者元数据、不改论文内容"的反事实评测在 9 个 LLM 上系统性地审计了 LLM 同行评议偏见，发现所有模型都对名校（RS）有显著好感、对资深 PI 与高产作者更宽容，且关键是：当模型在 hard rating 上看似中立时，soft rating（基于 token 概率的期望评分）暴露出更强的隐藏偏见，揭示了"对齐只是把偏好藏起来而没消除"的对齐失败模式。

**[LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection](social_computing/livefact_a_dynamic_time-aware_benchmark_for_llm-driven_fake_news_detection.md)**

:   LiveFact 把"假新闻检测"从静态二分类升级成一个按月更新、按时间切片证据的动态推理基准，用 Classification + Inference 双模式同时考察 LLM 的事实判断和"该说不知道就说不知道"的认知谦逊，并用 SSA 实体替换显式监控基准污染。

**[mdok-style at SemEval-2026 Task 9: Finetuning LLMs for Multilingual Polarization Detection](social_computing/mdok-style_at_semeval-2026_task_9_finetuning_llms_for_multilingual_polarization_.md)**

:   将原本用于多语言机器生成文本检测的 mdok 系统（QLoRA 微调 Qwen3-32B / Gemma-3-27B）迁移到 SemEval-2026 Task 9 多语言极化检测，并叠加匿名化、大小写、同形字四种数据增强，在 22 种语言上跑出比官方 baseline 平均高 3–4% Macro-F1 的结果。

**[MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection](social_computing/mm-stancedet_retrieval-augmented_multi-modal_multi-agent_stance_detection.md)**

:   作者把多模态立场检测重构成一个 4 阶段 multi-agent pipeline——CLIP 检索相似样本提供 few-shot CoT、文本/图像/跨模态冲突 3 个专家 agent 各自分析、3 个立场（支持/反对/中立）的 debater agent 互相辩论、最后一个 adjudicator agent 做 self-reflection 出最终标签，5 个数据集上 in-target 和 zero-shot 都超过包括 GPT-4V、TMPT、MV-Debate 在内的强 baseline。

**[Persona-E2: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events](social_computing/persona-e2_a_human-grounded_dataset_for_personality-shaped_emotional_responses_t.md)**

:   构建了首个将人格特质（MBTI + Big Five）与读者情感反应关联的大规模数据集 Persona-E2，包含 3111 个事件 × 36 名标注者共 11.2 万条标注，揭示 LLM 在模拟人格化情感反应时存在"人格幻觉"问题，且 Big Five 特征比 MBTI 更有效地缓解该问题。

**[Phase Transitions in Affective Meaning Divergence: The Hidden Drift Before the Break](social_computing/phase_transitions_in_affective_meaning_divergence_the_hidden_drift_before_the_br.md)**

:   这篇论文把对话破裂前的“同词不同情感理解”形式化为 Affective Meaning Divergence，并用熵正则博弈证明修复概率会发生鞍结分岔，再在 Conversations Gone Awry 上观察到方差上升等 critical slowing down 早期预警信号。

**[Point of Order: Action-Aware LLM Persona Modeling for Realistic Civic Simulation](social_computing/point_of_order_action-aware_llm_persona_modeling_for_realistic_civic_simulation.md)**

:   这篇论文把公开 Zoom 会议视频转成跨视频说话人可追踪、带行动标签和 persona 元数据的政府协商语料，并用 QLoRA 微调 LLM 生成特定参与者发言，使困惑度最多降低 67%，人类在图灵式测试中也很难区分模拟对话和真实会议片段。

**[Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation](social_computing/probing_multimodal_large_language_models_on_cognitive_biases_in_chinese_short-vi.md)**

:   这篇论文构建了一个 200 条中文健康短视频的高质量谣言评测集，用证据链、错误类型和社交线索系统评估 8 个前沿 MLLM，发现 Gemini-2.5-Pro 最稳，但多数模型在多模态谣言判断中仍受标签偏置、权威账号和流量指标影响。

**[Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning](social_computing/prompt-level_distillation_a_non-parametric_alternative_to_model_fine-tuning_for_.md)**

:   本文提出 Prompt-Level Distillation，把教师模型在训练样本上的推理规律抽取、聚类并消解冲突后写入学生模型的 system prompt，在不更新参数的情况下显著提升小模型的推理分类能力。

**[PSK@EEUCA 2026: Fine-Tuning Large Language Models with Synthetic Data Augmentation for Multi-Class Toxicity Detection in Gaming Chat](social_computing/pskeeuca_2026_fine-tuning_large_language_models_with_synthetic_data_augmentation.md)**

:   这篇系统论文面向 EEUCA 2026 游戏聊天毒性识别任务，用 Llama 3.1 8B + LoRA + 经过严格过滤的 5% 少数类合成改写数据，在六分类 macro-F1 上达到 0.6234，并揭示了验证集高分但测试集迁移差的“验证陷阱”。

**[Reheat Nachos for Dinner? Evaluating AI Support for Cross-Cultural Communication of Neologisms](social_computing/reheat_nachos_for_dinner_evaluating_ai_support_for_cross-cultural_communication_.md)**

:   本文通过 234 名非母语者和 144 名母语评价者的人类实验比较四类 AI/非 AI 支持，发现带语境解释的 AI Explanation 最能提升非母语者使用英语网络新词时的母语者评分，但学习者的自信和真实交际能力之间仍存在明显错位。

**[RV-HATE: Reinforced Multi-Module Voting for Implicit Hate Speech Detection](social_computing/rv-hate_reinforced_multi-module_voting_for_implicit_hate_speech_detection.md)**

:   RV-HATE 把隐式仇恨言论检测拆成四个面向不同数据特性的 BERT 对比学习模块，再用 PPO 学习数据集特定的软投票权重，在五个 benchmark 上把平均 macro-F1 提升到 84.47%，比 SharedCon 平均高 1.8 个百分点。

**[SMARTER: A Data-efficient Framework to Improve Toxicity Detection with Explanation via Self-augmenting Large Language Models](social_computing/smarter_a_data-efficient_framework_to_improve_toxicity_detection_with_explanatio.md)**

:   SMARTER 用少量标注样本让 LLM 为正确和错误标签分别生成解释，再用偏好优化和跨模型训练提升可解释毒性检测，在 3 个数据集上以 6%-57% 的训练数据达到全量训练 86%-100% 的性能。

**[SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](social_computing/spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)**

:   本文提出 SPAGBias 框架，首次系统评估 LLM 在城市微观空间语境中的性别偏见，通过显式偏见、概率偏见和建构偏见三个诊断层揭示了 LLM 中结构化的空间-性别关联模式，并追溯偏见在模型开发全流程中的嵌入与放大。

**[Splits! Flexible Sociocultural Linguistic Investigation at Scale](social_computing/splits_flexible_sociocultural_linguistic_investigation_at_scale.md)**

:   提出构建社会语言学"沙盒"的方法，从 Reddit 构建了按人口统计群体和讨论话题双重切分的 970 万帖子数据集 Splits!，并设计了基于 lift 和 triviality 的两阶段过滤流程，从 2.3 万条 LLM 生成的候选假设中高效筛选出值得深入研究的社会文化语言现象。

**[Synthia: Scalable Grounded Persona Generation from Social Media Data](social_computing/synthia_scalable_grounded_persona_generation_from_social_media_data.md)**

:   提出 Synthia 框架，基于真实社交媒体帖子（Bluesky）生成有根据的 LLM 人格叙事，在社会调查对齐度上比 SOTA 提升最高 11.6%，同时使用更小的模型，并保留社交网络拓扑结构支持网络感知分析。

**[The Proxy Presumption: From Semantic Embeddings to Valid Social Measures](social_computing/the_proxy_presumption_from_semantic_embeddings_to_valid_social_measures.md)**

:   这篇论文指出 NLP 中把 embedding 几何距离直接命名为“创造力、偏见、创新性”等社会构念是一种 Proxy Presumption，并提出 Construct Validity Protocol 与 Counterfactual Neutralization 来把启发式 proxy 变成可验证的测量工具。

**[To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs](social_computing/to_lie_or_not_to_lie_investigating_the_biased_spread_of_global_lies_by_llms.md)**

:   本文提出 GlobalLies——一个包含 440 个虚假信息生成模板和 6,867 个实体的多语言平行数据集（8 种语言、195 个国家），揭示了 LLM 在传播虚假信息时存在系统性的国家和语言偏差：对低 HDI 国家的虚假信息生成率显著更高（统计相关 $\rho=-0.355$, $p=5\times10^{-7}$），低资源语言的合规率高出英语 30% 以上，且现有安全分类器和 RAG 防护措施提供不均匀的保护。

**[ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](social_computing/toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)**

:   ToxiTrace 提出了一种面向 BERT 类编码器的可解释中文毒性检测方法，通过 CuSA（LLM 引导的弱标注）、GCLoss（梯度约束损失）和 ARCL（对抗推理对比学习）三个组件，在保持高效编码器推理的同时实现了句级分类准确率和连续有毒片段提取的双重提升。

**[VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking](social_computing/veritas_the_first_dynamic_benchmark_for_multimodal_automated_fact-checking.md)**

:   VeriTaS 用一个季度更新的七阶段自动化流水线，把专业事实核查机构的真实多语种图文视频声明转成标准化、可解释、可评测的多模态事实核查基准，并显示当前最强多模态模型离可靠 AFC 仍有明显距离。

**[When Bigger Isn't Better: A Comprehensive Fairness Evaluation of Political Bias in Multi-News Summarisation](social_computing/when_bigger_isn39t_better_a_comprehensive_fairness_evaluation_of_political_bias_.md)**

:   本文构建了首个带政治倾向标签的多文档新闻摘要数据集 FairNews，并通过五维公平性评估框架对 13 个 LLM 进行评估，发现中等规模模型在公平性和效率上优于大模型，且实体情感相似性是最难通过提示去偏的维度。

**[Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text](social_computing/who_gets_which_message_auditing_demographic_bias_in_llm-generated_targeted_text.md)**

:   本文首次系统分析 LLM 在人口统计条件下生成定向消息时的偏见行为，提出 Persuasion Bias Index (PBI) 指标，发现 GPT-4o/Llama/Mistral 在气候传播中对男性和年轻人使用更强势的说服策略，且上下文提示会系统性地放大这些差异。

**[Why Are We Moral? An LLM-based Agent Simulation Approach to Study Moral Evolution](social_computing/why_are_we_moral_an_llm-based_agent_simulation_approach_to_study_moral_evolution.md)**

:   这篇论文用 LLM agent 构建史前狩猎采集社会仿真平台，把道德类型、记忆、判断、协作和繁衍都纳入演化实验，发现合作和互助通常提升生存稳定性，而判断他人道德类型的认知成本会改变哪类道德策略胜出。

**[YEZE at SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization via Heterogeneous Ensembling](social_computing/yeze_at_semeval-2026_task_9_detecting_multilingual_multicultural_and_multievent_.md)**

:   YEZE 系统把 SemEval-2026 Task 9 的 22 语言在线极化识别拆成独立子任务，用 XLM-RoBERTa-large 与 mDeBERTa-v3-base 分别微调，再通过加权概率平均和 weighted BCE 缓解多标签稀疏问题，在细粒度极化类型与表现形式预测上取得稳定的官方 Top-10 排名。

---

## 👥 多智能体 { #multi_agent }

**[A Multi-Agent Framework for Feature-Constrained Difficulty Control in Reading Comprehension Item Generation](multi_agent/a_multi-agent_framework_for_feature-constrained_difficulty_control_in_reading_co.md)**

:   本文提出 MAFIG，一个用多智能体协作、特征级评估器和迭代修订来生成阅读理解选择题的框架，相比单轮提示能显著提高题目对词汇、篇幅、句长、推理复杂度、事实性和选项中立性等约束的满足率，并带来更稳定的难度递增。

**[AgenticEval: Toward Agentic and Self-Evolving Safety Evaluation of Large Language Models](multi_agent/agenticeval_toward_agentic_and_self-evolving_safety_evaluation_of_large_language.md)**

:   AgenticEval 把 LLM 安全评估重新定义为「持续、自我演化的红队过程」：Specialist 把非结构化法规文本拆成原子规则知识库，Generator 围绕每条规则生成多模态多形式的 Question Group，Evaluator + Analyst 不断把当轮失败转化为下一轮更狠的攻击策略，三轮迭代后 GPT-5 对 EU AI Act 的合规率从 72.50% 暴跌到 36.36%，揭示静态 benchmark 严重高估了大模型的安全水位。

**[ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](multi_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)**

:   提出 ATLAS 多智能体金融交易框架和 Adaptive-OPRO 提示优化方法，通过专业化分析师智能体准备异构市场信息，并基于延迟噪声反馈动态优化中央交易智能体的指令提示，在多种市场波动环境中显著超越基线。

**[AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage](multi_agent/autoreproduce_automatic_ai_experiment_reproduction_with_paper_lineage.md)**

:   AutoReproduce 提出了一个多智能体框架，通过"论文谱系"算法从引用文献中挖掘隐式领域知识，实现端到端的论文实验自动复现，在自建基准 ReproduceBench 上的代码执行率达 94.87%，性能差距仅 19.72%。

**[BookAgent: Orchestrating Safety-Aware Visual Narratives via Multi-Agent Cognitive Calibration](multi_agent/bookagent_orchestrating_safety-aware_visual_narratives_via_multi-agent_cognitive.md)**

:   BookAgent 是一个安全感知的多智能体框架，通过**价值对齐故事板（VAS）+ 迭代跨模态精炼（ICR）+ 时序认知校准（TCC）**三阶段闭环架构，从用户草稿端到端生成高质量、角色一致、内容安全的绘本故事。

**[CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](multi_agent/cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)**

:   本文提出 CIA（Communication Inference Attack），在严格黑盒只能观测最终输出的设定下，通过对抗性查询诱导多智能体系统暴露中间 agent 的推理输出，再用全局偏置解纠缠 + LLM 弱监督建模语义相关性，成功反演出 MAS 的通信拓扑，平均 AUC 0.87、峰值 0.99。

**[Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](multi_agent/collaborative_multi-agent_scripts_generation_for_enhancing_imperfect-information.md)**

:   提出一个协作式多智能体框架用于自动生成高质量剧本杀游戏脚本和训练数据，通过两阶段训练策略（CoT 微调 + GRPO 强化学习配合 ScoreAgent 奖励塑形）增强 VLM 在不完全信息下的多跳推理能力，在 WhodunitBench 上显著提升 VLM 的叙事推理、事实提取和欺骗抵御能力。

**[Conjunctive Prompt Attacks in Multi-Agent LLM Systems](multi_agent/conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)**

:   本文研究多智能体 LLM 系统中的联合提示攻击（conjunctive prompt attacks）：用户查询中嵌入的触发键和被入侵远程代理中的隐藏模板各自看起来无害，但当路由将它们带到同一代理时会激活有害行为，现有防御（PromptGuard、Llama-Guard 等）均无法可靠阻止。

**[ConSensus: Multi-Agent Collaboration for Multimodal Sensing](multi_agent/consensus_multi-agent_collaboration_for_multimodal_sensing.md)**

:   ConSensus 是一个无需训练的多智能体传感器融合框架，它把不同传感模态交给专门 agent 独立解释，再用语义融合、统计共识和混合仲裁得到最终判断，在 5 个多模态传感 benchmark 上比单 agent 平均提升 7.1% accuracy，并把融合 token 成本降到多轮 debate 方法的约 1/12.7。

**[Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection](multi_agent/debating_the_unspoken_role-anchored_multi-agent_reasoning_for_half-truth_detecti.md)**

:   提出RADAR框架，通过角色锚定（政客 vs 科学家）的多智能体辩论来检测基于遗漏上下文的半真半假信息，配合双阈值自适应早停机制，在噪声检索条件下一致超越单智能体和传统多智能体基线。

**[Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate](multi_agent/dialectic-med_mitigating_diagnostic_hallucinations_via_counterfactual_adversaria.md)**

:   提出 Dialectic-Med，一个受波普尔证伪主义启发的多智能体医学诊断框架，通过提议者（诊断假设）、反对者（视觉证伪模块主动检索矛盾视觉证据）和调解者（加权共识图决策）的对抗辩证推理，在 MIMIC-CXR-VQA、VQA-RAD 和 PathVQA 上取得 SOTA，解释忠实度提升 12.5%，显著缓解诊断幻觉。

**[Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](multi_agent/diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)**

:   本文通过评估超过 10,000 个研究提案，从模型智能、智能体认知和系统动力学三个层次系统揭示了多智能体 LLM 系统中的"多样性崩溃"现象：更强的模型、权威驱动的角色分配和密集的通信拓扑都会抑制语义多样性，根本原因是交互结构而非模型能力不足。

**[Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search](multi_agent/efficient_multi-agent_system_training_with_data_influence-oriented_tree_search.md)**

:   提出 DITS，把 "训练数据影响分数 (influence score)" 而非传统 Q-value 作为 MCTS 树搜索和偏好数据选择的指挥棒，并为不可微指标推导出一个 "前向推理就能算" 的影响分数估计公式，使 MAS 在 7 个数据集 / 3 个多智能体任务上平均比 Optima-iSFT-DPO 再提升 2.5–2.7%。

**[EvoSci: A Bio-Inspired Multi-Agent Framework for the Evolution of Scientific Discovery](multi_agent/evosci_a_bio-inspired_multi-agent_framework_for_the_evolution_of_scientific_disc.md)**

:   本文提出 EvoSci，把科研 idea 生成建模为多智能体协作和生物启发式演化循环，通过问题空间构建、团队式研究执行、评审反馈和实体级交叉/变异/选择，在 10 个科研主题上生成更有新颖性和整体质量的研究想法。

**[EvoSpark: Endogenous Interactive Agent Societies for Unified Long-Horizon Narrative Evolution](multi_agent/evospark_endogenous_interactive_agent_societies_for_unified_long-horizon_narrati.md)**

:   EvoSpark 提出一个支持长程叙事演化的多智能体框架，通过分层递归记忆（RSB 做社会认知代谢）、生成式场面调度（GMS 做角色-地点-情节对齐）和涌现角色锚定协议（ECGP 将 LLM 幻觉转化为持久角色）三重设计解决社会记忆堆叠和叙事-空间失谐问题。

**[Explicit Trait Inference for Multi-Agent Coordination](multi_agent/explicit_trait_inference_for_multi-agent_coordination.md)**

:   提出显式特质推理（ETI）方法，基于心理学中温暖和能力两个维度让LLM智能体推理并追踪合作伙伴的行为特征，在经济博弈中减少45-77%收益损失，在MultiAgentBench上提升3-29%任务表现。

**[MAGEO: From Experience to Skill — Multi-Agent Generative Engine Optimization via Reusable Strategy Learning](multi_agent/from_experience_to_skill_multi-agent_generative_engine_optimization_via_reusable.md)**

:   本文将生成引擎优化（GEO）从逐实例启发式优化重构为策略学习问题，提出 MAGEO 多智能体框架——执行层由偏好/规划/编辑/评估四个智能体协作，学习层将验证有效的编辑模式蒸馏为可复用的引擎特定策略技能，并引入 Twin Branch 因果评估协议和 DSV-CF 双轴指标，在三个主流引擎上显著优于启发式基线。

**[From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](multi_agent/from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)**

:   本文构建了JurisCQAD——一个包含43000+真实中文法律咨询的大规模数据集，并提出JurisMA多智能体框架，通过法律元素图进行结构化任务分解和动态多Agent协作（管理Agent+格式检查+法条检索），在LawBench上显著优于通用和法律专用LLM。

**[HACHIMI: Scalable and Controllable Student Persona Generation via Orchestrated Agents](multi_agent/hachimi_scalable_and_controllable_student_persona_generation_via_orchestrated_ag.md)**

:   HACHIMI 把"学生画像生成"形式化为 TAD-PG（理论对齐 + 分布可控）任务，用"提议–验证–修订"多智能体框架配合神经符号验证器和分层采样，产出 100 万条 1–12 年级合成学生画像；在 CEPS / PISA 2022 群体级评测中显示出明显的「保真梯度」——数学与好奇心相关构念高度对齐，而幸福感和家庭动态构念则只能弱对齐。

**[Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate](multi_agent/latent_agents_a_post-training_procedure_for_internalized_multi-agent_debate.md)**

:   提出 IMAD（Internalized Multi-Agent Debate）框架，用 SFT + GRPO 两阶段后训练把多智能体辩论"内化"进单个 LLM，token 消耗最多减少 93%，并通过激活引导证明内化后的模型在隐空间中保留了可分离、可控的"智能体子空间"。

**[LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey](multi_agent/llm-based_human-agent_collaboration_and_interaction_systems_a_survey.md)**

:   本文首次系统性梳理"LLM 基础的人-agent 协作系统（LLM-HAS）"——把人重新拉回 agent loop，从环境/画像、人类反馈、交互类型、编排范式、通信结构 5 个维度建立统一分类，并补充了一个 A1–A5 的 Human Agency Scale 量化"任务里到底该让人参与多深"。

**[MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing](multi_agent/masfactory_a_graph-centric_framework_for_orchestrating_llm-based_multi-agent_sys.md)**

:   MASFactory 把 LLM 多智能体系统建模成 Node / Edge 计算图，提出 "Vibe Graphing" 三阶段流水线（Role Assignment → Structure Design → Semantic Completion）把自然语言意图编译成可执行 MAS 工作流，并提供 Context/Message Adapter、ComposedGraph 模板复用与 VS Code 可视化；7 个 benchmark 上复现 5 个代表性 MAS 且效果持平甚至更好，端到端 Vibe Graphing 把 ChatDev 1511 行代码压到 45 行、API 成本比 Vibe Coding 低 1 个数量级。

**[MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering](multi_agent/mata_multi-agent_framework_for_reliable_and_flexible_table_question_answering.md)**

:   提出 MATA 多Agent表格问答框架，通过调度器优先选择推理路径（CoT/PoT/text2SQL）、置信度检查器筛选答案、法官Agent仲裁，实现模型无关的高效准确表格QA，在10个LLM上平均EM提升40.1%。

**[Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](multi_agent/memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)**

:   提出 MALMAS，一个记忆增强的 LLM 多智能体系统用于表格数据自动特征生成，通过六个专职 Agent 分工探索不同特征空间维度 + 三级记忆机制（过程/反馈/概念）实现跨轮迭代优化，在 16 个分类和 7 个回归数据集上超越现有基线。

**[Multi-Agent Reasoning Improves Compute Efficiency: Pareto-Optimal Test-Time Scaling](multi_agent/multi-agent_reasoning_improves_compute_efficiency_pareto-optimal_test-time_scali.md)**

:   这篇论文把 self-consistency、self-refinement、multi-agent debate 和 Mixture-of-Agents 放到统一计算预算下比较，发现多智能体推理尤其是 MoA 在 Pareto 前沿上更高效，最高可在约 20 倍 CoT 预算下把 MMLU-Pro 准确率从 64.3% 提升到 71.4%。

**[ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification](multi_agent/odutqa-mdc_a_task_for_open-domain_underspecified_tabular_qa_with_multi-turn_dial.md)**

:   本文提出 ODUTQA-MDC 任务和基准，首次系统研究开放域场景下用户查询模糊性的检测与多轮对话澄清问题，构建了包含 25,105 个 QA 对的大规模数据集，并设计了 MAIC-TQA 多智能体框架来完成"检测-澄清-推理"的端到端表格问答。

**[OxyGent: Making Multi-Agent Systems Modular, Observable, and Evolvable via Oxy Abstraction](multi_agent/oxygent_making_multi-agent_systems_modular_observable_and_evolvable_via_oxy_abst.md)**

:   OxyGent 把 agent、工具、LLM 与推理流程统一封装成可插拔的 Oxy 原子组件，并用权限驱动的动态规划与 OxyBank 数据回流机制，让工业级多智能体系统更容易搭建、监控和持续演化。

**[PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation](multi_agent/posterforest_hierarchical_multi-agent_collaboration_for_scientific_poster_genera.md)**

:   PosterForest 用一个同时编码论文层次语义和海报空间布局的 Poster Tree 作为中间表示，再让 Content Agent、Layout Agent 和 Feedback Agent 递归协同优化，训练-free 地生成科学海报，并在人评中获得 59.2% overall preference，明显优于 P2P 和 Paper2Poster。

**[Preference Estimation via Opponent Modeling in Multi-Agent Negotiation](multi_agent/preference_estimation_via_opponent_modeling_in_multi-agent_negotiation.md)**

:   提出将 LLM 提取的自然语言偏好信号与贝叶斯对手建模框架结合的偏好估计方法，在多方多议题谈判中通过语言似然函数融合定性线索和定量出价信息，将完全达成协议率从 37% 提升至 62%。

**[PROTEA: Offline Evaluation and Iterative Refinement for Multi-Agent LLM Workflows](multi_agent/protea_offline_evaluation_and_iterative_refinement_for_multi-agent_llm_workflows.md)**

:   PROTEA 是一个面向多智能体 LLM 工作流的离线调试平台，通过节点级评估、反向生成中间期望和可编辑 prompt 修订，把“最终答案变差了”定位到具体节点并闭环验证修改效果。

**[RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems](multi_agent/roadmapper_a_multi-agent_system_for_roadmap_generation_of_solving_complex_resear.md)**

:   本文提出 RoadMap 研究路线图生成基准和 RoadMapper 多智能体系统，用知识检索、逻辑/粒度批评、修订与 DPO 评估器组成闭环，在中英文复杂科研问题上比直接提示平均提升约 7-9 分，并显著降低专家设计路线图的时间成本。

**[Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](multi_agent/scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)**

:   提出 ExtAgents 多智能体框架，通过全局知识同步（所有Seeking Agent间交换信息）和知识累积推理（逐步向Reasoning Agent注入筛选后的知识）两个机制，解决现有多智能体方法在扩展外部知识输入超出上下文窗口时性能不升反降的瓶颈，在多跳QA和长综述生成任务上显著提升。

**[Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems](multi_agent/seeing_the_whole_elephant_a_benchmark_for_failure_attribution_in_llm-based_multi.md)**

:   TraceElephant 主张多智能体失败归因应在开发者真实可见的完整执行轨迹下评测，并提供 220 条失败 trace、责任 agent 与关键失败 step 标注，证明 full observability 能将 step-level 归因从输出-only 的 16% 提升到 28%-30% 以上。

**[SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](multi_agent/silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)**

:   本文提出 SILO-BENCH，一个角色无关的多智能体 LLM 分布式协调基准，包含 30 个算法任务、三个通信复杂度级别、54 种配置共 1620 个实验，揭示了关键的"通信-推理鸿沟"：智能体能自发形成合理通信拓扑并积极交换信息，但系统性地无法将分布式状态整合为正确答案。

**[Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives](multi_agent/social_dynamics_as_critical_vulnerabilities_that_undermine_objective_decision-ma.md)**

:   这篇论文证明，LLM 多智能体系统中的代表智能体不仅会受自身推理能力限制，还会被同伴数量、同伴能力、论证长度和修辞风格等“社会动力学”显著影响，从而在有客观答案的任务上做出错误决策。

**[To Trust or Not to Trust: Attention-Based Trust Management for LLM Multi-Agent Systems](multi_agent/to_trust_or_not_to_trust_attention-based_trust_management_for_llm_multi-agent_sy.md)**

:   本文为 LLM 多智能体系统（LLM-MAS）提出首个全面的"可信度"定义（基于 Grice 合作原则的六个正交维度），发现 LLM 的注意力模式可区分不同类型的可信度违规，据此设计了轻量级的 A-Trust 评估方法和端到端的信任管理系统（TMS），在多种攻击下将恶意消息检测率提升至 77-90%。

**[Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](multi_agent/towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)**

:   提出 SpreadsheetAgent，一种两阶段多智能体框架，通过代码执行、视觉和 LaTeX 三种格式的渐进式区域读取与交叉验证，在不超出 LLM 上下文限制的前提下实现鲁棒的真实世界电子表格理解。

**[Towards Self-Improving Error Diagnosis in Multi-Agent Systems](multi_agent/towards_self-improving_error_diagnosis_in_multi-agent_systems.md)**

:   提出 ErrorProbe 框架，通过 MAST 分类驱动的结构化分解、症状驱动的后向追踪和验证式记忆机制，在多智能体系统中实现自改进的语义故障归因，尤其在步骤级错误定位上大幅超越基线。

**[When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning](multi_agent/when_identity_skews_debate_anonymization_for_bias-reduced_multi-agent_reasoning.md)**

:   这篇论文指出多智能体辩论中的 LLM 会因为“谁说的”而不是“说了什么”改变立场，并通过响应匿名化与 Identity Bias Coefficient 量化和削弱这种身份驱动偏差。

---

## 📖 NLP 理解 { #nlp_understanding }

**[A Computational Method for Measuring "Open Codes" in Qualitative Analysis](nlp_understanding/a_computational_method_for_measuring_34open_codes34_in_qualitative_analysis.md)**

:   提出一种基于理论的计算方法，通过LLM增强的代码合并算法和四个无需ground truth的指标（Coverage, Overlap, Novelty, Divergence），系统评估人类和AI在归纳定性编码中的表现。

**[Accurate and Efficient Statistical Testing for Word Semantic Breadth](nlp_understanding/accurate_and_efficient_statistical_testing_for_word_semantic_breadth.md)**

:   本文指出"在上下文嵌入空间用置换检验直接比较两个词的语义广度"会因均值方向差异而严重虚高 Type-I 错误，提出用 Householder 反射先对齐均值方向再做置换的方法，把 Type-I 错误降低 32.5%，并给出 GPU 批量化实现实现 23 倍加速。

**[AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models](nlp_understanding/adaptime_enabling_adaptive_temporal_reasoning_in_large_language_models.md)**

:   本文提出 AdapTime，把"时间推理"抽象为 reformulate / rewrite / review 三个可复用的原子动作，由 LLM Planner 根据问题与上下文自适应决定执行哪几步、按什么顺序执行，无需任何外部工具、手工规则或微调即可显著提升 LLM 在时序 QA 上的表现，在 DeepSeek-V3 上把 TimeQA-Easy 推到 85.4 EM。

**[Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations](nlp_understanding/agree_disagree_explain_decomposing_human_label_variation_in_nli_through_the_lens.md)**

:   将LiTEx推理分类法从"标签一致下的解释变异"扩展到"标签不一致"场景，发现标注者可能标签不同但推理类似，推理类别的一致性比标签一致性更好地反映解释的语义相似度。

**[ASTRA: Adaptive Semantic Tree Reasoning Architecture for Complex Table Question Answering](nlp_understanding/astra_adaptive_semantic_tree_reasoning_architecture_for_complex_table_question_a.md)**

:   ASTRA 把复杂表格自适应重构为语义树，再用文本树导航和符号代码执行双模式推理回答问题，在 AIT-QA、SSTQA 和 HiTab 上分别达到 91.6%、81.9% 和 90.1% 准确率，超过强 LLM 和已有表格结构化方法。

**[Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering](nlp_understanding/beyond_chunking_discourse-aware_hierarchical_retrieval_for_long_document_questio.md)**

:   用修辞结构理论（RST）解析长文档的篇章结构，构建一棵句子级层级树并对中间节点做 LLM 摘要增强，最后在树上做结构感知的多粒度检索，使长文档 QA 在 QASPER / QuALITY / NarrativeQA / MultiFieldQA-zh 四个 benchmark 一致超越固定切块和 RAPTOR 语义聚类。

**[BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation](nlp_understanding/boundrl_efficient_structured_text_segmentation_through_reinforced_boundary_gener.md)**

:   BoundRL 将结构化文本分割重新定义为边界生成任务——仅生成每个片段的起始 token 而非完整文本，减少 90% 的输出 token 并消除幻觉风险，结合双目标奖励函数和选择性扰动策略的 RLVR 训练，使 1.7B 小模型超越了 Claude-4 Sonnet 的 few-shot 表现。

**[Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?](nlp_understanding/can_llms_estimate_cognitive_complexity_of_reading_comprehension_items.md)**

:   这篇论文构建 ReCo 阅读理解认知复杂度数据集，并系统评估 8 个 LLM 是否能自动判断题目所需的证据范围和表述转换层级，结果显示强模型能接近但仍明显低于专家，尤其不擅长识别完整证据集合和细粒度词序转换。

**[Commonsense Knowledge with Negation: A Resource to Enhance Negation Understanding](nlp_understanding/commonsense_knowledge_with_negation_a_resource_to_enhance_negation_understanding.md)**

:   提出自动为现有常识知识库增添否定的方法，构建超过 200 万三元组的否定常识语料库（¬Atomic 和 ¬Anion），并证明在其上预训练可以提升 LLM 的否定理解能力。

**[Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs](nlp_understanding/creating_conlangs_to_probe_the_metalinguistic_grammatical_knowledge_of_llms.md)**

:   本文提出 IASC（Interactive Agentic System for ConLangs），一个模块化的人造语言构建系统，通过让 LLM 按语言学规格执行形态句法变换来探测其元语言知识，发现 LLM 处理常见语言类型模式远优于罕见模式，且不同 LLM 之间能力差异悬殊。

**[DimABSA: Building Multilingual and Multidomain Datasets for Dimensional Aspect-Based Sentiment Analysis](nlp_understanding/dimabsa_building_multilingual_and_multidomain_datasets_for_dimensional_aspect-ba.md)**

:   作者构建了首个多语言（6 种语言）多领域（4 个领域）的维度型方面级情感分析数据集 DimABSA（76,958 个 aspect 实例 / 42,590 句），用连续的 valence–arousal 分数取代传统的「正/负/中」三分类，并设计 3 个新子任务和统一指标 cF1，对 6 个开/闭源 LLM 进行了系统评测。

**[DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition](nlp_understanding/diziner_disagreement-guided_instruction_refinement_via_pilot_annotation_simulati.md)**

:   DiZiNER 通过模拟人工标注中的"预标注"流程，利用多个异构 LLM 作为标注员、一个监督 LLM 分析模型间分歧并迭代优化任务指令，在18个NER基准上实现了14个数据集的零样本SOTA，平均提升+8.0 F1，且超越了作为监督者的GPT-5 mini。

**[EXCEEDS: Extracting Complex Events via Nugget-based Grid Modeling in Scientific Domain](nlp_understanding/exceeds_extracting_complex_events_via_nugget-based_grid_modeling_in_scientific_d.md)**

:   作者发现"科学文献摘要"这种 EE 场景同时存在**信息密度高**（每 100 token 5.54 个事件 + 12.82 个 argument）和**事件结构复杂**（重叠/不连续/逆序 nugget + 子事件）两个老 EE 数据集都没碰过的痛点，于是 (a) 标注了 2,508 文档/24,381 事件的 SciEvents 数据集，(b) 提出 EXCEEDS——把 EE 重构成"在 $l \times l$ word-word 网格上做多标签关系分类"的端到端框架，用 HTL/THL/EAL 三种边把 trigger/argument/sub-event 全部统一建模，在主指标和复杂场景上都打过 9 个 SOTA baseline。

**[Exploring Concreteness Through a Figurative Lens](nlp_understanding/exploring_concreteness_through_a_figurative_lens.md)**

:   作者用 prompt-based probing + DiffMean + SVD 拆解四个 LLM（Llama-3.1-8B / Qwen3-8B / Gemma2-9B / GPT-OSS-20B）内部的"具体性"（concreteness）表示，发现：早期层就已经能区分名词的字面用法（高 concrete）vs 比喻用法（低 concrete），中后期层把整个 concreteness 信息压缩到**一条一维方向**上，并展示这条 axis 既能做几乎和有监督 4096 维分类器持平的零样本 figurative text 分类、又能直接被加到 hidden state 上对生成做"字面 ↔ 比喻"的可控改写。

**[Filling the Gap: Is Commonsense Knowledge Generation useful for Natural Language Inference?](nlp_understanding/filling_the_gap_is_commonsense_knowledge_generation_useful_for_natural_language_.md)**

:   论文让 LLM 自己生成连接 premise 和 hypothesis 的自然语言"常识公理"，再用一个"factuality 判官"过滤掉不靠谱的公理只留下高质量的注入回 NLI 提示，结果 Llama-3.1-70B 和 gpt-oss-120b 在 SNLI/ANLI 上准确率提升 1.99-6.88%，并显著减弱了"宁可选 Neutral"的安全偏好。

**[HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction](nlp_understanding/hcre_llm-based_hierarchical_classification_for_cross-document_relation_extractio.md)**

:   提出 HCRE 模型，通过构建层次化关系树将跨文档关系抽取从大规模关系集的直接分类转化为逐层层次化分类，并设计预测-验证推理策略缓解层间错误传播，在 CodRED 数据集上显著超越 SLM 和 LLM 基线。

**[It's High Time: A Survey of Temporal Question Answering](nlp_understanding/it39s_high_time_a_survey_of_temporal_question_answering.md)**

:   本文提供了时序问答（TQA）的全面综述，提出了基于语料时间性、问题时间性和模型时间能力三个维度的统一分析框架，系统梳理了从规则管道到 Transformer/LLM 时代的 TQA 方法演进、基准数据集和评估策略，并识别了未来挑战。

**[Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation](nlp_understanding/knowledge-driven_augmentation_and_retrieval_for_integrative_temporal_adaptation.md)**

:   KARITA 把"时序漂移"拆成不确定性、特征距离和本体术语稀有度三种互补信号，对每个被命中的目标样本回溯检索语义相近的源样本，再用 LLM + 领域本体（MeSH / EuroVoc / CSO）生成同义词改写做数据增强，从而以纯数据驱动的方式把源时段模型迁移到未来时段，在临床、法律、科学三类长跨度多标签分类数据上稳定优于强基线。

**[LexRel: Benchmarking Legal Relation Extraction for Chinese Civil Cases](nlp_understanding/lexrel_benchmarking_legal_relation_extraction_for_chinese_civil_cases.md)**

:   构建了首个中国民事法律关系的结构化分类体系（9 大领域、265 种关系类型），并基于此提出 LexRel 基准（1,140 个专家标注样本），评估了主流 LLM 在法律关系抽取任务上的能力，发现当前模型在该任务上存在显著局限，同时证明了法律关系信息对下游法律 AI 任务的增益效果。

**[LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](nlp_understanding/llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)**

:   本文提出 LLM 引导的语义引导框架，通过 LLM 生成子意图和三阶段课程式合成数据训练非否定 Tsetlin Machine（NTM），提取高置信度符号特征注入真实数据，使标准 TM 在保持完全可解释性的同时逼近 BERT 的分类性能。

**[Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models](nlp_understanding/lost_in_the_prompt_order_revealing_the_limitations_of_causal_attention_in_langua.md)**

:   本文深入研究了大语言模型在多选题问答中对提示组件顺序的敏感性，通过系统性实验排除了训练偏差和记忆衰退假说，揭示了因果注意力掩码是导致 QOC（问题-选项-上下文）顺序性能大幅下降的根本机制。

**[MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](nlp_understanding/made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)**

:   本文提出 MADE——一个基于 FDA 医疗设备不良事件报告的"活"多标签文本分类基准，包含 1,154 个层次化标签和严格的时间分割，系统评估了 20+ 编码器/解码器模型在判别式微调、生成式微调和 few-shot 提示下的预测性能和不确定性量化（UQ）能力，揭示了关键权衡：小型判别式微调解码器在头到尾准确率上最优，生成式微调的 UQ 最可靠，大型推理模型提升稀有标签但 UQ 意外较弱。

**[MetFuse: Figurative Fusion between Metonymy and Metaphor](nlp_understanding/metfuse_figurative_fusion_between_metonymy_and_metaphor.md)**

:   作者提出一个三阶段（候选生成 → MLM 打分挑选 → LLM 润色）流水线，把一句字面句子同时改写成转喻 / 隐喻 / 混合（hybrid）三种 figurative 变体，构造出首个 1000 quadruplet × 4000 句的 MetFuse 数据集，并实证发现"隐喻动词的出现会让同句中的转喻名词变得更显性"，在 8 个 metonymy/metaphor 分类基准上做数据增强一致涨点。

**[MSMO-ABSA: Multi-Scale and Multi-Objective Optimization for Cross-Lingual Aspect-Based Sentiment Analysis](nlp_understanding/msmo-absa_multi-scale_and_multi-objective_optimization_for_cross-lingual_aspect-.md)**

:   针对跨语言 aspect-based 情感分析提出 MSMO 框架——句级用 Wasserstein 对抗训练 + 代码切换数据做语言判别器对齐，aspect 级用双向 KL 一致性训练对齐同情感 aspect 的预测分布，再叠加多教师知识蒸馏，在 SemEval-2016 4 个目标语言 + mBERT/XLM-R 上稳定刷出新 SOTA，并显著超过 GPT-4o / Qwen2.5-7B-LoRA 等 LLM 方案。

**[MTSQL-R1: Towards Long-Horizon Multi-Turn Text-to-SQL via Agentic Training](nlp_understanding/mtsql-r1_towards_long-horizon_multi-turn_text-to-sql_via_agentic_training.md)**

:   MTSQL-R1把多轮 Text-to-SQL 从“一次性翻译”改造成可与数据库和对话记忆交互的长程智能体训练问题，通过自教式 warm-start SFT 与多层级 GRPO 奖励，让小规模 Qwen3 模型在 CoSQL 和 SParC 上超过强闭源提示基线与短程 SFT/RL 基线。

**[Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs](nlp_understanding/reasoning-based_refinement_of_unsupervised_text_clusters_with_llms.md)**

:   提出基于推理的聚类精炼框架，将 LLM 作为语义判官（而非嵌入生成器）验证和重构无监督聚类的输出，通过一致性验证、冗余裁决和标签接地三个推理阶段，在社交媒体语料上显著提升聚类一致性和人类对齐的标注质量。

**[Refining and Reusing Annotation Guidelines for LLM Annotation](nlp_understanding/refining_and_reusing_annotation_guidelines_for_llm_annotation.md)**

:   这篇论文把传统人工标注项目中的 guideline reuse 和 moderation 流程迁移到 LLM 标注中，证明显式标注规范、推理型模型和少量 gold discrepancy 驱动的迭代规范细化，都能提升生物医学 NER 的严格 span+type F1。

**[SAM-NER: Semantic Archetype Mediation for Zero-Shot Named Entity Recognition](nlp_understanding/sam-ner_semantic_archetype_mediation_for_zero-shot_named_entity_recognition.md)**

:   SAM-NER 用“实体发现 → 14 类通用语义原型 → 目标类型定义校准”的三阶段中介框架缓解零样本 NER 的 schema drift，在 CrossNER 上取得 66.3 平均 micro-F1，超过一系列强基线。

**[Semantic Reranking at Inference Time for Hard Examples in Rhetorical Role Labeling](nlp_understanding/semantic_reranking_at_inference_time_for_hard_examples_in_rhetorical_role_labeli.md)**

:   提出 RiSE，一种推理时语义重排序框架，通过自动识别低置信度困难样本并利用对比学习的标签语义表示重排序模型输出，在 8 个修辞角色标注数据集上困难样本平均提升 +9.15 macro-F1。

**[Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](nlp_understanding/table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)**

:   全面综述了 LLM 时代表格问答（TQA）研究，从五个维度（表格格式、问题复杂度、答案格式、模态、领域）系统化分类任务设置，按核心挑战（表格理解、复杂查询、大输入、数据异构、知识集成）组织建模方法，覆盖 277 篇论文，并前瞻性讨论了强化学习、可解释性等新兴方向。

**[Test-Time Reasoners Are Strategic Multiple-Choice Test-Takers](nlp_understanding/test-time_reasoners_are_strategic_multiple-choice_test-takers.md)**

:   这篇论文系统比较 12 个推理 LLM 在完整多选题和只看选项的多选题上的表现，发现测试时推理确实会让模型在 choices-only 场景中高于随机，但推理轨迹显示其中不全是浅层作弊，也包含推断缺失问题、排除错误选项和调用事实知识等更像“策略性应试”的行为。

**[The Imperfective Paradox in Large Language Models](nlp_understanding/the_imperfective_paradox_in_large_language_models.md)**

:   这篇论文用新构造的 ImperfectiveNLI 诊断集检验 LLM 是否理解“正在做某事”不一定意味着“已经完成某事”，发现开源 LLM 普遍会把有目标事件误判为已完成，提示工程只能在减少完成幻觉和保留合法蕴含之间摇摆，真正问题在于推理阶段被目的论先验主导。

**[新闻文本中的时间框架揭示](nlp_understanding/uncovering_temporal_framing_in_the_news.md)**

:   本文提出了新闻文本中"时间框架"的概念——从社会科学理论出发，建立包含 8 类时间框架的分类体系，标注了英德双语新闻语料库，并用有监督和零样本两种方式训练模型进行时间框架检测。

---

## ⚖️ 对齐 / RLHF { #llm_alignment }

**[AgentV-RL: Scaling Reward Modeling with Agentic Verifier](llm_alignment/agentv-rl_scaling_reward_modeling_with_agentic_verifier.md)**

:   把奖励模型从"单轮打分"重塑为"前向+后向双 agent + 工具调用"的多轮审议流程，并通过 SFT+GRPO 把多 agent 能力蒸馏到单个 4B 模型中，使其在 BoN 选择上比 70B 量级 ORM 高 25.2%。

**[Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](llm_alignment/aligning_agents_via_planning_a_benchmark_for_trajectory-level_reward_modeling.md)**

:   提出 Plan-RewardBench，一个面向复杂工具增强场景的轨迹级偏好基准，用于评估奖励模型在多步规划、工具使用和错误恢复等场景下区分优劣智能体轨迹的能力。

**[Alignment Data Map for Efficient Preference Data Selection and Diagnosis](llm_alignment/alignment_data_map_for_efficient_preference_data_selection_and_diagnosis.md)**

:   提出 Alignment Data Map，一个通过联合考量回复质量(quality)和回复变异性(variability)来可视化、选择和诊断偏好数据的分析工具，仅用 33% 数据即可达到全量训练的对齐效果。

**[ARES: Adaptive Red-Teaming and End-to-End Repair of Policy-Reward System](llm_alignment/ares_adaptive_red-teaming_and_end-to-end_repair_of_policy-reward_system.md)**

:   ARES 通过一个能动态组合「话题 / 人设 / 目标 / 战术」四元结构的 Safety Mentor 同时探测 Core LLM 和 Reward Model 的「系统性弱点」（两者同时失守），然后用先修 RM 再修策略的两阶段闭环把 RedTeam 安全率从 0.28 提到 0.96，几乎不损失通用能力。

**[BACH-V: Bridging Abstract and Concrete Human-Values in Large Language Models](llm_alignment/bach-v_bridging_abstract_and_concrete_human-values_in_large_language_models.md)**

:   本文提出 abstraction-grounding 框架，把 LLM 的概念理解拆成"抽象-抽象 / 抽象-具体 / 具体-具体"三层，并用概念探针 + 激活引导在 6 个开源 LLM、10 个价值维度上证明：LLM 内部确实存在结构化的价值表示，能跨抽象层迁移、并因果地驱动具体决策。

**[Better Literary Translation: A Multi-Aspect Data Generation and LLM Training Approach](llm_alignment/better_literary_translation_a_multi-aspect_data_generation_and_llm_training_appr.md)**

:   这篇论文把文学翻译质量拆成“表达流畅”和“文学效果”两个维度，用专门 LLM 反复生成高质量参考译文和偏好对，再用 SFT + 显式奖励模型 + GRPO 训练 LitMT，使 8B/14B 小模型在英译中文学翻译上接近甚至超过部分大模型。

**[ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training](llm_alignment/consistrm_improving_generative_reward_models_via_consistency-aware_self-training.md)**

:   ConsistRM 提出基于一致性感知的自训练框架，通过时序一致性伪标签（融合在线状态和历史记忆的偏好一致性）和语义一致性批评奖励（衡量多次生成批评的语义相似度）两个模块，在无需人工标注的条件下将生成式奖励模型的五个基准平均性能提升 1.5%，同时显著缓解了位置偏差问题。

**[Debiasing Reward Models via Causally Motivated Inference-Time Intervention](llm_alignment/debiasing_reward_models_via_causally_motivated_inference-time_intervention.md)**

:   作者把 Bradley-Terry reward model 视作估计 total effect 的因果图，识别出与五类风格性偏差（长度 / 段落 / 词重叠 / 感叹号 / 粗体）激活高度相关的 bias-specific neurons（占总神经元 < 2%），在推理时把这些神经元激活替换为验证集中位数（估计 controlled direct effect），在 RewardBench / RM-Bench 上既不掉点又消除偏差，DPO 下游使用后让 8B 模型的对齐分数追平 70B SOTA reward model。

**[How Value Induction Reshapes LLM Behaviour](llm_alignment/how_value_induction_reshapes_llm_behaviour.md)**

:   本文用价值标注后的偏好数据子集对 8 个开源 LLM (3 系) × 15 个价值做 DPO 微调，发现价值之间存在系统性串扰 —— 诱导一个值会同时强化或抑制其他相关 / 对立值，正面价值能提升安全性但所有价值都会让模型更"拟人化"，使输出更易被感知为阿谀。

**[Large Language Models Are Overconfident in Their Own Responses](llm_alignment/large_language_models_are_overconfident_in_their_own_responses.md)**

:   这篇论文发现 instruction-tuned LLM 在评估“自己给出的答案”时存在显著 ownership bias，并提出把答案改写成用户输入再询问置信度的简单推理时策略，可在无需重训的情况下降低过度自信。

**[MAESTRO: Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization](llm_alignment/maestro_meta-learning_adaptive_estimation_of_scalarization_trade-offs_for_reward.md)**

:   本文提出 MAESTRO，将 GRPO 中的奖励标量化重新定义为上下文老虎机问题，通过轻量级 Conductor 网络利用模型末层隐藏状态自适应地为每个 prompt-response 对选择奖励权重，在七个开放域基准上一致超越静态奖励和单一奖励基线。

**[MDP-GRPO: Stabilized Group Relative Policy Optimization for Multi-Constraint Instruction Following](llm_alignment/mdp-grpo_stabilized_group_relative_policy_optimization_for_multi-constraint_inst.md)**

:   MDP-GRPO 针对多约束指令遵循中 GRPO 在离散低方差奖励下的不稳定问题，结合多温度采样、双锚点 advantage、prospect-theoretic shaping 和非对称 KL，使小模型在 IFEval、FollowBench 和自建多约束测试集上获得更稳的软/硬约束满足率。

**[Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](llm_alignment/mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)**

:   作者发现标准 GRPO 把同一题目的不同选项顺序当成独立 prompt 训练，导致模型在"换顺序"后选择会变化（permutation-blindness），于是提出 PA-GRPO：把同一语义实例的多种排列组成 permutation group，用跨排列的 advantage baseline + 一致性 reward 显式优化"换顺序不换选择"，在 7 个 MCQ/Judge benchmark 上把 selection bias 大幅压低同时保持准确率。

**[ModeX: Evaluator-Free Best-of-N Selection for Open-Ended Generation](llm_alignment/modex_evaluator-free_best-of-n_selection_for_open-ended_generation.md)**

:   把开放式文本生成的 Best-of-N 选样建模为「在生成文本的相似度图上找模态簇」的问题——用 n-gram Jaccard 构图 + 递归 Fiedler 向量谱聚类 + 中心度选 centroid，不需要任何 reward model / LLM-judge 就把 self-consistency 推广到摘要 / 代码 / 数学等无标准答案的任务。

**[On the Rejection Criterion for Proxy-Based Test-Time Alignment](llm_alignment/on_the_rejection_criterion_for_proxy-based_test-time_alignment.md)**

:   这篇论文把隐式奖励、Nudging 和 KAD 等 proxy-based test-time alignment 方法统一成一个“先采样再决定是否拒绝”的概率图模型，并提出用小对齐模型的最佳置信度作为参照的 conservative confidence bet，在多个数学与常识推理数据集上提升了混合解码精度。

**[P-Check: Advancing Personalized Reward Model via Learning to Generate Dynamic Checklist](llm_alignment/p-check_advancing_personalized_reward_model_via_learning_to_generate_dynamic_che.md)**

:   P-Check 把个性化奖励建模从“给用户历史塞进 judge”改成“先为当前用户和当前问题生成带权重的动态评价 checklist，再用它指导奖励打分”，在 PRISM、Arena、BESPOKE 的个性化偏好预测和下游生成任务上都显著优于 persona、记忆检索和微调奖励模型基线。

**[PERSA: Reinforcement Learning for Professor-Style Personalized Feedback with LLMs](llm_alignment/persa_reinforcement_learning_for_professor-style_personalized_feedback_with_llms.md)**

:   PERSA 用“教授示范 + 教授偏好奖励 + 只更新高层 LoRA 的 PPO”把通用 LLM 调成特定教师的编程反馈风格，在 APPS、PyFiXV、CodeReviewQA 上显著提升风格一致性，同时基本保持 100% 诊断正确率。

**[Pref-CTRL: Preference Driven LLM Alignment using Representation Editing](llm_alignment/pref-ctrl_preference_driven_llm_alignment_using_representation_editing.md)**

:   Pref-CTRL 在不更新大模型参数的测试时对齐框架 RE-Control 上，引入面向成对偏好数据的 margin loss 和 regularizer loss 来训练轻量价值函数，使表示编辑更符合人类偏好并在 SHP、HH-RLHF 与跨域数据上稳定优于 RE-Control。

**[RbtAct: Rebuttal as Supervision for Actionable Review Feedback Generation](llm_alignment/rbtact_rebuttal_as_supervision_for_actionable_review_feedback_generation.md)**

:   RbtAct 把作者 rebuttal 视为“哪些评审意见真的促成修改”的隐式监督，构建 7.5 万条 review-rebuttal 段级映射，并用 SFT+DPO 训练 8B 模型生成更具体、更可执行的论文评审反馈。

**[S2H-DPO: Hardness-Aware Preference Optimization for Vision-Language Models](llm_alignment/s2h-dpo_hardness-aware_preference_optimization_for_vision-language_models.md)**

:   提出 Simple-to-Hard（S2H）DPO 框架，通过构建三个递进难度级别的多图偏好数据（定点推理→跨图比较→全局视觉搜索），系统性地提升 VLM 的多图推理能力，同时保持单图性能。

**[SFTMix: Elevating Language Model Instruction Tuning with Mixup Recipe](llm_alignment/sftmix_elevating_language_model_instruction_tuning_with_mixup_recipe.md)**

:   本文提出 SFTMix，一种基于 Mixup 的指令微调方法，通过训练动态将 SFT 数据集分为高置信度和低置信度子集，在隐表示空间对两者进行线性插值并施加 Mixup 正则化，在不依赖高质量数据集的情况下，跨 LLM 家族和数据集规模一致性地提升指令遵循能力。

**[Student Guides Teacher: Weak-to-Strong Inference via Spectral Orthogonal Exploration](llm_alignment/student_guides_teacher_weak-to-strong_inference_via_spectral_orthogonal_explorat.md)**

:   本文把 LLM 在难题上反复沿同一错误逻辑采样的现象解释为隐藏状态低秩塌缩，并提出 Spectral Orthogonal Exploration (SOE)：用弱学生模型提供与教师当前 dominant subspace 正交的短 probe，迫使教师跳出原有 bias manifold，在 AIME/MATH/Olympiad 等难题子集上把 Pass@16 平均从 26.7% 提升到 45.9%。

**[Taming Extreme Tokens: Covariance-Aware GRPO with Gaussian-Kernel Advantage Reweighting](llm_alignment/taming_extreme_tokens_covariance-aware_grpo_with_gaussian-kernel_advantage_rewei.md)**

:   这篇论文把 GRPO 训练中的熵不稳定归因于少量极端 token 的“log 概率-优势”协方差贡献，并用无额外超参的高斯核对这些 token 的 advantage 做软抑制，从而在 1.5B 和 7B 数学推理模型上稳定提升性能。

**[Teaching LLM to be Persuasive: Reward-Enhanced Policy Optimization for Alignment from Heterogeneous Rewards](llm_alignment/teaching_llm_to_be_persuasive_reward-enhanced_policy_optimization_for_alignment_.md)**

:   这篇论文面向在线旅游平台的酒店降价谈判场景，提出 REPO 用偏好奖励模型、LLM 评审和规则函数三类奖励共同训练 Qwen3-32B，使模型在专家评价和 9653 场真实 A/B 对话中同时提升说服力、SOP 合规和坏例修复质量。

**[Team-Based Self-Play With Dual Adaptive Weighting for Fine-Tuning LLMs](llm_alignment/team-based_self-play_with_dual_adaptive_weighting_for_fine-tuning_llms.md)**

:   TPAW 将 LLM 自训练改造成“当前模型与历史模型组队博弈”的对齐过程，并用目标响应权重与主玩家权重两套自适应机制稳定偏好优化，在不额外引入人工偏好标注的情况下提升 Open LLM Leaderboard 与 GSM8K 表现。

**[Too Correct to Learn: Reinforcement Learning on Saturated Reasoning Data](llm_alignment/too_correct_to_learn_reinforcement_learning_on_saturated_reasoning_data.md)**

:   这篇论文指出强推理模型在“太容易、几乎全对”的训练集上做 GRPO 会因为组内奖励方差消失而停止学习，并提出 Mixed-CUTS 用受限 Top-K 均匀采样混合标准 rollout，重新制造有意义的探索差异，在 Qwen3-4B 上把 AIME25 Pass@1 相比标准 GRPO 提高 15.1%。

**[Topology-Enhanced Alignment for Large Language Models: Trajectory Topology Loss and Topological Preference Optimization](llm_alignment/topology-enhanced_alignment_for_large_language_models_trajectory_topology_loss_a.md)**

:   这篇论文把 LLM 对齐看成隐藏空间中的“语义轨迹”塑形问题，在 SFT 阶段用 0 维持久同调抽取 prompt-answer 拓扑桥并加入 TTL，在 DPO 阶段用主题偏好方向加入 TPO，使 UltraChat 和 HH-RLHF 上的奖励、胜率和无害性指标都稳定优于非拓扑基线。

**[Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms](llm_alignment/towards_bridging_the_reward-generation_gap_in_direct_alignment_algorithms.md)**

:   本文识别了直接对齐算法（DAAs）中的"奖励-生成鸿沟"——训练目标与自回归解码动态之间的不匹配，提出 POET（Prefix-Oriented Equal-length Training），通过将偏好响应对截断为较短者长度来隐式约束 token 级 MDP 在所有时间步上收敛，在 AlpacaEval 2 上最高提升 11.8 个百分点。

**[What Makes Good Instruction-Tuning Data? An In-Context Learning Perspective](llm_alignment/what_makes_good_instruction-tuning_data_an_in-context_learning_perspective.md)**

:   本文提出 weighted In-Context Influence (wICI)，用候选样本作为 one-shot demonstration 后能否降低相关困难 probe 的 instruction-following difficulty 来衡量指令数据价值，在 10% 数据预算下优于或匹配 IFD、DEITA、NUGGETS、SelectIT 等选择方法。

**[Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models](llm_alignment/why_supervised_fine-tuning_fails_to_learn_a_systematic_study_of_incomplete_learn.md)**

:   本文首次系统研究了 SFT 中的"不完全学习现象"（ILP）——即模型收敛后仍无法正确复现部分训练数据，识别了五种反复出现的原因（知识缺失、知识冲突、数据内部矛盾、左侧遗忘、不充分优化），并提出诊断框架和针对性缓解策略。

**[WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback](llm_alignment/wildfeedback_aligning_llms_with_in-situ_user_interactions_and_feedback.md)**

:   WildFeedback 从真实用户与 ChatGPT 的多轮对话中自动识别满意/不满意反馈，把自然发生的用户偏好转成偏好训练样本和逐例 checklist 评估标准，使小型开源指令模型在通用 benchmark 与真实用户偏好测试上都比 UltraFeedback 训练更贴近用户需求。

---

## 🗣️ 对话系统 { #dialogue }

**[APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI](dialogue/apex-mem_agentic_semi-structured_memory_with_temporal_reasoning_for_long-term_co.md)**

:   把对话长期记忆建成"领域无关本体支撑的属性图 + 只追加事件存储 + ReAct 多工具检索代理"三件套——构建时永不覆盖、检索时再做时序冲突解析，在 LOCOMO 拿到 88.88%（比 MIRIX 高 3.5 个点）、LongMemEval 拿到 86.2%（比最强 RAG baseline 高 13.7 个点）。

**[Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review](dialogue/author-in-the-loop_response_generation_and_evaluation_integrating_author_experti.md)**

:   本文将学术论文作者回复（rebuttal）生成重新定义为"作者在回路"任务，提出 Re3Align 数据集（3.4K 论文、440K 句级编辑标注、15K 审稿-回复-修改三元组）、REspGen 可控生成框架和 REspEval 20+ 指标评估套件，在 5 个 SOTA LLM 上系统验证了作者输入、可控性和评估引导精修的效果。

**[Codebook-Injected Dialogue Segmentation for Multi-Utterance Constructs Annotation: LLM-Assisted and Gold-Label-Free Evaluation](dialogue/codebook-injected_dialogue_segmentation_for_multi-utterance_constructs_annotatio.md)**

:   论文把 dialogue act 标注重新定义为"先分段、再贴标签"的两步问题，提出 codebook-injected 的 LLM 分割（System 1）和 Dial-Start 的 DA-aware 检索增强（System 2）两种方案，并给出无需 gold boundary 的三类评测指标（segment 内一致性 / 相邻段差异性 / 人-AI 分布对齐），在 TalkMoves 和 CLASS-annotated 两套教学对话上证明：DA-aware 提示能让 LLM 切出更同质的 segment，但和 coherence-based baseline 各占不同评测维度，没有单一最优。

**[CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](dialogue/codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)**

:   本文提出 CoDial，一个将预定义的对话流（task schema）转换为结构化异构图再自动生成 LLM 护栏代码（如 Colang）的框架，在推理阶段实现可解释且可控的任务型对话策略，在 STAR 基准上达到 SOTA，且无需训练数据。

**[Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](dialogue/cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)**

:   提出CoPoLLM框架，通过构建首个带认知扭曲标注的情感支持对话数据集CogBiasESC，结合认知策略强化学习（CPRL）引擎和双流条件优化（DSCO），使LLM能诊断8类认知扭曲并生成策略感知的干预回复，在15个SOTA基线上全面领先。

**[Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue](dialogue/context-agent_dynamic_discourse_trees_for_non-linear_dialogue.md)**

:   作者提出 Context-Agent，把多轮对话历史建模为"话题树森林"（每棵树代表一个独立话题、每条分支代表一次指令细化/分叉），按导航意图而非语义相似度组织节点，并配套提出 NTM 基准评测非线性长程对话，在多种 LLM 上同时提升任务完成率并降低 token 消耗。

**[Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky](dialogue/disambiguation-centric_finetuning_makes_enterprise_tool-calling_llms_more_realis.md)**

:   提出 DiaFORGE 框架，通过消歧中心的合成数据生成管线 + 推理链微调 + 动态评估体系，让开源 LLM 在面对近重复企业 API 时的工具调用成功率比 GPT-4o 高 27 个百分点、比 Claude-3.5-Sonnet 高 49 个百分点。

**[Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation](dialogue/discourse_coherence_and_response-guided_context_rewriting_for_multi-party_dialog.md)**

:   本文提出 DRCR，首个将上下文改写引入多方对话生成的框架，使用话语连贯性和回复质量双反馈信号构建偏好数据，通过动态自演化学习让改写器和回复器在迭代训练中相互增强。

**[Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents](dialogue/dual_hierarchical_dialogue_policy_learning_for_legal_inquisitive_conversational_.md)**

:   作者把"美国最高法院法官审律师"这种"AI 主动提问、对方未必合作"的对话定义为 Inquisitive Dialogue，提出 Dual Hierarchical RL 框架——一个 Appraisal Agent 实时打分律师回答（9 种 appraisal 类）、一个 Hierarchical Dialogue Agent 在三层（act/subtype/utterance）Poincaré 动作空间上做 DDQN 选 act，再叠加目标相关性/新颖性/简洁性三重 reward 与一个 conservative 正则项，在 Oyez Supreme Court 数据集上把 PES（探查有效性）从 baseline 的 4.22 推到 4.47，多轮 Coverage / MR 都最高。

**[ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](dialogue/ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)**

:   ETHICMIND 提出推理时（inference-time）的风险感知对齐框架，在多轮对话的每一轮中联合分析伦理风险和用户情感，规划高层响应策略，再生成兼顾伦理引导和情感共鸣的回复，无需额外训练即可在高风险和道德模糊场景中实现更一致的对齐表现。

**[Frame of Reference: Addressing the Challenges of Common Ground Representation in Dialogue](dialogue/frame_of_reference_addressing_the_challenges_of_common_ground_representation_in_.md)**

:   本文提出 IndiRef 基准测试，用于评估对话系统通过"关系指代"（如"昨天我们去的那个公园旁边的咖啡馆"）建立和利用持久共识（common ground）的能力，发现现有 LLM 在全上下文条件下准确率不超过 50%，并通过合成数据 + GRPO 强化学习训练将性能提升 15-20%。

**[GenesisFunc: Multi-Agent Data Generation for Accurate and Generalizable Function-Calling](dialogue/genesisfunc_multi-agent_data_generation_for_accurate_and_generalizable_function-.md)**

:   GenesisFunc 用可靠工具池、多代理对话生成和多阶段质检自动构造高质量函数调用训练数据，微调 Qwen3-8B 后在 BFCL、API-Bank 和 ACEBench 上超过同规模开源函数调用模型，并展示出向更多工具和多轮 RL 训练扩展的潜力。

**[LOCKET: Robust Feature-Locking Technique for Language Models](dialogue/locket_robust_feature-locking_technique_for_language_models.md)**

:   为 LLM "按特征付费解锁" 商业模式设计了一个不用密码、可扩展、抗越狱的 feature-locking 方案 LOCKET：每个待锁特征训一个 LoRA adapter（用 LAT 做对抗强化拒答），合并多个 adapter 时按层做频谱范数裁剪 (spectral norm clipping) 防止"过度拒答"塌缩，最终在 3 个模型 × 4 特征 (Math/SQL/Summarize/MMLU) 上拿到 100% 拒答率、≤7% 效用损失、≤5% 越狱攻击成功率，全面碾压 password-locking 基线。

**[MA$^2$P: A Meta-Cognitive Autonomous Intelligent Agents Framework for Complex Persuasion](dialogue/ma2p_a_meta-cognitive_autonomous_intelligent_agents_framework_for_complex_persua.md)**

:   MA$^2$P 将复杂说服对话拆成“元策略选择-任务级多智能体说服-事后知识更新”的闭环，在不训练底座 LLM 的情况下，把被说服者的信念、欲望和顾虑转成更具体的策略动作，并在 CToMPersu 上显著提升多种 LLM 的说服成功率。

**[Metro: Towards Strategy Induction from Expert Dialogue Transcripts for Non-collaborative Dialogues](dialogue/metro_towards_strategy_induction_from_expert_dialogue_transcripts_for_non-collab.md)**

:   Metro 把专家对话 transcript 自动归纳成一片 "Strategy Forest"——以 K-Means 聚类的对话状态为根节点的树，节点是 LLM 扩展的 micro-principle 动作、分支是按 Wilson 置信下界 + MCTS 风格价值回传打分剪枝后的完整动作轨迹，推理时直接 retrieve 一棵树、并行抽取 short-term（breadth）和 long-term（depth）建议，无需任何训练就在 P4G / CB 两个非合作对话任务上比 PRINCIPLES、PPDPP、GDP-Zero 等基线平均涨点约 10%。

**[Preference Learning Unlocks LLMs' Psycho-Counseling Skills](dialogue/preference_learning_unlocks_llms_psycho-counseling_skills.md)**

:   本文构建了面向心理咨询回复质量的 PsyCoPref 偏好数据集，并用奖励模型、DPO 与迭代式偏好学习训练 LLM，使 8B 模型在心理咨询回复上相对 GPT-4o 达到 87.0% 的胜率。

**[ReacTOD: Bounded Neuro-Symbolic Agentic NLU for Zero-Shot Dialogue State Tracking](dialogue/reactod_bounded_neuro-symbolic_agentic_nlu_for_zero-shot_dialogue_state_tracking.md)**

:   ReacTOD 将任务型对话状态跟踪拆成受限工具调用，并用确定性符号校验器拦截和反馈 LLM 错误，使 8B 到 32B 级模型也能在零样本 MultiWOZ 和 SGD 上获得强于先前大模型提示方法的 JGA。

**[Reasoning Gets Harder for LLMs Inside A Dialogue](dialogue/reasoning_gets_harder_for_llms_inside_a_dialogue.md)**

:   这篇论文提出 Boulder 动态基准，证明 LLM 在孤立推理题上表现很好，但同一问题嵌入任务型对话后会显著退化，主要原因来自多轮上下文、对话角色约束和工具调用负担。

**[Simulated Students in Tutoring Dialogues: Substance or Illusion?](dialogue/simulated_students_in_tutoring_dialogues_substance_or_illusion.md)**

:   这篇论文提出了一套面向数学辅导对话的模拟学生评测框架，发现简单 prompting 往往只会生成“看似会答题的学生”，而 SFT 和 DPO 更接近真实学生行为，但在错误复现和个体差异建模上仍然远未解决。

**[SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation](dialogue/spasm_stable_persona-driven_agent_simulation_for_multi-turn_dialogue_generation.md)**

:   本文提出 SPASM，一个以稳定性为核心的人设驱动多轮对话模拟框架，通过模块化人设生成、自我中心上下文投影（ECP）和终止检测三个组件，在 LLM-LLM 对话中大幅减少角色漂移和"回声"现象，构建了 45,000 段高质量多轮对话数据。

**[Stress-Testing Emotional Support Models: Moving from Homogeneous to Diverse Help Seekers](dialogue/stress-testing_emotional_support_models_moving_from_homogeneous_to_diverse_help_.md)**

:   这篇论文用 Reddit 情感支持对话构造九维求助者画像，并用带行为路由的 LoRA-MoE 训练可控求助者模拟器，从而让情感支持模型在更真实、更困难、更多样的人群上接受交互式压力测试。

**[STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems](dialogue/stride-ed_a_strategy-grounded_stepwise_reasoning_framework_for_empathetic_dialog.md)**

:   本文提出 STRIDE-ED 框架，通过构建覆盖正/中/负情绪的全面共情策略体系，设计任务对齐的多阶段认知CoT推理，结合策略感知数据精炼和SFT+PPO两阶段训练，在多个开源LLM上实现共情对话SOTA，情感准确率达57.25%，BLEU-4达4.67。

**[Surprisal Minimisation over Goal-directed Alternatives Predicts Production Choice in Dialogue](dialogue/surprisal_minimisation_over_goal-directed_alternatives_predicts_production_choic.md)**

:   这篇论文把自然对话中的话语生成建模为在上下文替代项中的成本敏感选择，并发现相对于“同一交际目标”的 goal-directed alternatives 最小化 surprisal，最能预测人类实际说出的 continuation。

**[Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](dialogue/template-assisted_contrastive_learning_of_task-oriented_dialogue_sentence_embedd.md)**

:   提出 TaDSE 框架，利用对话中现有的模板（template）信息作为辅助锚点，通过模板感知的数据增强、配对对比训练和语义压缩推理三个阶段，在无监督设置下显著提升任务型对话的句子嵌入质量，在五个基准上超越此前 SOTA 甚至优于有监督的商业嵌入模型。

**[Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation](dialogue/towards_proactive_information_probing_customer_service_chatbots_harvesting_value.md)**

:   本文提出 ProChatIP 框架，将客服聊天机器人从被动应答工具转变为主动信息采集引擎，通过专门的对话策略模块学习"何时探测"用户以获取预设的目标信息，同时最小化对话轮数和用户摩擦。

**[Your Students Don't Use LLMs Like You Wish They Did](dialogue/your_students_dont_use_llms_like_you_wish_they_did.md)**

:   这篇论文提出 6 个面向教育 AI 对话的可计算行为指标，并在 500 段真实学生-AI 对话中发现：学生常把本应促进学习的 LLM 工具用成答案提取器，且部署方式比系统设计或学生偏好更能决定这种错位。

---

## 🕸️ 图学习 { #graph_learning }

**[AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](graph_learning/agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)**

:   提出 AgentGL，首个基于强化学习的智能体图学习（AGL）框架，让 LLM 智能体通过图原生搜索工具自主导航文本属性图（TAG），在节点分类和链接预测任务上分别实现最高 17.5% 和 28.4% 的绝对准确率提升。

**[ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning](graph_learning/ark_answer-centric_retriever_tuning_via_kg-augmented_curriculum_learning.md)**

:   提出ARK框架，通过三维答案充分性评分（Forward+Backward+Retriever对齐）筛选正样本，利用LLM构建的知识图谱生成渐进难度的困难负样本进行课程对比学习，在10个数据集上平均提升14.5% F1。

**[Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval](graph_learning/autonomous_knowledge_graph_exploration_with_adaptive_breadth-depth_retrieval.md)**

:   本文提出 ARK：一个 training-free 的知识图谱检索 agent，只暴露「全局词法搜索」和「单跳邻居展开」两个最小工具，让 LLM 自主在广度和深度之间切换，无需种子节点或固定跳数；在 STaRK 三图上把 Hit@1 平均推到 59.1%，最高比 training-free baseline 提升 31.4%，并可把策略 label-free 蒸馏进 Qwen3-8B。

**[AutoPKG: An Automated Framework for Dynamic E-commerce Product-Attribute Knowledge Graph Construction](graph_learning/autopkg_an_automated_framework_for_dynamic_e-commerce_product-attribute_knowledg.md)**

:   提出 AutoPKG，一个多智能体 LLM 框架，从多模态电商商品内容自动构建 Product-Attribute 知识图谱（PKG），通过类型归纳 Agent、属性键发现 Agent、属性值提取 Agent 和集中式 KGD 决策 Agent 实现动态本体的持续演化和规范化，在 Lazada 数据集上取得 0.953 WKE（类型）和 0.724 WKE（属性键），线上 A/B 测试推荐 GMV 提升 7.89%。

**[CoG: Controllable Graph Reasoning via Relational Blueprints and Failure-Aware Refinement over Knowledge Graphs](graph_learning/cog_controllable_graph_reasoning_via_relational_blueprints_and_failure-aware_ref.md)**

:   CoG 是一个 training-free 的 KGQA 框架，把 Kahneman 的 Dual-Process Theory 落到 KG 推理上：System 1 离线把训练集 SPARQL 蒸馏成"关系蓝图"模板库，在线作为软结构约束指导 candidate relation 的 rerank 与剪枝；System 2 在搜索停滞时触发证据条件反思和定向回溯，纠正前期错误决策；在 CWQ / WebQSP / GrailQA 三个多跳 KGQA 基准上同时拿到 SOTA 准确率（GPT-4 backbone CWQ 77.8、WebQSP 89.7、GrailQA 86.4）和最低成本（CWQ 比 PoG 少 13% token、少 12% call）。

**[Collaboration of Fusion and Independence: Hypercomplex-driven Robust Multi-Modal Knowledge Graph Completion](graph_learning/collaboration_of_fusion_and_independence_hypercomplex-driven_robust_multi-modal_.md)**

:   M-Hyper 把多模态知识图谱实体编码为双四元数（biquaternion）的四个正交基，分别承载结构 / 视觉 / 文本三个独立模态以及一个融合模态，通过 Hamilton 乘积同时实现"模态独立保留"和"成对充分交互"，在 DB15K / MKG-W / MKG-Y 三个数据集上以最低显存、最短训练时间打败 18 个 baseline。

**[Comparing Human and Large Language Model Interpretation of Implicit Information](graph_learning/comparing_human_and_large_language_model_interpretation_of_implicit_information.md)**

:   本文提出隐含信息提取（IIE）任务和基于 LLM 的三阶段提取管道（信息提取→推理验证→时序分析），构建结构化知识图谱来表示文本的隐含含义，并通过众包人类判断对比发现 LLM 在社交丰富语境中比人类更保守，但在短事实语境中人类更保守。

**[ComplianceNLP: Knowledge-Graph-Augmented RAG for Multi-Framework Regulatory Gap Detection](graph_learning/compliancenlp_knowledge-graph-augmented_rag_for_multi-framework_regulatory_gap_d.md)**

:   ComplianceNLP 是一个端到端的金融监管合规系统，把 12,847 条 SEC / MiFID II / Basel III 法规构造成知识图谱来增强 RAG 检索，配合 LEGAL-BERT 的多任务义务抽取和门槛打分的差距分析，在 RegObligation / GapBench 上以 87.7 F1 击败 GPT-4o+RAG 3.5 个点，并通过领域知识蒸馏 + Medusa 推测解码实现 $2.8\times$ 推理加速；4 个月并行运行处理了 9,847 条更新，达到 96.0% 召回率和 3.1× 分析师效率提升。

**[CRAFTQA: A Code-Driven Adaptive Framework for Complex Structured Data Reasoning](graph_learning/craftqa_a_code-driven_adaptive_framework_for_complex_structured_data_reasoning.md)**

:   CRAFTQA 用 CodeSTEP 生成可执行的逐步 Python 推理代码，并在预定义操作不够时由 CRAFT 动态生成自定义函数，从而显著提升表格、知识图谱和时序知识图谱上的复杂结构化数据问答能力，GPT-4o 版本在复杂推理 Overall 上达到 76.6%。

**[EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment](graph_learning/ea-agent_a_structured_multi-step_reasoning_agent_for_entity_alignment.md)**

:   提出 EA-Agent，将实体对齐（EA）分解为结构化多步推理过程，通过工具池（三元组选择器+对齐工具+反思器）的规划和执行实现可解释的对齐决策，配合奖励引导的离线策略优化持续改进规划能力，在 DBP15K 上 Hits@1 提升高达 3.17%，同时减少冗余三元组带来的效率问题。

**[Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks](graph_learning/evaluating_llms_on_large-scale_graph_property_estimation_via_random_walks.md)**

:   现有 LLM 图推理 benchmark 只用 20–50 节点的小图、还要求全图可见；本文用「随机游走统计量」把最多 2.39M 节点的真实图压成 prompt，提出 EstGraph 评估 LLM 在节点/边数、社区数、图结构、影响节点 4 项估计任务上的表现，发现 LLM 在中等规模图上可达 < 20% 相对误差并能识别图结构。

**[From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](graph_learning/from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)**

:   本文提出 Gspell，一个轻量级后验解释框架，通过将 GNN 节点嵌入投影到 LLM 嵌入空间并构建混合提示（软提示+文本），使 LLM 能够直接推理 GNN 内部表示并生成自然语言解释和解释子图，在文本属性图（TAG）上实现了忠实性与可解释性的良好平衡。

**[Graph-Based Alternatives to LLMs for Human Simulation](graph_learning/graph-based_alternatives_to_llms_for_human_simulation.md)**

:   本文提出 GEMS（Graph-basEd Models for Human Simulation），将封闭式人类行为模拟任务建模为异构图上的链接预测问题，在三个数据集和三种评估设定下匹配或超越强 LLM 基线方法，同时参数量减少 3 个数量级。

**[GS-Quant: Granular Semantic and Generative Structural Quantization for Knowledge Graph Completion](graph_learning/gs-quant_granular_semantic_and_generative_structural_quantization_for_knowledge_.md)**

:   GS-Quant 把 KG 实体量化成"由粗到细"的离散 code 序列——用层次聚类树约束 RQ-VAE 让浅层 code 编码全局类别（如 "Person"）、深层 code 编码细粒度属性（如 "Artist"），再用 GPT-style decoder 重构 entity + ancestor 强制 code 之间产生因果依赖，最后把这些 code 加到 LLM 词表里做 LoRA 微调，在 WN18RR / FB15k-237 上 Hits@1 比 SOTA SSQR 高 2.2-2.4 个点。

**[IndustryAssetEQA: A Neurosymbolic Operational Intelligence System for Embodied Question Answering in Industrial Asset Maintenance](graph_learning/industryasseteqa_a_neurosymbolic_operational_intelligence_system_for_embodied_qu.md)**

:   本文把工业资产维护问答重新建模成"具身决策"任务，提出由 episode 化遥测、FMEA 知识图谱、参数化反事实风险模拟器、provenance 校验和安全门组成的神经符号系统 IndustryAssetEQA，在 4 个工业数据集上把结构有效性、反事实方向准确率、解释蕴含率分别提升最多 0.51 / 0.47 / 0.64，并把专家判定的严重过度断言从 28% 压到 2%。

**[LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning](graph_learning/legalgraphrag_multi-agent_graph_retrieval-augmented_generation_for_reliable_lega.md)**

:   LegalGraphRAG 用事实图、 ontology 图和规则图构成层级法律图谱，再用 Researcher-Auditor-Adjudicator 三代理流程完成检索、验证和裁决，在法律判决生成中提升准确性与证据可追溯性。

**[LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs](graph_learning/llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp.md)**

:   本文在六个关系抽取数据集上对比四个 LLM（7B-70B）和一个轻量级图解析器（124M参数），发现当文档的关系图平均边数超过约 18 条时，图解析器持续且显著优于 LLM，在最复杂的 ERFGC 数据集上 F1 差距达 13.2 个点，揭示了 LLM 在复杂语言图结构抽取上的根本局限。

**[LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval](graph_learning/logoskg_hardware-optimized_scalable_and_interpretable_knowledge_graph_retrieval.md)**

:   本文提出 LogosKG，一个硬件对齐的知识图谱检索框架，通过将图遍历转化为三元稀疏矩阵（SUB/OBJ/REL）的乘法运算，配合度感知图分区、跨图路由和按需缓存，在单设备上实现了对十亿边规模 KG 的可扩展、可解释高跳检索，并通过下游 KG-LLM 交互实验揭示了图拓扑结构对 LLM 诊断推理的影响。

**[MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation](graph_learning/megarag_multimodal_knowledge_graph-based_retrieval_augmented_generation.md)**

:   MegaRAG 利用 MLLM 在长文档每一页并行抽取实体-关系并合并为多模态知识图谱（MMKG），再用"子图引导"的二轮 refinement 补全跨模态、跨页关系，配合双路检索与两阶段答案生成显著优于 GraphRAG/LightRAG/VisRAG，在 SlideVQA(2k) 上准确率 64.85%（baseline 最高 27.66%）。

**[STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation](graph_learning/stem_structure-tracing_evidence_mining_for_knowledge_graphs-driven_retrieval-aug.md)**

:   STEM 将知识图谱多跳问答从逐步路径搜索改写为“先生成查询结构图、再按结构追踪证据子图”的问题，通过语义到结构投影、Triple-GNN 全局引导和结构匹配检索，在 WebQSP 与 CWQ 上显著提升 KG-RAG 的答案准确率和证据覆盖率。

**[TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation](graph_learning/tagrag_tag-guided_hierarchical_knowledge_graph_retrieval-augmented_generation.md)**

:   TagRAG 用“对象标签 + 领域标签链”替代 GraphRAG 中昂贵的实体社区划分和全图摘要，在保持全局知识整合能力的同时显著降低构图与检索成本，并在 UltraDomain 四个领域上以小模型 Qwen3-4B 获得高于 NaiveRAG、GraphRAG、LightRAG 和 MiniRAG 的胜率。

**[What Makes AI Research Replicable? Executable Knowledge Graphs as Scientific Knowledge Representations](graph_learning/what_makes_ai_research_replicable_executable_knowledge_graphs_as_scientific_know.md)**

:   本文提出 Executable Knowledge Graphs (xKG)，把论文中的技术概念和可运行代码片段组织成 Paper- Technique- Code 三层图结构，作为可插拔知识库辅助科研复现 agent，在 PaperBench Code-Dev 上为不同 agent 带来最高 10.90 个百分点的复制得分提升。

**[Which bird does not have wings: Negative-constrained KGQA with Schema-guided Semantic Matching and Self-directed Refinement](graph_learning/which_bird_does_not_have_wings_negative-constrained_kgqa_with_schema-guided_sema.md)**

:   本文提出了否定约束知识图谱问答（NEST KGQA）新任务和 NestKGQA 数据集，设计了 Python 格式逻辑形式 PyLF 来清晰表达否定约束，并提出 CUCKOO 框架通过约束感知草稿生成、Schema 引导语义匹配和自导向细化三个模块，在 few-shot 设置下实现了多约束问题的高效精确回答。

---

## ⚡ LLM 效率 { #llm_efficiency }

**[Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference](llm_efficiency/alloc-moe_budget-aware_expert_activation_allocation_for_efficient_mixture-of-exp.md)**

:   把 MoE 推理的"激活专家个数"抽象成全局预算 $B$，先用动态规划在层间做最优 Top-K 分配（Alloc-L），再用全局 Top-(K·T) 选择在 token 间重分配（Alloc-T），在 DeepSeek-V2-Lite 上把激活预算砍掉一半还能保持精度，prefill 加速 1.15×、decode 加速 1.34×。

**[Are Large Language Models Economically Viable for Industry Deployment?](llm_efficiency/are_large_language_models_economically_viable_for_industry_deployment.md)**

:   提出Edge-Eval框架，通过5个部署指标（经济盈亏平衡、智能功耗比、系统密度、冷启动税、量化保真度）在传统T4 GPU上全生命周期评估LLM，揭示<2B小模型在经济和生态维度全面优于7B模型，并发现QLoRA虽降低内存但能耗增加最高7倍的反常现象。

**[Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning](llm_efficiency/beyond_accuracy_unveiling_inefficiency_patterns_in_tool-integrated_reasoning.md)**

:   提出 PTE（Prefill Token Equivalents），一个基于硬件感知的工具集成推理效率度量指标，统一了内部推理和外部工具使用的成本，并通过大规模实验揭示了四种 TIR 低效模式：确认性工具使用、工具混合、缺乏工具先验和工具格式崩溃。

**[BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs](llm_efficiency/bosch_black-box_binary_optimization_for_short-context_attention-head_selection_i.md)**

:   提出 BOSCH，一种免训练的注意力头级别 SWA 混合方法，将 SWA 头选择建模为大邻域搜索问题并分解为三阶段优化（层重要性探测→自适应比例分配→分组头选择），在 4 个模型 4 种比例设置下系统性超越层级启发式和 6 种静态头级别方法。

**[Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](llm_efficiency/breaking_block_boundaries_anchor-based_history-stable_decoding_for_diffusion_lar.md)**

:   提出 AHD（Anchor-based History-stable Decoding），一种无需训练的即插即用动态解码策略，通过动态锚点回溯历史轨迹判定扩散LLM中跨块稳定token，实现早期解锁，在BBH上减少80%解码步数的同时提升3.67%性能。

**[CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling](llm_efficiency/comet_collaborative_memory_transformer_for_efficient_long_context_modeling.md)**

:   CoMeT 给已有 LLM 加一个"全局记忆 + FIFO 临时记忆"的双记忆插件，分块处理输入实现常数显存、线性时间复杂度，仅在 32k 上下文上微调就能在 1M token 内任意位置精确找回密码，并提出层级流水线并行让 16×80GB GPU 就能微调 128k 上下文。

**[CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](llm_efficiency/creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)**

:   本文提出 CreditDecoding，一种无需训练的并行解码加速方法，通过累积 token 级历史证据（轨迹信用）来增强正确但置信度不足的 token，在 LLaDA-8B-Instruct 上实现最高 5.48 倍加速且准确率提升 0.48。

**[Lizard: An Efficient Linearization Framework for Large Language Models](llm_efficiency/lizard_an_efficient_linearization_framework_for_large_language_models.md)**

:   Lizard 用一个"Gated Linear Attention（全局压缩）+ Anchor Window Attention（局部精度）+ 可学习 gate 替代 RoPE"的混合 subquadratic 注意力替换预训练 Transformer 的 softmax attention，只用 0.04B token 蒸馏就能在 5-shot MMLU 上把现有 linearization 方法甩开 9.4–24.5 分，并配套一个 tensor-core 友好的训练算法把吞吐量提升 32%。

**[MTRouter: Cost-Aware Multi-Turn LLM Routing with History-Model Joint Embeddings](llm_efficiency/mtrouter_cost-aware_multi-turn_llm_routing_with_history-model_joint_embeddings.md)**

:   MTRouter把多轮Agent中的“每一轮该调用哪个LLM”建模为成本约束下的逐轮路由问题，通过历史-模型联合嵌入预测候选模型对最终任务结果的贡献，在ScienceWorld和HLE上同时提升任务表现并显著降低总调用成本。

**[Multi-Drafter Speculative Decoding with Alignment Feedback](llm_efficiency/multi-drafter_speculative_decoding_with_alignment_feedback.md)**

:   本文提出 MetaSD，一个将多个异构草稿器整合到推测解码中的统一框架，将草稿器选择建模为多臂赌博机问题，通过块散度（Block Divergence）奖励信号动态选择与目标 LLM 最对齐的草稿器，在黑盒和白盒配置下一致优于单草稿器方法。

**[Native Hybrid Attention for Efficient Sequence Modeling](llm_efficiency/native_hybrid_attention_for_efficient_sequence_modeling.md)**

:   本文提出 Native Hybrid Attention (NHA)，将线性 RNN 的长期记忆槽与滑动窗口的短期精确 token 拼接后通过单次 softmax 注意力统一处理，实现层内和层间混合的原生统一——无需额外融合参数即可动态分配长短期注意力权重，在 recall 密集和常识推理任务上超越 Transformer 和其他混合基线。

**[RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding](llm_efficiency/racer_retrieval-augmented_contextual_rapid_speculative_decoding.md)**

:   RACER 提出了一种无需训练的推测解码方法，将基于检索的精确模式匹配与基于 logits 的未来预测统一起来，通过 copy-logit 策略构建 Logits Tree、LRU 驱逐的 AC 自动机构建 Retrieval Tree，在多个基准上实现了超过 2 倍的推理加速。

**[Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for DLMs](llm_efficiency/saber_an_efficient_sampling_with_adaptive_acceleration_and_backtracking_enhanced.md)**

:   本文提出 Saber，一个面向扩散语言模型（DLM）的免训练采样算法，通过自适应加速（根据已建立的上下文动态调整并行解码量）和回溯增强重遮蔽（撤销被新上下文证伪的 token）两种策略，在代码生成上平均提升 Pass@1 1.9% 的同时实现 251.4% 的推理加速。

**[SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration](llm_efficiency/specbound_adaptive_bounded_self-speculation_with_layer-wise_confidence_calibrati.md)**

:   提出 SpecBound 自草稿推测解码框架，通过逐层温度退火抑制浅层虚假高置信度预测，并设计有界推测算法自适应控制草稿的深度和宽度，在保持输出无损的同时实现最高 2.33× 的推理加速。

**[Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](llm_efficiency/speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)**

:   提出推测验证（Speculative Verification, SV），通过引入与草稿模型同等规模的伴随模型（companion model），利用草稿-伴随分布的相似性预测推测准确率，动态调整验证长度以最大化有效吞吐量，在大批量推理中实现相对标准推测解码平均1.4×、最高1.9×的加速。

**[StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference](llm_efficiency/structkv_preserving_the_structural_skeleton_for_scalable_long-context_inference.md)**

:   本文提出 StructKV，一个结构感知的 KV Cache 压缩框架，通过全局入度中心性（Global In-Degree Centrality）跨层累积注意力模式识别全局信息枢纽，动态枢纽层检测（Dynamic Pivot Detection）自适应定位最优压缩层，以及结构传播与解耦（Structural Propagation & Decoupling）分离计算预算和存储预算，在 LongBench 和 RULER 上以 60% prefill + 10% KV 实现了接近全上下文的性能。

**[Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning](llm_efficiency/tandem_riding_together_with_large_and_small_language_models_for_efficient_reason.md)**

:   Tandem 让大模型只生成 Goal / Planning / Retrieval / Action 四类短思维线索，再由小模型用困惑度和熵判断线索是否足够并完成答案，在 MATH、GSM8K 和 HumanEval 上用约 60% 的计算成本达到或超过单独大模型推理效果。

**[Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](llm_efficiency/task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)**

:   提出多层级任务画像引导的数据合成框架解决 LLM 路由的冷启动问题，并设计 TRouter——一种将任务类型作为隐变量的路由方法，通过变分推断建模查询-成本-性能关系，在冷启动和域内设置下均实现有效路由。

**[专业化的幻觉：揭示混合专家模型中的"常设委员会"](llm_efficiency/the_illusion_of_specialization_unveiling_the_domain-invariant_34standing_committ.md)**

:   通过引入 CommitteeAudit 框架，作者发现 MoE 模型中存在一个"常设委员会"——一个紧凑的、持久的专家组合，在不同领域始终被激活并占据大部分路由权重，这与广泛假设的领域特定专业化形成鲜明对比，揭示了稀疏计算内在的集中化结构。

**[阈值差分注意力：无 Sink、超稀疏且非分散的长上下文注意力](llm_efficiency/threshold_differential_attention_for_sink-free_ultra-sparse_and_non-dispersive_l.md)**

:   TDA 通过结合长度自适应阈值和差分抑制视图，实现无注意力 Sink、99% 精确稀疏、且性能竞争力的长上下文 Transformer 注意力。

**[TokenTiming: A Dynamic Alignment Method for Universal Speculative Decoding Model Pairs](llm_efficiency/tokentiming_a_dynamic_alignment_method_for_universal_speculative_decoding_model_.md)**

:   TokenTiming 把草稿模型生成的 token 序列重新编码到目标 tokenizer 空间，再用动态时间规整构造多对多 token 对齐，从而让不同词表的现成小模型也能作为投机解码 draft model，并在多个 14B-70B 目标模型上取得最高 1.57x 的异构投机解码加速。

**[Understanding LLM Performance Degradation in Multi-Instance Processing: The Roles of Instance Count and Context Length](llm_efficiency/understanding_llm_performance_degradation_in_multi-instance_processing_the_roles.md)**

:   这篇论文系统评测 16 个 LLM 在 multi-instance processing 中的退化规律，发现性能下降不只是上下文变长造成的，实例数量本身对成功率的影响更强，尤其在 1,000 个以上实例时几乎所有模型都会崩溃且很少主动提醒用户。

---

## 🎁 推荐系统 { #recommender }

**[Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders](recommender/bridging_language_and_items_for_retrieval_and_recommendation_benchmarking_llms_a.md)**

:   本文发布 Amazon Reviews 2023 大规模数据集（570M reviews / 48M items）并基于它构建 BLaIR 基准，覆盖序列推荐 / 协同过滤 / 商品搜索 (短 query + 复杂 query) 三大场景，benchmark 了 11 个顶尖 LLM 作为语义编码器，发现它们在 BLaIR 上的排名与 MTEB 几乎不相关（Spearman -0.476），并指出推荐场景对语义编码器有独特要求。

**[ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation](recommender/clusterrag_cluster-based_collaborative_filtering_for_personalized_retrieval-augm.md)**

:   ClusterRAG 把协同过滤引入个性化 RAG：先用用户历史文档构建用户表示并用 HDBSCAN 聚类，再从目标用户和相似用户中分层检索 profile 文档组成 prompt，在 LaMP 多任务基准上使 hybrid 模式全面优于 vanillaRAG、LaMP-IPA、ROPG 和 CFRAG。

**[Culinary Crossroads: A RAG Framework for Enhancing Diversity in Cross-Cultural Recipe Adaptation](recommender/culinary_crossroads_a_rag_framework_for_enhancing_diversity_in_cross-cultural_re.md)**

:   作者发现标准 RAG 在创意任务上"给了多样上下文也产出不多样"，于是设计 plug-and-play 的 CARRIAGE：查询重写 + diversity-aware MMR 重排 + sliding-window 动态上下文 + 对比性上下文注入，把"上下文多样性"真正传导到"输出多样性"，在西班牙语跨国菜谱适配上同时改善 lexical/semantic/ingredient diversity 与 CultureScore，对 closed-book LLM 达到 Pareto efficiency。

**[Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents](recommender/decisive_guiding_user_decisions_with_optimal_preference_elicitation_from_unstruc.md)**

:   提出 DECISIVE 交互式决策框架，通过从非结构化文档中提取客观选项评分矩阵，结合贝叶斯偏好推断自适应选择成对比较问题高效学习用户潜在偏好向量，在最小化用户交互负担的同时实现透明个性化推荐，决策准确率比强基线提升最高 20%。

**[From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation](recommender/from_past_to_path_masked_history_learning_for_next-item_prediction_in_generative.md)**

:   提出掩码历史学习（MHL）训练框架，通过在生成式推荐的自回归训练中加入掩码历史重建辅助任务，结合熵引导的自适应掩码策略和课程学习调度器，使模型从仅预测"下一个是什么"转向理解"为什么形成这条路径"，在三个数据集上显著超越SOTA。

**[From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents](recommender/from_recall_to_forgetting_benchmarking_long-term_memory_for_personalized_agents.md)**

:   本文提出Memora基准和FAMA指标，将长期记忆评估从浅层事实检索扩展到跨越数周至数月的记忆整合与突变处理，揭示现有LLM和记忆agent在处理频繁知识更新时的系统性失败。

**[HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](recommender/harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)**

:   提出 HARPO 框架，将对话推荐重新定义为以推荐质量为优化目标的结构化决策问题，通过层次化偏好学习、基于价值网络的树搜索推理、虚拟工具操作和多智能体精炼四大组件，在 ReDial、INSPIRED 和 MUSE 三个基准上显著超越现有方法。

**[HORIZON: A Benchmark for in-the-wild User Behaviour Modeling](recommender/horizon_a_benchmark_for_in-the-wild_user_behaviour_modeling.md)**

:   本文提出 HORIZON，首个全开源的大规模跨领域长期推荐基准，基于 Amazon Reviews 合并构建包含 54M 用户和 35M 商品的统一交互历史，设计了沿时间轴和用户维度解耦的四象限评估协议，揭示了 BERT4Rec 等模型在分布内表现强劲但在时序外推和未见用户场景下显著退化的现象，且 LLM 在用户行为建模上并未一致优于专用架构。

**[HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment](recommender/hsuga_llm-enhanced_recommendation_with_hierarchical_semantic_understanding_and_g.md)**

:   HSUGA 把 LLM 增强序列推荐的两个核心环节拆开来打补丁：用"阶段式 + 四类原子编辑（Add/Delete/Update/Retain）"的 HSU 模块把长交互序列的语义抽取做稳，再用按活跃度分组（20% 头部 / 80% 长尾）的 GAA 自蒸馏对齐解决长尾用户欠监督、活跃用户过对齐的问题，在 Steam/Fashion/Beauty 三个数据集 + GRU4Rec/BERT4Rec/SASRec 三个 backbone 上即插即用都涨点。

**[IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters](recommender/icebreaker_for_conversational_agents_breaking_the_first-message_barrier_with_per.md)**

:   本文提出 IceBreaker，通过两步"握手"——共鸣感知兴趣蒸馏捕获触发兴趣 + 交互导向启动语生成配合个性化偏好对齐——解决对话智能体的"首条消息壁垒"，在全球最大对话产品之一的 A/B 测试中提升用户活跃天数 +1.84‰ 和点击率 +94.25‰。

**[Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation](recommender/intent-driven_semantic_id_generation_for_grounded_conversational_news_recommenda.md)**

:   本文提出 NewsRec-Chat，把对话式新闻推荐从"先检索再生成"反转为"先生成 SID 再模糊匹配"，靠两阶段 SID 对齐 + GPT-4 CoT 蒸馏让 7B 模型直接生成层级 Semantic ID 前缀并与当日新闻池模糊匹配，腾讯新闻平台上 152K 开放生成空间里取得 12.4% L1（4× 随机）、0% 幻觉，并通过 Profile-Aware Dual-Signal Reasoning 让 0 历史用户达到 18.0% L1（其他基线 0%）。

**[Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](recommender/learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)**

:   本文提出 ReCAP 框架，通过可训练的查询生成器和用户画像生成器，从用户历史记录中检索与说服相关的信息并构建上下文感知的用户画像，显著提升个性化说服力预测的效果。

**[MemRec: Collaborative Memory-Augmented Agentic Recommender System](recommender/memrec_collaborative_memory-augmented_agentic_recommender_system.md)**

:   MemRec 用一个**轻量级 LLM 专门管理一张动态"协同记忆图"**（把多个 user 与 item 的语义记忆通过交互边相连），然后把蒸馏后的"协同记忆面（facets）"喂给重量级推理 LLM 做最终推荐；通过"Curate-then-Synthesize"压噪 + 异步 $O(1)$ 标签传播更新，在 4 个 benchmark 上 H@1 相对 SOTA i2Agent 提升 **+15% 到 +29%**，数据稀疏用户上更是相对 Vanilla LLM 提升 **+91.4%**。

**[Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation](recommender/mirroring_users_towards_building_preference-aligned_user_simulator_with_user_fee.md)**

:   作者把推荐系统里的"用户反馈日志"重写成一个 LLM 能理解的"用户记忆 + 曝光列表"统一仿真场景，再用 EKB 消费者决策模型生成显式的 chain-of-thought 决策过程作为"clarification"，通过不确定性分解 + 拒绝采样蒸馏出 10K 高质量 SFT/DPO 数据，让 3B 的 Llama 用户模拟器在 8 个领域的真实用户行为预测上超过 GPT-5 和 Gemini-2.5-Flash。

**[Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework](recommender/personalizing_llms_with_binary_feedback_a_preference-corrected_optimization_fram.md)**

:   这篇论文提出 C-BPO，把目标用户历史当作正反馈、其他用户历史当作带噪未标注负反馈，并用 PU 学习校正“偏好重叠”带来的误惩罚，从而让 LLM 学到用户独特偏好而不压制通用任务能力。

**[Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion](recommender/quality_over_clicks_intrinsic_quality-driven_iterative_reinforcement_learning_fo.md)**

:   提出 Cold-EQS，一个面向冷启动电商场景的查询建议框架，利用可回答性、事实准确性和信息增益作为内在质量奖励，通过迭代强化学习持续优化查询建议质量，在线 chatUV 提升 6.81%。

**[ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning](recommender/rerec_reasoning-augmented_llm-based_recommendation_assistant_via_reinforcement_f.md)**

:   本文提出 ReRec，一个基于强化微调（RFT）的推荐助手框架，通过双图增强的奖励塑形提供细粒度奖励信号、推理感知的优势估计对推理步骤进行差异化监督、以及在线课程调度器动态调整训练难度，使 LLM 能处理复杂的多步推理推荐查询，在 RecBench+ 基准上显著超越现有方法。

**[SenseJudge: Human-Centric Preference-Driven Judgment Framework](recommender/sensejudge_human-centric_preference-driven_judgment_framework.md)**

:   提出 SenseJudge，一种基于显式人类偏好的可定制化 LLM 判断框架，配合真实多轮对话基准 SenseBench，在个性化评判任务中平均准确率比基线高 16.08%，模型排名与真实人类排名一致。

**[What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty](recommender/what_makes_an_ideal_quote_recommending_34unexpected_yet_rational34_quotations_vi.md)**

:   NOVELQR 提出了一个新颖性驱动的引用推荐框架，通过生成式标签代理构建深层语义知识库实现语义理性检索，并用 token 级新颖性估计器缓解自回归续写偏差，在双语基准上显著提升推荐质量。

**[What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context](recommender/what_makes_llms_effective_sequential_recommenders_a_study_on_preference_intensit.md)**

:   本文揭示现有 LLM 推荐系统的二元偏好建模丢失了偏好强度和时间上下文两个关键信息，提出 RecPO 框架通过自适应奖励边际将这两个因素纳入偏好优化，在五个数据集上显著超越 S-DPO 等基线。

**[Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](recommender/where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)**

:   SiPeR 通过场景转换估计（"Where"）和贝叶斯逆推理（"What"）两个机制，解决情景对话推荐中用户偏好随环境动态变化且常常隐式表达的挑战，在 SIMMC 2.1 和 SCREEN 上分别提升 10.9% 和 10.6%。

---

## ✍️ 文本生成 { #nlp_generation }

**[Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search](nlp_generation/adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl.md)**

:   本文提出 PACO，把"多属性可控摘要"重新表述为一个寻找"属性控制顺序"的规划问题，并用一个定制的 Monte Carlo Tree Search（节点是完整摘要、动作是单属性调整）在 prompt 阶段就找到最优调整路径，无需任何属性专用训练，用 Llama-3.2-1B 即可达到 Llama-3.3-70B baseline 的可控性，70B+PACO 全面超越所有现有方法。

**[Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering](nlp_generation/are_emotion_and_rhetoric_neurons_in_llm_neuron_recognition_and_adaptive_masking_.md)**

:   系统研究LLM中情感和修辞神经元的表征机制及其内在关联，提出结合多维筛选的神经元识别框架和自适应遮蔽验证方法，实现了情感/修辞预测的定向诱导和修辞神经元辅助情感识别。

**[Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style](nlp_generation/can_you_make_it_sound_like_you_post-editing_llm-generated_text_for_personal_styl.md)**

:   作者设计一项 81 人预注册在线研究，让被试用 GPT-o4-mini 起草+人工 post-edit 重写婚礼誓词、道歉信等"在意个人风格"的文本，发现 post-edit 确实能显著拉近被试自身风格、远离 LLM 风格，但被编辑后的文本仍系统性地比独立写作更"AI 味"——而被试自己却感知不到这种残留风格痕迹。

**[Children's English Reading Story Generation via Supervised Fine-Tuning of Compact LLMs with Controllable Difficulty and Safety](nlp_generation/childrens_english_reading_story_generation_via_supervised_fine-tuning_of_compact.md)**

:   作者用 UFLI K–2 英语阅读课程对应的 2,580 篇 GPT-4o / Llama-3.3-70B 生成故事，对三个 8B 模型（Llama 3 / Granite 3.3 / Apertus）做 4 种 SFT 设计（baseline / Good Stories / Rewarded SFT / 模拟儿童读音错误），证明 **小模型 + 合适 SFT 策略** 可在 Spache 可读性、句法复杂度、毒性等 K-2 关键指标上**超过 zero-shot GPT-4o 与 Llama-3.3-70B**，其中 Rewarded SFT 最稳定、几乎无幻觉。

**[ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline](nlp_generation/conlangcrafter_constructing_languages_with_a_multi-hop_llm_pipeline.md)**

:   本文提出 ConlangCrafter，一个基于 LLM 的多跳管道，将构造语言（conlang）设计分解为音系、语法、词汇三个模块化阶段，通过随机性注入保证类型学多样性、通过自精炼循环保证内部一致性，并提出了一个包含类型学多样性分析和翻译一致性评估的自动评估框架。

**[Difficulty-Controllable Cloze Question Distractor Generation](nlp_generation/difficulty-controllable_cloze_question_distractor_generation.md)**

:   这篇论文提出 DCDG，通过双路干扰项数据增强、QA ensemble 难度聚类和多任务 seq2seq 训练，让完形填空干扰项生成模型可以按 easy/hard 控制难度，并在自动与人工评测中明显优于 GPT-4o。

**[EDUMATH: Generating Standards-aligned Educational Math Word Problems](nlp_generation/edumath_generating_standards-aligned_educational_math_word_problems.md)**

:   作者把"按 K-12 数学课程标准生成应用题（MWP）"任务系统化，搜集了 11,000+ 由真实美国教师标注的 MWP 训练数据 STEM，用 SFT + KTO + ModernBERT 过滤训出 EDUMATH-12B/30B 两个开源 SOTA 生成器，并在 3-5 年级真实学生身上做了第一个 RCT，发现学生在 LLM 题与人写题上正确率相当但**几乎一致偏好定制 LLM 题**。

**[FACTS: Table Summarization via Offline Template Generation with Agentic Workflows](nlp_generation/facts_table_summarization_via_offline_template_generation_with_agentic_workflows.md)**

:   本文提出 FACTS（Fast, Accurate, and Privacy-Compliant Table Summarization），通过三阶段 Agentic 工作流自动生成可复用的离线模板（SQL 查询 + Jinja2 模板），实现快速、准确、隐私合规的查询聚焦表格摘要，在 FeTaQA、QTSumm 和 QFMTS 三个基准上全面超越基线。

**[Frankentext: Stitching Random Text Fragments into Long-Form Narratives](nlp_generation/frankentext_stitching_random_text_fragments_into_long-form_narratives.md)**

:   提出Frankentext范式，让LLM在极端约束下（90%文本逐字复制自人类写作）拼接随机人类文本片段为连贯长篇叙事，揭示现有AI文本检测器在混合作者场景下的严重失败（72%的Frankentext被误判为人类写作）。

**[In-depth Research Impact Summarization through Fine-Grained Temporal Citation Analysis](nlp_generation/in-depth_research_impact_summarization_through_fine-grained_temporal_citation_an.md)**

:   这篇论文提出“科研影响力摘要”任务：先从论文的引文上下文中识别真正揭示影响的细粒度意图，再生成随时间演化的影响力叙事，比单纯引用数更能说明一篇论文如何被后续工作采用、批评和改造。

**[Investigating the Representation of Backchannels and Fillers in Fine-tuned Language Models](nlp_generation/investigating_the_representation_of_backchannels_and_fillers_in_fine-tuned_langu.md)**

:   论文在英日双语口语对话语料上用 MASK / NTP / TTP 三种微调任务训练 BERT / GPT-2 / TurnGPT / LLaMA-3 8B / Qwen-3 8B，再用 t-SNE 可视化和 silhouette 聚类量化"哼哈词"（backchannel 如 *uh-huh*）和"语气词"（filler 如 *um*）的表征质量；发现微调能让这些被视为"语义漂白"的功能词在嵌入空间里被显著区分开，并让模型在 NLG 生成时自然地说出多种 backchannel/filler，向"像人一样对话的 LM"迈出可量化的第一步。

**[Losses that Cook: Topological Optimal Transport for Structured Recipe Generation](nlp_generation/losses_that_cook_topological_optimal_transport_for_structured_recipe_generation.md)**

:   提出一种基于 Sinkhorn 散度的拓扑损失函数，将食材列表表示为嵌入空间中的点云，最小化预测与真实食材之间的几何差异，显著提升结构化食谱生成中食材召回率和数量精度，在人类评估中 62% 的情况被偏好。

**[Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](nlp_generation/planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)**

:   本文提出 PLOTTER 框架，首次将叙事规划从文本表示转移到图结构表示（事件图+角色图），通过多 agent 的 Evaluate-Plan-Revise 迭代循环在图拓扑上诊断和修复叙事缺陷，在叙事性、角色塑造、戏剧张力等维度上显著优于现有方法。

**[Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification](nlp_generation/right_at_my_level_a_unified_multilingual_framework_for_proficiency-aware_text_si.md)**

:   提出 Re-RIGHT 框架，通过三模块奖励（词汇覆盖率+语义保持+连贯性）的 GRPO 训练，用 4B 策略模型在英日韩中四种语言上实现按学习者熟练度等级（CEFR/JLPT/TOPIK/HSK）精确简化文本，超越 GPT-5.2 和 Gemini 2.5 等大模型。

**[SCURank: Ranking Multiple Candidate Summaries with Summary Content Units for Enhanced Summarization](nlp_generation/scurank_ranking_multiple_candidate_summaries_with_summary_content_units_for_enha.md)**

:   本文提出 SCURank，一种基于摘要内容单元（SCU）的排序框架，通过提取 SCU、跨摘要聚类估计信息重要性、按信息丰富度评分来排序候选摘要，替代不稳定的 LLM 直接排序和粗粒度的 ROUGE 排序，在多 LLM 蒸馏场景中配合 BRIO 对比学习显著提升了蒸馏模型的摘要性能。

**[ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts](nlp_generation/threadsumm_summarization_of_nested_discourse_threads_using_tree_of_thoughts.md)**

:   本文提出 ThreadSumm，一个多阶段 LLM 管道框架，将嵌套话语线程摘要建模为层次推理问题——先提取方面和原子内容单元进行内容规划，再通过句子排序构建线程感知序列，最后用 Tree of Thoughts 搜索生成和评分多个段落候选，在 Reddit/StackExchange 数据集上优于基线。

**[XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration](nlp_generation/xtragpt_context-aware_and_controllable_academic_paper_revision_via_human-ai_coll.md)**

:   本文提出 XtraGPT——首个面向学术论文修改的开源 LLM 套件（1.5B-14B），通过在 7,000 篇顶会论文和 140,000 个标准引导的指令-修改对上微调，实现上下文感知的段落级可控修改，7B 版本匹配 GPT-4o-mini，14B 版本超越 GPT-4o-mini，人类评估显示修改后论文预测评分平均提升 0.65 分。

---

## 📹 视频理解 { #video_understanding }

**[APB-V: Accelerating Long-Video Understanding via Sequence-Parallelism-aware Approximate Attention](video_understanding/apb-v_accelerating_long-video_understanding_via_sequence-parallelism-aware_appro.md)**

:   APB-V 用面向序列并行的近似注意力和系统级负载均衡加速长视频 LMM 推理，在保留完整视觉 embedding 的同时，在 64 帧 1440p 设置下相对 FlashAttn、ZigZagRing 和 APB 分别达到 12.72×、1.70× 和 1.18× 加速，且没有显著性能损失。

**[ArrowGEV: Grounding Events in Video via Learning the Arrow of Time](video_understanding/arrowgev_grounding_events_in_video_via_learning_the_arrow_of_time.md)**

:   提出 ArrowGEV，一个受物理学"时间之箭"启发的强化学习框架，通过区分时间敏感和时间不敏感事件来建模视频中的时间方向性，提升 VLM 的事件定位精度和时序理解能力。

**[Automated Knowledge Component Generation and Interpretable Knowledge Tracing in Coding Problems](video_understanding/automated_knowledge_component_generation_for_interpretable_knowledge_tracing_in_.md)**

:   这篇论文用 LLM 自动为开放式编程题生成和聚类 Knowledge Components，并提出 KCGen-KT 将学生在每个 KC 上的掌握度转成 soft token 输入 Llama 3，在 CodeWorkout 和 FalconCode 上同时提升正确率预测与学生代码生成。

**[Confidence Estimation for LLMs in Multi-turn Interactions](video_understanding/confidence_estimation_for_llms_in_multi-turn_interactions.md)**

:   首次系统研究多轮对话场景下的 LLM 置信度估计，提出两个核心准则（per-turn 校准 + 信息增加时单调性）、对应的 InfoECE 指标和 Kendall's $\tau$ 评估、Hinter-Guesser 数据集构造范式，并提出新颖的 P(SUFFICIENT) logit 探针——结果发现现有方法（verbalized / SC / P(TRUE)）在多轮场景中校准和单调性都很差，而 P(SUFFICIENT) 在 GUESS 上 InfoECE 降到 5.27（vs P(TRUE) 79.97）、$\tau$ 达 81.51，但任务远未解决。

**[CRAFT: Critic-Refined Adaptive Key-Frame Targeting for Multimodal Video Question Answering](video_understanding/craft_critic-refined_adaptive_key-frame_targeting_for_multimodal_video_question_.md)**

:   CRAFT 是一个面向新闻事件多视频问答的 claim-centric pipeline，它结合动态关键帧选择、ASR 转写、UNLI/MNLI/LLM critic 迭代修正和引用合并，在 MAGMaR-Test 上取得 0.739 macro average、0.810 reference recall 和 0.635 citation F1。

**[Distorted or Fabricated? A Survey on Hallucination in Video LLMs](video_understanding/distorted_or_fabricated_a_survey_on_hallucination_in_video_llms.md)**

:   本文首次对视频大语言模型（Vid-LLM）中的幻觉现象进行系统分类，提出"动态失真"（时空关系和引用一致性错误）和"内容捏造"（统计先验驱动和音视频冲突）的机制驱动分类体系，综述评估基准、缓解策略和根因分析。

**[DualFact: A Multimodal Fact Verification Framework for Procedural Video Understanding](video_understanding/dualfact_a_multimodal_fact_verification_framework_for_procedural_video_understan.md)**

:   作者把"做饭、家具制作"这类程序化视频字幕的事实评测拆成**双层事实**——conceptual facts（抽象角色，如 Action/Ingredient/Tool/Location）+ contextual facts（视频中可观察的 predicate–argument 关系，如 stir(soup, pot)），配套构建 YouCook3-Fact / CraftBench-Fact 两个标注隐式参数补全 (VIA) 与对比性事实的基准，并提出 MultiFactScore 用多模态/文本 NLI 在角色级别分别核查事实，进而把错误细分为 Hallucination / Saliency / Omission；实验发现 SOTA MLLM 字幕"流畅但事实残缺"，单看字幕会高估 Hallucination 一半左右，只有 video-grounded 评测才能区分 saliency 与真 hallucination。

**[GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents](video_understanding/gameplayqa_a_benchmarking_framework_for_decision-dense_pov-synced_multi-video_un.md)**

:   提出 GameplayQA，一个基于多人3D游戏视频的端到端基准框架，通过密集时间线标注（1.22标签/秒）和结构化干扰项分类学，系统评估多模态大模型在决策密集、多视角同步场景下的感知和推理能力，揭示前沿模型与人类表现仍有显著差距。

**[HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](video_understanding/hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)**

:   本文提出 HERMES，基于对 MLLM 解码器层级注意力偏好的机制性分析，将 KV 缓存概念化为层级记忆框架（浅层=感觉记忆、中层=工作记忆、深层=长期记忆），实现免训练的高效流式视频理解，在减少 68% 视频 token 的条件下仍保持或提升准确率，TTFT 延迟仅 <30ms，比前 SOTA 快 10 倍。

**[NSF-SciFy: Mining the NSF Awards Database for Scientific Claims](video_understanding/nsf-scify_mining_the_nsf_awards_database_for_scientific_claims.md)**

:   NSF-SciFy 从 NSF 奖项摘要中抽取 2.8M 科学 claims 和 investigation proposals，构建了比现有科学 claim 数据集大几个数量级的资源，并展示了它能显著提升 claim / proposal 抽取模型。

**[Probing for Reading Times](video_understanding/probing_for_reading_times.md)**

:   本文探测语言模型各层表示预测阅读时间的能力，发现早期层表示在预测早期注视指标上优于surprisal，而surprisal在晚期指标上更优，最佳预测器因语言和指标而异。

**[Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding](video_understanding/response-g1_explicit_scene_graph_modeling_for_proactive_streaming_video_understa.md)**

:   Response-G1 用查询引导的在线场景图、历史场景图检索和带时间戳的触发提示，把流式视频中的视觉证据和用户查询的响应条件显式对齐，在无需微调的情况下显著提升 Video-LLM 判断“现在是否该回答”的能力。

**[Rethinking the Idiomaticity Decomposability Hypothesis: Evidence from Distributional Learning](video_understanding/rethinking_the_idiomaticity_decomposability_hypothesis_evidence_from_distributio.md)**

:   这篇论文用上下文化语言模型作为“受控的分布式学习者”重新检验 Idiom Decomposability Hypothesis，发现模型派生的可分解性只弱相关于人类判断，并且与句法灵活性呈小而稳定的负相关，说明习语行为更像是由分布经验、surprisal 和表征稳定过程共同塑造。

**[TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](video_understanding/temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)**

:   本文提出 TemporalVLM，通过时间感知的片段编码器（重叠滑动 Video Q-Former + 融合模块）提取局部细粒度时间特征，再用 BiLSTM 聚合全局长程依赖，首次在 Video LLM 中引入 LSTM，在密集视频描述、时序定位、高光检测和动作分割四项任务上超越先前方法。

**[TRACE：基于证据定位的多视频事件理解与声明生成](video_understanding/trace_evidence_grounding-guided_multi-video_event_understanding_and_claim_genera.md)**

:   TRACE 通过"先定位后推理"的管道，先用 OCR 和目标检测构建文本可搜索的视频时间线，再用文本 LLM 进行查询条件的证据定位，最后由 LVLM 生成带引用的声明，在多视频事件理解任务上达到 SOTA，F1 从 0.705 提升到 0.811。

**[ViLL-E: Video LLM Embeddings for Retrieval](video_understanding/vill-e_video_llm_embeddings_for_retrieval.md)**

:   提出 ViLL-E，首个同时支持文本生成和 embedding 生成的 Video LLM 统一架构，通过三阶段生成-对比联合训练和自适应 KV-Former embedding head，在视频检索和时序定位上逼近专家模型，同时保持 VideoQA 竞争力。

**[VISTA: Verification In Sequential Turn-based Assessment](video_understanding/vista_verification_in_sequential_turn-based_assessment.md)**

:   VISTA 提出了一个基于声明级分解和顺序一致性追踪的多轮对话事实性评估框架，将不可验证内容细分为主观、矛盾、缺乏证据和弃权四类，在四个对话基准和八个 LLM 上显著优于 FActScore 和 LLM-as-Judge 基线。

---

## 🔎 AIGC 检测 { #aigc_detection }

**[AEGIS: A Holistic Benchmark for Evaluating Forensic Analysis of AI-Generated Academic Images](aigc_detection/aegis_a_holistic_benchmark_for_evaluating_forensic_analysis_of_ai-generated_acad.md)**

:   AEGIS 是首个面向学术图像伪造取证的综合基准，覆盖 7 大学术图类与 39 子类、4 种伪造策略（整图捏造、参考图改写、局部修复、局部编辑）和 25 个生成模型，提出取证范围判别、文字伪影识别、操作类型分类、篡改像素定位四项任务，对 25 个 MLLM 与 9 个专家模型联评后发现：即使 GPT-5.1 综合分仅 48.80%，专家模型像素 IoU 仅 30.09%，凸显「生成进化快于取证」与「MLLM 推理 vs 专家模型敏感度」的结构性互补。

**[Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](aigc_detection/beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)**

:   提出 RACE（Rhetorical Analysis for Creator-Editor Modeling），利用修辞结构理论(RST)构建逻辑图来建模文本"创作者"的思维架构，同时提取篇章单元级特征捕获"编辑者"的语言风格，实现四类细粒度 LLM 生成文本检测（人写/LLM写/LLM润色人文/人改写LLM文）。

**[BIASEDTALES-ML: A Multilingual Dataset for Analyzing Narrative Attribute Distributions in LLM-Generated Stories](aigc_detection/biasedtales-ml_a_multilingual_dataset_for_analyzing_narrative_attribute_distribu.md)**

:   BiasedTales-ML 构建了约 35 万篇覆盖 8 种语言的 LLM 生成儿童故事语料库，通过全排列提示设计和分布分析框架，揭示了**叙事中社会属性分布在不同语言间存在显著差异**，英语中心的评估无法反映多语言场景下的偏见模式。

**[C-ReD: A Comprehensive Chinese Benchmark for AI-Generated Text Detection Derived from Real-World Prompts](aigc_detection/c-red_a_comprehensive_chinese_benchmark_for_ai-generated_text_detection_derived_.md)**

:   C-ReD 构建了一个覆盖五类中文写作场景、九个 LLM 生成器和真实使用式 prompt 的中文 AI 生成文本检测基准，并显示检测难度强烈依赖领域、生成器和 prompt，而在 C-ReD 上微调能显著提升对未见模型和外部中文数据的泛化。

**[Can AI-Generated Persuasion Be Detected? Persuaficial Benchmark and AI vs. Human Linguistic Differences](aigc_detection/can_ai-generated_persuasion_be_detected_persuaficial_benchmark_and_ai_vs_human_l.md)**

:   本文引入 Persuaficial——一个覆盖六种语言的高质量 AI 生成说服性文本多语言基准，系统评估了 LLM 生成的说服性文本与人类撰写的说服性文本在自动检测难度上的差异，发现微妙的 AI 说服比人类说服更难检测（F1 下降约 20%），而过度强化的说服反而更容易被发现。

**[DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection](aigc_detection/detectrl-x_towards_reliable_multilingual_and_real-world_llm-generated_text_detec.md)**

:   DetectRL-X 构建了一个 345.6 万样本、多语言、多域、多攻击、多长度和二/三分类并行的 LLM 生成文本检测基准，证明现有检测器在真实多语言和人机协作写作场景下仍存在明显鲁棒性短板。

**[ExaGPT: Example-Based Machine-Generated Text Detection for Human Interpretability](aigc_detection/exagpt_example-based_machine-generated_text_detection_for_human_interpretability.md)**

:   ExaGPT 把"判定一段文本是人写还是 LLM 生成"这件事重构成"在数据存储里找哪一侧的相似 span 更多"，通过 BERT 嵌入 + k-NN 检索 + 动态规划做最优 span 切分，既给出可解释证据（最相似的检索 span 例子）又在 1% FPR 下把准确率刷到比此前可解释检测器最高高出 +37.0 个点。

**[Frame In, Frame Out: Measuring Framing Bias in LLM-Generated News Summaries](aigc_detection/frame_in_frame_out_measuring_framing_bias_in_llm-generated_news_summaries.md)**

:   本文提出 FIFO，用 LLM jury 加专家校准的方式在 XSum 上大规模测量 LLM 新闻摘要是否引入 framing bias，并发现若干高容量模型的框架化表达比例高于人工摘要基线。

**[From Scoring to Explanations: Evaluating SHAP and LLM Rationales for Rubric-based Teaching Quality Assessment](aigc_detection/from_scoring_to_explanations_evaluating_shap_and_llm_rationales_for_rubric-based.md)**

:   这篇论文提出一个面向 rubric 自动评分的句子级解释评估框架，在课堂教学反馈质量评分任务上比较微调 PLM、prompted LLM、SHAP 归因和 LLM rationale，发现 fine-tuned PLM 更准，而 SHAP 比 LLM 生成的解释更忠实、更可迁移。

**[GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization](aigc_detection/gigacheck_detecting_llm-generated_content_via_object-centric_span_localization.md)**

:   提出 GigaCheck，一个双策略框架：文档级使用微调 LLM 进行分类，片段级创新地将 AI 生成文本片段视为"目标"，用 DETR-like 架构实现端到端的字符级定位。

**[MASH: Evading Black-Box AI-Generated Text Detectors via Style Humanization](aigc_detection/mash_evading_black-box_ai-generated_text_detectors_via_style_humanization.md)**

:   本文提出 MASH（多阶段风格人性化对齐），通过风格注入 SFT → DPO 对齐 → 推理时精炼三阶段流水线，训练一个仅 0.1B 参数的改写器，在黑盒设置下以 92% 的平均攻击成功率规避 AI 文本检测器，同时保持优秀的语言质量。

**[mdok-style at SemEval-2026 Task 10: Finetuning LLMs for Conspiracy Detection](aigc_detection/mdok-style_at_semeval-2026_task_10_finetuning_llms_for_conspiracy_detection.md)**

:   作者把自己在 PAN@CLEF2025 拿冠军的 mdok（机器生成文本检测器）的 finetuning 范式平移到阴谋论检测：用四种数据增强（匿名化 / 大小写 / 同形字 / 去重）扩训练集，再做一轮自训练（只保留 ≥0.99 或 ≤0.01 的高置信伪标签），用 QLoRA 4-bit PEFT 微调 Qwen3-32B，最终在 SemEval-2026 Task 10 subtask 2 拿 Macro F1 = 0.78，排名 8/52（85 百分位）。

**[REFLEX: Self-Refining Explainable Fact-Checking via Verdict-Anchored Style Control](aigc_detection/reflex_self-refining_explainable_fact-checking_via_verdict-anchored_style_contro.md)**

:   REFLEX 将事实核查中的判决预测和解释生成绑定起来，用 backbone 与微调模型之间的自分歧样本构造内部 steering 向量，在不依赖搜索 API 或闭源教师模型的情况下提升判决 Macro-F1，并让解释更短、更一致、更少误导。

**[Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](aigc_detection/temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)**

:   本文通过构建跨12年的纵向写作数据集，发现LLM生成文本存在"时间扁平化"现象——虽然词汇多样性高，但在语义和认知情感维度上的时间漂移显著低于人类，仅凭时间变异模式就能以94%准确率区分人类与LLM文本。

**[When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection](aigc_detection/when_personalization_tricks_detectors_the_feature-inversion_trap_in_machine-gene.md)**

:   揭示了个性化场景下 MGT 检测器的"特征反转陷阱"——通用域中区分人写文本和机器文本的特征在个性化域中发生反转，导致检测器性能骤降甚至翻转，并提出 StyloCheck 框架通过量化检测器对反转特征的依赖程度来预测跨域性能变化，预测相关性达 0.85 以上。

**[Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](aigc_detection/who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)**

:   本文构建了首个面向LLM生成古典中文诗词的检测基准ChangAn（含30,664首诗），系统评估了12种AI检测方法在不同文本粒度和生成策略下的表现，揭示了当前中文文本检测器在古典诗词领域的严重局限性。

---

## 📚 预训练 { #llm_pretraining }

**[Compact Example-Based Explanations for Language Models](llm_pretraining/compact_example-based_explanations_for_language_models.md)**

:   本文提出选择相关性分数（Selection Relevance Score），一种无需重训练的指标来评估训练样本子集作为示例解释的质量，并证明常见的"选最高影响力"策略常不如随机选择，进而提出平衡影响力与代表性的新策略。

**[Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](llm_pretraining/data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)**

:   本文提出 Data Mixing Agent，首个基于模型的端到端领域重加权框架，通过在大量数据混合轨迹上使用 CQL 强化学习训练小型代理来学习可泛化的数据混合启发式，在数学推理持续预训练中平衡源领域和目标领域性能，且可泛化到未见过的源领域、目标模型和领域空间。

**[Demystifying Data Organization for Enhanced LLM Training](llm_pretraining/demystifying_data_organization_for_enhanced_llm_training.md)**

:   这篇论文系统研究 LLM 训练中“样本出现顺序”的影响，复用已有样本级质量/难度分数，提出边界强化、循环复习、连续课程和局部多样性四条数据组织原则，并用 STR 与 SAW 在预训练和 SFT 中稳定提升性能。

**[Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective](llm_pretraining/fine-tuning_vs_in-context_learning_in_large_language_models_a_formal_language_le.md)**

:   作者用概率层级化上下文无关文法 (HPCFG) 构造一组"无污染、有边界、可精确采样"的形式语言作为受控测试床，并提出"判别式 AUC 测试"作为统一指标，在 18 个 LLM、6 个家族、6 种语言上系统比较 FT 与 ICL：FT 在 in-distribution 上稳定胜出，但在 out-of-distribution 上两者打平，ICL 的归纳偏置与 FT 相近但对 token 敏感得多。

**[FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](llm_pretraining/forever_forgetting_curve-inspired_memory_replay_for_language_model_continual_lea.md)**

:   作者把 Ebbinghaus 遗忘曲线的"间隔回放"思路从"训练步数"重新对齐到"模型时间" (parameter update norm $\Delta_t = \|\Theta_t - \Theta_{t-1}\|_2$ 累积)——既用累积模型时间 $\tau_t$ 决定**何时回放**，又用最近更新强度 $\mu_t$ 与基线 $\mu_0$ 的不稳定比 $r_t$ 自适应控制**如何回放** (regularization 强度)；在 3 个 CL 基准、4 种 backbone (0.6B–13B) 上一致超越 SOTA，OP +1.2%、BWT +0.9% vs 最强 baseline VBM。

**[Is a Document Educational or Just Wikipedia-Style? -- Pitfalls of Classifier-Based Quality Filtering](llm_pretraining/is_a_document_educational_or_just_wikipedia-style_--_pitfalls_of_classifier-base.md)**

:   这篇论文发现Classifier-based Quality Filtering会把“Wikipedia式写法”误当成“更有教育价值”，简单改写就能让低质量网页越过预训练数据过滤阈值，FineWeb-Edu约7%的样本会因此翻转过滤决策。

**[KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates](llm_pretraining/koco_conditioning_language_model_pre-training_on_knowledge_coordinates.md)**

:   提出知识坐标条件化预训练（KoCo），将每个文档映射为三维语义坐标（来源、内容、稳定性），作为文本前缀注入预训练，使模型获得显式的上下文感知能力，在 10 个下游任务上提升性能、加速收敛约 30%，并有效缓解幻觉。

**[On the Proper Treatment of Units in Surprisal Theory](llm_pretraining/on_the_proper_treatment_of_units_in_surprisal_theory.md)**

:   这篇论文指出 surprisal theory 中“下一个单位”的单位选择一直被预训练语言模型 tokenizer 悄悄决定，因而提出一个把模型 token、语言学单位和实验 ROI 明确分离的有限状态转导框架，并在 MECO 眼动数据上验证不同单位库存会改变 surprisal 对阅读时间的预测问题本身。

**[SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](llm_pretraining/sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)**

:   本文提出 SAGE 优化器，通过 Lion 风格的符号更新方向和一个 $O(d)$ 内存开销的自适应阻尼缩放因子，解决了轻量级优化器在嵌入层上失败的"嵌入层困境"，在 Llama 模型（最大 1.3B）上以显著更低的优化器内存达到新的 SOTA 困惑度。

**[SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models](llm_pretraining/script_a_subcharacter_compositional_representation_injection_module_for_korean_p.md)**

:   本文提出 SCRIPT，一个模型无关的即插即用模块，通过双通道策略将韩文 Hangul 的子字符（Jamo）组合知识注入现有子词级 PLM 的嵌入层，无需重新预训练即可在韩语 NLU/NLG 任务上获得一致提升，并使嵌入空间更好地捕捉语法规律和语义变化。

**[Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement](llm_pretraining/toward_consistent_world_models_with_multi-token_prediction_and_latent_semantic_e.md)**

:   从理论上分析了多 Token 预测（MTP）如何通过梯度耦合机制诱导表示收缩性从而促进信念状态的涌现，但同时揭示了 MTP 的"结构性幻觉"问题（隐空间中的非法捷径），并提出 LSE-MTP 框架通过隐一致性损失和语义锚定损失将预测锚定到真实隐状态轨迹，在合成图和真实曼哈顿出租车导航上显著改善路径合法性和鲁棒性。

**[Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity](llm_pretraining/working_memory_constraints_scaffold_learning_in_transformers_under_data_scarcity.md)**

:   本文将人类工作记忆约束（固定窗口、指数衰减、逻辑衰减、首因-近因效应）集成到 GPT-2 注意力机制中，在发展可信的小规模语料（10M/100M 词）上从头训练，发现这些约束在数据稀缺时显著提升语法准确率和人类阅读时间的预测力，且促进注意力头的功能专门化。

---

## 🤖 机器人/具身智能 { #robotics }

**[Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents](robotics/breaking_down_and_building_up_mixture_of_skill-based_vision-and-language_navigat.md)**

:   SkillNav 把视觉语言导航任务拆解成 5 个原子技能（方向调整、垂直移动、停顿、地标识别、区域识别）+ 1 个时序规划技能，每个技能用合成数据微调一个 DUET 子 agent，再用 training-free 的 VLM router 做时序重排 + 子目标定位 + 技能选择，在 GSA-R2R 上取得 SOTA 泛化能力（Test-N-Scene SPL 48% vs. 之前最高 43%）。

**[Cultivating Forensic Reasoning for Generalizable Multimodal Manipulation Detection](robotics/cultivating_forensic_reasoning_for_generalizable_multimodal_manipulation_detecti.md)**

:   这篇论文提出 REFORM，把多模态伪造检测从“直接拟合标签”改成“学习可验证的取证推理过程”，并通过 ROM 推理标注数据集、双解码器和 GRPO 训练，在 ROM、DGM4 与 MMFakeBench 上取得更强的跨域泛化和可解释检测结果。

**[ElasticFlow: One-Step Physics-Consistent Policy with Elastic Time Horizons for Language-Guided Manipulation](robotics/elasticflow_one-step_physics-consistent_policy_with_elastic_time_horizons_for_la.md)**

:   提出 ElasticFlow：用平均速度场 (MeanFlow) 取代瞬时速度场学习语言条件机器人动作，配合 "弹性时间区间 $\Delta t=t-r$" 显式编码控制粒度，实现 1-NFE 单步推理 (∼71Hz)，在 LIBERO-Long、CALVIN ABC-D 等长程任务上超过 OpenVLA 与 $\pi_0$。

**[GoViG: Goal-Conditioned Visual Navigation Instruction Generation via Multimodal Reasoning](robotics/govig_goal-conditioned_visual_navigation_instruction_generation_via_multimodal_r.md)**

:   GoViG 提出一个**只靠第一视角初始与目标观测**就能生成导航指令的新任务，并把它拆成"先想象中间画面再写指令"两步，用 Anole-7B 在 token 级 MSE + 标签平滑 CE 双目标下联合训练，配合 one-pass / interleaved 两种多模态推理策略，把 BLEU-4 从基线 0.08 推到 0.32 并在跨域真实视频上保持 0.27。

**[GROKE: Vision-Free Navigation Instruction Evaluation via Graph Reasoning on OpenStreetMap](robotics/groke_vision-free_navigation_instruction_evaluation_via_graph_reasoning_on_opens.md)**

:   GROKE 提出**完全不用视觉**就评测导航指令好不好——把 OSM 地图序列化成 JSON，让 Gemini-3 Pro 当 follower agent 沿图执行指令，用 Navigation Error / SR / SDTW 反过来当指令质量的 proxy；相比启发式 baseline 在 Map2Seq 上降低 navigation error 68.5%，且 NE 与人类对"指令清晰度"的判断显著相关 ($r = -0.31, p < 0.01$)。

**[Libra-VLA: Achieving Learning Equilibrium via Asynchronous Coarse-to-Fine Dual-System](robotics/libra-vla_achieving_learning_equilibrium_via_asynchronous_coarse-to-fine_dual-sy.md)**

:   Libra-VLA 把机器人动作分解为"离散宏方向（macro-intent）+ 连续微姿态（micro-pose）"的混合动作空间，再用 System 2（VLM + 并行 coarse-action head）低频规划、System 1（diffusion transformer + 独立 SigLIP 编码器）高频精修，通过 intent buffer 实现真正异步执行，在 LIBERO 上拿到 97.2% SoTA、LIBERO-Plus 零样本 79.5%（比之前 OpenVLA-OFT+ 高 10 个点）。

**[Limited Linguistic Diversity in Embodied AI Datasets](robotics/limited_linguistic_diversity_in_embodied_ai_datasets.md)**

:   本文对主流 VLA 训练语料（RT-1、BRIDGE、TacoPlay、Language Table、LIBERO）做系统性"语言多样性体检"，从词汇/语义/句法三维度量化发现：VLA 数据**仅 < 2% 指令唯一、RT-1 整库只有 49 个 unique word、否定/条件句 < 1%**，远逊于指令调优语料（OASST2 93%、Alpaca 99.8% 唯一），这种"模板化贫乏"或许正是 VLA 模型对 paraphrase 脆弱、泛化失败的根源。

**[Mango: Multi-Agent Web Navigation via Global-View Optimization](robotics/mango_multi-agent_web_navigation_via_global-view_optimization.md)**

:   Mango 在网页导航前先构建网站的全局近似结构，再用 Thompson Sampling 在候选 URL 间动态分配有限导航预算，使 LLM web agent 不必总从首页盲目探索，并在 WebVoyager 和 WebWalkerQA 上显著超过 AgentOccam、WebWalker 等基线。

**[VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions](robotics/vln-nf_feasibility-aware_vision-and-language_navigation_with_false-premise_instr.md)**

:   本文提出 VLN-NF 基准——首个要求 VLN agent 在 3D 部分可观测环境中识别虚假前提指令并输出 NOT-FOUND 的任务，配套提出 REV-SPL 评估指标和 ROAM 两阶段混合框架，ROAM 达到 6.1 REV-SPL，比监督基线提升 45%。

**[视觉语言导航代理的能力导向失败归因](robotics/where_did_it_go_wrong_capability-oriented_failure_attribution_for_vision-and-lan.md)**

:   本文针对具身代理（特别是视觉语言导航 VLN 代理）的多层次能力失败问题，提出 CanTest 框架，通过能力导向的测试预言机与失败归因机制，精准定位导致任务失败的具体能力缺陷（感知/记忆/规划/决策），比现有方法发现的失败案例多 23–34%。

---

## ✏️ 知识编辑 { #knowledge_editing }

**[Aligning Language Models with Real-time Knowledge Editing](knowledge_editing/aligning_language_models_with_real-time_knowledge_editing.md)**

:   引入CRAFT（持续更新的中文金融知识编辑数据集）和KEDAS（基于多样化编辑增强和自适应推理的知识编辑对齐范式），解决现有知识编辑方法在实时场景中成功率-局部性-可迁移性难以兼顾的问题。

**[CLaRE-ty Amid Chaos: Quantifying Representational Entanglement to Predict Ripple Effects in LLM Editing](knowledge_editing/clare-ty_amid_chaos_quantifying_representational_entanglement_to_predict_ripple_.md)**

:   CLARE 提出了一种轻量级的表示层面方法，通过单个中间层的前向激活量化事实间的纠缠程度，用于预测模型编辑的连锁效应，相比梯度方法平均提升 62.2% Spearman 相关性，同时快 2.74 倍、内存减少 2.85 倍。

**[EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing](knowledge_editing/evoedit_evolving_null-space_alignment_for_robust_and_efficient_knowledge_editing.md)**

:   提出 EvoEdit，通过动态演化零空间投影器实现大规模序列知识编辑，在保持原有知识的同时高效注入新知识，在 10K 编辑量级下仍保持 SOTA 性能，且比 AlphaEdit 快 3.5 倍。

**[FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing](knowledge_editing/fable_fine-grained_fact_anchoring_for_unstructured_model_editing.md)**

:   本文发现现有非结构化模型编辑方法虽能整体性回忆编辑文本但无法进行细粒度事实访问，提出FABLE框架通过两阶段层次化策略将细粒度事实锚定到浅层、整体性叙事整合到深层，并构建UnFine诊断基准进行系统评估。

**[HiEdit: Lifelong Model Editing with Hierarchical Reinforcement Learning](knowledge_editing/hiedit_lifelong_model_editing_with_hierarchical_reinforcement_learning.md)**

:   HiEdit 用分层强化学习把"终身模型编辑"拆成 high-level 选层 + low-level 算梯度更新两个子任务，让 hypernetwork 按知识自适应地只动一半的层，把强基线 RLEdit 平均再提 8.48%。

**[One Mask to Rule Them All: On Hidden Facts after Editing and How to Find Them](knowledge_editing/one_mask_to_rule_them_all_on_hidden_facts_after_editing_and_how_to_find_them.md)**

:   这篇论文发现 ROME / MEMIT 并没有真正覆盖旧知识，而是通过共享的过度注意力机制压制旧知识；一个稀疏二值 mask 就能反转多数编辑，并把新编辑成功率从 98% 降到 38%。

**[Representation Interventions Enable Lifelong Knowledge Memory Control in LLMs](knowledge_editing/representation_interventions_enable_lifelong_knowledge_memory_control_in_llms.md)**

:   这篇论文提出 RILKE，把终身知识编辑从“改模型权重”转成“在隐藏表示空间施加低秩干预”，通过鲁棒训练、查询自适应路由和共享子空间模块，在 1,000 次非结构化知识编辑后仍保持接近满分的编辑成功率和较好的泛化能力，同时显著降低存储开销。

**[Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse](knowledge_editing/spectral_characterization_and_mitigation_of_sequential_knowledge_editing_collaps.md)**

:   论文从 SVD 谱结构解释顺序知识编辑为何会让 LLM 一般能力崩溃，并提出 REVIVE，在原始权重的奇异向量基中滤除会干扰 dominant singular subspace 的更新分量，使 MEMIT、RECT、AlphaEdit 等编辑器在 10,000 到 20,000 次连续编辑下同时保持编辑成功率和通用能力。

**[The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models](knowledge_editing/the_model_agreed_but_didn39t_learn_diagnosing_surface_compliance_in_large_langua.md)**

:   提出 SA-MCQ 诊断框架揭示知识编辑中的"表面合规"现象——编辑器在标准基准上达到高分但并未真正覆写内部信念，模型在判别式自评中会回退到原始参数记忆，递归编辑还会累积表征残留导致认知不稳定。

---

## 🔗 因果推理 { #causal_inference }

**[Better and Worse with Scale: How Contextual Entrainment Diverges with Model Size](causal_inference/better_and_worse_with_scale_how_contextual_entrainment_diverges_with_model_size.md)**

:   本文首次为"上下文夹带效应"（contextual entrainment）建立缩放定律，发现更大的模型在语义上下文中更能抵抗虚假信息（负指数），但在非语义上下文中更容易复制无关 token（正指数），揭示了语义过滤和机械复制两种功能的对立缩放行为。

**[ClimateCause: Complex and Implicit Causal Structures in Climate Reports](causal_inference/climatecause_complex_and_implicit_causal_structures_in_climate_reports.md)**

:   ClimateCause 构建了首个针对气候报告中复杂和隐式因果结构的专家标注数据集（874 条因果关系），支持嵌套因果、多事件拆解、相关性方向和时空语境标注，并提出基于因果图语义复杂度的可读性度量，LLM 基准测试显示因果链推理仍是重要挑战。

**[Evaluating Counterfactual Strategic Reasoning in Large Language Models](causal_inference/evaluating_counterfactual_strategic_reasoning_in_large_language_models.md)**

:   本文用重复囚徒困境和石头剪刀布的标签扰动、收益扰动与联合反事实版本评测 LLM 的策略适应能力，发现很多模型在熟悉博弈中看似会玩，但在收益结构改变后仍沿用模板化策略。

**[Function Words as Statistical Cues for Language Learning](causal_inference/function_words_as_statistical_cues_for_language_learning.md)**

:   作者一边用 186 种语言的 Universal Dependencies 语料证明"功能词高频 + 句法可预测 + 短语边界对齐"这三条分布性质是跨语种普适的，另一边在英语上构造 7 个反事实变体训练 GPT-2 small，证明 transformer 学习者只有在三条性质同时满足时学得最好，并发现一个 Goldilocks 效应——功能词必须既够高频又够多样才能既可靠又有区分度。

**[iTAG: Inverse Design for Natural Text Generation with Accurate Causal Graph Annotations](causal_inference/itag_inverse_design_for_natural_text_generation_with_accurate_causal_graph_annot.md)**

:   提出 iTAG 框架，通过逆向设计的三阶段流程（参数化因果图构建→基于 CoT 的概念赋值→结构保持的文本生成）生成同时具有极高因果图标注准确率和文本自然度的数据，可作为真实标注数据的实用替代品进行文本因果发现算法基准测试。

**[Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](causal_inference/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)**

:   本文提出 CmIR（因果模态不变表示学习），基于因果推理理论将每种模态显式解纠缠为因果不变表示和环境特定虚假表示，通过不变性约束+互信息约束+重建约束的优雅目标函数确保不变表示具有跨环境的稳定预测关系，在多模态情感/幽默/讽刺检测上取得 SOTA，尤其在 OOD 和噪声场景下表现突出。

**[Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation](causal_inference/parallel_universes_parallel_languages_a_comprehensive_study_on_llm-based_multili.md)**

:   本文系统研究了 LLM 在六种语言上的多语言反事实样本生成能力，通过直接生成和翻译两种路径对比，发现翻译路径的标签翻转率更高但需要更多编辑，识别出四类常见错误模式，并验证多语言反事实数据增强优于跨语言增强，尤其对低资源语言更有效。

---

## 📈 时间序列 { #time_series }

**[A Unified Framework for Modeling Heterogeneous Financial Data via Dual-Granularity Prompting](time_series/a_unified_framework_for_modeling_heterogeneous_financial_data_via_dual-granulari.md)**

:   提出FinLangNet框架，通过双模块架构（DeepFM处理静态特征 + 双粒度提示机制的Transformer处理时序行为）实现多尺度信用风险预测，在滴滴金融平台部署后实现KS提升6.3pp和坏账率下降9.9%。

**[ODTQA-FoRe: An Open-Domain Tabular Question Answering Dataset for Future Data Forecasting and Reasoning](time_series/odtqa-fore_an_open-domain_tabular_question_answering_dataset_for_future_data_for.md)**

:   ODTQA-FoRe 提出面向未来数值预测和预测后推理的开放域表格问答任务，并用 TimeFore 三代理框架把表格检索、SQL 取数、专用时间序列预测和答案规范化串成一个可评测 baseline。

**[STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation](time_series/stk-adapter_incorporating_evolving_graph_and_event_chain_for_temporal_knowledge_.md)**

:   本文提出 STK-Adapter，通过在 LLM 每一层嵌入三个 MoE 模块（ST-MoE 捕捉时空结构、EA-MoE 建模事件链语义、CMA-MoE 深度跨模态对齐），解决现有方法将 TKG 嵌入与 LLM 浅层对齐导致的时空信息丢失和逐层稀释问题，在四个基准数据集上显著超越 SOTA。

**[STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning](time_series/streasoner_empowering_llms_for_spatio-temporal_reasoning_in_time_series_via_spat.md)**

:   STReasoner 用网络 SDE 合成带图结构和文本语义的时空时间序列数据，再通过时间序列编码器、三阶段训练和空间感知 S-GRPO，让 LLM 学会基于时间动态与空间依赖做显式推理。

**[Temporal Leakage in Search-Engine Date-Filtered Web Retrieval: A Retrospective Forecasting Case Study](time_series/temporal_leakage_in_search-engine_date-filtered_web_retrieval_a_retrospective_fo.md)**

:   本文对 Google 和 DuckDuckGo 的日期过滤器进行系统审计，发现搜索引擎日期过滤在回顾性预测评估中严重失效——71%（Google）和 81%（DuckDuckGo）的问题至少有一个页面包含重大截止日期后信息泄漏，导致预测 Brier 分数从 0.24 虚降至 0.10。

**[Test of Time: Rethinking Temporal Signal of Benchmark Contamination](time_series/test_of_time_rethinking_temporal_signal_of_benchmark_contamination.md)**

:   这篇论文证明“cutoff 之后性能下降”并不是稳健的 benchmark contamination 证据：同一批源文档只要从原文填空题换成 LLM 改写题，时间衰减信号就会显著改变甚至消失。

**[Time-RA: Towards Time Series Reasoning for Anomaly Diagnosis with LLM Feedback](time_series/time-ra_towards_time_series_reasoning_for_anomaly_diagnosis_with_llm_feedback.md)**

:   定义 Time-RA 新任务将时间序列异常检测从二分类升级为生成式推理诊断（检测+分类+原因解释），构建首个包含约 4 万样本、10 个领域、20 种异常类型的多模态基准 RATs40K，并通过 AI 反馈标注流程和 LLM 微调验证了该范式的可行性。

---

## 🧬 计算生物 { #computational_biology }

**[AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling](computational_biology/aroma_augmented_reasoning_over_a_multimodal_architecture_for_virtual_cell_geneti.md)**

:   提出 AROMA 框架，通过整合文本证据、知识图谱拓扑信息和蛋白质序列特征的多模态架构，结合两阶段训练策略（SFT + GRPO），实现了可解释且精确的基因扰动效应预测。

**[BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models](computational_biology/biotool_a_comprehensive_tool-calling_dataset_for_enhancing_biomedical_capabiliti.md)**

:   BioTool 构建了一个覆盖 NCBI / Ensembl / UniProt 三大生物医学数据库 34 个常用工具、7,040 条经人工核验的「查询–API 调用」对的指令微调数据集，用它微调 4B 量级开源 LLM 后，工具调用质量超过 GPT-5.1 / Gemini-3 Pro / Claude-4.5-Sonnet 等商业大模型 15% 以上。

**[ChemAmp: Amplified Chemistry Tools via Composable Agents](computational_biology/chemamp_amplified_chemistry_tools_via_composable_agents.md)**

:   提出"工具放大"新范式（区别于传统的工具编排），通过 ChemAmp 框架将化学专用工具（UniMol2、Chemformer等）作为可组合积木块动态构建任务专用超级智能体，在分子设计、反应预测等四个核心化学任务上超越专用模型和通用LLM，同时推理token成本减少94%。

**[ProtoCycle: Reflective Tool-Augmented Planning for Text-Guided Protein Design](computational_biology/protocycle_reflective_tool-augmented_planning_for_text-guided_protein_design.md)**

:   ProtoCycle提出一种将LLM作为规划器、结合轻量级工具环境的**反思式智能体框架**，用于文本引导蛋白质序列设计：它用多轮“规划-工具调用-评估-反思”循环替代一次性文本到序列生成，在 Mol-Instructions 上将 ProTrek 提升到 14.681、Retrieval 提升到 0.936，并只用约 2,000 条 SFT 轨迹和在线 RL 达到接近/超过专用蛋白质设计模型的语言对齐效果。

**[ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway](computational_biology/toxreason_a_benchmark_for_mechanistic_chemical_toxicity_reasoning_via_adverse_ou.md)**

:   本文提出 ToxReason，一个基于不良结局路径 (AOP) 框架的化学毒性机理推理基准，整合药物-靶点实验数据与毒性标签，要求模型从分子起始事件推理到器官级不良结局；通过 GRPO 强化学习训练的 4B 模型在毒性预测（F1 71.4%）和推理质量上均超越 GPT-5 等大模型。

---

## 🎨 图像生成 { #image_generation }

**[ANCHOR: LLM-driven Subject Conditioning for Text-to-Image Synthesis](image_generation/anchor_llm-driven_subject_conditioning_for_text-to-image_synthesis.md)**

:   这篇论文提出 ANCHOR 数据集，用 70K+ 来自 5 家新闻媒体的抽象 caption 暴露 T2I 模型在多主体、上下文推理和细粒度 grounding 上的失败，并提出 SAFE 用 LLM 抽取关键主体、在 embedding 层强化主体表示来提升图文一致性。

**[From AR to Diffusion: Efficiently Adapting Large Language Models with Strictly Causal and Elastic Horizons](image_generation/from_ar_to_diffusion_efficiently_adapting_large_language_models_with_strictly_ca.md)**

:   本文提出 FLUID，用严格因果注意力和熵感知 Elastic Horizon 把预训练自回归 LLM 高效适配为扩散式并行生成模型，在只用 2.7B 适配 tokens 的情况下取得接近强 AR 模型、优于现有扩散基线的推理和代码生成表现。

**[MENTOR: Efficient Autoregressive Image Generation with Balanced Multimodal Control](image_generation/mentor_efficient_multimodal-conditioned_tuning_for_autoregressive_vision_generat.md)**

:   MENTOR 用统一自回归 decoder 和两阶段多模态训练，把参考图像与文本指令对齐到同一生成前缀中，在仅 3M 训练数据和 8 张 A100 约 1.5 天训练预算下，取得了较好的概念保持与 prompt following 平衡。

**[Multimodal Large Language Models for Multi-Subject In-Context Image Generation](image_generation/multimodal_large_language_models_for_multi-subject_in-context_image_generation.md)**

:   这篇论文提出 MUSIC，把多模态大语言模型的视觉推理能力引入多主体 in-context 图像生成，通过自动合成训练数据、视觉 CoT 和语义驱动空间布局规划，显著缓解多个参考主体同时生成时的主体遗漏、身份混淆和语义漂移问题。

**[Think Bright, Diffuse Nice: Enhancing T2I-ICL via Inductive-Bias Hint Instruction and Query Contrastive Decoding](image_generation/think_bright_diffuse_nice_enhancing_t2i-icl_via_inductive-bias_hint_instruction_.md)**

:   这篇论文提出训练无关的 TBDN 框架，用 Hint Instruction 让 LVLM 更关注最终 query，用 Query Contrastive Decoding 抑制先验幻觉，再把更准确的文本描述交给扩散模型，在 CoBSAT 和 T2I Fast Mini-ImageNet 上显著提升文本到图像上下文学习性能。

---

## 🛡️ AI 安全 { #ai_safety }

**[OmniCompliance-100K: A Multi-Domain Rule-Grounded Real-World Safety Compliance Dataset](ai_safety/omnicompliance-100k_a_multi-domain_rule-grounded_real-world_safety_compliance_da.md)**

:   本文构建了首个大规模、多领域、基于真实案例的 LLM 安全合规数据集 OmniCompliance-100K，包含 12,985 条人工整理的法规/政策规则和 106,009 条通过 Web 搜索智能体采集的真实合规案例，覆盖 AI 安全、数据隐私、金融、医疗等 9 个领域，并通过广泛的基准实验揭示了当前 LLM 在安全合规能力上的系统性短板。

**[On the (In-)Security of the Shuffling Defense in the Transformer Secure Inference](ai_safety/on_the_in-security_of_the_shuffling_defense_in_the_transformer_secure_inference.md)**

:   这篇论文指出 Transformer 安全推理中常用的“洗牌后公开中间激活”防御并不安全，并提出一种先把不同随机置换下的激活对齐、再解线性方程抽取权重的攻击，在 Pythia-70m 和 GPT-2 上能以约 1 美元查询成本恢复近似可用的模型权重。

**[Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF](ai_safety/reverse_constitutional_ai_a_framework_for_controllable_toxic_data_generation_via.md)**

:   提出 Reverse Constitutional AI (R-CAI)，通过反转 Constitutional AI 的原则为"毒性宪法"，结合批评-修正循环和概率截断的 RLAIF 机制，实现自动化、可控的多维度对抗毒性数据合成，同时通过概率截断解决奖励黑客导致的语义退化问题（语义连贯性提升15%）。

**[UniVid: 统一视频审核的视觉语言模型](ai_safety/univid_unified_vision-language_model_for_video_moderation.md)**

:   UniVid 通过用统一的策略感知字幕 VLM 替代 1000+ 个黑盒分类器，将视频审核系统从不可维护的"碎片化"架构演进为可解释、可复用的"端到端"审核系统，在 ByteDance 平台生产部署中相比传统方案违规泄漏率下降 42.7%。

---

## 🎬 视频生成 { #video_generation }

**[Accelerating Training of Autoregressive Video Generation Models via Local Optimization with Representation Continuity](video_generation/accelerating_training_of_autoregressive_video_generation_models_via_local_optimi.md)**

:   提出 Local Optimization + Representation Continuity (ReCo) 训练策略，通过在局部窗口内优化并约束隐状态的平滑过渡，实现自回归视频生成模型训练速度提升 2 倍且不牺牲生成质量。

**[OSCBench: Benchmarking Object State Change in Text-to-Video Generation](video_generation/oscbench_benchmarking_object_state_change_in_text-to-video_generation.md)**

:   提出 OSCBench——首个专门评估文生视频模型中物体状态变化（OSC）能力的基准，基于烹饪场景构建 1,120 条提示覆盖常规/新颖/组合三类场景，揭示即使最强 T2V 模型在 OSC 准确率上也仅达 0.786。

**[Self-Correcting Text-to-Video Generation with Misalignment Detection and Localized Refinement](video_generation/self-correcting_text-to-video_generation_with_misalignment_detection_and_localiz.md)**

:   提出 VideoRepair，首个免训练、模型无关的文本到视频自校正框架，通过 MLLM 检测细粒度文本-视频不对齐，保留正确区域并选择性修复问题区域，在 EvalCrafter 和 T2V-CompBench 上跨四种 T2V 骨干模型一致提升对齐质量。

**[TeachMaster: Generative Teaching via Code](video_generation/teachmaster_generative_teaching_via_code.md)**

:   TeachMaster 提出 Generative Teaching 范式，用代码作为教育视频的可解释中间表示，让规划、代码生成、配音、调试、同步和布局智能体协作生成完整课程视频，在接近人工质量的同时把 45 小时课程制作成本降到传统方式的约 0.3%。

---

## 🧑 人体理解 { #human_understanding }

**[CO-EVO: Co-evolving Semantic Anchoring and Style Diversification for Federated DG-ReID](human_understanding/co-evo_co-evolving_semantic_anchoring_and_style_diversification_for_federated_dg.md)**

:   CO-EVO 针对联邦域泛化行人重识别（FedDG-ReID）中的"语义-风格冲突"，提出 CSA（相机不变语义锚定）学习冻结的身份级文本原型作为"引力中心"+ GSD（全局风格多样化）用轻量 GCSB（全局相机风格库）合成真实跨域扰动，二者耦合优化在 Market-1501/MSMT17/CUHK03 leave-one-out 上 ViT mAP 平均比 SOTA 提升 14 个点（34.1→48.1）。

**[Hybrid Autoregressive-Diffusion Model for Real-Time Sign Language Production](human_understanding/hybrid_autoregressive-diffusion_model_for_real-time_sign_language_production.md)**

:   这篇论文提出 HybridSign，把自回归逐帧生成和 flow-based diffusion 细化结合起来，并加入三专家多尺度姿态表示与 confidence-aware causal attention，在 PHOENIX14T 和 How2Sign 上取得更好的手语生成质量-延迟折中。

**[Segment, Embed, and Align: A Universal Recipe for Aligning Subtitles to Signing](human_understanding/segment_embed_and_align_a_universal_recipe_for_aligning_subtitles_to_signing.md)**

:   SEA 将连续手语视频的字幕对齐拆成 sign segmentation、text-sign embedding 和 episode-level dynamic programming 三步，在 BOBSL、How2Sign、WMT-SLT SRF、SwissSLi 四个数据集上取得 SOTA F1@0.50，并能在 CPU 上高效处理长视频。

---

## 🧊 3D 视觉 { #3d_vision }

**[CodeBind: Decoupled Representation Learning for Multimodal Alignment with Unified Compositional Codebook](3d_vision/codebind_decoupled_representation_learning_for_multimodal_alignment_with_unified.md)**

:   CodeBind 用 shared-specific 表征解耦和组合式 VQ codebook 改造 ImageBind/ViT-Lens 式多模态对齐，在九种模态上同时提升跨模态分类/检索，并保留更强的模态特有细粒度信息。

---

## ✂️ 语义分割 { #segmentation }

**[AnchorSeg: Language Grounded Query Banks for Reasoning Segmentation](segmentation/anchorseg_language_grounded_query_banks_for_reasoning_segmentation.md)**

:   提出AnchorSeg，将推理分割重构为基于语言引导查询库的结构化条件生成过程，通过锚点查询显式解耦空间定位与语义推理，配合Token-Mask循环一致性训练目标，在ReasonSeg上达到SOTA（67.7% gIoU, 68.1% cIoU）。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[LLMSurgeon: Diagnosing Data Mixture of Large Language Models](self_supervised/llmsurgeon_diagnosing_data_mixture_of_large_language_models.md)**

:   LLMSurgeon 把“这个 LLM 到底训练在什么数据上”形式化为 Data Mixture Surgery，并用代理分类器的软混淆矩阵反演生成文本中的领域分布，从而在只访问模型输出的条件下估计预训练数据混合比例。

---

## 📂 其他 { #others }

**[Qayyem: A Real-time Platform for Scoring Proficiency of Arabic Essays](others/qayyem_a_real-time_platform_for_scoring_proficiency_of_arabic_essays.md)**

:   Qayyem 是首个支持跨题目（cross-prompt）多维度（multi-trait）阿拉伯语自动作文评分的 Web 平台，集成了从特征工程到 SOTA 神经模型的多种评分方案，支持端到端的学术写作评估工作流。

</div>