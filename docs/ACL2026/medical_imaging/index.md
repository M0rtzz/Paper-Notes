---
title: >-
  ACL2026 医学图像方向51篇论文解读
description: >-
  51篇ACL2026的医学图像方向论文解读，涵盖医学影像、LLM、推理、多模态、对话系统、强化学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "医学图像"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "LLM"
  - "推理"
  - "多模态"
  - "对话系统"
  - "强化学习"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**💬 ACL2026** · **51** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (17)](../../ICML2026/medical_imaging/index.md) · [📷 CVPR2026 (119)](../../CVPR2026/medical_imaging/index.md) · [🔬 ICLR2026 (79)](../../ICLR2026/medical_imaging/index.md) · [🤖 AAAI2026 (106)](../../AAAI2026/medical_imaging/index.md) · [🧠 NeurIPS2025 (147)](../../NeurIPS2025/medical_imaging/index.md) · [📹 ICCV2025 (38)](../../ICCV2025/medical_imaging/index.md)

🔥 **高频主题：** 医学影像 ×30 · LLM ×13 · 推理 ×9 · 多模态 ×5 · 对话系统 ×4

**["Excuse Me, May I Say Something…" CoLabScience: A Proactive AI Assistant for Biomedical Discovery](34excuse_me_may_i_say_something34_colabscience_a_proactive_ai_assistant_for_biom.md)**

:   CoLabScience 通过 PULI（正无标注学习干预）框架，训练一个能在生物医学团队讨论中**主动判断何时介入、如何介入**的 LLM 助手，利用 GRPO 和强化学习协调器从流式对话中自动识别最佳干预时机并生成科学建议。

**[Anonpsy: A Graph-Based Framework for Structure-Preserving De-identification of Psychiatric Narratives](anonpsy_a_graph-based_framework_for_structure-preserving_de-identification_of_ps.md)**

:   提出Anonpsy框架，将精神科叙事的去标识化重新定义为图引导的语义重写问题——先将叙事转换为语义图，在图上进行受约束的扰动以修改身份信息同时保持临床结构，最后通过图条件生成重建叙事。

**[AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling](aroma_augmented_reasoning_over_a_multimodal_architecture_for_virtual_cell_geneti.md)**

:   提出 AROMA 框架，通过整合文本证据、知识图谱拓扑信息和蛋白质序列特征的多模态架构，结合两阶段训练策略（SFT + GRPO），实现了可解释且精确的基因扰动效应预测。

**[Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders](benchmarking_and_enabling_efficient_chinese_medical_retrieval_via_asymmetric_enc.md)**

:   提出 CMedTEB（中文医学文本嵌入基准）和 CARE（非对称检索框架），前者通过多 LLM 投票+专家验证构建高质量的中文医学检索/重排/STS 基准，后者用轻量 BERT 编码查询+大型 LLM 编码文档的非对称架构，通过两阶段渐进对齐策略实现 LLM 级检索精度+BERT 级在线延迟。

**[Beyond Prompt: Fine-grained Simulation of Cognitively Impaired Standardized Patients via Stochastic Steering](beyond_prompt_fine-grained_simulation_of_cognitively_impaired_standardized_patie.md)**

:   提出 StsPatient，通过从对比指令/回复对中提取领域特定的转向向量（Steering Vector），配合随机 Token 调制（STM）机制控制注入概率来模拟不同认知障碍领域和严重程度的标准化病人，相比 prompt engineering 方法在临床真实性上平均提升 11.23%，在严重程度可控性上超越最佳基线 18.54%。

**[Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents](beyond_the_individual_virtualizing_multi-disciplinary_reasoning_for_clinical_int.md)**

:   提出 Aegle 框架，通过图结构多智能体架构虚拟化多学科会诊（MDT），将解耦并行推理和动态拓扑引入门诊问诊流程，在24个科室53项指标上超越SOTA模型。

**[Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)**

:   作者提出 MedCheck——首个面向医学 LLM benchmark **生命周期**的评估框架，把 benchmark 构建拆成 5 个阶段共 46 条标准，用它对 56 个医学 benchmark 做审计，发现医学 NLP 评测领域存在 3 个系统性问题：(1) 50% 不对齐任何医学标准（ICD/SNOMED），(2) 88% 不处理数据污染，(3) 89% 不测模型 robustness、91% 不测 uncertainty——结论是当前"排行榜进步"很大程度是幻象。

**[BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)**

:   BioHiCL 利用 MeSH（医学主题词）的**层级多标签标注**为稠密检索器提供结构化监督，通过深度加权的标签相似度对齐嵌入空间与 MeSH 语义空间，使 0.1B 模型在生物医学检索、句子相似度和问答任务上超越大多数专用模型。

**[BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models](biotool_a_comprehensive_tool-calling_dataset_for_enhancing_biomedical_capabiliti.md)**

:   BioTool 构建了一个覆盖 NCBI / Ensembl / UniProt 三大生物医学数据库 34 个常用工具、7,040 条经人工核验的「查询–API 调用」对的指令微调数据集，用它微调 4B 量级开源 LLM 后，工具调用质量超过 GPT-5.1 / Gemini-3 Pro / Claude-4.5-Sonnet 等商业大模型 15% 以上。

**[Calibrated? Not for Everyone: How Sexual Orientation and Religious Markers Distort LLM Accuracy and Confidence in Medical QA](calibrated_not_for_everyone_how_sexual_orientation_and_religious_markers_distort.md)**

:   研究社会身份标记（性取向和宗教信仰）如何扭曲LLM在医疗问答中的准确率和置信度校准，发现"同性恋"标记在9个LLM上一致导致性能下降和校准危机，且交叉身份产生非加性的特异性伤害。

**[Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?](can_continual_pre-training_bridge_the_performance_gap_between_general-purpose_an.md)**

:   本文通过构建高质量德语医学语料库 FineMed-de（从 FineWeb2 过滤 730 万文档/51 亿词），对三种 LLM（7B-24B）进行持续预训练和 SLERP 模型合并，创建 DeFineMed 模型家族，证明领域特化的 7B 模型可以在德语医学任务上显著缩小与 24B 通用模型的性能差距（胜率提升约 3.5 倍）。

**[Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)**

:   提出CoPoLLM框架，通过构建首个带认知扭曲标注的情感支持对话数据集CogBiasESC，结合认知策略强化学习（CPRL）引擎和双流条件优化（DSCO），使LLM能诊断8类认知扭曲并生成策略感知的干预回复，在15个SOTA基线上全面领先。

**[CT-FineBench: A Diagnostic Fidelity Benchmark for Fine-Grained Evaluation of CT Report Generation](ct-finebench_a_diagnostic_fidelity_benchmark_for_fine-grained_evaluation_of_ct_r.md)**

:   作者把"CT 报告好不好"这个模糊问题，拆成"每个 finding 的每条细粒度属性是否对得上"的 QA 检查清单，构建出 44k 题的 CT-FineBench 基准，对临床错误的敏感度和与人类专家打分的相关性都显著超过 BLEU/BERTScore/RadGraph/RaTEScore/GREEN 等已有指标。

**[CT-Flow: Orchestrating CT Interpretation Workflow with Model Context Protocol Servers](ct-flow_orchestrating_ct_interpretation_workflow_with_model_context_protocol_ser.md)**

:   作者把 3D CT 解读重新建模成"放射科医生用工具迭代探查"的智能体任务，用 Model Context Protocol（MCP）暴露 Data Ingestion / Global Navigation / Detailed Observation / Advanced Analysis 四类工具，构造 2000+300 条可执行轨迹的 CT-FlowBench，并 SFT 出 CT-Flow-8B：在 3D-RAD 上达到 69.46% ACC，比纯切片基线提升 +22.46%，工具调用名称错误率仅 0.007/case。

**[CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)**

:   CURA 提出一个双层不确定性校准框架：个体层面将预测不确定性与错误概率对齐，队列层面通过嵌入空间的邻域风险率正则化预测，在 MIMIC-IV 的五个临床风险预测任务上一致提升校准指标而不牺牲判别性能。

**[CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning](cure-med_curriculum-informed_reinforcement_learning_for_multilingual_medical_rea.md)**

:   作者构造了一个覆盖 13 种语言（含 Amharic、Yoruba、Swahili 等低资源语言）、共 15,774 条开放问答的医学推理数据集 CureMed-Bench，并提出 Cure-Med：一个两阶段「code-switching 感知的 SFT + 课程式 GRPO」框架，把推理正确性和语言一致性联合优化，在 7B 上将语言一致率 / 推理正确率拉到 85.21% / 54.35%，在 32B 上拉到 94.96% / 70.04%。

**[Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine](decide_less_communicate_more_on_the_construct_validity_of_end-to-end_fact-checki.md)**

:   作者用 5 位临床专家在 RedHOT（Reddit 健康讨论）社交媒体真实声明上做了一项 1,000 实例的标注研究，发现端到端医学事实核查（end-to-end fact-checking）在 construct validity 层面就站不住脚 —— 证据连接难、声明欠规约、严重程度判定主观，三大障碍即便专家也无法消除，因此提出应把医学 fact-checking 重构为「交互式医患沟通模型」而非「分类→裁决」管线。

**[Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate](dialectic-med_mitigating_diagnostic_hallucinations_via_counterfactual_adversaria.md)**

:   提出 Dialectic-Med，一个受波普尔证伪主义启发的多智能体医学诊断框架，通过提议者（诊断假设）、反对者（视觉证伪模块主动检索矛盾视觉证据）和调解者（加权共识图决策）的对抗辩证推理，在 MIMIC-CXR-VQA、VQA-RAD 和 PathVQA 上取得 SOTA，解释忠实度提升 12.5%，显著缓解诊断幻觉。

**[Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)**

:   本文提出临床诊断推理数据（CDRD）结构来捕获从症状到鉴别诊断的抽象临床推理逻辑，并基于 CDRD 通过 SFT+RL 两阶段训练构建 Dr. Assistant 模型（14B），在临床问诊基准上 ICD-Recall 超过 HuatuoGPT-o1-72B 13.59%，达到与 GPT-5 竞争的水平。

**[Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](efficient_and_effective_internal_memory_retrieval_for_llm-based_healthcare_predi.md)**

:   本文提出K2K框架，将LLM的FFN参数空间视为可检索的知识库，通过LoRA注入临床知识、激活引导的探针构建精确检索、交叉注意力重排序自适应整合，实现了无需外部检索延迟的医疗预测SOTA。

**[Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised Reinforcement Learning Approach](eliciting_medical_reasoning_with_knowledge-enhanced_data_synthesis_a_semi-superv.md)**

:   本文提出MedSSR框架，通过注入罕见病知识的可控数据合成和"自监督RL→监督RL"的半监督训练范式，高效提升LLM的医学推理能力，在罕见病任务上实现最高+5.93%的提升，突破了现有方法+3%的改进上限。

**[eTracer: Towards Traceable Text Generation via Claim-Level Grounding](etracer_towards_traceable_text_generation_via_claim-level_grounding.md)**

:   eTracer 把 RAG 响应拆成原子 claim 再去上下文里搜支持/反驳的句级证据，用三步流水线（分解 → 嵌入检索 → 蕴含判定）输出带符号分数矩阵，从而在生物医学场景下既能精确反查每条事实的出处、又能定量评估响应的忠实度。

**[Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence](faithfulness_vs_safety_evaluating_llm_behavior_under_counterfactual_medical_evid.md)**

:   本文构建 MedCounterFact 数据集——用无义词、医学术语、非医学物品和有毒物质系统替换临床试验中的干预措施——发现前沿 LLM 在反事实医疗证据面前几乎无条件遵从上下文，即便"证据"表明海洛因或芥子气有疗效也自信回答，揭示了忠实度与安全之间缺乏明确边界的严重问题。

**[From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning](from_answers_to_arguments_toward_trustworthy_clinical_diagnostic_reasoning_with_.md)**

:   本文将Toulmin论证模型适配到临床诊断过程，提出CGCL三阶段课程训练框架（事实收集→假设检验→综合结论），配合T-Eval量化评估推理结构完整性，在无需RL的情况下实现与RL方法可比的诊断推理质量。

**[HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for Medical Vision Language Tasks](heterorag_a_heterogeneous_retrieval-augmented_generation_framework_for_medical_v.md)**

:   HeteroRAG 构建了 270 万张图文对 + 五类语料的 MedAtlas 知识库，把医学多模态 RAG 拆成三件套——按模态分训的 ModCLIPs 拉报告、按语料定制 query 的 MQG 拉文档、用 HKPT 偏好微调对齐跨模态与多源知识——让 7B 模型在 11 个数据集上稳定击败 4-5× 参数量的开源 Med-LVLM。

**[HypEHR: Hyperbolic Modeling of Electronic Health Records for Efficient Question Answering](hypehr_hyperbolic_modeling_of_electronic_health_records_for_efficient_question_a.md)**

:   本文提出 HypEHR，一个仅 22M 参数的洛伦兹双曲模型，将医学编码、就诊记录和问题嵌入双曲空间，通过层级感知正则化对齐 ICD 本体结构，在 MIMIC-IV 电子病历问答任务上接近 LLM 方法的效果。

**[IndicMedDialog: A Parallel Multi-Turn Medical Dialogue Dataset for Accessible Healthcare in Indic Languages](indicmeddialog_a_parallel_multi-turn_medical_dialogue_dataset_for_accessible_hea.md)**

:   本文构造 IndicMedDialog——首个覆盖英文+9 种印度语系（Assamese / Bengali / Gujarati / Hindi / Marathi / Punjabi / Tamil / Telugu / Urdu）的**平行多轮**医学诊断对话数据集（2,980 段对话 × 10 语 = 29,800 个实例），用 LLaMA-3.3-70B 合成对话 + TranslateGemma 翻译 + native speaker 校对 + 脚本感知 post-processing 修音/拼/字符间距；并基于 4-bit 量化 LLaMA-3.2-3B + LoRA 训出 IndicMedLM，在英文/印地语/马拉地语等 7/10 种语言上拿到 post-processed accuracy 最高，同时 95.3% 医学安全通过率，揭示了 5 类系统性 failure mode（ID/LC/CDC/TTF/PLG）。

**[Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation](inflated_excellence_or_true_performance_rethinking_medical_diagnostic_benchmarks.md)**

:   本文提出 DyReMe 动态医学诊断评估框架，通过 DyGen 模块生成包含鉴别诊断和误诊因素等临床干扰项的全新诊断案例，并通过 EvalMed 模块从准确性、真实性、帮助性和一致性四个维度评估 LLM，揭示现有静态基准高估了 LLM 的诊断能力——GPT-5 在 DyReMe 上准确率下降 8.25%，12 个 LLM 均暴露出显著的可信度不足。

**[Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation](knowledge-driven_augmentation_and_retrieval_for_integrative_temporal_adaptation.md)**

:   KARITA 把"时序漂移"拆成不确定性、特征距离和本体术语稀有度三种互补信号，对每个被命中的目标样本回溯检索语义相近的源样本，再用 LLM + 领域本体（MeSH / EuroVoc / CSO）生成同义词改写做数据增强，从而以纯数据驱动的方式把源时段模型迁移到未来时段，在临床、法律、科学三类长跨度多标签分类数据上稳定优于强基线。

**[Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation](knowledge_poisoning_attacks_on_medical_multi-modal_retrieval-augmented_generatio.md)**

:   作者提出 M3Att——首个面向医学多模态 RAG 的 **query-agnostic** 知识投毒框架，用"分布引导的视觉 PGD 触发"做检索劫持 + "临床歧义引导的文本改写"绕过 LVLM 自纠错，在 5 个 LVLM × 5 数据集 × 4 个医学任务上以 <1% 的投毒率（无需查询知识、视觉扰动 $\epsilon=16/255$）平均把下游效用拉低 8.78%，且对图聚类 / 文本聚类 / 图文一致性等 3 种 pre-retrieval 防御鲁棒。

**[Language Reconstruction with Brain Predictive Coding from fMRI Data](language_reconstruction_with_brain_predictive_coding_from_fmri_data.md)**

:   本文提出 PredFT，一个结合主网络（语言解码）和侧网络（脑预测编码表征）的端到端 fMRI-to-Text 解码模型，通过从大脑预测相关脑区（PTO 区域）提取前瞻性语义表征并融合到解码过程中，在 LeBel 数据集上 BLEU-1 达 34.95%（Sub-1），相比最强基线 MapGuide 提升 7.84 个百分点。

**[Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness](learning_dynamic_representations_and_policies_from_multimodal_clinical_time-seri.md)**

:   提出 OPL-MT-MNAR 框架，通过 MNAR 感知的多模态编码器 + 贝叶斯滤波隐状态 + 离线策略学习，从结构化数据和临床文本的"缺失模式本身携带的信息"中学习 ICU 患者动态表示，实现优于临床医生行为的脓毒症治疗策略（FQE 0.679 vs 0.528）。

**[MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)**

:   本文提出 MARCH，一个模拟放射科住院医-专科医-主治医层级协作流程的多智能体框架，通过三阶段（初始报告起草、检索增强修订、共识驱动定稿）生成 CT 报告，在 RadGenome-ChestCT 数据集上 CE-F1 达 0.399，比最佳基线 Reg2RG 的 0.253 提升 57.7%。

**[Measuring What Matters!! Assessing Therapeutic Principles in Mental-Health Conversation](measuring_what_matters_assessing_therapeutic_principles_in_mental-health_convers.md)**

:   本文提出 CARE 框架和 FAITH-M 基准数据集，通过对话上下文编码与对比范例检索+知识蒸馏链式推理（KD-CoT），对 AI 生成的心理治疗对话进行六个治疗原则维度的细粒度序数评估，加权 F1 达 63.34，比最强基线 Qwen3 提升 64.26%。

**[MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)**

:   本文提出 R-MHSafe 角色感知心理健康安全分类体系和 MHSafeEval 闭环 agent 评估框架，通过对抗性多轮咨询交互系统性发现 LLM 在心理咨询场景中的角色依赖型累积安全失败，揭示了现有静态基准无法捕捉的交互层面危害。

**[Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)**

:   本文提出将话语分解为情感-逻辑-行为（ELB）三组件并用 LLM 推理多个认知扭曲实例，然后通过多视角门控注意力 MIL 框架进行 bag 级分类，在韩语（KoACD）和英语（Therapist QA）数据集上均优于 LLM 直接推理基线。

**[MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning](multidx_a_multi-source_knowledge_integration_framework_towards_diagnostic_reason.md)**

:   MultiDx 将网页检索、SOAP 结构化病例、相似病例库和细粒度推理片段检索合成一个两阶段诊断推理框架，先从多路证据生成候选疾病，再通过疾病匹配、投票和鉴别诊断重排，最终在 MedCaseReasoning 与 DiReCT 上同时提升诊断命中率和推理召回。

**[PCoA: A New Benchmark for Medical Aspect-Based Summarization With Phrase-Level Context Attribution](pcoa_a_new_benchmark_for_medical_aspect-based_summarization_with_phrase-level_co.md)**

:   PCoA 构建了一个面向随机对照试验摘要的医学方面级摘要基准，把每条方面摘要同时对齐到支撑句子和贡献短语，并用 claim、citation、phrase 三层指标系统评测 LLM 在可验证医学摘要上的能力。

**[PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment](principlismqa_a_philosophy-grounded_approach_to_assessing_llm-human_clinical_med.md)**

:   本文基于国际医学伦理黄金标准——Principlism（自主、不伤害、有益、公正四原则），构建了 PrinciplismQA 基准（3,648 题，含知识 MCQA 和开放式临床伦理困境），并配套专家校准的评估流水线，发现 LLM 在知识基准上的高准确率并不等于具备临床伦理推理能力——最强模型 o3 总分也仅 77.5%。

**[ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection](promedical_hierarchical_fine-grained_criteria_modeling_for_medical_llm_alignment.md)**

:   ProMedical 用医生参与构造的分层细粒度 clinical rubric 贯穿偏好数据、奖励模型和 benchmark，通过显式 criteria injection 训练多维 reward model，使 Qwen3-8B 在医学对齐中获得 22.3% overall accuracy 和 21.7% safety compliance 的提升。

**[Query Pipeline Optimization for Cancer Patient Question Answering Systems](query_pipeline_optimization_for_cancer_patient_question_answering_systems.md)**

:   本文提出 CoMeta，一个面向癌症患者问答（CPQA）的三层可控元数据感知 RAG 框架，通过临床混合语义-符号文档检索（CHSDR）融合 E-Utilities 实时布尔搜索与 MedCPT 语义检索，配合语义增强重叠分割（SEOS）防止上下文碎片化，在 CMMQA 数据集上将 Claude-3-Haiku 的回答准确率提升 5.24%（vs CoT）和约 3%（vs naive RAG）。

**[RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction](ra-rrg_multimodal_retrieval-augmented_radiology_report_generation_with_key_phras.md)**

:   提出 RA-RRG 框架，通过 LLM 从放射报告中提取临床关键短语并构建检索库，给定胸部 X 光影像后检索相关短语并输入 LLM 生成报告，无需 LLM 微调即可有效抑制幻觉，仅需 18 GPU 小时训练，在 CheXbert 指标上达到 SOTA。

**[RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)**

:   本文提出 RADS（Reinforcement Adaptive Domain Sampling），一种基于强化学习的样本选择策略，在极端低资源和类别不平衡的临床场景下，通过智能选择少量目标域样本进行标注和联合微调，显著提升跨域疾病检测的迁移效果。

**[Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework](region-grounded_report_generation_for_3d_medical_imaging_a_fine-grained_dataset_.md)**

:   本文提出首个带有细粒度 ROI 标注的 3D PET/CT 数据集 VietPET-RoI（越南语），以及模拟放射科医生诊断流程的层次化报告生成框架 HiRRA，通过图神经网络建模 ROI 间的空间-形态学关系，BLEU-4 提升 19.7%，临床指标 RoIQ 提升 45.8%。

**[ReMedi: Reasoner for Medical Clinical Prediction](remedi_reasoner_for_medical_clinical_prediction.md)**

:   ReMedi 将 EHR 临床预测改写成“理由-答案”生成与偏好学习问题，通过带真实结局提示的困难样本再生成、SFT 和 DPO 让医学 LLM 学会更细粒度地解释患者风险，在 MIMIC-IV 三类预测任务上相对 KARE 最高带来 19.9 个 F1 点提升。

**[RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models](reprompt_recurrent_prompt_tuning_for_integrating_structured_ehr_encoders_with_la.md)**

:   本文提出 RePrompT，一种时间感知的 LLM 框架，通过循环提示调优（将前一次就诊的隐状态作为下一次就诊的软提示）和结构化编码提示调优（注入群体级 EHR 编码器的嵌入）两种互补机制，在 MIMIC-III/IV 上的再入院和死亡率预测任务上一致超越 EHR 基线和 LLM 基线。

**[RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine](ritek_a_dataset_for_large_language_models_complex_reasoning_over_textual_knowled.md)**

:   RiTeK 构建了两个大规模医学文本知识图谱（TKG）和对应的复杂推理 QA 数据集，涵盖 6 种拓扑结构和丰富的文本描述，评估了 11 种检索方法并揭示了现有 LLM 驱动检索系统在医学 TKG 推理上的严重不足。

**[Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling](semi-supervised_diseased_detection_from_speech_dialogues_with_multi-level_data_m.md)**

:   本文提出一种纯音频的半监督学习框架，通过在会话级、片段级和帧级三个层次联合建模临床对话中的病理语音特征，利用 EMA 教师-学生网络动态生成高质量伪标签，在抑郁症和阿尔茨海默症检测中仅用 11 个标注样本即可达到全监督 90% 的性能。

**[Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)**

:   本文提出 CoMed，一种 LLM 赋能的图学习框架，通过结合 EHR 统计证据和类型约束 LLM 推理构建全局医学知识图谱，再用 LLM 生成节点描述和边理由丰富为文本属性图，最终联合训练 LoRA 微调的 LLaMA 编码器和异构 GNN 学习统一的医学概念嵌入，在 MIMIC-III/IV 上显著提升诊断预测性能。

**[TheraAgent: Self-Improving Therapeutic Agent for Precise and Comprehensive Treatment Planning](theraagent_self-improving_therapeutic_agent_for_precise_and_comprehensive_treatm.md)**

:   TheraAgent 把治疗方案生成从一次性回答改造成 generate-reflect-refine 的自改进 agent 流程，用临床维度化评估器 TheraJudge 和 score-aware memory 不断修正方案，在 HealthBench 治疗规划子集和医生盲评中显著超过强基线。

**[Thinking Like a Botanist: Challenging Multimodal Language Models with Intent-Driven Chain-of-Inquiry](thinking_like_a_botanist_challenging_multimodal_language_models_with_intent-driv.md)**

:   本文提出PlantInquiryVQA基准和Chain-of-Inquiry（CoI）框架，包含24,950张植物图像和138,068个问答对，模拟植物学家的适应性诊断提问策略，评估18个MLLM在植物病理诊断中的多步视觉推理能力，发现结构化提问显著提升诊断准确性并减少幻觉，但即使最强模型的临床实用性得分仅0.188。
