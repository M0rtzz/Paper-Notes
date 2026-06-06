---
title: >-
  ACL2025 医疗 NLP方向13篇论文解读
description: >-
  13篇ACL2025的医疗 NLP 方向论文解读，涵盖医学影像、对话系统、问答等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "医疗 NLP"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "对话系统"
  - "问答"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🩺 医疗 NLP

**💬 ACL2025** · **13** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (2)](../../ICML2026/medical_nlp/index.md) · [💬 ACL2026 (14)](../../ACL2026/medical_nlp/index.md) · [🔬 ICLR2026 (5)](../../ICLR2026/medical_nlp/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/medical_nlp/index.md) · [🧠 NeurIPS2025 (6)](../../NeurIPS2025/medical_nlp/index.md) · [🧪 ICML2025 (1)](../../ICML2025/medical_nlp/index.md)

🔥 **高频主题：** 医学影像 ×7 · 对话系统 ×4 · 问答 ×2

**[Adaptive-VP: A Framework for LLM-Based Virtual Patients that Adapts to Trainees' Dialogue to Facilitate Nurse Communication Training](adaptive-vp_a_framework_for_llm-based_virtual_patients_that_adapts_to_trainees_d.md)**

:   提出 Adaptive-VP 框架，利用 LLM 构建可根据护理学员沟通质量动态调整行为的虚拟病人（VP），通过多 Agent 评估→动态适应→对话生成→安全监控的四模块管线，在 28 名护理专家的 between-subjects 实验中显著提升了 VP 交互的感知真实感（角色保真度 $\eta_p^2 = 0.151$，对话真实感 $\eta_p^2 = 0.254$）。

**[AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset](afrimed_qa_pan_african.md)**

:   构建首个大规模泛非洲医学问答基准 AfriMed-QA（15,275 题，16 国 60+ 医学院校、32 个专科），系统评估 30 个 LLM 并发现非洲医疗场景下存在显著的地域性能差距和生物医学模型反不如通用模型的反直觉现象。

**[Are LLMs Effective Psychological Assessors? Leveraging Adaptive RAG for Interpretable Mental Health Screening through Psychometric Practice](are_llms_effective_psychological_assessors_leveraging_adaptive_rag_for_interpret.md)**

:   本文提出了一种基于问卷引导的心理健康筛查框架，通过自适应RAG从用户Reddit帖子中检索相关内容，再用LLM代为填写标准化心理量表（如BDI-II），在无需训练数据的情况下匹配或超越有监督方法的性能，同时提供了临床可解释的评估结果。

**[ArgHiTZ at ArchEHR-QA 2025: A Two-Step Divide and Conquer Approach to Patient Question Answering for Top Factuality](arghitz_at_archehr-qa_2025_a_two-step_divide_and_conquer_approach_to_patient_que.md)**

:   在 ArchEHR-QA 2025 共享任务中提出两阶段"分治"方法：先用重排序模型从电子健康记录中提取关键句子，再用小型医学 LLM 生成回复，在不使用外部知识的情况下取得事实性排名第一、总分第 8/30 的成绩。

**[Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge](biore_llm_judge_evaluation.md)**

:   本文首次系统研究了 LLM-as-Judge 在生物医学关系抽取评估中的表现，发现其准确率通常低于 50%，并提出结构化输出格式（JSON）和域适应技术来提升约 15% 的评估准确率。

**[CliniDial: A Naturally Occurring Multimodal Dialogue Dataset for Team Reflection in Action During Clinical Operation](clinidial_a_naturally_occurring_multimodal_dialogue_dataset_for_team_reflection_.md)**

:   构建了 CliniDial 数据集，收集自模拟临床手术中的自然对话，包含音频转录、双角度视频和患者生理信号等多模态数据，标注了团队反思行为编码，揭示了现有 LLM 在处理标签不均衡、自然对话交互和领域多模态数据方面的显著不足。

**[Enhancing Medical Dialogue Generation through Knowledge Refinement and Dynamic Prompt Adjustment](enhancing_medical_dialogue_generation_through_knowledge_refinement_and_dynamic_p.md)**

:   提出 MedRef，一种融合知识精炼机制和动态 Prompt 调整策略的医学对话系统，通过隐变量过滤无关知识图谱三元组、实体-行为联合预测、以及三元组过滤器和示例选择器动态构建系统 Prompt，在 MedDG 和 KaMed 两个基准上取得 SOTA 性能。

**[Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings](evaluation_of_llms_in_medical_text_summarization_the_role_of_vocabulary_adaptati.md)**

:   系统性基准研究发现 LLM 在高 OOV（词汇外词）和高新颖性医学文本摘要场景下性能显著下降，并通过多种词汇适配策略（MEDVOC、MEDVOC-LLM、ScafFix）证明即使 Llama-3.1（128K 词汇量）仍受过度分片问题困扰，词汇适配可带来显著改善。

**[LLMs Can Simulate Standardized Patients via Agent Coevolution](evopatient_standardized_patient.md)**

:   EvoPatient 提出了一个多智能体协同进化框架，通过患者 Agent 和医生 Agent 之间的自主模拟对话，让 LLM 无需人工监督即可学会模拟标准化病人（SP），在需求对齐度上超过现有推理方法 10%+。

**[Follow-up Question Generation for Enhanced Patient-Provider Conversations](follow-up_question_generation_for_enhanced_patient-provider_conversations.md)**

:   提出 FollowupQ 多智能体框架，结合 EHR 推理、鉴别诊断和消息澄清三类 Agent，为异步医患对话自动生成个性化追问列表，在真实和半合成数据集上分别比基线提升 17% 和 5% 的 RIM 分数，将医生需要额外发送的信息收集消息减少 34%。

**[ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents](reflectool_clinical_agent.md)**

:   ReflecTool 提出了一个反思感知的工具增强临床 Agent 框架，通过优化阶段积累成功轨迹和工具级经验，推理阶段检索相似案例并用验证器改进工具使用，在涵盖 18 个任务的 ClinicalAgent Bench 上超越纯 LLM 10+ 分、超越已有 Agent 方法 3 分。

**[Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies](urca_biomedical_evidence_extraction.md)**

:   本文提出 URCA（Uniform Retrieval Clustered Augmentation）框架，通过均匀检索+聚类+知识提取的 RAG 流程，从 RCT 研究全文中自动提取与临床问题相关的科学证据结论，在新构建的 CochraneForest 数据集上比最佳基线提升了 8.81% F1。

**[VITAL: A New Dataset for Benchmarking Pluralistic Alignment in Healthcare](vital_pluralistic_alignment_healthcare.md)**

:   本文构建了首个面向医疗健康领域的多元化对齐（pluralistic alignment）基准数据集 VITAL，包含 13.1K 价值观情境和 5.4K 多选题，并通过对 8 个 LLM 的广泛评估表明，现有多元化对齐技术（尤其是 ModPlural）在医疗场景下表现不佳，简单的 prompting 反而效果更好。
