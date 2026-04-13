---
title: >-
  ACL2025 医学图像方向 23篇论文解读
description: >-
  23篇ACL2025 医学图像方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**💬 ACL2025** · 共 **23** 篇

**[A Retrieval-Based Approach To Medical Procedure Matching In Romanian](a_retrieval-based_approach_to_medical_procedure_matching_in_romanian.md)**

:   提出基于检索的罗马尼亚语医疗程序名称匹配架构——将诊所的非标准程序描述匹配到保险公司标准化术语表，比较 BM25 稀疏检索、mE5/RoBERT/BioClinicalBERT 密集嵌入及 RRF 混合方法，在 14 万+映射对上评估，度量学习微调后 mE5 表现最佳。

**[Afrimed Qa Pan African](afrimed_qa_pan_african.md)**

:   构建了首个大规模泛非洲多专科医学问答基准 AfriMed-QA（15,275 题，来自 16 个国家 60+ 医学院，涵盖 32 个专科），评估 30 个 LLM 发现：大模型在非洲医疗问题上的准确率显著低于 USMLE，生物医学专用 LLM 反而不如通用模型，消费者盲评时更偏好 LLM 回答而非临床医生回答。

**[Automated Structured Radiology Report Generation](automated_structured_radiology_report_generation.md)**

:   提出结构化放射学报告生成（SRRG）新任务，利用LLM将自由文本报告重构为标准化格式，同时引入55标签的SRR-BERT疾病分类模型和F1-SRR-BERT评估指标，解决传统报告生成中风格多样导致的生成与评估困难。

**[Auxiliary Patient Data Xray](auxiliary_patient_data_xray.md)**

:   本文研究如何将急诊科患者数据（生命体征、药物、分诊信息等）整合到多模态语言模型中用于自动胸部X光报告生成，提出将异构表格数据、文本和图像转化为统一嵌入的方法，在MIMIC-CXR + MIMIC-IV-ED数据集上显著提升了报告的诊断准确性，超越了包括CXRMate-RRG24在内的多个基准模型。

**[Biore Llm Judge Evaluation](biore_llm_judge_evaluation.md)**

:   本文首次系统研究了 LLM-as-Judge 在生物医学关系抽取评估中的表现，发现其准确率通常低于 50%，并提出结构化输出格式（JSON）和域适应技术来提升约 15% 的评估准确率。

**[Chexalign Preference Finetuning](chexalign_preference_finetuning.md)**

:   CheXalign 提出了一种无需放射科医生反馈的自动化偏好数据生成管线，利用公开数据集中的参考报告和基于参考的评估指标（如 GREEN、BERTScore）构造偏好对，通过 DPO 等直接对齐算法对胸部X光报告生成模型进行偏好微调，在 MIMIC-CXR 上取得 SOTA CheXbert 分数。

**[Clinical Coding Eight Recommendations](clinical_coding_eight_recommendations.md)**

:   这篇 position paper 通过对 MIMIC 数据集和现有自动化临床编码研究的深入分析，指出当前评估方法（如仅关注前50个高频编码、使用不恰当指标）与真实临床场景严重脱节，并提出八条具体建议来改进评估方法和研究方向。

**[Clinidial A Naturally Occurring Multimodal Dialogue Dataset For Team Reflection ](clinidial_a_naturally_occurring_multimodal_dialogue_dataset_for_team_reflection_.md)**

:   构建了 CliniDial 数据集，收集自模拟临床手术中的自然对话，包含音频转录、双角度视频和患者生理信号等多模态数据，标注了团队反思行为编码，揭示了现有 LLM 在处理标签不均衡、自然对话交互和领域多模态数据方面的显著不足。

**[Cstrl Context-Driven Sequential Transfer Learning For Abstractive Radiology Repo](cstrl_context-driven_sequential_transfer_learning_for_abstractive_radiology_repo.md)**

:   提出 CSTRL，一种基于顺序迁移学习的放射学报告摘要生成方法，通过优化的间隔句生成（GSG）预训练、Fisher 矩阵正则化防止灾难性遗忘，并结合知识蒸馏实现模型压缩，在 MIMIC-CXR 和 Open-I 数据集上大幅超越现有方法。

**[Enhancing Medical Dialogue Generation Through Knowledge Refinement And Dynamic P](enhancing_medical_dialogue_generation_through_knowledge_refinement_and_dynamic_p.md)**

:   提出 MedRef，一种融合知识精炼机制和动态 Prompt 调整策略的医学对话系统，通过隐变量过滤无关知识图谱三元组、实体-行为联合预测、以及三元组过滤器和示例选择器动态构建系统 Prompt，在 MedDG 和 KaMed 两个基准上取得 SOTA 性能。

**[Evaluation Of Llms In Medical Text Summarization The Role Of Vocabulary Adaptati](evaluation_of_llms_in_medical_text_summarization_the_role_of_vocabulary_adaptati.md)**

:   系统性基准研究发现 LLM 在高 OOV（词汇外词）和高新颖性医学文本摘要场景下性能显著下降，并通过多种词汇适配策略（MEDVOC、MEDVOC-LLM、ScafFix）证明即使 Llama-3.1（128K 词汇量）仍受过度分片问题困扰，词汇适配可带来显著改善。

**[Learning From Negative Samples In Biomedical Generative Entity Linking](learning_from_negative_samples_in_biomedical_generative_entity_linking.md)**

:   提出 ANGEL 框架，首次在生成式生物医学实体链接（BioEL）中引入负样本训练，通过两阶段策略（正样本训练 + 负样本感知的偏好优化）显著提升模型区分表面形式相似但语义不同的实体的能力，在五个基准数据集上平均 top-1 准确率提升 1.7%。

**[Medbiorag Semantic Search And Retrieval-Augmented Generation For Biomedical Lite](medbiorag_semantic_search_and_retrieval-augmented_generation_for_biomedical_lite.md)**

:   MedBioRAG 提出了一种结合语义搜索、文档检索和微调 LLM 的检索增强生成框架，在生物医学问答的文本检索、封闭式 QA 和长文本 QA 三类任务上全面超越 GPT-4o 基线和此前 SOTA。

**[Medbiorag Semantic Search And Retrieval-Augmented Generation With Large Language](medbiorag_semantic_search_and_retrieval-augmented_generation_with_large_language.md)**

:   MedBioRAG 提出了一个集成语义搜索、文档检索和微调LLM的检索增强生成框架，用于生物医学问答任务，在文本检索（NFCorpus、TREC-COVID）、封闭式问答（MedQA、PubMedQA、BioASQ）和长文本问答四个维度的多个基准上均超越了先前SOTA和GPT-4o基线模型。

**[Multimed Multilingual Medical Speech Recognition Via Attention Encoder Decoder](multimed_multilingual_medical_speech_recognition_via_attention_encoder_decoder.md)**

:   发布 MultiMed——首个多语言医学 ASR 数据集（150小时，5种语言，10种录制场景，16种口音），配套小到大规模的端到端 Whisper 模型基线，首次系统研究医学领域的多语言 ASR：单语 vs 多语微调、AED vs Hybrid 架构对比，发现多语联合训练在小模型上有收益但大模型上可能退化。

**[Oisa Radiology Report Gen](oisa_radiology_report_gen.md)**

:   提出在线迭代自对齐（OISA）方法用于放射学报告生成——四阶段循环（自生成多样数据→自评估多目标偏好→自对齐多目标优化→自迭代进一步提升），无需额外人工标注即可迭代提升报告质量，在多个评估指标上达到 SOTA。

**[Omni Rag Medical](omni_rag_medical.md)**

:   针对医疗 LLM 需要多类型多结构知识源（教科书/指南/论文/知识图谱等）的特殊需求，提出 MedOmniKB 多源知识库和 Source Planning Optimization 方法——让模型学会"该从哪个源检索什么信息"，优化后的小模型在多源医疗知识利用上达到 SOTA。

**[One Size Fits None Rethinking Fairness In Medical Ai](one_size_fits_none_rethinking_fairness_in_medical_ai.md)**

:   本文在三个多模态医学预测任务（ICU死亡率、移植物失败、急诊分诊）上进行子群体性能分析，揭示聚合指标掩盖的群体间性能差异，主张将公平性与透明度紧密结合，通过常规化的子群体报告推动负责任的医学AI部署。

**[Radar Radiology Report Gen](radar_radiology_report_gen.md)**

:   提出 Radar 框架，通过"内部知识提取+外部补充知识检索+聚合注入"三步策略增强放射学报告生成——先提取 LLM 已有的与专家分类一致的知识，再检索缺失的补充知识，最终聚合两者生成更准确的放射学报告，在 MIMIC-CXR/CheXpert-Plus/IU X-ray 三个数据集上超越 SOTA。

**[Redactor An Llm-Powered Framework For Automatic Clinical Data De-Identification](redactor_an_llm-powered_framework_for_automatic_clinical_data_de-identification.md)**

:   提出 RedactX——一个全自动、多模态的临床数据去标识化框架，结合 LLM 多轮抽取、规则处理和检索式再词汇化，在 i2b2 数据集上实现了与专用商业系统可比的 F1（0.9646），同时优化了 token 使用效率。

**[Reflectool Clinical Agent](reflectool_clinical_agent.md)**

:   ReflecTool 提出了一个反思感知的工具增强临床 Agent 框架，通过优化阶段积累成功轨迹和工具级经验，推理阶段检索相似案例并用验证器改进工具使用，在涵盖 18 个任务的 ClinicalAgent Bench 上超越纯 LLM 10+ 分、超越已有 Agent 方法 3 分。

**[Secret Semi-Supervised Clinical Trial Document Similarity Search](secret_semi-supervised_clinical_trial_document_similarity_search.md)**

:   提出 SECRET，一种半监督临床试验协议相似性搜索方法，通过将临床试验文档转换为 Q/A 对表示，并结合局部（Q/A 级）和全局（试验级）对比学习来生成嵌入，在完整试验搜索的 recall@1 上相对最佳基线提升 78%。

**[Urca Biomedical Evidence Extraction](urca_biomedical_evidence_extraction.md)**

:   本文提出 URCA（Uniform Retrieval Clustered Augmentation）框架，通过均匀检索+聚类+知识提取的 RAG 流程，从 RCT 研究全文中自动提取与临床问题相关的科学证据结论，在新构建的 CochraneForest 数据集上比最佳基线提升了 8.81% F1。
