---
title: >-
  ACL2026 医疗 NLP方向14篇论文解读
description: >-
  14篇ACL2026的医疗 NLP 方向论文解读，涵盖医学影像、LLM、问答、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "医疗 NLP"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "LLM"
  - "问答"
  - "推理"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🩺 医疗 NLP

**💬 ACL2026** · **14** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (2)](../../ICML2026/medical_nlp/index.md) · [🔬 ICLR2026 (5)](../../ICLR2026/medical_nlp/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/medical_nlp/index.md) · [🧠 NeurIPS2025 (6)](../../NeurIPS2025/medical_nlp/index.md) · [🧪 ICML2025 (1)](../../ICML2025/medical_nlp/index.md) · [💬 ACL2025 (13)](../../ACL2025/medical_nlp/index.md)

🔥 **高频主题：** 医学影像 ×9 · LLM ×7 · 问答 ×3 · 推理 ×2

**[Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents](beyond_the_individual_virtualizing_multi-disciplinary_reasoning_for_clinical_int.md)**

:   提出 Aegle 框架，通过图结构多智能体架构虚拟化多学科会诊（MDT），将解耦并行推理和动态拓扑引入门诊问诊流程，在24个科室53项指标上超越SOTA模型。

**[BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)**

:   BioHiCL 利用 MeSH（医学主题词）的**层级多标签标注**为稠密检索器提供结构化监督，通过深度加权的标签相似度对齐嵌入空间与 MeSH 语义空间，使 0.1B 模型在生物医学检索、句子相似度和问答任务上超越大多数专用模型。

**[Calibrated? Not for Everyone: How Sexual Orientation and Religious Markers Distort LLM Accuracy and Confidence in Medical QA](calibrated_not_for_everyone_how_sexual_orientation_and_religious_markers_distort.md)**

:   研究社会身份标记（性取向和宗教信仰）如何扭曲LLM在医疗问答中的准确率和置信度校准，发现"同性恋"标记在9个LLM上一致导致性能下降和校准危机，且交叉身份产生非加性的特异性伤害。

**[Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](efficient_and_effective_internal_memory_retrieval_for_llm-based_healthcare_predi.md)**

:   本文提出K2K框架，将LLM的FFN参数空间视为可检索的知识库，通过LoRA注入临床知识、激活引导的探针构建精确检索、交叉注意力重排序自适应整合，实现了无需外部检索延迟的医疗预测SOTA。

**[HypEHR: Hyperbolic Modeling of Electronic Health Records for Efficient Question Answering](hypehr_hyperbolic_modeling_of_electronic_health_records_for_efficient_question_a.md)**

:   本文提出 HypEHR，一个仅 22M 参数的洛伦兹双曲模型，将医学编码、就诊记录和问题嵌入双曲空间，通过层级感知正则化对齐 ICD 本体结构，在 MIMIC-IV 电子病历问答任务上接近 LLM 方法的效果。

**[IndicMedDialog: A Parallel Multi-Turn Medical Dialogue Dataset for Accessible Healthcare in Indic Languages](indicmeddialog_a_parallel_multi-turn_medical_dialogue_dataset_for_accessible_hea.md)**

:   本文构造 IndicMedDialog——首个覆盖英文+9 种印度语系（Assamese / Bengali / Gujarati / Hindi / Marathi / Punjabi / Tamil / Telugu / Urdu）的**平行多轮**医学诊断对话数据集（2,980 段对话 × 10 语 = 29,800 个实例），用 LLaMA-3.3-70B 合成对话 + TranslateGemma 翻译 + native speaker 校对 + 脚本感知 post-processing 修音/拼/字符间距；并基于 4-bit 量化 LLaMA-3.2-3B + LoRA 训出 IndicMedLM，在英文/印地语/马拉地语等 7/10 种语言上拿到 post-processed accuracy 最高，同时 95.3% 医学安全通过率，揭示了 5 类系统性 failure mode（ID/LC/CDC/TTF/PLG）。

**[MedFact: Benchmarking the Fact-Checking Capabilities of Large Language Models on Chinese Medical Texts](medfact_benchmarking_the_fact-checking_capabilities_of_large_language_models_on_.md)**

:   MedFact 构建了一个覆盖真实中文医疗文本的专家标注事实核查 benchmark，并用 20 个 LLM 证明：当前模型较容易判断文本“有没有错”，但仍难以精确定位错误，RAG 有帮助，而多智能体和推理时扩展反而容易放大“过度批判”。

**[MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)**

:   MHGraphBench 从 PrimeKG 的心理健康子图自动构造 9 类多选任务，发现 LLM 在实体识别上接近满分，但在药物-疾病关系判断、禁忌边界和两跳 KG 推理上仍明显不足。

**[MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)**

:   本文提出 R-MHSafe 角色感知心理健康安全分类体系和 MHSafeEval 闭环 agent 评估框架，通过对抗性多轮咨询交互系统性发现 LLM 在心理咨询场景中的角色依赖型累积安全失败，揭示了现有静态基准无法捕捉的交互层面危害。

**[ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection](promedical_hierarchical_fine-grained_criteria_modeling_for_medical_llm_alignment.md)**

:   ProMedical 用医生参与构造的分层细粒度 clinical rubric 贯穿偏好数据、奖励模型和 benchmark，通过显式 criteria injection 训练多维 reward model，使 Qwen3-8B 在医学对齐中获得 22.3% overall accuracy 和 21.7% safety compliance 的提升。

**[Query Pipeline Optimization for Cancer Patient Question Answering Systems](query_pipeline_optimization_for_cancer_patient_question_answering_systems.md)**

:   本文提出 CoMeta，一个面向癌症患者问答（CPQA）的三层可控元数据感知 RAG 框架，通过临床混合语义-符号文档检索（CHSDR）融合 E-Utilities 实时布尔搜索与 MedCPT 语义检索，配合语义增强重叠分割（SEOS）防止上下文碎片化，在 CMMQA 数据集上将 Claude-3-Haiku 的回答准确率提升 5.24%（vs CoT）和约 3%（vs naive RAG）。

**[Ryze: Evidence-Enriched Data Synthesis from Biomedical Papers](ryze_evidence-enriched_data_synthesis_from_biomedical_papers.md)**

:   Ryze 将生物医学论文 PDF 自动转成保留图表、caption、结构化抽取和引用段落的证据增强 QA 数据，并用进度门控的 SFT+GRPO 训练 BioVLM-8B，在 LAB-Bench 上以 48.0% weighted accuracy 超过 Qwen3-VL-8B base 12.6 个百分点、超过 GPT-5.2 3.8 个百分点。

**[SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning](sema-rag_a_self-evolving_multi-agent_retrieval-augmented_generation_framework_fo.md)**

:   提出 SEMA-RAG，一种自演化多智能体 RAG 框架，通过三个专职智能体（解释器、探索器、仲裁器）模拟临床推理的分阶段工作流，在 5 个医学 QA 基准上平均超越最强基线 +6.46 个准确率点。

**[Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)**

:   本文提出 CoMed，一种 LLM 赋能的图学习框架，通过结合 EHR 统计证据和类型约束 LLM 推理构建全局医学知识图谱，再用 LLM 生成节点描述和边理由丰富为文本属性图，最终联合训练 LoRA 微调的 LLaMA 编码器和异构 GNN 学习统一的医学概念嵌入，在 MIMIC-III/IV 上显著提升诊断预测性能。
