---
title: >-
  NeurIPS2025 医疗 NLP方向6篇论文解读
description: >-
  6篇NeurIPS2025的医疗 NLP 方向论文解读，涵盖医学影像、多模态、推理、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "医疗 NLP"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "多模态"
  - "推理"
  - "LLM"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🩺 医疗 NLP

**🧠 NeurIPS2025** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (2)](../../ICML2026/medical_nlp/index.md) · [💬 ACL2026 (14)](../../ACL2026/medical_nlp/index.md) · [🔬 ICLR2026 (5)](../../ICLR2026/medical_nlp/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/medical_nlp/index.md) · [🧪 ICML2025 (1)](../../ICML2025/medical_nlp/index.md) · [💬 ACL2025 (13)](../../ACL2025/medical_nlp/index.md)

🔥 **高频主题：** 医学影像 ×3 · 多模态 ×2

**[CGBench: Benchmarking Language Model Scientific Reasoning for Clinical Genetics Research](cgbench_benchmarking_language_model_scientific_reasoning_for_clinical_genetics_r.md)**

:   提出 CGBench，一个基于 ClinGen 专家标注的临床遗传学 benchmark，从变异和基因策展角度评估 LLM 的科学文献推理能力，涵盖证据评分、证据验证和实验证据提取三个任务，发现推理模型在细粒度任务上表现最佳但在高层判断上不如非推理模型。

**[HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring](healthslm-bench_benchmarking_small_language_models_for_mobile_and_wearable_healt.md)**

:   首个系统评估小语言模型 (SLMs, 1-4B参数) 在移动与可穿戴健康监测任务上表现的基准，覆盖zero-shot/few-shot/指令微调三种范式，并在iPhone上验证了端侧部署的可行性。

**[LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation](llm-assisted_emergency_triage_benchmark_bridging_hospital-rich_and_mci-like_fiel.md)**

:   基于MIMIC-IV-ED构建了一个开放的、LLM辅助策划的急诊分诊基准数据集，定义了医院丰富资源和大规模伤亡事件(MCI)模拟两种场景，提供基线模型和SHAP可解释性分析，推动分诊预测研究的可复现性和普及化。

**[MedMKG: Benchmarking Medical Knowledge Exploitation with Multimodal Knowledge Graph](medmkg_benchmarking_medical_knowledge_exploitation_with_multimodal_knowledge_gra.md)**

:   构建了一个融合MIMIC-CXR影像数据和UMLS临床概念的医学多模态知识图谱MedMKG，提出Neighbor-aware Filtering(NaF)图像筛选算法，并在链接预测、文本-图像检索和VQA三大任务上对24种基线方法进行了全面基准测试。

**[Mind the Gap: Aligning Knowledge Bases with User Needs to Enhance Mental Health Retrieval](mind_the_gap_aligning_knowledge_bases_with_user_needs_to_enhance_mental_health_r.md)**

:   提出一种基于"需求差距"分析的知识库增强框架，通过叠加真实用户数据（论坛帖子）与现有心理健康资源库来识别内容空白，并用定向增强策略以最少的文档增量达到接近完整语料库的 RAG 检索质量。

**[MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology](mtbbench_a_multimodal_sequential_clinical_decision-making_benchmark_in_oncology.md)**

:   提出MTBBench——首个同时覆盖多模态、纵向时序和交互式Agent工作流三个维度的临床基准，模拟分子肿瘤委员会（MTB）的决策流程，评估并增强AI Agent在肿瘤学精准医疗中的多模态纵向推理能力。
