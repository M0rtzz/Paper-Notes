---
title: >-
  ACL2025 知识编辑方向 18篇论文解读
description: >-
  18篇ACL2025 知识编辑方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✏️ 知识编辑

**💬 ACL2025** · **18** 篇论文解读

**[A General Knowledge Injection Framework for ICD Coding](a_general_knowledge_injection_framework_for_icd_coding.md)**

:   > 本文提出 GKI-ICD，一个通用的知识注入框架，通过指南合成和多任务学习机制，无需额外网络模块即可同时整合 ICD Description、Synonym 和 Hierarchy 三种知识，在 MIMIC-III 基准上取得 SOTA 性能。

**[ToxEdit: Adaptive Detoxification Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing](adaptive_detoxification_safeguarding_general_capabilities_of_llms_through_toxici.md)**

:   提出 ToxEdit——毒性感知的知识编辑方法，在 LLM 前向传播早期层用 SVM 分类器检测有害隐藏状态，通过路由机制将有害输入导向编辑后的 FFN 副本、无害输入走原始 FFN，在 LLaMA3-8B/LLaMA2-7B/Mistral-7B 上同时实现了近 98% 去毒成功率和 95% 指令遵从保留（DL 指标），解决了知识编辑去毒中"去毒 vs 过度编辑"的核心矛盾。

**[BMIKE-53: Investigating Cross-Lingual Knowledge Editing with In-Context Learning](bmike-53_investigating_cross-lingual_knowledge_editing_with_in-context_learning.md)**

:   提出 BMIKE-53 —— 覆盖 53 种语言、整合 zsRE/CounterFact/WikiFactDiff 三个知识编辑数据集的跨语言基准，系统评估 zero-shot 到 8-shot 的上下文知识编辑方法，发现文字系统（拉丁 vs 非拉丁）比语言家族更能决定跨语言编辑效果，且 metric-specific 示例策略显著优于混合示例。

**[ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains](chainedit_propagating_ripple_effects_in_llm.md)**

:   提出 ChainEdit 框架，通过将知识图谱中挖掘的逻辑规则与 LLM 内在逻辑推理能力对齐，实现知识编辑时的链式更新，将逻辑泛化准确率从约 20% 提升至 58-65%。

**[CKnowEdit: A New Chinese Knowledge Editing Dataset for Linguistics, Facts, and Logic Error Correction in LLMs](cknowedit_chinese_knowledge_editing_dataset_llms.md)**

:   构建首个面向中文语言特性的知识编辑数据集 CKnowEdit，涵盖语言学（拼音/古诗/文言/成语/谚语）、事实（历史地理）和逻辑陷阱（谐音/推理/文字游戏）三大类共 1,854 条样本，系统评估五种主流知识编辑方法在四个中文 LLM 上的表现，揭示中文独有的编辑难题。

**[CompKe: Complex Question Answering under Knowledge Editing](compke_complex_question_answering_under_knowledge_editing.md)**

:   提出CompKe基准——包含11,924个复杂问题——用于评估知识编辑方法在涉及**一对多关系、逻辑操作（交集/并集）和条件确认**的复杂推理场景下的表现，揭示现有方法在复杂问答上的显著不足。

**[Context-Robust Knowledge Editing for Language Models](context-robust_knowledge_editing_for_language_models.md)**

:   发现现有知识编辑方法在前缀上下文存在时大幅失败（编辑成功率从 90.9% 降至 69.1%），提出 CHED 基准评估上下文鲁棒性，并设计 CoRE 方法通过多样化前缀上下文 + 跨前缀隐藏状态方差正则化来增强编辑的上下文鲁棒性，在保持模型通用能力的同时显著缩小有/无上下文的性能差距。

**[DocMEdit: Towards Document-Level Model Editing](docmedit_towards_document-level_model_editing.md)**

:   首次提出文档级模型编辑任务，构建包含 37,990 条数据、105,652 个编辑事实的 DocMEdit 基准，揭示现有编辑方法在长上下文、多事实并行编辑场景下的严重不足。

**[Efficient Knowledge Editing via Minimal Precomputation](efficient_knowledge_editing.md)**

:   证明了 MEMIT/ROME/EMMET 等知识编辑方法的预计算步骤（缓存 4400 万隐向量）可以减少到理论最小值的 2-10 倍（不到原来的 0.3%），将预计算时间从数十小时降到几分钟，且编辑性能基本无损。

**[MEGen: Generative Backdoor into Large Language Models via Model Editing](megen_generative_backdoor_into_large_language_models_via_model_editing.md)**

:   提出 MEGen，一种基于模型编辑的生成式后门攻击方法，能够仅通过少量样本修改少量局部参数，在 LLM 中注入生成式后门，使模型在触发时自由输出预设的危险内容。

**[Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning](memorizing_is_not_enough_deep_knowledge_injection_through_reasoning.md)**

:   提出四层知识注入框架（记忆→检索→推理→关联），构建 DeepKnowledge 合成测试平台，系统性揭示了知识注入各层级的关键因素：重复学习实现记忆、表达多样性实现检索、显式推理模式实现深度推理和关联，为 LLM 知识更新提供了完整的方法-层级映射。

**[Mitigating Negative Interference in Multilingual Sequential Knowledge Editing through Null-Space Constraints](mitigating_negative_interference_in_multilingual_sequential_knowledge_editing_th.md)**

:   本文提出 LangEdit 框架，通过将每种语言的参数更新投影到先前编辑语言的零空间上，实现多语言序列知识编辑中不同语言更新之间的数学隔离，有效缓解负干扰并保持多语言泛化能力。

**[Neuron-Level Sequential Editing for Large Language Models](neuron-level_sequential_editing_for_large_language_models.md)**

:   提出NSE方法用于LLM的序列化模型编辑，通过权重回退（weights rewinding）防止模型崩溃、基于激活值的神经元级选择性权重更新缓解模型遗忘、以及迭代多层编辑提高大规模知识更新的成功率。

**[Revealing the Deceptiveness of Knowledge Editing: A Mechanistic Analysis of Superficial Editing](revealing_the_deceptiveness_of_knowledge_editing_a_mechanistic_analysis_of_super.md)**

:   本文定义了"表面编辑"（superficial editing）现象——经过知识编辑的模型在常规提示下表现良好，但在特制攻击探针下会回退到原始知识——并通过机制分析揭示了早期层残差流和后期层特定注意力头是导致该现象的两个关键因素。

**[SAKE: Steering Activations for Knowledge Editing](sake_steering_activations_for_knowledge_editing.md)**

:   SAKE 提出将知识编辑建模为激活空间中的分布映射问题，通过对编辑事实生成改述和逻辑蕴含的提示集合来构建源/目标激活分布，再用最优传输的线性映射替换激活向量，实现比 ROME/MEMIT 等方法更鲁棒的事实编辑，在逻辑蕴含泛化和上下文鲁棒性上显著领先。

**[ScEdit: Script-based Assessment of Knowledge Editing](scedit_script-based_assessment_of_knowledge_editing.md)**

:   提出 ScEdit，一个基于脚本（Script）的知识编辑评估基准，将传统的"What"类事实回忆评估扩展为"How"类程序性推理评估，同时引入 token 级和文本级双层评估体系，揭示了现有知识编辑方法在实际应用场景中的显著不足。

**[Structure-aware Domain Knowledge Injection for Large Language Models](structure-aware_domain_knowledge_injection_for_large_language_models.md)**

:   提出 StructTuning 方法，通过自动提取训练语料的知识分类结构，设计结构感知的持续预训练（SCPT）和结构感知的监督微调（SSFT）两阶段策略，仅用传统方法5%的数据量即可达到100%的领域知识注入效果。

**[Towards a Principled Evaluation of Knowledge Editors](towards_a_principled_evaluation_of_knowledge_editors.md)**

:   本文系统性地揭示了知识编辑评估中评估方法、评估指标和编辑批量大小的选择会显著影响编辑器排名，并通过与LM Evaluation Harness集成来评估编辑对模型整体能力的副作用。
