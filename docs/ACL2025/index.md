---
title: >-
  ACL2025 1520篇论文解读
description: >-
  1520篇ACL2025论文深度解读，每篇5分钟读懂核心思想。覆盖LLM/NLP、多模态VLM、LLM评测、多语言/翻译、信息检索/RAG、模型压缩等41个研究领域，每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 ACL2025 论文笔记

共 **1520** 篇笔记，覆盖 **41** 个领域。

## 领域概览

| 领域 | 篇数 |
|:-----|-----:|
| 💬 [LLM/NLP](#llm_nlp) | 303 |
| 🧩 [多模态VLM](#multimodal_vlm) | 117 |
| 📊 [LLM评测](#llm_evaluation) | 73 |
| 🌐 [多语言/翻译](#multilingual_mt) | 73 |
| 🔍 [信息检索/RAG](#information_retrieval) | 65 |
| 📦 [模型压缩](#model_compression) | 62 |
| ⚖️ [对齐/RLHF](#llm_alignment) | 61 |
| 🦾 [LLM Agent](#llm_agent) | 45 |
| 💡 [LLM推理](#llm_reasoning) | 43 |
| 🔒 [LLM安全](#llm_safety) | 36 |
| 🎵 [音频/语音](#audio_speech) | 33 |
| 📚 [预训练/数据](#llm_pretraining) | 32 |
| ⚡ [LLM效率](#llm_efficiency) | 30 |
| 🛡️ [AI安全](#ai_safety) | 27 |
| 📖 [NLP理解](#nlp_understanding) | 26 |
| 🕸️ [图学习](#graph_learning) | 23 |
| 🏥 [医学图像](#medical_imaging) | 23 |
| 👥 [社会计算](#social_computing) | 23 |
| 🔬 [可解释性](#interpretability) | 22 |
| ✍️ [文本生成](#nlp_generation) | 22 |
| 💻 [代码智能](#code_intelligence) | 20 |
| ✏️ [知识编辑](#knowledge_editing) | 16 |
| 🔎 [AIGC检测](#aigc_detection) | 11 |
| 🗣️ [对话系统](#dialogue) | 10 |
| 🔗 [因果推理](#causal_inference) | 7 |
| 🎨 [图像生成](#image_generation) | 7 |
| 🎁 [推荐系统](#recommender) | 7 |
| 🎮 [强化学习](#reinforcement_learning) | 7 |
| 🤖 [机器人/具身智能](#robotics) | 7 |
| 🎬 [视频理解](#video_understanding) | 7 |
| 🔄 [自监督/表示学习](#self_supervised) | 6 |
| 📈 [时间序列](#time_series) | 5 |
| 🎯 [目标检测](#object_detection) | 4 |
| 🧑 [人体理解](#human_understanding) | 3 |
| ✂️ [语义分割](#segmentation) | 3 |
| 🖼️ [图像恢复](#image_restoration) | 2 |
| 📡 [信号/通信](#signal_comm) | 2 |
| 🧊 [3D视觉](#3d_vision) | 1 |
| 🚗 [自动驾驶](#autonomous_driving) | 1 |
| 📐 [优化/理论](#optimization) | 1 |
| 📂 [其他](#others) | 254 |

---

## 💬 LLM/NLP { #llm_nlp }

**[A Large-Scale Real-World Evaluation Of Llm-Based Virtual Teaching Assistant](llm_nlp/a_large-scale_real-world_evaluation_of_llm-based_virtual_teaching_assistant.md)**

:   在 KAIST 477 名研究生的 AI 编程课上部署基于 GPT-4o-mini + RAG 的虚拟教学助手（VTA），通过三轮大规模问卷调查和 3,869 条交互日志分析系统性评估 VTA 的有效性与接受度，发现 VTA 在编程和概念问题上有效但信任度随时间下降。

**[A Modular Dataset To Demonstrate Llm Abstraction Capability](llm_nlp/a_modular_dataset_to_demonstrate_llm_abstraction_capability.md)**

:   提出ArrangementPuzzle拼图数据集并训练LLM激活值分类器，发现分类器以>80%准确率识别推理正确性，揭示LLM在中间-后层Transformer层编码了区分逻辑等价与语义等价的抽象推理概念。

**[A Semantic-Aware Layer-Freezing Approach To Computation-Efficient Fine-Tuning Of](llm_nlp/a_semantic-aware_layer-freezing_approach_to_computation-efficient_fine-tuning_of.md)**

:   通过分析LLM推理过程中潜在表征的转移轨迹（transition traces）计算各层语义偏差，结合推导的缩放律公式估计各层对降低损失的贡献，从而确定"在哪些层微调"，实现与PEFT正交的高效微调方法。

**[A Survey Of Automatic Prompt Optimization With Instruction-Focused Heuristic-Bas](llm_nlp/a_survey_of_automatic_prompt_optimization_with_instruction-focused_heuristic-bas.md)**

:   系统综述 80+ 种基于启发式搜索算法的自动 Prompt 优化方法，提出五维分类体系（Where/What/What criteria/Which operators/Which algorithms）将碎片化研究统一到一个完整的分析框架下。

**[A Survey Of Large Language Models In Psychotherapy Current Landscape And Future ](llm_nlp/a_survey_of_large_language_models_in_psychotherapy_current_landscape_and_future_.md)**

:   系统综述 LLM 在心理治疗中的应用——提出评估（Assessment）→ 诊断（Diagnosis）→ 治疗（Treatment）三阶段概念分类法，覆盖症状检测、诊断推理、治疗对话策略等，揭示当前研究失衡（聚焦常见障碍、语言偏差、方法碎片化、理论整合不足），并提出连续多阶段建模和实时自适应系统等未来方向。

**[A Survey Of Llm-Based Agents In Medicine How Far Are We From Baymax](llm_nlp/a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)**

:   系统综述 LLM-based Agent 在医学中的应用——分析 Agent 架构（系统配置/临床规划/医学推理/外部能力增强）、应用场景（临床决策/文档/训练模拟/服务优化）和评估框架，覆盖 60 篇研究，识别幻觉管理、多模态整合、部署障碍和伦理问题等关键挑战。

**[A Survey On Efficient Large Language](llm_nlp/a_survey_on_efficient_large_language.md)**

:   首个系统性的数据高效 LLM 后训练综述，提出"数据价值飞轮"分类法，将方法分为五大类（数据选择、质量增强、合成生成、蒸馏压缩、自演进生态），覆盖 100+ 篇代表性工作并展望未来方向。

**[A Systematic Study Of Compositional Syntactic Transformer Language Models](llm_nlp/a_systematic_study_of_compositional_syntactic_transformer_language_models.md)**

:   对组合型句法 Transformer 语言模型（Compositional SLMs）进行系统研究——识别四个关键设计维度（句法树二叉化、线性化方向、组合函数、子成分掩码），提出统一框架涵盖 16 种变体（含 13 种全新变体），在语言建模、句法泛化、摘要、对话和推理效率上全面评估，给出多条设计建议。

**[A Training-Free Llm-Based Approach To General Chinese Character Error Correction](llm_nlp/a_training-free_llm-based_approach_to_general_chinese_character_error_correction.md)**

:   提出通用中文字符纠错任务C2EC（覆盖替换、缺失、冗余三种错误类型），通过扩展训练无关的CSC方法并结合Levenshtein距离和prompt-based LLM，使14B参数模型在不微调的条件下达到近50倍大模型的纠错性能。

**[Aad-Llm Neural Attention-Driven Auditory Scene Understanding](llm_nlp/aad-llm_neural_attention-driven_auditory_scene_understanding.md)**

:   提出意图感知听觉场景理解（II-ASU）范式和 AAD-LLM 原型系统——通过颅内脑电（iEEG）解码听者正在关注哪个说话人，将注意力状态注入听觉 LLM（Qwen2-Audio），使模型在多说话人场景中生成与听者感知对齐的回答，在描述/转录/提取/问答四个任务上主观和客观评估均优于无注意力感知的基线。

**[Adaptive-Vp A Framework For Llm-Based Virtual Patients That Adapts To Trainees D](llm_nlp/adaptive-vp_a_framework_for_llm-based_virtual_patients_that_adapts_to_trainees_d.md)**

:   提出 Adaptive-VP——基于 LLM 的虚拟病人对话生成框架，根据护理学员的沟通质量动态调整虚拟病人行为（沟通差→升级敌意，沟通好→缓和），包含案例开发管线+评估模块+动态适应模块+对话生成模块+安全监控模块五个组件，专家护士评估显示其交互自然度和真实感显著优于现有方法。

**[Afrobench How Good Are Large Language Models On African Languages](llm_nlp/afrobench_how_good_are_large_language_models_on_african_languages.md)**

:   构建 AfroBench——首个大规模非洲语言 LLM 多任务评测基准，覆盖 64 种非洲语言、15 个任务、22 个数据集（9 NLU + 6 生成 + 6 知识QA + 1 数学推理），系统对比提示式 LLM 与微调 BERT/T5 基线，发现非洲语言与英语之间存在巨大性能差距，且性能与单语资源可用性密切相关。

**[Aimscheck Modern Slavery](llm_nlp/aimscheck_modern_slavery.md)**

:   提出 AIMSCheck——使用 LLM 辅助审查企业现代奴隶制声明是否合规的端到端框架，构建英国和加拿大的新标注数据集（AIMS.uk/AIMS.ca），三层分解合规评估增强可解释性，在澳大利亚数据上训练的模型能有效跨司法管辖泛化到英国和加拿大。

**[Algorithmic Fidelity German Opinion](llm_nlp/algorithmic_fidelity_german_opinion.md)**

:   使用德国纵向选举研究（GLES）的开放式调查数据，评估 LLM 在生成反映德国亚群体公共舆论方面的"算法保真度"，发现 Llama2 在建模群体意见方面优于其他 LLM，但对左翼政党支持者的表征好于右翼（如 AfD），且提示中包含更多人口统计变量可改善表现。

**[Alignment Drift In Cefr-Prompted Llms For Interactive Spanish Tutoring](llm_nlp/alignment_drift_in_cefr-prompted_llms_for_interactive_spanish_tutoring.md)**

:   通过 LLM 模拟师生对话实验，发现基于 CEFR 等级的 system prompting 虽然能初步约束 LLM 输出的西班牙语难度，但随着对话轮次增加，这种约束效果逐渐衰减——作者将此现象命名为"alignment drift"，表明仅靠提示工程不足以支撑长期的自适应语言教学。

**[Arabic Dialects Assumptions Revisited](llm_nlp/arabic_dialects_assumptions_revisited.md)**

:   系统验证了阿拉伯语 NLP 中四个被广泛接受的假设，通过 978 个方言句子+33 名标注者的多标签标注数据集证明：56% 的方言句子在多个区域方言中有效，方言词表的区分度被高估，句子长度与方言歧义性的相关性远弱于方言化程度（ALDi），不同方言说话者对同一句子的 ALDi 评级差异显著。

**[Are Your Llms Capable Of Stable Reasoning](llm_nlp/are_your_llms_capable_of_stable_reasoning.md)**

:   提出 G-Pass@k 评估指标和 LiveMathBench 动态基准，从"性能上限"和"稳定性"两个维度全面评估LLM的推理能力，揭示了当前LLM在推理一致性上存在巨大提升空间。

**[Arithmattack Evaluating Robustness Of Llms To Noisy Context In Math Problem Solv](llm_nlp/arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)**

:   提出 ArithmAttack，通过在数学题上下文中随机插入标点符号（不改变任何单词）来测试 LLM 的鲁棒性，发现八个主流 LLM（包括 Llama3、Mistral、DeepSeek）在面对这种简单噪声时性能都显著下降。

**[Astute Rag Knowledge Conflicts](llm_nlp/astute_rag_knowledge_conflicts.md)**

:   Astute RAG 提出了一种对不完美检索具有鲁棒性的 RAG 方法，通过自适应生成 LLM 内部知识作为补充、带有来源标注的知识整合、以及基于可靠性的答案生成三个步骤，在 Gemini 和 Claude 上显著优于现有鲁棒 RAG 方法，且是唯一在最坏情况下（检索全部无用）不劣于无 RAG 基线的方法。

**[Atrie Legal Interpretation](llm_nlp/atrie_legal_interpretation.md)**

:   提出 ATRIE 框架模拟法学教义研究流程——自动从判例法中检索概念相关信息、解释法律概念、并通过下游任务（法律概念蕴涵LCE）自动评估解释质量，生成的解释在全面性和可读性上与专家相当。

**[Attention Speaks Volumes Localizing And Mitigating Bias In Language Models](llm_nlp/attention_speaks_volumes_localizing_and_mitigating_bias_in_language_models.md)**

:   提出Atlas（Attention-based Targeted Layer Analysis and Scaling），通过分析注意力分数定位LLM中偏见集中的层，然后在这些层进行推理时注意力缩放干预来缓解偏见，在BBQ、CrowS-Pairs和WinoGender三个数据集、四个模型上有效降低偏见，且仅增加0.82%的困惑度。

**[Autogui Scaling Gui Grounding With Automatic](llm_nlp/autogui_scaling_gui_grounding_with_automatic.md)**

:   提出 AutoGUI 自动标注管线——通过模拟交互比较 UI 状态变化 + LLM 推断元素功能 + LLM 验证过滤，构建 704K 高质量 UI 功能标注数据集，标注正确率 96.7% 可比人类，显著提升 VLM 的 UI grounding 能力且展现数据扩展效应。

**[Automatic Transmission For Llm Tiers Optimizing Cost And Accuracy In Large Langu](llm_nlp/automatic_transmission_for_llm_tiers_optimizing_cost_and_accuracy_in_large_langu.md)**

:   提出 LLM-AT（LLM Automatic Transmission）框架——无需训练即可动态选择 LLM 层级，Starter 通过准确率估计器选择初始层级，Generator 生成回答，Judge 评估有效性，无效则自动升级到更高层级，在不同难度任务上平衡准确率和成本，比始终用顶级模型更高效。

**[Awes Laws And Flaws From Todays Llm Research](llm_nlp/awes_laws_and_flaws_from_todays_llm_research.md)**

:   对 2020-2024 年 2000+ 篇 LLM 相关论文进行科学方法论的系统批判——基于统计检验/可复现性/伦理声明等标准评估研究质量，发现多种趋势：伦理声明减少、LLM 作为评估器增多、无人工评估的推理能力声称增多、统计严谨性下降，但会议检查清单（如 ACL 强制 limitations 章节）确实有效缓解部分问题。

**[Axis Efficient Human-Agent-Computer Interaction With Api-First Llm-Based Agents](llm_nlp/axis_efficient_human-agent-computer_interaction_with_api-first_llm-based_agents.md)**

:   提出 AXIS 框架，通过让 LLM Agent 优先调用 API 而非模拟人类 UI 操作来完成应用任务，在 Microsoft Word 实验中将任务完成时间缩短 65-70%，认知负荷降低 38-53%，同时保持 97-98% 的准确率。

**[Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](llm_nlp/bandit-based_prompt_design_strategy_selection_improves_prompt_optimizers.md)**

:   首次提出 Prompt 设计策略的显式选择机制 OPTS——将 CoT/角色提示/少样本等多种策略视为多臂老虎机的"臂"，用 Thompson 采样动态选择要应用的策略，集成到 EvoPrompt 后在 BIG-Bench Hard 上将 GPT-4o mini 性能提升最高 50%，超越隐式策略选择（APET）和均匀采样。

**[Bco Binary Classifier Alignment](llm_nlp/bco_binary_classifier_alignment.md)**

:   提出 BCO（Binary Classifier Optimization），从数学上证明二元交叉熵损失是 DPO 损失的上界，使 LLM 对齐仅需"点赞/踩"二元反馈而非成对偏好数据，并通过新颖的 reward shift 技术收紧上界，在配对偏好数据集上与 DPO 持平，在真实 Likert-5 标注数据上优于 DPO 和 KTO。

**[Beyond Dialogue A Profile-Dialogue Alignment Framework Towards General Role-Play](llm_nlp/beyond_dialogue_a_profile-dialogue_alignment_framework_towards_general_role-play.md)**

:   提出 Beyond Dialogue 框架，通过 Profile-Dialogue 对齐消除角色扮演训练中 profile 与对话之间的偏差，并引入句子级细粒度对齐任务，使模型更好地理解和表现角色特质。

**[Beyond In-Context Learning Aligning Long-Form Generation Of Large Language Model](llm_nlp/beyond_in-context_learning_aligning_long-form_generation_of_large_language_model.md)**

:   证明 ICL 示例不足以教会 LLM 任务的语言和格式分布，提出 LongGuide 自动生成质量指标和输出约束两种 guidelines 来增强长文本生成。

**[Beyond Output Matching Bidirectional Alignment For Enhanced In-Context Learning](llm_nlp/beyond_output_matching_bidirectional_alignment_for_enhanced_in-context_learning.md)**

:   提出 Bidirectional Alignment (BiAlign)，在传统知识蒸馏仅对齐输出分布的基础上，新增输入偏好对齐——通过 ranking loss 让学生模型学习教师模型对不同 ICL 示例的偏好排序，在语言理解、推理和代码 5 个任务上一致优于基线，GSM8K 提升 20%、LogiQA 提升 18%。

**[Beyond Profile From Surface-Level Facts To Deep Persona Simulation In Llms](llm_nlp/beyond_profile_from_surface-level_facts_to_deep_persona_simulation_in_llms.md)**

:   提出 CharacterBot，通过 4 个训练任务（视角重建预训练 + 选择题/生成式QA/风格迁移微调）和 CharLoRA 参数更新机制，从鲁迅 17 部杂文集中学习其语言风格和深层思想模式，在语言准确性和观点理解上显著超越各基线。

**[Beyond Prompt Engineering Robust Behavior Control In Llms Via Steering Target At](llm_nlp/beyond_prompt_engineering_robust_behavior_control_in_llms_via_steering_target_at.md)**

:   提出 STA（Steering Target Atoms），利用稀疏自编码器 (SAE) 将 LLM 的表示解耦为原子知识组件，通过激活幅度和频率筛选目标原子并操控，实现比提示工程更鲁棒、更精细的行为控制，在安全解毒和推理控制任务上效果优于现有 steering 方法。

**[Bias In Language Models Beyond Trick Tests Ruted Evaluation](llm_nlp/bias_in_language_models_beyond_trick_tests_ruted_evaluation.md)**

:   通过对比标准偏见基准（"trick tests"）与基于真实使用场景的 RUTEd 评估，发现标准偏见基准与真实场景中的偏见表现无显著相关性，主张偏见评估应面向具体应用场景。

**[Big5-Chat Shaping Llm Personalities Through Training On Human-Grounded Data](llm_nlp/big5-chat_shaping_llm_personalities_through_training_on_human-grounded_data.md)**

:   提出了 Big5-Chat 数据集（10万条对话），通过 SFT 和 DPO 训练方法将真实人类大五人格特质嵌入 LLM，效果显著优于基于提示的方法，且发现高尽责性/宜人性、低外向性/神经质的人格配置能提升模型推理能力。

**[Bipro Zero-Shot Chinese Poem Generation Via Block Inverse Prompting Constrained ](llm_nlp/bipro_zero-shot_chinese_poem_generation_via_block_inverse_prompting_constrained_.md)**

:   提出 BIPro 框架，利用块生成模型（Block Generative Model）的中间文本生成能力，通过"修订（revise）"和"重写（rewrite）"两种块逆提示方法，在无需领域特定训练的情况下使弱模型 GLM-10B 在开放式传统中国诗歌生成任务中超越 GPT-4 和最佳专用系统。

**[Boosting Llms Molecular Structure Elucidation With Knowledge Enhanced Tree Searc](llm_nlp/boosting_llms_molecular_structure_elucidation_with_knowledge_enhanced_tree_searc.md)**

:   提出 K-MSE，通过构建分子子结构知识库补充 LLM 化学知识 + 设计分子-光谱评分器作为奖励模型 + MCTS 树搜索推理框架，在分子结构解析任务上将 GPT-4o-mini 和 GPT-4o 的性能提升超过 20%。

**[Brevity Is The Soul Of Sustainability Characterizing Llm Response Lengths](llm_nlp/brevity_is_the_soul_of_sustainability_characterizing_llm_response_lengths.md)**

:   系统研究 12 个 LLM 在 5 个数据集上的响应长度行为，发现 LLM 普遍生成远超必要的冗长回复（核心答案仅占 42%），并提出多种提示策略将响应长度缩短 25-88%、推理能耗降低 25-60%，同时保持甚至提升 ROUGE-L F1 质量。

**[Buzzword Understanding Ugc](llm_nlp/buzzword_understanding_ugc.md)**

:   研究 LLM 能否通过用户生成内容（UGC）理解中文网络流行语——构建首个中文网络流行语数据集 Cheer（含定义和相关UGC），提出 Ress 方法引导 LLM 模拟人类语言学习过程来生成流行语定义，揭示了 LLM 在流行语理解上的三大共性挑战。

**[Cadllm Cad Modeling From Text](llm_nlp/cadllm_cad_modeling_from_text.md)**

:   提出一个语言引导的 CAD 设计自动化框架——通过半自动数据标注流水线、Transformer CAD 生成器（TCADGen）和 LLM 增强模型（CADLLM）三个创新，从文本参数和外观描述自动生成 CAD 建模序列，在精度和效率上超越传统方法。

**[Can Input Attributions Explain Inductive Reasoning In In-Context Learning](llm_nlp/can_input_attributions_explain_inductive_reasoning_in_in-context_learning.md)**

:   设计受控的合成归纳推理任务评估 4 种输入归因方法解释 ICL 的能力，发现最简单的梯度范数常常最好，但所有方法在不同任务和模型规模上表现不一致且不稳定——ICL 的可解释性比预期更难。

**[Can Language Models Reason About Individualistic Human Values And Preferences](llm_nlp/can_language_models_reason_about_individualistic_human_values_and_preferences.md)**

:   提出"个体价值对齐"范式，构建 IndieValueCatalog 数据集（基于世界价值观调查 WVS 的 9.3 万真人数据），评估并训练语言模型根据个人价值表达陈述推理其在新情境下的价值判断，揭示前沿 LM 仅达 55%-65% 准确率且存在显著的跨群体不公平性。

**[Can Language Models Replace Programmers For Coding Repocod Says Not Yet](llm_nlp/can_language_models_replace_programmers_for_coding_repocod_says_not_yet.md)**

:   构建了 RepoCod——一个包含980个来自11个大型 Python 项目的复杂代码生成任务的基准，具有真实的仓库级依赖和平均314个开发者测试用例，揭示了即使最先进的 LLM 也仅能达到不到30%的 Pass@1，远未能替代程序员完成真实编码任务。

**[Can Large Language Models Address Open-Target Stance Detection](llm_nlp/can_large_language_models_address_open-target_stance_detection.md)**

:   提出开放目标立场检测（OTSD）任务——目标在训练时未见且不作为输入提供，系统评估了 GPT、Gemini、LLaMA、Mistral 四个系列共 8 个 LLM 在目标生成和立场检测两阶段的表现，发现 LLM 整体优于现有 TSE 方法，但在目标未显式出现时表现明显下降。

**[Can Llms Help Uncover Insights About Llms A Large-Scale Evolving Literature Anal](llm_nlp/can_llms_help_uncover_insights_about_llms_a_large-scale_evolving_literature_anal.md)**

:   本文提出半自动化文献分析管线，利用LLM从arXiv论文中自动抽取实验结果构建可持续更新的LLMEvalDB数据集（18127条记录/1737篇论文），并通过该数据集复现并扩展了关于CoT和ICL提示策略在不同任务类型上有效性的关键发现。

**[Can Llms Identify Critical Limitations Within Scientific Research A Systematic E](llm_nlp/can_llms_identify_critical_limitations_within_scientific_research_a_systematic_e.md)**

:   提出 LimitGen 基准，系统评估 LLM 识别科研论文局限性的能力，包含合成数据集（通过受控扰动创建）和人类标注数据集（ICLR 2025 评审），并通过 RAG 增强文献检索来提升 LLM 生成更具体和建设性反馈的能力。

**[Can Llms Interpret And Leverage Structured Linguistic Representations A Case Stu](llm_nlp/can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)**

:   本文系统评估了 LLM 利用抽象语义表示（AMR）进行下游任务的能力，发现 AMR 增强的 prompt 在长上下文任务（如对话摘要）中显著提升 Llama 3.1 零样本性能（余弦相似度从 66% 提升至 76%），但在短上下文任务中通常会降低性能。

**[Can Llms Understand Unvoiced Speech Exploring Emg-To-Text Conversion With Llms](llm_nlp/can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)**

:   本文提出了一种基于可训练 EMG 适配器模块的方法，将无声肌电图（EMG）信号映射到大语言模型（LLM）的输入嵌入空间，在闭合词汇无声 EMG 转文本任务中实现了 0.49 的词错误率（WER），仅需 6 分钟训练数据即比专用模型提升约 20%。

**[Can We Retrieve Everything All At Once Arm An Alignment-Oriented Llm-Based Retri](llm_nlp/can_we_retrieve_everything_all_at_once_arm_an_alignment-oriented_llm-based_retri.md)**

:   提出 ARM（Alignment-oriented Retrieval Method），通过在 LLM 解码过程中融合信息对齐（N-gram 约束解码）、结构对齐（MIP 求解器推理数据对象间关系）和自验证聚合三个模块，实现"一次检索所有"所需数据对象，在 Bird 和 OTT-QA 数据集上显著优于标准 RAG（最高 +5.2pt）和 agentic RAG/ReAct（最高 +19.3pt）。

**[Cer Confidence Enhanced Reasoning](llm_nlp/cer_confidence_enhanced_reasoning.md)**

:   提出置信度增强推理框架 CER——在 CoT 推理的每个中间步骤中量化关键 token（数学任务的数值/开放域的专有名词）的置信度，用步间置信度乘积评估整条推理链的可靠性，用置信度加权聚合替代简单多数投票，在数学和开放域任务上比 self-consistency 分别提升最高 7.4% 和 5.8%。

**[Character Level Understanding](llm_nlp/character_level_understanding.md)**

:   提出 TIPA（Token Internal Position Awareness）方法，通过在 tokenizer 词汇表上进行逆序字符预测训练，增强 LLM 对 token 内部字符结构和位置的感知能力，显著提升中文拼写纠错等字符级任务的表现。

**[Chronosense Exploring Temporal Understanding In Large Language Models With Time ](llm_nlp/chronosense_exploring_temporal_understanding_in_large_language_models_with_time_.md)**

:   提出 ChronoSense 基准，系统评估 LLM 在 Allen 13 种时间区间关系和时间算术任务上的表现，揭示模型在时间理解方面的显著不足及对记忆的依赖。

**[Circuit Compositions Modular Structures](llm_nlp/circuit_compositions_modular_structures.md)**

:   通过在 PCFG SET 数据集上识别 10 个组合性字符串编辑操作的电路（circuits），研究 Transformer 中功能相关电路之间的模块化关系，发现功能相似的电路具有显著的节点重叠和跨任务忠实度，且电路可以通过集合运算（并集）组合以表示超出单个电路能力的更复杂功能。

**[Circuit Stability Characterizes Language Model Generalization](llm_nlp/circuit_stability_characterizes_language_model_generalization.md)**

:   本文提出"电路稳定性"作为评估语言模型泛化能力的新方法，通过数学形式化定义软电路和电路等价性，在算术推理、布尔表达式和体育理解三个案例研究中证明电路稳定性可以预测和刻画泛化行为。

**[Classifying Unreliable Narrators](llm_nlp/classifying_unreliable_narrators.md)**

:   借用叙事学理论定义三种不可靠叙事者类型，构建专家标注数据集 TUNa，系统评估 LLM 在零样本、少样本、微调和课程学习设定下的分类能力，发现该任务极具挑战性且课程学习对小模型有显著提升。

**[Codetool Enhancing Programmatic Tool Invocation Of Llms Via Process Supervision](llm_nlp/codetool_enhancing_programmatic_tool_invocation_of_llms_via_process_supervision.md)**

:   提出 CodeTool，一种逐步代码生成框架，通过即时奖励（On-the-spot Reward）和潜在奖励（Latent Reward）两种过程奖励机制引导 LLM 选择最优的工具调用路径，在 StableToolBench 和 RestBench-TMDB 上显著超越现有方法。

**[Cognibench Cognitive Faithfulness](llm_nlp/cognibench_cognitive_faithfulness.md)**

:   借鉴法律领域间接证据认定标准，提出分层评估框架和 CogniBench 数据集，首次系统性地定义和评估 LLM 在认知性陈述（推理、评价、解释）中的忠实度问题，并训练 CogniDet 检测器实现事实与认知幻觉的同时检测。

**[Cogsteer Cognition-Inspired Selective Layer Intervention For Efficiently Steerin](llm_nlp/cogsteer_cognition-inspired_selective_layer_intervention_for_efficiently_steerin.md)**

:   利用认知科学中的眼动数据分析 LLM 各层行为，发现中间层与人类注视相关性最高且最适合语义干预，提出 CogSteer 框架——仅微调最优单层（约 3% 参数）即可达到或超过全层微调的效果，在 GLUE/毒性控制任务上有效。

**[Coin Flips Bayesian](llm_nlp/coin_flips_bayesian.md)**

:   通过受控的有偏硬币抛掷实验，证明 LLM 在获得足够的上下文示例后能以贝叶斯方式更新其先验，但初始先验通常存在系统性偏差（偏向正面），且注意力幅度对贝叶斯推理影响甚微。

**[Combining The Best Of Both Worlds A Method For Hybrid Nmt And Llm Translation](llm_nlp/combining_the_best_of_both_worlds_a_method_for_hybrid_nmt_and_llm_translation.md)**

:   提出基于源句特征的NMT与LLM混合翻译调度策略（PPLT与JDM），在保持翻译质量最优的同时将LLM调用比例降至约25-30%，大幅减少计算开销。

**[Compositional Generalization Instruction](llm_nlp/compositional_generalization_instruction.md)**

:   提出 Ordered CommonGen 基准，通过要求 LLM 按指定顺序生成包含所有概念的句子，同时评估组合泛化与指令遵循能力，在 36 个 LLM 上发现即使最强模型也仅能达到约 75% 的有序覆盖率。

**[Computation Mechanism Behind Llm Position Generalization](llm_nlp/computation_mechanism_behind_llm_position_generalization.md)**

:   揭示 LLM 注意力 logit 学习了位置相关性和语义重要性的近似算术加法解耦（$W_{i,j} \approx f(\mathbf{q}, i-j) + g(\mathbf{q}, \mathbf{k})$，线性相关 0.959），发现了使这种解耦成立的中间表示模式，并用此解释了 LLM 的位置排列容忍性和长度泛化能力。

**[Conceptual Knowledge Org](llm_nlp/conceptual_knowledge_org.md)**

:   通过构建首个意大利语下位类别心理语言学数据集（187 个基本类别），系统对比了人类和 LLM 在下位概念层级上的类别组织结构，发现两者的对齐度较低但在不同语义领域存在显著差异。

**[Conformity In Large Language Models](llm_nlp/conformity_in_large_language_models.md)**

:   将心理学中的 Asch 从众实验范式迁移到 LLM 上，系统研究了 LLM 的从众行为（conformity），发现所有模型都会受多数意见影响改变答案，且不确定性越高越容易从众，并提出 Devil's Advocate 和 Question Distillation 两种干预方法有效缓解从众效应。

**[Consistencychecker Tree Evaluation](llm_nlp/consistencychecker_tree_evaluation.md)**

:   ConsistencyChecker 提出基于自一致性树（self-consistency tree）的无参考 LLM 评估框架，通过构建可逆变换的树状多步路径（如多语言往返翻译、代码等价重写），量化模型在迭代变换中的语义/功能保持能力，动态生成 benchmark 从根源消除数据泄露，且与 WMT 2024 权威排名的相关性 r > 0.7，证明无需配对数据即可可靠评估 LLM 泛化能力。

**[Contrastive Perplexity Controlled Gen](llm_nlp/contrastive_perplexity_controlled_gen.md)**

:   提出基于原型对比困惑度（Contrastive Perplexity, CP）的框架，通过构造语义相似但毒性属性不同的正负样本对，在困惑度空间中进行对比学习来微调 LLM，实现显著的毒性降低（Mistral-7b 毒性从 33.1% 降至 4.3%）且几乎不影响下游任务性能。

**[Contrastive Prompting Embeddings](llm_nlp/contrastive_prompting_embeddings.md)**

:   提出对比提示（Contrastive Prompting, CP）方法，通过引入辅助提示（引导编码非核心信息）并在推理时与正常提示的激活值做对比减法，过滤掉停用词等无关语义，使 LLM 的句子嵌入更聚焦核心语义，在 STS 和分类任务上一致提升现有提示方法。

**[Convert Language Model Into A Value-Based Strategic Planner](llm_nlp/convert_language_model_into_a_value-based_strategic_planner.md)**

:   提出 straQ* 框架，将 LLM 转化为基于 Q-learning 的策略规划器，用平均 logit 作为 Q 值实现策略级 MDP，在情感支持对话中基于长期回报选择最优策略，超越直接推理、CoT 和微调等基线。

**[Cool-Fusion Fuse Large Language Models Without Training](llm_nlp/cool-fusion_fuse_large_language_models_without_training.md)**

:   提出 Cool-Fusion，一种无需任何训练即可融合异构 LLM 的方法，通过在文本段粒度上让多个模型互相评估和重排序生成内容，在 GSM8K 上相对最强源模型提升 17.4% 准确率。

**[Cosmic Generalized Refusal Direction Identification In Llm Activations](llm_nlp/cosmic_generalized_refusal_direction_identification_in_llm_activations.md)**

:   提出 COSMIC（Cosine Similarity Metrics for Inversion of Concepts），一种基于余弦相似度的自动化方向选择框架，无需依赖模型输出 token 或预定义的拒绝模板即可在 LLM 激活空间中识别拒绝方向，在对抗场景和弱对齐模型中仍能有效执行拒绝引导。

**[Cot-Based Synthesizer Enhancing Llm Performance Through Answer Synthesis](llm_nlp/cot-based_synthesizer_enhancing_llm_performance_through_answer_synthesis.md)**

:   提出 CoT-based Synthesizer——一种新的推理扩展策略，通过利用 CoT 推理分析多个候选回答的互补信息来合成更优的最终答案，即便所有候选回答都是错误的也能综合出正确答案，在 MATH500 上对 Llama3-8B 提升 11.8%、对 GPT-4o 提升 10.3%。

**[Cross Model Transferability Sv](llm_nlp/cross_model_transferability_sv.md)**

:   提出 L-Cross Modulation 方法，通过简单线性变换将一个 LLM 的概念方向向量（steering vectors）迁移到另一个 LLM 中实现行为控制，发现三个关键结论：(1) 跨模型 SV 迁移有效；(2) 不同概念共享同一变换矩阵；(3) 小模型的 SV 可以控制大模型（弱到强迁移）。

**[Cuckoo An Ie Free Rider Hatched By Massive Nutrition In Llms Nest](llm_nlp/cuckoo_an_ie_free_rider_hatched_by_massive_nutrition_in_llms_nest.md)**

:   本文提出 Next Tokens Extraction (NTE) 范式，将 LLM 预训练数据中的下一个 token 预测转化为 BIO 标注的抽取任务，利用 C4 和 TuluV3 共 1.026 亿实例预训练 RoBERTa 标注器（Cuckoo），在少样本信息抽取任务上全面超越现有 IE 预训练模型。

**[Cultural Learning-Based Culture Adaptation Of Language Models](llm_nlp/cultural_learning-based_culture_adaptation_of_language_models.md)**

:   提出 CLCA 框架，借鉴文化学习理论，通过模拟社会交互生成文化适配对话数据，结合意图理解进行多任务训练，在 World Values Survey 上显著提升多种 LLM 的文化价值观对齐。

**[Culture Is Not Trivia Sociocultural Theory For Cultural Nlp](llm_nlp/culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)**

:   提出基于社会文化理论的文化 NLP 框架，批判现有将文化简化为"常识琐事"的做法，主张文化应被理解为动态的、情境化的社会建构过程，而非静态的知识标签。

**[Deal Decoding Time Alignment](llm_nlp/deal_decoding_time_alignment.md)**

:   DeAL 将 LLM 对齐问题重新形式化为解码时的启发式搜索问题，在推理阶段利用可定制的奖励函数（包括程序化约束和参数化 reward model）引导 token 选择，实现了灵活的多目标对齐且可与 RLHF 互补叠加。

**[Dehumanizing Metaphors Immigration](llm_nlp/dehumanizing_metaphors_immigration.md)**

:   提出结合 LLM 词级隐喻检测与 SBERT 篇章级语义关联的计算框架，在 40 万条美国移民推文上揭示保守派更多使用去人化隐喻、但生物类隐喻对自由派的用户互动效应更强的复杂图景。

**[Deontological Keyword Bias](llm_nlp/deontological_keyword_bias.md)**

:   本文揭示LLM存在"义务论关键词偏见"(DKB)——当提示中包含"must"、"ought to"等模态义务表达时，模型会将超过90%的常识场景误判为义务，并提出基于少样本示例与推理提示的去偏策略。

**[Derta Decoupled Refusal](llm_nlp/derta_decoupled_refusal.md)**

:   发现标准安全微调数据存在"拒绝位置偏差"——模型只学会在回答开头拒绝，中途发现不安全时无法中断。提出 DeRTa（Decoupled Refusal Training），通过"有害前缀+安全拒绝"的 MLE 训练和在每个位置模拟"从有害到安全"转换的 RTO 训练，让 LLM 能在回答的任何位置感知到不安全时拒绝，在六种攻击场景下超越 GPT-4 和 LLaMA3-Instruct。

**[Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Models](llm_nlp/detecting_referring_expressions_in_visually_grounded_dialogue_with_autoregressiv.md)**

:   提出使用自回归模型在视觉对话中检测指代表达（referring expressions），将 RE 检测建模为序列标注任务的自回归变体，在 PhotoBook 和 MeetUp 数据集上实现 SOTA。

**[Dice-Bench Evaluating The Tool-Use Capabilities Of Large Language Models In Mult](llm_nlp/dice-bench_evaluating_the_tool-use_capabilities_of_large_language_models_in_mult.md)**

:   提出 DICE-Bench 基准评估 LLM 在多轮对话中的工具使用能力，涵盖工具发现、调用、错误恢复和组合使用四个维度，发现即使最强模型在复杂多轮场景下也仅达 ~60% 成功率。

**[Difflm Controllable Synthetic Data Generation Via Diffusion Language Models](llm_nlp/difflm_controllable_synthetic_data_generation_via_diffusion_language_models.md)**

:   提出 DiffLM，将扩散模型应用于文本合成数据生成，通过引导函数实现对生成文本属性的精细控制，在多个下游任务上生成的合成数据质量超越 GPT-4 和传统增强方法。

**[Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence](llm_nlp/direct_confidence_alignment_aligning_verbalized_confidence_with_internal_confide.md)**

:   通过 DPO 将 LLM 的外化置信度（文字表达的概率）与内部置信度（token 概率）对齐，提升模型的透明度和可信度，Gemma-2-9B 上表现最佳。

**[Disco Device-Server Collaborative Llm-Based Text Streaming Services](llm_nlp/disco_device-server_collaborative_llm-based_text_streaming_services.md)**

:   提出 DiSCo 设备-服务器协作 LLM 文本流服务框架，端侧小模型快速生成初始 token 减少首 token 延迟（TTFT），服务端大模型并行处理后替换，实现低延迟高质量的文本流式服务。

**[Disentangle Memory Reasoning](llm_nlp/disentangle_memory_reasoning.md)**

:   提出将 LLM 的推理过程显式分解为"记忆回忆"和"逻辑推理"两个步骤——引入 `<memory>` 和 `<reason>` 两个可学习特殊 token 标记每步是知识回忆还是逻辑推理，用双 LLM 框架生成训练数据后 LoRA 微调，在 StrategyQA/CommonsenseQA/TruthfulQA 上提升性能并增强可解释性，8B 模型在 TruthfulQA 上超越 GPT-4o。

**[Dive Moe Reconstruction](llm_nlp/dive_moe_reconstruction.md)**

:   提出 DIVE，一种将 Dense LLM 重构为 MoE 架构的方法，核心洞察是不同领域的校准数据集会让结构化剪枝产生不同的剪枝结果，利用这种多样性构建领域特异的专家，配合高效的两阶段重训练（router dense训练 + expert LoRA稀疏训练），在仅调不到 1% 参数的情况下实现优于现有剪枝和 MoE 重构方法的效果。

**[Diversity Data Augmentation](llm_nlp/diversity_data_augmentation.md)**

:   提出 DoAug 框架，通过 SFT+DPO 微调 LLM 释义器并结合核心集选择与多样性采样，在保持语义一致性的同时显著提升增强数据集的多样性，在 12 个数据集上平均性能提升 10.52%，超出次优基线 3.76 个百分点。

**[Do Language Models Mirror Human Confidence?](llm_nlp/do_language_models_mirror_human_confidence_exploring_psychological_insights_to_a.md)**

:   探索 LLM 的置信度是否呈现与人类类似的心理学模式（如过度自信效应、达克效应），在 MMLU 等基准上发现 LLM 确实存在类人的过度自信偏差，但在某些维度上也有独特模式。

**[Do Language Models Understand Honorific Systems in Javanese?](llm_nlp/do_language_models_understand_honorific_systems_in_javanese.md)**

:   首次系统评估 LLM 对爪哇语敬语系统的理解能力，构建专门测试集发现即使是最强模型对复杂敬语层级的掌握也仅约 60% 准确率。

**[Do Language Models Understand the Cognitive Tasks Given to Them?](llm_nlp/do_language_models_understand_the_cognitive_tasks_given_to_them_investigations_w.md)**

:   用 N-back 任务系统分析 LLM 认知任务理解能力，发现性能低下的主因是任务理解不足而非工作记忆限制，挑战了 LLM 工作记忆容量约 3 的结论。

**[Do Large Language Models Perform Latent Multi-Hop Reasoning Without Exploiting S](llm_nlp/do_large_language_models_perform_latent_multi-hop_reasoning_without_exploiting_s.md)**

:   构建无快捷方式的评估数据集 SOCRATES，系统评估 41 个 LLM 在潜在多跳推理中的真实能力，发现模型在国家类桥接实体上可达 80% 组合率，但年份类仅约 5%。

**[Do LLMs Give Psychometrically Plausible Responses in Educational Assessments?](llm_nlp/do_llms_give_psychometrically_plausible_responses_in_educational_assessments.md)**

:   从心理测量学角度评估 18 个 LLM 在教育评估中的表现，发现大模型过度自信且无法预测干扰项选择模式。

**[Does Time Have Its Place Temporal Heads Where Language Models Recall Time-Specif](llm_nlp/does_time_have_its_place_temporal_heads_where_language_models_recall_time-specif.md)**

:   通过 EAP-IG 电路分析在 Llama-2/Qwen/Phi-3 中发现了专门处理时间条件知识的"时间头"（Temporal Heads），消融这些头只降低时间知识准确率（降 3-9%）而不影响时间无关知识和通用 QA，并展示了通过注入时间头激活值实现选择性时间知识编辑的可能性。

**[Dynamic Knowledge Integration for Evidence-Driven Counter-Argument Generation](llm_nlp/dynamic_knowledge_integration_for_evidence-driven_counter-argument_generation_wi.md)**

:   提出动态网络知识检索框架提升反驳论证生成质量，构建新评估数据集并用 LLM-as-Judge 进行评估。

**[Dynamic Parallel Tree Search For Efficient Llm Reasoning](llm_nlp/dynamic_parallel_tree_search_for_efficient_llm_reasoning.md)**

:   提出DPTS（Dynamic Parallel Tree Search）框架，通过并行流水线（Parallelism Streamline）解决树搜索中路径频繁切换难以并行化的问题，通过搜索与转换机制（Search and Transition Mechanism）的早停和深度搜索策略减少对低置信度路径的冗余探索，在Qwen-2.5和Llama-3上实现2-4倍推理加速，同时保持或超越MCTS等现有算法的推理准确率。

**[Eclm Entity Level Language Model Spoken Language Understanding](llm_nlp/eclm_entity_level_language_model_spoken_language_understanding.md)**

:   提出 ECLM 框架，将 LLM 应用于多意图口语理解：通过将 token 级槽填充转化为实体识别任务解决序列对齐问题，引入"意图链"（Chain of Intent）实现逐步多意图识别，在 MixATIS 和 MixSNIPS 上大幅超越 SOTA 基线。

**[Editext Diffusion Text Editing](llm_nlp/editext_diffusion_text_editing.md)**

:   提出 EdiText，基于嵌入扩散模型（LD4LG）的可控文本编辑框架，通过将 SDEdit 技术从图像域迁移到文本域实现粗粒度编辑（控制加噪时间步），并创新性地利用自条件化（self-conditioning）机制实现细粒度编辑（将参考文本嵌入注入为去噪条件），两者结合实现从粗到细的多粒度文本属性编辑。

**[Efficient Ensemble For Fine-Tuning Language Models On Multiple Datasets](llm_nlp/efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)**

:   提出 EnsembleLoRA——一种面向多数据集微调的高效集成方法，利用一阶 Taylor 近似快速估计任务亲和度将数据集分组，为每组训练一个 adapter 后加权组合，在 10 个 SuperGLUE 任务上以仅 9% 额外计算代价将 QLoRA 的平均测试准确率提升 10%。

**[Eli-Why Evaluating The Pedagogical Utility Of Language Model Explanations](llm_nlp/eli-why_evaluating_the_pedagogical_utility_of_language_model_explanations.md)**

:   构建了包含 13.4K "Why" 问题的 ELI-Why 基准，通过两项人类研究发现 GPT-4 生成的面向不同教育水平的解释仅 50% 能匹配目标年级（人工策划达 79%），且对学习者信息需求的满足度比人类答案低 20%。

**[Enabling Llm Knowledge Analysis Via Extensive Materialization](llm_nlp/enabling_llm_knowledge_analysis_via_extensive_materialization.md)**

:   本文提出通过递归查询和结果整合将 LLM 的事实知识大规模物化为知识库的方法论，构建了包含 1.01 亿三元组、290 万实体的 GPTKB，首次全面分析了 GPT-4o-mini 知识的规模、准确性、偏见、时效性和一致性。

**[Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding](llm_nlp/enhancing_input-label_mapping_in_in-context_learning_with_contrastive_decoding.md)**

**[Enhancing Spoken Discourse Modeling In Language Models Using Gestural Cues](llm_nlp/enhancing_spoken_discourse_modeling_in_language_models_using_gestural_cues.md)**

:   提出将手势动作序列（3D 人体运动数据）通过 VQ-VAE 编码为离散 gesture token，再经特征对齐映射到语言模型输入空间，用于增强口语篇章建模；在三类篇章标记（话语连接词、量词、立场标记）的文本填充任务上验证了手势信息对口语篇章理解的互补价值。

**[Enhancing Transformation from Natural Language to Signal Temporal Logic using LLMs](llm_nlp/enhancing_transformation_from_natural_language_to_signal_temporal_logic_using_ll.md)**

:   用 LLM 改进自然语言到信号时序逻辑（STL）的转换，提出两阶段方法（语义解析+逻辑生成）。

**[Episodic Grounding Experience](llm_nlp/episodic_grounding_experience.md)**

:   提出一个 weak-to-strong episodic grounding 框架，利用 MCTS 收集结构化经验数据，通过行为比率蒸馏将小模型的 episodic grounding 能力迁移到大模型，结合 DPO 优化实现从成功和失败经验中学习，在物理规划任务上超越 GPT-4o 等 SOTA 模型 3.45%。

**[Erm Prompt Optimization Memory](llm_nlp/erm_prompt_optimization_memory.md)**

:   提出 ERM 方法，通过指导性元提示生成带详细解题过程的 exemplar 来增强 feedback 质量，并引入 Feedback Memory 和 Exemplar Factory 两种长期记忆机制来高效存储和复用历史反馈与示例，在多个任务上以约一半的优化步数超越了 SOTA prompt 优化方法。

**[Escapebench Creative Agent](llm_nlp/escapebench_creative_agent.md)**

:   提出EscapeBench密室逃脱游戏基准（36个场景、3种难度）评估LM agent的创造性智能，并设计EscapeAgent通过Foresight（工具使用假设生成）和Reflection（未解任务追踪）模块将提示依赖降低约50%，但仍远落后于人类。

**[Evaluating Implicit Bias in LLMs by Attacking from a Psychometric Perspective](llm_nlp/evaluating_implicit_bias_in_large_language_models_by_attacking_from_a_psychometr.md)**

:   借鉴心理测量学中的隐式联想测验（IAT）方法评估 LLM 的隐式偏见，发现模型存在系统性偏见且与人类偏见模式部分相似。

**[Evaluating Lms Synthetic Data Gen](llm_nlp/evaluating_lms_synthetic_data_gen.md)**

:   提出 AgoraBench 基准，系统评估 6 个 LLM 在 3 个领域×3 种数据生成方式下的数据生成能力，通过训练 99 个学生模型发现：LLM 的数据生成能力与问题求解能力不直接相关，GPT-4o 在实例生成上最强而 Claude-3.5-Sonnet 在质量增强上最强。

**[Evopatient Standardized Patient](llm_nlp/evopatient_standardized_patient.md)**

:   EvoPatient 提出了一个多智能体协同进化框架，通过患者 Agent 和医生 Agent 之间的自主模拟对话，让 LLM 无需人工监督即可学会模拟标准化病人（SP），在需求对齐度上超过现有推理方法 10%+。

**[Explain-then-Process: Using Grammar Prompting to Enhance Grammatical Acceptability](llm_nlp/explain-then-process_using_grammar_prompting_to_enhance_grammatical_acceptabilit.md)**

:   提出先解释语法规则再判断可接受性的 Explain-then-Process 策略，在语法可接受性判断任务上超越直接 prompting 和 CoT。

**[Explica Evaluating Explicit Causal Reasoning In Large Language Models](llm_nlp/explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)**

:   提出 ExpliCa 数据集（4800 条问题，含因果和时间连接词），首次整合因果和时间关系评估并配以众包人类评分，发现即使顶级模型准确率也难超 0.80，且模型系统性地将时间关系误判为因果关系。

**[Exploring Explanations Improves The Robustness Of In-Context Learning](llm_nlp/exploring_explanations_improves_the_robustness_of_in-context_learning.md)**

:   提出 X²-ICL 框架，通过在上下文学习的示例中为所有可能的标签（而非仅观测标签）生成解释推理路径，系统性地探索隐变量推理空间，从而显著提升 ICL 在分布外（OOD）数据上的鲁棒性——在 5 个 LLM 上的 8 个 OOD 数据集中，X²-ICL 在 6-8 个上超越 ICL 和 X-ICL。

**[Exploring Graph Representations Of Logical Forms For Language Modeling](llm_nlp/exploring_graph_representations_of_logical_forms_for_language_modeling.md)**

:   提出 GFoLDS，一种在 DMRS 逻辑形式图表示上预训练的图 Transformer 语言模型，并提出"语言知识催化假说"(LKCH)：逻辑形式语言模型几乎立刻学会基础语言现象，进而加速复杂模式学习，在相同数据量下大幅超越 BERT。

**[Exploring The Potential Of Llms As](llm_nlp/exploring_the_potential_of_llms_as.md)**

:   提出 HiCUPID，首个全面满足个性化 AI 助手五大需求（用户信息遵循、隐含信息理解、多信息推理、长上下文建模、主动性回复）的基准，含 1,250 用户 × 25 人格 × 10 日程 + Llama-3.2 自动评估模型。

**[Foodtaxo Generating Food Taxonomies With Large Language Models](llm_nlp/foodtaxo_generating_food_taxonomies_with_large_language_models.md)**

:   提出 FoodTaxo，基于 Llama-3 的迭代自底向上分类法生成与补全算法，利用 CoT 提示 + RAG 检索 + NLI 验证三阶段流程，从已知叶节点概念出发逐步构建层次化 taxonomy；在五个基准数据集上与 TacoPrompt 等 SOTA 方法竞争，同时通过 reference-free 指标和消融实验揭示了非叶节点放置这一根本性瓶颈。

**[Foundation Lm Single Cell Survey](llm_nlp/foundation_lm_single_cell_survey.md)**

:   首篇从语言建模视角系统综述单细胞生物学基础语言模型的工作，将现有模型划分为 PLM（从头预训练）和 LLM（利用已有大模型）两大类，全面分析了数据 tokenization 策略、预训练/微调范式以及下游任务。

**[From Data To Knowledge Evaluating How Efficiently Language Models Learn Facts](llm_nlp/from_data_to_knowledge_evaluating_how_efficiently_language_models_learn_facts.md)**

:   首次直接研究事实在预训练数据中出现频次与 LLM 能否回忆该事实之间的关系，提出两种样本效率指标，发现不同架构/规模的模型在高频事实上表现相似但在低频事实上差异显著——低频事实的学习能力是区分模型样本效率的关键。

**[From Neurons To Semantics Evaluating Cross-Linguistic Alignment Capabilities Of ](llm_nlp/from_neurons_to_semantics_evaluating_cross-linguistic_alignment_capabilities_of_.md)**

:   提出基于神经元激活状态的跨语言对齐评估框架 NeuronXA，利用 FFN 层神经元状态作为语言的内在表征来衡量多语言 LLM 的跨语言对齐能力，仅需 100 对平行句子即可实现与下游任务性能 0.9556 的皮尔逊相关。

**[From Selection To Generation A Survey](llm_nlp/from_selection_to_generation_a_survey.md)**

:   首篇系统综述 LLM 时代的主动学习（Active Learning），提出以 Querying（选择/生成）和 Annotation（标注）为核心的分类体系，全面梳理 LLM 如何变革传统主动学习的选择-标注流程。

**[From Selection To Generation A Survey Of Llm-Based Active Learning](llm_nlp/from_selection_to_generation_a_survey_of_llm-based_active_learning.md)**

:   首篇系统梳理 LLM 时代主动学习的综述，提出以 Querying（从传统选择到 LLM 生成）和 Annotation（从人工标注到 LLM 标注）为双轴的统一分类体系，覆盖查询策略、标注方案、停止准则、AL 范式和应用领域。

**[Gapo Multi Objective Alignment](llm_nlp/gapo_multi_objective_alignment.md)**

:   提出GAPO，一种基于梯度自适应缩放的多目标策略优化方法，利用多梯度下降算法(MGDA)结合梯度归一化，平衡LLM在帮助性和无害性等冲突目标间的权衡，并通过P-GAPO支持用户偏好驱动的Pareto前沿生成。

**[Gapo Preferential Prompt](llm_nlp/gapo_preferential_prompt.md)**

:   提出 GAPO（Generative Adversarial Policy Optimization）框架，将 GAN 的对抗训练机制与 PPO 结合，使用 encoder-only 奖励模型替代传统 decoder-only 架构，通过"Preferential Prompt"（修改 prompt 中的约束而非 response）的新范式来增强 LLM 对细粒度约束的理解和遵循能力，在 IFEval 和产品描述生成任务上大幅超越 DPO/KTO/SimPO 等基线。

**[Generative Psycholexical Approach For Constructing Value](llm_nlp/generative_psycholexical_approach_for_constructing_value.md)**

:   提出生成式心理词汇方法（GPLA），自动化构建面向LLM的五因素价值体系（社会责任、冒险性、规则遵循、自我效能、理性），在结构效度、安全预测和价值对齐上优于经典Schwartz人类价值体系。

**[Genetic Instruct Scaling Up Synthetic Generation Of Coding Instructions For Larg](llm_nlp/genetic_instruct_scaling_up_synthetic_generation_of_coding_instructions_for_larg.md)**

:   提出 Genetic-Instruct 算法，借鉴进化算法的交叉和变异操作，从 512 个种子指令扩展生成 750 万+高质量编码指令，使用 Instructor-LLM/Coder-LLM/Judge-LLM 三角色流水线，训练后的模型在代码生成基准上超越 Self-Instruct 和 Evol-Instruct。

**[GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Substitution](llm_nlp/genknowsub_improving_modularity_and_reusability_of_llms_through_general_knowledg.md)**

:   提出 GenKnowSub 框架通过通用知识替换改善 LLM 的模块化和可重用性，实现知识组件的即插即用。

**[Geocultural Grounded Llm](llm_nlp/geocultural_grounded_llm.md)**

:   研究 RAG/搜索增强技术对 LLM 文化意识的影响——搜索增强显著提升了文化命题知识的选择题表现，但也增加了刻板印象风险，且在开放式文化流畅性的人工评估中改进不显著，揭示了"文化知识"和"文化流畅性"的本质区别。

**[Geometric Compositionality Lifetime](llm_nlp/geometric_compositionality_lifetime.md)**

:   通过将数据集的组合性程度与语言模型表示的非线性内在维度(I_d)和线性有效维度(d)联系起来，揭示了一个形式-意义二分：非线性 I_d 编码有意义的组合语义复杂度，而线性 d 编码表面词形复杂度；该对应关系在训练过程中随语言能力涌现而建立。

**[Gift-Sw Gaussian Noise Injected Fine-Tuning Of Salient Weights For Llms](llm_nlp/gift-sw_gaussian_noise_injected_fine-tuning_of_salient_weights_for_llms.md)**

:   提出 GIFT-SW，一种新型参数高效微调方法：仅更新权重矩阵中的"显著列"(salient columns)，同时对非显著列注入高斯噪声，在相同计算预算下超越全参微调和 LoRA/DoRA 等现代 PEFT 方法。

**[Goal Hijacking Attack](llm_nlp/goal_hijacking_attack.md)**

:   本文提出POUGH方法，通过高效的渐进式优化算法和两种语义引导的提示组织策略（采样策略+排序策略），实现了对LLM的高效通用目标劫持攻击，在四个开源LLM和十种恶意目标响应上平均攻击成功率达93.41%。

**[Good Natural Language Prompt](llm_nlp/good_natural_language_prompt.md)**

:   通过元分析150+篇prompting相关论文和博客，提出一个以属性为中心、以人为中心的prompt质量评估框架，涵盖6个维度21个属性，并发现单属性增强往往比多属性组合更有效。

**[Gorp Continual Gradient Projection](llm_nlp/gorp_continual_gradient_projection.md)**

:   GORP 提出将全秩参数和 LoRA 低秩参数的梯度统一投影到低秩梯度子空间中联合更新，利用 Adam 一阶矩隐式构建跨任务共享梯度空间来缓解灾难性遗忘，在 T5 和 LLaMA2 上持续学习性能接近多任务联合训练上界。

**[Graph Descriptive Order Llm](llm_nlp/graph_descriptive_order_llm.md)**

:   首次系统研究了图描述顺序（BFS、DFS、PageRank、PPR）对LLM解决图推理问题的影响，发现有序描述显著优于随机描述，且不同任务偏好不同的排列策略。

**[Hft Half Fine-Tuning For Large Language Models](llm_nlp/hft_half_fine-tuning_for_large_language_models.md)**

:   本文提出Half Fine-Tuning (HFT)，在每轮微调中随机冻结一半参数、只更新另一半，不改变模型架构的情况下显著缓解灾难性遗忘问题，同时在下游任务上取得与FFT相当甚至更好的性能，并减少约30%的训练时间。

**[How Llms Comprehend Temporal Meaning In Narratives A Case Study In Cognitive Eva](llm_nlp/how_llms_comprehend_temporal_meaning_in_narratives_a_case_study_in_cognitive_eva.md)**

:   通过认知语言学中的"体貌"（aspect）实验范式，系统评估LLM是否能像人类一样理解叙事中的时间意义，发现LLM过度依赖原型搭配、体貌判断不一致、缺乏远距因果推理能力，揭示其叙事理解与人类存在本质性差异。

**[How Numerical Precision Affects Arithmetical Reasoning Capabilities Of Llms](llm_nlp/how_numerical_precision_affects_arithmetical_reasoning_capabilities_of_llms.md)**

:   从电路复杂度理论出发，严格证明低精度（如 int4/int8）Transformer 在迭代加法和整数乘法上需要超多项式规模才能求解，而标准精度（float32）Transformer 仅需常数深度+多项式宽度即可高效求解三类算术任务，并在 LLaMA-3.1-8B 上实验验证了精度对算术能力的关键影响。

**[Human Nlp Cooperation Survey](llm_nlp/human_nlp_cooperation_survey.md)**

:   首次系统性地综述了人与NLP模型之间的合作范式，提出基于"谁为最终决策负责"的三类合作形式化分类体系（顺序合作、分诊合作、联合合作），并借鉴Grice合作原则定义了人机合作的基本原则，为后续研究提供了统一视角。

**[Humt Dumt Measuring And Controlling Human-Like Language In Llms](llm_nlp/humt_dumt_measuring_and_controlling_human-like_language_in_llms.md)**

:   提出基于 GPT-2 对数概率比的文本人类化语气度量 HumT 及其社会感知泛化版 SocioT，在 40 万+偏好样本上发现用户普遍偏好更低人类化的 LLM 输出且人类化语气与社交亲近（r=0.87）、低地位（r=-0.80）、女性化（r=0.47）强相关，进而通过仅 500 对偏好数据的 DPO 微调（DumT）有效降低人类化程度而不损模型性能。

**[Hygenar An Llm-Driven Hybrid Genetic Algorithm For Few-Shot Grammar Generation](llm_nlp/hygenar_an_llm-driven_hybrid_genetic_algorithm_for_few-shot_grammar_generation.md)**

:   构建 540 个挑战的文法生成数据集，设计 6 种评测指标，提出基于 LLM 驱动的混合遗传算法 HyGenar，显著提升 LLM 从少量示例生成 BNF 文法的能力。

**[Idea Enhancing The Rule Learning Ability Of Large Language Model Agent Through I](llm_nlp/idea_enhancing_the_rule_learning_ability_of_large_language_model_agent_through_i.md)**

:   提出 RULEARN benchmark（300 个手工交互式文字环境谜题，涵盖三类场景）和 IDEA 框架（溯因假设生成→演绎计划验证→归纳反馈修正的迭代循环），在 GPT-4o 上达到 50.33% 成功率（+7% vs ReAct baseline），但仍远低于人类 63.33%，细粒度人类评估揭示了 LLM 在假设修正阶段的根本瓶颈。

**[If Eleanor Rigby Had Met ChatGPT: A Study on Loneliness in a Post-LLM World](llm_nlp/if_eleanor_rigby_had_met_chatgpt_a_study_on_loneliness_in_a_post-llm_world.md)**

:   研究 LLM 聊天伴侣对孤独感的影响，发现 ChatGPT 交互可短期缓解孤独但长期效果有限，且可能减少人际互动动力。

**[Impossibility Fair Llms](llm_nlp/impossibility_fair_llms.md)**

:   系统分析了多种技术公平性框架（fairness through unawareness、group fairness、fair representations、multi-sided fairness等）在通用LLM上的适用性，论证了每种框架要么在逻辑上无法扩展到通用AI场景、要么在实践中不可行——主要源于非结构化训练数据的敏感属性不可剥离、用例/人群组合的组合爆炸、以及公平性不具备可组合性。

**[Improve Language Model And Brain Alignment Via Associative Memory](llm_nlp/improve_language_model_and_brain_alignment_via_associative_memory.md)**

:   通过模拟人类联想记忆（用相关概念扩展输入文本）提升语言模型与大脑 fMRI 活动的对齐度，发现联想记忆增强在内侧颞叶等记忆相关脑区效果最显著，并构建 Association 数据集通过 SFT 让 LLM 学会生成联想内容。

**[Improving Contextual Faithfulness Of Large Language Models Via Retrieval Heads-I](llm_nlp/improving_contextual_faithfulness_of_large_language_models_via_retrieval_heads-i.md)**

:   本文发现LLM中的"检索头"（retrieval heads）与长文本问答的上下文忠实度高度相关，据此提出Rhio框架：通过遮蔽检索头生成不忠实样本、引入控制令牌进行忠实度感知调优、再利用对比解码增强忠实度，在7B和13B模型上均超越GPT-4o。

**[Improving Preference Extraction In Llms By Identifying Latent Knowledge Through ](llm_nlp/improving_preference_extraction_in_llms_by_identifying_latent_knowledge_through_.md)**

:   本文提出使用线性分类探针（classifying probes）结合对比对（contrast pairs）来提取LLM的隐含偏好判断，在LLM-as-Judge任务中持续优于传统的生成式评估方法，且监督探针甚至超越微调评估器，同时保持类似的计算成本。

**[Infinisst Simultaneous Translation Of Unbounded Speech With Large Language Model](llm_nlp/infinisst_simultaneous_translation_of_unbounded_speech_with_large_language_model.md)**

:   提出 InfiniSST，将流式同声翻译建模为多轮对话任务，结合流式语音编码器和 KV cache 管理策略实现无限长语音输入的实时翻译，在 MuST-C 上减少 0.5-1s 计算感知延迟且不损失翻译质量。

**[Input Dependent Soft Prompting](llm_nlp/input_dependent_soft_prompting.md)**

:   提出 ID-SPAM，通过在输入 token 嵌入上施加可学习自注意力层并经瓶颈 MLP 生成**输入依赖**的软提示，仅在单层 Transformer 输入端拼接即可超越多种 Soft Prompt 基线，且具备优秀的零样本跨任务/跨领域迁移能力。

**[Interact Enabling Interactive Question-Driven Learning In Large Language Models](llm_nlp/interact_enabling_interactive_question-driven_learning_in_large_language_models.md)**

:   提出INTERACT框架，通过模拟师生对话让"学生"LLM通过主动提问向"教师"LLM学习新概念，在1,347个未见过的上下文上实验证明交互式学习最高可提升25%的理解准确率，且仅需5轮对话即可匹配静态学习基线。

**[Interactive And Expressive Code-Augmented Planning With Large Language Models](llm_nlp/interactive_and_expressive_code-augmented_planning_with_large_language_models.md)**

:   本文提出REPL-Plan，通过让LLM与扩展的REPL（Read-Eval-Print Loop）交互，实现既能充分利用代码表达力又能动态纠错和处理模糊子问题的自顶向下规划方法，在ALFWorld、WebShop和真实网页导航任务中取得强劲表现。

**[Internal and External Impacts of Natural Language Processing Papers](llm_nlp/internal_and_external_impacts_of_natural_language_processing_papers.md)**

:   从内部（NLP 社区）和外部（其他领域）两个维度衡量 NLP 论文影响力，构建影响力预测模型。

**[Investigating Context-Faithfulness In Large Language Models The Roles Of Memory ](llm_nlp/investigating_context-faithfulness_in_large_language_models_the_roles_of_memory_.md)**

:   通过测量 LLM 对同一问题不同释义的回答一致性来量化"记忆强度"，发现 LLM 对外部证据的接受度与记忆强度高度负相关，且改写式证据比重复或详细证据更有效。

**[Ipo Your Language Model Is Secretly A Preference Classifier](llm_nlp/ipo_your_language_model_is_secretly_a_preference_classifier.md)**

:   提出隐式偏好优化（IPO），利用生成式LLM自身作为偏好分类器（通过"Yes/No"token的概率），替代外部奖励模型来获取偏好信号，从而实现低成本的自对齐训练。

**[Is It Just Semantics? A Case Study of Discourse Particle Understanding in LLMs](llm_nlp/is_it_just_semantics_a_case_study_of_discourse_particle_understanding_in_llms.md)**

:   以德语语气词为案例研究 LLM 的语用理解能力，发现模型对语义细微差异（如 'doch''halt''ja'）的掌握远不如语法和词汇。

**[Jopa Explaining Large Language Models Generation Via Joint Prompt Attribution](llm_nlp/jopa_explaining_large_language_models_generation_via_joint_prompt_attribution.md)**

:   提出 JoPA（Joint Prompt Attribution）框架，将 LLM 生成任务的 prompt 归因建模为组合优化问题，用概率搜索算法高效寻找对输出有因果影响的输入 token 组合，解决了现有方法忽略 token 间协同效应的问题。

**[Just a Scratch: Enhancing LLM Capabilities for Self-Harm Detection through Intent Refinement](llm_nlp/just_a_scratch_enhancing_llm_capabilities_for_self-harm_detection_through_intent.md)**

:   通过意图精炼增强 LLM 自伤检测能力，将帖子按意图分类（求助/记录/警示等）再进行风险评估，准确率显著提升。

**[Kazmmlu Evaluating Language Models On Kazakh Russian And Regional Knowledge Of K](llm_nlp/kazmmlu_evaluating_language_models_on_kazakh_russian_and_regional_knowledge_of_k.md)**

:   提出 KazMMLU，首个专为哈萨克斯坦设计的 MMLU 风格双语（哈萨克语+俄语）评估基准，包含 23,000 道来自真实教育材料的多选题，覆盖 STEM、人文、社会科学等多学科多教育层次，评估了 27 个多语言 LLM，揭示了当前模型在哈萨克语上的显著不足。

**[Knockout: LLM Assessment Using Large Language Models for Evaluations through Iterative Elimination](llm_nlp/knockout_llm_assessment_using_large_language_models_for_evaluations_through_iter.md)**

:   提出 Knockout 淘汰赛式 LLM 评估方法，通过迭代配对比较逐步淘汰弱模型，比全配对比较效率高 O(n log n) vs O(n^2)。

**[Knowledge Boundary Crosslingual](llm_nlp/knowledge_boundary_crosslingual.md)**

:   通过探测 LLM 内部表示，揭示知识边界认知在多语言间呈线性结构，提出 training-free 对齐方法实现跨语言知识边界感知迁移，并发现"弱到强泛化"现象。

**[Knowledge Boundary Survey](llm_nlp/knowledge_boundary_survey.md)**

:   本文提出了 LLM 知识边界的形式化定义框架，将知识分为四种类型（PAK/PSK/MSU/MAU），并围绕"为什么研究知识边界""如何识别知识边界""如何缓解知识边界问题"三个核心问题系统综述了相关研究。

**[Language Codec Bridging Discrete Codec Speech Language Models](llm_nlp/language_codec_bridging_discrete_codec_speech_language_models.md)**

:   提出 Language-Codec，通过掩码通道残差向量量化（MCRVQ）机制和改进的傅里叶变换解码器，弥合离散编解码器表示与下游语音语言模型之间的鸿沟，仅用4个码本通道即实现高质量音频重建。

**[Language Model Fine-Tuning On Scaled Survey Data For Predicting Distributions Of](llm_nlp/language_model_fine-tuning_on_scaled_survey_data_for_predicting_distributions_of.md)**

:   提出直接在大规模公众意见调查数据（SubPOP，含 3362 道题目、70K 子群体-响应对）上微调 LLM，使其预测不同人口统计子群体的意见分布，相比 prompt engineering 基线将 Wasserstein 距离降低 32-46%，且泛化到未见过的调查和子群体。

**[Large Language Models Are Good Relational Learners](llm_nlp/large_language_models_are_good_relational_learners.md)**

:   提出 Rel-LLM 框架，利用 GNN 编码器从关系数据库中提取结构化子图表示，将其作为软提示注入冻结的 LLM，在 RelBench 基准上实现了关系深度学习（RDL）任务的 SOTA 性能，并支持零样本预测。

**[Large Language Models For Predictive Analysis How Far Are They](llm_nlp/large_language_models_for_predictive_analysis_how_far_are_they.md)**

:   提出 PredictiQ 基准（1130 查询×44 真实数据集×8 领域），从文本分析、代码生成和文本-代码对齐三个维度评估 12 个 LLM 的预测分析能力，发现现有 LLM 在此领域仍面临巨大挑战。

**[Large Language Models In Bioinformatics A Survey](llm_nlp/large_language_models_in_bioinformatics_a_survey.md)**

:   系统综述 LLM 在生物信息学中的应用，覆盖基因组序列建模、RNA 结构预测、蛋白质功能推断、单细胞转录组学四大方向，讨论数据稀缺、计算复杂度、跨组学整合等挑战。

**[Lazyreview Peer Review](llm_nlp/lazyreview_peer_review.md)**

:   构建首个标注"懒惰思维"细粒度类别的同行评审数据集 LazyReview——发现 LLM 在零样本下难以检测评审中的懒惰思维启发式，但在 LazyReview 上指令微调后性能提升 10-20 个点，且经懒惰思维反馈修改的评审显著更全面和可操作。

**[Length Controlled Generation For Black-Box Llms](llm_nlp/length_controlled_generation_for_black-box_llms.md)**

:   提出基于 Metropolis-Hastings 算法的迭代采样框架，结合重要性采样加速策略，在**不修改模型参数**的前提下实现黑盒 LLM 的精确长度控制，在 Llama3.1 上达到**100%**的长度控制成功率，最多仅需 5 次迭代，且不损害生成质量。

**[Lesa Learnable Llm Layer Scaling-Up](llm_nlp/lesa_learnable_llm_layer_scaling-up.md)**

:   提出 LESA，一种基于 SVD 发现层间潜在模式并通过神经网络预测中间层参数的可学习深度扩展方法，相比启发式层复制方法获得更好的初始化和更快的收敛速度，训练成本降低一半以上。

**[Leveraging Large Language Models To Measure Gender Representation Bias In Gender](llm_nlp/leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)**

:   提出基于 LLM 的方法量化有语法性别语言（西班牙语、巴伦西亚语）语料中的性别表示偏差，发现 4:1 到 6:1 的男性主导失衡，并证明少量反向偏差数据（5000句）可有效缓解模型输出中的性别不平衡。

**[Library-Like Behavior In Language Models Is Enhanced By Self-Referencing Causal ](llm_nlp/library-like_behavior_in_language_models_is_enhanced_by_self-referencing_causal_.md)**

:   提出"自引用因果循环"（ReCall）概念，揭示 LLM 预训练数据中自然存在的重复 token 序列如何形成循环引用，使自回归模型能够绕过单向因果限制、克服逆向诅咒（reversal curse），并据此设计了两步 ReCall-aware prompting 策略。

**[Limit Llm Planning Formalizer](llm_nlp/limit_llm_planning_formalizer.md)**

:   系统评估"LLM-as-Formalizer"方法论的极限——首次要求 LLM 生成完整 PDDL 表示（而非部分），从不同自然度的文本描述中形式化规划领域，发现最强模型（GPT-4o/o3-mini/DeepSeek-R1）可有效形式化超越直接规划，但描述越自然性能越低，弱模型卡在语法错误而强模型面临语义错误。

**[Llamaduo Llmops Pipeline For Seamless Migration From Service Llms To Small-Scale](llm_nlp/llamaduo_llmops_pipeline_for_seamless_migration_from_service_llms_to_small-scale.md)**

:   提出 LlamaDuo 自动化 LLMOps 流水线，通过服务 LLM 生成合成数据迭代微调小模型，使 2B-8B 本地模型在特定下游任务上逼近甚至匹敌 GPT-4o 等大模型性能，且长期部署成本显著降低。

**[Llm As Effective Streaming Processor Bridging Streaming-Batch Mismatches With Gr](llm_nlp/llm_as_effective_streaming_processor_bridging_streaming-batch_mismatches_with_gr.md)**

:   系统性地识别并量化了 batch-trained LLM 适配流式场景的三种不匹配（输入注意力 / 输出注意力 / 位置 ID），发现仅输入注意力不匹配才是关键瓶颈（+2.20 BLEU），据此提出组位置编码（Group Position Encoding）——源/目标各自维护连续位置 ID 即可，无需昂贵的 KV cache 重编码，在机器翻译和 ASR 两种跨模态任务上均超越专用流式架构。

**[Llm Braces Straightening](llm_nlp/llm_braces_straightening.md)**

:   LLMBraces 通过计算 FFN 层中各 value 向量与输入的相关性得分，动态调节子更新（sub-update）的贡献权重，用极少参数（比 LoRA 少 75%）同时提升模型预测精度和实现可控文本生成。

**[Llm Broken Telephone](llm_nlp/llm_broken_telephone.md)**

:   类比"传话游戏"研究 LLM 在迭代生成中的信息失真现象，通过多语言翻译链实验发现：信息失真随迭代累积，受中间语言选择和链复杂度影响，可通过温度控制和受限提示缓解但无法消除。

**[Llm Meets Scene Graph Can Large Language Models Understand And Generate Scene Gr](llm_nlp/llm_meets_scene_graph_can_large_language_models_understand_and_generate_scene_gr.md)**

:   构建 TSG Bench（120 场景、2041 描述、4289 文本场景图），系统评估 11 个 LLM 在场景图理解（SGDS/SGQA）和生成（SA-SGG/MA-SGG）四类任务上的能力，发现最强模型在理解上接近人类但生成任务仍有 15-17% 的差距。

**[Llm Robustness Incorrect Mcq](llm_nlp/llm_robustness_incorrect_mcq.md)**

:   提出"反思判断力"（Reflective Judgment）概念来衡量 LLM 在所有选项都错误的选择题中拒绝选择的能力，发现对齐后的模型（GPT-4o 等）往往盲目服从指令选择错误选项，而基座模型反而表现更好，且该能力随模型规模增大而涌现。

**[Llm Test Case Gen Bugs](llm_nlp/llm_test_case_gen_bugs.md)**

:   提出TrickCatcher——一种LLM驱动的测试用例生成方法，通过PUT引导的程序变体生成、基于生成器的输入生成和多样性驱动的差异测试三阶段流程，专门检测"plausible programs"（能通过现有测试套件但仍含隐蔽bug的程序）中的tricky bugs，F1分数达到SOTA基线的1.66倍。

**[Llm Vs Human Judges Study](llm_nlp/llm_vs_human_judges_study.md)**

:   构建包含20个NLP数据集（7万+实例）的 Judge-Bench 基准，系统评估11个LLM作为评判者与人类标注的一致性，发现模型在不同任务/属性/标注者专业水平上表现差异巨大，建议部署前必须针对特定任务做人类标注验证。

**[Llm Writing Assessment](llm_nlp/llm_writing_assessment.md)**

:   利用 L2 研究生文献综述语料库，系统评估了 LLM 在多维分析写作评估（评分+评论）上的能力，并提出可解释的反馈质量评估框架 ProEval。

**[Llms Can Be Easily Confused By Instructional Distractions](llm_nlp/llms_can_be_easily_confused_by_instructional_distractions.md)**

:   本文提出了 DIM-Bench 基准，系统评估了当输入文本本身就像指令时 LLM 的指令跟随能力，发现即使是 GPT-4o 等最先进的模型也极易受到这种"指令干扰"的影响而偏离用户真实意图。

**[Llms Know Their Vulnerabilities Uncover Safety Gaps Through Natural Distribution](llm_nlp/llms_know_their_vulnerabilities_uncover_safety_gaps_through_natural_distribution.md)**

:   提出 ActorBreaker 多轮攻击方法，基于 Latour 的行动者网络理论，利用与有害内容语义相关的良性 prompt（自然分布偏移）绕过安全机制，在 HarmBench 上达到 SOTA 攻击成功率，揭示了预训练数据与安全训练数据之间的语义覆盖差距。

**[Llms Persona-Plug Personalized Llms](llm_nlp/llms_persona-plug_personalized_llms.md)**

:   提出 PPlug 模型，通过轻量级插件式用户嵌入器将用户历史行为压缩为单一个性化嵌入，以 plug-and-play 方式引导 LLM 生成个性化输出，在 LaMP 基准上显著超越检索式和微调式基线，最高提升 35.8%。

**[Lm Graph Search Supervision](llm_nlp/lm_graph_search_supervision.md)**

:   本文证明了 path-star 图搜索任务在 decoder-only LM 上的失败并非 next-token prediction 范式的根本缺陷，而是由"监督污染"（supervision adulteration）导致的——过量的 teacher-forcing 监督信号诱导模型学到 Clever Hans Cheat 捷径，阻碍了子任务分解；通过 token masking、ranking-into-the-future、scratchpad、树形拓扑等六种正交方法均可使任务可学。

**[Locateandfocus Enhancing Terminology Translation In Speech](llm_nlp/locateandfocus_enhancing_terminology_translation_in_speech.md)**

:   提出Locate-and-Focus方法用于语音LLM的术语翻译：先用滑动窗口检索定位语音中包含术语的片段，再通过音频替换和Tag Cue引导模型聚焦翻译知识，在英中/英德方向上术语翻译成功率大幅提升。

**[Logical Forms Complement Probability In Understanding Language Model And Human P](llm_nlp/logical_forms_complement_probability_in_understanding_language_model_and_human_p.md)**

:   系统研究 LLM 在命题逻辑和模态逻辑推理上的能力，发现除了输入概率（perplexity）外，逻辑形式（modality、argument form）是预测 LLM 表现的重要互补因素，并通过人类行为数据对比揭示人机推理的异同。

**[Longdpo Unlock Better Long-Form Generation Abilities For Llms Via Critique-Augme](llm_nlp/longdpo_unlock_better_long-form_generation_abilities_for_llms_via_critique-augme.md)**

:   提出 LongDPO，通过 MCTS 收集步级偏好对、全局记忆池维护事实一致性、critique 增强低质量候选，再用步级 DPO 进行细粒度优化，在 LongBench-Write 上显著提升长文本生成质量，同时保持通用能力。

**[Lost In Literalism How Supervised Training Shapes Translationese In Llms](llm_nlp/lost_in_literalism_how_supervised_training_shapes_translationese_in_llms.md)**

:   本文系统研究了大语言模型在机器翻译中产生翻译腔（translationese）的现象，揭示了监督微调（SFT）数据中的翻译腔偏差是导致LLM翻译不自然的根本原因，并提出了通过润色训练参考译文和过滤不自然训练实例来缓解翻译腔的方法。

**[Mapping 1000 Models Loglikelihood](llm_nlp/mapping_1000_models_loglikelihood.md)**

:   提出用对数似然向量（log-likelihood vector）将 1000+ 语言模型映射到一个统一空间，证明向量间欧氏距离近似 KL 散度，可实现模型聚类可视化、基准性能预测（r=0.96）和数据泄漏检测。

**[Maps Personalized Search](llm_nlp/maps_personalized_search.md)**

:   首次建模电商搜索中的"搜索动机"——用户在搜索前的咨询行为蕴含的真实需求，提出MAPS框架融合LLM语义、MoAE池化和双重对齐机制，在真实商业数据上HR@10提升24.4%（从0.5685到0.7071）。

**[Masking In Multi-Hop Qa An Analysis Of How Language Models Perform With Context ](llm_nlp/masking_in_multi-hop_qa_an_analysis_of_how_language_models_perform_with_context_.md)**

:   系统分析因果掩码对 LLM 多跳 QA 的影响，发现 encoder-decoder（Flan-T5 770M）零样本多跳性能优于 decoder-only（Qwen 7B），且 prefix mask 可提升 decoder-only 的多跳推理能力 5.1%。

**[Mathfusion Instruction Fusion](llm_nlp/mathfusion_instruction_fusion.md)**

:   MathFusion 提出了跨问题指令融合的数学数据增强框架，通过顺序融合、并行融合和条件融合三种策略将两个数学问题合成新问题，仅用 45K 额外合成指令就在 6 个 benchmark 上平均提升 18 分准确率。

**[Mathneuro Math Reasoning Isolation](llm_nlp/mathneuro_math_reasoning_isolation.md)**

:   MathNeuro 提出了一种仅用前向传播就能隔离 LLM 中数学推理特定参数的方法，通过计算权重×激活的重要性分数并过滤掉通用语言任务也需要的参数，实现了精准的数学能力"手术"——剪除这些参数删除数学能力，缩放它们则提升 4-35% 数学性能。

**[Membench Towards More Comprehensive Evaluation On The Memory Of Llm-Based Agents](llm_nlp/membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)**

:   提出 MemBench，从多场景（参与/观察）、多层级（事实记忆/反思记忆）和多指标（准确率/召回率/容量/效率）三个维度全面评估 LLM Agent 的记忆能力，弥补了现有评估在记忆层级和交互场景上的不足。

**[Meraser Fingerprint Erasure](llm_nlp/meraser_fingerprint_erasure.md)**

:   提出 MEraser（Mismatched Eraser），通过两阶段微调策略（错配数据擦除 + 干净数据恢复）以不到 1000 条样本完全移除 LLM 中基于后门的指纹水印，同时保持模型性能，并首创可迁移的 LoRA 擦除适配器。

**[Mergeprint Fingerprint Ownership](llm_nlp/mergeprint_fingerprint_ownership.md)**

:   提出 MergePrint，首个针对模型合并（model merging）场景的 LLM 黑盒指纹验证方法，通过伪合并模型模拟合并行为并两阶段优化（输入优化 + 参数优化），使嵌入的指纹在合并后仍可被检测，实现高效、无害、抗篡改的所有权验证。

**[Mexgen Multi Level Explanations](llm_nlp/mexgen_multi_level_explanations.md)**

:   提出 MExGen（Multi-Level Explanations for Generative Language Models），将 LIME/SHAP 等扰动式归因方法扩展到 LLM 的上下文生成任务上——为摘要/问答等任务中 LLM 输出的每个部分量化上下文各段落的影响程度，比 LLM 自解释更忠实。

**[Mha2Mla Deepseek Latent Attention](llm_nlp/mha2mla_deepseek_latent_attention.md)**

:   MHA2MLA 首次提出将已训练好的 MHA 模型高效迁移到 DeepSeek 的 MLA 架构的方法，通过贡献度感知的 partial-RoPE 移除和联合 SVD 低秩近似，仅用 0.6%-1% 的训练数据即可恢复性能，将 Llama2-7B 的 KV cache 压缩 92.19% 且 LongBench 性能仅下降 1%。

**[Mind The Belief Gap Group Identity In The World Of Llms](llm_nlp/mind_the_belief_gap_group_identity_in_the_world_of_llms.md)**

:   构建多智能体信念一致性模拟框架，发现 LLM 比人类表现出更强的信念一致性偏见（gpt-3.5: 0.93 vs 人类: 0.2-0.62），导致虚假信息传播加剧、跨群体学习受阻，并探索接触假说等缓解策略（最优可改善 37%）。

**[Mirage Exploring How Large Language Models Perform In Complex Social Interactive](llm_nlp/mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)**

:   本文提出 MIRAGE 评估框架，通过八个精心设计的剧本杀场景和四项指标（信任倾向 TII、线索调查 CIC、交互能力 ICI、剧本合规 SCI）系统评估 LLM 在复杂社交互动环境中的表现，发现即使 GPT-4 也在这些场景中面临严峻挑战。

**[Mitigate Position Bias In Large Language Models Via Scaling A Single Dimension](llm_nlp/mitigate_position_bias_in_large_language_models_via_scaling_a_single_dimension.md)**

:   发现 LLM 隐状态中存在编码绝对位置信息的特定通道（positional hidden states），通过缩放这单一通道即可缓解"lost in the middle"位置偏差，在多文档 QA 基准上提升高达 15.2%，且不影响模型其他能力。

**[Mixtures Of In-Context Learners](llm_nlp/mixtures_of_in-context_learners.md)**

:   提出 MoICL 方法，将 ICL 的 demonstration 集合划分为多个子集（专家），通过可学习的权重函数融合各专家的 next-token 分布，在不修改 LLM 参数的前提下显著提升 ICL 的准确率、鲁棒性和效率。

**[Moral Values Western](llm_nlp/moral_values_western.md)**

:   提出通过词语联想（word association）而非直接提问来比较 LLM 与西方英语社会的道德价值观，发现 LLM 在正面道德维度上与人类更一致，但在情感多样性和具体性上存在系统性差异。

**[Multi-Prompting Decoder Helps Better Language Understanding](llm_nlp/multi-prompting_decoder_helps_better_language_understanding.md)**

:   提出 Multi-Prompting Decoder（MPD）框架，通过多提示查询 PLM 获取多组隐状态和类别分数，结合最优传输匹配和校准解码策略，在 MaaS（模型即服务）场景下的 few-shot 分类任务上显著超越现有方法。

**[Multi Attribute Steering](llm_nlp/multi_attribute_steering.md)**

:   提出 MAT-Steer，通过属性感知的 token 级 gating 机制和正交性约束，实现推理时对 LLM 多属性（如真实性、毒性、偏见）的同时精准干预，在 QA 和生成任务上全面超越现有 ITI 和微调方法。

**[Multiple Choice Eval](llm_nlp/multiple_choice_eval.md)**

:   系统性论证了 MCQA（多选题问答）作为 LLM 评测标准格式的三大缺陷——格式本身的局限性、数据集构建质量问题、以及 LLM 在 MCQA 上的特有错误——并从教育测试学中引入改进方案。

**[Natural Language Processing In Support Of Evidence-Based Medicine A Scoping Revi](llm_nlp/natural_language_processing_in_support_of_evidence-based_medicine_a_scoping_revi.md)**

:   对 129 篇研究进行范围综述，系统梳理了 NLP 技术在循证医学（EBM）五步流程（Ask-Acquire-Appraise-Apply-Assess）中的应用，覆盖证据检索、PICO 提取、质量评估、证据合成、摘要生成、临床试验匹配等任务，指出了 LLM 时代的新机遇与挑战。

**[Neural Topic Modeling With Large Language Models In The Loop](llm_nlp/neural_topic_modeling_with_large_language_models_in_the_loop.md)**

:   提出LLM-ITL框架，将LLM以"in-the-loop"方式集成到神经主题模型（NTM）训练中，通过基于最优传输的主题对齐目标和置信度加权机制，在保持文档表示质量和计算效率的同时显著提升主题可解释性。

**[Newsinterview A Dataset And A Playground To Evaluate Llms Grounding Gap Via Info](llm_nlp/newsinterview_a_dataset_and_a_playground_to_evaluate_llms_grounding_gap_via_info.md)**

:   构建了 4 万条新闻采访对话数据集，发现 LLM 在采访场景中缺乏 acknowledgement（少 50%）和话题转换能力（少 30%），并设计了含说服机制的模拟博弈环境（NewsInterview），证明最优 LLM（gpt-4o）也仅能提取 50.4% 的信息项。

**[Nudging Inference Time Alignment](llm_nlp/nudging_inference_time_alignment.md)**

:   提出 Nudging，一种免训练的推理时对齐算法，利用小型对齐模型在基础模型不确定时注入少量"nudging tokens"来引导输出，用 7-14 倍小的模型就能达到甚至超过大型对齐模型的性能。

**[Olmotrace Tracing Language Model Outputs Back To Trillions Of Training Tokens](llm_nlp/olmotrace_tracing_language_model_outputs_back_to_trillions_of_training_tokens.md)**

:   推出 OLMoTrace，首个能实时将语言模型输出追溯到万亿 token 训练数据的系统，基于扩展的 infini-gram 后缀数组索引，在 4.5 秒内完成逐字匹配追溯，支持事实核查、幻觉检测和创造力分析。

**[On Entity Identification In Language Models](llm_nlp/on_entity_identification_in_language_models.md)**

:   提出基于聚类的评估框架（Purity/Inverse Purity）分析 LLM 内部表示中的实体区分能力，发现实体信息在早期层（~归一化位置 0.2）的 20 维子空间中达到线性可分（F1~0.9），且不同大模型收敛到结构同构的实体编码——为"LLM 从纯文本训练中涌现离散知识结构"提供了系统性证据。

**[On The Mutual Influence Of Gender And Occupation In Llm Representations](llm_nlp/on_the_mutual_influence_of_gender_and_occupation_in_llm_representations.md)**

:   通过在 LLM 嵌入空间中近似性别方向（gender direction），系统研究了名字的性别表征与职业上下文之间的双向影响：职业上下文会偏移名字的性别表征，而名字的性别表征反过来影响 LLM 在职业预测任务中的偏差行为，但二者的相关性仅为中等强度。

**[On The Risk Of Evidence Pollution For Malicious Social Text Detection In The Era](llm_nlp/on_the_risk_of_evidence_pollution_for_malicious_social_text_detection_in_the_era.md)**

:   本文系统研究了LLM时代下恶意社交文本检测中的"证据污染"风险，提出13种污染方法和3种防御策略，发现LLM生成的虚假证据可导致检测器性能下降高达14.4%，且现有防御策略面临实际部署挑战。

**[Open-Set Living Need Prediction with Large Language Models](llm_nlp/open-set_living_need_prediction_with_large_language_models.md)**

:   提出开放集生活需求预测任务，利用 LLM 从用户描述中预测生活需求，不限于预定义类别，应用于养老服务、社区管理等场景。

**[Opencoder The Open Cookbook For Top-Tier Code Large Language Models](llm_nlp/opencoder_the_open_cookbook_for_top-tier_code_large_language_models.md)**

:   提出OpenCoder，一个完全开源的代码大语言模型（含1.5B和8B版本），不仅性能达到顶级水平，更作为"open cookbook"开放了可复现的数据处理流水线、预训练数据集、消融实验和训练协议，为代码智能研究提供基础设施。

**[Palm A Culturally Inclusive And Linguistically Diverse Dataset For Arabic Llms](llm_nlp/palm_a_culturally_inclusive_and_linguistically_diverse_dataset_for_arabic_llms.md)**

:   由 44 名阿拉伯世界研究者历时一年社区驱动构建的 Palm 数据集，涵盖全部 22 个阿拉伯国家、20 个文化主题、10 种方言，共 17,411 条人工创建的指令对，用于评估和提升 LLM 的阿拉伯文化和方言能力。

**[Paper 2312 17294](llm_nlp/paper_2312_17294.md)**

:   提出 OpenAgent 系统，通过自主从 GitHub 发现和集成专业工具来解决开放域任务，并构建 OpenAct 基准评测 LLM 在需要领域特定工具的开放域问题上的能力。

**[Perspective Transition Of Large Language Models For Solving Subjective Tasks](llm_nlp/perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)**

:   提出 RPT（Reasoning through Perspective Transition），通过在同一 prompt 中让 LLM 依次探索直接/角色扮演/第三人称三种视角、按置信度排序、选最优视角推理，在 12 个主观任务、4 个模型（GPT-4/GPT-3.5/Llama-3/Qwen-2）上均超越固定视角与集成基线，GPT-3.5 上平均提升 +4.56 点。

**[Pitfalls Of Scale Investigating The Inverse Task Of Redefinition In Large Langua](llm_nlp/pitfalls_of_scale_investigating_the_inverse_task_of_redefinition_in_large_langua.md)**

:   通过重定义任务（给著名物理常数或度量单位赋予新值并要求 LLM 据此推理），揭示大模型比小模型更容易锚定于先验知识，展现规模增长带来推理灵活性下降的逆向缩放现象。

**[Plangenllms Planning Survey](llm_nlp/plangenllms_planning_survey.md)**

:   PlanGenLLMs 是一篇系统性综述，基于经典 AI 规划评估框架提出完整性、可执行性、最优性、表示、泛化性和效率六大评估准则，全面梳理了 LLM 作为规划器的方法、评估和未来方向。

**[Planning-Driven Programming A Large Language Model Programming Workflow](llm_nlp/planning-driven_programming_a_large_language_model_programming_workflow.md)**

:   提出 LPW（LLM Programming Workflow），通过"方案生成→计划验证→代码实现→基于计划验证的精准调试"的两阶段工作流，显著提升 LLM 代码生成准确率，在 GPT-4o 上实现 HumanEval 98.2%、MBPP 84.8%、LiveCode 59.3% 的新 SOTA。

**[Plugin Finetuning Bridge](llm_nlp/plugin_finetuning_bridge.md)**

:   提出 PiFi 框架，将 LLM 的单个冻结层"插入"到 SLM 中再进行微调，以极低的额外计算成本将 LLM 的语言知识和泛化能力迁移到小模型，在 NLU 和 NLG 任务上均获得一致提升。

**[Polishing Every Facet Of The Gem](llm_nlp/polishing_every_facet_of_the_gem.md)**

:   提出 KoGEM（韩语语法评估基准），包含 1,524 道基于理论语言学分类的多选题，覆盖音韵/形态/句法/语义/规范 5 大类 16 子类，零样本评估 27 个 LLM 并与人类对比，揭示 LLM 在需要经验知识的语言子类（如发音规则、音韵变化）上远逊人类，而显式补充经验知识（发音文本、语素分解）后可大幅提升。

**[Political Bias Theory Grounded](llm_nlp/political_bias_theory_grounded.md)**

:   本文用政治科学中经过验证的 World Values Survey (WVS) 替代缺乏科学基础的 Political Compass Test (PCT)，设计 30 种提示变体在 11 个开源/商业 LLM 上收集 88,110 条开放式回复并训练立场分类器自动标注，发现指令微调模型普遍偏左但偏见度量对提示高度敏感，PCT 会夸大特定模型（如 GPT-3.5）的政治偏见。

**[Pragmatics Survey](llm_nlp/pragmatics_survey.md)**

:   全面梳理用于评估 NLP 系统语用能力的资源——按语用现象（隐含义、指称、言语行为、会话含义、预设等）分类数据集，分析任务设计、数据收集方法和评估方式，揭示了现代 LLM 在处理语境相关语言使用上的趋势、挑战和空白。

**[Praise Enhancing Product Descriptions With Llm-Driven Structured Insights](llm_nlp/praise_enhancing_product_descriptions_with_llm-driven_structured_insights.md)**

:   提出 PRAISE，一个 4 步 LLM pipeline（属性提取 → 跨产品对比 → 语义分组 → 结构化呈现），使用 Gemini 2.0 Flash 从 Amazon 产品描述中自动生成结构化洞察。在 90 个产品 × 9 个类别上验证，多步 pipeline 显著优于单次生成；效果与产品主观性高度相关（Arts&Crafts F1=0.82 vs Books F1=0.36），每产品仅需 $2R+1$ 次 API 调用。

**[Pre3 Deterministic Pda Structured Gen](llm_nlp/pre3_deterministic_pda_structured_gen.md)**

:   提出 Pre³，将 LR(1) 文法转化为确定性下推自动机（DPDA），通过预计算前缀条件边消除运行时非确定性探索，实现结构化 LLM 生成的显著加速——每 token 耗时降低最高 40%，吞吐提升最高 36%。

**[Probabilistic Aggregation And Targeted Embedding Optimization For Collective Mor](llm_nlp/probabilistic_aggregation_and_targeted_embedding_optimization_for_collective_mor.md)**

:   提出一种双阶段框架：先用截断正态分布EM算法将多个LLM的连续道德评分聚合为集体共识概率，再对偏离共识的模型进行道德理论token级嵌入优化，使其与集体意见对齐，实现多LLM间一致的道德推理。

**[Problem-Solving Logic Guided Curriculum In-Context Learning For Llms Complex Rea](llm_nlp/problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)**

:   提出问题解决逻辑引导的课程式 ICL，基于 QDMR 分析解题步骤结构来选择和排序 few-shot 示例（按步骤数从易到难），在多个复杂推理基准上超越现有 ICL 方法。

**[Psycholinguistic Word Features A New Approach For The Evaluation Of Llms Alignme](llm_nlp/psycholinguistic_word_features_a_new_approach_for_the_evaluation_of_llms_alignme.md)**

:   首次系统提出使用心理语言学词汇规范（Glasgow和Lancaster数据集共13种词汇特征）来评估LLM与人类的对齐程度，发现LLM在情感/概念特征上对齐较好，但在感知觉特征（味觉、嗅觉、触觉等）上对齐极差，揭示了LLM缺乏具身认知的根本局限。

**[Pugc Align Implicit Pref Ugc](llm_nlp/pugc_align_implicit_pref_ugc.md)**

:   提出 PUGC 框架，利用非标注用户生成内容（UGC）中的隐式人类偏好来生成偏好数据——将 UGC 转化为查询+参考文本，以此评分模型生成的响应，用 DPO 实现可扩展的领域特定对齐，在 Alpaca Eval 2 上基于 Mistral-7B 达到 35.93% 长度控制胜率 SOTA。

**[Quantifying Semantic Emergence In Language Models](llm_nlp/quantifying_semantic_emergence_in_language_models.md)**

:   提出了 Information Emergence (IE) 这一基于信息论的定量指标，通过比较 Transformer 各层中宏观（序列级）与微观（token级）的互信息差异，量化 LLM 从 token 中提取语义的能力。

**[Ranking Unraveled Recipes For Llm Rankings In Head-To-Head Ai Combat](llm_nlp/ranking_unraveled_recipes_for_llm_rankings_in_head-to-head_ai_combat.md)**

:   系统性地评估四种排名算法（Elo、Bradley-Terry、Glicko、Markov Chain）在LLM头对头评估中的表现，定义三条核心排名准则（传递性、预测准确率、超参数敏感性），发现广泛使用的 Elo 排名在稳定性和一致性方面存在严重缺陷，推荐 Glicko 用于大规模不均匀数据集、Bradley-Terry 用于小型可控数据集。

**[Re-Task Revisiting Llm Tasks From Capability Skill And Knowledge Perspectives](llm_nlp/re-task_revisiting_llm_tasks_from_capability_skill_and_knowledge_perspectives.md)**

:   借鉴 Bloom 分类学和知识空间理论，提出 Re-TASK 框架将 LLM 任务从"能力项-技能-知识"三层视角进行重新审视，并设计 Re-TASK prompting 策略通过针对性的知识注入和技能适配来增强 CoT 在领域任务上的表现，在法律任务上最高提升 45%。

**[Reason From Future Reverse Thought Chain Enhances Llm Reasoning](llm_nlp/reason_from_future_reverse_thought_chain_enhances_llm_reasoning.md)**

:   提出 Reason from Future（RFF）推理范式，通过交替进行逆向推理（从目标向前分解）和正向推理（从当前状态向目标逼近）实现双向推理，在 Game of 24、GSM8K、MATH-500 等基准上显著超越 CoT、ToT、CR 等方法，同时大幅减少搜索空间。

**[Recent Advances In Speech Language Models A Survey](llm_nlp/recent_advances_in_speech_language_models_a_survey.md)**

:   首篇 Speech Language Models (SpeechLMs) 综合综述，系统梳理从"ASR+LLM+TTS"级联架构到端到端语音语言模型的演进，提出按三大组件（speech tokenizer / language model / vocoder）和训练方案分类的分类体系，覆盖下游能力、评估指标、挑战与未来方向。

**[Recurrent Kif Continual Learning](llm_nlp/recurrent_kif_continual_learning.md)**

:   提出Recurrent-KIF持续学习框架，通过内外循环迭代机制动态估计参数重要性分布，利用基于重要性的二值掩码进行知识融合，有效缓解灾难性遗忘并促进知识迁移。

**[Red-Teaming Llm Multi-Agent Systems Via Communication Attacks](llm_nlp/red-teaming_llm_multi-agent_systems_via_communication_attacks.md)**

:   提出 Agent-in-the-Middle (AiTM) 攻击，通过拦截和篡改 LLM 多智能体系统中的 agent 间通信消息（而非直接修改 agent 本身），利用一个带反思机制的对抗性 agent 生成上下文感知的恶意指令，在多种框架/通信结构/真实应用上均取得 40%~100% 的攻击成功率。

**[Refining Salience-Aware Sparse Fine-Tuning Strategies For Language Models](llm_nlp/refining_salience-aware_sparse_fine-tuning_strategies_for_language_models.md)**

:   首次系统评估了 8 种显著性度量在稀疏参数高效微调（SPEFT）中的效果，发现简单的梯度基方法配合静态掩码即可一致性地超越 LoRA，挑战了"PEFT 需要复杂设计"的常见认知。

**[Repbend Representation Bending Safety](llm_nlp/repbend_representation_bending_safety.md)**

:   提出 RepBend，将 activation steering 的核心思想（安全/不安全表示的向量差异）引入 LoRA 微调的损失函数设计，通过"弯曲"模型的表示空间使安全和不安全状态在潜在空间中远离彼此，在多种越狱攻击基准上实现高达 95% 的攻击成功率降低，且对模型通用能力影响极小。

**[Representations Of Fact Fiction And Forecast In Large Language Models Epistemics](llm_nlp/representations_of_fact_fiction_and_forecast_in_large_language_models_epistemics.md)**

:   通过受控故事任务评估 8 个开源 LLM 对认识情态（may/must、know/believe/doubt）的语义知识，发现 LLM 在生成恰当认知表达方面表现有限且不鲁棒——必然性（must）优于可能性（may），事实陈述优于信念陈述。

**[Retrollm Empowering Large Language Models To Retrieve Fine-Grained Evidence With](llm_nlp/retrollm_empowering_large_language_models_to_retrieve_fine-grained_evidence_with.md)**

:   提出RetroLLM统一框架，将检索和生成集成为单一自回归解码过程，通过层级FM-Index约束和前瞻式受限解码，使LLM能直接从语料库中生成细粒度证据，同时显著减少token消耗。

**[Reversal Of Thought Enhancing Large Language](llm_nlp/reversal_of_thought_enhancing_large_language.md)**

:   提出 Reversal of Thought (RoT)，一个即插即用的推理框架，通过偏好引导的逆向推理预热策略，让 LLM 从示例中反向生成"LLM 口味"的最优 prompt，再通过认知偏好管理器自动区分已知/未知任务，在多种推理任务上超越 CoT/ToT/GoT 等基线。

**[Revisiting Epistemic Markers In Confidence Estimation Can Markers Accurately Ref](llm_nlp/revisiting_epistemic_markers_in_confidence_estimation_can_markers_accurately_ref.md)**

:   本文定义了"标记置信度"（marker confidence）概念来衡量 LLM 使用认知标记（如"fairly certain"）时的实际准确率，通过 7 个模型和 7 个数据集的系统实验发现：认知标记在分布内场景表现稳定，但在分布外场景下极不可靠。

**[Riot Efficient Prompt Refinement With Residual Optimization Tree](llm_nlp/riot_efficient_prompt_refinement_with_residual_optimization_tree.md)**

:   提出 Residual Optimization Tree（RiOT），一种自动 prompt 优化框架，通过树结构管理优化过程、基于困惑度的节点选择增强多样性、以及文本残差连接缓解语义漂移问题。

**[Rocoft Efficient Finetuning Of Large Language Models With Row-Column Updates](llm_nlp/rocoft_efficient_finetuning_of_large_language_models_with_row-column_updates.md)**

:   提出 RoCoFT，一种极简的参数高效微调方法：仅更新 Transformer 权重矩阵中少量行或列的参数，在 GLUE、QA、摘要生成和常识/数学推理等任务上达到与 LoRA 等 SOTA PEFT 方法相当的精度，同时更省内存和计算，并通过 Neural Tangent Kernel 理论解释了其有效性。

**[Safer Or Luckier Llms As Safety Evaluators Are Not Robust To Artifacts](llm_nlp/safer_or_luckier_llms_as_safety_evaluators_are_not_robust_to_artifacts.md)**

:   系统评估了11个LLM裁判在安全领域的鲁棒性，发现道歉前缀等表面文本特征（artifact）可将评估偏好扭曲高达98%，提出基于jury的多模型聚合方案但仍未完全解决该问题。

**[Salience Sparse Fine Tuning](llm_nlp/salience_sparse_fine_tuning.md)**

:   首次系统评估 8 种 salience 指标用于稀疏微调（SPEFT）的效果，发现简单的梯度指标 + 静态掩码即可提供最佳性价比，在 GSM8k 上比 LoRA 高出 22.6%，质疑了"复杂方法才能做好 PEFT"的假设。

**[Sarft Roleplay Safety](llm_nlp/sarft_roleplay_safety.md)**

:   首次系统评估了角色扮演微调（role-play fine-tuning）对 LLM 安全性的影响，发现安全退化程度与角色特质（特别是反派角色）正相关，并提出 SaRFT 框架，通过隐式奖励函数自适应识别对不同角色有害的训练数据子集，配合 KL 散度正则化实现角色表现力与安全性的 Pareto 最优平衡。

**[Sconu Selective Conformal Uncertainty In Large Language Models](llm_nlp/sconu_selective_conformal_uncertainty_in_large_language_models.md)**

:   提出选择性保形不确定性框架 SConU，通过构建保形 p-value 进行显著性检验，首次实现对违反可交换性假设的不确定性数据异常点的自动过滤，从而在单域和跨域 QA 场景中严格管理 LLM 的错误覆盖率。

**[Seed Stepwise Reasoning Disruption Attack](llm_nlp/seed_stepwise_reasoning_disruption_attack.md)**

:   提出 SEED（Stepwise rEasoning Error Disruption）攻击方法，通过在 LLM 的推理链前几步中巧妙注入细微错误（如微调计算数字），让模型在后续推理中自然传播错误得出错误答案，兼容零样本/少样本设置，GPT-4o 检测率低至 0.8%，揭示了 LLM 逐步推理过程的严重安全漏洞。

**[Segment Level Diffusion](llm_nlp/segment_level_diffusion.md)**

:   提出段落级扩散（Segment-Level Diffusion, SLD），将长文本输出切分为多个段落（如句子/对话轮次），对每个段落的潜在表示进行扩散建模，结合对比学习和对抗训练增强表示鲁棒性，在摘要、故事生成、对话生成等任务上实现了比现有扩散模型更好的长文本生成质量。

**[Self-Instructed Derived Prompt Generation Meets In-Context Learning Unlocking Ne](llm_nlp/self-instructed_derived_prompt_generation_meets_in-context_learning_unlocking_ne.md)**

:   提出一种自指导强化学习框架来训练"派生提示生成模型"，并将派生提示-响应对作为上下文学习（ICL）示例来增强原始提示的查询，在不修改黑盒 LLM（如 GPT-4）参数的情况下显著提升响应质量。

**[Self-Training Elicits Concise Reasoning In Large Language Models](llm_nlp/self-training_elicits_concise_reasoning_in_large_language_models.md)**

:   发现 LLM 输出分布中天然包含简洁推理路径，提出 FS-BoN（Few-shot 条件化 + Best-of-N 采样）自训练框架，从模型自身分布中筛选短且正确的推理样本进行微调，在 GSM8K 和 MATH 上跨 5 个模型族实现平均 30% token 缩减且不损准确率，效率为先前方法 Rational Metareasoning 的 2.4 倍。

**[Self-Tuning Instructing Llms To Effectively Acquire New Knowledge Through Self-T](llm_nlp/self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)**

:   受费曼学习法启发，提出 Self-Tuning 框架，通过记忆-理解-自省三层自教学策略，显著提升 LLM 从新文档中有效获取和回忆知识的能力。

**[Selfelicit Evidence Highlighting](llm_nlp/selfelicit_evidence_highlighting.md)**

:   SelfElicit 发现 LLM 深层注意力分数天然能识别上下文中的关键证据（即使回答错误时也是），据此提出推理时自动高亮关键证据句的上下文增强方法，无需训练即可显著提升基于证据的 QA 任务性能。

**[Simulating Diverse Students](llm_nlp/simulating_diverse_students.md)**

:   提出一种基于知识图谱认知原型的免训练框架，使LLM Agent能够模拟不同认知水平学生的学习行为（包括错误），在GPT-4o上实现94%的行为预测准确率，相比基线提升100%。

**[Skillaggregation Reference-Free Llm-Dependent Aggregation](llm_nlp/skillaggregation_reference-free_llm-dependent_aggregation.md)**

:   本文提出SkillAggregation方法，通过学习上下文相关的LLM评判者技能权重并利用后验估计进行推理，在无需参考标签的情况下有效聚合多个LLM评判者的预测，在多个任务上超越了现有聚合方法。

**[Skillverse Tree Eval](llm_nlp/skillverse_tree_eval.md)**

:   提出 SkillVerse，一种无监督的树结构 LLM 诊断框架——用 LLM-as-Judge 批评模型回答后组织为层次化技能树（dendrogram），可在任意粒度上分析模型能力，并用于改善 ICL（提升 25%）和预测新模型弱点（55% 成功率，高于基线 22pp）。

**[Socialeval Evaluating Social Intelligence Of Large Language Models](llm_nlp/socialeval_evaluating_social_intelligence_of_large_language_models.md)**

:   提出基于叙事脚本的双语社会智能基准 SocialEval，通过"世界树"结构整合结果导向的目标达成评估和过程导向的人际能力评估，全面评测 LLM 的社会智能。

**[Songcomposer Llm Lyric Melody Generation](llm_nlp/songcomposer_llm_lyric_melody_generation.md)**

:   提出 SongComposer，首个能同时生成歌词和旋律的大语言模型，通过元组格式对齐歌词与旋律、标量音高初始化和渐进式结构感知训练，在多个歌曲生成任务上超越 GPT-4。

**[Sqlong Enhanced Nl2Sql For Longer Contexts With Llms](llm_nlp/sqlong_enhanced_nl2sql_for_longer_contexts_with_llms.md)**

:   提出 SQLong，一种面向长上下文场景的 NL2SQL 数据增强框架，通过向训练数据中注入采样自其他数据库的合成 CREATE TABLE 语句来扩展上下文长度，使微调后的 LLM 在大规模 Schema 场景下显著提升 SQL 生成准确率。

**[Steering Off Course Reliability Challenges In Steering Language Models](llm_nlp/steering_off_course_reliability_challenges_in_steering_language_models.md)**

:   本文系统性地评估了三种主流的语言模型引导方法（DoLa、功能向量、任务向量）在多达36个模型上的泛化性，发现这些方法存在严重的脆弱性和高方差问题，并揭示了其底层假设的根本缺陷。

**[Stem-Pom Evaluating Language Models Math-Symbol Reasoning In Document Parsing](llm_nlp/stem-pom_evaluating_language_models_math-symbol_reasoning_in_document_parsing.md)**

:   提出 STEM-PoM 基准数据集（2K+ 数学符号实例），将 Part-of-Math Tagging 与文档解析结合，系统评估 LLM 对数学符号上下文多义性的分类能力，并证明符号分类能力的提升可迁移增强下游数学推理表现。

**[Stepwise Reasoning Disruption Attack Of Llms](llm_nlp/stepwise_reasoning_disruption_attack_of_llms.md)**

:   提出SEED（Stepwise rEasoning Error Disruption）攻击方法，通过在LLM推理链的早期步骤中注入细微错误来误导模型产生错误的后续推理和最终答案，在四个数据集和四个模型上验证了高攻击成功率和极低的检测率。

**[Stress-Testing Machine Generated Text Detection Shifting Language Models Writing](llm_nlp/stress-testing_machine_generated_text_detection_shifting_language_models_writing.md)**

:   通过 DPO 微调将 LLM 的写作风格对齐到人类文本的语言特征分布，生成更难被检测的机器文本，揭示了现有 MGT 检测器对浅层语言线索的过度依赖。

**[Structural Reasoning Improves Molecular Understanding Of Llm](llm_nlp/structural_reasoning_improves_molecular_understanding_of_llm.md)**

:   提出 MSR（Molecular Structural Reasoning）框架，通过显式引入六种关键分子结构信息作为推理中间步骤，分别设计分析推理（已知分子）和合成推理（未知分子）两种模式，显著提升化学 LLM 和通用 LLM 在分子相关任务上的表现。

**[Synapticrag Enhancing Temporal Memory Retrieval In Large Language Models Through](llm_nlp/synapticrag_enhancing_temporal_memory_retrieval_in_large_language_models_through.md)**

:   提出 SynapticRAG，借鉴神经科学中突触传播和漏积分发放（LIF）模型，将时序关联触发与语义相似度融合，在对话记忆检索任务上较 SOTA 提升最高 14.66%。

**[Synergizing Unsupervised Episode Detection With Llms For Large-Scale News Events](llm_nlp/synergizing_unsupervised_episode_detection_with_llms_for_large-scale_news_events.md)**

:   本文提出 EpiMine，一种无监督的 episode 检测框架，通过判别性词项共现驱动的文章分割与 LLM 协同，从新闻语料中检测关键事件下的 episode（子事件片段），在三个真实数据集上平均提升 59.2%。

**[Systematic Generalization In Language Models Scales With Information Entropy](llm_nlp/systematic_generalization_in_language_models_scales_with_information_entropy.md)**

:   证明语言模型的系统泛化能力与训练数据中成分分布的信息熵正相关——高熵训练分布下即使没有内置组合先验的标准 seq2seq 模型也能实现强系统泛化。

**[T5Score A Methodology For Automatically Assessing The Quality Of Llm Generated M](llm_nlp/t5score_a_methodology_for_automatically_assessing_the_quality_of_llm_generated_m.md)**

:   提出 T5Score 方法论，将 LLM 生成的自由文本主题集(FT-topics)的质量分解为五个可量化维度（可解释性、主题覆盖、文档覆盖、非重叠性、内部排序），通过简单标注任务实现高标注者一致性，并验证 LLM 可作为自动评估器替代人工。

**[Taxoadapt Aligning Llm-Based Multidimensional Taxonomy Construction To Evolving ](llm_nlp/taxoadapt_aligning_llm-based_multidimensional_taxonomy_construction_to_evolving_.md)**

:   提出 TaxoAdapt 框架，通过层次分类驱动的深度/宽度扩展和分类感知聚类，将 LLM 生成的多维度分类体系动态对齐到特定科学语料库，在粒度保持和兄弟节点一致性上分别超越最优基线 26.51% 和 50.41%。

**[Team Anotheroption At Semeval-2025 Task 8 Bridging The Gap Between Open-Source A](llm_nlp/team_anotheroption_at_semeval-2025_task_8_bridging_the_gap_between_open-source_a.md)**

:   提出一种混合 LLM 管道系统，通过 Text-to-SQL/Code 生成、RAG 检索、自纠错机制和 LLM Orchestrator 协调多个开源模型，在 SemEval-2025 Task 8（表格问答）中达到 80% 准确率（Top-13/38），开源模型组合（88%）超越 GPT-4o 单模型（74%）。

**[Tess 2 A Large-Scale Generalist Diffusion Language Model](llm_nlp/tess_2_a_large-scale_generalist_diffusion_language_model.md)**

:   提出 TESS 2，首个从已有自回归模型适配而来的大规模通用指令遵循扩散语言模型，通过 UL2 masking + label shifting + 双向注意力的适配训练方案 + reward guidance 推理引导，在 QA 和指令遵循任务上匹配甚至超越同等 AR 模型。

**[Testcase Eval Llm Test Gen](llm_nlp/testcase_eval_llm_test_gen.md)**

:   提出 TestCase-Eval 基准评估 LLM 生成算法题测试用例的能力，包含 500 道 Codeforces 算法题和 10 万条人工解答，聚焦两个任务——故障覆盖（测试集能覆盖多少潜在错误）和故障暴露（能否为特定错误代码生成暴露性测试），对 19 个 SOTA LLM 的评估揭示了当前模型在测试生成上的能力和局限。

**[The Nature Of Nlp Analyzing Contributions In Nlp Papers](llm_nlp/the_nature_of_nlp_analyzing_contributions_in_nlp_papers.md)**

:   提出 NLP 论文贡献的分类体系（知识/工件 × 8 子类），构建 ~2k 人工标注数据集 NLPContributions，训练 SciBERT 自动识别贡献声明，并对 ~29k 篇 ACL Anthology 论文做 50 年纵向趋势分析，揭示 NLP 研究从语言学导向转向方法/模型主导、近年又重拾人文与语言关注的演化轨迹。

**[Theory Of Llm Sampling](llm_nlp/theory_of_llm_sampling.md)**

:   本文从认知科学视角揭示了LLM的采样启发式机制与人类决策类似：采样不仅反映概念的统计规范（描述性成分），还系统性地偏向隐含的理想值（规范性成分），这种偏移在500个概念、15个模型上均显著，并可能导致医疗等应用中的有偏决策。

**[Theory Of Mind Llm](llm_nlp/theory_of_mind_llm.md)**

:   系统综述了 LLM 的心智理论（ToM）能力的评估基准（10+ story-based benchmarks）和增强策略（prompt-only 和 fine-tuning 两类方法），指出当前 LLM 在 ToM 推理上仍有显著不足，并提出未来方向。

**[Tigerllm - A Family Of Bangla Large Language Models](llm_nlp/tigerllm_-_a_family_of_bangla_large_language_models.md)**

:   针对孟加拉语（全球第5大语言）的 LLM 严重不足问题，构建高质量教科书语料 Bangla-TextBook（10M token）和原生指令数据 Bangla-Instruct（100K），训练的 TigerLLM 家族在六项基准上超越所有开源替代方案并胜过 GPT-3.5。

**[To Code Or Not To Code Adaptive Tool Integration For Math Language Models Via Ex](llm_nlp/to_code_or_not_to_code_adaptive_tool_integration_for_math_language_models_via_ex.md)**

:   提出基于EM框架的AutoCode方法，让数学LLM自主决定何时使用代码工具辅助推理，通过E-step引导探索高潜力代码触发决策+M-step离线RL优化，7B模型在MATH500上提升11%+。

**[Token Granularity Impact](llm_nlp/token_granularity_impact.md)**

:   本文系统研究了子词 token 粒度（词表大小 256~128K）对语言模型 surprisal 预测人类阅读时间能力的影响，发现约 8K 词表大小的中等粒度 token 在自然阅读时间预测上最优，而更粗粒度（更接近词级）的 token 在花园路径句法效应上表现更敏感。

**[Token Prepending Training Free](llm_nlp/token_prepending_training_free.md)**

:   提出 Token Prepending (TP) 技术，通过在每层将解码得到的句子嵌入前置到句子开头，使因果注意力机制下的早期 token 也能感知完整句子信息，无需训练即可显著提升 LLM 的句子嵌入质量。

**[Token Recycling](llm_nlp/token_recycling.md)**

:   提出Token Recycling——一种无需额外训练的投机解码方法，将解码过程中产生的候选token存入邻接矩阵，通过BFS算法构建draft tree并用tree attention验证，仅需<2MB额外存储即在所有规模LLM上实现约2倍加速，超越现有无训练方法30%和有训练方法25%。

**[Toolcoder A Systematic Code-Empowered Tool Learning Framework For Large Language](llm_nlp/toolcoder_a_systematic_code-empowered_tool_learning_framework_for_large_language.md)**

:   ToolCoder 将工具学习重新建模为代码生成任务，借鉴软件工程的需求分析、模块化设计、代码复用和错误诊断原则，让 LLM 通过生成并执行结构化 Python 代码来调用外部工具，在 RestBench 和 API-Bank 基准上显著超越 ReAct、CodeAct 等现有方法。

**[Toolcoder Code Empowered Tool Learning](llm_nlp/toolcoder_code_empowered_tool_learning.md)**

:   提出 ToolCoder 框架，将工具学习重新定义为代码生成任务，借鉴软件工程原则（需求分析→模块化设计→实现执行→错误调试→代码复用）让 LLM 通过生成和执行 Python 代码来完成多步工具调用，在 RestBench 和 API-Bank 上全面超越 ReAct、CodeAct 等基线方法。

**[Towards Harmonized Uncertainty Estimation For Large Language Models](llm_nlp/towards_harmonized_uncertainty_estimation_for_large_language_models.md)**

:   提出 CUE 框架，通过训练一个与目标 LLM 性能对齐的轻量级分类器（Corrector）来校正现有不确定性估计方法的分数，在指示性、精确-召回平衡和校准三个维度上实现协调一致的改进，最高提升达 60%。

**[Training Language Model To Critique For Better Refinement](llm_nlp/training_language_model_to_critique_for_better_refinement.md)**

:   提出 Refinement-oriented Critique Optimization（RCO），以"批判效用"（Critique Utility, CU）——即批判导致的精炼改善比例——作为奖励信号训练 critic 模型，通过 DPO 变体的 MSE 目标函数优化，无需直接评估批判质量；在对话生成、摘要、问答、数学推理、代码生成 5 个任务上，RCO 训练的 7B/13B critic 模型在 CU 和 RQS 指标上显著超过 70B 基线模型和 DPCO 方法。

**[Transforming Podcast Preview Generation From Expert Models To Llm-Based Systems](llm_nlp/transforming_podcast_preview_generation_from_expert_models_to_llm-based_systems.md)**

:   Spotify 提出用 LLM（Gemini 1.5 Pro）替代传统多模型特征工程流水线来生成播客预览片段，在离线人工评估和线上 A/B 测试中均显著优于传统系统，用户互动时长提升 4.6%，处理效率提升 5 倍。

**[Trates Trait-Specific Rubric-Assisted Cross-Prompt Essay Scoring](llm_nlp/trates_trait-specific_rubric-assisted_cross-prompt_essay_scoring.md)**

:   提出 TRATES 框架，重新定义 LLM 在自动作文评分中的角色——从直接评分者转变为**特质特征生成器与提取器**，通过 LLM 将评分标准(rubric)自动转化为评估问题(子特质)，结合通用写作质量特征和提示特定特征训练回归模型，在 ASAP 数据集 8 个特质上全部达到 SOTA，且首次在 ELLIPSE 数据集上建立跨提示特质评分基线。

**[Tremu Towards Neuro-Symbolic Temporal Reasoning For Llm-Agents With Memory In Mu](llm_nlp/tremu_towards_neuro-symbolic_temporal_reasoning_for_llm-agents_with_memory_in_mu.md)**

:   提出TReMu框架，通过时间感知记忆化（时间线摘要）和神经符号时间推理（LLM生成Python代码执行时间计算），将GPT-4o在多会话对话时间推理基准上的准确率从29.83%提升到77.67%。

**[Un-Considering Contextual Information Assessing Llms Understanding Of Indexical ](llm_nlp/un-considering_contextual_information_assessing_llms_understanding_of_indexical_.md)**

:   首次系统评估 LLM 对英语指示词（I/you/here/tomorrow）的理解能力，发现模型在 "I" 上表现接近人类但在 "you/here/tomorrow" 上严重依赖无关上下文信息而非语法规则，揭示了 LLM 在语用推理上的根本缺陷。

**[Uncertainty Unveiled Can Exposure To More In-Context Examples Mitigate Uncertain](llm_nlp/uncertainty_unveiled_can_exposure_to_more_in-context_examples_mitigate_uncertain.md)**

:   本文系统研究了长上下文 ICL 中增加示例数量对 LLM 预测不确定性的影响，通过不确定性分解揭示性能提升主要源于认知不确定性（EU）的降低，并从残差流投影角度解释了不确定性减少的内部机制。

**[Understanding And Meeting Practitioner Needs When Measuring Representational Har](llm_nlp/understanding_and_meeting_practitioner_needs_when_measuring_representational_har.md)**

:   通过对 12 位负责评估 LLM 系统表征性伤害（representational harms）的从业者进行半结构化访谈，发现公开可用的测量工具普遍无法满足实践者需求——要么因效度/特异性不足而"不好用"（not useful），要么因组织/制度壁垒而"用不了"（not used），并基于测量理论和实用测量框架提出系统性改进建议。

**[Understanding Silent Data Corruption In Llm Training](llm_nlp/understanding_silent_data_corruption_in_llm_training.md)**

:   本文首次系统研究了真实世界静默数据损坏（SDC）对LLM训练的影响，通过将不健康节点与健康节点配对并引入同步机制，在子模块计算、单步梯度、累积训练三个层面揭示了SDC的特征和影响模式。

**[Understanding The Repeat Curse In Large Language Models From A Feature Perspecti](llm_nlp/understanding_the_repeat_curse_in_large_language_models_from_a_feature_perspecti.md)**

:   从机制可解释性角度研究 LLM 重复生成问题（Repeat Curse），用 Sparse Autoencoder 提取单语义特征，定位中间层和最终层的"重复特征"，激活它们可诱导重复、关闭它们可缓解重复且不损害模型性能。

**[Unintended Harms Of Value-Aligned Llms Psychological And Empirical Insights](llm_nlp/unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)**

:   本文首次系统揭示了基于Schwartz价值观对齐的LLM存在非预期安全风险——特定价值维度与特定安全风险类别显著相关，并从心理学视角解释了这些关联的根源，进而提出通过提示抑制相关价值来降低有害行为的缓解策略。

**[Unleashing Llm Reasoning Capability Via Scalable](llm_nlp/unleashing_llm_reasoning_capability_via_scalable.md)**

:   提出 ScaleQuest，通过 Question Fine-Tuning (QFT) + Question Preference Optimization (QPO) 两阶段训练将 7B 解题模型变为出题模型，从零合成 100 万高质量数学问题-解答对，在四个基准上全面超越所有开源数据集，且数据量扩展至 1M 时性能持续提升未见饱和。

**[Unlocking Recursive Thinking Of Llms Alignment Via Refinement](llm_nlp/unlocking_recursive_thinking_of_llms_alignment_via_refinement.md)**

:   提出 AvR（Alignment via Refinement）两阶段框架，通过细化感知奖励（refinement-aware reward）和差分学习，让 LLM 学会"批评→改进"的递归思维能力，仅用 10k 数据即在 AlpacaEval 2 上将 LLaMA-3-8B-Instruct 的胜率提升超 26 个百分点。

**[Veracity Bias Llm Hidden Beliefs](llm_nlp/veracity_bias_llm_hidden_beliefs.md)**

:   揭示了 LLM 在推理任务中存在"真实性偏见"（Veracity Bias）——尽管显式对齐反对刻板印象，LLM 仍系统性地将正确答案归因于特定种族群体（归因偏差），并对相同解答因"作者"种族不同给出不同评价（评估偏差），在数学、编程、常识推理和写作任务中普遍存在。

**[Virsci Multi Agent Idea Gen](llm_nlp/virsci_multi_agent_idea_gen.md)**

:   提出基于 LLM 的多智能体系统 Virtual Scientists（VirSci），模拟真实科研团队的协作过程——组织多个 agent 团队协作生成、评估和改进科研 idea，在生成新颖科学想法方面超越单智能体 SOTA。

**[Warriorcoder Learning From Expert Battles To Augment Code Large Language Models](llm_nlp/warriorcoder_learning_from_expert_battles_to_augment_code_large_language_models.md)**

:   提出 WarriorCoder，通过构建多个专家代码 LLM 之间的竞技场（arena），让攻击者用自身擅长的领域挑战防御者，由裁判评估后用胜者回答训练目标模型，从而无需依赖专有模型或预存数据集即可从零生成高质量、高多样性的代码训练数据，实现 SOTA 性能。

**[When Large Language Models Meet Speech A Survey On Integration Approaches](llm_nlp/when_large_language_models_meet_speech_a_survey_on_integration_approaches.md)**

:   系统综述语音与大语言模型的集成方法，将现有工作分为文本级、隐表示级、音频token级三大类，覆盖 ASR/S2TT/S2ST/TTS 等应用场景，并给出各方法的优劣对比与未来挑战。

**[Which Demographics Do Llms Default To During Annotation](llm_nlp/which_demographics_do_llms_default_to_during_annotation.md)**

:   通过对比 LLM 在无人口统计信息(N)、有人口统计(SD)、安慰剂信息(P)三种 prompt 条件下的标注行为，揭示 LLM 在主观标注任务(冒犯性/礼貌性)中默认更接近白人、年轻、高学历群体的标注模式，且人口统计 prompting 确实产生了比安慰剂信息更系统性的影响。

**[Why Not Act On What You Know Unleashing Safety Potential Of Llms Via Self-Aware ](llm_nlp/why_not_act_on_what_you_know_unleashing_safety_potential_of_llms_via_self-aware_.md)**

:   发现 LLM 作为判别器时能准确识别越狱请求、但作为生成器时却仍产出有害内容的"判别-生成安全差距"，提出免训练的 SAGE（Self-Aware Guard Enhancement）策略，通过判别分析模块和判别响应模块将模型自身的安全鉴别能力桥接到生成行为，在 6 个模型上达到平均 99% 的防御成功率。

**[Zero-Shot Belief A Hard Problem For Llms](llm_nlp/zero-shot_belief_a_hard_problem_for_llms.md)**

:   本文提出了统一式和混合式两种零样本框架用于源-目标信念预测任务，使用 DeBERTa 事件标注器 + LLM 的混合方法在 FactBank 上达到新 SOTA（Full F1 72.0%），同时揭示了嵌套信念预测（Nested F1 仅 25.3%）对 LLM 而言仍是极大挑战。

---

## 🧩 多模态VLM { #multimodal_vlm }

**[Adammeme Adaptively Probe The Reasoning Capacity Of Multimodal Large Language Mo](multimodal_vlm/adammeme_adaptively_probe_the_reasoning_capacity_of_multimodal_large_language_mo.md)**

:   提出AdamMeme——一个基于多智能体协作的自适应评估框架，通过迭代生成更具挑战性的meme样本来探测多模态大语言模型(mLLM)在有害内容理解上的推理能力和特定弱点。

**[Adaptive Linguistic Prompting Alp Enhances Phishing Webpage Detection In Multimo](multimodal_vlm/adaptive_linguistic_prompting_alp_enhances_phishing_webpage_detection_in_multimo.md)**

:   提出 Adaptive Linguistic Prompting (ALP)，一种 8-shot 结构化提示方法，引导多模态 LLM 从 HTML 文本、截图和 URL 三个维度联合推理，检测钓鱼网页，在 GPT-4o 上组合分析达到 F1=0.93，超过传统零样本基线。

**[Adversarial Compositionality Clip](multimodal_vlm/adversarial_compositionality_clip.md)**

:   提出MAC基准和diversity-promoting自训练方法，通过让LLM生成欺骗性文本来系统暴露CLIP等预训练多模态表征的组合性漏洞，在图像/视频/音频三个模态上均显著超越已有方法。

**[Agent Rewardbench](multimodal_vlm/agent_rewardbench.md)**

:   Agent-RewardBench 提出了首个面向多模态 Agent 的奖励建模评测基准，覆盖感知/规划/安全 3 个维度、7 个真实场景、1136 个高质量样本，实验发现即使 GPT-4o 也仅达 61.4% 准确率，揭示了 Agent 奖励建模的巨大挑战。

**[Akan Cinematic Emotions Ace A Multimodal Multi-Party Dataset For Emotion Recogni](multimodal_vlm/akan_cinematic_emotions_ace_a_multimodal_multi-party_dataset_for_emotion_recogni.md)**

:   构建 AkaCE——首个非洲语言多模态对话情感识别数据集，覆盖阿坎语（加纳主要语言，约 2000 万使用者），含 385 段对话 6162 条发言（音频+视觉+文本三模态）、308 名说话人（性别平衡 155男/153女），并提供首个非洲语言词级韵律突出标注。

**[Aligning Vlm Assistants With Personalized Situated](multimodal_vlm/aligning_vlm_assistants_with_personalized_situated.md)**

:   基于社会学"角色集合"(Role-Set) 概念刻画用户多样性，提出 PCogAlign 框架，通过认知感知的动作导向奖励模型来为 VLM 助手生成个性化回复，使不同角色的用户在相同视觉场景下获得最适合自身需求的建议。

**[Alignmmbench Evaluating Chinese Multimodal Alignment In Large Vision-Language Mo](multimodal_vlm/alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)**

:   提出 AlignMMBench，首个面向中文视觉上下文的多模态对齐评测基准，涵盖 3 大类 13 项任务、1054 张图像和 4978 个 QA 对（含单轮/多轮对话），并训练了基于 ChatGLM3-6B 的评估器 CritiqueVLM，其评估一致性超过 GPT-4。

**[Aria-Ui Visual Grounding For Gui Instructions](multimodal_vlm/aria-ui_visual_grounding_for_gui_instructions.md)**

:   提出 Aria-UI，一个专为 GUI 视觉定位设计的纯视觉多模态模型，通过可扩展的指令合成数据管线和文本-图像交错的动作历史机制，在离线和在线 Agent 基准上均达到 SOTA，包括 AndroidWorld 第1名（44.8%）和 OSWorld 第3名（15.2%）。

**[Attacking Vl Agents Popups](multimodal_vlm/attacking_vl_agents_popups.md)**

:   系统性设计了一套对抗性弹窗攻击方法来攻击基于视觉语言模型的计算机操控 agent，在 OSWorld 和 VisualWebArena 上平均攻击成功率达 86%，任务成功率下降 47%，基础防御手段几乎无效。

**[Avg-Llava An Efficient Large Multimodal Model With Adaptive Visual Granularity](multimodal_vlm/avg-llava_an_efficient_large_multimodal_model_with_adaptive_visual_granularity.md)**

:   提出 AVG-LLaVA——在 LLaVA-NeXT 基础上引入视觉粒度缩放器（多级池化获取不同粒度视觉 token）+ 视觉粒度路由器（基于图像和指令自适应选择最合适粒度），并提出 RGLF 训练范式将路由器预测与 LMM 偏好对齐。在 11 个基准上实现更好性能同时大幅减少视觉 token（如 AI2D 上减少 85.3%，推理加速 2.53×）。

**[Branchlora Continual Instruction](multimodal_vlm/branchlora_continual_instruction.md)**

:   发现MoELoRA在多模态持续指令微调(MCIT)中存在参数低效——矩阵A跨任务趋同而B保持区分，提出BranchLoRA：共享一个A矩阵（树干）+ 多个专有B矩阵（树枝） + 灵活调参-冻结机制 + 任务特定路由器，在CoIN benchmark上显著超越MoELoRA，有效缓解灾难性遗忘。

**[Burn After Reading Do Multimodal Large Language Models Truly Capture Order Of Ev](multimodal_vlm/burn_after_reading_do_multimodal_large_language_models_truly_capture_order_of_ev.md)**

:   提出 TempVS 基准测试，系统评估 38 个 MLLM 在图像序列中对多事件时序关系的 grounding 和推理能力，揭示 SOTA 模型与人类之间存在巨大性能差距。

**[Can Mllms Understand The Deep Implication Behind Chinese Images](multimodal_vlm/can_mllms_understand_the_deep_implication_behind_chinese_images.md)**

:   提出 CII-Bench（Chinese Image Implication Understanding Benchmark），包含698张中国互联网/传统文化图像及800道选择题，系统评测MLLM对中文图像深层含义的高阶理解能力，发现最佳模型准确率仅64.4%，远低于人类平均78.2%，且模型在中国传统文化领域表现最差。

**[Can Vision-Language Models Evaluate Handwritten Math](multimodal_vlm/can_vision-language_models_evaluate_handwritten_math.md)**

:   本文提出FERMAT基准，通过609道人工策划的7-12年级数学题及其2200+份手写错误解答（覆盖计算、概念、符号、格式四类错误），系统评估9个VLM在手写数学内容的错误检测、定位和纠正能力，发现Gemini-1.5-Pro达到最高纠错率77%，但所有模型在处理手写内容时仍面临显著挑战。

**[Can Vision Language Models Understand Mimed Actions](multimodal_vlm/can_vision_language_models_understand_mimed_actions.md)**

:   构建 Mime 基准（86 个动作、860 个样本），利用动作捕捉 + 3D 渲染系统化评估 VLM 对哑剧动作的理解能力，发现 VLM 严重依赖场景上下文而无法像人类一样从纯肢体动作推断语义。

**[Cant See The Forest For The](multimodal_vlm/cant_see_the_forest_for_the.md)**

:   提出 MMSafeAware，首个同时评估"不安全内容识别"和"过度敏感"的多模态安全意识基准，包含 1,500 个跨 29 种安全场景的图文对，评估 9 个 MLLM 发现所有模型都存在安全与有用性的严重权衡——GPT-4V 将 36.1% 的不安全输入误判为安全，同时将 59.9% 的安全输入误判为不安全；三种改进方法均无法根本解决问题。

**[Centurio Multilingual Vlm](multimodal_vlm/centurio_multilingual_vlm.md)**

:   系统研究多语言LVLM训练策略，发现可以同时支持100种语言、只需25-50%非英文数据即可大幅提升多语言性能且不损英语性能，最终训练的Centurio在14个任务56种语言上达到SOTA。

**[Chartcoder Chart To Code](multimodal_vlm/chartcoder_chart_to_code.md)**

:   提出首个专用chart-to-code MLLM（ChartCoder），以Code LLM为语言骨干+160K大规模图表-代码数据集+Snippet-of-Thought逐步推理方法，7B模型在三个基准上超越所有开源MLLM，接近GPT-4o水平。

**[Code Guided Text Rich Image](multimodal_vlm/code_guided_text_rich_image.md)**

:   提出CoSyn框架，利用纯文本LLM的代码生成能力自动创建40万张文本丰富图像（图表、文档、图表等）+270万条指令微调数据，训练的7B VLM在7个基准上达到SOTA，超越GPT-4V和Gemini 1.5 Flash。

**[Coling-Unia At Scivqa 2025 Few-Shot Example Retrieval And Confidence-Informed En](multimodal_vlm/coling-unia_at_scivqa_2025_few-shot_example_retrieval_and_confidence-informed_en.md)**

:   本文提出了一种基于多模态大模型（MLLM）集成的科学图表视觉问答系统，通过 few-shot 示例检索策略和置信度感知的模型选择机制，在 SciVQA 2025 共享任务中获得第三名（平均 F1 = 85.12）。

**[Conflictvis Vision Knowledge Conflict](multimodal_vlm/conflictvis_vision_knowledge_conflict.md)**

:   首次系统探索 MLLM 中常识级别的视觉-知识冲突问题，提出自动化框架构建 ConflictVis 基准（374 图 + 1122 QA），发现 MLLM 在约 20% 的冲突场景中过度依赖参数化知识（尤其是 Yes-No 和动作类问题），并提出 Focus-on-Vision 提示策略进行缓解。

**[Cordial Can Multimodal Large Language Models Effectively Understand Coherence Re](multimodal_vlm/cordial_can_multimodal_large_language_models_effectively_understand_coherence_re.md)**

:   本文提出CORDIAL，首个用连贯关系（Coherence Relations）评估MLLM多模态话语分析能力的基准，涵盖灾难管理、社交媒体和在线文章3个话语领域的不同粒度连贯关系，实验发现即使Gemini 1.5 Pro和GPT-4o也无法匹配简单的CLIP分类器基线，揭示了MLLM在语用理解方面的根本不足。

**[Cosyn Code Guided Synthetic Data](multimodal_vlm/cosyn_code_guided_synthetic_data.md)**

:   提出 CoSyn 框架，利用纯文本 LLM 的代码生成能力自动合成多样化的文本丰富型图像及对应指令微调数据，构建 400K 图像 + 2.7M 指令数据集，在 7 个 benchmark 上达到开源 SOTA 并超越 GPT-4V。

**[Cracking Hallucination Vhd](multimodal_vlm/cracking_hallucination_vhd.md)**

:   提出Vision-aware Head Divergence (VHD)指标量化注意力头对视觉信息的敏感度，发现幻觉与模型过度依赖语言先验紧密相关，并提出Vision-aware Head Reinforcement (VHR)无训练方法，通过放大视觉敏感注意力头来缓解幻觉，在CHAIR上最高降低CHAIRS 16.36个点。

**[Craftext Benchmark Advancing Instruction Following In Complex Multimodal Open-En](multimodal_vlm/craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)**

:   提出 CrafText，一个基于 Craftax 开放世界环境的多模态指令跟随基准，包含 3,924 条指令和 3,423 个独特词汇，覆盖定位、条件、建造和成就四类任务，并设计双重评估协议测试智能体的语言泛化和目标泛化能力。

**[Dalr Dual-Level Alignment Learning For Multimodal Sentence Representation Learni](multimodal_vlm/dalr_dual-level_alignment_learning_for_multimodal_sentence_representation_learni.md)**

:   提出 DALR 框架，通过跨模态一致性学习 + 模态内排序蒸馏的双层对齐策略，解决多模态句子表示中的跨模态不对齐偏差（CMB）和模态内语义分歧（ISD）问题，在 STS 和 TR 任务上取得 SOTA。

**[Do Vision-Language Models Have Internal World Models Towards An Atomic Evaluatio](multimodal_vlm/do_vision-language_models_have_internal_world_models_towards_an_atomic_evaluatio.md)**

:   提出基于认知科学的双阶段框架（感知+预测），构建 WM-ABench 大规模基准（23 维度、6 模拟器、10 万+实例），通过 660 组实验系统揭示 15 个 SOTA VLM 在基本世界建模能力上的严重不足。

**[Donate Or Create Comparing Data Collection](multimodal_vlm/donate_or_create_comparing_data_collection.md)**

:   系统比较了两种情感标注数据的收集策略——"捐赠"真实社交媒体帖子 vs "创造"帖子——发现创造的帖子更长、更依赖文本、偏向原型化情感事件，但用创造数据训练的模型可以很好泛化到真实数据，只是需要真实数据来做靠谱的效果评估。

**[Dont Miss The Forest For The Trees Attentional Vision Calibration For Large Visi](multimodal_vlm/dont_miss_the_forest_for_the_trees_attentional_vision_calibration_for_large_visi.md)**

:   发现 LVLM 中存在"blind token"现象——少量语义无关的图像 token 吸引了不成比例的注意力权重，并提出 AvisC 方法通过测试时对比解码重新校准 blind token 影响，有效减轻视觉幻觉。

**[Effivlm Bench Acceleration](multimodal_vlm/effivlm_bench_acceleration.md)**

:   提出 EffiVLM-Bench，首个系统评估大型视觉语言模型（LVLM）训练免加速方法的统一框架，覆盖 17 个 benchmark、3 个前沿模型，引入泛化性和忠诚度等新指标，揭示了 token 压缩与参数压缩在不同场景下的性能-效率权衡。

**[Effivlm Bench Vlm Acceleration](multimodal_vlm/effivlm_bench_vlm_acceleration.md)**

:   提出 EffiVLM-Bench，一个统一评估框架，系统性地评估大型视觉语言模型(LVLM)的免训练加速方法，涵盖 token 压缩和参数压缩两大类，从性能、泛化性、忠实度和效率四个维度进行全面对比分析。

**[Enhance Multimodal Consistency And Coherence For Text-Image Plan Generation](multimodal_vlm/enhance_multimodal_consistency_and_coherence_for_text-image_plan_generation.md)**

:   本文提出一种自回归文本-图像计划生成框架（MPlanner），通过四阶段迭代——文本草拟、图像编辑、视觉信息提取、文本精炼——有效提升多模态计划中视觉步骤的连贯性和文本-图像的一致性。

**[Error-Driven Data-Efficient Large Multimodal Model Tuning](multimodal_vlm/error-driven_data-efficient_large_multimodal_model_tuning.md)**

:   提出一种错误驱动的数据高效微调框架，通过教师模型分析学生模型的错误推理步骤并识别缺失技能，从外部数据集检索针对性训练样本进行微调，无需任务特定数据即可实现平均 7.01% 的性能提升。

**[Evaluating Multimodal Language Models As Visual Assistants For Visually Impaired](multimodal_vlm/evaluating_multimodal_language_models_as_visual_assistants_for_visually_impaired.md)**

:   通过用户调查确定视障人群对 AI 视觉助手的核心需求与挑战，设计涵盖图像描述、多语言VQA、光学盲文识别、视频物体识别、视频问答五大用户中心任务的评估框架，系统评测 12 个 MLLM，揭示当前模型在文化理解、多语言支持、盲文阅读、辅助设备识别和幻觉控制方面的显著不足。

**[Evaluating Visual And Cultural Interpretation The K-Viscuit Benchmark With Human](multimodal_vlm/evaluating_visual_and_cultural_interpretation_the_k-viscuit_benchmark_with_human.md)**

:   本文提出了一种半自动化的文化 VLM 基准构建框架，通过人-VLM 协作生成多选 VQA 样本，并以此构建了聚焦韩国文化的 K-Viscuit 数据集（657 题），揭示了开源与闭源 VLM 在文化理解上的显著差距。

**[Exploring Compositional Generalization Of Multimodal Llms For Medical Imaging](multimodal_vlm/exploring_compositional_generalization_of_multimodal_llms_for_medical_imaging.md)**

:   提出 Med-MAT 数据集（106个医学数据集、53个子集），通过 MAT-Triplet（Modality-Anatomical area-Task）分解医学影像属性，首次系统验证了多模态大模型在医学影像上存在组合泛化（Compositional Generalization）现象，并证明组合泛化是多任务训练泛化增益的关键驱动因素。

**[Exploring How Generative Mllms Perceive More](multimodal_vlm/exploring_how_generative_mllms_perceive_more.md)**

:   系统探究为何生成式多模态LLM（如LLaVA）使用与CLIP相同的视觉编码器却能在视觉推理任务上大幅超越CLIP，发现patch token、位置编码和prompt加权是关键因素。

**[Fiha Autonomous Hallucination Evaluation In Vision-Language Models With Davidson](multimodal_vlm/fiha_autonomous_hallucination_evaluation_in_vision-language_models_with_davidson.md)**

:   本文提出 FIHA，一个无需 LLM 和人工标注的自动化细粒度幻觉评估框架，通过从图像和描述中提取实体、属性和关系生成 Q&A 对，并引入 Davidson 场景图（DSG）建模问题间的依赖关系，构建了 FIHA-v1 基准，全面评估了主流大视觉语言模型的幻觉水平。

**[Filter-And-Refine A Mllm Based Cascade System For Industrial-Scale Video Content](multimodal_vlm/filter-and-refine_a_mllm_based_cascade_system_for_industrial-scale_video_content.md)**

:   TikTok提出一种基于MLLM的两阶段级联内容审核系统（Router-Ranker），通过轻量级嵌入检索路由器过滤97.5%的合规流量，仅将高风险视频送入微调后的LLaVA进行精细分类，F1提升66.5%的同时部署成本降至直接全量部署的1.5%。

**[Finmme Benchmark Dataset For Financial Multi-Modal Reasoning Evaluation](multimodal_vlm/finmme_benchmark_dataset_for_financial_multi-modal_reasoning_evaluation.md)**

:   构建了一个包含 11,000+ 高质量金融多模态样本的评估基准 FinMME，涵盖 18 个金融领域和 10 种图表类型，提出了融合幻觉惩罚和领域归一化的 FinScore 评估体系，实验表明即使 GPT-4o 也仅得 47 分，揭示了 MLLM 在金融领域的显著不足。

**[Flagevalmm A Flexible Framework For Comprehensive Multimodal Model Evaluation](multimodal_vlm/flagevalmm_a_flexible_framework_for_comprehensive_multimodal_model_evaluation.md)**

:   提出 FlagEvalMM，一个开源的多模态模型评估框架，通过将模型推理与评估过程解耦的架构设计，统一支持视觉语言理解（VQA）、文生图/文生视频生成和图文检索等多种多模态任务的评估。

**[Harnessing Pdf Data For Improving Japanese Large Multimodal Models](multimodal_vlm/harnessing_pdf_data_for_improving_japanese_large_multimodal_models.md)**

:   提出一套全自动 PDF 数据提取管道，从日语 PDF 中提取图文对并生成指令数据，通过持续微调 LLaVA1.5 框架显著提升日语多模态模型性能，在 Heron-Bench 上实现 2.1%~13.8% 的提升。

**[Hidellava Hierarchical Decoupling For Continual Instruction](multimodal_vlm/hidellava_hierarchical_decoupling_for_continual_instruction.md)**

:   通过 CKA 分析发现 MLLM 顶层学任务特异信息而其余层学通用知识，提出 HiDe-LLaVA：顶层 LoRA 做 MoE 式任务特异扩展（双模态锚点匹配）+ 其余层 LoRA 做均匀融合，在新构建的无信息泄露基准 UCIT 上比最佳基线提升 5.8%。

**[Hierarchical Safety Realignment Lightweight Restoration Of Safety In Pruned Larg](multimodal_vlm/hierarchical_safety_realignment_lightweight_restoration_of_safety_in_pruned_larg.md)**

:   提出层次化安全重对齐方法HSR，通过先识别安全关键注意力头、再在这些头中定位并恢复被剪枝的安全关键神经元，以极低参数开销（万分之几）显著恢复被剪枝LVLM丢失的安全性能。

**[Hotelmatch Llm Retrieval](multimodal_vlm/hotelmatch_llm_retrieval.md)**

:   提出 HotelMatch-LLM，用 SLM 编码 query + LLM 编码酒店文档的非对称架构，配合三目标多任务优化（检索对齐 + MLM地理预测 + 视觉设施识别）和 patch 级 mean pooling 多图处理，在旅行领域多模态检索任务上显著超过 MARVEL/VISTA 等 SOTA。

**[Hscr Hierarchical Self-Contrastive Rewarding For Aligning Medical Vision Languag](multimodal_vlm/hscr_hierarchical_self-contrastive_rewarding_for_aligning_medical_vision_languag.md)**

:   提出层级自对比奖励方法 HSCR，通过视觉 token dropout 暴露模型内在的模态失对齐（misalignment），自动生成高质量偏好数据，并结合显式/隐式多层级偏好优化，仅用2000条训练样本即显著提升医学VLM的零样本性能和可信度。

**[Improving Medical Large Vision-Language Models With Abnormal-Aware Feedback](multimodal_vlm/improving_medical_large_vision-language_models_with_abnormal-aware_feedback.md)**

:   提出 UMed-LVLM，通过 Abnormal-Aware Instruction Tuning 和 Abnormal-Aware Rewarding（包含 Relevance Reward、Abnormal Localization Reward、Vision Relevance Reward）训练策略增强医学 LVLM 的异常区域定位能力，在 MAU 数据集上比基线提升 58%，并展现出优秀的跨模态和 OOD 泛化能力。

**[Inews A Multimodal Dataset For Modeling Personalized Affective Responses To News](multimodal_vlm/inews_a_multimodal_dataset_for_modeling_personalized_affective_responses_to_news.md)**

:   构建了一个包含 291 位英国标注者对 2,899 条 Facebook 多模态新闻帖子的个性化情感标注数据集 iNews，标注者特征（人口统计、人格、媒体信任等）可解释 15.2% 的标注方差，结合 persona 信息的 LLM 零样本预测准确率提升最高 7%。

**[Inference Compute Optimal Video Vlm](multimodal_vlm/inference_compute_optimal_video_vlm.md)**

:   首次系统研究视频VLM推理时的计算预算最优分配问题：在固定推理FLOPs下，如何在语言模型大小(x_N)、帧数(x_T)和每帧视觉token数(x_V)三个维度间做最优权衡，通过大规模训练扫描（~100k A100小时）和参数化建模得出实用的分配策略。

**[Internlm-Xcomposer25-Reward A Simple Yet Effective Multi-Modal Reward Model](multimodal_vlm/internlm-xcomposer25-reward_a_simple_yet_effective_multi-modal_reward_model.md)**

:   基于InternLM-XComposer2.5构建判别式多模态奖励模型IXC-2.5-Reward，通过精心构建跨文本/图像/视频的多领域偏好数据集训练，在多模态奖励基准VL-RewardBench上以70.0% Macro Acc超越GPT-4o（62.4%），并展示了RL训练、Best-of-N测试时缩放和数据清洗三大应用。

**[Jailbreak Large Vision-Language Models Through Multi-Modal Linkage](multimodal_vlm/jailbreak_large_vision-language_models_through_multi-modal_linkage.md)**

:   提出多模态链接（MML）攻击框架，通过跨模态加密-解密机制和"邪恶对齐"策略，以极高成功率（GPT-4o上达99%+）越狱当前最先进的视觉语言模型。

**[Jarvis-Vla Post-Training Large-Scale Vision Language Models To Play Visual Games](multimodal_vlm/jarvis-vla_post-training_large-scale_vision_language_models_to_play_visual_games.md)**

:   提出ActVLP训练范式，在动作模仿学习之前增加视觉语言后训练阶段（世界知识、视觉对齐、空间定位），构建首个能在Minecraft中执行1000+原子任务的VLA模型JARVIS-VLA，相比最佳基线提升40%。

**[Judging The Judges Can Large Vision-Language Models Fairly Evaluate Chart Compre](multimodal_vlm/judging_the_judges_can_large_vision-language_models_fairly_evaluate_chart_compre.md)**

:   系统评估了 13 个开源小型 LVLM（≤9B 参数）作为图表理解和推理任务的评判者，发现部分开源模型（如 LLaVA-Critic-7B）可达到接近 GPT-4 水平的评判能力（约 80% 一致率），但位置偏差和长度偏差等问题仍然普遍存在。

**[Logicqa Logical Anomaly Detection With Vision Language Model Generated Questions](multimodal_vlm/logicqa_logical_anomaly_detection_with_vision_language_model_generated_questions.md)**

:   提出 LogicQA 框架，利用预训练 VLM 自动生成异常相关问题并通过问答投票机制检测逻辑异常，在无需训练、无需标注的少样本设置下达到 SOTA 性能，同时提供自然语言的异常原因解释。

**[Longdocurl Multimodal Long Doc](multimodal_vlm/longdocurl_multimodal_long_doc.md)**

:   提出LongDocURL基准（396篇文档、2,325 QA对、33,000+页，含理解/推理/定位三类20个子任务），通过半自动构建流程生成高质量长文档评测数据，26种配置的实验结果显示最强GPT-4o仅64.5分，开源最高30.6分，远低于人类84.8分。

**[Madakv Adaptive Modality-Perception Kv Cache Eviction For Efficient Multimodal L](multimodal_vlm/madakv_adaptive_modality-perception_kv_cache_eviction_for_efficient_multimodal_l.md)**

:   本文提出MadaKV，一种模态感知的KV缓存逐出策略，通过模态偏好自适应（MPA）和层级压缩补偿（HCC）两个组件，在保持多模态长上下文任务性能的同时，显著降低KV缓存内存占用（80-95%）和解码延迟（1.3-1.5倍加速）。

**[Magic-Vqa Multimodal And Grounded Inference With Commonsense Knowledge For Visua](multimodal_vlm/magic-vqa_multimodal_and_grounded_inference_with_commonsense_knowledge_for_visua.md)**

:   提出MAGIC-VQA框架，通过三阶段流程（显式知识检索→按类型后处理→GNN隐式增强）将外部常识知识系统地注入LVLM，在ScienceQA、TextVQA、MMMU等基准上实现即插即用的常识推理增强，仅需0.33M可训练参数。

**[Mammoth Vl Multimodal Reasoning](multimodal_vlm/mammoth_vl_multimodal_reasoning.md)**

:   MAmmoTH-VL 提出了一种仅用开源模型构建 12M 多模态 CoT 推理指令数据的可扩展方法，通过数据收集→改写→自过滤三步管线，训练的 8B 模型在 MathVerse (+8.1%)、MMMU-Pro (+7%)、MuirBench (+13.3%) 上达到 SOTA。

**[Manu Modality Aware Unlearning](multimodal_vlm/manu_modality_aware_unlearning.md)**

:   提出MANU框架解决MLLM中的多模态遗忘不平衡问题：通过四种重要性函数（绝对/频率/方差/RMS）识别跨模态知识纠缠的神经元，选择性剪枝实现多模态输入和纯文本输入下的均衡知识遗忘，同时保持模型通用能力。

**[Mathcoder-Vl Bridging Vision And Code For Enhanced Multimodal Mathematical Reaso](multimodal_vlm/mathcoder-vl_bridging_vision_and_code_for_enhanced_multimodal_mathematical_reaso.md)**

:   提出利用代码作为跨模态对齐的监督信号，构建860万图像-代码对数据集ImgCode-8.6M和300万多模态数学指令微调数据集MM-MathInstruct-3M，训练的MathCoder-VL在开源模型中达到多模态数学推理SOTA，在几何问题上超越GPT-4o和Claude 3.5 Sonnet。

**[Mcts Video Captioning Eval](multimodal_vlm/mcts_video_captioning_eval.md)**

:   提出AutoCaption框架，利用蒙特卡洛树搜索(MCTS)自动迭代生成细粒度视频描述关键点（平均122个/视频），构建MCTS-VCB基准评估20+个MLLM的视频描述能力，并证明生成的数据可用于微调显著提升模型性能。

**[Megapairs Massive Data Synthesis For Universal Multimodal Retrieval](multimodal_vlm/megapairs_massive_data_synthesis_for_universal_multimodal_retrieval.md)**

:   提出 MegaPairs 数据合成方法，利用异构 KNN 三元组从开放域图像语料中挖掘相关图像对，结合 VLM/LLM 生成检索指令，合成 2600 万多模态训练实例，训练的 MMRet 模型仅用 0.5M 数据即超越使用 36.7M 数据的 MagicLens（70× 数据效率），在 4 个 CIR 基准和 MMEB 36 个数据集上达到 SOTA。

**[Meit Multimodal Electrocardiogram Instruction Tuning On Large Language Models Fo](multimodal_vlm/meit_multimodal_electrocardiogram_instruction_tuning_on_large_language_models_fo.md)**

:   提出 MEIT 框架，通过多模态指令微调将 ECG 信号与 LLM 对齐，利用轻量级拼接融合策略（无需额外参数）在 LLM 的自注意力层中注入 ECG 嵌入，实现自动 ECG 报告生成，并建立涵盖质量评估、零样本迁移、噪声鲁棒性和专家对齐四项任务的综合基准。

**[Mire Enhancing Multimodal Queries Representation Via Fusion-Free Modality Intera](multimodal_vlm/mire_enhancing_multimodal_queries_representation_via_fusion-free_modality_intera.md)**

:   提出MIRe框架，通过"无融合模态交互"（fusion-free modality interaction）在视觉-文本对齐阶段避免直接融合文本特征，利用查询引导注意力池化模块让文本嵌入引导视觉信息提取但不将文本信号反馈回视觉表示，有效缓解多模态检索中的文本主导问题，在四个基准上取得零样本SOTA。

**[Mixture Of Decoding An Attention-Inspired Adaptive Decoding Strategy To Mitigate](multimodal_vlm/mixture_of_decoding_an_attention-inspired_adaptive_decoding_strategy_to_mitigate.md)**

:   提出了 Mixture of Decoding (MoD)，通过 JS 散度衡量模型对图像 token 注意力的正确性，在注意力正确时采用互补解码放大关键信息，注意力错误时采用对比解码抑制误导信息，从而自适应地缓解多模态大模型的幻觉问题。

**[Mmboundary Advancing Mllm Knowledge Boundary Awareness Through Reasoning Step Co](multimodal_vlm/mmboundary_advancing_mllm_knowledge_boundary_awareness_through_reasoning_step_co.md)**

:   提出 MMBoundary 框架，通过对 MLLM 推理链中每一步进行置信度校准（而非仅对整体回答），结合文本+跨模态自奖励信号与强化学习，显著降低多模态置信度校准误差（平均 7.5%）并提升任务性能（最高 8.3%）。

**[Mmboundary Reasoning Step Confidence](multimodal_vlm/mmboundary_reasoning_step_confidence.md)**

:   提出 MMBoundary 框架，通过在推理链的每一步插入自然语言置信度表述（而非只在最终回答后给置信度），结合文本+跨模态的自奖励信号估计置信度，并用 SFT+RL 两阶段训练实现步级置信度校准，平均降低 7.5% 校准误差并提升 8.3% 任务准确率。

**[Mmina Benchmarking Multihop Multimodal Internet Agents](multimodal_vlm/mmina_benchmarking_multihop_multimodal_internet_agents.md)**

:   提出MMInA基准，包含1,050个人工编写的多跳多模态网页任务（覆盖14个真实动态网站，平均2.85跳），并设计逐跳评估协议和记忆增强方法，揭示当前最强Agent（GPT-4V仅21.8%任务成功率）在多跳网页导航上与人类（96.3%）的巨大差距。

**[Mmmu Pro Robust Benchmark](multimodal_vlm/mmmu_pro_robust_benchmark.md)**

:   提出 MMMU-Pro 基准，通过过滤纯文本可解题目、扩增选项至 10 个、引入 vision-only 输入设置三步法，构建更鲁棒的多学科多模态理解评测，所有模型性能显著下降 16.8%-26.9%。

**[Mmmupro A More Robust Multidiscipline Multimodal](multimodal_vlm/mmmupro_a_more_robust_multidiscipline_multimodal.md)**

:   MMMU-Pro 通过三步流程（过滤纯文本可解题、扩展选项至 10 个、引入纯视觉输入设置）构建更鲁棒的多学科多模态理解基准，模型性能比原 MMMU 下降 16.8%~26.9%，揭示当前模型依赖捷径而非真正多模态理解。

**[Mmscibench Benchmarking Language Models On Chinese Multimodal Scientific Problem](multimodal_vlm/mmscibench_benchmarking_language_models_on_chinese_multimodal_scientific_problem.md)**

:   提出 MMSciBench，一个包含 4,482 道中文高中数学和物理题目的多模态科学推理基准，涵盖选择题和问答题、纯文本和图文配对两种模态，并带有人工标注难度等级和三级知识分类体系；评估显示最强模型 Gemini 1.5 Pro 002 仅达 63.77% 准确率，且在图文题上性能大幅下降（36.28 个百分点差距）。

**[Multimm Cultural Metaphor](multimodal_vlm/multimm_cultural_metaphor.md)**

:   提出MultiMM——首个跨文化多模态隐喻数据集，包含8461个中英文广告图文对及细粒度标注，并设计SEMD模型融合情感特征增强隐喻检测。

**[Multimodal Coreference Resolution For Chinese Social Media Dialogues Dataset And](multimodal_vlm/multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)**

:   提出 TikTalkCoref，首个面向中文社交媒体对话的多模态共指消解数据集（基于抖音短视频），并构建了包含文本共指消解、视觉人物追踪和跨模态对齐三个模块的 pipeline benchmark。

**[Negvqa Can Vision Language Models Understand Negation](multimodal_vlm/negvqa_can_vision_language_models_understand_negation.md)**

:   提出 NegVQA 基准（7,379 道二选一 VQA 题），系统评估 20 个 VLM 对否定句的理解能力，发现所有模型在否定问题上性能大幅下降（平均 29.7%），并揭示"U 型"缩放趋势。

**[Omgm Orchestrate Multiple Granularities And Modalities For Efficient Multimodal ](multimodal_vlm/omgm_orchestrate_multiple_granularities_and_modalities_for_efficient_multimodal_.md)**

:   提出OMGM——一个面向知识密集型视觉问答(KB-VQA)的多模态RAG系统，通过粗到细三步检索策略协调查询与知识库在不同粒度和模态间的匹配，在InfoSeek和E-VQA上取得SOTA检索性能和极具竞争力的问答结果。

**[Omnialign-V Towards Enhanced Alignment Of Mllms With Human Preference](multimodal_vlm/omnialign-v_towards_enhanced_alignment_of_mllms_with_human_preference.md)**

:   构建了 OmniAlign-V（200K 高质量多模态 SFT 数据集）和 MM-AlignBench 评测基准，通过多样化图片来源、开放式问题设计和多样化回答格式，显著提升开源 MLLM 的人类偏好对齐能力，使 LLaVA-Next-32B 经 SFT+DPO 后超越 Qwen2VL-72B。

**[Patent Analysis Survey](multimodal_vlm/patent_analysis_survey.md)**

:   全面综述基于 NLP 和多模态 AI 的专利分析方法，按专利生命周期中的四大任务（分类、检索、质量分析、生成）提出新的分类体系，覆盖从传统神经网络到 PLM/LLM 的技术演进。

**[Performance Gap In Entity Knowledge Extraction Across Modalities In Vision Langu](multimodal_vlm/performance_gap_in_entity_knowledge_extraction_across_modalities_in_vision_langu.md)**

:   系统性地揭示了视觉语言模型（VLM）在视觉 vs 文本表征下实体知识提取的显著性能差距（最高达 18%），通过机制可解释性工具发现图像 token 的关键信息流发生在模型中间层很深处，导致后续事实推理的层数不足。

**[Progressive Multimodal Reasoning Via Active Retrieval](multimodal_vlm/progressive_multimodal_reasoning_via_active_retrieval.md)**

:   本文提出AR-MCTS框架，将主动检索（Active Retrieval）与蒙特卡洛树搜索（MCTS）结合，在多步多模态推理的每一步动态检索关键知识来替代传统beam search采样，自动生成逐步推理标注以渐进式对齐过程奖励模型（PRM），在MathVista、We-Math和GAOKAO-MM上显著提升了多种MLLM的推理性能。

**[Punchbench Mllm Punchline](multimodal_vlm/punchbench_mllm_punchline.md)**

:   提出PunchBench（6,000图文对、54,000 QA对），通过同义/反义标题替换消除文本捷径、多种问题格式与两层任务（感知+推理）全面评测MLLM的多模态梗图理解能力，并设计SC-CoQ由简到繁的提问策略显著提升表现，但GPT-4o仍远低于人类98%+水平。

**[R-Vlm Region-Aware Vision Language Model For Precise Gui Grounding](multimodal_vlm/r-vlm_region-aware_vision_language_model_for_precise_gui_grounding.md)**

:   提出R-VLM，将传统目标检测中的区域提议（region proposal）和IoU感知损失引入VLM的GUI元素定位，通过两阶段放大推理和IoU加权交叉熵损失，在ScreenSpot和AgentStudio上平均提升13%的grounding准确率。

**[Rate-Nav Region-Aware Termination Enhancement For Zero-Shot Object Navigation Wi](multimodal_vlm/rate-nav_region-aware_termination_enhancement_for_zero-shot_object_navigation_wi.md)**

:   提出 RATE-Nav，一种基于边际效用理论的零样本目标导航方法，通过几何预测区域分割和基于区域的探索率估计，结合 VLM 的宏观环境感知能力智能判断是否终止当前区域的探索，在 HM3D 上达到 67.8% 成功率和 31.3% SPL，在 MP3D 上比先前零样本方法提升约 10%。

**[Real-Mm-Rag A Real-World Multi-Modal Retrieval Benchmark](multimodal_vlm/real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)**

:   提出 REAL-MM-RAG 多模态文档检索基准，定义了真实世界检索基准的四大关键属性（多模态文档、增强难度、真实 RAG 查询、准确标注），引入多级查询改写鲁棒性评估，并通过针对性训练集（改写数据集+金融表格数据集）实现 SOTA 检索性能。

**[Redundancy Principles For Mllms Benchmarks](multimodal_vlm/redundancy_principles_for_mllms_benchmarks.md)**

:   本文从维度冗余、实例冗余和跨基准冗余三个层面系统量化了当前MLLM评测基准中的冗余现象，提出了基于性能排名相关性的冗余分析框架，为未来基准设计提供了原则性指导。

**[Redundancylens Revealing And Exploiting Visual Token Processing Redundancy For E](multimodal_vlm/redundancylens_revealing_and_exploiting_visual_token_processing_redundancy_for_e.md)**

:   提出 RedundancyLens 框架，系统揭示了 decoder-only MLLM 中视觉 token 在自注意力和 FFN 操作上存在大量结构化、聚簇式冗余，并利用这一发现实现免训练推理加速，与现有 token 压缩方法正交且可组合。

**[Reefknot A Comprehensive Benchmark For Relation Hallucination Evaluation Analysi](multimodal_vlm/reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)**

:   提出首个系统性评估多模态大模型**关系级幻觉**的综合基准 Reefknot（含 2 万+ 样本、三种任务），并基于置信度熵检测提出 Detect-then-Calibrate 缓解策略，平均降低幻觉率 9.75%。

**[Response Wide Shut Surprising Observations In Basic Vision Language Model Capabi](multimodal_vlm/response_wide_shut_surprising_observations_in_basic_vision_language_model_capabi.md)**

:   通过在VLM的三个中间特征空间（视觉编码器、VL投影层、语言解码器）上训练线性探针，系统揭示了一个反直觉的现象：对于大多数视觉任务，视觉编码器和VL投影层其实保留了充分的视觉信息，真正的瓶颈在于语言解码器的响应空间——信息在从投影层传递到最终文本输出的过程中大量丢失。

**[Retrieval Visual Contrastive Decoding To Mitigate Object Hallucinations In Large](multimodal_vlm/retrieval_visual_contrastive_decoding_to_mitigate_object_hallucinations_in_large.md)**

:   提出 RVCD（Retrieval Visual Contrastive Decoding），通过检索 AI 生成的单概念显式图像构建正/负 logit 集合，在解码阶段抑制 LVLM 的物体幻觉（Object Hallucination），无需额外训练即可显著优于现有解码方法。

**[Scalable Vision Language Model Training Via High Quality Data Curation](multimodal_vlm/scalable_vision_language_model_training_via_high_quality_data_curation.md)**

:   提出 SAIL-VL 系列开源视觉语言模型（2B/8B），核心贡献在于：构建了3亿规模最高质量的 SAIL-Caption 数据集，首次揭示了VLM预训练中的数据量对数缩放定律（655B token实验），并通过课程式三阶段SFT将缩放曲线从对数提升至近线性，在18个基准上达到SOTA。

**[Semeval-2025 Task 1 Admire -- Advancing Multimodal Idiomaticity Representation](multimodal_vlm/semeval-2025_task_1_admire_--_advancing_multimodal_idiomaticity_representation.md)**

:   设计了 SemEval-2025 AdMIRe 共享任务——通过图像排序和图像序列补全两个子任务，在多模态（文本+图像）和多语言（英语+巴西葡萄牙语）场景下评估模型对习语表达的理解能力，最佳系统通过混合专家和多查询平滑策略达到了接近人类水平的表现。

**[Singakids A Multilingual Multimodal Dialogic Tutor For Language Learning](multimodal_vlm/singakids_a_multilingual_multimodal_dialogic_tutor_for_language_learning.md)**

:   提出 SingaKids 系统，一个面向小学生的多语言多模态对话式语言学习辅导系统，通过图像描述任务整合稠密图像字幕、多语言对话、语音理解和儿童友好语音生成，支持英语、中文、马来语和泰米尔语四种语言的互动学习。

**[Sophia Efficient Long Video](multimodal_vlm/sophia_efficient_long_video.md)**

:   提出Sophia模型处理小时级长视频：通过Shot-adaptive Frame Pruning（基于镜头分割的两阶段帧剪枝）精准选择查询相关帧，结合O(N)复杂度的Hierarchical Attention替代全注意力，在8/8个长视频benchmark中6个SOTA，且注意力FLOPs仅为InternVL2的1/17。

**[Spatialmqa Mllm Spatial Relations](multimodal_vlm/spatialmqa_mllm_spatial_relations.md)**

:   提出SpatialMQA基准，通过5,392个基于COCO2017的人工标注多选题样本（无bbox、含视角替换、排除纯知识可答题），评测MLLM的空间关系推理能力，发现最强模型SpaceLLaVA仅达48.14%远低于人类98.40%。

**[Speaking Beyond Language](multimodal_vlm/speaking_beyond_language.md)**

:   提出 VENUS——首个大规模多模态对话数据集（89,459 段对话、14,910 小时），包含时间对齐的文本、3D 面部表情和肢体语言标注；基于该数据集开发 MARS 多模态语言模型，通过 VQ-VAE 将非语言线索离散化后与文本统一建模，实现对话中文本与非语言动作的联合理解和生成。

**[Sphere Unveiling Spatial Blind Spots In](multimodal_vlm/sphere_unveiling_spatial_blind_spots_in.md)**

:   提出 SPHERE 三层级空间推理评估框架（单技能→多技能→推理），基于 MS COCO 人工标注 2285 个 QA 对，发现 GPT-4o（67.9%）与人类（93.0%）差距 25%，尤其在距离判断、视角切换和物理推理上表现严重不足。

**[Symmetrical Visual Contrastive Optimization Aligning Visionlanguage](multimodal_vlm/symmetrical_visual_contrastive_optimization_aligning_visionlanguage.md)**

:   提出 S-VCO（对称视觉对比优化），一种新的 VLM 微调目标，通过对称地对齐/拒绝匹配/矛盾的图像-文本对来增强视觉依赖，配合最小视觉对比数据集 MVC，在幻觉检测上减少 22%，视觉依赖任务上显著提升。

**[Table Understanding And Multimodal Llms A Cross-Domain Case Study On Scientific ](multimodal_vlm/table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)**

:   提出 TableEval 基准（3017 张表格，5 种格式），系统比较了文本 LLM 和多模态 LLM 在科学 vs. 非科学表格理解任务上的表现，发现模型对表格模态（图像/文本）保持鲁棒但在科学表格上性能显著下降。

**[Teaching Vlm Ask Ambiguity](multimodal_vlm/teaching_vlm_ask_ambiguity.md)**

:   提出ClearVQA benchmark评估VLM处理歧义视觉问题的能力（覆盖指代歧义/意图欠明确/拼写歧义三类），并通过自动化pipeline生成歧义-澄清问题对用于SFT+DPO训练，使VLM学会"反问用户"而非猜测作答，VQA准确率相对提升13.3%。

**[Theorem Explain Agent](multimodal_vlm/theorem_explain_agent.md)**

:   提出 TheoremExplainAgent，一个基于多 Agent 协作的系统，能自动生成 5 分钟以上的定理讲解视频（Manim 动画+语音旁白），并构建了 TheoremExplainBench（240 个 STEM 定理，5 维评估指标）用于系统评估。

**[Token Pruning In Multimodal Large Language Models Are We Solving The Right Probl](multimodal_vlm/token_pruning_in_multimodal_large_language_models_are_we_solving_the_right_probl.md)**

:   通过大规模基准实验揭示了当前MLLM视觉token剪枝方法的多个根本性问题：精心设计的剪枝策略（FastV、SparseVLM）在多数基准上甚至不如随机选择和池化等朴素方法，原因在于注意力评分的位置偏差、对语言信息的误用、重要性与冗余性的失衡以及评估指标的不可靠。

**[Transferring Textual Preferences To Vision-Language Understanding Through Model ](multimodal_vlm/transferring_textual_preferences_to_vision-language_understanding_through_model_.md)**

:   提出一种免训练方法，通过模型参数合并（model merging）将纯文本奖励模型（RM）的偏好能力迁移到大视觉语言模型（LVLM）中，构建视觉语言奖励模型（VLRM），在多个多模态评估基准上超越LVLM直接评分和纯文本RM。

**[Trimllm Layer Dropping](multimodal_vlm/trimllm_layer_dropping.md)**

:   基于"层级特化"现象（不同层对不同领域重要性不同），提出TrimLLM在领域微调过程中渐进式丢弃最不重要的层，将LLaMA-7B压缩到50%大小时性能几乎无损，在消费级GPU上实现2.1-5.7×推理加速——无需任何特殊硬件或kernel支持。

**[Tvc Mitigating Visual Forgetting](multimodal_vlm/tvc_mitigating_visual_forgetting.md)**

:   发现MLLM长链推理中存在严重的"视觉遗忘"现象——推理进行到一半时移除图片仅导致2%精度下降，说明模型过度依赖文本输出而非视觉输入。提出Take-along Visual Conditioning (TVC)，在推理过程中周期性重新注入压缩后的图像特征，在5个数学推理基准上平均超越之前SOTA 3.4个点。

**[Unsolvable Problem Detection](multimodal_vlm/unsolvable_problem_detection.md)**

:   本文提出不可解问题检测（UPD）任务评估大型多模态模型（LMM）的鲁棒理解能力，包含三种不可解场景（答案缺失AAD、答案集不兼容IASD、图像问题不兼容IVQD），构建 MM-UPD Bench 基准，实验揭示现有 LMM 在标准 MCQA 上表现良好但在 UPD 上显著挣扎，且现有基准性能与 UPD 能力之间缺乏相关性。

**[Unveiling The Lack Of Lvlm Robustness To Fundamental Visual Variations Why And P](multimodal_vlm/unveiling_the_lack_of_lvlm_robustness_to_fundamental_visual_variations_why_and_p.md)**

:   提出 V2R-Bench 基准框架系统评估 21 个 LVLM 对位置/尺度/方向/上下文四种基本视觉变化的鲁棒性，揭示了即使先进模型在简单视觉任务上也存在显著脆弱性，并通过组件级分析证明这些漏洞根源在于多模态对齐不足和流水线架构的误差累积，而非数据不足。

**[Value Spectrum Vlm Pref](multimodal_vlm/value_spectrum_vlm_pref.md)**

:   提出 Value-Spectrum 基准，通过 50K+ 社交媒体短视频截图和 Schwartz 价值理论框架，系统评估 VLM 的内在价值偏好及角色扮演时的偏好适配能力。

**[Vf Eval Aigc Video Feedback](multimodal_vlm/vf_eval_aigc_video_feedback.md)**

:   提出VF-Eval基准，通过一致性验证、错误感知、错误类型检测、推理评估四大任务系统评估13个MLLM为AIGC视频提供反馈的能力，发现即使GPT-4.1也难以在所有任务上表现一致，揭示了AIGC视频理解的挑战性。

**[Vigil3D A Linguistically Diverse Dataset For 3D Visual Grounding](multimodal_vlm/vigil3d_a_linguistically_diverse_dataset_for_3d_visual_grounding.md)**

:   提出 ViGiL3D——一个语言多样性诊断数据集和自动化分析框架，用于评估 3D 视觉定位（3DVG）方法在否定、粗粒度指代、共指消解等多种语言现象上的表现，揭示现有方法在分布外提示上性能显著下降（最高达 20+ 点）。

**[Vision-Language Models Struggle To Align Entities Across Modalities](multimodal_vlm/vision-language_models_struggle_to_align_entities_across_modalities.md)**

:   提出 MATE 基准测试，包含 5,500 个评估实例，通过问答任务评估视觉语言模型（VLM）的跨模态实体链接能力，发现 VLM 在将同一实体的视觉和文本表示进行对齐方面远不如人类，尤其随场景中物体数量增加性能急剧下降。

**[Visual Evidence Prompting](multimodal_vlm/visual_evidence_prompting.md)**

:   提出Visual Evidence Prompting (VEP)，利用小型视觉专家模型（目标检测器、场景图生成器）的输出作为文本化"视觉证据"输入LVLM，无需训练即可在11个LVLM上显著降低幻觉——LLaVA-1.5在POPE上提升7.2%、Claude 3上提升12.1%。

**[Vlm2-Bench A Closer Look At How Well Vlms Implicitly Link Explicit Matching Visu](multimodal_vlm/vlm2-bench_a_closer_look_at_how_well_vlms_implicitly_link_explicit_matching_visu.md)**

:   本文提出VLM2-Bench，一个专门评估视觉语言模型（VLM）跨图像/帧"视觉线索关联"能力的基准，涵盖通用线索、物体中心线索和人物中心线索3大类9个子任务共3000+测试样本，发现即使最先进的商业模型在该任务上也落后人类30%以上，揭示了VLM在基础视觉匹配能力上的重大差距。

**[Vlminferslow Evaluating The Efficiency Robustness Of](multimodal_vlm/vlminferslow_evaluating_the_efficiency_robustness_of.md)**

:   首次在黑盒设置下研究 VLM 的效率鲁棒性，提出 VLMInferSlow 方法，通过零阶优化搜索对抗性图像扰动，迫使 VLM 生成更长序列，将计算成本最高增加 128.47%，揭示了 VLM 在 MLaaS 部署场景下的效率安全隐患。

**[Vlsbench Unveiling Visual Leakage In Multimodal Safety](multimodal_vlm/vlsbench_unveiling_visual_leakage_in_multimodal_safety.md)**

:   揭示现有多模态安全基准中存在的视觉安全信息泄露（VSIL）问题——图像中的危险内容已在文本查询中暴露，导致模型仅凭文本即可拒绝，从而使安全评估不可靠；为此构建了无泄露的VLSBench基准（2.2k图文对），发现多模态对齐在无VSIL场景中显著优于纯文本对齐。

**[Vrest Tree Search Vlm Reasoning](multimodal_vlm/vrest_tree_search_vlm_reasoning.md)**

:   提出VReST，首次将蒙特卡洛树搜索（MCTS）应用于多模态CoT推理：每个节点是一个推理步骤，通过多模态自奖励机制（sub-question有用性+答案正确性+视觉-语言线索相关性）评估推理质量，无需训练即在MathVista上达到64.50%（超越CoT的54.60%和ToT的60.20%），并展示出多模态测试时缩放定律。

**[We-Math Does Your Large Multimodal Model Achieve Human-Like Mathematical Reasoni](multimodal_vlm/we-math_does_your_large_multimodal_model_achieve_human-like_mathematical_reasoni.md)**

:   本文提出We-Math基准，首次通过将复合数学问题按知识概念分解为子问题，引入IK/IG/CM/RM四维指标来层次化评估LMM的推理过程（而非仅看最终结果），揭示了LMM普遍存在知识不足（IK）问题，且GPT-4o是首个从IK阶段迈入知识泛化（IG）阶段的模型。

**[Weaving Context Across Images Improving Vision-Language Models Through Focus-Cen](multimodal_vlm/weaving_context_across_images_improving_vision-language_models_through_focus-cen.md)**

:   提出 Focus-Centric Visual Chain 多图推理范式，通过问题分解和逐步聚焦关键视觉信息实现跨图推理，并构建 VISC-150K 数据集，在七个多图基准上实现 2-3% 的一致性提升。

**[Wikimixqa A Multimodal Benchmark For Question Answering Over Tables And Charts](multimodal_vlm/wikimixqa_a_multimodal_benchmark_for_question_answering_over_tables_and_charts.md)**

:   提出 WikiMixQA 基准，包含 1,000 道需要跨表格和图表进行多模态推理的多选题，评估 12 个 VLLM 后发现闭源模型在提供精确上下文时准确率约 70%，但需从长文档检索时性能骤降，开源模型最高仅 27%，揭示了当前视觉语言模型在长上下文多模态文档理解上的严重不足。

---

## 📊 LLM评测 { #llm_evaluation }

**[A Conformal Risk Control Framework For Granular Word Assessment And Uncertainty ](llm_evaluation/a_conformal_risk_control_framework_for_granular_word_assessment_and_uncertainty_.md)**

:   提出基于保形风险控制（Conformal Risk Control）框架校准 CLIPScore 的方法——通过对 CLIP 视觉/文本编码器的注意力掩码采样生成 CLIPScore 分布，然后利用保形风险控制 (1) 检测图像描述中的干扰词（foil words），(2) 生成校准置信区间，在 FOIL-it/FOIL-nocaps/Rich-HF 基准上以简单方法达到与复杂专用方法相当的干扰词检测性能，同时提供形式化风险保证。

**[MisMatched: A Benchmark for Scientific Natural Language Inference](llm_evaluation/a_mismatched_benchmark_for_scientific_natural_language_inference.md)**

:   引入 MisMatched——首个覆盖非 CS 领域（心理学、工程、公共卫生）的科学 NLI 评估基准，2700 对人工标注句子对，最佳基线 Macro F1 仅 78.17%，且发现训练时加入隐式关系句子对可提升性能。

**[Abgen Evaluating Large Language Models In](llm_evaluation/abgen_evaluating_large_language_models_in.md)**

:   提出 AbGen——首个评估 LLM 设计消融实验能力的基准（1500 条专家标注数据来自 807 篇 NLP 论文），发现最强 LLM (DeepSeek-R1) 与人类专家差距 14.4%，且 LLM-as-Judge 评分与人类评估严重不一致。

**[Access Denied Inc: The First Benchmark Environment for Sensitivity Awareness](llm_evaluation/access_denied_inc_the_first_benchmark_environment_for_sensitivity_awareness.md)**

:   提出敏感性感知（Sensitivity Awareness, SA）概念——评估 LLM 是否能遵守基于角色的访问控制规则——并构建首个评估基准 Access Denied Inc：模拟企业数据库 + 多用户组权限 + 自动化问卷+半自动评分（99.9%自动），揭示模型在拒绝未授权请求和响应合法查询上的显著差异。

**[Ad-Hoc Concept Forming In The Game Codenames As A Means For Evaluating Large Lan](llm_evaluation/ad-hoc_concept_forming_in_the_game_codenames_as_a_means_for_evaluating_large_lan.md)**

:   以桌游 Codenames 作为 LLM 评测工具——LLM 分别扮演线索给出者（Spymaster）和猜测者（Field Operative），通过控制词频/歧义性/具体性/风险等级/对手速度等变量系统评估 LLM 的临时概念形成、语义关联、合作推理和语用能力，发现 o3-mini 和 Claude-3.5 领先但所有模型在高风险和抽象词条件下均显著退化。

**[Ad-Llm Benchmarking Large Language Models For Anomaly Detection](llm_evaluation/ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)**

:   首个系统评估 LLM 在 NLP 异常检测中角色的基准 AD-LLM——覆盖三个关键任务：(1) 零样本检测（LLM 预训练知识直接做 AD），(2) 数据增强（生成合成数据/类别描述提升 AD 模型），(3) 模型选择（LLM 推荐无监督 AD 模型）。多数据集实验发现 LLM 零样本 AD 表现出色，精心设计的增强有用，但模型选择的可解释性仍是挑战。

**[Androidlab Autonomous Agent](llm_evaluation/androidlab_autonomous_agent.md)**

:   提出AndroidLab——首个统一训练和评估Android Agent的系统性框架，包含9个App上的138个可复现任务，同时支持纯文本（XML模式）和多模态（SoM模式）模型，并构建Android Instruct数据集（94.3k步骤），将开源LLM的成功率从4.59%提升至21.50%。

**[Antileakbench Preventing Data Contamination By Automatically Constructing Benchm](llm_evaluation/antileakbench_preventing_data_contamination_by_automatically_constructing_benchm.md)**

:   提出 AntiLeakBench——自动化反泄露基准框架，通过识别 LLM 知识截止后更新的真实世界新知识自动构建 QA 测试样本（而非简单收集新发布数据），确保测试知识严格不在训练集中，全自动流程无需人工标注，实验证实截止后性能普遍下降验证了数据污染的普遍存在。

**[Atomic Calibration Of Llms In Long-Form Generations](llm_evaluation/atomic_calibration_of_llms_in_long-form_generations.md)**

:   系统研究长文本生成中的原子级校准（Atomic Calibration）——将长回复分解为原子主张（atomic claims），为每个主张分配置信度分数，发现回复级校准良好的模型在原子级校准很差，将置信度获取方法分为判别式（内部状态）和生成式（外部评估）两类并发现它们互补，提出两种融合策略达到 SOTA 校准效果。

**[Batayan A Filipino Nlp Benchmark For Evaluating Large Language Models](llm_evaluation/batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)**

:   提出 Batayan——首个全面的菲律宾语 LLM 评测基准，覆盖理解/推理/生成三大能力的 8 个任务（含 3 个全新菲律宾语任务），由母语者翻译和标注确保语言真实性，评测 50+ 开源和商用 LLM 后发现菲律宾语表现显著落后于英语，显式菲律宾语支持和模型规模的提升均能带来明显增益。

**[Belarusian Glue](llm_evaluation/belarusian_glue.md)**

:   为白俄罗斯语（Belarusian，东斯拉夫语族）构建了首个NLU benchmark——BelarusianGLUE，包含5个任务约15K条实例，系统评估了BERT系列和LLM的表现，发现简单任务（情感分析）接近人类水平但难任务（Winograd）仍有显著差距，且最优模型类型因任务而异。

**[Besstie A Benchmark For Sentiment And Sarcasm Classification For Varieties Of En](llm_evaluation/besstie_a_benchmark_for_sentiment_and_sarcasm_classification_for_varieties_of_en.md)**

:   构建 BESSTIE，首个针对英语变体（澳大利亚/印度/英国英语）的情感分析和讽刺检测标注基准，通过 9 个微调 LLM 评估发现模型在印度英语（外圈变体）上表现显著差于内圈变体，跨变体泛化能力也有限。

**[Beyond One-Size-Fits-All Tailored Benchmarks For Efficient Evaluation](llm_evaluation/beyond_one-size-fits-all_tailored_benchmarks_for_efficient_evaluation.md)**

:   提出 TailoredBench 方法，为每个待评估的目标模型**自适应构建定制化核心集**（Native-coreset），而非使用所有模型共享的静态子集，通过自适应源模型选择、可扩展 K-Medoids 聚类和校准估计策略，在仅需 20-40 个样本的推理预算下将准确率估计的 MAE 平均降低 **31.4%**。

**[Browsing Lost Unformed Recollections A Benchmark For Tip-Of-The-Tongue Search An](llm_evaluation/browsing_lost_unformed_recollections_a_benchmark_for_tip-of-the-tongue_search_an.md)**

:   > 提出 BLUR（Browsing Lost Unformed Recollections），一个包含 573 道真实"话到嘴边"(tip-of-the-tongue) 已知物品搜索与推理问题的基准数据集，人类准确率 98%，而最佳 AI 系统仅约 56%，揭示了当前 AI 在工具使用和多跳推理上的巨大差距。

**[Calibraeval Calibrating Prediction Distribution To Mitigate Selection Bias In Ll](llm_evaluation/calibraeval_calibrating_prediction_distribution_to_mitigate_selection_bias_in_ll.md)**

:   提出 CalibraEval，一种无标签的推理时去偏方法，通过将去偏问题形式化为优化任务，利用非参数保序算法（NOA）学习校准函数，将 LLM 评判器的观测概率分布映射到无偏分布，有效缓解 LLM-as-Judge 中的选择偏差。

**[Calibration Confidence Text Gen](llm_evaluation/calibration_confidence_text_gen.md)**

:   针对文本生成中多个有效输出导致传统置信度指标失效的问题，提出两种任务无关的置信度度量——"比率"（头部vs中部概率比）和"尾部稀薄度"（分布尾部薄厚），仅依赖模型输出概率即可改善 BART/Flan-T5 在摘要、翻译、问答任务上的置信度校准。

**[Can External Validation Tools Improve Annotation Quality For Llm-As-A-Judge](llm_evaluation/can_external_validation_tools_improve_annotation_quality_for_llm-as-a-judge.md)**

:   提出 Evaluation Agent，一个工具增强的 LLM-as-a-Judge 框架，通过集成网络搜索（事实核查）、代码执行和数学验证工具，在长文本事实验证上将与人类一致性从 63% 提升到 81%，在编程评估上从 31% 提升到 71%，且对无关领域几乎无退化。

**[Cfbench A Comprehensive Constraints-Following Benchmark For Llms](llm_evaluation/cfbench_a_comprehensive_constraints-following_benchmark_for_llms.md)**

:   提出 CFBench——一个包含 1000 条精标样本、覆盖 200+ 真实场景和 50+ NLP 任务的中文大规模约束遵循基准，系统性地定义了 10 大类 25+ 子类的约束分类体系，并设计结合约束满足率（CSR）、指令满足率（ISR）和需求优先级满足率（PSR）的多维评估框架，揭示当前顶级 LLM 在约束遵循方面仍存在显著提升空间。

**[Chatbench From Static Benchmarks To Human-Ai Evaluation](llm_evaluation/chatbench_from_static_benchmarks_to_human-ai_evaluation.md)**

:   通过用户实验将 MMLU 静态基准转换为用户-AI 对话，构建 ChatBench 数据集（396 道题、7336 段对话），发现 AI-alone 准确率无法预测 user-AI 准确率，并训练用户模拟器使相关性提升 22-26 个百分点，为可扩展的交互式评估奠基。

**[Codemenv Benchmarking Large Language Models On Code Migration](llm_evaluation/codemenv_benchmarking_large_language_models_on_code_migration.md)**

:   提出 CodeMEnv，首个系统评估 LLM 跨环境代码迁移能力的基准，包含 922 个样本、19 个 Python/Java 包、3 个层次化任务（定位不兼容函数→描述变更→迁移代码），9 个 LLM 的平均 Pass@1 仅 26.50%，GPT-4o 最高 43.84%，揭示 LLM 更熟悉新版本函数且存在版本推理逻辑不一致问题。

**[Com2 Causal Commonsense](llm_evaluation/com2_causal_commonsense.md)**

:   提出Com2基准，利用因果事件图和因果理论（干预/反事实）构建复杂常识推理任务，发现LLM在推理深度和广度上存在不足，后训练和慢思考可部分缓解。

**[Culemo Cultural Lenses On Emotion - Benchmarking Llms For Cross-Cultural Emotion](llm_evaluation/culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)**

:   提出 CuLEmo，首个评估文化感知情感预测的多语言基准数据集，涵盖 6 种语言/文化（阿姆哈拉语、阿拉伯语、英语、德语、印地语、西班牙语），通过 400 个文化相关场景评估 LLM 的跨文化情感理解能力，发现情感表达在不同文化间存在显著差异且 LLM 表现参差不齐。

**[Culturalbench A Robust Diverse And Challenging Cultural Benchmark By Human-Ai Cu](llm_evaluation/culturalbench_a_robust_diverse_and_challenging_cultural_benchmark_by_human-ai_cu.md)**

:   通过 Human-AI CulturalTeaming（人机协作红队测试）流水线构建 CulturalBench，包含 1,696 个人类撰写并经五人独立验证的文化知识问题，覆盖 45 个全球地区和 17 个主题。CulturalBench-Hard（True/False格式）对最强模型（OpenAI o1）也仅 61.5%，远低于人类的 92.4%，揭示了模型在多答案问题上的模式寻求倾向和跨区域文化知识的不均衡表现。

**[Ecomscriptbench](llm_evaluation/ecomscriptbench.md)**

:   提出电商脚本规划（EcomScript）任务及其首个大规模benchmark EcomScriptBench（605K脚本、2.4M产品），通过购买意图（purchase intention）桥接用户行动步骤与产品检索的语义鸿沟，实验发现当前LLM在涉及产品的子任务上表现显著不足，注入意图知识可提升性能。

**[Educationq Evaluating Llms Teaching Capabilities Through Multi-Agent Dialogue Fr](llm_evaluation/educationq_evaluating_llms_teaching_capabilities_through_multi-agent_dialogue_fr.md)**

:   提出 EducationQ 多智能体对话框架，通过模拟真实课堂中教师-学生的形成性评估交互来评估 LLM 的教学能力，发现教学效果与模型规模或通用推理能力不呈线性关系，Llama 3.1 70B 在教学中表现最优。

**[Elaboration Competitive Programming](llm_evaluation/elaboration_competitive_programming.md)**

:   提出ELABORATION——首个全面评估人类-LLM协作竞赛编程的基准，包含覆盖编程全流程（理解→规划→编码→调试）的人类反馈分类体系和8320题精标注数据集，实验表明LLM在困难题上仅3.4% Pass@1，但人类反馈（特别是在编码阶段）可平均提升9.3%。

**[Evowiki Evaluating Llms On Evolving Knowledge](llm_evaluation/evowiki_evaluating_llms_on_evolving_knowledge.md)**

:   提出 EvoWiki，一个可自动更新的动态评估基准，将知识分为稳定（stable）、演化（evolved）和未知（uncharted）三个层级，用于评估 LLM 在知识持续演化场景下的利用能力，并揭示 RAG 与持续学习（CL）结合具有协同效应。

**[Exposing Numeracy Gaps A Benchmark To Evaluate Fundamental Numerical Abilities I](llm_evaluation/exposing_numeracy_gaps_a_benchmark_to_evaluate_fundamental_numerical_abilities_i.md)**

:   提出 NumericBench 综合基准，通过 6 类数据集评估 LLM 的 6 种基本数值能力（数字识别、算术运算、上下文检索、比较、汇总、逻辑推理），发现包括 GPT-4o、DeepSeek-V3 在内的 SOTA 模型在简单数值任务上仍表现极差，并深入分析了 5 种根因。

**[Financereasoning Benchmarking Financial Numerical Reasoning More](llm_evaluation/financereasoning_benchmarking_financial_numerical_reasoning_more.md)**

:   提出 FinanceReasoning benchmark，通过重标注公开数据集、构建 3,133 个 Python 金融函数库和新增 908 道专家标注难题，在可信度、全面性和挑战性三个维度提升金融数值推理评估能力。

**[From Tools To Teammates Evaluating Llms In Multi-Session Coding Interactions](llm_evaluation/from_tools_to_teammates_evaluating_llms_in_multi-session_coding_interactions.md)**

:   提出 MemoryCode 合成多会话数据集评估 LLM 在长期交互中追踪和执行编码指令的能力，发现即使 GPT-4o 在提供完整对话历史时准确率也下降 67%，揭示了当前 LLM 在前瞻性记忆和信息整合上的根本局限。

**[Grace A Granular Benchmark For Evaluating Model Calibration Against Human Calibr](llm_evaluation/grace_a_granular_benchmark_for_evaluating_model_calibration_against_human_calibr.md)**

:   提出GRACE基准，通过渐进式增量问答和真人vs模型竞赛收集1749个数据点，以人类校准表现为参照评估LLM校准能力，并引入CalScore指标发现：虽然人类准确率可能低于模型，但人类在校准方面普遍优于SOTA模型——模型在不确定时过度自信、在正确时又信心不足。

**[Guessarena Guess Who I Am A](llm_evaluation/guessarena_guess_who_i_am_a.md)**

:   提出 GuessArena，一种基于"猜猜我是谁"博弈游戏的自适应 LLM 评估框架，通过领域知识建模和多轮交互推理，在五个垂直行业中有效区分模型的领域知识和推理能力。

**[Hallulens Llm Hallucination Benchmark](llm_evaluation/hallulens_llm_hallucination_benchmark.md)**

:   提出了 HalluLens 幻觉基准，明确区分幻觉与事实性，建立了外在幻觉（与训练数据不一致）和内在幻觉（与输入上下文不一致）的清晰分类体系，引入三个动态可重生成的外在幻觉评估任务，并全面分析了现有基准的局限性。

**[Hellaswag-Pro A Large-Scale Bilingual Benchmark For Evaluating The Robustness Of](llm_evaluation/hellaswag-pro_a_large-scale_bilingual_benchmark_for_evaluating_the_robustness_of.md)**

:   构建首个大规模双语（中英）LLM 常识推理鲁棒性评估基准 HellaSwag-Pro，包含 7 种问题变体共 11,200 道题，系统评估 41 个 LLM 发现所有模型在常识推理上远未达到鲁棒。

**[Help Write Story Feedback](llm_evaluation/help_write_story_feedback.md)**

:   探索 LLM 能否为创意写作者提供有意义的写作反馈——构建包含 1300 个故意引入写作问题的故事测试集，评估常用 LLM 的写作反馈生成能力，发现模型虽能提供具体且多数准确的反馈，但常错过最重要的写作问题且不会恰当地在批评和鼓励之间切换。

**[Hpss Heuristic Prompting Strategy Search For Llm Evaluators](llm_evaluation/hpss_heuristic_prompting_strategy_search_for_llm_evaluators.md)**

:   整合 8 个影响 LLM 评估提示效果的关键因子（评分尺度、ICL 示例、评估标准、参考答案、CoT、AutoCoT、度量指标、组件顺序），提出基于遗传算法的启发式提示策略搜索方法 HPSS，在 12,960 种组合空间中高效找到最优提示策略，仅用基线 5% 的生成成本即超越 G-Eval 和 CloserLook。

**[Justrank Llm Judge System Ranking](llm_evaluation/justrank_llm_judge_system_ranking.md)**

:   首次大规模研究LLM判官在系统排名任务中的表现，提出JuStRank基准，揭示实例级判断能力与系统级排名能力之间的差距，并发现判官的"果断性"和"偏见"两个新兴特征。

**[Kitab-Bench A Comprehensive Multi-Domain Benchmark For Arabic Ocr And Document U](llm_evaluation/kitab-bench_a_comprehensive_multi-domain_benchmark_for_arabic_ocr_and_document_u.md)**

:   KITAB-Bench 是一个涵盖 9 大领域 36 个子领域共 8,809 个样本的综合性阿拉伯语 OCR 基准，评估结果显示现代视觉语言模型（如 GPT-4o、Gemini）在字符错误率上平均超过传统 OCR 方法 60%，但在 PDF-to-Markdown 转换中最优模型仅达到 65% 准确率，凸显了阿拉伯语文档理解的巨大挑战。

**[La Leaderboard Spanish](llm_evaluation/la_leaderboard_spanish.md)**

:   构建首个面向西班牙和拉丁美洲语言的开源LLM排行榜，整合66个数据集覆盖西班牙语、加泰罗尼亚语、巴斯克语、加利西亚语，评估50个模型并分析训练策略、算力与性能的关系。

**[Language Complexity Measurement as a Noisy Zero-Shot Proxy for Evaluating LLM Performance](llm_evaluation/language_complexity_measurement_as_a_noisy_zero-shot_proxy_for_evaluating_llm_pe.md)**

:   探索语言复杂度指标作为 LLM 性能的零样本代理评估，发现文本复杂度与 LLM 表现负相关但噪声大，仅可作为粗略参考。

**[Language Model Probabilities Are Not Calibrated In Numeric Contexts](llm_evaluation/language_model_probabilities_are_not_calibrated_in_numeric_contexts.md)**

:   系统研究了语言模型在数值上下文中的概率校准问题，发现即使在简单场景（如从袋中取弹珠）下，包括 GPT-4o 在内的所有测试模型均严重校准不良，存在基于词序、词频和词标识的系统性偏差（如某些模型总选第一个选项，其他模型总选第二个），指令微调加剧了模式崩塌。

**[Mars Benchmarking The Metaphysical Reasoning Abilities Of Language Models With A](llm_evaluation/mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)**

:   本文提出了 **Metaphysical Reasoning（形而上推理）** 的形式化定义，将分布变化下的推理分解为三步判别过程，并构建了首个大规模评估基准 Mars（355K 标注数据），实验表明 20+ 语言模型在该任务上表现均不理想，揭示了 LLM 在理解事件组成要素变化及其因果效应方面的显著短板。

**[Mcbe A Multi-Task Chinese Bias Evaluation Benchmark For Large Language Models](llm_evaluation/mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)**

:   提出首个多任务中文偏见评估基准 McBE，包含 4,077 条偏见评估实例（BEI），覆盖 12 种偏见类别和 82 个子类别，通过 5 种评估任务（偏好计算/子类别分类/场景选择/偏见分析/偏见评分）多角度量化 LLM 中的中文偏见，并揭示"参数越大偏见越强"的传统结论可能源于单任务评估的局限性。

**[Mdbench A Synthetic Multi-Document Reasoning Benchmark Generated With Knowledge ](llm_evaluation/mdbench_a_synthetic_multi-document_reasoning_benchmark_generated_with_knowledge_.md)**

:   提出 MDBench，一个通过「结构化知识→LLM 辅助增强→自然文本生成」管线合成的多文档推理 QA 基准，可控地注入跨文档依赖，对前沿 LLM 构成显著挑战（最佳模型 EM 仅~60%）。

**[Mis-Prompt Benchmarking Large Language Models For Proactive Error Handling](llm_evaluation/mis-prompt_benchmarking_large_language_models_for_proactive_error_handling.md)**

:   提出 Mis-prompt 基准，包含 4 项评估任务、14 类错误分类体系和 14,969 条数据集，系统研究 LLM 在无显式错误处理指令时的**主动**纠错能力，发现当前 LLM 主动纠错能力严重不足，SFT 可显著提升。

**[Mmlu-Cf A Contamination-Free Multi-Task Language Understanding Benchmark](llm_evaluation/mmlu-cf_a_contamination-free_multi-task_language_understanding_benchmark.md)**

:   提出 MMLU-CF，一个包含 20,000 道题的无数据污染多任务语言理解基准，通过从更广泛的来源收集数据并应用三条去污染规则（改写题目、打乱选项、随机替换选项）来避免无意和恶意的数据泄露，最强模型 GPT-4o 在该基准上仅获得 73.4%（MMLU 上为 88.0%）。

**[Movie101V2 Improved Movie Narration Benchmark](llm_evaluation/movie101v2_improved_movie_narration_benchmark.md)**

:   提出 Movie101v2，一个包含 203 部电影、46K 双语（中英文）视频-叙事对的大规模电影叙事基准，将自动电影叙事任务分解为三个渐进式目标（视觉事实描述 L1 → 情节叙述 L2 → 可部署 AD L3），并基于 LLM 提出新的评估框架，全面基线测试了包括 GPT-4V 在内的多种视觉语言模型。

**[Navigating Rifts In Human-Llm Grounding Study And Benchmark](llm_evaluation/navigating_rifts_in_human-llm_grounding_study_and_benchmark.md)**

:   系统研究人与 LLM 对话中的 grounding（建立共识）失败问题，发现 LLM 主动澄清的频率仅为人类的 1/3、主动追问的频率仅为 1/16，提出 Rifts 基准（约 1.8K 任务）评测 LLM 的 grounding 能力，并通过 grounding forecaster 实现初步干预。

**[Onebench To Test Them All Sample-Level Benchmarking Over Open-Ended Capabilities](llm_evaluation/onebench_to_test_them_all_sample-level_benchmarking_over_open-ended_capabilities.md)**

:   ONEBench提出了一种新的基准评测范式：将多个评测数据集的样本合并为统一数据池，通过Plackett-Luce排名聚合算法在样本级别进行模型比较，支持异构指标聚合、不完整数据处理和个性化能力探测。

**[Pap2Pat Benchmarking Outline-Guided Long-Text Patent Generation With Patent-Pape](llm_evaluation/pap2pat_benchmarking_outline-guided_long-text_patent_generation_with_patent-pape.md)**

:   构建了包含 1.8k 专利-论文配对的 Pap2Pat 基准，提出基于大纲的分块专利描述生成方法 COPGen，并设计了基于 NLI 的事实性/覆盖率/风格评估指标，系统评测了当前 LLM 在超长专利文档生成上的能力与不足。

**[Papersplease A Benchmark For Evaluating Motivational Values Of Large Language Mo](llm_evaluation/papersplease_a_benchmark_for_evaluating_motivational_values_of_large_language_mo.md)**

:   基于 Alderfer ERG 需求理论构建 3700 个道德困境场景（移民检查官角色扮演），评估 6 个 LLM 的动机价值偏好，发现 Claude 拒绝所有场景、GPT-4o-mini 对生存需求 99% 满足但对关系需求仅 47%，且模型对穆斯林/边缘化群体存在显著的隐性社会偏见。

**[Patch Psychometrics-Assisted Benchmarking Of Large Language Models Against Human](llm_evaluation/patch_psychometrics-assisted_benchmarking_of_large_language_models_against_human.md)**

:   提出 PATCH 框架，将心理测量学中的项目反应理论（IRT 3PL/2PL 模型）引入 LLM 基准测试，在 TIMSS 2011 八年级数学测试（88 道题、56 个国家/地区）上对比 GPT-4V、Gemini-Pro-Vision、Qwen-VL 与人类群体的能力值，发现 IRT 能力估计与简单准确率排名显著不同，GPT-4V 与韩国/新加坡/中国台北学生处于同一排名区间；同时发布 4 个高质量数据集（TIMSS 2011 & 2008 数学/科学/物理）。

**[Physreason A Comprehensive Benchmark Towards Physics-Based Reasoning](llm_evaluation/physreason_a_comprehensive_benchmark_towards_physics-based_reasoning.md)**

:   提出 PhysReason 基准，包含 1200 道物理题（平均 8.1 步解题），设计了答案级和步骤级两层自动评估框架 PSAS，揭示顶尖模型（Deepseek-R1、o3-mini）在物理推理上准确率不足 60%，并识别出四大推理瓶颈。

**[Readoc A Unified Benchmark For Realistic Document Structured Extraction](llm_evaluation/readoc_a_unified_benchmark_for_realistic_document_structured_extraction.md)**

:   READoc 提出了首个将文档结构化提取（DSE）定义为端到端 PDF 到 Markdown 转换的统一基准，包含 3,576 篇来自 arXiv/GitHub/Zenodo 的真实文档和三模块评估套件（标准化+分割+评分），首次揭示了当前 DSE 系统与真实场景需求之间的差距。

**[Realhitbench A Comprehensive Realistic Hierarchical Table Benchmark For Evaluati](llm_evaluation/realhitbench_a_comprehensive_realistic_hierarchical_table_benchmark_for_evaluati.md)**

:   提出 RealHiTBench——首个全面评估 LLM 对复杂层次化表格理解能力的基准，包含 708 张来自 13 个平台、24 个领域的真实复杂表格和 3,752 道题目，定义了 5 种复杂结构类型和 5 大任务类型，并提出基于树结构的 TreeThinker 推理管线显著提升模型对层次化表头的理解能力。

**[Retrieval Models Arent Tool-Savvy Benchmarking Tool Retrieval For Large Language](llm_evaluation/retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)**

:   提出 ToolRet——首个大规模工具检索基准（7.6k 任务 + 43k 工具语料库），系统评估了现有 IR 模型在工具检索场景下的表现，发现即使强力检索器也表现不佳，并贡献了超过 200k 训练实例显著提升检索质量。

**[Revisiting 3D Llm Benchmarks Are We Really Testing 3D Capabilities](llm_evaluation/revisiting_3d_llm_benchmarks_are_we_really_testing_3d_capabilities.md)**

:   揭示了 3D LLM 评测中的"2D-Cheating"问题——将点云渲染为图像后，2D VLM 在部分基准上超越 3D SOTA，说明这些基准未能有效评估真正的 3D 理解能力，并据此提出了有效 3D 评测的设计原则。

**[Right Answer Wrong Score Uncovering The Inconsistencies Of Llm Evaluation In Mul](llm_evaluation/right_answer_wrong_score_uncovering_the_inconsistencies_of_llm_evaluation_in_mul.md)**

:   系统揭示了LLM在多选问答(MCQA)评估中的不一致性——不同评估策略(RegEx/Logprobs/xFinder)和提示设置(约束/自由生成)组合会导致模型性能报告产生显著差异，且即使是SOTA的LLM-based答案提取器也无法可靠识别推理矛盾，呼吁建立标准化评估协议。

**[Rulearena Rule Guided Reasoning](llm_evaluation/rulearena_rule_guided_reasoning.md)**

:   提出 RuleArena——一个基于航空行李费、NBA交易规则、税务法规三个真实场景的benchmark，用于评估LLM遵循复杂自然语言规则进行推理的能力；实验发现即使最强模型（o1-preview）在最难任务上准确率也不足50%，暴露了LLM在规则召回、规则区分和数学计算三方面的系统性缺陷。

**[Sanskriti A Comprehensive Benchmark For Evaluating Language Models Knowledge Of ](llm_evaluation/sanskriti_a_comprehensive_benchmark_for_evaluating_language_models_knowledge_of_.md)**

:   提出 SANSKRITI——一个包含 21,853 个问答对、覆盖印度全部 28 个邦和 8 个联邦属地、涵盖 16 类文化属性的大规模基准数据集，用于评估语言模型对印度文化多样性的理解程度。

**[Seedbench A Multi-Task Benchmark For Evaluating Large Language Models In Seed Sc](llm_evaluation/seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)**

:   提出 SeedBench——首个面向种子科学（种子育种）的多任务 LLM 评测基准，包含 2,264 道专家验证题目，覆盖基因信息检索、基因功能调控和品种选育三大育种流程，对 26 个 LLM 进行系统评估，揭示了当前 LLM 与真实育种需求之间的显著差距。

**[Somethings Fishy In The Data Lake A Critical Re-Evaluation Of Table Union Search](llm_evaluation/somethings_fishy_in_the_data_lake_a_critical_re-evaluation_of_table_union_search.md)**

:   系统性分析了主流表联合搜索 (Table Union Search, TUS) 基准测试的三大结构性缺陷——过度重叠、语义简单、真值噪声，揭示简单的词袋 (BoW) 和预训练嵌入基线就能在这些基准上达到或超越复杂 SOTA 方法的效果，调研结论指出当前基准无法有效评估语义理解能力。

**[Structext Eval](llm_evaluation/structext_eval.md)**

:   提出StrucText-Eval——一个覆盖8种结构化语言（JSON/YAML/XML/Markdown/LaTeX/Org/CSV/Tree）和29个任务的自动生成评测基准，共5,800个样本，通过可控的嵌套深度和结构宽度调节难度。实验揭示开源LLM在标准集最高仅74.9%准确率，困难集降至45.8%，而人类在困难集达92.6%，暴露了LLM在复杂结构推理上的严重不足。

**[Structflowbench A Structured Flow Benchmark For Multi-Turn Instruction Following](llm_evaluation/structflowbench_a_structured_flow_benchmark_for_multi-turn_instruction_following.md)**

:   提出 StructFlowBench，一个融入结构流建模的多轮指令遵循基准测试，定义了六种基本的轮间关系（跟进、精炼、回忆、总结、扩展、不相关），建立了双层约束评估体系（轮内约束 + 轮间结构约束），系统评估了 13 个主流 LLM 在多轮对话结构理解上的能力。

**[Towards Dynamic Theory Of Mind Evaluating Llm Adaptation To Temporal Evolution O](llm_evaluation/towards_dynamic_theory_of_mind_evaluating_llm_adaptation_to_temporal_evolution_o.md)**

:   提出 DynToM 基准，通过 1,100 个社会情境中 5,500 个时序关联场景和 78,100 道题目，评估 LLM 追踪人类心理状态时序演化的能力，揭示模型平均落后人类 44.7%。

**[Towards Objective Fine-Tuning How Llms Prior Knowledge Causes Potential Poor Cal](llm_evaluation/towards_objective_fine-tuning_how_llms_prior_knowledge_causes_potential_poor_cal.md)**

:   揭示LLM的先验知识在微调过程中会导致校准退化（已知数据引发过度自信，未知数据反而有利于校准），提出CogCalib认知感知校准框架，在训练中根据知识偏差动态应用不同学习策略，在保持任务性能的同时平均降低57%的ECE。

**[Triptailor A Real-World Benchmark For Personalized Travel Planning](llm_evaluation/triptailor_a_real-world_benchmark_for_personalized_travel_planning.md)**

:   提出 TripTailor，一个基于真实数据的大规模旅行规划 benchmark，包含 40 个城市的 50 万+ POI 和近 4000 条真实行程，并引入可行性、合理性和个性化三维评估框架，发现最先进 LLM 生成的行程不到 10% 能达到人类水平。

**[Tumlu A Unified And Native Language Understanding Benchmark For Turkic Languages](llm_evaluation/tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)**

:   提出 TUMLU 和 TUMLU-mini，首个面向突厥语系（9 种语言）的原生语言理解基准，包含 38139 道中高中学科的多选题，覆盖拉丁、西里尔和阿拉伯三种文字系统，为低资源语言的 LLM 评估提供了高质量的原生基准。

**[Vital Pluralistic Alignment Healthcare](llm_evaluation/vital_pluralistic_alignment_healthcare.md)**

:   本文构建了首个面向医疗健康领域的多元化对齐（pluralistic alignment）基准数据集 VITAL，包含 13.1K 价值观情境和 5.4K 多选题，并通过对 8 个 LLM 的广泛评估表明，现有多元化对齐技术（尤其是 ModPlural）在医疗场景下表现不佳，简单的 prompting 反而效果更好。

**[Voxeval Benchmarking The Knowledge Understanding Capabilities Of End-To-End Spok](llm_evaluation/voxeval_benchmarking_the_knowledge_understanding_capabilities_of_end-to-end_spok.md)**

:   提出 VoxEval，首个支持端到端纯语音输入-输出评估的 SpeechQA 基准，涵盖 56 个学科、26 种输入音频条件，系统揭示了当前端到端语音大模型在知识理解和数学推理方面的严重不足。

**[Wicked A Simple Method To Make Multiple Choice Benchmarks More Challenging](llm_evaluation/wicked_a_simple_method_to_make_multiple_choice_benchmarks_more_challenging.md)**

:   提出 WiCkeD 方法，通过随机将多选题的一个选项替换为"以上都不对"来增加现有基准难度，使18个 LLM 的平均性能下降12.1个百分点，且链式思维推理也无法弥补这一下降。

**[Wximpactbench A Disruptive Weather Impact Understanding Benchmark For Evaluating](llm_evaluation/wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)**

:   提出 WXImpactBench，首个评估 LLM 理解极端天气社会影响的基准，包含经过 4 阶段管线处理的高质量历史报纸数据集和两个评估任务（多标签分类和排序问答）。

**[Yescieval Llm Judge Science](llm_evaluation/yescieval_llm_judge_science.md)**

:   提出YESciEval框架，结合九维细粒度评估准则和SFT+RL对齐策略来缓解LLM评估者的乐观偏差(optimism bias)，在科学问答场景下构建鲁棒的开源LLM-as-a-Judge系统，无需人类标注和闭源模型。

---

## 🌐 多语言/翻译 { #multilingual_mt }

**[A Case Study Of Cross-Lingual Zero-Shot Generalization For Classical Languages I](multilingual_mt/a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)**

:   系统评估 LLM（GPT-4o/Llama-3.1）在三种古典语言（梵语、拉丁语、古希腊语）上的零样本跨语言泛化能力——涵盖 NER、机器翻译、问答三个 NLU 任务，发现大模型在域外数据上可比肩甚至超越微调基线，模型规模是决定性因素，并贡献了一个 1501 对的梵语事实问答数据集。

**[Alleviating Distribution Shift In Synthetic Data For Machine Translation Quality](multilingual_mt/alleviating_distribution_shift_in_synthetic_data_for_machine_translation_quality.md)**

:   提出 DCSQE 框架，通过约束波束搜索生成更真实的合成翻译、利用独立的标注模型纠正标签偏差、以及 SPCE 算法将 token 级标签聚合为短语级标签，有效缓解合成 QE 数据的分布偏移问题，在有监督和无监督设置下均超越 CometKiwi 等 SOTA 基线。

**[An Expanded Massive Multilingual Dataset For High-Performance Language Technolog](multilingual_mt/an_expanded_massive_multilingual_dataset_for_high-performance_language_technolog.md)**

:   本文介绍 HPLT v2，一个从 4.5 PB 的 Internet Archive 和 Common Crawl 数据中提取的大规模多语言数据集，包含覆盖 193 种语言的 8 万亿 token 单语数据和覆盖 51 种语言的 3.8 亿句对平行数据，并通过改进的数据处理管线显著提升了数据质量。

**[Are Rules Meant To Be Broken Understanding Multilingual Moral Reasoning As A Com](multilingual_mt/are_rules_meant_to_be_broken_understanding_multilingual_moral_reasoning_as_a_com.md)**

:   提出UniMoral——一个跨6种语言的统一道德推理数据集，将道德推理建模为包含行为预测、道德类型分类、因素归因和后果生成的计算流水线，对3个LLM的基准测试揭示隐式道德语境能增强模型道德推理能力但仍需专门化方法。

**[Askqe Question Answering As Automatic Evaluation For Machine Translation](multilingual_mt/askqe_question_answering_as_automatic_evaluation_for_machine_translation.md)**

:   提出 AskQE——基于问答的机器翻译质量估计框架，通过对源文本生成问题、分别在源文本和回译输出上回答、对比答案差异来检测翻译错误，帮助不懂目标语言的用户判断翻译是否可接受，在 BioMQM 数据集上 Kendall's τ 相关和决策准确率均优于现有 QE 指标。

**[Assessing Agentic Large Language Models In Multilingual National Bias](multilingual_mt/assessing_agentic_large_language_models_in_multilingual_national_bias.md)**

:   首次研究 LLM 作为推理型 Agent 在多语言场景下的国籍偏见——在大学申请/旅行/搬迁三个决策场景中，让 GPT-3.5/GPT-4/Sonnet 对同一实体（大学/城市）用不同语言打分，发现普遍存在"本地语言偏向"（用中文问清华得 10 分，用英文问只得 7 分），GPT-4 在英语上偏见减少但非英语上偏见显著，CoT 不一定缓解反而可能放大偏差。

**[Beyond N-Grams Rethinking Evaluation Metrics And Strategies For Multilingual Abs](multilingual_mt/beyond_n-grams_rethinking_evaluation_metrics_and_strategies_for_multilingual_abs.md)**

:   系统评估了 n-gram 和神经指标在 8 种语言（4 种类型学家族）上与人工判断的相关性，发现 n-gram 指标在融合语言中可靠性差，而专门训练的神经指标 COMET 在所有语言上一致优于其他指标；还发现分词策略可以显著改善融合语言的评估效果。

**[Blessing Of Multilinguality A Systematic Analysis Of Multilingual In-Context Lea](multilingual_mt/blessing_of_multilinguality_a_systematic_analysis_of_multilingual_in-context_lea.md)**

:   系统分析多语言 ICL 策略，发现在 prompt 中混合多种高资源语言（HRL）的 demonstrations 一致性优于纯英文 demonstrations，尤其在低资源语言（LRL）上提升显著（Llama3.1 上 LRL 平均准确率提升 8.9~12.6%），甚至仅在 prompt 中加入不相关的非英语句子也能带来可测量的增益，揭示了"多语言暴露本身即有效"的现象。

**[Cchall A Novel Benchmark For Joint Cross-Lingual And Cross-Modal Hallucinations ](multilingual_mt/cchall_a_novel_benchmark_for_joint_cross-lingual_and_cross-modal_hallucinations_.md)**

:   提出首个**联合跨语言与跨模态**幻觉检测基准 CCHall，覆盖 9 种语言和 4 类多模态数据集，系统评估 6 款主流 MLLM 在联合场景下的幻觉表现，揭示当前模型在该联合场景中 F1 比单独跨模态低 10.9、比单独跨语言低 3.4，且提出多语提示和外部工具辅助两条缓解路径。

**[Clix Cross-Lingual Explanations Of Idiomatic Expressions](multilingual_mt/clix_cross-lingual_explanations_of_idiomatic_expressions.md)**

:   提出跨语言习语解释任务 CLIX，构建了包含英语习语及其西班牙语/德语解释的数据集，系统评估了 seq2seq 模型和 LLM 在该任务上的表现，发现 GPT-3.5 Turbo 的 pipeline 策略（先英文解释再翻译）配合 few-shot 效果最佳，人工评估流畅度和准确度高达 4.7+/5。

**[Cosmmic Commentsensitive Multimodal Multilingual Indian Corpus](multilingual_mt/cosmmic_commentsensitive_multimodal_multilingual_indian_corpus.md)**

:   构建首个面向印度语言的评论感知多模态多语言数据集 COSMMIC（9 种语言、4,959 篇文章-图像对、24,484 条读者评论），提出评论过滤（IndicBERT）和图像分类（CLIP）增强方案，用 GPT-4 和 LLama3 建立摘要和标题生成的基准。

**[Cross-Lingual Auto Evaluation For Assessing Multilingual Llms](multilingual_mt/cross-lingual_auto_evaluation_for_assessing_multilingual_llms.md)**

:   提出 CIA (Cross Lingual Auto Evaluation) Suite，一个跨语言 LLM 评估框架，包含评估模型 Hercule 和人工标注测试集 Recon，通过利用英语参考答案对非英语语言的 LLM 响应进行评分，8B 模型在多语言评估上超越了 GPT-4o 等闭源大模型。

**[Cross-Lingual Optimization For Language Transfer In Large Language Models](multilingual_mt/cross-lingual_optimization_for_language_transfer_in_large_language_models.md)**

:   提出 Cross-Lingual Optimization (CLO)，通过修改 DPO 损失函数实现跨语言偏好优化——给目标语言输入时偏好目标语言回复、给英语输入时偏好英语回复——在 5 个模型 × 6 种语言上一致超越 SFT，低资源语言中仅 3,200 样本的 CLO 即超越 6,400 样本的 SFT。

**[Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning](multilingual_mt/cross-lingual_representation_alignment_through_contrastive_image-caption_tuning.md)**

:   探索了一种无需平行语料的跨语言表示对齐方法——通过多语言图像-文本描述的对比学习（类 CLIP），让不同语言的文本表示在共享视觉空间中隐式对齐，并证明即使是编码器预训练中未见过的语言（如 Quechua）也能通过这种方式被纳入对齐体系。

**[Cross-Lingual Transfer Of Cultural Knowledge An Asymmetric Phenomenon](multilingual_mt/cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)**

:   通过构建可解释的实验框架，研究 LLM 语言适应过程中文化知识的跨语言迁移现象，发现高资源语言（中文、韩语）与英语之间存在双向迁移，而低资源语言（藏语、蒙古语）则呈现不对称迁移——知识主要从低资源语言流向英语，反向流动有限，并提出频率假说加以解释。

**[Cross-Lingual Transfer Of Debiasing And Detoxification In Multilingual Llms An E](multilingual_mt/cross-lingual_transfer_of_debiasing_and_detoxification_in_multilingual_llms_an_e.md)**

:   在 7 个 LLM 和 20 种语言上系统研究了英语去偏见/去毒化微调的跨语言迁移效果，发现 SFT 有效去偏见、DPO 有效去毒化，但迁移到非英语语言时普遍伴随语言生成能力下降（语言一致性、流畅度、多样性均受损），迁移效果可由预训练数据中目标语言的数据量预测。

**[Cross Lingual Neurons Compression](multilingual_mt/cross_lingual_neurons_compression.md)**

:   本文通过追踪多语言语言模型预训练过程中的检查点，发现模型从语言特定表示逐渐压缩为跨语言共享表示：中间层的语言识别能力下降、语义概念的"专家神经元"跨语言对齐，操控从西班牙语数据提取的概念神经元后模型反而生成语义相关的英语文本。

**[Crosslingual Pitfalls](multilingual_mt/crosslingual_pitfalls.md)**

:   提出基于束搜索和 LLM 仿真的自动化方法来高效发现多语言 LLM 的跨语言弱点，构建了覆盖 16 种语言的 6000+ 双语问答对数据集，揭示即使 GPT-4o 也存在超过 30% 的跨语言性能下降。

**[Cruxeval-X A Benchmark For Multilingual Code Reasoning Understanding And Executi](multilingual_mt/cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)**

:   提出 CruxEval-X，一个覆盖 19 种编程语言的多语言代码推理基准，通过全自动的测试引导翻译流水线从 Python 版 CruxEval 扩展而来，包含 12,660 个题目和 19K 测试用例，对 24 个 LLM 的评估揭示了编程语言间的相关性以及单语言训练模型的跨语言泛化能力。

**[Dictionaries To The Rescue Cross-Lingual Vocabulary Transfer For Low-Resource La](multilingual_mt/dictionaries_to_the_rescue_cross-lingual_vocabulary_transfer_for_low-resource_la.md)**

:   本文提出一种基于双语词典的跨语言词汇迁移方法，利用BPE分词器"删除子词后回退到更短子词"的特性，通过迭代删除-重分词-对齐的过程最大化目标语言子词的映射覆盖率，在低资源语言上显著优于依赖单语语料或平行语料的现有方法。

**[Disentangle Language Culture](multilingual_mt/disentangle_language_culture.md)**

:   提出 Dual Evaluation Framework，将多语言 LLM 评估沿"语言媒介"和"文化语境"两个维度解耦，发现"文化-语言协同"(Cultural-Linguistic Synergy) 现象——模型在文化语境与提问语言对齐时表现更好，并通过 FFN 神经元激活分析从可解释性角度给出解释。

**[Edit Once, Update Everywhere: Cross-Lingual Knowledge Synchronization](multilingual_mt/edit_once_update_everywhere_a_simple_framework_for_cross-lingual_knowledge_synch.md)**

:   提出 X-KDE 框架通过指令微调+偏好优化实现知识的跨语言同步编辑——在一种语言中编辑知识后自动在其他语言同步生效。

**[Execute A Multilingual Benchmark For Llm Token Understanding](multilingual_mt/execute_a_multilingual_benchmark_for_llm_token_understanding.md)**

:   扩展字符理解基准 CUTE 到 8 种语言和多种文字系统，提出 EXECUTE 框架，发现 LLM 在不同语言的字符/词/子字符级别表现差异巨大，且意外发现 LLM 对越不熟悉的语言反而在 token 理解任务上表现越好。

**[Exploring In-Context Example Generation For Machine Translation](multilingual_mt/exploring_in-context_example_generation_for_machine_translation.md)**

:   提出DAT(Demonstration Augmentation for Translation)——在**无需任何外部资源**的情况下，让LLM自动生成与用户查询相关且多样的源-目标句对作为in-context示例，在5个低资源语言翻译任务上超越zero-shot和固定示例的few-shot基线。

**[Exploring In-Image Machine Translation With Real-World Background](multilingual_mt/exploring_in-image_machine_translation_with_real-world_background.md)**

:   提出 DebackX 模型，通过将图像分离为背景和文字图像分别处理，首次解决了真实复杂背景下的图像内机器翻译 (IIMT) 任务，在翻译质量和视觉效果上均优于现有方法。

**[Flare Crosslingual Lora](multilingual_mt/flare_crosslingual_lora.md)**

:   FLARE 在 LoRA 适配器的低秩瓶颈中通过轻量线性/非线性变换融合源语言（英语）和目标语言的逐层表示，无需额外参数即可实现参数高效的跨语言迁移，在 Llama 3.1 上 QA 精确匹配提升 4.9%。

**[Grammamt Improving Machine Translation With Grammar-Informed In-Context Learning](multilingual_mt/grammamt_improving_machine_translation_with_grammar-informed_in-context_learning.md)**

:   提出 GrammaMT，利用语素间注释文本 (Interlinear Glossed Text, IGT) 的语法信息来增强 LLM 的 few-shot 机器翻译，在濒危语言上平均提升 12+ BLEU，在中高资源语言上也有一致改进。

**[Group Then Scale Dynamic Mixture-Of-Experts Multilingual Language Model](multilingual_mt/group_then_scale_dynamic_mixture-of-experts_multilingual_language_model.md)**

:   提出 DMoE——基于参数偏差的动态语言分组 + 选择性 MoE 层扩展方法，通过仅 10 步微调量化语言间相似性，将相似语言分组共享同一 expert，只在参数偏差大的层（语言特定层）扩展为 MoE 层，在 18~128 种语言上 PPL 比持续预训练降低 11.4%，用 3.6 倍少的参数超越 X-ELM 9.6%。

**[Hierarchical News Clustering](multilingual_mt/hierarchical_news_clustering.md)**

:   本文提出利用多语言 Matryoshka 嵌入的分层特性进行新闻文章聚类：低维捕捉主题级相似度、中维捕捉叙事级相似度、高维捕捉事件级相似度，结合改良的 RAC 层级聚类算法，在 SemEval 2022 Task 8 上达到 SOTA（Pearson ρ = 0.816）。

**[Just Go Parallel Improving The Multilingual Capabilities Of Large Language Model](multilingual_mt/just_go_parallel_improving_the_multilingual_capabilities_of_large_language_model.md)**

:   系统研究在 decoder-only LLM 训练中加入平行数据对多语言能力的影响，发现将平行数据放在训练末期效果最好，且平行数据显著优于等量的单语数据；LLM 无法自动泛化到训练方向的反向翻译。

**[Knowcoder-X Boosting Multilingual Information Extraction Via Code](multilingual_mt/knowcoder-x_boosting_multilingual_information_extraction_via_code.md)**

:   提出 KnowCoder-X，通过统一的 Python 类表示多语言 IE schema，并引入 IE 跨语言对齐指令微调阶段（含高质量 ParallelNER 数据集），在 64 个 IE 基准上大幅提升跨语言信息抽取性能。

**[Laca Crosslingual Absa](multilingual_mt/laca_crosslingual_absa.md)**

:   提出 LACA 框架，利用 LLM 为目标语言生成高质量伪标注数据（而非依赖机器翻译），在六种语言上显著提升跨语言 ABSA 性能，在 mBERT 和 XLM-R 上分别平均超过前 SOTA 1.50% 和 2.62%。

**[Langmark A Multilingual Dataset For Automatic Post-Editing](multilingual_mt/langmark_a_multilingual_dataset_for_automatic_post-editing.md)**

:   发布 LangMark——一个包含 206,983 个三元组、覆盖英语到七种语言的大规模多语言自动后编辑（APE）数据集，并证明 LLM 配合 few-shot prompting 能有效改善专有 NMT 引擎的输出质量。

**[Langsamp Multilingual Pretraining](multilingual_mt/langsamp_multilingual_pretraining.md)**

:   提出 LangSAMP 方法，在多语言预训练中将语言和文字系统 (script) embedding 添加到 Transformer 输出端（而非输入端），使模型主干学到更语言中立的表示，在 500+ 语言的零样本跨语言迁移中一致优于基线。

**[Lemonade A Large Multilingual Expert-Annotated Abstractive Event Dataset For The](multilingual_mt/lemonade_a_large_multilingual_expert-annotated_abstractive_event_dataset_for_the.md)**

:   发布 Lemonade——基于 ACLED 冲突数据的大规模多语言专家标注事件数据集（39,786 事件，20 种语言，171 个国家，10,707 实体），提出 Abstractive Event Extraction (AEE) 新任务范式，事件参数不限于文本 span 而是归一化为数值/类别/实体，配套 Zest 零样本实体链接系统在 AEL 子任务上 F1=45.7% 大幅超越 baseline 的 23.7%。

**[Less But Better Efficient Multilingual Expansion](multilingual_mt/less_but_better_efficient_multilingual_expansion.md)**

:   分析 LLM 不同层间的跨语言表征相似度，提出 LayerMoE 按层分配不同数量的新语言专家（高相似层少分配、低相似层多分配），用 60% 更少的专家参数超越 SOTA，并通过在高相似层添加路由分类器进一步缓解灾难性遗忘。

**[Lost In Multilinguality Dissecting Cross-Lingual Factual Inconsistency In Transf](multilingual_mt/lost_in_multilinguality_dissecting_cross-lingual_factual_inconsistency_in_transf.md)**

:   用机制可解释性方法解剖多语言 LLM 的跨语言事实不一致问题，发现模型在大多数层中以语言无关的概念空间处理知识，但在最后几层的"语言转换"过程中失败导致不一致，提出线性快捷方法绕过最后层以提升一致性和准确率。

**[Low Resource Translation](multilingual_mt/low_resource_translation.md)**

:   将语法书辅助的极低资源翻译（XLR MT）分解为语法规则检索和规则应用两步，并提出用代码格式表示语法规则以提升 LLM 在两步中的表现，在壮语翻译上实现了 13.1% BLEU 的提升。

**[M-Mad Multidimensional Multi-Agent Debate For Advanced Machine Translation Evalu](multilingual_mt/m-mad_multidimensional_multi-agent_debate_for_advanced_machine_translation_evalu.md)**

:   提出 M-MAD 框架，将 MQM 评估标准解耦为独立维度（准确性、流畅性、风格、术语），在每个维度内进行多智能体正反方辩论，最后由裁判智能体综合各维度结果，在 segment 级别显著超越已有 LLM-as-a-judge 方法，甚至用 GPT-4o mini 就能媲美 SOTA 有参考自动指标。

**[M2Rc-Eval Massively Multilingual Repository-Level Code Completion Evaluation](multilingual_mt/m2rc-eval_massively_multilingual_repository-level_code_completion_evaluation.md)**

:   提出覆盖18种编程语言的大规模多语言仓库级代码补全基准 M2rc-Eval，配合基于 AST 的桶级和语义级细粒度标注，并构建 M2rc-Instruct 指令语料以提升模型性能。

**[M3Finmeeting A Multilingual Multi-Sector And Multi-Task Financial Meeting Unders](multilingual_mt/m3finmeeting_a_multilingual_multi-sector_and_multi-task_financial_meeting_unders.md)**

:   构建了 M3FinMeeting——首个面向金融会议的多语言（中英日）、多行业、多任务评测基准，包含 600 场真实金融会议的摘要、QA 对抽取和问答三项任务，揭示了当前最先进 LLM 在金融会议理解上仍有显著提升空间。

**[M Rewardbench](multilingual_mt/m_rewardbench.md)**

:   构建首个多语言奖励模型评估基准M-RewardBench（23种语言、2.87K偏好实例，覆盖对话/安全/推理/翻译四类能力），系统评估多种RM后发现英语与非英语RM性能存在显著差距，且翻译质量和语言资源量对RM表现有重要影响。

**[Machine Translation Models Are Zero-Shot Detectors Of Translation Direction](multilingual_mt/machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)**

:   提出一种基于 NMT 模型翻译概率的无监督翻译方向检测方法：若 $p(\text{translation}|\text{original}) > p(\text{original}|\text{translation})$，则可零样本判断平行文本的原始翻译方向，NMT 翻译的文档级检测准确率达 96%。

**[Marco Bench Multilingual If](multilingual_mt/marco_bench_multilingual_if.md)**

:   将英文IFEval基准扩展到30种语言并进行文化本地化，揭示LLM在多语言指令遵循中高/低资源语言间25-35%的准确率差距，以及机器翻译数据低估模型性能7-22%。

**[Memorization Inheritance Seqkd](multilingual_mt/memorization_inheritance_seqkd.md)**

:   本文首次系统研究了序列级知识蒸馏（SeqKD）中教师模型的记忆行为如何传递给学生模型，发现学生模型虽未直接接触原始训练数据，但其提取式记忆率比基线模型高 57%，幻觉率也增加，并提出 Adaptive-SeqKD 通过在高质量子集上微调教师来缓解这些问题。

**[Mid Layer Crosslingual Alignment](multilingual_mt/mid_layer_crosslingual_alignment.md)**

:   通过分析 1000+ 语言对发现 LLM 中间层具有最强的跨语言对齐潜力，提出在任务训练中集成中间层对齐目标（对比损失），在槽填充（F1 61.7%）、机器翻译（BLEU 32.3）和结构化文本生成上显著提升跨语言迁移，对未见语言也有效。

**[Milic-Eval Benchmarking Multilingual Llms For Chinas Minority Languages](multilingual_mt/milic-eval_benchmarking_multilingual_llms_for_chinas_minority_languages.md)**

:   构建首个中国少数民族语言 LLM 评估基准 MiLiC-Eval，包含 24K 实例覆盖 9 个任务、聚焦藏语/维吴尔语/哈萨克语/蒙古语 4 种语言，发现开源 LLM 在语法密集型任务和多文字语言上表现极差。

**[Modular Sentence Encoders](multilingual_mt/modular_sentence_encoders.md)**

:   本文提出模块化多语言句子编码器训练方案：先训练语言特定模块（embedding + 语言适配器 + 句子编码适配器）缓解多语言诅咒，再训练跨语言对齐适配器同时使用平行和释义数据解决不同跨语言任务间的性能权衡，在 4 个任务和 23 种语言上全面优于单体模型训练。

**[Moscar A Large-Scale Multilingual And Multimodal Document-Level Corpus](multilingual_mt/moscar_a_large-scale_multilingual_and_multimodal_document-level_corpus.md)**

:   提出 mOSCAR——首个大规模多语言多模态文档级语料库（163种语言、303M文档、200B tokens、1.15B图片），从 Common Crawl 中提取交错的图文文档，并证明在此数据上训练的多语言 mLLM 能获得显著的 few-shot 学习提升。

**[Msqad Multilingual Ethical Bias](multilingual_mt/msqad_multilingual_ethical_bias.md)**

:   提出多语言敏感问答数据集MSQAD（基于Human Rights Watch 17个人权话题），通过McNemar检验和PERMANOVA检验两种统计假设检验方法，系统验证了LLM在不同语言下对相同敏感问题的回答存在显著伦理偏差——中文和印地语拒绝率最高，西班牙语和德语最容易生成不当回答，且该偏差在7个不同LLM中普遍存在。

**[Mt Eval Human Parity](multilingual_mt/mt_eval_human_parity.md)**

:   首次将人类基线引入 WMT Metrics Shared Task 的排名，发现最先进的自动指标经常与人类评估者排名持平甚至更高，但论证了现在声称"人类对等"为时尚早，并讨论了衡量 MT 评估进步的根本困难。

**[Mtvqa Benchmarking Multilingual Text-Centric Visual Question Answering](multilingual_mt/mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)**

:   构建了 MTVQA——首个覆盖 9 种语言的多语言文本中心视觉问答基准，通过人类专家标注解决翻译方法的"视觉-文本不对齐"问题，评估显示最佳 MLLM（InternVL-2.5，32.2%）与人类表现（79.7%）差距巨大，揭示了多语言文本理解的严峻挑战。

**[Multi-Perspective Alignment For Increasing Naturalness In Neural Machine Transla](multilingual_mt/multi-perspective_alignment_for_increasing_naturalness_in_neural_machine_transla.md)**

:   提出多视角对齐框架 (Multi-perspective Alignment)，同时奖励翻译自然度和内容保留，通过翻译体分类器和 COMET 的联合奖励信号对 NMT 模型进行强化学习微调，使译文词汇更丰富且不损失翻译准确度。

**[Multilingual Encoder Knows More Than You Realize Shared Weights Pretraining For ](multilingual_mt/multilingual_encoder_knows_more_than_you_realize_shared_weights_pretraining_for_.md)**

:   提出 XLM-SWCM 框架，通过将多语言编码器权重复用到解码器中（CustomDecoderLayer 共享 + NormalDecoderLayer 随机初始化交替插入），以 457M 参数在极低资源语言（藏语）上超越 13B 参数的 MC2-LLaMA，藏语摘要 ROUGE-L 达 25.7 vs 16.1。

**[Multilingual Llm English Accent](multilingual_mt/multilingual_llm_english_accent.md)**

:   本文揭示多语言 LLM 在非英语语言生成中存在"英语口音"——词汇和句法上偏向英语模式，提出了基于 JSD（词汇分布）和 WL 图核+MMD（句法依赖树）的语料级自然度指标，并通过 DPO 对齐方法有效提升目标语言的自然度。

**[Multilingual Speech Data Quality](multilingual_mt/multilingual_speech_data_quality.md)**

:   对三大公开多语言语音数据集（Common Voice、FLEURS、VoxPopuli）进行系统质量审计，发现低资源语言存在严重的微观和宏观质量问题，并提出基于社会语言学意识的数据集创建指南。

**[Nametag 3 A Tool And A Service For Multilingualmultitagset Ner](multilingual_mt/nametag_3_a_tool_and_a_service_for_multilingualmultitagset_ner.md)**

:   本文介绍 NameTag 3，一个开源的多语言、多数据集、多标签集命名实体识别工具和云服务，基于微调的预训练语言模型，单个 355M 参数模型在 15 种语言的 21 个测试集上达到 SOTA，同时比 DeepSeek-R1 等 LLM 快 10,000 倍以上。

**[Probing Llms For Multilingual Discourse Generalization Through A Unified Label S](multilingual_mt/probing_llms_for_multilingual_discourse_generalization_through_a_unified_label_s.md)**

:   本文提出首个跨框架、跨语言的统一篇章关系标签集（17类），并通过对23个LLM的注意力探针实验，证明多语言LLM能够在中间层编码跨语言可迁移的篇章级表征，且多语言训练和模型规模共同提升泛化能力。

**[Q2E Query-To-Event Decomposition For Zero-Shot Multilingual Text-To-Video Retrie](multilingual_mt/q2e_query-to-event_decomposition_for_zero-shot_multilingual_text-to-video_retrie.md)**

:   Q2E 提出了一种零样本的查询到事件分解方法，利用 LLM 和 VLM 的参数化世界知识将简单查询分解为前因/当前/后果事件，并结合视频的视觉描述和语音转录，通过逆熵融合排序实现 SOTA 的多语言文本到视频检索性能。

**[Registering Source Tokens To Target Language Spaces In Multilingual Neural Machi](multilingual_mt/registering_source_tokens_to_target_language_spaces_in_multilingual_neural_machi.md)**

:   提出 Registering 方法：在源语言和目标语言 token 之间插入一组目标语言标记（registers），通过修改注意力掩码使目标生成仅依赖 registers 的激活，彻底解决多语言翻译中的 off-target 问题，使小模型 MITRE-913M 超越 NLLB-3.3B。

**[Semantic Aware Linear Transfer By Recycling Pre-Trained Language Models For Cros](multilingual_mt/semantic_aware_linear_transfer_by_recycling_pre-trained_language_models_for_cros.md)**

:   提出 SALT（Semantic Aware Linear Transfer），通过为每个非共享词表 token 基于语义相似的共享 token 对构建独立的最小二乘变换矩阵，将目标语言 PLM 的丰富嵌入表示迁移到英语中心 LLM 的嵌入空间，在下游任务、持续预训练收敛速度和跨语言理解上均优于现有方法。

**[Seqpo-Simt Sequential Policy Optimization For Simultaneous Machine Translation](multilingual_mt/seqpo-simt_sequential_policy_optimization_for_simultaneous_machine_translation.md)**

:   将同步机器翻译（SiMT）建模为多步序列决策问题，提出 SeqPO-SiMT 策略优化框架，融合翻译质量和延迟的奖励信号，在 7B LLM 上实现 SiMT 性能媲美离线翻译的强模型。

**[Shifcon Nondominant Language](multilingual_mt/shifcon_nondominant_language.md)**

:   提出 ShifCon 框架，通过将非优势语言的表示 shift 到优势语言子空间以获取更丰富的模型知识，再 shift 回原语言子空间进行生成，结合多语言对比学习，显著提升低资源语言的表现。

**[Statement-Tuning Enables Efficient Cross-Lingual Generalization In Encoder-Only ](multilingual_mt/statement-tuning_enables_efficient_cross-lingual_generalization_in_encoder-only_.md)**

:   将 Statement-Tuning 方法扩展到多语言场景，证明仅 276M 参数的 mDeBERTa 编码器模型通过多语言 Statement-Tuning 微调后，能在未见任务和未见语言上实现跨语言零样本泛化，在多个 NLU 任务上匹敌甚至超越 70B+ 参数的生成式 LLM。

**[Team Ack At Semeval-2025 Task 2 Beyond Word-For-Word Machine Translation For Eng](multilingual_mt/team_ack_at_semeval-2025_task_2_beyond_word-for-word_machine_translation_for_eng.md)**

:   本文在 SemEval-2025 Task 2 中系统评估了 13 个模型（LLM + 传统 MT）在英韩实体密集文本翻译上的表现，通过自动指标和双语人工评估揭示了 LLM 虽优于传统 MT 但在需要文化适应的实体翻译上仍普遍失败，并构建了翻译错误分类体系。

**[The Hidden Space Of Safety Understanding Preference-Tuned Llms In Multilingual C](multilingual_mt/the_hidden_space_of_safety_understanding_preference-tuned_llms_in_multilingual_c.md)**

:   本文系统分析了偏好调优（RLHF/DPO 等）对 LLM 内部表示空间在多语言场景下的影响，发现对齐机制在英语上能有效分离有害/无害内容的隐空间表示，但在印地语、中文、德语等非英语语言上效果显著退化，揭示了当前对齐方法存在严重的单语偏差问题。

**[Thor-Moe Hierarchical Task-Guided And Context-Responsive Routing For Neural Mach](multilingual_mt/thor-moe_hierarchical_task-guided_and_context-responsive_routing_for_neural_mach.md)**

:   提出THOR-MoE框架，通过层级任务引导路由（自动预测领域/语言并生成混合任务表示来选任务级专家子集）和上下文响应路由（将全局上下文注入token表示以辅助专家选择），在多领域和多语言翻译中以更少激活参数获得显著性能提升。

**[Towards Global Ai Inclusivity A Large-Scale Multilingual Terminology Dataset Gis](multilingual_mt/towards_global_ai_inclusivity_a_large-scale_multilingual_terminology_dataset_gis.md)**

:   构建了首个大规模多语言 AI 术语数据集 GIST，包含从顶级 AI 会议论文中提取的 5K 术语及其阿拉伯语、中文、法语、日语和俄语翻译，并探索了三种无需重训练的术语集成方法来提升机器翻译质量。

**[Trans-Zero Self-Play Incentivizes Large Language Models For Multilingual Transla](multilingual_mt/trans-zero_self-play_incentivizes_large_language_models_for_multilingual_transla.md)**

:   提出 Trans-Zero 自博弈框架，仅使用单语数据，通过遗传蒙特卡洛树搜索（G-MCTS）在多语言翻译过程中探索语义一致的候选翻译，结合偏好优化实现无平行数据的多语言翻译训练，性能可媲美大规模监督微调方法。

**[Translation Robustness](multilingual_mt/translation_robustness.md)**

:   通过合成噪声和社交媒体文本的系统性实验，证明现代大规模预训练翻译模型（LLM）在未经任何专门鲁棒性训练的情况下，对多种输入噪声的鲁棒性已远超传统 NMT 模型，鲁棒性随模型规模增长自然提升。

**[Unveiling The Power Of Source Source-Based Minimum Bayes Risk Decoding For Neura](multilingual_mt/unveiling_the_power_of_source_source-based_minimum_bayes_risk_decoding_for_neura.md)**

:   提出 source-based MBR (sMBR) 解码方法，利用释义/回译生成的准源端句子作为"支持假设"，结合无参考 QE 指标作为效用函数，首次在 MBR 解码中完全依赖源端信息，在经典和 LLM 两种 NMT 设置下均优于 QE reranking 和标准 MBR 解码。

**[Watching The Watchers Exposing Gender Disparities In Machine Translation Quality](multilingual_mt/watching_the_watchers_exposing_gender_disparities_in_machine_translation_quality.md)**

:   系统性地揭示了机器翻译质量估计(QE)指标中的性别偏见：当源语言性别模糊时，阳性形式翻译得分系统性地高于阴性形式，性别中性翻译被惩罚；即使有上下文消歧线索，阴性指称的错误率仍显著高于阳性，且该偏见会传播到数据过滤和解码等下游任务中。

**[Zipa A Family Of Efficient Models For Multilingual Phone Recognition](multilingual_mt/zipa_a_family_of_efficient_models_for_multilingual_phone_recognition.md)**

:   提出 Zipa 系列高效语音模型，基于 Zipformer 骨干和 IpaPack++（17,132 小时多语言标注数据），在多语言音素识别上达到 SOTA，64M 参数模型即超越现有 300M 模型，并通过噪声学生训练在 4000+ 种语言上进一步提升。

---

## 🔍 信息检索/RAG { #information_retrieval }

**[A Reality Check On Context Utilisation For Retrieval-Augmented Generation](information_retrieval/a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)**

:   提出DRUID真实世界事实验证数据集和ACU评估指标，揭示合成数据集（CounterFact、ConflictQA）夸大了上下文特征的影响，导致对LLM上下文利用能力的过度乐观评估，呼吁使用真实检索数据研究RAG。

**[A Text Is Worth Several Tokens Text Embedding From Llms Secretly Aligns Well Wit](information_retrieval/a_text_is_worth_several_tokens_text_embedding_from_llms_secretly_aligns_well_wit.md)**

:   揭示 LLM 文本嵌入的有趣现象：将嵌入向量通过解码层映射回词表空间后，解码概率最高的 token 与输入文本的关键词高度对齐；进一步通过谱分析发现这一现象主要受第一主成分控制，并据此提出一种简洁的稀疏检索方法，达到原密集检索 80%+ 的效果。

**[Accelerating Adaptive Retrieval Augmented Generation Via Instruction-Driven Repr](information_retrieval/accelerating_adaptive_retrieval_augmented_generation_via_instruction-driven_repr.md)**

:   首次识别自适应 RAG（A-RAG）中多轮检索结果重叠导致的冗余计算问题，提出 IDR²（Instruction-Driven Representation Reduction）框架：跨迭代 KV 缓存共享（CICS）加速预填充 2.79 倍、指令引导去重增强（IDGR）帮助 LLM 正确处理缓存vs新文档、信息引导并行生成（IGPG）加速解码 2.33 倍，总体 A-RAG 流程加速 2.0 倍且不损失生成质量。

**[Air-Bench Automated Heterogeneous Information Retrieval Benchmark](information_retrieval/air-bench_automated_heterogeneous_information_retrieval_benchmark.md)**

:   提出 AIR-Bench——首个自动化、异构、动态的信息检索评测基准，通过 LLM 自动从真实语料生成高质量测试数据（三阶段管线：语料准备→候选生成→质量控制），覆盖 2 类任务、9 个领域、13 种语言共 69 个数据集，与人工标注数据高度一致，且持续动态更新避免数据泄露。

**[Beyond True Or False Retrieval-Augmented Hierarchical Analysis Of Nuanced Claims](information_retrieval/beyond_true_or_false_retrieval-augmented_hierarchical_analysis_of_nuanced_claims.md)**

:   提出 ClaimSpect 框架，将复杂声明自动分解为层次化的方面（aspect）树，并通过区分性检索从语料库中发现各方面的支持/中立/反对观点及其共识程度。

**[CART: A Generative Cross-Modal Retrieval Framework with Coarse-To-Fine Semantic Modeling](information_retrieval/cart_a_generative_cross-modal_retrieval_framework_with_coarse-to-fine_semantic_m.md)**

**[Coir A Comprehensive Benchmark For Code Information Retrieval Models](information_retrieval/coir_a_comprehensive_benchmark_for_code_information_retrieval_models.md)**

:   提出 CoIR，首个全面的代码信息检索基准，包含 10 个数据集、覆盖 4 大类 8 个子任务和 14 种编程语言，揭示了即使是 SOTA 检索模型在代码检索中也表现不佳，并指出许多模型已在现有排行榜上过拟合。

**[Collapse Dense Retrievers](information_retrieval/collapse_dense_retrievers.md)**

:   本文首次系统研究稠密检索器中多种启发式偏见（简短偏见、前置偏见、字面偏见、重复偏见）的个体和组合效应，发现当多种偏见叠加时，检索器选择包含答案的文档的概率低于10%，且这些偏见可被利用来操纵RAG系统，导致34%的性能下降。

**[Comrag Retrieval-Augmented Generation With Dynamic Vector Stores For Real-Time C](information_retrieval/comrag_retrieval-augmented_generation_with_dynamic_vector_stores_for_real-time_c.md)**

:   提出ComRAG——一个面向工业实时社区问答的检索增强生成框架，通过**静态知识向量库+高/低质量动态QA向量库**的三库架构和**质心记忆机制**，在三个CQA数据集上获得向量相似度最高25.9%的提升，同时降低延迟8.7%-23.3%。

**[Core Mmrag Knowledge Reconciliation](information_retrieval/core_mmrag_knowledge_reconciliation.md)**

:   CoRe-MMRAG 提出了一个端到端多模态 RAG 框架，通过四阶段流水线（参数知识生成→视觉-文本联合重排序→外部知识生成→内外知识整合）解决参数知识-检索知识不一致(PRKI)和视觉-文本知识不一致(VTKI)两个问题，在 InfoSeek 和 Encyclopedic-VQA 上分别提升 5.6% 和 9.3%。

**[Divide Then Align Rag Knowledge Boundary](information_retrieval/divide_then_align_rag_knowledge_boundary.md)**

:   DTA 提出将 RAG 查询按参数知识边界和检索知识边界划分为四个象限，对"两者都不知道"的查询构造偏好数据用 DPO 训练模型回答"我不知道"，解决了 RAFT 模型即使在检索完全噪声时也强行生成答案的问题，在准确率和适当弃权之间实现了有效平衡。

**[Dont Reinvent The Wheel Efficient Instruction-Following Text Embedding Based On ](information_retrieval/dont_reinvent_the_wheel_efficient_instruction-following_text_embedding_based_on_.md)**

:   提出 GSTransform 框架，通过轻量级空间变换将预计算的通用嵌入实时适配到用户指令指定的语义空间，避免每次新指令都重新编码全部语料，在 9 个数据集上平均得分 66.01（SOTA 基线 55.31），同时实现 6~300 倍实时加速。

**[Drag Distilling Rag Slm](information_retrieval/drag_distilling_rag_slm.md)**

:   DRAG 提出了一种从大模型向小模型蒸馏 RAG 能力的框架：用大模型（如 GPT-4o）为给定问题生成证据和知识图谱三元组，经排序过滤后作为结构化上下文输入给小模型（2B-9B），无需微调即可将小模型在 ARC-C 上提升高达 27.7%，同时显著减少幻觉。

**[DRAMA: Diverse Augmentation from LLMs to Smaller Dense Retrievers](information_retrieval/drama_diverse_augmentation_from_large_language_models_to_smaller_dense_retriever.md)**

:   通过多策略 LLM 数据增强+知识蒸馏创建小型高效检索器 Drama 0.3B，在 BEIR 上达到与 1B 模型相当的性能。

**[Empaths At Semeval-2025 Task 11 Retrieval-Augmented Approach To Perceived Emotio](information_retrieval/empaths_at_semeval-2025_task_11_retrieval-augmented_approach_to_perceived_emotio.md)**

:   提出 EmoRAG 系统，用检索增强生成（RAG）管道结合多 LLM 集成聚合，在 SemEval-2025 Task 11 多标签情感检测任务上无需额外训练即在 28 种语言中取得有竞争力的结果，平均 F1-micro 0.638。

**[Enhancing Lexicon-Based Text Embeddings with Large Language Models](information_retrieval/enhancing_lexicon-based_text_embeddings_with_large_language_models.md)**

:   用 LLM 增强基于词典的稀疏文本嵌入，在保持可解释性的同时提升检索性能。

**[Evaluation Of Attribution Bias In Generator-Aware Retrieval-Augmented Large Lang](information_retrieval/evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)**

:   定义并研究 RAG 中 LLM 对作者身份信息的归因敏感性和偏差，通过反事实评估发现告知 LLM 文档作者身份可显著改变归因质量 3-18%，且 LLM 存在对人类作者身份的归因偏差。

**[Exit Context-Aware Extractive Compression For Enhancing Retrieval-Augmented Gene](information_retrieval/exit_context-aware_extractive_compression_for_enhancing_retrieval-augmented_gene.md)**

:   提出 EXIT——一种抽取式上下文压缩框架，通过上下文感知的句子级二分类并行选取与查询相关的句子，在 QA 准确率和推理延迟上同时优于现有的抽生式和抽取式压缩方法。

**[Faithfulrag Fact Level Conflict](information_retrieval/faithfulrag_fact_level_conflict.md)**

:   发现现有忠实 RAG 方法通过强制抑制参数知识来实现上下文忠实，但这增加了误解上下文的风险（不忠实错误减少 6.65% 的同时错误匹配增加 6.42%）。提出 FaithfulRAG，通过事实级冲突识别（自事实挖掘）和冲突推理（自思考模块）解决知识冲突，在 FaithEval/SQuAD/MuSiQue/RealtimeQA 上超越最强基线 8-9 个百分点。

**[Flashbackefficient Retrieval-Augmented Language Modeling For Long Context Infere](information_retrieval/flashbackefficient_retrieval-augmented_language_modeling_for_long_context_infere.md)**

:   针对检索增强语言模型(RALM)中因检索内容前置(prepending)导致 KV cache 反复重算的推理效率问题，提出 FlashBack，将检索内容后置(appending)以保留输入的 KV cache，并用 Marking Token + LoRA 微调适配新的上下文模式，在 Llama 2-7B 上实现最高 4 倍推理加速且 perplexity 持平。

**[Flexrag A Flexible And Comprehensive Framework For Retrieval-Augmented Generatio](information_retrieval/flexrag_a_flexible_and_comprehensive_framework_for_retrieval-augmented_generatio.md)**

:   提出 FlexRAG，一个面向研究和原型开发的开源 RAG 框架，支持文本、多模态和 Web 检索三种模式，通过内存映射和异步处理实现比同类框架（FlashRAG）低一个数量级的资源开销。

**[From Ambiguity To Accuracy The Transformative Effect Of Coreference Resolution O](information_retrieval/from_ambiguity_to_accuracy_the_transformative_effect_of_coreference_resolution_o.md)**

:   本文系统研究了共指消解（coreference resolution）对 RAG 系统中文档检索和问答生成两阶段的影响，发现共指消解能一致性提升检索性能（尤其 mean pooling 模型受益最大），在 QA 任务中小模型的性能提升显著大于大模型，甚至使小模型达到大模型的基线水平。

**[Gainrag Preference Alignment](information_retrieval/gainrag_preference_alignment.md)**

:   发现 RAG 中检索器的"相关性"与 LLM 的"偏好"存在偏差——含正确答案的段落仍可能导致错误生成，而间接相关段落反而有用。提出 GainRAG，用基于对比解码困惑度的"增益"信号量化 LLM 偏好，训练轻量选择器在检索结果中选择真正有"增益"的段落，在 6 个 QA 数据集上显著超越 Standard RAG 和 Rerank 基线。

**[Garage A Benchmark With Grounding Annotations For Rag Evaluation](information_retrieval/garage_a_benchmark_with_grounding_annotations_for_rag_evaluation.md)**

:   GaRAGe 是一个包含 2366 个问题和超过 35K 条人工标注 grounding 段落的 RAG 基准，通过细粒度的 grounding 相关性标注，系统评估 LLM 在 RAG 场景下识别相关信息、拒绝回答和归因引用的能力。

**[Genie Worksheets Tod Agent](information_retrieval/genie_worksheets_tod_agent.md)**

:   Genie 提出了一个可编程的知识密集型任务导向对话框架，通过声明式 Worksheet 规范定义 Agent 策略，将 LLM 限制在语义解析和回复生成两个角色，由算法化运行时系统强制执行策略，实现从 21.8% 到 82.8% 的真实任务完成率提升。

**[Gor Rag Long Context Summary](information_retrieval/gor_rag_long_context_summary.md)**

:   提出 Graph of Records（GoR），将 LLM 历史响应与检索文本块构建为图结构，用 GNN 学习节点间的语义和逻辑关联，配合 BERTScore 自监督训练目标，在四个长文本全局摘要数据集上比检索基线提升 8-19%（ROUGE 指标）。

**[Graf Graph Retrieval Augmented By Facts For Romanian Legal Multi-Choice Question](information_retrieval/graf_graph_retrieval_augmented_by_facts_for_romanian_legal_multi-choice_question.md)**

:   提出GRAF算法，结合法律知识图谱（Law-RoG）和图注意力网络进行罗马尼亚语法律多选题问答，同时开源了首个罗马尼亚语法律MCQA数据集JuRO（10,836题）和法律语料库CROL。

**[Gumbel Reranking](information_retrieval/gumbel_reranking.md)**

:   将 RAG 系统中的重排序过程重新建模为文档级 Top-k 注意力掩码问题，利用 Gumbel 技巧和松弛 Top-k 采样实现端到端可微优化，直接最小化最终语言建模损失，在 HotpotQA 上 Recall@5 提升 10.4%。

**[Health-LLM: Personalized Retrieval-Augmented Disease Prediction System](information_retrieval/health-llm_personalized_retrieval-augmented_disease_prediction_system.md)**

:   提出 Health-LLM 个性化 RAG 疾病预测系统，结合可穿戴设备数据和医学知识库进行健康风险评估。

**[Helios Harmonizing Early Fusion Late Fusion And Llm Reasoning For Multi-Granular](information_retrieval/helios_harmonizing_early_fusion_late_fusion_and_llm_reasoning_for_multi-granular.md)**

:   提出 HELIOS 框架，结合早期融合（边级别二部子图检索）、晚期融合（查询相关节点扩展）和 LLM 推理（星图级逻辑推断），在 OTT-QA 上 recall 和 nDCG 分别提升达 42.6% 和 39.9%。

**[Hierarchical Document Refinement For Long-Context Retrieval-Augmented Generation](information_retrieval/hierarchical_document_refinement_for_long-context_retrieval-augmented_generation.md)**

:   提出 LongRefiner，一个即插即用的长文档精炼系统，通过双层查询分析、层次化文档结构化和自适应精炼三个步骤，在 7 个 QA 数据集上以**仅 1/10 的 token 预算**实现了优于全文输入的性能，同时延迟仅为最佳基线的 1/10。

**[Hoh A Dynamic Benchmark For Evaluating The Impact Of Outdated Information On Ret](information_retrieval/hoh_a_dynamic_benchmark_for_evaluating_the_impact_of_outdated_information_on_ret.md)**

:   本文提出 HoH，首个专门评估过时信息对 RAG 系统影响的大规模动态基准，包含 96,124 个 QA 对和 219,463 篇文档，揭示了过时信息对 RAG 性能和安全性的严重危害。

**[Hybgrag Hybrid Rag Skb](information_retrieval/hybgrag_hybrid_rag_skb.md)**

:   提出 HybGRAG 方法，通过检索器库（Retriever Bank）同时利用文本和关系信息，配合 Critic 模块的自反思迭代纠正问题路由错误，在半结构化知识库上的混合问答任务中 Hit@1 平均提升 51%。

**[Hypothetical Documents Or Knowledge Leakage Rethinking Llm-Based Query Expansion](information_retrieval/hypothetical_documents_or_knowledge_leakage_rethinking_llm-based_query_expansion.md)**

:   质疑 LLM-based 查询扩展（HyDE/Query2doc）的性能提升是否来自"假设性文档生成"，发现性能增益仅在 LLM 生成的文档包含与 gold evidence 语义一致的句子时才一致出现，揭示了 benchmark 中可能存在的知识泄露问题。

**[Investigating Language Preference Of Multilingual Rag Systems](information_retrieval/investigating_language_preference_of_multilingual_rag_systems.md)**

:   系统研究多语言RAG系统在检索和生成两个阶段的语言偏好问题，提出 MLRS 指标量化检索器语言偏好，并设计 DKM-RAG 框架通过融合翻译段落与模型内部知识来缓解偏好问题。

**[Investigating The Robustness Of Retrieval-Augmented Generation At The Query Leve](information_retrieval/investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)**

:   系统研究 RAG 管道对查询级扰动（冗余插入、正式语气、歧义引入、拼写错误）的敏感性，提出解耦各模块鲁棒性的分析框架，发现检索器是性能瓶颈且不同扰动类型的主导因素不同。

**[Knowshiftqa Rag Knowledge Shifts](information_retrieval/knowshiftqa_rag_knowledge_shifts.md)**

:   构建了 KnowShiftQA 数据集（3,005 道题，覆盖 5 个学科），通过假设性知识更新模拟教科书与 LLM 参数知识的差异，系统评估 RAG 系统面对知识偏移时的鲁棒性，发现现有 RAG 系统在知识偏移下性能下降 22-27%。

**[Ldir Low-Dimensional Dense And Interpretable Text Embeddings With Relative Repre](information_retrieval/ldir_low-dimensional_dense_and_interpretable_text_embeddings_with_relative_repre.md)**

:   提出 LDIR 方法，通过最远点采样选取锚文本（anchor texts），计算待编码文本与各锚文本的语义相关度，构建低维（≤500 维）、稠密且可解释的文本嵌入，性能接近黑盒模型并显著优于已有可解释嵌入方法。

**[Llm Reranking Harmful Content](information_retrieval/llm_reranking_harmful_content.md)**

:   提出基于 LLM 的成对偏好重排序方法，在零样本和少样本设置下对社交媒体推荐序列中的有害内容进行降级排序，显著优于 Perspective API 和 OpenAI Moderation API 等工业级分类器，同时引入 PP-k 和 EWN 两个新评估指标。

**[Logical Consistency Is Vital Neural-Symbolic Information Retrieval For Negative-](information_retrieval/logical_consistency_is_vital_neural-symbolic_information_retrieval_for_negative-.md)**

:   提出 NS-IR，通过将自然语言查询和文档转换为一阶逻辑（FOL），利用逻辑对齐和连接词约束两项技术优化稠密检索嵌入，显著提升了负约束查询等复杂逻辑检索场景的性能。

**[Main-Rag Multi-Agent Filtering Retrieval-Augmented Generation](information_retrieval/main-rag_multi-agent_filtering_retrieval-augmented_generation.md)**

:   提出 MAIN-RAG，一个无需训练的多 Agent RAG 过滤框架，通过 Predictor→Judge→Final-Predictor 三个 LLM Agent 协作评估检索文档的相关性，并设计自适应阈值（基于分数均值和标准差）动态过滤噪声文档，在 4 个 QA 基准上实现 2-11% 的准确率提升。

**[Maximal Matching Matters Preventing Representation Collapse For Robust Cross-Mod](information_retrieval/maximal_matching_matters_preventing_representation_collapse_for_robust_cross-mod.md)**

:   提出 MaxMatch 方法，通过基于匈牙利算法的最大配对相似度和两个新损失函数，解决集合嵌入方法中的稀疏监督和集合坍塌问题，在 MS-COCO 和 Flickr30k 上取得 SOTA 性能。

**[Memerag A Multilingual End-To-End Meta-Evaluation Benchmark For Retrieval Augmen](information_retrieval/memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)**

:   构建首个原生多语言 RAG 元评估基准 MEMERAG，覆盖 5 种语言，通过流程图引导的标注达到高标注者一致性，用于评估和比较多语言 RAG 自动评估器。

**[Mitigating Lost-In-Retrieval Problems In Retrieval Augmented Multi-Hop Question ](information_retrieval/mitigating_lost-in-retrieval_problems_in_retrieval_augmented_multi-hop_question_.md)**

:   本文识别 RAG 多跳问答中的"检索丢失"（lost-in-retrieval）问题——子问题分解后后续子问题因缺少关键实体导致检索性能骤降，提出 ChainRAG 框架通过构建句子图 + 渐进式检索 + 子问题重写（补全缺失实体）形成完整推理链，在 MuSiQue、2Wiki、HotpotQA 三个数据集上一致超越基线。

**[Moc Mixtures Of Text Chunking Learners For Retrieval-Augmented Generation System](information_retrieval/moc_mixtures_of_text_chunking_learners_for_retrieval-augmented_generation_system.md)**

:   提出 Boundary Clarity 和 Chunk Stickiness 两个直接量化分块质量的指标，以及基于粒度感知混合专家（MoC）的文本分块框架，通过正则表达式引导的轻量化分块策略在 RAG 系统中取得优于传统方法和大模型直接分块的性能。

**[Mt-Raig Novel Benchmark And Evaluation Framework For Retrieval-Augmented Insight](information_retrieval/mt-raig_novel_benchmark_and_evaluation_framework_for_retrieval-augmented_insight.md)**

:   提出MT-RAIG Bench——首个面向多表格检索增强洞察生成的大规模基准，以及MT-RAIG Eval——基于分解的细粒度自动评估框架，实验表明即使前沿LLM在多表格推理上仍表现不佳（忠实度仅约40%，完整度约60%）。

**[Multilingual Retrieval Augmented Generation For Culturally-Sensitive Tasks A Ben](information_retrieval/multilingual_retrieval_augmented_generation_for_culturally-sensitive_tasks_a_ben.md)**

:   构建了 BordIRLines 基准数据集，包含 49 种语言的领土争端查询及配对的 Wikipedia 检索文档，系统评估了多语言 RAG 环境下的跨语言鲁棒性，发现**检索多语言文档能比仅检索同语言文档更好地提高响应一致性并减少地缘政治偏差**。

**[On Synthetic Data Strategies For Domain-Specific Generative Retrieval](information_retrieval/on_synthetic_data_strategies_for_domain-specific_generative_retrieval.md)**

:   本文系统研究了针对领域特定语料库训练生成式检索模型的合成数据策略，提出多粒度查询生成、约束条件查询和基于硬负样本的偏好学习方法，显著提升检索性能。

**[Optimized Text Embedding Models And Benchmarks For Amharic Passage Retrieval](information_retrieval/optimized_text_embedding_models_and_benchmarks_for_amharic_passage_retrieval.md)**

:   针对低资源、形态丰富的阿姆哈拉语（Amharic），提出基于预训练 Amharic BERT/RoBERTa 的稠密检索模型和 ColBERT 晚期交互模型，在参数量远小于多语言基线的情况下大幅提升段落检索效果，并建立了该语言首个系统性检索基准。

**[Pandora Box Rag Noise](information_retrieval/pandora_box_rag_noise.md)**

:   本文从语言学视角定义了 RAG 系统中的 7 种噪声类型，构建了 NoiserBench 综合评测框架，通过 8 个 LLM 的大规模实验发现噪声可分为有害噪声（反事实、支持性、拼写）和有益噪声（语义、数据类型、非法句子），其中有益噪声反而能提升模型准确率 1-3%。

**[Parenting Optimizing Knowledge Selection Of Retrievalaugmented](information_retrieval/parenting_optimizing_knowledge_selection_of_retrievalaugmented.md)**

:   受人脑功能分区启发，提出 Parenting 框架，通过解耦并定位 LLM 参数空间中与"上下文遵循"(adherence)和"噪声鲁棒"(robustness)相关的子空间，并为不同子空间设计定制化微调策略，实现两种能力的平衡提升。

**[Psycholinguistic Visual Semantic](information_retrieval/psycholinguistic_visual_semantic.md)**

:   提出 Neighborhood Stability Measure (NSM)，仅通过文本嵌入空间中邻域的"尖锐度"来无监督估计词汇的可意象性(imageability)和具体性(concreteness)，无需视觉模态或生成模型即可超越现有方法。

**[Raemollm Retrieval Augmented Llms For Cross-Domain Misinformation Detection Usin](information_retrieval/raemollm_retrieval_augmented_llms_for_cross-domain_misinformation_detection_usin.md)**

:   提出 RAEmoLLM，首个基于情感信息检索的 RAG 框架，利用情感 LLM 的隐式嵌入构建检索数据库，为跨域虚假信息检测提供情感相关的 few-shot 示例，在三个基准上最高分别提升 15.64%、31.18% 和 15.73%（对比其他 few-shot 方法），无需微调。

**[Rare Retrieval Augmented Reasoning](information_retrieval/rare_retrieval_augmented_reasoning.md)**

:   提出 RARE，在 rStar 的 MCTS 推理框架中引入两个检索增强动作（A6: 基于原始问题生成搜索查询并检索，A7: 对子问题进行检索并重新回答），并用检索增强的事实性评分器（RAFS）替代原始判别器，使 LLaMA 3.1 在医学和常识推理任务上达到甚至超越 GPT-4o 的水平。

**[Redundancy Isotropy And Intrinsic Dimensionality Of Prompt-Based Text Embeddings](information_retrieval/redundancy_isotropy_and_intrinsic_dimensionality_of_prompt-based_text_embeddings.md)**

:   系统研究了基于Prompt的文本嵌入模型（如gte-Qwen2、E5-mistral等）在后处理降维下的性能鲁棒性，发现分类/聚类任务仅保留原始维度的0.5%即可基本保持性能，并通过内在维度（ID）和各向同性（IsoScore）两个指标定量解释了不同任务Prompt产生的嵌入冗余度差异。

**[Refind At Semeval-2025 Task 3 Retrieval-Augmented Factuality Hallucination Detec](information_retrieval/refind_at_semeval-2025_task_3_retrieval-augmented_factuality_hallucination_detec.md)**

:   提出 REFIND 框架，通过计算每个 token 在有无检索文档条件下的生成概率之比（Context Sensitivity Ratio, CSR），实现对 LLM 输出中幻觉片段的高效检测，在 SemEval-2025 Task 3 的 9 种语言上显著超越基线。

**[Reranking-Based Generation For Unbiased Perspective Summarization](information_retrieval/reranking-based_generation_for_unbiased_perspective_summarization.md)**

:   针对政治视角摘要任务，构建了受控测试集验证现有评估指标的可靠性，发现 LLM-based 指标远优于传统指标，并证明基于重排序（Reranking）的方法及在重排序数据上的 DPO 训练能显著提升摘要的覆盖性和忠实性。

**[Saferag Benchmarking Security In Retrieval-Augmented Generation Of Large Languag](information_retrieval/saferag_benchmarking_security_in_retrieval-augmented_generation_of_large_languag.md)**

:   提出 SafeRAG 安全基准，定义了四种增强型攻击任务（银噪声、上下文间冲突、软广告、白色拒绝服务），通过在 RAG 管道不同阶段注入攻击文本来系统评估 14 种 RAG 组件的安全漏洞，揭示了即使最明显的攻击也能轻松绕过现有防御。

**[Seal Scaling To Emphasize Attention For Long-Context Retrieval](information_retrieval/seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)**

:   通过剪枝分析发现特定注意力头与长上下文检索高度相关，提出 SEAL（可学习标量/向量缩放注意力头/通道），仅需 ~1024 个额外参数 + 50 个合成样本训练，即可将 LongChat-7B 在 31K 上的检索准确率从 0.32 提升到 0.88，且缩放参数可离线合并到权重中实现零推理开销。

**[Setr Set Selection Rag](information_retrieval/setr_set_selection_rag.md)**

:   提出从"逐个排序"到"集合选择"的 RAG 检索范式转换——SetR 通过 CoT 推理显式识别查询的信息需求，然后选择一组能共同满足这些需求的段落（而非逐个评分最相关的），在多跳 RAG 基准上超越 GPT-4o 级别的重排器。

**[Towards Adaptive Memory-Based Optimization For Enhanced Retrieval-Augmented Gene](information_retrieval/towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)**

:   提出 Amber 框架，通过 Agent 协作式记忆更新器、自适应信息收集器和多粒度内容过滤器三个组件协同工作，在迭代式 RAG 范式中提升开放域问答的检索效率和答案质量。

**[Towards Storage-Efficient Visual Document Retrieval An Empirical Study On Reduci](information_retrieval/towards_storage-efficient_visual_document_retrieval_an_empirical_study_on_reduci.md)**

:   系统性研究了视觉文档检索（VDR）中 patch 级别嵌入的压缩策略，发现 pruning 在 VDR 中本质不适用（简单随机剪枝反而最优），而 token merging 结合微调可在仅保留 2.8% 存储量时维持 94.6% 的检索性能（Light-ColPali/ColQwen2）。

**[Typed-Rag Type-Aware Decomposition Of Non-Factoid Questions For Retrieval-Augmen](information_retrieval/typed-rag_type-aware_decomposition_of_non-factoid_questions_for_retrieval-augmen.md)**

:   提出 Typed-RAG 框架，通过对非事实性问题（NFQ）进行类型感知的分解，将复杂的多方面问题拆解为单方面子查询，针对不同问题类型（辩论、经验、比较等）设计差异化的检索与生成策略，显著提升了 RAG 在 NFQA 中的表现。

**[Voxrag A Step Toward Transcription-Free Rag Systems In Spoken Question Answering](information_retrieval/voxrag_a_step_toward_transcription-free_rag_systems_in_spoken_question_answering.md)**

:   提出 VoxRAG，一个模块化的语音到语音检索增强生成系统，使用 CLAP 音频嵌入绕过转录直接从语音查询检索语义相关的音频片段，在播客问答场景中验证了无转录语音检索的可行性，Recall@10 在 somewhat relevant 片段上达到 0.60。

**[When Should Dense Retrievers Be Updated In Evolving Corpora Detecting Out-Of-Dis](information_retrieval/when_should_dense_retrievers_be_updated_in_evolving_corpora_detecting_out-of-dis.md)**

:   提出 GradNormIR，一种利用梯度范数在无 query 条件下检测 OOD 语料库的无监督方法，帮助判断稠密检索器何时需要更新以应对不断演化的文档集合。

---

## 📦 模型压缩 { #model_compression }

**[500Xcompressor Generalized Prompt Compression For Large Language Models](model_compression/500xcompressor_generalized_prompt_compression_for_large_language_models.md)**

:   提出 500xCompressor，将最多约 500 个自然语言 token 压缩为最少 1 个特殊 token 的 KV 值，实现 6x 到 480x 的压缩比，仅增加约 0.25% 的参数，LLM 在压缩后保留 62.26%-72.89% 的原始能力，显著超越 ICAE 基线。

**[Accurate Kv Cache Quantization With Outlier Tokens Tracing](model_compression/accurate_kv_cache_quantization_with_outlier_tokens_tracing.md)**

:   发现 KV Cache 的 outlier channel 中存在少量异常 token 偏离先前假设的均匀分布，提出 OTT（Outlier Tokens Tracing）方法，在量化过程中动态追踪并排除这些 token，在 2-bit 量化下实现 6.4x 内存压缩和 2.3x 吞吐提升，同时显著提高精度。

**[Aligndistil Token Level Alignment](model_compression/aligndistil_token_level_alignment.md)**

:   AlignDistil 证明了 RLHF 目标函数与 token 级蒸馏过程的理论等价性，并据此设计了一种简单的蒸馏方法：用 DPO 模型和反向 DPO 模型的 logit 分布线性组合构造教师分布，配合 token 自适应外推机制实现 token 级奖励优化，在 AlpacaEval 2.0、MT-Bench 和 Arena-Hard 上优于现有方法且收敛更快。

**[Apb Distributed Long Context](model_compression/apb_distributed_long_context.md)**

:   APB 提出了一种分布式长上下文推理框架，通过在序列并行框架中引入本地 KV cache 压缩和跨 GPU 传递压缩上下文块的机制，在不损失任务性能的前提下实现了相比 FlashAttn/RingAttn/StarAttn 分别高达 9.2x/4.2x/1.6x 的 prefill 加速。

**[Assigning Distinct Roles To Quantized And Low-Rank Matrices Toward Optimal Weigh](model_compression/assigning_distinct_roles_to_quantized_and_low-rank_matrices_toward_optimal_weigh.md)**

:   提出 ODLRI（Outlier-Driven Low-Rank Initialization）——在量化+低秩联合权重分解 $\mathbf{W} \approx \mathbf{Q} + \mathbf{LR}$ 中，将低秩分量专门分配给捕获激活敏感权重（异常值相关），量化分量处理剩余权重。通过这种"角色分配"初始化，稳定量化、降低激活感知误差，在 Llama2/3 和 Mistral 的低比特设置中一致提升困惑度和零样本精度。

**[Basic Reading Distillation](model_compression/basic_reading_distillation.md)**

:   本文提出基础阅读蒸馏（BRD），通过让教师LLM在通用语料上生成基础阅读行为数据（包括NER和问答），训练小型学生模型模仿这些行为，使564M参数的小模型在不接触下游任务数据的情况下就能在多种NLP任务上达到或超过20倍大的教师模型性能。

**[Beamlora Beam Constraint Lora](model_compression/beamlora_beam_constraint_lora.md)**

:   BeamLoRA 发现 LoRA 模块中不同 rank 的重要性存在显著差异且随训练动态演变，受 beam search 启发，提出在训练过程中动态评估 rank 重要性、剪枝不重要的 rank 并将参数空间扩展给重要 rank，在固定总 rank 下提升性能，在三个基座模型的 12 个数据集上持续优于 LoRA 及其变体。

**[Beyond Text Compression Tokenizers](model_compression/beyond_text_compression_tokenizers.md)**

:   本文系统评估了 6 种 tokenizer 在 350M 和 2.7B 参数模型上的影响，发现 tokenizer 选择对英文任务影响极小但对多语言任务（如机器翻译）有显著且跨尺度一致的影响，并提出了基于 Zipf 定律的新型内在评估指标，比文本压缩率能更好地预测多语言场景下的下游性能。

**[Bf16 Or Death Quantization Tradeoffs](model_compression/bf16_or_death_quantization_tradeoffs.md)**

:   这是迄今最全面的 LLM 量化实证研究，在 Llama-3.1 全系列（8B/70B/405B）上对 FP8/INT8/INT4 进行了超过 50 万次评估，发现 FP8 几乎无损、INT8 仅降 1-3%、INT4 出奇地有竞争力，并给出了不同部署场景的量化格式选择建议。

**[Blockpruner Fine-Grained Pruning For Large Language Models](model_compression/blockpruner_fine-grained_pruning_for_large_language_models.md)**

:   提出 BlockPruner，将 Transformer 层分解为 MHA 和 MLP 两个最小残差块，基于困惑度评估块重要性并通过迭代搜索进行细粒度剪枝，实现比层级剪枝更优的压缩效果。

**[Brainecho Semantic Brain Signal Decoding Through Vector-Quantized Spectrogram Re](model_compression/brainecho_semantic_brain_signal_decoding_through_vector-quantized_spectrogram_re.md)**

:   提出 BrainECHO 三阶段框架（自编码—对齐—微调），通过向量量化离散表示将脑信号映射到 Mel 频谱图空间，再借助 Whisper 完成非侵入式脑信号到文本的高质量解码。

**[Capture Key Cot Distillation](model_compression/capture_key_cot_distillation.md)**

:   提出 EDIT（mistakE-Driven key reasonIng step distillaTion），通过构造正确/错误配对的 dual CoTs 数据，利用最小编辑距离算法定位关键推理步骤，并以 token 级细粒度损失函数引导小模型聚焦学习这些关键步骤，而非简单模仿教师的推理形式。

**[Correcting Hallucinations In News Summaries Exploration Of Self-Correcting Llm M](model_compression/correcting_hallucinations_in_news_summaries_exploration_of_self-correcting_llm_m.md)**

:   系统性地探究了两种自纠正方法（CoVE 和 RARR）在新闻摘要幻觉纠正中的表现，比较了三种搜索引擎、多种检索设置和提示策略，发现 Bing 搜索片段 + RARR（few-shot）组合效果最佳，且 G-Eval 与人类评估高度一致。

**[Dac Prompt Compression](model_compression/dac_prompt_compression.md)**

:   DAC 提出动态注意力感知的 prompt 压缩方法，通过融合信息熵和注意力分数作为 token 重要性度量，并动态感知压缩过程中的熵偏移来进行细粒度压缩，在 LongBench 上比 SOTA 方法提升平均 1.33 分。

**[Data Laundering Artificially Boosting Benchmark Results Through Knowledge Distil](model_compression/data_laundering_artificially_boosting_benchmark_results_through_knowledge_distil.md)**

:   本文揭示了知识蒸馏可被滥用来人为提高基准测试分数的漏洞——通过"数据洗白"（Data Laundering）方法，将教师模型在测试集上学到的知识通过看似合法的中间训练步骤隐蔽地传递给学生模型，使一个2层BERT即可在GPQA上达到73.94%（接近OpenAI o1的77.30%），而该模型并未真正学会推理。

**[Deepsolution Boosting Complex Engineering Solution Design Via Tree-Based Explora](model_compression/deepsolution_boosting_complex_engineering_solution_design_via_tree-based_explora.md)**

:   提出面向复杂工程方案设计的新基准 SolutionBench 和新系统 SolutionRAG，通过树搜索探索+双视角思维（设计-审查交替）在 RAG 框架下逐步生成满足多约束的可靠工程方案，在 8 个工程领域达到 SOTA。

**[Denselora Dense Low-Rank Adaptation Of Large Language Models](model_compression/denselora_dense_low-rank_adaptation_of_large_language_models.md)**

:   本文提出DenseLoRA，通过引入跨层共享的Encoder-Decoder进行隐藏表示的压缩与重建，用一个稠密的小型低秩矩阵替代LoRA中两个冗余的低秩矩阵来进行适配，仅用0.01%可训练参数在LLaMA3-8B上达到83.8%准确率，超越了LoRA用0.70%参数达到的80.8%。

**[Direct Behavior Optimization Unlocking The Potential Of Lightweight Llms](model_compression/direct_behavior_optimization_unlocking_the_potential_of_lightweight_llms.md)**

:   提出 DeBoP 范式，将轻量级 LLM（LwLLM）的行为优化转化为对离散执行序列的优化，通过无梯度蒙特卡洛树搜索（MCTS）自动寻找最优 demonstration，使 LLaMA3-8B 在多数任务上超越 GPT-3.5 并减少约 60% 计算时间。

**[Drpruning Robust Pruning](model_compression/drpruning_robust_pruning.md)**

:   DRPruning 将分布稳健优化（DRO）引入 LLM 结构化剪枝，通过 scaling law 预测各领域最终 loss 作为参考、动态调整训练数据分布来平衡剪枝后各领域性能，在单语和多语设置下分别以 -5.59% PPL 和 +2.95% 下游任务的提升超越 Sheared LLaMA。

**[Eac Moe Expert Aware Compression](model_compression/eac_moe_expert_aware_compression.md)**

:   EAC-MoE 从 MoE 模型的专家选择特性出发，提出量化时校准路由器缓解 expert-shift 问题（QESC）+ 推理时基于专家选择频率动态剪枝不重要专家（PESF），在 Mixtral-8x7B 上实现 4.92× 内存压缩和 1.68× 推理加速且精度损失不到 1%。

**[Efficientqat](model_compression/efficientqat.md)**

:   EfficientQAT 提出两阶段 QAT 框架——先逐块训练所有参数（Block-AP）提供良好初始化，再端到端训练量化参数（E2E-QP）捕获跨块交互，在单张 A100 上 41 小时完成 Llama-2-70B 的 2-bit 量化，精度仅降 3 点。

**[Entropy-Based Exploration Conduction For Multi-Step Reasoning](model_compression/entropy-based_exploration_conduction_for_multi-step_reasoning.md)**

:   提出 Entro-duction 方法，通过监控 LLM 推理过程中输出的熵和方差熵变化来动态调整探索深度，使用 $\epsilon$-greedy 策略选择加深、扩展或停止三种探索行为，在避免冗余推理的同时提升推理准确率。

**[Explaining Puzzle Solutions In Natural Language An Exploratory Study On 6X6 Sudo](model_compression/explaining_puzzle_solutions_in_natural_language_an_exploratory_study_on_6x6_sudo.md)**

:   评估 5 个 LLM 在解决和解释 6×6 数独谜题上的能力，发现开源模型几乎无法正确求解（<1%），o1-preview 可解 65% 但其解释在正当性、清晰度和教育价值三个维度上均严重不足，揭示了 LLM 在多步推理解释方面的根本局限。

**[Fedex Lora Federated Exact Aggregation](model_compression/fedex_lora_federated_exact_aggregation.md)**

:   FedEx-LoRA 发现联邦学习中独立平均 LoRA 的 A 和 B 矩阵会导致不精确的全局更新（"乘积的均值≠均值的乘积"），通过在冻结权重矩阵中加入残差误差项实现精确聚合，在多个推理和 NLU 任务上一致优于 FedIT 和 FFA-LoRA。

**[Flipping Kd Small To Large](model_compression/flipping_kd_small_to_large.md)**

:   本文提出"反向知识蒸馏"范式——让 LLM 从微调过的小模型学习文本匹配的领域专家知识，通过将 decoder-only LLM 重新解释为 encoder-decoder 架构（用 LoRA 的压缩矩阵做 encoder）并设计 Margin-aware Contrastive Loss 来对齐表示相似度。

**[Gist Token Context Compression](model_compression/gist_token_context_compression.md)**

:   系统研究Gist Token上下文压缩方法，提出统一框架分类现有架构（记忆位置×粒度），发现Fine-grained KV Cache在RAG/QA上近无损但在合成召回上有明显缺陷，识别出三种失败模式（边界丢失/意外丢失/中途丢失），并提出细粒度自编码和分段token重要性估计两种改进策略。

**[Graph Counselor Multiagent Graphrag](model_compression/graph_counselor_multiagent_graphrag.md)**

:   Graph Counselor 提出了一个多智能体协作的 GraphRAG 推理框架，通过 Planning/Thought/Execution 三个 Agent 自适应提取图结构信息，并引入多视角自反思机制纠正推理偏差，在多个图推理任务上超越现有方法。

**[Iam Efficient Inference Through Attention Mapping Between Different-Scale Llms](model_compression/iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)**

:   发现不同规模 LLM 的注意力矩阵具有高度相似性，提出 IAM 框架——在 prefill 阶段建立小模型与大模型注意力头之间的余弦相似度映射，decode 阶段用小模型的注意力矩阵替代大模型映射层的注意力计算，实现 KV cache 减少 22% 和推理加速 11%，且与现有 KV cache 压缩方法正交。

**[L4Q Parameter Efficient Quantization Aware Finetuning](model_compression/l4q_parameter_efficient_quantization_aware_finetuning.md)**

:   提出 L4Q，将量化感知训练 (QAT) 与 LoRA 深度整合：先合并权重与LoRA参数再统一量化，通过定制反向传播路径消除权重梯度存储开销，实现联合优化量化与微调参数，在4-bit和3-bit量化下显著超越现有方法。

**[Lacuna Inc At Semeval-2025 Task 4 Lora-Enhanced Influence-Based Unlearning For L](model_compression/lacuna_inc_at_semeval-2025_task_4_lora-enhanced_influence-based_unlearning_for_l.md)**

:   提出 LIBU（LoRA 增强的影响函数遗忘算法），分两阶段实现 LLM 机器遗忘：Phase 1 用对角 Fisher 信息矩阵加权的影响函数更新参数精准遗忘，Phase 2 用 Sophia 二阶优化器稳定化训练，在 SemEval-2025 Task 4 的 OLMo-7B 上达到 0.283 遗忘率同时维持 0.469 MMLU 准确率。

**[Language Models Resist Alignment](model_compression/language_models_resist_alignment.md)**

:   本文从压缩理论视角揭示了LLM存在"弹性"(elasticity)现象——模型倾向于保持预训练分布而抵触对齐分布，且对齐后的模型在受到扰动时会以与数据量差距成反比的速率回弹到预训练状态，这解释了为什么对齐如此脆弱且容易被少量微调逆转。

**[Language Specific Features](model_compression/language_specific_features.md)**

:   利用 Sparse Autoencoders (SAEs) 分析多语言 LLM 的内部表示，发现存在强烈的语言特定 SAE features，这些 features 不仅与语言特有 token 相关还与语言上下文相关，消融它们只影响对应语言能力，且多个语言 features 之间存在协同效应；进一步利用这些 features 增强 steering vectors 实现对生成语言的精确控制。

**[Law Of Capacity Gap Distilling Language Models](model_compression/law_of_capacity_gap_distilling_language_models.md)**

:   揭示了语言模型蒸馏中的"容量差距定律"——最优教师模型的参数量与学生模型成线性关系（约 2.5 倍），将 LLM 蒸馏中的"不可能三角"转化为可解问题，并据此成功蒸馏出 3B 的 MiniMA 模型。

**[Limited-Resource Adapters Are Regularizers Not Linguists](model_compression/limited-resource_adapters_are_regularizers_not_linguists.md)**

:   本文将 adapter souping（权重平均）与交叉注意力微调结合用于低资源克里奥尔语机器翻译，发现虽然方法带来了显著提升（最高 +8 BLEU），但语言关联性与 adapter 性能无有意义的协变关系——随机初始化的未训练 adapter 表现同样优秀，表明 adapter 在此设定下的作用本质是**参数正则化而非语言信息迁移**。

**[Longred Mitigating Short-Text Degradation Of Long-Context Large Language Models ](model_compression/longred_mitigating_short-text_degradation_of_long-context_large_language_models_.md)**

:   本文系统分析了长上下文LLM在短文本任务上性能退化的两个原因（分布漂移和灾难性遗忘），并提出LongReD方法，通过短文本蒸馏和短到长蒸馏两个训练目标来最小化扩展模型与原始模型之间的分布差异，在保持长文本建模能力的同时将短文本性能保留至原始模型的99.4%。

**[Magnet Multi-Turn Tool-Use Data Synthesis And Distillation Via Graph Translation](model_compression/magnet_multi-turn_tool-use_data_synthesis_and_distillation_via_graph_translation.md)**

:   提出 Magnet 框架，基于函数依赖图的随机游走和节点操作（Insert/Merge/Split）构建高质量多轮 Function Calling 训练轨迹，结合基于提示的上下文蒸馏生成正负对比轨迹进行 SFT + mDPO 训练，使 14B 模型 Magnet-14B-mDPO 在 BFCL-v3 上达到 68.01（排名第 4），在多轮场景上大幅超越教师模型 Gemini-1.5-pro-002。

**[Moqae Mixed Precision Kv Cache](model_compression/moqae_mixed_precision_kv_cache.md)**

:   MoQAE 创造性地将不同量化比特宽度配置视为 MoE 中的"专家"，通过轻量路由器学习每个 chunk 的最优量化策略，结合路由冻结和路由共享机制，在几乎不损失精度的情况下大幅减少长上下文推理的 KV cache 内存。

**[Mplug Docowl2 Doc Compress](model_compression/mplug_docowl2_doc_compress.md)**

:   提出High-resolution DocCompressor模块，利用低分辨率全局视觉特征作为query通过交叉注意力将高分辨率文档图像压缩为仅324个token（不到同类方法的20%），在多页文档理解benchmark上达到SOTA且首token延迟降低50%+。

**[One QuantLLM for ALL: Fine-tuning Quantized LLMs Once for Efficient Deployments](model_compression/one_quantllm_for_all_fine-tuning_quantized_llms_once_for_efficient_deployments.md)**

**[Osrm Lora Merging Orthogonal](model_compression/osrm_lora_merging_orthogonal.md)**

:   OSRM 发现 LoRA 模型合并失败的根因是参数与数据分布的交互干扰（而非仅仅是参数冲突），提出在微调前通过数据协方差矩阵的特征分解来初始化 LoRA 矩阵 A，使其子空间与其他任务的数据分布正交，从而在合并时最小化跨任务干扰，在 8 个数据集、5 个模型上显著提升合并性能。

**[Outlier-Safe Pre-Training For Robust 4-Bit Quantization Of Large Language Models](model_compression/outlier-safe_pre-training_for_robust_4-bit_quantization_of_large_language_models.md)**

:   OSP（Outlier-Safe Pre-Training）框架通过三项创新——Muon 优化器（消除特权基方向）、Single-Scale RMSNorm（防止通道放大）和可学习嵌入投影层（重分布嵌入层激活），在预训练阶段主动防止异常值形成，训练的 1.4B 模型在 1T tokens 上实现近零超额峰度（0.04 vs 标准模型的 1818.56），在激进4-bit量化下平均分 35.7（Adam 为 26.5），仅 2% 训练开销。

**[Parameter-Efficient Fine-Tuning Via Circular Convolution](model_compression/parameter-efficient_fine-tuning_via_circular_convolution.md)**

:   提出Circular Convolution Adaptation (C3A)，用循环卷积算子替代LoRA的低秩矩阵分解来构造增量权重$\Delta W$——循环矩阵的秩与可训练参数数量解耦，实现"少参数+高秩"适配；同时利用FFT加速前向/反向传播，在计算和显存效率上均可与LoRA媲美。在GLUE、常识推理、数学推理、代码生成等任务上持续优于LoRA及其变体。

**[Pre-Training Distillation For Large Language Models A Design Space Exploration](model_compression/pre-training_distillation_for_large_language_models_a_design_space_exploration.md)**

:   系统性地探索大语言模型预训练蒸馏（Pre-training Distillation）的设计空间，从 logits 处理、损失函数选择、scaling law 和 offline/online logits 四个维度进行广泛实验，找到更优配置并得出有价值的结论。

**[Prompt Distill Teacher Student](model_compression/prompt_distill_teacher_student.md)**

:   提出候选标注+蒸馏范式（CanDist）——当 LLM 对样本不确定时输出所有可能标签（而非强制给唯一标签），然后用小语言模型（SLM）从候选标注中蒸馏出唯一标签，理论证明候选标注蒸馏比直接使用单标签有更好的理论保证，在六个文本分类任务上验证有效。

**[Ptq161 Low Bit Quantization](model_compression/ptq161_low_bit_quantization.md)**

:   首次将LLM权重真正量化到1.61-bit（此前号称sub-2bit的方法实际都超过2bit），通过一维结构化掩码（仅增加0.0002-bit/权重）保留显著通道、块级缩放因子优化和量化预处理三大创新，在LLaMA系列上以更低比特超越BiLLM和PB-LLM。

**[Quaff Quantized Peft](model_compression/quaff_quantized_peft.md)**

:   本文提出 Outlier Spatial Stability Hypothesis (OSSH)——微调期间激活异常通道的空间位置保持稳定——并基于此设计了 Quaff 框架，通过目标动量缩放仅处理少量不变的异常通道，实现 1.73× 延迟降低和 30% 内存节省，同时在 GPQA 上精度还提升了 0.6%。

**[Quantification Of Large Language Model Distillation](model_compression/quantification_of_large_language_model_distillation.md)**

:   本文提出了两种互补的LLM蒸馏量化方法——身份一致性评估（ICE）和响应相似性评估（RSE），通过越狱攻击挖掘模型身份信息泄露和多粒度响应相似性来衡量模型的蒸馏程度，发现大多数知名LLM（除Claude、Doubao和Gemini外）都表现出较高的蒸馏程度。

**[Revisiting Lora Through The Lens Of Parameter Redundancy Spectral Encoding Helps](model_compression/revisiting_lora_through_the_lens_of_parameter_redundancy_spectral_encoding_helps.md)**

:   本文系统研究了 LoRA 微调中的参数冗余问题，发现降低密度冗余不会损害表达能力（稀疏性质），并提出 SeLoRA——利用频谱变换（Fourier/Wavelet）从稀疏频谱子空间重参数化 LoRA 矩阵，以更少参数实现更优性能，且可即插即用地集成到多种 LoRA 变体中。

**[Rise Reasoning Enhancement Via Iterative Self-Exploration In Multi-Hop Question ](model_compression/rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)**

:   提出 RISE——结合 RAG 与自迭代训练的多跳问答框架，通过问题分解、检索阅读、自我批判三个动作的自我探索循环，迭代生成训练数据并多目标优化模型，在 2Wiki/HotpotQA/MuSiQue 上超越 GPT-3.5 和所有 8B 级基线。

**[Sci-Lora Mixture Of Scientific Loras For Cross-Domain Lay Paraphrasing](model_compression/sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)**

:   提出 Sci-LoRA——一种混合多领域 LoRA 的框架，通过对比学习训练文本编码器+动态权重生成器+LoRA 融合模块，在无需领域标签的情况下实现跨12个学科领域的科学文本通俗化改写，在5个数据集10个指标上超越 SOTA。

**[See Strategic Exploration Exploitation Prompt Optimization](model_compression/see_strategic_exploration_exploitation_prompt_optimization.md)**

:   本文提出 SEE 框架，首次将指令（instruction）和示例（examples）作为整体进行联合优化，采用元启发式优化原则设计四阶段探索-利用策略，在35个基准任务上实现平均13.94%的准确率提升并降低58.67%的计算成本。

**[Selection Bias Node Pruning](model_compression/selection_bias_node_pruning.md)**

:   提出Bias Node Pruning (BNP)和Auxiliary Option Injection (AOI)两种互补方法，从模型内部和输入端同时缓解LLM在多选题中的选择偏差，仅剪除0.002%权重即可将Llama-3准确率从52.3%提升至65.3%（+24.9%组合提升）。

**[Semantic Exploration Adaptive Gating](model_compression/semantic_exploration_adaptive_gating.md)**

:   针对 LLM 树搜索推理中"简单题也做复杂搜索"和"语义重复路径反复扩展"两大浪费问题，提出 SEAG 框架：先用 entropy 门控决定是否启动树搜索，再用语义聚类合并等价推理步骤，最终在准确率平均提升 4.3% 的同时仅需 RAP 31% 的推理开销。

**[State Offset Tuning Ssm Peft](model_compression/state_offset_tuning_ssm_peft.md)**

:   针对 SSM（如 Mamba）提出 State-offset Tuning，一种新的"状态基"PEFT 方法家族，通过在每个时间步直接注入可训练的状态偏移量 $h'$ 替代 Prefix-Tuning 的虚拟 token，解决了 prompt-based 方法在 SSM 上表达能力受限的问题，在更少参数量下持续优于 LoRA 和 Prefix-Tuning。

**[Stun Moe Pruning](model_compression/stun_moe_pruning.md)**

:   STUN 提出了结构化→非结构化的两阶段 MoE 剪枝方法，第一阶段用 O(1) 前向传播实现可扩展的专家级剪枝，第二阶段在剩余专家内做非结构化剪枝，在 480B 参数的 Snowflake Arctic 上以 40% 稀疏度几乎无性能损失。

**[Table Lora Structure Understanding](model_compression/table_lora_structure_understanding.md)**

:   TableLoRA 提出面向表格任务的专用 LoRA 模块，通过特殊 token 编码器改善表格序列化，并用 2D LoRA 编码单元格的行列位置信息，在参数高效微调设置下相比 vanilla LoRA 在 HiTab 上提升 5.9%，弥合了 LoRA 与全量微调之间 40.56% 的性能差距。

**[Tada Training-Free Recipe For Decoding With Adaptive Kv Cache Compression And Me](model_compression/tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)**

:   提出 TaDA——无需训练的 KV cache 压缩方法，通过对 K/V 激活做 head 维度均值中心化后量化偏差（而非原始激活），自动消除离群值问题，配合逐层自适应量化精度搜索，将 KV cache 压缩至原始 16 位的 27% 同时保持接近基线的精度。

**[Trans Peft Transferable](model_compression/trans_peft_transferable.md)**

:   Trans-PEFT 发现基座模型更新（如 Qwen2→Qwen2.5）主要改变 FFN 层的任务知识存储而较少影响 Attention 层的任务模式，据此提出层内知识掩码和跨层知识丢弃两种策略，使在旧版本上训练的 PEFT 模块可直接迁移到新版本而不需重新微调，性能提升可达 30%。

**[Uniicl Icl Framework](model_compression/uniicl_icl_framework.md)**

:   提出 UniICL 框架，用**一个冻结的 LLM** 同时完成 demonstration 压缩（compress→virtual tokens）、demonstration 选择（基于压缩后的 virtual token 相似度排序）和最终响应生成三个任务，仅需 17M 可训练参数（projection layer + learnable embedding），配合 Demonstration Bank 缓存机制避免重复压缩，实现 12× 压缩率下从 4-shot 扩展到 64-shot ICL（24GB 显存内），在多个 out-of-domain 数据集上超越 AutoCompressor、ICAE、LLMLingua 等基线。

**[Uniquanf Unified Quantization](model_compression/uniquanf_unified_quantization.md)**

:   UniQuanF 统一了均匀量化（UQ,表现力弱但优化性强）和二进制编码量化（BCQ,表现力强但优化性差）的优势，通过统一初始化、局部周期映射和统一定理，实现无额外部署开销的高精度 LLM 量化，在 GSM8K 上提升最高 4.60%。

**[Wanda Pruning Large Language Models Via Regional Gradients](model_compression/wanda_pruning_large_language_models_via_regional_gradients.md)**

:   提出 Wanda++——基于 decoder block 级别区域梯度的轻量级 LLM 剪枝框架，通过区域梯度评分（RGS）改进剪枝准则 + 区域优化（RO）最小化稠密/稀疏块输出差异，在 2:4 稀疏下 WikiText 困惑度较 Wanda 最高降低 32%，单 H100 GPU 10 分钟内完成 7B 模型剪枝。

**[Who Taught You That Tracing Teachers In Model Distillation](model_compression/who_taught_you_that_tracing_teachers_in_model_distillation.md)**

:   提出"教师模型归因"新问题：给定蒸馏后的学生模型，能否识别其教师模型？发现 n-gram 相似度和困惑度都不可靠，但词性（PoS）模板特征能有效捕捉教师模型在蒸馏中留下的语法"指纹"，在 5 个候选教师中达到 45-74% 的归因准确率（随机基线 20%）。

---

## ⚖️ 对齐/RLHF { #llm_alignment }

**[Agentalign Navigating Safety Alignment In The Shift From Informative To Agentic ](llm_alignment/agentalign_navigating_safety_alignment_in_the_shift_from_informative_to_agentic_.md)**

:   本文提出 AgentAlign 框架，利用抽象行为链作为中介，在模拟环境中合成高质量的 agent 安全对齐数据（有害+良性），通过 SFT 使三类开源模型的 agent 安全性提升35.8%-79.5%，同时保持甚至提升了任务能力。

**[Agentrm Enhancing Agent Generalization With Reward Modeling](llm_alignment/agentrm_enhancing_agent_generalization_with_reward_modeling.md)**

:   提出 AgentRM，一种可泛化的奖励模型，通过显式/隐式奖励建模和 LLM-as-a-judge 三种方式构建，配合 Best-of-N 采样和步级波束搜索引导策略模型进行测试时搜索，在 9 个 Agent 任务上平均提升 8.8 分，且展现出弱到强的泛化能力。

**[Aligned But Blind Implicit Bias](llm_alignment/aligned_but_blind_implicit_bias.md)**

:   发现 LLM 对齐训练的矛盾效应：对齐成功消除了显式偏见（Llama 3 70B 降至 8.13%），但反而放大了隐式偏见（从 64.1% 升至 91.4%），机制是对齐使模型在歧义上下文中不再表征种族概念（"种族盲视"），导致安全护栏无法在隐性场景中激活。通过在早期层注入种族感知激活可将隐式偏见从 97.3% 降至 71.2%。

**[Amopo Adaptive Multi-Objective Preference Optimization Without Reward Models And](llm_alignment/amopo_adaptive_multi-objective_preference_optimization_without_reward_models_and.md)**

:   提出AMoPO框架，通过将生成空间建模为高斯分布实现维度感知的自适应权重分配，在不依赖奖励模型和参考模型的情况下完成多目标偏好对齐，在HelpSteer2数据集上超越SOTA 28.5%，并在7B/14B/32B模型上验证了缩放能力。

**[Aspo Adaptive Sentence-Level Preference Optimization For Fine-Grained Multimodal](llm_alignment/aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)**

:   提出 ASPO（Adaptive Sentence-level Preference Optimization）——将 DPO 的偏好单元从"整个回复"细化到"每个句子"，为每个句子动态计算自适应奖励值（基于模型自身预测评估正确性和重要性），在多模态推理任务上显著优于传统回复级 DPO，有效减少幻觉并提升细粒度推理能力。

**[Atyaephyra At Semeval-2025 Task 4 Low-Rank Negative Preference Optimization](llm_alignment/atyaephyra_at_semeval-2025_task_4_low-rank_negative_preference_optimization.md)**

:   在 SemEval 2025 LLM 遗忘共享任务中，将负偏好优化 (NPO) 与低秩适配 (LoRA) 结合，利用 LoRA 的结构特性零开销获取原始模型分布来计算 KL 散度正则化，显著稳定了遗忘过程并超越了任务基线。

**[Automixalign Adaptive Data Mixing](llm_alignment/automixalign_adaptive_data_mixing.md)**

:   AutoMixAlign 提出了一种理论驱动的多任务偏好优化数据混合方法：先训练各任务的 specialist model 确定最优 loss 基线，再通过 minimax 优化自适应调整数据混合比例，优先处理 excess loss（与 specialist 的差距）最大的任务，在 helpfulness/harmlessness/reasoning 多任务 DPO 中平均提升 9.42%。

**[Beyond Surface-Level Patterns An Essence-Driven Defense Framework Against Jailbr](llm_alignment/beyond_surface-level_patterns_an_essence-driven_defense_framework_against_jailbr.md)**

:   提出 EDDF，一种基于"攻击本质"而非表面模式的越狱防御框架：离线提取已知攻击的本质策略存入向量数据库，在线时对新查询做本质抽象+检索+细粒度判断，将攻击成功率降低至少 20% 且误报率仅 2.18%。

**[Beyond The Tip Of Efficiency Uncovering The Submerged Threats Of Jailbreak Attac](llm_alignment/beyond_the_tip_of_efficiency_uncovering_the_submerged_threats_of_jailbreak_attac.md)**

:   系统评估 13 个 SOTA 小语言模型（<4B参数）在 5 种越狱攻击下的安全性，发现 SLM 虽能抵御直接攻击但在越狱攻击下显著比大模型脆弱，进一步分析了架构压缩、量化和知识蒸馏等 SLM 技术对安全性的影响。

**[Boosting Vulnerability Detection Of Llms Via Curriculum Preference Optimization ](llm_alignment/boosting_vulnerability_detection_of_llms_via_curriculum_preference_optimization_.md)**

:   提出 ReVD 框架，通过双向漏洞推理数据合成 + 三元组 SFT（同时学习漏洞代码/修复代码/代码差异的推理）+ 课程化在线偏好优化（COPO），将 LLM 的漏洞检测准确率提升 12-23%，在 PrimeVul 和 SVEN 上达到 SOTA。

**[Breaking The Ceiling Exploring The Potential Of Jailbreak Attacks Through Expand](llm_alignment/breaking_the_ceiling_exploring_the_potential_of_jailbreak_attacks_through_expand.md)**

:   基于精细化可能性模型 (ELM) 将越狱策略分解为四类可独立进化的组件（角色/内容支撑/语境/沟通技巧），提出 CL-GSO 遗传算法在组件级进行交叉与变异，将策略空间从既有方法的 40 种扩展到 839 种，在 Claude-3.5 上实现 96% 攻击成功率（此前方法最高仅 4%），同时提出基于意图一致性的评估机制，准确率达 96.5% 超越专用安全模型。

**[Call for Rigor in Reporting Quality of Instruction Tuning Data](llm_alignment/call_for_rigor_in_reporting_quality_of_instruction_tuning_data.md)**

**[Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step](llm_alignment/chain-of-jailbreak_attack_for_image_generation_models_via_ed.md)**

:   提出 CoJ（Chain-of-Jailbreak）攻击，将生成恶意图像的单步请求分解为多步编辑指令链（从无害种子图像逐步编辑到目标），绕过图像生成模型的安全过滤器，在 GPT-4o 等模型上达到高攻击成功率。

**[Chain-Of-Jailbreak Attack For Image Generation Models Via Editing Step By Step](llm_alignment/chain-of-jailbreak_attack_for_image_generation_models_via_editing_step_by_step.md)**

:   提出 Chain-of-Jailbreak（CoJ）攻击，将无法直接绕过安全护栏的恶意 query 分解为多步编辑子 query（删然后插、插然后删、改然后改回），在 GPT-4V/4o/Gemini 上达到 60%+ 越狱成功率；同时提出 Think-Twice Prompting 防御，拦截 95%+ 的 CoJ 攻击。

**[Cheems Chinese Reward Models](llm_alignment/cheems_chinese_reward_models.md)**

:   为弥补中文 Reward Model 资源的空白，本文构建了 CheemsBench（首个大规模中文 RM 评测基准）和 CheemsPreference（首个大规模中文偏好数据集），通过人机协作标注 + 远程监督过滤策略训练的 CheemsRM 在中文场景显著超越现有所有开源 RM。

**[Curiosity Driven Rlhf](llm_alignment/curiosity_driven_rlhf.md)**

:   CD-RLHF 将好奇心驱动探索（curiosity-driven RL）引入 RLHF，通过前向动力学模型的预测误差作为内在奖励，结合 top-k 门控过滤与 reward whitening，在不损失对齐质量的前提下大幅提升 LLM 输出多样性（Llama-3.2-1B 上 Diversity 提升 40.26%，EAD 提升 8.92%）。

**[Debate Reflect And Distill Multi-Agent Feedback With Tree-Structured Preference ](llm_alignment/debate_reflect_and_distill_multi-agent_feedback_with_tree-structured_preference_.md)**

:   提出 D&R 框架，让小模型（student）与多个大模型（teacher）进行多轮辩论并收集自我反思和教师反馈，然后将辩论日志组织为偏好树做 Tree-structured DPO (T-DPO) 蒸馏，在 MMLU Pro 和 MATH 上平均提升 14.18 分，且推理效率优于基线。

**[Diffpo Diffusion Alignment](llm_alignment/diffpo_diffusion_alignment.md)**

:   提出 DiffPO，将 LLM 对齐重新建模为句子级扩散去噪过程，通过 parallel decoding 实现高效推理时对齐，作为即插即用模块可增强任意底座模型的对齐质量。

**[Dynamic Scaling Of Unit Tests For Code Reward Modeling](llm_alignment/dynamic_scaling_of_unit_tests_for_code_reward_modeling.md)**

:   本文发现扩展LLM生成的单元测试数量可以持续提升代码奖励信号质量（尤其对困难问题效果更好），据此训练了轻量级单元测试生成模型CodeRM-8B并实现动态缩放策略，在多个代码生成基准上取得显著提升。

**[Expectation Confirmation Preference Optimization For Multi-Turn Conversational R](llm_alignment/expectation_confirmation_preference_optimization_for_multi-turn_conversational_r.md)**

:   提出 ECPO（Expectation Confirmation Preference Optimization），首个面向 LLM 对话推荐 Agent 的多轮偏好优化方法——基于心理学期望确认理论（ECT）显式建模用户满意度在多轮对话中的演变，通过前向期望确认定位不满意根因 + 后向期望推导重写回复构建 turn-level 偏好对，配合 AILO 用户模拟器，在 3 个数据集上显著优于现有 MTPO 方法。

**[Federated Data-Efficient Instruction Tuning For Large Language Models](llm_alignment/federated_data-efficient_instruction_tuning_for_large_language_models.md)**

:   提出 FedHDS（Federated Hierarchical Data Selection），通过 intra-client 和 inter-client 两级层次化数据选择消除联邦学习中客户端内部和跨客户端的数据冗余，结合多层 Transformer 特征融合提升 coreset 质量；仅用不到 1.5% 的数据，在 Rouge-L 上相对 SOTA 全数据联邦基线平均提升 10.72%，训练效率提升最高达 48.8 倍。

**[Finding The Sweet Spot Preference Data Construction For Scaling Preference Optim](llm_alignment/finding_the_sweet_spot_preference_data_construction_for_scaling_preference_optim.md)**

:   发现传统的 DPO 偏好数据构建策略（max-min）在增加采样量时性能反而下降，通过基于奖励分布的系统性探索发现 rejected 响应应选在 μ−2σ 而非最小值，据此提出了一种随采样量增加而持续提升的偏好数据构建方法。

**[Fine-Grained Video Dubbing Duration Alignment With Segment Supervised Preference](llm_alignment/fine-grained_video_dubbing_duration_alignment_with_segment_supervised_preference.md)**

:   提出 Segment Supervised Preference Optimization (SSPO)，将视频配音中译文与源语音的时长对齐问题建模为段级偏好优化，通过逐句采样+细粒度 DPO 损失实现每行对话的时长一致性，同时维持翻译质量和输出格式。

**[Focused-Dpo Enhancing Code Generation Through Focused Preference Optimization On](llm_alignment/focused-dpo_enhancing_code_generation_through_focused_preference_optimization_on.md)**

:   发现代码生成错误集中在特定"错误易发点"（error-prone points）——前缀/后缀通常正确，错误集中在中间代码段，提出 Focused-DPO 通过 PageRank 排序定位关键代码段并在 DPO 损失中加权放大，HumanEval+ 提升 4.41%、MBPP+ 提升 6.71%。

**[Haf-Rm A Hybrid Alignment Framework For Reward Model Training](llm_alignment/haf-rm_a_hybrid_alignment_framework_for_reward_model_training.md)**

:   提出混合对齐框架 HaF-RM，在奖励模型训练中保留策略层（policy layer），通过同时优化序列级奖励损失和 token 级策略损失来共同监督共享的内部偏好模型，在 5 个数据集上一致性超越标准 Baseline 和 DPO 方法。

**[Hiddendetect Detecting Jailbreak Attacks Against Multimodal Large Language Model](llm_alignment/hiddendetect_detecting_jailbreak_attacks_against_multimodal_large_language_model.md)**

:   提出 HiddenDetect，一种免训练（tuning-free）的基于内部激活状态的安全检测框架：通过监控 LVLM 推理时隐藏状态中的拒绝语义信号来检测越狱攻击，在多个模型和多模态基准上 AUROC 大幅超越现有方法。

**[Influence Functions Rlhf](llm_alignment/influence_functions_rlhf.md)**

:   首次将影响函数应用于 RLHF 奖励模型的反馈数据审计，结合 OPORP 向量压缩实现 2.5 倍加速，在偏差检测上超越 GPT-4o（AUC 0.8 vs 0.747），并从 Anthropic-HH 数据集中发现 47% 的错标样本。

**[Internal Value Alignment In Large Language Models Through Controlled Value Vecto](llm_alignment/internal_value_alignment_in_large_language_models_through_controlled_value_vecto.md)**

:   提出 ConVA（Controlled Value Vector Activation）框架，通过上下文控制的数据集精准识别 LLM 隐空间中的价值向量，并用门控最小扰动机制在推理时激活目标价值，在 Schwartz 10 种基本价值上实现平均 29.6% 的控制成功率提升，同时保持 97%+ 的文本流畅度和通用能力。

**[Iopo Input Output Preference](llm_alignment/iopo_input_output_preference.md)**

:   提出 IOPO（Input-Output Preference Optimization），在传统 DPO 仅优化输出偏好的基础上，引入输入偏好建模——让模型学习"给定回复 y，哪个指令 x 更匹配"，从而增强对复杂多约束指令的细粒度感知能力；同时构建了包含 120K 训练数据、1K 评测数据、覆盖 5 大类 26 个约束维度的 Trace 基准。

**[Jailbreaking One Step Is Enough](llm_alignment/jailbreaking_one_step_is_enough.md)**

:   本文提出REDA（Reverse Embedded Defense Attack）方法，将攻击意图伪装为"防御"有害内容的任务，通过反转攻击视角+ICL示例引导+请求意图削弱，实现一步生成、跨模型通用的高成功率越狱攻击。

**[Jailbreakradar Comprehensive Assessment Jailbreak Attacks](llm_alignment/jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)**

:   本文提出了一个全面的越狱攻击评估框架 JailbreakRadar，收集了17种代表性越狱攻击方法，建立了六类攻击分类体系，并在9个对齐LLM上进行了大规模系统性评测，揭示了不同类型攻击在实用性和防御鲁棒性上的关键差异。

**[Jsontuning Towards Generalizable Robust And Controllable Instruction Tuning](llm_alignment/jsontuning_towards_generalizable_robust_and_controllable_instruction_tuning.md)**

:   提出 JsonTuning——将指令微调的输入输出从自然语言文本替换为 JSON 结构化格式，通过显式表示任务元素、关系和输出约束（JSON Schema），在 7 个预训练模型和 6 类任务上一致超越传统 TextTuning，平均性能从 26.78 提升到 30.88，同时显著增强鲁棒性和可控性。

**[Kpo Protein Safety](llm_alignment/kpo_protein_safety.md)**

:   提出知识引导偏好优化（KPO）框架，通过蛋白质安全知识图谱识别安全/危险序列作为偏好信号，用强化学习训练蛋白质语言模型减少有害蛋白质序列的生成概率，同时保持功能性——为蛋白质生成的生物安全提供保障框架。

**[Llms Caught In The Crossfire Malware Requests And Jailbreak Challenges](llm_alignment/llms_caught_in_the_crossfire_malware_requests_and_jailbreak_challenges.md)**

:   构建 MalwareBench 基准（320 个手工恶意代码需求 × 11 种黑盒越狱方法 = 3520 个 prompt），系统评测 29 个 LLM 在恶意代码生成场景下的安全性，发现越狱攻击将平均拒绝率从 60.93% 降至 39.92%，且模型参数量与防御能力并非正比关系。

**[Lssf Safety Subspace](llm_alignment/lssf_safety_subspace.md)**

:   LSSF 提出 LLM 的安全信息存在于低秩子空间中的假设，通过 SVD 提取安全对齐模型的主成分，利用安全奇异值熵自适应确定每层的保留秩，最终将提取的安全主成分线性融合到微调后的模型中，无需额外训练即可恢复因微调而退化的安全对齐，同时保持下游任务性能。

**[M2S Multiturn To Singleturn Jailbreak In](llm_alignment/m2s_multiturn_to_singleturn_jailbreak_in.md)**

:   提出 M2S 框架，通过三种简单的格式转换方法（Hyphenize/Numberize/Pythonize）将多轮人类越狱对话压缩为单轮 prompt，不仅保持甚至超越原始多轮攻击效果（ASR 高达 95.9%，比多轮提升最多 17.5%），同时 token 使用量减半以上。

**[Measuring Data Diversity For Instruction Tuning A Systematic Analysis And A Reli](llm_alignment/measuring_data_diversity_for_instruction_tuning_a_systematic_analysis_and_a_reli.md)**

:   系统分析 11 种现有多样性度量方法的局限性，提出 NovelSum——一种同时考虑样本间差异和信息密度的数据多样性指标，与指令微调性能达到 0.97 相关性。

**[Mpo Multilingual Safety Alignment](llm_alignment/mpo_multilingual_safety_alignment.md)**

:   MPO 发现 LLM 在主导语言（英文）和目标语言间的隐式 Reward Gap 与安全性能强相关，提出直接最小化两者 Reward Gap 差异来将主导语言的安全对齐能力迁移到多语言，在三个模型上显著降低了低资源语言的攻击成功率且不损害通用能力。

**[Mtsa Multi-Turn Safety Alignment For Llms Through Multi-Round Red-Teaming](llm_alignment/mtsa_multi-turn_safety_alignment_for_llms_through_multi-round_red-teaming.md)**

:   提出MTSA框架，通过思维引导的多轮红队攻击学习和基于未来奖励的多轮强化学习算法，在对抗迭代优化中同时提升红队模型的攻击能力和目标模型的安全防御能力，在多个安全基准上达到SOTA，且不损失模型通用性能。

**[Mutual Taught Policy Reward](llm_alignment/mutual_taught_policy_reward.md)**

:   Mutual-Taught 提出了一种基于 EM 算法的自训练框架，在偏好优化过程中同时迭代更新 policy model 和 reward model：E-step 用当前 RM 优化 PM，M-step 用 PM 更新前后的输出差异构建伪偏好对来更新 RM，解决了分布偏移导致的 reward hacking 问题，8B 模型在 AlpacaEval-2 达到 54.1% LC win rate。

**[Otpo Token Weighting](llm_alignment/otpo_token_weighting.md)**

:   OTPO 利用无平衡最优传输（UOT）在 chosen/rejected 回复的 token 表示之间计算语义对齐权重，使偏好优化聚焦于关键差异 token 而非均等对待所有 token，在 AlpacaEval2 上将 DPO 的 LC WR 从 48.14% 提升至 55.84%，并将 DPO/SimPO/SamPO/LDDPO 统一为 token 加权的特例。

**[Personalized Preference Opt](llm_alignment/personalized_preference_opt.md)**

:   提出"溯因推理"视角的偏好个性化方法：先用 LLM 推断偏好 chosen/rejected 回答背后的用户画像（Persona Inference），再用画像增强的偏好数据训练模型（Persona Tailoring），显著提升模型对不同用户需求的个性化适配能力。

**[Pig Privacy Jailbreak](llm_alignment/pig_privacy_jailbreak.md)**

:   提出 PIG 框架，通过识别隐私查询中的 PII 实体类型、构建隐私上下文示例、并利用三种基于梯度的迭代优化策略更新上下文，实现对 LLM 的高效隐私越狱攻击，在白盒和黑盒模型上均达到 SOTA。

**[Pku-Saferlhf Towards Multi-Level Safety Alignment For Llms With Human Preference](llm_alignment/pku-saferlhf_towards_multi-level_safety_alignment_for_llms_with_human_preference.md)**

:   发布 PKU-SafeRLHF 大规模安全偏好数据集，包含 44.6k 精炼 prompt、265k 带安全元标签的 QA 对和 166.8k 偏好数据，首次引入 19 种危害类别和 3 级严重程度标注，并训练了严重程度敏感的审核模型（93% 准确率）和基于该数据的 SafeRLHF 对齐 pipeline。

**[Probability-Consistent Preference Optimization For Enhanced Llm Reasoning](llm_alignment/probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)**

:   提出 PCPO（Probability-Consistent Preference Optimization），在偏好对选择时同时考虑答案正确性和 token 级概率一致性（用 Levenshtein 距离过滤+概率一致性评分），并在 DPO 损失中按一致性加权，在 GSM8K/MATH-500/Olympiadbench 上一致超越标准 DPO 和 ScPO。

**[Queryattack Jailbreaking Aligned Large Language Models Using Structured Non-Natu](llm_alignment/queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)**

:   提出 QueryAttack，将恶意自然语言查询分解为三个语义组件（内容、修饰符、类别）并填入编程语言模板（SQL/URL/Python/Java/C++ 等 9 种），结合 ICL 引导目标 LLM 直接用自然语言回复有害内容，无需解密步骤，在 GPT-4o 上 Ensemble 配置达到 96.35% ASR，且提出的跨语言 CoT 防御可将 ASR 降低最多 64%。

**[Red Queen Safeguarding Large Language Models Against Concealed Multi-Turn Jailbr](llm_alignment/red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)**

:   提出 Red Queen Attack——首个基于 Theory of Mind（ToM）构建多轮对话场景并隐藏恶意意图的越狱攻击方法，生成 56K 多轮隐蔽攻击数据，在 GPT-4o 上达到 87.6% ASR；同时提出 Red Queen Guard 防御策略，通过多轮 DPO 数据训练将 ASR 降至 <1%，同时不影响通用基准性能。

**[Rethinking Table Instruction Tuning](llm_alignment/rethinking_table_instruction_tuning.md)**

:   系统消融表格指令微调中被忽视的超参数选择（学习率、数据量、epoch），揭示现有表格 LLM 因学习率过大（2e-5）导致通用能力严重退化（MMLU 降 14 分、AI2ARC 降 21 分），提出仅需 13 个数据集各 200 条（共 2600 条）+ 学习率 1e-6 + 2 epoch 微调 LLaMA 3.1 8B Instruct 即可构建 TAMA，在 13 个表格任务上匹配/超越 GPT-3.5 和 GPT-4，同时完整保持通用能力。

**[Reverse Preference Optimization For Complex Instruction Following](llm_alignment/reverse_preference_optimization_for_complex_instruction_following.md)**

:   提出反向偏好优化（RPO），通过动态反转指令中未满足的约束将任意回复转化为"完美"chosen 样本，消除多约束偏好对中的噪声，在多轮复杂指令遵循任务上显著超越 DPO 基线。

**[Reward Fairness Rlhf](llm_alignment/reward_fairness_rlhf.md)**

:   将 RLHF 中的各种奖励偏差（长度偏差、类别偏差、社会偏差）统一定义为"奖励不公平"问题，从资源分配视角提出两种偏差无关的缓解方法——公平正则化和公平系数——在不针对特定偏差设计的情况下有效缓解多种偏差，实现更公平的人类偏好对齐。

**[Reward Generalization In Rlhf A Topological Perspective](llm_alignment/reward_generalization_in_rlhf_a_topological_perspective.md)**

:   从信息拓扑的角度系统刻画 RLHF 中 reward 信息的流动——宏观层面将 RLHF 建模为自编码过程，微观层面提出 Induced Bayesian Network (IBN) 分析偏好数据拓扑对 reward 泛化的影响，进而提出树结构偏好数据方法，在 HH-RLHF/GSM-8K/DialogSum 三个任务上平均 65% win rate 超越链式 baseline。

**[Rewrite To Jailbreak Discover Learnable And Transferable Implicit Harmfulness In](llm_alignment/rewrite_to_jailbreak_discover_learnable_and_transferable_implicit_harmfulness_in.md)**

:   提出 R2J（Rewrite to Jailbreak），一种可学习、可迁移的黑盒越狱方法——通过迭代训练 attacker LLM 学习改写有害指令（仅改措辞不改意图），相比 GCG/AutoDAN 等方法攻击成功率提高 20%+，且无额外前缀/后缀，更隐蔽且跨模型可迁移。

**[Rise Error Inject Preference](llm_alignment/rise_error_inject_preference.md)**

:   RISE 发现 LLM 约 75% 的数学错误是微妙的步内错误（数字替换、操作数交换、步骤遗漏），通过让 LLM 自编辑向正确解注入预定义微妙错误来构造高质量难负样本，配合错误感知 DPO 训练，仅用 4.5K 样本在 GSM8K 提升 3.0%、MATH 提升 7.9%，并泛化到逻辑推理和代码生成。

**[Rpo Retrieval Preference Optimization For Robust Retrieval-Augmented Generation](llm_alignment/rpo_retrieval_preference_optimization_for_robust_retrieval-augmented_generation.md)**

:   提出 Retrieval Preference Optimization (RPO)，一种专为 RAG 设计的轻量级偏好对齐方法，通过将检索质量评估隐式地集成到生成过程中，使 LLM 能够自适应地在参数知识和检索知识之间做出选择，无需额外组件即可缓解知识冲突导致的幻觉问题。

**[Sea Lowresource Safety Alignment For Multimodal](llm_alignment/sea_lowresource_safety_alignment_for_multimodal.md)**

:   提出 SEA 框架，通过梯度优化生成合成模态 embedding（不需要真实图像/视频/音频），仅用文本安全数据就能实现多模态 LLM 的安全对齐，在单张 RTX3090 上 24 秒即可合成高质量 embedding，同时发布了视频和音频安全基准 VA-SafetyBench。

**[Synthesizeme Persona Prompts](llm_alignment/synthesizeme_persona_prompts.md)**

:   提出SynthesizeMe，通过bootstrap推理→合成用户画像→筛选信息性示例三步流程，无需微调即可为个性化奖励模型构建有效prompt，在Chatbot Arena上提升LLM-as-a-Judge个性化准确率4.4%。

**[Tabledreamer Progressive And Weakness-Guided Data Synthesis From Scratch For Tab](llm_alignment/tabledreamer_progressive_and_weakness-guided_data_synthesis_from_scratch_for_tab.md)**

:   提出 TableDreamer 两阶段数据合成框架：第一阶段从零合成多样化表格及种子指令数据，第二阶段通过弱点引导的迭代输入空间探索（在三个方向上演化数据，并用 LLM-as-Judge 筛选模型表现差的样本作为下一轮种子），仅用 27K GPT-4o 合成数据即将 Llama3.1-8B 的平均准确率提升 11.62%，超越使用 80K-100K 数据的所有基线方法。

**[Teaching An Old Llm Secure Coding](llm_alignment/teaching_an_old_llm_secure_coding.md)**

:   提出 DiSCo（从前沿 LLM 蒸馏的安全代码偏好数据集，10K 实例覆盖 431 种 CWE）和 LPO（局部偏好优化算法，仅在安全相关 token 上传播损失），在四个安全编码基准上减少 19-40% 的安全问题，同时提升 3-10% 的代码质量。

**[Think Cite Attributed Text Gen](llm_alignment/think_cite_attributed_text_gen.md)**

:   将归因文本生成（带引用的文本生成）建模为多步推理问题，提出自引导蒙特卡洛树搜索（SG-MCTS）结合进度奖励建模（PRM），通过多路径搜索+中间状态反思+生成/归因双维度进度奖励，在 ALCE 基准三个数据集上显著超越所有基线。

**[Tmcht Contagious Jailbreak Multiagent](llm_alignment/tmcht_contagious_jailbreak_multiagent.md)**

:   提出 TMCHT 多智能体多拓扑攻击评估框架和 ARCJ（对抗性复制传染越狱）方法，解决了多智能体系统中单智能体攻击方法面临的"毒性消散"问题——通过优化检索后缀确保中毒样本易被检索，优化复制后缀使中毒信息具有传染性自我复制能力，在 line/star 拓扑和 100 智能体系统中分别提升 23.51%/18.95%/52.93% 的攻击成功率。

**[Upcycling Instruction Tuning From Dense To Mixture-Of-Experts Via Parameter Merg](llm_alignment/upcycling_instruction_tuning_from_dense_to_mixture-of-experts_via_parameter_merg.md)**

:   本文提出UpIT (Upcycling Instruction Tuning)，利用密集模型指令微调过程中的中间checkpoint作为专业化专家，通过遗传算法扩展专家数量和路由预优化，实现数据高效且灵活的dense-to-MoE转换。

---

## 🦾 LLM Agent { #llm_agent }

**[A Multi-Agent Framework For Mitigating Dialect Biases In Privacy Policy Question](llm_agent/a_multi-agent_framework_for_mitigating_dialect_biases_in_privacy_policy_question.md)**

:   提出多 Agent 协作框架缓解隐私政策 QA 中的方言偏差——Dialect Agent 将方言查询翻译为标准美式英语（SAE）并验证意图保留，Privacy Policy Agent 利用领域专长生成答案，两者迭代协商至达成一致。在 PrivacyQA 和 PolicyQA 上将 GPT-4o-mini 零样本准确率从 0.394 提升至 0.601，方言间最大 F1 差距降低 82%。

**[Agentic Knowledgeable Self-Awareness](llm_agent/agentic_knowledgeable_self-awareness.md)**

:   本文提出 KnowSelf，一种数据驱动方法，通过在 agent 自探索轨迹上标注特殊 token 来标识不同思维情境（快速思考/慢速思考/知识思考），经两阶段训练（SFT + RPO）使 agent 模型学会自主判断何时需要调用外部知识，以最小知识消耗代价达到最优规划效果。

**[Agentic Reasoning Tools](llm_agent/agentic_reasoning_tools.md)**

:   Agentic Reasoning 提出了一个将 Web 搜索、代码执行和知识图谱记忆（Mind-Map）三种 Agent 工具集成到 LLM 推理过程中的框架，在 DeepSeek-R1 上将 Humanity's Last Exam 准确率从 9.4% 提升到 23.8%（+14.4%），GPQA 从 71.5% 到 81.2%，接近 OpenAI Deep Research 水平。

**[Androidgen Agent Data Scarcity](llm_agent/androidgen_agent_data_scarcity.md)**

:   提出AndroidGen——一个在数据稀缺条件下增强Android Agent能力的框架（ExpSearch+ReflectPlan+AutoCheck+StepCritic四模块），在AndroidWorld上以GPT-4o达到46.8%成功率（vs M3A的27.7%），并能自动生成高质量轨迹数据训练开源模型达到竞争力水平。

**[Bel Esprit Multi-Agent Framework For Building Ai Model Pipelines](llm_agent/bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)**

:   提出 Bel Esprit——基于多 Agent 框架的对话式 AI 管线构建系统，通过子 Agent 协作（需求澄清→管线构建→验证→模型填充）将用户的模糊需求转化为由多个 AI 模型组成的可执行管线（如多语言视频配音→语音识别+翻译×3+TTS×3），在人工策划和合成数据上验证有效性。

**[Beyond Numeric Rewards In-Context Dueling Bandits With Llm Agents](llm_agent/beyond_numeric_rewards_in-context_dueling_bandits_with_llm_agents.md)**

:   系统评估了 LLM 在 Dueling Bandits（偏好反馈强化学习）中的零样本上下文决策能力，发现 GPT-4 Turbo 在弱遗憾（weak regret）上表现出色但强遗憾（strong regret）存在差距，进而提出 LEAD 框架（LLM with Enhanced Algorithmic Dueling），通过将经典 DB 算法与 LLM 智能体细粒度自适应融合来同时获得理论保证和鲁棒性。

**[Caution Environment Gui Agent Distractions](llm_agent/caution_environment_gui_agent_distractions.md)**

:   本文首次系统研究了多模态 GUI Agent 对环境干扰（弹窗广告、推荐内容等）的脆弱性，在无恶意攻击的自然场景下，即使最强的 MLLM（包括GPT-4o）也有 20-40% 的概率被环境中的无关内容分散注意力而执行偏离用户目标的操作。

**[Dpt Agent Dual Process](llm_agent/dpt_agent_dual_process.md)**

:   提出 DPT-Agent，首个将双过程理论（Dual Process Theory）系统化地融入语言智能体框架的方法——用有限状态机(FSM)+code-as-policy 作为快速直觉的 System 1，用心智理论(ToM)+异步反思的 LLM 作为慢速深思的 System 2，首次实现了自主的实时同步人机协作（在 Overcooked 困难版中）。

**[Emulate A Multi-Agent Framework For Determining The Veracity Of Atomic Claims By](llm_agent/emulate_a_multi-agent_framework_for_determining_the_veracity_of_atomic_claims_by.md)**

:   提出 EMULATE 多智能体事实核查框架，通过 7 个专职 LLM agent 模拟人类验证声明的完整行为链（搜索→排序→内容评估→证据充分性判断→分类），在三个事实核查 benchmark 上的 Macro-F1 和 Weighted-F1 均超越现有方法。

**[Explorer Scaling Exploration-Driven Web Trajectory Synthesis For Multimodal Web ](llm_agent/explorer_scaling_exploration-driven_web_trajectory_synthesis_for_multimodal_web_.md)**

:   提出 Explorer——一个可扩展的多智能体 pipeline，通过自主网页探索和逐步精炼来合成大规模多模态 web 轨迹数据集（94K 成功轨迹，49K+ URL，720K 截图），训练的 Explorer-7B 在 Mind2Web-Live、MiniWob++ 等 benchmark 上达到甚至超过 GPT-4 水平。

**[Fact Audit Factcheck](llm_agent/fact_audit_factcheck.md)**

:   提出FACT-AUDIT——一个基于重要性采样和多智能体协作的自适应动态事实核查评估框架，通过动态生成测试数据、迭代探测模型弱点、并同时评估verdict预测和justification质量，全面审计LLM的事实核查能力边界。

**[Gui Explorer Autonomous](llm_agent/gui_explorer_autonomous.md)**

:   提出 GUI-explorer，一个无需训练的 GUI agent，通过自主探索收集功能感知的交互轨迹，并以无监督方式从状态转换三元组中挖掘 transition-aware 知识，在 SPA-Bench 和 AndroidWorld 上分别达到 53.7% 和 47.4% 的任务成功率。

**[Guicourse From General Vision Language Model To Versatile Gui Agent](llm_agent/guicourse_from_general_vision_language_model_to_versatile_gui_agent.md)**

:   本文提出GUICourse——一套用于从通用视觉语言模型（VLM）训练多功能GUI代理的数据集系列（GUIEnv/GUIAct/GUIChat），通过两阶段训练流程先增强OCR和定位能力、再注入GUI知识，使得3.1B参数的小模型也能在网页和手机GUI导航任务上取得有效表现。

**[Guidebench Guideline Following](llm_agent/guidebench_guideline_following.md)**

:   提出 GuideBench 基准测试，系统评估 LLM 在领域导向指南遵循方面的能力，覆盖 7 个任务类别共 1272 个实例，从规则遵循、规则更新鲁棒性和人类偏好对齐三个维度评估 18 个 LLM，发现当前模型在复杂领域规则遵循上仍有较大提升空间。

**[Gödel Agent A Self-Referential Agent Framework For Recursive Self-Improvement](llm_agent/gödel_agent_a_self-referential_agent_framework_for_recursive_self-improvement.md)**

:   提出 Gödel Agent，一种受 Gödel Machine 启发的自引用 Agent 框架，能通过 Python monkey patching 在运行时读取和修改自身代码（包括修改自身的修改逻辑），实现递归自改进，在 DROP/MGSM/MMLU/GPQA 上超越手工设计和元学习优化的 Agent。

**[Iagent Llm Agent As A Shield Between User And Recommender Systems](llm_agent/iagent_llm_agent_as_a_shield_between_user_and_recommender_systems.md)**

:   提出用户-Agent-平台三层范式，在用户和推荐系统之间插入 LLM Agent 作为保护层，通过指令解析、知识获取、重排序和动态用户画像实现个性化推荐，在四个数据集上平均提升 16.6%，同时有效缓解回音室效应和低活跃用户的不公平问题。

**[Legalagentbench Evaluating Llm Agents In Legal Domain](llm_agent/legalagentbench_evaluating_llm_agents_in_legal_domain.md)**

:   提出 LegalAgentBench，一个面向中国法律领域的 LLM Agent 综合评测基准，包含 17 个真实语料库、37 个工具和 300 个覆盖多跳推理与写作的任务，通过关键词匹配和过程进度率实现细粒度评估。

**[Llm Agent Image Classification](llm_agent/llm_agent_image_classification.md)**

:   提出 Conditional Concept Bottleneck Models (CoCoBMs) 和 LLM-driven Concept Agent 框架，通过类别条件化的概念评分机制和基于环境反馈的动态概念库优化，在 6 个数据集上提升分类准确率 6% 的同时将可解释性提升约 30%。

**[Locagent Graph-Guided Llm Agents For Code Localization](llm_agent/locagent_graph-guided_llm_agents_for_code_localization.md)**

:   LocAgent 将代码库解析为有向异构图（含 contain/import/invoke/inherit 四种关系），并设计统一工具（SearchEntity/TraverseGraph/RetrieveEntity）引导 LLM Agent 进行多跳推理，实现高精度代码定位，在文件级达到 92.7% 准确率，同时通过微调开源模型将成本降低 86%。

**[Meco Metacognition Tool Use](llm_agent/meco_metacognition_tool_use.md)**

:   提出 MeCo（Meta-Cognition Trigger），通过表示工程从 LLM 内部提取"元认知信号"——模型对自身能力的自我评估——来自适应决定是否需要调用外部工具，无需微调且计算开销极小，在多个骨干模型和基准上显著改善工具使用决策的准确性。

**[Metal A Multi-Agent Framework For Chart Generation With Test-Time Scaling](llm_agent/metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)**

:   提出 METAL，一个基于 VLM 的多智能体框架，将图表生成任务（chart-to-code）分解为生成、视觉评审、代码评审和修订四个专门化智能体的迭代协作，在 ChartMIMIC 基准上比现有最佳方法提升 5.2% F1，并展现了测试时缩放（test-time scaling）现象。

**[Mextra Agent Memory Privacy](llm_agent/mextra_agent_memory_privacy.md)**

:   本文系统研究了 LLM Agent 记忆模块的隐私风险，提出 MEXTRA 黑盒记忆提取攻击，通过精心设计的定位-对齐攻击 prompt 和自动化多样 prompt 生成方法，在医疗和网购两种 Agent 上成功提取大量私人查询记录。

**[Multi Agent Dialect Bias Privacy Qa](llm_agent/multi_agent_dialect_bias_privacy_qa.md)**

:   提出一个双 Agent 框架（Dialect Agent + Privacy Policy Agent），通过方言感知翻译和迭代协作来消除隐私政策QA系统在不同英语方言间的性能差距，无需重训练或方言特定微调，在 PrivacyQA 和 PolicyQA 上将方言间最大性能差距降低最高 82%。

**[Multiagentbench Evaluating The Collaboration And Competition Of Llm Agents](llm_agent/multiagentbench_evaluating_the_collaboration_and_competition_of_llm_agents.md)**

:   提出 MultiAgentBench 基准测试和 MARBLE 框架，系统评估 LLM 多智能体系统在协作与竞争场景中的表现，包含 6 种交互场景（研究、Minecraft、数据库、编程、讨价还价、狼人杀），引入基于里程碑的 KPI 指标和协调评分，发现 gpt-4o-mini 整体任务分最高、图结构协调协议在研究场景中表现最佳、认知规划可提升里程碑达成率 3%。

**[Multiple Llm Agents Debate For Equitable](llm_agent/multiple_llm_agents_debate_for_equitable.md)**

:   提出 Multi-Agent Debate 框架，让两个 LLM agent 围绕文化场景进行辩论并由 judge LLM 仲裁，在 NormAd-eti 基准上显著提升文化适应准确率和跨文化群体公平性，使 7-9B 小模型达到 27B 模型的性能水平。

**[Nexussum Narrative Summarization](llm_agent/nexussum_narrative_summarization.md)**

:   提出 NexusSum，一个三阶段多Agent LLM框架（对话转描述→层次摘要→迭代压缩），无需微调即可处理书籍/电影/电视剧等长叙事文本的摘要生成，在 BookSum 上 BERTScore 提升达 30%。

**[Os-Kairos Adaptive Interaction For Mllm-Powered Gui Agents](llm_agent/os-kairos_adaptive_interaction_for_mllm-powered_gui_agents.md)**

:   提出 OS-Kairos，通过协作探测框架标注每步置信度分数并微调进基座模型，使 GUI Agent 能在每步预测置信度、自主决定执行或请求人类干预，在复杂场景下任务成功率 (TSR) 从 OS-Atlas-Pro-7B 的 14.29% 提升到 88.20%，在 AITZ 和 Meta-GUI 基准上也有 24~87% 的绝对提升。

**[Os Agents A Survey On Mllm-Based Agents For General Computing Devices Use](llm_agent/os_agents_a_survey_on_mllm-based_agents_for_general_computing_devices_use.md)**

:   首篇全面综述 MLLM 驱动的操作系统代理（OS Agents），系统梳理其基础组件、构建方法、评估基准和未来方向。

**[Os Agents Survey Mllm](llm_agent/os_agents_survey_mllm.md)**

:   首个系统性综述基于（多模态）大语言模型的操作系统智能体（OS Agents），覆盖基础概念、构建方法（基础模型+Agent框架）、评估基准和商业产品，全面梳理了从CogAgent到Anthropic Computer Use等50+工作的技术演进。

**[Os Genesis Gui Agent Trajectory](llm_agent/os_genesis_gui_agent_trajectory.md)**

:   提出OS-Genesis，一种"先交互探索再逆向生成任务"的GUI agent轨迹数据合成范式，通过无人监督的UI元素遍历收集状态转移三元组，逆向合成任务指令后用Trajectory Reward Model质量控制，在AndroidWorld上将Qwen2-VL-7B性能从9.82%提升至17.41%，接近GPT-4o的23.70%。

**[Pasa An Llm Agent For Comprehensive Academic Paper Search](llm_agent/pasa_an_llm_agent_for_comprehensive_academic_paper_search.md)**

:   PaSa 是一个基于 LLM 的学术论文搜索智能体，通过自主调用搜索工具、阅读论文和导航引用网络来实现全面准确的学术文献检索，经 RL 训练后在真实场景中大幅超越 Google Scholar 和 GPT-4o。

**[Play2Prompt Zero-Shot Tool Instruction Optimization For Llm Agents Via Tool Play](llm_agent/play2prompt_zero-shot_tool_instruction_optimization_for_llm_agents_via_tool_play.md)**

:   提出 Play2Prompt，通过让 LLM 自主"玩"工具（试探输入输出行为）来零样本地生成工具使用示例和优化工具文档，无需任何标注数据即可显著提升 LLM agent 的工具调用能力。

**[R2D2 Reflective Agentic Memory](llm_agent/r2d2_reflective_agentic_memory.md)**

:   R2D2 提出了一个结合 Remember（经验回放缓冲区 + A* 搜索导航）和 Reflect（错误反思 + 反思记忆存储）两范式的 Web Agent 框架，将 Web 导航从 Unknown MDP 转化为 Known MDP，在 WebArena 上导航错误减少 50%，任务完成率提升 3 倍，超越 SOTA 17%。

**[Repro-Bench Can Agentic Ai Systems Assess The Reproducibility Of Research Claims](llm_agent/repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)**

:   本文提出 REPRO-Bench，一个包含 112 个社会科学论文实例的基准，用于评估 AI Agent 自动化评估论文可重复性的能力；现有最佳 Agent 准确率仅 21.4%（低于随机猜测的 25%），作者进一步开发的 REPRO-Agent 将准确率提升至 36.6%（71% 相对提升）。

**[Repro-Bench Can Agentic Ai Systems Assess The Reproducibility Of Social Science ](llm_agent/repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_social_science_.md)**

:   提出 REPRO-Bench，包含 112 个社会科学论文的可复现性评估任务，发现现有 AI Agent（最高准确率仅 21.4%）远不足以自动化该流程，并据此开发 REPRO-Agent 将准确率提升至 36.6%。

**[Select Read And Write A Multi-Agent Framework Of Full-Text-Based Related Work Ge](llm_agent/select_read_and_write_a_multi-agent_framework_of_full-text-based_related_work_ge.md)**

:   提出 Select-Read-Write 三 Agent 协同框架，通过图感知的阅读顺序决策和共享工作记忆机制，实现基于论文全文（而非摘要）的 Related Work 自动生成，在 Llama3-8B / Claude-3-Haiku / GPT-4o 三个基座模型上均取得一致提升，Citation Graph 策略效果最优。

**[Self Taught Agentic Long Ctx](llm_agent/self_taught_agentic_long_ctx.md)**

:   提出 AgenticLU 框架，通过 Chain-of-Clarifications (CoC) 工作流让 LLM 自主生成澄清问题并检索相关上下文，再通过 SFT+DPO 两阶段微调将树搜索路径蒸馏到模型中，使 8B 模型在 128K 长上下文 QA 任务上大幅超越基线。

**[Smart Self-Aware Agent For Tool Overuse Mitigation](llm_agent/smart_self-aware_agent_for_tool_overuse_mitigation.md)**

:   揭示 LLM Agent 中的"工具过度使用"现象（≥30% 的工具调用是不必要的），提出 SMART 元认知范式，通过在推理步骤中显式标注"知识驱动 vs 工具依赖"来训练 Agent 的自感知能力，7B 模型匹配 GPT-4o 水平。

**[Sudo Rm -Rf Agentic Security](llm_agent/sudo_rm_-rf_agentic_security.md)**

:   提出 SUDO 两阶段攻击框架针对计算机使用 Agent：静态阶段用 Detox2tox 将恶意请求去毒化→生成执行计划→回毒化恢复恶意载荷；动态阶段用检查清单迭代优化攻击，在 MANUS 上达到 63.19% 攻击成功率。

**[Synworld Agentic Action Knowledge](llm_agent/synworld_agentic_action_knowledge.md)**

:   SynWorld 提出让 Agent 在合成的虚拟场景中通过蒙特卡洛树搜索（MCTS）来探索和优化动作知识（工具描述和工作流），使 Agent 能够自主适应新环境的工具使用，在 ToolBench 上比 ReAct 基线提升约 9 个百分点。

**[Table Critic Multi Agent](llm_agent/table_critic_multi_agent.md)**

:   提出 Table-Critic 多智能体框架，通过 Judge-Critic-Refiner-Curator 四个专门化 Agent 的协作批评与迭代精化，配合自进化模板树累积批评知识，在 WikiTQ 和 TabFact 上分别实现 73.7% 和 91.7% 的准确率，大幅超越现有方法。

**[The Behavior Gap Evaluating Zero-Shot Llm Agents In Complex Task-Oriented Dialog](llm_agent/the_behavior_gap_evaluating_zero-shot_llm_agents_in_complex_task-oriented_dialog.md)**

:   提出一个综合评估框架来量化 LLM agent 与人类专家在任务导向对话中的"行为差距"（dialog acts、工具使用、知识利用三个维度），发现行为差距随任务复杂度增加显著扩大（相关系数 0.963），缩小行为差距可平均提升 24.3% 性能。

**[Theorem-Of-Thought A Multi-Agent Framework For Abductive Deductive And Inductive](llm_agent/theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)**

:   提出 Theorem-of-Thought (ToTh) 框架，通过三个并行智能体分别模拟溯因、演绎和归纳推理，将推理轨迹构建为形式推理图并利用 NLI 校准的贝叶斯置信传播选出最连贯推理链，在符号和数值推理上一致优于 CoT、Self-Consistency 和 CoT-Decoding。

**[Toolhop Multi Hop Tool Use](llm_agent/toolhop_multi_hop_tool_use.md)**

:   提出ToolHop——首个query-driven构建的多跳工具使用评估数据集（995个多跳查询+3912个本地可执行工具），评估14个LLM后发现最强的GPT-4o仅达49.04%准确率，揭示了不同模型家族在工具使用策略上的显著差异。

**[Toolhop Multi Hop Tool Use Benchmark](llm_agent/toolhop_multi_hop_tool_use_benchmark.md)**

:   提出 ToolHop——一个包含 995 个多跳查询和 3912 个本地可执行工具的基准数据集，通过"查询驱动"的数据构建方式（先有查询再造工具）确保工具间有真实依赖关系和可验证答案，评测 14 个 LLM 发现最强的 GPT-4o 准确率仅 49%，揭示了不同模型家族在工具使用上的显著策略差异。

---

## 💡 LLM推理 { #llm_reasoning }

**[Aristotle Logical Reasoning](llm_reasoning/aristotle_logical_reasoning.md)**

:   提出 Aristotle 逻辑推理框架，将符号表达式和逻辑规则全面融入 Decompose-Search-Resolve 的每个阶段，通过逻辑分解器、搜索路由器和消解器三大组件实现逻辑完备的推理，在多个逻辑推理基准上以 GPT-4 平均提升 4.5%、GPT-4o 平均提升 5.4% 超越 SOTA。

**[Bpp-Search Enhancing Tree Of Thought Reasoning For Mathematical Modeling Problem](llm_reasoning/bpp-search_enhancing_tree_of_thought_reasoning_for_mathematical_modeling_problem.md)**

:   提出 BPP-Search 算法，将 Beam Search、过程奖励模型 (PRM) 和 Pairwise Preference 机制整合到 Tree-of-Thought 框架中，用于运筹学数学建模问题的自动求解，在 StructuredOR 等数据集上以更少的推理步骤显著超越 CoT/SC/ToT 基线。

**[Can Large Language Models Detect Errors In Long Chain-Of-Thought Reasoning](llm_reasoning/can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)**

:   本文提出DeltaBench——首个系统评估o1类模型长CoT推理质量和现有LLM/PRM错误检测能力的基准数据集，通过对1,236个样本的精细人工标注，揭示了o1类模型约27%推理冗余、67.8%反思无效，以及最强critic模型GPT-4-turbo-128k也仅达F1=40.8%的令人警醒的现状。

**[Chain-Of-Reasoning Towards Unified Mathematical Reasoning In Large Language Mode](llm_reasoning/chain-of-reasoning_towards_unified_mathematical_reasoning_in_large_language_mode.md)**

:   提出Chain-of-Reasoning（CoR）统一框架，将自然语言推理(NLR)、算法推理(AR)和符号推理(SR)三种范式整合在同一推理链中协同工作，配合渐进式范式训练(PPT)策略，使7B模型在定理证明上零样本超越GPT-4o 41%，在MATH上超越RL方法15%。

**[Chain Of Reasoning Unified Math](llm_reasoning/chain_of_reasoning_unified_math.md)**

:   提出 Chain-of-Reasoning（CoR）框架，将自然语言推理（NLR）、算法推理（AR）和符号推理（SR）三种范式统一在一个推理链中，通过渐进范式训练（PPT）策略让 7B 模型（CoR-Math-7B）在零样本下超越 GPT-4o 41% 的定理证明准确率，在 MATH 基准上超过 RL 方法 15%。

**[Clozemath Improving Mathematical Reasoning In Language Models By Learning To Fil](llm_reasoning/clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)**

:   ClozeMath 提出了一种受人类完形填空学习启发的微调策略，通过掩码数学解答中的方程式并训练模型预测它们（text-infilling目标），与标准语言模型目标联合训练，在GSM8K和MATH上显著超越了强基线Masked Thought，并在推理时间扩展和鲁棒性测试中表现出更好的泛化能力。

**[Cot-Icl Lab A Synthetic Framework For Studying Chain-Of-Thought Learning From In](llm_reasoning/cot-icl_lab_a_synthetic_framework_for_studying_chain-of-thought_learning_from_in.md)**

:   提出 CoT-ICL Lab 框架，通过解耦因果结构（DAG）和 token 处理函数（MLP）生成可控的合成 token 化数据集，系统研究了 CoT 对 ICL 的加速效应、模型深度的关键作用、以及 Transformer 嵌入与注意力图对底层推理结构的学习机制。

**[Cot-Uq Improving Response-Wise Uncertainty Quantification In Llms With Chain-Of-](llm_reasoning/cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)**

:   针对 LLM 在推理任务中过度自信的问题，提出 CoT-UQ 框架，将 CoT 推理步骤中的关键词提取和重要性评分整合到不确定性量化过程中，在逻辑和数学推理任务上 AUROC 平均提升 5.9%。

**[Cot-Valve Length-Compressible Chain-Of-Thought Tuning](llm_reasoning/cot-valve_length-compressible_chain-of-thought_tuning.md)**

:   本文提出CoT-Valve，一种通过在参数空间中识别"长度控制方向"（以LoRA实现）来弹性控制推理链长度的方法，仅训练一次即可生成从长到短不同长度的推理路径，在QwQ-32B-Preview上将GSM8K推理链从741压缩至225 tokens且准确率仅降0.15%（95.07%→94.92%）。

**[Critic-Cot Boosting The Reasoning Abilities Of Large Language Model Via Chain-Of](llm_reasoning/critic-cot_boosting_the_reasoning_abilities_of_large_language_model_via_chain-of.md)**

:   提出 Critic-CoT 框架，通过逐步 Chain-of-Thought 批判范式和无需人工标注的弱监督数据自动构建，将 LLM 的自我批判从 System-1 式直觉判断推向 System-2 式慎重逐步分析；两阶段训练（GPT-4 蒸馏 + 自我批判）使 Llama-3-70B-Instruct 在 GSM8K 从 89.6% 提升至 95.4%，MATH500 从 50.4% 提升至 68.4%，并发现批判能力与任务求解能力可以相互增强。

**[Dcot Diverse Cot Refinement](llm_reasoning/dcot_diverse_cot_refinement.md)**

:   提出 Diverse Chain of Thought (DCoT) 训练方法，通过在单次推理中生成多条串行推理链实现"推理内自修正"（within-inference refinement），在 1.3B–70B 模型上均超越标准 CoT 基线，尤其在大输出空间任务（数值/抽取型）上提升显著。

**[Define Decision-Making With Analogical Reasoning Over Factor Profiles](llm_reasoning/define_decision-making_with_analogical_reasoning_over_factor_profiles.md)**

:   提出 DeFine 框架，从财报电话会议等复杂场景的语音转录文本中构建概率因子画像(factor profile)，结合 Bradley-Terry 模型识别关键因子并通过因子画像间的 KL 散度做类比推理，用于辅助 LLM 在不确定性下做投资决策，准确率和 F1 均超越基线。

**[Dgprm Dynamic Process Reward](llm_reasoning/dgprm_dynamic_process_reward.md)**

:   DG-PRM 提出了一种动态可泛化的过程奖励建模框架，通过奖励树存储多维度评估标准并动态选择步骤相关的奖励信号，用 Pareto 支配估计处理多面奖励，在 PRMBench 上达到 SOTA 且具有优异的跨领域泛化能力。

**[Drt Deep Reasoning Translation Via Long Chain-Of-Thought](llm_reasoning/drt_deep_reasoning_translation_via_long_chain-of-thought.md)**

:   将长 CoT 推理引入机器翻译，构建多智能体框架（翻译器→顾问→评估器）迭代精炼含比喻/隐喻的文学翻译，合成 22K 长思维翻译训练样本，训练的 DRT-14B 在文学翻译上超越 QwQ-32B 和 DeepSeek-R1-Distill-32B 等大模型。

**[Enhancing Mathematical Reasoning In Llms By Stepwise Correction](llm_reasoning/enhancing_mathematical_reasoning_in_llms_by_stepwise_correction.md)**

:   本文提出StepCo（Stepwise Correction），一种迭代式"验证-修正"框架：利用过程监督验证器（PSV）逐步定位LLM推理路径中的首个错误步骤并触发LLM修正，在8个数学推理基准上以GPT-4o为后端取得94.1%平均准确率，超越Best-of-10方法+2.4个点，同时减少77.8%的token消耗。

**[Enhancing Retrieval Systems With Inference-Time Logical Reasoning](llm_reasoning/enhancing_retrieval_systems_with_inference-time_logical_reasoning.md)**

:   提出推理时逻辑推理框架（ITLR），利用 LLM 将自然语言查询转换为逻辑表达式（AND/OR/NOT），然后基于模糊逻辑对各子项的 cosine similarity 分数进行组合，在合成数据和 NFCorpus/SciFact/ArguAna 三个真实数据集上一致性超越传统 dense retrieval 和 BRIGHT baseline，尤其在含否定的复杂查询上提升显著。

**[Finereason Evaluating And Improving Llms Deliberate Reasoning Through Reflective](llm_reasoning/finereason_evaluating_and_improving_llms_deliberate_reasoning_through_reflective.md)**

:   提出 FineReason——一个基于逻辑谜题的推理基准，通过"状态检查"（判断当前状态是否可解）和"状态转换"（决定下一步操作）两个任务，对LLM的审慎推理能力（反思、回溯、纠错）进行原子级粒度评估，并证明在谜题数据上的训练可迁移提升数学推理能力（GSM8K 提升 5.1%）。

**[Glore Long Cot Representation](llm_reasoning/glore_long_cot_representation.md)**

:   从表示空间角度发现 LLM 将长 CoT 推理编码为一种与普通 CoT 明确区分的通用能力，提出 GLoRE（General Long CoT Reasoning via Representation Engineering）——通过对比推理模式注入和领域特定表示调整来解锁长 CoT 能力，无需训练即可在域内和跨域场景下超越 SFT 方法。

**[Improve Vlm Cot Reasoning](llm_reasoning/improve_vlm_cot_reasoning.md)**

:   通过GPT-4o蒸馏193k CoT数据做SFT + 基于答案正确性构建偏好对做DPO，显著提升VLM的CoT推理能力（LLaVA-Reasoner在8个benchmark上CoT平均提升12.6%），且CoT训练还能反哺直接预测性能。

**[Improving Chain-Of-Thought Reasoning Via Quasi-Symbolic Abstractions](llm_reasoning/improving_chain-of-thought_reasoning_via_quasi-symbolic_abstractions.md)**

:   本文提出QuaSAR（Quasi-Symbolic Abstract Reasoning），一种CoT变体方法，通过引导LLM先对问题进行符号化抽象（提取变量/谓词）、再用半形式化表示重构问题、最后基于准符号推理链求解，在GPT-4o上相比CoT提高最多8%准确率，并显著增强了对对抗性变体（选项打乱、数值替换）的鲁棒性。

**[Local Look-Ahead Guidance Via Verifier-In-The-Loop For Automated Theorem Proving](llm_reasoning/local_look-ahead_guidance_via_verifier-in-the-loop_for_automated_theorem_proving.md)**

:   提出 LeanListener，在自动定理证明(ATP)中引入 verifier-in-the-loop 设计，利用 Lean 验证器在每步提供中间反馈（子目标数变化）而非仅轨迹级奖励，通过在线 GRPO 训练使 ReProver 的 tactic 有效率和证明率均获提升，证明速度快 20%。

**[Logicpro Program Guided Reasoning](llm_reasoning/logicpro_program_guided_reasoning.md)**

:   提出 LogicPro 数据合成方法，利用 LeetCode 算法题和 Python 代码解作为逻辑源，通过"问题生成→代码中间变量提取→程序引导推理生成"三步流水线，从 2360 道算法题合成 540K 高质量文本推理数据，在 BBH27、LogicBench、DROP 等多个 OOD 基准上显著超越现有推理数据集。

**[Marco-O1 V2 Towards Widening The Distillation Bottleneck For Reasoning Models](llm_reasoning/marco-o1_v2_towards_widening_the_distillation_bottleneck_for_reasoning_models.md)**

:   揭示了直接蒸馏大推理模型（如 DeepSeek-R1）的长 CoT 数据到小模型时的「形式化长时间思考」瓶颈，提出基于 MCTS 从头构造树状 CoT 数据并结合思维长度平衡、细粒度 DPO 和联合训练目标来缓解该问题。

**[Mclm Multilingual Test Time Scaling](llm_reasoning/mclm_multilingual_test_time_scaling.md)**

:   提出 MCLM（55 语言的竞赛级数学基准），发现三种 test-time scaling 方法（ORM/PRM/Budget Forcing）在英语上提升显著（如 AIME +20 分），但在其他语言上平均仅提升 1.94 分，表明 test-time scaling 的多语言泛化能力严重不足。

**[Mm-Verify Enhancing Multimodal Reasoning With Chain-Of-Thought Verification](llm_reasoning/mm-verify_enhancing_multimodal_reasoning_with_chain-of-thought_verification.md)**

:   本文提出MM-Verifier和MM-Reasoner两个模型，通过模拟搜索+拒绝采样合成长链CoT验证数据、以及文本蒸馏合成多模态推理数据，仅7B参数即在MathVista上达到65.3%准确率超越GPT-4o（63.8%）和人类表现（60.3%）。

**[On Generalization Across Measurement Systems Llms Entail More Test-Time Compute ](llm_reasoning/on_generalization_across_measurement_systems_llms_entail_more_test-time_compute_.md)**

:   系统研究 LLM 跨度量系统（货币、长度、重量）的泛化能力，发现模型默认使用训练数据中的主导度量（如美元、公制），对非主导度量查询准确率显著下降；CoT 推理可弥补但推理成本增加高达 300%，对欠代表文化用户构成系统性不公平。

**[One Missing Piece For Open-Source Reasoning Models A Dataset To Mitigate Cold-St](llm_reasoning/one_missing_piece_for_open-source_reasoning_models_a_dataset_to_mitigate_cold-st.md)**

:   提出 Long CoT Collection——一个由短链式思维 LLM（如 GPT-4o）标注的 100K 长链式推理数据集，通过从 o1 提取推理流程作为引导，使短 CoT LLM 也能生成长 CoT 数据，从而解决强化学习中的冷启动问题，训练在该数据上初始化的模型在后续 RL 中获得 2-3 倍的性能提升。

**[PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation](llm_reasoning/pcot_persuasion-augmented_chain_of_thought_for_detecting_fake_news_and_social_me.md)**

:   提出 PCoT，一种零样本方法，利用说服知识增强虚假信息检测——受心理学研究启发（识别说服谬误可提高假新闻检测），两阶段处理：先识别分析说服信号，再将说服分析融入推理判断，平均提升 15%。

**[Pcot Persuasion Chain Of Thought Fake News](llm_reasoning/pcot_persuasion_chain_of_thought_fake_news.md)**

:   提出 PCoT（说服增强的思维链）方法，通过两阶段推理——先让 LLM 分析文本中的说服策略，再结合说服分析结果判断是否为虚假信息——在零样本设置下，5 个 LLM 和 5 个数据集上平均提升 15% 的检测 F1。

**[Ranked Voting Based Self-Consistency Of Large Language Models](llm_reasoning/ranked_voting_based_self-consistency_of_large_language_models.md)**

:   将 Self-Consistency 的多数投票升级为排序投票，让 LLM 每次推理生成多个候选答案的偏好排序而非单一答案，用三种排序投票方法（IRV/BCV/MRRV）聚合多次推理的排序信息，在 6 个数据集上一致超越传统 SC，最高提升 12.46%。

**[Rethinking The Role Of Prompting Strategies In Llm Test-Time Scaling A Perspecti](llm_reasoning/rethinking_the_role_of_prompting_strategies_in_llm_test-time_scaling_a_perspecti.md)**

:   本文在 6 个 LLM × 8 种 prompting 策略 × 6 个 benchmark 上系统实验发现，随着 majority voting 采样次数增加，简单的 CoT 始终超越复杂 prompting 策略；并从概率论角度给出理论证明，提出 $O(1)$ 复杂度的 scaling 性能预测方法和两种改进策略。

**[Revisiting Self-Consistency From Dynamic Distributional Alignment Perspective On](llm_reasoning/revisiting_self-consistency_from_dynamic_distributional_alignment_perspective_on.md)**

:   将 Self-Consistency 重新理解为采样分布与真实答案分布的动态对齐问题，揭示温度不仅控制采样随机性还直接塑造真实答案分布，据此提出置信度驱动的三阶段动态温度调节机制（FSD 阈值理论推导），在 10 个模型 × GSM8K/MATH 上零训练开销同时提升平均和最佳性能。

**[Revisiting The Test-Time Scaling Of O1-Like Models Do They Truly Possess Test-Ti](llm_reasoning/revisiting_the_test-time_scaling_of_o1-like_models_do_they_truly_possess_test-ti.md)**

:   系统性地揭示了 QwQ/DeepSeek-R1/LIMO 等 o1-like 模型在测试时并不具备真正的顺序扩展 (sequential scaling) 能力——更长的 CoT 并不带来更高准确率，根因是自我修正 (self-revision) 能力不足——并据此提出 Shortest Majority Vote 并行扩展方法显著超越传统多数投票。

**[Safe Math Reasoning](llm_reasoning/safe_math_reasoning.md)**

:   提出 Safe 框架，首次利用 Lean 4 形式化语言对 LLM 数学推理的每一步进行回顾性逐步验证，通过自动形式化+自动定理证明检测幻觉，并与前瞻性 PRM 分数融合，在多个数学数据集上取得 SOTA，同时发布包含 30,809 条形式化声明的 FormalStep 基准。

**[Softcot Soft Chain Of Thought](llm_reasoning/softcot_soft_chain_of_thought.md)**

:   提出 SoftCoT，用一个冻结的小型辅助模型（如 LLaMA-3.2-1B）生成实例特定的"软思维 token"（连续隐状态），通过可训练的投影模块映射到主 LLM 的表示空间作为推理前缀，实现参数高效的连续空间 CoT 推理，避免了全模型微调导致的灾难性遗忘问题。

**[Stricta Structured Reasoning Peer Review](llm_reasoning/stricta_structured_reasoning_peer_review.md)**

:   提出 STRICTA 框架，将专家文本评估（如论文审稿）建模为基于结构因果模型（SCM）的逐步推理图，收集 40+ 生物医学专家对 22 篇论文的 4000+ 推理步骤数据，揭示先验知识差异是评审分歧的主因、写作风格对终审影响过大，并发现 LLM 在人工监督下可有效辅助结构化评估。

**[Test Time Scaling Selective Qa](llm_reasoning/test_time_scaling_selective_qa.md)**

:   首次评估 test-time scaling 模型（DeepSeek R1、S1）在选择性问答（允许拒绝回答）场景下的表现，发现增加推理计算不仅提高正确率还提升对正确答案的置信度，提出基于置信度阈值的选择函数和 "Jeopardy Odds" 效用函数来评估非零错误惩罚下的 test-time scaling 性能。

**[Thinkguard Deliberative Slow Thinking Leads To Cautious Guardrails](llm_reasoning/thinkguard_deliberative_slow_thinking_leads_to_cautious_guardrails.md)**

:   通过从 GPT-4o/DeepSeek-R1 蒸馏结构化批判（安全标签+详细推理理由），微调护栏模型实现"慢思考"式安全判断，在 4 个安全 benchmark 上达到最高平均 F1（75.5%）和 AUPRC（79.5%），相比 LLaMA Guard 3 准确率提升 16.1%、宏 F1 提升 27.0%。

**[Towards Better Chain-Of-Thought A Reflection On Effectiveness And Faithfulness](llm_reasoning/towards_better_chain-of-thought_a_reflection_on_effectiveness_and_faithfulness.md)**

:   从有效性和忠实性两个视角系统分析 CoT 的表现模式：发现问题难度、信息增益和信息流单调性决定 CoT 有效性，并揭示不忠实 CoT 的机制——模型在预测答案时从问题中召回了 CoT 遗漏的正确信息。在此基础上提出 QUIRE 算法，同时提升 CoT 的有效性（+2.4%）和忠实性（+5.6%）。

**[Towards Safety Reasoning In Llms Ai-Agentic Deliberation For Policy-Embedded Cot](llm_reasoning/towards_safety_reasoning_in_llms_ai-agentic_deliberation_for_policy-embedded_cot.md)**

:   提出 AIDsafe 多智能体迭代审议框架，自动生成嵌入安全策略的高质量 CoT 数据，微调后的模型在安全泛化和越狱鲁棒性上显著优于传统安全训练，同时引入 ear-whisperer agent 解决 DPO 偏好数据中 selected/rejected 难以区分的问题。

**[Tract Regression Cot](llm_reasoning/tract_regression_cot.md)**

:   提出 TRACT，一种两阶段回归感知微调方法，将 CoT 推理与回归损失（squared error）结合，用于提升 LLM-as-a-Judge 场景中的数值评分精度，显著优于仅用交叉熵训练或仅用回归损失的现有方案。

**[Training Turn-By-Turn Verifiers For Dialogue Tutoring Agents The Curious Case Of](llm_reasoning/training_turn-by-turn_verifiers_for_dialogue_tutoring_agents_the_curious_case_of.md)**

:   提出 **Traver**（Trace-and-Verify）agent 工作流，通过**知识追踪**显式估计学生知识状态 + **逐轮验证器**（turn-by-turn verifier）对候选辅导话语打分选优，并设计 **Dict** 自动评估协议（模拟学生 + 代码生成测试），在编程辅导场景中将学生 Pass 率从 38.7% 提升至 43.7%（相对提升 106.5%），显著超越 Vanilla Instruct、Self-Refine 和 TreeInstruct。

**[Unveiling The Key Factors For Distilling Chain-Of-Thought Reasoning](llm_reasoning/unveiling_the_key_factors_for_distilling_chain-of-thought_reasoning.md)**

:   系统研究影响 CoT 蒸馏的三大因素（粒度、格式、教师模型），发现 SLM 与粒度呈非单调关系、格式影响较小、强教师不总是更好。

---

## 🔒 LLM安全 { #llm_safety }

**[Agrail A Lifelong Agent Guardrail With Effective And Adaptive Safety Detection](llm_safety/agrail_a_lifelong_agent_guardrail_with_effective_and_adaptive_safety_detection.md)**

:   提出 AGrail，一个终身学习的 LLM Agent 安全护栏框架，通过双 LLM 协作（Analyzer + Executor）和记忆模块，在测试时自适应地生成和优化安全检查策略，有效防御任务特定风险和系统性风险。

**[Aligning Large Language Models To Follow Instructions And Hallucinate Less Via E](llm_safety/aligning_large_language_models_to_follow_instructions_and_hallucinate_less_via_e.md)**

:   提出 NOVA 框架——通过过滤掉 LLM "不熟悉"的指令数据来同时减少幻觉和保持指令遵从能力——Internal Consistency Probing（ICP）通过多次自生成回复的隐藏状态一致性评估 LLM 对指令的熟悉度，Semantic Equivalence Identification（SEI）通过语义聚类+投票评估 LLM 对目标回复的熟悉度，再用质量奖励模型确保数据质量。大幅减少幻觉同时保持指令遵从。

**[Answer When Needed Forget When Not Language Models Pretend To Forget Via In-Cont](llm_safety/answer_when_needed_forget_when_not_language_models_pretend_to_forget_via_in-cont.md)**

:   提出"上下文知识遗忘"（In-Context Knowledge Unlearning）——通过引入遗忘 token（`<<UNL>>...<<UNL>>`）使 LLM 在推理时根据查询上下文选择性地"遗忘"特定知识，微调后 LLM 在 TOFU/AGE/RWKU 上达到 95% 遗忘准确率同时保留 80% 无关知识。更深入的内部分析发现 LLM 在中间层仍生成正确答案，仅在最后一层决定"假装遗忘"。

**[Arghitz At Archehr-Qa 2025 A Two-Step Divide And Conquer Approach To Patient Que](llm_safety/arghitz_at_archehr-qa_2025_a_two-step_divide_and_conquer_approach_to_patient_que.md)**

:   在 ArchEHR-QA 2025 共享任务中提出两阶段"分治"方法：先用重排序模型从电子健康记录中提取关键句子，再用小型医学 LLM 生成回复，在不使用外部知识的情况下取得事实性排名第一、总分第 8/30 的成绩。

**[Chinese Simpleqa A Chinese Factuality Evaluation For Large Language Models](llm_safety/chinese_simpleqa_a_chinese_factuality_evaluation_for_large_language_models.md)**

:   提出 Chinese SimpleQA——首个全面的中文事实性评估基准，包含 3000 个高质量短问答（覆盖 6 大主题、99 个子主题），评估 41 个 LLM 后发现仅 o1-preview（63.8%）和 Doubao-pro-32k（61.9%）能通过，并系统揭示了"大模型更好"、"RAG缩小差距"、"对齐降低事实性"等关键洞察。

**[Cliperase Efficient Unlearning Of Visual-Textual Associations In Clip](llm_safety/cliperase_efficient_unlearning_of_visual-textual_associations_in_clip.md)**

:   提出 CLIPErase，一种专为 CLIP 多模态模型设计的机器遗忘框架，通过遗忘模块、保留模块和一致性模块三部分协同，选择性地移除特定视觉-文本关联，同时保持模型在保留数据上的性能。

**[Comparisonqa Evaluating Factuality Robustness Of Llms Through Knowledge Frequenc](llm_safety/comparisonqa_evaluating_factuality_robustness_of_llms_through_knowledge_frequenc.md)**

:   构建 ComparisonQA 基准（283K 配对问题），通过让高频和低频实体共享同一抽象问题实现受控对比，结合正确性和不确定性的两轮评估方法发现 LLM（包括 GPT-4o）对低频知识的鲁棒性极差。

**[Defense Prompt Injection](llm_safety/defense_prompt_injection.md)**

:   本文提出一种"以攻为防"的 prompt injection 防御策略：将已有的攻击技术（ignore、escape、fake completion）反转用于防御，在被注入的数据内容后追加 shield prompt + 原始指令，使 LLM 忽略注入指令而执行原始指令，在多种攻击场景下将 ASR 降至接近零。

**[Exploring Forgetting In Large Language Model Pre-Training](llm_safety/exploring_forgetting_in_large_language_model_pre-training.md)**

:   系统性地探索了 LLM 预训练阶段的灾难性遗忘问题，提出了基于实体记忆的新指标（M_ex、M_in）替代传统 PPL 来检测遗忘，并验证了周期性高强度 memory replay 策略在缓解预训练遗忘中的有效性。

**[Factual Knowledge In Language Models Robustness And Anomalies Under Simple Tempo](llm_safety/factual_knowledge_in_language_models_robustness_and_anomalies_under_simple_tempo.md)**

:   发布 TimeStress 数据集（521K 陈述，2003 条时间事实），评估 18 个 LLM 在时间上下文变化下的事实知识鲁棒性，发现最好的模型仅对 11% 的事实实现完美鲁棒，且存在人类不会犯的关键错误。

**[From Misleading Queries To Accurate Answers A Three-Stage Fine-Tuning Method For](llm_safety/from_misleading_queries_to_accurate_answers_a_three-stage_fine-tuning_method_for.md)**

:   提出三阶段微调方法（误导检测->查询纠正->准确回答）增强 LLM 处理含误导信息输入的能力，在误导检测和 QA 任务上显著提升准确率，同时减少幻觉生成。

**[Hallucination Detox Send](llm_safety/hallucination_detox_send.md)**

:   本文揭示了 LLM 训练过程中幻觉行为的振荡现象，提出 Sensitivity Dropout（SenD）训练协议——通过识别并确定性丢弃高变异敏感嵌入索引来降低训练中的幻觉方差，同时提出计算高效的 Efficient EigenScore（EES）近似方法，在 Pythia 和 Llama 模型上实现高达 17% 的测试时可靠性提升。

**[Halogen Hallucinations](llm_safety/halogen_hallucinations.md)**

:   提出 HALoGEN——覆盖 9 个领域（含编程、科学引用、摘要等）的 10,923 条 prompt 的大规模幻觉评测框架，配套原子级自动验证器，在 14 个 LLM 的约 150,000 条生成上系统性评估幻觉，发现即使最佳模型也可能有高达 86% 的原子事实存在幻觉，并提出 Type A/B/C 三类错误分类法。

**[Hd-Ndes Neural Differential Equations For Hallucination Detection In Llms](llm_safety/hd-ndes_neural_differential_equations_for_hallucination_detection_in_llms.md)**

:   本文首次将神经微分方程（Neural DEs）应用于LLM幻觉检测，通过对隐空间中token激活的连续轨迹建模来系统评估陈述的真实性，在True-False数据集上AUC-ROC超过SOTA 14%以上。

**[Improving Factuality With Explicit Working Memory](llm_safety/improving_factuality_with_explicit_working_memory.md)**

:   提出 Ewe（Explicit Working mEmory），在 LLM 解码过程中引入由多个 KV cache 单元组成的显式工作记忆，实时接收检索知识反馈和事实核查反馈，检测到错误时删除错误句子并用更新后的记忆重新生成，在 4 个事实性长文本生成基准上将 VeriScore F1 提升 2–6 分且不损失回答有用性。

**[Improving Model Factuality With Fine-Grained Critique-Based Evaluator](llm_safety/improving_model_factuality_with_fine-grained_critique-based_evaluator.md)**

:   训练细粒度的事实性评估器 FenCE，通过在公开数据集上增强文本批评（critique）和多工具获取的多样化源文档来提升评估准确率，并利用 FenCE 对生成器响应进行修订和评分以构建偏好训练数据，使 Llama2-7B/Llama3-8B 在 FActScore 上分别提升 16.86%/14.45%。

**[Indirect Prompt Injection Detection](llm_safety/indirect_prompt_injection_detection.md)**

:   本文系统研究间接 prompt injection 攻击的检测与移除：构建评估基准，发现现有检测模型对间接攻击表现不佳但专门训练的模型可达 99% 准确率，提出分割移除和抽取移除两种方法，并将检测+移除组合为过滤管道，有效降低间接 prompt injection 的攻击成功率。

**[Intent Hallucination Eval](llm_safety/intent_hallucination_eval.md)**

:   本文提出"意图幻觉"（Intent Hallucination）概念——LLM 在处理复杂多条件查询时遗漏或误解部分意图约束导致的偏离用户意图的生成，构建 FaithQA 基准（20,068 题）和 Constraint Score 评估指标，实验表明意图幻觉在 SOTA 模型中普遍存在且随查询复杂度增加而加剧。

**[Language Models Can Subtly Deceive Without Lying A Case Study On Strategic Phras](llm_safety/language_models_can_subtly_deceive_without_lying_a_case_study_on_strategic_phras.md)**

:   构建了一个立法环境测试平台（LobbyLens），研究 LLM 是否能通过策略性措辞（strategic phrasing）——即不说谎但有意操纵表达方式——来隐藏修正案中对特定公司的利益导向，发现 LLM 经过 re-planning 可使欺骗率提升最多 40 个百分点。

**[Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection In Open](llm_safety/learning_auxiliary_tasks_improves_reference-free_hallucination_detection_in_open.md)**

:   系统性地研究了开放域长文本生成中的无参考幻觉检测问题，发现 LLM 内部状态（概率/熵）不足以可靠区分事实与幻觉内容，并提出 RATE-FT（Rationale and Auxiliary Task Enhanced Fine-Tuning），通过引入推理解释和辅助 QA 任务增强微调，在 LongFact 上比普通微调提升 3% 以上。

**[Mamba Knockout For Unraveling Factual Information Flow](llm_safety/mamba_knockout_for_unraveling_factual_information_flow.md)**

:   将 Transformer 上的 Attention Knockout 可解释性方法迁移至 Mamba-1 和 Mamba-2，揭示了 SSM 模型中事实信息的流动模式——发现 Mamba 与 Transformer 共享"主语 token 在中后层向最后 token 传递关键信息"的普遍模式，但在首 token 偏置和关系 token 依赖等方面存在架构特异性差异。

**[Monitoring Decoding Mitigating Hallucination Via Evaluating The Factuality Of Pa](llm_safety/monitoring_decoding_mitigating_hallucination_via_evaluating_the_factuality_of_pa.md)**

:   提出 Monitoring Decoding (MD) 框架，在生成过程中动态监控部分响应的事实性，通过监控函数识别易产生幻觉的 token 并利用树搜索策略选择性地修正这些关键 token，从而在保持效率的同时显著提升事实准确性。

**[Odysseus Dynamic Focus Decoding](llm_safety/odysseus_dynamic_focus_decoding.md)**

:   提出动态聚焦解码（DFD），通过追踪 LLM 各层间分布差异（KL 散度）来识别知识密集型解码步骤，自适应调整温度——知识密集步用低温保事实性，非知识密集步用高温促多样性——在七个数据集上同时提升事实性和多样性。

**[On-Policy Self-Alignment With Fine-Grained Knowledge Feedback For Hallucination ](llm_safety/on-policy_self-alignment_with_fine-grained_knowledge_feedback_for_hallucination_.md)**

:   提出 RLFH（Reinforcement Learning for Hallucination），一种在策略（on-policy）自对齐方法，让 LLM 自己作为评判者，将回复分解为原子事实并进行真实性和信息量评估，生成 token 级别的密集奖励信号，通过在线 PPO 优化来有效缓解幻觉问题。

**[Opt-Out Investigating Entity-Level Unlearning For Large Language Models Via Opti](llm_safety/opt-out_investigating_entity-level_unlearning_for_large_language_models_via_opti.md)**

:   提出 Opt-Out，一种基于最优传输理论的实体级 LLM 遗忘方法，利用 Sliced Wasserstein Distance 正则化参数偏移实现精细遗忘；同时构建首个实体级遗忘数据集 ELUDe（20 目标实体 + 144 邻居实体，15K+ forget / 90K+ retain QA 对），在 Llama-3.1-8B 和 Phi-3.5 上全面超越现有方法。

**[Revs Unlearning Sensitive Information In Language Models Via Rank Editing In The](llm_safety/revs_unlearning_sensitive_information_in_language_models_via_rank_editing_in_the.md)**

:   提出 REVS，一种无梯度的模型编辑方法，通过在 FF2 层中定位与敏感 token 关联最强的神经元，将其投影到词汇空间后迭代降低目标 token 排名，在 SSN/Email/URL 三类敏感数据上 Unlearning Score 显著超越 6 种基线（89.58 vs 36.98），同时通用能力几乎零损（MMLU 61.05→60.87），且对 Logit-Lens 和 Delta 提取攻击高度鲁棒。

**[Saferoute Adaptive Model Selection For Efficient And Accurate Safety Guardrails ](llm_safety/saferoute_adaptive_model_selection_for_efficient_and_accurate_safety_guardrails_.md)**

:   提出 SafeRoute，一个二分类路由器，根据输入难度自适应地在小型和大型安全护栏模型之间选择，仅对约5%的"困难"样本使用大模型，在保持安全检测精度的同时大幅降低计算开销。

**[Stochastic Chameleons Irrelevant Context Hallucinations Reveal Class-Based Misge](llm_safety/stochastic_chameleons_irrelevant_context_hallucinations_reveal_class-based_misge.md)**

:   通过行为分析和机械可解释性实验揭示 LLM 无关上下文幻觉的内部机制：模型在底层构建抽象类别表示（如"语言"），然后两条竞争电路（query-based vs context-based）争夺特征选择权，相对激活强度决定正确泛化还是产生幻觉。

**[Towards Context-Robust Llms A Gated Representation Fine-Tuning Approach](llm_safety/towards_context-robust_llms_a_gated_representation_fine-tuning_approach.md)**

:   提出 Grft（Gated Representation Fine-Tuning），一种轻量级即插即用的门控表示微调方法，仅需不到 200 个训练样本和模型 0.0004% 的参数，即可让 LLM 在面对矛盾、无用的外部上下文时表现出类似人类的鲁棒认知行为。

**[Treecut A Synthetic Unanswerable Math Word Problem Dataset For Llm Hallucination](llm_safety/treecut_a_synthetic_unanswerable_math_word_problem_dataset_for_llm_hallucination.md)**

:   提出 TreeCut，一种基于树结构的合成数据集生成方法，通过在树路径上移除必要条件边来系统性生成无穷多的不可回答数学应用题，用以评估 LLM 在面对不可解问题时的幻觉行为。

**[Truth Knows No Language Evaluating Truthfulness Beyond English](llm_safety/truth_knows_no_language_evaluating_truthfulness_beyond_english.md)**

:   构建首个专业翻译的多语言 TruthfulQA 基准（巴斯克语、加泰罗尼亚语、加利西亚语、西班牙语），发现 LLM 的跨语言真实性差异小于预期，且 LLM-as-a-Judge 比多选题指标更贴合人类判断。

**[Ualign Leveraging Uncertainty Estimations For Factuality Alignment On Large Lang](llm_safety/ualign_leveraging_uncertainty_estimations_for_factuality_alignment_on_large_lang.md)**

:   提出 UAlign 框架，利用置信度分数和语义熵两种不确定性估计来显式建模 LLM 知识边界，并将其作为输入特征融入 PPO 对齐训练，使模型自信回答已知问题、坚定拒绝未知问题，在多个知识 QA 数据集上显著提升可靠性与泛化性。

**[Uaqfact Evaluating Factual Knowledge Utilization Of Llms On Unanswerable Questio](llm_safety/uaqfact_evaluating_factual_knowledge_utilization_of_llms_on_unanswerable_questio.md)**

:   提出双语不可回答问题数据集UAQFact（13,970题），每个问题附带知识图谱事实知识，定义三个评估任务分别衡量LLM区分不可回答问题（UAQ）与可回答问题（ABQ）、利用内部/外部事实知识处理UAQ的能力，实验揭示即使LLM已存储相关知识也难以有效利用。

**[Unveiling And Addressing Pseudo Forgetting In Large Language Models](llm_safety/unveiling_and_addressing_pseudo_forgetting_in_large_language_models.md)**

:   揭示 LLM 持续学习中的"伪遗忘"现象：性能下降并非因为模型丧失了旧任务能力，而是指令无法正确激活已有能力。通过归因分析证明遗忘模型的指令依赖度降低，并提出基于 Rationale-Guidance Difficulty（RGD）的动态数据回放框架 RGD-R 来缓解伪遗忘。

**[Which Retain Set Matters For Llm Unlearning A Case Study On Entity Unlearning](llm_safety/which_retain_set_matters_for_llm_unlearning_a_case_study_on_entity_unlearning.md)**

:   系统研究实体遗忘中 retain set 的选择问题，提出 Syntactically Similar Neighbor Set，发现句法相似性（而非领域/实体相似性）才是遗忘过程中知识退化的主要驱动因素，用句法相似的 retain set 做正则化可同时最优保护所有类型的邻居知识。

**[Zjuklab At Semeval-2025 Task 4 Unlearning Via Model Merging](llm_safety/zjuklab_at_semeval-2025_task_4_unlearning_via_model_merging.md)**

:   在 SemEval-2025 Task 4（LLM 敏感内容遗忘）中获得第二名，核心思路是训练两个互补模型（一个过度遗忘、一个遗忘不足），通过 TIES-Merging 合并得到平衡遗忘的模型，本地实验达到近乎完美的 MIA 分数 0.501。

---

## 🎵 音频/语音 { #audio_speech }

**[Aae Voice Chatbot](audio_speech/aae_voice_chatbot.md)**

:   研究将非裔美式英语（AAE）整合到聊天机器人中的效果——开发文本和语音 AAE 聊天机器人并用 AAE 说话者评估，发现文本 AAE 聊天机器人常表现不佳（方言生成不够自然），但语音聊天机器人结合非裔声音和 AAE 元素时用户体验更好，揭示了语言个性化的复杂性。

**[Advancing Zero-Shot Text-To-Speech Intelligibility Across Diverse Domains Via Pr](audio_speech/advancing_zero-shot_text-to-speech_intelligibility_across_diverse_domains_via_pr.md)**

:   提出INTP（Intelligibility Preference Speech Dataset）数据集和面向多种TTS架构的DPO扩展方法，通过偏好对齐显著提升零样本TTS系统在绕口令、重复词、中英混合、跨语言等挑战场景下的可懂度，并验证了弱模型到强模型的泛化能力。

**[Ai4Reading Chinese Audiobook Interpretation System Based On Multi-Agent Collabor](audio_speech/ai4reading_chinese_audiobook_interpretation_system_based_on_multi-agent_collabor.md)**

:   提出 AI4Reading，一个基于 11 个专业化 LLM Agent 协作的中文有声书解读系统，通过主题分析、案例扩展、编辑润色、口语化改写和整合修订等阶段自动生成解读稿，并用 TTS 合成音频，在解读脚本质量（简洁性、完整性、准确性、连贯性）上超过专业人工解读平台樊登读书。

**[Atri Mitigating Multilingual Audio Text Retrieval Inconsistencies By Reducing Da](audio_speech/atri_mitigating_multilingual_audio_text_retrieval_inconsistencies_by_reducing_da.md)**

:   从理论上分析多语言音频文本检索（ML-ATR）中跨语言不一致性的根本原因是训练数据分布误差，并提出 1-to-K 对比学习（KCL）和音频-英语共锚对比学习（CACL）两种策略来减少该误差，在召回率和一致性上达到 SOTA。

**[Audio Dialogue Benchmark](audio_speech/audio_dialogue_benchmark.md)**

:   提出 ADU-Bench，一个包含 20,000+ 开放式音频对话的综合基准，覆盖 3 种通用场景、12 项技能、9 种语言和 4 类歧义处理，首次系统评估大型音频语言模型（LALM）的音频对话理解能力，在 16 个模型上的实验揭示了现有 LALM 在数学符号、角色扮演、多语言和语音歧义处理上的显著不足。

**[Autoregressive Speech Synthesis Without Vq](audio_speech/autoregressive_speech_synthesis_without_vq.md)**

:   MELLE 提出了一种基于连续 mel-spectrogram 帧的自回归语言模型 TTS 方法，通过回归损失 + 变分推断采样模块 + spectrogram flux loss 直接预测连续频谱帧，避免了向量量化带来的保真度损失和采样鲁棒性问题，单阶段模型即可达到与人类水平相当的语音合成质量。

**[Chain-Talker Chain Understanding And Rendering For Empathetic Conversational Spe](audio_speech/chain-talker_chain_understanding_and_rendering_for_empathetic_conversational_spe.md)**

:   提出 Chain-Talker，通过三阶段链式建模（情感理解→语义理解→共情渲染）实现可解释的共情对话语音合成，并开发 CSS-EmCap 自动标注管道为对话语音生成情感描述。

**[Clamp 3 Universal Music Information Retrieval Across Unaligned Modalities And Un](audio_speech/clamp_3_universal_music_information_retrieval_across_unaligned_modalities_and_un.md)**

:   提出 CLaMP 3 统一框架，通过对比学习将乐谱、演奏信号、音频录音与多语言文本对齐到共享表示空间，在无配对训练数据的模态间实现跨模态检索，并展现出对未见语言的强泛化能力。

**[Controlspeech Zero Shot](audio_speech/controlspeech_zero_shot.md)**

:   ControlSpeech 是首个同时实现零样本音色克隆和零样本语言风格控制的TTS系统，通过离散编解码器空间中的解耦表示和风格混合语义密度（SMSD）模块解决了风格控制中的多对多问题。

**[Dialectal Coverage And Generalization In Arabic Speech Recognition](audio_speech/dialectal_coverage_and_generalization_in_arabic_speech_recognition.md)**

:   系统研究阿拉伯语方言覆盖对 ASR 性能的影响，通过多方言预训练和联合微调扩展 ArTST 模型覆盖 17 个阿拉伯国家的语音变体，并探索了代码切换场景下的多语言优化策略。

**[Different Speech Translation Models Encode And Translate Speaker Gender Differen](audio_speech/different_speech_translation_models_encode_and_translate_speaker_gender_differen.md)**

:   通过注意力探针分析不同架构的语音翻译模型如何编码说话人性别信息，发现传统编码器-解码器模型能较好保留性别信息，而新型 speech+MT 架构的适配器会显著擦除性别信息，导致翻译中出现更严重的阳性默认偏差。

**[Distilling An End-To-End Voice Assistant Without Instruction Training Data](audio_speech/distilling_an_end-to-end_voice_assistant_without_instruction_training_data.md)**

:   提出DiVA（Distilled Voice Assistant），通过将文本LLM对转录文本的响应作为自监督信号进行跨模态蒸馏，无需任何语音指令训练数据即可训练端到端语音LLM——仅用3.5k小时ASR数据就泛化到口语问答、分类和翻译任务，且在用户偏好测试中以72%胜率碾压Qwen 2 Audio（使用100倍以上训练计算量）。

**[Dncasr End-To-End Training For Speaker-Attributed Asr](audio_speech/dncasr_end-to-end_training_for_speaker-attributed_asr.md)**

:   提出 DNCASR，一种端到端可训练的说话人归因 ASR 系统，通过链接神经聚类解码器和 ASR 解码器，联合训练生成带说话人标识的转录文本，在 AMI 会议数据上实现 cpWER 9.0% 的相对降低。

**[Does Your Voice Assistant Remember Analyzing Conversational Context Recall And U](audio_speech/does_your_voice_assistant_remember_analyzing_conversational_context_recall_and_u.md)**

:   系统性评估开源语音交互模型的对话历史回忆能力，提出 ContextDialog 基准，发现这些模型在回忆过去语音信息方面远弱于文本模型，且 RAG 方法也难以有效弥补这一差距。

**[Double Entendre Robust Audio-Based Ai-Generated Lyrics Detection Via Multi-View ](audio_speech/double_entendre_robust_audio-based_ai-generated_lyrics_detection_via_multi-view_.md)**

:   提出 DE-detect，一个仅以音频为输入的多视角晚期融合管线，通过结合自动转录歌词的文本特征和语音模型提取的歌词相关音频特征，实现了对 AI 生成歌词的鲁棒检测，在域内外均优于单模态方法。

**[Eta-Wavlm Efficient Speaker Identity Removal In Self-Supervised Speech Represent](audio_speech/eta-wavlm_efficient_speaker_identity_removal_in_self-supervised_speech_represent.md)**

:   提出 Eta-WavLM，通过简单的线性方程将 WavLM 自监督语音表示分解为说话人相关和说话人无关两个分量，无需复杂训练即可生成高质量的说话人解耦表示，在语音转换任务上全面超越现有方法。

**[Gigaspeech2 Low Resource Asr](audio_speech/gigaspeech2_low_resource_asr.md)**

:   GigaSpeech 2 构建了一个约 30,000 小时的大规模低资源语言（泰语、印尼语、越南语）ASR 语料库，通过自动化爬取-转录-精炼管线从无标注 YouTube 视频生成高质量伪标签，训练的模型仅用 10% 参数量即可将 WER 比 Whisper large-v3 降低 25%-40%。

**[Investigating And Enhancing Vision-Audio Capability In Omnimodal Large Language ](audio_speech/investigating_and_enhancing_vision-audio_capability_in_omnimodal_large_language_.md)**

:   发现当前全模态大语言模型（OLLMs）在视觉-音频任务上显著弱于视觉-文本任务，原因在于视觉与音频模态之间缺乏直接对齐，并提出 Self-KD（自知识蒸馏）方法，利用 OLLM 自身的视觉-文本组件作为教师来增强视觉-音频能力。

**[Mind The Gap Static And Interactive Evaluations Of Large Audio Models](audio_speech/mind_the_gap_static_and_interactive_evaluations_of_large_audio_models.md)**

:   本文通过收集 484 名参与者的 7,500 次交互评估数据，首次系统比较了大型音频模型（LAM）的静态基准和交互式评估表现，发现两者之间存在显著差距（$R^2=0.30$），并揭示了用户对 LAM 的真实使用场景和偏好。

**[Mms-Llama Efficient Llm-Based Audio-Visual Speech Recognition With Minimal Multi](audio_speech/mms-llama_efficient_llm-based_audio-visual_speech_recognition_with_minimal_multi.md)**

:   提出 MMS-LLaMA，通过早期音视频融合、动态查询分配的 AV Q-Former 和语速预测器三个模块，将多模态语音 token 压缩至每秒仅 3.5 个，在 LRS3 上以 0.72% WER 达到 SOTA 的同时减少 86% token 用量和 35.7% FLOPs。

**[Omniflatten An End-To-End Gpt Model For Seamless Voice Conversation](audio_speech/omniflatten_an_end-to-end_gpt_model_for_seamless_voice_conversation.md)**

:   提出 OmniFlatten——基于 Qwen2-0.5B 的端到端全双工语音对话模型，通过三阶段渐进式后训练（模态对齐→半双工→全双工对话学习）和统一的 flatten 操作，在不修改 GPT 架构的前提下实现了低延迟的自然全双工语音交互，turn-taking 响应时间仅 193ms，显著优于 Moshi 的 553ms。

**[On The Robust Approximation Of Asr Metrics](audio_speech/on_the_robust_approximation_of_asr_metrics.md)**

:   提出一种无需真实标签的 ASR 性能指标近似方法，利用多模态统一 embedding 空间中的语音-文本相似度和高质量代理模型的 proxy metrics，训练回归模型预测 WER/CER，在 40+ 模型和 14 个数据集上绝对误差控制在个位数以内，超过最新基线 50% 以上。

**[Predicting Turn-Taking And Backchannel In Human-Machine Conversations Using Ling](audio_speech/predicting_turn-taking_and_backchannel_in_human-machine_conversations_using_ling.md)**

:   提出首个融合语言、声学和视觉三模态信号预测对话中轮换（turn-taking）和反馈通道（backchannel）动作的端到端框架，并构建了包含 210+ 小时的 MM-F2F 面对面对话数据集，turn-taking F1 提升 10%，backchannel F1 提升 33%。

**[Spark-Tts An Efficient Llm-Based Text-To-Speech Model With Single-Stream Decoupl](audio_speech/spark-tts_an_efficient_llm-based_text-to-speech_model_with_single-stream_decoupl.md)**

:   提出 Spark-TTS，基于新型单流语音编解码器 BiCodec 和 Qwen2.5 LLM 的高效 TTS 系统，通过将语音解耦为低码率语义 token 和固定长度全局 token，实现零样本语音克隆和从粗到细的属性控制，在 Seed-TTS-eval 上达到 SOTA 可懂度。

**[Sparsify Music Avqa](audio_speech/sparsify_music_avqa.md)**

:   Sparsify 提出三层稀疏化策略（稀疏掩码+自适应稀疏合并+关键子集选择）用于音乐表演视听问答（Music AVQA），在 MUSIC-AVQA 和 v2.0 两个 benchmark 上达到 SOTA（81.75%/81.30%），训练时间减少 28.32%，25% 数据即保持 74% 的全量性能。

**[Speechiq Speechagentic Intelligence Quotient Across Cognitive](audio_speech/speechiq_speechagentic_intelligence_quotient_across_cognitive.md)**

:   提出 SpeechIQ，一个基于 Bloom 认知分类学的层次化语音理解评估框架，从 Remember（WER）、Understand（语义相似度）、Apply（QA 准确率）三个层次综合评估语音 LLM 的智能水平，发现级联 ASR+LLM 系统在同规模下优于端到端多模态模型。

**[Speechweave Diverse Multilingual Synthetic Text Audio Data Generation Pipeline F](audio_speech/speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)**

:   SpeechWeave 提出了一套端到端的合成语音数据生成流水线，通过关键词采样提升文本多样性、在生成时即完成文本归一化（准确率达97%）、并利用跨语言语音克隆实现说话人标准化，生成的数据在多样性上比直接提示LLM高出10-48%，并能有效提升下游TTS模型性能。

**[T2A Feedback Audio Gen](audio_speech/t2a_feedback_audio_gen.md)**

:   提出三个细粒度 AI 音频评分管线（事件出现/事件顺序/声学和谐质量）替代人工标注构建大规模音频偏好数据集 T2A-FeedBack（41K提示+249K音频），用偏好调优增强 TTA 模型的基础能力，在简单（AudioCaps）和复杂（T2A-EpicBench）场景下都显著提升多事件音频生成质量。

**[Tas Audio Spatialization](audio_speech/tas_audio_spatialization.md)**

:   提出 TAS（Text-guided Audio Spatialization）框架，用灵活的文本提示（3D 空间位置描述或声源间相对位置描述）引导潜在扩散模型将单声道音频转换为双耳音频，构建了 376K 样本的 SpatialTAS 数据集，在模拟和真实录制数据上均超越现有方法，并基于 Llama-3.1-8B 开发了空间语义一致性评估模型。

**[Tcsinger 2 Customizable Multilingual Zero-Shot Singing Voice Synthesis](audio_speech/tcsinger_2_customizable_multilingual_zero-shot_singing_voice_synthesis.md)**

:   提出 TCSinger 2，一个多任务多语言零样本歌声合成模型，通过模糊边界编码器、对比学习音频编码器和基于 Flow 的自定义 Transformer（含 Cus-MOE），实现基于歌声/语音/文本提示的风格迁移与多层级风格控制。

**[Towards Reliable Large Audio Language Model](audio_speech/towards_reliable_large_audio_language_model.md)**

:   本文首次系统研究大型音频语言模型（LALM）的可靠性问题，提出训练无关方法（IDK/MCoT/Task Agent）和训练方法（基于模型特定 IDK 数据集的 LoRA SFT），并设计 Reliability Gain Index（RGI）指标来评估可靠性提升效果，发现"知道说不知道"是可跨音频模态迁移的元能力。

**[Who Can Withstand Chat-Audio Attacks An Evaluation Benchmark For Large Audio-Lan](audio_speech/who_can_withstand_chat-audio_attacks_an_evaluation_benchmark_for_large_audio-lan.md)**

:   提出 Chat-Audio Attacks (CAA) 基准，包含四类通用对抗音频攻击（内容攻击、情感攻击、显式噪声攻击、隐式噪声攻击），通过三种评估方法系统评测六个 SOTA 大型音频语言模型的鲁棒性，发现 GPT-4o 表现最优但所有模型均存在显著脆弱性。

**[Zero-Shot Text-To-Speech For Vietnamese](audio_speech/zero-shot_text-to-speech_for_vietnamese.md)**

:   针对越南语零样本TTS缺乏高质量长音频数据集的问题，构建了941小时的PhoAudiobook数据集，并在VALL-E、VoiceCraft和XTTS-v2三个SOTA零样本TTS模型上进行系统实验，证明PhoAudiobook显著提升了模型性能，其中XTTS-v2在长句合成上全面超越基线viXTTS，而VALL-E和VoiceCraft在短句合成上更具鲁棒性。

---

## 📚 预训练/数据 { #llm_pretraining }

**[Adversarial Tokenization](llm_pretraining/adversarial_tokenization.md)**

:   本文发现 LLM 管线中 BPE tokenizer 只使用唯一一种分词方式，但同一字符串存在指数级多种合法分词；通过对抗性地选择非标准分词方案，可以在不改变原始文本的情况下绕过安全对齐，攻击成功率与现有 SOTA 文本级攻击方法相当。

**[Autonomous Data Selection With Zero-Shot Generative Classifiers For Mathematical](llm_pretraining/autonomous_data_selection_with_zero-shot_generative_classifiers_for_mathematical.md)**

:   提出 AutoDS——用基座 LLM 自身作为零样本"生成分类器"自动评估数学文本质量。通过两个 yes/no 问题的 logits 计算连续 LM-Score（而非二分类），筛选高质量数学文本做持续预训练，在 MATH/GSM8K/BBH 上大幅提升并实现约 2 倍 token 效率提升。发布 AutoMathText 数据集。

**[Between Circuits Chomsky](llm_pretraining/between_circuits_chomsky.md)**

:   提出在自然语言预训练前先在形式语言上进行"pre-pretraining"，发现具有层级依赖结构的形式语言（如 k-Shuffle Dyck）能为 Transformer 提供有效的归纳偏置，使 1B 参数模型以 33% 更少的 token 达到相同的语言建模损失。

**[Data-Constrained Synthesis Of Training Data For De-Identification](llm_pretraining/data-constrained_synthesis_of_training_data_for_de-identification.md)**

:   本文系统研究了在数据受限条件下，如何利用领域适应的LLM生成合成临床文本，并通过机器标注训练NER模型进行个人身份信息（PII）检测，发现机器标注器的质量而非生成模型的规模是决定合成数据效用的关键因素。

**[Data Caricatures On The Representation Of African American Language In Pretraini](llm_pretraining/data_caricatures_on_the_representation_of_african_american_language_in_pretraini.md)**

:   结合定量实验、人工判断和定性分析，系统评估了 12 个开源预训练语料库中非裔美国人语言（AAL）的数量与质量：发现 AAL 仅占 0.007%–0.18% 的文档（远低于人口比例），C4 中 28.9% 的 AAL 文本被判为不适合 LLM 生成、24.5% 强化有害刻板印象，且 16 种自动过滤器中有 13 种系统性地偏向保留白人主流英语（WME）而非 AAL。

**[Data Whisperer Data Selection](llm_pretraining/data_whisperer_data_selection.md)**

:   Data Whisperer 提出一种无需训练的注意力加权 few-shot ICL 数据选择方法，利用预训练模型自身的 ICL 能力和注意力分数为训练样本打分，仅用 10% 数据即可超越全量微调性能，同时比现有方法快 7-20 倍。

**[Davir Data Selection Via Implicit Reward For Large Language Models](llm_pretraining/davir_data_selection_via_implicit_reward_for_large_language_models.md)**

:   提出 DavIR 数据选择方法，通过对基座模型与参考模型的损失差进行**参考模型损失归一化**（而非 token 数归一化），有效消除 RHO 目标中的序列长度依赖，使仅 **6%** 的 Alpaca 数据集（3K/52K）训练出的模型优于全量数据训练模型，同时将归一化思想推广到 DPO 得到 DavIR-DPO，在 AlpacaEval 上提升 Zephyr 8% 的对齐性能。

**[Diversity Explains Inference Scaling Laws Through A Case Study Of Minimum Bayes ](llm_pretraining/diversity_explains_inference_scaling_laws_through_a_case_study_of_minimum_bayes_.md)**

:   从 bias-diversity 分解的理论视角重新解释 MBR 解码：质量估计误差 MSE = Bias - Diversity，增加 diversity（伪参考的多样性）是提升 MBR 性能的关键；进一步通过信息论扩展到一般推理方法，揭示 diversity 是推理 scaling law（增加采样提升性能但边际递减）的理论根源，并在机器翻译、摘要、图像描述任务上实证验证。

**[Dual Stage Curriculum Learning Sequence Labeling](llm_pretraining/dual_stage_curriculum_learning_sequence_labeling.md)**

:   提出面向序列标注任务的双阶段课程学习（DCL）框架，通过数据级和模型级两阶段的由易到难训练策略以及基于贝叶斯不确定性的动态难度度量，在提升性能的同时加速训练超过 25%。

**[Emergent Abilities Continued Pt](llm_pretraining/emergent_abilities_continued_pt.md)**

:   揭示了持续预训练（CPT）进行语言适应时，混入英文数据对保留模型上下文学习（ICL）能力和下游涌现能力至关重要——尽管不影响验证困惑度；并提出课程学习和 EMA 权重平均作为替代方案。

**[Fr Spec Speculative Sampling](llm_pretraining/fr_spec_speculative_sampling.md)**

:   发现大词表LLM（如LLaMA-3的128k词表）中投机采样的瓶颈从Transformer层转移到LM Head，提出FR-Spec通过频率排序将草稿模型的词表压缩75%（128k→32k），在EAGLE-2基础上额外获得1.12×加速，且保证最终输出分布数学等价。

**[How Do Llms Acquire New Knowledge A Knowledge Circuits Perspective On Continual ](llm_pretraining/how_do_llms_acquire_new_knowledge_a_knowledge_circuits_perspective_on_continual_.md)**

:   从知识电路（knowledge circuits）角度研究 LLM 在持续预训练中如何获取新知识：新知识的获取依赖于与已有知识的关联性，电路经历"形成→优化"的阶段转变，且呈现从深层到浅层的演化模式。

**[Improving Continual Pre-Training Through Seamless Data Packing](llm_pretraining/improving_continual_pre-training_through_seamless_data_packing.md)**

:   提出 Seamless Packing (SP) 数据打包策略，通过两阶段方法——滑动窗口处理长文本 + FFD 算法打包短文本——在持续预训练中保持上下文连续性、最小化截断和填充，在 99% 的实验设置中超越基线方法。

**[Inconsistent Tokenizations Cause Language Models To Be Perplexed By Japanese Gra](llm_pretraining/inconsistent_tokenizations_cause_language_models_to_be_perplexed_by_japanese_gra.md)**

:   揭示了 tokenizer 的不一致分词是导致 LLM 无法遵守日语"第一人称心理谓词限制"等细微语法规则的根本原因——当限制测试句子为一致分词时，Llama 3 的困惑度差异可改善28倍。

**[Incorporating Domain Knowledge Into Materials Tokenization](llm_pretraining/incorporating_domain_knowledge_into_materials_tokenization.md)**

:   提出 MATTER——一种面向材料科学的领域感知分词框架，通过训练材料概念检测器 MatDetector 并将检测结果注入分词的合并排序中，避免领域术语碎片化，在生成和分类任务上分别平均提升 4% 和 2%。

**[Inserter Speech Instruction](llm_pretraining/inserter_speech_instruction.md)**

:   提出 InSerter（交错语音-文本预训练）方法，通过 TTS 将大规模文本语料合成为交错的语音-文本序列进行预训练，大幅提升 SpeechLLM 的语音指令遵循能力，并构建首个全面的语音指令遵循基准 SpeechInstructBench。

**[Large Vocabulary Size Improves Large Language Models](llm_pretraining/large_vocabulary_size_improves_large_language_models.md)**

:   实证研究词表大小与 LLM 性能的关系，在英语和日语上证明更大的词表（从 5K 到 500K）一致带来更好的下游性能，并提出在继续训练场景中替换词表的方法。

**[Making Llms Better Many-To-Many Speech-To-Text Translators With Curriculum Learn](llm_pretraining/making_llms_better_many-to-many_speech-to-text_translators_with_curriculum_learn.md)**

:   提出 LLM-SRT，将语音到文本翻译（S2TT）任务转化为语音识别与翻译联合任务（SRT），并通过三阶段课程学习策略（ASR→SMT→SRT）有效利用 LLM 的机器翻译能力，在低资源场景（每种语言不到 10 小时数据）下实现 15×14 语言对的 SOTA 多对多语音翻译性能。

**[Metarater A Multidimensional Data Selection Method](llm_pretraining/metarater_a_multidimensional_data_selection_method.md)**

:   提出Meta-rater多维数据选择框架，定义PRRC四个质量维度（专业性/可读性/推理性/清洁度），通过proxy模型回归学习多个质量分数的最优加权组合，使1.3B模型训练收敛速度翻倍、下游任务提升3.23%。

**[Model Performance-Guided Evaluation Data Selection For Effective Prompt Optimiza](llm_pretraining/model_performance-guided_evaluation_data_selection_for_effective_prompt_optimiza.md)**

:   提出 IPOMP——一种两阶段评估数据选择方法，第一阶段通过语义聚类和边界分析选取多样化样本，第二阶段利用提示优化过程中的实时模型性能迭代替换冗余样本，在 BIG-bench 和 LIAR 上将提示优化效果提升 1.6%-3.1%，稳定性提升 50%+，额外开销不到 1%。

**[Nemotron Cc Pretraining Data](llm_pretraining/nemotron_cc_pretraining_data.md)**

:   Nemotron-CC 通过分类器集成、合成数据改写和减少启发式过滤三种策略，从 Common Crawl 构建了 6.3T token 的长期预训练数据集，在 15T token 训练中超越 Llama 3.1 8B。

**[Optimizing Pre-Training Data Mixtures With Mixtures Of Data Expert Models](llm_pretraining/optimizing_pre-training_data_mixtures_with_mixtures_of_data_expert_models.md)**

:   提出Mixture of Data Experts (MDE)方法，通过在各数据域上独立训练专家模型并用混合权重进行概率级集成，高效近似不同数据混合比下的语言模型损失，大幅提升预训练数据混合比例的搜索效率和预测精度。

**[Pre-Training Curriculum For Multi-Token Prediction In Language Models](llm_pretraining/pre-training_curriculum_for_multi-token_prediction_in_language_models.md)**

:   针对小语言模型（SLM）难以直接受益于多 token 预测（MTP）目标的问题，提出前向/反向课程学习策略——前向课程（NTP→MTP）使 SLM 在保持自推测解码加速的同时提升生成质量，反向课程（MTP→NTP）在 NTP 性能上更优但失去推理加速优势。

**[Scar Style Consistency Data Selection](llm_pretraining/scar_style_consistency_data_selection.md)**

:   SCAR 识别出回复的"语言形式"和"指令惊奇度"是影响 LLM 指令微调效果的两个关键风格因素，并提出基于风格一致性的排序方法自动选择高质量训练数据，仅用 0.7% 的原始数据就能让微调后的 LLM 匹配甚至超越全数据集训练的性能。

**[Second Language Arabic Acquisition Of Llms Via Progressive Vocabulary Expansion](llm_pretraining/second_language_arabic_acquisition_of_llms_via_progressive_vocabulary_expansion.md)**

:   受人类第二语言习得启发，提出渐进式词表扩展（Progressive Vocabulary Expansion）方法，通过分阶段指数增长地扩展阿拉伯语子词到 LLaMA2 词表中，在保留原模型英语知识的同时高效适配阿拉伯语，构建出 AraLLaMA 7B/13B 模型。

**[Splintering Nonconcatenative Languages For Better Tokenization](llm_pretraining/splintering_nonconcatenative_languages_for_better_tokenization.md)**

:   提出 Splinter，一种预分词步骤，通过迭代剪除模板字符将非拼接性语言（希伯来语、阿拉伯语、马来语）的词重排为线性形式，使标准 BPE/UnigramLM 能发现形态学上有意义的连续片段，在内在指标和希伯来语下游任务上均优于原始分词。

**[Stealing Training Data From Large Language Models In Decentralized Training Thro](llm_pretraining/stealing_training_data_from_large_language_models_in_decentralized_training_thro.md)**

:   提出 Activation Inversion Attack (AIA)，首次揭示了在去中心化训练框架中，恶意阶段可以通过截获的中间激活值重构训练数据，在 GPT2-XL 微调场景下可精确恢复 62% 的私人邮件地址。

**[Tokalign Vocab Adaptation](llm_pretraining/tokalign_vocab_adaptation.md)**

:   提出 TokAlign，基于 Token 共现信息学习两个词表之间的一对一映射矩阵，高效替换 LLM 的词表，实现跨语言知识迁移和跨模型 token 级蒸馏。

**[Tokenization Is Sensitive To Language Variation](llm_pretraining/tokenization_is_sensitive_to_language_variation.md)**

:   系统研究了 BPE tokenizer 的三个关键设计选择（拟合语料、pre-tokenizer、词表大小）对语言变体鲁棒性任务和敏感性任务下游性能的差异化影响，并提出基于 logistic regression 的 task-aware tokenizer 评估指标，显著优于 Rényi efficiency 等 task-agnostic 指标。

**[Training Dynamics Underlying Language Model Scaling Laws Loss Deceleration And Z](llm_pretraining/training_dynamics_underlying_language_model_scaling_laws_loss_deceleration_and_z.md)**

:   发现语言模型训练中存在 loss deceleration（损失减速）现象——损失曲线在 log-log 空间呈分段线性，根因是 zero-sum learning（ZSL）：per-token 梯度系统性对立导致破坏性干涉，将一部分样本的改善抵消另一部分的恶化；scale up 通过降低减速触发损失 $L_d$ 和提升减速后斜率 $r_d$ 来缓解 ZSL，为突破 scaling law 瓶颈提供了可直接干预的机制。

**[Unsupervised Morphological Tree Tokenizer](llm_pretraining/unsupervised_morphological_tree_tokenizer.md)**

:   提出 TreeTok，一种基于无监督神经形态结构归纳的分词器，通过 MorphOverriding 机制和自监督目标学习字符级树结构，以自顶向下词表匹配方式进行分词，在形态分割和语言建模任务上均优于 BPE/WordPiece。

**[Velocitune A Velocity-Based Dynamic Domain Reweighting Method For Continual Pre-](llm_pretraining/velocitune_a_velocity-based_dynamic_domain_reweighting_method_for_continual_pre-.md)**

:   提出 Velocitune 框架，通过学习速度（learning velocity）动态调整持续预训练中各数据域的采样权重——优先加大学习较慢的域的权重，并利用 scaling law 低成本估计目标损失，在数学/代码推理和系统命令生成任务上显著优于静态混合基线。

---

## ⚡ LLM效率 { #llm_efficiency }

**[Adaptive Grouped Pe Context Window](llm_efficiency/adaptive_grouped_pe_context_window.md)**

:   提出 AdaGroPE（Adaptive Grouped Positional Encoding），一种无需训练的即插即用方法，通过让位置复用次数随距离递增式增长、并根据输入序列长度动态调整位置编码映射，将 LLM 上下文窗口外推到远超预训练长度，在多个 benchmark 上达到 SOTA 甚至超过原生长上下文模型。

**[Clasp Self Speculative Decoding](llm_efficiency/clasp_self_speculative_decoding.md)**

:   CLaSp 提出一种无需训练的自推测解码方法，通过动态规划算法在每个验证步骤后根据上下文动态调整跳层策略，利用上一次验证的完整隐状态作为目标来选择最优跳层集合，在 LLaMA3 系列上实现 1.3-1.7× 加速且不改变生成分布。

**[Cnnsum Exploring Long-Context Summarization With Large Language Models In Chines](llm_efficiency/cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines.md)**

:   构建了 CNNSum——基于中文小说的多尺度长文本摘要基准（695 样本，16k-128k tokens），通过人工标注确保质量，系统测评了 20+ 个 LLM，发现高级 LLM 倾向生成主观评述导致摘要模糊、小模型性价比更高、Base 版微调效果优于 Chat 版，且用短文本数据微调即可显著提升长文本摘要能力。

**[Design Choices For Extending The Context Length Of Visual Language Models](llm_efficiency/design_choices_for_extending_the_context_length_of_visual_language_models.md)**

:   系统性地探索了将现有视觉语言模型（VLM）的上下文窗口扩展到128K的设计空间，从数据配方、位置编码扩展到上下文利用三个维度提出最佳实践，并提出 M-RoPE++ 和混合分辨率训练两项技术，构建的 Giraffe 模型在长上下文 VLM 中达 SOTA。

**[Dynamic Chunking And Selection For Reading Comprehension Of Ultra-Long Context I](llm_efficiency/dynamic_chunking_and_selection_for_reading_comprehension_of_ultra-long_context_i.md)**

:   提出 Dynamic Chunking and Selection (DCS)，通过基于语义相似度的动态分块和问题感知分类器的块选择，解决长文本固定分块导致的语义断裂问题，在 12 个长文本 QA 数据集上以 Llama3 为基座实现 single-hop 平均 35.50（+28.6%）和 multi-hop 平均 29.07（+20.0%）的提升，且在 256k token 输入下保持鲁棒。

**[Efficient Many-Shot In-Context Learning With Dynamic Block-Sparse Attention](llm_efficiency/efficient_many-shot_in-context_learning_with_dynamic_block-sparse_attention.md)**

:   提出 Dynamic Block-Sparse Attention (DBSA)，一种无需训练的推理框架，通过结构化块稀疏注意力编码和动态检索 KV 缓存，在多示例上下文学习中实现接近微调的推理延迟，同时保持 >95% 的最佳方法准确率。

**[Entailment-Preserving First-Order Logic Representations In Natural Language Enta](llm_efficiency/entailment-preserving_first-order_logic_representations_in_natural_language_enta.md)**

:   形式化定义了蕴含保持一阶逻辑表示（EPF）任务及无参考评价指标（EPR系列），提出迭代learning-to-rank训练方法，通过BRIO损失优化T5模型的NL→FOL翻译，使其生成的FOL表示能被自动定理证明器验证蕴含关系，在三个数据集上EPR提升1.8-2.7%、EPR@16提升17.4-20.6%。

**[Focusllm Precise Understanding Of Long Context By Dynamic Condensing](llm_efficiency/focusllm_precise_understanding_of_long_context_by_dynamic_condensing.md)**

:   提出FocusLLM框架，通过将长文本分块并为每块注入动态提示（dynamic prompt），用可训练的候选token浓缩各块的关键信息，再通过并行解码机制聚合到本地上下文中生成下一个token，仅用8K训练长度和0.5B训练预算即可扩展LLaMA-2到400K上下文，在LongBench和∞-Bench上超越所有基线。

**[Gigachat Family Efficient Russian Language Modeling Through Mixture Of Experts A](llm_efficiency/gigachat_family_efficient_russian_language_modeling_through_mixture_of_experts_a.md)**

:   介绍 GigaChat 系列——首个从头为俄语设计并预训练的 MoE 架构 LLM 家族，包含 20B 总参数/3.3B 激活参数的基座和指令微调模型，在俄语 benchmark 上达到同规模 SOTA，训练速度是同量级 dense 模型的 2 倍，推理延迟降低 40%。

**[Gradot Offsite Tuning](llm_efficiency/gradot_offsite_tuning.md)**

:   从优化理论角度首次系统分析 Offsite-tuning 问题，提出梯度保持压缩分数（GCS），并基于此设计了 GradOT 方法，对 MHA 使用动态秩分解（DRD）、对 MLP 使用选择性通道剪枝（SCP），在免训练条件下同时实现性能保持和隐私保护。

**[Kv Latent Cache Reduction](llm_efficiency/kv_latent_cache_reduction.md)**

:   提出KV-Latent范式，通过对KV头的维度进行降采样将其映射到潜空间，仅需不到1%预训练量的额外训练即可恢复性能，在LLaMA-3-8B上KV Cache减少50%（dqk=dvo=64→50%缓存），同时TTFT延迟降低8%。

**[Ladm Long Context Data](llm_efficiency/ladm_long_context_data.md)**

:   提出 LADM 框架，利用注意力机制的内在检索能力来度量长上下文数据中的跨 span 依赖关系，从大规模预训练语料中高效筛选高质量长上下文训练数据，仅用 1B token 持续预训练即可显著提升多种 LLM 的长上下文能力。

**[Literary Evidence Retrieval Via Long-Context Language Models](llm_efficiency/literary_evidence_retrieval_via_long-context_language_models.md)**

:   将 RELiC 数据集改造为长上下文文学证据检索 benchmark（292 个高质量样本），要求模型在完整小说文本（45k-125k tokens）中为文学评论找到缺失引用；Gemini Pro 2.5 以 62.5% 准确率首次超越人类专家（55%），但最佳开源模型 DeepSeek-R1 仅 29.1%，揭示了闭源/开源模型在解释性推理上的巨大鸿沟。

**[Longreward Improving Long-Context Large Language Models With Ai Feedback](llm_efficiency/longreward_improving_long-context_large_language_models_with_ai_feedback.md)**

:   提出 LongReward，利用现成LLM从帮助性、逻辑性、忠实性和完整性四个维度为长上下文模型回复自动打分，结合DPO实现长上下文SFT模型的进一步提升。

**[Longsafety Evaluating Long-Context Safety Of Large Language Models](llm_efficiency/longsafety_evaluating_long-context_safety_of_large_language_models.md)**

:   提出LongSafety——首个专门针对开放式长上下文任务的LLM安全评估基准，包含7类安全问题和6种任务类型共1,543个测试用例，揭示大多数模型安全率低于55%，且短上下文安全能力无法迁移到长上下文场景。

**[Many Shot Attacks Long Context](llm_efficiency/many_shot_attacks_long_context.md)**

:   系统分析 Many-Shot Jailbreaking（MSJ）攻击的关键因素，发现上下文长度是攻击成功的决定性因素，而内容的有害性、主题、格式几乎不重要——即使重复安全内容、随机无意义文本（Lorem Ipsum）都能在长上下文下突破模型安全对齐。

**[Mitigating Posterior Salience Attenuation In Long-Context Llms With Positional C](llm_efficiency/mitigating_posterior_salience_attenuation_in_long-context_llms_with_positional_c.md)**

:   发现长上下文LLM中的后验显著性衰减（PSA）现象——gold token的显著性随上下文增长而下降但仍保持高排名，由此提出无需训练的位置对比解码（PCD）方法，通过对比长距离感知注意力和局部感知注意力的logits来放大长距离信号，在多个长上下文基准上取得SOTA。

**[Native Sparse Attention](llm_efficiency/native_sparse_attention.md)**

:   DeepSeek提出NSA——一种原生可训练的稀疏注意力机制，通过"压缩+选择+滑动窗口"的层次化稀疏策略和硬件对齐的Triton kernel设计，在27B参数模型上实现了超越Full Attention的性能，同时在64k序列上获得前向9倍、解码11.6倍的加速。

**[Neko Cross-Modality Post-Recognition Error Correction With Tasks-Guided Mixture-](llm_efficiency/neko_cross-modality_post-recognition_error_correction_with_tasks-guided_mixture-.md)**

:   提出 NeKo，使用任务引导的 Mixture-of-Experts 语言模型进行跨模态后识别纠错（ASR/ST/OCR/TEC），通过将每个专家指派给特定任务的数据集实现专业化，在 Open ASR Leaderboard 上达到 SOTA，零样本下超越 GPT-3.5 和 Claude-3.5。

**[On Many-Shot In-Context Learning For Long-Context Evaluation](llm_efficiency/on_many-shot_in-context_learning_for_long-context_evaluation.md)**

:   深入研究 many-shot ICL 用于长上下文语言模型评估，提出 Sample Learning Ratio 指标区分 SSL 和 ASL 任务，构建 ManyICLBench 基准全面评测 12 个 LCLM。

**[Ref-Long Benchmarking The Long-Context Referencing Capability Of Long-Context La](llm_efficiency/ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)**

:   提出 Ref-Long benchmark，从"引用定位"（给定 key 识别哪些文档引用了它并返回索引）这一被忽视的维度评估长上下文模型，包含 3 个子集（合成→真实）共 4300 个任务；发现即使 GPT-4o 在 Multi-Hard-24K 上 ExAcc 仅 19%，远低于人类 92%，且 prompt 工程和专项微调均无法根本解决该问题。

**[Refreshkv Updating Small Kv Cache During Long-Form Generation](llm_efficiency/refreshkv_updating_small_kv_cache_during_long-form_generation.md)**

:   提出RefreshKV推理方法，通过在生成过程中周期性地在全KV缓存注意力和小KV缓存注意力之间交替，并基于全注意力步的注意力模式动态更新小KV缓存，在不永久丢弃任何token的前提下，实现与驱逐式方法相当的加速且大幅提升长文本生成任务性能。

**[Robust Utility-Preserving Text Anonymization Based On Large Language Models](llm_efficiency/robust_utility-preserving_text_anonymization_based_on_large_language_models.md)**

:   提出RUPTA框架，通过隐私评估器、效用评估器和优化器三个LLM组件协同工作，迭代编辑文本以实现防御LLM重识别攻击的同时保留下游任务效用，并通过DPO蒸馏将匿名化能力迁移到轻量模型。

**[Sam Decoding Speculative Decoding Via Suffix Automaton](llm_efficiency/sam_decoding_speculative_decoding_via_suffix_automaton.md)**

:   提出SAM-Decoding，利用后缀自动机（Suffix Automaton）对通用文本语料和当前文本序列进行最长后缀匹配来高效生成推测解码的草稿，平均O(1)时间复杂度，在Spec-Bench上比现有检索式方法快18%+，并可与EAGLE-2等方法互补组合进一步提速3.28%-11.13%。

**[Scaling Context Not Parameters Training A Compact 7B Language Model For Efficien](llm_efficiency/scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)**

:   提出 MegaBeam-Mistral-7B，一个支持 512K token 上下文长度的 7B 语言模型，通过四阶段渐进式训练、RoPE theta 调优、bfloat16 精度修复和 XLA 编译器内存优化等工程实践，使紧凑型模型在长上下文任务上达到甚至超越大参数模型（如 Llama-3.1-70B、GPT-4）的性能。

**[Sliding Windows Full Ranking](llm_efficiency/sliding_windows_full_ranking.md)**

:   本文系统研究了长上下文LLM在段落排序中的应用，提出用 full ranking（一次性排序所有段落）替代传统滑动窗口策略，并设计了多轮滑动窗口标签构造方法和重要性感知损失函数来微调 full ranking 模型，在效率提升约30-65%的同时实现了排序效果的全面超越。

**[Smarter Better Faster Longer A Modern Bidirectional Encoder For Fast Memory Effi](llm_efficiency/smarter_better_faster_longer_a_modern_bidirectional_encoder_for_fast_memory_effi.md)**

:   提出 ModernBERT，将现代 LLM 架构优化（RoPE、GeGLU、交替局部/全局注意力、unpadding）系统性地引入 encoder-only 模型，在 2T token 上训练并原生支持 8192 上下文长度，在分类和检索任务上全面超越 BERT/RoBERTa/DeBERTaV3，同时推理速度和显存效率大幅领先。

**[Spindlekv Layered Kv Cache](llm_efficiency/spindlekv_layered_kv_cache.md)**

:   SpindleKV 提出分层处理 KV cache 压缩的策略——深层使用注意力驱动的 token eviction（利用稀疏注意力），浅层使用基于相似性学习的 codebook 替换（利用 token 间高相似度），并解决了 GQA 兼容性问题，实现 50% KV cache 缩减而不损失性能。

**[Train Long Context Effectively](llm_efficiency/train_long_context_effectively.md)**

:   本文系统研究了如何通过持续预训练和监督微调（SFT）有效训练长上下文语言模型，提出了包括数据配比、训练长度缩放等一系列关键发现，最终训练出的 ProLong-8B 模型仅用 Llama-3.1 5% 的长上下文训练数据量即在 128K 长度上达到同规模最优性能。

**[What Are The Essential Factors In Crafting Effective Long Context Multi-Hop Inst](llm_efficiency/what_are_the_essential_factors_in_crafting_effective_long_context_multi-hop_inst.md)**

:   提出多智能体交互式多跳生成（MIMG）框架，通过质量验证、单跳问题生成、多问题采样和多跳合并四个模块，系统性地合成高质量长上下文多跳指令数据，训练后模型平均提升7.54%，甚至超越更大规模人工标注数据集。

---

## 🛡️ AI安全 { #ai_safety }

**[Cavgan Unifying Jailbreak And Defense Of Llms Via Generative Adversarial Attacks](ai_safety/cavgan_unifying_jailbreak_and_defense_of_llms_via_generative_adversarial_attacks.md)**

:   提出 CAVGAN 框架，利用生成对抗网络在 LLM 内部表示空间中同时学习越狱攻击（生成器）和安全防御（判别器），首次将攻防统一到同一框架中实现"攻防共进"，攻击成功率平均 88.85%，防御成功率平均 84.17%。

**[Centaur Bridging The Impossible Trinity Of](ai_safety/centaur_bridging_the_impossible_trinity_of.md)**

:   提出 Centaur 框架，融合随机置换矩阵和安全多方计算（SMPC）来打破隐私保护 Transformer 推理（PPTI）中的"不可能三角"——同时实现强隐私保护、5-30x 加速和明文级别推理精度。

**[Dialect Fairness Robustness](ai_safety/dialect_fairness_robustness.md)**

:   本文提出首个系统评估LLM在非标准方言（AAVE）推理任务中公平性与鲁棒性的研究，构建了包含1.2K+平行查询对的ReDial基准，发现几乎所有主流LLM在AAVE输入上表现出显著的性能下降和不公平。

**[Elba-Bench An Efficient Learning Backdoor Attacks Benchmark For Large Language M](ai_safety/elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)**

:   建立了 ELBA-Bench——一个涵盖 12 种攻击方法、18 个数据集和 12 个 LLM 的综合后门攻击基准，系统评估 PEFT 和无微调两种范式下 LLM 后门攻击的有效性和隐蔽性。

**[Ensemble Watermarks Llm](ai_safety/ensemble_watermarks_llm.md)**

:   提出集成水印方法，将文体特征（藏头词 acrostic + 感觉运动词 sensorimotor norms）与已有红绿水印组合，在 paraphrasing 攻击后三特征集成检测率达 95%，而单独红绿水印仅 49%。

**[Estimating Privacy Leakage Of Augmented Contextual Knowledge In Language Models](ai_safety/estimating_privacy_leakage_of_augmented_contextual_knowledge_in_language_models.md)**

:   本文提出context influence指标，基于差分隐私框架量化语言模型在解码时对增强上下文知识的隐私泄露程度，并系统分析了模型大小、上下文大小、生成位置等因素对隐私泄露的影响。

**[Fairi Tales Evaluation Of Fairness In Indian Contexts With A Focus On Bias And S](ai_safety/fairi_tales_evaluation_of_fairness_in_indian_contexts_with_a_focus_on_bias_and_s.md)**

:   发布 Indic-Bias——首个印度社会文化的大规模 LLM 公平性 benchmark，覆盖 85 个身份群体（种姓/宗教/地区/部落），20,000 个场景模板+3 类评估任务，14 个 LLM 评测揭示模型对达利特等边缘群体系统性负面偏见且 70%+ 案例中强化刻板印象。

**[Fairness Difference Awareness](ai_safety/fairness_difference_awareness.md)**

:   本文挑战了主流公平性研究中"对所有群体一视同仁即为公平"的假设，提出"差异意识"(Difference Awareness)概念，构建了包含8个基准共16k问题的评测套件，发现现有"最公平"的LLM在该维度上表现不佳，且现有去偏方法会适得其反。

**[From Tradeoff To Synergy A Versatile](ai_safety/from_tradeoff_to_synergy_a_versatile.md)**

:   提出SymMark共生水印框架，融合logits-based和sampling-based两类水印方法（串行/并行/混合三种策略），通过token熵和语义熵自适应选择水印策略，在可检测性、鲁棒性、文本质量和安全性上实现SOTA。

**[Gifi Gender Fairness](ai_safety/gifi_gender_fairness.md)**

:   提出 GIFI（Gender Inclusivity Fairness Index），一个多层次综合评估框架，涵盖代词识别、情感中立性、毒性、反事实公平性、刻板印象关联、职业公平性和推理性能一致性七个维度，在 22 个 LLM 上系统评估二元与非二元性别的公平性。

**[Improved Unbiased Watermark For Large Language](ai_safety/improved_unbiased_watermark_for_large_language.md)**

:   提出 MCmark，一族基于多通道（Multi-Channel）的无偏水印算法，通过将词表分割为 $l$ 个段并在选中段内提升 token 概率来嵌入统计信号，在保持 LLM 原始输出分布的同时，可检测性比现有无偏水印提升超 10%。

**[Improving Fairness Of Large Language Models In Multi-Document Summarization](ai_safety/improving_fairness_of_large_language_models_in_multi-document_summarization.md)**

:   提出 FairPO（Fair Preference Optimization），通过扰动式偏好对生成和公平感知偏好调优，同时优化多文档摘要中的摘要级和语料级公平性。

**[Llm Watermark Distillation Robustness](ai_safety/llm_watermark_distillation_robustness.md)**

:   本文首次系统研究 LLM 水印在防止未授权知识蒸馏中的鲁棒性，提出三种水印去除攻击（无目标/有目标释义 + 推理时水印中和），发现有目标释义和水印中和可以彻底去除继承的水印，其中水印中和在保持知识迁移效率的同时实现零额外训练开销的水印去除。

**[Morphmark Adaptive Watermarking](ai_safety/morphmark_adaptive_watermarking.md)**

:   MorphMark 通过多目标权衡分析框架揭示了绿表概率 P_G 在水印效果与文本质量之间的关键作用，并据此提出自适应调整水印强度 r 的方法——当 P_G 高时增强水印、P_G 低时减弱水印，实现了在不依赖额外模型训练的前提下同时提升水印可检测性和文本质量。

**[Multi-Task Adversarial Attacks Against Black-Box Model With Few-Shot Queries](ai_safety/multi-task_adversarial_attacks_against_black-box_model_with_few-shot_queries.md)**

:   提出 CEMA（Cluster and Ensemble Multi-task Text Adversarial Attack）方法，通过训练"深层替代模型"将复杂的多任务黑盒攻击转化为单任务文本分类攻击，仅需约 100 次查询即可同时攻击分类、翻译、摘要、文生图等多种任务，并在 ChatGPT-4o、百度翻译、Stable Diffusion 等商用模型上验证了有效性。

**[Privacibench Evaluating Privacy With Contextual Integrity](ai_safety/privacibench_evaluating_privacy_with_contextual_integrity.md)**

:   提出 PrivaCI-Bench，基于 Contextual Integrity 理论构建了目前最大的上下文隐私评估基准（154K 实例），涵盖真实法院案例、隐私政策和 EU AI Act 合规检查器合成数据，评估 LLM 在 HIPAA/GDPR/AI Act 下的法律合规能力。

**[Private Memorization Editing Turning Memorization Into A Defense To Strengthen D](ai_safety/private_memorization_editing_turning_memorization_into_a_defense_to_strengthen_d.md)**

:   提出 PME（Private Memorization Editing），将 LLM 的记忆化特性从安全弱点转化为防御手段，通过编辑 Feed Forward 层参数来移除已记忆的个人身份信息（PII），实现无需重训的隐私保护。

**[Quantifying Misattribution Unfairness In Authorship Attribution](ai_safety/quantifying_misattribution_unfairness_in_authorship_attribution.md)**

:   本文提出MAUI_k指标量化作者归因系统中"错误归因不公平性"——某些作者系统性地更容易被误判为可疑作者，并发现这种不公平与作者嵌入在向量空间中距质心的距离高度相关。

**[Robust And Minimally Invasive Watermarking For Eaas](ai_safety/robust_and_minimally_invasive_watermarking_for_eaas.md)**

:   提出 ESpeW（Embedding-Specific Watermark），一种嵌入特异性水印方法，通过在每个嵌入向量的不同位置注入独特水印，实现对 Embeddings as a Service (EaaS) 的鲁棒版权保护，抵抗各种水印移除攻击且对嵌入质量的影响小于 1%。

**[Robust Data Watermarking In Language Models By Injecting Fictitious Knowledge](ai_safety/robust_data_watermarking_in_language_models_by_injecting_fictitious_knowledge.md)**

:   提出一种基于虚构知识（Fictitious Knowledge）的数据水印方法，通过在训练数据中注入虚构但合理的实体及其属性描述，实现对 LLM 训练数据所有权的可追溯验证，水印抗数据预处理过滤且支持黑盒 QA 验证。

**[Sandcastles Watermarking Impossibility](ai_safety/sandcastles_watermarking_impossibility.md)**

:   本文通过大规模实验和人类评估挑战了 "Watermarks in the Sand" (WITS) 的理论不可能性结论：证明随机游走攻击的两个关键假设在实践中不成立——混合(mixing)速度极慢（100% 的攻击文本仍可追溯原始来源）且质量预言机(quality oracle)不可靠（仅 77% 准确率），自动攻击仅 26% 成功率，人类质量审核后降至 10%。

**[Speechfake A Largescale Multilingual Speech Deepfake](ai_safety/speechfake_a_largescale_multilingual_speech_deepfake.md)**

:   构建 SpeechFake，目前最大的语音深度伪造检测数据集——超 300 万样本、3000+ 小时、40 种生成工具（含最新 TTS/VC/Neural Vocoder）、46 种语言，在自有及未见测试集上展现强基线性能。

**[Tip Iceberg Adversarial Attacks](ai_safety/tip_iceberg_adversarial_attacks.md)**

:   本文提出 Task-in-Prompt (TIP) 攻击——一类通过在 prompt 中嵌入序列到序列任务（如密码解码、谜语、代码执行）来间接生成违禁内容的新型越狱攻击类别，并构建 PHRYGE benchmark 系统评估，证明该攻击可成功绕过 GPT-4o、LLaMA 3.2 等六种 SOTA LLM 的安全防护。

**[Towards Fairness Assessment Of Dutch Hate Speech Detection](ai_safety/towards_fairness_assessment_of_dutch_hate_speech_detection.md)**

:   本文系统评估了荷兰语仇恨言论检测模型的反事实公平性，提出四种反事实数据生成方法（LLMdef、LLMlist、SLL、MGS），并通过在 BERTje 模型上微调验证了反事实数据增强对模型性能和公平性的改进效果。

**[Tug Of War Fairness Privacy](ai_safety/tug_of_war_fairness_privacy.md)**

:   发现 LLM 通过 SFT 增强隐私意识会显著降低公平性意识（trade-off），提出无训练方法 SPIN（抑制公平-隐私耦合神经元），基于信息论解耦两种意识，在 Qwen2-7B 上同时提升公平性 12.2% 和隐私意识 14.0%。

**[Watermark Segment Detection](ai_safety/watermark_segment_detection.md)**

:   提出两种高效方法（Geometric Cover Detector 和 Adaptive Online Locator）用于在长文混合来源文本中检测和精确定位水印片段，时间复杂度从 O(n²) 降至 O(n log n)，在三种主流水印技术上均显著优于 baseline。

**[Wet Eaas Watermark](ai_safety/wet_eaas_watermark.md)**

:   揭示了现有 EaaS 嵌入水印（EmbMarker/WARDEN）可被改写攻击绕过，提出 WET（线性变换水印），通过秘密循环矩阵对嵌入做线性变换注入水印，理论和实验证明其对改写攻击具有鲁棒性，验证 AUC 接近 100%。

---

## 📖 NLP理解 { #nlp_understanding }

**[A Comprehensive Graph Framework For Question Answering With Mode-Seeking Prefere](nlp_understanding/a_comprehensive_graph_framework_for_question_answering_with_mode-seeking_prefere.md)**

:   提出GraphMPA框架，通过构建基于通用相似度度量的层次化文档图实现全局文档理解，并引入mode-seeking偏好优化替代传统DPO实现更精准的人类偏好对齐，在6个QA数据集上全面超越现有RAG方法。

**[A Variational Approach For Mitigating Entity Bias In Relation Extraction](nlp_understanding/a_variational_approach_for_mitigating_entity_bias_in_relation_extraction.md)**

:   将变分信息瓶颈（VIB）应用于关系抽取的实体去偏——将实体映射到概率分布 $\mathcal{N}(\mu,\sigma)$，通过方差控制实体信息的压缩程度（高方差=更多依赖上下文），在 TACRED/REFinD/BioRED 三个领域（通用/金融/生物医学）的域内和域外设置上达到 SOTA，同时方差分析提供可解释性。

**[Adapting Psycholinguistic Research For Llms Gender-Inclusive Language In A Coref](nlp_understanding/adapting_psycholinguistic_research_for_llms_gender-inclusive_language_in_a_coref.md)**

:   将心理语言学方法从法语适配到英语和德语，研究 LLM 如何处理性别包容性语言——发现英语 LLM 基本保持先行词性别一致但内含男性默认偏见（不愿用 they 单数），德语 LLM 男性偏见更强烈（压倒所有性别中性化策略），但德语性别包容形式确实增加了女性/中性性别的出现概率。

**[Analyzing Political Bias In Llms Via Target-Oriented Sentiment Classification](nlp_understanding/analyzing_political_bias_in_llms_via_target-oriented_sentiment_classification.md)**

:   提出基于目标导向情感分类（TSC）不一致性的 LLM 政治偏见分析新方法——在 450 个政治句子中插入 1319 名不同政治光谱/人口特征的政治家名字，用 7 个模型×6 种语言预测情感，定义熵基不一致性指标量化预测变异性，发现所有模型均存在显著偏见（左翼正面/极右翼负面），大模型偏见更强且更一致，用虚构名字替换可部分缓解。

**[Automatic Generation Of Inference Making Questions For Reading Comprehension Ass](nlp_understanding/automatic_generation_of_inference_making_questions_for_reading_comprehension_ass.md)**

:   开发了一套阅读理解推理题分类法（代词桥接/文本连接/填补空白），用 GPT-4o few-shot 提示自动生成针对特定推理类型的多项选择题；93.8% 的题目质量合格，但仅 42.6% 准确匹配目标推理类型，说明 LLM 在精确推理能力控制上仍有不足。

**[Belle A Bi-Level Multi-Agent Reasoning Framework For Multi-Hop Question Answerin](nlp_understanding/belle_a_bi-level_multi-agent_reasoning_framework_for_multi-hop_question_answerin.md)**

:   提出 BELLE 双层多智能体辩论框架，先将多跳问题分类为四种类型，再通过双层辩论机制（第一层正反方辩论 + 第二层快/慢辩论者监督）动态规划 CoT、单步检索、迭代检索等算子的组合方案，实现面向问题类型的自适应多跳推理。

**[Bookcoref Book Scale](nlp_understanding/bookcoref_book_scale.md)**

:   提出首个书级别共指消解基准BookCoref，通过角色链接+LLM过滤+窗口扩展的自动标注管线，在50本完整小说上生成高质量银标注数据，平均文档长度超过20万tokens。

**[Calmqa Cultural Multilingual Qa](nlp_understanding/calmqa_cultural_multilingual_qa.md)**

:   构建了首个多语言长文本问答数据集 CaLMQA（51.7K 问题，23 种语言），通过无翻译方式收集文化特异性问题，发现 LLM 回答文化特异性问题的事实性（45-52%）显著低于文化无关问题（64-71%），低资源语言表现尤其差。

**[Can Llms Reliably Simulate Real Students Abilities In Mathematics And Reading Co](nlp_understanding/can_llms_reliably_simulate_real_students_abilities_in_mathematics_and_reading_co.md)**

:   利用项目反应理论(IRT)将11个LLM与真实学生放在同一能力量表上评估，发现在无引导情况下强模型远超学生平均水平，而"扮演某年级学生"的提示虽能改变表现，但**没有任何模型-提示组合**能在所有学科和年级上可靠模拟平均学生。

**[Disambiguate First Parse Later Generating Interpretations For Ambiguity Resoluti](nlp_understanding/disambiguate_first_parse_later_generating_interpretations_for_ambiguity_resoluti.md)**

:   提出"先消歧、后解析"的模块化方法，利用LLM生成默认解释并训练专门的infilling模型补全缺失解释，将歧义自然语言问题转化为多个明确解释后再分别进行SQL解析。

**[Dot Absa Template](nlp_understanding/dot_absa_template.md)**

:   提出 Dynamic Order Template（DOT）方法用于生成式方面级情感分析——为每个实例动态创建最优的预测模板顺序（只含必要的视角），在 ASQP 和 ACOS 数据集上提升 F1 的同时显著减少推理时间。

**[Embqa Embedding Odqa](nlp_understanding/embqa_embedding_odqa.md)**

:   EmbQA 提出嵌入级 ODQA 框架，用轻量线性层和无监督对比学习优化查询表示实现段落重排序，并引入基于序统计量的探索性嵌入扩展候选答案多样性，配合熵选择机制自动选答，在 4 个 ODQA 数据集上以更低计算成本超越 SuRe 等 prompt 级方法。

**[End-To-End Dialog Neural Coreference Resolution Balancing Efficiency And Accurac](nlp_understanding/end-to-end_dialog_neural_coreference_resolution_balancing_efficiency_and_accurac.md)**

:   提出一个端到端神经共指消解系统，通过结合上下文嵌入、层次化注意力机制和优化策略（剪枝/量化），在OntoNotes等基准数据集上实现效率与准确率的平衡，SpanBERT达到87.3 F1。

**[Generalized Open Relation Extract](nlp_understanding/generalized_open_relation_extract.md)**

:   提出 MixORE 框架，在更通用的 Open Relation Extraction 设定下（无标注数据同时包含已知和新颖关系，且不做长尾或预分割假设），通过 Semantic Autoencoder 检测新关系 + 开放世界半监督联合学习，在 FewRel/TACRED/Re-TACRED 上全面超越 SOTA。

**[Generating Diverse Training Samples For Relation Extraction With Large Language ](nlp_understanding/generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)**

:   研究如何用LLM为关系抽取（RE）生成高质量且多样化的训练样本，提出基于ICL的逐条生成策略和基于DPO的多样性微调方法，生成的训练数据可有效提升few-shot RE模型性能。

**[Hierarchical Retrieval With Evidence Curation For Open-Domain Financial Question](nlp_understanding/hierarchical_retrieval_with_evidence_curation_for_open-domain_financial_question.md)**

:   HiREC 提出分层检索与证据策展框架，先检索相关文档再从中选取段落，并通过过滤无关段落 + 自动生成补充查询来补全缺失信息，在包含 14.5 万篇 SEC 文档的 LOFin 基准上相比最优 RAG 基线提升 13%+ 答案准确率。

**[Iquest An Iterative Question-Guided Framework For Knowledge Base Question Answer](nlp_understanding/iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)**

:   iQUEST 提出迭代式子问题引导框架，在每一步推理中动态生成当前可解答的子问题以维持推理方向，并结合 GNN 聚合二跳邻居语义信息实现"前瞻性"实体探索，在 CWQ、WebQSP、WebQuestions、GrailQA 四个基准上取得 SOTA 或接近 SOTA 的性能，且无需微调 LLM。

**[Multi-Hop Reasoning For Question Answering With Hyperbolic Representations](nlp_understanding/multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)**

:   通过在 T5 编码器-解码器模型中简单插入一层 Poincaré 双曲层，以最小改动将欧几里得嵌入映射到双曲空间进行多跳推理，实验在四个数据集上一致性地超越欧几里得对照组，并证明了基于 δ-hyperbolicity 初始化曲率的有效性以及双曲空间在层次结构更强的数据集上优势更明显。

**[Neusym Rag Pdf Qa](nlp_understanding/neusym_rag_pdf_qa.md)**

:   NeuSym-RAG 提出了一个混合神经-符号检索框架，将 PDF 文档通过多视角分块解析同时存入关系数据库和向量库，LLM Agent 通过可执行动作（SQL 查询 + 向量检索 + 查看图片等）迭代式交互检索，在学术论文 QA 上比经典 RAG 提升 17.3%。

**[Qqsum A Novel Task And Model Of Quantitative Query-Focused Summarization For Rev](nlp_understanding/qqsum_a_novel_task_and_model_of_quantitative_query-focused_summarization_for_rev.md)**

:   提出 QQSUM 任务和 QQSUM-RAG 框架，通过 KP 导向检索与聚类、Next-KP-Generation 训练策略，从产品评论中生成包含多元观点及其流行度量化的 Key Point 摘要，解决传统 PQA 只输出单一视角答案的问题。

**[Recursive Question Understanding For Complex Question Answering Over Heterogeneo](nlp_understanding/recursive_question_understanding_for_complex_question_answering_over_heterogeneo.md)**

:   提出 ReQAP 方法，通过递归问题分解构建可执行算子树，在结构化+非结构化的异构个人数据上实现复杂问答，支持端侧轻量部署。

**[Rescore Multihop Qa](nlp_understanding/rescore_multihop_qa.md)**

:   提出 ReSCORE，利用 LLM 生成的文档-问题相关性（relevance）和文档-答案一致性（consistency）的联合概率作为伪标签，在迭代 RAG 框架中无监督训练 dense retriever，在三个多跳 QA 数据集上达到 SOTA。

**[Rethinking Semantic Parsing For Large Language Models Enhancing Llm Performance ](nlp_understanding/rethinking_semantic_parsing_for_large_language_models_enhancing_llm_performance_.md)**

:   针对"语义解析结果直接输入LLM反而降低性能"这一反直觉现象，提出SENSE——一种在prompt中嵌入语义提示（而非显式解析结果）的零样本方法，在GLUE理解任务、机器翻译、复述和简化等生成任务上一致性地提升了LLM表现。

**[Self-Critique Guided Iterative Reasoning For Multi-Hop Question Answering](nlp_understanding/self-critique_guided_iterative_reasoning_for_multi-hop_question_answering.md)**

:   提出 SiGIR 框架，通过端到端训练使模型具备迭代问题分解、检索、推理和自我评估能力，在推理阶段利用自我批评反馈引导 iteration-level beam search 选择最优推理路径，在三个多跳 QA 数据集上平均超越 SOTA 8.6%。

**[Sentiment Reasoning For Healthcare](nlp_understanding/sentiment_reasoning_for_healthcare.md)**

:   提出"情感推理"（Sentiment Reasoning）新任务，要求模型在预测医疗对话情感标签的同时生成解释理据，并构建了覆盖五种语言的 30K 样本多模态情感分析数据集，通过理据增强训练在分类准确率和 macro-F1 上均提升约 2%。

**[Syngraph A Dynamic Graph-Llm Synthesis Framework For Sparse Streaming User Senti](nlp_understanding/syngraph_a_dynamic_graph-llm_synthesis_framework_for_sparse_streaming_user_senti.md)**

:   提出SynGraph框架，在连续时间动态图上将稀疏用户分为mid-tail/long-tail/extreme三类，针对不同稀疏程度利用LLM合成增强数据（结合局部-全局图理解、高阶关系和画像生成），有效缓解流式评论情感分析中的数据稀疏问题。

---

## 🕸️ 图学习 { #graph_learning }

**[Beyond Completion A Foundation Model For General Knowledge Graph Reasoning](graph_learning/beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)**

:   提出 MERRY，一个统一处理 KG 内（零样本 KGC）和 KG 外（KGQA）推理任务的知识图谱基础模型，通过多视角条件消息传递 (CMP) 融合文本和结构信息，在 28 个数据集上超越现有方法。

**[Can Graph Neural Networks Learn Language](graph_learning/can_graph_neural_networks_learn_language.md)**

:   提出 Morpher，首个图-文多模态 prompt learning 范式——在冻结 GNN 和 LLM 参数的前提下，同时学习图 prompt 和文本 prompt + 跨模态投影器，用极弱文本监督（仅类别名几个词）将 GNN 表征对齐到 LLM 语义空间，首次实现 GNN 的 CLIP 式零样本图分类。

**[Claimpkg Enhancing Claim Verification Via Pseudo-Subgraph Generation With Lightw](graph_learning/claimpkg_enhancing_claim_verification_via_pseudo-subgraph_generation_with_lightw.md)**

:   提出 ClaimPKG 框架，通过轻量级专用 LLM 将文本声明转换为伪子图表示，再从知识图谱中检索相关子图作为证据，最终由通用 LLM 进行推理验证，在 FactKG 数据集上比 SOTA 高出 9%-12% 准确率。

**[Croppable Knowledge Graph Embedding](graph_learning/croppable_knowledge_graph_embedding.md)**

:   提出 MED 框架训练"可裁剪"知识图谱嵌入——一次训练同时优化 64 个不同维度的子模型（共享嵌入前缀），通过互学习、进化改进和动态损失权重，各维度子模型直接裁剪使用即超越独立训练和蒸馏方法，训练速度快 10 倍。

**[Cross-Document Contextual Coreference Resolution In Knowledge Graphs](graph_learning/cross-document_contextual_coreference_resolution_in_knowledge_graphs.md)**

:   提出基于知识图谱的跨文档共指消解方法，通过动态链接机制将文本实体提及与知识图谱节点关联，结合上下文嵌入和图消息传递推理提升跨文档实体识别的精度和召回率，在多个基准数据集上超越传统方法。

**[Disentangled Multi-Span Evolutionary Network Against Temporal Knowledge Graph Re](graph_learning/disentangled_multi-span_evolutionary_network_against_temporal_knowledge_graph_re.md)**

:   提出 DiMNet，通过多跨度演化策略和跨时间解耦机制，分离节点语义的活跃/稳定特征，显著提升时序知识图谱（TKG）外推推理性能，在四个基准数据集上取得 SOTA。

**[Extending Complex Logical Queries Uncertain Knowledge Graphs](graph_learning/extending_complex_logical_queries_uncertain_knowledge_graphs.md)**

:   提出在不确定知识图谱（Uncertain KG）上进行软查询（Soft Query）的新问题设定 SQUK，结合必要性（necessity）和重要性（importance）扩展一阶逻辑查询语义，并设计带校准的神经符号推理方法 SRC，避免前向推理中的级联错误。

**[Fast-And-Frugal Text-Graph Transformers Are Effective Link Predictors](graph_learning/fast-and-frugal_text-graph_transformers_are_effective_link_predictors.md)**

:   提出 Fast-and-Frugal Text-Graph (FnF-TG) Transformer，通过 Transformer 的自注意力机制统一编码文本描述和图结构（ego-graph），在归纳链接预测任务上以小 BERT 模型超越了使用大 BERT+MPNN 的 SOTA，同时首次扩展到完全归纳设置（关系也可归纳）。

**[Fidelis Faithful Reasoning In Large Language Model For Knowledge Graph Question ](graph_learning/fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)**

:   提出 FiDeLiS 框架，通过 Path-RAG 预选候选集缩小搜索空间 + 演绎验证beam search (DVBS) 逐步构建并验证推理路径，在无需训练的情况下提升 LLM 在知识图谱问答中的准确性和可解释性。

**[Graphcheck Breaking Long-Term Text Barriers With Extracted Knowledge Graph-Power](graph_learning/graphcheck_breaking_long-term_text_barriers_with_extracted_knowledge_graph-power.md)**

:   GraphCheck 提出一种图增强的事实核查框架，利用 LLM 从文档和声明中提取知识图谱三元组，通过 GNN 编码图结构并作为 soft prompt 注入冻结的 LLM 验证器，在单次推理调用中实现细粒度事实核查，在7个基准上平均提升 7.1%，且在医学领域表现出强泛化能力。

**[Graphnarrator](graph_learning/graphnarrator.md)**

:   提出GraphNarrator——首个为图神经网络生成自然语言解释的方法，通过将显著性图解释"语言化"为文本段落、用Expert Iteration迭代优化伪标签质量、最终蒸馏到端到端解释器模型，在三个数据集上生成的解释在忠实度、简洁性和人类偏好上均优于GPT-4o零样本解释。

**[Kg Llm Trustworthy Qa](graph_learning/kg_llm_trustworthy_qa.md)**

:   提出开放域知识图谱问答基准 OKGQA 及其扰动变体 OKGQA-P，通过统一的图引导检索-生成框架系统性地验证了 KG 增强可以有效降低 LLM 幻觉率（FActScore 提升约 20 个百分点），子图检索在各类查询上表现最优且对 KG 噪声具有鲁棒性。

**[Kg Rag Recommendation](graph_learning/kg_rag_recommendation.md)**

:   提出 K-RagRec 框架，将知识图谱（KG）中的结构化关系信息引入 LLM 推荐系统的 RAG 流程——从 KG 中检索高质量的结构化实体关系信息来增强推荐生成，解决纯文本 RAG 忽略结构关系和引入噪声的问题。

**[M3Hg Multimodal Multi-Scale And Multi-Type Node Heterogeneous Graph For Emotion ](graph_learning/m3hg_multimodal_multi-scale_and_multi-type_node_heterogeneous_graph_for_emotion_.md)**

:   提出 M3HG 模型，通过构建多模态多类型节点异构图来显式建模对话中的情感与原因上下文，并在句间和句内两个尺度上融合语义信息，实现多模态对话中情感-原因三元组的端到端提取。同时构建了首个中文多场景 MECTEC 数据集 MECAD。

**[Mrakl Multilingual Retrieval-Augmented Knowledge Graph Construction For Low-Reso](graph_learning/mrakl_multilingual_retrieval-augmented_knowledge_graph_construction_for_low-reso.md)**

:   将多语言知识图谱构建（mKGC）重新定义为 QA 任务，提出基于 RAG 的 mRAKL 系统，利用非结构化单语数据作为检索源来克服低资源语言中结构化数据匮乏的困难，在 Tigrinya 和 Amharic 两种低资源语言上显著超越已有方法。

**[Multimodal Transformers Are Hierarchical Modal-Wise Heterogeneous Graphs](graph_learning/multimodal_transformers_are_hierarchical_modal-wise_heterogeneous_graphs.md)**

:   从图论视角证明了多模态 Transformer（MulTs）本质上是层次化模态异质图（HMHG），并基于此提出 GsiT 模型，通过 Interlaced Mask 机制实现仅 1/3 参数的 All-Modal-In-One 融合，同时性能显著超越传统 MulTs。

**[Ontology-Guided Reverse Thinking Makes Large Language Models Stronger On Knowled](graph_learning/ontology-guided_reverse_thinking_makes_large_language_models_stronger_on_knowled.md)**

:   提出 ORT（Ontology-Guided Reverse Thinking），利用知识图谱本体结构从目标逆向构建标签推理路径，引导知识检索，显著提升 LLM 的知识图谱问答能力。

**[Paper 2401 14640](graph_learning/paper_2401_14640.md)**

:   提出 CAQA 基准，利用知识图谱自动生成包含四类归因类别（支持、部分支持、矛盾、无关）和四种推理复杂度的大规模问答归因评估数据集（161K 样本），系统性地评测了 25 种自动归因评估器的能力。

**[Predicate-Conditional Conformalized Answer Sets For Knowledge Graph Embeddings](graph_learning/predicate-conditional_conformalized_answer_sets_for_knowledge_graph_embeddings.md)**

:   提出 CondKGCP——基于谓词条件的 conformal prediction 方法用于知识图谱嵌入的不确定性量化，通过合并相似谓词增大校准集+双重校准（score+rank）减小预测集大小，在保证谓词级条件覆盖率的同时输出更紧凑的答案集，在多个KGE基准上优于5个baseline。

**[Rscf Relationsemantics Consistent Filter For Entity](graph_learning/rscf_relationsemantics_consistent_filter_for_entity.md)**

:   提出 RSCF 插件式 KGE 方法，通过共享仿射变换 + 根植实体变换 + 归一化三特征确保"语义相似的关系产生相似的实体变换"（关系语义一致性），在距离模型和张量分解模型上均显著超越 SOTA，并从理论和实验上验证了一致性保持率。

**[Simgrag Leveraging Similar Subgraphs For Knowledge Graphs Driven Retrieval-Augme](graph_learning/simgrag_leveraging_similar_subgraphs_for_knowledge_graphs_driven_retrieval-augme.md)**

:   提出 SimGRAG 方法，通过"查询→模式图→子图"两阶段对齐策略，利用 LLM 将查询转化为图模式，再用图语义距离（GSD）度量在知识图谱中高效检索语义最相似的子图，实现即插即用的 KG 驱动 RAG，在问答和事实验证任务上超越所有现有方法。

**[The Role Of Exploration Modules In Small Language Models For Knowledge Graph Comp](graph_learning/the_role_of_exploration_modules_in_small_language_models_for_knowledge_graph_comp.md)**

:   本文发现小语言模型（SLM）在知识图谱问答中的性能瓶颈在于图探索（exploration）阶段，提出用轻量级检索模块（SentenceBERT/GTR）替代 SLM 自身进行知识图谱遍历，显著提升了 SLM 在 KGQA 任务上的表现。

**[The Role Of Exploration Modules In Small Language Models For Knowledge Graph Que](graph_learning/the_role_of_exploration_modules_in_small_language_models_for_knowledge_graph_que.md)**

:   本文发现小语言模型（SLM）在知识图谱问答（KGQA）中使用Think-on-Graph框架时，性能瓶颈在于KG探索阶段而非推理阶段，通过引入轻量级段落检索模块（SentenceBERT/GTR，~110M参数）替代SLM自身进行KG遍历，可显著提升SLM在KGQA上的表现。

---

## 🏥 医学图像 { #medical_imaging }

**[A Retrieval-Based Approach To Medical Procedure Matching In Romanian](medical_imaging/a_retrieval-based_approach_to_medical_procedure_matching_in_romanian.md)**

:   提出基于检索的罗马尼亚语医疗程序名称匹配架构——将诊所的非标准程序描述匹配到保险公司标准化术语表，比较 BM25 稀疏检索、mE5/RoBERT/BioClinicalBERT 密集嵌入及 RRF 混合方法，在 14 万+映射对上评估，度量学习微调后 mE5 表现最佳。

**[Afrimed Qa Pan African](medical_imaging/afrimed_qa_pan_african.md)**

:   构建了首个大规模泛非洲多专科医学问答基准 AfriMed-QA（15,275 题，来自 16 个国家 60+ 医学院，涵盖 32 个专科），评估 30 个 LLM 发现：大模型在非洲医疗问题上的准确率显著低于 USMLE，生物医学专用 LLM 反而不如通用模型，消费者盲评时更偏好 LLM 回答而非临床医生回答。

**[Automated Structured Radiology Report Generation](medical_imaging/automated_structured_radiology_report_generation.md)**

:   提出结构化放射学报告生成（SRRG）新任务，利用LLM将自由文本报告重构为标准化格式，同时引入55标签的SRR-BERT疾病分类模型和F1-SRR-BERT评估指标，解决传统报告生成中风格多样导致的生成与评估困难。

**[Auxiliary Patient Data Xray](medical_imaging/auxiliary_patient_data_xray.md)**

:   本文研究如何将急诊科患者数据（生命体征、药物、分诊信息等）整合到多模态语言模型中用于自动胸部X光报告生成，提出将异构表格数据、文本和图像转化为统一嵌入的方法，在MIMIC-CXR + MIMIC-IV-ED数据集上显著提升了报告的诊断准确性，超越了包括CXRMate-RRG24在内的多个基准模型。

**[Biore Llm Judge Evaluation](medical_imaging/biore_llm_judge_evaluation.md)**

:   本文首次系统研究了 LLM-as-Judge 在生物医学关系抽取评估中的表现，发现其准确率通常低于 50%，并提出结构化输出格式（JSON）和域适应技术来提升约 15% 的评估准确率。

**[Chexalign Preference Finetuning](medical_imaging/chexalign_preference_finetuning.md)**

:   CheXalign 提出了一种无需放射科医生反馈的自动化偏好数据生成管线，利用公开数据集中的参考报告和基于参考的评估指标（如 GREEN、BERTScore）构造偏好对，通过 DPO 等直接对齐算法对胸部X光报告生成模型进行偏好微调，在 MIMIC-CXR 上取得 SOTA CheXbert 分数。

**[Clinical Coding Eight Recommendations](medical_imaging/clinical_coding_eight_recommendations.md)**

:   这篇 position paper 通过对 MIMIC 数据集和现有自动化临床编码研究的深入分析，指出当前评估方法（如仅关注前50个高频编码、使用不恰当指标）与真实临床场景严重脱节，并提出八条具体建议来改进评估方法和研究方向。

**[Clinidial A Naturally Occurring Multimodal Dialogue Dataset For Team Reflection ](medical_imaging/clinidial_a_naturally_occurring_multimodal_dialogue_dataset_for_team_reflection_.md)**

:   构建了 CliniDial 数据集，收集自模拟临床手术中的自然对话，包含音频转录、双角度视频和患者生理信号等多模态数据，标注了团队反思行为编码，揭示了现有 LLM 在处理标签不均衡、自然对话交互和领域多模态数据方面的显著不足。

**[Cstrl Context-Driven Sequential Transfer Learning For Abstractive Radiology Repo](medical_imaging/cstrl_context-driven_sequential_transfer_learning_for_abstractive_radiology_repo.md)**

:   提出 CSTRL，一种基于顺序迁移学习的放射学报告摘要生成方法，通过优化的间隔句生成（GSG）预训练、Fisher 矩阵正则化防止灾难性遗忘，并结合知识蒸馏实现模型压缩，在 MIMIC-CXR 和 Open-I 数据集上大幅超越现有方法。

**[Enhancing Medical Dialogue Generation Through Knowledge Refinement And Dynamic P](medical_imaging/enhancing_medical_dialogue_generation_through_knowledge_refinement_and_dynamic_p.md)**

:   提出 MedRef，一种融合知识精炼机制和动态 Prompt 调整策略的医学对话系统，通过隐变量过滤无关知识图谱三元组、实体-行为联合预测、以及三元组过滤器和示例选择器动态构建系统 Prompt，在 MedDG 和 KaMed 两个基准上取得 SOTA 性能。

**[Evaluation Of Llms In Medical Text Summarization The Role Of Vocabulary Adaptati](medical_imaging/evaluation_of_llms_in_medical_text_summarization_the_role_of_vocabulary_adaptati.md)**

:   系统性基准研究发现 LLM 在高 OOV（词汇外词）和高新颖性医学文本摘要场景下性能显著下降，并通过多种词汇适配策略（MEDVOC、MEDVOC-LLM、ScafFix）证明即使 Llama-3.1（128K 词汇量）仍受过度分片问题困扰，词汇适配可带来显著改善。

**[Learning From Negative Samples In Biomedical Generative Entity Linking](medical_imaging/learning_from_negative_samples_in_biomedical_generative_entity_linking.md)**

:   提出 ANGEL 框架，首次在生成式生物医学实体链接（BioEL）中引入负样本训练，通过两阶段策略（正样本训练 + 负样本感知的偏好优化）显著提升模型区分表面形式相似但语义不同的实体的能力，在五个基准数据集上平均 top-1 准确率提升 1.7%。

**[Medbiorag Semantic Search And Retrieval-Augmented Generation For Biomedical Lite](medical_imaging/medbiorag_semantic_search_and_retrieval-augmented_generation_for_biomedical_lite.md)**

:   MedBioRAG 提出了一种结合语义搜索、文档检索和微调 LLM 的检索增强生成框架，在生物医学问答的文本检索、封闭式 QA 和长文本 QA 三类任务上全面超越 GPT-4o 基线和此前 SOTA。

**[Medbiorag Semantic Search And Retrieval-Augmented Generation With Large Language](medical_imaging/medbiorag_semantic_search_and_retrieval-augmented_generation_with_large_language.md)**

:   MedBioRAG 提出了一个集成语义搜索、文档检索和微调LLM的检索增强生成框架，用于生物医学问答任务，在文本检索（NFCorpus、TREC-COVID）、封闭式问答（MedQA、PubMedQA、BioASQ）和长文本问答四个维度的多个基准上均超越了先前SOTA和GPT-4o基线模型。

**[Multimed Multilingual Medical Speech Recognition Via Attention Encoder Decoder](medical_imaging/multimed_multilingual_medical_speech_recognition_via_attention_encoder_decoder.md)**

:   发布 MultiMed——首个多语言医学 ASR 数据集（150小时，5种语言，10种录制场景，16种口音），配套小到大规模的端到端 Whisper 模型基线，首次系统研究医学领域的多语言 ASR：单语 vs 多语微调、AED vs Hybrid 架构对比，发现多语联合训练在小模型上有收益但大模型上可能退化。

**[Oisa Radiology Report Gen](medical_imaging/oisa_radiology_report_gen.md)**

:   提出在线迭代自对齐（OISA）方法用于放射学报告生成——四阶段循环（自生成多样数据→自评估多目标偏好→自对齐多目标优化→自迭代进一步提升），无需额外人工标注即可迭代提升报告质量，在多个评估指标上达到 SOTA。

**[Omni Rag Medical](medical_imaging/omni_rag_medical.md)**

:   针对医疗 LLM 需要多类型多结构知识源（教科书/指南/论文/知识图谱等）的特殊需求，提出 MedOmniKB 多源知识库和 Source Planning Optimization 方法——让模型学会"该从哪个源检索什么信息"，优化后的小模型在多源医疗知识利用上达到 SOTA。

**[One Size Fits None Rethinking Fairness In Medical Ai](medical_imaging/one_size_fits_none_rethinking_fairness_in_medical_ai.md)**

:   本文在三个多模态医学预测任务（ICU死亡率、移植物失败、急诊分诊）上进行子群体性能分析，揭示聚合指标掩盖的群体间性能差异，主张将公平性与透明度紧密结合，通过常规化的子群体报告推动负责任的医学AI部署。

**[Radar Radiology Report Gen](medical_imaging/radar_radiology_report_gen.md)**

:   提出 Radar 框架，通过"内部知识提取+外部补充知识检索+聚合注入"三步策略增强放射学报告生成——先提取 LLM 已有的与专家分类一致的知识，再检索缺失的补充知识，最终聚合两者生成更准确的放射学报告，在 MIMIC-CXR/CheXpert-Plus/IU X-ray 三个数据集上超越 SOTA。

**[Redactor An Llm-Powered Framework For Automatic Clinical Data De-Identification](medical_imaging/redactor_an_llm-powered_framework_for_automatic_clinical_data_de-identification.md)**

:   提出 RedactX——一个全自动、多模态的临床数据去标识化框架，结合 LLM 多轮抽取、规则处理和检索式再词汇化，在 i2b2 数据集上实现了与专用商业系统可比的 F1（0.9646），同时优化了 token 使用效率。

**[Reflectool Clinical Agent](medical_imaging/reflectool_clinical_agent.md)**

:   ReflecTool 提出了一个反思感知的工具增强临床 Agent 框架，通过优化阶段积累成功轨迹和工具级经验，推理阶段检索相似案例并用验证器改进工具使用，在涵盖 18 个任务的 ClinicalAgent Bench 上超越纯 LLM 10+ 分、超越已有 Agent 方法 3 分。

**[Secret Semi-Supervised Clinical Trial Document Similarity Search](medical_imaging/secret_semi-supervised_clinical_trial_document_similarity_search.md)**

:   提出 SECRET，一种半监督临床试验协议相似性搜索方法，通过将临床试验文档转换为 Q/A 对表示，并结合局部（Q/A 级）和全局（试验级）对比学习来生成嵌入，在完整试验搜索的 recall@1 上相对最佳基线提升 78%。

**[Urca Biomedical Evidence Extraction](medical_imaging/urca_biomedical_evidence_extraction.md)**

:   本文提出 URCA（Uniform Retrieval Clustered Augmentation）框架，通过均匀检索+聚类+知识提取的 RAG 流程，从 RCT 研究全文中自动提取与临床问题相关的科学证据结论，在新构建的 CochraneForest 数据集上比最佳基线提升了 8.81% F1。

---

## 👥 社会计算 { #social_computing }

**[A Survey On Proactive Defense Strategies Against Misinformation In Large Languag](social_computing/a_survey_on_proactive_defense_strategies_against_misinformation_in_large_languag.md)**

:   系统综述 LLM 主动防御错误信息的策略——提出"三支柱"框架：(1) 知识可信度（训练数据质量+知识编辑+RAG）, (2) 推理可靠性（自对齐+解码策略）, (3) 输入鲁棒性（对抗攻击防御+输入净化）。127 种技术的分类映射，48 项基准研究的元分析显示主动防御比传统检测方法提升 42-63%。

**[Banstereoset A Dataset To Measure Stereotypical Social Biases In Llms For Bangla](social_computing/banstereoset_a_dataset_to_measure_stereotypical_social_biases_in_llms_for_bangla.md)**

:   构建 BanStereoSet，一个包含 1194 条填空式样本、覆盖 9 类偏见（种族/性别/宗教/职业/美貌/年龄/种姓/地区等）的孟加拉语刻板印象偏见数据集，用于评估多语言 LLM 在孟加拉语中的社会偏见，发现 GPT-4o 偏见最高，Mistral 最低。

**[Biasguard A Reasoning-Enhanced Bias Detection Tool For Large Language Models](social_computing/biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)**

:   提出 BiasGuard，通过显式推理公平性规范来检测 LLM 输出偏见：第一阶段用教师模型生成推理轨迹做 SFT 初始化，第二阶段用 DPO 强化推理质量，在 5 个数据集上超越分类器和 LLM-as-Judge 方法且降低过度公平误判。

**[Can Community Notes Replace Professional Fact-Checkers](social_computing/can_community_notes_replace_professional_fact-checkers.md)**

:   通过大规模分析 Twitter/X 的 Community Notes 数据（66.4 万条），发现社区笔记对专业事实核查的依赖远超此前报告（至少 5-7%），证明高质量社区审核离不开专业事实核查，尤其在涉及阴谋论和更广泛虚假叙事的高风险内容上。

**[Conspiracy Theories And Where To Find Them On Tiktok](social_computing/conspiracy_theories_and_where_to_find_them_on_tiktok.md)**

:   首个TikTok阴谋论系统性分析：通过官方API收集美国150万条长视频，利用标签富集和远程监督识别阴谋论内容（每月约1000条新视频），评估TikTok创作者激励计划的影响，并测试开源LLM（Llama3、Mistral、Gemma）在基于音频转录的阴谋论检测上的效果（精确率高达96%但整体水平与微调RoBERTa相当）。

**[Detection Of Human And Machine-Authored Fake News In Urdu](social_computing/detection_of_human_and_machine-authored_fake_news_in_urdu.md)**

:   本文提出了乌尔都语四分类假新闻检测任务（人类假/人类真/机器假/机器真），构建了首个乌尔都语机器生成新闻数据集，并提出层次化检测方法将四分类分解为机器文本检测和假新闻检测两个子任务，在域内和跨域设置中均优于基线。

**[Explicit Vs Implicit Investigating Social Bias In Large Language Models Through ](social_computing/explicit_vs_implicit_investigating_social_bias_in_large_language_models_through_.md)**

:   借鉴社会心理学中隐式联想测验（IAT）和自我报告评估（SRA），提出自反思评估框架系统研究 LLM 的显式和隐式偏见，发现 LLM 与人类一样存在显式-隐式偏见不一致——显式偏见轻微但隐式偏见强烈，且模型越大/对齐训练越多，这种不一致越严重。

**[Exploring Multimodal Challenges In Toxic Chinese Detection Taxonomy Benchmark An](social_computing/exploring_multimodal_challenges_in_toxic_chinese_detection_taxonomy_benchmark_an.md)**

:   这篇工作把中文毒性文本中的“形、音、义混合扰动”系统化为 3 类 8 种策略，构建了大规模扰动基准 CNTP，并证明当前中美主流 LLM 在这类中文多模态毒性检测上都明显不稳，而小样本 ICL / SFT 虽能抬高检出率，却容易把正常内容一起误杀。

**[Exploring The Impact Of Instruction-Tuning On Llms Susceptibility To Misinformat](social_computing/exploring_the_impact_of_instruction-tuning_on_llms_susceptibility_to_misinformat.md)**

:   首次系统研究指令微调如何影响 LLM 对虚假信息的易感性，发现指令微调使模型从偏信 assistant-role 转变为偏信 user-role，当虚假信息以独立 user-turn 呈现时易感性最高，揭示了指令微调的"副作用"。

**[Fairsteer Inference Time Debiasing For Llms With Dynamic Activation Steering](social_computing/fairsteer_inference_time_debiasing_for_llms_with_dynamic_activation_steering.md)**

:   提出 FairSteer，一种推理时去偏框架，通过轻量线性分类器检测激活中的偏见信号，再用对比 prompt 对计算的去偏转向向量（DSV）动态调整隐藏层激活，无需重训即可在多任务上有效缓解 LLM 的社会偏见。

**[Gg-Bbq German Gender Bias Benchmark For Question Answering](social_computing/gg-bbq_german_gender_bias_benchmark_for_question_answering.md)**

:   将英语BBQ偏见基准数据集的性别子集翻译为德语，经人工审校后创建GG-BBQ德语性别偏见评估基准，揭示了机器翻译在性别偏见评估数据集构建中的局限性，并评估了多个德语LLM的偏见表现。

**[Hateday Global Hate Speech](social_computing/hateday_global_hate_speech.md)**

:   HateDay 构建了首个全球代表性仇恨言论数据集——24 万条随机采样的 Twitter 推文覆盖 8 种语言和 4 个英语国家，揭示了学术数据集大幅高估了检测模型在真实场景中的表现，尤其对非欧洲语言检测能力极差。

**[How Does Misinformation Affect Large Language](social_computing/how_does_misinformation_affect_large_language.md)**

:   构建了目前最大的误信息评估基准 MisBench（1034 万条误信息），从知识冲突类型和文本风格两个维度系统分析 LLM 对误信息的行为和偏好，并提出 RtD 方法结合外部知识源提升误信息检测能力。

**[Implihatevid Video Hate](social_computing/implihatevid_video_hate.md)**

:   首次提出视频中隐性仇恨言论检测任务，构建2009个视频的ImpliHateVid数据集，并设计两阶段对比学习框架融合文本、图像、音频三模态特征。

**[Is Llm An Overconfident Judge Unveiling The Capabilities Of Llms In Detecting Of](social_computing/is_llm_an_overconfident_judge_unveiling_the_capabilities_of_llms_in_detecting_of.md)**

:   系统评估了多个 LLM 在攻击性语言检测中面对标注分歧时的表现，发现 LLM 在标注者高度一致的样本上表现优异（GPT-4o F1 85.24%）但在低一致度样本上骤降至 57.06%，且模型对不确定样本表现出严重的过度自信；进一步通过 few-shot 和指令微调实验证明，在训练中引入分歧样本可同时提升检测准确率和人-AI 对齐度。

**[Llm Label Propagation](social_computing/llm_label_propagation.md)**

:   提出 GLPN-LLM 框架，通过 mask-based 全局标签传播机制有效整合 LLM 生成的伪标签，解决了 LLM 伪标签直接组合效果不佳的问题，在 Twitter/PHEME/Weibo 三个数据集上全面超越 SOTA。

**[Llm Personalized Disinformation](social_computing/llm_personalized_disinformation.md)**

:   系统评估了 6 个主流 LLM 生成个性化虚假信息的能力，发现大多数 LLM 能生成高质量个性化虚假新闻，且个性化请求反而降低了安全过滤器的触发率（相当于一种 jailbreak），同时轻微降低了机器生成文本的可检测性。

**[Mdit-Bench Evaluating The Dual-Implicit Toxicity In Large Multimodal Models](social_computing/mdit-bench_evaluating_the_dual-implicit_toxicity_in_large_multimodal_models.md)**

:   提出"双模态隐式毒性"(dual-implicit toxicity)概念——仅当结合图文两个模态时才能被识别的偏见与歧视，构建了包含317K问题、12类23子类的MDIT-Bench基准，并通过长上下文越狱揭示了主流多模态大模型中大量可被激活的隐藏毒性。

**[Measuring Social Biases In Masked Language Models By Proxy Of Prediction Quality](social_computing/measuring_social_biases_in_masked_language_models_by_proxy_of_prediction_quality.md)**

:   提出了注意力加权的预测质量代理度量 Δpa 和 CRRA，在迭代掩码实验（IME）下评估 MLM 的社会偏见，并引入模型比较函数 BSRT 来估计重训练引入的偏见，发现所提方法比 CSPS、AUL、AULA 等现有方法更准确、更敏感。

**[Silencing Empowerment Allowing Bigotry Auditing The Moderation Of Hate Speech On](social_computing/silencing_empowerment_allowing_bigotry_auditing_the_moderation_of_hate_speech_on.md)**

:   对 Twitch 平台的自动化内容审核工具 AutoMod 进行大规模审计，发送超过 10.7 万条消息，发现 AutoMod 在最严格设置下仅能标记 22% 的仇恨内容，高度依赖侮辱性词汇作为检测信号，同时错误屏蔽高达 89.5% 的教育性/赋权性内容。

**[State Toxicn A Benchmark For Span-Level Target-Aware Toxicity Extraction In Chin](social_computing/state_toxicn_a_benchmark_for_span-level_target-aware_toxicity_extraction_in_chin.md)**

:   构建了首个中文 span 级仇恨言论检测数据集 STATE ToxiCN（8029 条帖子、9533 个四元组标注），提出 Target-Argument-Hateful-Group 四元组标注体系，并首次建立了中文仇恨俚语标注词典（830 条），系统评估了多种 LLM 在 span 级中文仇恨言论检测上的能力。

**[Taz2024Full Analysing German Newspapers For Gender Bias And Discrimination Acros](social_computing/taz2024full_analysing_german_newspapers_for_gender_bias_and_discrimination_acros.md)**

:   本文发布了 taz2024full——迄今最大的公开德语报纸语料库（180 万篇文章，1980-2024 年），并通过可扩展的分析管道展示了四十多年来德语新闻报道中性别表征的演变趋势。

**[Translate With Care Addressing Gender Bias Neutrality And Reasoning In Large Lan](social_computing/translate_with_care_addressing_gender_bias_neutrality_and_reasoning_in_large_lan.md)**

:   提出 Translate-with-Care (TWC) 数据集（3,950 条跨 6 种无性别语言的翻译挑战），系统揭示 GPT-4、Google Translate 等模型在无性别→有性别语言翻译中的性别偏见和推理错误，并通过微调 mBART-50 在偏见消除和翻译准确率上大幅超越闭源 LLM。

---

## 🔬 可解释性 { #interpretability }

**[A Dual-Perspective Nlg Meta-Evaluation Framework With Automatic Benchmark And Be](interpretability/a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)**

:   揭示传统 NLG 元评估的局限（人工评分平均聚合不合理、相关系数选择模糊），提出双视角元评估框架：全局视角（序数分类，评估粗粒度评级能力）+ 局部视角（相邻成对比较，评估细粒度区分能力），并引入基于可控错误注入的自动基准构建方法，在 16 个 LLM 上验证不同模型在两个视角上的能力分布差异显著。

**[Around The World In 24 Hours Probing Llm Knowledge Of Time And Place](interpretability/around_the_world_in_24_hours_probing_llm_knowledge_of_time_and_place.md)**

:   本文提出 GeoTemp 数据集（320k 提示，覆盖 289 个城市和 37 个时区），首次评估 LLM 联合时间和空间推理的能力，发现模型能独立处理时间计算和地理知识，但在需要结合两者时性能急剧下降。

**[Bias Attribution In Filipino Language Models Extending A Bias Interpretability M](interpretability/bias_attribution_in_filipino_language_models_extending_a_bias_interpretability_m.md)**

:   将信息论偏见归因分数指标扩展到黏着语（菲律宾语），通过对子词分数取均值来处理复杂词素结构，在 4 个多语言 PLM 上揭示菲律宾语模型的偏见由实体类主题词（人物/物品/关系）驱动，与英语中动作类主题词（犯罪/性行为）形成鲜明对比。

**[Cleme2 Gec Evaluation](interpretability/cleme2_gec_evaluation.md)**

:   本文提出 CLEME2.0，一种可解释的 GEC 参考评估指标，通过将编辑解耦为四类（正确纠正 TP、错误纠正 FPne、欠纠正 FN、过纠正 FPun）并结合编辑加权技术，在 GJG15 和 SEEDA 两个人工评判数据集上达到了与人工判断最高相关性的 SOTA 结果。

**[Cracking Factual Knowledge A Comprehensive Analysis Of Degenerate Knowledge Neur](interpretability/cracking_factual_knowledge_a_comprehensive_analysis_of_degenerate_knowledge_neur.md)**

:   本文从结构和功能两个角度对大语言模型中的退化知识神经元（DKN）进行了严格定义和系统研究，提出了神经拓扑聚类（NTC）方法来精确获取DKN，并通过34个实验验证了DKN与模型鲁棒性、可进化性和复杂性之间的关系。

**[Expert An Explainable Image Captioning Evaluation Metric With Structured Explana](interpretability/expert_an_explainable_image_captioning_evaluation_metric_with_structured_explana.md)**

:   本文提出 EXPERT，一种基于 VLM 微调的无参考图像描述评估指标，通过构建大规模结构化解释数据集并设计两阶段评估模板，在多个基准数据集上达到 SOTA 的同时，提供基于流畅度、相关性、描述性三个维度的高质量结构化解释。

**[Irt Router Multi Llm](interpretability/irt_router_multi_llm.md)**

:   IRT-Router 借鉴心理测量学的项目反应理论（IRT），将 LLM 视为"考生"、query 视为"考题"，学习多维能力向量和难度/区分度参数实现可解释的多 LLM 路由，在 OOD 场景下达 87%+ 准确率且成本仅为 GPT-4o 的 1/30。

**[Llama See Llama Do Entrainment](interpretability/llama_see_llama_do_entrainment.md)**

:   本文发现并定义了"上下文夹带"(contextual entrainment)现象——LLM会对上下文中出现过的任意token赋予更高概率，并通过可微掩码方法定位了负责该现象的entrainment heads，关闭这些头后可显著抑制干扰效应。

**[Mechanistic Interpretability Of Emotion Inference In Large Language Models](interpretability/mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)**

:   通过 probing、activation patching 和 generation steering 三种机制可解释性技术，发现 LLM 的情感表征功能性地定位于中间层的 MHSA 单元，并基于认知评估理论（appraisal theory）证明这些表征具有心理学合理性，成功通过干预评估概念（如 self-agency、pleasantness）引导情感输出。

**[Normalized Aopc Faithfulness Metrics](interpretability/normalized_aopc_faithfulness_metrics.md)**

:   本文揭示了广泛使用的 AOPC（扰动曲线下面积）忠实度指标在跨模型比较时会产生误导性结论（因为不同模型的 AOPC 上下界差异巨大），提出 Normalized AOPC (NAOPC) 通过 min-max 归一化消除模型间的不可比性，实验表明归一化可以根本性地改变模型忠实度排名。

**[Output Centric Interpretability](interpretability/output_centric_interpretability.md)**

:   提出两种以输出为中心的特征描述方法（VocabProj 和 TokenChange），弥补现有基于输入激活的自动可解释性流程只能捕获"什么激活了特征"而忽略"特征如何影响输出"的缺陷，并证明输入+输出方法的集成能生成最忠实的特征描述。

**[Position-Aware Automatic Circuit Discovery](interpretability/position-aware_automatic_circuit_discovery.md)**

:   提出位置感知的边归因修补方法（PEAP）和数据集 Schema 机制，解决了自动电路发现中忽略位置信息导致的抵消效应和重要性高估问题，实现了更小且更忠实的电路发现。

**[Prism Political Bias Embeddings](interpretability/prism_political_bias_embeddings.md)**

:   提出PRISM框架——首个专为政治偏见嵌入设计的方法，通过争议性话题偏见指标挖掘和Cross-Encoder政治偏见评分两阶段，生成可解释的偏见感知嵌入，在政治偏见分类上超越通用文本嵌入模型。

**[Probing Subphonemes In Morphology Models](interpretability/probing_subphonemes_in_morphology_models.md)**

:   本文提出了一种语言无关的探测方法，研究在形态学变形任务上训练的 Transformer 模型如何隐式习得音韵特征，发现局部特征（如末辅音清化）在音素嵌入中编码良好，而长距离依赖（如元音和谐）在编码器层的上下文化表示中更显著。

**[Probing The Geometry Of Truth Consistency And Generalization Of Truth Directions](interpretability/probing_the_geometry_of_truth_consistency_and_generalization_of_truth_directions.md)**

:   系统性研究LLM内部"真值方向"(truth direction)的一致性与泛化能力，发现只有能力较强的模型才稳定展现一致的真值方向，且基于简单原子陈述训练的真实性探针可泛化至逻辑变换、问答任务和上下文知识场景。

**[Reasoning Circuits In Language Models A Mechanistic Interpretation Of Syllogisti](interpretability/reasoning_circuits_in_language_models_a_mechanistic_interpretation_of_syllogisti.md)**

:   用机械可解释性技术（激活补丁 + Logit Lens + 电路消融）发现语言模型中实现三段论推理的完整电路：三阶段机制——长归纳偏差→中间项抑制（h11.10）→传递项移动，该电路在符号输入上既充分又必要，可迁移到自然语言输入，且跨 GPT-2/Pythia/LLaMA/Qwen 四种架构存在兼容模式。

**[Retrieve To Explain Drug Target Identification](interpretability/retrieve_to_explain_drug_target_identification.md)**

:   提出 R2E (Retrieve to Explain)，一种基于检索的架构，通过从文献语料库中检索证据来评分和排序所有候选答案，并利用 Shapley 值将预测忠实地归因到支撑证据，在药物靶点识别任务上超越了遗传学基线和 GPT-4 基线。

**[Safety Is Not Only About Refusal Reasoning-Enhanced Fine-Tuning For Interpretabl](interpretability/safety_is_not_only_about_refusal_reasoning-enhanced_fine-tuning_for_interpretabl.md)**

:   提出 Rational 框架，通过推理增强微调让 LLM 在回答前进行显式的安全推理（分析意图、伦理和潜在危害），而非依赖僵硬的拒绝启发式，在保持有用性的同时显著提升对推理层面对抗攻击的鲁棒性。

**[Separating Tongue From Thought Activation Patching Reveals Language-Agnostic Con](interpretability/separating_tongue_from_thought_activation_patching_reveals_language-agnostic_con.md)**

:   通过激活修补实验，首次提供了因果性证据证明大语言模型内部存在与语言解耦的概念表示——模型先确定输出语言，再确定概念，并且跨语言平均的概念表示不仅不损害翻译能力，反而能提升翻译准确率。

**[Shortcut Neuron Eval](interpretability/shortcut_neuron_eval.md)**

:   提出通过对比分析和因果分析定位污染模型中的"捷径神经元"（shortcut neurons），并通过 activation patching 抑制这些神经元，实现更可信的 LLM 评估，与 MixEval 的 Spearman 相关系数超过 0.95。

**[The Anatomy Of Evidence An Investigation Into Explainable Icd Coding](interpretability/the_anatomy_of_evidence_an_investigation_into_explainable_icd_coding.md)**

:   本文对 MDACE 数据集和当前可解释 ICD 编码系统进行了深入的应用导向分析，揭示了人工标注证据与代码描述的重叠规律、证据在文档中的分布特征，并提出了新的匹配度量来评估模型解释的实用性。

**[Towards Explainable Temporal Reasoning In Large Language Models A Structure-Awar](interpretability/towards_explainable_temporal_reasoning_in_large_language_models_a_structure-awar.md)**

:   提出 GETER 框架，通过轻量级 Structure-Text Adapter 将时序知识图谱的结构信息注入 LLM，使模型在时序推理任务中既能给出准确预测又能生成可解释的推理说明。

---

## ✍️ 文本生成 { #nlp_generation }

**[A Representation Level Analysis Of Nmt Model Robustness To Grammatical Errors](nlp_generation/a_representation_level_analysis_of_nmt_model_robustness_to_grammatical_errors.md)**

:   从模型内部表示视角分析 NMT 对语法错误的鲁棒性机制——发现编码器先"检测"语法错误（GED 探测精度在前半层上升），再"纠正"它（不正确词的表示向正确形式靠拢），并识别出"鲁棒性头"（Robustness Heads）——特定注意力头关注可解释的语言单元以修正错误表示，微调后模型更多依赖这些头。

**[An Empirical Study Of Manytomany Summarization](nlp_generation/an_empirical_study_of_manytomany_summarization.md)**

:   首次系统研究LLM在多对多摘要（M2MS）任务上的表现，整合8个数据集构建涵盖5个领域6种语言的47.8K样本基准，评测18个LLM发现零样本LLM可媲美微调传统模型，指令微调后显著超越，但事实性问题仍是关键瓶颈。

**[Atgen A Framework For Active Text Generation](nlp_generation/atgen_a_framework_for_active_text_generation.md)**

:   推出 ATGen——首个将主动学习（AL）与文本生成（NLG）任务桥接的综合框架，支持人类标注和 LLM 自动标注，集成 SOTA AL 策略和实验设计方法，提供 Web 标注界面和统一基准平台。实验证明 AL 显著减少人工标注时间和 LLM API 调用成本。

**[Cocolex Legal Text Gen](nlp_generation/cocolex_legal_text_gen.md)**

:   提出 CoCoLex，一种无需训练的解码策略，通过置信度引导的动态插值将模型词表分布与上下文复制分布结合，鼓励从检索上下文中直接复制 token，在五个法律文本生成基准上显著提升生成文本对源文档的忠实性。

**[Context-Aware Hierarchical Merging For Long Document Summarization](nlp_generation/context-aware_hierarchical_merging_for_long_document_summarization.md)**

:   提出上下文感知的层次合并（CAHM）方法，通过在层次合并摘要过程中引入源文档的相关上下文（抽取/检索/引用三种方式），有效缓解 LLM 在超长文档（>100K tokens）摘要中的幻觉问题。

**[Dehumanizing Machines Anthropomorphic](nlp_generation/dehumanizing_machines_anthropomorphic.md)**

:   系统研究如何缓解文本生成系统的拟人化行为——编制基于文献和众包的干预手段清单，发展概念框架来表征干预空间、区分干预类型和评估干预效果，为减少用户对 AI 的过度依赖和情感依附提供理论和实证基础。

**[Doc Level Mbr Optimal Transport](nlp_generation/doc_level_mbr_optimal_transport.md)**

:   提出 MBR-OT，将最优传输（Wasserstein距离）引入最小贝叶斯风险（MBR）解码，实现用句子级效用函数评估文档级输出质量，在文档级机器翻译、文本简化和密集图像描述任务上显著优于标准 MBR 解码。

**[Dtcrs Dynamic Tree Construction For Recursive Summarization](nlp_generation/dtcrs_dynamic_tree_construction_for_recursive_summarization.md)**

:   提出 DTCRS 方法，根据文档结构和查询语义动态构建摘要树，通过问题分解和子问题引导聚类减少冗余摘要节点，在三个 QA 数据集上显著优于静态摘要树方法 RAPTOR。

**[Enhancing Text Editing For Grammatical Error Correction Arabic As A Case Study](nlp_generation/enhancing_text_editing_for_grammatical_error_correction_arabic_as_a_case_study.md)**

:   本文提出一种无需语言特定编辑集的通用文本编辑方法（SWEET），通过数据驱动的编辑标签自动提取和压缩策略，首次成功将文本编辑范式应用于阿拉伯语语法纠错，在多个基准上达到SOTA且推理速度提升6倍以上。

**[Event Graph Bias Mitigation Summarization](nlp_generation/event_graph_bias_mitigation_summarization.md)**

:   提出基于多文档事件关系图的中立化摘要方法，通过构建包含文内事件关系（时间/因果/子事件/共指）、跨文档事件共指和事件级道德观点的知识图，以图文本化（硬提示）和图提示调优（软提示）两种方式引导 LLM 生成去偏见的中立摘要，在内容保持和偏见消除上均优于基线。

**[Gec-Metrics A Unified Library For Grammatical Error Correction Evaluation](nlp_generation/gec-metrics_a_unified_library_for_grammatical_error_correction_evaluation.md)**

:   提出 gec-metrics 统一库，将 10 种语法纠错 (GEC) 评估指标整合到统一接口中，并提供元评估功能，解决了现有 GEC 评估实现碎片化、不可复现、难以扩展的问题。

**[Impara-Ged Grammatical Error Detection Is Boosting Reference-Free Grammatical Er](nlp_generation/impara-ged_grammatical_error_detection_is_boosting_reference-free_grammatical_er.md)**

:   在 IMPARA 的质量估计器构建之前，增加一步语法错误检测（GED）预训练，同时去掉失效的相似度估计器，使无参考 GEC 评估在 SEEDA 上达到句子级最高相关性。

**[Personalized Text Generation With Contrastive Activation Steering](nlp_generation/personalized_text_generation_with_contrastive_activation_steering.md)**

:   提出 StyleVector——一个无需训练的个性化文本生成框架，通过对比用户真实响应与模型生成的无风格响应之间的隐层激活差异来提取"风格向量"，在推理时通过简单的线性激活干预引导 LLM 生成符合用户写作风格的文本，在 LaMP 和 LongLaMP 基准上实现 8% 的相对提升，同时将存储需求降低至 PEFT 方法的 1/1700。

**[Persphere A Comprehensive Framework For Multi-Faceted Perspective Retrieval And ](nlp_generation/persphere_a_comprehensive_framework_for_multi-faceted_perspective_retrieval_and_.md)**

:   > 提出 PerSphere 基准数据集和 MURS（Multi-faceted perspective retrieval and summarization）任务，旨在从文档集中检索并全面总结争议性问题的多面向观点，并提出分层多智能体总结系统 HierSphere 来缓解长上下文和观点提取的挑战。

**[Rethinking Evaluation Metrics For Grammatical Error Correction Why Use A Differe](nlp_generation/rethinking_evaluation_metrics_for_grammatical_error_correction_why_use_a_differe.md)**

:   指出自动 GEC 评估与人类评估在聚合方式上的差距（人类用 TrueSkill 做成对比较后聚合，自动评估用平均/求和后排序），提出对所有自动指标统一使用 TrueSkill 聚合，在 SEEDA 基准上大幅提升多数指标与人类评估的相关性。

**[Tagrouter Learning Route To Llms Through Tags For Open-Domain Text Generation Ta](nlp_generation/tagrouter_learning_route_to_llms_through_tags_for_open-domain_text_generation_ta.md)**

:   这篇论文提出 TagRouter，用一个小型标签生成器把开放域文本生成请求先压缩成一组语义标签，再基于标签统计每个候选 LLM 的相对优势并进行路由，从而在不重新训练路由器的前提下，把多模型系统的接受率做得比单个大模型更高，同时显著降低推理成本。

**[Tell Dont Show Leveraging Language Models Abstractive Retellings To Model Litera](nlp_generation/tell_dont_show_leveraging_language_models_abstractive_retellings_to_model_litera.md)**

:   提出 Retell 方法：利用小型 LM 对文学段落进行抽象复述（abstractive retelling），

**[Theme-Explanation Structure For Table Summarization Using Large Language Models ](nlp_generation/theme-explanation_structure_for_table_summarization_using_large_language_models_.md)**

:   提出 Tabular-TX 管线，通过多步 CoT 推理实现深度表格理解、记者角色 prompt 生成清晰句子、并将输出结构化为 Theme（主题状语）+ Explanation（解释谓语）的格式，在韩语行政表格摘要基准上不依赖微调即实现 ROUGE-1 0.51 的最佳性能，显著超越微调和纯 ICL 方法。

**[Towards Better Open-Ended Text Generation A Multicriteria Evaluation Framework](nlp_generation/towards_better_open-ended_text_generation_a_multicriteria_evaluation_framework.md)**

:   针对开放式文本生成中多指标（coherence/diversity/perplexity）之间的权衡问题，提出三种互补的多准则评估方法——Extended Bradley-Terry 模型（序数排名）、Union-Free Generic Depth（允许不可比性的偏序）和 Q*Text（基数评估综合指标），在6个 LLM × 59种解码策略 × 180万+生成文本上验证，发现中等超参配置普遍优于极端配置，小模型+合理解码策略可匹敌大模型。

**[Unveiling Attractor Cycles In Large Language Models A Dynamical Systems View Of ](nlp_generation/unveiling_attractor_cycles_in_large_language_models_a_dynamical_systems_view_of_.md)**

:   本文从动力系统理论出发，发现LLM在连续释义（successive paraphrasing）过程中输出会收敛至稳定的2-周期吸引子循环，而非探索广阔的释义空间，揭示了LLM生成能力的固有局限性。

**[Video Text Summarization](nlp_generation/video_text_summarization.md)**

:   构建了 VISTA 数据集（18,599 个 AI 会议演讲视频-摘要对），首次系统性基准测试科学视频到文本摘要任务，并提出基于计划（plan-based）的框架，通过显式建模摘要结构来提升生成质量和事实一致性。

**[Writing Like Best Exemplar](nlp_generation/writing_like_best_exemplar.md)**

:   定义"基于范例的说明文生成"新任务——给定一篇关于源主题的范例文本，生成关于目标主题的说明文，提出 Recurrent Plan-then-Adapt（RePA）框架，通过逐段模仿规划+检索增强自适应生成+双记忆机制，在 Wikipedia/RoleEE/USNews 三个数据集上显著优于 GPT-4 和 o1 基线。

---

## 💻 代码智能 { #code_intelligence }

**[Benchmarking Long-Context Language Models On Long Code Understanding](code_intelligence/benchmarking_long-context_language_models_on_long_code_understanding.md)**

:   提出 LongCodeU 基准，从代码单元感知、单元内理解、单元间关系理解和长文档理解四个维度（8 个任务）系统评估长上下文语言模型的长代码理解能力，揭示了当代码长度超过 32K token 时 LCLM 性能急剧下降，远未达到其声称的 128K-1M 上下文窗口能力。

**[Codedpo Code Alignment](code_intelligence/codedpo_code_alignment.md)**

:   提出 CodeDPO，通过 PageRank 启发的自验证评分机制从自生成代码中构造高质量偏好对（93K 正确性 + 21K 效率），DPO 训练后在 8 个代码模型上 HumanEval 平均提升 10+ 分，同时提升代码执行效率 1.25-1.45×。

**[Codeif Benchmarking The Instruction-Following Capabilities Of Large Language Mod](code_intelligence/codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)**

:   提出 CodeIF，第一个系统性评估 LLM 在代码生成中指令遵循能力的基准，含 8 大类 50 个细粒度约束指令、4 种新评估指标，并对 35 个 SOTA 模型进行全面评估。

**[Codereviewqa The Code Review Comprehension Assessment For Large Language Models](code_intelligence/codereviewqa_the_code_review_comprehension_assessment_for_large_language_models.md)**

:   提出 CodeReviewQA 基准，将代码审查自动修正（ACR）任务分解为三个中间推理步骤——变更类型识别（CTR）、变更定位（CL）、解决方案识别（SI），各自设计为不同难度的多选题探测，在 900 个人工验证的高质量样例（9 种语言）上评测 72 个 LLM，揭示了模型在代码审查理解中的具体弱点。

**[Compileagent Automated Real-World Repo-Level Compilation With Tool-Integrated Ll](code_intelligence/compileagent_automated_real-world_repo-level_compilation_with_tool-integrated_ll.md)**

:   提出 CompileAgent，首个面向仓库级代码编译的 LLM Agent 框架，集成五种专用工具和流程化 Agent 策略，在 100 个 C/C++ 真实项目的 CompileAgentBench 上将编译成功率最高提升 71%，平均每个项目仅需 $0.22。

**[Coret Improved Retriever For Code Editing](code_intelligence/coret_improved_retriever_for_code_editing.md)**

:   提出 CoRet，一个面向代码编辑任务的稠密检索模型，通过整合代码语义、仓库文件层级结构和调用图依赖关系，并使用针对仓库级检索设计的对数似然损失函数，在 SWE-bench 和 Long Code Arena 上比现有模型的 Recall 至少提升 15 个百分点。

**[Dynacode A Dynamic Complexity-Aware Code Benchmark For Evaluating Large Language](code_intelligence/dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)**

:   提出 DynaCode，一个动态复杂度感知的代码生成基准，通过将代码问题按圈复杂度分类并用调用图（Call Graph）组合嵌套，生成约 1.89 亿个唯一问题，有效缓解数据污染并系统评估 LLM 在不同复杂度下的代码生成能力。

**[Etf An Entity Tracing Framework For Hallucination Detection In Code Summaries](code_intelligence/etf_an_entity_tracing_framework_for_hallucination_detection_in_code_summaries.md)**

:   提出 Entity Tracing Framework (ETF)，一种通过静态程序分析提取代码实体、再用 LLM 验证这些实体在生成摘要中是否被正确描述的幻觉检测框架，配合首创的 CodeSumEval 数据集（~10K样本），在代码摘要幻觉检测上达到 73% F1。

**[Exploracoder Advancing Code Generation For Multiple Unseen Apis Via Planning And](code_intelligence/exploracoder_advancing_code_generation_for_multiple_unseen_apis_via_planning_and.md)**

:   提出无需额外训练的 ExploraCoder 框架，通过任务规划将复杂多 API 编程问题分解为子任务，再通过链式 API 探索（CoAE）逐步实验并积累正确的 API 用法经验，在多 API 不可见库基准上 pass@10 绝对提升最高 17.28%。

**[Feabench Repo Code Gen](code_intelligence/feabench_repo_code_gen.md)**

:   提出 FEA-Bench——首个评估 LLM 在仓库级代码库中实现新特性（Feature Implementation）能力的基准，包含来自 83 个 GitHub 仓库的 1401 个任务实例，每个实例配有单元测试。最强模型 DeepSeek-R1 仅解决约 10% 的任务，揭示了仓库级增量开发对当前 LLM 的巨大挑战。

**[Galla Graph Aligned Large Language Models](code_intelligence/galla_graph_aligned_large_language_models.md)**

:   提出 GALLa，通过 GNN 编码代码的 AST/DFG 结构图并用跨模态适配器对齐到 LLM 嵌入空间，在微调时作为辅助任务注入代码结构信息，推理时丢弃 GNN 和 adapter 实现零额外开销，在 5 个代码任务 × 7 个基线 LLM（350M-14B）上持续提升。

**[Gift Gibbs Fine Tuning Code Gen](code_intelligence/gift_gibbs_fine_tuning_code_gen.md)**

:   提出 Gibbs Fine-Tuning（GiFT），受 Gibbs 采样启发，通过"代码→描述→代码"的迭代翻译从边际分布而非条件分布中采样自生成代码，结合困惑度引导的长尾数据选择，在 APPS+/MBPP+/CodeInsight 上比标准自训练提升最高 9.8%。

**[Mldebugging Towards Benchmarking Code Debugging Across Multi-Library Scenarios](code_intelligence/mldebugging_towards_benchmarking_code_debugging_across_multi-library_scenarios.md)**

:   本文提出 MLDebugging——首个面向**多库 Python 代码调试**的综合基准，涵盖 126 个 Python 库和 7 种 bug 类型（共 1175 个样本），系统评估主流开源和闭源 LLM 在多库调试场景下的能力，发现当前 LLM 在此任务上仍有很大提升空间。

**[Oasis Order-Augmented Strategy For Improved Code Search](code_intelligence/oasis_order-augmented_strategy_for_improved_code_search.md)**

:   提出OASIS方法，通过为负样本对引入基于序的相似度标签来捕捉代码语义中的细微差异，结合InfoNCE和CoSENT双重损失函数训练代码嵌入模型，在CoSQA、AdvTest和CodeSearchNet三个基准的NL2Code和Code2Code搜索任务上全面超越现有SOTA。

**[Personality Guided Code Gen](code_intelligence/personality_guided_code_gen.md)**

:   让 GPT-4o 为每个编程任务生成适配的 MBTI 人格类型和描述，再让 LLM 以该人格角色扮演程序员生成代码，在 28 个 LLM-数据集组合中 23 个取得 pass rate 提升，最高达 12.9%，且可与 CoT 等策略叠加使用。

**[Revisit Self-Debugging With Self-Generated Tests For Code Generation](code_intelligence/revisit_self-debugging_with_self-generated_tests_for_code_generation.md)**

:   系统性地研究了使用 LLM 自生成测试进行自调试（self-debugging）的效果，发现基于后执行信息的自调试在基础编程问题上反而降低性能（因自生成测试偏差），但基于执行中间状态（in-execution）的自调试可有效规避该偏差，在基础和竞赛题上均有提升。

**[Texpert A Multi-Level Benchmark For Evaluating Latex Code Generation By Llms](code_intelligence/texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)**

:   提出TeXpert——首个系统评估LLM从自然语言指令生成科学文档LaTeX代码能力的多难度级别基准，包含440个高质量样本（Simple/Average/Hard三级），在9个开闭源LLM上的评估揭示了LaTeX生成是LLM的显著短板（Hard任务准确率普遍低于17.5%），逻辑错误和格式错误是主要瓶颈。

**[Tree-Of-Code A Tree-Structured Exploring Framework For End-To-End Code Generatio](code_intelligence/tree-of-code_a_tree-structured_exploring_framework_for_end-to-end_code_generatio.md)**

:   提出 Tree-of-Code（ToC）框架，通过树结构组织端到端的完整代码程序（CodeProgram）节点，结合基于执行结果的反思机制和提示/模型随机探索策略，在无需标注数据的零样本设置下，以不到 1/4 的交互轮次实现了比 CodeAct 高近 20% 的复杂任务准确率。

**[Tree Of Evolution Code Gen](code_intelligence/tree_of_evolution_code_gen.md)**

:   提出Tree-of-Evolution (ToE)——一种树结构的代码指令合成框架，通过多路径进化和质量驱动优化来克服现有方法（如Code Evol-Instruct/OSS-Instruct）的单向合成和随机生成限制，仅用75K合成数据微调base model即可达到或超越Qwen2.5-Coder-Instruct（数百万样本微调）的性能。

**[Utboost Rigorous Evaluation Of Coding Agents On Swe-Bench](code_intelligence/utboost_rigorous_evaluation_of_coding_agents_on_swe-bench.md)**

:   本文提出 UTBoost 框架，通过基于 LLM 的测试用例生成器 UTGenerator 和改进的解析器来增强 SWE-Bench 的测试用例覆盖率，发现 36 个测试不充分的实例和 345 个被错误标记为通过的补丁，导致 SWE-Bench Lite 排行榜 40.9% 和 SWE-Bench Verified 24.4% 的排名发生变化。

---

## ✏️ 知识编辑 { #knowledge_editing }

**[A General Knowledge Injection Framework For Icd Coding](knowledge_editing/a_general_knowledge_injection_framework_for_icd_coding.md)**

:   提出 GKI-ICD，首个无需额外专用网络模块即可同时注入三种 ICD 编码知识（描述、同义词、层级结构）的通用框架——通过合成知识引导文本（Guideline Synthesis）+ 多任务学习实现知识注入，在 MIMIC-III 和 MIMIC-III-50 上大多数指标达到 SOTA。

**[Adaptive Detoxification Safeguarding General Capabilities Of Llms Through Toxici](knowledge_editing/adaptive_detoxification_safeguarding_general_capabilities_of_llms_through_toxici.md)**

:   提出 ToxEdit——毒性感知的知识编辑方法，在前向传播中动态检测毒性激活模式（SVM 二分类器检测有害隐藏状态），将计算路由到原始 FFN 或编辑后的 FFN，实现自适应去毒而不过度编辑。增强 SafeEdit 基准加入指令遵从评估，在多个 LLM 上去毒能力和通用能力保留均 SOTA。

**[Bmike-53 Investigating Cross-Lingual Knowledge Editing With In-Context Learning](knowledge_editing/bmike-53_investigating_cross-lingual_knowledge_editing_with_in-context_learning.md)**

:   提出 BMIKE-53，覆盖 53 种语言的跨语言上下文知识编辑 (IKE) 基准，统一了 zsRE/CounterFact/WikiFactDiff 三个知识编辑数据集，系统评估发现模型规模和示例对齐对跨语言 IKE 效果至关重要，文字系统类型是影响跨语言性能差异的关键因素。

**[Chainedit Propagating Ripple Effects In Llm](knowledge_editing/chainedit_propagating_ripple_effects_in_llm.md)**

:   提出 ChainEdit 框架，通过将知识图谱中挖掘的逻辑规则与 LLM 内在逻辑推理能力对齐，实现知识编辑时的链式更新，将逻辑泛化准确率从约 20% 提升至 58-65%。

**[Cknowedit Chinese Knowledge Editing Dataset Llms](knowledge_editing/cknowedit_chinese_knowledge_editing_dataset_llms.md)**

:   提出首个面向中文语言特征的知识编辑数据集 CKnowEdit，涵盖语言学、事实性和逻辑性三大类共10个子类的1,854条样本，揭示了当前知识编辑方法在中文场景下的不足。

**[Compke Complex Question Answering Under Knowledge Editing](knowledge_editing/compke_complex_question_answering_under_knowledge_editing.md)**

:   提出CompKe基准——包含11,924个复杂问题——用于评估知识编辑方法在涉及**一对多关系、逻辑操作（交集/并集）和条件确认**的复杂推理场景下的表现，揭示现有方法在复杂问答上的显著不足。

**[Context-Robust Knowledge Editing For Language Models](knowledge_editing/context-robust_knowledge_editing_for_language_models.md)**

:   发现现有知识编辑方法在前缀上下文存在时大幅失败（编辑成功率从 90.9% 降至 69.1%），提出 CHED 基准评估上下文鲁棒性，并设计 CoRE 方法通过多样化前缀上下文 + 跨前缀隐藏状态方差正则化来增强编辑的上下文鲁棒性，在保持模型通用能力的同时显著缩小有/无上下文的性能差距。

**[Docmedit Towards Document-Level Model Editing](knowledge_editing/docmedit_towards_document-level_model_editing.md)**

:   首次提出文档级模型编辑任务，构建包含 37,990 条数据、105,652 个编辑事实的 DocMEdit 基准，揭示现有编辑方法在长上下文、多事实并行编辑场景下的严重不足。

**[Efficient Knowledge Editing](knowledge_editing/efficient_knowledge_editing.md)**

:   证明了 MEMIT/ROME/EMMET 等知识编辑方法的预计算步骤（缓存 4400 万隐向量）可以减少到理论最小值的 2-10 倍（不到原来的 0.3%），将预计算时间从数十小时降到几分钟，且编辑性能基本无损。

**[Megen Generative Backdoor Into Large Language Models Via Model Editing](knowledge_editing/megen_generative_backdoor_into_large_language_models_via_model_editing.md)**

:   提出 MEGen，一种基于模型编辑的生成式后门攻击方法，能够仅通过少量样本修改少量局部参数，在 LLM 中注入生成式后门，使模型在触发时自由输出预设的危险内容。

**[Memorizing Is Not Enough Deep Knowledge Injection Through Reasoning](knowledge_editing/memorizing_is_not_enough_deep_knowledge_injection_through_reasoning.md)**

:   提出四层知识注入框架（记忆→检索→推理→关联），构建 DeepKnowledge 合成测试平台，系统性揭示了知识注入各层级的关键因素：重复学习实现记忆、表达多样性实现检索、显式推理模式实现深度推理和关联，为 LLM 知识更新提供了完整的方法-层级映射。

**[Mitigating Negative Interference In Multilingual Sequential Knowledge Editing Th](knowledge_editing/mitigating_negative_interference_in_multilingual_sequential_knowledge_editing_th.md)**

:   本文提出 LangEdit 框架，通过将每种语言的参数更新投影到先前编辑语言的零空间上，实现多语言序列知识编辑中不同语言更新之间的数学隔离，有效缓解负干扰并保持多语言泛化能力。

**[Neuron-Level Sequential Editing For Large Language Models](knowledge_editing/neuron-level_sequential_editing_for_large_language_models.md)**

:   提出NSE方法用于LLM的序列化模型编辑，通过权重回退（weights rewinding）防止模型崩溃、基于激活值的神经元级选择性权重更新缓解模型遗忘、以及迭代多层编辑提高大规模知识更新的成功率。

**[Revealing The Deceptiveness Of Knowledge Editing A Mechanistic Analysis Of Super](knowledge_editing/revealing_the_deceptiveness_of_knowledge_editing_a_mechanistic_analysis_of_super.md)**

:   本文定义了"表面编辑"（superficial editing）现象——经过知识编辑的模型在常规提示下表现良好，但在特制攻击探针下会回退到原始知识——并通过机制分析揭示了早期层残差流和后期层特定注意力头是导致该现象的两个关键因素。

**[Scedit Script-Based Assessment Of Knowledge Editing](knowledge_editing/scedit_script-based_assessment_of_knowledge_editing.md)**

:   提出 ScEdit，一个基于脚本（Script）的知识编辑评估基准，将传统的"What"类事实回忆评估扩展为"How"类程序性推理评估，同时引入 token 级和文本级双层评估体系，揭示了现有知识编辑方法在实际应用场景中的显著不足。

**[Towards A Principled Evaluation Of Knowledge Editors](knowledge_editing/towards_a_principled_evaluation_of_knowledge_editors.md)**

:   本文系统性地揭示了知识编辑评估中评估方法、评估指标和编辑批量大小的选择会显著影响编辑器排名，并通过与LM Evaluation Harness集成来评估编辑对模型整体能力的副作用。

---

## 🔎 AIGC检测 { #aigc_detection }

**[A Rose By Any Other Name Llm-Generated Explanations Are Good Proxies For Human E](aigc_detection/a_rose_by_any_other_name_llm-generated_explanations_are_good_proxies_for_human_e.md)**

:   研究 LLM 生成的解释能否替代昂贵的人工解释来近似 NLI 的人工判断分布（HJD）——发现在提供人工标签的条件下，LLM 生成的解释与人工解释在近似 HJD 方面效果相当（"名字不重要，玫瑰依然芬芳"），且方法可推广到无人工解释的数据集和域外测试集。

**[Aigt Social Media Monitoring](aigc_detection/aigt_social_media_monitoring.md)**

:   首次大规模量化社交媒体上 AI 生成文本(AIGT)的占比变化——收集 Medium/Quora/Reddit 上 240 万帖子，构建 AIGTBench 训练最佳检测器 OSM-Det，发现 2022-2024 年间 Medium 和 Quora 的 AIGT 占比从~2% 飙升至~37-39%，而 Reddit 仅从 1.3% 增至 2.5%。

**[Chatgpt User Ai Text Detection](aigc_detection/chatgpt_user_ai_text_detection.md)**

:   通过 1,740 条标注实验发现，经常使用 LLM 进行写作任务的人类标注者可以极高精度（5人投票仅错 1/300）检测 AI 生成文本，即使面对改写和人性化逃逸策略也显著优于大多数自动检测器。

**[Greater Adversarial Mgt Detection](aigc_detection/greater_adversarial_mgt_detection.md)**

:   提出 GREATER 对抗训练框架，同步训练对抗攻击器（Greater-A）和 MGT 检测器（Greater-D），对抗器通过代理模型梯度识别关键 token 并在嵌入空间扰动生成对抗样本，检测器从课程式对抗样本中学习泛化防御，在 16 种攻击下 ASR 降至 5.53%（SOTA 为 6.20%），攻击效率比 SOTA 快 4 倍。

**[Haco-Det A Study Towards Fine-Grained Machine-Generated Text Detection Under Hum](aigc_detection/haco-det_a_study_towards_fine-grained_machine-generated_text_detection_under_hum.md)**

:   提出人机共创场景下的细粒度MGT检测任务和数据集HACo-Det，通过多轮LLM改写构建了11,200篇带词级别标注的共创文本，将7种主流检测器适配到词级/句级任务后发现：微调方法（DeBERTa）远优于基于度量的方法，但整体检测水平仍远未解决该问题。

**[Learning To Rewrite Generalized Llm-Generated Text Detection](aigc_detection/learning_to_rewrite_generalized_llm-generated_text_detection.md)**

:   提出Learning2Rewrite（L2R）框架，通过微调LLM的改写模型来放大人写文本和AI生成文本在改写编辑距离上的差异，从而实现跨领域高度泛化的AI文本检测——在21个独立领域上平均AUROC达0.9009，域外测试超越RAIDAR达4.67%、超越直接分类微调达51.35%。

**[Llm Vs Human Formal Syntax](aigc_detection/llm_vs_human_formal_syntax.md)**

:   首次使用形式句法理论（HPSG）系统比较六个 LLM 生成的纽约时报风格文本与真实人类撰写的 NYT 文本，发现 LLM 和人类写作在 HPSG 语法类型分布上存在系统性差异，揭示了 LLM 句法行为与人类的本质不同。

**[Low-Perplexity Llm-Generated Sequences And Where To Find Them](aigc_detection/low-perplexity_llm-generated_sequences_and_where_to_find_them.md)**

:   提出系统化 pipeline 分析 LLM 生成的低困惑度序列（token 预测概率 ≥0.9）并追溯到训练数据来源，发现 30-60% 的低困惑度片段无法匹配训练数据，将可匹配片段分为四种记忆行为类别。

**[Multisocial Mgt Detection](aigc_detection/multisocial_mgt_detection.md)**

:   构建首个多语言(22种语言)、多平台(5个社交媒体)、多生成器(7个LLM)的社交媒体机器生成文本检测基准 MultiSocial（47万文本），填补了社交媒体短文本+非英语场景下 MGT 检测研究的空白，发现微调检测器可在社交媒体文本上有效训练且训练平台选择很重要。

**[Reliably Bounding False Positives A Zero-Shot Machine-Generated Text Detection F](aigc_detection/reliably_bounding_false_positives_a_zero-shot_machine-generated_text_detection_f.md)**

:   提出基于多尺度保形预测（MCP）的零样本机器生成文本检测框架，通过文本长度感知的分组分位数计算，在严格约束假阳性率（FPR）上界的同时显著提升检测性能，并构建了覆盖15个领域、22个LLM的大规模双语基准数据集RealDet。

**[Who Writes What Ai Detection](aigc_detection/who_writes_what_ai_detection.md)**

:   揭示作者的社会语言学属性（性别、CEFR水平、学科领域、语言环境）会系统性地影响AI生成文本检测器的准确率，其中语言水平和语言环境的偏差最为显著且一致，提出了基于多因素WLS+ANOVA的偏差量化框架。

---

## 🗣️ 对话系统 { #dialogue }

**[Dialogue Systems For Emotional Support Via Value Reinforcement](dialogue/dialogue_systems_for_emotional_support_via_value_reinforcement.md)**

:   提出 ES-VR，首个将人类价值观强化融入情感支持对话系统的方法，通过目标价值检测器和参考生成器（均在 Reddit 数据上训练），结合 SFT + DPO 两阶段训练，使支持者模型不仅能缓解求助者的负面情绪，还能探索和强化其积极价值观，实现更深层的内在转变。

**[Enabling Chatbots With Eyes And Ears An Immersive Multimodal Conversation System](dialogue/enabling_chatbots_with_eyes_and_ears_an_immersive_multimodal_conversation_system.md)**

:   本文提出赋予聊天机器人"眼睛和耳朵"的沉浸式多模态对话系统，构建了融合视觉与听觉的多会话多方对话数据集 M3C，并设计了包含对话模块和多模态记忆检索模块的对话模型，实现了多说话者共享视听体验的动态长期对话。

**[Enstom Enhancing Dialogue Systems With Entropy-Scaled Steering Vectors For Topic](dialogue/enstom_enhancing_dialogue_systems_with_entropy-scaled_steering_vectors_for_topic.md)**

:   提出 EnSToM，一种基于熵缩放转向向量的轻量级方法，通过利用 LLM 内部层级熵分布差异来动态调整转向强度，在不修改模型参数的情况下提升任务导向对话系统的主题维持能力。

**[Know You First And Be You Better Modeling Human-Like User Simulators Via Implici](dialogue/know_you_first_and_be_you_better_modeling_human-like_user_simulators_via_implici.md)**

:   本文提出 USP（User Simulator with Implicit Profiles）框架，通过从人机对话中提取隐式用户画像，并结合条件监督微调和基于循环一致性的强化学习，在真实性、一致性和多样性三个维度上显著超越基线方法，语义相似度和风格相似度分别提升约 34% 和 43%。

**[Know Your Mistakes Towards Preventing Overreliance On Task-Oriented Conversation](dialogue/know_your_mistakes_towards_preventing_overreliance_on_task-oriented_conversation.md)**

:   本文提出面向任务型对话系统的 Accountability Model，在 LLM 中加入额外的 accountability head 作为二分类器预测对话状态中各 slot 的概率，从而检测并自校正假阳性和假阴性错误，在 MultiWOZ 上将 JGA 从 64.34 提升到 70.51（↑9.6%），达到 SOTA。

**[Kokorochat A Japanese Psychological Counseling Dialogue](dialogue/kokorochat_a_japanese_psychological_counseling_dialogue.md)**

:   提出 KokoroChat，一个通过训练有素的咨询师角色扮演收集的日语心理咨询对话数据集，包含 6,589 段长对话及详细的客户反馈评分，用于提升 LLM 的心理咨询回复生成和对话评估能力。

**[Persona Sentiment Dialogue](dialogue/persona_sentiment_dialogue.md)**

:   大规模分析 LLM 对人设情感极性的敏感性，发现负面人设导致过度强调人设属性和对话矛盾、弱/中性人设产生低质量对话，提出结合逐轮生成、人设排序和情感感知提示的对话生成框架来缓解这些问题。

**[Personalens A Benchmark For Personalization Evaluation In Conversational Ai Assi](dialogue/personalens_a_benchmark_for_personalization_evaluation_in_conversational_ai_assi.md)**

:   提出 PersonaLens，一个面向任务导向型 AI 助手个性化能力的综合评测基准，包含 1500 个丰富用户画像、20 个领域 111 个任务、用户模拟 Agent 和 Judge Agent，通过大规模自动化评估揭示当前 LLM 助手在个性化方面的显著不足。

**[Reflectdiffu Empathetic Response](dialogue/reflectdiffu_empathetic_response.md)**

:   提出轻量级共情对话框架 ReflectDiffu，融合情感传染（捕捉情绪）、意图二次机制（Exploring-Sampling-Correcting将情绪映射为行动意图）和扩散模型生成，在相关性、可控性和信息量上全面超越现有基线和 Llama-3.1-8B。

**[Uniconv Retrieval Response Gen](dialogue/uniconv_retrieval_response_gen.md)**

:   探索如何将对话场景中的稠密检索和响应生成统一到单个 LLM 中，通过三个联合训练目标（对话检索 + 响应生成 + 上下文识别指令）和数据差异缓解机制，在五个对话搜索数据集上实现检索和生成的相互促进，超越分离式基线。

---

## 🔗 因果推理 { #causal_inference }

**[Causal Graph Based Event Reasoning Using Semantic Relation Experts](causal_inference/causal_graph_based_event_reasoning_using_semantic_relation_experts.md)**

:   本文提出了一种基于多个语义关系专家（时间、篇章、前置条件、常识）协作讨论的因果事件图生成方法，用于显式建模事件间的因果连接，并在事件预测、事件预报等多个下游任务上取得了有竞争力的结果。

**[Causalrag Integrating Causal Graphs Into Retrieval-Augmented Generation](causal_inference/causalrag_integrating_causal_graphs_into_retrieval-augmented_generation.md)**

:   提出 CausalRAG，将因果图集成到 RAG 的检索过程中——从文档构建文本图并识别因果关系，在查询时通过因果路径发现和因果摘要生成来检索上下文，在文档问答中显著提升上下文精度（92.86%）和检索召回率。

**[Fitcf A Framework For Automatic Feature Importance-Guided Counterfactual Example](causal_inference/fitcf_a_framework_for_automatic_feature_importance-guided_counterfactual_example.md)**

:   提出 FitCF 框架，利用 BERT 特征归因方法（LIME/IG/SHAP等）提取重要词来引导 LLM 在 zero-shot 下生成反事实样本（ZeroCF），再经标签翻转验证筛选后作为 few-shot 示例，在新闻分类和情感分析任务上一致性超越 Polyjuice、BAE、FIZLE 三种基线。

**[Iris An Iterative And Integrated Framework](causal_inference/iris_an_iterative_and_integrated_framework.md)**

:   提出 IRIS 框架——仅需一组初始变量名作为输入，即可自动检索文档、提取变量值构建结构化数据、通过混合因果发现（GES 统计算法 + LLM 因果关系验证）构建因果图，并通过缺失变量提议组件迭代扩展变量集合，放松了传统方法的无环和因果充分性假设，在 Cancer、Diabetes、Obesity、ADNI、Insurance 等 6 个数据集上 F1 全面超越 0-shot/CoT/RAG 基线。

**[Leveraging Variation Theory In Counterfactual Data Augmentation For Optimized Ac](causal_inference/leveraging_variation_theory_in_counterfactual_data_augmentation_for_optimized_ac.md)**

:   本文将变异理论(Variation Theory)引入反事实数据增强(CDA)框架，通过保留神经符号模式的方式使用LLM生成反事实样本，并结合三级过滤流水线筛选高质量数据，用于优化主动学习中的少样本文本分类，在多个数据集上取得显著F1提升。

**[Llm Causal Discovery Reliability](causal_inference/llm_causal_discovery_reliability.md)**

:   利用开源 LLM（OLMo、BLOOM）可访问的预训练语料库，实证验证了"因果鹦鹉"假说——LLM 识别因果关系的能力与预训练数据中该关系的出现频率高度相关（Spearman r=0.9），且错误因果关系的存在和上下文变化都会显著影响预测可靠性。

**[Reasoning Is All You Need For Video Generalization A Counterfactual Benchmark Wi](causal_inference/reasoning_is_all_you_need_for_video_generalization_a_counterfactual_benchmark_wi.md)**

:   提出 COVER（COunterfactual VidEo Reasoning），一个多维度视频反事实推理 benchmark，将评估任务按抽象-具体和感知-认知两个维度分为四象限共 13 类任务，并通过将复杂问题分解为子问题（必要条件）来揭示——子问题准确率与反事实推理能力强相关，提升推理能力是改善视频理解鲁棒性的关键。

---

## 🎨 图像生成 { #image_generation }

**[A Unified Agentic Framework For Evaluating Conditional Image Generation](image_generation/a_unified_agentic_framework_for_evaluating_conditional_image_generation.md)**

:   提出 CIGEval，一个基于大型多模态模型（LMM）的统一 Agent 评估框架，通过工具集成（Grounding、Highlight、Difference、Scene Graph）和分而治之的评估策略，在 7 种条件图像生成任务上达到与人类标注者相当的相关性（0.4625 vs 人类间 0.47），且仅用 2.3K 训练数据微调 7B 模型即超越 GPT-4o 版 SOTA。

**[D-Gen Automatic Distractor Generation And Evaluation For Reliable Assessment Of ](image_generation/d-gen_automatic_distractor_generation_and_evaluation_for_reliable_assessment_of_.md)**

:   提出 D-GEN——首个开源干扰项生成模型（LLaMA微调，8B/70B），自动将开放式评测题转为多选题格式，配套排名对齐+熵分析两种评估方法验证干扰项质量，在 MMLU 上 Spearman's ρ=0.99 保持模型排名一致性。

**[Difftod Diffusion Dialogue Planning](image_generation/difftod_diffusion_dialogue_planning.md)**

:   DiffTOD 将对话规划建模为轨迹生成问题，利用掩码扩散语言模型实现非顺序对话规划，并设计三种引导机制（词级/语义级/搜索级）灵活控制对话朝目标推进，在谈判/推荐/闲聊三种场景上显著超越基线。

**[Flashaudio Rectified Flow Tta](image_generation/flashaudio_rectified_flow_tta.md)**

:   将整流流（Rectified Flow）引入文本转音频生成，通过双焦采样器优化时间步分布、不混溶流减少数据-噪声总距离、锚定优化修正 CFG 引导误差，实现单步生成 FAD=1.49 超越百步扩散模型，生成速度达实时 400 倍。

**[Multimodal Pragmatic Jailbreak On Text-To-Image Models](image_generation/multimodal_pragmatic_jailbreak_on_text-to-image_models.md)**

:   提出"多模态语用越狱"（Multimodal Pragmatic Jailbreak）新型攻击方式，通过让T2I模型生成包含视觉文字的图像，使得图像内容和文字内容单独看都安全但组合后产生不安全内容，揭示了所有测试模型（包括DALL·E 3）均受此攻击影响。

**[Ozspeech One-Step Zero-Shot Speech Synthesis With Learned-Prior-Conditioned Flow](image_generation/ozspeech_one-step_zero-shot_speech_synthesis_with_learned-prior-conditioned_flow.md)**

:   提出OZSpeech，首个将最优传输条件流匹配(OT-CFM)与学习先验分布相结合实现单步采样的零样本TTS系统，在内容准确性(WER)、推理速度和模型大小上均大幅领先现有方法。

**[Rvc Rhythm Voice Conversion](image_generation/rvc_rhythm_voice_conversion.md)**

:   R-VC 是首个实现节奏可控的零样本语音转换系统，通过 Mask Transformer 时长模型建模目标说话人的节奏风格，结合 Shortcut Flow Matching 的 DiT 解码器实现仅 2 步采样的高效高质量语音生成，在 LibriSpeech 上 WER 3.51、说话人相似度 0.930。

---

## 🎁 推荐系统 { #recommender }

**[Beyond Single Labels Improving Conversational Recommendation Through Llm-Powered](recommender/beyond_single_labels_improving_conversational_recommendation_through_llm-powered.md)**

:   针对对话推荐系统中的假阴性问题（用户可能喜欢的item被错误标记为负样本），提出基于LLM的数据增强框架，通过语义检索+相关性打分生成合成标签，再通过两阶段训练策略平衡语义相关性和协同信息。

**[Cove Compressed Vocabulary Expansion Makes Better Llm-Based Recommender Systems](recommender/cove_compressed_vocabulary_expansion_makes_better_llm-based_recommender_systems.md)**

:   提出 CoVE 框架，通过扩展 LLM 词表为每个物品分配唯一 token ID 和嵌入，将序列推荐任务转化为 next-token prediction，相比现有方法推荐准确率提升最高 62%，推理速度提升约 100 倍，并通过哈希嵌入压缩解决大规模场景的内存问题。

**[Gram Generative Recommendation](recommender/gram_generative_recommendation.md)**

:   提出 GRAM 生成式推荐模型，通过语义到词汇的翻译（将隐式物品关系编码到 LLM 词汇空间）和多粒度迟融合（分别编码不同粒度提示后在解码时融合），在四个基准上比八个 SOTA 方法在 Recall@5 上提升 11.5-16.0%。

**[Kerl Knowledge-Enhanced Personalized Recipe Recommendation Using Large Language ](recommender/kerl_knowledge-enhanced_personalized_recipe_recommendation_using_large_language_.md)**

:   提出 KERL 统一食品推荐系统，结合 FoodKG 知识图谱和 Phi-3-mini 多 LoRA 微调，实现个性化食谱推荐（F1=0.973）、食谱生成和微量营养素估算三个功能，大幅超越基线 LLM 和传统嵌入方法。

**[Lotus A Leaderboard For Detailed Image Captioning From Quality To Societal Bias ](recommender/lotus_a_leaderboard_for_detailed_image_captioning_from_quality_to_societal_bias_.md)**

:   提出 LOTUS 排行榜，从描述质量（对齐性、描述性、语言复杂度）、副作用（幻觉、有害性）和社会偏见（性别、肤色）三个维度统一评估大型视觉语言模型的详细图像描述能力，并支持基于用户偏好的定制化评估。

**[Mira Empowering One-Touch Ai Services On Smartphones With Mllm-Based Instruction](recommender/mira_empowering_one-touch_ai_services_on_smartphones_with_mllm-based_instruction.md)**

:   提出 MIRA 框架，通过结构化推理、模板增强推理和前缀树约束解码，让用户在智能手机上长按文本或图片即可获得上下文相关的 AI 服务指令推荐，在 7B 模型上超越 GPT-4V（F1: 0.9121 vs 0.879），token 使用量仅为 1/7。

**[Reclm Recommendation Instruction Tuning](recommender/reclm_recommendation_instruction_tuning.md)**

:   提出 RecLM，一个模型无关的推荐指令微调框架，通过两轮对话式指令微调将协同过滤信号注入 LLM 生成的用户/商品画像，再用 RLHF（PPO）精炼画像质量，在 MIND/Netflix/工业数据集上作为即插即用组件为 BiasMF/NCF/LightGCN/SGL/SimGCL 一致带来提升，尤其在冷启动场景效果显著。

---

## 🎮 强化学习 { #reinforcement_learning }

**[Align-Slm Textless Spoken Language Models With Reinforcement Learning From Ai Fe](reinforcement_learning/align-slm_textless_spoken_language_models_with_reinforcement_learning_from_ai_fe.md)**

:   首次将偏好优化（DPO + RLAIF）应用于无文本口语语言模型（SLM）——从预训练 TWIST 模型生成多个语音续写候选，通过 ASR→LLM 评分自动创建偏好数据对，用 DPO 训练 SLM 一致性地生成语义更好的语音续写，结合课程学习进一步提升。在 ZeroSpeech/StoryCloze 基准上达到 SLM SOTA（sWUGGY 77.9%、S-StoryCloze 61.1%、T-StoryCloze 86.8%）。

**[Eierl Dialogue Policy](reinforcement_learning/eierl_dialogue_policy.md)**

:   提出 EIERL 方法，将进化算法（EA）的全局搜索能力与深度强化学习（DRL）的局部优化能力结合用于任务导向对话策略学习，并设计精英个体注入（EII）机制自适应地将高性能个体注入 EA 种群以加速进化，在 4 个数据集上显著提升探索-利用平衡。

**[Learning To Generate Structured Output With Schema Reinforcement Learning](reinforcement_learning/learning_to_generate_structured_output_with_schema_reinforcement_learning.md)**

:   提出 SchemaBench 基准（约4万条 JSON schema）和 Schema Reinforcement Learning (SRL) 训练框架，通过细粒度 schema 验证器提供密集奖励信号，结合 Thoughts of Structure (ToS) 推理机制，将 LLM 的复杂 JSON 生成准确率提升高达16%，同时不损害通用推理能力。

**[Llm-Enhanced Self-Evolving Reinforcement Learning For Multi-Step E-Commerce Paym](reinforcement_learning/llm-enhanced_self-evolving_reinforcement_learning_for_multi-step_e-commerce_paym.md)**

:   将电商支付欺诈检测建模为多步 MDP，用 LLM（Mixtral/LLaMA/Gemma）通过进化算法自动生成和优化 RL 奖励函数，在 eBay 真实交易数据上比人工设计奖励函数和传统 SL 基线显著提升 dollar-wise precision。

**[Maporl Multi-Agent Post-Co-Training For Collaborative Large Language Models With](reinforcement_learning/maporl_multi-agent_post-co-training_for_collaborative_large_language_models_with.md)**

:   提出 MAPoRL——一种基于多智能体强化学习的后训练范式，通过让多个 LLM 在辩论框架中共同训练（co-training），配合验证器评分和协作激励机制，显著提升多 LLM 协作的效果，并展现出跨任务的泛化能力。

**[Prompt-Based Personality Profiling Reinforcement Learning For Relevance Filterin](reinforcement_learning/prompt-based_personality_profiling_reinforcement_learning_for_relevance_filterin.md)**

:   提出RL-Profiler方法，用强化学习训练一个帖子相关性过滤器（SelNet），从用户Profile的大量帖子中筛选出与人格特征相关的少量帖子，再交给LLM零样本预测人格，在大幅减少上下文长度的同时保持接近使用全部帖子的预测效果。

**[Treerl Tree Search Rl](reinforcement_learning/treerl_tree_search_rl.md)**

:   提出 TreeRL，将基于熵引导的树搜索（EPTree）直接集成到 LLM 的 on-policy 强化学习训练中，通过在高不确定性 token 处分叉来扩展推理路径多样性，并利用树结构提供的全局+局部优势作为过程监督信号，在数学和代码推理任务上超过传统的多链采样 RL。

---

## 🤖 机器人/具身智能 { #robotics }

**[Cheer-Ekman Fine-Grained Embodied Emotion Classification](robotics/cheer-ekman_fine-grained_embodied_emotion_classification.md)**

:   本文提出CHEER-Ekman数据集，将CHEER数据集的二元具身情感标注扩展为Ekman六类基础情绪，并采用基于LLM的自动Best-Worst Scaling（BWS）技术实现无需任务特定训练的细粒度情感分类，性能超越有监督BERT。

**[Dice Idiomaticity](robotics/dice_idiomaticity.md)**

:   提出 DICE 数据集，通过严格控制习语表面形式不变而仅改变上下文，揭示 LLM 在习语消歧任务上的高准确率很大程度来自对表面线索的依赖而非真正的上下文理解，并发现句子概率和搭配频率对模型表现有复杂影响。

**[Do Emotions Really Affect Argument Convincingness A Dynamic Approach With Llm-Ba](robotics/do_emotions_really_affect_argument_convincingness_a_dynamic_approach_with_llm-ba.md)**

:   提出一种受心理学操控检验启发的动态框架，利用LLM调节论证的情感强度，系统考察情感对论证说服力的因果影响，发现超过半数情况下人类的说服力判断不受情感变化影响，而当情感有影响时更多是增强而非削弱说服力。

**[Drae Dynamic Retrieval-Augmented Expert Networks For Lifelong Learning And Task ](robotics/drae_dynamic_retrieval-augmented_expert_networks_for_lifelong_learning_and_task_.md)**

:   提出 DRAE 框架，整合动态 MoE 路由、参数化 RAG（P-RAG）、三层认知控制架构（ReflexNet-SchemaPlanner-HyperOptima）和 DPMM 终身知识保留，在机器人操作和自动驾驶任务上平均成功率达 82.5%，有效缓解灾难性遗忘。

**[Hierarchical-Task-Aware Multi-Modal Mixture Of Incremental Lora Experts For Embo](robotics/hierarchical-task-aware_multi-modal_mixture_of_incremental_lora_experts_for_embo.md)**

:   提出层次化具身持续学习设置（HEC），将 agent 学习分为高层指令和低层动作两级，并设计 Task-aware MoILE 方法——通过跨模态聚类识别任务、双路由器选择 LoRA 专家、SVD 正交训练保留旧知识，在 5 种增量学习场景中遗忘率降至 3.37%（vs 前 SOTA 7.44%）。

**[Self Percept Manipulation Detection](robotics/self_percept_manipulation_detection.md)**

:   提出 SELF-PERCEPT 两阶段 prompting 框架，借鉴心理学自我知觉理论（Self-Perception Theory），引导 LLM 先观察对话参与者的行为线索再推断内在态度，显著提升多人多轮对话中心理操纵的检测效果。

**[Vulnerability Of Llms To Vertically Aligned Text Manipulations](robotics/vulnerability_of_llms_to_vertically_aligned_text_manipulations.md)**

:   本文系统揭示了LLM对垂直排列文本输入的严重脆弱性：仅将少量关键词垂直排列即可导致文本分类准确率下降25-45个百分点，CoT推理无法缓解此问题，但精心设计的few-shot learning可有效恢复性能。

---

## 🎬 视频理解 { #video_understanding }

**[Addressing Blind Guessing Calibration Of Selection Bias In Multiple-Choice Quest](video_understanding/addressing_blind_guessing_calibration_of_selection_bias_in_multiple-choice_quest.md)**

:   首次系统性研究视频语言模型（VLM）在多选题回答中的选项选择偏差问题，通过任务分解分析偏差来源，提出BOLD后处理校准技术，在减少偏差的同时提升模型性能。

**[From Teacher To Student Tracking Memorization Through Model Distillation](video_understanding/from_teacher_to_student_tracking_memorization_through_model_distillation.md)**

:   系统研究了知识蒸馏（KD）对大语言模型记忆化行为的影响，发现蒸馏不仅能压缩模型，还能显著降低对训练数据的逐字记忆风险——其中反向 KL 蒸馏（RKLD/MiniLLM）将记忆化比例从 SFT 的 65.4% 降至最低 6.0%。

**[Generative Frame Sampler For Long Video Understanding](video_understanding/generative_frame_sampler_for_long_video_understanding.md)**

:   提出 GenS，一个基于 VideoLLM 的生成式帧采样模块，用自然语言输出question-aware的相关帧时间段和置信度分数，作为即插即用模块在 LongVideoBench/MLVU/HourVideo 上为多种 VideoLLM 带来 2-4 个点的一致提升。

**[Icr Probe Tracking Hidden State Dynamics For Reliable Hallucination Detection In](video_understanding/icr_probe_tracking_hidden_state_dynamics_for_reliable_hallucination_detection_in.md)**

:   提出 ICR Score（Information Contribution to Residual Stream），通过测量 MHSA 和 FFN 模块对隐状态更新的贡献一致性来量化残差流动态，构建仅 16K 参数的 ICR Probe，在 4 个数据集 × 3 个 LLM 上幻觉检测 AUROC 全面超越基线。

**[Improving Dialogue State Tracking Through Combinatorial Search For In-Context Ex](video_understanding/improving_dialogue_state_tracking_through_combinatorial_search_for_in-context_ex.md)**

:   提出 CombiSearch 方法，通过组合式评分为对话状态追踪（DST）选择最优 in-context 示例组合，在仅用 5% 训练数据的情况下超越所有使用 100% 数据的 baseline，理想设置下 JGA 上界比传统方法高 12%。

**[Raven Robust Advertisement Video Violation Temporal Grounding Via Reinforcement ](video_understanding/raven_robust_advertisement_video_violation_temporal_grounding_via_reinforcement_.md)**

:   本文提出RAVEN框架，将课程强化学习与多模态LLM结合，通过分层奖励机制和渐进式训练策略，实现广告视频违规内容的精确时序定位和类别预测，无需显式推理标注数据即可激发涌现推理能力。

**[Vidcapbench A Comprehensive Benchmark Of Video Captioning For Controllable Text-](video_understanding/vidcapbench_a_comprehensive_benchmark_of_video_captioning_for_controllable_text-.md)**

:   提出 VidCapBench，首个专为可控文生视频（T2V）设计的视频描述评估 benchmark，从美学/内容/运动/物理规律四个维度评估 caption 质量，643 个视频+10,644 个 QA 对，实验证明 VidCapBench 分数与 T2V 生成质量高度正相关。

---

## 🔄 自监督/表示学习 { #self_supervised }

**[Improving Low-Resource Morphological Inflection Via Self-Supervised Objectives](self_supervised/improving_low-resource_morphological_inflection_via_self-supervised_objectives.md)**

:   系统探索 13 种自监督辅助目标（自编码、CMLM、T5-style 等）在极低资源形态变化任务中的效果，发现无标注数据极少时自编码最优，数据增多后字符级 MLM 更好，按形态素边界采样掩码是最有前景的方向。

**[Llm Back Gen Treebank](self_supervised/llm_back_gen_treebank.md)**

:   提出 LLM 反向生成方法自动构建跨领域成分句法树库——给定只有领域关键词叶节点的不完整句法树，用 LLM 填充缺失词汇生成完整的跨领域句法树库，结合 span 级对比学习预训练，在 MCTB 五个目标领域上达到跨领域成分句法分析 SOTA。

**[Magnet Augmenting Generative Decoders With Representation Learning And Infilling](self_supervised/magnet_augmenting_generative_decoders_with_representation_learning_and_infilling.md)**

:   提出 Magnet 方法，通过混合注意力机制（双向+因果）和三个自监督目标（掩码预测+对比学习+缺失片段生成），将纯解码器 LLM 同时增强为文本编码器和填充模型，在 token 级和句子级表示学习任务上超越 LLM2Vec 等专用方法，同时避免了双向化带来的严重文本重复问题。

**[Qaencoder Aligned Representation](self_supervised/qaencoder_aligned_representation.md)**

:   提出 QAEncoder，一种免训练方法通过蒙特卡洛采样估计文档对应查询的期望嵌入作为文档表示的代理，配合文档指纹保持区分性，在 BEIR 上将 bge-large 从 58.5 提升到 61.8 NDCG@10，零额外存储和延迟开销。

**[Shubert Self-Supervised Sign Language Representation Learning Via Multi-Stream C](self_supervised/shubert_self-supervised_sign_language_representation_learning_via_multi-stream_c.md)**

:   提出 SHuBERT（Sign Hidden-Unit BERT），将语音自监督学习模型 HuBERT 的 masked cluster prediction 范式迁移到手语视频——对手部、面部、身体姿态四个流分别聚类并同时预测 masked 帧的聚类标签，在约 984 小时 ASL 视频上预训练后，在翻译/孤立识别/指拼检测多任务上达到公开数据 SOTA。

**[Whispa Semantically And Psychologically Aligned Whisper With Self-Supervised Con](self_supervised/whispa_semantically_and_psychologically_aligned_whisper_with_self-supervised_con.md)**

:   提出 WhiSPA，通过对比学习将 Whisper 音频编码器的潜在空间与 SBERT 语义表征和心理学维度（情感、人格）对齐，消除语音处理中对额外文本 LM 的依赖，在心理学评估任务上误差降低 73-84%。

---

## 📈 时间序列 { #time_series }

**[Context Aware Sentiment Forecasting Agents](time_series/context_aware_sentiment_forecasting_agents.md)**

:   提出一个基于 LLM 的多视角角色扮演框架（MPR），通过主观 Agent 模拟用户发帖、客观 Agent（微调的"心理学家"LLM）审查行为一致性，以迭代纠正的方式预测社交媒体用户对实时事件的未来情感反应，在宏观和微观层面均大幅超越传统方法。

**[Ctpd Cross-Modal Temporal Pattern Discovery For Enhanced Multimodal Electronic H](time_series/ctpd_cross-modal_temporal_pattern_discovery_for_enhanced_multimodal_electronic_h.md)**

:   提出 CTPD 框架，利用 Slot Attention 从多模态 EHR 数据（不规则时间序列+临床笔记）中发现跨模态共享的时序原型模式，通过 TP-NCE 对比损失对齐两模态的时序语义，在 MIMIC-III 的死亡率预测和表型分类任务上取得 SOTA。

**[G2S A General-To-Specific Learning Framework For Temporal Knowledge Graph Foreca](time_series/g2s_a_general-to-specific_learning_framework_for_temporal_knowledge_graph_foreca.md)**

:   提出 G2S 框架，将时序知识图谱（TKG）预测中的通用模式（时序结构规律）与场景信息（具体实体/关系）解耦，先在匿名化时序结构上学习通用模式，再注入场景信息，有效提升 LLM 在 TKG 预测中的泛化能力。

**[Lets-C Leveraging Text Embedding For Time Series Classification](time_series/lets-c_leveraging_text_embedding_for_time_series_classification.md)**

:   提出 LETS-C——将时间序列数字化为文本字符串后用 text embedding 模型编码，与原始时间序列元素级相加融合后送入轻量 CNN+MLP 分类头，在 UEA 10 个多变量时间序列数据集上以仅 14.5% 的可训练参数量超越 OneFitsAll（GPT-2 微调）等 27 个 baseline 达到 SOTA。

**[Time-Mqa Time Series Multi-Task Question Answering With Context Enhancement](time_series/time-mqa_time_series_multi-task_question_answering_with_context_enhancement.md)**

:   提出Time-MQA框架和TSQA数据集（~200k QA对），将时间序列的预测、填补、异常检测、分类和开放式推理问答统一到自然语言问答范式下，通过持续预训练LLM使其具备时间序列理解和推理能力。

---

## 🎯 目标检测 { #object_detection }

**[Anchored Answers Unravelling Positional Bias In Gpt-2S Multiple-Choice Questions](object_detection/anchored_answers_unravelling_positional_bias_in_gpt-2s_multiple-choice_questions.md)**

:   首次对 GPT-2 系列在选择题中的"锚定偏差"（始终偏好选项 A）进行全面机械可解释性分析——通过 Logit Lens 定位导致偏差的 MLP 值向量和注意力头，然后更新值向量+重校准注意力权重，以最小干预消除偏差并将 MCQ 准确率平均提升 70%+。

**[Dolphin Document Image Parsing Via Heterogeneous Anchor Prompting](object_detection/dolphin_document_image_parsing_via_heterogeneous_anchor_prompting.md)**

:   提出 Dolphin，一个轻量级（322M）的文档图像解析模型，采用"先分析后解析"（analyze-then-parse）两阶段范式——先进行页面级布局分析生成阅读顺序的元素序列，再利用异构锚点提示（heterogeneous anchor prompting）并行解析各元素内容，以仅 322M 参数在页面级和元素级解析任务上超越 7B+ 模型和商业系统。

**[Weed Out Then Harvest Dual Low-Rank Adaptation Is An Effective Noisy Label Detec](object_detection/weed_out_then_harvest_dual_low-rank_adaptation_is_an_effective_noisy_label_detec.md)**

:   提出Delora框架，通过引入clean LoRA和noisy LoRA双模块构建噪声标签检测器，将样本选择与模型训练解耦，打破传统"小损失"方法中样本选择与训练互相影响的恶性循环。

**[Why Safeguarded Ships Run Aground Aligned Large Language Models Safety Mechanism](object_detection/why_safeguarded_ships_run_aground_aligned_large_language_models_safety_mechanism.md)**

:   揭示了安全对齐LLM的一个普遍现象：安全机制过度锚定在chat template区域（TASA），导致越狱攻击可通过干扰template区域的信息处理来绕过安全防线，并提出通过将安全探针从template区域迁移到生成阶段来缓解该漏洞。

---

## 🧑 人体理解 { #human_understanding }

**[BQA: Body Language Question Answering Dataset for Video Large Language Models](human_understanding/bqa_body_language_question_answering_dataset_for_video_large_language_models.md)**

**[I See What You Mean Co-Speech Gestures For Reference Resolution In Multimodal Di](human_understanding/i_see_what_you_mean_co-speech_gestures_for_reference_resolution_in_multimodal_di.md)**

:   提出自监督预训练方法学习表征性共语手势（co-speech iconic gestures）的嵌入表示，将骨骼动作 grounded 到语言中，在面对面对话的指称消解任务上证明手势与语音的互补性——手势+语音准确率 31% 远超单独语音 24% 或手势 19%。

**[Transbench Breaking Barriers For Transferable Graphical User Interface Agents In](human_understanding/transbench_breaking_barriers_for_transferable_graphical_user_interface_agents_in.md)**

:   提出首个系统评估 GUI Agent **迁移性**（跨版本/跨平台/跨应用）的 benchmark TransBench，涵盖 81 个中文 App、1459 张截图、22K+ 标注指令，实验表明在旧版本上微调可有效迁移到新版本和其他平台，而跨平台迁移中 Android 数据的泛化性最强。

---

## ✂️ 语义分割 { #segmentation }

**[Def-Dts Deductive Reasoning For Open-Domain Dialogue Topic Segmentation](segmentation/def-dts_deductive_reasoning_for_open-domain_dialogue_topic_segmentation.md)**

:   提出 DEF-DTS，一种基于 LLM 多步演绎推理的对话话题分割方法——通过双向上下文摘要 → 话语意图分类（5 类） → 演绎话题转移判断三步 pipeline，在 TIAGE、SuperDialseg、Dialseg711 三个数据集上取得无监督/prompt 方法 SOTA，在 Dialseg711 上超越监督方法。

**[Instructpart Task-Oriented Part Segmentation With Instruction Reasoning](segmentation/instructpart_task-oriented_part_segmentation_with_instruction_reasoning.md)**

:   提出 InstructPart，首个将任务导向指令与部件级分割结合的真实世界 benchmark——2400 张图像、48 类物体、44 类部件、9600 条人工标注的任务指令，评估发现当前 VLM 在指令驱动的部件分割上严重不足，基于 LISA+DINOv2 的 baseline 微调后性能提升约 100%。

**[Pixel-Level Reasoning Segmentation Via Multi-Turn Conversations](segmentation/pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)**

:   提出像素级推理分割（Pixel-level RS）新任务，通过多轮对话逐步理解用户意图，实现细粒度的像素级分割，并构建了包含24k句话、8.3k多轮对话场景的PRIST数据集和MIRAS框架。

---

## 🖼️ 图像恢复 { #image_restoration }

**[Diffusedef Adversarial Defense](image_restoration/diffusedef_adversarial_defense.md)**

:   DiffuseDef 提出了一种将扩散层作为去噪器插入编码器和分类器之间的对抗防御方法，通过扩散训练学会预测隐状态噪声，推理时对对抗隐状态加噪+迭代去噪+集成，在黑盒和白盒攻击下达 SOTA 鲁棒性。

**[Prep-Ocr A Complete Pipeline For Document Image Restoration And Enhanced Ocr Acc](image_restoration/prep-ocr_a_complete_pipeline_for_document_image_restoration_and_enhanced_ocr_acc.md)**

:   提出 PreP-OCR 两阶段流水线：先用合成退化数据训练的 ResShift 模型修复历史文档图像（多方向 patch 提取+中值融合），再用 ByT5 做 OCR 后语义纠错，在 13,831 页真实历史文档上降低 CER 63.9-70.3%。

---

## 📡 信号/通信 { #signal_comm }

**[Toolspectrum Towards Personalized Tool Utilization For Large Language Models](signal_comm/toolspectrum_towards_personalized_tool_utilization_for_large_language_models.md)**

:   提出 ToolSpectrum benchmark，首次评估 LLM 在用户画像和环境因素双维度下的个性化工具选择能力，发现现有 SOTA 模型在联合推理两个维度时表现显著下降。

**[Wirelessmathbench A Mathematical Modeling Benchmark For Llms In Wireless Communi](signal_comm/wirelessmathbench_a_mathematical_modeling_benchmark_for_llms_in_wireless_communi.md)**

:   本文提出WirelessMathBench，一个包含587道题目的无线通信数学建模基准，从40篇前沿论文中提取，系统评估LLM在领域特定数学推导上的能力，揭示即使最强的DeepSeek-R1平均准确率也仅38.05%，完整公式推导仅7.83%。

---

## 🧊 3D视觉 { #3d_vision }

**[Slamming Training A Speech Language Model On One Gpu In A Day](3d_vision/slamming_training_a_speech_language_model_on_one_gpu_in_a_day.md)**

:   提出 Slam 训练配方，通过系统化的模型初始化、架构选择、合成数据、偏好优化等环节优化，在单张 A5000 GPU 上 24 小时内训练出性能媲美大规模 SLM 的语音语言模型。

---

## 🚗 自动驾驶 { #autonomous_driving }

**[Embracing Large Language Models In Traffic Flow Forecasting](autonomous_driving/embracing_large_language_models_in_traffic_flow_forecasting.md)**

:   提出 LEAF 框架，用图分支（pair-wise关系）和超图分支（non-pair-wise关系）的双分支预测器生成候选预测，再用冻结的 LLM 作为选择器（判别而非生成）挑选最优预测，通过 ranking loss 反馈优化预测器，在 PEMS 数据集上取得 SOTA。

---

## 📐 优化/理论 { #optimization }

**[Scalebio Bilevel Data Reweighting](optimization/scalebio_bilevel_data_reweighting.md)**

:   ScaleBiO 提出基于罚函数重构的全一阶双层优化算法，首次将双层优化应用于 30B+ 参数 LLM 的数据源重加权，在 Qwen-2.5-32B 上实现 GSM8K +9%、MATH +5.8% 的提升。

---

## 📂 其他 { #others }

**[A3Cg Esg Greenwashing](others/a3cg_esg_greenwashing.md)**

:   提出 A3CG 数据集和方面-行动分析任务（从可持续性声明中提取方面及其行动类型：已实施/计划中/不确定），通过跨类别泛化设置评估 NLP 方法抵御漂绿风险的鲁棒性，发现监督学习（GRACE F1=47.51）优于 LLM（Claude 3.5 F1=42.03）但泛化效率更差。

**[A Large And Balanced Corpus For Fine-Grained Arabic Readability Assessment](others/a_large_and_balanced_corpus_for_fine-grained_arabic_readability_assessment.md)**

:   构建 Barec（Balanced Arabic Readability Evaluation Corpus）——首个大规模细粒度阿拉伯语可读性评估语料库，包含 69K+ 句子（100万+词），覆盖 19 个可读性等级（从幼儿园到研究生），由 6 名专业阿拉伯语教育者标注，并在多种分级粒度（19/7/5/3级）上基准测试自动可读性评估模型。

**[A Little Human Data Goes A Long Way](others/a_little_human_data_goes_a_long_way.md)**

:   首次系统研究合成数据在事实验证和问答中能否替代人工标注——替换 90% 仅轻微下降，但替换最后 10% 严重退化；仅 125 条人工数据可显著提升纯合成模型，等效增益需 10 倍以上合成数据。

**[A Measure Of The System Dependence Of Automated Metrics](others/a_measure_of_the_system_dependence_of_automated_metrics.md)**

:   指出机器翻译自动评估指标存在被忽视的"系统依赖性"问题：同一指标分数对不同翻译系统对应不同的人类评分，提出 SysDep 度量来量化这一效应，揭示即使是 WMT23 最佳指标 XCOMET 也存在严重的系统依赖性导致错误排名。

**[A Practical Approach For Building Production-Grade Conversational Agents With Wo](others/a_practical_approach_for_building_production-grade_conversational_agents_with_wo.md)**

:   提出基于有向无环图（DAG）工作流的混合对话 Agent 框架——每个图节点有独立系统提示/工具/执行规则以处理特定场景约束。结合原型 Agent 数据收集、状态感知的响应掩码微调策略，在 Kakao 移动电商场景中任务准确率提升 52%、格式遵从度提升 50%，超越 GPT-4o。

**[A Semi-Supervised Scalable Unified Framework For E-Commerce Query Classification](others/a_semi-supervised_scalable_unified_framework_for_e-commerce_query_classification.md)**

:   提出电商查询分类的半监督可扩展统一框架 SSUF——三个可插拔模块：知识增强（LLM 世界知识+后验点击）解决短查询信息不足、标签增强（语义编码+半监督信号）打破对后验标签的依赖、结构增强（共现+语义+层级图 GCN）传播长尾标签梯度。已在 JD.COM 部署，离线和在线 A/B 实验均显著超越 SOTA。

**[A Spatio-Temporal Point Process For Fine-Grained Modeling Of Reading Behavior](others/a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)**

:   提出基于标记时空点过程（Marked Spatio-Temporal Point Process）的阅读行为概率模型——用 Hawkes 过程建模跳视（何时何处注视），用对数正态分布+卷积建模注视时长（持续多久），避免传统聚合测量的信息丢失。实证发现：Hawkes 模型比基线更好拟合跳视数据，但上下文 surprisal 作为预测因子仅带来边际改善——surprisal 理论难以解释细粒度眼动。

**[Acecoder Acing Coder Rl Via Automated](others/acecoder_acing_coder_rl_via_automated.md)**

:   构建 AceCode-87K（87K 编码题 + 138 万自动合成测试用例），训练代码专用 Reward Model（7B 超越 340B Nemotron），Best-of-N 提升 Llama-3.1-8B 平均 8.9 分，R1 风格从 base 直接 RL 仅 80 步 HumanEval+ 提升 22.5%。

**[Acord An Expert-Annotated Retrieval Dataset For Legal Contract Drafting](others/acord_an_expert-annotated_retrieval_dataset_for_legal_contract_drafting.md)**

:   构建首个面向合同起草的专家标注条款检索基准 ACORD——114 个律师编写的查询、126,000+ 查询-条款对、1-5 星相关性评分，聚焦责任限制/赔偿/控制权变更/最惠国等复杂条款；bi-encoder 检索 + LLM 点式重排序表现有前景但距律师需求仍有显著差距。LLM 直接起草对比律师修改暴露了多种缺陷。

**[Adaptive Retrieval Without Self-Knowledge Bringing Uncertainty Back Home](others/adaptive_retrieval_without_self-knowledge_bringing_uncertainty_back_home.md)**

:   对 35 种自适应检索方法（含 8 种最新方法和 27 种不确定性估计方法）进行了全面评测，发现经典的不确定性估计技术在效率和自知能力方面往往优于复杂的专用流水线，同时保持相当的 QA 性能。

**[Advancing Sequential Numerical Prediction In Autoregressive Models](others/advancing_sequential_numerical_prediction_in_autoregressive_models.md)**

:   提出 NTIL（Numerical Token Integrity Loss）解决自回归模型数值预测的两大缺陷——(1) token 级用 EMD 替代交叉熵保留数字间序数关系+指数位置加权，(2) 序列级通过可微数值构建+相对偏差度量评估整体数值误差。首次将 EMD 用于自回归模型优化，在目标检测/文本识别/数学推理上显著提升。

**[Aide Attribute-Guided Multi-Hop Data Expansion For Data Scarcity In Task-Specifi](others/aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)**

:   提出 AIDE——属性引导的多跳数据扩展框架，从仅 10 个种子数据点出发，通过提取主题/属性/关系三元组引导 LLM 多跳递归合成新数据，加入 Persona 增加多样性和残差连接防止偏离，在 Mistral-7B/Llama-3.1-8B/Llama-3.2-3B 上超越人工标注数据微调，比 Evol-Instruct 等 SOTA 提升 30%+。

**[Aligned But Blind Implicit Bias](others/aligned_but_blind_implicit_bias.md)**

:   揭示对齐训练的"种族盲视"副作用：对齐使 LLM 在歧义上下文中不再将 black/white 表征为种族概念，安全护栏因此无法激活，导致隐式偏见从 64.1% 飙升至 91.4%；反直觉地，在早期层注入种族感知激活（而非遗忘）可将隐式偏见从 97.3% 降至 42.4%。

**[Ambik Dataset Of Ambiguous Tasks In Kitchen Environment](others/ambik_dataset_of_ambiguous_tasks_in_kitchen_environment.md)**

:   提出 AmbiK，一个专门用于厨房环境中歧义指令检测的纯文本数据集，包含 1000 对歧义/非歧义指令，按三种歧义类型（用户偏好/常识/安全）分类标注，并评估了多种基于 conformal prediction 的歧义检测方法，发现现有方法在该基准上表现很差。

**[An Analysis Of Datasets Metrics And Models In Keyphrase Generation](others/an_analysis_of_datasets_metrics_and_models_in_keyphrase_generation.md)**

:   对关键短语生成（keyphrase generation）领域50+篇论文进行系统性分析，揭示了基准数据集高度相似、评估指标计算不一致导致性能被高估等关键问题，并发布了一个强力PLM-based模型以促进未来研究。

**[Analytickws Towards Exemplar-Free Analytic Class Incremental Learning For Small-](others/analytickws_towards_exemplar-free_analytic_class_incremental_learning_for_small-.md)**

:   提出 AnalyticKWS，一种无需存储历史样本的关键词检测增量学习方法，通过冻结特征提取器 + 递归最小二乘解析解更新分类器，在 GSC 和 SC-100 数据集上超过了所有基于样本回放的方法，且训练时间和内存开销极低。

**[Anything Goes A Crosslinguistic Study Of Impossible Language Learning In Lms](others/anything_goes_a_crosslinguistic_study_of_impossible_language_learning_in_lms.md)**

:   跨语言研究 LM 能否区分可能语言和不可能语言——在 12 种语言（4 个语系）上训练 GPT-2 Small 的可能/不可能/未见证变体，发现模型大体上能区分可能 vs 不可能语言（单语言内），但**跨语言**时区分能力减弱，且对类型学未见证语言（Greenberg Universal 20 的未见证词序）的区分仅在泛化测试中有效而在困惑度上无效——LM 有部分人类样的归纳偏置但弱于人类。

**[Are Any-To-Any Models More Consistent Across Modality Transfers Than Specialists](others/are_any-to-any_models_more_consistent_across_modality_transfers_than_specialists.md)**

:   本文提出 ACON 数据集和三种一致性评估标准（循环一致性、前向等变性、共轭等变性），发现当前 any-to-any 模型在逐点评估中并不比专用模型组合更具跨模态一致性，但通过多编辑操作的分布式分析可以观察到弱一致性。

**[Are Bias Evaluation Methods Biased](others/are_bias_evaluation_methods_biased.md)**

:   严格控制变量后比较三种主流偏见评估方法（结构化问答 BBQ、LLM-as-a-Judge、情感分析），发现不同方法对同一组 LLM 产生显著不同的偏见排名——偏见评估方法本身就是有偏的，企业不应依赖单一偏见基准来选择模型。

**[Arise Risk Adaptive Search](others/arise_risk_adaptive_search.md)**

:   提出 ARise 框架，将贝叶斯风险评估与动态 RAG 集成到蒙特卡洛树搜索中，解决知识增强推理中的错误传播和验证瓶颈问题，在多跳QA任务上平均准确率超 SOTA KAR 方法 23.10%，超 RAG-equipped 推理模型（DeepSeek-R1）25.37%。

**[Attention Entropy Parallel Encoding](others/attention_entropy_parallel_encoding.md)**

:   发现并行上下文编码导致 query token 的注意力熵异常升高是性能下降的关键因素，并提出 Attention Sink 共享前缀和 Selective Attention 两种免微调方法有效缓解该问题。

**[Autalic A Dataset For Anti-Autistic Ableist Language In Context](others/autalic_a_dataset_for_anti-autistic_ableist_language_in_context.md)**

:   提出 Autalic——首个专注于上下文中反自闭症残障歧视语言检测的数据集，包含 2,400 条 Reddit 句子及上下文标注，由神经多样性背景的专家标注，实验揭示当前 LLM（包括 DeepSeek、Llama3、Gemma2、Mistral）在识别反自闭症歧视语言时与人类判断严重不一致（平均 Cohen's Kappa 仅 0.091），凸显该任务的困难性。

**[Behavioural Vs Representational Systematicity In End-To-End Models An Opinionate](others/behavioural_vs_representational_systematicity_in_end-to-end_models_an_opinionate.md)**

:   区分行为系统性（模型能否正确处理新组合）与表征系统性（模型内部表征是否结构化），指出当前基准和模型主要测试行为系统性却常声称解决了 Fodor-Pylyshyn 对表征系统性的挑战。基于 Hadley (1994) 的三级分类（弱/准/强系统性）分析语言和视觉关键基准的测试范围，最终呼吁用机械可解释性方法在行为评估之上补充表征分析。

**[Better Embeddings With Coupled Adam](others/better_embeddings_with_coupled_adam.md)**

:   从理论上证明 Adam 优化器的逐 token 二阶矩是导致 LLM 词嵌入各向异性（均值偏移）的根因，提出 Coupled Adam——对嵌入层的二阶矩取词汇平均——消除了各向异性问题，并在大规模实验中提升了嵌入质量和下游性能。

**[Beyond Frameworks Multi Agent Collaboration](others/beyond_frameworks_multi_agent_collaboration.md)**

:   本文系统化地将多智能体协作分解为四个维度（治理模式、参与控制、交互模式、上下文管理），通过两个上下文依赖任务的大量实验证明：集中治理+指导者控制参与+有序交互+指导者摘要的组合最优，在保持甚至提升准确率的同时减少高达 93% 的 token 消耗。

**[Beyond Position The Emergence Of Wavelet-Like Properties In Transformers](others/beyond_position_the_emergence_of_wavelet-like_properties_in_transformers.md)**

:   通过频率分析和小波分解，揭示了使用 RoPE 位置编码的 Transformer 模型中注意力头自发涌现出类小波（wavelet-like）的多分辨率处理特性，以弥补 RoPE 在位置精度和频率分辨率之间的固有权衡。

**[Big-Bench Extra Hard](others/big-bench_extra_hard.md)**

:   为应对 BIG-Bench Hard 被前沿模型饱和的问题，Google DeepMind 推出 BIG-Bench Extra Hard (BBEH)，用 23 个更难的任务替换 BBH 中的对应任务，最强通用模型仅达 9.8%（调和平均）、最强推理模型达 44.8%，揭示了 LLM 在通用推理上的巨大差距。

**[Bone Soups Multi Objective Gen](others/bone_soups_multi_objective_gen.md)**

:   提出 Bone Soup 模型合并方法，通过先构造"骨架奖励"（多目标奖励的组合）训练骨架模型、再用对称循环矩阵映射确定合并系数，解决了 Rewarded Soup 中单目标模型合并的次优性问题，在三个多目标生成任务上实现更好的 Pareto 前沿和可控性。

**[Bregman Conditional Random Fields Sequence Labeling With Parallelizable Inferenc](others/bregman_conditional_random_fields_sequence_labeling_with_parallelizable_inferenc.md)**

:   提出 Bregman CRF (Bcrf)，一种基于均值正则化（mean regularization）的新型序列标注判别模型，使用迭代 Bregman 投影实现可并行化的推理算法，替代传统 CRF 中固有顺序的 Viterbi/Forward 算法，在 POS/NER/分词任务上性能与标准 CRF 持平但更快，且在有禁止标签转移约束的场景下优于 Mean Field 方法。

**[Building Better Avoiding Pitfalls In Developing Language Resources When Data Is ](others/building_better_avoiding_pitfalls_in_developing_language_resources_when_data_is_.md)**

:   通过对 81 位从事中低资源语言 NLP 研究的从业者进行问卷调查，系统性地揭示了低资源语言数据收集与标注中的质量问题和伦理隐患，特别是参与式研究框架中对社区贡献者的剥削问题，并提出了以语言社区为中心的改进建议。

**[Byte Latent Transformer](others/byte_latent_transformer.md)**

:   Meta FAIR提出BLT——首个在大规模（8B参数/4T字节）上匹配基于tokenizer的LLM性能的字节级架构，通过基于下一字节熵的动态分组（patching）将字节聚合为可变长度patch，在保持性能的同时实现最高50%推理FLOP节省，并开辟了"同时增大模型和patch尺寸"的全新scaling维度。

**[Cadreview Automatically Reviewing Cad Programs With Error Detection And Correcti](others/cadreview_automatically_reviewing_cad_programs_with_error_detection_and_correcti.md)**

:   提出 CAD 程序审查任务及 ReCAD 框架，基于参考图像自动检测 CAD 程序中的错误并生成修正反馈，构建了包含 20K+ 样本（8 类错误）的 CADReview 数据集。

**[Can Third Parties Read Our Emotions](others/can_third_parties_read_our_emotions.md)**

:   本文通过人类被试实验，系统比较了第三方标注者（人类标注者和LLM）与第一方（作者自标注）在情感识别任务中的对齐程度，发现第三方标注与作者真实情感之间存在显著差距，LLM虽优于人类标注者，但仍表现不佳；人口统计学相似性可提升标注质量。

**[Can Uniform Meaning Representation Help Gpt-4 Translate From Indigenous Language](others/can_uniform_meaning_representation_help_gpt-4_translate_from_indigenous_language.md)**

:   探索将统一意义表示（UMR）语义图纳入 GPT-4 提示中，翻译三种原住民语言（纳瓦霍语、阿拉帕霍语、库卡马语），发现在大多数情况下 UMR 的加入带来统计显著的性能提升。

**[Capacity Matters A Proof-Of-Concept For Transformer Memorization On Real-World D](others/capacity_matters_a_proof-of-concept_for_transformer_memorization_on_real-world_d.md)**

:   本文以SNOMED医学知识图谱为数据源，系统研究了decoder-only Transformer在结构化数据上的记忆容量，发现嵌入维度是决定学习速度和容量的主要因素，而增加层数收效甚微，Softmax激活函数表现最稳定。

**[Causal Tokenisation Bias](others/causal_tokenisation_bias.md)**

:   本文首次将 tokeniser 选择对语言模型输出的影响定义为"分词偏差"(tokenisation bias)，并利用因果推断中的断点回归设计(RDD)来量化这一效应——发现当一个 subword 被纳入词表时，其对应字符串的概率最高可提升 17 倍（小模型），揭示分词是语言建模中一个被低估的关键设计选择。

**[Cautious Next Token Prediction](others/cautious_next_token_prediction.md)**

:   提出 Cautious Next Token Prediction (CNTP)，一种无需训练的自适应解码策略：在模型预测熵较高（不确定）时采样多条候选路径至标点处，选择困惑度最低的路径作为最终续写，从而在不牺牲多样性的前提下显著提升准确性。

**[Chartlens Fine-Grained Visual Attribution In Charts](others/chartlens_fine-grained_visual_attribution_in_charts.md)**

:   提出图表的事后细粒度视觉归因（Post-Hoc Fine-grained Visual Attribution）任务，设计 ChartLens 算法利用分割技术标记图表元素并通过 Set-of-Marks 提示多模态 LLM 进行精确归因，同时构建 ChartVA-Eval 基准，在三类图表上取得 26-66% 的 F1 提升。

**[Childmandarin A Comprehensive Mandarin Speech Dataset For Young Children Aged 3-](others/childmandarin_a_comprehensive_mandarin_speech_dataset_for_young_children_aged_3-.md)**

:   提出 ChildMandarin，一个面向 3-5 岁幼儿的普通话语音数据集，包含 397 名说话人、41.25 小时语音、覆盖中国 22 个省级行政区，并在 ASR 和说话人验证任务上提供了全面的基线评估。

**[Chulo Chunk-Level Key Information Representation For Long Document Understanding](others/chulo_chunk-level_key_information_representation_for_long_document_understanding.md)**

:   ChuLo 的核心不是单纯把长文档切小，而是先在全文范围内找出最关键的语义短语，再把这些关键信息重新注入每个 chunk 的表示里，从而在只用紧凑块表示的前提下，同时保住全局语义和细粒度 token 信息。

**[Citeeval Principle-Driven Citation Evaluation For Source Attribution](others/citeeval_principle-driven_citation_evaluation_for_source_attribution.md)**

:   本文提出 CiteEval，一个基于原则驱动的引用评估框架，通过考虑完整检索上下文、超越检索的多种上下文以及细粒度评价标准，构建了 CiteBench 基准和 CiteEval-Auto 自动指标，在引用质量评估上显著优于基于 NLI 的现有方法。

**[Clac At Semeval-2025 Task 6 A Multi-Architecture Approach For Corporate Environm](others/clac_at_semeval-2025_task_6_a_multi-architecture_approach_for_corporate_environm.md)**

:   本文针对SemEval-2025 Task 6（PromiseEval）的企业ESG报告承诺验证任务，探索了三种递进的模型架构：ESG-BERT基线、语言特征增强版、以及融合注意力池化和多目标学习的联合子任务模型，最终以0.5268的私榜分数略超基线（0.5227），验证了语言特征工程和多任务学习在ESG承诺验证中的有效性。

**[Coachme Sport Instruction](others/coachme_sport_instruction.md)**

:   提出 CoachMe，通过对比学习者动作与参考动作的差异（时间+物理两个维度），自动生成运动特异性的教练指导文本，在花样滑冰和拳击上分别超过 GPT-4o 31.6% 和 58.3%（G-Eval）。

**[Coam Corpus Of All-Type Multiword Expressions](others/coam_corpus_of_all-type_multiword_expressions.md)**

:   构建了一个高质量的全类型多词表达(MWE)识别数据集 CoAM（1.3K句），通过多步质量保障流程解决了现有数据集标注不一致的问题，并发现微调大语言模型在 MWE 识别任务上显著优于此前的 SOTA 方法 MWEasWSD。

**[Cola Collaborative Low-Rank Adaptation](others/cola_collaborative_low-rank_adaptation.md)**

:   提出 CoLA，一种灵活的 LoRA 架构，打破矩阵 A 和 B 之间的固定数量约束（#A=M, #B=N），并设计三种协作策略（全协作/随机协作/启发式协作），结合扩展的 PiSSA 初始化，在低样本场景下显著优于现有 PEFT 方法。

**[Comfyui-Copilot An Intelligent Assistant For Automated Workflow Development](others/comfyui-copilot_an_intelligent_assistant_for_automated_workflow_development.md)**

:   提出 ComfyUI-Copilot，一个基于 LLM 的层次化多 agent 框架，作为 ComfyUI 插件提供智能节点/模型推荐和一键式工作流构建，覆盖 7K 节点、62K 模型和 9K 工作流的知识库，在线服务 22 国的 19K 用户并处理了 85K+ 查询。

**[Commonsense Arab Culture](others/commonsense_arab_culture.md)**

:   构建首个阿拉伯文化特定常识推理数据集 ArabCulture（3482 道由母语者原创的题目，覆盖 13 国×54 主题），评估多种 LLM 发现即使 32B 参数模型也在文化常识推理上表现不佳，且不同地区表现差异显著，地理/文化上下文线索的加入仅部分有效。

**[Completing A Systematic Review In Hours](others/completing_a_systematic_review_in_hours.md)**

:   提出 InsightAgent，一个以人为中心的交互式多 Agent 系统，通过语义聚类分区、多 agent 并行阅读和实时用户交互，将医学系统综述的撰写时间从数月缩短到约 1.5 小时，达到人类撰写质量的 79.7%。

**[Conceptcarve Dynamic Realization Of Evidence](others/conceptcarve_dynamic_realization_of_evidence.md)**

:   提出 ConceptCarve 框架，通过 LLM 与传统检索器的交互式协作，动态构建概念树来表征证据在特定社区中的实现形式，解决了证据检索中的推理鸿沟和领域敏感性两大挑战。

**[Conect Dataset Overcoming Data Scarcity In Context-Aware E-Commerce Mt](others/conect_dataset_overcoming_data_scarcity_in_context-aware_e-commerce_mt.md)**

:   构建了 ConECT——一个包含 11,400 句对的捷克-波兰电商产品翻译数据集，配有图片和类目路径元数据，系统评估了 VLM、NMT+类目路径和 NMT+图像描述三种上下文感知翻译方法，证明视觉和类目上下文能提升翻译质量，但合成图像描述反而有害。

**[Consistent Client Simulation For Motivational Interviewing-Based Counseling](others/consistent_client_simulation_for_motivational_interviewing-based_counseling.md)**

:   提出一种面向动机性访谈（MI）心理咨询的一致性客户模拟框架，通过状态转换、行动选择、信息选择和回复生成四个模块，确保模拟客户的行为与其预设的画像（动机、信念、改变计划、配合度）保持一致，在自动和专家评估中均优于基线方法。

**[Consultant Decoding Yet Another Synergistic Mechanism](others/consultant_decoding_yet_another_synergistic_mechanism.md)**

:   提出 Consultant Decoding (CD)，一种基于目标模型负对数似然（NLL）验证 draft token 的新型协同解码机制，相比传统 Speculative Decoding 的似然比验证方法，能大幅提升接受率、降低大模型调用频率，同时保持甚至超越目标模型的生成质量。

**[Coral Speculative Drafting](others/coral_speculative_drafting.md)**

:   CORAL 通过跨步表示对齐（CSRA）改进多步训练中 draft 模型的特征一致性，并用权重分组机制压缩大词表 LM head 的推理延迟，在 LLaMA3/Qwen2.5 上实现 2.50-4.07× 加速，超越 EAGLE-2 和 HASS。

**[Cortexdebate Debating Sparsely And Equally For Multi-Agent Debate](others/cortexdebate_debating_sparsely_and_equally_for_multi-agent_debate.md)**

:   提出 CortexDebate，一种受人脑皮层工作机制启发的多智能体辩论方法，通过构建稀疏动态辩论图和基于 McKinsey 信任公式的评估模块（MDM），同时解决了现有 MAD 方法中"输入上下文过长"和"过度自信导致不平等辩论"两大核心问题。

**[Cramming Tokens Embedding Capacity](others/cramming_tokens_embedding_capacity.md)**

:   通过逐样本优化方法将文本压缩到可训练的 [mem] 向量中，发现 Llama-3.1-8B 可以将 1568 个 token 无损压缩到单个输入向量中，揭示了现有方法（约 x10 压缩比）与实际可达极限（x1500+）之间存在两个数量级的差距。

**[Decoding Knowledge Attribution In Mixture-Of-Experts A Framework Of Basic-Refine](others/decoding_knowledge_attribution_in_mixture-of-experts_a_framework_of_basic-refine.md)**

:   提出跨层级知识归因算法，系统解析 MoE 模型中共享专家与路由专家的"基础-精炼"协作框架，揭示 MoE 相比稠密模型实现 31% 更高的逐层效率，并通过语义驱动路由机制（注意力头-专家相关性 r=0.68）和专家阻断实验验证了架构深度对鲁棒性的决定性影响。

**[Decoding Reading Goals From Eye Movements](others/decoding_reading_goals_from_eye_movements.md)**

:   本文首次提出从眼动轨迹中解码读者阅读目标（信息检索 vs. 普通阅读）的任务，通过 12 种模型的系统比较发现基于 Transformer 的扫视路径+语言建模方案（RoBERTa-Eye-F）最优，可在阅读早期即实现高精度实时预测。

**[Deeprtl2 A Versatile Model For Rtl-Related Tasks](others/deeprtl2_a_versatile_model_for_rtl-related_tasks.md)**

:   DeepRTL2是首个统一处理RTL（寄存器传输级）相关生成任务与嵌入任务的LLM，通过精心构建的数据集和GRIT训练策略，在代码生成、代码理解、自然语言代码搜索、功能等价检查和性能预测五大任务上达到SOTA。

**[Demo Reframing Dialogue Interaction With Fine-Grained Element Modeling](others/demo_reframing_dialogue_interaction_with_fine-grained_element_modeling.md)**

:   本文提出对话元素建模（Dialogue Element Modeling）这一新任务，系统定义了对话生命周期中从"前奏"到"尾声"的全面元素体系，构建了包含元素感知和对话智能体交互两大能力的DEMO benchmark，并通过模仿学习训练DEMO agent在域内外任务上均表现优异。

**[Developmentally-Plausible Working Memory Shapes A Critical Period For Language A](others/developmentally-plausible_working_memory_shapes_a_critical_period_for_language_a.md)**

:   受"Less-is-More"假说启发，本文提出 DynamicLimit-Exp 方法，将人类工作记忆在关键期内的指数增长特征集成到语言模型训练中（通过动态调节 ALiBi 斜率），在 Child-Directed Speech 数据上训练的 GPT-2 模型在句法评估中显著优于无记忆约束和静态约束的基线。

**[Distractor Gen Multiple Choice](others/distractor_gen_multiple_choice.md)**

:   提出选择题干扰项生成的三步流水线：(1) 训练配对排序器预测学生误选哪个干扰项；(2) 用排序器构造偏好数据集；(3) 用 DPO 训练生成器产生更具迷惑性的干扰项。在 CS 领域（Python/DB/ML）实验中，生成的干扰项更难以区分，且题目鉴别指数(DI)更高。

**[Do Not Abstain Identify And Solve The Uncertainty](others/do_not_abstain_identify_and_solve_the_uncertainty.md)**

:   本文提出ConfuseBench基准和基于inquiry answer唯一性判断不确定性来源的方法，并通过InteractDPO在策略训练中动态生成偏好对来提升inquiry质量，使LLM能主动识别并解决不确定性而非简单回避。

**[Docagent A Multi-Agent System For Automated Code Documentation Generation](others/docagent_a_multi-agent_system_for_automated_code_documentation_generation.md)**

:   提出 DocAgent，一个基于拓扑依赖排序的多智能体代码文档生成系统，通过 Reader-Searcher-Writer-Verifier 协作流程增量构建上下文，在完整性、实用性和真实性三个维度上显著优于 FIM 和 Chat 基线。

**[Dolphin Moving Towards Closed-Loop Auto-Research Through Thinking Practice And F](others/dolphin_moving_towards_closed-loop_auto-research_through_thinking_practice_and_f.md)**

:   > 提出 Dolphin，一个闭环自动科研框架，包含"想法生成→实验验证→结果反馈"三阶段循环，通过任务属性引导的论文排序和异常回溯引导的调试流程，在 3D 分类等任务上自动提出并验证了接近人类设计 SOTA 的方法。

**[Domix An Efficient Framework For Exploiting](others/domix_an_efficient_framework_for_exploiting.md)**

:   提出 DoMIX，将各领域知识用独立 LoRA 模块存储后通过对角初始化的 bridge 矩阵在微调时灵活组合利用，在持续领域适应预训练场景下减少 58% 预训练时间和 87% GPU 内存，同时性能超越 SOTA。

**[Dpp Diverse Multidoc Summary](others/dpp_diverse_multidoc_summary.md)**

:   用行列式点过程（DPP）替代 LLM 的隐式内容选择来生成多样化多文档摘要——将任务分解为"提取原子关键点→DPP 选择多样子集→LLM 重写为摘要"三步，解决了 LLM 的"lost in the middle"问题，在 DiverseSumm 基准上一致提升了源覆盖率。

**[Dress Dataset Rubric Based Essay Scoring Efl Writing](others/dress_dataset_rubric_based_essay_scoring_efl_writing.md)**

:   发布 DREsS——一个面向 EFL（英语作为外语）写作教育的大规模标准化评分量规数据集（48.9K 样本），并提出基于文本损坏的 CASE 数据增强策略，将基线性能提升 45.44%。

**[Drs Deep Question Reformulation With Structured Output](others/drs_deep_question_reformulation_with_structured_output.md)**

:   提出 DRS（Deep Question Reformulation with Structured Output），一种零样本方法，通过实体驱动的 DFS 搜索 + 结构化输出约束，将 GPT-3.5 的问题重构准确率从 23.03% 提升至 70.42%，使 LLM 能有效帮助用户将不可回答的问题转化为可回答的形式。

**[Dta Llama Parallel Tool Invocation](others/dta_llama_parallel_tool_invocation.md)**

:   提出 DTA-Llama，将传统树搜索的串行工具调用路径转换为有向无环图（DAG）结构实现并行调用，设计 Process/Thread 推理框架使 LLM 在每轮中可分解任务并并行执行多个工具，在 StableToolBench 上使 Llama2-7B 达到 GPT-3.5 Parallel Function Calling 的水平。

**[Dynamic Label Name Refinement for Few-Shot Dialogue Intent Classification](others/dynamic_label_name_refinement_for_few-shot_dialogue_intent_classification.md)**

**[Efficient Opamp Adaptation For Zoom Attention To Golden Contexts](others/efficient_opamp_adaptation_for_zoom_attention_to_golden_contexts.md)**

:   受运算放大器（OpAmp）电路启发，提出 OpAmp Adaptation 方法通过 adapter 高效改造预训练 Transformer 的注意力机制，在噪声上下文场景下让 LLM 更精准聚焦于 golden document，Qwen2.5-OpAmp-72B 在多个噪声上下文基准上超越 DeepSeek-V3 和 GPT-4o。

**[Enhancing Conversational Agents With Theory Of Mind Aligning Beliefs Desires And](others/enhancing_conversational_agents_with_theory_of_mind_aligning_beliefs_desires_and.md)**

:   本文探索了从开源 LLM（LLaMA）内部表征中提取心智理论（ToM）相关信息的可行性，并利用 BDI（信念-愿望-意图）框架操纵这些表征来生成更符合人类社交认知的对话回复，ToM 对齐后的模型在 3B 和 8B 上分别达到 67% 和 63% 的胜率。

**[Enhancing Fol Entailment](others/enhancing_fol_entailment.md)**

:   系统性研究 Transformer 在一阶逻辑蕴涵任务中的泛化推理能力，揭示了查询语法、token 嵌入和 Transformer 架构（特别是位置编码）的影响，并提出 TEGA（Transformer Encoder with Guided Attention）在相对位置编码设定下显著提升逻辑推理性能。

**[Enhancing Marker Scoring Accuracy Through Ordinal Confidence Modelling In Educat](others/enhancing_marker_scoring_accuracy_through_ordinal_confidence_modelling_in_educat.md)**

:   本文提出了一种基于核加权序数分类交叉熵（KWOCCE）的置信度建模方法，通过利用 CEFR 等级的序数结构和分数分箱策略，实现最高 47% 评分在 100% CEFR 一致性下释放，99% 在 ≥95% 一致性下释放，显著优于无置信度过滤时的约 92%。

**[Enhancing The Comprehensibility Of Text Explanations Via Unsupervised Concept Di](others/enhancing_the_comprehensibility_of_text_explanations_via_unsupervised_concept_di.md)**

:   提出 ECO-Concept 框架，通过 slot attention 机制自动提取文本概念，并利用 LLM 作为人类代理评估概念的可理解性，用可理解性反馈损失指导模型微调，在无概念标注的情况下实现了兼具高分类精度和人类可理解性的概念解释。

**[Entailed Between The Lines Incorporating Implication Into Nli](others/entailed_between_the_lines_incorporating_implication_into_nli.md)**

:   形式化定义"隐含蕴涵"（implied entailment）任务，将传统NLI的三分类扩展为四分类（隐式蕴涵/显式蕴涵/中立/矛盾），构建包含10K前提和40K假设的INLI数据集，实验表明微调后的模型能有效识别隐含蕴涵并跨领域泛化。

**[Entity Framing And Role Portrayal In The News](others/entity_framing_and_role_portrayal_in_the_news.md)**

:   本文构建了一个包含 5 种语言、1378 篇新闻文章、5800+ 实体标注的多语言层次化实体框架语料库，提出含 22 种精细角色的叙事角色分类体系（主角 / 反派 / 无辜者三大框架下），并在微调多语言 Transformer 和 LLM 层次零样本学习上建立了基准。

**[Entropy-Uid A Method For Optimizing Information Density](others/entropy-uid_a_method_for_optimizing_information_density.md)**

:   提出 Entropy-UID 方法，在自回归语言模型的解码过程中联合最小化熵和 surprisal 的加权组合，以实现信息密度的均匀分布。在 WikiText-2、OpenWebText 和 WMT 数据集上，该方法实现了最低的熵标准差（≈2.8）和稳定的 surprisal（≈5.7），优于单目标优化策略。

**[Epicode Boosting Model Performance Beyond Training With Extrapolation And Contra](others/epicode_boosting_model_performance_beyond_training_with_extrapolation_and_contra.md)**

:   提出 EpiCoDe，一种结合模型外推（Model Extrapolation）和对比解码（Contrastive Decoding）的无训练方法，在数据稀缺场景中通过参数空间外推和推理时logit差异对比来提升微调模型性能，并从logit误差角度给出了理论分析框架。

**[Epman Episodic Memory Attention For Generalizing To Longer Contexts](others/epman_episodic_memory_attention_for_generalizing_to_longer_contexts.md)**

:   提出 EpMAN 方法，通过情景记忆模块估计上下文块的相对相关性，用该相关性重新加权解码器的自注意力（differentiating attention），配合噪声训练和注意力范围扩展策略，在 16k-256k 上下文长度范围内实现了比长上下文 LLM 和 RAG 更强且更鲁棒的表现。

**[Evaluating Design Decisions For Dual Encoder-Based Entity Disambiguation](others/evaluating_design_decisions_for_dual_encoder-based_entity_disambiguation.md)**

:   系统评估了 Dual Encoder 在实体消歧（ED）任务中的关键设计选择（损失函数、相似度度量、标签语义化格式、负采样策略），并基于最优设计构建了 VerbalizED 系统，在 ZELDA 基准上达到了新的 SOTA，同时探索了一种迭代预测策略来利用已消歧的邻居实体改进困难样本。

**[Evaluating The Evaluation Of Diversity In Commonsense Generation](others/evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)**

:   对常识生成（GCR）任务中的12种多样性评估指标进行系统元评估，发现基于形式（n-gram）的指标在低质量生成上严重高估多样性，而基于内容（句子嵌入）的指标与人类判断一致性更高，推荐未来 GCR 研究使用 VS-Embed 或 Chamfer Distance 等内容级指标。

**[Explaining Matters Leveraging Definitions And Semantic Expansion For Sexism Dete](others/explaining_matters_leveraging_definitions_and_semantic_expansion_for_sexism_dete.md)**

:   针对在线性别歧视检测中的数据稀疏和细粒度分类歧义问题，提出两种基于prompt的数据增强技术——定义驱动数据增强（DDA）利用类别定义生成语义对齐的合成样本，上下文语义扩展（CSE）通过分析模型错误的语义特征丰富训练数据——并结合 Mistral-7B 回退集成策略，在 EDOS 数据集上实现全任务 SOTA。

**[Explicit And Implicit Data Augmentation For Social Event Detection](others/explicit_and_implicit_data_augmentation_for_social_event_detection.md)**

:   本文提出SED-Aug，一个结合显式（LLM文本增强）和隐式（特征空间扰动）的双重数据增强框架用于社交事件检测，在Twitter2012和Twitter2018上分别超越最优基线17.67%和15.57%的平均F1。

**[Expo Model Extrapolation](others/expo_model_extrapolation.md)**

:   基于"对齐训练仅产生微小参数变化"的观察，提出ExPO方法——通过放大SFT→DPO的参数变化方向（$\theta_2 = \theta_1 + \alpha\Delta\theta$），在零额外训练开销下提升对齐性能，使仅训练20%步骤的DPO模型超越完整训练的版本。

**[Fastdraft How To Train Your Draft](others/fastdraft_how_to_train_your_draft.md)**

:   提出 FastDraft，一套高效的 draft 模型预训练与对齐流程，可在24小时内用单节点8卡训练出约50M参数的 draft 模型，配合 Speculative Decoding 实现最高3倍内存带宽加速和2倍实际推理加速。

**[Fcmr Robust Evaluation Of Financial Cross-Modal Multi-Hop Reasoning](others/fcmr_robust_evaluation_of_financial_cross-modal_multi-hop_reasoning.md)**

:   构建了金融领域跨模态多跳推理基准 FCMR，包含文本、表格和图表三种模态，分 Easy/Medium/Hard 三个难度等级，最强模型 Claude 3.5 Sonnet 在 Hard 级别仅达 30.4% 准确率，揭示了 MLLM 在信息检索阶段的关键瓶颈。

**[Feat A Preference Feedback Dataset Through A Cost-Effective Auto-Generation And ](others/feat_a_preference_feedback_dataset_through_a_cost-effective_auto-generation_and_.md)**

:   提出 FEAT 框架，通过 LLM 自动生成和标注教师反馈偏好数据集用于英语辅导系统，发现仅混入 5-10% 人工标注数据就能超越 100% 人工数据的排序性能。

**[Federated Lora Heterogeneous](others/federated_lora_heterogeneous.md)**

:   提出 LoRA-A²（Low Rank Adaptation with Alternating freeze and Adaptive rank selection），通过交替冻结 A/B 矩阵解决联邦 LoRA 聚合不一致问题，并结合自适应秩选择机制在大幅压缩上传参数量（最高减少 99.8%）的同时保持鲁棒性，尤其在低秩+高数据异构场景下显著优于现有方法。

**[Follow-Up Question Generation For Enhanced Patient-Provider Conversations](others/follow-up_question_generation_for_enhanced_patient-provider_conversations.md)**

:   提出 FollowupQ 多智能体框架，结合 EHR 推理、鉴别诊断和消息澄清三类 Agent，为异步医患对话自动生成个性化追问列表，在真实和半合成数据集上分别比基线提升 17% 和 5% 的 RIM 分数，将医生需要额外发送的信息收集消息减少 34%。

**[Foreplay Polish Erotic Detection](others/foreplay_polish_erotic_detection.md)**

:   构建了首个波兰语色情内容检测数据集 forePLay（24,768 句，5 类标签），提出涵盖模糊性、暴力和社会不可接受行为的多维标注体系，评估发现专用波兰语模型显著优于多语言模型，且 Transformer 编码器模型在不平衡类别处理上表现最强。

**[Fractal Fine-Grained Scoring From Aggregate Text Labels](others/fractal_fine-grained_scoring_from_aggregate_text_labels.md)**

:   提出 FRACTAL 方法，将回复级别（response-level）的聚合标签分解为句子级别（sentence-level）的伪标签，利用多实例学习（MIL）和标签比例学习（LLP）技术结合先验信息（文档-句子余弦相似度）训练句子级评分模型，覆盖检索、问答、摘要和数学推理四类任务。

**[Frictional Agent Alignment](others/frictional_agent_alignment.md)**

:   提出摩擦对齐框架 FAAF（Frictional Agent Alignment Framework），通过双策略（frictive state policy + intervention policy）目标函数，训练 LLM 在协作对话中识别信念冲突并生成促进反思与审议的"摩擦"干预，超越 DPO/IPO/PPO 等对齐方法。

**[From Lists To Emojis How Format Bias Affects Model Alignment](others/from_lists_to_emojis_how_format_bias_affects_model_alignment.md)**

:   本文系统研究了 RLHF 中偏好模型（包括人类评估者、GPT-4 和开源模型）对粗体、列表、emoji 等格式模式的偏好偏差，展示了不到 1% 的偏差数据即可显著注入偏差，并提出了双头奖励模型的去偏方法。

**[Ga-S3 Comprehensive Social Network Simulation With Group Agents](others/ga-s3_comprehensive_social_network_simulation_with_group_agents.md)**

:   提出基于"群体智能体"（Group Agent）的社交网络模拟系统 GA-S3，将具有相似行为的个体聚合为群体代理，通过层次化生成、马尔可夫网络推理和行为模块实现大规模社交网络的高效精确模拟。

**[Gear Generation Augmented Retrieval](others/gear_generation_augmented_retrieval.md)**

:   GeAR 在传统 bi-encoder 检索框架上引入融合编码器和文本解码器，通过生成任务增强检索模型对文档内部细粒度语义的理解能力，同时不增加全局检索的计算开销。

**[Generating Synthetic Relational Tabular Data Via Structural Causal Models](others/generating_synthetic_relational_tabular_data_via_structural_causal_models.md)**

:   本文扩展了 TabPFN 的基于结构因果模型（SCM）的合成数据生成方法，提出了一个能够生成多表关联（relational）合成表格数据的框架，通过耦合节点和隐因果关系实现跨表依赖建模。

**[Genre A French Gender-Neutral Rewriting System Using Collective Nouns](others/genre_a_french_gender-neutral_rewriting_system_using_collective_nouns.md)**

:   GeNRe 是首个法语性别中性重写系统，利用集体名词（collective nouns）替代阳性泛指（masculine generics），提出规则系统、微调模型和指令模型三种方案，其中规则系统和 Claude 3 Opus + 词典方案效果最好。

**[Getreason Enhancing Image Context Extraction Through Hierarchical Multi-Agent Re](others/getreason_enhancing_image_context_extraction_through_hierarchical_multi-agent_re.md)**

:   提出 GETReason，一个层级化多智能体框架，通过将公共事件图像的上下文提取分解为地理空间、时间和事件三个子任务，并由专门化的 Agent 协作完成，实现比现有方法更准确的图像上下文推理。

**[Gpt-4 As A Homework Tutor Can Improve Student Engagement And Learning Outcomes](others/gpt-4_as_a_homework_tutor_can_improve_student_engagement_and_learning_outcomes.md)**

:   在意大利高中进行了为期 8 周的随机对照试验（RCT），用 GPT-4 替代传统英语作业作为互动辅导工具，发现 GPT-4 组学生在参与度（有趣性、资源充分性显著提升）和特定条件下的学习增益（三年级 Cohen's d=0.603）方面有所改善，仅需教师提供作业目标和描述即可实施，幻觉率低于 1%，且所有在校学生均表示希望继续使用。

**[Graph-Guided Cross-Composition Feature Disentanglement For Compositional Zero-Sh](others/graph-guided_cross-composition_feature_disentanglement_for_compositional_zero-sh.md)**

:   DCDA 提出图引导的跨组合特征解耦方案，通过双适配器（L-Adapter 用于文本端 GNN 特征聚合、V-Adapter 用于视觉端跨注意力解耦）注入冻结 CLIP，在组合零样本学习任务上显著超越现有方法。

**[Graph-Structured Trajectory Extraction From Travelogues](others/graph-structured_trajectory_extraction_from_travelogues.md)**

:   提出使用图结构（而非序列）表示旅行轨迹，构建了包含地理层级关系（inclusion）和时间顺序关系（transition）的访问顺序图，并建立了日语游记轨迹提取的基准数据集ATD-VSO。

**[Graphically Speaking Unmasking Abuse In Social Media With Conversation Insights](others/graphically_speaking_unmasking_abuse_in_social_media_with_conversation_insights.md)**

:   提出一种基于图注意力网络（GAT）的上下文感知滥用语言检测框架，将 Reddit 对话建模为图结构（节点=评论，边=回复关系），利用基于 Reddit 界面渲染逻辑的 affordance-based 图裁剪策略保留关键上下文，3 层 GAT 模型达到 F1=0.7624，显著优于无上下文基线和扁平化上下文方法，在上下文敏感样本上提升尤为明显（+4.75%）。

**[Guidelines For Fine-Grained Sentence-Level Arabic Readability Annotation](others/guidelines_for_fine-grained_sentence-level_arabic_readability_annotation.md)**

:   本文提出了 BAREC 语料库及其标注指南，这是一个拥有 69K+ 句子、覆盖 19 个可读性等级的大规模阿拉伯语句子级可读性评估资源，并在此基础上建立了自动可读性评估的基准模型。

**[Hanging In The Balance Pivotal Moments In Crisis Counseling Conversations](others/hanging_in_the_balance_pivotal_moments_in_crisis_counseling_conversations.md)**

:   本文提出了一种无监督方法来检测对话中的"关键时刻"（pivotal moments）——即下一步回应可能极大影响对话结局的节点，并在危机心理咨询场景中验证了该方法的有效性。

**[Hard Negative Mining For Domain-Specific Retrieval In Enterprise Systems](others/hard_negative_mining_for_domain-specific_retrieval_in_enterprise_systems.md)**

:   本文提出了一种面向企业级领域特定检索的可扩展硬负样本挖掘框架，通过融合多种嵌入模型、PCA 降维和双语义条件筛选来动态选择高质量硬负样本，在内部云服务数据集和公开基准上均取得了显著提升。

**[Hash-Rag Bridging Deep Hashing With Retriever For Efficient Fine Retrieval And A](others/hash-rag_bridging_deep_hashing_with_retriever_for_efficient_fine_retrieval_and_a.md)**

:   Hash-RAG 将深度哈希技术系统集成到 RAG 框架中，实现了仅需传统方法 10% 检索时间的高效检索，并通过 Prompt-Guided Chunk-to-Context（PGCC）模块在保持效率的同时提升了生成质量。

**[Hata Trainable And Hardware-Efficient Hash-Aware Top-K Attention For Scalable La](others/hata_trainable_and_hardware-efficient_hash-aware_top-k_attention_for_scalable_la.md)**

:   HATA 提出了一种将 learning-to-hash 技术集成到 top-k 注意力机制的方法，通过将查询和键映射为二进制哈希码来获取相对 qk 分数排序（而非绝对分数估计），在保持模型精度的同时实现了相对全注意力最高 7.2 倍的加速。

**[Helpsteer3 Human-Annotated Feedback And Edit Data To Empower Inference-Time Scal](others/helpsteer3_human-annotated_feedback_and_edit_data_to_empower_inference-time_scal.md)**

:   NVIDIA 发布 HelpSteer3 数据集（7000+标注员、80+国家），训练专用的 Feedback 和 Edit 模型，在推理时通过"初始响应→反馈→编辑"循环实现开放域通用任务的推理时扩展，基于 Llama 3 系列 70B 模型在 Arena Hard 上达到 92.7 分，超越 OpenAI o1-preview (90.4) 和 DeepSeek R1 (92.3)。

**[Hierarchical Attention Generates Better Proofs](others/hierarchical_attention_generates_better_proofs.md)**

:   提出 Hierarchical Attention 正则化方法，通过建立五层语义层次结构来引导 LLM 的注意力机制，使其与数学推理的自然信息流对齐，在 miniF2F 和 ProofNet 上分别提升证明成功率 2.05% 和 1.69%，同时降低证明复杂度 23.81% 和 16.50%。

**[Hierarchical Bracketing Dep Parsing](others/hierarchical_bracketing_dep_parsing.md)**

:   提出层次化括号编码家族用于依存句法分析的序列标注范式，证明现有4-bit编码是该家族的非最优特例，推导出仅需12个标签的最优编码，并将其推广到处理任意非投射性。

**[Hierarchical Memory Wikipedia Gen](others/hierarchical_memory_wikipedia_gen.md)**

:   提出 Memory Organization-based Generation（MOG）框架，从网页文档中提取细粒度记忆单元（factoid），通过递归聚类-摘要算法组织为层次化 Wikipedia 大纲结构，使每个章节都有直接的记忆支撑，在 FreshWiki 和 WikiStart 数据集上信息量、引用率和可验证性全面超越 RAG 和 STORM 基线。

**[Hippro Counterspeech Gen](others/hippro_counterspeech_gen.md)**

:   提出 HiPPrO 两阶段框架用于多条件反仇恨言论生成——第一阶段通过层次化前缀学习在多个属性（策略+情感）空间中优化反言论生成，第二阶段用无参考无奖励的偏好优化提升建设性，策略一致性提升 ~38%，ROUGE 指标提升 2-3%。

**[How To Mitigate Overfitting In Weak-To-Strong Generalization](others/how_to_mitigate_overfitting_in_weak-to-strong_generalization.md)**

:   提出两阶段训练框架解决弱到强泛化中的过拟合问题：第一阶段通过基于不确定性的过滤提高弱监督信号质量，第二阶段用已微调的强模型为被丢弃的难题重新生成答案以恢复问题质量，在 GSM8k 和 MATH 上将 PGR 从 7.19% 提升到 120.50%。

**[Hyperbole Metaphor Detection](others/hyperbole_metaphor_detection.md)**

:   提出 EmoBi 框架，通过情感分析→情感引导的域映射→双向动态交互三阶段 prompting 流程，利用 LLM 挖掘夸张和隐喻背后的情感线索及二者的互促关系，在四个数据集上大幅超越 SoTA（TroFi 上夸张检测 F1 提升 28.1%，HYPO-L 上隐喻检测 F1 提升 23.1%）。

**[I0T Embedding Standardization Method Towards Zero Modality Gap](others/i0t_embedding_standardization_method_towards_zero_modality_gap.md)**

:   提出 I0T 框架，通过发现并消除 CLIP 中图像/文本编码器各自学到的模态特异性特征（表现为归一化嵌入中的峰值激活），将模态差距降低至接近零，同时保持甚至提升下游任务性能，并提出了比 CLIPScore 更具可解释性的自动评估指标 I0T-Score。

**[If Attention Serves As A Cognitive](others/if_attention_serves_as_a_cognitive.md)**

:   通过将 Transformer Grammar（TG）的注意力机制与人类阅读时间数据关联，首次证明在句法结构上操作的注意力比在 token 序列上操作的普通 Transformer 注意力能更好地预测人类阅读行为，揭示人类句子处理涉及"句法结构+词序列"的双重记忆表征。

**[If Attention Serves As A Cognitive Model Of Human Memory Retrieval What Is The P](others/if_attention_serves_as_a_cognitive_model_of_human_memory_retrieval_what_is_the_p.md)**

:   本文探究 Transformer Grammar（TG）的注意力机制能否作为人类记忆检索的认知模型，通过 Normalized Attention Entropy（NAE）将模型与人类阅读时间关联，发现基于句法结构的注意力比基于 token 的注意力更能解释人类句子处理行为，且两者提供独立互补的贡献。

**[Implicit Arguments Video Instructions](others/implicit_arguments_video_instructions.md)**

:   提出 Implicit-VidSRL 数据集与 iSRL-Qwen2-VL 模型，针对过程性视频指令中省略的隐含论元（食材成分）进行预测，通过 SRL 框架将多步指令分解为 {verb, what, where/with} 三元组，在银标数据上微调后在隐含论元 F1 上超越 GPT-4o 达 17%。

**[Improve Rule Retrieval And Reasoning With Self-Induction And Relevance Reestimat](others/improve_rule_retrieval_and_reasoning_with_self-induction_and_relevance_reestimat.md)**

:   针对规则检索中查询（具体实例化事实）与规则（抽象变量形式）之间的语义鸿沟，提出 SIAR（自归纳增强检索）和 R3（规则相关性重评估）两种方法，通过将查询映射到规则语义空间并重新评估规则相关性，显著提升规则检索和下游推理性能。

**[Improving Language And Modality Transfer In](others/improving_language_and_modality_transfer_in.md)**

:   提出基于字符级编码器 charSONAR 的跨语言跨模态翻译方法，通过 teacher-student 训练获得字符级文本编码器，再用轻量适配器连接 1000+ 语言的 CTC ASR 模型（MMS），在 75 语言文本翻译和 33 语言语音翻译上实现 SOTA，零资源低资源场景表现尤其突出。

**[Inducing Lexicons Of In-Group Language With Socio-Temporal Context](others/inducing_lexicons_of_in-group_language_with_socio-temporal_context.md)**

:   提出 LISTN（Lexicon Induction with Socio-Temporal Nuance）框架，利用动态词嵌入和用户嵌入联合建模社区语言的社会结构和时间演化，在反女性在线社区（manosphere）的群体内词汇归纳任务上达到 0.77 的平均精度，显著超越现有方法。

**[Inferring Functionality Of Attention Heads From Their Parameters](others/inferring_functionality_of_attention_heads_from_their_parameters.md)**

:   提出MAPS框架，通过将注意力头参数投影到词汇空间构建token映射矩阵$M$，无需任何推理或训练即可推断注意力头实现的功能，在6个LLM上验证了20种关系操作的映射准确性，并开发自动化pipeline发现了大量此前未被识别的注意力头功能。

**[Infogen Generating Complex Statistical Infographics From Documents](others/infogen_generating_complex_statistical_infographics_from_documents.md)**

:   提出Infogen框架，将文本文档转化为复杂统计信息图（多子图组合），采用两阶段设计——先用微调LLM生成结构化中间元数据，再用LLM代码生成器和反馈模块迭代生成最终信息图代码。

**[Inner Thinking Transformer Leveraging Dynamic Depth Scaling To Foster Adaptive I](others/inner_thinking_transformer_leveraging_dynamic_depth_scaling_to_foster_adaptive_i.md)**

:   提出 Inner Thinking Transformer (ITT)，通过自适应 token 路由和残差思维连接，在不增加参数的情况下为关键 token 动态分配更多计算步骤，实现隐式深度推理，162M 参数即可达到 466M Transformer 96.5% 的性能。

**[Inspiredebate Multidim Evaluation Debating](others/inspiredebate_multidim_evaluation_debating.md)**

:   提出双组件框架：InspireScore（融合4个主观维度+2个客观维度的辩论评估系统）和 InspireDebate（通过CoT-SFT + 多维DPO + Web-RAG 三阶段优化的辩论框架），评估系统与专家判断相关性提高 44%，辩论性能超越基线 57%。

**[Instruction-Tuning Data Synthesis From Scratch Via Web Reconstruction](others/instruction-tuning_data_synthesis_from_scratch_via_web_reconstruction.md)**

:   提出 Web Reconstruction (WebR)，一种从原始网页文档全自动合成高质量指令微调数据的框架，通过"Web作为指令"和"Web作为回复"双视角范式，无需人工标注即可生成优于现有SOTA的IT数据。

**[Intuitive Fine Tuning](others/intuitive_fine_tuning.md)**

:   通过MDP框架将SFT和偏好优化统一建模为"偏好估计+转移优化"两个子过程，揭示SFT本质上是偏好优化的特殊退化形式（使用偏差先验），提出IFT方法通过时间残差连接在仅使用SFT格式数据的条件下实现接近或超越SFT+PO顺序训练的对齐效果。

**[Iris Interactive Research Ideation System For Accelerating Scientific Discovery](others/iris_interactive_research_ideation_system_for_accelerating_scientific_discovery.md)**

:   提出 IRIS，一个开源的交互式研究构思系统，通过蒙特卡洛树搜索（MCTS）扩展测试时计算、细粒度反馈机制和基于查询的文献综合，实现人机协作的科学假设生成。

**[Is Linguistically-Motivated Data Augmentation Worth It](others/is_linguistically-motivated_data_augmentation_worth_it.md)**

:   系统比较语言学驱动和非语言学（随机扰动）数据增强策略在两种低资源语言上的效果，发现语言学方法仅在生成的样本接近训练数据分布时才有优势，否则可能有害。

**[Its Not A Walk In The Park Challenges Of Idiom Translation In Speech-To-Text Sys](others/its_not_a_walk_in_the_park_challenges_of_idiom_translation_in_speech-to-text_sys.md)**

:   本文首次系统比较了语音到文本翻译（SLT）、文本机器翻译（MT）和大语言模型（LLM）在习语翻译任务上的表现，发现 SLT 系统在处理习语时性能大幅下降，即便在编码器高层仍倾向于字面翻译，而 MT 和 LLM 对习语的处理能力明显更优。

**[Knowledge Tracing In Programming Education Integrating Students Questions](others/knowledge_tracing_in_programming_education_integrating_students_questions.md)**

:   本文提出 SQKT（Students' Question-based Knowledge Tracing）模型，首次将学生提问和自动提取的技能信息整合到知识追踪中，用于预测编程教育中学生对后续编程题的完成情况，域内实验 AUC 提升高达 33.1%。

**[Kodcode A Diverse Challenging And Verifiable Synthetic Dataset For Coding](others/kodcode_a_diverse_challenging_and_verifiable_synthetic_dataset_for_coding.md)**

:   KodCode 提出一套三阶段合成数据管线（编程题目合成→解决方案+单元测试自验证→后训练数据合成），构建了 447K 经过验证的编程 question-solution-test 三元组，微调后的模型在 HumanEval、MBPP、BigCodeBench、LiveCodeBench 等基准上超越 Qwen2.5-Coder-32B-Instruct 和 DeepSeek-R1-Distill-Llama-70B。

**[Ladder Language-Driven Slice Discovery And Error Rectification In Vision Classif](others/ladder_language-driven_slice_discovery_and_error_rectification_in_vision_classif.md)**

:   提出 LADDER 框架，利用 LLM 的推理能力和潜在领域知识，通过分析文本（图像描述/医学报告/元数据）自动发现视觉分类器中的系统性偏差切片（error slices），并通过伪标签生成和属性重平衡实现无需标注的多偏差缓解。

**[Laquer Localized Attribution](others/laquer_localized_attribution.md)**

:   提出 Localized Attribution Queries (LAQuer) 任务——将生成文本中用户选定的片段精确定位到源文档的对应片段，实现比句子级归因更精细、比子句级归因更用户导向的溯源，在多文档摘要和长文本问答上显著减少了归因文本长度。

**[Latim Measuring Latent Token-To-Token Interactions In Mamba Models](others/latim_measuring_latent_token-to-token_interactions_in_mamba_models.md)**

:   提出 LaTIM，一种针对 Mamba-1 和 Mamba-2 的 token 级分解方法，将 SSM 的隐式计算重构为类似 Transformer 注意力的 token-to-token 贡献矩阵，实现对 Mamba 模型的细粒度可解释性分析。

**[Learning To Align Multi-Faceted Evaluation A Unified And Robust Framework](others/learning_to_align_multi-faceted_evaluation_a_unified_and_robust_framework.md)**

:   提出 ARJudge 评估框架，通过微调 Analyzer 自适应生成评估标准并执行文本+代码双驱动分析，配合无需微调的 Refiner 综合判断，在多个评估基准上超越现有微调评估器，尤其在指令遵循评估上通过代码驱动分析提升高达 11.1%。

**[Length-Induced Embedding Collapse In Plm-Based Models](others/length-induced_embedding_collapse_in_plm-based_models.md)**

:   发现并严格证明了 PLM 文本嵌入模型中的"长度坍缩"现象——长文本嵌入趋于聚集，源于 self-attention 作为低通滤波器随文本长度增加而滤波率增强，高频信息被过度抑制；提出 TempScale 方法通过降低 attention 温度来缓解长短文本嵌入分布差异，在 MTEB 上提升 0.94%、LongEmbed 上提升 1.10%。

**[Limited Generalizability In Argument Mining State-Of-The-Art Models Learn Datase](others/limited_generalizability_in_argument_mining_state-of-the-art_models_learn_datase.md)**

:   对 4 种 Transformer 模型在 17 个英语句子级论辩挖掘数据集上进行首次大规模跨数据集泛化评估，发现 SOTA 模型主要学到了数据集特有的词汇模式而非论辩的结构性信号，泛化能力远低于基准表现，但任务相关预训练和联合数据训练可部分缓解这一问题。

**[Literature Meets Data Hypothesis](others/literature_meets_data_hypothesis.md)**

:   提出首个将文献驱动和数据驱动假设生成进行协同整合的方法，通过 Refinement 和 Union 两种策略让 LLM 从论文摘要和观测数据中联合生成更具泛化性的假设，在五个社会科学分类任务的 OOD 数据集上比纯数据驱动方法平均提升 3.37%，并首次通过人类实验证明 LLM 生成的假设能显著改善人类决策准确率（+7.44% / +14.19%）。

**[Logu Longform Gen Uncertainty](others/logu_longform_gen_uncertainty.md)**

:   定义"长文本不确定性生成"（LoGU）任务，识别不确定性抑制和不确定性错位两个子挑战，提出基于分解的数据构造框架和 SFT+DPO 两阶段训练流水线，使 LLM 在长文本生成中对不确定事实显式表达不确定性，在三个数据集上将 Llama3-8B 的事实准确率从 51.9% 提升到 71.6%，错误声明数从 20.4 降到 5.81。

**[Low-Rank Interconnected Adaptation Across Layers](others/low-rank_interconnected_adaptation_across_layers.md)**

:   提出 Lily（Low-rank Interconnected Adaptation across Layers），通过将 LoRA 的 A/B 适配器跨层解耦并互联共享，配合数据依赖的路由机制，在相同或更少参数下实现高秩权重更新，在多模态、多架构、多规模场景中均优于 LoRA。

**[Macp Minimal Yet Mighty Adaptation Via Hierarchical Cosine Projection](others/macp_minimal_yet_mighty_adaptation_via_hierarchical_cosine_projection.md)**

:   本文提出 MaCP——一种基于离散余弦变换（DCT）的参数高效微调方法，通过将权重变化投影到余弦频域并分层选择最关键的频率分量，在极低参数量（比 LoRA 少 99.7%）下实现了优于或媲美现有 PEFT 方法的性能。

**[Making Fetch Happen Finding Emergent Dog Whistles Through Common Habitats](others/making_fetch_happen_finding_emergent_dog_whistles_through_common_habitats.md)**

:   提出 FETCH! 基准和 EarShot 系统，用于在大规模社交媒体语料库中发现新兴的"狗哨"（dog whistle，即具有双重含义的编码表达），利用向量数据库和 LLM 的结合实现了比现有方法高 2-20 个 F-score 百分点的提升。

**[Mapping The Podcast Ecosystem With The Structured Podcast Research Corpus](others/mapping_the_podcast_ecosystem_with_the_structured_podcast_research_corpus.md)**

:   构建并发布了 SPoRC——一个包含 110 万集播客转录的大规模数据集（含元数据、推断的说话者角色和 37 万集的音频特征），并通过话题分析、嘉宾共现网络分析和 George Floyd 事件响应性分析，首次全面刻画了播客生态系统的内容、结构和响应性。

**[Mapqator An Extensible Framework For Efficient Annotation Of Map-Based Qa Datase](others/mapqator_an_extensible_framework_for_efficient_annotation_of_map-based_qa_datase.md)**

:   提出 MapQaTor——一个可扩展的开源 Web 框架，通过集成多种地图 API（Google Maps、OpenStreetMap 等），将地理空间 QA 数据集的标注速度提升至少 30 倍，同时通过 API 响应缓存确保数据可复现性。

**[Mdcure A Scalable Pipeline For Multi-Document Instruction-Following](others/mdcure_a_scalable_pipeline_for_multi-document_instruction-following.md)**

:   提出 MDCure 框架，通过两阶段流程（生成+过滤）自动构建高质量的多文档指令数据，并训练 MDCureRM 多目标奖励模型进行数据过滤，使微调后的 LLM（最高 70B）在多文档和长上下文任务上相比基线提升高达 75.1%，且实现跨任务、跨领域的强泛化能力。

**[Meaning Beyond Truth Conditions Evaluating Discourse Level Understanding Via Ana](others/meaning_beyond_truth_conditions_evaluating_discourse_level_understanding_via_ana.md)**

:   本文提出语义理解能力的层级框架（词汇/句子/话语），构建了基于照应可及性（anaphora accessibility）的评估数据集，发现 LLM 在某些结构上与人类一致但在其他结构上存在系统性分歧——LLM 依赖词汇线索而非结构化抽象。

**[Measuring The Effect Of Transcription Noise On Downstream Language Understanding](others/measuring_the_effect_of_transcription_noise_on_downstream_language_understanding.md)**

:   提出ENDow框架，首次系统化地分析ASR转录噪声对下游NLU任务的影响，通过可配置的pipeline评估不同噪声强度和类型下任务模型的行为，发现命名实体是最关键的词类型，且模型能容忍一定程度的噪声。

**[Memorization A Close Look At Books](others/memorization_a_close_look_at_books.md)**

:   系统研究 Llama 3 系列模型对完整书籍的记忆化程度，发现书籍提取率与其流行度（训练数据重复度代理）高度正相关，并通过 LoRA 微调揭示指令微调的抗反刍缓解措施仅涉及极少量集中在底层 transformer block 的权重变化。

**[Meta-Learning Neural Mechanisms Rather Than Bayesian Priors](others/meta-learning_neural_mechanisms_rather_than_bayesian_priors.md)**

:   挑战了"元学习在神经网络中蒸馏贝叶斯简单性先验"的主流观点，通过形式语言实验证明元学习实际上是在模型中植入有用的**神经机制**（如计数器），而非学习简单性偏好。

**[Mexma Token-Level Objectives Improve Sentence Representations](others/mexma_token-level_objectives_improve_sentence_representations.md)**

:   提出 MEXMA，一种结合句子级和 token 级目标的跨语言句子编码器训练方法：用一种语言的句子表示去预测另一种语言的被掩码 token，同时让句子和 token 的梯度都直接更新编码器，在双文本挖掘和多项下游任务上超越 SONAR 和 LaBSE。

**[Micro Act Knowledge Conflict Reasoning](others/micro_act_knowledge_conflict_reasoning.md)**

:   提出 Micro-Act 框架，通过层次化动作空间（导航/功能/桥接动作）和自适应粒度分解，让 LLM 自动感知上下文复杂度并逐层拆解知识对比，在 5 个知识冲突基准上全面超越 SOTA，同时在无冲突场景下也保持鲁棒。

**[Mindref Mimicking Human Memory Hierarchical Reference Retrieval](others/mindref_mimicking_human_memory_hierarchical_reference_retrieval.md)**

:   提出 MindRef 框架，模拟人类先回忆文档标题再定位具体段落的两阶段记忆模式，通过 Trie 和 FM-Index 约束解码让 LLM 独立召回参考段落，无需额外检索模型或预分段。

**[Mitigating Confounding In Speech-Based Dementia Detection Through Weight Masking](others/mitigating_confounding_in_speech-based_dementia_detection_through_weight_masking.md)**

:   本文针对基于语音转录文本的痴呆检测任务中的**性别混淆偏差**问题，提出两种权重掩码方法（Extended Confounding Filter 和 Dual Filter），通过识别并消除 Transformer 网络中与性别关联的权重来实现去混淆，在保持痴呆检测性能的同时显著降低性别间的公平性差距。

**[Mitigating Shortcut Learning With Interpolated Learning](others/mitigating_shortcut_learning_with_interpolated_learning.md)**

:   提出 InterpoLL，通过将多数样本的表征与同类少数样本的表征做插值 $z_i = (1-\lambda)f_{enc}(x_i) + \lambda f_{enc}(x_j)$ 来弱化捷径特征影响，在 MNLI/FEVER/QQP 等 NLU 任务上少数样本泛化显著超越 ERM 和 SOTA 捷径缓解方法，且不损失多数样本准确率。

**[Mockconf A Student Interpretation Dataset Analysis Word- And Span-Level Alignmen](others/mockconf_a_student_interpretation_dataset_analysis_word-_and_span-level_alignmen.md)**

:   本文构建了 MockConf——一个以捷克语为中心的**学生同声传译数据集**（7 小时，5 种欧洲语言），提供人工标注的 span 级和 word 级对齐，同时发布了专用标注工具 InterAlign，并建立了自动对齐的基线和评估指标体系。

**[More A Mixture Of Low-Rank Experts For Adaptive Multi-Task Learning](others/more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)**

:   提出 MoRE (Mixture of Low-Rank Experts)，将 LoRA 中的不同秩视为不同专家，通过自适应秩选择器为每个任务动态选择最合适的秩，配合对比学习优化的任务嵌入和平衡数据采样策略，使用单个 LoRA 模块实现高效的多任务微调。

**[Mosaic Multiple Observers Spotting Ai Content](others/mosaic_multiple_observers_spotting_ai_content.md)**

:   基于信息论中的通用压缩原理，提出 MOSAIC——多 LLM 集成的 AI 生成文本检测方法，通过 Blahut-Arimoto 算法为多个 detector LLM 计算最优组合权重，构建混合分布作为观察者，比较文本的实际 surprisal 与混合模型的期望交叉熵差异来判断是否为 AI 生成，在多个域/语言/生成器上鲁棒优于单模型和双模型（如 Binoculars）方法。

**[Multi-Agent Collaboration Via Cross-Team Orchestration](others/multi-agent_collaboration_via_cross-team_orchestration.md)**

:   提出 Cross-Team Orchestration (Croto)，一个可扩展的多团队协作框架，通过将多个独立 agent 团队组织起来进行跨团队交互，利用层次化分组 (Hierarchy Partitioning) 和贪心聚合 (Greedy Aggregation) 机制将各团队的多样化解决方案融合为更优结果。

**[Multi-Facet Blending For Faceted Query-By-Example Retrieval](others/multi-facet_blending_for_faceted_query-by-example_retrieval.md)**

:   > 提出 FaBle（Multi-Facet Blending）数据增强方法，通过对文档进行面向分解（decomposition）、面向生成（generation）、面向重组（recomposition）三阶段，仅用 1K 文档合成出面向条件的训练三元组，在数据稀缺条件下显著提升分面 QBE 检索效果，特别是在最具挑战性的 method 分面上超越了使用 130 万+ 数据训练的强基线。

**[Multi-Hop Question Generation Via Dual-Perspective Keyword Guidance](others/multi-hop_question_generation_via_dual-perspective_keyword_guidance.md)**

:   定义了双视角关键词——问题关键词（捕捉提问者意图）和文档关键词（反映 QA 对相关内容），并提出 DPKG 框架，通过扩展 Transformer 编码器和两个答案感知解码器，将关键词无缝集成到多跳问题生成过程中。

**[My Life Is Miserable Have To Sign 500 Autographs Everyday Exposing Humblebraggin](others/my_life_is_miserable_have_to_sign_500_autographs_everyday_exposing_humblebraggin.md)**

:   首次将 humblebragging（谦虚式自夸）检测引入计算语言学领域，提出了一个4元组形式化定义，构建了 HB-24 合成数据集，并在 ML/DL/LLM 上进行了全面基准评估，GPT-4o 在 zero-shot+定义 设定下达到 0.88 F1，超越人类标注者。

**[Narrative Media Framing In Political Discourse](others/narrative_media_framing_in_political_discourse.md)**

:   将叙事学理论与媒体框架分析相结合，提出了包含角色（英雄/反派/受害者）、冲突/解决、文化故事三个结构化组件的叙事框架分析体系，在气候变化和 COVID-19 两个领域验证了该框架的有效性和可迁移性。

**[Neodiff Unified Text Diffusion](others/neodiff_unified_text_diffusion.md)**

:   提出 NeoDiff，通过引入"外在时间"（句子级扩散进度）和"内在时间"（token 级扩散进度）的双时间框架，利用 Poisson 过程为每个 token 独立分配细粒度噪声水平，并用上下文感知的时间预测器自适应调节去噪进度，统一了离散和连续文本扩散模型的理论框架，在机器翻译、复述、文本简化等多个任务上超越现有扩散基线。

**[Neural Parameter Search For Slimmer Fine-Tuned Models And Better Transfer](others/neural_parameter_search_for_slimmer_fine-tuned_models_and_better_transfer.md)**

:   提出Neural Parameter Search (NPS)，通过在task vector的低秩子空间中搜索最优权重系数来提升微调模型的剪枝效率，在知识迁移（+1.5%）、模型融合（+2.1%）和压缩（40%效率提升）三个场景下均取得显著改进。

**[Neuron Empirical Gradient Discovering And Quantifying Neurons Global Linear Cont](others/neuron_empirical_gradient_discovering_and_quantifying_neurons_global_linear_cont.md)**

:   揭示了预训练语言模型 FF 层神经元激活值与模型输出之间存在全局线性关系，提出了神经元经验梯度（NEG）来量化这种线性关系，并设计了高效估算方法 NeurGrad，最终通过技能神经元探测实验证明 NEG 能有效表征多种语言技能。

**[On Support Samples Of Next Word Prediction](others/on_support_samples_of_next_word_prediction.md)**

:   基于表示定理（representer theorem），研究语言模型下一词预测中训练样本的角色，发现两类支持样本（促进预测和抑制预测），并证明支持样本是样本的内在属性（训练前即可预测），而非支持样本对表示学习至关重要。

**[One For All Update Parameterized Knowledge Across Multiple Models With Once Edit](others/one_for_all_update_parameterized_knowledge_across_multiple_models_with_once_edit.md)**

:   提出 OnceEdit，通过编辑一个轻量级插件模型并利用异构模型集成技术将编辑后的知识迁移到多个 LLM，实现"一次编辑，多模型更新"，在 ZsRE 和 Counterfact 数据集上显著超越现有方法。

**[Optimizing Decomposition For Optimal Claim Verification](others/optimizing_decomposition_for_optimal_claim_verification.md)**

:   提出动态分解（Dynamic Decomposition）框架，通过强化学习从验证器反馈中学习分解策略，将声明（claim）分解为验证器偏好的原子性粒度，弥合分解器与验证器之间的性能差距。

**[Partial Colexifications Improve Concept Embeddings](others/partial_colexifications_improve_concept_embeddings.md)**

:   首次将部分共词化（partial colexifications）数据整合到概念嵌入训练中，通过图嵌入技术在跨语言共词化网络上学习概念向量表示，在语义相似度、语义变化预测和词汇联想预测三项任务上均显著优于仅使用完全共词化的基线。

**[Patclaimeval Patent Evaluation](others/patclaimeval_patent_evaluation.md)**

:   提出首个专利权利要求评估基准 Patent-CE（1228 个专家标注的比较评估数据点）和专用评估方法 PatClaimEval（基于 Longformer + 对比学习变体），在特征完整性、概念清晰度、术语一致性、逻辑连接和整体质量五个维度上与人类专家评估的相关性全面超越 13 种现有指标（包括 G-Eval-4），整体质量维度的 Spearman 提升 58%。

**[Persistent Homology Of Topic Networks For The Prediction Of Reader Curiosity](others/persistent_homology_of_topic_networks_for_the_prediction_of_reader_curiosity.md)**

:   > 将文本的主题网络结构用持续同调 (Persistent Homology) 量化为拓扑空洞（连通分量、环、空腔），以此作为"信息空白"的代理变量来预测读者好奇心，在《饥饿游戏》小说上实现了 73% 的解释偏差（vs 基线 30%）。

**[Persona Dynamics Unveiling The Impact Of Persona Traits On Agents In Text-Based ](others/persona_dynamics_unveiling_the_impact_of_persona_traits_on_agents_in_text-based_.md)**

:   提出 PANDA 方法，将人类人格特质（Big Five + Dark Triad 共8种）投射到文本游戏智能体的策略学习中，通过人格分类器引导 Q 值调整，发现高开放性（Openness）人格在冒险类文本游戏中表现显著优于其他人格类型。

**[Personabench Evaluating Ai Models On Understanding Personal Information Through ](others/personabench_evaluating_ai_models_on_understanding_personal_information_through_.md)**

:   提出一套合成数据生成管线，创建包含多样化用户画像和模拟私有文档（对话记录、AI 交互、购买历史）的 PersonaBench 基准，用于评估 AI 模型从嘈杂用户数据中提取个人信息的能力，实验表明当前 RAG 方案远不能胜任这一任务。

**[Personalized Generation In Large Model Era A Survey](others/personalized_generation_in_large_model_era_a_survey.md)**

:   首篇跨模态个性化生成（PGen）综合综述，提出统一的用户中心视角将 NLP/CV/IR 社区的研究纳入一个框架，系统化梳理了文本/图像/视频/音频/3D/跨模态六大模态下的个性化技术、数据集和评估指标，涵盖 200+ 篇文献，并指出可扩展性、偏好演化、隐私公平等关键挑战。

**[Plagiarism Ai Generated Research](others/plagiarism_ai_generated_research.md)**

:   在对自主科研 Agent（如 AI Scientist）生成的研究文档进行专家审查后发现，24% 的文档是"智能剽窃"——方法论与已有工作一一对应但不引用原始来源，且现有剽窃检测工具无法识别这种"改头换面"的抄袭。

**[Popalign Diversifying Contrasting Patterns For A More Comprehensive Alignment](others/popalign_diversifying_contrasting_patterns_for_a_more_comprehensive_alignment.md)**

:   提出PopAlign框架，从Prompt、Model、Pipeline三个层面构建六种多样化对比策略（包括创新的Elicitive Contrast），无需额外人工标注即可合成高质量偏好数据，实现更全面的LLM对齐。

**[Predicting Through Generation Why Generation Is Better For Prediction](others/predicting_through_generation_why_generation_is_better_for_prediction.md)**

:   本文从信息论角度证明了token级生成比pooled表示保留更多互信息，提出PredGen框架通过scheduled sampling和task adapter解决生成式预测中的exposure bias和格式不匹配问题，并设计了Writer-Director Alignment Loss统一生成与预测目标。

**[Preventing Rogue Agents Improves Multi-Agent Collaboration](others/preventing_rogue_agents_improves_multi-agent_collaboration.md)**

:   提出一种通过实时监控 Agent 不确定性来检测"失控 Agent"（rogue agent）并进行干预的框架，在自建的 WhoDunitEnv 多智能体协作环境以及代码生成和资源可持续性任务上分别取得高达 17.4%、2.5% 和 20% 的性能提升。

**[Principled Generalization Arithmetic](others/principled_generalization_arithmetic.md)**

:   建立首个统一理论框架来理解 Transformer 在算术任务（加法/乘法/模运算）上的泛化行为——从任务性质（平移不变性）和位置编码类型（APE/RPE）的交互出发，解释了之前困扰领域的多个泛化谜题（如加法能泛化但乘法不能，模100能泛化但模101不能），实验验证了理论预测。

**[Proactive Conversational Coaching](others/proactive_conversational_coaching.md)**

:   通过健康教练领域的专家访谈和用户研究（31 名参与者、155 段对话），系统评估了五种不同对话风格（Directive、Interrogative、Facilitative）的 LLM 教练 Agent，发现用户高度重视核心功能性（substance）而对缺乏功能性时的风格修饰（style）持负面态度，同时揭示了用户第一人称评价与专家/LLM 第三方评价之间的显著不一致。

**[Proxann Topic Model Eval](others/proxann_topic_model_eval.md)**

:   ProxAnn 设计了一个面向实际使用场景的主题模型/文档聚类评估协议——标注者先从模型输出推断类别再将类别应用到新文档——并证明 LLM 代理标注者可以统计不可分辨地替代人类标注者，同时发现经典 LDA 在此评估下表现不弱于现代方法。

**[Pvp An Image Dataset For Personalized](others/pvp_an_image_dataset_for_personalized.md)**

:   构建了首个大规模个性化视觉说服数据集 PVP（28,454 张图像、596 条消息、9 种说服策略、2,521 位标注者的心理特征），并提出说服图像生成和评估两个任务，发现融入观看者的心理特征能显著提升说服图像的生成和评估效果。

**[Qg-Sms Enhancing Test Item Analysis Via Student Modeling And Simulation](others/qg-sms_enhancing_test_item_analysis_via_student_modeling_and_simulation.md)**

:   QG-SMS 提出用单个 LLM 模拟不同理解水平的学生群体，通过学生画像生成、表现预测和分析三步流程，弥补了现有 LLM 评估器在考后分析维度（题目难度、区分度、干扰项效率）上的严重不足，在多个数据集上实现了最高一致性准确率。

**[Qualispeech A Speech Quality Assessment Dataset With Natural Language Reasoning ](others/qualispeech_a_speech_quality_assessment_dataset_with_natural_language_reasoning_.md)**

:   本文提出 QualiSpeech，首个包含 11 个维度标注和详细自然语言推理描述的语音质量评估数据集，以及配套的评测基准，证明了微调后的听觉 LLM 能生成关于噪声和失真的详细描述，并展示了推理增强质量评估的潜力。

**[Quantifying Lexical Semantic Shift Via Unbalanced Optimal Transport](others/quantifying_lexical_semantic_shift_via_unbalanced_optimal_transport.md)**

:   将 Unbalanced OT 应用于上下文化词嵌入来量化词义变迁，提出 Sense Usage Shift (SUS) 指标在每个用法实例级别量化语义变化（义项使用频率增减），统一解决实例级变化检测、词级变化幅度量化、词义扩展/缩小判定。

**[Rank Chunk And Expand Lineage-Oriented Reasoning For Taxonomy Expansion](others/rank_chunk_and_expand_lineage-oriented_reasoning_for_taxonomy_expansion.md)**

:   LORex 提出了一个即插即用的分类体系扩展框架，结合判别式排序器 TEMPORA（基于欧拉路径的分类路径语言化）和迭代式 LLM 推理（语义过滤→父节点检索→路径验证），无需微调 LLM，在 4 个基准上实现了 12% 的准确率提升和 5% 的 Wu&P 提升。

**[Reidentification Deidentified](others/reidentification_deidentified.md)**

:   提出一种基于 RAG 的再识别攻击方法，通过稀疏检索+稠密检索+自回归填充的三阶段流程，对去标识化文档进行逆向攻击，在三个数据集上高达 80% 的被遮蔽文本片段可被成功恢复，揭示了当前去标识化方法的脆弱性。

**[Reliable Eval Metrics Scientific](others/reliable_eval_metrics_scientific.md)**

:   系统分析了传统相似度指标（ROUGE、BERTScore 等）在科学文本修订评估中的局限性，发现它们强相关于编辑距离且惩罚深度修改，并证明结合 LLM-as-Judge 和任务特定指标的混合方法最能对齐人类评判。

**[Rep Robust Knowledge Editing](others/rep_robust_knowledge_editing.md)**

:   揭示locate-and-edit知识编辑方法中语义键的根本缺陷——内部表示无法同时满足鲁棒性和特异性，提出REP模块通过对比学习解耦编辑键，在鲁棒性测试上提升最高66.4%。

**[Repanda Pandas-Powered Tabular Verification And Reasoning](others/repanda_pandas-powered_tabular_verification_and_reasoning.md)**

:   提出 RePanda，将自然语言声明翻译为可执行的 pandas 查询来实现表格事实验证，在 TabFact 上达到 84.09% 准确率，在 OOD 的 WikiFact 上无需额外微调达 84.72%，同时以仅 7B 参数的模型逼近 671B DeepSeek-Chat 的零样本性能，并扩展至表格问答任务取得 75.1% 准确率。

**[Research Borderlands Analysing Writing Across Research Cultures](others/research_borderlands_analysing_writing_across_research_cultures.md)**

:   通过访谈跨学科研究者构建学术写作文化规范框架（结构/风格/修辞/引用四类），并用计算指标量化11个CS社区的写作差异，揭示LLM在跨社区写作改编时存在严重的"同质化"倾向。

**[Retrospective Learning From Interactions](others/retrospective_learning_from_interactions.md)**

:   提出 ReSpect 方法，让多模态 LLM 通过回顾性地解码用户在多轮交互中的隐式反馈信号来自我改进，无需任何外部标注，在数千次人机交互中将任务完成率从 31% 提升至 82%。

**[Revisiting Weak-To-Strong Generalization In Theory And Practice Reverse Kl Vs Fo](others/revisiting_weak-to-strong_generalization_in_theory_and_practice_reverse_kl_vs_fo.md)**

:   在 Weak-to-Strong Generalization (W2SG) 框架中，提出用 reverse KL 替代 forward KL 作为损失函数——理论证明 reverse KL 的 mode-seeking 特性可保证强模型超过弱监督者至少"分歧量"的幅度，实验在 GPT-2/Pythia/Qwen2.5 系列上验证 reverse KL/CE 在 12/12 设置中超越 forward KL 且噪声鲁棒性更好。

**[Rmoa Optimizing Mixture-Of-Agents Through Diversity Maximization And Residual Co](others/rmoa_optimizing_mixture-of-agents_through_diversity_maximization_and_residual_co.md)**

:   受 ResNet 残差学习启发，提出 RMoA 框架，通过嵌入式多样性贪心选择、残差提取/聚合智能体和自适应终止机制来优化多智能体协作架构，在降低计算开销的同时实现 SOTA 性能。

**[Rotor Towards More Reliable Responses For Order-Invariant Inputs](others/rotor_towards_more_reliable_responses_for_order-invariant_inputs.md)**

:   提出 RoToR，一种基于全局排序和循环位置编码分配的零样本顺序不变语言模型，通过最小化位置 ID 修改来实现稳定的顺序不变性，并设计选择路由（Selective Routing）机制自适应处理混合输入类型。

**[Rubriks Cube Testing A New Rubric For Evaluating Explanations On The Cube Datase](others/rubriks_cube_testing_a_new_rubric_for_evaluating_explanations_on_the_cube_datase.md)**

:   提出 Rubrik——受教育评估启发的解释质量评价量规，基于三层嵌套类型体系（Commentary⊆Justification⊆Argument）+ 8 维质量维度，配套 CUBE 数据集（26K 条由人类和 6 个 LLM 生成的解释），发现 LLM 解释低质主因是缺乏简洁性而非连贯性。

**[S2Wtm Spherical Sliced-Wasserstein Autoencoder For Topic Modeling](others/s2wtm_spherical_sliced-wasserstein_autoencoder_for_topic_modeling.md)**

:   提出 S2WTM，一种基于球面切片 Wasserstein 自编码器的主题模型，在超球面潜空间上对齐聚合后验与先验分布，有效避免 VAE 的后验坍塌问题，同时在主题连贯性和多样性上超越现有 SOTA。

**[S3 - Semantic Signal Separation](others/s3_-_semantic_signal_separation.md)**

:   S3将主题建模概念化为发现语义空间中独立语义轴的过程，利用独立成分分析（ICA）分解文档嵌入矩阵，无需预处理即可产生高度连贯且多样化的主题，同时是最快的上下文主题模型（平均比BERTopic快4.5倍）。

**[Sdd Self-Degraded Defense Against Malicious Fine-Tuning](others/sdd_self-degraded_defense_against_malicious_fine-tuning.md)**

:   SDD通过训练LLM对有害指令生成高质量但无关的良性回复来实现防御：当攻击者进行恶意微调时，模型的通用能力会显著下降，从而无法有效执行恶意指令。

**[Self-Correction Is More Than Refinement A Learning Framework For Visual And Lang](others/self-correction_is_more_than_refinement_a_learning_framework_for_visual_and_lang.md)**

:   提出 Self-Correction Learning (SCL)，通过将 VLM 自身产生的自纠正数据（成功和失败的纠正样本）分类为偏好/非偏好对，利用 DPO 进行偏好微调，从根本上提升模型直接生成正确答案的能力，而非仅仅依赖推理时的迭代修正。

**[Self-Foveate Enhancing Diversity And Difficulty Of Synthesized Instructions From](others/self-foveate_enhancing_diversity_and_difficulty_of_synthesized_instructions_from.md)**

:   提出 Self-Foveate 方法，受人类视觉注视机制启发，通过"微观-散射-宏观"三级注视策略，从无监督文本中系统性提取多粒度信息，合成具有更高多样性和难度的指令数据，用于 LLM 的指令微调。

**[Seoe Semantic Eval](others/seoe_semantic_eval.md)**

:   针对开放域事件检测（ODED）评估的两大痛点——有限 benchmark 缺乏真实世界代表性、token 级匹配指标无法捕捉语义相似性——提出 SEOE 框架，构建包含 564 种事件类型覆盖 7 大领域的可扩展 benchmark，并引入基于 LLM 的语义 F1 评估指标。

**[Share An Slm-Based Hierarchical Action Correction Assistant For Text-To-Sql](others/share_an_slm-based_hierarchical_action_correction_assistant_for_text-to-sql.md)**

:   提出 SHARE 框架，通过三个专门化小语言模型 (SLM, <8B) 的顺序管道协作，将声明式 SQL 转换为步骤化动作轨迹以暴露推理路径，再从 Schema 和逻辑两个维度分阶段纠正错误，实现高效低成本的 Text-to-SQL 自纠错。

**[Share Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed Fro](others/share_shared_memory-aware_open-domain_long-term_dialogue_dataset_constructed_fro.md)**

:   提出了基于电影剧本构建的长期对话数据集 SHARE，首次引入「共享记忆」概念，并设计了 EPISODE 对话框架来管理个人信息、个人事件和共享记忆，使长期对话更具亲密感和参与度。

**[Share Text To Sql Correction](others/share_text_to_sql_correction.md)**

:   提出 SHARE 框架，通过三个专用小语言模型（SLM）组成的顺序管道，将 SQL 查询转换为逐步动作轨迹并分别修正 schema 错误和逻辑错误，从而以低成本高效辅助 LLM 进行 Text-to-SQL 自纠正。

**[Sightation Counts Leveraging Sighted User Feedback In Building A Blv-Aligned Dat](others/sightation_counts_leveraging_sighted_user_feedback_in_building_a_blv-aligned_dat.md)**

:   构建了首个经盲人/低视力（BLV）专业教育者验证的大规模图表描述数据集 Sightation，通过 VLM 引导生成 + 视力正常者评估 + BLV 偏好对齐的流程，在多项下游任务上显著提升了图表描述的可访问性。

**[Sleepless Nights Sugary Days Creating Synthetic Users With Health Conditions For](others/sleepless_nights_sugary_days_creating_synthetic_users_with_health_conditions_for.md)**

:   提出一个端到端框架，基于真实人口学、健康/生活方式和行为/心理特征数据生成有健康状况的合成用户（涵盖睡眠和糖尿病管理），用于评估健康教练Agent的交互质量，并通过人类专家评估验证其显著优于通用合成用户。

**[Spot Bridging Natural Language And Geospatial Search For Investigative Journalis](others/spot_bridging_natural_language_and_geospatial_search_for_investigative_journalis.md)**

:   提出 SPOT——一个面向调查记者的开源自然语言地理空间搜索工具，通过微调 LLaMA 3 将非结构化场景描述转换为 YAML 查询，结合 OSM 标签语义捆绑系统和 Elasticsearch 索引，实现对 OpenStreetMap 的准确自然语言访问。

**[Spotting Out-Of-Character Behavior Atomic-Level Evaluation Of Persona Fidelity I](others/spotting_out-of-character_behavior_atomic-level_evaluation_of_persona_fidelity_i.md)**

:   提出原子级（句子级）评估框架，通过三个指标（ACC_atom、IC_atom、RC_atom）细粒度检测大语言模型在开放式文本生成中的角色偏离（Out-of-Character）行为，弥补了传统整体评分方法无法捕捉长文本中微妙人格不一致的问题。

**[Statistical Deficiency Task Inclusion](others/statistical_deficiency_task_inclusion.md)**

:   基于统计缺陷性（statistical deficiency）理论，提出一种理论驱动的任务包含关系（task inclusion）定义与度量框架，以信息充分性（information sufficiency, IS）作为可计算代理指标，通过比较微调模型的中间层表征来估计任务间的包含程度，并在合成数据和真实NLP任务上成功重建了经典NLP pipeline的层次关系。

**[Stricta Structured Reasoning In Critical Text Assessment For Peer Review And Bey](others/stricta_structured_reasoning_in_critical_text_assessment_for_peer_review_and_bey.md)**

:   提出 STRICTA 框架，基于结构因果模型（SCM）将文本评审建模为显式的逐步推理图（workflow），在生物医学论文评审中收集 40+ 位专家的 4000+ 推理步骤数据集，发现先验知识差异是专家分歧主因、写作风格对最终评审有因果影响，LLM 存在错误传播但人类监督可有效缓解。

**[Subword Models Struggle With Word Learning But Surprisal Hides It](others/subword_models_struggle_with_word_learning_but_surprisal_hides_it.md)**

:   本文通过心理语言学的词汇判断任务（lexical decision task）揭示了使用子词（BPE）分词的语言模型在单词学习方面存在严重缺陷，而基于字符级分词的模型能轻松完成该任务；当使用 surprisal（在语境中的出乎意料程度）来评估时，这一差距被掩盖了。

**[Sudolm Authorization Alignment](others/sudolm_authorization_alignment.md)**

:   SudoLM 提出了一种 LLM 参数化知识访问控制框架，通过"SUDO key"机制让授权用户解锁受限知识（如医学领域知识），未授权用户则只能访问公开知识，用 DPO 的 authorization alignment 在一个模型内实现了传统需要多版本模型才能完成的分级访问控制。

**[Synergistic Weak-Strong Collaboration By Aligning Preferences](others/synergistic_weak-strong_collaboration_by_aligning_preferences.md)**

:   本文提出 CoWest 框架，通过让专业化的弱模型（如 LLaMA3-8B）生成初始草稿，再由通用强模型（如 GPT-4）精炼，并利用协作反馈通过 DPO 微调弱模型以对齐强模型偏好，在反事实推理、医学和伦理三个领域显著超越单模型和已有协作方法。

**[Synthia Novel Concept Design With Affordance Composition](others/synthia_novel_concept_design_with_affordance_composition.md)**

:   Synthia 提出了一种基于 affordance（功能可供性）组合的新颖概念设计框架，通过层次化概念本体、affordance 采样策略和课程学习微调 T2I 模型，生成既视觉新颖又功能连贯的创新设计。

**[Tabxeval Why This Is A Bad Table An Exhaustive Rubric For Table Evaluation](others/tabxeval_why_this_is_a_bad_table_an_exhaustive_rubric_for_table_evaluation.md)**

:   TabXEval 提出了一种基于结构化评分规则（rubric）的两阶段表格评估框架——先通过 TabAlign 对齐参考表和生成表的结构，再通过 TabCompare 进行语义和语法层面的细粒度比较，同时构建了多领域基准 TabXBench。

**[Taclr A Scalable And Efficient Retrieval-Based Method For Industrial Product Att](others/taclr_a_scalable_and_efficient_retrieval-based_method_for_industrial_product_att.md)**

:   TACLR 提出了首个基于检索范式的产品属性值识别（PAVI）方法，通过分类感知对比学习和自适应推理机制，在处理隐含值、OOD 值和归一化输出方面全面超越分类和生成方法，并已成功部署在闲鱼（Xianyu）平台。

**[Tag-Evol Achieving Efficient Instruction Evolving Via Tag Injection](others/tag-evol_achieving_efficient_instruction_evolving_via_tag_injection.md)**

:   Tag-Evol 提出了一种基于知识标签注入的指令进化框架，通过构建多步细粒度标签池和预算控制注入机制，无需迭代即可生成不同难度的高质量进化指令数据，在多任务多骨干上显著优于 Evol-Instruct。

**[Task-Informed Anti-Curriculum By Masking Improves Downstream Performance On Text](others/task-informed_anti-curriculum_by_masking_improves_downstream_performance_on_text.md)**

:   TIACBM 提出了一种任务感知的反课程掩码微调策略：利用下游任务知识（如情感极性、词性标签）决定哪些 token 被掩码，并采用周期衰减的掩码率，在情感分析、文本分类和作者归属三个任务上均取得统计显著的性能提升。

**[Testnuc Enhancing Test-Time Computing Approaches And Scaling Through Neighboring](others/testnuc_enhancing_test-time_computing_approaches_and_scaling_through_neighboring.md)**

:   TestNUC 提出了一种线性扩展的测试时推理增强方法，通过检索测试样本的近邻无标注数据，让 LLM 同时预测测试样本及其邻居，再通过加权多数投票聚合，稳定提升分类准确率。

**[The Harmonic Structure Of Information Contours](others/the_harmonic_structure_of_information_contours.md)**

:   提出 Harmonic Surprisal (HS) 假说——文本中 surprisal 曲线呈周期性波动且周期与语篇结构（EDU/句子/段落）对齐，用带时间缩放的谐波回归检验，在 6 种语言上发现一致的周期模式，精化了经典的 Uniform Information Density 假说。

**[The Noisy Path From Source To Citation Measuring How Scholars Engage With Past R](others/the_noisy_path_from_source_to_citation_measuring_how_scholars_engage_with_past_r.md)**

:   本文构建了一个大规模计算管道来量化引用保真度（citation fidelity），基于 1300 万引用句对分析发现：引用保真度与时间/学科接近度、开放获取和团队规模等因素显著相关，并通过准实验证实了中介引用导致信息失真的"电话效应"。

**[Tiser Timeline Self Reflection Temporal](others/tiser_timeline_self_reflection_temporal.md)**

:   提出 TISER 框架，通过"推理→时间线构建→自反思→答案生成"四阶段管道实现LLM时间推理的test-time scaling，配合合成推理轨迹数据微调，让 7B 开源模型在多个时间推理基准上超越 GPT-4，在TGQA等任务上达到 SOTA。

**[Tokenisation Is Np-Complete](others/tokenisation_is_np-complete.md)**

:   本文证明了以压缩为目标的分词问题（包括直接分词和自底向上分词两种变体）是NP完全的，这意味着不太可能存在高效算法找到最优分词器，从而为BPE等近似算法的使用提供了理论依据。

**[Towards Comprehensive Argument Analysis In Education Dataset Tasks And Method](others/towards_comprehensive_argument_analysis_in_education_dataset_tasks_and_method.md)**

:   本文针对中文高中议论文，提出包含纵向（论证关系）和横向（话语关系）两个维度共 14 种细粒度论证关系类型的标注方案，并在论证成分检测、关系预测和自动评分三个任务上建立了全面的 benchmark。

**[Towards Style Alignment In Cross-Cultural Translation](others/towards_style_alignment_in_cross-cultural_translation.md)**

:   本文提出跨文化翻译中的"风格对齐"概念，发现当前 LLM 翻译会破坏文本的风格信息（尤其偏向中性化），并提出 RASTA 方法通过学习嵌入空间中的风格概念来检索风格对齐的少样本示例，在不损害翻译质量的前提下显著提升风格保持。

**[Towards Text-Image Interleaved Retrieval](others/towards_text-image_interleaved_retrieval.md)**

:   定义文本-图像交错检索（TIIR）新任务，构建基于 wikiHow 的首个 TIIR 基准数据集（155K 文档、7654 测试对），并提出 Matryoshka Multimodal Embedder（MME）通过多粒度视觉 token 压缩解决 MLLM 中视觉 token 过多导致的效率和语义偏差问题，大幅提升检索性能。

**[Tree-Of-Debate Multi-Persona Debate Trees Elicit Critical Thinking For Scientifi](others/tree-of-debate_multi-persona_debate_trees_elicit_critical_thinking_for_scientifi.md)**

:   提出Tree-of-Debate (ToD)框架，将科学论文转化为LLM persona进行树结构化辩论，通过自我审议、迭代检索和主持人引导的层级子话题扩展，生成细粒度、上下文化的论文对比摘要，在领域专家评估中显著优于基线方法。

**[Trove A Challenge For Finegrained Text](others/trove_a_challenge_for_finegrained_text.md)**

:   提出TROVE文本溯源挑战，将目标文本中每个句子追溯到源文档中的具体源句，并分类其细粒度关系（引用、压缩、推理等），覆盖多文档和长文档场景。

**[Tuna Temporal Understanding](others/tuna_temporal_understanding.md)**

:   本文提出 Tuna，一个面向密集动态视频的细粒度时序理解基准，包含 1000 个精心标注的视频和两个互补任务（Tuna-cap 字幕生成和 Tuna-mcq 视频问答），覆盖相机状态、场景、动作、属性四大动态要素，评估 21 个模型后发现即使 GPT-4o 也仅达 58.5% F1，揭示了当前模型在时序理解上的重大不足。

**[Understanding Common Ground Misalignment In Goal-Oriented Dialog A Case-Study Wi](others/understanding_common_ground_misalignment_in_goal-oriented_dialog_a_case-study_wi.md)**

:   本文通过在 Ubuntu IRC 技术支持对话中标注"对话摩擦"（conversational friction），实证揭示了共识基础（common ground）的失配与任务成功率之间的显著关联，并发现 LLM 能识别显式的对话摩擦但难以处理需要语用或领域推理的隐式摩擦。

**[Understanding Cross-Domain Adaptation In Low-Resource Topic Modeling](others/understanding_cross-domain_adaptation_in_low-resource_topic_modeling.md)**

:   首次将领域自适应形式化引入低资源主题建模，推导有限样本泛化上界指导方法设计，提出 DALTA 框架通过共享编码器、领域专用解码器和对抗对齐实现跨领域主题知识的选择性迁移。

**[Unifying Language Agent Algorithms With Graph-Based Orchestration Engine For Rep](others/unifying_language_agent_algorithms_with_graph-based_orchestration_engine_for_rep.md)**

:   提出 AGORA 框架，通过图编排引擎统一实现 10 种主流 Agent 推理算法，并建立标准化评测体系，系统比较不同算法在多种 LLM 上的表现，发现简单的 CoT 方法在成本效益上往往优于复杂算法。

**[Unique Hard Attention A Tale Of Two Sides](others/unique_hard_attention_a_tale_of_two_sides.md)**

:   本文揭示了Transformer中唯一硬注意力（UHA）的平局打破方向（最左vs最右）对模型表达能力有根本性影响：最左UHA等价于LTL[◇⁻]片段且与软注意力等价，而最右UHA等价于完整LTL，严格更强。

**[Unlocking Speech Instruction Data Potential With Query Rewriting](others/unlocking_speech_instruction_data_potential_with_query_rewriting.md)**

:   提出基于多 LLM 知识融合的查询重写框架，通过零样本重写将文本指令转换为更适合 TTS 模型合成的分布，结合多 Agent 标注与验证机制，将语音指令数据可用率从 72% 提升至 93%。

**[Unveiling Dual Quality In Product Reviews An Nlp-Based Approach](others/unveiling_dual_quality_in_product_reviews_an_nlp-based_approach.md)**

:   首次将"产品双重质量"（同一品牌产品在不同市场存在质量差异）问题定义为 NLP 任务，构建了 1957 条波兰语评论的标注数据集，并系统评估了 SetFit、Transformer 编码器和 LLM 等多种方法在该任务上的表现。

**[Usdc A Dataset Of Underlineuser Underlinestance And Underlinedogmatism In Long U](others/usdc_a_dataset_of_underlineuser_underlinestance_and_underlinedogmatism_in_long_u.md)**

:   构建 USDC——首个用户级长对话立场和教条主义数据集，764 个多用户 Reddit 对话（22 子版块），用 {Mistral Large, GPT-4} × {zero/one/few-shot} 共 6 设置多数投票标注立场(5级)+教条程度(4级)，并用 7 个 SLM 微调/指令微调建立基线。

**[Using Shapley Interactions To Understand How Models Use Structure](others/using_shapley_interactions_to_understand_how_models_use_structure.md)**

:   利用 Shapley Taylor 交互指数（STII）分析语言模型和语音模型如何在内部表征中编码输入的结构关系，发现自回归模型在句法邻近度、非组合语义和语音协同发音方面均展现出有意义的非线性交互模式。

**[Using Source-Side Confidence Estimation For Reliable Translation Into Unfamiliar](others/using_source-side_confidence_estimation_for_reliable_translation_into_unfamiliar.md)**

:   提出一种基于梯度归因的源端置信度估计方法，通过测量输出序列对源端嵌入的敏感度来识别可能误译的源端词汇，无需词对齐，在误译检测任务上显著优于传统对齐方法。

**[Value Residual Learning](others/value_residual_learning.md)**

:   提出 ResFormer 和 SVFormer，通过在注意力机制中引入第一层 Value 向量到后续层的残差连接，增强初始 token 级信息在深层网络中的传播，以比标准 Transformer 少 16.11% 的参数和 20.3% 的训练数据达到同等性能，SVFormer 还能减少近一半 KV 缓存。

**[Vaquum Are Vague Quantifiers Grounded In Visual Data](others/vaquum_are_vague_quantifiers_grounded_in_visual_data.md)**

:   构建 VAQUUM 数据集（20300 条人类评分），系统评估视觉语言模型在视觉场景中对模糊量词（few, many 等）理解和使用的能力，发现 VLM 的生成概率与人类判断存在一定对齐但在不同评估范式间表现不一致。

**[Verbosity-Aware Rationale Reduction Effective Reduction Of Redundant Rationale V](others/verbosity-aware_rationale_reduction_effective_reduction_of_redundant_rationale_v.md)**

:   提出 VARR 框架，以句子为单位并利用基于似然度的"冗余度（verbosity）"标准识别和移除推理路径中的冗余句子，在多种推理任务上平均提升 7.71% 准确率同时减少 19.87% 的 token 生成。

**[Visual Cues Enhance Predictive Turn-Taking For Two-Party Human Interaction](others/visual_cues_enhance_predictive_turn-taking_for_two-party_human_interaction.md)**

:   提出 MM-VAP 多模态话轮转换预测模型，结合语音与面部表情、头部姿态、注视方向等视觉线索，在视频会议对话中以 84% 平衡准确率显著优于纯语音 SOTA 模型的 79%，并首次按转换沉默时长分析性能。

**[Voting Or Consensus Decision-Making In Multi-Agent Debate](others/voting_or_consensus_decision-making_in_multi-agent_debate.md)**

:   系统评估了 7 种决策协议（4 种投票 + 3 种共识）对多 Agent LLM 辩论的影响，发现投票在推理任务上优于共识 13.2%，共识在知识任务上优于投票 2.8%，并提出 AAD 和 CI 两种提升答案多样性的新方法。

**[Well Begun Is Half Done Low-Resource Preference Alignment By Weak-To-Strong Deco](others/well_begun_is_half_done_low-resource_preference_alignment_by_weak-to-strong_deco.md)**

:   提出 Weak-to-Strong Decoding (WSD) 框架，利用一个小型对齐模型为大型基座模型起草对齐的开头，再由大模型续写，以低资源方式实现偏好对齐且不产生 alignment tax。

**[What Matters In Evaluating Book-Length Stories A Systematic Study Of Long Story ](others/what_matters_in_evaluating_book-length_stories_a_systematic_study_of_long_story_.md)**

:   本文系统研究了书籍级长篇故事（>100K tokens）的自动评估问题，构建了首个大规模长篇故事评估基准LongStoryEval（600本新出版小说、340K条读者评论），提出分层评价标准体系，比较三种评估策略的有效性，并训练了专用评估模型NovelCritique-8B，在与人类评分的对齐度上超越GPT-4o。

**[When To Speak When To Abstain](others/when_to_speak_when_to_abstain.md)**

:   提出 CDA（Contrastive Decoding with Abstention），一种免训练解码方法，通过熵校准的不确定性估计让 LLM 在参数/上下文知识可用时生成正确回答、在两者都不可靠时主动弃权，覆盖全部四种知识可用性场景。

**[Why Are Positional Encodings Nonessential For Deep Autoregressive Transformers R](others/why_are_positional_encodings_nonessential_for_deep_autoregressive_transformers_r.md)**

:   这是一篇综述/回顾性论文，系统解释了多层自回归 Transformer 无需显式位置编码即可区分不同序列的原理，并追溯该结论的历史起源（Irie et al., 2019），纠正了社区中将此发现误归因于后来工作的现象。

**[Words Of Warmth Trust And Sociability Norms For Over 26K English Words](others/words_of_warmth_trust_and_sociability_norms_for_over_26k_english_words.md)**

:   构建了首个大规模人工标注的英文词汇-温暖（Warmth）关联词库 Words of Warmth，覆盖 26k+ 词汇的信任（Trust）和社会性（Sociability）维度评分，并利用该资源研究了儿童词汇习得和社交媒体刻板印象。

**[Xturing Enhanced Turing Test](others/xturing_enhanced_turing_test.md)**

:   提出 X-Turing 框架，通过引入 burst 对话模式和伪对话生成技术来增强和高效化图灵测试，能够评估 LLM 在长期对话中的人类模仿能力，发现 LLM 随着对话轮次增加表现显著下降。

**[You Need To Mimic To Get Fame Solving Meeting Transcript Scarcity With A Multi-A](others/you_need_to_mimic_to_get_fame_solving_meeting_transcript_scarcity_with_a_multi-a.md)**

:   提出 MIMIC 框架，通过多智能体辩论模拟生成合成会议转录，构建了包含 800 场会议的 FAME 数据集（500 英语 + 300 德语），并设计了基于心理学的行为真实性评估框架。

**[Zero-Shot Conversational Stance Detection Dataset And Approaches](others/zero-shot_conversational_stance_detection_dataset_and_approaches.md)**

:   构建了首个零样本多轮多方对话立场检测数据集 ZS-CSD（280 个目标、17k+ 样本），并提出 SITPCL 模型，通过说话者交互建模和目标感知原型对比学习实现了 SOTA 性能（F1-macro 43.81%）。
