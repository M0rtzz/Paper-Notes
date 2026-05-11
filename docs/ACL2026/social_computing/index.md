---
title: >-
  ACL2026 社会计算方向9篇论文解读
description: >-
  9篇ACL2026的社会计算方向论文解读，涵盖 LLM、推理、语音、情感分析等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 社会计算

**💬 ACL2026** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (4)](../../CVPR2026/social_computing/index.md) · [🔬 ICLR2026 (10)](../../ICLR2026/social_computing/index.md) · [🤖 AAAI2026 (11)](../../AAAI2026/social_computing/index.md) · [🧠 NeurIPS2025 (18)](../../NeurIPS2025/social_computing/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/social_computing/index.md) · [🧪 ICML2025 (7)](../../ICML2025/social_computing/index.md)

🔥 **高频主题：** LLM ×2 · 推理 ×2

**[Among Us: Language of Conspiracy Theorists on Mainstream Reddit](among_us_language_of_conspiracy_theorists_on_mainstream_reddit.md)**

:   分析5亿条Reddit评论的10年纵向数据，发现活跃于阴谋论社区的用户在主流社区中也展现出可检测的独特语言模式（平均87%分类准确率），但这些模式高度依赖社区上下文，社区特定模型比全局模型高出最多17个百分点。

**[Explain the Flag: Contextualizing Hate Speech Beyond Censorship](explain_the_flag_contextualizing_hate_speech_beyond_censorship.md)**

:   本文提出一种混合方法，结合 LLM 和三种语言（英/法/希腊语）的人工策展词汇表来检测和解释仇恨言论——术语管道通过词汇匹配+LLM 语义消歧检测固有贬损用语，无术语管道用 LLM 检测群体针对性内容，两者融合生成有据可查的解释。

**[How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects](how_language_models_conflate_logical_validity_with_plausibility_a_representation.md)**

:   通过表示分析揭示 LLM 中"逻辑有效性"和"合理性"两个概念在隐层空间中高度对齐，导致模型将合理性与有效性混淆（内容效应），并构造去偏转向向量有效解耦这两个概念，减少内容效应同时提升推理准确率。

**[Is this chart lying to me? Automating the detection of misleading visualizations](is_this_chart_lying_to_me_automating_the_detection_of_misleading_visualizations.md)**

:   提出 Misviz（2604张真实世界误导性可视化）和 Misviz-synth（57665张合成可视化）基准，覆盖12种误导类型，系统评估MLLM、规则检查器和图像分类器在检测误导性图表上的表现，揭示该任务仍极具挑战性。

**[On the Step Length Confounding in LLM Reasoning Data Selection](on_the_step_length_confounding_in_llm_reasoning_data_selection.md)**

:   本文发现基于自然度的 LLM 推理数据选择方法存在"步长混淆"问题——系统性地偏好每步更长的样本而非更高质量的样本，根因是推理步骤首 token 的低概率被长步骤稀释。提出 Aslec-drop（丢弃首 token 概率）和 Aslec-casl（因果回归去偏）两种校正方法，平均准确率提升 6-9%。

**[Persona-E2: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events](persona-e2_a_human-grounded_dataset_for_personality-shaped_emotional_responses_t.md)**

:   构建了首个将人格特质（MBTI + Big Five）与读者情感反应关联的大规模数据集 Persona-E2，包含 3111 个事件 × 36 名标注者共 11.2 万条标注，揭示 LLM 在模拟人格化情感反应时存在"人格幻觉"问题，且 Big Five 特征比 MBTI 更有效地缓解该问题。

**[SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)**

:   本文提出 SPAGBias 框架，首次系统评估 LLM 在城市微观空间语境中的性别偏见，通过显式偏见、概率偏见和建构偏见三个诊断层揭示了 LLM 中结构化的空间-性别关联模式，并追溯偏见在模型开发全流程中的嵌入与放大。

**[ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)**

:   ToxiTrace 提出了一种面向 BERT 类编码器的可解释中文毒性检测方法，通过 CuSA（LLM 引导的弱标注）、GCLoss（梯度约束损失）和 ARCL（对抗推理对比学习）三个组件，在保持高效编码器推理的同时实现了句级分类准确率和连续有毒片段提取的双重提升。

**[ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway](toxreason_a_benchmark_for_mechanistic_chemical_toxicity_reasoning_via_adverse_ou.md)**

:   本文提出 ToxReason，一个基于不良结局路径 (AOP) 框架的化学毒性机理推理基准，整合药物-靶点实验数据与毒性标签，要求模型从分子起始事件推理到器官级不良结局；通过 GRPO 强化学习训练的 4B 模型在毒性预测（F1 71.4%）和推理质量上均超越 GPT-5 等大模型。
