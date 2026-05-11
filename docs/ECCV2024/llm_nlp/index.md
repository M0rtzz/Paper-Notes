---
title: >-
  ECCV2024 LLM / NLP方向18篇论文解读
description: >-
  18篇ECCV2024的 LLM / NLP 方向论文解读，涵盖少样本学习、人脸/视线、持续学习、Agent、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM / NLP

**🎞️ ECCV2024** · **18** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (36)](../../ACL2026/llm_nlp/index.md) · [📷 CVPR2026 (9)](../../CVPR2026/llm_nlp/index.md) · [🔬 ICLR2026 (39)](../../ICLR2026/llm_nlp/index.md) · [🤖 AAAI2026 (38)](../../AAAI2026/llm_nlp/index.md) · [🧠 NeurIPS2025 (53)](../../NeurIPS2025/llm_nlp/index.md) · [📹 ICCV2025 (8)](../../ICCV2025/llm_nlp/index.md)

🔥 **高频主题：** 少样本学习 ×3

**[APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension](apl_anchor-based_prompt_learning_for_one-stage_weakly_supervised_referring_expre.md)**

:   本文提出锚框提示学习方法 APL，通过设计锚框提示编码器（APE）生成位置、颜色、类别三类判别性提示，动态融入锚框特征以丰富视觉语义，再配合文本重构损失和视觉对齐损失实现精确的视觉-语言对齐，在四个 REC 基准上超越现有弱监督方法（如 RefCOCO 上比 RefCLIP 高 6.44%）。

**[ControlLLM: Augment Language Models with Tools by Searching on Graphs](controlllm_augment_language_models_with_tools_by_searching_on_graphs.md)**

:   提出 ControlLLM 框架，通过在预构建的工具图（Tool Graph）上进行图搜索（Thoughts-on-Graph）来规划多模态工具调用，显著提升了复杂任务中工具选择和参数赋值的准确性。

**[Cultural Value Differences of LLMs: Prompt, Language, and Model Size](cultural_value_differences_llms.md)**

:   本文使用 Hofstede 文化维度问卷系统性地研究 LLM 表达文化价值观的行为模式，发现提示语言（中文 vs 英文）和模型规模对文化价值差异的影响远大于模型架构差异和问题顺序变化。

**[DreamStruct: Understanding Slides and User Interfaces via Synthetic Data Generation](dreamstruct_understanding_slides_and_user_interfaces_via_synthetic_data_generati.md)**

:   提出利用代码生成合成结构化视觉数据（幻灯片和UI），用于训练理解模型，减少人工标注需求。

**[Evaluating Text-to-Visual Generation with Image-to-Text Generation](evaluating_text-to-visual_generation_with_image-to-text_generation.md)**

:   提出VQAScore，利用VQA模型替代CLIP来评估文本-视觉生成质量，在复杂组合性提示上大幅超越CLIPScore，并发布GenAI-Bench基准。

**[FunQA: Towards Surprising Video Comprehension](funqa_towards_surprising_video_comprehension.md)**

:   构建了大规模反直觉视频问答基准 FunQA（4.3K 视频、312K QA 对），覆盖幽默/创意/魔术三类令人惊讶的视频，并提出 FunMentor 智能体通过多轮对话增强 VLM 的反常识推理能力。

**[Meta-Prompting for Automating Zero-Shot Visual Recognition with LLMs](meta-prompting_for_automating_zero-shot_visual_recognition_with_llms.md)**

:   提出 MPVR（Meta-Prompting for Visual Recognition），通过两阶段 meta-prompting 策略自动化生成多样化的类别特定 VLM prompt，无需人工设计 LLM 查询即可显著提升 CLIP 等模型的 zero-shot 识别性能。

**[One-stage Prompt-based Continual Learning](one-stage_prompt-based_continual_learning.md)**

:   提出 OS-Prompt 框架，通过直接使用 ViT 中间层 token embedding 作为 prompt query（而非额外的 query ViT 前向传播），将 Prompt-based Continual Learning 的计算成本降低约 50%，并通过 Query-Pool Regularization (QR) loss 补偿表征能力损失，在 CIFAR-100、ImageNet-R、DomainNet 上超越 CodaPrompt 约 1.4%。

**[PromptIQA: Boosting the Performance and Generalization for No-Reference Image Quality Assessment via Prompts](promptiqa_boosting_the_performance_and_generalization_for_no-reference_image_qua.md)**

:   提出 PromptIQA，通过少量"图像-分数对"（ISP）作为 prompt 的方式，使 NR-IQA 模型训练完成后无需微调即可自适应适配新的质量评估需求，在 12 个数据集、5 类 IQA 任务上均达到 SOTA 性能和泛化能力。

**[Propose, Assess, Search: Harnessing LLMs for Goal-Oriented Planning in Instructional Videos](propose_assess_search_harnessing_llms_for_goal-oriented_planning_in_instructiona.md)**

:   VidAssist提出"提议-评估-搜索"三步框架，利用LLM作为知识库和评估工具，结合广度优先搜索算法，在教学视频的目标导向规划任务中以零/少样本方式超越全监督SOTA，few-shot在COIN上比全监督VLaMP高+7.7% SR。

**[Reprojection Errors as Prompts for Efficient Scene Coordinate Regression](reprojection_errors_as_prompts_for_efficient_scene_coordinate_regression.md)**

:   本文提出 EGFS（Error-Guided Feature Selection）机制，利用低重投影误差区域作为 SAM 的 point prompts 扩展为语义掩码，迭代地筛选可靠训练样本，在 Cambridge Landmarks 和 Indoor6 数据集上以更小模型和更少训练时间超越现有无 3D 信息依赖的 SCR 方法。

**[RoadPainter: Points Are Ideal Navigators for Topology Transformer](roadpainter_points_are_ideal_navigators_for_topology_transformer.md)**

:   提出 RoadPainter，通过先回归车道中心线点再利用实例 mask 精炼的两阶段策略，结合混合注意力机制和真实-虚拟车道分离策略，在 OpenLane-V2 数据集上实现 SOTA 的拓扑推理性能。

**[Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities](self-adapting_large_visual-language_models_to_edge_devices_across_visual_modalit.md)**

:   提出EdgeVL框架，通过两阶段适配（双模态知识蒸馏+量化感知对比学习），将大规模VLM（如CLIP）适配到边缘设备上，实现无需人工标注的跨模态（RGB和非RGB）开放词汇分类，达到最高15.4%的准确率提升和93倍的模型压缩。

**[SLIMER: Show Less, Instruct More - Enriching Prompts with Definitions and Guidelines for Zero-Shot NER](slimer_zero_shot_ner.md)**

:   SLIMER 通过在提示中注入实体定义和标注指南来增强 LLM 的零样本命名实体识别能力，仅用 391 个实体类别训练即可在从未见过的实体标签上达到与使用 13000+ 实体类别训练的 SOTA 方法相当的性能。

**[Stripe Observation Guided Inference Cost-Free Attention Mechanism](stripe_observation_guided_inference_cost-free_attention_mechanism.md)**

:   本文通过深入分析Transformer中注意力权重矩阵的条纹（stripe）模式现象，提出一种推理阶段完全无额外计算开销的注意力增强机制——仅在训练阶段通过辅助模块学习条纹引导的注意力修正，并在推理时将其重参数化融入标准注意力权重中，实现"免费午餐"式的性能提升。

**[TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)**

:   利用两个语言模型分别进行布局规划和布局编码，实现更灵活、更多样化的视觉文本渲染，在文本准确性和风格多样性之间取得更好的平衡。

**[Towards Open-Ended Visual Recognition with Large Language Model](towards_open-ended_visual_recognition_with_large_language_models.md)**

:   提出 OmniScient Model (OSM)——一个基于冻结 CLIP-ViT + 可训练 MaskQ-Former + 冻结 LLM (Vicuna-7B) 的生成式 mask 分类器，将视觉识别从"从预定义词表中选择类别"转变为"直接生成类别名称"，消除了训练和测试时对预定义词表的依赖，在 COCO 全景分割上超越 DaTaSeg +4.3 PQ。

**[Zero-Shot Object Counting with Good Exemplars (VA-Count)](zeroshot_object_counting_with_good_exemplars.md)**

:   提出 VA-Count，一种基于视觉关联的零样本物体计数框架，通过 Grounding DINO 驱动的样例增强模块和对比学习噪声抑制模块，为任意类别建立高质量样例与图像间的鲁棒视觉关联。
