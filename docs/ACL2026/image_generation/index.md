---
title: >-
  ACL2026 图像生成方向11篇论文解读
description: >-
  11篇ACL2026的图像生成方向论文解读，涵盖对话系统、语音、文生图等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎨 图像生成

**💬 ACL2026** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (208)](../../CVPR2026/image_generation/index.md) · [🔬 ICLR2026 (154)](../../ICLR2026/image_generation/index.md) · [🤖 AAAI2026 (78)](../../AAAI2026/image_generation/index.md) · [🧠 NeurIPS2025 (247)](../../NeurIPS2025/image_generation/index.md) · [📹 ICCV2025 (213)](../../ICCV2025/image_generation/index.md) · [🧪 ICML2025 (112)](../../ICML2025/image_generation/index.md)

🔥 **高频主题：** 对话系统 ×2 · 语音 ×2 · 文生图 ×2

**[AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning in E-commerce](afmrl_attribute-enhanced_fine-grained_multi-modal_representation_learning_in_e-c.md)**

:   提出 AFMRL 框架，将电商产品的细粒度理解定义为属性生成任务，通过 MLLM 生成关键属性来增强对比学习（AGCL），并用检索性能作为奖励信号反向优化属性生成器（RAR），在大规模电商数据集上实现 SOTA 检索性能。

**[BookAgent: Orchestrating Safety-Aware Visual Narratives via Multi-Agent Cognitive Calibration](bookagent_orchestrating_safety-aware_visual_narratives_via_multi-agent_cognitive.md)**

:   BookAgent 是一个安全感知的多智能体框架，通过**价值对齐故事板（VAS）+ 迭代跨模态精炼（ICR）+ 时序认知校准（TCC）**三阶段闭环架构，从用户草稿端到端生成高质量、角色一致、内容安全的绘本故事。

**[CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)**

:   本文提出 CoDial，一个将预定义的对话流（task schema）转换为结构化异构图再自动生成 LLM 护栏代码（如 Colang）的框架，在推理阶段实现可解释且可控的任务型对话策略，在 STAR 基准上达到 SOTA，且无需训练数据。

**[ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling](controlaudio_tackling_text-guided_timing-indicated_and_intelligible_audio_genera.md)**

:   本文提出 ControlAudio，一个统一的渐进式扩散建模框架，通过三阶段渐进训练（TTA 预训练→时序控制微调→时序+可懂语音联合训练）和渐进引导采样，在单个扩散模型中实现文本引导、时序精确控制和可懂语音生成三种能力，在时序精度和语音清晰度上显著超越现有方法。

**[Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models](follow_the_flow_on_information_flow_across_textual_tokens_in_text-to-image_model.md)**

:   本文通过因果干预框架系统研究了文本到图像模型中文本编码器输出的 token 级信息分布，发现词汇项的语义通常集中在 1-2 个代表性 token 上，且跨项信息流在 11% 的情况下会导致语义泄漏和图像错误解读，并提出了简单有效的 token 级干预方法来改善对齐。

**[From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation](from_past_to_path_masked_history_learning_for_next-item_prediction_in_generative.md)**

:   提出掩码历史学习（MHL）训练框架，通过在生成式推荐的自回归训练中加入掩码历史重建辅助任务，结合熵引导的自适应掩码策略和课程学习调度器，使模型从仅预测"下一个是什么"转向理解"为什么形成这条路径"，在三个数据集上显著超越SOTA。

**[Investigating Counterfactual Unfairness in LLMs towards Identities through Humor](investigating_counterfactual_unfairness_in_llms_towards_identities_through_humor.md)**

:   本文通过幽默场景系统调查 LLM 的反事实不公平性——交换说话者/听众身份后观察模型行为变化，发现特权群体说的笑话被拒绝率高达 67.5%，被判定为恶意的概率高 64.7%，且社会危害评分高达 1.5 分（5分制），揭示了模型内化了固定的社会特权层级而非进行真正的社会推理。

**[Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions](large_language_models_are_bad_dice_players_llms_struggle_to_generate_random_numb.md)**

:   本文首次大规模系统审计了 11 个前沿 LLM 在 15 种概率分布上的原生采样能力，发现 LLM 严重缺乏内在概率采样机制，且这种缺陷会传导到下游应用中造成系统性偏差。

**[MASH: Evading Black-Box AI-Generated Text Detectors via Style Humanization](mash_evading_black-box_ai-generated_text_detectors_via_style_humanization.md)**

:   本文提出 MASH（多阶段风格人性化对齐），通过风格注入 SFT → DPO 对齐 → 推理时精炼三阶段流水线，训练一个仅 0.1B 参数的改写器，在黑盒设置下以 92% 的平均攻击成功率规避 AI 文本检测器，同时保持优秀的语言质量。

**[VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)**

:   本文提出 Visualize-then-Retrieve (VisRet)，一种将文本查询先通过 T2I 生成模型可视化为图像、再在图像模态内进行检索的新范式，在四个基准上平均提升 nDCG@30 0.125（CLIP）和 0.121（E5-V），下游 VQA 准确率在 Visual-RAG-ME 上提升 15.7%。

**[ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching](zipvoice-dialog_non-autoregressive_spoken_dialogue_generation_with_flow_matching.md)**

:   提出 ZipVoice-Dialog，首个基于流匹配的非自回归零样本对话语音生成模型，通过课程学习策略和说话人轮次嵌入两个简单设计，解决了流匹配直接用于对话场景时的语音不可懂和轮次混乱问题，同时发布了首个大规模开源对话语音数据集 OpenDialog（6.8k 小时）。
