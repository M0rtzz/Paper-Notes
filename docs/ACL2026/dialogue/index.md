---
title: >-
  ACL2026 对话系统方向9篇论文解读
description: >-
  9篇ACL2026的对话系统方向论文解读，涵盖对话系统、推理、强化学习、对齐/RLHF、情感分析、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🗣️ 对话系统

**💬 ACL2026** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (1)](../../CVPR2026/dialogue/index.md) · [🔬 ICLR2026 (5)](../../ICLR2026/dialogue/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/dialogue/index.md) · [🧠 NeurIPS2025 (5)](../../NeurIPS2025/dialogue/index.md) · [🧪 ICML2025 (3)](../../ICML2025/dialogue/index.md) · [💬 ACL2025 (22)](../../ACL2025/dialogue/index.md)

🔥 **高频主题：** 对话系统 ×7

**[Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)**

:   提出ConvAgent，通过将RL训练奖励分解为结果奖励、信息增益奖励和混合主动行为奖励三个互补组件，训练对话式搜索智能体在多轮交互中交替进行搜索和推理。

**[Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review](author-in-the-loop_response_generation_and_evaluation_integrating_author_experti.md)**

:   本文将学术论文作者回复（rebuttal）生成重新定义为"作者在回路"任务，提出 Re3Align 数据集（3.4K 论文、440K 句级编辑标注、15K 审稿-回复-修改三元组）、REspGen 可控生成框架和 REspEval 20+ 指标评估套件，在 5 个 SOTA LLM 上系统验证了作者输入、可控性和评估引导精修的效果。

**[Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky](disambiguation-centric_finetuning_makes_enterprise_tool-calling_llms_more_realis.md)**

:   提出 DiaFORGE 框架，通过消歧中心的合成数据生成管线 + 推理链微调 + 动态评估体系，让开源 LLM 在面对近重复企业 API 时的工具调用成功率比 GPT-4o 高 27 个百分点、比 Claude-3.5-Sonnet 高 49 个百分点。

**[Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation](discourse_coherence_and_response-guided_context_rewriting_for_multi-party_dialog.md)**

:   本文提出 DRCR，首个将上下文改写引入多方对话生成的框架，使用话语连贯性和回复质量双反馈信号构建偏好数据，通过动态自演化学习让改写器和回复器在迭代训练中相互增强。

**[ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)**

:   ETHICMIND 提出推理时（inference-time）的风险感知对齐框架，在多轮对话的每一轮中联合分析伦理风险和用户情感，规划高层响应策略，再生成兼顾伦理引导和情感共鸣的回复，无需额外训练即可在高风险和道德模糊场景中实现更一致的对齐表现。

**[SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation](spasm_stable_persona-driven_agent_simulation_for_multi-turn_dialogue_generation.md)**

:   本文提出 SPASM，一个以稳定性为核心的人设驱动多轮对话模拟框架，通过模块化人设生成、自我中心上下文投影（ECP）和终止检测三个组件，在 LLM-LLM 对话中大幅减少角色漂移和"回声"现象，构建了 45,000 段高质量多轮对话数据。

**[Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](template-assisted_contrastive_learning_of_task-oriented_dialogue_sentence_embedd.md)**

:   提出 TaDSE 框架，利用对话中现有的模板（template）信息作为辅助锚点，通过模板感知的数据增强、配对对比训练和语义压缩推理三个阶段，在无监督设置下显著提升任务型对话的句子嵌入质量，在五个基准上超越此前 SOTA 甚至优于有监督的商业嵌入模型。

**[Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation](towards_proactive_information_probing_customer_service_chatbots_harvesting_value.md)**

:   本文提出 ProChatIP 框架，将客服聊天机器人从被动应答工具转变为主动信息采集引擎，通过专门的对话策略模块学习"何时探测"用户以获取预设的目标信息，同时最小化对话轮数和用户摩擦。

**[VoxMind: An End-to-End Agentic Spoken Dialogue System](voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)**

:   提出 VoxMind，一个赋予端到端语音对话模型智能体能力的统一框架：通过"Think-before-Speak"机制实现显式推理，结合多智能体动态工具管理架构解耦推理延迟与工具规模，任务完成率从基线 34.88% 提升至 74.57%，超越 Gemini-2.5-Pro。
