---
title: >-
  ACL2025 multi_agent方向33篇论文解读
description: >-
  33篇ACL2025的 multi_agent 方向论文解读，涵盖 Agent、LLM、推理、对抗鲁棒、强化学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📄 multi_agent

**💬 ACL2025** · **33** 篇论文解读

🔥 **高频主题：** Agent ×31 · LLM ×8 · 推理 ×5 · 对抗鲁棒 ×2 · 强化学习 ×2

**[AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](agentdropout-dynamic-agent-elimination-for-multi-agent-collaboration.md)**

:   本文提出 AgentDropout，通过在多轮讨论中动态消除冗余 Agent（节点剪枝）和冗余通信边（边剪枝），在降低 21.6% prompt token 消耗的同时提升了 1.14 分的任务性能。

**[Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks](agents_under_siege_breaking_pragmatic_multi-agent_llm_systems_with_optimized_pro.md)**

:   本文首次系统研究了在带宽约束、延迟和安全机制的现实多智能体LLM系统中的对抗攻击问题，提出基于最大流最小费用的拓扑优化和排列不变蒙骗损失（PIEL）的攻击方法，在多个LLM架构上实现了高达7倍于传统攻击的成功率。

**[Auto-TA: Towards Scalable Automated Thematic Analysis (TA) via Multi-Agent Large Language Models with Reinforcement Learning](auto-ta_towards_scalable_automated_thematic_analysis_ta_via_multi-agent_large_la.md)**

:   提出一个基于多智能体 LLM 的全自动主题分析（Thematic Analysis）流水线，通过专业角色分工和可选的 RLHF 微调，实现对临床叙事的端到端主题提取，消除了人工编码和全文审阅的需求。

**[Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)**

:   提出 Bel Esprit 多 Agent 对话框架，通过 Mentalist（需求澄清）→ Builder（管线构建）→ Inspector（验证）→ Matchmaker（模型分配）四步协作，将用户模糊的自然语言需求自动转化为多模型 AI 管线图，在 441 条管线数据上达到 25.2% EM 和 37.0 GED（GPT-4o Builder）。

**[Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems](beyond_frameworks_multi_agent_collaboration.md)**

:   本文系统化地将多智能体协作分解为四个维度（治理模式、参与控制、交互模式、上下文管理），通过两个上下文依赖任务的大量实验证明：集中治理+指导者控制参与+有序交互+指导者摘要的组合最优，在保持甚至提升准确率的同时减少高达 93% 的 token 消耗。

**[CoMet: Metaphor-Driven Covert Communication for Multi-Agent Language Games](comet_metaphor-driven_covert_communication_for_multi-agent_language_games.md)**

:   本文提出 CoMet 框架，通过整合基于假设检验的隐喻推理器和自改进式隐喻生成器，使 LLM 智能体能在多智能体语言博弈中运用隐喻进行隐蔽通信和语义规避，在 Undercover 和 Adversarial Taboo 两个游戏中显著提升了智能体的策略沟通能力（胜率从 0.20 提升至 0.70）。

**[CortexDebate: Debating Sparsely and Equally for Multi-Agent Debate](cortexdebate_debating_sparsely_and_equally_for_multi-agent_debate.md)**

:   提出 CortexDebate，一种受人脑皮层工作机制启发的多智能体辩论方法，通过构建稀疏动态辩论图和基于 McKinsey 信任公式的评估模块（MDM），同时解决了现有 MAD 方法中"输入上下文过长"和"过度自信导致不平等辩论"两大核心问题。

**[Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement](debate_reflect_and_distill_multi-agent_feedback_with_tree-structured_preference_.md)**

:   提出 D&R 框架，让小模型（student）与多个大模型（teacher）进行多轮辩论并收集自我反思和教师反馈，然后将辩论日志组织为偏好树做 Tree-structured DPO (T-DPO) 蒸馏，在 MMLU Pro 和 MATH 上平均提升 14.18 分，且推理效率优于基线。

**[DocAgent: A Multi-Agent System for Automated Code Documentation Generation](docagent_a_multi-agent_system_for_automated_code_documentation_generation.md)**

:   提出 DocAgent，一个基于拓扑依赖排序的多智能体代码文档生成系统，通过 Reader-Searcher-Writer-Verifier 协作流程增量构建上下文，在完整性、实用性和真实性三个维度上显著优于 FIM 和 Chat 基线。

**[EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework](educationq_evaluating_llms_teaching_capabilities_through_multi-agent_dialogue_fr.md)**

:   提出 EducationQ 多智能体对话框架，通过模拟真实课堂中教师-学生的形成性评估交互来评估 LLM 的教学能力，发现教学效果与模型规模或通用推理能力不呈线性关系，Llama 3.1 70B 在教学中表现最优。

**[EMULATE: A Multi-Agent Framework for Determining the Veracity of Atomic Claims by Emulating Human Actions](emulate_a_multi-agent_framework_for_determining_the_veracity_of_atomic_claims_by.md)**

:   提出 EMULATE 多智能体事实核查框架，通过 7 个专职 LLM agent 模拟人类验证声明的完整行为链（搜索→排序→内容评估→证据充分性判断→分类），在三个事实核查 benchmark 上的 Macro-F1 和 Weighted-F1 均超越现有方法。

**[GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning](getreason_enhancing_image_context_extraction_through_hierarchical_multi-agent_re.md)**

:   提出 GETReason，一个层级化多智能体框架，通过将公共事件图像的上下文提取分解为地理空间、时间和事件三个子任务，并由专门化的 Agent 协作完成，实现比现有方法更准确的图像上下文推理。

**[Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](graph_counselor_multiagent_graphrag.md)**

:   Graph Counselor 提出了一个多智能体协作的 GraphRAG 推理框架，通过 Planning/Thought/Execution 三个 Agent 自适应提取图结构信息，并引入多视角自反思机制纠正推理偏差，在多个图推理任务上超越现有方法。

**[LLMSR@XLLM25: Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation](llmsrxllm25_less_is_more_enhancing_structured_multi-agent_reasoning_via_quality-.md)**

:   本文提出 Less is More 框架，在仅有 24 个标注样本的极端低资源条件下，通过逆向提示归纳、GPT-4o 增强的检索式推理合成和双阶段奖励引导过滤三个阶段，蒸馏出高质量的结构化推理数据来微调 LLaMA3-8B 多智能体系统，在 XLLM@ACL2025 共享任务中获得第三名。

**[M-MAD: Multidimensional Multi-Agent Debate for Advanced Machine Translation Evaluation](m-mad_multidimensional_multi-agent_debate_for_advanced_machine_translation_evalu.md)**

:   提出 M-MAD 框架，将 MQM 评估标准解耦为独立维度（准确性、流畅性、风格、术语），在每个维度内进行多智能体正反方辩论，最后由裁判智能体综合各维度结果，在 segment 级别显著超越已有 LLM-as-a-judge 方法，甚至用 GPT-4o mini 就能媲美 SOTA 有参考自动指标。

**[MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation](main-rag_multi-agent_filtering_retrieval-augmented_generation.md)**

:   提出 MAIN-RAG，一个无需训练的多 Agent RAG 过滤框架，通过 Predictor→Judge→Final-Predictor 三个 LLM Agent 协作评估检索文档的相关性，并设计自适应阈值（基于分数均值和标准差）动态过滤噪声文档，在 4 个 QA 基准上实现 2-11% 的准确率提升。

**[MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration](mam_modular_multi-agent_framework_for_multi-modal_medical_diagnosis_via_role-spe.md)**

:   提出模块化多智能体框架 MAM，将医学诊断过程分解为全科医生、专科团队、放射科医生、医学助手和主任医师五个角色，通过角色专业化协作实现多模态（文本/图像/音频/视频）医学诊断，在多个公开数据集上比基线模型提升 18%~365%。

**[MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](maporl_multi-agent_post-co-training_for_collaborative_large_language_models_with.md)**

:   提出 MAPoRL——一种基于多智能体强化学习的后训练范式，通过让多个 LLM 在辩论框架中共同训练（co-training），配合验证器评分和协作激励机制，显著提升多 LLM 协作的效果，并展现出跨任务的泛化能力。

**[MasRouter: Learning to Route LLMs for Multi-Agent Systems](masrouter_learning_to_route_llms_for_multi-agent_systems.md)**

:   首次提出多智能体系统路由（MASR）问题，设计 MasRouter 级联控制器网络，依次决定协作模式、角色分配和 LLM 路由，在保持高性能的同时将 MAS 的推理成本降低最高 52%，实现效果与效率的平衡。

**[METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)**

:   提出 METAL，一个基于 VLM 的多智能体框架，将图表生成任务（chart-to-code）分解为生成、视觉评审、代码评审和修订四个专门化智能体的迭代协作，在 ChartMIMIC 基准上比现有最佳方法提升 5.2% F1，并展现了测试时缩放（test-time scaling）现象。

**[MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection](mind_a_multi-agent_framework_for_zero-shot_harmful_meme_detection.md)**

:   提出 MIND 框架，通过相似样本检索、双向洞察推导和多智能体辩论三个阶段实现零样本有害梗图（meme）检测，无需标注数据即可在三个数据集上超越现有零样本方法，并在不同模型架构和规模上展现强泛化性。

**[Multi-Agent Collaboration via Cross-Team Orchestration](multi-agent_collaboration_via_cross-team_orchestration.md)**

:   提出 Cross-Team Orchestration (Croto)，一个可扩展的多团队协作框架，通过将多个独立 agent 团队组织起来进行跨团队交互，利用层次化分组 (Hierarchy Partitioning) 和贪心聚合 (Greedy Aggregation) 机制将各团队的多样化解决方案融合为更优结果。

**[A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems](multi_agent_dialect_bias_privacy_qa.md)**

:   构建 Dialect Agent（方言翻译+审查）与 Privacy Policy Agent（领域回答）的双 Agent 迭代协作框架，通过注入方言语言学知识的提示工程，在无需重训练的前提下同时提升隐私政策 QA 的整体准确率和跨方言公平性。

**[MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents](multiagentbench_evaluating_the_collaboration_and_competition_of_llm_agents.md)**

:   提出 MultiAgentBench 基准测试和 MARBLE 框架，系统评估 LLM 多智能体系统在协作与竞争场景中的表现，包含 6 种交互场景（研究、Minecraft、数据库、编程、讨价还价、狼人杀），引入基于里程碑的 KPI 指标和协调评分，发现 gpt-4o-mini 整体任务分最高、图结构协调协议在研究场景中表现最佳、认知规划可提升里程碑达成率 3%。

**[Preventing Rogue Agents Improves Multi-Agent Collaboration](preventing_rogue_agents_improves_multi-agent_collaboration.md)**

:   提出一种通过实时监控 Agent 不确定性来检测"失控 Agent"（rogue agent）并进行干预的框架，在自建的 WhoDunitEnv 多智能体协作环境以及代码生成和资源可持续性任务上分别取得高达 17.4%、2.5% 和 20% 的性能提升。

**[Red-Teaming LLM Multi-Agent Systems via Communication Attacks](red-teaming_llm_multi-agent_systems_via_communication_attacks.md)**

:   提出 Agent-in-the-Middle (AiTM) 攻击，通过拦截和篡改 LLM 多智能体系统中的 agent 间通信消息（而非直接修改 agent 本身），利用一个带反思机制的对抗性 agent 生成上下文感知的恶意指令，在多种框架/通信结构/真实应用上均取得 40%~100% 的攻击成功率。

**[Select, Read, and Write: A Multi-Agent Framework of Full-Text-based Related Work Generation](select_read_and_write_a_multi-agent_framework_of_full-text-based_related_work_ge.md)**

:   提出 Select-Read-Write 三 Agent 协同框架，通过图感知的阅读顺序决策和共享工作记忆机制，实现基于论文全文（而非摘要）的 Related Work 自动生成，在 Llama3-8B / Claude-3-Haiku / GPT-4o 三个基座模型上均取得一致提升，Citation Graph 策略效果最优。

**[Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation](synthesizing_post-training_data_for_llms_through_multi-agent_simulation.md)**

:   本文提出 MATRIX 多智能体模拟器和 MATRIX-Gen 场景驱动指令生成器，通过模拟真实社会场景来合成高质量的 LLM 后训练数据，仅用 20K 条合成数据训练的 Llama-3-8B 在 AlpacaEval 2 和 Arena-Hard 上超过了使用超过 10M 数据训练的 Meta 官方 Llama-3-8B-Instruct。

**[Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](table_critic_multi_agent.md)**

:   提出 Table-Critic 多智能体框架，通过 Judge-Critic-Refiner-Curator 四个专门化 Agent 的协作批评与迭代精化，配合自进化模板树累积批评知识，在 WikiTQ 和 TabFact 上分别实现 73.7% 和 91.7% 的准确率，大幅超越现有方法。

**[Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models](theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)**

:   提出 Theorem-of-Thought (ToTh) 框架，通过三个并行智能体分别模拟溯因、演绎和归纳推理，将推理轨迹构建为形式推理图并利用 NLI 校准的贝叶斯置信传播选出最连贯推理链，在符号和数值推理上一致优于 CoT、Self-Consistency 和 CoT-Decoding。

**[A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns](tmcht_contagious_jailbreak_multiagent.md)**

:   提出TMCHT（大规模多智能体多拓扑文本攻击评估框架）和ARCJ（对抗性复制传染越狱）方法——通过优化检索后缀提高毒性样本被检索概率+优化复制后缀使毒性信息具有自我复制传染能力，解决了多智能体系统中单智能体攻击方法面临的"毒性消散"问题。

**[Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System](virsci_multi_agent_idea_gen.md)**

:   提出 VirSci 多 agent 系统，用真实科学家数据构建虚拟科研生态，通过 5 步协作流程和创新的组间+组内讨论机制生成科学 idea，在新颖性和潜在影响力上显著超越单 agent 系统。

**[Voting or Consensus? Decision-Making in Multi-Agent Debate](voting_or_consensus_decision-making_in_multi-agent_debate.md)**

:   系统性对比了多智能体辩论中 7 种决策协议（投票 vs 共识），发现共识协议在知识任务上提升 2.8%、投票协议在推理任务上提升 13.2%，并提出 AAD 和 CI 两种增强答案多样性的新方法，分别带来 3.3% 和 7.4% 的性能提升。
