---
title: >-
  ICLR2026 医疗 NLP方向5篇论文解读
description: >-
  5篇ICLR2026的医疗 NLP 方向论文解读，涵盖医学影像、对话系统、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "医疗 NLP"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "对话系统"
  - "推理"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🩺 医疗 NLP

**🔬 ICLR2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (2)](../../ICML2026/medical_nlp/index.md) · [💬 ACL2026 (14)](../../ACL2026/medical_nlp/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/medical_nlp/index.md) · [🧠 NeurIPS2025 (6)](../../NeurIPS2025/medical_nlp/index.md) · [🧪 ICML2025 (1)](../../ICML2025/medical_nlp/index.md) · [💬 ACL2025 (13)](../../ACL2025/medical_nlp/index.md)

🔥 **高频主题：** 医学影像 ×3 · 对话系统 ×2 · 推理 ×2

**[ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue.md)**

:   提出 ATPO（自适应树策略优化）算法，将多轮医疗对话建模为层级马尔可夫决策过程（H-MDP），通过不确定性感知的自适应树扩展机制动态分配rollout预算，结合Bellman误差和动作值方差的复合不确定性度量来引导探索，在三个医学对话基准上以Qwen3-8B超越GPT-4o。

**[BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases](biomedsql_text-to-sql_for_scientific_reasoning_on_biomedical_knowledge_bases.md)**

:   提出 BiomedSQL，首个专门评估 Text-to-SQL 系统在生物医学知识库上科学推理能力的基准，包含 68,000 个问题/SQL/答案三元组，揭示当前最强模型（GPT-o3-mini 62.6%）与领域专家（90%）之间仍有巨大差距。

**[CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](counselbench_llm_mental_health_qa.md)**

:   联合100名持证心理健康专家构建CounselBench双组件基准——CounselBench-EVAL（2,000条六维度专家评估）和CounselBench-Adv（120个对抗性问题+1,080条响应标注），系统性揭示LLM在心理健康开放式问答中表面得分高但存在过度泛化、擅自医疗建议等安全隐患，同时证明LLM-as-Judge在安全关键领域严重不可靠。

**[From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents](from_conversation_to_query_execution_benchmarking_user_and_tool_interactions_for.md)**

:   提出EHR-ChatQA基准，首次评估数据库Agent在电子病历场景中的端到端交互工作流（澄清模糊查询→解决术语不匹配→生成SQL→返回答案），发现最强模型(o4-mini)的Pass@5超90%但Pass∧5(全部成功)大幅下降(差距达60%)，暴露了安全关键领域的鲁棒性缺陷。

**[MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science](medagentgym_agentic_training_biomedical.md)**

:   构建了首个统一的生物医学数据科学 Agent 训练环境 MedAgentGym，包含 72,413 个任务实例（覆盖 12 个真实场景、129 个类别），配备可执行沙盒和可验证 ground truth，系统基准评估 29 个 LLM 揭示商业/开源差距，并通过高效多线程轨迹采样 + 离线/在线 RL 训练出 Med-Copilot，分别获得 +43.02%/+45.28% 提升，达到与 GPT-4o 竞争的性能。
