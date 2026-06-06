---
title: >-
  NeurIPS2025 对话系统方向5篇论文解读
description: >-
  5篇NeurIPS2025的对话系统方向论文解读，涵盖 LLM、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "对话系统"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "Agent"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🗣️ 对话系统

**🧠 NeurIPS2025** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (4)](../../ICML2026/dialogue/index.md) · [💬 ACL2026 (26)](../../ACL2026/dialogue/index.md) · [📷 CVPR2026 (1)](../../CVPR2026/dialogue/index.md) · [🔬 ICLR2026 (5)](../../ICLR2026/dialogue/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/dialogue/index.md) · [🧪 ICML2025 (3)](../../ICML2025/dialogue/index.md)

🔥 **高频主题：** LLM ×2

**[AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs](aclora_almost_trainingfree_access_controlaware_multimodal_ll.md)**

:   设计 AC-LoRA 端到端系统，为不同权限数据集训练独立的 LoRA 适配器，推理时根据用户查询的 cosine 相似度和权限动态检索并无训练合并多个 LoRA 输出，在保证强信息隔离的同时匹配或超越 SOTA LoRA 混合方法的回答质量。

**[Bridging Human and LLM Judgments: Understanding and Narrowing the Gap](bridging_human_and_llm_judgments_understanding_and_narrowing_the_gap.md)**

:   提出Bridge统计框架，通过序数logistic回归建模人类和LLM评判之间的潜在关系，以少量人类标签改善LLM评判的校准和对齐，同时支持对系统性偏差的正式统计检验。

**[HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location](hygen_efficient_llm_serving_via_elastic_online-offline_request_co-location.md)**

:   提出HyGen——干扰感知的LLM推理系统，通过精准的批次延迟预测器、SLO感知的性能分析器和前缀共享最大化调度策略，实现在线和离线工作负载的弹性共置，在保证严格SLO合规的同时获得3.87-5.84倍吞吐提升。

**[MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems](metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)**

:   提出 MetaMind——一个受心理学元认知理论启发的多智能体框架，通过 ToM Agent（心理状态假设生成）、Moral Agent（社会规范约束精炼）和 Response Agent（响应生成与自我验证）三阶段协作，显著提升 LLM 的社会推理能力，在多个社会智能基准上达到 SOTA 并首次接近人类水平。

**[SciArena: An Open Evaluation Platform for Non-Verifiable Scientific Literature-Grounded Tasks](sciarena_an_open_evaluation_platform_for_non-verifiable_scientific_literature-gr.md)**

:   构建 SciArena 社区驱动的科学文献评估开放平台，采用 Chatbot Arena 式的人类偏好投票方式对 47 个基础模型进行排名，收集超过 20,000 条投票数据，并发布 SciArena-Eval 元基准来评测自动评估系统对文献任务答案质量的判断能力。
