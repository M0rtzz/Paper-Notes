---
title: >-
  AAAI2026 社会计算方向 10篇论文解读
description: >-
  10篇AAAI2026 社会计算方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 社会计算

**🤖 AAAI2026** · **10** 篇论文解读

**[Argumentative Debates for Transparent Bias Detection](argumentative_debates_for_transparent_bias_detection_technic.md)**

:   提出 ABIDE（Argumentative BIas Detection by DEbate），通过基于邻域属性的论证方案（argument schemes）构建量化双极论证框架（QBAF），将偏见检测过程建模为结构化辩论，实现从单邻域到全局的透明偏见推理，并形式化证明 QBAF 语义与偏见检测期望行为之间的对应关系。

**[Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)**

:   本文提出ED2D框架，在多智能体辩论（MAD）系统中引入证据检索模块来增强虚假信息检测准确率，并通过受控人类实验首次对比了AI生成的辩论稿与专家人工fact-check在说服力和信念纠正方面的效果，揭示了AI辩论系统在正确时具有专家级说服力、但在错误时可能加剧误导的双刃剑效应。

**[Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition](cross-modal_prompting_for_balanced_incomplete_multi-modal_emotion_recognition.md)**

:   提出 Cross-modal Prompting (ComP) 方法，通过渐进式提示生成+跨模态知识传播+动态调度器来解决不完整多模态情感识别中的模态不平衡问题，在 4 个数据集、 7 种缺失率下均达到 SOTA。

**[Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System](fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)**

:   提出 Fact2Fiction，首个针对 Agent 化事实核查系统（如 DEFAME、InFact）的投毒攻击框架：通过 Planner Agent 模拟声明分解生成子问题，利用系统的 justification 反向工程关键推理点来制作定向恶意证据，并按重要性分配投毒预算，在仅 1% 投毒率下比 SOTA PoisonedRAG 高 8.9%-21.2% 的攻击成功率。

**[FactGuard: Event-Centric and Commonsense-Guided Fake News Detection](factguard_event-centric_and_commonsense-guided_fake_news_detection.md)**

:   提出 FactGuard 框架，利用 LLM 提取事件核心内容（去风格化）并生成常识推理，通过 Rationale Usability Evaluator 动态评估 LLM 建议的可信度，并通过知识蒸馏获得无需 LLM 的轻量版 FactGuard-D，在假新闻检测中兼顾鲁棒性和效率。

**[Factor(U,T): Controlling Untrusted AI by Monitoring their Plans](factorut_controlling_untrusted_ai_by_monitoring_their_plans.md)**

:   研究不可信 AI 做任务分解、可信 AI 做执行的 Factor(U,T) 协议安全性，发现监控分解计划的 AUROC 仅 0.52（接近随机），而监控具体代码实现可达 0.96——恶意意图在抽象计划中难以检测但在具体实现中暴露，结论是"结构性预防（可信分解器）优于事后监控"。

**[Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering](multi-modal_dynamic_proxy_learning_for_personalized_multiple_clustering.md)**

:   本文提出Multi-DProxy框架，通过门控跨模态融合、双约束代理优化和动态候选词管理三大创新机制，利用可学习的文本代理实现个性化多重聚类，在全部公开基准上达到SOTA。

**[Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference](reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in.md)**

:   提出OmiGraph，首个基于"遗漏感知"的虚假信息检测框架，通过构建遗漏感知图、利用LLM推理遗漏意图、以及遗漏导向的消息传递与聚合机制，从"未说出的内容"中提取欺骗模式，在双语数据集上平均提升+5.4% F1和+5.3% ACC。

**[SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation](scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval.md)**

:   提出SceneJailEval，一个场景自适应的多维度越狱评估框架，定义14个越狱场景和10个评估维度，通过场景分类→维度动态选择→多维检测→加权危害评分的流程，在自建数据集上F1达0.917（超SOTA 6%），在JBB上达0.995（超SOTA 3%），同时支持危害程度量化而非仅二分类。

**[T2Agent: A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search](t2agent_a_tool-augmented_multimodal_misinformation_detection_agent_with_monte_ca.md)**

:   提出 T2Agent，一个集成可扩展工具集与蒙特卡洛树搜索（MCTS）的虚假信息检测智能体，通过多源验证机制将检测任务分解为针对不同伪造源的子任务，在 MMfakebench 上以 GPT-4o 为骨干将基线 MMDAgent 的准确率提升 28.7%，达到新 SOTA。
