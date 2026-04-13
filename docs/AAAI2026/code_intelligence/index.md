---
title: >-
  AAAI2026 代码智能方向 9篇论文解读
description: >-
  9篇AAAI2026 代码智能方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💻 代码智能

**🤖 AAAI2026** · 共 **9** 篇

**[Diffbench Meets Diffagent End-To-End Llm-Driven Diffusion Ac](diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac.md)**

:   提出DiffBench（604个扩散模型加速任务的评估基准，分5个难度等级）和DiffAgent（集成规划-编码-调试三Agent + 遗传算法选择器的闭环框架），在Claude Sonnet 4上将扩散加速代码生成通过率从54.30%提升到81.59%，复杂优化任务达成率68.27%。

**[Equacode A Multi-Strategy Jailbreak Approach For Large Language Models Via Equat](equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat.md)**

:   提出EquaCode多策略越狱方法，将恶意查询分解为方程求解（B+C+x=A）和代码补全（补全Solver类的solve()方法）的跨域组合，在GPT系列上平均攻击成功率92.78%，在最新模型（Gemini/DeepSeek/Grok）上接近100%。

**[Extracting Events Like Code A Multi-Agent Programming Framework For Zero-Shot Ev](extracting_events_like_code_a_multi-agent_programming_framework_for_zero-shot_ev.md)**

:   提出 Agent-Event-Coder (AEC)，将零样本事件抽取类比为软件工程流程，用4个专职Agent（Retrieval→Planning→Coding→Verification）协作完成抽取，并将事件schema编码为可执行Python类实现编译器式确定性验证与双循环迭代修正，在5个领域、6个LLM上全面超越零样本基线。

**[Mose Hierarchical Self-Distillation Enhances Early Layer Embeddings](mose_hierarchical_self-distillation_enhances_early_layer_embeddings.md)**

:   提出 ModularStarEncoder（MoSE），一个 10 亿参数的多出口编码器，通过新颖的自蒸馏机制（高层引导低层训练）显著增强早期层表示，在 CodeSearchNet 等代码理解任务上超越所有开源模型，同时支持灵活的计算-精度权衡部署。

**[Recode Updating Code Api Knowledge With Reinforcement Learning](recode_updating_code_api_knowledge_with_reinforcement_learning.md)**

:   提出 ReCode 框架，通过基于规则的强化学习（而非 SFT）训练 LLM 在 prompt 中正确利用 API 更新文档完成代码版本迁移，使 7B 模型在 CodeUpdateArena 上超越 32B 模型。

**[Span Benchmarking And Improving Cross-Calendar Temporal Reasoning Of Large Langu](span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)**

:   提出SPAN跨日历时间推理基准（6种日历×10推理方向×100年范围×37380实例），发现基础LLM平均仅34.5%准确率（无一超过80%），揭示Future-Date Degradation和Calendar Asymmetry Bias两种系统性失败模式，工具增强的Time Agent达95.31%——证明跨日历推理需要外部工具而非参数化知识。

**[Tapas Are Free Training-Free Adaptation Of Programmatic Agen](tapas_are_free_training-free_adaptation_of_programmatic_agen.md)**

:   TAPA 将 LLM 定位为符号动作空间的"智能调制器"而非直接决策者，通过 LLM 引导的程序合成动态适配程序化 Agent 的符号动作，无需重新训练即可适应动态环境，在网络安全 DDoS 防御（77.7% 网络正常运行率）和群体智能编队控制中表现优异。

**[Towards Better Code Understanding In Decoder-Only Large Language Models Via Hie](towards_better_code_understanding_in_decoder-only_large_language_models_via_hie.md)**

:   提出CL4D对比学习框架，通过继续预训练将decoder-only代码生成模型适配到代码理解任务（代码搜索、克隆检测），在不重新训练encoder模型的前提下实现了与同等规模encoder-only模型相当甚至更优的性能。

**[Towards Better Code Understanding In Decoder-Only Models With Contrastive Learni](towards_better_code_understanding_in_decoder-only_models_with_contrastive_learni.md)**

:   本文提出CL4D框架，通过对比学习对预训练的decoder-only代码生成模型进行继续预训练，使其能够有效提取代码表示并在代码搜索和克隆检测等理解任务上达到甚至超越同规模encoder-only模型的性能。
