---
title: >-
  ACL2026 视频理解方向17篇论文解读
description: >-
  17篇ACL2026的视频理解方向论文解读，涵盖多模态、压缩/编码、问答、推理、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "视频理解"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "压缩/编码"
  - "问答"
  - "推理"
  - "LLM"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📹 视频理解

**💬 ACL2026** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (17)](../../ICML2026/video_understanding/index.md) · [📷 CVPR2026 (83)](../../CVPR2026/video_understanding/index.md) · [🔬 ICLR2026 (22)](../../ICLR2026/video_understanding/index.md) · [🤖 AAAI2026 (33)](../../AAAI2026/video_understanding/index.md) · [🧠 NeurIPS2025 (59)](../../NeurIPS2025/video_understanding/index.md) · [📹 ICCV2025 (57)](../../ICCV2025/video_understanding/index.md)

🔥 **高频主题：** 多模态 ×2 · 压缩/编码 ×2

**[APB-V: Accelerating Long-Video Understanding via Sequence-Parallelism-aware Approximate Attention](apb-v_accelerating_long-video_understanding_via_sequence-parallelism-aware_appro.md)**

:   APB-V 用面向序列并行的近似注意力和系统级负载均衡加速长视频 LMM 推理，在保留完整视觉 embedding 的同时，在 64 帧 1440p 设置下相对 FlashAttn、ZigZagRing 和 APB 分别达到 12.72×、1.70× 和 1.18× 加速，且没有显著性能损失。

**[ArrowGEV: Grounding Events in Video via Learning the Arrow of Time](arrowgev_grounding_events_in_video_via_learning_the_arrow_of_time.md)**

:   提出 ArrowGEV，一个受物理学"时间之箭"启发的强化学习框架，通过区分时间敏感和时间不敏感事件来建模视频中的时间方向性，提升 VLM 的事件定位精度和时序理解能力。

**[Automated Knowledge Component Generation and Interpretable Knowledge Tracing in Coding Problems](automated_knowledge_component_generation_for_interpretable_knowledge_tracing_in_.md)**

:   这篇论文用 LLM 自动为开放式编程题生成和聚类 Knowledge Components，并提出 KCGen-KT 将学生在每个 KC 上的掌握度转成 soft token 输入 Llama 3，在 CodeWorkout 和 FalconCode 上同时提升正确率预测与学生代码生成。

**[Confidence Estimation for LLMs in Multi-turn Interactions](confidence_estimation_for_llms_in_multi-turn_interactions.md)**

:   首次系统研究多轮对话场景下的 LLM 置信度估计，提出两个核心准则（per-turn 校准 + 信息增加时单调性）、对应的 InfoECE 指标和 Kendall's $\tau$ 评估、Hinter-Guesser 数据集构造范式，并提出新颖的 P(SUFFICIENT) logit 探针——结果发现现有方法（verbalized / SC / P(TRUE)）在多轮场景中校准和单调性都很差，而 P(SUFFICIENT) 在 GUESS 上 InfoECE 降到 5.27（vs P(TRUE) 79.97）、$\tau$ 达 81.51，但任务远未解决。

**[CRAFT: Critic-Refined Adaptive Key-Frame Targeting for Multimodal Video Question Answering](craft_critic-refined_adaptive_key-frame_targeting_for_multimodal_video_question_.md)**

:   CRAFT 是一个面向新闻事件多视频问答的 claim-centric pipeline，它结合动态关键帧选择、ASR 转写、UNLI/MNLI/LLM critic 迭代修正和引用合并，在 MAGMaR-Test 上取得 0.739 macro average、0.810 reference recall 和 0.635 citation F1。

**[Distorted or Fabricated? A Survey on Hallucination in Video LLMs](distorted_or_fabricated_a_survey_on_hallucination_in_video_llms.md)**

:   本文首次对视频大语言模型（Vid-LLM）中的幻觉现象进行系统分类，提出"动态失真"（时空关系和引用一致性错误）和"内容捏造"（统计先验驱动和音视频冲突）的机制驱动分类体系，综述评估基准、缓解策略和根因分析。

**[DualFact: A Multimodal Fact Verification Framework for Procedural Video Understanding](dualfact_a_multimodal_fact_verification_framework_for_procedural_video_understan.md)**

:   作者把"做饭、家具制作"这类程序化视频字幕的事实评测拆成**双层事实**——conceptual facts（抽象角色，如 Action/Ingredient/Tool/Location）+ contextual facts（视频中可观察的 predicate–argument 关系，如 stir(soup, pot)），配套构建 YouCook3-Fact / CraftBench-Fact 两个标注隐式参数补全 (VIA) 与对比性事实的基准，并提出 MultiFactScore 用多模态/文本 NLI 在角色级别分别核查事实，进而把错误细分为 Hallucination / Saliency / Omission；实验发现 SOTA MLLM 字幕"流畅但事实残缺"，单看字幕会高估 Hallucination 一半左右，只有 video-grounded 评测才能区分 saliency 与真 hallucination。

**[GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents](gameplayqa_a_benchmarking_framework_for_decision-dense_pov-synced_multi-video_un.md)**

:   提出 GameplayQA，一个基于多人3D游戏视频的端到端基准框架，通过密集时间线标注（1.22标签/秒）和结构化干扰项分类学，系统评估多模态大模型在决策密集、多视角同步场景下的感知和推理能力，揭示前沿模型与人类表现仍有显著差距。

**[HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)**

:   本文提出 HERMES，基于对 MLLM 解码器层级注意力偏好的机制性分析，将 KV 缓存概念化为层级记忆框架（浅层=感觉记忆、中层=工作记忆、深层=长期记忆），实现免训练的高效流式视频理解，在减少 68% 视频 token 的条件下仍保持或提升准确率，TTFT 延迟仅 <30ms，比前 SOTA 快 10 倍。

**[NSF-SciFy: Mining the NSF Awards Database for Scientific Claims](nsf-scify_mining_the_nsf_awards_database_for_scientific_claims.md)**

:   NSF-SciFy 从 NSF 奖项摘要中抽取 2.8M 科学 claims 和 investigation proposals，构建了比现有科学 claim 数据集大几个数量级的资源，并展示了它能显著提升 claim / proposal 抽取模型。

**[Probing for Reading Times](probing_for_reading_times.md)**

:   本文探测语言模型各层表示预测阅读时间的能力，发现早期层表示在预测早期注视指标上优于surprisal，而surprisal在晚期指标上更优，最佳预测器因语言和指标而异。

**[Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding](response-g1_explicit_scene_graph_modeling_for_proactive_streaming_video_understa.md)**

:   Response-G1 用查询引导的在线场景图、历史场景图检索和带时间戳的触发提示，把流式视频中的视觉证据和用户查询的响应条件显式对齐，在无需微调的情况下显著提升 Video-LLM 判断“现在是否该回答”的能力。

**[Rethinking the Idiomaticity Decomposability Hypothesis: Evidence from Distributional Learning](rethinking_the_idiomaticity_decomposability_hypothesis_evidence_from_distributio.md)**

:   这篇论文用上下文化语言模型作为“受控的分布式学习者”重新检验 Idiom Decomposability Hypothesis，发现模型派生的可分解性只弱相关于人类判断，并且与句法灵活性呈小而稳定的负相关，说明习语行为更像是由分布经验、surprisal 和表征稳定过程共同塑造。

**[TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)**

:   本文提出 TemporalVLM，通过时间感知的片段编码器（重叠滑动 Video Q-Former + 融合模块）提取局部细粒度时间特征，再用 BiLSTM 聚合全局长程依赖，首次在 Video LLM 中引入 LSTM，在密集视频描述、时序定位、高光检测和动作分割四项任务上超越先前方法。

**[TRACE：基于证据定位的多视频事件理解与声明生成](trace_evidence_grounding-guided_multi-video_event_understanding_and_claim_genera.md)**

:   TRACE 通过"先定位后推理"的管道，先用 OCR 和目标检测构建文本可搜索的视频时间线，再用文本 LLM 进行查询条件的证据定位，最后由 LVLM 生成带引用的声明，在多视频事件理解任务上达到 SOTA，F1 从 0.705 提升到 0.811。

**[ViLL-E: Video LLM Embeddings for Retrieval](vill-e_video_llm_embeddings_for_retrieval.md)**

:   提出 ViLL-E，首个同时支持文本生成和 embedding 生成的 Video LLM 统一架构，通过三阶段生成-对比联合训练和自适应 KV-Former embedding head，在视频检索和时序定位上逼近专家模型，同时保持 VideoQA 竞争力。

**[VISTA: Verification In Sequential Turn-based Assessment](vista_verification_in_sequential_turn-based_assessment.md)**

:   VISTA 提出了一个基于声明级分解和顺序一致性追踪的多轮对话事实性评估框架，将不可验证内容细分为主观、矛盾、缺乏证据和弃权四类，在四个对话基准和八个 LLM 上显著优于 FActScore 和 LLM-as-Judge 基线。
