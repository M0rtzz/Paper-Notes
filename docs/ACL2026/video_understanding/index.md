---
title: >-
  ACL2026 视频理解方向11篇论文解读
description: >-
  11篇ACL2026的视频理解方向论文解读，涵盖压缩/编码、多模态、推理、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "视频理解"
  - "论文解读"
  - "论文笔记"
  - "压缩/编码"
  - "多模态"
  - "推理"
  - "LLM"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📹 视频理解

**💬 ACL2026** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (8)](../../ICML2026/video_understanding/index.md) · [📷 CVPR2026 (77)](../../CVPR2026/video_understanding/index.md) · [🔬 ICLR2026 (22)](../../ICLR2026/video_understanding/index.md) · [🤖 AAAI2026 (33)](../../AAAI2026/video_understanding/index.md) · [🧠 NeurIPS2025 (59)](../../NeurIPS2025/video_understanding/index.md) · [📹 ICCV2025 (57)](../../ICCV2025/video_understanding/index.md)

🔥 **高频主题：** 压缩/编码 ×2

**[ArrowGEV: Grounding Events in Video via Learning the Arrow of Time](arrowgev_grounding_events_in_video_via_learning_the_arrow_of_time.md)**

:   提出 ArrowGEV，一个受物理学"时间之箭"启发的强化学习框架，通过区分时间敏感和时间不敏感事件来建模视频中的时间方向性，提升 VLM 的事件定位精度和时序理解能力。

**[Confidence Estimation for LLMs in Multi-turn Interactions](confidence_estimation_for_llms_in_multi-turn_interactions.md)**

:   首次系统研究多轮对话场景下的 LLM 置信度估计，提出两个核心准则（per-turn 校准 + 信息增加时单调性）、对应的 InfoECE 指标和 Kendall's $\tau$ 评估、Hinter-Guesser 数据集构造范式，并提出新颖的 P(SUFFICIENT) logit 探针——结果发现现有方法（verbalized / SC / P(TRUE)）在多轮场景中校准和单调性都很差，而 P(SUFFICIENT) 在 GUESS 上 InfoECE 降到 5.27（vs P(TRUE) 79.97）、$\tau$ 达 81.51，但任务远未解决。

**[Distorted or Fabricated? A Survey on Hallucination in Video LLMs](distorted_or_fabricated_a_survey_on_hallucination_in_video_llms.md)**

:   本文首次对视频大语言模型（Vid-LLM）中的幻觉现象进行系统分类，提出"动态失真"（时空关系和引用一致性错误）和"内容捏造"（统计先验驱动和音视频冲突）的机制驱动分类体系，综述评估基准、缓解策略和根因分析。

**[DualFact: A Multimodal Fact Verification Framework for Procedural Video Understanding](dualfact_a_multimodal_fact_verification_framework_for_procedural_video_understan.md)**

:   作者把"做饭、家具制作"这类程序化视频字幕的事实评测拆成**双层事实**——conceptual facts（抽象角色，如 Action/Ingredient/Tool/Location）+ contextual facts（视频中可观察的 predicate–argument 关系，如 stir(soup, pot)），配套构建 YouCook3-Fact / CraftBench-Fact 两个标注隐式参数补全 (VIA) 与对比性事实的基准，并提出 MultiFactScore 用多模态/文本 NLI 在角色级别分别核查事实，进而把错误细分为 Hallucination / Saliency / Omission；实验发现 SOTA MLLM 字幕"流畅但事实残缺"，单看字幕会高估 Hallucination 一半左右，只有 video-grounded 评测才能区分 saliency 与真 hallucination。

**[GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents](gameplayqa_a_benchmarking_framework_for_decision-dense_pov-synced_multi-video_un.md)**

:   提出 GameplayQA，一个基于多人3D游戏视频的端到端基准框架，通过密集时间线标注（1.22标签/秒）和结构化干扰项分类学，系统评估多模态大模型在决策密集、多视角同步场景下的感知和推理能力，揭示前沿模型与人类表现仍有显著差距。

**[HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)**

:   本文提出 HERMES，基于对 MLLM 解码器层级注意力偏好的机制性分析，将 KV 缓存概念化为层级记忆框架（浅层=感觉记忆、中层=工作记忆、深层=长期记忆），实现免训练的高效流式视频理解，在减少 68% 视频 token 的条件下仍保持或提升准确率，TTFT 延迟仅 <30ms，比前 SOTA 快 10 倍。

**[Probing for Reading Times](probing_for_reading_times.md)**

:   本文探测语言模型各层表示预测阅读时间的能力，发现早期层表示在预测早期注视指标上优于surprisal，而surprisal在晚期指标上更优，最佳预测器因语言和指标而异。

**[Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding](response-g1_explicit_scene_graph_modeling_for_proactive_streaming_video_understa.md)**

:   Response-G1 用查询引导的在线场景图、历史场景图检索和带时间戳的触发提示，把流式视频中的视觉证据和用户查询的响应条件显式对齐，在无需微调的情况下显著提升 Video-LLM 判断“现在是否该回答”的能力。

**[TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)**

:   本文提出 TemporalVLM，通过时间感知的片段编码器（重叠滑动 Video Q-Former + 融合模块）提取局部细粒度时间特征，再用 BiLSTM 聚合全局长程依赖，首次在 Video LLM 中引入 LSTM，在密集视频描述、时序定位、高光检测和动作分割四项任务上超越先前方法。

**[ViLL-E: Video LLM Embeddings for Retrieval](vill-e_video_llm_embeddings_for_retrieval.md)**

:   提出 ViLL-E，首个同时支持文本生成和 embedding 生成的 Video LLM 统一架构，通过三阶段生成-对比联合训练和自适应 KV-Former embedding head，在视频检索和时序定位上逼近专家模型，同时保持 VideoQA 竞争力。

**[VISTA: Verification In Sequential Turn-based Assessment](vista_verification_in_sequential_turn-based_assessment.md)**

:   VISTA 提出了一个基于声明级分解和顺序一致性追踪的多轮对话事实性评估框架，将不可验证内容细分为主观、矛盾、缺乏证据和弃权四类，在四个对话基准和八个 LLM 上显著优于 FActScore 和 LLM-as-Judge 基线。
