---
title: >-
  ACL2026 视频理解方向11篇论文解读
description: >-
  11篇ACL2026的视频理解方向论文解读，涵盖压缩/编码、Agent、目标跟踪、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📹 视频理解

**💬 ACL2026** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (77)](../../CVPR2026/video_understanding/index.md) · [🔬 ICLR2026 (24)](../../ICLR2026/video_understanding/index.md) · [🤖 AAAI2026 (33)](../../AAAI2026/video_understanding/index.md) · [🧠 NeurIPS2025 (58)](../../NeurIPS2025/video_understanding/index.md) · [📹 ICCV2025 (57)](../../ICCV2025/video_understanding/index.md) · [🧪 ICML2025 (7)](../../ICML2025/video_understanding/index.md)

**[ArrowGEV: Grounding Events in Video via Learning the Arrow of Time](arrowgev_grounding_events_in_video_via_learning_the_arrow_of_time.md)**

:   提出 ArrowGEV，一个受物理学"时间之箭"启发的强化学习框架，通过区分时间敏感和时间不敏感事件来建模视频中的时间方向性，提升 VLM 的事件定位精度和时序理解能力。

**[Distorted or Fabricated? A Survey on Hallucination in Video LLMs](distorted_or_fabricated_a_survey_on_hallucination_in_video_llms.md)**

:   本文首次对视频大语言模型（Vid-LLM）中的幻觉现象进行系统分类，提出"动态失真"（时空关系和引用一致性错误）和"内容捏造"（统计先验驱动和音视频冲突）的机制驱动分类体系，综述评估基准、缓解策略和根因分析。

**[GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents](gameplayqa_a_benchmarking_framework_for_decision-dense_pov-synced_multi-video_un.md)**

:   提出 GameplayQA，一个基于多人3D游戏视频的端到端基准框架，通过密集时间线标注（1.22标签/秒）和结构化干扰项分类学，系统评估多模态大模型在决策密集、多视角同步场景下的感知和推理能力，揭示前沿模型与人类表现仍有显著差距。

**[HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)**

:   本文提出 HERMES，基于对 MLLM 解码器层级注意力偏好的机制性分析，将 KV 缓存概念化为层级记忆框架（浅层=感觉记忆、中层=工作记忆、深层=长期记忆），实现免训练的高效流式视频理解，在减少 68% 视频 token 的条件下仍保持或提升准确率，TTFT 延迟仅 <30ms，比前 SOTA 快 10 倍。

**[Preference Estimation via Opponent Modeling in Multi-Agent Negotiation](preference_estimation_via_opponent_modeling_in_multi-agent_negotiation.md)**

:   提出将 LLM 提取的自然语言偏好信号与贝叶斯对手建模框架结合的偏好估计方法，在多方多议题谈判中通过语言似然函数融合定性线索和定量出价信息，将完全达成协议率从 37% 提升至 62%。

**[Probing for Reading Times](probing_for_reading_times.md)**

:   本文探测语言模型各层表示预测阅读时间的能力，发现早期层表示在预测早期注视指标上优于surprisal，而surprisal在晚期指标上更优，最佳预测器因语言和指标而异。

**[RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora](rare_redundancy-aware_retrieval_evaluation_framework_for_high-similarity_corpora.md)**

:   本文提出 RARE 框架，通过将文档分解为原子事实来追踪跨文档冗余，并设计 CRRF（基于独立准则排序的倒数排名融合）稳定 LLM 多准则判断，在金融/法律/专利等高冗余企业语料上构建了 RedQA 基准，揭示主流检索器在 4-hop 高重叠设置下 PerfRecall@10 从 66.4% 暴跌至 5.0-27.9%。

**[Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for DLMs](saber_an_efficient_sampling_with_adaptive_acceleration_and_backtracking_enhanced.md)**

:   本文提出 Saber，一个面向扩散语言模型（DLM）的免训练采样算法，通过自适应加速（根据已建立的上下文动态调整并行解码量）和回溯增强重遮蔽（撤销被新上下文证伪的 token）两种策略，在代码生成上平均提升 Pass@1 1.9% 的同时实现 251.4% 的推理加速。

**[VC-Inspector: Advancing Reference-free Evaluation of Video Captions with Factual Analysis](vc-inspector_advancing_reference-free_evaluation_of_video_captions_with_factual_.md)**

:   本文提出 VC-Inspector，一个基于开源轻量级多模态模型（Qwen2.5-VL 3B/7B）的无参考视频字幕评估指标，通过可控事实错误合成流水线生成训练数据，在 VATEX-Eval 上达到 $\tau_b$=42.58 的人类判断相关性，超越依赖 GPT-4o 的 G-VEval（$\tau_b$=39.40），且在幻觉检测基准上达到 99.6% 准确率。

**[ViLL-E: Video LLM Embeddings for Retrieval](vill-e_video_llm_embeddings_for_retrieval.md)**

:   提出 ViLL-E，首个同时支持文本生成和 embedding 生成的 Video LLM 统一架构，通过三阶段生成-对比联合训练和自适应 KV-Former embedding head，在视频检索和时序定位上逼近专家模型，同时保持 VideoQA 竞争力。

**[VISTA: Verification In Sequential Turn-based Assessment](vista_verification_in_sequential_turn-based_assessment.md)**

:   VISTA 提出了一个基于声明级分解和顺序一致性追踪的多轮对话事实性评估框架，将不可验证内容细分为主观、矛盾、缺乏证据和弃权四类，在四个对话基准和八个 LLM 上显著优于 FActScore 和 LLM-as-Judge 基线。
