---
title: >-
  ACL2026 LLM 安全方向21篇论文解读
description: >-
  21篇ACL2026的 LLM 安全方向论文解读，涵盖 LLM、水印/隐写、文本摘要、压缩/编码、语音、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ACL2026"
  - "LLM 安全"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "水印/隐写"
  - "文本摘要"
  - "压缩/编码"
  - "语音"
  - "对抗鲁棒"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM 安全

**💬 ACL2026** · **21** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (16)](../../CVPR2026/llm_safety/index.md) · [🔬 ICLR2026 (36)](../../ICLR2026/llm_safety/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/llm_safety/index.md) · [🧠 NeurIPS2025 (59)](../../NeurIPS2025/llm_safety/index.md) · [📹 ICCV2025 (8)](../../ICCV2025/llm_safety/index.md) · [🧪 ICML2025 (32)](../../ICML2025/llm_safety/index.md)

🔥 **高频主题：** LLM ×8 · 水印/隐写 ×2

**[Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)**

:   提出自适应文本匿名化框架，通过进化式提示优化自动为LLM发现任务特定的匿名化指令，在多个隐私-效用权衡场景中超越手工设计的策略，且可在开源模型上运行。

**[AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation](agsc_adaptive_granularity_and_semantic_clustering_for_uncertainty_quantification.md)**

:   AGSC 提出了一个针对长文本生成的不确定性量化框架，通过 NLI 中立概率触发自适应粒度分解（减少 60% 推理时间），并使用 GMM 软聚类捕捉潜在语义主题进行主题感知的加权聚合，在 BIO 和 LongFact 基准上达到 SOTA 的事实性相关性。

**[Beyond End-to-End: Dynamic Chain Optimization for Private LLM Adaptation on the Edge](beyond_end-to-end_dynamic_chain_optimization_for_private_llm_adaptation_on_the_e.md)**

:   提出 ChainFed，一种打破内存墙的链式联邦微调范式，通过逐层顺序训练-冻结适配器使资源受限边缘设备也能参与 LLM 微调，结合动态层协调、全局感知优化和功能导向自适应三项技术，平均准确率提升最高 46.46%。

**[Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors](compiling_activation_steering_into_weights_via_null-space_constraints_for_stealt.md)**

:   本文提出 STEEREDIT，将动态激活转向编译为静态权重修改的后门注入框架，通过提取顺从方向并利用零空间约束确保仅在触发词存在时激活，在多个安全对齐 LLM 上实现高攻击成功率同时保持非触发场景下的安全性和通用性。

**[De-Anonymization at Scale via Tournament-Style Attribution](de-anonymization_at_scale_via_tournament-style_attribution.md)**

:   本文提出 DAS（De-Anonymization at Scale），一种基于 LLM 的大规模作者去匿名化方法，采用锦标赛式淘汰策略+密集检索预过滤+多轮投票聚合，可在数万候选文本中进行作者匹配，揭示了 LLM 对匿名平台（如双盲评审）的隐私威胁。

**[DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)**

:   本文提出 DUET，一个结合直接代码执行和 LLM 伪代码执行的双路框架，通过功能多数投票融合两种互补的执行路径——前者在代码正确时可靠但受实现错误影响，后者绕过实现细节但可能产生执行幻觉——在 LiveCodeBench 测试输出预测上提升 Pass@1 13.6 个百分点。

**[Enhancing Hallucination Detection via Future Context](enhancing_hallucination_detection_via_future_context.md)**

:   本文提出利用采样生成的"未来上下文"（后续句子）来增强黑盒场景下的幻觉检测，利用幻觉一旦出现就倾向于持续传播的"滚雪球效应"，在 SelfCheckGPT 和 SC 等多种采样方法上一致提升检测性能。

**[FACTS: Table Summarization via Offline Template Generation with Agentic Workflows](facts_table_summarization_via_offline_template_generation_with_agentic_workflows.md)**

:   本文提出 FACTS（Fast, Accurate, and Privacy-Compliant Table Summarization），通过三阶段 Agentic 工作流自动生成可复用的离线模板（SQL 查询 + Jinja2 模板），实现快速、准确、隐私合规的查询聚焦表格摘要，在 FeTaQA、QTSumm 和 QFMTS 三个基准上全面超越基线。

**[Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens](forget_what_matters_keep_the_rest_selective_unlearning_of_informative_tokens.md)**

:   提出 Entropy-guided Token Weighting (ETW)，利用预测分布的熵值作为 token 信息量的代理指标，选择性地对信息性 token 施加更强的遗忘惩罚，在有效遗忘目标知识的同时更好地保持模型通用能力。

**[Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages](indic-codecfake_meets_satyam_towards_detecting_neural_audio_codec_synthesized_sp.md)**

:   本文构建了首个多印度语言的 CodecFake 检测基准 ICF，并提出 SATYAM——一个双曲音频大语言模型，通过在双曲空间中用 Bhattacharyya 距离对齐语义和副语言表示再与提示对齐，仅训练 3.75M 参数即达到 98.32% 的检测准确率。

**[Jailbreaking Large Language Models with Morality Attacks](jailbreaking_large_language_models_with_morality_attacks.md)**

:   本文构建10.3K道德攻击数据集（价值模糊+价值冲突），通过四种对抗策略操纵LLM道德判断，发现LLM和guardrail模型对道德攻击极度脆弱，且更大模型反而更容易被攻破。

**[KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates](koco_conditioning_language_model_pre-training_on_knowledge_coordinates.md)**

:   提出知识坐标条件化预训练（KoCo），将每个文档映射为三维语义坐标（来源、内容、稳定性），作为文本前缀注入预训练，使模型获得显式的上下文感知能力，在 10 个下游任务上提升性能、加速收敛约 30%，并有效缓解幻觉。

**[Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning](maximizing_local_entropy_where_it_matters_prefix-aware_localized_llm_unlearning.md)**

:   本文提出 PALU（Prefix-Aware Localized Unlearning），从时间和词表两个维度实现局部化的熵最大化遗忘：在时间维度仅对敏感前缀 token 施加遗忘目标，在词表维度仅对 top-K logits 进行平坦化，以最小的参数扰动实现高效遗忘并保持模型通用能力。

**[MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs](meashalu_mitigation_of_scientific_measurement_hallucinations_for_large_language_.md)**

:   本文提出MeasHalu框架，通过细粒度测量幻觉分类法和两阶段优化（推理感知SFT+幻觉靶向GRPO奖励）缓解LLM在科学测量抽取中的幻觉，在MeasEval上显著超越基线。

**[Synthia: Scalable Grounded Persona Generation from Social Media Data](synthia_scalable_grounded_persona_generation_from_social_media_data.md)**

:   提出 Synthia 框架，基于真实社交媒体帖子（Bluesky）生成有根据的 LLM 人格叙事，在社会调查对齐度上比 SOTA 提升最高 11.6%，同时使用更小的模型，并保留社交网络拓扑结构支持网络感知分析。

**[Topic-Based Watermarks for Large Language Models](topic-based_watermarks_for_large_language_models.md)**

:   本文提出基于主题的轻量水印方案 TBW，将词表按语义主题聚类为"绿色列表"（而非随机分区），根据输入提示选择语义对齐的主题列表进行 logit 偏置，在保持与无水印文本相当的困惑度的同时，显著提升了对释义和词汇扰动攻击的鲁棒性。

**[Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement](toward_consistent_world_models_with_multi-token_prediction_and_latent_semantic_e.md)**

:   从理论上分析了多 Token 预测（MTP）如何通过梯度耦合机制诱导表示收缩性从而促进信念状态的涌现，但同时揭示了 MTP 的"结构性幻觉"问题（隐空间中的非法捷径），并提出 LSE-MTP 框架通过隐一致性损失和语义锚定损失将预测锚定到真实隐状态轨迹，在合成图和真实曼哈顿出租车导航上显著改善路径合法性和鲁棒性。

**[Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations](two_pathways_to_truthfulness_on_the_intrinsic_encoding_of_llm_hallucinations.md)**

:   本文发现 LLM 内部编码真实性信号存在两条不同的信息通路：Question-Anchored（依赖问题到回答的信息流）和 Answer-Anchored（从生成答案本身提取自包含证据），两者与知识边界紧密关联，并据此提出 Mixture-of-Probes 和 Pathway Reweighting 两种通路感知的幻觉检测方法，AUC 提升达 10%。

**[Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text](who_gets_which_message_auditing_demographic_bias_in_llm-generated_targeted_text.md)**

:   本文首次系统分析 LLM 在人口统计条件下生成定向消息时的偏见行为，提出 Persuasion Bias Index (PBI) 指标，发现 GPT-4o/Llama/Mistral 在气候传播中对男性和年轻人使用更强势的说服策略，且上下文提示会系统性地放大这些差异。

**[Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models](why_supervised_fine-tuning_fails_to_learn_a_systematic_study_of_incomplete_learn.md)**

:   本文首次系统研究了 SFT 中的"不完全学习现象"（ILP）——即模型收敛后仍无法正确复现部分训练数据，识别了五种反复出现的原因（知识缺失、知识冲突、数据内部矛盾、左侧遗忘、不充分优化），并提出诊断框架和针对性缓解策略。

**[XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts](xmark_reliable_multi-bit_watermarking_for_llm-generated_texts.md)**

:   提出 XMark，一种基于 Leave-one-Shard-out（LoSo）策略和 evergreen list 的多比特文本水印方法，通过跨多个词表排列的绿色列表交集和约束 token-shard 映射矩阵，在保持文本质量的同时显著提升了有限 token 条件下的解码准确率。
