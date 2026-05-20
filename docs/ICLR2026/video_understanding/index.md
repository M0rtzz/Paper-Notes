---
title: >-
  ICLR2026 视频理解方向22篇论文解读
description: >-
  22篇ICLR2026的视频理解方向论文解读，涵盖 LLM、推理、目标跟踪等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "视频理解"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "推理"
  - "目标跟踪"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📹 视频理解

**🔬 ICLR2026** · **22** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (8)](../../ICML2026/video_understanding/index.md) · [💬 ACL2026 (11)](../../ACL2026/video_understanding/index.md) · [📷 CVPR2026 (77)](../../CVPR2026/video_understanding/index.md) · [🤖 AAAI2026 (33)](../../AAAI2026/video_understanding/index.md) · [🧠 NeurIPS2025 (59)](../../NeurIPS2025/video_understanding/index.md) · [📹 ICCV2025 (57)](../../ICCV2025/video_understanding/index.md)

🔥 **高频主题：** LLM ×5 · 推理 ×3 · 目标跟踪 ×3

**[AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference](adaem_an_adaptively_and_automated_extensible_measurement_of_llms_value_differenc.md)**

:   提出 AdAEM，一个自适应、自扩展的 LLM 价值观评估框架，通过信息论优化自动生成能最大化揭示不同 LLM 价值差异的测试问题，解决现有静态基准无法区分模型价值取向的"信息量不足"问题。

**[A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)**

:   提出 A.I.R.，一种无需训练的自适应-迭代-推理驱动帧选择框架，通过两阶段策略（GMM 自适应初始采样 + 迭代式 VLM 精细分析）解决 VideoQA 中轻量模型（CLIP）相似度不准确和 VLM 分析成本爆炸的双重困境，在最坏情况下也仅需分析 72 帧（vs 基线 128 帧），同时显著提升多个长视频 benchmark 性能。

**[AnveshanaAI: A Multimodal Platform for Adaptive AI/ML Education through Automated Question Generation and Interactive Assessment](anveshanaai_a_multimodal_platform_for_adaptive_aiml_education_through_automated_.md)**

:   提出 AnveshanaAI，一个基于 Bloom 认知分类学的自适应 AI/ML 教育平台，通过自动化题目生成（基于微调的 GPT-2）、语义相似度检测去重、XAI 可解释性技术和游戏化机制（积分/徽章/排行榜），实现了覆盖数据科学到多模态 AI 七大领域的个性化学习评估系统，实验表明微调后困惑度显著下降且学习者参与度明显提升。

**[Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss](coupling_experts_and_routers_in_mixture-of-experts_via_an_auxiliary_loss.md)**

:   提出 Expert-Router Coupling (ERC) Loss，一种轻量级辅助损失函数，通过将路由器参数视为聚类中心的代理 token 并约束专家对其激活范数，实现路由器决策与专家能力的紧密耦合，仅需 $n^2$ 次激活计算即可显著提升 MoE-LLM 性能。

**[Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading](decoding_open-ended_information_seeking_goals_from_eye_movements_in_reading.md)**

:   提出从阅读时眼动轨迹解码开放式信息检索目标的新任务，基于 OneStop 眼动数据集（360人、486问题、162段落），开发判别式和生成式多模态模型；RoBERTEye-Fixations 在三选一目标选择上达 49.3%（随机 33%），不同 critical span 达 70.9%；DalEye-Llama/GPT 在目标重建中也显著优于无眼动基线。

**[Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought](emergence_of_superposition_unveiling_the_training_dynamics_of_chain_of_continuou.md)**

:   从理论上分析了两层 Transformer 在有向图可达性问题上使用连续 Chain-of-Thought（Coconut）训练时的训练动力学，揭示了"叠加态"（superposition）机制如何自然涌现：index-matching logit 先增长后有界，从而在探索与利用之间取得平衡。

**[FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging](flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat.md)**

:   提出 FlashVID，一个免训练的视频大语言模型推理加速框架，通过树状时空 token 合并（TSTM）联合建模空间和时间冗余，仅保留 10% 的视觉 token 就能保持 LLaVA-OneVision 99.1% 的性能，并能将 Qwen2.5-VL 的输入帧数提升 10 倍。

**[FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)**

:   提出 FLoC，基于设施选址函数（facility location function）的视觉 token 压缩框架，通过子模优化在给定预算下快速选择兼具代表性和多样性的 token 子集，实现无训练、模型无关、查询无关的长视频理解 token 压缩。

**[From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning](from_vicious_to_virtuous_cycles_synergistic_representation_learning_for_unsuperv.md)**

:   发现 slot-based 目标中心学习中编码器（产生尖锐但有噪声的注意力图）与解码器（产生空间一致但模糊的重建掩码）之间的恶性循环，提出同步对比学习目标和 slot 正则化预热策略将其转化为良性循环，在 MOVi 和 YouTube-VIS 上大幅提升物体发现性能。

**[Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding](lets_split_up_zero-shot_classifier_edits_for_fine-grained_video_understanding.md)**

:   提出了"类别拆分"(Category Splitting)新任务，通过挖掘视频分类器权重中的潜在组合结构，在零样本条件下将粗粒度动作类别拆分为细粒度子类别，无需重训或额外数据。

**[Log Probability Tracking of LLM APIs](log_probability_tracking_of_llm_apis.md)**

:   提出 Logprob Tracking (LT) 方法，仅用单token输入和单token输出的log概率即可检测LLM API的微小变更（如单步微调），灵敏度比现有方法高2-3个数量级，成本低1000倍。

**[LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals](lumina_detecting_hallucinations_in_rag_system_with_context-knowledge_signals.md)**

:   提出 Lumina 框架，通过"上下文-知识信号"检测RAG系统中的幻觉：用MMD度量**外部上下文利用**程度，用跨层token预测演化度量**内部知识利用**程度，无需超参调优即可泛化。

**[Mamba-3: Improved Sequence Modeling using State Space Principles](mamba-3_improved_sequence_modeling_using_state_space_principles.md)**

:   从SSM视角提出三项核心改进：指数-梯形离散化、复值状态空间、多输入多输出(MIMO)公式化，在不增加解码延迟的前提下显著提升模型质量和状态追踪能力，推进性能-效率Pareto前沿。

**[Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs](map_the_flow_revealing_hidden_pathways_of_information_in_videollms.md)**

:   首次用机制可解释性工具（Attention Knockout + Logit Lens）系统逆向工程VideoLLM的时序推理过程，揭示出"早中层跨帧交互→中层视频-语言整合→中后层答案生成"的三阶段信息流蓝图，并证明仅保留42%注意力边即可几乎无损保持VideoQA性能。

**[NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks](nerve_nonlinear_eigenspectrum_dynamics_in_llm_feed-forward_networks.md)**

:   提出 NerVE，一个轻量级的特征谱分析框架，通过四个互补指标（频谱熵、参与比、特征值早期富集、JS 散度）系统揭示了 LLM 中 FFN 非线性如何重新注入方差、重塑特征谱，以及架构和优化器选择如何印刻独特的频谱签名。

**[Online Time Series Prediction Using Feature Adjustment](online_time_series_prediction_using_feature_adjustment.md)**

:   提出 ADAPT-Z（Automatic Delta Adjustment via Persistent Tracking in Z-space），将在线时序预测的适应目标从模型参数更新转移到特征空间修正，通过轻量 adapter 融合当前特征与历史梯度来应对多步预测中的延迟反馈问题，在13个数据集上一致超越现有在线学习方法。

**[Paper Copilot: Tracking the Evolution of Peer Review in AI Conferences](paper_copilot_tracking_the_evolution_of_peer_review_in_ai_conferences.md)**

:   构建 Paper Copilot——跨数十个 AI/ML 会议的同行评审持久数字档案与分析平台：通过 OpenReview API、网页抓取、社区贡献三源混合收集评审数据，实时归档评分时间快照（含 rebuttal 前后动态变化），揭示 ICLR 2025 年决策熵反常下降——评审体系从概率性分层转向近确定性分数驱动决策的结构性变化，并通过 LLM 驱动的作者-机构元数据提取支持人才轨迹追踪。

**[Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning](stabilizing_policy_gradients_for_sample-efficient_reinforcement_learning_in_llm_.md)**

:   提出 CAPO（Curvature-Aware Policy Optimization），通过在 LM head 最后一层建模二阶优化几何来预测并过滤会导致策略崩溃的 token 更新，在激进超参数（5× 学习率、1/12 batch size）下仍保持训练稳定，实现 MATH 上相较标准 GRPO 的 30× 样本效率提升。

**[The Expressive Limits of Diagonal SSMs for State-Tracking](the_expressive_limits_of_diagonal_ssms_for_state-tracking.md)**

:   建立了输入依赖复数对角（DCD）SSM 在群状态追踪任务上的完整表达能力刻画：单层不能追踪任何非阿贝尔群，$k$ 层能追踪群 $G$ 当且仅当 $G$ 存在长度为 $k$ 的子正规链且因子均为阿贝尔群——精确定义了深度对表达能力的严格提升，同时实验揭示表达能力与可学习性之间的显著 gap。

**[Towards Reliable Benchmarking: A Contamination Free, Controllable Evaluation Framework for Multi-step LLM Function Calling](towards_reliable_benchmarking_a_contamination_free_controllable_evaluation_frame.md)**

:   提出 FuncBenchGen 框架，通过将多步函数调用建模为 DAG 图遍历问题，实现无数据污染、可精细控制任务难度的 LLM 工具使用能力评估，并揭示了推理模型在长调用链和连接型干扰函数下的关键失败模式。

**[Video-KTR: Reinforcing Video Reasoning via Key Token Attribution](video-ktr_reinforcing_video_reasoning_via_key_token_attribution.md)**

:   提出 Video-KTR，一种模态感知的策略塑造框架，通过反事实分析识别视觉感知型、时序敏感型和高熵 Token 三类关键 Token，仅对这些 Token 执行选择性强化学习更新，在多个视频推理基准上达到 SOTA（Video-Holmes 42.7%，超越 GPT-4o）。

**[VideoNSA: Native Sparse Attention Scales Video Understanding](videonsa_native_sparse_attention_scales_video_understanding.md)**

:   本文提出 VideoNSA，将 Native Sparse Attention（NSA）引入视频语言模型，通过压缩、选择和滑动窗口三分支动态门控的混合稀疏注意力机制，在仅使用 3.6% 注意力预算的条件下实现 128K token 的视频理解，在长视频理解、时序推理和空间理解任务上全面超越 token 压缩和无训练稀疏注意力基线。
