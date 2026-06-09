---
title: >-
  ICLR2026 推荐系统方向9篇论文解读
description: >-
  9篇ICLR2026的推荐系统方向论文解读，涵盖推荐系统、LLM、对抗鲁棒、个性化生成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "推荐系统"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对抗鲁棒"
  - "个性化生成"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎁 推荐系统

**🔬 ICLR2026** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (11)](../../ICML2026/recommender/index.md) · [💬 ACL2026 (21)](../../ACL2026/recommender/index.md) · [🤖 AAAI2026 (27)](../../AAAI2026/recommender/index.md) · [🧠 NeurIPS2025 (24)](../../NeurIPS2025/recommender/index.md) · [🧪 ICML2025 (17)](../../ICML2025/recommender/index.md) · [💬 ACL2025 (7)](../../ACL2025/recommender/index.md)

🔥 **高频主题：** 推荐系统 ×3 · LLM ×3

**[C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems](c2al_cohort-contrastive_auxiliary_learning_for_large-scale_recommendation_system.md)**

:   提出 C2AL（Cohort-Contrastive Auxiliary Learning），通过数据驱动地发现分布差异最大的用户群体对，构建对比性辅助二分类任务正则化共享编码器，使 FM 注意力权重从稀疏变为稠密，缓解大规模推荐系统中少数群体的表征偏差，在 Meta 6 个生产模型（数十亿数据点）上验证有效。

**[CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation](collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco.md)**

:   观察到序列推荐中不同用户的 KV cache 具有显著跨用户相似性（协同信号），提出 CollectiveKV 将 KV 分解为低维用户特有部分和从全局 KV 池检索的高维共享部分，实现 0.8% 的压缩率且性能不降。

**[From Evaluation to Defense: Advancing Safety in Video Large Language Models](from_evaluation_to_defense_advancing_safety_in_video_large_language_models.md)**

:   构建 VideoSafetyEval（11.4k 视频-查询对覆盖 19 种风险类别）揭示视频模态使安全性能下降 34.2%，提出 VideoSafety-R1 三阶段框架（报警 Token+SFT+Safety-guided GRPO）在 VSE-HH 上提升 71.1% 防御成功率。

**[GoalRank: Group-Relative Optimization for a Large Ranking Model](goalrank_group-relative_optimization_for_a_large_ranking_model.md)**

:   理论证明任意 Multi-Generator-Evaluator 排序系统都存在一个更大的 generator-only 模型以更小的误差逼近最优策略且满足 scaling law，据此提出 GoalRank——用 reward model 构建 group-relative 参考策略来训练大型 generator-only 排序模型，在线 A/B 测试中显著优于 SOTA。

**[In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations](in_agents_we_trust_but_who_do_agents_trust_latent_source_preferences_steer_llm_g.md)**

:   通过对来自6家提供商的12个LLM在新闻、学术、电商三大领域的大规模控制实验，揭示了LLM存在系统性的**隐式信息源偏好**（latent source preferences）——当内容语义完全相同时，仅更换来源标签就能显著改变模型的信息选择行为，且这种偏好无法通过提示工程消除。

**[ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)**

:   提出ProPerSim模拟框架，构建基于大五人格的32种用户persona在Smallville家庭环境中的日常行为模拟，AI助手通过每2.5分钟的主动推荐决策和DPO偏好学习，在14天模拟中将用户满意度从2.2/4提升至3.3/4，首次验证了主动性+个性化统一的可行性。

**[RAE: A Neural Network Dimensionality Reduction Method for Nearest Neighbors Preservation in Vector Search](rae_a_neural_network_dimensionality_reduction_method_for_nearest_neighbors_prese.md)**

:   提出 RAE（Regularized Auto-Encoder），通过线性自编码器 + Frobenius 范数正则化实现降维，理论证明正则化系数 $\lambda$ 通过 Rayleigh 商性质约束编码器矩阵的条件数 $\kappa(W)$，从而保证范数失真率有界、k-NN 结构被保持。在 4 个数据集上一致优于 PCA/UMAP/MDS/ISOMAP，余弦距离下比 PCA 至少高 12%，且训练仅需 8 秒、推理毫秒级。

**[Search Arena: Analyzing Search-Augmented LLMs](search_arena_analyzing_search-augmented_llms.md)**

:   构建 Search Arena——首个大规模搜索增强 LLM 人类偏好数据集（24069 对话 + 12652 偏好投票，71 种语言），发现用户偏好受引用数量影响（即使引用不支持声明），社区驱动平台比 Wikipedia 更受偏好，搜索增强不降低通用聊天性能但通用 LLM 在搜索场景显著退化。

**[Token-Efficient Item Representation via Images for LLM Recommender Systems](token-efficient_item_representation_via_images_for_llm_recommender_systems.md)**

:   提出 I-LLMRec，利用商品图像替代冗长文本描述来表示推荐系统中的物品语义，通过 RISA 对齐模块和 RERI 检索模块，在仅用单个token表示物品的同时保留丰富语义，推理速度提升约2.93倍且推荐性能超越文本描述方法。
