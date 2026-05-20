---
title: >-
  AAAI2026 预训练方向5篇论文解读
description: >-
  5篇AAAI2026的预训练方向论文解读，涵盖 LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "预训练"
  - "论文解读"
  - "论文笔记"
  - "LLM"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📚 预训练

**🤖 AAAI2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (17)](../../ICML2026/llm_pretraining/index.md) · [💬 ACL2026 (10)](../../ACL2026/llm_pretraining/index.md) · [📷 CVPR2026 (8)](../../CVPR2026/llm_pretraining/index.md) · [🔬 ICLR2026 (26)](../../ICLR2026/llm_pretraining/index.md) · [🧠 NeurIPS2025 (47)](../../NeurIPS2025/llm_pretraining/index.md) · [📹 ICCV2025 (9)](../../ICCV2025/llm_pretraining/index.md)

**[ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences](elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)**

:   ELSPR 将 LLM 评估器的成对偏好建模为锦标赛图，通过强连通分量 (SCC) 识别非传递偏好，提出归一化有向图结构熵指标，并基于图重构过滤有问题的训练数据——过滤后的评估器非传递性降低 13.8%、结构熵降低 0.088，且丢弃数据的人类一致性仅 34.4%（vs 保留数据 52.6%）。

**[Learning Procedural-aware Video Representations through State-Grounded Hierarchy Unfolding](learning_procedural-aware_video_representations_through_state-grounded_hierarchy.md)**

:   提出 Task-Step-State（TSS）三层语义框架，在传统的任务-步骤层次中引入"状态"作为视觉锚定层，并设计渐进式预训练策略（Task→Step→State→Step→Task）逐步展开 TSS 层次，在 COIN 和 CrossTask 数据集上的任务识别、步骤识别和步骤预测任务上全面超越 SOTA。

**[Learning Time in Static Classifiers](learning_time_in_static_classifiers.md)**

:   提出 Support-Exemplar-Query (SEQ) 学习框架，通过损失函数设计（而非架构修改）为标准前馈分类器注入时序推理能力，利用软DTW将预测序列与类别时序原型对齐，在细粒度图像分类和视频异常检测上均取得提升。

**[No-Regret Strategy Solving in Imperfect-Information Games via Pre-Trained Embedding](no-regret_strategy_solving_in_imperfect-information_games_via_pre-trained_embedd.md)**

:   提出 Embedding CFR 算法，将不完美信息博弈中的信息集映射到连续低维嵌入空间（而非离散聚类），在相同空间开销下实现更快的可利用性收敛和更高质量的策略求解。

**[PrefixGPT: Prefix Adder Optimization by a Generative Pre-trained Transformer](prefixgpt_prefix_adder_optimization_by_a_generative_pre-trained_transformer.md)**

:   提出PrefixGPT，将前缀加法器优化建模为序列生成问题，通过定制的GPT模型预训练学习设计规则后用RL微调生成优化设计，在面积-延迟乘积(ADP)上取得SOTA且对初始化不敏感。
