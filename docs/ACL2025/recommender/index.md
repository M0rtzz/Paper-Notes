---
title: >-
  ACL2025 推荐系统方向 7篇论文解读
description: >-
  7篇ACL2025 推荐系统方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎁 推荐系统

**💬 ACL2025** · **7** 篇论文解读

**[Beyond Single Labels: Improving Conversational Recommendation through LLM-Powered Data Augmentation](beyond_single_labels_improving_conversational_recommendation_through_llm-powered.md)**

:   针对对话推荐系统中的假阴性问题（用户可能喜欢的item被错误标记为负样本），提出基于LLM的数据增强框架，通过语义检索+相关性打分生成合成标签，再通过两阶段训练策略平衡语义相关性和协同信息。

**[CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems](cove_compressed_vocabulary_expansion_makes_better_llm-based_recommender_systems.md)**

:   提出 CoVE 框架，通过扩展 LLM 词表为每个物品分配唯一 token ID 和嵌入，将序列推荐任务转化为 next-token prediction，相比现有方法推荐准确率提升最高 62%，推理速度提升约 100 倍，并通过哈希嵌入压缩解决大规模场景的内存问题。

**[GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion](gram_generative_recommendation.md)**

:   提出 GRAM 生成式推荐框架，通过**语义到词汇翻译**将隐式物品层次/协同关系编码到 LLM 词汇空间，并用**多粒度迟融合**独立编码不同粒度提示再在解码端融合，在四个基准上 Recall@5 提升 11.5–16.0%、NDCG@5 提升 5.3–13.6%。

**[KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models](kerl_knowledge-enhanced_personalized_recipe_recommendation_using_large_language_.md)**

:   提出 KERL 统一食品推荐系统，结合 FoodKG 知识图谱和 Phi-3-mini 多 LoRA 微调，实现个性化食谱推荐（F1=0.973）、食谱生成和微量营养素估算三个功能，大幅超越基线 LLM 和传统嵌入方法。

**[LOTUS: A Leaderboard for Detailed Image Captioning from Quality to Societal Bias and User Preferences](lotus_a_leaderboard_for_detailed_image_captioning_from_quality_to_societal_bias_.md)**

:   提出 LOTUS 排行榜，从描述质量（对齐性、描述性、语言复杂度）、副作用（幻觉、有害性）和社会偏见（性别、肤色）三个维度统一评估大型视觉语言模型的详细图像描述能力，并支持基于用户偏好的定制化评估。

**[MIRA: Empowering One-Touch AI Services on Smartphones with MLLM-based Instruction Recommendation](mira_empowering_one-touch_ai_services_on_smartphones_with_mllm-based_instruction.md)**

:   提出 MIRA 框架，通过结构化推理、模板增强推理和前缀树约束解码，让用户在智能手机上长按文本或图片即可获得上下文相关的 AI 服务指令推荐，在 7B 模型上超越 GPT-4V（F1: 0.9121 vs 0.879），token 使用量仅为 1/7。

**[RecLM: Recommendation Instruction Tuning](reclm_recommendation_instruction_tuning.md)**

:   提出 RecLM，一个模型无关的推荐指令微调框架，通过两轮对话式指令微调将协同过滤信号注入 LLM 生成的用户/商品画像，再用 RLHF（PPO）精炼画像质量，在 MIND/Netflix/工业数据集上作为即插即用组件为 BiasMF/NCF/LightGCN/SGL/SimGCL 一致带来提升，尤其在冷启动场景效果显著。
