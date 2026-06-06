---
title: >-
  [论文解读] Laser: Bi-Tuning with Collaborative Information for Controllable LLM-Based Sequential Recommendation
description: >-
  [ACL 2025][推荐系统][序列推荐] 本文提出Laser框架，通过在LLM输入的前缀和后缀分别插入可训练虚拟token（Bi-Tuning），将用户-物品协同信息注入冻结的LLM，并设计基于MoE的M-Former来捕获不同类型用户的差异化特征，实现参数高效的序列推荐。
tags:
  - "ACL 2025"
  - "推荐系统"
  - "序列推荐"
  - "大语言模型"
  - "双端调优"
  - "协同信息"
  - "参数高效"
---

# Laser: Bi-Tuning with Collaborative Information for Controllable LLM-Based Sequential Recommendation

**会议**: ACL 2025  
**arXiv**: [2409.01605](https://arxiv.org/abs/2409.01605)  
**代码**: 无  
**领域**: 推荐系统  
**关键词**: 序列推荐、大语言模型、双端调优、协同信息、参数高效

## 一句话总结
本文提出Laser框架，通过在LLM输入的前缀和后缀分别插入可训练虚拟token（Bi-Tuning），将用户-物品协同信息注入冻结的LLM，并设计基于MoE的M-Former来捕获不同类型用户的差异化特征，实现参数高效的序列推荐。

## 研究背景与动机

**领域现状**：序列推荐系统通过分析用户的历史交互序列来预测下一个可能感兴趣的物品。近年来，利用LLM的语义理解能力来增强序列推荐成为热点，典型方法包括将物品标题作为LLM输入、用LLM编码物品语义等。

**现有痛点**：现有LLM-based推荐方法有两个主要问题：（1）大多数方法需要全量或大规模参数微调，资源消耗巨大；（2）虽然LLM擅长编码文本语义，但序列推荐的核心信号——用户-物品协同信息（collaborative information）——无法直接通过文本传递给LLM。现有工作简单地将物品ID或描述转为文本，忽略了协同过滤信号中蕴含的用户行为模式。

**核心矛盾**：LLM擅长理解语义但不擅长建模协同关系；传统推荐模型擅长协同过滤但语义理解不足。如何在参数高效的前提下将两者的优势结合？

**本文目标**：设计一个参数高效的框架，既能利用LLM的语义能力，又能注入传统推荐模型的协同信息。

**切入角度**：受prompt tuning启发，通过在输入两端插入可训练token来调整LLM行为——前缀负责注入协同信息并适配任务，后缀负责将LLM的语言空间输出转换到推荐空间。

**核心 idea**：前缀+后缀的Bi-Tuning范式 + MoE-based查询变换器（M-Former）实现用户差异化的协同信息注入。

## 方法详解

### 整体框架
Laser框架包含三个模块：（1）一个冻结的ID-based序列推荐模型（如SASRec），提供协同过滤信号；（2）一个冻结的LLM，作为语义编码器；（3）可训练的Bi-Tuning模块——前缀端的M-Former和后缀端的投影层。用户历史交互序列首先通过文本化输入LLM，M-Former从冻结的协同信息中提取查询，注入前缀位置；后缀token将LLM输出映射到推荐空间，与候选物品embedding计算匹配分数。

### 关键设计

1. **Bi-Tuning策略（前缀+后缀双端调优）**:

    - 功能：在冻结LLM参数的前提下实现任务适配
    - 核心思路：前缀包含 $N_p$ 个可训练虚拟token，插入输入序列头部，作用类似prefix-tuning，引导LLM关注推荐相关信息。后缀包含 $N_s$ 个可训练虚拟token，附加在输入末尾，其对应的LLM隐藏层输出经过一个线性投影层后作为用户表示。实际训练时只优化前缀/后缀token、M-Former和投影层的参数，LLM完全冻结。
    - 设计动机：前缀负责"适配输入"（将推荐任务信息注入LLM），后缀负责"适配输出"（将语言空间表示转换为推荐空间表示），两端协同实现最小化参数的任务适配。

2. **M-Former（MoE-based Querying Transformer）**:

    - 功能：为不同类型用户生成差异化的协同信息前缀
    - 核心思路：M-Former基于Querying Transformer架构（类似Q-Former），维护一组可学习的查询token，通过交叉注意力从冻结的序列推荐模型输出中提取协同信息。关键创新在于使用MoE（Mixture of Experts）替代标准FFN，每个expert对应一种用户行为模式。对于每个用户，门控网络根据其历史序列特征选择性激活不同的expert组合，实现用户差异化的信息提取。
    - 设计动机：不同用户（活跃/非活跃、偏好集中/分散）对协同信息的需求不同，单一的查询变换难以适配所有用户类型。MoE结构以较小的计算开销实现了多样化的信息提取策略。

3. **协同信息桥接**:

    - 功能：将传统推荐模型的协同过滤表示注入LLM
    - 核心思路：使用已训练好的SASRec等序列推荐模型作为"协同编码器"，冻结其参数。M-Former通过交叉注意力机制来"阅读"SASRec输出的物品序列表示，提取与当前推荐任务相关的协同信号，转化为前缀token注入LLM。这样LLM在处理文本化的物品描述时，同时拥有语义和协同两路信息。
    - 设计动机：ID-based模型和LLM各有所长，通过桥接机制让两者互补，避免了从头训练一个兼具两种能力的大模型。

### 损失函数 / 训练策略
采用softmax交叉熵损失进行下一项预测训练。训练时冻结LLM和SASRec参数，只优化M-Former、前缀/后缀token和投影层。

## 实验关键数据

### 主实验

| 数据集 | 指标 | Laser | SASRec | P5 | TALLRec | 提升(vs best baseline) |
|--------|------|-------|--------|-----|---------|----------------------|
| Beauty | HR@10 | 5.82 | 4.93 | 4.21 | 5.15 | +13.0% |
| Beauty | NDCG@10 | 3.21 | 2.68 | 2.25 | 2.81 | +14.2% |
| Sports | HR@10 | 4.15 | 3.52 | 3.10 | 3.73 | +11.3% |
| Sports | NDCG@10 | 2.18 | 1.85 | 1.58 | 1.92 | +13.5% |
| Toys | HR@10 | 6.47 | 5.51 | 4.89 | 5.82 | +11.2% |

### 消融实验

| 配置 | HR@10 | NDCG@10 | 说明 |
|------|-------|---------|------|
| Laser (Full) | 5.82 | 3.21 | 完整模型 |
| w/o M-Former (random prefix) | 4.91 | 2.62 | 去掉协同信息注入，性能大幅下降 |
| w/o MoE (single expert) | 5.43 | 2.95 | 去掉MoE后下降，说明用户差异化建模有效 |
| w/o suffix (mean pooling) | 5.21 | 2.78 | 后缀对输出空间转换很重要 |
| w/o prefix (suffix only) | 5.05 | 2.71 | 仅用后缀，缺少协同信息 |
| Full fine-tuning LLM | 5.68 | 3.12 | 全参数调优反而不如Bi-Tuning |

### 关键发现
- M-Former贡献最大（去掉后HR@10下降15.6%），说明协同信息注入是核心价值
- MoE相比单expert提升7.2%，验证了用户差异化建模的必要性
- 前缀和后缀都不可或缺，双端调优的性能远超单端
- 有趣的是全细调LLM反而不如Bi-Tuning，可能因为过拟合或破坏了LLM预训练知识

## 亮点与洞察
- Bi-Tuning的前缀/后缀分工非常清晰优雅——前缀负责输入端适配（注入协同信息），后缀负责输出端适配（空间转换），是一种值得推广的LLM适配范式。
- 用冻结的传统推荐模型作为"协同信息提供者"的设计很实用，避免了从头在LLM中学习协同关系的巨大成本。
- M-Former中的MoE设计体现了"不同用户需要不同程度/类型的协同信息"的洞察，这在推荐系统中是很自然但之前被忽视的点。

## 局限与展望
- 依赖预训练好的ID-based推荐模型提供协同信息，其质量直接影响Laser的性能上限
- 未探讨冷启动场景——新用户或新物品几乎没有协同信息可供M-Former提取
- MoE中expert数量的选择对性能的影响未充分分析
- 可以考虑将Bi-Tuning范式扩展到对话推荐、多模态推荐等场景

## 相关工作与启发
- **vs P5/TALLRec**: 这些方法将推荐转化为纯文本生成任务，丢失了协同信号；Laser通过M-Former显式注入协同信息
- **vs SASRec**: 传统序列推荐无法利用语义信息，Laser借助LLM补足了语义理解能力
- **vs LoRA-based方法**: LoRA调整LLM内部参数，Laser通过外部token注入信息，两种思路互补

## 评分
- 新颖性: ⭐⭐⭐⭐ Bi-Tuning + M-Former的组合设计新颖，MoE的引入有合理动机
- 实验充分度: ⭐⭐⭐⭐ 多个数据集和baseline对比充分，消融实验到位
- 写作质量: ⭐⭐⭐⭐ 框架图和方法描述清晰
- 价值: ⭐⭐⭐⭐ 为LLM-based推荐提供了参数高效且效果显著的方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation](../../ICLR2026/recommender/collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco.md)
- [\[ACL 2025\] RecLM: Recommendation Instruction Tuning](reclm_recommendation_instruction_tuning.md)
- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/recommender/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)
- [\[ACL 2025\] Beyond Single Labels: Improving Conversational Recommendation through LLM-Powered Data Augmentation](beyond_single_labels_improving_conversational_recommendation_through_llm-powered.md)
- [\[ACL 2025\] CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems](cove_compressed_vocabulary_expansion_makes_better_llm-based_recommender_systems.md)

</div>

<!-- RELATED:END -->
