---
title: >-
  [论文解读] Beyond Single Labels: Improving Conversational Recommendation through LLM-Powered Data Augmentation
description: >-
  [ACL 2025][推荐系统][对话推荐系统] 针对对话推荐系统中的假阴性问题（用户可能喜欢的item被错误标记为负样本），提出基于LLM的数据增强框架，通过语义检索+相关性打分生成合成标签，再通过两阶段训练策略平衡语义相关性和协同信息。
tags:
  - "ACL 2025"
  - "推荐系统"
  - "对话推荐系统"
  - "数据增强"
  - "假阴性问题"
  - "大语言模型"
  - "两阶段训练"
---

# Beyond Single Labels: Improving Conversational Recommendation through LLM-Powered Data Augmentation

**会议**: ACL 2025  
**arXiv**: [2508.05657](https://arxiv.org/abs/2508.05657)  
**代码**: [github.com/xu1110/FNSCRS](https://github.com/xu1110/FNSCRS)  
**领域**: 推荐系统  
**关键词**: 对话推荐系统, 数据增强, 假阴性问题, 大语言模型, 两阶段训练

## 一句话总结

针对对话推荐系统中的假阴性问题（用户可能喜欢的item被错误标记为负样本），提出基于LLM的数据增强框架，通过语义检索+相关性打分生成合成标签，再通过两阶段训练策略平衡语义相关性和协同信息。

## 研究背景与动机

对话推荐系统（CRS）通过多轮对话与用户交互来提供推荐，但在训练过程中面临严重的假阴性问题：

- **问题实例**：用户说"我想看搞笑警察电影"，训练数据中只有一部电影标记为正样本，其他符合条件的搞笑警察电影都被错误视为负样本。
- **CRS场景的独特挑战**：与传统推荐系统不同，CRS数据集富含丰富的语义信息（对话上下文），增强标签时需要同时保证：(1) 与对话上下文的语义相关性；(2) 保留数据集中固有的协同信息（用户行为的共性和趋势）。
- **LLM的局限性**：虽然LLM擅长理解语义相关性，但难以有效捕获协同信息。过度依赖LLM建议的标签可能导致推荐偏向语义一致性而忽略协同信息，降低用户满意度。

现有方法要么将假阴性样本视为噪声进行缓解（如降低负采样概率），要么通过增强数据集来扩展标签集，但在CRS场景中缺乏有效的语义相关性与协同信息的平衡机制。

## 方法详解

### 整体框架

方法分为两个阶段：数据合成阶段和模型训练阶段。

**数据合成阶段**：使用LLM进行语义检索和相关性评分，生成合成训练数据。
**模型训练阶段**：两阶段训练——先用合成数据预训练学习语义关系，再用原始数据微调整合协同信息。

### 关键设计

1. **LLM语义检索器（Relevant Items Retrieval）**：

    - 核心思路：仅基于语义信息检索候选item，不考虑协同信息，从而避免流行度偏差等协同信息带来的偏见。
    - 使用GritLM作为文本编码器，将item描述文本和对话上下文编码为稠密向量。
    - 通过最大内积搜索为每个对话上下文检索top-50最相似item。
    - 设计动机：初始阶段忽略协同信息可以覆盖更广范围的item，避免过度聚焦热门item。

2. **LLM相关性评分器（Relevance Estimation）**：

    - 使用GPT-4生成context-item-score三元组作为训练数据（链式思维提示）。
    - 训练Gemma2-9b为每个候选item打0-4分的细粒度相关性分数。
    - 阈值设为3.5，保留高分item构成合成训练数据集。
    - 效果：ReDial原始29,810个正样本扩展到377,313个，INSPIRED从1,404扩展到15,891个。

3. **两阶段训练策略**：

    - **阶段一（预训练）**：在合成数据集上使用标准交叉熵损失训练推荐器，学习用户偏好与item之间的语义关系，避免原始数据中的偏见。
    - **阶段二（微调）**：在原始真实数据集上微调，整合协同信息。引入标签平滑项（基于KL散度），使用预训练模型的输出作为软标签，通过系数α控制对协同信息的依赖程度。
    - 设计动机：先学语义再学协同，允许以可控方式整合两类信息。

### 损失函数 / 训练策略

预训练阶段：标准交叉熵损失
$$L_{pre} = -\sum_{i=1}^{N}\sum_{j=1}^{M} y_{i,j} \cdot \log P(i,j)$$

微调阶段：交叉熵 + 标签平滑
$$L_{finetune} = L_{ce} + \alpha \cdot L_{soft}$$
$$L_{soft} = \sum_{i=1}^{N} D_{KL}(P(i), \hat{y_i})$$

其中α越大表示对协同信息的依赖越小。

## 实验关键数据

### 主实验

| 模型 | ReDial R@1 | ReDial R@10 | ReDial R@50 | INSPIRED R@1 | INSPIRED R@10 | INSPIRED R@50 |
|------|-----------|-------------|-------------|-------------|--------------|--------------|
| BARCOR | 3.13 | 17.34 | 36.32 | 2.86 | 11.06 | 30.81 |
| BARCOR + ours | **4.31** | **21.26** | **43.84** | **3.73** | **21.12** | **43.11** |
| UniCRS | 3.53 | 19.60 | 40.50 | 3.97 | 20.00 | 40.66 |
| UniCRS + ours | **3.76** | **20.93** | **42.74** | **5.43** | **22.91** | 39.47 |
| Llama2 | 3.93 | 20.74 | 41.34 | 4.46 | 11.68 | 34.16 |
| Llama2 + ours | **4.46** | **22.37** | **44.20** | **9.32** | **28.26** | **50.93** |

用户模拟器评估中提升更为显著：Llama2在INSPIRED上R@50从34.78提升到73.29（+111%）。

### 消融实验

| 配置 | ReDial R@10 | INSPIRED R@10 | 说明 |
|------|------------|--------------|------|
| BARCOR基线 | 17.34 | 11.06 | 无增强 |
| + Self-Distillation | 19.95 | 19.38 | 使用协同+语义检索 |
| + CFCRS | 18.98 | 20.50 | 反事实对话模拟 |
| + Ours | **21.26** | **21.12** | 语义优先+两阶段 |

### 关键发现

- **一致性提升**：方法在三个骨干模型、两个数据集、两种评估方式下均稳定提升，展现了强泛健壮性。
- **超越零样本LLM**：即使使用更小的模型，本方法也优于GPT-3.5和GPT-4o的零样本推荐（Llama2+ours R@10 22.37 vs GPT-4o 17.20）。
- **语义优先优于混合检索**：与同时使用协同和语义信息的Self-Distillation相比，初始阶段仅用语义信息反而效果更好，验证了避免协同偏见的策略有效性。
- **合成数据量级**：ReDial合成数据约扩大12.7倍，INSPIRED约扩大11.3倍。

## 亮点与洞察

- **假阴性问题系统化**：首次在CRS场景下系统研究假阴性问题，并提出了清晰的解决框架。
- **语义与协同的解耦设计**：不同于先前方法将两类信息混合使用，本文通过两阶段训练实现可控的信息整合，设计理念优雅。
- **与现有方法兼容**：本方法增强的是标签而非对话，可与现有对话增强方法（如CFCRS）互补。
- **实践价值高**：方法适用于各种CRS推荐器骨干，无需修改推荐器架构。

## 局限与展望

- 相关性评分器训练依赖GPT-4生成的数据，存在成本和偏见传递问题。
- 标签平滑系数α需要手动调整，缺乏自适应机制。
- 仅在电影推荐数据集上验证，未涉及其他推荐领域（如音乐、商品）。
- 两阶段训练增加了训练复杂度，预训练阶段的数据规模较大（37万+样本）。
- 可探索合成对话与合成标签的协同增强效果。

## 相关工作与启发

本文工作位于CRS、LLM增强推荐和假阴性处理的交汇处。与Wei等人（2024）在传统推荐系统中的假阴性检测工作不同，本文在检索阶段不使用协同信息以避免偏见，并通过两阶段训练以可控方式引入协同信息。这种"先语义后协同"的思路对推荐系统中如何有效利用LLM能力有普遍参考价值。

## 评分

- 新颖性: ⭐⭐⭐⭐ 在CRS场景首次系统研究假阴性问题，两阶段训练策略有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 三个骨干，两个数据集，两种评估方式，深度分析充分
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图示直观，但部分符号较多
- 价值: ⭐⭐⭐⭐ 方法简洁有效，易于在实际CRS中应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Laser: Bi-Tuning with Collaborative Information for Controllable LLM-Based Sequential Recommendation](bi-tuning_with_collaborative_information_for_controllable_llm-based_sequential_r.md)
- [\[ICML 2025\] MATCHA: Toward Safe and Human-Aligned Game Conversational Recommendation via Multi-Agent Decomposition](../../ICML2025/recommender/toward_safe_and_human-aligned_game_conversational_recommendation_via_multi-agent.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](../../ACL2026/recommender/harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[NeurIPS 2025\] VisualLens: Personalization through Task-Agnostic Visual History](../../NeurIPS2025/recommender/visuallens_personalization_through_task-agnostic_visual_history.md)
- [\[ACL 2026\] Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](../../ACL2026/recommender/where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)

</div>

<!-- RELATED:END -->
