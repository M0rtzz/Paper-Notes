---
title: >-
  [论文解读] Entity Framing and Role Portrayal in the News
description: >-
  [ACL 2025][实体框架] 本文构建了一个包含 5 种语言、1378 篇新闻文章、5800+ 实体标注的多语言层次化实体框架语料库，提出含 22 种精细角色的叙事角色分类体系（主角 / 反派 / 无辜者三大框架下），并在微调多语言 Transformer 和 LLM 层次零样本学习上建立了基准。
tags:
  - ACL 2025
  - 实体框架
  - 叙事角色
  - 多语言标注
  - 层次分类
  - 零样本学习
---

# Entity Framing and Role Portrayal in the News

**会议**: ACL 2025  
**arXiv**: [2502.14718](https://arxiv.org/abs/2502.14718)  
**代码**: [数据集页面](https://mbzuai-nlp.github.io/entity-framing/)  
**领域**: NLP / 媒体分析 / 实体框架  
**关键词**: 实体框架, 叙事角色, 多语言标注, 层次分类, 零样本学习

## 一句话总结

本文构建了一个包含 5 种语言、1378 篇新闻文章、5800+ 实体标注的多语言层次化实体框架语料库，提出含 22 种精细角色的叙事角色分类体系（主角 / 反派 / 无辜者三大框架下），并在微调多语言 Transformer 和 LLM 层次零样本学习上建立了基准。

## 研究背景与动机

在社交媒体和新闻媒体高度发达的信息时代，新闻报道中对实体（个人、组织、群体）的"框架化"呈现方式对公共认知有深远影响：

**情感框架的影响力**：称同一群体为"自由战士"vs"恐怖分子"会激活完全不同的情感反应，但现有 NLP 研究主要停留在情感极性层面

**粗粒度框架的局限**：已有研究多将实体简单划分为英雄/恶棍/受害者三类，但现实报道中角色远比这复杂——同一实体可能在不同段落扮演不同角色

**缺乏多语言和领域多样性**：现有数据集多为单语言、单领域

**实体级 vs 文章级分析**：大多数框架分析在文章级别进行，缺少对具体实体的精细刻画

核心创新：从叙事功能而非道德判断出发重新定义框架类别，用 protagonist/antagonist/innocent 取代 hero/villain/victim，关注实体在叙事中的功能角色。

## 方法详解

### 整体框架

#### 分类体系设计

22 种精细角色嵌套在三大叙事框架下：
- **Protagonist（主角）**：Guardian, Martyr, Underdog, Peacemaker, Rebel, Virtuous, Unifier 等
- **Antagonist（反派）**：Tyrant, Deceiver, Bigot, Foreign Aggressor, Instigator, Corrupt, Incompetent 等
- **Innocent（无辜者）**：Victim, Scapegoat, Exploited, Forgotten

任务形式化：给定文章 $S$ 和实体mention的 span $[i,j]$，预测角色集合 $\{r_1, r_2, ..., r_k\} \subseteq R$。

### 关键设计

1. **语料库构建流程**：

    - **文章选取**：从大规模新闻聚合工具获取候选文章 → 关键词过滤（>250词）→ 人工审核（Perfect Fit / Average Fit / Uncertain / Unfit）→ 零样本分类器和说服力评分进一步筛选
    - **覆盖范围**：5 种语言（保加利亚语、英语、印地语、葡萄牙语、俄语），2 个领域（俄乌战争 + 气候变化）
    - **标注过程**：每篇由 2 名标注者标注 → curator 审核和整合 → 定期随机质检
    - **标注工具**：INCEpTION

2. **XLM-R 微调实验设计**：

    - 输入格式：`entity mention + [SEP] + title + [SEP] + context`
    - 三种上下文粒度：全文（DOC）、段落（PAR）、句子（SEN）
    - 处理长文档：通过缩小上下文窗口到段落/句子级别绕过 512 token 限制
    - 多标签分类：sigmoid 激活 + Binary Cross-Entropy 损失

3. **LLM 层次零样本学习**：

    - **单步法（Single-Step）**：一个 prompt 同时预测主框架和精细角色
    - **多步法（Multi-Step）**：先预测主框架（主角/反派/无辜者），再基于此预测精细角色
    - 使用 GPT-4o 进行零样本推理

### 损失函数 / 训练策略

- XLM-R：Binary Cross-Entropy 损失用于多标签分类
- 基于文章级别划分 train/dev/test，防止数据泄露
- 多语言联合训练 vs 单语言训练两种设置

## 实验关键数据

### 主实验：不同上下文粒度的性能（表格）

| 上下文 | 主框架 Accuracy | 主框架 Balanced Acc | 精细角色 Micro F1 | 精细角色 Macro F1 |
|-------|---------------|-------------------|-----------------|-----------------|
| DOC | 0.601 / 0.723* | 0.590 / 0.724* | 0.391 | 0.231 |
| PAR | **0.738** / **0.753*** | **0.739** / **0.755*** | 0.421 | 0.239 |
| SEN | 0.718 / 0.750* | 0.712 / 0.750* | **0.434** | **0.253** |

> M = 仅训练主框架，F = 训练精细角色后评估主框架。段落级在主框架最优，句子级在精细角色最优。

### 零样本 vs 微调对比（表格）

| 方法 | 主框架 Accuracy | 精细 Micro F1 | 精细 Macro F1 | 成本(USD) |
|------|---------------|-------------|-------------|----------|
| GPT-4o 单步 | 0.703 | 0.382 | **0.310** | $5.32 |
| GPT-4o 多步 | 0.705 | 0.317 | 0.277 | $3.19 |
| XLM-R (PAR) | **0.753** | **0.421** | 0.239 | - |

> XLM-R 在 Micro F1 上胜出，但零样本在 Macro F1 上更好——因为 XLM-R 在稀有角色上训练数据不足。

### 关键发现

1. **段落级上下文最优**：全文信息过多反而干扰，句子太短缺乏叙事背景
2. **多语言训练全面优于单语言**：跨语言迁移显著提升了所有语言的性能
3. **类别极度不平衡**：innocent 类中 83.6% 是 victim，74% 的实体仅出现一次
4. **角色转换罕见但存在**：1378 篇文章中仅 99 篇有主框架角色变化，但转换序列很有信息量
5. **多步法更省钱**：比单步法便宜 40%，主框架性能相当但精细角色更差（错误传播）
6. **Macro F1 一致很低**：所有方法在稀有角色上表现都很差，反映了数据不平衡的挑战

## 亮点与洞察

- **分类体系设计精良**：22 种叙事角色的层次分类体系经过大量实际标注验证，比简单的三分类提供了丰富得多的分析维度
- **"叙事功能"取代"道德判断"**：protagonist/antagonist/innocent 比 hero/villain/victim 更客观，减少了标注时的主观偏见
- **多语言覆盖有实际价值**：5 种语言 × 2 个地缘政治敏感领域的组合，为跨文化媒体分析提供了独特资源
- **角色共现和转换分析**：发现 peacemaker 常与 guardian 共现，scapegoat 常与 exploited 共现——这反映了真实叙事的复杂性
- **零样本 vs 微调的互补性**：零样本处理稀有类别更好，微调处理常见类别更好，暗示两者结合可能是最优方案

## 局限与展望

1. **领域受限**：仅覆盖俄乌战争和气候变化两个领域
2. **标注主观性**：虽有详细指南，但实体框架标注本质上主观味浓
3. **IAA 中等**：Krippendorff's α 在 0.43-0.73 之间，可接受但不算高
4. **类别不平衡严重**：多数精细角色样本很少，影响模型训练
5. **零样本依赖闭源模型**：GPT-4o 可能被废弃，影响可复现性
6. **未探索 few-shot 和多任务学习**：可能进一步提升稀有类别性能

## 相关工作与启发

- **Sharma et al. (2023)** 在 meme 中识别英雄/恶棍/受害者，但仅限视觉模态
- **Bergstrand & Jasper (2018)** 分析了英雄/恶棍/受害者三元组的道德属性，本文的功能性角色定义是对此的改进
- **ABSA 研究**：本文任务与方面级情感分析相关但不同——不分配情感极性而是分配叙事角色
- 启发：实体框架分析可与虚假信息检测、逆向宣传分析等任务结合，形成更完整的媒体素养工具链

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 22 种角色的层次分类体系和多语言语料库是全新资源 |
| 实验充分度 | 4 | 微调 + 零样本 + 语料库分析，覆盖面广 |
| 写作质量 | 4 | 数据集论文写法规范，统计分析详尽 |
| 价值 | 4.5 | 作为社区资源价值很高，应用前景广泛 |

<!-- RELATED:START -->

## 相关论文

- [Narrative Media Framing in Political Discourse](narrative_media_framing_in_political_discourse.md)
- [Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation](evaluating_design_decisions_for_dual_encoder-based_entity_disambiguation.md)
- [SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL](share_text_to_sql_correction.md)
- [Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [Decoding Reading Goals from Eye Movements](decoding_reading_goals_from_eye_movements.md)

<!-- RELATED:END -->
