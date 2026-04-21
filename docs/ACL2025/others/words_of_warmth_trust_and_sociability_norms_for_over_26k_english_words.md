---
title: >-
  [论文解读] Words of Warmth: Trust and Sociability Norms for over 26k English Words
description: >-
  [ACL 2025][warmth] 通过严格的众包标注流程构建了首个大规模词汇-温暖（Warmth）、信任（Trust）和社交性（Sociability）关联词典（覆盖 26k+ 英语单词），并通过儿童词汇习得分析和社交媒体刻板印象案例研究，展示了该资源在社会认知研究中的广泛价值。
tags:
  - ACL 2025
  - warmth
  - trust
  - sociability
  - lexicon
  - stereotypes
  - social cognition
---

# Words of Warmth: Trust and Sociability Norms for over 26k English Words

**会议**: ACL 2025  
**arXiv**: [2506.03993](https://arxiv.org/abs/2506.03993)  
**代码**: [http://saifmohammad.com/warmth.html](http://saifmohammad.com/warmth.html)  
**领域**: NLP / 计算社会科学 / 词汇资源  
**关键词**: warmth, trust, sociability, lexicon, stereotypes, social cognition

## 一句话总结

通过严格的众包标注流程构建了首个大规模词汇-温暖（Warmth）、信任（Trust）和社交性（Sociability）关联词典（覆盖 26k+ 英语单词），并通过儿童词汇习得分析和社交媒体刻板印象案例研究，展示了该资源在社会认知研究中的广泛价值。

## 研究背景与动机

社会心理学研究表明，温暖（Warmth, W）和能力（Competence, C）是人类评估他人和群体的两个核心维度（Stereotype Content Model, Fiske et al. 2002），深刻影响着人际交往、情绪调节、职场表现乃至政治感知。近年研究进一步将温暖分解为两个独立子维度：信任（Trust, T，包含道德、诚信、可靠性等）和社交性（Sociability, S，包含友好、合群、热情等），这为更精细的社会认知分析奠定了理论基础。

然而，资源层面存在严重不对称：能力维度已有大规模词汇标注资源（如 NRC VAD Lexicon 覆盖 20k+ 词的 dominance 标注），但温暖维度仅有极小规模的标注词典（Nicolas et al. 2021 仅 341 词）。此前尝试通过 WordNet 同义词和词嵌入自动扩展的方法效果不佳——近义词也可能传达截然不同的温暖含义（如 slip vs. fault, skinny vs. slender）。Fraser et al. (2024) 实验证实自动扩展词典在捕获 W-C 维度上无效。

这一资源缺口严重制约了计算社会科学（刻板印象追踪、公共话语分析）、发展心理学（WCTS 维度的儿童发展规律）以及 NLP 应用（偏见检测、情感分析）的研究进展。本文通过大规模众包标注填补这一空白。

## 方法详解

### 整体框架

通过精心设计的众包标注流程，为 26k+ 英语单词分别标注信任（Trust）和社交性（Sociability）得分，再通过取并集策略生成温暖（Warmth）综合得分，最终形成三套互补的词典资源（Words of Warmth / WCTS Lexicons）。

### 关键设计

1. **词汇来源选择与筛选**:

    - 功能：确定待标注的词汇集合
    - 核心思路：从 NRC VAD Lexicon v2 的约 44k 单词中筛选，排除情感高度中性的词汇（valence 得分在 -0.2 到 +0.2 之间），最终保留 26,188 个单词
    - 设计动机：既要覆盖足够多的常用英语词汇，又要侧重有情感关联的词——这些词更可能携带温暖/冷漠的含义，从而使标注资源的信息密度更高

2. **双维度独立标注体系**:

    - 功能：分别获取 Trust 和 Sociability 两个子维度的人工标注
    - 核心思路：设计两套独立的 7 级 Likert 量表（-3 到 +3），分别对应"非常值得信任/非常不值得信任"和"非常社交/非常不社交"。每个词获得 9 人（Trust 为 12 人）独立标注，最终取平均分
    - 设计动机：社会认知理论表明 Trust 和 Sociability 是温暖的两个独立子维度，独立标注能更精确地捕获各维度的差异（如 homosexual 的 T 分数远低于 S 分数）

3. **双层黄金标准质量控制**:

    - 功能：确保标注质量并防止作弊
    - 核心思路：约 2% 的题目为预先标注的控制题，其中一半为 popup gold（答错时立即弹窗反馈），另一半为 no-popup gold（静默监控，防止标注者共享控制题答案）。准确率低于 80% 的标注者的所有标注被丢弃
    - 设计动机：众包标注中标注者可能共享答案或不认真作答，双层机制从即时纠偏和隐蔽监控两个方向保障质量

### 温暖得分聚合策略

Warmth 词典采用"取绝对值更大者"的 union/or 策略：对于每个词，取 Trust 和 Sociability 中绝对值更大的分数作为 Warmth 分数。例如 uplifting（S=3, T=0.67）的 W 分数为 3；birdbrain（S=-1.71, T=-2.62）的 W 分数为-2.62。

这一设计基于社会认知理论：我们感知一个人温暖可能因为信任、因为友善，或两者兼有——不要求两个子维度同时满足。

## 实验关键数据

### 主实验：标注质量验证

| 维度 | 标注词数 | 平均标注数/词 | SHR (Spearman ρ) | SHR (Pearson r) |
|------|---------|--------------|------------------|-----------------|
| Sociability | 26,123 | 7.9 | 0.965 | 0.969 |
| Trust | 26,185 | 11.4 | 0.943 | 0.957 |
| Warmth | 26,085 | 8.8 | 0.965 | 0.974 |

Split-half reliability 均超过 0.94（1000 次随机划分取均值），远高于类似工作的可靠性水平（如词汇-焦虑关联研究的 SHR 在 0.8s）。

### 词汇分布统计

| 维度 | 正向占比 | 中性占比 | 负向占比 |
|------|---------|---------|---------|
| Trust | 28.9%（值得信任） | 38.6% | 32.4%（不值得信任） |
| Sociability | - | 较少 | 大量无生命物体被标为轻微不社交 |
| Warmth | 42%（温暖） | 10.5% | 47.5%（冷漠） |

### 消融实验：儿童词汇习得分析

基于 Kuperman et al. (2012) 的年龄习得数据集（30k 词），与 WCTS 词典交叉分析：

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 极性 W 词 vs 极性 C 词 | 各年龄段极性 W 词占比始终高于极性 C 词 | 支持"情感优先假说"（primacy of valence） |
| 早期高温暖 | 幼儿习得更多高 W 词和低 C 词 | 符合幼儿依赖照顾者温暖的发展特征 |
| S 先于 T | 早期极性 S 词占比远高于极性 T 词 | 社交性维度在儿童早期比信任维度更重要 |
| T 随年龄增长 | 低信任词在 3 岁时极少，随年龄稳步增加 | 道德/信任概念是逐步发展的 |

### 关键发现：刻板印象案例研究

使用 2015-2021 年美/加 Twitter 语料：

| 分析类型 | 目标 | 关键发现 |
|---------|------|---------|
| 直接查询 | 社会群体词的 W-C 坐标 | god: 高 W 高 C；criminal: 极低 W；disabled: 极低 C |
| 共现词分析 | Twitter 语料中 WCTS 共现 | muslim/jew/immigrant 得到低 W 分数（反映负面刻板印象） |
| 性别差异 | 性别代词/称谓 | 男性 C 更高、女性 W 更高（性别刻板印象在语言中的体现） |
| 内群-外群 | 加拿大人 vs 美国人互评 | 加拿大人自评更温暖更有能力（内群偏好）；美国人也认为加拿大人更温暖 |
| 代词分析 | I/me/you/we | I/me 低 C（符合低权力地位标记假说）；we 高 W（正面语境标记） |

## 亮点与洞察

- 资源规模的质变：比此前最大温暖词典（341 词）大 75 倍以上，从"玩具数据集"跃升为真正可用的研究工具
- T 和 S 的独立标注揭示了温暖维度的内部异质性——例如 homosexual 的 T 分数远低于 S 分数，反映了部分群体将同性恋与"不道德"而非"不友好"联系起来
- 双层黄金标准质量控制（popup + no-popup gold）可推广到其他众包标注任务
- 词汇习得分析提供了计算心理语言学的新证据：极性温暖词vs能力词的年龄分布清晰地支持了情感优先假说
- 共现词 WCTS 分析方法简单高效——仅使用词频统计即可捕获复杂的社会认知模式

## 局限与展望

- 仅覆盖英语单词，跨语言/跨文化版本待开发——不同文化中 W-C 维度的权重可能不同
- 标注者以美国人为主（69%），其次为印度/英国/加拿大，可能存在文化偏差
- 词级别标注无法捕获上下文中的动态温暖感知——同一个词在不同语境中可表达不同温暖含义
- Twitter 案例研究的地理定位可能存在噪声
- 未直接评估与下游 NLP 任务（情感分析、偏见检测、有害内容过滤）的集成效果
- 词典条目反映标注时期的文化认知，不是恒定不变的——需定期更新

## 相关工作与启发

- **vs NRC VAD Lexicon (Mohammad 2018/2025)**: VAD 覆盖 Competence（dominance）维度，本文补齐了 Warmth 维度的缺口；两者组合形成完整的 WCTS 框架
- **vs Nicolas et al. (2021)**: 仅 341 词的温暖词典，通过 WordNet/embedding 自动扩展效果差；本文通过大规模人工标注从根本上解决了覆盖率问题
- **vs 词嵌入偏见研究 (Caliskan et al. 2017)**: WIAT 通过嵌入空间检测隐式偏见；本文的词汇级 WCTS 标注提供了更直接、可解释的分析路径
- **vs 生成式 AI 偏见研究 (Kotek et al. 2023)**: Words of Warmth 可用于系统性评估 LLM 生成文本中的温暖/能力偏见

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个大规模 W-T-S 词典，填补了重要资源空白；将温暖分解为 Trust 和 Sociability 独立标注是重要创新
- 实验充分度: ⭐⭐⭐⭐ 可靠性验证、词汇习得分析、多维度刻板印象案例研究（社会群体、性别、内群外群、代词），展示了词典的广泛适用性
- 写作质量: ⭐⭐⭐⭐⭐ Saif Mohammad 一贯的高质量写作，组织清晰，伦理讨论详尽负责
- 价值: ⭐⭐⭐⭐ 作为基础资源贡献，对社会认知研究、偏见检测和计算社会科学有长期影响力

<!-- RELATED:START -->

## 相关论文

- [Behind Closed Words: Creating and Investigating the forePLay Annotated Dataset for Polish Erotic Discourse](foreplay_polish_erotic_detection.md)
- [FEAT: A Preference Feedback Dataset through a Cost-Effective Auto-Generation and Labeling Framework for English AI Tutoring](feat_a_preference_feedback_dataset_through_a_cost-effective_auto-generation_and_.md)
- [Decoding Reading Goals from Eye Movements](decoding_reading_goals_from_eye_movements.md)
- [SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL](share_text_to_sql_correction.md)
- [Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)

<!-- RELATED:END -->
