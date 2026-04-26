---
title: >-
  [论文解读] taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades
description: >-
  [ACL 2025][德语语料库] 构建迄今最大的公开德语新闻语料库 taz2024full（180万+篇文章，1980-2024），并适配actor级语篇分析管线至德语，揭示四十余年间新闻报道中持续存在的性别表征失衡与情感偏差。
tags:
  - ACL 2025
  - 德语语料库
  - 性别偏见
  - 媒体分析
  - 纵向研究
  - NER
---

# taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades

**会议**: ACL 2025  
**arXiv**: [2506.05388](https://arxiv.org/abs/2506.05388)  
**代码**: [Ognatai/corpus_pipeline](https://github.com/Ognatai/corpus_pipeline)  
**领域**: 社会计算 / 偏见检测  
**关键词**: 德语语料库, 性别偏见, 媒体分析, 纵向研究, NER

## 一句话总结

构建迄今最大的公开德语新闻语料库 taz2024full（180万+篇文章，1980-2024），并适配actor级语篇分析管线至德语，揭示四十余年间新闻报道中持续存在的性别表征失衡与情感偏差。

## 研究背景与动机

**领域现状**：新闻媒体是公众获取信息的核心渠道，媒体中的性别偏见会潜移默化地影响社会认知。然而对性别偏见的长时间纵向量化研究极为稀缺，主要受限于大规模、长跨度语料库的匮乏。

**现有痛点**：公开语料库生态由英语主导，德语大规模新闻资源严重匮乏。现有德语语料库（DWDS、DeReKo、Leipzig Wortschatz等）多限于关键词搜索或句子级查询，不支持全量下载与大规模分析。绝大多数德语报纸因许可费和法律限制无法公开使用。此外，现有偏见检测方法多基于词嵌入或统计关联，缺乏基于actor级别的语篇分析。

**核心矛盾**：要回答"性别偏见如何随时间演变"需要同时具备两个条件——覆盖数十年的大规模公开语料库和能在角色粒度上分析性别表征的自动化管线——目前德语NLP生态中两者都不存在。

**本文目标** (1) 构建并公开发布首个支持44年纵向研究的大型德语新闻语料库；(2) 将Urchs et al. (2024)的actor级偏见检测管线从英语适配到德语，支持全量语料库分析。

**切入角度**：利用柏林左翼日报taz作为唯一允许免费学术使用的德语报纸数据源，爬取1980-2024年全部公开文章，并刻意选择传统可解释方法（而非LLM）进行偏见分析——避免"用有偏见的工具检测偏见"的方法论矛盾。

**核心 idea**：大规模语料库构建+actor级语篇分析管线的德语适配，实现44年跨度的新闻性别偏见纵向量化研究。

## 方法详解

### 整体框架

工作包含两个核心部分：(1) 语料库构建——从柏林左翼日报 taz 网站爬取 1980-2024 年间所有公开文章并结构化存储；(2) 偏见分析管线——适配 Urchs et al. (2024) 的 actor 级语篇分析方法至德语，从单篇处理扩展到全量语料库分析，输出年度歧视报告。

### 关键设计

1. **大规模德语语料库构建（taz2024full）**:

    - 功能：提供首个公开可用的、覆盖44年的大规模德语新闻语料库
    - 核心思路：数据来源为德国柏林左翼日报taz（die Tageszeitung），唯一允许免费学术使用的德语报纸。2024年8-11月间爬取全部公开文章，以JSON格式存储，每条记录包含元数据（发布时间、作者、关键词、token数、是否含人物实体）和文本（标题、导语、正文）。使用SoMaJo分词器处理并筛除≤3 token的碎片，最终在Zenodo公开发布
    - 设计动机：现有德语语料库不支持全量下载和大规模分析，且多数因版权无法公开。taz的宽松许可政策使其成为唯一可行的数据源

2. **Actor级偏见检测管线**:

    - 功能：在角色粒度上自动化分析性别表征，输出年度歧视报告
    - 核心思路：管线流程为NER人物实体提取（spaCy）→通用指称词补充（如"母亲""父亲"）→同名实体合并→共指消解（coreferee）将代词链接至对应actor→基于代词分布进行性别分类（>70%阴性/阳性代词即分类为女/男）→提取每个actor的歧视标记（数量、提及频率、情感倾向、性别编码词汇、PMI最高形容词）。德语适配包括：改用代词驱动分析、仅保留有共指链的actor、新增泛阳性和性别中立语言检测、情感分析替换为german-sentiment-bert
    - 设计动机：基于Urchs et al. (2024)的英语管线进行语言适配，确保方法可比性的同时处理德语特有的语法性别问题

3. **刻意不使用LLM的方法论选择**:

    - 功能：确保偏见检测工具链本身不引入系统性偏差
    - 核心思路：选择传统可解释方法（spaCy + coreferee + PMI），避免LLM自身包含的性别/政治偏见污染分析结果。歧视分析以报告形式输出而非最终判定，将解释权留给研究者
    - 设计动机：用有偏见的工具检测偏见在方法论上自相矛盾；传统方法输出透明可审计，确保研究结论的可信度

## 实验关键数据

### 主实验

| 指标 | 值 |
|------|------|
| 文章总数 | 1,834,370 篇 |
| 时间跨度 | 1980-2024（44年） |
| 唯一 token 数 | 6,944,197 |
| 含人物实体的文章占比 | 83% |
| 平均 token 长度 | 5.15 字符 |
| 平均句子长度 | 20.07 tokens |
| 平均文章长度 | 396.89 tokens / 19.77 句 |
| 中位文章长度 | 276 tokens / 13 句 |
| 文章数峰值 | 2004 年（73,002 篇） |
| 2007年后趋势 | 因付费内容增加导致公开文章数持续下降 |

### 消融实验

| 分析维度 | 结果 |
|---------|------|
| Actor 性别比例 | 1990s 起男性 actor 数量和提及频率持续显著高于女性 |
| 时序演变 | 2010s 起女性 actor 纳入比例逐渐增加，但男性 mentions 仍占优 |
| 媒体可见性 | 即使近年 actor 数量接近平衡，男性仍获得更多文本篇幅 |
| 情感倾向 | 整体偏中性略负；44 年间女性 actor 情感得分始终略低于男性 |
| 性别编码词汇 | Gender Decoder 词表中的性别编码词使用极少 |
| PMI 形容词 | 高 PMI 形容词在性别间差异不大且随时间稳定 |
| 性别中立语言 | taz 几乎未系统性采用德语性别中立表达（Gendern） |
| Neo-pronouns | 全语料库仅 5 篇文章包含德语新代词 |

### 关键发现

- **男性主导贯穿四十年**：无论actor出现数量还是提及频率，男性始终占主导地位；即便2010s后actor数趋于平衡，mentions差距依然存在
- **持续微妙的情感偏差**：对女性actor的描述情感始终略低于男性，差异虽小但44年从未逆转
- **进步媒体的盲区**：taz作为左翼进步媒体，在性别中立语言的使用上并未走在前列

## 亮点与洞察

- **44年纵向性别偏见研究极为罕见**：提供独特的长时间维度洞察，揭示偏见消除是一个比想象中更缓慢的过程
- **不用LLM的设计体现方法论自洽性**：避免"用有偏见的工具检测偏见"的方法论矛盾，在偏见研究中值得借鉴
- **管线语言无关设计**：核心管线可迁移至其他语言，歧视报告不做最终判定，留给用户解释

## 局限与展望

- 仅来自单一左翼媒体taz，无法代表整个德语媒体生态
- 德语共指消解模型精度有限，影响性别推断准确性
- 限于二元性别分析（德语缺乏通用非二元代词）
- 2007年后因付费内容增加导致可用文章数下降，可能引入采样偏差

## 相关工作与启发

- **vs Urchs et al. (2024)**：他们提出英语actor级偏见检测管线，本文将其适配到德语并扩展了泛阳性检测和PMI分析，证明了框架的跨语言可迁移性
- **vs 嵌入偏见检测方法**：WEAT/SEAT等方法在词/句嵌入层面检测偏见，本文在actor级别分析，粒度更细且与真实新闻报道直接对应

## 评分
- 新颖性: ⭐⭐⭐ 语料库构建有价值但技术方法相对常规
- 实验充分度: ⭐⭐⭐ 44年纵向分析覆盖全面但缺少与其他媒体的对比
- 写作质量: ⭐⭐⭐⭐ 研究动机、方法论选择的讨论充分且有深度
- 价值: ⭐⭐⭐⭐ 大规模公开德语语料库本身就是重要资源贡献

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] GG-BBQ: German Gender Bias Benchmark for Question Answering](gg-bbq_german_gender_bias_benchmark_for_question_answering.md)
- [\[ACL 2025\] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language](exploring_gender_bias_in_large_language_models_an_in-depth_dive_into_the_german_.md)
- [\[ACL 2025\] Translate With Care: Addressing Gender Bias, Neutrality, and Reasoning in Large Language Model Translations](translate_with_care_addressing_gender_bias_neutrality_and_reasoning_in_large_lan.md)
- [\[NeurIPS 2025\] Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs](../../NeurIPS2025/social_computing/auto-search_and_refinement_an_automated_framework_for_gender_bias_mitigation_in_.md)
- [\[ACL 2025\] BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)

<!-- RELATED:END -->
