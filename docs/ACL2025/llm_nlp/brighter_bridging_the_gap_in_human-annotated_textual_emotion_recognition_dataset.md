---
title: >-
  [论文解读] BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages
description: >-
  [ACL 2025 (Best Resource Paper)][LLM/NLP][情感识别] 本文构建了覆盖28种语言的多标签情感标注数据集BRIGHTER，重点覆盖非洲、亚洲、东欧和拉美的低资源语言，由母语使用者标注，并在单语和跨语言情感识别任务上建立了基准实验结果。
tags:
  - "ACL 2025 (Best Resource Paper)"
  - "LLM/NLP"
  - "情感识别"
  - "多语言数据集"
  - "低资源语言"
  - "多标签标注"
  - "跨语言迁移"
---

# BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages

**会议**: ACL 2025 (Best Resource Paper)  
**arXiv**: N/A  
**链接**: [ACL Anthology](https://aclanthology.org/2025.acl-long.436/)
**代码**: 有（数据集公开发布）  
**领域**: NLP理解 / 情感识别 / 多语言NLP  
**关键词**: 情感识别, 多语言数据集, 低资源语言, 多标签标注, 跨语言迁移

## 一句话总结

本文构建了覆盖28种语言的多标签情感标注数据集BRIGHTER，重点覆盖非洲、亚洲、东欧和拉美的低资源语言，由母语使用者标注，并在单语和跨语言情感识别任务上建立了基准实验结果。

## 研究背景与动机

**领域现状**：文本情感识别（Emotion Recognition）是NLP中的重要任务，被广泛应用于对话系统、社交媒体分析、心理健康监测等场景。当前的情感识别研究和数据集主要集中在英语等高资源语言，形成了严重的资源不均衡。

**现有痛点**：低资源语言（如非洲的豪萨语、约鲁巴语，亚洲的乌尔都语、僧伽罗语，东欧的罗马尼亚语等）严重缺乏高质量的情感标注数据集。现有的多语言情感数据集要么语言覆盖有限，要么依赖机器翻译而非母语者标注，导致情感表达的文化细微差异被忽略。

**核心矛盾**：情感的表达方式具有强烈的文化和语言依赖性——不同语言中"愤怒"、"悲伤"等情感的表达方式、强度和社会规范截然不同。依赖翻译或高资源语言代理模型无法真正解决低资源语言的情感理解问题。

**本文目标**：(1) 为28种语言创建由母语者标注的多标签情感数据集；(2) 覆盖多种文本领域（社交媒体、新闻等）；(3) 建立单语和跨语言的情感识别基准。

**切入角度**：作者组建了一个大规模国际合作团队（40+位共同作者），每种语言由相应的母语研究者负责数据收集和标注，确保标注质量和文化适切性。

**核心 idea**：通过国际协作构建大规模、多领域、多语言的情感标注数据集，填补低资源语言情感识别的数据空白，并以此为基础评估当前模型（含LLM）在跨语言情感识别中的表现差异。

## 方法详解

### 整体框架

BRIGHTER数据集的构建遵循"数据收集→标注方案设计→母语者标注→质量控制→基准实验"的流程。输入是来自多种文本领域的原始文本，输出是带有多标签情感标注的数据实例。情感标签采用Ekman的六种基本情感（愤怒、厌恶、恐惧、快乐、悲伤、惊讶）加上"中性"类别。

### 关键设计

1. **多语言数据收集与领域覆盖**:

    - 功能：为每种语言收集来自多种文本领域的样本
    - 核心思路：根据各语言的实际数字内容生态，从Twitter/X、新闻评论、Reddit、本地论坛等渠道收集文本。不强制所有语言使用相同来源，而是根据各语言的网络生态灵活选择
    - 设计动机：不同语言的数字内容分布差异巨大，灵活的收集策略能获得更自然、更具代表性的语料

2. **多标签情感标注方案**:

    - 功能：支持一个文本实例同时标注多种情感
    - 核心思路：每个标注者对每条文本标注所有适用的情感标签（从7个类别中选择），并可标注情感强度。采用多标注者机制（每条至少2-3位标注者），通过标注者间一致性（inter-annotator agreement）衡量标注质量
    - 设计动机：真实文本往往同时表达多种情感（如"又惊又喜"），单标签方案会丢失丰富的情感信息

3. **跨语言实验框架**:

    - 功能：评估单语训练、跨语言零样本迁移和多语言联合训练的效果
    - 核心思路：使用XLM-RoBERTa等多语言预训练模型，分别在单语数据上微调、在英语上微调后零样本迁移到目标语言、以及在多语言混合数据上联合训练。同时评估GPT-4等LLM在零样本/少样本设置下的表现
    - 设计动机：全面评估当前技术栈在低资源语言情感识别上的真实能力，揭示资源不均衡带来的性能差距

### 训练策略

基准实验采用标准的多标签分类框架，使用二元交叉熵损失。对XLM-RoBERTa进行语言特定微调和多语言联合微调。LLM实验使用零样本和少样本prompt。

## 实验关键数据

### 主实验

| 语言组 | 模型 | 平均F1 (加权) | 与英语差距 |
|--------|------|--------------|-----------|
| 高资源语言（英/法/中） | XLM-R fine-tuned | ~72-78% | 基准 |
| 中资源语言（印地/阿拉伯） | XLM-R fine-tuned | ~62-70% | -8~15% |
| 低资源语言（豪萨/约鲁巴/僧伽罗） | XLM-R fine-tuned | ~48-58% | -20~30% |
| 跨语言零样本 | XLM-R (en→target) | ~45-65% | 变化大 |
| 零样本 | GPT-4 | ~50-65% | 语言间差异显著 |

### 消融实验

| 配置 | 平均F1 | 说明 |
|------|--------|------|
| 单语微调 | 最优 | 语言特定数据效果最好 |
| 多语言联合训练 | 接近单语 | 对低资源语言有轻微提升 |
| 零样本跨语言迁移 | 明显下降 | 语言距离越远效果越差 |
| GPT-4 零样本 | 中等 | 对部分低资源语言表现出乎意料地差 |

### 关键发现
- 低资源语言的情感识别性能与高资源语言之间存在20-30%的F1差距，即使使用最先进的多语言模型也难以完全弥合
- LLM（如GPT-4）在低资源语言上的零样本表现不如预期，特别是在非拉丁字母语言上
- 情感强度识别（intensity prediction）比类别识别更具挑战性，跨语言差异更大
- 标注者一致性在不同语言间有显著差异，反映了情感表达的文化差异性

## 亮点与洞察
- **大规模国际协作的数据建设范式**值得借鉴——通过分布式标注（每种语言由当地研究者负责）既保证了规模又保证了质量，这种模式可推广到其他多语言NLP任务
- **多标签+强度的标注方案**比传统单标签方案更贴近真实情感表达，为后续研究提供了更丰富的标注信号
- **获得Best Resource Paper奖**说明低资源语言数据集建设是当前NLP社区的核心关切之一

## 局限与展望
- 部分语言的数据规模仍然较小（数百条），可能不足以训练可靠的模型
- 标注方案基于Ekman的基本情感理论，在不同文化中可能存在适用性问题（如某些文化特有的情感类别未被覆盖）
- 未来可探索利用BRIGHTER数据集进行更深入的跨文化情感对比研究
- 可以将数据集扩展到对话场景和代码混用（code-switching）文本

## 相关工作与启发
- **vs GoEmotions**: GoEmotions提供了英语的细粒度情感标注，但只覆盖英语；BRIGHTER牺牲了细粒度但获得了语言广度
- **vs SemEval情感任务**: SemEval历年情感任务陆续增加了语言覆盖，但缺乏统一的标注框架；BRIGHTER提供了统一方案
- **vs AfriSenti**: AfriSenti专注非洲语言情感分析，BRIGHTER可视为其全球化扩展版

## 评分
- 新颖性: ⭐⭐⭐⭐ 数据集构建思路不新颖，但28种语言的覆盖规模和母语者标注策略具有重要价值
- 实验充分度: ⭐⭐⭐⭐ 单语、跨语言、LLM多种设置都有涵盖，分析较全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，对数据收集挑战的讨论很有价值
- 价值: ⭐⭐⭐⭐⭐ Best Resource Paper实至名归，填补了多语言情感识别的重要数据空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)
- [\[ACL 2025\] Team Anotheroption at SemEval-2025 Task 8: Bridging the Gap Between Open-Source and Proprietary LLMs in Table QA](team_anotheroption_at_semeval-2025_task_8_bridging_the_gap_between_open-source_a.md)
- [\[ACL 2025\] Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](knowledge_boundary_crosslingual.md)
- [\[ACL 2025\] AfroBench: How Good are Large Language Models on African Languages?](afrobench_how_good_are_large_language_models_on_african_languages.md)
- [\[ACL 2025\] Mind the (Belief) Gap: Group Identity in the World of LLMs](mind_the_belief_gap_group_identity_in_the_world_of_llms.md)

</div>

<!-- RELATED:END -->
