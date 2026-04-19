---
title: >-
  [论文解读] Tell, Don't Show: Leveraging Language Models' Abstractive Retellings to Model Literary Themes
description: >-
  [ACL2025][文本生成][topic modeling] 提出 Retell 方法：利用小型 LM 对文学段落进行抽象复述（abstractive retelling），
tags:
  - ACL2025
  - 文本生成
  - topic modeling
  - literary analysis
  - abstractive retelling
  - LDA
  - cultural analytics
---

# Tell, Don't Show: Leveraging Language Models' Abstractive Retellings to Model Literary Themes

**会议**: ACL2025  
**arXiv**: [2505.23166](https://arxiv.org/abs/2505.23166)  
**代码**: [lucy3/tell_dont_show](https://github.com/lucy3/tell_dont_show)  
**作者**: Li Lucy, Camilla Griffiths, Sarah Levine, Jennifer L. Eberhardt, Dorottya Demszky, David Bamman
**机构**: UC Berkeley, Stanford University
**领域**: nlp_generation  
**关键词**: topic modeling, literary analysis, abstractive retelling, LDA, cultural analytics

## 一句话总结

提出 Retell 方法：利用小型 LM 对文学段落进行抽象复述（abstractive retelling），
将叙事中"展示"（show）的感官细节转化为"告知"（tell）的高层概念，
再对复述文本运行 LDA 主题建模，
在资源受限条件下显著优于直接 LDA 和直接向 LM 询问主题标签的基线方法。

## 研究背景与动机

文学文本分析中的主题建模是文化分析（cultural analytics）的重要任务。
传统的词袋方法如 LDA 在处理文学文本时面临独特挑战：
文学创作的黄金法则是 **"展示，而非告知"**（show, don't tell）——
好的叙事通过低层次的感官细节（角色动作、对话、场景描写）
而非高层次的抽象说明来传达主题。
这导致 LDA 依赖的词汇级特征难以捕捉跨文档的深层主题。

例如，一段描写角色拖着身体缓慢移动的文字，
LDA 只能看到 "sluggishly"、"arms"、"legs" 等表面词汇，
而无法提炼出"暴力后果"或"身体创伤"这样的高层主题。

同时存在一个实际问题：
虽然强大的 LM（如 GPT-4）开辟了新可能，
但人文学科的研究者往往受限于 API 成本和计算资源。
已有的 LM 主题建模方法（如 TopicGPT）需要复杂的多步提示链，
且小型 LM 在直接生成主题标签时表现不稳定：
Llama 3.1 8B 在 TopicGPT 框架下为 100 个文档产生了 486 个主题，
生成的标签容易过于宽泛（如 "life" 覆盖 32.9% 段落）。

**核心洞察**：
与其让 LM 直接输出主题标签，
不如让 LM "告诉"（tell）我们文学段落在"展示"（show）什么——
即做抽象复述，将叙事的表面形式翻译为高层概念，
然后将经典 LDA 应用于这些复述文本上。

## 方法详解

### Retell 框架

方法分为两步：

**步骤一：抽象复述**

用小型指令微调 LM 对每个文学段落（不超过 250 词）生成复述。
使用简短的单次提示模板，核心指令为：
"In one paragraph, [VERB] the following book excerpt
for a literary scholar analyzing narrative content."

尝试三种动词 VERB：
- describe（描述）：鼓励高层抽象
- summarize（摘要）：鼓励高层抽象
- paraphrase（改写）：保留更多原文低层细节

复述平均长度约 105-170 词。

**步骤二：LDA 主题建模**

在复述文本上运行 Mallet LDA，预处理步骤包括：
- 小写化，去除少于 3 字符的词
- 去除出现在超过 25% 文档中的高频词
- 去除出现在少于 5 篇复述中的低频词
- 用 spaCy NER 去除角色名字（避免按书聚类）

### 测试模型

四个资源高效的小型指令微调 LM：
- GPT-4o mini（闭源）
- Llama 3.1 8B（开源）
- Phi-3.5-mini / 3.8B（开源）
- Gemma 2 2B（开源）

### 基线方法

1. **Default LDA**：直接在原始段落上运行 LDA
2. **TopicGPT-lite**：改编自 TopicGPT 的两阶段方案
    - 阶段一（主题生成）：LM 在 N=1000 采样文档上逐文档提出一个主题
    - 阶段二（主题分配）：LM 为所有文档分配主题标签
    - 限制单标签生成以缓解小型 LM 的主题数爆炸问题

### Retell 的实际优势

- 每段文本仅需一次 LM 推理加 LDA，运行效率高于 TopicGPT-lite
- 主题数 k 可快速调整而无需重新运行 LM
- 单条提示即可完成，无需复杂的提示工程

## 实验关键数据

### 实验一：段落集标签相关性评估（Table 2）

**数据集构建**：
- 从 Project Gutenberg 和当代畅销书列表收集 50.7k 标题
- 使用 Goodreads 读者标签、SparkNotes 和 LitCharts 主题标签作为金标准
- 人工将标签归为 27 个通用主题（如 gender, race, war, love 等）
- 最终获得 11.6k 标注段落（732 本书，21.1k 主题-段落对）
- 补充等量随机段落以稳定 LDA 估计，数据总计 5.02M 词

**评估方式**：
Prolific 众包标注者评判预测主题与金标准的语义相关性，
采用 3 分制评分，时薪 $16，标注者一致性加权 Cohen's kappa = 0.70。

| 方法 | 非常相关 | 不相关 |
|------|---------|--------|
| **Retell-describe** | **0.60** | **0.10** |
| **Retell-summarize** | **0.59** | 0.11 |
| Retell-paraphrase | 0.50 | 0.14 |
| Default LDA | 0.38 | 0.27 |
| TopicGPT-lite | 0.22-0.35 | 0.17-0.68 |

**关键发现**：
- Retell-describe/summarize 大幅优于所有基线
- 抽象动词 describe/summarize 优于 paraphrase（0.60 vs 0.50），验证了 telling > showing 假设
- Default LDA 的主题充斥功能词（如 n't, got, say），语义模糊
- TopicGPT-lite 产生过于宽泛的标签（如 loneliness 覆盖 education 主题）

### 实验二：段落级主题入侵测试（Table 3）

由具有影视和文学标注经验的内部标注者评估
（加权 Cohen's kappa = 0.66），对 50 段文本的 top-3 预测主题加入侵主题评分：

| 方法 | Top-1 | Top-2 | Top-3 | 入侵者 |
|------|-------|-------|-------|--------|
| Retell-desc (GPT-4o mini, k=50) | **2.81** | **2.51** | 2.23 | 1.63 |
| Retell-desc (GPT-4o mini, k=89) | 2.60 | 2.53 | **2.40** | 1.77 |
| TopicGPT-lite (GPT-4o mini, k=89) | 2.59 | 2.48 | 2.51 | 1.52 |
| Retell-summ (Llama 8B, k=50) | 2.36 | 2.30 | 2.12 | 1.67 |

所有方法的 top 主题得分显著高于入侵者（U 检验 p<0.05），
Retell 在段落级与 TopicGPT-lite 表现可比，但 Retell 更轻量高效。

### 案例研究：ELA 教材中的种族主题（Table 4-5）

**数据**：
396 本美国高中英语教材（AP Literature 考题书目加教师推荐书目），
1,645 段人工标注段落（401 mention + 198 discuss + 其余 neither），
由社会心理学专家领导的本科生团队历时四个月编码。

**发现**：
- Retell 产生了与种族身份高度相关的主题词
  如 "black, racial, white, community, individuals"
- 这些主题在 discuss 段落中概率显著高于 mention（U 检验 p<0.001）
- Retell 的两个相关主题联合使用可提高召回率而不降低精度
- Default LDA 的对应高频词（"black, people, white"）区分力弱
- TopicGPT-lite 的标签（Identity, Family, Work）在三类段落中无显著差异

## 亮点

- **概念创新**：巧妙利用文学创作 "show vs. tell" 原则转化为计算方法论，
  让 LM 充当叙事细节到抽象概念的"翻译层"
- **极简设计**：一条提示加标准 LDA，无需复杂提示链或模型微调，
  对人文学者和资源受限场景极为友好
- **跨学科价值**：在 NLP 方法创新与人文教育应用之间搭建了优秀桥梁，
  案例研究展示了对种族议题的实际分析能力
- **多维评估**：众包评估加专家标注加主题入侵测试加案例研究，
  覆盖主题级和段落级两个粒度

## 局限与展望

- LM 的复述只代表一种解读，文学阅读本质上是主观且文化建构的过程
- LM 可能利用预训练中的书籍知识补充段落中未出现的信息，引入上下文偏差
- 仅关注显式种族提及，隐式种族线索的识别需更深入研究
- 复述可能遗漏关键内容（如种族刻板印象描写），摘要内容选择行为值得研究
- 未充分探索更大模型表现（GPT-4o 初步结果显示强 LM 直接生成标签也很有效）
- 金标准标签来自特定在线资源，可能存在内容生产和覆盖偏差

## 与相关工作的对比

- **LDA (Blei et al., 2003)**：经典概率主题模型，依赖词汇表面形式，
  是本文的基础组件，运行于 LM 复述之上
- **TopicGPT (Pham et al., 2024)**：直接从 LM 获取主题标签，多步提示，
  是本文基线，小型 LM 上表现不稳定
- **BERTopic 等嵌入主题模型**：基于文档嵌入聚类，
  与 Retell 不同路线，后者利用生成式 LM 的抽象能力
- **隐含信息补全 (Zhong et al., 2022; Hoyle et al., 2023)**：
  用 LM 描述文档以挖掘隐含信息，最接近的先驱工作
- **计算人文 (Piper, 2018; Underwood, 2019)**：
  远读（distant reading）传统，Retell 为远读提供了新工具

## 评分

- 新颖性: ⭐⭐⭐⭐
  "show to tell" 隐喻转化为方法论极具巧思，抽象复述加 LDA 的组合简洁新颖
- 实验充分度: ⭐⭐⭐⭐
  多模型多动词多 k 值的系统比较，众包加专家加案例研究多层评估
- 写作质量: ⭐⭐⭐⭐⭐
  跨学科论文写作范例，动机清晰，叙述流畅，伦理讨论周全
- 价值: ⭐⭐⭐⭐
  对人文计算和文化分析有直接应用价值，方法简洁易推广至其他叙事文本分析

<!-- RELATED:START -->

## 相关论文

- [Theme-Explanation Structure for Table Summarization Using Large Language Models](theme-explanation_structure_for_table_summarization_using_large_language_models_.md)
- [An Empirical Study of Many-to-Many Summarization with Large Language Models](an_empirical_study_of_manytomany_summarization.md)
- [Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing](unveiling_attractor_cycles_in_large_language_models_a_dynamical_systems_view_of_.md)
- [FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](../../ICLR2026/nlp_generation/fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)
- [A Representation Level Analysis of NMT Model Robustness to Grammatical Errors](a_representation_level_analysis_of_nmt_model_robustness_to_grammatical_errors.md)

<!-- RELATED:END -->
