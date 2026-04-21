---
title: >-
  [论文解读] Research Borderlands: Analysing Writing Across Research Cultures
description: >-
  [ACL 2025][research culture] 通过访谈跨学科研究者构建学术写作文化规范框架（结构/风格/修辞/引用四类），并用计算指标量化11个CS社区的写作差异，揭示LLM在跨社区写作改编时存在严重的"同质化"倾向。
tags:
  - ACL 2025
  - research culture
  - cultural norms
  - scientific writing
  - LLM evaluation
  - interdisciplinary research
  - cultural competence
---

# Research Borderlands: Analysing Writing Across Research Cultures

**会议**: ACL 2025  
**arXiv**: [2506.00784](https://arxiv.org/abs/2506.00784)  
**代码**: [shaily99/research_borderlands](https://github.com/shaily99/research_borderlands)  
**领域**: others  
**关键词**: research culture, cultural norms, scientific writing, LLM evaluation, interdisciplinary research, cultural competence  
**作者**: Shaily Bhatt (CMU), Tal August (UIUC), Maria Antoniak (Copenhagen)  

## 一句话总结

通过访谈跨学科研究者构建学术写作文化规范框架（结构/风格/修辞/引用四类），并用计算指标量化11个CS社区的写作差异，揭示LLM在跨社区写作改编时存在严重的"同质化"倾向。

## 研究背景与动机

- **现有问题**：LLM文化能力评估中，"文化"概念定义模糊，多数工作依赖国籍、语言等粗粒度代理变量，缺乏与社区成员的深入互动
- **切入点**：学术写作是文化规范的显性载体——不同研究社区（如NLP、HCI、教育学）对论文结构、用词、论证方式有截然不同的隐性期望
- **核心动机**：以"以人为中心"(human-centered)的方式发现和度量文化规范，而非自上而下地预设代理变量；同时评估LLM作为写作工具时能否适应不同研究社区的文化规范
- **生态效度**：先通过问卷（N=78）验证"跨社区改编论文"是研究者真实面临的任务，再以此为基础开展研究

## 方法详解

### 1. 定性研究：发现文化规范

- **问卷调查（N=78）**：面向跨学科研究者，确认论文改编是真实需求；66人报告有跨社区改编经验；改编时几乎总会调整Introduction部分
- **半结构化访谈（N=10）**：选取资深跨学科学者，每人60分钟；访谈前要求提供同一论文在不同社区投稿的多个Introduction版本用于对比
- **编码分析**：两位作者独立编码前两份访谈转录，经三周迭代讨论后形成统一规范框架

### 2. 文化规范框架（四大类）

| 类别 | 规范维度 | 示例差异 |
|------|---------|---------|
| **结构规范** | 篇幅长度、表格/图的使用 | NLP论文8-9页 vs FAccT论文14页；CV社区图多，NLP社区表多 |
| **风格规范** | 术语/行话、可读性、正式程度、冗余程度 | NLP社区"RoBERTa"无需解释；教育学用词如"preponderance"；人文学科允许非正式散文 |
| **修辞规范** | 定量证据、比喻性语言、框架（framing）、叙事组织 | ML/NLP重视数字证据；人文侧重故事性叙述；9/10受访者认为"重构框架"是最关键的改编操作 |
| **引用规范** | 经典引用、引用互动风格 | 同一概念在不同社区有不同经典引用（如认知科学"mental models" vs HCI"folk theories"）；人文学科常在开头直接引用原文 |

### 3. 计算化评价套件（Evaluation Suite）

将框架中可量化的规范操作化为计算指标：

- **结构**：词数、句数（NLTK分词）、表/图出现率（正则匹配）
- **风格**：术语专有性（NPMI specificity score）、正式度（DeBERTa-large微调GYAFC）、可读性（Flesch reading-ease）
- **修辞**：定量证据比例（Llama 3.1 70B as judge，与人工标注93%一致）、叙事组织（句子功能分类的位置偏度skew）、价值框架（10维价值向量，lexicon分类器，precision 72.95%）

### 4. LLM文化能力评估

- **任务**：给定来源社区的Introduction，让LLM改编为目标社区风格
- **数据采样**：(a)每对社区随机100篇；(b)每对社区选specificity最高的100篇（更贴近真实改编场景）
- **模型**：GPT-3.5 Turbo、GPT-4o Mini、Llama 3.1 8B、Llama 3.3 70B、Mistral Ministral 8B；每prompt采样5次，共55万次生成

## 实验关键数据

### 11个CS社区写作规范差异（Section 6，Figure 3）

| 指标 | 关键发现 |
|------|---------|
| 篇幅 | Economics & Computation最长，NLP相对较短 |
| 图/表 | CV社区图最多，NLP社区表最多 |
| 术语专有性 | 各社区均为正值，Education社区最具独特性 |
| 正式度 | 社区间差异不大（均为CS子领域） |
| 定量证据 | ML/NLP/AI方差最小，说明定量证据是其强文化规范 |
| 叙事组织 | ML/NLP/AI中目标句(objective)出现位置更靠前；AI社区结果句(results)也更早出现 |

### Table 2：LLM文化能力评估（Section 7，以ML和NLP为目标社区）

| 观察 | 详情 |
|------|------|
| **词汇改编成功** | 所有LLM改编后specificity几乎总是提升，模型确实知道不同社区的词汇差异 |
| **其他维度同质化** | 篇幅总是缩短、表/图提及率总是降低、可读性总是下降、定量证据比例总是略增，仅在目标方向恰好一致时碰巧正确 |
| **叙事组织固化** | 背景句和方法句偏度升高、目标句偏度降低，所有模型趋向统一模板 |
| **框架相似度** | 改编前后framing cosine similarity几乎不变，LLM未能根据目标社区调整价值框架 |

## 亮点

- **方法论创新**：以自下而上(bottom-up)、以人为中心的方式发现文化规范，通过访谈跨学科专家而非预设代理变量，框架四大类涵盖结构/风格/修辞/引用全方位
- **大规模验证**：81,178篇论文、11个CS社区、38个会议的定量分析成功复现了受访者描述的质性观察
- **LLM同质化发现**：首次系统证明LLM在跨社区写作改编中除词汇外全面同质化，55万次生成的大规模实验极具说服力
- **开源评价套件**：提供可复用的计算指标和代码，可用于科学计量学和LLM文化能力评价

## 局限性

- **社区范围**：仅覆盖CS子领域，未涉及社会学、生物学、艺术史等差异更大的学科
- **受访者偏差**：10位受访者主要来自ML/NLP/计算社会科学，通过社交媒体招募存在选择偏差
- **指标局限**：正式度指标在CS社区间区分度不足；verbosity和figurative language因缺乏可靠指标被排除；引用规范因无法映射行内引用而未计算化
- **评估方式**：LLM仅在zero-shot设置下评估，未探索few-shot或RAG等提升文化能力的方法
- **代理选择**：以"社区"而非"会议"为文化单元，虽经问卷验证但可能忽略同一社区内子方向的差异

## 相关工作

- **理解研究社区**：Lucy et al. (2023)分析社区间词汇特异性；Birhane et al. (2022)与Jiang et al. (2025)研究ML研究中编码的价值观；Michael et al. (2023)调查NLP社区信念
- **LLM科研工具**：Si et al. (2024)用于科研创意生成；Robinson et al. (2024)聚焦单一社区写作辅助——均未考虑跨社区文化差异
- **LLM文化能力**：Adilazuarda et al. (2024)综述指出"文化"定义模糊；Rao et al. (2024)用专家文档构建地理文化评测但设置人工；本文以真实任务+社区成员驱动，方法论更接地气
- **LLM写作同质化**：Liang et al. (2024)发现LLM辅助论文更短更相似；Guo et al. (2024)发现语言多样性下降；Xu et al. (2025)发现修辞多样性降低

## 评分

- 新颖性: ⭐⭐⭐⭐ — 以人类学/质性方法切入NLP文化能力评估，视角新颖
- 实验充分度: ⭐⭐⭐⭐ — 81K论文+55万LLM生成，规模大且指标设计有说服力
- 写作质量: ⭐⭐⭐⭐⭐ — 混合方法叙述流畅，质性与定量结合紧密
- 价值: ⭐⭐⭐⭐ — 对LLM科研写作工具的文化适应性提出了重要警示，评价套件可复用

<!-- RELATED:START -->

## 相关论文

- [笔记5：ReSearch - 学习通过搜索推理](../../NeurIPS2025/others/research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)
- [Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus](mapping_the_podcast_ecosystem_with_the_structured_podcast_research_corpus.md)
- [All That Glitters is Not Novel: Plagiarism in AI Generated Research](plagiarism_ai_generated_research.md)
- [IRIS: Interactive Research Ideation System for Accelerating Scientific Discovery](iris_interactive_research_ideation_system_for_accelerating_scientific_discovery.md)
- [Unifying Language Agent Algorithms with Graph-based Orchestration Engine for Reproducible Agent Research](unifying_language_agent_algorithms_with_graph-based_orchestration_engine_for_rep.md)

<!-- RELATED:END -->
