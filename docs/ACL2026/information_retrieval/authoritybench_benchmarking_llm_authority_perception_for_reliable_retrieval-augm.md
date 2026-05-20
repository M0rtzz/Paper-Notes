---
title: >-
  [论文解读] AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation
description: >-
  [ACL 2026][信息检索/RAG][权威感知] AuthorityBench 用 10K 网页域名（PageRank 真值）+22K 实体（Wikipedia 跨语言 sitelink 真值）+120 RAG 问题构造首个 LLM「权威感知」基准…
tags:
  - "ACL 2026"
  - "信息检索/RAG"
  - "权威感知"
  - "RAG 过滤"
  - "PageRank"
  - "实体流行度"
  - "LLM-as-a-Judge"
---

# AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation

**会议**: ACL 2026  
**arXiv**: [2603.25092](https://arxiv.org/abs/2603.25092)  
**代码**: [Trustworthy-Information-Access/AuthorityBench](https://github.com/Trustworthy-Information-Access/AuthorityBench)  
**领域**: 信息检索 / RAG / 可信 AI  
**关键词**: 权威感知, RAG 过滤, PageRank, 实体流行度, LLM-as-a-Judge

## 一句话总结
AuthorityBench 用 10K 网页域名（PageRank 真值）+22K 实体（Wikipedia 跨语言 sitelink 真值）+120 RAG 问题构造首个 LLM「权威感知」基准，发现 ListJudge / PairJudge + PointScore 输出最准，加入网页文本反而拖后腿，且把权威信号用作 RAG 过滤能把答案准确率最多提 14 个百分点。

## 研究背景与动机

**领域现状**：RAG 已成 LLM 减少幻觉的标配，但生成质量高度依赖检索来源的可靠性。当前 LLM-as-a-Judge 工作集中在「相关性」和「效用」两条线，对「来源是否权威」探究极少。

**现有痛点**：低权威源（如个人博客、营销文）可能输出与 Mayo Clinic 矛盾的医疗建议；RAG 若不能辨别权威差异、单纯按文笔流畅度选答案，就会传播错误信息。已有方法要么手动标注 credibility，要么依赖外部打分，缺乏「LLM 自己能否感知权威」的系统评估。

**核心矛盾**：权威是「内容无关」的来源属性（一个匿名博客即使语言精美也不应被信任），但 LLM 通常被训练成「读什么信什么」；它能否在没有内容指引时仅凭 URL/实体名识别权威，是 RAG 可靠性的关键前提。

**本文目标**：(1) 给出可量化的权威定义；(2) 评估主流 LLM 在多种 prompt 范式下的权威感知能力；(3) 验证「权威过滤」能否改善真实 RAG 答案正确率。

**切入角度**：把权威拆成两条公认代理——源权威（用 Web 图 PageRank）和实体权威（用 Wikipedia 跨语言 sitelink 数量）；这两个代理客观、可扩展、内容无关，是验证 LLM 「先验」感知的好工具。

**核心 idea**：构造三套数据集（DomainAuth / EntityAuth / RAGAuth）+ 三种 LLM-as-a-Judge 范式（Point / Pair / List）+ 两种输出格式（直接排序 vs 打绝对分），系统性地拷问「LLM 能感知权威吗、怎么问最有效、对 RAG 有没有用」。

## 方法详解
> 这是基准 + 评测论文，不是新模型论文。下面把「数据构造 + 评测协议 + RAG 实验」当作方法详解的三个组件来讲。

### 整体框架
- **数据**：DomainAuth（10K 网页域名 + Google Toolbar PageRank 0-9 真值，含细粒度 10 级和粗粒度 5 级两种标签）；EntityAuth（22K 实体跨 Basketball / Movies / Songs 三领域，Wikipedia sitelink 数经对数分箱映射到 0-9 真值）；RAGAuth（120 道 yes/no 问题 × 10 篇高低权威混合检索文档，专挑医疗 / 时事这类容易被误信息污染的话题）。
- **评测**：Listwise（10 项排序，Spearman ρ / Kendall τ）+ Pairwise（5K easy 对 +5K hard 对，paired-preference accuracy）+ RAG 答案准确率。
- **被测 LLM**：Qwen3-8B/14B/32B、Llama-3-8B、Llama-3.1-8B；temperature=0，Qwen3 关闭 thinking。

### 关键设计

1. **三种 Judge 范式 × 两种输出格式**:

    - 功能：穷举可能的 prompt 方式，找出最能激发 LLM 权威感知的设置。
    - 核心思路：**PointJudge** 单条输入打绝对分（按分排序）；**PairJudge** 两两比较，输出可以是 PairRank（直接判赢家）或 PointScore（给每条打绝对分，再做 BubbleSort / AverageScore）；**ListJudge** 整列输入，输出可以是 ListRank（直接给重排列）或 PointScore（给每条打分再排）。fine-grained 时 Pair 用 anchor-based 近似（10 项中选 5 个 anchor），coarse-grained 用全 pair。
    - 设计动机：通过对比这 5 种 setting，分离两类问题——「让 LLM 比较」vs「让 LLM 给绝对值」哪个更好；以及「输出是排序还是分数」哪个更稳定。AverageScore 比 BubbleSort 更稳，因为对非传递判断更鲁棒。

2. **加 / 不加网页文本的对比 (Ctx vs w/o Ctx)**:

    - 功能：测试「权威是否等同于文本质量」。
    - 核心思路：每个 judge 都同时跑「只给 domain name / entity name」和「再加上网页文本片段」两版。
    - 设计动机：如果文本能稳定提升判断，说明权威可被语言风格充分代理；如果反而拖后腿，说明权威是独立信号。结果显示 List / Pair 设置加文本几乎都降分（如 Qwen3-8B PointScore 从 71.35→63.91），但在 hard pair 上文本反而救场（accuracy 大涨 30+ 点）——文本是「结构信号模糊时」的补偿。

3. **Authority-Aware RAG 过滤管线**:

    - 功能：把权威感知放进真实 RAG，看能不能转化为答案准确率。
    - 核心思路：统一用 ListJudge + PointScore 协议对 10 篇文档打分，按三种 criteria 排序后选 top-$k$ 喂给 generator——(a) Relevance Filter（按查询相关性）、(b) Utility Filter（先生成伪答案再按 utility 打分，沿用 Zhang 2024b）、(c) Authority Filter（只看源 URL，不读文档内容）。在 $k\in\{1,3,5\}$ 设置下比答案准确率。
    - 设计动机：用同一 judge 协议保证公平对比；让 Authority Filter 只用 URL 是为了证明「权威信号」纯净——不能借助任何文档内容信息。

### 损失函数 / 训练策略
评测论文，无训练。Spearman / Kendall（listwise）+ paired accuracy（pairwise）+ answer accuracy（RAG）三套指标。

## 实验关键数据

### 主实验：DomainAuth（细粒度 10 级，Spearman ρ %）

| Model | Ctx | PointJudge | List+ListRank | List+PointScore | Pair+PairRank | Pair+PointScore |
|------|------|------|------|------|------|------|
| Qwen3-32B | w/o | 73.72 | 73.63 | 74.41 | 72.10 | **75.28** |
| Qwen3-32B | w/ | 73.57 | 55.85 | 63.10 | 66.32 | 69.93 |
| Qwen3-14B | w/o | 71.97 | 72.02 | 73.09 | 70.21 | **73.43** |
| Llama-3-8B | w/o | 63.87 | 57.53 | 66.08 | 61.05 | 64.83 |
| Qwen3-8B | w/o | 41.97 | 54.01 | 67.11 | 15.18 | **71.35** |

EntityAuth（Basketball, w/o text, Spearman ρ %）：Llama-3-8B PointScore 达 **88.90**，Qwen3-32B 达 85.94——比 DomainAuth 普遍高 10+ 点，说明「实体权威」对 LLM 比「网页权威」更容易感知。

### RAG 答案准确率（RAGAuth, 120 题, %）

| 模型 | k | Relevance | Utility | **Authority** | w/o Filter |
|------|------|------|------|------|------|
| Qwen3-14B | 1 | 51.67 | 60.00 | **76.67** | 58.33 |
| Qwen3-14B | 3 | 45.00 | 66.67 | **75.00** | 58.33 |
| Qwen3-32B | 1 | 63.33 | 65.00 | **70.00** | 55.00 |
| Llama-3-8B | 3 | 41.67 | 52.50 | **64.17** | 50.83 |
| Llama-3.1-8B | 3 | 55.00 | 48.33 | **71.76** | 57.50 |

### 关键发现
- **ListJudge / PairJudge + PointScore 最强**：让模型先「看到上下文」再「打绝对分」效果最好；纯 PointJudge（孤立打分）信号最弱。原因是模型缺乏跨条目的标定基准。
- **ListRank 不如 PointScore**：强迫模型给出完整排序在近平局时容易混乱；分数虽 smooth 但更稳定。
- **模型规模单调改善**：Qwen3 8B → 14B → 32B Spearman 单调上升，说明权威感知是「世界知识」依赖型能力。
- **加网页文本通常拖后腿**（List/Pair 设置降 10-20 点），但**对 hard pair 反而救场**（accuracy 涨 30+ 点）——证明文本是「结构信号缺失时的补偿」而非默认增益，验证了「权威 ≠ 文风」假设。
- **RAG 实验最关键**：Authority Filter 全面优于 Relevance / Utility，最高在 Qwen3-14B 上达 76.67%（比 w/o Filter 提 18.3 点）；说明真实 RAG 误差中有大量是「相关但低权威源」造成的。
- **打分偏差**：所有模型对 .gov / .edu 都打高分（+2.5~+3.3 偏差），对社交媒体也有 +1~+2.4 高估——LLM 已经把 TLD 内化为权威信号，但对结构权威之外的「品牌效应」过度敏感，是潜在的误导源。

## 亮点与洞察
- **首次把「权威感知」单独拎出来作为 LLM 能力维度**，过去这部分被埋在 credibility / trustworthy 等大词里，本文把它操作化为可量化指标。
- **「权威 ≠ 文本质量」的实证**是值得记的洞察——加文本反而降效说明 LLM 在 list/pair 上会被「文风流畅度」误导，是 RAG 反幻觉里被忽略的弱点。
- **ListJudge + PointScore 的工程性价比最高**（一次 forward 给所有文档打分），作者实测把它落地到 RAG 过滤管线，是个直接可复用的实践 cookbook。
- **bias 分析（.gov/.edu 高估、社交媒体高估）**给后续做 calibration 留了清晰靶子。

## 局限与展望
- **PageRank 作为真值的天花板**：niche 但权威的专家博客可能 PR 低；某些政府小站也因外链少被低估。需要更动态、主题相关的权威度量。
- **RAGAuth 规模小（120 题）**：足以演示价值但不足以做跨主题统计；扩到 1K+ 是未来工作。
- 仅评测开源 5 个 LLM，闭源旗舰（GPT-5、Claude 4）未覆盖；权威感知很可能与 RLHF 数据强相关，闭源结论可能不同。
- 「权威」与「权威错误」之间的关系（高权威源也偶发错误，如 Reuters 撤稿）未探讨。

## 相关工作与启发
- **vs 相关性 ranker (RankGPT / Setwise)**：他们做 topical match，本文做 source prior；二者正交，可叠加。
- **vs 效用 ranker (Zhang 2024b)**：utility 看「能否生成答案」，authority 看「该不该信」；RAGAuth 显示两者互补，authority 在医疗/时事更重要。
- **vs Credibility-aware RAG (Pan 2024 / Deng 2025)**：他们靠预标签或交叉一致性，本文测试「LLM 内在知否」并显示无需训练即可获益。
- **vs Fake news detection**：那条线以内容真假为目标，本文把 source authority 作为独立先验拎出来——这是 RAG 流水线里能直接落地的中间信号。

## 评分
- 新颖性: ⭐⭐⭐⭐ 权威作为独立维度被系统评测在 LLM-as-a-Judge 文献里首次出现。
- 实验充分度: ⭐⭐⭐⭐ 3 个数据集 + 5 个模型 + 5 种 judge + 两种粒度 + RAG 端到端，覆盖面广；遗憾是 RAGAuth 规模偏小、闭源模型未测。
- 写作质量: ⭐⭐⭐⭐ 图 1 motivation 一击中要害；ListRank vs PointScore 的解释清晰；附录 bias 分析的细致度尤其值得点赞。
- 价值: ⭐⭐⭐⭐ 给 RAG 工程团队一个简单可落地的 cookbook：用 ListJudge+PointScore 过权威，能直接拿 10-20 点准确率提升。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[ACL 2026\] Reliable Evaluation Protocol for Low-Precision Retrieval](reliable_evaluation_protocol_for_low-precision_retrieval.md)
- [\[ACL 2026\] Optimizing User Profiles via Contextual Bandits for Retrieval-Augmented LLM Personalization](optimizing_user_profiles_via_contextual_bandits_for_retrieval-augmented_llm_pers.md)
- [\[NeurIPS 2025\] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](../../NeurIPS2025/information_retrieval/retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
