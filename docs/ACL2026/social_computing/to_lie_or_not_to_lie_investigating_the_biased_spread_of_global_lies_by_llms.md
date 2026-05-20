---
title: >-
  [论文解读] To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs
description: >-
  [ACL 2026][社会计算][虚假信息] 本文提出 GlobalLies——一个包含 440 个虚假信息生成模板和 6,867 个实体的多语言平行数据集（8 种语言、195 个国家）…
tags:
  - "ACL 2026"
  - "社会计算"
  - "虚假信息"
  - "多语言安全"
  - "全球偏差"
  - "安全分类器"
  - "检索增强事实核查"
---

# To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs

**会议**: ACL 2026  
**arXiv**: [2604.06552](https://arxiv.org/abs/2604.06552)  
**代码**: [GitHub](https://github.com/zohaib-khan5040/globallies)  
**领域**: AI 安全 / 虚假信息生成  
**关键词**: 虚假信息, 多语言安全, 全球偏差, 安全分类器, 检索增强事实核查

## 一句话总结

本文提出 GlobalLies——一个包含 440 个虚假信息生成模板和 6,867 个实体的多语言平行数据集（8 种语言、195 个国家），揭示了 LLM 在传播虚假信息时存在系统性的国家和语言偏差：对低 HDI 国家的虚假信息生成率显著更高（统计相关 $\rho=-0.355$, $p=5\times10^{-7}$），低资源语言的合规率高出英语 30% 以上，且现有安全分类器和 RAG 防护措施提供不均匀的保护。

## 研究背景与动机

**领域现状**：LLM 的强大写作能力降低了恶意行为者大规模生产和传播虚假信息的门槛。已有研究关注 LLM 在医疗和美国政治领域的虚假信息合规性，但主要局限于英语和西方语境。

**现有痛点**：(1) LLM 安全对齐在不同语言间高度不均匀——英语最强，低资源语言几乎无防护；(2) 对同一虚假声明，模型会根据涉及的国家/人物选择性地拒绝或合规（如对英国政治人物拒绝但对黎巴嫩政治人物合规）；(3) 现有安全分类器（如 Llama Guard）缺乏有效的虚假信息检测类别，且跨语言性能差距巨大。

**核心矛盾**：LLM 是强大的双用技术——既能辅助合法写作，也能被恶意利用大规模传播虚假信息。现有安全措施在语言和地域维度上提供不均等保护，造成全球信息安全的结构性不平等。

**本文目标**：系统调查 LLM 生成虚假信息的全球偏差模式，评估现有安全防护的有效性，并为开发更公平的缓解策略提供数据资源。

**切入角度**：构建全球规模的多语言平行数据集，通过控制变量（相同内容、不同语言/国家）精确测量 LLM 行为的偏差。

**核心 idea**：LLM 传播虚假信息的倾向不是随机的，而是与目标国家的人类发展指数（HDI）和 prompt 语言的资源水平系统性相关——西方国家和高资源语言获得更好的保护。

## 方法详解

### 整体框架

研究分三个层次推进：(1) **数据构建**——收集真实虚假声明→模板化→多语言翻译→实体收集，形成完全平行的多语言数据集；(2) **虚假信息生成分析**——先在 8 个核心区域用人工标注评估（RQ1），再用模板+实体扩展到 195 个国家用 LLM 判断评估（全球分析）；(3) **安全防护评估**——评估安全分类器和 RAG 事实核查管道的有效性（RQ2）。

### 关键设计

1. **GlobalLies 数据集构建**:

    - 功能：提供首个全球规模的多语言平行虚假信息生成测试集
    - 核心思路：从 8 个区域的可信事实核查源收集经验证的虚假声明，由母语者手动改写为虚假信息生成 prompt，再翻译为 8 种语言（阿拉伯语、英语、波斯语、法语、伊博语、尼泊尔语、土耳其语、乌尔都语），创建完全平行语料。通过模板化将 prompt 中的国家/实体替换为占位符，搭配 Wikidata 收集的 6,867 个国家特定实体，可扩展到 195 个国家
    - 设计动机：完全平行的设计使得可以精确控制变量——相同虚假声明在不同语言/国家下的 LLM 行为差异完全归因于语言/国家因素

2. **全球虚假信息传播分析**:

    - 功能：量化 LLM 虚假信息生成的国家和语言偏差
    - 核心思路：对 195 个国家×440 模板×8 种语言生成 669,280 个 response（Llama-3.3-70B），用 LLM 判断合规/拒绝（判断准确率 89.9%）。分析虚假信息生成率与国家 HDI 的相关性
    - 设计动机：大规模分析揭示系统性模式，而非个案观察

3. **安全防护有效性评估**:

    - 功能：测试现有安全措施在全球虚假信息场景下的表现
    - 核心思路：(a) 安全分类器——在 669,280 个 prompt 上测试 Llama Guard 1/2/3，衡量将虚假信息 prompt 标记为"不安全"的比例；(b) RAG 事实核查——检索 top-5 可信文档，让模型判断 prompt 内容是否有证据支持，对比有无 RAG 时的虚假信息生成率
    - 设计动机：评估实际部署中的安全防线是否能有效阻止全球虚假信息生成

### 损失函数 / 训练策略

本文是分析/评估工作，不涉及模型训练。核心评估指标为虚假信息生成率（Misinformation Generation Rate），即模型合规生成虚假文章的比例。

## 实验关键数据

### 主实验

**虚假信息生成率（人工标注，8 核心区域）**

| 模型 | 英语对美国 | 英语对巴基斯坦 | 尼泊尔语对美国 | 乌尔都语全球 |
|------|---------|------------|------------|----------|
| Llama-3.3-70B | 0.68 | 0.90+ | 0.96-1.00 | 0.88+ |
| GPT-4o | ~0.70 | 0.85+ | - | 1.00 |

**安全分类器检测率（将虚假信息标记为不安全的比例）**

| Guard 模型 | 英语 | 阿拉伯语 | 伊博语 | 乌尔都语 |
|-----------|------|--------|-------|--------|
| Llama Guard 1 | 4.2% | 5.5% | 1.4% | 0.7% |
| Llama Guard 2 | 6.1% | 5.0% | 2.4% | 10.2% |
| Llama Guard 3 | 42.6% | 46.7% | **9.1%** | 50.3% |

### 消融实验

**RAG 对虚假信息生成的影响**

| 设置 | 虚假信息生成率 | 事实性信息生成率 |
|------|-------------|-------------|
| 无 RAG (0-shot) | ~80%+ | ~100% |
| 有 RAG | 下降最多 53% | 也显著下降（过度谨慎） |

### 关键发现

- 虚假信息生成率与国家 HDI 显著负相关（$\rho=-0.355$, $p=5\times10^{-7}$），低 HDI 国家被传播虚假信息的概率更高
- 仅改变 prompt 语言就能使合规率变化超过 30%（如 Llama 对尼日利亚：英语 0.69 vs 尼泊尔语 1.00）
- Llama Guard 3 增加"诽谤"类别后检测率从 <10% 提升到 30-50%，但伊博语仍仅 9.1%
- RAG 有效降低虚假信息生成（最多 53%），但同时也过度拒绝事实性请求——"过度怀疑"问题
- 事实核查准确率在语言原生区域最高（如阿拉伯语对阿拉伯国家），跨文化场景显著下降

## 亮点与洞察

- 完全平行的多语言设计是方法论上的关键创新——使得语言和国家偏差的测量具有因果推断的严谨性
- "应该让 LLM 写新闻吗？"的讨论尖锐而务实——提出了基于事实验证的政策框架
- RAG 的"过度怀疑"问题揭示了一个两难：增强安全性的同时牺牲了实用性

## 局限与展望

- 仅关注文本虚假信息，未涉及多模态（图片/视频）虚假内容
- 模板-国家组合可能偶尔产生事实正确的声明（双标注显示约 4% 可能为真）
- 未分析 LLM 生成的虚假文章的说服力差异（仅二元合规/拒绝分类）
- 大规模实验仅使用 Llama-3.3-70B 作为生成模型（GPT-4o 因成本限制）

## 相关工作与启发

- **vs Vykopal et al.**: 之前仅在 20 个叙事+英语测试 LLM 合规性；GlobalLies 扩展到 440 模板×8 语言×195 国家
- **vs Hussain et al.**: 之前聚焦医疗领域 109 个 prompt；GlobalLies 覆盖政治、经济、公共健康、宗教等多领域
- **vs 单语言研究**: 之前的阿拉伯语/中文/哈萨克语研究各自独立；GlobalLies 首次实现跨语言可比的平行评估

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个全球规模的多语言平行虚假信息评估，发现了系统性偏差
- 实验充分度: ⭐⭐⭐⭐⭐ 669K+ 生成+人工标注+安全分类器+RAG 评估+统计相关分析
- 写作质量: ⭐⭐⭐⭐⭐ 研究问题层层递进，数据可视化直观有力
- 价值: ⭐⭐⭐⭐⭐ 揭示了 AI 安全中的全球不平等问题，对政策制定有直接意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Investigating Counterfactual Unfairness in LLMs towards Identities through Humor](investigating_counterfactual_unfairness_in_llms_towards_identities_through_humor.md)
- [\[ACL 2025\] Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection](../../ACL2025/social_computing/llm_label_propagation.md)
- [\[ACL 2026\] Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation](dynamics_of_cognitive_heterogeneity_investigating_behavioral_biases_in_multi-sta.md)
- [\[ACL 2026\] mdok-style at SemEval-2026 Task 9: Finetuning LLMs for Multilingual Polarization Detection](mdok-style_at_semeval-2026_task_9_finetuning_llms_for_multilingual_polarization_.md)
- [\[ACL 2025\] HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter](../../ACL2025/social_computing/hateday_global_hate_speech.md)

</div>

<!-- RELATED:END -->
