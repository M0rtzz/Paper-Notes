---
title: >-
  [论文解读] Stop Automating Peer Review Without Rigorous Evaluation
description: >-
  [ICML 2026][LLM/NLP][AI 审稿] 这是一篇立场论文：作者通过对 ICLR 2026 真实评审和 60 篇模拟评审的实证测量，发现当前 LLM 审稿存在 hivemind（高度趋同）+ paper laundering（零样本改写就能涨 0.45 分）两大失效，因此论证「在没有严格评估之前…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "AI 审稿"
  - "Hivemind 效应"
  - "Paper Laundering"
  - "算法单一化"
  - "评审多样性"
---

# Stop Automating Peer Review Without Rigorous Evaluation

**会议**: ICML 2026  
**arXiv**: [2605.03202](https://arxiv.org/abs/2605.03202)  
**代码**: 无  
**领域**: LLM / NLP / AI 治理 / 同行评审  
**关键词**: AI 审稿、Hivemind 效应、Paper Laundering、算法单一化、评审多样性

## 一句话总结
这是一篇立场论文：作者通过对 ICLR 2026 真实评审和 60 篇模拟评审的实证测量，发现当前 LLM 审稿存在 hivemind（高度趋同）+ paper laundering（零样本改写就能涨 0.45 分）两大失效，因此论证「在没有严格评估之前，不应让 LLM 直接生成审稿意见」，并呼吁建立一门"审稿自动化的科学"。

## 研究背景与动机

**领域现状**：顶会投稿量爆炸式增长（NeurIPS / ICLR / ICML 每年涨数千篇），合格审稿人池跟不上，于是各大会议开始引入 LLM：AAAI 2025 与人类审稿并行生成 LLM 评审；ICLR 2026 允许审稿人用 LLM；ICML 2026 推出"作者选用"二元政策；ICLR 2025 部署了 review feedback agent。各家政策极度碎片化、缺乏共识。

**现有痛点**：现有关于 AI 审稿的工作几乎全在测"和人类打分的相关性"或"是否倾向于通过 prompt injection 攻击"。这两个维度都不够：相关性高不代表无害（可能错得一致），而 prompt injection 已被会议明文禁止，并非典型威胁模型。真正棘手的是合法但低成本的策略性写作行为，以及 AI 审稿在群体层面的"同质化"。

**核心矛盾**：人类审稿的偏差是**分布式**的（不同 reviewer 偏差方向不同，聚合后部分抵消），而 LLM 审稿的偏差是**中心化**的（同一基座模型训练数据相似 → 错误高度相关）。这就是 Kleinberg & Raghavan 所说的 algorithmic monoculture：每个个体决策看似合理，聚合后整体却变差，且对同一种攻击全军覆没。

**本文目标**：把审稿自动化的"必要条件"从模糊的"和人相关"明确为两条可测量条件——(C1) 评审多样性 / 非趋同；(C2) 非可博弈 / 抗策略性改写——并实证检验当前 SOTA LLM 审稿是否满足。

**切入角度**：作者一边拿 ICLR 2026 全部 75,800 篇真实评审（已带 AI 标签）做"野生数据"分析，一边用 Bianchi et al. 的 AI reviewer agent 对随机 60 篇论文做控制实验，构造 4 prompt × 2 改写模型 × 3 审稿模型 = 24 个条件的完整 grid，把"hivemind"和"laundering"分别量化。

**核心 idea**：用 IntraSim / InterSim 两个嵌入相似度指标度量"hivemind 效应"，再用零样本 LaTeX 改写定义 "paper laundering"，证明 AI 审稿两个必要条件都不满足，从而支持「先建立审稿自动化科学，再谈部署」的立场。

## 方法详解
本文没有提模型，方法即一套"度量 + 攻击"的实证协议，分两大支柱。

### 整体框架
输入：(1) ICLR 2026 全量 75,800 评审 + AI 标签（来自 Emi 2025 的 EditLens 分类器，21% 被判为 AI 生成）；(2) 60 篇随机 ICLR 论文 + AI reviewer agent 生成的合成评审；(3) 同 60 篇论文经零样本 LLM 改写后的"laundered"版本。

输出：两条立场论据——hivemind 效应统计学显著、laundering 平均涨分 +0.45（$p<0.0001$）。

中间环节包含：(a) 用 `text-embedding-3-small` 把每篇 review 编码成向量，计算同篇内 reviewer 间相似度（IntraSim）与不同篇 reviewer 间相似度（InterSim）；(b) 对每篇论文喂入 4 种 zero-shot prompt（含一条命令 launderer 主动 jailbreak 审稿模型的版本），分别用 GPT-5.1 / GPT-5.4 重写整份 LaTeX，再编译回 PDF 喂给审稿 agent；(c) Wilcoxon signed-rank 检验配对分差。

### 关键设计

1. **Hivemind 指标 IntraSim / InterSim**:

    - 功能：把"评审多样性"从模糊概念变成可比的标量，用于跨人类 vs AI、跨真实 vs 模拟两组对照。
    - 核心思路：对论文 $p$ 的所有评审向量集合 $\mathcal{R}(p)$，IntraSim 是两两 cosine 相似度的平均：$\mathrm{IntraSim}(p)=\frac{2}{m_p(m_p-1)}\sum_{i<j}\mathrm{sim}(r_i,r_j)$；InterSim 则跨论文对 $p\neq q$ 求 $\mathrm{sim}(r,r')$ 的双重平均。InterSim 只在"同一类 reviewer 跨不同论文"内比较，避免人类 vs AI 文本风格差异污染信号。
    - 设计动机：单看打分相关性会把"AI 都给高分"和"AI 都说同样的话"混在一起；嵌入空间的成对相似度能直接衡量「第二份评审是否带来新信息」，这正是多 reviewer 制度的核心价值。

2. **Paper Laundering 协议**:

    - 功能：构造一种**完全合规**的"博弈"AI 审稿的攻击（不是 prompt injection，不是隐藏指令），用来检验非可博弈性 C2。
    - 核心思路：把整份论文 LaTeX 源码作为输入，让 launderer LLM 在零样本 prompt 下重写一遍（成本约 $0.25/篇），重新编译成 PDF 后交给 AI reviewer agent；通过 4 prompt × 2 launderer × 3 reviewer 的 24 cell 网格统计 paired 分差。Wilcoxon test 在几乎所有 cell 里都 $p<0.001$，平均涨分 $+0.45$，自夸偏差（self-preference bias）让 GPT 审稿被 GPT 改写涨得更多。
    - 设计动机：现有对抗攻击研究关注 prompt injection，但那已被会议明文禁止；laundering 是"作者明面上承认用 AI 润色文字"，因此即便有效也无法靠政策禁止——这才是真正威胁评审公信力的失效模式。

3. **Monoculture 级联测量**:

    - 功能：把"hivemind"从评审层延伸到论文层，论证若 laundering 被普遍采用，整个学术写作会向 AI 审稿偏好收敛。
    - 核心思路：对 60 篇论文的 abstract+introduction 嵌入做两两 cosine 相似度，共 6,903 对（多次采样），比较原始 vs laundered 版本——后者平均相似度从 $0.497$ 升到 $0.529$，相对涨幅 $+6.5\%$，Cohen's $d=1.02$（大效应）。
    - 设计动机：算法单一化的危害不只是"评审趋同"，更在于"被评审塑造的写作也趋同"，作者用一个简单的嵌入聚合实验把因果链从 reviewer 一路画到 author 的写作行为，强化了立场论据的链路完整性。

### 损失函数 / 训练策略
本文不训练新模型。所有"训练"成本集中在：(i) 调 OpenAI / Anthropic API 跑 60 × 24 = 1,440 个审稿；(ii) 单篇 laundering 成本 ≈ $0.25；(iii) ICLR 全量评审嵌入和相似度计算属于一次性离线开销。

## 实验关键数据

### 主实验
ICLR 2026 真实评审上的 hivemind 效应：

| 数据切片 | InterSim 均值 | $p$ 值 | Cohen's $d$ |
|---|---|---|---|
| 纯 AI 评审 | 0.486 | $<10^{-4}$ | 0.29 |
| 含人类贡献的评审 | 0.467 | — | — |
| 仅取 weaknesses+questions 段 | — | $<10^{-4}$ | 0.35 |

模拟实验（60 篇 ICLR 论文）上 AI vs 人类的趋同程度：

| 设置 | IntraSim | InterSim (vs 人类 0.470) |
|---|---|---|
| 人类 ICLR 评审 | 0.811 | 0.470 |
| GPT-5.1 评 原始论文 | 0.882 (+8.7%) | 0.646 (+37.4%, $d=3.55$) |
| GPT-5.1 评 laundered 版 | 0.891 (+9.8%) | 0.657 (+39.8%, $d=3.79$) |
| Claude 评 原始论文 | — | 0.553 (+17.6%, $d=1.41$) |

Laundering 涨分（1–10 分制，24 cells 平均）：

| 指标 | 数值 |
|---|---|
| 平均 paired 涨分 | $+0.45$ |
| Wilcoxon 显著性 | $p<0.001$（几乎全部 cell） |
| 单篇 laundering 成本 | ≈ $0.25 |
| 论文相似度涨幅 | +6.5% ($d=1.02$) |

### 消融实验

| 切片 / 配置 | 关键发现 | 说明 |
|---|---|---|
| 限制评审段为 weakness+question | IntraSim $d$ 从 1.47 → 1.93 | 趋同不只来自 boilerplate，而是渗透到批判性段落 |
| GPT reviewer + GPT launderer | 涨分最大 | 自夸偏差 (Panickssery 2024) 复现 |
| Claude reviewer | 涨分较小 | 跨家族 launderer 攻击仍生效，但弱 |
| AI vs 人类对最终录用的预测力 | AUC 0.710 vs 0.822 (n=8,015) | 人类打分更能预测 final decision |
| AI–AI 评分相关 vs AI–人类 | $r=0.49$ vs $r=0.15$ | 错得一致而非和人对齐 |

### 关键发现
- AI reviewer 大量复用通用句式：GPT 最常出现的短语 "if not, can you comment on" 出现在 13.3% 论文里，Claude 的 "how does the method handle" 出现在 21.7%，而人类评审里最常见短语都 <1%——直接证据表明 AI 审稿的"问题"几乎与论文无关。
- 涨分主要来自 hedging 词（"may", "typically", "suggests"）与 emphasis 词（"strong", "robust", "consistent"）增多，以及 hallucinated 的结果"额外解读"，并非实质性实验新增。
- AI 评审平均给 GPT 7.3、Claude 6.1，人类平均 4.3，存在显著系统性虚高。

## 亮点与洞察
- **把"AI 审稿好不好"重新定义为两条可测试条件**：相关性思路下永远扯不清，多样性 + 抗博弈这两条把讨论拉回工程基线，研究者立即能拿来跑测。
- **Laundering 是"合规攻击"的优雅样本**：和 prompt injection 完全不同——它正好暴露"会议政策可以禁止越界，但禁止不了零样本润色"这一治理空白，论据迁移性极强。
- **算法单一化在两个层级被分别证伪**：reviewer 层（评审同质化）+ author 层（论文向 AI 偏好趋同），这种"双层 monoculture"框架可复用到推荐系统、自动判分等任何被 LLM 中介化的多评估场景。
- **公开 Appendix H 直接给本文的 AI 评审**——立场论文罕见的"以身试法"，本身就是反讽式 demo。

## 局限与展望
- 作者承认：必要条件不等于充分条件，即使未来某代 LLM 真的非趋同 + 抗博弈，依然要回答问责制、民主合法性等问题。
- 实证局限：60 篇模拟样本量小、AI 标签来自 EditLens 分类器（有 misclassification 风险，作者在 Appendix G.3 验证）；laundering 的"质量是否真的没变"主要靠 word-level 分析判断，缺少人类盲评。
- 改进方向：把"抗博弈"做成 benchmark（laundering robust score）纳入未来 AI reviewer 系统的标准评测；研究 launderer 不止改 style 时（例如加入伪 experiment）的失效边界。

## 相关工作与启发
- **vs Bianchi et al. 2025b（AI reviewer agent）**：本文直接复用其 agent，但把研究问题从"能不能审"转到"应不应该自动化"，并提供反例。
- **vs Goldberg et al. 2024（NeurIPS LLM checklist）**：同样揭示"AI 工具可被博弈"，但本文把博弈门槛降到零样本零优化，威胁更普遍。
- **vs Lin et al. 2025b（targeted adversarial attacks）**：他们要做字符级扰动，本文只需自然语言改写，攻击者门槛低一个数量级。
- **vs Kleinberg & Raghavan 2021（algorithmic monoculture）**：把这一理论框架首次实证用于学术评审场景。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把"评审自动化"问题首次拆成两条可测条件，并提出 paper laundering 这一新型威胁
- 实验充分度: ⭐⭐⭐⭐ 真实数据 + 控制实验 + 24-cell grid 兼顾外/内部效度，但 60 篇规模仍偏小
- 写作质量: ⭐⭐⭐⭐⭐ 立场论文典范，逻辑链清晰、反方观点逐条回应（§5）、附录给出自己被 AI 审稿的结果
- 价值: ⭐⭐⭐⭐⭐ 议题极及时，对会议政策与社区共识有直接影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future](../../ACL2026/llm_nlp/can_ai_be_a_good_peer_reviewer_a_survey_of_peer_review_process_evaluation_and_th.md)
- [\[AAAI 2026\] Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback](../../AAAI2026/llm_nlp/position_on_llm-assisted_peer_review_addressing_reviewer_gap_through_mentoring_a.md)
- [\[ACL 2025\] ATRIE: Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation](../../ACL2025/llm_nlp/atrie_legal_interpretation.md)
- [\[AAAI 2026\] STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model](../../AAAI2026/llm_nlp/stem_efficient_relative_capability_evaluation_of_llms_through_structured_transit.md)
- [\[ICML 2026\] Margin-Adaptive Confidence Ranking for Reliable LLM Judgement](margin-adaptive_confidence_ranking_for_reliable_llm_judgement.md)

</div>

<!-- RELATED:END -->
