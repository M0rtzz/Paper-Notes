---
title: >-
  [论文解读] Prompts for Public-Sector LLMs Should Be Governed as Commons
description: >-
  [ICML 2026][推荐系统][提示词治理] 这是一篇 position paper：作者主张公共部门用的 LLM 提示词应当像开源 commons 一样被版本化、有出处、可审计、可否决，并用一座北美城市的 443 条社区提示词（增强到 3…
tags:
  - "ICML 2026"
  - "推荐系统"
  - "提示词治理"
  - "公共部门"
  - "commons"
  - "城市 AI"
  - "多元价值聚合"
---

# Prompts for Public-Sector LLMs Should Be Governed as Commons

**会议**: ICML 2026  
**arXiv**: [2606.00873](https://arxiv.org/abs/2606.00873)  
**代码**: 无（position paper + pilot 数据集）  
**领域**: AI 治理 / Position Paper  
**关键词**: 提示词治理, 公共部门, commons, 城市 AI, 多元价值聚合

## 一句话总结
这是一篇 position paper：作者主张公共部门用的 LLM 提示词应当像开源 commons 一样被版本化、有出处、可审计、可否决，并用一座北美城市的 443 条社区提示词（增强到 3,317 条）跑了一个含五种治理状态的 pilot benchmark，给出可证伪的三个预测——治理化提示能改变输出分布、提升可审计性、缩短故障修复时延。

## 研究背景与动机

**领域现状**：公共部门正在用 LLM 起草公文、汇总记录、分流市民诉求、准备公众参与材料。Model Cards、Datasheets、RLHF/Constitutional AI、平台政策这些既有治理工具，都没有覆盖"本地部署时实际在用的那套提示词模板"。

**现有痛点**：在实际部署里，提示词模板通常在团队、外包、供应商之间**非正式**流转，不进政策审查；而提示词本身会编码角色、受众、价值取舍——同一模型同一输入，换一套提示词输出会显著不同。一旦某条提示词成了默认模板，它内嵌的偏好就会被误以为是"模型的结论"或"政策的结论"。

**核心矛盾**：账责被切成三段——模型供应商管权重和系统策略、集成商管提示词和工作流、公共机构承担输出后果但**没有审计轨迹**。提示词层是真正的"配置层"，却没有任何治理基元。

**本文目标**：把提示词当成一类**被治理的工件**（governed artefact）来对待，给出一套可在 Git 类仓库上落地的治理原语（版本、provenance、许可证、否决、配额、申诉），并用一个可证伪的 pilot 证明这些治理状态确实能改变可观测的输出分布与运维指标。

**切入角度**：借用 Ostrom 的 commons 治理理论（边界、监督、冲突解决可以映射为仓库工作流）与开源社区经验（许可证、PR、议题、CHANGELOG），把"提示词"做成一个由社区维护的 commons。

**核心 idea**：**Prompt Commons** = 版本化的、有出处与许可证、有可审计变更日志的提示词模板仓库 + 三档治理状态（open / curated / veto-enabled）+ 一个让冲突显式化的"协商式聚合提示词"。

## 方法详解

### 整体框架
Prompt Commons 不是一个新模型，而是一套**治理协议 + 仓库结构 + 评测协议**：仓库里每条提示词是带 metadata 的纯文本条目（作者群体、地点、价值主张、同意级别、变更日志），变更走 issue / PR 留下时间戳与理由；治理状态有三档可枚举；协商式聚合把多条来自不同利益相关方的提示词当作"提案"，再用一条版本化的"聚合提示词"指挥模型识别共识、列出分歧、提出折中；评测时固定模型、温度、top-$p$、解码长度、评分量表，比较五种方法 M0–M4。整套设计的目的是把"提示词"从一次性输入变成可以被引用、被合规审查、被回滚的**发布工件**。

### 关键设计

**1. 三档治理状态 + 可强制执行的仓库规则：把"谁能改、什么能发、出问题怎么停"映射成 Git 规则。**

commons 治理失败的常见原因是规则只写在文档里、不可执行；Prompt Commons 把治理直接嵌进仓库工作流，分三档可枚举的状态。Open 状态下任何认证用户都能提案，维护者只清理 spam 与基本安全风险；Curated 状态下合并需维护者审、必须填齐 provenance 字段、释放必须满足跨群体/地区的覆盖配额、走 checklist；Veto-enabled 状态在 curated 之上加正式 quarantine——被指定的代表性组织可发"否决记录"把某条提示词暂时下架，并限时复议（accept / modify / reject）。每一档都对应明确的 PR 模板与 CI 检查项，不依赖任何特殊基础设施，于是"治理"变成日常 PR 流程的一部分而不是事后补救。

**2. 协商式聚合提示词：把"如何处理分歧"写成可审计的 artefact，而不是隐藏的 ensemble。**

当多个利益相关方各自提交了带不同价值主张（可达性、安全、成本、气候韧性、程序公平）的提示词时，传统做法要么把分歧硬塞进一条提示词（多数压倒少数），要么用 hidden ensemble（少数派被静默掩盖）。这里改用一条**被版本化、被审计**的"元提示词"，让模型显式识别共识、列出分歧、说明价值张力、给出折中方案或排序的备选项——这与社会选择理论里"聚合规则塑造结果"的思想对齐。pilot 里它实现为 M4：分层抽样 $k=6$ 条按作者群体均衡的提示词 + 一个固定的聚合指令。把聚合规则做成可审计 artefact，受影响群体才能事后追问"聚合是否抹平了少数派关切"。

**3. 可证伪的评测协议（5 种方法 × 3 类指标）：把"治理是否真起作用"变成能被数据反驳的命题。**

position paper 的论证必须可被推翻，所以作者固定一个 instruction-tuned chat LLM（temperature 0, top-$p$ 1, max 256 tokens）、固定 $N=50$ 条街道与公共空间取舍的 contested-choice 小情景、固定三选一标签（车辆通行优先 / 主动交通与可达性优先 / 混合或折中），再比较 M0（单作者提示词）、M1（open commons 随机抽 1 条）、M2（curated 按覆盖约束抽样）、M3（curated + veto）、M4（协商式聚合）。指标分三类：输出分布（折中率、commitment $D=1-p_{\text{mixed}}$）、主观可接受度（6 组共 12 名 raters 的 7 点量表）、运维响应时延（50 起合成事件的 time-to-remediation）。作者明写出三个预测 P1/P2/P3 并声明：若治理化版本在这三类指标上与单作者版本无差异甚至更差，则本文立场被证伪。

### 损失函数 / 训练策略
本文不训练模型；治理协议本身就是"训练"——通过 issue/PR/veto 这套社区流程不断迭代提示词集合。所有数值（提示词长度、词汇熵、token 频率、可接受度、修复时延）都是描述性统计，**不是优化目标**。

## 实验关键数据

### 主实验
pilot 在 443 条人写提示词上展开，经去重、价值保留改写（每条最多 5 条 paraphrase）和情景扩展，得到 3,317 条增强提示词；人写提示词平均 22.6 词（中位 19），增强后 31.7 词，词汇熵从 7.53 bits 涨到 8.39 bits。下表给出"折中率（mixed or compromise）"与 commitment $D=1-p_{\text{mixed}}$ 的对比：

| 方法 | 折中率 (%) | $D$ | 说明 |
|------|-----------|------|------|
| M0 单作者提示词 | 24 | 0.76 | 路线明确但偏窄 |
| M1 open commons | 48–52 | 0.48–0.52 | 折中显著上升 |
| M2 curated commons | 48–52 | 0.48–0.52 | 同上，且覆盖更均衡 |
| M3 curated + veto | 48–52 | 0.48–0.52 | 同上，叠加可控撤回 |
| M4 协商式聚合 | — | 0.49 | 显式识别分歧后给出折中 |

### 消融实验
作者还给出主观可接受度与运维时延：

| 治理状态 | 跨群体平均可接受度 (7 点) | Gini（跨组离散度） | 平均修复时延 |
|---------|-------------------------|--------------------|--------------|
| M0 单作者 | $4.35\pm0.86$ | 0.096 | — |
| M2 curated | $4.92\pm0.44$ | 0.043 | $11.8$ h |
| M3 curated + veto | $5.48\pm0.66$ | — | $5.6$ h |
| Open（仅参考） | — | — | $30.5$ h |

### 关键发现
- 治理状态确实**改变输出分布**：从单作者到 commons，折中率从 24% 跳到 ~50%，且跨群体可接受度上升、离散度下降。这印证了 P1。
- 治理流程能显著**缩短修复时延**：open → curated → veto 一路从 30.5 h 降到 5.6 h（虽然是合成事件流量）。这印证了 P3。
- 折中率本身不是目标——对应急任务过高的折中率反而有害；作者反复强调"指标随任务模式而变"，避免把描述性统计当成规范性目标。

## 亮点与洞察
- 把"提示词"从工程优化对象（prompt engineering）重新定义为**治理面（governance surface）**，是这篇 position paper 最大的视角切换；这一步走通后，社会选择、commons 理论、开源治理的整套词汇都可以平移过来。
- "协商式聚合提示词"把 ensemble 从隐藏方法变成了可审计 artefact——这一招对任何多 stakeholder 系统（不只是公共部门）都可借鉴。
- 主动写出三条可证伪预测（P1/P2/P3），并指明"如果证伪则立场不成立"，这种 ML position paper 的写法很值得推广。

## 局限与展望
- pilot 只用了 1 个 API 模型、$N=50$ 条情景、12 名 raters、1 个城市的招募池；外部效度受限，作者自己也承认这是"易复现的最小可证伪 testbed"，不是普适效应估计。
- 事件响应时延用的是合成 arrival log，不是真实机构的反应时间。
- Prompt Commons 自身可能被资源充裕的利益方"捕获"，所谓的"社区合法性"也可能只是表象——作者用 provenance + 配额 + 否决来缓解，但这条对抗动力学没量化评估。
- 透明披露提示词会扩大攻击面（prompt injection、jailbreak），需要分级访问控制，这块只给了原则未给方案。

## 相关工作与启发
- **vs Model Cards / Datasheets (Mitchell 2019; Gebru 2021)**：那两条线治理"模型/数据集"，本文治理"部署时的提示词集合"，正交且可叠加。
- **vs RLHF / Constitutional AI (Christiano 2017; Ouyang 2022; Bai 2022)**：那条线在训练阶段约束模型全局行为，本文在部署阶段约束本地框架；模型层无法穷尽地方化价值取舍。
- **vs OWASP Top 10 for LLM Applications**：本文把 quarantine 和 rollback 提升为治理 primitive，与安全工程的事件响应工作流接轨。
- **vs social choice for AI alignment (Conitzer 2024; Huang 2025)**：本文把社会选择视角下沉到"聚合提示词"这一可审计 artefact，让 alignment 的多元价值聚合走向运维可落地。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把提示词治理化为 commons 这一定位是少见的清晰主张。
- 实验充分度: ⭐⭐⭐ pilot 易复现但外部效度有限，作者本人也明确点出。
- 写作质量: ⭐⭐⭐⭐ 论点—反驳—证伪条件结构非常工整，符合 ICML position paper 范式。
- 价值: ⭐⭐⭐⭐ 给出可在 GitHub 即刻落地的治理原语，对正在采购 LLM 的公共部门尤其有用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] NeurIPS Should Lead Scientific Consensus on AI Policy](../../NeurIPS2025/recommender/neurips_should_lead_scientific_consensus_on_ai_policy.md)
- [\[ICLR 2026\] Search Arena: Analyzing Search-Augmented LLMs](../../ICLR2026/recommender/search_arena_analyzing_search-augmented_llms.md)
- [\[ACL 2026\] Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework](../../ACL2026/recommender/personalizing_llms_with_binary_feedback_a_preference-corrected_optimization_fram.md)
- [\[ACL 2026\] Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders](../../ACL2026/recommender/bridging_language_and_items_for_retrieval_and_recommendation_benchmarking_llms_a.md)
- [\[AAAI 2026\] Evaluating LLMs for Police Decision-Making: A Framework Based on Police Action Scenarios](../../AAAI2026/recommender/evaluating_llms_for_police_decision-making_a_framework_based_on_police_action_sc.md)

</div>

<!-- RELATED:END -->
