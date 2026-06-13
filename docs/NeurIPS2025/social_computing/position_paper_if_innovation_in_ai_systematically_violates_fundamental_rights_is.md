---
title: >-
  [论文解读] Position Paper: If Innovation in AI Systematically Violates Fundamental Rights, Is It Innovation at All?
description: >-
  [NeurIPS 2025 Oral][社会计算][AI监管] 本文挑战"监管与创新对立"的固有信念，通过制药、航空、福利系统的历史类比和 Collingridge 困境分析论证良好设计的监管是创新的基础而非阻碍，并以 EU AI Act 的监管沙盒、中小企业支持等机制为范例展示监管如何加速而非延缓负责任的技术进步。
tags:
  - "NeurIPS 2025 Oral"
  - "社会计算"
  - "AI监管"
  - "基本权利"
  - "EU AI Act"
  - "负责任创新"
  - "Collingridge困境"
---

# Position Paper: If Innovation in AI Systematically Violates Fundamental Rights, Is It Innovation at All?

**会议**: NeurIPS 2025 Oral  
**arXiv**: [2511.00027](https://arxiv.org/abs/2511.00027)  
**代码**: 无  
**领域**: AI Ethics / AI Governance  
**关键词**: AI监管, 基本权利, EU AI Act, 负责任创新, Collingridge困境

## 一句话总结

本文挑战"监管与创新对立"的固有信念，通过制药、航空、福利系统的历史类比和 Collingridge 困境分析论证良好设计的监管是创新的基础而非阻碍，并以 EU AI Act 的监管沙盒、中小企业支持等机制为范例展示监管如何加速而非延缓负责任的技术进步。

## 研究背景与动机

**领域现状**：AI 已渗透到关键基础设施和决策系统中，其失败产生社会、经济和民主层面的真实伤害。同时，去监管派以"促进创新"为由推动放松 AI 管制——美国新政府撤销了行政命令 14110，AI Action Plan 2025 明确要求"消除繁文缛节和繁重监管"。

**现有痛点**：去监管叙事忽略了 Collingridge 困境的关键洞察——技术早期难以预见所有风险，但一旦深度嵌入社会则变更成本极高甚至不可能。AI 的现实危害已经在发生：2023 年斯洛伐克选举中的深伪音频动摇了民主进程；AI 生成的非同意性内容（包括涉及未成年人的 CSAM）已造成实际伤害（西班牙 Almendralejo 案件、Taylor Swift 深伪事件）；荷兰 SyRI 系统因不透明的 AI 决策系统歧视性地针对低收入社区而被法院裁定违反人权，最终导致政府辞职。

**核心矛盾**：去监管派将"监管 vs 创新"构建为零和博弈，但历史证据表明这是一个虚假二分法。没有一个具有重大公共影响的领域在缺乏监管框架的情况下蓬勃发展。

**本文目标** 从理论和实证两个维度论证：（1）监管和创新不是对立的；（2）EU AI Act 提供了一个可行的基于风险的治理模型；（3）重新定义"创新"——系统性侵犯基本权利的技术不配称为创新。

**切入角度**：跨学科论证——结合科技政策学（Collingridge困境）、经济学（Schumpeter的创造性破坏、Porter假说）、法学（EU AI Act 条文分析）和历史案例研究。

**核心 idea**：监管不是创新的刹车，而是创新得以持久的地基——技术雄心需要民主价值观和基本权利的纪律。

## 方法详解

### 整体框架

本文采用多层次论证结构：历史类比证伪"监管阻碍创新"→ 列举去监管 AI 的具体风险 → 解析 EU AI Act 作为治理范例 → 分析透明度/影响评估/问责制/AI素养作为操作工具 → 回应替代观点。

### 关键设计

1. **虚假二分法的历史驳斥**：

    - 三个关键历史案例：（a）沙利度胺丑闻导致 10,000+ 婴儿严重畸形 → Kefauver-Harris 修正案建立了现代药物安全协议的基础；（b）1959 年美国每次起飞有 1/25,000 致命概率 → FAA 成立和系统性安全法规后改善 1000+ 倍至 1/29,000,000；（c）荷兰 SyRI 自动欺诈检测系统因不透明和歧视性靶向被法院裁定违反人权
    - 核心论点：这些案例中的伤害不是因为有了监管，而是因为缺乏或失败的监管。历史上有效监管往往催生了而非阻碍了重大技术进步

2. **Collingridge 困境与监管时机**：

    - 困境表述："在变更容易时看不到变更的需要；在看到变更需要时变更已变得昂贵、困难和耗时"
    - 在 AI 语境中：AI 在高风险领域的持续部署意味着不良后果可能在造成重大伤害后才被识别
    - 但 Collingridge 本人并不认为该困境不可克服——他倡导适应新证据的前瞻性治理机制
    - EU AI Act 正是对此的回应：通过基于风险的分类 + 监管沙盒等自适应机制，在预防伤害和保持灵活性之间取得平衡

3. **EU AI Act 的具体机制**：

    - **监管沙盒（Art. 57）**：2026年8月前每个成员国强制设立。不是去监管区而是共监管空间——提供法律确定性（迭代式监管指导）、风险缓解（可暂停实验）、免罚保护（遵循计划则无行政罚款）、跨部门协作。西班牙已启动首个国家沙盒
    - **实际测试（Art. 60）**：特定高风险 AI 系统可在知情同意和监督下进行真实世界测试，市场监管机构可否决或停止测试
    - **中小企业支持（Art. 62）**：沙盒优先准入、意识培训、专门沟通渠道、促进标准化参与。微企业（<10人/€200万营收）可简化质量管理体系
    - **基本权利影响评估 FRIA（Art. 27）**：将治理从被动合规转向前瞻性设计，通过事前评估潜在的权利侵害

### 去监管 AI 的具体风险

- **偏见和歧视**：AI 从历史数据中学习并放大种族、性别、社会经济偏见。文献综述识别了 152 项具体的偏见缓解措施
- **不可问责的决策**：缺乏人在环或人在环形同虚设（人类审查者缺乏培训、权限或时间）。"道德外包"现象——将伦理责任推给算法，开发者和部署机构推卸问责
- **诱导性非反思**：自动化系统迫使人类在不质疑作者身份或后果的情况下做决定，侵蚀反思性判断

## 实验关键数据

### 监管影响数据

| 历史案例 | 监管前 | 监管后 | 改善倍数 |
|---------|--------|--------|---------|
| 美国航空致命事故概率 | 1/25,000（1959） | 1/29,000,000（近年） | 1,000x+ |
| 药物安全（沙利度胺后） | 无系统性临床试验 | Kefauver-Harris 强制试验 | 建立现代安全体系 |
| GDPR 影响 | — | 催生 PET 技术（差分隐私、联邦学习） | 技术创新驱动 |

### EU AI Act 认知表

| 感知负担 | 实际收益 |
|---------|---------|
| "监管阻碍创新" | 监管沙盒提供安全实验环境 |
| "合规成本高" | 中小企业支持 + 简化模板 |
| "限制技术进步" | 合规驱动创新（水印、PET 等） |
| "竞争劣势" | 伦理领导力 = 市场差异化优势 |

### 关键发现

- **零和叙事是错误的**：历史上所有具有重大公共影响的领域都因有效监管而非因去监管而蓬勃发展
- **监管可以催生创新**：GDPR 催生了差分隐私和联邦学习等隐私增强技术，AI Act 正在推动水印和版权保护技术发展
- **Porter 假说在 AI 领域适用**：精心设计的监管激励企业以产生社会和市场双重收益的方式创新
- **先行者优势**：主动合规的企业可以塑造新兴市场、捕获先行者优势

## 亮点与洞察

- **历史类比的说服力极强**：航空安全 1000 倍改善、沙利度胺悲剧等案例比任何抽象论证都有力
- **重新定义"创新"的理论价值**：引用 OECD 负责任创新定义，将创新 ≡ 负责任创新，明确拒绝脱离问责和伦理考量的"创新"概念
- **Collingridge 困境的精准运用**：不只引用困境本身，还指出 Collingridge 认为困境可克服——关键在于前瞻性自适应治理
- **Apple 隐私案例**：将用户数据隐私定位为商业战略和伦理要务，成功将合规转化为市场差异化

## 局限与展望

- **欧洲视角偏向**：大量篇幅讨论 EU AI Act，对美国以外的监管路径（中国、日本等）着墨不多
- **缺乏定量实证**：主要依靠历史类比和政策分析，缺乏"有/无监管"对创新产出的统计因果分析
- **对创新的计量偏窄**：主要从企业合规和市场信任角度衡量创新价值，未充分考虑基础研究层面的影响
- **自我选择偏差**：所引案例（航空、制药）都是"监管成功故事"，未讨论监管确实阻碍创新的反面案例
- **对 Wachter 等批评的回应不够深入**：仅简要提及 EU AI Act 的法律漏洞问题

## 相关工作与启发

- **vs Bradford 的"Brussels Effect"**：Bradford 论证欧盟监管通过全球扩散效应成为事实标准，本文在此基础上进一步论证监管本身的创新促进作用
- **vs Draghi 报告**：Draghi 报告警告过度监管阻碍欧盟数字竞争力，本文直接回应并反驳这一立场
- **vs Castro & ITIF**：针对"预防性原则损害 AI 进步"的论点进行逐条反驳

## 评分

- 新颖性: ⭐⭐⭐ 论点本身不新但论证系统性和历史深度值得认可
- 实验充分度: ⭐⭐⭐ Position paper 不需要实验，但可加入更多定量证据
- 写作质量: ⭐⭐⭐⭐⭐ 论证结构严密，层层递进，替代观点的回应公正且有理有据
- 价值: ⭐⭐⭐⭐ 在当前去监管浪潮中提供了重要的反思视角，对政策制定者和 AI 从业者都有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Policy-as-Prompt: Turning AI Governance Rules into Guardrails for AI Agents](policy-as-prompt_turning_ai_governance_rules_into_guardrails_for_ai_agents.md)
- [\[NeurIPS 2025\] IF-GUIDE: Influence Function-Guided Detoxification of LLMs](if-guide_influence_function-guided_detoxification_of_llms.md)
- [\[NeurIPS 2025\] Don't Let It Fade: Preserving Edits in Diffusion Language Models via Token Timestep Allocation](dont_let_it_fade_preserving_edits_in_diffusion_language_mode.md)
- [\[ACL 2026\] Imperfectly Cooperative Human-AI Interactions: Comparing the Impacts of Human and AI Attributes in Simulated and User Studies](../../ACL2026/social_computing/imperfectly_cooperative_human-ai_interactions_comparing_the_impacts_of_human_and.md)
- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)

</div>

<!-- RELATED:END -->
