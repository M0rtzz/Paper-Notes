---
title: >-
  [论文解读] Can Community Notes Replace Professional Fact-Checkers?
description: >-
  [ACL 2025][Community Notes] 大规模分析 Twitter/X 社区笔记 66.4 万条，发现社区笔记对专业事实核查的依赖是此前报告的 5 倍（≥5-7%），涉及阴谋论/虚假叙事的内容引用事实核查来源的概率是其他内容的 2 倍，证明高质量社区审核与专业事实核查深度交织、不可替代。
tags:
  - ACL 2025
  - Community Notes
  - 事实核查
  - 虚假信息
  - 社区众包审核
  - Twitter/X
---

# Can Community Notes Replace Professional Fact-Checkers?

**会议**: ACL 2025  
**arXiv**: [2502.14132](https://arxiv.org/abs/2502.14132)  
**代码**: 无  
**领域**: NLP / 社会计算与虚假信息治理  
**关键词**: Community Notes, 事实核查, 虚假信息, 社区众包审核, Twitter/X

## 一句话总结

大规模分析 Twitter/X 社区笔记 66.4 万条，发现社区笔记对专业事实核查的依赖是此前报告的 5 倍（≥5-7%），涉及阴谋论/虚假叙事的内容引用事实核查来源的概率是其他内容的 2 倍，证明高质量社区审核与专业事实核查深度交织、不可替代。

## 研究背景与动机

- **任务定义**：量化 Twitter/X Community Notes（社区笔记）对专业事实核查机构工作的依赖程度，识别依赖事实核查来源的帖子与笔记特征。
- **现实背景**：Meta 2025 年宣布终止与事实核查机构的合作转向社区审核模式，暗示两种策略独立甚至对立；Twitter/X 自 2022 年起全面推行 Community Notes 作为治理虚假信息的主要手段。
- **已有局限**：Kangur et al. (2024) 报告 Community Notes 中仅 1% 引用事实核查来源，但其使用的事实核查机构列表较小，且将新闻媒体的事实核查栏目（如 AP Fact Check）归为"新闻"类，导致严重低估。
- **核心问题**：(RQ1) 社区笔记在多大程度上依赖专业事实核查？(RQ2) 哪些类型的帖子和笔记更依赖事实核查来源？

## 方法详解

### 整体框架

从 Twitter/X 官方下载 2021.1–2025.1 全部 Community Notes 原始数据（150 万条），经语言过滤（去除 52.6 万非英语）→ 去除"不具误导性"笔记（26.8 万）→ 去除广告/垃圾（4.4 万），最终保留 66.4 万条英语笔记。对笔记中 URL 进行 13 类来源分类，再子采样 2.55 万条"有用"笔记抓取对应帖子文本（记为 $\mathcal{S}_\text{text}$），进行主题分析与叙事/阴谋论标注。

### 关键设计

**1. 五步级联 URL 来源分类管道**

解决仅靠域名匹配无法捕获新闻媒体事实核查栏目的问题。按优先级逐步分类：① 域名匹配手动整理的事实核查机构列表（Snopes、PolitiFact、AFP Fact Check 等 30+ 家）；② URL 中搜索"fact-check"变体（捕获 AP News 的 `/fact-checking/` 路径等）；③ 域名匹配作者手动标注的 top-100 常见域名；④ 用 GPT-4o 对剩余域名分类；⑤ GPT-4 失败标为"未知"。最终成功分类 95% 的 URL 到 13 个类别。

**2. 零样本主题分类与人工验证**

对 $\mathcal{S}_\text{text}$ 子集使用 ModernBERT-large-zeroshot 模型，以"Tweet:\<帖子\>; Note:\<笔记\>"拼接形式输入，零样本分类到 13 个主题（健康、政治、科技等）。作者人工评估准确率达 90%，主要错误为 AI 生成图片相关内容被误分到"科技"类。

**3. LLM 驱动的叙事/阴谋论检测**

用 GPT-4o 判断 8K 平衡采样的\<帖子, 笔记\>对是否涉及更广泛虚假叙事或阴谋论。两位作者独立标注 100 对进行验证（一致率 0.88，分歧经讨论解决），模型 F1 = 0.85。另外作者对 400 对进行细粒度人工标注，分析反驳策略（提供缺失上下文 / 质疑来源 / 引用科学证据等）。

## 实验关键数据

### RQ1：社区笔记对事实核查的依赖程度

| 笔记类型 | 引用事实核查来源比例 | 备注 |
|:---------|:---:|:-----|
| 所有英语笔记 | ≥5% | 此前报告仅 1.2% (Kangur et al.) |
| 评为"有用"的笔记 | 7% | 事实核查来源与高质量正相关 |
| 评为"无用"的笔记 | 1% | 低质量笔记很少引用事实核查 |

- 对比 Kangur et al. (2024) 报告的 1.2%，本文发现高达 **5 倍**
- 含事实核查来源的笔记在用户评分中"HelpfulGoodSources"维度显著更高
- 高风险主题（健康、科学、诈骗）中事实核查引用比例更高；科技、体育主题更低

### RQ2：涉及叙事/阴谋论的内容与事实核查的关系

| | 含事实核查来源 | 不含事实核查来源 |
|:---|:---:|:---:|
| 涉及更广泛叙事/阴谋论 | 22% | 11% |
| 不涉及 | 28% | 39% |

- 涉及更广泛叙事/阴谋论的内容引用事实核查来源的概率是其他内容的 **2 倍**
- 400 对人工标注进一步揭示反驳策略差异：涉及复杂叙事时更依赖外部事实核查链接；涉及误导性媒体时更多直接提供反例或缺失上下文
- 事实核查来源主要用于质疑声明来源可信度和提供科学证据，很少用于补充缺失上下文

### 笔记来源分类分布（top-5 类别）

| 来源类别 | 所有笔记占比 | "有用"笔记占比 |
|:---------|:---:|:---:|
| 新闻 | 最高 | 最高 |
| 社交媒体 | 较高 | 较高 |
| 参考资料 | 中等 | 中等 |
| 事实核查 | ≥5% | 7% |
| 学术 | 较低 | 较低 |

## 亮点与洞察

1. **政策回应性**：直接用数据回应 Meta 终止事实核查合作的决策——社区审核与专业事实核查是共生关系而非替代关系，削弱事实核查将连锁削弱社区笔记质量
2. **方法论改进**：五步级联分类管道比简单域名匹配多发现 5 倍事实核查引用，揭示此前研究的系统性低估
3. **共生机制**：专业事实核查做深度调查研究 → 社区笔记引用并传播研究成果 → 形成信息治理生态闭环
4. **partisan 困境**：仅 11% 社区笔记达到"有用"状态（需跨观点共识），平均耗时 15.5 小时，党派性议题效率尤低

## 局限与展望

1. 仅分析英语笔记（排除 50 万+ 非英语笔记），结论可能偏向英语圈公共话语
2. 大部分笔记无法获取原始推文文本（仅子集 $\mathcal{S}_\text{text}$ 有帖子文本），深度分析受限
3. 人工标注规模有限（400 对细粒度标注、100 对验证集），可用众包扩展
4. 未区分社区笔记写作者的专业背景——部分可能本身就是事实核查从业者
5. 阴谋论判定标准基于西方科学家视角，可能存在文化偏差

## 相关工作与启发

- **Community Notes 分析**：Pröllochs (2022) 分析了来源可信度与"有用"评级的关系；本文深化了事实核查维度的定量分析
- **事实核查生态**：Graves & Anderson (2020) 研究了平台与事实核查机构的合作模式；本文从用户端补充了依赖证据
- **众包验证**：Martel et al. (2024) 证明众包可有效识别虚假信息；Zhao & Naaman (2023) 发现业余核查者在医学等专业领域倾向于参考专业事实核查
- **启发**：社区驱动的知识验证系统（学术同行评审、Wiki 编辑等）或许也存在类似的对专业审核的隐性依赖

## 评分

- **新颖性**: ★★★★☆ — 首次系统性量化社区笔记对事实核查的依赖程度，揭示 5 倍低估
- **技术深度**: ★★★☆☆ — 以统计分析和 LLM 标注为主，无新模型或新算法
- **实验充分性**: ★★★★☆ — 66.4 万条大规模数据 + 人工验证 + 多角度多粒度分析
- **实用性**: ★★★★★ — 对社交平台虚假信息治理政策有直接参考价值

<!-- RELATED:START -->

## 相关论文

- [DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts](../../ICML2025/social_computing/defame_dynamic_evidence-based_fact-checking_with_multimodal_experts.md)
- [Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System](../../AAAI2026/social_computing/fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)
- [Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation](llm_personalized_disinformation.md)
- [Detection of Human and Machine-Authored Fake News in Urdu](detection_of_human_and_machine-authored_fake_news_in_urdu.md)
- [How does Misinformation Affect Large Language Model Behaviors and Preferences?](how_does_misinformation_affect_large_language.md)

<!-- RELATED:END -->
