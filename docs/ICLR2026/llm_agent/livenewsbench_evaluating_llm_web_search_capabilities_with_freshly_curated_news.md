---
title: >-
  [论文解读] LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News
description: >-
  [ICLR 2026][LLM Agent][LLM Web Search] 提出 LiveNewsBench，一个自动从近期新闻生成的、定期更新的 benchmark，通过多跳、事实性问答评估 LLM 的 agentic web search 能力，有效分离模型内部知识与检索能力…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "LLM Web Search"
  - "Agentic Search"
  - "benchmark"
  - "新闻QA"
  - "多跳检索"
---

# LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News

**会议**: ICLR 2026  
**arXiv**: [2602.13543](https://arxiv.org/abs/2602.13543)  
**代码**: [https://livenewsbench.com](https://livenewsbench.com)  
**领域**: LLM Agent  
**关键词**: LLM Web Search, Agentic Search, benchmark, 新闻QA, 多跳检索

## 一句话总结
提出 LiveNewsBench，一个自动从近期新闻生成的、定期更新的 benchmark，通过多跳、事实性问答评估 LLM 的 agentic web search 能力，有效分离模型内部知识与检索能力，性能范围从 11% 到 90%，展现出强区分力。

## 研究背景与动机
**领域现状**：配备 agentic web search 的 LLM（如 GPT-5.2、DeepSeek V3.2）在需要实时信息的任务中表现出色，但评估这些系统面临根本性挑战。

**现有痛点**：现有 benchmark 分三类——(a) 学术推理 benchmark（如 HLE）实际测的是领域知识而非搜索能力；(b) 事实型 QA（如 SimpleQA、BrowseComp）使用静态问答对，模型可通过记忆直接作答（GPT-5.2 在 SimpleQA 上无网络达 62.5%）；(c) Deep Research benchmark 依赖主观评分指标（完整性、洞察力），无法验证事实正确性。

**核心矛盾**：当 LLM 的训练数据日益丰富，静态 benchmark 越来越难区分"模型靠记忆答对"还是"模型真正搜索到了答案"。时间敏感型 benchmark（FreshQA、SealQA）虽然让答案随时间变化，但问题本身固定且简单，前沿模型不联网仍能在 FreshQA 上达 74.3%。

**本文目标**：如何设计一个可以持续更新、抗记忆污染、要求多跳搜索、且具有客观事实答案的 LLM web search 评估基准？

**切入角度**：利用维基百科当前事件存档为"种子"，自动抓取近期新闻文章，通过 LLM 生成跨多篇文章的多跳问答对。问题的答案在模型训练截止日期之后才出现，从根本上限制记忆。

**核心idea**：全自动、定期刷新、多跳新闻QA pipeline + 人工验证子集 = 持续有效的 agentic search benchmark。

## 方法详解

### 整体框架
LiveNewsBench 想做一件事：用"模型没见过、又能客观核对"的新闻问题，把 LLM 的真实检索能力从它的记忆里剥离出来。整条流水线分两段。前一段是数据构建——先从近期新闻里抓文章、按事件聚成簇，再从每个簇里生成需要跨多篇文章才能答出的多跳问答对，经多轮自动验证 + 人工抽检后入库，并按事件时间切分数据。后一段是评估——把题目交给一个自定义的 ReAct 风格 agentic search agent，在统一的搜索预算下作答，再用 LLM judge 判分。关键在于，问题的答案出现在模型训练截止之后，所以"靠背"这条路被从源头堵死。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    SEED["Wikipedia Current Events<br/>Archive（事件种子）"]
    A["新闻文章检索与聚类<br/>白名单+时间窗，按事件聚簇（5.3 篇/事件）"]
    B["多跳 QA 对生成与验证<br/>自一致性过滤 + 准则改写 + 人工抽检"]
    C["数据划分与更新策略<br/>按事件时间切 train/val/test，每季度刷新"]
    D["Agentic Web Search 评估框架<br/>ReAct：Search/Visit/Finish，预算 5+5"]
    OUT["GPT-4.1 judge 判分 → 准确率/排名"]
    SEED --> A --> B --> C --> D --> OUT
```

### 关键设计

**1. 新闻文章检索与聚类：把"出题素材"按事件聚簇，天然生出多跳**

要持续产出新题，得先有稳定、可靠的新闻素材源。流水线以 Wikipedia Current Events Archive 为种子——它每天汇总全球重大事件摘要，本身就是高质量、去噪过的事件清单。对每个事件，用 GPT-4.1 把摘要改写成搜索查询，再经 Brave Search API 拉回相关新闻文章。为控质量，检索施加两道闸门：约 100 家可靠新闻源的白名单，以及"事件日期前 3 天到后 11 天"的时间窗口；最后再用 GPT-4.1 逐篇验证文章确实在讲这个事件。平均每个事件能关联到 5.3 篇文章。按事件聚簇是这一步的关键——同一事件下的多篇报道各自补充不同细节，后续生成问答时就能强制"答案散落在多篇文章里"，多跳特性是聚簇结构自带的，而非事后硬凑。

**2. 多跳 QA 对生成与验证：自一致性 + 准则改写，双重过滤又不误杀难题**

有了文章簇，用 GPT-5.1 Thinking 从中生成候选问答对，并通过一份详细 guidelines 约束三件事：问题必须依赖多篇文章的信息（多跳），答案必须事实、客观、简短。难点在于自动生成难免出错题，但又不能用弱模型去过滤——那会把真正难的题一并删掉。这里用了两道自动闸门。一是**自一致性过滤**：同一模型对同一题独立作答两次，只有两次答案一致的才保留，借此剔除模糊或有歧义的题；它比"换个弱模型核对"更稳，因为不会因为弱模型答不上来就误判题目无效。二是**准则遵循验证**：用一份改写过措辞的 guidelines 重新评估同一道题，措辞扰动能暴露那些只是恰好符合原始模板、实则不够严谨的题，提升鲁棒性。两道自动闸门之后还有人工抽检，人工拒绝率约 15%，独立标注员之间的一致率达 92%，说明自动 pipeline 留下的题质量已相当可靠。

**3. 数据划分与更新策略：按事件时间切分，让测试集永远是"最新的"**

抗记忆不只靠出题，还靠划分方式。train/val/test 不是随机切，而是严格按新闻事件的时间序：最近两个月的事件进测试集（模型见过它们的可能性最低），第三个月进验证集，更早的进训练集。这样测试集对任何固定截止日期的模型而言都是"未来"，记忆污染被结构性地压到最低。当前版本规模为训练集 600+、验证集 170、测试集 340（其中人工验证子集 200）。更重要的是这套划分支持滚动更新——论文承诺每季度刷新一次，单版构建成本约 \$700，使 benchmark 能长期保持"模型没背过"的有效性，而不是发布即过期。值得一提的是，流水线顺带产出的 600+ 条带客观答案的训练数据也一并开源，可直接用于 agentic search 模型的 RLVR（可验证奖励强化学习）训练，填补该方向训练数据稀缺的空白。

**4. Agentic Web Search 评估框架：自定义 ReAct + 受控搜索预算，保证公平可比**

评估端没有直接套用 LangChain 等通用开源 agent 框架——它们是为生成深度研究报告设计的，面对简短事实问答会消耗过多 token，且各家行为差异大、不可比。LiveNewsBench 自建了一个轻量的 ReAct 风格框架，每一步 agent 只能从三个动作里选：Search（发一条搜索查询）、Visit（读取某网页全文）、或 Finish&Answer（给出最终答案）。标准配置把预算卡在最多 5 次 Search + 5 次 Visit，底层用 Tavily Search API；最终答案由 GPT-4.1 以 SimpleQA 风格 prompt 作 judge 判对错。统一动作空间 + 固定搜索预算，使不同模型在同一条件下比较检索能力，而不是比谁舍得烧更多 token。

## 实验关键数据

### 主实验

| 模型 | 推理模型? | 开源? | 平均搜索数 | 平均访问数 | 准确率(%) |
|------|----------|-------|-----------|-----------|----------|
| GPT-5.2 Official API | ✓ | ✗ | N/A | N/A | **90.0** |
| DeepSeek V3.2 Thinking | ✓ | ✓ | 3.3 | 2.6 | 84.5 |
| DeepSeek V3.2 (No Think) | ✗ | ✓ | 3.4 | 2.6 | 83.0 |
| Claude Sonnet 4.5 | ✗ | ✗ | 2.9 | 1.3 | 82.0 |
| Grok 4 | ✓ | ✗ | 2.7 | 1.7 | 82.0 |
| GPT-5.2 | ✓ | ✗ | 2.9 | 1.8 | 74.0 |
| GPT-4.1 | ✗ | ✗ | 1.7 | 0.6 | 72.5 |
| Gemini 3 Pro | ✓ | ✗ | 3.4 | 0.6 | 60.5 |
| Kimi K2 Thinking | ✓ | ✓ | 2.9 | 1.1 | 48.0 |
| Llama 3.1 8B | ✗ | ✓ | 3.9 | 0.3 | 11.0 |

### 消融实验（搜索预算影响）

| 模型 | Budget=1 | Budget=3 | Budget=5 | Budget=7 | 提升(1→7) |
|------|---------|---------|---------|---------|----------|
| DeepSeek V3.2 Thinking | 48.5 | 80.5 | 84.5 | 84.5 | +36.0 |
| DeepSeek V3.2 (No Think) | 20.0 | 79.0 | 83.0 | 84.5 | +64.5 |
| Claude Sonnet 4.5 | 53.5 | 79.0 | 82.0 | 67.0 | +13.5 |
| GPT-5.2 | 62.5 | 72.5 | 74.0 | 74.5 | +12.0 |
| Kimi K2 Thinking | 7.5 | 47.0 | 48.0 | 52.0 | +44.5 |

### 关键发现
- **记忆污染极低**：无网络条件下，最强模型（GPT-5.2）准确率仅 21.5%，而有搜索时达 74.0%，差距 52.5%。对比 FreshQA（无网络 72.2%）和 SealQA-Hard（47.4%），LiveNewsBench 有效避免了记忆问题
- **搜索预算对性能至关重要**：Budget 从 1 增到 7，所有模型都有显著提升（4%~64.5%），证实了问题的多跳性质
- **工具调用能力差异巨大**：Kimi K2 Thinking 有 44% 的样本无法正确执行搜索动作，而 Claude Sonnet 4.5 仅 0.5% 失败
- **模型偏好搜索摘要而非全文阅读**：所有模型的搜索次数均高于页面访问次数
- **官方API不一定更好**：GPT-5.2 用官方 API 提升 16%，但 Claude 用官方 API 反降 42%

## 亮点与洞察
- **全自动更新 pipeline 是最大亮点**：从新闻采集、QA 生成到验证全程自动化，成本仅 $700/版，可持续更新。对比 FreshQA 等需要人工出题的 benchmark，可扩展性质的飞跃
- **多层验证确保质量**：自一致性过滤 → 准则遵循验证 → 人工检查，三层漏斗。人工验证子集和全集排名一致，说明自动 pipeline 质量可靠
- **同时提供大规模训练数据**：解决了 agentic search 领域训练数据稀缺的问题，600+ 条开源训练集可用于 RLVR

## 局限与展望
- 尽管设计了抗记忆机制，前沿模型仍可通过推理猜对部分答案（如 GPT-5.2 无网络仍有 21.5%），完全消除记忆影响很难
- 问题来源限于英文新闻和西方媒体（白名单约100家），可能存在地域和语言偏差
- 评估使用 GPT-4.1 作为 judge，judge 本身的局限可能影响评估准确性
- 未研究答案随事件发展变化的时间敏感场景（论文明确标注为 future work）
- 搜索引擎 API（Tavily）的质量会影响结果，但论文未消融不同搜索引擎的影响

## 相关工作与启发
- **vs FreshQA/SealQA**：它们使用固定问题+变化答案，但问题本身已被模型记住。LiveNewsBench 同时更新问题和答案，更彻底地抗记忆
- **vs BrowseComp**：BrowseComp 问题静态且不更新，主要测试搜索深度而非时效性。LiveNewsBench 侧重新闻时效+多跳组合
- **vs RealTimeQA**：同为定期更新，但 RealTimeQA 每次仅约 10 个样本，来自手动整理的新闻网站 trivia。LiveNewsBench 自动化 pipeline 规模大得多（>1000/版）
- **vs Deep Research Bench**：评估长报告的主观质量 vs 评估简短事实答案的客观正确性，互补而非替代

## 评分
- 新颖性: ⭐⭐⭐⭐ 全自动、定期更新、抗记忆的 search benchmark 设计确实新颖，但核心思路（用新闻出题抗记忆）相对直觉
- 实验充分度: ⭐⭐⭐⭐⭐ 13个LLM + 2个API，多个消融（搜索预算、记忆测试、全集vs人工集），分析非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，motivation 充分，与相关工作的对比表格一目了然
- 价值: ⭐⭐⭐⭐ 填补了 agentic search 评估的重要空白，且可持续更新，对社区有长期价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](mc-search_evaluating_and_enhancing_multimodal_agentic_search_with_structured_lon.md)
- [\[ICLR 2026\] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [\[ACL 2026\] Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems](../../ACL2026/llm_agent/rethinking_reasoning-intensive_retrieval_evaluating_and_advancing_retrievers_in_.md)
- [\[ICLR 2026\] ZeroDayBench: Evaluating LLM Agents on Unseen Zero-Day Vulnerabilities for Cyberdefense](zerodaybench_evaluating_llm_agents_on_unseen_zero-day_vulnerabilities_for_cyberd.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)

</div>

<!-- RELATED:END -->
