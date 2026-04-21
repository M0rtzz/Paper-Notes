---
title: >-
  [论文解读] LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News
description: >-
  [ICLR 2026][LLM Agent][LLM Web Search] 提出 LiveNewsBench，一个自动从近期新闻生成的、定期更新的 benchmark，通过多跳、事实性问答评估 LLM 的 agentic web search 能力，有效分离模型内部知识与检索能力，性能范围从 11% 到 90%，展现出强区分力。
tags:
  - ICLR 2026
  - LLM Agent
  - LLM Web Search
  - Agentic Search
  - benchmark
  - 新闻QA
  - 多跳检索
---

# LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News

**会议**: ICLR 2026  
**arXiv**: [2602.13543](https://arxiv.org/abs/2602.13543)  
**代码**: https://livenewsbench.com (有)  
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
LiveNewsBench 的数据构建分两个阶段：(1) 从网络检索新闻文章并按事件聚类；(2) 从文章聚类中生成 QA 对，经多轮自动验证和人工检查。评估阶段使用自定义的 ReAct 风格 agentic search 框架，控制搜索预算进行公平对比。

### 关键设计

1. **新闻文章检索与聚类**:

    - 功能：从 Wikipedia Current Events Archive 获取全球重大新闻事件摘要，用 GPT-4.1 改写为搜索查询，通过 Brave Search API 检索相关新闻文章
    - 核心思路：采用白名单（约100家可靠新闻源），时间窗口限制（事件日期前3天到后11天），最后用 GPT-4.1 验证文章与事件的相关性。每个事件平均关联 5.3 篇文章
    - 设计动机：按事件聚类使得后续生成的问答可以引用多篇文章，天然形成多跳问题；白名单和时间窗口确保数据质量

2. **多跳QA对生成与验证**:

    - 功能：用 GPT-5.1 Thinking 从文章聚类中生成候选问答对
    - 核心思路：通过详细 guidelines 确保问题需要多篇文章信息（多跳要求），答案事实性、客观性、简短。生成后经 **自一致性过滤**（同一模型独立回答，两次一致才保留）和 **准则遵循验证**（用改写版 guidelines 重新评估）双重自动验证
    - 设计动机：自一致性比用弱模型过滤更好，避免过滤掉难题；改写 guidelines 增加鲁棒性。人工拒绝率约 15%，独立标注员间一致率 92%

3. **数据划分与更新策略**:

    - 功能：按新闻事件时间做 train/val/test 划分
    - 核心思路：最近两个月的事件作为测试集（记忆可能性最低），第三个月作验证集，更早的做训练集。当前版本：训练集 600+，验证集 170，测试集 340（其中人工验证子集 200）
    - 设计动机：时间序划分确保测试集对模型而言是"最新的"；承诺每季度更新，构建成本约 $700/版

4. **Agentic Web Search 评估框架**:

    - 功能：自定义 ReAct 风格框架，每步可执行 Search（发搜索查询）、Visit（访问网页全文）或 Finish&Answer
    - 核心思路：标准配置允许最多 5 次搜索 + 5 次页面访问。使用 Tavily Search API。用 GPT-4.1 作评判（SimpleQA 风格 prompt）
    - 设计动机：通用开源框架（LangChain 等）设计用于深度研究报告而非事实问答，token 消耗过大；控制搜索预算实现公平对比

### 损失函数 / 训练策略
本文为 benchmark 工作，不涉及模型训练。但提供了 600+ 条开源训练数据，可用于 agentic search 模型的 RLVR 训练。

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

## 相关论文

- [MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](mc-search_evaluating_and_enhancing_multimodal_agentic_search_with_structured_lon.md)
- [ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [Judge Reliability Harness: Stress Testing the Reliability of LLM Judges](judge_reliability_harness_stress_testing_the_reliability_of_llm_judges.md)

<!-- RELATED:END -->
