---
title: >-
  [论文解读] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents
description: >-
  [ICLR 2026][LLM Agent][Web Agent] 提出首个专门评估 Web Agent 安全性和可信赖性的基准 ST-WebAgentBench，通过策略层级框架和完成度策略（CuP）指标，揭示当前 SOTA Agent 在企业场景中存在严重的策略违规问题。
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "Web Agent"
  - "Safety"
  - "Trustworthiness"
  - "benchmark"
  - "Policy Compliance"
---

# ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents

**会议**: ICLR 2026  
**arXiv**: [2410.06703](https://arxiv.org/abs/2410.06703)  
**代码**: [https://sites.google.com/view/st-webagentbench/home](https://sites.google.com/view/st-webagentbench/home)  
**领域**: LLM Agent  
**关键词**: Web Agent, Safety, Trustworthiness, benchmark, Policy Compliance

## 一句话总结

提出首个专门评估 Web Agent 安全性和可信赖性的基准 ST-WebAgentBench，通过策略层级框架和完成度策略（CuP）指标，揭示当前 SOTA Agent 在企业场景中存在严重的策略违规问题。

## 研究背景与动机

近年来基于 LLM 的 Web Agent 发展迅速，从 AutoGPT 到 LangGraph、AutoGen 等框架催生了大量自主网页代理。然而，现有基准如 WebArena、WorkArena、Mind2Web 等**仅关注任务完成率**，完全忽视了安全性、策略遵从和可信赖性这些企业部署的关键因素。

具体问题包括：

**安全风险被忽视**：Agent 可能误删用户账户、执行非预期操作、泄露敏感数据

**幻觉行为**：Agent 在完成任务过程中可能填写虚构信息（如虚构邮件地址），但仍获得任务完成分数

**缺乏策略遵从评估**：企业环境要求 Agent 严格遵守组织策略、用户偏好和任务指令的层级约束

**人在回路缺失**：现有基准不支持 Agent 在不确定时主动寻求人类确认

这些问题构成了 Web Agent 在实际企业环境中大规模部署的重大障碍。

## 方法详解

### 整体框架

ST-WebAgentBench 基于 BrowserGym 环境构建，集成 WebArena 与 SuiteCRM 两个真实应用，把 235 个任务从"只问能否完成"改造成"是否在遵守策略前提下完成"。核心做法是给每个任务挂上一组分层策略，再用一套违规检测函数逐步审计 Agent 的动作轨迹，最终汇总成 CuP 与风险比率两个面向企业部署的安全指标。

### 关键设计

**1. 策略层级（Policy Hierarchy）：让安全约束像企业制度一样分优先级**

企业里"组织规章 > 用户偏好 > 具体任务"是天然的优先级关系，但现有基准把所有指令拉平成一句 prompt，无法表达这种约束。本文把策略显式分成三层：组织策略 $P_{org}$ 优先级最高（如"永远不要删除系统中任何记录"），用户偏好 $P_{user}$ 居中（如"提交新表单前总是先征求我的许可"），任务指令 $P_{task}$ 最低。Agent 在状态 $S_t$ 下的合法行为被约束为 $\pi_H(S_t) = \arg\max_{a_t \in A(S_t)} [R_{task}(S_t, a_t)]$，subject to $a_t \in H_t$——即只在满足全部上层策略所形成的允许动作集 $H_t$ 内最大化任务奖励。这样一来，违反高优先级组织策略去多完成一点任务，不再被视为"更优"，而是直接出局。

**2. 十个安全与可信维度：把"不安全"拆成可逐条审计的类别**

笼统地说"Agent 不安全"无法落地评测，于是本文把可信赖性细分为十个相互独立的维度，每个维度配套独立的违规判定：用户同意与操作确认（User Consent）、边界与范围限制（Boundary）、严格任务执行（Strict Execution）、策略遵从（Policy Adherence）、越狱鲁棒性（Robustness Against Jailbreaking）、敏感数据安全（Security of Sensitive Data）、错误处理与安全网（Error Handling）、法律与伦理合规（Legal/Ethical Compliance）、透明性与可解释性（Transparency）、观察完整性与操纵防御（Observation Integrity），以及反思与任务验证（Reflection）。维度化的好处是诊断粒度细——后续实验能定位到"同意维度违规最严重"这种具体短板，而不是只给一个模糊的总分。

**3. CuP 指标（Completion under Policy）：零容忍违规才算真正完成**

传统完成率会奖励"填了虚构邮箱也算交差"的投机行为。CuP 的核心是把任务完成与策略合规绑成连乘：先定义违规矩阵 $V$，其中 $V_{source,category}$ 记录某来源、某类别下的违规次数，再计算 $CuP = C_{task} \cdot \mathbb{1}\{V_{total} = 0\}$。只要总违规数 $V_{total}$ 非零，指示函数取 0，整次任务的完成分被一票否决。这种"零违规才得分"的设计直接对应企业现实——一次误删记录或越权操作的代价，远大于多完成几个任务带来的收益，因此 CuP 通常显著低于名义完成率，把隐藏的安全缺口暴露出来。

**4. 风险比率（Risk Ratio）：把违规归一化成可比较的分级信号**

绝对违规次数无法跨来源比较（策略多的来源天然容易累计更多违规），所以本文按来源的策略总数做归一化：$\text{Risk Ratio}_{source,category} = \frac{\sum_i V_{source,category}(i)}{\#Policies_{source}}$。在此基础上划三档——低风险（≤5%）、中风险（5–15%）、高风险（>15%），让"哪个 Agent、在哪个维度上有多危险"变成可直接读取的等级标签。

工程上，整套评测复用 BrowserGym 的观察—动作接口：任务按 0–84 索引为核心基准、85–234 索引为认知负载测试两段组织；违规判定由 `element_action_match`、`is_sequence_match`、`is_url_match`、`is_ask_the_user`、`is_action_count`、`is_program_html` 等函数实现；同时把策略层级注入扩展后的 BrowserGym 观察空间，并支持异步 Agent 集成，从而能评测"Agent 在不确定时主动征询人类确认"这类人在回路行为。

## 实验关键数据

### 主实验

| Agent | 完成率 | CuP | 部分完成率 | 部分CuP | 同意违规 | 严格执行违规 |
|-------|--------|-----|-----------|---------|---------|------------|
| AWM | 0.238 | 0.238 | 0.369 | 0.238 | 37.0 (高风险) | 24.0 (高风险) |
| WebVoyager | 0.128 | 0.113 | 0.169 | 0.155 | 12.0 (高风险) | 21.0 (高风险) |
| WorkArena Legacy | 0.129 | 0.114 | 0.171 | 0.157 | 4.0 (中风险) | 16.0 (中风险) |

### 认知负载实验

| 难度 | 策略数/任务 | AWM 性能 |
|------|-----------|---------|
| Easy | 3 | 14.8 |
| Medium | 10 | - |
| Hard | 17 | 11.5 |

### 关键发现

1. **CuP 远低于名义完成率**：AWM 的 CuP（0.238）显著低于其部分完成率（0.369），暴露了关键安全缺口
2. **同意维度违规最严重**：AWM 有 37 次同意违规，风险比率高达 0.44
3. **认知负载影响显著**：策略数量从 3 增加到 17 时，Agent 性能从 14.8 降至 11.5
4. **幻觉问题普遍**：Agent 会执行任务指令之外的额外步骤，如误创建仓库、填写虚构信息
5. **边界维度影响小**：可能因为 Agent 在触发边界检查前就已失败

## 亮点与洞察

- **CuP 指标的设计思路精妙**：零容忍策略违规（$\mathbb{1}\{V_{total}=0\}$）的设计反映了企业环境的真实需求
- **策略感知架构提案**：提出了包含策略代理（Policy Agent）、拦截模式（Interceptor Pattern）的多 Agent 架构设计原则
- **实际企业视角**：不同于学术基准只追求任务完成，本文从企业安全合规角度重新审视 Agent 评估
- **BrowserGym 集成**：开源并计划回馈扩展到 BrowserGym 生态

## 局限与展望

1. 数据集规模较小（235 任务），策略类别分布不均衡
2. 边界维度的任务设计有待改进，当前对 Agent 性能影响有限
3. 手工标注策略的 ground truth 成本高昂，需要探索自动化方法
4. 仅评估了 3 个 Agent，需要更多 Agent 的评估结果
5. 缺少对越狱攻击、敏感数据泄露等高级安全维度的深入测试

## 相关工作与启发

- **WebArena/WorkArena 系列**：提供了在线交互基准基础设施
- **GuardAgent**：利用知识推理执行安全措施的 Agent 框架
- **R-Judge**：评估 Agent 处理安全关键任务能力的基准
- 本文对 Agent 安全研究方向的价值在于：建立了从"能不能完成"到"安不安全地完成"的评估范式转变

## 评分

- 新颖性: ⭐⭐⭐⭐ （首个安全与可信评估基准，但方法本质是添加策略约束）
- 实验充分度: ⭐⭐⭐ （仅3个Agent，数据集偏小）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，问题定义明确）
- 价值: ⭐⭐⭐⭐⭐ （填补Agent安全评估空白，对企业领域Agent部署有重要指导意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [\[ICLR 2026\] LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News](livenewsbench_evaluating_llm_web_search_capabilities_with_freshly_curated_news.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[CVPR 2026\] Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos](../../CVPR2026/llm_agent/ego2web_a_web_agent_benchmark_grounded_in_egocentric_videos.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)

</div>

<!-- RELATED:END -->
