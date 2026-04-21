---
title: >-
  [论文解读] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents
description: >-
  [ICLR 2026][LLM Agent][Web Agent] 提出首个专门评估 Web Agent 安全性和可信赖性的基准 ST-WebAgentBench，通过策略层级框架和完成度策略（CuP）指标，揭示当前 SOTA Agent 在企业场景中存在严重的策略违规问题。
tags:
  - ICLR 2026
  - LLM Agent
  - Web Agent
  - Safety
  - Trustworthiness
  - benchmark
  - Policy Compliance
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

ST-WebAgentBench 基于 BrowserGym 环境构建，集成了 WebArena 和 SuiteCRM 的应用环境，包含 235 个策略增强任务，覆盖多个安全类别。

### 关键设计

1. **安全与可信行为的策略层级（Policy Hierarchy）**

    - **组织策略 P_org**（最高优先级）：如"永远不要删除系统中的任何记录"
    - **用户偏好 P_user**（中等优先级）：如"提交新表单前总是询问我的许可"
    - **任务指令 P_task**（最低优先级）：特定任务的执行指令
    - Agent 的行为必须满足：$\pi_H(S_t) = \arg\max_{a_t \in A(S_t)} [R_{task}(S_t, a_t)]$ subject to $a_t \in H_t$

2. **安全与可信维度（10个评估维度）**

    - 用户同意与操作确认（User Consent）
    - 边界与范围限制（Boundary）
    - 严格任务执行（Strict Execution）
    - 策略遵从（Policy Adherence）
    - 越狱鲁棒性（Robustness Against Jailbreaking）
    - 敏感数据安全（Security of Sensitive Data）
    - 错误处理与安全网（Error Handling）
    - 法律与伦理合规（Legal/Ethical Compliance）
    - 透明性与可解释性（Transparency）
    - 观察完整性与操纵防御（Observation Integrity）
    - 反思与任务验证（Reflection）

3. **CuP 指标（Completion under Policy）**

    - 定义策略违规矩阵 $V$，其中 $V_{source,category}$ 表示特定来源和类别的违规次数
    - 指标计算：$CuP = C_{task} \cdot \mathbb{1}\{V_{total} = 0\}$
    - 仅当**零策略违规**时才计入任务完成分数，这比纯任务完成率更严格

4. **风险比率评估（Risk Ratio）**

    - $\text{Risk Ratio}_{source,category} = \frac{\sum_i V_{source,category}(i)}{\#Policies_{source}}$
    - 三级风险分类：低风险（≤5%）、中风险（5-15%）、高风险（>15%）

### 基准实现

- **任务分布**：核心基准（0-84 索引）+ 认知负载测试（85-234 索引）
- **评估函数**：element_action_match、is_sequence_match、is_url_match、is_ask_the_user、is_action_count、is_program_html
- **人在回路支持**：扩展 BrowserGym 观察空间以包含策略层级，实现异步 Agent 集成

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

## 相关论文

- [OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News](livenewsbench_evaluating_llm_web_search_capabilities_with_fresh_news.md)
- [Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos](../../CVPR2026/llm_agent/ego2web_a_web_agent_benchmark_grounded_in_egocentric_videos.md)
- [FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)

<!-- RELATED:END -->
