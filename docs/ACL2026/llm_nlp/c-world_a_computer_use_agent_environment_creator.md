---
title: >-
  [论文解读] C-World: A Computer Use Agent Environment Creator
description: >-
  [ACL 2026][LLM/NLP][MCP 工具] 作者将"agent 环境"形式化为 Action / Task / Transition / Reward 四元组并实现为 C-World：用 5…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "MCP 工具"
  - "长程任务"
  - "World Model"
  - "状态扰动"
  - "评测+训练数据引擎"
---

# C-World: A Computer Use Agent Environment Creator

**会议**: ACL 2026  
**arXiv**: [2601.06328](https://arxiv.org/abs/2601.06328)  
**代码**: https://ziqiao-git.github.io/C-World/ (有)  
**领域**: LLM Agent / 计算机使用 / Agent 环境  
**关键词**: MCP 工具, 长程任务, World Model, 状态扰动, 评测+训练数据引擎

## 一句话总结
作者将"agent 环境"形式化为 Action / Task / Transition / Reward 四元组并实现为 C-World：用 5,571 个真实 MCP 工具 + 自动任务合成 + state controller 扰动 + 双信号 reward 提供高保真评测，又用一个"World Engine"在无 live API 下模拟工具响应实现可规模化训练；评测 9 个前沿 LLM 发现"规划普遍强、执行普遍弱"，仅用 1,170 条 C-World 轨迹微调即可超过用 119k 样本训练的 baseline。

## 研究背景与动机

**领域现状**：单轮 function-calling 已接近饱和（BFCL 等基准均被刷到很高），但真实"计算机使用"任务往往涉及几十轮交互、跨多个 app、含模糊约束和动态失败。现有 agent benchmark（AgentBench、ToolBench、WebArena 等）要么工具数有限（<600），要么是静态题集，无法支持持续训练。

**现有痛点**：(1) 工具池小、领域单一，难以体现真实工作流的广度；(2) 任务静态，不能"自我进化"出新约束；(3) 缺少"扰动"——所有任务都是 happy path，无法测 agent 的 recovery / replan；(4) 评测信号简单（pass/fail 或 LLM judge），无法分离"规划失败"和"执行失败"；(5) 训练完全依赖真 API，受成本、速率、状态不稳定限制，做不到大规模轨迹采集。

**核心矛盾**：要让 agent 像人一样在复杂环境中学，就需要"广、深、可扰动、可低成本规模化"的环境，但人工搭这种环境本身贵到不可行；如果靠 live API，又被外部服务的速率/费用/不稳定卡死。

**本文目标**：(1) 形式化"agent 环境"应该包含哪些组件；(2) 构造一个可按需生成新环境的系统而不是固定 benchmark；(3) 提供一个不依赖 live API 的"world model"模式，让训练侧能规模化。

**切入角度**：把 RL 里成熟的 MDP 四元组（动作空间、转移、奖励、任务分布）原样套到 LLM agent，并把"工具调用响应"建模成一个可由 LLM 模拟的转移函数 → world engine。

**核心 idea**：定义 $(\mathcal{A},\mathcal{T},\mathcal{F},\mathcal{R})$ 四组件 + 真实模式 / 合成模式双轨实现；4 个组件全部用"自动合成 + LLM 模拟"替代人工。

## 方法详解

### 整体框架
C-World 把 agent 环境形式化为 4 个组件 + 一个 World Engine。
- $\mathcal{A}$ 动作空间：从 Smithery 注册中心抓 276 个 MCP server、5,571 个工具（覆盖 204 个常用 app，如 Gmail/GitHub/Slack），统一用 MCP 协议封装并做三阶段执行验证；运行时通过 BGE-M3 + FAISS 提供 `search_tools(query, k)` 让 agent 自检索。
- $\mathcal{T}$ 任务分布：tool candidate sampling（按 server 做 round-robin 防止任务被单一 app 主导）→ check-then-revise 合成长程任务（含 wild constraints）→ 模糊改写防止 keyword 短路。
- $\mathcal{F}$ 转移：State Controller 中间件拦截 tool call，按预设策略注入 tool-level（超时/限流）、state-level（结果延迟/被改写）、constraint-level（中途加新需求）三类扰动。
- $\mathcal{R}$ reward：verifiable（schema/order/diversity）+ judge-based（completeness/grounding/tradeoff，3 个前沿 LLM majority vote）。
- World Engine：以工具响应类别卡片（email/calendar/code-hosting 等）+ schema + 会话级 log 为条件，让 LLM 根据 `(state, action)` 生成"以假乱真"的响应，从而不依赖 live API 就能跑大量轨迹。

外加一个 Planner-Actor agent 框架：Planner 把任务拆 sub-goal graph，Actor 走 ReAct 调工具，Planner 每步 verify 并给反馈。

### 关键设计

1. **统一动作空间 + 工具检索（Action Space）**:

    - 功能：把 5,571 个真实 MCP 工具变成一个可被检索、可被认证执行的统一动作池。
    - 核心思路：通过自动 registry 爬虫 + 人工补充收集工具；为需鉴权的服务配置专用虚拟账号；执行三阶段验证（authenticated availability → successful invocation → usable responses）过滤死工具；最后把"server identity + tool name + 描述 + schema"拼成文档，BGE-M3 编码后存 FAISS 提供 `search_tools(query, k)` 接口。Agent 不会一次拿到全部工具，必须自己查、按需 load。
    - 设计动机：之前 benchmark 要么硬给一小撮工具（不真实），要么给一大堆但都死的（不可执行）。真正逼近真实场景必须既"全"又"活"，且不能让 agent 走 keyword 捷径——所以工具检索接口是唯一入口。

2. **check-then-revise 任务合成 + 反捷径机制（Task Distribution）**:

    - 功能：在不依赖人工标注的情况下，生成"长程 + 多 app + 含 wild constraint + 抗 keyword 短路"的任务。
    - 核心思路：先采样 1~3 个 seed tools，用它们的描述做查询、用 `search_tools` 召回更大候选集；按 server 做 round-robin 采样保证跨 app（强制最少 server 数）。然后让 LLM 生成初始 query，再进入 check-then-revise 循环：自动评估 (i) tool coverage（是否合理地激活全部候选工具）+ (ii) constraint quality（约束是否多样、互相耦合、产生长程依赖），不达标就反馈重写。最后做模糊化（"send the summary to the team" 而非 "use slack_post_message"），并只暴露 `search_tools` 接口，迫使 agent 自己分解 sub-query 检索。
    - 设计动机：之前的合成任务要么短，要么虽然长但靠堆 steps 凑数；本文用"工具覆盖 + 约束密度"两个 metric 强行把任务推向长程；fuzzy 改写则解决"评测时被合成 prompt 泄露"的隐性 leakage 问题。

3. **三层 State Controller + World Engine（Transition Function）**:

    - 功能：让环境既能模拟真实失败（rate limit、state drift、需求变更），又能无 live API 跑大规模训练。
    - 核心思路：State Controller 是一个嵌在 agent runtime 内部的轻量 Python 中间件，拦截 MCP 出入流量，按"adversity budget"注入 (a) tool-level（超时/不可用/限流）、(b) state-level（payload 截断/session 失效）、(c) constraint-level（中途加新规则）三类扰动；为公平性，每个模型遭遇的扰动总量恒等但触发时机随机。World Engine 把工具按功能类别（email/code-hosting/calendar 等）建"category card"含典型响应模式、字段结构、常见失败，再给 schema + few-shot + 会话 log → LLM 直接生成响应，因为只依赖 schema 和卡片，所以能对同类下没见过的工具零样本泛化，甚至能模拟现实不存在的企业级环境。在 50 个评测任务上跟 live 执行做对比，Spearman $\rho=0.883$。
    - 设计动机：直接 random 噪声是没用的（agent 学不到什么），所以要"可复现 + 针对性"的扰动；World Engine 这一层是把 agent 训练从"被 API 卡死"中解放出来的关键——可以批量造轨迹做 SFT/RL，且对不存在的工具也能模拟，便于做 stress test。

### 损失函数 / 训练策略
训练侧把 50 个 seed task 的"首轮有效动作"（abstract intent → 具体工具检索/调用）筛出 1,170 条样本，转成 ms-swift 格式按 Hermes-style agent supervision（显式建模 tool invocation / tool response）做 SFT；和 Toucan（119k）、ToolACE（11.3k）公平对比。

## 实验关键数据

### 主实验
9 个前沿 LLM 在 C-World 真实模式下的总分（10 分制 + %）：

| 模型 | Overall | Completeness | Recovery% | Format% | Tool Calls |
|------|---------|-----|-----|-----|-----|
| gemini-3-pro-preview | **5.87** | 4.75 | 89.0 | 53.9 | 47.9 |
| claude-opus-4.5 | 5.42 | 4.70 | 83.7 | 51.0 | 45.2 |
| deepseek-v3.2 | 4.97 | 4.00 | **90.6** | 39.5 | 21.7 |
| grok-4 | 4.78 | 3.80 | 89.0 | **68.3** | 27.4 |
| gpt-oss-120b | 4.66 | 3.42 | 72.7 | 35.8 | 14.4 |
| gpt-5.2 | 4.43 | 3.42 | 79.3 | 12.4 | 29.2 |
| qwen3-235b-a22b | 3.53 | 2.56 | 88.1 | 31.3 | 11.2 |
| gpt-4o-mini | 3.07 | 1.13 | 50.6 | 3.3 | 51.7 |

### 消融实验（World Engine 训练数据效率 + 模拟保真度）

| 模型 / 训练数据 | 样本数 | BFCL | MCP-Universe |
|------|-----|------|------|
| Qwen2.5-7B base | – | 19.93% | 4.40% |
| + Toucan | 119k | 27.18% | 15.28% |
| + ToolACE | 11.3k | 27.06% | 2.23% |
| **+ C-World** | **1.2k** | **28.58%** | **15.30%** |
| Qwen3-8B base | – | 18.32% | 6.35% |
| + Toucan | 119k | 27.39% | 6.67% |
| + ToolACE | 11.3k | 29.49% | 3.29% |
| **+ C-World** | **1.2k** | **30.05%** | **8.86%** |

| 评测模式 | Spearman ρ |
|------|------|
| World Engine (合成) vs Real Exec | **0.883** |
| DeepSeek-V3.2 judge vs Human | 0.759 |
| GPT-5.1 judge vs Human | 0.733 |
| Human vs Human ceiling | 0.773 |

### 关键发现
- **执行能力是 bottleneck**：所有模型的 Goal Decomposition 得分都在 7.7~8.6（规划没差距），但 Completeness 跨模型从 1.13 到 4.75（差距 4×），说明问题在"做"不在"想"。
- **工具调用数 ≠ 成功率**：gpt-4o-mini 调 51.7 次最多却最差（陷入循环式重复调用），Gemini-3-Pro 用相近 47.9 次拿最高分；说明高活动量必须配上高 reasoning 才有用。
- **约束遵循是主要失败模式**：Format compliance 跨模型从 3.3% 到 68.3%，比 tool invocation 的差距大得多。
- **World Engine 几乎可平替 live API**：模型排名 Spearman 0.883；judge 集成也接近人类 ceiling 0.773（仅低 0.014）。
- **数据效率惊人**：1,170 条 C-World 轨迹 SFT 反超 119k Toucan 样本，证明"长程 + 约束密集 + 含扰动"的轨迹比海量 happy path 更值钱。

## 亮点与洞察
- 把 RL 教科书里的 $(\mathcal{A},\mathcal{T},\mathcal{F},\mathcal{R})$ 形式化套到 LLM agent 是非常聪明的"还原"——之前 agent 论文经常把环境讲成黑盒，本文显式拆开后立刻清楚每个组件该怎么独立改、独立评测。
- World Engine 的"category card"设计是 zero-shot 泛化到同类未见工具的关键——比 per-tool demonstration 便宜得多，且让"完全不存在的工具"也能拥有可信的响应分布，可以用于做 stress test。
- "adversity budget 恒定 + 触发时机随机化" 是漂亮的公平性设计，回避了不同模型遭遇不同难度的偏差。
- "Planner-Actor 同模型 + Planner 每步 verify sub-goal" 是给长程任务找到的轻量解药——比 multi-agent 角色专业化简单，但 Table 3 显示 Gemini-3-Pro 凭它从首轮第 5 名爬到第 1 名。
- 1,170 条 > 119k 这种"训练数据效率"结论可能比 benchmark 数字本身更值得整个社区关注：暗示工具学习的核心是"少而硬"的轨迹，而不是"多而易"。

## 局限与展望
- 评测集只 50 个 seed scenario + 254 LongSeal 题，对工具-server-约束的组合空间覆盖远不充分。
- 主分析集中在 frontier / 大开源模型，sub-10B 模型只在 Appendix H 给了 4 个数据点；C-World 框架本身能不能稳定指导小模型训练有待验证。
- World Engine 虽然 Spearman 高，但绝对分数和 live execution 有系统偏差（如 deepseek-v3.2 合成 Pass% 只有 38.9%，远低于真实 87.5%），用作训练监督时可能引入偏差。
- 改进思路：(1) 把任务合成做成持续进化（new server / new failure mode 自动接入）；(2) world engine 输出做置信度估计，低置信时 fallback 到 live；(3) 把 sub-10B 模型纳入主分析，找到"对工具学习友好"的小模型起点。

## 相关工作与启发
- **vs AgentBench / WebArena**: 都是静态 benchmark，工具数少（18~600 vs 5,571）、不可重生成；C-World 的环境-级合成 + 状态扰动 + 模糊指令是真正"可演化"的环境。
- **vs ToolBench / Toucan**: 都做工具学习数据合成，但只是任务静态、无 transition 扰动；本文的 1.2k > 119k 直接打脸"靠堆数据涨点"的做法。
- **vs StableToolBench / MCP-Bench**: 同样关注 MCP 工具调用，但本文是第一个把"工具响应模拟器"（World Engine）形式化做出来并实证可平替 live API 的工作。
- **启发**：(1)「显式 $(\mathcal{A},\mathcal{T},\mathcal{F},\mathcal{R})$ 拆分」可迁移到任何需要 agent 评测的领域（多模态 agent、autonomous coding）；(2) category card 思路可推广到其他模拟器（如代码执行环境、浏览器）减少 per-instance 监督。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "环境创建系统"而非"benchmark"是范式转移；World Engine 用 LLM 模拟 MCP 响应是新方向。
- 实验充分度: ⭐⭐⭐⭐ 9 个 frontier LLM + 真/合成双模式 + 训练侧 SFT 对比 + Per-event-type Recovery + 人类对齐 + 5 维 persona 分析，但 seed task 数（50）和 small model 覆盖偏少。
- 写作质量: ⭐⭐⭐⭐ 形式化清晰，每个组件都有独立 section 和示例；附录 F 把扰动类型用 4 个真实 trajectory 讲清楚，可读性很高。
- 价值: ⭐⭐⭐⭐⭐ 5,571 工具 + 双模式 + 公开代码，是目前最实用的 computer-use agent 训练评测基础设施之一；1.2k 反超 119k 的发现也直接改变工具学习的数据收集策略。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents](../../ACL2025/llm_nlp/axis_efficient_human-agent-computer_interaction_with_api-first_llm-based_agents.md)
- [\[NeurIPS 2025\] Do Language Models Use Their Depth Efficiently?](../../NeurIPS2025/llm_nlp/do_language_models_use_their_depth_efficiently.md)
- [\[ICML 2026\] Express Your Doubts: Probabilistic World Modeling Should Not Be Based on Token logprobs](../../ICML2026/llm_nlp/express_your_doubts_--_probabilistic_world_modeling_should_not_be_based_on_token.md)
- [\[ICML 2026\] Multi-Agent Teams Hold Experts Back: 自组织 LLM 团队为什么留不住「专家」](../../ICML2026/llm_nlp/multi-agent_teams_hold_experts_back.md)
- [\[ICLR 2026\] How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use](../../ICLR2026/llm_nlp/how_far_are_llms_from_professional_poker_players_revisiting_game-theoretic_reaso.md)

</div>

<!-- RELATED:END -->
