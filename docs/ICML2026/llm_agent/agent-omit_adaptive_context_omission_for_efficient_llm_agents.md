---
title: >-
  [论文解读] Agent-Omit: Adaptive Context Omission for Efficient LLM Agents
description: >-
  [ICML 2026][LLM Agent][上下文管理] 通过 Monte-Carlo rollout 量化"哪些回合的 thought / observation 可以省"，再用冷启动 SFT + 双采样 omit-aware GRPO 训出能自适应跳过冗余思考和观测的 8B agent…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "上下文管理"
  - "思维省略"
  - "观察省略"
  - "GRPO"
  - "双采样"
---

# Agent-Omit: Adaptive Context Omission for Efficient LLM Agents

**会议**: ICML 2026  
**arXiv**: [2602.04284](https://arxiv.org/abs/2602.04284)  
**代码**: https://github.com/usail-hkust/Agent-Omit (有)  
**领域**: LLM Agent / 高效推理 / Agentic Reinforcement Learning  
**关键词**: 上下文管理, 思维省略, 观察省略, GRPO, 双采样

## 一句话总结
通过 Monte-Carlo rollout 量化"哪些回合的 thought / observation 可以省"，再用冷启动 SFT + 双采样 omit-aware GRPO 训出能自适应跳过冗余思考和观测的 8B agent，五个基准上 token 用量大降而准确率与七大前沿模型持平。

## 研究背景与动机

**领域现状**：LLM agent 通过 thought→action→observation 多轮循环解决任务（ReAct / agentic RL），Kimi-K2、DeepSeek-V3.2 等 agent 已经在 deep search、网页购物、具身决策、科学发现等场景展现强能力。但多轮交互让上下文越拉越长、token 成本飞涨。

**现有痛点**：现有提效方法分三类——只压思考（ToolLight、DEPO）、只剪观察（Observation-Mask、DeepMiner）、或两者合二为一做摘要（MEM-Agent、ReSum）。它们都把整条轨迹"一视同仁"地压缩，忽略了不同回合贡献度差异巨大。

**核心矛盾**：thought 和 observation 的"必要性"是 turn-dependent 的——早期高层规划往往把后续几轮的思考直接决定了；末轮做汇总时大多数早期观测早就过时。一刀切的压缩既会误删必要信息（影响准确率），也会保留无用 token（影响效率）。

**本文目标**：分两步：(1) 用可控干预定量证明"按回合选择性省略"的可行性；(2) 训练一个能在交互过程中自适应决定"这一轮的思考要不要写、之前哪些观测要不要丢"的策略。

**切入角度**：把 agent 的省略行为本身建模成动作空间的一部分——thought 输出空字符串、观测通过特殊 token `<omit_tool_response_N>` 显式删除——这样省略就可以在 SFT 和 RL 框架里自然学习。

**核心 idea**：让 agent 主动输出"思考省略"和"观测省略"动作，再用一个把"任务奖励"和"token 节省"耦合（且 task 失败时省略奖励清零）的 omit-aware GRPO 来训练，并辅以双采样解决"省略后看不到原信息"的归因难题。

## 方法详解

### 整体框架
两阶段优化：(a) Agent Omission Behavior Synthesis（冷启动 SFT）—— 通过 Monte-Carlo rollout 识别每条轨迹中"可省"的 thought/observation 回合，构造单轮 + 多轮两套合成数据，把"省略格式"和"在被省略上下文下继续推理"两件事一起教给基础模型。(b) Omit-Aware Agentic RL —— 引入 dual sampling（同时采全轨迹 + 每个省略点的部分轨迹）和 omit-aware reward（task reward + omit reward），用 GRPO 优化。理论上证明学到的省略策略偏离最优策略的程度被 KL 散度上界约束。

### 关键设计

1. **量化分析 + 省略动作显式化**:

    - 功能：先定量证明 "selective omission 真的能减 token 而不掉点"，再把省略本身设计成 agent 可输出的 token 模式，让后续 SFT/RL 能直接学。
    - 核心思路：在 WebShop + Qwen3-8B 上分别"挖掉第 $t$ 轮的 $\tau_t$ 或 $o_t$"再让 agent 接着走完，统计 token 与 Pass@1。结果：thought 占 45.1%、observation 占 52.2%、action 仅 2.7%；中间轮思考可省、末轮观测不能省、首轮思考不能省，存在大量"灰区"——准确率不降但 token 显著减。在动作端，思考省略用空 `<think> </think>`；观测省略用 `<omit_tool_response_N_...>`，对历史观测集 $\Gamma \subseteq \{1,\dots,t-1\}$ 显式 mask 掉。
    - 设计动机：把启发式（按时间窗删）升级为可学习策略，需要先验证"省略空间确实非空"，再给省略行为一个明确的语言学接口；后两个要素让 SFT 和 RL 都能监督。

2. **冷启动数据合成（Omission Behavior Synthesis）**:

    - 功能：把通用 LLM 教成 omission-aware 的 agent，提供初始策略让 RL 不至于探索灾难。
    - 核心思路：对训练轨迹做 forward rollout 识别"可省回合"——只要某回合省去后 token 减少且 accuracy 不降则标记为 omittable（论文 Figure 4 给出 $\tau_2,\tau_3,o_3$ 等示例）。然后分层构造：(i) Single-Turn omission，用专门 system prompt 教 agent 输出空 thought 或 omit_tool_response 命令；(ii) Multi-Turn omission，把整条轨迹中的可省 thought/observation 换成对应省略符号，强迫 agent 学会在历史被省后继续保持推理连续性，避免 context-lost。最后做全参 SFT，损失 $\mathcal{L} = -\mathbb{E}_{(x,y)\sim \mathcal{D}_{single}\cup\mathcal{D}_{multi}}[\log \mathcal{P}_{\pi_\theta}(y\mid x)]$，对环境观测部分加 loss mask。
    - 设计动机：直接上 RL 会因为 agent 不会输出省略符号而完全采不到正样本；先 SFT 把格式打开，是把"省略是一个合法动作"的认知植入模型的最低成本路径。

3. **Omit-aware Agentic RL：双采样 + 双奖励 + GRPO**:

    - 功能：把"省略策略"作为一阶决策学习目标，同时保证 task 准确率不被 reward hacking 牺牲。
    - 核心思路：dual sampling—— 对每个输入采全轨迹 $y$（执行省略动作的完整 episode），再针对每个发生省略的回合把"省略前的上下文 + 该轮 thought/action"截出作为部分轨迹 $y'$，每个 $y$ 派生 $p(y)$ 个 $y'$；这样 agent 在 $y'$ 上能"看到尚未省略时的上下文"来对省略决策学习归因，避开"省略后再也看不到原信息"导致策略不可学的死结。奖励上：task reward $R_{task}$ 对全 / 部分轨迹都给；omit reward $R_{omit}=\mathrm{Tok}(\tau_{omitted})/\mathrm{Tok}(y) + \mathrm{Tok}(o_{omitted})/\mathrm{Tok}(y)$ 只对全轨迹给，且在 $R_{task}=0$ 时强制清零，防止 agent "为省而省"。综合奖励 $r(\cdot)=(1-\mu)R_{task}+\mu R_{omit}$（$\mu=0.2$），$r'(\cdot)=R_{task}$。用 GRPO 优化，并加 KL 约束 $-\beta \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]$。
    - 设计动机：直接对 omission 决策做 credit assignment 需要"反事实的非省略上下文"——这正是普通 agentic RL 拿不到的；dual sampling 用一个工程巧思补上了这个缺口，使省略策略可学。task-conditioned omit reward 则把"提速不能降准确率"显式编码，比单纯加权和更稳健。

### 损失函数 / 训练策略
SFT 阶段标准 LM loss + 环境观测 loss mask；RL 阶段目标
$\max_{\pi_\theta} \mathbb{E}_{x,\{y_i,\{y'_{i,j}\}\}}\big[\tfrac{1}{n}\sum_i \big(r(x,y_i) + \tfrac{1}{p(y_i)}\sum_j r'(x,y'_{i,j})\big)\big] - \beta \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]$。基础模型 Qwen3-8B。理论上作者在 semantic Lipschitz 假设下证明效果 / 效率偏差被 $\delta + K' \cdot \mathrm{KL}(\pi^\ast,\pi_\theta)$ 上界约束，说明随 KL 减小可以单调逼近最优省略策略。

## 实验关键数据

### 主实验
五个 agent 环境（DeepSearch、WebShop、TextCraft、BabyAI、SciWorld），与七个前沿 LLM（DeepSeek-R1-0528、DeepSeek-V3.2、o3 / o4-mini、Qwen3-235B-A22B、Qwen3-Next-80B-A3B、Qwen3-32B）以及七个高效 agent 构造方法比较。

| 对比对象 | Pass@1 准确率 | Token 成本 | 备注 |
|----------|---------------|-------------|------|
| Agent-Omit-8B（基于 Qwen3-8B） | 与七个前沿 LLM 整体相当 | 显著更低 | 8B 用大模型一半甚至更少的 token 达到同级准确率 |
| 七个高效 agent 方法（TM / OM / TOM） | 各有所长 | 各有所长 | Agent-Omit 取得"最佳效果-效率 trade-off" |
| Qwen3-8B 原生 | 基线 | 基线 | 不省略时 thought 45.1% + observation 52.2% 占比 |

### 消融实验

| 配置 | 关键现象 | 解读 |
|------|----------|------|
| 仅 SFT（无 RL） | 学到省略格式但收益有限 | RL 才能学到自适应何时省 |
| 无 dual sampling | 省略策略难收敛 | 部分轨迹是省略 credit assignment 的必要桥梁 |
| 无 $R_{omit}$ | 与原 agent 几乎一致 | 缺乏显式效率激励 |
| $R_{omit}$ 不与 $R_{task}$ 耦合 | 出现 reward hacking，准确率掉 | 强约束 "task 失败则 omit 奖励为 0" 是必要的 |
| 单轮 omission only | 多轮场景泛化差 | 多轮合成数据强迫模型学习"无原信息也能续推" |
| 训练后 agent 行为分析 | 自适应省 3–4 轮 thought/observation，集中在中间轮 | 与 Section 3 量化分析的"可省灰区"高度一致 |

### 关键发现
- 一刀切的 TM / OM / TOM 方法因为漠视回合差异，在准确率或 token 上总要牺牲一边；Agent-Omit 同时拿到准确率和 token 上的最佳前沿。
- "首尾不能省、中间可省"的规律在五个环境中跨任务一致，说明省略策略本身具有跨域可迁移性。
- 理论的 KL 上界与实际训练曲线一致：随 GRPO 训练进行，agent 不断逼近 Monte-Carlo 标注的最优省略前沿。
- 把 omission 当作一阶动作学习要比把它当作 post-hoc 后处理（如摘要）更有效，因为前者能利用 RL 的 task-aware 反馈。

## 亮点与洞察
- 把"上下文压缩"从一个静态后处理问题转成"agent 自身的一阶决策"，这是范式上的位移——之前的工作都是"模型在外面被压缩"，本文是"模型自己决定省什么"。
- dual sampling 解决"省略后无法归因"的死锁，是 agentic RL 处理"删信息"类动作的可复用 trick；这套思路可以迁移到任何带"删除 / 合并"动作的策略学习场景。
- 显式 token 接口（`<omit_tool_response_N>`）让省略与现有 LLM tokenizer / API 完全兼容，部署成本低；对生产系统是个很好的"软改造"路径。
- "task 失败则 omit 奖励清零"是个简单但极其关键的 reward shaping 设计，避免了任何 efficiency-only reward 都会遇到的 collapse 模式。

## 局限与展望
- 实验都在 Qwen3-8B + 五个文本类 agent 环境上，跨更大规模模型 / 多模态 / 长 horizon (>20 轮) 任务效果尚待验证。
- 省略动作目前只覆盖 thought 全省和历史 observation 删除，但部分压缩（只省一部分思考、用摘要替换观测）的"细粒度省略"空间还没探索。
- dual sampling 让 RL 采样成本翻倍以上，计算开销大；对实际部署到 100B+ 规模的训练成本是潜在瓶颈。
- 理论分析依赖 semantic Lipschitz 假设，实际 LLM 在 prompt 微小变化下的奖励不连续性可能让上界变松。

## 相关工作与启发
- **vs ToolLight / DEPO（思考压缩）**：他们做 token-level 压缩；本文做 turn-level 决策，更精细且 RL 可学。
- **vs Observation-Mask / DeepMiner（启发式删观察）**：他们用固定规则；本文用学到的策略，跨环境一致。
- **vs MEM-Agent / ReSum（LLM 摘要）**：摘要会引入 LLM 召唤成本和信息扭曲；省略直接 mask，无信息扭曲，且省 token 更彻底。
- **vs Agentic RL 主流（如 GRPO/Verl 训练 search agent）**：本文在 GRPO 之上引入双采样和 omit-aware 奖励，是对 agentic RL 的扩展，可与 ReAct / search agent 框架正交叠加。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把上下文压缩重新定义为 agent 的一阶动作，并解决 credit assignment 的双采样设计有清晰创新。
- 实验充分度: ⭐⭐⭐⭐ 五个异质环境 + 七 LLM + 七效率方法的对比足够全面，唯独跨更大模型尺寸缺少 scaling 曲线。
- 写作质量: ⭐⭐⭐⭐ 从量化分析 → 框架 → 理论 → 实验逻辑顺畅，图 3 的可视化非常说服。
- 价值: ⭐⭐⭐⭐⭐ 对真实部署 agent 系统直接有用——上下文成本是当前 agent 落地最贵的部分之一，本方法可与现有 RL pipeline 即插即用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search](../../AAAI2026/llm_agent/agentswift_efficient_llm_agent_design_via_value-guided_hierarchical_search.md)
- [\[ICLR 2026\] Efficient Agent Training for Computer Use](../../ICLR2026/llm_agent/efficient_agent_training_for_computer_use.md)
- [\[ICML 2026\] Adaptive Querying with AI Persona Priors](adaptive_querying_with_ai_persona_priors.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/llm_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[AAAI 2026\] When Refusals Fail: Unstable Safety Mechanisms in Long-Context LLM Agents](../../AAAI2026/llm_agent/when_refusals_fail_unstable_safety_mechanisms_in_long-context_llm_agents.md)

</div>

<!-- RELATED:END -->
