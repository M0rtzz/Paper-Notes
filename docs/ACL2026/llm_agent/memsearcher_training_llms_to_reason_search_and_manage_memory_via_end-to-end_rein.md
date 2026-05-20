---
title: >-
  [论文解读] MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End RL
description: >-
  [ACL 2026][LLM Agent][搜索 Agent] MemSearcher 把搜索 agent 的"历史拼接"换成"LLM 自管理的紧凑内存"——每轮只看 `(question, memory)` 而不是 `(question, t₁, a₁, o₁, …)`…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "搜索 Agent"
  - "内存管理"
  - "GRPO"
  - "Multi-Context RL"
  - "ReAct 替代"
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End RL

**会议**: ACL 2026  
**arXiv**: [2511.02805](https://arxiv.org/abs/2511.02805)  
**代码**: https://github.com/icip-cas/MemSearcher (有)  
**领域**: LLM Agent / RL / Search Agent  
**关键词**: 搜索 Agent, 内存管理, GRPO, Multi-Context RL, ReAct 替代

## 一句话总结
MemSearcher 把搜索 agent 的"历史拼接"换成"LLM 自管理的紧凑内存"——每轮只看 `(question, memory)` 而不是 `(question, t₁, a₁, o₁, …)`，并用 multi-context GRPO 把整条 trajectory 的 advantage 传播到每一轮独立优化，在 7 个 QA benchmark 上 3B/7B/14B 全面超过同尺寸 ReAct baseline（7B 甚至超 32B ReSearch），context 长度恒定 <4K token。

## 研究背景与动机
**领域现状**：LLM-based search agent（Search-R1, ReSearch, AutoRefine, R1-Searcher 等）大多走 ReAct 范式 —— 每轮把 `thought-action-observation` 累加到 context 里，让 LLM 决定下一步搜什么。这种"对话历史拼接"是当前 RL-based search agent 的事实标准，且配合 GRPO/PPO 端到端训练已经能拿到很强的多跳问答效果。

**现有痛点**：ReAct 把所有历史塞进 context 有两大致命问题：
1. **Context 线性膨胀**：搜索 agent 的 observation 是检索回来的 passage，每轮可能数百到数千 token，多轮跑下来 context 轻松破 10K。Lost-in-the-middle、长 context 性能衰减、KV cache GPU 显存爆炸全都来了。
2. **噪声压制信号**：检索 passage 大部分与问题无关，混在历史里让 LLM 难以聚焦关键事实；论文 Figure 2 给的反例就是 ReAct 把"角色的女友"和"演员现实中的对象"搞混了。

**核心矛盾**：Search agent 必须看多轮历史才能多跳推理，但"看历史 = 拼接全部"在长跑场景下又不可持续。这两者构成根本张力。

**本文目标**：(i) 让 context 长度对轮数 $n$ 保持 $O(1)$；(ii) 同时让 LLM 自己学会"该记什么、该丢什么"；(iii) 端到端用 RL 训练（不用人工标注内存状态）。

**切入角度**：与其外挂 RAG/KG/structured memory 一类附加模块，作者干脆**让同一个 backbone LLM 兼任 reasoning 和 memory manager** —— 一次 prompt 既输出 thought+action，也在 observation 回来后输出更新后的 memory。这种"自反思式内存"训练困难在于：每轮的输入 context $c_{i,j} = (q, m_{i,j-1})$ 都不一样，使一条 trajectory 变成"多个独立优化目标"，vanilla GRPO 算 advantage 时只对整条 trajectory 算一次 reward，没法直接套用。

**核心 idea**：(1) 框架：用 `<memory>` 标签存自然语言内存，每轮 LLM 既是 actor 又是 memory manager。(2) 训练：提出 multi-context GRPO ——一条 trajectory 算一个 reward → 一个 advantage → 把这个 advantage **均匀传播** 给该 trajectory 内所有 turn → 把每个 turn 当独立优化目标。

## 方法详解

### 整体框架
**Trajectory 表示**：第 $i$ 条 trajectory 有 $n_i$ 个 turn，第 $j$ turn 形如 $(q, m_{i,j-1}, t_{i,j}, a_{i,j}, o_{i,j}, m_{i,j})$（最后一 turn 没有 $o, m$，因为已经 boxed 答案了）。$t$ 在 `<think>` / $a$ 在 `<tool_call>` / $o$ 在 `<tool_response>` / $m$ 在 `<memory>` 里。

**每轮交互**：
1. LLM 输入 $(q, m_{i,j-1})$；
2. 输出 thought + action（要么调 wikipedia_search 工具，要么 `\boxed{}` 给最终答案终止）；
3. 环境返回 observation；
4. LLM 再次被调用，把 $(o_{i,j}, m_{i,j-1})$ 融合写成新 $m_{i,j}$，且 token 数 ≤ 预设上限 1024；
5. 进入下一轮，循环直到答出来或超最大轮数。

**计算复杂度对比**：

| 方法 | 每轮 Context | 每轮 FLOPs | 总 FLOPs | GPU Memory |
|------|-------------|------------|----------|-----------|
| ReAct | $O(n)$ | $O(n)$ | $O(n^2)$ | $O(n)$ |
| **MemSearcher** | $O(1)$ | $O(1)$ | $O(n)$ | $O(1)$ |

### 关键设计

1. **LLM-as-Memory-Manager 的紧凑内存范式（框架核心）**:

    - 功能：把 ReAct 的"线性拼接历史"换成"固定长度内存 + 自然语言摘要"，使 context 在多轮搜索下保持恒定。
    - 核心思路：每轮 LLM 看到的 context $c_i = (q, m_{i-1})$ 而不是 $c_i = (q, t_1, a_1, o_1, \ldots)$。memory $m$ 用自然语言写在 `<memory>` 标签内，由 backbone LLM 自己 overwrite 生成。生成 prompt 指引："仔细读 $o_i$，把对回答 $q$ 有用的新信息整合进去，同时保留 $m_{i-1}$ 中的所有相关细节"。最大 memory 长度 1024 token（在 256-2048 区间做了 ablation）。
    - 设计动机：传统外挂 memory（RAG / KG / Mem0）要么需要单独训练 retriever、要么牺牲端到端可微性；让 backbone LLM 自己 manage 内存的好处是 (1) 不引入额外模型，(2) 整条 pipeline 还是同一个 LLM 在 act，RL 训练可以端到端覆盖。Memory 用自然语言（而非 latent token）保证可解释、可调试。

2. **Multi-Context GRPO（端到端训练核心）**:

    - 功能：解决"一条 trajectory 内多 turn 各自有不同 context"导致 vanilla GRPO 不能直接用的问题。
    - 核心思路：先按 GRPO 套路对每个 question $q$ 采样 group of $G$ trajectories，每条 trajectory 算 final reward $R_i$（format reward + F1-based answer reward），按 group 内 mean/std 标准化得到 trajectory-level advantage $A_i = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}$。**关键步**：把 $A_i$ uniformly propagate 到该 trajectory 的所有 turn，即 $A_{i,j} = A_i$ for all $j \in [1, n_i]$。然后把每 turn 当独立 PPO/GRPO 优化目标，objective 改为对所有 $(i,j)$ 求和，importance ratio $r_{i,j}(\theta) = \pi_\theta(T_{i,j}|c_{i,j}) / \pi_{\theta_{\text{old}}}(T_{i,j}|c_{i,j})$。最后还要对 search engine 返回的 observation token 做 loss mask（不算 policy gradient），稳定训练。
    - 设计动机：直接对整条 trajectory 算 ratio 在 multi-context 下既数值不稳又信号稀疏；按 turn 拆分后每个 turn 都有自己的梯度信号，但 reward 又只有 final——所以用 trajectory-level advantage 强制对齐"哪条 trajectory 好"，让 reward 信号沿所有 turn 反传。这是把"sparse outcome reward + dense per-turn optimization"在 GRPO 框架下漂亮地缝合起来。

3. **Reward 设计与训练稳定性**:

    - 功能：纯规则 reward，无 process supervision，结合 format check + F1 answer 评估。
    - 核心思路：$R = 0$ 若格式错误；$R = 0.1$ 若格式对但 F1=0（鼓励先学会格式）；$R = F1$ 若 F1>0。Group 内归一化得 $A_i$。训练超参 lr=1e-6，KL coef=0.001，clip 0.2，rollout group=5，temperature=1.0。Search engine token 全 mask 掉。
    - 设计动机：format reward 当作"warm-up bonus"避免模型早期完全 zero reward 无信号；F1 而非 EM 当 fine-grained reward 让部分正确的回答也有梯度。这是 DeepSeek-R1 风格的 rule-based reward 在 multi-turn search 场景的合理迁移。

### 损失函数 / 训练策略
基于 verl 库训练，backbone 是 Qwen2.5-3B/7B/14B-Instruct，知识源是 2018 Wikipedia dump，retriever 是 E5。训练数据用 Search-R1 公开的 NQ + HotpotQA train split。3B/7B 跑 8×H100，14B 跑 2×8×H100。一个 epoch 即可（256 batch、5 rollout）。Reward 曲线两阶段：前 25 step 急升（学会基础工具+memory 使用），之后缓慢上行（精细化策略）。

## 实验关键数据

### 主实验

**7 benchmark EM 平均分（Avg.）：3B/7B/14B vs SOTA baseline**：

| 尺寸 | 最强 baseline | 本文 | 绝对增益 |
|------|--------------|------|---------|
| 3B | AutoRefine-3B-base = 40.5 | **MemSearcher-3B = 43.8** | +3.3 |
| 7B | ReSearch-7B = 43.6 / R1-Searcher-7B = 40.2 | **MemSearcher-7B = 48.9** | +5.3 |
| 14B+ | Search-R1-14B-base = 47.8 / ReSearch-32B = 48.3 | **MemSearcher-14B = 51.7** | +3.4 vs 32B |

**关键观察**：MemSearcher-3B (43.8) 已超过所有 7B baseline；MemSearcher-7B (48.9) 已超过 32B ReSearch。这说明节省 context 后省下来的 model capacity 全花在了真正的搜索推理上。

**分数据集主表（节选）**：

| 数据集 | Search-R1-7B-base | ReSearch-7B | **MemSearcher-7B** |
|--------|-------------------|-------------|--------------------|
| NQ | 48.0 | 40.9 | **52.7** |
| TriviaQA | 63.8 | 63.7 | **68.1** |
| PopQA | 45.7 | 44.6 | 47.8 |
| HotpotQA | 43.3 | 43.5 | **50.8** |
| 2Wiki | 38.2 | 47.6 | 48.6 |
| Musique | 19.6 | 22.3 | **25.8** |
| Bamboogle | 43.2 | 42.4 | **48.8** |

### 消融与分析

**RL training vs no training（Qwen2.5-Instruct base）**：

| 模型 | w/o training | w/ MemSearcher RL | 提升 |
|------|--------------|-------------------|------|
| Qwen2.5-3B-Instruct | 14.4 | 43.8 | **+29.4** |
| Qwen2.5-7B-Instruct | 25.8 | 48.9 | **+23.1** |
| Qwen2.5-14B-Instruct | 27.7 | 51.7 | **+24.0** |

→ 框架本身需要 RL 才能 unlock，纯 prompting 不够。

**RL vs SFT（Qwen2.5-3B）**：

| 方法 | Avg |
|------|-----|
| SFT (Qwen2.5-72B 蒸馏轨迹) | 28.5 |
| **RL** | **43.8** |

→ SFT 用 72B 蒸馏轨迹比 RL 差 15.3 分，因为 72B 自己也没掌握 MemSearcher，做不出好 teacher；RL 直接奖励"答对"，让模型自学 what-to-retain。

**Memory 长度 ablation（256/512/1024/2048 tokens）**：

- 简单数据集如 Bamboogle 256 token 即饱和；
- 复杂多跳 Musique 256→1024 持续上升；
- 默认 1024 是 trade-off sweet spot。

**Context 长度对比（vs ReAct-based ReSearch）**：MemSearcher 多轮交互平均 context 始终 <4K token，几乎水平线；ReSearch 线性增长，5 轮后突破 10K。

### 关键发现
- **小模型超大模型**：MemSearcher-3B > 7B baseline；7B > 32B ReSearch。压缩 context 节省下来的算力让模型把"capacity 用在刀刃上"。
- **甚至超 Google Web Search**：MemSearcher 在本地 wiki dump 上的成绩超过用 Google 真实搜索的 R1-Searcher 和 ZeroSearch，说明 memory 设计的收益大过 web 索引质量收益。
- **训练曲线两阶段**：前 25 步快速学会格式 + 工具调用，之后缓慢学 memory 策略；和 DeepSeek-R1 的两阶段学习模式呼应。
- **SFT 蒸馏天花板低**：因为 teacher（72B）自己也没掌握 MemSearcher。这是 framework innovation + RL 训练的典型案例——新范式的小模型必须靠 RL 自己探索，而不是从老范式的大模型蒸。
- **Memory 长度敏感性**：复杂任务需要更长 memory，但 1024 是工程上的 sweet spot；这也启发将来可做"自适应 memory 长度"。

## 亮点与洞察
- **"backbone 兼 memory manager"是优雅的极简设计**：不引入额外模块，不破坏端到端可训练性，单 LLM 同时承担 reasoning、acting、memorizing 三职。比 RAG/KG/structured memory 这些外挂方案更内聚。
- **Multi-context GRPO 是个普适算法贡献**：凡是"trajectory 内每 turn context 不同 + final reward 稀疏"的 multi-turn RL 场景（如工具使用、多轮对话、长程规划）都可以套用 trajectory-level advantage propagation。
- **Context 恒定 → 工业可部署**：搜索 agent 在线服务最大成本就是 long context 的 KV cache，MemSearcher 直接砍到 $O(1)$，对低显存服务器（如 4090 / A10）部署友好。
- **"通过让 backbone 学会忘"反而提升性能**：传统直觉是"更多 context = 更好"，本文反证 selective forgetting + 紧凑 memory 在 search 任务上 dominate cluttered history。和 attention 机制本质（关注少量、忽略多数）一致。
- **3B 超 32B 的 capability density 提升**：是个非常有冲击力的实验——说明"以 paradigm 创新换 model scaling"在 agent 时代是可行路径。

## 局限与展望
- **作者承认**：(1) 内存机制简单（纯自然语言 overwrite），更复杂的 RAG-like / structured memory 没探索；(2) Multi-context GRPO 仍可能有 length bias（长 trajectory 贡献多）等待优化。
- **自查**：(1) Memory overwrite 是 destructive 的，没有 long-term archival 机制，对超长任务可能丢失早期重要信息；(2) 实验只在静态 wiki dump，未跑真实 web；(3) 对内存的 quality 没有 explicit reward，只能靠 final answer 反推；(4) Qwen2.5 系列以外的模型族（如 Llama / Mistral）未验证；(5) Format reward = 0.1 虽然帮助但容易引导模型"勉强写对格式但答非所问"，可能掩盖部分失败模式。
- **改进方向**：(1) 分层内存（短期 working memory + 长期 archival）；(2) 在 reward 中加入 memory quality auxiliary signal（比如内存与最终答案的因果归因度）；(3) 自适应 memory length 而非固定 1024；(4) 把 multi-context GRPO 拓展到 dense per-turn reward（如 turn-level relevance reward）；(5) 跨模型族泛化验证。

## 相关工作与启发
- **vs ReAct / Search-R1 / ReSearch / AutoRefine（ReAct 路线）**：他们都拼接全部历史，context 线性涨；本文砍到 $O(1)$ context 并 outperform 他们。
- **vs R1-Searcher / ZeroSearch（真 Web）**：用真 Google API 反而不如本文用静态 wiki，说明 paradigm 本身的优势大过数据源。
- **vs MEM1 / MemAgent（其他 memory-based agent）**：本文是首个把 memory 管理与 RL 端到端结合并用 multi-context GRPO 训练的，是 paradigm 层面的算法贡献。
- **vs HippoRAG / Mem0 / Zep（外挂结构化 memory）**：他们靠 KG / vector store 离线索引；本文用 LLM 自管理在线 memory，零外部依赖。
- **启发**：(1) 凡是 multi-turn agent，都应考虑"压缩历史"而非"拼接历史"；(2) sparse outcome reward + per-turn optimization 可以通过 advantage broadcast 这种简单 trick 缝合；(3) 让 backbone 同时担任多种角色（reasoner + actor + manager）是节省 capacity、保 end-to-end 可训练性的好范式；(4) "新 paradigm 必须 RL 训不能蒸"的现象提醒我们：framework 创新的真实潜力只能用 exploration 而非 imitation 解锁。

## 评分
- 新颖性: ⭐⭐⭐⭐ Memory-managed search agent + multi-context GRPO 的组合在 RL-based search agent 领域是清晰的 paradigm shift；个别组件（GRPO、ReAct 替代思路）不新但拼合得当。
- 实验充分度: ⭐⭐⭐⭐⭐ 7 数据集 × 3 模型尺寸 × 8 baseline + RL/SFT/no-training 三对照 + memory 长度 ablation + context 曲线 + 训练 reward 曲线 + 案例分析，覆盖完整。
- 写作质量: ⭐⭐⭐⭐ Figure 1/2 直观对比 ReAct，公式推导严谨，table 复杂度对比一目了然；附录给出完整训练超参与案例研究。
- 价值: ⭐⭐⭐⭐⭐ 公开代码，工业可部署（<4K context），3B 超 7B baseline 的 capability density 极具吸引力；multi-context GRPO 算法可推广到所有 multi-turn agent RL 训练，影响面广。

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers](../../NeurIPS2025/llm_agent/lc-opt_benchmarking_reinforcement_learning_and_agentic_ai_for_end-to-end_liquid_.md)
- [\[ACL 2026\] StructMem: Structured Memory for Long-Horizon Behavior in LLMs](structmem_structured_memory_for_long-horizon_behavior_in_llms.md)
- [\[ACL 2026\] BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search](bapo_boundary-aware_policy_optimization_for_reliable_agentic_search.md)
- [\[ACL 2026\] GOAT: A Training Framework for Goal-Oriented Agent with Tools](goat_a_training_framework_for_goal-oriented_agent_with_tools.md)
- [\[ACL 2026\] Supplement Generation Training for Enhancing Agentic Task Performance](supplement_generation_training_for_enhancing_agentic_task_performance.md)

</div>

<!-- RELATED:END -->
