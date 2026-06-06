---
title: >-
  [论文解读] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle
description: >-
  [ICML 2026][LLM Agent][经验生命周期] EvolveR 给 LLM agent 套一个「在线交互 → 离线自蒸馏成原则库 → GRPO 策略进化」的闭环生命周期：agent 不再丢弃过去轨迹，而是把自己的成功失败抽象成可检索的「策略原则」…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "经验生命周期"
  - "自蒸馏原则库"
  - "动态评分"
  - "GRPO"
  - "多跳问答"
---

# EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle

**会议**: ICML 2026  
**arXiv**: [2510.16079](https://arxiv.org/abs/2510.16079)  
**代码**: https://github.com/Edaizi/EvolveR (有)  
**领域**: LLM Agent / 持续学习 / 强化学习  
**关键词**: 经验生命周期、自蒸馏原则库、动态评分、GRPO、多跳问答

## 一句话总结
EvolveR 给 LLM agent 套一个「在线交互 → 离线自蒸馏成原则库 → GRPO 策略进化」的闭环生命周期：agent 不再丢弃过去轨迹，而是把自己的成功失败抽象成可检索的「策略原则」，再用 RL 学会**如何用自己的原则**去解新问题，在 7 个多跳 QA benchmark 上明显跑赢 Search-R1 等 RL agent baseline。

## 研究背景与动机

**领域现状**：LLM agent（ReAct、Reflexion、ExpeL、Search-R1 等）在工具调用上已能跑通，但绝大多数是「无状态」的：每次任务独立、过去经验要么直接丢、要么靠外部 LLM 教师 distill 出 hint 临时灌入。

**现有痛点**：(1) Reflexion 类方法把反思当成「一次性 hint」，agent 内在 policy 未更新；(2) 用 raw trajectory 检索（Case-based）容易在新任务上 overfit / 直接抄答案，而非抽象策略；(3) 用外部强教师 distill 经验，可能与 agent 自身能力分布不匹配（cognitive misalignment），小模型尤其如此；(4) Search-R1 / O2-Searcher 之类 RL agent 把「与外部搜索」的策略学得很好，但完全没解决「从自己经验中学习」的问题。

**核心矛盾**：人类专家通过「交互—反思—抽象」的连续循环成长；现有 agent 框架要么短路了反思（无状态），要么短路了抽象（raw case），要么短路了内化（仅 prompt，不更新 policy）。

**本文目标**：构造一个完整闭环 — agent 自己产生轨迹、自己蒸馏出可复用的策略原则、自己用 RL 学会用这些原则，整套系统不依赖外部教师。

**切入角度**：把「原则库」做成 agent 显式可检索的工具（与 search engine 同等地位）；让 GRPO 不仅学「怎么解题」还学「怎么用经验」。

**核心 idea**：自蒸馏原则 + 动态评分维护 + 实验经验作为 action — 把经验 lifecycle 与 RL policy 进化拧成一根绳。

## 方法详解

### 整体框架
EvolveR 主循环交替执行两阶段。**Online phase**：agent 在 Think-Act-Observe 循环中可发三类 action — `<search_experience>` 查自己的经验库 $\mathcal E$、`<search_knowledge>` 查外部搜索、`<answer>` 给最终答案；轨迹 $\tau_{\text{new}}$ 会被收集供后续训练。**Offline phase**：参数冻结，agent 用自己的 policy $\pi_\theta$ 扮演「专家」角色 review 最近一批轨迹，按结果蒸馏成「成功原则 / 失败原则」 — 每个原则 = 自然语言描述 + 若干结构化知识三元组。新蒸馏出的原则经过去重、相似度合并、动态打分维护后写入 $\mathcal E$。最后用 GRPO 在 $\tau$ 上更新 $\pi_\theta$，闭环。

冷启动用 ~700 条 NQ/HotpotQA CoT 轨迹做 LoRA SFT 稳住早期 RL，然后才正式进入 lifecycle 迭代。

### 关键设计

1. **自蒸馏 + 双层去重整合的经验库 $\mathcal E$**:

    - 功能：把 agent 自身的成功/失败轨迹转化为可检索、不冗余、能滚动更新的策略原则集合。
    - 核心思路：(a) 每条轨迹 $\tau$ 由 $\pi_\theta$ 自身（不是外部教师）按 prompt 抽出一条原则 $p_{\text{cand}}$；(b) 第一层去重 — 同问题下因 GRPO 多次采样得到的语义等价原则合并；(c) 第二层合并 — 在全库 $\mathcal E$ 内做 embedding 检索 + 二分类语义判定，若 $\max_{p\in\mathcal E}\text{sim}(p_{\text{cand}},p)<\theta_{\text{sim}}$ 则作为新条目 $\mathcal E\leftarrow\mathcal E\cup\{p_{\text{cand}}\}$，否则把 $\tau_{\text{src}}$ Merge 到最相似条目 $p^*$ 下，丰富其证据但不增加冗余。
    - 设计动机：自蒸馏避免外部教师的能力分布不匹配；双层整合避免 raw case 库爆炸 + 重复条目稀释检索质量。

2. **动态评分 + 阈值剪枝的质量控制**:

    - 功能：让经验库自我「优胜劣汰」，让真正高复用价值的原则在 retrieval 中排前面。
    - 核心思路：每条原则 $p$ 跟踪使用次数 $c_{\text{use}}(p)$ 与成功次数 $c_{\text{succ}}(p)$，按 Laplace 平滑公式 $s(p)=\frac{c_{\text{succ}}(p)+1}{c_{\text{use}}(p)+2}$ 算分。低于阈值 $\theta_{\text{prune}}$ 周期性剪掉，避免库变垃圾堆。
    - 设计动机：Laplace 平滑使得低使用次数的新原则有合理默认分（不至于一上来就被剪），高使用次数下分数趋近真实成功率；prune 是经验库长寿的关键。

3. **经验作为 RL action + GRPO 闭环训练**:

    - 功能：让 agent 不仅会读经验，更会在 policy 层面学到「什么时候该查经验、查什么经验最有用」。
    - 核心思路：reward 是 outcome reward（EM 与 ground truth）+ format reward（鼓励 think、search、answer 都至少各出现一次，且 search_experience 与 search_knowledge 都有调用）的加权和：$R(\tau)=w_o R_{\text{outcome}}+w_f R_{\text{format}}$。policy 用 GRPO 优化：$\mathcal J_{\text{GRPO}}(\theta)=\mathbb E_\tau[\sum_t \min(\rho_t \hat A_t, \text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t) - \beta D_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}]]$，每 batch 对每 prompt 采 $G=8$ 条轨迹做相对优势估计。
    - 设计动机：GRPO 无需 critic、训练稳定，相对优势在 experience-guided 轨迹间做对比，能强化「成功原则 → 成功结果」的因果连接 — 这正是 EvolveR 闭环关键。

### 损失函数 / 训练策略
冷启动：LoRA SFT (LLama_Factory) on 700 CoT 样本；RL 阶段：GRPO + Verl 框架，batch 128 prompt、$G=8$、Adam lr $1\times 10^{-6}$、warmup 20、mini-batch 128，8 张 A100。reward 中 $R_{\text{format}}=\mathbb I(\tau_{\text{complete}})\cdot (R_{\text{think}}+R_{\text{search}})/2$ — 同时奖励合理的 think 步数与多类型 search 调用。

## 实验关键数据

### 主实验
7 个 QA benchmark，分 In-domain（NQ、HotpotQA）与 OOD（TriviaQA、PopQA、2Wiki、Musique、Bamboogle）。EM 作为主指标，Qwen2.5-3B 与 7B 全面对比。

| 模型 | 方法 | NQ | HotpotQA | TriviaQA | PopQA | 2Wiki | Musique | Bamboogle | **Avg** |
|---|---|---|---|---|---|---|---|---|---|
| 3B | Direct | .106 | .149 | .288 | .108 | .244 | .020 | .024 | .134 |
| 3B | RAG | .348 | .255 | .544 | .387 | .226 | .047 | .080 | .270 |
| 3B | Search-R1-instruct | .341 | .324 | .545 | .378 | .319 | .103 | .264 | .325 |
| 3B | **EvolveR** | **.434** | **.373** | .584 | .434 | **.381** | **.137** | **.328** | **.382** |
| 7B | RAG | .349 | .299 | .585 | — | — | — | — | — |
| 7B | **EvolveR** | — | — | — | — | — | — | — | **.417** |

（更多 7B 行号原文 Table 1 给出；3B 上 EvolveR 平均 +5.7 EM 优于最强 baseline Search-R1-instruct。）

### 消融实验

| 配置 | 平均 EM 变化 | 说明 |
|---|---|---|
| Full EvolveR | 0.382 (3B) | 完整经验生命周期 |
| 去掉自蒸馏，用外部强教师 distill | 3B 下降，7B 持平 | 验证 cognitive alignment 在小模型上更重要 |
| 去掉去重 + 评分 | 库膨胀，性能下降 | curation 关键 |
| 去掉 `<search_experience>` action | 退化为 Search-R1 | RL 学不到「用自己经验」 |
| 仅 prompt + raw case 检索 | 显著下降 | 抽象原则 ≫ raw trajectory |

### 关键发现
- 在 3B 小模型上**自蒸馏胜过用更强外部教师 distill** — 这反直觉但合理：教师产出的原则可能超出 agent 自身执行能力，反而不可用。
- experience action 与 RL 的协同最关键：单纯把原则库当 RAG 用不更新 policy，提升幅度远小于 EvolveR。
- OOD 数据集（Bamboogle 等对抗多跳）上提升更显著，说明蒸馏出的「策略原则」相比记住具体事实有更好的泛化迁移。

## 亮点与洞察
- **「经验作为可学习 action」的设计**：把 `<search_experience>` 与 `<search_knowledge>` 并列为 first-class action，让 GRPO 直接对「查/不查经验」打梯度 — 这是 EvolveR 与所有 prompt-only memory 框架的本质分水岭。
- **认知对齐 (cognitive alignment)**：自蒸馏让经验匹配 agent 自身能力分布，是开放性洞察 — 暗示 self-improvement 系统的「教师」不一定越强越好。
- **闭环可滚动**：动态评分 + 剪枝让库长期可维护，避免了 ExpeL 类「经验越用越脏」的常见塌缩。

## 局限与展望
- 在 7B 模型上自蒸馏与外部 distill 表现接近，说明随着 base model 变强，「认知对齐」红利减弱 — 在 30B/70B 上是否还成立未知。
- 实验集中在多跳 QA，对长程 agentic 任务（web nav、code agent）尚未验证；而那才是 lifecycle 最该发光的场景。
- 原则库随训练增长，retrieval 延迟、embedding drift 等工程问题在长期部署中未充分讨论。
- format reward 工艺化（强制各类 action 都出现）可能引入虚假调用，是否伤害真实最优策略仍需进一步验证。

## 相关工作与启发
- **vs Reflexion / ExpeL**：他们存反思 / 经验但不更新 policy；EvolveR 经验同时驱动 retrieval 与 RL。
- **vs Search-R1 / O2-Searcher**：它们用 RL 学外部知识检索；EvolveR 多走一步，把 agent 自身经验也变成可学习目标。
- **vs Mem0 / G-Memory**：原则结构受其启发（自然语言 + 知识三元组），但 EvolveR 把这种结构嵌入完整 RL lifecycle，而非单纯做检索。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把经验生命周期 + 自蒸馏 + GRPO 三件套拧成闭环，是 agent 持续学习方向上少见的端到端方案
- 实验充分度: ⭐⭐⭐⭐ 7 个 benchmark + 多 scale 消融 + cognitive alignment 对照，覆盖面广
- 写作质量: ⭐⭐⭐⭐ 系统组件介绍清晰、Figure 1 四种 paradigm 对照直观
- 价值: ⭐⭐⭐⭐ 提供了「agent 自我进化」的可复现工程范式，对 long-horizon agent 路线有方法贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation](towards_feedback-to-plan_decisions_for_self-evolving_llm_agents_in_cuda_kernel_g.md)
- [\[ACL 2026\] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](../../ACL2026/llm_agent/mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)
- [\[ICML 2026\] Talk, Judge, Cooperate: Gossip-Driven Indirect Reciprocity in Self-Interested LLM Agents](talk_judge_cooperate_gossip-driven_indirect_reciprocity_in_self-interested_llm_a.md)
- [\[ICML 2026\] AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions](autorpa_efficient_gui_automation_through_llm-driven_code_synthesis_from_interact.md)
- [\[ICML 2026\] SE-GA: Memory-Augmented Self-Evolution for GUI Agents](se-ga_memory-augmented_self-evolution_for_gui_agents.md)

</div>

<!-- RELATED:END -->
