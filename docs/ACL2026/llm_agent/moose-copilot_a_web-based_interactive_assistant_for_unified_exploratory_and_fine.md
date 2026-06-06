---
title: >-
  [论文解读] MOOSE-Copilot: A Web-Based Interactive Assistant for Unified Exploratory and Fine-Grained Scientific Hypothesis Discovery
description: >-
  [ACL 2026][LLM Agent][科学假设发现] MOOSE-Copilot 把发散式科研 idea 探索和收敛式细粒度假设 refinement 统一到一个可视化人机协同系统中，并用初始蓝图、阶段路由和反馈三类显式人工信号显著提升科学假设发现效果。
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "科学假设发现"
  - "人机协同"
  - "探索-利用"
  - "交互式代理"
  - "MOOSE-Chem"
---

# MOOSE-Copilot: A Web-Based Interactive Assistant for Unified Exploratory and Fine-Grained Scientific Hypothesis Discovery

**会议**: ACL 2026  
**arXiv**: [2605.29475](https://arxiv.org/abs/2605.29475)  
**代码**: https://moosedemo.com (演示网站；缓存未提供 GitHub 仓库链接)  
**领域**: LLM Agent / 科学发现  
**关键词**: 科学假设发现, 人机协同, 探索-利用, 交互式代理, MOOSE-Chem  

## 一句话总结
MOOSE-Copilot 把发散式科研 idea 探索和收敛式细粒度假设 refinement 统一到一个可视化人机协同系统中，并用初始蓝图、阶段路由和反馈三类显式人工信号显著提升科学假设发现效果。

## 研究背景与动机
**领域现状**：LLM 已经被用于科研流程中的假设生成、实验设计、论文写作和评审辅助，其中“科学假设发现”位于研究流程的早期，直接影响后续实验方向和潜在价值。现有自动化发现系统大致分成两类：一类偏发散探索，从研究背景出发生成多样化高层想法；另一类偏细粒度优化，从一个初始概念出发补全方法、实验和执行细节。

**现有痛点**：这两类系统通常被当成独立任务处理。探索式系统能扩展方向，但输出往往粗糙、欠具体；细粒度系统能把一个 idea 打磨成可执行方案，却依赖已经选好的起点。更重要的是，许多 agent 工作流基本自治运行，领域专家只能在事后筛选，难以及时纠偏。

**核心矛盾**：科学发现同时需要 exploration 和 exploitation。完全自动化地在高层灵感空间和细粒度实验空间中联合搜索，会遇到巨大的组合爆炸；但如果只保留人工手工控制，又会损失 LLM 在大规模生成和局部 refinement 上的优势。

**本文目标**：作者希望建立一个统一框架，把 MOOSE-Chem 的探索式搜索和 MOOSE-Chem2 的细粒度 refinement 串起来，同时让人类专家在关键节点注入方向性信息，决定从哪里开始、何时转入细化、以及如何根据反馈重生成。

**切入角度**：论文把 human-in-the-loop 不是当成 UI 功能，而是形式化为 Human-AI Interaction Interface (HAII) protocol。也就是说，人类输入被建模为搜索过程中的路由算子和约束信号，用来裁剪搜索空间、指定粒度转换、修正当前轨迹。

**核心 idea**：用结构化人类信号把“发散搜索 + 收敛优化”的联合空间分解成一条可控的人机协同搜索轨迹。

## 方法详解
MOOSE-Copilot 的核心不是提出一个新的单阶段生成器，而是把已有的 MOOSE-Chem 和 MOOSE-Chem2 放进一个统一状态机里，并给这个状态机配上可视化交互界面。系统左侧负责探索，从背景和 inspiration corpus 中扩展 hypothesis tree；右侧负责 exploitation，把某个粗粒度节点逐层细化成更完整的研究方案。用户在中间扮演导航者，决定哪些节点值得继续探索、哪些节点应该下钻细化，以及哪些生成结果需要带着反馈再生成。

### 整体框架
输入端包括研究问题、可选 literature survey、可选 inspiration knowledge corpus，以及用户自己的初始蓝图。系统首先通过 MOOSE-Chem 把背景 $b$ 和 inspiration knowledge $i_j$ 结合，逐步生成一棵假设树；每条路径代表一串 inspiration-driven updates。随后，用户可以从树上选中某个假设节点，将其路由到 MOOSE-Chem2。MOOSE-Chem2 再用层级搜索策略，从高抽象层的 correction 开始，逐步向方法细节和实验设计层收敛。

输出端不是单个答案，而是一条带历史的 search trajectory。界面提供输入页、树视图、排名页和反馈页：树视图显示假设如何由不同 inspiration 演化；排名页展示 LLM 自评得分；反馈页允许用户选择继续探索还是进入细粒度细化，并输入定向 feedback。

### 关键设计
1. **探索-利用统一状态机**:

	- 功能：把 MOOSE-Chem 的发散探索和 MOOSE-Chem2 的细粒度 refinement 放进同一个工作流。
	- 核心思路：探索阶段近似 $P(h \mid b)$，通过不断选择 inspiration 更新中间假设；细化阶段把初始 hypothesis $h_0$ 视为待优化对象，在不同抽象层上逐步修正，使其更可执行、更一致。
	- 设计动机：单独探索容易得到粗糙 idea，单独 refinement 又缺少起点选择机制；统一框架让两者形成 exploration-exploitation 循环。

2. **三类 HAII 指导信号**:

	- 功能：让人类专家以显式、可复用的方式影响搜索过程。
	- 核心思路：$f_{init}$ 作为 initial blueprint 约束根节点搜索边界；$f_{route}$ 作为 inter-stage routing，让用户决定从概念空间 $\mathcal{C}$ 下钻到执行空间 $\mathcal{E}$，或从细化结果再回到探索；$f_{dir}$ 作为 intra-stage feedback，把 critiques 合入上下文并触发 regenerative generation。
	- 设计动机：自动系统很难判断何时继续发散、何时收敛，而领域专家通常能快速识别有潜力的分支和错误方向。

3. **交互式树形界面**:

	- 功能：降低命令行 agent 工具的使用门槛，并让假设演化过程可追踪。
	- 核心思路：每个 hypothesis 是树中的节点，用户可以视觉化查看路径、比较排名、选择节点、输入反馈，再决定下一步调用 MOOSE1 或 MOOSE2。
	- 设计动机：科学家需要的不只是自动答案，还需要知道 idea 如何生成、从哪里分叉、为什么某个节点值得继续推进。

### 损失函数 / 训练策略
本文没有训练新的大模型，主要评估系统对人类指导信号的响应能力。实验使用 oracle-simulated evaluation：oracle LLM 能访问 ground-truth fine-grained hypothesis，但只能生成方向性 critique，不能直接泄露答案；节点选择也通过 oracle ranking 模拟高质量专家路由。这样做的目的不是估计普通用户的真实表现，而是给出结构化专家信号下的性能上界。

## 实验关键数据

### 主实验
实验在 TOMATO-Chem2 上进行，包含 51 篇顶级论文的研究问题、literature survey 和细粒度 hypothesis 标注。指标是生成假设对 ground-truth elements 的 recall。

| 方法 | 主要设置 | Recall | Search Steps |
|------|----------|--------|--------------|
| baseline_MC | 只用 MOOSE-Chem 探索 | 11.44% | 未报告 |
| baseline_MC2 | 只用 MOOSE-Chem2 细化 | 10.33% | 478.6 |
| MC_with_hint | MOOSE-Chem + initial blueprint | 15.37% | 未报告 |
| MC_with_feedback_with_hint | 初始蓝图 + oracle ranking + feedback + MOOSE-Chem | 16.93% | 未报告 |
| MC2_with_MC_input_oracle_rank | 探索后用 oracle 选节点进入 MOOSE-Chem2 | 18.26% | 336.6 |
| MC2_with_feedback_oracle_rank | oracle 选节点 + 1 次 feedback refinement | 21.98% | 166.1 |
| MC2_with_strong_feedback_x4_oracle_rank | oracle 选节点 + 4 次 strong feedback refinement | 26.96% | 90.1 |

### 消融实验
| 指导信号 | 对比设置 | 观察 |
|----------|----------|------|
| 初始蓝图 | MC 11.44% vs MC_with_hint 15.37% | 初始约束能明显缩小搜索范围，提高粗粒度探索质量 |
| 阶段路由 | self-ranking 进入 MC2 为 12.74%，oracle-ranking 进入 MC2 为 18.26% | 选中哪个节点下钻几乎决定后续 refinement 的上限 |
| 方向反馈 | 普通 feedback x1 为 21.98%，strong feedback x4 为 26.96% | 更强、更明确的反馈能持续推高 recall，并减少 search steps |
| 纯自治细化 | baseline_MC2 为 10.33%，需要 478.6 steps | 没有人类路由和反馈时，细粒度搜索成本高且效果低 |

### 关键发现
- initial blueprint 把探索限制在更合理的起点附近，使系统不必在过大的概念空间中盲搜。
- routing 的作用非常大：同样是从 MOOSE-Chem 输出进入 MOOSE-Chem2，自排序节点只有 12.74%，oracle 选节点达到 18.26%。
- feedback 不只是微调语言表达，而是在细化阶段改变搜索轨迹；strong feedback x4 达到最高 recall 26.96%，同时 search steps 降到 90.1。

## 亮点与洞察
- 最有价值的地方是把“人类参与”形式化成搜索中的三个控制信号，而不是泛泛说 human-in-the-loop。这个抽象让后续系统可以比较不同人类信号的边际贡献。
- 论文很好地区分了探索式 hypothesis discovery 和细粒度 hypothesis discovery。很多科研 agent 论文只展示最终 idea，MOOSE-Copilot 更强调 idea 从粗到细的演化路径。
- 树形界面对科研场景很贴切，因为研究者通常不是一次性接受某个答案，而是在多个分支之间反复比较、回退和下钻。
- oracle-simulated evaluation 的意义在于测系统上限：如果专家信号足够好，框架能不能接得住。结果显示接得住，但也说明真实用户研究仍然必要。

## 局限与展望
- 作者明确承认系统还没有集成自动实验执行，因此从“假设生成”到“实验证伪”的闭环尚未完成。
- 系统也没有使用专门面向科学假设发现的 post-training 方法，生成质量仍依赖底层 LLM 和既有 MOOSE 系列模块。
- 当前评估用 oracle 模拟高质量专家信号，能证明协议有效，但不能直接代表真实科学家的使用成本、认知负担和反馈质量。
- 后续可以把实验执行、文献检索、失败案例回传和 hypothesis ranking 打通，让 $f_{dir}$ 不只来自人类文本反馈，也来自真实实验结果。

## 相关工作与启发
- **vs MOOSE-Chem**: MOOSE-Chem 把探索式假设发现建模为 inspiration-driven search，本文保留这个发散能力，但加入人工蓝图、路由和后续细化。
- **vs MOOSE-Chem2**: MOOSE-Chem2 专注于把一个初始假设逐级细化，本文解决的是初始假设从哪里来、何时进入细化、如何根据反馈再生成。
- **vs IdeaSynth / NOVA / LLM-SR**: 这些系统强调自动生成或迭代 idea，MOOSE-Copilot 更强调交互协议和可视化可控性。
- **启发**: 对科研 agent 来说，“可控的中间状态”可能比“端到端自动生成”更重要；把用户操作抽象为可评估信号，是设计下一代科学发现系统的关键。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 将探索与细化统一到 HAII 协议中很有辨识度，但底层生成模块主要复用 MOOSE 系列。
- 实验充分度: ⭐⭐⭐☆☆ 有清晰消融和数字，但主要是 oracle-simulated setting，真实用户实验不足。
- 写作质量: ⭐⭐⭐⭐☆ 动机、协议和界面对应关系清楚，读者容易理解系统为什么这样设计。
- 价值: ⭐⭐⭐⭐☆ 对科研 agent 的人机协同设计有直接参考价值，尤其适合复杂科学发现流程。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SR-Scientist: Scientific Equation Discovery With Agentic AI](../../ICLR2026/llm_agent/sr-scientist_scientific_equation_discovery_with_agentic_ai.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](../../ICLR2026/llm_agent/newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[ACL 2026\] Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh](mina_a_multilingual_llm-powered_legal_assistant_agent_for_bangladesh_for_empower.md)
- [\[ICML 2025\] Evaluating Retrieval-Augmented Generation Agents for Autonomous Scientific Discovery in Astrophysics](../../ICML2025/llm_agent/evaluating_retrieval-augmented_generation_agents_for_autonomous_scientific_disco.md)
- [\[ICML 2025\] Open Source Planning & Control System with Language Agents for Autonomous Scientific Discovery](../../ICML2025/llm_agent/open_source_planning_control_system_with_language_agents_for_autonomous_scientif.md)

</div>

<!-- RELATED:END -->
