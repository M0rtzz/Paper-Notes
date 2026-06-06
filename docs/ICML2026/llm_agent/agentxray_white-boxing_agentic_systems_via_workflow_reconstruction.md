---
title: >-
  [论文解读] AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction
description: >-
  [ICML 2026][LLM Agent][Agentic Workflow Reconstruction] 作者把"对黑盒 agent 系统反推一个等价白盒 workflow"作为新任务 AWR，用 MCTS 在 agent 原语序列空间中搜索…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Agentic Workflow Reconstruction"
  - "MCTS"
  - "剪枝"
  - "黑盒解释"
  - "多智能体"
---

# AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction

**会议**: ICML 2026  
**arXiv**: [2602.05353](https://arxiv.org/abs/2602.05353)  
**代码**: 论文未明确标注（无明确链接）  
**领域**: LLM Agent / 可解释性 / 组合优化  
**关键词**: Agentic Workflow Reconstruction, MCTS, Red-Black Pruning, 黑盒解释, 多智能体

## 一句话总结
作者把"对黑盒 agent 系统反推一个等价白盒 workflow"作为新任务 AWR，用 MCTS 在 agent 原语序列空间中搜索，再配上一种基于评分动态着色的 Red-Black 剪枝来平衡深度与宽度，在五个真实领域上实现可解释的白盒重建。

## 研究背景与动机

**领域现状**：LLM agent / 多智能体系统（MAS）通过 role specialization 和工具调用解决复杂任务（ChatDev、MetaGPT 等）。但实际部署的高性能 agent 通常是黑盒——内部的 prompt、agent 拓扑、工具链都不可见。

**现有痛点**：用户只能看到输入和输出，无法理解决策过程；调试、改造、安全审计都受阻；现有 agent 可解释性研究要么针对单步 LLM 推理，要么需要白盒访问（如模型蒸馏），无法对纯黑盒 API 工作。

**核心矛盾**：黑盒系统的内部状态空间巨大（agent 角色 × 模型 × 思维模式 × 工具集 × 顺序），即便允许我们采样输入输出对，也无法穷举搜索；而经典蒸馏需要模型参数，根本不适用。

**本文目标**：定义一个新任务 Agentic Workflow Reconstruction (AWR)：仅靠 $(\tau, o^\ast)$ 输入输出对，合成一个显式、可解释、可编辑的白盒 workflow，使其执行后输出与黑盒尽量一致。

**切入角度**：(1) Linearity Hypothesis——多数实际 agent 系统执行时都串行化成一条 action-observation 序列（即便设计上是图），所以把搜索空间限制为长度 $\le L_{\max}$ 的原语链。(2) 输出相似度作为 proxy 指标，绕开真值功能等价的判定难题。

**核心 idea**：把 AWR 表述为离散原语序列空间上的组合优化，再用 MCTS + Red-Black 剪枝在 token 预算下高效逼近最优 workflow。

## 方法详解

### 整体框架
输入是一个数据集 $\mathcal{D}=\{(\tau_i, o_i^\ast)\}$，每条来自黑盒系统 $\mathcal{M}_{\text{black}}$。统一原语空间 $\Omega$：每个原语 $p=\langle \rho, \mu, \pi, T_{\text{local}}\rangle$（角色、底层模型、思维模式、工具集），既包括纯推理 agent 也包括工具增强 agent。Workflow 表示为线性序列 $\mathbf{s}=[s_1,\dots,s_L]$，$L \le L_{\max}$。目标
$\mathbf{s}^\ast = \arg\max_{\mathbf{s}} \mathbb{E}_{(\tau,o^\ast)}[\mathrm{Sim}(\Phi(\mathbf{s},\tau), o^\ast)]$，
其中 $\mathrm{Sim}$ 是任务特定的代理度量（代码用 AST，文本用余弦）。AgentXRay 用 MCTS：每个节点是一条 workflow 前缀，每条边代表追加一个原语；通过 Red-Black 着色决定该节点偏好"深探（refine）"还是"广扩（branch）"。

### 关键设计

1. **统一原语空间 + Linearity Hypothesis**:

    - 功能：把异构的 agent / 工具 / 单 agent 多 agent 系统统一在同一搜索单元下，并把搜索复杂度从图拓扑 $O(2^{|\Omega|^2})$ 降到序列 $O(|\Omega|^{L_{\max}})$。
    - 核心思路：每个原语统一是 $\langle$role, model, thought pattern, local tools$\rangle$；纯推理 agent 即 $T_{\text{local}}=\emptyset$，工具增强 agent 即 $T_{\text{local}}\ne\emptyset$。再借 MacNet (Qian 2025) 指出多 agent DAG 执行时也会拓扑排序、ReAct/WebArena 等交互天然就是 ordered trace 等观察，限制搜索空间到线性序列。
    - 设计动机：纯图拓扑搜索在中等规模 $\Omega$ 上就不可行；而 "behavioral fidelity"（输入输出匹配）本身只需要复现可观测序列，而非内部拓扑，因此线性化是一个 task-aligned 的剪枝。

2. **MCTS 搜索循环（带 UCB + 早停 rollout）**:

    - 功能：处理 $\mathrm{Sim}$ 这个只有走到接近完整 workflow 才能观测的稀疏 / 延迟奖励信号。
    - 核心思路：每次迭代抽一条 $(\tau, o^\ast)$；从根选路径，到达待扩节点时执行 sample-rollout：把序列采样填到 $L_{\max}$ 并真正执行 workflow 得到输出 $o$；若执行失败 $r=0$，否则 $r=\mathrm{Sim}(o, o^\ast)$；沿路径回溯更新 $N(v), Q(v)$。每个节点的子选择用 UCB 平衡 exploration / exploitation。
    - 设计动机：与"穷举 $|\Omega|^{L_{\max}}$"不同，MCTS 用统计采样把搜索代价 amortize；UCB 在异构 action 空间（不同 role、不同模型、不同工具）中表现稳健；早停让无效原语不浪费完整 rollout 的 token。

3. **Red-Black Pruning（评分驱动的动态着色）**:

    - 功能：在固定 iteration / token 预算下，自动决定哪些节点应继续加深探索（depth refine）、哪些节点该开新分支（width expand），缓解组合爆炸。
    - 核心思路：在每次迭代前用 ColorTree 对当前树重新着色：Red 节点表示当前选择已"稳定"（评分高 + 访问次数足够），选子节点走 UCB 走深；Black 节点表示当前选择尚未充分探索，优先创建新子节点扩宽。整个 search loop（Algorithm 1）由 color-guided descent (Line 9) + 早停 rollout (Lines 11–13) + reward backprop (Line 22) 组成。
    - 设计动机：标准 MCTS 在大 $\Omega$ 上常陷入"宽得没法走深"或"深陷一个坏分支"的两端；Red-Black 把"是否有信心继续 refine 当前路径"用评分动态量化，让搜索资源被引导到"既有潜力又有深度可挖"的子树上，从而在同等 iteration 下走更深、查更优。

### 损失函数 / 训练策略
非梯度方法，无训练阶段。"损失"是负代理相似度 $-\mathrm{Sim}(\Phi(\mathbf{s},\tau), o^\ast)$，"优化器"就是 MCTS + Red-Black Pruning。每次执行 workflow 时调用真实 LLM API（用 GPT / Gemini 等），所以预算用 iteration 数 $N$ 和总 token 来度量。

## 实验关键数据

### 主实验
五个领域、五个目标系统：软件开发 (ChatDev) / 数据分析 (MetaGPT) / 教育 (TeachMaster) / 3D 建模 (ChatGPT GPT-5.2 API) / 科学计算 (Gemini 3 Pro)。代理相似度用 Static Functional Equivalence (SFE)。

| 领域 / 目标系统 | 度量 | AgentXRay 平均 SFE | 备注 |
|-----------------|------|--------------------|------|
| 软件开发 / ChatDev | AST-based | 高 SFE（综合均值 0.426） | 重建出可执行的 dev workflow |
| 数据分析 / MetaGPT | AST + 文本 | 同上 | 多 agent 协作被线性化复现 |
| 教育 / TeachMaster | 文本相似度 | 同上 | 教学流程被还原 |
| 3D 建模 / ChatGPT | 输出对比 | 同上 | 单 agent + 工具调用链 |
| 科学计算 / Gemini 3 Pro | 输出对比 | 同上 | 长链科学推理也能近似 |
| 综合 | — | 0.426 SFE | 比无剪枝基线明显更高 |

### 消融实验

| 配置 | 现象 | 解读 |
|------|------|------|
| 完整 AgentXRay（MCTS + Red-Black） | 最佳 SFE，token 减少 8–22% | 剪枝在同预算下让搜索更深 |
| 无 Red-Black 剪枝（纯 MCTS） | SFE 较低 + token 多 | 节点选择缺乏评分引导，资源被均匀分散 |
| 无线性化假设（图拓扑搜索） | 不可行 | $O(2^{|\Omega|^2})$ 搜索爆炸 |
| 不同 $L_{\max}$ | 中等长度最优 | 太短表达力不够，太长 rollout 失败率上升 |
| 不同评分函数（仅 Sim vs Sim + 深度） | 多维评分更优 | "代理质量 + 搜索深度"联合评分让 Red-Black 更敏感 |

### 关键发现
- Red-Black 剪枝是 token 效率的关键开关：同样的 iteration 预算下，剪枝后能走到更深的 workflow 层级，从而拿到更好的 fidelity。
- Linearity Hypothesis 在五个截然不同的领域（含真正多 agent 的 ChatDev、MetaGPT）下都能给出可用 fidelity，验证了"执行时拓扑序"是黑盒可观测的主要信号。
- 即便目标系统是 GPT-5.2 或 Gemini 3 Pro 这种闭源 API，AgentXRay 仍能在仅 IO 访问下逼近行为；这意味着白盒重建对真正的"商用黑盒"也有效。
- 重建出的 workflow 是可编辑的——使用者可以替换某个角色 / 工具，从而做下游适配；这是与模型蒸馏的根本差异。

## 亮点与洞察
- 把可解释性问题转化为可观测层面的"行为等价 + 结构白盒"，避免了必须打开模型参数的不可能任务；这是把"interpretability"务实落地的一个范式。
- 统一原语定义同时覆盖 agent 与工具，使搜索空间在概念上对单 agent + tool-use 系统也成立——这扩大了适用面，不只限于多 agent 系统。
- Red-Black 剪枝把"剪 vs 不剪"做成节点级的动态决策（依赖评分），比静态阈值剪枝更稳健，可迁移到任何稀疏奖励的 LLM agent 搜索。
- 用 SFE 作为代理度量绕开"true functional equivalence"的不可判定性，是面对开放式多文件输出的现实折衷，思路可复用到 code synthesis / agent eval 领域。

## 局限与展望
- Linearity Hypothesis 是上界：真正强依赖并发 / 异步多 agent 的系统（同步对话、循环反馈）可能被线性序列错过本质行为。
- 评估指标 SFE 是代理度量；在某些任务上 AST 匹配或文本相似度并不能区分真正功能差异——可能导致评分误导 MCTS。
- Rollout 需要真实执行 workflow，单次成本就是若干次 LLM 调用，搜索 $N$ 次后 token 开销巨大；目前 8–22% 的节省主要是相对的，绝对成本仍高。
- 原语空间 $\Omega$ 需要事先准备 role / model / pattern / tool 候选；如果黑盒里用了未在 $\Omega$ 的特殊 trick，则永远重建不出来。

## 相关工作与启发
- **vs 模型蒸馏**：蒸馏要参数访问、产出黑盒小模型；AWR 仅要 IO，产出白盒可编辑 workflow。
- **vs MacNet / 多 agent 图结构（Qian 2025）**：MacNet 用 DAG 训练新 agent；本文相反——从黑盒 agent 反推一个等价线性 workflow。
- **vs ReAct / WebArena 等交互 agent**：那些工作设计 agent；本文用观察来逆向 agent，给出可解释表示。
- **vs MCTS-for-LLM 路线（如 ToT、AgentTrek）**：他们用 MCTS 搜索单条推理路径；本文用 MCTS 搜索"agent 构造图本身"，是更高一层抽象。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ AWR 任务定义本身是新的，Red-Black 评分剪枝也是对 MCTS 的实质改进。
- 实验充分度: ⭐⭐⭐⭐ 五个领域 + 真实闭源 API 覆盖，但每个领域细节展开不够，统计显著性可再加强。
- 写作质量: ⭐⭐⭐⭐ 任务动机、统一原语、Linearity 论证都很清晰；Algorithm 1 写得直接可复现。
- 价值: ⭐⭐⭐⭐ 对 agent 部署的可解释 / 可控 / 可审计有直接帮助，可成为"逆向工程闭源 agent API"的实用工具。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] A2Flow: Automating Agentic Workflow Generation via Self-Adaptive Abstraction Operators](../../AAAI2026/llm_agent/a2flow_automating_agentic_workflow_generation_via_self-adaptive_abstraction_oper.md)
- [\[ICML 2026\] Answer Only as Precisely as Justified: Calibrated Claim-Level Specificity Control for Agentic Systems](answer_only_as_precisely_as_justified_calibrated_claim-level_specificity_control.md)
- [\[ACL 2026\] Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems](../../ACL2026/llm_agent/rethinking_reasoning-intensive_retrieval_evaluating_and_advancing_retrievers_in_.md)
- [\[AAAI 2026\] With Great Capabilities Come Great Responsibilities: Introducing the Agentic Risk & Capability Framework for Governing Agentic AI Systems](../../AAAI2026/llm_agent/with_great_capabilities_come_great_responsibilities_introducing_the_agentic_risk.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science?](../../ACL2025/llm_agent/repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_social_science_.md)

</div>

<!-- RELATED:END -->
