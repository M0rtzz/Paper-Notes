---
title: >-
  [论文解读] Knowing Isn't Understanding: Re-Grounding Generative Proactivity with Epistemic and Behavioral Insight
description: >-
  [ICML2026 (Position Paper)][生成式主动性] 这是一篇 ICML2026 Position 论文，主张生成式智能体的"主动性"不能只看是否更早、更自主、更持续地行动…
tags:
  - "ICML2026 (Position Paper)"
  - "生成式主动性"
  - "认识论-行为耦合"
  - "未知的未知"
  - "承诺校准"
  - "认识论伙伴"
---

# Knowing Isn't Understanding: Re-Grounding Generative Proactivity with Epistemic and Behavioral Insight

**会议**: ICML2026 (Position Paper)  
**arXiv**: [2602.15259](https://arxiv.org/abs/2602.15259)  
**代码**: 无（立场论文）  
**领域**: LLM 对齐 / 主动式智能体 / 认识论建模  
**关键词**: 生成式主动性, 认识论-行为耦合, 未知的未知, 承诺校准, 认识论伙伴

## 一句话总结
这是一篇 ICML2026 Position 论文，主张生成式智能体的"主动性"不能只看是否更早、更自主、更持续地行动，而必须由两条联合约束规制——认识论合法性（agent 是否真的"理解"了情境）与行为承诺度（介入是否可逆、是否被迫升级），并把幻觉、对齐失败、不安全自治重新解释成"知道 / 行动"之间的错耦合（mis-coupling）。

## 研究背景与动机

**领域现状**：当前主动式智能体研究主要沿三条路径堆能力——(i) 预测式 IR / 推荐（anticipatory）：从历史信号外推下一步需要什么；(ii) 自主规划 / 工具调用 LLM agent（autonomous）：把"主动"等同于多步执行 + 自我反思；(iii) 混合主动（mixed-initiative）：把"何时介入、以何强度介入"建成显式控制变量。所有这些都把主动性看作"在既定任务框架下的动作选择"，认识不确定性被降级为对已知变量的置信度。

**现有痛点**：把无知一律塌缩成"对已知维度的不确定性"会丢掉三类东西——(a) 错误被当成知识（error-as-knowledge：LLM 给出流畅自信但错的解释）；(b) 信号被压制（denial：为了维持任务推进而平滑掉异常信号）；(c) **未知的未知**（unknown unknowns, UU）——既不在任务框架里，也无从被置信度刻画。表 1 把现有工作按其能触及的最高认识论状态分类（KK / KU / UK / UU），结果是几乎没有任何主流方法能达到 UU。

**核心矛盾**：把主动性当成"更强的初始化能力"会系统性放大风险——介入越早、越果断，越会改写环境本身、抹掉本可暴露错耦合的证据；优化目标只看任务完成 / 连贯性 / 速度，等价于在奖励 "behavioral momentum"，而对认识论稳健性几乎不给信号。

**本文目标**：(i) 显式地把认识论维度（什么能被合法宣称为已理解）作为主动性的第一性约束；(ii) 给出诊断框架解释"幻觉""自治不安全""对齐失败"这些表面失败的共同结构性根源；(iii) 提出 epistemic partnership 作为下一代主动式 agent 的方向。

**切入角度**：借用 Kerwin 的无知哲学（ignorance philosophy）和 Parker 等的组织行为学"倒甜甜圈"（inverted doughnut）模型——前者把无知拆成 error / tacit / taboo / denial / UU 等结构化形式，后者把"自主行为是否合法"约束在 role scope + recoverability + 社会反馈三维空间。两条线分别解决"知道什么"和"以多大力度做"，但单独使用都漏掉关键约束。

**核心 idea**：把主动性建模成"承诺 × 合法性"二维联合空间，并要求二者保持**耦合**——承诺度（commitment）必须随认识论合法性（epistemic legitimacy）动态降档；当合法性下滑时，行动必须可逆、可中断、能放大不确定性而非平滑掉它。

## 方法详解

### 整体框架
本文不提算法，而是给一个**诊断 + 设计原则**框架，分四步推进：(1) 综述现有主动性范式，定位其共同盲点；(2) 引入认识论奠基（epistemic grounding）讨论"无知不只是不确定性"；(3) 引入行为奠基（behavioral grounding）讨论"主动不等于更多初始化"；(4) 提出 epistemic-behavioral coupling 联合模型，把现有失败模式归类为耦合失配，并给出 5 个开放研究问题与最小行为约束清单。

### 关键设计

**1. 认识论奠基（Epistemic Grounding）：把无知从"对已知变量的置信度"拆成多种结构化形式，让 agent 能显式表示"什么我没建模"**

当前主动 agent 失败的根因是把 ignorance ≈ uncertainty——任务框架本身错的时候，置信度反而被自我强化（impoverished model 下的 low uncertainty），而不是预警。本文援引 Kerwin 的 ignorance philosophy，把无知细分成五种：uncertainty（已知变量上的置信不足）、error（把错的当对的并捍卫之）、tacit（隐性可执行但说不出口）、taboo（被规范或激励禁问的问题）、denial（主动压制威胁性信息）——这些形式没有一个能被概率建模捕捉到。表 1 用这套分类盘点了 7 个代表性范式（Anticipatory IR、Web/OS agent、Planning+Tool LLM、Mixed-initiative 等），结论是所有主流方法的"认识天花板"最高只到 UK，UU（未知的未知）完全没人碰。要破这一点，agent 必须拥有显式表示"什么我没建模"的能力，这是 confidence calibration 根本修不到的层级。

**2. 行为奠基（Behavioral Grounding）：用"倒甜甜圈 + 可逆性边界"约束介入力度，避免认识论合法但行为越界**

光知道得对还不够，还要约束"以多大力度、多大范围、多强承诺去介入"。本文借组织行为学 Parker et al. (2010) 的 inverted doughnut model——中心是 prescribed core（必须执行的核心责任），中圈是 discretionary zone（鼓励主动），外圈是 overreach（越界，社会成本高）。但作者尖锐指出该模型只规制"沿 role scope 的偏离"，**不规制行动者对情境的理解是否正确**；人类靠社会反馈、机构信号补齐这一层，agent 却几乎没有这些稳定信号——它的优化目标只奖励 task completion，恰恰把"放手退档"的行为系统性地不奖励。换句话说，把人类组织里"靠规范加反馈自我设限"的行为主动性原样搬到 agent 上，等于只给马力不给刹车；所以必须给行为侧加一个新硬约束——commitment 必须与 epistemic recoverability 联动。

**3. 认识论-行为耦合（Coupling）：把主动性建在 (承诺, 合法性) 的二维空间里，把幻觉、runaway、信号压制统一诊断为错耦合**

本文把主动性放进 (commitment, epistemic legitimacy) 的二维联合空间，分四象限：（高合法 + 低承诺）= 观察 / 澄清；（高合法 + 高承诺）= 合理介入；（低合法 + 低承诺）= 探索 / 试探；（低合法 + 高承诺）= **epistemic overreach**。三种典型失败由此被统一解释——epistemic overreach（hallucination 被工具调用放大）、suppressed epistemic signals（连贯性奖励压制掉异常证据）、runaway commitment under false certainty（自反思 agent 把 error 强化成知识）。据此给出 4 条最小行为约束：commitment 必须随 recoverability 缩放；主动行为必须保留而非压制不确定性；commitment 必须能被认识论退化中断；不确定性必须主动调制初始化而非事后标注。这个框架最有冲击力的判断是：真正的控制变量不是 autonomy（谁能行动）而是 commitment（行动有多不可逆）——一旦把两轴联合起来，"幻觉 = 高承诺低合法"和"恭顺 = 低承诺高合法"就不再是孤立现象，而是同一空间里可一致评估和约束的不同点。

### 训练策略 / 实现指引（针对立场论文）
本文不给具体算法，但给出**五条研究议程**（Q1-Q5）：怎么表示 epistemic legitimacy？哪些信号必须在行动中保留？怎么及时检出退化？何时退档 / 弃权才算"正确的主动"？怎么评估 coupling quality（在行动时而非事后判断）？第 7 节进一步指向 epistemic partnership——三个能力：主动问 UU、长视野思考、test-time proactivity（部署期实时调节 initiative）。

## 实验关键数据

立场论文无定量实验。下面两表是论文的**核心定性分析**——一张盘点现状，一张分类失败模式。

### 主"对比表"：现有主动性范式的认识论天花板（论文 Table 1 重排）

| 主动性范式 | KK | KU | UK | UU | 结构性缺口 |
|------------|----|----|----|----|------------|
| Anticipatory IR / Proactive Retrieval | ✓ | ✓ | ✗ | ✗ | 只能在已知信息空间里预测，UK/UU 完全够不着 |
| Sequential / Basket Recommendation | ✓ | ✗ | ✗ | ✗ | 只在固定 catalog 上选择，连 KU 都不显式建模 |
| Web/OS/Embodied Agent | ✓ | ✗ | ✗ | ✗ | benchmark 把成功定义死，没有"重定义任务"的接口 |
| Planning + Tool-using LLM | ✓ | ✓ | ✗ | ✗ | 优化已知工具下的动作，不重构应建模什么 |
| Proactive Conversational (human-centered) | ✓ | ✓ | ✓ | ✗ | 能调何时介入但还在预设维度里 |
| Mixed-initiative Clarification | ✓ | ✓ | ✓ | $\sim$ | 能掘出隐性意图，但难以surface "缺失维度" |
| **本文 epistemic partnership（愿景）** | ✓ | ✓ | ✓ | ✓ | 唯一显式以 UU 为一阶目标 |

### 失败模式分类表（论文第 5 节归纳）

| 失败模式 | 二维定位 | 典型表现 | 现有缓解机制 | 为何不够 |
|----------|----------|----------|--------------|----------|
| Epistemic overreach | 高承诺 + 低合法 | LLM 流畅自信地调工具改外部状态 | confidence calibration | 在错框架下的"高置信"被当成 epistemic OK |
| Suppressed signals | 高承诺 + 退化中的合法性 | self-improve loop 把异常平滑掉 | uncertainty estimation | 优化目标奖连贯，置信度跨分布漂移时反而升高 |
| Runaway commitment | 升级中的承诺 + denial | 反思 agent 把 error 加固为 knowledge | self-reflection | 反思本身被同样的"完成度"信号驱动 |
| Premature steering | 高承诺 + UK 未处理 | 早期决定性行动抹掉本可暴露错误的证据 | mixed-initiative 协调 | 协调时已经介入，太晚 |

### 关键发现
- **真正缺失的控制变量是 commitment 不是 autonomy**——这是本文最有冲击力的判断。permissioning / tool access 控的是 autonomy；真正决定害处的，是行动有多大幅度地改写未来状态。
- **当前 benchmark 系统性奖错的人**：reward 给 task completion / coherence / speed 等价于 reward 给 momentum，自然挑选出"宁可错也要前进"的策略——这一点直接挑战了 ReAct/Reflexion 等主流评估范式。
- **认识论合法性不能由 confidence 单值充当**——KK/KU/UK/UU 是四种不同状态，需要四种不同的代理变量（agent representation 还远没到这一步）。
- **proactive 与 restraint 是同一硬币的两面**：epistemic partnership 的判定标准不是"问得多 / 帮得多"，而是"是否在该退档时退档"——这对评估协议有立刻可操作的影响。

## 亮点与洞察
- **借哲学武器**：从 Kerwin ignorance philosophy 抽出 error / tacit / taboo / denial / UU 这五种"非概率无知"，给 ML 圈一种久违的、能解释 hallucination 真因的概念语言；这种"非置信式不确定性"完全可以拓展到 RAG、tool-use safety、自动驾驶等场景。
- **借社会科学武器**：Inverted doughnut model 直接给 agent 设计提供了"discretionary zone vs overreach"的可视化模型——比"reward shaping + RLHF"那套更接近企业治理。
- **重新定义"对齐"**：本文实质上把 alignment 从 "value alignment" 拓展为 "commitment alignment"——对齐的不只是目标，更是"在合法性不足时收手"的能力。这给 RLHF / DPO / Constitutional AI 之外打开了第三条路线。
- **可迁移的设计原则**：四条最小行为约束（recoverability scaling / signal preservation / interruptibility / active modulation）是可立刻落地到现有 agent 框架的 hard constraint；任何 ReAct / AutoGen / OpenAI Assistants 风格的实现都可以在其 tool-use 层强加它们。

## 局限与展望
- **没给实现路线图**：四条最小行为约束如何 operationalize 仍是开放的——比如 "commitment must be interruptible by epistemic degradation" 需要先把"epistemic degradation"做成可探测信号；论文承认这本身就是 Q3 开放问题。
- **未覆盖多 agent / 社会规模耦合**：耦合模型是单 agent 视角；当多个主动 agent 同时在场（multi-agent / agentic society），合法性 / 承诺如何在系统级耦合未讨论。
- **评估协议缺失**：作者主张"评 coupling quality 要在行动时评不能事后评"，但没给具体 benchmark / metric——这是后续工作非常关键的一步。
- **认识论分类的可学性**：KK/KU/UK/UU 是哲学层的分类，能否被现行 LLM 学到、能否泛化、能否抗投毒，全是空白。
- **正向改进**：(i) 把 epistemic degradation 操作化为 OOD score / consistency check / counterfactual probing 等具体信号；(ii) 设计 "abstention reward" 让 RL agent 学到 calibrated restraint；(iii) 构造显式 UU benchmark——例如"任务描述刻意省去关键变量，看 agent 是否会问而非猜"。

## 相关工作与启发
- **vs Horvitz (mixed-initiative) 1999/2007**：本文承认 mixed-initiative 给出了"何时介入"的精彩理论，但批评它默认任务框架已被正确指定；本文加的是"先问框架本身是否合法"。
- **vs ReAct / Reflexion / planning agent 系列**：本文把这些工作归类为"高 autonomy 但 commitment 不被显式管控"的代表——其行为 momentum 正是失败放大器。
- **vs Hendrycks et al. distribution shift 系列**：本文借用了"calibration degrades under shift"的实证发现，但把它从"confidence is unreliable"提升到"confidence 本身代表错了的东西"。
- **vs 近期 epistemic agent 工作（COLLABLLM, DYNA-THINK, ProPer）**：本文承认这些已朝 epistemic partnership 走，但仍把 collaboration ≈ more interaction；本文要求 collaboration = calibrated intervention，把约束移到了更前置的层级。
- **vs Constitutional AI / RLHF**：那些主要做 value alignment（行为不该是什么）；本文做 commitment alignment（行动有多坚定该取决于知识有多扎实），二者正交。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "认识论-行为耦合"这一框架在 ML 文献里前所未见，把 ignorance philosophy 严肃引入主动 agent 设计，是真正的概念突破。
- 实验充分度: ⭐⭐⭐ 立场论文性质，无定量实验；论证靠综述 + 概念演绎，但提出的 Q1-Q5 可指导实证后续。
- 写作质量: ⭐⭐⭐⭐ 概念推导有结构感，表 1 的盘点尤其清晰；少数段落（第 5/6 节）有概念密度过高、读者负担偏重的问题。
- 价值: ⭐⭐⭐⭐⭐ 对当前 agent / alignment 社区有矫正性影响——把 "更主动 / 更自治"的隐性目标揭穿成可能放大伤害的优化方向，且给出可立即试用的最小约束清单。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Principled Understanding of Generalization for Generative Transformer Models in Arithmetic Reasoning Tasks](../../ACL2025/others/principled_generalization_arithmetic.md)
- [\[ICML 2026\] On the Epistemic Uncertainty of Overparametrized Neural Networks](on_the_epistemic_uncertainty_of_overparametrized_neural_networks.md)
- [\[AAAI 2026\] Why Isn't Relational Learning Taking Over the World?](../../AAAI2026/others/why_isnt_relational_learning_taking_over_the_world.md)
- [\[AAAI 2026\] An Epistemic Perspective on Agent Awareness](../../AAAI2026/others/an_epistemic_perspective_on_agent_awareness.md)
- [\[ICML 2026\] MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation](mmd-balls_as_credal_sets_a_pac-bayesian_framework_for_epistemic_uncertainty_in_t.md)

</div>

<!-- RELATED:END -->
