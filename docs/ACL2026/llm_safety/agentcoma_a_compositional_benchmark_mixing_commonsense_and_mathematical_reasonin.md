---
title: >-
  [论文解读] AgentCoMa: A Compositional Benchmark Mixing Commonsense and Mathematical Reasoning in Real-World Scenarios
description: >-
  [ACL2026][LLM安全][混合类型组合推理] AgentCoMa 构造了一个把常识选择和单步数学运算强制组合起来的 agentic benchmark，并在 61 个 LLM 上发现：模型通常能分别做对两个子问题，但组合后平均准确率从“两个子步都能独立答对”的 80% 掉到 51%，暴露出混合类型组合推理中的显著脆弱性。
tags:
  - "ACL2026"
  - "LLM安全"
  - "混合类型组合推理"
  - "常识推理"
  - "数学推理"
  - "Agent评测"
  - "模型脆弱性"
---

# AgentCoMa: A Compositional Benchmark Mixing Commonsense and Mathematical Reasoning in Real-World Scenarios

**会议**: ACL2026  
**arXiv**: [2508.19988](https://arxiv.org/abs/2508.19988)  
**代码**: https://agentcoma.github.io  
**领域**: LLM安全 / LLM推理评测  
**关键词**: 混合类型组合推理、常识推理、数学推理、Agent评测、模型脆弱性  

## 一句话总结
AgentCoMa 构造了一个把常识选择和单步数学运算强制组合起来的 agentic benchmark，并在 61 个 LLM 上发现：模型通常能分别做对两个子问题，但组合后平均准确率从“两个子步都能独立答对”的 80% 掉到 51%，暴露出混合类型组合推理中的显著脆弱性。

## 研究背景与动机
**领域现状**：当前 LLM 在常识推理和数学推理上都已有大量 benchmark。常识侧关注日常场景中的空间、时间、社会关系、因果知识；数学侧覆盖从小学应用题到竞赛数学的不同难度。另一方面，agentic benchmark 往往加入工具调用、长任务链、动态环境等复杂因素，用来衡量 LLM agent 的综合能力。

**现有痛点**：这些评测各自都有价值，但很难单独回答一个更细的问题：当一个任务同时需要常识判断和数学计算时，模型到底是在“组合失败”，还是只是被长上下文、工具调用、环境变化等额外因素拖累？如果 benchmark 本身混入太多干扰项，就无法精确定位错误来源。

**核心矛盾**：真实世界 agent 任务经常需要跨推理类型切换，例如先用常识判断哪些物品能常温保存，再计算总价；但现有 compositional benchmark 多半组合的是同一类推理步骤，比如两个知识检索步骤或两个数学步骤。模型在单一推理类型上表现好，并不意味着它能稳定地把不同推理类型接起来。

**本文目标**：作者希望构造一个受控评测环境，每个样本都能拆成一个常识子问题和一个数学子问题，并且每个子问题本身对人类和强模型都不难。这样一来，若模型在组合题上失败，就可以更明确地归因到“跨类型组合”本身，而不是题目单步太难。

**切入角度**：论文选择 commonsense 与 math 作为两类互补推理：前者更接近快速、直觉式的 System 1，后者更接近慢速、显式计算的 System 2。把两者放进同一个现实 agent 场景，可以测试模型是否真的会在同一条推理链中调动不同能力。

**核心 idea**：用“常识选择 + 单步算术”的人工构造任务，把 agent 场景中的混合类型组合推理从其他复杂因素里剥离出来，再通过子问题对照、人体实验和解释性分析定位 LLM 的组合脆弱性。

## 方法详解

### 整体框架

这篇论文不提新模型，而是提出一个 benchmark 加一套诊断 protocol，要回答的核心问题是：当一个任务同时需要常识判断和数学计算时，模型到底是「组合失败」，还是只是被长上下文、工具调用、环境变化等额外因素拖累？关键在于每道题都同时保留「组合题」和「两个可独立评测的子题」，于是实验能比较模型在三种输入形态上的表现：只做常识、只做数学、把两步放在一起做。

AgentCoMa 的输入是一个面向 LLM agent 的现实任务描述，输出通常是一个数值答案。一个样本包含四个核心对象：组合问题、常识子问题、数学子问题，以及三者各自的标准答案。组合问题先要求模型从多个选项中基于常识做选择，再对被选中的对象做一次算术操作——例如 garage 工具整理样例里，模型要先判断 power drill、extension cords、leaf blower 属于电器，再把对应数量相加；常识子问题只问「哪些物品应该放入防水柜」，数学子问题则把常识选择结果直接写明、只留加法计算，这样就把「知识选择失败」和「组合失败」拆开了。数据集覆盖 house working、web shopping、science experiments、smart assistant、travel agent 五个 agentic 场景，每个场景的数学步骤只含一次基本四则运算，五个场景与四种运算类型在开发集和测试集里都被均衡分布。最终规模为 260 个样本（80 开发 + 180 测试），没有训练集，因为它定位是评测预训练或指令模型。

### 关键设计

**1. 可拆解的混合推理样本：把每道组合题显式拆成常识与数学两个子题，凑成可对照的三路输入**

传统 benchmark 只能告诉你「模型答错了」，却说不清它是不会常识、不会算术、还是不会把两者连起来。AgentCoMa 让组合题的常识步骤决定参与计算的对象、数学步骤只做一次小数值运算，子题则各自删掉另一类推理负担。如果模型能分别答对两个子题、却答错组合题，就说明失败不主要来自单步能力不足，而来自跨类型组合本身——这正是它比单看最终准确率更有诊断力的地方。

**2. 人工编写与多轮专家验证：保证「子问题很简单但组合很难」这个对照关系成立**

论文的核心结论完全建立在「子题简单、组合才难」之上，一旦样本有歧义或数学步骤暗含多步推理，这个对照就塌了，所以数据质量控制本身就是方法的一部分。样本全部由专家人工编写、不用 LLM 自动生成，随后经过二元检查、独立求解、答案比对和歧义反馈，任一检查不过就重写并重新验证。为降低作者偏差，除一个评估步骤外，验证都由不同于样本作者的专家完成。

**3. 从行为到机制的诊断链：不止报准确率，还追问组合差距到底来自哪里**

如果只看到准确率下降，很容易草率归因于上下文太长或 prompt 不好。作者于是先把 AgentCoMa 与 Bamboogle、MultiArith 的组合差距相互对比，再检查额外上下文能否解释性能下降；随后用 Min-K%++ 估计混合题型与训练分布的相似度，用 lookback attention ratio 分析上下文利用程度，用 QRNCA 比较组合题与两个子题激活的相关神经元重叠。多层证据叠起来才支撑得起更强的解释：混合类型组合题在训练分布里更少见，模型推理时更倾向激活数学相关 circuit、却没有同步调动常识相关 circuit。

### 损失函数 / 训练策略
本文没有训练新模型，因此不存在模型损失函数。实验采用统一的推理与评测策略：所有 LLM 用 two-shot chain-of-thought prompt 和 greedy decoding 生成答案；数值答案从 CoT 输出中用正则抽取最终数字，并与标准答案精确匹配；常识子题的非数值答案由 LLM-as-a-judge 对照标准答案判定。作者还额外测试 self-ask 分解式 prompting，发现平均组合差距仍接近 CoT 设置，说明简单提示分解不能消除 AgentCoMa 暴露的问题。

人类对照实验使用 45 名非专家 crowdworker，要求具备高中教育水平和英语流利度，并且不用计算器或搜索工具手算。每名标注者回答 12 个问题，且同一样本的组合题和子题不会交给同一个人，以减少答案泄漏。

## 实验关键数据

### 主实验
论文一共评测 61 个 LLM，并选取 16 个近期模型展示细粒度结果，覆盖 instruction-tuned、SFT reasoning、RL reasoning 三类训练策略，模型大小从代表性表中的 3B 到 141B 不等。核心发现是：模型平均可以在 80% 的样本上分别答对两个子步骤，但组合题平均准确率只有 51%，形成 29% 的平均 compositionality gap。

| 对象 / 基准 | 子步都独立答对 | 组合准确率 | 组合差距 / 说明 |
|-------------|----------------|------------|-----------------|
| AgentCoMa 上 16 个代表性 LLM 平均 | 80% | 51% | 平均差距 29%，即主要问题出在组合而非单步能力 |
| AgentCoMa 非专家人类 | 78.9% | 82.8% | 人类没有明显组合崩塌，组合题甚至略高 |
| Bamboogle 上 LLM 平均 | 53% | 52% | 两步同属知识推理，组合差距可忽略 |
| MultiArith 上 LLM 平均 | 近乎完美 | 近乎完美 | 两步同属数学推理，平均组合差距小于 1% |

从具体模型看，强模型也没有完全免疫。例如 Llama3.3 70B IT 的两个子步都答对比例为 90.0%，组合准确率为 73.3%；Qwen3 14B 为 88.9% vs 60.6%；SimpleRL 32B 为 93.9% vs 66.7%。较小或较弱模型的崩塌更明显，如 SimpleRL 8B 的两个子步都答对比例为 56.7%，组合准确率只有 25.0%。

### 消融实验
这篇论文的“消融”更像诊断实验：作者逐步排除上下文长度、prompt 分解、同类组合难度等解释，再观察注意力和神经元模式。

| 分析项 | 关键结果 | 说明 |
|--------|----------|------|
| 失败来源分解 | AgentCoMa 组合失败中约 0.74 来自“两子步都能单独答对”的样本 | 失败不是因为模型不会做单步，而是不会可靠组合两类推理 |
| Self-Ask prompting | 平均组合差距约 27%，CoT 下为 29% | 显式分解提示只能小幅缓解，不能解决核心问题 |
| Lookback attention | commonsense 71.49，math 72.20，composition 70.75 | 组合题中模型对上下文的回看注意力更低，更容易出现上下文幻觉 |
| Neuron overlap: Llama 3.1 8B | composition-math 39%，composition-commonsense 3% | 组合题更像激活数学 circuit，而不是同时激活常识 circuit |
| Neuron overlap: GeneralReasoner 4B | composition-math 54%，composition-commonsense 10% | reasoning 模型也呈现相似偏向，说明问题不只属于普通指令模型 |

### 关键发现
- AgentCoMa 的差距明显大于同类推理组合 benchmark。MultiArith 主要是数学加数学，Bamboogle 主要是知识加知识；它们的组合准确率与子步联合正确率接近，而 AgentCoMa 的红线明显断开。
- RL 或 SFT 的 reasoning 优化没有消除该问题。论文中特别指出，reasoning 模型和 instruction-tuned 模型一样会出现大组合差距，这说明“会长链推理”不等于“会跨类型组合”。
- 训练分布相似度分析支持一个合理解释：混合常识与数学的组合题型相对少见。Min-K%++ 分数显示，组合问题比单独常识或单独数学问题更不像模型训练中常见的模式。
- 机制分析给出更具体的失败图景：模型面对组合题时，往往主要沿着数学相关神经元模式走，常识相关神经元没有被充分激活，因此会做出形式上流畅但上下文使用错误的推理。

## 亮点与洞察
- 最有价值的设计是把 benchmark 做成“组合题 + 两个子题”的三联结构。它把通常含混的错误诊断变成了可量化对照：模型不是不会买东西、也不是不会加法，而是把“该买什么”和“怎么算钱”接起来时掉链子。
- AgentCoMa 的难点不是题目绝对难，而是推理类型切换难。这对 agent 评测很重要，因为真实任务里很多失败并不来自高级数学或长规划，而来自把简单能力按正确顺序组合。
- 论文没有停在排行榜层面，而是把行为差距连接到训练分布、注意力和神经元重叠。虽然这些解释性分析不等同于因果证明，但它们让“混合类型任务是未充分学习模式”这个结论更可信。
- 对安全与可靠性评估也有启发：如果模型在低风险、短上下文、无工具的受控题里都不能稳定组合常识和算术，那么在真实 agent 场景中，把模型部署为自动决策者时需要额外的分步验证和中间状态检查。

## 局限与展望
- AgentCoMa 是刻意简化的受控实验。每道题只组合两类推理，数学部分也限制为一次基本运算；这有利于定位问题，但不能覆盖真实 agent 任务中的长任务链、多约束、多轮交互和工具调用。
- 数据规模相对小，测试集为 180 个样本，总样本数 260。这个规模与 Bamboogle、MultiArith 等经典组合 benchmark 接近，但如果要细分场景、算术类型、推理顺序和语言变体，仍需要更大数据。
- 论文只初步探索了推理顺序影响。作者在局限中提到，构造完整 reversed-order 数据集需要大量专家标注，因此当前还不能系统回答“先数学后常识”和“先常识后数学”的差异。
- 常识子题的自动评测依赖 LLM-as-a-judge。作者做了与人类评估相关性的验证，但在更开放的常识答案空间中，评测器偏差仍可能影响细粒度结论。
- 后续可以扩展到更多混合类型，例如常识 + 约束规划、空间推理 + 算术、社会常识 + 资源分配，并把 benchmark 接入实际 agent trace，观察受控差距是否能预测真实任务失败。

## 相关工作与启发
- **vs 常识推理 benchmark**: 常识 benchmark 主要测试模型是否掌握日常知识和直觉推理，本文则把常识作为组合链条的一环，重点看它是否能与显式计算协同。
- **vs 数学推理 benchmark**: MultiArith、GSM 类任务考察数学计算或数学文字题能力；AgentCoMa 的数学步骤本身很简单，真正考察的是模型是否能先用常识选对计算对象。
- **vs Bamboogle / MultiArith 组合推理**: 这些 benchmark 也有多步结构，但通常组合的是同一类型推理。本文的关键差异是跨类型组合，因此能暴露同类组合任务看不到的脆弱性。
- **vs agentic benchmark**: 许多 agent benchmark 同时包含工具调用、长 horizon 和动态环境，评测更贴近真实部署但不易归因。AgentCoMa 牺牲一部分真实复杂度，换来更清晰的因果诊断窗口。
- **对后续研究的启发**: 训练与推理方法不应只奖励最终答案或长 CoT，还应显式监督中间类型切换。例如可以训练模型先标注当前需要的推理类型、输出被选择对象，再进入计算；也可以用 verifier 检查常识选择是否被正确传递到数学步骤。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 选取“常识 + 数学”的混合类型组合推理作为受控 benchmark，问题定义清楚且有辨识度。
- 实验充分度: ⭐⭐⭐⭐☆ 61 个 LLM、人类对照、同类 benchmark 对比和多种解释性分析都比较扎实，但数据规模和 reversed-order 分析仍可扩展。
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构清晰，从数据构造到行为结果再到机制分析层层推进，主结论和支持证据对应得很好。
- 价值: ⭐⭐⭐⭐⭐ 对 LLM agent 可靠性评估很有价值，提醒我们不要把单项推理能力误读为真实场景中的稳定组合能力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](../../ICLR2026/llm_safety/measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)
- [\[ICLR 2026\] Moving Beyond Medical Exams: A Clinician-Annotated Fairness Dataset of Real-World Tasks and Ambiguity in Mental Healthcare](../../ICLR2026/llm_safety/moving_beyond_medical_exams_a_clinician-annotated_fairness_dataset_of_real-world.md)
- [\[NeurIPS 2025\] SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications](../../NeurIPS2025/llm_safety/swe-sql_illuminating_llm_pathways_to_solve_user_sql_issues_in_real-world_applica.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)
- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)

</div>

<!-- RELATED:END -->
