---
title: >-
  [论文解读] StratMem-Bench: Evaluating Strategic Memory Use in Virtual Character Conversation Beyond Factual Recall
description: >-
  [ACL2026][LLM评测][战略记忆使用] StratMem-Bench 把角色对话中的记忆分成必需、可增益和无关三类，并用 SMC、MIQ、PES、CIR 四个指标评测模型是否能在事实正确、主动丰富和避免跑题之间做战略选择。
tags:
  - "ACL2026"
  - "LLM评测"
  - "战略记忆使用"
  - "虚拟角色"
  - "长期记忆"
  - "对话生成"
---

# StratMem-Bench: Evaluating Strategic Memory Use in Virtual Character Conversation Beyond Factual Recall

**会议**: ACL2026  
**arXiv**: [2604.26243](https://arxiv.org/abs/2604.26243)  
**代码**: https://github.com/seucoin/StratMem-Bench.git  
**领域**: LLM 评测 / 虚拟角色对话 / 长期记忆  
**关键词**: strategic memory use、虚拟角色、长期对话、记忆选择、LLM 评测

## 一句话总结
StratMem-Bench 将虚拟角色对话中的记忆分成 must、nice 和 irr 三类，评估模型是否能在保证事实需求的同时主动加入有益记忆并抑制无关记忆，揭示当前强 LLM 在“支持性记忆选择”上仍明显不稳。

## 研究背景与动机
**领域现状**：长期对话和虚拟角色系统通常会给模型配备外部记忆，让角色能记住过去经历、用户偏好和人物设定。现有 benchmark 多数评估 factual recall，也就是相关事实是否被找回并反映在回答中。

**现有痛点**：人类对话中的记忆使用不是越多越好。某些记忆是回答问题必须使用的，有些只是能让回答更自然、更有共情或更个性化，还有些虽然在记忆库里但当前场景不该提。如果 benchmark 只看是否召回事实，就无法衡量这种选择能力。

**核心矛盾**：虚拟角色既要像真实人一样主动、贴心，又不能把无关私事硬塞进回答。模型需要在 proactivity 和 risk aversion 之间动态平衡。

**本文目标**：构建一个明确区分 required、supportive 和 irrelevant memories 的评测集，并设计指标衡量模型是否能“该用的全用、可用的适度用、不该用的不用”。

**切入角度**：论文借鉴 Gricean Maxims，把 must 记忆对应事实正确性，把 nice 记忆对应适量信息与社交连贯性，把 irr 记忆对应相关性违反。

**核心 idea**：把记忆从静态事实仓库改写为对话中的动态资源，评估模型在当前 query、persona 和 history 下对每条记忆的功能性判断。

## 方法详解
StratMem-Bench 的任务是条件响应生成。每个样本包含对话历史、当前用户问题、角色 persona 和未标注的记忆池。模型看不到 must、nice、irr 标签，必须自己判断哪些记忆对当前回答有功能贡献。

### 整体框架
数据来源于 LoCoMo 的多会话虚拟角色对话。作者从前若干 session 中抽取记忆池和 persona，再用后续 session 作为当前对话历史，生成一个当前用户 query。这样保证记忆来自过去，回答发生在之后，避免时间泄漏。

每条记忆按当前 query 的功能角色标注为三类。must 是回答正确性所必需的记忆；nice 不是正确性必需，但能提升个性化、共情或社交连贯性；irr 与当前对话目标无关，使用后会偏题或显得突兀。

标注流程先由 GPT-5.1 生成初始标签，再由多人类专家复核和讨论。作者报告三名标注者在专家讨论前的 Fleiss' kappa 为 0.81，说明角色划分虽然有主观性，但协议相对稳定。

### 关键设计
1. **三类记忆角色的 instance-level 标注**:

    - 功能：让同一条记忆在不同 query 下可以拥有不同角色。
    - 核心思路：标签不由关键词重叠决定，而由该记忆对当前对话目标的功能贡献决定。例如“搬到新城市”在询问住址时是 must，在询问近况时可能是 nice，在询问音乐偏好时是 irr。
    - 设计动机：虚拟角色的记忆使用本质上是语境决策，如果把记忆固定为相关或不相关，会忽略对话目标的变化。

2. **Strict Memory Compliance**:

    - 功能：用硬性规则衡量模型是否满足战略记忆使用的基本约束。
    - 核心思路：must-only 样本要求所有 must 被用且 irr 不被用；nice-only 要求至少用一条 nice 且不用 irr；must+nice 要求 must 全用、nice 至少用一条且 irr 不被用。
    - 设计动机：传统平均质量分会掩盖严重错误，SMC 把战略选择先变成 pass/fail，便于发现记忆选择瓶颈。

3. **MIQ、PES 与 CIR 行为画像**:

    - 功能：把“选对记忆”和“用好记忆”分开评估。
    - 核心思路：MIQ 用 1-5 分评价被选记忆是否自然融入回答；PES 衡量模型在可使用 nice 记忆时是否主动丰富回答；CIR 衡量存在 nice 记忆时模型误用 irr 的比例。
    - 设计动机：有些模型很保守，CIR 低但回答不够丰富；有些模型很主动，PES 高但容易引入无关记忆。两个指标一起看才能描述真实行为。

### 损失函数 / 训练策略
这是一篇 benchmark 论文，没有训练新模型。评测时所有模型使用统一 instruction template，输入 persona、history、query 和未标注 memory pool，零样本生成单轮回答。作者把 nice 记忆过多的样本下采样到两条 nice 记忆，以降低随机碰中 nice 的概率。

自动评估由 DeepSeek-V3.2 完成。记忆使用检测要求评估器引用回答中的具体证据，并重复采样三次做多数投票；与人类专家在 1,130 个 memory-response pair 上的 Cohen's kappa 为 0.96。MIQ 与 300 条人工标注回答的 Cohen's kappa 为 0.69。

## 实验关键数据

### 主实验
数据集共有 657 个样本，must+nice 场景占多数，也是最接近真实角色对话的困难设置。

| 场景 | 样本数 | 平均记忆条数 | 平均每条记忆词数 | 评测含义 |
|------|--------|--------------|------------------|----------|
| must-only | 50 | 6.24 | 9.53 | 只需满足必要记忆并抑制无关记忆 |
| nice-only | 132 | 9.12 | 10.09 | 没有硬性事实需求，考察主动丰富 |
| must+nice | 475 | 8.97 | 9.75 | 同时满足事实、丰富性和抑制无关信息 |
| Overall | 657 | 8.79 | 9.81 | 全量评测 |

| 模型 | SMC must-only | SMC nice-only | SMC must+nice | SMC All | MIQ All on pass |
|------|---------------|---------------|---------------|---------|-----------------|
| GPT-5.2 | 88.00 | 57.58 | 41.89 | 48.55 | 4.45 |
| GPT-5-chat | 90.00 | 46.21 | 41.68 | 46.27 | 4.56 |
| Claude Sonnet 4.5 | 90.00 | 53.03 | 46.95 | 51.45 | 4.37 |
| Gemini 3 Pro | 78.00 | 49.24 | 48.21 | 50.68 | 4.21 |
| DeepSeek-reasoner | 76.00 | 48.48 | 39.16 | 43.84 | 4.12 |
| Qwen3-235B | 92.45 | 46.56 | 42.28 | 47.18 | 4.24 |

强模型在 must-only 上普遍能达到 76%-92% SMC，但 nice-only 和 must+nice 明显下降。也就是说，模型能处理“必须事实”，但不擅长判断“可以让对话更好的支持性记忆”。

### 消融实验
这篇论文没有模型结构消融，但提供了行为维度分解，可以视作评测诊断表。

| 模型 | must-used MIQ | nice-used MIQ | irr-used MIQ | PES All | CIR All |
|------|---------------|---------------|--------------|---------|---------|
| GPT-5.2 | 4.48 | 4.22 | 2.99 | 56.01 | 13.01 |
| GPT-5-chat | 4.55 | 4.38 | 2.81 | 51.91 | 7.91 |
| Claude Sonnet 4.5 | 4.36 | 4.18 | 3.05 | 62.48 | 15.82 |
| Gemini 3 Pro | 3.92 | 3.73 | 2.63 | 73.33 | 31.96 |
| DeepSeek-chat | 4.32 | 4.12 | 2.75 | 56.96 | 15.49 |
| Qwen3-Max | 4.14 | 4.04 | 2.64 | 57.76 | 19.77 |

### 关键发现
- must 记忆一旦被正确选择，MIQ 往往较高，说明当前瓶颈主要在“选择哪些记忆”，而不是语言表达能力。
- nice 记忆带来 enrichment tax：nice-used MIQ 通常比 must-used MIQ 低约 0.2，说明额外个性化信息更容易造成生硬或牵强整合。
- irr 记忆会导致质量崩塌，irr-used MIQ 多在 2.6-3.1，说明无关记忆不仅偏题，也会破坏角色回答的连贯性。
- PES 和 CIR 存在清晰 trade-off。GPT-5-chat 的 CIR 最低约 7.91，但 PES 也较保守；Gemini 3 Pro 的 PES 最高约 73.33，但 CIR 达到 31.96，主动性伴随更高无关记忆风险。

## 亮点与洞察
- 这篇论文把长期记忆评测从“能不能记得”推进到“该不该说”。对真实角色和个人助理系统来说，这比单纯 recall 更接近产品风险。
- must、nice、irr 的划分非常实用。它承认对话质量不只是事实正确，还包括适度个性化和相关性控制。
- SMC 是硬指标，MIQ 是质量指标，PES/CIR 是行为倾向指标。四者组合能把模型失败拆成“没用必需记忆、不会主动丰富、乱用无关记忆、整合质量差”等不同模式。
- 论文发现强模型在 SMC-pass 时 MIQ 仍高，说明提升战略记忆使用可能需要更好的 memory selection 或 policy，而不是单纯更强生成器。

## 局限与展望
- 评测只覆盖单轮 response generation，没有考察多轮对话中记忆选择策略如何随互动动态变化。
- 当前只处理文本记忆，没有覆盖声音、图像、外貌、位置等多模态角色记忆。
- 数据来自 LoCoMo 转换和合成流程，虽然有人类复核，但真实用户长期互动中的隐私、情绪和社交边界更复杂。
- 自动评估依赖 LLM judge，虽然与人工一致性较高，但仍可能对某些表达风格或模型家族存在偏好。

## 相关工作与启发
- **vs LoCoMo / LongMemEval**: 这些 benchmark 主要评估长程记忆检索和 factual recall；StratMem-Bench 进一步要求模型判断记忆的对话功能角色。
- **vs Personalized RAG**: 个性化 RAG 关注引入用户偏好提升回答相关性，本文强调在生成时动态选择 must、nice、irr，并评估过度个性化的风险。
- **vs 角色一致性评测**: 传统角色扮演评测看 persona consistency，本文看角色是否像人一样“知道什么时候提过去的事”。
- **启发**: 面向真实助手系统，可以在 memory manager 中显式建模 must/nice/irr 或类似标签，并将 CIR 作为安全和用户体验指标。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 从 factual recall 转向 strategic memory use，问题定义很清楚，指标组合也有现实感。
- 实验充分度: ⭐⭐⭐⭐☆ 覆盖多家强模型并有人类一致性验证，但缺少多轮和真实用户场景评测。
- 写作质量: ⭐⭐⭐⭐☆ 任务、指标和结论组织清晰，部分自动评估细节放在附录，主文读起来比较顺。
- 价值: ⭐⭐⭐⭐⭐ 对虚拟角色、个人助理和长期记忆 RAG 都很有参考价值，尤其能帮助诊断“太冷淡”和“太多嘴”两类失败。# StratMem-Bench: Evaluating Strategic Memory Use in Virtual Character Conversation Beyond Factual Recall

**会议**: ACL2026  
**arXiv**: [2604.26243](https://arxiv.org/abs/2604.26243)  
**代码**: https://github.com/seucoin/StratMem-Bench.git  
**领域**: LLM评测 / 虚拟角色对话  
**关键词**: 战略记忆使用, 虚拟角色, 长期记忆, LLM评测, 对话生成  

## 一句话总结
StratMem-Bench 把角色对话中的记忆分成必需、可增益和无关三类，并用 SMC、MIQ、PES、CIR 四个指标评测模型是否能在事实正确、主动丰富和避免跑题之间做战略选择。

## 研究背景与动机
**领域现状**：长期记忆是虚拟角色和个性化对话系统的重要能力。现有 memory-augmented generation、long-term dialogue 和 personalized RAG 评测通常关注“模型能不能记住并召回事实”，例如用户之前说过什么、事件发生在哪里、某个偏好是什么。

**现有痛点**：人类对话中的记忆使用并不是把所有相关片段都倒出来。说话者需要判断哪些记忆是回答当前问题的硬需求，哪些记忆只是能让回答更自然、更有共情，哪些记忆虽然真实但此刻提出来会打断话题。传统 factual recall 指标只能奖励“用了该用的事实”，却很难惩罚“用了不该用的事实”或衡量“可选记忆用得是否自然”。

**核心矛盾**：角色对话需要的是动态、情境化的记忆控制，而不是静态知识库检索。模型如果过于保守，只用必需记忆，回答会正确但单薄；如果过于主动，可能把无关私密信息或背景故事强行插入，导致角色对话失焦。

**本文目标**：作者希望让 strategic memory use 可测量。一个好虚拟角色应该做到三件事：必须使用所有 must memories，适当使用一些 nice memories 来丰富对话，同时抑制 irr memories。

**切入角度**：论文没有提出新的对话模型，而是提出一个 benchmark 和评测框架。它从 LoCoMo 这类长会话数据中构造角色 persona、历史对话、当前 query 和未标注 memory pool，然后在评测端用隐藏标注判断模型选记忆和整合记忆的质量。

**核心 idea**：把“记忆相关”细化成 must / nice / irr 三种功能角色，并把评测拆成硬约束合规、整合质量、主动丰富倾向和无关插入风险四个维度。

## 方法详解
这篇论文的“方法”主要是数据集构建和评测指标设计。它解决的是：如何把角色对话里的记忆使用从模糊的人类直觉，变成可以自动评估、又尽量贴近人类判断的任务。

### 整体框架
每个样本包含四类输入：历史对话 $h$、当前用户 query $q$、persona $P$ 和 memory pool $M$。模型只看到未标注的自然语言 memory pool，不知道每条记忆属于 must、nice 还是 irr。模型需要生成角色回复 $\hat{y}=f(h,q,M,P)$。评测时，隐藏标注会把 $M$ 分成三个不相交集合：$M_{must}$、$M_{nice}$、$M_{irr}$。系统先检测每条记忆是否被回复使用，再根据不同场景计算 Strict Memory Compliance，并进一步用 LLM judge 评估 Memory Integration Quality。

### 关键设计
1. **must / nice / irr 的功能性记忆标注**:

	- 功能：把记忆从“是否相关”拆成三种对当前对话目标的功能角色。
	- 核心思路：must memories 是满足当前 query 必不可少的信息，漏用会导致回答错误或幻觉；nice memories 不是事实正确性的硬需求，但能增强个性化、共情或社交连贯性；irr memories 与当前目标无关，应该被抑制。关键点是这些标签是 instance-level 的，同一条记忆在不同 query 下可能变成不同角色。例如“John 最近搬家”在问住址时是 must，在问近况时可能是 nice，在问喜欢的音乐时就是 irr。
	- 设计动机：传统 RAG 只问“检索结果是否相关”，但角色对话中的语用目标更细。Grice 的 Relation 和 Quantity 原则说明，一个好回答既要相关，也要信息量适当。

2. **从 LoCoMo 构造 StratMem-Bench**:

	- 功能：生成带 persona、history、query 和异质记忆池的角色对话评测样本。
	- 核心思路：作者从 LoCoMo 的多 session 对话和结构化 summary 出发，使用窗口 $w\in\{1,2,3\}$ 提取过去 session 形成 memory pool，并把紧随其后的 session 作为 dialogue history，保证记忆获得和当前对话有时间分离。query 和 persona 由 LLM 生成，memory 初始标签由 GPT-5.1 自动标注，再由多名人工专家审核、合并重复片段、删除歧义样本。最终得到 657 个样本，并报告标注前 Fleiss' κ=0.81。
	- 设计动机：如果直接从真实长对话中抽 query，很难控制 must / nice / irr 三类记忆的混合。用 LoCoMo 的长期会话和 summary 做源数据，可以保留角色对话的自然性，又能构造可控的记忆池。

3. **四维评测指标：SMC、MIQ、PES、CIR**:

	- 功能：同时测量硬约束、整合质量、主动性和风险偏好。
	- 核心思路：SMC 是严格二值合规指标。must-only 场景要求所有 must 都被用且 irr 不被用；nice-only 要求至少一个 nice 被用且 irr 不被用；must+nice 要求所有 must 被用、至少一个 nice 被用且 irr 不被用。MIQ 是 1-5 分的 memory integration quality，重点不是流畅度，而是记忆是否自然、相关、无误归因。PES 衡量在 must 已满足时模型是否主动使用 nice，CIR 衡量存在 nice 时模型是否误用 irr。
	- 设计动机：单一准确率无法区分“保守但正确”和“主动但冒险”。PES / CIR 把 proactivity-risk trade-off 显式化，让模型行为画像更清楚。

### 损失函数 / 训练策略
本文不训练新模型。评测流程中，模型生成回复时使用统一 prompt 和温度 $T=0.6$。记忆使用检测与 MIQ 评分由 DeepSeek-V3.2 执行，memory-use detection 使用三次重复采样和多数投票，并要求 judge 引用回复中的具体证据再做 used/not used 判断。自动评估经过人工校验：记忆使用检测在 1,130 个 memory-response pair 上与专家标注 Cohen's κ=0.96，MIQ 在 300 个回复上与人工评分 Cohen's κ=0.69。

## 实验关键数据

### 主实验
数据集共有 657 个样本，must+nice 是最大也最难的场景，占 475 个。整体来看，所有模型在 must-only 中表现不错，但进入 nice-only 和 must+nice 后显著下降。

| 场景 | 样本数 | 平均记忆数 | 平均每条记忆词数 | 评测含义 |
|------|--------|------------|------------------|----------|
| must-only | 50 | 6.24 | 9.53 | 用完必需记忆并避开无关记忆 |
| nice-only | 132 | 9.12 | 10.09 | 主动选用支持性记忆并避开无关记忆 |
| must+nice | 475 | 8.97 | 9.75 | 同时满足事实需求、主动丰富和抑制无关记忆 |
| Overall | 657 | 8.79 | 9.81 | 完整战略记忆使用评测 |

### 消融实验
这里没有模型消融，主要实验是多模型评测。表中 SMC 是严格合规率，MIQ 是只在 SMC-pass 样本上计算的记忆整合质量。

| 模型 | must-only SMC | nice-only SMC | must+nice SMC | All SMC | All MIQ on pass |
|------|---------------|---------------|---------------|---------|-----------------|
| GPT-5.2 | 88.00 | 57.58 | 41.89 | 48.55 | 4.45 |
| GPT-5-chat | 90.00 | 46.21 | 41.68 | 46.27 | 4.56 |
| Claude Sonnet 4.5 | 90.00 | 53.03 | 46.95 | 51.45 | 4.37 |
| Gemini 3 Pro | 78.00 | 49.24 | 48.21 | 50.68 | 4.21 |
| DeepSeek-reasoner | 76.00 | 48.48 | 39.16 | 43.84 | 4.12 |
| Llama 4 Maverick | 79.25 | 53.44 | 46.09 | 50.23 | 4.44 |
| Qwen3-235B | 92.45 | 46.56 | 42.28 | 47.18 | 4.24 |

### 记忆类型与行为倾向

| 模型 | must-used MIQ | nice-used MIQ | irr-used MIQ | PES All | CIR All |
|------|---------------|---------------|--------------|---------|---------|
| GPT-5.2 | 4.48 | 4.22 | 2.99 | 56.01 | 13.01 |
| GPT-5-chat | 4.55 | 4.38 | 2.81 | 51.91 | 7.91 |
| Claude Sonnet 4.5 | 4.36 | 4.18 | 3.05 | 62.48 | 15.82 |
| Gemini 3 Pro | 3.92 | 3.73 | 2.63 | 73.33 | 31.96 |
| DeepSeek-chat | 4.32 | 4.12 | 2.75 | 56.96 | 15.49 |
| Llama 4 Maverick | 4.44 | 4.23 | 2.66 | 57.27 | 12.52 |
| Qwen3-Max | 4.14 | 4.04 | 2.64 | 57.76 | 19.77 |

### 关键发现
- must-only 场景 SMC 约为 76%-92%，说明强模型能较好地区分必需事实和无关记忆。
- nice-only 场景 SMC 降到 46%-57%，must+nice 全部模型低于 50%，说明“支持性但非必需”的记忆是当前 LLM 的主要盲区。
- SMC-pass 的 MIQ 通常较高，说明一旦模型选对记忆，整合质量往往还不错。瓶颈主要在 selection，而不是语言表达。
- nice-used MIQ 通常比 must-used 低约 0.2，作者称为 enrichment tax；而 irr-used MIQ 跌到 2.6-3.1，说明无关记忆不是轻微噪声，而是会明显破坏对话连贯性。
- PES 和 CIR 存在明显 trade-off。Gemini 3 Pro 最主动，PES 73.33%，但 CIR 也高达 31.96%；GPT-5-chat 最保守，CIR 7.91%，但 PES 只有 51.91%。

## 亮点与洞察
- must / nice / irr 的划分非常贴近真实角色对话。很多系统只会做 factual recall，但“提不提某段真实记忆”本身就是社交策略。
- SMC 的严格定义很有诊断价值。must+nice 场景要求 all must、at least one nice、zero irr，虽然苛刻，但能精准暴露模型在异质记忆池中的决策不稳定。
- MIQ 不追求细腻审美评分，而是 failure-sensitive，这降低了 LLM judge 的主观性。作者明确列出 topic drift、forced over-association、fabrication、overexpansion、misattribution 等错误类型，便于复现和分析。
- PES / CIR 的二维图比单一排行榜更有用。不同产品可能宁愿保守不冒险，也可能希望角色更主动，Benchmark 能帮助开发者按场景选模型。

## 局限与展望
- StratMem-Bench 只评测单轮响应生成，没有评测记忆使用策略在多轮互动中的长期演化。真实角色可能需要先少说、后追问、再逐步揭示记忆。
- 数据只包含文本记忆，没有涉及声音、图像、外貌、地点等多模态记忆。具身角色和游戏 NPC 场景中，多模态记忆会更重要。
- 样本来自 LoCoMo 转换和 LLM 生成，虽然有人工审核，但仍可能和真实用户长期互动存在分布差异。
- 自动评测依赖 DeepSeek-V3.2 judge。尽管人工一致性较高，未来仍需要更多开放 evaluator 或人工基准来降低模型自评风险。

## 相关工作与启发
- **vs LoCoMo / LongMemEval**: 这些 benchmark 主要评估跨 session 事实召回与推理；StratMem-Bench 评估的是在已有记忆池中选择什么、压制什么，以及怎么自然整合。
- **vs Personalized RAG**: 个性化 RAG 重视用户偏好是否被反映，StratMem-Bench 更关注生成时的战略使用，尤其是“可选但有帮助”和“真实但此刻不该说”的边界。
- **vs 角色一致性评测**: 传统 role-play benchmark 多测 persona consistency，本文把角色能力扩展到 turn-level memory control，对长期陪伴型 agent 更关键。
- **对系统设计的启发**: 实际应用中可以把记忆检索器输出也分成 hard-required、soft-enrichment、suppress candidates，让生成模型显式知道哪些记忆必须用、哪些要谨慎用。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 把记忆从 factual recall 推进到 strategic use，问题定义很清楚。
- 实验充分度: ⭐⭐⭐⭐☆ 数据构建、模型覆盖、自动评估人类验证和行为分析都扎实，但缺少多轮设置。
- 写作质量: ⭐⭐⭐⭐☆ 指标解释和 case analysis 友好，读完能明确知道每个分数代表什么。
- 价值: ⭐⭐⭐⭐⭐ 对虚拟角色、长期记忆 Agent 和个性化对话系统都有直接评测价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[ACL 2026\] Evaluating Memory Capability in Continuous Lifelog Scenario](evaluating_memory_capability_in_continuous_lifelog_scenario.md)
- [\[ACL 2026\] Stress Testing Factual Consistency Metrics for Long-Document Summarization](stress_testing_factual_consistency_metrics_for_long-document_summarization.md)
- [\[AAAI 2026\] Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](../../AAAI2026/llm_evaluation/beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)
- [\[ICLR 2026\] DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](../../ICLR2026/llm_evaluation/dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)

</div>

<!-- RELATED:END -->
