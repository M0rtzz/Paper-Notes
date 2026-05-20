---
title: >-
  [论文解读] Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding
description: >-
  [ACL 2026][视频理解][主动响应] Response-G1 在免微调设置下把流式视频中的视觉证据和查询触发条件都显式表示为 scene graph，并通过历史图检索增强每帧的 silence/response 决策…
tags:
  - "ACL 2026"
  - "视频理解"
  - "主动响应"
  - "流式视频理解"
  - "场景图"
  - "证据检索"
  - "免微调"
---

# Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding

**会议**: ACL2026  
**arXiv**: [2605.07575](https://arxiv.org/abs/2605.07575)  
**代码**: https://github.com/kadmkbl/Response-G1  
**领域**: video_understanding  
**关键词**: 流式视频理解、主动响应、场景图、检索增强、Video-LLM

## 一句话总结
Response-G1 用查询引导的在线场景图、历史场景图检索和带时间戳的触发提示，把流式视频中的视觉证据和用户查询的响应条件显式对齐，在无需微调的情况下显著提升 Video-LLM 判断“现在是否该回答”的能力。

## 研究背景与动机
**领域现状**：Video-LLM 已经能处理视频问答和长视频理解，Streaming Video Understanding 则进一步要求模型在视频持续到来时增量感知、推理和交互。多数现有系统仍是 reactive：用户在某个时间点提问，模型立刻基于已观察片段回答。

**现有痛点**：现实交互里很多问题是 anticipatory 的，例如“当某个人开始做某件事时告诉我”。这类问题在提问时答案条件可能还没出现，模型必须先保持沉默，等证据满足条件再回答。已有 proactive 方法要么训练 EOS/二分类触发头，依赖细粒度帧级标注；要么用帧差阈值或多 agent prompt，容易忽略查询语义。

**核心矛盾**：主动响应的关键不是“视频有没有变化”，而是“当前累积证据是否满足这个查询隐含的响应条件”。如果视觉证据和查询条件都只隐式存在于 Video-LLM hidden states 或 prompt 里，模型很难稳定对齐它们，也难解释为何此刻触发。

**本文目标**：作者希望设计一个不需要微调和帧级标签的框架，让 Video-LLM 在流式视频中显式建模查询相关证据、检索历史证据，并据此决定 silence/response。

**切入角度**：用户查询通常描述一个由对象、属性和关系组成的目标场景，例如“穿红衣服的男孩正在和别人交谈”。这天然可以表示成 scene graph；如果视频片段也转成 scene graph，就能在同一结构空间里做证据-条件匹配。

**核心 idea**：把流式视频的观察证据和查询条件都转成场景图，通过 top-K 场景图检索把最相关历史证据送回 Video-LLM，再用触发 prompt 做逐帧响应时机判断。

## 方法详解
Response-G1 是一个 fine-tuning-free pipeline。它没有训练新的触发分类器，而是把 Video-LLM 自身用作场景图生成器、文本编码器和最终决策器。方法的关键是把长视频历史压缩成结构化、查询相关、可检索的图记忆，而不是把所有视觉 token 都塞进上下文。

### 整体框架
输入是流式视频帧序列和用户在某个时刻提出的查询，输出是每个时间步的 `silence` 或 `response` 决策，以及触发后的自然语言答案。框架包含三步：先对当前 clip 做 query-guided scene graph generation，抽取对象-谓词-对象三元组；再把历史场景图存入 memory bank，并与查询条件图计算相似度，检索 top-K 相关片段；最后把视觉 token、带时间戳的检索场景图和触发指令一起输入 Video-LLM，让它判断是否回答。

### 关键设计
1. **查询引导的在线场景图生成**:

    - 功能：把每个 streaming clip 转成只保留查询相关细节的结构化证据。
    - 核心思路：对时间 $t$ 附近的视频片段 $C_t$，用 Video-LLM 根据原始查询 $Q$ 生成场景图 $G_t=(O_t,P_t)$，其中节点是对象及属性，边是时空关系。场景图可写成三元组集合 $G_t=\{(o_i,p_{ij},o_j)\}$。查询被直接注入生成 prompt，使模型优先描述与触发条件有关的对象和关系。
    - 设计动机：无查询的场景图会生成大量无关 triplet，增加检索噪声；直接塞入目标对象又可能诱发 hallucination。保留原始查询作为软指导，在相关性和真实性之间更平衡。

2. **记忆式场景图检索**:

    - 功能：从历史视频片段中找出最能支持当前响应条件的证据。
    - 核心思路：每个场景图三元组被线性化成自然语言短语，整图表示为短语拼接 $\Phi_t$。查询也被解析成查询条件图 $G_q$ 并线性化为 $\Phi_q$，从而和视频图保持相同格式。方法用 Video-LLM 文本编码器做 mean pooling 得到图向量 $g_t$ 和查询向量 $g_q$，再用余弦相似度检索 memory bank 中 top-K 场景图。
    - 设计动机：直接用原始查询文本和视频图做相似度会有格式不一致问题；把两者都变成 graph text，检索更关注对象关系是否匹配。

3. **带时间戳的检索增强触发决策**:

    - 功能：让模型在每个时间步判断是否已有足够证据回答。
    - 核心思路：检索出的场景图会加上文本时间戳，例如 `<2.0s>`，再编码进上下文。触发阶段的输入是视觉帧 embedding、检索场景图上下文和类似“Should I answer now? Yes or No.”的指令。若输出 silence，就继续处理下一帧；若输出 response，就用同一组检索增强上下文和原始查询生成最终答案。
    - 设计动机：主动响应不仅要知道“是否出现目标关系”，还要知道证据在时间上的先后和当前是否已足够。时间戳让图记忆从静态检索结果变成可用于 temporal grounding 的证据链。

### 损失函数 / 训练策略
Response-G1 不做参数训练，核心是 inference-time pipeline。实验使用 Qwen3-VL-8B 作为 Video-LLM backbone，OVO-Bench 采用默认 1 FPS；StreamingBench 按官方规则采样，短视频 1 FPS，中等长度 0.5 FPS，长视频 0.2 FPS。所有实验在 A100 80GB 上以 FP16 运行。作者还做了延迟分析：原始 Response-G1 embedding 版本每帧约 825ms，对应 1.2 FPS；使用 streaming KV-Cache 后约 473ms，对应 2.1 FPS，满足 1 FPS 设置。

## 实验关键数据

### 主实验
在 OVO-Bench 上，Response-G1 对 open-source streaming Video-LLM 的优势集中体现在 Forward Active Responding，也就是最能体现 proactive 能力的部分。它的整体分数未超过 Gemini 1.5 Pro 等闭源模型，但在开源流式模型中明显领先。

| 模型 | 参数 | Real-Time Visual Perception Avg↑ | Backward Tracing Avg↑ | Forward Active Responding Avg↑ | Overall Avg↑ |
|------|------|----------------------------------|-----------------------|--------------------------------|--------------|
| GPT-4o | - | 63.6 | 58.7 | 53.4 | 58.6 |
| Gemini 1.5 Pro | - | 70.8 | 62.3 | 57.2 | 65.3 |
| TimeChat-Online | 7B | 58.6 | 42.0 | 36.4 | 45.6 |
| StreamAgent | 7B | 61.3 | 41.7 | 45.4 | 49.4 |
| Response-G1 | 8B | 73.6 | 52.1 | 58.2 | 61.3 |

在 StreamingBench 上，Response-G1 在开源模型中取得最高 Overall，并把 proactive output 从约 29 提升到 44。闭源 GPT-4o 的 PO 仍更高，但 Response-G1 显著缩小了开源模型差距。

| 模型 | 参数 | Real-Time Visual Understanding Avg↑ | PO↑ | Overall Avg↑ | 说明 |
|------|------|-------------------------------------|-----|--------------|------|
| GPT-4o | - | 73.3 | 56.9 | 70.5 | 闭源强基线 |
| LLaVA-OneVision | 7B | 71.1 | 29.6 | 66.3 | 强开源 Video-LLM |
| TimeChat-Online | 7B | 75.4 | 28.8 | 70.9 | 开源流式基线 |
| StreamAgent | 7B | 74.3 | 28.9 | 70.2 | 多 agent prompt 基线 |
| Response-G1 | 8B | 77.5 | 44.0 | 73.7 | 开源流式模型中最高 overall 和 PO |

### 消融实验
检索增强和时间戳都有效。去掉 retrieval augmentation 后，主动任务和反应任务同时下降；去掉 timestamp encoding 后，CRR/PO 这类需要时间定位的任务受影响更明显。

| 配置 | OVO ACR↑ | OVO HLD↑ | OVO CRR↑ | Streaming CS↑ | Streaming PR↑ | Streaming PO↑ |
|------|----------|----------|----------|---------------|---------------|---------------|
| w/o Retrieval Augmentation | 66.1 | 28.0 | 55.4 | 83.6 | 79.6 | 36.8 |
| w/o Timestamp Encoding | 74.0 | 33.6 | 60.4 | 87.7 | 82.9 | 43.6 |
| Full | 74.3 | 33.9 | 61.7 | 88.0 | 83.3 | 44.0 |

查询引导也是关键。直接把解析出的对象关系塞进 prompt 会提高相关性，但容易让模型提前“看见”尚未出现的目标；原始查询引导最稳。

| 场景图生成策略 | Streaming PO↑ | OVO REC↑ | OVO SSR↑ | OVO CRR↑ | 解释 |
|----------------|---------------|----------|----------|----------|------|
| w/o Guidance | 38.8 | 34.1 | 66.9 | 59.4 | 生成很多无关 triplet |
| Object-Guidance | 43.6 | 40.2 | 67.9 | 61.3 | 更相关，但有 hallucination 风险 |
| Query-Guidance | 44.0 | 41.9 | 71.1 | 61.7 | 相关性和事实性最平衡 |

### 关键发现
- 显式结构化证据对 proactive timing 特别有用。Response-G1 在 OVO 的 FAR 和 StreamingBench 的 PO 上提升最明显，说明场景图确实帮助模型判断“条件是否满足”。
- 检索不是只为压缩上下文，它还把长视频历史按查询语义重新排序，使触发决策看到的是最相关证据而不是最近证据。
- 图文本格式一致性很重要。论文比较原始查询文本和 query graph text，后者在相似度检索上更好，说明跨模态检索前的表示格式对齐不能忽略。
- KV-Cache 让方法从概念验证接近可实时部署。1.2 FPS 到 2.1 FPS 的延迟结果说明额外 SGG/SGR 成本可控，但仍适合低 FPS 流式理解而非高帧率控制。

## 亮点与洞察
- 论文的强点是把 proactive response timing 具体化为 evidence-condition alignment，而不是笼统地让模型“判断是否该回答”。这个问题重述本身很有价值。
- 场景图在这里不是传统视觉解析模块，而是 Video-LLM 自己生成的开放词表结构记忆。它牺牲了一点严谨性，但换来了无需检测器和无需微调的通用性。
- Query-guided SGG 的消融很有启发：过少指导会有噪声，过强 object guidance 会幻觉，原始查询作为软条件反而最合适。
- 方法也可迁移到机器人/Agent 记忆：把在线感知片段转成结构化事件图，再按任务意图检索历史证据，用于触发动作或回答。

## 局限与展望
- 作者指出，场景图的对象-关系表示不能覆盖所有推理需求，尤其是 why-style 问题、因果解释和隐含动机推理。
- 当前 clip size 固定，可能错过事件边界或把一个语义事件切碎。未来可以用事件级触发或语义变化检测决定何时生成场景图。
- LLM-based open-set SGG 仍有 hallucination 风险。Object-Guidance 的失败案例说明，如果提示过度暗示目标对象，模型可能提前生成不存在的 triplet。
- 方法依赖 Video-LLM 的文本编码和 prompt 遵循能力，换 backbone 后虽有附录验证，但仍可能需要重调 prompt、K 值和采样率。
- 目前主要在 benchmark 的 1 FPS 或更低采样下验证；如果用于自动驾驶或高频机器人控制，延迟和安全性还远远不够。

## 相关工作与启发
- **vs VideoLLM-online / Flash-Vstream**: 这些方法侧重流式视频 token 处理和效率；Response-G1 更关注 query-aware proactive timing，用结构化图记忆补充视觉 token。
- **vs Dispider / StreamBridge**: 这类方法通常训练激活模型或辅助分类头；Response-G1 不依赖帧级触发标注，而是用 prompt 和检索实现触发决策。
- **vs StreamAgent**: StreamAgent 用多 agent prompting 规划响应时机；Response-G1 用场景图把证据和条件显式对齐，解释性更强、PO 提升更明显。
- **vs 传统 scene graph generation**: 传统 SGG 依赖闭集检测器；本文把 Video-LLM 作为开放词表图生成器，更适合真实长尾视频，但也更需要抑制幻觉。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把场景图检索引入 proactive streaming video understanding，问题切入和结构化记忆设计都很清楚。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 OVO-Bench、StreamingBench、消融、案例、延迟和跨架构验证，但高帧率真实部署仍需更多实验。
- 写作质量: ⭐⭐⭐⭐ 方法图和分阶段叙述易懂，表格较完整；部分表格列很多，读数需要仔细对照。
- 价值: ⭐⭐⭐⭐ 对视频助手、具身智能和流式监控场景有实际启发，尤其适合无需微调的开源 Video-LLM 增强。# Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding

**会议**: ACL 2026  
**arXiv**: [2605.07575](https://arxiv.org/abs/2605.07575)  
**代码**: https://github.com/kadmkbl/Response-G1  
**领域**: 视频理解 / 流式多模态交互 / Scene Graph  
**关键词**: 主动响应、流式视频理解、场景图、证据检索、免微调

## 一句话总结
Response-G1 在免微调设置下把流式视频中的视觉证据和查询触发条件都显式表示为 scene graph，并通过历史图检索增强每帧的 silence/response 决策，使 Video-LLM 在 OVO-Bench 和 StreamingBench 的主动响应与反应式理解任务上显著优于开源流式 Video-LLM。

## 研究背景与动机
**领域现状**：流式视频理解要求模型在视频尚未完全观看时持续处理新帧。多数 Video-LLM 以 reactive interaction 为主：用户在某个时刻提问，模型基于当前已见内容立刻回答。更难的 proactive interaction 则要求模型判断“现在是否已经看到足够证据，可以回答”。

**现有痛点**：主动响应方法通常依赖帧级 EOS 训练、额外二分类头或提示式 planner。微调方法需要细粒度帧级 silence/response 标注，但相邻帧视觉差异很小却可能标签相反，决策边界难学；免微调方法若只看帧差或简单 prompt，又常忽略查询语义。

**核心矛盾**：主动响应的关键不是“视频有没有变化”，而是“已观察证据是否满足用户查询中隐含的响应条件”。现有方法大多隐式地把视频 token 和查询 token 丢给模型，缺少一个结构化中间层来对齐对象、关系和时间证据。

**本文目标**：作者希望用 scene graph 显式表达视频中出现的对象和关系，同时把查询也解析成条件图，让模型在同一结构空间中比较“已见证据”和“应该等待的条件”。

**切入角度**：很多主动响应查询天然描述一个预期场景，例如“当穿红衣服的男孩和别人说话时回答”。这样的条件本身就是对象-关系结构，因此 scene graph 比原始文本或视觉 token 更适合作为证据检索和触发判断的桥梁。

**核心 idea**：把流式视频前缀压缩成查询相关 scene graph memory，再检索最相关的历史图作为结构化证据，辅助 Video-LLM 决定何时响应。

## 方法详解
Response-G1 是一个不训练基座 Video-LLM 的 pipeline。它持续从视频片段中生成 query-guided scene graph，把每个图存在 memory bank 中；每来一帧，就把用户查询解析为 query graph，和历史 scene graph 做相似度检索；最后把当前帧、检索到的带时间戳图证据和 trigger prompt 一起输入 Video-LLM，让模型输出 Yes/No 式触发决策。触发后，再用相同的图增强上下文生成最终答案。

### 整体框架
输入是一个正在到来的帧序列和某个时刻发出的用户查询。系统对以当前时间为中心的视频 clip 生成 scene graph $G_t=(O_t,P_t)$，其中节点是对象及属性，边是时空关系。memory bank 保存从过去到当前的图。推理时，系统先检索 top-K 个与查询条件最相关的历史图，再把它们编码到 trigger 阶段和 answer 阶段。

### 关键设计
1. **在线查询引导的场景图生成**:

	- 功能：把流式视频片段转成与当前查询相关的对象-关系三元组。
	- 核心思路：对视频 clip $C_t$，用 Video-LLM 生成 $G_t=\mathcal{S}(C_t;Q)$；图由三元组 $(o_t^i,p_t^{ij},o_t^j)$ 组成。prompt 中直接放入用户查询，让生成器优先描述查询相关证据，而不是枚举所有画面细节。
	- 设计动机：流式视频中无关细节很多。如果 scene graph 不受查询约束，后续检索会被噪声干扰；如果只注入对象列表，又可能诱发模型提前幻想目标对象。原始查询引导在相关性和事实性之间更平衡。

2. **基于记忆的图检索**:

	- 功能：从历史视频证据中找出最能支持当前响应条件的片段。
	- 核心思路：先把每个 scene graph 三元组线性化成自然语言短语，例如 “woman in red”，再把整个图拼接成文本；同时把用户查询解析成 query condition graph，而不是直接用原始查询文本。二者用 Video-LLM 文本编码器均值池化得到向量，再用 cosine similarity 取 top-K。
	- 设计动机：视频图和查询条件如果格式不一致，相似度会偏向语言表面而不是图结构。把查询也转成图文本，可以让检索比较“对象-关系证据”而非普通句子。

3. **检索增强的触发与回答**:

	- 功能：让模型在每个时间步判断 silence 或 response，并在触发后生成答案。
	- 核心思路：触发阶段输入为当前视频帧 token、带时间戳的检索图 $\Psi(G_t^{ctx})$ 和 trigger instruction；若模型输出 silence，则继续等待下一帧；若输出 response，则用视频前缀、同一批图证据和原始查询生成答案。
	- 设计动机：主动响应既需要语义匹配，也需要时间定位。给 scene graph 加时间戳能帮助模型知道证据何时出现，避免只看到相关对象却误判响应时机。

### 损失函数 / 训练策略
Response-G1 本身没有训练损失，是免微调 pipeline。实验用 Qwen3-VL-8B 作为 Video-LLM backbone，OVO-Bench 按默认 1 FPS 评估；StreamingBench 按官方采样策略，短于 300 帧的视频 1 FPS、300 到 600 帧 0.5 FPS、长于 600 帧 0.2 FPS。实验在 A100 80GB 上用 FP16 运行。

## 实验关键数据

### 主实验
实验分 proactive mode 和 reactive mode。OVO-Bench 的 Forward Active Responding 以及 StreamingBench 的 PO 是主动响应任务；其他子任务用于验证 scene graph retrieval 对普通流式视频理解是否也有帮助。

| 方法 | OVO RTVP | OVO BT | OVO FAR | OVO Overall | Streaming RTVU | Streaming PO | Streaming Overall |
|------|----------|--------|---------|-------------|----------------|--------------|-------------------|
| TimeChat-Online | 58.6 | 42.0 | 36.4 | 45.6 | 75.4 | 28.8 | 70.9 |
| StreamAgent | 61.3 | 41.7 | 45.4 | 49.4 | 74.3 | 28.9 | 70.2 |
| Response-G1 | 73.6 | 52.1 | 58.2 | 61.3 | 77.5 | 44.0 | 73.7 |

Response-G1 在 OVO-Bench 的主动响应 FAR 上比第二名开源流式 Video-LLM 高 12.8 个点，在 StreamingBench 的 PO 上高 15.1 个点。它在 reactive 任务也有提升，说明图检索不是只对触发判断有用，也增强了对象关系和时空理解。

### 消融实验
作者主要验证 retrieval augmentation、timestamp encoding、query-guided SGG 和 query graph retrieval 的作用。

| 配置 | ACR | HLD | CRR | CS | PR | PO | 说明 |
|------|-----|-----|-----|----|----|----|------|
| w/o Retrieval Augmentation | 66.1 | 28.0 | 55.4 | 83.6 | 79.6 | 36.8 | 不加入图证据，主动和反应式任务都弱 |
| w/o Timestamp Encoding | 74.0 | 33.6 | 60.4 | 87.7 | 82.9 | 43.6 | 图证据有用，但缺少时间线索 |
| Full | 74.3 | 33.9 | 61.7 | 88.0 | 83.3 | 44.0 | 完整方法最佳 |

论文还报告 query-guided scene graph generation 优于无引导和直接对象引导。直接注入解析对象会让模型过度关注预期对象，甚至生成尚未出现的三元组，导致过早响应；用原始查询引导则更稳。

### 关键发现
- 显式 scene graph retrieval 对主动响应提升最大，因为主动响应本质上就是“证据是否满足条件”的判断。
- Timestamp encoding 对 CRR 这类需要定位线索出现时刻的任务特别重要，说明结构化语义和时间顺序缺一不可。
- 查询解析成 graph text 后再检索，比直接用原始 query text 更适合与视频 scene graph 做相似度比较。
- 延迟分析显示 Response-G1 embedding 版本每帧约 825 ms，最大 1.2 FPS；加入 streaming KV-Cache 后约 473 ms，最大 2.1 FPS，基本满足 1 FPS 流式设置。

## 亮点与洞察
- 论文把主动响应从一个黑盒分类问题变成可解释的结构化证据匹配问题。检索到的 scene graph 可以直接展示给人看，帮助解释模型为什么此刻回答。
- 免微调是很强的工程优势。很多流式视频标注昂贵且任务差异大，Response-G1 通过 prompt、memory 和 retrieval 就能提升现有 Video-LLM。
- Query-guided SGG 的细节很重要：只给对象关系会让模型“先入为主”，给完整查询反而能保留条件语义和事实约束。
- 这个框架可以扩展到机器人、AR 眼镜和视频客服场景，因为这些系统都需要边看边判断何时打断、提醒或回答。

## 局限与展望
- 作者指出 scene graph 的对象-关系表示不能覆盖所有推理需求，尤其是“为什么”类问题，未来可能需要因果图或事件图。
- 当前在线 SGG 使用固定 clip size，不会根据事件边界自适应决定何时生成图，也可能带来冗余计算。
- LLM-based open-set scene graph generation 存在相关性和幻觉之间的权衡；过强查询引导可能提前生成不存在的三元组。
- Top-K 相似度检索较简单，尚未探索更复杂的长期记忆、证据去重、冲突处理和跨片段事件合并。
- 虽然延迟能满足 1 FPS，但更高帧率、边缘设备和多路视频流仍会受到 SGG 计算成本限制。

## 相关工作与启发
- **vs VideoLLM-online / Flash-Vstream**: 这些方法重点是在线处理和减少视觉 token 冗余，Response-G1 更关注查询条件与累积证据的结构化对齐。
- **vs TimeChat-Online**: TimeChat-Online 通过提示或阈值做触发，Response-G1 加入 scene graph memory，使触发判断更 query-aware 和可解释。
- **vs StreamAgent**: StreamAgent 用多智能体/planner 规划响应时机，Response-G1 用统一的图表示和检索完成证据组织，结构更轻。
- **启发**: 流式多模态系统可以把中间记忆从“所有视觉 token”转为“可检索结构图”。这能同时提高可解释性、压缩历史和增强触发决策。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 将 scene graph 用于 proactive streaming response timing 很自然但抓住了关键痛点。
- 实验充分度: ⭐⭐⭐⭐☆ 两个 benchmark、主动与反应式任务、消融和延迟都有覆盖，但缺少真实交互用户研究。
- 写作质量: ⭐⭐⭐⭐☆ 方法图和实验表清晰，部分表格列较多但结论明确。
- 价值: ⭐⭐⭐⭐⭐ 对流式视频助手、具身智能和可穿戴 AI 的响应时机控制很有实际意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](../../CVPR2026/video_understanding/streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[ACL 2026\] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)
- [\[CVPR 2025\] HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation](../../CVPR2025/video_understanding/hyperglm_hypergraph_for_video_scene_graph_generation_and_anticipation.md)
- [\[AAAI 2026\] Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction](../../AAAI2026/video_understanding/explicit_temporal-semantic_modeling_for_dense_video_captioning_via_context-aware.md)
- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](../../CVPR2026/video_understanding/fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)

</div>

<!-- RELATED:END -->
