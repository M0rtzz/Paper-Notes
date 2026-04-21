---
title: >-
  [论文解读] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering
description: >-
  [AAAI 2026][AI安全][隐私保护RAG] 首次探索知识图谱问答（KGQA）中的隐私保护 RAG 场景，提出 ARoG（Abstraction Reasoning on Graph）框架，通过关系中心抽象和结构导向抽象两种策略，在实体被匿名化（替换为无意义的 MID）的条件下，仍能有效检索和利用知识图谱回答问题。
tags:
  - AAAI 2026
  - AI安全
  - 隐私保护RAG
  - 知识图谱问答
  - 实体匿名化
  - 抽象推理
  - LLM隐私
---

# Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering

**会议**: AAAI 2026  
**arXiv**: [2508.08785](https://arxiv.org/abs/2508.08785)  
**代码**: [https://github.com/NLPGM/ARoG](https://github.com/NLPGM/ARoG)  
**领域**: AI安全  
**关键词**: 隐私保护RAG, 知识图谱问答, 实体匿名化, 抽象推理, LLM隐私

## 一句话总结

首次探索知识图谱问答（KGQA）中的隐私保护 RAG 场景，提出 ARoG（Abstraction Reasoning on Graph）框架，通过关系中心抽象和结构导向抽象两种策略，在实体被匿名化（替换为无意义的 MID）的条件下，仍能有效检索和利用知识图谱回答问题。

## 研究背景与动机

RAG（检索增强生成）通过从外部知识源（如知识图谱 KG）检索事实信息来增强 LLM 的输出质量，缓解幻觉和知识过时问题。然而，许多实际 KG 包含**敏感隐私信息**（个人数据、公司机密等）。

**核心隐私风险**: 当使用私有 KG 回答问题时，现有 RAG 系统必须将相关三元组暴露给 LLM。例如回答"Bronny 住在哪里？"时，三元组 (Bronny, lives in, L.A.) 会被发送到 LLM。由于 LLM 的**黑盒特性**和数据传输的潜在不安全性，这构成严重的隐私泄露风险，特别是使用第三方 LLM API 时。

**隐私保护 RAG 场景定义**: KG 中的实体对 LLM 是匿名的——实体被替换为加密的唯一机器标识符（MID），不含任何语义信息。LLM 无法了解实体的类型、名称或描述。

**两大核心挑战**:

**如何将匿名实体转化为可检索信息？** MID 无语义，LLM 无法通过匹配 MID 与问题来检索相关知识

**如何检索与问题相关的匿名实体？** 问题是自然语言，而 KG 中实体是无意义标识符，两者之间没有语义桥梁

**关键洞察**: 虽然实体被匿名化，但 KG 中的**关系（relations）** 定义的是模式结构而非敏感信息，可以安全地与 LLM 共享。通过实体周围的关系信息，可以推断出实体的抽象概念（如一个实体是 time_zones、contained_by、population 的主语，citytown 的宾语 → 概念为 "geographic location"）。

## 方法详解

### 整体框架

ARoG 采用 **Retrieval-then-Generation** 流水线，包含四个模块：

1. **关系中心抽象（RA）**: 将匿名实体通过其邻接关系抽象为高层概念
2. **结构导向抽象（SA）**: 将非结构化问题转化为结构化的抽象概念路径
3. **抽象驱动检索（AR）**: 基于两种抽象进行多轮迭代检索
4. **生成器（Generator）**: 基于检索到的三元组推断答案

### 关键设计

#### 1. **关系中心抽象模块（Relation-centric Abstraction）**

将关系视为"谓语动词"，实体视为"主语/宾语名词"，通过关系推断实体概念。包含三步：

**步骤一：关系检索** — 从 $n$ 个主题实体出发，提取邻接关系，用 LLM 选择 $W$ 个最相关关系：

$$R_{\text{t,opt}} = \arg\max_{S \subseteq R_t, |S| \leq W} \text{LLM}(R_t, q, Inst_{rr}, E_{rr})$$

**步骤二：关系过滤** — 对候选实体集中的每个实体，用 SentenceTransformer 按余弦相似度选择 top-$K$（$K=5$）个与问题最相关的关系：

$$R_{v,opt} = \arg\max_{S \subseteq R_v, |S| \leq K} \sum_{r \in S} \cos(\text{Emb}(q), \text{Emb}(r))$$

**步骤三：实体抽象** — 用 LLM 基于过滤后的关系推断实体的抽象概念，附加到 MID 后：

$$e^{abs} = \text{LLM}(R_{v,opt}, e, Inst_{ea}, E_{ea})$$

关键设计：同一实体集群内的实体共享同一概念（减少 LLM 调用）；多个概念以逗号连接。

#### 2. **结构导向抽象模块（Structure-oriented Abstraction）**

将非结构化问题转化为结构化的**抽象概念路径** $P_q$。核心优势：**路径的有效性不依赖于正确的实体名称**。

例如，问题 "What is the daughter of the artist who had The Mrs. Carter Show World Tour?" 的抽象概念路径：

```
Nicki Minaj (artist) → had → The Mrs. Carter Show World Tour
Nicki Minaj (artist) → has daughter named → Chiara Fattorini (person)
```

虽然 "Nicki Minaj" 和 "Chiara Fattorini" 都是错误的实体，但包含 "artist" 和 "person" 概念的路径与 KG 中抽象后的三元组在语义上仍然对齐。

使用 CoT（链式思维）方式生成：

$$Rat, P_q = \text{LLM}(q, Inst_{sa}, E_{sa})$$

#### 3. **抽象驱动检索模块（Abstraction-driven Retrieval）**

多轮迭代检索，参数为宽度 $W$ 和深度 $D$（均默认为 3）：

每轮迭代中，基于关系中心抽象获得候选三元组 $T_q^{abs}$，然后用余弦相似度将其与抽象概念路径 $P_q$ 中的三元组匹配，选择 $W$ 个最相关的：

$$T_{q,\text{opt}}^{abs} = \arg\max_{S \subseteq T_q^{abs}, |S| \leq W} \sum_{t^{abs} \in S} \max_{t_p \in P_q} \cos(\text{Emb}(t^{abs}), \text{Emb}(t_p))$$

新发现的实体替换主题实体进入下一轮迭代。

#### 4. **生成器模块**

将累积的相关三元组 $T_{q,all}^{abs}$ 和原始问题 $q$ 输入 LLM 生成答案。输出包含一个 Flag：若为正则直接输出答案，否则继续下一轮检索。

**答案中的 MID 在用户侧替换为真实名称**（用户保有 MID → 名称的映射表），这确保了隐私保护的完整闭环。

### 损失函数 / 训练策略

ARoG 是纯推理时框架，不涉及模型训练。关键配置：

- **LLM**: gpt-4o-mini-2024-07-18
- **温度**: SA 和 Generator 设为 0（确定性输出），RA 设为 0.4（引入多样性）
- **检索参数**: 宽度 $W = 3$, 深度 $D = 3$
- **嵌入模型**: SentenceTransformer 用于关系过滤和三元组匹配
- **实验重复 3 次取平均**

## 实验关键数据

### 主实验

| 类型 | 方法 | WebQSP #Tot | WebQSP #Fil | CWQ #Tot | CWQ #Fil | GrailQA #Tot | GrailQA #Fil |
|---|---|---|---|---|---|---|---|
| Pure | CoT | 67.2 | 0.0 | 55.1 | 0.0 | 35.5 | 0.0 |
| RAG | ToG | 64.9 | 8.2 | 54.1 | 4.9 | 38.9 | 17.0 |
| RAG | PoG | 61.4 | 12.6 | 49.8 | 16.3 | 49.7 | 36.1 |
| RAG | GoG | 62.3 | 26.9 | 48.1 | 16.5 | 29.1 | 12.4 |
| **RAG** | **ARoG** | **74.7** | **58.9** | **60.0** | **36.3** | **78.7** | **71.8** |
| | 提升 | +6.1 | +5.8 | +4.9 | +19.8 | +16.4 | +9.0 |

**#Filtered 设定**（仅含 LLM 内部知识无法回答的问题）下，ARoG 的优势更为显著。ToG 在 WebQSP #Fil 上仅 8.2%，而 ARoG 达到 58.9%。GrailQA 上 ARoG 以 78.7% 的 #Tot 和 71.8% 的 #Fil 大幅领先。

### 消融实验

| 关系检索 | 关系过滤 | 实体抽象 | 结构抽象 | WebQSP #Tot | CWQ #Tot | GrailQA #Tot |
|---|---|---|---|---|---|---|
| ✓ | × | × | × | 63.9 | 36.4 | 72.6 |
| ✓ | × | ✓ | × | 68.3 | 47.9 | 76.9 |
| ✓ | × | ✓ | ✓ | 72.9 | 59.5 | 78.3 |
| ✓ | ✓ | ✓ | × | 68.1 | 50.3 | 76.3 |
| ✓ | ✓ | ✓ | ✓ | **74.7** | **60.0** | **78.7** |

- 移除 RA：#Filtered 下降至少 3.8%（私有知识检索依赖 RA）
- 移除 SA：CWQ 上下降最多（多跳推理依赖结构化路径）
- 两个抽象策略互补，共同使用效果最佳

### 关键发现

1. **现有 RAG 方法在隐私保护场景下近乎失效**: ToG、PoG 等严重依赖实体语义信息，匿名化后性能大幅下降
2. **SP 方法的稳定性**: 语义解析方法不依赖 KG 中实体信息，在 #Total 到 #Filtered 转变中性能相对稳定
3. **抽象概念路径优于 CoT 和问题分解**: 与 CoT rationale 和子问题分解相比，抽象概念路径在检索中更有效，因为它包含了对匿名实体的概念推断
4. **检索效率**: 在 GrailQA 上 ARoG 的总 token 使用量最低（5605），效率最优

## 亮点与洞察

- **问题定义新颖**: 首次定义并系统解决"隐私保护 RAG"场景，为 KG 隐私保护提供了全新视角
- **关系作为语义桥梁**: 利用 KG 中的关系（模式信息）推断匿名实体的语义，是一个巧妙的设计——关系本身不是隐私数据，但可以用来重建语义
- **概念路径的错误容忍性**: 抽象概念路径即使包含错误的实体名称也能有效工作，这大大提高了系统在开放域问题上的鲁棒性
- **完整的隐私闭环**: MID 仅在用户侧还原为真实名称，LLM 端自始至终无法获取实体语义

## 局限与展望

1. **依赖关系信息**: 如果 KG 中关系也需要匿名化，当前方法将失效
2. **LLM API 调用开销**: 每次检索需要多次 LLM 调用（平均 12-25 次），在成本敏感场景下可能不可接受
3. **仅评估 Freebase**: 未在其他类型的 KG（如 Wikidata、领域专用 KG）上验证
4. **主题实体假设**: 假设问题中的主题实体名称是可用的，这在某些隐私场景下可能不成立
5. **概念抽象质量依赖 LLM**: 抽象质量受限于 LLM 的推理能力，对复杂领域可能不够准确

## 相关工作与启发

- RAG 系统从"直接暴露知识"到"隐私保护检索"的演进是必然趋势
- 知识图谱中关系和实体的"解耦"策略可推广到其他隐私保护场景
- 抽象概念推理（从具体到抽象再到具体）的思路可应用于联邦知识图谱查询
- 未来可结合安全多方计算或同态加密进一步增强隐私保证

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (首次定义隐私保护RAG场景，抽象推理策略新颖)
- 实验充分度: ⭐⭐⭐⭐⭐ (3数据集2设定，消融+效率+定量+深度分析，非常全面)
- 写作质量: ⭐⭐⭐⭐⭐ (问题定义清晰，方法描述细致，图示直观)
- 价值: ⭐⭐⭐⭐⭐ (开创隐私保护RAG新方向，实用价值高)

<!-- RELATED:START -->

## 相关论文

- [Estimating Privacy Leakage of Augmented Contextual Knowledge in Language Models](../../ACL2025/ai_safety/estimating_privacy_leakage_of_augmented_contextual_knowledge_in_language_models.md)
- [Ask and Remember: A Questions-Only Replay Strategy for Continual Visual Question Answering](../../ICCV2025/ai_safety/ask_and_remember_a_questions-only_replay_strategy_for_continual_visual_question_.md)
- [Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [Resource-Adaptive Federated Text Generation with Differential Privacy](../../ICLR2026/ai_safety/resource-adaptive_federated_text_generation_with_differential_privacy.md)
- [PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration](prism_privacy-aware_routing_for_adaptive_cloud-edge_llm_inference_via_semantic_s.md)

<!-- RELATED:END -->
