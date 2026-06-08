---
title: >-
  [论文解读] Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval
description: >-
  [ACL 2026][图学习][知识图谱] 本文提出 ARK：一个 training-free 的知识图谱检索 agent，只暴露「全局词法搜索」和「单跳邻居展开」两个最小工具，让 LLM 自主在广度和深度之间切换，无需种子节点或固定跳数；在 STaRK 三图上把 Hit@1 平均推到 59.1%…
tags:
  - "ACL 2026"
  - "图学习"
  - "知识图谱"
  - "自适应检索"
  - "广度深度权衡"
  - "工具使用"
  - "轨迹蒸馏"
---

# Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval

**会议**: ACL 2026  
**arXiv**: [2601.13969](https://arxiv.org/abs/2601.13969)  
**代码**: https://github.com/mims-harvard/ark  
**领域**: 图学习 / 知识图谱检索 / RAG / LLM Agent  
**关键词**: 知识图谱, 自适应检索, 广度深度权衡, 工具使用, 轨迹蒸馏

## 一句话总结
本文提出 ARK：一个 training-free 的知识图谱检索 agent，只暴露「全局词法搜索」和「单跳邻居展开」两个最小工具，让 LLM 自主在广度和深度之间切换，无需种子节点或固定跳数；在 STaRK 三图上把 Hit@1 平均推到 59.1%，最高比 training-free baseline 提升 31.4%，并可把策略 label-free 蒸馏进 Qwen3-8B。

## 研究背景与动机
**领域现状**：把知识图谱（KG）接进 RAG 已是主流，因为 KG 把证据组织成「实体 + 类型化边」，能复用、能做关系约束。当前 KG 检索有两条路线：(i) 相似度检索（BM25、ada-002、GritLM、KAR）覆盖广但浅；(ii) 多跳遍历（Think-on-Graph、GraphFlow）能跟关系链但要先找 seed entity，且常常需要任务特定训练。

**现有痛点**：相似度方法把局部邻居编码进 embedding 后就「冻结」了，多跳查询要么扩展上下文、要么堆 message passing，复杂度爆炸；遍历方法对 seed 敏感——种子选错或不全，搜索就被困在局部、永远摸不到正确证据。更糟的是大量方法依赖图特定训练，换图就完蛋。

**核心矛盾**：KG 查询本质上同时有两种需求——**广度**（涉及多实体或松散概念，需要全图覆盖）和**深度**（证据藏在多跳路径上）；但现有系统只能在一种模式下擅长，没法在一条 trajectory 内自适应切换。

**本文目标**：搭一个 training-free 框架，让 LLM 在一条 trajectory 内自由选「全局搜」还是「关系展开」，不锁定 seed，也不预设跳数；同时设计能用 label-free 方式蒸馏进小模型的工具接口。

**切入角度**：把检索从「在固定索引上打分」改成「带工具的交互式 agent 决策」，工具集刻意做到最小（只 2 个原语），让能力来自 LLM 的工具使用，而非工具本身的复杂度。

**核心 idea**：用「全局 BM25 搜索 + 类型过滤的单跳邻居」两个原语 + 一个 ReAct 风格 agent，把多跳遍历表达成「邻居调用的多次组合」，把广度-深度切换交给 LLM 自己决定。

## 方法详解

### 整体框架
ARK 把知识图谱检索从"在固定索引上打分"改造成"带工具的交互式 agent 决策"，核心是让 LLM 在一条轨迹里自己决定该"广撒"还是"深挖"。图被形式化为 $G = \langle V, E, \phi_V, \phi_E, d_V \rangle$，agent $\mathcal{A} = \langle \text{LLM}, \mathcal{T} \rangle$ 拿到查询 $Q$ 后产出轨迹 $\tau = ((s_1, A_1, o_1), \dots, (s_T, A_T, o_T))$：每一步从两个工具里挑一个调用、拿到新观测，同时维护一个有序候选列表 $\mathcal{R}$（把工具返回的节点 append 进去，或调用 `finish` 收尾）。文本相关性自始至终用 BM25 的 $\operatorname{rel}(q, d_V(v))$ 算，快且稳，适合 agent 在轨迹中反复发短查询；可选地并发跑 $n$ 个独立 agent 再投票聚合。整套框架本体 training-free，能力全压在 LLM 的工具使用上，而非工具本身的复杂度。

### 关键设计

**1. 极简双工具接口：只给两个原语，把多跳交给组合。** 

传统遍历 agent 的老毛病是"seed 选错就全盘皆输"，一旦起点不对就被困在局部、永远摸不到正确证据。ARK 干脆只暴露两个工具：Global Search $\operatorname{Search}_G(q, k)$ 在全图节点描述上做 BM25 取 Top-k，提供"进入图的全局锚点"；Neighborhood Exploration $\operatorname{Neighbors}(v, q, F)$ 在节点 $v$ 的一跳邻域 $N_F(v) = \{u \in N(v) \mid \phi_V(u) \in F_V, \phi_E(\{u,v\}) \in F_E\}$ 内按 BM25 排前 k，支持节点/边类型过滤 $F = (F_V, F_E)$ 和子查询 $q$。LLM 通过 ReAct 提示交替调用：文本主导的查询（如 AMAZON）就连续 global search 拿候选；关系主导的查询（如 MAG 找作者论文）就先 global search 锚一个起点，再用多次 `Neighbors` 把多跳路径拼出来。关键在于 global search 在整条轨迹里永远可用，等于给 agent 一个随时能"重新看全图"的逃生口，根治了 seed 锚定病；多跳深度也不预设，agent 看够了自己 `finish`。

**2. 并发自一致：靠投票筛出稳定共识而非更多候选。** 

单个 agent 在高分支的图上很容易 drift，越搜越偏。ARK 借鉴 LLM 推理里的 self-consistency，同时跑 $n$ 个带随机解码的独立 agent，每个产出自己的 $\mathcal{R}^{(i)}$，最终按节点在所有轨迹里的出现频次排序，平票时用"最早出现位置"破——既奖励多数共识又奖励早发现。整个 rank fusion 用最简单的"拼接 + 频次排序"实现。实验里 voting 明显强于 ordering（拼接保首次）和 random（乱序），说明并发的价值不是"凑更多候选"而是"凑稳定信号"；而且 agent 之间彼此独立，端到端延迟取决于最慢的那个而非求和，几乎是白捡的性能。

**3. Label-free 轨迹蒸馏：把工具使用策略灌进 8B 学生。** 

ARK 用大闭源模型推理贵，于是把策略蒸馏到小模型。做法是拿 GPT-4.1 当 teacher 在训练集上跑 ARK 收集轨迹（每 query 3 条、最多 20 步、不做拒绝采样），这些轨迹本身就是"带工具调用 + 工具返回"的完整对话；学生 Qwen3-8B 用 next-token loss 只在 assistant token 上微调，只学"调哪个工具、参数怎么填"，全程不碰任何 ground-truth 相关性标签。训练用 LoRA、16384 上下文、lr=1e-5 跑一个 epoch（单 H100 约五小时），按 validation early stop。label-free 是这套蒸馏真正实用的关键——上一张新图只要 teacher 跑得起就能产出训练数据，不需要人工 relevance 标注。

### 损失函数 / 训练策略
ARK 本体 training-free，只有蒸馏阶段做 SFT。总预算约 18,000 trajectories / 图、约 94.4M tokens；训练时用 mask 把 user 消息和 tool output 屏蔽掉，loss 只在 assistant 的工具调用 token 上计算。聚合规则（频次 + 先发现优先）是写死的、不参与学习。

## 实验关键数据

### 主实验（STaRK，GPT-4.1 backbone）

| 类别 | 方法 | AMAZON Hit@1 | MAG Hit@1 | PRIME Hit@1 | 平均 Hit@1 | 平均 MRR |
|---|---|---|---|---|---|---|
| Training-free / Retrieval | BM25 | 44.94 | 25.85 | 12.75 | 27.85 | 36.68 |
| Training-free / Retrieval | KAR | 54.20 | 50.47 | 30.35 | 45.01 | 52.67 |
| Training-free / Agent | Think-on-Graph + GPT-4o | 20.67 | 23.33 | 16.67 | 20.22 | 31.43 |
| **Training-free / Agent** | **ARK** | **55.82** | **73.40** | **48.20** | **59.14** | **67.44** |
| 需训练 / Retrieval | mFAR | 53.0 | 55.9 | 40.0 | 49.63 | 60.20 |
| 需训练 / Retrieval | MoR | 52.19 | 58.19 | 36.41 | 48.93 | 58.77 |
| 需训练 / Agent | GraphFlow | 47.85 | 39.09 | 51.39 | 46.11 | 54.89 |
| 需训练（蒸馏） | ARK distilled (Qwen3-8B) | 54.99 | 61.66 | 31.87 | 49.51 | 58.47 |

蒸馏学生在 AMAZON 几乎追平 teacher（55.82 → 54.99），MAG +26.6 / PRIME +13.5 绝对点优于 base Qwen3-8B；teacher 保留率最高 98.5%。

### 消融实验（10% 测试子集）

| 配置 | AMAZON Hit@1 | MAG Hit@1 | PRIME Hit@1 | 含义 |
|---|---|---|---|---|
| Full ARK | 58.5 | 79.2 | 49.2 | 完整双工具 |
| w/o Neighbors | 54.5 | **30.5** | **23.1** | 砍掉邻居展开，MAG/PRIME 灾难性掉点 |
| Neighbors w/o $q$ | 56.0 | 72.1 | 44.7 | 邻居不按子查询排序，温和掉点 |
| Neighbors w/o $F$ | 55.5 | 79.2 | 42.2 | 关掉类型过滤，PRIME（异构 biomed）掉 7 个点 |

聚合策略对比：Voting 在 MAG 73.40 / Ordering 71.24 / Random 38.04（Hit@1），证明并发不是「凑更多候选」而是「凑稳定共识」。

### 关键发现
- **ARK 会按查询自适应调工具**：AMAZON 上 87.7% 调用是 global search（文本主导），MAG / PRIME 上 65.3% / 52.3% 调用转向邻居展开（关系主导），完全不需要人为告诉它「这是文本查询还是关系查询」。
- **类型过滤是异构图的关键**：PRIME 是生医多类型图，关掉 $F$ 后掉 7 个 Hit@1 点；同质图（AMAZON）影响小。这说明在重 schema 的图上类型约束几乎和遍历同等重要。
- **成功 trajectory 用邻居很节制**：失败 case 要么完全不调邻居（不知道该多跳），要么调超过 10 次（漂移）；成功 trajectory 在选择性扩张上止于约 10 次以内，验证了「自适应停止」的重要性。
- **BM25 反而比 dense 强**：在 ARK 内 BM25 在 AMAZON / PRIME 上 Hit@1 比 text-embedding-3-large 高 5–9 点；猜测原因是 agent 可以多轮迭代「补救」词法不匹配，dense 的优势在单次召回，但在多步搜索中被稀释。

## 亮点与洞察
- **「最小工具集 + 强 LLM」打败「复杂工具集 + 训练」**：只用 2 个原语就吊打 GraphFlow 这种 RL-trained agent（除了 PRIME），强调当前阶段架构创新可能不如「把搜索决策权交回 LLM」更有效。
- **Global search 永不退休**：让 global search 在整条 trajectory 都保留可用，agent 就有「回到全图」的逃生口，根除遍历 agent 的 seed 锚定病。这个设计可以迁移到任何需要「全局视野 + 局部精修」的检索任务（数据库 query、代码导航、文件系统搜索）。
- **Label-free distillation 的可复用性**：把 teacher 的工具使用轨迹直接蒸到 student，省掉了 relevance 标注的成本，这套范式对任何「LLM agent + 工具」类系统都适用——只要 teacher 跑得起就能蒸。

## 局限与展望
- 延迟代价：agent 需要多轮 LLM 调用，端到端延迟比单次稠密检索高得多；高并发 / 在线场景吃力。
- 全局搜索是 BM25 的，对释义、跨语言别名、领域同义词不友好；在低文本密度的图（如纯关系图）上表现可能掉很多。
- 最强配置依赖大闭源 LLM（GPT-4.1），开源 backbone 即便蒸馏后在 PRIME 这种长 horizon 任务上仍明显落后 teacher。
- 只在 STaRK 三图上评测，是否在工业级图（动态、亿级节点）成立尚未验证；类型 schema 噪声大时类型过滤可能反成负担。

## 相关工作与启发
- **vs Think-on-Graph**: ToG 也是 training-free LLM 多跳遍历，但靠 beam search 扩 path、没有 global search 退路，容易锚错；ARK 把 global search 内化为可随时回退的工具。
- **vs GraphFlow**: GraphFlow 用 GFlowNets RL 学遍历策略，PRIME 上更强但需要图特定训练；ARK training-free 而且工具接口更小。
- **vs KAR / mFAR / MoR**: 这些是 retriever-style 方法，靠学多字段融合或 query expansion；ARK 把检索改成 agent 交互，能力上限不被 ranker 训练所限。
- **vs ReAct / AvaTaR**: 把 ReAct 范式专门套到 KG 检索上，给出了「最小工具集 + 投票 + 蒸馏」的完整 recipe，可借鉴到通用 tool-using agent 设计。

## 评分
- 新颖性: ⭐⭐⭐⭐ 工具极简化的想法本身朴素，但「training-free + label-free 蒸馏」组合很有意思
- 实验充分度: ⭐⭐⭐⭐⭐ STaRK 三图 + 多 backbone + 蒸馏曲线 + 工具/聚合/相关性多重 ablation
- 写作质量: ⭐⭐⭐⭐ 形式化清晰，把广度-深度 trade-off 讲得很直观
- 价值: ⭐⭐⭐⭐⭐ 给 KG-RAG 一个干净的「baseline + 蒸馏」方案，工业可直接 fork

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Explore-on-Graph: Incentivizing Autonomous Exploration of LLMs on Knowledge Graphs](../../ICLR2026/graph_learning/explore-on-graph_incentivizing_autonomous_exploration_of_large_language_models_o.md)
- [\[ACL 2026\] MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation](megarag_multimodal_knowledge_graph-based_retrieval_augmented_generation.md)
- [\[ACL 2026\] LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval](logoskg_hardware-optimized_scalable_and_interpretable_knowledge_graph_retrieval.md)
- [\[ACL 2026\] TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation](tagrag_tag-guided_hierarchical_knowledge_graph_retrieval-augmented_generation.md)
- [\[AAAI 2026\] Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily](../../AAAI2026/graph_learning/beyond_fixed_depth_adaptive_graph_neural_networks_for_node_classification_under_.md)

</div>

<!-- RELATED:END -->
