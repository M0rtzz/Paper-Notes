---
title: >-
  [论文解读] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation
description: >-
  [ACL 2026][信息检索/RAG][RAG] 作者提出 Disco-RAG，把修辞结构理论（RST）显式注入 RAG pipeline——对每个 chunk 解析 intra-chunk RST 树（局部层级）+ 跨 chunk 构建 inter-chunk 修辞图（全局连贯）+ 生成 discours…
tags:
  - "ACL 2026"
  - "信息检索/RAG"
  - "RAG"
  - "RST"
  - "篇章结构"
  - "修辞图"
  - "长文档推理"
---

# Disco-RAG: Discourse-Aware Retrieval-Augmented Generation

**会议**: ACL 2026  
**arXiv**: [2601.04377](https://arxiv.org/abs/2601.04377)  
**代码**: https://dongqi.me/projects/Disco-RAG (有)  
**领域**: 信息检索 / RAG / 篇章结构  
**关键词**: RAG、RST、篇章结构、修辞图、长文档推理

## 一句话总结
作者提出 Disco-RAG，把修辞结构理论（RST）显式注入 RAG pipeline——对每个 chunk 解析 intra-chunk RST 树（局部层级）+ 跨 chunk 构建 inter-chunk 修辞图（全局连贯）+ 生成 discourse-aware blueprint 引导回答，在 Loong / ASQA / SciNews 三个长文档基准上 training-free 即拿下 SOTA（Loong overall +12.74 LLM Score）。

## 研究背景与动机

**领域现状**：RAG（Retrieval-Augmented Generation）是把外部知识接入 LLM 的主流方案，标准流程是把文档切 chunk → 向量化入库 → 查询时检索 Top-k chunk → 拼接进 prompt 让 LLM 生成。GraphRAG / RQ-RAG / StructRAG 等结构化变体陆续出现，通过知识图谱、子图、层级树等给检索增加结构信号。

**现有痛点**：现有 RAG（包括 GraphRAG 等结构化变体）有两个被忽略的篇章层面缺陷：(1) **intra-chunk structural blindness**——chunk 内部的修辞层级（哪句是核心、哪句是补充、哪些是因果/对比关系）未被建模；(2) **inter-chunk coherence gaps**——多个 chunk 之间的修辞连接（如 chunk A 的结论与 chunk B 的反例之间的对照关系）无法被识别。

**核心矛盾**：举一个反例——Chunk A 说「研究发现 12% 较低的发病率」，Chunk B 说「整体效应不显著」。标准 RAG 不知道 A 是 conditional finding（仅适用于冬季缺 D 成人），会粗放地总结为「维生素 D 降低流感风险」。本质问题是：**RAG 检索回的是 chunk 级证据，但生成需要 discourse 级推理**——这之间隔着「孤立证据」vs「连贯论证链」的鸿沟。

**本文目标**：在 inference-time（不微调）让 LLM 既能看到 chunk，又能看到 chunk 内/chunk 间的修辞结构，并在此基础上做出 plan，再生成。

**切入角度**：修辞结构理论（RST, Mann & Thompson 1987/1988）天然提供了 nucleus（核心）/satellite（卫星）+ Elaboration / Contrast / Cause 等关系标签，过去主要用于摘要和神经生成模型，本文首次系统迁移到 RAG。

**核心 idea**：用 LLM 同时充当 RST parser（解析 chunk 内 EDU + 关系）+ rhetorical graph constructor（预测 chunk 间关系）+ planner（生成基于结构的 blueprint）+ generator，4 个角色串成 pipeline，无需任何额外训练。

## 方法详解

Disco-RAG 是 inference-time strategy，不动 retriever 和 generator 的参数。整个 pipeline 在标准 RAG 的 retrieve + generate 之间插入「discourse modeling + planning」两段。

### 整体框架

标准 RAG 形式化为 $y = \arg\max_{y'} P(y' \mid q, \mathcal{C}(q; \mathcal{D}))$，其中 $\mathcal{C} = \{c_1, \dots, c_k\}$ 是 Top-$k$ 检索的 chunk。Disco-RAG 在此基础上增加三个阶段：

1. **Intra-chunk RST tree** $t_i$：对每个 chunk $c_i$ 让 LLM-based parser $\mathcal{A}$ 同时做 EDU 切分 + nucleus/satellite 角色分配 + 关系标注，得 tree $t_i = (V_i, E_i)$。**离线**完成。
2. **Inter-chunk rhetorical graph** $\mathcal{G}$：把所有 retrieved chunk **一次性 listwise** 喂给 $\mathcal{A}$，让它对每对 chunk 预测 rhetorical relation 或 UNRELATED 标签，构成有向图 $\mathcal{G} = (\mathcal{C}, \mathcal{F})$。
3. **Discourse-driven planning blueprint** $\mathcal{B}$：把 $(q, \mathcal{C}, \mathcal{T}, \mathcal{G})$ 喂给 $\mathcal{A}$ 生成 plan，列出要选哪些 salient content、按什么 argumentative flow 组织、优先呈现哪些证据。

最终生成阶段以 $(q, \mathcal{C}, \mathcal{T}, \mathcal{G}, \mathcal{B})$ 四件套作为条件：$y = \arg\max_{y'} P(y' \mid q, \mathcal{C}, \mathcal{T}, \mathcal{G}, \mathcal{B})$。

### 关键设计

1. **Intra-chunk RST tree（局部层级）**:

    - 功能：把每个 chunk 内部分解为 EDU（Elementary Discourse Unit），并在 EDU 之间建立 nucleus/satellite + Elaboration/Contrast/Cause 等关系，得到一棵 RST tree。
    - 核心思路：LLM parser $\mathcal{A}$ 联合执行 EDU 切分 + 关系预测，形式化为 $P(t_i \mid c_i; \theta_\mathcal{A}) = \prod_j P(e_{i_j} \mid c_i; \theta_\mathcal{A}) \cdot \prod_{(u,v)} P(r_{u,v} \mid e_{i_u}, e_{i_v}; \theta_\mathcal{A})$，前者是 EDU 边界概率，后者是关系概率。为节省 inference 成本，所有 intra-chunk tree 都**离线**预解析。
    - 设计动机：标准 RAG 把 chunk 当成 bag-of-tokens 抛给 generator，丢失了「核心结论 vs 旁证」的区别。RST 树明确告诉 generator「这段的核心句是哪条，哪些只是 elaboration、哪些是 condition」，避免被 satellite 信息误导（如把局部条件结论当成普遍结论）。

2. **Inter-chunk rhetorical graph（全局连贯）**:

    - 功能：在 retrieved chunks 之间建立有向修辞连接，标记每对 chunk 的关系（如 Elaboration / Contrast / Cause）或 UNRELATED。
    - 核心思路：用 **listwise** 推理——所有 $k$ 个 chunk 一次性喂给 $\mathcal{A}$，让它联合预测 $k(k-1)$ 个 ordered pair 的关系：$P(\mathcal{G} \mid \mathcal{C}) = \prod_{i=1}^k \prod_{j \ne i} P(r_{i,j} \mid \mathcal{C})$。allow UNRELATED 让模型主动剪枝。
    - 设计动机：相比 pairwise 推理（chunk A vs chunk B 单独询问），listwise 让 parser 能看到全局上下文，更容易识别「A 是 B 的反例」「C 支持 A 但反驳 B」这类需要三方对照才能判定的关系。这是 GraphRAG 等基于 entity-edge 图最缺的「论证级」结构。

3. **Discourse-aware planning blueprint**:

    - 功能：在生成前先输出一份 plan，明确「先讲什么、再讲什么、用哪些证据支撑、如何处理冲突证据」。
    - 核心思路：把 $(q, \mathcal{C}, \mathcal{T}, \mathcal{G})$ 全部传给 $\mathcal{A}$，让它产出一个动态 blueprint $\mathcal{B}$。这个 plan 既不是抽取式（不是从 chunk 复制）也不是 free-form（受 RST 树和修辞图约束），而是「discourse-aware reasoning steps」——按修辞结构组织叙述流。
    - 设计动机：相比直接 generation，先 plan 再写让 LLM 把 high-level 决策（选证据、定顺序）与 low-level 决策（措辞、连接词）解耦；相比 generic plan（不看结构），discourse-aware plan 能利用 RST 关系决定「先讲 nucleus 还是 satellite」「contrast 关系怎么呈现」。

### 损失函数 / 训练策略
**完全 training-free**，所有四个 LLM 角色（parser / graph constructor / planner / generator）共享同一基模（Llama-3.1-8B、Llama-3.3-70B 或 Qwen2.5-72B）。Retriever 用 Qwen3-Embedding-8B，chunk size = 256 tokens（无 sliding window），Top-10 retrieval，beam search width = 3。各模块通过 prompt 驱动（论文附录给出完整 prompt 模板）。

## 实验关键数据

### 主实验：3 个长文档基准（节选）

**Loong**（4 个长度档，10K → 250K tokens；4 种任务）综合表现：

| 长度档 | 方法 | Backbone | LLM Score↑ | EM↑ |
|--------|------|----------|------------|------|
| Set 1 (10K-50K) | Standard RAG | Llama-3.3-70B | 62.78 | 0.34 |
| Set 1 | StructRAG (prev SOTA) | – | 69.43 | 0.35 |
| Set 1 | **Disco-RAG** | Llama-3.3-70B | **71.00** | **0.38** |
| Set 2 (50K-100K) | Standard RAG | Llama-3.3-70B | 53.77 | 0.18 |
| Set 2 | **Disco-RAG** | Llama-3.3-70B | **63.61** | **0.28** |
| Set 4 (200K-250K) | Standard RAG | Llama-3.3-70B | 35.61 | 0.07 |
| Set 4 | StructRAG | – | 51.42 | 0.10 |
| Set 4 | **Disco-RAG** | Llama-3.3-70B | **54.62** | **0.11** |

**ASQA**：Disco-RAG (Llama-3.3-70B) 拿到 EM=42.0 / RL=42.3 / DR=32.8，全维度超越 MAIN-RAG-Llama3-8B（39.2 / 42.0 / —）与 Tree of Clarifications（— / 39.7 / 36.6）。

**SciNews**：Disco-RAG (Llama-3.3-70B) 拿到 RL=21.11 / BERTScore=65.67 / SARI=44.37 / SummaC=69.48，多数指标超越 RSTformer（20.12 / 62.80 / 41.56 / —）和 Plan-Input（— / 65.32 / — / 72.40）。

### 消融实验（Loong benchmark, Llama-3.3-70B）

| 方法 | Overall LLM Score | Overall EM | 说明 |
|------|-------------------|------------|------|
| **Disco-RAG (full)** | **62.07** | **0.24** | 三模块齐全 |
| w/o RST tree | 56.22 | 0.20 | 去掉 intra-chunk 树 → 跌 5.85 |
| w/o rhetorical graph | 57.10 | 0.21 | 去掉 inter-chunk 图 → 跌 4.97 |
| w/o planning | 59.75 | 0.22 | 去掉 planner → 跌 2.32 |
| Standard RAG | 49.33 | 0.17 | 基线 |
| w/ retrieve-and-plan | 50.64 | 0.18 | 标准 RAG + free-form plan（无结构） |
| w/ plan-and-retrieve | 51.38 | 0.18 | 先 plan 再 retrieve（无结构） |

generic planning 只比 standard RAG 涨 1.3–2.0 分，而 discourse-aware planning 涨 12+ 分，证明结构 prior 不可替代。

### 关键发现
- **结构模块 > planner 的贡献**：RST 树和修辞图各贡献 ~5 分，planner 只贡献 ~2 分。结构是基础，plan 是放大器。
- **越长文档增益越大**：Set 1（短）Disco-RAG 比 Standard RAG 高 8.22 分，Set 4（200K+ tokens）则高 19 分。证明 discourse-aware 在长文档下尤其关键——证据分散时更需要修辞 scaffold 来串联。
- **对 retrieval 噪声极鲁棒**：替换 Top-10 中 20–40% 为无关 chunk，Standard RAG 从 49.33 跌到 45.23，Disco-RAG 仍维持 56.17，说明修辞结构能让 generator 自动识别 UNRELATED 段。
- **结构扰动实验**：随机 shuffle RST 关系标签 → 62.07 → 55.48；翻转图边方向 → 55.82；shuffle plan steps → 57.50。但都仍优于 standard RAG（49.33），证明性能来自 structural 信号本身而非「多了 token」。
- **混合模型部署**：8B parser + 70B generator 拿到 60.52，逼近全 70B 的 62.07，远超 standard RAG 的 49.33——意味着结构模块可以下放到小模型节省成本。
- **SFT 正交可叠加**：在 SciNews 上 fine-tune generator 后再加 discourse 输入，RL 从 22.8 涨到 23.3，SummaC 从 72.3 涨到 74.0，证明 discourse signal 与参数学习互补。
- **人工评估**：3 名 PhD 评分（3 分 Likert），Disco-RAG 在 Faithfulness 上 2.53 vs Standard RAG 1.67，几乎追近人写参考的 2.88。

## 亮点与洞察
- **「discourse」是 RAG 一直缺的那块拼图**：GraphRAG 等结构化变体盯着 entity-level KG，本文 zoom out 到 argument-level discourse，捕捉因果/对比/elaboration 这类「论证关系」，刚好是 LLM 在多文档合成时最容易出错的层面。这个视角转换很巧妙。
- **Listwise inter-chunk relation prediction**：相比 pairwise 询问，让 LLM 一次看完所有 chunk 再决定关系，能利用全局上下文做更精准的修辞推断。这个 listwise trick 可以迁移到任何「文档间关系建模」任务（如多文档摘要的冲突检测、新闻事件聚类）。
- **三模块解耦 + 同一基模复用**：parser / graph constructor / planner / generator 都用同一个 LLM，只换 prompt，工程上极简且部署灵活。混合模型实验进一步表明可以「8B 跑结构 + 70B 跑生成」省成本。
- **结构 ablation 仍优于无结构**：即便随机扰动后的结构信号都比 standard RAG 强，说明 discourse-aware framework 的 robustness 来自「让 generator 关注结构」这件事本身，而非依赖完美 parsing。这给实际部署降低了 parser 质量门槛。
- **修辞结构在长文档下增益翻倍**：从 Set 1 的 +8 分到 Set 4 的 +19 分，证明 RAG 系统在长文档场景下最该补的能力就是 discourse modeling。

## 局限与展望
- **额外 LLM 调用 → 高 latency / token 消耗**：作者承认 RST parsing + graph construction + planning 各需一次 LLM 调用，比 standard RAG 大约慢 3-4 倍，对延迟敏感场景需要缓存复用结构、batching 或蒸馏 lighter parser。
- **依赖 backbone 的 discourse 理解能力**：在小模型（如 Llama-3.1-8B）上 parser 质量不如 70B，全 8B 版本只到 58.94 vs 全 70B 的 62.07。对更小或非 instruction-tuned 模型可能完全失效。
- **数据集偏 narrow**：三个 benchmark 都是英文 + 学术/百科风格，对其他语言、对话/代码/数学类文档的适用性未知。
- **修辞关系标签集固定**：只用了经典 RST 的少数关系（Elaboration, Contrast, Cause 等），对法律/医学等专域可能需要扩展标签集。
- **未对比 entity-level + discourse-level 联合方法**：作者把 GraphRAG 当 baseline 但没尝试「GraphRAG 的实体图 + Disco-RAG 的修辞图」融合，这块潜力未挖。

## 相关工作与启发
- **vs GraphRAG (Edge et al. 2024) / KG-RAG**：GraphRAG 用 entity-level KG，把证据按实体共现组织；Disco-RAG 用 argument-level RST，把证据按修辞关系组织。两者粒度不同——entity 解决「我提到了什么」，discourse 解决「我在论证什么」。本文在 Loong 上比 GraphRAG 高 30+ 分，证明 discourse 是当前 RAG 更紧迫的瓶颈。
- **vs StructRAG (Li et al. 2025b)**：StructRAG 用 hybrid information structurization（动态选 table/tree/graph 等格式），是 SOTA 的训练时方法；Disco-RAG training-free 且持续小幅优于 StructRAG（71.00 vs 69.43 in Set 1, 54.62 vs 51.42 in Set 4），证明结构化信号的「类型」（修辞 vs 通用）比「形式」更重要。
- **vs RST-LoRA (Liu & Demberg 2024) / RSTformer (Liu et al. 2024)**：这两者把 RST 注入摘要模型的参数（LoRA / 编码器），Disco-RAG 把 RST 注入 prompt（inference-time），更易迁移到任意 frozen LLM。
- **vs Tree of Clarifications / RQ-RAG / MAIN-RAG**：这些方法专注 query refinement / multi-agent filtering，Disco-RAG 关注「retrieved evidence 的结构」，两类工作互补，可叠加。
- **vs FLARE (Jiang et al. 2023)**：FLARE 是 active retrieval，Disco-RAG 是 post-retrieval structural enhancement，可结合：先 FLARE 决定何时检索，再 Disco-RAG 解析结构。

## 评分
- 新颖性: ⭐⭐⭐⭐ RST 注入 RAG 这一具体组合是新的，但 RST 本身和 RAG 结构化都有 prior work
- 实验充分度: ⭐⭐⭐⭐⭐ 3 数据集 × 3 backbone × 4 长度档 + 消融 + 结构扰动 + 混合部署 + SFT 叠加 + 人工评估，极其详尽
- 写作质量: ⭐⭐⭐⭐ 公式化 pipeline 清晰，图 1 的反例说明 motivation 直观；附录 prompt 全公开
- 价值: ⭐⭐⭐⭐⭐ Training-free 即可大幅提升长文档 RAG，且模块化解耦工程友好，直接可部署


## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ACL 2025\] Typed-RAG: Type-Aware Decomposition of Non-Factoid Questions for Retrieval-Augmented Generation](../../ACL2025/information_retrieval/typed-rag_type-aware_decomposition_of_non-factoid_questions_for_retrieval-augmen.md)
- [\[ACL 2026\] CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs](codepromptzip_code-specific_prompt_compression_for_retrieval-augmented_generatio.md)

</div>

<!-- RELATED:END -->
