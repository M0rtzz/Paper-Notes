---
title: >-
  [论文解读] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering
description: >-
  [ACL 2025][NLP理解][知识图谱问答] iQUEST 提出迭代式子问题引导框架，在每一步推理中动态生成当前可解答的子问题以维持推理方向，并结合 GNN 聚合二跳邻居语义信息实现"前瞻性"实体探索，在 CWQ、WebQSP、WebQuestions、GrailQA 四个基准上取得 SOTA 或接近 SOTA 的性能，且无需微调 LLM。
tags:
  - ACL 2025
  - NLP理解
  - 知识图谱问答
  - 多跳推理
  - 子问题引导
  - 图神经网络
  - 实体探索
---

# iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering

**会议**: ACL 2025  
**arXiv**: [2506.01784](https://arxiv.org/abs/2506.01784)  
**代码**: [GitHub](https://github.com/Wangshuaiia/iQUEST)  
**领域**: NLP理解  
**关键词**: 知识图谱问答, 多跳推理, 子问题引导, GNN, 实体探索

## 一句话总结

iQUEST 提出迭代式子问题引导框架，在每一步推理中动态生成当前可解答的子问题以维持推理方向，并结合 GNN 聚合二跳邻居语义信息实现"前瞻性"实体探索，在 CWQ、WebQSP、WebQuestions、GrailQA 四个基准上取得 SOTA 或接近 SOTA 的性能，且无需微调 LLM。

## 研究背景与动机

**多跳 KBQA 的核心挑战**：复杂问题需要在知识图谱中进行多跳推理（如"北太平洋地区，热带风暴 Fabio 影响区域的官方花是什么？"），但存在两个关键难题：
   - **(1) 推理路径连贯性难以维持**：随着推理步骤增多，模型容易"迷失方向"，歧义实体（如 Mexico）引入噪声进一步干扰推理
   - **(2) 关键多跳连接被过早丢弃**：现有方法多依赖一跳邻居的局部相关性评分，可能误删看似不相关但二跳邻居高度匹配的关键路径（如 "Harry Potter Films" 的二跳邻居 "John Williams" 才是关键）

**现有方法不足**：ToG、Interactive-KBQA 等代理式方法虽有效但缺乏持续推理引导；问题分解方法多为一次性静态分解，无法根据推理进展自适应调整

**人类认知启发**：研究表明人类通过"提出并解决小问题"可维持更高注意力水平。iQUEST 将此洞见应用于 LLM 推理引导

## 方法详解

### 整体框架

iQUEST 由三个核心模块组成，通过迭代循环协同工作：

1. **迭代问题引导（IQG）**：在每一步生成可由当前上下文回答的子问题
2. **二跳实体探索**：用 GNN 聚合二跳邻居信息选择最相关实体
3. **答案提取（AE）**：根据检索到的证据回答子问题，判断是否充分

每轮迭代流程：生成子问题 → 检索一跳邻居 → GNN 评估（含二跳）→ 回答子问题 → 更新上下文 → 判断是否充分

### 关键设计

**1. 迭代问题引导（Iterative Question Guidance）**

与传统问题分解的核心区别：不是对原始问题做一次性静态拆分，而是在每一步根据**当前推理状态动态生成新子问题**：

$$Q_{\text{sub}}^{(n)} = \text{IQG-LLM}(Q, \mathcal{C})$$

其中上下文 $\mathcal{C} = [Q_{\text{sub}}^{(1)}, A_{\text{sub}}^{(1)}, Q_{\text{sub}}^{(2)}, \dots, A_{\text{sub}}^{(n-1)}]$ 包含所有已解答的子问题和答案。每步 LLM 同时判断是否需要进一步分解，若不需要则当前子问题直接用于 KG 探索。

**例子**：原问题"北太平洋地区，Fabio 影响区域的官方花是什么？" → 第一步子问题："北太平洋地区哪个区域受 Tropical Storm Fabio 影响？" → 得到答案后 → 第二步子问题："该区域的官方花是什么？"

**2. 二跳邻居聚合（Two-Hop Entity Exploration with GNN）**

**Step A: 邻居检索** — 通过 SPARQL 模板检索当前实体的所有一跳邻居

**Step B: GNN 聚合二跳信息** — 对每个一跳邻居再执行 SPARQL 获取其二跳邻居，然后用 GraphSAGE 风格的聚合更新一跳邻居表示：

$$\hat{\mathbf{h}}_{1h} = \sigma\left(\mathbf{W} \cdot [\mathbf{h}_{1h} \| \text{AGG}\{\mathbf{h}_{2h} \mid e_{2h} \in \mathcal{N}(e_{1h})\}]\right)$$

- $\mathbf{h}_{1h}$/$\mathbf{h}_{2h}$：一跳/二跳邻居的 BERT 编码表示
- AGG：mean-pooling 聚合（可处理任意数量邻居）
- 拼接操作保留中心节点原始信息，避免身份稀释

**Step C: 相关性分类** — 将更新后的表示 $\hat{\mathbf{h}}_{1h}$ 与子问题表示拼接，通过两层 MLP 做二分类：

$$\hat{\mathbf{y}} = \text{Softmax}(\mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2)$$

取"相关"类概率作为得分，选 top-$k$（$k=3$）实体作为支持证据。

**3. 答案提取 LLM（AE-LLM）**

根据选出的 top-$k$ 实体和子问题用 LLM 生成中间答案，加入上下文 $\mathcal{C}$，然后判断是否已有足够信息直接回答原始问题。若充分则综合所有子问题和答案生成最终回答。

### 损失函数 / 训练策略

- **GNN 训练**：交叉熵损失 $L = -\sum_{i=1}^{2} y_i \log(\hat{y}_i)$，训练数据来自单跳推理样本，负样本通过随机采样生成
- **编码器**：bert-base-uncased（隐层 768 维），GNN 隐层 128 维
- **LLM 不微调**：IQG-LLM 和 AE-LLM 均使用现成 API（GPT-4o / DeepSeek-R1 / LLaMA）

## 实验关键数据

### 主实验：四个 KBQA 基准（Hit@1）

| 方法 | CWQ | WebQSP | WebQuestions | GrailQA |
|------|-----|--------|-------------|---------|
| Interactive-KBQA | 49.07 | 71.20 | - | - |
| ToG (GPT-4) | 69.50 | 82.60 | 57.90 | 81.40 |
| KG-CoT | 62.30 | 84.90 | 68.00 | - |
| Chain-of-Question | 78.80 | 78.10 | - | - |
| **iQUEST (GPT-4o)** | **73.85** | **88.93** | **81.23** | 73.52 |

WebQSP（88.93）和 WebQuestions（81.23）SOTA，CWQ 和 GrailQA 排第二。

### 消融实验：GNN 的一致性提升

| 模型组合 | CWQ | WebQSP | WebQuestions | GrailQA |
|----------|-----|--------|-------------|---------|
| LLaMA 3B + GPT-4o (IQG) | 20.14 | 40.42 | 48.65 | 32.87 |
| + GNN | 23.66 (+3.52) | 43.73 (+3.31) | 50.11 (+1.46) | 34.19 (+1.32) |
| GPT-4o + GPT-4o (IQG) | 68.42 | 88.10 | 80.20 | 69.30 |
| + GNN | **73.85 (+5.43)** | 88.93 (+0.83) | 81.23 (+1.03) | **73.52 (+4.22)** |

GNN 在所有模型组合和数据集上都带来一致提升。

### AE-LLM 内部知识 vs 推理能力

| AE-LLM (IQG=GPT-4o, +GNN) | CWQ | WebQSP |
|---------------------------|-----|--------|
| DeepSeek-R1 (强推理) | 55.64 | 83.21 |
| LLaMA 70B (弱推理) | 50.30 (-5.34) | 84.36 (+1.15) |
| GPT-4o (强知识) | **73.85 (+18.21)** | **88.93 (+5.72)** |

AE-LLM 更依赖内部知识（GPT-4o +18%）而非推理能力。

### 关键发现

- **IQG 存在收益递减**：IQG-LLM 推理能力超过阈值后边际收益迅速减小
- **KG 对小模型可能有害**：LLaMA 3B + KG 性能下降，小模型难以有效整合外部知识
- **WebQuestions +23% vs ToG**：得益于原始问题持续引导，有效应对查询歧义

## 亮点与洞察

1. **"引导而非分解"的范式转换**：在每步动态生成子问题，根据推理状态自适应调整，更符合人类认知模式
2. **GNN 实现"前瞻性"推理**：聚合二跳邻居信息让模型考虑下一步潜在路径，避免短视决策
3. **消融设计精巧**：基于假设区分推理能力与内部知识的贡献，得出"IQG 分担推理，AE 依赖知识"的清晰结论
4. **轻量训练**：仅训练 GNN，LLM 完全不微调

## 局限与展望

1. **双 LLM 计算开销**：IQG + AE 两次 LLM 调用增加延迟和成本
2. **GNN 仅限二跳**：对需要更深跳数推理的领域可能不够
3. **仅在 Freebase 上验证**：对 Wikidata 等其他 KG 的适应性未测试
4. **小模型无法有效利用 KG**：限制低资源场景应用

## 相关工作与启发

- **KBQA 方法**：iQUEST 结合了 IR-based 的实体探索和 SP-based 的 SPARQL 查询
- **问题分解**：CoQ 等方法侧重静态分解；iQUEST 转向动态引导
- **启发**：(1) 迭代子问题生成可推广到多步推理 NLP 任务；(2) GNN 前瞻机制可借鉴到图上检索增强场景

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|------------|------|
| 新颖性 | 8 | 迭代引导 + GNN 前瞻组合设计有创意 |
| 技术深度 | 7 | GNN 设计合理，消融深入有说服力 |
| 实验充分性 | 8 | 4 数据集 × 4 LLM，消融全面 |
| 写作质量 | 7 | 结构清晰，假设-验证式消融专业 |
| 实用价值 | 7 | 无需微调 LLM，可直接集成 |
| 总分 | 7.5 | 方法新颖有效，消融设计值得学习 |

<!-- RELATED:START -->

## 相关论文

- [Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering](self-critique_guided_iterative_reasoning_for_multi-hop_question_answering.md)
- [Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering](embqa_embedding_odqa.md)
- [A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment](a_comprehensive_graph_framework_for_question_answering_with_mode-seeking_prefere.md)
- [ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision](rescore_multihop_qa.md)
- [BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering](belle_a_bi-level_multi-agent_reasoning_framework_for_multi-hop_question_answerin.md)

<!-- RELATED:END -->
