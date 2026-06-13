---
title: >-
  [论文解读] SynapticRAG: Enhancing Temporal Memory Retrieval in Large Language Models through Synaptic Mechanisms
description: >-
  [ACL2025][LLM 其他][记忆检索] 提出 SynapticRAG，借鉴神经科学中突触传播和漏积分发放（LIF）模型，将时序关联触发与语义相似度融合，在对话记忆检索任务上较 SOTA 提升最高 14.66%。 LLM 长期记忆不足：大语言模型在多轮对话场景中难以维持连贯的长期记忆，尤其在对话上下文跨越多个会话时…
tags:
  - "ACL2025"
  - "LLM 其他"
  - "记忆检索"
  - "时序关联"
  - "突触机制"
  - "对话系统"
  - "RAG"
---

# SynapticRAG: Enhancing Temporal Memory Retrieval in Large Language Models through Synaptic Mechanisms

**会议**: ACL2025  
**arXiv**: [2410.13553](https://arxiv.org/abs/2410.13553)  
**代码**: [tamoharu/SynapticRAG-Benchmark](https://github.com/tamoharu/SynapticRAG-Benchmark)  
**领域**: LLM/NLP  
**关键词**: 记忆检索, 时序关联, 突触机制, 对话系统, RAG

## 一句话总结

提出 SynapticRAG，借鉴神经科学中突触传播和漏积分发放（LIF）模型，将时序关联触发与语义相似度融合，在对话记忆检索任务上较 SOTA 提升最高 14.66%。

## 研究背景与动机

**LLM 长期记忆不足**：大语言模型在多轮对话场景中难以维持连贯的长期记忆，尤其在对话上下文跨越多个会话时，当前 SOTA 记忆检索方法的响应相关性下降约 17%。

**现有 RAG 依赖纯语义相似度**：传统 RAG 及其变体主要基于向量余弦相似度检索知识，忽略了对话中时序分布的依赖关系，难以处理跨时间段的复杂上下文。

**认知科学的启示**：人类记忆检索依赖时序触发（Tulving, 1985）和突触连接（Hebb, 1949），而非单纯的记忆强度。时序上相近的记忆往往构成连贯的对话链。

**已有时序增强方法不足**：MemoryBank 使用艾宾浩斯遗忘曲线、MyAgent 使用记忆巩固模型，但都未能有效捕捉对话中复杂的时序关联。

**缺乏合适的评估基准**：现有数据集（如 PerLTQA）仅部分满足时序感知对话评估需求，缺少专门包含复杂时序依赖和显式标注的数据集。

**跨语言泛化需求**：对话系统需要在英/中/日等多种语言中都能有效工作，需要语言无关的记忆检索机制。

## 方法详解

### 整体框架

SynapticRAG 包含四个核心组件：(1) 记忆构建与存储——将对话轮次嵌入为向量节点构建属性图；(2) 记忆节点选择——基于语义和时序的双重过滤；(3) 刺激传播机制——在分层图中模拟突触信号传播；(4) 刺激激活——基于动态 LIF 模型决定节点是否"发放"以被选为检索结果。

### 关键设计 1：记忆构建与时序关联计算

- **功能**：将对话每一轮编码为向量，存储为图中的记忆节点，节点间以余弦相似度加权连边。每个节点维护加权脉冲序列 $[(t_i, s_i)]_{i=1}^n$ 记录刺激历史。
- **为什么**：仅靠语义相似度无法捕捉对话中的时序依赖，需要时序维度的信息来补充。
- **怎么做**：利用动态时间规整（DTW）对齐两个节点的脉冲序列，构建距离矩阵并通过指数衰减转化为关联强度矩阵 $W(i,j) = \exp(-D(i,j)/\bar{\tau})$，再通过 DTW 累积得到时序关联分数 $T_{\text{score}}$，最终与余弦相似度 $C_{\text{score}}$ 相乘得到传播分数 $P_{\text{score}} = T_{\text{score}} \cdot C_{\text{score}}$。

### 关键设计 2：刺激传播机制

- **功能**：在分层图中逐层传播刺激信号，构建从输入节点到深层记忆节点的激活路径。
- **为什么**：模拟生物神经系统中突触信号在网络中的传导过程，使语义和时序上相关联的记忆节点能被"连锁"激活。
- **怎么做**：从初始节点出发（$S_{X_0}=1$），刺激强度沿路径衰减 $S_{X_{i+1}} = P_{\text{score}}(X_i, X_{i+1}) \cdot S_{X_i}$。每层通过阈值 $\text{stim}_{\text{th}}$ 筛选合格父节点，通过 $\cos_{\text{th}}$ 确定候选子节点，当子节点从多个父节点接收刺激时选择提供最大刺激强度的父节点。传播在无合格父节点时终止。

### 关键设计 3：动态漏积分发放（LIF）激活

- **功能**：引入增强版 LIF 模型，将离散刺激转为连续输入电流，通过膜电位的动态演化决定哪些节点被激活。
- **为什么**：标准 LIF 模型的固定时间常数 τ 无法平衡近期输入与长期记忆，需要自适应机制。
- **怎么做**：输入电流生成 $\tau_X \frac{dI_X}{dt} = -I_X(t) + \sum_{t_s \in \Gamma} S_X(t)\delta(t-t_s)$；膜电位演化 $\tau_X \frac{dV_X}{dt} = -V_X(t) + I_X(t)$；动态时间常数更新 $\tau_X(t+\Delta t) = \tau_X(t) + \frac{1-\exp(-\Delta t)}{1+\exp(-\Delta t)}$（频繁输入趋近0保持敏感，稀疏输入趋近1保留长期记忆）。当膜电位超过阈值 $V_{\text{th}}$ 时节点"发放"，发放后重置为静息状态。

### 损失函数/训练策略

本文不涉及端到端训练，而是使用 Optuna 对超参数进行优化（$\cos_{\text{th}}=0.262$, $\tau_{\text{init}}=43.07$, $V_{\text{th}}=0.099$, $\text{stim}_{\text{th}}=0.037$, $V_{\text{rest}}=2.903$, $I_{\text{rest}}=-7.128$）。文本向量化使用 text-embedding-3-large，生成使用 GPT-4o。

## 实验关键数据

### 主实验：动态检索精度（ERC / ERC-MG）

| 方法 | SMRCs-EN | SMRCs-JA | PerLTQA-EN | PerLTQA-CN | 平均 |
|------|----------|----------|------------|------------|------|
| RAG | 71.55 | 81.58 | 82.14 | 84.95 | 80.15 |
| MemoryBank | 32.60 | 49.34 | 33.62 | 36.75 | 38.11 |
| MemoryBank(Adt) | 62.36 | 72.15 | 67.24 | 71.31 | 68.37 |
| MyAgent | 62.14 | 70.61 | 61.99 | 65.14 | 65.05 |
| MyAgent(Adt) | 63.24 | 77.19 | 66.49 | 69.54 | 69.19 |
| **SynapticRAG** | **86.21** | **92.32** | **90.47** | **93.12** | **90.53** |
| Δ% | +14.66 | +10.74 | +8.33 | +8.17 | +10.38 |

SynapticRAG 在所有4个数据集上均大幅超越最强基线，平均提升 10.38%，最高提升 14.66%（SMRCs-EN）。在最严格的 ERC-MG 指标下同样保持 7.76%–14.66% 的优势。

### 消融实验

| 消融变体 | SMRCs-EN | SMRCs-JA | PerLTQA-EN | PerLTQA-CN | 平均 |
|---------|----------|----------|------------|------------|------|
| 完整 SynapticRAG | +14.66 | +10.74 | +8.33 | +8.17 | +10.41 |
| 去除时序整合 | -11.82 | -8.55 | -9.68 | -9.54 | -9.90 |
| 去除刺激传播 | -8.31 | -2.41 | -3.60 | -3.36 | -4.35 |
| 去除动态时间常数 | -14.66 | -10.96 | -8.29 | -8.24 | -10.54 |

三个组件中，动态时间常数影响最大（平均-10.54%），其次是时序整合（-9.90%）和刺激传播（-4.35%）。

## 亮点与洞察

1. **跨学科创新**：将神经科学中突触传播、LIF 模型和动态时间常数等概念系统性地引入对话记忆检索，不同于简单的遗忘曲线，提供了更丰富的生物启发机制。
2. **自建高质量数据集 SMRCs**：专门构建包含复杂时序依赖的双语（英/日）对话数据集，填补了该领域评估基准的空白，Krippendorff's Alpha 达到 0.782。
3. **提出动态评估指标 ERC/ERC-MG**：针对不同查询需要不同数量记忆的实际场景，设计了自适应检索数量的评估指标，更贴近真实应用。
4. **跨语言一致性**：在英/中/日三种语言上均取得稳定提升，验证了方法的语言无关性。

## 局限与展望

1. **过度检索风险**：突触传播可能导致过多节点被激活，增加生成 prompt 长度和计算开销。可引入早停机制和剪枝策略。
2. **计算复杂度 O(n²)**：余弦相似度过滤过程导致二次复杂度，不利于超大规模记忆库。论文提出了 O(n) 的质心向量优化方案但细节在附录中。
3. **参数敏感性**：特别是 τ_scale 参数对性能影响极大，超过最优值后急剧下降，不同任务可能需要独立调参。
4. **可解释性不足**：复杂的刺激传播和发放机制降低了相比简单相似度方法的可解释性。
5. **数据集规模有限**：SMRCs 仅 101 条对话 / 456 个任务，可扩展到更大规模以验证方法鲁棒性。

## 相关工作与启发

- **RAG 系列**（Lewis 2020, Borgeaud 2022）：提供了基于语义相似度的基础检索范式，是本文的基线起点。
- **MemoryBank**（Zhong 2023）：引入艾宾浩斯遗忘曲线调整记忆权重，但时序建模能力有限。
- **MyAgent**（Hou 2024）：采用认知启发的记忆回忆模型，考虑相关性、时间和回忆概率。
- **HippoRAG**（Gutiérrez 2025）：同样从神经科学（海马体）出发设计 LLM 长期记忆，与本文关注点互补。
- **启发**：本文的分层图传播 + LIF 激活思路可以迁移到其它需要时序感知检索的场景（如事件抽取、时间线推理）。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将突触传播和动态 LIF 模型系统引入对话记忆检索，跨学科创新程度高
- **实验充分度**: ⭐⭐⭐⭐ — 4个数据集、5个基线、3种语言、完整消融和参数分析，但数据集规模偏小
- **写作质量**: ⭐⭐⭐⭐ — 公式推导完整清晰，框架图直观，但部分符号定义可以更紧凑
- **价值**: ⭐⭐⭐⭐ — 为对话系统记忆管理提供了新的生物启发范式，对 RAG 时序增强有实用参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues](tremu_towards_neuro-symbolic_temporal_reasoning_for_llm-agents_with_memory_in_mu.md)
- [\[ACL 2025\] Disentangling Memory and Reasoning Ability in Large Language Models](disentangle_memory_reasoning.md)
- [\[ACL 2025\] ChronoSense: Exploring Temporal Understanding in Large Language Models with Time Intervals of Events](chronosense_exploring_temporal_understanding_in_large_language_models_with_time_.md)
- [\[ACL 2025\] Investigating Context-Faithfulness in Large Language Models: The Roles of Memory Strength and Evidence Style](investigating_context-faithfulness_in_large_language_models_the_roles_of_memory_.md)
- [\[ACL 2025\] Enhancing the Rule Learning Ability of Large Language Model Agent through Induction, Deduction, and Abduction](idea_enhancing_the_rule_learning_ability_of_large_language_model_agent_through_i.md)

</div>

<!-- RELATED:END -->
