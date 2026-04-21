---
title: >-
  [论文解读] Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs
description: >-
  [ICLR 2026][图学习][LLM引用检测] 通过构建 10000 篇论文的配对引用图（人类 vs GPT-4o 生成 vs 随机基线），发现 LLM 生成的参考文献在图拓扑结构上与人类几乎不可区分（RF 仅 60% 准确率），但语义嵌入可有效检测（RF 83%，GNN 93%），说明 LLM 精确模仿了引用拓扑但留下了可检测的语义指纹。
tags:
  - ICLR 2026
  - 图学习
  - LLM引用检测
  - 引用图
  - 图神经网络
  - 语义嵌入
  - 学术诚信
---

# Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs

**会议**: ICLR 2026  
**arXiv**: [2601.20704](https://arxiv.org/abs/2601.20704)  
**代码**: 无  
**领域**: AI安全 / 图学习  
**关键词**: LLM引用检测, 引用图, 图神经网络, 语义嵌入, 学术诚信

## 一句话总结
通过构建 10000 篇论文的配对引用图（人类 vs GPT-4o 生成 vs 随机基线），发现 LLM 生成的参考文献在图拓扑结构上与人类几乎不可区分（RF 仅 60% 准确率），但语义嵌入可有效检测（RF 83%，GNN 93%），说明 LLM 精确模仿了引用拓扑但留下了可检测的语义指纹。

## 研究背景与动机

**领域现状**：LLM 越来越多地被用于合成科学知识、起草文献综述和建议参考文献。先前研究发现 LLM 生成的参考文献在粗粒度指标上与人类相似（标题长度、团队规模、引用数），但在细节上有系统偏差（马太效应加强、偏好近期论文、减少自引用）。

**现有痛点**：尚不清楚能否可靠地区分 LLM 和人类生成的参考文献列表。单条引用审计（如 LLM-Check）不足以捕获列表级别的模式。

**核心矛盾**：LLM 是否真正理解引用结构，还是只是表面模仿？如果拓扑结构相同，差异在哪里？

**本文目标**：系统评估 LLM 生成的引用图与人类引用图在结构和语义两个维度上的差异，并开发检测方法。

**切入角度**：渐进式建模策略——从可解释的图结构特征到语义嵌入，再到 GNN，逐步分解拓扑 vs 语义的贡献。

**核心 idea**：LLM 参考文献"结构上像人类，语义上有偏差"——检测应针对内容信号而非图结构。

## 方法详解

### 整体框架

从 SciSciNet 采样 10000 篇论文 → 分别构建真实引用图、GPT-4o 生成引用图、领域匹配随机基线图 → 提取结构特征（度中心性/接近中心性/特征向量中心性/聚类系数/边数）→ 提取语义嵌入（OpenAI 3072-D）→ RF + GNN 三分类评估。

### 关键设计

1. **引用图构建**:

    - **功能**：为每篇论文构建配对的真实/生成引用图
    - **核心思路**：焦点论文为主节点，引用论文为子节点，引用关系从 SciSciNet 检索。GPT-4o 通过标题+摘要+作者等信息纯参数化生成。随机基线按领域均匀重排引用保持度分布
    - **设计动机**：控制实验——同一焦点论文的三种引用图直接可比

2. **结构特征 vs 语义嵌入对比**:

    - **结构特征**：度/接近/特征向量中心性、聚类系数、边数 → RF 分类
    - **语义嵌入**：OpenAI text-embedding-3-large (3072-D) → 图级聚合 → RF / 作为 GNN 节点特征
    - **设计动机**：分离拓扑信号和内容信号的贡献

3. **GNN 图分类**:

    - **功能**：用 GCN/GAT/GIN/GraphSAGE 进行图级二分类
    - **核心思路**：节点特征为结构属性（5-D）或语义嵌入（3072-D），图级 readout 后二分类
    - **设计动机**：GNN 能联合利用结构和语义信号

### 损失函数 / 训练策略

Adam 优化器，70/15/15 分割，平衡数据集。GPT-4o + Claude Sonnet 4.5 双 LLM 验证鲁棒性。SPECTER + OpenAI 双嵌入模型验证。

## 实验关键数据

### 主实验

| 方法 | GT vs GPT | GT vs Random | GPT vs Random |
|------|----------|-------------|--------------|
| RF (结构特征) | 0.608 | 0.896 | 0.928 |
| RF (语义嵌入) | **0.835** | 0.908 | 0.953 |
| GNN (结构特征) | ~0.55 | ~0.90 | ~0.93 |
| GNN (语义嵌入) | **0.93** | ~0.95 | ~0.97 |

### 消融实验

| 配置 | GT vs GPT 准确率 | 说明 |
|------|----------------|------|
| GNN + 嵌入 | 93% | 最佳 |
| RF + 嵌入 | 83.5% | 语义嵌入贡献大 |
| RF + 结构 | 60.8% | 接近随机 |
| GNN + 结构 | ~55% | 结构完全不够 |
| 随机嵌入替换 | ~50% | 确认非维度效应 |
| 跨生成器（GPT训练→Claude测试） | ~72% | 泛化到其他LLM |

### 关键发现
- **拓扑几乎不可区分**：GPT 引用图的中心性、聚类系数与真实图高度重叠，RF 仅 60%
- **语义指纹可检测**：嵌入特征将准确率从 60% 提升到 83%（RF）/ 93%（GNN）
- **随机基线容易区分**：真实 vs 随机 89%+，GPT vs 随机 93%+——说明 GPT 确实生成了结构合理的引用
- **跨 LLM 泛化**：GPT-4o 训练的分类器对 Claude 仍有 72% 准确率
- **用随机嵌入替换后准确率降到 50%**，确认是语义结构而非维度带来的区分力

## 亮点与洞察
- **"结构像人，语义有偏"的发现**对审计和去偏策略有直接指导意义——应关注内容信号而非图结构
- **领域匹配随机基线**的设计严谨——同领域重排引用控制了主题分布
- **渐进式分析**（结构→嵌入→GNN）清晰展示了每个层次的贡献

## 局限与展望
- 仅测试了参数化生成（无 RAG），实际应用中 LLM 可能有检索增强
- 语义差异的具体维度（近期偏好、声望偏好等）未深入分析
- 3072-D 嵌入的哪些维度驱动区分力？
- 仅二分类，未探索多分类（部分 LLM 参考）

## 相关工作与启发
- **vs LLM-Check**：LLM-Check 审计单条引用存在性，本文评估整个引用列表的图级模式
- **vs Algaba et al.**：先前工作发现粗粒度一致性，本文通过 GNN + 嵌入实现高准确率自动检测

## 评分
- 新颖性: ⭐⭐⭐⭐ 引用图+GNN 的组合新颖，但分析框架本身较直接
- 实验充分度: ⭐⭐⭐⭐⭐ 10000 图、双 LLM、双嵌入模型、多基线、随机嵌入控制，非常全面
- 写作质量: ⭐⭐⭐⭐ 可视化出色，逐层分析清晰
- 价值: ⭐⭐⭐⭐ 对学术诚信和 AI 辅助写作有实际意义

<!-- RELATED:START -->

## 相关论文

- [LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification](../../ICML2025/graph_learning/llm_enhancers_for_gnns_an_analysis_from_the_perspective_of_causal_mechanism_iden.md)
- [On the Expressive Power of GNNs for Boolean Satisfiability](on_the_expressive_power_of_gnns_for_boolean_satisfiability.md)
- [RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation](ras_retrieval-and-structuring_for_knowledge-intensive_llm_generation.md)
- [Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)
- [Sentient: Detecting APTs Via Capturing Indirect Dependencies and Behavioral Logic](../../AAAI2026/graph_learning/sentient_detecting_apts_via_capturing_indirect_dependencies_and_behavioral_logic.md)

<!-- RELATED:END -->
