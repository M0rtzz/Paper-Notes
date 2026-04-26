---
title: >-
  [论文解读] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs
description: >-
  [NeurIPS 2025][图学习][知识图谱问答] 提出 DP（Deliberation on Priors）框架，通过渐进式知识蒸馏利用知识图谱的结构先验生成忠实的关系路径，并通过推理内省策略基于约束先验验证推理可靠性，在 KGQA 基准上达到新 SOTA。
tags:
  - NeurIPS 2025
  - 图学习
  - 知识图谱问答
  - 可信推理
  - 结构先验
  - 约束推理
  - 知识蒸馏
---

# Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs

**会议**: NeurIPS 2025  
**arXiv**: [2505.15210](https://arxiv.org/abs/2505.15210)  
**代码**: [GitHub](https://github.com/mira-ai-lab/Deliberation-on-Priors)  
**领域**: 图学习 / KGQA  
**关键词**: 知识图谱问答, 可信推理, 结构先验, 约束推理, 知识蒸馏

## 一句话总结

提出 DP（Deliberation on Priors）框架，通过渐进式知识蒸馏利用知识图谱的结构先验生成忠实的关系路径，并通过推理内省策略基于约束先验验证推理可靠性，在 KGQA 基准上达到新 SOTA。

## 研究背景与动机

基于知识图谱的检索增强生成（KG-RAG）旨在为 LLM 提供最新的外部知识以减少幻觉。然而，现有方法未能充分利用知识图谱中蕴含的先验知识，具体表现在两个方面：

(1) **结构信息未被充分利用**: 从主题实体到答案实体的关系路径可以增强 LLM 对 KG 结构模式的感知，提升推理的忠实性。(2) **显式和隐式约束被忽视**: 如类型约束、多实体约束、时间约束等可用于筛选候选路径和指导回溯，提升推理可靠性。

现有方法要么端到端检索相关三元组后直接生成，要么逐步分解问题后检索，但都没有充分利用这些先验知识。特别是在复杂问题上（如 CWQ 数据集需要 4-hop 推理），LLM 的推理可靠性严重不足。

## 方法详解

### 整体框架

DP 框架包含四个核心模块：**蒸馏（Distillation）**→**规划（Planning）**→**实例化（Instantiation）**→**内省（Introspection）**。分为离线和在线两个阶段：离线阶段通过 SFT + KTO 将结构先验蒸馏到 LLM 中；在线阶段通过路径生成、选择、实例化和约束验证实现可信推理。

### 关键设计

**1. 渐进式知识蒸馏：SFT + KTO**

- **功能**: 将知识图谱的结构模式注入 LLM，使其能生成忠实的关系路径
- **核心思路**: 首先收集弱监督信号——从训练集中提取问题到关系路径的映射 $\boldsymbol{\mathcal{P}}_w(q) = \text{ShortestPath}_{\text{Dijkstra}}(\mathcal{G}_k(e_s), e_s, e_t)$。然后使用 SFT 训练 LLM 最大化条件对数似然 $\mathcal{L}_{\text{SFT}} = \sum_{t=1}^{T} \log P_\theta(r_t^* | r_{<t}^*, q, e_s)$。接着通过 KTO（Kahneman-Tversky Optimization）进一步优化——构造正/负路径数据（通过路径截断、实体-路径交换、关系删除三种扰动），解决正负样本 1:3 的严重不平衡问题
- **设计动机**: 单纯 SFT 不足以让 LLM 区分语义有效和无效的路径；KTO 相比 DPO 更适合处理正负样本不平衡的情况，其基于 Kahneman-Tversky 前景理论的效用最大化目标更为鲁棒

**2. 推理内省策略：约束提取 + 验证 + 回溯**

- **功能**: 确保最终推理路径满足问题的约束条件，提升回答可靠性
- **核心思路**: 预定义 5 类约束（类型、多实体、显式时间、隐式时间、序数）。在线推理时，先由 LLM 提取问题中的约束 $\mathcal{C}(q)$，然后验证实例化的推理路径 $\mathbb{P}$ 是否满足约束：$\mathcal{J}(q, e_s, \mathbb{P}) = 1$ 当 $\mathbb{P} \models \mathcal{C}(q)$。若不满足则触发回溯机制，反馈违反的约束信息，重新选择和验证关系路径
- **设计动机**: 现有方法在获得推理路径后直接生成答案，缺乏验证机制。约束先验提供了可执行的检查条件，回溯机制减少了假阳性推理路径的负面影响

**3. 一对多路径映射**

- **功能**: 提高路径生成的覆盖率和多样性
- **核心思路**: 不同于 RoG 的一对一映射，DP 收集从主题实体到答案实体的**所有最短路径**作为弱监督信号，形成一对多的问题-路径映射。对于多主题实体的问题，独立为每个实体生成候选路径后合并
- **设计动机**: 一对一映射可能错过有效的替代路径，一对多映射在 WebQSP 上路径生成 F1 从 59.3% 提升到 76.7%（相对提升 29.3%）

### 损失函数 / 训练策略

路径生成器基于 LLaMA3.1-8B-Instruct，使用 LoRA 高效适配。SFT 训练 2 epoch，KTO 训练 1 epoch。KTO 损失：$\mathcal{L}_{\text{KTO}} = \mathbb{E}_{(x,y) \sim D}[\lambda_y - v(x,y)]$，其中价值函数 $v(x,y)$ 根据正负样本分别计算。在线推理阶段，路径选择和约束验证使用 few-shot prompting。

## 实验关键数据

### 主实验

| 方法 | 类型 | WebQSP H@1 | WebQSP F1 | CWQ H@1 | CWQ F1 |
|------|------|-----------|-----------|---------|---------|
| RoG | SL | 80.8 | 70.8 | 57.8 | 56.2 |
| GNN-RAG | SL | 82.8 | 73.5 | 62.8 | 60.4 |
| DoG (GPT-4) | ICL | 65.4 | 55.6 | 41.0 | 46.4 |
| LightPROF | HL | 83.8 | - | 59.3 | - |
| **DP (GPT-4.1)** | HL | **86.7** | **80.1** | **75.8** | **69.4** |

### 消融实验

| 设置 | WebQSP H@1 | WebQSP F1 | CWQ H@1 | CWQ F1 |
|------|-----------|-----------|---------|---------|
| DP (GPT-4.1) | **86.7** | **80.1** | **75.8** | **69.4** |
| GPT-4.1 (vanilla) | 71.0 | 54.6 | 53.0 | 48.9 |
| w/o KTO | 84.7 | 77.3 | 74.6 | 67.3 |
| w/o Introspection | 82.0 | 75.7 | 70.8 | 65.2 |
| w/o 约束预定义 | 83.4 | 76.4 | 74.4 | 68.5 |

### 关键发现

1. **CWQ 上 H@1 提升 13%**: 相比前 SOTA LightPROF，DP 在 CWQ 上 H@1 从 59.3 提升至 75.8
2. **内省模块是最关键组件**: 移除内省导致最大性能下降（WebQSP H@1 ↓4.7%，CWQ H@1 ↓5.0%）
3. **LLM 交互次数仅需 2.5-2.9 次**: 远少于 ToG（15.9-22.6 次）和 PoG（9.0-13.3 次），token 消耗也最低
4. **H 和 H@1 的差距仅约 10%**: 远优于 ToG 的 46.2% 差距，体现了 DP 的推理可靠性
5. **GPT-4.1 触发回溯比 GPT-3.5 更频繁**: 更强的指令遵循能力带来更严格的约束检查

## 亮点与洞察

- 将 KG 先验知识（结构 + 约束）系统化整合到 LLM 推理中，设计优雅且有效
- KTO 在正负样本严重不平衡下的应用具有技术创新性
- 回溯机制为 LLM 推理引入了自我纠错能力，提升了系统的可靠性
- 极低的 LLM 交互次数（2-3 次）使得该方法在实际部署中更加实用
- 指出了现有工作中 H 和 H@1 指标被混用的问题，具有学术规范价值

## 局限与展望

- 约束类型需要人工预定义（5 类），在垂直领域的扩展需要额外人力
- 路径生成器依赖弱监督信号的质量，训练集标注不准确时可能传播错误
- 未来计划研究约束类型的自动提取和总结方法，减少人工干预
- 可探索将内省机制应用到其他需要可信推理的 LLM 任务中

## 相关工作与启发

- **KG-RAG 方法**: RoG、ToG、DoG 等方法逐步完善了 LLM 与 KG 的交互，但缺乏对先验知识的充分利用
- **KTO 优化**: 相比 DPO，KTO 更适合不平衡数据，本文的应用为 KGQA 领域提供了新的对齐范式
- **启发**: 知识图谱不仅是 LLM 的外部知识源，其结构本身就包含丰富的推理先验，系统化利用这些先验可以显著提升推理的忠实性和可靠性

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 先验知识的系统化利用 + KTO 在路径生成中的创新应用 + 约束驱动的内省回溯机制
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个数据集 + 多 LLM 集成验证 + 全面消融 + 效率分析 + 错误分析
- **写作质量**: ⭐⭐⭐⭐ 方法描述详尽，公式严谨
- **价值**: ⭐⭐⭐⭐⭐ KGQA 新 SOTA + 高效实用 + 可信推理框架具有广泛适用性

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models](../../ICML2025/graph_learning/graph-constrained_reasoning_faithful_reasoning_on_knowledge_graphs_with_large_la.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](table_as_a_modality_for_large_language_models.md)
- [\[NeurIPS 2025\] Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs](dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)
- [\[ACL 2025\] Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](../../ACL2025/graph_learning/kg_llm_trustworthy_qa.md)
- [\[ACL 2025\] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](../../ACL2025/graph_learning/fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)

<!-- RELATED:END -->
