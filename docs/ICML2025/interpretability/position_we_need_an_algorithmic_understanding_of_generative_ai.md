---
title: >-
  [论文解读] Position: We Need An Algorithmic Understanding of Generative AI
description: >-
  [ICML 2025][算法理解] 提出 AlgEval 框架，倡导系统性地研究生成式 AI 学习和使用的算法——包括算法原语（vocabulary）及其组合（grammar）——作为替代纯粹规模扩展的理解路径，并通过图导航任务的案例研究展示了 top-down 假说与 bottom-up 验证相结合的方法论。
tags:
  - ICML 2025
  - 算法理解
  - 可解释性
  - 机制解释性
  - 算法原语
  - LLM
  - 图搜索
  - 推理时计算
  - AlgEval
---

# Position: We Need An Algorithmic Understanding of Generative AI

**会议**: ICML 2025  
**arXiv**: [2507.07544](https://arxiv.org/abs/2507.07544)  
**代码**: 无（立场论文）  
**领域**: 可解释性  
**关键词**: 算法理解, 可解释性, 机制解释性, 算法原语, LLM, 图搜索, 推理时计算, AlgEval

## 一句话总结

提出 AlgEval 框架，倡导系统性地研究生成式 AI 学习和使用的算法——包括算法原语（vocabulary）及其组合（grammar）——作为替代纯粹规模扩展的理解路径，并通过图导航任务的案例研究展示了 top-down 假说与 bottom-up 验证相结合的方法论。

## 研究背景与动机

### 领域现状

**领域现状**：LLM 取得了巨大成功，但一个根本性问题未被充分重视：**LLM 到底用什么算法来解决问题？**

当前研究优先级的失衡：

**规模至上**：自 Bitter Lesson (Sutton 2019) 以来，领域重心放在数据和计算规模上，而非理解

**机制解释性的局限**：现有工作集中于底层电路分析（attention head、neuron），但缺乏算法层面的系统性理解

**可持续性危机**：数据和计算成本不断攀升，收益递减；生物智能提供了远更高效的存在证明

**推理时计算兴起**：o1/R1 等模型转向推理时计算，但其中涉及的算法仍不清楚

**核心论点**：ML 社区应优先研究生成式 AI 的算法理解，而非仅追求性能提升。算法理解可以带来：

### 现有痛点

**现有痛点**：更高效的训练和推理

### 核心矛盾

**核心矛盾**：更可信赖的模型

### 解决思路

**解决思路**：更好的多智能体系统设计

### 补充说明

**补充说明**：合规性和安全性

## 方法详解

### AlgEval 框架

AlgEval 是一个用于算法评估和理解的研究框架，核心包含两个层面：

**算法原语（Algorithmic Primitives）**——即"词汇表"：

- **基础原语**：已发现的包括函数向量（in-context learning 映射）、向量算术计算、steering 向量、复制抑制、key-value 记忆检索等
- **原语作为基础算法**：组合多个原语形成更复杂的功能电路，如间接宾语识别（induction + inhibition + duplicate token + copy heads）
- **算法作为原语**：Transformer 可展示经典 ML 方法（核方法、SVM、马尔可夫链、高阶优化）

**算法组合（Algorithmic Composition）**——即"语法"：

- 原语如何组合成更复杂的算法？
- 组合是否具有组合性（compositionality）？
- 已有证据表明 LLM 在组合推理方面存在持续失败

### 方法论路径

AlgEval 主张结合以下方法：

1. **表示和注意力分析**：层间表示相似性矩阵、注意力流分析
2. **子图和电路发现**：激活补丁、自动电路发现、归因补丁
3. **推理时计算分析**：分析序列输出中的算法行为（如回溯、搜索）
4. **五步研究流程**：(a) 识别原语库 → (b) 构建简单任务集 → (c) 创建机制库 → (d) 分析组合 → (e) 消融验证

### 案例研究：图导航任务

以 Llama-3.1-8B/70B 为对象，研究 LLM 在简单树形图导航中的算法行为。提示模型判断从起点到目标节点是否存在直接路径。

**假说**：模型可能使用经典搜索算法（BFS/DFS/Dijkstra）。

**注意力分析结果**：
- 最终 token 显著更多地关注正确路径上的节点（$b = 0.33, p < .001$），32 层中 14 层显著
- 早中层关注成对节点链接，第 13-14 层关注目标节点，第 19 层左右注意力转向正确答案
- 呈现**级联注意力扩散**模式：从各节点向前驱逐步传播

**表示分析结果**：
- t-SNE 可视化显示 "lobby"（起点）token 在跨层过程中逐渐与其他节点分离
- 目标节点 W 与最近竞争节点 Q 的表示距离逐层增大（"竞争性分离"）
- 表示在连续层间变化平滑（相似度 > 0.95），不呈现明显的离散步骤

**与经典算法的比较**：
- 将层间最高表示相似度的节点对序列与 BFS/DFS 所有可能轨迹比较
- 匹配程度极低：BFS 平均 0.18，DFS 平均 0.24
- **结论**：LLM 不执行经典搜索算法，而是使用某种依赖策略的增量注意力机制

**8B vs 70B 对比**：两个模型展现相似的注意力和表示模式，但 70B 的正确/错误路径分离更显著。搜索策略无本质差异。

## 实验关键数据

### 案例研究定量结果

- 最终 token 对正确 vs 错误路径的注意力：$b = 0.33, SE = 0.07, t(2015) = 4.51, p < .001$
- 32 层中 14 层对正确路径有显著更高注意力，仅 3 层反向
- BFS/DFS 轨迹匹配率：BFS 0.18, DFS 0.24
- 连续层间表示相似度：> 0.95（第 4-32 层）

## 亮点与洞察

1. **Marr 三层分析的算法层**：明确主张在算法层面（而非仅计算层面或实现层面）理解 LLM，填补了当前研究重心的重要空白。

2. **top-down 与 bottom-up 结合**：不同于机制解释性的纯 bottom-up 探索，AlgEval 强调先形成算法假说再进行验证，方法论上更加科学。

3. **案例研究的否定性发现**：LLM 不使用 BFS/DFS，而是采用某种基于注意力的增量搜索策略——这正是 AlgEval 框架的价值所在，能发现模型与经典算法的偏离。

4. **推理时计算的算法分析**：o1/R1 等模型的推理时计算为算法分析提供了比前馈计算更易分析的空间。

5. **实际影响**：算法理解可带来更高效的训练（算法感知的架构设计）、更低的排放、更好的安全合规。

## 局限与展望

- 案例研究仅限于简单的两步树图导航，复杂图上的行为未知
- AlgEval 目前更多是研究愿景而非成熟方法论，许多关键步骤（如自动原语发现、组合分析）仍待开发
- 分析聚焦于 Llama-3.1，未验证其他模型架构
- 理论框架缺乏形式化保证
- 作为立场论文，实验贡献有限

## 相关工作

- **机制解释性**：Olah 2020 (电路)、Wang 2023 (IOI 电路)、Olsson 2022 (induction heads)
- **算法推理**：Weiss 2021 (RASP)、Zhou 2024 (RASP 扩展)
- **表示分析**：Todd 2024 (函数向量)、Kornblith 2019 (CKA)
- **推理时计算**：Snell 2024 (推理扩展)、DeepSeek-AI 2025 (R1)

## 评分

⭐⭐⭐ — 作为立场论文提出了重要的研究方向和系统性框架，案例研究提供了具体示例。但实质性技术贡献有限，许多研究方向停留在展望层面。核心洞察（LLM 不用经典搜索）虽有趣但并不意外。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Position: The Reasoning Trap — Logical Reasoning as a Mechanistic Pathway to Advanced AI Self-Awareness](../../ICLR2026/interpretability/the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_situational_.md)
- [\[ICML 2025\] SafetyAnalyst: Interpretable, Transparent, and Steerable Safety Moderation for AI Behavior](safetyanalyst_interpretable_transparent_and_steerable_safety_moderation_for_ai_b.md)
- [\[NeurIPS 2025\] Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms](../../NeurIPS2025/interpretability/tropical_attention_neural_algorithmic_reasoning_for_combinatorial_algorithms.md)
- [\[ACL 2025\] Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework](../../ACL2025/interpretability/towards_explainable_temporal_reasoning_in_large_language_models_a_structure-awar.md)
- [\[ACL 2025\] Position-aware Automatic Circuit Discovery](../../ACL2025/interpretability/position-aware_automatic_circuit_discovery.md)

</div>

<!-- RELATED:END -->
