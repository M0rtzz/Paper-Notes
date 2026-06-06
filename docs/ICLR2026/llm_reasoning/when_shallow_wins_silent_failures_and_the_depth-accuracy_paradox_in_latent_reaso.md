---
title: >-
  [论文解读] When Shallow Wins: Silent Failures and the Depth-Accuracy Paradox in Latent Reasoning
description: >-
  [ICLR 2026][LLM推理][latent reasoning] 本文系统分析了 Qwen2.5-Math-7B 在 GSM8K 上的隐式推理行为，发现 81.6% 的正确预测来自计算不一致的路径，8.8% 为静默失败（高置信错误），并揭示了推理深度与准确率之间的悖论关系。
tags:
  - "ICLR 2026"
  - "LLM推理"
  - "latent reasoning"
  - "faithfulness metrics"
  - "silent failures"
  - "depth-accuracy paradox"
  - "computational stability"
---

# When Shallow Wins: Silent Failures and the Depth-Accuracy Paradox in Latent Reasoning

**会议**: ICLR 2026  
**arXiv**: [2603.03475](https://arxiv.org/abs/2603.03475)  
**代码**: [github.com/SubramanyamSahoo/When-Shallow-Wins](https://github.com/SubramanyamSahoo/When-Shallow-Wins)  
**领域**: LLM推理  
**关键词**: latent reasoning, faithfulness metrics, silent failures, depth-accuracy paradox, computational stability

## 一句话总结

本文系统分析了 Qwen2.5-Math-7B 在 GSM8K 上的隐式推理行为，发现 81.6% 的正确预测来自计算不一致的路径，8.8% 为静默失败（高置信错误），并揭示了推理深度与准确率之间的悖论关系。

## 研究背景与动机

**领域现状**：Chain-of-Thought (CoT) 提示极大地提升了 LLM 的多步推理能力，但显式推理消耗上下文窗口、引入延迟，且可能并不反映真实的计算过程。近期架构展示了**隐式推理**（latent reasoning）——在激活空间内完成多跳推理而无需语言化表达。

**现有痛点**：当前基准测试仅关注单样本准确率，无法衡量模型内部计算的可靠性。一个正确的答案可能来自稳定的推理路径，也可能来自脆弱的启发式捷径。在教育辅导、自动评分等高风险场景下，这种不透明性带来部署安全隐患。

**核心矛盾**：基准准确率 ≠ 计算可靠性。模型可以通过统计捷径达到看似不错的准确率，但底层推理路径高度不稳定，一旦输入稍作变化就可能产生截然不同的结果。

**本文方案**：提出一套组合式 faithfulness 度量（激活稳定性 $\mathcal{S}$、推理跳数对齐 $\mathcal{A}$、深度效率 $\mathcal{E}$），通过多次前向传播的激活分析，量化隐式推理的真实计算质量，并构建安全评估框架识别静默失败。

## 方法详解

### 整体框架

本文围绕三个核心研究问题展开：(1) 如何量化隐式推理的 faithfulness；(2) 隐式推理是压缩版 CoT 还是不同的计算策略；(3) 模型能否同时通过稳定和不稳定路径达到高准确率。方法论包含 faithfulness 度量设计、层级可解释性分析、安全评估框架和压缩假设检验四个模块。

### 关键设计1: 组合式 Faithfulness 度量

度量由三个可解释分量加权组成：

$$\mathcal{F}(q) = 0.35 \cdot \mathcal{S}(q) + 0.35 \cdot \mathcal{A}(q) + 0.30 \cdot \mathcal{E}(q)$$

- **激活稳定性 $\mathcal{S}$**：对同一问题执行两次独立前向传播，计算各层激活的余弦相似度均值，并乘以方差惩罚项 $(1 - \min(\sigma^2, 1))$，同时捕捉平均一致性和跨层稳定性
- **推理跳数对齐 $\mathcal{A}$**：检测激活幅度变化超过第 75 百分位的层作为推理转折点，通过对数比率衡量观测到的转折频率与期望推理步数的匹配程度
- **深度效率 $\mathcal{E}$**：综合活跃层比例、跳数密度和幅度分布，与理论最优深度 $\mathcal{D}_{\text{opt}} = \min(s/L, 1)$ 的偏离程度

判定阈值为 $\mathcal{F} \geq 0.60$、$\mathcal{S} \geq 0.65$、$\mathcal{E} \geq 0.60$，三者需同时满足。

### 关键设计2: 安全评估框架与静默失败检测

基于激活稳定性和正确性的二维分类，将模型输出划分为四种模式：

| 模式 | 条件 | 风险等级 |
|------|------|----------|
| True Positive | 正确 ∧ $\mathcal{S} \geq 0.65$ | 低 |
| Lucky Guess | 正确 ∧ $\mathcal{S} < 0.65$ | 中 |
| True Negative | 错误 ∧ $\mathcal{S} < 0.65$ | 预期 |
| Silent Failure | 错误 ∧ $\mathcal{S} \geq 0.65$ | **高** |

静默失败率 $\text{SFR} = |\text{Silent Failures}| / |\mathcal{P}|$ 量化了模型在高置信下产生错误输出的比例。

### 关键设计3: 压缩假设检验

通过比较三种推理模式（隐式、显式 CoT、简洁推理）的层级激活幅度轨迹的余弦相似度：

$$\text{SR} = \frac{1}{|\mathcal{P}|} \sum_{q \in \mathcal{P}} \mathbb{I}[\text{sim}_{\text{traj}}(q, \text{impl}, \text{conc}) \geq 0.7]$$

若 $\text{SR} \geq 0.75$ 则支持压缩假设，$\text{SR} < 0.50$ 则拒绝。

## 实验关键数据

### 主实验

在 500 个 GSM8K 问题上评估 Qwen2.5-Math-7B：

| 指标 | 均值 | 标准差 |
|------|------|--------|
| 准确率 | 0.610 | 0.488 |
| 推理深度 $\mathcal{D}$ | 0.514 | 0.012 |
| 激活熵 $H$ | 0.090 | 0.041 |
| 稳定性 $\mathcal{S}$ | 0.600 | 0.200 |
| 对齐 $\mathcal{A}$ | 0.687 | 0.139 |
| 效率 $\mathcal{E}$ | 0.737 | 0.030 |
| 整体 Fidelity $\mathcal{F}$ | 0.672 | 0.092 |

失败模式分布：Lucky Guess 占 49.8%（249例），True Negative 占 30.2%（151例），True Positive 占 11.2%（56例），Silent Failure 占 8.8%（44例）。仅 20% 的响应满足严格 faithfulness 标准。

### 消融实验

| 配置 | 平均 Fidelity | 与正确性的相关性 |
|------|---------------|------------------|
| Full | 0.642 | -0.315 |
| No Stability | 0.718 | -0.315 |
| No Alignment | 0.611 | -0.314 |
| No Efficiency | 0.600 | -0.311 |

跨模型比较（7B vs 1.5B）：

| 指标 | 7B | 1.5B | Δ |
|------|-----|------|---|
| 准确率 | 0.610 | 0.610 | 0.000 |
| 推理深度 | 0.514 | 0.479 | +0.034 |
| 激活熵 | 0.090 | 0.169 | -0.079 |

### 关键发现

1. **深度-准确率悖论**：Fidelity 与正确性呈弱负相关（$r = -0.21$, $p = 0.002$），但连续分析 AUROC 达 0.78，表明这是二分类阈值效应
2. **模型缩放无收益**：1.5B→7B（4.7× 参数）在评估子集上准确率完全相同（61%），大模型推理更深但未转化为性能提升
3. **隐式推理 ≠ 压缩 CoT**：仅约 20% 的隐式推理轨迹与 CoT 模式的相似度 ≥ 0.7，平均相似度仅 0.43，说明隐式推理采用了多样化的计算策略
4. **中间层因果重要性**：噪声干预实验揭示中间层（6-9层）因果贡献最大（$\gamma_6 = 0.34$），而激活幅度高峰在后期层（20-28层），暗示了双阶段计算模型

## 亮点与洞察

- **Lucky Guess 主导**：81.6% 的正确预测来自不稳定路径，表明基准准确率严重高估了真实推理能力
- **静默失败的安全隐患**：8.8% 的预测表现为"高置信错误"，在教育、医疗决策等场景中将产生严重后果
- **双阶段计算模型的发现**：中间层负责关键推理操作，后期层负责放大和输出格式化，与电路发现（circuit discovery）的研究一致
- **评估改革的呼吁**：单样本准确率不足以保证计算可靠性，需要多次推理一致性评估和稳定性加权的评分机制

## 局限与展望

- 仅在 GSM8K 的 6%（500题）上评估，结论推广需全数据集验证
- Faithfulness 度量缺乏理论基础，阈值选择为经验性的
- 仅聚焦 Qwen 单一模型家族，对其他架构的适用性未知
- 稳定性估计需要多次前向传播，限制了大模型场景的可扩展性
- 噪声干预方法粒度较粗，activation patching 等更精细技术可能更有效

## 相关工作与启发

- **CoT 推理与解释忠实度**：Lanham et al. (2023) 和 Turpin et al. (2023) 质疑语言化推理是否反映真实计算，本文将这一质疑延伸到隐式推理领域
- **机制可解释性**：Wang et al. (2023) 的电路发现和 Meng et al. (2023) 的因果干预方法为本文的层级分析提供了方法论基础
- **信息瓶颈理论**：Tishby & Zaslavsky (2015) 的信息瓶颈理论在本文中得到经验验证——后期层的激活熵急剧压缩与高激活区域重合

## 评分

⭐⭐⭐

本文提出了有价值的 faithfulness 度量框架并揭示了隐式推理的多种有趣现象，但评估范围偏小（仅 500 题）、度量缺乏理论支撑、且部分结论（如参数缩放无收益）可能受限于评估子集，整体贡献处于中等水平。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning](../../ICML2026/llm_reasoning/when_to_re-plan_subgoal_persistence_in_hierarchical_latent_reasoning.md)
- [\[ICLR 2026\] No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes](no_answer_needed_predicting_llm_answer_accuracy_from_question-only_linear_probes.md)
- [\[ICLR 2026\] When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models](when_reasoning_meets_compression_understanding_the_effects_of_pruning_and_quant.md)
- [\[ACL 2025\] Large Language and Reasoning Models are Shallow Disjunctive Reasoners](../../ACL2025/llm_reasoning/large_language_and_reasoning_models_are_shallow_disjunctive_reasoners.md)
- [\[AAAI 2026\] Answering the Unanswerable Is to Err Knowingly: Analyzing and Mitigating Abstention Failures in Large Reasoning Models](../../AAAI2026/llm_reasoning/answering_the_unanswerable_is_to_err_knowingly_analyzing_and.md)

</div>

<!-- RELATED:END -->
