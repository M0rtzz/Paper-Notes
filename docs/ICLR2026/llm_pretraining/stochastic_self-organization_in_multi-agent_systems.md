---
title: >-
  [论文解读] Stochastic Self-Organization in Multi-Agent Systems
description: >-
  [ICLR 2026][预训练][multi-agent systems] 提出 SelfOrg 框架，基于 Agent 响应的语义相似度和 Shapley 值贡献估计，动态构建有向无环通讯图（DAG），实现多 Agent 系统的自组织协作。在弱模型场景下优势尤为显著。
tags:
  - "ICLR 2026"
  - "预训练"
  - "multi-agent systems"
  - "GAN"
  - "Shapley value"
  - "communication graph"
  - "DAG"
  - "LLM collaboration"
---

# Stochastic Self-Organization in Multi-Agent Systems

**会议**: ICLR 2026  
**arXiv**: [2510.00685](https://arxiv.org/abs/2510.00685)  
**代码**: 待确认  
**领域**: LLM预训练  
**关键词**: multi-agent systems, self-organization, Shapley value, communication graph, DAG, LLM collaboration

## 一句话总结

提出 SelfOrg 框架，基于 Agent 响应的语义相似度和 Shapley 值贡献估计，动态构建有向无环通讯图（DAG），实现多 Agent 系统的自组织协作。在弱模型场景下优势尤为显著。

## 研究背景与动机

基于 LLM 的多 Agent 系统（MAS）理论上能解决单个 LLM 无法处理的任务，但协作效果高度依赖通讯拓扑结构。现有方法的核心问题：

**固定拓扑**（链、树、全连接图）：无法适应不同任务和实例

**可优化拓扑**（GPTSwarm、AgentPrune）：需要策略梯度或掩码训练，开销大

**外部 LLM 裁判**（DyLAN）：引入额外 LLM 评估开销

**预训练图生成器**（G-Designer、MAS-GPT）：需要额外训练

本文的关键洞察是：由于 LLM 本质上是随机的，同一个 Agent 对同一问题的不同运行可能产生完全不同的答案。因此，**通讯结构应该基于当前响应状态动态决定**，而不是基于任务类型或问题本身。特别是在弱模型场景下，编排系统的价值在于放大稀少的正确响应并抑制噪声。

## 方法详解

### 整体框架

SelfOrg 包含四个阶段（如 Algorithm 1）：

1. **去中心化初始化**（$t=0$）：每个 Agent 独立生成响应 $\mathcal{R}_n^{(0)}$
2. **贡献估计**：基于 Shapley 值近似计算每个 Agent 的贡献
3. **通讯图构建**：形成 DAG，高贡献 Agent 在上游
4. **响应传播与聚合**：通过 DAG 传播信息，最终选择最接近加权质心的响应

### 关键设计

**Shapley 值近似**：真实 Shapley 值需要 $2^N$ 次评估，不可行。本文用余弦相似度近似：

$$\phi_n \approx \psi_n := \cos(\mathbf{r}_n, \mathbf{r}_{\text{avg}})$$

其中 $\mathbf{r}_n = f(\mathcal{R}_n)$ 是用轻量嵌入模型（如 all-MiniLM-L6）得到的响应嵌入，$\mathbf{r}_{\text{avg}}$ 是所有响应的平均嵌入。复杂度从指数级降到线性。

**近似质量保证**（Theorem 1）：当嵌入范数相等且内积有下界时，近似误差有上界 $I\Gamma^2$。Corollary 1 进一步保证：当两个 Agent 的贡献差距足够大时，排序是稳定的。

**DAG 构建规则**：
- 边 $e_{m \to n}$ 激活条件：$\cos(\mathbf{r}_n, \mathbf{r}_m) \geq \tau$ 且 $\psi_m > \psi_n$
- 检测并消除环：删除环中最弱 Agent 指向最强 Agent 的边
- 拓扑排序：按贡献值打破平局

**响应聚合**：使用贡献加权质心：

$$\mathbf{r}_{\text{centroid}}^{(T)} = \frac{\sum_{n=1}^N \psi_n^{(T)} \mathbf{r}_n^{(T)}}{\sum_{n=1}^N \psi_n^{(T)}}$$

最终选择最接近质心的响应：$n_\star = \arg\max_{n} \cos(\mathbf{r}_n^{(T)}, \mathbf{r}_{\text{centroid}}^{(T)})$

### 理论分析

**正确性放大机制**基于两个关键引理：

**Lemma 1（一致性集中）**：对两个独立 Agent，正确答案一致的概率 $\Pr[X_c] = p^2$ 大于任何错误答案一致的概率之和 $\sum_k p_k^2$，前提是错误答案足够分散（实验验证了这一点：正确答案在 100 次运行中反复出现，而错误答案高度分散）。

**Lemma 2（贡献主导）**：在正确答案嵌入紧密聚类、错误答案嵌入分散的假设下（Assumption 1），正确 Agent 的贡献值严格高于错误 Agent。

## 实验关键数据

### 主实验

**弱模型场景（Qwen-2.5-1.5B）**：

| 方法 | MATH | GSM8K | AQUA | GSM-H | MMLU | MMLU-P | AIME | AVG | AVG-R |
|------|------|-------|------|-------|------|--------|------|-----|-------|
| Single | 49.20 | 70.40 | 51.18 | 36.20 | 49.60 | 28.80 | 3.33 | 41.24 | 2.57 |
| DyLAN | 49.80 | 67.80 | 51.18 | 27.20 | 50.00 | 15.40 | 3.33 | 37.82 | 4.00 |
| AgentVerse | 45.20 | 69.00 | 50.39 | 27.80 | 38.20 | 24.00 | 0.00 | 36.37 | 4.86 |
| AutoGen | 11.60 | 69.40 | 28.74 | 5.40 | 12.20 | 5.20 | 0.00 | 18.93 | 6.06 |
| **SelfOrg** | **52.40** | **74.60** | **58.27** | **38.00** | **53.80** | **31.60** | **6.67** | **45.05** | **1.00** |

SelfOrg 比最强单 Agent 基线提升约 **+4 个百分点**，是唯一排名稳定第一的方法。

**强模型场景（LLaMA-3.3-70B）**：

| 方法 | MATH | GSM8K | AQUA | MMLU | GPQA | AIME | AVG | AVG-R |
|------|------|-------|------|------|------|------|-----|-------|
| CoT | 75.00 | 95.80 | 79.92 | 85.20 | 56.70 | 26.67 | 68.46 | 2.50 |
| MacNet | 74.80 | 96.00 | 79.13 | 83.00 | 58.26 | 26.67 | 67.31 | 3.63 |
| **SelfOrg** | **79.80** | **96.60** | **81.10** | 85.00 | **59.82** | **30.00** | **70.19** | **1.25** |

### 消融实验

**扩展性分析（Qwen-2.5-X）**：

| 模型规模 | AQUA Single | AQUA SelfOrg | Δ | MMLU-P Single | MMLU-P SelfOrg | Δ |
|---------|-------------|-------------|---|--------------|----------------|---|
| 1.5B | 51.18 | 58.27 | +7.09 | 28.80 | 31.60 | +2.80 |
| 3B | 65.35 | 73.62 | +8.27 | 42.60 | 46.20 | +3.60 |
| 7B | 73.62 | 78.35 | +4.73 | 53.20 | 56.40 | +3.20 |
| 72B | 81.10 | 80.71 | -0.39 | 70.60 | 71.20 | +0.60 |

增益在弱/中等模型最大，72B 模型时几乎消失（AQUA 甚至微降），符合理论预期。

### 关键发现

1. **弱模型收益最大**：现有 MAS 基线在 1.5B 模型上全部 "崩溃"（部分甚至低于单 Agent），SelfOrg 是唯一显著提升的方法
2. **异构 Agent 有效**：混合 Qwen/Falcon/LLaMA/Mistral 4 种 7B 模型，SelfOrg 将混合池的随机基线从 53.94 提升到 66.14（AQUA-RAT）
3. **贡献排名合理**：强模型（Qwen、Falcon）一致获得高排名，弱模型（Mistral）被降级
4. **实践中 2 轮协作通常足够**：第一轮探索，第二轮巩固

## 亮点与洞察

1. **响应条件化 > 任务条件化**：打破了"每个任务类型有一个最优拓扑"的假设，改为基于当前响应动态构建拓扑，这是更合理的建模
2. **无外部裁判、无预训练、无 RL**：使用轻量嵌入模型（6 层 MiniLM）替代昂贵的 LLM 裁判，大幅降低开销
3. **理论与实验高度统一**：概率模型准确预测了正确答案聚类、错误答案分散的实验现象
4. **Shapley 值的巧妙近似**：从指数复杂度降到线性，同时保证排序稳定性
5. **弱模型场景的实际价值**：当使用成本较低的小模型时，SelfOrg 提供了显著且稳定的收益

## 局限性

1. 依赖嵌入模型的质量——如果嵌入无法区分正确和错误响应的语义差异，贡献估计将失效
2. "多数即正确"的隐含假设——如果大多数 Agent 都错误且错误一致，SelfOrg 可能放大错误
3. 72B 规模模型收益消失，说明方法主要适用于中小规模 LLM
4. 评估局限于推理类 Benchmark，开放式生成任务的效果未知

## 相关工作与启发

- 与 GPTSwarm（参数化拓扑优化）和 G-Designer（预训练图生成器）相比，SelfOrg 是零开销的在线方法
- Shapley 值在联邦学习中的应用被成功迁移到 MAS 场景
- 启发：可以将此类自组织机制与专家混合（MoE）结合，实现推理时的动态专家路由
- DAG 构建中的"贡献主导信息流"原则可推广到更大规模的 Agent 编排系统

## 评分

- **创新性**: ⭐⭐⭐⭐ — 响应条件化拓扑构建是重要概念贡献，理论分析扎实
- **实验充分性**: ⭐⭐⭐⭐⭐ — 7 个 Benchmark、多种骨干模型（1.5B~72B）、异构设置、扩展性分析
- **实用性**: ⭐⭐⭐⭐ — 方法轻量无需训练，但需多次 LLM 调用
- **写作质量**: ⭐⭐⭐⭐ — 理论推导清晰，实验全面
- **综合评分**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation](../../ACL2025/llm_pretraining/synthesizing_post-training_data_for_llms_through_multi-agent_simulation.md)
- [\[ACL 2026\] Demystifying Data Organization for Enhanced LLM Training](../../ACL2026/llm_pretraining/demystifying_data_organization_for_enhanced_llm_training.md)
- [\[AAAI 2026\] ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences](../../AAAI2026/llm_pretraining/elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)
- [\[ICLR 2026\] CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images](chammi-75_pre-training_multi-channel_models_with_heterogeneous_microscopy_images.md)
- [\[ACL 2026\] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](../../ACL2026/llm_pretraining/data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)

</div>

<!-- RELATED:END -->
