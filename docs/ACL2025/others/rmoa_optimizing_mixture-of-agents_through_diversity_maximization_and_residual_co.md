---
title: >-
  [论文解读] RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation
description: >-
  [ACL 2025][Mixture-of-Agents] 受 ResNet 残差学习启发，提出 RMoA 框架，通过嵌入式多样性贪心选择、残差提取/聚合智能体和自适应终止机制来优化多智能体协作架构，在降低计算开销的同时实现 SOTA 性能。 基于大语言模型的多智能体系统（MAS）在多种任务上展现了强大能力…
tags:
  - "ACL 2025"
  - "Mixture-of-Agents"
  - "残差连接"
  - "多样性选择"
  - "自适应终止"
  - "多智能体协作"
---

# RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation

**会议**: ACL 2025  
**arXiv**: [2505.24442](https://arxiv.org/abs/2505.24442)  
**代码**: 有 ([https://github.com/mindhunter01/RMoA](https://github.com/mindhunter01/RMoA))  
**领域**: 其他  
**关键词**: Mixture-of-Agents, 残差连接, 多样性选择, 自适应终止, 多智能体协作

## 一句话总结

受 ResNet 残差学习启发，提出 RMoA 框架，通过嵌入式多样性贪心选择、残差提取/聚合智能体和自适应终止机制来优化多智能体协作架构，在降低计算开销的同时实现 SOTA 性能。

## 研究背景与动机

基于大语言模型的多智能体系统（MAS）在多种任务上展现了强大能力，其中 Mixture-of-Agents（MoA）架构通过多层 Agent 并行处理和聚合来提升性能。然而，现有的 MoA 和 SMoA 方法存在三个核心问题：

**计算开销高**：每层所有 Agent 的响应都被拼接传递给下一层，token 数量急剧膨胀

**信息损失**：多层迭代聚合过程中关键信息逐渐丢失，导致深层性能退化

**鲁棒性不足**：依赖 Judge 模型评估响应质量，但 Judge 模型本身的区分能力有限

SMoA 虽然引入了 Judge 模型来筛选响应以降低 token 数量，但 Judge 模型的质量评估不够可靠。更重要的是，随着层数增加，聚合过程中的信息损失问题始终未被有效解决。

作者的核心洞察是：**可以借鉴 ResNet 的残差连接思想**——保留层间差异信息以缓解信息丢失；**用嵌入相似度代替 Judge 模型**来选择多样化响应以保证信息异质性；**引入自适应终止**以避免不必要的计算。

## 方法详解

### 整体框架

RMoA 在标准 MoA 架构基础上引入三个核心改进：
1. **贪心多样性嵌入选择（Greedy Diversity Embedding Selection）**：用 embedding 模型代替 Judge 模型，从 N 个响应中贪心选择 K 个最多样化的响应
2. **残差智能体（Residual Agent）**：包含残差提取智能体和残差聚合智能体，捕获层间差异信息
3. **自适应终止机制（Adaptive Termination）**：根据残差收敛情况动态决定是否继续迭代

### 关键设计

1. **贪心多样性嵌入选择**

    - 功能：从第 l 层 N 个 Agent 的响应中选出 K 个语义最多样的响应
    - 核心思路：
        - 用 embedding 模型将所有响应转为向量表示
        - 构建余弦相似度矩阵 S ∈ R^{N×N}
        - 初始化：选择全局平均相似度最低的响应作为第一个
        - 迭代选择：每轮选择与已选集合最大相似度最小的响应（min-max策略）
        - 重复直到选满 K 个
    - 设计动机：解决"注意力分散"问题——太多语义重叠的响应会增加 self-attention 的认知负担，贪心选择确保信息异质性最大化

2. **残差提取智能体（Residual Extraction Agent）**

    - 功能：捕获连续两层响应之间的关键差异
    - 核心思路：
        - 将第 l 层的 K 个候选响应与第 l-1 层的 K 个历史响应拼接
        - 使用预定义 prompt 让残差提取 Agent 识别有意义的变化
        - 提取的残差 ΔR_l 与上一层响应拼接，作为下一层的参考输入
    - 设计动机：直接参考 ResNet——通过保留层间增量信息来缓解深层迭代中的信息退化

3. **残差聚合智能体（Residual Aggregation Agent）**

    - 功能：在最终层整合上一层参考响应和当前层残差，生成最终输出
    - 核心思路：综合考虑上一层的完整响应和当前层的残差变化
    - 设计动机：确保最终输出既保留了长期信息又整合了短期改进

4. **自适应终止机制**

    - 功能：动态决定何时停止迭代
    - 核心思路：如果连续 m 层都没有检测到有意义的残差（残差为"无变化"或"无更新"），提前终止
    - 设计动机：避免不必要的推理层带来的额外计算开销和潜在的幻觉累积

5. **角色扮演多样性**

    - 每个 Agent 被赋予不同的角色扮演人设，以增强输出的多样性和创造性

### 损失函数 / 训练策略

RMoA 是一个纯推理框架，无需训练。关键超参数：
- 层数 L：默认 6 层
- 多样性选择数 K：默认 3（实验验证最优）
- 自适应终止的连续无残差层数 m
- embedding 模型：BGE-m3（对模型选择不敏感，±0.6%）

## 实验关键数据

### 主实验（不同模型在四个基准上的表现）

| 模型 | 方法 | AlpacaEval 2.0 | MATH | CRUX | MMLU-r | 平均 |
|------|------|---------------|------|------|--------|------|
| Qwen2.5-7B | 基线 | 37.94 | 74.94 | 57.31 | 69.90 | 60.02 |
| | +MoA | 31.77 | 75.28 | 56.81 | 62.70 | 56.64↓ |
| | +SMoA | 40.79 | 76.98 | 59.93 | 72.00 | 62.43↑ |
| | **+RMoA** | **41.01** | **77.20** | **61.00** | **71.80** | **62.75↑4.55%** |
| Gemma2-9B | 基线 | 45.15 | 36.64 | 47.50 | 63.90 | 48.30 |
| | **+RMoA** | **45.61** | **50.44** | **50.50** | **66.10** | **53.16↑10.06%** |
| Llama3.1-8B | 基线 | 22.93 | 48.18 | 40.62 | 58.60 | 42.58 |
| | **+RMoA** | **32.86** | **52.10** | **42.65** | **61.63** | **47.41↑11.10%** |
| GPT-4o | 基线 | 55.18 | 76.60 | 75.80 | 83.73 | 72.83 |
| | **+RMoA** | **63.29** | **81.16** | **87.37** | **86.67** | **79.62↑9.32%** |

### 消融实验与扩展分析

| 分析维度 | 发现 |
|---------|------|
| K 值选择 (K=2,3,4,5) | K=3 为最优平衡点，K>3 性能下降 |
| 自适应终止 vs 无终止 | AT 将幻觉率从平均 5.2% 降至 1.6% |
| Embedding 模型 | BGE-m3/SGPT/E5-large 差异 <0.6%，方法对 embedding 选择鲁棒 |
| 层数扩展 (1-6层) | 性能随层数稳步提升且不退化 |
| 大模型 (72B/DeepSeek) | Qwen2.5-72B: 80.00→87.80 (+7.8%); DeepSeek-R1: 78.04→82.92 (+4.88%) |
| 终止判断方法 | LLM 判断(80.2%) > 方差指标(79.4%) > 相似度阈值(78.8%) |

### 关键发现

1. RMoA 在所有模型和基准上一致优于 MoA 和 SMoA
2. 原始 MoA 在小模型上可能导致性能退化（如 Qwen2.5-7B 的 MMLU 下降 7.2%），RMoA 避免了这一问题
3. 残差机制有效缓解了信息损失，使深层架构始终带来性能提升
4. 自适应终止显著降低了幻觉率（从 ~5% 降至 ~1.6%）
5. 框架可与更强大的基座模型（如 DeepSeek-R1）叠加使用

## 亮点与洞察

- ResNet 残差思想的跨域应用十分巧妙：将深度学习中解决梯度消失/信息退化的经典方案应用到多智能体协作场景
- 用 embedding 多样性替代 Judge 模型评估是一个既简洁又有效的设计选择
- 自适应终止机制同时解决了效率和幻觉两个问题
- 框架的模型无关性（从 7B 到 GPT-4o 均有效）体现了良好的通用性

## 局限与展望

1. **计算开销仍较高**：虽然相比 MoA 有所优化，但多层多 Agent 的推理成本仍然显著
2. **残差提取依赖 prompt 工程**：残差提取智能体的效果受 prompt 设计影响
3. **角色扮演的效果未充分分析**：不同角色设计策略的影响缺乏系统消融
4. **缺乏与 self-consistency 等方法的直接比较**
5. **实验场景可扩展**：未涉及开放式生成、对话等场景

## 相关工作与启发

- MoA 和 SMoA 是直接基线，RMoA 在其基础上解决了信息损失和效率问题
- ResNet 的残差连接启发了跨模态的信息保留策略
- 可启发在 RAG、多模型投票、ensemble 等场景中引入残差信息保留机制
- 嵌入多样性选择方法可推广到检索增强生成中的文档选择

## 评分

- **新颖性**: ⭐⭐⭐⭐ — ResNet 残差思想在多智能体协作中的应用是新颖的，嵌入多样性选择替代 Judge 模型也是有价值的创新
- **实验充分度**: ⭐⭐⭐⭐ — 四个多样化基准、四种模型规模、丰富的消融实验和扩展分析，覆盖面广
- **写作质量**: ⭐⭐⭐ — 整体结构清晰，但部分公式符号较密集，可读性一般
- **价值**: ⭐⭐⭐⭐ — 为多智能体协作提供了一个即插即用的通用优化框架，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Randomized Dimensionality Reduction for Euclidean Maximization and Diversity Measures](../../ICML2025/others/randomized_dimensionality_reduction_for_euclidean_maximization_and_diversity_mea.md)
- [\[ACL 2025\] Value Residual Learning](value_residual_learning.md)
- [\[ACL 2025\] A New Formulation of Zipf's Meaning-Frequency Law through Contextual Diversity](a_new_formulation_of_zipfs_meaning-frequency_law_through_contextual_diversity.md)
- [\[ACL 2025\] ACT: Knowledgeable Agents to Design and Perform Complex Tasks](act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)
- [\[ACL 2025\] Optimizing Decomposition for Optimal Claim Verification](optimizing_decomposition_for_optimal_claim_verification.md)

</div>

<!-- RELATED:END -->
