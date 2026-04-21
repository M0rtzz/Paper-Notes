---
title: >-
  [论文解读] Mitigating Negative Interference in Multilingual Sequential Knowledge Editing through Null-Space Constraints
description: >-
  [ACL 2025][多语言知识编辑] 本文提出 LangEdit 框架，通过将每种语言的参数更新投影到先前编辑语言的零空间上，实现多语言序列知识编辑中不同语言更新之间的数学隔离，有效缓解负干扰并保持多语言泛化能力。
tags:
  - ACL 2025
  - 多语言知识编辑
  - 零空间投影
  - 负干扰
  - 序列编辑
  - LLM
---

# Mitigating Negative Interference in Multilingual Sequential Knowledge Editing through Null-Space Constraints

**会议**: ACL 2025  
**arXiv**: [2506.10800](https://arxiv.org/abs/2506.10800)  
**代码**: [有 (GitHub)](https://github.com/VRCMF/LangEdit.git)  
**领域**: NLP / 知识编辑 / 多语言LLM  
**关键词**: 多语言知识编辑, 零空间投影, 负干扰, 序列编辑, LLM

## 一句话总结

本文提出 LangEdit 框架，通过将每种语言的参数更新投影到先前编辑语言的零空间上，实现多语言序列知识编辑中不同语言更新之间的数学隔离，有效缓解负干扰并保持多语言泛化能力。

## 研究背景与动机

大型语言模型（LLM）在编码和检索事实知识方面表现出色，知识编辑（knowledge editing）是一种高效更新知识而无需完全重训的方法。然而在**多语言场景**下，高效更新知识面临极大挑战：对一种语言进行知识编辑可能损害模型在其他语言上的表现。

为每种语言单独维护编辑模型成本过高。更现实的方案是将所有语言的知识更新整合到一个统一模型中。但实验表明，跨语言的序列编辑会导致**破坏性参数干扰**（destructive parameter interference），即"**负干扰**"——在一种语言上的编辑会降低先前编辑语言的准确性和模型的多语言泛化能力。

现有的单语知识编辑方法（如 ROME、MEMIT、AlphaEdit）在处理这一新场景时表现不佳。特别是，AlphaEdit 虽然也使用零空间投影，但它使用的是**静态零空间**（基于原始预训练知识），而不同语言之间的动态交互需要不同的投影空间。

## 方法详解

### 整体框架

LangEdit 在 LLM 的 MLP 层（具体是 W_out 矩阵）上进行知识编辑。将多语言数据流 {L_1, L_2, ..., L_T} 按时间步依次注入，每一步对应某种语言的知识更新。核心思想是：当编辑语言 j 的知识时，将参数更新投影到先前所有已编辑语言的零空间中，确保新编辑不破坏旧知识。

### 关键设计

1. **MLP 作为知识存储器**：将知识事实 (subject, relation, object) 编码为 MLP 层的 key-value 对。W_in 层的输出对应 key（subject + relation），W_out 层的输出对应 value（object）。知识编辑的目标是优化 W_out。

2. **零空间投影**：

    - 对先前已编辑的所有 key 矩阵计算非中心协方差矩阵 K̄_{t-1}
    - 通过递推公式增量更新协方差矩阵（避免存储完整历史）
    - 对协方差矩阵做 SVD 分解，保留零特征值对应的特征向量构造投影矩阵 P_{t-1}
    - 将新参数更新 ΔW_t 投影到 P_{t-1} 表示的零空间中

3. **优化目标（闭合解）**：

    - 目标函数包含两项：正则化项（限制参数扰动大小）和编辑精度项（确保编辑后 key-value 匹配）
    - 闭合解：ΔW_t = R_t · K_t^T · P_{t-1} · (K_t · K_t^T · P_{t-1} + I)^{-1}
    - 其中 R_t = V_t - W_{t-1} · K_t 是残差

4. **与 AlphaEdit 的核心区别**：

    - AlphaEdit：静态零空间（仅基于预训练知识 K_0），不区分语言
    - LangEdit：动态零空间（每步 t 的投影矩阵 P_{t-1} 不同），语言特定的隔离

### 训练策略

- 每步编辑 100 个样本
- K_0 从 Wikipedia 随机采样 100,000 个三元组计算
- 对 GPT-J-6B / Llama3-8B / Qwen2.5-7B 分别选择关键层进行编辑
- 关键层通过 causal tracing 技术确定

## 实验关键数据

### 主实验：多语言序列知识编辑（mzsre 数据集，6 语言 × 400 样本 = 2400 编辑）

| 模型 | 方法 | Efficacy↑ | Generality↑ | Specificity↑ | XTREME F1↑ |
|------|------|-----------|-------------|-------------|-----------|
| Llama3-8B | Pre-edited | 31.15 | 31.01 | 31.93 | 69.30 |
| Llama3-8B | MEMIT | 1.45 | 1.46 | 0.67 | 4.54 |
| Llama3-8B | AlphaEdit | 80.34 | 75.84 | 30.91 | 60.59 |
| Llama3-8B | **LangEdit** | **82.54** | **77.53** | **31.90** | **66.24** |
| Qwen2.5-7B | AlphaEdit | 93.50 | 87.18 | 42.58 | 73.01 |
| Qwen2.5-7B | **LangEdit** | **93.90** | **87.02** | **42.64** | **74.06** |
| GPT-J-6B | AlphaEdit | 83.59 | 78.34 | 26.55 | 36.74 |
| GPT-J-6B | **LangEdit** | **84.27** | **79.74** | **27.23** | **38.59** |

### 消融：负干扰量化（Llama3-8B）

| 方法 | Efficacy 差距 vs 单语 | F1 差距 vs 单语 |
|------|---------------------|----------------|
| AlphaEdit (multi) | +0.10 ~ +3.27 | +3.20 ~ +15.19 |
| **LangEdit** | 部分优于单语 | **+0.20 ~ +9.29** |

LangEdit 在英、德、荷、法四种语言上 Efficacy 超越了单语 AlphaEdit。

### 关键发现

1. **LangEdit 全面优于 SOTA**：在三种模型架构和两个数据集上，Efficacy 平均提升 +2.20，XTREME F1 平均提升 +5.65（Llama3-8B 最大提升）。

2. **ROME、FT 在序列多语言编辑下灾难性崩溃**：大量编辑后 XTREME F1 降至个位数，说明现有方法完全无法处理此场景。

3. **编辑效果与预训练数据量负相关**：英语（预训练数据多）提升较小，西班牙语（预训练数据少）提升高达 +9.00 F1。

4. **多语言编辑可增强泛化能力**：对 GPT-J-6B 和 Qwen2.5-7B，LangEdit 编辑后的 XTREME 分数甚至超过未编辑模型，说明注入多语言知识本身就有正向迁移效应。

5. **零空间投影有效隔离语言间干扰**：不使用零空间投影时，多语言编辑显著劣于单语编辑；使用后差距大幅缩小甚至反超。

## 亮点与洞察

- **新任务定义**：首次形式化"多语言序列知识编辑"任务，为多语言 LLM 的维护提供了新范式。
- **数学保证的语言隔离**：零空间投影从理论上保证不同语言的参数更新不相互覆盖。
- **增量协方差更新**：避免存储和重计算完整的历史 key 矩阵，计算效率高。
- **闭合解**：无需迭代优化，计算一步到位。
- **实用场景**：对多语言信息检索、多语言 LLM 的事实更新等应用有直接价值。

## 局限与展望

- 零空间维度随编辑量增加而缩小，当编辑量非常大时可能出现"零空间耗尽"
- 仅在 MLP 层的 W_out 上编辑，注意力层中的知识未被编辑
- 关键层的选择依赖 causal tracing，不同模型和语言可能需要不同配置
- 实验仅覆盖 6 种语言，对更多样的语言组合（如低资源语言）的效果未知
- 编辑的知识类型限于事实三元组，更复杂的知识类型未涉及

## 相关工作与启发

- **ROME** (Meng et al., 2022)：开创性的定位-编辑框架，通过 rank-one 更新编辑单层 MLP
- **MEMIT** (Meng et al., 2023)：扩展 ROME 到多层并行编辑
- **RECT** (Gu et al., 2024)：通过相对权重变化控制更新幅度
- **PRUNE** (Ma et al., 2025)：通过条件数约束限制权重退化
- **AlphaEdit** (Fang et al., 2025)：单语零空间投影编辑，本文的直接基线
- 持续学习中的零空间方法 (Wang et al., 2021)：本文借鉴的核心技术

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 多语言序列编辑是新任务，动态零空间投影是对 AlphaEdit 的有意义扩展
- **实验充分度**: ⭐⭐⭐⭐⭐ — 3 模型 × 2 数据集 × 6 语言 × 4 下游任务 × 多基线 × 逐语言分析 × 负干扰量化
- **写作质量**: ⭐⭐⭐⭐ — 问题动机清晰，数学推导完整，图表信息量大
- **价值**: ⭐⭐⭐⭐ — 为多语言 LLM 知识维护提供了实用框架，有明确的工程价值

<!-- RELATED:START -->

## 相关论文

- [Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning](memorizing_is_not_enough_deep_knowledge_injection_through_reasoning.md)
- [ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains](chainedit_propagating_ripple_effects_in_llm.md)
- [ToxEdit: Adaptive Detoxification Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing](adaptive_detoxification_safeguarding_general_capabilities_of_llms_through_toxici.md)
- [Neuron-Level Sequential Editing for Large Language Models](neuron-level_sequential_editing_for_large_language_models.md)
- [Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)](../../AAAI2026/knowledge_editing/multiplicative_orthogonal_sequential_editing_for_language_models.md)

<!-- RELATED:END -->
