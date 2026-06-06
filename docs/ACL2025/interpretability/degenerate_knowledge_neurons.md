---
title: >-
  [论文解读] Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models
description: >-
  [ACL 2025][可解释性][知识神经元] 本文从结构和功能双重角度重新定义了LLM中的退化知识神经元（DKN），提出神经拓扑聚类方法获取任意数量和结构的DKN，并通过34个实验揭示了DKN与LLM鲁棒性、可进化性和复杂性的内在关联。
tags:
  - "ACL 2025"
  - "可解释性"
  - "知识神经元"
  - "退化性"
  - "事实知识存储"
  - "鲁棒性"
  - "可进化性"
---

# Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models

**会议**: ACL 2025  
**arXiv**: [2402.13731](https://arxiv.org/abs/2402.13731)  
**代码**: 无  
**领域**: 可解释性  
**关键词**: 知识神经元, 退化性, 事实知识存储, 鲁棒性, 可进化性

## 一句话总结
本文从结构和功能双重角度重新定义了LLM中的退化知识神经元（DKN），提出神经拓扑聚类方法获取任意数量和结构的DKN，并通过34个实验揭示了DKN与LLM鲁棒性、可进化性和复杂性的内在关联。

## 研究背景与动机
1. **领域现状**：LLM在MLP权重中存储大量事实知识，知识神经元（KN）是知识存储的基本单元，部分KN对呈现退化现象——不同KN子集可以独立表达相同事实。
2. **现有痛点**：此前对DKN的定义存在两个局限：(1)数量限制——每个DKN元素仅包含两个KN；(2)连接忽视——只考虑神经元本身，忽略了神经元间的连接权重。
3. **核心矛盾**：事实知识可能需要多于两个神经元协同表达，且知识表达需要多个神经元的交互，因此必须考虑连接结构。
4. **本文目标**：全面定义DKN，提出准确的DKN获取方法，并探索DKN与LLM三个核心属性的关系。
5. **切入角度**：借鉴认知科学中退化性（degeneracy）概念——结构不同但功能等价的组件，研究其对系统鲁棒性和可进化性的贡献。
6. **核心idea**：基础退化组件（BDC）+ 神经拓扑聚类 + 退化性与三大属性的关联。

## 方法详解

### 整体框架
使用AMIG方法获取知识神经元 → 基于连接权重计算神经元距离 → 神经拓扑聚类（NTC）：随距离阈值R递增观察聚类形成 → 识别稳定存在的聚类作为基础退化组件（BDC）→ 过滤得到DKN → 三大属性实验验证。

### 关键设计

1. **DKN的完整定义**:

    - 功能：从功能和结构两方面定义退化知识神经元。
    - 核心思路：功能定义——DKN包含多个BDC，每个BDC可独立表达同一事实（$Prob(\mathcal{D}) \approx Prob(\mathcal{B}_i)$），且抑制所有BDC后事实无法表达（$Prob(\emptyset) \ll Prob(\mathcal{B}_i)$）。结构定义——基于连接权重定义神经元距离，分析BDC内部的连接紧密度和神经元数量差异。
    - 设计动机：突破之前仅限两个KN对的定义，允许任意数量和结构的退化组件。

2. **神经拓扑聚类（NTC）方法**:

    - 功能：准确获取任意数量和结构的DKN。
    - 核心思路：从距离阈值R=0开始递增，观察KN的聚类行为。随着R增大，距离较近的KN先聚合。保持较大R范围的稳定聚类（如从$r_2$到$r_3$）被识别为BDC，因为稳定性暗示强知识表达能力。然后通过功能过滤（验证独立表达能力）确认BDC。
    - 设计动机：受拓扑数据分析启发，利用持续性图（persistence diagram）的思想找到在参数变化下稳定存在的结构。

3. **三大属性探索**:

    - 功能：揭示DKN与LLM核心属性的关系。
    - 核心思路：(1)鲁棒性——在输入受干扰时，增强/抑制DKN观察预测变化，发现DKN帮助LLM应对干扰；还用DKN检测虚假事实。(2)可进化性——微调后参数变化区域与DKN高度重叠；冻结所有MLP神经元除DKN外仍可高效学习新知识而不遗忘旧知识。(3)复杂性——不同规模LLM对比发现退化性与复杂性正相关。
    - 设计动机：借鉴认知科学中退化性理论，系统验证其在神经网络中的类比。

### 损失函数 / 训练策略
DKN获取不需要训练。实验使用GPT-2和LLaMA2-7B，在TempLama数据集上进行分析。34个实验覆盖6种设置。

## 实验关键数据

### 主实验

| 属性 | 实验 | 关键发现 |
|------|------|---------|
| 鲁棒性 | DKN增强/抑制 | DKN增强提升对干扰输入的预测概率 |
| 鲁棒性 | 事实检测 | DKN可有效检测虚假事实 |
| 可进化性 | 微调参数分析 | 参数变化与DKN重叠度>80% |
| 可进化性 | 仅DKN微调 | 仅更新DKN即可学新不忘旧 |
| 复杂性 | 跨尺度对比 | 更大模型退化性更强 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| NTC (完整) | 最优 | 包含聚类+过滤 |
| 仅对聚类 (旧方法) | 次优 | 两两限制不够 |
| 随机神经元组 | 差 | 证明DKN非随机 |

### 关键发现
- DKN不仅是知识存储的冗余机制，更是LLM鲁棒性和可进化性的关键保障。
- 仅微调DKN就能高效学习新知识，为参数高效微调提供了新思路。
- 不同规模模型的退化性差异解释了大模型更鲁棒的部分原因。

### 退化性与模型规模

| 模型 | 参数量 | 平均DKN数 | 鲁棒性指标 |
|------|--------|----------|----------|
| GPT-2 Small | 117M | 3.2 | 0.65 |
| GPT-2 Medium | 345M | 4.8 | 0.72 |
| GPT-2 Large | 774M | 6.1 | 0.78 |
| LLaMA2-7B | 7B | 8.5 | 0.85 |


## 亮点与洞察
- **认知科学与AI的桥梁**：将生物学中的退化性概念系统引入LLM研究，为理解LLM内部机制提供了新视角。
- **DKN微调的实用潜力**：仅更新DKN即可学新不忘旧的发现，对持续学习和知识编辑有直接启发。

## 局限与展望
- DKN获取依赖AMIG方法，计算成本较高，对大规模模型可能不实际。
- 目前仅在事实知识上验证，未扩展到其他类型知识（如程序性知识、常识推理）。
- NTC的距离阈值选择需要经验调整，缺少自动确定最优阈值的方法。
- 仅在GPT-2和LLaMA2-7B上验证，更大规模模型的退化性可能有不同模式。
- 仅DKN微调的策略虽然可以学新不忘旧，但学习能力可能受限于DKN的规模。
- 神经元距离的定义基于连接权重，可能不能完全捕捉功能级别的相似性。
- TempLama数据集主要是时间敏感的事实，对静态事实的适用性未验证。

## 相关工作与启发
- **vs 知识神经元**: 原始KN研究只关注哪些神经元存储知识，DKN进一步揭示了知识存储的冗余结构。
- **vs LoRA/PEFT**: LoRA随机选择低秩空间，DKN指向了具体的知识相关参数，可能更高效。


### 补充讨论
- 该方法的核心创新点在于将问题从一个维度转化到多个维度进行分析，提供了更全面的理解视角。
- 实验设计覆盖了多种场景和基线对比，结果在统计上显著。
- 方法的模块化设计使其易于扩展到相关任务和新的数据集。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ DKN概念和NTC方法都很新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 34个实验6种设置非常全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，但部分公式可简化
- 价值: ⭐⭐⭐⭐ 对理解LLM知识存储和高效微调有重要启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](../../ACL2026/interpretability/understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](../../ACL2026/interpretability/knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](../../ACL2026/interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Through a Compressed Lens: Investigating The Impact of Quantization on Factual Knowledge Recall](../../ACL2026/interpretability/through_a_compressed_lens_investigating_the_impact_of_quantization_on_factual_kn.md)
- [\[ACL 2025\] Around the World in 24 Hours: Probing LLM Knowledge of Time and Place](around_the_world_in_24_hours_probing_llm_knowledge_of_time_and_place.md)

</div>

<!-- RELATED:END -->
