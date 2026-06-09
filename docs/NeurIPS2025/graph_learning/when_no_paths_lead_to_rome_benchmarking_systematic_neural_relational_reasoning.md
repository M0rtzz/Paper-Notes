---
title: >-
  [论文解读] When No Paths Lead to Rome: Benchmarking Systematic Neural Relational Reasoning
description: >-
  [NeurIPS 2025][图学习][关系推理] 提出NoRA benchmark，系统性地打破现有关系推理benchmark中"推理可归约为路径组合"的假设，引入非路径推理、歧义事实和多关系等挑战，揭示包括o3在内的所有现有模型在off-path推理上的根本缺陷。
tags:
  - "NeurIPS 2025"
  - "图学习"
  - "关系推理"
  - "benchmark"
  - "非路径推理"
  - "系统性泛化"
  - "图神经网络"
---

# When No Paths Lead to Rome: Benchmarking Systematic Neural Relational Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2510.23532](https://arxiv.org/abs/2510.23532)  
**代码**: [GitHub](https://github.com/axd353/WhenNoPathsLeadToRome/)  
**领域**: 图学习  
**关键词**: 关系推理, benchmark, 非路径推理, 系统性泛化, 图神经网络

## 一句话总结

提出NoRA benchmark，系统性地打破现有关系推理benchmark中"推理可归约为路径组合"的假设，引入非路径推理、歧义事实和多关系等挑战，揭示包括o3在内的所有现有模型在off-path推理上的根本缺陷。

## 研究背景与动机

**领域现状**：系统性关系推理是设计能以组合方式运用学到规则的模型的核心挑战。CLUTRR是最流行的benchmark，聚焦于家庭关系推理。SOTA方法包括CTP（可微逻辑编程）、R5/NCRL（路径组合）、Edge Transformer和EpiGNN等。

**现有痛点**：
   - CLUTRR的所有推理都可归约为**路径组合**：沿连接源与目标的单一路径组合关系即可得出答案
   - 许多SOTA模型的架构中硬编码了路径偏置（如R5, NCRL只能沿路径推理）
   - STaR虽需组合多条路径，但主要推理方式仍是路径型的

**核心矛盾**：现实世界的关系推理常需"绕道"——即利用不在源-目标路径上的信息来得出结论。例如：知道wes"无女儿"（off-path信息）才能推断ann是todd的"母方姨妈"。现有benchmark无法测试这种能力。

**本文目标**：构建一个系统性地挑战路径偏置的benchmark，精确量化非路径推理的难度，驱动更通用推理模型的研发。

**切入角度**：设计包含性别特定家庭关系、日常关系（同学、邻居等）和歧义事实的丰富规则集，使推理必须涉及路径外的节点和约束消歧。

**核心 idea**：NoRA通过引入off-path推理、歧义消解和多关系来打破路径组合假设，揭示系统性推理的真正瓶颈。

## 方法详解

### NoRA Benchmark设计

#### 三大创新特征

1. **非路径推理**：推理需"绕道"使用不在源-目标路径上的边。如图1所示，推断ann是todd的maternal aunt需通过off-path节点wes的"无女儿"属性。

2. **歧义事实**：使用ASP语法编码 $l\{r_1(x_1,y_1),\ldots,r_n(x_n,y_n)\}u$，表示其中 $l$ 到 $u$ 个事实为真。模型需通过约束消歧来确定哪些关系成立。

3. **多关系**：实体间可同时存在多种关系——层次型（aunt且maternal_aunt）或独立型（兄弟且同学）。

#### 难度度量

- **Reasoning Depth（深度）**：推导所需的最少规则应用步数
- **Reasoning Width（宽度）**：唯一推导路径数，量化歧义程度
- **Backtrack Load (BL)**：推理步数与涉及实体数的比值，衡量是否需在路径上来回推理
- **Off-Path Edge Count (OPEC)**：推导中使用的不在源-目标任何直接路径上的边数

#### 数据集划分

| 划分 | 深度 | 宽度 | BL | OPEC |
|------|------|------|-----|------|
| Train-a | ≤6 | ≤5 | ≤1.5 | ≤2 |
| Test-D | **>6** | ≤5 | ≤1.5 | ≤2 |
| Test-W | ≤6 | **>5** | ≤1.5 | ≤2 |
| Test-BL | ≤6 | ≤5 | **>1.5** | - |
| Test-OPEC | - | - | - | **≥3** |

### 推理框架

基于Answer Set Programming (ASP)：
- **World rules**: 确定性规则（如 living_in_same_place 传递性）+ 约束规则（未成年人不能是家长）
- **Answer set**: 故事的所有非矛盾推演结果
- **查询**: 给定故事 $\mathcal{S}$ 和实体对 $(x,y)$，预测所有可推得的关系 $\mathcal{R} = \bigcap\{\text{rels}(x,y,\mathcal{A}) | \mathcal{A}\text{ is answer set}\}$

### NoRA v1.1: 扩展版

使用递归子图扩展技术生成更高OPEC/深度/BL的测试实例，保证每个测试实例都可通过"拼接"训练实例得到，确保组合泛化的可行性。

## 实验关键数据

### 主实验：Exact-Match Accuracy

| 模型 | In-dist | Test-D | Test-W | Test-BL | Test-OPEC |
|------|---------|--------|--------|---------|-----------|
| ET (single-edge) | 0.885 | 0.741 | 0.703 | 0.245 | **0.060** |
| ET (multi-edge) | **0.900** | 0.493 | **0.790** | **0.785** | 0.037 |
| RAT (multi-edge) | 0.900 | 0.676 | 0.668 | 0.540 | 0.028 |
| EpiGNN-mul (BCE) | 0.520 | 0.604 | 0.491 | 0.156 | 0.009 |
| NBFNet (BCE) | 0.576 | 0.531 | 0.460 | 0.153 | 0.009 |
| R-GCN | 0.347 | 0.672 | 0.283 | 0.051 | 0.032 |

### 无歧义测试集

| 模型 | In-dist-na | D-na | BL-na | OPEC-na |
|------|------------|------|-------|---------|
| ET (single-edge) | 0.800 | 0.822 | 0.104 | 0.110 |
| ET (multi-edge) | 0.800 | 0.494 | 0.056 | 0.077 |
| EpiGNN-mul (BCE) | 0.539 | 0.716 | 0.027 | 0.045 |

### LRM评估（o3, 给定世界规则）

- OPEC=0时：o3近乎完美
- OPEC=3, depth=7时：o3准确率**降至0**
- o4-mini表现略差于o3

### 消融：控制变量后的ET性能

- 控制BL和宽度后，从depth 4（>0.8）到depth 6（0.2-0.6）性能急剧下降
- 证实单一维度的"容易"测试实例掩盖了真实困难

### 关键发现

1. **OPEC是最困难的维度**：所有模型在Test-OPEC上接近完全失败（<10%），包括明确给定规则的o3
2. **BL同样极具挑战**：无歧义设置下所有模型在BL-na上<11%
3. **歧义意外不难**（但有快捷方式）：Test-W的表现合理，但进一步分析发现模型在利用捷径
4. **GNN方法整体表现差**：even在分布内测试集上也远低于ET，因路径偏置太强
5. **ET是最佳但远不够好**：最佳OPEC准确率仅6%

## 亮点与洞察

1. **精确量化推理难度**：提出的四维度（Depth/Width/BL/OPEC）量化体系非常系统，可泛化到其他推理任务
2. **打破路径假设的实证力量**：所有模型（包括LRM o3）在off-path推理上的失败令人震撼
3. **benchmark生成的严谨性**：NoRA v1.1通过递归拼接保证测试的组合泛化可行性
4. **多维度独立控制**：每个测试集仅沿一个难度轴外推，避免混淆

## 局限与展望

1. 随机采样难以获得多个难度维度同时高的实例
2. Depth/BL等指标依赖知识库编码方式，存在灵敏度问题
3. 歧义并非真正挑战（因捷径），需改进歧义事实的生成机制
4. 未测试更多GNN变体（如高阶GNN、子图GNN）
5. 推理规则固定——更通用的设置应允许规则集变化

## 相关工作与启发

- **CLUTRR** [Sinha et al., 2019]：经典家庭关系推理benchmark，所有推理均为路径型
- **STaR** [Khalid & Schockaert, 2025]：时空推理benchmark，需组合多路径但仍主要是路径型
- **Edge Transformer** [Bergen et al., 2021]：三角注意力机制的Transformer，最通用的系统推理架构
- **EpiGNN** [Khalid & Schockaert, 2025]：支持多路径组合的GNN，在HetioNet上优于ET
- **Dziri et al. (2023)**：论证LLM在组合推理上的根本局限

## 评分

⭐⭐⭐⭐⭐

极其出色的benchmark工作。四维度难度量化精确且有洞察力。最核心贡献是实证表明所有现有模型——包括LRM——在非路径推理上本质性地失败，为新型推理架构的研发提供了清晰的方向。NoRA有潜力成为CLUTRR之后关系推理评估的新标准。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MedMKG: Benchmarking Medical Knowledge Exploitation with Multimodal Knowledge Graph](medmkg_benchmarking_medical_knowledge_exploitation_with_multimodal_knowledge_gra.md)
- [\[NeurIPS 2025\] Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking](diagnosing_and_addressing_pitfalls_in_kg-rag_datasets_toward_more_reliable_bench.md)
- [\[ICLR 2026\] Relatron: Automating Relational Machine Learning over Relational Databases](../../ICLR2026/graph_learning/relatron_automating_relational_machine_learning_over_relational_databases.md)
- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[NeurIPS 2025\] MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning](moemeta_mixture-of-experts_meta_learning_for_few-shot_relational_learning.md)

</div>

<!-- RELATED:END -->
