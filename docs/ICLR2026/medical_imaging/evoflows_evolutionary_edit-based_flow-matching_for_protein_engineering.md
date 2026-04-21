---
title: >-
  [论文解读] EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering
description: >-
  [ICLR 2026 (Workshop on Foundation Models for Science)][医学图像][Protein Engineering] EvoFlows 提出一种基于编辑操作的 Flow Matching 方法，通过学习进化相关蛋白质序列间的突变轨迹，能在模板序列上执行可控数量的突变（插入、删除、替换），同时预测"突变什么"和"在哪里突变"。
tags:
  - ICLR 2026 (Workshop on Foundation Models for Science)
  - 医学图像
  - Protein Engineering
  - Flow Matching
  - Edit Operations
  - Sequence-to-Sequence
  - Evolutionary Trajectories
---

# EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering

**会议**: ICLR 2026 (Workshop on Foundation Models for Science)  
**arXiv**: [2603.11703](https://arxiv.org/abs/2603.11703)  
**代码**: 无  
**领域**: 生物医学 / 蛋白质设计  
**关键词**: Protein Engineering, Flow Matching, Edit Operations, Sequence-to-Sequence, Evolutionary Trajectories

## 一句话总结

EvoFlows 提出一种基于编辑操作的 Flow Matching 方法，通过学习进化相关蛋白质序列间的突变轨迹，能在模板序列上执行可控数量的突变（插入、删除、替换），同时预测"突变什么"和"在哪里突变"。

## 研究背景与动机

蛋白质工程的核心目标是基于已知蛋白质序列（模板），生成功能性变体。这需要模型能够在模板基础上引入合理的突变。现有蛋白质语言模型在优化任务中存在多重局限：

**自回归模型（如 ESM、ProtGPT2）**: 需要从头生成完整序列，无法直接在模板上做局部修改，也难以控制与模板的距离（突变数量）。

**掩码语言模型/离散扩散模型（如 ESM-MLM、EvoDiff）**: 依赖预先指定的突变位置（哪些位置被 mask），但在实际蛋白质工程中，最优突变位置通常未知。这些方法无法自主发现突变位点。

**不支持插入和删除（indels）**: 绝大多数现有方法仅处理固定长度序列的替换突变，而自然进化中大量的适应性变化来自序列长度的变化——即插入和删除操作。

总结来说，现有方法要么不支持模板条件生成，要么需要已知突变位置，要么忽略了 indels——这使得它们与真实蛋白质工程的需求存在显著差距。

## 方法详解

### 整体框架

EvoFlows 是一种**可变长度的序列到序列**建模方法。其核心思想是：将蛋白质工程视为从模板序列到目标变体序列的"编辑流"（edit flow）——即一系列编辑操作（插入、删除、替换）的连续轨迹。通过 Flow Matching 框架，模型学习这些进化轨迹，然后在推理时可以在模板上执行可控数量的编辑操作。

### 关键设计

1. **编辑操作建模（Edit-Based Representation）**:

    - **功能**: 将两个蛋白质序列之间的差异表示为编辑操作序列
    - **核心思路**: 给定模板序列 A 和目标序列 B，通过序列对齐（如 Needleman-Wunsch 算法）计算最小编辑距离，得到一系列操作：替换（substitution）、插入（insertion）、删除（deletion）。每个操作包含位置和具体的氨基酸变化。
    - **设计动机**: 将突变过程显式表示为编辑操作，使模型能同时预测"在哪里突变"和"突变成什么"，并且自然支持序列长度变化（通过 indels）。这比固定长度的掩码/替换范式灵活得多。

2. **进化轨迹学习（Evolutionary Trajectory Learning via Flow Matching）**:

    - **功能**: 利用 Flow Matching 框架学习从模板到变体的连续编辑流
    - **核心思路**: 从进化相关的蛋白质家族中采样序列对（如同一 UniRef 簇中的不同序列），构建编辑操作空间中的概率流。Flow Matching 在这个空间中拟合速度场，从模板出发沿着学到的流场进行 ODE 积分，可以生成连续且合理的变体序列。
    - **设计动机**: Flow Matching 相比离散扩散有更稳定的训练和更好的样本效率。在编辑操作空间（而非序列空间）做流匹配，天然支持可变长度且能保持与自然进化轨迹的一致性。

3. **可控突变数量（Controllable Number of Mutations）**:

    - **功能**: 在推理时控制生成变体与模板之间的编辑距离
    - **核心思路**: 通过控制 ODE 积分的步长/终止时间，可以调节从模板出发"走多远"——短距离产生保守突变（少量替换），长距离产生激进突变（更多替换 + indels）。
    - **设计动机**: 在蛋白质工程中，突变数量的控制至关重要——太少可能无法改善功能，太多可能破坏折叠稳定性。这种可控性是 EvoFlows 的核心实用特性。

### 损失函数 / 训练策略

- **Flow Matching 目标**: 标准的条件流匹配损失，最小化预测速度场与真实速度之间的 MSE
- **训练数据**: 从 UniRef（通用蛋白质参考簇）和 OAS（抗体序列数据库）中提取进化相关的蛋白质家族，构建序列对作为训练轨迹
- **编辑对齐**: 训练前预处理步骤——对每对序列计算最优编辑对齐，作为 flow matching 的目标轨迹

## 实验关键数据

### 实验设置
- 评估数据：UniRef 和 OAS 中多样化的蛋白质家族
- 评估方式：in silico（计算评估），非湿实验验证
- 核心评估维度：生成变体的自然性（是否与天然蛋白质家族分布一致）和探索范围（与模板的距离）

### 主实验

| 方法 | 家族一致性 | 模板距离 | indels支持 | 位置预测 |
|------|-----------|----------|-----------|----------|
| 自回归模型 | 中 | 不可控 | 有限 | 不适用（全序列生成） |
| 掩码语言模型 | 高（保守） | 需预设位置 | 不支持 | 不支持（需先指定） |
| 离散扩散模型 | 中高 | 需预设位置 | 不支持 | 不支持（需先指定） |
| **EvoFlows** | **高** | **更远且可控** | **支持** | **自动预测** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅替换（无 indels） | 探索范围受限 | 验证了 indels 支持的必要性 |
| 不同 ODE 积分步数 | 突变数量连续可控 | 验证了流场的平滑性 |
| 不同蛋白质家族 | 一致表现良好 | 泛化能力可靠 |

### 关键发现

1. **EvoFlows 生成的变体与天然蛋白质家族分布一致**: 说明学到的编辑流确实捕获了自然进化的模式
2. **探索范围远超基线**: 能生成距模板更远的变体同时保持合理性，意味着更大的功能探索空间
3. **同时预测"哪里"和"什么"**: 不需要先验的突变位置知识，这对实际蛋白质工程非常重要

## 亮点与洞察

- **问题定义精准**: 准确识别了现有蛋白质语言模型在工程任务中的三个核心短板（无模板条件、需预知位置、不支持 indels），并用一个统一框架同时解决
- **编辑空间做 Flow Matching**: 将 flow matching 从序列空间转移到编辑操作空间，是一个巧妙的建模选择——自然处理变长序列且物理意义更直观
- **可控性**: 通过 ODE 积分步长控制突变程度，提供了实用的旋钮，工程师可根据需求调节保守/激进程度
- **连接进化与生成**: 利用进化相关序列作为训练信号，使生成过程隐式地遵循自然选择的约束

## 局限与展望

1. **仅 in silico 验证**: 所有实验为计算评估，缺乏湿实验验证。生成变体的实际功能性（酶活性、结合亲和力等）未知
2. **Workshop 论文**: 作为 workshop 论文，方法和实验的详细程度有限，大规模评估尚不充分
3. **编辑对齐的质量**: 训练依赖序列对齐计算编辑操作，对齐质量可能影响学到的流场；对于高度发散的序列对，最优编辑路径的选择不唯一
4. **结构信息缺失**: 当前方法仅在序列层面操作，未利用蛋白质三维结构信息。结构约束可能进一步提升变体的合理性
5. **扩展性**: 对超长蛋白质序列（>1000 残基）的处理效率和质量需要进一步验证
6. **多步编辑的组合效应**: 单对序列的编辑轨迹可能无法捕获多步进化中的协同突变效应

## 相关工作与启发

- **与 EvoDiff 的关系**: EvoDiff 使用离散扩散在序列空间直接生成，需预设突变位置；EvoFlows 在编辑空间做连续流匹配，不需预设位置
- **与 ESM 系列的关系**: ESM 的掩码语言模型擅长评估突变效果但不擅长设计突变方案；EvoFlows 直接面向突变设计
- **Flow Matching 在生物学中的应用**: 这是 flow matching 在蛋白质序列建模中的早期尝试，与分子构象生成中的 flow matching 方法形成呼应
- **对药物设计的启发**: 抗体亲和力成熟、酶工程等应用场景中，可控的序列编辑能力尤为关键

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Flexibility-conditioned Protein Structure Design with Flow Matching](../../ICML2025/medical_imaging/flexibility-conditioned_protein_structure_design_with_flow_matching.md)
- [How to Make the Most of Your Masked Language Model for Protein Engineering](how_to_make_the_most_of_your_masked_language_model_for_protein_engineering.md)
- [Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation](learning_patient-specific_disease_dynamics_with_latent_flow_matching_for_longitu.md)
- [Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation](../../AAAI2026/medical_imaging/ambiguity-aware_truncated_flow_matching_for_ambiguous_medica.md)
- [Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](../../NeurIPS2025/medical_imaging/energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)

<!-- RELATED:END -->
