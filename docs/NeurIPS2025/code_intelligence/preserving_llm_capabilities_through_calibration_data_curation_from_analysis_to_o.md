---
title: >-
  [论文解读] Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization
description: >-
  [NeurIPS 2025][LLM压缩] 系统研究了校准数据的组成特性（序列长度/样本量/来源/格式）和领域对应关系对LLM压缩后能力保持的影响，发现激活空间中的代表性和多样性是数据质量的本质决定因素，并据此提出三阶段校准数据策展框架COLA。
tags:
  - NeurIPS 2025
  - LLM压缩
  - 校准数据
  - 后训练量化
  - 后训练剪枝
  - 能力保持
  - 激活空间
---

# Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2510.10618](https://arxiv.org/abs/2510.10618)  
**作者**: Bowei He, Lihao Yin, Huiling Zhen, Shuqi Liu, Han Wu, Xiaokun Zhang, Mingxuan Yuan, Chen Ma (City University of Hong Kong, Huawei)  
**代码**: 已公开  
**领域**: model_compression  
**关键词**: LLM压缩, 校准数据, 后训练量化, 后训练剪枝, 能力保持, 激活空间  

## 一句话总结

系统研究了校准数据的组成特性（序列长度/样本量/来源/格式）和领域对应关系对LLM压缩后能力保持的影响，发现激活空间中的代表性和多样性是数据质量的本质决定因素，并据此提出三阶段校准数据策展框架COLA。

## 研究背景与动机

### 问题背景
后训练压缩（剪枝和量化）是部署LLM的主流方案。在压缩过程中，**校准数据**用于评估权重重要性和激活动态范围，对压缩质量至关重要。然而，现有压缩方法普遍假设对校准数据分布具有鲁棒性，缺乏系统研究。

### 已有工作的不足
- 早期工作仅从**孤立视角**（样本量、数据来源、序列长度）研究校准数据影响
- 评估局限于基础的**语言建模困惑度**和**常识推理**，未涉及数学推理、代码生成等高级能力
- 缺乏对底层影响机制的探索，不清楚什么构成**最优校准数据**
- 未建立系统的校准数据策展策略

### 核心动机
回答四个关键问题：(Q1) 校准数据的组成特性如何影响能力保持？(Q2) 领域对应关系如何影响能力保持？(Q3) 什么是最优校准数据？(Q4) 如何从可用数据中策展最优校准数据？

## 方法详解

### 实验设置
- **模型**：LLaMA3-8B-Instruct、Qwen2.5-7B-Instruct
- **剪枝方法**：SparseGPT（50%非结构化）、Wanda（4:8半结构化）
- **量化方法**：GPTQ（4-bit）、AWQ（4-bit）
- **校准数据源**：C4、WikiText、SlimPajama（预训练数据）；CommonsenseQA、MathQA、CodeQA（下游数据）
- **评估维度**：语言建模(PPL)、常识推理(CS)、数学推理(Math)、代码生成(Code)、多语言理解

### Q1：组成特性的影响

**序列长度**：数学推理对长度最敏感——SparseGPT在短序列(128)下性能骤降25.5%；代码生成呈非单调变化（AWQ在38.71%~47.53%间波动）。剪枝方法比量化方法对长度更敏感，AWQ的逐通道缩放提供显著鲁棒性。

**样本量**：超过64-128个样本后收益递减。但代码生成出现反常模式——LLaMA3-8B+AWQ中，16个样本(46.40%)优于128个(38.71%)；Qwen2.5-7B+GPTQ从16样本的57.67%暴跌至128样本的34.03%。额外样本可能引入次优样例。

**数据来源**：来源选择的影响甚至**超过压缩方法本身的差异**。C4在代码生成上优势明显（LLaMA3-8B上比Wikipedia高19.4%），SlimPajama在数学推理上表现更好（Qwen2.5-7B上比Wikipedia高8.7%）。

**数据格式**：包含显式推理链(Q&A w/ ERC)的格式效果最佳——Qwen2.5-7B+AWQ的数学任务提升9.5%（47.34%→51.84%），推理链激活并保持了模型内部推理机制。

### Q2：领域对应关系的影响

**语言对齐**：英语校准数据对英语为主的模型效果最好；数学任务上，LLaMA3-8B+GPTQ中英语(31.22%)比日语(16.72%)高46.4%。多语言基准上，匹配评估语言比默认英语更优。

**领域匹配**：领域匹配显著增强对应能力但损害其他能力。MathQA校准使量化数学性能提升5.92个百分点，CodeQA使量化代码生成提升7.49个百分点，但两者都增加困惑度(2-3点)。

### Q3：最优校准数据的定义

核心发现：**激活空间中的代表性(Representativeness)和多样性(Diversity)** 从根本上决定校准数据质量。
- **代表性**：样本能否触发目标领域典型的激活模式
- **多样性**：触发的唯一激活模式的覆盖广度

### Q4：COLA框架（三阶段策展）

**Stage 1 — 数据集选择（领域对应）**：
根据部署场景（通用/专用）选择源数据集，考虑语言对齐、领域覆盖、推理难度。形式化为覆盖优化问题：

$$S = \arg\max_{S \subseteq \mathcal{D}} \sum_{c \in C} w_c \cdot \text{coverage}(S, c)$$

**Stage 2 — 数据处理（组成特性）**：
优化序列长度（通常2048 tokens，AWQ可用较短）；格式增强，将数据转换为含显式推理链的Q&A格式。

**Stage 3 — 样本选择（激活空间代表性与多样性）**：
1. 对候选样本前向传播，提取逐层激活向量 $\mathbf{a}_i = [\mathbf{h}_i^1, \ldots, \mathbf{h}_i^L]$
2. 随机投影降维：$\mathbf{a}_i' = \frac{1}{\sqrt{d}} \mathbf{R} \mathbf{a}_i$
3. K-means聚类划分激活空间
4. 从每个簇选取最接近质心的样本：$x_j^* = \arg\min_{x_i \in C_j} \|\mathbf{a}_i' - \mu_j\|_2$

簇数$k$直接控制最终样本量，AWQ用较少样本，剪枝方法用更多样本。

## 实验关键数据

### 实验1：通用部署场景性能对比

| 压缩方法 | 校准数据 | LLaMA3-8B PPL | CS | Math | Code | Qwen2.5-7B PPL | CS | Math | Code |
|---------|---------|----------|------|-------|-------|----------|------|-------|-------|
| AWQ (4-bit) | WikiText (random) | 15.86 | 65.26 | 36.46 | 38.71 | 17.36 | 66.42 | 47.34 | 62.10 |
| AWQ (4-bit) | C4 (random) | 15.48 | 66.21 | 37.19 | 39.87 | 17.00 | 67.42 | 48.29 | 63.72 |
| AWQ (4-bit) | Self-Gen | 15.59 | 67.08 | 37.51 | 39.75 | 17.12 | 68.04 | 48.66 | 63.67 |
| AWQ (4-bit) | **COLA** | **15.41** | **67.42** | **37.85** | **40.17** | **16.95** | **68.47** | **49.02** | **64.15** |
| SparseGPT (50%) | WikiText (random) | 20.15 | 41.85 | 19.18 | 15.34 | 21.54 | 42.23 | 17.85 | 13.45 |
| SparseGPT (50%) | **COLA** | **19.31** | **44.23** | **20.12** | **16.14** | **20.72** | **44.47** | **18.65** | **14.10** |
| GPTQ (4-bit) | WikiText (random) | 16.29 | 65.23 | 31.22 | 34.83 | 17.22 | 65.84 | 35.85 | 34.03 |
| GPTQ (4-bit) | **COLA** | **15.83** | **67.52** | **32.56** | **36.18** | **16.79** | **68.15** | **37.23** | **35.22** |

COLA在所有压缩方法和模型上均一致优于随机采样和Self-Gen基线。改进在剪枝方法上更显著（SparseGPT常识推理+2.38pp），与剪枝对校准数据更敏感的观察一致。

### 实验2：领域匹配校准的影响（AWQ 4-bit）

| 校准数据 | LLaMA3-8B CS | Math | Code | Qwen2.5-7B CS | Math | Code |
|---------|----------|-------|-------|----------|-------|-------|
| WikiText | 65.26 | 36.46 | 38.71 | 66.42 | 47.34 | 62.10 |
| CommonsenseQA | 69.37 | 34.21 | 36.22 | 72.86 | 45.21 | 58.64 |
| MathQA | 64.15 | **41.85** | 35.89 | 65.38 | **54.42** | 57.21 |
| CodeQA | 63.92 | 33.42 | **44.62** | 64.97 | 43.85 | **68.73** |

领域匹配校准在目标能力上带来大幅提升（MathQA→Math: +5.39pp; CodeQA→Code: +5.91pp），但以其他能力下降为代价。这揭示了校准数据选择本身是一个**能力权衡**问题。

## 亮点

- **系统性研究**：首次从组成特性和领域对应两大维度全面研究校准数据对LLM压缩的影响，覆盖数学推理和代码生成等高级能力
- **机制洞察**：揭示激活空间的代表性和多样性是校准数据质量的根本决定因素，超越了数据来源等表面特征
- **实用框架**：提出的COLA三阶段框架（选择→处理→采样）与现有压缩方法正交，可即插即用，在通用和专用部署场景下均有效
- **关键发现**：校准数据来源选择的影响可超过压缩方法本身的差异；显式推理链格式对保持推理能力至关重要

## 局限与展望

- **额外计算开销**：Stage 3需要对候选样本做前向传播和聚类，增加了预处理成本
- **模型规模有限**：仅在7-8B参数模型上验证，未测试更大规模模型（如70B+）
- **绝对提升较小**：通用场景下COLA相比随机采样的提升通常在1-2个百分点，边际改进有限
- **未考虑压缩方法特异性**：当前框架对所有压缩方法使用统一流程，未针对剪枝/量化的算法特性定制
- **聚类参数敏感性**：K-means的簇数$k$和降维维度$d$的选择缺乏理论指导

## 与相关工作的对比

- **Williams & Aletras (2024)**：仅发现校准数据变化带来性能差异的表面现象，本文深入分析底层激活空间机制
- **Bandari et al. (2024)**：研究C4是否为最优剪枝校准数据，但局限于剪枝且未给出系统策展方案
- **Ji et al.**：提出Self-Gen方法用模型自生成数据校准，但在本文实验中被COLA一致超越
- **Lee et al. (2023)**：关注序列长度对齐，本文在此基础上发现长度影响具有能力依赖性和方法依赖性
- **Zhang et al. (2024), Jaiswal et al. (2024)**：探索样本量影响，本文进一步揭示样本量增加可能损害特定能力（如代码生成）
- **SparseGPT, Wanda, GPTQ, AWQ**：主流压缩方法均可受益于本文的COLA框架，验证了方法正交性

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统研究校准数据对高级推理能力的影响并提出激活空间机制解释
- 实验充分度: ⭐⭐⭐⭐ — 覆盖2模型×4压缩方法×多维度变量，消融全面，但模型规模有限
- 写作质量: ⭐⭐⭐⭐ — 问题驱动(Q1-Q4)结构清晰，从分析到优化逻辑连贯
- 价值: ⭐⭐⭐⭐ — 对LLM压缩实践有直接指导意义，COLA框架简单有效且即插即用

<!-- RELATED:START -->

## 相关论文

- [CoRe: Benchmarking LLMs' Code Reasoning Capabilities through Static Analysis Tasks](core_benchmarking_llms_code_reasoning_capabilities_through_static_analysis_tasks.md)
- [DyCodeEval: Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination](../../ICML2025/code_intelligence/dynamic_benchmarking_of_reasoning_capabilities_in_code_large_language_models_und.md)
- [A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions](a_stochastic_differential_equation_framework_for_multi-objective_llm_interaction.md)
- [FractalBench: Diagnosing Visual-Mathematical Reasoning Through Recursive Program Synthesis](fractalbench_diagnosing_visual-mathematical_reasoning_through_recursive_program_.md)
- [Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning](co-evolving_llm_coder_and_unit_tester_via_reinforcement_learning.md)

<!-- RELATED:END -->
