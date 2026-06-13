---
title: >-
  [论文解读] Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes
description: >-
  [ACL 2025][LLM 其他][math reasoning] 提出 MathNeuro，一种仅需前向传播的计算高效方法，通过过滤掉对通用语言任务同样重要的参数来定位 LLM 中数学推理专属的参数，剪枝这些参数可删除数学能力，缩放这些参数可提升 4-35% 的数学性能。 研究领域现状：数学推理是 LLM 研究的核心能力…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "math reasoning"
  - "parameter importance"
  - "skill localization"
  - "剪枝"
  - "neuron isolation"
---

# Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes

**会议**: ACL 2025  
**arXiv**: [2410.16930](https://arxiv.org/abs/2410.16930)  
**代码**: [https://github.com/bryanchrist/MathNeuro](https://github.com/bryanchrist/MathNeuro)  
**领域**: LLM / 可解释性 / 数学推理  
**关键词**: math reasoning, parameter importance, skill localization, pruning, neuron isolation  

## 一句话总结

提出 MathNeuro，一种仅需前向传播的计算高效方法，通过过滤掉对通用语言任务同样重要的参数来定位 LLM 中数学推理专属的参数，剪枝这些参数可删除数学能力，缩放这些参数可提升 4-35% 的数学性能。

## 研究背景与动机

**研究领域现状：** 数学推理是 LLM 研究的核心能力之一，但关于数学推理如何编码在模型参数中、能否被定位和隔离的研究极少。已有的技能/知识定位方法主要聚焦于语言特定参数或事实知识，未专门研究数学推理。

**现有方法的局限性：**（1）基于梯度的参数重要性方法（如 Panigrahi et al. 2023）计算开销大，对大模型不可行；（2）基于前向传播的方法如 Wanda（Sun et al. 2023）能找到对数学重要的参数，但无法隔离数学专属参数——因为这些参数与其他任务的重要参数高度重叠；（3）LAPE（Tang et al. 2024）在不同模型上表现不一致。

**核心挑战：** 数学推理不仅涉及计算，还与自然语言理解深度交织，使得数学专属参数难以与通用语言参数区分。

## 方法详解

### 整体框架

MathNeuro 分三步：（1）分别用数学数据和非数学数据计算每个参数的重要性分数；（2）分别取 Top-K% 最重要参数；（3）取数学 Top-K 与非数学 Top-K 的差集作为数学专属参数。

### 关键设计

1. **基于权重×激活值的参数重要性计算：** 沿用 Wanda 的核心思想，对每个参数 $(i,j)$ 计算 $S_{ij} = |W_{ij}| \cdot \|X_j\|_2$，同时考虑权重大小和激活强度。对 N 个样本求和以获得鲁棒估计。不需要梯度，仅需前向传播。

2. **任务差异化过滤：** 分别在 attention 和 MLP 层中，用数学数据（GSM8K/MATH）和非数学数据（MMLU/RACE）计算重要性分数。取各自的 Top-K% 参数后做差集：$T_{math} = \text{TopK}_{math} \setminus \text{TopK}_{non\text{-}math}$，过滤掉对通用语言同样重要的参数。

3. **数据高效性：** 实验表明，仅用单个数学样本和单个非数学样本即可有效定位数学专属参数，虽然效果略逊于 500 样本，但依然显著优于 baseline。

## 实验

### 剪枝实验（Llama 3.2 1B IT，TopK=15%）

| 方法 | GSM8K 准确率变化 | RACE 准确率变化 | MMLU 准确率变化 |
|------|-----------------|----------------|----------------|
| MathNeuro (RACE) | 大幅下降至 ~0% | 小幅下降（≈随机剪枝）| 小幅下降 |
| MathNeuro (MMLU) | 大幅下降至 ~0% | 小幅下降 | 小幅下降（≈随机剪枝）|
| Wanda | 大幅下降 | 大幅下降 | 大幅下降 |
| LAPE | 不一致 | 不一致 | 不一致 |
| Random | 适度下降 | 适度下降 | 适度下降 |

### 缩放实验（缩放因子 1.1，TopK=5%）

| 模型 | 方法 | GSM8K 提升 | 非数学任务影响 |
|------|------|-----------|---------------|
| Llama 3.2 1B IT | MathNeuro | +4-17% | 无显著变化 |
| Gemma 2 2B IT | MathNeuro | +4-17% | 无显著变化 |
| Llama 3.1 8B IT | MathNeuro (×1.01) | +4-17% | 无显著变化 |
| Phi 1.5 (预训练) | MathNeuro | +5-35% (MATH) | 无显著变化 |

### 参数一致性分析

| 样本数 | 两次独立识别的参数重叠率 |
|--------|----------------------|
| 1 | ~70-80% |
| 10 | ~85-90% |
| 100 | ~95%+ |
| 500 | ~97%+ |

### 关键发现

- MathNeuro 识别的参数仅占模型总参数的 ~1.5-1.8%，却承载了几乎所有数学推理能力
- 数学专属参数在各 decoder block 中分布较为均匀，说明数学推理编码在整个模型中而非集中在特定层
- 在 GSM8K 上识别的数学参数可泛化到 MATH、EGSM 等未见数学任务
- 剪枝后模型在非数学任务上的退化程度与随机剪枝相当，证实了参数隔离的有效性

## 亮点

- 方法极其简洁——仅需前向传播和集合差运算，无需梯度或复杂优化
- 数据效率极高：甚至单个样本就能定位数学专属参数
- 双向验证设计完善：剪枝删除能力 + 缩放增强能力，互相印证
- 发现数学推理参数均匀分布在模型各层，为理解 LLM 编码技能的方式提供新洞察
- 在 5 个不同规模模型（1B-8B）上一致有效

## 局限性

- 仅在 1B-8B 规模模型上验证，未测试更大模型（>8B）
- 缩放因子是经验性选择的（小模型 1.1，大模型 1.01），缺乏系统性超参搜索
- 采用"数学 vs. 非数学"的二元划分过于简化，数学推理包含多种子技能（算术、代数、几何等）
- 评估以 GSM8K/MATH 为主，这些数据集可能无法代表所有数学推理类型
- 方法基于 Wanda 的权重×激活值公式，理论解释较弱

## 相关工作

- **技能定位：** Wanda (Sun et al. 2023) 基于权重×激活值的剪枝；LAPE (Tang et al. 2024) 基于激活概率熵的语言定位
- **数学推理：** GSM8K (Cobbe et al. 2021)、MATH (Hendrycks et al. 2021b) 基准；Hanna et al. 2023 分析加减法概念的处理
- **知识编码：** Dai et al. 2022 基于梯度的知识神经元；Suau et al. 2024 语言特定参数干预
- **模型剪枝：** Chang et al. 2024 参数重要性综述

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 8 |
| 技术深度 | 7 |
| 实验充分性 | 9 |
| 写作质量 | 8 |
| 实用价值 | 7 |
| 总分 | 7.8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] STEM-PoM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing](stem-pom_evaluating_language_models_math-symbol_reasoning_in_document_parsing.md)
- [\[ACL 2025\] ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)
- [\[ACL 2025\] Assessing and Enhancing the Causal Reasoning Abilities of Language Models via Faithful Textual Interpretation](assessing_and_enhancing_the_causal_reasoning_abilities_of_language_models_via_fai.md)
- [\[ACL 2025\] To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization](to_code_or_not_to_code_adaptive_tool_integration_for_math_language_models_via_ex.md)
- [\[ACL 2025\] Disentangling Memory and Reasoning Ability in Large Language Models](disentangle_memory_reasoning.md)

</div>

<!-- RELATED:END -->
