---
title: >-
  [论文解读] Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models
description: >-
  [ACL 2025][LLM/NLP][合成数据] 提出 Genetic-Instruct 算法，借鉴进化算法的交叉和变异操作，从 512 个种子指令扩展生成 750 万+高质量编码指令，使用 Instructor-LLM/Coder-LLM/Judge-LLM 三角色流水线，训练后的模型在代码生成基准上超越 Self-Instruct 和 Evol-Instruct。
tags:
  - ACL 2025
  - LLM/NLP
  - 合成数据
  - 代码生成
  - evolutionary algorithm
  - 指令微调
  - 可扩展性
---

# Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2407.21077](https://arxiv.org/abs/2407.21077)  
**代码**: https://huggingface.co/datasets/nvidia/OpenCodeGeneticInstruct  
**领域**: LLM/NLP  
**关键词**: 合成数据, 代码生成, evolutionary algorithm, 指令微调, 可扩展性

## 一句话总结
提出 Genetic-Instruct 算法，借鉴进化算法的交叉和变异操作，从 512 个种子指令扩展生成 750 万+高质量编码指令，使用 Instructor-LLM/Coder-LLM/Judge-LLM 三角色流水线，训练后的模型在代码生成基准上超越 Self-Instruct 和 Evol-Instruct。

## 研究背景与动机

**领域现状**：LLM 代码生成能力的提升依赖高质量指令数据，但人工标注昂贵。合成指令生成（Self-Instruct, Evol-Instruct）是替代方案。

**现有痛点**：Self-Instruct 生成的指令多简单不够挑战；Evol-Instruct 单靠变异增加复杂度；基于代码的方法（OSS-Instruct）依赖大规模高质量代码库。

**核心矛盾**：如何从极少量种子扩展到百万级多样且高质量的编码指令？

**本文目标** 设计可大规模并行的合成指令生成算法。

**切入角度**：进化算法思路——交叉扩主题覆盖，变异增局部多样性，适应度函数保质量。

**核心 idea**：用进化算法的交叉+变异+选择，配合三个 LLM 角色（指令生成/代码编写/质量评判），从 512 种子无限扩展。

## 方法详解

### 整体框架
种子指令（512个）-> 进化循环：随机选择交叉或变异操作 -> Instructor-LLM 生成新指令 -> Coder-LLM 生成代码 -> Judge-LLM 评估质量 -> 通过的加入种子池 -> 重复直到目标规模。多 colony 并行 + 多代（generation）迭代。

### 关键设计

1. **交叉操作 (Crossover)**

    - Self-Instruct 风格：从种子池选多个示例做 few-shot，生成新指令
    - 作用：扩展主题覆盖范围
    - 设计动机：跳出种子的局部主题空间

2. **变异操作 (Mutation)**

    - Evol-Instruct 风格：基于预定义规则演化单个指令
    - 作用：增加局部多样性和复杂度
    - 设计动机：在已有主题内增加难度和变化

3. **三角色 LLM 流水线**

    - Instructor-LLM：生成指令
    - Coder-LLM：生成对应代码
    - Judge-LLM：评估指令-代码对的质量
    - 设计动机：分工明确，每个角色可独立优化

4. **多 colony 并行 + 去污染**

    - 多个 colony 并行执行进化循环
    - 最终去污染（decontaminate）确保不包含评估基准中的题目
    - 设计动机：线性扩展生成速度

## 实验关键数据

### 主实验 -- 代码生成基准性能（SFT 后）

| 训练数据 | HumanEval | MBPP | MultiPL-E | 平均 |
|---------|-----------|------|-----------|------|
| Self-Instruct | ~60% | ~55% | ~45% | ~53% |
| Evol-Instruct | ~65% | ~58% | ~48% | ~57% |
| OSS-Instruct | ~68% | ~60% | ~50% | ~59% |
| **Genetic-Instruct (7.5M)** | **~75%** | **~67%** | **~58%** | **~67%** |

### 消融 -- 数据量与性能

| 数据量 | HumanEval |
|--------|-----------|
| 100K | ~62% |
| 1M | ~68% |
| 3M | ~72% |
| 7.5M | **~75%** |

### 关键发现
- **7.5M 数据带来一致且显著的提升**
- **交叉+变异的耦合是关键**：单独使用任一操作都不如组合
- **弱模型也能生成有效数据**：不需要最强的 LLM 做 Instructor
- **完全开源**：不依赖闭源模型或数据

## 亮点与洞察
- **进化算法与 LLM 合成数据的结合**非常自然——交叉负责探索、变异负责利用、适应度函数保证质量
- **三角色分工**比单模型生成+自评估更可靠
- **从 512 种子到 750 万数据**展示了强大的可扩展性

## 局限与展望
- 计算成本高（需要三个 LLM 推理）
- 仅针对编码任务
- 改进方向：扩展到通用指令、自适应交叉/变异比例

## 相关工作与启发
- **vs Self-Instruct**：Self-Instruct 只有交叉，Genetic-Instruct 加入了变异和质量评判
- **vs Evol-Instruct**：Evol-Instruct 只有变异，缺少交叉的主题扩展

## 评分
- 新颖性: ⭐⭐⭐⭐ 进化算法+LLM 合成数据的优雅结合
- 实验充分度: ⭐⭐⭐⭐⭐ 750 万数据 + 多基准 + 消融 + 全开源
- 写作质量: ⭐⭐⭐⭐ 算法描述清晰
- 价值: ⭐⭐⭐⭐⭐ 开源数据集对社区有直接价值

<!-- RELATED:START -->

## 相关论文

- [DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](difflm_controllable_synthetic_data_generation_via_diffusion_language_models.md)
- [LESA: Learnable LLM Layer Scaling-Up](lesa_learnable_llm_layer_scaling-up.md)
- [Evaluating Language Models as Synthetic Data Generators](evaluating_lms_synthetic_data_gen.md)
- [Scaling Up Active Testing to Large Language Models](../../NeurIPS2025/llm_nlp/scaling_up_active_testing_to_large_language_models.md)
- [HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation](hygenar_an_llm-driven_hybrid_genetic_algorithm_for_few-shot_grammar_generation.md)

<!-- RELATED:END -->
