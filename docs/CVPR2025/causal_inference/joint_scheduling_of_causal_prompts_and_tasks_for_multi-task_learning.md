---
title: >-
  [论文解读] Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning
description: >-
  [CVPR 2025][因果提示] 提出 JSCPT，通过联合调度因果提示（消除虚假相关）和任务学习顺序（利用动态任务关系），优化多任务提示学习的性能。
tags:
  - CVPR 2025
  - 因果提示
  - 任务调度
  - 多任务学习
  - VLM
  - 虚假相关
---

# Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: 待确认  
**领域**: 因果推理 / 多任务学习  
**关键词**: 因果提示, 任务调度, 多任务学习, VLM, 虚假相关

## 一句话总结
提出 JSCPT，通过联合调度因果提示（消除虚假相关）和任务学习顺序（利用动态任务关系），优化多任务提示学习的性能。

## 研究背景与动机
**领域现状**：多任务提示学习利用共享提示来微调预训练 VLM 以适应多个下游任务。

**现有痛点**：忽视了两个问题 — (1) 训练数据中的虚假相关导致提示学习到错误模式；(2) 不同任务之间的关系在训练过程中动态变化，固定的联合训练策略不够灵活。

**本文目标** 同时解决提示中的虚假相关和任务间的动态关系建模问题。

**核心 idea**：用因果推理去除提示中的虚假特征，用自适应调度优化任务学习顺序。

## 方法详解

### 关键设计
1. **因果提示学习**：通过因果干预去除提示中的虚假相关特征，保留与任务因果相关的信号。
2. **自适应任务调度**：根据训练过程中任务间的梯度关系动态调整学习权重和顺序。
3. **联合优化框架**：将因果提示和任务调度统一到一个优化目标中。

## 实验关键数据

### 关键发现
- 在多任务基准上显著优于固定提示+均匀任务训练的基线
- 因果提示有效减少了由虚假相关导致的过拟合
- 自适应调度在任务冲突场景下优势更明显

## 亮点与洞察
- 将因果推理和课程学习两个思想有机结合到提示学习框架中
- 任务关系的动态建模比固定权重更符合实际训练过程

## 局限与展望
- 因果变量的定义和干预设计依赖领域知识
- 任务数量较多时调度空间指数增长，需要高效搜索策略

<!-- RELATED:START -->

## 相关论文

- [A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](../../NeurIPS2025/causal_inference/a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)
- [Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains](../../ICML2025/causal_inference/learning_time-aware_causal_representation_for_model_generalization_in_evolving_d.md)
- [Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](../../ICLR2026/causal_inference/function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)
- [Characterization and Learning of Causal Graphs from Hard Interventions](../../NeurIPS2025/causal_inference/characterization_and_learning_of_causal_graphs_from_hard_interventions.md)
- [Do-PFN: In-Context Learning for Causal Effect Estimation](../../NeurIPS2025/causal_inference/do-pfn_in-context_learning_for_causal_effect_estimation.md)

<!-- RELATED:END -->
