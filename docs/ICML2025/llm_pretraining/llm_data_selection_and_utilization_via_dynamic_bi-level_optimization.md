---
title: >-
  [论文解读] LLM Data Selection and Utilization via Dynamic Bi-level Optimization
description: >-
  [ICML2025][数据选择] 提出动态数据加权模型(DWM)，通过双层优化在LLM训练过程中实时调整每批数据的权重，捕捉模型动态变化的数据偏好，比静态数据选择方法一致提升性能且可迁移到不同模型规模。
tags:
  - ICML2025
  - 数据选择
  - 双层优化
  - 数据加权
  - LLM预训练
  - 动态偏好
---

# LLM Data Selection and Utilization via Dynamic Bi-level Optimization

**会议**: ICML2025  
**arXiv**: [2507.16178](https://arxiv.org/abs/2507.16178)  
**代码**: 待确认  
**领域**: llm_nlp  
**关键词**: 数据选择, 双层优化, 数据加权, LLM预训练, 动态偏好

## 一句话总结
提出动态数据加权模型(DWM)，通过双层优化在LLM训练过程中实时调整每批数据的权重，捕捉模型动态变化的数据偏好，比静态数据选择方法一致提升性能且可迁移到不同模型规模。

## 研究背景与动机

### 静态数据选择的局限
DoReMi/DSIR等方法在训练前选择数据，忽略了模型在训练过程中数据偏好的变化。且样本独立选择忽略了批内样本的联合效应。

### DWM的目标
1. 在训练中动态调整数据权重
2. 考虑批内样本的联合效应
3. 可迁移到不同规模的模型

## 方法详解

### 双层优化框架
**下层**：用DWM加权的数据训练LLM一步
**上层**：基于下层更新后LLM的验证性能优化DWM参数
通过链式法则实现梯度反传。

### 动态学习
在训练的不同阶段（每隔一定步数）重新运行双层优化，捕捉模型持续变化的数据偏好。

### DWM设计
输入每个样本的特征，输出其在当前批内的相对权重。即插即用，与数据选择方法正交。

## 实验关键数据

### 370M模型预训练

| 方法 | 数据选择 | +DWM | 改善 |
|------|---------|------|------|
| 随机选 | 基线 | **优于基线** | 显著 |
| DSIR选 | 较好 | **更好** | 一致 |
| DsDM选 | 较好 | **更好** | 一致 |

### 跨模型迁移

| 训练规模 | DWM训练于 | 效果 |
|---------|---------|------|
| 370M | 370M | 基线 |
| 1.3B | 370M(迁移) | **一致提升** |

### 数据偏好演化分析

| 训练阶段 | 偏好特征 |
|---------|---------|
| 早期 | 偏好多样性高的数据 |
| 中期 | 偏好难度适中的数据 |
| 后期 | 偏好与验证集相似的数据 |

### 关键发现
1. DWM对随机选择和精心选择的数据都有效
2. 可从小模型迁移到大模型
3. 模型的数据偏好在训练中确实动态变化
4. 批内样本的联合效应对更新方向有实质影响

## 亮点与洞察

1. 动态 vs 静态的核心区别：模型不同阶段需要不同数据。
2. 双层优化实现了"模型告诉你它需要什么数据"。
3. 跨规模迁移使DWM更实用——小模型学到的权重可指导大模型。
4. 数据偏好演化分析提供了训练过程的新洞察。
5. 即插即用设计与现有数据选择方法兼容。

## 局限与展望

1. 双层优化增加了训练计算开销（每隔N步需要额外前向/反向传播）。
2. 验证集的选择影响DWM学到的偏好。
3. 更大规模(>10B)的验证缺失。
4. 与课程学习思路的对比不够。

## 相关工作与启发

- 与DoReMi的区别：DoReMi是静态域重加权，DWM是动态样本加权。
- 与MATES的关系：MATES做在线选择但不做批内加权。
- 启发：双层优化可用于其他需要动态适配的训练场景。

## 评分
- 新颖性: 4.0/5 — 双层优化在数据加权的应用
- 实验充分度: 4.5/5 — 多方法+迁移+分析
- 写作质量: 4.0/5
- 价值: 4.5/5 — 提供了数据利用的新维度

## 补充

### 数据偏好演化的实践启示
早期模型偏好多样性高的数据，后期偏好与验证集相似的数据。这与课程学习思路一致。

### 与课程学习的关联
DWM可视为自动化的课程学习——模型自己决定每个阶段的数据偏好，而非人工设计。

### 计算效率权衡
双层优化每隔N步需额外前反向传播，增加约10-15%训练时间。但带来的性能提升值得这个开销。

<!-- RELATED:START -->

## 相关论文

- [Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization](../../ACL2025/llm_pretraining/model_performance-guided_evaluation_data_selection_for_effective_prompt_optimiza.md)
- [Enhancing Training Data Attribution with Representational Optimization](../../NeurIPS2025/llm_pretraining/enhancing_training_data_attribution_with_representational_optimization.md)
- [Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](../../NeurIPS2025/llm_pretraining/breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)
- [Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](../../ACL2025/llm_pretraining/data_whisperer_data_selection.md)
- [DavIR: Data Selection via Implicit Reward for Large Language Models](../../ACL2025/llm_pretraining/davir_data_selection_via_implicit_reward_for_large_language_models.md)

<!-- RELATED:END -->
